"""The sensitivity row of an amount that is reported as a concentration in a
compartment whose size changes (issue #742).

A species with ``hasOnlySubstanceUnits`` is stored as its amount over the
compartment's size at load. Where a rate rule or an assignment rule resizes the
compartment, ``run()`` reports the species as ``amount/V_live(t)``: the column
is multiplied by ``V_static/V_live`` when the result is made. The sensitivity
row was left as the integrator wrote it, the derivative of the stored value:
too large by ``V_live/V_static``, and an exact 0 in the column of a parameter
that only moves the volume, with nothing raised or logged.

With ``V' = g`` from 1 and ``X ->`` at ``k*X`` from an amount of 2, the
reported column is ``X = 2*exp(-k*t)/(1 + g*t)``. At t = 4, dX/dk came back
-5.3626 for -1.7875 and dX/dg 0 for -0.5958.

The row is the quotient rule now. Expected values are closed forms, and central
differences of runs with no sensitivities for the model that has none.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

K, G, X0 = 0.1, 0.5, 2.0
RUN = {"t_span": (0.0, 4.0), "n_points": 5, "rtol": 1e-11, "atol": 1e-13}
T = np.linspace(0.0, 4.0, 5)
REST = "substanceOnly species X in V; X = 2; k = 0.1; R1: X -> ; k*X\n"
MODELS = {
    "a-rate-rule": ("compartment V = {v0}; V' = g; g = 0.5; " + REST, "g"),
    "an-assignment-rule": ("compartment V; V := {v0} + g*time; g = 0.5; " + REST, "g"),
}


def _model(kind: str, v0: float = 1.0) -> bngsim.Model:
    return bngsim.Model.from_antimony_string(MODELS[kind][0].format(v0=v0))


def _x(k=K, g=G, x0=X0, v0=1.0):
    """The reported column: the amount over the live size."""
    return x0 * np.exp(-k * T) / (v0 + g * T)


def _dk(v0=1.0, k=K):
    return -T * _x(k=k, v0=v0)


def _dg(v0=1.0):
    return -_x(v0=v0) * T / (v0 + G * T)


def _row(result, name="X"):
    return np.asarray(result.sensitivities)[:, list(result.species_names).index(name), :]


def _column(result, name="X"):
    return np.asarray(result.species)[:, list(result.species_names).index(name)]


@pytest.mark.parametrize("kind", sorted(MODELS))
@pytest.mark.parametrize("v0", [1.0, 2.0])
def test_the_row_is_the_derivative_of_the_column_that_is_reported(kind, v0):
    """In the column of ``k`` the row was the amount's, too large by the live
    size over the size at load, 3 at t = 4. In the column of ``g``, which only
    moves the volume, it was 0."""
    result = bngsim.Simulator(_model(kind, v0), method="ode", sensitivity_params=["k", "g"]).run(
        **RUN
    )
    np.testing.assert_allclose(_column(result), _x(v0=v0), rtol=1e-8)
    row = _row(result)
    np.testing.assert_allclose(row[:, 0], _dk(v0), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(row[:, 1], _dg(v0), rtol=1e-6, atol=1e-9)
    assert result.ar_sensitivity_refused == frozenset()


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_the_row_of_its_own_initial_value(kind):
    """The initial-condition axis: dX(t)/dX(0) is ``exp(-k*t)/(1 + g*t)``, and
    was ``exp(-k*t)``."""
    result = bngsim.Simulator(_model(kind), method="ode", sensitivity_ic=["X"]).run(**RUN)
    block = np.asarray(result.sensitivities_ic)
    row = block[:, list(result.species_names).index("X"), 0]
    np.testing.assert_allclose(row, np.exp(-K * T) / (1.0 + G * T), rtol=1e-6, atol=1e-9)


def test_the_row_of_the_compartment_s_own_initial_size():
    """A rate-rule compartment is a state, and its initial size has an axis.
    The amount does not move with it and the live size does, one for one:
    dX/dV(0) is ``-X/V_live``. It was 0."""
    result = bngsim.Simulator(_model("a-rate-rule"), method="ode", sensitivity_ic=["V"]).run(**RUN)
    block = np.asarray(result.sensitivities_ic)
    row = block[:, list(result.species_names).index("X"), 0]
    np.testing.assert_allclose(row, -_x() / (1.0 + G * T), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_every_column_at_once_and_a_selector(kind):
    """``compute_all_sensitivities`` stamps each chunk, and
    ``output_sensitivities("species:X")`` reads the same row."""
    sim = bngsim.Simulator(_model(kind), method="ode")
    result = sim.compute_all_sensitivities(params=["k", "g"], chunk_size=1, **RUN)
    row = _row(result)
    np.testing.assert_allclose(row[:, 0], _dk(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(row[:, 1], _dg(), rtol=1e-6, atol=1e-9)
    result = bngsim.Simulator(_model(kind), method="ode", sensitivity_params=["k", "g"]).run(**RUN)
    chosen = np.asarray(result.output_sensitivities(["species:X"]))
    np.testing.assert_allclose(chosen.reshape(len(T), 2)[:, 1], _dg(), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_a_batch_row(kind):
    """Each row is stamped against its own clone."""
    sim = bngsim.Simulator(_model(kind), method="ode", sensitivity_params=["k", "g"])
    rows = sim.run_batch(params=[{"k": 0.3}], **RUN)
    np.testing.assert_allclose(_row(rows[0])[:, 0], _dk(k=0.3), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_a_gradient_contracts_the_same_rows(kind):
    """``Result.gradient`` of the sum of X squared over the samples:
    [-9.2974, -4.6568], and was [-19.79, 0]."""
    result = bngsim.Simulator(_model(kind), method="ode", sensitivity_params=["k", "g"]).run(**RUN)
    column = list(result.species_names).index("X")

    def of_x_alone(species, time):
        weight = np.zeros_like(species)
        weight[:, column] = 2.0 * species[:, column]
        return weight

    got = np.asarray(result.gradient(of_x_alone))
    want = [np.sum(2.0 * _x() * _dk()), np.sum(2.0 * _x() * _dg())]
    np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_allclose(want, [-9.2974, -4.6568], rtol=1e-4)


GROWN = (
    "compartment cell = 1; cell' = kv*(2 - cell); kv = 0.5; ksyn = 3; kdeg = 0.2;\n"
    "substanceOnly species P in cell; P = 0; species Y in cell; Y = 1;\n"
    "J1: -> P; ksyn\nJ2: P -> ; kdeg*P\nJ3: Y -> ; kdeg*Y\n"
)


def _by_plain_runs(text, name, param, value, h=1e-5):
    def column(at):
        model = bngsim.Model.from_antimony_string(text)
        model.set_param(param, at)
        model.reset()
        return _column(bngsim.Simulator(model, method="ode").run(**RUN), name)

    return (column(value * (1 + h)) - column(value * (1 - h))) / (2 * h * value)


@pytest.mark.parametrize("param,value", [("ksyn", 3.0), ("kv", 0.5), ("kdeg", 0.2)])
def test_a_species_made_in_a_compartment_that_grows(param, value):
    """A cell that grows toward 2 with an amount made in it at ``ksyn``.
    Against differences of runs of the reported column: at t = 4, dP/dksyn
    was 2.7534 for 1.4766, dP/dkdeg -14.34 for -7.69, and dP/dkv 0 for
    -1.2860."""
    model = bngsim.Model.from_antimony_string(GROWN)
    result = bngsim.Simulator(model, method="ode", sensitivity_params=[param]).run(**RUN)
    want = _by_plain_runs(GROWN, "P", param, value)
    np.testing.assert_allclose(_row(result, "P")[:, 0], want, rtol=2e-5, atol=1e-8)
    assert np.max(np.abs(want)) > 1e-3


@pytest.mark.parametrize("param,value", [("kv", 0.5), ("kdeg", 0.2)])
def test_a_concentration_in_the_same_compartment_is_as_it_was(param, value):
    """Control. ``Y`` is held as a concentration: its dilution is integrated,
    its column is not rescaled, and its row was right."""
    model = bngsim.Model.from_antimony_string(GROWN)
    result = bngsim.Simulator(model, method="ode", sensitivity_params=[param]).run(**RUN)
    want = _by_plain_runs(GROWN, "Y", param, value)
    np.testing.assert_allclose(_row(result, "Y")[:, 0], want, rtol=2e-5, atol=1e-8)


def test_an_amount_in_a_compartment_that_does_not_change_is_as_it_was():
    """Control. No rescale, no change: dX/dk = -t*X."""
    model = bngsim.Model.from_antimony_string("compartment V = 2; " + REST)
    result = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(**RUN)
    column = _column(result)
    np.testing.assert_allclose(_row(result)[:, 0], -T * column, rtol=1e-6, atol=1e-9)


DECLINED = "compartment V; V := piecewise(1 + g*time, time < 100, 1); g = 0.5; " + REST


def test_a_size_whose_own_derivative_is_not_to_be_had_leaves_the_row_nan():
    """The compartment's rule holds a condition, and codegen declines its
    output sensitivity (GH #198). The species' row needs it: NaN, named in a
    warning and in ``ar_sensitivity_refused``, where it held the derivative of
    the stored amount."""
    model = bngsim.Model.from_antimony_string(DECLINED)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k", "g"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(**RUN)
    assert np.isnan(_row(result)).all()
    assert "X" in result.ar_sensitivity_refused
    assert any("#742" in str(w.message) and "'X'" in str(w.message) for w in caught)
    np.testing.assert_allclose(_column(result), _x(), rtol=1e-8)


def test_a_run_with_no_row_for_the_rule_leaves_the_species_row_nan(monkeypatch):
    """The same where the result holds no output-sensitivity row for the
    compartment's rule at all: nothing to take the size's derivative from."""
    real = bngsim.Simulator._stamp

    def stamped(self, result, **kwargs):
        result._expression_sensitivities = np.empty((0, 0, 0))
        return real(self, result, **kwargs)

    monkeypatch.setattr(bngsim.Simulator, "_stamp", stamped)
    sim = bngsim.Simulator(_model("an-assignment-rule"), method="ode", sensitivity_params=["k"])
    with pytest.warns(UserWarning, match="#742"):
        result = sim.run(**RUN)
    assert np.isnan(_row(result)).all()
    assert result.ar_sensitivity_refused == frozenset({"X"})


def test_both_kinds_of_refused_row_are_listed():
    """``Z := 3*X`` is an assignment-rule species in the same compartment,
    refused as it was (GH #221). ``X`` is refused beside it, and the list
    holds both."""
    text = DECLINED + "substanceOnly species Z in V; Z := 3*X\n"
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["k", "g"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(**RUN)
    assert result.ar_sensitivity_refused == frozenset({"X", "Z"})
