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


# ── Where the amount at load was read off the size ──────────────────────────
#
# The quotient rule takes the stored amount's own row from the integrator. An
# amount that was made at load by converting a concentration with the size the
# rule had there is a number in the model: the parameters that size was read
# from do not reach it, in a write or in a column. Main returned 0 for such a
# column, right only where the two missing terms cancel.

_M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)
BY_RULE = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfCompartments><compartment id="V" constant="false"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="V" {{declared}} boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="v0" value="{{v0}}" constant="true"/>
      <parameter id="g" value="0.5" constant="true"/>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfRules><assignmentRule variable="V"><math {_M}>
      <apply><plus/><ci>v0</ci><apply><times/><ci>g</ci>{_TIME}</apply></apply>
    </math></assignmentRule></listOfRules>
    <listOfReactions>
      <reaction id="R1" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math {_M}><apply><times/><ci>k</ci><ci>X</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""
HELD_AS_AN_AMOUNT = 'initialConcentration="2" hasOnlySubstanceUnits="true"'
GIVEN_AS_AN_AMOUNT = 'initialAmount="2" hasOnlySubstanceUnits="true"'
A_CONCENTRATION_GIVEN_AS_AN_AMOUNT = 'initialAmount="2" hasOnlySubstanceUnits="false"'


def _by_rule(declared: str, v0: float = 1.5) -> bngsim.Model:
    return bngsim.Model.from_sbml_string(BY_RULE.format(declared=declared, v0=v0))


@pytest.mark.parametrize(
    "load",
    [
        lambda: _by_rule(HELD_AS_AN_AMOUNT),
        lambda: _by_rule(A_CONCENTRATION_GIVEN_AS_AN_AMOUNT),
        lambda: bngsim.Model.from_antimony_string(
            "compartment V; V := v0 + g*time; v0 = 1.5; g = 0.5; "
            "substanceOnly species X in V; X = 2*V; k = 0.1; R1: X -> ; k*X\n"
        ),
    ],
    ids=["a-concentration-held-as-an-amount", "an-amount-held-as-a-concentration", "set-to-2*V"],
)
def test_what_sized_the_compartment_when_a_value_was_converted_is_refused(load):
    """``V := v0 + g*time`` with the species' initial value converted by V(0),
    or assigned from it. For a concentration of 2 held as an amount, dX/dv0 is
    [0, 0.2262, 0.2620, 0.2469, 0.2189]; main returned 0, and the quotient
    rule alone [-1.3333, -0.6786, ...]. A write to ``v0`` left the amount where
    it was. Refused, with what the rule reads (#696's record)."""
    model = load()
    assert {"v0", "g"} <= set(model.frozen_params)
    with pytest.raises(bngsim.ParameterError, match="read\\s+once"):
        model.set_param("v0", 3.0)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="'v0'"):
        bngsim.Simulator(load(), method="ode", sensitivity_params=["v0"])
    # A column of what the size was not read from is still to be had.
    result = bngsim.Simulator(load(), method="ode", sensitivity_params=["k"]).run(**RUN)
    assert np.isfinite(_row(result)).all()
    assert np.abs(_row(result)[-1, 0]) > 0.1


def test_the_size_s_own_parameter_where_nothing_was_converted():
    """The amount is given as an amount, so nothing was read off the size at
    load: ``X = 2*exp(-k*t)/(v0 + g*t)`` and dX/dv0 = -X/(v0 + g*t), -0.8889 at
    the start. It was 0. Checked against a model rebuilt at another ``v0``
    too."""
    model = _by_rule(GIVEN_AS_AN_AMOUNT)
    assert model.frozen_params == []
    result = bngsim.Simulator(model, method="ode", sensitivity_params=["v0", "g"]).run(**RUN)
    np.testing.assert_allclose(_row(result)[:, 0], -_x(v0=1.5) / (1.5 + G * T), rtol=1e-6)
    np.testing.assert_allclose(_row(result)[:, 1], _dg(1.5), rtol=1e-6, atol=1e-9)
    h = 1e-4
    rebuilt = [
        _column(bngsim.Simulator(_by_rule(GIVEN_AS_AN_AMOUNT, v0), method="ode").run(**RUN))
        for v0 in (1.5 + h, 1.5 - h)
    ]
    np.testing.assert_allclose(
        _row(result)[:, 0], (rebuilt[0] - rebuilt[1]) / (2 * h), rtol=1e-5, atol=1e-8
    )


BY_A_STATE = (
    "compartment W = 1; species Z in W; Z = 0; kz = 0.4; aux := 3*kz;\n"
    "compartment V; V := 1 + 0.5*Z;\n"
    "substanceOnly species X in V; X = 2; k = 0.1;\nR0: -> Z; kz\nR1: X -> ; k*X\n"
)


@pytest.mark.parametrize("param,value", [("kz", 0.4), ("k", 0.1)])
def test_a_compartment_sized_by_a_rule_that_reads_a_state(param, value):
    """``V := 1 + 0.5*Z`` with ``Z`` made at ``kz``, the shape of
    BIOMD0000000856's total volume, beside another rule's expression. Against
    models rebuilt from the text at another value."""
    result = bngsim.Simulator(
        bngsim.Model.from_antimony_string(BY_A_STATE), method="ode", sensitivity_params=[param]
    ).run(**RUN)

    def rebuilt(at):
        text = BY_A_STATE.replace(f"{param} = {value}", f"{param} = {at!r}")
        assert text != BY_A_STATE
        model = bngsim.Model.from_antimony_string(text)
        return _column(bngsim.Simulator(model, method="ode").run(**RUN))

    h = 1e-4 * value
    want = (rebuilt(value + h) - rebuilt(value - h)) / (2 * h)
    np.testing.assert_allclose(_row(result)[:, 0], want, rtol=2e-5, atol=1e-8)
    assert np.max(np.abs(want)) > 1e-2


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_a_squeezed_batch_has_the_rows_of_its_columns(kind):
    """``run_batch(squeeze=True)`` stacks the rows: each is the derivative of
    the column beside it, in the volume's parameter too."""
    sim = bngsim.Simulator(_model(kind), method="ode", sensitivity_params=["k", "g"])
    stacked = sim.run_batch(params=[{"k": 0.1}, {"k": 0.3}], squeeze=True, **RUN)
    block = np.asarray(stacked.sensitivities)
    column = list(stacked.species_names).index("X")
    np.testing.assert_allclose(block[1, :, column, 0], _dk(k=0.3), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(block[0, :, column, 1], _dg(), rtol=1e-6, atol=1e-9)
