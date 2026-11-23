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

``steady_state()`` reported the same species over the size at load, and its
``dY_ss/dp`` row as the stored amount's, until it ran the same passes (issue
#787): ``H`` made at 1 and lost at ``0.5*H`` in ``V := 1 + P``, with ``P``
settling at 5, came back 2.0 for 2/6, where ``run()`` to the same state
reports 2/6. That section checks it against closed forms.

The last section is ``run()`` again, for an amount that an assignment rule
sets in such a compartment: its row was NaN, refused by name, where the
steady state had it (issue #787).
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

K, G, X0 = 0.1, 0.5, 2.0
RUN = {"t_span": (0.0, 4.0), "n_points": 5, "rtol": 1e-11, "atol": 1e-13}
T = np.linspace(0.0, 4.0, 5)
# The warning that a species row is NaN for want of its size's derivative.
HELD_BEFORE = "derivative of the stored amount, which is what the row held before"
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
    assert any(HELD_BEFORE in str(w.message) and "'X'" in str(w.message) for w in caught)
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
    with pytest.warns(UserWarning, match=HELD_BEFORE):
        result = sim.run(**RUN)
    assert np.isnan(_row(result)).all()
    assert result.ar_sensitivity_refused == frozenset({"X"})


def test_both_kinds_of_refused_row_are_listed():
    """``Z := 3*X`` is an assignment-rule species in the same compartment. Its
    row is its rule's put through the quotient rule (issue #787), which needs
    the size's row as X's does, so it is refused beside X, and the list holds
    both. Its selector is refused by name, as before."""
    text = DECLINED + "substanceOnly species Z in V; Z := 3*X\n"
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["k", "g"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(**RUN)
    assert result.ar_sensitivity_refused == frozenset({"X", "Z"})
    assert np.isnan(_row(result, "Z")).all()
    with pytest.raises(ValueError, match=r"'species:Z' is reported over the live size"):
        result.output_sensitivities("species:Z")


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
      {{more_species}}
    </listOfSpecies>
    <listOfParameters>
      <parameter id="v0" value="{{v0}}" constant="true"/>
      <parameter id="g" value="0.5" constant="true"/>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    {{assignments}}
    <listOfRules><assignmentRule variable="V"><math {_M}>
      <apply><plus/><ci>v0</ci><apply><times/><ci>g</ci>{_TIME}</apply></apply>
    </math></assignmentRule>{{more_rules}}</listOfRules>
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


SET_FROM_THE_SIZE = (
    '<listOfInitialAssignments><initialAssignment symbol="X">'
    f"<math {_M}><apply><times/><cn>2</cn><ci>V</ci></apply></math>"
    "</initialAssignment></listOfInitialAssignments>"
)


A_RULE_TARGET = (
    '<species id="Z" compartment="V" initialConcentration="5" hasOnlySubstanceUnits="true"'
    ' boundaryCondition="false" constant="false"/>'
)
ITS_RULE = (
    f'<assignmentRule variable="Z"><math {_M}><apply><times/><cn>3</cn><ci>X</ci></apply></math>'
    "</assignmentRule>"
)


def _by_rule(
    declared: str,
    v0: float = 1.5,
    assignments: str = "",
    more_species: str = "",
    more_rules: str = "",
) -> bngsim.Model:
    return bngsim.Model.from_sbml_string(
        BY_RULE.format(
            declared=declared,
            v0=v0,
            assignments=assignments,
            more_species=more_species,
            more_rules=more_rules,
        )
    )


@pytest.mark.parametrize(
    "load",
    [
        lambda: _by_rule(HELD_AS_AN_AMOUNT),
        lambda: _by_rule(A_CONCENTRATION_GIVEN_AS_AN_AMOUNT),
    ],
    ids=["a-concentration-held-as-an-amount", "an-amount-held-as-a-concentration"],
)
def test_what_sized_the_compartment_when_a_value_was_converted_is_refused(load):
    """``V := v0 + g*time`` with the species' initial value converted by V(0).
    For a concentration of 2 held as an amount, dX/dv0 is
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


def test_an_initial_assignment_from_such_a_size_was_refused_already():
    """Control. ``X = 2*V`` written as an initialAssignment is not kept as an
    expression, and #696 lists what it reads."""
    model = _by_rule(GIVEN_AS_AN_AMOUNT, assignments=SET_FROM_THE_SIZE)
    assert {"v0", "g"} <= set(model.frozen_params)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="'v0'"):
        bngsim.Simulator(model, method="ode", sensitivity_params=["v0"])


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


def test_a_size_that_stands_still_at_these_values_still_has_its_column():
    """``V := 1 + g*time`` at ``g = 0``: the size is 1 at every sample and the
    column is not rescaled, and dX/dg is ``-X*t`` all the same."""
    text = MODELS["an-assignment-rule"][0].format(v0=1.0).replace("g = 0.5", "g = 0")
    result = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["g"]
    ).run(**RUN)
    np.testing.assert_allclose(_row(result)[:, 0], -_x(g=0.0) * T, rtol=1e-6, atol=1e-9)


def test_an_initial_value_over_a_constant_rule_is_kept_as_an_expression():
    """Control. ``S = 2*q`` under ``q := 3*p``: the rule is a constant over
    parameters, the initial value is kept as an expression of ``p``, and a
    write to ``p`` moves it. Only a rule that reads the time or a state is a
    slot an initial value cannot rest on."""
    model = bngsim.Model.from_antimony_string(
        "species S; p = 2; k = 1; q := 3*p; S = 2*q;\nJ: S -> ; k*S\n"
    )
    assert model.frozen_params == []
    model.set_param("p", 3.0)
    model.reset()
    start = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1.0), n_points=2)
    assert _column(start, "S")[0] == pytest.approx(18.0)


def test_a_rule_s_own_target_in_the_compartment_converts_nothing():
    """``Z := 3*X`` declares a concentration too, which its rule overrides: no
    value was converted, and ``v0`` and ``g`` keep their columns of ``X``."""
    model = _by_rule(GIVEN_AS_AN_AMOUNT, more_species=A_RULE_TARGET, more_rules=ITS_RULE)
    assert model.frozen_params == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bngsim.Simulator(model, method="ode", sensitivity_params=["v0", "g"]).run(**RUN)
    np.testing.assert_allclose(_row(result)[:, 0], -_x(v0=1.5) / (1.5 + G * T), rtol=1e-6)
    np.testing.assert_allclose(_row(result)[:, 1], _dg(1.5), rtol=1e-6, atol=1e-9)


def test_a_size_that_stands_still_under_a_rule_with_no_derivative_is_nan_too():
    """``V := piecewise(1, time < 5, 2)`` sampled before 5: the column is not
    rescaled and the rule's derivative is not to be had. NaN and named, where
    the stored amount's row happens to be right."""
    text = DECLINED.replace("piecewise(1 + g*time, time < 100, 1)", "piecewise(1, time < 5, 2)")
    assert text != DECLINED
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["k"]
    )
    with pytest.warns(UserWarning, match=HELD_BEFORE):
        result = sim.run(**RUN)
    assert np.isnan(_row(result)).all()
    assert "X" in result.ar_sensitivity_refused


# ── The steady state (issue #787) ───────────────────────────────────────────
#
# A size that settles where it did not start: ``V' = a - b*V`` from 1 to
# a/b = 6, or ``V := 1 + P`` with ``P`` made at ``kp`` and lost at ``kd*P``,
# 1 + kp/kd = 6. The amount ``H`` in it is made and lost, set by a rule, or set
# by a rate rule.

SETTLES = (
    "compartment D = 1; species P in D; P = 0; kp = 1; kd = 0.2;\nJ3: => P; kp; J4: P => ; kd*P;\n"
)
SIZES = {
    "a-rate-rule": "compartment V = 1; V' = a - b*V; a = 3; b = 0.5;\n",
    "an-assignment-rule": "compartment V; V := 1 + P;\n",
}
AN_AMOUNT = "substanceOnly species H in V; "
AMOUNTS = {
    "made": AN_AMOUNT + "H = 0; ks = 1; kh = 0.5;\nJ1: => H; ks; J2: H => ; kh*H;\n",
    "set-by-a-rule": AN_AMOUNT + "H := 2*P;\n",
    "set-by-a-rate-rule": AN_AMOUNT + "H = 0; ks = 1; kh = 0.5; H' = ks - kh*H;\n",
}
NOMINAL = {"ks": 1.0, "kh": 0.5, "kp": 1.0, "kd": 0.2, "a": 3.0, "b": 0.5}
SS = {"tol": 1e-12, "max_time": 1e5}


def _settles(size: str, amount: str) -> bngsim.Model:
    return bngsim.Model.from_antimony_string(SETTLES + SIZES[size] + AMOUNTS[amount])


def _settled(size: str, amount: str, p: dict) -> float:
    """[H] at the steady state, the amount over the live size: 2/6, or 10/6
    where the rule sets it to 2*P."""
    v = p["a"] / p["b"] if size == "a-rate-rule" else 1.0 + p["kp"] / p["kd"]
    held = 2.0 * p["kp"] / p["kd"] if amount == "set-by-a-rule" else p["ks"] / p["kh"]
    return held / v


def _settled_row(size: str, amount: str, params: list[str]) -> np.ndarray:
    """d[H]/dp of the closed form, by central differences of it."""
    row = []
    for name in params:
        h = 1e-6 * NOMINAL[name]
        up, down = dict(NOMINAL), dict(NOMINAL)
        up[name] += h
        down[name] -= h
        row.append((_settled(size, amount, up) - _settled(size, amount, down)) / (2 * h))
    return np.array(row)


EVERY_KIND = [(size, amount) for size in sorted(SIZES) for amount in sorted(AMOUNTS)]


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("size,amount", EVERY_KIND)
def test_a_steady_state_reports_the_amount_over_the_live_size(size, amount, method):
    """2/6 or 10/6, where it was 2 or 10: the amount over the size at load."""
    ss = bngsim.Simulator(_settles(size, amount), method="ode").steady_state(method=method, **SS)
    assert ss.converged
    want = _settled(size, amount, NOMINAL)
    assert ss["H"] == pytest.approx(want, rel=1e-8)
    assert ss.to_dict()["H"] == pytest.approx(want, rel=1e-8)
    assert ss["P"] == pytest.approx(5.0, rel=1e-8)


@pytest.mark.parametrize("size,amount", EVERY_KIND)
def test_a_steady_state_is_where_a_long_run_ends(size, amount):
    """Every species, the size's own among them where a rate rule moves it."""
    ss = bngsim.Simulator(_settles(size, amount), method="ode").steady_state(**SS)
    run = bngsim.Simulator(_settles(size, amount), method="ode").run(
        t_span=(0.0, 400.0), n_points=3, rtol=1e-11, atol=1e-13
    )
    for name in run.species_names:
        assert ss[name] == pytest.approx(_column(run, name)[-1], rel=1e-7, abs=1e-12), name


@pytest.mark.parametrize("size,amount", EVERY_KIND)
def test_a_steady_state_row_is_the_derivative_of_what_it_reports(size, amount):
    """The quotient rule through the size's own row. Main returned the row of
    the stored amount over the size at load: d[H]/dks 2 for 1/3 for the amount
    made in either compartment, d[H]/db 0 for 2/3 in ``V' = a - b*V``, and
    d[H]/dkp 10 for 0.2778 for ``H := 2*P`` in ``V := 1 + P``."""
    model = _settles(size, amount)
    params = [name for name in NOMINAL if name in model.param_names]
    ss = bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=params, **SS)
    want = _settled_row(size, amount, params)
    row = np.asarray(ss.sensitivity)[list(ss.species_names).index("H")]
    np.testing.assert_allclose(row, want, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        ss.output_sensitivities(["species:H"])[0], want, rtol=1e-6, atol=1e-9
    )
    assert np.max(np.abs(want)) > 0.1


@pytest.mark.parametrize("size", sorted(SIZES))
def test_each_entry_of_a_steady_state_batch(size):
    """Each entry is reported against its own clone, at its own size."""
    sim = bngsim.Simulator(_settles(size, "made"), method="ode")
    entries = [{"ks": 2.0}] + (
        [{"a": 4.0}, {"b": 0.25}] if size == "a-rate-rule" else [{"kp": 2.0}, {"kd": 0.4}]
    )
    for entry, ss in zip(entries, sim.steady_state_batch(entries, **SS), strict=True):
        assert ss["H"] == pytest.approx(_settled(size, "made", {**NOMINAL, **entry}), rel=1e-8)


@pytest.mark.parametrize("size", sorted(SIZES))
def test_a_steady_concentration_beside_it_is_as_it_was(size):
    """Control. ``G`` is held as a concentration, made at ``kg*V`` and lost at
    ``kh*G*V``: 2 at every size, and its row was right."""
    text = SETTLES + SIZES[size] + AMOUNTS["made"]
    text += "species G in V; G = 0; kg = 1; J5: => G; kg*V; J6: G => ; kh*G*V;\n"
    model = bngsim.Model.from_antimony_string(text)
    ss = bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=["kg", "kh"], **SS)
    assert ss["G"] == pytest.approx(2.0, rel=1e-8)
    row = np.asarray(ss.sensitivity)[list(ss.species_names).index("G")]
    np.testing.assert_allclose(row, [2.0, -4.0], rtol=1e-6)


def test_a_steady_state_with_an_accumulator_left_out():
    """``S`` collects what ``H`` loses at ``0.1*H`` and has no steady value, so
    ``mask=`` leaves it out and its row is NaN, as it was, with nothing said
    about it here. ``H`` is ks/(kh + 0.1) over 6, and its row is right."""
    text = SETTLES + SIZES["an-assignment-rule"] + AMOUNTS["made"]
    text += "substanceOnly species S in V; S = 0; J7: H => S; 0.1*H;\n"
    model = bngsim.Model.from_antimony_string(text)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ss = bngsim.Simulator(model, method="ode").steady_state(
            sensitivity_params=["ks", "kd"], mask=~model.is_pure_sink(), **SS
        )
    names = list(ss.species_names)
    assert np.isnan(np.asarray(ss.sensitivity)[names.index("S")]).all()
    assert ss["H"] == pytest.approx(1.0 / 0.6 / 6.0, rel=1e-8)
    want = [1.0 / 0.6 / 6.0, (1.0 / 0.6) / 36.0 * 25.0]
    np.testing.assert_allclose(np.asarray(ss.sensitivity)[names.index("H")], want, rtol=1e-6)


def test_a_steady_state_that_left_out_the_size_refuses_the_row():
    """``mask=`` leaves out ``V`` and keeps ``H``, whose row needs ``V``'s.
    Main returned the stored amount's row. Refused, by name."""
    model = _settles("a-rate-rule", "made")
    keep = [name for name in model.species_names if name != "V"]
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"\['H'\].*mask="):
        sim.steady_state(sensitivity_params=["ks", "a"], mask=keep, **SS)


# ── An amount a rule sets, on a time course (issue #787) ────────────────────
#
# ``H := 2*P`` in either compartment above, run rather than settled. ``run()``
# reports the rule over the live size, and its row was NaN, refused by name,
# where the steady state above gives it: the varvol passes' quotient rule ran
# before the rule's row was put in. With P(t) = P0*e^(-kd*t) + (kp/kd)*(1 -
# e^(-kd*t)) and V either 1 + P or a/b + (1 - a/b)*e^(-b*t), the reported
# column has a closed form. Each row is checked against differences of the
# closed form and against central differences of runs.


def _p_of_t(p: dict, p0: float = 0.0) -> np.ndarray:
    decay = np.exp(-p["kd"] * T)
    return p0 * decay + p["kp"] / p["kd"] * (1.0 - decay)


def _live_size(size: str, p: dict, p0: float = 0.0) -> np.ndarray:
    if size == "a-rate-rule":
        held = p["a"] / p["b"]
        return held + (1.0 - held) * np.exp(-p["b"] * T)
    return 1.0 + _p_of_t(p, p0)


def _ruled(size: str, p: dict, p0: float = 0.0) -> np.ndarray:
    """[H] = 2P/V_live."""
    return 2.0 * _p_of_t(p, p0) / _live_size(size, p, p0)


def _ruled_rows(size: str, params: list[str], p0: float = 0.0) -> np.ndarray:
    """d[H]/dp of the closed form, by central differences of it."""
    rows = []
    for name in params:
        h = 1e-6 * NOMINAL[name]
        up, down = dict(NOMINAL), dict(NOMINAL)
        up[name] += h
        down[name] -= h
        rows.append((_ruled(size, up, p0) - _ruled(size, down, p0)) / (2 * h))
    return np.stack(rows, axis=1)


def _ruled_text(size: str, p0: float = 0.0, extra: str = "") -> str:
    return (SETTLES + SIZES[size] + AMOUNTS["set-by-a-rule"] + extra).replace(
        "P = 0;", f"P = {p0};"
    )


@pytest.mark.parametrize("p0", [0.0, 1.0])
@pytest.mark.parametrize("size", sorted(SIZES))
def test_a_rule_s_amount_on_a_time_course_has_its_row(size, p0):
    """At t = 4 in ``V := 1 + P`` from P = 0, d[H]/dkp is 0.3909 and d[H]/dkd
    -0.6786; the row was NaN, and H was in ``ar_sensitivity_refused``. From
    P = 1 the size at load is 2, which the rule's value is divided by before
    the live size replaces it."""
    text = _ruled_text(size, p0)
    model = bngsim.Model.from_antimony_string(text)
    params = [name for name in NOMINAL if name in model.param_names]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(**RUN)
    assert result.ar_sensitivity_refused == frozenset()
    np.testing.assert_allclose(_column(result, "H"), _ruled(size, NOMINAL, p0), rtol=1e-8)
    row = _row(result, "H")
    np.testing.assert_allclose(row, _ruled_rows(size, params, p0), rtol=1e-6, atol=1e-9)
    for c, name in enumerate(params):
        want = _by_plain_runs(text, "H", name, NOMINAL[name])
        np.testing.assert_allclose(row[:, c], want, rtol=2e-5, atol=1e-8, err_msg=name)
    np.testing.assert_array_equal(result.output_sensitivities("species:H")[:, 0, :], row)
    if size == "an-assignment-rule" and p0 == 0.0:
        np.testing.assert_allclose(row[-1], [0.3909, -0.6786], atol=1e-4)


@pytest.mark.parametrize("size", sorted(SIZES))
def test_a_rule_s_amount_has_the_row_of_what_it_reads_at_the_start(size):
    """The initial-condition axis: d[H]/dP(0) is 2*e^(-kd*t)/V, less, where P
    sizes the compartment, [H]*e^(-kd*t)/V. It was NaN."""
    text = _ruled_text(size)
    model = bngsim.Model.from_antimony_string(text)
    result = bngsim.Simulator(model, method="ode", sensitivity_ic=["P"]).run(**RUN)
    row = np.asarray(result.sensitivities_ic)[:, list(result.species_names).index("H"), 0]
    decay = np.exp(-NOMINAL["kd"] * T)
    v = _live_size(size, NOMINAL)
    dv = decay if size == "an-assignment-rule" else 0.0
    want = 2.0 * decay / v - _ruled(size, NOMINAL) * dv / v
    np.testing.assert_allclose(row, want, rtol=1e-6, atol=1e-9)
    h = 1e-5

    def column(at):
        m = bngsim.Model.from_antimony_string(text)
        m.set_concentration("P", at)
        return _column(bngsim.Simulator(m, method="ode").run(**RUN), "H")

    np.testing.assert_allclose(row, (column(h) - column(-h)) / (2 * h), rtol=2e-5, atol=1e-8)


def test_a_rule_s_amount_beside_a_concentration_whose_dilution_reads_a_rate():
    """A concentration ``G`` in ``V := 1 + P`` is diluted at the rate of P, and
    the dilution term reads that rate (issue #1052). Codegen declined the
    output sensitivity of every expression for it, the size's own among them,
    so H's row had no dV/dθ to take: NaN. Only the dilution term reads a rate,
    and only it is declined now."""
    text = _ruled_text(
        "an-assignment-rule", extra="species G in V; G = 2; kg = 0.5; J5: G => ; kg*G;\n"
    )
    params = ["kp", "kd"]
    course = {**RUN, "rtol": 1e-10, "atol": 1e-12}
    model = bngsim.Model.from_antimony_string(text)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(**course)
    assert result.ar_sensitivity_refused == frozenset()
    row = _row(result, "H")
    np.testing.assert_allclose(
        row, _ruled_rows("an-assignment-rule", params), rtol=1e-6, atol=1e-9
    )
    for c, name in enumerate(params):
        want = _by_plain_runs(text, "H", name, NOMINAL[name])
        np.testing.assert_allclose(row[:, c], want, rtol=2e-5, atol=1e-8, err_msg=name)
    size = np.asarray(result.output_sensitivities("expression:V"))[:, 0, :]
    np.testing.assert_allclose(size[-1], [2.7534, -4.7802], atol=1e-4)
