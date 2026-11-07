"""SBML ``quotient`` and ``rem`` truncate, as MathML defines them (issue #1522).

SBML L3V2 takes ``<quotient/>`` and ``<rem/>`` from MathML 2.0, whose Appendix
C.2.3.1 defines ``quotient(a, b)`` as the q with a = b*q + r, |r| < |b| and
a*r >= 0, and C.2.3.9 has a = b*quotient(a, b) + rem(a, b). That is C's
truncation: quotient(-5, 3) = -1 and rem(-5, 3) = -2, the remainder taking a's
sign. libRoadRunner and COPASI read them so, and so does bngsim's own reading of
libSBML's ``%``. Both of bngsim's MathML translators floored instead, so a
negative operand gave quotient(-5, 3) = -2 and rem(-5, 3) = 1: in rules,
kinetic laws and initialAssignments, interpreted and compiled, with no warning.

Oracle: ``math.trunc`` and ``math.fmod``, which are MathML's definitions; and
for an operand that moves with time, the libSBML ``%`` form of the same
remainder, which bngsim already read as MathML does.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
_MATH = '<math xmlns="http://www.w3.org/1998/Math/MathML">'

# (a, b): every sign, a non-integer operand, an exact multiple, 0, and |a| < |b|.
OPERANDS = [
    (-5, 3),
    (5, -3),
    (-5, -3),
    (5, 3),
    (-6, 3),
    (-5.5, 2.0),
    (0, 3),
    (-0.5, 3.0),
    (-5.0, 3.0),
]
IDS = [f"{a}_{b}" for a, b in OPERANDS]


def _q(a, b):
    return float(math.trunc(a / b))


def _r(a, b):
    return math.fmod(a, b)


def _cn(v):
    """An integer as libSBML's AST_INTEGER, anything else as AST_REAL."""
    return f'<cn type="integer">{v}</cn>' if isinstance(v, int) else f"<cn>{v}</cn>"


def _apply(op, a, b):
    return f"<apply><{op}/>{_cn(a)}{_cn(b)}</apply>"


def _rules_model(a, b, *, as_rule):
    """Parameters q and r set to quotient(a, b) and rem(a, b), by an assignment
    rule (the runtime emit) or by an initialAssignment (the load-time fold)."""
    params = '<parameter id="q" constant="false"/><parameter id="r" constant="false"/>'
    if as_rule:
        body = "".join(
            f'<assignmentRule variable="{v}">{_MATH}{_apply(op, a, b)}</math></assignmentRule>'
            for v, op in (("q", "quotient"), ("r", "rem"))
        )
        block = f"<listOfRules>{body}</listOfRules>"
    else:
        body = "".join(
            f'<initialAssignment symbol="{v}">{_MATH}{_apply(op, a, b)}</math></initialAssignment>'
            for v, op in (("q", "quotient"), ("r", "rem"))
        )
        block = f"<listOfInitialAssignments>{body}</listOfInitialAssignments>"
    return (
        f'{_HDR}\n<model id="m"><listOfParameters>{params}</listOfParameters>{block}'
        "</model></sbml>"
    )


@pytest.mark.parametrize(("a", "b"), OPERANDS, ids=IDS)
def test_an_initial_assignment_folds_them_truncated(a, b):
    """On main, (-5, 3) folded to q = -2 and r = 1."""
    model = bngsim.Model.from_sbml_string(_rules_model(a, b, as_rule=False))
    assert model.get_param("q") == _q(a, b)
    assert model.get_param("r") == pytest.approx(_r(a, b), abs=1e-15)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
@pytest.mark.parametrize(("a", "b"), OPERANDS, ids=IDS)
def test_an_assignment_rule_reads_them_truncated(a, b, codegen):
    """On main, (-5, 3) read q = -2 and r = 1 at run time, compiled or not."""
    model = bngsim.Model.from_sbml_string(_rules_model(a, b, as_rule=True))
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(t_span=(0, 1), n_points=2)
    exprs = {n: float(np.asarray(result.expressions[n])[-1]) for n in ("q", "r")}
    assert exprs["q"] == _q(a, b)
    assert exprs["r"] == pytest.approx(_r(a, b), abs=1e-15)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_a_kinetic_law_reads_them_truncated(codegen):
    """S' = -(4 + rem(-5, 3))·S and X' = quotient(-5, 3) + 2: S(1) = e^-2 and
    X(1) = 1. On main the rate constants came out 5 and 0: S(1) = e^-5, X(1) = 0."""
    law_s = (
        f"<apply><times/><apply><plus/><cn>4</cn>{_apply('rem', -5, 3)}</apply><ci>S</ci></apply>"
    )
    law_x = f"<apply><plus/>{_apply('quotient', -5, 3)}<cn>2</cn></apply>"
    sbml = (
        f'{_HDR}\n<model id="m">'
        '<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>'
        "<listOfSpecies>"
        '<species id="S" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false"'
        ' boundaryCondition="false" constant="false"/>'
        '<species id="X" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false"'
        ' boundaryCondition="false" constant="false"/>'
        "</listOfSpecies><listOfReactions>"
        '<reaction id="J0" reversible="false"><listOfReactants><speciesReference species="S"'
        ' stoichiometry="1" constant="true"/></listOfReactants>'
        f"<kineticLaw>{_MATH}{law_s}</math></kineticLaw></reaction>"
        '<reaction id="J1" reversible="false"><listOfProducts><speciesReference species="X"'
        ' stoichiometry="1" constant="true"/></listOfProducts>'
        f"<kineticLaw>{_MATH}{law_x}</math></kineticLaw></reaction>"
        "</listOfReactions></model></sbml>"
    )
    model = bngsim.Model.from_sbml_string(sbml)
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        t_span=(0, 1), n_points=2, rtol=1e-10, atol=1e-12
    )
    names = list(result.species_names)
    final = np.asarray(result.species)[-1]
    assert final[names.index("S")] == pytest.approx(math.exp(-2.0), rel=1e-7)
    assert final[names.index("X")] == pytest.approx(1.0, rel=1e-7)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_a_remainder_of_the_time_less_a_number_truncates(codegen):
    """The issue's rem(time - 5, 3) at t = 1 is rem(-4, 3) = -1, and
    quotient(time - 5, 3) is -1; on main, 2 and -2. A difference is not
    non-negative by its form, so it takes the sign test (the issue's own row)."""
    time = "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>"
    arg = f"<apply><minus/>{time}<cn>5</cn></apply><cn>3</cn>"
    rules = "".join(
        f'<assignmentRule variable="{v}">{_MATH}<apply><{op}/>{arg}</apply></math>'
        "</assignmentRule>"
        for v, op in (("q", "quotient"), ("r", "rem"))
    )
    sbml = (
        f'{_HDR}\n<model id="m"><listOfParameters>'
        '<parameter id="q" constant="false"/><parameter id="r" constant="false"/>'
        f"</listOfParameters><listOfRules>{rules}</listOfRules></model></sbml>"
    )
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(sbml), method="ode", codegen=codegen
    ).run(t_span=(0, 1), n_points=2)
    assert float(np.asarray(result.expressions["q"])[-1]) == -1.0
    assert float(np.asarray(result.expressions["r"])[-1]) == pytest.approx(-1.0, abs=1e-15)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_a_remainder_of_the_time_truncates_on_a_run_from_before_0(codegen):
    """A run can start before 0, so the time is not non-negative by its form.
    Over t = -4.5 .. 1.5, rem(time, 3) is fmod(t, 3), -1.5 at t = -4.5, and
    quotient(time, 3) is trunc(t/3), -1 there; read as the floor they were 1.5
    and -2, wrong at every row before 0. The bound is rounding: each row's t is
    a sum of steps from -4.5."""
    time = "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>"
    rules = "".join(
        f'<assignmentRule variable="{v}">{_MATH}<apply><{op}/>{time}<cn type="integer">3</cn>'
        "</apply></math></assignmentRule>"
        for v, op in (("q", "quotient"), ("r", "rem"))
    )
    sbml = (
        f'{_HDR}\n<model id="m"><listOfParameters>'
        '<parameter id="q" constant="false"/><parameter id="r" constant="false"/>'
        f"</listOfParameters><listOfRules>{rules}</listOfRules></model></sbml>"
    )
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(sbml), method="ode", codegen=codegen
    ).run(t_span=(-4.5, 1.5), n_points=7)
    t = np.asarray(result.time)
    assert t[0] == -4.5
    np.testing.assert_array_equal(np.asarray(result.expressions["q"]), [_q(x, 3) for x in t])
    np.testing.assert_allclose(
        np.asarray(result.expressions["r"]), [_r(x, 3) for x in t], rtol=0, atol=1e-12
    )


def _schedule_sbml(t0, *, native):
    """X is made at k while rem(time - t0, P) >= d, and lost at kd. ``native``
    writes <rem/>; otherwise libSBML's own `%`, which bngsim read as MathML does."""
    antimony = pytest.importorskip("antimony")
    rem = "rem(time - t0, P)" if native else "(time - t0) % P"
    ant = (
        f"model m\n compartment C = 1\n species X in C = 0\n"
        f" k = 0.3; kd = 0.1; P = 24; d = 7; t0 = {t0}\n"
        f" J0: => X; k*piecewise(1, {rem} >= d, 0)\n J1: X => ; kd*X\nend"
    )
    assert antimony.loadAntimonyString(ant) >= 0, antimony.getLastError()
    sbml = antimony.getSBMLString("m")
    assert ("<rem/>" in sbml) == native
    return sbml


@pytest.mark.parametrize("t0", [28.0, 0.0, -5.0])
def test_a_remainder_over_time_is_the_one_libsbml_writes(t0):
    """Before t0 the remainder is negative, so the schedule is off: native <rem/>
    now runs the same trajectory as libSBML's `%` of the same operands. On main,
    at t0 = 28, native <rem/> turned the schedule on for 10 hours before t0 and
    gave X(60) = 1.4908 for 1.3872."""
    t = np.linspace(0.0, 60.0, 121)
    runs = [
        np.asarray(
            bngsim.Simulator(
                bngsim.Model.from_sbml_string(_schedule_sbml(t0, native=native)), method="ode"
            )
            .run(sample_times=t, rtol=1e-10, atol=1e-12)
            .species
        )[:, 0]
        for native in (True, False)
    ]
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-8, atol=1e-12)


def test_a_remainder_schedule_from_zero_keeps_its_sensitivities():
    """rem(time, P) >= d is still read as the repeating schedule it is: the
    sensitivity run is compensated, and its column matches a central difference
    of plain runs."""
    antimony = pytest.importorskip("antimony")
    ant = (
        "model m\n compartment C = 1\n species X in C = 0\n"
        " k = 0.3; kd = 0.1; P = 24; d = 7\n"
        " J0: => X; k*piecewise(1, rem(time, P) >= d, 0)\n J1: X => ; kd*X\nend"
    )
    assert antimony.loadAntimonyString(ant) >= 0, antimony.getLastError()
    sbml = antimony.getSBMLString("m")
    assert "<rem/>" in sbml
    run = {"t_span": (0.0, 60.0), "n_points": 61}

    def plain(k):
        model = bngsim.Model.from_sbml_string(sbml)
        model.set_param("k", k)
        r = bngsim.Simulator(model, method="ode").run(**run, rtol=1e-11, atol=1e-13)
        return np.asarray(r.species)[:, 0]

    h = 0.3e-4
    fd = (plain(0.3 + h) - plain(0.3 - h)) / (2 * h)
    sens = bngsim.Simulator(
        bngsim.Model.from_sbml_string(sbml), method="ode", sensitivity_params=["k"]
    ).run(**run, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(sens.sensitivities)[:, 0, 0], fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max()
    )


def test_a_converted_net_keeps_them_truncated(tmp_path):
    """bngsim.convert writes the .net from the loaded model, so it carried the
    floor too; read back, X(1) = 0 for 1."""
    from bngsim import convert

    law_x = f"<apply><plus/>{_apply('quotient', -5, 3)}<cn>2</cn></apply>"
    sbml = (
        f'{_HDR}\n<model id="m">'
        '<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>'
        '<listOfSpecies><species id="X" compartment="C" initialConcentration="0"'
        ' hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        "</listOfSpecies><listOfReactions>"
        '<reaction id="J1" reversible="false"><listOfProducts><speciesReference species="X"'
        ' stoichiometry="1" constant="true"/></listOfProducts>'
        f"<kineticLaw>{_MATH}{law_x}</math></kineticLaw></reaction>"
        "</listOfReactions></model></sbml>"
    )
    src, net = tmp_path / "m.xml", tmp_path / "m.net"
    src.write_text(sbml)
    convert.sbml_to_net(str(src), str(net))
    result = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
        t_span=(0, 1), n_points=2, rtol=1e-10, atol=1e-12
    )
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(1.0, rel=1e-8)


def _emitted(mathml):
    import libsbml
    from bngsim import _sbml_loader

    return _sbml_loader._ast_to_exprtk_recursive(libsbml.readMathMLFromString(mathml), {}, {})


@pytest.mark.parametrize(
    "operands",
    [
        "<cn>9</cn><cn>2</cn>",
        "<apply><plus/><cn>9</cn><cn>0.5</cn></apply><apply><times/><cn>2</cn><cn>3</cn></apply>",
    ],
    ids=["two numbers", "a sum and a product of numbers"],
)
def test_operands_non_negative_by_their_form_keep_the_floor_text(operands):
    """Where both operands are non-negative numbers by their form, truncation is
    the floor, and the text written is the one main wrote, so nothing that read
    it reads anything new."""
    for op in ("quotient", "rem"):
        text = _emitted(f"{_MATH}<apply><{op}/>{operands}</apply></math>")
        assert "ceil" not in text and "if(" not in text, text


@pytest.mark.parametrize(
    "operand",
    [
        "<ci>a</ci>",
        "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>",
        "<apply><minus/><cn>3</cn><cn>5</cn></apply>",
    ],
    ids=["a parameter", "the time", "a difference of numbers"],
)
def test_a_parameter_the_time_or_a_difference_takes_the_sign_test(operand):
    """A parameter can be set negative after load, a run can start before 0,
    and a difference of non-negative numbers can be negative, so each one's
    remainder truncates by the sign test, whatever it holds at load."""
    text = _emitted(f"{_MATH}<apply><rem/>{operand}<cn>3</cn></apply></math>")
    assert "ceil" in text and "floor" in text and "if(" in text, text


def test_a_far_line_quotient_of_the_time_is_refused():
    """quotient(1e10 + time, 0.01) is floor(1e12 + 100t) on its treads from 0,
    which main integrated exactly to 1e12 + 49.5 over (0, 1). It reads the time,
    which a run can start before 0, so it takes the sign test, whose ceil is a
    second step on the far line: the far line's reading (#1018) refuses the run
    by name rather than read it wrong."""
    pytest.importorskip("antimony")
    import antimony

    ant = (
        "model m\n compartment C = 1\n species X in C = 0\n"
        " J0: => X; quotient(1e10 + time, 0.01)\nend"
    )
    assert antimony.loadAntimonyString(ant) >= 0, antimony.getLastError()
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(antimony.getSBMLString("m")), method="ode"
    )
    with pytest.raises(bngsim.SimulationError, match="switches on"):
        sim.run(t_span=(0, 1), n_points=2, rtol=1e-8, atol=1e-10)


def test_a_far_line_remainder_by_a_parameter_is_refused():
    """rem(1.1e12 + time, P), P = 0.0048828125, integrates to 0.2317905426 tread
    by tread. Main ran it to 0.2317929216, 1e-5 off at rtol 1e-8 with no error:
    the stops the far line needs were not there (#1018). Its sign test makes it a
    law the far-line reading refuses, by name."""
    pytest.importorskip("antimony")
    import antimony

    ant = (
        "model m\n compartment C = 1\n species X in C = 0\n P = 0.0048828125\n"
        " J0: => X; 100*rem(1.1e12 + time, P)\nend"
    )
    assert antimony.loadAntimonyString(ant) >= 0, antimony.getLastError()
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(antimony.getSBMLString("m")), method="ode"
    )
    with pytest.raises(bngsim.SimulationError, match="switches on"):
        sim.run(t_span=(0, 1), n_points=2, rtol=1e-8, atol=1e-10)
