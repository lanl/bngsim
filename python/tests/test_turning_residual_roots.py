"""Issue #904, part 2: a window whose residual turns within a step.

``if(abs(Aobs-5) < 0.01, k, 0)`` and ``if((Aobs-5)^2 < 0.0001, k, 0)`` are the
window ``4.99 < A < 5.01``. Each is one comparison, and its residual
(``abs(A-5) - 0.01``, ``(A-5)^2 - 1e-4``) falls to zero and comes back up as A
passes 5: within one step it has the same sign at both ends, so a root on it
reports nothing, and the window was stepped over. Y came back 0 for 0.1 at
every tolerance, and with sensitivities ``(A-5)^2`` gave Y = 0 and dY/dk = 0.

Such an atom is now rooted at its edges, ``A - 5`` against ``0.01`` and
``-0.01``, each monotone along a flow that moves A one way, and on the
sensitivity path jumped there as any crossing is.

A = 2t, so A is in the window for 0.01 of time and Y = k * 0.01. With the rate
``k*A`` and the window centred on a parameter c, Y = k * 0.01 * c.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SensitivityUnsupportedError
from bngsim._switch_sensitivity import (
    _monotone_pieces,
    state_switch_conditions,
    state_switch_reactions,
)

NET = """begin parameters
    1 k 10
    2 two 2
    3 c 5
end parameters
begin functions
    1 f() {cond}
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 two
    2 0 2 f
end reactions
begin groups
    1 Aobs 1
end groups
"""

WINDOWS = [
    "if((Aobs>4.99)&&(Aobs<5.01),k,0)",
    "if(abs(Aobs-5)<0.01,k,0)",
    "if((Aobs-5)^2<0.0001,k,0)",
    "k*(abs(Aobs-5)<0.01)",
]
IDS = ["two-comparisons", "abs", "square", "abs-as-a-number"]


def _model(tmp_path, cond: str) -> bngsim.Model:
    path = tmp_path / "window.net"
    path.write_text(NET.format(cond=cond))
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
@pytest.mark.parametrize("cond", WINDOWS, ids=IDS)
def test_a_window_one_comparison_spells_is_not_stepped_over(tmp_path, cond, rtol):
    run = bngsim.Simulator(_model(tmp_path, cond), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(0.1, rel=1e-5)


@pytest.mark.parametrize("cond", WINDOWS[1:], ids=IDS[1:])
def test_its_edges_are_the_roots(tmp_path, cond):
    roots = _model(tmp_path, cond).state_switch_root_conditions()
    assert len(roots) == 2
    assert all("Aobs-5" in r and "abs" not in r and "^" not in r for r in roots)


@pytest.mark.parametrize(
    "cond",
    ["if((Aobs>c-0.01)&&(Aobs<c+0.01),k*Aobs,0)", "if((Aobs-c)^2<0.0001,k*Aobs,0)"],
    ids=["two-comparisons", "square"],
)
def test_the_sensitivity_jumps_at_each_edge(tmp_path, cond):
    """Y = k * 0.01 * c: dY/dc = 0.1 moves both edges, dY/dk = 0.05."""
    sim = bngsim.Simulator(_model(tmp_path, cond), method="ode", sensitivity_params=["c", "k"])
    run = sim.run(t_span=(0.0, 10.0), n_points=2, rtol=1e-9, atol=1e-12)
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(0.5, rel=1e-6)
    d_c, d_k = np.asarray(run.sensitivities)[-1, 1, :]
    assert d_c == pytest.approx(0.1, rel=1e-5)
    assert d_k == pytest.approx(0.05, rel=1e-6)


@pytest.mark.parametrize("cond", WINDOWS[1:3], ids=IDS[1:3])
def test_each_edge_is_read_by_the_reaction_that_reads_the_window(tmp_path, cond):
    """The jump at an edge is judged on the flux of the reactions that read it
    (issue #763); an edge no reaction is listed for is taken as continuous."""
    core = _model(tmp_path, cond)._core
    conditions = state_switch_conditions(core)
    assert len(conditions) == 2
    assert state_switch_reactions(core, conditions) == [[1], [1]]


@pytest.mark.parametrize(
    ("atom", "pieces"),
    [
        ("abs(Aobs-5)<0.01", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-(0.01))"]),
        ("0.01>abs(Aobs-5)", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-(0.01))"]),
        # Each edge keeps the operator written, the far one mirrored, so the
        # crossing stop places it where the atom's truth changes (issue #1673).
        ("abs(Aobs-c)<=w", ["(Aobs-c)<=(w)", "(Aobs-c)>=(-(w))"]),
        ("w>=abs(Aobs-c)", ["(Aobs-c)<=(w)", "(Aobs-c)>=(-(w))"]),
        ("abs(Aobs-5)>=0.5", ["(Aobs-5)>=(0.5)", "(Aobs-5)<=(-(0.5))"]),
        ("(Aobs-5)^2<0.0001", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-0.01)"]),
        ("Aobs^2>4", ["(Aobs)>(2.0)", "(Aobs)<(-2.0)"]),
        ("4<Aobs^2", ["(Aobs)>(2.0)", "(Aobs)<(-2.0)"]),
        # as the SBML loader writes a condition
        ("(abs((A-5))<0.01)", ["((A-5))<(0.01)", "((A-5))>(-(0.01))"]),
    ],
)
def test_the_pieces_of_a_turning_comparison(atom, pieces):
    assert _monotone_pieces(atom) == pieces


@pytest.mark.parametrize(
    "atom",
    [
        "Aobs<5",  # monotone already
        "abs(A)*abs(B)<1",  # not one abs()
        "abs(A)<abs(B)",  # turns on both sides
        "(A-5)^2<w",  # a square against a bound that is not a number
        "(A-5)^2<-1",  # against a negative one: no edge at all
        "(A-5)^2<0",  # against 0: one point, no window
        "abs(A-5)<-1",  # no window either
        "abs(A-5)>0",  # false at one point only
        "A^2.5<1",  # not a square
        "(A-5)^2+(B-5)^2<1",  # a sum: only its last group is squared
        "(A-5)*(B)^2<4",  # a product: likewise
        "trig>abs(-1)",  # an abs() of a number does not turn
        "(2-5)^2<1",  # nor does a square of one
    ],
)
def test_other_comparisons_are_left_as_they_are(atom):
    assert _monotone_pieces(atom) == []


DISK = """begin parameters
    1 k 10
    2 five 5
    3 one 1
    4 c 5
end parameters
begin functions
    1 f() if((Aobs-c)^2+(Bobs-5)^2<0.75,k,0)
end functions
begin species
    1 A() 0
    2 Y() 0
    3 B() 5.25
end species
begin reactions
    1 0 1 five
    2 1 0 one
    3 0 2 f
end reactions
begin groups
    1 Aobs 1
    2 Bobs 3
end groups
"""


def test_a_sum_of_squares_keeps_its_own_root(tmp_path):
    """``(A-c)^2+(B-5)^2`` squares its last group only, so it is no square of
    one side and its comparison is rooted whole, at the crossing it has.

    A = 5(1 - e^-t) enters the disk and stays: with B = 5.25 it is inside once
    25 e^(-2t) < 0.6875, from t* = -ln(0.0275)/2, so Y = k(10 - t*) and
    dY/dc = -k/A'(t*) = -2/sqrt(0.0275). Read as the square of
    ``(A-c)^2+(B-5)``, the disk was rooted at A - c = -0.866, where nothing
    switches, its own crossing was not jumped, and dY/dc came back 0.
    """
    path = tmp_path / "disk.net"
    path.write_text(DISK)
    roots = bngsim.Model.from_net(str(path)).state_switch_root_conditions()
    assert roots == ("(Aobs-c)^2+(Bobs-5)^2<0.75",)
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["c", "k"]
    )
    run = sim.run(t_span=(0.0, 10.0), n_points=2, rtol=1e-9, atol=1e-12)
    t_star = -np.log(0.0275) / 2
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(10 * (10 - t_star), rel=1e-6)
    d_c, d_k = np.asarray(run.sensitivities)[-1, 1, :]
    assert d_c == pytest.approx(-2 / np.sqrt(0.0275), rel=1e-5)
    assert d_k == pytest.approx(10 - t_star, rel=1e-6)


def _sbml_window(
    tmp_path,
    condition: str | None = None,
    *,
    trigger: str | None = None,
    rules: dict[str, str] | None = None,
    functions: dict[str, str] | None = None,
) -> str:
    """A = 2t as an SBML document, with Y' = piecewise(k, condition, 0), or
    with an event on *trigger* that adds 1 to Y and writes the time to T.
    *rules* are parameters set by assignment rules, by name, and *functions*
    are function definitions, by name."""
    import libsbml

    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("window")
    comp = model.createCompartment()
    comp.setId("C")
    comp.setSize(1)
    comp.setConstant(True)
    for sid in ("A", "Y", "T"):
        sp = model.createSpecies()
        sp.setId(sid)
        sp.setCompartment("C")
        sp.setInitialConcentration(0)
        sp.setHasOnlySubstanceUnits(False)
        sp.setBoundaryCondition(False)
        sp.setConstant(False)
    k = model.createParameter()
    k.setId("k")
    k.setValue(10)
    k.setConstant(True)
    for name, body in (functions or {}).items():
        function = model.createFunctionDefinition()
        function.setId(name)
        function.setMath(libsbml.parseL3Formula(body))
    for name, formula in (rules or {}).items():
        param = model.createParameter()
        param.setId(name)
        param.setConstant(False)
        rule = model.createAssignmentRule()
        rule.setVariable(name)
        rule.setMath(libsbml.parseL3Formula(formula))
    laws = [("R1", "A", "2")]
    if condition is not None:
        laws.append(("R2", "Y", f"piecewise(k, {condition}, 0)"))
    for rid, product, law in laws:
        rxn = model.createReaction()
        rxn.setId(rid)
        rxn.setReversible(False)
        ref = rxn.createProduct()
        ref.setSpecies(product)
        ref.setStoichiometry(1)
        ref.setConstant(True)
        rxn.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    if trigger is not None:
        event = model.createEvent()
        event.setId("E")
        event.setUseValuesFromTriggerTime(True)
        when = event.createTrigger()
        when.setMath(libsbml.parseL3Formula(trigger))
        when.setInitialValue(False)
        when.setPersistent(True)
        for target, value in (("Y", "Y + 1"), ("T", "time")):
            assignment = event.createEventAssignment()
            assignment.setVariable(target)
            assignment.setMath(libsbml.parseL3Formula(value))
    path = tmp_path / "window.xml"
    path.write_text(libsbml.writeSBMLToString(doc))
    return str(path)


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
@pytest.mark.parametrize(
    "condition",
    ["abs(A - 5) < 0.01", "pow(A - 5, 2) < 0.0001", "abs(time - 2.5) < 0.005"],
    ids=["abs", "pow", "abs-of-time"],
)
def test_an_sbml_kinetic_law_spelled_so_is_not_stepped_over(tmp_path, condition, rtol):
    """The SBML loader registers the comparison's two edges, so the run has a
    root or a stop at each and roots nothing besides. It registered the whole
    comparison, which turns within the step as the ``.net`` one does; on time,
    the scan that recovers clock conditions then found the whole one again
    beside its edges."""
    path = _sbml_window(tmp_path, condition)
    model = bngsim.Model.from_sbml(path)
    covered = model.time_discontinuity_conditions()
    assert len(covered) == 2
    assert all("abs" not in c and "^" not in c for c in covered)
    assert model.state_switch_root_conditions() == ()
    run = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    y = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))["Y"]
    assert y == pytest.approx(0.1, rel=1e-5)


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_a_square_against_a_rule_bound_keeps_the_rate_laws_edges(tmp_path, rtol):
    """The loader cannot split ``(A - 5)^2 < w``: ``w`` is no number in its
    text, so it registers the whole comparison, whose root never fires. The
    rate-law scan writes ``w := 0.0001`` in and roots the edges. Read as
    covering those edges, the whole one left the window stepped over: Y = 0."""
    path = _sbml_window(tmp_path, "(A - 5)^2 < w", rules={"w": "0.0001"})
    roots = bngsim.Model.from_sbml(path).state_switch_root_conditions()
    assert roots == ("(A-5)<(0.01)", "(A-5)>(-0.01)")
    run = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    y = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))["Y"]
    assert y == pytest.approx(0.1, rel=1e-5)


def _fired(
    tmp_path, trigger: str, rtol: float, functions: dict[str, str] | None = None
) -> tuple[float, float]:
    """How often the event on *trigger* fired by t = 10, and when it last did."""
    model = bngsim.Model.from_sbml(_sbml_window(tmp_path, trigger=trigger, functions=functions))
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    last = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))
    return float(last["Y"]), float(last["T"])


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
@pytest.mark.parametrize(
    "trigger",
    ["abs(A - 5) < 0.01", "0.01 > abs(A - 5)", "(A - 5)^2 < 0.0001", "pow(A - 5, 2) < 0.0001"],
    ids=["abs", "abs-reversed", "square", "pow"],
)
def test_an_sbml_event_on_a_turning_trigger_fires(tmp_path, trigger, rtol):
    """The trigger holds while 4.99 < A < 5.01, from t = 2.495. Rooted whole,
    it is false at both ends of the step that passes A = 5, and the event
    never fired: Y = 0 at both tolerances, where ``(A > 4.99) && (A < 5.01)``
    fired."""
    y, fired_at = _fired(tmp_path, trigger, rtol)
    assert y == 1.0
    assert fired_at == pytest.approx(2.495, rel=1e-6)


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_an_sbml_event_outside_a_window_fires_again_after_it(tmp_path, rtol):
    """``abs(A - 5) > 0.01`` holds at the start and fires there, is false only
    while A passes 5, and fires again at t = 2.505. It fired once."""
    y, fired_at = _fired(tmp_path, "abs(A - 5) > 0.01", rtol)
    assert y == 2.0
    assert fired_at == pytest.approx(2.505, rel=1e-6)


NEAR = "lambda(x, c, w, abs(x - c) < w)"


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
@pytest.mark.parametrize(
    ("trigger", "body", "edges"),
    [
        ("near(A, 5, 0.01)", NEAR, {"A-5<0.01", "A-5>-0.01"}),
        ("near(A, 4.99, 5.01)", "lambda(x, lo, hi, (x > lo) && (x < hi))", {"A>4.99", "A<5.01"}),
        ("near(time, 2.5, 0.005)", NEAR, {"time-2.5<0.005", "time-2.5>-0.005"}),
        # The formal A is what the call binds, time, and not the species A.
        (
            "near(time, 2.5, 0.005)",
            "lambda(A, c, w, abs(A - c) < w)",
            {"time-2.5<0.005", "time-2.5>-0.005"},
        ),
    ],
    ids=["abs", "two-comparisons", "time", "formal-named-A"],
)
def test_an_sbml_event_on_a_window_in_a_function_definition_fires(
    tmp_path, trigger, body, edges, rtol
):
    """``near(A, 5, 0.01)`` holds its window in the function's body, and the
    loader scanned only the trigger's own math, which has no comparison in it.
    Rooted whole, the window was stepped over and the event never fired: Y = 0
    at both tolerances, where the same window written in the trigger fired.
    The body's two edges are registered now, in what the call binds."""
    path = _sbml_window(tmp_path, trigger=trigger, functions={"near": body})
    covered = bngsim.Model.from_sbml(path).time_discontinuity_conditions()
    assert {c.replace("(", "").replace(")", "") for c in covered} == edges
    y, fired_at = _fired(tmp_path, trigger, rtol, functions={"near": body})
    assert y == 1.0
    assert fired_at == pytest.approx(2.495, rel=1e-6)


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_an_sbml_event_on_an_equality_in_a_function_definition_fires(tmp_path, rtol):
    """``hit(time, 2.5)`` with ``hit := x == c`` holds at one double, and the run
    stops there, as it does for ``time == 2.5`` written in the trigger. The
    body was not read, and the event never fired."""
    y, fired_at = _fired(
        tmp_path, "hit(time, 2.5)", rtol, functions={"hit": "lambda(x, c, x == c)"}
    )
    assert y == 1.0
    assert fired_at == pytest.approx(2.5, rel=1e-12)


def test_an_equality_in_a_function_definition_that_never_holds_is_refused(tmp_path):
    """``hit(time - 1, 1.3)``: ``t - 1`` steps over 1.3 in doubles, and the run
    refuses it, as it does ``time - 1 == 1.3`` written in the trigger. The body
    was not read, and the run went on with an event that never fired."""
    path = _sbml_window(
        tmp_path, trigger="hit(time - 1, 1.3)", functions={"hit": "lambda(x, c, x == c)"}
    )
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").run(
            t_span=(0.0, 10.0), n_points=2
        )


def test_an_sbml_event_on_a_touch_in_a_function_definition_fires_on_any_grid(tmp_path):
    """``peak(time)`` with ``peak := sin(s) >= 1`` is true at one instant a
    period. A trigger's touch keeps its stop (issue #714), and the event fires
    16 times by t = 100 on any grid, as on ``sin(time) >= 1``. The body was not
    read, and it never fired."""
    path = _sbml_window(
        tmp_path, trigger="peak(time)", functions={"peak": "lambda(s, sin(s) >= 1)"}
    )
    for n_points in (2, 11, 1001):
        run = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").run(
            t_span=(0.0, 100.0), n_points=n_points
        )
        y = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))["Y"]
        assert y == 16.0


COUNTER = """begin parameters
    1 k 10
    2 one 1
    3 c 5
end parameters
begin functions
    1 f() if(abs(Aobs-c)<0.01,k*Aobs,0)
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 one
    2 0 2 f
end reactions
begin groups
    1 Aobs 1
end groups
"""


def test_an_abs_window_on_a_counter_whose_edges_a_column_moves_is_refused(tmp_path):
    """A is a counter: Y = k*c*0.02 = 1, dY/dc = 0.2, dY/dk = 0.1.

    abs() has no analytic derivative, so the columns are CVODES' difference
    quotient. Issue #48 holds a counter's threshold while the quotient is
    taken, but only where it reads the condition, and it does not read this
    one. With c requested the quotient moved the edges the run now stops at,
    and dY/dc came back 7.03. A column that moves neither is right.
    """
    path = tmp_path / "counter.net"
    path.write_text(COUNTER)
    with pytest.raises(SensitivityUnsupportedError, match=r"abs\(Aobs-c\)<0\.01"):
        bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["c"]
        ).run(t_span=(0.0, 10.0), n_points=2, rtol=1e-8, atol=1e-12)
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["k"]
    )
    run = sim.run(t_span=(0.0, 10.0), n_points=2, rtol=1e-8, atol=1e-12)
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(1.0, rel=1e-6)
    assert float(np.asarray(run.sensitivities)[-1, 1, 0]) == pytest.approx(0.1, rel=1e-6)


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_a_counters_window_is_stopped_on_and_not_rooted_as_well(tmp_path, rtol):
    """A counter's ``abs(Aobs-c) < 0.01`` is a condition on time, and the run
    stops at both of its edges. Its edges were rooted as well, a second
    restart at each stop. Y = k*c*0.02 = 1."""
    path = tmp_path / "counter.net"
    path.write_text(COUNTER)
    model = bngsim.Model.from_net(str(path))
    assert model.time_discontinuity_conditions() == ("abs(Aobs-c)<0.01",)
    assert model.state_switch_root_conditions() == ()
    run = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(1.0, rel=1e-4)
