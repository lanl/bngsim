"""Issue #1054 — a piecewise assignment-rule compartment size that jumps.

SBML keeps a species' amount across a change of its compartment's size, so a
concentration jumps by V_before/V_after where the size does. The §8c dilution
term carries the size's rate between jumps, but a jump is a delta no ODE term
carries, and the concentration was held there instead:

- ``cv := piecewise(1 + 0.1*S, S > 3, 1 + 0.2*S)``, S decaying from 10 at 0.5,
  jumps 1.30 -> 1.60 at t = 2.41, and [Sn] stayed high by 1.60/1.30 after;
- ``cv := piecewise(1 + t, t < 2, 5)`` jumps 3 -> 5 at t = 2, and [Sn] stayed
  at 1.667 where the amount, 5, over 5 is 1.

Conditions that change at one root, ``t < 2`` and ``t >= 2`` in a piecewise
with no otherwise, are one jump, not one each. Each case is checked against
amount/V(t) in closed form.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

M = 'xmlns="http://www.w3.org/1998/Math/MathML"'
T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'


def _pw(*pieces, otherwise=None):
    body = "".join(f"<piece>{v}{c}</piece>" for v, c in pieces)
    tail = "" if otherwise is None else f"<otherwise>{otherwise}</otherwise>"
    return f"<piecewise>{body}{tail}</piecewise>"


def _plus(a, b):
    return f"<apply><plus/>{a}{b}</apply>"


def _times(a, b):
    return f"<apply><times/>{a}{b}</apply>"


def _rel(op, a, b):
    return f"<apply><{op}/>{a}{b}</apply>"


def _cn(x):
    return f"<cn>{x}</cn>"


def _ci(x):
    return f"<ci>{x}</ci>"


def _doc(rules, *, functions="", params="", events="", species=""):
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {M}>{b}</math></assignmentRule>' for v, b in rules
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfFunctionDefinitions>{functions}</listOfFunctionDefinitions>
 <listOfCompartments><compartment id="c" size="1" constant="true"/>
  <compartment id="cv" size="1" constant="false"/></listOfCompartments>
 <listOfSpecies>
  <species id="S" compartment="c" initialAmount="10" hasOnlySubstanceUnits="false"
   boundaryCondition="false" constant="false"/>
  <species id="Sn" compartment="cv" initialConcentration="5"
   hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>{species}
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.5" constant="true"/>{params}</listOfParameters>
 <listOfRules>{rule_xml}</listOfRules>
 <listOfReactions><reaction id="degS" reversible="false">
  <listOfReactants>
   <speciesReference species="S" stoichiometry="1" constant="true"/>
  </listOfReactants>
  <kineticLaw><math {M}>{_times(_times(_ci("k"), _ci("S")), _ci("c"))}</math></kineticLaw>
 </reaction></listOfReactions>
 <listOfEvents>{events}</listOfEvents>
</model></sbml>"""


S_SWITCH = _pw(
    (_plus(_cn(1), _times(_cn(0.1), _ci("S"))), _rel("gt", _ci("S"), _cn(3))),
    otherwise=_plus(_cn(1), _times(_cn(0.2), _ci("S"))),
)
T_STEP = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), otherwise=_cn(5))
# up at t = 1 and back down at t = 3
T_WINDOW = _pw(
    (_cn(2), f"<apply><and/>{_rel('gt', T, _cn(1))}{_rel('lt', T, _cn(3))}</apply>"),
    otherwise=_cn(1),
)
# no otherwise: `t < 2` and `t >= 2` flip at one root
T_NO_OTHERWISE = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), (_cn(5), _rel("geq", T, _cn(2))))
S_NO_OTHERWISE = _pw(
    (_plus(_cn(1), _times(_cn(0.1), _ci("S"))), _rel("gt", _ci("S"), _cn(3))),
    (_plus(_cn(1), _times(_cn(0.2), _ci("S"))), _rel("leq", _ci("S"), _cn(3))),
)
# no piece at t = 2.1 itself; the jump's rescale divided by the 0 read there
T_GAP = _pw((_cn(1), _rel("lt", T, _cn(2.1))), (_cn(5), _rel("gt", T, _cn(2.1))))
# continuous at its switch: 1 + t meets 3 at t = 2
T_CONTINUOUS = _pw((_plus(_cn(1), T), _rel("lt", T, _cn(2))), otherwise=_cn(3))
STEP_FN = (
    '<functionDefinition id="step"><math '
    + M
    + "><lambda><bvar><ci>x</ci></bvar>"
    + _pw((_plus(_cn(1), _ci("x")), _rel("lt", _ci("x"), _cn(2))), otherwise=_cn(5))
    + "</lambda></math></functionDefinition>"
)


def _s(t):
    return 10 * np.exp(-0.5 * t)


CASES = {
    # the size jumps 1.30 -> 1.60 as S falls through 3; V(0) = 2, amount 10
    "on a species": (
        _doc([("cv", S_SWITCH)]),
        lambda t: 10 / np.where(_s(t) > 3, 1 + 0.1 * _s(t), 1 + 0.2 * _s(t)),
    ),
    "on time": (_doc([("cv", T_STEP)]), lambda t: 5 / np.where(t < 2, 1 + t, 5)),
    "both ways": (
        _doc([("cv", T_WINDOW)]),
        lambda t: 5 / np.where((t > 1) & (t < 3), 2, 1),
    ),
    "through another rule": (
        _doc(
            [("cv", _plus(_cn(0), _ci("q"))), ("q", T_STEP)],
            params='<parameter id="q" value="1" constant="false"/>',
        ),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    "through a function definition": (
        _doc([("cv", f"<apply><ci>step</ci>{T}</apply>")], functions=STEP_FN),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    # each pair is one jump, not two rescales at the instant
    "with no otherwise, on time": (
        _doc([("cv", T_NO_OTHERWISE)]),
        lambda t: 5 / np.where(t < 2, 1 + t, 5),
    ),
    "with no otherwise, on a species": (
        _doc([("cv", S_NO_OTHERWISE)]),
        lambda t: 10 / np.where(_s(t) > 3, 1 + 0.1 * _s(t), 1 + 0.2 * _s(t)),
    ),
    "with no piece at the switch": (
        _doc([("cv", T_GAP)]),
        lambda t: 5 / np.where(t < 2.1, 1, 5),
    ),
    # control: no jump, so nothing to rescale
    "continuous (control)": (
        _doc([("cv", T_CONTINUOUS)]),
        lambda t: 5 / np.where(t < 2, 1 + t, 3),
    ),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_amount_is_kept_across_the_jump(case, codegen):
    xml, closed_form = CASES[case]
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", codegen=codegen
    ).run(t_span=(0.0, 4.0), n_points=17)
    t = np.asarray(result.time)
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    np.testing.assert_allclose(sn, closed_form(t), rtol=1e-5, atol=1e-8)


# ─── Two jumps a run constant puts on one instant (#1124) ──────────────────
#
# ``cv := 1 + piecewise(0, t < t1, 1) + piecewise(0, t < t2, 1)``: with t1 = t2
# the size goes 1 -> 3 at once, and [Sn] to 5/3. The two jump events read the
# state at their trigger time, so the last rescale is what stays, and each read
# the other's condition as already flipped: [Sn] went to 5·2/3 = 3.33.


def _step_at(threshold, *, on="time"):
    reads = T if on == "time" else _ci("S")
    op = "lt" if on == "time" else "gt"
    return _pw((_cn(0), _rel(op, reads, _ci(threshold))), otherwise=_cn(1))


def _two_steps(a, b, *, on="time"):
    names = ("t1", "t2") if on == "time" else ("a", "b")
    size = _plus(_plus(_cn(1), _step_at(names[0], on=on)), _step_at(names[1], on=on))
    params = "".join(
        f'<parameter id="{n}" value="{v!r}" constant="true"/>'
        for n, v in zip(names, (a, b), strict=True)
    )
    return _doc([("cv", size)], params=params)


def _sn_at(model, times, *, codegen=False):
    result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
        sample_times=times, rtol=1e-10, atol=1e-12
    )
    return np.asarray(result.species)[:, list(result.species_names).index("Sn")]


def _n_steps(*thresholds):
    names = tuple(f"t{i + 1}" for i in range(len(thresholds)))
    size = _cn(1)
    for name in names:
        size = _plus(size, _step_at(name))
    params = "".join(
        f'<parameter id="{n}" value="{v!r}" constant="true"/>'
        for n, v in zip(names, thresholds, strict=True)
    )
    return _doc([("cv", size)], params=params)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize(
    "thresholds",
    [(2.0, 2.0, 2.0), (2.0, 2.0, 3.0), (3.0, 2.0, 2.0)],
    ids=["three together", "two then one", "one then two"],
)
def test_three_jumps_on_one_instant_rescale_once(thresholds, codegen):
    """#1132's review: the sibling rule was shown on pairs; three steps on one
    instant are one rescale too, on both backends."""
    times = np.array([0.0, 1.9, 2.5, 3.5])
    got = _sn_at(bngsim.Model.from_sbml_string(_n_steps(*thresholds)), times, codegen=codegen)
    size = 1 + sum((times >= t).astype(float) for t in thresholds)
    np.testing.assert_allclose(got, 5 / size, rtol=1e-8)


def test_the_reversed_spelling_of_a_sibling_is_one_jump():
    """`piecewise(1, t >= t2, 0)` is `piecewise(0, t < t2, 1)` spelled the
    other way round; beside `piecewise(0, t < t1, 1)` at t1 = t2 the two are
    one instant and one rescale, 5 -> 5/3."""
    reversed_step = _pw((_cn(1), _rel("geq", T, _ci("t2"))), otherwise=_cn(0))
    size = _plus(_plus(_cn(1), _step_at("t1")), reversed_step)
    params = "".join(f'<parameter id="{n}" value="2.0" constant="true"/>' for n in ("t1", "t2"))
    times = np.array([0.0, 1.9, 2.5, 3.5])
    got = _sn_at(bngsim.Model.from_sbml_string(_doc([("cv", size)], params=params)), times)
    np.testing.assert_allclose(got, 5 / (1 + 2 * (times >= 2.0)), rtol=1e-8)


@pytest.mark.parametrize(
    ("t1", "t2"),
    [(2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (2.0, 2.0000000001)],
    ids=["together", "apart", "apart, the other way", "a hair apart"],
)
def test_two_jumps_on_one_instant_rescale_once(t1, t2):
    times = np.array([0.0, 1.9, 2.5, 3.5])
    got = _sn_at(bngsim.Model.from_sbml_string(_two_steps(t1, t2)), times)
    np.testing.assert_allclose(got, 5 / (1 + (times >= t1) + (times >= t2)), rtol=1e-8)


@pytest.mark.parametrize(("t2", "now"), [(3.0, 2.0), (2.0, 3.0)], ids=["onto", "off"])
def test_a_set_param_moves_one_jump_onto_the_other_or_off_it(t2, now):
    model = bngsim.Model.from_sbml_string(_two_steps(2.0, t2))
    model.set_param("t2", now)
    times = np.array([0.0, 2.5, 3.5])
    np.testing.assert_allclose(
        _sn_at(model, times), 5 / (1 + (times >= 2.0) + (times >= now)), rtol=1e-8
    )


@pytest.mark.parametrize(("a", "b"), [(5.0, 5.0), (5.0, 4.0)], ids=["together", "apart"])
def test_two_jumps_on_one_species_threshold(a, b):
    """S = 10·e^(−t/2) passes 5 at t = 1.386 and 4 at t = 1.833."""
    times = np.array([0.0, 1.0, 1.5, 2.0])
    s = _s(times)
    got = _sn_at(bngsim.Model.from_sbml_string(_two_steps(a, b, on="species")), times)
    np.testing.assert_allclose(got, 5 / (1 + (s <= a) + (s <= b)), rtol=1e-8)


@pytest.mark.parametrize(
    ("on", "scaled"),
    [("time", 4.0), ("time", 5.0), ("species", 10.0), ("species", 8.0)],
    ids=["time, together", "time, apart", "species, together", "species, apart"],
)
def test_two_jumps_a_scale_puts_on_one_instant(on, scaled):
    """`2*t < t2` beside `t < t1` with t2 = 2·t1, or `2*S > b` beside `S > a`
    with b = 2a: the roots coincide, but the differences are not ± each other,
    and [Sn] went to 5·2/3 for 5/3."""
    reads = T if on == "time" else _ci("S")
    first, op, second = (("t1", 2.0), "lt", "t2") if on == "time" else (("a", 5.0), "gt", "b")
    size = _plus(
        _plus(_cn(1), _step_at(first[0], on=on)),
        _pw((_cn(0), _rel(op, _times(_cn(2), reads), _ci(second))), otherwise=_cn(1)),
    )
    params = "".join(
        f'<parameter id="{n}" value="{v!r}" constant="true"/>'
        for n, v in (first, (second, scaled))
    )
    times = np.array([0.0, 1.0, 1.5, 1.9, 2.5, 3.5])
    got = _sn_at(bngsim.Model.from_sbml_string(_doc([("cv", size)], params=params)), times)
    if on == "time":
        size_then = 1 + (times >= first[1]) + (2 * times >= scaled)
    else:
        s = _s(times)
        size_then = 1 + (s <= first[1]) + (2 * s <= scaled)
    np.testing.assert_allclose(got, 5 / size_then, rtol=1e-8)


def _set_sn_at_2(priority):
    return (
        '<event id="e" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math {M}>'
        f"{_rel('geq', T, _cn(2))}</math></trigger>"
        + (f"<priority><math {M}>{_cn(priority)}</math></priority>" if priority else "")
        + f'<listOfEventAssignments><eventAssignment variable="Sn"><math {M}>{_cn(10)}'
        "</math></eventAssignment></listOfEventAssignments></event>"
    )


@pytest.mark.parametrize(
    "priority", [None, 5, "1e308"], ids=["no priority", "a priority", "the jump's priority"]
)
def test_an_event_at_the_jump_keeps_the_concentration_it_sets(priority):
    """``at (t >= 2): Sn = 10`` fires where ``cv := piecewise(1 + t, t < 2,
    5)`` jumps. Sn is 10 after, the concentration the event sets in the
    compartment as it is then; the jump's rescale of Sn from before the instant
    overwrote it whenever it ran second, by the priority or the draw between
    equal ones. At the jump's own priority, 1e308, the draw was a fair one, and
    the model did not load: 1e308 is an integer too large for the int field a
    constant priority was given."""
    xml = _doc([("cv", T_STEP)], events=_set_sn_at_2(priority))
    for seed in range(4):
        result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
            t_span=(0.0, 4.0), n_points=17, seed=seed
        )
        t = np.asarray(result.time)
        sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
        np.testing.assert_allclose(sn, np.where(t < 2, 5 / (1 + t), 10), rtol=1e-5, atol=1e-8)


# ─── What an event at the jump reads (issue #1054's review) ─────────────────
#
# `cv := piecewise(1 + t, t < 2, 5)` takes [Sn] from 5/3 just before t = 2 to 1
# at t = 2. The jump's rescale executed first, but an event triggered at the
# instant had its trigger, values and delay read with the rest of the batch,
# before it, at [Sn] = 5/3: `Sn = Sn + 1` left 8/3 where [Sn] at t = 2 is 1, and
# libRoadRunner gives 2.


def _event(eid, trigger, assignments, *, uvftt=True, persistent=True, delay=None, priority=None):
    body = "".join(
        f'<eventAssignment variable="{v}"><math {M}>{e}</math></eventAssignment>'
        for v, e in assignments
    )
    return (
        f'<event id="{eid}" useValuesFromTriggerTime="{str(uvftt).lower()}">'
        f'<trigger initialValue="false" persistent="{str(persistent).lower()}">'
        f"<math {M}>{trigger}</math></trigger>"
        + (f"<delay><math {M}>{delay}</math></delay>" if delay is not None else "")
        + (f"<priority><math {M}>{priority}</math></priority>" if priority is not None else "")
        + f"<listOfEventAssignments>{body}</listOfEventAssignments></event>"
    )


def _sp(sid):
    return (
        f'<species id="{sid}" compartment="c" initialConcentration="0" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


AT_2 = _rel("geq", T, _cn(2))
SN_PLUS_1 = _plus(_ci("Sn"), _cn(1))


def _sn(after):
    return lambda t: np.where(t < 2, 5 / (1 + t), after)


def _from(t0, value):
    return lambda t: np.where(t < t0, 0.0, value)


AT_THE_JUMP = {
    "a value frozen at the trigger": ([_event("e", AT_2, [("Sn", SN_PLUS_1)])], {"Sn": _sn(2.0)}),
    # control: read when it executes, after the jump, it was right
    "a value read when it executes": (
        [_event("e", AT_2, [("Sn", SN_PLUS_1)], uvftt=False)],
        {"Sn": _sn(2.0)},
    ),
    "another species set from it": ([_event("e", AT_2, [("X", _ci("Sn"))])], {"X": _from(2, 1.0)}),
    "a priority": (
        [_event("e", AT_2, [("X", _ci("Sn"))], priority=_cn(5))],
        {"X": _from(2, 1.0)},
    ),
    "after a delay": (
        [_event("e", AT_2, [("X", _ci("Sn"))], delay=_cn(0.9))],
        {"X": _from(2.9, 1.0)},
    ),
    # 0.6·[Sn] is 0.6 at t = 2, and was read 1
    "a delay that reads it": (
        [_event("e", AT_2, [("X", _cn(1))], delay=_times(_cn(0.6), _ci("Sn")))],
        {"X": _from(2.6, 1.0)},
    ),
    # [Sn] > 1.5 is false at t = 2, and was read true
    "a trigger that reads it": (
        [
            _event(
                "e",
                f"<apply><and/>{AT_2}{_rel('gt', _ci('Sn'), _cn(1.5))}</apply>",
                [("X", _cn(7))],
            )
        ],
        {"X": _from(0, 0.0)},
    ),
    # control: the jump cancelled it, as its trigger fell
    "a trigger that reads it, not persistent": (
        [
            _event(
                "e",
                f"<apply><and/>{AT_2}{_rel('gt', _ci('Sn'), _cn(1.5))}</apply>",
                [("X", _cn(7))],
                persistent=False,
            )
        ],
        {"X": _from(0, 0.0)},
    ),
    # Y is frozen at the instant too: before e sets X, after the jump
    "two events in turn": (
        [
            _event("e", AT_2, [("X", _ci("Sn"))], priority=_cn(2)),
            _event("f", AT_2, [("Y", _plus(_ci("X"), _ci("Sn")))], priority=_cn(1)),
        ],
        {"X": _from(2, 1.0), "Y": _from(2, 1.0)},
    ),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(AT_THE_JUMP))
def test_an_event_at_the_jump_reads_the_concentration_after_it(case, codegen):
    events, closed_forms = AT_THE_JUMP[case]
    xml = _doc([("cv", T_STEP)], events="".join(events), species=_sp("X") + _sp("Y"))
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", codegen=codegen
    ).run(t_span=(0.0, 4.0), n_points=17)
    t = np.asarray(result.time)
    names = list(result.species_names)
    for sid, closed_form in closed_forms.items():
        got = np.asarray(result.species)[:, names.index(sid)]
        np.testing.assert_allclose(got, closed_form(t), rtol=1e-5, atol=1e-8, err_msg=sid)


def _sensitivities(events):
    xml = _doc([("cv", T_STEP)], events="".join(events), species=_sp("X") + _sp("Y"))
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", sensitivity_params=["k"]
    )
    return sim.run(t_span=(0.0, 4.0), n_points=17)


def test_the_sensitivity_of_a_value_frozen_after_the_jump():
    """``X = k*Sn`` at the jump: dX/dk is [Sn] at t = 2, 1. Read before the
    jump it was 5/3, beside an X of 5/6."""
    result = _sensitivities([_event("e", AT_2, [("X", _times(_ci("k"), _ci("Sn")))])])
    t = np.asarray(result.time)
    x = list(result.species_names).index("X")
    np.testing.assert_allclose(
        np.asarray(result.species)[:, x], _from(2, 0.5)(t), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[:, x, 0], _from(2, 1.0)(t), rtol=1e-5, atol=1e-8
    )


def test_the_sensitivity_of_a_value_frozen_between_two_fires_is_refused():
    """f's values are frozen after the jump and before e sets X, which they
    read: that state is neither the one before the batch nor the one at f, the
    two the sensitivity jump reads at."""
    events = AT_THE_JUMP["two events in turn"][0]
    with pytest.raises(bngsim.SimulationError, match=r"frozen between two fires of one instant"):
        _sensitivities(events)


def test_run_diffrax_names_the_jump():
    """The model has no event of its own; the two are the jump's, each way."""
    pytest.importorskip("diffrax")
    from bngsim._diffrax_solver import run_diffrax

    model = bngsim.Model.from_sbml_string(_doc([("cv", S_SWITCH)]))
    assert model.n_events == 2
    with pytest.raises(ValueError, match=r"keeps the amounts in compartment cv across a jump"):
        run_diffrax(model, t_end=4.0)


def test_a_condition_an_event_also_moves_is_refused():
    """``t > p`` flips with the clock and when the event sets p: the event's
    own rescale (#74) and the jump's would both apply, so it is refused."""
    size = _pw((_cn(2), _rel("gt", T, _ci("p"))), otherwise=_cn(1))
    event = (
        '<event id="e" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math {M}>'
        f"{_rel('gt', T, _cn(0.5))}</math></trigger>"
        f'<listOfEventAssignments><eventAssignment variable="p"><math {M}>{_cn(0.25)}'
        "</math></eventAssignment></listOfEventAssignments></event>"
    )
    xml = _doc(
        [("cv", size)],
        params='<parameter id="p" value="3" constant="false"/>',
        events=event,
    )
    with pytest.raises(bngsim.ModelError, match=r"p, which an event changes"):
        bngsim.Model.from_sbml_string(xml)


# ─── A jump another compartment's size reads ────────────────────────────────
#
# `cv := piecewise(1, t < 2, 5)` takes [Sn] from 5 to 1 at t = 2, and a size
# rule that reads [Sn] jumps with it, whether or not a condition of its own
# flips there. Where one did, `cw := piecewise(1, t < 2, 1 + Sn)`, cw's own
# jump read [Sn] from before cv's and went to 1 + 5 where SBML has 1 + 1 at the
# instant, and [Sw] went from 3 to 3/6 for 3/2 (libRoadRunner 1.5). Where none
# did, `cw := 1 + Sn`, nothing rescaled [Sw] at all.


def _sized_doc(compartments, species, rules, *, params="", reactions="", functions="", amounts=()):
    comp_xml = "".join(
        f'<compartment id="{c}" size="1" constant="{str(c == "c").lower()}"/>'
        for c in compartments
    )
    sp_xml = "".join(
        f'<species id="{s}" compartment="{c}" '
        + (
            f'initialAmount="{x}" hasOnlySubstanceUnits="true" '
            if s in amounts
            else f'initialConcentration="{x}" hasOnlySubstanceUnits="false" '
        )
        + 'boundaryCondition="false" constant="false"/>'
        for s, c, x in species
    )
    rule_xml = "".join(
        f'<assignmentRule variable="{v}"><math {M}>{b}</math></assignmentRule>' for v, b in rules
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
 <listOfFunctionDefinitions>{functions}</listOfFunctionDefinitions>
 <listOfCompartments>{comp_xml}</listOfCompartments>
 <listOfSpecies>{sp_xml}</listOfSpecies>
 <listOfParameters><parameter id="k" value="0.5" constant="true"/>{params}</listOfParameters>
 <listOfRules>{rule_xml}</listOfRules>
 <listOfReactions>{reactions}</listOfReactions>
</model></sbml>"""


def _step(before, cond, after):
    return _pw((before, cond), otherwise=after)


BEFORE_2 = _rel("lt", T, _cn(2))
CV_JUMP = ("cv", _step(_cn(1), BEFORE_2, _cn(5)))  # [Sn] = 5 goes to 1 at t = 2
ONE_PLUS_SN = _plus(_cn(1), _ci("Sn"))
S_DECAYS = (
    '<reaction id="degS" reversible="false"><listOfReactants>'
    '<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>'
    f"<kineticLaw><math {M}>{_times(_times(_ci('k'), _ci('S')), _ci('c'))}</math></kineticLaw>"
    "</reaction>"
)
T_S = 2 * np.log(10 / 3)  # where S, 10 at 0 and decaying at 0.5, falls through 3


def _jump_to(at, before, after):
    return lambda t: np.where(t < at, before, after)


FOLLOWERS = {
    "cw jumps with cv": (
        _sized_doc(
            ["cv", "cw"],
            [("Sn", "cv", 5), ("Sw", "cw", 3)],
            [CV_JUMP, ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN))],
        ),
        {"Sn": _jump_to(2, 5, 1), "Sw": _jump_to(2, 3, 1.5)},
    ),
    "in reverse document order": (
        _sized_doc(
            ["cw", "cv"],
            [("Sw", "cw", 3), ("Sn", "cv", 5)],
            [("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)), CV_JUMP],
        ),
        {"Sn": _jump_to(2, 5, 1), "Sw": _jump_to(2, 3, 1.5)},
    ),
    # [Su] 8 -> 2, cv 1 -> 3 and [Sn] 6 -> 2, cw 1 -> 3 and [Sw] 3 -> 1
    "three in a chain": (
        _sized_doc(
            ["cu", "cv", "cw"],
            [("Su", "cu", 8), ("Sn", "cv", 6), ("Sw", "cw", 3)],
            [
                ("cu", _step(_cn(1), BEFORE_2, _cn(4))),
                ("cv", _step(_cn(1), BEFORE_2, _plus(_cn(1), _ci("Su")))),
                ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
            ],
        ),
        {"Su": _jump_to(2, 8, 2), "Sn": _jump_to(2, 6, 2), "Sw": _jump_to(2, 3, 1)},
    ),
    "on a species": (
        _sized_doc(
            ["c", "cv", "cw"],
            [("S", "c", 10), ("Sn", "cv", 5), ("Sw", "cw", 3)],
            [
                ("cv", _step(_cn(1), _rel("gt", _ci("S"), _cn(3)), _cn(5))),
                ("cw", _step(_cn(1), _rel("gt", _ci("S"), _cn(3)), ONE_PLUS_SN)),
            ],
            reactions=S_DECAYS,
        ),
        {"Sn": _jump_to(T_S, 5, 1), "Sw": _jump_to(T_S, 3, 1.5)},
    ),
    # 1 + 5 before and 2 + 3 after: the amount, 18, over 5
    "read on both sides": (
        _sized_doc(
            ["cv", "cw"],
            [("Sn", "cv", 5), ("Sw", "cw", 3)],
            [
                CV_JUMP,
                ("cw", _step(ONE_PLUS_SN, BEFORE_2, _plus(_cn(2), _times(_cn(3), _ci("Sn"))))),
            ],
        ),
        {"Sw": _jump_to(2, 3, 3.6)},
    ),
    # cw goes from 6 to 2, and nothing of cw's own flips: 18 over 2
    "with no condition of its own": (
        _sized_doc(
            ["cv", "cw"], [("Sn", "cv", 5), ("Sw", "cw", 3)], [CV_JUMP, ("cw", ONE_PLUS_SN)]
        ),
        {"Sn": _jump_to(2, 5, 1), "Sw": _jump_to(2, 3, 9)},
    ),
    # [Sn] > 3 is true before and false after: cw goes from 1 to 2
    "in a condition": (
        _sized_doc(
            ["cv", "cw"],
            [("Sn", "cv", 5), ("Sw", "cw", 3)],
            [
                CV_JUMP,
                (
                    "cw",
                    _step(
                        _cn(1),
                        f"<apply><or/>{BEFORE_2}{_rel('gt', _ci('Sn'), _cn(3))}</apply>",
                        _cn(2),
                    ),
                ),
            ],
        ),
        {"Sw": _jump_to(2, 3, 1.5)},
    ),
    # two clocks, not one root, that agree in this run
    "at two thresholds that are equal": (
        _sized_doc(
            ["cv", "cw"],
            [("Sn", "cv", 5), ("Sw", "cw", 3)],
            [
                ("cv", _step(_cn(1), _rel("lt", T, _ci("t1")), _cn(5))),
                ("cw", _step(_cn(1), _rel("lt", T, _ci("t2")), ONE_PLUS_SN)),
            ],
            params='<parameter id="t1" value="2" constant="true"/>'
            '<parameter id="t2" value="2" constant="true"/>',
        ),
        {"Sw": _jump_to(2, 3, 1.5)},
    ),
    # cw goes to 1 + 5 at t = 2, and with [Sn] to 1 + 1 at t = 3
    "at two thresholds that differ": (
        _sized_doc(
            ["cv", "cw"],
            [("Sn", "cv", 5), ("Sw", "cw", 3)],
            [
                ("cv", _step(_cn(1), _rel("lt", T, _ci("t1")), _cn(5))),
                ("cw", _step(_cn(1), _rel("lt", T, _ci("t2")), ONE_PLUS_SN)),
            ],
            params='<parameter id="t1" value="3" constant="true"/>'
            '<parameter id="t2" value="2" constant="true"/>',
        ),
        {"Sw": lambda t: np.where(t < 2, 3, np.where(t < 3, 0.5, 1.5))},
    ),
    # control: cw jumps just after t = 2, where 4 - 2t < 0 first holds, and
    # cv's jump at t = 2 has executed by then
    "on a strict inequality": (
        _sized_doc(
            ["cv", "cw"],
            [("Sn", "cv", 5), ("Sw", "cw", 3)],
            [
                CV_JUMP,
                (
                    "cw",
                    _step(
                        ONE_PLUS_SN,
                        _rel("lt", f"<apply><minus/>{_cn(4)}{_times(_cn(2), T)}</apply>", _cn(0)),
                        _cn(1),
                    ),
                ),
            ],
        ),
        {"Sw": lambda t: np.where(t > 2, 1.5, 3)},
    ),
}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize("case", list(FOLLOWERS))
def test_a_size_that_reads_a_concentration_a_jump_moves(case, codegen):
    xml, closed_forms = FOLLOWERS[case]
    for seed in range(4):  # the order the jumps of an instant execute in is drawn
        result = bngsim.Simulator(
            bngsim.Model.from_sbml_string(xml), method="ode", codegen=codegen
        ).run(t_span=(0.0, 4.0), n_points=17, seed=seed)
        t = np.asarray(result.time)
        names = list(result.species_names)
        for sid, closed_form in closed_forms.items():
            got = np.asarray(result.species)[:, names.index(sid)]
            np.testing.assert_allclose(
                got, closed_form(t), rtol=1e-5, atol=1e-8, err_msg=f"{sid}, seed {seed}"
            )


def test_the_sensitivity_of_a_concentration_a_jump_moves():
    """Sn decays at k in cv, and cw := piecewise(1, t < 2, 1 + Sn) from t = 2
    on, the amount of Sw, 3, held: [Sw] = 3/(1 + e^(-kt)) there, and d[Sw]/dk =
    3t·e^(-kt)/(1 + e^(-kt))²."""
    decay = (
        '<reaction id="degSn" reversible="false"><listOfReactants>'
        '<speciesReference species="Sn" stoichiometry="1" constant="true"/></listOfReactants>'
        f"<kineticLaw><math {M}>{_times(_times(_ci('k'), _ci('Sn')), _ci('cv'))}</math>"
        "</kineticLaw></reaction>"
    )
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [CV_JUMP, ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN))],
        reactions=decay,
    ).replace('id="k" value="0.5"', 'id="k" value="0.1"')
    result = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", sensitivity_params=["k"]
    ).run(t_span=(0.0, 4.0), n_points=17)
    t = np.asarray(result.time)
    w = list(result.species_names).index("Sw")
    e = np.exp(-0.1 * t)
    np.testing.assert_allclose(
        np.asarray(result.species)[:, w], np.where(t < 2, 3, 3 / (1 + e)), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[:, w, 0],
        np.where(t < 2, 0, 3 * t * e / (1 + e) ** 2),
        rtol=1e-4,
        atol=1e-7,
    )


def test_a_ring_of_new_sizes_is_refused():
    """cv's new size reads [Sw] and cw's reads [Sn]: each is the other's
    amount over a size that reads it, and libRoadRunner refuses the pair as a
    recursive assignment rule."""
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            ("cv", _step(_cn(1), BEFORE_2, _plus(_cn(1), _ci("Sw")))),
            ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
        ],
    )
    with pytest.raises(bngsim.ModelError, match=r"cv reads Sw in cw; cw reads Sn in cv"):
        bngsim.Model.from_sbml_string(xml)


def _ring_through_a_condition():
    # cv reads [Sw] only in a condition that holds false on both sides, and cw
    # reads [Sn] in its new size: each jump rescales the other's species.
    return _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            (
                "cv",
                _step(
                    _cn(1),
                    f"<apply><or/>{BEFORE_2}{_rel('gt', _ci('Sw'), _cn(100))}</apply>",
                    _ci("b"),
                ),
            ),
            ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
        ],
        params='<parameter id="b" value="5" constant="true"/>',
    )


def test_a_ring_through_a_value_and_a_condition_is_refused():
    """Issue #1185, item 2. cw's size reads [Sn] in cv, and cv's jump condition
    reads [Sw] in cw: a ring through a value and a condition. This one has a
    single consistent state after t = 2 (cv's condition holds false on both
    sides), and bngsim ran it. It is refused with the rest: telling a ring with
    one consistent state from one with two is a solve bngsim does not make, and
    libRoadRunner refuses both as a recursive assignment rule."""
    with pytest.raises(
        bngsim.ModelError,
        match=r"through their values and their jump conditions \(cv's jump condition reads "
        r"Sw in cw; cw's size reads Sn in cv\)",
    ):
        bngsim.Model.from_sbml_string(_ring_through_a_condition())


def _value_and_condition_ring():
    # cv := 1 + [Sw]; cw := piecewise(5, time >= 2 && [Sn] > 8, 1)
    cond = f"<apply><and/>{_rel('geq', T, _cn(2))}{_rel('gt', _ci('Sn'), _cn(8))}</apply>"
    return _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [("cv", _plus(_cn(1), _ci("Sw"))), ("cw", _step(_cn(5), cond, _cn(1)))],
    )


def test_a_ring_with_two_consistent_states_is_refused():
    """Issue #1185, item 2, its reproducer. After t = 2 cw at 1 ([Sn] 5, so
    [Sn] > 8 is false) and cw at 5 ([Sw] 0.6, cv 1.6, [Sn] 12.5, so it is true)
    are both consistent. bngsim kept cw at 1, with no word of the other."""
    with pytest.raises(
        bngsim.ModelError,
        match=r"cv's size reads Sw in cw; cw's jump condition reads Sn in cv\), so where a "
        r"jump flips one of those conditions, the sizes after it can be read more than one way",
    ):
        bngsim.Model.from_sbml_string(_value_and_condition_ring())


def test_a_ring_of_three_through_values_and_conditions_is_refused():
    """One condition link and two value links: cu's condition reads [Sw] in cw,
    cw's size reads [Sn] in cv, and cv's size reads [Su] in cu. The message
    names each link in the ring's order, with its kind."""
    cond_on_sw = f"<apply><or/>{BEFORE_2}{_rel('gt', _ci('Sw'), _cn(100))}</apply>"
    xml = _sized_doc(
        ["cu", "cv", "cw"],
        [("Su", "cu", 8), ("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            ("cu", _step(_cn(1), cond_on_sw, _cn(5))),
            ("cv", _plus(_cn(1), _ci("Su"))),
            ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
        ],
    )
    with pytest.raises(
        bngsim.ModelError,
        match=r"cu's jump condition reads Sw in cw; cw's size reads Sn in cv; "
        r"cv's size reads Su in cu",
    ):
        bngsim.Model.from_sbml_string(xml)


def test_a_chain_through_a_value_and_a_condition_runs():
    """Negative control: a condition link and a value link that do not close.
    cu's condition reads [Sw] in cw, and cw's size reads [Sn] in cv, but cv's
    size reads nothing. At t = 2 cv goes to 5 ([Sn] 5 -> 1), cw to 1 + 1
    ([Sw] 3 -> 1.5), and cu, its condition false for [Sw] 3 or 1.5, to 4
    ([Su] 8 -> 2), whatever order the jumps of the instant are drawn in."""
    cond_on_sw = f"<apply><or/>{BEFORE_2}{_rel('gt', _ci('Sw'), _cn(100))}</apply>"
    xml = _sized_doc(
        ["cu", "cv", "cw"],
        [("Su", "cu", 8), ("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            ("cu", _step(_cn(1), cond_on_sw, _cn(4))),
            CV_JUMP,
            ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
        ],
    )
    closed_forms = {"Su": _jump_to(2, 8, 2), "Sn": _jump_to(2, 5, 1), "Sw": _jump_to(2, 3, 1.5)}
    for seed in range(4):
        result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
            t_span=(0.0, 4.0), n_points=17, seed=seed
        )
        t = np.asarray(result.time)
        names = list(result.species_names)
        for sid, closed_form in closed_forms.items():
            got = np.asarray(result.species)[:, names.index(sid)]
            np.testing.assert_allclose(
                got, closed_form(t), rtol=1e-5, atol=1e-8, err_msg=f"{sid}, seed {seed}"
            )


def test_a_ring_of_one_with_one_consistent_state_jumps():
    """Issue #1185, item 3, recorded rather than changed. A ring of one,
    a size whose jump condition reads its own concentration, is not refused.
    With two consistent states the run keeps the one it is in
    (test_a_ring_of_one_runs_and_stays). With the threshold at 100 instead of 2,
    only cv = 5 is consistent after t = 2, and the run goes there: refusing the
    ring of one would refuse this model too."""
    cond = f"<apply><or/>{BEFORE_2}{_rel('gt', _ci('Sn'), _cn(100))}</apply>"
    xml = _sized_doc(["cv"], [("Sn", "cv", 5)], [("cv", _step(_cn(1), cond, _cn(5)))])
    result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        t_span=(0.0, 4.0), n_points=5
    )
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    np.testing.assert_allclose(sn, [5.0, 5.0, 1.0, 1.0, 1.0], rtol=1e-6)


def _on_or_after_2(read, above):
    # piecewise(1, t < 2 || [read] > above, 5): 1 until t = 2, and on after it
    # while [read] stays above *above*
    return _step(
        _cn(1), f"<apply><or/>{BEFORE_2}{_rel('gt', _ci(read), _cn(above))}</apply>", _cn(5)
    )


def test_a_ring_through_conditions_alone_is_refused():
    """Issue #1124, item 3. cv jumps at t = 2 unless [Sw] > 2, and cw unless
    [Sn] > 2. Both staying at 1 ([Sn] 5, [Sw] 3) and both going to 5 ([Sn] 1,
    [Sw] 0.6) are consistent after t = 2. bngsim kept both at 1, with no word
    of it; libRoadRunner refuses the pair as a recursive assignment rule."""
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [("cv", _on_or_after_2("Sw", 2)), ("cw", _on_or_after_2("Sn", 2))],
    )
    with pytest.raises(
        bngsim.ModelError,
        match=r"cv reads Sw in cw; cw reads Sn in cv\), so where a jump flips those "
        r"conditions the sizes after it can be read more than one way",
    ):
        bngsim.Model.from_sbml_string(xml)


def test_a_ring_of_three_through_conditions_is_refused():
    xml = _sized_doc(
        ["cu", "cv", "cw"],
        [("Su", "cu", 8), ("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            ("cu", _on_or_after_2("Sw", 2)),
            ("cv", _on_or_after_2("Su", 2)),
            ("cw", _on_or_after_2("Sn", 2)),
        ],
    )
    with pytest.raises(
        bngsim.ModelError, match=r"cu reads Sw in cw; cw reads Sn in cv; cv reads Su in cu"
    ):
        bngsim.Model.from_sbml_string(xml)


def test_a_ring_through_a_rule_is_refused():
    """#1134's review: cv's new size reads [Sw] through the parameter rule
    `q := 1 + Sw`, not by name. The refusal sees through the rule."""
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            ("cv", _step(_cn(1), BEFORE_2, _ci("q"))),
            ("q", _plus(_cn(1), _ci("Sw"))),
            ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
        ],
        params='<parameter id="q" value="4" constant="false"/>',
    )
    with pytest.raises(bngsim.ModelError, match=r"cv reads Sw in cw; cw reads Sn in cv"):
        bngsim.Model.from_sbml_string(xml)


def test_a_ring_through_a_function_definition_is_refused():
    """The same ring with cv's read inside `plus1(Sw)`, a function definition."""
    plus1 = (
        '<functionDefinition id="plus1"><math '
        + M
        + "><lambda><bvar><ci>x</ci></bvar>"
        + _plus(_cn(1), _ci("x"))
        + "</lambda></math></functionDefinition>"
    )
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            ("cv", _step(_cn(1), BEFORE_2, f"<apply><ci>plus1</ci>{_ci('Sw')}</apply>")),
            ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
        ],
        functions=plus1,
    )
    with pytest.raises(bngsim.ModelError, match=r"cv reads Sw in cw; cw reads Sn in cv"):
        bngsim.Model.from_sbml_string(xml)


def test_an_amount_link_is_no_ring():
    """Control. Sw is an amount (hasOnlySubstanceUnits), so cw's size does not
    dilute it and cv's read of it closes no ring: the model loads and runs."""
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [
            ("cv", _step(_cn(1), BEFORE_2, _plus(_cn(1), _ci("Sw")))),
            ("cw", _step(_cn(1), BEFORE_2, ONE_PLUS_SN)),
        ],
        amounts={"Sw"},
    )
    result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        t_span=(0.0, 4.0), n_points=5
    )
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    assert sn[0] == 5.0 and sn[-1] == pytest.approx(5 / 4, rel=1e-6)


def test_a_ring_of_one_runs_and_stays():
    """The ring of one: `cv := piecewise(1, t < 2 || [Sn] > 2, 5)` alone has the
    same two consistent states after t = 2 (stay at 1 with [Sn] = 5, or jump
    to 5 with [Sn] = 1) as the pair #1134 refuses, and is not refused: the
    run keeps the state it is in. Pinned here so a change in that choice is
    seen (#1134's review; libRoadRunner refuses it as recursive)."""
    xml = _sized_doc(["cv"], [("Sn", "cv", 5)], [("cv", _on_or_after_2("Sn", 2))])
    result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        t_span=(0.0, 4.0), n_points=5
    )
    sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
    np.testing.assert_allclose(sn, 5.0, rtol=1e-8)


def test_a_condition_read_one_way_is_no_ring():
    """Control: cw's condition reads [Sn], and cv's reads nothing of cw. cv
    goes to 5 at t = 2, [Sn] to 1, and cw follows: [Sw] 3 -> 0.6."""
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [("cv", _step(_cn(1), BEFORE_2, _cn(5))), ("cw", _on_or_after_2("Sn", 2))],
    )
    for seed in range(4):
        result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
            t_span=(0.0, 4.0), n_points=17, seed=seed
        )
        t = np.asarray(result.time)
        sw = np.asarray(result.species)[:, list(result.species_names).index("Sw")]
        np.testing.assert_allclose(sw, _jump_to(2, 3, 0.6)(t), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "xml, param, match",
    [
        # A ring through a condition reached "two rescales of one concentration
        # at one instant" too; it is refused at load now (issue #1185), above.
        (
            FOLLOWERS["three in a chain"][0],
            "k",
            r"derivative of that composition is not implemented",
        ),
    ],
    ids=["three in a chain"],
)
def test_the_sensitivity_of_a_rescale_on_a_rescale_is_refused(xml, param, match):
    """A concentration rescaled at one instant by a jump's rescale of what
    another rescale there moved: the sensitivity jump has no derivative for the
    second. (Its sibling, one concentration rescaled by two jumps, needs a ring
    of jumps, and every ring is refused at load now, issue #1185.)"""
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", sensitivity_params=[param]
    )
    with pytest.raises(bngsim.SimulationError, match=match):
        sim.run(t_span=(0.0, 4.0), n_points=17)


# ─── Jumps the integrator puts on one instant (#1181) ──────────────────────
#
# The roots above coincide by a run constant, and the loader reads each on the
# side it leaves. CVODE batches roots within its own tolerance, and a stop
# merges the times within its instant, whatever relates them: t2 a few ulp past
# t1, `time/s < tm` beside `time < t1`, an equality beside a step. Each jump
# froze its own share of the instant and the last write stayed, [Sn] 5·2/3 for
# 5/3. The first jump of the instant now rescales by the whole of it and the
# rest stand down.


def _ulps_past(x, n=1):
    for _ in range(n):
        x = math.nextafter(x, math.inf)
    return x


def _steps(*thresholds, extra=""):
    size = _cn(1)
    for name, _ in thresholds:
        size = _plus(size, _pw((_cn(0), _rel("lt", T, _ci(name))), otherwise=_cn(1)))
    params = "".join(f'<parameter id="{n}" value="{v!r}" constant="true"/>' for n, v in thresholds)
    return size, params + extra


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
@pytest.mark.parametrize(
    ("t1", "t2"),
    [
        (2.0, _ulps_past(2.0)),
        (2.0, 2 + 1e-15),
        (2.0, 2 + 1e-14),
        (2.0, 2 + 1e-13),
        (0.3, 0.1 + 0.2),
    ],
    ids=["one ulp", "1e-15", "1e-14", "1e-13", "0.1 + 0.2"],
)
def test_two_jumps_a_few_ulp_apart_rescale_once(t1, t2, codegen):
    size, params = _steps(("t1", t1), ("t2", t2))
    times = np.array([0.0, 0.2, 1.9, 2.5, 3.5])
    for seed in range(3):  # the order the jumps of an instant execute in is drawn
        result = bngsim.Simulator(
            bngsim.Model.from_sbml_string(_doc([("cv", size)], params=params)),
            method="ode",
            codegen=codegen,
        ).run(sample_times=times, rtol=1e-10, atol=1e-12, seed=seed)
        sn = np.asarray(result.species)[:, list(result.species_names).index("Sn")]
        np.testing.assert_allclose(sn, 5 / (1 + (times >= t1) + (times >= t2)), rtol=1e-8)


def test_three_jumps_a_few_ulp_apart_rescale_once():
    """Steps 0, 1 and 2 ulp past t = 2: one instant to the run, which no run
    constant relates (three on one root are #1132's, above)."""
    size, params = _steps(("t1", 2.0), ("t2", _ulps_past(2.0)), ("t3", _ulps_past(2.0, 2)))
    times = np.array([0.0, 1.9, 2.5])
    got = _sn_at(bngsim.Model.from_sbml_string(_doc([("cv", size)], params=params)), times)
    np.testing.assert_allclose(got, [5, 5, 5 / 4], rtol=1e-8)


def test_a_step_up_and_a_step_down_on_one_instant_leave_the_size():
    """`1 + piecewise(0, t < t1, 1) + piecewise(1, t < t2, 0)` with t2 one ulp
    past t1 is 2 throughout, and [Sn] stays 5; it went to 7.5."""
    up = _pw((_cn(0), _rel("lt", T, _ci("t1"))), otherwise=_cn(1))
    down = _pw((_cn(1), _rel("lt", T, _ci("t2"))), otherwise=_cn(0))
    params = (
        '<parameter id="t1" value="2.0" constant="true"/>'
        f'<parameter id="t2" value="{_ulps_past(2.0)!r}" constant="true"/>'
    )
    times = np.array([0.0, 1.9, 2.5, 3.5])
    xml = _doc([("cv", _plus(_plus(_cn(1), up), down))], params=params)
    np.testing.assert_allclose(_sn_at(bngsim.Model.from_sbml_string(xml), times), 5, rtol=1e-8)


def test_a_scaled_clock_beside_a_plain_one_rescales_once():
    """`time/s < tm` with s = 60 and tm = 2/60 crosses where 60·tm rounds to,
    a few ulp from `time < t1`; no number relates the two at load."""
    scaled = _pw(
        (_cn(0), _rel("lt", f"<apply><divide/>{T}{_ci('s')}</apply>", _ci("tm"))), otherwise=_cn(1)
    )
    size, params = _steps(
        ("t1", 2.0),
        extra=f'<parameter id="s" value="60" constant="true"/>'
        f'<parameter id="tm" value="{2 / 60!r}" constant="true"/>',
    )
    times = np.array([0.0, 1.9, 2.5])
    got = _sn_at(
        bngsim.Model.from_sbml_string(_doc([("cv", _plus(size, scaled))], params=params)), times
    )
    np.testing.assert_allclose(got, [5, 5, 5 / 3], rtol=1e-8)


def test_a_clock_beside_a_species_threshold_on_one_instant():
    """S = 10·e^(-t/2) passes 5 at t = 2·ln 2, where `time < t1` flips too."""
    t1 = 2 * math.log(2)
    on_s = _pw((_cn(0), _rel("gt", _ci("S"), _ci("a"))), otherwise=_cn(1))
    size, params = _steps(("t1", t1), extra='<parameter id="a" value="5" constant="true"/>')
    times = np.array([0.0, 1.0, 1.3, 1.5, 2.5])
    got = _sn_at(
        bngsim.Model.from_sbml_string(_doc([("cv", _plus(size, on_s))], params=params)), times
    )
    np.testing.assert_allclose(got, 5 / (1 + 2 * (times >= t1)), rtol=1e-8)


def test_jumps_on_one_instant_beside_a_size_that_moves():
    """cv := 1 + 0.1·S + two steps one ulp apart: the amount, 10, over cv."""
    size, params = _steps(("t1", 2.0), ("t2", _ulps_past(2.0)))
    size = _plus(size, _times(_cn(0.1), _ci("S")))
    times = np.array([0.0, 1.0, 1.9, 2.5, 3.5])
    got = _sn_at(bngsim.Model.from_sbml_string(_doc([("cv", size)], params=params)), times)
    np.testing.assert_allclose(got, 10 / (1 + 0.1 * _s(times) + 2 * (times >= 2)), rtol=1e-7)


@pytest.mark.parametrize(("t1", "t2"), [(2.0, 2.0), (2.0, 3.0)], ids=["together", "apart"])
def test_an_equality_beside_a_step(t1, t2):
    """`1 + piecewise(1, time == t1, 0) + piecewise(0, time < t2, 1)`: with t1 =
    t2 the size is 3 at the instant and 2 after it, and [Sn] 5/2 after (RR:
    2.5). It read the step as passed both where the equality was reached and
    where it was left, and [Sn] stayed at 5."""
    eq = _pw((_cn(1), _rel("eq", T, _ci("t1"))), otherwise=_cn(0))
    step = _pw((_cn(0), _rel("lt", T, _ci("t2"))), otherwise=_cn(1))
    params = (
        f'<parameter id="t1" value="{t1!r}" constant="true"/>'
        f'<parameter id="t2" value="{t2!r}" constant="true"/>'
    )
    times = np.array([0.0, 1.5, 2.5, 3.5])
    xml = _doc([("cv", _plus(_plus(_cn(1), eq), step))], params=params)
    got = _sn_at(bngsim.Model.from_sbml_string(xml), times)
    np.testing.assert_allclose(got, 5 / (1 + (times >= t2)), rtol=1e-8)


def test_a_follower_of_jumps_on_one_instant_takes_the_whole_instant():
    """cv goes 1 -> 3 by two steps one ulp apart, and cw := 1 + [Sn] from 6
    to 1 + 5/3: [Sw] = 18/(8/3) after, the amount of Sw held."""
    size, params = _steps(("t1", 2.0), ("t2", _ulps_past(2.0)))
    xml = _sized_doc(
        ["cv", "cw"],
        [("Sn", "cv", 5), ("Sw", "cw", 3)],
        [("cv", size), ("cw", ONE_PLUS_SN)],
        params=params,
    )
    times = np.array([0.0, 1.9, 2.5])
    result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        sample_times=times, rtol=1e-10, atol=1e-12
    )
    x = np.asarray(result.species)
    names = list(result.species_names)
    np.testing.assert_allclose(x[:, names.index("Sn")], [5, 5, 5 / 3], rtol=1e-8)
    np.testing.assert_allclose(x[:, names.index("Sw")], [3, 3, 18 / (1 + 5 / 3)], rtol=1e-8)


def test_the_sensitivity_of_jumps_a_constant_puts_on_one_instant_stands():
    """With t1 = t2 the loader reads each jump on the side it leaves, and the
    run's columns are what they were: d[Sn]/d[Sn]0 = 1/3 after."""
    model = bngsim.Model.from_sbml_string(_two_steps(2.0, 2.0))
    result = bngsim.Simulator(
        model, method="ode", sensitivity_params=["k"], sensitivity_ic=["Sn"]
    ).run(sample_times=[0.0, 1.9, 2.5], rtol=1e-10, atol=1e-12)
    j = list(result.species_names).index("Sn")
    np.testing.assert_allclose(np.asarray(result.species)[:, j], [5, 5, 5 / 3], rtol=1e-8)
    np.testing.assert_allclose(
        np.asarray(result.sensitivities_ic)[:, j, 0], [1, 1, 1 / 3], atol=1e-8
    )


def test_the_sensitivity_of_jumps_a_few_ulp_apart_is_refused():
    """It returned [Sn] = 10/3 with no word; the derivative of the instant's
    one rescale is not implemented."""
    model = bngsim.Model.from_sbml_string(_two_steps(2.0, _ulps_past(2.0)))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    with pytest.raises(
        bngsim.SimulationError, match=r"compartment 'cv' jumps 2 times at one instant.*#1181"
    ):
        sim.run(sample_times=[0.0, 1.9, 2.5], rtol=1e-10, atol=1e-12)


def test_the_sensitivity_of_an_equality_beside_a_step_names_what_it_refuses():
    """t1 = t2 relates the two roots, and the equality is read as it stands, so
    the instant's rescale is the engine's; the refusal no longer says the roots
    are unrelated."""
    eq = _pw((_cn(1), _rel("eq", T, _ci("t1"))), otherwise=_cn(0))
    step = _pw((_cn(0), _rel("lt", T, _ci("t2"))), otherwise=_cn(1))
    params = (
        '<parameter id="t1" value="2.0" constant="true"/>'
        '<parameter id="t2" value="2.0" constant="true"/>'
    )
    xml = _doc([("cv", _plus(_plus(_cn(1), eq), step))], params=params)
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml_string(xml), method="ode", sensitivity_params=["k"]
    )
    with pytest.raises(bngsim.SimulationError, match=r"jumps 2 times at one instant") as err:
        sim.run(sample_times=[0.0, 1.5, 2.5], rtol=1e-10, atol=1e-12)
    assert "no run constant relates" not in str(err.value)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "codegen"])
def test_a_clone_rescales_once_too(codegen):
    """Every parallel worker integrates a clone. The size before the jump is an
    expression of the clone's own evaluator; left at the source's index, the
    clone read another expression there and gave [Sn] 2.5 for 5/3."""
    size, params = _steps(("t1", 2.0), ("t2", _ulps_past(2.0)))
    model = bngsim.Model.from_sbml_string(_doc([("cv", size)], params=params))
    times = np.array([0.0, 1.9, 2.5])
    np.testing.assert_allclose(
        _sn_at(model.clone(), times, codegen=codegen), [5, 5, 5 / 3], rtol=1e-8
    )


def test_the_jumps_that_stand_down_are_drawn_as_before():
    """Jumps of one instant tie on their priority, and the draw among them is
    made as it was before #1181: a seed's later draws stay where they were.
    Marked done before the draw, the one that stands down took no draw, and
    with t1 = t2, where nothing else changed, every later tie-break moved.

    Oracle: the same model with two plain events tied at t = 2 in place of the
    jumps, which makes the same one draw there."""
    at = lambda t: _rel("geq", T, _cn(t))  # noqa: E731
    later = _event("E1", at(3), [("X", _cn(1))]) + _event("E2", at(3), [("X", _cn(2))])
    tie = _event("F1", at(2), [("Y", _cn(1))]) + _event("F2", at(2), [("Y", _cn(2))])
    params = (
        '<parameter id="t1" value="2.0" constant="true"/>'
        '<parameter id="t2" value="2.0" constant="true"/>'
    )
    jumps = _plus(_plus(_cn(1), _step_at("t1")), _step_at("t2"))
    species = _sp("X") + _sp("Y")
    with_jumps = _doc([("cv", jumps)], params=params, events=later, species=species)
    with_events = _doc([("cv", _cn(1))], params=params, events=later + tie, species=species)

    def x_after(xml, seed):
        result = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
            sample_times=[0.0, 2.5, 3.5], rtol=1e-10, atol=1e-12, seed=seed
        )
        names = list(result.species_names)
        x = np.asarray(result.species)
        return x[-1, names.index("X")], x[-1, names.index("Sn")]

    got = [x_after(with_jumps, seed) for seed in range(16)]
    want = [x_after(with_events, seed)[0] for seed in range(16)]
    assert len(set(want)) == 2  # the draw decides it
    assert [x for x, _ in got] == want
    np.testing.assert_allclose([sn for _, sn in got], 5 / 3, rtol=1e-8)
