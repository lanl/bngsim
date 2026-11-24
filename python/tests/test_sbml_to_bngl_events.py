"""sbml_to_bngl's events → BNGL actions, read against what bngsim's own run does.

A fixed-time SBML event becomes ``simulate`` phases with a ``setConcentration``
at each fire time (``convert/_events.py``). The protocol the report carries is
read here as its dose schedule, ``[(time, target, value), …]``, in the order the
actions apply. Where the converter accepts a model, replaying that schedule over
the decay it doses gives what bngsim's run of the source SBML gives, so the
``.bngl`` runs as the model does. Where it cannot know what the model does, it
refuses. Nothing here needs BNG2.pl.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import ConversionError
from bngsim.convert import StateChange, sbml_to_bngl
from bngsim.convert._events import sbml_events_to_protocol

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

_M = '<math xmlns="http://www.w3.org/1998/Math/MathML">'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)
_K = 0.1  # every species decays at this rate
_X0 = {"A": 1.0, "B": 2.0}


def _cn(x: object) -> str:
    return f"<cn>{x}</cn>"


def _ci(name: str) -> str:
    return f"<ci>{name}</ci>"


def _at(threshold: str, op: str = "geq") -> str:
    """The trigger ``time <op> threshold``."""
    return f"<apply><{op}/>{_TIME}{threshold}</apply>"


def _event(
    eid: str,
    trigger: str,
    assignments: dict[str, str],
    *,
    initial_value: bool = True,
    priority: str | None = None,
    delay: str | None = None,
) -> str:
    sets = "".join(
        f'<eventAssignment variable="{var}">{_M}{math_}</math></eventAssignment>'
        for var, math_ in assignments.items()
    )
    return (
        f'<event id="{eid}" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="{str(initial_value).lower()}" persistent="true">'
        f"{_M}{trigger}</math></trigger>"
        + (f"<delay>{_M}{delay}</math></delay>" if delay is not None else "")
        + (f"<priority>{_M}{priority}</math></priority>" if priority is not None else "")
        + f"<listOfEventAssignments>{sets}</listOfEventAssignments></event>"
    )


def _species(sid: str, x0: float) -> str:
    return (
        f'<species id="{sid}" compartment="C" initialConcentration="{x0}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
    )


def _decay(sid: str) -> str:
    return (
        f'<reaction id="J{sid}" reversible="false"><listOfReactants>'
        f'<speciesReference species="{sid}" stoichiometry="1" constant="true"/>'
        f"</listOfReactants><kineticLaw>{_M}<apply><times/><ci>k</ci><ci>{sid}</ci></apply>"
        "</math></kineticLaw></reaction>"
    )


def _model_body(*events: str, parameters: str = "", extra: str = "", compartment: str = "") -> str:
    """A and B decaying at k in C, and the events. ``extra`` holds the
    initialAssignments and rules, in SBML's order."""
    return (
        "<listOfCompartments>"
        + (compartment or '<compartment id="C" size="1" spatialDimensions="3" constant="true"/>')
        + "</listOfCompartments><listOfSpecies>"
        + "".join(_species(s, x0) for s, x0 in _X0.items())
        + '</listOfSpecies><listOfParameters><parameter id="k" value="0.1" constant="true"/>'
        + parameters
        + "</listOfParameters>"
        + extra
        + "<listOfReactions>"
        + "".join(_decay(s) for s in _X0)
        + "</listOfReactions>"
        + (f"<listOfEvents>{''.join(events)}</listOfEvents>" if events else "")
    )


def _sbml(*events: str, **body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        f'<model id="m">{_model_body(*events, **body)}</model></sbml>'
    )


def _comp_sbml(*events: str, **body: str) -> str:
    """The same model as the one submodel ``sub`` of a comp document, whose top
    model holds nothing else."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" '
        'xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1" '
        'level="3" version="1" comp:required="true">'
        '<model id="top"><comp:listOfSubmodels>'
        '<comp:submodel comp:id="sub" comp:modelRef="inner"/>'
        "</comp:listOfSubmodels></model>"
        '<comp:listOfModelDefinitions><comp:modelDefinition id="inner">'
        f"{_model_body(*events, **body)}"
        "</comp:modelDefinition></comp:listOfModelDefinitions></sbml>"
    )


def _write(tmp_path: Path, xml: str, name: str = "m.xml") -> Path:
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return path


def _convert(tmp_path: Path, xml: str, t_span: tuple[float, float] = (0.0, 3.0)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sbml_to_bngl(_write(tmp_path, xml), None, strict=True, t_span=t_span, n_points=7)


def _schedule(report) -> list[tuple[float, str, float]]:
    """``(time, target, value)`` for each ``setConcentration``, in the order the
    actions apply it: the time is where the phase before it ends."""
    out: list[tuple[float, str, float]] = []
    steps = report.protocol.steps if report.protocol is not None else ()
    # One before the first phase is at the start, where that phase begins.
    t = next((float(s.t_span[0]) for s in steps if not isinstance(s, StateChange)), 0.0)
    for step in steps:
        if isinstance(step, StateChange):
            assert step.kind == "set_concentration", step
            out.append((t, step.target, float(step.value)))
        else:
            t = float(step.t_span[1])
    return out


def _replay(schedule, name: str, x0: float, t0: float, t: float) -> float:
    """Species ``name`` at ``t`` under the schedule: the last dose before ``t``
    (or ``x0`` at the start), decayed from there."""
    x, since = x0, t0
    for ft, target, value in schedule:
        if target == name and ft < t:
            x, since = value, ft
    return x * math.exp(-_K * (t - since))


def _assert_runs_as_the_model(
    xml: str, report, t_span: tuple[float, float] = (0.0, 3.0), prefix: str = ""
):
    """The schedule, replayed, gives A and B (named ``prefix`` + id) as
    bngsim's run of ``xml`` gives them, at every sample but the fire times
    (where a sample is read before or after the dose as the run lands)."""
    schedule = _schedule(report)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_sbml_string(xml)
        res = bngsim.Simulator(model, method="ode").run(
            t_span=t_span, n_points=25, rtol=1e-10, atol=1e-12
        )
    times = np.asarray(res.time)
    columns = np.asarray(res.species)
    names = list(res.species_names)
    fire_times = {ft for ft, _, _ in schedule}
    for sid, x0 in _X0.items():
        name = prefix + sid
        for i, t in enumerate(times):
            if any(abs(t - ft) < 1e-9 for ft in fire_times):
                continue
            want = columns[i, names.index(name)]
            got = _replay(schedule, name, x0, t_span[0], float(t))
            assert got == pytest.approx(want, rel=1e-6), (name, float(t), schedule)


# ─── a comp submodel's events (issue #1787) ────────────────────────────────
# Model.from_sbml flattens a comp document before it reads it, so the
# submodel's events are the model's. The converter read the raw document,
# whose top model has none, and wrote no actions: ok=True, lossy=[].


def test_a_comp_submodels_fixed_time_event_is_carried_into_actions(tmp_path):
    xml = _comp_sbml(_event("E1", _at(_cn(1)), {"A": _cn(5)}))
    rep = _convert(tmp_path, xml)
    assert rep.ok, rep.lossy
    assert _schedule(rep) == [(1.0, "sub__A", 5.0)]
    _assert_runs_as_the_model(xml, rep, prefix="sub__")


def test_a_comp_submodels_state_triggered_event_is_refused(tmp_path):
    trigger = f"<apply><lt/>{_ci('A')}{_cn(0.5)}</apply>"
    xml = _comp_sbml(_event("E1", trigger, {"A": _cn(5)}))
    with pytest.raises(ConversionError, match="'sub__E1' is state-triggered"):
        _convert(tmp_path, xml)


def test_a_model_event_the_document_does_not_declare_is_refused(tmp_path):
    """The backstop: where the model was loaded with an event the document read
    for translation lacks, the converter says so rather than drop it."""
    one = _event("E1", _at(_cn(1)), {"A": _cn(5)})
    two = _event("E2", _at(_cn(2)), {"A": _cn(7)})
    model = bngsim.Model.from_sbml_string(_sbml(one, two))
    protocol, lossy = sbml_events_to_protocol(_write(tmp_path, _sbml(one)), model)
    assert protocol is None
    assert len(lossy) == 1 and "declares 1, but the loaded model has 2" in lossy[0]


def test_the_loaders_size_jump_events_are_not_counted_as_the_models(tmp_path):
    """A piecewise size adds two events of the loader's own (``_size_jump_*``),
    which the document does not declare and the backstop must not count."""
    piece = (
        f"<piecewise><piece>{_cn(1)}<apply><lt/>{_TIME}{_cn(2)}</apply></piece>"
        f"<otherwise>{_cn(5)}</otherwise></piecewise>"
    )
    xml = _sbml(
        _event("E1", _at(_cn(1)), {"A": _cn(5)}),
        compartment='<compartment id="C" size="1" spatialDimensions="3" constant="false"/>',
        extra=f'<listOfRules><assignmentRule variable="C">{_M}{piece}</math></assignmentRule>'
        "</listOfRules>",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_sbml_string(xml)
    assert list(model._core.event_size_jump_compartments) == ["", "C", "C"]
    protocol, lossy = sbml_events_to_protocol(_write(tmp_path, xml), model)
    assert lossy == []
    assert protocol is not None


# ─── what a trigger time, delay or value folds from (issue #1784) ──────────
# Each parameter and compartment folded at its declared value. One an
# initialAssignment sets starts at that value instead; one a rule or an event
# changes has no one value to fold.


def _param(pid: str, value: object, constant: bool = True) -> str:
    return f'<parameter id="{pid}" value="{value}" constant="{str(constant).lower()}"/>'


def _initial(**symbols: str) -> str:
    return (
        "<listOfInitialAssignments>"
        + "".join(
            f'<initialAssignment symbol="{s}">{_M}{m}</math></initialAssignment>'
            for s, m in symbols.items()
        )
        + "</listOfInitialAssignments>"
    )


def _rules(**rules: tuple[str, str]) -> str:
    """``var=(kind, math)``, kind ``assignmentRule`` or ``rateRule``."""
    return (
        "<listOfRules>"
        + "".join(f'<{k} variable="{v}">{_M}{m}</math></{k}>' for v, (k, m) in rules.items())
        + "</listOfRules>"
    )


def _times(*factors: str) -> str:
    return f"<apply><times/>{''.join(factors)}</apply>"


def _plus(*terms: str) -> str:
    return f"<apply><plus/>{''.join(terms)}</apply>"


_C_AND_D = (
    '<compartment id="C" size="1" spatialDimensions="3" constant="true"/>'
    '<compartment id="D" size="1" spatialDimensions="3" constant="{constant}"/>'
)
_AT_HALF = "E0", _at(_cn(0.5))


@pytest.mark.parametrize(
    ("events", "body", "want"),
    [
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _cn(5)}, delay=_ci("d"))],
            {"parameters": _param("d", 0), "extra": _initial(d=_cn(1))},
            [(2.0, "A", 5.0)],
            id="a delay an initialAssignment sets",
        ),
        pytest.param(
            [_event("E1", _at(_ci("T")), {"A": _cn(5)})],
            {"parameters": _param("T", 1), "extra": _initial(T=_cn(2))},
            [(2.0, "A", 5.0)],
            id="a trigger time an initialAssignment sets",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _ci("v")})],
            {"parameters": _param("v", 1), "extra": _initial(v=_cn(5))},
            [(1.0, "A", 5.0)],
            id="a value an initialAssignment sets",
        ),
        pytest.param(
            [_event("E1", _at(_ci("T")), {"A": _cn(5)})],
            {
                "parameters": _param("T", 0) + _param("h", 0),
                "extra": _initial(T=_times(_cn(2), _ci("h")), h=_cn(1)),
            },
            [(2.0, "A", 5.0)],
            id="an initialAssignment that reads a later one",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _cn(5)}, delay=_ci("d"))],
            {
                "parameters": _param("d", 0, constant=False) + _param("h", 0.5),
                "extra": _rules(d=("assignmentRule", _times(_cn(2), _ci("h")))),
            },
            [(2.0, "A", 5.0)],
            id="a delay an assignment rule over constants sets",
        ),
        pytest.param(
            [
                _event("E1", _at(_cn(1)), {"A": _ci("v0")}),
                _event("E2", _at(_cn(2)), {"v": _cn(7)}),
            ],
            {
                "parameters": _param("v", 4, constant=False) + _param("v0", 0),
                "extra": _initial(v0=_ci("v")),
            },
            [(1.0, "A", 4.0), (2.0, "v", 7.0)],
            id="an initialAssignment that reads the start of what an event changes",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"v": _plus(_ci("v"), _cn(4)), "A": _ci("v")})],
            {"parameters": _param("v", 1, constant=False)},
            [(1.0, "v", 5.0), (1.0, "A", 1.0)],
            id="a value that reads the target only its own event changes",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _ci("D")})],
            {"compartment": _C_AND_D.format(constant="true"), "extra": _initial(D=_cn(3))},
            [(1.0, "A", 3.0)],
            id="a compartment size an initialAssignment sets",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _ci("D")})],
            {
                "compartment": _C_AND_D.format(constant="false"),
                "extra": _rules(D=("assignmentRule", _cn(3))),
            },
            [(1.0, "A", 3.0)],
            id="a compartment size an assignment rule over constants sets",
        ),
    ],
)
def test_a_name_folds_to_the_value_the_run_gives_it(tmp_path, events, body, want):
    xml = _sbml(*events, **body)
    rep = _convert(tmp_path, xml)
    assert rep.ok, rep.lossy
    assert _schedule(rep) == want
    _assert_runs_as_the_model(xml, rep)


@pytest.mark.parametrize(
    ("events", "body", "note"),
    [
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _cn(5)}, delay=_ci("d"))],
            {
                "parameters": _param("d", 0, constant=False),
                "extra": _rules(d=("assignmentRule", _ci("B"))),
            },
            "'E1' has a non-constant delay",
            id="a delay an assignment rule over a species sets",
        ),
        pytest.param(
            [_event("E1", _at(_times(_cn(2), _ci("T"))), {"A": _cn(5)})],
            {
                "parameters": _param("T", 1, constant=False),
                "extra": _rules(T=("rateRule", _cn(1))),
            },
            "'E1' is state-triggered",
            id="a trigger time a rate rule moves",
        ),
        pytest.param(
            [
                _event(*_AT_HALF, {"T": _cn(3)}),
                _event("E1", _at(_ci("T")), {"A": _cn(5)}),
            ],
            {"parameters": _param("T", 1, constant=False)},
            "'E1' is state-triggered",
            id="a trigger time another event resets",
        ),
        pytest.param(
            [_event("E1", _at(_ci("T")), {"T": _plus(_ci("T"), _cn(1)), "A": _cn(5)})],
            {"parameters": _param("T", 1, constant=False)},
            "'E1' is state-triggered",
            id="a trigger time its own event moves on",
        ),
        pytest.param(
            [
                _event(*_AT_HALF, {"d": _cn(2)}),
                _event("E1", _at(_cn(1)), {"A": _cn(5)}, delay=_ci("d")),
            ],
            {"parameters": _param("d", 0, constant=False)},
            "'E1' has a non-constant delay",
            id="a delay another event resets",
        ),
        pytest.param(
            [
                _event(*_AT_HALF, {"v": _cn(7)}),
                _event("E1", _at(_cn(1)), {"A": _ci("v")}),
            ],
            {"parameters": _param("v", 1, constant=False)},
            "'E1' assigns 'A' a non-constant value",
            id="a value another event resets",
        ),
        pytest.param(
            [
                _event(*_AT_HALF, {"v": _plus(_ci("v"), _cn(1))}),
                _event("E1", _at(_cn(1)), {"v": _plus(_ci("v"), _cn(1))}),
            ],
            {"parameters": _param("v", 1, constant=False)},
            "'E0' assigns 'v' a non-constant value",
            id="a target two events step",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _ci("v")})],
            {"parameters": _param("v", 1), "extra": _initial(v=_ci("B"))},
            "'E1' assigns 'A' a non-constant value",
            id="a value whose initialAssignment reads a species",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _ci("v")})],
            {
                "parameters": _param("v", 1),
                "extra": _initial(
                    v=f"<apply><power/>{_cn(-8)}<apply><divide/>{_cn(1)}{_cn(3)}</apply></apply>"
                ),
            },
            "'E1' assigns 'A' a non-constant value",
            id="a value whose initialAssignment has no real value",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _ci("D")})],
            {
                "compartment": _C_AND_D.format(constant="false"),
                "extra": _rules(D=("assignmentRule", _plus(_cn(1), _TIME))),
            },
            "'E1' assigns 'A' a non-constant value",
            id="a compartment size an assignment rule over the time sets",
        ),
        pytest.param(
            [
                _event(*_AT_HALF, {"D": _cn(3)}),
                _event("E1", _at(_cn(1)), {"A": _ci("D")}),
            ],
            {"compartment": _C_AND_D.format(constant="false")},
            "'E1' assigns 'A' a non-constant value",
            id="a compartment size another event resets",
        ),
    ],
)
def test_a_name_a_rule_or_event_moves_is_not_folded(tmp_path, events, body, note):
    """Refused, naming the event: its time or value moves with the run."""
    xml = _sbml(*events, **body)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_sbml_string(xml)
    protocol, lossy = sbml_events_to_protocol(_write(tmp_path, xml), model)
    assert protocol is None
    assert any(note in n for n in lossy), lossy
    with pytest.raises(ConversionError, match=note):
        _convert(tmp_path, xml)


# ─── a trigger true at the start, and events at one instant (issue #1785) ──
# initialValue is the trigger's value just before the run starts, so a trigger
# already true there fires only where it is false. Events at one instant
# execute as the run executes them: those whose strict trigger turns true just
# past the instant after the rest, and each part in priority order, higher
# first, an unset priority counting as 0.


def _gt(threshold: str) -> str:
    return _at(threshold, "gt")


def _a(eid: str, trigger: str, value: object, **kw) -> str:
    """An event setting A to ``value``."""
    return _event(eid, trigger, {"A": _cn(value)}, **kw)


@pytest.mark.parametrize(
    ("events", "t_span", "want"),
    [
        pytest.param(
            [_a("E1", _at(_cn(0)), 5)], (0, 3), [], id="time >= 0, initialValue true: never"
        ),
        pytest.param(
            [_a("E1", _at(_cn(0)), 5, initial_value=False)],
            (0, 3),
            [(0.0, "A", 5.0)],
            id="time >= 0, initialValue false: at the start",
        ),
        pytest.param(
            [_a("E1", f"<apply><leq/>{_cn(0)}{_TIME}</apply>", 5)],
            (0, 3),
            [],
            id="0 <= time, initialValue true: never",
        ),
        pytest.param(
            [_a("E1", _gt(_cn(0)), 5)],
            (0, 3),
            [(0.0, "A", 5.0)],
            id="time > 0, initialValue true: just past the start",
        ),
        pytest.param(
            [_a("E1", _gt(_cn(-1)), 5)], (0, 3), [], id="time > -1, initialValue true: never"
        ),
        pytest.param(
            [_a("E1", _gt(_cn(-1)), 5, initial_value=False)],
            (0, 3),
            [(0.0, "A", 5.0)],
            id="time > -1, initialValue false: at the start",
        ),
        pytest.param(
            [_a("E1", _at(_cn(1)), 5)], (2, 5), [], id="time >= 1 from t = 2, initialValue true"
        ),
        pytest.param(
            [_a("E1", _at(_cn(1)), 5, initial_value=False)],
            (2, 5),
            [(2.0, "A", 5.0)],
            id="time >= 1 from t = 2, initialValue false: at the start",
        ),
        pytest.param(
            [_a("E1", _at(_cn(1)), 5, initial_value=False, delay=_cn(1))],
            (2, 5),
            [(3.0, "A", 5.0)],
            id="time >= 1 from t = 2, initialValue false, delay 1: from the start",
        ),
        pytest.param(
            [_event("E1", _at(_cn(0)), {"A": _ci("B")})],
            (0, 3),
            [],
            id="an event that never fires is not refused for what it reads",
        ),
    ],
)
def test_a_trigger_true_at_the_start_fires_only_where_initial_value_is_false(
    tmp_path, events, t_span, want
):
    xml = _sbml(*events)
    rep = _convert(tmp_path, xml, t_span=t_span)
    assert rep.ok, rep.lossy
    assert _schedule(rep) == want
    _assert_runs_as_the_model(xml, rep, t_span=t_span)


@pytest.mark.parametrize(
    ("events", "want"),
    [
        pytest.param(
            [
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
                _a("E2", _at(_cn(1)), 10, priority=_cn(2)),
            ],
            [10.0, 5.0],
            id="the higher priority first",
        ),
        pytest.param(
            [_a("E2", _at(_cn(1)), 10, priority=_cn(-1)), _a("E1", _at(_cn(1)), 5)],
            [5.0, 10.0],
            id="an unset priority counts as 0",
        ),
        pytest.param(
            [
                _a("E2", _gt(_cn(1)), 10, priority=_cn(2)),
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
            ],
            [5.0, 10.0],
            id="a strict trigger after the rest, whatever its priority",
        ),
        pytest.param(
            [
                _a("E2", _gt(_cn(0.5)), 5, priority=_cn(2), delay=_cn(0.5)),
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
                _event("E3", _at(_cn(1)), {"B": _cn(7)}),
            ],
            [5.0, 5.0],
            id="a strict trigger's delayed execution beside events that agree",
        ),
        pytest.param(
            [
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
                _a("E2", _at(_cn(0.5)), 10, priority=_cn(2), delay=_cn(0.5)),
            ],
            [10.0, 5.0],
            id="a delayed execution by its priority",
        ),
        pytest.param(
            [
                _a("E3", _at(_cn(1)), 7, priority=_cn(1)),
                _a("E1", _at(_cn(1)), 5, priority=_cn(2)),
                _a("E2", _at(_cn(1)), 10, priority=_cn(2)),
            ],
            [5.0, 10.0, 7.0],
            id="a tie that a lower priority overwrites",
        ),
        pytest.param(
            [_a("E1", _at(_cn(1)), 5), _a("E2", _at(_cn(1)), 5)],
            [5.0, 5.0],
            id="equal priorities that agree",
        ),
        pytest.param(
            [
                _a("E1", _at(_cn(1)), 5, priority=_ci("B")),
                _a("E2", _at(_cn(2)), 10, priority=_cn(1)),
            ],
            [5.0, 10.0],
            id="a priority that is not a constant, alone at its instant",
        ),
    ],
)
def test_events_at_one_instant_execute_in_the_order_the_run_gives(tmp_path, events, want):
    xml = _sbml(*events)
    rep = _convert(tmp_path, xml)
    assert rep.ok, rep.lossy
    assert [v for _, target, v in _schedule(rep) if target == "A"] == want
    _assert_runs_as_the_model(xml, rep)


def test_equal_priorities_that_set_different_species_are_carried(tmp_path):
    xml = _sbml(
        _event("E1", _at(_cn(1)), {"A": _cn(5)}),
        _event("E2", _at(_cn(1)), {"B": _cn(7)}),
    )
    rep = _convert(tmp_path, xml)
    assert rep.ok, rep.lossy
    assert _schedule(rep) == [(1.0, "A", 5.0), (1.0, "B", 7.0)]
    _assert_runs_as_the_model(xml, rep)


@pytest.mark.parametrize(
    ("events", "note"),
    [
        pytest.param(
            [
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
                _a("E2", _at(_cn(1)), 10, priority=_cn(1)),
            ],
            "events 'E1' and 'E2' set 'A' to different values at t = 1 with equal priority",
            id="equal priorities",
        ),
        pytest.param(
            [_a("E1", _at(_cn(1)), 5), _a("E2", _at(_cn(1)), 10)],
            "events 'E1' and 'E2' set 'A' to different values at t = 1 with equal priority",
            id="no priorities",
        ),
        pytest.param(
            [_a("E1", _at(_cn(1)), 5), _a("E2", _at(_cn(1)), 10, priority=_cn(0))],
            "events 'E1' and 'E2' set 'A' to different values at t = 1 with equal priority",
            id="an unset priority beside 0",
        ),
        pytest.param(
            [
                _a("E1", _at(_cn(1)), 5, priority=_ci("B")),
                _a("E2", _at(_cn(1)), 10, priority=_cn(1)),
            ],
            "'E1' has a non-constant priority",
            id="a priority that is not a constant",
        ),
        pytest.param(
            [
                _a("E2", _gt(_cn(0.5)), 10, priority=_cn(2), delay=_cn(0.5)),
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
            ],
            "'E2', delayed from a strict trigger, executes before or after the other",
            id="a strict trigger's delayed execution, which the run's rounding places",
        ),
    ],
)
def test_events_whose_order_is_not_known_ahead_are_refused(tmp_path, events, note):
    """Where the run leaves a species at a value its random draw between
    events decides, or its rounding, the actions have no one value to set."""
    with pytest.raises(ConversionError, match=note):
        _convert(tmp_path, _sbml(*events))
