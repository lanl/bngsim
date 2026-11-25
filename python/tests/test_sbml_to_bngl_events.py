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
import sys
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import ConversionError
from bngsim.convert import StateChange, sbml_to_bngl
from bngsim.convert._events import _instant_reach, sbml_events_to_protocol

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


def _convert(
    tmp_path: Path, xml: str, t_span: tuple[float, float] = (0.0, 3.0), n_points: int = 7
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sbml_to_bngl(
            _write(tmp_path, xml), None, strict=True, t_span=t_span, n_points=n_points
        )


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
    (where a sample is read before or after the dose as the run lands). For
    several seeds, grids and tolerances."""
    schedule = _schedule(report)
    fire_times = {ft for ft, _, _ in schedule}
    # The grid and the tolerance move the run's stops and steps; the seed, its
    # draws among equal priorities.
    for seed, (n_points, rtol) in enumerate(
        [(25, 1e-10), (7, 1e-10), (101, 1e-9), (2, 1e-10), (31, 1e-8), (13, 1e-10)]
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
                t_span=t_span, n_points=n_points, rtol=rtol, atol=rtol * 1e-2, seed=seed
            )
        times = np.asarray(res.time)
        columns = np.asarray(res.species)
        names = list(res.species_names)
        for sid, x0 in _X0.items():
            name = prefix + sid
            for i, t in enumerate(times):
                if any(abs(t - ft) < 1e-9 for ft in fire_times):
                    continue
                want = columns[i, names.index(name)]
                got = _replay(schedule, name, x0, t_span[0], float(t))
                assert got == pytest.approx(want, rel=1e-6), (seed, name, float(t), schedule)


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
            "'E1' assigns 'A' a value that reads 'v', which keeps one value through the run, "
            "but its initialAssignment reads a species' initial value",
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
            "'E1' assigns 'A' a value that reads 'v', which keeps one value through the run, "
            "but its initialAssignment does not fold to a number",
            id="a value whose initialAssignment has no real value",
        ),
        pytest.param(
            [
                _event(
                    "E1",
                    _at(_cn(1)),
                    {"A": f"<apply><power/>{_cn(0)}{_cn(-1)}</apply>"},
                )
            ],
            {},
            "'E1' assigns 'A' a value of inf, which is not a finite number",
            id="a value that is 0 to a negative power",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _cn(5)}, delay=_ci("d"))],
            {"parameters": _param("d", 1), "extra": _initial(d=_cn(-0.5))},
            "'E1' has a delay of -0.5, which the run refuses",
            id="a delay an initialAssignment makes negative",
        ),
        pytest.param(
            [
                _event(
                    "E1",
                    _at(_cn(1)),
                    {"A": _cn(5)},
                    delay=f"<apply><divide/>{_cn(1)}{_cn(0)}</apply>",
                )
            ],
            {},
            "'E1' has a delay of inf, which the run refuses: an event delay must be a "
            "finite, non-negative number",
            id="a delay of 1/0",
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
# execute as the run executes them: in priority order, higher first, an unset
# priority counting as 0. Of equal priority, delayed executions keep the order
# they were triggered in, and where no event has a priority, they follow the
# rest; the run draws among what is left.


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
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5)),
                _a("E1", _at(_cn(1)), 5),
            ],
            [5.0, 10.0],
            id="with no priorities, a delayed execution after one not delayed",
        ),
        pytest.param(
            [
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5), priority=_cn(0)),
                _a("E1", _at(_cn(1)), 5, priority=_cn(0)),
            ],
            [5.0, 10.0],
            id="with priorities a plain 0, a delayed execution after one not delayed",
        ),
        pytest.param(
            [
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5)),
                _a("E1", _at(_cn(0.25)), 5, delay=_cn(0.75)),
            ],
            [5.0, 10.0],
            id="delayed executions in the order they were triggered",
        ),
        pytest.param(
            [
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5), priority=_cn(1)),
                _a("E1", _at(_cn(0.25)), 5, delay=_cn(0.75), priority=_cn(1)),
            ],
            [5.0, 10.0],
            id="delayed executions of one priority in the order they were triggered",
        ),
        pytest.param(
            [
                _a("E1", _gt(_cn(0.3)), 10, priority=_cn(-1)),
                _a("E2", _at(_cn(0.1)), 5, delay=_cn(0.2000001)),
            ],
            [10.0, 5.0],
            id="an execution due well past a strict trigger's time",
        ),
        pytest.param(
            [
                _a("E1", _at(_cn(0.7)), 5, delay=_cn(0.5), priority=_cn(2)),
                _a("E2", _at(_cn(1.2)), 10, priority=_cn(1)),
            ],
            [5.0, 10.0],
            id="a delayed execution due on an instant, with no stop near its trigger",
        ),
        pytest.param(
            [
                _a("E1", _at(_cn(0.3)), 5, priority=_cn(1)),
                _a("E2", _at(_cn(0.3)), 10, priority=_cn(2)),
                _event("EB", _at(_times(_cn(3), _cn(0.1))), {"B": _cn(7)}),
            ],
            [10.0, 5.0],
            id="triggers on one double, read together beside a stop a double after it",
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
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5), priority=_cn(1)),
            ],
            "events 'E1' and 'E2' set 'A' to different values at t = 1 with equal priority",
            id="a delayed execution beside one not delayed, of one priority",
        ),
        pytest.param(
            [
                _a("E1", _at(_cn(1)), 5),
                _event(
                    "E3",
                    _at(_cn(1)),
                    {"B": _cn(7)},
                    priority=f"<apply><minus/>{_ci('k')}{_ci('k')}</apply>",
                ),
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5)),
            ],
            "events 'E1' and 'E2' set 'A' to different values at t = 1 with equal priority",
            id="a delayed execution beside one not delayed, in a tie with a priority",
        ),
        pytest.param(
            [
                _a("E1", _at(_cn(0.5)), 5, delay=_cn(0.5)),
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5)),
            ],
            "events 'E1' and 'E2' set 'A' to different values at t = 1 with equal priority",
            id="delayed executions triggered at one instant",
        ),
        pytest.param(
            [
                _a("E3", _at(_cn(1)), 7, priority=_cn(1)),
                _a("E1", _at(_cn(1)), 5),
                _a("E2", _at(_cn(1)), 10),
            ],
            "events 'E1' and 'E2' set 'A' to different values at t = 1 with equal priority",
            id="the note names the tie, not an event before it",
        ),
        pytest.param(
            [
                _event("E1", _gt(_cn(0.7)), {"B": _cn(10)}, delay=_cn(0.2)),
                _event("E2", _at(_cn(0.1)), {"B": _cn(5)}, delay=_cn(0.8)),
            ],
            "events 'E1' and 'E2' set 'B' to different values at t = 0.8999999999999999 and "
            "t = 0.9, close enough together that the translation cannot know which the run",
            id="a strict trigger's delayed execution a double before another",
        ),
        pytest.param(
            [
                _a("E1", _gt(_cn(0.3)), 10, priority=_cn(-1)),
                _a("E2", _at(_cn(0.1)), 5, delay=_cn(0.2), priority=_cn(0)),
            ],
            "events 'E1' and 'E2' set 'A' to different values at t = 0.3 and "
            "t = 0.30000000000000004, close enough together that the translation cannot know",
            id="an execution due a double past a strict trigger's time, of higher priority",
        ),
        pytest.param(
            [
                _a("E1", _gt(_cn(0.3)), 10, priority=_cn(1)),
                _a("E2", _at(_cn(0.1)), 5, delay=_cn(0.2), priority=_cn(1)),
            ],
            "events 'E1' and 'E2' set 'A' to different values at t = 0.3 and "
            "t = 0.30000000000000004, close enough together that the translation cannot know",
            id="an execution due a double past a strict trigger's time, of its priority",
        ),
    ],
)
def test_events_whose_order_is_not_known_ahead_are_refused(tmp_path, events, note):
    """Where the run leaves a species at a value its random draw between
    events decides, or its rounding, the actions have no one value to set.
    Where the note says the run's order is random, it is: A after the
    instant varies with the seed."""
    xml = _sbml(*events)
    with pytest.raises(ConversionError, match=note):
        _convert(tmp_path, xml)
    if "equal priority" in note:
        after = set()
        for seed in range(12):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
                    t_span=(0.0, 3.0), n_points=7, seed=seed
                )
            after.add(round(float(np.asarray(res.species)[3, res.species_names.index("A")]), 6))
        assert len(after) > 1, after


# ─── executions the run can take as one instant or apart ───────────────────
# The run merges two stops on literal time closer than (64 + 4)·ε·max(|a|, |b|)
# into the later one, reads a strict trigger a little past its threshold, and
# counts a delay from where it read the trigger. So executions due within that
# reach of each other, or whose trigger has a stop that close after it, run as
# one instant by priority or apart in time, as the rounding and the run's other
# stops fall: the grid and the seed can decide.

_SEVEN_TENTHS = {"parameters": _param("T3", 0), "extra": _initial(T3=_times(_cn(7), _cn(0.1)))}


@pytest.mark.parametrize(
    ("events", "body", "note"),
    [
        pytest.param(
            [
                _a("E1", _at(_cn(0.7)), 5, delay=_cn(0.5), priority=_cn(2)),
                _a("E2", _at(_cn(1.2)), 10, priority=_cn(1)),
                _event("E3", _at(_ci("T3")), {"B": _cn(7)}),
            ],
            _SEVEN_TENTHS,
            "events 'E1' and 'E2' set 'A' to different values at t = 1.2, close enough together",
            id="a delay counted from a trigger read on a stop a double after it",
        ),
        pytest.param(
            [
                _a("S", _gt(_cn(0.7)), 10, priority=_cn(2)),
                _a("N", _at(_ci("T")), 5, priority=_cn(1)),
                _event("E3", _at(_ci("T3")), {"B": _cn(7)}),
            ],
            {
                "parameters": _param("T3", 0) + _param("T", 0.7),
                "extra": _initial(T3=_times(_cn(7), _cn(0.1))),
            },
            "set 'A' to different values at t = 0.7, close enough together",
            id="a strict trigger read with the rest on a stop a double after them",
        ),
        pytest.param(
            [_a("E1", _at(_cn(0.3)), 5), _a("E2", _at(_ci("T")), 10)],
            {"parameters": _param("T", 0), "extra": _initial(T=_times(_cn(3), _cn(0.1)))},
            "events 'E1' and 'E2' set 'A' to different values at t = 0.3 and "
            "t = 0.30000000000000004",
            id="two triggers a double apart",
        ),
        pytest.param(
            [
                _a("I", _at(_cn(1)), 5),
                _a("D1", _at(_cn(0.5)), 10, delay=_cn(0.5)),
                _event(
                    "S2",
                    _gt(_cn(repr(math.nextafter(1.0, 0.0)))),
                    {"B": _cn(7)},
                    priority=_ci("p0"),
                ),
            ],
            {"parameters": _param("p0", 0)},
            "events 'I' and 'D1' set 'A' to different values at t = 1.0",
            id="a strict trigger a double before a tie, with a priority it brings",
        ),
    ],
)
def test_events_the_run_can_take_as_one_instant_or_apart_are_refused(tmp_path, events, body, note):
    with pytest.raises(ConversionError, match=note):
        _convert(tmp_path, _sbml(*events, **body))


@pytest.mark.parametrize("n_points", [11, 7])
def test_a_trigger_just_after_another_events_instant_is_refused(tmp_path, n_points):
    """Just after EB executes at 0.6 (on the row there with 11 points), the
    run steps over the stop of E1's trigger, a double later at 6*0.1, and
    finds it by its root dozens of doubles late, so E1's delayed execution
    comes due after E0, not with it at 1.1."""
    xml = _sbml(
        _event("EB", _at(_cn(0.6)), {"B": _cn(7)}),
        _a("E1", _at(_times(_cn(6), _cn(0.1))), 4, delay=_cn(0.5), priority=_cn(2)),
        _a("E0", _at(_cn(1.1)), 5, priority=_cn(1)),
    )
    with pytest.raises(ConversionError, match="events 'E1' and 'E0' set 'A' to different"):
        _convert(tmp_path, xml, n_points=n_points)


_D_3_7 = _times(_ci("d"), _cn(3), _cn(7))  # the run's evaluator folds it as d*21


@pytest.mark.parametrize(
    ("events", "body"),
    [
        pytest.param(
            [
                _a("E1", _at(_ci("T")), 4, delay=_cn(0.5), priority=_cn(1)),
                _a(
                    "E3",
                    _at(_plus(_cn(0.5), _times(_cn(0.1), _cn(3), _cn(7)))),
                    5,
                    priority=_cn(2),
                ),
            ],
            {
                "parameters": _param("T", 0) + _param("h", 0.1),
                "extra": _initial(T=_times(_ci("h"), _cn(3), _cn(7))),
            },
            id="an initialAssignment the run reads as 2.1",
        ),
    ],
)
def test_times_a_double_apart_as_the_run_reads_them_are_refused(tmp_path, events, body):
    """Folded left to right here, d*3*7 is 2.1000000000000005, and E1 seemed
    due with E3. The run reads d*21 = 2.1, and executes E1 at 2.6, before E3."""
    with pytest.raises(
        ConversionError,
        match="events 'E3' and 'E1' set 'A' to different values at t = 2.6000000000000005 "
        "and t = 2.6",
    ):
        _convert(tmp_path, _sbml(*events, **body))


def test_a_delay_the_run_computes_otherwise_is_refused(tmp_path):
    """The run reads d*3*7 as d*21 = 2.1, where it folds here left to right to
    2.1000000000000005: the two disagree, so the delay is refused."""
    xml = _sbml(
        _a("E1", _at(_cn(0.5)), 4, delay=_D_3_7, priority=_cn(1)),
        _a("E3", _at(_cn(2.6)), 5, priority=_cn(2)),
        parameters=_param("d", 0.1),
    )
    with pytest.raises(
        ConversionError,
        match="'E1' has a delay that the run computes as 2.1 and the translation folds to "
        "2.1000000000000005",
    ):
        _convert(tmp_path, xml)


# A strict trigger is read a little past its threshold, where the run can take
# it with the other events of the instant, after them, or, where it has a twin
# on the same residual, in a second part of the instant: which, its spelling
# and the run's other stops decide. Delayed, it keeps that offset. So beside an
# event at or just after its time that sets a species differently, it is
# refused.


@pytest.mark.parametrize(
    "events",
    [
        pytest.param(
            [
                _a("E2", _gt(_cn(1)), 10, priority=_cn(2)),
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
            ],
            id="a strict trigger beside one not strict on its threshold",
        ),
        pytest.param(
            [
                _a("E0", _gt(_cn(0)), 10, delay=_cn(1)),
                _a("E1", _at(_cn(1)), 5),
                _a("E2", _at(_cn(1)), 7, priority=_cn(0)),
            ],
            id="a strict trigger's delayed execution on a tie with no priorities",
        ),
        pytest.param(
            [
                _a("E0", _gt(_cn(0.5)), 10, delay=_cn(0.5), priority=_cn(-1)),
                _a("E1", _at(_cn(1)), 5),
                _a("E2", _at(_cn(1)), 7),
            ],
            id="a strict trigger's delayed execution of lower priority on a tie",
        ),
        pytest.param(
            [
                _a("E0", _gt(_cn(0.5)), 7, delay=_cn(0.5), priority=_cn(1)),
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5), priority=_cn(1)),
            ],
            id="a strict trigger's delayed execution beside one triggered before it",
        ),
        pytest.param(
            [
                _a("E0", _gt(_cn(0)), 7, delay=_cn(1), priority=_cn(1)),
                _a("E2", _at(_cn(0)), 10, delay=_cn(1), priority=_cn(1), initial_value=False),
            ],
            id="a strict trigger's delayed execution beside one triggered at the start",
        ),
        pytest.param(
            [
                _a("E2", _at(_cn(0.5)), 10, delay=_cn(0.5), priority=_cn(1)),
                _a("E3", _at(_cn(0.5)), 5, delay=_cn(0.5), priority=_cn(1)),
                _a("E0", _gt(_cn(0.5)), 7, delay=_cn(0.5), priority=_cn(1)),
            ],
            id="a strict trigger's delayed execution on a tie of delayed ones",
        ),
        pytest.param(
            [
                _a("E2", _gt(_cn(0.5)), 10, priority=_cn(2), delay=_cn(0.5)),
                _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
            ],
            id="a strict trigger's delayed execution beside one not delayed",
        ),
        pytest.param(
            [
                _a("E2", _at(_cn(0.25)), 10, delay=_cn(0.75), priority=_cn(1)),
                _a("E0", _gt(_cn(0)), 7, delay=_cn(1), priority=_cn(1)),
            ],
            id="a strict trigger's delayed execution beside one triggered after it",
        ),
    ],
)
def test_a_strict_trigger_beside_an_event_that_sets_a_species_differently_is_refused(
    tmp_path, events
):
    with pytest.raises(
        ConversionError,
        match="close enough together that the translation cannot know which the run executes last",
    ):
        _convert(tmp_path, _sbml(*events))


def test_strict_triggers_with_and_without_a_twin_are_refused(tmp_path):
    """E1's and E4's strict triggers have a twin on their residual, E0's
    non-strict one, and the run holds them to a second part of the instant,
    where they tie; E3's reads the same double through another spelling and
    runs in the first part. By priority alone E3 sets B last, at 7; the run
    leaves 5 or 7 as its draw falls, on some grids."""
    x = f"<apply><times/>{_cn('0.009523809523809525')}{_cn(3)}{_cn(7)}</apply>"
    y = f"<apply><times/>{_ci('p')}{_cn(3)}{_cn(7)}</apply>"
    xml = _sbml(
        _event(
            "E0",
            f"<apply><leq/>{x}{_TIME}</apply>",
            {"A": _cn(10)},
            initial_value=False,
            priority=_cn(2),
            delay=_cn(1),
        ),
        _event(
            "E1",
            f"<apply><lt/>{x}{_TIME}</apply>",
            {"B": _cn(7)},
            initial_value=False,
            priority=_cn(2),
        ),
        _event("E4", _gt(x), {"B": _cn(5)}, initial_value=False, priority=_cn(2)),
        _event(
            "E3", f"<apply><lt/>{y}{_TIME}</apply>", {"A": _cn(10), "B": _cn(7)}, priority=_cn(0)
        ),
        parameters=_param("p", "0.009523809523809525"),
    )
    with pytest.raises(ConversionError, match="set 'B' to different values"):
        _convert(tmp_path, xml)
    after = set()
    for seed in range(6):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
                t_span=(0.0, 3.0), n_points=101, rtol=1e-4, atol=1e-6, seed=seed
            )
        times = np.asarray(res.time)
        i = int(np.searchsorted(times, 0.21 - 1e-9))
        b = float(np.asarray(res.species)[i, res.species_names.index("B")])
        after.add(round(b * math.exp(_K * (times[i] - 0.2)), 3))
    assert after == {5.0, 7.0}


def test_writers_the_run_may_order_either_way_are_named_as_such(tmp_path):
    """E2 is due 400 doubles after E1, with EB between: the reach of the run's
    stops chains E1's to E2's, so the note says only what is true, that the
    translation cannot know which the run executes last."""
    eps = sys.float_info.epsilon
    xml = _sbml(
        _a("E1", _at(_cn(1)), 5),
        _event("EB", _at(_cn(repr(1 + 200 * eps))), {"B": _cn(7)}),
        _a("E2", _at(_cn(repr(1 + 400 * eps))), 10),
    )
    with pytest.raises(ConversionError) as refused:
        _convert(tmp_path, xml)
    note = str(refused.value)
    assert "events 'E1' and 'E2' set 'A' to different values" in note
    assert "close enough together that the translation cannot know which the run" in note
    assert "in either order" not in note


_ABS_MINUS_1 = f"<apply><abs/>{_cn(-1)}</apply>"
_ZERO_BY_ZERO = f"<apply><divide/>{_cn(0)}{_cn(0)}</apply>"


@pytest.mark.parametrize(
    "priority",
    [
        pytest.param(_ABS_MINUS_1, id="abs(-1)"),
        pytest.param(f"<apply><floor/>{_cn(1.5)}</apply>", id="floor(1.5)"),
    ],
)
def test_a_priority_the_run_reads_as_a_number_is_ordered_by_it(tmp_path, priority):
    """Not folded here, but it reads nothing that moves: the run's evaluator
    reads 1, so E2 (priority 2) runs first and E1 sets A last."""
    xml = _sbml(
        _a("E1", _at(_cn(1)), 5, priority=priority),
        _a("E2", _at(_cn(1)), 10, priority=_cn(2)),
    )
    rep = _convert(tmp_path, xml)
    assert rep.ok, rep.lossy
    assert [v for _, target, v in _schedule(rep) if target == "A"] == [10.0, 5.0]
    _assert_runs_as_the_model(xml, rep)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"priority": _ZERO_BY_ZERO}, id="0/0"),
        pytest.param(
            {"priority": _ci("p"), "params": _param("p", 1), "extra": _initial(p=_ZERO_BY_ZERO)},
            id="a parameter an initialAssignment sets to 0/0",
        ),
    ],
)
def test_a_priority_the_run_reads_as_nan_is_refused_as_the_run_refuses_it(tmp_path, body):
    xml = _sbml(
        _a("E1", _at(_cn(1)), 5, priority=body["priority"]),
        parameters=body.get("params", ""),
        extra=body.get("extra", ""),
    )
    with pytest.raises(
        ConversionError,
        match="'E1' has a priority that is NaN, which the run refuses: an event priority must "
        "be a number",
    ):
        _convert(tmp_path, xml)
    with warnings.catch_warnings(), pytest.raises(Exception, match="priority that is NaN"):
        warnings.simplefilter("ignore")
        bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
            t_span=(0.0, 3.0), n_points=7
        )


_H_3_7 = _times(_ci("h"), _cn(3), _cn(7))  # the run reads h*21 = 2.1


def test_a_value_the_run_rounds_otherwise_is_refused(tmp_path):
    """Folded left to right here, h*3*7 is 2.1000000000000005; the run sets 2.1."""
    xml = _sbml(
        _event("E1", _at(_cn(1)), {"A": _H_3_7}),
        _event("E2", _at(_cn(1)), {"A": _cn(2.1)}),
        parameters=_param("h", 0.1),
    )
    with pytest.raises(
        ConversionError,
        match="'E1' assigns 'A' a value that the run computes as 2.1 and the translation folds "
        "to 2.1000000000000005",
    ):
        _convert(tmp_path, xml)


def test_values_a_double_apart_in_a_tie_are_refused(tmp_path):
    """The run sets 2.1 and 2.1000000000000005 in a tie, in either order."""
    xml = _sbml(
        _event("E1", _at(_cn(1)), {"A": _cn(2.1)}),
        _event("E2", _at(_cn(1)), {"A": _cn("2.1000000000000005")}),
    )
    with pytest.raises(ConversionError, match="with equal priority, and the run executes them"):
        _convert(tmp_path, xml)


def test_a_value_is_the_models_not_the_runs_storage(tmp_path):
    """The run stores an amount-valued species over its compartment's size,
    and the loader's assignment says 5/C; the value is the model's 5."""
    xml = _sbml(
        _event("E1", _at(_cn(1)), {"A": _cn(5)}),
        _event("E2", _at(_cn(2)), {"B": _cn(7)}),
        compartment='<compartment id="C" size="2" spatialDimensions="3" constant="true"/>',
    )
    amount = 'initialConcentration="1.0" hasOnlySubstanceUnits="true"'
    xml = xml.replace('initialConcentration="1.0" hasOnlySubstanceUnits="false"', amount, 1)
    assert amount in xml
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_sbml_string(xml)
    protocol, lossy = sbml_events_to_protocol(_write(tmp_path, xml), model)
    assert lossy == []
    sets = [s for s in protocol.steps if isinstance(s, StateChange)]
    assert [(s.target, s.value) for s in sets] == [("A", 5.0), ("B", 7.0)]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            _times(f"<apply><minus/>{_H_3_7}{_cn(2.1)}</apply>", _cn("1e16")),
            id="(h*3*7 - 2.1)*1e16",
        ),
        pytest.param(
            f"<apply><minus/>{_times(_cn('1e16'), _H_3_7)}{_cn('2.1e16')}</apply>",
            id="1e16*(h*3*7) - 2.1e16",
        ),
    ],
)
def test_a_value_that_cancels_otherwise_in_the_run_is_refused(tmp_path, value):
    """Folded left to right here, h*3*7 - 2.1 is 4.4e-16, scaled to 4.4 or 4.0;
    the run reads h*21 - 2.1 = 0. The two disagree, so the value is refused."""
    xml = _sbml(_event("E1", _at(_cn(1)), {"A": value}), parameters=_param("h", 0.1))
    with pytest.raises(
        ConversionError, match="'E1' assigns 'A' a value that the run computes as 0.0 and"
    ):
        _convert(tmp_path, xml)


def test_a_writer_surely_after_the_rest_sets_the_value(tmp_path):
    """E2 is due one and a half reaches after E1, past anything the strict EB
    can land on: it surely executes last, so A is 10 after it."""
    late = 1.0 + 1.5 * _instant_reach(1.0, 3.0)
    xml = _sbml(
        _a("E1", _at(_cn(1)), 5),
        _event("EB", _gt(_cn(1)), {"B": _cn(7)}),
        _a("E2", _at(_cn(repr(late))), 10),
    )
    rep = _convert(tmp_path, xml)
    assert rep.ok, rep.lossy
    assert _schedule(rep) == [(1.0, "A", 5.0), (1.0, "B", 7.0), (late, "A", 10.0)]
    _assert_runs_as_the_model(xml, rep)


def test_the_note_names_the_writer_that_keeps_the_rest_from_one_instant(tmp_path):
    """E1 and E2 are on one double, ordered by priority; the strict E3 is
    what the translation cannot place."""
    xml = _sbml(
        _a("E1", _at(_cn(1)), 5, priority=_cn(1)),
        _a("E2", _at(_cn(1)), 10, priority=_cn(2)),
        _a("E3", _gt(_cn(1)), 5),
    )
    with pytest.raises(ConversionError, match="events 'E3' and 'E2' set 'A' to different values"):
        _convert(tmp_path, xml)


_ABS = "<apply><abs/>{}</apply>"


@pytest.mark.parametrize(
    ("events", "want"),
    [
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _ABS.format(_cn(-5))})],
            [(1.0, "A", 5.0)],
            id="a value of abs(-5)",
        ),
        pytest.param(
            [_a("E1", _at(_ABS.format(_cn(-1))), 5)],
            [(1.0, "A", 5.0)],
            id="a trigger time of abs(-1)",
        ),
    ],
)
def test_a_constant_not_folded_here_is_read_from_the_run(tmp_path, events, want):
    xml = _sbml(*events)
    rep = _convert(tmp_path, xml)
    assert rep.ok, rep.lossy
    assert _schedule(rep) == want
    _assert_runs_as_the_model(xml, rep)


_DIV = "<apply><divide/>{}{}</apply>"


@pytest.mark.parametrize(
    ("threshold", "shown"),
    [
        pytest.param(_DIV.format(_cn(0), _cn(0)), "nan", id="0/0"),
        pytest.param(f"<apply><ln/>{_cn(-1)}</apply>", "nan", id="ln(-1)"),
        pytest.param(f"<apply><power/>{_cn(-2)}{_cn(0.5)}</apply>", "nan", id="(-2)^0.5"),
        pytest.param("<notanumber/>", "nan", id="notanumber"),
        pytest.param(_DIV.format(_cn(1), _cn(0)), "inf", id="1/0"),
        pytest.param(_times(_cn("1e308"), _cn(10)), "inf", id="1e308*10"),
        pytest.param(_DIV.format(_cn(-1), _cn(0)), "-inf", id="-1/0"),
    ],
)
def test_a_trigger_time_that_is_not_a_finite_number_is_refused(tmp_path, threshold, shown):
    """The run never fires `time >= NaN` or `time >= inf`; the actions must not
    dose it at the start, nor carry a time no phase can end at."""
    with pytest.raises(
        ConversionError, match=f"'E1' has a trigger time of {shown}, which is not a finite number"
    ):
        _convert(tmp_path, _sbml(_a("E1", _at(threshold), 5)))


def test_a_trigger_time_that_is_nan_beside_another_event_returns(tmp_path):
    """A NaN among the run's stops never ended the walk along them. The
    conversion runs in a daemon thread, so a hang fails the test, not the run."""
    import threading

    xml = _sbml(_a("E1", _at(_DIV.format(_cn(0), _cn(0))), 5), _a("E2", _at(_cn(1)), 7))
    outcome: list[BaseException] = []

    def convert() -> None:
        try:
            _convert(tmp_path, xml)
        except BaseException as e:  # the refusal, or anything else, to check below
            outcome.append(e)

    worker = threading.Thread(target=convert, daemon=True)
    worker.start()
    worker.join(timeout=120)
    assert not worker.is_alive(), "sbml_to_bngl did not return"
    assert outcome and isinstance(outcome[0], ConversionError), outcome
    assert "'E1' has a trigger time of nan" in str(outcome[0])


@pytest.mark.parametrize(
    ("value", "body"),
    [
        pytest.param(
            _times(
                f"<apply><minus/>{_times(_ci('p'), _cn(3), _cn(7))}{_cn(2.1)}</apply>", _cn("1e16")
            ),
            {"parameters": _param("p", 0.1, constant=False)},
            id="(p*3*7 - 2.1)*1e16",
        ),
        pytest.param(
            _times(f"<apply><minus/>{_ci('p')}{_cn(2.1)}</apply>", _cn("1e16")),
            {
                "parameters": _param("p", 0, constant=False) + _param("h", 0.1),
                "extra": _initial(p=_H_3_7),
            },
            id="(p - 2.1)*1e16, p set by an initialAssignment to h*3*7",
        ),
    ],
)
def test_a_value_on_its_own_start_value_the_run_may_round_otherwise_is_refused(
    tmp_path, value, body
):
    """The run reads p*3*7 as p*21, and holds h*3*7 as 2.1: from either, it
    sets p to 0, where the fold here gives 4.4."""
    xml = _sbml(_event("E1", _at(_cn(1)), {"p": value}), **body)
    with pytest.raises(
        ConversionError, match="'E1' assigns 'p' a value that reads its own start value"
    ):
        _convert(tmp_path, xml)


_Q_RULE = {
    "parameters": '<parameter id="q" constant="false"/>',
    "extra": _rules(q=("assignmentRule", _times(_ci("k"), _cn(3), _cn(7)))),
}
_Q_CANCELS = _times(f"<apply><minus/>{_ci('q')}{_cn(2.1)}</apply>", _cn("1e16"))


@pytest.mark.parametrize(
    "events",
    [
        pytest.param([_event("E1", _at(_cn(1)), {"A": _Q_CANCELS})], id="a value"),
        pytest.param(
            [_a("E1", _at(_plus(_times(_cn(0.3378), _Q_CANCELS), _cn(1))), 5)], id="a trigger time"
        ),
    ],
)
def test_a_name_an_assignment_rule_sets_that_the_run_computes_otherwise_is_refused(
    tmp_path, events
):
    """q := k*3*7 is held before the run as 2.1000000000000005, the fold; the
    run computes k*21 = 2.1 from the rule."""
    with pytest.raises(
        ConversionError, match="that reads 'q', which an assignment rule sets, and which the run"
    ):
        _convert(tmp_path, _sbml(*events, **_Q_RULE))


def _with_functions(xml: str, **bodies: str) -> str:
    """``xml`` with a functionDefinition ``name() = body`` for each."""
    defs = "".join(
        f'<functionDefinition id="{name}">{_M}<lambda>{body}</lambda></math></functionDefinition>'
        for name, body in bodies.items()
    )
    return xml.replace(
        '<model id="m">',
        f'<model id="m"><listOfFunctionDefinitions>{defs}</listOfFunctionDefinitions>',
        1,
    )


@pytest.mark.parametrize(
    ("events", "bodies", "note"),
    [
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": _times("<apply><ci>f</ci></apply>", _cn(3))})],
            {"f": _ci("B")},
            "'E1' assigns 'A' a non-constant value",
            id="a value through f() = B",
        ),
        pytest.param(
            [_a("E1", _at("<apply><ci>f</ci></apply>"), 5)],
            {"f": _ci("B")},
            "'E1' is state-triggered",
            id="a trigger time through f() = B",
        ),
        pytest.param(
            [_event("E1", _at(_cn(1)), {"A": "<apply><ci>f</ci></apply>"}, delay=_cn(1))],
            {"f": _TIME},
            "'E1' assigns 'A' a non-constant value",
            id="a value through f() = time",
        ),
    ],
)
def test_what_a_function_definition_reads_is_seen(tmp_path, events, bodies, note):
    with pytest.raises(ConversionError, match=note):
        _convert(tmp_path, _with_functions(_sbml(*events), **bodies))


def test_the_note_names_a_strict_writer_not_one_on_the_double(tmp_path):
    """ED's delayed execution lands on 1 exactly, with E2, and priority orders
    them; the strict E3 is what the translation cannot place."""
    xml = _sbml(
        _a("ED", _at(_cn(0.5)), 10, delay=_cn(0.5), priority=_cn(1)),
        _a("E2", _at(_cn(1)), 5, priority=_cn(2)),
        _a("E3", _gt(_cn(1)), 5),
    )
    with pytest.raises(ConversionError, match="events 'E3' and 'ED' set 'A' to different values"):
        _convert(tmp_path, xml)


def test_a_value_that_reads_rate_of_is_refused_as_such(tmp_path):
    rate_of = (
        '<apply><csymbol encoding="text" '
        'definitionURL="http://www.sbml.org/sbml/symbols/rateOf">rateOf</csymbol>'
        f"{_ci('k')}</apply>"
    )
    xml = _sbml(_event("E1", _at(_cn(1)), {"A": _plus(rate_of, _cn(5))}))
    with pytest.raises(
        ConversionError, match="'E1' assigns 'A' a value that reads rateOf, which the translation"
    ):
        _convert(tmp_path, xml)
