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
    t = 0.0
    for step in report.protocol.steps if report.protocol is not None else ():
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
