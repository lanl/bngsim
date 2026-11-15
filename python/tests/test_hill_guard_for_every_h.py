"""Issue #1636: a Hill exponent moved into (0, 1) after load is read as 0 at S = 0.

#1629 (issue #1497) wrote the `.net` loader's Hill function
``Vmax*S^(h-1)/(Kh^h + S^h)`` as ``if((S > 0) or (h <= 0) or (h >= 1), body, 0)``,
which the reaction multiplies by S, but only where h was in (0, 1) at load: a
root on ``S > 0`` costs every step it is watched, and 300 Hill reactions with
h >= 1 ran 10x slower with it. So a model whose h was moved into (0, 1) after it
was loaded evaluated ``inf*0`` at S = 0, and every run that reached S = 0 failed.

Every Hill law now takes the guard, and a plain run folds a condition its
parameter values decide (``(h <= 0) or (h >= 1)`` true at h = 2) to a constant
before choosing its roots, so a model at h >= 1 roots nothing for it.

The model is the issue's: S fed at kin = 1 from 0 and drained by Hill(1, 2, h).
BNG2.pl 2.9.3 with run_network 3.0 gives S(5) = 2.796028127 at h = 0.5 and
2.92167346 at h = 2 (the issue's table); a stiff solve of the same ODE agrees.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil

import bngsim
import numpy as np
import pytest

# The loader says it rewrote a legacy Hill token, on every load here.
pytestmark = pytest.mark.filterwarnings("ignore:Legacy/deprecated BioNetGen .net rate law token")

S_AT_HALF, S_AT_TWO = 2.796028127, 2.92167346
TOL = 2e-8  # the reference's own nine digits, and rtol 1e-10 here

NET = """begin parameters
 1 Vmax 1
 2 Kh 2
 3 h {h}
 4 kin 1
end parameters
begin species
 1 S() 0
 2 P() 0
end species
begin reactions
 1 1 2 Hill Vmax Kh h
 2 0 1 kin
end reactions
begin groups
 1 S_obs 1
 2 P_obs 2
end groups
"""
RUN = {"t_span": (0.0, 5.0), "n_points": 2, "rtol": 1e-10, "atol": 1e-12}


def _model(tmp_path, h=2.0):
    path = tmp_path / "hill.net"
    path.write_text(NET.format(h=h))
    return bngsim.Model.from_net(str(path))


def _s_end(result) -> float:
    return float(np.asarray(result.species)[-1, list(result.species_names).index("S()")])


def test_the_function_is_guarded_whatever_h_is_at_load(tmp_path):
    for h in (2.0, 1.0, 0.5, 0.0, -1.0):
        (body,) = _model(tmp_path, h)._core.function_expressions
        assert body.startswith("if((") and "or (h <= 0) or (h >= 1)" in body, (h, body)


def test_h_set_into_the_open_interval_after_load(tmp_path):
    # On main: CV_FIRST_RHSFUNC_ERR, `inf*0` at S = 0.
    model = _model(tmp_path, 2.0)
    model.set_param("h", 0.5)
    assert abs(_s_end(bngsim.Simulator(model, "ode").run(**RUN)) - S_AT_HALF) < TOL


def test_a_scan_and_a_batch_over_h(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path, 2.0), "ode")
    scanned = [_s_end(r) for r in sim.parameter_scan("h", [0.5, 2.0], **RUN)]
    rows = [_s_end(r) for r in sim.run_batch(params=[{"h": 0.5}, {"h": 2.0}], **RUN)]
    for got in (scanned, rows):
        assert abs(got[0] - S_AT_HALF) < TOL and abs(got[1] - S_AT_TWO) < TOL, got


def test_a_run_at_h_2_then_one_at_one_half_on_the_same_simulator(tmp_path):
    # The roots kept for h = 2 must not be reused at h = 0.5.
    model = _model(tmp_path, 2.0)
    sim = bngsim.Simulator(model, "ode")
    assert abs(_s_end(sim.run(**RUN)) - S_AT_TWO) < TOL
    model.reset()
    model.set_param("h", 0.5)
    assert abs(_s_end(sim.run(**RUN)) - S_AT_HALF) < TOL
    model.reset()
    model.set_param("h", 2.0)
    assert abs(_s_end(sim.run(**RUN)) - S_AT_TWO) < TOL


def test_loaded_at_one_half_it_runs_as_before(tmp_path):
    assert abs(_s_end(bngsim.Simulator(_model(tmp_path, 0.5), "ode").run(**RUN)) - S_AT_HALF) < TOL


def _roots_handed_to_the_run(model, sim=None) -> list[str]:
    seen: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    log = logging.getLogger("bngsim")
    old = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        # At h <= 0 the law itself is infinite at S = 0 (run_network does not
        # read it as 0 there either), so the run from S = 0 fails at its first
        # step; the roots are handed to it before that.
        with contextlib.suppress(bngsim.SimulationError):
            (sim or bngsim.Simulator(model, "ode")).run(**RUN)
    finally:
        log.removeHandler(handler)
        log.setLevel(old)
    return [m for m in seen if m.startswith("State-switch roots for a plain run")]


@pytest.mark.parametrize(("h", "rooted"), [(2.0, False), (1.0, False), (-1.0, False), (0.5, True)])
def test_the_guard_is_rooted_only_where_h_is_in_the_open_interval(tmp_path, h, rooted):
    # The route, not only the answer: at h >= 1 or h <= 0 the guard is true
    # whatever S is, and the run is handed no root for it.
    assert bool(_roots_handed_to_the_run(_model(tmp_path, h))) is rooted


def test_the_root_comes_back_when_h_moves_into_the_interval(tmp_path):
    # The roots kept for one set of decisions are not those of another: the
    # same Simulator at h = 2, then 0.5, then 2.
    model = _model(tmp_path, 2.0)
    sim = bngsim.Simulator(model, "ode")
    assert not _roots_handed_to_the_run(model, sim)
    model.reset()
    model.set_param("h", 0.5)
    assert _roots_handed_to_the_run(model, sim)
    model.reset()
    model.set_param("h", 2.0)
    assert not _roots_handed_to_the_run(model, sim)


def test_a_sensitivity_run_is_not_folded(tmp_path):
    # Not this change's business: its roots come by another route, which reads
    # no parameter value. dS/dVmax at t = 5 at h = 2, against a difference.
    model = _model(tmp_path, 2.0)
    sens = bngsim.Simulator(model, "ode", sensitivity_params=["Vmax"]).run(**RUN)
    column = float(np.asarray(sens.sensitivities)[-1, 0, 0])
    step = 1e-6
    up, down = _model(tmp_path, 2.0), _model(tmp_path, 2.0)
    up.set_param("Vmax", 1 + step)
    down.set_param("Vmax", 1 - step)
    fd = (
        _s_end(bngsim.Simulator(up, "ode").run(**RUN))
        - _s_end(bngsim.Simulator(down, "ode").run(**RUN))
    ) / (2 * step)
    assert abs(column - fd) < 1e-6 * max(1.0, abs(fd))


M = "http://www.w3.org/1998/Math/MathML"


def _sbml(event: bool) -> str:
    # S decays at k*S while (S > 1) or (p >= 1), the condition's parameter part
    # true at p = 2, beside an event that never fires (time >= 1e9). It assigns
    # S: an event that assigned p would make p part of the state, which no
    # fold reads as a constant; one that assigns anything is enough to keep
    # every root.
    ev = (
        '<listOfEvents><event id="E" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math xmlns="{M}"><apply><geq/>'
        '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
        "<cn>1e9</cn></apply></math></trigger><listOfEventAssignments>"
        f'<eventAssignment variable="S"><math xmlns="{M}"><cn>0</cn></math></eventAssignment>'
        "</listOfEventAssignments></event></listOfEvents>"
        if event
        else ""
    )
    law = (
        f'<math xmlns="{M}"><apply><piecewise><piece><apply><times/><ci>k</ci><ci>S</ci></apply>'
        "<apply><or/><apply><gt/><ci>S</ci><cn>1</cn></apply>"
        "<apply><geq/><ci>p</ci><cn>1</cn></apply></apply></piece><otherwise><cn>0</cn></otherwise>"
        "</piecewise></apply></math>"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"><model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies><species id="S" compartment="C" initialConcentration="3"
 hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="k" value="1" constant="true"/>
<parameter id="p" value="2" constant="true"/></listOfParameters>
<listOfReactions><reaction id="R" reversible="false"><listOfReactants>
<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
<kineticLaw>{law}</kineticLaw></reaction></listOfReactions>
{ev}</model></sbml>"""


def test_a_model_with_events_keeps_its_roots():
    # The condition is decided at p = 2 either way; with an event, which may
    # set p during the run, nothing is folded. The answer is e^-t both ways.
    plain = bngsim.Model.from_sbml_string(_sbml(event=False))
    with_event = bngsim.Model.from_sbml_string(_sbml(event=True))
    folded = plain._state_switch_roots_at_values(plain.time_discontinuity_conditions())
    assert folded is not None and folded[0] == ()
    assert (
        with_event._state_switch_roots_at_values(with_event.time_discontinuity_conditions())
        is None
    )
    for model in (plain, with_event):
        result = bngsim.Simulator(model, "ode").run((0.0, 5.0), 6, rtol=1e-10, atol=1e-12)
        s = np.asarray(result.species)[:, 0]
        assert np.abs(s - 3 * np.exp(-np.asarray(result.time))).max() < 1e-7


@pytest.mark.skipif(
    not (os.environ.get("BNGPATH") and shutil.which("perl")), reason="BNG2.pl not available"
)
def test_the_issues_bngl_setparameter_after_generate_network(tmp_path):
    bngl = tmp_path / "hill.bngl"
    bngl.write_text(
        "begin model\nbegin parameters\n  Vmax 1\n  Kh 2\n  h 2\n  kin 1\nend parameters\n"
        "begin molecule types\n  S()\n  P()\nend molecule types\n"
        "begin seed species\n  S() 0\n  P() 0\nend seed species\n"
        "begin observables\n  Molecules S_obs S()\n  Molecules P_obs P()\nend observables\n"
        "begin reaction rules\n  S() -> P() Hill(Vmax,Kh,h)\n  0 -> S() kin\nend reaction rules\n"
        'end model\ngenerate_network({overwrite=>1})\nsetParameter("h",0.5)\n'
        'simulate({method=>"ode",t_end=>5,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n'
    )
    result = bngsim.run_bngl(str(bngl)).runs[-1].result
    assert abs(_s_end(result) - S_AT_HALF) < TOL


# ── The fold itself ─────────────────────────────────────────────────────────

CONSTANTS = frozenset({"h", "k", "on"})


def fold_parameter_decided(text, constants, value_of):
    # Imported here, so the run tests above still run where it is absent.
    from bngsim._switch_sensitivity import fold_parameter_decided as fold

    return fold(text, constants, value_of)


def _fold(text, value_of):
    return fold_parameter_decided(text, CONSTANTS, value_of).replace(" ", "")


def _values(**at):
    def value_of(side: str):
        side = side.strip()
        try:
            return float(side)
        except ValueError:
            return at.get(side)

    return value_of


def test_an_or_with_a_true_parameter_part_folds_to_1():
    text = "if((S > 0) or (h <= 0) or (h >= 1), body, 0)"
    assert _fold(text, _values(h=2.0)) == "if(1,body,0)"
    assert _fold(text, _values(h=0.5)) == text.replace(" ", "")


def test_an_and_with_a_false_parameter_part_folds_to_0():
    text = "if((A > 1) and (on >= 1), k, 0)"
    assert _fold(text, _values(on=0.0)) == "if(0,k,0)"
    assert "A > 1" in fold_parameter_decided(text, CONSTANTS, _values(on=1.0))


def test_a_condition_with_no_parameter_part_is_left_alone():
    text = "if((A > 1) or (B < 2), k, 0)"
    assert _fold(text, _values()) == text.replace(" ", "")


@pytest.mark.parametrize(
    "part",
    ["(time() >= 1)", "(time >= 1)", "(S >= 1)", "(h >= unknown)"],
    ids=["the clock", "the clock, bare", "a species", "a name it cannot read"],
)
def test_a_part_that_is_not_over_constants_decides_nothing(part):
    # Every name here reads 5, the clock and the species included, so only the
    # rule that a part reads constants alone keeps these from folding.
    text = f"if((A > 1) or {part}, k, 0)"
    readable = _values(h=2.0, **{"time()": 5.0, "time": 5.0, "S": 5.0, "unknown": 1.0})
    assert "A > 1" in fold_parameter_decided(text, CONSTANTS, readable)


def test_nested_conditions_are_folded_inside_out():
    text = "if((A > 1) or (h >= 1), if((B > 1) and (h <= 0), 1, 2), 3)"
    assert _fold(text, _values(h=2.0)) == "if(1,if(0,1,2),3)"


def test_a_side_that_cannot_be_read_leaves_the_condition_alone():
    text = "if((S > 0) or (h >= 1), body, 0)"
    assert "S > 0" in fold_parameter_decided(text, CONSTANTS, lambda side: None)
