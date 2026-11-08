"""RoadRunner is run through the instants its event triggers switch at.

RoadRunner fires a ``time == c`` trigger, or an open window such as
``(time > a) && (time < b)``, only when one of its output times lands on the
instant or inside the window. bngsim fires the event wherever the instant falls
(#1015), so on a job grid with no sample there the sweep read RoadRunner's miss
as a bngsim divergence; 10 BioModels had to be recorded as known artifacts for it
(#1021). ``_rr_common.event_instants`` finds the instants from the SBML, and
``rr_ode(..., through_event_instants=True)`` integrates RoadRunner through them
while reporting the job's own grid.

The derivation tests are libsbml only. The RoadRunner tests skip where
roadrunner is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libsbml")

import _rr_common as rc  # noqa: E402


def _sbml(trigger: str, params: str = "", extra: str = "") -> str:
    """An L3V2 model: species S at 0 with no reactions, constant parameters
    from ``params``, one event on ``trigger`` (MathML body) that sets S = 1."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
  </listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>{params}</listOfParameters>
  {extra}
  <listOfEvents>
   <event id="E" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true">
     <math xmlns="http://www.w3.org/1998/Math/MathML">{trigger}</math>
    </trigger>
    <listOfEventAssignments>
     <eventAssignment variable="S">
      <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
     </eventAssignment>
    </listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>
"""


T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'


def _cn(v) -> str:
    return f"<cn>{v}</cn>"


def _rel(op: str, a: str, b: str) -> str:
    return f"<apply><{op}/>{a}{b}</apply>"


def _param(pid: str, value: float, constant: bool = True) -> str:
    return f'<parameter id="{pid}" value="{value}" constant="{str(constant).lower()}"/>'


def _on_instants(instants: list[float]) -> list[float]:
    """The instants, with a 'hair after' one rounded back onto its boundary."""
    return sorted({round(t, 6) for t in instants})


def test_an_equality_on_time_gives_its_instant():
    assert rc.event_instants(_sbml(_rel("eq", T, _cn(20))), 0.0, 39.0) == [20.0]


def test_an_inequality_on_time_gives_its_instant_and_a_sample_just_after():
    """`time != 5` is false only at 5; it turns true again just after it."""
    got = rc.event_instants(_sbml(_rel("neq", T, _cn(5))), 0.0, 10.0)
    assert len(got) == 2 and got[0] == 5.0 and 5.0 < got[1] < 5.0 + 1e-6


@pytest.mark.parametrize("op", ["gt", "geq", "lt", "leq"])
def test_an_inequality_gives_a_sample_just_after_its_boundary_never_on_it(op):
    """Just after: inside any window the comparison opens. Never on it: a sample
    on the instant of a jump makes RoadRunner's integrator fail at the harness
    tolerances (MODEL0911120000's `time >= 16.16` pulse end: CV_CONV_FAILURE)."""
    (got,) = rc.event_instants(_sbml(_rel(op, T, _cn(16))), 0.0, 39.0)
    assert 16.0 < got < 16.0 + 1e-6


def test_the_constant_may_be_a_parameter_or_an_expression_of_them():
    trig = _rel("eq", T, "<apply><plus/><ci>a</ci><ci>b</ci></apply>")
    sbml = _sbml(trig, _param("a", 5) + _param("b", 360))
    assert _on_instants(rc.event_instants(sbml, 0.0, 700.0)) == [365.0]
    # A caller's value (RoadRunner's, after the initial assignments) wins.
    got = rc.event_instants(sbml, 0.0, 700.0, value_of={"a": 1.0, "b": 2.0}.get)
    assert _on_instants(got) == [3.0]


def test_an_open_window_gets_a_sample_just_inside_it():
    trig = f"<apply><and/>{_rel('gt', T, _cn(179))}{_rel('lt', T, _cn(179.5))}</apply>"
    got = rc.event_instants(_sbml(trig), 0.0, 1400.0)
    assert any(179.0 < t < 179.5 for t in got)
    assert _on_instants(got) == [179.0, 179.5]


def test_a_closed_window_gets_a_sample_inside_it_too():
    """RoadRunner skips a window narrower than its step whatever its edges."""
    trig = f"<apply><and/>{_rel('geq', T, _cn(16))}{_rel('leq', T, _cn(17))}</apply>"
    got = rc.event_instants(_sbml(trig), 0.0, 40.0)
    assert any(16.0 < t < 17.0 for t in got)


def test_a_quotient_is_affine():
    trig = _rel("eq", f"<apply><divide/>{T}{_cn(3)}</apply>", _cn(2))
    assert _on_instants(rc.event_instants(_sbml(trig), 0.0, 10.0)) == [6.0]


def test_a_compartment_size_is_a_constant():
    """Valued from the SBML when no value_of is given (it used to raise there)."""
    sbml = _sbml(_rel("eq", T, "<ci>C</ci>")).replace('size="1"', 'size="5"')
    assert rc.event_instants(sbml, 0.0, 10.0) == [5.0]


def test_an_affine_argument_is_solved():
    trig = _rel("eq", f"<apply><times/>{_cn(2)}{T}</apply>", _cn(5))
    assert _on_instants(rc.event_instants(_sbml(trig), 0.0, 10.0)) == [2.5]


def test_rem_of_time_equal_to_a_level_gives_the_level_every_period():
    rem = f"<apply><rem/>{T}{_cn(28)}</apply>"
    trig = f"<apply><and/>{_rel('eq', rem, _cn(27))}{_rel('gt', T, _cn(54))}</apply>"
    got = _on_instants(rc.event_instants(_sbml(trig), 0.0, 120.0))
    assert got == [27.0, 54.0, 55.0, 83.0, 111.0]  # 54: just after the window's edge


def test_rem_of_time_past_a_level_also_gives_where_it_wraps():
    """`(time % 24) > 12` turns true just after 12 + 24k and false at each wrap."""
    rem = f"<apply><rem/>{T}{_cn(24)}</apply>"
    got = rc.event_instants(_sbml(_rel("gt", rem, _cn(12))), 0.0, 50.0)
    assert _on_instants(got) == [12.0, 24.0, 36.0, 48.0]
    assert all(t > round(t, 6) for t in got)  # each a hair after its boundary


def _copasi_mod(x: str, period: str) -> str:
    """``x % P`` as COPASI exports it: piecewise(x - P*ceil(x/P), xor(x < 0, P < 0),
    x - P*floor(x/P)). libsbml prints it back as ``%``; the AST is the expansion."""

    def branch(fn: str) -> str:
        return (
            f"<apply><minus/>{x}<apply><times/>{period}"
            f"<apply><{fn}/><apply><divide/>{x}{period}</apply></apply></apply></apply>"
        )

    cond = f"<apply><xor/>{_rel('lt', x, _cn(0))}{_rel('lt', period, _cn(0))}</apply>"
    return f"<piecewise><piece>{branch('ceiling')}{cond}</piece><otherwise>{branch('floor')}</otherwise></piecewise>"


def test_copasis_expansion_of_modulo_is_read_as_one():
    trig = _rel("eq", _copasi_mod(T, _cn(12)), _cn(0))
    got = _on_instants(rc.event_instants(_sbml(trig), 0.0, 50.0))
    assert got == [12.0, 24.0, 36.0, 48.0]


def test_a_shifted_dosing_cycle_of_parameters():
    """``((time - Dose) % Cycle) == 0`` (BIOMD0000000234's shape)."""
    shifted = f"<apply><minus/>{T}<ci>Dose</ci></apply>"
    trig = _rel("eq", _copasi_mod(shifted, "<ci>Cycle</ci>"), _cn(0))
    sbml = _sbml(trig, _param("Dose", 1) + _param("Cycle", 3))
    assert _on_instants(rc.event_instants(sbml, 0.0, 10.0)) == [1.0, 4.0, 7.0, 10.0]


@pytest.mark.parametrize(
    "trig, params",
    [
        (_rel("gt", "<ci>S</ci>", _cn(5)), ""),  # on state: RoadRunner roots it itself
        (_rel("eq", T, "<ci>S</ci>"), ""),  # against a species
        (_rel("eq", f"<apply><times/>{T}{T}</apply>", _cn(4)), ""),  # not affine
        (_rel("eq", T, "<ci>p</ci>"), _param("p", 3, constant=False)),  # p is assigned
    ],
)
def test_comparisons_it_cannot_place_give_no_instants(trig, params):
    extra = ""
    if 'constant="false"' in params:
        extra = (
            '<listOfRules><assignmentRule variable="p"><math '
            'xmlns="http://www.w3.org/1998/Math/MathML"><cn>3</cn></math>'
            "</assignmentRule></listOfRules>"
        )
    assert rc.event_instants(_sbml(trig, params, extra), 0.0, 10.0) == []


def test_a_parameter_an_algebraic_rule_may_determine_is_not_a_constant():
    algebraic = (
        '<listOfRules><algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<apply><minus/><ci>p</ci><cn>3</cn></apply></math></algebraicRule></listOfRules>"
    )
    sbml = _sbml(_rel("eq", T, "<ci>p</ci>"), _param("p", 3, constant=False), algebraic)
    assert rc.event_instants(sbml, 0.0, 10.0) == []


@pytest.mark.parametrize("level", [-1, 3, 4])
def test_a_rem_level_it_never_takes_gives_no_level_instants(level):
    """rem(time, 3) for time >= 0 is in [0, 3): it never equals -1, 3 or 4."""
    rem = f"<apply><rem/>{T}{_cn(3)}</apply>"
    assert rc.event_instants(_sbml(_rel("eq", rem, _cn(level))), 0.0, 10.0) == []


def test_only_instants_inside_the_run_are_kept():
    trig = f"<apply><or/>{_rel('eq', T, _cn(0))}{_rel('eq', T, _cn(5))}{_rel('eq', T, _cn(11))}</apply>"
    assert _on_instants(rc.event_instants(_sbml(trig), 0.0, 10.0)) == [5.0]


def test_a_modulo_too_fine_for_the_run_is_left_alone():
    rem = f"<apply><rem/>{T}{_cn(1e-6)}</apply>"
    assert rc.event_instants(_sbml(_rel("eq", rem, _cn(0))), 0.0, 1000.0) == []


def test_a_model_with_no_events_has_no_instants():
    sbml = (
        _sbml(_rel("eq", T, _cn(1)))
        .replace("<listOfEvents>", "<!--")
        .replace("</listOfEvents>", "-->")
    )
    assert rc.event_instants(sbml, 0.0, 10.0) == []


# ── RoadRunner ──────────────────────────────────────────────────────────────

_GRID = (0.0, 1.0, 11)  # samples every 0.1; none at 0.33


def _s_column(t, v, names):
    return v[:, names.index("S")] if "S" in names else v[:, names.index("[S]")]


def test_roadrunner_skips_an_event_between_its_samples():
    """The premise: if this starts failing, RoadRunner has learned to fire the
    event on its own and the instants are no longer needed."""
    pytest.importorskip("roadrunner")
    t, v, names, _ = rc.rr_ode(_sbml(_rel("eq", T, _cn(0.33))), *_GRID, 1e-8, 1e-12)
    assert np.all(_s_column(t, v, names) == 0.0)


def test_through_the_instants_it_fires_and_reports_the_job_grid():
    pytest.importorskip("roadrunner")
    sbml = _sbml(_rel("eq", T, _cn(0.33)))
    t, v, names, timing = rc.rr_ode(sbml, *_GRID, 1e-8, 1e-12, through_event_instants=True)
    np.testing.assert_allclose(t, np.linspace(*_GRID), rtol=0, atol=1e-12)
    s = _s_column(t, v, names)
    assert np.all(s[t < 0.33] == 0.0) and np.all(s[t > 0.33] == 1.0)
    assert timing["event_instants"] == 1


def test_the_parameter_is_valued_after_its_initial_assignment():
    """p is 0.5 in the SBML and 0.33 after its initialAssignment; RoadRunner's
    value is the one the event fires at."""
    pytest.importorskip("roadrunner")
    ia = (
        '<listOfInitialAssignments><initialAssignment symbol="p"><math '
        'xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.33</cn></math>'
        "</initialAssignment></listOfInitialAssignments>"
    )
    sbml = _sbml(_rel("eq", T, "<ci>p</ci>"), _param("p", 0.5), ia)
    t, v, names, _ = rc.rr_ode(sbml, *_GRID, 1e-8, 1e-12, through_event_instants=True)
    s = _s_column(t, v, names)
    assert s[np.isclose(t, 0.3)][0] == 0.0 and s[np.isclose(t, 0.4)][0] == 1.0


def test_a_narrow_closed_window_fires_through_the_instants():
    pytest.importorskip("roadrunner")
    trig = f"<apply><and/>{_rel('geq', T, _cn(16))}{_rel('leq', T, _cn(17))}</apply>"
    sbml = _sbml(trig)
    grid = (0.0, 40.0, 5)  # samples every 10: none in [16, 17]
    plain = rc.rr_ode(sbml, *grid, 1e-9, 1e-12)
    through = rc.rr_ode(sbml, *grid, 1e-9, 1e-12, through_event_instants=True)
    assert np.all(_s_column(*plain[:3]) == 0.0)  # the premise: RoadRunner skips it
    assert _s_column(*through[:3])[-1] == 1.0


def test_a_model_without_time_triggers_runs_as_before():
    pytest.importorskip("roadrunner")
    sbml = _sbml(_rel("gt", "<ci>S</ci>", _cn(5)))
    plain = rc.rr_ode(sbml, *_GRID, 1e-8, 1e-12)
    through = rc.rr_ode(sbml, *_GRID, 1e-8, 1e-12, through_event_instants=True)
    np.testing.assert_array_equal(plain[0], through[0])
    np.testing.assert_array_equal(plain[1], through[1])
    assert through[3]["event_instants"] == 0


def _ulps_from(x: float, k: int) -> float:
    for _ in range(abs(k)):
        x = float(np.nextafter(x, np.inf if k > 0 else -np.inf))
    return x


# Grids and the sample each case is near. 0.30000000000000004 and 1.0 sit at
# different places between powers of two, so CVODE's refusal (closer than
# 2·eps·max(|t0|, |tout|)) reaches 2 ulps before the first and 3 before the
# second; 1e7's ulps are past the 1e-9 a returned time used to be held to.
_NEAR = {
    "0.3": ((0.0, 1.0, 11), 3),
    "1.0": ((0.0, 1.0, 11), 10),
    "1e7": ((0.0, 2e7, 3), 1),
}


# The ulps from the sample each case is run at: 1 to 5 before, on it, after it,
# and 1e-12 before 0.3, where main ran.
_KS = {
    "0.3": (-1, -2, -3, -4, -5, 0, 1, 5, "1e-12"),
    "1.0": (-1, -2, -3, -4, 0, 1),
    "1e7": (-1, -2, -3, -4, 0, 1),
}


@pytest.mark.parametrize("op", ["eq", "neq"])
@pytest.mark.parametrize("near, k", [(near, k) for near in _NEAR for k in _KS[near]])
def test_a_comparison_ulps_from_a_sample_reads_as_bngsim_reports_it(op, near, k):
    """`time == p` fires at p, so S is 1 on every sample at or after p; `time
    != p`, false at p, fires just after it, so S is 1 on every sample after p.
    bngsim reports both so, and the job grid comes back exactly.

    Where p is so shortly before the sample that CVODE will not start the step
    to it after the event (1 and 2 ulps before 0.3, 1 to 3 before 1.0, 1 and 2
    before 1e7), RoadRunner raised CV_TOO_CLOSE (#1558); there the sample now
    runs a few ulps later. Every other case ran before and reads the same."""
    pytest.importorskip("roadrunner")
    grid_args, i = _NEAR[near]
    grid = np.linspace(*grid_args)
    p = float(grid[i]) - 1e-12 if k == "1e-12" else _ulps_from(float(grid[i]), k)
    sbml = _sbml(_rel(op, T, f"<cn>{p!r}</cn>"))
    if op == "neq":  # starts true, so it fires just after p, not at the start
        sbml = sbml.replace('initialValue="false"', 'initialValue="true"')
    t, v, names, _ = rc.rr_ode(sbml, *grid_args, 1e-8, 1e-12, through_event_instants=True)
    np.testing.assert_array_equal(t, grid)
    after = grid >= p if op == "eq" else grid > p
    np.testing.assert_array_equal(_s_column(t, v, names), after.astype(float))


def test_only_a_sample_cvode_would_not_step_to_from_an_instant_moves():
    """Every other instant gives main's union of grid and instants. One CVODE
    would not step from stays, and the sample moves on a few ulps."""
    grid = np.linspace(0.0, 1.0, 11)
    g3 = float(grid[3])
    for instants in ([0.33], [g3], [_ulps_from(g3, 1)], [_ulps_from(g3, -3)], [g3 - 1e-12]):
        times, rows = rc.rr_output_times(grid, instants)
        assert times == list(np.union1d(grid, instants)), instants
        assert [times[r] for r in rows] == list(grid)
    for g, k in ((g3, -1), (g3, -2), (1.0, -3)):
        p = _ulps_from(g, k)
        times, rows = rc.rr_output_times(grid, [p])
        moved = times[rows[list(grid).index(g)]]
        assert p in times and g not in times and len(times) == len(grid) + 1
        assert g < moved <= _ulps_from(g, 4)
        assert [times[r] for r in rows if times[r] != moved] == [x for x in grid if x != g]


def test_a_sample_that_cannot_move_past_no_instant_stays():
    """An instant on the sample, or one so shortly after it that the move
    would reach it: main's union, so the run fails as before."""
    grid = np.linspace(0.0, 1.0, 11)
    g3 = float(grid[3])
    p = _ulps_from(g3, -1)
    for instants in ([p, g3], [p, _ulps_from(g3, 1)]):
        times, rows = rc.rr_output_times(grid, instants)
        assert times == list(np.union1d(grid, instants)), instants
        assert [times[r] for r in rows] == list(grid)


def _either(*instants: float) -> str:
    return (
        "<apply><or/>" + "".join(_rel("eq", T, f"<cn>{c!r}</cn>") for c in instants) + "</apply>"
    )


@pytest.mark.parametrize(
    "instants, grid_args",
    [
        ((-1, 0), (0.0, 1.0, 11)),
        ((-2, -1), (0.0, 1.0, 11)),
        ((-1, 1), (0.0, 1.0, 11)),
        ((0, 1), (0.0, 1.0, 11)),
    ],
)
def test_two_instants_closer_than_cvode_steps_still_fail(instants, grid_args):
    """Two event times, ulps from the sample at 0.30000000000000004, that must
    each be an output time and are closer together than CVODE will step after
    the first fires. No choice of output times runs that, so RoadRunner raises,
    a reference failure rather than a row missing an event."""
    pytest.importorskip("roadrunner")
    g = float(np.linspace(*grid_args)[3])
    sbml = _sbml(_either(*(_ulps_from(g, k) for k in instants)))
    with pytest.raises(RuntimeError, match="CV_TOO_CLOSE"):
        rc.rr_ode(sbml, *grid_args, 1e-8, 1e-12, through_event_instants=True)


def test_an_instant_ulps_after_a_nonzero_start_still_fails():
    """CVODE will not start its first step to an output time that close to
    the start, and neither the start nor the instant can move."""
    pytest.importorskip("roadrunner")
    sbml = _sbml(_rel("eq", T, f"<cn>{_ulps_from(100.0, 1)!r}</cn>"))
    with pytest.raises(RuntimeError, match="CV_TOO_CLOSE"):
        rc.rr_ode(sbml, 100.0, 101.0, 11, 1e-8, 1e-12, through_event_instants=True)
