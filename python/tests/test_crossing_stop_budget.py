"""A rate law that switches more often than a run has stops for is not
stepped over (issue #890).

A clock schedule gets a crossing stop on each of its edges, and the number of
edges is the run's length over the period, not a property of the model. Past
8,192 of them none was placed: the run went on with a warning in the log and
stepped over the pulses, X = 0.1 where the answer is 6000.

The budget is now 1,000,000 stops for the run, over every condition. A
condition past it has none of its stops placed, and:

* where it is a repeating schedule and nothing else in the run switches, the
  step is bounded at half the shorter stretch of its period, which no step can
  then span;
* otherwise a ``max_step`` the caller passed stands for the stops, and without
  one the run is refused, by name;
* a stochastic run, which has no step bound to fall back on, is refused.

The bound is not taken from the one condition's spacing where anything else
switches. A rate law that goes off on one schedule and on again at another's
stop is the same at both ends of a step across the two: bounded at half of a
0.2-wide pulse, beside a second schedule that was stopped at, the run gained
0.05 wherever the two met within 0.05, and returned 81 for 78.

Oracle: the model is ``0 -> X`` at a unit pulse, so X(T) is the on-time,
``w`` a period. The budget is lowered for the tests of what happens past it,
which would otherwise each integrate a million pulses.
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest
from bngsim import _switch_sensitivity as sw

_NET = """begin parameters
    1 P {P}
    2 w {w}
end parameters
begin species
    1 X() 0
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""
_SCHEDULE = "if(time() - P*floor(time()/P) < w, 1, 0)"
_FLOORS = "floor(time()/P) - floor((time()-w)/P)"


def _model(tmp_path, dose=_SCHEDULE, P=1.0, w=0.1):
    path = tmp_path / "m.net"
    path.write_text(_NET.format(P=P, w=w, dose=dose), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _x_end(model, t_end, method="ode", **kw):
    r = bngsim.Simulator(model, method=method).run(t_span=(0.0, t_end), n_points=3, **kw)
    return float(np.asarray(r.species)[-1, 0])


@pytest.fixture
def budget(monkeypatch):
    """Lower the run's budget of stops, with the resolvers' memo emptied on
    each side: an answer is remembered against the budget it was made under."""

    def set_to(n):
        sw._SCHEDULE_CACHE.clear()
        monkeypatch.setattr(sw, "_RUN_STOP_BUDGET", n)

    sw._SCHEDULE_CACHE.clear()
    yield set_to
    sw._SCHEDULE_CACHE.clear()


# ── Inside the budget: every edge is a stop ─────────────────────────────────
@pytest.mark.parametrize(
    "P, w, t_end, exact",
    [(1.0, 0.5, 6000.0, 3000.0), (1.0, 0.1, 6000.0, 600.0), (0.7, 0.05, 5000.0, 357.15)],
    ids=["12000_edges", "12000_narrow", "14285_edges"],
)
def test_the_issues_schedules_are_integrated(tmp_path, P, w, t_end, exact):
    """The issue's table. 0.7 goes into 5000 a little over 7142 times, and the
    last pulse is whole: 7143 pulses of 0.05."""
    assert _x_end(_model(tmp_path, P=P, w=w), t_end) == pytest.approx(exact, rel=1e-4)


def test_every_edge_of_a_long_schedule_is_a_stop(tmp_path):
    model = _model(tmp_path)
    conds = model.time_discontinuity_conditions()
    found, _, _ = sw.crossings_with_periodic(model._core, 0.0, 30000.0, conds)
    assert len(found) == 60000
    assert [s.time for s in found[:4]] == pytest.approx([0.1, 1.0, 1.1, 2.0])


def test_the_enumeration_is_the_one_the_sensitivity_path_lists(tmp_path):
    """`_schedule_edge_times` is `_schedule_edges` without the partials, and
    has to give the same doubles in the same order."""
    terms = sw.ScheduleTerms(0.7, 0.3, 0.05, {}, {}, {}, True)
    assert sw._schedule_edge_times(terms, 2.0, 500.0) == [
        value for value, _ in sw._schedule_edges(terms, 2.0, 500.0, 10**9)
    ]


# ── Past the budget ─────────────────────────────────────────────────────────
def test_a_schedule_past_the_budget_bounds_the_step(tmp_path, budget, caplog):
    budget(1000)
    model = _model(tmp_path)
    with caplog.at_level(logging.INFO, logger="bngsim"):
        assert _x_end(model, 2000.0) == pytest.approx(200.0, rel=1e-4)
    assert any("the step is bounded at 0.05" in r.getMessage() for r in caplog.records)


def test_the_bound_is_half_the_shorter_stretch(tmp_path, budget):
    """On for 0.9 of every period: the stretch that can be stepped over is the
    0.1 it is off for."""
    budget(1000)
    over: list = []
    model = _model(tmp_path, w=0.9)
    conds = model.time_discontinuity_conditions()
    found, stops, _ = sw.crossings_with_periodic(model._core, 0.0, 2000.0, conds, over_budget=over)
    assert found == [] and stops == []
    (item,) = over
    assert item.condition == conds[0] and item.changes == 4000
    assert item.min_gap == pytest.approx(0.1) and item.union_gap is None
    assert _x_end(model, 2000.0) == pytest.approx(1800.0, rel=1e-4)


class _Options:
    def __init__(self, max_step_size=0.0):
        self.max_step_size = max_step_size


def test_alone_a_schedule_is_bounded_at_half_its_shorter_stretch():
    bound = bngsim.Simulator._bound_step_past_stop_budget
    item = sw.StopsOverBudget("c", 4000, 0.1)
    opts = _Options()
    bound(opts, [item], 0.0, 2000.0, alone=True)
    assert opts.max_step_size == 0.05
    tighter = _Options(0.01)
    bound(tighter, [item], 0.0, 2000.0, alone=True)
    assert tighter.max_step_size == 0.01
    looser = _Options(0.5)
    bound(looser, [item], 0.0, 2000.0, alone=True)
    assert looser.max_step_size == 0.05


def test_beside_anything_else_its_own_spacing_bounds_nothing():
    """Not alone, with no `max_step` from the caller: refused, though its own
    shortest stretch is known."""
    bound = bngsim.Simulator._bound_step_past_stop_budget
    with pytest.raises(bngsim.SimulationError, match="#890"):
        bound(_Options(), [sw.StopsOverBudget("c", 4000, 0.1)], 0.0, 2000.0, alone=False)


@pytest.mark.parametrize(
    "beside",
    [
        "if(time() > 3, 0.5, 0)",
        "floor(time()/700)",
        "max(0, 1 - abs(time()-120))",
    ],
    ids=["another_condition", "a_step_call", "a_bending_call"],
)
def test_a_schedule_past_the_budget_beside_another_switch_is_refused(tmp_path, budget, beside):
    budget(1000)
    model = _model(tmp_path, dose=f"{_SCHEDULE} + {beside}")
    with pytest.raises(bngsim.SimulationError, match="#890"):
        _x_end(model, 2000.0)


def test_a_schedule_past_the_budget_beside_an_event_is_refused(budget):
    budget(1000)
    model = bngsim.Model.from_antimony_string(
        "species X = 0; P = 1; w = 0.1;\n"
        "J: -> X; piecewise(1, time - P*floor(time/P) < w, 0);\n"
        "E: at (X > 50): X = 0;\n"
    )
    with pytest.raises(bngsim.SimulationError, match="#890"):
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, 2000.0), n_points=3)


def test_a_condition_with_no_known_spacing_is_refused(tmp_path, budget):
    budget(1000)
    model = _model(tmp_path, dose=_FLOORS)
    with pytest.raises(bngsim.SimulationError, match="more than the 1,000,000 stops"):
        _x_end(model, 2000.0)


def test_it_names_the_condition_and_what_to_pass(tmp_path, budget):
    budget(1000)
    model = _model(tmp_path, dose=_FLOORS)
    with pytest.raises(bngsim.SimulationError) as err:
        _x_end(model, 2000.0)
    message = str(err.value)
    assert "floor(time()/P)" in message
    assert "max_step" in message and "run_until" in message and "#890" in message


def test_the_callers_step_bound_stands_for_it(tmp_path, budget):
    budget(1000)
    model = _model(tmp_path, dose=_FLOORS)
    assert _x_end(model, 2000.0, max_step=0.04) == pytest.approx(200.0, rel=1e-4)


_TWO = "if(mod(time(), P) < w, 1, 0) + if(mod(time(), 1.25*P) < 2*w, 1, 0)"


def test_one_budget_is_shared_by_every_condition(tmp_path, budget):
    """Two schedules of 600 and 480 edges do not both fit in 1000. The first
    is stopped at, the second is not, and the run is refused."""
    budget(1000)
    model = _model(tmp_path, dose=_TWO)
    over: list = []
    conds = model.time_discontinuity_conditions()
    found, stops, _ = sw.crossings_with_periodic(model._core, 0.0, 300.0, conds, over_budget=over)
    # Every edge of the first in (0, 300], give or take the one at 300 itself.
    assert len(found) in (599, 600) and stops == []
    (item,) = over
    assert item.condition == conds[1] and item.changes in (479, 480)
    with pytest.raises(bngsim.SimulationError, match="#890"):
        _x_end(model, 300.0)


def test_the_stretch_that_counts_is_between_any_two_changes(tmp_path, budget):
    """The second schedule's own pulses are 0.2 wide. Every fifth one ends
    0.05 before a pulse of the first begins: 3.95 and 4.0. The rate law is 1 on
    both sides of that stretch, so a step across it sees nothing."""
    budget(1000)
    model = _model(tmp_path, dose=_TWO)
    over: list = []
    sw.crossings_with_periodic(
        model._core, 0.0, 300.0, model.time_discontinuity_conditions(), over_budget=over
    )
    (item,) = over
    assert item.min_gap == pytest.approx(0.2)
    assert item.union_gap == pytest.approx(0.05)


def test_the_callers_step_bound_is_tightened_to_the_stretch_that_counts(tmp_path, budget):
    """300 pulses of 0.1 and 240 of 0.2. A `max_step` of 0.1 is half the
    second schedule's own stretch and twice the 0.05 one: taken as passed, the
    run returned 81."""
    budget(1000)
    model = _model(tmp_path, dose=_TWO)
    assert _x_end(model, 300.0, max_step=0.1) == pytest.approx(78.0, rel=1e-4)


def test_a_stochastic_run_past_the_budget_is_refused(tmp_path, budget):
    budget(1000)
    model = _model(tmp_path)
    with pytest.raises(bngsim.SimulationError, match="more than the 1,000,000 breakpoints"):
        _x_end(model, 2000.0, method="ssa", seed=1)


def test_a_condition_no_resolver_reads_is_counted_too(tmp_path, budget):
    budget(1000)
    model = _model(tmp_path, dose="if(frac(time()/P) < w, 1, 0)")
    with pytest.raises(bngsim.SimulationError, match="#890"):
        _x_end(model, 2000.0)
    assert _x_end(model, 2000.0, max_step=0.04) == pytest.approx(200.0, rel=1e-4)


def test_a_run_inside_the_budget_is_not_refused_after_one_that_was_not(tmp_path, budget):
    """The memo holds the refusal against the window it was made for."""
    budget(1000)
    model = _model(tmp_path, dose=_FLOORS)
    with pytest.raises(bngsim.SimulationError):
        _x_end(model, 2000.0)
    assert _x_end(_model(tmp_path, dose=_FLOORS), 200.0) == pytest.approx(20.0, rel=1e-4)


# ── What the first review found ─────────────────────────────────────────────
class _Opts:
    def __init__(self, max_step=0.0):
        self.max_step_size = max_step


def _over(**kw):
    kw.setdefault("condition", "c")
    kw.setdefault("changes", 2_000_000)
    kw.setdefault("min_gap", None)
    return sw.StopsOverBudget(**kw)


def test_a_step_bound_the_caller_chose_too_wide_is_tightened():
    """`max_step=1` beside a schedule of 0.01-wide windows: the bound was taken
    as standing for the stops, and stepped over them."""
    opts = _Opts(max_step=1.0)
    bngsim.Simulator._bound_step_past_stop_budget(opts, [_over(min_gap=0.01)], 0.0, 1000.0)
    assert opts.max_step_size == pytest.approx(0.005)
    # One already narrow enough is the caller's, and is left.
    opts = _Opts(max_step=0.001)
    bngsim.Simulator._bound_step_past_stop_budget(opts, [_over(min_gap=0.01)], 0.0, 1000.0)
    assert opts.max_step_size == 0.001


def test_a_bound_that_would_take_more_steps_than_a_run_has_is_refused():
    """Half of a 1e-6 stretch over 4,000 time units is 8e9 steps: a run that
    asked for nine minutes of one core and counting, with nothing said."""
    for opts in (_Opts(), _Opts(max_step=1.0)):
        with pytest.raises(bngsim.SimulationError, match="8e\\+09 steps or more"):
            bngsim.Simulator._bound_step_past_stop_budget(
                opts, [_over(min_gap=1e-6)], 0.0, 4000.0, alone=True
            )


def test_a_bound_wider_than_half_the_mean_stretch_cannot_stand_for_the_stops():
    """Where no stretch is known, 2,000,000 changes in 1,000 time units still
    say the shortest is at most 5e-4."""
    with pytest.raises(bngsim.SimulationError, match="more than half the mean"):
        bngsim.Simulator._bound_step_past_stop_budget(_Opts(max_step=0.01), [_over()], 0.0, 1000.0)
    opts = _Opts(max_step=1e-4)
    bngsim.Simulator._bound_step_past_stop_budget(opts, [_over()], 0.0, 1000.0)
    assert opts.max_step_size == 1e-4


# ── What the second review found ────────────────────────────────────────────
def test_a_sinusoid_past_the_budget_is_not_stepped_over(tmp_path, budget):
    """`sin(50 t) > 0.5` is true a third of the time. Past the budget its
    resolver warned in the log and returned no stops, and nothing else was
    asked: 2e-5 for 80. Alone it is integrated under a step bound; beside
    another switch it is refused."""
    budget(1000)
    alone = _model(tmp_path, dose="if(sin(50*time()) > 0.5, 1, 0)")
    assert _x_end(alone, 240.0) == pytest.approx(80.0, rel=2e-3)
    beside = _model(tmp_path, dose="if(sin(50*time()) > 0.5, 1, 0) + if(time() > 100, 1, 0)")
    with pytest.raises(bngsim.SimulationError, match="#890"):
        _x_end(beside, 240.0)


_REPORTED = """begin parameters
    1 k 0.1
end parameters
begin species
    1 X() 0
end species
begin functions
    1 dose() k
    2 reported() {reported}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""


@pytest.mark.parametrize(
    "reported",
    ["mod(time(), 0.01)", "if(mod(time(), 0.01) < 0.005, 1, 0)", "if(sin(50*time()) > 0, 1, 0)"],
)
def test_a_function_no_rate_law_reads_is_no_reason_to_refuse(tmp_path, budget, reported):
    """A function that only reports changes an output and not the trajectory.
    One with more changes than a run has stops for refused the run as if the
    rate law switched on it."""
    budget(1000)
    path = tmp_path / "reported.net"
    path.write_text(_REPORTED.format(reported=reported), encoding="utf-8")
    assert _x_end(bngsim.Model.from_net(str(path)), 240.0) == pytest.approx(24.0)


def test_which_conditions_a_rate_law_reads(tmp_path):
    path = tmp_path / "reported.net"
    path.write_text(_REPORTED.format(reported="if(time() > 3, 1, 0)"), encoding="utf-8")
    core = bngsim.Model.from_net(str(path))._core
    assert sw.conditions_a_rate_law_reads(core, ["time() > 3"]) == frozenset()
    model = _model(tmp_path)
    (cond,) = model.time_discontinuity_conditions()
    assert sw.conditions_a_rate_law_reads(model._core, [cond]) == frozenset({cond})
