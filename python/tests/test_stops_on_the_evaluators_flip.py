"""A crossing stop sits on the double at which the model's evaluator reads
the new branch (issue #891).

The stop resolvers work a clock schedule's edges out from folded constants: a
period, an offset and a duty, or ``(n - beta)/alpha``. That is not the
arithmetic the evaluator computes the condition with, and where an edge is no
exact double the stop could land a double before the one the condition flips
at. The run then restarted on the branch that had just ended, and the jump
fell inside the first step after the restart, which has no history to see it
with. With P = 0.3 and w = 0.1 the error at rtol 1e-8 was 3e-6 and at 1e-10
still 3e-8: it did not follow the tolerance.

Each stop is now moved onto the first double at which the evaluator reads the
condition's new value, found by asking it: on the doubles around the stop,
and, for a condition whose argument carries an offset, by bisection between
the stop's neighbours. With `1e12 + time()` in the argument the two are up to
a double of the argument apart, 1.2e-4 of time, and with the stop past the
change the run restarted half a tread into the next branch: the remainder of a
far clock in issue #998's table lost every period after the 72nd.

A pulse between such stops then follows the tolerance, in every spelling of
its schedule alike. (Read the step that ends on a stop from one double before
it, and the pulse is integrated to rounding in a few steps a stop; but a law
that jumps on every stop then hides a pulse inside its stretches from the
integrator, which main finds by walking the stretch. That is not in this
change.)

A condition an event's trigger reads keeps its stop where the resolvers put
it: on the first true double of `time > 1`, a delay of 1.5 added to the
trigger time rounds to 2.5, and the execution came before a row at 2.5 that it
is after. How two events on one instant execute is issue #947.

Where a moved stop is beside another on a different double within one
instant, the run cannot take the two apart and no one stop serves both
conditions, so every stop of that run is where the resolvers put it, as on
main. A run at rtol 1e-3 or looser keeps the resolvers' stops too.

Oracles: closed forms where the clock is read bare (the model is ``0 -> X`` at
a unit pulse, so X(T) is the on-time; two events on one instant leave 0.15 in
sequence, as on main). For a law of a far clock, the law summed between its
own changes, each found by bisection on the model's evaluator: what the
evaluator computes, by none of the code that places stops.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import crossings_with_periodic

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

_SPELLINGS = {
    "remainder": "if(time() - P*floor(time()/P) < w, 1, 0)",
    "mod": "if(mod(time(), P) < w, 1, 0)",
    "floor_difference": "floor(time()/P) - floor((time()-w)/P)",
}
# (P, w, t_end): edges that are exact doubles, and edges that are not.
_SCHEDULES = [(1.0, 0.5, 600.0), (24.0, 1.0, 2400.0), (1.0, 0.1, 600.0), (0.3, 0.1, 300.0)]


def _model(tmp_path, dose, P, w):
    path = tmp_path / "m.net"
    path.write_text(_NET.format(P=P, w=w, dose=dose), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _on_time(P, w, t_end):
    n = math.floor(t_end / P)
    return n * w + min(t_end - n * P, w)


@pytest.mark.parametrize("rtol", [1e-8, 1e-10])
@pytest.mark.parametrize("P, w, t_end", _SCHEDULES)
@pytest.mark.parametrize("dose", _SPELLINGS.values(), ids=_SPELLINGS.keys())
def test_a_pulse_follows_the_tolerance(tmp_path, dose, P, w, t_end, rtol):
    """The issue's table. With P = 0.3 and w = 0.1 the remainder spelling was
    320 times rtol off at 1e-8 and 330 times at 1e-10, and each spelling of a
    schedule had an error of its own (with P = 1 and w = 0.1, 71 and 69 times
    rtol at 1e-8). With every stop on its change, all three are 11 and 4 times
    and 14 and 1 times."""
    sim = bngsim.Simulator(_model(tmp_path, dose, P, w), method="ode", codegen=False)
    r = sim.run(t_span=(0.0, t_end), n_points=11, rtol=rtol, atol=rtol * 1e-2)
    x_end = float(np.asarray(r.species)[-1, 0])
    assert x_end == pytest.approx(_on_time(P, w, t_end), rel=25 * rtol)


@pytest.mark.parametrize("P, w, t_end", _SCHEDULES)
@pytest.mark.parametrize("dose", _SPELLINGS.values(), ids=_SPELLINGS.keys())
def test_every_stop_is_the_first_double_of_its_branch(tmp_path, dose, P, w, t_end):
    """What the rate law reads one double before each stop differs from what
    it reads on it. As placed, 841 of 2000 stops of the P = 0.3 schedule read
    the same on both."""
    model = _model(tmp_path, dose, P, w)
    conds = model.time_discontinuity_conditions()
    found, _, _ = crossings_with_periodic(model._core, 0.0, t_end, conds, on_the_flip=True)
    stops = np.unique([s.time for s in found if s.time < t_end])
    assert stops.size >= math.floor(t_end / P)
    rate = dose
    here = model._core.evaluate_over_times(rate, stops)
    before = model._core.evaluate_over_times(rate, np.nextafter(stops, -np.inf))
    assert np.all(here != before)


@pytest.mark.parametrize("P, w, t_end", [(0.3, 0.1, 300.0), (0.7, 0.05, 700.0)])
def test_the_compiled_right_hand_side_flips_where_the_interpreter_does(tmp_path, P, w, t_end):
    """The stops are placed on the interpreter's flip. Compiled with fused
    multiply-add, `time() - P*floor(time()/P)` rounds once where the
    interpreter rounds twice, and flipped a double away at some edges: the
    compiled run was ten times less accurate at rtol 1e-10. Compiled without
    it, the two runs are one. The MIR JIT, which compiles the same source in
    process (BNGSIM_CODEGEN_JIT=mir, issue #1217), gives the same run too:
    skipping it there skipped the check under the JIT, with a reason that said
    the C compiler was missing."""
    ends = []
    for codegen in (False, True):
        sim = bngsim.Simulator(
            _model(tmp_path, _SPELLINGS["remainder"], P, w), method="ode", codegen=codegen
        )
        if codegen and sim.codegen_backend not in ("cc", "mir"):
            pytest.skip("no C compiler: the right-hand side is not compiled")
        r = sim.run(t_span=(0.0, t_end), n_points=11, rtol=1e-10, atol=1e-12)
        ends.append(float(np.asarray(r.species)[-1, 0]))
    assert ends[1] == pytest.approx(_on_time(P, w, t_end), rel=5e-9)
    assert ends[1] == pytest.approx(ends[0], rel=1e-12)


def test_a_stop_that_was_on_its_flip_is_not_moved(tmp_path):
    model = _model(tmp_path, _SPELLINGS["remainder"], 1.0, 0.5)
    conds = model.time_discontinuity_conditions()
    found, _, _ = crossings_with_periodic(model._core, 0.0, 6.0, conds, on_the_flip=True)
    assert [s.time for s in found] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]


def test_a_run_with_sensitivity_columns_keeps_its_stops_as_placed(tmp_path):
    """Such a run reads every condition itself, and its switch times are the
    resolvers' own numbers: the two must stay one."""
    model = _model(tmp_path, _SPELLINGS["remainder"], 0.3, 0.1)
    conds = model.time_discontinuity_conditions()
    plain, _, _ = crossings_with_periodic(model._core, 0.0, 30.0, conds, on_the_flip=True)
    with_columns, _, _ = crossings_with_periodic(
        model._core, 0.0, 30.0, conds, sensitivity_params=["P"]
    )
    assert len(plain) == len(with_columns)
    moved = sum(a.time != b.time for a, b in zip(plain, with_columns, strict=True))
    assert moved > 0
    assert all(
        abs(a.time - b.time) <= 4 * math.ulp(a.time)
        for a, b in zip(plain, with_columns, strict=True)
    )


class _Options:
    """What `_apply_crossing_stops` reads and writes, kept where it can be read."""

    max_step_size = 0.0

    def __init__(self, rtol):
        self.rtol = rtol
        self.stops = []

    def set_crossing_stops(self, stops, restarts=()):
        self.stops = [time for time, _clock, _value in stops]

    def set_crossing_probes(self, probes):
        pass


@pytest.mark.parametrize(
    "rtol, before", [(1e-2, True), (1e-3, True), (1e-4, False), (1e-8, False)]
)
def test_a_run_at_a_loose_tolerance_stops_a_double_before_each_change(tmp_path, rtol, before):
    """Moved onto its change at rtol 1e-2, the step reaching a stop read the
    new rate at its end and passed: ``mod(time(), 0.03) < 0.01`` came out 82%
    high where main, its stops a double early, was 4%. A double before each
    change it is exact to rounding (issue #1226); at 1e-4 and tighter the
    stops stay on the change."""
    model = _model(tmp_path, _SPELLINGS["remainder"], 0.3, 0.1)
    conds = model.time_discontinuity_conditions()
    flipped, _, _ = crossings_with_periodic(model._core, 0.0, 30.0, conds, on_the_flip=True)
    opts = _Options(rtol)
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 0.0, 30.0)
    assert opts.stops == [math.nextafter(s.time, -math.inf) if before else s.time for s in flipped]


# ── A threshold on a counter ────────────────────────────────────────────────
_COUNTER_NET = """begin parameters
    1 P 0.3
    2 w 0.1
    3 one 1
end parameters
begin species
    1 X() 0
    2 T() {t0}
end species
begin functions
    1 dose() if(t - P*floor(t/P) < w, 1, 0)
end functions
begin reactions
    1 0 1 dose #_R1
    2 0 2 one #clock
end reactions
begin groups
    1 t 2
end groups
"""


@pytest.mark.parametrize("t0", [0, 0.7, 5.05])
def test_a_schedule_on_a_counter_is_as_accurate(tmp_path, t0):
    """The stop carries the counter's value on the flip, which the counter is
    put on. Started at 0.7 the schedule is 0.1 into a period, with that pulse
    just over. (One text for every start value: the stops are remembered by
    the counter's window, not the run's.)"""
    path = tmp_path / "m.net"
    path.write_text(_COUNTER_NET.format(t0=t0), encoding="utf-8")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=False)
    r = sim.run(t_span=(0.0, 300.0), n_points=11, rtol=1e-10, atol=1e-12)
    exact = _on_time(0.3, 0.1, 300.0 + t0) - _on_time(0.3, 0.1, t0)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=5e-9)


_TWO_NET = """begin parameters
    1 P 0.3
    2 w 0.1
end parameters
begin species
    1 X() 0
    2 Y() 0
end species
begin functions
    1 f1() if(time() - P*floor(time()/P) < w, 1, 0)
    2 f2() if(mod(time(), P) < w, 1, 0)
end functions
begin reactions
    1 0 1 f1 #_R1
    2 0 2 f2 #_R2
end reactions
"""


def test_two_spellings_of_one_schedule_keep_the_stops_main_has(tmp_path):
    """Two rate laws on one schedule in two spellings change a double apart at
    a third of their edges. One stop taken for both, on the earlier change, left
    the `mod` law restarting a double before its own: 305 times rtol, where
    main is 33. Such a run keeps the stops where the resolvers put them."""
    path = tmp_path / "two.net"
    path.write_text(_TWO_NET, encoding="utf-8")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=False)
    r = sim.run(t_span=(0.0, 300.0), n_points=3, rtol=1e-8, atol=1e-10)
    y = float(np.asarray(r.species)[-1, 1])
    assert y == pytest.approx(_on_time(0.3, 0.1, 300.0), rel=60e-8)


# ── Events beside a moved stop ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "params, rate, t_star",
    [
        ("tau = 3", "piecewise(5, time > tau + 0, 0)", 3.0),
        ("tau = 3", "piecewise(5, time > 3, 0)", 3.0),
        ("tau = 3; t_dose = 3", "piecewise(5, time > t_dose, 0)", 3.0),
        ("tau = 7.3", "piecewise(5, time > tau + 0, 0)", 7.3),
        ("tau = 123.456", "piecewise(5, time > tau + 0, 0)", 123.456),
    ],
)
def test_events_on_a_rate_laws_change_execute_as_on_main(params, rate, t_star):
    """`time >= tau` then `time > tau`, beside a rate law that switches on
    the same instant. The rate law's stop is moved onto its change, a double
    after the events'; merged onto it, the two events executed as one batch
    a double late, Q = 0.5 where main executes them in sequence, 0.15."""
    text = (
        f"model m\n species X = 1, Q = 1\n {params}\n J0: -> X; {rate}\n"
        " E1: at (time > tau), priority = 3: Q = 0.3*Q\n"
        " E2: at (time >= tau), priority = 2: Q = 0.5*Q\nend"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    r = sim.run(sample_times=[0.0, t_star / 2, 1.4 * t_star, 2.1 * t_star], rtol=1e-10, atol=1e-12)
    assert float(np.asarray(r.species)[-1, 1]) == pytest.approx(0.15, rel=1e-9)


def test_a_delayed_event_beside_a_rate_laws_change_comes_after_its_row():
    """`at 1.5 after (time > 1)` beside a rate law on `time > t_on`, t_on = 1.
    Read on the rate law's moved stop, 1.0000000000000002, the trigger plus
    the delay rounded to 2.5 and the row at 2.5 already held the execution;
    main records the row first."""
    text = (
        "model m\n species X = 0, Q = 1\n t_on = 1\n J0: -> X; piecewise(5, time > t_on, 0)\n"
        " E1: at 1.5 after (time > 1): Q = 2*Q\nend"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    r = sim.run(t_span=(0.0, 5.0), n_points=11)
    q = dict(zip(np.round(np.asarray(r.time), 9), np.asarray(r.species)[:, 1], strict=True))
    assert (q[2.0], q[2.5], q[3.0]) == (1.0, 1.0, 2.0)


# ── A clock with an offset ──────────────────────────────────────────────────
_FAR_NET = """begin parameters
    1 one 1
end parameters
begin species
    1 X() 0
end species
begin functions
    1 dose() {law}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""
_FAR_LAWS = [
    ("floor(mod(1e9 + time(), 0.3)/0.3 + 0.0005)", 240.0),
    ("floor(mod(1e12 + time(), 0.3)/0.3 + 0.25)", 100.0),
    ("if(mod(1e12 + time(), 0.3) < 0.1, 1, 0)", 100.0),
    ("floor(mod(time(), 0.3)/0.3 + 0.25)", 400.0),
]


def _far_model(tmp_path, law):
    path = tmp_path / "far.net"
    path.write_text(_FAR_NET.format(law=law), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _changes(core, law, t_end, n=2_000_001, t_start=0.0):
    """Every change of a law that holds its value between changes, each on the
    first double of its new value, by bisection on the evaluator."""
    grid = np.linspace(t_start, t_end, n)
    v = core.evaluate_over_times(law, grid)
    i = np.nonzero(v[1:] != v[:-1])[0]
    lo, hi, before = grid[i].copy(), grid[i + 1].copy(), v[i].copy()
    for _ in range(80):
        mid = lo + 0.5 * (hi - lo)
        live = (mid > lo) & (mid < hi)
        if not live.any():
            break
        at_mid = before.copy()
        at_mid[live] = core.evaluate_over_times(law, mid[live])
        same = (at_mid == before) | ~live
        lo = np.where(same & live, mid, lo)
        hi = np.where(~same & live, mid, hi)
    return hi


def _integral_between_changes(core, law, t_end, t_start=0.0):
    edges = np.concatenate(([t_start], _changes(core, law, t_end, t_start=t_start), [t_end]))
    value = core.evaluate_over_times(law, 0.5 * (edges[:-1] + edges[1:]))
    return float(np.sum(value * np.diff(edges)))


def _stop_times(model, t_end):
    conds = model.time_discontinuity_conditions()
    found, scanned, _ = crossings_with_periodic(model._core, 0.0, t_end, conds, on_the_flip=True)
    return np.unique([s.time for s in (*found, *scanned)])


@pytest.mark.parametrize("law, t_end", _FAR_LAWS)
def test_every_stop_of_a_law_of_a_far_clock_is_on_the_evaluators_change(tmp_path, law, t_end):
    """As placed: 0 of 1,600, 0 of 666 and 1,288 of 2,666 for the three step
    laws, up to 6.1e-5 from the change at an offset of 1e12."""
    model = _far_model(tmp_path, law)
    stops = _stop_times(model, t_end)
    changes = _changes(model._core, law, t_end)
    assert stops.size == changes.size
    assert np.array_equal(stops, changes)


@pytest.mark.parametrize("rtol", [1e-6, 1e-8, 1e-10])
@pytest.mark.parametrize("law, t_end", _FAR_LAWS)
def test_a_pulse_of_a_far_clock_follows_the_tolerance(tmp_path, law, t_end, rtol):
    """The comment on the issue: 0.0720 for 0.1200 and 20.25 for 25.0 at the
    default tolerances, the second at any tolerance. Most of that was issue
    #998's lost stretches; with those found, a stop half a tread of the
    argument past each change still left the second law 8e-3 off at rtol 1e-6
    and the first 45 times rtol at 1e-8. Against what the evaluator computes,
    which for the second law is 24.9751: its pulses are whole treads of
    `1e12 + time()`."""
    model = _far_model(tmp_path, law)
    exact = _integral_between_changes(model._core, law, t_end)
    sim = bngsim.Simulator(_far_model(tmp_path, law), method="ode")
    r = sim.run(t_span=(0.0, t_end), n_points=3, rtol=rtol, atol=rtol * 1e-2)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=25 * rtol)


# ── The ends of a run ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "law, t_end, last",
    [
        ("9*mod(time(), 1)^8", 5.0, 5.0),
        ("if(mod(time(), 1) < 0.5, 3, 1)", 5.0, 5.0),
        ("floor(time())", 5.0, 5.0),
        # Half a double of the argument before the end: inside the run.
        ("100*mod(1000000000000.0 + time(), 0.125)", 0.625, 0.62493896484375),
    ],
)
def test_a_change_on_the_runs_last_instant_has_its_stop(tmp_path, law, t_end, last):
    """The resolvers of steps and schedules list no change on the last instant
    of a run, and a run of whole periods ends on one. With an offset in the
    argument the evaluator's change is before the end, and went unstopped:
    the last period of the far-clock remainder was lost."""
    stops = _stop_times(_far_model(tmp_path, law), t_end)
    assert stops[-1] == last


def test_a_leg_that_ends_on_a_threshold_keeps_its_stop_there(tmp_path):
    """`time() <= 0.3` changes a double after 0.3, which is past a leg that
    ends on 0.3. The stop stays on the end: a last step that is not stopped is
    taken past the end and read back across the change, and the next leg then
    started from a state that had seen it."""
    model = _far_model(tmp_path, "if(time() <= 0.3, 0, 1)")
    assert list(_stop_times(model, 0.3)) == [0.3]
    assert list(_stop_times(model, 1.0)) == [float(np.nextafter(0.3, 1.0))]


# ── Stops at a large time ───────────────────────────────────────────────────
_LATE = "if(mod(time(), 0.3) < 0.1, 1, 0)"


@pytest.mark.parametrize("t0, rtol", [(1e6, 1e-9), (1e6, 1e-11), (1e7, 1e-10)])
def test_a_dose_schedule_started_late_is_integrated_at_a_tight_tolerance(tmp_path, t0, rtol):
    """CVODE takes a stop as reached once a step ends within 100*uround*t of
    it, without integrating the rest. Stopped on the stop's own double, the
    step cut to end there rounds onto it at a large time, reads the new
    branch, fails its error test, and the run lost 2e-8 at each stop where the
    dose switched off: 180 times rtol at t0 = 1e6, rtol 1e-9 (main refused it
    for want of progress). Stopped a double before, with that double then
    integrated on the branch that ends, the dose is right to rounding."""
    model = _far_model(tmp_path, _LATE)
    exact = _integral_between_changes(model._core, _LATE, t0 + 30.0, t_start=t0)
    sim = bngsim.Simulator(_far_model(tmp_path, _LATE), method="ode")
    r = sim.run(t_span=(t0, t0 + 30.0), n_points=3, rtol=rtol, atol=rtol * 1e-2)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=1e-8)


@pytest.mark.parametrize("t0, rtol", [(1e6, 1e-8), (1e4, 1e-10), (1e3, 1e-11), (0.0, 1e-11)])
def test_a_start_whose_stops_resolve_to_the_tolerance_runs(tmp_path, t0, rtol):
    """Nearer the clock's origin, the same schedule follows the tolerance."""
    model = _far_model(tmp_path, _LATE)
    exact = _integral_between_changes(model._core, _LATE, t0 + 30.0, t_start=t0)
    sim = bngsim.Simulator(_far_model(tmp_path, _LATE), method="ode")
    r = sim.run(t_span=(t0, t0 + 30.0), n_points=3, rtol=rtol, atol=rtol * 1e-2)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=25 * rtol)
