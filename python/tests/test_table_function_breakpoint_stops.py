"""The integrator stops on each breakpoint of a table function read on the
clock, so a dosing pulse written as a ``tfun`` is not stepped over (issue #709).

A ``tfun`` over time is constant (``step``) or straight (``linear``) between
its breakpoints. Over a segment that begins and ends at the same value CVODE's
error estimate sees nothing, so the step grew past the whole of
``tfun([0,10,12,50], [0,5,0,0], time, method=>"step")`` and the input was
dropped: A + B stayed 3 where the answer is 13, with the same Result reporting
F = 5 on [10, 12). Pulses 0.1 and 0.5 wide were lost on a model that was still
relaxing.

Oracle: mass balance. A <-> B conserves A + B, and the only other reaction is
the zeroth-order input at rate F, so A + B at T is 3 plus the integral of F: an
exact number that uses none of bngsim's code. The stochastic run is held to
the same number through its mean.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import table_function_stops

_NET = """begin parameters
    1 kf {kf}
    2 kr {kr}
    3 one 1
end parameters
begin species
    1 $Src() 1
    2 A() {A0}
    3 B() {B0}
{species}
end species
begin groups
    1 Atot 2,3
{groups}
end groups
begin functions
{functions}
end functions
begin reactions
    1 1 1,2 F #in
    2 2 3 kf #f
    3 3 2 kr #r
{reactions}
end reactions
"""

_STEP = '    1 F() tfun([0,10,12,50],[0,5,0,0], time, method=>"step")'
# A unit-rate counter read through a group: how a BNGL model spells the clock.
_COUNTER = {
    "species": "    4 T() {t0}",
    "groups": "    2 Tobs 4",
    "reactions": "    4 0 4 one #clock",
}


def _model(tmp_path, functions, *, kf=1, kr=0.5, A0=1, B0=2, counter_from=None):
    extra = {k: "" for k in _COUNTER}
    if counter_from is not None:
        extra = {k: v.format(t0=counter_from) for k, v in _COUNTER.items()}
    path = tmp_path / "m.net"
    path.write_text(
        _NET.format(kf=kf, kr=kr, A0=A0, B0=B0, functions=functions, **extra), encoding="utf-8"
    )
    return bngsim.Model.from_net(str(path))


def _total(model, t_end=50.0, method="ode", **kw):
    run_kw = {k: kw.pop(k) for k in ("seed", "n_points") if k in kw}
    run_kw.setdefault("n_points", 6)
    r = bngsim.Simulator(model, method=method, **kw).run(t_span=(0.0, t_end), **run_kw)
    x = np.asarray(r.species)[-1]
    return float(x[1] + x[2])


# (functions block, A + B at t = 50)
_TABLES = {
    "step": (_STEP, 13.0),
    "step_in_arithmetic": (
        '    1 F() 2*tfun([0,10,12,50],[0,5,0,0], time, method=>"step")',
        23.0,
    ),
    "linear_triangle": ("    1 F() tfun([0,10,11,12,50],[0,0,5,0,0], time)", 8.0),
    "linear_triangle_two_hundredths_wide": (
        "    1 F() tfun([0,10,10.01,10.02,50],[0,0,5,0,0], time)",
        3.05,
    ),
    "step_a_thousandth_wide": (
        '    1 F() tfun([0,30,30.001,50],[0,5,0,0], time, method=>"step")',
        3.005,
    ),
    "a_gate_on_a_step_table": (
        _STEP.replace("F()", "G()") + "\n    2 F() if(G > 2, 1, 0)",
        5.0,
    ),
    "a_gate_on_a_linear_table": (
        "    1 G() tfun([0,10,11,12,50],[0,0,5,0,0], time)\n    2 F() if(G > 2, 1, 0)",
        3.0 + 1.2,
    ),
}


@pytest.mark.parametrize("functions, exact", _TABLES.values(), ids=_TABLES.keys())
def test_the_input_of_a_table_over_time_is_integrated(tmp_path, functions, exact):
    assert _total(_model(tmp_path, functions)) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("n_points", [2, 6, 501])
def test_the_answer_does_not_depend_on_the_output_grid(tmp_path, n_points):
    assert _total(_model(tmp_path, _STEP), n_points=n_points) == pytest.approx(13.0, rel=1e-6)


def test_the_compiled_right_hand_side_is_stopped_too(tmp_path):
    assert _total(_model(tmp_path, _STEP), codegen=True) == pytest.approx(13.0, rel=1e-6)


def test_a_file_backed_table_is_stopped_at(tmp_path):
    (tmp_path / "dose_file.tfun").write_text("# time F\n0 0\n10 5\n12 0\n50 0\n", encoding="utf-8")
    m = _model(tmp_path, "    1 F() tfun('dose_file.tfun', time, method=>\"step\")")
    assert _total(m) == pytest.approx(13.0, rel=1e-6)


@pytest.mark.parametrize("width", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_a_pulse_on_a_model_that_is_still_relaxing(tmp_path, width):
    """The issue's sweep: widths 0.1 and 0.5 gave 3.0, the rest were right."""
    table = f'    1 F() tfun([0,30,{30 + width},50],[0,5,0,0], time, method=>"step")'
    m = _model(tmp_path, table, kf=0.05, kr=0.02, A0=3, B0=0)
    assert _total(m) == pytest.approx(3.0 + 5.0 * width, rel=1e-6)


def test_a_run_that_ends_inside_the_pulse(tmp_path):
    assert _total(_model(tmp_path, _STEP), t_end=11.0) == pytest.approx(8.0, rel=1e-6)


def test_a_run_in_legs_stops_in_each_leg(tmp_path):
    m = _model(tmp_path, _STEP)
    sim = bngsim.Simulator(m, method="ode")
    for t_end in (5.0, 10.5, 20.0, 50.0):
        r = sim.run_until(t_end)
    x = np.asarray(r.species)[-1]
    assert float(x[1] + x[2]) == pytest.approx(13.0, rel=1e-6)


@pytest.mark.parametrize("counter_from, exact", [(0, 13.0), (7, 13.0), (11, 8.0), (20, 3.0)])
def test_a_table_indexed_by_a_counter_is_stopped_where_the_counter_arrives(
    tmp_path, counter_from, exact
):
    """`tfun(..., Tobs)` over a counter that starts at 7 pulses on t in [3, 5).
    Started at 11 it is half through the pulse already, and at 20 past it."""
    table = '    1 F() tfun([0,10,12,50],[0,5,0,0], Tobs, method=>"step")'
    m = _model(tmp_path, table, counter_from=counter_from)
    assert _total(m) == pytest.approx(exact, rel=1e-6)


def test_the_stochastic_mean_is_the_same_number(tmp_path):
    """The input is zeroth order at rate F, so A + B at 50 is 3 plus a Poisson
    count of mean 10: variance 10, and 400 runs put the mean within 0.16."""
    totals = [_total(_model(tmp_path, _STEP), method="ssa", seed=s) for s in range(1, 401)]
    z = (np.mean(totals) - 13.0) / math.sqrt(10.0 / len(totals))
    assert abs(z) < 4.5


# ── What is stopped at ──────────────────────────────────────────────────────
def _stop_times(model, t_start=0.0, t_end=50.0):
    return [stop.time for stop in table_function_stops(model._core, t_start, t_end)]


def test_a_step_table_is_stopped_where_its_value_changes(tmp_path):
    """50 is a breakpoint where the value stays 0: nothing breaks there."""
    assert _stop_times(_model(tmp_path, _STEP)) == [10.0, 12.0]


def test_a_linear_table_is_stopped_where_its_slope_changes(tmp_path):
    """(20, 2.5) lies on the line from (10, 0) to (30, 5): no bend there. The
    table is held at its end values outside itself, so a first or last segment
    that slopes bends at its end."""
    m = _model(tmp_path, "    1 F() tfun([5,10,20,30,40],[0,0,2.5,5,1], time)")
    assert _stop_times(m) == [10.0, 30.0, 40.0]
    assert _stop_times(m, t_start=10.0, t_end=30.0) == [30.0]


def test_a_table_indexed_by_a_counter_carries_the_counters_value(tmp_path):
    table = '    1 F() tfun([0,10,12,50],[0,5,0,0], Tobs, method=>"step")'
    stops = table_function_stops(_model(tmp_path, table, counter_from=7)._core, 0.0, 50.0)
    assert [(s.time, s.clock_species_idx, s.threshold) for s in stops] == [
        (3.0, 3, 10.0),
        (5.0, 3, 12.0),
    ]


def test_a_table_indexed_by_a_parameter_gets_no_stop(tmp_path):
    m = _model(tmp_path, '    1 F() tfun([0,10,12,50],[0,5,0,0], one, method=>"step")')
    assert _stop_times(m) == []


def test_the_core_hands_over_each_table(tmp_path):
    (table,) = _model(tmp_path, _STEP)._core.table_function_tables
    assert table == {
        "name": "F",
        "index_kind": "time",
        "index_param_idx": -1,
        "index_obs_idx": -1,
        "method": "step",
        "xs": [0.0, 10.0, 12.0, 50.0],
        "ys": [0.0, 5.0, 0.0, 0.0],
    }


class _Options:
    """What `_apply_crossing_stops` writes to, kept where it can be read."""

    max_step_size = 0.0

    def __init__(self):
        self.stops = None

    def set_crossing_stops(self, stops, restarts=()):
        self.stops = [time for time, _clock, _value in stops]
        self.restarts = list(restarts)

    def set_crossing_probes(self, probes):
        pass


def test_the_run_is_handed_the_breakpoints(tmp_path):
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, _STEP), method="ode")._apply_crossing_stops(opts, 0.0, 50.0)
    assert opts.stops == [10.0, 12.0]
    assert opts.restarts == [True, True]


def test_a_stop_along_a_smooth_stretch_of_a_linear_table_does_not_restart(tmp_path):
    """A step table jumps, and the history behind the stop belongs to the
    branch that ended. A linear one only bends, and along a dense table of a
    smooth signal the step ends on a bend now and then and the run keeps its
    history. At the bends of a feature (a triangle's foot, apex and end) it
    restarts: see the tests of ramps below."""
    xs = np.linspace(0.0, 100.0, 1001)
    m = _model(tmp_path, _dense("linear", 1.0 + np.sin(xs / 3.0) ** 2))
    stops = table_function_stops(m._core, 0.0, 200.0)
    kept = [s for s in stops if not s.restart]
    assert 5 <= len(kept) <= 60
    # It restarts at the table's ends and where the signal changes curvature,
    # where a bend is next to one a quarter its size: a few dozen times.
    restarts = [s.time for s in stops if s.restart]
    assert restarts[0] == pytest.approx(0.1) and restarts[-1] == 100.0
    assert len(restarts) <= 40
    triangle = _model(tmp_path, "    1 F() tfun([0,10,11,12,50],[0,0,5,0,0], time)")
    opts = _Options()
    bngsim.Simulator(triangle, method="ode")._apply_crossing_stops(opts, 0.0, 50.0)
    assert (opts.stops, opts.restarts) == ([10.0, 11.0, 12.0], [True, True, True])


def test_a_stop_that_ends_a_step_on_the_instant_of_a_jump_restarts(tmp_path):
    """One stop for the two, and it restarts: the jump needs it."""
    xs = np.linspace(0.0, 100.0, 1001)
    table = _dense("linear", 1.0 + np.sin(xs / 3.0) ** 2)
    kept = [
        s.time
        for s in table_function_stops(_model(tmp_path, table)._core, 0.0, 200.0)
        if not s.restart
    ]
    both = f"{table.replace('1 F()', '1 G()')}\n    2 F() G + if(time()>={kept[0]!r},1,0)"
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, both), method="ode")._apply_crossing_stops(opts, 0.0, 200.0)
    at = opts.stops.index(kept[0])
    assert opts.restarts[at] is True
    assert opts.restarts[opts.stops.index(kept[1])] is False


def test_a_dense_linear_table_is_integrated_as_accurately_as_before(tmp_path):
    """1 + sin(t/7) tabulated at 1001 points, driving dX/dt = F - 0.05 X: a
    table the integrator was already resolving. A step now ends on each of its
    999 bends, with no restart, and the answer at rtol 1e-8 stays within 3e-7
    of the run at 1e-12 (main: 1.3e-7), for 6% more steps."""
    xs = np.linspace(0.0, 240.0, 1001)
    ys = 1.0 + np.sin(xs / 7.0)
    table = (
        f"    1 F() tfun([{','.join(repr(float(x)) for x in xs)}],"
        f"[{','.join(repr(float(y)) for y in ys)}], time)"
    )
    path = tmp_path / "dense.net"
    path.write_text(
        "begin parameters\n    1 kd 0.05\nend parameters\n"
        "begin species\n    1 X() 0\nend species\n"
        f"begin functions\n{table}\nend functions\n"
        "begin reactions\n    1 0 1 F #in\n    2 1 0 kd #out\nend reactions\n",
        encoding="utf-8",
    )

    def x_end(rtol):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
        r = sim.run(t_span=(0.0, 240.0), n_points=25, rtol=rtol, atol=rtol * 1e-3)
        return float(np.asarray(r.species)[-1, 0])

    assert x_end(1e-8) == pytest.approx(x_end(1e-12), rel=1e-6)


def test_a_model_with_no_table_and_no_condition_gets_no_stops(tmp_path):
    """The common case, whose stepping must be untouched."""
    opts = _Options()
    m = _model(tmp_path, "    1 F() kf*0")
    bngsim.Simulator(m, method="ode")._apply_crossing_stops(opts, 0.0, 50.0)
    assert opts.stops is None


# ── What the first review found ─────────────────────────────────────────────
_GATE = (
    "    1 G() tfun([0,10,11,12,50],[0,0,{peak},0,0], time)\n"
    "    2 F() 1000*if(abs(G-2) < 0.01, 1, 0)"
)


@pytest.mark.parametrize(("peak", "exact"), [(5, 11.0), (50, 3.8)])
def test_a_condition_that_gates_on_a_table_is_a_condition_on_the_clock(tmp_path, peak, exact):
    """`abs(G - 2) < 0.01` with G a triangle over time: two windows, where the
    table passes 2 on its way up and on its way down, each 0.02/slope wide.
    A table call was not a read of the clock, so the condition was none, and
    A + B stayed at 3. The second table has the first one's breakpoints and
    ten times its values: what is remembered about one is not the other's."""
    assert _total(_model(tmp_path, _GATE.format(peak=peak))) == pytest.approx(exact, rel=1e-5)


def _dense(method: str, ys) -> str:
    xs = np.linspace(0.0, 100.0, len(ys))
    extra = ', method=>"step"' if method == "step" else ""
    return (
        f"    1 F() tfun([{','.join(repr(float(x)) for x in xs)}],"
        f"[{','.join(repr(float(y)) for y in ys)}], time{extra})"
    )


@pytest.mark.parametrize(("method", "at_most"), [("linear", 120), ("step", 600)])
def test_a_dense_table_of_a_smooth_signal_is_not_stopped_at_every_sample(
    tmp_path, method, at_most
):
    """10,000 samples of 1 + sin^2. Each break is like the one before it and
    the one after it: the curvature of what was sampled. A stop on each took
    the run from 0.3 ms to 44 (and to 300 at 100,000). The stops are the
    table's ends, the few samples around each place it turns or changes
    curvature, and one for each twentieth of its range it rises or falls by:
    a number the density of the samples does not change."""
    counts = []
    for n in (10_000, 40_000):
        xs = np.linspace(0.0, 100.0, n)
        m = _model(tmp_path, _dense(method, 1.0 + np.sin(xs / 3.0) ** 2))
        ((_, _, restarts, ends),) = m.table_function_breaks()
        # A step table restarts at every stop; a linear one at its ends and
        # where the signal changes curvature, whatever the number of samples.
        assert len(ends) == 0 if method == "step" else len(restarts) <= 40
        counts.append(len(restarts) + len(ends))
    assert counts[0] <= at_most and counts[1] <= at_most


@pytest.mark.parametrize("method", ["linear", "step"])
def test_a_pulse_in_a_dense_table_is_still_stopped_at(tmp_path, method):
    """The same samples with three of them lifted by 2: a break of more than
    a hundredth of the range."""
    xs = np.linspace(0.0, 100.0, 10_000)
    ys = 1.0 + np.sin(xs / 3.0) ** 2
    ys[5000:5003] += 2.0
    stops = _stop_times(_model(tmp_path, _dense(method, ys)), 0.0, 200.0)
    assert any(abs(t - xs[5000]) < 0.02 for t in stops)
    assert any(abs(t - xs[5003]) < 0.02 for t in stops)


def test_a_small_break_beside_a_stretch_with_none_is_a_stop(tmp_path):
    """A pulse of 0.9 on a baseline of 100, in a table that is flat on both
    sides of it: under a hundredth of the largest value, and the whole of the
    table's range. Flat on both sides is where a step grows across it."""
    table = '    1 F() tfun([0,10,12,50],[100,100.9,100,100], time, method=>"step")'
    assert _stop_times(_model(tmp_path, table)) == [10.0, 12.0]


def test_the_breaks_are_read_once_for_a_model_and_again_after_a_table_is_added(tmp_path):
    m = _model(tmp_path, _STEP)
    first = m.table_function_breaks()
    assert m.table_function_breaks() is first
    assert m.clone().table_function_breaks() is first
    m.add_table_function("late", times=[0.0, 20.0, 21.0, 50.0], values=[0.0, 1.0, 0.0, 0.0])
    assert len(m.table_function_breaks()) == 2


# ── What the second review found ────────────────────────────────────────────
def _doses(amp: float) -> tuple[str, float]:
    """A loading dose of 1000 for an hour, then *amp* for an hour every day."""
    xs, ys = [0.0, 1.0, 2.0], [0.0, 1000.0, 0.0]
    for day in range(1, 8):
        xs += [24.0 * day, 24.0 * day + 1.0]
        ys += [amp, 0.0]
    return f'    1 F() tfun({xs},{ys}, time, method=>"step")', 3.0 + 1000.0 + 7.0 * amp


@pytest.mark.parametrize("amp", [5.0, 9.0, 11.0])
def test_a_break_is_not_judged_by_its_size_beside_the_tables_largest(tmp_path, amp):
    """Maintenance doses of 5 after a loading dose of 1000 are half a percent
    of the table's range, and every one is a pulse on a model at rest. Judged
    against the range they got no stop: 1008 for 1038."""
    table, exact = _doses(amp)
    m = _model(tmp_path, table)
    assert _total(m, t_end=200.0) == pytest.approx(exact, rel=1e-6)
    assert len(_stop_times(m, 0.0, 200.0)) == 16


def test_a_low_bump_beside_a_tall_spike_is_stopped_at(tmp_path):
    """A triangle 0.4 high and a spike 100 high in one linear table."""
    table = "    1 F() tfun([0,10,11,12,50,50.001,50.002,100],[0,0,0.4,0,0,100,0,0], time)"
    stops = _stop_times(_model(tmp_path, table), 0.0, 100.0)
    assert stops == [10.0, 11.0, 12.0, 50.0, 50.001, 50.002]


def test_a_run_of_like_breaks_is_stopped_by_what_it_adds_up_to(tmp_path):
    """A staircase of 200 equal rises: each is like its neighbours, and a step
    across some of them sees the table higher at its end. One stop for each
    twentieth of the range, with the first and the last."""
    xs = [float(i) for i in range(201)]
    table = f'    1 F() tfun({xs},{xs}, time, method=>"step")'
    ((_, _, breaks, ends),) = _model(tmp_path, table).table_function_breaks()
    assert 20 <= len(breaks) <= 24 and len(ends) == 0
    assert breaks[0] == 1.0 and breaks[-1] == 200.0


# ── What the third review found ─────────────────────────────────────────────
def _ramp_model(tmp_path, xs, ys, x0=0.0):
    path = tmp_path / "ramp.net"
    path.write_text(
        "begin parameters\n    1 one 1\nend parameters\n"
        f"begin species\n    1 X() {x0}\nend species\n"
        f"begin functions\n    1 F() tfun({xs},{ys}, time)\nend functions\n"
        "begin reactions\n    1 0 1 F #in\nend reactions\n",
        encoding="utf-8",
    )
    return bngsim.Model.from_net(str(path))


def _x(model, t_end, **kw):
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=3, **kw)
    return float(np.asarray(r.species)[-1, 0])


@pytest.mark.parametrize(("start", "height"), [(10, 1e4), (100, 1e3), (4000, 100), (1e5, 1)])
def test_a_ramp_after_a_flat_stretch_is_integrated(tmp_path, start, height):
    """The step grows on the flat stretch, and a stop at the foot of the ramp
    that only ended a step left it that long: seven failures of the error
    test did not bring it down, and the run died (`CV_ERR_FAILURE`) where it
    had run before there was a stop at all. The integrator restarts there."""
    m = _ramp_model(tmp_path, [float(start), 2.0 * start], [0.0, float(height)])
    # The triangle under the ramp and the rectangle after it.
    exact = 0.5 * height * start + height * start
    assert _x(m, 3.0 * start) == pytest.approx(exact, rel=1e-6)


def test_a_run_that_ends_inside_a_ramp_sees_the_ramp(tmp_path):
    """Flat to 20, up to 1 at 40, back to 0 at 50, and a run to 35. The
    breakpoints at 40 and 50 are past the end and are no stops of this run,
    so a step from the foot went past 50, where the table is flat again, and
    the row at 35 was read back from it: X = 0 for 5.625."""
    xs, ys = [0.0, 20.0, 40.0, 50.0], [0.0, 0.0, 1.0, 0.0]
    assert _x(_ramp_model(tmp_path, xs, ys), 35.0) == pytest.approx(0.5 * 15.0 * 0.75, rel=1e-6)
    assert _x(_ramp_model(tmp_path, xs, ys), 60.0) == pytest.approx(15.0, rel=1e-6)


def test_a_linear_table_restarts_the_integrator_at_every_bend_of_a_feature(tmp_path):
    """On a segment the rate law is straight and the step grows to its
    length, so at the next bend it is too long by however steep the turn is."""
    table = "    1 F() tfun([0,10,11,12,30,50,55,80],[0,0,5,0,0,4,0,0], time)"
    ((_, _, restarts, ends),) = _model(tmp_path, table).table_function_breaks()
    assert restarts.tolist() == [10.0, 11.0, 12.0, 30.0, 50.0, 55.0]
    assert ends.tolist() == []


# ── What the fourth review found ────────────────────────────────────────────
def _exact(xs, ys, t_end, step=False) -> float:
    """The integral of the table over (0, t_end), held at its ends outside."""
    t = np.linspace(0.0, t_end, 2_000_001)
    if step:
        f = np.asarray(ys, dtype=float)[np.clip(np.searchsorted(xs, t, side="right") - 1, 0, None)]
    else:
        f = np.interp(t, xs, ys)
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))


@pytest.mark.parametrize("wiggle", [1e-6, 1e-3])
@pytest.mark.parametrize("n_points", [2, 11])
def test_a_baseline_that_wiggles_is_no_neighbour_to_the_foot_of_a_rise(tmp_path, wiggle, n_points):
    """A baseline of 50 with one sample a millionth off (data rounded to six
    digits), then a rise. The wiggle's bend counted as a neighbour of the
    rise's first bend, which was then accounted for and no stop, and a run
    that ended inside the rise returned the integral of the baseline: 175,000
    for 300,000."""
    xs = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0]
    ys = [50.0, 50.0 + wiggle, 50.0, 50.0, 1050.0, 4050.0, 30050.0, 50.0]
    m = _ramp_model(tmp_path, xs, ys)
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, 3500.0), n_points=n_points)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(_exact(xs, ys, 3500.0), rel=1e-6)


@pytest.mark.parametrize("t_end", [150.0, 250.0, 400.0])
def test_a_rise_beside_a_wiggle_does_not_stall_the_integrator(tmp_path, t_end):
    """The same foot as a stop that only ended a step: `CV_ERR_FAILURE`."""
    xs, ys = [0.0, 100.0, 200.0, 300.0, 400.0], [0.0, 1e-6, 1000.0, 4000.0, 0.0]
    got = _x(_ramp_model(tmp_path, xs, ys), t_end)
    assert got == pytest.approx(_exact(xs, ys, t_end), rel=1e-6)


def test_a_step_table_restarts_where_it_turns(tmp_path):
    """Twenty samples of a hump, as a step table. The two jumps beside its
    top, one up and one down, are each the mean of their neighbours, and
    between the stops before and after them the table is a pulse: the cap was
    lost, at any tolerance."""
    xs = np.linspace(0.0, 1.0, 21)
    ys = 10.0 * np.sin(np.pi * xs)
    table = (
        f"    1 F() tfun([{','.join(repr(float(x)) for x in xs)}],"
        f'[{",".join(repr(float(y)) for y in ys)}], time, method=>"step")'
    )
    path = tmp_path / "hump.net"
    path.write_text(
        "begin parameters\n    1 one 1\nend parameters\n"
        "begin species\n    1 X() 0\nend species\n"
        f"begin functions\n{table}\nend functions\n"
        "begin reactions\n    1 0 1 F #in\nend reactions\n",
        encoding="utf-8",
    )
    exact = float(np.sum(ys[:-1] * np.diff(xs)))
    got = _x(bngsim.Model.from_net(str(path)), 1.0)
    assert got == pytest.approx(exact, rel=1e-6)
