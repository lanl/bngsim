"""An event whose time lies a few doubles after an output sample fires, and
the sample before it keeps the state before it (issue #950).

CVODE reports an event as a root: a change of sign of the trigger between two
times it reads it at. The run also stops on the instant a clock trigger rises
(`time >= tau`), and as a rule the rise is a root on that stop. It is not
where an output sample lies within the stop's roundoff before it, which is
100 ulps of the time by CVODE's own measure. The step that reaches the stop is
searched for roots only as far as the sample, where the trigger is still
false; the next call returns the stop without searching the stretch between;
and the run restarts on the stop, where CVODE takes the trigger, now true, as
the state it starts from. The event never fired: X(6) = 2 for 5 with tau one
to 256 ulps after a sample at 3, and with tau = 0.1 + 0.2 on any grid that has
0.3. Every trigger is now read at a stop the run restarts on.

Also here, since they are the same lines:

* a delayed event executed one or two doubles before a sample at t = 3e6 ended
  the run with CVODE's "tout too close" error: the window in which an output
  counts as reached was an absolute 1e-12, less than a double there;
* issue #739's cases (a delayed execution one or two doubles after a sample
  ran a whole output interval late), fixed in #955 and not tested since.

Oracles: closed forms. The count of executions, and the exponential decay
from the time an event executes at.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

_SET = "species X; X = 2; a = 5; tau = 3\nE1: at (time {op} tau): X = a\n"


def _after(x: float, ulps: int) -> float:
    for _ in range(abs(ulps)):
        x = math.nextafter(x, math.inf if ulps > 0 else -math.inf)
    return x


def _run(text, times, params=None, **kw):
    model = bngsim.Model.from_antimony_string(text)
    for name, value in (params or {}).items():
        model.set_param(name, value)
    return bngsim.Simulator(model, method="ode").run(sample_times=times, **kw)


def _x(text, times, params=None, **kw):
    return [float(v) for v in np.asarray(_run(text, times, params, **kw).species)[:, 0]]


# ── #950 ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ulps", [0, 1, 2, 8, 64, 100, 256, 300, 5000])
@pytest.mark.parametrize("sample", [0.3, 3.0, 300.0, 3.0e6])
def test_an_event_just_after_a_sample_fires(sample, ulps):
    """The issue's table, at four magnitudes of the time: the window scales
    with it. Every row was 2.0 at the end for 1 to 256 ulps."""
    times = [0.0, 0.5 * sample, sample, 1.5 * sample, 2.0 * sample]
    x = _x(_SET.format(op=">="), times, {"tau": _after(sample, ulps)})
    assert x[-2:] == [5.0, 5.0]


@pytest.mark.parametrize("ulps", [1, 8, 256])
def test_the_sample_before_the_event_holds_the_state_before_it(ulps):
    """The event is after the sample at 3, by however little: that row is 2.
    On the sample itself (0 ulps) the row is the event's 5."""
    times = [0.0, 1.5, 3.0, 4.5, 6.0]
    assert _x(_SET.format(op=">="), times, {"tau": _after(3.0, ulps)}) == [2, 2, 2, 5, 5]
    assert _x(_SET.format(op=">="), times, {"tau": 3.0}) == [2, 2, 5, 5, 5]


@pytest.mark.parametrize("op", [">=", ">"])
@pytest.mark.parametrize("ulps", [-256, -8, -1, 0, 1, 8, 256])
def test_it_fires_whichever_side_of_the_sample_it_is_on(op, ulps):
    x = _x(_SET.format(op=op), [0.0, 1.5, 3.0, 4.5, 6.0], {"tau": _after(3.0, ulps)})
    assert x[-1] == 5.0


def test_a_dose_time_computed_as_a_sum_fires_on_a_round_grid():
    """How it is met in practice: 0.1 + 0.2 is one double above 0.3."""
    text = "species X; X = 2; a = 5; t1 = 0.1; t2 = 0.2\nE1: at (time >= t1 + t2): X = a\n"
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1.0), n_points=11)
    assert float(np.asarray(r.species)[-1, 0]) == 5.0


def test_several_samples_within_a_few_doubles_of_the_event():
    """Samples 2 and 1 doubles before the event, on it, and 1 after."""
    tau = _after(3.0, 5)
    times = [0.0, _after(tau, -2), _after(tau, -1), tau, _after(tau, 1), 6.0]
    assert _x(_SET.format(op=">="), times, {"tau": tau}) == [2, 2, 2, 5, 5, 5]


def test_the_decay_after_the_event_starts_at_the_event():
    """X is set to 10 at tau and decays at 0.5: X(6) = 10 exp(-0.5 (6 - tau)).
    One sample late would be 10 exp(-0.5 * 1.5)."""
    text = "species X = 1; J: X -> ; 0.5*X; tau = 3\nE: at (time >= tau): X = 10\n"
    tau = _after(3.0, 8)
    x = _x(text, [0.0, 1.5, 3.0, 4.5, 6.0], {"tau": tau}, rtol=1e-10, atol=1e-12)
    assert x[-1] == pytest.approx(10.0 * math.exp(-0.5 * (6.0 - tau)), rel=1e-8)
    assert x[2] == pytest.approx(math.exp(-1.5), rel=1e-8)


def test_two_events_just_after_a_sample_keep_their_priorities():
    """Both rise on the stop: the higher priority executes first and the
    lower one's value stays."""
    text = (
        "species X; X = 2; tau = 3\n"
        "E1: at (time >= tau), priority = 1: X = 7\n"
        "E2: at (time >= tau), priority = 9: X = 9\n"
    )
    assert _x(text, [0.0, 3.0, 6.0], {"tau": _after(3.0, 4)})[-1] == 7.0


def test_a_delayed_event_triggered_just_after_a_sample_is_delayed_from_there():
    text = "species X = 1; J: X -> ; 0.5*X; tau = 3\nE: at 1 after (time >= tau): X = 10\n"
    tau = _after(3.0, 8)
    x = _x(text, [0.0, 3.0, 6.0], {"tau": tau}, rtol=1e-10, atol=1e-12)
    assert x[-1] == pytest.approx(10.0 * math.exp(-0.5 * (6.0 - tau - 1.0)), rel=1e-8)


def test_a_run_in_legs_that_end_on_the_sample_fires_it_once():
    model = bngsim.Model.from_antimony_string(
        "species X; X = 0; tau = 3\nE1: at (time >= tau): X = X + 1\n"
    )
    model.set_param("tau", _after(3.0, 8))
    sim = bngsim.Simulator(model, method="ode")
    for t_end in (1.5, 3.0, 4.5, 6.0):
        r = sim.run_until(t_end)
    assert float(np.asarray(r.species)[-1, 0]) == 1.0


# ── An execution a double before a sample, at a large time ──────────────────
# x is the clock, read into y when the event executes; c counts executions.
_DELAYED = (
    "x = 0; x' = 1; y = 0; c = 0; dly = 1;\n"
    "E: at dly after (time >= 0), t0=false, fromTrigger=false: y = x, c = c + 1;\n"
)


def _y_and_count(t_exec, t_end, n_points):
    model = bngsim.Model.from_antimony_string(_DELAYED)
    model.set_param("dly", t_exec)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=1e-10, atol=1e-12
    )
    names = list(r.observable_names)
    obs = np.asarray(r.observables)[-1]
    return float(obs[names.index("y")]), float(obs[names.index("c")])


@pytest.mark.parametrize("ulps", [-3, -2, -1, 0, 1, 2, 3])
def test_a_delayed_execution_beside_a_sample_at_a_large_time(ulps):
    """Samples every 1e6 to 1e7. One or two doubles before the sample at 3e6
    the run failed with CV_TOO_CLOSE."""
    t_exec = _after(3.0e6, ulps)
    y, counted = _y_and_count(t_exec, 1.0e7, 11)
    assert y == pytest.approx(t_exec, rel=1e-9)
    assert counted == 1.0


# ── #739 ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n_points", [11, 101, 1000, 1001, 1002])
@pytest.mark.parametrize(
    "trigger, delay",
    [(1.1, 1.3), (0.1, 0.2), (0.7, 0.1), (2.2, 1.1), (0.3, 0.6), (1.0, 1.4), (4.35, 0.05)],
)
def test_a_delayed_execution_beside_a_sample_runs_at_its_time(trigger, delay, n_points):
    """1.1 + 1.3 is one double above the sample 2.4 of a 1001-point grid, and
    the event ran at 2.41: Z(10) was 0.2316 for 0.2304."""
    text = (
        "species Z; Z = 1; Jd: Z -> ; 0.5*Z;\n"
        f"E: at {delay!r} after (time >= {trigger!r}), t0=true, fromTrigger=false: Z = Z + 10;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 10.0), n_points=n_points, rtol=1e-10, atol=1e-12
    )
    exact = math.exp(-5.0) + 10.0 * math.exp(-0.5 * (10.0 - (trigger + delay)))
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=1e-8)


@pytest.mark.parametrize("ulps", [0, 1, 2, 3])
@pytest.mark.parametrize("n_points", [11, 101, 1001])
def test_a_second_delayed_event_a_double_after_the_first(ulps, n_points):
    """Ea is due at 0.01 + 8.29 = 8.299999999999999, Eb a few doubles later:
    X is set to 2 there and decays at 0.1."""
    due = _after(0.01 + 8.29, ulps)
    text = (
        "species X; X = 1; J: X -> ; 0.1*X; m = 0;\n"
        "Ea: at 8.29 after (time >= 0.01), t0=false: m = 1;\n"
        f"Eb: at {due!r} after (time >= 0), t0=false: X = 2;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 10.0), n_points=n_points, rtol=1e-10, atol=1e-12
    )
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(
        2.0 * math.exp(-0.1 * (10.0 - due)), rel=1e-8
    )


@pytest.mark.parametrize("n_points", [2, 11, 51, 101, 1001])
def test_a_chain_of_delayed_events_loses_none(n_points):
    """One execution every 0.1, each triggering the next: 99 by t = 9.95. On
    an 11-point grid 20 of 100 were lost."""
    text = (
        "n = 0; q = 0;\n"
        "E: at 0.1 after (q < 0.5), t0=false: n = n + 1, q = 1;\n"
        "R: at (q > 0.5): q = 0;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 9.95), n_points=n_points)
    assert float(np.asarray(r.observables)[-1, list(r.observable_names).index("n")]) == 99.0


# ── What the first review found ─────────────────────────────────────────────
# The window in which an output counts as reached was read before a step was
# asked for, with an absolute floor of 1e-12: thousands of doubles at t = 1,
# and the whole of a run on a picosecond scale.
_DECAY_EVENT = (
    "species X = 1; J: X -> ; k*X; k = 1; tau = 1; n = 0;\n"
    "E: at (time >= tau), fromTrigger=false: X = X + 10, n = n + 1;\n"
)


def _n(result) -> list[float]:
    column = list(result.observable_names).index("n")
    return [float(v) for v in np.asarray(result.observables)[:, column]]


def test_a_run_on_a_picosecond_scale_is_integrated():
    """`X -> ; 1e13 X` over (0, 1e-12): every row came back at t = 0 with X = 1."""
    text = "species X = 1; J: X -> ; k*X; k = 1e13;\ny = 0; E: at (time >= 1): y = 1;\n"
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1e-12), n_points=11)
    assert np.asarray(r.time).tolist() == np.linspace(0.0, 1e-12, 11).tolist()
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(math.exp(-10.0), rel=1e-3)


def test_an_event_on_a_picosecond_scale_fires():
    model = bngsim.Model.from_antimony_string(_DECAY_EVENT)
    model.set_param("k", 1e13)
    model.set_param("tau", 5e-13)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1e-12), n_points=11)
    assert _n(r)[-1] == 1.0
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(
        math.exp(-10.0) + 10.0 * math.exp(-5.0), rel=1e-4
    )


@pytest.mark.parametrize(
    ("times", "tau", "counts"),
    [
        # An event on the last sample, the one before it 5e-13 earlier.
        ([0.0, 0.5, 1.0 - 5e-13, 1.0], 1.0, [0, 0, 0, 1]),
        # An event 2e-13 before the last sample.
        ([0.0, 0.5, 1.0 - 5e-13, 1.0], 1.0 - 2e-13, [0, 0, 0, 1]),
        # At t = 1e-3, where 5e-13 is two million doubles.
        ([0.0, 1e-3 - 8e-13, 1e-3, 2e-3], 1e-3 - 5e-13, [0, 0, 1, 1]),
    ],
)
def test_an_event_under_1e_12_after_a_sample_is_integrated_to(times, tau, counts):
    r = _run(_DECAY_EVENT, times, {"tau": tau})
    assert _n(r) == counts
    # Each row is recorded at the time it was asked for.
    assert np.asarray(r.time).tolist() == times


_PENDING = (
    "species X = 1; J: X -> ; k*X; k = 5; n = 0;\n"
    "E: at 0.4 after (time >= 0.1), fromTrigger=false: X = X + 10, n = n + 1;\n"
)
_FIRED = (
    "species X = 1; J: X -> ; k*X; k = 5; n = 0;\n"
    "E: at (time >= 0.05), t0=false, fromTrigger=false: X = X + 10, n = n + 1;\n"
)


def _legs(text, ends, params=None):
    model = bngsim.Model.from_antimony_string(text)
    for name, value in (params or {}).items():
        model.set_param(name, value)
    sim = bngsim.Simulator(model, method="ode")
    counts, times = [], []
    for end in ends:
        r = sim.run_until(end)
        counts.append(_n(r)[-1])
        times.append(float(np.asarray(r.time)[-1]))
    return counts, times


@pytest.mark.parametrize(
    ("text", "counts"), [(_PENDING, [0, 0, 1]), (_FIRED, [1, 1, 1])], ids=["pending", "fired"]
)
def test_a_leg_one_double_long_hands_on_the_events_it_was_handed(text, counts):
    """`run_until(0.3)` then `run_until(0.1 + 0.2)`: a leg too short to
    integrate. Its row was recorded at the time it started from, the next leg
    asked for the event carry of that time, and found none: a delayed
    execution waiting on 0.5 was dropped, and an event that had fired once
    fired again."""
    ends = [0.3, 0.1 + 0.2, 0.6]
    got, times = _legs(text, ends)
    assert got == counts
    assert times == ends


def test_a_leg_that_ends_a_double_after_a_delayed_execution_keeps_the_carry():
    """Due one double before the end of the first leg, which is taken as
    reached from there: every later leg fired the event again."""
    text = "n = 0; dl = 0.3;\nE: at dl after (time >= 0), t0=false: n = n + 1;\n"
    counts, _ = _legs(text, [0.3, 0.6, 0.9], {"dl": _after(0.3, -1)})
    assert counts == [1, 1, 1]


@pytest.mark.parametrize("ulps", [1, 2, 3, 70])
def test_a_run_that_starts_a_double_before_the_event_fires_it(ulps):
    """A stop closer ahead of the start than CVODE integrates over ended the
    run in its "tout too close" error, as a leg and as a run of its own."""
    text = (
        "species X = 1; J: X -> ; k*X; k = 5; n = 0; tau = 1;\nE: at (time >= tau): n = n + 1;\n"
    )
    counts, _ = _legs(text, [0.3, 0.6], {"tau": _after(0.3, ulps)})
    assert counts == [0, 1]
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("tau", _after(0.3, ulps))
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.3, 0.6), n_points=3)
    assert _n(r) == [0, 1, 1]


def test_an_event_inside_a_leg_four_doubles_long_at_a_large_time():
    text = (
        "species X = 1; J: X -> ; k*X; k = 1e-7; n = 0; tau = 1;\n"
        "E: at (time >= tau), fromTrigger=false: X = X + 10, n = n + 1;\n"
    )
    counts, times = _legs(text, [3e6, _after(3e6, 4), 6e6], {"tau": _after(3e6, 2)})
    assert counts == [0, 1, 1]
    assert times == [3e6, _after(3e6, 4), 6e6]


# ── What the second review found ────────────────────────────────────────────
_DUE = "n = 0;\nE: at 1 after (time >= 1), fromTrigger=false: n = n + 1;\n"


@pytest.mark.parametrize("scale", [1e-3, 0.35, 1.0, 3e6])
@pytest.mark.parametrize("ulps", [1, 2, 3, 6])
def test_a_delayed_execution_due_on_a_row_reached_from_short_is_in_the_row(scale, ulps):
    """Due at T, with the sample before it a double or two short of T: the
    row at T was taken as reached and recorded without the execution."""
    text = f"n = 0; d = {scale!r};\nE: at d after (time >= d), fromTrigger=false: n = n + 1;\n"
    due = 2.0 * scale
    r = _run(text, [0.0, scale, _after(due, -ulps), due, 1.5 * due])
    assert _n(r)[-3:] == [0.0, 1.0, 1.0]


def test_a_leg_that_ends_on_a_delayed_execution_from_a_double_short_runs_it():
    counts, times = _legs(_DUE, [_after(2.0, -1), 2.0, 3.0])
    assert counts == [0, 1, 1]
    assert times == [_after(2.0, -1), 2.0, 3.0]


@pytest.mark.parametrize("ulps", [1, 5, 20, 60, 100, 200, 400])
def test_a_sample_just_before_an_instant_leaves_its_priorities_alone(ulps):
    """An immediate event of priority 10 and a delayed one of priority 1 on
    t = 2, with a sample a few doubles before 2. CVODE returns the stop in
    place of the sample; the delayed execution ran there, before the row and
    before the event of higher priority, which was then taken for a cascade:
    x = 1 for 2, and the row before 2 held both."""
    text = (
        "x = 0;\n"
        "I: at (time >= 2), priority = 10: x = 1;\n"
        "D: at 1 after (time >= 1), priority = 1: x = 2;\n"
    )
    r = _run(text, [0.0, 1.5, _after(2.0, -ulps), 3.0])
    x = [float(v) for v in np.asarray(r.observables)[:, list(r.observable_names).index("x")]]
    assert x == [0.0, 0.0, 0.0, 2.0]


@pytest.mark.parametrize("ulps", [1, 2, 3, 150])
def test_every_row_carries_the_time_that_was_asked_for(ulps):
    """Samples a few doubles before a stop came back labelled with the stop."""
    times = [0.0, 1.0, _after(2.0, -ulps), 2.0, 3.0]
    r = _run("n = 0;\nE: at (time >= 2): n = n + 1;\n", times)
    assert np.asarray(r.time).tolist() == times
    assert _n(r) == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_a_run_with_sensitivity_columns_refuses_a_distance_it_cannot_integrate():
    """Its row is read from the integrator's history, and before the first
    step there is none: a first row a double after the start, taken as
    reached, came back as a row of zeros. It keeps CVODE's own refusal."""
    model = bngsim.Model.from_antimony_string("species X = 1; J: X -> ; k*X; k = 1;")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    with pytest.raises(bngsim.SimulationError, match="CV_TOO_CLOSE"):
        sim.run(sample_times=[5.0, _after(5.0, 1), 6.0])


# ── What the third review found ─────────────────────────────────────────────
def _col(result, name):
    names = list(result.observable_names)
    return [float(v) for v in np.asarray(result.observables)[:, names.index(name)]]


def test_what_is_due_a_double_before_a_stop_executes_before_the_stop():
    """`at 0.1 after (time >= 0.7)` is due on 0.7999999999999999, one double
    before an event on 0.8. CVODE returns the stop in place of the execution's
    time, and read as one batch with the stop's the later event ran first: a
    dose added after a doubling instead of before it, in a model with no
    priorities, which main had right."""
    text = (
        "species Z = 1; JZ: Z -> ; 0.3*Z;\n"
        "A: at 0.1 after (time >= 0.7), fromTrigger=false: Z = Z + 1;\n"
        "B: at (time >= 0.8), fromTrigger=false: Z = 2*Z;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 2.0), n_points=21)
    # (Z(0.8) + 1) doubled, decayed to 2.
    exact = 2.0 * (math.exp(-0.3 * 0.8) + 1.0) * math.exp(-0.3 * 1.2)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("ulps", [1, 2, 5, 40, 99, 400])
def test_instants_a_few_doubles_apart_keep_the_order_of_time(ulps):
    """A delayed event of priority 1 due at 2 and an immediate one of priority
    10 a few doubles later: two instants, whatever the priorities."""
    text = (
        "x = 0; tau = 2;\n"
        "A: at 1 after (time >= 1), priority = 1: x = 1;\n"
        "B: at (time >= tau), priority = 10: x = 2;\n"
    )
    r = _run(text, [0.0, 1.5, 3.0], {"tau": _after(2.0, ulps)})
    assert _col(r, "x")[-1] == 2.0


@pytest.mark.parametrize("ulps", [1, 2, 8, 40, 99, 150, 400])
def test_a_row_on_a_due_time_holds_the_execution_whatever_stop_follows(ulps):
    """Due at 2 with a row on 2, and a stop a few doubles later that CVODE
    returns in its place. The instant waited for the row; what belongs to the
    row's own time is read before it."""
    text = (
        "n = 0; m = 0; tau = 2;\n"
        "D: at 1 after (time >= 1): n = n + 1;\n"
        "E: at (time > tau): m = m + 1;\n"
    )
    times = [0.0, 1.5, 2.0, 2.5]
    r = _run(text, times, {"tau": _after(2.0, ulps)})
    assert np.asarray(r.time).tolist() == times
    assert _col(r, "n") == [0.0, 0.0, 1.0, 1.0]
    assert _col(r, "m") == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("ulps", [1, 8, 60, 99])
def test_what_a_row_reads_of_the_clock_is_read_at_the_rows_time(ulps):
    """A sample a few doubles before a switch of an assignment rule, returned
    from the stop on the switch: labelled with its own time, the row held the
    rule's value from the other side."""
    text = (
        "species X = 1; J: X -> ; k*X; k = 1; tau = 2;\n"
        "dose := piecewise(1, time >= tau, 0);\n"
        "n = 0; E: at (time >= tau): n = n + 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sample = _after(2.0, -ulps)
    r = bngsim.Simulator(model, method="ode").run(sample_times=[0.0, sample, 3.0])
    assert float(np.asarray(r.time)[1]) == sample
    dose = np.asarray(r.expressions)[:, list(r.expression_names).index("dose")]
    assert [float(v) for v in dose] == [0.0, 0.0, 1.0]
    assert _n(r) == [0.0, 0.0, 1.0]


def test_a_trigger_on_a_stop_inside_a_two_double_leg_is_read_at_the_stop():
    """Legs to a - 1 ulp and a + 1 ulp around a trigger time a: the trigger
    was read at the leg's end, the delayed event it starts fell due a double
    after an immediate one meant for the same instant, and x came out 1."""
    a, d = 0.3, 0.6
    text = (
        "x = 0; y = 0; a = 0.3; d = 0.6; c = 1;\n"
        "D: at d after (time >= a), priority = 6, fromTrigger=true: y = x + y + 1;\n"
        "I: at (time >= c), fromTrigger=false: x = x + y + 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("c", a + d)
    sim = bngsim.Simulator(model, method="ode")
    for end in (_after(a, -1), _after(a, 1), 2.0 * (a + d)):
        r = sim.run_until(end)
    assert _col(r, "x")[-1] == 2.0


def test_a_run_with_sensitivity_columns_keeps_its_rows_where_they_stood():
    """Its columns are read where the run stands. A sample a few doubles
    before a switch time comes back from the stop on it, with the switch's
    jump in the column: labelled with the stop, as before, where that is
    right."""
    text = "species X = 1; k1 = 1; k2 = 2; tau = 1;\nJ: X -> ; piecewise(k1, time < tau, k2)*X;\n"
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    r = sim.run(sample_times=[0.0, _after(1.0, -8), 2.0])
    assert float(np.asarray(r.time)[1]) == 1.0
    assert float(np.asarray(r.sensitivities)[1, 0, 0]) == pytest.approx(math.exp(-1.0), rel=1e-6)


# ── What the fourth review found ────────────────────────────────────────────
def _last(result, name):
    values = dict(zip(result.species_names, np.asarray(result.species)[-1], strict=True))
    values.update(zip(result.observable_names, np.asarray(result.observables)[-1], strict=True))
    return float(values[name])


def _trigger_time_for(due: float, delay: float) -> float:
    """A trigger time t with ``t + delay == due`` in doubles."""
    for ulps in (0, 1, -1, 2, -2, 3, -3):
        t = _after(due - delay, ulps)
        if t + delay == due:
            return t
    raise AssertionError("no trigger time")


_THREE = (
    "species Z = 1; JZ: Z -> ; 0.3*Z; x = 0; y = 0; xp = -1; xd = -1;\n"
    "aP = 1; aD = 1; dD = 1; a0 = 1; d0 = 1;\n"
    "D0: at d0 after (time >= a0), fromTrigger=false: y = y + 1;\n"
    "P: at (time >= aP), priority = 1, fromTrigger=false: x = 1, xp = x;\n"
    "D: at dD after (time >= aD), priority = 10, fromTrigger=false: x = 2, xd = x;\n"
)


def test_a_stop_behind_a_row_is_still_read_at_its_own_time():
    """A sample on T with a delayed execution due on it, an event P on a stop
    three doubles later, and a delayed execution D due two doubles after
    that. CVODE returns the stop in place of the sample. The row's execution
    ran, the run was restarted on the row's time with the stop already taken,
    and P was left to the root search of the next step, which returned it
    with D and ordered the two by priority: D saw x = 0 and P wrote last."""
    T = 1.3
    stop = _after(T, 3)
    due = _after(stop, 2)
    params = {
        "a0": _trigger_time_for(T, 0.5),
        "d0": 0.5,
        "aP": stop,
        "aD": _trigger_time_for(due, 0.25),
        "dD": 0.25,
    }
    r = _run(_THREE, [0.0, T, 2.3], params, rtol=1e-9, atol=1e-12)
    assert (_last(r, "xp"), _last(r, "xd"), _last(r, "x")) == (0.0, 1.0, 2.0)


def test_a_rate_law_that_switches_on_a_stop_behind_a_row_is_integrated():
    """The stop is a dose that starts at tau = 3000000.5, and a delayed
    execution is due on the sample one double before it. Restarted on the
    sample with the stop taken, the integrator had a discontinuity a double
    after its start and made no progress."""
    tau = 3000000.5
    text = (
        "species X = 1; J: X -> ; 0.3*X; tau = 1; tq = 1; m = 0;\n"
        "dose := piecewise(1, time >= tau, 0); J2: -> X; dose;\n"
        "Q: at 0.25 after (time >= tq), fromTrigger=false: m = m + 1;\n"
    )
    before = _after(tau, -1)
    params = {"tau": tau, "tq": _trigger_time_for(before, 0.25)}
    r = _run(text, [0.0, before, tau + 10.0], params, rtol=1e-9, atol=1e-12)
    assert _last(r, "m") == 1.0
    assert _last(r, "X") == pytest.approx((1.0 - math.exp(-3.0)) / 0.3, rel=1e-7)


def test_an_execution_due_before_a_rooted_event_runs_before_it():
    """A leg that ends three doubles before a delayed execution D's time, and
    an event P two doubles after that time. The next leg is stopped for a
    little past D's time, P is rooted in between, and D, still waiting, was
    taken into P's batch and ran after it: P saw x = 0 and D wrote last."""
    T = 1.3
    text = (
        "species Z = 1; JZ: Z -> ; 0.3*Z; x = 0; xd = -1; xp = -1; aD = 1; dD = 1; aP = 1;\n"
        "D: at dD after (time >= aD), fromTrigger=false: x = 1, xd = x;\n"
        "P: at (time >= aP), fromTrigger=false: x = 2, xp = x;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    for name, value in {
        "aD": _trigger_time_for(T, 0.25),
        "dD": 0.25,
        "aP": _after(T, 2),
    }.items():
        model.set_param(name, value)
    sim = bngsim.Simulator(model, method="ode")
    sim.run_until(_after(T, -3), n_points=2, rtol=1e-9, atol=1e-12)
    r = sim.run_until(2.3, n_points=2, rtol=1e-9, atol=1e-12)
    assert (_last(r, "xd"), _last(r, "xp"), _last(r, "x")) == (0.0, 1.0, 2.0)


@pytest.mark.parametrize("gap", [110, 150, 1000, 20000])
@pytest.mark.parametrize("T", [1e-3, 0.3])
def test_a_stop_many_doubles_ahead_of_a_time_near_zero_is_not_yet_reached(T, gap):
    """A delayed execution on T, an event P on a stop `gap` doubles later
    (priority 1) and a delayed execution D three doubles after that (priority
    10). Times under 1 were compared by the doubles of 1, so the return at T
    took the stop as reached; P was then left to the root search, which
    returned it past D's time, and the two were ordered by priority."""
    stop = _after(T, gap)
    due = _after(stop, 3)
    delay = T / 4.0
    params = {
        "a0": _trigger_time_for(T, 3.0 * T / 8.0),
        "d0": 3.0 * T / 8.0,
        "aP": stop,
        "aD": _trigger_time_for(due, delay),
        "dD": delay,
    }
    text = _THREE.replace("0.3*Z", "Z")
    r = _run(text, [0.0, 2.0 * T], params, rtol=1e-9, atol=1e-12)
    assert (_last(r, "xp"), _last(r, "xd"), _last(r, "x")) == (0.0, 1.0, 2.0)


@pytest.mark.parametrize("t_end", [10.0, 1e4, 1e6])
def test_how_far_ahead_an_execution_is_stopped_for_does_not_grow_with_the_run(t_end):
    """P sets X = 1 at t = 1 and D reads it 1e-12 later, with X decaying at
    1e11: y = exp(-0.1). The least distance ahead that a due execution is
    stopped for was taken on the scale of the run's length, and over a run of
    1e6 D ran 1.8e-12 after P and read 0.837."""
    text = (
        "species X = 0; JX: X -> ; k*X; k = 1e11; y = -1; tau = 1; dly = 1e-12;\n"
        "P: at (time >= tau), fromTrigger=false: X = 1;\n"
        "D: at dly after (time >= tau), fromTrigger=false: y = X;\n"
    )
    r = _run(text, [0.0, t_end], rtol=1e-10, atol=1e-14)
    assert _last(r, "y") == pytest.approx(math.exp(-0.1), rel=2e-3)


# ── What the fifth review found ─────────────────────────────────────────────
@pytest.mark.parametrize("T, after", [(0.013, 1065), (1.3e-6, 10_000_000), (13.0, 1065)])
def test_a_stop_is_not_merged_into_one_the_loop_takes_for_another_instant(T, after):
    """An event P on T, a delayed execution D due 65 doubles later (priority
    above P's), and an unrelated event Q `after` doubles past T, a distance
    that is under 68 doubles of 1 and far more than 68 doubles of T. The
    stops of P and Q were merged as one instant and Q's kept, which the loop
    does not take for P's; P was left to the root search, came back on D's
    time, and ran after it by priority."""
    text = (
        "species Z = 1; JZ: Z -> ; kz*Z; kz = 1; x = 0; y = 0; xp = -1; xd = -1;\n"
        "aP = 1; aD = 1; dD = 1; aQ = 1;\n"
        "P: at (time >= aP), priority = 1, fromTrigger=false: x = 1, xp = x;\n"
        "D: at dD after (time >= aD), priority = 10, fromTrigger=false: x = 2, xd = x;\n"
        "Q: at (time >= aQ): y = 1;\n"
    )
    due = _after(T, 65)
    params = {
        "kz": 0.3 / T,
        "aP": T,
        "aD": _trigger_time_for(due, 0.25 * T),
        "dD": 0.25 * T,
        "aQ": _after(T, after),
    }
    r = _run(text, [0.0, _after(T, -60), due, 2.0 * T], params, rtol=1e-9, atol=1e-12)
    assert (_last(r, "xp"), _last(r, "xd"), _last(r, "x")) == (0.0, 1.0, 2.0)
    assert _col(r, "x")[2] == 2.0


@pytest.mark.parametrize(
    "priorities, order",
    [(("", ""), (0.0, 2.0, 1.0)), ((10, 1), (0.0, 2.0, 1.0)), ((1, 10), (1.0, 0.0, 2.0))],
    ids=["none", "P>D", "D>P"],
)
@pytest.mark.parametrize("T", [0.6, 1.3, 7.0])
def test_a_trigger_an_earlier_execution_opens_rises_with_the_root(T, priorities, order):
    """A leg ends five doubles before T. G is due three doubles before T and
    sets g = 1; P is `at (time >= T && g > 0.5)`; D is due on T. The next leg
    returns a root on T with G still waiting. G runs first, at its own time,
    and opens P's trigger, which the step never saw change sign: P was not
    in the root's batch, and ran after D whatever their priorities."""
    pp, pd = (f", priority = {p}" if p != "" else "" for p in priorities)
    text = (
        "species Z = 1; JZ: Z -> ; kz*Z; kz = 1; x = 0; g = 0; xp = -1; xd = -1;\n"
        "aP = 1; aG = 1; dG = 1; aD = 1; dD = 1;\n"
        "G: at dG after (time >= aG), fromTrigger=false: g = 1;\n"
        f"P: at (time >= aP && g > 0.5){pp}, fromTrigger=false: x = 2, xp = x;\n"
        f"D: at dD after (time >= aD){pd}, fromTrigger=false: x = 1, xd = x;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    for name, value in {
        "kz": 0.3 / T,
        "aP": T,
        "aG": _trigger_time_for(_after(T, -3), 0.25 * T),
        "dG": 0.25 * T,
        "aD": _trigger_time_for(T, 0.375 * T),
        "dD": 0.375 * T,
    }.items():
        model.set_param(name, value)
    sim = bngsim.Simulator(model, method="ode")
    sim.run_until(_after(T, -5), n_points=2, rtol=1e-9, atol=1e-12)
    r = sim.run_until(2.0 * T, n_points=2, rtol=1e-9, atol=1e-12)
    assert (_last(r, "xp"), _last(r, "xd"), _last(r, "x")) == order


def test_two_events_a_few_denormals_apart_are_one_instant():
    """`time >= 0` and `time >= 7.4e-322`. Measured by doubles of the time
    alone they were two instants, the step from one to the other underflowed
    in the integrator, and the second event never fired, with no error."""
    text = (
        "species Z = 1; JZ: Z -> ; 0.3*Z; n1 = 0; n2 = 0; a2 = 1;\n"
        "E1: at (time >= 0): n1 = n1 + 1;\n"
        "E2: at (time >= a2): n2 = n2 + 1;\n"
    )
    r = _run(text, [-1.0, 0.0, 1.0], {"a2": _after(0.0, 150)}, rtol=1e-9, atol=1e-12)
    assert (_last(r, "n1"), _last(r, "n2")) == (1.0, 1.0)
    assert _last(r, "Z") == pytest.approx(math.exp(-0.6), rel=1e-7)


def test_a_grid_that_names_each_time_twice_gets_each_row_twice():
    """Ten thousand stops under a grid with every time given twice. The row
    reached without a step asked every stop left in the run whether it lay
    before it (eight times the run time); the rows are the same pairs."""
    text = (
        "species X = 1; J: X -> ; 0.001*X; n = 0;\n"
        "E: at (time - floor(time) >= 0.5), fromTrigger=false: n = n + 1;\n"
    )
    times = np.repeat(np.linspace(0.0, 2000.0, 8001), 2)
    r = _run(text, times.tolist())
    n = np.array(_col(r, "n"))
    assert np.array_equal(n[0::2], n[1::2])
    assert n[-1] == 2000.0
    assert np.array_equal(np.asarray(r.species)[0::2], np.asarray(r.species)[1::2])


# ── What the sixth review found ─────────────────────────────────────────────
@pytest.mark.parametrize("t0", [0.013, 1.0, 300.0])
def test_a_trigger_an_execution_opens_is_not_the_start_read_again(t0):
    """A leg starts at t0 with A, `at (g*(time - a) > 0)`, exactly on its
    threshold (g = 0), which marks it as resting there. G is due three
    doubles later and sets g = 1; a root comes three doubles after that. A's
    rise is G's write and the clock, with no root of its own: read through
    the mark, which is for a root at the start, it was refused and never
    fired."""
    t_g = _after(t0, 3)
    text = (
        "species Z = 1; JZ: Z -> ; 0.1*Z; g = 0; na = 0; nb = 0; a = 1; aG = 1; dG = 1; T = 1;\n"
        "G: at dG after (time >= aG), fromTrigger=false: g = 1;\n"
        "A: at (g*(time - a) > 0), fromTrigger=false: na = na + 1;\n"
        "B: at (time >= T), fromTrigger=false: nb = nb + 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    delay = 0.25 * t0
    for name, value in {
        "a": t_g,
        "aG": _trigger_time_for(t_g, delay),
        "dG": delay,
        "T": _after(t0, 6),
    }.items():
        model.set_param(name, value)
    sim = bngsim.Simulator(model, method="ode")
    sim.run_until(t0, n_points=2, rtol=1e-9, atol=1e-12)
    r = sim.run_until(2.0 * t0, n_points=2, rtol=1e-9, atol=1e-12)
    assert (_last(r, "g"), _last(r, "na"), _last(r, "nb")) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "a1, a2, times",
    [
        (1e-310, 1e-100, [0.0, 1.0]),
        (1e-200, 1e-110, [0.0, 1e-200, 1.0]),
        (5e-324, 4.94e-304, [0.0, 1.0]),
        (1e-310, 0.5, [0.0, 1.0]),
    ],
    ids=["a_denormal", "a_sample_on_it", "two_under_the_floor", "alone"],
)
def test_a_stop_nearer_zero_than_the_integrator_steps_is_taken_where_the_run_stands(a1, a2, times):
    """An event at 1e-310 beside one at 1e-100. The two are not one instant,
    so each has its stop, and from t = 0 the integrator's first step to the
    first underflows: it reported the output as reached with no step taken,
    neither event fired, and Z(1) came back as Z(0), with no error."""
    text = (
        "species Z = 1; JZ: Z -> ; 0.3*Z; n1 = 0; n2 = 0; a1 = 1; a2 = 1;\n"
        "E1: at (time >= a1): n1 = n1 + 1;\n"
        "E2: at (time >= a2): n2 = n2 + 1;\n"
    )
    r = _run(text, times, {"a1": a1, "a2": a2}, rtol=1e-9, atol=1e-12)
    assert (_last(r, "n1"), _last(r, "n2")) == (1.0, 1.0)
    assert _last(r, "Z") == pytest.approx(math.exp(-0.3), rel=1e-7)


# ── What the seventh review found ───────────────────────────────────────────
@pytest.mark.parametrize("due_after", [1, 3, 5])
def test_an_execution_due_just_into_a_leg_does_not_fire_an_event_resting_at_its_start(due_after):
    """`E0: at (S > 0)` with S = 0 and S' = 0 at the start of the second leg:
    resting on its threshold, by the rule for a trigger that does not leave it
    at first order (issue #340; what it should do is issue #934). An unrelated
    delayed execution due a few doubles into the leg is read as a stop is,
    with every trigger, and that read fired E0, which the same leg without
    the execution never fires."""
    t0 = 0.3
    due = _after(t0, due_after)
    text = (
        "species S = 0, A = 0; JS: -> S; A; JA: -> A; piecewise(0, time <= 0.3, 1);\n"
        "fired = 0; q = 0; aD = 1; dD = 1;\n"
        "E0: at (S > 0): fired = fired + 1;\n"
        "D: at dD after (time >= aD): q = q + 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("dD", 0.2)
    model.set_param("aD", _trigger_time_for(due, 0.2))
    sim = bngsim.Simulator(model, method="ode")
    sim.run_until(t0, n_points=2, rtol=1e-9, atol=1e-12)
    r = sim.run_until(1.0, n_points=2, rtol=1e-9, atol=1e-12)
    assert (_last(r, "q"), _last(r, "fired")) == (1.0, 0.0)


@pytest.mark.parametrize(
    "a1, times",
    [
        (5e-324, [-1.0, 0.0, 1.0]),
        (0.0, [-1.0, -5e-324, 1.0]),
        (1e-200, [-1e-130, 0.0, 1.0]),
    ],
    ids=["a_denormal_after_a_sample", "a_sample_a_denormal_before", "from_just_under_zero"],
)
def test_a_row_read_behind_a_root_does_not_fire_its_event_again(a1, times):
    """`at (time >= 5e-324)` with a sample on 0. The distance from the sample
    to the event underflows in the integrator's own bookkeeping, it searches
    for roots past the sample, and the event fires with the run standing past
    the row. The row is then read at its own, earlier time, where the trigger
    is false; written down as fallen, it rose again at the next read and the
    event fired twice."""
    text = "species Z = 1; JZ: Z -> ; 0.3*Z; n1 = 0; a1 = 1;\nE1: at (time >= a1): n1 = n1 + 1;\n"
    r = _run(text, times, {"a1": a1}, rtol=1e-9, atol=1e-12)
    assert _col(r, "n1")[-1] == 1.0


# ── What the eighth review found ────────────────────────────────────────────
@pytest.mark.parametrize("t0", [0.3, 5.0])
@pytest.mark.parametrize("later", [1, 10, 100])
@pytest.mark.parametrize("into", [3, 10, 40])
def test_a_write_ends_the_rest_of_a_trigger_at_the_start(t0, into, later):
    """A, `at (g*(time - a) > 0)`, rests on its threshold at the start with
    g = 0. G sets g = 1 a few doubles into the run and a root or a stop
    follows. The mark for a trigger that rests at the start is for a model
    that has not moved: kept through G's write, it refused A at the read of
    the stop, and A, opened by that write and the clock, never fired."""
    t_g = _after(t0, into)
    text = (
        "species Z = 1; JZ: Z -> ; 0.1*Z; g = 0; na = 0; nb = 0; a = 1; t0 = 0; dG = 1; T = 1;\n"
        "G: at dG after (time >= t0), t0=false, fromTrigger=false: g = 1;\n"
        "A: at (g*(time - a) > 0), fromTrigger=false: na = na + 1;\n"
        "B: at (time >= T), fromTrigger=false: nb = nb + 1;\n"
    )
    params = {"t0": t0, "dG": t_g - t0, "a": t_g, "T": _after(t_g, later)}
    r = _run(text, [t0, t0 + 1.0], params, rtol=1e-9, atol=1e-12)
    assert (_last(r, "g"), _last(r, "na"), _last(r, "nb")) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("ts", [0.3, 9.1, 3.0e6])
@pytest.mark.parametrize("before, after", [(9, 17), (40, 40), (80, 3), (3, 80)])
def test_a_fall_just_before_a_sample_is_read_at_the_sample(ts, before, after):
    """W, `at (time < b || time >= c)`, falls a few doubles before a sample
    and rises a few after it, with another event's stop on the sample. The
    row is read behind where the run stands, and W reads false there having
    been true: its fall, not yet read. Taken for a rise that came after the
    row, the fall was dropped, and at c W went from true to true: no event."""
    text = (
        "species Z = 1; JZ: Z -> ; 0.04*Z; nw = 0; n1 = 0; b = 1; c = 1; a1 = 1; b1 = 2;\n"
        "W: at (time < b || time >= c): nw = nw + 1;\n"
        "E1: at (time >= a1 && time < b1): n1 = n1 + 1;\n"
    )
    b = _after(ts, -before)
    params = {"b": b, "c": _after(ts, after), "a1": _after(ts, -150), "b1": ts}
    r = _run(text, [0.0, _after(b, -6), ts, 1.5 * ts], params, rtol=1e-9, atol=1e-12)
    assert _last(r, "nw") == 1.0


@pytest.mark.parametrize("c", [1e-20, 1e-100, 1e-130])
def test_a_row_behind_a_root_is_no_rise_of_what_fell_there(c):
    """`W: at (time < 5e-324 || time >= c)` beside `E1: at (time >= 5e-324)`,
    with a sample on 0. Both change on the first double after the sample, and
    the root for it is returned past the sample. Read at the sample
    afterwards, W is true again, as it was before it fell: taken for a rise,
    it fired on a row before its only one, and the batch it fired in read E1
    as fallen, which then fired twice."""
    text = (
        "species Z = 1; JZ: Z -> ; 0.3*Z; nw = 0; n1 = 0; a1 = 1; c = 1;\n"
        "W: at (time < a1 || time >= c): nw = nw + 1;\n"
        "E1: at (time >= a1): n1 = n1 + 1;\n"
    )
    r = _run(text, [-1.0, 0.0, 1.0], {"a1": 5e-324, "c": c}, rtol=1e-9, atol=1e-12)
    assert (_last(r, "nw"), _last(r, "n1")) == (1.0, 1.0)
    assert _col(r, "nw")[1] == 0.0


# ── What the ninth review found ─────────────────────────────────────────────
_RESTS = (
    "species S = 0, A = 0; S' = A; A' = ka; ka = 1; fired = 0; q = 0; aU = 0;\n"
    "E0: at (S > 0), t0=false: fired = fired + 1;\n"
)


@pytest.mark.parametrize(
    "t0, span, into",
    [(0.0, 2.0, 1e-300), (0.0, 2.0, 1e-140), (0.3, 0.7, None), (300.0, 300.0, None)],
)
def test_a_write_to_what_a_resting_trigger_does_not_read_leaves_it_resting(t0, span, into):
    """E0, `at (S > 0)`, rests on its threshold at the start: S = 0 and
    S' = A = 0 there, and S leaves at second order. It does not fire (issue
    #340), alone. U sets q, which nothing reads, on the first double of the
    run. With every mark for a resting trigger dropped at any write, E0 was
    taken to rise at its first root, 6.9e-27 after the start."""
    a_u = into if into is not None else _after(t0, 1)
    text = _RESTS + "U: at (time >= aU): q = q + 1;\n"
    r = _run(text, [t0, t0 + span / 2, t0 + span], {"aU": a_u}, rtol=1e-9, atol=1e-12)
    assert (_last(r, "q"), _last(r, "fired")) == (1.0, 0.0)


def test_a_cell_that_rests_at_the_start_lives_beside_a_write_to_something_else():
    """BIOMD0000000285, issue #340's model: `PIdeath > 0` rests on its
    threshold at t = 0 and must not kill the cell there. Beside one event
    that sets a parameter nothing reads on `time >= 1e-300`, it did:
    kalive = 0 and MisP(100) = 6.0e-27 for 38.9126."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks/sbml_events/BIOMD0000000285.xml"
    if not path.exists():
        pytest.skip("BIOMD0000000285.xml is not in this checkout")
    time = (
        '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">'
        " t </csymbol>"
    )
    math_ns = 'xmlns="http://www.w3.org/1998/Math/MathML"'
    event = (
        '      <event id="Unrelated">\n'
        f"        <trigger><math {math_ns}><apply><geq/>{time}"
        '<cn type="e-notation"> 1 <sep/> -300 </cn></apply></math></trigger>\n'
        '        <listOfEventAssignments><eventAssignment variable="qq">'
        f"<math {math_ns}><cn> 1 </cn></math></eventAssignment></listOfEventAssignments>\n"
        "      </event>\n"
    )
    xml = path.read_text()
    assert "    </listOfEvents>" in xml and "    </listOfParameters>" in xml
    xml = xml.replace("    </listOfEvents>", event + "    </listOfEvents>").replace(
        "    </listOfParameters>",
        '      <parameter constant="false" id="qq" value="0"/>\n    </listOfParameters>',
    )
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode")
    r = sim.run(t_span=(0.0, 100.0), n_points=11, rtol=1e-9, atol=1e-12)
    species = dict(zip(r.species_names, np.asarray(r.species)[-1], strict=True))
    assert species["MisP"] == pytest.approx(38.912573, rel=1e-6)


@pytest.mark.parametrize("into", [1, 3, 40])
def test_a_write_two_rate_laws_upstream_of_a_resting_trigger_ends_its_rest(into):
    """E0, `at (S > 0)`, rests at the start with S' = A, A' = B and B = 0. G
    sets B = 1 a few doubles into the run: nothing the trigger reads has
    changed and nor has S', but B feeds A and A feeds S, and S leaves zero
    because of the write. E0 fires, once."""
    t0 = 0.3
    text = (
        "species S = 0, A = 0, B = 0; S' = A; A' = B; B' = 0; fired = 0; aG = 0;\n"
        "E0: at (S > 0), t0=false: fired = fired + 1;\n"
        "G: at (time >= aG): B = 1;\n"
    )
    r = _run(text, [t0, t0 + 0.5, t0 + 1.0], {"aG": _after(t0, into)}, rtol=1e-9, atol=1e-12)
    assert _last(r, "fired") == 1.0
    assert _last(r, "S") == pytest.approx(0.5, rel=1e-5)


# ── What the tenth review found ─────────────────────────────────────────────
_THROUGH_RATEOF = {
    "a rate rule": "species S = 0, A = 0, B = 0; S' = rateOf(A); A' = B; B' = 0;",
    "an assignment rule": "species S = 0, A = 0, B = 0; S' = r; r := rateOf(A); A' = B; B' = 0;",
    "a kinetic law": (
        "compartment c = 1; species S in c = 0, A in c = 0, B in c = 0;"
        " J0: -> S; rateOf(A); J1: -> A; B;"
    ),
    "two rate laws": "species S = 0, Z = 0, A = 0, B = 0; S' = Z; Z' = rateOf(A); A' = B; B' = 0;",
}


@pytest.mark.parametrize("how", sorted(_THROUGH_RATEOF))
@pytest.mark.parametrize("t0, into", [(0.0, None), (0.3, 1), (0.3, 40)])
def test_a_write_that_feeds_a_resting_trigger_through_rateof_ends_its_rest(how, t0, into):
    """E0, `at (S > 0)`, rests at the start with S' = rateOf(A), A' = B and
    B = 0. G sets B = 1 a few doubles into the run, and S' is 1 from then
    on: E0 fires, once, as in libRoadRunner. The Jacobian's pattern does not
    follow a rate law through a rateOf it reads, B was not seen to feed S,
    and E0 kept its rest and never fired."""
    a_g = 1e-300 if into is None else _after(t0, into)
    text = (
        _THROUGH_RATEOF[how]
        + " fired = 0; aG = 0;\nE0: at (S > 0), t0=false: fired = fired + 1;\n"
        + "G: at (time >= aG): B = 1;\n"
    )
    r = _run(text, [t0, t0 + 0.5, t0 + 1.0], {"aG": a_g}, rtol=1e-9, atol=1e-12)
    assert _last(r, "fired") == 1.0


@pytest.mark.parametrize("t0, span", [(0.0, 2.0), (0.3, 0.7), (300.0, 300.0)])
def test_a_trigger_on_a_rateof_rests_beside_a_write_to_something_else(t0, span):
    """E0, `at (rateOf(S) > 0)`, rests at the start: S' = A = 0 there, and A
    leaves zero at second order. A trigger that reads a rateOf was taken to
    read every species, U's write to q to feed it, and E0 fired 6.9e-27
    after the start, where it does not fire alone."""
    a_u = 1e-300 if t0 == 0.0 else _after(t0, 1)
    text = (
        "species S = 0, A = 0, Bb = 0; S' = A; A' = Bb; Bb' = kb; kb = 1; q = 0; fired = 0;"
        " aU = 0;\nE0: at (rateOf(S) > 0), t0=false: fired = fired + 1;\n"
    )
    alone = _run(text, [t0, t0 + span / 2, t0 + span], {}, rtol=1e-9, atol=1e-12)
    beside = _run(
        text + "U: at (time >= aU): q = q + 1;\n",
        [t0, t0 + span / 2, t0 + span],
        {"aU": a_u},
        rtol=1e-9,
        atol=1e-12,
    )
    assert (_last(alone, "fired"), _last(beside, "fired"), _last(beside, "q")) == (0.0, 0.0, 1.0)


# ── What the eleventh review found ──────────────────────────────────────────
_RESTS_AT_THIRD_ORDER = (
    "species S = 0, A = 0, Bb = 0, W = 0, V = 0, Y = 0; S' = A; A' = Bb; Bb' = kb; kb = 1;"
    " V' = W; W' = 0; z = 0; fired = 0; aU = 0;\n"
)


@pytest.mark.parametrize(
    "rateof",
    ["Y' = rateOf(V);\n", "Y' = 0;\nZ: at (time >= 0.9): z = rateOf(V);\n"],
    ids=["in a rate rule", "in an assignment"],
)
@pytest.mark.parametrize("t0, span", [(0.0, 2.0), (0.3, 0.7), (300.0, 300.0)])
def test_a_rateof_elsewhere_does_not_make_every_write_feed_every_trigger(rateof, t0, span):
    """E0, `at (S > 0)`, rests at the start. U adds to W, which only V' reads,
    on the first double of the run; the model has a rateOf(V) in it, in a
    rate rule or in another event's assignment, and nothing of S's reads
    it. Taken to feed every trigger because the model uses a rateOf, the
    write ended E0's rest, and E0 fired 6.9e-27 after the start."""
    a_u = 1e-300 if t0 == 0.0 else _after(t0, 1)
    text = (
        _RESTS_AT_THIRD_ORDER
        + rateof
        + "E0: at (S > 0), t0=false: fired = fired + 1;\nU: at (time >= aU): W = W + 1;\n"
    )
    r = _run(text, [t0, t0 + span / 2, t0 + span], {"aU": a_u}, rtol=1e-9, atol=1e-12)
    assert (_last(r, "W"), _last(r, "fired")) == (1.0, 0.0)


@pytest.mark.parametrize("t0, span", [(0.0, 2.0), (0.3, 0.7)])
def test_a_write_to_the_species_itself_does_not_feed_its_rateof(t0, span):
    """E0, `at (rateOf(S) > 0)` with S' = A, rests at the start. U writes S,
    which S' does not read: rateOf(S) is what it was."""
    a_u = 1e-300 if t0 == 0.0 else _after(t0, 1)
    text = (
        "species S = 0, A = 0, Bb = 0; S' = A; A' = Bb; Bb' = kb; kb = 1; fired = 0; aU = 0;\n"
        "E0: at (rateOf(S) > 0), t0=false: fired = fired + 1;\n"
        "U: at (time >= aU): S = S + 1;\n"
    )
    r = _run(text, [t0, t0 + span / 2, t0 + span], {"aU": a_u}, rtol=1e-9, atol=1e-12)
    assert _last(r, "fired") == 0.0


_NESTED_RATEOF = {
    "two deep": ("species S = 0, A = 0, B = 0; S' = rateOf(A); A' = B; B' = 0;", "B"),
    "through a rule": (
        "species S = 0, A = 0, B = 0; S' = r; r := rateOf(A); A' = B; B' = 0;",
        "B",
    ),
    "three deep": (
        "species S = 0, A = 0, B = 0, D = 0; S' = rateOf(A); A' = rateOf(B); B' = D; D' = 0;",
        "D",
    ),
}


@pytest.mark.parametrize("how", sorted(_NESTED_RATEOF))
@pytest.mark.parametrize("t0, a_g", [(0.0, 0.25), (0.3, 0.7)])
def test_a_trigger_on_a_nested_rateof_rises_with_the_write_that_opens_it(how, t0, a_g):
    """E0, `at (rateOf(S) > 0.5)` with S' = rateOf(A) and A' = B. G sets
    B = 1 from a delay, at a stop, and rateOf(S) is 1 from then on: E0 fires
    there, as in libRoadRunner. One refresh of the rateOf buffer
    reads what the refresh before it left, so after the write rateOf(S) was
    a write behind: false at the stop, true at the restart, where a trigger
    already true is no rise, and E0 never fired."""
    dynamics, written = _NESTED_RATEOF[how]
    text = (
        dynamics
        + " fired = 0; tf = -1; aG = 1;\n"
        + "E0: at (rateOf(S) > 0.5), t0=false: fired = fired + 1, tf = time;\n"
        + f"G: at 0.1 after (time >= aG - 0.1): {written} = 1;\n"
    )
    r = _run(text, [t0, t0 + 0.5, t0 + 1.0], {"aG": a_g}, rtol=1e-9, atol=1e-12)
    assert _last(r, "fired") == 1.0
    assert _last(r, "tf") == pytest.approx(a_g, abs=1e-12)


# ── What the twelfth review found ───────────────────────────────────
@pytest.mark.parametrize("index", ["time", "kb", "V"])
@pytest.mark.parametrize("t0, span", [(0.0, 2.0), (0.3, 0.7), (300.0, 300.0)])
def test_a_table_function_does_not_make_every_write_feed_every_trigger(index, t0, span):
    """The model of the eleventh review's control with a table function
    added to it that nothing reads: over the clock, over a constant, or over
    V, which U's write to W feeds. A table's index is not among the addresses
    its call reads, and with one not indexed by the clock nothing was taken
    as known of any expression: U's write ended E0's rest, and E0 fired
    6.9e-27 after the start. A table is read by the expressions that call it,
    with what its index reads."""
    a_u = 1e-300 if t0 == 0.0 else _after(t0, 1)
    text = (
        _RESTS_AT_THIRD_ORDER
        + "Y' = V;\nE0: at (S > 0), t0=false: fired = fired + 1;\n"
        + "U: at (time >= aU): W = W + 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("aU", a_u)
    model.add_table_function("g", times=[0.0, 1.0, 2.0], values=[0.0, 1.0, 2.0], index=index)
    r = bngsim.Simulator(model, method="ode").run(
        sample_times=[t0, t0 + span / 2, t0 + span], rtol=1e-9, atol=1e-12
    )
    assert (_last(r, "W"), _last(r, "fired")) == (1.0, 0.0)
