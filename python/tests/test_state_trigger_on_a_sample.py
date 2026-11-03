"""An event whose trigger on the state crosses on an output sample fires in
that sample's row (issue #1124, item 2).

``q' = 1; at (q >= 2.5): S = 1`` with a sample at 2.5: CVODE reaches
q(2.5) = 2.4999999999999996, a few doubles short, so the trigger read false
at the row, the root was found a few doubles after it, and the row held the
state from before the event ([S] = 3; libRoadRunner and SSA give 1). The root
search places a crossing no closer than its own tolerance, so a root that
near is not after the row by anything the run can tell. The triggers at an
output are now also read with the state half that tolerance ahead, and the
clock at the row.

A run with sensitivity columns does the same (issue #1182), and takes the
event's jump in its columns there as at a located crossing: the row is t*,
within half a root tolerance of it. It had waited for the next step's root
search, and recorded the row before the event that the run without columns
recorded after it.

A trigger on the clock is exact, and keeps #950's answer: an event a double
after a sample leaves that row before it.

Oracles: closed forms. q = t, so the event is at t = 2.5; its count of
executions; the columns' closed forms through t* = 2.5/k; and for the clock,
the order of two doubles.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

_RISE = "species S = 3; q = 0; q' = 1; E1: at (q {op} 2.5): S = 1;"

_GRIDS = {
    "a_sample_on_it": [0.0, 1.0, 2.0, 2.5, 3.0],
    "only_it": [0.0, 2.5],
    "half_steps": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "linspace": list(np.linspace(0.0, 5.0, 11)),
}


def _column(text, times, name, params=None, **kw):
    model = bngsim.Model.from_antimony_string(text)
    for key, value in (params or {}).items():
        model.set_param(key, value)
    r = bngsim.Simulator(model, method="ode").run(sample_times=times, **kw)
    return [float(v) for v in np.asarray(r.species)[:, list(r.species_names).index(name)]]


def _row(times, t):
    return [float(x) for x in times].index(t)


@pytest.mark.parametrize("grid", sorted(_GRIDS))
def test_the_row_on_the_crossing_holds_the_event(grid):
    """The issue's repro, on four grids. Each was 3 at t = 2.5."""
    times = _GRIDS[grid]
    s = _column(_RISE.format(op=">="), times, "S")
    at = _row(times, 2.5)
    assert s[:at] == [3.0] * at
    assert s[at:] == [1.0] * (len(times) - at)


def test_it_agrees_with_ssa():
    times = _GRIDS["a_sample_on_it"]
    model = bngsim.Model.from_antimony_string(_RISE.format(op=">="))
    r = bngsim.Simulator(model, method="ssa").run(sample_times=times, seed=1)
    ssa = [float(v) for v in np.asarray(r.species)[:, list(r.species_names).index("S")]]
    assert _column(_RISE.format(op=">="), times, "S") == ssa


def test_it_fires_once():
    """Fired at the row, the root a few doubles later is behind the run."""
    text = "species n = 0; q = 0; q' = 1; E1: at (q >= 2.5): n = n + 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0, 4.0], "n") == [0, 0, 0, 1, 1, 1]


def test_a_falling_trigger():
    text = "species S = 3; q = 5; q' = -1; E1: at (q <= 2.5): S = 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S") == [3, 3, 3, 1, 1]


def test_a_delayed_execution_runs_its_delay_from_the_row():
    text = "species S = 3; q = 0; q' = 1; E1: at 1 after (q >= 2.5): S = 1;"
    assert _column(text, [0.0, 2.5, 3.4, 3.5, 4.0], "S") == [3, 3, 3, 1, 1]


def test_at_a_tight_tolerance_as_well():
    s = _column(_RISE.format(op=">="), _GRIDS["a_sample_on_it"], "S", rtol=1e-10, atol=1e-12)
    assert s == [3, 3, 3, 1, 1]


def test_an_event_on_the_row_that_turns_the_state_back_is_read_ahead_from():
    """An event on a stop at the sample turns q back before it reaches 2.5.

    q tops out a few doubles short of 2.5 and falls, so E1 never fires. Read
    ahead on the derivative from before E0 (q' = +1), q crossed 2.5 and E1 fired
    in the row.
    """
    text = (
        "species S = 3; q = 0; k = 1; q' = k;"
        " E0: at (time >= 2.5): k = -1; E1: at (q >= 2.5): S = 1;"
    )
    for times in ([0.0, 1.0, 2.0, 2.5, 3.0], [0.0, 2.5, 3.0]):
        assert _column(text, times, "S") == [3.0] * len(times)
        assert max(_column(text, times, "q")) < 2.5


def test_a_crossing_after_the_sample_by_more_than_rounding_is_after_it():
    """1e-9 after the sample is far past the root tolerance: the row is 3."""
    text = "species S = 3; q = 0; q' = 1; E1: at (q >= 2.5 + 1e-9): S = 1;"
    assert _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S") == [3, 3, 3, 3, 1]


@pytest.mark.parametrize("ulps", [1, 8])
def test_a_clock_trigger_a_double_after_the_sample_is_after_it(ulps):
    """Only the state is read ahead: #950's row before the event stands."""
    tau = 2.5
    for _ in range(ulps):
        tau = math.nextafter(tau, math.inf)
    text = "species S = 3; tau = 0; E1: at (time >= tau): S = 1;"
    s = _column(text, [0.0, 1.0, 2.0, 2.5, 3.0], "S", params={"tau": tau})
    assert s == [3, 3, 3, 3, 1]


# A run with sensitivity columns fires at the row too, with the jump of a
# located crossing (issue #1182). Before, its fire waited for the next step's
# root search, a few doubles later, and the row held the state from before the
# event. `S' = 1` and `S = S/2` at q = k·t = 2.5 give closed forms: t* = 2.5/k,
# so past it S = (3 + t*)/2 + (t - t*), dS/dk = 2.5/(2k²) = 1.25 at k = 1,
# dS/dq0 = 0.5 (t* = (2.5 - q0)/k) and dS/dS0 = 0.5. Before it, 0, 0 and 1.
_HALVE = "species S = 3; q = 0; k = 1; q' = k; S' = 1; E1: at (q {op} 2.5): S = S/2;"


def _sens_run(text, times, **kw):
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode", **kw).run(sample_times=times)
    j = list(r.species_names).index("S")
    s = [float(v) for v in np.asarray(r.species)[:, j]]
    return r, j, s


@pytest.mark.parametrize("grid", sorted(_GRIDS))
def test_a_sensitivity_run_agrees_with_a_plain_run_at_the_row(grid):
    """The same model and grid, with and without sensitivities, at the crossing row."""
    text = "species S = 3; q = 0; k = 1; q' = k; E1: at (q >= 2.5): S = 1;"
    times = _GRIDS[grid]
    _, _, s = _sens_run(text, times, sensitivity_params=["k"])
    np.testing.assert_allclose(s, _column(text, times, "S"), rtol=1e-12)
    assert s == [3.0 if t < 2.5 else 1.0 for t in times]


@pytest.mark.parametrize("op", [">=", ">"])
def test_a_sensitivity_run_jumps_its_columns_at_the_row(op):
    text = _HALVE.format(op=op)
    times = [0.0, 1.0, 2.0, 2.5, 3.0]
    r, j, s = _sens_run(text, times, sensitivity_params=["k"], sensitivity_ic=["q", "S"])
    np.testing.assert_allclose(s, _column(text, times, "S"), rtol=1e-12)
    np.testing.assert_allclose(s, [3, 4, 5, 2.75, 3.25], rtol=1e-6)
    after = np.array(times) >= 2.5
    dk = np.asarray(r.sensitivities)[:, j, 0]
    np.testing.assert_allclose(dk, np.where(after, 1.25, 0.0), atol=1e-6)
    dic = np.asarray(r.sensitivities_ic)[:, j, :]
    np.testing.assert_allclose(dic[:, 0], np.where(after, 0.5, 0.0), atol=1e-6)
    np.testing.assert_allclose(dic[:, 1], np.where(after, 0.5, 1.0), atol=1e-6)


def test_a_sensitivity_run_from_a_later_start_fires_at_the_row():
    """From t = 1 the crossing is at 3.5: t* = 1 + 2.5/k."""
    times = [1.0, 2.0, 3.5, 4.0]
    r, j, s = _sens_run(
        _HALVE.format(op=">="), times, sensitivity_params=["k"], sensitivity_ic=["q"]
    )
    np.testing.assert_allclose(s, _column(_HALVE.format(op=">="), times, "S"), rtol=1e-12)
    np.testing.assert_allclose(s, [3, 4, 2.75, 3.25], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(r.sensitivities)[:, j, 0], [0, 0, 1.25, 1.25], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(r.sensitivities_ic)[:, j, 0], [0, 0, 0.5, 0.5], atol=1e-6
    )


def test_two_events_on_one_crossing_at_the_row_fire_together():
    """Both rise on q = 2.5; E2 reads S from before the batch (3 + 2.5)."""
    text = (
        _HALVE.format(op=">=").replace("species S = 3;", "species S = 3, R = 1;")
        + " E2: at (q >= 2.5): R = R + S;"
    )
    times = [0.0, 2.5, 3.0]
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(sample_times=times)
    names = list(r.species_names)
    x = np.asarray(r.species)
    plain = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode").run(
        sample_times=times
    )
    np.testing.assert_allclose(x, np.asarray(plain.species), rtol=1e-12)
    np.testing.assert_allclose(x[:, names.index("R")], [1, 6.5, 6.5], rtol=1e-6)
    # R = 1 + 3 + t*: dR/dk = dt*/dk = -2.5
    np.testing.assert_allclose(
        np.asarray(r.sensitivities)[:, names.index("R"), 0], [0, -2.5, -2.5], atol=1e-6
    )


def test_a_falling_trigger_at_the_row_agrees_with_a_plain_run():
    text = "species S = 3; q = 5; k = 1; q' = -k; S' = 1; E1: at (q <= 2.5): S = S/2;"
    times = [0.0, 1.0, 2.5, 3.0]
    r, j, s = _sens_run(text, times, sensitivity_params=["k"])
    # (Off the crossing the two runs step apart by rounding: CVODE's error
    # test reads the columns too.)
    np.testing.assert_allclose(s, _column(text, times, "S"), rtol=1e-12)
    np.testing.assert_allclose(s, [3, 4, 2.75, 3.25], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(r.sensitivities)[:, j, 0], [0, 0, 1.25, 1.25], atol=1e-6)


# Issue #1182's model beside a switch on V = rate·t, which reads S, `gap`
# before the row or after it: further apart than one step's tolerances and
# within #1025's window, so the order is confirmed by a plain run at a
# hundredth of the tolerances. That run, and the one it confirms, fire the
# event at the row, and the fire there has to be counted as the root's are
# for the two to name it alike. t* = 2.5/k, S halves at t*.
_BESIDE_A_SWITCH = (
    "species S, W, V; S = 3; W = 0; V = 0; q = 0; k = 1; c = 0.7\n"
    "q' = k; S' = 1; V' = {rate}\n"
    "J2: -> W; piecewise(c*S, V >= {threshold!r}, 0)\n"
    "E1: at (q >= 2.5): S = S/2\n"
)


@pytest.mark.parametrize("rate", [0.25, 1.5, 4.0])
@pytest.mark.parametrize("gap", [5e-5, -5e-5], ids=["switch-first", "event-first"])
def test_a_fire_on_the_row_is_named_to_the_run_that_confirms_its_order(rate, gap):
    """Uncounted, the tighter run's fire at the row was not found, and the
    run was refused with 'does not fire it there' where its order is plain
    (V is exact). Switch first, W gathers c·S from 2.5 − gap and
    dW/dk = c·t*·dt*/dk = −4.375; event first, dW/dk = c·(3 − t_s)/2·2.5."""
    text = _BESIDE_A_SWITCH.format(rate=rate, threshold=rate * (2.5 - gap))
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
        sample_times=[0.0, 1.0, 2.0, 2.5, 3.0], rtol=1e-6, atol=1e-9, timeout=60
    )
    names = list(run.species_names)
    assert np.asarray(run.species)[3, names.index("S")] == pytest.approx(2.75, rel=1e-12)
    t_switch = 2.5 - gap
    truth = -0.7 * 2.5 * 2.5 if gap > 0 else 0.7 * 0.5 * (3.0 - t_switch) * 2.5
    got = np.asarray(run.sensitivities)[-1, names.index("W"), 0]
    assert got == pytest.approx(truth, rel=1e-8)


# The same, with the switch beside a second event at t2 = 2.8/k, which the
# run finds by root: its name to the tighter run is its count, and the count
# of every fire before it, the one on the row included.
_AFTER_A_ROW_FIRE = (
    "species S, W, V; S = 3; W = 0; V = 0; q = 0; k = 1; c = 0.7\n"
    "q' = k; S' = 1; V' = 1.5\n"
    "J2: -> W; piecewise(c*S, V >= {threshold!r}, 0)\n"
    "E1: at (q >= 2.5): S = S/2\nE2: at (q >= 2.8): S = S/2\n"
)


@pytest.mark.parametrize("gap", [5e-5, -5e-5], ids=["switch-first", "event-first"])
def test_a_fire_after_one_on_the_row_is_named_with_it_counted(gap):
    """With the row's fire left out of a count, the two runs disagreed on
    how many fires came before E2, and the run was refused with 'fires other
    events before it'. S halves at t1 = 2.5/k and t2 = 2.8/k. Switch first
    (t_s = t2 − gap): dW/dk = c·[(t2 − t_s)/2 + (3 − t2)/4]·2.5
    − c·[S(t2⁻)/2 − (3 − t2)/2]·t2, with S(t2⁻) = 2.75 + t2 − 2.5. Event
    first: dW/dk = c·(3 − t_s)·(0.625 + t2/2)."""
    t1, t2, t_s = 2.5, 2.8, 2.8 - gap
    model = bngsim.Model.from_antimony_string(_AFTER_A_ROW_FIRE.format(threshold=1.5 * t_s))
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
        sample_times=[0.0, 1.0, 2.0, 2.5, 3.0], rtol=1e-6, atol=1e-9, timeout=60
    )
    names = list(run.species_names)
    assert np.asarray(run.species)[3, names.index("S")] == pytest.approx(2.75, rel=1e-12)
    if gap > 0:
        before_t2 = 2.75 + t2 - t1
        truth = 0.7 * (((t2 - t_s) / 2 + (3 - t2) / 4) * t1 - (before_t2 / 2 - (3 - t2) / 2) * t2)
    else:
        truth = 0.7 * (3 - t_s) * (0.625 + t2 / 2)
    got = np.asarray(run.sensitivities)[-1, names.index("W"), 0]
    assert got == pytest.approx(truth, rel=1e-8)


# ── A row reached from short of the sample (issue #1276) ─────────────────────
#
# A sensitivity run takes a row as reached from up to 1e-12 short of it after a
# step. Where another root stops it that close before the sample (E2, on q a
# hair below 2.5), it never read ahead, and the row held the state from before
# E1, which crosses at the sample: [S] 5.5 at t = 2.4999999999999, where the run
# without columns held 2.75 at 2.5.
#
# The bounds: q and S are linear in t, so the run integrates them to rounding
# (4.9e-15 from the closed form, measured), and the run without columns reaches
# the same state at the row (2.2e-16). The columns carry CVODES' own error
# (7.4e-11, measured).

_SHORT = (
    "species S, R; S = 3; R = 0; q = 0; k = 1; q' = k; S' = 1;"
    "E1: at (q >= 2.5): S = S/2; E2: at (q >= {e2!r}): R = 1;"
)
_SHORT_TIMES = [0.0, 1.0, 2.0, 2.5, 3.0]


@pytest.mark.parametrize("off", [1e-13, 5e-13, 2e-12])
def test_a_row_reached_from_short_of_the_sample_holds_the_event(off):
    """2e-12 is outside the 1e-12 the row is taken from, and agreed before."""
    text = _SHORT.format(e2=2.5 - off)
    r, j, s = _sens_run(text, _SHORT_TIMES, sensitivity_params=["k"], sensitivity_ic=["S"])
    assert float(np.asarray(r.time)[3]) == 2.5
    np.testing.assert_allclose(s, _column(text, _SHORT_TIMES, "S"), rtol=1e-12)
    np.testing.assert_allclose(s, [3, 4, 5, 2.75, 3.25], rtol=1e-12)
    names = list(r.species_names)
    assert float(np.asarray(r.species)[3, names.index("R")]) == 1.0


@pytest.mark.parametrize("off", [1e-13, 5e-13])
def test_its_columns_jump_at_the_row(off):
    """t* = 2.5/k: S = (3 + t*)/2 + (t − t*) after it, so dS/dk = 1.25 and
    dS/dS0 = 0.5 from the row on; before it, 0 and 1."""
    r, j, _ = _sens_run(
        _SHORT.format(e2=2.5 - off), _SHORT_TIMES, sensitivity_params=["k"], sensitivity_ic=["S"]
    )
    after = np.array(_SHORT_TIMES) >= 2.5
    np.testing.assert_allclose(
        np.asarray(r.sensitivities)[:, j, 0], np.where(after, 1.25, 0.0), atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(r.sensitivities_ic)[:, j, 0], np.where(after, 0.5, 1.0), atol=1e-9
    )


# A clock trigger at a row reached short: E1 on the clock at the sample, E2 a
# hair before it on q. With q' = 1 neither fire's time moves with k (k reaches S
# only), so the two are one instant, and fire at the row as the run without
# columns fires them. S = 3 + k·t, halved by E1 at 2.5 under ``>=`` (before the
# row) or a double after it under ``>``.

_CLOCK_SHORT = (
    "species S, R; S = 3; R = 0; q = 0; k = 1; q' = {q_rate}; S' = {s_rate};"
    "E1: at (time {op} 2.5): S = S/2; E2: at (q >= {e2!r}): R = 1;"
)


@pytest.mark.parametrize("off", [1e-13, 5e-13])
@pytest.mark.parametrize(("op", "s_row", "dk_row"), [(">=", 2.75, 1.25), (">", 5.5, 2.5)])
def test_a_clock_trigger_at_a_row_reached_short(op, s_row, dk_row, off):
    """dS/dk = t until E1, halved there: 1.25 or 2.5 at the row, 1.75 at 3."""
    text = _CLOCK_SHORT.format(q_rate="1", s_rate="k", op=op, e2=2.5 - off)
    r, j, s = _sens_run(text, _SHORT_TIMES, sensitivity_params=["k"])
    assert float(np.asarray(r.time)[3]) == 2.5
    np.testing.assert_allclose(s, _column(text, _SHORT_TIMES, "S"), rtol=1e-12)
    np.testing.assert_allclose(s, [3, 4, 5, s_row, 3.25], rtol=1e-12)
    names = list(r.species_names)
    assert float(np.asarray(r.species)[3, names.index("R")]) == 1.0
    np.testing.assert_allclose(
        np.asarray(r.sensitivities)[:, j, 0], [0, 1, 2, dk_row, 1.75], atol=1e-9
    )


@pytest.mark.parametrize("off", [1e-13, 5e-13])
def test_a_clock_fire_beside_a_moving_one_at_a_row_reached_short_is_refused(off):
    """With q' = k, E2's crossing moves with k and E1's on the clock does not.
    Fired together at the row, they are refused by name (issue #49), where main
    recorded the row short of both (the #1276 defect)."""
    text = _CLOCK_SHORT.format(q_rate="k", s_rate="1", op=">=", e2=2.5 - off)
    refusal = r"(?s)'E1' and 'E2' fire at the same instant.*move differently"
    with pytest.raises(bngsim.SimulationError, match=refusal):
        _sens_run(text, _SHORT_TIMES, sensitivity_params=["k"])
