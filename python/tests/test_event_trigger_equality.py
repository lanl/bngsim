"""An event whose trigger is an equality on the clock fires at that instant,
on any output grid.

``at (time == 2)`` is true at one double. It is no ordering, so no crossing
stop was placed for it, and no root, since the boolean does not change sign
across a step that passes over 2. The event fired only where an output sample
happened to land on 2: with ``n_points=6`` over (0, 5) it fired, with
``n_points=2`` or samples at 1.9 and 3.3 it did not, and the run returned the
state of a model with no event, with no warning. The run now stops on the
instant.

The instant is the double the model's evaluator reads the equality as true
at: ``time == tau/7`` holds where the clock equals the evaluator's own
rounding of ``tau/7``, which is where the stop is put. An equality that no
double satisfies (``time - 1 == 1.3``) cannot fire, and is refused by name.

Oracle: the count of executions, from the trigger's meaning. It does not
depend on the output grid, so every grid is held to the same count.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# No sample on the instant; one on it; an uneven grid; a dense one.
_GRIDS = {
    "two_points": {"t_span": (0.0, 5.0), "n_points": 2},
    "a_sample_on_it": {"t_span": (0.0, 5.0), "n_points": 6},
    "uneven": {"sample_times": [0.0, 1.9, 3.3, 5.0]},
    "dense": {"t_span": (0.0, 5.0), "n_points": 1001},
}


def _count(text, grid, method="ode", **kw):
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method=method).run(**_GRIDS[grid], **kw)
    return float(np.asarray(r.species)[-1, list(r.species_names).index("n")])


# (model, executions by t = 5)
_TRIGGERS = {
    "time_equals_a_number": ("species n = 0; E: at (time == 2): n = n + 1;", 1),
    "written_the_other_way_round": ("species n = 0; E: at (2 == time): n = n + 1;", 1),
    "a_parameter": ("species n = 0; tau = 2.2; E: at (time == tau): n = n + 1;", 1),
    # The evaluator's 15.4/7 is one double; Python's solve of it another.
    "a_quotient": ("species n = 0; tau = 15.4; E: at (time == tau/7): n = n + 1;", 1),
    "a_sum": ("species n = 0; tau = 2.3; E: at (time == tau + 0.1): n = n + 1;", 1),
    "two_instants": ("species n = 0; E: at (time == 2 || time == 3): n = n + 1;", 2),
    "beside_a_state_condition": ("species n = 0; E: at (time == 2 && n < 1): n = n + 1;", 1),
    # True from the start, false at 2 alone, and true again: one rise.
    "not_equal": ("species n = 0; E: at (time != 2): n = n + 1;", 1),
    "delayed": ("species n = 0; E: at 1 after (time == 2): n = n + 1;", 1),
    "outside_the_run": ("species n = 0; E: at (time == 7): n = n + 1;", 0),
    "at_the_end": ("species n = 0; E: at (time == 5): n = n + 1;", 1),
    "at_the_start": ("species n = 0; E: at (time == 0), t0=false: n = n + 1;", 1),
}


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("text, fires", _TRIGGERS.values(), ids=_TRIGGERS.keys())
def test_an_equality_trigger_fires_on_every_grid(text, fires, grid):
    assert _count(text, grid) == fires


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("name", ["time_equals_a_number", "a_quotient", "two_instants"])
def test_a_stochastic_run_fires_it_too(name, grid):
    text, fires = _TRIGGERS[name]
    assert _count(text, grid, method="ssa", seed=1) == fires


# A trigger that is a window of the clock in a shape no stop resolver reads
# fired the same way, only where a sample landed inside it (issue #904).
_WINDOW_TRIGGERS = {
    # Opens at 1.7488, clear of every grid's samples: an edge a few doubles
    # after a sample is issue #950's.
    "abs_window": "abs(time - 2) < 0.2512",
    "abs_window_a_billionth_wide": "abs(time - 2) < 1e-9",
    "abs_touch": "abs(time - 2) <= 0",
    "gaussian_window": "exp(-(time - 2)^2) > 0.999999",
}


@pytest.mark.parametrize("method", ["ode", "ssa"])
@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("trigger", _WINDOW_TRIGGERS.values(), ids=_WINDOW_TRIGGERS.keys())
def test_a_trigger_on_a_window_of_the_clock_fires_on_every_grid(trigger, grid, method):
    text = f"species n = 0; E: at ({trigger}): n = n + 1;"
    assert _count(text, grid, method=method, **({"seed": 1} if method == "ssa" else {})) == 1


def test_the_state_after_the_event_is_the_events(tmp_path):
    """X decays at 0.5 and is reset to 10 at t = 2: X(5) = 10 exp(-1.5)."""
    text = "species X = 1; J: X -> ; 0.5*X; E: at (time == 2): X = 10;"
    for grid in _GRIDS:
        model = bngsim.Model.from_antimony_string(text)
        r = bngsim.Simulator(model, method="ode").run(**_GRIDS[grid], rtol=1e-10, atol=1e-12)
        assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(10 * np.exp(-1.5), rel=1e-8)


def test_the_stop_is_the_double_the_evaluator_holds_the_equality_at():
    from bngsim._switch_sensitivity import crossings_with_periodic

    model = bngsim.Model.from_antimony_string(
        "species n = 0; tau = 15.4; E: at (time == tau/7): n = n + 1;"
    )
    conds = model.time_discontinuity_conditions()
    assert conds == ("(time()==(tau/7))",)
    _, stops, _ = crossings_with_periodic(
        model._core, 0.0, 5.0, conds, touches=model._event_disc_conditions
    )
    (stop,) = stops
    assert stop.touch
    assert model._core.evaluate_over_times(conds[0], np.array([stop.time]))[0] == 1.0


@pytest.mark.parametrize(
    "trigger", ["time - 1 == 1.3", "time/3 == 0.7"], ids=["shifted", "scaled"]
)
def test_an_equality_no_double_satisfies_is_refused(trigger):
    """2.3 - 1 is 1.2999999999999998 and the next double gives
    1.3000000000000003: the two sides pass each other without meeting, the
    trigger never changes value, and the event could not fire."""
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, 5.0), n_points=6)


def test_it_is_not_refused_where_the_instant_is_outside_the_run():
    model = bngsim.Model.from_antimony_string("species n = 0; E: at (time - 1 == 1.3): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 2.0), n_points=3)
    assert float(np.asarray(r.species)[-1, 0]) == 0.0


def test_an_equality_in_a_rate_law_is_still_no_stop(tmp_path):
    """To a rate law the instant has no measure, and a restart on it would
    carry the branch that holds there over a whole step."""
    model = bngsim.Model.from_antimony_string(
        "species X = 0; J: -> X; piecewise(1e9, time == 2, 0);"
    )
    assert model.time_discontinuity_conditions() == ()
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 5.0), n_points=6)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(0.0, abs=1e-9)


# ── What the first review found ─────────────────────────────────────────────
# (model, executions by t = 5)
_MORE = {
    # The instant is wherever the evaluator's two sides meet, whatever they are.
    "pi": ("species n = 0; E: at (time == pi): n = n + 1;", 1),
    "exp_1": ("species n = 0; E: at (time == exp(1)): n = n + 1;", 1),
    "a_root_of_two": ("species n = 0; E: at (time == 2^0.5 + 1): n = n + 1;", 1),
    "a_square": ("species n = 0; E: at (time^2 == 4): n = n + 1;", 1),
    "a_product": ("species n = 0; E: at (time*time == 2.25): n = n + 1;", 1),
    "an_assigned_parameter": (
        "species n = 0; a = 0.7; b := a*3; E: at (time == b): n = n + 1;",
        1,
    ),
    "a_large_offset": ("species n = 0; E: at (time + 1e6 == 1e6 + 2): n = n + 1;", 1),
    "never": ("species n = 0; E: at (sin(time) == 2): n = n + 1;", 0),
    # An equality of step calls holds for a stretch: a window like any other.
    "a_floor": ("species n = 0; E: at (floor(time) == 3): n = n + 1;", 1),
    # From 2.262, which no grid here samples: a sample a double from the
    # stop is issue #950's.
    "a_scaled_floor": ("species n = 0; E: at (floor(time/0.377) == 6): n = n + 1;", 1),
    # An equality no double satisfies, in a trigger that can fire without it.
    "beside_an_ordering": (
        "species n = 0; E: at (time - 1 == 1.3 || time >= 4): n = n + 1;",
        1,
    ),
    "not_equal_beside_an_ordering": (
        "species n = 0; E: at (time > 3 && time - 1 != 1.3): n = n + 1;",
        1,
    ),
    "not_equal_alone": ("species n = 0; E: at (time - 1 != 1.3), t0=false: n = n + 1;", 1),
    "negated": ("species n = 0; E: at (!(time - 1 == 1.3)), t0=false: n = n + 1;", 1),
}


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("text, fires", _MORE.values(), ids=_MORE.keys())
def test_what_only_the_evaluator_places_fires_on_every_grid(text, fires, grid):
    assert _count(text, grid) == fires


# ── What the third review found ─────────────────────────────────────────────
@pytest.mark.parametrize("trigger", ["abs(time - 5) == 0.3", "max(2*time - 3, 0) == 1.6"])
def test_an_equality_over_a_call_that_holds_its_value_is_refused_where_it_cannot_hold(trigger):
    """Read as a window, an equality over `abs` or `max` that no double meets
    was a window that never opens: the event silently never fired, where the
    same equality written on `time` is refused."""
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=3)


@pytest.mark.parametrize(
    ("trigger", "fires"),
    [("abs(time - 5) == 0.25", 2), ("max(abs(time - 5), 0.5) == 0.5", 1), ("floor(time) == 3", 1)],
)
def test_an_equality_over_a_call_that_holds_its_value_fires_where_it_holds(trigger, fires):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=2)
    assert float(np.asarray(r.species)[-1, 0]) == fires


# ── What the fourth review found ────────────────────────────────────────────
# (trigger, executions by t = 10)
_PERIODIC = {
    "rem(time, 3) == 1": 4,
    "rem(time, 3) == 1.5": 3,
    "time - 3*floor(time/3) == 1": 4,
    "floor(time/3)*3 + 1 == time": 4,
    "rem(floor(time), 3) == 1": 4,
    "floor(time) - 4*floor(time/4) == 2": 3,
    "min(rem(time, 3), 1) == 0.5": 4,
}


@pytest.mark.parametrize("n_points", [2, 38])
@pytest.mark.parametrize("trigger, fires", _PERIODIC.items(), ids=range(len(_PERIODIC)))
def test_a_side_that_jumps_past_the_other_is_no_near_miss(trigger, fires, n_points):
    """`rem(time, 3) == 1` holds at 1, 4, 7 and 10. At 3, 6 and 9 the left
    side wraps from 3 to 0, past 1 without being near it. Read as two sides
    passing each other without meeting, each wrap refused the trigger."""
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=n_points)
    assert float(np.asarray(r.species)[-1, 0]) == fires


def test_a_staircase_that_never_takes_a_value_is_a_trigger_that_never_holds():
    """`floor(time/0.4) == 6.5`: no near miss, nothing to refuse, no event."""
    model = bngsim.Model.from_antimony_string(
        "species n = 0; E: at (floor(time/0.4) == 6.5): n = n + 1;"
    )
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=3)
    assert float(np.asarray(r.species)[-1, 0]) == 0.0


# ── What the fifth review found ─────────────────────────────────────────────
# Equalities no double satisfies, written with a side that is zero or with a
# subtraction that cancels. The two sides were taken to have nearly met only
# where they were within a few doubles of their own size of each other, and
# `time - 100` misses 0.3 by a rounding of 100.3, which is many doubles of
# 0.3: each of these went unrefused and never fired.
_CANCELLING = [
    ("time - 100 == 0.3", 200.0),
    ("time - 1e6 == 0.3", 2e6),
    ("time/3 - 0.1 == 0", 5.0),
    ("sin(time) == 0", 10.0),
    ("time*time - 2 == 0", 5.0),
    ("exp(time) - 3 == 0", 5.0),
    ("0.1*time - 0.23 == 0", 5.0),
    ("abs(time - 5) - 0.3 == 0", 10.0),
    ("2*sin(time) - 1 == 0", 10.0),
    ("rem(time, 3) - 1.3 == 0", 10.0),
]


@pytest.mark.parametrize("trigger, t_end", _CANCELLING, ids=range(len(_CANCELLING)))
def test_a_near_miss_is_refused_whatever_size_its_sides_are(trigger, t_end):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=3)


# The same spellings where a double does satisfy them.
_CANCELLING_AND_MET = [
    ("rem(time, 3) - 1 == 0", 10.0, 4),
    ("time - 2.3 == 0", 5.0, 1),
    ("(time - 100)*(time - 300) == 0", 400.0, 2),
    ("sqrt(time) - 1000.5 == 0", 2e6, 1),
    ("time - 0.1 - 0.2 == 0", 5.0, 1),
]


@pytest.mark.parametrize(
    "trigger, t_end, fires", _CANCELLING_AND_MET, ids=range(len(_CANCELLING_AND_MET))
)
def test_a_zero_side_that_is_met_fires(trigger, t_end, fires):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=3)
    assert float(np.asarray(r.species)[-1, 0]) == fires


# ── What the sixth review found ─────────────────────────────────────────────
# A side coarser than the clock holds one value for thousands of doubles and
# then moves by one of its own last places, past the other side. Nothing
# around the pass says that move was not a jump, and read by the jump alone
# these were taken to have one and went unrefused.
_COARSE_SIDES = [
    ("(time + 1024)*1.9 == 1946.17", 5.0),
    ("(time + 273.15)*1.8 == 493", 5.0),
    ("(time + 4100)/0.6 == 6836", 5.0),
    ("(time + 1000)*(time + 1000)/6.0 == 166900", 5.0),
]


@pytest.mark.parametrize("n_points", [2, 38])
@pytest.mark.parametrize("trigger, t_end", _COARSE_SIDES, ids=range(len(_COARSE_SIDES)))
def test_a_near_miss_of_a_side_coarser_than_the_clock_is_refused(trigger, t_end, n_points):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=n_points)


# The same kind of side where a double does give the other.
_COARSE_SIDES_MET = [
    ("(time + 1e6)/1e6 == 1.000002", 5.0),
    ("(time + 1e6)*0.3 == 300000.9", 5.0),
    ("sqrt(time + 1e6) == 1000.001", 5.0),
]


@pytest.mark.parametrize("trigger, t_end", _COARSE_SIDES_MET, ids=range(len(_COARSE_SIDES_MET)))
def test_a_side_coarser_than_the_clock_that_is_met_fires(trigger, t_end):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=3)
    assert float(np.asarray(r.species)[-1, 0]) >= 1.0


# ── What the seventh review found ───────────────────────────────────────────
# A side that jumps past the other by a few of its own last places, and equals
# it a little later. Called a near miss for being within 64 doubles of the
# other on both doubles of the jump, each was refused.
_SMALL_JUMPS = {
    "1e15 + rem(time, 3) == 1e15 + 1": 4,
    "1e16 + 2*floor(rem(time, 4)) == 1e16 + 4": 3,
    "4096 + rem(time, 3)*1e-11 == 4096.00000000001": 4,
}


@pytest.mark.parametrize("n_points", [2, 38])
@pytest.mark.parametrize("trigger, fires", _SMALL_JUMPS.items(), ids=range(len(_SMALL_JUMPS)))
def test_a_jump_of_a_few_last_places_past_the_other_side_is_no_near_miss(trigger, fires, n_points):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=n_points)
    assert float(np.asarray(r.species)[-1, 0]) == fires


# Near misses of a side that, after a subtraction, moves by hundreds of its
# own last places once in hundreds of doubles of the clock. Over the 64
# doubles beside the pass it had not moved, and within 64 of its own last
# places it had not come: neither reading refused these.
_COARSE_AFTER_A_SUBTRACTION = [
    "exp(time/1e3) - 1 == 0.0023",
    "(time + 1e3)*1e-3 - 1 == 2.3e-3",
    "sqrt(time + 1e6) - 1000 == 0.00115",
    "(time*3 + 1e4)/3 - 1e4/3 == 2.3",
]


@pytest.mark.parametrize("n_points", [2, 38])
@pytest.mark.parametrize(
    "trigger", _COARSE_AFTER_A_SUBTRACTION, ids=range(len(_COARSE_AFTER_A_SUBTRACTION))
)
def test_a_near_miss_is_told_by_where_the_difference_next_moves(trigger, n_points):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, 5.0), n_points=n_points)


# ── What the eighth review found ────────────────────────────────────────────
# The clock on both sides. Where the sides meet, their difference goes back
# and forth around zero at the rounding level, and the equality holds on a
# few of the doubles inside that. Read only at the ends of it, as a condition
# on a call that flickers is, each of these never fired.
_BOTH_SIDES_MOVE = [
    "0.5*time + 100 == time + 98",
    "time + 100 == 2*time + 96",
    "10*time == 9*time + 4",
    "time/2 + 10 == time + 8",
    "sqrt(time) + 10 == 0.5*time + 10",
    "abs((0.5*time + 100) - (time + 98)) < 1e-13",
    "(0.5*time + 100 == time + 98) || time > 100",
]


@pytest.mark.parametrize("n_points", [2, 38])
@pytest.mark.parametrize("trigger", _BOTH_SIDES_MOVE, ids=range(len(_BOTH_SIDES_MOVE)))
def test_an_equality_with_the_clock_on_both_sides_fires(trigger, n_points):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 5.0), n_points=n_points)
    assert float(np.asarray(r.species)[-1, 0]) == 1.0


# A staircase that steps past the other side and takes its value two time
# units later. Looked for out to 2^32 doubles, which at t = 1e7 is eight time
# units, the staircase's next step was found and taken for how much a coarse
# side moves, and the step past was refused as a near miss.
_STEPS_AT_LARGE_TIMES = [
    ("floor(time/2)*2 - 3*floor(time/10000005) == 10000003", 9999980.0, 10000020.0),
    ("floor(time*8)/8 - 0.1875*floor(time/300000.3) == 300000.1875", 299961.0, 300001.0),
]


@pytest.mark.parametrize(
    "trigger, lo, hi", _STEPS_AT_LARGE_TIMES, ids=range(len(_STEPS_AT_LARGE_TIMES))
)
def test_a_staircases_next_step_is_not_how_much_a_coarse_side_moves(trigger, lo, hi):
    model = bngsim.Model.from_antimony_string(f"species n = 0; E: at ({trigger}): n = n + 1;")
    r = bngsim.Simulator(model, method="ode").run(t_span=(lo, hi), n_points=3)
    assert float(np.asarray(r.species)[-1, 0]) == 1.0
