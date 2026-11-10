"""An event on ``time == c``, where an event writes c, fires on every grid
(issue #1519).

An event-written parameter is held in a species slot, so it is no run
constant, and the stop that makes ``time == c`` fire (issue #1015's, for a
constant c) was not placed: the resolver declined the equality, and the
per-stretch reading of #1244 left event triggers out. No root finds an
instant either, so the event fired only where an output sample, or a point
of the SSA's trigger-probe lattice, landed on it. A dose counter
``at (time == c && c < 91): c = c + 1`` from c = 0.5 gave all 10 doses over
(0, 10) at ``n_points=101`` and none at 100 or 102, with nothing said;
BIOMD0000000620 (Palmer2014) is that model.

The instants are now placed stretch by stretch between the executions of the
events that write c, the trigger's own included, taken in time order: those
that fire at a fixed time, on such an instant, or on a flag only events set,
and set values known before the run. Where they do not, the run is refused
by name.

Oracle: the count of executions, from the trigger's meaning. ``time == c``
turns true at t = c, so the counter's instants are 0.5, 1.5, ..., 9.5, and
the count does not depend on the output grid or the engine.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest

_SPAN = (0.0, 10.0)
# No sample on any instant; the grids either side of the one with a sample on
# every half-integer, which was main's only right answer; an uneven one.
_GRIDS = {
    "two_points": {"t_span": _SPAN, "n_points": 2},
    "n_points_100": {"t_span": _SPAN, "n_points": 100},
    "n_points_102": {"t_span": _SPAN, "n_points": 102},
    "uneven": {"sample_times": [0.0, 1.9, 3.3, 7.04, 10.0]},
}
# The SSA probes the triggers on a lattice of a thousandth of the run, which
# over (0, 10) lands on every half-integer; over (0, 10.3) on none.
_SSA_SPAN = (0.0, 10.3)
_SSA_GRIDS = {
    "two_points": {"t_span": _SSA_SPAN, "n_points": 2},
    "n_points_100": {"t_span": _SSA_SPAN, "n_points": 100},
    "uneven": {"sample_times": [0.0, 1.9, 3.3, 7.04, 10.3]},
}

_COUNTER = "E: at (time == c && c < 91): n = n + 1, c = c + 1;"

# (model body, executions of E by t = 10, by t = 10.3 where that differs)
_CASES = {
    "the_issue_s_counter": ("var c = 0.5; " + _COUNTER, 10),
    # c is written only after its one instant, by an event at a fixed time.
    "written_after_its_instant": (
        "var c = 0.5037; F: at (time >= 5): c = 0.25; E: at (time == c): n = n + 1;",
        1,
    ),
    # Read through an assignment rule.
    "through_an_assignment_rule": (
        "var c = 0.5; b := c; E: at (time == b && c < 91): n = n + 1, c = c + 1;",
        10,
    ),
    "through_a_scaled_rule": (
        "var c = 0.25; b := 2*c; E: at (time == b): n = n + 1, c = c + 0.5;",
        10,
    ),
    # The value read through one (interval := 1*k; nxt := c + 1).
    "a_value_through_a_rule_of_constants": (
        "k = 1; interval := 1*k; var c = 0.5; E: at (time == c): n = n + 1, c = c + interval;",
        10,
    ),
    "a_value_through_a_rule_of_c": (
        "var c = 0.5; nxt := c + 1; E: at (time == c): n = n + 1, c = nxt;",
        10,
    ),
    # A species held by a rate rule of 0 is no more integrated than a parameter.
    "a_species_held_by_c_prime_zero": ("species c = 0.5; c' = 0; " + _COUNTER, 10),
    # Fired at 0.6, 1.6, ...: the instant is the trigger's, not the execution's.
    "delayed": (
        "var c = 0.5; E: at 0.1 after (time == c && c < 91): n = n + 1, c = c + 1;",
        10,
    ),
    # Delayed past the next instant: the trigger at 0.5 sets c = 1.5 at 2.5,
    # after 1.5, so the equality never holds again.
    "delayed_past_the_next_instant": (
        "var c = 0.5; E: at 2 after (time == c): n = n + 1, c = c + 1;",
        1,
    ),
    # The next dose read from the clock at the instant.
    "next_dose_from_the_clock": (
        "var c = 0.5; E: at (time == c): n = n + 1, c = time + 1;",
        10,
    ),
    "next_dose_from_the_clock_frozen_at_the_trigger": (
        "var c = 0.5; E: at 0.1 after (time == c): n = n + 1, c = time + 1;",
        10,
    ),
    # c + 0.1 rounds at every step; each instant is the double the sum makes.
    "a_step_that_rounds": (
        "var c = 0.5037; E: at (time == c): n = n + 1, c = c + 0.1;",
        95,
        98,
    ),
    # 0.1 + 0.1 + ... lands within rounding of whole numbers, where the SSA
    # keeps a count on the number itself.
    "a_step_of_a_tenth": ("var c = 0.1; E: at (time == c): n = n + 1, c = c + 0.1;", 100, 103),
    "hourly_in_days": (
        "var c = 0.041666666666666664; E: at (time == c): n = n + 1, c = c + 1/24;",
        240,
        247,
    ),
    # The guard reads a second parameter, which an event sets at 5.2.
    "a_guard_on_another_written_parameter": (
        "var c = 0.5; var d = 0; F: at (time >= 5.2): d = 5; "
        "E: at (time == c && d < 3): n = n + 1, c = c + 1;",
        5,
    ),
    # True from the start, false at c alone, and true again: one rise each time
    # c is passed, at 0.5 and at 7.
    "not_equal": (
        "var c = 0.5; E: at (time != c): n = n + 1; F: at (time >= 5): c = 7;",
        2,
    ),
    # Fires at the start too: at 0, 1, ..., 10.
    "from_the_start": (
        "var c = 0; E: at (time == c), t0=false: n = n + 1, c = c + 1;",
        11,
    ),
    # True at the start with initialValue true, so no fire there; F then adds
    # 3.5 to c at 2.2, and E fires at 3.5, 4.5, ..., 9.5.
    "true_before_the_start_then_set_later": (
        "var c = 0; E: at (time == c): n = n + 1, c = c + 1; F: at (time >= 2.2): c = c + 3.5;",
        7,
    ),
    # A first dose, or a later one (BIOMD0000000613's shape): 0.5, then c + 0.5
    # while before 9.6.
    "a_first_dose_or_a_later_one": (
        "var c = 0; E: at (time == 0.5 || (time == c + 0.5 && time < 9.6)): n = n + 1, c = c + 1;",
        10,
    ),
    # Two counters on one schedule, which set nothing the other reads.
    "two_counters_on_one_schedule": (
        "var c1 = 0.5; var c2 = 0.5; E1: at (time == c1): n = n + 1, c1 = c1 + 1; "
        "E2: at (time == c2): c2 = c2 + 1;",
        10,
    ),
    # An unrelated switch on the instant of a dose: d, which a rate law reads.
    "beside_an_unrelated_switch_on_an_instant": (
        "var c = 0.5; var d = 100; " + _COUNTER + " "
        "F: at (time >= 2.5): d = 3; species Y = 0; Y' = piecewise(1, time > d, 0);",
        10,
    ),
    # F sets c to the time it fires at, and E's trigger rises on that instant:
    # the engine fires E there, at 5, then 6, ..., 10.
    "a_fixed_event_sets_the_instant_now": (
        "var c = 100; F: at (time >= 5): c = 5; E: at (time == c): n = n + 1, c = c + 1;",
        6,
    ),
    # F executes at 4 + 1, on that double, and sets c to it: E's trigger rises
    # there, and the engine fires E at 5, then 6, ..., 10.
    "a_delayed_fixed_event_sets_the_instant_now": (
        "var c = 100; F: at 1 after (time >= 4): c = 5; E: at (time == c): n = n + 1, c = c + 1;",
        6,
    ),
    # E sets d to the instant it fires on, and G's trigger rises on it, so G
    # fires there too and sets the next instant.
    "a_cascade_on_the_instant": (
        "var c = 0.5; var d = 100; E: at (time == c): n = n + 1, d = time; "
        "G: at (time == d): c = c + 1;",
        10,
    ),
    # E raises a flag, F (on the flag) sets the next instant a quarter on and
    # lowers it, and G on that instant moves c: the shape of BIOMD0000000628.
    "a_flag_set_on_an_instant": (
        "var c = 0.5; var on = 0; var t1 = 0; E: at (time == c): n = n + 1, on = 1; "
        "F: at (on == 1): t1 = time + 0.25, on = 0; G: at (time == t1): c = c + 1;",
        10,
    ),
    # The guard is false at 3.5, so c stays there until F adds 3.5 at 6.2;
    # E then fires at 7, 8, 9, 10. A schedule that took the instant at 3.5
    # as a fire put c at 10 by then.
    "a_guard_false_on_an_instant": (
        "var c = 0.5; E: at (time == c && (time < 3 || time > 6)): n = n + 1, c = c + 1; "
        "F: at (time >= 6.2): c = c + 3.5;",
        7,
    ),
    # Executed 0.25 after its instant, on that double, with the time then.
    "a_delayed_value_from_the_clock": (
        "var c = 0.5; E: at 0.25 after (time == c), fromTrigger=false: n = n + 1, "
        "c = time + 0.75;",
        10,
    ),
    # F's value is read at its trigger, 2, with what holds then (c = 2.5,
    # from E at 1.5), and set at 3: c = 7.5, and E fires at 7.5, 8.5, 9.5.
    "a_value_frozen_at_its_trigger": (
        "var c = 0.5; F: at 1 after (time >= 2): c = c + 5; "
        "E: at (time == c): n = n + 1, c = c + 1;",
        6,
    ),
    # Frozen at 1 (x = 1), before G sets x = 5 at 1.2: c = 3.
    "a_value_frozen_before_a_later_write": (
        "var c = 100; var x = 1; var w = 0.25; W: at (time == w): w = w + 100; "
        "F: at 0.5 after (time >= 1): c = x + 2; G: at (time >= 1.2): x = 5; "
        "E: at (time == c): n = n + 1;",
        1,
    ),
    # The clock frozen at the trigger, 1/3: c = 4/3, not the 1.4333 of the
    # execution's time.
    "a_clock_frozen_at_its_trigger": (
        "var c = 100; var w = 0.25; W: at (time == w): w = w + 100; "
        "F: at 0.1 after (time >= 1/3): c = time + 1; E: at (time == c): n = n + 1;",
        1,
    ),
    # A counter one double after a stop on `time >= 0.3`, which is no instant
    # of its own: the instant's stop is kept where the two merge.
    "beside_a_switch_one_double_before": (
        "var c = 0.1; E: at (time == c): n = n + 1, c = c + 0.1; species D = 0; "
        "F: at (time >= 0.3): D = D + 1;",
        100,
        103,
    ),
    # A reaction beside it: the SSA's probe windows start at random times.
    "beside_a_reaction": ("species X = 0; J: -> X; 5; var c = 0.5; " + _COUNTER, 10),
    # A rate law that reads c, stiffly: c stays the double the event set.
    "read_by_a_stiff_rate_law": (
        "species X = 1; J: X -> ; 1000*c*X + 0.5*X; var c = 0.5; " + _COUNTER,
        10,
    ),
}

# Whose instants a sample of the grid of 101 over (0, 10) lands on.
_ON_THE_GRID = [
    name
    for name in _CASES
    if name
    not in (
        "written_after_its_instant",
        "a_step_that_rounds",
        "a_step_of_a_tenth",
        "hourly_in_days",
        "a_flag_set_on_an_instant",
        "a_delayed_value_from_the_clock",
        "a_clock_frozen_at_its_trigger",
        "beside_a_switch_one_double_before",
    )
]


def _model(body: str) -> bngsim.Model:
    return bngsim.Model.from_antimony_string(f"model m\n species n = 0\n {body}\nend")


def _count(body: str, method: str = "ode", **grid) -> float:
    kw = {"seed": 1} if method == "ssa" else {}
    r = bngsim.Simulator(_model(body), method=method).run(**grid, **kw)
    return float(np.asarray(r.species)[-1, list(r.species_names).index("n")])


@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("name", _CASES)
def test_it_fires_on_every_grid(name, grid):
    """Main fired none of these off the grid that lands on the instants."""
    body, fires = _CASES[name][:2]
    assert _count(body, **_GRIDS[grid]) == fires


@pytest.mark.parametrize("name", _ON_THE_GRID)
def test_control_a_grid_on_the_instants(name):
    """Control: a sample on every half-integer fires each instant on main
    too, and the count is the same as off it."""
    body, fires = _CASES[name][:2]
    assert _count(body, t_span=_SPAN, n_points=101) == fires


@pytest.mark.parametrize("grid", _SSA_GRIDS)
@pytest.mark.parametrize("name", _CASES)
def test_a_stochastic_run_fires_it_too(name, grid):
    """The SSA's breakpoints come from the same resolver. Off its probe
    lattice main gave none of these."""
    body = _CASES[name][0]
    fires = _CASES[name][2] if len(_CASES[name]) > 2 else _CASES[name][1]
    assert _count(body, method="ssa", **_SSA_GRIDS[grid]) == fires


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_a_span_off_the_probe_lattice(method):
    """Over (0, 10.3) main gave 0 of 10 in both engines."""
    assert _count("var c = 0.5; " + _COUNTER, method, t_span=(0.0, 10.3), n_points=2) == 10


def test_the_ssa_starts_a_species_counter_on_a_whole_count():
    """``species c = 0.5`` is a molecule count to the SSA, which rounds it to
    1 as the run starts (and warns): its instants are 1, 2, ..., 10, where
    the ODE's are 0.5, 1.5, ..., 9.5. Main fired none off the lattice."""
    body = "species c = 0.5; E: at (time == c): n = n + 1, c = c + 1;"
    assert _count(body, "ssa", t_span=_SSA_SPAN, n_points=2) == 10
    assert _count(body, "ode", t_span=_SSA_SPAN, n_points=2) == 10


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_control_a_trigger_true_before_the_start_does_not_fire_there(method):
    """Control: c = 0 at t = 0, with initialValue true (the default), so the
    trigger does not rise at the start, c stays 0, and the event never
    fires. As on main. Taken to fire at the start, it fires 11 times."""
    body = "var c = 0; E: at (time == c): n = n + 1, c = c + 1;"
    assert _count(body, method, t_span=_SPAN, n_points=2) == 0


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_control_a_delayed_instant_that_is_not_persistent_never_executes(method):
    """Control: the trigger is true at 0.5 alone, so by its execution at 0.6
    it is false, and an event that is not persistent is not taken. c stays
    0.5, and nothing executes. As on main; not refused."""
    body = "var c = 0.5; E: at 0.1 after (time == c), persistent=false: n = n + 1, c = c + 1;"
    assert _count(body, method, t_span=_SPAN, n_points=2) == 0


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_control_a_constant_side(method):
    """Control: with c a constant, main fired it on every grid already."""
    body = "c = 0.5037; E: at (time == c): n = n + 1;"
    assert _count(body, method, t_span=_SPAN, n_points=100) == 1


@pytest.mark.parametrize("legs", [[(0.0, 4.75), (4.75, 10.0)], [(0.0, 5.5), (5.5, 10.0)]])
def test_a_continued_run_places_the_rest(legs):
    """A run that continues the last one reads c as the last one left it."""
    model = _model(f"var c = 0.5; {_COUNTER}")
    sim = bngsim.Simulator(model, method="ode")
    for lo, hi in legs:
        r = sim.run(t_span=(lo, hi), n_points=2)
    assert float(np.asarray(r.species)[-1, 0]) == 10.0
    assert model.get_concentration("c") == 10.5


def test_a_parameter_that_changes_after_it_is_read_leaves_the_schedule_known():
    """F reads on1 at t = 1, when it is 0, so c = 2 and E fires once. W1 and
    W2 set on1 to two values at one instant, 5, so on1 is not known from 5
    on; that is no reason to refuse c, read before then, nor d, which G sets
    at 8 and a rate law reads (a window of 0.01 after 8.5). Main gave 0 and
    0.01 here."""
    body = (
        "var c = 100; var on1 = 0; var d = 100; F: at (time >= 1): c = 2 + on1; "
        "W1: at (time == 5): on1 = 1; W2: at (time == 5): on1 = 2; E: at (time == c): n = n + 1; "
        "G: at (time >= 8): d = 8.5; species Y = 0; "
        "Y' = piecewise(1, time > d && time < d + 0.01, 0);"
    )
    r = bngsim.Simulator(_model(body), method="ode").run(
        t_span=_SPAN, n_points=2, rtol=1e-10, atol=1e-12
    )
    x = np.asarray(r.species)[-1]
    names = list(r.species_names)
    assert x[names.index("n")] == 1.0
    assert x[names.index("Y")] == pytest.approx(0.01, rel=1e-6)


def test_a_counter_beside_two_writers_at_one_instant():
    """W1 and W2 both set on1 at 5.25, in an order not known; that leaves
    on1 not known from 5.25 on, and nothing else. The counter beside them
    gives its 10. Main gave 0."""
    body = (
        "var c = 0.5; var on1 = 0; " + _COUNTER + " W1: at (time == 5.25): on1 = 1; "
        "W2: at (time == 5.25): on1 = 2; species Y = 0; Y' = piecewise(1, time > 9 + on1, 0);"
    )
    for method, span in (("ode", _SPAN), ("ssa", _SSA_SPAN)):
        assert _count(body, method, t_span=span, n_points=2) == 10


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_two_fixed_executions_at_one_instant_beside_a_delayed_one(method):
    """E's delayed execution and F both set c at 1.5: their order is not
    known, so c is not, but d, which K sets at 2 and W reads, is, and so are
    W's instants: 1.75 sets e = 2.75, and 5.5 sets e = 6.5. The schedule read
    the instant's executions by their places in a list a give-up had just
    rewritten, and took K's in place of W's: 1 for 2. Main gave 0."""
    body = (
        "var c = 100; var d = 1.75; var e = 100; E: at 1 after (time == 0.5): c = 7; "
        "F: at (time >= 1.5): c = 9; K: at (time >= 2): d = 5.5; W: at (time == d): "
        "e = time + 1; H: at (time == e): n = n + 1; species Y = 0; "
        "Y' = piecewise(1, time > c, 0);"
    )
    assert _count(body, method, t_span=_SPAN, n_points=2) == 2


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_control_an_instant_on_a_constant_beside_that_give_up(method):
    """Control: H fires at 2.25, a constant, as on main. The same give-up
    read a list index past its end, and the run went on with no stops at
    all and a warning: 0."""
    body = (
        "var c = 100; var d = 2.25; E: at 1 after (time == 0.5): c = 7; "
        "F: at (time >= 1.5): c = 9; H: at (time == d): n = n + 1; species Y = 0; "
        "Y' = piecewise(1, time > c, 0);"
    )
    assert _count(body, method, t_span=_SPAN, n_points=2) == 1


def test_an_instant_a_double_after_another_is_refused_where_the_stops_merge():
    """A counter by 0.1 reaches 0.30000000000000004, one double after G's
    `time == 0.3`. The integrator merges two such stops into the earlier,
    and the counter, never stopped on, fired 2 times for 100: refused. The
    SSA keeps both stops, and gives 100."""
    body = (
        "var c = 0.1; E: at (time == c): n = n + 1, c = c + 0.1; species g = 0; "
        "G: at (time == 0.3): g = g + 1;"
    )
    with pytest.raises(bngsim.SimulationError, match="within a rounding of another stop"):
        _count(body, t_span=_SPAN, n_points=2)
    assert _count(body, "ssa", t_span=_SSA_SPAN, n_points=2) == 103


def test_an_amount_is_read_as_an_amount():
    """c is an amount in a compartment of 3, and the trigger reads the amount:
    0.1, 0.2, ..., 10. Read as its concentration, the instants were a third
    of those. Main gave 0."""
    body = (
        "compartment V = 3; substanceOnly species c in V = 0.1; compartment W = 1; "
        "species n in W; E: at (time == c): n = n + 1, c = c + 0.1;"
    )
    r = bngsim.Simulator(bngsim.Model.from_antimony_string(f"model m\n {body}\nend")).run(
        t_span=_SPAN, n_points=2
    )
    assert float(np.asarray(r.species)[-1, list(r.species_names).index("n")]) == 100.0


def test_a_rate_law_reads_a_species_held_by_c_prime_zero_stretch_by_stretch():
    """#1244's per-stretch stops, now for a species held by ``c' = 0`` too:
    a window of 0.25 every c, with c from 1 to 0.5 at t = 2, is 0.5 by then
    and 4 after, 4.5 in all. Main stepped over the windows after 2: 2.25."""
    model = bngsim.Model.from_antimony_string(
        "model m\n species c = 1; c' = 0; F: at (time >= 2): c = 0.5; species Y = 0; "
        "Y' = piecewise(1, time - c*floor(time/c) < 0.25, 0);\nend"
    )
    r = bngsim.Simulator(model, method="ode").run(t_span=_SPAN, n_points=2, rtol=1e-10, atol=1e-12)
    y = float(np.asarray(r.species)[-1, list(r.species_names).index("Y")])
    assert y == pytest.approx(4.5, rel=1e-6)


def test_a_rate_law_reads_a_value_set_through_an_assignment_rule():
    """#1244's per-stretch stops, for a value read through an assignment
    rule: w2 := 2*k is 2, so a daily pulse of width 1 becomes 2 from t = 50,
    and X(240) is 17 (#1244's own case). Main refused it as not known before
    the run."""
    pulse = "piecewise(1, (time - 24*floor(time/24)) < w, 0)"
    model = bngsim.Model.from_antimony_string(
        "model m\n compartment C = 1; species X in C; X = 0; w = 1; k = 1; w2 := 2*k;\n"
        f" J0: -> X; {pulse}\n E1: at (time >= 50): w = w2\nend"
    )
    r = bngsim.Simulator(model, method="ode").run(sample_times=[0, 240], rtol=1e-8, atol=1e-10)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(17, rel=1e-6)


def test_six_hundred_daily_doses():
    """A daily dose at 8 for 600 days, D' = -0.1 D. Main gave every dose only
    where an hourly grid landed on them; at two points, none."""
    body = (
        "var c = 8; species D = 0; D' = -0.1*D; "
        "E: at (time == c): n = n + 1, D = D + 100, c = c + 24;"
    )
    r = bngsim.Simulator(_model(body), method="ode").run(
        t_span=(0.0, 14400.0), n_points=2, rtol=1e-10, atol=1e-12
    )
    x = np.asarray(r.species)[-1]
    names = list(r.species_names)
    exact = sum(100 * math.exp(-0.1 * (14400 - (8 + 24 * k))) for k in range(600))
    assert x[names.index("n")] == 600.0
    assert x[names.index("D")] == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize(("n_points", "rtol"), [(100, 1e-6), (102, 1e-6), (100, 1e-10)])
def test_biomd0000000620_gives_every_dose(n_points, rtol):
    """Palmer2014's daily anakinra dose, ``time == counter && counter < 91``
    from 0.5: main gave Anakinrasc(10) = 0 and the counter 0.5 off the grid
    of 101, at rtol 1e-10 too. Its 14222.31 there is the oracle."""
    path = Path(__file__).resolve().parents[2] / "benchmarks/sbml_events/BIOMD0000000620.xml"

    def at_ten(n, tol):
        r = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode").run(
            t_span=(0, 10), n_points=n, rtol=tol, atol=1e-9
        )
        x = np.asarray(r.species, float)[-1, list(r.species_names).index("Anakinrasc")]
        names = list(r.observable_names)
        counter = np.asarray(r.observables, float)[-1, names.index("Anakinra_dose_counter")]
        return x, counter

    oracle, _ = at_ten(101, rtol)
    dose, counter = at_ten(n_points, rtol)
    assert counter == 10.5
    assert dose == pytest.approx(oracle, rel=1e-5)
    assert oracle == pytest.approx(14222.3, rel=1e-4)


def test_biomd0000000613_gives_every_dose():
    """Teriparatide at 12, then every 24 while before 84 (BIOMD0000000613),
    on ``time == t0 || (time == counter + t0 && time < 84)``: three doses, so
    the counter reads 72. Main gave 24, one dose, off the grid of 101 over
    (0, 100), with nothing said."""
    path = Path(__file__).resolve().parents[2] / "benchmarks/sbml_events/BIOMD0000000613.xml"

    def at_100(n):
        r = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode").run(
            t_span=(0, 100), n_points=n, rtol=1e-10, atol=1e-20
        )
        x = np.asarray(r.species, float)[-1, list(r.species_names).index("TERISC")]
        names = list(r.observable_names)
        return np.asarray(r.observables, float)[-1, names.index("event_counter")], x

    oracle = at_100(101)
    for n in (2, 1000, 1002):
        counter, dose = at_100(n)
        assert counter == 72.0
        assert dose == pytest.approx(oracle[1], rel=1e-6)


def test_biomd0000000628_ends_every_pulse():
    """BIOMD0000000628's 100 stimulation pulses from t = 300, each ended at
    ``time == parameter_7 + 0.008``, where parameter_7 is set to the time by
    an event on ``parameter_6 == 1`` in the instant another event raises that
    flag. Main never ended the first pulse, on any grid: parameter_6 = 1 and
    parameter_1 = 0.00717 at t = 1000. Run on a sample at every instant, main
    agrees with this to 4e-10."""
    path = Path(__file__).resolve().parents[2] / "benchmarks/sbml_events/BIOMD0000000628.xml"
    r = bngsim.Simulator(bngsim.Model.from_sbml(str(path)), method="ode").run(
        t_span=(0, 302), n_points=2
    )
    names = list(r.species_names) + list(r.observable_names)
    x = np.concatenate([np.asarray(r.species, float)[-1], np.asarray(r.observables, float)[-1]])

    def at(name):
        return float(x[names.index(name)])

    assert at("parameter_3") == 100.0
    assert at("parameter_6") == 0.0
    assert at("parameter_1") == 0.0


# ── Where the instants cannot be placed: refused by name ────────────────────


@pytest.mark.parametrize("method", ["ode", "ssa"])
@pytest.mark.parametrize(
    ("body", "why"),
    [
        (
            "species X = 0; X' = 1; var c = 100; F: at (X > 2): c = time + 1; "
            "E: at (time == c): n = n + 1;",
            "event F on X>2 sets it, a trigger that is no fixed time",
        ),
        (
            "species X = 1; X' = 1; var c = 0.5; E: at (time == c): n = n + 1, c = c + X;",
            "sets it to (c+X), which is not known before the run",
        ),
        (
            "var c = 0.5; E: at (time == c): n = n + 1, c = c + 1; F: at (time >= 2.5): c = 4.5;",
            "with another event at t=2.5, in an order not known before the run",
        ),
        # E's trigger is not an equality that holds at an instant: it rises
        # just after c, at a double no stop names.
        (
            "var c = 0.5; E: at (time != c): n = n + 1, c = c + 1;",
            "a trigger that is no fixed time",
        ),
        # E's guard reads X = 2t, which moves: whether E fires at 0.5 is not
        # known before the run (it does not; F then sets c = 2.5, and E fires
        # at 2.5, 3.5, ..., 9.5).
        (
            "species X = 0; X' = 2; var c = 0.5; "
            "E: at (time == c && X > 4): n = n + 1, c = c + 1; F: at (time >= 2.2): c = 2.5;",
            "on an instant that reads more than the clock and what events alone set",
        ),
        # G fires on an equality on a counter, which no stop is placed on:
        # whether it fires is not known.
        (
            "species tau = 0; tau' = 1; var c = 0.5; G: at (tau == 3): c = 7.5; "
            "E: at (time == c): n = n + 1, c = c + 1;",
            "event G on tau==3 sets it, a trigger that is no fixed time",
        ),
        # The increment reads on, which a state trigger sets (at 1.6): from
        # then on the instants are 4.75, 6, ... (9 in all), and c's schedule
        # read with on at its start gave 4.
        (
            "var c = 0.5; var on = 0; species X = 0; X' = 2; F: at (X > 3.2): on = 1; "
            "E: at (time == c): n = n + 1, c = c + 1 + 0.25*on;",
            "which is not known then",
        ),
        # The guard reads a flag a state trigger raises (at 0.15): read at its
        # start, the flag was down, and the counter fired once for 10.
        (
            "var c = 0.5; var on = 0; species X = 0; X' = 2; G: at (X > 0.3): on = 1; "
            "E: at (time == c && on == 1): n = n + 1, c = c + 1;",
            "its trigger reads on",
        ),
        # F's value reads on, which a state trigger sets at 1.5, before F at 2.
        (
            "var c = 100; var on = 0; species X = 0; X' = 2; G: at (X > 3): on = 2; "
            "F: at (time >= 2): c = 3 + on; E: at (time == c): n = n + 1;",
            "which is not known then",
        ),
        # c*0.7 + 1.1 closes in on 11/3: its instants come within a rounding
        # of each other, and a stop cannot tell two such apart.
        (
            "var c = 0.5; E: at (time == c): n = n + 1, c = c*0.7 + 1.1;",
            "closer than a stop tells apart",
        ),
        # `time > 5` holds from a root a little after 5 (5.00000000000008 at
        # two points), where c = 5 has passed, so E does not fire there; the
        # schedule, which reads F at 5, cannot tell.
        (
            "var c = 100; F: at (time > 5): c = 5; E: at (time == c): n = n + 1, c = c + 1;",
            "known only to within the tolerance",
        ),
        # F (`time > 2`) executes at a root a little after E's instant at 2,
        # so G, on the flag F raises, reads the clock where the schedule does
        # not know it.
        (
            "var c = 0.5; var d = 0; var e = 100; E: at (time == c): c = c + 1.5; "
            "F: at (time > 2): d = 1; G: at (d == 1): e = time + 1; H: at (time == e): n = n + 1;",
            "known only to within the tolerance",
        ),
        # F's value is read at its trigger, 1, the instant E sets x: which the
        # engine does first is not known here.
        (
            "var c = 1; var x = 0; var y = 100; E: at (time == c): x = x + 10, c = c + 100; "
            "F: at 0.5 after (time >= 1): y = x + 3; H: at (time == y): n = n + 1;",
            "reads its value at its trigger, where another event sets it",
        ),
        # E1 and P set a and b at 1, apart, but G reads both: it fires between
        # them in one order and not in the other.
        (
            "var c1 = 1; var a = 0; var b = 0; var c = 100; var w = 0.5; "
            "E1: at (time == c1): a = 1, c1 = c1 + 100; P: at 0.5 after (time == w): b = 1, "
            "w = w + 100; G: at (a == 1 && b == 0): c = 5; H: at (time == c): n = n + 1;",
            "and its trigger reads what events set at t=1.0",
        ),
        # E sets d to the instant it fires on, and two events on that instant
        # set c, so which sets it last is not known.
        (
            "var c = 0.5; var d = 100; E: at (time == c): n = n + 1, d = time; "
            "G: at (time == d): c = c + 1; H: at (time == d): c = c + 2;",
            "in an order not known before the run",
        ),
    ],
    ids=[
        "a-state-trigger",
        "a-value-of-the-state",
        "two-writers-at-one-instant",
        "a-writer-on-not-equal",
        "a-guard-on-the-state",
        "a-writer-on-a-counter-equality",
        "a-value-from-a-parameter-a-state-trigger-sets",
        "a-guard-on-a-flag-a-state-trigger-raises",
        "a-fixed-value-from-a-parameter-a-state-trigger-sets",
        "instants-closer-than-a-stop",
        "a-strict-trigger-sets-its-own-threshold",
        "a-root-beside-an-instant",
        "a-value-frozen-where-another-event-sets-it",
        "an-order-a-third-trigger-reads",
        "a-cascade-of-two-writers",
    ],
)
def test_a_schedule_not_known_before_the_run_is_refused(body, why, method):
    """Main fired none of these off the instants, with nothing said."""
    with pytest.raises(bngsim.SimulationError, match="issue #1519") as e:
        _count(body, method, t_span=_SPAN, n_points=2)
    assert why in str(e.value)
    assert re.search(r"and only events set \w+:", str(e.value))


def test_past_the_execution_budget_it_is_refused_and_legs_are_not(monkeypatch):
    """More executions than the run follows is refused, with the way out, and
    the same run in legs is not."""
    from bngsim import _switch_sensitivity

    monkeypatch.setattr(_switch_sensitivity, "_HELD_EXECUTIONS_MOST", 6)
    body = "var c = 0.5; " + _COUNTER
    with pytest.raises(bngsim.SimulationError, match="more than 6 times in the run") as e:
        _count(body, t_span=_SPAN, n_points=2)
    assert "shorter legs" in str(e.value)
    model = _model(body)
    sim = bngsim.Simulator(model, method="ode")
    for lo, hi in ((0.0, 3.0), (3.0, 6.0), (6.0, 10.0)):
        r = sim.run(t_span=(lo, hi), n_points=2)
    assert float(np.asarray(r.species)[-1, 0]) == 10.0


def test_an_instant_no_double_meets_is_refused_as_for_a_constant():
    """``time - 1 == c`` with c set to 1.3 at 0.5: 2.3 - 1 is
    1.2999999999999998 and the next double gives 1.3000000000000003, so the
    event cannot fire, as for ``time - 1 == 1.3`` (issue #1015). Main ran it,
    and it fired no time, with nothing said."""
    body = "var c = 0; F: at (time >= 0.5): c = 1.3; E: at (time - 1 == c): n = n + 1;"
    with pytest.raises(bngsim.SimulationError, match="never equal in double arithmetic"):
        _count(body, t_span=_SPAN, n_points=2)


# ── Read as before ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_control_an_equality_of_a_flag_a_state_trigger_sets(method):
    """Control: ``flag == 1`` reads no clock, and changes only where an event
    sets the flag, where the engine reads the triggers again. No stop is
    needed, and a state trigger's write is no reason to refuse it."""
    body = (
        "species X = 0; X' = 1; var flag = 0; F: at (X > 2): flag = 1; "
        "E: at (flag == 1): n = n + 1;"
    )
    assert _count(body, method, t_span=_SPAN, n_points=2) == 1


def test_control_an_equality_on_a_counter_is_read_as_before():
    """Control: no stop is placed on an equality on a counter species, which
    is integrated and does not land on the value (the exception events.md
    names), whatever its other side reads, so one a state trigger writes is
    not refused either. As on main, the event does not fire."""
    body = (
        "species tau = 0; tau' = 1; species X = 0; X' = 2; var c = 0.5; "
        "F: at (X > 6): c = 5.5; E: at (tau == c): n = n + 1;"
    )
    assert _count(body, t_span=_SPAN, n_points=2) == 0
