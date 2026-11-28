"""Event cases (issue #1913: filed bugs in #1781's area, each a strict expected
failure until its fix lands).

Every model here is Antimony, so its events are SBML's. None has a spec: the
spec holds no resets, and a touch surface is never seen on both of its sides.
Each case keeps points that pass (a parameter row off the bug, or a grid that
dodges it), which show that the case and its closed form are written right.
"""

from __future__ import annotations

from _case import Axes, Case, Judge, Known, Sampled, Switch, Uniform, point
from _closed_forms import (
    event_double_apart,
    event_row_short,
    event_touch_tie,
    event_trigger_step,
    event_unlisted_value,
)

# ── #1618: a touch on the same instant as an event whose time moves ─────────

TOUCH_TIE = """\
species X; X = 1; k = 0.1; tau = 2.5
J0: X -> ; k*X
E1: at (time > tau): X = X + 1
E2: at (sin(2*3.141592653589793*time/10) >= 1): X = 2*X
"""

TIE_KEYS = {
    f"event_touch_tie[tau=2.5]|rtol={r},grid=sampled4,sens=tau": Known(
        1618,
        AssertionError,
        "a touch event on the instant of an event whose time moves escapes the same-instant refusal",
    )
    for r in ("1e-08", "1e-10")
}

# ── #1626: a rate law on a species an event sets to a value not listed ──────
#
# F2 fires at 1.2 here, after F's trigger at 1 (the issue's at 1), so F reads
# x = 0 with no same-instant order to decide: the same y = 3, the same 0.05.

UNLISTED = """\
model m
 var x = 0; var y = 100
 F2: at (time >= 1.2): x = 10
 F: at 0.5 after (time >= 1): y = x + 3
 species Y = 0
 Y' = piecewise(1, time - y*floor(time/y) < 0.05, 0)
end
"""

# ── #1627: two events a double apart share one stop ─────────────────────────

DOUBLE_APART = """\
model m
 k = 3; j = 3; var amt = 1
 species g = 0; species h = 0; species D = 0
 G: at (time == 0.3): g = g + 1
 H: at (time == 0.1*k): h = h + 1
 E: at (time == 0.1*j): D = D + amt
 F: at (time >= 0.3): amt = 2
end
"""

# ── #1799: a sensitivity run records a row from up to 1e-12 short of it ─────

ROW_SHORT = """\
species X, Y; X = 1; Y = 0; k = 1e11
J1: X -> ; k*X
E1: at (time >= 2.95e-11): Y = 1
"""

# ── #1804: a state trigger that steps at its own fire instant ───────────────

TRIGGER_STEP = """\
species B, C; B = 0; C = 0; T0 = 0.5
J0: -> B; 1
J1: -> C; 1
E1: at (C + piecewise(0, time < T0 + 1, 5) >= T0 + 1): B = 0
"""

CASES = (
    Case(
        name="event_touch_tie",
        family="event",
        model=TOUCH_TIE,
        fmt="antimony",
        switches=(Switch("E1", "time", "jump"), Switch("E2", "time", "jump")),
        params={"k": 0.1, "tau": 2.5},
        t_span=(0.0, 6.0),
        observe=("X",),
        reference=event_touch_tie(),
        axes=Axes(
            params=({"tau": 2.4}, {"tau": 2.5}, {"tau": 2.6}),
            rtol=(1e-8, 1e-10),
            grids=(Sampled(4),),
            sens=(("tau",),),
        ),
        slice=(
            point(1e-10, Sampled(4), ("tau",), tau=2.5),
            point(1e-8, Sampled(4), ("tau",), tau=2.5),
            point(1e-10, Sampled(4), ("tau",), tau=2.4),
            point(1e-8, Sampled(4), ("tau",), tau=2.6),
        ),
        judge=Judge(),
        known=TIE_KEYS,
        note=(
            "#1618: E1 adds 1 to X at tau, E2 doubles it at 2.5, where sin(2πt/10) touches "
            "1. At tau = 2.5 the order decides X(6), so dX/dtau has no value and the run owes "
            "a refusal; at 2.4 and 2.6 the closed form holds."
        ),
    ),
    Case(
        name="event_unlisted_value",
        family="event",
        model=UNLISTED,
        fmt="antimony",
        switches=(Switch("window", "time", "jump"),),
        params={},
        t_span=(0.0, 20.0),
        observe=("Y",),
        reference=event_unlisted_value(),
        axes=Axes(rtol=(1e-6, 1e-8, 1e-10), grids=(Uniform(2), Uniform(101))),
        slice=(
            point(1e-8, Uniform(2)),
            point(1e-6, Uniform(101)),
            point(1e-10, Uniform(101)),
        ),
        judge=Judge(),
        known={
            f"event_unlisted_value|rtol={r},grid=u2,sens=none": Known(
                1626,
                AssertionError,
                "a condition on a species an event sets to a value not listed before the run "
                "is stepped over",
            )
            for r in ("1e-06", "1e-08", "1e-10")
        },
        note=(
            "#1626: a window on time - y*floor(time/y), with y set by an event to x + 3, a "
            "value read from the state. A grid of 2 rows steps over every window after 1.5; "
            "a grid of 101 samples each."
        ),
    ),
    Case(
        name="event_double_apart",
        family="event",
        model=DOUBLE_APART,
        fmt="antimony",
        switches=(
            Switch("G", "time", "jump"),
            Switch("H", "time", "jump"),
            Switch("E", "time", "jump"),
            Switch("F", "time", "jump"),
        ),
        params={"k": 3.0, "j": 3.0},
        t_span=(0.0, 1.0),
        observe=("g", "h", "D"),
        reference=event_double_apart(),
        axes=Axes(
            params=({}, {"k": 4.0, "j": 4.0}),
            rtol=(1e-6, 1e-8),
            grids=(Uniform(2), Uniform(101)),
        ),
        slice=(
            point(1e-8, Uniform(2)),
            point(1e-6, Uniform(101)),
            point(1e-8, Uniform(2), k=4.0, j=4.0),
            point(1e-6, Uniform(101), k=4.0, j=4.0),
        ),
        judge=Judge(),
        known={
            f"event_double_apart|rtol={r},grid={g},sens=none": Known(
                1627, AssertionError, "two event stops a double apart are merged into one"
            )
            for r in ("1e-06", "1e-08")
            for g in ("u2", "u101")
        },
        note=(
            "#1627: equalities on 0.3 and on 0.1*3 = 0.30000000000000004, a switch on "
            "time >= 0.3 and an instant a double after it. Each must fire on its own double, "
            "in time order. At k = j = 4 the instants lie apart, and the run is right."
        ),
    ),
    Case(
        name="event_row_short",
        family="event",
        model=ROW_SHORT,
        fmt="antimony",
        switches=(Switch("E1", "time", "jump"),),
        params={"k": 1e11},
        t_span=(0.0, 1e-10),
        observe=("X", "Y"),
        reference=event_row_short(),
        axes=Axes(rtol=(1e-8, 1e-10), grids=(Sampled(11),), sens=((), ("k",))),
        slice=(
            point(1e-10, Sampled(11), ("k",)),
            point(1e-8, Sampled(11), ("k",)),
            point(1e-10, Sampled(11)),
            point(1e-8, Sampled(11)),
        ),
        atol_ratio=1e-4,
        judge=Judge(),
        known={
            f"event_row_short|rtol={r},grid=sampled11,sens=k": Known(
                1799, AssertionError, "a sensitivity run records the row at 3e-11 from 2.95e-11"
            )
            for r in ("1e-08", "1e-10")
        },
        note=(
            "#1799, on #765's model: rows every 1e-11 and an event at 2.95e-11. The row asked "
            "for at 3e-11 must come back at 3e-11. Plain runs do; sensitivity runs record it "
            "from the event's time. A Uniform grid would not show it (its interior rows are "
            "judged where the run put them), so the grid is Sampled."
        ),
    ),
    Case(
        name="event_trigger_step",
        family="event",
        model=TRIGGER_STEP,
        fmt="antimony",
        switches=(Switch("E1", "state", "jump"),),
        params={"T0": 0.5},
        t_span=(0.0, 4.0),
        observe=("B", "C"),
        reference=event_trigger_step(),
        axes=Axes(
            params=({}, {"T0": 1.3}),
            rtol=(1e-8, 1e-10),
            grids=(Sampled(3),),
            sens=((), ("T0",)),
        ),
        slice=(
            point(1e-10, Sampled(3), ("T0",)),
            point(1e-8, Sampled(3), ("T0",), T0=1.3),
            point(1e-10, Sampled(3)),
            point(1e-8, Sampled(3), T0=1.3),
        ),
        judge=Judge(),
        known={
            f"{row}|rtol={r},grid=sampled3,sens=T0": Known(
                1804,
                AssertionError,
                "dt*/dT0 comes from differences straddling the trigger's own step",
            )
            for row in ("event_trigger_step", "event_trigger_step[T0=1.3]")
            for r in ("1e-08", "1e-10")
        },
        note=(
            "#1804: B is reset when C + piecewise(0, time < T0+1, 5) >= T0+1, which steps at "
            "its own fire instant t* = T0 + 1. dB/dT0 = -1 past t*; the run gives -(T0+1)/T0."
        ),
    ),
)
