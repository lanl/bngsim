"""Event cases: #1913's filed bugs in #1781's area, each a strict expected
failure until its fix lands, and #1915's events that set or move the state.

Every model here is Antimony, so its events are SBML's. #1913's cases have no
spec (a touch surface is never seen on both of its sides); #1915's do, with
the events as resets (``_branchwise.Reset``), and the spec check compares their
flows with bngsim's. Each case keeps points that pass (a parameter row off the
bug, or a grid that dodges it), which show that the case and its closed form are
written right.
"""

from __future__ import annotations

from _branchwise import Reset, Spec, Surface
from _case import Axes, Case, Judge, Known, Sampled, Switch, Uniform, point
from _closed_forms import (
    clock_reset,
    cross_switch,
    event_double_apart,
    event_row_short,
    event_touch_tie,
    event_trigger_step,
    event_unlisted_value,
    state_reset,
)

from _cases import _families as F
from _cases.pulse import MOVED_ROW, _known
from _cases.step import STATE_GRIDS

#: The window peak's reason for a crossing on the state (as ``step_state``'s).
_LOCATED = (
    "{what} is a crossing on the state, located only to the run's own accuracy, about "
    "rtol*X/|X'|, so a row on it reads columns that jump there at that distance into the far "
    "side; scaled by its samples alone, a column that decays past it would be judged against "
    "its tail"
)
RESET_JUDGE = Judge(
    peak="window",
    converge=False,
    why=_LOCATED.format(what="each firing")
    + ". The convergence rows are off: a plain run's own global error at rtol 1e-12 is 35 times "
    "rtol before any firing (X(5) 1.3e-10 off), so its convergence rows fail the rule on every one "
    "of 240 runs whose atol differed in its last digits on 15 groups, and on 1 to 141 of them on 5 "
    "more. Every point passes on every one of 480 runs, under 338 times its tolerance, 3 times "
    "under k",
)
CROSS_JUDGE = Judge(
    peak="window",
    converge=False,
    why=_LOCATED.format(what="X falling through th")
    + ". The convergence rows are off: its plain points reach 148 to 203 times their tolerance at "
    "rtol 1e-10 and 1e-12 on aligned rows, and its convergence rows fail the rule on 1 to 148 of "
    "240 runs whose atol differed in its last digits. Every point passes on every one of 480 "
    "runs, under 362 times its tolerance, 2.8 times under k",
)
RESIDUE = (
    AssertionError,
    "#1997: past the pulse dX/dw keeps a residue that does not fall with rtol, 1.3e-7 to 2.2e-7 "
    "where it is 0",
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


# ── #1915: events that set or move the state ────────────────────────────────

STATE_RESET = """\
species X, T; X = 10; T = 0
k = 0.2; th = 3; a = 1; d = 4; kc = 2
J0: -> T; kc
J1: X -> ; k*X
E1: at (X < th): X = a*X + d
"""
STATE_RESET_PARAMS = {"k": 0.2, "th": 3.0, "a": 1.0, "d": 4.0, "kc": 2.0}
STATE_RESET_COLS = ("k", "th", "a", "d")

CROSS_SWITCH = """\
species X, T; X = 10; T = 0
k1 = 0.5; k2 = 0.05; th = 3; tau = 6; xr = 8; kc = 2
J0: -> T; kc
J1: X -> ; piecewise(k1, X > th, k2)*X
E1: at (time >= tau): X = xr
"""
CROSS_SWITCH_PARAMS = {"k1": 0.5, "k2": 0.05, "th": 3.0, "tau": 6.0, "xr": 8.0, "kc": 2.0}
CROSS_SWITCH_COLS = ("k1", "k2", "th", "tau", "xr")

CLOCK_RESET = """\
species T, X, Y; T = 0; X = 10; Y = 0
kx = 1; kc = 2; c = 30; w = 0.1; tau = 10
J0: -> T; kc
J1: X -> Y; piecewise(kx/w, (T > c) && (T < c + w), 0)*X
E1: at (time >= tau): T = 0
"""
CLOCK_RESET_PARAMS = {"kx": 1.0, "kc": 2.0, "c": 30.0, "w": 0.1, "tau": 10.0}
CLOCK_RESET_COLS = ("kx", "kc", "c", "w", "tau")

#: The clock reset's judge: a pulse 0.05 wide in time (1/80 at kc = 8), on a
#: counter. Its numbers are this case's own (see ``why``).
CLOCK_RESET_JUDGE = Judge(
    peak="window",
    converge=False,
    why=(
        "a clock pulse 0.05 wide in time: inside it the c and w columns run 1e2 to 1e3 times "
        "their end values, which are 0, and CVODES holds them to rtol of that running size, so "
        "each column is scaled by its peak over the window, as #1914's narrow pulses are. Past "
        "the pulse those columns keep a residue that does not fall with rtol (dX/dc 9.2e-10 and "
        "dX/dw 1.8e-7 at rtol 1e-10 and 1e-12, the same in the .net spelling: #1997, marked at "
        "1e-12), so the convergence rows are off. Over 240 runs whose atol differed in its last "
        "digits, and 240 more at random, every point outside the marks and _flips stays under "
        "404 times its tolerance, 2.5 times under k, and the lowest mark is 2.7e3, 2.7 times "
        "over it. A review's 320 further draws found three more points between 578 and 589 "
        "times their tolerance, which _flips leaves out too"
    ),
)

CASES = (
    *CASES,
    Case(
        name="event_state_reset",
        family="event",
        model=STATE_RESET,
        fmt="antimony",
        switches=(Switch("E1", "state", "jump"),),
        params=STATE_RESET_PARAMS,
        t_span=(0.0, 30.0),
        observe=("X", "T"),
        reference=state_reset(t_end=30.0),
        spec=Spec(
            states=("X", "T"),
            params=tuple(STATE_RESET_PARAMS),
            rhs=lambda x, s, t, p: {"X": -p["k"] * x["X"], "T": p["kc"]},
            surfaces=(),
            x0=lambda p: {"X": 10, "T": 0},
            resets=(
                Reset(
                    "E1",
                    lambda x, t, p: p["th"] - x["X"],
                    lambda x, t, p: {"X": p["a"] * x["X"] + p["d"]},
                ),
            ),
        ),
        axes=Axes(
            params=(
                {},  # 6 firings, every 4.24 from t = 6.02
                {"d": 20.0},  # 3 firings
                {"a": 0.0, "d": 9.0},  # each firing sets X to 9: 5 firings
                {"k": 0.05},  # one firing, at t = 24.1
                {"k": 0.3, "d": 2.0},  # 16 firings, every 1.70
                {"a": 0.5, "d": 6.0},  # each firing halves X and adds 6
            ),
            grids=STATE_GRIDS,
            sens=((), STATE_RESET_COLS),
        ),
        slice=(
            point(1e-12, F.Aligned(31), STATE_RESET_COLS),
            point(1e-10, STATE_GRIDS[-1], STATE_RESET_COLS, k=0.3, d=2.0),
            point(1e-8, F.Aligned(2), STATE_RESET_COLS, a=0.0, d=9.0),
            point(1e-6, F.Uniform(31), STATE_RESET_COLS, d=20.0),
            point(1e-12, F.Uniform(4), (), a=0.5, d=6.0),
        ),
        judge=RESET_JUDGE,
        note=(
            "X decays at k, and each time it falls through th an event sets X = a*X + d: a "
            "trigger on the state, firing again and again. Firing j comes at t1 + (j - 1)*D, "
            "t1 = ln(10/th)/k and D = ln((a*th + d)/th)/k, so every column carries each firing "
            "time's derivative."
        ),
    ),
    Case(
        name="event_cross_switch",
        family="event",
        model=CROSS_SWITCH,
        fmt="antimony",
        switches=(Switch("X>th", "state", "jump"), Switch("E1", "time", "jump")),
        params=CROSS_SWITCH_PARAMS,
        t_span=(0.0, 30.0),
        observe=("X", "T"),
        reference=cross_switch(),
        spec=Spec(
            states=("X", "T"),
            params=tuple(CROSS_SWITCH_PARAMS),
            rhs=lambda x, s, t, p: {
                "X": -(p["k2"] + (p["k1"] - p["k2"]) * s[0]) * x["X"],
                "T": p["kc"],
            },
            surfaces=(Surface("X>th", lambda x, t, p: x["X"] - p["th"]),),
            x0=lambda p: {"X": 10, "T": 0},
            resets=(Reset("E1", lambda x, t, p: t - p["tau"], lambda x, t, p: {"X": p["xr"]}),),
        ),
        axes=Axes(
            params=(
                {},  # X falls through 3 at 2.41; the event at 6 sets it to 8, above
                {"xr": 2.0},  # set below: it stays below
                {
                    "tau": 1.0,
                    "xr": 2.0,
                },  # set below before it got there: the branch flips at the event
                {"tau": 1.0},  # set to 8 before it got there: no flip
                {"tau": 25.0},  # set above late: it falls through again at 26.96
                {"tau": 25.37},
                {"xr": 20.0},
            ),
            grids=STATE_GRIDS,
            sens=((), CROSS_SWITCH_COLS),
        ),
        slice=(
            point(1e-12, F.Aligned(31), CROSS_SWITCH_COLS),
            point(1e-10, STATE_GRIDS[-1], CROSS_SWITCH_COLS, tau=1.0, xr=2.0),
            point(1e-8, F.Aligned(2), CROSS_SWITCH_COLS, tau=25.0),
            point(1e-6, F.Uniform(31), CROSS_SWITCH_COLS, xr=2.0),
            point(1e-12, F.Uniform(4), (), tau=25.37),
        ),
        judge=CROSS_JUDGE,
        note=(
            "X decays at piecewise(k1, X > th, k2), and an event at tau sets X = xr: the event "
            "moves the state across the switch's surface with no root, so the branch it lands on "
            "must be read from the state it sets. Above th, X falls through it again at "
            "tau + ln(xr/th)/k1."
        ),
    ),
    Case(
        name="event_clock_reset",
        family="event",
        model=CLOCK_RESET,
        fmt="antimony",
        switches=(
            Switch("open", "state", "jump"),
            Switch("shut", "state", "jump"),
            Switch("E1", "time", "jump"),
        ),
        params=CLOCK_RESET_PARAMS,
        t_span=(0.0, 30.0),
        observe=("T", "X", "Y"),
        reference=clock_reset(),
        spec=Spec(
            states=("T", "X", "Y"),
            params=tuple(CLOCK_RESET_PARAMS),
            rhs=lambda x, s, t, p: {
                "T": p["kc"],
                "X": -p["kx"] / p["w"] * s[0] * s[1] * x["X"],
                "Y": p["kx"] / p["w"] * s[0] * s[1] * x["X"],
            },
            surfaces=(
                Surface("open", lambda x, t, p: x["T"] - p["c"]),
                Surface("shut", lambda x, t, p: p["c"] + p["w"] - x["T"]),
            ),
            x0=lambda p: {"T": 0, "X": 10, "Y": 0},
            resets=(Reset("E1", lambda x, t, p: t - p["tau"], lambda x, t, p: {"T": 0}),),
        ),
        axes=Axes(
            params=(
                {},  # the reset at 10 moves the pulse from 15 to 25
                {"tau": 10.37},  # to 25.37, between grid points
                {"tau": 20.0},  # after the pulse at 15: the next would be at 35
                {"tau": 15.02},  # inside the pulse: the reset cuts it short
                {"kc": 8.0, "c": 120.0},  # a faster clock: the pulse at 25, 1/80 wide
                {"w": 4.0},  # a wide pulse, 25 to 27
                {"c": 18.0, "tau": 16.0},  # two pulses: at 9, and at 25 after the reset
            ),
            grids=F.GRIDS,
            sens=((), CLOCK_RESET_COLS),
        ),
        slice=(
            point(1e-12, F.Aligned(31), CLOCK_RESET_COLS),
            point(1e-10, F.Offset(31), CLOCK_RESET_COLS, kc=8.0, c=120.0),
            point(1e-8, F.Aligned(2), CLOCK_RESET_COLS, c=18.0, tau=16.0),
            point(1e-6, F.Uniform(31), CLOCK_RESET_COLS, w=4.0),
            point(1e-12, F.Uniform(4), (), kc=8.0, c=120.0),
        ),
        judge=CLOCK_RESET_JUDGE,
        note=(
            "A counter T at kc, X -> Y at kx/w while c < T < c + w, and an event at tau that sets "
            "T back to 0: the reset moves the crossings the clock carries, from c/kc to "
            "tau + c/kc, and inside the pulse it cuts the pulse short."
        ),
    ),
)

_CLOCK_MARKS = {
    (1997, RESIDUE): [
        *(
            (row, CLOCK_RESET_COLS, ("1e-12",), F.ALL_GRIDS)
            for row in ({}, {"tau": 10.37}, {"kc": 8.0, "c": 120.0})
        ),
        *(
            (row, CLOCK_RESET_COLS, ("1e-12",), ("u4", "u31", "offset31"))
            for row in ({"tau": 20.0}, {"c": 18.0, "tau": 16.0})
        ),
    ],
    (1799, MOVED_ROW): [
        (row, CLOCK_RESET_COLS, ("1e-06", "1e-08", "1e-10", "1e-12"), F.ALIGNED)
        for row in ({"tau": 20.0}, {"c": 18.0, "tau": 16.0})
    ],
}
CASES = tuple(
    F.with_marks(c, _known(c, _CLOCK_MARKS)) if c.name == "event_clock_reset" else c for c in CASES
)
