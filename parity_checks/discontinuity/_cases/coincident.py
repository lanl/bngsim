"""Coincident surfaces (issue #1915, phase 0 of #1781): two switches on one
instant, and a switch on the instant of an event, or a double from it (the
#1627 line).

Each case sweeps the two instants together, a double apart either way, 1e-9
apart and well apart. Where the columns have a value at the tie, the run must
give it; where they have none (the order of the two decides them), the run owes
a refusal of those columns, and an ulp from the tie it may decline them
(``Permitted``): whether instants a double apart are kept apart is the
contract's to say (#1916). The species have a value everywhere.

The references are closed forms read through exact indicators
(``_closed_forms.coincident_time``, ``coincident_state``, ``switch_event``), so a
sample on one of two instants a double apart is on that one only. Each is
checked against central differences of its own species, and against its spec
integrated branch by branch on the rows whose instants differ.
"""

from __future__ import annotations

import math

from _branchwise import Reset, Spec, Surface
from _case import Axes, Case, Judge, Switch, point
from _closed_forms import coincident_state, coincident_time, switch_event

from _cases import _families as F
from _cases.pulse import MOVED_ROW, _known
from _cases.step import STATE_GRIDS

#: What each mark fails with, and why.
TOO_CLOSE = (
    "SimulationError",
    "#1646: with a row on each of two instants a double apart, the sensitivity run fails with "
    "CV_TOO_CLOSE at the first",
)
MERGED = (
    AssertionError,
    "#2003: the row on one of two switches a double apart carries the other's column jump",
)
REFUSED = (
    "SensitivityRefusedDuringRunError",
    "#2004: an event a double after a row or a stop is refused with #950's refusal, though the "
    "columns it refuses have values",
)

UP = math.nextafter(25.0, math.inf)  # 25.000000000000004
DOWN = math.nextafter(25.0, 0.0)  # 24.999999999999996
#: #1627's doubles: 0.3, and 0.1*3 one double after it.
P3, P3UP = 0.3, 0.1 * 3

TIME_NET = """\
begin parameters
{params}
end parameters
begin functions
    1 rate() {law}
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 rate
end reactions
{groups}"""


def _net(params: dict, law: str, groups: str = "") -> str:
    lines = "\n".join(f"    {i} {k} {v!r}" for i, (k, v) in enumerate(params.items(), 1))
    return TIME_NET.format(params=lines, law=law, groups=groups)


def _spec(params, surfaces, rate) -> Spec:
    def rhs(x, s, t, p):
        flow = rate(x, s, t, p) * x["X()"]
        return {"T()": p["k_clock"], "X()": -flow, "Y()": flow}

    return Spec(
        states=("T()", "X()", "Y()"),
        params=tuple(params),
        rhs=rhs,
        surfaces=surfaces,
        x0=lambda p: {"T()": 0, "X()": 10, "Y()": 0},
    )


# ── Two time switches on one instant ────────────────────────────────────────

SUM_PARAMS = {"k0": 0.1, "d1": 0.2, "d2": 0.3, "c1": 25.0, "c2": 25.0, "k_clock": 2.0}
AND_PARAMS = {"k0": 0.1, "d1": 0.2, "c1": 25.0, "c2": 25.0, "k_clock": 2.0}
TIME_SURFACES = (
    Surface("c1", lambda x, t, p: t - p["c1"]),
    Surface("c2", lambda x, t, p: t - p["c2"]),
)
TIME_ROWS = (
    {},  # the tie, on a grid point
    {"c2": UP},  # a double after
    {"c2": DOWN},  # a double before
    {"c2": 25.0 + 1e-9},
    {"c2": 25.5},
    {"c1": 25.37, "c2": 25.37},  # the tie between grid points
    {"c1": P3, "c2": P3UP},  # #1627's doubles
)
SUM_COLS = ("k0", "d1", "d2", "c1", "c2")
AND_RATES, AND_COLS = ("k0", "d1"), ("k0", "d1", "c1", "c2")

# ── Two thresholds on one species ───────────────────────────────────────────

STATE_PARAMS = {"k0": 0.1, "d1": 0.2, "d2": 0.3, "th1": 3.0, "th2": 3.0, "k_clock": 2.0}
STATE_ROWS = (
    {},  # the tie: X falls through both at t = 2.0066
    {"th2": math.nextafter(3.0, 0.0)},  # th1 a double first
    {"th2": math.nextafter(3.0, math.inf)},  # th2 a double first
    {"th2": 3.0 * (1 - 1e-4)},  # apart by far more than any rtol locates a crossing to
    {"th2": 2.7},
    {"th2": 1.0},  # well apart: t = 2.01 and 4.75
    {"th1": 6.0, "th2": 6.0},  # the tie early, at t = 0.85
)
STATE_RATES, STATE_COLS = ("k0", "d1", "d2"), ("k0", "d1", "d2", "th1", "th2")
STATE_JUDGE = Judge(
    peak="window",
    converge=False,
    why=(
        "a state switch's crossing is located only to the run's own accuracy, about "
        "rtol*X/|X'|, so a row on it reads a column that jumps there at that distance into "
        "the far branch; scaled by its samples alone, a column that decays past the crossings "
        "(to e^-2.8 of itself by t = 30) would be judged against its tail. The convergence rows "
        "are off: on the tie at 6 the plain rows on aligned2 and aligned31 reach 66 times their "
        "tolerance at rtol 1e-12 on one run in 240, from at most 3.6 at the looser tolerances, "
        "and fail the rule there. Over 480 runs whose atol differed in its last digits, every "
        "point outside _flips passes on every run, under 67 times its tolerance"
    ),
)
SWITCH_EVENT_JUDGE = Judge(
    converge=False,
    why=(
        "the a column on the event's row is X just before the event, and at rtol 1e-12 it comes "
        "back about 8e-11 of itself off, 93 times its tolerance against 7.8 at 1e-10, where X "
        "itself is within 1.5 times its own: its convergence rows fail the rule on 52 to 185 of "
        "240 runs whose atol differed in its last digits. Every passing point stays under 146 "
        "times its tolerance on every one of 480 runs, 6.9 times under k"
    ),
)

# ── A switch on the instant of an event ─────────────────────────────────────

SWITCH_EVENT = """\
species X, T; X = 10; T = 0
k0 = 0.1; k1 = 0.5; c = 25; te = 25; a = 1; d = 1; kc = 2
J0: -> T; kc
J1: X -> ; piecewise(k1, time > c, k0)*X
E1: at (time >= te): X = a*X + d
"""
SWITCH_EVENT_PARAMS = {"k0": 0.1, "k1": 0.5, "c": 25.0, "te": 25.0, "a": 1.0, "d": 1.0, "kc": 2.0}
SWITCH_EVENT_SPEC = Spec(
    states=("X", "T"),
    params=tuple(SWITCH_EVENT_PARAMS),
    rhs=lambda x, s, t, p: {
        "X": -(p["k0"] + (p["k1"] - p["k0"]) * s[0]) * x["X"],
        "T": p["kc"],
    },
    surfaces=(Surface("switch", lambda x, t, p: t - p["c"]),),
    x0=lambda p: {"X": 10, "T": 0},
    resets=(
        Reset("E1", lambda x, t, p: t - p["te"], lambda x, t, p: {"X": p["a"] * x["X"] + p["d"]}),
    ),
)
SWITCH_EVENT_ROWS = (
    {},  # the tie: the event adds 1 as the rate steps from 0.1 to 0.5
    {"te": UP},  # the event a double after the switch
    {"te": DOWN},  # a double before it
    {"te": 25.5},
    {"te": 24.5},
    {"a": 0.0, "d": 5.0},  # the tie, the event setting X = 5
    {"a": 0.5, "d": 2.0, "te": 12.0},  # far from it, scaling X
    {"c": P3, "te": P3UP},  # #1627's doubles: the switch at 0.3, the event a double after
    {"c": P3UP, "te": P3},  # the event a double before the switch
)
SE_RATES, SE_COLS = ("k0", "k1", "a", "d"), ("k0", "k1", "a", "d", "c", "te")


def _marked(case, table):
    return F.with_marks(case, _known(case, table))


ALL = ("1e-06", "1e-08", "1e-10", "1e-12")
ON_BOTH = ("aligned2", "aligned31")

CASES = (
    Case(
        name="coincident_sum",
        family="coincident",
        model=_net(SUM_PARAMS, "k0+d1*if(time()>c1,1,0)+d2*if(time()>c2,1,0)"),
        fmt="net",
        switches=(Switch("c1", "time", "jump"), Switch("c2", "time", "jump")),
        params=SUM_PARAMS,
        t_span=(0.0, 30.0),
        observe=("T()", "X()", "Y()"),
        reference=coincident_time(interacting=False),
        spec=_spec(
            SUM_PARAMS, TIME_SURFACES, lambda x, s, t, p: p["k0"] + p["d1"] * s[0] + p["d2"] * s[1]
        ),
        axes=Axes(params=TIME_ROWS, grids=F.GRIDS, sens=((), SUM_COLS)),
        slice=(
            point(1e-12, F.Aligned(31), SUM_COLS),
            point(1e-10, F.Offset(31), SUM_COLS, c2=UP),
            point(1e-8, F.Aligned(2), SUM_COLS, c1=P3, c2=P3UP),
            point(1e-6, F.Uniform(31), SUM_COLS, c2=DOWN),
            point(1e-12, F.Uniform(4), (), c1=25.37, c2=25.37),
        ),
        note=(
            "X -> Y at k0 + d1*[t>c1] + d2*[t>c2]: two time switches, each adding its own term. "
            "At c1 = c2 the order does not matter, so every column has its value there. A double "
            "apart the case holds the run to keeping the two instants apart, as #1627 reads them "
            "(#2003's marks); if the contract (#1916) makes instants an ulp apart one, those marks "
            "come out."
        ),
    ),
    Case(
        name="coincident_and",
        family="coincident",
        model=_net(AND_PARAMS, "k0+d1*if(time()>c1&&time()>c2,1,0)"),
        fmt="net",
        switches=(Switch("c1", "time", "jump"), Switch("c2", "time", "jump")),
        params=AND_PARAMS,
        t_span=(0.0, 30.0),
        observe=("T()", "X()", "Y()"),
        reference=coincident_time(interacting=True),
        spec=_spec(AND_PARAMS, TIME_SURFACES, lambda x, s, t, p: p["k0"] + p["d1"] * s[0] * s[1]),
        axes=Axes(params=TIME_ROWS, grids=F.GRIDS, sens=((), AND_RATES, AND_COLS)),
        slice=(
            point(1e-12, F.Aligned(31), AND_COLS),
            point(1e-10, F.Offset(31), AND_RATES),
            point(1e-8, F.Aligned(2), AND_COLS, c2=25.5),
            point(1e-6, F.Uniform(31), AND_COLS, c2=UP),
            point(1e-12, F.Uniform(4), (), c1=P3, c2=P3UP),
        ),
        note=(
            "X -> Y at k0 + d1*[t>c1 && t>c2]: one term both switches turn on. At c1 = c2 the c "
            "columns have no value (0 from one side, X*d1 from the other), so the run owes "
            "their refusal; an ulp from the tie it may decline them. The rate columns have "
            "their values everywhere."
        ),
    ),
    Case(
        name="coincident_state",
        family="coincident",
        model=_net(
            STATE_PARAMS,
            "k0+d1*if(Xobs>th1,1,0)+d2*if(Xobs>th2,1,0)",
            "begin groups\n    1 Xobs 2\nend groups\n",
        ),
        fmt="net",
        switches=(Switch("th1", "state", "jump"), Switch("th2", "state", "jump")),
        params=STATE_PARAMS,
        t_span=(0.0, 30.0),
        observe=("T()", "X()", "Y()"),
        reference=coincident_state(),
        spec=_spec(
            STATE_PARAMS,
            (
                Surface("th1", lambda x, t, p: x["X()"] - p["th1"]),
                Surface("th2", lambda x, t, p: x["X()"] - p["th2"]),
            ),
            lambda x, s, t, p: p["k0"] + p["d1"] * s[0] + p["d2"] * s[1],
        ),
        axes=Axes(params=STATE_ROWS, grids=STATE_GRIDS, sens=((), STATE_RATES, STATE_COLS)),
        slice=(
            point(1e-12, F.Aligned(31), STATE_COLS),
            point(1e-10, STATE_GRIDS[-1], STATE_RATES),
            point(1e-8, F.Aligned(2), STATE_COLS, th2=1.0),
            point(1e-6, F.Uniform(31), STATE_COLS, th2=3.0 * (1 - 1e-4)),
            point(1e-12, F.Uniform(4), (), th1=6.0, th2=6.0),
        ),
        judge=STATE_JUDGE,
        note=(
            "X -> Y at k0 + d1*[X>th1] + d2*[X>th2]: X falls through two thresholds on itself. "
            "Their order is fixed by their values, since both read the same X. At th1 = th2 the "
            "th columns have no value (the crossing time moves with the rate between the two, "
            "which differs from each side), so the run owes their refusal, and may decline them "
            "an ulp from it; the rate columns have their values there."
        ),
    ),
    Case(
        name="switch_event",
        family="coincident",
        model=SWITCH_EVENT,
        fmt="antimony",
        switches=(Switch("switch", "time", "jump"), Switch("E1", "time", "jump")),
        params=SWITCH_EVENT_PARAMS,
        t_span=(0.0, 30.0),
        observe=("X", "T"),
        reference=switch_event(),
        spec=SWITCH_EVENT_SPEC,
        judge=SWITCH_EVENT_JUDGE,
        axes=Axes(params=SWITCH_EVENT_ROWS, grids=F.GRIDS, sens=((), SE_RATES, SE_COLS)),
        slice=(
            point(1e-12, F.Aligned(31), SE_COLS),
            point(1e-10, F.Offset(31), SE_RATES, a=0.0, d=5.0),
            point(1e-8, F.Aligned(2), SE_COLS, te=25.5),
            point(1e-6, F.Uniform(31), SE_COLS, a=0.5, d=2.0, te=12.0),
            point(1e-12, F.Uniform(4), (), c=P3, te=P3UP),
            point(1e-10, F.Aligned(31), (), c=P3UP, te=P3),
            point(1e-8, F.Uniform(31), SE_RATES, te=UP),  # #2004
        ),
        note=(
            "X decays at piecewise(k1, time > c, k0), and an event at time >= te sets "
            "X = a*X + d (a = 1 adds d, a = 0 sets X to d). The event sets the state, the switch "
            "moves its rate: at te = c the order does not move X, but dX/dte is d*k0*e^(..) from "
            "below and d*k1*e^(..) from above, so the c and te columns have no value there and "
            "the run owes their refusal. A double apart (#1627's 0.3 and 0.1*3, either way), the "
            "two must run in time order."
        ),
    ),
)

#: What fails on main, on every one of 480 runs whose atol differed in its last
#: digits (``_flips``): (row, columns, rtols, grids) per mark kind.
_MARKS = {
    "coincident_sum": {
        (1646, TOO_CLOSE): [
            (row, SUM_COLS, ALL, ON_BOTH)
            for row in ({"c2": UP}, {"c2": DOWN}, {"c1": P3, "c2": P3UP})
        ],
        (2003, MERGED): [
            ({"c2": UP}, SUM_COLS, ALL, ("u31", "offset31")),
            ({"c2": DOWN}, SUM_COLS, ALL, ("u31",)),
        ],
        (1799, MOVED_ROW): [({"c2": DOWN}, SUM_COLS, ALL, ("offset31",))],
    },
    "coincident_and": {
        (1799, MOVED_ROW): [
            ({"c2": UP}, AND_RATES, ALL, (*ON_BOTH, "offset31")),
            ({"c2": DOWN}, AND_RATES, ALL, ON_BOTH),
            ({"c1": P3, "c2": P3UP}, AND_RATES, ALL, ON_BOTH),
        ],
    },
    "switch_event": {
        (1646, TOO_CLOSE): [
            ({"te": UP}, SE_COLS, ALL, ON_BOTH),
            ({"te": DOWN}, SE_RATES, ALL, ON_BOTH),
            ({"c": P3, "te": P3UP}, SE_COLS, ALL, ON_BOTH),
            ({"c": P3UP, "te": P3}, SE_RATES, ALL, ON_BOTH),
        ],
        (2004, REFUSED): [
            ({"te": UP}, SE_RATES, ALL, ("u31", "aligned2", "aligned31", "offset31")),
            ({"c": P3, "te": P3UP}, SE_RATES, ALL, ON_BOTH),
        ],
    },
}
CASES = tuple(_marked(c, _MARKS[c.name]) if c.name in _MARKS else c for c in CASES)
