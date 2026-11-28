"""Pulse cases (issue #1912's seed; #1914 adds the family), and #1913's filed
bugs on pulses, each a strict expected failure until its fix lands."""

from __future__ import annotations

import sympy
from _branchwise import Spec, Surface
from _case import Aligned, Axes, Case, Judge, Known, Offset, Switch, Uniform, point
from _closed_forms import (
    gaussian_pulse,
    periodic_window,
    pulse_exposure,
    table_pulse,
    window_decay,
)

from _cases import _families as F

WINDOW_NET = """\
begin parameters
    1 kx 1
    2 t_on 7.3
    3 t_off 12.1
end parameters
begin functions
    1 window() kx*(if(time()>t_on,1,0)-if(time()>t_off,1,0))
end functions
begin species
    1 X() 10
    2 Y() 0
end species
begin reactions
    1 1 2 window
end reactions
"""

ALL = ("kx", "t_on", "t_off")


def _window_rhs(x, s, t, p):
    rate = p["kx"] * (s[0] - s[1]) * x["X()"]
    return {"X()": -rate, "Y()": rate}


WINDOW_SPEC = Spec(
    states=("X()", "Y()"),
    params=ALL,
    rhs=_window_rhs,
    surfaces=(
        Surface("on", lambda x, t, p: t - p["t_on"]),
        Surface("off", lambda x, t, p: t - p["t_off"]),
    ),
    x0=lambda p: {"X()": 10, "Y()": 0},
)

CASES = (
    Case(
        name="time_window",
        family="pulse",
        model=WINDOW_NET,
        fmt="net",
        switches=(Switch("on", "time", "jump"), Switch("off", "time", "jump")),
        params={"kx": 1.0, "t_on": 7.3, "t_off": 12.1},
        t_span=(0.0, 30.0),
        observe=("X()", "Y()"),
        reference=window_decay("X()", "Y()", 10.0),
        spec=WINDOW_SPEC,
        axes=Axes(
            params=(
                {},
                {"t_off": 7.31},  # narrow: a hundredth wide
                {"t_on": 10.0, "t_off": 20.0},  # on the uniform grids' points
                {"kx": 0.05, "t_on": 0.5, "t_off": 29.5},  # wide and slow
            ),
            # Aligned(2) is the window's ends and its crossings, nothing else:
            # #1984 showed only there.
            grids=(Uniform(4), Uniform(31), Aligned(2), Aligned(31), Offset(31)),
            sens=((), ALL),
        ),
        slice=(
            point(1e-10, Aligned(31), ALL),
            point(1e-6, Aligned(31), ALL),
            point(1e-12, Offset(31), ALL, t_off=7.31),
            point(1e-8, Uniform(31), ALL, t_on=10.0, t_off=20.0),
            point(1e-8, Uniform(4), ()),
            point(1e-10, Aligned(2), ALL),
        ),
        note="A rate switched on and off at two times: both times' columns jump.",
    ),
)


# ── #1913: filed bugs on pulses ─────────────────────────────────────────────

# #1657: a step table indexed by a function that reads the clock, h() = time().
TABLE_INDEX_NET = """\
begin species
    1 $Src() 1
    2 A() 0
end species
begin reactions
    1 1 1,2 F
end reactions
begin functions
    1 h() time()
    2 F() tfun([0,10,12,50],[0,5,0,0],h,method=>"step")
end functions
"""

TABLE_INDEX_SPEC = Spec(
    states=("Src()", "A()"),
    params=(),
    rhs=lambda x, s, t, p: {"Src()": 0, "A()": 5 * (s[0] - s[1]) * x["Src()"]},
    surfaces=(Surface("on", lambda x, t, p: t - 10), Surface("off", lambda x, t, p: t - 12)),
    x0=lambda p: {"Src()": 1, "A()": 0},
)

# #1805: a rate law on a parameter-free periodic window of the clock.
PERIODIC_NET = """\
begin parameters
    1 c 1.0
    2 kd 0.1
end parameters
begin functions
    1 L() if(sin(10*time())>0.99,c,0)
end functions
begin species
    1 X() 0
end species
begin reactions
    1 0 1 L
    2 1 0 kd
end reactions
"""

PERIODIC_SPEC = Spec(
    states=("X()",),
    params=("c", "kd"),
    rhs=lambda x, s, t, p: {"X()": p["c"] * s[0] - p["kd"] * x["X()"]},
    surfaces=(Surface("window", lambda x, t, p: sympy.sin(10 * t) - sympy.Rational(99, 100)),),
    x0=lambda p: {"X()": 0},
)

# #1795: a narrow Gaussian pulse of the clock, area kx, run by run_diffrax.
GAUSSIAN_NET = """\
begin parameters
    1 kx 100
    2 c 50
    3 s 0.05
end parameters
begin functions
    1 f() kx/(s*sqrt(_pi))*exp(-((time()-c)/s)^2)
end functions
begin species
    1 X() 0
end species
begin reactions
    1 0 1 f
end reactions
"""

GAUSSIAN_SPEC = Spec(
    states=("X()",),
    params=("kx", "c", "s"),
    rhs=lambda x, s, t, p: {
        "X()": p["kx"]
        / (p["s"] * sympy.sqrt(sympy.pi))
        * sympy.exp(-(((t - p["c"]) / p["s"]) ** 2))
    },
    surfaces=(),
    x0=lambda p: {"X()": 0},
)

CASES = (
    *CASES,
    Case(
        name="table_index_function",
        family="pulse",
        model=TABLE_INDEX_NET,
        fmt="net",
        switches=(Switch("on", "time", "jump"), Switch("off", "time", "jump")),
        params={},
        t_span=(0.0, 50.0),
        observe=("A()",),
        reference=table_pulse(),
        spec=TABLE_INDEX_SPEC,
        axes=Axes(rtol=(1e-8, 1e-12), grids=(Uniform(6), Uniform(5001), Aligned(6))),
        slice=(
            point(1e-8, Uniform(6)),
            point(1e-12, Uniform(5001)),
            point(1e-8, Aligned(6)),
            point(1e-12, Aligned(6)),
        ),
        judge=Judge(),
        known={
            f"table_index_function|rtol={r},grid={g},sens=none": Known(
                1657,
                AssertionError,
                "a step tfun indexed by h() = time() gets no breakpoint stops",
            )
            for r in ("1e-08", "1e-12")
            for g in ("u6", "u5001", "aligned6")
        },
        note=(
            "#1657: 0 -> A at a step table that is 5 on [10, 12), indexed by h() = time(), "
            "so A(50) = 10. The run steps over the pulse on every grid: rows at 10 and 12 "
            "are interpolated, not stopped at. No point passes, so the spec check, which "
            "reads bngsim's rate inside the pulse, is what shows the case is written right."
        ),
    ),
    Case(
        name="periodic_window",
        family="pulse",
        model=PERIODIC_NET,
        fmt="net",
        switches=(Switch("window", "time", "jump"),),
        params={"c": 1.0, "kd": 0.1},
        t_span=(0.0, 10.0),
        observe=("X()",),
        reference=periodic_window(),
        spec=PERIODIC_SPEC,
        axes=Axes(rtol=(1e-8, 1e-10), grids=(Uniform(2),), sens=((), ("kd",), ("c",))),
        slice=(
            point(1e-10, Uniform(2), ("kd",)),
            point(1e-8, Uniform(2), ("c",)),
            point(1e-10, Uniform(2)),
            point(1e-8, Uniform(2)),
        ),
        judge=Judge(
            converge=False,
            why=(
                "the plain run's error is not monotone in rtol on any grid (0.3x its tolerance "
                "at 1e-8, 82x at 1e-10, 13x at 1e-12), which the convergence rule reads as an "
                "error that stops falling; each point stays well inside k = 1e3"
            ),
        ),
        known={
            f"periodic_window|rtol={r},grid=u2,sens={q}": Known(
                1805,
                "SensitivityUnsupportedError",
                "refused for every parameter, though no parameter moves the window's edges",
            )
            for r in ("1e-08", "1e-10")
            for q in ("kd", "c")
        },
        note=(
            "#1805: X is made at c while sin(10t) > 0.99. The window's 32 edges in (0, 10) "
            "are fixed numbers, so dX/dkd and dX/dc exist; the run refuses them both."
        ),
    ),
    Case(
        name="diffrax_gaussian",
        family="pulse",
        model=GAUSSIAN_NET,
        fmt="net",
        switches=(),
        params={"kx": 100.0, "c": 50.0, "s": 0.05},
        t_span=(0.0, 100.0),
        observe=("X()",),
        reference=gaussian_pulse(),
        spec=GAUSSIAN_SPEC,
        axes=Axes(
            params=({}, {"s": 0.5}),
            rtol=(1e-6, 1e-8, 1e-10),
            grids=(Uniform(3),),
            engines=("ode", "diffrax"),
        ),
        slice=(
            point(1e-8, Uniform(3), engine="diffrax"),
            point(1e-6, Uniform(3), engine="diffrax", s=0.5),
            point(1e-10, Uniform(3), engine="ode"),
            point(1e-6, Uniform(3), engine="ode", s=0.5),
        ),
        atol_ratio=1.0,
        judge=Judge(),
        known={
            f"{row}|rtol={r},grid=u3,sens=none,engine=diffrax": Known(
                1795,
                AssertionError,
                "run_diffrax places none of the smooth-pulse stops Simulator.run does",
            )
            for row in ("diffrax_gaussian", "diffrax_gaussian[s=0.5]")
            for r in ("1e-06", "1e-08", "1e-10")
        },
        note=(
            "#1795: 0 -> X at a Gaussian of the clock, area kx = 100, centred at 50. It has no "
            "switch (and declares none), but a step over it misses it whole, so it needs the "
            "stops Simulator.run places (#1286). run_diffrax places none and returns X(100) = 0. "
            "atol = rtol, as the issue ran it."
        ),
    ),
)


# ── #1914: the pulse family, swept ──────────────────────────────────────────
#
# A rectangle of area kx on a condition that reads time() or a counter clock,
# narrow and wide, and the Gaussian of the triangle's area and width that has
# no corner: the bend family's control (``bend.py``).


PULSE_JUDGE = Judge(
    k_state=5e3,
    k_sens=5e3,
    peak="window",
    converge=False,
    why=(
        "a pulse 0.003 wide drives its columns 1e4 to 1e5 times their end values inside it, "
        "and CVODES holds them to rtol of that running size there, so each column is scaled by "
        "its peak over the window (a grid with no sample inside the pulse would judge it against "
        "its end value). Over 263 runs whose atol differed by up to 2.5e-11 of itself, these "
        "cases' passing points reach 2.15e3 times that tolerance at rtol 1e-12 "
        "(pulse_clock[c=50.74], offset31, a plain run; triangle_clock at k_clock = 8, u4), and "
        "2.33e3 on another set of draws (pulse_clock at k_clock = 8, offset31, a plain run that "
        "usually reads about 2e2); the corner-free control gauss_clock, which has no switch, "
        "2.13e3, and gauss_time 962, on columns that cancel to 0 past the pulse (c and w); at "
        "1e-10 they reach 776. So k = 5e3 is 2.1 times the worst of them, and the marks of "
        "#1997 and #1998 judged by it lie 1.5 "
        "times above it at their lowest (triangle_clock on one run of 263) and 2.2 times "
        "otherwise. The convergence rows are off: from 1e-10 to 1e-12 the control's error grows "
        "about 25 to 30 times, which the rule reads as an error that stops falling"
    ),
)

TIME_ROWS = (
    {},  # 0.003 wide, height 333: the narrow pulse
    {"w": 0.1},
    {"w": 4.0},  # wide
    F.at25("time", t=25.37),  # between every uniform point
    {"kx": 0.003},  # narrow, height 1
    {"c": 2.5, "w": 0.1},  # early, where the doubles are finer
)
CLOCK_ROWS = (
    {},  # the counter at 50 across the pulse
    F.at25("clock", T0=-49.95),  # the counter at 0.05 across it
    F.at25("clock", k_clock=0.5),
    F.at25("clock", k_clock=8.0),
    {"w": 0.1},
    {"w": 4.0},
    F.at25("clock", t=25.37),
)
GAUSS_TIME_ROWS = ({}, {"w": 0.1}, F.at25("time", t=25.37))
GAUSS_CLOCK_ROWS = ({}, F.at25("clock", T0=-49.95), {"w": 0.1})


T0_ROW = F.at25("clock", T0=-49.95)
STALL = (
    "SimulationError",
    "#1995: a sensitivity run makes no progress at the time switch's stop, with h below the ulp of t",
)
CLOCK_PLAIN = (
    "SimulationError",
    "#1996: a plain run fails with CV_ERR_FAILURE just past the clock pulse's leading edge",
)
W_COLUMN = (
    AssertionError,
    "#1997: dX/dw ends about 1e-3 off where it is 0, at every rtol; on some step sequences the "
    "row on the trailing edge also comes back short of its time (#1799), which the judge reports "
    "first",
)
MOVED_ROW = (
    AssertionError,
    "#1799: a sensitivity run records the row on a stop up to 1e-12 from the time asked",
)


def _known(case, table):
    """``{(issue, raises, reason): cells}`` as Known marks."""
    return {
        k: Known(issue, raises, reason)
        for (issue, (raises, reason)), cells in table.items()
        for k in F.keys(case, cells)
    }


def _pulse_time():
    cols = F.columns("window", "time")
    case = F.family_case(
        "pulse_time",
        "pulse",
        "window",
        "time",
        TIME_ROWS,
        pulse_exposure("window", "time"),
        slice_=(
            point(1e-10, F.Uniform(4), cols),
            point(1e-12, F.Offset(31), cols, w=0.1),
            point(1e-12, F.Aligned(31), cols, w=4.0),
            point(1e-8, F.Aligned(2), cols),
            point(1e-6, F.Uniform(31), ()),
        ),
        judge=PULSE_JUDGE,
        note=(
            "A rectangle of area kx on time(), from c to c + w: both edges jump. Its sensitivity "
            "runs make no progress at the leading edge's stop once the jump is large there (#1995): "
            "0.003 wide (height 333) from rtol 1e-10, 0.1 wide at 1e-12."
        ),
    )
    return F.with_marks(
        case,
        _known(
            case,
            {
                (1995, STALL): [
                    ({}, cols, ("1e-10", "1e-12"), F.ALL_GRIDS),
                    ({"c": 25.37}, cols, ("1e-10", "1e-12"), F.ALL_GRIDS),
                    ({"w": 0.1}, cols, ("1e-12",), F.ALL_GRIDS),
                ]
            },
        ),
    )


def _pulse_clock():
    cols = F.columns("window", "clock")
    case = F.family_case(
        "pulse_clock",
        "pulse",
        "window",
        "clock",
        CLOCK_ROWS,
        pulse_exposure("window", "clock"),
        slice_=(
            point(1e-12, F.Uniform(4), ()),
            point(1e-12, F.Uniform(31), cols),
            point(1e-10, F.Offset(31), cols, k_clock=8.0, c=200.0),
            point(1e-12, F.Uniform(4), cols, **T0_ROW),
            point(1e-8, F.Aligned(2), cols),
            point(1e-6, F.Aligned(31), cols, w=4.0),
        ),
        judge=PULSE_JUDGE,
        note=(
            "A rectangle of area kx on a counter clock Tobs, from c to c + w in T. Its w column "
            "ends 1e-3 off where it is 0 (#1997), and its plain runs fail at rtol 1e-12 (#1996). "
            "With the counter at 0.05 across it (T0 = -49.95) the w column is 1,700 times "
            "nearer, and its points pass."
        ),
    )
    kc05, kc8, late = (
        F.at25("clock", k_clock=0.5),
        F.at25("clock", k_clock=8.0),
        F.at25("clock", t=25.37),
    )
    return F.with_marks(
        case,
        _known(
            case,
            {
                (1996, CLOCK_PLAIN): [
                    ({}, (), ("1e-12",), ("u4", "u31")),
                    (kc8, (), ("1e-12",), ("u4", "u31")),
                    (kc05, (), ("1e-12",), ("u31",)),
                ],
                (1997, W_COLUMN): [
                    (row, cols, rtols, F.ALL_GRIDS)
                    for row, rtols in (
                        ({}, ("1e-10", "1e-12")),
                        (kc8, ("1e-10", "1e-12")),
                        (late, ("1e-10", "1e-12")),
                        (kc05, ("1e-12",)),
                        ({"w": 0.1}, ("1e-12",)),
                    )
                ],
            },
        ),
    )


def _gauss(reads, rows, slice_, note):
    cols = F.columns("gauss", reads)
    case = F.family_case(
        f"gauss_{reads}",
        "pulse",
        "gauss",
        reads,
        rows,
        pulse_exposure("gauss", reads),
        slice_=slice_(cols),
        judge=PULSE_JUDGE,
        note=note,
    )
    marks = {(1799, MOVED_ROW): [({}, cols, tuple(F.RTOL_LABELS), ("aligned31", "offset31"))]}
    return F.with_marks(case, _known(case, marks))


CASES = (
    *CASES,
    _pulse_time(),
    _pulse_clock(),
    _gauss(
        "time",
        GAUSS_TIME_ROWS,
        lambda cols: (
            point(1e-10, F.Aligned(31), cols),
            point(1e-12, F.Uniform(4), cols),
            point(1e-6, F.Offset(31), ()),
            point(1e-8, F.Aligned(2), cols, w=0.1),
            point(1e-12, F.Uniform(31), cols, c=25.37),
        ),
        (
            "The control beside triangle_time: a Gaussian of the triangle's area, 1/2, and "
            "s = w/4, on time(). It has no corner and no switch, so it declares none. bngsim "
            "stops at its middle, t = 25, and a sensitivity run records a row asked for there "
            "an ulp short (#1799). It has no crossing, so its aligned31 and offset31 grids ask "
            "for the same rows; its rows are the triangles' narrow, 0.1-wide and off-grid ones."
        ),
    ),
    _gauss(
        "clock",
        GAUSS_CLOCK_ROWS,
        lambda cols: (
            point(1e-8, F.Offset(31), cols),
            point(1e-12, F.Uniform(31), cols),
            point(1e-12, F.Aligned(2), cols, **T0_ROW),
            point(1e-6, F.Uniform(4), ()),
            point(1e-10, F.Aligned(31), cols, w=0.1),
        ),
        (
            "The control beside triangle_clock: the Gaussian of gauss_time on a counter clock, no "
            "corner and no switch. With the counter at 0.05 its middle row has #1998's moved "
            "state on 107 of 263 runs whose atol differed by up to 2.5e-11 of itself (left out), "
            "where triangle_clock's apex rows have it on every run, at 1.2e4 to 6.2e4 times their "
            "tolerance. It has no crossing, so its aligned31 and offset31 grids ask for the same "
            "rows; its rows are the triangles' narrow and 0.1-wide ones."
        ),
    ),
)
