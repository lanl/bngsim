"""Bend cases (issue #1912's seed; #1914 adds the family)."""

from __future__ import annotations

from _branchwise import Branchwise, Spec, Surface
from _case import Aligned, Axes, Case, Judge, Offset, Switch, Uniform, point
from _closed_forms import abs_bend, min_bend, pulse_exposure

from _cases import _families as F
from _cases.pulse import CLOCK_ROWS, MOVED_ROW, PULSE_JUDGE, T0_ROW, TIME_ROWS, _known
from _cases.step import MOVED_STATE

DECAY_BEND_NET = """\
begin parameters
    1 kd 0.2
    2 kp 0.5
    3 th 3.0
end parameters
begin functions
    1 source() kp*max(0,th-Xobs)
end functions
begin species
    1 X() 10
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 0 2 source
end reactions
begin groups
    1 Xobs 1
end groups
"""

COLS = ("kd", "kp", "th")

DECAY_BEND_SPEC = Spec(
    states=("X()", "Y()"),
    params=COLS,
    rhs=lambda x, s, t, p: {
        "X()": -p["kd"] * x["X()"],
        "Y()": p["kp"] * (p["th"] - x["X()"]) * s[0],
    },
    surfaces=(Surface("max", lambda x, t, p: p["th"] - x["X()"]),),
    x0=lambda p: {"X()": 10, "Y()": 0},
)

CASES = (
    Case(
        name="decay_bend",
        family="bend",
        model=DECAY_BEND_NET,
        fmt="net",
        switches=(Switch("max", "state", "bend"),),
        params={"kd": 0.2, "kp": 0.5, "th": 3.0},
        t_span=(0.0, 30.0),
        observe=("X()", "Y()"),
        # Integrated branch by branch, so the sweep exercises that reference;
        # _closed_forms.decay_bend is its closed form, which a frame test
        # holds it to.
        reference=Branchwise(DECAY_BEND_SPEC, expect=1),
        spec=DECAY_BEND_SPEC,
        axes=Axes(
            params=(
                {},  # t* = 6.0
                {"th": 9.0},  # t* = 0.53, early
                {"kd": 0.05, "th": 5.0},  # t* = 13.9, slow
            ),
            # Aligned(2) is the window's ends and its crossings, nothing else:
            # #1984 showed only there.
            grids=(Uniform(4), Uniform(31), Aligned(2), Aligned(31), Offset(31)),
            sens=((), COLS),
        ),
        slice=(
            point(1e-10, Aligned(31), COLS),
            point(1e-6, Aligned(31), COLS),
            point(1e-12, Offset(31), COLS, th=9.0),
            point(1e-8, Uniform(31), COLS, kd=0.05, th=5.0),
            point(1e-8, Uniform(4), ()),
            point(1e-10, Aligned(2), COLS),
        ),
        note="A source that starts as its decaying species falls through a threshold: a bend on the state.",
    ),
)


# ── #1914: the bend family, swept ───────────────────────────────────────────
#
# The corners of max and abs in a triangle of area kx/2 on time() and on a
# counter clock (#1285, #1456 and #1777's model, which ``pulse.py``'s
# Gaussians stand beside as corner-free controls), and the corners of min and
# abs on a decaying state.


MIN_NET = """\
begin parameters
    1 vmax 1.0
    2 k 0.5
    3 kd 0.2
end parameters
begin functions
    1 source() min(vmax,k*Xobs)
end functions
begin species
    1 X() 10
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 0 2 source
end reactions
begin groups
    1 Xobs 1
end groups
"""
MIN_COLS = ("vmax", "k", "kd")
MIN_SPEC = Spec(
    states=("X()", "Y()"),
    params=MIN_COLS,
    rhs=lambda x, s, t, p: {
        "X()": -p["kd"] * x["X()"],
        "Y()": s[0] * p["k"] * x["X()"] + (1 - s[0]) * p["vmax"],
    },
    surfaces=(Surface("min", lambda x, t, p: p["vmax"] - p["k"] * x["X()"]),),
    x0=lambda p: {"X()": 10, "Y()": 0},
)

ABS_NET = """\
begin parameters
    1 kx 1.0
    2 th 3.0
    3 kd 0.2
end parameters
begin functions
    1 source() kx*abs(Xobs-th)
end functions
begin species
    1 X() 10
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 0 2 source
end reactions
begin groups
    1 Xobs 1
end groups
"""
ABS_COLS = ("kx", "th", "kd")
ABS_SPEC = Spec(
    states=("X()", "Y()"),
    params=ABS_COLS,
    rhs=lambda x, s, t, p: {
        "X()": -p["kd"] * x["X()"],
        "Y()": p["kx"] * (2 * s[0] - 1) * (x["X()"] - p["th"]),
    },
    surfaces=(Surface("abs", lambda x, t, p: x["X()"] - p["th"]),),
    x0=lambda p: {"X()": 10, "Y()": 0},
)


TRI = {r: F.columns("triangle", r) for r in ("time", "clock")}


def _triangle(reads):
    cols = TRI[reads]
    if reads == "time":
        slice_ = (
            point(1e-8, Aligned(2), cols),
            point(1e-12, Uniform(31), cols),
            point(1e-12, Offset(31), cols, w=4.0),
            point(1e-10, Uniform(4), cols, c=25.37),
            point(1e-6, Aligned(31), ()),
        )
        note = (
            "A triangle of area kx/2 on time(), max(0, 1 - |v|/(w/2))/w: corners of max at its "
            "feet and of abs at its apex, t = 25. gauss_time is its corner-free control. bngsim's "
            "stop at its near foot, c - w/2, falls an ulp past the double the grid asks for, and "
            "a sensitivity run records the row there (#1799)."
        )
        marks = {
            (1799, MOVED_ROW): [
                (row, cols, tuple(F.RTOL_LABELS), F.ALIGNED)
                for row in (
                    {},
                    {"w": 0.1},
                    F.at25("time", t=25.37),
                    {"kx": 0.003},
                    {"c": 2.5, "w": 0.1},
                )
            ]
        }
    else:
        kc05, kc8, late = (
            F.at25("clock", k_clock=0.5),
            F.at25("clock", k_clock=8.0),
            F.at25("clock", t=25.37),
        )
        slice_ = (
            point(1e-12, Uniform(31), cols, **T0_ROW),
            point(1e-12, Aligned(2), cols, **kc8),
            point(1e-12, Uniform(4), cols, **T0_ROW),
            point(1e-10, Aligned(31), cols),
            point(1e-8, Offset(31), cols, w=4.0),
            point(1e-6, Uniform(31), ()),
        )
        note = (
            "#1285, #1456 and #1777's triangle on a counter clock, its apex at t = 25. "
            "gauss_clock is its corner-free control. At rtol 1e-12 the 0.003-wide pulse's rows on "
            "and beside its crossings carry #1998's moved state, at 7.7e3 to 2.5e5 times their "
            "tolerance (1.2e4 to 6.2e4 with the counter at 0.05 across it: #1777's lead); u4, "
            "with no row there, is right, and so are the wider pulses (w = 0.1 and 4). At "
            "k_clock = 0.5 the state is moved on 116 of 263 runs, so those rows are left out."
        )
        marks = {
            (1998, MOVED_STATE): [
                ({}, cols, ("1e-12",), F.ON_CROSSING),
                (T0_ROW, cols, ("1e-12",), F.ON_CROSSING),
                (kc8, cols, ("1e-12",), F.ON_CROSSING),
                (kc05, cols, ("1e-12",), ("aligned2",)),
                (late, cols, ("1e-12",), ("aligned2", "aligned31", "offset31")),
            ]
        }
    case = F.family_case(
        f"triangle_{reads}",
        "bend",
        "triangle",
        reads,
        TIME_ROWS if reads == "time" else CLOCK_ROWS,
        pulse_exposure("triangle", reads),
        slice_=slice_,
        judge=PULSE_JUDGE,
        note=note,
    )
    return F.with_marks(
        case,
        _known(case, marks),
    )


def _state_slice(cols):
    return (
        point(1e-6, Uniform(4), cols),
        point(1e-8, Uniform(31), ()),
        point(1e-10, Aligned(2), cols),
        point(1e-12, Aligned(31), cols),
        point(1e-8, Offset(31), cols),
    )


CASES = (
    *CASES,
    _triangle("time"),
    _triangle("clock"),
    Case(
        name="min_state",
        family="bend",
        model=MIN_NET,
        fmt="net",
        switches=(Switch("min", "state", "bend"),),
        params={"vmax": 1.0, "k": 0.5, "kd": 0.2},
        t_span=(0.0, 30.0),
        observe=("X()", "Y()"),
        reference=min_bend(),
        spec=MIN_SPEC,
        axes=Axes(
            params=(
                {},  # t* = 8.05
                {"vmax": 4.0},  # t* = 1.12
                {"kd": 0.06},  # t* = 26.8, late
                {"kd": 2.0},  # t* = 0.80, fast
                {"k": 0.09},  # k*X never reaches vmax: no crossing
            ),
            grids=F.GRIDS,
            sens=((), MIN_COLS),
        ),
        slice=_state_slice(MIN_COLS),
        spec_rows=({}, {"vmax": 4.0}),
        judge=Judge(
            converge=False,
            why=(
                "its plain points pass at every rtol, but from 2x their tolerance at rtol 1e-8 "
                "they reach 32x at 1e-10 and 78x at 1e-12 (aligned2, and u4 at kd = 2), past the "
                "rule's floor of 30, so the convergence rows sit at 1.05 to 2.2 times their "
                "allowance while every point stays under 1e2"
            ),
        ),
        note="Y is made at min(vmax, k*X) on a decaying X: a corner of min on the state.",
    ),
    Case(
        name="abs_state",
        family="bend",
        model=ABS_NET,
        fmt="net",
        switches=(Switch("abs", "state", "bend"),),
        params={"kx": 1.0, "th": 3.0, "kd": 0.2},
        t_span=(0.0, 30.0),
        observe=("X()", "Y()"),
        reference=abs_bend(),
        spec=ABS_SPEC,
        axes=Axes(
            params=(
                {},  # t* = 6.02
                {"th": 9.0},  # t* = 0.53
                {"kd": 0.05, "th": 5.0},  # t* = 13.9, slow
                {"kd": 2.0, "kx": 0.7},  # t* = 0.60, fast
            ),
            grids=F.GRIDS,
            sens=((), ABS_COLS),
        ),
        slice=_state_slice(ABS_COLS),
        judge=Judge(
            converge=False,
            why=(
                "on some step sequences the kd column at the corner comes back about 1e-7 of "
                "itself off at rtol 1e-10 (#1999), on 2 to 5 of 263 runs whose atol differed by up "
                "to 2.5e-11 of itself; those points are left out (_flips). The convergence rows "
                "read the same columns, and would ride on whether a run hits it"
            ),
        ),
        note=(
            "Y is made at kx*|X - th| on a decaying X: a corner of abs on the state. On some step "
            "sequences its kd column at the corner is about 1e-7 of itself off at rtol 1e-10 "
            "(#1999); those points are left out."
        ),
    ),
)
