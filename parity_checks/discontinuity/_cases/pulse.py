"""Pulse cases (issue #1912's seed; #1914 adds the family), and #1913's filed
bugs on pulses, each a strict expected failure until its fix lands."""

from __future__ import annotations

import sympy
from _branchwise import Spec, Surface
from _case import Aligned, Axes, Case, Judge, Known, Offset, Switch, Uniform, point
from _closed_forms import gaussian_pulse, periodic_window, table_pulse, window_decay

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
        judge=Judge(converge=False),
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
        judge=Judge(converge=False),
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
        judge=Judge(converge=False),
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
