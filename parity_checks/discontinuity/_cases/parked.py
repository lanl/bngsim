"""Parked cases: a state held on a surface, or crawling up to one (issue #1913:
filed bugs in #1781's area, each a strict expected failure until its fix
lands).

* #1796: ``steady_state()`` stalls one ulp short of a threshold the state
  approaches slowly, and returns that state with ``converged=False``.
* #1991: ``run()`` of #1796's model refuses the slow approach as "at rest"
  when an output row falls inside the tolerance band short of the threshold.
* #1798: a rate law gated on a window of zero width that the state sits on,
  ``if(Bobs<=0 && Bobs>=0, kb, 0)``, loses the exit time's columns.
"""

from __future__ import annotations

from _branchwise import Spec, Surface
from _case import Axes, Case, Judge, Known, SteadyState, Switch, Uniform, point
from _closed_forms import parked_zero_window, slow_threshold
from _reference import Limit

# ── #1796: a slowly approached threshold, through steady_state() ────────────
#
# A starts 1e3*r short of 5 and rises at r while A < 5, so it reaches 5 at
# t = 1000; past 5 it relaxes to 6. Y relaxes to 0.01 while A < 5 and decays to 0
# past it.

SLOW_THRESHOLD_NET = """\
begin parameters
    1 r 1e-9
    2 kf 1
    3 ky 1
    4 a0 4.999999
end parameters
begin functions
    1 up() if(Aobs<5,r,kf)
    2 dn() if(Aobs<5,0,kf/6)
    3 yin() if(Aobs<5,0.01*ky,0)
end functions
begin species
    1 A() a0
    2 Y() 0.01
end species
begin reactions
    1 0 1 up
    2 1 0 dn
    3 0 2 yin
    4 2 0 ky
end reactions
begin groups
    1 Aobs 1
end groups
"""

SLOW_THRESHOLD_SPEC = Spec(
    states=("A()", "Y()"),
    params=("r", "kf", "ky", "a0"),
    rhs=lambda x, s, t, p: {
        "A()": s[0] * p["r"] + (1 - s[0]) * (p["kf"] - p["kf"] / 6 * x["A()"]),
        "Y()": s[0] * p["ky"] / 100 - p["ky"] * x["Y()"],
    },
    surfaces=(Surface("A<5", lambda x, t, p: 5 - x["A()"]),),
    x0=lambda p: {"A()": p["a0"], "Y()": 0.01},
)

# ── #1798: a zero-width window the state sits on ────────────────────────────
#
# A decays from 10; B is made at kp*(q - A) once A < q, so it sits at exactly 0
# until t* = ln(10/q)/a; Y is made at kb while B is 0. The spec's branch
# variables (g > 0) cannot hold "B is exactly 0", so there is none; the plain
# points and the kb column, which pass, stand in.

ZERO_WINDOW_NET = """\
begin parameters
    1 a 0.5
    2 q 2.0
    3 kb 1.0
    4 kp 1.0
end parameters
begin functions
    1 fB() kp*if(q-Aobs>0,q-Aobs,0)
    2 fY() if(Bobs<=0&&Bobs>=0,kb,0)
end functions
begin species
    1 A() 10
    2 B() 0
    3 Y() 0
end species
begin reactions
    1 1 0 a
    2 0 2 fB
    3 0 3 fY
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
end groups
"""

ZERO_COLS = ("a", "q", "kb")

CASES = (
    Case(
        name="steady_slow_threshold",
        family="parked",
        model=SLOW_THRESHOLD_NET,
        fmt="net",
        switches=(Switch("A<5", "state", "jump"),),
        params={"r": 1e-9, "kf": 1.0, "ky": 1.0, "a0": 0.0},
        derive=lambda p: {"a0": 5.0 - p["r"] * 1e3},
        t_span=(0.0, 1100.0),
        observe=("A()", "Y()"),
        reference=Limit(lambda p: {"A()": 6.0, "Y()": 0.0}),
        spec=SLOW_THRESHOLD_SPEC,
        axes=Axes(
            params=({}, {"r": 1e-6}),
            rtol=(1e-6, 1e-8, 1e-10),
            grids=(SteadyState("integration"), SteadyState("newton")),
        ),
        slice=(
            point(1e-8, SteadyState("integration")),
            point(1e-6, SteadyState("newton")),
            point(1e-8, SteadyState("newton"), r=1e-6),
            point(1e-10, SteadyState("integration"), r=1e-6),
            point(1e-6, SteadyState("integration")),
        ),
        judge=Judge(
            converge=False,
            why="a steady state's error is set by steady_state's tol, not its rtol",
        ),
        known={
            f"steady_slow_threshold|rtol={r},grid=steady-{m},sens=none": Known(
                1796,
                AssertionError,
                "steady_state() stalls one ulp short of a slowly approached threshold",
            )
            for r in ("1e-08", "1e-10")
            for m in ("integration", "newton")
        },
        spec_rows=({"r": 1e-6},),
        note=(
            "#1796: A rises at r to a threshold at 5, which it reaches at t = 1000, and past "
            "it relaxes to 6. The steady state is A = 6, Y = 0. At r = 1e-9 the march stalls "
            "one ulp short of 5 and returns that state with converged=False, from rtol 1e-8 "
            "on; at rtol 1e-6, and at r = 1e-6, it crosses and is right. The spec check reads "
            "r = 1e-6, as a plain run at r = 1e-9 meets #1991."
        ),
    ),
    Case(
        name="slow_threshold_run",
        family="parked",
        model=SLOW_THRESHOLD_NET,
        fmt="net",
        switches=(Switch("A<5", "state", "jump"),),
        params={"r": 1e-9, "kf": 1.0, "ky": 1.0, "a0": 0.0},
        derive=lambda p: {"a0": 5.0 - p["r"] * 1e3},
        t_span=(0.0, 1100.0),
        observe=("A()", "Y()"),
        reference=slow_threshold(),
        spec=SLOW_THRESHOLD_SPEC,
        spec_rows=({"r": 1e-7},),
        axes=Axes(
            params=({}, {"r": 1e-7}),
            rtol=(1e-6, 1e-8, 1e-10),
            grids=(Uniform(2), Uniform(31)),
        ),
        slice=(
            point(1e-8, Uniform(31)),
            point(1e-8, Uniform(2)),
            point(1e-6, Uniform(31)),
            point(1e-10, Uniform(31), r=1e-7),
        ),
        judge=Judge(
            converge=False,
            why=(
                "at r = 1e-7 the 31-row grid goes from 1.7x its tolerance at rtol 1e-8 to 43x at "
                "1e-10 (#1989's late crossing, smaller there), 1.4 times the convergence "
                "allowance: a knife edge, where the r = 1e-9 point is marked"
            ),
        ),
        known={
            "slow_threshold_run|rtol=1e-08,grid=u31,sens=none": Known(
                1991,
                "SimulationError",
                "a slow approach is refused as at rest when a row falls inside its band",
            ),
            "slow_threshold_run[r=1e-07]|rtol=1e-06,grid=u31,sens=none": Known(
                1991,
                "SimulationError",
                "the same refusal at r = 1e-7, where rtol 1e-6 leaves a wider band",
            ),
            "slow_threshold_run|rtol=1e-10,grid=u31,sens=none": Known(
                1989,
                AssertionError,
                "the slowly crossed switch is taken 1.3e-3 late, A(1026.7) 2.5e-6 low",
            ),
        },
        note=(
            "#1991, on #1796's model as a time course: at r = 1e-9 and rtol 1e-8, the row at "
            "t = 880 falls while A is 1.2e-7 short of 5, inside the 2.8e-7 band, and the run "
            "refuses it as come to rest, though r carries A onto the surface at t = 1000. "
            "Two rows run to A = 6, and so does r = 1e-7 but at rtol 1e-6, whose wider band "
            "it lands in at t = 880 too. At rtol 1e-10 the 31 rows also show #1989: the "
            "switch is taken 1.3e-3 late."
        ),
    ),
    Case(
        name="parked_zero_window",
        family="parked",
        model=ZERO_WINDOW_NET,
        fmt="net",
        switches=(Switch("source", "state", "bend"), Switch("zero", "state", "jump")),
        params={"a": 0.5, "q": 2.0, "kb": 1.0, "kp": 1.0},
        t_span=(0.0, 8.0),
        observe=("A()", "B()", "Y()"),
        reference=parked_zero_window(),
        axes=Axes(rtol=(1e-8, 1e-10), grids=(Uniform(81),), sens=((), ZERO_COLS)),
        slice=(
            point(1e-10, Uniform(81), ZERO_COLS),
            point(1e-8, Uniform(81), ZERO_COLS),
            point(1e-10, Uniform(81)),
        ),
        judge=Judge(),
        known={
            f"parked_zero_window|rtol={r},grid=u81,sens=a+q+kb": Known(
                1798,
                AssertionError,
                "a zero-width window the state sits on loses the exit time's columns",
            )
            for r in ("1e-08", "1e-10")
        },
        note=(
            "#1798: Y is made at kb while B sits at exactly 0, gated on "
            "if(Bobs<=0&&Bobs>=0,kb,0), so Y(8) = kb*t* with t* = ln(10/q)/a. The run gives "
            "dY/da = dY/dq = 0 for -6.44 and -1. The == spelling is refused."
        ),
    ),
)
