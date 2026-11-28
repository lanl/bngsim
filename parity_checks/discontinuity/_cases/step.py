"""Step cases (issue #1912's seed; #1914 adds the family), and #1913's filed
bugs on steps, each a strict expected failure until its fix lands."""

from __future__ import annotations

from _branchwise import Spec, Surface
from _case import Aligned, Axes, Case, Judge, Known, Offset, Sampled, Switch, Uniform, point
from _closed_forms import (
    clock_exposure,
    cube_switch,
    late_switch,
    table_kink_switch_time,
    table_step_switch_time,
)

# #1776's model, with the residual's coefficients as parameters so the sweep
# can move them. u = a*time() - b*Tobs runs at r = a - b*k_clock; c is placed
# so the step sits at t = 25.
CLOCK_STEP_NET = """\
begin parameters
    1 kx 1
    2 k_clock 2.0
    3 a 3.0
    4 b 1.0
    5 c 25.0
end parameters
begin functions
    1 law() kx*if(a*time()-b*Tobs>c,1,0)
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 law
end reactions
begin groups
    1 Tobs 1
end groups
"""

COLS = ("kx", "k_clock", "c")


def _clock_step_rhs(x, s, t, p):
    rate = p["kx"] * s[0] * x["X()"]
    return {"T()": p["k_clock"], "X()": -rate, "Y()": rate}


CLOCK_STEP_SPEC = Spec(
    states=("T()", "X()", "Y()"),
    params=("kx", "k_clock", "a", "b", "c"),
    rhs=_clock_step_rhs,
    surfaces=(Surface("u>c", lambda x, t, p: p["a"] * t - p["b"] * x["T()"] - p["c"]),),
    x0=lambda p: {"T()": 0, "X()": 10, "Y()": 0},
)

CASES = (
    Case(
        name="clock_step",
        family="step",
        model=CLOCK_STEP_NET,
        fmt="net",
        switches=(Switch("u>c", "state", "jump"),),
        params={"kx": 1.0, "k_clock": 2.0, "a": 3.0, "b": 1.0, "c": 25.0},
        derive=lambda p: {"c": 25.0 * (p["a"] - p["b"] * p["k_clock"])},
        t_span=(0.0, 30.0),
        observe=("T()", "X()", "Y()"),
        reference=clock_exposure("step"),
        spec=CLOCK_STEP_SPEC,
        axes=Axes(
            params=(
                {},  # the clock carries u up, the species back (b > 0)
                {"a": 2.5, "b": 0.5},
                {"a": 1.0, "b": -1.0},  # the species carry u with the clock
            ),
            # Aligned(2) is the window's ends and its crossings, nothing else:
            # #1984 showed only there.
            grids=(Uniform(4), Uniform(31), Aligned(2), Aligned(31), Offset(31)),
            sens=((), COLS),
        ),
        slice=(
            point(1e-10, Aligned(31), COLS),
            point(1e-6, Aligned(31), COLS),
            point(1e-12, Offset(31), (), a=1.0, b=-1.0),
            point(1e-8, Uniform(31), COLS, a=2.5, b=0.5),
            point(1e-8, Uniform(4), ()),
            point(1e-8, Aligned(2), COLS),
        ),
        # Every column point at rtol 1e-12 carries #1985, at 93-286x its
        # tolerance (macOS arm64 and Linux x86_64 alike), where rtol 1e-6 to 1e-10 hold every column
        # within 31x. A limit of 50 sits between the two with room either side.
        # The convergence rows are off: #1985 grows by 4x to 46x from rtol
        # 1e-10 to 1e-12 across the groups, a spread no such rule could split
        # without flapping, and the limit carries the check instead.
        judge=Judge(
            k_sens=50.0,
            converge=False,
            why=(
                "#1985: every column point at rtol 1e-12 is 93-286x its tolerance, where 1e-6 "
                "to 1e-10 hold every column within 31x, so k_sens = 50 sits between the two. "
                "From 1e-10 to 1e-12 the error grows 4x to 46x across the groups, a spread no "
                "convergence rule could split without flapping, so the limit carries the check."
            ),
        ),
        known={
            **{
                # Asked for a row at t = 25, the crossing, the sensitivity run
                # records it from one ulp short: 24.999999999999996 (macOS and Linux).
                f"clock_step[a=2.5,b=0.5]|rtol=1e-12,grid={g},sens=kx+k_clock+c": Known(
                    1799,
                    AssertionError,
                    "a sensitivity run records the row at the crossing an ulp short",
                )
                for g in ("aligned2", "aligned31", "offset31")
            },
            **{
                f"{row}|rtol=1e-12,grid={g},sens=kx+k_clock+c": Known(
                    1985, AssertionError, "the k_clock column stops converging at rtol 1e-12"
                )
                for row, grids in (
                    ("clock_step", ("u4", "u31", "aligned2", "aligned31", "offset31")),
                    ("clock_step[a=2.5,b=0.5]", ("u4", "u31")),
                    (
                        "clock_step[a=1.0,b=-1.0]",
                        ("u4", "u31", "aligned2", "aligned31", "offset31"),
                    ),
                )
                for g in grids
            },
            # Rows at 0, 25 and 30 only: from the crossing on, dX/dk_clock and
            # dX/dc come back exactly 0 at rtol 1e-10.
            "clock_step[a=1.0,b=-1.0]|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c": Known(
                1984, AssertionError, "the step's jump is lost when a row falls on the crossing"
            ),
        },
        note=(
            "#1776's step: a state switch the clock carries against its species. "
            "u31 holds t = 25, so a uniform grid samples the crossing too."
        ),
    ),
)


# ── #1913: filed bugs on steps ──────────────────────────────────────────────

# #1797, case 1: a fitted switch time on a step table's breakpoint.
TABLE_STEP_NET = """\
begin parameters
    1 a 2
    2 k 0.5
    3 q 0.7
    4 tau 3.0
end parameters
begin functions
    1 tf() tfun([0,3],[0,1],time(),method=>"step")
    2 fY() if(time()>=tau,k*Xo,0)
    3 fW() q*Xo*tf()
    4 fX() a
end functions
begin species
    1 X() 0
    2 Y() 0
    3 W() 0
end species
begin reactions
    1 0 1 fX
    2 0 2 fY
    3 0 3 fW
end reactions
begin groups
    1 Xo 1
end groups
"""

TABLE_STEP_SPEC = Spec(
    states=("X()", "Y()", "W()"),
    params=("a", "k", "q", "tau"),
    rhs=lambda x, s, t, p: {
        "X()": p["a"],
        "Y()": s[0] * p["k"] * x["X()"],
        "W()": s[1] * p["q"] * x["X()"],
    },
    surfaces=(
        Surface("tau", lambda x, t, p: t - p["tau"]),
        Surface("table", lambda x, t, p: t - 3),
    ),
    x0=lambda p: {"X()": 0, "Y()": 0, "W()": 0},
)

# #1797, case 2: the same breakpoint under a gate, where dY/dtau has a kink.
TABLE_KINK_NET = """\
begin parameters
    1 r 1
    2 tau 3.0
end parameters
begin functions
    1 g() tfun([0,3],[0.5,1],time(),method=>"step")
    2 fY() r*if(time()>=tau,1,0.25)*g()
end functions
begin species
    1 Y() 0
end species
begin reactions
    1 0 1 fY
end reactions
"""

TABLE_KINK_SPEC = Spec(
    states=("Y()",),
    params=("r", "tau"),
    rhs=lambda x, s, t, p: {"Y()": p["r"] * (0.25 + 0.75 * s[0]) * (0.5 + 0.5 * s[1])},
    surfaces=(
        Surface("tau", lambda x, t, p: t - p["tau"]),
        Surface("table", lambda x, t, p: t - 3),
    ),
    x0=lambda p: {"Y()": 0},
)

# #1802: an integer-power state switch, compiled. A = 1 + eps*t crosses
# c^(1/3) at t* = 2. The thresholds c are the issue's thr**3 (the platform's
# pow) for thr 6 ulp either side of 1 + 2e-8 - ulp, written as literals so every
# platform has the same rows; offset 0 is the issue's own c.
CUBE_NET = """\
begin parameters
    1 eps 1e-8
    2 c 1.0000000600000007
    3 kb 3
end parameters
begin functions
    1 fY() if(Aobs^3>c,kb,0)
end functions
begin species
    1 A() 1
    2 Y() 0
end species
begin reactions
    1 0 1 eps
    2 0 2 fY
end reactions
begin groups
    1 Aobs 1
end groups
"""

CUBE_SPEC = Spec(
    states=("A()", "Y()"),
    params=("eps", "c", "kb"),
    rhs=lambda x, s, t, p: {"A()": p["eps"], "Y()": s[0] * p["kb"]},
    surfaces=(Surface("cube", lambda x, t, p: x["A()"] ** 3 - p["c"]),),
    x0=lambda p: {"A()": 1, "Y()": 0},
)

#: c by ulp offset of thr. On macOS arm64 and Linux x86_64 alike the compiled
#: run never takes the switch at -5, -3, 0, +3 and +6 (#1802); at every offset
#: both engines take it about 7e-13 of A late (#1989).
CUBE_C = {
    -6: 1.0000000599999967,
    -5: 1.0000000599999974,
    -4: 1.000000059999998,
    -3: 1.0000000599999987,
    -2: 1.0000000599999994,
    -1: 1.00000006,
    0: 1.0000000600000007,
    1: 1.0000000600000014,
    2: 1.000000060000002,
    3: 1.0000000600000027,
    4: 1.0000000600000034,
    5: 1.000000060000004,
    6: 1.0000000600000047,
}
CUBE_NEVER = (-5, -3, 0, 3, 6)
#: The control: at eps = 1e-2 an ulp of A moves t* by 2e-14, and both engines
#: take the switch on time.
CUBE_CONTROL = {"eps": 1e-2, "c": 1.0612079999999993}


def _cube_row(c: float) -> str:
    return "cube_switch" if c == CUBE_C[0] else f"cube_switch[c={c!r}]"


CUBE_KNOWN = {
    **{
        f"{_cube_row(c)}|rtol=1e-10,grid=u2,sens=none,engine=ode-interpreted": Known(
            1989, AssertionError, "a slowly crossed state switch is taken 7e-13 of A late"
        )
        for c in CUBE_C.values()
    },
    **{
        f"{_cube_row(c)}|rtol=1e-10,grid=u2,sens=none,engine=ode-compiled": (
            Known(1802, AssertionError, "pow() and ExprTk disagree on the side: never switched")
            if k in CUBE_NEVER
            else Known(1989, AssertionError, "a slowly crossed state switch is taken late")
        )
        for k, c in CUBE_C.items()
    },
}

# #1836: a switch late in a long run, where the doubles are coarse.
LATE_NET = """\
begin parameters
    1 c0 0.5
    2 sigma 1048574.0
    3 k 2
    4 rc 1
end parameters
begin functions
    1 rate_X() if(t>=sigma,k*(1-x),0)
end functions
begin species
    1 C() c0
    2 X() 0
end species
begin reactions
    1 0 1 rc
    2 0 2 rate_X
end reactions
begin groups
    1 t 1
    2 x 2
end groups
"""

LATE_SPEC = Spec(
    states=("C()", "X()"),
    params=("c0", "sigma", "k", "rc"),
    rhs=lambda x, s, t, p: {"C()": p["rc"], "X()": s[0] * p["k"] * (1 - x["X()"])},
    surfaces=(Surface("late", lambda x, t, p: x["C()"] - p["sigma"]),),
    x0=lambda p: {"C()": p["c0"], "X()": 0},
)

#: sigma = 2^e - 2, so the crossing is 2.5 before a window that ends at 2^e.
#: Not 2^30 or 2^40, where the outcome rides on the step sequence: over 2^30 the
#: columns at rtol 1e-8 failed 3 of 30 runs whose atol differed by 1 part in
#: 1e12, and its plain run at 1e-10 ranged 6.5x to 854x its tolerance; over 2^40
#: the plain run at 1e-8 sat at 735x. Over 2^46 and 2^50 every point fails, and
#: over 2^20 every point passes.
LATE_SIGMA = {e: 2.0**e - 2.0 for e in (20, 46, 50)}

CASES = (
    *CASES,
    Case(
        name="table_step_switch_time",
        family="step",
        model=TABLE_STEP_NET,
        fmt="net",
        switches=(Switch("tau", "time", "jump"), Switch("table", "time", "jump")),
        params={"a": 2.0, "k": 0.5, "q": 0.7, "tau": 3.0},
        t_span=(0.0, 7.5),
        observe=("X()", "Y()", "W()"),
        reference=table_step_switch_time(),
        spec=TABLE_STEP_SPEC,
        axes=Axes(
            params=({}, {"tau": 3.3}),
            rtol=(1e-8, 1e-10),
            grids=(Sampled(16),),
            sens=((), ("tau",)),
        ),
        slice=(
            point(1e-10, Sampled(16), ("tau",)),
            point(1e-8, Sampled(16), ("tau",)),
            point(1e-10, Sampled(16), ("tau",), tau=3.3),
            point(1e-8, Sampled(16)),
        ),
        judge=Judge(),
        known={
            f"table_step_switch_time|rtol={r},grid=sampled16,sens=tau": Known(
                1797,
                AssertionError,
                "a switch time on a step tfun's breakpoint takes the table's jump into its column",
            )
            for r in ("1e-08", "1e-10")
        },
        note=(
            "#1797, case 1: Y switches on at tau, and W is made at q*X*tf() with a step table "
            "tf that jumps at 3. W never reads tau, so dW/dtau = 0, also with tau on the "
            "breakpoint; the run gives q*X(3) = -4.2. At tau = 3.3 it is right."
        ),
    ),
    Case(
        name="table_kink_switch_time",
        family="step",
        model=TABLE_KINK_NET,
        fmt="net",
        switches=(Switch("tau", "time", "jump"), Switch("table", "time", "jump")),
        params={"r": 1.0, "tau": 3.0},
        t_span=(0.0, 6.0),
        observe=("Y()",),
        reference=table_kink_switch_time(),
        spec=TABLE_KINK_SPEC,
        axes=Axes(
            params=({}, {"tau": 3.3}),
            rtol=(1e-8, 1e-10),
            grids=(Sampled(5),),
            sens=((), ("tau",)),
        ),
        slice=(
            point(1e-10, Sampled(5), ("tau",)),
            point(1e-8, Sampled(5), ("tau",)),
            point(1e-10, Sampled(5), ("tau",), tau=3.3),
            point(1e-8, Sampled(5)),
        ),
        judge=Judge(),
        known={
            f"table_kink_switch_time|rtol={r},grid=sampled5,sens=tau": Known(
                1797,
                AssertionError,
                "a switch time on a step tfun's breakpoint, at a kink, returns neither side",
            )
            for r in ("1e-08", "1e-10")
        },
        note=(
            "#1797, case 2: Y is made at r*if(time()>=tau,1,0.25)*g() with a step table g "
            "that doubles at 3. At tau = 3, dY/dtau is -0.375 from below and -0.75 from "
            "above, so the run owes a refusal; it returns -0.875. At tau = 3.3 it is right."
        ),
    ),
    Case(
        name="cube_switch",
        family="step",
        model=CUBE_NET,
        fmt="net",
        switches=(Switch("cube", "state", "jump"),),
        params={"eps": 1e-8, "c": CUBE_C[0], "kb": 3.0},
        t_span=(0.0, 2.05),
        observe=("A()", "Y()"),
        reference=cube_switch(),
        spec=CUBE_SPEC,
        axes=Axes(
            params=(*({"c": c} for c in CUBE_C.values() if c != CUBE_C[0]), {}, CUBE_CONTROL),
            rtol=(1e-10,),
            grids=(Uniform(2),),
            engines=("ode-interpreted", "ode-compiled"),
        ),
        slice=(
            point(1e-10, Uniform(2), engine="ode-compiled"),
            point(1e-10, Uniform(2), engine="ode-interpreted"),
            point(1e-10, Uniform(2), engine="ode-compiled", **CUBE_CONTROL),
            point(1e-10, Uniform(2), engine="ode-interpreted", **CUBE_CONTROL),
        ),
        judge=Judge(
            k_state=1e5,
            converge=False,
            why=(
                "one ulp of A moves t* by 2.2e-8 at eps = 1e-8, and Y by 6.6e-8, 4e3 times its "
                "tolerance at rtol 1e-10, so k_state = 1e5 allows about 20 ulp; the case runs at "
                "one rtol, so it has no convergence row either"
            ),
        ),
        known=CUBE_KNOWN,
        timeout=20.0,
        spec_rows=(CUBE_CONTROL,),
        note=(
            "#1802: Y is made at kb once Aobs^3 > c, with A = 1 + eps*t crossing at t* = 2. "
            "Compiled, pow() and ExprTk's residual disagree about the side at the stalled A, "
            "and the switch is never taken (Y = 0 for 0.15). Both engines also take it 7e-13 "
            "of A late at eps = 1e-8 (#1989). k_state is 1e5: one ulp of A moves t* by "
            "2.2e-8 there, Y by 6.6e-8, 4e3 times the tolerance at rtol 1e-10, so 1e5 allows "
            "about 20 ulp. For the same reason the case runs at rtol 1e-10 only."
        ),
    ),
    Case(
        name="late_switch",
        family="step",
        model=LATE_NET,
        fmt="net",
        switches=(Switch("late", "state", "jump"),),
        params={"c0": 0.5, "sigma": LATE_SIGMA[20], "k": 2.0, "rc": 1.0},
        t_span=lambda p: (0.0, p["sigma"] + 2.0),
        observe=("C()", "X()"),
        reference=late_switch(),
        spec=LATE_SPEC,
        axes=Axes(
            params=tuple({"sigma": s} for s in LATE_SIGMA.values()),
            rtol=(1e-8, 1e-10),
            grids=(Sampled(2),),
            sens=((), ("sigma",)),
        ),
        slice=(
            point(1e-10, Sampled(2), ("sigma",), sigma=LATE_SIGMA[20]),
            point(1e-8, Sampled(2), sigma=LATE_SIGMA[20]),
            point(1e-10, Sampled(2), ("sigma",), sigma=LATE_SIGMA[46]),
            point(1e-10, Sampled(2), sigma=LATE_SIGMA[50]),
            point(1e-8, Sampled(2), ("sigma",), sigma=LATE_SIGMA[50]),
        ),
        judge=Judge(
            converge=False,
            why=(
                "over 2^46 and 2^50 #1836's error grows as rtol tightens, which its marks hold; "
                "over 2^20 the column at rtol 1e-10 ranges 18x to 315x its tolerance as atol "
                "moves by 1 part in 1e12, too near the convergence allowance (about 400x) to hold"
            ),
        ),
        known={
            **{
                f"late_switch[sigma={LATE_SIGMA[e]!r}]|rtol={r},grid=sampled2,sens={q}": Known(
                    1836,
                    AssertionError,
                    "a switch late in a long run is taken where the doubles are coarse",
                )
                for e in (46, 50)
                for r in ("1e-08", "1e-10")
                for q in ("none", "sigma")
            },
        },
        note=(
            "#1836: a counter from 0.5 switches X on when it reaches sigma = 2^e - 2, 2.5 before "
            "the window's end at 2^e. Past 2^30 the restart's small steps round where the "
            "doubles are coarse; over 2^50 the jump is lost."
        ),
    ),
)
