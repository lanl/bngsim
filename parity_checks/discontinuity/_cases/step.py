"""Step cases (issue #1912's seed; #1914 adds the family)."""

from __future__ import annotations

from _branchwise import Spec, Surface
from _case import Aligned, Axes, Case, Known, Offset, Switch, Uniform, point
from _closed_forms import clock_exposure

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
            point(1e-12, Offset(31), COLS, a=1.0, b=-1.0),
            point(1e-8, Uniform(31), COLS, a=2.5, b=0.5),
            point(1e-8, Uniform(4), ()),
            point(1e-10, Aligned(2), COLS),
        ),
        known={
            **{
                # Asked for a row at t = 25, the crossing, the sensitivity run
                # records it from one ulp short: 24.999999999999996 (macOS arm64).
                f"clock_step[a=2.5,b=0.5]|rtol=1e-12,grid={g},sens=kx+k_clock+c": Known(
                    1799,
                    AssertionError,
                    "a sensitivity run records the row at the crossing an ulp short",
                )
                for g in ("aligned2", "aligned31", "offset31")
            },
            **{
                # A row at or before the crossing leaves dX/dk_clock 286x the
                # tolerance at rtol 1e-12, where 1e-10 holds it to 6x.
                f"{row}|converge,grid={g},sens=kx+k_clock+c": Known(
                    1985, AssertionError, "the k_clock column stops converging at rtol 1e-12"
                )
                for row, grids in (
                    ("clock_step", ("u31", "aligned2", "aligned31", "offset31")),
                    ("clock_step[a=2.5,b=0.5]", ("u31",)),
                    ("clock_step[a=1.0,b=-1.0]", ("u31", "aligned31")),
                )
                for g in grids
            },
            **{
                # Rows at 0, 25 and 30 only: from the crossing on, dX/dk_clock
                # and dX/dc come back exactly 0 at rtol 1e-10.
                f"clock_step[a=1.0,b=-1.0]|{m},sens=kx+k_clock+c": Known(
                    1984,
                    AssertionError,
                    "the step's jump is lost when a row falls on the crossing",
                )
                for m in ("rtol=1e-10,grid=aligned2", "converge,grid=aligned2")
            },
        },
        note=(
            "#1776's step: a state switch the clock carries against its species. "
            "u31 holds t = 25, so a uniform grid samples the crossing too."
        ),
    ),
)
