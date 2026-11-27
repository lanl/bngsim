"""Step cases (issue #1912's seed; #1914 adds the family)."""

from __future__ import annotations

from _branchwise import Spec, Surface
from _case import Aligned, Axes, Case, Offset, Switch, Uniform, point
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
            grids=(Uniform(4), Uniform(31), Aligned(31), Offset(31)),
            sens=((), COLS),
        ),
        slice=(
            point(1e-10, Aligned(31), COLS),
            point(1e-6, Aligned(31), COLS),
            point(1e-12, Offset(31), COLS, a=1.0, b=-1.0),
            point(1e-8, Uniform(31), COLS, a=2.5, b=0.5),
            point(1e-8, Uniform(4), ()),
        ),
        note=(
            "#1776's step: a state switch the clock carries against its species. "
            "u31 holds t = 25, so a uniform grid samples the crossing too."
        ),
    ),
)
