"""Pulse cases (issue #1912's seed; #1914 adds the family)."""

from __future__ import annotations

from _branchwise import Spec, Surface
from _case import Aligned, Axes, Case, Offset, Switch, Uniform, point
from _closed_forms import window_decay

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
            grids=(Uniform(4), Uniform(31), Aligned(31), Offset(31)),
            sens=((), ALL),
        ),
        slice=(
            point(1e-10, Aligned(31), ALL),
            point(1e-6, Aligned(31), ALL),
            point(1e-12, Offset(31), ALL, t_off=7.31),
            point(1e-8, Uniform(31), ALL, t_on=10.0, t_off=20.0),
            point(1e-8, Uniform(4), ()),
        ),
        note="A rate switched on and off at two times: both times' columns jump.",
    ),
)
