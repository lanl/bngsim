"""Bend cases (issue #1912's seed; #1914 adds the family)."""

from __future__ import annotations

from _branchwise import Branchwise, Spec, Surface
from _case import Aligned, Axes, Case, Offset, Switch, Uniform, point

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
