"""Sawtooth cases (issue #1914, phase 0 of #1781): a pulse that jumps up at c
and falls back to 0 along a line by c + w, a jump then a bend, on time(), on a
counter clock, and on #1776's residual ``a*time() - b*Tobs``.

#1779's weakly coupled sawtooths are in ``clock.py``; ``sawtooth_coupled`` is
#1776's own model swept, with its healthy couplings as rows."""

from __future__ import annotations

import dataclasses

from _case import point
from _closed_forms import pulse_exposure

from _cases import _families as F
from _cases.pulse import (
    CLOCK_PLAIN,
    CLOCK_ROWS,
    MOVED_ROW,
    PULSE_JUDGE,
    STALL,
    T0_ROW,
    TIME_ROWS,
    _known,
)
from _cases.step import MOVED_STATE

#: The clock-carried sawtooths' judge: PULSE_JUDGE at a looser k. At rtol 1e-12
#: #1998's moved state puts their rows on and beside the jump at 3.0e3 to 7.8e3
#: times their tolerance, the same value on every draw of atol, and a plain
#: run's species reach 4.3e3 on some draws (sawtooth_clock[c=50.74], offset31).
#: 1e4 is 1.3 times the first and 2.3 times the second, so #1998 is marked on
#: these cases only where it passes 1e4 (k_clock = 8).
CLOCK_SAWTOOTH_JUDGE = dataclasses.replace(
    PULSE_JUDGE,
    k_state=1e4,
    k_sens=1e4,
    why=PULSE_JUDGE.why
    + ". On a clock-carried sawtooth k is 1e4: at rtol 1e-12 #1998's moved state puts the rows "
    "on and beside its jump at 3.0e3 to 7.8e3 times their tolerance, the same on every draw of "
    "atol, and a plain run's species reach 4.3e3 on some draws (sawtooth_clock[c=50.74], "
    "offset31); 1e4 sits 1.3 times above the first and 2.3 times above the second",
)

COUPLED_ROWS = (
    {},  # a = 3, b = 1: #1776's own model, its covered row
    {"a": 2.5, "b": 0.5},
    {"a": 1.0, "b": -1.0},  # the species carry u with the clock
    {"w": 0.1},
)


COLS = {r: F.columns("sawtooth", r) for r in ("time", "clock", "coupled")}
KC8 = F.at25("clock", k_clock=8.0)


def _case(reads, rows, slice_, judge, note, marks, exclude=None, derive=None):
    case = F.family_case(
        f"sawtooth_{reads}",
        "sawtooth",
        "sawtooth",
        reads,
        rows,
        pulse_exposure("sawtooth", reads),
        slice_=slice_,
        judge=judge,
        note=note,
        derive=derive,
    )
    return F.with_marks(case, _known(case, marks), exclude and exclude(case))


CASES = (
    _case(
        "time",
        TIME_ROWS,
        (
            point(1e-10, F.Aligned(31), COLS["time"]),
            point(1e-12, F.Uniform(4), COLS["time"], kx=0.003),
            point(1e-12, F.Offset(31), COLS["time"], w=4.0),
            point(1e-8, F.Aligned(2), COLS["time"], c=25.37),
            point(1e-6, F.Uniform(31), ()),
        ),
        PULSE_JUDGE,
        (
            "A sawtooth of area kx on time(): a jump up at c, a bend back to 0 at c + w. Its "
            "sensitivity runs stall at the jump's stop (#1995): 0.003 wide from rtol 1e-10, and "
            "at 1e-12 even at height 2 (kx = 0.003) or 0.1 wide."
        ),
        {
            (1995, STALL): [
                ({}, COLS["time"], ("1e-10", "1e-12"), F.ALL_GRIDS),
                ({"c": 25.37}, COLS["time"], ("1e-10", "1e-12"), F.ALL_GRIDS),
                *(
                    (row, COLS["time"], ("1e-12",), F.ALL_GRIDS)
                    for row in ({"w": 0.1}, {"kx": 0.003}, {"c": 2.5, "w": 0.1})
                ),
            ]
        },
    ),
    _case(
        "clock",
        CLOCK_ROWS,
        (
            point(1e-12, F.Uniform(4), ()),
            point(1e-12, F.Aligned(31), COLS["clock"], **KC8),
            point(1e-12, F.Uniform(31), COLS["clock"]),
            point(1e-10, F.Offset(31), COLS["clock"], **T0_ROW),
            point(1e-8, F.Aligned(2), COLS["clock"], w=4.0),
            point(1e-6, F.Offset(31), ()),
        ),
        CLOCK_SAWTOOTH_JUDGE,
        (
            "The sawtooth of sawtooth_time on a counter clock Tobs. At rtol 1e-12 its plain runs "
            "fail on n_points 4 (#1996), and its sensitivity runs carry #1998's moved state from "
            "the jump on: on rows on and beside it at 5.4e3 times their tolerance, inside this "
            "case's k of 1e4, and at k_clock = 8 at 1.6e4."
        ),
        {
            (1996, CLOCK_PLAIN): [
                (row, (), ("1e-12",), ("u4",))
                for row in ({}, F.at25("clock", k_clock=0.5), KC8, F.at25("clock", t=25.37))
            ],
            (1998, MOVED_STATE): [(KC8, COLS["clock"], ("1e-12",), F.ON_CROSSING)],
        },
        lambda case: {
            **{
                k: F.flip(8, what=": the rest had #1998's moved state beside the jump at 4.3e4")
                for k in F.keys(case, [(T0_ROW, COLS["clock"], ("1e-12",), ("offset31",))])
            },
            **{
                k: F.flip(9, what=": the rest had #1998's moved state beside the jump at 4.4e4")
                for k in F.keys(
                    case, [(F.at25("clock", t=25.37), COLS["clock"], ("1e-12",), ("offset31",))]
                )
            },
        },
    ),
    _case(
        "coupled",
        COUPLED_ROWS,
        (
            point(1e-12, F.Uniform(4), ()),
            point(1e-6, F.Aligned(2), COLS["coupled"], a=1.0, b=-1.0),
            point(1e-12, F.Aligned(31), COLS["coupled"]),
            point(1e-10, F.Offset(31), COLS["coupled"], a=2.5, b=0.5),
            point(1e-8, F.Uniform(31), COLS["coupled"], w=0.1),
        ),
        CLOCK_SAWTOOTH_JUDGE,
        (
            "#1776's sawtooth: u = a*time() - b*Tobs runs at r = a - b*k_clock, and c is placed "
            "so the pulse starts at t = 25. a = 3, b = 1 is #1776's own model. Its plain run fails "
            "at rtol 1e-12 on n_points 4 (#1996); #1998's moved state puts its rows on the jump at "
            "3.7e3 to 7.2e3 times their tolerance, inside this case's k of 1e4."
        ),
        {
            (1996, CLOCK_PLAIN): [
                (row, (), ("1e-12",), ("u4",)) for row in ({}, {"a": 1.0, "b": -1.0})
            ],
            (1799, MOVED_ROW): [({"a": 1.0, "b": -1.0}, COLS["coupled"], ("1e-06",), F.ALIGNED)],
        },
        derive=lambda p: {"c": 25.0 * (p["a"] - p["b"] * p["k_clock"])},
    ),
)
