"""Clock cases: switches a counter clock carries (issue #1913: filed bugs in
#1781's area, each a strict expected failure until its fix lands).

* #1779: a pulse on ``u = a*time() - b*Tobs`` with the species' share b far
  below the clock's, as a triangle, a step and a sawtooth. Closed forms from
  ``clock_exposure``. Each case also holds #1776's own row (a = 3, b = 1), which
  the #1776 fix covers, as its control.
* #1794: a narrow window on a unit-rate counter, through ``steady_state()``.
* #1807: an SBML counter-clock pulse, whose onset column the loader's extra
  root breaks.
"""

from __future__ import annotations

import dataclasses
import math

import sympy
from _branchwise import Spec, Surface
from _case import Axes, Case, Judge, Known, SteadyState, Switch, Uniform, point
from _closed_forms import clock_exposure, counter_onset
from _reference import Limit

# ── #1779: a crossing the clock carries far faster than its species ────────

_U = "a*time()-b*Tobs"
CLOCK_LAWS = {
    "triangle": f"kx*(1/w)*max(0,1-abs({_U}-c)/(w/2))",
    "step": f"kx*if({_U}>c,1,0)",
    "sawtooth": f"kx*(2/w)*if({_U}>c,if({_U}<c+w,1-({_U}-c)/w,0),0)",
}

CLOCK_NET = """\
begin parameters
    1 kx 1
    2 k_clock 2.0
    3 a 3.0
    4 b 0.05
    5 c 22.5
    6 w 0.003
end parameters
begin functions
    1 pulse() {law}
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 pulse
end reactions
begin groups
    1 Tobs 1
end groups
"""

CLOCK_COLS = ("kx", "k_clock")
CLOCK_PARAMS = ("kx", "k_clock", "a", "b", "c", "w")
#: #1776's own row: the #1776 fix covers it, so every shape passes there.
COVERED = {"a": 3.0, "b": 1.0}


def _clock_spec(shape: str, literal: tuple[float, float] | None = None) -> Spec:
    """The case's model by hand. ``literal`` is (a, b) written into the law as
    numbers, which the spec then reads as numbers too: moved, the parameters
    a and b would move nothing in such a text."""

    def v(x, t, p):
        a, b = literal if literal is not None else (p["a"], p["b"])
        return a * t - b * x["T()"] - p["c"]

    def rhs(x, s, t, p):
        w, vv = p["w"], v(x, t, p)
        if shape == "step":
            h = s[0]
        elif shape == "triangle":  # s: v > -w/2, v > 0, v > w/2
            h = (1 / w) * s[0] * (1 - s[2]) * (1 - (2 * s[1] - 1) * vv / (w / 2))
        else:  # sawtooth; s: v > 0, v > w
            h = (2 / w) * s[0] * (1 - s[1]) * (1 - vv / w)
        rate = p["kx"] * h * x["X()"]
        return {"T()": p["k_clock"], "X()": -rate, "Y()": rate}

    offsets = {"step": (0,), "triangle": (-1, 0, 1), "sawtooth": (0, 2)}[shape]
    surfaces = tuple(
        Surface(f"u{o:+d}", lambda x, t, p, o=o: v(x, t, p) - o * p["w"] / 2) for o in offsets
    )
    return Spec(
        states=("T()", "X()", "Y()"),
        params=CLOCK_PARAMS
        if literal is None
        else tuple(q for q in CLOCK_PARAMS if q not in ("a", "b")),
        rhs=rhs,
        surfaces=surfaces,
        x0=lambda p: {"T()": 0, "X()": 10, "Y()": 0},
    )


def _clock_case(shape, rows, known, slice_, note):
    spec = _clock_spec(shape)
    return Case(
        name=f"clock_{shape}_weak",
        family="clock",
        model=CLOCK_NET.format(law=CLOCK_LAWS[shape]),
        fmt="net",
        switches=tuple(
            Switch(s.switch, "state", "jump" if shape != "triangle" else "bend")
            for s in spec.surfaces
        ),
        params={"kx": 1.0, "k_clock": 2.0, "a": 3.0, "b": 0.05, "c": 0.0, "w": 0.003},
        derive=lambda p: {"c": 25.0 * (p["a"] - p["b"] * p["k_clock"])},
        t_span=(0.0, 30.0),
        observe=("T()", "X()", "Y()"),
        reference=clock_exposure(shape),
        spec=spec,
        spec_rows=(COVERED,),
        axes=Axes(
            params=rows,
            rtol=(1e-6, 1e-9, 1e-11),
            grids=(Uniform(4),),
            sens=((), CLOCK_COLS),
        ),
        slice=slice_,
        judge=Judge(converge=False, peak="samples" if shape == "step" else "window"),
        known=known,
        timeout=10.0,
        note=note,
    )


def _rows(*bs: float) -> tuple[dict, ...]:
    """a = 3 for b >= 0 and a = 1 for b < 0, as #1779 ran them; #1776's row last."""
    return (*({"a": 3.0 if b >= 0 else 1.0, "b": b} for b in bs), COVERED)


TRIANGLE_ROWS = _rows(0.05, 0.03, 0.01)
STEP_ROWS = _rows(0.08, 0.05, 0.03, 1e-10, -1e-10)
SAWTOOTH_ROWS = _rows(0.08, 0.05, 0.03, 0.01, 1e-3, 1e-4, 1e-6, 1e-10, 0.0, -1e-10, -1e-6, -1e-4)

STALL = ("SimulationError", "a stall: CVODE makes no progress at the crossing")
CUT = ("SimulationTimeout", "a stall, cut off by the case's time limit")
WRONG = (AssertionError, "wrong columns, with no error")
PLAIN_FAILS = ("SimulationError", "the plain run fails too at rtol 1e-11")


def _key(shape: str, b: float, rtol: str, cols: bool, literal: bool = False) -> str:
    sens = "+".join(CLOCK_COLS) if cols else "none"
    if literal:
        return f"clock_{shape}_literal|rtol={rtol},grid=u4,sens={sens}"
    a = 3.0 if b >= 0 else 1.0
    return f"clock_{shape}_weak[a={a!r},b={b!r}]|rtol={rtol},grid=u4,sens={sens}"


def _marks(shape: str, table, literal: bool = False) -> dict:
    """``(b, rtol, cols): (raises, reason)`` as ``Known`` marks of #1779."""
    return {
        _key(shape, b, r, cols, literal): Known(1779, raises, reason)
        for (b, r, cols), (raises, reason) in table.items()
    }


# What each fails with on macOS arm64, by (b, rtol, columns). #1779's table
# was run with a and b written into the rate law; here they are parameters, so
# that the sweep can move them, and the step at b = +-1e-10 and the sawtooth at
# b = 0 are right this way (the literal spelling is in the two cases below).
TRIANGLE_MARKS = _marks(
    "triangle",
    {
        (0.05, "1e-06", True): CUT,
        **{(b, r, True): STALL for b in (0.05, 0.03, 0.01) for r in ("1e-09", "1e-11")},
    },
)
STEP_MARKS = _marks(
    "step",
    {
        (0.08, "1e-09", True): WRONG,
        (0.05, "1e-09", True): WRONG,
        (0.03, "1e-06", True): WRONG,
        **{(b, "1e-11", True): CUT for b in (0.08, 0.05, 0.03)},
    },
)
SAWTOOTH_MARKS = _marks(
    "sawtooth",
    {
        **{(b, "1e-06", True): WRONG for b in (0.08, 0.05, 0.03, 0.01)},
        **{(b, "1e-09", True): STALL for b in (0.08, 0.05)},
        **{(b, "1e-09", True): CUT for b in (0.03, 0.01, 1e-3, 1e-4, 1e-6, 0.0, -1e-10)},
        **{(b, "1e-09", True): WRONG for b in (1e-10, -1e-6, -1e-4)},
        **{(b, "1e-09", False): CUT for b in (0.0, -1e-10)},
        **{
            (b, "1e-11", True): WRONG if b == 1e-10 else STALL
            for b in (0.08, 0.05, 0.03, 0.01, 1e-3, 1e-4, 1e-6, 1e-10, 0.0, -1e-10, -1e-6, -1e-4)
        },
        # The plain run at rtol 1e-11 fails on every row but b = 1e-10, #1776's
        # own covered row included, as #1779 says it does on main.
        **{
            (b, "1e-11", False): PLAIN_FAILS
            for b in (0.08, 0.05, 0.03, 0.01, 1e-3, 1e-4, 1e-6, 0.0, -1e-10, -1e-6, -1e-4, 1.0)
        },
    },
)


def _p(rtol, b=None, cols=True):
    row = COVERED if b is None else {"a": 3.0 if b >= 0 else 1.0, "b": b}
    return point(rtol, Uniform(4), CLOCK_COLS if cols else (), **row)


TRIANGLE_SLICE = (
    _p(1e-9, 0.03),
    _p(1e-11, 0.01),
    _p(1e-6),
    _p(1e-11),
    _p(1e-9, 0.03, cols=False),
)
STEP_SLICE = (_p(1e-9, 0.05), _p(1e-6, 0.03), _p(1e-11), _p(1e-6, cols=False))
SAWTOOTH_SLICE = (
    _p(1e-6, 0.05),
    _p(1e-9, 1e-10),
    _p(1e-11, 1e-10),
    _p(1e-11),
    _p(1e-9, cols=False),
    _p(1e-11, cols=False),
)

#: #1779's own spelling, a and b written into the law, for the row where the
#: spelling decides: the sawtooth at b = 0, whose pulse the run misses whole.
#: (The step at b = +-1e-10, "42% off" in #1779's table, is not a failure by
#: this judge either way: that column is 5.6e-11, inside its absolute
#: tolerance at loose rtol, and 473 times its tolerance at 1e-11.)
LITERAL = {"sawtooth": (3.0, 0.0)}
LITERAL_MARKS = {
    "sawtooth": _marks(
        "sawtooth",
        {
            **{(0.0, r, True): WRONG for r in ("1e-06", "1e-09", "1e-11")},
            (0.0, "1e-09", False): CUT,
            (0.0, "1e-11", False): PLAIN_FAILS,
        },
        literal=True,
    )
}


def _literal_case(shape: str) -> Case:
    a, b = LITERAL[shape]
    law = CLOCK_LAWS[shape].replace("a*time()-b*Tobs", f"{a!r}*time()-{b!r}*Tobs")
    base = _clock_case(shape, (COVERED,), {}, (_p(1e-6),), "")
    marks = LITERAL_MARKS[shape]
    return dataclasses.replace(
        base,
        name=f"clock_{shape}_literal",
        model=CLOCK_NET.format(law=law).replace("    4 b 0.05", f"    4 b {b!r}"),
        params={**base.params, "a": a, "b": b},
        spec=_clock_spec(shape, literal=(a, b)),
        reference=clock_exposure(shape),
        spec_rows=None,
        axes=dataclasses.replace(base.axes, params=({},)),
        slice=(
            point(1e-9, Uniform(4), CLOCK_COLS),
            point(1e-6, Uniform(4), CLOCK_COLS),
            point(1e-11, Uniform(4), CLOCK_COLS),
            point(1e-6, Uniform(4)),
        ),
        known=marks,
        note=(
            f"#1779's {shape} at b = {b!r}, with a and b written into the rate law as the "
            "issue ran it. With them as parameters the same row is right; spelled this way "
            "the run misses the pulse whole, Y(30) = 0, and its plain runs time out at rtol "
            "1e-9 and fail at 1e-11."
        ),
    )


# ── #1794: steady_state() over a narrow window on a unit-rate counter ───────

COUNTER_WINDOW_NET = """\
begin parameters
    1 kc 1.0
    2 k 100
    3 kd 0.5
end parameters
begin functions
    1 win() if((cobs>5)&&(cobs<5.01),k,0)
end functions
begin species
    1 c() 0
    2 A() 1
    3 B() 0
    4 D() 1
end species
begin reactions
    1 0 1 kc
    2 2 3 win
    3 4 0 kd
end reactions
begin groups
    1 cobs 1
    2 Aobs 2
end groups
"""

COUNTER_WINDOW_SPEC = Spec(
    states=("c()", "A()", "B()", "D()"),
    params=("kc", "k", "kd"),
    rhs=lambda x, s, t, p: {
        "c()": p["kc"],
        "A()": -p["k"] * s[0] * s[1] * x["A()"],
        "B()": p["k"] * s[0] * s[1] * x["A()"],
        "D()": -p["kd"] * x["D()"],
    },
    surfaces=(
        Surface("open", lambda x, t, p: x["c()"] - 5),
        Surface("shut", lambda x, t, p: sympy.Rational(501, 100) - x["c()"]),
    ),
    x0=lambda p: {"c()": 0, "A()": 1, "B()": 0, "D()": 1},
)


def _counter_window_limit(p):
    a = math.exp(-p["k"] * 0.01 / p["kc"])
    return {"A()": a, "B()": 1.0 - a, "D()": 0.0}


# ── #1807: an SBML counter-clock pulse and its onset column ────────────────

COUNTER_ONSET = """\
model m
 compartment C = 1
 species Tc in C, X in C
 Tc = 0; X = 0
 J0: -> Tc; 1
 J1: -> X; k0 + piecewise(piecewise(k1*((Tc-on)/D)^(a-1)*(1-(Tc-on)/D), Tc <= on + D, 0), Tc >= on, 0)
 J2: X -> ; kdeg*X
 a = 1.2; D = 4.0; k1 = 2.0; k0 = 0.1; kdeg = 0.3; on = 3.0
end
"""


def _onset_rhs(x, s, t, p):
    # Off the pulse the power is never taken (the model's piecewise does not
    # evaluate it either): the spec check moves a below 1, where 0^(a-1) fails.
    z = (x["Tc"] - p["on"]) / p["D"]
    pulse = sympy.Piecewise((p["k1"] * z ** (p["a"] - 1) * (1 - z), s[0] * s[1] > 0), (0, True))
    return {"Tc": 1, "X": p["k0"] + pulse - p["kdeg"] * x["X"]}


COUNTER_ONSET_SPEC = Spec(
    states=("Tc", "X"),
    params=("a", "D", "k1", "k0", "kdeg", "on"),
    rhs=_onset_rhs,
    surfaces=(
        Surface("onset", lambda x, t, p: x["Tc"] - p["on"]),
        Surface("end", lambda x, t, p: p["on"] + p["D"] - x["Tc"]),
    ),
    x0=lambda p: {"Tc": 0, "X": 0},
)

_NOTE = (
    "#1779: a counter T at k_clock and X -> Y at a {shape} on u = a*time() - b*Tobs, which "
    "runs at r = a - b*k_clock, centred at t = 25. With b far below the clock's share the "
    "run stalls, times out or returns wrong columns with no error. Every row's columns are "
    "kx and k_clock; a = 3, b = 1 is #1776's own row, which its fix covers, the control."
)

CASES = (
    _clock_case(
        "triangle",
        TRIANGLE_ROWS,
        TRIANGLE_MARKS,
        TRIANGLE_SLICE,
        _NOTE.format(shape="triangle")
        + " Judged by each column's window peak: inside the pulse the k_clock column runs "
        "about 3e4 times its end value.",
    ),
    _clock_case("step", STEP_ROWS, STEP_MARKS, STEP_SLICE, _NOTE.format(shape="step")),
    _clock_case(
        "sawtooth",
        SAWTOOTH_ROWS,
        SAWTOOTH_MARKS,
        SAWTOOTH_SLICE,
        _NOTE.format(shape="sawtooth") + " Judged by each column's window peak, as the triangle.",
    ),
    _literal_case("sawtooth"),
    Case(
        name="steady_counter_window",
        family="clock",
        model=COUNTER_WINDOW_NET,
        fmt="net",
        switches=(Switch("open", "state", "jump"), Switch("shut", "state", "jump")),
        params={"kc": 1.0, "k": 100.0, "kd": 0.5},
        t_span=(0.0, 60.0),
        observe=("A()", "B()", "D()"),
        reference=Limit(_counter_window_limit),
        spec=COUNTER_WINDOW_SPEC,
        axes=Axes(
            params=({}, {"kc": 1.0001}),
            rtol=(1e-6, 1e-8),
            grids=(SteadyState("integration"), SteadyState("newton")),
        ),
        slice=(
            point(1e-8, SteadyState("integration")),
            point(1e-6, SteadyState("newton")),
            point(1e-8, SteadyState("newton"), kc=1.0001),
            point(1e-6, SteadyState("integration"), kc=1.0001),
        ),
        judge=Judge(converge=False),
        known={
            f"steady_counter_window|rtol={r},grid=steady-{m},sens=none": Known(
                1794,
                AssertionError,
                "steady_state() steps over a window on a unit-rate counter clock",
            )
            for r in ("1e-06", "1e-08")
            for m in ("integration", "newton")
        },
        steady_mask=("A()", "B()", "D()"),
        note=(
            "#1794: A -> B at k = 100 while 5 < c < 5.01, c a counter at kc, and D decays so "
            "the march runs past c = 5. A's steady value is exp(-k*0.01/kc). At kc = 1 the "
            "counter is a clock, and the march neither roots nor stops on its conditions; at "
            "kc = 1.0001 it roots them and is right. The clock is masked out of the test, as "
            "the steady_state docstring says to do for a write-only accumulator."
        ),
    ),
    Case(
        name="counter_onset_sbml",
        family="clock",
        model=COUNTER_ONSET,
        fmt="antimony",
        switches=(Switch("onset", "state", "bend"), Switch("end", "state", "bend")),
        params={"a": 1.2, "D": 4.0, "k1": 2.0, "k0": 0.1, "kdeg": 0.3, "on": 3.0},
        t_span=(0.0, 10.0),
        observe=("Tc", "X"),
        reference=counter_onset(),
        spec=COUNTER_ONSET_SPEC,
        axes=Axes(rtol=(1e-4, 1e-6, 1e-8), grids=(Uniform(11),), sens=((), ("on",))),
        slice=(
            point(1e-6, Uniform(11), ("on",)),
            point(1e-8, Uniform(11), ("on",)),
            point(1e-4, Uniform(11), ("on",)),
            point(1e-8, Uniform(11)),
        ),
        judge=Judge(converge=False),
        known={
            "counter_onset_sbml|rtol=1e-06,grid=u11,sens=on": Known(
                1807,
                AssertionError,
                "the loader's extra root on Tc >= on breaks the comoving onset column",
            ),
            "counter_onset_sbml|rtol=1e-08,grid=u11,sens=on": Known(
                1807, "SimulationError", "the run makes no progress at the onset, t = 3"
            ),
        },
        note=(
            "#1807: #750's pulse on a counter clock Tc, k1*s^(a-1)*(1-s) while "
            "on <= Tc <= on + D, as Antimony. The SBML loader adds a root on Tc >= on that "
            "the .net path does not, and the onset column comes out 2.6% off at rtol 1e-6; "
            "tighter, the run makes no progress at t = 3."
        ),
    ),
)
