"""The model texts, specs and axes #1914's families share (issue #1914, phase 0
of #1781): pulses, steps and sawtooths on a condition linear in t, and the
corners of max, min and abs.

A family case is one shape of rate law on one spelling of its condition:

* ``time``: ``time()`` itself. The crossing time is a formula of the
  parameters, so bngsim stops on it.
* ``clock``: ``Tobs``, a counter clock ``0 -> T`` at ``k_clock`` from ``T0``.
  bngsim finds such a clock from the model's structure and stops on it too.
* ``coupled``: #1776's residual ``a*time() - b*Tobs``.

Every model keeps the clock T, read or not. Its exact value, ``T0 + k_clock*t``,
is a witness of which time the state a run reports belongs to: #1914's sweep
first showed a row carrying the state of a later time that way.

Each pulse is placed so its crossing sits at t = 25, a point of the
``Uniform(31)`` grid (the window's or sawtooth's leading edge, a triangle's or
Gaussian's middle). A row at ``c`` for t = 25.37 puts it between every uniform
point. The references are ``_closed_forms.pulse_exposure`` and ``level_step``.
"""

from __future__ import annotations

import sympy
from _branchwise import Spec, Surface
from _case import Aligned, Axes, Case, Offset, Switch, Uniform

#: The grids every family case sweeps: none near a crossing (u4), one on it
#: (u31 holds t = 25), the window's ends and its crossings alone (#1984 showed
#: only there), a sample on every crossing, and one a hair either side.
GRIDS = (Uniform(4), Uniform(31), Aligned(2), Aligned(31), Offset(31))

#: A condition's spelling in a rate law.
U = {"time": "time()", "clock": "Tobs", "coupled": "(a*time()-b*Tobs)"}

#: Each shape's rate law, in u; its switch offsets of v = u - c (as multiples
#: of w), with what each does to the rate there.
LAWS = {
    "window": "if({u}>c&&{u}<c+w,kx/w,0)",
    "triangle": "kx*(1/w)*max(0,1-abs({u}-c)/(w/2))",
    "sawtooth": "kx*(2/w)*if({u}>c,if({u}<c+w,1-({u}-c)/w,0),0)",
    "gauss": "0.5*kx/((w/4)*sqrt(_pi))*exp(-(({u}-c)/(w/4))^2)",
    "step": "if({u}>c,k1,k0)",
}
SURFACES = {
    "window": ((0.0, "jump"), (1.0, "jump")),
    "triangle": ((-0.5, "bend"), (0.0, "bend"), (0.5, "bend")),
    "sawtooth": ((0.0, "jump"), (1.0, "bend")),
    "gauss": (),
    "step": ((0.0, "jump"),),
}

NET = """\
begin parameters
{params}
end parameters
begin functions
    1 rate() {law}
end functions
begin species
    1 T() {t0}
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 rate
end reactions
begin groups
    1 Tobs 1
end groups
"""


def base_params(shape: str, reads: str) -> dict[str, float]:
    """Every parameter a family case's model holds, at its base row: a pulse
    0.003 wide in u, its crossing at t = 25."""
    p = {"k0": 0.1, "k1": 1.0} if shape == "step" else {"kx": 1.0}
    p["k_clock"] = 2.0
    if reads == "coupled":
        p.update({"a": 3.0, "b": 1.0})
    p["c"] = {"time": 25.0, "clock": 50.0, "coupled": 25.0}[reads]
    if shape != "step":
        p["w"] = 0.003
    if reads == "clock":
        p["T0"] = 0.0
    return p


def columns(shape: str, reads: str) -> tuple[str, ...]:
    """The columns a family case asks for: every parameter its law reads."""
    own = ("k0", "k1") if shape == "step" else ("kx",)
    line = ("c",) if reads == "time" else ("k_clock", "c")
    return own + line + (() if shape == "step" else ("w",))


def net(shape: str, reads: str) -> str:
    params = base_params(shape, reads)
    lines = "\n".join(f"    {i} {k} {v!r}" for i, (k, v) in enumerate(params.items(), 1))
    return NET.format(
        params=lines,
        law=LAWS[shape].format(u=U[reads]),
        t0="T0" if reads == "clock" else "0",
    )


def spec(shape: str, reads: str) -> Spec:
    """The model by hand, with one branch variable per surface ``v - o*w``."""
    params = tuple(base_params(shape, reads))

    def u(x, t, p):
        if reads == "time":
            return t
        if reads == "clock":
            return x["T()"]
        return p["a"] * t - p["b"] * x["T()"]

    def rhs(x, s, t, p):
        if shape == "step":
            rate = p["k0"] + (p["k1"] - p["k0"]) * s[0]
        else:
            w, v = p["w"], u(x, t, p) - p["c"]
            if shape == "window":
                h = s[0] * (1 - s[1]) / w
            elif shape == "triangle":  # s: v > -w/2, v > 0, v > w/2
                h = (1 / w) * s[0] * (1 - s[2]) * (1 - (2 * s[1] - 1) * v / (w / 2))
            elif shape == "sawtooth":  # s: v > 0, v > w
                h = (2 / w) * s[0] * (1 - s[1]) * (1 - v / w)
            else:
                q = w / 4
                h = sympy.Rational(1, 2) / (q * sympy.sqrt(sympy.pi)) * sympy.exp(-((v / q) ** 2))
            rate = p["kx"] * h
        flow = rate * x["X()"]
        return {"T()": p["k_clock"], "X()": -flow, "Y()": flow}

    def surface(o):
        def g(x, t, p):
            w = p.get("w", 0)
            return u(x, t, p) - p["c"] - o * w

        return g

    return Spec(
        states=("T()", "X()", "Y()"),
        params=params,
        rhs=rhs,
        surfaces=tuple(Surface(_switch_name(o), surface(o)) for o, _ in SURFACES[shape]),
        x0=lambda p: {"T()": p["T0"] if reads == "clock" else 0, "X()": 10, "Y()": 0},
    )


def _switch_name(o: float) -> str:
    return "v" if o == 0 else f"v{o:+g}w"


def switches(shape: str, reads: str) -> tuple[Switch, ...]:
    what = "time" if reads == "time" else "state"
    return tuple(Switch(_switch_name(o), what, effect) for o, effect in SURFACES[shape])


def at25(reads: str, *, k_clock: float = 2.0, T0: float = 0.0, t: float = 25.0, **row) -> dict:
    """A row whose crossing sits at time t: c for the condition to reach there."""
    out = dict(row)
    if reads == "clock":
        out["c"] = T0 + k_clock * t
        if k_clock != 2.0:
            out["k_clock"] = k_clock
        if T0 != 0.0:
            out["T0"] = T0
    elif t != 25.0:
        out["c"] = t
    return out


def family_case(
    name: str,
    family: str,
    shape: str,
    reads: str,
    rows: tuple[dict, ...],
    reference,
    *,
    slice_,
    note: str,
    judge=None,
    known=None,
    exclude=None,
    timeout: float = 20.0,
    derive=None,
    rtol=None,
) -> Case:
    """One shape on one spelling of its condition, swept over ``rows``, every
    tolerance, ``GRIDS`` and its columns."""
    kw = {} if judge is None else {"judge": judge}
    return Case(
        name=name,
        family=family,
        model=net(shape, reads),
        fmt="net",
        switches=switches(shape, reads),
        params=base_params(shape, reads),
        derive=derive,
        t_span=(0.0, 30.0),
        observe=("T()", "X()", "Y()"),
        reference=reference,
        spec=spec(shape, reads),
        axes=Axes(
            params=rows,
            grids=GRIDS,
            sens=((), columns(shape, reads)),
            **({} if rtol is None else {"rtol": rtol}),
        ),
        slice=slice_,
        known=known or {},
        exclude=exclude or {},
        timeout=timeout,
        note=note,
        **kw,
    )


# ── Marks and exclusions ────────────────────────────────────────────────────

#: A grid by its label, for writing marks.
BY_LABEL = {g.label: g for g in GRIDS}
RTOL_LABELS = {"1e-06": 1e-6, "1e-08": 1e-8, "1e-10": 1e-10, "1e-12": 1e-12}


def keys(case: Case, cells) -> list[str]:
    """The keys of ``cells``: (row, columns, rtols, grid labels) each, the
    columns ``()`` for a plain run; a key the sweep has not is an error."""
    from _case import CaseError, point

    out = []
    grids = {g.label: g for g in case.axes.grids}
    for row, sens, rtols, labels in cells:
        for r in rtols:
            for g in labels:
                k = case.key(point(RTOL_LABELS[r], grids[g], tuple(sens), **row))
                if k not in {case.key(p) for p in case._product()}:
                    raise CaseError(f"case {case.name!r}: no point {k!r} to mark")
                out.append(k)
    return out


def with_marks(case: Case, known: dict) -> Case:
    """``case`` with its known failures. Its coin flips come from ``_flips``,
    which ``_cases`` adds to every case."""
    import dataclasses

    return dataclasses.replace(case, known=dict(known))


ALL_GRIDS = tuple(g.label for g in GRIDS)
ON_CROSSING = ("u31", "aligned2", "aligned31", "offset31")
ALIGNED = ("aligned2", "aligned31")
