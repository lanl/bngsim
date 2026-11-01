"""Stops on a smooth bump of the clock in a rate law (issue #1286).

A rate law shaped as a narrow smooth bump of a clock, a Gaussian
``exp(-((T - c)/s)^2)`` or a ``1/cosh((T - c)/s)^2``, underflows to exactly 0
away from it: ``exp(-10^4)`` is 0 in doubles. From the start of the run to the
bump, the right-hand side is then exactly what it would be with no bump, CVODE's
error estimate is 0, and its step grows until one step spans the bump. Y came
back 0 for 2.212 at every tolerance, on dense output points and in sensitivity
runs alike, with no word. A corner or a condition gets a stop (#714, #1068,
#1127); a smooth bump has neither, so nothing marked where it was.

This finds such bumps in the text and places a stop at each one's centre. A
step that would have spanned the bump ends on its peak instead, where the
rate is largest, so the error test sees the bump and shrinks the step onto it,
and the run restarts there with a small step for the fall. Stops four widths
either side as well were measured no more accurate, on a Gaussian of widths
0.005 to 10, two bumps in one run, and a run that starts on a bump's rise, at
rtol 1e-6 and 1e-12 (within the tolerance of a max_step reference throughout),
and were two restarts more. A stop costs a restart and changes nothing else,
so a bump found where none matters (wide, or never reached) costs only that.

A bump centred past the end of the run with its rise inside the run gets a stop
at the end instead. The step that reaches the last output is taken past it and
read back, and taken over the rise and the peak it read 0 at both ends: with the
centre a width past t = 30, Y(30) was 0 for 0.386.

Recognized, after the run's parameter values are put in:

- ``exp(q)`` with ``q`` a quadratic in one clock with a negative leading
  coefficient (a Gaussian, at any scale and offset), also spelled ``_e^q`` or
  ``pow(_e, q)``;
- ``cosh(u)`` and ``tanh(u)`` with ``u`` affine in one clock (sech², spelled
  ``1/cosh(u)^2`` or ``1 - tanh(u)^2``, whose tanh saturates to exactly 1; a
  tanh step is one stop's worth of restarts more). ExprTk has no ``sech``.

A clock is literal ``time()``, or an observable whose species all change at a
constant rate on two probes of the right-hand side (a counter fed at
``k_clock``, at any rate; not a feed that reads the state). Its value at ``t``
is read from the run's start and that rate, which holds until an event assigns
the counter: a counter an event resets has its bump stopped on where it first
reaches it, and a warning says the passes after a reset are not.
"""

from __future__ import annotations

import contextlib
import functools
import math
import re
import warnings
import weakref
from typing import Any

# A rate law with none of these spells no bump, and is not parsed. `_e^q` and
# `pow(_e, q)` are parsed as exp(q).
_BUMP_CALL = re.compile(r"(?<![A-Za-z0-9_.])(?:(?:exp|cosh|tanh)\s*\(|_e\s*\^|pow\s*\(\s*_e\s*,)")
# The clock as the parse reads it: `time()` and no other spelling (`t()` is a
# name in the model, #659).
_TIME_CALL = re.compile(r"(?<![A-Za-z0-9_.])time\s*\(\s*\)")
_NAME = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*")
# How far from its centre, in widths, a bump is still more than a rounding of
# its peak: exp(-6^2) is 2.3e-16 of it, and sech² and 1 - tanh² fall as
# 4·e^(-2u), 1.2e-16 at u = 19. A bump centred past the end of the run by less
# than this has its rise inside the run.
_REACH_EXP = 6.0
_REACH_HYPERBOLIC = 19.0


@functools.lru_cache(maxsize=4096)
def _parsed(text: str):
    from bngsim._jacobian import _exprtk_to_sympy

    return _exprtk_to_sympy(text)


def _counter_clocks(core, t_start: float, spelled) -> dict[str, tuple[float, float, bool]]:
    """Observable name, as *spelled* in a parse -> ``(value at t_start, rate,
    whether an event assigns one of its species)`` for each observable all of
    whose species change at one constant rate, read from two probes of the
    right-hand side at shifted states and times."""
    import numpy as np

    from bngsim._codegen import _amount_volume_factors

    data = core.codegen_data()
    try:
        conc = np.asarray(core.get_state(), dtype=float)
        d1 = np.asarray(core._eval_rhs(float(t_start), conc))
        d2 = np.asarray(core._eval_rhs(float(t_start) + 1.0, conc + 1.0))
    except Exception:  # pragma: no cover - defensive
        return {}
    constant = {i for i in range(len(d1)) if d1[i] == d2[i] and np.isfinite(d1[i])}
    # An observable reads an amount-valued (SBML substance-units) species as its
    # stored value times its compartment's size (GH #75). A counter so declared
    # in a compartment of size 2 moves at twice its stored rate, and read at the
    # stored one its bump's stop came at twice the bump's time.
    factor, live = _amount_volume_factors(data["species"])
    names = list(core.param_names)
    factor.update({i: float(core.get_param(names[k])) for i, k in live.items()})
    assigned = set(core.event_assigned_species)
    out: dict[str, tuple[float, float, bool]] = {}
    for obs in data["observables"]:
        entries = [(si, w * factor.get(si, 1.0)) for si, w in obs.get("entries") or []]
        if not entries or not all(si in constant for si, _w in entries):
            continue
        rate = sum(w * float(d1[si]) for si, w in entries)
        if rate == 0.0 or not math.isfinite(rate):
            continue
        out[spelled(obs["name"])] = (
            sum(w * float(conc[si]) for si, w in entries),
            rate,
            any(si in assigned for si, _w in entries),
        )
    return out


def _value_calls(expr, sp) -> list[Any]:
    """The ``exp``/``cosh``/``tanh`` calls of *expr* that are part of
    its value, not of a condition. A bump inside a condition (``if(exp(...) >
    0.5, ...)``, a threshold written with ``tanh``) is a crossing, which the
    condition scan places or declines (#904, #1068); a stop here would change
    what that scan decided."""
    calls: list[Any] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, (sp.exp, sp.cosh, sp.tanh)):
            calls.append(node)
        if isinstance(node, sp.Piecewise):
            stack.extend(value for value, _cond in node.args)
        elif isinstance(node, (sp.core.relational.Relational, sp.logic.boolalg.Boolean)):
            continue
        else:
            stack.extend(node.args)
    return calls


def _bumps(expr, sp) -> list[tuple[str, float, float, float]]:
    """``(symbol, centre, width, reach)`` in the symbol's units, for each bump
    of *expr* in one symbol. The reach is how far from the centre the bump is
    more than a rounding of its peak."""
    found: list[tuple[str, float, float, float]] = []
    for node in _value_calls(expr, sp):
        arg = node.args[0]
        if len(arg.free_symbols) != 1:
            continue
        (sym,) = arg.free_symbols
        try:
            poly = sp.Poly(sp.expand(arg), sym)
            coeffs = [float(c) for c in poly.all_coeffs()]
        except (sp.polys.polyerrors.BasePolynomialError, TypeError, ValueError):
            # Not a polynomial in it, or one with a coefficient that is no real
            # number: a width of 0 at a scan point puts zoo in the argument, and
            # raised out of the whole model's stops, every run.
            continue
        if isinstance(node, sp.exp):
            if poly.degree() != 2 or not coeffs[0] < 0.0:
                continue
            a, b = coeffs[0], coeffs[1]
            centre, width, reach = -b / (2.0 * a), 1.0 / math.sqrt(-a), _REACH_EXP
        else:
            if poly.degree() != 1 or coeffs[0] == 0.0:
                continue
            a, b = coeffs
            centre, width, reach = -b / a, 1.0 / abs(a), _REACH_HYPERBOLIC
        if math.isfinite(centre) and math.isfinite(width) and width > 0.0:
            found.append((sym.name, centre, width, reach * width))
    return found


# The stops each model was asked for, by everything else they read: a run, a
# scan point or a fit evaluation at the same values asks again for nothing. Kept
# per model, whose rate laws and reactions are its core's for its life: two
# models with the same functions, values and state are not one model. Keyed on
# those alone, a model whose counter is fed at 1 was handed the stop of one fed
# at 2, at half its bump's time, and stepped over the bump.
_STOPS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_STOPS_MOST = 64


def smooth_pulse_stops(core, t_start: float, t_end: float) -> list[Any]:
    """The stops in ``(t_start, t_end]`` for the smooth bumps of the clock in
    the rate laws, as crossing stops on literal time."""
    # The raw texts first: a model none of whose functions spells one of these
    # calls has no bump, and is answered without the inlined context, which
    # runs to tens of thousands of entries on a large model.
    raw = tuple(core.function_expressions)
    if not any(_BUMP_CALL.search(t) for t in raw):
        return []
    key = (
        raw,
        tuple(float(core.get_param(n)) for n in core.param_names),
        tuple(float(x) for x in core.get_state()),
        float(t_start),
        float(t_end),
    )
    asked: dict[tuple, list[Any]] = {}
    with contextlib.suppress(TypeError):  # a core that takes no weak reference
        asked = _STOPS.setdefault(core, asked)
    stops = asked.get(key)
    if stops is None:
        stops = _resolve(core, float(t_start), float(t_end))
        if len(asked) >= _STOPS_MOST:
            asked.clear()
        asked[key] = stops
    return list(stops)


def _call_arguments(text: str):
    """The text of each bump call's argument in *text*: what a balanced
    ``exp(``, ``cosh(`` or ``tanh(`` holds, the exponent of ``pow(_e, q)``, and
    all that follows ``_e^``."""
    for m in _BUMP_CALL.finditer(text):
        if m.group(0).startswith("_e"):
            yield text[m.end() :]
            continue
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        yield text[m.end() : i]


def _a_clock_in_a_bump(core, t_start: float, raw: tuple[str, ...]) -> bool:
    """Whether a bump call's argument can be a function of a clock alone, read
    from the function texts before anything is inlined or parsed.

    A bump's argument has the clock as its one symbol once the parameter values
    are put in. One that reaches no clock, through the functions and derived
    parameters it names, is no bump; nor is one that reaches a species or an
    observable that is no clock, which stays a second symbol. Most `exp` calls
    in rate laws are one or the other (a Boltzmann factor, a rate of a
    voltage), and inlining and parsing every rate law that holds one took over
    7 minutes a run on MODEL1006230090, whose 22 are all of them."""
    clocks = set(_counter_clocks(core, t_start, lambda name: name))
    state = (set(core.observable_names) | set(core.species_names)) - clocks
    defs = dict(zip(core.function_names, raw, strict=True))
    for name, is_expr, expr in zip(
        core.param_names, core.param_is_expression, core.param_expressions, strict=True
    ):
        if is_expr and expr:
            defs.setdefault(name, expr)

    def reads(text: str) -> tuple[bool, bool, set[str]]:
        names = set(_NAME.findall(text))
        clock = bool(_TIME_CALL.search(text) or names & clocks)
        return clock, bool(names & state), names & defs.keys()

    direct = {name: reads(text) for name, text in defs.items()}
    reach = {name: [clock, other] for name, (clock, other, _named) in direct.items()}
    grew = True
    while grew:
        grew = False
        for name, (_clock, _other, named) in direct.items():
            for k in (0, 1):
                if not reach[name][k] and any(reach[n][k] for n in named):
                    reach[name][k] = grew = True
    for text in raw:
        for arg in _call_arguments(text):
            clock, other, named = reads(arg)
            if (clock or any(reach[n][0] for n in named)) and not (
                other or any(reach[n][1] for n in named)
            ):
                return True
    return False


def _resolve(core, t_start: float, t_end: float) -> list[Any]:
    from bngsim._switch_sensitivity import _rate_law_flat_parts, _write_flats

    if not _a_clock_in_a_bump(core, t_start, tuple(core.function_expressions)):
        return []
    made = _rate_law_flat_parts(core)
    texts = [t for t in _write_flats(core, made)[: made[1]] if _BUMP_CALL.search(t)]
    if not texts:
        return []
    try:
        import sympy as sp
    except ImportError:  # pragma: no cover
        return []
    from bngsim._codegen import (
        _PY_KEYWORD_PARAM_NAMES,
        _alias_keyword_param,
        _inline_derived_param_refs,
    )
    from bngsim._jacobian import _TIME_SYM
    from bngsim._switch_sensitivity import CrossingStop

    def spelled(name: str) -> str:
        """*name* as the parse spells it: a Python keyword is aliased (GH #108),
        and a width named `lambda` was left in the argument unread."""
        return _alias_keyword_param(name) if name in _PY_KEYWORD_PARAM_NAMES else name

    names = list(core.param_names)
    derived = {
        n: e
        for n, is_expr, e in zip(
            names, core.param_is_expression, core.param_expressions, strict=True
        )
        if is_expr and e
    }
    values = {spelled(n): float(core.get_param(n)) for n in names}
    bumps: list[tuple[str, float, float, float]] = []
    for text in texts:
        expr = _parsed(_inline_derived_param_refs(text, derived))
        if expr is None:
            continue
        subs = {s: values[s.name] for s in expr.free_symbols if s.name in values}
        bumps.extend(_bumps(expr.subs(subs) if subs else expr, sp))
    counters = (
        _counter_clocks(core, t_start, spelled)
        if any(bump[0] != _TIME_SYM for bump in bumps)
        else {}
    )
    times: set[float] = set()
    for sym, centre, width, reach in bumps:
        if sym == _TIME_SYM:
            when, span = centre, reach
        elif sym in counters:
            v0, rate, reset = counters[sym]
            when, span = t_start + (centre - v0) / rate, reach / abs(rate)
            if reset:
                warnings.warn(
                    f"A smooth bump of {sym} at {sym} = {centre:g} is stopped on "
                    f"where {sym} first reaches it, from its value at the start of "
                    f"the run and its rate; an event assigns {sym}, and a pass "
                    "through the bump after that is not stopped on and may be "
                    "stepped over (issue #1286). Pass max_step no larger than the "
                    f"bump's width in time, {width / abs(rate):g}.",
                    stacklevel=2,
                )
        else:
            continue
        if not math.isfinite(when):
            continue
        if t_start < when <= t_end:
            times.add(when)
        elif t_end < when < t_end + span:
            times.add(t_end)
    return [CrossingStop(t, -1, 0.0) for t in sorted(times)]


def apart_from(pulses: list[Any], stops: list[Any], floor: float) -> list[Any]:
    """Those of *pulses* not on the instant of one of *stops* (time-ordered).

    A bump needs a step to end on its instant, not on any particular double,
    and a stop already there is one. Merged as the later of two on literal time,
    a bump centred on a switch took the switch's stop off the double before its
    flip (#1226): Y(20) was 10.0005 for 10 at rtol 1e-2."""
    import bisect

    from bngsim._switch_sensitivity import _same_instant

    times = [stop.time for stop in stops]

    def on_one(t: float) -> bool:
        i = bisect.bisect_left(times, t)
        return any(_same_instant(t, times[j], floor) for j in (i - 1, i) if 0 <= j < len(times))

    return [pulse for pulse in pulses if not on_one(pulse.time)]
