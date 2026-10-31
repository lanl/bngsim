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

Recognized, after the run's parameter values are put in:

- ``exp(q)`` with ``q`` a quadratic in one clock with a negative leading
  coefficient (a Gaussian, at any scale and offset);
- ``cosh(u)``, ``sech(u)`` and ``tanh(u)`` with ``u`` affine in one clock
  (sech², spelled ``1/cosh(u)^2`` or ``1 - tanh(u)^2``, whose tanh saturates
  to exactly 1; a tanh step is one stop's worth of restarts more).

A clock is literal ``time()``, or an observable whose species all change at a
constant rate (a counter fed at ``k_clock``, at any rate). Its value at ``t`` is
read from the run's start and that rate.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any

# A rate law with none of these spells no bump, and is not parsed.
_BUMP_CALL = re.compile(r"(?<![A-Za-z0-9_.])(?:exp|cosh|sech|tanh)\s*\(")
# Where the stops go, in widths from the centre: the peak.
_STOP_WIDTHS = (0.0,)
_TIME_SYMBOL = "_bngsim_time_csymbol"


@functools.lru_cache(maxsize=4096)
def _parsed(text: str):
    from bngsim._jacobian import _exprtk_to_sympy

    return _exprtk_to_sympy(text)


def _rate_law_texts(core) -> list[str]:
    """Each functional rate law with functions inlined, as the switch scanners
    read it."""
    from bngsim._jacobian import _inline_functions
    from bngsim._switch_sensitivity import _scan_function_map, _spell_mod

    data = core.codegen_data()
    func_map = _scan_function_map(core.functional_jacobian_context())
    out: list[str] = []
    for rxn in data["reactions"]:
        if rxn.get("type") != "functional":
            continue
        fname = rxn.get("function_name")
        body = func_map.get(fname) if fname else _spell_mod(rxn.get("rate_expression") or "")
        if body:
            out.append(_inline_functions(body, func_map) or body)
    return out


def _counter_clocks(core, t_start: float) -> dict[str, tuple[float, float]]:
    """Observable name -> ``(value at t_start, rate)`` for each observable all
    of whose species change at one constant rate, read from two probes of the
    right-hand side at shifted states and times."""
    import numpy as np

    data = core.codegen_data()
    try:
        conc = np.asarray(core.get_state(), dtype=float)
        d1 = np.asarray(core._eval_rhs(float(t_start), conc))
        d2 = np.asarray(core._eval_rhs(float(t_start) + 1.0, conc + 1.0))
    except Exception:  # pragma: no cover - defensive
        return {}
    constant = {i for i in range(len(d1)) if d1[i] == d2[i] and np.isfinite(d1[i])}
    out: dict[str, tuple[float, float]] = {}
    for obs in data["observables"]:
        entries = obs.get("entries") or []
        if not entries or not all(si in constant for si, _w in entries):
            continue
        rate = sum(w * float(d1[si]) for si, w in entries)
        if rate == 0.0 or not math.isfinite(rate):
            continue
        out[obs["name"]] = (sum(w * float(conc[si]) for si, w in entries), rate)
    return out


def _value_calls(expr, sp) -> list[Any]:
    """The ``exp``/``cosh``/``sech``/``tanh`` calls of *expr* that are part of
    its value, not of a condition. A bump inside a condition (``if(exp(...) >
    0.5, ...)``, a threshold written with ``tanh``) is a crossing, which the
    condition scan places or declines (#904, #1068); a stop here would change
    what that scan decided."""
    calls: list[Any] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, (sp.exp, sp.cosh, sp.sech, sp.tanh)):
            calls.append(node)
        if isinstance(node, sp.Piecewise):
            stack.extend(value for value, _cond in node.args)
        elif isinstance(node, (sp.core.relational.Relational, sp.logic.boolalg.Boolean)):
            continue
        else:
            stack.extend(node.args)
    return calls


def _bumps(expr, clocks: set[str], sp) -> list[tuple[str, float, float]]:
    """``(clock, centre, width)`` in the clock's units, for each bump of *expr*
    in one clock."""
    found: list[tuple[str, float, float]] = []
    for node in _value_calls(expr, sp):
        arg = node.args[0]
        names = {str(s) for s in arg.free_symbols}
        if len(names) != 1 or not names <= clocks:
            continue
        (clock,) = names
        sym = sp.Symbol(clock)
        try:
            poly = sp.Poly(sp.expand(arg), sym)
        except sp.PolynomialError:
            continue
        coeffs = [float(c) for c in poly.all_coeffs()]
        if isinstance(node, sp.exp):
            if poly.degree() != 2 or not coeffs[0] < 0.0:
                continue
            a, b = coeffs[0], coeffs[1]
            found.append((clock, -b / (2.0 * a), 1.0 / math.sqrt(-a)))
        else:
            if poly.degree() != 1 or coeffs[0] == 0.0:
                continue
            a, b = coeffs
            found.append((clock, -b / a, 1.0 / abs(a)))
    return [f for f in found if all(math.isfinite(x) for x in f[1:]) and f[2] > 0.0]


# The stops of a model at its values, keyed by everything they read: a run, a
# scan point or a fit evaluation at the same values asks again for nothing.
_STOPS_CACHE: dict[tuple, list[Any]] = {}
_STOPS_CACHE_MAX = 1024


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
    hit = _STOPS_CACHE.get(key)
    if hit is not None:
        return list(hit)
    stops = _resolve(core, float(t_start), float(t_end))
    if len(_STOPS_CACHE) >= _STOPS_CACHE_MAX:
        _STOPS_CACHE.clear()
    _STOPS_CACHE[key] = stops
    return list(stops)


def _resolve(core, t_start: float, t_end: float) -> list[Any]:
    texts = [t for t in _rate_law_texts(core) if _BUMP_CALL.search(t)]
    if not texts:
        return []
    try:
        import sympy as sp
    except ImportError:  # pragma: no cover
        return []
    from bngsim._codegen import _inline_derived_param_refs
    from bngsim._switch_sensitivity import CrossingStop, switch_condition_scope

    derived = dict(switch_condition_scope(core).derived_exprs)
    values = {n: float(core.get_param(n)) for n in core.param_names}
    counters = _counter_clocks(core, t_start)
    clocks = {_TIME_SYMBOL, *counters}
    times: set[float] = set()
    for text in texts:
        expr = _parsed(_inline_derived_param_refs(text, derived))
        if expr is None:
            continue
        subs = {s: values[str(s)] for s in expr.free_symbols if str(s) in values}
        expr = expr.subs(subs) if subs else expr
        for clock, centre, width in _bumps(expr, clocks, sp):
            for k in _STOP_WIDTHS:
                value = centre + k * width
                if clock == _TIME_SYMBOL:
                    when = value
                else:
                    v0, rate = counters[clock]
                    when = t_start + (value - v0) / rate
                if math.isfinite(when) and t_start < when <= t_end:
                    times.add(when)
    return [CrossingStop(t, -1, 0.0) for t in sorted(times)]
