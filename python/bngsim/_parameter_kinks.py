"""A sensitivity parameter that sits on a kink of a rate law (issue #1332).

``abs``, ``min``, ``max``, ``clamp`` and ``if()`` bend or switch where their
argument crosses a value. A rate law whose bend reads parameters only, and whose
parameters put it exactly there, has no derivative in a parameter that moves
the bend: ``10*abs(q) + s`` at q = 0 has slopes -10 and +10 in q, and
``10*min(q, 1) + s`` at q = 1 has 10 and 0. The compiled ``∂f/∂p`` differentiates
each branch and takes ``sign(0) = 0`` and the average at a tie, so a forward
sensitivity there came back 0 for a slope of +-47.62 at the steady state, and
23.81 for 47.62 or 0, with no error. Such a column is refused.

Only a bend over parameters alone is judged here, where the parameters' values
decide it before the run. A bend over the state is a crossing in time, which the
sensitivity envelope already locates or refuses (issues #146, #414, #938, #960).

The test is exact, at the values in double precision, as the engine evaluates
them: a bend an ulp away is the one-sided derivative of the function the engine
computes, and is left alone.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from typing import Any

# A rate law with none of these has no bend to sit on, and is not parsed.
_BENDS = re.compile(r"(?<![A-Za-z0-9_.])(?:abs|min|max|clamp|if)\s*\(")


def _rate_law_texts(core) -> list[tuple[str, str]]:
    """``(where, text)`` of each functional rate law, with function references
    inlined, as the switch scanners read it."""
    from bngsim._jacobian import _inline_functions
    from bngsim._switch_sensitivity import _scan_function_map, _spell_mod

    data = core.codegen_data()
    func_map = _scan_function_map(core.functional_jacobian_context())
    out: list[tuple[str, str]] = []
    for i, rxn in enumerate(data["reactions"]):
        if rxn.get("type") != "functional":
            continue
        fname = rxn.get("function_name")
        body = func_map.get(fname) if fname else _spell_mod(rxn.get("rate_expression") or "")
        if not body:
            continue
        where = f"reaction {i + 1}" + (f" ({fname})" if fname else "")
        out.append((where, _inline_functions(body, func_map) or body))
    return out


def _bend_points(expr, sp) -> list[tuple[str, Any, list[Any]]]:
    """``(construct, at, branches)`` for each bend of *expr*: ``at`` is the
    quantity that is 0 on it, and ``branches`` the expressions whose
    derivatives must agree there (for ``abs``, ``±u``; for a tie of ``min`` /
    ``max``, the tied arguments, which the caller finds)."""
    from bngsim._jacobian import _kink_bindings

    clampval = _kink_bindings(sp)["clampval"]
    found: list[tuple[str, Any, list[Any]]] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, sp.Abs):
            u = node.args[0]
            found.append((f"abs({u})", u, [u, -u]))
        elif isinstance(node, (sp.Max, sp.Min)):
            found.append((f"{type(node).__name__.lower()}{node.args}", None, list(node.args)))
        elif type(node) is clampval:
            lo, x, hi = node.args
            found.append((f"clamp({lo}, {x}, {hi})", x - lo, [lo, x]))
            found.append((f"clamp({lo}, {x}, {hi})", x - hi, [x, hi]))
        elif isinstance(node, sp.Piecewise):
            for value, cond in node.args:
                for rel in cond.atoms(sp.core.relational.Relational):
                    found.append((f"if({rel}, ...)", rel.lhs - rel.rhs, []))
                stack.append(value)
            continue
        stack.extend(node.args)
    return found


@functools.lru_cache(maxsize=4096)
def _parsed(text: str):
    """The sympy parse of a rate law's text, kept: a batch asks each row's
    clone, whose laws read the same."""
    from bngsim._jacobian import _exprtk_to_sympy

    return _exprtk_to_sympy(text, bends=True)


def parameters_on_a_kink(core, param_names: Sequence[str]) -> list[str]:
    """Each of *param_names* that a rate law bends at, at the model's present
    values: one line per (parameter, bend), for the refusal to list."""
    names = list(dict.fromkeys(param_names))
    if not names:
        return []
    laws = [(w, t) for w, t in _rate_law_texts(core) if _BENDS.search(t)]
    if not laws:
        return []
    try:
        import sympy as sp
    except ImportError:
        return []
    from bngsim._codegen import _inline_derived_param_refs
    from bngsim._switch_sensitivity import switch_condition_scope

    derived = dict(switch_condition_scope(core).derived_exprs)
    values = {n: float(core.get_param(n)) for n in core.param_names}
    out: list[str] = []
    for p in names:
        if p not in values:
            continue
        # Every derived parameter but the one asked about is inlined, so a bend
        # written over a derived parameter is attributed to what it is built
        # from, and the one asked about keeps its name (issue #475).
        exprs = {k: v for k, v in derived.items() if k != p}
        sym = sp.Symbol(p)
        for where, text in laws:
            expr = _parsed(_inline_derived_param_refs(text, exprs))
            if expr is None or sym not in expr.free_symbols:
                continue
            for construct, at, branches in _bend_points(expr, sp):
                line = _on_the_bend(construct, at, branches, sym, values, sp)
                if line is not None:
                    out.append(f"{p!r} in {where}: {line}")
    return out


def _on_the_bend(construct, at, branches, sym, values, sp) -> str | None:
    """A description of the bend if the parameters sit on it and *sym* moves it
    with different slopes on its sides; ``None`` otherwise, and for a bend that
    reads anything but parameters (the state, the clock)."""
    pool = branches + ([at] if at is not None else [])
    free = set().union(*(e.free_symbols for e in pool)) if pool else set()
    if not free or not free <= {sp.Symbol(n) for n in values} or sym not in free:
        return None
    subs = {s: values[str(s)] for s in free}

    def val(e) -> float:
        return float(sp.lambdify(list(subs), e, "math")(*subs.values()))

    def slope(e) -> float:
        return val(sp.diff(e, sym))

    try:
        if at is None:
            # min / max: a tie of the extreme argument, between arguments whose
            # slopes in the parameter differ.
            vals = [val(a) for a in branches]
            extreme = max(vals) if construct.startswith("max") else min(vals)
            tied = [a for a, v in zip(branches, vals, strict=True) if v == extreme]
            if len(tied) < 2 or len({slope(a) for a in tied}) < 2:
                return None
            return f"{construct}, whose arguments {tied} tie at {extreme:g}"
        if val(at) != 0.0:
            return None
        if branches:
            if len({slope(b) for b in branches}) < 2:
                return None
        elif slope(at) == 0.0:
            return None
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return f"{construct}, on its switch at {sym} = {values[str(sym)]:g}"
