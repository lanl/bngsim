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

Whether the values sit on a bend is asked exactly, in double precision, as the
engine evaluates them: a bend an ulp away is the one-sided derivative of the
function the engine computes, and is left alone. Whether the law has a
derivative there is asked of the whole law, read with each side of the bend put
in for it, and not of the bend alone: ``10*abs(q)*q`` at q = 0 is smooth, and so
is a law whose ``abs`` sits in a branch the present values do not take.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from typing import Any

# sympy's names for the calls a description prints, back as a law writes them.
_AS_WRITTEN = {"Abs": "abs", "Min": "min", "Max": "max", "clampval": "clamp"}
_SYMPY_CALL = re.compile(r"\b(Abs|Min|Max|clampval)\(")


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


def _bend_points(expr, sp) -> list[tuple[str, Any, Any, list[Any]]]:
    """``(construct, at, node, sides)`` for each bend of *expr*: ``at`` is the
    quantity that is 0 on it, and ``sides`` what *node* is on either side of it,
    to be put in for *node* to read the law there: ``±u`` for ``abs(u)``, the
    two values of ``clamp`` at an edge, and true and false for a comparison of
    an ``if()`` condition. A ``min`` / ``max`` has no ``at``: its ``sides`` are
    its arguments, of which the tied ones are the caller's to find."""
    from bngsim._jacobian import _kink_bindings

    clampval = _kink_bindings(sp)["clampval"]
    found: list[tuple[str, Any, Any, list[Any]]] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, sp.Abs):
            u = node.args[0]
            found.append((f"abs({u})", u, node, [u, -u]))
        elif isinstance(node, (sp.Max, sp.Min)):
            found.append(
                (f"{type(node).__name__.lower()}{node.args}", None, node, list(node.args))
            )
        elif type(node) is clampval:
            lo, x, hi = node.args
            found.append((f"clamp({lo}, {x}, {hi})", x - lo, node, [lo, x]))
            found.append((f"clamp({lo}, {x}, {hi})", x - hi, node, [x, hi]))
        elif isinstance(node, sp.Piecewise):
            for value, cond in node.args:
                for rel in cond.atoms(sp.core.relational.Relational):
                    found.append((f"if({rel}, ...)", rel.lhs - rel.rhs, rel, [sp.true, sp.false]))
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


def _clamp(lo, x, hi):
    """ExprTk's ``e_clamp``, for the parse's ``clampval``: ``lo`` where x < lo,
    else ``hi`` where x > hi, else ``x``."""
    return lo if x < lo else (hi if x > hi else x)


def _values(exprs, subs, sp) -> list[float]:
    """*exprs* at the parameter values *subs*, in double precision. Each Float is
    printed as the double it holds: sympy's own printer writes 15 digits, and
    ``q/3.0 - 1`` at q = 3, which the engine reads 0, read -1e-15."""
    from bngsim._switch_sensitivity import _double_printer

    # The settings lambdify gives a printer it makes itself, which one handed to
    # it does not get: `exp`, not `math.exp`, which its namespace has no `math`
    # for, and an unknown call by name, for the namespace to find `clampval`.
    printer = _double_printer()(
        {"fully_qualified_modules": False, "inline": True, "allow_unknown_functions": True}
    )
    f = sp.lambdify(
        list(subs), list(exprs), modules=[{"clampval": _clamp}, "math"], printer=printer
    )
    return [float(v) for v in f(*subs.values())]


def parameters_on_a_kink(core, param_names: Sequence[str]) -> list[str]:
    """Each of *param_names* that a rate law bends at, at the model's present
    values: one line per (parameter, bend), for the refusal to list."""
    names = list(dict.fromkeys(param_names))
    if not names:
        return []
    from bngsim._jacobian import has_condition_construct
    from bngsim._switch_sensitivity import _BEND_CALL

    # A rate law with no bend and no condition has nothing to sit on, and is not parsed.
    laws = [
        (w, t)
        for w, t in _rate_law_texts(core)
        if _BEND_CALL.search(t) or has_condition_construct(t)
    ]
    if not laws:
        return []
    try:
        import sympy as sp
    except ImportError:
        return []
    from bngsim._codegen import (
        _PY_KEYWORD_PARAM_NAMES,
        _alias_keyword_param,
        _inline_derived_param_refs,
    )

    all_names = list(core.param_names)
    values = {n: float(core.get_param(n)) for n in all_names}
    # The parser spells a parameter named after a Python keyword (`lambda`) by
    # an alias, and the laws' symbols are read back through it.
    spelled = {
        (_alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n): n for n in all_names
    }
    # The derived parameters as switch_condition_scope reads them, without its
    # probe of the right-hand side for the clocks.
    derived = {
        n: e
        for n, is_e, e in zip(
            all_names, core.param_is_expression, core.param_expressions, strict=True
        )
        if is_e and e
    }
    asked = [p for p in names if p in values]
    # Every derived parameter but the one asked about is inlined, so a bend
    # written over a derived parameter is attributed to what it is built from,
    # and the one asked about keeps its name (issue #475): one reading of the
    # laws for all the primaries, and one more per derived parameter asked about.
    groups = [({p for p in asked if p not in derived}, derived)]
    groups += [({p}, {k: v for k, v in derived.items() if k != p}) for p in asked if p in derived]
    out: list[str] = []
    for group, exprs in groups:
        if not group:
            continue
        for where, text in laws:
            law = _parsed(_inline_derived_param_refs(text, exprs) or text)
            if law is None:
                continue
            for p, bend in _kinks(law, group, spelled, values, sp):
                out.append(f"{p!r} in {where}: {bend}")
    return out


def _kinks(law, asked, spelled, values, sp) -> list[tuple[str, str]]:
    """``(parameter, description)`` for each bend of *law* that the present
    values sit on, that a parameter of *asked* moves, and where the law's two
    sides differ in that parameter. A bend that reads anything but parameters
    (the state, the clock) is a crossing in time, and is not judged here."""
    params = {s: spelled[s.name] for s in law.free_symbols if s.name in spelled}
    if not any(p in asked for p in params.values()):
        return []
    out: list[tuple[str, str]] = []
    for construct, at, node, sides in _bend_points(law, sp):
        pool = sides if at is None else [at]
        free = set().union(*(e.free_symbols for e in pool))
        movers = sorted((s for s in free if params.get(s) in asked), key=str)
        if not movers or not free <= set(params):
            continue
        try:
            got = _values(pool, {s: values[params[s]] for s in free}, sp)
        except Exception:
            # A call this cannot evaluate (an engine function): not judged.
            continue
        if at is None:
            # min / max: a tie of the extreme argument.
            extreme = max(got) if construct.startswith("max") else min(got)
            on = [a for a, v in zip(sides, got, strict=True) if v == extreme]
            if len(on) < 2:
                continue
            bend = f"{construct}, whose arguments {on} tie at {extreme:g}"
        else:
            if got[0] != 0.0:
                continue
            on = sides
            bend = None
        for s in movers:
            p = params[s]
            if _sides_differ(law, node, on, s, params, values, sp):
                line = bend or f"{construct}, on its switch at {p} = {values[p]:g}"
                out.append((p, _as_written(line, params)))
    return out


def _as_written(text: str, params) -> str:
    """*text* with sympy's call names and the parser's keyword aliases
    (``_BNG_KW_lambda``) spelled as the law writes them."""
    text = _SYMPY_CALL.sub(lambda m: _AS_WRITTEN[m.group(1)] + "(", text)
    for s, p in params.items():
        if s.name != p:
            text = re.sub(rf"\b{re.escape(s.name)}\b", p, text)
    return text


def _sides_differ(law, node, sides, sym, params, values, sp) -> bool:
    """Whether *law*, read with each of *sides* put in for *node*, differs in
    slope in *sym* at the parameters' present values, or for a comparison of an
    ``if()``, in value. What else the law reads (the state) is left a symbol, so
    the sides agree only where they agree for every state. A difference this
    cannot finish counts as one: the values are on the bend.

    It is read over real symbols, because over the parser's plain ones ``d|u|``
    is ``re()`` / ``im()`` and a ``Derivative`` that nothing evaluates:
    ``abs(abs(q) - 1)`` at q = 1 raised from the printer rather than being
    refused. Only here, where the values are on a bend: made real, a law costs
    as much again as its parse."""
    jumps = isinstance(node, sp.core.relational.Relational)
    try:
        real = {s: sp.Symbol(s.name, real=True) for s in law.free_symbols}
        law, node, sym = law.xreplace(real), node.xreplace(real), real[sym]
        if not law.has(node):
            # Read over real symbols, the law took the bend into something else.
            return True
        subs = {real[s]: sp.Float(values[p]) for s, p in params.items()}
        first, *others = (law.xreplace({node: side.xreplace(real)}) for side in sides)
        for other in others:
            gaps = [sp.diff(first, sym) - sp.diff(other, sym)]
            if jumps:
                gaps.append(first - other)
            if any(sp.expand(gap.xreplace(subs)) != 0 for gap in gaps):
                return True
    except Exception:
        return True
    return False
