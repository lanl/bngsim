"""Symbolic differentiation of rate-law expressions for the analytical Jacobian
(GH #76).

This module is the **consumer-agnostic symbolic core**. It takes a rate-law
expression written in the engine's ExprTk grammar, differentiates it with
sympy, and returns the partial derivatives. Two facts shape the design:

1. **The interpreted engine syncs live state through *observables*, not species.**
   During CVODE integration only ``obs.total`` is refreshed from the live state
   vector; species ExprTk variables are stale. So an interpreted-engine
   derivative expression must be written in ``{observable, parameter, time}``
   symbols. The canonical primitive this module produces is therefore
   ``∂(rate)/∂(observable_k)`` — every consumer derives from it:

   * interpreted SBML (per-species): ``∂rate/∂x_j = Σ_k ∂rate/∂obs_k · factor_{k→j}``
     keeping observable symbols (live), emitted once per dependent species.
   * interpreted ``.net`` (per-observable): the ``∂rate/∂obs_k`` strings as-is,
     scattered by the C++ callback (which also handles the mass-action species
     factor).
   * **codegen (future):** the *same* ``∂rate/∂obs_k`` sympy expressions, emitted
     as C instead of ExprTk — a separate emitter, no re-derivation. The codegen
     RHS reads ``y[i]`` live, so it can reference the C-generated observable
     intermediates and scatter identically.

2. **Differentiation is separated from emission.** ``differentiate_rate_law``
   returns sympy expressions; ``sympy_to_exprtk`` is the (replaceable) ExprTk
   emitter. A C emitter is the only thing codegen adds.

Robustness contract: every entry point returns ``None`` (never raises) when an
expression cannot be parsed, classified, differentiated, or emitted. The caller
treats ``None`` as "no analytical Jacobian for this reaction" and the model
falls back to the finite-difference Jacobian — exactly the pre-#76 behavior.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable

# Reuse the BNGL/ExprTk ``if(c,t,f)`` → sympy ``Piecewise`` rewriter and the
# balanced-paren / top-level-comma helpers from the codegen path (issue #27).
from bngsim._codegen import (
    _PY_KEYWORD_PARAM_NAMES,
    _alias_keyword_param,
    _translate_bngl_if_to_piecewise,
)

logger = logging.getLogger("bngsim")

# Placeholder symbol standing in for ExprTk ``time()`` / ``t()`` while the
# expression lives in sympy. It is a constant w.r.t. species, and the ExprTk
# emitter maps it back to ``time()``.
_TIME_SYM = "_bngsim_time_csymbol"

# Python-keyword names that are *literals*, not identifiers — they must never be
# aliased to a parameter symbol (e.g. the ``True`` default in a Piecewise
# condition). Subset of ``_PY_KEYWORD_PARAM_NAMES`` left for sympy to interpret.
_LITERAL_KEYWORDS = frozenset({"True", "False", "None"})

# ExprTk built-in / reserved function names that may appear inside a rate law.
# Identifiers followed by ``(`` that are in this set parse as sympy functions
# rather than free symbols. ``log`` is the natural logarithm in both ExprTk and
# sympy (the SBML loader maps MathML ``ln`` → ExprTk ``log``).
_EXPRTK_TO_SYMPY_FUNC = {
    "exp": "exp",
    "log": "log",  # natural log
    "ln": "log",
    "log10": None,  # handled specially: log10(x) → log(x)/log(10)
    "log2": None,
    "sqrt": "sqrt",
    "abs": "Abs",
    "sign": "sign",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "min": "Min",
    "max": "Max",
    "floor": "floor",
    "ceil": "ceiling",
}

# sympy function names that the ExprTk emitter can represent. A derivative
# containing any other function (Heaviside, DiracDelta, erf, gamma, …) is not
# representable and triggers the FD fallback.
_SYMPY_FUNC_TO_EXPRTK = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",  # also produced from Pow(_, 1/2); see _ExprTkPrinter
    "Abs": "abs",
    "sign": "sign",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "Min": "min",
    "Max": "max",
    "floor": "floor",
    "ceiling": "ceil",
}

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_IDENT_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


# ─── ExprTk string → sympy ────────────────────────────────────────────────


def _preprocess_exprtk(expr: str) -> str:
    """Rewrite an ExprTk expression string into a form ``sympy.parse_expr`` can
    tokenize: ``time()``→placeholder, ``if(c,t,f)``→Piecewise, ``^``→``**``,
    word logicals → symbolic logicals."""
    s = expr.strip()
    # time() / t() → constant placeholder (whole-word, parens required so a
    # parameter literally named ``t`` is untouched).
    s = re.sub(r"\b(?:time|t)\s*\(\s*\)", _TIME_SYM, s)
    # if(c, t, f) → Piecewise((t, c), (f, True))
    s = _translate_bngl_if_to_piecewise(s)
    # power operator
    s = s.replace("^", "**")
    # word-form logicals → symbolic so parse_expr does not try to evaluate the
    # truth value of a symbolic operand. ExprTk parenthesizes each operand
    # (see _ast_to_exprtk_recursive), so operator precedence is preserved. ``not(x)``
    # becomes ``Not(x)`` (bound in the local dict).
    s = re.sub(r"\bnot\s*\(", "Not(", s)
    s = re.sub(r"\band\b", "&", s)
    s = re.sub(r"\bor\b", "|", s)
    return s


def _build_local_dict(preprocessed: str, sp):
    """Bind every identifier in ``preprocessed`` to a plain sympy Symbol (so
    names like ``E``, ``S``, ``I``, ``N`` are not mistaken for sympy constants
    or functions), and bind recognized call identifiers to sympy functions.

    Returns ``(local_dict, alias_of)`` where ``alias_of`` maps an original
    identifier to the (possibly keyword-aliased) symbol name used in sympy, or
    ``None`` if a ``log10``/``log2`` rewrite is needed first.
    """
    called = {m.group(1) for m in _IDENT_CALL_RE.finditer(preprocessed)}
    all_idents = set(_IDENT_RE.findall(preprocessed))

    local: dict = {"Piecewise": sp.Piecewise, "Not": sp.Not, "And": sp.And, "Or": sp.Or}
    alias_of: dict[str, str] = {}

    for ident in all_idents:
        if ident in ("Piecewise", "Not", "And", "Or") or ident in _LITERAL_KEYWORDS:
            # Bound/handled by sympy as literals; never a parameter symbol.
            continue
        if ident in called and ident in _EXPRTK_TO_SYMPY_FUNC:
            mapped = _EXPRTK_TO_SYMPY_FUNC[ident]
            if mapped is None:
                # log10 / log2 are rewritten in _exprtk_to_sympy before this.
                continue
            local[ident] = getattr(sp, mapped)
            continue
        if ident in called:
            # An un-inlined user-function call — caller must inline first.
            # Bind nothing; parse will fail or leave a free function, caught
            # by the free-symbol check.
            continue
        # A variable identifier. Keyword-named params (e.g. ``lambda``) get a
        # safe alias so parse_expr can tokenize them.
        alias = _alias_keyword_param(ident) if ident in _PY_KEYWORD_PARAM_NAMES else ident
        alias_of[ident] = alias
        local[alias] = sp.Symbol(alias)

    return local, alias_of


def _exprtk_to_sympy(expr: str):
    """Parse an ExprTk rate-law string into a sympy expression, or ``None``."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return None

    s = _preprocess_exprtk(expr)
    # log10(x) → log(x)/log(10); log2(x) → log(x)/log(2). Done on the string so
    # the function-call form is gone before parsing.
    s = re.sub(r"\blog10\s*\(", "(1.0/log(10))*log(", s)
    s = re.sub(r"\blog2\s*\(", "(1.0/log(2))*log(", s)

    local, alias_of = _build_local_dict(s, sp)

    # Apply keyword aliases (longest first so e.g. ``lambda`` is replaced before
    # a hypothetical ``lambda2`` substring would be — word-boundary anchored).
    for ident in sorted(alias_of, key=len, reverse=True):
        if alias_of[ident] != ident:
            s = re.sub(rf"\b{re.escape(ident)}\b", alias_of[ident], s)

    try:
        return parse_expr(s, local_dict=local, evaluate=True)
    except Exception:
        return None


# ─── Function inlining ─────────────────────────────────────────────────────


def _inline_functions(
    expr: str, func_map: dict[str, str], _depth: int = 0, _seen: frozenset | None = None
) -> str | None:
    """Recursively substitute ``name`` → ``(expression)`` for every user
    function ``name`` referenced in ``expr``. Assignment rules and SBML
    functionDefinition results are stored as functions, so a rate law that
    references a derived quantity is flattened until only observables,
    parameters and ``time()`` remain. Returns ``None`` on cycle / excessive
    depth (algebraic rules are acyclic per SBML, so this guards malformed
    input only)."""
    if _seen is None:
        _seen = frozenset()
    if _depth > 64:
        return None

    # Intersect the identifiers actually present in ``expr`` with ``func_map``,
    # rather than running a regex for every function name. ``func_map`` can hold
    # tens of thousands of entries on genome-scale mechanistic models, where the
    # per-name regex scan dominated build time; this is O(len(expr)).
    referenced = [n for n in set(_IDENT_RE.findall(expr)) if n in func_map]
    if not referenced:
        return expr

    out = expr
    for name in referenced:
        if name in _seen:
            return None  # cycle
        body = _inline_functions(func_map[name], func_map, _depth + 1, _seen | {name})
        if body is None:
            return None
        # Replace the bare identifier (the engine already resolved ``name()`` →
        # ``name`` for inter-function references at build time).
        out = re.sub(rf"\b{re.escape(name)}\b", f"({body})", out)
    return out


# ─── Core: differentiate w.r.t. observables ────────────────────────────────


def differentiate_rate_law(
    rate_expr: str,
    func_map: dict[str, str],
    observable_names: set[str],
    constant_names: set[str],
    deadline: float | None = None,
):
    """Differentiate a rate-law expression w.r.t. each observable it depends on.

    The **consumer-agnostic primitive**. Returns ``{observable_name:
    sympy_expr}`` for every observable whose partial derivative is non-zero, or
    ``None`` to signal the FD fallback. Returned expressions are in observable /
    parameter / time symbols (live in the interpreted evaluator) and are the
    shared input for the ExprTk emitter now and a C emitter later.

    Parameters
    ----------
    rate_expr : str
        The reaction's rate-law expression in ExprTk grammar.
    func_map : dict[str, str]
        ``function_name → expression`` for every user function (assignment
        rules, inlined functionDefinitions, nested rate-law helpers).
    observable_names : set[str]
        Names that are state-coupled (differentiation variables).
    constant_names : set[str]
        Names safe to treat as constants (plain / expression parameters,
        compartment volumes). Any free symbol that is neither an observable,
        a constant, nor ``time`` triggers the fallback — this is what rejects a
        hidden state (e.g. a rate-rule-target parameter) instead of silently
        differentiating it as a constant.
    deadline : float, optional
        Absolute ``time.perf_counter()`` value past which the differentiation is
        abandoned by raising :class:`_DerivationBudgetExceeded` (GH #95), checked
        before each per-observable ``sp.diff`` so one pathological rate law cannot
        run the build-time derivation unbounded. ``None`` (the default, used by
        the codegen emitter) means no budget.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    inlined = _inline_functions(rate_expr, func_map)
    if inlined is None:
        return None

    sym_expr = _exprtk_to_sympy(inlined)
    if sym_expr is None:
        return None

    # Build aliased name sets to match what _exprtk_to_sympy produced.
    def _alias(n: str) -> str:
        return _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n

    obs_alias = {_alias(o): o for o in observable_names}
    allowed = set(obs_alias) | {_alias(c) for c in constant_names} | {_TIME_SYM}

    free = {str(s) for s in sym_expr.free_symbols}
    if not free.issubset(allowed):
        # An un-inlined function name or an unrecognized (possibly state)
        # symbol survived → cannot guarantee a correct analytical derivative.
        return None

    result: dict = {}
    for alias, obs_name in obs_alias.items():
        if alias not in free:
            continue
        if deadline is not None and time.perf_counter() > deadline:
            # GH #95: bail out of an over-budget derivation mid-rate-law so a
            # single law coupling many observables cannot blow the budget.
            raise _DerivationBudgetExceeded
        deriv = sp.diff(sym_expr, sp.Symbol(alias))
        if deriv == 0:
            continue
        if not _is_emittable(deriv):
            return None
        result[obs_name] = deriv
    # Empty dict is a *success* (the rate has no state dependence ⇒ a zero
    # Jacobian column), distinct from None (could-not-differentiate ⇒ FD). The
    # caller must not treat the constant-rate case as a fallback trigger.
    return result


def _is_emittable(expr) -> bool:
    """True iff every function in ``expr`` maps to an ExprTk builtin. Rejects
    derivatives that introduced Heaviside / DiracDelta / special functions."""
    try:
        import sympy as sp
    except ImportError:
        return False
    for fn in expr.atoms(sp.Function):
        name = type(fn).__name__
        # Piecewise is a Function subclass in sympy but the printer emits it as
        # nested if(); relational conditions inside are handled by StrPrinter.
        if name == "Piecewise":
            continue
        if name not in _SYMPY_FUNC_TO_EXPRTK:
            return False
    return True


def _remove_removable_power_denominators(expr):
    """Rewrite ``base(x)^n / x`` terms when ``base(x)`` is a linear multiple of
    ``x``.

    SymPy often differentiates Hill/power laws as ``n*base^n/base * dbase/dx``.
    For rate-law bases such as ``x/K`` or plain ``x``, that can print as
    ``(x/K)^n/x``. Mathematically the singularity is removable for the
    non-negative concentration domain BNGsim evaluates, but the raw emitted
    ExprTk/C form produces ``0/0`` at zero-valued initial conditions. This local
    rewrite avoids a global ``simplify()`` pass on large BioModels.
    """
    import sympy as sp
    from sympy.core.traversal import bottom_up

    def rewrite_mul(node):
        if not isinstance(node, sp.Mul):
            return node

        factors = list(node.args)
        changed = True
        while changed:
            changed = False
            for denom_i, factor in enumerate(factors):
                if not (
                    isinstance(factor, sp.Pow)
                    and factor.exp == -1
                    and factor.base.is_Symbol
                ):
                    continue

                sym = factor.base
                for power_i, power_factor in enumerate(factors):
                    if power_i == denom_i or not isinstance(power_factor, sp.Pow):
                        continue
                    base, exp = power_factor.base, power_factor.exp
                    if exp == -1 or not base.has(sym):
                        continue
                    quotient = sp.cancel(base / sym)
                    if quotient.has(sym):
                        continue

                    factors[power_i] = sp.Pow(base, exp - 1, evaluate=False)
                    factors[denom_i] = quotient
                    changed = True
                    break
                if changed:
                    break

        if factors == list(node.args):
            return node
        return sp.Mul(*factors, evaluate=False)

    return bottom_up(expr, rewrite_mul)


# ─── sympy → ExprTk emitter ────────────────────────────────────────────────


def _make_printer():
    """Build an ExprTk StrPrinter subclass (lazily, so sympy import stays
    optional)."""
    from sympy.printing.str import StrPrinter

    class _ExprTkPrinter(StrPrinter):
        def _print_Pow(self, expr):
            from sympy import S

            b, e = expr.base, expr.exp
            if e == S.Half:
                return f"sqrt({self._print(b)})"
            if e == -S.Half:
                return f"(1.0/sqrt({self._print(b)}))"
            if e == -S.One:
                return f"(1.0/({self._print(b)}))"
            return f"(({self._print(b)})^({self._print(e)}))"

        def _print_Piecewise(self, expr):
            # ((v1, c1), (v2, c2), ..., (vn, True)) → if(c1, v1, if(c2, v2, ... vn))
            args = list(expr.args)
            # Default else: last True branch, or 0.0.
            else_str = "0.0"
            pieces = []
            for val, cond in args:
                if cond is True or (hasattr(cond, "is_Boolean") and cond == True):  # noqa: E712
                    else_str = self._print(val)
                    break
                pieces.append((self._print(cond), self._print(val)))
            out = else_str
            for cond_s, val_s in reversed(pieces):
                out = f"if({cond_s},{val_s},{out})"
            return out

        def _print_And(self, expr):
            return "(" + " and ".join(self._print(a) for a in expr.args) + ")"

        def _print_Or(self, expr):
            return "(" + " or ".join(self._print(a) for a in expr.args) + ")"

        def _print_Not(self, expr):
            return f"(not({self._print(expr.args[0])}))"

        def _print_Function(self, expr):
            name = type(expr).__name__
            mapped = _SYMPY_FUNC_TO_EXPRTK.get(name, name)
            return f"{mapped}({self.stringify(expr.args, ',')})"

        def _print_Abs(self, expr):
            return f"abs({self._print(expr.args[0])})"

        def _print_Float(self, expr):
            # Full round-trippable precision; ExprTk parses standard C floats.
            return repr(float(expr))

        def _print_Symbol(self, expr):
            if expr.name == _TIME_SYM:
                return "time()"
            return expr.name

        def _print_Exp1(self, expr):
            return "exp(1)"

        def _print_Pi(self, expr):
            return "_pi"

    return _ExprTkPrinter


_printer_cache: list = []


def sympy_to_exprtk(expr) -> str | None:
    """Emit a sympy expression as an ExprTk string, or ``None`` if it contains
    a construct the emitter cannot represent."""
    try:
        import sympy as sp  # noqa: F401
    except ImportError:
        return None
    if not _is_emittable(expr):
        return None
    try:
        expr = _remove_removable_power_denominators(expr)
    except Exception:
        return None
    if not _printer_cache:
        _printer_cache.append(_make_printer()())
    try:
        s = _printer_cache[0].doprint(expr)
    except Exception:
        return None
    # Never hand a non-finite literal to the engine: a NaN/Inf in a derivative
    # means the model state itself is degenerate (e.g. unset IC / volume), and
    # the analytical Jacobian must defer to FD rather than poison the matrix.
    if re.search(r"(?<![A-Za-z0-9_])(nan|inf|-inf)(?![A-Za-z0-9_])", s, re.IGNORECASE):
        return None
    return s


# ─── sympy → C emitter (codegen) ───────────────────────────────────────────
#
# The codegen counterpart of ``sympy_to_exprtk`` (GH #76, Task 4): emits the
# *same* ``differentiate_rate_law`` sympy expressions as C (``math.h``) source
# instead of ExprTk, so a compiled ``.so`` Jacobian reuses the symbolic core
# with no re-derivation. Two differences from the ExprTk emitter:
#
#   * **Symbols are not globals.** ExprTk registers observable/parameter names as
#     evaluator variables; C needs every free symbol mapped to a concrete
#     expression (``y[i]``, ``data->param_values[idx]``, the time variable). The
#     caller supplies that mapping via ``resolve_symbol``; the printer owns only
#     operator / function / literal syntax. A symbol the resolver cannot map
#     fails the whole emission (``None``) rather than referencing an undefined C
#     variable — same fail-safe contract as the rest of the module.
#   * **C idioms.** ``^``→``pow``/repeated multiply, ``if(c,t,f)``→ternary,
#     word logicals→``&&``/``||``/``!``, ``abs``→``fabs``, ``min``/``max``→
#     ``fmin``/``fmax``, integers→double literals (no C integer division).

_SYMPY_FUNC_TO_C = {
    "exp": "exp",
    "log": "log",  # natural log (C ``log``)
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    # Abs, sign, Min, Max, floor, ceiling have dedicated _print_* methods below.
}


class _CEmitError(Exception):
    """Internal: an unresolvable symbol or unmappable construct. Caught by
    ``sympy_to_c`` and converted to a ``None`` return (FD fallback)."""


def _make_c_printer():
    """Build a C ``StrPrinter`` subclass (lazily, so sympy import stays
    optional). Mirrors ``_ExprTkPrinter`` term-for-term in C syntax; every
    symbol is routed through the instance ``_resolver`` callback."""
    from sympy.printing.str import StrPrinter

    class _CPrinter(StrPrinter):
        # Set per call by sympy_to_c; None outside an emission.
        _resolver: Callable[[str], str | None] | None = None

        def _print_Pow(self, expr):
            from sympy import S

            b, e = expr.base, expr.exp
            if e == S.Half:
                return f"sqrt({self._print(b)})"
            if e == -S.Half:
                return f"(1.0/sqrt({self._print(b)}))"
            if e == -S.One:
                return f"(1.0/({self._print(b)}))"
            if e.is_Integer:
                n = int(e)
                # Small integer powers → repeated multiply (cheaper than pow(),
                # mirrors the RHS codegen's power handling). Larger / fractional
                # exponents fall through to pow().
                if 1 <= n <= 4:
                    bs = self._print(b)
                    return "(" + "*".join(f"({bs})" for _ in range(n)) + ")"
                if -4 <= n <= -1:
                    bs = self._print(b)
                    return "(1.0/(" + "*".join(f"({bs})" for _ in range(-n)) + "))"
            return f"pow({self._print(b)}, {self._print(e)})"

        def _print_Piecewise(self, expr):
            # ((v1, c1), …, (vn, True)) → (c1) ? (v1) : ((c2) ? (v2) : … : else)
            args = list(expr.args)
            else_str = "0.0"
            pieces = []
            for val, cond in args:
                if cond is True or (hasattr(cond, "is_Boolean") and cond == True):  # noqa: E712
                    else_str = self._print(val)
                    break
                pieces.append((self._print(cond), self._print(val)))
            out = else_str
            for cond_s, val_s in reversed(pieces):
                out = f"(({cond_s}) ? ({val_s}) : ({out}))"
            return out

        def _print_And(self, expr):
            return "(" + " && ".join(self._print(a) for a in expr.args) + ")"

        def _print_Or(self, expr):
            return "(" + " || ".join(self._print(a) for a in expr.args) + ")"

        def _print_Not(self, expr):
            return f"(!({self._print(expr.args[0])}))"

        def _print_Function(self, expr):
            name = type(expr).__name__
            mapped = _SYMPY_FUNC_TO_C.get(name)
            if mapped is None:
                raise _CEmitError(f"function {name}")
            return f"{mapped}({self.stringify(expr.args, ', ')})"

        def _print_Abs(self, expr):
            return f"fabs({self._print(expr.args[0])})"

        def _print_sign(self, expr):
            # double-valued signum ∈ {-1.0, 0.0, 1.0}; arises from d/dx |f|.
            a = self._print(expr.args[0])
            return f"((double)((0.0 < ({a})) - (({a}) < 0.0)))"

        def _print_floor(self, expr):
            return f"floor({self._print(expr.args[0])})"

        def _print_ceiling(self, expr):
            return f"ceil({self._print(expr.args[0])})"

        def _print_Min(self, expr):
            return self._cfold("fmin", expr.args)

        def _print_Max(self, expr):
            return self._cfold("fmax", expr.args)

        def _cfold(self, fn, args):
            ps = [self._print(a) for a in args]
            out = ps[0]
            for p in ps[1:]:
                out = f"{fn}({out}, {p})"
            return out

        def _print_Float(self, expr):
            return repr(float(expr))

        def _print_Integer(self, expr):
            # Double literal: no sub-expression can trigger C integer division.
            return f"{int(expr)}.0"

        def _print_Rational(self, expr):
            return f"({int(expr.p)}.0/{int(expr.q)}.0)"

        def _print_Symbol(self, expr):
            return self._resolve(expr.name)

        def _resolve(self, name):
            mapped = self._resolver(name) if self._resolver is not None else None
            if mapped is None:
                raise _CEmitError(f"symbol {name}")
            return mapped

        def _print_Exp1(self, expr):
            return "exp(1.0)"

        def _print_Pi(self, expr):
            return "M_PI"

    return _CPrinter


_c_printer_cache: list = []


def sympy_to_c(expr, resolve_symbol) -> str | None:
    """Emit a sympy expression as C (``math.h``) source, or ``None``.

    ``resolve_symbol(name) -> str | None`` maps each free-symbol name (an
    observable, a parameter, or the ``_TIME_SYM`` time placeholder) to a C
    expression. A ``None`` from the resolver — an unknown / un-mappable symbol —
    fails the whole emission so the model keeps its existing Jacobian rather than
    emit a reference to an undefined C variable. ``None`` is also returned for an
    un-representable construct (special function, NaN/Inf literal) — the same
    fail-safe contract as ``sympy_to_exprtk``.
    """
    try:
        import sympy as sp  # noqa: F401
    except ImportError:
        return None
    if not _is_emittable(expr):
        return None
    try:
        expr = _remove_removable_power_denominators(expr)
    except Exception:
        return None
    if not _c_printer_cache:
        _c_printer_cache.append(_make_c_printer()())
    printer = _c_printer_cache[0]
    printer._resolver = resolve_symbol
    try:
        s = printer.doprint(expr)
    except Exception:
        # _CEmitError (unresolvable symbol / unmappable fn) or any printer
        # failure → FD fallback.
        return None
    finally:
        printer._resolver = None
    # Never hand a non-finite literal to the compiler: a NaN/Inf in a derivative
    # means a degenerate model state; defer to FD rather than poison the matrix.
    if re.search(r"(?<![A-Za-z0-9_])(nan|inf|-inf)(?![A-Za-z0-9_])", s, re.IGNORECASE):
        return None
    return s


# ─── GH #198: expression / global-function output-sensitivity partials ──────
#
# These constructs make a function's output sensitivity *unsupported* and must
# fail loudly (#198), not silently FD-fall-back like the Jacobian path. They
# either have no continuous derivative (comparisons, logical operators,
# rounding) or a derivative whose boundary jump the analytic path would silently
# drop (``if()``/Piecewise, ``abs``, ``min``/``max``, ``floor``/``ceil`` — sympy
# happily differentiates these, dropping the delta, so a token pre-scan is the
# only reliable rejection). Each maps to the human-readable reason surfaced in
# the error. Names are matched only as call heads (``\bname\s*\(``) so a model
# symbol like ``absorbance`` or ``minutes`` is never falsely flagged.
_UNSUPPORTED_EXPR_CONSTRUCTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bif\s*\("), "if() conditional"),
    (re.compile(r"==|!=|<=|>=|<|>"), "comparison operator"),
    (re.compile(r"&&|\|\||\band\b|\bor\b|\bnot\b"), "logical operator"),
    (re.compile(r"(?<![=!<>])!(?!=)"), "logical-not operator"),
    (re.compile(r"\babs\s*\("), "abs()"),
    (re.compile(r"\bmin\s*\("), "min()"),
    (re.compile(r"\bmax\s*\("), "max()"),
    (re.compile(r"\bfloor\s*\("), "floor()"),
    (re.compile(r"\bceil\s*\("), "ceil()"),
    (re.compile(r"\b(?:round|rint|nint)\s*\("), "rounding function (round/rint/nint)"),
]


def unsupported_expr_construct(body: str) -> str | None:
    """Return the reason a function body is unsupported for #198 output
    sensitivities, or ``None`` if no rejected construct is present."""
    for pat, name in _UNSUPPORTED_EXPR_CONSTRUCTS:
        if pat.search(body):
            return name
    return None


def differentiate_expression_output_partials(
    body: str,
    *,
    species_cref: dict[str, str],
    observable_cref: dict[str, str],
    param_cref: dict[str, str],
    function_cref: dict[str, str],
):
    """Differentiate a global-function body w.r.t. each *directly referenced*
    symbol (species / observable / parameter / earlier function), WITHOUT
    inlining, for the #198 output-sensitivity chain rule.

    Unlike :func:`differentiate_rate_law` (which inlines functions and
    differentiates only w.r.t. observables, treating params as constants), this
    treats every directly-referenced symbol as an independent differentiation
    variable, so the caller can assemble

        df/dθ = Σ_i ∂f/∂x_i·dx_i/dθ + Σ_j ∂f/∂obs_j·dobs_j/dθ
              + Σ_k ∂f/∂p_k·dp_k/dθ + Σ_m ∂f/∂f_m·df_m/dθ

    over the same expression graph the value codegen uses. The time term is
    dropped (``d time/dθ = 0``); ``_pi`` / ``_e`` are likewise constant.

    Returns ``(partials, None)`` on success, or ``(None, reason)`` on an
    unsupported construct — #198 fails loudly rather than falling back to FD.
    ``partials`` has keys ``"species"`` / ``"observable"`` / ``"param"`` /
    ``"function"``; each maps a referenced *original* symbol name to the C
    expression for that partial (zero partials omitted). The C expressions
    reference ``y[i]`` / ``obs[j]`` / ``p[k]`` / ``func[l]`` / ``t`` via the
    supplied ``*_cref`` maps, so they stay byte-consistent with the value codegen
    (``_emit_function_lines``). On a name collision across kinds, precedence
    matches the value path's ``_build_ident_lookup_model``: function > observable
    > species > parameter.
    """
    reason = unsupported_expr_construct(body)
    if reason is not None:
        return None, f"uses unsupported construct: {reason}"

    try:
        import sympy as sp
    except ImportError:  # pragma: no cover - sympy is a hard dep of codegen sens
        return None, "sympy is required for expression output sensitivities"

    sym_expr = _exprtk_to_sympy(body)
    if sym_expr is None:
        return None, "could not parse expression for differentiation"

    # Keyword-named params get a safe alias in _exprtk_to_sympy; round-trip it
    # here so the sympy symbol name resolves back to the right kind / C ref.
    def _alias(n: str) -> str:
        return _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n

    # Lowest precedence first so a higher-precedence kind overwrites on collision
    # (mirrors _build_ident_lookup_model: param < species < observable < func).
    kinds = (
        ("param", param_cref),
        ("species", species_cref),
        ("observable", observable_cref),
        ("function", function_cref),
    )
    alias_kind: dict[str, str] = {}
    alias_orig: dict[str, str] = {}
    alias_cref: dict[str, str] = {}
    for kind, cref in kinds:
        for name, c in cref.items():
            a = _alias(name)
            alias_kind[a] = kind
            alias_orig[a] = name
            alias_cref[a] = c

    ignored = {_TIME_SYM, "_pi", "_e"}

    free = {str(s) for s in sym_expr.free_symbols}
    unknown = free - set(alias_cref) - ignored
    if unknown:
        return None, "references unrecognized symbol(s): " + ", ".join(sorted(unknown))

    def resolve(name: str) -> str | None:
        if name == _TIME_SYM:
            return "t"
        if name == "_pi":
            return "M_PI"
        if name == "_e":
            return "M_E"
        return alias_cref.get(name)

    partials: dict[str, dict[str, str]] = {
        "species": {},
        "observable": {},
        "param": {},
        "function": {},
    }
    for alias in free:
        if alias in ignored:
            continue
        deriv = sp.diff(sym_expr, sp.Symbol(alias))
        if deriv == 0:
            continue
        c_expr = sympy_to_c(deriv, resolve)
        if c_expr is None:
            return None, (
                f"derivative w.r.t. {alias_orig[alias]} is not representable in C "
                "(non-differentiable or unsupported function)"
            )
        partials[alias_kind[alias]][alias_orig[alias]] = c_expr
    return partials, None


# ─── Consumer-facing packaging ─────────────────────────────────────────────


def build_per_species_sympy(
    rate_expr: str,
    func_map: dict[str, str],
    obs_groups: dict[str, list],
    species_amount: dict[int, tuple],
    constant_names: set[str],
    deadline: float | None = None,
):
    """Chain-rule ``∂rate/∂obs_k`` through each observable's species group into
    one per-species *sympy* derivative:

        ∂rate/∂x_j = Σ_k (∂rate/∂obs_k) · factor_{k→j} · (V_j if x_j amount-valued)

    The consumer-agnostic per-species core shared by the ExprTk packaging
    (:func:`build_per_species_terms`) and the codegen C emitter
    (``_codegen.generate_jacobian_from_model``). Returns ``[(species_idx0,
    sympy_expr)]`` (expressions still in observable / parameter / time symbols),
    or ``None`` to fall back. ``[]`` is a *success* (constant-rate ⇒ zero column),
    distinct from ``None`` (differentiation failure ⇒ FD fallback).

    ``obs_groups``: ``observable_name → [(species_idx0, factor), …]``.
    ``species_amount``: ``species_idx0 → (amount_valued: bool, volume_factor: float)``.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    dd = differentiate_rate_law(rate_expr, func_map, set(obs_groups), constant_names, deadline)
    if dd is None:
        return None

    import math

    per_species: dict = {}
    for obs_name, dexpr in dd.items():
        for sp_idx, factor in obs_groups[obs_name]:
            amount_valued, vol = species_amount.get(sp_idx, (False, 1.0))
            coeff = float(factor) * (float(vol) if amount_valued else 1.0)
            if not math.isfinite(coeff):
                # Degenerate chain-rule factor (e.g. an unset compartment volume
                # surfaced as NaN). Defer the whole model to FD.
                return None
            if coeff == 0.0:
                continue
            per_species[sp_idx] = per_species.get(sp_idx, sp.Integer(0)) + sp.Float(coeff) * dexpr

    out = []
    for sp_idx, expr in per_species.items():
        # No sympy.simplify(): it is cosmetic and can dominate load time on
        # large rate laws. diff() already yields a compact form, and a term
        # that happens to be structurally zero just contributes a zero column.
        if expr == 0:
            continue
        out.append((sp_idx, expr))
    return out


def build_per_species_terms(
    rate_expr: str,
    func_map: dict[str, str],
    obs_groups: dict[str, list],
    species_amount: dict[int, tuple],
    constant_names: set[str],
    deadline: float | None = None,
) -> list[tuple[int, str]] | None:
    """SBML / ``apply_species_factor=false`` path.

    Emits the :func:`build_per_species_sympy` per-species derivatives as ExprTk
    strings. Returns ``[(species_idx0, exprtk_str)]`` (expressions still in
    observable / parameter / time symbols, hence live-evaluable), or ``None`` to
    fall back. ``deadline`` (GH #95) is forwarded to the symbolic core.

    GH #151: a rate law in the recognized saturable family (Hill / rational /
    basal+regulated / multi-regulator) is differentiated and emitted natively —
    no SymPy, no derivation budget. The native path returns ``None`` for anything
    outside the family, so the SymPy path below remains the fallback.
    """
    native = _native_per_species_terms(
        rate_expr, func_map, obs_groups, species_amount, constant_names
    )
    if native is not None:
        return native

    terms = build_per_species_sympy(
        rate_expr, func_map, obs_groups, species_amount, constant_names, deadline
    )
    if terms is None:
        return None
    out: list[tuple[int, str]] = []
    for sp_idx, expr in terms:
        s = sympy_to_exprtk(expr)
        if s is None:
            return None
        out.append((sp_idx, s))
    # [] = covered with a zero column (constant-rate functional reaction); None
    # was already returned above on a genuine differentiation failure.
    return out


def build_per_observable_terms(
    rate_expr: str,
    func_map: dict[str, str],
    observable_names: set[str],
    constant_names: set[str],
    deadline: float | None = None,
) -> list[tuple[str, str]] | None:
    """``.net`` / ``apply_species_factor=true`` path.

    Emits ``∂rate/∂obs_k`` per observable as ExprTk; the C++ callback scatters
    each through the observable's species group and combines with the
    mass-action species factor via the product rule. Returns
    ``[(observable_name, exprtk_str)]`` or ``None``. ``deadline`` (GH #95) is
    forwarded to the symbolic core.

    GH #151: tries the native saturable path (no SymPy) first — the legacy
    ``.net`` ``Sat``/``Hill`` rewrites (#48) land here, and their derivatives are
    closed-form rational/power expressions the native differentiator emits
    directly. SymPy below is the fallback for everything else.
    """
    native = _native_per_observable_terms(rate_expr, func_map, observable_names, constant_names)
    if native is not None:
        return native

    dd = differentiate_rate_law(rate_expr, func_map, observable_names, constant_names, deadline)
    if dd is None:
        return None
    out: list[tuple[str, str]] = []
    for obs_name, dexpr in dd.items():
        s = sympy_to_exprtk(dexpr)
        if s is None:
            return None
        out.append((obs_name, s))
    # [] = covered with a zero contribution (constant-rate functional reaction).
    return out


# ─── GH #151 native saturable-family path (no SymPy) ────────────────────────
#
# The native engine (``bngsim._saturable_jacobian``) recognizes the fixed
# saturable rate-law family — Hill, rational/saturation (the legacy ``Sat``/
# ``Hill`` ``.net`` rewrites, #48), basal+regulated production, and products /
# shared-denominator sums of those over several regulators — and differentiates
# it in closed form, emitting ExprTk or C with zero SymPy invocations and no
# derivation-budget pressure (#95). Every helper returns ``None`` for an
# expression outside the family, so the SymPy path above stays the fallback. The
# import is local so ``_saturable_jacobian`` (which imports names from this
# module) loads without a cycle, and SymPy is never imported on the native path.


def _native_per_observable_terms(rate_expr, func_map, observable_names, constant_names):
    """Native ∂func/∂obs_k as ExprTk strings (``.net`` per-observable path), or
    ``None`` if the rate law is outside the saturable family."""
    from bngsim import _saturable_jacobian as _sat

    dd = _sat.differentiate_rate_law_native(rate_expr, func_map, observable_names, constant_names)
    if dd is None:
        return None
    out: list[tuple[str, str]] = []
    for obs_name, node in dd.items():
        s = _sat.emit_exprtk(node)
        if s is None:
            return None
        out.append((obs_name, s))
    return out


def _native_per_species_terms(rate_expr, func_map, obs_groups, species_amount, constant_names):
    """Native per-species derivatives as ExprTk strings (SBML path), or
    ``None``."""
    from bngsim import _saturable_jacobian as _sat

    terms = _sat.build_per_species_native(
        rate_expr, func_map, obs_groups, species_amount, constant_names
    )
    if terms is None:
        return None
    out: list[tuple[int, str]] = []
    for sp_idx, node in terms:
        s = _sat.emit_exprtk(node)
        if s is None:
            return None
        out.append((sp_idx, s))
    return out


def build_per_species_c(
    rate_expr, func_map, obs_groups, species_amount, constant_names, resolve_symbol
) -> list[tuple[int, str]] | None:
    """Codegen per-species path: ``[(species_idx0, c_str)]`` or ``None``.

    Tries the native saturable family first (no SymPy), then the SymPy path
    (:func:`build_per_species_sympy` + :func:`sympy_to_c`). ``[]`` is a *success*
    (constant-rate ⇒ zero column). A native success whose C emission fails (an
    unresolvable symbol) falls through to SymPy rather than declining outright."""
    from bngsim import _saturable_jacobian as _sat

    native = _sat.build_per_species_native(
        rate_expr, func_map, obs_groups, species_amount, constant_names
    )
    if native is not None:
        out: list[tuple[int, str]] = []
        emitted = True
        for sp_idx, node in native:
            c = _sat.emit_c(node, resolve_symbol)
            if c is None:
                emitted = False
                break
            out.append((sp_idx, c))
        if emitted:
            return out

    terms = build_per_species_sympy(
        rate_expr, func_map, obs_groups, species_amount, constant_names
    )
    if terms is None:
        return None
    out2: list[tuple[int, str]] = []
    for sp_idx, expr in terms:
        c = sympy_to_c(expr, resolve_symbol)
        if c is None:
            return None
        out2.append((sp_idx, c))
    return out2


def differentiate_rate_law_c(
    rate_expr, func_map, observable_names, constant_names, resolve_symbol
) -> list[tuple[str, str]] | None:
    """Codegen per-observable path: ordered ``[(observable_name, c_str)]`` or
    ``None``. Native saturable family first (no SymPy), then SymPy."""
    from bngsim import _saturable_jacobian as _sat

    nd = _sat.differentiate_rate_law_native(rate_expr, func_map, observable_names, constant_names)
    if nd is not None:
        out: list[tuple[str, str]] = []
        emitted = True
        for obs_name, node in nd.items():
            c = _sat.emit_c(node, resolve_symbol)
            if c is None:
                emitted = False
                break
            out.append((obs_name, c))
        if emitted:
            return out

    dd = differentiate_rate_law(rate_expr, func_map, observable_names, constant_names)
    if dd is None:
        return None
    out2: list[tuple[str, str]] = []
    for obs_name, dexpr in dd.items():
        c = sympy_to_c(dexpr, resolve_symbol)
        if c is None:
            return None
        out2.append((obs_name, c))
    return out2


# ─── Build-time derivation budget (GH #95) ─────────────────────────────────
#
# The #76 analytical Jacobian is a *bet*: pay a one-time symbolic-derivation cost
# at build so every Newton Jacobian-setup in the solve is an O(nnz) eval instead
# of FD's (n+1) RHS evals. The bet wins only when derivation cost ≪ solve savings,
# and per-derivative sympy cost grows super-linearly with (inlined) rate-law size
# and observable coupling. On a handful of large BioModels (e.g. BIOMD0000000496,
# 0000000628, 0000000595, MODEL1001200000) the derivation runs 40 s–>1 min while
# the ODE solve is already sub-second under FD — measured: analytical and FD solve
# times are identical to within noise, so the derivation is pure wasted build
# time. Worse, the rr_parity harness times build+solve together, so a slow build
# reads as an ODE "timeout" (GH #95).
#
# Fix: bound the derivation wall-time. A model that derives under budget keeps the
# analytical Jacobian; one that does not falls back to the finite-difference
# Jacobian instead of hanging — adaptive, and a strict win on the losers (same
# solve, build collapses 47×–>100×). The budget is checked both between reactions
# and inside differentiate_rate_law's per-observable loop, so overshoot is bounded
# to one rate law's derivative even for a pathological single reaction.

# Base wall-clock budget (seconds) for the build-time derivation on a *small*
# model.
#
# Choosing the value: the budget must (a) exceed the derivation time of every model
# whose *solve* genuinely needs the analytical Jacobian, or it would regress that
# model from PASS to a solver failure, and (b) stay below the pathological
# derivations it exists to cut off. A full rr_parity corpus classification (build
# analytical vs finite-difference, then solve at the harness's 1e-9/1e-12
# tolerance) found the two populations cleanly separated:
#
#   * needs-analytical (FD solve *fails*): only BIOMD0000000457 among the corpus —
#     derives in ~12 s (even under 4-worker contention) and must keep its
#     analytical Jacobian.
#   * derivation losers (FD solve identical, derivation pure waste): the next
#     slowest is BIOMD0000000496 at ~41 s, then 595/574/628 at 52–75 s.
#
# 20 s sits in the [~12 s, ~41 s] gap with ~1.7x margin over the slowest
# needs-analytical model and ~2x under the fastest loser — the widest separation a
# single fixed budget gives this pair. Crucially every model in that pair is small
# (<= 295 species), so on the model-size-scaled budget below they all stay pinned
# to this base. Models that derive quickly (the vast majority, << 1 s) are
# unaffected. Override / disable with BNGSIM_JAC_DERIV_BUDGET_S (<= 0 or
# "inf"/"none" → unbounded, the pre-#95 behavior; raise it on a slow machine if a
# needs-analytical model logs a fallback).
_DEFAULT_DERIVATION_BUDGET_S = 20.0

# GH #187: scale the budget with model size, and make the finite-difference (FD)
# fallback safe at scale.
#
# The fixed base above is tuned to cut off pathological *small* models, where the
# FD Jacobian is a perfectly good fallback — its (n+1) RHS-eval cost per Newton
# setup is negligible at small n, so a dropped derivation costs nothing. That
# assumption inverts at scale: an FD Jacobian needs ~n_species RHS evals *per
# formation*, so on a genome-scale model (tens of thousands of species) it is not a
# viable solver path at all — falling back to it is effectively non-terminating,
# not merely wasteful. The fixed wall-clock budget is also machine-dependent, which
# turned it into a silent correctness/performance cliff that fired only on slower
# or busier nodes that finished derivation just over the cap (the GS-SPARCED case:
# 74,795 species, ~33 s derivation — cut off at the 20 s default, dropped to an
# unrunnable FD solve).
#
# Two model-size-aware adjustments keep the #95 small-model protection while
# removing the cliff for large ones. The budget is keyed on n_species — the same
# quantity that drives FD cost — so it is tight exactly where FD is cheap and
# generous exactly where FD is unviable, and the #95 losers (139–295 species, slow
# derivation) stay pinned to the base, far below the scaling knee:
#
#   1. Scale: budget = max(base, _BUDGET_PER_SPECIES_S * n_species). At
#      ~0.5 ms/species observed (GS-SPARCED), the 5 ms/species slope is ~10x the
#      real derivation rate, so a model whose derivation scales like GS-SPARCED is
#      never cut off. The base dominates below the ~4000-species knee (base /
#      slope), covering every BioModels-scale model unchanged.
#   2. Gate: at/above _FD_NONVIABLE_SPECIES the budget is unbounded outright —
#      there is no good fallback to cut over to (FD never converges), so cutting
#      the derivation off only breaks the solve. This is the hard guarantee that
#      backs up the (already generous) scaling for super-linear derivations.
#
# An explicit BNGSIM_JAC_DERIV_BUDGET_S still wins over both (absolute seconds, or
# inf/none/off/0 for unbounded — the documented genome-scale workaround).

# Per-species derivation allowance (seconds). 5 ms/species; the base dominates
# below ~4000 species (= base / slope), so small models are unaffected.
_BUDGET_PER_SPECIES_S = 0.005

# Species count at/above which a finite-difference Jacobian is not a viable solver
# path (it needs ~n_species RHS evals per Newton Jacobian setup). At/above this the
# analytical Jacobian is mandatory: the derivation budget is unbounded regardless
# of wall-clock, because falling back to FD here would not converge.
_FD_NONVIABLE_SPECIES = 20000

# Species count above which an FD fallback is costly enough that a budget expiry is
# escalated from INFO to a WARNING carrying the BNGSIM_JAC_DERIV_BUDGET_S=inf
# workaround — so the degradation is loud, not silent (GH #187 option 3). Below it
# FD is cheap (the #95 case) and the fallback is logged at INFO as before.
_FD_COSTLY_SPECIES = 2000


class _DerivationBudgetExceeded(Exception):
    """Internal signal: the build-time symbolic derivation passed its wall-clock
    budget. Caught by :func:`attach_functional_jacobian`, which logs the fallback
    and leaves the model on the finite-difference Jacobian (GH #95)."""


def _derivation_budget_s(n_species: int = 0) -> float | None:
    """Resolve the build-time derivation budget in seconds, or ``None`` for
    unbounded.

    An explicit ``BNGSIM_JAC_DERIV_BUDGET_S`` wins over everything: an absolute
    number of seconds, or ``inf``/``none``/``off``/``0`` for unbounded (the pre-#95
    and documented genome-scale workaround). A non-positive or non-finite value
    also disables the budget. With the env var unset (or malformed) the budget is
    derived from ``n_species`` (GH #187): unbounded once the model is too large for
    a finite-difference Jacobian to be a viable fallback
    (``n_species >= _FD_NONVIABLE_SPECIES``), otherwise the #95 base scaled up by
    species count."""
    # Size-derived default, used when the env var is unset or malformed.
    if n_species >= _FD_NONVIABLE_SPECIES:
        default: float | None = None  # FD non-viable → analytical mandatory
    else:
        default = max(_DEFAULT_DERIVATION_BUDGET_S, _BUDGET_PER_SPECIES_S * n_species)

    raw = os.environ.get("BNGSIM_JAC_DERIV_BUDGET_S")
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in ("inf", "none", "off", "0"):
        return None
    try:
        val = float(raw)
    except ValueError:
        return default
    if val <= 0 or val != val or val == float("inf"):
        return None
    return val


def eager_jacobian_requested(defer_jacobian: bool | None = None) -> bool:
    """Whether the analytical Functional Jacobian should be derived eagerly at
    model load (GH #145 escape hatch).

    The default is lazy: the derivation is deferred off the load path to the
    first ODE-solve setup (``Model.prepare_analytical_jacobian``). Eager restores
    the pre-#145 derive-at-load behavior for A/B and safety, selected by an
    explicit ``defer_jacobian=False`` or ``BNGSIM_EAGER_JACOBIAN=1``. The env var
    is checked for every load path; the ``defer_jacobian`` argument is the
    per-call override exposed by ``from_sbml`` / ``from_net``."""
    if defer_jacobian is False:
        return True
    return os.environ.get("BNGSIM_EAGER_JACOBIAN") == "1"


# ─── Post-build attach driver (Python drives C++; the engine never calls us) ──


def attach_functional_jacobian(core) -> bool:
    """Differentiate every Functional rate law of a freshly-built model and
    attach the analytical Jacobian terms.

    Reads the model's functional context, runs the symbolic core, and writes the
    derivative expressions back via ``core.set_functional_jacobian``. Returns
    ``True`` if the analytical Jacobian was populated, ``False`` if anything fell
    back (the model then keeps the finite-difference Jacobian — no error). Never
    raises: a model that cannot be differentiated simply runs as before.

    This is the single uniform entry point for both loaders — ``Model.from_sbml``
    and ``Model.from_net`` call it after construction. It runs once at load time;
    the integration loop never touches Python.
    """
    # ON by default (GH #76). Validated across the full BioModels SBML corpus
    # (1597 models): every attached analytical Jacobian matches the engine's own
    # RHS derivative — zero wrong attaches — and the in-C++ FD self-validation
    # gate (NetworkModel::set_functional_jacobian, reliability-gated two-step
    # finite differences + a non-finite-entry guard) provably bails to the
    # finite-difference Jacobian for the cases the symbolic core cannot handle
    # exactly (singular-at-init derivatives, residual inlining divergences). So
    # the analytical path is a strict speedup where it attaches and byte-identical
    # to before where it does not. All-Elementary models are unaffected (there are
    # no Functional reactions to differentiate; this returns early below).
    #
    # Escape hatch: set BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0 to force the
    # finite-difference Jacobian on every model (e.g. to A/B the feature).
    if os.environ.get("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC") == "0":
        return False

    # GH #151: no hard SymPy gate here. Each rate law is tried on the native
    # saturable path first (no SymPy); SymPy is imported lazily only for the
    # genuine fallback (``differentiate_rate_law`` self-guards on ImportError).
    # So a model whose Functional rate laws are entirely within the saturable
    # family attaches a complete analytical Jacobian with zero SymPy invocations
    # — even with SymPy uninstalled.

    try:
        ctx = core.functional_jacobian_context()
    except Exception:
        return False
    rxns = ctx.get("functional_reactions") or []
    if not rxns:
        return False  # no Functional reactions — Elementary path already handled

    func_map = dict(ctx["function_map"])
    obs_groups = {name: [(int(si), float(f)) for si, f in grp] for name, grp in ctx["observables"]}
    obs_idx = {name: i for i, (name, _grp) in enumerate(ctx["observables"])}
    species_meta = {i: (bool(av), float(vf)) for i, (av, vf) in enumerate(ctx["species_meta"])}
    constants = set(ctx["constant_names"])
    obs_names = set(obs_groups)

    # GH #95: bound the build-time symbolic derivation. A model that derives under
    # budget keeps the analytical Jacobian; one that exceeds it falls back to the
    # finite-difference Jacobian rather than hanging the load. GH #187: the budget
    # scales with species count and goes unbounded once an FD Jacobian is no longer
    # a viable fallback, so a genome-scale model is never silently dropped to an
    # intractable FD solve.
    n_species = len(species_meta)
    budget = _derivation_budget_s(n_species=n_species)
    start = time.perf_counter()
    deadline = (start + budget) if budget is not None else None

    all_terms = []
    processed = 0
    try:
        for rxn in rxns:
            if deadline is not None and time.perf_counter() > deadline:
                # Between-reaction check: catches the accumulating cost of many
                # moderately-priced reactions (the per-observable check inside
                # differentiate_rate_law catches a single expensive rate law).
                raise _DerivationBudgetExceeded
            rate_expr = rxn["rate_expr"]
            # The mass-action species factor (∏ reactant conc) only contributes a
            # product-rule term when it is non-trivial. apply_species_factor with no
            # reactants (rate-rule reactions, SBML kinetic laws) has ∏ = 1, so the
            # rate is just func·stat ⇒ the per-species path applies. Only a true
            # non-empty species factor (.net Functional) needs per-observable +
            # the product rule.
            has_species_factor = rxn["apply_species_factor"] and len(rxn["reactant_idx0"]) > 0
            if has_species_factor:
                # .net per-observable path: emit ∂func/∂obs_k; the C++ callback
                # scatters through the observable group and applies the mass-action
                # species-factor product rule. (Engaged once the C++ per-observable
                # path lands; until then set_functional_jacobian rejects it and the
                # model falls back to FD — never wrong, just not yet accelerated.)
                obs_terms = build_per_observable_terms(
                    rate_expr, func_map, obs_names, constants, deadline
                )
                if obs_terms is None:
                    return False
                keyed = [(obs_idx[name], expr) for name, expr in obs_terms]
                all_terms.append((rxn["rxn_idx"], True, keyed))
            else:
                # SBML per-species path.
                sp_terms = build_per_species_terms(
                    rate_expr, func_map, obs_groups, species_meta, constants, deadline
                )
                if sp_terms is None:
                    return False
                all_terms.append((rxn["rxn_idx"], False, [(int(j), expr) for j, expr in sp_terms]))
            processed += 1
    except _DerivationBudgetExceeded:
        elapsed = time.perf_counter() - start
        if n_species >= _FD_COSTLY_SPECIES:
            # GH #187 option 3: on a large model the FD fallback is costly (it needs
            # ~n_species RHS evals per Newton setup) and may not converge at all, so
            # make the degradation loud and name the exact workaround. This only
            # fires when the budget was forced finite below the model's needs — the
            # size-scaled default goes unbounded at _FD_NONVIABLE_SPECIES precisely
            # to avoid reaching here on a genome-scale model.
            logger.warning(
                "GH#76 analytical Jacobian: build-time derivation exceeded the %.1fs "
                "budget after %d/%d functional reactions (%.1fs elapsed) on a large "
                "model (%d species); falling back to the finite-difference Jacobian, "
                "which needs ~%d RHS evaluations per Newton step and may be extremely "
                "slow or fail to converge at this scale. Set "
                "BNGSIM_JAC_DERIV_BUDGET_S=inf to keep the analytical Jacobian.",
                budget,
                processed,
                len(rxns),
                elapsed,
                n_species,
                n_species,
            )
        else:
            logger.info(
                "GH#76 analytical Jacobian: build-time derivation exceeded the %.1fs "
                "budget after %d/%d functional reactions (%.1fs elapsed); using the "
                "finite-difference Jacobian instead. Set BNGSIM_JAC_DERIV_BUDGET_S to "
                "raise or disable the budget.",
                budget,
                processed,
                len(rxns),
                elapsed,
            )
        return False

    try:
        return bool(core.set_functional_jacobian(all_terms))
    except Exception:
        return False
