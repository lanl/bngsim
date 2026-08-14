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
import threading
import time
from collections.abc import Callable

# Reuse the BNGL/ExprTk string rewriters from the codegen path: ``if(c,t,f)`` →
# sympy ``Piecewise`` (issue #27) and the logical-operator rewrite (issues #53,
# #56). Both live in ``_codegen`` so the derived-parameter chain-rule path there
# and this symbolic core share one implementation, with the import going one
# way only.
from bngsim._codegen import (
    _PY_KEYWORD_PARAM_NAMES,
    _alias_keyword_param,
    _rewrite_logicals,
    _translate_bngl_if_to_piecewise,
)

logger = logging.getLogger("bngsim")

# Placeholder symbol standing in for ExprTk ``time()`` / ``t()`` while the
# expression lives in sympy. It is a constant w.r.t. species, and the ExprTk
# emitter maps it back to ``time()``.
_TIME_SYM = "_bngsim_time_csymbol"

# The prefix ``_alias_keyword_param`` prepends, taken FROM that function so the
# alias and its inverse cannot drift apart. ``sympy_to_exprtk`` is the only
# emitter that has to undo the alias itself (see ``_print_Symbol``).
_KW_ALIAS_PREFIX = _alias_keyword_param("")

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
# A zero-arg call: `divide()`. BNGL accepts an Observable (and any other
# scalar) written as a call wherever the bareword is valid, and BNG2.pl
# preserves whichever form the user wrote (issue #28). No ExprTk built-in
# takes zero arguments except ``time()``/``t()`` — which _preprocess_exprtk
# rewrites to a placeholder *before* this pattern runs — so an empty argument
# list is unambiguously a scalar reference and the parens must go.
_EMPTY_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\)")


# ─── ExprTk string → sympy ────────────────────────────────────────────────


def _preprocess_exprtk(expr: str) -> str:
    """Rewrite an ExprTk expression string into a form ``sympy.parse_expr`` can
    tokenize: ``time()``→placeholder, ``obs()``→``obs``, ``if(c,t,f)``→Piecewise,
    ``^``→``**``, logicals → sympy ``And``/``Or``/``Not`` calls."""
    s = expr.strip()
    # time() / t() → constant placeholder (whole-word, parens required so a
    # parameter literally named ``t`` is untouched).
    s = re.sub(r"\b(?:time|t)\s*\(\s*\)", _TIME_SYM, s)
    # obs() → obs for every remaining zero-arg call (issue #28), mirroring
    # ``ExprTkEvaluator::compile``'s strip_empty_parens (src/expression.cpp).
    # Without it ``parse_expr`` builds an *applied undefined function* with no
    # arguments, which shares no symbol with the bareword form: differentiating
    # ``100*divide()`` w.r.t. the observable ``divide`` then yields 0 — a
    # silently wrong Jacobian entry / output-sensitivity partial — while
    # ``k1*Atot()`` yields a derivative the C printer cannot render, i.e. a
    # spurious "not representable" decline. Runs after the time rewrite, which
    # needs the parens it matches on.
    s = _EMPTY_CALL_RE.sub(r"\1", s)
    # if(c, t, f) → Piecewise((t, c), (f, True))
    s = _translate_bngl_if_to_piecewise(s)
    # power operator
    s = s.replace("^", "**")
    # Logicals → sympy call form, so parse_expr never evaluates the truth value
    # of a symbolic operand and BNGL's precedence survives. ``not(x)`` becomes
    # ``Not(x)`` first (all three are bound in the local dict).
    s = re.sub(r"\bnot\s*\(", "Not(", s)
    s = _rewrite_logicals(s)
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
        # safe alias so parse_expr can tokenize them. Keywords only: this module
        # emits through :func:`sympy_to_c`, whose resolve callback never lets
        # sympy print a symbol name, so a C reserved word needs no alias to
        # survive the round trip (GH #108; the rule is in
        # :func:`bngsim._codegen._alias_keyword_param`).
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


# GH #250: functions the emitter accepts as *input* but whose derivative it can
# never print, so a rate law using one over a differentiation variable is going to
# decline however long the derivation is allowed to run.
#
# Derived rather than listed by hand: differentiate every name in
# _SYMPY_FUNC_TO_EXPRTK and check whether the result survives _is_emittable and
# the printers. Exactly six do not, in two flavours —
#
#   * Abs / Min / Max produce re()/im()/Heaviside, which are Functions outside the
#     emitter map, so _is_emittable already rejects them *after* the derivation;
#   * ceiling / floor / sign produce an unevaluated Derivative, which is not a
#     Function at all and needed _is_emittable's own fix above.
#
# Catching them before ``sp.diff`` is a build-time saving, not a behaviour change:
# the model declines either way. The saving is not marginal, because these are
# exactly the constructs sympy is worst at. BIOMD0000000385 carries three Abs over
# its differentiation variables and spends **138 s** discovering the decline — a
# 6.9x overshoot of the 20 s budget, and unbounded by it, since the budget can only
# be tested between sp.diff calls and one of these *is* a single call. Subdividing
# the derivation does not help: recursing to 117 checkable steps still leaves the
# two dAbs at 62.4 s and 34.8 s, with every other step under 0.04 s. The 138 s
# becomes ~0.5 s here, and the model lands on the same FD Jacobian it did before.
#
# Matched on the type name during one traversal, because Min/Max are Application
# but *not* Function subclasses, so ``atoms(sp.Function)`` misses them.
_NONDIFFERENTIABLE_EMITTER_FUNCS = frozenset({"Abs", "Max", "Min", "ceiling", "floor", "sign"})


def _nondifferentiable_over(expr, targets: set[str]) -> str | None:
    """Name of a :data:`_NONDIFFERENTIABLE_EMITTER_FUNCS` node in a position that
    will be differentiated with respect to one of ``targets``, or ``None`` if the
    expression can be differentiated and emitted for all of them (GH #250).

    Two things keep this from over-declining, and the corpus has a model for each:

    * **The target intersection.** ``Abs(k) * A`` differentiates to ``Abs(k)``,
      perfectly emittable — only a *differentiation variable* under one of these
      functions is a problem, not a parameter.
    * **Piecewise conditions are not differentiated.** ``d/dx Piecewise((f, c), …)``
      is ``Piecewise((df/dx, c), …)``: the conditions are copied through verbatim,
      so a blocked function inside one is emitted unchanged and stays legal.
      ``MODEL1006230034`` is the case — ``Piecewise(…, mincond_J_K < Abs(deltaPsi))``
      over ``deltaPsi``, a differentiation variable — and it keeps a complete
      analytical Jacobian today. A position-blind scan would take it away.

    Hence an explicit walk rather than ``preorder_traversal``: the recursion has to
    skip the condition half of every ``ExprCondPair``.
    """
    try:
        import sympy as sp
    except ImportError:
        return None
    stack = [expr]
    while stack:
        node = stack.pop()
        name = type(node).__name__
        if (
            name in _NONDIFFERENTIABLE_EMITTER_FUNCS
            and {str(s) for s in node.free_symbols} & targets
        ):
            return name
        if isinstance(node, sp.Piecewise):
            stack.extend(value for value, _cond in node.args)
        else:
            stack.extend(node.args)
    return None


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

    # Build aliased name sets to match what _exprtk_to_sympy produced. Python
    # keywords only, matching `_build_local_dict` above — nothing here reaches
    # `sp.ccode`, so C reserved words are not a round-trip hazard (GH #108).
    def _alias(n: str) -> str:
        return _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n

    obs_alias = {_alias(o): o for o in observable_names}
    allowed = set(obs_alias) | {_alias(c) for c in constant_names} | {_TIME_SYM}

    free = {str(s) for s in sym_expr.free_symbols}
    if not free.issubset(allowed):
        # An un-inlined function name or an unrecognized (possibly state)
        # symbol survived → cannot guarantee a correct analytical derivative.
        return None

    # GH #250: fall back *before* differentiating when the answer is already
    # decided. See _NONDIFFERENTIABLE_EMITTER_FUNCS — this is the same decline the
    # _is_emittable check below would reach, minus the derivation, and on the one
    # corpus model that hits it that is 138 s minus.
    blocked = _nondifferentiable_over(sym_expr, {a for a in obs_alias if a in free})
    if blocked is not None:
        logger.debug(
            "GH#76 analytical Jacobian: rate law uses %s over a differentiation "
            "variable, whose derivative cannot be emitted; using finite differences.",
            blocked,
        )
        return None

    result: dict = {}
    # Sorted so the emitted ExprTk/C derivative ordering is deterministic
    # regardless of set hash-seed — ``observable_names`` is a set, so iterating it
    # ordered the result (and every downstream ``d0``/``d1`` temporary and its
    # scatter) by PYTHONHASHSEED. Codegen output is content-addressed and cached,
    # so a rate law reading two or more observables re-hashed on every process and
    # missed its own ``.so``. The saturable twin
    # (``_saturable_jacobian.differentiate_rate_law_native``) already sorts for
    # exactly this reason; this is the sympy branch catching up. Ordering never
    # affects the numerical result.
    for alias, obs_name in sorted(obs_alias.items()):
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
    derivatives that introduced Heaviside / DiracDelta / special functions, and
    any unevaluated ``Derivative`` (GH #250)."""
    try:
        import sympy as sp
    except ImportError:
        return False
    # An unevaluated Derivative is sympy saying it *cannot* differentiate the
    # node: sign, floor and ceiling all come back as `Derivative(f(x), x)`. It is
    # not a Function, so the atoms() scan below never sees it, and both printers
    # then fall through to printing it verbatim — `Derivative(sign(x), x)` is not
    # an ExprTk builtin, and `Derivative(...)` is not a declared C function, so
    # what reached the emitter was a broken derivative rather than a fallback to
    # FD. Checked first because it is the cheap structural case (GH #250).
    if expr.has(sp.Derivative):
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


def _linear_multiple_quotient(base, sym, sp):
    """``base / sym`` when ``base`` is a linear multiple of ``sym`` — i.e. when
    the quotient is free of ``sym`` — computed structurally, or ``None`` (GH #96).

    This is the exact question :func:`_remove_removable_power_denominators` asks,
    and the only one it can act on: it needs ``base = c·sym`` so that
    ``base^n / sym`` can become ``c · base^(n-1)``. ``sp.cancel`` answered it by
    normalising the whole rational expression first, which is unbounded work on a
    large ``Add`` and is thrown away whenever the quotient still contains ``sym``
    — the overwhelmingly common case, since a base that merely *mentions* ``sym``
    (``Km + sym``) is not a multiple of it.

    Handles the three shapes that can be one:

    * ``sym`` itself → ``1``;
    * a ``Mul`` carrying ``sym`` as a first-power factor → the other factors,
      provided none of *them* mentions ``sym`` (``sym·(a + sym)`` is quadratic,
      not linear, so its quotient is rejected);
    * an ``Add`` whose every term is itself a linear multiple → the sum of the
      term quotients (``a·sym + b·sym`` → ``a + b``), which is the one case a
      purely local ``Mul`` test would miss and ``cancel`` did catch.

    Anything else — a genuine sum like ``Km + sym``, a power ``sym^2``, a
    transcendental — is not a linear multiple and returns ``None``. That is the
    same rejection ``cancel`` reached, just without the normalisation.
    """
    if base == sym:
        return sp.S.One
    if isinstance(base, sp.Mul):
        rest = []
        found = False
        for arg in base.args:
            if not found and arg == sym:
                found = True
                continue
            rest.append(arg)
        if not found:
            return None
        quotient = sp.Mul(*rest)
        return None if quotient.has(sym) else quotient
    if isinstance(base, sp.Add):
        parts = []
        for term in base.args:
            part = _linear_multiple_quotient(term, sym, sp)
            if part is None:
                return None
            parts.append(part)
        return sp.Add(*parts)
    return None


def _remove_removable_power_denominators(expr):
    """Rewrite ``base(x)^n / x`` terms when ``base(x)`` is a linear multiple of
    ``x``.

    SymPy often differentiates Hill/power laws as ``n*base^n/base * dbase/dx``.
    For rate-law bases such as ``x/K`` or plain ``x``, that can print as
    ``(x/K)^n/x``. Mathematically the singularity is removable for the
    non-negative concentration domain BNGsim evaluates, but the raw emitted
    ExprTk/C form produces ``0/0`` at zero-valued initial conditions. This local
    rewrite avoids a global ``simplify()`` pass on large BioModels.

    The quotient ``base/sym`` is taken **structurally**
    (:func:`_linear_multiple_quotient`), never with ``sp.cancel`` (GH #96). That
    call was the entire cost of this function, and it was cost with nothing to
    show for it: it ran on every ``Pow`` base merely *containing* ``sym`` —
    including whole rational ``Add`` sub-expressions, where it performs full
    multivariate rational normalisation and then, because the result still
    contains ``sym``, the caller discards it and moves on. On
    ``BIOMD0000000217``'s ``∂vdead/∂l1`` one such call ran for minutes. The
    structural test answers the same question in microseconds, for exactly the
    set of bases the rewrite can actually use.
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
                if not (isinstance(factor, sp.Pow) and factor.exp == -1 and factor.base.is_Symbol):
                    continue

                sym = factor.base
                for power_i, power_factor in enumerate(factors):
                    if power_i == denom_i or not isinstance(power_factor, sp.Pow):
                        continue
                    base, exp = power_factor.base, power_factor.exp
                    if exp == -1 or not base.has(sym):
                        continue
                    quotient = _linear_multiple_quotient(base, sym, sp)
                    if quotient is None:
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


def _is_log_carrier(node, base, sp):
    """Is ``node``'s entire dependence on ``base`` confined to ``log`` arguments?

    That is the test for "this factor grows at most like a power of ``ln(base)``
    as ``base → 0+``". A *polynomial* in ``log(…base…)`` over ``base``-free
    coefficients is ``O(ln(base)^m)`` for some finite ``m``, which ``base^exp``
    with ``exp > 0`` dominates — so the product tends to ``0``.

    A factor that mentions ``base`` *outside* a ``log`` is not a carrier and is
    left strictly alone: ``1/(base + K)`` is finite at ``base == 0`` and needs no
    help, while ``1/base`` is a genuine pole whose divergence is not logarithmic
    and which a blanket ``0`` would silently swallow.

    The test is "blank the logarithms, then check what is left", in two parts,
    and both parts are load-bearing.

    *``base`` must not survive the blanking.* That has to be asked of the whole
    expression rather than by walking ``node.args``, because ``Mul``/``Add``
    match *sub-products*: ``thr/(alpha*k1thr)`` contains ``thr/k1thr`` while none
    of its three arguments does. A structural recursion bottoms out reporting "no
    ``base`` below this point" and calls an ordinary rational factor logarithmic
    — which misfires the guard onto expressions holding no logarithm at all, as
    BIOMD0000000066 and BIOMD0000000247 demonstrate.

    *What is left must be a polynomial in the blanked logarithm.* A polynomial of
    degree ``m`` is ``O(|ln base|^m)``, which is the growth the limit argument
    needs. "``base`` appears only under a ``log``" does **not** on its own imply
    it: ``exp`` inverts the logarithm, so ``exp(log(X)/g)`` is ``X^(1/g)`` — a
    power of ``base``, not a logarithm of one (BIOMD0000000613 writes a Hill
    exponent this way). Left unchecked, that reasoning would answer ``0`` for
    ``base^n·exp(-2·ln base)`` = ``base^(n-2)``, which is ``+inf`` at ``base = 0``
    for ``n < 2``. Non-polynomial dependence is declined instead: ``1/ln(base)``
    and ``1/(a + ln base)`` tend to ``0`` rather than diverging, so they never
    produced the NaN this guard exists to remove, and declining them costs a
    branch that was never needed.
    """
    if not node.has(sp.log) or not node.has(base):
        return False
    placeholder = sp.Dummy("_log")
    stripped = node.replace(lambda e: isinstance(e, sp.log), lambda e: placeholder)
    if stripped.has(base):
        return False
    return bool(stripped.is_polynomial(placeholder))


def _guard_exponent_log_at_zero(expr):
    """Guard ``base**exp · (logarithmic in base)`` against ``base == 0`` (GH #310).

    Differentiating a Hill/power law w.r.t. its **exponent** produces
    ``d/dn base^n = base^n · ln(base)``. On the non-negative concentration domain
    BNGsim evaluates, a species that is exactly zero — an unset initial condition
    is the common case — makes that ``0 · (-inf)`` = ``NaN`` in floating point,
    even though the limit exists and is ``0`` for every ``exp > 0``. One NaN in
    ``∂f/∂p`` is enough to poison that parameter's whole sensitivity column, or
    to defeat the corrector outright when it is the only column.

    The limit argument does not depend on the logarithm appearing exactly once.
    ``base^exp`` with ``exp > 0`` decays faster than *any* power of ``ln(base)``
    diverges, so for each ``Mul`` this collects the ``Pow(base, exp)`` factor
    together with every sibling factor whose dependence on ``base`` is confined to
    ``log`` arguments (:func:`_is_log_carrier`) and rewrites that
    sub-product to its limit at ``base == 0``:

    * ``exp`` a positive number → ``Piecewise((0, Eq(base, 0)), (raw, True))``,
      the branch decided at build time;
    * ``exp`` symbolic → ``Piecewise((0, Eq(base, 0) & (exp > 0)), (raw, True))``,
      the sign of the exponent settled at run time against its current value;
    * ``exp`` a non-positive number → left alone. ``base^exp·ln(base)`` has no
      finite limit there (``base^exp`` is already ``inf``/``1``), so a NaN is the
      honest answer and a blanket ``0`` would paper over a real singularity.

    Taking the whole logarithmic sub-product rather than one sibling ``log`` node
    is what closes GH #317. A single-``log`` pairing misses every shape where the
    divergence is spelled differently — ``ln(base)^2`` from a rate law that itself
    contains a logarithm, ``(a + ln base)`` from the product rule, ``ln(base/K)``
    whose argument is not structurally ``base`` — each of which a *first-order*
    ``sp.diff`` of ordinary SBML produces, and each of which then NaNs at
    ``base == 0`` with the pairing guard sitting right beside it.

    Siblings that mention ``base`` outside a ``log``, and siblings free of ``base``
    altogether, stay outside the ``Piecewise`` and multiply through as before, so a
    singularity that is not this one still reports itself.

    Only ``base == 0`` is caught, never ``base < 0``: a negative base under a
    fractional exponent is a genuine NaN and stays one.

    Applied *after* :func:`_remove_removable_power_denominators`, so the exponent
    tested is the one that survives that rewrite (``base^n/base`` → ``base^(n-1)``
    turns the run-time test into ``n - 1 > 0``, which is the correct condition for
    the rewritten term).
    """
    import sympy as sp
    from sympy.core.traversal import bottom_up

    def rewrite_mul(node):
        if not isinstance(node, sp.Mul):
            return node
        # Every carrier holds a logarithm, so a Mul without one can never be
        # rewritten. Answering that in one subtree walk keeps the scan below off
        # the overwhelming majority of nodes (2331 of 124132 corpus expressions
        # contain a logarithm at all).
        if not node.has(sp.log):
            return node

        factors = list(node.args)
        guarded = False
        changed = True
        while changed:
            changed = False
            for pow_i, power_factor in enumerate(factors):
                if not isinstance(power_factor, sp.Pow):
                    continue
                base = power_factor.base
                # ``2^x·log(2)`` and friends are constant and finite at every
                # point of the domain; nothing to guard.
                if base.is_number or base.is_positive:
                    continue
                # A base that is itself a logarithm cannot be tested for by
                # :func:`_is_log_carrier`, which blanks logarithms to look for it.
                if base.has(sp.log):
                    continue
                exp = power_factor.exp
                if exp.is_number:
                    if not exp.is_positive:
                        continue
                    cond = sp.Eq(base, 0)
                else:
                    cond = sp.And(sp.Eq(base, 0), exp > 0)

                carriers = [
                    i
                    for i, factor in enumerate(factors)
                    if i != pow_i and _is_log_carrier(factor, base, sp)
                ]
                if not carriers:
                    continue

                raw = sp.Mul(power_factor, *(factors[i] for i in carriers), evaluate=False)
                absorbed = set(carriers)
                factors[pow_i] = sp.Piecewise((sp.Integer(0), cond), (raw, True))
                factors = [f for i, f in enumerate(factors) if i not in absorbed]
                guarded = changed = True
                break

        if not guarded:
            return node
        return sp.Mul(*factors, evaluate=False)

    return bottom_up(expr, rewrite_mul)


# A rate law can only need the zero-base logarithm guard if it contains a
# logarithm, and a substring test settles that without touching sympy. That
# matters: over the 1644-model corpus only 2.09% of rate laws mention a
# logarithm at all, so gating on this turns a 21.4 s whole-corpus sympy pass
# into a 0.9 s one — and leaves the 97.9% that cannot be affected untouched,
# which is what keeps this off the derivation budget (#96, #97).
_LOG_CALL_RE = re.compile(r"\b(?:ln|log|log10|log2)\s*\(")


def guard_rate_law_text(text: str) -> str | None:
    """Guarded ExprTk spelling of one rate law, or ``None`` if it needs no guard.

    The single implementation of GH #333's rewrite, so the model path
    (:func:`guard_function_expressions`) and the ``.net`` codegen emitter, which
    builds its C from the file rather than from a loaded model, cannot drift
    apart — they are two RHS emitters for the same rate law and a guard on one
    only would make them disagree about the value at zero.

    Not a second copy of the guard itself: parsing to sympy and printing back
    through :func:`sympy_to_exprtk` applies :func:`_guard_exponent_log_at_zero`
    on the way out. Do **not** pre-apply the guard before calling that, or the
    emitter wraps an already-wrapped expression.

    ``None`` for anything this must not touch — no logarithm present, an existing
    conditional (which keeps the pass idempotent and leaves a modeller's own
    ``if()`` alone), a text that does not parse, or one that cannot be re-emitted.
    Turning a working model into a broken one is the only outcome not on the
    table, so every uncertain case declines.
    """
    if not _LOG_CALL_RE.search(text) or "if(" in text:
        return None
    try:
        import sympy  # noqa: F401
    except ImportError:
        return None
    sym = _exprtk_to_sympy(text)
    if sym is None:
        return None
    guarded = sympy_to_exprtk(sym)
    if guarded is None or guarded == text or "if(" not in guarded:
        # An unguarded round trip only re-spells an expression (``a*b`` → ``b*a``);
        # only a rewrite that actually introduced the branch is worth taking.
        return None
    return guarded


def guard_function_expressions(core) -> list[tuple[str, str, str]]:
    """Rewrite a model's rate laws to their limit at a zero logarithm base (GH #333).

    #310 and #317 guard ``base^exp · ln(base)`` at ``base == 0`` on the way out
    of the two *derivative* emitters. The rate law's own value never passes
    through them — ExprTk evaluates it straight from the model's text, and the
    codegen C emitter prints that same text — so a law containing ``ln(S)``
    answered ``NaN`` at ``S = 0`` while its own ``∂f/∂n`` answered the limit.
    One value apart, at ``S = 1e-30``, the identical model ran to completion.

    The rewrite itself is the emitter's, not a second implementation: parsing to
    sympy and printing back through :func:`sympy_to_exprtk` applies
    :func:`_guard_exponent_log_at_zero` on the way out, so the value path and the
    derivative path cannot drift apart by construction. Do **not** pre-apply the
    guard here — the emitter would then wrap an already-wrapped expression.

    The rewrite lands on the function's *evaluation* expression
    (``core.set_function_eval_expression``), never on its declared one. That
    separation is the whole design: the guard's ``S == 0`` branch is a
    state-dependent condition, and a rate law carrying one is read by the
    forward-sensitivity path as a state switch whose crossing time moves, which
    it refuses (the #52 / #150 machinery). Overwriting the declared law therefore
    bought a finite value at one point and lost the analytic sensitivity RHS for
    the entire run — measured, not theorised. Differentiation keeps reading the
    smooth declared law and the derivative emitters apply their own guard (#310,
    #317) to the result.

    Returns the ``(name, before, after)`` triples that changed, which is what the
    tests assert on. Silent no-op for a function that does not parse or cannot be
    re-emitted: this must never be able to turn a working model into a broken
    one, so anything the round trip cannot express is left exactly as declared.
    """
    changed: list[tuple[str, str, str]] = []
    try:
        names = list(core.function_names)
        texts = list(core.function_expressions)
    except Exception:
        return changed
    # The overwhelmingly common case, and the one worth settling before anything
    # else: no function mentions a logarithm, so nothing here can apply.
    candidates = [(n, t) for n, t in zip(names, texts, strict=False) if _LOG_CALL_RE.search(t)]
    if not candidates:
        return changed

    for name, text in candidates:
        guarded = guard_rate_law_text(text)
        if guarded is None:
            continue
        try:
            if core.set_function_eval_expression(name, guarded):
                changed.append((name, text, guarded))
        except Exception:
            continue
    return changed


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

        def _print_Relational(self, expr):
            # sympy's StrPrinter prints Eq/Ne in *function* form (``Eq(a, b)``),
            # which ExprTk does not know. Every relational it can produce has an
            # infix ExprTk spelling, so print them all that way (GH #310).
            lhs, rhs = (self._print(a) for a in expr.args)
            return f"({lhs} {expr.rel_op} {rhs})"

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
            # Undo the Python-keyword alias `_exprtk_to_sympy` applied on the way
            # in. The C twin of this emitter (`sympy_to_c`) resolves every symbol
            # through a callback keyed by alias and so never had to; this one
            # prints names straight through, so without the reverse step an
            # ExprTk derivative over a parameter named `def` / `lambda` / `is`
            # comes out reading `_BNG_KW_def`, ExprTk rejects it as an undefined
            # symbol, and the WHOLE model silently drops to the FD Jacobian —
            # visible only under BNGSIM_JAC_DEBUG. (Found via #170: putting the
            # storage divide back into a Functional law is what first made a
            # keyword-named *compartment* appear inside a differentiated rate.)
            # Guarded on the suffix actually being a keyword, so an ordinary
            # parameter that merely starts with the prefix is left alone.
            name = expr.name
            if name.startswith(_KW_ALIAS_PREFIX):
                stem = name[len(_KW_ALIAS_PREFIX) :]
                if stem in _PY_KEYWORD_PARAM_NAMES:
                    return stem
            return name

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
        expr = _guard_exponent_log_at_zero(_remove_removable_power_denominators(expr))
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

        def _print_Relational(self, expr):
            # As in the ExprTk printer: StrPrinter's ``Eq(a, b)`` / ``Ne(a, b)``
            # function form is not C, and every relational has an infix C
            # spelling (GH #310).
            lhs, rhs = (self._print(a) for a in expr.args)
            return f"({lhs} {expr.rel_op} {rhs})"

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


_c_printer_local = threading.local()


def _c_printer():
    printer = getattr(_c_printer_local, "printer", None)
    if printer is None:
        printer = _make_c_printer()()
        _c_printer_local.printer = printer
    return printer


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
        expr = _guard_exponent_log_at_zero(_remove_removable_power_denominators(expr))
    except Exception:
        return None
    printer = _c_printer()
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
#
# The third field marks the **conditional class** — the constructs that select a
# branch rather than bend a curve. Their dropped delta is a jump at a *crossing
# time*, which is the one flavour of dropped delta something else can supply:
# issue #48's switch-time jump does exactly that for a clock threshold. So
# ``allow_conditions=True`` lets a caller that has checked the crossings are
# compensated (GH #68's gate, :func:`bngsim._switch_sensitivity.
# uncompensated_condition_params`) accept them, while ``abs``/``min``/``max``/
# ``floor``/``ceil``/``round`` — kinks in the *state*, with no crossing time and
# nothing to compensate them — stay rejected for every caller.
_UNSUPPORTED_EXPR_CONSTRUCTS: list[tuple[re.Pattern, str, bool]] = [
    (re.compile(r"\bif\s*\("), "if() conditional", True),
    (re.compile(r"==|!=|<=|>=|<|>"), "comparison operator", True),
    (re.compile(r"&&|\|\||\band\b|\bor\b|\bnot\b"), "logical operator", True),
    (re.compile(r"(?<![=!<>])!(?!=)"), "logical-not operator", True),
    (re.compile(r"\babs\s*\("), "abs()", False),
    (re.compile(r"\bmin\s*\("), "min()", False),
    (re.compile(r"\bmax\s*\("), "max()", False),
    (re.compile(r"\bfloor\s*\("), "floor()", False),
    (re.compile(r"\bceil\s*\("), "ceil()", False),
    (re.compile(r"\b(?:round|rint|nint)\s*\("), "rounding function (round/rint/nint)", False),
]


def unsupported_expr_construct(body: str, *, allow_conditions: bool = False) -> str | None:
    """Return the reason a function body is unsupported for #198 output
    sensitivities, or ``None`` if no rejected construct is present.

    ``allow_conditions`` skips the conditional class (``if()``, comparisons,
    logicals). Only a caller that has separately established the crossings are
    compensated may set it — see the table above.
    """
    for pat, name, conditional in _UNSUPPORTED_EXPR_CONSTRUCTS:
        if conditional and allow_conditions:
            continue
        if pat.search(body):
            return name
    return None


def has_condition_construct(body: str) -> bool:
    """True when *body* carries a construct from the conditional class.

    The cheap pre-check that decides whether a caller needs to build the
    switch-condition scope at all — same table, so it cannot drift from what
    ``allow_conditions`` actually waives.
    """
    return any(
        pat.search(body) for pat, _name, conditional in _UNSUPPORTED_EXPR_CONSTRUCTS if conditional
    )


def differentiate_expression_output_partials(
    body: str,
    *,
    species_cref: dict[str, str],
    observable_cref: dict[str, str],
    param_cref: dict[str, str],
    function_cref: dict[str, str],
    deadline: float | None = None,
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

    ``deadline`` (GH #97) is a ``time.perf_counter()`` stamp bounding the caller's
    whole build-time analysis. It is checked on entry — parsing the body is itself
    unbounded work — and again before each ``sp.diff``, so one pathological
    function overshoots by at most one partial. Expiry raises
    :class:`_DerivationBudgetExceeded` rather than returning a reason: it is a
    property of the *build*, not of this function, and the caller has to know the
    difference (it marks every remaining function unsupported instead of
    attributing the failure to this body). ``None`` (the default) is unbounded.
    """
    reason = unsupported_expr_construct(body)
    if reason is not None:
        return None, f"uses unsupported construct: {reason}"

    try:
        import sympy as sp
    except ImportError:  # pragma: no cover - sympy is a hard dep of codegen sens
        return None, "sympy is required for expression output sensitivities"

    if deadline is not None and time.perf_counter() > deadline:
        raise _DerivationBudgetExceeded

    sym_expr = _exprtk_to_sympy(body)
    if sym_expr is None:
        return None, "could not parse expression for differentiation"

    # Keyword-named params get a safe alias in _exprtk_to_sympy; round-trip it
    # here so the sympy symbol name resolves back to the right kind / C ref.
    # Keywords only, deliberately: the resolve callback built below is what
    # `sympy_to_c` prints through, so no symbol name reaches a C printer that
    # would rename a reserved word (GH #108).
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
    # Sorted, not set order: this dict's insertion order reaches the emitted C
    # through the caller's per-kind partial maps, so iterating the `free` *set*
    # made the #198 evaluator's source depend on PYTHONHASHSEED — a different
    # artifact hash every process, and a content-addressed `.so` that can never
    # hit. Measured on 121 of the 585 .net corpus models. Exactly the defect #68
    # fixed in `differentiate_rate_law`; it survived here because a function with
    # a single referenced symbol cannot show it.
    for alias in sorted(free):
        if alias in ignored:
            continue
        if deadline is not None and time.perf_counter() > deadline:
            # GH #97: bail mid-function so one body referencing many symbols
            # cannot run the whole analysis past its budget in a single call.
            raise _DerivationBudgetExceeded
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
    species_volume_sym: dict[int, str] | None = None,
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
    ``species_volume_sym`` (issue #170 stage 2): ``species_idx0 → the name of the
    parameter V_j IS``, for species whose compartment size is writable. Those keep
    V_j as a **symbol** rather than folding it into the coefficient: the fold happens
    once at Simulator-construction time, and this derivative is not only a
    preconditioner — it is the ``J·yS`` half of the forward-sensitivity RHS, so a
    volume written afterwards leaves a *wrong sensitivity* (measured at 100% on a
    400x write), not merely a slower solve. Absent/empty ⇒ every coefficient is
    folded exactly as before, so a model with no writable compartment size emits
    byte-identical derivatives.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    dd = differentiate_rate_law(rate_expr, func_map, set(obs_groups), constant_names, deadline)
    if dd is None:
        return None

    import math

    vol_sym = species_volume_sym or {}
    per_species: dict = {}
    for obs_name, dexpr in dd.items():
        for sp_idx, factor in obs_groups[obs_name]:
            amount_valued, vol = species_amount.get(sp_idx, (False, 1.0))
            vname = vol_sym.get(sp_idx) if amount_valued else None
            if vname:
                if float(factor) == 0.0:
                    continue
                # `differentiate_rate_law` aliased any Python-keyword identifier on
                # the way in, so this symbol has to carry the alias too: `sympy_to_c`
                # resolves by aliased name and `sympy_to_exprtk` undoes it on print.
                alias = _alias_keyword_param(vname) if vname in _PY_KEYWORD_PARAM_NAMES else vname
                term = sp.Float(float(factor)) * sp.Symbol(alias) * dexpr
                per_species[sp_idx] = per_species.get(sp_idx, sp.Integer(0)) + term
                continue
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
    species_volume_sym: dict[int, str] | None = None,
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
        rate_expr, func_map, obs_groups, species_amount, constant_names, species_volume_sym
    )
    if native is not None:
        return native

    terms = build_per_species_sympy(
        rate_expr,
        func_map,
        obs_groups,
        species_amount,
        constant_names,
        deadline,
        species_volume_sym,
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


def _native_per_species_terms(
    rate_expr, func_map, obs_groups, species_amount, constant_names, species_volume_sym=None
):
    """Native per-species derivatives as ExprTk strings (SBML path), or
    ``None``."""
    from bngsim import _saturable_jacobian as _sat

    terms = _sat.build_per_species_native(
        rate_expr, func_map, obs_groups, species_amount, constant_names, species_volume_sym
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
    rate_expr,
    func_map,
    obs_groups,
    species_amount,
    constant_names,
    resolve_symbol,
    deadline=None,
    species_volume_sym=None,
) -> list[tuple[int, str]] | None:
    """Codegen per-species path: ``[(species_idx0, c_str)]`` or ``None``.

    Tries the native saturable family first (no SymPy), then the SymPy path
    (:func:`build_per_species_sympy` + :func:`sympy_to_c`). ``[]`` is a *success*
    (constant-rate ⇒ zero column). A native success whose C emission fails (an
    unresolvable symbol) falls through to SymPy rather than declining outright.

    ``deadline`` (GH #90) is forwarded to the SymPy fallback only — the native
    path runs no SymPy and needs no bound. Passed by the sensitivity RHS, whose
    ``J·yS`` half re-derives this on its own build; ``None`` (the Jacobian
    emitter's call) is the unbudgeted behaviour."""
    from bngsim import _saturable_jacobian as _sat

    native = _sat.build_per_species_native(
        rate_expr, func_map, obs_groups, species_amount, constant_names, species_volume_sym
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
        rate_expr,
        func_map,
        obs_groups,
        species_amount,
        constant_names,
        deadline,
        species_volume_sym,
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
    rate_expr, func_map, observable_names, constant_names, resolve_symbol, deadline=None
) -> list[tuple[str, str]] | None:
    """Codegen per-observable path: ordered ``[(observable_name, c_str)]`` or
    ``None``. Native saturable family first (no SymPy), then SymPy.

    ``deadline`` (GH #90) is forwarded to the SymPy fallback only; see
    :func:`build_per_species_c`."""
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

    dd = differentiate_rate_law(rate_expr, func_map, observable_names, constant_names, deadline)
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
#
# That bound is real but not tight, because "one derivative" is not a bounded
# quantity: the deadline can only be tested between sp.diff calls, never inside
# one. Measured against the 20 s default, #245 found the corpus overshooting by
# 1.0x on MODEL1006230053 (20.2 s) and MODEL1006230090 (20.1 s) — and by **6.9x on
# BIOMD0000000385**, whose five rate laws inline to a single 47 k-token expression
# that took 138 s to reach the first deadline check and then declined anyway.
#
# GH #250 closed that without touching the deadline, because subdividing the
# derivation could not have closed it: recursing 385's rate law to 117
# deadline-checkable steps still leaves two dAbs at 62.4 s and 34.8 s, every other
# step under 0.04 s. Abs is an atomic leaf and sp.diff on it is one call. What the
# profile showed instead is that those 138 s were spent deriving something the
# emitter can never print, so the answer was known before the derivation started —
# see _NONDIFFERENTIABLE_EMITTER_FUNCS. 385 now declines in 0.8 s, and the worst
# overshoot left in the corpus is 1.0x.

# Base wall-clock budget (seconds) for the build-time derivation on a *small*
# model.
#
# Choosing the value: the budget must (a) exceed the derivation time of every model
# the analytical Jacobian actually buys something on, or it regresses that model,
# and (b) stay below the pathological derivations it exists to cut off. #95 read
# (a) as "the solve *fails* on FD" and found the two populations 3.4x apart —
# BIOMD0000000457 needing ~12 s to derive, the cheapest waste (BIOMD0000000496) at
# ~41 s — and put 20 s in the middle.
#
# Both endpoints have since moved by more than the gap between them, so the value
# was re-derived from the corpus rather than adjusted (issues #249 and #245; the
# sweeps behind every number here, and the floor the value is held to, are in
# python/tests/test_sbml_jacobian_budget_biomd496.py). Unbudgeted derivation over
# 1319 rr_parity ODE models, then analytical / FD-forced / RoadRunner at the
# harness's 1e-9/1e-12 tolerance in fresh processes. **Solve times are medians of
# repeats on a warm codegen cache** — a single cold-cache sample reports
# FD/analytical ratios wrong by up to 9x in either direction, which is what made
# 496 read as a model worth protecting and MODEL1603150001 read as one that is not.
#
#   * (a) is no longer a *correctness* constraint at all (issue #249): forcing FD
#     on all 1,218 attaching models finds none that needs the analytical Jacobian
#     to solve. BIOMD0000000457, the model #95 sized the budget against, is the one
#     apparent exception and is not one — its FD solve fails at exactly rtol 1e-9 on
#     x86_64 and nowhere else, succeeding on both neighbouring tolerances and on
#     arm64 at all of them. That is an arithmetic knife-edge, not a model property,
#     and it is unpinnable in either direction (issue #245).
#   * (a) survives as a *cost* constraint, which is what sets the floor:
#       - BIOMD0000000608 solves 4.2x faster with the analytical Jacobian
#         (0.015 s vs 0.065 s) and derives in **4.76 s**. FD *works* here, so the
#         #95 screen could not see it; it is nonetheless the most expensive
#         derivation in the corpus that pays for itself, and it sets the floor.
#         Not a lone fixture either — MODEL1603150001 (3.0x), MODEL1601050000
#         (2.7x), MODEL1602080000 (1.7x) and MODEL1504130000 (1.4x) derive in
#         2.2-3.0 s and pay too. Every one of them reads as a *loss* (0.2-0.3x) in
#         a single cold-cache sample, which is the same artifact from the other
#         side: whichever mode runs first absorbs the codegen warm-up.
#   * (b) the cheapest derivation that does NOT pay for itself is BIOMD0000000628
#     at 59.3 s, whose analytical solve is if anything slower than its FD one. Above
#     it: MODEL1006230049/077/053 at 85-133 s, BIOMD0000000385 (118 s, and it
#     *declines* after paying — waste at any budget), and MODEL1006230090, whose
#     unbudgeted derivation ran past a 400 s probe cap without a verdict.
#   * Between 4.76 s and 59.3 s there is nothing to get right. 496 (10.9 s) and 497
#     (11.1 s) derive to completion on this default and measure 1.02x and 1.25x —
#     the waste #245 reported, and real, but ~22 s of build across the whole corpus.
#
# So the window is (4.76 s, 59.3 s): **12.5x wide, against #95's 3.4x**. The gap did
# not close, it moved and opened. 20 s sits 4.2x above the floor and 3.0x below the
# ceiling — deliberately above the 16.8 s geometric centre, because the two ways to
# be wrong do not cost the same: too high spends build seconds once, too low buys a
# permanently slower solve. That asymmetry is also what makes the value survive the
# machine spread
# (~3.3x between two development machines, ratios travel and seconds do not): on a
# machine 3.3x slower, 608 derives in 15.7 s and is still kept; on one 3.3x faster,
# 628 derives in 18 s and is kept, which costs 18 s of build and no correctness.
# Both models that set the window are small — 608 at 52 species, 628 at 139 — so on
# the model-size-scaled budget below they see this base and not the scaled value,
# which is what makes the base the thing being reasoned about here rather than a
# lower bound on something else. Models that derive quickly (the
# vast majority, << 1 s) are unaffected. Override / disable with
# BNGSIM_JAC_DERIV_BUDGET_S (<= 0 or "inf"/"none" → unbounded, the pre-#95
# behavior; raise it on a slow machine if a model whose derivation pays for itself
# logs a fallback — no corpus model needs it to *solve*, so this is a speed knob).
#
# Why this stays keyed on wall-clock, which is #245's other half. Seconds do not
# travel between machines, so the obvious repair is to key the cut-off on something
# that does — derivation cost per reaction, inlined rate-law size, a #97-style step
# count. Measured over the same corpus, each of those predicts derivation cost far
# worse than the machine spread it would be replacing: per-inlined-token cost runs
# 2.5-2351 us/token over the 256 models with >= 1000 tokens (**923x** end to end,
# 142x from the median up), and BIOMD0000000385 and BIOMD0000000246 have the same
# largest inlined rate law (~47 k tokens) and derive 19x apart. Best log-log
# correlation of any static key is 0.685 (total inlined tokens, the metric behind
# every number above). Concretely: the loosest such budget that cuts
# nothing it cuts today needs 8 ms/token, which hands MODEL1006230049 5838 s —
# 290x what wall-clock gives it. A key that travels is not the same as a key that
# predicts, and this one does neither well enough to replace the clock. What #187
# and #97 already do is the sound version of the idea and stays: keep the clock as
# the mechanism, and let a size that travels scale the *allowance* upward.
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


# GH #90: the sensitivity ∂f/∂p derivation (``_codegen._functional_dfdp_terms``
# and the derived-parameter chain rules on the same build) gets its own budget,
# resolved by _sens_derivation_budget_s below. Same base, same slope, same
# override grammar as the Jacobian's — deliberately, so the two policies cannot
# drift — but its own env var, and one substantive difference: it never goes
# unbounded by size.
#
# The difference is about the FALLBACK, not the derivation. _FD_NONVIABLE_SPECIES
# exists because past that size there is nothing to fall back TO: an FD Jacobian
# needs ~n_species RHS evals per Newton setup and simply does not converge, so
# cutting the derivation off breaks the solve outright and the analytical Jacobian
# is mandatory. The sensitivity path has no such cliff — declining hands the
# columns to CVODES' internal difference quotient, which is what every Functional
# model used before #55 and is correct at every scale (measured 9-37x more
# expensive per column, not divergent). So the sensitivity budget stays finite
# everywhere: a build that would otherwise appear to HANG instead declines, says
# so, and solves. Anyone who would rather wait sets BNGSIM_SENS_DERIV_BUDGET_S.
_JAC_BUDGET_ENV = "BNGSIM_JAC_DERIV_BUDGET_S"
_SENS_BUDGET_ENV = "BNGSIM_SENS_DERIV_BUDGET_S"

# GH #97: per-derivation-step allowance for the #198 output-sens analysis, the
# third budgeted phase on a sensitivity build. A step is one expression parsed or
# one ∂/∂symbol taken — see _codegen._output_sens_derivation_steps. 50 ms/step;
# the base dominates below 400 steps (= base / slope), so small models are
# unaffected. See _output_sens_derivation_budget_s for why this path is keyed on
# the step count rather than on species count, and where the slope comes from.
_BUDGET_PER_STEP_S = 0.05


class _DerivationBudgetExceeded(Exception):
    """Internal signal: the build-time symbolic derivation passed its wall-clock
    budget. Caught by :func:`attach_functional_jacobian`, which logs the fallback
    and leaves the model on the finite-difference Jacobian (GH #95), and by
    ``_codegen.generate_sens_from_model`` / ``generate_sens_rhs_c``, which decline
    the analytic sensitivity RHS and leave the model on CVODES' internal
    difference quotient (GH #90)."""


def _budget_env_override(env_var: str, default: float | None) -> float | None:
    """Apply an explicit ``env_var`` budget over a size-derived ``default``.

    The override grammar, shared by every derivation budget: an absolute number of
    seconds, or ``inf``/``none``/``off``/``0`` for unbounded (the pre-#95 and
    documented genome-scale workaround). A non-positive or non-finite value also
    disables the budget; a value that does not parse falls through to ``default``,
    so a typo degrades to the policy rather than to no budget at all.
    """
    raw = os.environ.get(env_var)
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

    return _budget_env_override(_JAC_BUDGET_ENV, default)


def _sens_derivation_budget_s(n_species: int = 0) -> float | None:
    """Resolve the build-time budget for the *sensitivity* ∂f/∂p derivation in
    seconds, or ``None`` for unbounded (GH #90).

    The sensitivity counterpart of the Jacobian's :func:`_derivation_budget_s`,
    sharing its base, its per-species slope and its override grammar — but read
    from ``BNGSIM_SENS_DERIV_BUDGET_S``, and **never unbounded by species count**.
    See the block comment above ``_SENS_BUDGET_ENV`` for why
    ``_FD_NONVIABLE_SPECIES`` has no counterpart here: the sensitivity fallback
    (CVODES' internal difference quotient) stays viable at every model size, so
    there is never a reason to let this derivation run without a bound.

    The two budgets are independent: setting ``BNGSIM_JAC_DERIV_BUDGET_S=inf`` to
    keep a genome-scale model's analytical Jacobian does not also uncap this one,
    because they buy different things and fall back to different paths.
    """
    default = max(_DEFAULT_DERIVATION_BUDGET_S, _BUDGET_PER_SPECIES_S * n_species)
    return _budget_env_override(_SENS_BUDGET_ENV, default)


def _output_sens_derivation_budget_s(n_steps: int = 0) -> float | None:
    """Resolve the build-time budget for the #198 output-sensitivity
    ``d func/dθ`` derivation in seconds, or ``None`` for unbounded (GH #97).

    The third phase on a sensitivity build, after the analytical Jacobian and
    ∂f/∂p: ``_codegen._analyze_output_sens`` parses every user function and every
    derived-parameter expression and takes one derivative per symbol each
    references. It reads the *same* env var as ∂f/∂p — one knob for one build —
    but resolves its own deadline, so neither phase can starve the other (see
    ``_codegen._output_sens_derivation_deadline``).

    **Keyed on the derivation-step count, not on species**, which is where this
    policy departs from the other two. Same shape as #187 —
    ``max(base, slope × size)`` — over the size that actually drives the cost here.
    A model can carry thousands of global functions on a few hundred species
    (``MODEL1112100000``: 1265 species, 3633 functions, 14532 steps), so a
    species-scaled curve is loose exactly where the work is, and
    ``_codegen._output_sens_derivation_steps`` counts the steps from the reference
    graph's own tokens before any sympy runs.

    Measured over the BioModels SBML corpus on the current emitters, which is the
    only population worth sizing against and a different one from #97's issue
    text: that was written before #96's printer fix, which took
    ``BIOMD0000000217`` from 900 s to 1.2 s. ``BIOMD0000000063`` — the model the
    issue names as the worst that completes, at 10.2 s on nine species, and the
    evidence for its "expression-driven, not size-driven" reading — now derives in
    0.86 s. With those outliers gone what is left is size-driven:

    | model            | species | steps | analysis | ms/step | headroom |
    |------------------|--------:|------:|---------:|--------:|---------:|
    | MODEL1603150001  |    6047 | 15568 |   85.7 s |     5.5 |     9.1x |
    | MODEL1504130000  |    5063 | 14880 |   67.2 s |     4.5 |    11.1x |
    | MODEL1112100000  |    1265 | 14532 |   13.3 s |     0.9 |    54.7x |
    | BIOMD0000000497  |     295 |  3986 |   19.2 s |     4.8 |    10.4x |
    | BIOMD0000000470  |     786 |  4756 |   16.9 s |     3.6 |    14.1x |
    | BIOMD0000000247  |      31 |   249 |    1.4 s |     5.7 |    14.1x |

    So the slope is ~9x the worst rate anything real derives at, and the base ~14x
    the worst model below the knee: a model that derives at a *normal* rate is
    never cut off however large it is, and what expires is an anomalous per-step
    cost — which is what a hang looks like from here. A flat budget was the other
    candidate and is measurably wrong: 20 s cuts the first two models, which today
    complete and emit.

    Expiry does not decline anything. Unlike ∂f/∂p — one callback for every
    column, so one undifferentiable rate law takes the whole model — output
    sensitivities are per function, and a function that ran out of clock is marked
    ``unsupported`` exactly as an undifferentiable one is: the emitted C carries a
    NaN sentinel and the ``Result`` raises the reason at selection time. Every
    function derived before the deadline keeps working. That graceful degradation
    is also why this budget does not need #187's ``_FD_NONVIABLE_SPECIES`` gate:
    there is no model size at which an expiry here breaks the solve.
    """
    default = max(_DEFAULT_DERIVATION_BUDGET_S, _BUDGET_PER_STEP_S * n_steps)
    return _budget_env_override(_SENS_BUDGET_ENV, default)


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
    # (#170 stage 2) The compartment-size parameter each amount factor IS, for the
    # species whose volume is writable — carried as a symbol instead of folded, so a
    # volume written after this derivation still moves J (and with it the J*yS half
    # of the sensitivity RHS). Empty for every model with no writable compartment
    # size, which keeps their ExprTk derivative text unchanged.
    species_volume_sym = {i: n for i, n in enumerate(ctx.get("species_volume_param") or ()) if n}
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
                    rate_expr,
                    func_map,
                    obs_groups,
                    species_meta,
                    constants,
                    deadline,
                    species_volume_sym,
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
