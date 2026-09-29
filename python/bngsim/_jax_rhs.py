"""bngsim._jax_rhs — JAX-based ODE RHS and AD Jacobian for BNGsim.

Generates a JAX-traced RHS function from a built model, then uses
``jax.jacfwd`` to compute exact dense Jacobians via automatic
differentiation. This provides exact Jacobians for ALL rate law types
(Elementary, Functional, MichaelisMenten) without manual derivatives.

Architecture:
  1. generate_jax_rhs(model) -> Callable: build a JAX RHS from codegen_data()
  2. generate_jax_jacobian(model) -> Callable: jacfwd(rhs) wrapper
  3. screen_for_discontinuities(model) -> list: Check for floor/ceil/etc.

Each takes a :class:`bngsim.Model` or the path of a ``.net`` file, which is
loaded with :meth:`bngsim.Model.from_net`. Until issue #803 step 4 the path was
re-read here by codegen's private ``.net`` parser, a second reading of the file
that disagreed with the loader's (#608, #784) — and the JAX RHS was wrong on 249
of 767 corpus networks, most of them because a synthesis reaction's null
reactant multiplied the rate by the last species.

The JAX Jacobian is fed back to CVODE via the existing user-Jacobian
callback mechanism (dense matrix). JAX runs on CPU only.

Optional dependency: ``pip install bngsim[jax]``.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import os
import re
from typing import Any

from bngsim._codegen import (
    _BUILTIN_CONSTANT_VALUES,
    _RATEOF_PREFIX,
    CodegenDeclined,
    _find_close_paren_strict,
    _multiplicity,
    _normalize_exprtk_operators,
    _split_top_level_commas,
    _topological_function_order,
)

logger = logging.getLogger("bngsim")

# ─── Availability check ─────────────────────────────────────────────────────

_JAX_AVAILABLE: bool | None = None


def jax_available() -> bool:
    """Check if JAX is importable (cached).

    Also enables 64-bit precision (required for CVODE compatibility).
    """
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is None:
        try:
            import jax

            # Enable 64-bit precision — CVODE uses double, and float32
            # Jacobians would corrupt Newton convergence.
            jax.config.update("jax_enable_x64", True)
            import jax.numpy  # noqa: F401

            _JAX_AVAILABLE = True
        except ImportError:
            _JAX_AVAILABLE = False
    return _JAX_AVAILABLE


# ─── Discontinuity screening ────────────────────────────────────────────────

# Functions that produce useless AD gradients (piecewise constant).
_DISCONTINUOUS_FUNCS = {"floor", "ceil", "rint", "round", "Heaviside"}


def screen_for_discontinuities(model: Any) -> list[str]:
    """Scan function expressions for constructs that defeat AD.

    Returns a list of problematic function names found, or empty list
    if the model is safe for JAX AD. ``model`` is a :class:`bngsim.Model` or a
    ``.net`` path (see the module docstring).
    """
    problems = []
    for f in _codegen_data(_as_model(model))["functions"]:
        for disc_fn in _DISCONTINUOUS_FUNCS:
            if re.search(rf"\b{disc_fn}\b", f["expression"]):
                problems.append(f"function '{f['name']}' uses {disc_fn}()")
    return problems


# ─── The built model, and what the JAX RHS implements of it ─────────────────


def _as_model(model: Any) -> Any:
    """A built model: the argument itself, or the ``.net`` file at that path loaded
    with :meth:`bngsim.Model.from_net` -- never a second reading of the file."""
    if isinstance(model, (str, os.PathLike)):
        from bngsim._model import Model

        return Model.from_net(os.fspath(model))
    return model


def _codegen_data(model: Any) -> dict:
    return (model._core if hasattr(model, "_core") else model).codegen_data()


def _refuse_unsupported(data: dict) -> None:
    """Raise ``ValueError`` naming the first construct the JAX RHS does not implement.

    It implements what a ``.net`` model contains -- measured, not assumed: over the
    774 corpus networks of up to 1,000 reactions, every reaction and species field
    of ``codegen_data()`` sits at one value except the rate-law type and ``fixed``
    (issue #803). That is Elementary, Functional and Michaelis-Menten reactions
    with a stat factor and the reactant factor applied, fixed species, observables
    over unit-volume species, and functions of parameters, observables, species and
    other functions. Everything else the engine's RHS handles -- compartment
    volumes, amount-valued species, per-species volume scaling, rate rules, rateOf,
    table functions -- is refused here rather than approximated, whatever the
    model was loaded from.
    """

    def refuse(what: str) -> None:
        raise ValueError(
            f"the JAX RHS (jacobian='jax', run_diffrax) does not implement {what}. "
            "Use jacobian='auto' (the default), which evaluates this model through the "
            "engine instead."
        )

    for s in data["species"]:
        name = s["name"]
        if s.get("amount_valued", False):
            refuse(f"an amount-valued species ({name})")
        if float(s.get("volume_factor", 1.0)) != 1.0:
            refuse(f"a species in a compartment of size other than 1 ({name})")
        if int(s.get("volume_param_idx0", -1)) >= 0 or int(s.get("ode_live_volume_idx0", -1)) >= 0:
            refuse(f"a species whose compartment size is a parameter or a state ({name})")
    function_names = {f["name"] for f in data["functions"]}
    for r in data["reactions"]:
        if r["type"] not in ("elementary", "functional", "mm"):
            refuse(f"a reaction of type {r['type']!r}")
        if r["type"] == "mm":
            # A rate constant that is also a function's name (the #266 shape): the
            # engine reads the live function there, and the JAX RHS the parameter row.
            for i in r["rate_param_indices"]:
                name = data["parameters"][int(i)]["name"]
                if name in function_names:
                    refuse(f"a Michaelis-Menten rate constant that a function defines ({name})")
        if not r.get("apply_species_factor", True):
            refuse("a reaction whose kinetic law carries its own reactant factor (SBML)")
        if r.get("per_species_volume_scaling", False):
            refuse("a cross-compartment reaction (per-species volume scaling)")
        if r.get("is_rate_rule_ode", False):
            refuse("a rate rule")
    if data.get("table_functions"):
        refuse("a table function (tfun)")
    for f in data["functions"]:
        if _RATEOF_PREFIX in f["expression"]:
            refuse(f"rateOf (function {f['name']})")
        if "%" in f["expression"]:
            # ExprTk's % is C's fmod; Python's, which the translated text would
            # use, is floor-mod, and the two differ on a negative operand.
            refuse(f"the % operator (function {f['name']})")


# ─── Expression translator (.net expression -> JAX/Python) ──────────────────

# Stand-in for `time()` while the model-name substitutions run, so no parameter
# or observable can rewrite the clock (issue #659). Not a valid BNG identifier,
# so it cannot collide with a model name.
_CLOCK_SYM = "__bngsim_clock__"

# A zero-arg call — an observable, parameter or any other scalar written as
# `name()` (issue #28). `time()` is the only zero-argument built-in and is
# rewritten before this runs.
_EMPTY_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\)")

# The engine's math functions and their jax.numpy spelling. Applied in a single
# pass (see _translate_expr_jax), so no entry's replacement can be rewritten by
# another entry's pattern.
_JAX_MATH_FUNCS: dict[str, str] = {
    "ln": "jnp.log",
    "log": "jnp.log",
    "sqrt": "jnp.sqrt",
    "exp": "jnp.exp",
    "sin": "jnp.sin",
    "cos": "jnp.cos",
    "tan": "jnp.tan",
    "asin": "jnp.arcsin",
    "acos": "jnp.arccos",
    "atan": "jnp.arctan",
    "abs": "jnp.abs",
    "min": "jnp.minimum",
    "max": "jnp.maximum",
    "pow": "jnp.power",
    # Not jnp.round, which rounds a half to EVEN — see _jax_round below.
    "rint": "__bngsim_rint__",
    "round": "__bngsim_round__",
    "trunc": "jnp.trunc",
    "floor": "jnp.floor",
    "ceil": "jnp.ceil",
    # GH #565 — the engine's reserved list (reserved_names(), src/expression.cpp)
    # carries these too, and jax.numpy spells every one of them directly. Their
    # absence was not a decision: `log10` never matched the `log` rule (`1` is a
    # word character, so the boundary held), so it reached the generated source
    # untranslated and raised NameError inside the RHS at solve time, on a model
    # the ODE backend and jacobian="auto" both handle.
    "log10": "jnp.log10",
    "log2": "jnp.log2",
    "sinh": "jnp.sinh",
    "cosh": "jnp.cosh",
    "tanh": "jnp.tanh",
    "asinh": "jnp.arcsinh",
    "acosh": "jnp.arccosh",
    "atanh": "jnp.arctanh",
    "sign": "jnp.sign",
    "sgn": "jnp.sign",
    # ExprTk takes `not` only as a call, `not(x)`. Python's `not` asks a traced
    # value for bool() under jax.jit, and ExprTk reads a number as its truth
    # value, which jnp.logical_not does too (GH #579).
    "not": "jnp.logical_not",
}

# The engine's reserved constants, bound on every expression it compiles. Same
# story as the functions above: nothing substitutes them, so `_pi` reached the
# RHS as a bare name and raised NameError there. The physical constants have no
# jax.numpy name, so they are emitted as the values the engine binds.
_JAX_CONSTANTS: dict[str, str] = {
    **{name: repr(value) for name, value in _BUILTIN_CONSTANT_VALUES.items()},
    "_pi": "jnp.pi",
    "_e": "jnp.e",
    # The bare spelling of the clock. `time()` is parked above as _CLOCK_SYM;
    # the engine accepts `time` without parentheses too, and _BUILTIN_IDENT_MAP
    # (the C path) maps it the same way. (`t()` is not the clock — issue #659.)
    "time": "t",
}

_JAX_NAMES: dict[str, str] = {**_JAX_MATH_FUNCS, **_JAX_CONSTANTS}


def _jax_inv_hill_power(x: Any, n: Any) -> Any:
    """Evaluate ``1 / (1 + x**n)`` without overflowing for positive ``x``.

    JAX's derivative of the direct power can become ``inf/inf`` even while the
    Hill fraction itself has cleanly saturated to zero (issue #838). For
    positive bases, with ``z = n*log(x) = log(x**n)``, the fraction is
    ``1/(1 + e^z)`` for ``z <= 0`` and ``e^-z/(1 + e^-z)`` for ``z > 0``. Each
    exponential is at most 1, so neither the value nor its tangent overflows,
    and neither form subtracts from 1: ``sigmoid(-z)`` has the same value, but
    its tangent ``s*(1 - s)`` cancels to 0 once ``x**n`` drops below ~1e-16,
    where the true derivative ``-n*x**(n-1)`` can be huge (x = 1e-300, n = 0.5).
    Keep the direct expression for non-positive bases, where logarithms would
    change the real-valued behavior of integer powers. Every branch reads masked
    inputs, ``n`` included, so the branch not taken cannot leak a NaN tangent or
    cotangent through ``where`` (``0**n`` is inf for a negative ``n``), and this
    stays elementwise under ``vmap``, unlike ``lax.cond``.
    """
    import jax.numpy as jnp

    pos = x > 0.0
    z = n * jnp.log(jnp.where(pos, x, 1.0))
    low = z <= 0.0
    p = jnp.exp(jnp.where(low, z, 0.0))  # x**n, at most 1
    q = jnp.exp(-jnp.where(low, 0.0, z))  # x**-n, below 1
    positive = jnp.where(low, 1.0 / (1.0 + p), q / (1.0 + q))
    direct = 1.0 / (1.0 + jnp.power(jnp.where(pos, 0.0, x), jnp.where(pos, 1.0, n)))
    return jnp.where(pos, positive, direct)


# The engine's two roundings, neither of which jax.numpy spells. jnp.round
# rounds a half to EVEN, so mapping either name onto it moved every exact half:
# round(2.5) came out 2 here and 3 in the engine. They are not each other
# either. `rint` is BNG's floor(x + 0.5) (expr_compat::rint in
# src/expression.cpp, issue #771), so a half always rounds up: rint(-2.5) is -2.
# `round` is ExprTk's floor(x + 0.5), or ceil(x - 0.5) below zero, so a
# negative half rounds away from zero: round(-2.5) is -3. They agree at and
# above zero. The RHS namespace binds them as helpers.
def _jax_round(x: Any) -> Any:
    """ExprTk's ``round``: ``floor(x + 0.5)``, or ``ceil(x - 0.5)`` below zero."""
    import jax.numpy as jnp

    return jnp.where(x < 0, jnp.ceil(x - 0.5), jnp.floor(x + 0.5))


def _jax_rint(x: Any) -> Any:
    """The engine's ``rint``: BNG's ``floor(x + 0.5)``, a half rounded up."""
    import jax.numpy as jnp

    return jnp.floor(x + 0.5)


_JAX_HELPERS: dict[str, Any] = {
    "__bngsim_round__": _jax_round,
    "__bngsim_rint__": _jax_rint,
    "__bngsim_inv_hill_power__": _jax_inv_hill_power,
}

# What the generated RHS binds around the eval (see generate_jax_rhs), plus the
# Python keywords an expression may legitimately contain. Every other bare name
# left after translation is a NameError waiting for solve time.
_JAX_EVAL_NAMES = frozenset({"jnp", "t", "params", "obs", "y", *_JAX_HELPERS})
# Not `and`/`or`/`not`: _regroup_for_python and the table turn those into
# jnp.logical_* calls, and one that survived would ask a tracer for bool()
# under jax.jit (GH #579).
_PY_KEYWORDS = frozenset({"if", "else", "True", "False", "None"})

# An identifier that is not an attribute access: `jnp.log` is one name, not two.
_BARE_IDENT_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)")

# Quoted text is data, not a name — a table function's file name would
# otherwise be read as a pile of undefined identifiers.
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_JAX_NUMERIC_LITERAL_RE = re.compile(
    r"(?P<string>'[^']*'|\"[^\"]*\")|"
    r"(?P<number>(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?(?![\w.]))"
)


def _jax_numeric_literals_as_floats(expr: str) -> str:
    """Make integer literals floating-point without touching strings or indices.

    The JAX expression evaluator mixes translated model literals with JAX
    arrays. A literal-only subexpression such as ``1500^6`` is evaluated by
    Python as an arbitrary-precision ``int`` before it reaches JAX; JAX then
    raises when that value cannot be represented as int64. The engine evaluates
    all model numbers as doubles, so spell integer tokens as floats here too.
    """

    def replace(match: re.Match[str]) -> str:
        string = match.group("string")
        if string is not None:
            return string
        number = match.group("number")
        assert number is not None
        return number + ".0" if re.fullmatch(r"\d+", number) else number

    return _JAX_NUMERIC_LITERAL_RE.sub(replace, expr)


# Longest name first so `asin` wins over `sin`, and no match may start straight
# after a word character or a `.` — the latter keeps an already-emitted
# `jnp.log` from being read as the function `log` should this ever run twice.
_JAX_MATH_RE = re.compile(
    r"(?<![\w.])(" + "|".join(sorted(map(re.escape, _JAX_NAMES), key=len, reverse=True)) + r")\b"
)


# `max(`/`min(` as a call, not as the tail of a longer name or an attribute.
_MINMAX_CALL_RE = re.compile(r"(?<![\w.])(max|min)\s*\(")


def _fold_minmax(expr: str) -> str:
    """Fold an n-ary ``max``/``min`` into nested binary calls, a unary one into
    its parenthesized argument, and leave a binary one byte-for-byte.

    The engine's are ExprTk's, which are variadic: ``max(a, b, c)`` is the
    largest of three and ``max(a)`` is ``a``. jnp.maximum/jnp.minimum take
    exactly two arguments, so either form reached the RHS as a TypeError at
    solve time — past _reject_untranslated_names, since ``jnp.maximum`` is a
    perfectly good name. The C path folds the same way (GH #556). Runs before
    the name passes, while the arguments are still the model's own text.
    """
    out: list[str] = []
    cursor = 0
    while (m := _MINMAX_CALL_RE.search(expr, cursor)) is not None:
        close = _find_close_paren_strict(expr, m.end() - 1)
        if close < 0:
            break  # unbalanced: leave the rest as written
        parts = [_fold_minmax(part) for part in _split_top_level_commas(expr[m.end() : close])]
        out.append(expr[cursor : m.start()])
        if len(parts) == 2:
            out.append(f"{expr[m.start() : m.end()]}{parts[0]},{parts[1]})")
        elif len(parts) == 1:
            out.append(f"({parts[0]})" if parts[0].strip() else expr[m.start() : close + 1])
        else:
            folded = parts[0].strip()
            for part in parts[1:]:
                folded = f"{m.group(1)}({folded},{part.strip()})"
            out.append(folded)
        cursor = close + 1
    out.append(expr[cursor:])
    return "".join(out)


# The two conjunctions, in both spellings the evaluator takes: `&&`/`||` are
# rewritten to ` and `/` or ` before ExprTk ever sees them
# (replace_logical_operators, src/expression.cpp), so a model may write either.
# And the six comparisons, longer spelling first so `<=` is not read as `<`.
_LOGICAL_OPS = ("&&", "||", "and", "or")
_COMPARISON_OPS = ("<=", ">=", "==", "!=", "<", ">")


def _split_top_level_ops(expr: str, ops: tuple[str, ...]) -> list[str]:
    """Split ``expr`` at paren-depth-zero occurrences of ``ops``, keeping the
    operators as their own odd-indexed parts. One part means no split.

    A word-spelled operator matches only as a whole word, so ``band`` and
    ``or_rate`` are names, not conjunctions.
    """
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return [expr]  # unbalanced: leave it to the caller
        elif depth == 0:
            for op in ops:
                if not expr.startswith(op, i):
                    continue
                if op[0].isalpha() and not _is_whole_word(expr, i, len(op)):
                    continue
                parts.append(expr[start:i])
                parts.append(op)
                i += len(op)
                start = i
                break
            else:
                i += 1
            continue
        i += 1
    if depth != 0:
        return [expr]
    parts.append(expr[start:])
    return parts


def _is_whole_word(expr: str, start: int, length: int) -> bool:
    before = expr[start - 1] if start else ""
    after = expr[start + length] if start + length < len(expr) else ""
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _regroup_for_python(expr: str) -> str:
    """Bracket the source grammar's grouping into the text before it becomes
    Python, where two of its precedences are inverted.

    ExprTk and C both put ``&&``/``||`` *below* the comparisons they join, and
    both read a comparison run left to right. Python does neither: ``&`` and
    ``|`` bind *tighter* than a comparison, and ``a < b < c`` is a chain, not
    ``(a < b) < c``. So the textual ``&&`` -> ``&`` below used to invert the
    grouping of every gated rate law:

        if(A > 2 && 4 < B, 10, 20)   ->   jnp.where(A > (2 & 4) < B, 10, 20)

    ``2 & 4`` is ``0``, and what is left is a Python comparison chain that
    happens to evaluate — so the run returned 10 where the interpreter and the
    compiled C both returned 20. On float operands the same shape raises
    ``TypeError: and does not accept dtype float64`` instead, at solve time.

    Both are grouping, not translation, so both are fixed here in one place:
    bracket each ``&&``/``||`` operand, and left-fold a comparison run. An
    expression with no logical operator and fewer than two comparisons is
    returned byte-for-byte, which is nearly all of them.
    """
    # Innermost first: rewrite every parenthesized group, then this level.
    buf: list[str] = []
    i = 0
    while i < len(expr):
        if expr[i] == "(":
            close = _find_close_paren_strict(expr, i)
            if close < 0:
                buf.append(expr[i:])  # unbalanced: leave the rest as written
                i = len(expr)
                break
            buf.append("(" + _regroup_for_python(expr[i + 1 : close]) + ")")
            i = close + 1
        else:
            buf.append(expr[i])
            i += 1
    rebuilt = "".join(buf)

    # An argument list is a sequence of regions, each grouped on its own.
    return ",".join(_regroup_operand(arg) for arg in _split_top_level_commas(rebuilt))


def _regroup_operand(expr: str) -> str:
    """One region between commas: bracket its logical operands, or, failing
    that, left-fold its comparison run."""
    parts = _split_top_level_ops(expr, _LOGICAL_OPS)
    if len(parts) > 1:
        # Emitted as calls, not operators. Python's `and`/`or` ask a traced value
        # for bool(), which fails under the jax.jit the Jacobian is built with,
        # and `&`/`|` refuse a float operand — where ExprTk reads any number as
        # its truth value (`2 and 3` is 1). jnp.logical_and/or do both. `and`
        # binds tighter than `or`, as ExprTk has it (`1 or 0 and 0` is 1), so
        # each run of conjunctions is folded first and the disjuncts after.
        operands = [_regroup_operand(part).strip() for part in parts[0::2]]
        disjuncts: list[str] = []
        current = operands[0]
        for op, operand in zip(parts[1::2], operands[1:], strict=True):
            if op in ("&&", "and"):
                current = f"jnp.logical_and({current}, {operand})"
            else:
                disjuncts.append(current)
                current = operand
        disjuncts.append(current)
        folded = disjuncts[0]
        for disjunct in disjuncts[1:]:
            folded = f"jnp.logical_or({folded}, {disjunct})"
        return folded

    parts = _split_top_level_ops(expr, _COMPARISON_OPS)
    if len(parts) <= 3:
        return expr  # one comparison or none — Python already agrees
    folded = f"({parts[0].strip()}{parts[1]}{parts[2].strip()})"
    for idx in range(3, len(parts) - 1, 2):
        folded = f"({folded}{parts[idx]}{parts[idx + 1].strip()})"
    return folded


def _translate_expr_jax(
    expr: str,
    param_names: dict[str, int],
    obs_names: dict[str, int],
    func_names_set: set[str],
    func_order: list[str],
) -> str:
    """Translate a model function expression to JAX-compatible Python.

    Replaces:
      - parameter names -> params[idx]
      - observable names -> obs[idx]
      - function names -> func_<name> (local variable)
      - a scalar written as a zero-arg call (`Atot()`) -> the scalar (issue #28)
      - time() -> t
      - if(cond,a,b) -> jnp.where(cond,a,b)
      - ln() -> jnp.log()
      - common math -> jnp.<func>()
      - && -> & (JAX boolean), || -> | (JAX boolean)
    """
    # ExprTk's reading of =, <>, -- and relational chains made explicit, as in
    # the C translators and the sympy parsers (issue #734).
    c = _normalize_exprtk_operators(expr)
    c = _jax_numeric_literals_as_floats(c)

    # Bracket the source's grouping before the operators become Python ones,
    # whose precedence differs (GH #579).
    c = _regroup_for_python(c)

    # _regroup_for_python has already turned every balanced &&/|| into a
    # jnp.logical_* call; this only reaches text it left as written.
    c = c.replace("&&", " & ")
    c = c.replace("||", " | ")

    # The clock, parked under a placeholder no model name can collide with.
    #
    # Issue #659 — `time` is the only clock symbol the evaluator binds; `t` is
    # deliberately left free as an ordinary model identifier
    # (src/expression.cpp), and `t()` is that identifier written as a zero-arg
    # call. Rewriting both to a bare `t` here put the clock and a model symbol
    # named `t` into the same token, and the observable pass below then rewrote
    # it: in a model with an observable `t`, `time()` came out as `obs[i]` — the
    # clock silently replaced by a population. The placeholder survives every
    # substitution below and is spent last.
    c = re.sub(r"\btime\s*\(\s*\)", _CLOCK_SYM, c)

    # A scalar written as a zero-arg call (issue #28): BNGL accepts `Atot()`
    # wherever `Atot` is valid and BNG2.pl preserves whichever the user wrote,
    # so the engine strips the parens for any registered scalar
    # (strip_empty_parens, src/expression.cpp). Do the same before the name
    # passes below, or `Atot()` becomes `obs[1]()` — a call on a JAX array —
    # and `t()` becomes `obs[3]()`. `time()` is already gone, and it is the only
    # zero-arg built-in, so every remaining empty argument list is a scalar.
    c = _EMPTY_CALL_RE.sub(r"\1", c)

    c = _fold_minmax(c)

    # Replace if(cond, a, b) -> jnp.where(cond, a, b)
    # This handles nested if() via repeated application
    for _ in range(10):  # max nesting depth
        new_c = re.sub(r"\bif\s*\(", "jnp.where(", c)
        if new_c == c:
            break
        c = new_c

    # Replace function references: funcName() or bare funcName
    # Must do BEFORE parameter replacement
    for fname in func_order:
        safe = _safe_py_name(fname)
        c = re.sub(rf"\b{re.escape(fname)}\(\)", f"func_{safe}", c)
        c = re.sub(rf"\b{re.escape(fname)}\b", f"func_{safe}", c)

    # Replace observable and parameter names, each longest first so a name is
    # never rewritten inside a longer one. The trailing `(?!\[)` keeps a later
    # pass off an index an earlier pass emitted: model text never indexes, so a
    # name followed by `[` is always `obs[` or `params[`, and a parameter named
    # `obs` must not rewrite it. Species are not read by name: only SBML does,
    # and _refuse_unsupported refuses SBML, so a species pass could only shadow
    # a parameter of the same name, which the engine reads (issue #803).
    for name in sorted(obs_names.keys(), key=len, reverse=True):
        idx = obs_names[name]
        c = re.sub(
            rf"(?<!func_)\b{re.escape(name)}\b(?!\[)",
            f"obs[{idx}]",
            c,
        )

    for name in sorted(param_names.keys(), key=len, reverse=True):
        idx = param_names[name]
        c = re.sub(
            rf"(?<!func_)\b{re.escape(name)}\b(?!\[)",
            f"params[{idx}]",
            c,
        )

    # Replace math functions with jnp equivalents, in ONE pass (GH #564).
    # Run as a sequence of re.sub calls, each rule saw what the rules before it
    # had already emitted: `ln` became `jnp.log`, and the very next rule's
    # `\blog\b` matched the `log` inside it — `.` is not a word character, so
    # the boundary holds there — leaving `jnp.jnp.log` and an
    # "AttributeError: module 'jax.numpy' has no attribute 'jnp'" at
    # evaluation. One pass over the source cannot rewrite its own output, which
    # ends the whole class of collision rather than the one instance of it.
    c = _JAX_MATH_RE.sub(lambda m: _JAX_NAMES[m.group(1)], c)

    # Replace ^ with ** for exponentiation
    c = c.replace("^", "**")

    # `1/(1+x**n)` is a reciprocal Hill term. Differentiating its direct power
    # can produce inf/inf after the fraction has saturated. Rewrite the exact
    # denominator shape to a stable sigmoid before JAX traces the expression.
    c = _rewrite_hill_power_denominators(c)

    # Spend the clock placeholder last: `t` is the JAX RHS's own time argument,
    # and nothing above may rewrite it (issue #659).
    c = c.replace(_CLOCK_SYM, "t")

    # After the clock is spent, not before: the scan below rejects any bare name
    # it cannot evaluate, and `__bngsim_clock__` is one until it becomes `t`.
    _reject_untranslated_names(expr, c)

    return c


class _HillPowerDenominator(ast.NodeTransformer):
    """Replace ``1 + base**exponent`` denominator terms with a stable helper."""

    def __init__(self) -> None:
        self.changed = False

    @staticmethod
    def _is_one(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and node.value == 1

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)  # rewrites the children in place
        if not isinstance(node.op, ast.Div):
            return node
        denominator = node.right
        if not isinstance(denominator, ast.BinOp) or not isinstance(denominator.op, ast.Add):
            return node
        power = None
        if self._is_one(denominator.left):
            power = denominator.right
        elif self._is_one(denominator.right):
            power = denominator.left
        if not isinstance(power, ast.BinOp) or not isinstance(power.op, ast.Pow):
            return node
        stable_fraction = ast.Call(
            func=ast.Name(id="__bngsim_inv_hill_power__", ctx=ast.Load()),
            args=[power.left, power.right],
            keywords=[],
        )
        self.changed = True
        return ast.copy_location(
            ast.BinOp(left=node.left, op=ast.Mult(), right=stable_fraction), node
        )


def _rewrite_hill_power_denominators(expr: str) -> str:
    """Rewrite reciprocal power-sum denominators before JAX differentiates."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr
    transformer = _HillPowerDenominator()
    rewritten = transformer.visit(tree)
    if not transformer.changed:
        return expr
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def _reject_untranslated_names(expr: str, translated: str) -> None:
    """Raise if anything survived translation that the RHS cannot evaluate.

    The generated RHS evaluates each translated body with ``{"__builtins__":
    {}}`` and a namespace holding only ``jnp``/``t``/``params``/``obs``/``y``
    and the previously computed ``func_*`` locals, so a name that reaches it
    untranslated is a guaranteed ``NameError`` — raised deep inside the solve,
    naming a symbol the caller never wrote in Python (GH #565). Saying so here,
    while the expression is still in hand, costs nothing and names the function
    and the model text it came from.
    """
    scanned = _STRING_LITERAL_RE.sub("''", translated)
    unknown = sorted(
        {
            name
            for name in _BARE_IDENT_RE.findall(scanned)
            if name not in _JAX_EVAL_NAMES
            and name not in _PY_KEYWORDS
            and not name.startswith("func_")
        }
    )
    if unknown:
        raise ValueError(
            f"jacobian='jax' cannot translate {', '.join(repr(u) for u in unknown)} "
            f"in the expression {expr!r}: no jax.numpy equivalent is mapped for it. "
            "Use jacobian='auto' (the default), which evaluates this model through "
            "the engine instead."
        )


def _safe_py_name(name: str) -> str:
    """Convert a BNG name to a safe Python identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ─── JAX RHS generator ──────────────────────────────────────────────────────


def generate_jax_rhs(model: Any) -> Any:
    """Generate a JAX-traced RHS function from a built model.

    The returned function has signature::

        rhs(y: jnp.ndarray, t: float, params: jnp.ndarray) -> jnp.ndarray

    where y is species (n_species,), params is (n_params,) in the model's
    parameter order (``param_names``, derived parameters included at their
    evaluated values), and the return is dydt (n_species,).

    All operations use jnp so the function is JAX-traceable for AD.

    Parameters
    ----------
    model : Model or str
        The built model, or a ``.net`` path loaded with ``Model.from_net``.

    Returns
    -------
    Callable
        JAX-traceable RHS function.

    Raises
    ------
    ImportError
        If JAX is not installed.
    ValueError
        If the model uses a construct the JAX RHS does not implement (see
        ``_refuse_unsupported``) or an expression it cannot translate.
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    import jax.numpy as jnp

    data = _codegen_data(_as_model(model))
    _refuse_unsupported(data)
    params_list = data["parameters"]
    species_list = data["species"]
    reactions = data["reactions"]
    observables = data["observables"]
    functions = data["functions"]

    n_sp = len(species_list)
    n_params = len(params_list)

    # Index maps, all 0-based, from the loader's own reading of the model.
    param_idx = {p["name"]: i for i, p in enumerate(params_list)}
    obs_idx = {o["name"]: k for k, o in enumerate(observables)}
    func_names_set = {f["name"] for f in functions}
    func_order = [f["name"] for f in functions]
    fixed_sp = frozenset(i for i, s in enumerate(species_list) if s["fixed"])

    # obs[k] = sum_j W[k, j] * y[j]. A model with no observables gets a (0, n)
    # matrix, whose product with y is an empty vector -- not a (0,) array that the
    # matmul rejects, which is what the .net path built (issue #803).
    obs_weights = [[0.0] * n_sp for _ in observables]
    for k, o in enumerate(observables):
        for sp_i, factor in o["entries"]:
            obs_weights[k][int(sp_i)] += float(factor)
    obs_w_array = jnp.array(obs_weights, dtype=jnp.float64).reshape(len(observables), n_sp)

    # Functions in dependency order, so one that reads another declared after it
    # sees its value (issue #699 on the .net path). The value is the guarded
    # ``eval_expression`` where the loader provides one (GH #333), as in the
    # compiled RHS.
    try:
        order = _topological_function_order(list(functions))
    except CodegenDeclined as exc:
        raise ValueError(f"the JAX RHS cannot order this model's functions: {exc}") from exc
    func_exprs = []
    for i in order:
        f = functions[i]
        expr = f.get("eval_expression") or f["expression"]
        jax_expr = _translate_expr_jax(expr, param_idx, obs_idx, func_names_set, func_order)
        func_exprs.append((f["name"], compile(jax_expr, f"<jax:{f['name']}>", "eval")))

    rxn_data = []
    for r in reactions:
        reactants = [int(i) for i in r["reactants"]]
        products = [int(i) for i in r["products"]]
        sf = float(r["stat_factor"])
        rate_params = [int(i) for i in r["rate_param_indices"]]
        if r["type"] == "elementary":
            if not rate_params:
                raise ValueError("the JAX RHS found an elementary reaction with no rate constant")
            kind: tuple = ("elementary", rate_params[0], sf)
        elif r["type"] == "functional":
            if r["function_name"] not in func_names_set:
                raise ValueError(
                    f"the JAX RHS found a reaction driven by an unknown function "
                    f"{r['function_name']!r}"
                )
            kind = ("functional", r["function_name"], sf)
        else:  # "mm"; _refuse_unsupported admits no other type
            if len(rate_params) < 2 or len(reactants) < 2:
                raise ValueError("the JAX RHS found a malformed Michaelis-Menten reaction")
            kind = ("mm", rate_params[0], rate_params[1], sf)
        rxn_data.append((reactants, products, kind))

    def rhs(y, t, params):
        """JAX-traced ODE RHS: dy/dt = f(y, t, params)."""
        obs = obs_w_array @ y

        # Evaluate functions in dependency order
        func_vals = {}
        for fname, code in func_exprs:
            local_ns = {
                **_JAX_HELPERS,
                "jnp": jnp,
                "t": t,
                "params": params,
                "obs": obs,
                "y": y,
            }
            # Add previously computed functions
            for prev_name, prev_val in func_vals.items():
                local_ns[f"func_{_safe_py_name(prev_name)}"] = prev_val
            func_vals[fname] = eval(code, {"__builtins__": {}}, local_ns)  # noqa: S307

        dydt = jnp.zeros(n_sp, dtype=y.dtype)
        for reactants, products, kind in rxn_data:
            if kind[0] == "elementary":
                _, p_i, sf = kind
                rate = params[p_i] * sf
                for ri in reactants:
                    rate = rate * y[ri]
            elif kind[0] == "functional":
                _, fname, sf = kind
                rate = func_vals[fname] * sf
                for ri in reactants:
                    rate = rate * y[ri]
            else:
                _, kcat_i, km_i, sf = kind
                kcat = params[kcat_i]
                km = params[km_i]
                e = y[reactants[0]]
                s = y[reactants[1]]
                # Stable positive root of x² − delta·x − km·s = 0 (GH #89):
                # ½(delta + D) cancels for delta < 0, so that branch uses the
                # conjugate form. The denominator is masked before the divide
                # so the unselected branch cannot inject a NaN into a tangent.
                delta = s - km - e
                d_mm = jnp.sqrt(delta * delta + 4.0 * km * s)
                neg = delta < 0.0
                denom = jnp.where(neg, d_mm - delta, 1.0)
                s_free = jnp.where(neg, 2.0 * km * s / denom, 0.5 * (delta + d_mm))
                # GH #93: no clamp on s_free (it is negative exactly when s
                # is, and the rate continues smoothly there), but the rate's
                # denominator is guarded — it vanishes when km*e == 0. Masked
                # before the divide for the same reason as above: an unmasked
                # 0/0 in the unselected branch NaNs the tangent.
                kps = km + s_free
                live = kps > 0.0
                rate = jnp.where(live, sf * kcat * s_free * e / jnp.where(live, kps, 1.0), 0.0)

            # One update per species of multiplicity * rate, as the C++ right-hand
            # side and the C emitter do (issue #801): a coefficient is one index
            # entry per unit, and a million entries is a million traced ops.
            for ri, m in _multiplicity(reactants):
                dydt = dydt.at[ri].add(-rate if m == 1 else -(m * rate))
            for pi, m in _multiplicity(products):
                dydt = dydt.at[pi].add(rate if m == 1 else m * rate)

        for si in fixed_sp:
            dydt = dydt.at[si].set(0.0)

        return dydt

    # Attach metadata (function attributes — Any-typed to satisfy mypy)
    rhs_obj: Any = rhs
    rhs_obj.n_species = n_sp
    rhs_obj.n_params = n_params
    rhs_obj.fixed_species = fixed_sp

    return rhs_obj


#: When ``check_rhs_against_engine`` is not told the horizon (``jacobian="jax"``,
#: whose ``run()`` has not been called yet): a spread, so a time-dependent rate law
#: that goes wrong only past ``t = 0`` is seen.
_CHECK_TIMES = (0.0, 1.7, 13.0, 250.0)


def check_rhs_against_engine(model: Any, rhs: Any, times: Any = None) -> None:
    """Raise ``ValueError`` unless the JAX RHS matches the engine's.

    ``_refuse_unsupported`` names every construct the JAX RHS is known not to
    implement; this catches one it does not know about. It compares against
    :meth:`bngsim.Model.rhs` -- the interpreted RHS the solver integrates -- at
    three strictly positive states (so no rate term vanishes into a zero species)
    and at each of ``times`` (default ``_CHECK_TIMES``; ``run_diffrax`` passes
    points across its own interval), so a time-dependent rate law that goes wrong
    only later is seen too. A mismatch refuses rather than hand the caller a
    Jacobian, or a trajectory, for a different model.

    Each species is held to 1e-7 of its own gross flux -- the sum of ``|rate|``
    over every reaction it takes part in, catalysts included -- far above
    roundoff and far below any semantic difference. A scale shared by every
    species let a fast reaction hide a 4x error on a slow one (issue #803).
    Values equal on both sides agree, infinities included; two NaNs agree.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    core = model._core
    data = _codegen_data(model)
    n_sp = len(data["species"])
    # Gross incidence: each appearance on either side, so `A -> A + B` counts A.
    incidence = np.zeros((n_sp, len(data["reactions"])))
    for j, r in enumerate(data["reactions"]):
        for i in list(r["reactants"]) + list(r["products"]):
            incidence[int(i), j] += 1.0

    y0 = np.abs(np.asarray(core.get_initial_state(), dtype=np.float64))
    rng = np.random.default_rng(803)
    scale = max(float(y0.max(initial=0.0)), 1.0)
    states = [
        y0 * (1.0 + 0.25 * rng.random(n_sp)) + 0.1 * scale * (0.5 + rng.random(n_sp)),
        y0 * rng.random(n_sp) + 0.5 * scale * rng.random(n_sp) + 1e-3 * scale,
        scale * (1.0 + 2.0 * rng.random(n_sp)),
    ]
    ts = [float(t) for t in (_CHECK_TIMES if times is None else times)]
    if not core.functions_use_time:
        ts = ts[:1]  # nothing reads the clock, so one time says it all
    ys = np.array([y for y in states for _ in ts], dtype=np.float64)
    tt = np.array([t for _ in states for t in ts], dtype=np.float64)
    p = np.array([core.get_param(n) for n in core.param_names], dtype=np.float64)
    # Never jit: a compile cost more than the check (5.3 s -> 8.8 s of setup at
    # 1,000 reactions). At 1,032 reactions an eager evaluation is 0.29 s a point
    # and a vmap pass a flat 1.6 s, so a few points go eagerly and many in one pass.
    pj = jnp.asarray(p)
    if len(ys) <= 5:
        f_jax_all = np.array(
            [np.asarray(rhs(jnp.asarray(y), t, pj)) for y, t in zip(ys, tt, strict=True)]
        )
    else:
        f_jax_all = np.asarray(
            jax.vmap(rhs, in_axes=(0, 0, None))(jnp.asarray(ys), jnp.asarray(tt), pj)
        )
    f_jax_all = f_jax_all.astype(np.float64)
    names = list(model.species_names)
    for y, t, f_jax in zip(ys, tt, f_jax_all, strict=True):
        f_eng = np.asarray(model.rhs(y, t), dtype=np.float64)
        gross = incidence @ np.abs(np.asarray(model.propensities(y, t), dtype=np.float64))
        finite = gross[np.isfinite(gross)]
        tol = 1e-7 * gross + 1e-12 * float(finite.max(initial=0.0))
        agree = (
            (f_jax == f_eng) | (np.isnan(f_eng) & np.isnan(f_jax)) | (np.abs(f_jax - f_eng) <= tol)
        )
        if not agree.all():
            i = int(np.argmin(agree))
            raise ValueError(
                f"the JAX RHS disagrees with the engine's for species {names[i]} at "
                f"t = {t:g} (JAX {f_jax[i]!r}, engine {f_eng[i]!r}), so it would "
                "describe a different model; refusing. Use jacobian='auto' (the default)."
            )


# ─── JAX Jacobian wrapper ───────────────────────────────────────────────────


def generate_jax_jacobian(model: Any) -> Any:
    """Generate a JAX AD Jacobian function from a built model.

    Returns a function::

        jac_fn(y: ndarray, t: float, params: ndarray) -> ndarray

    that computes the exact N×N dense Jacobian ∂f/∂y via forward-mode AD.

    Parameters
    ----------
    model : Model or str
        The built model, or a ``.net`` path loaded with ``Model.from_net``.

    Returns
    -------
    Callable
        Function that returns (n_species, n_species) Jacobian matrix.

    Raises
    ------
    ImportError
        If JAX is not installed.
    ValueError
        If the model uses a construct the JAX RHS does not implement.

    A model whose functions use floor/ceil/etc. is built, with a logged warning:
    its Jacobian is zero across each step, which costs CVODE step size, not
    correctness.
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    # Screen for discontinuities — warn but don't reject.
    # CVODE uses the Jacobian for Newton convergence, not solution
    # correctness. A locally-zero Jacobian at a discontinuity just
    # means CVODE takes smaller steps there (same as FD behavior).
    model = _as_model(model)
    problems = screen_for_discontinuities(model)
    if problems:
        logger.warning(
            "Model functions may produce zero JAX AD gradients at "
            "discontinuities (CVODE will adapt step size):\n  %s",
            "\n  ".join(problems),
        )

    import jax

    rhs = generate_jax_rhs(model)

    # Forward-mode AD: differentiate RHS w.r.t. y (argnums=0)
    # Returns (n_species, n_species) Jacobian matrix
    jac_fn_raw = jax.jacfwd(rhs, argnums=0)

    def jac_fn(y, t, params):
        """Compute dense Jacobian J[i][j] = ∂f_i/∂y_j."""
        return jac_fn_raw(y, t, params)

    # Attach metadata (function attributes — Any-typed to satisfy mypy)
    jac_fn_obj: Any = jac_fn
    jac_fn_obj.n_species = rhs.n_species
    jac_fn_obj.n_params = rhs.n_params
    jac_fn_obj.rhs = rhs

    return jac_fn_obj


# ─── Prepare JAX Jacobian for CVODE ─────────────────────────────────────────


def prepare_jax_jacobian(model: Any) -> tuple:
    """Prepare a JAX Jacobian evaluator for use with CVODE.

    Returns a tuple (jac_fn, n_species) where jac_fn is a callable that
    takes (y_flat, t, param_flat) and returns a flat column-major Jacobian
    array, the layout CVODE's dense matrix uses.

    The JAX RHS is first checked against the engine's
    (:func:`check_rhs_against_engine`), so a model it would misdescribe is
    refused rather than solved with a Jacobian of some other system.

    Parameters
    ----------
    model : Model or str
        The built model, or a ``.net`` path loaded with ``Model.from_net``.

    Returns
    -------
    tuple
        (evaluate_jacobian, n_species) where evaluate_jacobian is
        a callable (y_flat, t, param_flat) -> flat_jac_array.
    """
    if not jax_available():
        raise ImportError(
            "JAX is required for jacobian='jax'. Install with: pip install jax jaxlib"
        )

    import jax
    import jax.numpy as jnp
    import numpy as np

    model = _as_model(model)
    jac_fn = generate_jax_jacobian(model)
    check_rhs_against_engine(model, jac_fn.rhs)
    n_sp = jac_fn.n_species

    # JIT-compile for speed
    jac_fn_jit = jax.jit(jac_fn)

    # Warm up with dummy data to trigger compilation
    dummy_y = jnp.ones(n_sp, dtype=jnp.float64)
    dummy_p = jnp.ones(jac_fn.n_params, dtype=jnp.float64)
    # warmup may fail with dummy params; that's OK
    with contextlib.suppress(Exception):
        _ = jac_fn_jit(dummy_y, 0.0, dummy_p)

    def evaluate_jacobian(y_flat, t, param_flat):
        """Evaluate Jacobian, return column-major flat array for CVODE.

        CVODE dense matrix is column-major (Fortran order).
        """
        y_jax = jnp.array(y_flat, dtype=jnp.float64)
        p_jax = jnp.array(param_flat, dtype=jnp.float64)
        J = jac_fn_jit(y_jax, t, p_jax)
        # J is (n_sp, n_sp) with J[i][j] = df_i/dy_j
        # CVODE dense matrix is column-major: column j, row i
        # np.asfortranarray gives column-major layout
        J_np = np.asarray(J, dtype=np.float64)
        return J_np.flatten(order="F")

    return evaluate_jacobian, n_sp
