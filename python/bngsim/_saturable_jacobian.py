"""Native, **SymPy-free** closed-form differentiation of saturable rate laws
(GH #151).

The GH #76 analytical Jacobian differentiates Functional rate laws with SymPy
(``bngsim._jacobian.differentiate_rate_law`` → ``sympy_to_exprtk`` /
``sympy_to_c``). SymPy is correct but slow: a model with many saturable
Functional reactions can blow the per-build derivation budget (#95), and because
``NetworkModel::analytical_jacobian_complete()`` is all-or-nothing, a single
un-derived reaction discards the whole model's closed-form Jacobian.

Saturable kinetics are a small, fixed algebraic family — Hill terms, rational /
saturation terms (the legacy ``Sat``/``Hill`` ``.net`` tokens, #48), basal +
regulated production, and products / shared-denominator sums of those over
several regulators. Every member is built from ``+ - * / ^`` (and a handful of
unary functions) over species and parameters, with the Hill exponents and
half-saturation constants being *parameters* (constant w.r.t. species). Their
derivatives therefore have simple closed forms (sum / product / quotient / power
/ chain rule) that can be emitted directly, with no CAS in the loop.

This module is that direct path. It:

1. **tokenizes + parses** an (already function-inlined) ExprTk rate-law string
   into a tiny arithmetic AST — only ``+ - * / ^``, unary ``±``, parentheses,
   ``time()``, and the unary functions ``exp``/``log``/``ln``/``sqrt`` plus
   two-arg ``pow``. Anything else (``if(``, comparisons, logical operators,
   an un-whitelisted function, an un-inlined user call) makes parsing fail;
2. **differentiates** the AST in closed form w.r.t. a single named variable,
   treating every other identifier as constant; and
3. **emits** the derivative as an ExprTk string *or* a C (``math.h``) string,
   reusing the same power-emission idioms as the SymPy emitters.

Robustness contract (identical to ``bngsim._jacobian``): every entry point
returns ``None`` (never raises, never a wrong derivative) when an expression
falls outside the recognized family. The caller then drops to the SymPy path,
and failing that to the finite-difference Jacobian — so the native path is a
strict speedup where it engages and byte-identical to before where it does not.

The whole module imports no SymPy. ``bngsim._jacobian`` tries it first and only
imports SymPy for the genuine fallback, so a model whose Functional rate laws
are entirely within this family obtains a complete analytical Jacobian with zero
SymPy invocations — even with SymPy uninstalled.
"""

from __future__ import annotations

import re
from collections.abc import Set as AbstractSet

# Reuse the function-inliner and keyword set from the SymPy path. These are
# pure-Python (no SymPy at import); the time placeholder must match so the
# codegen C resolver maps it to the time variable identically for both paths.
from bngsim._codegen import _PY_KEYWORD_PARAM_NAMES
from bngsim._jacobian import _TIME_SYM, _inline_functions

# Unary functions whose derivative is known and emittable. ``ln`` canonicalizes
# to ``log`` (natural log in both ExprTk and C). Anything not here fails parsing
# → SymPy fallback.
_UNARY_FUNCS = {"exp", "log", "ln", "sqrt"}
_CANONICAL_FUNC = {"exp": "exp", "log": "log", "ln": "log", "sqrt": "sqrt"}

# time() / t() spellings (parens required, matching _jacobian._preprocess_exprtk
# so a parameter literally named ``t`` is left as a variable).
_TIME_NAMES = {"time", "t"}

_NAN_INF_RE = re.compile(r"(?<![A-Za-z0-9_])(nan|inf|-inf)(?![A-Za-z0-9_])", re.IGNORECASE)


class _NativeError(Exception):
    """Internal: the expression is outside the recognized saturable family
    (unparseable, an un-whitelisted construct, or an unresolvable symbol).
    Caught at every public boundary and converted to a ``None`` return."""


# ─── AST ────────────────────────────────────────────────────────────────────
#
# Tagged tuples, immutable and cheap to hash/compare:
#   ('num', float)            numeric literal
#   ('var', name)             identifier — a state observable or a constant param
#   ('time',)                 time() / t()
#   ('neg', a)                unary minus
#   ('+'|'-'|'*'|'/'|'^', a, b)   binary op
#   ('call', name, (arg,))    exp/log/sqrt (canonical name)

_ZERO = ("num", 0.0)
_ONE = ("num", 1.0)


def _num(x: float):
    return ("num", float(x))


def _is_num(n) -> bool:
    return n[0] == "num"


def _is_zero(n) -> bool:
    return n[0] == "num" and n[1] == 0.0


def _is_one(n) -> bool:
    return n[0] == "num" and n[1] == 1.0


# Smart constructors: fold constants and collapse identities so differentiation
# does not build a tree full of ``0 +``, ``1 *`` and ``x - 0`` that would bloat
# the emitted string. Each returns a normalized AST node.


def _mk_neg(a):
    if _is_zero(a):
        return _ZERO
    if _is_num(a):
        return _num(-a[1])
    if a[0] == "neg":
        return a[1]
    return ("neg", a)


def _mk_add(a, b):
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    if _is_num(a) and _is_num(b):
        return _num(a[1] + b[1])
    # a + (-b) → a - b
    if b[0] == "neg":
        return _mk_sub(a, b[1])
    if a[0] == "neg":
        return _mk_sub(b, a[1])
    return ("+", a, b)


def _mk_sub(a, b):
    if _is_zero(b):
        return a
    if _is_zero(a):
        return _mk_neg(b)
    if _is_num(a) and _is_num(b):
        return _num(a[1] - b[1])
    if b[0] == "neg":  # a - (-b) → a + b
        return _mk_add(a, b[1])
    return ("-", a, b)


def _mk_mul(a, b):
    if _is_zero(a) or _is_zero(b):
        return _ZERO
    if _is_one(a):
        return b
    if _is_one(b):
        return a
    if _is_num(a) and _is_num(b):
        return _num(a[1] * b[1])
    return ("*", a, b)


def _mk_div(a, b):
    if _is_zero(a):
        return _ZERO
    if _is_one(b):
        return a
    if _is_num(a) and _is_num(b) and b[1] != 0.0:
        return _num(a[1] / b[1])
    return ("/", a, b)


def _mk_pow(a, b):
    if _is_zero(b):
        return _ONE
    if _is_one(b):
        return a
    if _is_num(a) and _is_num(b):
        try:
            v = a[1] ** b[1]
        except (ValueError, OverflowError, ZeroDivisionError):
            return ("^", a, b)
        if v == v and v not in (float("inf"), float("-inf")):
            return _num(v)
    return ("^", a, b)


def _mk_call(name: str, arg):
    return ("call", name, (arg,))


# ─── Tokenizer ────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _tokenize(s: str):
    """Split an ExprTk arithmetic string into tokens. Raises ``_NativeError`` on
    any character outside the recognized subset (so comparison / logical
    operators, ``%``, ``&``, ``|``, ``!`` etc. all defer to the SymPy path)."""
    toks: list[tuple[str, object]] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/^(),":
            toks.append((c, c))
            i += 1
            continue
        m = _NUM_RE.match(s, i)
        if m and (c.isdigit() or c == "."):
            toks.append(("num", float(m.group())))
            i = m.end()
            continue
        m = _IDENT_RE.match(s, i)
        if m:
            toks.append(("ident", m.group()))
            i = m.end()
            continue
        raise _NativeError(f"unexpected character {c!r} at {i}")
    toks.append(("eof", None))
    return toks


# ─── Recursive-descent parser ─────────────────────────────────────────────────
#
# Grammar (precedence low→high), matching ExprTk arithmetic:
#   expr   := add
#   add    := mul (('+'|'-') mul)*
#   mul    := unary (('*'|'/') unary)*
#   unary  := ('+'|'-') unary | power
#   power  := atom ('^' unary)?            # right-associative, exponent may be unary
#   atom   := num | ident | ident '(' args ')' | '(' expr ')'


class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.pos = 0

    def _peek(self):
        return self.toks[self.pos]

    def _advance(self):
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind):
        tok = self._advance()
        if tok[0] != kind:
            raise _NativeError(f"expected {kind!r}, got {tok[0]!r}")
        return tok

    def parse(self):
        node = self._add()
        if self._peek()[0] != "eof":
            raise _NativeError("trailing tokens")
        return node

    def _add(self):
        node = self._mul()
        while self._peek()[0] in ("+", "-"):
            op = self._advance()[0]
            rhs = self._mul()
            node = _mk_add(node, rhs) if op == "+" else _mk_sub(node, rhs)
        return node

    def _mul(self):
        node = self._unary()
        while self._peek()[0] in ("*", "/"):
            op = self._advance()[0]
            rhs = self._unary()
            node = _mk_mul(node, rhs) if op == "*" else _mk_div(node, rhs)
        return node

    def _unary(self):
        kind = self._peek()[0]
        if kind == "+":
            self._advance()
            return self._unary()
        if kind == "-":
            self._advance()
            return _mk_neg(self._unary())
        return self._power()

    def _power(self):
        base = self._atom()
        if self._peek()[0] == "^":
            self._advance()
            exp = self._unary()  # right-assoc; exponent may carry its own sign
            return _mk_pow(base, exp)
        return base

    def _atom(self):
        tok = self._advance()
        kind = tok[0]
        if kind == "num":
            return _num(tok[1])
        if kind == "(":
            node = self._add()
            self._expect(")")
            return node
        if kind == "ident":
            name = tok[1]
            if self._peek()[0] == "(":
                return self._call(name)
            return ("var", name)
        raise _NativeError(f"unexpected token {kind!r}")

    def _call(self, name: str):
        self._expect("(")
        args = []
        if self._peek()[0] != ")":
            args.append(self._add())
            while self._peek()[0] == ",":
                self._advance()
                args.append(self._add())
        self._expect(")")

        if name in _TIME_NAMES and not args:
            return ("time",)
        if name in _UNARY_FUNCS and len(args) == 1:
            return _mk_call(_CANONICAL_FUNC[name], args[0])
        if name == "pow" and len(args) == 2:
            return _mk_pow(args[0], args[1])
        # Un-whitelisted function (e.g. if(), an un-inlined user call, a builtin
        # with no known derivative) → outside the family.
        raise _NativeError(f"unsupported call {name}({len(args)} args)")


def _parse(expr: str):
    return _Parser(_tokenize(expr)).parse()


# ─── Closed-form differentiation ────────────────────────────────────────────────


def _diff(node, target: str):
    """∂node/∂target in closed form. ``target`` is the only non-constant
    identifier; every other ``('var', …)`` and ``('time',)`` is constant."""
    tag = node[0]
    if tag == "num" or tag == "time":
        return _ZERO
    if tag == "var":
        return _ONE if node[1] == target else _ZERO
    if tag == "neg":
        return _mk_neg(_diff(node[1], target))
    if tag == "+":
        return _mk_add(_diff(node[1], target), _diff(node[2], target))
    if tag == "-":
        return _mk_sub(_diff(node[1], target), _diff(node[2], target))
    if tag == "*":
        a, b = node[1], node[2]
        return _mk_add(_mk_mul(_diff(a, target), b), _mk_mul(a, _diff(b, target)))
    if tag == "/":
        a, b = node[1], node[2]
        da, db = _diff(a, target), _diff(b, target)
        # Numerically stable quotient derivative: da/b − (a/b)·(db/b).
        # Mathematically identical to (da·b − a·db)/b², but every intermediate
        # stays at result scale instead of result×b². The naive numerator
        # overflows to ±inf (→ nan) for saturable terms like u/(1+u) whose u can
        # be astronomically large at extreme states — e.g. a Hill term in
        # conc/volume with a very small compartment volume. The stable form never
        # forms that product.
        if _is_zero(db):  # b constant: (a/b)' = da/b
            return _mk_div(da, b)
        a_over_b = _mk_div(a, b)
        if _is_zero(da):  # a constant: (a/b)' = −(a/b)·(db/b)
            return _mk_neg(_mk_mul(a_over_b, _mk_div(db, b)))
        return _mk_sub(_mk_div(da, b), _mk_mul(a_over_b, _mk_div(db, b)))
    if tag == "^":
        return _diff_pow(node[1], node[2], target)
    if tag == "call":
        return _diff_call(node, target)
    raise _NativeError(f"cannot differentiate {tag!r}")


def _diff_pow(base, exp, target: str):
    da = _diff(base, target)
    de = _diff(exp, target)
    de_zero = _is_zero(de)
    da_zero = _is_zero(da)
    if de_zero and da_zero:
        return _ZERO
    if de_zero:
        # d/dx base^exp = exp · base^(exp−1) · d(base)/dx   (exp constant).
        # Emits a MULTIPLY (base^(exp−1)), never base^exp / base — so it cannot
        # produce a removable x^n/x → 0/0 the way the SymPy path's raw output can
        # (which is why no _remove_removable_power_denominators rewrite is needed).
        new_exp = _mk_sub(exp, _ONE)
        return _mk_mul(_mk_mul(exp, _mk_pow(base, new_exp)), da)
    if da_zero:
        # d/dx a^exp = a^exp · ln(a) · d(exp)/dx   (base constant: a^x exponential).
        return _mk_mul(_mk_mul(_mk_pow(base, exp), _mk_call("log", base)), de)
    # General base^exp with the SAME state variable in BOTH base and exponent
    # (e.g. A^A). The closed form base^exp·(de·ln(base) + exp·da/base) divides by
    # the base AND takes ln(base) — both singular at base=0 (a removable 0/0 of the
    # x^x kind). This is outside the saturable family and SymPy simplifies it
    # better (e.g. A/A → 1), so defer it rather than emit the fragile form.
    raise _NativeError("general power: state in both base and exponent")


def _diff_call(node, target: str):
    name, arg = node[1], node[2][0]
    da = _diff(arg, target)
    if _is_zero(da):
        return _ZERO
    if name == "exp":
        return _mk_mul(_mk_call("exp", arg), da)
    if name == "log":  # natural log: d/dx log(a) = da/a
        return _mk_div(da, arg)
    if name == "sqrt":  # d/dx sqrt(a) = da / (2·sqrt(a))
        return _mk_div(da, _mk_mul(_num(2.0), _mk_call("sqrt", arg)))
    raise _NativeError(f"cannot differentiate call {name}")


# ─── Helpers over the AST ───────────────────────────────────────────────────────


def _var_names(node, out: set[str]):
    tag = node[0]
    if tag == "var":
        out.add(node[1])
    elif tag == "neg":
        _var_names(node[1], out)
    elif tag in ("+", "-", "*", "/", "^"):
        _var_names(node[1], out)
        _var_names(node[2], out)
    elif tag == "call":
        _var_names(node[2][0], out)
    # num / time carry no variable names
    return out


# ─── Emitters ────────────────────────────────────────────────────────────────


def _fmt_num(x: float) -> str:
    """Shortest round-trippable literal. ``repr`` yields a decimal point for
    every float (``2.0`` not ``2``), so the C emitter never risks integer
    division and the ExprTk parser reads a real."""
    return repr(float(x))


def _emit_exprtk(node) -> str:
    tag = node[0]
    if tag == "num":
        return _fmt_num(node[1])
    if tag == "var":
        return node[1]
    if tag == "time":
        return "time()"
    if tag == "neg":
        return f"(-({_emit_exprtk(node[1])}))"
    if tag in ("+", "-", "*", "/"):
        return f"({_emit_exprtk(node[1])} {tag} {_emit_exprtk(node[2])})"
    if tag == "^":
        b, e = node[1], node[2]
        if _is_num(e):
            if e[1] == -1.0:
                return f"(1.0/({_emit_exprtk(b)}))"
            if e[1] == 0.5:
                return f"sqrt({_emit_exprtk(b)})"
            if e[1] == -0.5:
                return f"(1.0/sqrt({_emit_exprtk(b)}))"
        return f"(({_emit_exprtk(b)})^({_emit_exprtk(e)}))"
    if tag == "call":
        return f"{node[1]}({_emit_exprtk(node[2][0])})"
    raise _NativeError(f"cannot emit {tag!r}")


def _emit_c(node, resolve_symbol) -> str:
    tag = node[0]
    if tag == "num":
        return _fmt_num(node[1])
    if tag == "var":
        mapped = resolve_symbol(node[1])
        if mapped is None:
            raise _NativeError(f"unresolved symbol {node[1]}")
        return mapped
    if tag == "time":
        mapped = resolve_symbol(_TIME_SYM)
        if mapped is None:
            raise _NativeError("unresolved time symbol")
        return mapped
    if tag == "neg":
        return f"(-({_emit_c(node[1], resolve_symbol)}))"
    if tag in ("+", "-", "*", "/"):
        return f"({_emit_c(node[1], resolve_symbol)} {tag} {_emit_c(node[2], resolve_symbol)})"
    if tag == "^":
        return _emit_c_pow(node[1], node[2], resolve_symbol)
    if tag == "call":
        # exp/log/sqrt map 1:1 to C math.h.
        return f"{node[1]}({_emit_c(node[2][0], resolve_symbol)})"
    raise _NativeError(f"cannot emit {tag!r}")


def _emit_c_pow(b, e, resolve_symbol) -> str:
    """Mirror ``bngsim._jacobian._CPrinter._print_Pow``: small integer powers →
    repeated multiply (cheaper than ``pow``), ``±0.5`` → ``sqrt``, else ``pow``."""
    bs = _emit_c(b, resolve_symbol)
    if _is_num(e):
        ev = e[1]
        if ev == 0.5:
            return f"sqrt({bs})"
        if ev == -0.5:
            return f"(1.0/sqrt({bs}))"
        if ev == -1.0:
            return f"(1.0/({bs}))"
        if ev == float(int(ev)):
            ni = int(ev)
            if 1 <= ni <= 4:
                return "(" + "*".join(f"({bs})" for _ in range(ni)) + ")"
            if -4 <= ni <= -1:
                return "(1.0/(" + "*".join(f"({bs})" for _ in range(-ni)) + "))"
    return f"pow({bs}, {_emit_c(e, resolve_symbol)})"


def emit_exprtk(node) -> str | None:
    """Emit ``node`` as an ExprTk string, or ``None`` if a non-finite literal
    slipped through (a degenerate derivative → defer to FD)."""
    try:
        s = _emit_exprtk(node)
    except _NativeError:
        return None
    if _NAN_INF_RE.search(s):
        return None
    return s


def emit_c(node, resolve_symbol) -> str | None:
    """Emit ``node`` as C (``math.h``) source via ``resolve_symbol``, or
    ``None`` on an unresolvable symbol / non-finite literal."""
    try:
        s = _emit_c(node, resolve_symbol)
    except _NativeError:
        return None
    if _NAN_INF_RE.search(s):
        return None
    return s


# ─── Public differentiation API (mirrors bngsim._jacobian signatures) ──────────


def differentiate_rate_law_native(
    rate_expr: str,
    func_map: dict[str, str],
    observable_names: AbstractSet[str],
    constant_names: set[str],
):
    """Native counterpart of ``bngsim._jacobian.differentiate_rate_law``.

    Differentiate ``rate_expr`` w.r.t. each observable it depends on, in closed
    form and with no SymPy. Returns ``{observable_name: ast_node}`` for every
    non-zero partial (``{}`` for a constant-rate law — a *success*, zero column),
    or ``None`` when the expression is outside the recognized saturable family
    (then the caller falls back to the SymPy path)."""
    inlined = _inline_functions(rate_expr, func_map)
    if inlined is None:
        return None
    try:
        node = _parse(inlined)
    except _NativeError:
        return None

    names: set[str] = _var_names(node, set())
    # Keyword-named identifiers (``lambda`` etc.) are aliased only by the SymPy
    # parser; defer them so the C resolver's aliased keying still matches.
    if any(name in _PY_KEYWORD_PARAM_NAMES for name in names):
        return None
    # Per-name membership instead of ``observable_names | constant_names``: on a
    # large model that union is ~10^5 elements and was rebuilt per reaction. An
    # unknown / un-inlined / possibly-state symbol surviving here means the closed
    # form cannot be guaranteed, so defer.
    for nm in names:
        if nm not in observable_names and nm not in constant_names:
            return None

    # Differentiate only the observables that actually appear (scanning all of
    # ``observable_names`` was O(n_obs) per reaction). Sorted so the emitted
    # ExprTk/C derivative ordering is deterministic regardless of set hash-seed
    # (codegen output is cached and byte-determinism-checked); ordering never
    # affects the numerical result.
    result: dict = {}
    try:
        for obs in sorted(names & observable_names):
            deriv = _diff(node, obs)
            if _is_zero(deriv):
                continue
            result[obs] = deriv
    except _NativeError:
        # A construct the closed-form differentiator declines (e.g. a general
        # power with the state in both base and exponent) → defer to SymPy.
        return None
    return result


def build_per_species_native(
    rate_expr: str,
    func_map: dict[str, str],
    obs_groups: dict[str, list],
    species_amount: dict[int, tuple],
    constant_names: set[str],
):
    """Native counterpart of ``bngsim._jacobian.build_per_species_sympy``.

    Chain-rule ``∂rate/∂obs_k`` through each observable's species group into one
    per-species AST derivative:

        ∂rate/∂x_j = Σ_k (∂rate/∂obs_k) · factor_{k→j} · (V_j if amount-valued)

    Returns ``[(species_idx0, ast_node)]`` or ``None``. ``[]`` is a *success*
    (constant-rate ⇒ zero column), distinct from ``None`` (not in the family)."""
    import math

    # ``obs_groups.keys()`` is set-like (supports ``in`` and ``&``); passing it
    # avoids materializing a fresh ~n_obs set on every reaction.
    dd = differentiate_rate_law_native(rate_expr, func_map, obs_groups.keys(), constant_names)
    if dd is None:
        return None

    per_species: dict = {}
    for obs_name, dexpr in dd.items():
        for sp_idx, factor in obs_groups[obs_name]:
            amount_valued, vol = species_amount.get(sp_idx, (False, 1.0))
            coeff = float(factor) * (float(vol) if amount_valued else 1.0)
            if not math.isfinite(coeff):
                return None  # degenerate chain-rule factor → defer whole model to FD
            if coeff == 0.0:
                continue
            term = _mk_mul(_num(coeff), dexpr)
            per_species[sp_idx] = _mk_add(per_species.get(sp_idx, _ZERO), term)

    out = []
    for sp_idx in sorted(per_species):  # deterministic species order for codegen
        expr = per_species[sp_idx]
        if _is_zero(expr):
            continue
        out.append((sp_idx, expr))
    return out
