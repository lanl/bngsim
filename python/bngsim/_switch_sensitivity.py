"""Switch-time forward sensitivity: crossing detection and ∂t*/∂p (issue #48).

A **switch time** is a fitted parameter that sets *when* a step in the dynamics
occurs, rather than how fast something happens inside a branch — the
social-distancing onset ``sigma`` / ``t0`` / ``tau1`` of the Lin2021 COVID model,
where rate laws read ``if(t>=sigma, ...)`` over a unit-rate counter clock.

Such a parameter is invisible to the variational source term: ``∂f/∂sigma`` is a
clean ``0`` inside each smooth branch, because ``sympy.diff`` of the
``if``→``Piecewise`` rewrite drops the boundary delta when the parameter appears
only in the *condition*. The entire gradient is a finite jump in the sensitivity
column at the crossing ``t*``::

    s(t*⁺) = s(t*⁻) + (f⁻ − f⁺) · ∂t*/∂p

which ``CvodeSimulator::run()`` applies (see the ``SwitchTimeSens`` note in
``types.hpp``). This module supplies what the core cannot work out for itself:

1. **Where the crossings are.** ``.net``/BNGL models register no discontinuity
   triggers — only the SBML loader calls ``add_discontinuity_trigger`` — so the
   ``if()`` conditions have to be recovered from the function bodies.
2. **∂t*/∂p, chain-ruled to the fitted primaries.** For a unit-rate clock the
   crossing time is the threshold, so ``∂t*/∂p = ∂(threshold)/∂p``; a derived
   threshold such as ``sigma = t0 + t_delta`` puts a jump for ``t0`` at *both*
   the ``t0`` and ``sigma`` switches but for ``t_delta`` only at ``sigma``. The
   partials come from the same sympy machinery as the issue #43 IC seeds.

Only crossings that actually move with a requested sensitivity parameter are
emitted. A model with ``if()`` conditions but no fitted switch time yields an
empty list, so the integration loop is left untouched and its stepping stays
bit-for-bit identical to the pre-#48 path.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import NamedTuple

from bngsim._codegen import (
    _BUILTIN_CONSTANT_VALUES,
    _derived_expr_partials_numeric,
    _find_close_paren_strict,
    _inline_derived_param_refs,
    _split_top_level_commas,
)
from bngsim._exceptions import SensitivityUnsupportedError

logger = logging.getLogger(__name__)

# `if` as a call, not as part of an identifier like `stiff` — mirrors the
# whole-word guard in _translate_bngl_if_to_piecewise.
_IF_CALL = re.compile(r"(?<![A-Za-z0-9_])if\s*\(")

# Boolean connectives that join relational atoms inside a condition. ExprTk
# accepts both the C and the word forms.
_LOGICAL = re.compile(r"&&|\|\||(?<![A-Za-z0-9_])(?:and|or|nand|nor|xor)(?![A-Za-z0-9_])")

# Longest-first so `<=` is never read as `<` followed by `=`.
_RELATIONAL = re.compile(r"<=|>=|==|!=|<|>")

# The two spellings of negation, defined together because they mean the same
# thing and every reader of one has to read the other (issue #234). Reading them
# differently is what let `!((a>1) && (b>2))` split into its two surfaces while
# `not((a>1) and (b>2))` — the same window, and the only one of the two a model
# can be written in at all — was kept whole and declined.

# A bare `!` that is not part of `!=`; the rest of the operator surface is
# covered by _RELATIONAL / _LOGICAL. This build's ExprTk rejects `!` outright
# (ERR007/ERR248), so no *loaded* model reaches here carrying one — but the
# pattern is what `uncompensated_condition_reason` and the issue #49 event path
# refuse on, and both of those read text before anything has compiled it.
_NOT_OP = re.compile(r"(?<![=!<>])!(?!=)")

# `not` as a call — what the SBML loader emits for <not/> (_ast_to_exprtk),
# and the only negation spelling ExprTk actually compiles.
_NOT_CALL = re.compile(r"(?<![A-Za-z0-9_])not\s*\(")

# ExprTk exposes simulation time as a nullary function (see expression.cpp), and
# BNGL models conventionally spell it bare; accept both.
_TIME_SYMBOLS = frozenset({"time", "time()"})

# A clock's slope must be exactly 1 for `t* = threshold` to hold. The check is an
# equality rather than a tolerance on purpose: a counter is written as a
# zeroth-order synthesis with rate constant 1, so its RHS row is exactly 1.0, and
# anything else is a different kind of variable we should not treat as a clock.
_CLOCK_SLOPE = 1.0


def _strip_redundant_parens(s: str) -> str:
    """Drop fully-enclosing parentheses: ``((a+b))`` → ``a+b``."""
    s = s.strip()
    while s.startswith("(") and _find_close_paren_strict(s, 0) == len(s) - 1:
        s = s[1:-1].strip()
    return s


def _strip_negation(s: str) -> str:
    """Peel every negation that wraps the whole of *s*, in either spelling.

    ``!((a>1) && (b>2))`` and ``not(((a>1) and (b>2)))`` both come back as
    ``(a>1) && (b>2)`` / ``(a>1) and (b>2)`` — the compound the negation was
    applied to, ready for :func:`_split_logical_atoms` to split at depth 0.

    Dropping the negation is sound for the issue #48/#150 callers for the reason
    :func:`_split_logical_atoms` documents: they want to know *where* the
    branch flips, and ¬c flips wherever c does. It is emphatically NOT sound for
    the issue #49 event path, which reduces a trigger to its *rising* edge and
    therefore reads the sense of every comparison; that path refuses a negated
    trigger outright (see :func:`_analyze_event_trigger`) before it ever gets
    here.

    Redundant parentheses are stripped on both sides of each peel, so the two
    spellings converge on one string rather than on two that differ by a pair of
    parens. Without the trailing strip, ``!(t>=sigma)`` came back as
    ``(t>=sigma)``, which :func:`_relational_split_op` does not read as a
    relational atom at all (it stops looking at depth > 0) — so the ``!``
    spelling of a *clock* threshold was refused where ``not(t>=sigma)`` is now
    admitted. Not model-reachable today, since ExprTk rejects ``!``; asserted at
    this level so the two spellings cannot drift apart again.
    """
    s = _strip_redundant_parens(s)
    while s:
        if _NOT_OP.match(s) is not None:
            s = _strip_redundant_parens(s[1:])
            continue
        m = _NOT_CALL.match(s)
        # Only a call whose argument list closes at the very end wraps the whole
        # of `s`; `not(a) > 1` is a comparison of a negation, not a negated one.
        if m is not None and _find_close_paren_strict(s, m.end() - 1) == len(s) - 1:
            s = _strip_redundant_parens(s[m.end() : -1])
            continue
        break
    return s


def _iter_if_conditions(expr: str):
    """Yield the condition string of every BNGL ``if(c, t, f)`` in *expr*.

    Recurses through both branches and through the condition itself, so the
    nested form the Lin2021 model uses —
    ``if((t>=sigma)&&(t<tau1), lambda0, if((t>=tau1)&&(t<200), lambda1, 0))`` —
    surfaces all four of its conditions.
    """
    cursor = 0
    while True:
        m = _IF_CALL.search(expr, cursor)
        if m is None:
            return
        open_paren = m.end() - 1
        close_paren = _find_close_paren_strict(expr, open_paren)
        if close_paren < 0:
            return  # malformed; the codegen path reports it properly
        args = _split_top_level_commas(expr[open_paren + 1 : close_paren])
        if len(args) == 3:
            yield args[0].strip()
            for arg in args:
                yield from _iter_if_conditions(arg)
        cursor = close_paren + 1


def _split_logical_atoms(cond: str) -> list[str]:
    """Split a boolean condition into its relational atoms.

    ``((t>=sigma)&&(t<tau1))`` → ``['t>=sigma', 't<tau1']``. Splits only at
    paren depth 0, then re-descends into parts that were parenthesised as a
    group or negated.

    Negation is peeled rather than interpreted — both spellings, through
    :func:`_strip_negation`. It flips which branch is taken but not *where* the
    crossing is, and the core reads f⁻/f⁺ by evaluating the real RHS on each side
    rather than by interpreting the condition. So a negated *compound* yields the
    surfaces of its parts: ``not((X<hi) and (X>lo))`` — what the SBML loader
    emits for a ``<not/>`` around an ``<and/>`` — splits into
    ``['X<hi', 'X>lo']``, the same pair the un-negated spelling gives, which is
    what stops one window written two ways from reaching two different machines
    (issue #234). De Morgan is the warrant: ∂(¬(A∧B)) ⊆ ∂A ∪ ∂B, so the peeled
    reading names no surface the condition does not have, and the pair it names
    is the pair ``(A and B)`` already registers.

    Both peels and the paren strip only ever shorten, which is what bounds the
    re-descent — and the re-descent happens only on a part this pass actually
    *reduced*. A logical that is at depth > 0 and under no negation leaves ``p``
    equal to what came in, and re-descending on an unchanged string never
    terminates: ``max(a, b and c) > 1`` recursed until the interpreter gave up,
    taking out the switch gate, the crossing scan and the ``.so`` cache key with
    it (issue #216, found while verifying issue #232).

    Keeping such a part whole is the conservative reading and the one the callers
    already handle: an atom nobody can split is an atom neither
    :func:`_clock_threshold_splits` nor :func:`state_switch_residual` claims, so
    :func:`uncompensated_condition_reason` declines it as a crossing nothing
    compensates.
    """
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(cond):
        c = cond[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            m = _LOGICAL.match(cond, i)
            if m is not None:
                parts.append(cond[start : m.start()])
                start = i = m.end()
                continue
        i += 1
    parts.append(cond[start:])

    atoms: list[str] = []
    for part in parts:
        p = _strip_negation(part)
        if not p:
            continue
        # Every step above only ever shortens, so ``p != cond.strip()`` means this
        # pass made progress and the recursion is finite; equality means it did
        # not, and recursing would repeat this call forever (see the docstring).
        if p != cond.strip() and _LOGICAL.search(p):
            atoms.extend(_split_logical_atoms(p))
        else:
            atoms.append(p)
    return atoms


def _relational_split_op(atom: str) -> tuple[str, str, str] | None:
    """Split a relational atom into ``(lhs, operator, rhs)``.

    Returns ``None`` when the atom carries no comparison (a bare boolean flag,
    say) — such a condition has no time threshold to stop at.
    """
    depth = 0
    i = 0
    while i < len(atom):
        c = atom[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            m = _RELATIONAL.match(atom, i)
            if m is not None:
                return atom[:i].strip(), m.group(0), atom[m.end() :].strip()
        i += 1
    return None


def _relational_split(atom: str) -> tuple[str, str] | None:
    """``(lhs, rhs)`` of a relational atom, discarding the operator.

    The issue #48 callers care only about *where* the threshold is: the core
    reads f⁻/f⁺ by evaluating the real RHS on each side of the crossing rather
    than by interpreting the comparison. The issue #49 event detector does need
    the operator (a trigger's rising edge is at its lower bound), and calls
    :func:`_relational_split_op` directly.
    """
    split = _relational_split_op(atom)
    return None if split is None else (split[0], split[2])


def _condition_spans(expr: str) -> list[tuple[int, int]]:
    """Character spans of every ``if()`` *condition* in *expr*.

    Used to decide whether a parameter reaches the RHS only by choosing a
    branch. Nested ``if``s are covered because the scan resumes just inside each
    ``if(``, so an ``if`` in a branch (or in a condition) is found on a later
    pass and contributes its own span.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        m = _IF_CALL.search(expr, cursor)
        if m is None:
            return spans
        open_paren = m.end() - 1
        close_paren = _find_close_paren_strict(expr, open_paren)
        if close_paren < 0:
            return spans
        args = _split_top_level_commas(expr[open_paren + 1 : close_paren])
        if len(args) == 3:
            spans.append((open_paren + 1, open_paren + 1 + len(args[0])))
        cursor = open_paren + 1


def _condition_only_params(
    core,
    candidates: set[str],
    derived_exprs: dict[str, str],
) -> set[str]:
    """Subset of *candidates* that reach the RHS only through ``if()`` conditions.

    Such a parameter has ``∂f/∂p ≡ 0`` in every branch interior — the premise the
    whole jump formula rests on — which is also what makes it safe to pin against
    CVODES' finite-difference probe. A parameter that ALSO scales something
    inside a branch (``if(t>=sigma, sigma*k, 0)``) has a genuine in-branch
    derivative and is excluded, so the caller can refuse instead of silently
    dropping it.

    Both the functional rate laws and the elementary rate constants are checked,
    each with function references and derived-parameter references inlined first,
    so a threshold spelled ``sigma = t0 + t_delta`` is attributed to
    ``t0``/``t_delta`` and a clock hidden behind a helper function — ``clk =
    t - sigma`` in ``if(clk>=0, k, 0)`` — is judged by its inlined text.

    Inlining is *not* optional: it is what keeps this predicate scanning the same
    text as :func:`compute_switch_time_sens`. Scanning a helper's raw body in
    isolation reads ``sigma`` in ``clk = t - sigma`` as an in-branch use and
    refuses a parameter that is condition-only — which is why the functions
    scanned here are the reaction rate laws (fully inlined), not every helper
    function standalone.
    """
    if not candidates:
        return set()
    from bngsim._jacobian import _inline_functions

    data = core.codegen_data()
    params = data["parameters"]
    func_map = dict(core.functional_jacobian_context()["function_map"])

    def _flatten(expr: str) -> str:
        return _inline_derived_param_refs(_inline_functions(expr, func_map) or expr, derived_exprs)

    pats = {
        name: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        for name in candidates
    }
    pure = set(candidates)

    # The reaction rate laws, fully inlined. A helper function reaches the RHS
    # only where it is referenced, so inlining it into the rate law that uses it
    # puts each of its parameters in the branch or the condition it actually
    # occupies there — which scanning the helper on its own cannot know.
    for rxn in data["reactions"]:
        if rxn.get("type") != "functional":
            continue
        fname = rxn.get("function_name")
        body = func_map.get(fname) if fname else rxn.get("rate_expression")
        if not body:
            continue
        flat = _flatten(body)
        if not flat:
            continue
        spans = _condition_spans(flat)
        for name in list(pure):
            for m in pats[name].finditer(flat):
                if not any(lo <= m.start() and m.end() <= hi for lo, hi in spans):
                    pure.discard(name)
                    break

    # An elementary reaction's rate constant is read directly, never through a
    # condition. (A *functional* reaction's rate_param_indices entry is the
    # synthesized holder for the function's value, whose body is covered by the
    # rate-law scan above.)
    for rxn in data["reactions"]:
        if rxn.get("type") != "elementary":
            continue
        for pidx in rxn.get("rate_param_indices", []):
            if not (0 <= pidx < len(params)):
                continue
            p = params[pidx]
            flat = _flatten(p.get("expression", "") or p["name"])
            for name in list(pure):
                if pats[name].search(flat):
                    pure.discard(name)

    return pure


def _unit_rate_clock_indices(core) -> frozenset[int]:
    """Every species index obeying ``dc/dt = 1``, from two RHS probes.

    Split out of :func:`_unit_rate_clock_species` because it is the half that
    needs no ``functional_jacobian_context()``: two evaluations of the right-hand
    side and nothing else. :func:`time_discontinuity_conditions` asks the
    question in that cheap form, so that a model with a conditional rate law but
    no counter can be dismissed without building a context that runs to tens of
    thousands of entries on a large model.
    """
    n_species = core.n_species
    if n_species == 0:
        return frozenset()
    conc = [core.get_concentration(name) for name in core.species_names]
    try:
        deriv = core._eval_rhs(0.0, conc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: RHS probe for clock detection failed: %s", exc)
        return frozenset()
    clock_idx = {i for i in range(n_species) if deriv[i] == _CLOCK_SLOPE}
    if not clock_idx:
        return frozenset()
    # Confirm the slope is state-independent: a species whose RHS merely happens
    # to equal 1 at the initial state is not a clock. Probing at a perturbed
    # state is enough to reject every state-dependent rate law.
    probe = [c + 1.0 for c in conc]
    try:
        deriv2 = core._eval_rhs(1.0, probe)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: second RHS probe failed: %s", exc)
        return frozenset()
    return frozenset(i for i in clock_idx if deriv2[i] == _CLOCK_SLOPE)


def _unit_rate_clock_species(core, ctx=None) -> dict[str, int]:
    """Map every symbol that reads a unit-rate clock to its species index.

    A *clock* is a species obeying ``dc/dt = 1`` — the counter idiom BNGL uses to
    make simulation time available to rate laws (``counter()`` with a
    zeroth-order synthesis at rate 1, exposed through an observable, commonly
    named ``t``). Its value is time plus a constant offset, so a threshold on it
    is a threshold on time and the crossing time is known a priori.

    Both the observable name and the species name map to the same index: a
    condition may be written against either. Returns ``{}`` for the overwhelming
    majority of models, which have no such species.
    """
    clock_idx = _unit_rate_clock_indices(core)
    if not clock_idx:
        return {}

    symbols: dict[str, int] = {}
    for i, name in enumerate(core.species_names):
        if i in clock_idx:
            symbols[name] = i
    if ctx is None:
        ctx = core.functional_jacobian_context()
    for obs_name, entries in ctx["observables"]:
        group = list(entries)
        if len(group) == 1 and group[0][0] in clock_idx and group[0][1] == 1.0:
            symbols[obs_name] = group[0][0]
    return symbols


def _clock_symbol_sub(expr: str, sym: str, repl: str) -> str:
    """Replace whole-word occurrences of clock symbol *sym* in *expr*.

    ``time()`` is a call and ``time`` is a bare name; both are in
    ``clock_symbols``. The bare pattern excludes a following ``(`` so it cannot
    eat the call form's head and leave ``()`` behind.
    """
    if sym.endswith("()"):
        return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(sym[:-2])}\s*\(\s*\)", repl, expr)
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_(])", repl, expr)


# Placeholder the clock is solved for. Leading underscore and a `bng` infix so it
# cannot collide with a model parameter, since it is parsed alongside them.
_CLOCK_SOLVE_SYMBOL = "_bng_clock_t"


def _clock_free(text: str, clock_symbols: AbstractSet[str]) -> bool:
    """True when *text* reads none of the model's clock symbols back."""
    return not any(_clock_symbol_sub(text, c, "\x00") != text for c in clock_symbols)


def _clock_solve_residual(atom: str, clock_symbols: AbstractSet[str]) -> tuple[str, str] | None:
    """``(clock_symbol, residual)`` for a relational atom that compares exactly
    one clock against something reading no clock back — else ``None``.

    The shared preamble of the three *solving* recognizers below
    (:func:`_clock_affine_threshold`, :func:`_clock_monomial_threshold`,
    :func:`_clock_quadratic_thresholds`), which differ only in what they do with
    the residual once they hold it. It lives in one place because the rules it
    encodes are exactly the ones a fourth solver would be most likely to get
    subtly different, and a hand-copied preamble is how paired sites drift:

    * exactly one clock symbol appears in the atom, matched longest-first so
      ``time()`` is consumed before ``time`` can match its head;
    * exactly one *side* reads it. ``t < 2*t`` is affine and does solve, to
      ``t* = 0``, but :func:`_clock_threshold_split_bare` rejects it deliberately
      and the issue #150 state path claims it; admitting it here would move a
      crossing between two machineries for no gain;
    * no *second* clock symbol survives into the residual, since two unit-rate
      clocks carry different offsets and nothing here knows them.

    The residual is ``(lhs)-(rhs)`` with the clock rewritten to
    :data:`_CLOCK_SOLVE_SYMBOL`, ready to hand to sympy.
    """
    split = _relational_split(atom)
    if split is None:
        return None
    lhs, rhs = split
    # Longest first so `time()` is consumed before `time` can match its head.
    present = [
        c
        for c in sorted(clock_symbols, key=len, reverse=True)
        if _clock_symbol_sub(atom, c, "\x00") != atom
    ]
    if not present:
        return None  # not a clock atom at all — the overwhelmingly common case
    clock_sym = present[0]
    on_left = not _clock_free(lhs, {clock_sym})
    on_right = not _clock_free(rhs, {clock_sym})
    if on_left == on_right:
        return None
    residual = _clock_symbol_sub(f"({lhs})-({rhs})", clock_sym, _CLOCK_SOLVE_SYMBOL)
    if not _clock_free(residual, clock_symbols):
        return None  # a second clock symbol survives: two clocks, unknown offsets
    return clock_sym, residual


def _clock_affine_threshold(atom: str, clock_symbols: AbstractSet[str]) -> tuple[str, str] | None:
    """``(clock_symbol, threshold_expr)`` for an atom that is *affine in a clock*
    but does not put that clock bare on one side — or ``None``.

    Issue #355. :func:`_clock_threshold_split_bare` recognizes a threshold by its
    spelling: exactly one side must be the clock symbol itself. Two shapes that
    are the same threshold fail that test, and both are in the corpus:

        (time()-Tdam) < 0                 the PEtab spelling of time() < Tdam
        0 >= Dam0 - krepair*(time()-Tdam) affine with a parameter-dependent slope

    Neither reads live state, and both have a crossing time in closed form, so
    both are exactly what issue #48's machinery exists to compensate — the bare
    test is simply not asking the right question. Declining them is not free: the
    gate is per model, so one of these takes the whole model off the analytic
    sensitivity RHS and onto CVODES' difference quotient, which integrates
    straight through the crossing and drops the jump (issue #232 measured 53%).

    So: form the residual ``lhs − rhs``, and if it is degree 1 in the clock,
    solve it. ``a·t + b = 0`` gives ``t* = −b/a``, and the returned threshold
    expression is that quotient — which is what ``∂t*/∂p`` must be differentiated
    from, and is why this is a symbolic solve rather than the numeric two-probe
    that :func:`_crossing_time_of_condition` uses for the same shapes. That one
    needs a *number* (where to stop); this one needs an *expression* (what to
    chain-rule).

    Tried only after the bare test declines, never instead of it, so no atom
    recognized today changes path, spelling, or threshold text. Every failure
    here returns ``None``, which is exactly the status quo for these atoms.

    Rejected, deliberately:

    * more than one distinct clock symbol — two unit-rate clocks carry different
      offsets and nothing here knows them;
    * a residual not linear in the clock (``t*t < k``), where ``−b/a`` is not the
      crossing;
    * a clock coefficient that is not free of the clock, or is a literal zero
      (no crossing at all);
    * an atom reading the clock on BOTH sides. ``t < 2*t`` is affine and does
      solve (to ``t* = 0``), but :func:`_clock_threshold_split_bare` rejects it
      deliberately and the state path claims it instead; admitting it here would
      move a crossing between two machineries for no gain, which is not what this
      issue measured. One side, as before — the change is where in that side the
      clock may sit, not how many sides it may sit on.

    A threshold reading live state is NOT rejected here — it does not have to be.
    The callers already resolve the returned expression against the primaries and
    get nothing back for a free symbol that is not one, which is the same door
    every other unresolvable threshold leaves by.
    """
    head = _clock_solve_residual(atom, clock_symbols)
    if head is None:
        return None
    clock_sym, residual = head

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        from bngsim._codegen import _preprocess_derived_expr

        expr = parse_expr(_preprocess_derived_expr(residual), evaluate=True)
        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        if t not in expr.free_symbols:
            return None
        a = sp.diff(expr, t)
        if t in a.free_symbols or a == 0:
            return None  # not linear in the clock, or no crossing
        threshold = sp.simplify(-(expr - a * t) / a)
        if t in threshold.free_symbols:  # pragma: no cover - implied by linearity
            return None
        text = str(threshold).replace("**", "^")
    except Exception as exc:  # noqa: BLE001 - an unparseable atom is just declined
        logger.debug("clock affine solve declined %r: %s", atom, exc)
        return None

    if not _clock_free(text, clock_symbols):
        return None
    return clock_sym, text


def _clock_monomial_threshold(
    atom: str, clock_symbols: AbstractSet[str]
) -> tuple[str, str] | None:
    """``(clock_symbol, threshold_expr)`` for a clock threshold whose residual is
    a single power of the clock — ``c·clock^n`` versus a threshold, ``n ≥ 2`` —
    else ``None``.

    Issue #418. :func:`_clock_affine_threshold` solves a residual that is degree
    **1** in the clock; this solves the next tractable shape up, a residual whose
    clock-bearing part is a single monomial ``c·clock^n``. ``time()*time() >=
    thresh`` is the corpus case (issue #414 refused it, since neither #48's affine
    solver nor #150's state root brackets it). ``c·clock^n`` is strictly monotonic
    on ``clock ≥ 0``, so it has exactly ONE crossing there, at

        clock = (−const/c)^(1/n)     — the principal (positive real) root,

    which is the clock **value** at the crossing and therefore exactly the
    "threshold expression" the issue #48 machinery already differentiates for
    ``∂t*/∂p`` and evaluates for ``t*`` — the same ``(clock_symbol, threshold_expr)``
    contract the affine solve returns, so nothing downstream changes. For
    ``time²>=thresh`` that root is ``sqrt(thresh)`` and ``∂t*/∂thresh =
    1/(2·sqrt(thresh))``; a threshold that comes out negative at run time
    (``thresh<0``, the condition always true) evaluates to a non-real value, which
    :func:`compute_switch_time_sens` reads as "no crossing" exactly as it does for
    an out-of-window one — correctly, since ``∂f/∂thresh`` is then a clean 0.

    Value-free by construction: the single positive root of a clock monomial is
    unambiguous without the run window or parameter values, which is what lets the
    *one* recognizer keep serving both the gate (:func:`uncompensated_condition_reason`)
    and the detector (:func:`compute_switch_time_sens`) — the #68 invariant. That
    is also the boundary of this first step: anything with more than one clock term
    is declined and stays refused (``(clock−5)^2``, two crossings; ``clock^2 +
    clock``, mixed and not a bare power), because picking THE crossing there needs
    the window this text transform does not have.

    Tried only after the bare and affine tests both decline, so no atom recognized
    today changes path, spelling, or threshold text.
    """
    head = _clock_solve_residual(atom, clock_symbols)
    if head is None:
        return None
    clock_sym, residual = head

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        from bngsim._codegen import _preprocess_derived_expr

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        expr = sp.expand(parse_expr(_preprocess_derived_expr(residual), evaluate=True))
        if t not in expr.free_symbols:
            return None
        const = expr.subs(t, 0)
        clock_part = sp.expand(expr - const)
        # The clock-bearing part must be a single monomial c·t^n. `Poly` raises on
        # a non-polynomial power (t^0.5, t^k); an atom that is not a clean integer
        # power of the clock is declined rather than guessed at.
        terms = sp.Poly(clock_part, t).terms()
        if len(terms) != 1:
            return None  # more than one clock term — ambiguous crossing
        (degree,), coeff = terms[0]
        if degree < 2 or t in coeff.free_symbols or coeff == 0:
            return None  # degree 1 is the affine solve's; 0 cannot reach here
        # c·t^n + const = 0  ⇒  t = (−const/c)^(1/n), the principal positive root.
        root = sp.simplify(sp.root(sp.simplify(-const / coeff), int(degree)))
        if t in root.free_symbols:  # pragma: no cover - implied by the single term
            return None
        text = str(root).replace("**", "^")
    except Exception as exc:  # noqa: BLE001 - an unsolvable atom is just declined
        logger.debug("clock monomial solve declined %r: %s", atom, exc)
        return None

    if not _clock_free(text, clock_symbols):
        return None
    return clock_sym, text


def _clock_quadratic_thresholds(
    atom: str, clock_symbols: AbstractSet[str]
) -> tuple[str, list[str]] | None:
    """``(clock_symbol, [threshold_expr, threshold_expr])`` for a clock threshold
    whose residual is a **quadratic** in the clock — else ``None``.

    Issue #421. The three recognizers before this one each name at most ONE
    crossing: :func:`_clock_threshold_split_bare` and
    :func:`_clock_affine_threshold` because a residual of degree 1 has one root,
    :func:`_clock_monomial_threshold` because ``c·clock^n`` is monotonic where the
    clock lives and so crosses once there. A quadratic is the first shape with
    genuinely more than one crossing, and it is the shape the corpus writes as a
    window: ``(time()-5)*(time()-5) >= thresh`` is true early, false through the
    middle, true again late, and ``time()*time()+time() >= thresh`` is the same
    residual with a linear term. Issue #414 refused both.

    Both crossings are still closed form — the quadratic formula — so each one is
    an expression the issue #48 machinery already knows what to do with: evaluate
    it for ``t*``, differentiate it for ``∂t*/∂p``. Differentiating the root
    expression IS the implicit function theorem for this residual, so no numeric
    root-find is needed and no new kind of record is either. What is new is only
    that ONE atom now yields TWO thresholds, which is why every caller takes a
    list (:func:`_clock_threshold_splits`).

    The two roots of ``a·clock² + b·clock + c`` are returned in the formula's own
    order, ``(−b ∓ sqrt(b²−4ac))/(2a)``; the detector sorts its records by time,
    so nothing downstream depends on which comes first. A residual whose
    discriminant is a **literal zero** is a tangency rather than a crossing and
    yields the single repeated root, so two identical records are never emitted
    for it.

    Value-free, as the recognizers above it are: which of the two roots is real,
    or in the run window, is a question about the parameter point and is answered
    by the callers that hold one — :func:`compute_switch_time_sens` filters by the
    window, and both it and :func:`clock_crossing_compensated` read a non-real
    root as the crossing that does not happen (:func:`_threshold_is_non_real`).
    Keeping it out of the recognizer is what lets the one recognizer keep serving
    both the gate and the detector, which is the issue #68 invariant.

    Degree 3 and up stay declined, deliberately, and not because sympy cannot
    write their roots down. A cubic with three real roots has none expressible in
    real radicals (the *casus irreducibilis*), so the closed forms sympy returns
    route through complex intermediates and would be read here as crossings that
    do not happen — silently dropping real jumps, which is the one failure mode
    worse than refusing. From degree 5 there is no radical form at all and sympy
    answers with ``CRootOf`` objects, which are not expressions any of this can
    evaluate. Both want a numeric root-find over the run window rather than a text
    transform, which is a different piece of machinery.

    Tried only after the bare, affine and monomial tests all decline, so no atom
    recognized today changes path, spelling, or threshold text.
    """
    head = _clock_solve_residual(atom, clock_symbols)
    if head is None:
        return None
    clock_sym, residual = head

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        from bngsim._codegen import _preprocess_derived_expr

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        expr = sp.expand(parse_expr(_preprocess_derived_expr(residual), evaluate=True))
        if t not in expr.free_symbols:
            return None
        # `Poly` raises on a non-polynomial power (t^0.5, t^k) and on anything
        # the clock sits inside a call in, so those are declined rather than
        # guessed at.
        poly = sp.Poly(expr, t)
        if poly.degree() != 2:
            return None
        a, b, c = poly.all_coeffs()
        # A coefficient reading the clock back cannot happen for a Poly in t;
        # a leading coefficient that is a literal zero would not be degree 2.
        disc = sp.simplify(b * b - 4 * a * c)
        if disc == 0:
            roots = [sp.simplify(-b / (2 * a))]
        else:
            r = sp.sqrt(disc)
            roots = [sp.simplify((-b - r) / (2 * a)), sp.simplify((-b + r) / (2 * a))]
        if any(t in root.free_symbols for root in roots):  # pragma: no cover
            return None
        texts = [str(root).replace("**", "^") for root in roots]
    except Exception as exc:  # noqa: BLE001 - an unsolvable atom is just declined
        logger.debug("clock quadratic solve declined %r: %s", atom, exc)
        return None

    if not all(_clock_free(text, clock_symbols) for text in texts):
        return None
    return clock_sym, texts


# The integer the recognizer below substitutes for ``floor(...)`` while it works
# out whether the residual is a schedule. It is the period *index* — which whole
# period the clock is in — and is eliminated again before anything is returned.
_PERIOD_INDEX_SYMBOL = "_bng_period_k"

# `ceil` is ExprTk's spelling and `ceiling` is sympy's; nothing between the two
# translates for the recognizers here, which parse the residual text directly
# rather than going through `_exprtk_to_sympy`.
_CEIL_CALL = re.compile(r"(?<![A-Za-z0-9_])ceil\s*\(")

# How many nested `Piecewise` collapses one condition may need (issue #465).
# libSBML's `rem()` expansion is one. The bound is a runaway guard on a `subs`
# loop, not a modelling limit.
_MAX_PIECEWISE_COLLAPSES = 8


class PeriodicSchedule(NamedTuple):
    """A clock condition that switches on a repeating schedule (issue #436).

    ``period``, ``offset`` and ``duty`` are expression *texts* over the model's
    parameters, in the same form as the threshold texts the other recognizers
    return. Together they say: starting at ``offset``, and every ``period``
    thereafter, the condition turns over at ``offset + k*period + duty`` and back
    at ``offset + (k+1)*period``, for every whole ``k``.

    There is no list of crossing times here on purpose. How many crossings a
    schedule has depends on how long the run is, and the recognizers in this
    module are window-free by design so that the issue #68 gate — which has no
    window — and the run-time detector can share one answer about a condition.
    A schedule keeps that property by describing the *pattern* and leaving the
    enumeration to :func:`compute_switch_time_sens`, which does hold a window.
    """

    clock: str
    period: str
    offset: str
    duty: str


def _collapse_clock_piecewise(expr, t, scope: SwitchConditionScope | None):
    """*expr* with every ``Piecewise`` over the clock replaced by the one branch
    the run actually takes, or ``None`` when that branch is not decidable.

    Issue #465. libSBML does not emit ``rem(a, b)``; it expands it into a sign
    test over two remainders::

        if(sign(a) != sign(b), a - b*ceil(a/b), a - b*floor(a/b))

    so a model that writes ``rem(time(), P) >= d`` — the same repeating schedule
    issue #436 compensated — arrives with the schedule behind an ``if()`` inside
    the condition itself, and :func:`_clock_periodic_schedule` declines it for
    having two step functions rather than one.

    The two branches are genuinely different functions: for a positive ``P`` the
    ``floor`` remainder runs over ``[0, P)`` and the ``ceil`` one over
    ``(-P, 0]``, so they differ by ``P`` everywhere except at the multiples of it.
    Which one the run takes is therefore not a question about the text, and it
    cannot be left to :func:`_schedule_matches_residual` to settle afterwards:
    the ``ceil`` branch of a positive-period schedule reads as a *negative*
    period, whose probe points land at negative clock values, where the guard
    really does select that branch and the residual really does hold one sign for
    the whole period. It would be accepted as a schedule that never turns over —
    the quiet failure that function exists to prevent — and the model would be
    admitted with every one of its real crossings uncompensated.

    So the guard is resolved here instead, against the run's own clock domain and
    the model's parameter point:

    * ``sign(clock)`` is 1, because a simulation clock is positive. The one point
      where it is not is ``t = 0``, and there both remainders are 0, so nothing
      downstream can see which branch was taken;
    * what is left must be clock-free. A guard that still reads the clock changes
      which branch is live *partway through the run*, so no single period, offset
      and duty describes the window — MODEL1006230027's shape, and out of scope
      for issue #465;
    * and it must evaluate at the parameter point, through
      :func:`_evaluate_threshold`, which binds parameter names the way the model
      does (GH #108) rather than the way sympy's parser would.

    Fail-closed at every step. Without a *scope* there is no parameter point to
    resolve a guard against, so a clock-bearing ``Piecewise`` is declined rather
    than guessed at, which keeps a caller that has no scope from disagreeing with
    one that has.
    """
    import sympy as sp

    for _ in range(_MAX_PIECEWISE_COLLAPSES):
        pieces = [p for p in expr.atoms(sp.Piecewise) if t in p.free_symbols]
        if not pieces:
            return expr
        # Innermost first, so a guard that is itself a Piecewise is resolved
        # before the branch it selects is read.
        piece = min(pieces, key=lambda p: len(p.atoms(sp.Piecewise)))
        chosen = None
        for value, cond in piece.args:
            if cond is sp.true:
                chosen = value
                break
            decided = _guard_holds(cond, t, scope)
            if decided is None:
                return None
            if decided:
                chosen = value
                break
        if chosen is None:
            return None  # every branch guard is false; the model has no value here
        expr = expr.subs(piece, chosen)
    return None  # pragma: no cover - more nesting than any corpus model writes


def _guard_holds(cond, t, scope: SwitchConditionScope | None) -> bool | None:
    """Whether *cond* holds for the whole of the run's clock domain, or ``None``
    when it is not decidable there. See :func:`_collapse_clock_piecewise`."""
    import sympy as sp

    # A simulation clock is positive, so `sign(clock)` is 1. Applied before the
    # clock-free test below, which is what lets the sign test resolve at all.
    cond = cond.replace(sp.sign(t), sp.Integer(1))
    if t in cond.free_symbols:
        return None  # the live branch changes partway through the run
    if cond is sp.true:
        return True
    if cond is sp.false:
        return False
    if scope is None:
        return None  # no parameter point to resolve it against
    bindings = {}
    for sym in cond.free_symbols:
        value = _evaluate_threshold(str(sym), scope.param_idx, scope.values, scope.derived_exprs)
        if value is None or not (abs(value) < float("inf")):
            return None
        bindings[sym] = sp.Float(value)
    resolved = cond.subs(bindings)
    if resolved is sp.true:
        return True
    if resolved is sp.false:
        return False
    return None  # a guard that did not reduce to a truth value


def _clock_guard_cannot_cross(atom: str, scope: SwitchConditionScope) -> bool:
    """True when *atom* reads the clock only through ``sign()``, and so holds one
    truth value for the whole run (issue #465).

    The companion to :func:`_collapse_clock_piecewise`. libSBML's expansion of
    ``rem(a, b)`` puts an ``if()`` inside the condition, and
    :func:`_iter_condition_atoms` descends into an ``if()``'s condition as
    readily as into its branches — rightly, since a nested ``if()`` in a *branch*
    is a real discontinuity someone has to compensate. So the guard
    ``sign(time()) != sign(P)`` arrives as an atom in its own right, is neither a
    clock threshold nor a comparison over state, and refuses the model even
    though the schedule wrapped around it is fully compensated.

    It is not a crossing. ``sign(clock)`` is 1 for the whole of a run, so the
    guard picks its branch before the first step and holds it to the last, which
    is what :func:`condition_cannot_cross` says about a comparison over
    run-constants — the same ground, reached by reading a value rather than a
    name. It cannot go in that function: that one is documented as structural,
    never numeric, and :func:`switch_gate_cache_digest` relies on it, so this is
    asked from :func:`clock_crossing_compensated` instead, which is
    value-dependent already and is what the digest carries.

    Deliberately narrow. Only an atom that *reads the clock* and stops reading it
    once ``sign(clock)`` is resolved is claimed here, so ``time() >= sigma`` —
    whose clock survives the substitution — is left to the recognizers exactly as
    before, and an atom with no clock in it at all is left to
    :func:`condition_cannot_cross`. Widening it to those would admit a derived
    parameter that function deliberately refuses, which is a different question
    from this one.

    The one instant the substitution is wrong about is ``t = 0``, where
    ``sign(0)`` is 0. For the expansion this exists to read, both branches are
    equal there — a remainder is 0 at 0 either way — so nothing downstream can
    see it. A guard whose branches genuinely differ at exactly the run's first
    instant would be read as not crossing when it does; no corpus model writes
    one, and the schedule the guard sits inside is checked against the model's
    own residual afterwards regardless
    (:func:`_schedule_matches_residual`).
    """
    if _clock_free(atom, scope.clock_symbols):
        return False  # not a clock atom; condition_cannot_cross judges it
    present = [
        c
        for c in sorted(scope.clock_symbols, key=len, reverse=True)
        if _clock_symbol_sub(atom, c, "\x00") != atom
    ]
    if not present:
        return False
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        from bngsim._codegen import _preprocess_derived_expr

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        text = _clock_symbol_sub(atom, present[0], _CLOCK_SOLVE_SYMBOL)
        cond = parse_expr(_preprocess_derived_expr(_CEIL_CALL.sub("ceiling(", text)))
        if not isinstance(cond, sp.logic.boolalg.Boolean) or t not in cond.free_symbols:
            return False
        return _guard_holds(cond, t, scope) is not None
    except Exception as exc:  # noqa: BLE001 - an unreadable atom is simply not claimed
        logger.debug("clock guard cross test declined %r: %s", atom, exc)
        return False


def _clock_periodic_schedule(
    atom: str,
    clock_symbols: AbstractSet[str],
    scope: SwitchConditionScope | None = None,
) -> PeriodicSchedule | None:
    """The repeating schedule *atom* switches on, or ``None`` when it is not one.

    Issue #436. The four recognizers above this one each name a fixed number of
    crossing times, because their residuals are polynomials in the clock and a
    polynomial has finitely many roots. The shape this one recognizes has a
    crossing in every period for as long as the run lasts::

        if(time() - 24*floor(time()/24) >= 7, on, off)

    which is "on for the last 17 hours of every 24 hour day". Written out with a
    start time and a fitted period it is how a model spells repeated dosing, a
    light and dark cycle, or a train of stimulus pulses, and it is by a wide
    margin the most common rate-law crossing bngsim could not compensate.

    The recognizer works on the residual rather than on the spelling, which is
    the lesson of issue #355: ``7 <= time() - 24*floor(time()/24)`` and
    ``(time()-start) - floor((time()-start)/P)*P <= duration`` are the same
    schedule, and one corpus model writes the whole remainder in seconds and
    divides back to hours afterwards. So:

    0. any ``if()`` *inside* the condition is collapsed to the branch the run
       takes (:func:`_collapse_clock_piecewise`), which is how libSBML's
       expansion of ``rem(time(), P)`` is read as the schedule it is (issue
       #465);
    1. one ``floor`` (or ``ceil``, rewritten to a floor by the exact identity
       ``ceil(x) = -floor(-x)``) reads the clock, and nothing else non-linear
       does;
    2. its argument is affine in the clock, ``(clock - offset)/period``, which is
       what fixes the period and the phase;
    3. with that floor replaced by a whole number ``k``, the residual is affine
       in the clock and in ``k`` together — ``a*clock + b1*k + b0``;
    4. and ``a*period + b1 == 0``, which is what makes the schedule *repeat*.
       Without it the residual reads differently in each period, so the pattern
       of crossings is not the same in every one and the window-free description
       this returns would not be true. ``rem(t, P) >= t/2`` is the shape that
       fails here: it is enumerable, but not by a period, an offset and a duty.

    From (3) and (4) the residual on the whole of period ``k`` is
    ``a*(clock - k*period) + b0``, so it vanishes at ``offset + k*period + duty``
    with ``duty = -b0/a - offset``, and it jumps where the floor does, at
    ``offset + k*period``. Those are the two edges per period, and both are
    ordinary issue #48 crossing times: an expression to evaluate for ``t*`` and
    to differentiate for ``∂t*/∂p``.

    Whether the schedule crosses at all is a question about values, not text —
    it needs ``0 < duty/period < 1``, and a corpus model really does write
    ``rem(time(), P) >= 0``, which is true for every t and never crosses — so it
    is answered by :func:`_periodic_schedule_terms` where the parameter point is
    known, in the same place the other recognizers' callers ask whether a root
    is real.

    Tried only after all four polynomial recognizers decline, so no atom
    recognized today changes path, spelling or threshold text.

    *scope* is read only to resolve a guard in step 0, and only a condition
    carrying an ``if()`` needs it — every atom recognized without one is
    recognized identically with it. Passing it is what keeps the issue #68 gate
    and the run-time detector answering the same way about such a condition, so
    all five callers do.
    """
    head = _clock_solve_residual(atom, clock_symbols)
    if head is None:
        return None
    clock_sym, residual = head

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        from bngsim._codegen import _preprocess_derived_expr

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        expr = parse_expr(_preprocess_derived_expr(_CEIL_CALL.sub("ceiling(", residual)))
        # An `if()` inside the condition — libSBML's expansion of `rem()` is the
        # one the corpus writes — reaches here as a `Piecewise`. Collapsing it to
        # the branch the run takes is what lets the rest of this read the
        # schedule behind it, and it has to happen before the step functions are
        # counted, since the two branches contribute one each (issue #465).
        expr = _collapse_clock_piecewise(expr, t, scope)
        if expr is None:
            return None
        # `ceil(x) = -floor(-x)` holds for every real x, integers included, so
        # this is a rewrite and not an approximation. Doing it here means the
        # rest of the recognizer has one step function to reason about.
        expr = expr.replace(sp.ceiling, lambda x: -sp.floor(-x))
        if t not in expr.free_symbols:
            return None
        steps = {f for f in expr.atoms(sp.floor) if t in f.free_symbols}
        if len(steps) != 1:
            return None  # a schedule of schedules; nothing here enumerates that
        step = next(iter(steps))
        arg = step.args[0]
        if arg.atoms(sp.floor):
            return None  # a floor inside the floor's own argument
        alpha = sp.simplify(sp.diff(arg, t))
        if t in alpha.free_symbols or alpha == 0:
            return None  # the floor's argument is not affine in the clock
        beta = sp.simplify(arg - alpha * t)
        if t in beta.free_symbols:
            return None
        k = sp.Symbol(_PERIOD_INDEX_SYMBOL)
        flat = expr.subs(step, k)
        if flat.atoms(sp.floor) or t not in flat.free_symbols:
            return None
        a = sp.simplify(sp.diff(flat, t))
        b1 = sp.simplify(sp.diff(flat, k))
        if a == 0 or {t, k} & (a.free_symbols | b1.free_symbols):
            return None  # not affine in the clock and the period index together
        b0 = sp.simplify(flat - a * t - b1 * k)
        if {t, k} & b0.free_symbols:
            return None
        period = sp.simplify(1 / alpha)
        offset = sp.simplify(-beta / alpha)
        if sp.simplify(a * period + b1) != 0:
            return None  # the residual does not repeat period to period
        duty = sp.simplify(-b0 / a - offset)
        texts = [str(e).replace("**", "^") for e in (period, offset, duty)]
    except Exception as exc:  # noqa: BLE001 - an unreadable atom is just declined
        logger.debug("clock periodic schedule declined %r: %s", atom, exc)
        return None

    if not all(_clock_free(text, clock_symbols) for text in texts):
        return None
    return PeriodicSchedule(clock_sym, *texts)


def _clock_threshold_splits(
    atom: str, clock_symbols: AbstractSet[str]
) -> tuple[str, list[str]] | None:
    """``(clock_symbol, [threshold_expr, ...])`` for a clock threshold, one entry
    per crossing it has, else ``None``.

    Four recognizers, asked in order, and the order is the blast radius: the
    spelling test below answers first and unchanged, so every atom admitted
    before issue #355 is admitted by the same code with the same threshold text.
    Only an atom it *declines* reaches :func:`_clock_affine_threshold`, which
    solves a residual degree 1 in the clock instead of matching on where the clock
    sits (``(time()-Tdam)<0``, ``0>=Dam0-krepair*(time()-Tdam)``). Only an atom
    that too declines reaches :func:`_clock_monomial_threshold` (issue #418), which
    solves the next shape up — a single power ``c·clock^n``, ``time()*time()>=thresh``.
    Only an atom all three decline reaches :func:`_clock_quadratic_thresholds`
    (issue #421), the first recognizer that can answer with more than one crossing
    (``(time()-5)*(time()-5)>=thresh``, a window with an edge at each end).

    The list is why this is plural. The first three recognizers each answer with
    exactly one threshold and are wrapped into a one-element list here, so a caller
    reading the list back gets the same single crossing it always did.

    :func:`_clock_threshold_split_oriented` deliberately does NOT come through
    here — see its docstring.
    """
    for recognizer in (
        _clock_threshold_split_bare,
        _clock_affine_threshold,
        _clock_monomial_threshold,
    ):
        single = recognizer(atom, clock_symbols)
        if single is not None:
            return single[0], [single[1]]
    return _clock_quadratic_thresholds(atom, clock_symbols)


def _clock_threshold_split_bare(
    atom: str, clock_symbols: AbstractSet[str]
) -> tuple[str, str] | None:
    """Split a relational atom into ``(clock_symbol, threshold_expr)``, or
    ``None`` when it is not a clock-versus-threshold comparison.

    This is the *one* place that decides what a **recognized clock threshold**
    is, and it has two callers that must never drift apart (issue #68, and the
    #56 lesson before it):

    * :func:`compute_switch_time_sens`, which turns each one it finds into a
      crossing to stop at and a ``∂t*/∂p`` to jump by;
    * :func:`uncompensated_condition_reason`, the codegen gate that decides
      whether a condition-bearing Functional rate law may use the analytic
      sensitivity RHS — which is sound *only* for the conditions the first
      caller compensates.

    An atom qualifies when exactly one side is bare clock symbol and the other
    side reads no clock back. ``t < 2*t`` fails the second test (no fixed
    crossing time) and ``X > thresh`` fails the first (a state threshold, whose
    crossing moves with the trajectory and is nobody's ``∂t*/∂p``).
    """
    split = _relational_split(atom)
    if split is None:
        return None
    lhs_bare = _strip_redundant_parens(split[0])
    rhs_bare = _strip_redundant_parens(split[1])
    lhs_clock = lhs_bare in clock_symbols
    rhs_clock = rhs_bare in clock_symbols
    # Exactly one side must be the clock; `t < 2*t` is not a fixed crossing, and
    # neither-side is not a time threshold at all.
    if lhs_clock == rhs_clock:
        return None
    clock_sym, threshold_expr = (lhs_bare, rhs_bare) if lhs_clock else (rhs_bare, lhs_bare)
    # A threshold that reads the clock back (`t < 2*t`) has no fixed crossing
    # time. A threshold over some *other* state is caught a step later by the
    # caller: the partials come back empty because the expression has a free
    # symbol that is not a primary parameter.
    if any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_])", threshold_expr)
        for sym in clock_symbols
    ):
        return None
    return clock_sym, threshold_expr


# Which side of its threshold a clock has to be on for a relational atom to be
# true: "lower" ⇒ true for large clock values (`t >= T0`, `T0 <= t`), so the
# atom's own crossing is a false→true edge; "upper" ⇒ true for small ones
# (`t <= toff`), so it bounds the interval from above and its crossing is a
# true→false edge.
_LOWER_OPS = frozenset({">", ">="})
_UPPER_OPS = frozenset({"<", "<="})


def _clock_threshold_split_oriented(
    atom: str, clock_symbols: AbstractSet[str]
) -> tuple[str, str, str] | None:
    """:func:`_clock_threshold_splits` plus which side of the threshold is true.

    Returns ``(clock_symbol, threshold_expr, "lower" | "upper")``, or ``None``
    when the atom is not a clock-versus-threshold comparison *or* its operator
    does not cut the time axis into a half-line. ``==`` / ``!=`` are rejected on
    the second ground: an equality on a continuous clock is true on a measure-
    zero set that the root finder cannot reliably straddle, so there is no
    well-defined rising edge to differentiate.

    Used only by the issue #49 event-time detector. The issue #48 rate-law path
    keeps calling :func:`_clock_threshold_splits`, which is deliberately
    orientation-blind: an ``if()`` branch flips at the threshold whichever way
    the comparison points, and the core reads f⁻/f⁺ by evaluating the RHS on
    each side.

    Bound to :func:`_clock_threshold_split_bare`, NOT to the widened
    :func:`_clock_threshold_splits` (issue #355). Orientation here is read off
    *which side the clock sits on*, and that is only meaningful when the clock is
    bare: for ``0 >= Dam0 - krepair*(time()-Tdam)`` the clock is on the right, so
    this would answer "upper" where the solved threshold
    ``time() >= Tdam + Dam0/krepair`` is plainly a lower one. Getting it right
    would mean reading the sign of the clock's coefficient, which is a parameter
    expression and so not always statically signed. Events are a separate
    machinery with its own crossing detector; widening them is not what #355
    measured, so this stays where it was.
    """
    split = _clock_threshold_split_bare(atom, clock_symbols)
    if split is None:
        return None
    op_split = _relational_split_op(atom)
    if op_split is None:  # pragma: no cover - _clock_threshold_splits implies one
        return None
    _lhs, op, _rhs = op_split
    clock_sym, threshold_expr = split
    clock_on_left = _strip_redundant_parens(op_split[0]) == clock_sym
    if op in _LOWER_OPS:
        kind = "lower" if clock_on_left else "upper"
    elif op in _UPPER_OPS:
        kind = "upper" if clock_on_left else "lower"
    else:
        return None
    return clock_sym, threshold_expr, kind


class SwitchConditionScope(NamedTuple):
    """Everything :func:`uncompensated_condition_reason` needs about a model.

    Built once by :func:`switch_condition_scope` from the same ``core`` the
    switch-time detector reads, so the gate and the detector cannot disagree
    about which symbols are clocks or which parameters a threshold reduces to.
    """

    # The C++ ``NetworkModel`` the rest of this was read from. Carried because
    # deciding whether a *state* condition's crossing is compensated is not a
    # text question — the solver has to be able to split the atom into a
    # residual it can root on and differentiate, which only
    # ``state_switch_residual`` knows (issue #150).
    core: object
    # Unit-rate clock symbol → the species index it reads. Literal simulation
    # `time` is not in here (it is no species); ``clock_symbols`` is the union.
    clocks: dict[str, int]
    clock_symbols: frozenset[str]
    param_names: tuple[str, ...]
    param_pats: dict[str, re.Pattern]
    primary_names: frozenset[str]
    param_idx: dict[str, int]
    values: tuple[float, ...]
    derived_exprs: dict[str, str]
    # Names whose value is fixed for the whole run, so a comparison written over
    # nothing else cannot change its truth value while the solver integrates
    # (:func:`condition_cannot_cross`). Primary parameters, less the slots a
    # model *function* owns — ``evaluate_functions()`` rewrites those from the
    # function's own expression before every derivative evaluation (issues #227,
    # #266), so their value moves with the trajectory even though the address is
    # a parameter's — and less any clock symbol.
    run_constants: frozenset[str]
    # Every name the model binds to a function. A call to one hides whatever its
    # body reads from a scan of the call site, so :func:`condition_cannot_cross`
    # declines rather than guess.
    function_names: frozenset[str]


def switch_condition_scope(core, ctx=None) -> SwitchConditionScope:
    """Assemble the model-level context both switch-condition callers read.

    One RHS probe for the clocks, one pass over the parameter table, shared by
    :func:`compute_switch_time_sens` and :func:`uncompensated_condition_reason`
    so neither can be built on a different view of the model than the other.
    """
    param_names = list(core.param_names)
    is_expr = list(core.param_is_expression)
    exprs = list(core.param_expressions)
    clocks = _unit_rate_clock_species(core, ctx)
    clock_symbols = frozenset(clocks) | _TIME_SYMBOLS
    function_names = frozenset(core.function_names)
    return SwitchConditionScope(
        core=core,
        clocks=clocks,
        clock_symbols=clock_symbols,
        param_names=tuple(param_names),
        param_pats={
            n: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])") for n in param_names
        },
        primary_names=frozenset(n for i, n in enumerate(param_names) if not is_expr[i]),
        param_idx={n: i for i, n in enumerate(param_names)},
        values=tuple(core.get_param(n) for n in param_names),
        derived_exprs={
            param_names[i]: exprs[i] for i in range(len(param_names)) if is_expr[i] and exprs[i]
        },
        run_constants=frozenset(
            n
            for i, n in enumerate(param_names)
            if not is_expr[i] and n not in function_names and n not in clock_symbols
        ),
        function_names=function_names,
    )


# An identifier, excluding the exponent letter of a numeric literal (`1e5`) and
# a member-ish suffix. An atom with none of these is a comparison between
# literals — a compile-time constant, with no crossing at all.
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*")


def condition_cannot_cross(atom_flat: str, scope: SwitchConditionScope) -> bool:
    """True when *atom_flat* holds ONE truth value for the whole run (issue #382).

    *atom_flat* is one condition atom with derived-parameter references already
    inlined (:func:`_inline_derived_param_refs`), which is the form
    :func:`uncompensated_condition_reason` judges. The question here is narrower
    than "is this crossing compensated": it is whether there is a crossing **in
    the run window** at all. A comparison written over nothing but run-constants
    picks its branch before the first step and holds it to the last, so ``f`` has
    no discontinuity for the trajectory to meet, and there is no jump for issue
    #48 to stop at or issue #150 to root on. The in-branch derivative is then the
    whole story and the analytic sensitivity RHS is admissible — which is ground
    3 of :func:`uncompensated_condition_reason`, widened from the literals-only
    ``0>0`` it used to mean to the spelling models actually carry.

    The four `MODEL09112*` witnesses of issue #382 are reduced Guyton circulation
    models: most of the loop is cut away and what is left refers to the removed
    parts through frozen ``<parameter>`` declarations, so ``CRRFLX>1e-07``
    (``CRRFLX = 0``) and ``PO2ART<80.0`` (``PO2ART = 97.0439``) are conditions
    that cannot move. Declining them cost those models the analytic RHS entirely
    and handed the whole sensitivity solve to CVODES' difference quotient, which
    then failed ``CV_CONV_FAILURE`` at the first output point.

    Admissible names, and why every other one is refused:

    * a **primary parameter** that no model function owns and that is no clock —
      ``scope.run_constants``. A function's backing slot is excluded because
      ``evaluate_functions()`` rewrites it before every derivative evaluation
      (issues #227, #266), so its value moves with the trajectory even though its
      address is a parameter's;
    * a **call** to anything that is not a model function — ``if``, ``exp``,
      ``floor``. The callee is then an arithmetic builtin, and its arguments are
      scanned by this same loop, so state reached through one is still seen. A
      call to a *model* function is refused: its body is not at the call site and
      may read state this scan cannot see, which is the direction that would
      admit a crossing rather than miss one.

    Everything else declines, including a name this does not recognize at all: a
    species or observable (state, so the condition moves with the trajectory), a
    ``rate_of__`` accessor, a clock symbol or ``time`` in either spelling, and a
    derived parameter that survived inlining (its leaves are then unverified).

    **The invariant this rests on** is the SBML loader's: a parameter whose value
    can move mid-run is not left a parameter. A ``<rateRule>`` target and an
    ``<eventAssignment>`` target are both *promoted to species*
    (``rate_rule_targets`` / ``event_promoted_params`` in
    :mod:`bngsim._sbml_loader`), so they arrive here as state and decline on that
    ground rather than on a special case. What is left in ``param_names`` that
    still moves is the function's backing slot, which is why that is the one
    exclusion spelled out above.

    Structural, not numeric: the verdict reads which *names* an atom carries and
    what kind each is, never a parameter's value. So moving a rate constant
    cannot flip it, and :func:`switch_gate_cache_digest` — which exists to carry
    the value-dependent half of this gate — does not have to know about it.

    A run-constant condition sitting exactly ON its threshold (``ANPKNS>0.0``
    with ``ANPKNS = 0``, three of the four witnesses) is still admitted, and the
    column it yields is the one-sided derivative: the branch does not move for
    any perturbation on the closed side, and an infinitesimal one on the open
    side flips the whole run at once rather than moving a crossing through it.
    Nothing bngsim or AMICI carries compensates a discontinuity in *parameter*
    space — it is not a crossing in time, so no saltation jump applies — and both
    engines report the branch that is taken. What this changes is that bngsim now
    reports it from the analytic RHS instead of refusing every column in the
    model over it.
    """
    for m in _IDENTIFIER.finditer(atom_flat):
        name = m.group(0)
        if name in scope.run_constants:
            continue
        # A clock is never a run-constant, in either spelling, so it is tested
        # before the call check: `time()` is written as a call and would
        # otherwise be waved through as a builtin.
        if name in scope.clock_symbols:
            return False
        if atom_flat[m.end() :].lstrip().startswith("(") and name not in scope.function_names:
            continue
        return False
    return True


class UncompensatedCrossingReason(str):
    """A decline reason for which the difference-quotient fallback is ALSO wrong.

    Every other reason the analytic sensitivity RHS is declined for — an
    underivable rate law (#56/#66), a derivation budget (#90) — leaves CVODES'
    internal difference quotient a correct, slower answer to the same smooth
    problem. A reason from :func:`uncompensated_condition_reason` may not: what
    it reports is a *crossing*, and the difference quotient integrates the
    variational equation straight through one, missing exactly the jump the
    analytic path was declined for (issue #146).

    Issue #150 took the common case out of this class entirely, by removing the
    decline rather than re-labelling it: a condition that reduces to a single
    relational comparison over live state — ``Virus < 1`` — now has its crossing
    located as a CVODE root and its saltation jump applied there
    (:func:`state_switch_conditions`), so the in-branch derivative is again the
    whole in-branch story and the analytic RHS is admitted. Issue #382 took
    another case out the same way, from the other end: a condition written over
    run-constants alone — ``CRRFLX>1e-07`` where ``CRRFLX`` is a frozen
    ``<parameter>`` — has no crossing in the run window to compensate, so it too
    is admitted rather than declined (:func:`condition_cannot_cross`). Issue #418
    took a further case out from the clock end: a threshold that is a single power
    of the clock — ``time()*time()>=thresh`` — has its crossing solved in closed
    form (``time = sqrt(thresh)``) by :func:`_clock_monomial_threshold` and is now
    compensated by the issue #48 machinery like any affine clock threshold. What is
    left in this class is the crossing nothing compensates — a comparison inside
    a call argument, a comparison outside an ``if()`` head, or a clock threshold
    that neither reduces to a constant nor to a single clock power. A conjunction
    is not one of them, and
    neither is a negation: :func:`_split_logical_atoms` reduces both to the
    surfaces underneath, so ``not((X<hi) and (X>lo))`` is admitted on ground 2
    exactly as ``(X<hi) and (X>lo)`` is (issue #234). Nor is an *equality* over
    live state, since issue #381: ``X == 1`` bounds its own true-set with the
    surface ``X − 1 = 0``, which is the one ``X < 1`` names, so it resolves to
    the same residual and is claimed by the same root.

    A ``str`` subclass rather than a second return value because the reason is
    cached, stored in dicts and formatted at half a dozen sites between here and
    the warning; carrying the distinction on the value itself means none of them
    has to be taught to thread it, and none of them can drop it.

    :class:`DeclinedAtMovingCrossingReason` is the *other* producer of the same
    fallback-is-wrong-too verdict, arrived at from the opposite direction.
    """

    __slots__ = ()


class DeclinedAtMovingCrossingReason(UncompensatedCrossingReason):
    """A decline for an unrelated reason, on a model that HAS a moving crossing.

    The parent class is reached by asking "is this crossing compensated?" and
    answering no. This one is reached by asking nothing about the crossing at
    all: the analytic sensitivity RHS was declined because some rate law calls
    an unsupported function, or its derivative will not render as C, or the
    derivation budget expired — and the model happens also to carry a branch
    condition :func:`model_moving_crossings` recognizes. ``CVodeSensInit1`` takes
    ONE callback for every column, so that decline puts the *whole* model on the
    difference quotient, crossing included, and the fallback is then wrong for
    exactly the reason the parent class exists (issue #232).

    Kept apart from the parent only so the warning can end with the right
    sentence: what is missing here is not machinery nobody has written, it is
    the named decline above. Remove that and the analytic path — which does
    apply the jump — comes back.
    """

    __slots__ = ()


# `==`, `!=` and ExprTk's bare `=`, at paren depth 0. Longest-first so `<=` and
# `>=` are never read as a bare `=`, and `!=` is never read as a negation.
_EQUALITY_OP = re.compile(r"==|!=|(?<![<>=!])=(?!=)")


def is_equality_atom(atom: str) -> bool:
    """True when *atom*'s own comparison is an equality rather than an ordering.

    Read at depth 0, the same level :func:`_relational_split_op` splits at, so an
    equality buried inside a call argument does not count as this atom's operator.
    """
    depth = 0
    i = 0
    while i < len(atom):
        c = atom[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            m = _RELATIONAL.match(atom, i)
            if m is not None:
                return _EQUALITY_OP.fullmatch(m.group(0)) is not None
            if _EQUALITY_OP.match(atom, i) is not None:
                return True
        i += 1
    return False


def state_switch_residual(core, atom: str) -> str:
    """The residual text the solver would root on for condition *atom*, or ``""``.

    Thin, exception-swallowing wrapper over
    ``NetworkModel.state_switch_residual``. A non-empty answer means the solver
    can locate this crossing and differentiate ``dt*/dθ`` there, so the
    saltation jump ``(f⁻−f⁺)·dt*/dθ`` will be applied and the condition needs no
    warning; the string itself identifies the *crossing* rather than its
    spelling, so ``X<1`` and ``X<=1`` come back equal.

    Both callers in this module route through here — the run-time detector that
    registers the roots and the codegen gate that decides whether to warn — so
    the two cannot classify a condition differently. That is the same
    single-recognizer requirement issue #68 imposed on the clock path, for the
    same reason: a gate that disagrees with the machinery it stands in for is
    worse than no gate.
    """
    try:
        residual, _why = core.state_switch_residual(atom)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("state switch: %r could not be resolved: %s", atom, exc)
        return ""
    return residual


def _iter_condition_atoms(expr: str):
    """Every relational atom of every ``if()`` condition in *expr*."""
    for cond in _iter_if_conditions(expr):
        yield from _split_logical_atoms(cond)


# `time` written bare or as ExprTk's nullary call, as a whole word.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?:\s*\(\s*\))?(?![A-Za-z0-9_(])")

# How far off zero a residual may sit at the crossing it just predicted, as a
# fraction of the slope times the run's own time scale. This is a *linearity*
# check, not a root-finding tolerance: the two probes below already solve the
# linear case exactly, so anything failing it is a residual that is not linear
# in time, whose crossing this cannot claim to know.
_CROSSING_RESIDUAL_TOL = 1e-9

# Resolved crossing times, keyed on the condition text, the run window and the
# values of every parameter the condition reads once derived names are inlined —
# which is everything the answer depends on. Two sympy round trips per condition
# is ~2 ms, and a scan or a fit calls run() thousands of times while the
# parameters a *schedule* reads (an experimental-condition dose time) change
# once per experiment, so the hit rate is close to 1. Bounded and cleared whole
# rather than LRU-evicted: the population is tiny and the cost of a cold miss is
# the 2 ms it always was.
_CROSSING_CACHE: dict[tuple, float | None] = {}
_CROSSING_CACHE_MAX = 4096


def _time_alias_bodies(ctx) -> dict[str, str]:
    """Every function/assignment-rule name whose value is a function of time.

    ``model_time := time`` is the shape (GH #259): a condition may threshold the
    *alias* rather than the csymbol, and the alias is also a plain model
    parameter carrying a stale number. Reading that number is how a residual
    that is genuinely time-dependent comes back looking constant, so this map
    is needed twice over — to inline the aliases that resolve, and to refuse the
    ones that do not.

    Transitive: an alias of an alias reads time too.
    """
    bodies = dict(ctx["function_map"])
    aliases: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name, body in bodies.items():
            if name in aliases or not isinstance(body, str):
                continue
            if _TIME_REF.search(body) or any(
                re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}(?![A-Za-z0-9_])", body)
                for a in aliases
            ):
                aliases[name] = body
                changed = True
    return aliases


def _inline_time_aliases(expr: str, aliases: dict[str, str]) -> str:
    """Substitute every *bare* reference to a time alias by its body.

    Bare only: ``f(x)`` is a call whose body takes an argument, and pasting the
    body over the call site would drop the argument. A call to a time-dependent
    function therefore survives the substitution and is caught by the caller's
    identifier check, which is the conservative direction.
    """
    for _ in range(len(aliases) + 1):
        before = expr
        for name, body in aliases.items():
            expr = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_(])", f"({body})", expr
            )
        if expr == before:
            break
    return expr


def _crossing_time_of_condition(
    cond: str,
    scope: SwitchConditionScope,
    t_start: float,
    t_end: float,
    aliases: dict[str, str] | None = None,
) -> float | None:
    """The model time at which *cond* flips, or ``None`` when it is not a fixed
    time crossing this function can resolve exactly.

    *cond* is one registered GH #72 discontinuity trigger, verbatim — the text
    the root function evaluates, not a re-derivation of it. Resolving it means
    answering "at what t does the sign of ``lhs − rhs`` change", and the answer
    has to be a constant of the run: a condition reading live state
    (``time < S1``) has no such answer and is declined, as is one whose residual
    is not linear in time.

    Deliberately *not* routed through :func:`_clock_threshold_splits`, which
    requires a **bare** clock symbol on one side and would decline the shape
    this issue is about (``(time - PdBu_time) < 0``, the PEtab spelling of
    ``time < PdBu_time``). That function answers a different question — whether
    a crossing's ``∂t*/∂p`` can be chain-ruled to fitted primaries — and the
    #259 lesson is that root registration and threshold *recognition* must not
    be held to the same spelling: registration already admits either side being
    time-dependent, and a crossing that is registered but unresolvable here is
    exactly the one that wedges.

    Two probes decide it. The residual is evaluated at two times, which solves
    the linear case exactly, and then re-evaluated at the predicted crossing:
    a residual that is not linear in time fails that check and is declined
    rather than stopped at in the wrong place.
    """
    # The registered text is a whole condition, so it arrives wrapped —
    # `((time()-PdBu_time)<0)` — and the relational splitter only matches at
    # paren depth 0.
    split = _relational_split(_strip_redundant_parens(cond.strip()))
    if split is None:
        return None
    lhs, rhs = split
    aliases = aliases or {}
    # An alias of `time` is a plain parameter as far as the tables below are
    # concerned, and carries a stale number, so it has to be substituted BEFORE
    # anything reads a value — otherwise `model_time >= 0.7` looks constant.
    # Substituting also keeps the #259 property that the alias spelling and the
    # csymbol spelling are the same run to the last bit.
    residual = _inline_time_aliases(f"({lhs})-({rhs})", aliases)
    flat = _inline_derived_param_refs(residual, scope.derived_exprs) or residual
    if not _TIME_REF.search(flat):
        return None
    # Every other identifier must be a model parameter, and must not be one of
    # the time-dependent names — a surviving alias is one this could only read
    # as a constant, and a species or observable name means the crossing moves
    # with the trajectory (issue #150's business, not a fixed stop time).
    read: list[str] = []
    for m in _IDENTIFIER.finditer(_TIME_REF.sub(" 0 ", flat)):
        if m.group(0) not in scope.param_idx or m.group(0) in aliases:
            return None
        read.append(m.group(0))

    key = (
        cond,
        t_start,
        t_end,
        tuple(sorted((n, scope.values[scope.param_idx[n]]) for n in set(read))),
    )
    if key in _CROSSING_CACHE:
        return _CROSSING_CACHE[key]

    def at(t: float) -> float | None:
        return _evaluate_threshold(
            _TIME_REF.sub(f"({t!r})", residual), scope.param_idx, scope.values, scope.derived_exprs
        )

    def resolve() -> float | None:
        scale = max(abs(t_start), abs(t_end), 1.0)
        r0, r1 = at(0.0), at(scale)
        if r0 is None or r1 is None:
            return None
        slope = (r1 - r0) / scale
        if slope == 0.0 or not (abs(slope) < float("inf")):
            return None  # `time` cancels out: no crossing to stop at
        t_star = -r0 / slope
        check = at(t_star)
        if check is None or abs(check) > _CROSSING_RESIDUAL_TOL * abs(slope) * scale:
            logger.debug(
                "crossing stop: %r is not linear in time (residual %r at t=%r); skipping",
                cond,
                check,
                t_star,
            )
            return None
        return t_star

    answer = resolve()
    if len(_CROSSING_CACHE) >= _CROSSING_CACHE_MAX:
        _CROSSING_CACHE.clear()
    _CROSSING_CACHE[key] = answer
    return answer


def _blank_clock_refs(text: str, scope: SwitchConditionScope) -> str:
    """*text* with every clock reference replaced by a literal, so what is left
    is the things the clock is being compared against."""
    blanked = _TIME_REF.sub(" 0 ", text)
    for sym in scope.clocks:
        blanked = _clock_symbol_sub(blanked, sym, " 0 ")
    return blanked


def _switches_on_clock_alone(atom: str, scope: SwitchConditionScope) -> bool:
    """True when *atom* compares a clock against things that never move.

    The admission rule for :func:`time_discontinuity_conditions` (issue #440),
    and it is deliberately narrow. ``time() >= 100`` and ``time() - 24*floor(
    time()/24) >= 7`` qualify; ``S1 > 0.5`` does not, because a state threshold
    crosses at a time no one knows in advance; and ``time() < S1`` does not
    either, for the same reason written the other way round.

    A *clock* is literal simulation time or a counter species: one fed by a
    zeroth-order reaction at rate 1 and read back through a group, which is how
    a BNGL model makes time available to a rate law (issue #443). The two are
    interchangeable here because a counter's value is time plus a constant
    offset, so the time at which it reaches a threshold is just as knowable:
    ``t_start + threshold − c(t_start)``, the conversion issue #48 already
    makes. Which one an atom is written against is recorded, because a stop
    placed on a counter needs the counter landed exactly on its threshold and a
    stop placed on literal time does not — see :func:`fixed_crossing_stops`.

    "Never move" is :attr:`SwitchConditionScope.run_constants` — primary
    parameters, less the slots a model *function* owns, whose value is rewritten
    from the function body before every derivative evaluation and so tracks the
    trajectory. Derived parameters are inlined first, so ``time() >= onset``
    with ``onset = t0 + delay`` is read as the comparison against ``t0 + delay``
    that it is.

    Orderings only. An equality is excluded, which is what the SBML scan does
    too — see the comment on the check.
    """
    if is_equality_atom(atom):
        # `time() == T` is true for one instant of measure zero, so its branch
        # contributes nothing to the integral and there is nothing to miss.
        # Stopping there would be worse than not: the step that restarts AT the
        # crossing reads the rate law where the equality holds and carries that
        # value forward over a whole step. The SBML scan registers only
        # orderings for the same reason.
        return False
    flat = _inline_derived_param_refs(atom, scope.derived_exprs) or atom
    if _clock_free(flat, scope.clock_symbols):
        return False
    blanked = _blank_clock_refs(flat, scope)
    for m in _IDENTIFIER.finditer(blanked):
        name = m.group(0)
        if name in scope.function_names:
            # A call to a model function that inlining left standing. Its body
            # can read anything, so the atom is only as knowable as the body is,
            # and this declines rather than guess.
            return False
        if blanked[m.end() :].lstrip().startswith("("):
            # An engine built-in — `floor(`, `min(`, `exp(`. ExprTk compiles a
            # call only to one of those or to a model function, and the model
            # functions were just excluded, so what is left is arithmetic over
            # arguments this same loop goes on to read.
            continue
        if name in scope.run_constants or name in _BUILTIN_CONSTANT_VALUES:
            continue
        return False
    return True


def time_discontinuity_conditions(core, ctx=None) -> tuple[str, ...]:
    """Every rate-law branch condition this model switches on a *clock* alone with.

    The ``.net``/BNGL answer to the question the SBML loader answers at load
    time by walking the libSBML tree and calling ``add_discontinuity_trigger``
    (issue #440). A BNGL model is built entirely in C++, so there is no
    build-time seam to register a root at; what there is instead is the run-time
    stop time (:func:`fixed_crossing_stops`,
    ``SolverOptions.set_crossing_stops``), which lands the step exactly on
    the crossing and restarts there. That is the half of issue #305 that does
    the work here.

    Why any of it is needed: inside each branch of ``if(time() >= 100, k, 0)``
    the right-hand side is a constant, so CVODE's local error estimate over a
    step that spans the whole branch is near zero and nothing stops the step
    from growing until it swallows the window. The reported trajectory is then
    the one where the branch never turned on. Tightening ``rtol`` does not help,
    because there is no error to see.

    The scan mirrors :func:`model_moving_crossings` — same texts, same inlining,
    same atom split — so a threshold written inside a called function definition
    is found under its call site. It differs only in which atoms it keeps:
    :func:`_switches_on_clock_alone`, which is the set whose crossing times are
    knowable before the run rather than the set whose crossing times move.

    Empty for a model with no functions, and for the far more common model whose
    conditions read state rather than a clock, so nothing about its stepping
    changes.
    """
    from bngsim._jacobian import _inline_functions, has_condition_construct

    if core.n_functions == 0:
        return ()
    if ctx is None:
        # The raw texts first, which is the cheap half: a model with no
        # conditional rate law, and no clock for one to threshold, short-circuits
        # here rather than paying for ``functional_jacobian_context()``, whose
        # function_map is built from every function the model has and runs to
        # tens of thousands of entries on a genome-scale one. Neither inlining a
        # function nor inlining a derived parameter can introduce a conditional
        # or a ``time`` that none of these texts already spells, which is why
        # both halves are sound read here — the same argument the issue #333
        # guard makes for reading ``function_expressions`` instead of the
        # context. Measured on this repository's ``.net`` corpus: 80 of 585
        # models carry a conditional rate law, so the second half is what keeps
        # a 43 ms scan off most of them.
        #
        # A counter clock cannot be recognized from the texts, because what
        # names it is the shape of the model's right-hand side rather than
        # anything the condition spells: the group is called ``t`` in one model
        # and ``Time_`` in the next (issue #443). Two evaluations of the RHS
        # answer it instead, which is far cheaper than the context this is
        # guarding, and the answer is reused by ``switch_condition_scope``
        # below.
        texts_raw = (*core.function_expressions, *core.param_expressions)
        if not any(has_condition_construct(t) for t in texts_raw):
            return ()
        if not any(_TIME_REF.search(t) for t in texts_raw) and not _unit_rate_clock_indices(core):
            return ()
        ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    texts = [
        *func_map.values(),
        *(str(r.get("rate_expr", "")) for r in ctx["functional_reactions"]),
    ]
    conditional = [t for t in texts if has_condition_construct(t)]
    if not conditional:
        return ()
    try:
        scope = switch_condition_scope(core, ctx)
    except Exception as exc:  # pragma: no cover - defensive
        # Same fallback as the other model-level scans: with no scope nothing
        # can be classified, and the pre-#440 stepping is what a model gets.
        logger.debug("time-discontinuity scan: scope unavailable (%s)", exc)
        return ()

    found: list[str] = []
    for text in conditional:
        flat = _inline_functions(text, func_map) or text
        for atom in _iter_condition_atoms(flat):
            if atom not in found and _switches_on_clock_alone(atom, scope):
                found.append(atom)
    return tuple(found)


# Resolved schedule edge lists, keyed the same way :data:`_CROSSING_CACHE` is:
# on the condition text, the run window, and the value of every parameter the
# condition reads once derived names are inlined. Recognizing a schedule and
# then checking it against the model's own residual is seven sympy round trips,
# which is 7 ms on a model spelling 38 conditions and is paid on every ``run()``
# — so a fit paid it on every evaluation. Everything a schedule's answer depends
# on is in the key, and the parameters a schedule reads (a dose period, a
# stimulus start) change once per experiment rather than once per evaluation, so
# the hit rate is close to 1.
#
# The cached list is never handed out for mutation: the one caller iterates it.
_SCHEDULE_CACHE: dict[tuple, list[float] | None] = {}
_SCHEDULE_CACHE_MAX = 4096


def _schedule_stop_times(
    cond: str, scope: SwitchConditionScope, t_start: float, t_end: float
) -> list[float] | None:
    """Every edge of a repeating schedule in ``(t_start, t_end]``, or ``None``
    when *cond* is not one this can place edges for (issue #440).

    :func:`_crossing_time_of_condition` solves a residual that is linear in
    time, which is one crossing. A schedule — ``time() - 24*floor(time()/24) >=
    7``, the light-and-dark cycle and the repeated-dose idiom — has one in every
    period, and its residual is a sawtooth that no two probes can solve. Issue
    #436 already recognizes the pattern for forward sensitivity and enumerates
    its edges from the period, the offset and the duty; this is the same
    enumeration read for the far simpler purpose of stopping the step at each
    one.

    Only a schedule over literal simulation time. A counter clock reaches this
    already rewritten into one by :func:`_rewrite_counter_clock`, so the two
    spellings arrive here as the same text and there is nothing here that has to
    know which one the model wrote (issue #443).

    Read for a condition the SBML loader registered as well as for one derived
    from a BNGL function body. The registered CVODE root does not cover a
    repeating schedule: it is evaluated on the boolean, which reads the same on
    both sides of a step spanning a whole period.

    The chain rule ``_periodic_schedule_terms`` computes is not needed here — a
    stop carries no ``∂t*/∂p`` — so the numbers are read straight through
    :func:`_evaluate_threshold`. The residual round-trip is kept: it is what
    catches a schedule sympy's parser mis-read (a parameter named ``I`` folding
    ``I*I`` to ``-1``), and placing stops where the model has no edge would be a
    pure perturbation of its stepping.
    """
    # A condition can arrive wrapped — `((time()-P*floor(time()/P))>=D)` is how
    # the SBML loader registers one — and the recognizer, like the relational
    # splitter under it, only reads its operator at paren depth 0. Stripping
    # first is the same thing :func:`_crossing_time_of_condition` does with the
    # same text, and it is what lets an SBML model be read at all: without it
    # every registered schedule declines for a reason that is about a paren
    # rather than about the schedule. Done before the memo key is built, so two
    # spellings of one condition share a cache entry.
    atom = _strip_redundant_parens(cond.strip())
    read = sorted(
        {
            m.group(0)
            for m in _IDENTIFIER.finditer(
                _inline_derived_param_refs(atom, scope.derived_exprs) or atom
            )
            if m.group(0) in scope.param_idx
        }
    )
    key = (
        atom,
        t_start,
        t_end,
        tuple((n, scope.values[scope.param_idx[n]]) for n in read),
    )
    if key in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[key]
    answer = _resolve_schedule_stop_times(atom, scope, t_start, t_end)
    if len(_SCHEDULE_CACHE) >= _SCHEDULE_CACHE_MAX:
        _SCHEDULE_CACHE.clear()
    _SCHEDULE_CACHE[key] = answer
    return answer


def _resolve_schedule_stop_times(
    atom: str, scope: SwitchConditionScope, t_start: float, t_end: float
) -> list[float] | None:
    """:func:`_schedule_stop_times` without the memo, on an already-unwrapped atom."""
    sched = _clock_periodic_schedule(atom, scope.clock_symbols, scope)
    if sched is None or sched.clock not in _TIME_SYMBOLS:
        return None
    values: list[float] = []
    for text in (sched.period, sched.offset, sched.duty):
        value = _evaluate_threshold(text, scope.param_idx, scope.values, scope.derived_exprs)
        if value is None or not (abs(value) < float("inf")):
            return None
        values.append(float(value))
    period, offset, duty = values
    if period == 0.0:
        return None
    # A duty that does not land strictly inside the period is a condition that
    # holds one truth value forever — `rem(time(), P) >= 0` is a corpus model —
    # so there is no edge to stop at. `duty/period` rather than `0 < duty <
    # period` so a `ceil()` spelling, whose recognized period is negative, is
    # judged the same way.
    if not 0.0 < duty / period < 1.0:
        return None
    if not _schedule_matches_residual(atom, sched, period, duty, offset, True, scope):
        return None
    terms = ScheduleTerms(
        period=period,
        offset=offset,
        duty=duty,
        d_period={},
        d_offset={},
        d_duty={},
        crosses=True,
    )
    edges = _schedule_edges(terms, t_start, t_end, _SCHEDULE_EDGE_BUDGET)
    if edges is None:
        # Over budget. A plain run has no gradient to be quietly wrong, but it
        # does have a trajectory, and this is the one case where bngsim knows
        # the schedule is there and cannot stop at it — so say so rather than
        # leave the stepping to be discovered wrong later.
        logger.warning(
            "Periodic schedule %r has more than %d edges between t=%r and t=%r; the "
            "integrator will step over them unclamped and may miss whole windows "
            "(issue #440). Pass max_step to bound the step instead.",
            atom,
            _SCHEDULE_EDGE_BUDGET,
            t_start,
            t_end,
        )
        return None
    return [value for value, _partials in edges]


class CrossingStop(NamedTuple):
    """One model time the integrator has to land exactly on (issues #305, #443).

    ``clock_species_idx`` is the 0-based index of the counter species whose
    threshold this crossing is, or ``-1`` when the condition reads literal
    simulation time. ``threshold`` is the value that counter holds at ``time``,
    and is meaningless (0.0) for a literal-time crossing.

    The counter fields are what separate this from a bare list of times, and
    they exist because a counter is *integrated*. Stopping the step at ``time``
    puts the clock at ``threshold`` to within the integrator's own error, which
    is a couple of parts in 1e14 short of it — and on the short side the
    condition is still false, so the run restarts on the branch that just ended
    and meets the discontinuity inside the first step after a restart that has
    no history to fall back on. That is issue #82, and the core repairs it by
    setting the counter to its exact value at the crossing before it restarts.
    Literal simulation time needs none of this, because the stop is computed
    from the same clock the condition reads.
    """

    time: float
    clock_species_idx: int
    threshold: float


def _rewrite_counter_clock(
    core, cond: str, scope: SwitchConditionScope, t_start: float
) -> tuple[str, int, float] | None:
    """*cond* with its counter clock rewritten as an expression in simulation
    time, plus the counter's species index and its offset from time.

    A counter obeys ``dc/dt = 1``, so ``c(t) = t + (c(t_start) − t_start)`` for
    the whole of the run: one substitution turns a condition on the counter into
    the condition on time that it already is, and every resolver below can then
    read it without knowing a counter was ever involved. That is the whole of
    what issue #443 needs on this side — ``t >= 100`` on a counter starting from
    0 becomes ``time() + 0.0 >= 100``, and the linear solve places the stop at
    100 exactly as it would for the literal spelling.

    ``(cond, -1, 0.0)`` for a condition that reads literal time and no counter,
    which leaves it untouched. ``None`` where no stop can be placed at all: a
    condition reading two different counters, since both would have to be landed
    on their own threshold at the stop and the record carries one (no model in
    this repository's corpus writes it), or a counter whose value is not finite.
    """
    syms = [sym for sym in scope.clocks if not _clock_free(cond, {sym})]
    if not syms:
        return cond, -1, 0.0
    indices = {scope.clocks[sym] for sym in syms}
    if len(indices) != 1:
        logger.debug("crossing stop: %r reads more than one counter clock; skipping", cond)
        return None
    idx = next(iter(indices))
    offset = float(core.get_concentration(core.species_names[idx])) - t_start
    if not math.isfinite(offset):
        # A counter seeded nan or inf (issue #353 leaves a nan concentration
        # rather than refusing the model). Every crossing time computed from it
        # would be nan, and a stop at nan is not a stop.
        logger.debug("crossing stop: counter clock for %r reads %r; skipping", cond, offset)
        return None
    rewritten = cond
    for sym in syms:
        rewritten = _clock_symbol_sub(rewritten, sym, f"(time()+({offset!r}))")
    return rewritten, idx, offset


def fixed_crossing_stops(core, t_start: float, t_end: float, conditions=()) -> list[CrossingStop]:
    """Crossings in ``(t_start, t_end]`` at which a registered time discontinuity
    flips, for ``SolverOptions.set_crossing_stops`` (issue #305).

    The core stops the integration step exactly on each of these. That is not a
    refinement of the GH #72 root — it is what makes the root reachable at all.
    CVODE tests for a root only on a step it **accepts**, and where the branch
    jump is large enough that the error test rejects every step spanning the
    crossing, the accepted steps land short, ``t`` creeps to the last double
    below ``t*``, and every remaining step is under one ulp: ``t + h == t``,
    with ``g`` never once evaluated past the crossing. On Weber_BMC2015 that
    kills 6% of a fitting box outright, with zero root returns in the whole run.

    Since issue #440 a stop is also what a ``.net``/BNGL model gets *instead* of
    a root: those models are built entirely in C++, with no build-time seam for
    the loader to register one at, and stopping the step on the crossing and
    reinitialising there is the whole of what the root would have bought. So
    ``conditions`` is the gate rather than the registered-root count — the SBML
    loader hands over the set it registered, and
    :func:`time_discontinuity_conditions` derives the same thing for a model
    whose loader could not.

    Two kinds of crossing are placed. A residual linear in time
    (:func:`_crossing_time_of_condition`) has one, solved exactly. A repeating
    schedule (:func:`_schedule_stop_times`) has one per period, enumerated from
    the pattern issue #436 recognizes.

    Either may be written against a counter species rather than against literal
    simulation time, which is the spelling BNGL models actually use and issue
    #443 is about: 37 of the 585 ``.net`` models in this repository's corpus
    threshold a counter, against none that threshold ``time()``.
    :func:`_rewrite_counter_clock` turns such a condition into the condition on
    time it already is before either resolver sees it, so neither of them, and
    nothing below them, needs a second code path.

    A schedule is placed whether the condition was registered by the SBML loader
    or derived from a BNGL function body, because the registered root does not
    cover it. The root is evaluated on the *boolean*, and the boolean of a
    repeating schedule reads the same on both sides of a step that spans a whole
    period, so there is no sign change for the root finder to see — which is the
    same reason issue #440 needed stops rather than roots in the first place. A
    ``piecewise`` on ``time - 24*floor(time/24) >= 7`` in SBML reports 10.6 on
    the accumulator issue #440 uses, where the answer is 17.

    Empty (so: no change to any model's stepping) unless some condition's
    crossing time is a constant of the run. Resolution reads the *current*
    parameter values, so a condition is answered for the phase it is asked in —
    the same experimental-condition parameter can put the crossing inside the
    window in one phase and outside it in another, and stopping at a time that
    phase has no crossing at is a pure perturbation of its stepping.
    """
    if not conditions:
        return []
    ctx = core.functional_jacobian_context()
    scope = switch_condition_scope(core, ctx)
    aliases = _time_alias_bodies(ctx)
    out: list[CrossingStop] = []
    for cond in conditions:
        rewrite = _rewrite_counter_clock(core, cond, scope, t_start)
        if rewrite is None:
            continue
        text, clock_idx, offset = rewrite
        t_star = _crossing_time_of_condition(text, scope, t_start, t_end, aliases)
        times: list[float] | None = None
        if t_star is not None:
            times = [t_star]
        else:
            times = _schedule_stop_times(text, scope, t_start, t_end)
        for t_cross in times or ():
            if not (t_start < t_cross <= t_end):
                continue
            # The counter's exact value at the crossing. Reading it back off the
            # rewrite rather than off the recognized threshold text is what keeps
            # the two in step whichever resolver placed the stop, and for a
            # schedule there is no single threshold text to read.
            stop = CrossingStop(t_cross, clock_idx, t_cross + offset if clock_idx >= 0 else 0.0)
            near = [
                i
                for i, seen in enumerate(out)
                if abs(t_cross - seen.time) <= 1e-12 * max(abs(t_cross), 1.0)
            ]
            if not near:
                out.append(stop)
            elif clock_idx >= 0 and out[near[0]].clock_species_idx < 0:
                # Two conditions crossing at one instant, one on a counter and
                # one on literal time. Keep the counter record: it does
                # everything the plain one does and also lands the counter on
                # its threshold, which the plain one would leave a couple of ulp
                # short and the counter's own condition reading false.
                out[near[0]] = stop
    out.sort()
    return out


def fixed_time_crossings(core, t_start: float, t_end: float, conditions=()) -> list[float]:
    """Just the times from :func:`fixed_crossing_stops`, in order."""
    return [stop.time for stop in fixed_crossing_stops(core, t_start, t_end, conditions)]


def fixed_clock_threshold(atom: str, scope: SwitchConditionScope) -> bool:
    """True when *atom* is a clock threshold against a value nothing moves.

    ``t < 14`` — or ``t < half_life`` where ``half_life`` reduces to literals —
    crosses at a time that is the same for every parameter, so ``∂t*/∂p`` is
    exactly 0 and the crossing contributes no jump to any sensitivity column.
    That is what makes it harmless twice over: :func:`clock_crossing_compensated`
    admits it because there is nothing to compensate, and
    :func:`model_moving_crossings` excludes it because there is nothing for the
    difference-quotient fallback to miss either. One definition, two readers, so
    they cannot answer the same question differently (issue #232) — the first of
    them now through :func:`_threshold_crossing_terms`, which asks it of one
    crossing at a time because an atom can have two (issue #421) and they need
    not be alike: ``(time()-5)*(time()-thresh) >= 0`` has one edge nothing moves
    and one every fitted run does.
    """
    split = _clock_threshold_splits(atom, scope.clock_symbols)
    if split is None:
        # A repeating schedule (issue #436) whose period, offset and duty are all
        # literal has an edge at a fixed time in every period, so no parameter
        # moves any of them either. ``rem(time(), 24) >= 7`` — the light and dark
        # cycle six corpus models write — is the case.
        sched = _clock_periodic_schedule(atom, scope.clock_symbols, scope)
        if sched is None:
            return False
        return all(
            _fixed_threshold_expr(text, scope) for text in (sched.period, sched.offset, sched.duty)
        )
    # EVERY crossing the atom has must be fixed: a threshold with two of them
    # (issue #421) is only free of a jump if no parameter moves either one.
    return all(_fixed_threshold_expr(thr, scope) for thr in split[1])


def _threshold_has_no_parameter(threshold_expr: str, scope: SwitchConditionScope) -> bool:
    """True when no model parameter appears in *threshold_expr* once derived
    references are inlined."""
    thr_flat = _inline_derived_param_refs(threshold_expr, scope.derived_exprs) or threshold_expr
    return not any(scope.param_pats[n].search(thr_flat) for n in scope.param_names)


def _fixed_threshold_expr(threshold_expr: str, scope: SwitchConditionScope) -> bool:
    """True when this one crossing sits at the same time whatever is fitted: no
    model parameter appears in the threshold, **and** the threshold evaluates to
    a number.

    The second half is not redundant with the first. ``time() < END_M`` where
    ``END_M`` is a *species* names no parameter, so the text scan alone called it
    fixed — and BIOMD0000000675 is what that costs: the gate admitted the model
    on the ground that its three clock crossings do not move, the issue #48
    detector then emitted no record for them because the threshold does not
    evaluate to a constant, and the issue #150 state root stood off because the
    clock path had claimed them. Three crossings that move with the trajectory,
    compensated by nobody, with no warning. Requiring the threshold to evaluate
    is what tells a literal (``t < 14``, genuinely fixed) from a name this cannot
    read (which is live state, and issue #150's).
    """
    if not _threshold_has_no_parameter(threshold_expr, scope):
        return False
    return (
        _evaluate_threshold(threshold_expr, scope.param_idx, scope.values, scope.derived_exprs)
        is not None
    )


def model_moving_crossings(core, ctx=None) -> tuple[str, ...]:
    """Every rate-law branch condition in the model whose crossing *time* moves.

    The question the decline warning has to ask (issue #232). A model carrying
    one of these is a model for which declining the analytic sensitivity RHS is
    not a free choice: CVODES' internal difference quotient integrates the
    variational equation smoothly through a crossing, so it drops the jump
    ``(f⁻−f⁺)·dt*/dθ`` outright — and for a *state* threshold it is worse than
    that, because its probe evaluates ``f`` at ``y + σ·s``, which just past the
    surface lands on the other branch (the note in
    :func:`uncompensated_condition_reason` works that through). Measured on
    issue #232's two-spelling reproduction: 53 % error at ``rtol=1e-8``, 273
    steps against 179, and no result at all below ``rtol=1e-9``, on a model whose
    analytic RHS is right to 2e-10 at every tolerance. So whenever this returns
    something, the warning must stop calling the fallback "correct, but slower".

    Deliberately coarse, in the safe direction: it asks only whether an atom
    *can* cross at a moving time, never whether anything compensates the
    crossing. Two grounds exclude an atom, both borrowed from
    :func:`uncompensated_condition_reason` so the two agree about what is not a
    crossing at all — a comparison naming no symbol (``0>0``, decided at load),
    and a :func:`fixed_clock_threshold` (``t<14``, whose ``∂t*/∂p`` is exactly
    0). Everything else reads live state or a parameter, so some θ moves it.

    It follows that an atom here may be one the analytic path *would* have
    compensated — issue #48's clock jump, issue #150's saltation jump. That is
    the point: this is about what happens once the model is on the fallback,
    where neither jump is applied.

    The scan mirrors :func:`switch_gate_cache_digest` — same ``has_condition_construct``
    pre-gate over the same function bodies and functional rate expressions, and
    the same inlining before the atoms are read — so a threshold written inside
    a called function definition is found under its call site.
    """
    from bngsim._jacobian import _inline_functions, has_condition_construct

    if core.n_functions == 0:
        return ()
    if ctx is None:
        ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    texts = [
        *func_map.values(),
        *(str(r.get("rate_expr", "")) for r in ctx["functional_reactions"]),
    ]
    conditional = [t for t in texts if has_condition_construct(t)]
    if not conditional:
        return ()
    try:
        scope = switch_condition_scope(core, ctx)
    except Exception as exc:  # pragma: no cover - defensive
        # Same fallback as the gate's: with no scope nothing can be classified,
        # and reporting a crossing we cannot name would not help anyone.
        logger.debug("moving-crossing scan: scope unavailable (%s)", exc)
        return ()

    found: list[str] = []
    for text in conditional:
        flat = _inline_functions(text, func_map) or text
        for atom in _iter_condition_atoms(flat):
            if atom in found:
                continue
            atom_flat = _inline_derived_param_refs(atom, scope.derived_exprs) or atom
            if not _IDENTIFIER.search(atom_flat):
                continue
            if fixed_clock_threshold(atom, scope):
                continue
            found.append(atom)
    return tuple(found)


def model_uncompensated_crossing_reason(core, ctx=None) -> UncompensatedCrossingReason | None:
    """The first rate-law branch crossing this model leaves uncompensated, or ``None``.

    The model-level twin of the per-rate-law gate codegen runs
    (:func:`uncompensated_condition_reason`, at its call site in
    :func:`bngsim._codegen._functional_rate_law_partials`): it scans every
    condition-bearing **reaction rate expression** — inlined exactly as codegen
    inlines it — and returns the reason the first one carries that no machinery
    can bracket (a comparison outside an ``if()`` head or buried in a call
    argument, or a clock threshold that neither reduces to a constant nor to a
    single clock power — an equality over live state is compensated since #381,
    and a clock monomial since #418). ``None`` means every crossing this model
    has is one issue #48 stops at or issue #150 roots on, so its jump is applied
    at run time — by
    :meth:`Simulator._apply_switch_time_sens` /
    :meth:`Simulator._apply_state_switch_sens` — even when the analytic
    sensitivity RHS is declined and the run is on CVODES' difference quotient, and
    nothing is dropped.

    This is the exact fact issue #414's refusal keys on, through the same
    recognizer codegen declines with, so the run-time gate and the build cannot
    disagree about which crossings are compensated. It differs from
    :func:`model_moving_crossings` in two ways the refusal needs and the warning
    does not. It reports only the crossings nothing brackets (that function is
    coarse in the safe direction and reports compensated crossings too — right for
    a warning, wrong for a refusal, because a compensated crossing on the
    difference quotient still gets its jump). And it scans **only** reaction rate
    expressions, not every function body: ``∂f/∂p`` is declined only over a rate
    law, so a condition living in an observable or expression function that no
    reaction uses as its rate law — ``if_fn() = if(A_obs>1, A_obs, 0)`` reported
    as ``expression:if_fn`` — is not a rate-law crossing and must not refuse the
    run (its own output-sensitivity request is refused on its own terms, GH #198).
    """
    from bngsim._jacobian import _inline_functions, has_condition_construct

    if core.n_functions == 0:
        return None
    if ctx is None:
        ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    # Reaction rate expressions only — the exact texts _functional_dfdp_terms
    # differentiates (frxn["rate_expr"] per functional reaction). A function that
    # no rate law reaches is inert here, so it is never scanned. Inlined BEFORE
    # the condition test and the recognizer, exactly as codegen inlines it: a
    # rate law written as a bare function name — or one nesting the condition a
    # level down — only reveals its ``if()`` after inlining.
    texts = [str(r.get("rate_expr", "")) for r in ctx["functional_reactions"]]
    flats = [_inline_functions(t, func_map) or t for t in texts]
    conditional = [f for f in flats if has_condition_construct(f)]
    if not conditional:
        return None
    try:
        scope = switch_condition_scope(core, ctx)
    except Exception as exc:  # pragma: no cover - defensive
        # Same fallback as the gate's and model_moving_crossings': with no scope
        # nothing can be classified, so report no uncompensated crossing rather
        # than refuse a run over a crossing we cannot actually name.
        logger.debug("uncompensated-crossing scan: scope unavailable (%s)", exc)
        return None
    for flat in conditional:
        reason = uncompensated_condition_reason(flat, scope)
        if reason is not None:
            return reason
    return None


def clock_crossing_compensated(atom: str, scope: SwitchConditionScope) -> bool:
    """Will :func:`compute_switch_time_sens` account for every one of *atom*'s
    crossings?

    True when the atom is a clock threshold and each crossing it has leaves the
    issue #48 machinery nothing for anyone else to do — the three grounds
    :func:`_threshold_crossing_terms` names, which are that the crossing resolves
    (the detector emits a record, the solver stops at ``t*`` and applies
    ``(f⁻−f⁺)·∂t*/∂p`` there), that nothing moves it (``t<14``, so ``∂t*/∂p`` is
    exactly 0 and there is no jump to make), or that it does not happen at all
    (a root off the real line).

    True as well for a repeating schedule — ``time() - 24*floor(time()/24) >= 7``
    — whose period, offset and duty each pass the same test (issue #436). A
    schedule has a crossing in every period rather than a fixed number of them,
    so what the detector emits for it depends on the run window; whether each of
    those crossings is compensated does not, which is why this predicate can
    still answer without one.

    False for a clock threshold whose threshold does not reduce to a constant
    over the primaries — the detector would silently skip that crossing — and
    for anything that is neither a clock threshold nor a schedule.

    This is the predicate that keeps the clock path and the state path from
    fighting over the same crossing. A BNGL counter clock is a *species*, so
    ``t >= sigma`` reads live state and :func:`state_switch_residual` would
    happily claim it; letting both claim it would apply the jump twice. Asked
    first, in both the gate and the run-time detector, so the two cannot split
    the difference.
    """
    if _clock_guard_cannot_cross(atom, scope):
        # Not a crossing at all: the atom reads the clock only through `sign()`,
        # so it holds one truth value for the whole run (issue #465). Asked
        # before the recognizers because none of them would claim it and the
        # gate would refuse the model over it.
        return True
    split = _clock_threshold_splits(atom, scope.clock_symbols)
    if split is None:
        # Asked last, so an atom any polynomial recognizer claims keeps the path
        # and the threshold text it had before issue #436.
        return _schedule_compensated(atom, scope)
    # EVERY crossing the atom has has to be accounted for. A quadratic threshold
    # (issue #421) has two, and compensating one of them while the other flips
    # the branch unjumped is the same silent zero as compensating neither.
    # Judged one crossing at a time rather than through
    # :func:`fixed_clock_threshold`, because the two roots need not be alike:
    # ``(time()-5)*(time()-thresh) >= 0`` has one crossing nothing moves and one
    # every fitted run does, and the atom is compensated on both counts.
    return all(_threshold_compensated(thr, scope) for thr in split[1])


class CrossingTerms(NamedTuple):
    """What one crossing time of a clock threshold contributes.

    ``value`` is the clock value at the crossing, or ``None`` for a root that
    does not occur at this parameter point (a non-real one). ``partials`` is
    ``∂threshold/∂primary`` over the primaries with a non-zero partial, empty for
    a crossing nothing moves.
    """

    partials: dict[str, float]
    value: float | None


def _threshold_crossing_terms(
    threshold_expr: str, scope: SwitchConditionScope
) -> CrossingTerms | None:
    """``CrossingTerms`` for one crossing time, or ``None`` when nothing
    compensates it.

    The one rule both the gate (:func:`clock_crossing_compensated`) and the
    detector (:func:`compute_switch_time_sens`) read, so they cannot answer
    differently about a crossing — the issue #68 invariant, now that an atom can
    have more than one (issue #421). Three ways to be compensated:

    * nothing moves this crossing, so ``∂t*/∂p`` is exactly 0 and there is no jump
      to make — :func:`fixed_clock_threshold` asked of one root instead of the
      whole atom, because two roots of one atom need not be alike;
    * the threshold reduces to the primaries and evaluates, so the detector emits
      a record and the solver stops there;
    * the threshold resolves to a number **off the real line**, which is a root
      the run never reaches: there is no branch flip and nothing to compensate.
      ``time()^2 >= thresh`` at ``thresh = -4`` crosses nowhere — the condition is
      simply true throughout — and the quadratic formula puts a whole region of
      parameter space in that case, since the discriminant of
      ``(time()-5)^2 >= thresh`` goes negative as soon as ``thresh`` does. Reading
      that as "unreadable threshold" refuses a model whose gradient is a correct
      clean zero.

    A threshold that *steps* as a parameter moves — ``floor(P)``, ``ceil(P)``,
    ``sign(P)`` — is none of the three, and falls through to the refusal at the
    end. Its partials come back empty because sympy has no derivative for a step
    function (issue #441 makes that an ordinary decline rather than the stack
    overflow it used to be), and it does name a parameter, so nothing here claims
    to compensate it. That is the honest answer: a crossing time that jumps as a
    parameter moves has no chain rule to the primaries even where its derivative
    is a.e. zero. A step over *constants* is a different thing — ``floor(5)`` is
    5, a crossing nothing moves — and is compensated on the first ground above.
    """
    # ``warn_on_failure=False`` for the same reason the detector passes it: an
    # empty result is the supported "not a switch time" answer, which the caller
    # reports (or hands to the state path) rather than warns about.
    partials = _derived_expr_partials_numeric(
        threshold_expr,
        set(scope.primary_names),
        scope.param_idx,
        list(scope.values),
        scope.derived_exprs,
        warn_on_failure=False,
    )
    value = _evaluate_threshold(threshold_expr, scope.param_idx, scope.values, scope.derived_exprs)
    if value is None:
        if _threshold_is_non_real(
            threshold_expr, scope.param_idx, scope.values, scope.derived_exprs
        ):
            return CrossingTerms({}, None)  # a root that does not occur
        return None
    # ``value`` is already known to be a number here, so the text scan is the
    # whole of :func:`_fixed_threshold_expr` that is left to check.
    if partials or _threshold_has_no_parameter(threshold_expr, scope):
        return CrossingTerms({n: float(v) for n, v in partials.items() if v != 0.0}, value)
    return None


def _threshold_compensated(threshold_expr: str, scope: SwitchConditionScope) -> bool:
    """Whether this one crossing time needs no compensation or gets it."""
    return _threshold_crossing_terms(threshold_expr, scope) is not None


class ScheduleTerms(NamedTuple):
    """What a repeating schedule contributes, before a window is applied.

    The three numbers are the schedule read at the current parameter point, and
    the three dictionaries are ``∂(that number)/∂primary`` over the primaries
    with a non-zero partial. ``crosses`` is whether the condition turns over at
    all: it needs the duty to fall strictly inside a period, and a corpus model
    really does write ``time() - P*floor(time()/P) >= 0``, which is true at every
    instant of the run and never crosses.

    Everything a caller needs to place and differentiate the edges is here, and
    nothing that needs a run window is: the edge at ``offset + k*period + duty``
    has ``∂t*/∂p = ∂offset/∂p + k*∂period/∂p + ∂duty/∂p``, which is this record
    read once and combined per whole ``k``.
    """

    period: float
    offset: float
    duty: float
    d_period: dict[str, float]
    d_offset: dict[str, float]
    d_duty: dict[str, float]
    crosses: bool


# How far the model's own residual may sit from zero at the crossing time a
# recognized schedule predicts, and how far two periods of it may differ, both
# relative to the residual's size half a duty either side of the crossing. A
# schedule that is really there puts an exact zero at its own edge and repeats to
# the last bit; the tolerance is for the roundoff of taking a remainder at a large
# clock value.
_SCHEDULE_RESIDUAL_TOL = 1e-9


def _schedule_matches_residual(
    atom: str,
    sched: PeriodicSchedule,
    period: float,
    duty: float,
    offset: float,
    crosses: bool,
    scope: SwitchConditionScope,
) -> bool:
    """Check a recognized schedule against the condition the model evaluates.

    :func:`_clock_periodic_schedule` reads the residual through ``sympy``'s
    parser, which binds a handful of one-letter names to its own objects: a model
    parameter called ``I`` arrives as the imaginary unit, ``S`` as the singleton
    registry, ``E`` as Euler's number. Most of the time that is harmless, because
    those objects obey the same arithmetic a symbol would and the recognizer's
    answer comes back spelled with the same name. It is not harmless always —
    ``I*I`` folds to ``-1`` — and the failure it produces is the quiet one: a
    schedule that reads as never crossing, which the gate then admits with
    nothing behind it.

    So the schedule is checked against the residual evaluated the *model's* way,
    through :func:`_evaluate_threshold`, which binds parameter names before it
    parses (GH #108). Four evaluations, at clock values one period apart and
    either side of the predicted edge, ask the three things the schedule claims:
    that the residual changes sign where the schedule says it does (or holds one
    sign for the whole period when the schedule says it never crosses), that it
    is zero at the edge itself, and that it repeats a period later.

    Cheap for what it covers. A recognizer that got the period, the phase or the
    duty wrong fails here rather than placing stop times where nothing happens,
    and a residual reading anything :func:`_evaluate_threshold` cannot resolve —
    live state, most obviously — fails here too.
    """
    head = _clock_solve_residual(atom, scope.clock_symbols)
    if head is None:  # pragma: no cover - the recognizer already required one
        return False
    # `ceil` is ExprTk's spelling of `ceiling` and sympy's parser does not know
    # it, so a residual written with one would evaluate to nothing here and every
    # schedule spelled that way would be declined for the wrong reason. The
    # rewrite is the same exact identity the recognizer applies.
    residual = _CEIL_CALL.sub("ceiling(", head[1])

    def at(clock_value: float) -> float | None:
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(_CLOCK_SOLVE_SYMBOL)}(?![A-Za-z0-9_])",
            f"({clock_value!r})",
            residual,
        )
        return _evaluate_threshold(text, scope.param_idx, scope.values, scope.derived_exprs)

    # Clock values to read the residual at, as a fraction of the period past the
    # offset. When the schedule crosses the first two straddle the edge; when it
    # does not they spread over the period, which is where a constant sign has to
    # hold. The third is the first one a period later, and the fourth is the edge
    # itself.
    phi = duty / period
    fractions = (phi / 2.0, (1.0 + phi) / 2.0) if crosses else (0.25, 0.75)
    points = [offset + period * f for f in fractions]
    points.append(offset + period * (fractions[0] + 1.0))
    if crosses:
        points.append(offset + duty)

    probes: list[float] = []
    for point in points:
        value = at(point)
        if value is None or not (abs(value) < float("inf")):
            return False
        probes.append(value)
    first, second, repeat = probes[0], probes[1], probes[2]
    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return False
    if abs(repeat - first) > _SCHEDULE_RESIDUAL_TOL * scale:
        return False  # the residual does not repeat a period later
    if crosses:
        if first * second >= 0.0:
            return False  # no sign change across the edge the schedule names
        if abs(probes[3]) > _SCHEDULE_RESIDUAL_TOL * scale:
            return False  # the residual is not zero at the edge
    elif first * second <= 0.0:
        return False  # a sign change inside a schedule that claims none
    return True


def _periodic_schedule_terms(
    atom: str, sched: PeriodicSchedule, scope: SwitchConditionScope
) -> ScheduleTerms | None:
    """:class:`ScheduleTerms` for a recognized schedule, or ``None`` when nothing
    compensates its crossings (issue #436).

    The schedule's twin of :func:`_threshold_crossing_terms`, and it asks that
    function of each of the three expressions the schedule is made of, so a
    period, an offset and a duty are judged by exactly the rule one crossing time
    is: the expression reduces to the primaries and evaluates, or it carries no
    parameter at all and so moves for nobody. Both the issue #68 gate
    (:func:`clock_crossing_compensated`) and the run-time detector
    (:func:`compute_switch_time_sens`) read this one function, which is what
    stops them answering differently about a schedule.

    A period of zero is refused: there is then no schedule, only a division by
    zero waiting to happen in the enumeration. So is a schedule the model's own
    residual does not actually follow — see :func:`_schedule_matches_residual`.
    """
    values: list[float] = []
    partials: list[dict[str, float]] = []
    for text in (sched.period, sched.offset, sched.duty):
        part = _threshold_crossing_terms(text, scope)
        if part is None or part.value is None:
            return None
        values.append(float(part.value))
        partials.append(part.partials)
    period, offset, duty = values
    if period == 0.0 or not (abs(period) < float("inf")):
        return None
    # The condition turns over once per period exactly when the duty lands
    # strictly inside the period. `duty/period` rather than `0 < duty < period`
    # so a schedule written with `ceil()` — whose remainder runs from -period to
    # 0, and whose recognized period is therefore negative — is judged the same
    # way as one written with `floor()`.
    crosses = 0.0 < duty / period < 1.0
    if not _schedule_matches_residual(atom, sched, period, duty, offset, crosses, scope):
        return None
    return ScheduleTerms(
        period=period,
        offset=offset,
        duty=duty,
        d_period=partials[0],
        d_offset=partials[1],
        d_duty=partials[2],
        crosses=crosses,
    )


def _schedule_compensated(atom: str, scope: SwitchConditionScope) -> bool:
    """Whether *atom* is a repeating schedule whose crossings are compensated."""
    sched = _clock_periodic_schedule(atom, scope.clock_symbols, scope)
    return sched is not None and _periodic_schedule_terms(atom, sched, scope) is not None


# How many crossings one repeating schedule may contribute to a single run.
#
# Every crossing is a stop time: the solver ends a step exactly on it, reads the
# rate law a few ulp either side to get the branch jump, applies the jump and
# restarts. That is cheap but not free, and unlike every other crossing in this
# module the count is not a property of the model — it is the run window divided
# by the period, so a long enough run at a short enough period asks for an
# unbounded number of them.
#
# A guard against unbounded work rather than a tight performance limit, and the
# number comes from measurement. On a dosing model with a 0.25 period a stop costs
# 20-40 us of wall clock, so the cap is about a quarter of a second of stopping
# added to one run, plus 45 ms to detect them. It admits the schedules people
# actually write with room to spare: a hundred days of hourly dosing is 4800
# edges, and the largest any model in this repository's corpus asks for over its
# own reported time course is 200 (MODEL0406553884).
#
# Beyond it bngsim refuses the run rather than dropping the extra edges, because a
# schedule compensated up to its budget and not after it is a gradient that is
# right at the start of the run and wrong at the end, which is the silent-zero
# failure this whole module exists to avoid.
_SCHEDULE_EDGE_BUDGET = 8192


def _schedule_index_window(
    base: float, period: float, v_lo: float, v_hi: float
) -> tuple[int, int] | None:
    """Whole ``k`` for which ``base + k*period`` can land in ``(v_lo, v_hi]``.

    Widened by one on each end and then re-checked against the window by the
    caller, so a rounding of the division cannot drop an edge that is genuinely
    inside it. ``None`` when the arithmetic does not resolve to a finite range,
    which is a schedule nothing can enumerate rather than an empty one.
    """
    try:
        x_lo = (v_lo - base) / period
        x_hi = (v_hi - base) / period
    except (ZeroDivisionError, OverflowError):
        return None
    if not (math.isfinite(x_lo) and math.isfinite(x_hi)):
        return None
    lo, hi = (x_lo, x_hi) if x_lo <= x_hi else (x_hi, x_lo)
    return math.floor(lo) - 1, math.ceil(hi) + 1


def _schedule_edges(
    terms: ScheduleTerms, v_lo: float, v_hi: float, limit: int
) -> list[tuple[float, dict[str, float]]] | None:
    """Every edge of a repeating schedule in ``(v_lo, v_hi]``, as
    ``(clock value, ∂(clock value)/∂primary)`` — or ``None`` when there are more
    than ``limit`` of them (issue #436).

    Two edges per period, and they are different kinds of edge. The condition
    turns over inside the period where the residual passes through zero, at
    ``offset + k*period + duty``; it turns back at the period boundary itself,
    ``offset + (k+1)*period``, where the ``floor`` steps and the remainder drops
    from a whole period to nothing. Both are ordinary crossing times, and both
    differentiate by inspection — the duty moves only the first of the pair, and
    the period moves the ``k``-th edge ``k`` times as far as the first one, which
    is the whole of why a fitted period is worth compensating at all.

    Enumerated rather than searched for, which is what the closed form buys: the
    edges are known from three numbers and a whole ``k``, so nothing here roots
    on anything or steps through the window.

    ``limit`` is counted against the candidate indices, which is the number of
    edges returned plus at most the two each family is widened by — near enough
    at any budget worth setting, and it is what lets the count be decided by
    arithmetic instead of by enumerating first and counting after.
    """
    if not terms.crosses:
        return []
    # (base of the family, the partials that family adds to offset + k*period).
    families = (
        (terms.offset + terms.duty, terms.d_duty),
        (terms.offset, None),
    )
    windows: list[tuple[int, int]] = []
    total = 0
    for base, _extra in families:
        window = _schedule_index_window(base, terms.period, v_lo, v_hi)
        if window is None:
            return None
        total += window[1] - window[0] + 1
        if total > limit:
            return None
        windows.append(window)

    edges: list[tuple[float, dict[str, float]]] = []
    for (base, extra), (k_lo, k_hi) in zip(families, windows, strict=True):
        for k in range(k_lo, k_hi + 1):
            value = base + k * terms.period
            if not (v_lo < value <= v_hi):
                continue
            partials: dict[str, float] = dict(terms.d_offset)
            for name, coeff in terms.d_period.items():
                partials[name] = partials.get(name, 0.0) + k * coeff
            if extra is not None:
                for name, coeff in extra.items():
                    partials[name] = partials.get(name, 0.0) + coeff
            edges.append((value, {n: c for n, c in partials.items() if c != 0.0}))
    return edges


def state_switch_conditions(core, ctx=None) -> list[str]:
    """Rate-law conditions whose crossing the solver locates and jumps (#150).

    A condition that reads model state — ``piecewise(0, Virus < 1, Virus*rho)``
    — flips a branch of ``f`` at a crossing whose time moves with *every*
    parameter through the trajectory. ``∂f/∂θ`` is right inside each branch (the
    ``Piecewise`` derivative carries no boundary delta and does not need one),
    but ``∂x/∂θ`` is discontinuous at the crossing by the saltation term

        s(t*⁺) = s(t*⁻) + (f⁻ − f⁺)·dt*/dθ

    which neither the analytic sensitivity RHS nor CVODES' internal difference
    quotient supplies: both integrate the variational equation smoothly across.
    Handing the conditions to ``SolverOptions.set_state_switch_conditions`` is
    what gets each one registered as a CVODE root — so the crossing is *located*
    rather than chased, which is also what keeps the run out of issue #82's
    collapsed step — and jumped there with ``dt*/dθ`` differentiated by the
    implicit function theorem, exactly as issue #144 does for a state-dependent
    event trigger.

    Clock thresholds are deliberately excluded: issue #48 already compensates
    those, at a crossing time it knows a priori, and a second jump at the same
    instant would double-count it.

    Returns ``[]`` for the overwhelming majority of models — anything with no
    conditional rate law at all short-circuits before the model is probed.
    Deduplicated by *residual*, so one crossing written two ways is one entry.
    """
    from bngsim._jacobian import _inline_functions

    if core.n_functions == 0:
        return []
    if ctx is None:
        ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    bodies = list(func_map.values())
    # Same cheap first gate as compute_switch_time_sens: no `if()` anywhere
    # means no branch to cross, and no RHS probe is paid.
    if not any(_IF_CALL.search(body) for body in bodies):
        return []

    scope = switch_condition_scope(core, ctx)
    conditions: list[str] = []
    seen_residual: set[str] = set()
    for body in bodies:
        # Inlined, because that is the text the GATE judges — and a condition
        # can only become a state condition under inlining. BIOMD0000000837
        # writes `Lymphocyte_Term` as `piecewise(…, 1 - Total_Lymphocytes/K > 0,
        # 0)` where `Total_Lymphocytes` is an assignment-rule parameter, i.e. a
        # *parameter* address that reads no live state; the gate sees it after
        # substitution as `1 - (B+C_e+C_m+H_e+H_m+L)/K > 0` and admits. Scanning
        # the raw body registered nothing for it, so the gate lifted the decline
        # with no crossing behind it — the silent zero the #68 gate exists to
        # stop, reintroduced from the other side. Measured on the corpus: 3 of
        # 182 condition-carrying rr_parity models, invisible to every test.
        flat = _inline_functions(body, func_map) or body
        for atom in _iter_condition_atoms(flat):
            if clock_crossing_compensated(atom, scope):
                continue  # issue #48's crossing; jumping it here would double it
            if is_equality_atom(atom):
                continue  # measure-zero: rooting on it would MAKE the branch (below)
            residual = state_switch_residual(core, atom)
            if residual and residual not in seen_residual:
                seen_residual.add(residual)
                conditions.append(atom)
    return conditions


def switch_gate_cache_digest(core, ctx=None) -> tuple:
    """The part of the issue #68 codegen gate's verdict that parameter VALUES decide.

    Almost everything the model-based codegen reads is structure — stoichiometry,
    rate-law text, the derived-parameter attachment vector — and none of it moves
    when a fit moves a rate constant. This gate is the exception, in two places:

    * :func:`_unit_rate_clock_species` **probes the RHS**, so which species count
      as clocks depends on every parameter value and on the species initial
      concentrations;
    * :func:`clock_crossing_compensated` **evaluates a clock threshold
      numerically** and admits the condition only when that evaluation resolves.

    Either one flipping flips whether the analytic sensitivity RHS is emitted at
    all, which changes the generated source. So a ``.so`` cache key that drops
    parameter values (:func:`bngsim._codegen.compute_model_codegen_hash`) cannot
    just assume they never reach the emitted C — it has to carry this verdict.

    Carrying the *verdict* rather than the values is the whole point: moving a
    rate constant does not move any of these booleans, so the key stays put
    across a fit, while a change that really would re-emit the source moves it.

    The scan mirrors what :func:`bngsim._codegen._functional_dfdp_terms` does —
    same :func:`has_condition_construct` pre-gate over the same function bodies
    and functional rate expressions, and the same inlining before the atoms are
    read — so the digest is non-empty exactly when the gate builds a scope at all.
    Returns ``()`` for the condition-free majority, which pays only the pre-gate
    text scan.
    """
    from bngsim._jacobian import _inline_functions, has_condition_construct

    if core.n_functions == 0:
        return ()
    if ctx is None:
        ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    texts = [
        *func_map.values(),
        *(str(r.get("rate_expr", "")) for r in ctx["functional_reactions"]),
    ]
    conditional = [t for t in texts if has_condition_construct(t)]
    if not conditional:
        return ()
    try:
        scope = switch_condition_scope(core, ctx)
    except Exception as exc:  # pragma: no cover - defensive
        # The gate's own fallback: with no scope every condition declines, and
        # that decline reads no value, so there is nothing to carry.
        logger.debug("switch-gate digest: scope unavailable (%s)", exc)
        return ()
    rows = set()
    for text in conditional:
        flat = _inline_functions(text, func_map) or text
        for atom in _iter_condition_atoms(flat):
            rows.add(
                (
                    atom,
                    clock_crossing_compensated(atom, scope),
                    bool(state_switch_residual(core, atom)),
                )
            )
    return (tuple(sorted(scope.clocks.items())), tuple(sorted(rows)))


def uncompensated_condition_reason(
    expr: str, scope: SwitchConditionScope
) -> UncompensatedCrossingReason | None:
    """Why *expr*'s conditions block the analytic sensitivity RHS, or ``None``
    when every one of them is a discontinuity something already compensates —
    issue #48 by stopping at it, or issue #150 by rooting on it (issue #68).

    Every reason this returns is an :class:`UncompensatedCrossingReason`: what
    is left after #150 is the crossing nothing brackets, and for those the
    difference-quotient fallback is wrong too. A condition that IS compensated
    is admitted outright rather than declined with a milder label.

    ``sympy.diff`` of the ``Piecewise`` an ``if(c, a, b)`` becomes returns a
    clean ``0`` w.r.t. a parameter appearing only in ``c`` — no Dirac delta — so
    :func:`bngsim._jacobian._is_emittable` will never reject it and nothing
    downstream notices. That ``0`` is:

    * **correct** for a clock threshold (``if(t>=sigma, ...)``): it is the whole
      in-branch story, and :func:`compute_switch_time_sens` supplies the rest as
      the crossing jump ``s⁺ = s⁻ + (f⁻−f⁺)·∂t*/∂sigma``;
    * **correct** for a state threshold (``if(X>=thresh, ...)``) too, since issue
      #150: its crossing time moves with every parameter through the trajectory,
      but that crossing is now located as a root and its saltation jump applied
      there, so the in-branch zero is again the whole in-branch story;
    * **wrong** for anything nobody locates the crossing of.

    So an atom is admissible on exactly three grounds:

    1. :func:`_clock_threshold_splits` recognizes it *and* every crossing it
       names is one :func:`_threshold_crossing_terms` accounts for — the exact
       conditions under which the detector emits the compensating record, or has
       nothing to compensate. A crossing it would silently skip is no better than
       an uncompensated state threshold here, and since issue #421 an atom can
       have two, so a *partly* compensated one is refused with the rest. Since
       issue #436 the atom may instead be a repeating schedule
       (``time() - 24*floor(time()/24) >= 7``), which has a crossing in every
       period and is admitted when its period, offset and duty each pass that
       same test.
    2. :func:`state_switch_residual` splits it into a residual over live state,
       which is what :func:`state_switch_conditions` hands the solver to root on
       and jump at. Since issue #381 an *equality* splits too: ``X == 1`` is
       satisfied only on the surface ``X − 1 = 0``, which is where ``X < 1``
       changes branch as well, so the two spell one crossing and share one root.
       (The event path still refuses an equality, which needs a rising edge
       rather than a surface — see ``NetworkModel::state_switch``.)
    3. :func:`condition_cannot_cross` finds it written over run-constants alone
       (``0>0``, and equally ``CRRFLX>1e-07`` for a frozen ``CRRFLX``), so it
       holds one truth value for the whole run and never crosses (issue #382).

    Admitting (2) is not a nicety: with the crossing resolved to a root, CVODES'
    difference-quotient fallback becomes *worse* than it was, not better. Its
    probe evaluates f at ``y + σ·s`` with ``σ ≈ √rtol``, and just past a
    crossing ``σ·s`` is easily wide enough to put the probe back on the other
    branch — on the issue #150 reproduction that injects ``rho·X/σ ≈ 2.7e4``
    into ``ds/dt`` for the sliver of time the state stays within ``σ·|s|`` of the
    surface, and the column comes out 28% high. The analytic RHS differentiates
    each branch where it is live and never probes across, so a condition whose
    crossing IS compensated has to reach it.

    Derived-parameter references are inlined before the scan, so a threshold
    spelled ``sigma = t0 + t_delta`` clears ``t0`` and ``t_delta`` too, and a
    condition written over a derived parameter is reported against its primaries.
    """
    # A comparison that is not inside an if() condition — `beta*(I>1)`, the
    # boolean-as-a-number idiom — is a branch with no locatable threshold at all.
    # Checked first, and over the whole expression, because everything below
    # reasons about `if()` conditions and would simply not see it.
    #
    # `_NOT_CALL` joins the scan for the same reason issue #234 taught the
    # splitter to peel it: `not()` is the SBML spelling of `!`, and only the `!`
    # spelling was being watched here. A rate law of `not(X)` is a step at X=0 —
    # the same boolean-as-a-number idiom as `(X>0)`, which this rejects — but it
    # carries no operator either pattern matches, so it was admitted, sympy
    # differentiated `~X` to a clean 1, and nothing warned. Both spellings now
    # land in the same class.
    spans = _condition_spans(expr)
    for pat in (_RELATIONAL, _LOGICAL, _NOT_OP, _NOT_CALL):
        for m in pat.finditer(expr):
            if not any(lo <= m.start() and m.end() <= hi for lo, hi in spans):
                # `not(` is matched with its paren; report the operator alone.
                op = m.group(0).rstrip("( \t")
                return UncompensatedCrossingReason(
                    f"the comparison {op!r} in {expr!r} is not inside an if() "
                    "condition, so there is no threshold to locate its crossing at and "
                    "nothing can compensate the jump"
                )

    for cond in _iter_if_conditions(expr):
        for atom in _split_logical_atoms(cond):
            # Ground 1 — issue #48 stops at this crossing (or there is none to
            # stop at). Asked FIRST, and through the same predicate the run-time
            # state-switch detector skips on, so a counter-clock threshold —
            # which reads a species and would otherwise qualify on ground 2 as
            # well — is claimed by exactly one of the two.
            if clock_crossing_compensated(atom, scope):
                continue
            # Ground 2 — an equality over live state selects no branch over any
            # INTERVAL, so there is no crossing to compensate. Asked before
            # ground 3 and separately from it, because nothing roots on it:
            # `state_switch_conditions` skips it for the reason spelled out
            # there, and admitting it under ground 3's description would claim a
            # jump that is never applied. The residual is still required to
            # resolve, which is what rejects an equality between run-constants
            # (issue #382's ground, not this one) and one whose sides are
            # themselves comparisons (a boolean difference, whose true-set IS an
            # interval).
            if is_equality_atom(atom):
                if state_switch_residual(scope.core, atom):
                    continue
            # Ground 3 — issue #150 roots on this crossing and jumps it.
            elif state_switch_residual(scope.core, atom):
                continue
            atom_flat = _inline_derived_param_refs(atom, scope.derived_exprs) or atom
            # Ground 3 — a comparison over run-constants holds one truth value
            # for the whole run, so there is no crossing at all to compensate.
            if condition_cannot_cross(atom_flat, scope):
                continue
            split = _clock_threshold_splits(atom, scope.clock_symbols)
            if split is None:
                sched = _clock_periodic_schedule(atom, scope.clock_symbols, scope)
                if sched is None:
                    return _not_a_clock_threshold(atom, atom_flat, scope)
                # A repeating schedule bngsim reads as a schedule, but whose
                # period, offset or duty does not reduce to a constant over the
                # primaries — so the detector could not place its edges (issue
                # #436). Named separately from the crossing-time case below
                # because what is unreadable is a piece of the schedule, not a
                # crossing time: saying "the clock crossing 'stim_period'" would
                # be describing the wrong thing.
                unreadable_parts = [
                    f"{label} {text!r}"
                    for label, text in (
                        ("period", sched.period),
                        ("offset", sched.offset),
                        ("duty", sched.duty),
                    )
                    if not _threshold_compensated(text, scope)
                ]
                if unreadable_parts:
                    return UncompensatedCrossingReason(
                        f"the condition {atom!r} switches on a repeating schedule whose "
                        + " and ".join(unreadable_parts)
                        + " does not reduce to a constant expression over the model's primary "
                        "parameters, so bngsim cannot say when the schedule's edges fall and "
                        "neither the issue #48 switch-time jump nor the issue #150 crossing "
                        "root can compensate them"
                    )
                # Each piece reads back on its own, so what failed is the check
                # that the pieces really describe this condition: the residual
                # does not repeat a period later, or does not change sign where
                # the schedule says it does (:func:`_schedule_matches_residual`).
                return UncompensatedCrossingReason(
                    f"the condition {atom!r} looks like a repeating schedule of period "
                    f"{sched.period!r} and duty {sched.duty!r}, but the condition the model "
                    "actually evaluates does not follow that schedule, so bngsim will not "
                    "place stop times from it and neither the issue #48 switch-time jump nor "
                    "the issue #150 crossing root can compensate its crossings"
                )
            # A clock threshold that neither reduces to a constant over the
            # primaries (so the issue #48 detector would silently skip its
            # crossing) nor reads live state (so issue #150 cannot root on it).
            # Its ∂t*/∂p reaches nobody, and the Piecewise derivative's zero
            # would be the whole gradient. Named one crossing time at a time,
            # since an atom may have two (issue #421) and only one of them be
            # the unreadable one.
            unreadable = [t for t in split[1] if not _threshold_compensated(t, scope)]
            return UncompensatedCrossingReason(
                "the clock crossing "
                + " and ".join(repr(t) for t in unreadable)
                + f" in the condition {atom!r} does not reduce "
                "to a constant expression over the model's primary parameters and does not "
                "read model state either, so neither the issue #48 detector nor the issue "
                "#150 crossing root can compensate it and the Piecewise derivative's zero "
                "would be the whole gradient"
            )
    return None


def _not_a_clock_threshold(
    atom: str, atom_flat: str, scope: SwitchConditionScope
) -> UncompensatedCrossingReason:
    """The decline message for a condition atom nobody compensates the crossing of.

    Reached only after :func:`state_switch_residual` has already declined it, so
    the crossing is neither a clock threshold issue #48 stops at nor a state
    threshold issue #150 roots on: a logical the splitter could not reduce (one
    buried in a call argument, say), a comparison whose residual will not
    compile. A plain conjunction or negation is *not* one of these —
    :func:`_split_logical_atoms` hands their surfaces over one at a time, so this
    is never reached for them (issues #232, #234). Names the
    parameters the atom carries when it has any — a *fitted* threshold is the
    case issue #68 was most concerned with — and otherwise says that the crossing
    moves through the trajectory instead.
    """
    named = sorted(n for n in scope.param_names if scope.param_pats[n].search(atom_flat))
    if named:
        many = len(named) > 1
        return UncompensatedCrossingReason(
            f"the parameter{'s' if many else ''} "
            + ", ".join(repr(n) for n in named)
            + f" appear{'' if many else 's'} in the condition {atom!r}, which is neither a "
            f"recognized clock threshold nor a single comparison over model state, so moving "
            f"{'them' if many else 'it'} moves a branch crossing that neither the issue #48 "
            "switch-time jump nor the issue #150 saltation jump can be run on"
        )
    return UncompensatedCrossingReason(
        f"the condition {atom!r} is not a recognized clock threshold (it reads model state), "
        "and it is not a single comparison whose residual bngsim can root on either, so its "
        "crossing time moves with the trajectory and therefore with every parameter and "
        "nothing compensates the jump — the rate-law twin of the state-dependent event "
        "trigger refused for issue #52"
    )


def _switch_params_in_uncompensated_conditions(
    scope: SwitchConditionScope, function_bodies, candidates: set[str]
) -> set[str]:
    """Subset of *candidates* that also read a condition nothing compensates.

    A switch-time parameter is *pinned* against CVODES' difference-quotient probe
    (issue #48): the probe is held at the parameter's nominal value so it cannot
    drag the switch into the approach and stall the integrator. That is safe only
    when every crossing the parameter moves is compensated — stopped at (issue
    #48) or rooted on (issue #150). If the parameter ALSO reads a condition whose
    crossing nothing brackets, the probe was the only thing that would have
    captured that crossing's dependence on it, and pinning drops it: the column
    comes back a silent zero (MODEL1708310001's ``cycle_int`` measured
    ∂/∂cycle_int = 0 against a finite difference peaking at ~19). Such a parameter
    cannot be pinned, and un-pinning reintroduces the issue #48 stall, so the
    caller refuses it.

    A plain dose schedule, ``if(time - floor(time/period)*period >= offset, …)``,
    used to be the example of such a condition and is one no longer: issue #436
    enumerates its edges and compensates each of them. MODEL1708310001 is still
    the witness because it writes a remainder *of a remainder*, one level deeper
    than that recognizer goes.

    The atom test is :func:`uncompensated_condition_reason`'s — compensated by
    issue #48, then issue #150, then :func:`condition_cannot_cross` — so the two
    cannot disagree about which crossings are compensated. Only relevant on the
    difference-quotient path: an uncompensated crossing is exactly what declines
    the analytic sensitivity RHS (issue #68), so a model that reaches here with an
    analytic RHS has none of these conditions and this returns empty.
    """
    if not candidates:
        return set()
    unsafe: set[str] = set()
    seen: set[str] = set()
    for body in function_bodies:
        for atom in _iter_condition_atoms(body):
            # Same reason the detector's own loop does this: the verdict depends
            # on the atom text and the scope, so one look answers for every rate
            # law that spells it.
            if atom in seen:
                continue
            seen.add(atom)
            if clock_crossing_compensated(atom, scope):
                continue
            if state_switch_residual(scope.core, atom):
                continue
            atom_flat = _inline_derived_param_refs(atom, scope.derived_exprs) or atom
            if condition_cannot_cross(atom_flat, scope):
                continue
            names_in = {m.group(0) for m in _IDENTIFIER.finditer(atom_flat)}
            unsafe |= candidates & names_in
            if unsafe == candidates:
                return unsafe  # nothing more to find
    return unsafe


class SwitchCrossing(NamedTuple):
    """One crossing for ``SolverOptions.set_switch_time_sens`` (issues #48, #375).

    The first four fields are the record as issue #48 shipped it. The last two
    are the isolation bump: parameter deltas the core applies *only* while it
    reads this crossing's ``f⁻``, chosen so this condition alone falls back to
    its before-branch while every other condition landing on the same instant
    stays on its after-branch. Empty — the overwhelmingly common case — means no
    other condition crosses here and the plain ``f⁻ − f⁺`` is this crossing's
    jump on its own.
    """

    t_star: float
    clock_idx0: int
    threshold: float
    dtstar: list[float]
    isolate_param_idx0: list[int]
    isolate_delta: list[float]


class _Crossing(NamedTuple):
    """A detected clock threshold, before it is decided whether it emits."""

    t_star: float
    clock_idx0: int
    threshold: float
    dtstar: list[float]
    # ∂threshold/∂primary over ALL primaries, not just the requested columns.
    # This is the crossing's identity: two spellings of one threshold share it,
    # two thresholds that merely share a value do not (issue #375).
    partials: dict[str, float]


def _q(x: float) -> str:
    """A relative-precision key for a float, for grouping crossings."""
    return f"{x:.12g}"


def _is_same_crossing(a: _Crossing, b: _Crossing) -> bool:
    """Whether two detected thresholds are the SAME crossing written twice.

    Same clock, same value, and the same ∂threshold/∂primary — which is what
    distinguishes `t>=t0` appearing in six rate laws (one crossing) from
    `t>=tau0` and `t>=tauPIP2syn` where the two parameters happen to hold the
    same number (two crossings, issue #375). The partials are compared with a
    tolerance because two spellings can reduce through different arithmetic.
    """
    if a.clock_idx0 != b.clock_idx0 or _q(a.threshold) != _q(b.threshold):
        return False
    if a.partials.keys() != b.partials.keys():
        return False
    return all(
        abs(v - b.partials[n]) <= 1e-9 * max(abs(v), abs(b.partials[n]), 1.0)
        for n, v in a.partials.items()
    )


def _crossing_bucket(cross: _Crossing) -> tuple[int, str]:
    """The two fields :func:`_is_same_crossing` requires before it compares
    anything else: same clock, same threshold value to 12 significant digits.

    Used to bucket the crossings found so far, so absorbing one does not have to
    walk all of them. Not a second definition of sameness — a bucket collision is
    still decided by :func:`_is_same_crossing` — just a way of skipping the
    comparisons that answer no on their first line.
    """
    return cross.clock_idx0, _q(cross.threshold)


def _absorb_crossing(
    found: list[_Crossing],
    cand: _Crossing,
    index: dict[tuple[int, str], list[int]] | None = None,
) -> None:
    """Add ``cand``, or fold it into the crossing it duplicates.

    Folding keeps the larger-magnitude partial per column rather than summing,
    which is the pre-#375 rule and the right one *within* a crossing: one
    threshold gating six rate laws must be jumped across once, not six times.

    ``index`` buckets ``found`` by :func:`_crossing_bucket`, which turns the scan
    from every crossing found so far into the handful sharing this one's instant.
    Optional because it changes no answer, and load-bearing since issue #436: a
    repeating schedule can contribute a thousand crossings to one run, and at that
    size the linear scan is the whole cost of detection (1600 crossings measured
    440 ms of comparisons, against 10 ms for everything else the detector does).
    """
    bucket = None if index is None else index.setdefault(_crossing_bucket(cand), [])
    for i in range(len(found)) if bucket is None else bucket:
        seen = found[i]
        if _is_same_crossing(seen, cand):
            merged = list(seen.dtstar)
            for c, v in enumerate(cand.dtstar):
                if abs(v) > abs(merged[c]):
                    merged[c] = v
            found[i] = seen._replace(dtstar=merged)
            return
    if bucket is not None:
        bucket.append(len(found))
    found.append(cand)


# How far a crossing's threshold is pushed to take it off the instant, relative
# to the threshold's own magnitude. Large enough to clear the core's clock nudge
# (64 ulp) by orders of magnitude; small enough that anything else the bumped
# parameter reaches — an in-branch coefficient, on an issue #358 impure switch
# time — moves by a part in a million and cannot be mistaken for the jump.
_ISOLATION_REL = 1e-6
# The same bump has to survive being added to the threshold, so it may not sink
# into the last bits of it.
_ISOLATION_MIN_ULP = 1024.0
_EPS = sys.float_info.epsilon


def _isolation_bump(
    cross: _Crossing,
    group: Sequence[_Crossing],
    param_idx: dict[str, int],
    thresholds_on_clock: AbstractSet[float],
) -> tuple[int, float]:
    """The ``(param_idx0, delta)`` that takes ``cross`` alone off this instant.

    The core reads a crossing's jump by nudging the *clock* a few ulp either
    side of the threshold, which flips every condition thresholding that clock
    there at once — so when several land on one instant, the difference it reads
    is their sum and no amount of re-keying separates them (issue #375). What
    does separate them is moving one condition's threshold instead: raise it by
    a hair and, with the clock held on the after side, that condition alone
    reads its before-branch. The difference is then this crossing's own jump.

    Which parameter to raise is the whole of the choice. It has to be one this
    crossing's threshold depends on and *no* coinciding threshold does, or the
    bump moves the others too and isolates nothing; when no such parameter
    exists the crossings are not separable this way and bngsim refuses rather
    than return the merged answer #375 reported.
    """
    others: set[str] = set()
    for other in group:
        if other is not cross:
            others |= set(other.partials)
    private = {n: v for n, v in cross.partials.items() if n not in others and n in param_idx}
    if not private:
        shared = sorted(set(cross.partials) & others)
        raise SensitivityUnsupportedError(
            "Forward sensitivity is not supported on this model: the switch times "
            f"crossing together at t={cross.t_star:.6g} cannot be told apart. bngsim reads "
            "a crossing's jump by nudging the clock across the threshold, which flips "
            "every condition thresholding it at that instant at once, so a coinciding "
            "crossing is separated by moving its threshold off the instant instead "
            "(issue #375) — and every parameter this one's threshold depends on ("
            + ", ".join(f"'{n}'" for n in sorted(cross.partials))
            + ") is also read by a threshold it coincides with"
            + (" (" + ", ".join(f"'{n}'" for n in shared) + ")" if shared else "")
            + ". Returning the merged jump would charge each parameter with the other's, "
            "so bngsim refuses. Separating the switch times — they are equal only by "
            "coincidence of value — makes this model supported."
        )
    # The largest partial gives the smallest parameter step for a given move of
    # the threshold, so it is the one least able to disturb anything else.
    name = max(private, key=lambda n: abs(private[n]))
    span = max(abs(cross.threshold), 1.0)
    delta_threshold = _ISOLATION_REL * span
    # Never step so far that the bumped threshold reaches a DIFFERENT threshold
    # on the same clock: that would flip a condition this crossing does not own,
    # which is the very contamination being removed.
    gaps = [abs(t - cross.threshold) for t in thresholds_on_clock if _q(t) != _q(cross.threshold)]
    if gaps:
        delta_threshold = min(delta_threshold, 0.25 * min(gaps))
    # Unreachable while :func:`_q` groups on 12 significant digits: the closest
    # threshold it calls *different* is far enough away that a quarter of the gap
    # still clears this floor. Widening `_q` would silently start producing steps
    # that vanish into the threshold's last bits, flipping nothing — and a jump
    # read from two identical RHS evaluations is an exact zero, the one outcome
    # worth failing over. Kept as the thing that fails instead, with the margin
    # itself pinned by test_the_quantisation_leaves_room_for_the_isolation_step.
    floor = _ISOLATION_MIN_ULP * _EPS * span
    if delta_threshold < floor:
        raise SensitivityUnsupportedError(
            "Forward sensitivity is not supported on this model: the switch times "
            f"crossing together at t={cross.t_star:.6g} sit closer to a neighbouring "
            "threshold than the step that would separate them, so isolating this "
            "crossing's jump would flip a condition it does not own (issue #375). "
            "bngsim refuses rather than return a merged jump."
        )
    return param_idx[name], delta_threshold / private[name]


def _emit_switch_records(
    found: list[_Crossing], param_idx: dict[str, int]
) -> list[SwitchCrossing]:
    """Turn detected crossings into records, isolating any that coincide.

    A crossing no requested parameter moves emits nothing — but it is still in
    ``found``, and its presence on an instant is exactly what puts the crossings
    that DO emit there onto the isolation path. That is why issue #375
    reproduced with a single parameter requested: the coinciding condition
    contaminates the core's ``f⁻`` whether or not anyone asked about it.
    """
    by_instant: dict[str, list[_Crossing]] = {}
    thresholds_on_clock: dict[int, set[float]] = {}
    for cross in found:
        by_instant.setdefault(_q(cross.t_star), []).append(cross)
        thresholds_on_clock.setdefault(cross.clock_idx0, set()).add(cross.threshold)

    records: list[SwitchCrossing] = []
    for group in by_instant.values():
        for cross in group:
            if not any(v != 0.0 for v in cross.dtstar):
                continue  # no requested parameter moves this crossing
            if len(group) > 1:
                idx0, delta = _isolation_bump(
                    cross, group, param_idx, thresholds_on_clock[cross.clock_idx0]
                )
                isolate_idx, isolate_delta = [idx0], [delta]
            else:
                isolate_idx, isolate_delta = [], []
            records.append(
                SwitchCrossing(
                    t_star=cross.t_star,
                    clock_idx0=cross.clock_idx0,
                    threshold=cross.threshold,
                    dtstar=cross.dtstar,
                    isolate_param_idx0=isolate_idx,
                    isolate_delta=isolate_delta,
                )
            )
    records.sort(key=lambda r: r.t_star)
    return records


def _absorb_schedule_crossings(
    found: list[_Crossing],
    found_index: dict[tuple[int, str], list[int]],
    atom: str,
    scope: SwitchConditionScope,
    core,
    col_of: dict[str, int],
    t_start: float,
    t_end: float,
    n_cols: int,
) -> None:
    """Add every edge of *atom*'s repeating schedule that falls in the run window
    (issue #436), or do nothing when *atom* is not one.

    Split out of :func:`compute_switch_time_sens`'s atom loop because this is the
    one crossing shape whose *count* depends on the run window. Everything else
    about it is ordinary: each edge becomes a ``_Crossing`` with a clock value and
    a ``∂t*/∂p``, absorbed by the same rule as any other, so a schedule edge that
    lands on the same instant as some other crossing merges or isolates exactly as
    two thresholds would.

    Raises
    ------
    SensitivityUnsupportedError
        When the window holds more edges than :data:`_SCHEDULE_EDGE_BUDGET` *and*
        a requested parameter moves them. Refusing is the only honest answer
        there: the jumps that fit inside a budget and the ones that do not are
        the same gradient, so returning the first few thousand would be right
        early in the run and silently wrong afterwards. A schedule no requested
        parameter moves emits nothing either way, so an over-budget one is simply
        not enumerated.
    """
    sched = _clock_periodic_schedule(atom, scope.clock_symbols, scope)
    if sched is None:
        return
    terms = _periodic_schedule_terms(atom, sched, scope)
    if terms is None or not terms.crosses:
        # Unreadable (the gate declines the model over it, so no analytic RHS is
        # emitted and there is nothing to compensate here), or a schedule that
        # never turns over — `time() - P*floor(time()/P) >= 0` is true at every
        # instant of the run.
        return

    if sched.clock in _TIME_SYMBOLS:
        clock_idx0 = -1
        clock_now = t_start
    else:
        clock_idx0 = scope.clocks[sched.clock]
        # dc/dt = 1, so the clock reads t_start's value now and advances with the
        # run; the window in clock values is the window in time, shifted.
        clock_now = core.get_concentration(core.species_names[clock_idx0])

    edges = _schedule_edges(terms, clock_now, clock_now + (t_end - t_start), _SCHEDULE_EDGE_BUDGET)
    moved_names = set(terms.d_period) | set(terms.d_offset) | set(terms.d_duty)
    if edges is None:
        if moved_names & set(col_of):
            raise SensitivityUnsupportedError(
                f"Forward sensitivity is not supported on this run: the condition {atom!r} "
                "switches on a repeating schedule, and the reported time window holds more "
                f"than {_SCHEDULE_EDGE_BUDGET} of its edges (period "
                f"{terms.period:.6g}, window {t_end - t_start:.6g}). bngsim compensates such "
                "a schedule by stopping the solver at every edge and applying the branch "
                "jump there (issue #436), and it will not compensate only the first few "
                "thousand: that would give a gradient that is correct early in the run and "
                "silently wrong after it. Shorten the reported time window, lengthen the "
                "period, or drop "
                + ", ".join(f"'{n}'" for n in sorted(moved_names & set(col_of)))
                + " from sensitivity_params."
            )
        logger.debug(
            "switch-time: schedule %r has more than %d edges in the window, and no "
            "requested parameter moves it; not enumerated",
            atom,
            _SCHEDULE_EDGE_BUDGET,
        )
        return

    for value, partials in edges:
        t_star = t_start + (value - clock_now)
        # The same half-open window the rest of the loop applies, re-checked in
        # time rather than in clock values so a schedule on a counter clock is
        # filtered by exactly the test a threshold on it would be.
        if not (t_start < t_star <= t_end):
            continue
        dtstar = [0.0] * n_cols
        for prim_name, coeff in partials.items():
            col = col_of.get(prim_name)
            if col is not None and coeff != 0.0:
                dtstar[col] += float(coeff)
        _absorb_crossing(
            found,
            _Crossing(
                t_star=t_star,
                clock_idx0=clock_idx0,
                threshold=value,
                dtstar=dtstar,
                partials=dict(partials),
            ),
            found_index,
        )


def compute_switch_time_sens(
    core,
    sens_param_names,
    t_start: float,
    t_end: float,
    has_analytic_sens_rhs: bool = False,
) -> tuple[list[SwitchCrossing], list[int]]:
    """Switch-time crossings and their ``∂t*/∂p``, plus the parameters to pin.

    Parameters
    ----------
    core
        The C++ ``NetworkModel``.
    sens_param_names
        Requested sensitivity parameters, in the column order the run will use.
    t_start, t_end
        Reported time window; crossings outside it contribute nothing.
    has_analytic_sens_rhs
        Whether this run installs an analytic sensitivity RHS
        (``bngsim_codegen_sens_rhs``) rather than falling back to CVODES'
        internal difference quotient. It decides one thing only: whether a
        parameter that is *both* a switch time and an in-branch coefficient is
        accepted or refused (issue #358, see below). Left ``False`` by default so
        a caller that cannot tell keeps the conservative refusal.

    Returns
    -------
    records
        :class:`SwitchCrossing` values sorted by ``t_star``, for
        ``SolverOptions.set_switch_time_sens``. Empty unless some ``if()``
        threshold actually moves with one of ``sens_param_names`` — a model whose
        switch times are all fixed constants needs no jump and is left alone.

        One record per DISTINCT crossing, where distinctness is
        ``∂threshold/∂primary`` rather than the threshold's value: one threshold
        gating six rate laws is still one crossing (jumping it six times would be
        as wrong as merging), while two thresholds that merely hold the same
        number are two (issue #375). Records that share an instant carry the
        isolation bump that lets the core read each one's own jump.
    pinned
        0-based indices of the switch-time parameters, for
        ``SolverOptions.set_switch_pinned_params``.

    Raises
    ------
    SensitivityUnsupportedError
        If a requested parameter sets a switch time *and* scales something inside
        a branch, *and* ``has_analytic_sens_rhs`` is false. On the difference
        quotient the jump alone is not the whole gradient and pinning holds the
        genuine in-branch ``∂f/∂p`` at a wrong 0, so bngsim refuses rather than
        return a partially-correct derivative; an analytic RHS carries that term
        itself and the pair is accepted (issue #358). Typed (issue #320) so a
        caller can tell this declared gap from a bug without matching on message
        text; it still inherits ``ValueError``, which this raised before.

        Also if two crossings land on the same instant and no parameter is
        private to one of them, so neither can be moved off it without moving the
        other (issue #375). The merged jump that used to be returned there
        charged each parameter with the other's.
    """
    names = list(sens_param_names)
    if not names or core.n_functions == 0:
        return [], []

    ctx = core.functional_jacobian_context()
    func_map = dict(ctx["function_map"])
    # Cheapest possible gate, checked before anything else: a model with no
    # conditional rate law has no switch to find. On the RAW bodies, so a
    # genome-scale model with tens of thousands of functions and no `if()`
    # short-circuits before any inlining — inlining cannot create an `if()` that
    # no body already has. Keeps the RHS probes and the sympy work off the path
    # of every ordinary sensitivity run, including the per-row batch path.
    if not any(_IF_CALL.search(body) for body in func_map.values()):
        return [], []
    # Inline function references before scanning, exactly as the issue #150 state
    # detector (:func:`state_switch_conditions`) already does — "the gate and the
    # detector must scan the SAME text". A clock threshold can hide behind a
    # function: BIOMD0000001007 writes `heav_x = if((x<0), 0, if((x>0), 1, 0))`
    # with `x = time() - ModelValue_27`, so the raw body's atom is `x<0`, which
    # names no clock and is silently skipped — while `state_switch_conditions`
    # inlines to `(time()-ModelValue_27)<0`, recognizes the clock crossing and
    # `continue`s past it as #48's job. The crossing was then compensated by
    # neither detector, dropping the saltation jump for every parameter that moves
    # it (∂t*/∂Tdam=1 through ModelValue_27) and leaving that column short.
    from bngsim._jacobian import _inline_functions

    function_bodies = [_inline_functions(body, func_map) or body for body in func_map.values()]

    # Counter-species clocks (the BNGL idiom) plus literal simulation time (what
    # SBML's `time` csymbol emits, and what a BNGL model may write directly).
    # `time` needs no counter, so this is never empty — an SBML piecewise-in-time
    # rate law is detected on exactly the same path as a `.net` counter switch.
    # Assembled through the scope #68's codegen gate also builds, so the two
    # cannot disagree about which symbols are clocks or what a threshold reduces
    # to (that divergence is the whole hazard #68 was filed against).
    scope = switch_condition_scope(core, ctx)
    clocks = scope.clocks
    clock_symbols = set(scope.clock_symbols)
    param_idx = scope.param_idx
    derived_exprs = scope.derived_exprs
    # A requested parameter that is itself derived has no independent axis: its
    # partials are attributed to the primaries it is built from, exactly as the
    # #41/#43 chain rules do. Columns for such a name stay 0.
    col_of = {name: c for c, name in enumerate(names)}

    # Every clock threshold this model spells, one entry per DISTINCT crossing.
    # "Distinct" is decided by ∂threshold/∂primary, not by the threshold's value:
    # the same threshold appearing in many rate laws — `t>=t0` gates six
    # functions in Lin2021 — has one set of partials and collapses to one
    # crossing, while two thresholds that merely happen to share a *number*
    # (`tau0 = tauPIP2syn = 0.05` on BIOMD0000000075) have different partials and
    # stay apart. Keying on the value alone is issue #375: it merged those two
    # into one record whose ∂t*/∂p was the union, and the core then charged each
    # parameter with the other's jump.
    #
    # Crossings that no *requested* parameter moves are collected here too, and
    # deliberately: they emit no record, but a condition flipping at the same
    # instant still contaminates the f⁻−f⁺ the core reads, so their presence is
    # what turns isolation on below. Dropping them early is why #375 reproduced
    # even when a single parameter was requested.
    found: list[_Crossing] = []
    # ``found`` bucketed by clock and threshold value, so absorbing a crossing
    # does not walk every crossing found so far (see :func:`_absorb_crossing`).
    found_index: dict[tuple[int, str], list[int]] = {}
    # One pass per DISTINCT atom text. Everything below reads the atom and the
    # model-level scope and nothing else, so a second look at the same text
    # re-derives the same crossing and `_absorb_crossing` folds it back into
    # itself: the same answer for the same sympy work. A meal-timing model spells
    # its six conditions in twenty rate laws, so that is 120 recognizer passes for
    # 6 answers, and the recognizers are where a detection pass spends its time.
    seen_atoms: set[str] = set()

    for body in function_bodies:
        for cond in _iter_if_conditions(body):
            for atom in _split_logical_atoms(cond):
                if atom in seen_atoms:
                    continue
                seen_atoms.add(atom)
                # The recognizer #68's codegen gate shares (see
                # _clock_threshold_splits): the gate may only admit a condition
                # this loop turns into a compensating jump.
                #
                # :func:`clock_crossing_compensated` is this loop's acceptance
                # test stated as a predicate, for the gate and for the issue
                # #150 detector — which must skip exactly what this loop claims,
                # or a counter-clock threshold (a *species*, hence live state)
                # would be jumped twice. Since issue #421 the two share the
                # per-crossing rule below (:func:`_threshold_crossing_terms`) as
                # well as the recognizer, and TestTheGateAndTheDetectorsAgree
                # still checks the behaviour rather than the sharing.
                split = _clock_threshold_splits(atom, clock_symbols)
                if split is None:
                    _absorb_schedule_crossings(
                        found,
                        found_index,
                        atom,
                        scope,
                        core,
                        col_of,
                        t_start,
                        t_end,
                        len(names),
                    )
                    continue
                clock_sym, threshold_exprs = split

                # One atom, one crossing — until issue #421, where a quadratic
                # clock threshold answers with the two edges of a window.
                #
                # Resolved through the same rule the gate reads, so the two cannot
                # disagree about a crossing (issue #68). A `None` here is a
                # crossing nothing compensates, and it takes the WHOLE atom's
                # jumps with it: compensating one edge of a window and leaving the
                # other to flip the branch unjumped is the silent-zero half of the
                # answer, which is worse than none of it. The crossings still go
                # into `found` — a condition flipping at an instant contaminates
                # the f⁻−f⁺ read there whether or not anyone can jump it (issue
                # #375) — they just carry no ∂t*/∂p.
                terms = [_threshold_crossing_terms(e, scope) for e in threshold_exprs]
                compensated = all(t is not None for t in terms)

                for threshold_expr, term in zip(threshold_exprs, terms, strict=True):
                    if term is None or term.value is None:
                        # Unreadable, or a root off the real line — a crossing
                        # that does not happen at this parameter point. Neither
                        # records anything.
                        logger.debug(
                            "switch-time: threshold %r has no real constant value; skipping",
                            threshold_expr,
                        )
                        continue
                    partials = term.partials if compensated else {}
                    threshold_value = term.value
                    dtstar = [0.0] * len(names)
                    for prim_name, coeff in partials.items():
                        col = col_of.get(prim_name)
                        if col is not None and coeff != 0.0:
                            dtstar[col] += float(coeff)

                    if clock_sym in _TIME_SYMBOLS:
                        clock_idx0 = -1
                        t_star = threshold_value
                    else:
                        clock_idx0 = clocks[clock_sym]
                        # dc/dt = 1, so c(t) = c(t_start) + (t − t_start) and the
                        # crossing is offset by the clock's current value. Lin2021
                        # seeds counter() at 1, shifting every threshold by a day.
                        clock_now = core.get_concentration(core.species_names[clock_idx0])
                        t_star = t_start + (threshold_value - clock_now)

                    # Half-open window, matching the core's own filter: a crossing
                    # on t_end still jumps into the final recorded column; one on
                    # t_start would precede the run's initial recording.
                    if not (t_start < t_star <= t_end):
                        continue

                    _absorb_crossing(
                        found,
                        _Crossing(
                            t_star=t_star,
                            clock_idx0=clock_idx0,
                            threshold=threshold_value,
                            dtstar=dtstar,
                            partials=dict(partials),
                        ),
                        found_index,
                    )

    records = _emit_switch_records(found, param_idx)
    if not records:
        return [], []

    # Every parameter that moves at least one detected crossing is a switch-time
    # parameter, and must be pinned against CVODES' FD probe — pinning is both
    # what makes ∂f/∂p come out as the correct 0 and what keeps the probe from
    # displacing the switch into the approach (which stalls the integrator).
    switch_params = {
        names[c]
        for _t, _ci, _thr, dtstar, _ii, _id in records
        for c in range(len(names))
        if dtstar[c] != 0.0
    }
    # A switch-time parameter is safe to pin only if EVERY crossing it moves is
    # compensated. One that also reads a condition nothing brackets — a dose
    # schedule written as a remainder of a remainder, one level past what issue
    # #436 reads — cannot be: pinning holds the probe that was the uncompensated
    # crossing's only handler, so the column comes back a silent zero
    # (MODEL1708310001's `cycle_int`). Refused before the in-branch
    # test below, and regardless of the sensitivity RHS, because an uncompensated
    # crossing is what put this model on the difference quotient to begin with
    # (issue #68) — there is no analytic path here to combine anything on.
    unsafe = _switch_params_in_uncompensated_conditions(scope, function_bodies, switch_params)
    if unsafe:
        raise SensitivityUnsupportedError(
            "Forward sensitivity w.r.t. "
            + ", ".join(f"'{p}'" for p in sorted(unsafe))
            + " is not supported on this model: each moves a detected switch time AND "
            "also reads a condition whose crossing nothing compensates. Pinning it against "
            "CVODES' difference-quotient probe (issue #48) would hold that crossing's "
            "dependence on it at a wrong 0, and un-pinning would drag the switch into "
            "the approach and stall (issue #358). bngsim refuses rather than return a "
            "partially-correct derivative. Drop it from sensitivity_params."
        )

    # A parameter that ALSO scales something inside a branch (`if(t>=sigma,
    # sigma*k, 0)`) has a genuine non-zero in-branch ∂f/∂p, so its gradient is
    # the interior variational term PLUS the crossing jump, not the jump alone.
    #
    # Whether bngsim can deliver that sum is decided entirely by the sensitivity
    # RHS this run installs (issue #358):
    #
    # * On the **analytic** path, `bngsim_dfdp` already emits the in-branch
    #   ∂f/∂p — the clean `Piecewise` derivative, gated by the same clock
    #   threshold, no boundary delta (`d/dsigma if(t>=sigma, sigma*k, 0)` is
    #   `if(t>=sigma, k, 0)`) — so the variational equation integrates the
    #   interior term and the saltation jump `(f⁻−f⁺)·∂t*/∂p` adds the boundary.
    #   The two already-computed terms sum to the correct total derivative, and
    #   pinning below is inert (CVODES never perturbs `sens_p` when a user sens
    #   RHS is set, so there is no probe to hold nominal). Accept.
    # * On the **difference-quotient** path there is no `bngsim_dfdp`; the whole
    #   ∂f/∂p is CVODES' internal probe, and pinning a switch-time parameter is
    #   load-bearing in both directions the header comment names — it forces the
    #   in-branch term to a WRONG 0 for an impure parameter, and un-pinning to
    #   recover it would drag the switch into the approach and stall. bngsim
    #   cannot combine the two, so it refuses rather than return a
    #   partially-correct derivative (the same hazard the issue #68 gate exists
    #   for). Refuse.
    pure = _condition_only_params(core, switch_params, derived_exprs)
    impure = sorted(switch_params - pure)
    if impure and not has_analytic_sens_rhs:
        raise SensitivityUnsupportedError(
            "Forward sensitivity w.r.t. "
            + ", ".join(f"'{p}'" for p in impure)
            + " is not supported on this model: each sets an if() switch time AND "
            "appears inside a branch (or as a rate constant), so its gradient is "
            "the in-branch ∂f/∂p plus the crossing jump — and this run has no "
            "analytic sensitivity RHS, so ∂f/∂p is CVODES' internal difference "
            "quotient, which pinning holds at a wrong 0 for such a parameter "
            "(issue #358). An analytic sensitivity RHS combines the two; a model "
            "reaches the difference quotient only when a rate law it must "
            "differentiate carries a non-smooth builtin or a state-dependent "
            "condition (issue #68). Split the parameter into a separate switch "
            "time and rate constant, or drop it from sensitivity_params."
        )

    pinned = sorted(param_idx[p] for p in switch_params if p in param_idx)
    return records, pinned


class ThresholdScope(NamedTuple):
    """How an event trigger's threshold is allowed to be written (issue #49).

    ``exprs`` is the inlining map: every name in it stands for the expression it
    maps to, flattened recursively by :func:`_inline_derived_param_refs`.
    ``primaries`` is what may survive that flattening — an identifier outside it
    means the threshold is not a constant over the model's fitted parameters and
    the trigger must be refused, not guessed at.
    """

    exprs: dict[str, str]
    primaries: frozenset[str]


def _threshold_scope(scope: SwitchConditionScope, ctx) -> ThresholdScope:
    """Widen the derived-parameter map with *rule-bound* parameters.

    An SBML ``<assignmentRule>`` on a parameter arrives as a model **function**
    whose name matches a parameter's; the engine copies the function's value
    into the parameter before each RHS evaluation (the ``var_param_binding``
    idiom). Such a parameter is NOT a constant, even though
    ``param_is_expression`` is false for it and reading its current value looks
    like reading a literal.

    That distinction is load-bearing. BIOMD0000000301 writes its pulse schedule
    as ``pulse2_start = pulse1_start + pulse1_length + pulse_interval``, so an
    event triggered on ``time >= pulse2_start`` moves when ``pulse1_start``
    does. Treating ``pulse2_start`` as a primary put the entire ∂t*/∂p on the
    wrong column and left ``pulse1_start``'s at zero — a confidently wrong
    gradient, which is the failure mode this module exists to avoid.

    A rule-bound parameter joins the inlining map only when its body reduces to
    arithmetic over parameters (iterated to a fixed point, so a rule written
    over another rule — ``pulse3_start`` over ``pulse2_start`` — is admitted on
    a later pass). One that reads a species, an observable, ``time``, or any
    function call is left out, and a threshold naming it is then refused by the
    leftover-identifier check in :func:`_analyze_event_trigger` rather than
    silently evaluated at its current value.
    """
    exprs = dict(scope.derived_exprs)
    rule_bound = {
        name: body
        for name, body in dict(ctx["function_map"]).items()
        if name in scope.param_idx and body
    }
    # `f(` anywhere means a call — a rule over floor()/if() is not arithmetic
    # over parameters, and its derivative is not a constant.
    call = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_]\w*\s*\(")
    resolvable: dict[str, str] = {}
    for _ in range(len(rule_bound) + 1):
        grew = False
        for name, body in rule_bound.items():
            if name in resolvable or call.search(body):
                continue
            idents = {m.group(0) for m in _IDENTIFIER.finditer(body)}
            unknown = idents - set(exprs) - set(resolvable)
            # A name that is a *parameter* is fine only when it is not itself an
            # unresolved rule: otherwise the flattening would stop on it.
            if unknown - (set(scope.param_idx) - set(rule_bound)):
                continue
            resolvable[name] = body
            grew = True
        if not grew:
            break
    exprs.update(resolvable)
    return ThresholdScope(
        exprs=exprs,
        # `rule_bound` is subtracted whole, not just the resolvable part: a
        # rule-bound parameter has ``param_is_expression`` false, so it is in
        # ``primary_names`` and would otherwise pass the leftover check as a
        # fitted constant — which is the silent-zero this scope exists to stop.
        primaries=frozenset(scope.primary_names) - set(exprs) - set(rule_bound),
    )


class EventTimeSensResult(NamedTuple):
    """What :func:`compute_event_time_sens` found about a model's events."""

    # ``(event_idx0, [∂t*/∂p per requested column])`` for every event whose
    # crossing lands in the reported window and moves with a requested
    # parameter. Feeds ``SolverOptions.set_event_time_sens``.
    records: list[tuple[int, list[float]]]
    # 0-based indices of events whose trigger reduces entirely to clock
    # thresholds, so ∂t*/∂p is known (possibly exactly 0) and the core's
    # parameter-dependent-trigger refusal can be lifted for them.
    compensated: list[int]
    # event_idx0 → why the trigger could not be reduced, for the events that
    # are NOT compensated. Only populated for triggers that carry a symbol at
    # all, so an ordinary literal-time event contributes nothing.
    reasons: dict[int, str]
    # The subset of ``reasons`` whose event must be REFUSED rather than merely
    # left uncompensated: the trigger transitively reads a requested
    # sensitivity parameter, so ∂t*/∂p is non-zero and unavailable. The core's
    # own check tests the trigger's *bound addresses*, which cannot see through
    # an assignment-rule parameter (``t_rule = t_first + …`` binds the trigger
    # to ``t_rule``'s address, never to ``t_first``'s) — so this closes the same
    # hole for the wider notion of "threshold" issue #49 introduced.
    blocked: dict[int, str]


def compute_event_time_sens(
    core,
    sens_param_names,
    t_start: float,
    t_end: float,
) -> EventTimeSensResult:
    """Event-time sensitivities ``∂t*/∂p`` for time-triggered events (issue #49).

    A switch time encoded as an SBML **event** — ``time >= T0`` firing
    ``on := 1``, with the rate laws reading ``on`` — is the same modelling
    intent as the ``piecewise(kin, time >= T0, 0)`` that issue #48 covers, and
    has the same gradient. What differs is that the state jumps as well as the
    time moving, so the forward-sensitivity jump carries all four terms::

        s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p

    ``∂h/∂x`` and ``∂h/∂p`` are already differenced by the core at each fire
    (GH #212); this function supplies the missing ``∂t*/∂p``.

    A trigger qualifies when it is a conjunction of **clock thresholds** — the
    same recognizer :func:`_clock_threshold_splits` applies to ``if()``
    conditions, so the event path and the rate-law path cannot drift about what
    a locatable crossing is. The rising edge of a conjunction of half-lines is
    the largest of its lower bounds, and it is that atom's threshold whose
    derivative moves the fire, so ``∂t*/∂p = ∂(that threshold)/∂p``.

    Everything else is left uncompensated (and refused upstream by
    :func:`NetworkModel::event_sensitivity_unsupported_reason` when it actually
    reads a requested parameter): a state-dependent trigger, whose crossing
    moves through the trajectory; a disjunction, whose rising edge is the
    minimum over branches and can hand off between them as a parameter moves; a
    negation, which turns a rising edge into a falling one; an equality on a
    continuous clock; and a tie between two lower bounds, where ``t*(p)`` has a
    kink rather than a derivative.
    """
    names = list(sens_param_names)
    n_events = core.n_events
    if n_events == 0:
        return EventTimeSensResult([], [], {}, {})

    triggers = list(core.event_trigger_sources())
    ctx = core.functional_jacobian_context()
    scope = switch_condition_scope(core, ctx)
    clocks = scope.clocks
    clock_symbols = set(scope.clock_symbols)
    thresholds = _threshold_scope(scope, ctx)
    col_of = {name: c for c, name in enumerate(names)}
    # Every body a trigger symbol can stand for, resolvable or not — used only
    # to decide whether an UNREDUCED trigger still depends on a requested
    # parameter. Kept separate from `thresholds.exprs`, which admits only the
    # bodies that reduce to arithmetic over primaries.
    all_bodies = dict(scope.derived_exprs)
    all_bodies.update({n: b for n, b in dict(ctx["function_map"]).items() if b})
    requested = set(names)

    records: list[tuple[int, list[float]]] = []
    compensated: list[int] = []
    reasons: dict[int, str] = {}
    blocked: dict[int, str] = {}

    for ei in range(min(n_events, len(triggers))):
        trigger = triggers[ei] or ""
        analysis = _analyze_event_trigger(
            core, trigger, scope, thresholds, clocks, clock_symbols, t_start
        )
        if isinstance(analysis, str):
            # Only worth reporting when the trigger names something; a literal
            # comparison the recognizer declined carries no gradient anyway.
            if _IDENTIFIER.search(trigger):
                reasons[ei] = analysis
                flat = _inline_derived_param_refs(trigger, all_bodies) or trigger
                moved_by = sorted(requested & {m.group(0) for m in _IDENTIFIER.finditer(flat)})
                if moved_by:
                    blocked[ei] = (
                        "the crossing time of event "
                        + repr(core.event_trigger_sources()[ei])
                        + " moves with "
                        + ", ".join(repr(n) for n in moved_by)
                        + ", but "
                        + analysis
                    )
            continue
        compensated.append(ei)
        if analysis is None:
            continue  # no rising edge at all (or none inside the window)
        t_star, partials = analysis
        if not (t_start < t_star <= t_end):
            # The event either already fired at (or before) t_start — where its
            # firing time is pinned to the run's own start and does not move —
            # or never fires in this run. Either way there is no jump to make,
            # and ∂t*/∂p is correctly absent rather than merely unknown.
            continue
        dtstar = [0.0] * len(names)
        moved = False
        for prim_name, coeff in partials.items():
            col = col_of.get(prim_name)
            if col is not None and coeff != 0.0:
                dtstar[col] += float(coeff)
                moved = True
        if moved:
            records.append((ei, dtstar))

    return EventTimeSensResult(records, compensated, reasons, blocked)


def _analyze_event_trigger(
    core,
    trigger: str,
    scope: SwitchConditionScope,
    thresholds: ThresholdScope,
    clocks: dict[str, int],
    clock_symbols: set[str],
    t_start: float,
):
    """Reduce an event trigger to its rising edge.

    Returns ``(t_star, {primary: ∂threshold/∂primary})`` when the trigger is a
    conjunction of clock thresholds with a well-defined false→true edge,
    ``None`` when it is such a conjunction but has no rising edge (no lower
    bound, or an empty true-interval), or a reason string when it cannot be
    reduced at all.
    """
    expr = trigger.strip()
    if not expr:
        return "the event has no trigger expression"
    # Only conjunction is reducible: the true-set of an `&&` of half-lines is an
    # interval, whose left endpoint is one atom's threshold. `||` unions
    # intervals (the rising edge is the earliest crossing and can hand off from
    # one branch to another as a parameter moves), and nand/nor/xor negate.
    for m in _LOGICAL.finditer(expr):
        if m.group(0) not in ("&&", "and"):
            return (
                f"the trigger {trigger!r} joins its comparisons with {m.group(0)!r}; only a "
                "conjunction has a rising edge that is one threshold's crossing, so ∂t*/∂p "
                "is not a single threshold's derivative here"
            )
    # Both spellings: ExprTk's `!` and the `not(...)` call form the SBML loader
    # emits for <not/> (_ast_to_exprtk). Negation turns the false→true edge this
    # jump is derived for into a true→false one.
    #
    # This refusal and the rate-law path's *peel* (:func:`_strip_negation`, issue
    # #234) look like two answers to one question; they are answers to two. A
    # rate-law switch needs only to know WHERE the branch flips: the core reads
    # f⁻/f⁺ by evaluating the real RHS on each side of the located crossing, so
    # ¬c and c name the same surface and the sense is never consulted. This
    # reduction needs to know WHICH crossing is the rising edge, which is a
    # statement about the sense — it orients every atom into a lower or an upper
    # bound (`_clock_threshold_split_oriented`) and takes `t* = max(lower)`.
    # Negation swaps those roles, so peeling here would hand back a confidently
    # wrong ∂t*/∂p rather than a coarser one. Refusing anywhere in the trigger,
    # rather than per atom, keeps that true under `&&` as well.
    if _NOT_OP.search(expr) or _NOT_CALL.search(expr):
        return (
            f"the trigger {trigger!r} is negated; negation turns the false→true edge this "
            "jump is derived for into a true→false one"
        )

    lower: list[tuple[float, dict[str, float]]] = []  # (t_star, partials)
    upper: list[float] = []
    for atom in _split_logical_atoms(expr):
        split = _clock_threshold_split_oriented(atom, clock_symbols)
        if split is None:
            if not _IDENTIFIER.search(_inline_derived_param_refs(atom, thresholds.exprs) or atom):
                continue  # a literal comparison: constant, never crosses
            return (
                f"the atom {atom!r} in the trigger {trigger!r} is not a comparison of "
                "simulation time (or a unit-rate counter) against a threshold, so its "
                "crossing time is not a threshold bngsim can differentiate"
            )
        clock_sym, threshold_expr, kind = split
        # Everything that survives the flattening must be a primary parameter.
        # An identifier that does not — an assignment-rule parameter whose rule
        # is not arithmetic over parameters, a `floor()`/`ceil()` dose-schedule
        # counter, a species name — means the threshold is not a constant, and
        # reading its current value would hand back a plausible number attached
        # to the wrong parameter (see :func:`_threshold_scope`).
        thr_flat = _inline_derived_param_refs(threshold_expr, thresholds.exprs) or threshold_expr
        leftover = sorted(
            {m.group(0) for m in _IDENTIFIER.finditer(thr_flat)} - thresholds.primaries
        )
        if leftover:
            return (
                f"the threshold {threshold_expr!r} in the trigger {trigger!r} does not reduce "
                "to arithmetic over the model's primary parameters — "
                + ", ".join(repr(n) for n in leftover)
                + " is not one, so the threshold is not a constant and ∂t*/∂p cannot be "
                "attributed to a fitted parameter"
            )
        value = _evaluate_threshold(
            threshold_expr, scope.param_idx, scope.values, thresholds.exprs
        )
        if value is None:
            return (
                f"the threshold {threshold_expr!r} in the trigger {trigger!r} does not "
                "reduce to a constant expression over the model's primary parameters"
            )
        if clock_sym in _TIME_SYMBOLS:
            t_atom = value
        else:
            # dc/dt = 1, so c(t) = c(t_start) + (t − t_start) and the crossing is
            # offset by the clock's value at the start of the run.
            clock_idx0 = clocks[clock_sym]
            clock_now = core.get_concentration(core.species_names[clock_idx0])
            t_atom = t_start + (value - clock_now)
        if kind == "upper":
            upper.append(t_atom)
            continue
        partials = _derived_expr_partials_numeric(
            threshold_expr,
            set(thresholds.primaries),
            scope.param_idx,
            list(scope.values),
            thresholds.exprs,
            warn_on_failure=False,
        )
        lower.append((t_atom, partials))

    if not lower:
        # `time <= toff` alone: true from t_start, so the event fires at the
        # run's start (or not at all) and its firing time does not move.
        return None
    t_star = max(t for t, _ in lower)
    winners = [p for t, p in lower if t == t_star]
    if len(winners) > 1 and any(w != winners[0] for w in winners[1:]):
        return (
            f"two atoms of the trigger {trigger!r} put the rising edge at the same time "
            f"t={t_star:.6g} with different derivatives, so t*(p) has a kink there rather "
            "than a derivative"
        )
    if upper and t_star > min(upper):
        return None  # the true-interval is empty; the event never fires
    return t_star, winners[0]


# The only function names the clock solvers above put into a threshold
# expression. Anything else in one means it was not written by them.
_SOLVED_THRESHOLD_FUNCS = frozenset({"sqrt", "Abs"})


def _threshold_is_non_real(
    expr: str,
    param_idx: dict,
    values: Sequence[float],
    derived_exprs: dict[str, str],
) -> bool:
    """True when *expr* resolves at this parameter point but to a number that is
    not on the real line.

    :func:`_evaluate_threshold` answers ``None`` to two very different questions
    — "I could not read this threshold" and "I read it, and this crossing does
    not happen" — and the switch-time gate has to separate them (issue #421). A
    clock crossing solved in closed form goes non-real exactly where the crossing
    stops existing: ``time()^2 >= thresh`` at ``thresh = -4`` is true for the
    whole run, and the quadratic formula does the same thing over a whole region
    of parameter space, since the discriminant of ``(time()-5)^2 >= thresh`` turns
    negative as soon as ``thresh`` does. There the branch never flips, ``∂f/∂p``
    is a correct clean zero, and refusing the model would be refusing a gradient
    bngsim can compute.

    Deliberately narrow. A threshold reading a species, one whose parameters do
    not resolve, and a degenerate ``1/0`` all stay ``False`` here and so stay
    refused, because none of them is the statement that a crossing is absent.
    Shares one preparation and one parse with :func:`_evaluate_threshold`, which
    is the whole point of routing it through the same
    :func:`bngsim._codegen._derived_expr_value_numeric`: two paths that disagree
    about which expressions they can read is issue #105.
    """
    from bngsim._codegen import _derived_expr_value_numeric

    s = expr.strip()
    if s in param_idx:
        return False
    try:
        float(s)
        return False
    except ValueError:
        pass
    # Every name has to be a model parameter or one of the handful of functions
    # the clock solvers themselves emit. ``I`` is why this guard exists: sympy
    # reads a bare ``I`` as the imaginary unit, and an SIR model spells its
    # infected observable exactly that, so ``t >= sigma*I`` would otherwise
    # "resolve" to a non-real number and be read here as a crossing that does not
    # happen. It is a threshold over live state, it belongs to issue #150, and it
    # has to keep reaching it. Anything else unrecognised answers ``False`` too,
    # which is the conservative direction: the crossing stays refused.
    if any(
        m.group(0) not in param_idx and m.group(0) not in _SOLVED_THRESHOLD_FUNCS
        for m in _IDENTIFIER.finditer(s)
    ):
        return False
    value = _derived_expr_value_numeric(
        s,
        set(param_idx) - set(derived_exprs),
        derived_exprs.keys(),
        param_idx,
        values,
        allow_complex=True,
    )
    return isinstance(value, complex) and value.imag != 0.0


def _evaluate_threshold(
    expr: str,
    param_idx: dict,
    values: Sequence[float],
    derived_exprs: dict[str, str],
) -> float | None:
    """Numeric value of a threshold expression at the current parameter point.

    A bare parameter name (the common case: ``t>=sigma``) is a dict lookup. Any
    other arithmetic expression — including one over derived parameters — goes
    through the *same* preparation as :func:`_derived_expr_partials_numeric`,
    which since GH #108 means literally the same function
    (:func:`bngsim._codegen._prepare_derived_expr`, reached here through
    :func:`bngsim._codegen._derived_expr_value_numeric`): the ExprTk surface
    rewrite, the keyword/reserved symbol aliasing, and one parse. The two
    therefore cannot disagree about which expressions they can handle, so the
    value and its derivative always come from the same round trip — which is the
    whole point, since a caller that gets partials but no value (or the reverse)
    drops the crossing entirely (issue #105).

    Without the aliasing, a threshold that is *arithmetic over* a parameter whose
    name is a Python keyword (``del+gap``) fails at tokenization, so the value
    came back ``None`` while the partials of the identical expression came back
    correct. A *bare* keyword name never reached sympy, which is why only the
    arithmetic form was affected.

    Sharing the preparation also retired the last place this path flattened the
    derived-parameter DAG (GH #99 moved the partials off it and this site was
    missed): a derived name now substitutes its current value rather than the
    expression it stands for. ``ode/synthesis_v3``'s ``F0+Fh`` inlined to 61 KB
    and 1.2 s of sympy for a number the partials already had.
    """
    from bngsim._codegen import _derived_expr_value_numeric

    s = expr.strip()
    if s in param_idx:
        return float(values[param_idx[s]])
    try:
        return float(s)
    except ValueError:
        pass
    # Anything in `param_idx` that is not one of the expression-valued names is
    # a primary here, exactly as the partials twin is called (`thresholds.exprs`
    # also carries the rule-bound parameters, which are not constants either).
    value = _derived_expr_value_numeric(
        s,
        set(param_idx) - set(derived_exprs),
        derived_exprs.keys(),
        param_idx,
        values,
    )
    # `allow_complex` is off, so a complex is unreachable; the narrowing is for
    # the type checker, since the twin above turns that flag on.
    return None if isinstance(value, complex) else value
