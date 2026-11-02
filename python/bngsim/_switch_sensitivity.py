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

import ast
import bisect
import contextlib
import functools
import itertools
import logging
import math
import re
import sys
import weakref
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import NamedTuple

from bngsim import _term_order
from bngsim._codegen import (
    _BUILTIN_CONSTANT_VALUES,
    _DERIVED_BARE_NAME,
    _ORDERING_OPS,
    _derived_expr_partials_numeric,
    _find_close_paren_strict,
    _inline_derived_param_refs,
    _spell_mod,
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


_ATOMIC_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?")


def _fully_parenthesised(expr: str) -> bool:
    """Whether the opening ``(`` of *expr* is closed by its last character."""
    if not (expr.startswith("(") and expr.endswith(")")):
        return False
    return _find_close_paren_strict(expr, 0) == len(expr) - 1


def _is_name_or_call(text: str) -> bool:
    """A lone name or number, or a name applied to one parenthesised group."""
    if _ATOMIC_TOKEN.fullmatch(text):
        return True
    m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text)
    if m is None:
        return False
    return _fully_parenthesised(text[m.end() :])


def _condition_key(cond: str) -> str:
    """The text of a condition with the parentheses the SBML loader's
    ``_atomic`` adds around a call dropped, and those around a lone name or
    number, so that one crossing reached two ways has one key (issue #1082).

    A function-definition formal bound to ``time()`` is spliced into the body
    as ``(time())``, so ``near(time, 2.5, 0.005)`` with ``near := abs(x - c) <
    w`` and ``abs(time - 2.5) < 0.005`` written directly gave two spellings of
    each edge, and two roots. A group that is a call's own argument list is
    left alone, and so is any group with an operator in it. Lives here, beside
    :func:`_strip_redundant_parens`, so the loader that registers conditions
    and the recovery in ``Model.time_discontinuity_conditions`` compare on one
    form without the loader (and libsbml) being imported for it.
    """
    prev = None
    while prev != cond:
        prev = cond
        out: list[str] = []
        i = 0
        n = len(cond)
        while i < n:
            ch = cond[i]
            # An opening parenthesis that is not a call's own (no name before it).
            if ch == "(" and not (out and (out[-1][-1:].isalnum() or out[-1][-1:] == "_")):
                close = _find_close_paren_strict(cond, i)
                inner = cond[i + 1 : close] if close >= 0 else ""
                if close >= 0 and _is_name_or_call(inner):
                    out.append(inner)
                    i = close + 1
                    continue
            out.append(ch)
            i += 1
        cond = "".join(out)
    return cond


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


def _turning_side(expr: str) -> tuple[str, str] | None:
    """``("abs", u)`` for ``abs(u)``, ``("square", u)`` for ``(u)^2`` or ``u^2``
    with ``u`` one name or one parenthesized group; ``None`` otherwise.

    The group must close where the ``^`` stands. ``(A-5)^2+(B-5)^2`` and
    ``(A-5)*(B)^2`` each balance as a whole and end in ``^2``, but only the
    last group is squared; read as the square of everything before the ``^``,
    the disk ``(A-c)^2+(B-5)^2 < 0.75`` was rooted at a surface it does not
    have and lost its own crossing, and dY/dc came back 0 for -12.06.
    """
    text = _strip_redundant_parens(expr.strip())
    m = re.match(r"abs\s*\(", text)
    if m is not None and _find_close_paren_strict(text, m.end() - 1) == len(text) - 1:
        return "abs", text[m.end() : -1].strip()
    m = re.fullmatch(r"(\(.*\)|[A-Za-z_][A-Za-z0-9_]*)\s*\^\s*2(?:\.0*)?", text, re.S)
    if m is None:
        return None
    base = m.group(1)
    if base.startswith("(") and _find_close_paren_strict(base, 0) != len(base) - 1:
        return None
    return "square", _strip_redundant_parens(base).strip()


def _monotone_pieces(atom: str) -> list[str]:
    """The comparisons whose crossings are this atom's, one per turn of its
    residual, where the atom compares ``abs(u)`` or ``u^2`` with a bound.

    ``abs(A - 5) < 0.01`` is a window: its residual ``abs(A-5) - 0.01`` falls
    to zero and comes back up within one step as A passes 5, so a root on it
    sees the same sign at both ends of the step and reports nothing, and the
    window was stepped over (issue #904). Its edges are where ``A - 5`` crosses
    ``0.01`` and ``-0.01``, and each of those residuals is monotone along a
    flow that moves A one way. ``u^2`` against a bound is the same at
    ``±sqrt(bound)``, taken only for a bound that is a number, where the square
    root is one. A bound that is a number must be above 0: at or below it the
    atom holds at one point or nowhere, its edges would be restarts at
    crossings that switch nothing (two of them on one surface, at 0), and it is
    rooted whole as before. So is ``trig > abs(-1)``, whose ``abs()`` reads no
    name and so does not turn: its edges would put a restart at ``trig = -1``,
    where nothing switches. ``[]`` for any other atom. Parentheses around the
    whole atom are read through: the SBML loader writes ``(abs((A-5))<0.01)``.
    """
    split = _relational_split_op(_strip_redundant_parens(atom))
    if split is None:
        return []
    lhs, _op, rhs = split
    side = _turning_side(lhs)
    bound = rhs
    if side is None:
        side = _turning_side(rhs)
        bound = lhs
    if side is None or _turning_side(bound) is not None:
        return []
    kind, inner = side
    if not _IDENTIFIER.search(inner):
        return []
    try:
        value: float | None = float(_strip_redundant_parens(bound.strip()))
    except ValueError:
        value = None
    if value is not None and not (0.0 < value < math.inf):
        return []
    if kind == "abs":
        return [f"({inner})<({bound})", f"({inner})>(-({bound}))"]
    if value is None:
        return []
    root = repr(math.sqrt(value))
    return [f"({inner})<({root})", f"({inner})>(-{root})"]


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


class _SpanIndex:
    """Which character spans hold a match, answered by bisection (issue #1256).

    ``contains(start, end)`` is ``any(lo <= start and end <= hi for lo, hi in
    spans)``, and ``covers(start)`` is ``any(lo <= start < hi ...)``. Asked of
    every match of a pattern against every ``if()`` condition span, as four
    scans here did, that is a product of the two counts, and a rate law written
    out through a diamond of rules holds thousands of both: a sensitivity
    simulator for #1145's size jump took 298 s to build at 20 rules, and takes
    35 s with the spans indexed.

    With the spans sorted by start and the furthest end of those so far kept
    beside each, some span holds ``[start, end]`` exactly when, of the spans
    that start at or before ``start``, the furthest end is at or after ``end``:
    the same answer, in ``O(log n)``. (A match across two adjacent spans is in
    their union, but in neither, and is not held: the furthest end belongs to
    one span.)
    """

    __slots__ = ("_starts", "_furthest")

    def __init__(self, spans) -> None:
        ordered = sorted(spans)
        self._starts = [lo for lo, _hi in ordered]
        self._furthest = list(itertools.accumulate((hi for _lo, hi in ordered), max))

    def _reach(self, start: int):
        k = bisect.bisect_right(self._starts, start)
        return self._furthest[k - 1] if k else None

    def contains(self, start: int, end: int) -> bool:
        far = self._reach(start)
        return far is not None and far >= end

    def covers(self, start: int) -> bool:
        far = self._reach(start)
        return far is not None and far > start


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
    func_map = _scan_function_map(core.functional_jacobian_context())

    pats = {
        name: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        for name in candidates
    }
    pure = set(candidates)

    # A *derived* candidate cannot be judged on the fully inlined text, because
    # inlining is exactly what removes its name from it (issue #475). It is
    # asked on text where everything derived EXCEPT itself is inlined, so its
    # own occurrences stay visible while the primaries underneath the others are
    # still attributed. One flattening for all the primary candidates, one more
    # per derived candidate — and there are never many of those.
    #
    # It matters: BIOMD0000000636 and BIOMD0000000808 each write a derived
    # threshold parameter that also appears inside a branch. Read on the inlined
    # text they look condition-only, which on the difference-quotient path would
    # pin a parameter whose in-branch term is real and hold it at a wrong 0 —
    # the failure issue #358 refuses over.
    groups = [(set(candidates) - derived_exprs.keys(), derived_exprs)]
    groups += [
        ({name}, {k: v for k, v in derived_exprs.items() if k != name})
        for name in sorted(candidates & derived_exprs.keys())
    ]

    # The reaction rate laws, fully inlined. A helper function reaches the RHS
    # only where it is referenced, so inlining it into the rate law that uses it
    # puts each of its parameters in the branch or the condition it actually
    # occupies there — which scanning the helper on its own cannot know.
    bodies: list[str] = []
    for rxn in data["reactions"]:
        if rxn.get("type") != "functional":
            continue
        fname = rxn.get("function_name")
        body = func_map.get(fname) if fname else _spell_mod(rxn.get("rate_expression") or "")
        if body:
            bodies.append(_inline_functions(body, func_map) or body)

    # An elementary reaction's rate constant is read directly, never through a
    # condition. (A *functional* reaction's rate_param_indices entry is the
    # synthesized holder for the function's value, whose body is covered by the
    # rate-law scan above.)
    constants: list[str] = []
    for rxn in data["reactions"]:
        if rxn.get("type") != "elementary":
            continue
        for pidx in rxn.get("rate_param_indices", []):
            if 0 <= pidx < len(params):
                p = params[pidx]
                constants.append(p.get("expression", "") or p["name"])

    for group_names, exprs in groups:
        for body in bodies:
            asking = pure & group_names
            if not asking:
                break
            flat = _inline_derived_param_refs(body, exprs)
            if not flat:
                continue
            spans = _SpanIndex(_condition_spans(flat))
            for name in sorted(asking):
                for m in pats[name].finditer(flat):
                    if not spans.contains(m.start(), m.end()):
                        pure.discard(name)
                        break
        for text in constants:
            asking = pure & group_names
            if not asking:
                break
            flat = _inline_derived_param_refs(text, exprs)
            for name in sorted(asking):
                if pats[name].search(flat):
                    pure.discard(name)

    return pure


def _unit_rate_clock_indices(core, candidates: frozenset[int] | None = None) -> frozenset[int]:
    """Every species index obeying ``dc/dt = 1``, from two RHS probes.

    Split out of :func:`_unit_rate_clock_species` because it is the half that
    needs no ``functional_jacobian_context()``: two evaluations of the right-hand
    side and nothing else. :func:`time_discontinuity_conditions` asks the
    question in that cheap form, so that a model with a conditional rate law but
    no counter can be dismissed without building a context that runs to tens of
    thousands of entries on a large model.

    ``candidates`` is :func:`unit_rate_clock_candidates` at the model's current
    values, for a caller that has already probed them (issue #726).
    """
    if candidates is None:
        candidates = unit_rate_clock_candidates(core)
    if not candidates:
        return frozenset()
    return _structurally_unit_rate(core, set(candidates))


def texts_spell_a_switch(core) -> bool:
    """Whether any function or derived-parameter text spells a condition, a
    step, a comparison or a bend.

    The cheap first gate of :func:`time_discontinuity_conditions`, which returns
    ``()`` from the texts alone where this is false and never probes for a
    clock. :meth:`bngsim.Model.time_discontinuity_conditions` reads the same
    gate to decide whether its cache key needs the probe at all (issue #726),
    so the two cannot disagree about which models the clocks matter to. Read
    from the texts alone: structural.
    """
    from bngsim._jacobian import has_condition_construct

    if core.n_functions == 0:
        return False
    return any(
        has_condition_construct(t)
        or _STEP_CALL.search(_spell_mod(t))
        or _RELATIONAL.search(t)
        or _BEND_CALL.search(t)
        for t in (*core.function_expressions, *core.param_expressions)
    )


def unit_rate_clock_candidates(core) -> frozenset[int]:
    """The species indices at ``dc/dt = 1`` on both of two RHS probes.

    The half of :func:`_unit_rate_clock_indices` that parameter values decide:
    what follows it reads only structure. A counter fed at ``k_t`` passes at
    ``k_t = 1`` and not at 0 or 0.5, so whatever is derived from which species
    are clocks is derived again when this moves (issue #726). It costs two RHS
    evaluations, where the structural half reads the whole model.
    """
    import numpy as np

    if core.n_species == 0:
        return frozenset()
    conc = core.get_state()
    try:
        deriv = np.asarray(core._eval_rhs(0.0, conc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: RHS probe for clock detection failed: %s", exc)
        return frozenset()
    clock_idx = np.flatnonzero(deriv == _CLOCK_SLOPE)
    if clock_idx.size == 0:
        return frozenset()
    # A species whose RHS merely happens to equal 1 at the initial state is not
    # a clock, so probe again at a shifted state and time. That is only a
    # filter: shift-invariant and time-gated rates pass it, and
    # _structurally_unit_rate decides (issue #733).
    try:
        deriv2 = np.asarray(core._eval_rhs(1.0, conc + 1.0))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: second RHS probe failed: %s", exc)
        return frozenset()
    return frozenset(int(i) for i in clock_idx if deriv2[i] == _CLOCK_SLOPE)


def clock_volume_params(core) -> dict[int, str]:
    """For each amount-valued species whose compartment size is a live
    parameter, the name of that parameter.

    The other value :func:`_unit_rate_clock_species` reads besides the RHS: what
    a rate law reads under such a species' name is its stored value times the
    size, so it is a unit-rate clock only while the size is 1. A ``set_param`` on
    the size moves the verdict without moving the probes, so
    :meth:`bngsim.Model.time_discontinuity_conditions` keys its cache on these
    parameters' values as well (issue #726). Which species they are is
    structural; ``.net`` models have none.
    """
    names = list(core.param_names)
    out: dict[int, str] = {}
    for i, sp in enumerate(core.codegen_data()["species"]):
        k = int(sp.get("volume_param_idx0", -1))
        if sp.get("amount_valued", False) and 0 <= k < len(names):
            out[i] = names[k]
    return out


# The clock, read by a rate law that is therefore not constant. rateOf accessors
# (`rate_of__X`) are caught by their prefix.
_TIME_TOKENS = frozenset({"time"})


def _structurally_unit_rate(core, candidates: set[int]) -> frozenset[int]:
    """The candidates whose dc/dt is 1 for the whole run, by construction.

    Two RHS probes agreeing on 1 do not make a clock (issue #733). Three kinds
    of species pass them and are not one: a rate gated in time that happens to
    be 1 at t = 0 and t = 1 (``if(time() < Toff, 1, 0)``); a species an event
    assigns (an SBML timer reset ``x := 0``), whose value is not time plus a
    constant after the reset; and a species a shift of every state by 1 leaves
    at slope 1 (``A <-> B`` with kf = kr). Treated as clocks, their thresholds
    were jumped once at a crossing time computed as though c = c0 + t, with the
    wrong ∂t*/∂θ, and their real crossings were never rooted.

    A candidate is kept only when no event assigns it, no reaction consumes it
    on net, and every reaction producing it on net has a rate that is a
    constant: mass action with no reactants, or a rate law (an SBML kinetic law
    is the whole flux, whatever reactants it lists) that, inlined through every
    function and assignment rule, reads no species, observable, time, rateOf or
    table function. The probes already say that constant is 1.
    """
    from bngsim._jacobian import _IDENT_RE

    assigned = set(core.event_assigned_species)
    keep = {i for i in candidates if i not in assigned}
    if not keep:
        return frozenset()
    try:
        data = core.codegen_data()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: clock structure unavailable: %s", exc)
        return frozenset()
    func_map = {f["name"]: f["expression"] for f in data.get("functions", ())}
    # A table function reads its index (time, or an observable) outside its text.
    # A function body calls it through its accessor, `tfun_<name>()`.
    tables = {t["name"] for t in data.get("table_functions", ()) if isinstance(t, dict)}
    tables |= {f"tfun_{name}" for name in tables}
    param_expr = {p["name"]: p.get("expression", "") for p in data.get("parameters", ())}
    params = list(data.get("parameters", ()))
    state_names = set(core.species_names) | {
        ob["name"] for ob in data.get("observables", ()) if isinstance(ob, dict)
    }

    def reads_state_or_time(text: str, depth: int = 0) -> bool:
        if depth > 64:
            return True  # a cycle or a chain too deep to trust: not a clock
        for tok in _IDENT_RE.findall(text):
            if (
                tok in state_names
                or tok in _TIME_TOKENS
                or tok in tables
                or tok.startswith("rate_of__")
            ):
                return True
            body = func_map.get(tok)
            if body is None and param_expr.get(tok):
                body = param_expr[tok]
            if body is not None and reads_state_or_time(body, depth + 1):
                return True
        return False

    for rxn in data.get("reactions", ()):
        reactants = list(rxn.get("reactants", ()))
        products = list(rxn.get("products", ()))
        # Net stoichiometry, so a clock that only catalyses (Tc -> Tc + X, a
        # reactant and a product at once) keeps its slope and stays a clock.
        net = {i: products.count(i) - reactants.count(i) for i in keep}
        keep -= {i for i, n in net.items() if n < 0}  # consumed: slope follows state
        produced = {i for i, n in net.items() if n > 0 and i in keep}
        if not produced:
            continue
        if reactants and rxn.get("apply_species_factor", True):
            keep -= produced  # mass action: the reactants multiply the rate
            continue
        # Otherwise the rate law is the whole flux (an SBML kinetic law lists
        # its reactants but does not multiply by them), and its text decides.
        if rxn.get("type") == "functional":
            text = func_map.get(rxn.get("function_name") or "", "")
        else:
            idx = [i for i in rxn.get("rate_param_indices", ()) if 0 <= i < len(params)]
            text = " ".join(params[i]["name"] for i in idx)
        if not text or reads_state_or_time(text):
            keep -= produced
    return frozenset(keep)


def _unit_rate_clock_species(core, ctx=None, candidates=None) -> dict[str, int]:
    """Map every symbol that reads a unit-rate clock to its species index.

    A *clock* is a species obeying ``dc/dt = 1`` — the counter idiom BNGL uses to
    make simulation time available to rate laws (``counter()`` with a
    zeroth-order synthesis at rate 1, exposed through an observable, commonly
    named ``t``). Its value is time plus a constant offset, so a threshold on it
    is a threshold on time and the crossing time is known a priori.

    Both the observable name and the species name map to the same index: a
    condition may be written against either. Returns ``{}`` for the overwhelming
    majority of models, which have no such species. ``candidates`` is passed to
    :func:`_unit_rate_clock_indices`.
    """
    clock_idx = _unit_rate_clock_indices(core, candidates)
    if not clock_idx:
        return {}
    # What a rate law reads under a species' name is its amount when the species
    # is declared in substance units: the stored value times its compartment's
    # size. In a compartment of size 2 that moves at 2 per unit time and is no
    # unit-rate clock, though the stored value is. Read as one, its threshold
    # was crossed at the wrong time: an onset `T >= 3` was jumped at t = 3,
    # where it is at 1.5.
    from bngsim._codegen import _amount_volume_factors

    try:
        scaled, _live = _amount_volume_factors(core.codegen_data()["species"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: species volumes unavailable: %s", exc)
        return {}
    clock_idx = frozenset(i for i in clock_idx if scaled.get(i, 1.0) == 1.0)
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

# The calls the clock solvers legitimately need sympy's own meaning for: the
# step functions a schedule is written with, and what `_preprocess_derived_expr`
# rewrites `if`/`&&`/`||`/`!`/comparisons into.
_CLOCK_PARSE_CALLS = ("floor", "ceiling", "Piecewise", "And", "Or", "Not", "Eq", "Ne")


def _parse_clock_expr(text: str):
    """``(sympy_expr, dealias)`` for a clock-solver residual or condition, with
    every bare model name bound as a plain ``Symbol`` (issue #757).

    ``parse_expr`` with no ``local_dict`` resolves a bare name through sympy's
    own namespace first, so a model parameter spelled ``E`` came back as Euler's
    number, ``S`` as the ``S`` singleton, ``N`` as ``evalf``, ``Q`` as the
    assumptions object, ``O`` as ``Order``, and ``beta``/``gamma``/``zeta`` as
    special functions. A constant silently dropped the parameter from the solved
    threshold (``time()-E*E >= 0`` solved to ``exp(2)``, so ``dX/dE`` came back
    an exact 0); anything else raised, and the recognizer's decline cost a
    periodic schedule every stop on a plain run and turned an affine threshold
    into a spurious ``SensitivityUnsupportedError``. ``_codegen``'s
    ``_prepare_derived_expr`` binds every parameter for exactly this reason;
    these five sites came later and never did.

    A name followed by ``(`` is a call and keeps its default meaning (only the
    step functions and logic classes are pinned); every other name is a
    ``Symbol``. Python keywords are aliased so the text parses, and ``dealias``
    maps a printed result back to the model's names. Keywords only, not C
    reserved words too: these results are printed with ``str()``, never through
    ``sp.ccode``, which is the narrow side of issue #108's aliasing rule. A name
    used both ways, or two names that would share an alias, raises, which every
    caller already reads as a decline.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    from bngsim._codegen import (
        _PY_KEYWORD_PARAM_NAMES,
        _alias_keyword_param,
        _preprocess_derived_expr,
        _substitute_symbols_once,
    )
    from bngsim._jacobian import builtin_shadow_bindings

    s = _preprocess_derived_expr(text)
    calls: set[str] = set()
    names: set[str] = set()
    for m in _DERIVED_BARE_NAME.finditer(s):
        (calls if m.group(2) else names).add(m.group(1))
    names -= {"True", "False"}
    if names & calls:
        raise ValueError(f"{sorted(names & calls)} used both as a value and as a call")
    alias = {n: _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n for n in names}
    if len(set(alias.values())) != len(alias):
        raise ValueError("two names share a sympy alias")
    renamed = {n: a for n, a in alias.items() if n != a}
    if renamed:
        s = _substitute_symbols_once(s, renamed)

    local_dict: dict = {c: getattr(sp, c) for c in _CLOCK_PARSE_CALLS}
    # `round` would otherwise be Python's, which folds round(2.5) to 2 (#684).
    local_dict.update(builtin_shadow_bindings(sp))
    local_dict.update({a: sp.Symbol(a) for a in alias.values()})
    expr = parse_expr(s, local_dict=local_dict, evaluate=True)

    back = {a: n for n, a in renamed.items()}

    def dealias(printed: str) -> str:
        return _substitute_symbols_once(printed, back) if back else printed

    return expr, dealias


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

        expr, dealias = _parse_clock_expr(residual)
        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        if t not in expr.free_symbols:
            return None
        a = sp.diff(expr, t)
        if t in a.free_symbols or a == 0:
            return None  # not linear in the clock, or no crossing
        threshold = sp.simplify(-(expr - a * t) / a)
        if t in threshold.free_symbols:  # pragma: no cover - implied by linearity
            return None
        text = dealias(str(threshold).replace("**", "^"))
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

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        expr, dealias = _parse_clock_expr(residual)
        expr = sp.expand(expr)
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
        text = dealias(str(root).replace("**", "^"))
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

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        expr, dealias = _parse_clock_expr(residual)
        expr = sp.expand(expr)
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
        texts = [dealias(str(root).replace("**", "^")) for root in roots]
    except Exception as exc:  # noqa: BLE001 - an unsolvable atom is just declined
        logger.debug("clock quadratic solve declined %r: %s", atom, exc)
        return None

    if not all(_clock_free(text, clock_symbols) for text in texts):
        return None
    return clock_sym, texts


def clock_guard_cells(expr, clock_names: AbstractSet[str], sp) -> list[tuple]:
    """``[(expr_on_cell, guards, truth)]``: *expr* collapsed on each cell of the
    clock axis its clock guards cut it into (issue #545).

    A guard is a comparison of one clock with a number inside a ``Piecewise``
    condition — the year selection ``if(t<365, d_2021, if(t<730, 365+d_2022, …))``.
    ``guards`` is every one of them, in a fixed order, and ``truth`` says which
    side of each the cell is on; ``expr_on_cell`` has every guard replaced by that
    truth value, which is what collapses such a chain to the one branch the cell
    takes. The cells are the distinct truth assignments the guards take along the
    axis, probed at every guard value and between them, so an assignment that
    holds only exactly at a threshold (``t<730`` beside ``t<=730``) is a cell of its
    own. With no guards, or guards on more than one clock, *expr* is its own and
    only cell, with no guards.

    Read by the clock-threshold recognizer below and by the generator's comoving
    columns (``bngsim._codegen._functional_comoving_plan``), which both need an
    onset written through such a chain as the plain expression it is on the cell
    where it crosses.
    """
    guards = set()
    for pw in expr.atoms(sp.Piecewise):
        for _value, cond in pw.args:
            if isinstance(cond, bool) or cond in (sp.true, sp.false):
                continue
            for rel in cond.atoms(sp.core.relational.Relational):
                if isinstance(rel, (sp.Eq, sp.Ne)):
                    continue
                syms = rel.free_symbols
                if len(syms) != 1:
                    continue
                (sym,) = syms
                if sym.name not in clock_names:
                    continue
                if (rel.lhs == sym and rel.rhs.is_Number) or (
                    rel.rhs == sym and rel.lhs.is_Number
                ):
                    guards.add(rel)
    if not guards:
        return [(expr, (), ())]
    clocks = {next(iter(rel.free_symbols)) for rel in guards}
    if len(clocks) != 1:
        return [(expr, (), ())]
    (clock,) = clocks
    # (One clock against one number each: no sum in them for a key to tie on.)
    ordered = tuple(sorted(guards, key=sp.srepr))
    points = sorted({float(rel.rhs if rel.rhs.is_Number else rel.lhs) for rel in ordered})
    probes = [points[0] - 1.0]
    for lo, hi in zip(points, points[1:], strict=False):
        probes += [lo, 0.5 * (lo + hi)]
    probes += [points[-1], points[-1] + 1.0]
    cells: dict[tuple[bool, ...], tuple] = {}
    for x in probes:
        truth = tuple(bool(rel.subs(clock, x)) for rel in ordered)
        if truth in cells:
            continue
        on_cell = expr.xreplace(
            {
                rel: (sp.true if held else sp.false)
                for rel, held in zip(ordered, truth, strict=False)
            }
        )
        cells[truth] = (on_cell, ordered, truth)
    return list(cells.values())


class GuardedThreshold(NamedTuple):
    """One crossing candidate of a clock threshold written through clock guards
    (issue #545): the threshold on one cell of the guards, and that cell.

    A cell's threshold is a crossing only where the clock value it names falls in
    the cell itself — the else-branch of a year chain names a day in some other
    year, and the condition does not flip there — which is a question about the
    parameter point, so it is asked by the callers that hold one
    (:meth:`holds_at`), not by the value-free recognizer.
    """

    threshold: str
    clock: str
    guards: tuple
    truth: tuple

    def holds_at(self, clock_value: float) -> bool:
        import sympy as sp

        clock = sp.Symbol(self.clock)
        return all(
            bool(rel.subs(clock, clock_value)) == held
            for rel, held in zip(self.guards, self.truth, strict=True)
        )


_GUARDED_CACHE: dict[
    tuple[str, frozenset[str]], tuple[str, tuple[GuardedThreshold, ...]] | None
] = {}


def _clock_guarded_thresholds(
    atom: str, clock_symbols: AbstractSet[str]
) -> tuple[str, tuple[GuardedThreshold, ...]] | None:
    """``(clock_symbol, (GuardedThreshold, …))`` for an atom comparing a bare clock
    against something that reads the clock only through guards — else ``None``.

    Issue #545. A season's onset written as a year selection,
    ``t > if(t<365, 0+d_2021, if(t<730, 365+d_2022, …))``, crosses at a time the
    run knows in advance on every cell of its guards — ``d_2021``, ``365+d_2022``,
    … — but reads the clock on both sides, so :func:`_clock_threshold_split_bare`
    declines it and the issue #150 state path claims it as a switch over live state.
    That path can only root on the crossing, and CVODE tests for a root on a step it
    accepts: where the rate law past the onset is a power rising from zero, every
    step spanning it fails the error test, and the run wedges just short of a root
    it never reaches. Recognized here instead, each cell's threshold is a crossing
    the solver stops on exactly, with the ``∂t*/∂p`` of that cell.

    Deliberately narrow: exactly one side is a bare clock, the other side reads
    that clock only inside ``Piecewise`` conditions comparing it with a number
    (never in a value, never another clock), and every cell's threshold is free of
    the clock. Anything else returns ``None``, which is the status quo. Tried after
    every polynomial recognizer and before the schedules, so no atom recognized
    today changes path.
    """
    key = (atom, frozenset(clock_symbols))
    if key in _GUARDED_CACHE:
        return _GUARDED_CACHE[key]
    answer = None
    split = _relational_split(atom)
    if split is not None:
        lhs_bare = _strip_redundant_parens(split[0])
        rhs_bare = _strip_redundant_parens(split[1])
        lhs_clock = lhs_bare in clock_symbols
        rhs_clock = rhs_bare in clock_symbols
        if lhs_clock != rhs_clock:
            clock_sym, other = (lhs_bare, rhs_bare) if lhs_clock else (rhs_bare, lhs_bare)
            others = {c for c in clock_symbols if c != clock_sym}
            if not _clock_free(other, {clock_sym}) and _clock_free(other, others):
                answer = _guarded_cells_of(clock_sym, other)
    if len(_GUARDED_CACHE) >= _CROSSING_CACHE_MAX:
        _GUARDED_CACHE.clear()
    _GUARDED_CACHE[key] = answer
    return answer


def _guarded_cells_of(
    clock_sym: str, other: str
) -> tuple[str, tuple[GuardedThreshold, ...]] | None:
    try:
        import sympy as sp

        from bngsim._jacobian import _TIME_SYM, _exprtk_to_sympy, _value_symbol_names

        expr = _exprtk_to_sympy(other)
        if expr is None:
            return None
        clock_name = _TIME_SYM if clock_sym in _TIME_SYMBOLS else clock_sym
        if clock_name in _value_symbol_names(expr, sp):
            return None  # the clock in a value: not a threshold at all
        cells = clock_guard_cells(expr, {clock_name}, sp)
        if not cells[0][1]:
            return None  # read the clock, but not through a guard this can cut on
        out = []
        seen = set()
        for on_cell, guards, truth in cells:
            if clock_name in {s.name for s in on_cell.free_symbols}:
                return None
            text = str(on_cell).replace("**", "^")
            if (text, truth) in seen:
                continue
            seen.add((text, truth))
            out.append(GuardedThreshold(text, clock_name, guards, truth))
    except Exception as exc:  # noqa: BLE001 - an unreadable atom is just declined
        logger.debug("clock guarded threshold declined %r: %s", other, exc)
        return None
    return clock_sym, tuple(out)


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
      downstream can see which branch was taken. A guard that spells the same
      sign test as a comparison — ``clock < 0`` false, ``clock >= 0`` true — is
      resolved on the same ground and without that caveat (issue #473);
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
    # The same fact, for a guard that spells the sign test as a comparison rather
    # than as a call (issue #473). A run's clock domain is `[0, inf)`, so
    # `clock < 0` is false and `clock >= 0` is true for the whole of it. Both are
    # exact in a way `sign(clock) = 1` is not: they hold at `t = 0` too, so
    # neither carries the single-instant caveat written up on
    # :func:`_clock_guard_cannot_cross`.
    #
    # Only those two, on purpose. `clock > 0` and `clock <= 0` say the same thing
    # everywhere except at `t = 0`, where they are wrong, and what excuses the
    # `sign(0) = 0` caveat is an argument about the shape of libSBML's remainder
    # expansion — both of its branches are equal at 0 — not about clock guards in
    # general. Nothing here can check that for an arbitrary guard, so those two
    # keep declining.
    #
    # Collected from the relationals actually present rather than substituted by
    # a literal key, because `clock < 0` and `clock < 0.0` are different sympy
    # expressions and the corpus writes both. sympy puts the clock on the left
    # for either way round, so `0 > clock` arrives here as `clock < 0`.
    zero_tests = {
        rel: (sp.false if isinstance(rel, sp.StrictLessThan) else sp.true)
        for rel in cond.atoms(sp.StrictLessThan, sp.GreaterThan)
        if rel.lhs == t and rel.rhs.is_zero
    }
    if zero_tests:
        cond = cond.subs(zero_tests)
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
    """True when *atom* reads the clock only in ways that hold one truth value
    for the whole run — through ``sign()`` (issue #465), or as a comparison
    against zero (issue #473).

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
    once :func:`_guard_holds` has made its substitutions is claimed here, so
    ``time() >= sigma`` — whose clock survives them — is left to the recognizers
    exactly as before, and an atom with no clock in it at all is left to
    :func:`condition_cannot_cross`. Widening it to those would admit a derived
    parameter that function deliberately refuses, which is a different question
    from this one.

    The one instant the ``sign()`` substitution is wrong about is ``t = 0``,
    where ``sign(0)`` is 0. For the expansion this exists to read, both branches
    are equal there — a remainder is 0 at 0 either way — so nothing downstream
    can see it. A guard whose branches genuinely differ at exactly the run's
    first instant would be read as not crossing when it does; no corpus model
    writes one, and the schedule the guard sits inside is checked against the
    model's own residual afterwards regardless
    (:func:`_schedule_matches_residual`). The comparison spelling carries no such
    caveat, which is why only the two exact ones are claimed there.
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

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        text = _clock_symbol_sub(atom, present[0], _CLOCK_SOLVE_SYMBOL)
        cond, _ = _parse_clock_expr(_CEIL_CALL.sub("ceiling(", text))
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

        t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
        expr, dealias = _parse_clock_expr(_CEIL_CALL.sub("ceiling(", residual))
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
        texts = [dealias(str(e).replace("**", "^")) for e in (period, offset, duty)]
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


@functools.lru_cache(maxsize=16384)
def _whole_word(name: str) -> re.Pattern:
    """*name* as a whole word. Kept: the scope is built before every run that
    asks about a crossing, and a model with more parameters than :mod:`re`
    keeps patterns for compiled every one of them again each time, 20 ms a
    run at 800."""
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def switch_condition_scope(core, ctx=None, *, clock_candidates=None) -> SwitchConditionScope:
    """Assemble the model-level context both switch-condition callers read.

    One RHS probe for the clocks, one pass over the parameter table, shared by
    :func:`compute_switch_time_sens` and :func:`uncompensated_condition_reason`
    so neither can be built on a different view of the model than the other.
    ``clock_candidates``, when given, is that probe already made at the model's
    current values (:func:`unit_rate_clock_candidates`).
    """
    param_names = list(core.param_names)
    is_expr = list(core.param_is_expression)
    # As written, and not through _spell_mod: a derived parameter is a
    # constant of the run, so a `%` in one switches nothing, and the
    # resolvers that inline these texts decline a call they do not know.
    # `thr = base % 10` read as `mod(base, 10)` lost the exact crossing of
    # `time() > thr`, and with it the SSA's piecewise-constant rate (#1127).
    exprs = list(core.param_expressions)
    clocks = _unit_rate_clock_species(core, ctx, clock_candidates)
    clock_symbols = frozenset(clocks) | _TIME_SYMBOLS
    function_names = frozenset(core.function_names)
    return SwitchConditionScope(
        core=core,
        clocks=clocks,
        clock_symbols=clock_symbols,
        param_names=tuple(param_names),
        param_pats={n: _whole_word(n) for n in param_names},
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


def _iter_comparisons_anywhere(expr: str):
    """Every comparison in *expr*, whether or not an ``if()`` holds it.

    ExprTk reads a comparison as the number 1 or 0, so a window is also written
    with no ``if()`` at all: ``k*((time()>=100)*(time()<=140))``. The rate law
    jumps where such a comparison flips exactly as it does for a condition, and
    :func:`_iter_condition_atoms` saw none of them (issue #904).

    Each argument of each call and each parenthesized group is an expression of
    its own, cut at its commas and logical operators, and a piece whose own
    operator is a comparison is an atom. The groups inside an atom are read
    too, for a comparison used as a number inside another's side.

    One pass over the text with a stack of open groups, not a function that
    calls itself: a sum a thousand parentheses deep overran the interpreter's
    stack, the scan that calls this gave up on the whole model, and a window
    in another function of it was stepped over.
    """
    # Per open group: where its current piece starts, and whether that piece
    # has a comparison at the group's own depth.
    starts = [0]
    compared = [False]
    out: list[str] = []

    def close(end: int) -> None:
        if compared[-1]:
            atom = _strip_redundant_parens(expr[starts[-1] : end])
            if atom:
                out.append(atom)

    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c == "(":
            starts.append(i + 1)
            compared.append(False)
        elif c == ")":
            close(i)
            if len(starts) > 1:
                starts.pop()
                compared.pop()
        elif c == ",":
            close(i)
            starts[-1], compared[-1] = i + 1, False
        elif c in "<>=!":
            m = _RELATIONAL.match(expr, i)
            if m is not None:
                compared[-1] = True
                i = m.end()
                continue
        elif c in "&|anox":
            m = _LOGICAL.match(expr, i)
            if m is not None:
                close(i)
                starts[-1], compared[-1] = m.end(), False
                i = m.end()
                continue
        i += 1
    while starts:
        close(n)
        starts.pop()
        compared.pop()
    yield from out


# `time` written bare or as ExprTk's nullary call, as a whole word.
_TIME_REF = re.compile(r"(?<![A-Za-z0-9_])time(?:\s*\(\s*\))?(?![A-Za-z0-9_(])")

# How far off zero a residual may sit at the crossing it just predicted, as a
# fraction of the slope times the run's own time scale. This is a *linearity*
# check, not a root-finding tolerance: the two probes below already solve the
# linear case exactly, so anything failing it is a residual that is not linear
# in time, whose crossing this cannot claim to know.
_CROSSING_RESIDUAL_TOL = 1e-9

# Resolved crossing times, keyed on the text actually solved (function slots and
# derived names inlined), the run window and the values of every parameter that
# text reads — which is everything the answer depends on. The inlined text, not
# the condition as written: the cache outlives a model, and two models can spell
# `time()>=on` over an `on` defined as 2*a in one and 3*a in the other, which
# read the same leaf values and cross at different times (issue #729, the half
# #883 left when it fixed the schedule key). Two sympy round trips per condition
# is ~2 ms, and a scan or a fit calls run() thousands of times while the
# parameters a *schedule* reads (an experimental-condition dose time) change
# once per experiment, so the hit rate is close to 1. Bounded and cleared whole
# rather than LRU-evicted: the population is tiny and the cost of a cold miss is
# the 2 ms it always was.
_CROSSING_CACHE: dict[tuple, float | None] = {}
_CROSSING_CACHE_MAX = 4096


def _function_slot_bodies(ctx) -> dict[str, str]:
    """Every function a condition can name bare, mapped to its body.

    An SBML assignment rule such as ``model_time := time`` (GH #259) or
    ``g := 4*S`` is a function whose value the evaluator also writes into a
    parameter slot of the same name, and a condition can threshold that name.
    Read as the parameter its address is, the slot is a stale number:
    - a residual that moves with time comes back looking constant (#259);
    - one that moves with the state comes back as a crossing at a time nothing
      crosses (``time >= g`` stopped at 4 while the crossing was at 1.7).
    So the resolver inlines every one of them, and what the bodies read decides:
    time resolves, a parameter is read at its live value, and a species declines.
    """
    return {n: b for n, b in _scan_function_map(ctx).items() if isinstance(b, str)}


def _inline_function_slots(expr: str, bodies: dict[str, str]) -> str:
    """Substitute every *bare* reference to a function by its body, transitively.

    Bare only: ``f(x)`` is a call whose body takes an argument, and pasting the
    body over the call site would drop the argument. A call therefore survives
    the substitution and is caught by the caller's identifier check, which is the
    conservative direction. Only the names present are substituted, so a model
    with hundreds of functions pays for the few a condition reads.
    """
    for _ in range(len(bodies) + 1):
        present = set(_IDENTIFIER.findall(expr)) & bodies.keys()
        if not present:
            break
        before = expr
        for name in present:
            expr = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_(])",
                f"({bodies[name]})",
                expr,
            )
        if expr == before:
            break
    return expr


def _crossing_time_of_condition(
    cond: str,
    scope: SwitchConditionScope,
    t_start: float,
    t_end: float,
    bodies: dict[str, str] | None = None,
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
    bodies = bodies or {}
    # A function's slot is a plain parameter as far as the tables below are
    # concerned, and carries a stale number, so it has to be substituted BEFORE
    # anything reads a value — otherwise `model_time >= 0.7` looks constant, and
    # `time >= g` with `g := 4*S` looks like a crossing at a fixed time.
    # Substituting also keeps the #259 property that the alias spelling and the
    # csymbol spelling are the same run to the last bit.
    residual = _inline_function_slots(f"({lhs})-({rhs})", bodies)
    flat = _inline_derived_param_refs(residual, scope.derived_exprs) or residual
    if not _TIME_REF.search(flat):
        return None
    # Every other identifier must be a model parameter, and must not be a
    # function — one that survives inlining is a call this could only read as a
    # constant — and a species or observable name means the crossing moves with
    # the trajectory (issue #150's business, not a fixed stop time).
    read: list[str] = []
    for m in _IDENTIFIER.finditer(_TIME_REF.sub(" 0 ", flat)):
        if m.group(0) not in scope.param_idx or m.group(0) in bodies:
            return None
        read.append(m.group(0))

    # The window enters only as the scale the residual is probed over, and a
    # linear crossing does not depend on it. A power of two at or above the
    # window's reach keys it, so consecutive run_until legs (whose windows all
    # differ) share an answer instead of each re-parsing the residual three
    # times, which cost a leg ~1.5 ms of sympy.
    scale = 2.0 ** math.ceil(math.log2(max(abs(t_start), abs(t_end), 1.0)))
    key = (
        flat,
        scale,
        tuple(sorted((n, scope.values[scope.param_idx[n]]) for n in set(read))),
    )
    if key in _CROSSING_CACHE:
        return _CROSSING_CACHE[key]

    def at(t: float) -> float | None:
        return _evaluate_threshold(
            _TIME_REF.sub(f"({t!r})", residual),
            scope.param_idx,
            scope.values,
            scope.derived_exprs,
        )

    def resolve() -> float | None:
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

    Orderings, and an equality of things that hold their values. An equality
    of things that move is excluded, which is what the SBML scan does too — see
    the comment on the check.
    """
    if is_equality_atom(atom) and not (_STEP_CALL.search(atom) or _BEND_CALL.search(atom)):
        # `time() == T` is true for one instant of measure zero, so its branch
        # contributes nothing to the integral and there is nothing to miss.
        # Stopping there would be worse than not: the step that restarts AT the
        # crossing reads the rate law where the equality holds and carries that
        # value forward over a whole step. The SBML scan registers only
        # orderings for the same reason.
        #
        # `floor(time()/24) == 3` is true for a day: a window like any other,
        # which a step call makes of an equality. So does a call that holds its
        # value over a stretch: `max(abs(time()-120), 0.5) == 0.5`.
        return False
    flat = _inline_derived_param_refs(atom, scope.derived_exprs) or atom
    return _reads_clock_and_run_constants(flat, scope)


def _reads_clock_and_run_constants(flat: str, scope: SwitchConditionScope) -> bool:
    """True when *flat* reads a clock, and otherwise only run constants.

    The body of :func:`_switches_on_clock_alone`, shared with the step terms
    :func:`time_discontinuity_conditions` admits outside any condition (issue
    #869), which answer the same question about a ``floor()`` argument that
    this answers about a comparison. *flat* has its derived names inlined.
    """
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


def time_discontinuity_conditions(
    core, ctx=None, *, steps_only: bool = False, clock_candidates: frozenset[int] | None = None
) -> tuple[str, ...]:
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

    Since issue #869 it also returns each step call (``floor``, ``ceil``,
    ``mod``, ...) that reads a clock outside every condition, such as the
    ``floor(...)`` terms of a pulse written as a difference of floors. That is
    not a condition, but it is a jump at a knowable time, and the one consumer,
    :func:`fixed_crossing_stops`, places a stop at each of its jumps. A step
    inside another step's argument is returned as the outer call, whose jumps
    are the rate law's, rather than on its own.

    Empty for a model with no functions, and for the far more common model whose
    conditions read state rather than a clock, so nothing about its stepping
    changes.

    ``steps_only`` returns the step calls alone. The sensitivity detector reads
    the conditions itself, with what each threshold's crossing moves with, and
    needs only what it does not read (issue #944).

    ``clock_candidates`` is :func:`unit_rate_clock_candidates` already probed at
    the model's current values, which :meth:`bngsim.Model.time_discontinuity_conditions`
    has in hand for its cache key (issue #726); otherwise the scan probes once.
    """
    from bngsim._jacobian import _inline_functions, has_condition_construct

    if core.n_functions == 0:
        return ()
    # A table function read on the clock is a function of the time like any
    # builtin: `abs(tfun_G() - 2) < 0.01` is a window on the clock, wherever the
    # table passes 2. For the question of what a text reads, such a call is the
    # clock.
    time_tables = (
        {table["name"] for table in core.table_function_tables if table["index_kind"] == "time"}
        if core.n_table_functions and not steps_only
        else set()
    )

    def as_clock(text: str) -> str:
        if not time_tables:
            return text
        return _TABLE_CALL.sub(
            lambda m: "time()" if m.group(1) in time_tables else m.group(0), text
        )

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
        if not texts_spell_a_switch(core):
            return ()
        texts_raw = (*core.function_expressions, *core.param_expressions)
        if not any(_TIME_REF.search(t) for t in texts_raw) and not time_tables:
            if clock_candidates is None:
                clock_candidates = unit_rate_clock_candidates(core)
            if not _unit_rate_clock_indices(core, clock_candidates):
                return ()
        ctx = core.functional_jacobian_context()
    func_map = _scan_function_map(ctx)
    texts = [
        *func_map.values(),
        *(_scan_rate_law(r) for r in ctx["functional_reactions"]),
    ]
    conditional = [t for t in texts if has_condition_construct(t)]
    stepped = [t for t in texts if _STEP_CALL.search(t)]
    # A comparison standing as a number, outside every if() (issue #904).
    compared = [] if steps_only else [t for t in texts if _RELATIONAL.search(t)]
    # ...and a call that bends where a step call jumps.
    bent = [] if steps_only else [t for t in texts if _BEND_CALL.search(t)]
    if not conditional and not stepped and not compared and not bent:
        return ()
    try:
        scope = switch_condition_scope(core, ctx, clock_candidates=clock_candidates)
    except Exception as exc:  # pragma: no cover - defensive
        # Same fallback as the other model-level scans: with no scope nothing
        # can be classified, and the pre-#440 stepping is what a model gets.
        logger.debug("time-discontinuity scan: scope unavailable (%s)", exc)
        return ()

    found: list[str] = []
    for text in conditional:
        flat = _inline_functions(text, func_map) or text
        for atom in _iter_condition_atoms(flat):
            if atom not in found and _switches_on_clock_alone(as_clock(atom), scope):
                found.append(atom)
    # The comparisons standing as numbers that are conditions on the clock. A
    # step or bending call inside one reaches the rate law through it alone,
    # as one inside an if() condition does.
    numbers: list[str] = []
    for text in compared:
        flat = _inline_functions(text, func_map) or text
        for atom in _iter_comparisons_anywhere(flat):
            if _switches_on_clock_alone(as_clock(atom), scope):
                if atom not in numbers:
                    numbers.append(atom)
                if atom not in found:
                    found.append(atom)

    def inside_a_number(text: str) -> list[tuple[int, int]]:
        spans = []
        for atom in numbers:
            start = text.find(atom)
            while start >= 0:
                spans.append((start, start + len(atom)))
                start = text.find(atom, start + 1)
        return spans

    # A step call outside every condition jumps the rate law itself (issue
    # #869): `floor((time()-t0)/P) - floor((time()-t0-w)/P)` is a unit pulse
    # written with no comparison at all. Between its jumps the rate is flat, so
    # nothing stops CVODE's step from growing across a whole pulse. The call is
    # handed over as its own entry, which :func:`_step_edge_stop_times` reads.
    #
    # The outermost call, not every call. A step inside another step's argument
    # reaches the rate law only through the outer one, and its own jumps are
    # not jumps of the rate law: `floor(floor(time()+phase)/cyclePeriod)`, the
    # day counter of the Pokhilko/Adams circadian light functions, changes once
    # a cycle while its inner floor changes every hour, and a stop at each of
    # those hours was a CVODE restart for nothing (4x the Jacobian evaluations
    # on BIOMD0000000412). Only a call that cannot be admitted whole, because
    # it reads state, is looked inside, for the steps of time it holds.
    def admit(text: str) -> None:
        for call, arg in _iter_step_calls(text, outermost=True, skip=inside_a_number(text)):
            inlined = _inline_derived_param_refs(call, scope.derived_exprs) or call
            if _reads_clock_and_run_constants(as_clock(inlined), scope):
                if call not in found:
                    found.append(call)
            else:
                admit(arg)

    n_conditions = len(found)
    for text in stepped:
        admit(_inline_functions(text, func_map) or text)

    # A bending call of the clock outside every condition, by the same rule: the
    # outermost that reads the clock and run constants alone, and inside one
    # that reads state, the calls of time it holds.
    def admit_bends(text: str) -> None:
        for call, arg in _iter_step_calls(
            text, outermost=True, pattern=_BEND_CALL, skip=inside_a_number(text)
        ):
            inlined = _inline_derived_param_refs(call, scope.derived_exprs) or call
            if _reads_clock_and_run_constants(as_clock(inlined), scope):
                if call not in found:
                    found.append(call)
            else:
                admit_bends(arg)

    for text in bent:
        admit_bends(_inline_functions(text, func_map) or text)
    return tuple(found[n_conditions:] if steps_only else found)


# Resolved schedule and step-edge lists, keyed by :func:`_stop_memo_key`: the
# condition text with derived names inlined, the run window, and the value of
# every parameter that text reads. Recognizing a schedule and
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
    catches a schedule the recognizer mis-read (before issue #757, sympy's
    namespace folded ``I*I`` to ``-1`` for a parameter named ``I``), and placing
    stops where the model has no edge would be a pure perturbation of its
    stepping.
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
    return _memoized_stops(
        _stop_memo_key(atom, scope, t_start, t_end),
        lambda: _resolve_schedule_stop_times(atom, scope, t_start, t_end),
    )


def _memoized_stops(key, resolve):
    """*resolve*'s answer under *key* in :data:`_SCHEDULE_CACHE`. A condition
    over the run's budget is remembered as that, and raises again."""
    if key in _SCHEDULE_CACHE:
        answer = _SCHEDULE_CACHE[key]
    else:
        try:
            answer = resolve()
        except _OverStopBudget as over:
            answer = over
        if len(_SCHEDULE_CACHE) >= _SCHEDULE_CACHE_MAX:
            _SCHEDULE_CACHE.clear()
        _SCHEDULE_CACHE[key] = answer
    if isinstance(answer, _OverStopBudget):
        raise answer
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
    # Two edges a period. Past the run's budget of stops none is listed, and
    # the caller bounds the step by half the shorter of the two stretches a
    # period is cut into, which no step can then span (issue #890).
    count = 2.0 * (t_end - t_start) / abs(period)
    if not math.isfinite(count):
        return None
    if count > _RUN_STOP_BUDGET:
        raise _OverStopBudget(count, min(abs(duty), abs(period) - abs(duty)))
    return _schedule_edge_times(terms, t_start, t_end)


def _step_edge_stop_times(
    cond: str, scope: SwitchConditionScope, t_start: float, t_end: float
) -> list[float] | None:
    """Every instant in ``(t_start, t_end]`` where *cond* changes value because a
    step function of time in it jumps or its residual crosses zero between two
    jumps, or ``None`` when *cond* is not one this can read (issue #869).

    :func:`_schedule_stop_times` places a repeating schedule from its period,
    offset and duty, and so only reads a condition with exactly one ``floor``.
    A pulse is also written with two, as ``floor((time()-t0)/P) -
    floor((time()-t0-w)/P) > 0.5``, with ``mod``, as ``mod(time()-t0, P) < w``,
    or as the difference of floors bare, with no comparison at all. None of
    those was stopped at, so CVODE stepped over every pulse.

    This reads them all the same way. Every ``floor``, ``ceil``, ``rint`` and
    ``mod`` in *cond* must take an argument affine in time, so its jumps can be
    listed. Between two consecutive jumps each of them is a constant, and a
    comparison's residual must then be affine in time too, which gives its one
    crossing in that interval exactly. A jump is kept only where the value it
    feeds changes, the comparison's truth or the bare step term itself.

    *cond* is a comparison atom or a bare step call as
    :func:`time_discontinuity_conditions` returns it, with any counter clock
    already rewritten onto ``time()``. Memoized on the same key as
    :func:`_schedule_stop_times`.
    """
    atom = _strip_redundant_parens(cond.strip())

    def resolve():
        try:
            return _resolve_step_edge_stop_times(atom, scope, t_start, t_end)
        except _OverStopBudget:
            raise
        except Exception as exc:  # noqa: BLE001 - an unreadable atom is just declined
            logger.debug("step-edge stops declined %r: %s", atom, exc)
            return None

    return _memoized_stops(("step-edges", *_stop_memo_key(atom, scope, t_start, t_end)), resolve)


def _resolve_step_edge_stop_times(
    atom: str, scope: SwitchConditionScope, t_start: float, t_end: float
) -> list[float] | None:
    """:func:`_step_edge_stop_times` without the memo, on an unwrapped atom."""
    import sympy as sp
    from sympy.core.function import AppliedUndef

    split = _relational_split_op(atom)
    if split is not None:
        lhs, op, rhs = split
        # The truth of the comparison, read off the sign of ``lhs - rhs``.
        compare = _ORDERING_OPS.get(op)
        if compare is None:
            return None  # an equality; see _switches_on_clock_alone
        text = f"({lhs})-({rhs})"
    else:
        compare = None
        text = atom
    text = _inline_derived_param_refs(text, scope.derived_exprs) or text
    for sym in sorted(_TIME_SYMBOLS, key=len, reverse=True):
        text = _clock_symbol_sub(text, sym, _CLOCK_SOLVE_SYMBOL)
    if not _clock_free(text, scope.clock_symbols):
        return None  # a counter clock the caller did not rewrite

    expr, dealias = _parse_clock_expr(_CEIL_CALL.sub("ceiling(", text))
    t = sp.Symbol(_CLOCK_SOLVE_SYMBOL)
    values = {}
    for sym in expr.free_symbols - {t}:
        name = dealias(str(sym))
        if name in scope.run_constants:
            values[sym] = float(scope.values[scope.param_idx[name]])
        elif name in _BUILTIN_CONSTANT_VALUES:
            values[sym] = float(_BUILTIN_CONSTANT_VALUES[name])
        else:
            return None  # reads something that moves during the run

    # A mod of constants is a number, and C's fmod is what the evaluator
    # computes it with. Left as `_bng_trunc(24/7)` below it never became one,
    # and the atom was declined (issue #899). Folded before the values go in,
    # so that a whole number stands in the expression as a literal `3` would,
    # and sympy takes it out of the floor it sits in: added to a decimal first,
    # it moved every edge read off that floor by the rounding of the sum.
    while True:
        constant = [
            e
            for e in expr.atoms(AppliedUndef)
            if e.func.__name__ == "mod"
            and len(e.args) == 2
            and t not in e.free_symbols
            and not any(x.func.__name__ == "mod" for x in e.atoms(AppliedUndef) - {e})
        ]
        if not constant:
            break
        for e in constant:
            value = math.fmod(*(float(arg.subs(values)) for arg in e.args))
            expr = expr.xreplace(
                {e: sp.Integer(int(value)) if value == int(value) else sp.Float(value)}
            )
    expr = expr.replace(lambda e: isinstance(e, sp.floor), lambda e: sp.floor(*e.args))
    expr = expr.subs(values)

    # One step function per kind of jump. `rint` is BNG's floor(x + 0.5) and
    # `ceil(x)` is -floor(-x), both exactly; ExprTk's `mod` is C's fmod,
    # a - b*trunc(a/b), whose trunc jumps at every nonzero integer and not at 0.
    trunc = sp.Function("_bng_trunc")
    expr = expr.replace(sp.ceiling, lambda x: -sp.floor(-x))
    while True:
        # Innermost first; a `_bng_trunc` already rewritten inside does not
        # count, so `mod(mod(time(), 24), 12)` is read too.
        calls = [
            e
            for e in expr.atoms(AppliedUndef)
            if e.func.__name__ in ("rint", "mod")
            and not any(x.func.__name__ in ("rint", "mod") for x in e.atoms(AppliedUndef) - {e})
        ]
        if not calls:
            break
        for e in calls:
            if e.func.__name__ == "rint" and len(e.args) == 1:
                expr = expr.xreplace({e: sp.floor(e.args[0] + sp.Rational(1, 2))})
            elif e.func.__name__ == "mod" and len(e.args) == 2:
                a, b = e.args
                expr = expr.xreplace({e: a - b * trunc(a / b)})
            else:
                return None
    if expr.has(sp.Piecewise) or any(e.func is not trunc for e in expr.atoms(AppliedUndef)):
        return None  # a nested if(), or a call this does not know the jumps of

    steps = [e for e in expr.atoms(sp.floor) | expr.atoms(trunc) if t in e.free_symbols]
    if not steps:
        return None

    # A step may hold other steps in its argument, as the day counter
    # `floor(floor(time()+phase)/24)` does. Each argument is read as affine in
    # time with the steps inside it held, so its crossings are listed piece by
    # piece between theirs. A step's argument holds only steps with fewer steps
    # in their own, so ordering by that count reads every inner step first.
    def n_inner(e) -> int:
        return len(e.args[0].atoms(sp.floor) | e.args[0].atoms(trunc))

    # Within a count, in an order with no ties: the steps come from a set.
    by_count: dict[int, list] = {}
    for e in steps:
        by_count.setdefault(n_inner(e), []).append(e)
    steps = [e for n in sorted(by_count) for e in _term_order.in_order(by_count[n])]
    n_leaves = sum(1 for e in steps if n_inner(e) == 0)
    marks = [sp.Symbol(f"_bng_step_{i}") for i in range(len(steps))]
    to_mark = dict(zip(steps, marks, strict=True))

    # Every coefficient below is read off an expanded sympy form, which is not
    # the model's arithmetic: `floor(t)/24` is `0.041666666666666664*m` there,
    # and `x/49` is `(1/49)*x`, so a value the model computes as exactly 1 or 0
    # comes out an ulp or two away, and `1/49*49` floors to 0. Each is therefore
    # read with the sum of its terms' magnitudes, and a value within
    # :data:`_ROUNDING` of that scale from the integer (or zero) it stands for
    # is that integer. A decision to the exact bit, `a == 0.0` or `floor(v)`,
    # would otherwise take the rounding's side.
    def affine(expr):
        """*expr* and the size of its terms, as functions of the steps' values."""
        size = sp.Add(*(sp.Abs(term) for term in sp.Add.make_args(expr)))
        return _lambdify_doubles(marks, expr), _lambdify_doubles(marks, size)

    def read(part, held: list[float], to_integer: bool) -> float:
        """*part* at *held*, snapped to the integer (or zero) it is rounding away from."""
        value_of, size_of = part
        v = float(value_of(*held))
        r = float(round(v)) if to_integer else 0.0
        return r if abs(v - r) <= _ROUNDING * float(size_of(*held)) else v

    args = []  # (slope, intercept) of each step's argument, over the inner steps' values
    for e in steps:
        u = sp.expand(e.args[0].xreplace(to_mark))
        alpha = sp.diff(u, t)
        beta = sp.expand(u - alpha * t)
        if t in alpha.free_symbols or t in beta.free_symbols:
            return None  # the argument is not affine in time
        if not (alpha.free_symbols or beta.free_symbols):
            alpha_f, beta_f = float(alpha), float(beta)
            if alpha_f == 0.0 or not (math.isfinite(alpha_f) and math.isfinite(beta_f)):
                return None
        args.append((affine(alpha), affine(beta)))
    is_trunc = [e.func is trunc for e in steps]

    def held_at(x: float, settled: int) -> list[float]:
        """Every step's value at *x*, the middle of a piece, which is on no
        edge. *settled* is how many of the steps have their edges listed where
        the piece is between two of those, and 0 where one of its ends is an
        end of the run.

        Where one of the settled steps is within a few last places of this
        arithmetic of a whole number at the middle of a piece between two
        edges all the same, the two are closer than it can tell apart, and the
        atom is declined: at `1e12 + time()` an argument is read to 5e-4 of a
        level, `floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)` is one for 1.5e-4
        before each wrap, and the middle of that pulse read as past the wrap,
        for 79 of 800 pulses found in a run of 240.

        Not asked of the piece at either end of the run. A run that begins a
        few doubles before an edge (the next leg of a run made in legs) has
        its first piece that short for no fault of the arithmetic:
        `floor(time() - 1e9)` from eight doubles before a whole number was
        declined, the scan takes a change that close to the start for the
        start, and the run died for want of the stop.
        """
        held = [0.0] * len(steps)
        for i, (slope, intercept) in enumerate(args):
            a = read(slope, held, to_integer=False)
            if a == 0.0:
                # Flat between the inner steps' jumps, where it can sit on an
                # integer for a whole piece: `floor(t)/24` is 1 on [24, 25).
                v = read(intercept, held, to_integer=True)
            else:
                v = a * x + float(intercept[0](*held))
                if i < settled and abs(v - round(v)) <= 4.0 * _EPS * (
                    abs(a * x) + float(intercept[1](*held))
                ):
                    raise ValueError(f"two edges within the rounding of {steps[i]} about t={x!r}")
            held[i] = float(math.trunc(v) if is_trunc[i] else math.floor(v))
        return held

    # The jumps: each argument's integer crossings in (t_start, t_end]. A step of
    # time itself is read over the whole window at once, and a step holding
    # others piece by piece between the edges already found, theirs among them.
    edges: list[float] = []
    for i, (slope, intercept) in enumerate(args):
        bounds = [t_start, t_end] if i < n_leaves else [t_start, *edges, t_end]
        jumps: list[float] = []
        for k, (lo, hi) in enumerate(zip(bounds, bounds[1:], strict=False)):
            if hi <= lo:
                continue
            between_edges = i >= n_leaves and 0 < k < len(bounds) - 2
            held = held_at(0.5 * (lo + hi), i if between_edges else 0)
            a = read(slope, held, to_integer=False)
            if a == 0.0:
                continue  # flat here: it can only jump where an inner step does
            b = float(intercept[0](*held))
            u0, u1 = a * lo + b, a * hi + b
            n_lo, n_hi = math.ceil(min(u0, u1)), math.floor(max(u0, u1))
            if n_hi - n_lo + 1 > _RUN_STOP_BUDGET - len(edges) - len(jumps):
                # Past the run's budget of stops (issue #890). The count is of
                # the edges this argument could make, which for a step inside
                # another is more than the jumps the outer one keeps.
                raise _OverStopBudget(len(edges) + len(jumps) + n_hi - n_lo + 1)
            for n in range(n_lo, n_hi + 1):
                if is_trunc[i] and n == 0:
                    continue
                tn = (n - b) / a
                if lo < tn <= hi:
                    jumps.append(tn)
        merged: list[float] = []
        for tn in sorted([*edges, *jumps]):
            if not merged or not _same_instant(tn, merged[-1]):
                merged.append(tn)
        edges = merged

    # Between two edges every step is a constant. Read each interval at its
    # midpoint, never on an edge, so a step's value there is unambiguous.
    flat = sp.expand(expr.xreplace(to_mark))
    slope = sp.diff(flat, t)
    if t in slope.free_symbols:
        return None  # not affine in time between jumps: nothing here solves it
    intercept = sp.expand(flat - slope * t)
    if t in intercept.free_symbols:
        return None
    residual_slope, residual_intercept = affine(slope), affine(intercept)

    def piece(lo: float, hi: float, between_edges: bool) -> tuple[float, float, float]:
        """The residual's slope and intercept on the piece, and the size of the
        intercept's terms."""
        held = held_at(0.5 * (lo + hi), len(steps) if between_edges else 0)
        a = read(residual_slope, held, to_integer=False)
        if a == 0.0:
            return 0.0, read(residual_intercept, held, to_integer=False), 0.0
        return (
            a,
            float(residual_intercept[0](*held)),
            float(residual_intercept[1](*held)),
        )

    def at(a: float, b: float, size: float, x: float) -> float:
        """The residual at the bound *x*, 0 where it is rounding away from 0."""
        v = a * x + b
        return 0.0 if abs(v) <= _ROUNDING * (abs(a * x) + size) else v

    bounds = [t_start, *edges]
    if edges[-1:] != [t_end]:
        bounds.append(t_end)
    edge_set = set(edges)
    out: list[float] = []
    before: float | None = None  # the limit just before the current edge
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        a, b, size = piece(lo, hi, i > 0 and bounds[i + 1] in edge_set)
        at_lo, at_hi = at(a, b, size, lo), at(a, b, size, hi)
        if not (math.isfinite(at_lo) and math.isfinite(at_hi)):
            return None
        if compare is not None:
            # A comparison is judged on its one-sided limits inside the piece,
            # not on its value at a bound. A residual that is exactly 0 at an
            # edge holds that value for one instant, and the side it is on is
            # the side its slope comes from: `mod(time(), 24) < 24` reaches 0
            # only at each jump and is true everywhere else, so nothing changes.
            if at_lo == 0.0:
                at_lo = a
            if at_hi == 0.0:
                at_hi = -a
        if before is not None:
            if compare is not None:
                changed = compare(before, 0.0) != compare(at_lo, 0.0)
            else:
                changed = before != at_lo
            if changed:
                out.append(lo)
        if compare is not None and a != 0.0 and compare(at_lo, 0.0) != compare(at_hi, 0.0):
            root = -b / a
            if lo < root <= hi:
                out.append(root)
        before = at_hi
    return sorted(set(out))


def _inner_step_stop_times(
    cond: str, scope: SwitchConditionScope, t_start: float, t_end: float
) -> list[float] | None:
    """The stops of the step calls inside a bare step call that
    :func:`_step_edge_stop_times` could not read, or ``None``.

    :func:`time_discontinuity_conditions` hands over a step inside another
    step's argument as the outer call. When the outer one is a call nothing
    here knows the jumps of, the inner calls are stopped at instead, each read
    the same way, which is what main placed for such a model. That covers every
    jump the outer call makes where an inner one does, at the cost of a restart
    where it does not change, and not a jump it makes between them:
    `sign(mod(time(), P) - 12)` jumps at 12 as well as at each period. A
    comparison is not looked inside: its truth is the condition, and a jump of
    a step within it need not move that.
    """
    text = _strip_redundant_parens(cond.strip())
    m = _STEP_CALL.match(text)
    if m is None or _find_close_paren_strict(text, m.end() - 1) != len(text) - 1:
        return None
    found: list[float] | None = None
    for call, _arg in _iter_step_calls(text[m.end() : -1], outermost=True):
        times = _step_edge_stop_times(call, scope, t_start, t_end)
        if times is None:
            times = _inner_step_stop_times(call, scope, t_start, t_end)
        if times is not None:
            found = sorted({*(found or ()), *times})
    return found


def _stop_memo_key(atom: str, scope: SwitchConditionScope, t_start: float, t_end: float):
    """The memo key the stop resolvers share: the text with derived names
    inlined, the run window, and the value of every parameter that text reads,
    with whether it is a run constant.

    The inlined text, not the atom as written. The cache outlives a model, and
    two models can spell one atom over a derived name defined differently in
    each (``onset = t0 + 1`` in one, ``t0 * 5`` in the other), which read the
    same leaf values and still have different edges."""
    flat = _inline_derived_param_refs(atom, scope.derived_exprs) or atom
    read = sorted({m.group(0) for m in _IDENTIFIER.finditer(flat)} & scope.param_idx.keys())
    return (
        flat,
        t_start,
        t_end,
        tuple((n, scope.values[scope.param_idx[n]], n in scope.run_constants) for n in read),
    )


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
    # A stop ON a touch of the threshold (issue #714): the condition is true at
    # that one double alone, so the stop is never merged into another.
    touch: bool = False
    # Whether the integrator restarts here. Not where the right-hand side only
    # bends, at a breakpoint of a `linear` table function (issue #709): there
    # the step is made to end on the stop and the run keeps its history.
    restart: bool = True
    # A stop moved onto the double at which its condition changes for the
    # evaluator (issue #891), and where the resolvers had put it (None for one
    # they did not list); see :func:`merge_crossing_stops`.
    on_flip: bool = False
    placed: float | None = None


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
    return merge_crossing_stops(all_fixed_crossings(core, t_start, t_end, conditions))


def all_fixed_crossings(core, t_start: float, t_end: float, conditions=()) -> list[CrossingStop]:
    """Every crossing :func:`fixed_crossing_stops` resolves, before the ones that
    share an instant are merged to one stop: one entry per condition and
    crossing, in time order.

    The integration loop stops once at an instant, so it takes the merged list.
    The event sensitivity jump asks each fixed switch on an event's instant for
    its own jump (issue #767), and two conditions that cross together, one on
    each of two counters, are two switches.
    """
    return _resolve_crossings(core, t_start, t_end, conditions, periodic=False)[0]


def crossings_with_periodic(
    core,
    t_start: float,
    t_end: float,
    conditions=(),
    touches=frozenset(),
    over_budget=None,
    sensitivity_params=None,
    on_the_flip=False,
    before_the_flip=False,
    by_tread=False,
) -> tuple[list[CrossingStop], list[CrossingStop], list[CrossingStop]]:
    """:func:`all_fixed_crossings`, and for the conditions it leaves out, their
    stops and their crossings, from one pass over the conditions.

    A condition whose residual is a sinusoid or a polynomial in time or in a
    counter (:func:`_periodic_stop_times`) is not monotone in time, so none of
    the fixed resolvers places it, and a window narrower than one integrator
    step, or than one panel of the SSA's continuous loop, can lie wholly
    between two of their nodes and be stepped over without a trace (issue
    #714). Its crossings are constants of the run all the same. A stop lands a
    little past its crossing (see :func:`_periodic_stop_times`); the crossing
    itself is what the event sensitivity jump asks about an instant.

    A condition of any other shape that reads the clock and run constants alone
    is evaluated over the window with the model's own evaluator, and stopped at
    wherever its value changes (:func:`_engine_scan_stop_times`, issue #904).
    What a fixed resolver placed is checked against the evaluator the same way.

    *touches* are the conditions an event trigger reads; a touch of the
    threshold without a crossing is a stop for these alone, and so is the one
    instant an equality holds at.

    *over_budget*, a list, receives a :class:`StopsOverBudget` for each
    condition that changes more often than the run's budget of stops allows
    (issue #890). None of that condition's stops is placed, and the caller
    bounds the step or refuses the run.

    *sensitivity_params* is ``None`` for a run with no sensitivity columns, and
    otherwise the parameters the run differentiates (empty for initial
    conditions alone). Such a run classifies every condition itself: as a
    switch time it has a derivative for, a root with a jump, or a refusal. A
    condition on a counter that no resolver reads (``exp(t) >= 20``) is one of
    its roots, and a stop on the same instant would be a second reading of that
    switch. So there the evaluator is asked only about a condition on literal
    time that reads none of those parameters: its changes are fixed times of
    the run, with no jump to make, and without the stop the columns were those
    of the stepped-over trajectory.

    *on_the_flip* moves each periodic stop onto the double the evaluator
    changes on (issue #891, :func:`_stops_on_the_flip`); the ODE path asks for
    it in a run with no sensitivity columns. Not the SSA's: there a step rate on
    `1e12 + t` switched a double of the clock, 1.2e-4 of time, later than the
    same run from the origin, and the run far from it was no longer that run.

    *before_the_flip* moves each such stop, and each the scan places on the
    evaluator's change, onto the last double before that change, on literal
    time: the ODE path asks for it at the tolerances that are not on the flip
    (issue #1226). A step ends on a stop, and the implicit step reads the rate
    at its end: on the first double of the next stretch it is taken at the
    next stretch's rate, and where the tolerance lets it span the stretch,
    `if(mod(time(), 24) < 1, 1, 0)` over 1,000 days at rtol 1e-2 came to
    14,627 for 1,000. A double earlier it is exact to rounding. Only for a
    condition whose rate laws hold between its changes: one that reads the
    clock outside every condition that holds (a sawtooth, a ramp, a pulse
    times the clock) keeps its stops where they were put
    (`changes_between_its_stops` in :func:`_resolve_crossings`).

    *by_tread* reads a condition on a line far from zero tread by tread where
    the scan refuses it or leaves it without stops (issue #1018,
    :func:`_tread_stop_times`); the ODE path asks for it in a run with no
    sensitivity columns. Not the SSA's: its stops are a double before each
    change, and a step rate read there is the old one.
    """
    return _resolve_crossings(
        core,
        t_start,
        t_end,
        conditions,
        periodic=True,
        touches=touches,
        over_budget=over_budget,
        sensitivity_params=sensitivity_params,
        on_the_flip=on_the_flip,
        before_the_flip=before_the_flip,
        by_tread=by_tread,
    )


class StopsOverBudget(NamedTuple):
    """A condition that changes value more often in a run than stops are placed
    for (issue #890)."""

    condition: str
    # How many changes were counted before the count was given up.
    changes: int
    # The shortest time between two consecutive changes of this condition,
    # where it is known. A step no longer than half of it cannot span a
    # stretch between two of them. That is the whole of what keeps a step off
    # the rate law's windows only where nothing else in the run switches: a
    # stretch that ends at another condition's stop can be shorter than any of
    # this condition's own.
    min_gap: float | None
    # The shortest time between two consecutive changes of any condition,
    # this one's and every one that was stopped at, where this one's changes
    # were listed (it did not fit beside the others'). ``None`` where they were
    # only counted.
    union_gap: float | None = None
    # The changes could not be counted: each finer grid the condition was
    # evaluated on found more of them (:class:`bngsim._clock_scan.Unresolved`).
    unresolved: bool = False
    # The call, and the stretch of the run, where a step call's argument is
    # within its own rounding of a level it crosses: so that the call reads on
    # either side of it by turns and where it changes is not a time, or (the
    # last entry true) for more of the run than the call can be read over
    # (:class:`bngsim._clock_scan.Flickers`). Such a condition is `unresolved`.
    flicker: tuple[str, float, float, bool] | None = None
    # A leg of the run that would fit, where the condition was read tread by
    # tread (issue #1018) and is refused by name past the budget: without a
    # stop on each change a step bound need not keep it within the tolerance.
    legs: float | None = None


class _OverStopBudget(Exception):
    """Raised by a resolver whose condition is past :data:`_RUN_STOP_BUDGET`."""

    def __init__(
        self,
        count: float,
        min_gap: float | None = None,
        unresolved: bool = False,
        flicker: tuple[str, float, float, bool] | None = None,
        legs: float | None = None,
    ):
        super().__init__(count)
        self.count = int(min(count, 2**62))
        self.min_gap = min_gap
        self.unresolved = unresolved
        self.flicker = flicker
        self.legs = legs


class StopsMoveWithAnEvent(Exception):
    """A rate law's condition reads a species only events write, at times or
    to values not known before the run, and it changes more than once in it
    (issue #1244): no stop can be placed for it, and a step can lie across a
    whole window of it."""


# Times or values an event's schedule is read from: ``time() >= c`` and its
# spellings, with c the run's constant.
_TIME_TRIGGER = re.compile(r"^time\(\)\s*(>=|>)\s*(.+)$|^(.+?)\s*(<=|<)\s*time\(\)$")
# Stretches a schedule is placed over, at most; past it the run is refused as
# one whose schedule is not known.
_HELD_STRETCHES_MOST = 512


def _event_held_species(core, conditions=None) -> dict[str, int]:
    """The species events assign and nothing else moves: no reaction, rate rule
    or rescale reads them as a reactant or a product. Between two of those
    events each holds its value. Name to 0-based index; of those *conditions*
    read, where they are given."""
    assigned = set(core.event_assigned_species)
    if not assigned:
        return {}
    names = list(core.species_names)
    if conditions is not None:
        read = {m.group(0) for cond in conditions for m in _IDENTIFIER.finditer(cond)}
        assigned = {i for i in assigned if names[i] in read}
        if not assigned:
            return {}  # the model as a dict is not built for a run that asks nothing of it
    data = core.codegen_data()
    for rxn in data["reactions"]:
        assigned -= set(rxn["reactants"]) | set(rxn["products"])
    return {names[i]: i for i in sorted(assigned)}


def _reads_run_constants_alone(text: str, scope: SwitchConditionScope) -> bool:
    """True when *text* reads no clock, no state and no model function: only
    numbers, builtins and run constants, read directly or through a derived
    parameter, which is a constant of the run too (`w = k2`, k2 = 2*w2)."""
    text = _inline_derived_param_refs(text, scope.derived_exprs) or text
    if not _clock_free(text, scope.clock_symbols):
        return False
    for m in _IDENTIFIER.finditer(text):
        name = m.group(0)
        if name in scope.function_names:
            return False
        if text[m.end() :].lstrip().startswith("("):
            continue  # an engine builtin; its arguments are read by this loop
        if name in scope.run_constants or name in _BUILTIN_CONSTANT_VALUES:
            continue
        return False
    return True


def _with_values(text: str, values: Mapping[str, float]) -> str:
    """*text* with each name in *values* written as its number."""
    return _IDENTIFIER.sub(
        lambda m: f"({values[m.group(0)]!r})" if m.group(0) in values else m.group(0), text
    )


def _held_schedule(
    core,
    scope: SwitchConditionScope,
    held: Mapping[str, int],
    t_start: float,
    t_end: float,
    sensitivity_params,
) -> tuple[list[tuple[float, dict[str, float]]], dict[str, str], dict[str, set[float]]]:
    """When each held species changes in ``(t_start, t_end)``, and to what, as
    ``[(t_start, values), (t1, values from t1), ...]``; for each held species
    whose changes are not known before the run, why; and for each, the numbers
    an event is known to set it to, whenever that event fires.

    Known: every event that writes it fires on ``time() >= c`` (or ``>``, or
    the same written the other way), with c a run constant, after a constant
    delay, and sets a value that reads only run constants and held species
    known then. Such a trigger rises once, so the event fires once. A run that
    starts where the last one ended continues its events (``event_carry``, issue
    #693): a trigger rises in it only where it was false there, and the
    executions the last run delayed into it are taken with the values they
    froze. A fresh run's trigger was false before it unless its initialValue
    says true. Not known: two writers executing at one instant, which are not ordered here;
    a delayed fire whose value, read at its trigger, reads a held species that
    another fire sets in between; and, in a run with sensitivity columns, a
    time or a value that reads a requested parameter, or one derived from
    them, whose stops would move with it."""
    names_of = {i: name for name, i in held.items()}
    unknown: dict[str, str] = {}
    requested = set(sensitivity_params or ())

    def moved(text: str) -> list[str]:
        """The requested parameters *text* reads, directly or through a
        derived parameter, as :func:`_resolve_flat` reads a condition."""
        flat = _inline_derived_param_refs(text, scope.derived_exprs) or text
        return sorted(set(_IDENTIFIER.findall(flat)) & requested)

    param_names = list(core.param_names)
    if sensitivity_params:
        for sp, p in core.species_ic_param_refs:
            if sp in names_of and moved(param_names[p]):
                unknown[names_of[sp]] = (
                    f"its initial value is {param_names[p]}, which a requested "
                    "sensitivity parameter moves"
                )

    def value_of(text: str, at: float) -> tuple[float | None, str]:
        """The number *text* reads, or None and why not."""
        if not _reads_run_constants_alone(text, scope):
            return None, "which is not known before the run"
        hit = moved(text)
        if hit:
            return None, f"which a requested sensitivity parameter moves ({', '.join(hit)})"
        v = float(core.evaluate_over_times(text, [float(at)], 0.0)[0])
        return (v, "") if math.isfinite(v) else (None, f"which reads {v}")

    start = {name: float(core.get_concentration(name)) for name in held}
    # What each event is known to set a held species to, read with the others
    # as the run starts, whenever it fires: a condition whose stops cannot be
    # placed is read with each of these as well as with its start.
    sets: dict[str, set[float]] = {}

    def may_set(writes) -> None:
        for name, text in writes:
            text = _with_values(text, start)
            if _reads_run_constants_alone(text, scope):
                v = float(core.evaluate_over_times(text, [float(t_start)], 0.0)[0])
                if math.isfinite(v):
                    sets.setdefault(name, set()).add(v)

    # The last run's events, where this one continues it: what each trigger read
    # where it ended, and its executions not yet applied.
    carry = core.event_carry()
    continued = carry is not None and float(carry[0]) == float(t_start)
    was_true: list = list(carry[1]) if continued else []
    events = core.events_as_written()

    # (execution time, trigger time, label, [(species, value text)], from trigger)
    fires: list[tuple[float, float, str, list[tuple[str, str]], bool]] = []
    if continued:
        for i, at, frozen in carry[2]:
            ev = events[i]
            writes = [
                (names_of[sp], (repr(float(frozen[k])) if frozen else text))
                for k, (sp, text) in enumerate(ev["assignments"])
                if sp in names_of
            ]
            if not writes:
                continue
            may_set(writes)
            label = f"event {ev['id'] or '?'}, delayed from the last run"
            if frozen and len(frozen) != len(ev["assignments"]):
                for name, _ in writes:
                    unknown.setdefault(name, f"{label} carries values it cannot be read by")
                continue
            if float(at) < t_end:
                fires.append((float(at), t_start, label, writes, False))
    for i, ev in enumerate(events):
        writes = [(names_of[sp], text) for sp, text in ev["assignments"] if sp in names_of]
        if not writes:
            continue
        may_set(writes)
        trigger = _strip_redundant_parens(ev["trigger"])
        label = f"event {ev['id'] or '?'} on {trigger}"
        m = _TIME_TRIGGER.match(trigger.replace(" ", ""))
        threshold, why = None, "a trigger that is no fixed time"
        if m is not None and ev["delay_expr"]:
            why = f"after a delay of {ev['delay_expr']}, not a constant"
        elif m is not None:
            op, c = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            threshold, why = value_of(c, t_start)
            why = f"at time {c}, {why}"
        if threshold is None:
            for name, _ in writes:
                unknown.setdefault(name, f"{label} sets it, {why}")
            continue
        true_at_start = t_start >= threshold if op in (">=", "<=") else t_start > threshold
        true_before = bool(was_true[i]) if continued else bool(ev["initial_value"])
        if true_at_start and true_before:
            continue  # it never rises in this run
        rises = t_start if true_at_start else threshold
        at = rises + float(ev["delay"])
        if at < t_end:
            from_trigger = float(ev["delay"]) > 0 and ev["use_values_from_trigger_time"]
            fires.append((at, rises, label, writes, from_trigger))
    fires.sort(key=lambda f: f[0])
    for a, b in zip(fires, fires[1:], strict=False):
        if abs(b[0] - a[0]) <= 1e-12 * max(1.0, abs(a[0])):
            for name, _ in a[3] + b[3]:
                unknown.setdefault(name, f"{a[2]} and {b[2]} set it at one instant")
    if len(fires) > _HELD_STRETCHES_MOST:
        for f in fires:
            for name, _ in f[3]:
                unknown.setdefault(name, f"events set it {len(fires)} times in the run")

    values = dict(start)
    schedule = [(t_start, dict(values))]
    for at, rises, label, writes, from_trigger in fires:
        if from_trigger and any(rises < other[0] < at for other in fires):
            # Its value is read at its trigger, before a fire it would read.
            for name, _ in writes:
                unknown.setdefault(
                    name, f"{label} reads its value at its trigger, before another event sets it"
                )
            continue
        new = {}
        for name, text in writes:
            v, why = value_of(_with_values(text, values), at)
            if v is None:
                unknown.setdefault(name, f"{label} sets it to {text}, {why}")
            else:
                new[name] = v
        if new:
            values = {**values, **new}
            if at > t_start:
                schedule.append((at, dict(values)))
            else:
                schedule[0] = (t_start, dict(values))
            for name, v in new.items():
                sets.setdefault(name, set()).add(v)
    return schedule, unknown, sets


def _past_budget_in_both(a: StopsOverBudget, b: StopsOverBudget) -> StopsOverBudget:
    """One condition past the budget in two stretches of a run (issue #1244),
    as one: the changes of both, the shorter of each stretch between two, and
    what either could not count or read. A step bound made from the last
    stretch alone can span the windows of an earlier, narrower one."""

    def least(x, y):
        return y if x is None else x if y is None else min(x, y)

    return a._replace(
        changes=a.changes + b.changes,
        min_gap=least(a.min_gap, b.min_gap),
        union_gap=least(a.union_gap, b.union_gap),
        unresolved=a.unresolved or b.unresolved,
        flicker=a.flicker or b.flicker,
        legs=least(a.legs, b.legs),
    )


def _resolve_crossings(
    core,
    t_start: float,
    t_end: float,
    conditions,
    periodic: bool,
    touches=frozenset(),
    over_budget=None,
    sensitivity_params=None,
    on_the_flip=False,
    before_the_flip=False,
    by_tread=False,
) -> tuple[list[CrossingStop], list[CrossingStop], list[CrossingStop]]:
    """:func:`_resolve_flat`, with a condition that reads a species only events
    write resolved stretch by stretch between their executions (issue #1244).

    Such a species is no run constant, so a condition that reads it was given
    no stop at all, and a schedule of pulses whose width an event writes was
    stepped over before the event as well as after: X(240) came to 1 for 17.
    Between two of the events it holds, and where when they execute and what
    they set is known before the run (:func:`_held_schedule`), each stretch is
    resolved with the species written as its number there. Where it is not
    known, a condition a rate law reads that changes more than once over the
    run, as its start reads it or with the species at any number an event is
    known to set it to, is refused by name (:class:`StopsMoveWithAnEvent`): a
    step can lie across a whole window of it. One that changes once is a
    root's sign change, and an event's trigger is located as a root, so both
    are read as before; so is one that only an output reads.

    The run has one budget of stops (:data:`_RUN_STOP_BUDGET`), whatever its
    stretches, and each call reads the rate laws for every condition of the
    run as the model spells it."""
    kw = dict(
        periodic=periodic,
        over_budget=over_budget,
        sensitivity_params=sensitivity_params,
        on_the_flip=on_the_flip,
        before_the_flip=before_the_flip,
        by_tread=by_tread,
    )
    held = _event_held_species(core, [c for c in conditions if c not in touches])
    reads = {
        cond: sorted(set(_IDENTIFIER.findall(cond)) & held.keys())
        for cond in conditions
        if cond not in touches
    }
    reads = {cond: names for cond, names in reads.items() if names}
    if not reads:
        return _resolve_flat(core, t_start, t_end, conditions, touches=touches, **kw)
    ctx = core.functional_jacobian_context()
    scope = switch_condition_scope(core, ctx)
    schedule, unknown, sets = _held_schedule(core, scope, held, t_start, t_end, sensitivity_params)
    known = {cond: names for cond, names in reads.items() if not set(names) & unknown.keys()}
    # What is not known is read as before: given no stop, or refused below.
    rest = [cond for cond in conditions if cond not in known]
    unplaced = [cond for cond in reads if cond not in known]
    # Only where a rate law reads it: a step across a window of a condition
    # that only an output reads moves nothing (as past the budget,
    # :func:`conditions_a_rate_law_reads`).
    switching = conditions_a_rate_law_reads(core, unplaced) if unplaced else frozenset()
    for cond in unplaced:
        if cond not in switching:
            continue
        names = reads[cond]
        # As its start reads it, and with each held species it reads at each
        # number an event is known to set it to: more than one change over the
        # run, in any of these, and a step can lie across one of its windows,
        # as across a pulse. (Quiet at its start, a width w = 0 pulses once a
        # state trigger sets w = 1.)
        at_start = schedule[0][1]
        readings = [(None, at_start)] + [
            (name, {**at_start, name: v})
            for name in names
            for v in sorted(sets.get(name, ()))
            if v != at_start[name]
        ]
        for set_by, values in readings:
            quiet: list = []
            stops = [
                stop
                for part in _resolve_flat(
                    core,
                    t_start,
                    t_end,
                    [_with_values(cond, values)],
                    periodic=True,
                    over_budget=quiet,
                )[:2]
                for stop in part
            ]
            if len(stops) <= 1 and not quiet:
                continue
            name = next(n for n in names if n in unknown)
            how_often = f"{len(stops)} times" if not quiet else "more often than stops are placed"
            where = (
                "as the run starts"
                if set_by is None
                else f"with {set_by} at {values[set_by]!r}, which an event sets it to,"
            )
            raise StopsMoveWithAnEvent(
                f"A rate law switches at {_strip_redundant_parens(cond.strip())!r}, which "
                f"reads {name}, and only events set {name}: {unknown[name]}. So its stops "
                f"cannot be placed before the run, and {where} the condition "
                f"changes {how_often}: a step can lie across a whole window of it with "
                "nothing said (issue #1244). bngsim refuses rather than step over it. An "
                f"event that sets {name} at a fixed time (`time >= c`) to a constant, "
                "with no requested sensitivity parameter in either, has the condition's "
                "stops placed stretch by stretch."
            )
    # Each call is asked about some of the run's conditions, and reads the rate
    # laws for all of them, as the model spells them.
    written = {cond: cond for cond in conditions}
    found = list(_resolve_flat(core, t_start, t_end, rest, touches=touches, written=written, **kw))
    if known:
        # Past the budget in a stretch is reported as the condition the model
        # holds, which is what the run's switches are matched against, once.
        past: dict[str, StopsOverBudget] = {}
        # One budget for the run: each stretch's stops count against it with
        # every stop placed before them.
        placed = [stop.time for stop in (*found[0], *found[1])]
        ends = [at for at, _ in schedule[1:]] + [t_end]
        for (lo, values), hi in zip(schedule, ends, strict=True):
            if hi <= lo:
                continue
            asked = {_with_values(cond, values): cond for cond in known}
            here: list[StopsOverBudget] = []
            part = _resolve_flat(
                core,
                lo,
                hi,
                list(asked),
                touches=touches,
                written={**written, **asked},
                placed_before=placed,
                **{**kw, "over_budget": here},
            )
            for k in range(3):
                found[k] = [*found[k], *part[k]]
            placed.extend(stop.time for stop in (*part[0], *part[1]))
            for item in here:
                cond = asked.get(item.condition, item.condition)
                item = item._replace(condition=cond)
                past[cond] = item if cond not in past else _past_budget_in_both(past[cond], item)
        for item in past.values():
            if over_budget is None:
                _warn_over_budget(item.condition, item.changes, t_start, t_end)
            else:
                over_budget.append(item)
    fixed, stops, crossings = (sorted(f, key=lambda stop: stop.time) for f in found)
    return fixed, stops, crossings


def _resolve_flat(
    core,
    t_start: float,
    t_end: float,
    conditions,
    periodic: bool,
    touches=frozenset(),
    over_budget=None,
    sensitivity_params=None,
    on_the_flip=False,
    before_the_flip=False,
    by_tread=False,
    written=None,
    placed_before=(),
) -> tuple[list[CrossingStop], list[CrossingStop], list[CrossingStop]]:
    """The resolvers over *conditions*, for :func:`_resolve_crossings`.

    *written*, where it asks about some of the run's conditions, maps each
    condition asked here, and every other the run asks about, to the model's
    spelling of it: a stretch's is written with a held species as its number
    (issue #1244), and the rate laws are read for what the model spells.
    *placed_before* are the times of the stops the run has placed already,
    which count against its one budget."""
    if not conditions:
        return [], [], []
    ctx = core.functional_jacobian_context()
    scope = switch_condition_scope(core, ctx)
    bodies = _function_slot_bodies(ctx)
    spelled = dict(written) if written is not None else {}
    every = list(dict.fromkeys(spelled.values())) if written is not None else list(conditions)
    found: list[CrossingStop] = []
    stops: list[CrossingStop] = []
    crossings: list[CrossingStop] = []
    # The changes of each condition that did not fit, where they were listed.
    unplaced: dict[int, list[float]] = {}
    # Stops read tread by tread (issue #1018), placed after every other's.
    treaded: list[tuple[str, int, _TreadStops]] = []
    # The conditions whose value holds between two changes (a comparison, a
    # step call): the rest of a rate law's reads of the clock are where it
    # changes between its stops (issue #1226).
    held_conditions = (
        sorted((c for c in every if _held_between_changes(c)), key=len, reverse=True)
        if before_the_flip
        else []
    )

    reads_here: list[list[tuple[str, bool]] | None] = []

    def changes_between_its_stops(cond: str) -> bool:
        """Whether a rate law that reads *cond* reads the clock outside every
        condition that holds between its changes: a sawtooth (a `mod`, `frac`
        or `floor` remainder), a ramp, a pulse times the clock. Spanned by one
        step, such a stretch is read at one end, and a double before the change
        that end is not the right one more than the other: from X(0) = 1e4 at
        rtol 1e-3, `mod(time(), 0.7)` over 70 was 19.3 over a double before
        every change and 0.45 as the resolvers put them. So it keeps them. A
        law that holds between its changes is exact a double before. (Each
        rate law flattened, with what it calls written into it; past
        :data:`_HELD_READ_MOST`, not read, and taken to change.)"""
        cond = spelled.get(cond, cond)
        if not reads_here:
            try:
                reads_here.append(_rate_laws_read_held(core, tuple(held_conditions)))
            except Exception as exc:  # noqa: BLE001 - then it is taken to
                logger.debug("the rate laws: not read (%s)", exc)
                reads_here.append(None)
        reads = reads_here[0]
        if reads is None:
            return True
        return any(clock for flat, clock in reads if cond in flat)

    flats: list[list[str]] = []

    def read_by_a_rate_law(cond: str) -> bool:
        """Whether a rate law reads *cond*, and none that does reads the clock
        in a far line outside every condition (:func:`_a_far_line_outside`).
        Asked only of a condition about to be read tread by tread. A function
        that only reports is no reason for a stop: a stop on each tread of an
        output of `100*mod(1e12 + time(), P)` took 1.6 million steps where the
        run took 10."""
        cond = spelled.get(cond, cond)
        if not flats:
            try:
                flats.append(_rate_law_flats(core))
            except Exception as exc:  # noqa: BLE001 - then the reading is not made
                logger.debug("the rate laws: not read (%s)", exc)
                flats.append([])
        mine = [flat for flat in flats[0] if cond in flat]
        return bool(mine) and not any(
            _a_far_line_outside(core, flat, every, scope, t_start, t_end) for flat in mine
        )

    whole_triggers: set[str] | None = None

    def over(cond, count, min_gap, listed=None, unresolved=False, flicker=None, legs=None) -> None:
        if over_budget is None:
            _warn_over_budget(cond, count, t_start, t_end)
        else:
            if listed is not None:
                unplaced[len(over_budget)] = listed
            over_budget.append(
                StopsOverBudget(
                    cond,
                    int(count),
                    min_gap,
                    unresolved=unresolved,
                    flicker=flicker,
                    legs=legs,
                )
            )

    def requested_in(asked: _EngineCondition) -> list[str]:
        """The requested sensitivity parameters the condition reads."""
        flat = _inline_derived_param_refs(asked.inlined, scope.derived_exprs) or asked.inlined
        return sorted(
            {m.group(0) for m in _IDENTIFIER.finditer(flat)} & set(sensitivity_params or ())
        )

    def asks_evaluator(engine, clock_idx: int, trigger: bool = False) -> bool:
        """Whether the evaluator's reading of this condition places stops.
        *engine* gives the condition as the evaluator is asked about it."""
        if not periodic:
            return False
        asked = engine()
        if asked is None:
            return False
        if sensitivity_params is None:
            return True
        if asked.kind == "bend":
            # The rate law is continuous across a bend, so there is no jump
            # to make whatever moves it: the stop is all the columns need.
            return True
        if asked.kind == "value":
            # A step call outside every condition is no root of the run's: its
            # stops are the whole of how it is read, as they are for one a
            # resolver lists (issue #944), on a counter as on the clock. One a
            # requested parameter moves was refused above.
            return True
        # An event's trigger is located as a root, where the event's jump in
        # the columns is made; a stop on its rise would hand the run a fire
        # with no located crossing (issue #910).
        return clock_idx < 0 and not trigger and not requested_in(asked)

    for cond in conditions:
        rewrite = _rewrite_counter_clock(core, cond, scope, t_start)
        if rewrite is None:
            continue
        text, clock_idx, offset = rewrite
        # The condition as the evaluator is asked about it, made when something
        # asks: a condition an exact resolver places, in a run that has made
        # this pass before, never does.
        made: list[_EngineCondition | None] = []

        def engine(cond=cond, rewrite=rewrite, made=made) -> _EngineCondition | None:
            if not made:
                made.append(_engine_condition(core, cond, rewrite, scope, bodies))
            return made[0]

        asked = engine() if sensitivity_params else None
        if asked is not None and asked.kind in ("value", "value?"):
            moved = requested_in(asked)
            if moved:
                # The sensitivity detector reads a step call outside every
                # condition as a jump no parameter moves (issue #944). One
                # that a requested parameter does move has a jump in the
                # column at each step, which nothing computes, and CVODES'
                # difference quotient across the step ground the run to a
                # halt instead: minutes at t = 23.9999 for a pulse of floors.
                raise SensitivityUnsupportedError(
                    "Forward sensitivity is not supported for this model: the rate law "
                    f"steps at {_strip_redundant_parens(cond.strip())!r}, a step call "
                    "outside any condition, and the times it steps at move with the "
                    f"requested parameter{'s' if len(moved) > 1 else ''} "
                    f"{', '.join(moved)}. The jump each step makes in those columns is "
                    "not computed. Drop "
                    f"{'them' if len(moved) > 1 else 'it'} from sensitivity_params, or "
                    "difference plain runs."
                )
        # (The search first: it answers for every ordering at a tenth of the
        # cost of reading the atom's own operator.)
        equality = _EQUALITY_OP.search(text) is not None and is_equality_atom(
            _strip_redundant_parens(text.strip())
        )
        # An equality of two things that move smoothly holds at an instant. Of
        # two that hold their values (`floor(time()/0.5) == 240`) it holds for
        # a stretch, and is a window like any other.
        instant = equality and not (_STEP_CALL.search(text) or _BEND_CALL.search(text))
        if equality:
            # Measure zero to a rate law (see :func:`_switches_on_clock_alone`);
            # the one double an event's trigger is true at.
            # In a run with sensitivity columns too, where the instant reads
            # no requested parameter: no root ever located it.
            #
            # An equality over a call that holds its value is asked the same
            # way before it is read as a window: `abs(time - 5) == 0.3` holds
            # at two instants or, where no double meets it, at none, and that
            # trigger is refused like any other that cannot fire.
            asked = engine() if cond in touches and clock_idx < 0 and periodic else None
            if asked is not None and not (sensitivity_params is not None and requested_in(asked)):
                if whole_triggers is None:
                    whole_triggers = {
                        _strip_redundant_parens(src).replace(" ", "")
                        for src in core.event_trigger_sources()
                    }
                alone = _strip_redundant_parens(cond).replace(" ", "") in whole_triggers
                for at in _equality_stops(
                    core, cond, text, asked, scope, bodies, t_start, t_end, alone
                ):
                    stops.append(CrossingStop(at, clock_idx, 0.0, True))
                    crossings.append(CrossingStop(at, clock_idx, 0.0))
            if instant:
                continue
        times: list[float] | None = None
        pairs: list[tuple[float, float]] = []
        # Tread by tread (issue #1018): a plain ODE run, on literal time, and not
        # for an event's trigger. (On a counter the integrated clock lands a few
        # doubles past a stop's value, and a stop a double before a change is
        # not a stop before it.)
        treads = by_tread and sensitivity_params is None and clock_idx < 0 and cond not in touches
        # (time, the counter's value there, on a touch) for a scanned condition.
        scanned: list[tuple[float, float, bool]] | None = None
        try:
            try:
                t_star = (
                    None
                    if equality
                    else _crossing_time_of_condition(text, scope, t_start, t_end, bodies)
                )
                if t_star is not None:
                    times = [t_star]
                else:
                    times = _schedule_stop_times(text, scope, t_start, t_end)
                    if times is None and not equality:
                        times = _step_edge_stop_times(text, scope, t_start, t_end)
                if times is not None and t_star is not None and _reads_time_once(text, scope):
                    # The linear resolver's answer for a comparison that reads
                    # the clock once, through arithmetic alone, is the whole of
                    # what that comparison does: there is nothing to check.
                    pass
                elif times is not None:
                    # What a resolver read is not always what the evaluator
                    # computes: where the two disagree about a change, the
                    # evaluator is the one the rate law follows. A plain run
                    # that has asked before is answered from the memo, without
                    # the condition being made for the evaluator at all.
                    asks = (
                        periodic
                        if sensitivity_params is None
                        else asks_evaluator(engine, clock_idx, cond in touches)
                    )
                    if asks and not _agrees_with_evaluator(
                        core, engine, text, scope, bodies, t_start, t_end, times
                    ):
                        logger.debug("stops for %r disagree with the evaluator; scanning", cond)
                        times = None
                        scanned = _engine_scan_stop_times(
                            core,
                            cond,
                            engine(),
                            offset,
                            scope,
                            t_start,
                            t_end,
                            cond in touches,
                            by_tread=treads,
                            reads=read_by_a_rate_law,
                        )
                    elif not times and treads and _a_whole_sawtooth_call(text):
                        # A sawtooth of a line far from zero, which the step
                        # resolver reads as within its rounding of zero on both
                        # sides of every wrap: `100*mod(1.1e12 + time(),
                        # 0.0048828125)` had no stop and was 21% over by the
                        # sixth of six legs of 0.2. Read tread by tread, where
                        # it can be (issue #1018).
                        scanned = _left_alone_tread_stops(
                            core,
                            cond,
                            engine(),
                            scope,
                            t_start,
                            t_end,
                            offset,
                            reads=read_by_a_rate_law,
                        )
                elif periodic:
                    shaped: list[bool] = []
                    if not equality:
                        pairs = _periodic_stop_times(
                            text,
                            scope,
                            t_start,
                            t_end,
                            bodies,
                            touches=cond in touches,
                            shaped=shaped,
                        )
                    if not shaped and asks_evaluator(engine, clock_idx, cond in touches):
                        scanned = _engine_scan_stop_times(
                            core,
                            cond,
                            engine(),
                            offset,
                            scope,
                            t_start,
                            t_end,
                            cond in touches,
                            by_tread=treads,
                            reads=read_by_a_rate_law,
                        )
                if times is None and not pairs and scanned is None and not equality:
                    times = _inner_step_stop_times(text, scope, t_start, t_end)
            except _OverStopBudget as exc:
                # Many jumps are not many changes: `floor(time()/0.0001) < 5000`
                # holds 2.4 million of the first and one of the second. (Not
                # where the changes were found and could not be counted: the
                # sampling that reads the first kind sees nothing of a pulse
                # between two of its points, and `if(sign(q) > 0, 1, 0)` with
                # a q that flickers came back as never changing.)
                few = (
                    _few_changes(core, engine(), offset, scope, t_start, t_end)
                    if not exc.unresolved and asks_evaluator(engine, clock_idx, cond in touches)
                    else None
                )
                if few is None:
                    raise
                times, pairs, scanned = None, [], few
        except _OverStopBudget as exc:
            over(
                cond,
                exc.count,
                exc.min_gap,
                unresolved=exc.unresolved,
                flicker=exc.flicker,
                legs=exc.legs,
            )
            continue
        if isinstance(scanned, _TreadStops):
            if scanned.past is not None:
                _say_left_alone(cond, *scanned.past)
                scanned = None
            else:
                # Taken once every other condition has its stops: before, a law
                # main leaves without any took the budget a schedule beside it
                # fits in, and the run was refused.
                treaded.append((cond, clock_idx, scanned))
                continue
        inside = [t for t in times or () if t_start < t <= t_end]
        # One budget for the run, not one per condition: each stop costs a
        # restart, and a few conditions near it took a gigabyte.
        mine = len(inside) + len(pairs) + len(scanned or ())
        if len(placed_before) + len(found) + len(stops) + mine > _RUN_STOP_BUDGET:
            changes = sorted(
                [*inside, *(ts for _, ts in pairs), *(t for t, _, _ in scanned or ())]
            )
            legs = (
                _leg_that_fits(
                    [*placed_before, *(s.time for s in found), *(s.time for s in stops), *changes]
                )
                if isinstance(scanned, _TreadStops)
                else None
            )
            over(cond, mine, _shortest_stretch(changes), changes, legs=legs)
            continue
        for t_cross, t_stop in pairs:
            crossings.append(
                CrossingStop(t_cross, clock_idx, t_cross + offset if clock_idx >= 0 else 0.0)
            )
            stops.append(
                CrossingStop(
                    t_stop,
                    clock_idx,
                    t_stop + offset if clock_idx >= 0 else 0.0,
                    touch=t_stop == t_cross,
                )
            )
        # (The scan's stops are on the evaluator's change. A double before it,
        # where the run asks for that and no event's trigger reads it: issue
        # #1226, above.)
        before = (
            before_the_flip
            and clock_idx < 0
            and cond not in touches
            and bool(scanned or inside)
            and not changes_between_its_stops(cond)
        )
        for t_cross, value, touch in scanned or ():
            if before and not touch and t_start < math.nextafter(t_cross, -math.inf):
                t_cross = math.nextafter(t_cross, -math.inf)
            crossings.append(CrossingStop(t_cross, clock_idx, value if clock_idx >= 0 else 0.0))
            stops.append(
                CrossingStop(t_cross, clock_idx, value if clock_idx >= 0 else 0.0, touch=touch)
            )
        # Each onto the double the evaluator changes on (issue #891), in a run
        # that reads no condition for itself: one with sensitivity columns has
        # its switch times from the resolvers' own numbers, and the two must
        # stay one. Not for a condition an event's trigger reads: a trigger on
        # `time > 1` stopped on its first true double, 1.0000000000000002, and
        # a delay of 1.5 added to that rounds to 2.5, so the execution came
        # before a row at 2.5 that it is after (issue #947 is where events on
        # one instant are ordered).
        on_flip = (
            _stops_on_the_flip(core, engine, text, offset, scope, bodies, t_start, t_end, inside)
            if inside
            and periodic
            and (on_the_flip or before)
            and sensitivity_params is None
            and cond not in touches
            else None
        )
        if on_flip is not None:
            as_placed: list[float | None] = [*inside, *(None for _ in on_flip[1])]
            for t_placed, (t_cross, value) in zip(
                as_placed, [*on_flip[0], *on_flip[1]], strict=True
            ):
                if t_cross > t_end and t_placed is not None:
                    # The change is a double past the run's last instant, as
                    # `time <= 0.3` is for a leg that ends on 0.3. The stop
                    # stays on the end: a last step that is not stopped is
                    # taken past the end and read back, across the change.
                    t_cross, value = t_placed, t_placed + offset
                elif before:
                    # The last double before the change (issue #1226).
                    t_cross = math.nextafter(t_cross, -math.inf)
                    value = t_cross + offset
                if t_start < t_cross <= t_end:
                    found.append(
                        CrossingStop(
                            t_cross,
                            clock_idx,
                            value if clock_idx >= 0 else 0.0,
                            on_flip=True,
                            placed=t_placed,
                        )
                    )
            continue
        for t_cross in inside:
            # The counter's exact value at the crossing. Reading it back off the
            # rewrite rather than off the recognized threshold text is what keeps
            # the two in step whichever resolver placed the stop, and for a
            # schedule there is no single threshold text to read.
            found.append(
                CrossingStop(t_cross, clock_idx, t_cross + offset if clock_idx >= 0 else 0.0)
            )
    if len(treaded) > 1 or (treaded and over_budget):
        # Main's answer for each, where another law of the run is read so too
        # or is past the budget. Two lines whose treads end on the same
        # instants (`1e12 + time()`, `1e12 + 3*time()`) put two stops a double
        # or two apart, which a run merges onto the later, on the earlier's
        # change, and a third of the changes had no stop before them (+977
        # local tolerances where main refuses). Beside a condition past the
        # budget, these stops cut the stretch its step bound is made from, and
        # a run main makes was refused for a bound of 3e8 steps.
        for cond, _clock_idx, scanned in treaded:
            if scanned.main is not None:
                over(
                    cond,
                    scanned.main.count,
                    scanned.main.min_gap,
                    unresolved=scanned.main.unresolved,
                    flicker=scanned.main.flicker,
                )
        treaded = []
    for cond, clock_idx, scanned in treaded:
        others = len(placed_before) + len(found) + len(stops)
        if others + len(scanned) > _RUN_STOP_BUDGET:
            fits = _leg_that_fits(
                [
                    *placed_before,
                    *(s.time for s in found),
                    *(s.time for s in stops),
                    *(t for t, _, _ in scanned),
                ]
            )
            if scanned.main is not None:
                over(cond, len(scanned), _shortest_stretch([t for t, _, _ in scanned]), legs=fits)
            else:
                _say_left_alone(cond, len(scanned), t_start, t_end, fits=fits, others=others)
            continue
        for t_cross, value, touch in scanned:
            crossings.append(CrossingStop(t_cross, clock_idx, value if clock_idx >= 0 else 0.0))
            stops.append(
                CrossingStop(t_cross, clock_idx, value if clock_idx >= 0 else 0.0, touch=touch)
            )
        if scanned.coarse is not None:
            _say_beyond_2_48(cond, scanned.coarse)
    # The sort is stable, so among equal times the conditions keep their order.
    found.sort(key=lambda stop: stop.time)
    stops.sort(key=lambda stop: stop.time)
    crossings.sort(key=lambda stop: stop.time)
    if unplaced and over_budget is not None:
        # A window of the rate law can open on one condition and close on
        # another, so what a step must not span is a stretch between two
        # changes of any of them.
        placed = [*placed_before, *(stop.time for stop in found), *(stop.time for stop in stops)]
        shortest = _shortest_stretch(
            sorted([*placed, *(t for times in unplaced.values() for t in times)])
        )
        for i in unplaced:
            over_budget[i] = over_budget[i]._replace(union_gap=shortest)
    return found, stops, crossings


# For a condition's text: whether it reads ``time()`` once and calls nothing
# else, and the names it reads. Asked of every condition on every run, so kept.
_READS_TIME_ONCE: dict[str, tuple[bool, tuple[str, ...]]] = {}
_ANY_CALL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(")


def _reads_time_once(text: str, scope: SwitchConditionScope) -> bool:
    """True for a comparison whose text reads ``time()`` once, holds no other
    call, no power and no modulo, and reads nothing else but run constants:
    ``time() >= t_on + delay``.

    Such a residual is a ratio of two expressions of first degree in time at
    most, which is monotone on each side of its pole. Where the linear
    resolver's three probes accept it, its one crossing is the comparison's
    only change, so the evaluator has nothing to add (and is not asked: this
    is the path nearly every model with a clock condition takes, on every
    run). A step call, a sinusoid or a product of two reads of time can pass
    those probes at one of several changes, and is checked.

    Run constants only, because a name can hide a second read of the clock:
    an SBML assignment rule ``g := (time - 100)*(time - 100.5)`` makes
    ``(time - 256)*g > 0`` read time once by its text and three times by what
    it computes.
    """
    known = _READS_TIME_ONCE.get(text)
    if known is None:
        calls = [m.group(0).rstrip("( \t") for m in _ANY_CALL.finditer(text)]
        plain = "^" not in text and "%" not in text and calls == ["time"]
        names = tuple(
            dict.fromkeys(
                m.group(0)
                for m in _IDENTIFIER.finditer(text)
                if m.group(0) != "time" and m.group(0) not in _BUILTIN_CONSTANT_VALUES
            )
        )
        if len(_READS_TIME_ONCE) >= 4096:
            _READS_TIME_ONCE.clear()
        known = _READS_TIME_ONCE[text] = (plain, names)
    plain, names = known
    return plain and all(name in scope.run_constants for name in names)


def conditions_a_rate_law_reads(core, conditions: Sequence[str]) -> frozenset[str]:
    """Those of *conditions* that stand in a rate law or in a function a rate
    law reaches.

    :func:`time_discontinuity_conditions` reads every function of the model,
    and one that no rate law reaches changes an output and not the trajectory.
    A stop on its conditions is a restart for nothing, which is cheap. A
    refusal over them is not: a run whose only fault was an output function
    with more changes than a run has stops for (``mod(time(), 0.0001)``,
    plotted) was refused as if the rate law switched on it.
    """
    if not conditions:
        return frozenset()
    flats = _rate_law_flats(core)
    return frozenset(cond for cond in conditions if any(cond in flat for flat in flats))


# What :func:`_rate_law_flat_parts` made, for each model it was made for (a
# model's rate laws and functions are its core's for its life), and by the
# texts it was made from, for a clone (a row of `run_batch` has a core of its
# own): a loop of legs on one model asks on every leg, and the inlining was
# half of such a leg's time.
_FLATS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_FLATS_BY_TEXT: dict[tuple, tuple[list[str], int, float]] = {}
_FLATS_BY_TEXT_MOST = 32
# The longest a rate law may flatten to and be read for whether it holds
# between its changes (issue #1226). Past it the inliner writes each text
# through sympy: a chain of 600 functions took 27 s to read where the run took
# 0.2, and main reads no rate law there. (The inliner's own limit.)
_HELD_READ_MOST = 20_000
# What :func:`_rate_laws_read_held` made, for each model, by the conditions
# written out.
_HELD_READS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _flattened_length(rate_laws, func_map) -> float:
    """The most any of *rate_laws* would be with every function it calls
    written out, from the lengths of the bodies, without writing one out."""
    from bngsim._jacobian import _IDENT_RE

    sizes: dict[str, float] = {}

    def size(name: str) -> float:
        if name not in sizes:
            sizes[name] = math.inf  # a cycle reads as too long
            sizes[name] = len(func_map[name]) + sum(
                size(ref) + 2 - len(ref)
                for ref in _IDENT_RE.findall(func_map[name])
                if ref in func_map
            )
        return sizes[name]

    try:
        return max(
            (
                len(text)
                + sum(
                    size(ref) + 2 - len(ref) for ref in _IDENT_RE.findall(text) if ref in func_map
                )
                for text in rate_laws
            ),
            default=0.0,
        )
    except RecursionError:
        return math.inf


def _rate_law_flat_parts(core) -> tuple[list[str], int, float]:
    """``(flats, n, longest)``: each rate law and then each function a rate law
    reaches, with the functions it calls written out (the first *n* are the
    rate laws'), and the most a rate law flattens to. The flats are made only
    where that is within :data:`_HELD_READ_MOST` or a caller asks for them
    whatever it is (:func:`_rate_law_flats`)."""
    with contextlib.suppress(TypeError):
        made = _FLATS.get(core)
        if made is not None:
            return made
    ctx = core.functional_jacobian_context()
    func_map = _scan_function_map(ctx)
    rate_laws = dict.fromkeys(_scan_rate_law(r) for r in ctx["functional_reactions"])
    key = (tuple(rate_laws), tuple(sorted(func_map.items())))
    made = _FLATS_BY_TEXT.get(key)
    if made is None:
        made = ([], len(rate_laws), _flattened_length(rate_laws, func_map))
        if len(_FLATS_BY_TEXT) >= _FLATS_BY_TEXT_MOST:
            _FLATS_BY_TEXT.clear()
        _FLATS_BY_TEXT[key] = made
    with contextlib.suppress(TypeError):
        _FLATS[core] = made
    return made


def _write_flats(core, made: tuple[list[str], int, float]) -> list[str]:
    """Fill *made*'s flats in place, once."""
    from bngsim._jacobian import _IDENT_RE, _inline_functions

    flats = made[0]
    if flats:
        return flats
    ctx = core.functional_jacobian_context()
    func_map = _scan_function_map(ctx)
    rate_laws = dict.fromkeys(_scan_rate_law(r) for r in ctx["functional_reactions"])
    reached: dict[str, None] = {}
    pending = [n for text in rate_laws for n in _IDENT_RE.findall(text) if n in func_map]
    while pending:
        name = pending.pop()
        if name not in reached:
            reached[name] = None
            pending.extend(n for n in _IDENT_RE.findall(func_map[name]) if n in func_map)
    memo: dict[str, str] = {}
    flats.extend(
        _inline_functions(text, func_map, memo=memo) or text
        for text in (*rate_laws, *(func_map[name] for name in reached))
    )
    return flats


def _rate_law_flats(core) -> list[str]:
    """Each rate law, and each function a rate law reaches, with the functions
    it calls written out."""
    return _write_flats(core, _rate_law_flat_parts(core))


def _rate_laws_read_held(core, held: tuple[str, ...]) -> list[tuple[str, bool]] | None:
    """For each rate law flattened, ``(flat, reads the clock)`` with every one
    of *held* written out of it; ``None`` where a rate law flattens past
    :data:`_HELD_READ_MOST`, and is not read (issue #1226). Made once a model
    and conditions: written out once a condition, a regimen of 600 windows in
    one rate law took 21.8 s for main's 1.5."""
    from bngsim import _clock_scan

    made = _rate_law_flat_parts(core)
    if made[2] > _HELD_READ_MOST:
        return None
    try:
        reads = _HELD_READS.setdefault(core, {})
    except TypeError:
        reads = {}
    if held not in reads:
        flats = _write_flats(core, made)[: made[1]]
        out = []
        for flat in flats:
            rest = flat
            for text in held:
                rest = rest.replace(text, "0")
            out.append((flat, bool(_clock_scan._READS_CLOCK.search(rest))))
        if len(reads) >= 4:
            reads.clear()
        reads[held] = out
    return reads[held]


def _a_far_line_outside(core, flat: str, conditions, scope, t_start, t_end) -> bool:
    """Whether *flat* reads the clock, outside every one of *conditions*,
    inside a line in time far enough from zero to hold each of its doubles
    for longer than an instant of the clock from *t_start* to *t_end*.

    Such a rate law is a staircase there too, and a stop on each change of
    the condition is not a stop on each of its steps: SBML's `rem(1.1e12 +
    time, P)` loads as `x - P*floor(x/P)`, and with a stop on each change of
    the `floor` it was 9e-5 off at rtol 1e-8, where main refused it. (A
    condition has its own stops, and a read of the clock near zero, as in
    `exp(-time())`, is the integrator's to follow.)"""
    import numpy as np

    from bngsim import _clock_scan

    rest = flat
    for cond in sorted(conditions, key=len, reverse=True):
        rest = rest.replace(cond, "0")
    ends = np.array([float(t_start), float(t_end)])
    instant = _clock_scan.TOUCH_ULPS * float(np.spacing(np.max(np.abs(ends))))
    for clock in _clock_scan._READS_CLOCK.finditer(rest):
        parts = _clock_scan.lines_holding_the_clock(rest, scope.run_constants, at=clock.start())
        if not parts:
            continue
        try:
            x = np.asarray(core.evaluate_over_times(f"({parts[-1]})", ends), dtype=float)
        except Exception:  # noqa: BLE001 - not read: then it may be one
            return True
        if not np.all(np.isfinite(x)):
            return True
        moved = abs(float(x[1] - x[0]))
        if (
            moved > 0.0
            and float(np.spacing(np.max(np.abs(x)))) * (ends[1] - ends[0]) / moved > instant
        ):
            return True
    return False


def _shortest_stretch(times: Sequence[float]) -> float | None:
    """The shortest time between two consecutive instants of *times* (sorted),
    two within a few doubles of each other being one instant."""
    import numpy as np

    at = np.asarray(times, dtype=float)
    if at.size < 2:
        return None
    gap = np.diff(at)
    apart = gap > _INSTANT_ULPS * np.spacing(np.abs(at[:-1]))
    return float(gap[apart].min()) if apart.any() else None


def _engine_text(text: str) -> str:
    """*text* as the evaluator compiles it: the clock as its nullary call."""
    return _TIME_REF.sub("time()", text)


def _evaluate_over(core, text: str, times, shift: float = 0.0):
    import numpy as np

    return core.evaluate_over_times(text, np.ascontiguousarray(times, dtype=float), shift)


# The evaluator's variable for a counter's offset from the clock, set at each
# call of ``evaluate_over_times``. A condition that reads a counter and literal
# time together is written over it, so its text is one for every run.
_SHIFT = "bngsim_scan_shift_"
_TABLE_CALL = re.compile(r"(?<![A-Za-z0-9_])tfun_([A-Za-z0-9_]+)\s*\(\s*\)")


class _EngineCondition(NamedTuple):
    """A condition as the model's evaluator can be asked about it at a time."""

    text: str  # what the evaluator compiles, the clock as `time()`
    inlined: str  # the same before the clock is respelled, for the memo key
    kind: str  # "comparison", "truth", "value" or "bend" (see _clock_scan)
    sides: tuple[str, str] | None  # a comparison's two sides
    # What is added to a time to get the value the text is evaluated at: the
    # counter's offset for a condition on a counter alone, which is read with
    # the counter written as the clock, and 0 otherwise.
    shift: float
    # The offset of a counter a condition reads beside literal time, which the
    # evaluator holds in a variable (:data:`_SHIFT`); 0 otherwise.
    offset: float
    knots: tuple[float, ...]  # breakpoints of the tables read, in those values
    tables: tuple  # their contents, for the memo key

    def evaluate(self, core, text: str, times):
        return _evaluate_over(core, text, times, self.offset)


def _engine_condition(
    core, cond: str, rewrite, scope: SwitchConditionScope, bodies
) -> _EngineCondition | None:
    """*cond* as the evaluator can be asked about it, or ``None`` where it reads
    anything but the clock, run constants and builtins: then its value at a
    time is not a function of the time alone.

    A condition on a counter is read with the counter written as the clock,
    and its times carried back by the counter's offset: no offset is written
    into the text, so one text serves every run. One that reads a counter and
    literal time together has the counter written as the clock plus a variable
    the evaluator is handed the offset in.

    A table function read on the clock is a function of the time like any
    builtin. One indexed by anything else is not.
    """
    from bngsim import _clock_scan

    _, clock_idx, offset = rewrite
    text = cond
    shift = held = 0.0
    if clock_idx >= 0:
        alone = _clock_free(cond, _TIME_SYMBOLS)
        for sym in sorted(scope.clocks, key=len, reverse=True):
            text = _clock_symbol_sub(text, sym, "time()" if alone else f"(time()+{_SHIFT})")
        if alone:
            shift = offset
        else:
            held = offset
    atom = _strip_redundant_parens(text.strip())
    inlined = _inline_function_slots(atom, bodies)
    flat = _inline_derived_param_refs(inlined, scope.derived_exprs) or inlined
    if not _clock_free(flat, frozenset(scope.clocks)):
        return None
    knots: list[float] = []
    # What the tables read hold, for the memo: it outlives a model, and two
    # models can call a table of one name and different contents.
    read_tables: list[tuple] = []
    gate = flat.replace(_SHIFT, "0")
    named = {m.group(1) for m in _TABLE_CALL.finditer(flat)}
    if named:
        tables = {t["name"]: t for t in core.table_function_tables}
        for name in sorted(named):
            table = tables.get(name)
            if table is None or table["index_kind"] != "time":
                return None
            knots.extend(float(x) + shift for x in table["xs"])
            read_tables.append((name, table["method"], tuple(table["xs"]), tuple(table["ys"])))
        gate = _TABLE_CALL.sub("time()", gate)
    if not _reads_clock_and_run_constants(gate, scope):
        return None
    engine = _engine_text(inlined)
    split = _relational_split_op(engine)
    sides = None
    if split is not None:
        kind, sides = "comparison", (split[0], split[2])
    else:
        kind = "truth"
        for pattern, whole_kind in ((_STEP_CALL, "value"), (_BEND_CALL, "bend")):
            m = pattern.match(engine)
            if m is not None and _find_close_paren_strict(engine, m.end() - 1) == len(engine) - 1:
                kind = whole_kind
        if kind == "value" and not _clock_scan.knows_steps_of(engine):
            kind = "value?"  # a step call whose jumps the scan does not know the levels of
    return _EngineCondition(
        engine, inlined, kind, sides, shift, held, tuple(knots), tuple(read_tables)
    )


def _sawtooth_of_a_line(asked: _EngineCondition, scope: SwitchConditionScope):
    """The phase of *asked* where it is a sawtooth (``mod``, ``frac``) of a
    line in time standing as a number, and ``None`` otherwise."""
    from bngsim import _clock_scan

    if asked.kind != "value":
        return None
    for phase in _clock_scan.phases(asked.text):
        if phase.call == asked.text and phase.call.partition("(")[0].strip() in ("mod", "frac"):
            return phase if _clock_scan._a_line_in_time(phase, scope.run_constants) else None
    return None


def _coarse_sawtooth(core, asked: _EngineCondition, phase, lo: float, hi: float) -> bool:
    """Whether the argument of the sawtooth *asked* is coarser than the clock
    of the run from *lo* to *hi*: it holds each of its doubles for more than
    a few tens of the clock's (:data:`_clock_scan.TOUCH_ULPS` of them), with
    a quotient the scan reads (one it does not, it refuses by name).

    A run that restarts on each wrap of such a sawtooth restarts where the
    rate is flat, for as long as the argument holds its value, and where a
    period is a whole number of doubles of the argument the rate is the same
    on every wrap too. The integrator can then take a period in one step and
    add nothing for it (issue #998): with a stop on every wrap of
    `100*mod(1.1e12 + time(), 0.0048828125)` over (0, 20.2), each on its
    wrap, X was 0.0826 for 4.685, where main, with no stop, has it right. Two
    stops to the wrap held for X(0) = 0 and not for X(0) = 1e4. So the scan's
    stops are not placed for it, and it is left as on main. The run now looks
    inside such a stretch before it takes it in one step (issue #998), which
    makes a stop on every wrap add every period for X(0) = 0; whether to place
    them again, and what to do where the dose is small against the tolerance,
    is issue #1018.

    How long a flat stretch it takes depends on the rate and the tolerances,
    not on the run: at a clock of 1e6, `frac((3e9 + time())/0.0009765625)`
    holds a double for one instant of a run of 1 (4,096 doubles of the clock)
    and lost all but a five-hundredth of X to it. No step of the integrator
    is shorter than a hundred doubles of the clock, which is what the line
    here is drawn under: an argument as fine as that is no flat stretch to a
    step.
    """
    import numpy as np

    from bngsim import _clock_scan

    # (By the size of a double of the argument against how far the argument
    # moves in the run, and not by reading it at points along the run: 65
    # evenly spaced points fall on one place in the argument's tread wherever
    # the run is a whole number of its doubles long, and a law that was left
    # alone over (1e6, 1e6 + 1) had its stops, and 0.001 of its X, over
    # (1000000.023, 1000001.023).)
    argument = _clock_scan._split_commas(phase.call.partition("(")[2][:-1])[0].strip()
    ends = np.array([lo, hi], dtype=float)
    try:
        x = asked.evaluate(core, f"({argument})", ends)
        size = np.abs(asked.evaluate(core, phase.phase, ends))
    except Exception:  # noqa: BLE001 - the scan is asked, and says what it finds
        return False
    if not np.all(np.isfinite(x)) or np.any(
        size[np.isfinite(size)] >= _clock_scan._SAWTOOTH_AT_MOST
    ):
        return False
    double_of_the_argument = float(np.spacing(np.max(np.abs(x))))
    double_of_the_clock = float(np.spacing(max(abs(lo), abs(hi))))
    moved = abs(float(x[1] - x[0]))
    # One double of the argument lasts `double_of_the_argument/moved` of the run.
    return (
        double_of_the_argument * (hi - lo)
        > 0.5 * _clock_scan.TOUCH_ULPS * double_of_the_clock * moved
    )


def _engine_scan_stop_times(
    core,
    cond: str,
    asked: _EngineCondition | None,
    offset: float,
    scope: SwitchConditionScope,
    t_start,
    t_end,
    touches: bool,
    by_tread: bool = False,
    reads: Callable[[str], bool] | None = None,
) -> list[tuple[float, float, bool]] | None:
    """``(time, counter value, touch)`` for each change of *cond* in
    ``(t_start, t_end]``, found by evaluating it (:mod:`bngsim._clock_scan`),
    or ``None`` for a condition the evaluator cannot be asked about.

    The last resolver asked, for a condition no other reads (issue #904). The
    time is the first double at which the model's evaluator reads the new
    value. *touch* marks a stop on a window a few doubles wide, kept for an
    event trigger alone.

    With *by_tread*, a condition on a line far from zero that the scan refuses
    or leaves without stops is read tread by tread instead, where it can be
    (:func:`_tread_stop_times`) and where *reads* says a rate law reads it,
    and not for an event's trigger: those stops are a double before each
    change. The scan's own answer is remembered as on main, and the reading
    apart from it, by the condition and the window alone.
    """
    import numpy as np

    from bngsim import _clock_scan

    if asked is None or asked.kind == "value?":
        return None
    shift = asked.shift
    lo, hi = float(t_start) + shift, float(t_end) + shift
    treads = by_tread and not touches and reads is not None
    key = (
        "scan",
        asked.kind,
        touches,
        asked.offset,
        asked.tables,
        *_stop_memo_key(asked.inlined, scope, lo, hi),
    )

    def as_stops(edges):
        at = np.asarray(edges, dtype=float)
        touch = np.zeros(at.size, dtype=bool)
        if touches and at.size > 1:
            narrow = np.diff(at) <= _clock_scan.TOUCH_ULPS * np.spacing(np.abs(at[:-1]))
            touch[:-1] |= narrow
        out = [
            (float(c - shift), float(c), bool(k))
            if shift
            else (float(c), float(c) + offset, bool(k))
            for c, k in zip(at, touch, strict=True)
        ]
        return _TreadStops(out, edges.coarse) if isinstance(edges, _TreadStops) else out

    sawtooth = _sawtooth_of_a_line(asked, scope)
    if sawtooth is not None and _coarse_sawtooth(core, asked, sawtooth, lo, hi):
        read = (
            _left_alone_tread_stops(core, cond, asked, scope, t_start, t_end, offset, reads)
            if treads
            else None
        )
        if read is None:
            logger.debug(
                "%r: a sawtooth of a line coarser than the clock gets no stops (#1018)", cond
            )
        return read

    class Unreadable(Exception):
        pass

    def evaluate(text, at):
        try:
            return asked.evaluate(core, text, at)
        except Exception as exc:  # noqa: BLE001 - whatever the evaluator raises
            raise Unreadable(str(exc)) from None

    def resolve():
        try:
            edges = _clock_scan.scan_changes(
                evaluate,
                asked.text,
                lo,
                hi,
                kind=asked.kind,
                sides=asked.sides,
                knots=asked.knots,
                budget=_RUN_STOP_BUDGET,
                touches=touches,
                constants=scope.run_constants,
            )
        except _clock_scan.Flickers as exc:
            raise _OverStopBudget(
                exc.count,
                unresolved=True,
                flicker=(
                    exc.call,
                    exc.lo - shift,
                    exc.hi - shift,
                    "coarse" if exc.coarse else exc.unread,
                ),
            ) from None
        except _clock_scan.Unresolved as exc:
            raise _OverStopBudget(exc.count, unresolved=True) from None
        except _clock_scan.OverBudget as exc:
            raise _OverStopBudget(exc.count) from None
        except Unreadable as exc:
            # A text the evaluator cannot be asked about is declined, and has
            # the stops it had before there was a scan: none.
            logger.debug("engine scan declined %r: %s", cond, exc)
            return None
        except Exception as exc:  # noqa: BLE001 - said, where it was declined with the rest
            # Anything else that goes wrong in the scan is a defect of it. The
            # condition is left without stops, as every failure to place them
            # leaves a run (`Simulator._apply_crossing_stops`), and said so
            # as that is: declined in silence, an IndexError in here was a
            # window stepped over with no word.
            logger.warning(
                "The scan of the time condition %r failed (%s: %s). No stop is placed for "
                "it, and a rate law that switches on it may be stepped over: bound the "
                "step with max_step. This is a defect in bngsim; please report it.",
                cond,
                type(exc).__name__,
                exc,
            )
            return None
        return edges

    main: _OverStopBudget | None = None
    try:
        edges = _memoized_stops(key, resolve)
    except _OverStopBudget as refused:
        # Refused where it cannot be read every quarter instant: read once on
        # each tread, where it can be, and refused as before where it cannot.
        if refused.flicker is None or not treads or reads is None or not reads(cond):
            raise
        coarse = refused.flicker[0] if refused.flicker[3] == "coarse" else None
        edges = _memoized_stops(
            ("treads", *key),
            lambda: _tread_stop_times(core, cond, asked, scope, lo, hi, coarse=coarse),
        )
        if edges is None:
            raise refused from None
        main = refused
    if edges is None:
        return None
    out = as_stops(edges)
    if isinstance(out, _TreadStops):
        out.main = main
    return out


def _tread_stop_times(
    core,
    cond: str,
    asked: _EngineCondition,
    scope: SwitchConditionScope,
    lo: float,
    hi: float,
    left_alone: bool = False,
    coarse: str | None = None,
) -> list[float] | None:
    """The stops of *asked* from ``lo`` to ``hi`` on the evaluator's clock,
    read once for each double of the line its one read of the clock is in
    (:func:`bngsim._clock_scan.tread_changes`, issue #1018), for a condition
    the scan refuses or, with *left_alone*, leaves without stops. ``None``
    where it cannot be read that way, and the condition is left as the scan
    left it.

    Between two stops the rate law is then exactly constant: the rows of the
    issue the scan refused (`floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)`
    over 240, a dozen doubles of `1e12 + time()` to a `mod`) are exact to
    rounding, and `100*mod(1.1e12 + time(), 0.0048828125)`, which the scan
    left alone, was 21% over by the sixth of six legs of 0.2.

    *coarse* is the sawtooth call the scan refused as beyond 2^48: the run is
    made as the evaluator reads the law, a staircase, and is warned that the
    call resolves only a few steps of its period.

    Past the run's budget of stops, a condition the scan refused is over it,
    and the rule for that stands: a step bound of half its shortest stretch
    where it is all that switches, refused by name otherwise. One the scan
    left alone is left so, and the run is warned. Bounded at a tread,
    `100*frac(1e6 + time())` over (3, 10) is 6e10 steps, past what a bound
    may cost, where the run without stops is right; over (1e6, 1e6 + 1),
    `100*frac((3e9 + time())/0.0009765625)` came out 4.9e-4 off, for 2.2e-5
    without.
    """
    from bngsim import _clock_scan

    text = asked.text
    if len(_clock_scan._READS_CLOCK.findall(text)) != 1:
        return None
    parts = _clock_scan.lines_holding_the_clock(text, scope.run_constants)
    if not parts:
        return None
    read: _clock_scan.Treads | None = None
    unread: tuple[str, _clock_scan.Treads] | None = None
    # The innermost part of the line that holds the clock first: the doubles
    # of `(1.1e12 + time())/0.0048828125` skip, as a quotient by 5/1024 of
    # doubles 2.4e-4 apart is, and that sawtooth was left without stops, where
    # the sum inside takes each of its doubles in turn. A part too fine to read
    # is passed over for the next one out: `time() - 1` inside
    # `1e12 + (time() - 1)` holds a double of the clock to a tread.
    line = parts[-1]
    for part in parts:
        try:
            got = _clock_scan.tread_changes(
                lambda what, at: asked.evaluate(core, what, at),
                text,
                part,
                lo,
                hi,
                _RUN_STOP_BUDGET,
            )
        except Exception as exc:  # noqa: BLE001 - declined, and left as the scan left it
            logger.debug("tread-by-tread reading declined %r: %s", cond, exc)
            return None
        if got is None:
            continue
        if not got.read:
            if part == parts[-1]:
                unread = (part, got)
            continue
        read, line = got, part
        break
    if read is None:
        if unread is None:
            return None
        line, read = unread
    if read.stops is not None:
        return _TreadStops([float(t) for t in read.stops], coarse)
    if not left_alone:
        if read.read:
            # Refused by name, with a leg that fits (#890's step bound in place
            # of the stops left `100*mod(1e12 + time(), 12u)` over 130 0.4% off
            # at the default tolerances, and 4.5% from X(0) = 1e4; over 100,
            # each change stopped at, it is exact to rounding).
            raise _OverStopBudget(read.changes, min_gap=read.shortest, legs=read.fits)
        return None
    shift = asked.shift
    return _TreadStops(
        [],
        past=(read.changes, lo - shift, hi - shift, None if read.read else line, read.fits),
    )


class _TreadStops(list):
    """Stops read tread by tread (issue #1018), and what the run is told of
    them: *coarse*, the sawtooth call beyond 2^48 they are the steps of;
    *past*, ``(changes, t0, t1, line, leg that fits)`` where there were more
    than a run places or reads, and none are placed.
    What a reading finds is the condition's and the window's alone, and is
    remembered so: a law in a function no rate law reads, read and remembered
    as no stops, was no stops for the next model that did read it."""

    def __init__(
        self,
        items=(),
        coarse: str | None = None,
        past: tuple | None = None,
        main: _OverStopBudget | None = None,
    ):
        super().__init__(items)
        self.coarse = coarse
        self.past = past
        # Main's answer where it refused the law: what the run is given again
        # where these stops are not placed.
        self.main = main


def _say_beyond_2_48(cond: str, call: str) -> None:
    _say_once(
        cond,
        "2^48",
        "The rate law switches on %r, and the whole number of periods `%s` takes off its "
        "argument is beyond 2^48 in this run: the call resolves only a few steps of each "
        "period, each a large part of it. The run is made as the evaluator reads the rate "
        "law, a staircase with a stop on each step (issue #1018). For a sawtooth, take the "
        "offset out of what the call divides: `mod(time(), P)`, not `mod(1e12 + time(), P)`.",
        cond,
        call.partition("(")[0].strip(),
    )


# The conditions and kinds of warning :func:`_say_once` has said.
_SAID: set[tuple[str, str]] = set()


# What the readings of the run being placed have to say, said once its stops
# are taken (:func:`say_what_was_read`).
_PENDING: list[tuple[str, str, str, tuple]] = []


def _say_once(cond: str, kind: str, message: str, *args) -> None:
    """Warn *message* the first time a run of this session meets *cond* for
    *kind*: a loop of a thousand legs is one warning, not a thousand. Said
    when the run's stops are taken: said as another condition refused the
    run, the warning was spent, and the legs run next had none."""
    if (cond, kind) not in _SAID:
        _PENDING.append((cond, kind, message, args))


def say_what_was_read(taken: bool) -> None:
    """Say what the readings of the run just placed found, where its stops
    were *taken*; drop it where they were not."""
    pending = list(_PENDING)
    _PENDING.clear()
    if not taken:
        return
    for cond, kind, message, args in pending:
        if (cond, kind) not in _SAID:
            _SAID.add((cond, kind))
            logger.warning(message, *args)


def _three_places_down(x: float) -> str:
    """*x* to three significant figures, rounded down: a leg of 0.477 where
    0.476837 fits held 1,000,341 changes, and was over the budget again."""
    if not (x > 0.0 and math.isfinite(x)):
        return f"{x:g}"
    scale = 10.0 ** (math.floor(math.log10(x)) - 2)
    return f"{math.floor(x / scale) * scale:.3g}"


def _say_left_alone(
    cond: str,
    changes: int,
    t0: float,
    t1: float,
    line: str | None = None,
    fits: float | None = None,
    others: int = 0,
) -> None:
    """The warning for a law read tread by tread that is integrated as before
    issue #1018 after all, for want of the stops or the reads. *fits* is the
    longest leg that would have them."""
    from bngsim import _clock_scan

    span = abs(t1 - t0)
    if line is not None:
        what = (
            f"is read through {line!r}, which takes {changes} doubles between t={t0!r} and "
            f"t={t1!r}, more than bngsim reads for one law in one run "
            f"({_clock_scan.TREAD_READS:,})"
        )
        leg = span * _clock_scan.TREAD_READS / max(changes, 1)
    elif others:
        what = (
            f"changes {changes} times between t={t0!r} and t={t1!r}, which with the "
            f"{others:,} stops of the run's other switches is more than the "
            f"{_RUN_STOP_BUDGET:,} bngsim places in one run"
        )
        leg = span * _RUN_STOP_BUDGET / max(changes + others, 1)
    else:
        what = (
            f"changes {changes} times between t={t0!r} and t={t1!r}, more than the "
            f"{_RUN_STOP_BUDGET:,} stops bngsim places in one run"
        )
        leg = span * _RUN_STOP_BUDGET / max(changes, 1)
    _say_once(
        cond,
        "left alone",
        "The rate law switches on %r, which %s. It is integrated without a stop on each "
        "change, as before issue #1018, and is as near as the error test finding each "
        "change makes it. Run it in legs of %s or shorter (run_until) to have them.",
        cond,
        what,
        _three_places_down(min(leg, fits) if fits is not None else leg),
    )


def _leg_that_fits(times) -> float:
    """The shortest stretch holding one stop more than a run's budget: a leg
    shorter than it holds no more than the budget."""
    import numpy as np

    at = np.sort(np.asarray(list(times), dtype=float))
    if at.size <= _RUN_STOP_BUDGET:
        return math.inf
    return float(np.min(at[_RUN_STOP_BUDGET:] - at[:-_RUN_STOP_BUDGET]))


class _LeftAloneStops(_TreadStops):
    """The stops :func:`_left_alone_tread_stops` read for a law main leaves
    without any, which a run takes once every other condition has its own."""


# A period of a sawtooth of a line, in doubles of its argument, up to which
# the sawtooth is read tread by tread where main leaves it without stops.
# Finer, it is a staircase of steps under a thousandth of the period, and main
# integrates it as the line it nearly is: `100*mod(1.7e9 + time(), 86400)/86400`
# over 0.2 took 14 steps and was right to 4e-10, and a stop on each of its
# 838,861 treads took 1.7 million steps and 5.6 s. With 20 doubles to a period,
# `100*mod(1.1e12 + time(), 0.0048828125)` was 21% over on main.
_TREADS_TO_A_PERIOD = 1024.0


def _sawtooth_treads(core, asked: _EngineCondition, phase, lo: float, hi: float):
    """``(doubles of the argument to a period, beyond 2^48)`` for the sawtooth
    *phase* of *asked* from *lo* to *hi*, the fewer doubles of the two ends;
    ``None`` where it cannot be read."""
    import numpy as np

    from bngsim import _clock_scan

    argument = _clock_scan._split_commas(phase.call.partition("(")[2][:-1])[0].strip()
    ends = np.array([lo, hi], dtype=float)
    try:
        x = np.abs(np.asarray(asked.evaluate(core, f"({argument})", ends), dtype=float))
        size = np.abs(np.asarray(asked.evaluate(core, phase.phase, ends), dtype=float))
    except Exception:  # noqa: BLE001 - the reading is declined
        return None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(size))) or np.any(size == 0.0):
        return None
    if np.any(x == 0.0):
        return None
    doubles = float(np.min((x / size) / np.spacing(x)))
    return doubles, bool(np.any(size >= _clock_scan._SAWTOOTH_AT_MOST))


def _left_alone_tread_stops(
    core,
    cond: str,
    asked: _EngineCondition | None,
    scope: SwitchConditionScope,
    t_start,
    t_end,
    offset: float,
    reads: Callable[[str], bool] | None = None,
) -> _LeftAloneStops | None:
    """``(time, counter value, touch)`` for each change of a sawtooth of a line
    far from zero standing as a number, which main leaves without stops (the
    scan declines it as coarser than the clock, or a resolver lists no change
    of it), read tread by tread (issue #1018). ``None`` where it is left so:
    where a period is more than :data:`_TREADS_TO_A_PERIOD` doubles of its
    argument, or where it cannot be read.

    One beyond 2^48 is read so too, and warned of where its stops are placed;
    where it cannot be read, or past the budget, it is left as main leaves it.
    (Refused there, it refused runs main has right: `100*mod(1e8 + time(),
    20u)` at t = 1e6, whose stops CVODE cannot take apart, is within a tenth
    of a local tolerance on main.)
    """
    if asked is None:
        return None
    phase = _sawtooth_of_a_line(asked, scope)
    if phase is None:
        return None
    shift = asked.shift
    lo, hi = float(t_start) + shift, float(t_end) + shift
    treads = _sawtooth_treads(core, asked, phase, lo, hi)
    if treads is None or treads[0] > _TREADS_TO_A_PERIOD:
        return None
    # (Last: it reads every rate law, which a leg of a model with an ordinary
    # schedule paid for at three times main's cost a leg.)
    if reads is None or not reads(cond):
        return None
    beyond = treads[1]
    key = (
        "treads",
        asked.kind,
        asked.offset,
        asked.tables,
        *_stop_memo_key(asked.inlined, scope, lo, hi),
    )

    def resolve():
        return _tread_stop_times(
            core,
            cond,
            asked,
            scope,
            lo,
            hi,
            left_alone=True,
            coarse=phase.call if beyond else None,
        )

    edges = _memoized_stops(key, resolve)
    if edges is None or not (edges or edges.past):
        return None
    stops = [
        (float(c - shift), float(c), False) if shift else (float(c), float(c) + offset, False)
        for c in edges
    ]
    return _LeftAloneStops(stops, edges.coarse, edges.past)


def _few_changes(
    core, asked: _EngineCondition | None, offset: float, scope, t_start, t_end
) -> list[tuple[float, float, bool]] | None:
    """The changes of a comparison whose step calls jump more often than the
    run has stops for, where it can be seen not to follow them
    (:func:`bngsim._clock_scan.few_changes`), or ``None``."""
    from bngsim import _clock_scan

    if asked is None or asked.kind != "comparison" or asked.sides is None:
        return None
    shift = asked.shift
    lo, hi = float(t_start) + shift, float(t_end) + shift

    def resolve():
        try:
            return _clock_scan.few_changes(
                lambda t, at: asked.evaluate(core, t, at),
                asked.text,
                lo,
                hi,
                asked.sides,
                budget=_RUN_STOP_BUDGET,
            )
        except _clock_scan.Flickers as exc:
            # (Named, and not folded into the budget's refusal, which a
            # `max_step` can answer.)
            raise _OverStopBudget(
                exc.count,
                unresolved=True,
                flicker=(
                    exc.call,
                    exc.lo - shift,
                    exc.hi - shift,
                    "coarse" if exc.coarse else exc.unread,
                ),
            ) from None
        except Exception as exc:  # noqa: BLE001 - not read: the caller reports the budget
            logger.debug("few-changes reading declined %r: %s", asked.text, exc)
            return None

    key = ("few", asked.offset, asked.tables, *_stop_memo_key(asked.inlined, scope, lo, hi))
    edges = _memoized_stops(key, resolve)
    if edges is None:
        return None
    return [(c - shift, c, False) if shift else (c, c + offset, False) for c in edges]


# How many doubles of the time on each side of a stop the evaluator is asked on
# before anything is searched for. A stop read off folded constants lands
# within one or two of the double a condition on the bare clock flips at.
_FLIP_REACH = 8
# How many stops' doubles are read at once.
_FLIP_BLOCK = 1 << 16


def _stops_on_the_flip(
    core, engine, text: str, offset: float, scope, bodies, t_start, t_end, times: Sequence[float]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    """Each of *times* moved onto the first double at which the model's
    evaluator reads the condition's new value, as ``(time, counter value)``,
    and after them any change of it at the ends of the run that *times* does
    not list; ``None`` where the evaluator cannot be asked (issue #891).

    The resolvers work a stop out from folded constants: a period, an offset
    and a duty, or ``(n - beta)/alpha``. That is not the arithmetic the
    evaluator computes the condition with, and the stop can sit before the
    double the condition changes on. The run then restarts on the branch that
    just ended, and the jump falls inside the first step after the restart,
    which has no history to see it with. With the clock read bare the two are
    a double of the time apart, and the error (3e-6 at rtol 1e-8 with a period
    of 0.3) did not follow the tolerance down. With an offset in the argument
    they are up to a double of the *argument* apart, which is 1.2e-4 of time
    at ``1e12 + time()``: billions of doubles of the clock.

    So the evaluator is asked. First on the doubles around each stop, which
    is where a condition on the bare clock changes; then, for a stop with no
    change that near, between the two times half way to its neighbours, by
    bisection. A stop with no change of the condition between those is left
    where it was put, and so is one whose neighbours crowd it out: a change
    further from its stop than half the way to the next is not this stop's.

    *engine* makes the condition for the evaluator and is called only where
    the answer is not remembered; *text* is the condition with its counter
    written as time, which with the window and the times is what it is
    remembered under.
    """
    import numpy as np

    from bngsim import _clock_scan

    if not times:
        return None

    def resolve():
        asked = engine()
        if asked is None or asked.kind not in ("comparison", "truth", "value"):
            return None
        shift = asked.shift

        def ev(expr, at):
            return asked.evaluate(core, expr, at)

        if asked.kind == "value":
            # What the call holds from one jump to the next: its value where it
            # steps, the whole number it takes off its argument where it is a
            # sawtooth.
            held = _clock_scan._held(ev, asked.text)
            if held is None:
                return None
            reading, by = held.reading, held.by
        else:

            def reading(at):
                values = ev(asked.text, at)
                return np.where(np.isnan(values), 2.0, (values != 0.0).astype(float))

            by = 0.0

        def differ(a, b):
            return _clock_scan._differs(a, b, by)

        order = np.argsort(np.asarray(times, dtype=float), kind="stable")
        at = np.asarray(times, dtype=float)[order] + shift
        moved = at.copy()
        try:
            # The doubles around each stop, a block of stops at a time: the
            # grid and its readings for a million stops at once took 1.2 GB.
            reach = _FLIP_REACH
            gaps = np.arange(2 * reach)
            # The change nearest the stop as placed; of two as near, the later.
            near = np.abs(gaps + 0.5 - reach)[None, :] - 1e-3 * gaps[None, :]
            found = np.zeros(at.size, dtype=bool)
            for lo_i in range(0, at.size, _FLIP_BLOCK):
                block = at[lo_i : lo_i + _FLIP_BLOCK]
                grid = np.empty((block.size, 2 * reach + 1))
                grid[:, reach] = block
                down = up = block
                for j in range(1, reach + 1):
                    down = np.nextafter(down, -np.inf)
                    up = np.nextafter(up, np.inf)
                    grid[:, reach - j] = down
                    grid[:, reach + j] = up
                read = reading(grid.ravel()).reshape(grid.shape)
                changed = differ(read[:, 1:], read[:, :-1])
                pick = np.argmin(np.where(changed, near, np.inf), axis=1)
                hit = changed.any(axis=1)
                rows = np.nonzero(hit)[0]
                moved[lo_i + rows] = grid[rows, pick[rows] + 1]
                found[lo_i : lo_i + block.size] = hit
            far = np.nonzero(~found)[0]
            if far.size:
                # Between the times half way to its neighbours, and inside the
                # run: one side of the change at one end, the other at the other.
                lo, hi = float(t_start) + shift, float(t_end) + shift
                mid = 0.5 * (at[:-1] + at[1:])
                a = np.concatenate(([lo], mid))[far]
                b = np.concatenate((mid, [hi]))[far]
                side_a, side_b = reading(a), reading(b)
                across = differ(side_a, side_b) & (b > a)
                if across.any():
                    low = side_a[across]

                    def side_of(t, rows):
                        # On the low end's side or not: a third reading, where
                        # more than one change lies between, is not.
                        return np.where(differ(reading(t), low[rows]), 1.0, 0.0)

                    moved[far[across]] = _clock_scan._bisect(
                        side_of, a[across], b[across], np.zeros(int(across.sum()))
                    )
            # A change the resolvers put on the run's last instant or past it,
            # or on its first or before it, they do not list, and the
            # evaluator can have it inside the run: the wrap of
            # `mod(1e12 + time(), 0.125)` that ends a run of whole periods is
            # half a double of the argument before the end. So the reading
            # after the last stop is held against the one on the run's last
            # instant, and the reading on its first against the one before the
            # first stop.
            more: list[float] = []
            lo, hi = float(t_start) + shift, float(t_end) + shift
            first, last = float(np.min(moved)), float(np.max(moved))
            below = float(np.nextafter(first, -np.inf))
            for a, b in ((last, hi), (lo, below)):
                if not b > a:
                    continue
                ends = reading(np.array([a, b]))
                if differ(ends[:1], ends[1:])[0]:

                    def side_at_an_end(t, rows, low=ends[:1]):
                        return np.where(differ(reading(t), low[rows]), 1.0, 0.0)

                    more.append(
                        float(
                            _clock_scan._bisect(
                                side_at_an_end, np.array([a]), np.array([b]), np.zeros(1)
                            )[0]
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - the stops stand as placed
            logger.debug("stops on the flip: %r not evaluated (%s)", text, exc)
            return None
        back = np.empty_like(moved)
        back[order] = moved
        return [shift, back.tolist(), more]

    key = (
        "flip",
        tuple(times) if len(times) <= 64 else (len(times), times[0], times[-1], math.fsum(times)),
        *_stop_memo_key(
            _inline_function_slots(text, bodies) if bodies else text, scope, t_start, t_end
        ),
    )
    answer = _memoized_stops(key, resolve)
    if answer is None:
        return None
    shift, moved, more = answer

    def stop(c):
        return (c - shift, c) if shift else (c, c + offset)

    return [stop(c) for c in moved], [stop(c) for c in more]


# Grid cells the evaluator is asked on when a resolver's stops are checked.
_AGREEMENT_CELLS = 1 << 12


def _agrees_with_evaluator(
    core, engine, text: str, scope, bodies, t_start, t_end, times: Sequence[float]
) -> bool:
    """Whether every change of the condition the evaluator shows on a grid over
    the window has one of *times* in its cell.

    A resolver reads a text's shape, and a shape can be misread: a staircase
    passes the linear resolver's two probes (``floor(time()/0.7) >= 3`` was
    stopped at 2.104 for 2.1), and the step resolver reported no edge at all
    for ``mod(rint(time()-0.01), 0.1) > floor(time())``. The evaluator is what
    the rate law follows, so a change it shows between two grid points with no
    stop there means the stops are not this condition's.

    *text* is the condition with its counter written as time. With the bodies
    of the function slots it names written in, the window and the values it
    reads, that is everything the answer depends on, and what it is remembered
    under: two SBML models can both spell a trigger ``g > 0`` over an
    assignment rule ``g`` that is a line in one and a cubic in the other.
    *engine* makes the condition for the evaluator, and is called only where
    the answer is not remembered.

    A condition is checked by its truth. A step call standing as a number is
    checked by its value where that holds between jumps (``sign``, ``floor``):
    ``sign(min((time()-0.3)/0.0003, ln(time()-8.4)))`` was stopped at 0.3 and
    9.4 and not at 8.4, where the logarithm starts being a number. A call
    whose value moves in most cells (a sawtooth) says nothing this way and is
    not checked. True where the evaluator cannot be asked.
    """
    import numpy as np

    slack_ulps = float(_INSTANT_ULPS)

    def resolve():
        asked = engine()
        if asked is None or asked.kind not in ("comparison", "truth", "value"):
            return [1.0]
        if _sawtooth_of_a_line(asked, scope) is not None:
            # Two values of a sawtooth a cell apart are no jump in the cell,
            # in however few cells they differ: where its argument is coarse
            # against the run (a leg under a thousand doubles of
            # `1.1e12 + time()`, a window of 4,096 periods to the cell), few
            # do, a resolver's "no change" was held against them, and the
            # scan was asked for a law main leaves alone and has right.
            return [1.0]
        shift = asked.shift
        lo, hi = float(t_start) + shift, float(t_end) + shift
        nodes = np.linspace(lo, hi, _AGREEMENT_CELLS + 1)
        # A change on the run's first or last instant is no stop of the run's:
        # a resolver lists neither, and where it puts one a few doubles from
        # where the evaluator does (issue #891), either may fall just outside.
        # (Doubles of the window's scale: at lo = 0 a few of its own are
        # subnormal, and `time()/1000` underflows on the first of them.)
        edge = 2.0 * slack_ulps * float(np.spacing(max(abs(lo), abs(hi))))
        nodes[0] = lo + edge
        nodes[-1] = hi - edge
        nodes = np.unique(nodes)
        if nodes.size < 2:
            return [1.0]
        try:
            values = asked.evaluate(core, asked.text, nodes)
        except Exception as exc:  # noqa: BLE001 - nothing to check against
            logger.debug("agreement check: %r not evaluated (%s)", asked.text, exc)
            return [1.0]
        if asked.kind == "value":
            # Not a number is a value of its own, equal to itself.
            held = np.where(np.isnan(values), np.inf, values)
            changed = np.nonzero(held[:-1] != held[1:])[0]
            if changed.size > nodes.size // 4:
                return [1.0]
        else:
            if asked.kind == "comparison" and asked.sides is not None:
                # A comparison is read between the grid points as well, where
                # the difference of its sides turns: `(time()-5)^4 < 1e-10`
                # holds for 0.006 around 5, inside one cell of a run 600
                # long, and the polynomial resolver, which samples the window
                # too, had no stop for it there and nothing here said so.
                from bngsim import _clock_scan

                try:
                    residual = _clock_scan._residual(
                        lambda text, at: asked.evaluate(core, text, at),
                        asked.sides[0],
                        asked.sides[1],
                        lo,
                    )
                    turns = _clock_scan._extrema(residual, nodes, residual(nodes))
                    turns = turns[(turns > nodes[0]) & (turns < nodes[-1])]
                    if turns.size:
                        nodes = np.union1d(nodes, turns)
                        values = asked.evaluate(core, asked.text, nodes)
                except Exception as exc:  # noqa: BLE001 - the grid alone, as before
                    logger.debug("agreement check: %r not read between (%s)", asked.text, exc)
            truth = np.where(np.isnan(values), 2.0, (values != 0.0).astype(float))
            changed = np.nonzero(truth[:-1] != truth[1:])[0]
        if changed.size == 0:
            return [1.0]
        at = np.sort(np.asarray(times, dtype=float) + shift)
        # A stop on the cell's own edge counts for the cell on either side.
        slack = slack_ulps * np.spacing(np.abs(nodes))
        first = np.searchsorted(at, nodes[changed] - slack[changed], side="left")
        last = np.searchsorted(at, nodes[changed + 1] + slack[changed + 1], side="right")
        return [1.0 if bool(np.all(last > first)) else 0.0]

    key = (
        "agrees",
        tuple(times) if len(times) <= 64 else (len(times), times[0], times[-1]),
        *_stop_memo_key(
            _inline_function_slots(text, bodies) if bodies else text, scope, t_start, t_end
        ),
    )
    return bool(_memoized_stops(key, resolve)[0])


# Doubles on each side of a solved instant on which an equality is read.
_EQUALITY_REACH = 8
# How many times a near miss is looked four times further on each side for
# where the difference of the sides next moves: out to 2^22 doubles, a
# billionth of the time. Further, the next move found is a staircase's next
# step as often as a coarse side's next rounding: at t = 1e7, 2^32 doubles is
# eight time units, and `floor(time/2)*2` stepping past a value it takes two
# units later was refused as never equal to it.
_NEAR_MISS_REACH = 8


class TriggerEqualityNeverHolds(Exception):
    """An event's trigger is an equality on the clock that no double satisfies
    inside the run, so the event could never fire."""


def _equality_stops(
    core, cond: str, text: str, asked: _EngineCondition, scope, bodies, t_start, t_end, alone: bool
) -> list[float]:
    """The doubles in ``(t_start, t_end]`` at which the equality *cond* holds
    (or, for ``!=``, fails): the first of each run of them.

    ``time == tau/7`` is true where the clock equals the evaluator's own
    rounding of ``tau/7``, so the evaluator is asked: its two sides are read
    for where they pass each other (:func:`bngsim._clock_scan.scan_changes`
    with a trigger's touches kept), which finds ``time == pi`` and
    ``time^2 == 4`` as well as a parameter.

    Raises :class:`TriggerEqualityNeverHolds` for a ``==`` that is the whole of
    an event's trigger (*alone*) and whose sides pass each other between two
    adjacent doubles without being equal on either: ``time - 1 == 1.3`` has no
    solution in doubles, since 2.3 - 1 rounds to 1.2999999999999998 and the
    next double gives 1.3000000000000003. Inside a larger trigger the event
    can still fire through the rest of it, and a ``!=`` that never fails is
    simply true.
    """
    import numpy as np

    from bngsim import _clock_scan

    if asked.sides is None:
        return []
    shift = asked.shift
    lo, hi = float(t_start) + shift, float(t_end) + shift
    lhs, rhs = asked.sides
    is_ne = "!=" in asked.text[len(lhs) : len(asked.text) - len(rhs)]

    def resolve():
        def ev(text, at):
            return asked.evaluate(core, text, at)

        # `time == 12`: the instant is solved, and the double the evaluator
        # holds the equality at is one of the few around it. Searching the
        # whole window for it took a model with 62 such triggers from 3 ms to
        # most of a second.
        if _reads_time_once(text, scope):
            solved = _crossing_time_of_condition(text, scope, t_start, t_end, bodies)
            if solved is not None:
                if not (lo - 1.0 < solved + shift <= hi + 1.0):
                    return [[], 0.0]
                near = np.full(2 * _EQUALITY_REACH + 1, solved + shift)
                for k in range(_EQUALITY_REACH):
                    near[: _EQUALITY_REACH - k] = np.nextafter(
                        near[: _EQUALITY_REACH - k], -np.inf
                    )
                    near[_EQUALITY_REACH + 1 + k :] = np.nextafter(
                        near[_EQUALITY_REACH + 1 + k :], np.inf
                    )
                try:
                    holds = ev(asked.text, near) != 0.0
                except Exception as exc:  # noqa: BLE001 - searched for below instead
                    logger.debug("equality stop: %r not evaluated (%s)", cond, exc)
                    holds = None
                if holds is not None:
                    met = ~holds if is_ne else holds
                    if met.any() and not met.all():
                        # The first double of the run of them.
                        first = int(np.argmax(met))
                        return [[float(near[first])], 0.0]

        try:
            # Where the equality's truth changes, a window a double wide kept.
            edges = _clock_scan.scan_changes(
                ev,
                asked.text,
                lo,
                hi,
                kind="comparison",
                sides=asked.sides,
                knots=asked.knots,
                budget=_RUN_STOP_BUDGET,
                touches=True,
            )
            # Where the sides pass each other with no double between: the
            # ordering of the same two sides changes there, and the equality
            # does not.
            passes = _clock_scan.scan_changes(
                ev,
                f"({lhs})>({rhs})",
                lo,
                hi,
                kind="comparison",
                sides=asked.sides,
                knots=asked.knots,
                budget=_RUN_STOP_BUDGET,
                touches=True,
            )
        except Exception as exc:  # noqa: BLE001 - not read: no stop, as before
            logger.debug("equality stop: %r not evaluated (%s)", cond, exc)
            return [[], 0.0]
        at = np.asarray(edges, dtype=float)
        held = ev(asked.text, at) != 0.0 if at.size else np.zeros(0, dtype=bool)
        # The first double of each run the equality holds on (or `!=` fails on).
        starts = at[~held if is_ne else held]
        missed = 0.0
        for t in passes:
            near = np.abs(at - t) <= 4.0 * _INSTANT_ULPS * np.spacing(abs(t)) if at.size else []
            if np.any(near):
                continue
            # A near miss is the two sides passing each other as they move.
            # Where one of them jumps across the other (the wrap of
            # `rem(time, 3)` past 1, three times in ten time units) nothing was
            # nearly equal, and the trigger holds elsewhere or not at all.
            #
            # A jump is told by the difference of the sides changing more
            # across the two doubles of the pass than it does where it next
            # moves on either side of them. (Not by how close the sides are in
            # doubles of their own size: `time - 100 == 0.3` misses by a
            # rounding of 100.3, which is many doubles of 0.3, and went
            # unrefused; and `1e15 + rem(time, 3)` wraps by 24 doubles of
            # itself past `1e15 + 1`, which it does equal a second later, and
            # was refused.) Where it next moves, and not a fixed few doubles
            # on: a side coarser than the clock, `(time + 1024)*1.9`, holds
            # one value for four thousand doubles, and read over 64 it had
            # not moved at all beside a pass of one of its own last places.
            beside = np.array([np.nextafter(t, -np.inf), t])
            d = ev(f"({lhs})", beside) - ev(f"({rhs})", beside)
            across = abs(d[1] - d[0])

            # (A pass beside a double the sides are equal on is the end of a
            # stretch on which the equality held, and no miss.)
            drift = 0.0
            for k in (0, 1):
                step = (1.0 if k else -1.0) * _INSTANT_ULPS * float(np.spacing(abs(t)))
                for _ in range(_NEAR_MISS_REACH):
                    probe = np.array([beside[k] + step])
                    moved = abs(float((ev(f"({lhs})", probe) - ev(f"({rhs})", probe))[0]) - d[k])
                    if moved != 0.0:  # a move, or not a number
                        drift = max(drift, moved)
                        break
                    step *= 4.0
            if bool(np.all(np.isfinite(d))) and bool(np.all(d != 0.0)) and across <= 2.0 * drift:
                missed = float(t)
                break
        return [[float(x) for x in starts], missed]

    key = ("equality", asked.offset, asked.tables, *_stop_memo_key(asked.inlined, scope, lo, hi))
    starts, missed = _memoized_stops(key, resolve)
    if missed and alone and not is_ne:
        raise TriggerEqualityNeverHolds(
            f"The event trigger {_strip_redundant_parens(cond.strip())!r} is an equality "
            f"whose two sides are never equal in double arithmetic near t={missed - shift!r}: "
            "where they pass each other there, they differ by a rounding on every double, "
            "so the trigger does not turn true and the event does not fire. Write the "
            "instant itself on one side (time == 2.3 rather than time - 1 == 1.3), or use "
            "an ordering (>=)."
        )
    return [t - shift for t in starts if t_start < t - shift <= t_end]


# Which breakpoints of a table function are stops (issue #709).
#
# A stop exists for a feature a step could span unseen: one that leaves and
# comes back, so that the table is the same at both ends of the step. Such a
# feature has breaks that differ: a pulse is a jump up and a jump down, a
# triangle a turn one way, twice that the other way, and the first again. A
# break that its two neighbours account for (it is their mean, to within a
# quarter of the largest of the three) is in the middle of a stretch where the
# table only goes on rising or falling, or bending the one way, or passes
# smoothly from bending one way to the other. A step across that sees it, at
# its end.
#
# So a breakpoint is a stop where its neighbours do not account for its break,
# or where the breakpoint before it or after it has no break to speak of, under
# a quarter of its own (the foot of a feature on a flat or straight stretch,
# and the ends of the table), or, in a step table, where the jump beside it
# goes the other way. Within a stretch of breaks that do account for each
# other, there is one each time they have added up to this fraction of the
# range of the table's values.
#
# That keeps the stops of a table of many samples of a smooth signal to its
# ends and a few along it: a stop on each of 100,000 samples took a run from
# 0.3 ms to 300. It does not judge a break by its size beside the table's
# largest: maintenance doses of 5 after a loading dose of 1000 are half a
# percent of the range, every one a pulse.
_TABLE_RUN = 0.05
# How far a break may be from the mean of its neighbours and be accounted for
# by them, as a fraction of the largest of the three.
_TABLE_UNLIKE = 0.25
# A break below this many ulps of the table's largest value is the rounding of
# a slope, on a stretch that is straight.
_TABLE_STRAIGHT_ULPS = 64.0


def table_function_breaks(core) -> list[tuple[str, int, object, object]]:
    """``(index kind, observable index, restarts, ends)`` for each table
    function of the model that breaks anywhere: the breakpoints, as two
    arrays, at which the run is stopped, and restarted or not.

    A ``step`` table breaks where its value jumps, by the jump. A ``linear``
    one breaks where its slope turns, by the turn over the shorter segment
    beside it. Which breaks are stops is the rule above :data:`_TABLE_RUN`.

    The integrator restarts at every stop of a ``step`` table, where the rate
    law jumps, and at every bend of a ``linear`` table that its neighbours do
    not account for. On one segment of such a table the rate law is straight,
    which the integrator follows exactly, so its step grows to the length of
    the segment, and at the bend that step is too long by however steep the
    table turns. Left to its error test there, it either gave up (seven
    failures did not bring the step down from a flat stretch of 100 to a ramp
    of slope 10), or, in a run that ended inside the ramp, stepped past the end
    of the feature and read the table as flat. Only at a stop along a stretch
    of small, like bends (a dense table of a smooth signal) does it end a step
    and go on.

    Read from the tables alone, so a model's is the same for every run:
    :meth:`bngsim.Model.table_function_breaks` keeps it. A table no expression
    of the model calls (one added after load) is left out.
    """
    import numpy as np

    out: list[tuple[str, int, object, object]] = []
    for table in core.table_function_tables:
        if not table.get("read", True):
            # Added after load, and called by no expression (issue #772): the
            # run never reads it, so it has nowhere to break.
            continue
        xs = np.asarray(table["xs"], dtype=float)
        ys = np.asarray(table["ys"], dtype=float)
        if xs.size < 2:
            continue
        bends = table["method"] != "step"
        width = np.diff(xs)
        if bends:
            # Held at its end values outside the table, so the ends bend too.
            slope = np.concatenate(([0.0], np.diff(ys) / width, [0.0]))
            reach = np.minimum(
                np.concatenate(([width[0]], width)), np.concatenate((width, [width[-1]]))
            )
            size = np.diff(slope) * reach
            knots = xs
        else:
            size = np.diff(ys)
            knots = xs[1:]
        size = np.where(
            np.abs(size) <= _TABLE_STRAIGHT_ULPS * np.spacing(float(np.abs(ys).max())), 0.0, size
        )
        real = size != 0.0
        if not real.any():
            continue
        before = np.concatenate(([0.0], size[:-1]))
        after = np.concatenate((size[1:], [0.0]))
        # A neighbour under a quarter of the break is none: asked for an exact
        # zero, a baseline that wiggles by a millionth (data rounded to six
        # digits) was a neighbour, the foot of a rise beside it was accounted
        # for, and a run that ended inside the rise read the table as flat.
        beside_none = (np.abs(before) < _TABLE_UNLIKE * np.abs(size)) | (
            np.abs(after) < _TABLE_UNLIKE * np.abs(size)
        )
        largest = np.maximum(np.abs(size), np.maximum(np.abs(before), np.abs(after)))
        unlike = np.abs(size - 0.5 * (before + after)) > _TABLE_UNLIKE * largest
        feature = real & (beside_none | unlike)
        if not bends:
            # A step table turns where a jump up meets a jump down: the top of
            # a sampled hump, whose two jumps are each the mean of what is
            # beside them and between them are a pulse.
            feature |= real & ((before * size < 0.0) | (after * size < 0.0))
        # Within a stretch of breaks that account for each other: what they
        # have added up to since the last stop, in steps of the fraction of
        # the range.
        total = np.cumsum(np.abs(size))
        since = total - np.maximum.accumulate(np.where(feature, total, 0.0))
        level = np.floor(since / (_TABLE_RUN * float(np.ptp(ys))))
        along = real & ~feature & (level > np.concatenate(([0.0], level[:-1])))
        hard = feature if bends else feature | along
        soft = along & ~hard
        out.append((table["index_kind"], int(table["index_obs_idx"]), knots[hard], knots[soft]))
    return out


def table_function_stops(core, t_start: float, t_end: float, breaks=None) -> list[CrossingStop]:
    """A stop in ``(t_start, t_end]`` on each breakpoint of a table function
    that is read on the clock, where the table breaks (issue #709).

    A ``tfun`` over time is constant (``step``) or straight (``linear``)
    between its breakpoints, so over a segment that starts and ends at the same
    value the integrator's error estimate sees nothing, and a step can span the
    whole of a dosing pulse: ``tfun([0,10,12,50], [0,5,0,0], time)`` left A + B
    at 3 where the answer is 13. A stop on each such breakpoint puts every
    segment in a step of its own.

    Which breakpoints, and at which of them the integrator restarts, is
    :func:`table_function_breaks`'s answer (*breaks*, or read from *core*). A
    table indexed by an observable that is a unit-rate counter is placed too,
    at the time the counter reaches each breakpoint, as a counter's stop. One
    indexed by anything else moves with the state and gets none.
    """
    if breaks is None:
        breaks = table_function_breaks(core)
    out: list[CrossingStop] = []
    clocks: dict[int, tuple[int, float]] | None = None
    for kind, obs_idx, restarts, ends in breaks:
        if kind == "time":
            for at, restart in ((restarts, True), (ends, False)):
                inside = at[(at > t_start) & (at <= t_end)]
                out.extend(CrossingStop(x, -1, 0.0, restart=restart) for x in inside.tolist())
        elif kind == "observable":
            if clocks is None:
                clocks = _counter_observables(core, t_start)
            counter = clocks.get(obs_idx)
            if counter is None:
                continue
            idx, offset = counter
            # The counter is put on the breakpoint at the stop, which moves the
            # state, so the run restarts there whichever kind of stop it is.
            for x in sorted([*restarts.tolist(), *ends.tolist()]):
                when = x - offset
                if t_start < when <= t_end:
                    out.append(CrossingStop(when, idx, x))
    out.sort(key=lambda stop: stop.time)
    return out


def _counter_observables(core, t_start: float) -> dict[int, tuple[int, float]]:
    """Observable index → ``(species index, value − time)`` for each observable
    that is a unit-rate counter."""
    scope = switch_condition_scope(core)
    names = list(core.observable_names)
    out: dict[int, tuple[int, float]] = {}
    for sym, idx in scope.clocks.items():
        if sym in names:
            offset = float(core.get_concentration(core.species_names[idx])) - t_start
            if math.isfinite(offset):
                out[names.index(sym)] = (idx, offset)
    return out


# One budget of stops for a run, over every condition and every kind of
# resolver (issue #890). Each stop is a restart of the integrator, about 13 µs
# on a small model, and a record on both sides of the binding. A condition past
# it has none of its stops placed, and the run is step-bounded or refused.
_RUN_STOP_BUDGET = 1_000_000
# How many crossings of the periodic conditions are placed in a run; past this a
# condition is left to the integrator as before, with a warning.
_PERIODIC_STOP_BUDGET = _RUN_STOP_BUDGET
_POLY_MAX_DEGREE = 16
# The evaluator's named constants, as the loaders spell them (SBML <pi/> is
# `_pi`), read as numbers when no parameter has the name.
_EVALUATOR_CONSTANTS = {"_pi": math.pi, "pi": math.pi, "_e": math.e, "exponentiale": math.e}
# What a residual of either shape has and a linear one does not: a sine or a
# cosine, a power, or the clock read twice. A condition without one is not
# parsed: sympy costs about a millisecond a condition, on every run.
_PERIODIC_HINT = re.compile(r"(?<![A-Za-z0-9_])(?:sin|cos|pow)\s*\(|\^|\*\*")
# A condition's shape, by its residual text and the values of the parameters it
# reads, which is what a scan or a fit varies; bounded and cleared whole, like
# _CROSSING_CACHE.
_PERIODIC_SHAPES: dict[tuple, tuple | None] = {}
# How far past a crossing its stop lands: a multiple of the evaluator's rounding
# band there, and at most 32 ulps of the time. CVODE takes a stop as reached
# within about 100 ulps short of it, so a jump that close before the stop is
# never stepped across, while one farther before it is approached until the
# step collapses: the stop has to sit just past the evaluator's flip, not past
# its whole band.
_STOP_NUDGE_SAFETY = 4.0
_STOP_NUDGE_MAX_ULPS = 32.0


_NOISY_TRIGGERS_WARNED: set[str] = set()


def _warn_noisy_trigger(cond) -> None:
    """Once per condition: an event trigger whose crossings lie inside the
    evaluator's rounding noise. A sum of terms that nearly cancel at its peak
    (`sin(time) + cos(time) > 1.4142135623`) reads true and false by turns
    across a band ~1e-9 wide, and the event fires as many times as the rounding
    decides (23 for 16, or 0), on any engine that evaluates it in doubles."""
    if cond in _NOISY_TRIGGERS_WARNED:
        return
    _NOISY_TRIGGERS_WARNED.add(cond)
    logger.warning(
        "The event trigger %r crosses its threshold within the rounding of its own "
        "evaluation: near each crossing it reads true and false by turns, and the "
        "event can fire more or fewer times than it crosses. Move the threshold off "
        "the extremum or write the trigger so it does not cancel there.",
        cond,
    )


def _warn_over_budget(cond, n, t_start, t_end) -> None:
    logger.warning(
        "The condition %r crosses about %d times in (%g, %g], past the %d stops "
        "placed in one run; the integrator approaches them unclamped and can step "
        "over a narrow window.",
        cond,
        n,
        t_start,
        t_end,
        _PERIODIC_STOP_BUDGET,
    )


def _periodic_stop_times(
    cond, scope, t_start, t_end, bodies, touches: bool = False, shaped: list | None = None
) -> list[tuple[float, float]]:
    """``(crossing, stop)`` for each crossing in ``(t_start, t_end]`` of one
    condition whose residual is a sum of sinusoids of one frequency
    (``a·sin(ω·t + φ) + b·cos(ω·t + ψ) + c``) or a polynomial in time, or an
    ``if()`` over pieces of those shapes: ``sin(10*time()) > 0.99``,
    ``(time - 3)^2 < 4e-4``. Empty for a condition of any other shape, or that
    reads any state.

    The stop lands a few ulps past its crossing, never on it. On the crossing
    the rounding reads either branch, and where it read the one ending there,
    the restart's first step held the jump within ulps of its start and its
    step size collapsed: at tight tolerances CVODE gave up.

    A touch of the threshold without a crossing (``sin(time) >= 1``, true at one
    instant) has a stop on it only when *touches*, for an event trigger, which
    fires there. To a rate law the instant has no measure, and the evaluator
    reads the touched branch over a band 2e-8 wide around it: a stop in the band
    held a jump of 1e6 within it, where a run with no stop steps over it.
    """
    split = _relational_split(_strip_redundant_parens(cond.strip()))
    if split is None:
        return []
    lhs, rhs = split
    residual = _inline_function_slots(f"({lhs})-({rhs})", bodies)
    flat = _inline_derived_param_refs(residual, scope.derived_exprs) or residual
    n_clock = len(_TIME_REF.findall(flat))
    if n_clock == 0 or (n_clock == 1 and not _PERIODIC_HINT.search(flat)):
        return []
    # Every other name is a parameter (a value for this run), or a call to a
    # builtin; anything else (a species, an observable, a model function) moves.
    probe = _TIME_REF.sub(" 0 ", flat)
    params: set[str] = set()
    constants: set[str] = set()
    for m in _IDENTIFIER.finditer(probe):
        name = m.group(0)
        if probe[m.end() :].lstrip().startswith("("):
            if name in scope.function_names or name in bodies:
                return []
            continue
        if name in _EVALUATOR_CONSTANTS and name not in scope.param_idx:
            constants.add(name)
            continue
        if name not in scope.param_idx or name in bodies:
            return []
        params.add(name)
    key = (flat, tuple(sorted((n, scope.values[scope.param_idx[n]]) for n in params)))
    if key in _PERIODIC_SHAPES:
        shape = _PERIODIC_SHAPES[key]
    else:
        try:
            shape = _periodic_shape(flat, params, constants, scope)
        except Exception as e:  # sympy and mpmath can raise anything on odd input
            logger.debug("periodic crossing stops: %r declined: %s", cond, e)
            shape = None
        if len(_PERIODIC_SHAPES) >= _CROSSING_CACHE_MAX:
            _PERIODIC_SHAPES.clear()
        _PERIODIC_SHAPES[key] = shape
    if shape is None:
        return []
    if shaped is not None:
        # Read, whether or not it crosses in this window: nothing else is asked.
        shaped.append(True)
    return _shape_stop_times(shape, cond, float(t_start), float(t_end), touches)


def _shape_stop_times(
    shape, cond, lo: float, hi: float, touches: bool
) -> list[tuple[float, float]]:
    if shape[0] == "const":
        return []
    if shape[0] == "poly":
        kept = [e for e in shape[1] if lo < e[0] <= hi and (touches or not e[2])]
        if touches and any(e[3] for e in kept):
            _warn_noisy_trigger(cond)
        return [(tc, min(ts, hi)) for tc, ts, _, _ in kept]
    if shape[0] == "piecewise":
        import numpy as np

        out: list[tuple[float, float]] = []
        for i, (_, sub) in enumerate(shape[1]):
            pairs = _shape_stop_times(sub, cond, lo, hi, touches)
            if pairs:
                held = shape[2](np.asarray([tc for tc, _ in pairs], dtype=float)) == i
                out.extend(p for p, keep in zip(pairs, held, strict=False) if keep)
        return sorted(out)
    return _trig_stop_times(shape, cond, lo, hi, touches)


@functools.lru_cache(maxsize=256)
def _parse_periodic_text(text: str):
    """:func:`_parse_clock_expr`, once per text: the parameters stay symbols
    through it, so a scan or a fit that moves their values reuses the parse."""
    return _parse_clock_expr(text)


def _periodic_shape(flat, params, constants, scope) -> tuple | None:
    """The residual's shape: ``("poly", ((crossing, stop), ...))`` over every
    real root of a polynomial; ``("trig", R, ω, φ, c, noise_amp, max_phase)``
    for ``R·sin(ω·t + φ) + c``; ``("const",)``; or ``("piecewise", ((guard,
    shape), ...), branch_of)`` for an ``if()`` over pieces of those shapes whose
    guards read only the time, as an assignment rule ``S1 := if(time <= 2,
    sin(10*time), 1)`` brings into ``S1 < 0`` (SBML suite 00936); else
    ``None``.

    The parameters stay symbols through the parse and take their values after
    it, a polynomial's as exact rationals: parsed as decimals, ``tc^2 - d`` was
    folded in doubles first and its crossing moved by 17 ulps.
    """
    import sympy as sp

    text = _TIME_REF.sub(" bngsim_clock_t ", flat)
    for name in constants:
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_(])",
            f"({_EVALUATOR_CONSTANTS[name]!r})",
            text,
        )
    expr, dealias = _parse_periodic_text(text)
    t = sp.Symbol("bngsim_clock_t")
    if t not in expr.free_symbols:
        return None
    values: dict = {}
    for sym in expr.free_symbols - {t}:
        name = dealias(sym.name)
        if name not in params:
            return None
        values[sym] = scope.values[scope.param_idx[name]]
    if not expr.has(sp.Piecewise):
        return _shape_of(expr, t, values)
    # Each branch a shape of its own, kept where its guard selects it: the
    # crossings at a guard's own edges are the guard's, a condition of its own.
    folded = sp.piecewise_fold(expr)
    if not isinstance(folded, sp.Piecewise):
        return None
    pieces = []
    for value, guard in folded.args:
        if value.has(sp.Piecewise) or not guard.free_symbols <= {t} | set(values):
            return None
        sub = _shape_of(value, t, values)
        if sub is None:
            return None
        pieces.append((guard, sub))
    floats = {s: sp.Float(v) for s, v in values.items()}
    guards = [sp.lambdify(t, guard.subs(floats), "numpy") for guard, _ in pieces]

    def branch_of(times):
        import numpy as np

        out = np.full(times.shape, -1)
        for i, holds in enumerate(guards):
            out[(out < 0) & np.broadcast_to(np.asarray(holds(times), dtype=bool), times.shape)] = i
        return out

    return ("piecewise", tuple(pieces), branch_of)


def _shape_of(expr, t, values) -> tuple | None:
    import sympy as sp

    numeric = expr.subs({s: sp.Float(v) for s, v in values.items()}) if values else expr
    if not numeric.has(t):
        return ("const",)
    try:
        sp.Poly(numeric, t)
    except sp.PolynomialError:
        return _trig_shape(numeric, t)
    exact = expr.xreplace({s: sp.Rational(v) for s, v in values.items()})
    exact = exact.xreplace({f: sp.Rational(f) for f in exact.atoms(sp.Float)})
    return _poly_shape(exact, numeric, t)


def _trig_shape(numeric, t) -> tuple | None:
    """``("trig", R, ω, φ, c, noise_amp, max_phase)`` for a residual that is a
    sum of sines and cosines of one ``ω·t + φ_i`` plus a constant, folded to
    ``R·sin(ω·t + φ) + c`` by angle addition; else ``None``."""
    import sympy as sp

    atoms = list(numeric.atoms(sp.sin, sp.cos))
    if not atoms:
        return None
    s_sym, c_sym = sp.Dummy("s"), sp.Dummy("c")
    slots = [sp.Dummy(f"f{i}") for i in range(len(atoms))]
    w = None
    phases = []
    repl = {}
    for f, slot in zip(atoms, slots, strict=False):
        try:
            arg = sp.Poly(sp.expand(f.args[0]), t)
        except sp.PolynomialError:
            return None
        if arg.degree() != 1:
            return None
        wi, pi = (float(x) for x in arg.all_coeffs())
        if w is None:
            w = wi
        elif abs(wi - w) > 4.0 * _EPS * abs(w):
            return None  # two frequencies
        cp, sn = math.cos(pi), math.sin(pi)
        # sin(ωt + φ) = sin ωt·cos φ + cos ωt·sin φ; cos(ωt + φ) = cos ωt·cos φ − sin ωt·sin φ.
        repl[slot] = s_sym * cp + c_sym * sn if isinstance(f, sp.sin) else c_sym * cp - s_sym * sn
        phases.append(abs(pi))
    held = numeric.xreplace(dict(zip(atoms, slots, strict=False)))
    if held.has(t):
        return None  # the clock outside a sine
    try:
        lin = sp.Poly(sp.expand(held.xreplace(repl)), s_sym, c_sym)
        # Each term's own coefficient, which its rounding scales with.
        noise_amp = sum(abs(float(sp.diff(held, slot))) for slot in slots)
    except (sp.PolynomialError, TypeError):
        return None
    if lin.total_degree() != 1:
        return None
    a = float(lin.coeff_monomial(s_sym))
    b = float(lin.coeff_monomial(c_sym))
    c = float(lin.coeff_monomial(1))
    r = math.hypot(a, b)
    if w is None or not all(math.isfinite(x) for x in (r, w, c, float(noise_amp))):
        return None
    if r == 0.0 or w == 0.0:
        return ("const",)
    # a·sin ωt + b·cos ωt = R·sin(ωt + φ) with φ = atan2(b, a).
    return ("trig", r, w, math.atan2(b, a), c, max(noise_amp, r), max(phases))


def _poly_shape(exact, numeric, t) -> tuple | None:
    """Every real root of a polynomial residual, as ``(crossing, stop)``.

    The roots are those of the polynomial with exact coefficients: expanded in
    doubles, ``(time - 50)^8 - 1e-3`` loses its 1e-3 to the 3.9e13 beside it,
    and the roots came back at 49.145 and 50.864 for 49.578 and 50.422. They
    are found per squarefree factor, at a working precision the coefficients
    cannot exhaust; sympy's exact isolation took 77 s at degree 13.

    A root of even multiplicity touches zero without a sign change; its stop is
    on it, and marked a touch (see :func:`_periodic_stop_times`). One of odd
    multiplicity m leaves the evaluator's rounding band ``e`` about
    ``(e/|c_m|)^(1/m)`` past it, ``c_m`` the m-th Taylor coefficient there.
    """
    import mpmath
    import sympy as sp

    poly = sp.Poly(exact, t)
    if not 1 <= poly.degree() <= _POLY_MAX_DEGREE:
        return None
    if not (poly.domain.is_QQ or poly.domain.is_ZZ):
        # An irrational coefficient (``sqrt(2)``): to 60 digits, as a rational.
        poly = sp.Poly([sp.Rational(sp.N(k, 60)) for k in poly.all_coeffs()], t, domain=sp.QQ)
    roots: list[tuple[float, int]] = []
    for factor, mult in poly.sqf_list()[1]:
        coeffs = [sp.Rational(k) for k in factor.all_coeffs()]
        if len(coeffs) < 2:
            continue
        digits = 30 + max(len(str(abs(k.p))) + len(str(k.q)) for k in coeffs)
        with mpmath.workdps(digits):
            mp = [mpmath.mpf(k.p) / k.q for k in coeffs]
            if len(mp) == 2:
                found = [-mp[1] / mp[0]]
            else:
                found = mpmath.polyroots(mp, maxsteps=400, extraprec=2 * digits)
            tol = mpmath.mpf(10) ** (-(digits // 2))
            for z in found:
                z = mpmath.mpc(z)
                if abs(z.imag) <= tol * max(1, abs(z)):
                    roots.append((float(z.real), mult))
    roots.sort()
    out = []
    for i, (tr, m) in enumerate(roots):
        if m % 2 == 0:
            out.append((tr, tr, True, False))
            continue
        gap = min(
            (abs(tr - roots[j][0]) for j in (i - 1, i + 1) if 0 <= j < len(roots)),
            default=math.inf,
        )
        coeff = abs(float(poly.diff((t, m)).eval(sp.Rational(tr)))) / math.factorial(m)
        err = _rounding_bound(numeric, t, tr)
        nudge = (
            (_STOP_NUDGE_SAFETY * err / coeff) ** (1.0 / m)
            if coeff > 0.0 and math.isfinite(err)
            else math.inf
        )
        # Ulps of the time itself: CVODE's reach for a stop shrinks with t, and
        # an absolute floor of eps put a stop near t = 0.01 2048 ulps past.
        ulp = math.ulp(tr)
        noisy = nudge > _STOP_NUDGE_MAX_ULPS * ulp
        out.append(
            (tr, tr + min(max(nudge, ulp), _STOP_NUDGE_MAX_ULPS * ulp, gap / 4.0), False, noisy)
        )
    return ("poly", tuple(out))


def _rounding_bound(expr, t, tv: float) -> float:
    """A bound on the rounding error of *expr* evaluated in doubles at
    ``t = tv``, by the usual forward analysis: a sum adds its terms' errors and
    a rounding of each partial sum, a product its factors' relative errors, and
    ``sin`` and ``cos`` pass their argument's on. ``inf`` past any other node."""
    import sympy as sp

    def walk(e) -> tuple[float, float]:
        if e == t:
            return tv, 0.0
        if e.is_Number:
            v = float(e)
            return v, _EPS * abs(v)
        if isinstance(e, sp.Add):
            parts = [walk(a) for a in e.args]
            err = sum(ei for _, ei in parts) + _EPS * len(parts) * sum(abs(x) for x, _ in parts)
            return math.fsum(x for x, _ in parts), err
        if isinstance(e, sp.Mul):
            parts = [walk(a) for a in e.args]
            v = math.prod(x for x, _ in parts)
            err = _EPS * len(parts) * abs(v)
            for i, (_, ei) in enumerate(parts):
                err += ei * math.prod(abs(x) for j, (x, _) in enumerate(parts) if j != i)
            return v, err
        if isinstance(e, sp.Pow) and e.exp.is_Integer:
            b, eb = walk(e.base)
            n = int(e.exp)
            if b == 0.0 and n < 1:
                return math.nan, math.inf
            v = b**n
            return v, abs(n) * abs(b) ** (n - 1) * eb + 2.0 * _EPS * abs(v)
        if isinstance(e, (sp.sin, sp.cos)):
            x, ex = walk(e.args[0])
            return (math.sin(x) if isinstance(e, sp.sin) else math.cos(x)), ex + _EPS
        return math.nan, math.inf

    return walk(expr)[1]


def _trig_stop_times(
    shape, cond, lo: float, hi: float, touches: bool
) -> list[tuple[float, float]]:
    """``(crossing, stop)`` for ``R·sin(ω·t + φ) + c`` in ``(lo, hi]``."""
    import numpy as np

    _, r, w, ph, c, noise_amp, max_phase = shape
    v = -c / r
    if abs(v) > 1.0 or (abs(v) == 1.0 and not touches):
        return []  # never reaches its threshold, or only touches it
    bases = sorted({math.asin(v), math.pi - math.asin(v)})
    two_pi = 2.0 * math.pi
    # Counted before any is placed: sin(1e6*time) over a long run has more
    # crossings than memory.
    spans = []
    total = 0
    for th in bases:
        ends = sorted(((w * lo + ph - th) / two_pi, (w * hi + ph - th) / two_pi))
        if not all(math.isfinite(e) for e in ends):
            return []
        k0, k1 = math.floor(ends[0]), math.ceil(ends[1])
        spans.append((th, k0, k1))
        total += k1 - k0 - 1
    if total > _RUN_STOP_BUDGET:
        # The two stretches a level cuts a period into; the shorter is what a
        # step must not span. (This warned and returned no stops, and the run
        # stepped over every window: 2e-5 for 80 with sin(1e5*time()) > 0.5.)
        up = (math.pi - 2.0 * math.asin(v)) / abs(w)
        raise _OverStopBudget(total, min(up, two_pi / abs(w) - up) or two_pi / abs(w))
    times = np.sort(
        np.concatenate(
            [(th + two_pi * np.arange(k0, k1 + 1, dtype=float) - ph) / w for th, k0, k1 in spans]
        )
    )
    times = times[(times > lo) & (times <= hi)]
    # The rounding band: the rounding of the argument (about |ω·t| + |φ|), of
    # each sine and of the sum, over the slope R·|ω|·√(1 − v²) every crossing
    # shares. The nudge is never past a quarter of the way to the next crossing.
    s = math.sqrt(1.0 - v * v)
    if s == 0.0:
        # A touch: the stop is on it. Both bases are the one extremum then.
        return [(float(x), float(x)) for x in np.unique(times)]
    err = _EPS * (noise_amp * (2.0 * (np.abs(w * times) + max_phase) + 2.0) + abs(c))
    band = _STOP_NUDGE_SAFETY * err / (r * abs(w) * s)
    sep = (bases[1] - bases[0]) % two_pi
    gap = min(sep, two_pi - sep) / abs(w)
    ulp = np.spacing(np.abs(times))
    if touches and noise_amp > r * (1.0 + 1e-12) and np.any(band > _STOP_NUDGE_MAX_ULPS * ulp):
        _warn_noisy_trigger(cond)
    nudge = np.minimum(np.maximum(band, ulp), _STOP_NUDGE_MAX_ULPS * ulp)
    nudge = np.minimum(nudge, gap / 4.0)
    return [
        (float(a), float(b)) for a, b in zip(times, np.minimum(times + nudge, hi), strict=False)
    ]


def merge_crossing_stops(found: Sequence[CrossingStop], floor: float = 1.0) -> list[CrossingStop]:
    """Merge the time-ordered stops that land on one instant.

    In time order each needs only the last one kept: comparing each against
    every one kept was quadratic, and a step of time can put thousands of stops
    in a window (issue #869). On one instant a touch of a threshold wins, then a
    counter's stop, then, of two on literal time, the later; of a stop moved
    onto its condition's change (issue #891) and one on the same double, the
    one left in place; and where a moved stop is beside one on another double,
    every stop goes back to where the resolvers put it. The merged stop
    restarts the integrator if any of them does.

    *floor* as for :func:`_same_instant`. With 1 under a plain run, an event
    at t = 0.013 lost its stop to another's 1,065 doubles later, which the
    loop does not take for the same instant; the event was left to the root
    search and ran after a delayed execution due 65 doubles after it.
    """
    # Issue #891: a stop moved onto its condition's change beside another on a
    # double or two from it is one the run cannot take apart from it (it takes
    # the second as reached where it restarts on the first), and no one stop is
    # right for both conditions: folded onto the later, the step arriving
    # there crossed the earlier change and read its new branch (S 6.8e-6 off
    # in a slide model); onto the earlier, the law that changes later
    # restarted a double before its own change (`time() - P*floor(time()/P)`
    # beside `mod(time(), P)`: 305 times rtol, main 33); onto a moved one, an
    # event trigger's stop was read a double late. Where any is, every stop
    # the resolvers placed goes back to where they put it. (A scanned stop has
    # no placing of theirs, and stays a double before its change where it was
    # put there, issue #1226.)
    if any(stop.on_flip for stop in found) and any(
        (a.on_flip or b.on_flip) and a.time != b.time and _same_instant(a.time, b.time, floor)
        for a, b in zip(found, found[1:], strict=False)
    ):
        found = sorted(
            (
                stop._replace(
                    time=stop.placed if stop.on_flip and stop.placed is not None else stop.time,
                    on_flip=False,
                )
                for stop in found
            ),
            key=lambda stop: stop.time,
        )
    out: list[CrossingStop] = []
    for stop in found:
        if not out or not _same_instant(stop.time, out[-1].time, floor):
            out.append(stop)
            continue
        # A restart that any stop on the instant asks for is kept.
        if out[-1].restart and not stop.restart:
            stop = stop._replace(restart=True)
        elif stop.restart and not out[-1].restart:
            out[-1] = out[-1]._replace(restart=True)
        if stop.touch or out[-1].touch:
            # A touch of a threshold is true at one double alone: the stop on it
            # is never merged away (issue #714).
            if stop.touch and not out[-1].touch:
                out[-1] = stop
        elif stop.clock_species_idx >= 0 and out[-1].clock_species_idx < 0:
            # Two conditions crossing at one instant, one on a counter and
            # one on literal time. Keep the counter record: it does
            # everything the plain one does and also lands the counter on
            # its threshold, which the plain one would leave a couple of ulp
            # short and the counter's own condition reading false.
            out[-1] = stop
        elif stop.clock_species_idx < 0 and out[-1].clock_species_idx < 0:
            if stop.on_flip or out[-1].on_flip:
                # On one double (crowded ones are main's, above): one stop, the
                # one left in place if one is.
                if not stop.on_flip:
                    out[-1] = stop
                continue
            # Two on literal time: the later one. A periodic stop lands past
            # its crossing on purpose (issue #714), and a linear crossing at
            # the same instant is behind it too; the earlier one left the
            # periodic condition reading the branch that ends there.
            out[-1] = stop
    return out


# A clock reference anywhere in a text, read generously: `2time()` (ExprTk's
# 2*time()) included, and an identifier that merely ends in `time` after a digit
# too. Taking too much for a clock only costs speed below.
_ANY_CLOCK = re.compile(r"(?<![A-Za-z_])time(?![A-Za-z0-9_])")


def _condition_resolves_exactly(core, cond, scope, t_start, t_end, bodies) -> bool:
    """True when *cond* holds one truth value between consecutive breakpoints
    of this window, because the one time it flips is one of them.

    Stricter than :func:`fixed_crossing_stops`, whose stops cost nothing when
    wrong: here a wrong one would freeze a rate. Only a comparison of the bare
    clock with a threshold that reads no clock, on literal time (a counter
    crosses when something fires, not at a time), whose crossing
    :func:`_crossing_time_of_condition` solves: ``time() > 10``,
    ``time() < t_on + 0.5``. A schedule or a step call (``mod``, ``floor``) is
    left a function of time: between its listed edges it need not be constant
    (a sawtooth), and the resolvers do not promise every edge.
    """
    rewrite = _rewrite_counter_clock(core, cond, scope, t_start)
    if rewrite is None:
        return False
    text, clock_idx, _ = rewrite
    if clock_idx >= 0:
        return False
    split = _relational_split(_strip_redundant_parens(text.strip()))
    if split is None:
        return False
    lhs, rhs = (_strip_redundant_parens(x.strip()) for x in split)
    for clock, other in ((lhs, rhs), (rhs, lhs)):
        if not re.fullmatch(r"time\s*\(\s*\)", clock):
            continue
        # The threshold as the model evaluates it: a function slot or derived
        # parameter that reads the clock (`T := time > 0.5`) moves it.
        flat = _inline_function_slots(other, bodies)
        flat = _inline_derived_param_refs(flat, scope.derived_exprs) or flat
        if _ANY_CLOCK.search(flat) or any(
            m.group(0) in bodies or m.group(0) in scope.function_names
            for m in _IDENTIFIER.finditer(flat)
        ):
            return False
        return _crossing_time_of_condition(text, scope, t_start, t_end, bodies) is not None
    return False


# What may stand next to a whole condition in an expression: a bracket, a comma,
# a logical operator, or the text's end. `time()>5` is not a whole condition in
# `time()>5*A`.
_ATOM_LEFT = re.compile(r"(?:^|[(,!&|]|\b(?:and|or|not))\s*$")
_ATOM_RIGHT = re.compile(r"^\s*(?:$|[),&|]|(?:and|or)\b)")


def _remove_whole_atoms(expr: str, atom: str) -> str:
    """*expr* with each occurrence of *atom* that stands as a whole condition
    replaced by `` 0 ``; an occurrence that is part of a larger operand stays."""
    out, i = [], 0
    while True:
        j = expr.find(atom, i)
        if j < 0:
            out.append(expr[i:])
            return "".join(out)
        k = j + len(atom)
        if _ATOM_LEFT.search(expr[:j]) and _ATOM_RIGHT.match(expr[k:]):
            out.append(expr[i:j] + " 0 ")
        else:
            out.append(expr[i:k])
        i = k


def piecewise_constant_time_functions(
    core, t_start: float, t_end: float, conditions=(), clock_functions=None
) -> list[str]:
    """Functions that read the clock only inside conditions resolved exactly over
    ``(t_start, t_end]`` (:func:`_condition_resolves_exactly`), so are constant
    between the window's breakpoints. The SSA holds a rate that reads the clock
    only through these constant and re-reads it at each breakpoint, rather than
    integrating it (issue #719 follow-up): ``if((time()>10) && (time()<150),
    2.6, 0)`` is a step, not a curve.

    A function qualifies when removing every exactly resolved condition that
    stands as a whole condition in its text leaves no clock reference.
    Anything left over (a clock in arithmetic, a condition no resolver placed
    exactly, a condition that is part of a larger operand) keeps it a function
    of time. ``clock_functions`` is ``[(name, expression)]`` for the functions
    whose text reads the clock, when the caller has it.
    """
    if not conditions:
        return []
    if clock_functions is None:
        clock_functions = [
            (f["name"], f["expression"])
            for f in core.codegen_data()["functions"]
            if _ANY_CLOCK.search(f["expression"])
        ]
    if not clock_functions:
        return []
    ctx = core.functional_jacobian_context()
    scope = switch_condition_scope(core, ctx)
    bodies = _function_slot_bodies(ctx)
    exact = sorted(
        (
            c
            for c in conditions
            if _condition_resolves_exactly(core, c, scope, t_start, t_end, bodies)
        ),
        key=len,
        reverse=True,
    )
    if not exact:
        return []
    out = []
    for name, expr in clock_functions:
        # Spelled as the recovered conditions are (issue #1127), or a condition
        # found in a `%` text could not be found in it again.
        rest = _spell_mod(expr)
        for c in exact:
            rest = _remove_whole_atoms(rest, c)
        if not _ANY_CLOCK.search(rest):
            out.append(name)
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
        # A threshold written through clock guards (issue #545) is fixed when every
        # cell's threshold is.
        guarded = _clock_guarded_thresholds(atom, scope.clock_symbols)
        if guarded is not None:
            return all(_fixed_threshold_expr(g.threshold, scope) for g in guarded[1])
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


# Builtins whose value jumps, where their argument crosses an integer (the
# rounding family) or zero (``sign``), or at a modulus boundary. The analytic
# sensitivity RHS declines every one of them, so a model that calls one on a
# moving argument is on CVODES' difference quotient with that jump in it.
# ``abs``/``min``/``max`` are not here: they bend ``f`` without breaking it, and
# the difference quotient integrates through a kink correctly.
# A condition that is a sawtooth call, whole (`mod(1.1e12 + time(), P)`): the
# shape :func:`_left_alone_tread_stops` reads, asked of a resolver's "no change"
# before the condition is made for the evaluator.
_SAWTOOTH_CALL = re.compile(r"\s*(?:mod|frac)\s*\(")


# The calls whose value holds between two of its changes.
_HELD_CALLS = frozenset(
    {"floor", "ceil", "round", "roundn", "rint", "nint", "trunc", "sign", "sgn"}
)


def _held_between_changes(cond: str) -> bool:
    """Whether *cond* holds its value between two of its changes: a comparison
    (or a combination of them), or a whole step call. Not a sawtooth (`mod`,
    `frac`), which climbs between its wraps, nor a bend. Read at the top
    level only: a comparison in parentheses, as every one an SBML model loads
    is, is not taken to, and keeps its stops where they were put."""
    text = cond.strip()
    depth = 0
    for i, ch in enumerate(text):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth == 0 and ch in "<>" or (depth == 0 and text[i : i + 2] in ("==", "!=")):
            return True
    name, paren, _ = text.partition("(")
    if not paren or name.strip() not in _HELD_CALLS:
        return False
    depth = 0
    for i in range(len(name), len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return i == len(text) - 1
    return False


def _a_whole_sawtooth_call(text: str) -> bool:
    """Whether *text* is one `mod` or `frac` call, and not a comparison of
    one (`mod(time(), 24) < 12`)."""
    head = _SAWTOOTH_CALL.match(text)
    if head is None:
        return False
    depth = 0
    body = text.rstrip()
    for i in range(head.end() - 1, len(body)):
        depth += {"(": 1, ")": -1}.get(body[i], 0)
        if depth == 0:
            return i == len(body) - 1
    return False


_STEP_CALL = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?:floor|ceil|round|roundn|rint|nint|trunc|frac|sign|sgn|mod|fmod|rem|iclamp|inrange)"
    r"\s*\("
)


def _scan_function_map(ctx) -> dict:
    """``ctx["function_map"]`` with each body as the scanners read it (:func:`_spell_mod`)."""
    return {
        name: _spell_mod(body) if isinstance(body, str) else body
        for name, body in dict(ctx["function_map"]).items()
    }


def _scan_rate_law(rxn) -> str:
    """A functional reaction's rate law as the scanners read it (:func:`_spell_mod`)."""
    return _spell_mod(str(rxn.get("rate_expr", "")))


# Calls that bend rather than jump: continuous, with a slope that is not. A
# pulse written with them alone, `max(0, 1 - abs(time()-t0)/w)`, is flat on
# both sides of itself, and a step spans it as it would a window (issue #904).
_BEND_CALL = re.compile(r"(?<![A-Za-z0-9_.])(?:abs|min|max|clamp)\s*\(")


def _turn_atoms(call: str) -> list[str]:
    """Comparisons whose residuals change sign where a step or bend *call* jumps or bends.

    A plain run roots on them (issue #1068). ``floor((Tobs-50)/100)`` over a
    counter that is not a clock has jumps whose times the state decides, so no
    stop can be placed at them, and between the jumps the rate law is flat:
    one step spanned a pulse written as a difference of two floors and Y came
    out 0. The rounding family jumps where its argument crosses an integer,
    which is where ``sin(pi*u)`` changes sign (``cos(pi*u)`` for the
    half-integers ``round`` jumps at); ``mod(a, b)`` where ``a/b`` does. A bend
    turns where its arguments meet. Like the edges of #904, a residual that
    passes two jumps inside one step shows one sign at both ends, so a step
    that spans two jumps still misses them. ``[]`` for a call of an arity
    nothing here knows.
    """
    name, _, rest = call.partition("(")
    name = name.strip()
    args = [a.strip() for a in _split_top_level_commas(rest[:-1])]
    pi = repr(math.pi)
    if len(args) == 1:
        (u,) = args
        if name in ("floor", "ceil", "trunc", "frac"):
            return [f"sin({pi}*({u}))<0"]
        if name in ("round", "rint", "nint"):
            return [f"cos({pi}*({u}))<0"]
        if name in ("sign", "sgn", "abs"):
            return [f"({u})<0"]
    if len(args) == 2:
        a, b = args
        if name in ("mod", "fmod", "rem"):
            return [f"sin({pi}*({a})/({b}))<0"]
        if name == "roundn":
            return [f"cos({pi}*({a})*10^({b}))<0"]
    if len(args) == 3:
        lo, x, hi = args
        if name in ("inrange", "clamp"):
            return [f"({x})<({lo})", f"({x})<({hi})"]
        if name == "iclamp":
            # It bends at both bounds and jumps at the midpoint between them.
            return [f"({x})<({lo})", f"({x})<({hi})", f"({x})<(({lo})+({hi}))/2"]
    if name in ("min", "max") and len(args) >= 2:
        return [f"({a})<({b})" for i, a in enumerate(args) for b in args[i + 1 :]]
    return []


def _bend_branches(atom: str) -> list[str]:
    """*atom* with each bend call in it written as one of its branches, one atom per choice.

    ``0 < 1 - abs(T-50.005)/0.005``, the turn of a triangle pulse written with
    ``max()``, rises and falls back inside the step that spans the pulse, so a
    root on it saw one sign at both ends (issue #1068); so does
    ``0 < min((T-50)/w, (50.01-T)/w)``. ``abs(u)`` is ``u`` or ``-u``, and
    ``min``, ``max`` and ``clamp`` are one of their arguments. The branches
    are monotone where their arguments are, and between them they cross at
    both of the pulse's feet. A branch also crosses where it is not the one
    the call takes, which is a restart where nothing switches. ``[]`` with no
    bend to unfold, with one inside another, and past four branches.
    """
    calls: list[tuple[int, int, list[str]]] = []
    for m in _BEND_CALL.finditer(atom):
        close = _find_close_paren_strict(atom, m.end() - 1)
        if close < 0 or (calls and m.start() < calls[-1][1]):
            return []  # unbalanced, or a bend inside another one
        inner = atom[m.end() : close]
        if m.group().startswith("abs"):
            branches = [f"({inner})", f"(-({inner}))"]
        else:
            branches = [f"({a.strip()})" for a in _split_top_level_commas(inner)]
        calls.append((m.start(), close + 1, branches))
    if not calls or math.prod(len(b) for _, _, b in calls) > 4:
        return []
    out = [""]
    cursor = 0
    for start, end, branches in calls:
        out = [f"{piece}{atom[cursor:start]}{b}" for piece in out for b in branches]
        cursor = end
    return [piece + atom[cursor:] for piece in out]


def _call_turn_pieces(call: str) -> list[str]:
    """The roots that stand for a step or bend *call*: each of its
    :func:`_turn_atoms`, cut at its own edges or unfolded into its bend's
    branches (issue #1068). What :func:`state_switch_root_conditions` roots
    and :func:`state_switch_reactions` reads, so the two cannot drift apart."""
    return [
        piece
        for atom in _turn_atoms(call)
        for piece in (_monotone_pieces(atom) or _bend_branches(atom) or [atom])
    ]


def _iter_step_calls(
    expr: str,
    outermost: bool = False,
    pattern: re.Pattern | None = None,
    skip: Sequence[tuple[int, int]] = (),
    with_start: bool = False,
):
    """``(call, argument)`` for every :data:`_STEP_CALL` outside an ``if()`` condition,
    and the call's offset in *expr* after them with *with_start*.

    One inside a condition is that condition's business: its atom is scanned
    whole, floor and all, so reporting the call again would only repeat it.

    With *outermost*, a call inside another call's argument is not reported
    either: it is part of that argument, which the caller has in hand.

    *pattern* reads another family of calls the same way (:data:`_BEND_CALL`).
    *skip* are further spans of *expr* whose calls are not reported: the
    comparisons outside every ``if()`` that are conditions in their own right.
    """
    spans = _SpanIndex([*_condition_spans(expr), *skip])
    covered = -1  # the close paren of the last call reported, under *outermost*
    for m in (pattern or _STEP_CALL).finditer(expr):
        if m.start() < covered:
            continue
        if spans.covers(m.start()):
            continue
        open_paren = m.end() - 1
        close_paren = _find_close_paren_strict(expr, open_paren)
        if close_paren < 0:
            continue
        if outermost:
            covered = close_paren
        if with_start:
            yield expr[m.start() : close_paren + 1], expr[open_paren + 1 : close_paren], m.start()
        else:
            yield expr[m.start() : close_paren + 1], expr[open_paren + 1 : close_paren]


_PAREN_OR_COMMA = re.compile(r"[(),]")


class _BranchGuards:
    """Every ``if()`` of a text, read once, so :func:`_branch_guard` can be
    asked about any offset of it.

    :func:`state_switch_root_conditions` asks for the guard of every step or
    bend call in a rate law, and each question used to match every ``if()``
    before the call to its close paren and split its arguments again, the
    ones holding the call included: the product of the two counts, on every
    plain run. One ``if()`` holding 2,000 ``max()`` calls, read whole for each
    of them, spent 10.7 s there.

    One pass over the parentheses and commas matches each ``if(`` to its close
    paren and finds the commas between its arguments, where
    ``_find_close_paren_strict`` and ``_split_top_level_commas`` find them, and
    an ``if(`` left open is no ``if()`` at all, as there. Matched parens nest,
    so a sweep over the ``if()``s in order keeps, for each stretch between two
    edges of their argument lists, the innermost one holding it. The ``if()``s
    enclosing an offset are then that one and the ones holding it, found by
    bisection and read innermost first.
    """

    __slots__ = ("_expr", "_edges", "_inner", "_ifs")

    def __init__(self, expr: str) -> None:
        self._expr = expr
        # Each matched if(): its open paren, its close paren, its top-level commas.
        found: list[tuple[int, int, list[int]]] = []
        if_opens = {m.end() - 1 for m in _IF_CALL.finditer(expr)}
        if if_opens:
            groups: list[tuple[int, list[int] | None]] = []  # the open parens
            for m in _PAREN_OR_COMMA.finditer(expr):
                c = m.group()
                if c == "(":
                    groups.append((m.start(), [] if m.start() in if_opens else None))
                elif not groups:
                    continue  # a comma or close paren outside every group
                elif c == ",":
                    commas = groups[-1][1]
                    if commas is not None:
                        commas.append(m.start())
                else:
                    open_paren, commas = groups.pop()
                    if commas is not None:
                        found.append((open_paren, m.start(), commas))
            found.sort()
        # Per if(): its open paren, where its then- and else-branches start
        # (-1 for one without three arguments), and the if() holding it.
        self._ifs: list[tuple[int, int, int, int]] = []
        # The innermost if() holding each offset from an edge to the next.
        self._edges: list[int] = []
        self._inner: list[int] = []
        holding: list[int] = []

        def leave() -> None:
            self._edges.append(found[holding.pop()][1])
            self._inner.append(holding[-1] if holding else -1)

        for i, (open_paren, _close_paren, commas) in enumerate(found):
            while holding and found[holding[-1]][1] < open_paren:
                leave()
            then_start = else_start = -1
            if len(commas) == 2:
                then_start, else_start = commas[0] + 1, commas[1] + 1
            self._ifs.append((open_paren, then_start, else_start, holding[-1] if holding else -1))
            holding.append(i)
            self._edges.append(open_paren + 1)
            self._inner.append(i)
        while holding:
            leave()

    def at(self, pos: int) -> str | None:
        """:func:`_branch_guard` of offset *pos*."""
        k = bisect.bisect_right(self._edges, pos) - 1
        i = self._inner[k] if k >= 0 else -1
        factors: list[str] = []
        while i >= 0:
            open_paren, then_start, else_start, i = self._ifs[i]
            if then_start < 0:
                return None  # not the form read here: no guard to give
            cond = self._expr[open_paren + 1 : then_start - 1]
            if then_start <= pos < else_start - 1:
                factors.append(f"(({cond})!=0)")
            elif pos >= else_start:
                factors.append(f"(({cond})==0)")
            else:
                return None  # in the condition itself
        return "*".join(reversed(factors)) if factors else None


def _branch_guard(expr: str, pos: int) -> str | None:
    """Whether the ``if()`` branches enclosing offset *pos* of *expr* are on, as
    an expression that reads 1 where they all are and 0 where one is not:
    ``((c)!=0)`` for a then-branch and ``((c)==0)`` for an else-branch,
    multiplied, outermost first. ``None`` where no ``if()`` branch encloses it
    (issue #1147).

    A condition is ExprTk's to read, and ExprTk takes a branch wherever its
    condition is nonzero, not only where it is a comparison's 1: ``if(F, ...)``
    with ``F = 0.3`` runs its then-branch, and ``(c)`` read against ``> 0.5``
    called that branch off and stepped over its kink.

    A caller asking about many offsets of one text builds its
    :class:`_BranchGuards` once."""
    return _BranchGuards(expr).at(pos)


def _fixed_clock_step(arg_flat: str, scope: SwitchConditionScope) -> bool:
    """True when a step call's argument reads a clock and nothing else that moves.

    ``floor(time()/24)`` steps at the same instants in every run, so each step's
    ``∂t*/∂p`` is 0 and the difference quotient misses nothing there, which is
    :func:`fixed_clock_threshold`'s ground for a comparison. A parameter beside
    the clock (``floor(time()/P)``) moves the steps, and so does state.
    """
    for m in _IDENTIFIER.finditer(arg_flat):
        name = m.group(0)
        if name in scope.clock_symbols:
            continue
        if arg_flat[m.end() :].lstrip().startswith("(") and name not in scope.function_names:
            continue
        return False
    return True


def model_moving_crossings(core, ctx=None) -> tuple[str, ...]:
    """Every rate-law branch condition, and every step call, whose crossing *time* moves.

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
    crossing. Three grounds exclude an atom, all borrowed from
    :func:`uncompensated_condition_reason` so the two agree about what is not a
    crossing at all — a comparison naming no symbol (``0>0``, decided at load),
    a :func:`fixed_clock_threshold` (``t<14``, whose ``∂t*/∂p`` is exactly 0),
    and a :func:`condition_cannot_cross` atom (``a<b`` over run-constants, which
    picks its branch before the first step and holds it; issue #824). Everything
    else reads live state, so some θ moves it.

    It follows that an atom here may be one the analytic path *would* have
    compensated — issue #48's clock jump, issue #150's saltation jump. That is
    the point: this is about what happens once the model is on the fallback,
    where neither jump is applied.

    A condition is not the only way ``f`` jumps. ``floor(Atot)`` steps every time
    ``Atot`` crosses an integer, at a time every rate constant moves, and the
    analytic RHS declines it ("unsupported construct: floor()"), so a model that
    calls one on its state is on the difference quotient with those steps in it.
    Without this scan the decline warning called that fallback correct: on
    ``k1*floor(Atot)`` it is 57 % wrong at ``t = 0.5``. Every :data:`_STEP_CALL`
    outside a condition is reported the same way, unless its argument names no
    symbol, only run-constants (:func:`condition_cannot_cross`), or only a clock
    and literals (:func:`_fixed_clock_step`).

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
    func_map = _scan_function_map(ctx)
    texts = [
        *func_map.values(),
        *(_scan_rate_law(r) for r in ctx["functional_reactions"]),
    ]
    conditional = [t for t in texts if has_condition_construct(t)]
    stepped = [t for t in texts if _STEP_CALL.search(t)]
    if not conditional and not stepped:
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
            # A condition over run-constants alone holds one truth value for the
            # whole run, so it has no crossing time to move (issue #824). The
            # warning used to name `a<b` or `(a==b)<1` as a moving crossing
            # whenever something else declined the analytic RHS.
            if condition_cannot_cross(atom_flat, scope):
                continue
            found.append(atom)
    for text in stepped:
        flat = _inline_functions(text, func_map) or text
        for call, arg in _iter_step_calls(flat):
            if call in found:
                continue
            arg_flat = _inline_derived_param_refs(arg, scope.derived_exprs) or arg
            if not _IDENTIFIER.search(arg_flat):
                continue
            if condition_cannot_cross(arg_flat, scope) or _fixed_clock_step(arg_flat, scope):
                continue
            found.append(call)
    return tuple(found)


def model_uncompensated_crossing_reason(
    core, ctx=None, kept: dict | None = None
) -> UncompensatedCrossingReason | None:
    """The first rate-law branch crossing this model leaves uncompensated, or ``None``.

    ``kept`` keeps each law's answer between one scan of a model and the next,
    beside what it went by (:func:`_reason_goes_by`): a batch asks this of
    every row, and a row that changes nothing a condition reads is not
    scanned again.

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
    sensitivity RHS is declined and the run is on CVODES' difference quotient.
    That the jump is applied does not make such a run right: the quotient has
    read across a state crossing before the run gets there, and a run on it is
    refused by :func:`fallback_crossing` where the rate law jumps (issue #938).

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
    func_map = _scan_function_map(ctx)
    # Reaction rate expressions only — the exact texts _functional_dfdp_terms
    # differentiates (frxn["rate_expr"] per functional reaction). A function that
    # no rate law reaches is inert here, so it is never scanned. Inlined BEFORE
    # the condition test and the recognizer, exactly as codegen inlines it: a
    # rate law written as a bare function name — or one nesting the condition a
    # level down — only reveals its ``if()`` after inlining.
    texts = [_scan_rate_law(r) for r in ctx["functional_reactions"]]
    flats = [_inline_functions(t, func_map) or t for t in texts]
    conditional = [f for f in flats if has_condition_construct(f)]
    if not conditional:
        return None
    # Not caught: the gate that calls this refuses a model whose conditions
    # cannot be classified (issue #938), where it used to be run.
    scope = switch_condition_scope(core, ctx)
    for flat in conditional:
        if kept is None:
            reason = uncompensated_condition_reason(flat, scope)
        else:
            key = _reason_goes_by(flat, scope)
            last = kept.get(flat)
            if last is None or last[0] != key:
                last = (key, uncompensated_condition_reason(flat, scope))
                kept[flat] = last
            reason = last[1]
        if reason is not None:
            return reason
    return None


def _reason_goes_by(flat: str, scope: SwitchConditionScope) -> tuple:
    """What :func:`uncompensated_condition_reason` goes by for the rate law
    *flat*, beside its text: the law with its derived parameters written
    out, what each parameter in that is, and which species are clocks."""
    written = _inline_derived_param_refs(flat, scope.derived_exprs)
    names = set(_IDENTIFIER.findall(written)) & scope.param_idx.keys()
    return (
        written,
        tuple((name, float(scope.values[scope.param_idx[name]])) for name in sorted(names)),
        tuple(sorted(scope.clocks.items())),
    )


# ``abs``, ``max`` and ``min``: the law takes one of two expressions, with no
# condition written.
_CHOICE_NAMES = frozenset({"abs", "max", "min"})
_CHOICE_CALL = re.compile(r"(?<![A-Za-z0-9_.])(?:abs|max|min)\s*\(")


def _syntax_tree(expr: str) -> ast.Expression | None:
    """The Python syntax tree of an ExprTk expression, or ``None`` where it
    is not Python once :func:`bngsim._jacobian._preprocess_exprtk` has it."""
    from bngsim._codegen import _PY_KEYWORD_PARAM_NAMES, _alias_keyword_param
    from bngsim._jacobian import _LITERAL_KEYWORDS, _preprocess_exprtk

    try:
        text = _preprocess_exprtk(expr)
        # A parameter named with a Python keyword, ``lambda``, under the alias
        # the symbolic path gives it.
        for name in set(_IDENTIFIER.findall(text)) & (
            _PY_KEYWORD_PARAM_NAMES - _LITERAL_KEYWORDS - {"and", "or", "not"}
        ):
            text = re.sub(rf"\b{re.escape(name)}\b", _alias_keyword_param(name), text)
        return ast.parse(text.strip(), mode="eval")
    except Exception:  # noqa: BLE001 - not an expression this reads
        return None


class _Facts(NamedTuple):
    """What one node of a rate law's syntax tree reads, and what is known of
    its value whatever the state is."""

    state: bool = False  # reads the state, or a counter a requested column moves
    clock: bool = False  # reads literal time or a counter nothing moves
    asked: bool = False  # reads a requested parameter
    choice: bool = False  # holds an abs, max or min that is live
    exotic: bool = False  # holds a live call this does not know to be smooth: clamp
    unknown: bool = False  # holds a live call this does not know at all: atan2
    # Holds one of those with no sum over it, so that it may be 0 where the
    # choice is: ``sqrt(abs(e))``, ``exp(log(abs(e)))``, and not ``1 + abs(e)``.
    bare: bool = False
    positive: bool = False
    nonnegative: bool = False
    negative: bool = False
    # Has a value wherever its operands have: no logarithm of what may be 0,
    # no division by it. ``exp`` of what has none can be 0.
    finite: bool = True
    # What it is, where it reads numbers and parameters alone.
    value: float | None = None
    size: int = 1

    @property
    def live(self) -> bool:
        """Whether a column moves what this reads: the state, or a requested
        parameter, which the quotient moves as it does the state."""
        return self.state or self.asked

    @property
    def zero(self) -> bool:
        return self.value == 0.0


_NO_FACTS = _Facts()
# The two factors of a product are compared, for a square, where each is this small.
_SQUARE_MAX_NODES = 50

# Calls that make no choice between two expressions and take no step:
# analytic wherever they have a value.
_SMOOTH_CALLS = frozenset(
    {"exp", "expm1", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"}
    | {"asinh", "acosh", "atanh", "erf", "erfc", "log", "ln", "log2", "log10", "log1p", "pow"}
    | {"avg", "sum", "mul", "hypot"}
    # The engine's own: Kummer's ratio and the gamma function, analytic where
    # they have a value.
    | {"mratio", "tgamma"}
)
# Of those, the ones with poles: no value is claimed for them.
_POLE_CALLS = frozenset({"mratio", "tgamma"})
_LOG_CALLS = frozenset({"log", "ln", "log2", "log10"})
_STEP_NAMES = frozenset(
    {"floor", "ceil", "round", "roundn", "rint", "nint", "trunc", "frac", "sign", "sgn"}
    | {"mod", "fmod", "rem", "iclamp", "inrange"}
)
_STRUCTURE_CALLS = frozenset({"Piecewise", "And", "Or", "Not", "Eq", "Ne"})
# What a call on numbers alone comes to.
_VALUE_CALLS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "max": max,
    "min": min,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "pow": math.pow,
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
}


def _operands(node: ast.AST) -> list[ast.AST]:
    """The expressions under *node*: a call's arguments and not its name."""
    if isinstance(node, ast.Call):
        return list(node.args)
    return [child for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)]


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _number(node: ast.AST) -> float | None:
    """The value of a numeric literal, signed or not."""
    sign = 1.0
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -sign if isinstance(node.op, ast.USub) else sign
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
        return sign * float(node.value)
    return None


def _value_of(node: ast.AST, operands: Sequence[float | None]) -> float | None:
    """What *node* comes to on operands that are numbers, or ``None`` where
    one is not, or where it has no value."""
    if any(v is None for v in operands):
        return None
    try:
        if isinstance(node, ast.UnaryOp) and len(operands) == 1:
            if isinstance(node.op, ast.USub):
                return -operands[0]  # type: ignore[operator]
            if isinstance(node.op, ast.UAdd):
                return operands[0]
            return None
        if isinstance(node, ast.BinOp) and len(operands) == 2:
            a, b = operands
            if isinstance(node.op, ast.Add):
                out = a + b  # type: ignore[operator]
            elif isinstance(node.op, ast.Sub):
                out = a - b  # type: ignore[operator]
            elif isinstance(node.op, ast.Mult):
                out = a * b  # type: ignore[operator]
            elif isinstance(node.op, ast.Div):
                out = a / b  # type: ignore[operator]
            elif isinstance(node.op, ast.Pow):
                out = math.pow(a, b)  # type: ignore[arg-type]
            else:
                return None
            return float(out)
        call = _VALUE_CALLS.get(_call_name(node))
        if call is not None and operands:
            return float(call(*operands))
    except (ArithmeticError, ValueError, TypeError):
        return None
    return None


def _facts_of(
    node: ast.AST,
    under: list[_Facts],
    values: Mapping[str, float],
    clocks: AbstractSet[str],
    asked: AbstractSet[str],
) -> _Facts:
    """The facts of *node*, from those of its operands, *under*."""

    def valued(value: float, **reads) -> _Facts:
        return _Facts(
            positive=value > 0.0,
            nonnegative=value >= 0.0,
            negative=value < 0.0,
            finite=math.isfinite(value),
            value=value,
            **reads,
        )

    size = 1 + sum(f.size for f in under)
    if isinstance(node, ast.Name):
        if node.id in values:
            return valued(values[node.id], asked=node.id in asked)
        if node.id in clocks:
            return _Facts(clock=True)
        return _Facts(state=True)
    if isinstance(node, ast.Constant):
        number = _number(node)
        return _NO_FACTS if number is None else valued(number)
    state = any(f.state for f in under)
    clock = any(f.clock for f in under)
    reads_asked = any(f.asked for f in under)
    name = _call_name(node)
    if not state and not clock:
        value = _value_of(node, [f.value for f in under])
        if value is not None:
            facts = valued(value, asked=reads_asked, size=size)
            return facts._replace(
                choice=any(f.choice for f in under) or (name in _CHOICE_NAMES and reads_asked),
                exotic=any(f.exotic for f in under),
                unknown=any(f.unknown for f in under),
            )
    positive = nonnegative = negative = False
    finite = all(f.finite for f in under)
    if isinstance(node, ast.UnaryOp) and len(under) == 1:
        (one,) = under
        if isinstance(node.op, ast.USub):
            positive, negative = one.negative, one.positive
        elif isinstance(node.op, ast.UAdd):
            positive, nonnegative, negative = one.positive, one.nonnegative, one.negative
    elif isinstance(node, ast.BinOp) and len(under) == 2:
        left, right = under
        if isinstance(node.op, ast.Add):
            positive = (left.positive and right.nonnegative) or (
                left.nonnegative and right.positive
            )
            nonnegative = left.nonnegative and right.nonnegative
            negative = left.negative and right.negative
        elif isinstance(node.op, ast.Sub):
            positive = (left.positive or left.nonnegative) and right.negative
            negative = left.negative and (right.positive or right.nonnegative)
        elif isinstance(node.op, ast.Mult):
            positive = (left.positive and right.positive) or (left.negative and right.negative)
            negative = (left.positive and right.negative) or (left.negative and right.positive)
            nonnegative = (left.nonnegative and right.nonnegative) or (
                # A square, written as a product.
                left.size == right.size
                and left.size <= _SQUARE_MAX_NODES
                and ast.dump(node.left) == ast.dump(node.right)
            )
        elif isinstance(node.op, ast.Div):
            positive = (left.positive and right.positive) or (left.negative and right.negative)
            negative = (left.positive and right.negative) or (left.negative and right.positive)
            nonnegative = left.nonnegative and right.positive
            finite = finite and (right.positive or right.negative)
        elif isinstance(node.op, ast.Pow):
            power = _number(node.right)
            even = power is not None and power > 0.0 and power % 2.0 == 0.0
            # ``2^(log(0))`` is 0: a positive base is above 0 to a power that
            # has a value.
            positive = left.positive and right.finite
            nonnegative = left.nonnegative or even
            finite = finite and (right.nonnegative or left.positive or left.negative)
        else:
            finite = False
    elif name in ("abs", "sqrt") and len(under) == 1:
        (one,) = under
        nonnegative = True
        positive = one.positive or (name == "abs" and one.negative)
    elif name == "exp" and len(under) == 1:
        # ``exp(log(abs(e)))`` is 0 where e is: above 0 only of what has a value.
        nonnegative = True
        positive = under[0].finite
    elif name == "max" and under:
        positive = any(f.positive for f in under)
        nonnegative = any(f.nonnegative for f in under)
        negative = all(f.negative for f in under)
    elif name == "min" and under:
        positive = all(f.positive for f in under)
        nonnegative = all(f.nonnegative for f in under)
        negative = any(f.negative for f in under)
    elif name == "pow" and len(under) == 2:
        positive = under[0].positive and under[1].finite
        nonnegative = under[0].nonnegative
        finite = finite and (under[1].nonnegative or under[0].positive or under[0].negative)
    elif name in _LOG_CALLS and len(under) == 1:
        finite = finite and under[0].positive
    elif name == "Piecewise" and under:
        # Each argument is a (value, condition) pair, with the value's facts.
        positive = all(f.positive for f in under)
        nonnegative = all(f.nonnegative for f in under)
        negative = all(f.negative for f in under)
    elif isinstance(node, ast.Tuple) and len(under) == 2:
        # A Piecewise pair: what the pair is, is what its value is.
        positive, nonnegative, negative, finite = (
            under[0].positive,
            under[0].nonnegative,
            under[0].negative,
            under[0].finite,
        )
    elif name and (name in _POLE_CALLS or not (name in _SMOOTH_CALLS or name in _STRUCTURE_CALLS)):
        finite = False
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        bare = False
    elif isinstance(node, ast.Tuple):
        bare = bool(under) and under[0].bare  # a Piecewise pair: its value
    elif isinstance(node, (ast.Compare, ast.BoolOp)) or name in ("And", "Or", "Not", "Eq", "Ne"):
        bare = False
    else:
        bare = any(f.bare for f in under)
    facts = _Facts(
        state=state,
        clock=clock,
        asked=reads_asked,
        choice=any(f.choice for f in under),
        exotic=any(f.exotic for f in under),
        unknown=any(f.unknown for f in under),
        bare=bare,
        positive=positive,
        nonnegative=nonnegative or positive,
        negative=negative,
        finite=finite,
        size=size,
    )
    if name in _CHOICE_NAMES and facts.live:
        facts = facts._replace(choice=True, bare=True)
    elif facts.live and _unknown_call(name):
        facts = facts._replace(
            exotic=True, bare=True, unknown=facts.unknown or name not in _BEND_CALLS
        )
    return facts


# A call that is known to be continuous and to bend: asked about as a choice
# is, by what the law divides by. Any other call this does not know could be
# anything, a branch cut (``atan2``) or a pole, and is refused where it is live.
# A clamp bends only with its bounds in order; crossed, it is a step, which
# :func:`_quotient_across_a_choice` names (issue #1227).
_BEND_CALLS = frozenset({"clamp"})


def _unknown_call(name: str) -> bool:
    """Whether *name* is a call this does not know to be smooth, a choice or
    a step: ``clamp``, ``atan2``."""
    return bool(name) and not (
        name in _SMOOTH_CALLS
        or name in _CHOICE_NAMES
        or name in _STEP_NAMES
        or name in _STRUCTURE_CALLS
    )


def _facts_of_tree(
    tree: ast.Expression,
    values: Mapping[str, float],
    clocks: AbstractSet[str],
    asked: AbstractSet[str],
) -> tuple[list[ast.AST], dict[int, _Facts]]:
    """Every node of *tree*, parents before what is under them, and the facts
    of each by its ``id``."""
    order: list[ast.AST] = []
    stack: list[ast.AST] = [tree.body]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(_operands(node))
    facts: dict[int, _Facts] = {}
    for node in reversed(order):
        facts[id(node)] = _facts_of(
            node, [facts[id(child)] for child in _operands(node)], values, clocks, asked
        )
    return order, facts


def _either_way(node: ast.AST) -> list[str]:
    """*node* as text, and a difference with its sides swapped as well:
    ``X − thr`` is 0 where ``thr − X`` is."""
    out = [ast.dump(node)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        out.append(ast.dump(ast.BinOp(left=node.right, op=ast.Sub(), right=node.left)))
    return out


def _power(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """``(base, power)`` of a power, written ``a^b`` or ``pow(a, b)``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return node.left, node.right
    if _call_name(node) == "pow" and len(node.args) == 2:  # type: ignore[attr-defined]
        return node.args[0], node.args[1]  # type: ignore[attr-defined]
    return None


def _factors(node: ast.AST, facts: Mapping[int, _Facts], vanishing: bool = False) -> list[ast.AST]:
    """What *node* is a product of, so that it is 0 where one of them is:
    through a sign, a product, the numerator of a quotient, a square root,
    a power that is known not to be below 0, and each branch of a
    conditional.

    With *vanishing*, what *node* goes to 0 with, no faster than in
    proportion: not through a root or a conditional, and through a power
    only where it is known to be 1 or more. ``e^0`` is 1 where ``e`` is 0,
    and ``sqrt(e)`` leaves 0 with a slope that has no bound."""
    out: list[ast.AST] = []
    stack = [node]
    while stack:
        one = stack.pop()
        power = _power(one)
        if isinstance(one, ast.UnaryOp) and isinstance(one.op, (ast.USub, ast.UAdd)):
            stack.append(one.operand)
        elif isinstance(one, ast.BinOp) and isinstance(one.op, ast.Mult):
            stack.extend((one.left, one.right))
        elif isinstance(one, ast.BinOp) and isinstance(one.op, ast.Div):
            stack.append(one.left)
        elif not vanishing and _call_name(one) == "sqrt" and len(one.args) == 1:  # type: ignore[attr-defined]
            stack.append(one.args[0])  # type: ignore[attr-defined]
        elif not vanishing and _call_name(one) == "Piecewise":
            # 0 where the branch it takes is.
            stack.extend(
                pair.elts[0]
                for pair in one.args  # type: ignore[attr-defined]
                if isinstance(pair, ast.Tuple) and len(pair.elts) == 2
            )
        elif power is not None and (
            (facts[id(power[1])].value or 0.0) >= 1.0
            if vanishing
            else _divides_by_nothing(facts[id(power[1])])
        ):
            stack.append(power[0])
        else:
            out.append(one)
    return out


def _divides_by_nothing(power: _Facts) -> bool:
    """Whether a power with these facts is known not to be below 0: ``x^2``
    and ``x^n`` at an ``n`` of 2, and not ``x^-1`` or ``x^Y``."""
    return power.nonnegative


# A factor is compared with what a choice flips on only where it is this
# small: a sign or a step is written over a difference, not over a page of
# rate law.
_FLIP_MAX_NODES = 200
# How far the quotient moves a parameter: its own size times the root of the
# relative tolerance, a quarter of it at a tolerance of 0.06. And how fine the
# moves tried are, from that down: a comparison that flips and flips back
# inside the quarter is found where one of them lands between the two.
_QUOTIENT_REACH = 0.25
_QUOTIENT_HALVINGS = 12


def _moves_tried(value: float) -> list[float]:
    """The values a requested parameter at *value* is tried at: a quarter of
    itself away and by halves of that down, either way; a quarter where it
    is 0."""
    reach = _QUOTIENT_REACH * (abs(value) if value != 0.0 else 1.0)
    return [
        value + sign * reach / 2.0**halving
        for halving in range(_QUOTIENT_HALVINGS)
        for sign in (1.0, -1.0)
    ]


# What a compiled expression may call: the calls :func:`_value_of` has a value
# for, and the power as it takes it.
_VALUE_GLOBALS: dict = {"__builtins__": {}, "_pow": math.pow, **_VALUE_CALLS}
_VALUE_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}


def _value_code(node: ast.AST, taken: AbstractSet[str] = frozenset()):
    """*node* compiled to what :func:`_facts_of` gives as its value, where it
    is made of numbers, names, signs, sums, products, quotients, powers and
    the calls that have a value, and ``None`` where it holds anything else.

    The sign of an expression over parameters is asked at 24 values of each
    requested parameter in it, and through :func:`_facts_of_tree` that is a
    tuple built for every node each time: 49 passes over 500 nodes for one
    ``max`` in ``synthesis_v3``, 0.9 s after each ``set_param``. Compiled, it is
    the same arithmetic on the same doubles. A name that is not a parameter,
    and anything that has no value, raises where the tree gives ``None``.

    *taken* are the names that have a value where the code will be run. An
    expression that calls one of them, or that has a power while one of them
    is the name the power is called by here, is left to the tree: a parameter
    named ``abs`` or ``_pow`` is a parameter.
    """

    def written(one: ast.AST) -> str:
        if isinstance(one, ast.Name):
            return one.id
        if isinstance(one, ast.Constant) and isinstance(one.value, (bool, int, float)):
            number = float(one.value)
            if math.isnan(number):
                raise ValueError("nan")
            # A literal past a double is written as one: ``inf`` is a name.
            return (
                repr(number) if math.isfinite(number) else ("1e999" if number > 0 else "(-1e999)")
            )
        if isinstance(one, ast.UnaryOp) and isinstance(one.op, (ast.USub, ast.UAdd)):
            return f"({'-' if isinstance(one.op, ast.USub) else '+'}{written(one.operand)})"
        if isinstance(one, ast.BinOp) and type(one.op) in _VALUE_BINOPS:
            left, right = written(one.left), written(one.right)
            op = _VALUE_BINOPS[type(one.op)]
            if op == "**" and "_pow" in taken:
                raise ValueError("_pow")
            return f"_pow({left}, {right})" if op == "**" else f"({left} {op} {right})"
        if _call_name(one) in taken:
            raise ValueError(_call_name(one))
        if _call_name(one) in _VALUE_CALLS and not one.keywords:  # type: ignore[attr-defined]
            return f"{_call_name(one)}({', '.join(written(arg) for arg in one.args)})"  # type: ignore[attr-defined]
        raise ValueError(type(one).__name__)

    try:
        return compile(written(node), "<an expression over parameters>", "eval")
    except (RecursionError, SyntaxError, ValueError, MemoryError):
        return None


def _flips_with_a_parameter(
    node: ast.AST, values: Mapping[str, float], asked: AbstractSet[str]
) -> bool:
    """Whether *node*, an expression over numbers and parameters, is 0 or
    changes sign with a requested parameter in it moved as the quotient may
    move it (:func:`_moves_tried`); or cannot be read."""

    compiled = _value_code(node, values.keys())

    def sign_at(at: Mapping[str, float]) -> int | None:
        if compiled is not None:
            try:
                value = float(eval(compiled, _VALUE_GLOBALS, at))  # noqa: S307 - see _value_code
            except (ArithmeticError, ValueError, TypeError, NameError):
                value = None
        else:
            _order, facts = _facts_of_tree(
                ast.Expression(body=node),  # type: ignore[arg-type]
                at,
                frozenset(),
                frozenset(),
            )
            value = facts[id(node)].value
        if value is None or not math.isfinite(value):
            return None
        return (value > 0.0) - (value < 0.0)

    now = sign_at(values)
    if now is None or now == 0:
        return True
    for name in sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & set(asked)):
        if name not in values:
            return True
        for moved in _moves_tried(values[name]):
            if sign_at({**values, name: moved}) != now:
                return True
    return False


# The least power of a bend the quotient is let read across (see below).
_LEAST_POWER_OF_A_BEND = 1.0
# Calls whose slope has no bound where their argument is 0, and at an
# argument of 1 or −1.
_SHARP_AT_ZERO = frozenset({"sqrt"}) | _LOG_CALLS
_SHARP_AT_ONE = frozenset({"asin", "acos", "acosh", "atanh"})


def _quotient_across_a_choice(
    tree: ast.Expression,
    values: Mapping[str, float],
    clocks: AbstractSet[str],
    asked: AbstractSet[str],
) -> str | None:
    """A division in the rate law *tree* that jumps where an ``abs``, ``max``
    or ``min`` flips, as text, or ``None``.

    Each of the three is continuous: ``max(X, c)`` bends where ``X`` passes
    ``c``, and the difference quotient is right across a bend. A law jumps
    across one by dividing by what is 0 there: ``(X − thr)/abs(X − thr)`` is
    a sign and ``max(X − thr, 0)/(X − thr)`` a step, with nothing in the text
    that says so. What is named is a division, in a law where one of the
    three reads what a column moves, by a product with a factor that is

    - one of the three, or holds one with no sum over it, and is not known
      to be nonzero: ``e/abs(e)``, ``1/max(X − thr, 0)``,
      ``e/exp(log(abs(e)))``. ``v/max(X, 0.01)`` is known nonzero; or
    - a factor of what one of them flips on: ``abs(e)/e``, ``max(e, 0)/e``,
      ``max(e, −e)/e`` and ``abs(u·v)/u``.

    A negative or unknown power is a division by its base, and a
    conditional is 0 where a branch of it is. What is known nonzero is known
    by what a number or a parameter is, and by what ``abs``, ``exp`` of what
    has a value, an even power, a sum and a product keep: nothing is assumed
    of the state, which a concentration below 0 would break.

    Four more are named:

    - a ``clamp`` whose bounds are numbers or parameters with ``lo`` above
      ``hi``, over the state or over a requested parameter that moves its
      argument across ``lo``: a step, not a bend (issue #1227);
    - a power under 1, a root, a logarithm or a ``hypot`` of what holds one
      of the three and is not known to be nonzero, and an inverse sine or
      cosine of what holds one: continuous, and leaving the bend with a
      slope that has no bound;
    - any division by what is not known to be nonzero, in a law that holds a
      call this does not know to be smooth, ``clamp(0, e, 1)/e``;
    - an ``abs``, ``max`` or ``min`` over parameters alone that a requested
      one is within a quarter of itself of flipping, ``min(k1, k2)`` at
      k2 = 1.001·k1: a bend in the parameter, which the quotient straddles
      for the whole run and not for an instant of it.

    A sign written any other way is not found: with the choice inside a sum
    the law divides by, ``e/(abs(e) + 0·X)``, or with the two sides of the
    quotient written differently, ``abs(X − thr)/(X/thr − 1)``. Nor is a
    pole cut off on both sides, ``min(max(k/(X − thr), −5), 5)``.
    """
    order, facts = _facts_of_tree(tree, values, clocks, asked)
    root = facts[id(tree.body)]
    if not root.choice and not root.exotic:
        return None
    if root.unknown:
        for node in order:
            name = _call_name(node)
            if _unknown_call(name) and name not in _BEND_CALLS and facts[id(node)].live:
                return _clipped(node)
    # A clamp bends only with its bounds in order. With lo above hi, ExprTk's
    # clamp(lo, x, hi) is lo below x = lo and hi from there on, a step (issue
    # #1227), and the quotient reads across it as across any jump:
    # ``0.01*kx*clamp(1, (Tobs - 40)/10, 0)`` returned dY/dk_clock = -1.41 for
    # -0.97 at every tolerance. One over the state is named, as a step call is;
    # one over parameters alone where a requested one moves x across lo.
    for node in order:
        if _call_name(node) != "clamp" or not facts[id(node)].live:
            continue
        bounds = node.args  # type: ignore[attr-defined]
        if len(bounds) != 3:
            continue
        lo, x, hi = bounds
        bottom, top = facts[id(lo)].value, facts[id(hi)].value
        if bottom is None or top is None or bottom <= top:
            continue
        over = facts[id(x)]
        if over.state or (
            over.asked
            and not over.clock
            and _flips_with_a_parameter(ast.BinOp(left=x, op=ast.Sub(), right=lo), values, asked)
        ):
            return _clipped(node)

    def small(node: ast.AST) -> bool:
        return facts[id(node)].live and facts[id(node)].size <= _FLIP_MAX_NODES

    # What each live choice flips on: the factors of an ``abs``'s argument,
    # and of one side where the other is 0 or is the first with its sign
    # turned; else the difference of the two sides, either way round.
    flips: set[str] = set()
    # A derived parameter written out is in the law as often as it is read:
    # one law of ``synthesis_v3`` comes to 16,283 nodes with the same ``max``
    # over parameters in it 1,145 times. Each is asked about once.
    asked_already: dict[str, bool] = {}
    for node in order:
        name = _call_name(node)
        known = facts[id(node)]
        if name not in _CHOICE_NAMES or not known.live:
            continue
        args = node.args  # type: ignore[attr-defined]
        pairs: list[tuple[ast.expr, ast.expr | None]]
        if name == "abs":
            pairs = [(args[0], None)] if len(args) == 1 else []
        else:
            pairs = [(a, b) for i, a in enumerate(args) for b in args[i + 1 :]]
        if not known.state and not known.clock:
            # Over parameters alone: a bend the quotient straddles for good.
            written = ast.dump(node)
            found = asked_already.get(written)
            if found is None:
                found = asked_already[written] = any(
                    _flips_with_a_parameter(
                        a if b is None else ast.BinOp(left=a, op=ast.Sub(), right=b),
                        values,
                        asked,
                    )
                    for a, b in pairs
                )
            if found:
                return _clipped(node)
            continue
        for a, b in pairs:
            if b is not None and facts[id(b)].zero:
                b = None
            elif b is not None and facts[id(a)].zero:
                a, b = b, None
            elif b is not None and (
                (isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.USub) and _same(b.operand, a))
                or (
                    isinstance(a, ast.UnaryOp)
                    and isinstance(a.op, ast.USub)
                    and _same(a.operand, b)
                )
            ):
                # ``max(e, −e)`` is ``abs(e)``.
                a, b = (a if isinstance(b, ast.UnaryOp) else b), None
            if b is None:
                for factor in _factors(a, facts):
                    if small(factor):
                        flips.update(_either_way(factor))
            elif facts[id(a)].size + facts[id(b)].size < _FLIP_MAX_NODES:
                flips.update(_either_way(ast.BinOp(left=a, op=ast.Sub(), right=b)))
    for node in order:
        divisor: ast.AST | None = None
        power = _power(node)
        # A power under 1, a root or a logarithm of what holds a choice and
        # may be 0: ``max(thr − X, 0)^0.1`` is continuous and leaves 0 with a
        # slope that has no bound, and the quotient reads it much as it reads
        # a jump. At a relative tolerance of 1e-6 the column of the rate
        # constant was 2.5e-5 off at a power of 0.5, 1.3e-4 at 0.4, 3.8e-3 at
        # 0.3, 1.9e-2 at 0.2 and 10% at 0.1, where the run does not finish at
        # 1e-8; and the column of the threshold was 2.6e-4 off at a power of
        # 0.5, and 5.8e-5 at a tolerance of 1e-8. None of them is let through.
        # The choice may be anywhere in what the power is of:
        # ``(thr − min(X, thr))^0.1`` is the same function.
        name = _call_name(node)
        sharp: list[ast.AST] = []
        if power is not None:
            by = facts[id(power[1])]
            if (
                by.value is None
                or (by.value < _LEAST_POWER_OF_A_BEND and by.value != 0.0)
                or (
                    by.asked
                    and _flips_with_a_parameter(
                        ast.BinOp(
                            left=power[1],  # type: ignore[arg-type]
                            op=ast.Sub(),
                            right=ast.Constant(value=_LEAST_POWER_OF_A_BEND),
                        ),
                        values,
                        asked,
                    )
                )
            ):
                sharp = [power[0]]
        elif name in _SHARP_AT_ZERO or name == "hypot":
            sharp = list(node.args)  # type: ignore[attr-defined]
        if any(
            facts[id(arg)].live and _holds_a_choice(facts[id(arg)]) for arg in sharp
        ) and not any(facts[id(arg)].positive or facts[id(arg)].negative for arg in sharp):
            return _clipped(node)
        if name in _SHARP_AT_ONE and any(
            facts[id(arg)].live and _holds_a_choice(facts[id(arg)])
            for arg in node.args  # type: ignore[attr-defined]
        ):
            return _clipped(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            divisor = node.right
        elif power is not None and not _divides_by_nothing(facts[id(power[1])]):
            divisor = power[0]
        if divisor is None or not facts[id(divisor)].live:
            continue
        if root.exotic and not (facts[id(divisor)].positive or facts[id(divisor)].negative):
            return _clipped(node)
        for factor in _factors(divisor, facts):
            known = facts[id(factor)]
            if not known.live or known.positive or known.negative:
                continue
            if known.bare:
                return _clipped(node)
            if small(factor) and ast.dump(factor) in flips:
                return _clipped(node)
    return None


def _holds_a_choice(known: _Facts) -> bool:
    """Whether what has these facts holds a live ``abs``, ``max`` or ``min``,
    or a call that bends as they do, anywhere in it."""
    return known.choice or known.exotic


def _same(a: ast.AST, b: ast.AST) -> bool:
    """Whether two expressions are written the same."""
    return ast.dump(a) == ast.dump(b)


# Calls that are continuous wherever they have a value. A logarithm, a root
# and a power that is not a whole number are, where what they are of is known
# to be where they have one; a step call is asked about on its own, as a step.
_CONTINUOUS_CALLS = frozenset(
    {"exp", "sin", "cos", "sinh", "cosh", "tanh", "atan", "sqrt", "erf", "erfc"}
    | {"abs", "max", "min", "pow", "Piecewise", "And", "Or", "Not"}
)


def _only_bends(
    tree: ast.Expression,
    values: Mapping[str, float],
    clocks: AbstractSet[str],
    asked: AbstractSet[str],
) -> bool:
    """Whether the rate law *tree* is continuous across every condition in it
    that reads the state, by what it is made of.

    The difference quotient is right across a bend, and whether a law bends
    or jumps where a condition flips is not told by evaluating it (issue
    #938). It is proved here, for the ways a bend is written, or not at all:

    - ``if(e > 0, e·g, 0)``: one branch is 0 and the other is a product with
      everything ``e`` is 0 with among its factors, where ``e`` is the
      difference the condition compares. The signed rate
      ``if(v < 0, −v/max(X, 0.01), 0)`` and a ramp ``if(X < thr,
      kb·(thr − X), 0)`` are this. So is ``if(e > 0, e·g, e·h)``, with both
      branches such products;
    - ``if(a < b, a, b)``: the two branches are the two sides of the one
      comparison, the lesser or the greater of two written out.

    And the law is made of nothing else that could jump there: every division
    is by what is known to be nonzero; every power is a whole number that is
    not below 0, or of what is known not to be below 0; a root is of what is
    known not to be below 0 and a logarithm of what is known to be above it,
    since each is no number past the edge of where it has a value, and a
    comparison with no number is false; and every call is one that is
    continuous (:data:`_CONTINUOUS_CALLS`). An equality, a condition that is
    not a comparison, a condition of more than two branches, a comparison
    outside a condition, and anything this does not read, is not proved.
    """
    order, facts = _facts_of_tree(tree, values, clocks, asked)

    def whole(power: _Facts) -> bool:
        return power.value is not None and power.value >= 0.0 and power.value == int(power.value)

    def in_its_domain(base: _Facts, power: _Facts) -> bool:
        """A power that has a value on both sides of anything its base can
        do, and is continuous there."""
        if whole(power):
            return True
        return (base.nonnegative and power.nonnegative) or base.positive

    proved: set[int] = set()
    for node in order:
        name = _call_name(node)
        known = facts[id(node)]
        if isinstance(node, (ast.Name, ast.Constant, ast.Tuple, ast.Compare)):
            # A comparison is asked about below, where it is proved or not.
            pass
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                return False
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                divisor = facts[id(node.right)]
                if not (divisor.positive or divisor.negative):
                    return False
            elif isinstance(node.op, ast.Pow):
                if not in_its_domain(facts[id(node.left)], facts[id(node.right)]):
                    return False
            elif not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                return False
        elif name in _LOG_CALLS:
            if len(node.args) != 1 or not facts[id(node.args[0])].positive:  # type: ignore[attr-defined]
                return False
        elif name == "sqrt":
            if len(node.args) != 1 or not facts[id(node.args[0])].nonnegative:  # type: ignore[attr-defined]
                return False
        elif name in _STEP_NAMES:
            if known.state:
                return False
        elif name == "pow":
            base, power = _operands(node)
            if not in_its_domain(facts[id(base)], facts[id(power)]):
                return False
        elif name not in _CONTINUOUS_CALLS:
            return False
        if name != "Piecewise":
            continue
        pairs = node.args  # type: ignore[attr-defined]
        if not (
            len(pairs) == 2
            and all(isinstance(pair, ast.Tuple) and len(pair.elts) == 2 for pair in pairs)
            and isinstance(pairs[1].elts[1], ast.Constant)
            and pairs[1].elts[1].value is True
        ):
            continue
        (taken, condition), (other, _always) = pairs[0].elts, pairs[1].elts
        # The comparisons the condition is made of, through And, Or and Not.
        # Anything else in it that reads the state is a condition that is a
        # number, true where it is not 0, and is not proved.
        atoms: list[ast.Compare] = []
        stack = [condition]
        while stack:
            one = stack.pop()
            if isinstance(one, ast.Compare):
                atoms.append(one)
            elif _call_name(one) in ("And", "Or", "Not"):
                stack.extend(one.args)  # type: ignore[attr-defined]
            elif facts[id(one)].state:
                return False
        for atom in atoms:
            if not facts[id(atom)].state or len(atom.ops) != 1:
                continue
            if not isinstance(atom.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                continue
            left, right = atom.left, atom.comparators[0]
            # What the comparison's difference is 0 with: each factor of one
            # side where the other is 0, else the difference itself.
            if facts[id(right)].zero:
                needed = [f for f in _factors(left, facts) if facts[id(f)].live]
            elif facts[id(left)].zero:
                needed = [f for f in _factors(right, facts) if facts[id(f)].live]
            else:
                needed = [ast.BinOp(left=left, op=ast.Sub(), right=right)]
                facts[id(needed[0])] = _Facts(
                    state=True, size=facts[id(left)].size + facts[id(right)].size + 1
                )
            if not needed or any(facts[id(f)].size > _FLIP_MAX_NODES for f in needed):
                continue

            def vanishes(product: ast.AST, needed=needed) -> bool:
                """Whether *product* is 0, or has every one of them among
                its factors."""
                if facts[id(product)].zero:
                    return True
                has = {
                    text
                    for factor in _factors(product, facts, vanishing=True)
                    if facts[id(factor)].size <= _FLIP_MAX_NODES
                    for text in _either_way(factor)
                }
                return all(ast.dump(f) in has for f in needed)

            if vanishes(taken) and vanishes(other):
                proved.add(id(atom))
            elif (
                atom is condition
                and facts[id(taken)].size + facts[id(other)].size <= _FLIP_MAX_NODES
                and {ast.dump(taken), ast.dump(other)} == {ast.dump(left), ast.dump(right)}
            ):
                # The lesser or the greater of the two sides, written out.
                proved.add(id(atom))
    return all(
        id(node) in proved
        for node in order
        if isinstance(node, ast.Compare) and facts[id(node)].state
    )


def _clipped(node: ast.AST, width: int = 120) -> str:
    """*node* as text, cut to *width*."""
    try:
        text = ast.unparse(node)
    except RecursionError:
        text = type(node).__name__
    return text if len(text) <= width else text[: width - 3] + "..."


def _parameter_crossing(flat: str, scope: SwitchConditionScope, asked: AbstractSet[str]) -> bool:
    """Whether the quotient reads across *flat*, a comparison or a step call's
    argument over run constants alone, by moving a requested parameter.

    Such a condition holds one way for the whole run, and is still a surface
    in the parameters. The quotient reads the rate law at ``p + σ`` and
    ``p − σ``, and with ``p`` within ``σ`` of where the comparison flips one
    of the two is on the other branch: ``if(n > 1, kb, 0)`` at n = 1 + 1e-6
    returned dY/dn = 901 for 0, and 1.8e10 at n = 1.

    A comparison is one where it comes out differently with a requested
    parameter in it moved by up to a quarter of itself, either way; an
    equality, where it holds; and anything else that reads a requested
    parameter, a step call's argument, whatever its value.
    """
    names = sorted(set(_IDENTIFIER.findall(flat)) & set(asked) & set(scope.param_idx))
    if not names:
        return False
    split = _relational_split_op(flat)
    if split is None:
        return True
    lhs, _op, rhs = split

    def side(values: Sequence[float]) -> int | None:
        both = [
            _evaluate_threshold(text, scope.param_idx, values, scope.derived_exprs)
            for text in (lhs, rhs)
        ]
        if any(v is None or not math.isfinite(v) for v in both):
            return None
        return (both[0] > both[1]) - (both[0] < both[1])  # type: ignore[operator]

    now = side(scope.values)
    if now is None or now == 0:
        return True
    if is_equality_atom(flat):
        return False
    for name in names:
        index = scope.param_idx[name]
        moved = list(scope.values)
        for value in _moves_tried(float(scope.values[index])):
            moved[index] = value
            if side(moved) != now:
                return True
    return False


def fallback_crossing(
    core,
    sens_param_names: Sequence[str] = (),
    ic_species: Sequence[int] = (),
    ctx=None,
    parsed: dict | None = None,
    declared: Mapping[str, Mapping[str, float]] | None = None,
) -> str | None:
    """What a time course on CVODES' difference quotient cannot differentiate
    through in this model, as the text of the first one found, or ``None``.

    The quotient forms each column's right-hand side as ``f`` at ``y + σ·s`` and
    ``p + σ`` less ``f`` at ``y`` and ``p`` (issue #938). Beside a surface where
    a rate law jumps, the moved point is on the other branch for any column
    that moves the state across it: the difference is the jump over ``σ``, and
    the column takes part of the jump before the crossing is reached. The jump
    applied at the crossing itself (issue #48 for a counter, issue #150 for a
    state threshold) is then added to it. ``if(Aobs>thr,kb,0)`` with A made at
    rate ``k``, beside a rate law the analytic path declines, returned
    dY/dk = 14.52 for 10.2. Where the step size collapses on the same
    differences instead, the run ends in CVODE's no-progress error or in the
    wall clock, 2e-5 to 2e-8 short of the crossing (issue #932).

    Nothing at the crossing can put that right: the column is wrong before the
    run gets there, by more the looser the tolerance, and a run that ends short
    of the crossing has returned it. So a model that has one of these is
    refused before a time course starts:

    - a rate-law condition that reads the state, ``Aobs < thr`` or
      ``rateOf(X) > 0``, unless the law is proved to bend there and not jump
      (:func:`_only_bends`): ``if(v > 0, v, 0)`` and ``if(a < b, a, b)``,
      with nothing in the law dividing by what could be 0. The quotient is
      right across a bend. Telling one from a jump by evaluating the law was
      tried and is not sound (issue #938), so what is not proved is refused;
    - a condition on a counter species that a requested column moves (its rate
      constant, its initial amount). A counter nothing moves is read at ``y``
      in both terms of the quotient, and its threshold's parameters are held
      while the quotient is taken (issue #436) where issue #48 reads the
      condition. One it does not read, ``abs(Aobs-c) < 0.01``, is refused where
      a requested parameter moves its threshold: the quotient moves it across;
    - an equality on the state. It holds at one state and over no interval
      of them, unless the state sits there: ``B == 0`` with B made from where
      another species passes a threshold holds until it does;
    - a division the law may jump across through an ``abs``, ``max`` or
      ``min``: ``(thr − X)/abs(thr − X)``
      (:func:`_quotient_across_a_choice`). One that stands alone is a bend,
      and runs;
    - a step call on the state, ``floor(Atot)``, or on a moved counter; and a
      ``clamp`` over one whose bounds are crossed, ``clamp(1, Aobs, 0)``,
      which is a step (issue #1227);
    - a step call on a clock, literal time or a counter nothing moves, whose
      argument reads a requested parameter, ``floor(time()/P)`` with ``P``
      requested: the steps move with ``P``, and nothing jumps a column at a
      step. In a condition as well as outside one;
    - a table function that steps, indexed by an observable, by a function,
      or by a requested parameter or one made of it;
    - a comparison over parameters alone that a requested one is within a
      quarter of itself of flipping, ``if(n > 1, kb, 0)`` at n = 1.001 with n
      requested, an ``abs``, ``max`` or ``min`` over parameters likewise,
      and a step call on a requested parameter
      (:func:`_parameter_crossing`): the quotient moves a parameter as it
      moves the state;
    - a power under 1, a root or a logarithm of what holds an ``abs``,
      ``max`` or ``min`` and may be 0, ``max(thr − X, 0)^0.1`` and
      ``sqrt(thr − min(X, thr))``: continuous, with a slope that has no
      bound. An inverse sine or cosine of one likewise;
    - a comparison that is not the condition of an ``if()``, ``kb*(X > 1)``
      or ExprTk's single ``=``, on what a column moves; in a derived
      parameter as in a rate law, ``kd = 3*(n > 1)``
      (:func:`_comparison_a_column_moves`);
    - a call this does not know, ``atan2``, where a column moves what it
      reads;
    - a rate law that is not read: its functions nest too deep to write out,
      or it holds one of the above in a form the scan does not parse.

    A derived parameter that holds any of these is read as a rate law is:
    written into the laws that read it, and as a law of its own where it is
    the rate constant of a reaction no function writes, or is written in one.
    One that is in no rate, an initial amount, is not asked about. A requested derived
    parameter is asked about both as its own column moves it, held where it
    is set, and as the column of a primary it is written in does.

    A crossing on literal time is not one of these.

    What is asked is what the text says and what each parameter is when the
    scan is made: no rate law is evaluated at a state, nothing is integrated,
    and nothing is assumed of the state.

    Scans the reaction rate expressions with their functions inlined, as
    :func:`model_uncompensated_crossing_reason` does. An atom that names no
    symbol is no crossing.
    """
    from bngsim._jacobian import _TIME_SYM, _inline_functions, has_condition_construct

    # Not caught here: the caller refuses a model whose scan fails, and a
    # model whose tables cannot be listed is one whose step tables are not known.
    data = core.codegen_data()
    stepping = [table for table in data.get("table_functions", ()) if table.get("step")]
    requested = frozenset(sens_param_names)
    param_names = list(core.param_names)
    is_expr = list(core.param_is_expression)
    exprs = [_spell_mod(e) for e in core.param_expressions]
    derived = {
        param_names[i]: exprs[i] for i in range(len(param_names)) if is_expr[i] and exprs[i]
    }
    # ``parsed`` keeps what is made of each rate law's text between one scan
    # of a model and the next, and beside it the last answer for the law with
    # what it went by. What is made of a law has its derived parameters
    # written in, so it is kept for as long as they are written as they were:
    # one that ``set_param`` overrides is a number from then on, and a law
    # that read it is another law.
    #
    # One store for each way they are written, and none emptied under a scan:
    # the rows of a threaded batch share ``parsed``, and a row that overrides
    # a derived parameter is scanned beside one that does not. Emptied and
    # filled again, the store handed one row the laws another had written
    # out, and 4 batches in 3,000 returned a column of 6.94 for 2.46.
    stores: dict = {} if parsed is None else parsed
    written_as = tuple(sorted(derived.items()))
    found_store = stores.get(written_as)
    if found_store is None:
        if len(stores) >= _STORES_KEPT:
            stores.clear()  # a scan under way keeps the store it holds
        found_store = stores.setdefault(written_as, {})
    kept_all: dict = found_store
    for table in stepping:
        index = table.get("index_param_idx", -1)
        moved_index = False
        if table.get("index_kind") == "parameter" and 0 <= index < len(param_names):
            # Indexed by a requested parameter, or by one that is made of
            # one, through however many derived parameters: the quotient
            # moves the index. ``idx = tD + 1`` with ``tD = 2*n`` is moved by
            # tD, which is neither the index nor a primary it is written in.
            moved_index = bool(_written_in(param_names[index], derived) & requested)
        # Indexed by an observable, or by a function, ``fIdx() = Aobs``: by the
        # state, which the quotient moves.
        if table.get("index_kind") not in ("time", "parameter") or moved_index:
            return f'tfun {table.get("name")} (method=>"step")'
    # The derived parameters that hold a condition, a step call, a choice or
    # a call this does not know, themselves or through one they are written
    # in: ``kd = if(n > 1, 3, 0)`` is a rate constant that jumps in n.
    found_marked: frozenset[str] | None = kept_all.get(_MARKED)
    if found_marked is None:
        found_marked = kept_all[_MARKED] = _marked_derived(derived, has_condition_construct)
    marked: frozenset[str] = found_marked
    # Of those, the ones a reaction's rate constant is, or is written in. One
    # that only sets an initial amount, ``R0 = rint(7.47e4*f)``, is in no rate:
    # its seed is differentiated and not differenced.
    found_rates: frozenset[str] | None = kept_all.get(_IN_A_RATE)
    if found_rates is None:
        reach: set[str] = set()
        stack = [
            param_names[i]
            for reaction in data.get("reactions", ())
            if reaction.get("type") != "functional"
            for i in reaction.get("rate_param_indices", ())
            if 0 <= i < len(param_names) and param_names[i] in derived
        ]
        # And the ones a table function is indexed by: the table is read by
        # a rate law as a call, with its index nowhere in the law's text.
        stack += [
            param_names[table.get("index_param_idx", -1)]
            for table in data.get("table_functions", ())
            if table.get("index_kind") == "parameter"
            and 0 <= table.get("index_param_idx", -1) < len(param_names)
            and param_names[table.get("index_param_idx", -1)] in derived
        ]
        while stack:
            name = stack.pop()
            if name not in reach:
                reach.add(name)
                stack.extend(set(_IDENTIFIER.findall(derived[name])) & derived.keys())
        found_rates = kept_all[_IN_A_RATE] = frozenset(reach)
    rate_constants: frozenset[str] = marked & found_rates
    if core.n_functions == 0 and not rate_constants:
        return None
    if ctx is None:
        ctx = core.functional_jacobian_context()
    func_map = _scan_function_map(ctx)
    texts = [_scan_rate_law(reaction) for reaction in ctx["functional_reactions"]]
    # A table a rate law reads is indexed by what its index reads, which the
    # law's text, ``tfun_g()``, does not show. Where the index is a function,
    # ``tfun([0,1],[0,3], h)`` with ``h() = if(Aobs<thr,1,0)``, the
    # function is read as a law of its own: the table of what it holds jumps
    # where it jumps and bends where it bends. Compiled code reads the index
    # live since issue #965, so the quotient reads across that crossing. An
    # index function may read a table of its own, ``h() = tfun(..., h0)``, so
    # what the indices read is asked in turn until nothing new is found. Kept
    # beside the laws, and only for a model that has such a table.
    function_names = [str(f["name"]) for f in data.get("functions", ())]
    table_indices = {
        str(table.get("name")): function_names[table["index_func_idx"]]
        for table in data.get("table_functions", ())
        if table.get("index_kind") == "function"
    }
    if table_indices:
        found_indices: tuple[str, ...] | None = kept_all.get(_TABLE_INDICES)
        if found_indices is None:
            table_call = re.compile(
                r"(?<![A-Za-z0-9_])tfun_(" + "|".join(map(re.escape, table_indices)) + r")\("
            )
            memo: dict[str, str] = {}
            reached: set[str] = set()
            frontier = list(texts)
            while frontier:
                read = " ".join(
                    _inline_functions(text, func_map, memo=memo) or text for text in frontier
                )
                new = {table_indices[name] for name in table_call.findall(read)} - reached
                reached |= new
                frontier = sorted(new)
            found_indices = kept_all[_TABLE_INDICES] = tuple(sorted(reached))
        texts += found_indices
    moved_clocks: list[frozenset[int]] = []

    def scan(held: frozenset[str]) -> str | None:
        """The scan with the derived parameters in *held* left as names, at
        the values they have, and every other one written out."""
        kept = kept_all.setdefault((_VIEW, held), {})
        live = {n: e for n, e in derived.items() if n not in held} if held else derived
        laws: list[_Law] = []
        for text in texts:
            law = kept.get(text)
            if law is None:
                flat = _inline_functions(text, func_map)
                if flat is None:
                    # Functions nested past what is written out, or a cycle. Read
                    # as its bare name, it would hold no condition at all.
                    return f"{text} (its functions nest too deep to read)"
                law = kept[text] = _read_law(
                    text, flat, live, has_condition_construct, marked - held
                )
            if law.atoms or law.steps or law.choice or law.unread or law.compared:
                laws.append(law)
        # A marked derived parameter may be the rate constant of a reaction no
        # function writes, which is in none of the texts above. Each such is
        # read as a rate law of its own, over parameters alone.
        for name in sorted(rate_constants - held):
            law = kept.get((_DERIVED_LAW, name))
            if law is None:
                law = kept[(_DERIVED_LAW, name)] = _read_law(
                    f"{name} := {derived[name]}",
                    derived[name],
                    live,
                    has_condition_construct,
                    marked - held,
                )
            if law.atoms or law.steps or law.choice or law.unread or law.compared:
                laws.append(law)
        if not laws:
            return None
        scope = switch_condition_scope(core, ctx)
        if held:
            scope = scope._replace(
                derived_exprs=live,
                primary_names=scope.primary_names | held,
                run_constants=scope.run_constants
                | (held - scope.function_names - scope.clock_symbols),
            )
        if not moved_clocks:
            # The same in every view: which species is a clock, and which
            # parameter seeds or drives it, is not a matter of what is held.
            moved_clocks.append(
                _clocks_moved(
                    core, scope.clocks, list(sens_param_names), ic_species, True, declared
                )
            )
        moved = moved_clocks[0]
        # What each parameter is; and the clocks no column moves, literal time
        # and a counter with nothing requested that moves it.
        values = {name: float(scope.values[scope.param_idx[name]]) for name in scope.run_constants}
        for name, value in _BUILTIN_CONSTANT_VALUES.items():
            values.setdefault(name, float(value))
        clocks = frozenset({_TIME_SYM}) | {
            n for n, idx in scope.clocks.items() if idx not in moved
        }

        def crosses(flat: str) -> str:
            """Whether the quotient reads across the surface *flat* names, and
            in what: ``"parameter"``, ``"state"``, or ``""`` for neither."""
            if not _IDENTIFIER.search(flat):
                return ""
            if condition_cannot_cross(flat, scope):
                # Asked again only where a parameter the comparison reads has
                # changed, or the columns have.
                key = (
                    tuple(
                        float(scope.values[scope.param_idx[n]])
                        for n in sorted(set(_IDENTIFIER.findall(flat)) & set(scope.param_idx))
                    ),
                    requested,
                )
                last = kept.get(("parameter", flat))
                if last is None or last[0] != key:
                    last = (key, _parameter_crossing(flat, scope, requested))
                    kept[("parameter", flat)] = last
                return "parameter" if last[1] else ""
            if _reads_clock_and_run_constants(flat, scope):
                names = set(_IDENTIFIER.findall(flat))
                read = {scope.clocks[n] for n in names if n in scope.clocks}
                if {i for i in read if i >= 0} & moved:
                    return "state"
                # A requested parameter moves the threshold, which issue #48
                # holds while the quotient is taken only where it reads the
                # atom. It does not read `abs(Aobs-c) < 0.01` on a counter,
                # whose edges are rooted as crossings of the state instead
                # (issue #904), and the quotient moved c across them: dY/dc
                # came back 7.03 for 0.2. A parameter the quotient moves across
                # a surface, as `_parameter_crossing` finds: a law that only
                # bends there is proved so of its conditions on the state, and
                # this one reads none.
                if not names & requested:
                    return ""
                held_by = (
                    tuple(
                        float(scope.values[scope.param_idx[n]])
                        for n in sorted(names & set(scope.param_idx))
                    ),
                    requested,
                    clocks,
                )
                verdict = kept.get(("held", flat))
                if verdict is None or verdict[0] != held_by:
                    verdict = (held_by, clock_crossing_compensated(flat, scope))
                    kept[("held", flat)] = verdict
                return "" if verdict[1] else "parameter"
            return "state"

        def moved_by_a_column(flat: str) -> bool:
            """Whether a column moves anything *flat* reads: the state, a
            counter one moves, or a requested parameter."""
            names = set(_IDENTIFIER.findall(flat))
            if names & requested:
                return True
            if condition_cannot_cross(flat, scope):
                return False
            if _reads_clock_and_run_constants(flat, scope):
                read = {scope.clocks[n] for n in names if n in scope.clocks}
                return bool({i for i in read if i >= 0} & moved)
            return True

        def went_by(law: _Law) -> tuple:
            """What an answer about *law*'s tree goes by: what each parameter it
            reads is, which of its names are clocks, and which are requested."""
            return (
                tuple((n, values[n]) for n in sorted(law.names) if n in values),
                law.names & clocks,
                law.names & requested,
            )

        def bends(law: _Law) -> bool:
            """Whether *law* is proved continuous across its conditions on the
            state (:func:`_only_bends`)."""
            if law.tree is None:
                return False
            key = went_by(law)
            last = kept.get(("bends", law.text))
            if last is None or last[0] != key:
                last = (key, _only_bends(law.tree, values, clocks, requested))
                kept[("bends", law.text)] = last
            return last[1]

        def step_moves(flat: str, raw: str) -> bool:
            """A step on a clock that a requested parameter moves: on literal
            time, or on a counter no column moves, which is the same clock."""
            if not _reads_clock_and_run_constants(flat, scope):
                return False
            return bool(
                (set(_IDENTIFIER.findall(flat)) | set(_IDENTIFIER.findall(raw))) & requested
            )

        for law in laws:
            if law.unread and moved_by_a_column(law.full):
                return f"{_clip_text(law.full)} ({law.unread})"
            if law.compared and law.tree is not None:
                key = went_by(law)
                last = kept.get(("compared", law.text))
                if last is None or last[0] != key:
                    last = (key, _comparison_a_column_moves(law.tree, values, clocks, requested))
                    kept[("compared", law.text)] = last
                if last[1] is not None:
                    return last[1]
            for atom, atom_flat in law.atoms:
                across = crosses(atom_flat)
                if across == "parameter" or (across == "state" and not bends(law)):
                    return atom
                # A step in a condition on a clock: its edges move with a
                # requested parameter as they do outside one.
                if _STEP_CALL.search(atom_flat) and step_moves(atom_flat, atom):
                    return atom
            for call, arg, arg_flat in law.steps:
                if crosses(arg_flat) or step_moves(arg_flat, arg):
                    return call
            if law.choice:
                if law.tree is None:
                    return f"{_clip_text(law.full)} (not read)"
                key = went_by(law)
                last = kept.get(("quotient", law.text))
                if last is None or last[0] != key:
                    last = (
                        key,
                        _quotient_across_a_choice(law.tree, values, clocks, requested),
                    )
                    kept[("quotient", law.text)] = last
                if last[1] is not None:
                    return last[1]
        return None

    # A requested derived parameter is asked about twice. Its column moves it
    # where it is set, with what it is written in held (issue #750): it is a
    # name there, at the value it has. And the column of a primary it is
    # written in moves it through that: there it is written out.
    #
    # Several are held together where none is written in another. Where one
    # is, ``tD = 2*n`` and ``kd = if(tD > 1, 3, 0)`` with both requested, each
    # is held alone: held together, ``kd`` was a name in tD's view, and what
    # it does as tD's column moves it was not read (18 returned for 0).
    held = frozenset(requested & derived.keys())
    views: list[frozenset[str]] = [frozenset()]
    if held:
        nested = kept_all.get((_NESTED, held))
        if nested is None:
            nested = kept_all[(_NESTED, held)] = _one_written_in_another(held, derived)
        views += [frozenset({name}) for name in sorted(held)] if nested else [held]
    for view in views:
        found = scan(view)
        if found is not None:
            return found
    return None


def _written_in(name: str, derived: Mapping[str, str]) -> set[str]:
    """*name*, and every name it is written in: the derived parameters on the
    way down, and what they are written in at the end."""
    seen = {name}
    stack = [name]
    while stack:
        text = derived.get(stack.pop())
        if text is None:
            continue
        for read in set(_IDENTIFIER.findall(text)) - seen:
            seen.add(read)
            stack.append(read)
    return seen


def _one_written_in_another(held: AbstractSet[str], derived: Mapping[str, str]) -> bool:
    """Whether one of the derived parameters *held* is written in another of
    them, itself or through derived parameters that are not."""
    for name in held:
        seen: set[str] = set()
        stack = [name]
        while stack:
            for read in set(_IDENTIFIER.findall(derived[stack.pop()])) & derived.keys():
                if read in held:
                    return True
                if read not in seen:
                    seen.add(read)
                    stack.append(read)
    return False


# How many ways of writing the derived parameters a scan keeps laws for.
_STORES_KEPT = 8
# Keys of what a scan keeps that are not a rate law's text.
_MARKED = ("derived parameters that hold what is asked about",)
_IN_A_RATE = ("derived parameters a rate constant is, or is written in",)
_TABLE_INDICES = ("functions a table a rate law reads is indexed by",)
_VIEW = "held"
_NESTED = "held, one written in another"
_DERIVED_LAW = "derived"


class _Law(NamedTuple):
    """What a rate law's text comes to, which no parameter's value changes."""

    text: str
    flat: str  # with its functions inlined
    # And with every derived parameter that holds a condition, a step call, a
    # choice or an unknown call written out, so that those are in the text.
    full: str
    # Each condition atom, as written and with derived parameters inlined.
    atoms: tuple[tuple[str, str], ...]
    # Each step call outside a condition: the call, its argument, and the
    # argument with derived parameters inlined.
    steps: tuple[tuple[str, str, str], ...]
    choice: bool  # holds an abs, max or min, or a call this does not know
    # Compares outside an ``if()``, ``beta*(I > 1)``: its conditions are read
    # off the tree, and are not in ``atoms``.
    compared: bool
    unread: str  # what it holds that is not read at all, or ""
    tree: ast.Expression | None  # its syntax tree, where it holds one
    names: frozenset[str]  # what the tree reads


# A name written as a call.
_ANY_CALL = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
# Words ExprTk and the loaders write before a parenthesis that are no call:
# ``a and (b)``, which is what an SBML ``<and/>`` arrives as.
_NOT_CALLS = frozenset(
    {"if", "time", "and", "or", "not", "nand", "nor", "xor", "xnor", "true", "false"}
)
# ExprTk's equality written with one ``=``: ``kb*(max(A - thr, 0) = 0)`` is a
# step, outside any ``if()``, that the condition scan does not read.
_SINGLE_EQUALS = re.compile(r"(?<![<>=!:~])=(?![=>])")
# ExprTk's conditional written ``c ? a : b``.
_TERNARY = re.compile(r"\?")


def _compares_outside_an_if(text: str) -> bool:
    """Whether *text* holds a comparison, a logical operator, a ``c ? a : b``
    or a single ``=`` that is not in the condition of an ``if()``: the
    boolean read as a number, ``beta*(I > 1)``, which is 1 or 0. The scan of
    conditions reads those of an ``if()`` and does not find it."""
    if _SINGLE_EQUALS.search(text) or _TERNARY.search(text):
        return True
    spans = _SpanIndex(_condition_spans(text))
    return any(
        not spans.contains(m.start(), m.end())
        for pat in (_RELATIONAL, _LOGICAL, _NOT_OP, _NOT_CALL)
        for m in pat.finditer(text)
    )


def _comparison_a_column_moves(
    tree: ast.Expression,
    values: Mapping[str, float],
    clocks: AbstractSet[str],
    asked: AbstractSet[str],
) -> str | None:
    """A condition of the rate law *tree* that a column moves, or may, as
    text, or ``None``. For a law that compares outside an ``if()``, whose
    conditions are read here off the tree, where each is the condition of a
    ``Piecewise``.

    One that reads the state, or a counter a column moves, is named: nothing
    proves the law bends there. So is one on a clock that reads a requested
    parameter. One over numbers and parameters alone is named where a
    requested parameter in it is within a quarter of itself of flipping it,
    as :func:`_parameter_crossing` has it for the condition of an ``if()``;
    an equality, where it holds or would.
    """
    order, facts = _facts_of_tree(tree, values, clocks, asked)

    def moved(cond: ast.AST) -> bool:
        if isinstance(cond, ast.Constant):
            return False  # the ``True`` of the branch that is taken otherwise
        known = facts[id(cond)]
        if known.state:
            return True
        if not known.asked:
            return False
        if known.clock:
            return True
        name = _call_name(cond)
        if name in ("And", "Or", "Not"):
            return any(moved(arg) for arg in cond.args)  # type: ignore[attr-defined]
        if name in ("Eq", "Ne") and len(cond.args) == 2:  # type: ignore[attr-defined]
            left, right = cond.args  # type: ignore[attr-defined]
        elif isinstance(cond, ast.Compare) and len(cond.comparators) == 1:
            left, right = cond.left, cond.comparators[0]
        else:
            return True
        return _flips_with_a_parameter(
            ast.BinOp(left=left, op=ast.Sub(), right=right),  # type: ignore[arg-type]
            values,
            asked,
        )

    for node in order:
        if _call_name(node) != "Piecewise":
            continue
        for pair in node.args:  # type: ignore[attr-defined]
            if not (isinstance(pair, ast.Tuple) and len(pair.elts) == 2):
                return _clipped(node)
            if moved(pair.elts[1]):
                return _clipped(pair.elts[1])
    return None


def _calls_what_is_not_known(text: str) -> bool:
    return any(
        _unknown_call(m.group(1)) and m.group(1) not in _NOT_CALLS
        for m in _ANY_CALL.finditer(text)
    )


def _marked_derived(derived: Mapping[str, str], has_condition_construct) -> frozenset[str]:
    """The derived parameters whose value can jump or bend with what they are
    written in: one that holds a condition, a step call, an ``abs``, ``max``
    or ``min``, a call this does not know or a single ``=``, and every one
    written in such a one."""
    marked = {
        name
        for name, text in derived.items()
        if has_condition_construct(text)
        or _STEP_CALL.search(text)
        or _CHOICE_CALL.search(text)
        or _SINGLE_EQUALS.search(text)
        or _TERNARY.search(text)
        or _calls_what_is_not_known(text)
    }
    if not marked:
        return frozenset()
    reads = {
        name: set(_IDENTIFIER.findall(text)) & derived.keys() for name, text in derived.items()
    }
    grew = True
    while grew:
        grew = False
        for name, read in reads.items():
            if name not in marked and read & marked:
                marked.add(name)
                grew = True
    return frozenset(marked)


def _read_law(
    text: str,
    flat: str,
    derived_exprs,
    has_condition_construct,
    marked: AbstractSet[str] = frozenset(),
) -> _Law:
    # Written out only where it shows something: flattening every derived
    # parameter of a law is exponential in how deep they are defined.
    full = flat
    if marked and not marked.isdisjoint(_IDENTIFIER.findall(flat)):
        full = _inline_derived_param_refs(flat, derived_exprs) or flat
    atoms: tuple[tuple[str, str], ...] = ()
    if has_condition_construct(full):
        atoms = tuple(
            (atom, _inline_derived_param_refs(atom, derived_exprs) or atom)
            for atom in _iter_condition_atoms(full)
        )
    steps = tuple(
        (call, arg, _inline_derived_param_refs(arg, derived_exprs) or arg)
        for call, arg in _iter_step_calls(full)
    )
    choice = _CHOICE_CALL.search(full) is not None or _calls_what_is_not_known(full)
    compared = _compares_outside_an_if(full)
    tree = None
    names: frozenset[str] = frozenset()
    if choice or atoms or compared:
        # With derived parameters written out, so that what is left to name
        # is a primary, a clock or the state.
        tree = _syntax_tree(_inline_derived_param_refs(full, derived_exprs) or full)
        if tree is not None:
            names = frozenset(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
    # ``c ? a : b``, ``!c`` and ``xor`` are not in the tree's grammar.
    unread = "a comparison outside an if(), not read" if compared and tree is None else ""
    return _Law(text, flat, full, atoms, steps, choice, compared, unread, tree, names)


def _clip_text(text: str, width: int = 120) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


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
        # A threshold written through clock guards (issue #545) has one crossing
        # candidate per cell, and is compensated when each of them is — the ones
        # that fall outside their own cell are read as crossings that do not happen.
        guarded = _clock_guarded_thresholds(atom, scope.clock_symbols)
        if guarded is not None:
            return all(_threshold_compensated(g.threshold, scope) for g in guarded[1])
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
    ``∂threshold/∂p`` over the parameters with a non-zero partial, empty for a
    crossing nothing moves — the primaries always, plus any *derived* parameter
    the caller asked to be differentiated on its own terms (issue #475).

    Only the primaries decide whether the crossing is compensated, so the gate
    and the detector read one rule whatever either of them asked for.
    """

    partials: dict[str, float]
    value: float | None


def _threshold_crossing_terms(
    threshold_expr: str,
    scope: SwitchConditionScope,
    own_names: AbstractSet[str] = frozenset(),
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

    ``own_names`` are derived parameters a caller wants a column for. Each one
    the threshold reaches gets ``∂threshold/∂(that name)`` in ``partials``
    alongside the primaries, with the name held as an independent input rather
    than chain-ruled away into what it is defined from (issue #475). That is the
    same question the smooth half of its column already answers, since the
    emitted right-hand side reads the parameter's own slot; computing the jump
    any other way leaves the two halves of one column disagreeing. It is added
    after the decision above and never feeds it, so what makes a crossing
    compensated is a question about the primaries alone whoever asked.

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
    wanted = {n for n in own_names if n in scope.derived_exprs}
    walked = _derived_expr_partials_numeric(
        threshold_expr,
        set(scope.primary_names),
        scope.param_idx,
        list(scope.values),
        scope.derived_exprs,
        warn_on_failure=False,
        include_derived=bool(wanted),
    )
    # Split rather than widened. The primaries alone decide whether the crossing
    # is compensated, so that decision reads exactly what it read before issue
    # #475 and the gate cannot answer it differently from the detector.
    partials = {n: v for n, v in walked.items() if n in scope.primary_names}
    value = _evaluate_threshold(threshold_expr, scope.param_idx, scope.values, scope.derived_exprs)
    if value is None:
        if _threshold_is_non_real(
            threshold_expr, scope.param_idx, scope.values, scope.derived_exprs
        ):
            return CrossingTerms({}, None)  # a root that does not occur
        return None
    if not all(math.isfinite(v) for v in partials.values()):
        # A crossing that occurs but whose time has no finite derivative in a
        # primary (``thr = 1/a`` at a = 0, ``E^n`` at E = 0 with n < 1) has
        # nothing to jump by, so it is not compensated. The walk keeps such a
        # partial rather than dropping it (issue #720), since a dropped one
        # reads as a zero. Asked after the non-real check: a root that does not
        # occur has no crossing time to differentiate.
        return None
    # ``value`` is already known to be a number here, so the text scan is the
    # whole of :func:`_fixed_threshold_expr` that is left to check.
    if partials or _threshold_has_no_parameter(threshold_expr, scope):
        out = {n: float(v) for n, v in partials.items() if v != 0.0}
        # Added AFTER the decision above, never before it (issue #475).
        out.update({n: float(v) for n, v in walked.items() if n in wanted and v != 0.0})
        return CrossingTerms(out, value)
    return None


def _threshold_compensated(threshold_expr: str, scope: SwitchConditionScope) -> bool:
    """Whether this one crossing time needs no compensation or gets it."""
    return _threshold_crossing_terms(threshold_expr, scope) is not None


class ScheduleTerms(NamedTuple):
    """What a repeating schedule contributes, before a window is applied.

    The three numbers are the schedule read at the current parameter point, and
    the three dictionaries are ``∂(that number)/∂p`` over the parameters with a
    non-zero partial — the primaries, plus any derived parameter the caller
    asked :func:`_threshold_crossing_terms` to keep on its own terms (issue
    #475). ``crosses`` is whether the condition turns over at
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

    :func:`_clock_periodic_schedule` reads the residual symbolically, and a
    misreading fails quietly: a schedule that reads as never crossing is one the
    gate admits with nothing behind it. The case that motivated this check was
    sympy's own namespace, which read a model parameter ``I`` as the imaginary
    unit and folded ``I*I`` to ``-1``. Model names are bound as plain symbols now
    (issue #757), and the check stays as the net under whatever the recognizer
    gets wrong next.

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
    atom: str,
    sched: PeriodicSchedule,
    scope: SwitchConditionScope,
    own_names: AbstractSet[str] = frozenset(),
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
        part = _threshold_crossing_terms(text, scope, own_names)
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

# How far a value read off an expanded sympy form may sit from the integer or
# zero it stands for, relative to the sum of its terms' magnitudes, and still be
# that integer (:func:`_resolve_step_edge_stop_times`). Each constant is off the
# model's own arithmetic by an ulp or two (a reciprocal where the model divides),
# and a few terms add a few more; 64 ulps covers that with room. A non-integer
# is never read as one unless it is within 1.4e-14 of the scale, so a large
# offset keeps its fraction: (15 + 2e9)/24 is 83333333.958, not 83333334.
_ROUNDING = 64 * sys.float_info.epsilon


def _lambdify_doubles(args, expr):
    """``sp.lambdify`` over the math module, printing each Float as the double it
    holds. The default printer writes 15 significant digits, so ``1/24`` came
    back ``0.0416666666666667`` and moved every value read through it."""
    import sympy as sp

    return sp.lambdify(args, expr, modules="math", printer=_double_printer()())


@functools.cache
def _double_printer():
    from sympy.printing.pycode import PythonCodePrinter

    class _DoublePrinter(PythonCodePrinter):
        def _print_Float(self, expr):
            return repr(float(expr))

    return _DoublePrinter


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


def _schedule_edge_times(terms: ScheduleTerms, v_lo: float, v_hi: float) -> list[float] | None:
    """The times :func:`_schedule_edges` lists, to the bit, without the partials
    a stop has no use for: a million of them were a second of dictionaries
    (issue #890). ``None`` where the window does not resolve to a finite range.
    """
    import numpy as np

    if not terms.crosses:
        return []
    out = []
    for base in (terms.offset + terms.duty, terms.offset):
        window = _schedule_index_window(base, terms.period, v_lo, v_hi)
        if window is None:
            return None
        k = np.arange(window[0], window[1] + 1, dtype=np.int64).astype(float)
        value = base + k * terms.period
        out.append(value[(value > v_lo) & (value <= v_hi)])
    return np.concatenate(out).tolist()


def _schedule_edges(
    terms: ScheduleTerms, v_lo: float, v_hi: float, limit: int
) -> list[tuple[float, dict[str, float]]] | None:
    """Every edge of a repeating schedule in ``(v_lo, v_hi]``, as
    ``(clock value, ∂(clock value)/∂p)`` over whatever parameters *terms* carries
    — or ``None`` when there are more than ``limit`` of them (issue #436).

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
    # Same cheap first gate as compute_switch_time_sens, on the raw bodies: no
    # `if()` anywhere means no branch to cross, and no RHS probe is paid.
    if not any(_IF_CALL.search(body) for body in dict(ctx["function_map"]).values()):
        return []
    func_map = _scan_function_map(ctx)
    bodies = list(func_map.values())

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
        for whole in _iter_condition_atoms(flat):
            if clock_crossing_compensated(whole, scope):
                continue  # issue #48's crossing; jumping it here would double it
            if is_equality_atom(whole):
                continue  # measure-zero: rooting on it would MAKE the branch (below)
            # A residual that turns within a step is rooted, and jumped, at
            # each of its edges instead of on itself (issue #904): a root on
            # `(A-5)^2 - 1e-4` saw one sign at both ends of the step that held
            # the window, and the run returned Y = 0 and dY/dk = 0. The jump at
            # each edge is read off the right-hand side either side of it, as
            # at any crossing; with the whole atom rooted beside its edges, a
            # step that found both at one instant would jump it twice.
            for atom in _monotone_pieces(whole) or [whole]:
                residual = state_switch_residual(core, atom)
                if residual and residual not in seen_residual:
                    seen_residual.add(residual)
                    conditions.append(atom)
    return conditions


def rate_law_equalities(core, ctx=None) -> list[tuple[str, str]]:
    """The plain equalities over live state a functional rate law branches on.

    Each as ``(atom, residual)``, one per residual: an equality (or ``!=``) of
    things that move smoothly, in a rate law or a function one reaches (one an
    output alone reads moves nothing). An equality over a step or bend call
    (``floor(T/24) == 3``) is a window like any other and is not one. A state
    on one of these surfaces holds its branch for a stretch, and a
    sensitivity run roots where it leaves (issues #1228, #1279). ``[]`` for a
    model with no conditional rate law, before anything is probed.
    """
    from bngsim._jacobian import _inline_functions

    if core.n_functions == 0:
        return []
    if ctx is None:
        ctx = core.functional_jacobian_context()
    if not any(_IF_CALL.search(body) for body in dict(ctx["function_map"]).values()):
        return []
    func_map = _scan_function_map(ctx)
    memo: dict[str, str] = {}
    # The rate laws, with the functions they reach written in. Each law and each
    # atom is read once, however many reactions share it.
    atoms: dict[str, None] = {}
    for law in dict.fromkeys(_scan_rate_law(rxn) for rxn in ctx["functional_reactions"]):
        flat = _inline_functions(law, func_map, memo=memo) or law
        if _IF_CALL.search(flat):
            atoms.update(dict.fromkeys(_iter_condition_atoms(flat)))
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for whole in atoms:
        if not is_equality_atom(whole) or _STEP_CALL.search(whole) or _BEND_CALL.search(whole):
            continue
        residual = state_switch_residual(core, whole)
        if residual and residual not in seen:
            seen.add(residual)
            candidates.append((whole, residual))
    return candidates


def state_switch_reactions(core, conditions: Sequence[str], ctx=None) -> list[list[int]]:
    """For each condition, the 0-based functional reactions whose rate law reads it.

    Parallel to ``conditions`` (the list :func:`state_switch_conditions`
    returns, and the stops a run with sensitivities adds for step and bend
    calls), for ``SolverOptions.set_state_switch_reactions``. A law reads a stop
    when it holds a step call rooted there; a bend lists no law, since it does
    not jump. The solver asks
    whether a crossing is continuous of these reactions' flux alone (issue
    #763). Asked of every species' whole dx/dt, a species the switch never
    touches decided it: one whose rate law cancels internally (``0.3*P -
    0.1*Q`` over two 1e8 pools) contributes pure roundoff that reads as a jump,
    and a switched species' own large turnover dilutes a real one. A rate law
    reads a condition when an atom of its inlined text has the same crossing
    surface (:func:`_surface_key`), so ``X<1`` and ``X<=1`` both match. An
    empty entry means no functional rate law reads it (only an output does), so
    the solver takes the crossing as continuous; a tangent crossing, which its
    probes cannot see past, is still judged over the whole right-hand side.

    The list is of functional rate laws whose own text reads the condition. A
    law can read one without it: through a table function indexed by a function
    that holds the condition, ``tfun([0,1],[0,3], h)`` with
    ``h() = if(X<1, 1, 0)``, and is not listed. (A Michaelis-Menten ``kcat``
    that a function writes was another such reader; that model is refused where
    it is built, issue #931.) The solver covers that: a jump the pre-#763 test
    reads, and that the listed reactions do not account for, is judged as
    before. A table has no analytic sensitivity right-hand side, so a run whose
    rate reads one through a function that holds a crossing is refused by
    :func:`fallback_crossing` before it gets there (issues #938, #965).
    """
    from bngsim._jacobian import _inline_functions

    out: list[list[int]] = [[] for _ in conditions]
    if not conditions:
        return out
    if ctx is None:
        ctx = core.functional_jacobian_context()
    func_map = _scan_function_map(ctx)
    memo: dict[str, str] = {}

    def inline(text: str) -> str:
        return _inline_functions(text, func_map, memo=memo) or text

    key_of: dict[str, str | None] = {}

    def surface(atom: str) -> str | None:
        if atom not in key_of:
            residual = state_switch_residual(core, atom)
            key_of[atom] = _surface_key(residual) if residual else None
        return key_of[atom]

    wanted: dict[str, list[int]] = {}
    for i, cond in enumerate(conditions):
        key = surface(inline(cond))
        if key is not None:
            wanted.setdefault(key, []).append(i)
    if not wanted:
        return out
    for rxn in ctx["functional_reactions"]:
        flat = inline(_scan_rate_law(rxn))
        hit: set[int] = set()
        if _IF_CALL.search(flat):
            for written in _iter_condition_atoms(flat):
                for whole in _split_logical_atoms(written):
                    # The edges a turning residual is rooted at (issue #904) are
                    # read by the rate law that reads the whole atom.
                    for atom in (whole, *_monotone_pieces(whole)):
                        hit.update(wanted.get(surface(atom) or "", ()))
        # A step call jumps the law that holds it at the roots that stand for
        # it (issue #1068). sign() over the state has an analytic sensitivity
        # right-hand side since issue #1128, so the solver applies that jump, and
        # it reads it from the laws listed here. Unlisted, the jump was read
        # against the largest |dx/dt| anywhere: a 1e8-molecule pool turning over
        # beside it took the saltation term away, and dX/dk came back 96% off.
        if _STEP_CALL.search(flat):
            for call, _arg in _iter_step_calls(flat):
                for atom in _call_turn_pieces(call):
                    hit.update(wanted.get(surface(atom) or "", ()))
        for i in hit:
            out[i].append(int(rxn["rxn_idx"]))
    return out


def state_switch_root_conditions(
    core,
    covered: Sequence[str] = (),
    ctx=None,
    *,
    calls_only: bool = False,
    with_guards: bool = False,
):
    """Rate-law conditions over model state that a run without sensitivities roots on (#897).

    Inside each branch of ``if((time() >= onset) && (time() < onset + w), k, 0)``
    with ``onset = 4*S`` the rate is constant, so a step that spans the whole
    window reads the branch as off at both ends and CVODE's error estimate over
    it is near zero. The window is lost, with no warning at any tolerance. A
    fixed crossing stop cannot help, because the crossing time moves with ``S``.
    Registering the atom's residual as a root is what stops the step inside the
    window. A sensitivity run gets that root from :func:`state_switch_conditions`
    (issue #150); this is the set for every other run, and it gets no jump.

    It differs from :func:`state_switch_conditions` in three ways, all because
    no jump rides on it:

    * It never asks :func:`clock_crossing_compensated`, which differentiates
      every threshold with sympy to decide whether issue #48 can jump it. That
      costs up to ~0.3 s a call on the ``.net`` corpus, and nothing here needs
      a derivative. What stands in for it is ``covered``: the conditions the run
      already stops on or roots, which are
      :meth:`bngsim.Model.time_discontinuity_conditions`. A counter-clock
      threshold reads a species, so its residual resolves too, and rooting it
      as well as stopping on it would put a second restart at the instant the
      issue #443 landing has already handled. For an SBML model, ``covered`` is
      the set its loader registered as roots, state thresholds included
      (GH #194), so a crossing is not rooted twice.
    * It scans the functional rate laws and the functions they reach, rather
      than every function body. A condition in a function that no rate law
      reads changes an output, not the trajectory, so a root on it would buy a
      restart and nothing else.
    * It reads no parameter value, so the answer depends on the model's
      structure and ``covered`` alone, and
      :meth:`bngsim.Model.state_switch_root_conditions` caches it for each
      ``covered``. ``covered`` itself moves with a parameter value where a
      counter is a clock at one rate and not at another (issue #726), so a fit
      over such a rate pays once per verdict, and any other fit once.

    A step or bend call over state is rooted where it jumps or bends, as a
    comparison is where it flips (issue #1068), so ``max(A, 0)`` alone gets a
    root at ``A = 0``.

    With *calls_only*, only the roots that stand for those calls are returned:
    the set a run with sensitivities adds as stops to the conditions
    :func:`state_switch_conditions` jumps. It is this set with the comparisons
    taken out, deduplicated as here.

    With *with_guards*, returns ``(conditions, guards)``: for each condition,
    the ``if()`` branches its step or bend calls stand in, as an expression
    that reads 1 where one of them is on, or ``""`` where any of its readings
    stands in no branch (issue #1147). A crossing whose every branch is off
    changes nothing, and a plain run steps on through it rather than restart.
    Those branches' own conditions are rooted, so they cannot turn at it.

    Equalities are skipped for the reason :func:`state_switch_conditions` gives.
    Deduplicated by residual, read through :func:`_surface_key`, so one crossing
    written two ways is one root. Empty for a model with no conditional rate
    law and no step or bend call, before the model is probed.
    """
    from bngsim._jacobian import _IDENT_RE, _inline_functions

    if core.n_functions == 0:
        return ((), ()) if with_guards else ()

    # The raw texts gate first, as in time_discontinuity_conditions: inlining
    # cannot introduce an `if(` or a comparison that none of them spells, and
    # the context this skips runs to tens of thousands of entries on a
    # genome-scale model. A comparison is read wherever it stands: one used as
    # a number, `k*((time()>=onset())*(time()<onset()+w))`, is the same window
    # as the `if()` spelling and was not rooted (issue #904). So is a step or
    # bend call, which spells the same pulse with no comparison (issue #1068).
    def switches(text: str) -> bool:
        return bool(
            _IF_CALL.search(text)
            or _RELATIONAL.search(text)
            or _STEP_CALL.search(_spell_mod(text))
            or _BEND_CALL.search(text)
        )

    if not any(switches(t) for t in (*core.function_expressions, *core.param_expressions)):
        return ((), ()) if with_guards else ()
    if ctx is None:
        ctx = core.functional_jacobian_context()
    func_map = _scan_function_map(ctx)
    memo: dict[str, str] = {}

    def inline(text: str) -> str:
        return _inline_functions(text, func_map, memo=memo) or text

    # A covered condition is inlined before it is compared. The SBML loader
    # registers `time() >= onset` for an assignment rule `onset := 4*S`, and
    # `onset` is a parameter slot, so that text reads no state and has no
    # residual; the rate law's own atom arrives inlined, as `time() >= 4*S`.
    #
    # A covered comparison that turns covers its edges (issue #904). A clock's
    # `abs(Tobs-5) < 0.01` is stopped on at both of them, and its edges, rooted
    # as well, put a second restart at each stop. The SBML loader registers the
    # edges themselves, which read as they are. It is split as it is written,
    # before it is inlined: one the loader registered whole, `(A-5)^2 < w`
    # with `w := 0.0001`, is a root on the whole that never fires, and split
    # after `w` was written in, it marked the rate law's edges as covered and
    # the window was stepped over again.
    #
    # A covered step or bend call of a clock (issue #869) is stopped on at each
    # of its jumps, and covers the roots that stand for them and for the calls
    # inside it (issue #1068).
    def is_call(text: str) -> bool:
        return bool(_STEP_CALL.match(text) or _BEND_CALL.match(text)) and (
            _find_close_paren_strict(text, text.index("(")) == len(text) - 1
        )

    def call_atoms(text: str) -> list[str]:
        if not is_call(text):
            return []
        return _call_turn_pieces(text)

    # One atom is met again from every rate law and function that reads it, and
    # a min() of two inlined assignment rules has a residual 11k characters
    # long, whose key costs milliseconds (BIOMD0000000385).
    keys: dict[str, str | None] = {}

    def surface(atom: str) -> str | None:
        if atom not in keys:
            residual = state_switch_residual(core, atom)
            keys[atom] = _surface_key(residual) if residual else None
        return keys[atom]

    seen: set[str] = set()
    for cond in covered:
        nested = []
        if is_call(cond):
            for pattern in (_STEP_CALL, _BEND_CALL):
                for call, _arg in _iter_step_calls(cond, pattern=pattern):
                    nested.extend(call_atoms(call))
        for atom in nested or _monotone_pieces(cond) or [cond]:
            key = surface(inline(atom))
            if key is not None:
                seen.add(key)

    # Every `if()` a rate law reads, visited once: in the rate law's own text or
    # in a function it reaches. Flattening each rate law whole would visit a
    # shared function's conditions once per rate law that reaches it.
    rate_laws = dict.fromkeys(_scan_rate_law(r) for r in ctx["functional_reactions"])
    reached: dict[str, None] = {}
    pending = [n for text in rate_laws for n in _IDENT_RE.findall(text) if n in func_map]
    while pending:
        name = pending.pop()
        if name not in reached:
            reached[name] = None
            pending.extend(n for n in _IDENT_RE.findall(func_map[name]) if n in func_map)

    conditions: list[str] = []
    # Each kept condition's surface, and the branches its readings stand in:
    # None once one reading stands in none (issue #1147).
    kept_keys: list[str] = []
    branches: dict[str, list[str] | None] = {}

    def admit(atom: str, keep: bool = True, guard: str | None = None) -> None:
        key = surface(atom)
        if key is None:
            return
        if key not in seen:
            seen.add(key)
            if keep:
                conditions.append(atom)
                kept_keys.append(key)
                branches[key] = None if guard is None else [guard]
        elif key in branches:
            had = branches[key]
            branches[key] = None if had is None or guard is None else [*had, guard]

    for text in (*rate_laws, *(func_map[name] for name in reached)):
        if not switches(text):
            continue
        # The conditions of each if(), then every comparison standing anywhere
        # else. The second pass meets the first's atoms again, and the surface
        # key below keeps one of each.
        compared = [*_iter_condition_atoms(text), *_iter_comparisons_anywhere(text)]
        for written in compared:
            # Inlining can expose more logic than the text spelled — a condition
            # `flag` over `flag() = (A>1)||(B<2)` — so the atoms are split again.
            for whole in _split_logical_atoms(inline(written)):
                if is_equality_atom(whole):
                    continue
                # A residual that turns within a step (`abs(A-5) < 0.01`) is
                # rooted at each of its edges instead (issue #904).
                for atom in _monotone_pieces(whole) or [whole]:
                    admit(atom, keep=not calls_only)
        # Every step or bend call outside the comparisons, nested ones too: the
        # abs() inside `max(0, 1 - abs(u)/w)` bends the rate law as the max()
        # does (issue #1068). One inside a comparison jumps it only through
        # that comparison, whose root is above. Each atom's text is found
        # once: written N times, it is in `compared` N times, and finding
        # every copy for every copy listed N² spans, which the index the call
        # scan puts them in then sorted (issue #1256).
        spans = [
            (start, start + len(atom))
            for atom in dict.fromkeys(compared)
            for start in (m.start() for m in re.finditer(re.escape(atom), text))
        ]
        guard_at = _BranchGuards(text).at
        for pattern in (_STEP_CALL, _BEND_CALL):
            for call, _arg, start in _iter_step_calls(
                text, pattern=pattern, skip=spans, with_start=True
            ):
                guard = guard_at(start)
                for atom in call_atoms(inline(call)):
                    admit(atom, guard=None if guard is None else inline(guard))
    if not with_guards:
        return tuple(conditions)
    guards = []
    for key in kept_keys:
        on = branches[key]
        guards.append("" if not on else (on[0] if len(on) == 1 else f"max({', '.join(on)})"))
    return tuple(conditions), tuple(guards)


# ExprTk spellings Python's parser reads differently or not at all. `if` is a
# keyword there, so the call is renamed; the rest map to their Python twins.
_SURFACE_KEY_SUBS = (
    (_IF_CALL, "_if_("),
    (re.compile(r"&&"), " and "),
    (re.compile(r"\|\|"), " or "),
)


@functools.lru_cache(maxsize=256)
def _surface_key(residual: str) -> str:
    """One spelling per crossing surface, for :func:`state_switch_root_conditions`.

    ``NetworkModel.state_switch_residual`` writes ``(lhs)-(rhs)`` from the text it
    is given, so one surface reached by two routes can come back as two strings:
    ``(time())-(((4*S)))`` against ``(time())-((4*S))``, or with the sides
    swapped when one route wrote ``b < a`` and the other ``a > b``. Python's
    parser drops redundant parentheses, and the smaller of the two operand
    orders is kept. Text it cannot parse is its own key, which only costs a
    duplicate root.

    Kept for the residuals last asked about: the key is a function of the text
    alone, and a fresh model of the same structure asks about the same ones,
    which on BIOMD0000000385 are 11k characters long (issue #1068).
    """
    import ast

    text = residual.replace("^", "**")
    for pattern, repl in _SURFACE_KEY_SUBS:
        text = pattern.sub(repl, text)
    # Anything the parser or the printer cannot manage is its own key: a sum of
    # 400 observables nests too deep for ``ast.unparse`` (RecursionError), and
    # letting that escape dropped every root of the model, not just this one.
    try:
        tree = ast.parse(text, mode="eval").body
        key = ast.unparse(tree)
        if isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Sub):
            key = min(key, ast.unparse(ast.BinOp(tree.right, ast.Sub(), tree.left)))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return residual
    return key


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
    func_map = _scan_function_map(ctx)
    texts = [
        *func_map.values(),
        *(_scan_rate_law(r) for r in ctx["functional_reactions"]),
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
    spans = _SpanIndex(_condition_spans(expr))
    for pat in (_RELATIONAL, _LOGICAL, _NOT_OP, _NOT_CALL):
        for m in pat.finditer(expr):
            if not spans.contains(m.start(), m.end()):
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
            # interval). An equality that holds a step call holds over an
            # interval too, `floor(X/thr) == 0` for X under thr, and is not
            # admitted: read as holding nowhere, it returned a column of 0 for
            # 2.46 with nothing logged.
            if is_equality_atom(atom):
                if not _STEP_CALL.search(atom) and state_switch_residual(scope.core, atom):
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
    probe = _blank_clock_refs(atom_flat, scope)
    if not any(
        m.group(0) not in scope.param_idx and not probe[m.end() :].lstrip().startswith("(")
        for m in _IDENTIFIER.finditer(probe)
    ):
        # It reads nothing but the clock and constants: a shape of time the
        # switch-time jump does not recognize (`sin(10*time) > 0.99`), not state.
        return UncompensatedCrossingReason(
            f"the condition {atom!r} reads only the time, but in a shape the issue #48 "
            "switch-time jump does not recognize (it places a crossing linear in time, or "
            "a repeating schedule), so nothing compensates the jump at its crossings"
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

    ``fixed_on_instant`` says a crossing no requested column moves is on this
    one's clock and within one instant of it (issue #767). That one emits no
    record, so the core cannot see it, and it flips with every nudge of the
    clock that reads this one. An event on the instant comes apart from it
    under any parameter that moves the event.

    ``instant_clocks`` lists the other crossings on this instant that a rate
    law reads together with this one, each as its clock and the value it
    crosses at: ``(-1, t*)`` for the time and ``(species, threshold)`` for a
    counter. The core's test of whether the conditions on an instant commute
    needs them all flipped (issue #951): a nudge of this crossing's clock does
    not flip a condition on another, and one about this crossing's own time
    does not reach one on the same clock that is a few ulp on.

    ``instant_other_clocks`` lists every crossing on the instant that is on
    another clock, in the same form, composed with this one or not. A column
    that enters a comoving frame at this crossing enters it against the whole
    right-hand side, which has to be read with those made.
    """

    t_star: float
    clock_idx0: int
    threshold: float
    dtstar: list[float]
    isolate_param_idx0: list[int]
    isolate_delta: list[float]
    fixed_on_instant: bool = False
    instant_clocks: list[tuple[int, float]] = []
    instant_other_clocks: list[tuple[int, float]] = []


class _Crossing(NamedTuple):
    """A detected clock threshold, before it is decided whether it emits."""

    t_star: float
    clock_idx0: int
    threshold: float
    dtstar: list[float]
    # ∂threshold/∂primary over ALL primaries, not just the requested columns,
    # and primaries ONLY (:func:`_crossing_identity`). This is the crossing's
    # identity: two spellings of one threshold share it, two thresholds that
    # merely share a value do not (issue #375). ``dtstar`` above is where a
    # requested derived parameter's own-terms partial lands (issue #475), so the
    # column a caller reads and the identity two crossings are compared on stay
    # separate questions.
    partials: dict[str, float]
    # The least and the greatest of the values folded into this crossing, on
    # its clock, where they are not all one: two thresholds that agree to 12
    # digits are one crossing (:func:`_is_same_crossing`) and need not be one
    # double. ``threshold`` is the first one found, which depends on the order
    # the rate laws are written in; what is on the instant does not.
    lo: float | None = None
    hi: float | None = None
    # The condition atoms, and step calls, this crossing was found from. Which
    # rate laws read them decides whether two crossings on one instant can be
    # composed at all (:func:`_emit_switch_records`). Empty where that is not
    # known, which is read as "any of them".
    atoms: frozenset[str] = frozenset()


def _on_clock(cross: _Crossing) -> float:
    """Where ``cross`` sits on its own clock: a time, or a counter's value."""
    return cross.threshold if cross.clock_idx0 >= 0 else cross.t_star


def _extent(cross: _Crossing) -> tuple[float, float]:
    """The least and the greatest value folded into ``cross``, on its clock."""
    at = _on_clock(cross)
    return (at if cross.lo is None else cross.lo, at if cross.hi is None else cross.hi)


def _q(x: float) -> str:
    """A relative-precision key for a float, for grouping crossings."""
    return f"{x:.12g}"


def _crossing_identity(
    partials: dict[str, float], scope: SwitchConditionScope
) -> dict[str, float]:
    """*partials* reduced to the primaries, which is what identifies a crossing.

    Since issue #475 a requested *derived* parameter also gets a partial, and
    that one must not reach :func:`_is_same_crossing`: two spellings of one
    threshold — ``sigma`` and the ``t0 + t_delta`` it is defined as — reach the
    same instant through the same primaries and are one crossing, but only one
    of them names the derived parameter. Letting that into the identity would
    split them, and two crossings on one instant go down the issue #375
    isolation path, which can refuse a model that has nothing wrong with it.
    """
    return {n: v for n, v in partials.items() if n in scope.primary_names}


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
            lo = min(*_extent(seen), _on_clock(cand))
            hi = max(*_extent(seen), _on_clock(cand))
            atoms = seen.atoms | cand.atoms if seen.atoms and cand.atoms else frozenset()
            found[i] = seen._replace(dtstar=merged, lo=lo, hi=hi, atoms=atoms)
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
    composed: Sequence[_Crossing] = (),
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
    # which is the very contamination being removed. The thresholds of this
    # group are not such neighbours. The parameter bumped is one none of them
    # reads, so they stay where they are, and since issue #737 a group can hold
    # two thresholds that differ in their last digits.
    grouped = {other.threshold for other in group if other.clock_idx0 == cross.clock_idx0}
    gaps = [
        abs(t - cross.threshold)
        for t in thresholds_on_clock
        if t not in grouped and _q(t) != _q(cross.threshold)
    ]
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
    # The core reads at the whole hair, at half of it and, where it asks
    # whether the instant commutes, at a quarter, with the clock a nudge past
    # the crossing — or, where a rate law reads it together with others on the
    # instant (``composed``), a nudge past the latest of those. The shortest
    # reading has to clear that, or it has this condition already made: with
    # two gates of its own law 2,000 and 9,000 ulp after a switch the hair is
    # capped at 2,252 ulp, half of it is short of 2,064, and the column came
    # back +1 for −1.
    extent = max(
        (
            abs(at - _on_clock(cross))
            for other in composed
            if other.clock_idx0 == cross.clock_idx0
            for at in _extent(other)
        ),
        default=0.0,
    )
    # The quarter is read only where the crossing is composed with another.
    shortest = 4.0 if any(other is not cross for other in composed) else 2.0
    floor = max(floor, shortest * (extent + 2.0 * _INSTANT_ULPS * _EPS * span))
    if delta_threshold < floor:
        raise SensitivityUnsupportedError(
            "Forward sensitivity is not supported on this model: the switch times "
            f"crossing together at t={cross.t_star:.6g} sit closer to a neighbouring "
            "threshold than the step that would separate them, so isolating this "
            "crossing's jump would flip a condition it does not own (issue #375). "
            "bngsim refuses rather than return a merged jump."
        )
    return param_idx[name], delta_threshold / private[name]


# How many (step call, moved crossing) pairs are evaluated one at a time, for
# a step call whose edges cannot be listed. Each is two sympy evaluations.
_STEP_PROBE_PAIRS = 2048


def _steps_on_instants(
    core,
    ctx,
    scope: SwitchConditionScope,
    t_start: float,
    t_end: float,
    moved: Sequence[_Crossing],
) -> list[_Crossing]:
    """The step calls outside every condition that step on a moved crossing's
    instant, each as a crossing no parameter moves (issue #944).

    A call whose edges :func:`_step_edge_stop_times` lists, ``floor(time/3)``
    or the sawtooth of a dose ramp, is asked from that list, which the stops
    of the run already paid for: an edge within the core's nudge of a moved
    crossing is on its instant. The window is taken a nudge past its end,
    because an edge on the last instant of a run is one too.

    A call it cannot list, ``floor(time^2/9)``, ``floor(exp(time - 3))`` or
    one with more edges in the window than are listed, is evaluated a nudge
    before and a nudge after each moved crossing instead, which asks the one
    thing that matters here whatever the call is. A call that cannot be
    evaluated, or that reads two clocks, is taken to step: reading a crossing
    apart from a step that is not there costs nothing.

    Raises
    ------
    SensitivityUnsupportedError
        Where the calls that have to be evaluated and the moved crossings are
        more than :data:`_STEP_PROBE_PAIRS` pairs. Leaving the rest unasked
        would be right for the first two thousand crossings and silently wrong
        after them.
    """
    calls = time_discontinuity_conditions(core, ctx, steps_only=True)
    if not calls:
        return []
    times = sorted({cross.t_star for cross in moved})

    def reach(t: float, offset: float) -> float:
        # The nudge of the clock the call reads: a counter is nudged about its
        # own value, which is the time plus where the counter started.
        return (_INSTANT_ULPS + 4.0) * _EPS * max(abs(t), abs(t + offset), 1.0)

    def on(t: float, clock_idx: int, offset: float, call: str) -> _Crossing:
        return _Crossing(
            t_star=t,
            clock_idx0=clock_idx,
            threshold=t + offset if clock_idx >= 0 else t,
            dtstar=[],
            partials={},
            atoms=frozenset({call}),
        )

    out: list[_Crossing] = []
    pairs = 0
    for call in calls:
        rewrite = _rewrite_counter_clock(core, call, scope, t_start)
        if rewrite is None:
            out.extend(on(t, -1, 0.0, call) for t in times)
            continue
        text, clock_idx, offset = rewrite
        edges = _step_edge_stop_times(text, scope, t_start, t_end + 2.0 * reach(t_end, offset))
        if edges is not None:
            listed = sorted(edges)
            for t in times:
                i = bisect.bisect_left(listed, t)
                near = [listed[j] for j in (i - 1, i) if 0 <= j < len(listed)]
                if any(abs(edge - t) <= reach(t, offset) for edge in near):
                    out.append(on(t, clock_idx, offset, call))
            continue
        pairs += len(times)
        if pairs > _STEP_PROBE_PAIRS:
            raise SensitivityUnsupportedError(
                "Forward sensitivity is not supported on this run: the rate law steps at "
                f"{call!r}, "
                "a step call whose edges bngsim cannot list, and the requested parameters move "
                f"{len(times)} switch times in the reported window. Whether one of them lands on "
                "a step is asked by evaluating the call beside each, and that is not done for "
                f"more than {_STEP_PROBE_PAIRS} of them: the rest would be read with the step "
                "inside their bracket (issue #944). Shorten the reported time window, or drop "
                "the parameters that move the switch times from sensitivity_params."
            )

        def at(t: float, text: str = text) -> float | None:
            try:
                return _evaluate_threshold(
                    _TIME_REF.sub(f"({t!r})", text),
                    scope.param_idx,
                    scope.values,
                    scope.derived_exprs,
                )
            except Exception:  # noqa: BLE001 - a call this cannot read is taken to step
                return None

        for t in times:
            before, after = at(t - reach(t, offset)), at(t + reach(t, offset))
            if before is None or after is None or before != after:
                out.append(on(t, clock_idx, offset, call))
    return out


def _clocks_moved(
    core,
    clocks,
    names: Sequence[str],
    ic_species: Sequence[int],
    by_text: bool = False,
    declared: Mapping[str, Mapping[str, float]] | None = None,
) -> frozenset[int]:
    """The clock species a requested column moves (issue #725).

    A clock moves with its own initial-condition axis, with a parameter its IC
    is seeded from, and with a parameter its rate law reads (inlined through
    functions and derived parameters): BNG2.pl's ``_rateLaw1`` for
    ``0 -> counter() 1``, which compute_all_sensitivities requests by default.

    ``by_text`` reads which parameter a clock's IC is seeded from off the
    text, every name its IC parameter is written in, and not off the seed's
    value: no fewer clocks than the seeds give, and nothing differentiated.
    The refusal of :func:`fallback_crossing` asks it that way, each time a
    parameter is set.

    ``declared`` are the rows a caller gave with
    ``Model.declare_ic_sensitivity``: a clock started by hand at a value that
    moves with a parameter. The run is seeded with them, and a clock they
    move was not taken as moved: its crossing's jump was not applied, and
    dY/dn came back 0 for 3.
    """
    from bngsim._codegen import compute_ic_param_sens_seed
    from bngsim._jacobian import _IDENT_RE, _inline_functions

    idx = {i for i in clocks.values() if i >= 0}
    if not idx:
        return frozenset()
    wanted = set(names)
    moved = {i for i in idx if i in set(ic_species)}
    if declared and wanted:
        species = list(core.species_names)
        for name, row in declared.items():
            if name in species and any(row.get(p, 0.0) != 0.0 for p in wanted):
                moved |= {species.index(name)} & idx
    if wanted and idx - moved:
        pnames = list(core.param_names)
        data = core.codegen_data()
        params = list(data.get("parameters", ()))
        func_map = {f["name"]: f["expression"] for f in data.get("functions", ())}
        derived = {p["name"]: p["expression"] for p in params if p.get("expression")}
        if by_text:
            # As each derived parameter is written now: one that ``set_param``
            # has overridden is a number, and what it was written in no
            # longer reaches the clock.
            written = {
                name: text
                for name, is_expr, text in zip(
                    pnames, core.param_is_expression, core.param_expressions, strict=True
                )
                if is_expr and text
            }
            seeded = [
                (sp, name)
                for sp, pidx in core.species_ic_param_refs
                if sp in idx and 0 <= pidx < len(pnames)
                for name in _written_in(pnames[pidx], written)
            ]
            seeded += [
                (sp, pnames[pidx])
                for sp, pidx, _coeff in getattr(core, "compartment_ic_sens_seeds", ())
                if sp in idx and 0 <= pidx < len(pnames)
            ]
            moved |= {sp for sp, name in seeded if name in wanted}
        else:
            for sp, pidx, _coeff in compute_ic_param_sens_seed(core):
                if sp in idx and 0 <= pidx < len(pnames) and pnames[pidx] in wanted:
                    moved.add(sp)

        def reads(text: str, depth: int = 0) -> bool:
            if depth > 64:
                return True
            for tok in _IDENT_RE.findall(text):
                if tok in wanted:
                    return True
                if tok in derived and reads(derived[tok], depth + 1):
                    return True
            return False

        for rxn in data.get("reactions", ()):
            produced = {
                i
                for i in idx - moved
                if list(rxn.get("products", ())).count(i) > list(rxn.get("reactants", ())).count(i)
            }
            if not produced:
                continue
            if rxn.get("type") == "functional":
                body = func_map.get(rxn.get("function_name") or "", "")
                text = _inline_functions(body, func_map) or body
            else:
                ids = [i for i in rxn.get("rate_param_indices", ()) if 0 <= i < len(params)]
                text = " ".join(params[i]["name"] for i in ids)
            if reads(text):
                moved |= produced
    return frozenset(moved)


# The core reads a crossing by nudging its clock this many ulp either side of
# the threshold (``kSwitchInstantUlps`` in cvode_simulator.cpp), and treats two
# times within that reach as one instant.
_INSTANT_ULPS = 64.0


# The least time scale the instants of a run with no sensitivity columns are
# measured on (``kPlainInstantFloor`` in cvode_simulator.cpp).
_PLAIN_INSTANT_FLOOR = 1e-125


def _same_instant(a: float, b: float, floor: float = 1.0) -> bool:
    """Whether the core's clock nudge about ``a`` can reach ``b`` (issue #737).

    The core's reach and a few ulp, so that a crossing time this module and the
    core round differently in the last place is still on the safe side. Not
    more: a pair grouped here has to be isolated, and two thresholds that share
    their only parameter cannot be, so every pair grouped beyond the nudge's
    reach is a run refused for nothing.

    *floor* is the least scale the reach is measured on: 1 for the nudge, and
    :data:`_PLAIN_INSTANT_FLOOR` for a run with no sensitivity columns, whose
    loop takes a stop as reached within doubles of the time itself.
    """
    return abs(a - b) <= (_INSTANT_ULPS + 4.0) * _EPS * max(abs(a), abs(b), floor)


def _instant_groups(found: Sequence[_Crossing]) -> list[list[_Crossing]]:
    """The crossings the core cannot read one at a time, grouped.

    Two crossings belong together when the nudge that reads one flips the other
    as well: their times are one instant, or they threshold one counter clock at
    values within the nudge of each other (a counter that starts at 1e6 is
    nudged by 1e-8, whatever the time). The 12-digit key :func:`_q` is kept as a
    third reason, so nothing grouped before issue #737 is split.

    ``_q`` alone was not enough. It is a bucket, not a distance: two times an
    ulp apart can fall either side of a bucket edge, and below t = 1 its buckets
    are narrower than the nudge. Either way two crossings the nudge flips
    together went ungrouped, and each was charged with the other's jump.
    """
    parent = list(range(len(found)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def chain(order: list[int], value) -> None:
        for i, j in zip(order, order[1:], strict=False):
            if _same_instant(value(found[i]), value(found[j])):
                parent[root(j)] = root(i)

    everything = list(range(len(found)))
    chain(sorted(everything, key=lambda i: found[i].t_star), lambda c: c.t_star)
    by_clock: dict[int, list[int]] = {}
    by_key: dict[str, int] = {}
    for i, cross in enumerate(found):
        if cross.clock_idx0 >= 0:
            by_clock.setdefault(cross.clock_idx0, []).append(i)
        first = by_key.setdefault(_q(cross.t_star), i)
        parent[root(i)] = root(first)
    for members in by_clock.values():
        chain(sorted(members, key=lambda i: found[i].threshold), lambda c: c.threshold)

    groups: dict[int, list[_Crossing]] = {}
    for i, cross in enumerate(found):
        groups.setdefault(root(i), []).append(cross)
    return list(groups.values())


def _emit_switch_records(
    found: list[_Crossing],
    param_idx: dict[str, int],
    moved_clocks: frozenset[int] = frozenset(),
    atom_laws: Mapping[str, AbstractSet[int]] | None = None,
) -> list[SwitchCrossing]:
    """Turn detected crossings into records, isolating any that coincide.

    ``atom_laws`` says which rate laws read each condition. Two crossings on
    one instant can be composed only where some rate law reads both: a
    reaction's rate is one law, and the right-hand side adds reactions up.
    Each record lists, in ``instant_clocks``, the others on its instant that a
    rate law reads together with it; where there are none the core has
    nothing to ask about commuting, and reads the crossing's jump as it
    always did. Without ``atom_laws`` every crossing on the instant is listed.

    A crossing no requested parameter moves emits nothing — but it is still in
    ``found``, and its presence on an instant is exactly what puts the crossings
    that DO emit there onto the isolation path. That is why issue #375
    reproduced with a single parameter requested: the coinciding condition
    contaminates the core's ``f⁻`` whether or not anyone asked about it.
    """
    thresholds_on_clock: dict[int, set[float]] = {}
    for cross in found:
        thresholds_on_clock.setdefault(cross.clock_idx0, set()).add(cross.threshold)

    def emits(cross: _Crossing) -> bool:
        # No requested column moves a crossing that does not emit: not its
        # threshold, and not the clock it is read on (issue #725). One on a
        # moved clock is kept even inside a coinciding group, isolated below by
        # its own threshold's parameters like any other member.
        return any(v != 0.0 for v in cross.dtstar) or cross.clock_idx0 in moved_clocks

    def laws(cross: _Crossing) -> AbstractSet[int] | None:
        if atom_laws is None or not cross.atoms:
            return None
        read: set[int] = set()
        for atom in cross.atoms:
            if atom not in atom_laws:
                return None
            read |= atom_laws[atom]
        return read

    def one_law(a: _Crossing, b: _Crossing) -> bool:
        la, lb = laws(a), laws(b)
        return la is None or lb is None or bool(la & lb)

    def one_nudge(a: _Crossing, b: _Crossing) -> bool:
        if a.clock_idx0 != b.clock_idx0:
            return False
        if a.clock_idx0 >= 0:
            return _same_instant(a.threshold, b.threshold)
        return _same_instant(a.t_star, b.t_star)

    records: list[SwitchCrossing] = []
    for group in _instant_groups(found):
        for cross in group:
            if not emits(cross):
                continue
            if len(group) > 1:
                idx0, delta = _isolation_bump(
                    cross,
                    group,
                    param_idx,
                    thresholds_on_clock[cross.clock_idx0],
                    [other for other in group if other is cross or one_law(other, cross)],
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
                    fixed_on_instant=any(
                        other is not cross and not emits(other) and one_nudge(other, cross)
                        for other in group
                    ),
                    instant_clocks=_composed_with(cross, group, one_law),
                    instant_other_clocks=sorted(
                        {
                            (other.clock_idx0, at)
                            for other in group
                            if other.clock_idx0 != cross.clock_idx0
                            for at in _extent(other)
                        }
                    ),
                )
            )
    records.sort(key=lambda r: r.t_star)
    return records


def _composed_with(
    cross: _Crossing, group: Sequence[_Crossing], one_law
) -> list[tuple[int, float]]:
    """The other crossings in ``group`` that a rate law reads together with
    ``cross``, each as its clock and the least and greatest value folded into
    it; with ``cross``'s own extent where there is one and it is not a point."""
    others = [other for other in group if other is not cross and one_law(other, cross)]
    if not others:
        return []
    return sorted(
        {(other.clock_idx0, at) for other in others for at in _extent(other)}
        | {(cross.clock_idx0, at) for at in _extent(cross) if at != _on_clock(cross)}
    )


def _columns_of(names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Every sensitivity column each requested name occupies (issue #759).

    ``sensitivity_params`` may name a parameter twice — a list built by
    concatenation does it easily — and each occurrence is a column of its own
    that must carry the same ∂t*/∂p as the first. A ``{name: column}`` map keeps
    only the last occurrence, which left the earlier ones without the crossing
    jump: all zeros for a pure switch or event time.
    """
    cols: dict[str, list[int]] = {}
    for c, name in enumerate(names):
        cols.setdefault(name, []).append(c)
    return {name: tuple(c) for name, c in cols.items()}


def _dtstar_columns(
    partials: dict[str, float], col_of: dict[str, tuple[int, ...]], n_cols: int
) -> tuple[list[float], bool]:
    """Spread ``∂t*/∂primary`` over the requested columns (see :func:`_columns_of`).

    Also returns whether any requested column received a non-zero partial.
    """
    dtstar = [0.0] * n_cols
    moved = False
    for prim_name, coeff in partials.items():
        if coeff == 0.0:
            continue
        for col in col_of.get(prim_name, ()):
            dtstar[col] += float(coeff)
            moved = True
    return dtstar, moved


def _absorb_schedule_crossings(
    found: list[_Crossing],
    found_index: dict[tuple[int, str], list[int]],
    atom: str,
    scope: SwitchConditionScope,
    core,
    col_of: dict[str, tuple[int, ...]],
    t_start: float,
    t_end: float,
    n_cols: int,
    own_names: AbstractSet[str] = frozenset(),
    uncompensated_out: list[str] | None = None,
) -> None:
    """Add every edge of *atom*'s repeating schedule that falls in the run window
    (issue #436), or do nothing when *atom* is not one.

    ``uncompensated_out`` is handed *atom* where it is a schedule whose period,
    offset or duty does not resolve at the model's values now (issue #708).

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
    terms = _periodic_schedule_terms(atom, sched, scope, own_names)
    if terms is None:
        # Unreadable. The gate declines a model over it, so a build at these
        # values has no analytic RHS and there is nothing to compensate here.
        # A build from before a write may have one (issue #708): the caller is
        # told.
        if uncompensated_out is not None:
            uncompensated_out.append(atom)
        return
    if not terms.crosses:
        # A schedule that never turns over — `time() - P*floor(time()/P) >= 0`
        # is true at every instant of the run.
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
        dtstar, _ = _dtstar_columns(partials, col_of, n_cols)
        _absorb_crossing(
            found,
            _Crossing(
                t_star=t_star,
                clock_idx0=clock_idx0,
                threshold=value,
                dtstar=dtstar,
                partials=_crossing_identity(partials, scope),
                atoms=frozenset({atom}),
            ),
            found_index,
        )


def compute_switch_time_sens(
    core,
    sens_param_names,
    t_start: float,
    t_end: float,
    has_analytic_sens_rhs: bool = False,
    ic_species: Sequence[int] = (),
    declared: Mapping[str, Mapping[str, float]] | None = None,
    uncompensated_out: list[str] | None = None,
) -> tuple[list[SwitchCrossing], list[int]]:
    """Switch-time crossings and their ``∂t*/∂p``, plus the parameters to pin.

    ``uncompensated_out``, where given, is handed each clock-threshold
    condition and each repeating schedule this pass could not compensate at the
    model's values now (issue #708): the rule is the one the codegen gate
    admits a condition by, so one listed here in a run on the analytic
    right-hand side was admitted at other values, when the code was built. The
    pass reads every function, and the gate the rate laws alone: a caller
    confirms with :func:`model_uncompensated_crossing_reason`.

    ``ic_species`` are the 0-based species of the run's initial-condition
    columns. A counter clock's crossing moves with the clock's own sensitivity
    as well as its threshold's (issue #725), so a crossing on a clock that a
    requested column moves is recorded even when no column moves the threshold;
    the core subtracts the clock's own sensitivity from each column there.

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
    if (not names and not ic_species) or core.n_functions == 0:
        return [], []

    ctx = core.functional_jacobian_context()
    # Cheapest possible gate, checked before anything else: a model with no
    # conditional rate law has no switch to find. On the RAW bodies, so a
    # genome-scale model with tens of thousands of functions and no `if()`
    # short-circuits before any inlining — inlining cannot create an `if()` that
    # no body already has, and neither can spelling a `%` as `mod()`. Keeps the
    # RHS probes and the sympy work off the path of every ordinary sensitivity
    # run, including the per-row batch path.
    if not any(_IF_CALL.search(body) for body in dict(ctx["function_map"]).values()):
        return [], []
    func_map = _scan_function_map(ctx)
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
    col_of = _columns_of(names)
    # A requested parameter that is itself derived gets a column of its own, on
    # the same terms the smooth half of that column already uses: the derivative
    # of writing that slot, holding what it is built from (issue #475). It used
    # to get a hard 0 here while `bngsim_dfdp` gave it a real in-branch term, so
    # a model whose switch time was written as a derived parameter — `sigma =
    # t0 + t_delta` is the shape four hand-written corpus models use — reported
    # a column that was missing exactly its crossing jump.
    own_names = frozenset(names) & set(derived_exprs)

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
    # The rate laws that read each atom, by position in ``function_bodies``,
    # which are inlined: a law that reads a condition through another function
    # reads it.
    atom_laws: dict[str, set[int]] = {}

    for law, body in enumerate(function_bodies):
        for cond in _iter_if_conditions(body):
            for atom in _split_logical_atoms(cond):
                atom_laws.setdefault(atom, set()).add(law)
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
                # Issue #545: a threshold written through clock guards, one crossing
                # candidate per cell, asked in the same order clock_crossing_compensated
                # asks it.
                guarded = None
                if split is None:
                    guarded = _clock_guarded_thresholds(atom, clock_symbols)
                    if guarded is not None:
                        split = (guarded[0], [g.threshold for g in guarded[1]])
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
                        own_names,
                        uncompensated_out,
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
                terms = [_threshold_crossing_terms(e, scope, own_names) for e in threshold_exprs]
                compensated = all(t is not None for t in terms)
                if not compensated and uncompensated_out is not None:
                    uncompensated_out.append(atom)

                for k, (threshold_expr, term) in enumerate(
                    zip(threshold_exprs, terms, strict=True)
                ):
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
                    if guarded is not None and not guarded[1][k].holds_at(threshold_value):
                        # A cell's threshold that names a clock value outside that
                        # cell: the guards select another branch there, and the
                        # condition does not flip (issue #545).
                        continue
                    dtstar, _ = _dtstar_columns(partials, col_of, len(names))

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
                            partials=_crossing_identity(partials, scope),
                            atoms=frozenset({atom}),
                        ),
                        found_index,
                    )

    # Issue #944: a step call on a clock outside every condition is on its
    # instants too. The loop above reads conditions, and `floor(time/3)` as a
    # factor of a rate law is not one: each of its steps was placed as a stop
    # (#869) and not grouped with the records, so a fitted switch that landed on
    # one was read with the step inside its bracket, and took the step's jump
    # into its own column (dW/dtau = −4.2 for 0). A step that sits on a moved
    # crossing's instant is absorbed here as a crossing no parameter moves,
    # which is what puts the record onto the isolation path.
    moved_clocks = _clocks_moved(core, clocks, names, ic_species, declared=declared)
    moved = [c for c in found if any(v != 0.0 for v in c.dtstar) or c.clock_idx0 in moved_clocks]
    if moved:
        for step in _steps_on_instants(core, ctx, scope, float(t_start), float(t_end), moved):
            for call in step.atoms:
                if call not in atom_laws:
                    atom_laws[call] = {
                        law for law, body in enumerate(function_bodies) if call in body
                    }
            _absorb_crossing(found, step._replace(dtstar=[0.0] * len(names)), found_index)

    # A rate law that reads ``rateOf(X)`` reads every rate law that changes X,
    # which the text of neither says: a gate in X's reaction and one in the
    # law that reads its rate are composed. With `rateOf` in the model every
    # crossing on an instant is taken as read together with every other.
    reads_rates = bool(getattr(core, "uses_rateof", False))
    records = _emit_switch_records(
        found, param_idx, moved_clocks, None if reads_rates else atom_laws
    )
    if not records:
        return [], []

    # Every parameter that moves at least one detected crossing is a switch-time
    # parameter, and must be pinned against CVODES' FD probe — pinning is both
    # what makes ∂f/∂p come out as the correct 0 and what keeps the probe from
    # displacing the switch into the approach (which stalls the integrator).
    switch_params = {
        names[c] for record in records for c in range(len(names)) if record.dtstar[c] != 0.0
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
        for name, body in _scan_function_map(ctx).items()
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
    col_of = _columns_of(names)
    # Every body a trigger symbol can stand for, resolvable or not — used only
    # to decide whether an UNREDUCED trigger still depends on a requested
    # parameter. Kept separate from `thresholds.exprs`, which admits only the
    # bodies that reduce to arithmetic over primaries.
    all_bodies = dict(scope.derived_exprs)
    all_bodies.update({n: b for n, b in _scan_function_map(ctx).items() if b})
    requested = set(names)
    # A requested parameter that is itself derived gets a column of its own, on
    # the same terms the rate-law switch times use (issue #475). Restricted to
    # genuinely derived names: `thresholds.exprs` also holds the rule-bound
    # parameters, and a column for one of those is refused outright elsewhere
    # because a function owns the slot (issues #227, #329).
    own_names = requested & set(scope.derived_exprs)

    records: list[tuple[int, list[float]]] = []
    compensated: list[int] = []
    reasons: dict[int, str] = {}
    blocked: dict[int, str] = {}

    for ei in range(min(n_events, len(triggers))):
        trigger = triggers[ei] or ""
        analysis = _analyze_event_trigger(
            core, trigger, scope, thresholds, clocks, clock_symbols, t_start, own_names
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
        dtstar, moved = _dtstar_columns(partials, col_of, len(names))
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
    own_names: AbstractSet[str] = frozenset(),
):
    """Reduce an event trigger to its rising edge.

    Returns ``(t_star, {name: ∂threshold/∂name})`` when the trigger is a
    conjunction of clock thresholds with a well-defined false→true edge,
    ``None`` when it is such a conjunction but has no rising edge (no lower
    bound, or an empty true-interval), or a reason string when it cannot be
    reduced at all.

    ``own_names`` are derived parameters the caller wants a column for, read the
    way :func:`_threshold_crossing_terms` reads them: as an independent input
    rather than chain-ruled away into what the name is defined from (issue
    #475). They join the returned partials and nothing else — the kink test
    below still compares two atoms on their primaries alone, because two
    spellings of one threshold reach the same instant through the same
    primaries and only one of them names the derived parameter.
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

    # (t_star, ∂threshold/∂primary, ∂threshold/∂(requested derived name)).
    # The last is kept apart from the second so the kink test below can
    # compare two atoms on the primaries alone (issue #475).
    lower: list[tuple[float, dict[str, float], dict[str, float]]] = []
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
            {m.group(0) for m in _IDENTIFIER.finditer(thr_flat)}
            - thresholds.primaries
            - _BUILTIN_CONSTANT_VALUES.keys()
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
        wanted = {n for n in own_names if n in thresholds.exprs}
        walked = _derived_expr_partials_numeric(
            threshold_expr,
            set(thresholds.primaries),
            scope.param_idx,
            list(scope.values),
            thresholds.exprs,
            warn_on_failure=False,
            include_derived=bool(wanted),
        )
        partials = {n: v for n, v in walked.items() if n in thresholds.primaries}
        own = {n: v for n, v in walked.items() if n in wanted}
        lower.append((t_atom, partials, own))

    if not lower:
        # `time <= toff` alone: true from t_start, so the event fires at the
        # run's start (or not at all) and its firing time does not move.
        return None
    t_star = max(t for t, _p, _o in lower)
    winners = [(p, o) for t, p, o in lower if t == t_star]
    # Compared on the primaries only, for the reason in the docstring: a derived
    # partial belongs to a column, never to the question of whether two atoms
    # describe the same edge (issue #475).
    if len(winners) > 1 and any(w[0] != winners[0][0] for w in winners[1:]):
        return (
            f"two atoms of the trigger {trigger!r} put the rising edge at the same time "
            f"t={t_star:.6g} with different derivatives, so t*(p) has a kink there rather "
            "than a derivative"
        )
    if upper and t_star > min(upper):
        return None  # the true-interval is empty; the event never fires
    return t_star, {**winners[0][0], **winners[0][1]}


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
