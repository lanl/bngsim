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
import re
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import NamedTuple

from bngsim._codegen import (
    _derived_expr_partials_numeric,
    _find_close_paren_strict,
    _inline_derived_param_refs,
    _split_top_level_commas,
)

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
    :func:`_clock_threshold_split` nor :func:`state_switch_residual` claims, so
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

    Both the function bodies and the elementary rate constants are checked, each
    with derived-parameter references inlined first so a threshold spelled
    ``sigma = t0 + t_delta`` is attributed to ``t0``/``t_delta``.
    """
    if not candidates:
        return set()
    data = core.codegen_data()
    params = data["parameters"]

    pats = {
        name: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        for name in candidates
    }
    pure = set(candidates)

    for fn in data["functions"]:
        flat = _inline_derived_param_refs(fn.get("expression", ""), derived_exprs)
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
    # synthesized holder for the function's value, whose body is already covered
    # by the function scan above.)
    for rxn in data["reactions"]:
        if rxn.get("type") != "elementary":
            continue
        for pidx in rxn.get("rate_param_indices", []):
            if not (0 <= pidx < len(params)):
                continue
            p = params[pidx]
            flat = _inline_derived_param_refs(p.get("expression", "") or p["name"], derived_exprs)
            for name in list(pure):
                if pats[name].search(flat):
                    pure.discard(name)

    return pure


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
    n_species = core.n_species
    if n_species == 0:
        return {}
    conc = [core.get_concentration(name) for name in core.species_names]
    try:
        deriv = core._eval_rhs(0.0, conc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: RHS probe for clock detection failed: %s", exc)
        return {}
    clock_idx = {i for i in range(n_species) if deriv[i] == _CLOCK_SLOPE}
    if not clock_idx:
        return {}
    # Confirm the slope is state-independent: a species whose RHS merely happens
    # to equal 1 at the initial state is not a clock. Probing at a perturbed
    # state is enough to reject every state-dependent rate law.
    probe = [c + 1.0 for c in conc]
    try:
        deriv2 = core._eval_rhs(1.0, probe)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("switch-time: second RHS probe failed: %s", exc)
        return {}
    clock_idx = {i for i in clock_idx if deriv2[i] == _CLOCK_SLOPE}
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


def _clock_threshold_split(atom: str, clock_symbols: AbstractSet[str]) -> tuple[str, str] | None:
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
    """:func:`_clock_threshold_split` plus which side of the threshold is true.

    Returns ``(clock_symbol, threshold_expr, "lower" | "upper")``, or ``None``
    when the atom is not a clock-versus-threshold comparison *or* its operator
    does not cut the time axis into a half-line. ``==`` / ``!=`` are rejected on
    the second ground: an equality on a continuous clock is true on a measure-
    zero set that the root finder cannot reliably straddle, so there is no
    well-defined rising edge to differentiate.

    Used only by the issue #49 event-time detector. The issue #48 rate-law path
    keeps calling :func:`_clock_threshold_split`, which is deliberately
    orientation-blind: an ``if()`` branch flips at the threshold whichever way
    the comparison points, and the core reads f⁻/f⁺ by evaluating the RHS on
    each side.
    """
    split = _clock_threshold_split(atom, clock_symbols)
    if split is None:
        return None
    op_split = _relational_split_op(atom)
    if op_split is None:  # pragma: no cover - _clock_threshold_split implies one
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
    return SwitchConditionScope(
        core=core,
        clocks=clocks,
        clock_symbols=frozenset(clocks) | _TIME_SYMBOLS,
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
    )


# An identifier, excluding the exponent letter of a numeric literal (`1e5`) and
# a member-ish suffix. An atom with none of these is a comparison between
# literals — a compile-time constant, with no crossing at all.
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*")


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
    whole in-branch story and the analytic RHS is admitted. What is left in this
    class is the crossing nothing compensates — a comparison inside a call
    argument, an equality, a comparison outside an ``if()`` head, or a clock
    threshold that does not reduce to a constant. A conjunction is not one of
    them, and neither is a negation: :func:`_split_logical_atoms` reduces both to
    the surfaces underneath, so ``not((X<hi) and (X>lo))`` is admitted on ground
    2 exactly as ``(X<hi) and (X>lo)`` is (issue #234).

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

    Deliberately *not* routed through :func:`_clock_threshold_split`, which
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


def fixed_time_crossings(core, t_start: float, t_end: float, conditions=()) -> list[float]:
    """Times in ``(t_start, t_end]`` at which a registered time discontinuity
    flips, for ``SolverOptions.set_crossing_stop_times`` (issue #305).

    The core stops the integration step exactly on each of these. That is not a
    refinement of the GH #72 root — it is what makes the root reachable at all.
    CVODE tests for a root only on a step it **accepts**, and where the branch
    jump is large enough that the error test rejects every step spanning the
    crossing, the accepted steps land short, ``t`` creeps to the last double
    below ``t*``, and every remaining step is under one ulp: ``t + h == t``,
    with ``g`` never once evaluated past the crossing. On Weber_BMC2015 that
    kills 6% of a fitting box outright, with zero root returns in the whole run.

    Empty (so: no change to any model's stepping) unless the model registered a
    discontinuity trigger AND its crossing time is a constant of the run.
    Resolution reads the *current* parameter values, so a condition is answered
    for the phase it is asked in — the same experimental-condition parameter can
    put the crossing inside the window in one phase and outside it in another,
    and stopping at a time that phase has no crossing at is a pure perturbation
    of its stepping.
    """
    if not conditions or core.n_discontinuity_triggers == 0:
        return []
    ctx = core.functional_jacobian_context()
    scope = switch_condition_scope(core, ctx)
    aliases = _time_alias_bodies(ctx)
    out: list[float] = []
    for cond in conditions:
        t_star = _crossing_time_of_condition(cond, scope, t_start, t_end, aliases)
        if t_star is None or not (t_start < t_star <= t_end):
            continue
        if not any(abs(t_star - seen) <= 1e-12 * max(abs(t_star), 1.0) for seen in out):
            out.append(t_star)
    out.sort()
    return out


def fixed_clock_threshold(atom: str, scope: SwitchConditionScope) -> bool:
    """True when *atom* is a clock threshold against a value nothing moves.

    ``t < 14`` — or ``t < half_life`` where ``half_life`` reduces to literals —
    crosses at a time that is the same for every parameter, so ``∂t*/∂p`` is
    exactly 0 and the crossing contributes no jump to any sensitivity column.
    That is what makes it harmless twice over: :func:`clock_crossing_compensated`
    admits it because there is nothing to compensate, and
    :func:`model_moving_crossings` excludes it because there is nothing for the
    difference-quotient fallback to miss either. One definition, two readers, so
    they cannot answer the same question differently (issue #232).
    """
    split = _clock_threshold_split(atom, scope.clock_symbols)
    if split is None:
        return False
    thr_flat = _inline_derived_param_refs(split[1], scope.derived_exprs) or split[1]
    return not any(scope.param_pats[n].search(thr_flat) for n in scope.param_names)


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


def clock_crossing_compensated(atom: str, scope: SwitchConditionScope) -> bool:
    """Will :func:`compute_switch_time_sens` account for *atom*'s crossing?

    True on two grounds, both of which mean the issue #48 machinery leaves
    nothing for anyone else to do:

    * the atom is a clock threshold whose value and partials both resolve, so
      the detector emits a record, the solver stops at ``t*`` and applies
      ``(f⁻−f⁺)·∂t*/∂p`` there;
    * the atom is a clock threshold against a *literal* (``t<14``), so ``∂t*/∂p``
      is exactly 0 for every parameter and there is no jump to make.

    False for a clock threshold whose threshold does not reduce to a constant
    over the primaries — the detector would silently skip that crossing — and
    for anything that is not a clock threshold at all.

    This is the predicate that keeps the clock path and the state path from
    fighting over the same crossing. A BNGL counter clock is a *species*, so
    ``t >= sigma`` reads live state and :func:`state_switch_residual` would
    happily claim it; letting both claim it would apply the jump twice. Asked
    first, in both the gate and the run-time detector, so the two cannot split
    the difference.
    """
    split = _clock_threshold_split(atom, scope.clock_symbols)
    if split is None:
        return False
    threshold_expr = split[1]
    if fixed_clock_threshold(atom, scope):
        return True  # a literal threshold: fixed, so nothing moves the crossing
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
    return bool(partials) and value is not None


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

    1. :func:`_clock_threshold_split` recognizes it *and* the threshold reduces
       to primaries and evaluates to a constant — the two conditions under which
       the detector actually emits the compensating record. A threshold it would
       silently skip is no better than an uncompensated state threshold here.
    2. :func:`state_switch_residual` splits it into a residual over live state,
       which is what :func:`state_switch_conditions` hands the solver to root on
       and jump at.
    3. it names no symbol at all (``0>0``), so it is a constant and never
       crosses.

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
            # Ground 2 — issue #150 roots on this crossing and jumps it.
            if state_switch_residual(scope.core, atom):
                continue
            atom_flat = _inline_derived_param_refs(atom, scope.derived_exprs) or atom
            # Ground 3 — a comparison between literals is a compile-time
            # constant with no crossing at all.
            if not _IDENTIFIER.search(atom_flat):
                continue
            split = _clock_threshold_split(atom, scope.clock_symbols)
            if split is None:
                return _not_a_clock_threshold(atom, atom_flat, scope)
            # A clock threshold that neither reduces to a constant over the
            # primaries (so the issue #48 detector would silently skip its
            # crossing) nor reads live state (so issue #150 cannot root on it).
            # Its ∂t*/∂p reaches nobody, and the Piecewise derivative's zero
            # would be the whole gradient.
            return UncompensatedCrossingReason(
                f"the clock threshold {split[1]!r} in the condition {atom!r} does not reduce "
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
    buried in a call argument, say), an equality, a comparison whose residual
    will not compile. A plain conjunction or negation is *not* one of these —
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


def compute_switch_time_sens(
    core,
    sens_param_names,
    t_start: float,
    t_end: float,
) -> tuple[list[tuple[float, int, float, list[float]]], list[int]]:
    """Switch-time crossings and their ``∂t*/∂p``, plus the parameters to pin.

    Parameters
    ----------
    core
        The C++ ``NetworkModel``.
    sens_param_names
        Requested sensitivity parameters, in the column order the run will use.
    t_start, t_end
        Reported time window; crossings outside it contribute nothing.

    Returns
    -------
    records
        ``(t_star, clock_species_idx0, threshold, [∂t*/∂p per column])`` sorted by
        ``t_star``, for ``SolverOptions.set_switch_time_sens``. Empty unless some
        ``if()`` threshold actually moves with one of ``sens_param_names`` — a
        model whose switch times are all fixed constants needs no jump and is
        left alone.
    pinned
        0-based indices of the switch-time parameters, for
        ``SolverOptions.set_switch_pinned_params``.

    Raises
    ------
    ValueError
        If a requested parameter sets a switch time *and* scales something inside
        a branch. The jump alone is then not the whole gradient, and pinning
        would discard the genuine in-branch ``∂f/∂p``; bngsim refuses rather than
        return a partially-correct derivative.
    """
    names = list(sens_param_names)
    if not names or core.n_functions == 0:
        return [], []

    ctx = core.functional_jacobian_context()
    function_bodies = list(ctx["function_map"].values())
    # Cheapest possible gate, checked before anything else: a model with no
    # conditional rate law has no switch to find. Keeps the RHS probes and the
    # sympy work off the path of every ordinary sensitivity run, including the
    # per-row batch path.
    if not any(_IF_CALL.search(body) for body in function_bodies):
        return [], []

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
    values = list(scope.values)
    primary_names = set(scope.primary_names)
    derived_exprs = scope.derived_exprs
    # A requested parameter that is itself derived has no independent axis: its
    # partials are attributed to the primaries it is built from, exactly as the
    # #41/#43 chain rules do. Columns for such a name stay 0.
    col_of = {name: c for c, name in enumerate(names)}

    # Keyed on (clock species, threshold value) so the same threshold appearing
    # in many rate laws — `t>=t0` gates six functions in Lin2021 — is one
    # crossing, stopped at and jumped across once.
    found: dict[tuple[int, float], tuple[float, list[float]]] = {}

    for body in function_bodies:
        for cond in _iter_if_conditions(body):
            for atom in _split_logical_atoms(cond):
                # The recognizer #68's codegen gate shares (see
                # _clock_threshold_split): the gate may only admit a condition
                # this loop turns into a compensating jump.
                #
                # :func:`clock_crossing_compensated` is this loop's acceptance
                # test stated as a predicate, for the gate and for the issue
                # #150 detector — which must skip exactly what this loop claims,
                # or a counter-clock threshold (a *species*, hence live state)
                # would be jumped twice. The two are held together
                # behaviourally, by TestTheGateAndTheDetectorsAgree, rather than
                # structurally; a change here needs a change there.
                split = _clock_threshold_split(atom, clock_symbols)
                if split is None:
                    continue
                clock_sym, threshold_expr = split

                # ``warn_on_failure=False``: this is a *scan* over candidate
                # thresholds, so an expression that does not reduce to primaries
                # is the supported "not a switch time" answer (see the comment
                # above), not a chain rule this feature lost. The warning that
                # issue #56 added is for callers where empty means a dropped
                # gradient; here it would fire on every state-dependent
                # condition in the model.
                partials = _derived_expr_partials_numeric(
                    threshold_expr,
                    primary_names,
                    param_idx,
                    values,
                    derived_exprs,
                    warn_on_failure=False,
                )
                dtstar = [0.0] * len(names)
                moved = False
                for prim_name, coeff in partials.items():
                    col = col_of.get(prim_name)
                    if col is not None and coeff != 0.0:
                        dtstar[col] += float(coeff)
                        moved = True
                if not moved:
                    # A crossing that no requested parameter moves contributes a
                    # zero jump. Skipping it keeps this feature's footprint to
                    # models that actually fit a switch time.
                    continue

                threshold_value = _evaluate_threshold(
                    threshold_expr, param_idx, values, derived_exprs
                )
                if threshold_value is None:
                    logger.debug(
                        "switch-time: threshold %r is not a constant expression; skipping",
                        threshold_expr,
                    )
                    continue

                if clock_sym in _TIME_SYMBOLS:
                    clock_idx0 = -1
                    t_star = threshold_value
                else:
                    clock_idx0 = clocks[clock_sym]
                    # dc/dt = 1, so c(t) = c(t_start) + (t − t_start) and the
                    # crossing is offset by the clock's current value. Lin2021
                    # seeds counter() at 1, which shifts every threshold by a day.
                    clock_now = core.get_concentration(core.species_names[clock_idx0])
                    t_star = t_start + (threshold_value - clock_now)

                # Half-open window, matching the core's own filter: a crossing on
                # t_end still jumps into the final recorded column; one on
                # t_start would precede the run's initial recording.
                if not (t_start < t_star <= t_end):
                    continue

                key = (clock_idx0, threshold_value)
                prev = found.get(key)
                if prev is None:
                    found[key] = (t_star, dtstar)
                else:
                    # Same crossing reached through a different (but equal)
                    # threshold spelling — keep the larger-magnitude partial
                    # rather than double-counting the jump.
                    for c in range(len(names)):
                        if abs(dtstar[c]) > abs(prev[1][c]):
                            prev[1][c] = dtstar[c]

    records = [
        (t_star, clock_idx0, threshold, dtstar)
        for (clock_idx0, threshold), (t_star, dtstar) in found.items()
    ]
    records.sort(key=lambda r: r[0])
    if not records:
        return [], []

    # Every parameter that moves at least one detected crossing is a switch-time
    # parameter, and must be pinned against CVODES' FD probe — pinning is both
    # what makes ∂f/∂p come out as the correct 0 and what keeps the probe from
    # displacing the switch into the approach (which stalls the integrator).
    switch_params = {
        names[c]
        for _t, _ci, _thr, dtstar in records
        for c in range(len(names))
        if dtstar[c] != 0.0
    }
    pure = _condition_only_params(core, switch_params, derived_exprs)
    impure = sorted(switch_params - pure)
    if impure:
        raise ValueError(
            "Forward sensitivity w.r.t. "
            + ", ".join(f"'{p}'" for p in impure)
            + " is not supported: each sets an if() switch time AND appears inside "
            "a branch (or as a rate constant), so its gradient is not the crossing "
            "jump alone — there is also a non-zero in-branch ∂f/∂p that bngsim "
            "cannot currently combine with the jump (issue #48). bngsim refuses "
            "rather than return a partially-correct derivative. Split the "
            "parameter into a separate switch time and rate constant, or drop it "
            "from sensitivity_params."
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
    same recognizer :func:`_clock_threshold_split` applies to ``if()``
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
    return _derived_expr_value_numeric(
        s,
        set(param_idx) - set(derived_exprs),
        derived_exprs.keys(),
        param_idx,
        values,
    )
