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
    paren depth 0, then re-descends into atoms that were parenthesised as a
    group. A leading ``!`` is dropped: negation flips which branch is taken but
    not *where* the crossing is, and the core reads f⁻/f⁺ by evaluating the real
    RHS on each side rather than by interpreting the condition.
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
        p = _strip_redundant_parens(part).lstrip("!").strip()
        if not p:
            continue
        if _LOGICAL.search(p):
            atoms.extend(_split_logical_atoms(p))
        else:
            atoms.append(p)
    return atoms


def _relational_split(atom: str) -> tuple[str, str] | None:
    """Split a relational atom into ``(lhs, rhs)`` at its depth-0 comparison.

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
                return atom[:i].strip(), atom[m.end() :].strip()
        i += 1
    return None


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
    clocks = _unit_rate_clock_species(core, ctx)
    clock_symbols = set(clocks) | set(_TIME_SYMBOLS)

    param_names = list(core.param_names)
    is_expr = list(core.param_is_expression)
    exprs = list(core.param_expressions)
    param_idx = {n: i for i, n in enumerate(param_names)}
    values = [core.get_param(n) for n in param_names]
    primary_names = {param_names[i] for i in range(len(param_names)) if not is_expr[i]}
    derived_exprs = {
        param_names[i]: exprs[i] for i in range(len(param_names)) if is_expr[i] and exprs[i]
    }
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
                split = _relational_split(atom)
                if split is None:
                    continue
                lhs, rhs = split
                lhs_bare = _strip_redundant_parens(lhs)
                rhs_bare = _strip_redundant_parens(rhs)
                # Exactly one side must be the clock; `t < 2*t` is not a fixed
                # crossing, and neither-side is not a time threshold at all.
                lhs_clock = lhs_bare in clock_symbols
                rhs_clock = rhs_bare in clock_symbols
                if lhs_clock == rhs_clock:
                    continue
                clock_sym, threshold_expr = (
                    (lhs_bare, rhs_bare) if lhs_clock else (rhs_bare, lhs_bare)
                )
                # A threshold that reads the clock back (`t < 2*t`) has no fixed
                # crossing time. A threshold over some *other* state is caught a
                # step later: the partials come back empty because the expression
                # has a free symbol that is not a primary parameter, so the
                # crossing is skipped rather than pinned to a time it does not
                # have.
                if any(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_])",
                        threshold_expr,
                    )
                    for sym in clock_symbols
                ):
                    continue

                partials = _derived_expr_partials_numeric(
                    threshold_expr, primary_names, param_idx, values, derived_exprs
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


def _evaluate_threshold(
    expr: str,
    param_idx: dict,
    values: list,
    derived_exprs: dict[str, str],
) -> float | None:
    """Numeric value of a threshold expression at the current parameter point.

    A bare parameter name (the common case: ``t>=sigma``) is a dict lookup. Any
    other arithmetic expression — including one over derived parameters — is
    evaluated by sympy after the same nested-derived inlining the partials use,
    so the value and its derivative always come from the same expression.
    """
    from bngsim._codegen import _inline_derived_param_refs

    s = expr.strip()
    if s in param_idx:
        return float(values[param_idx[s]])
    try:
        return float(s)
    except ValueError:
        pass
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:  # pragma: no cover - sympy is a hard dep of codegen
        return None
    flat = _inline_derived_param_refs(s, derived_exprs)
    referenced = sorted(p for p in param_idx if re.search(rf"\b{re.escape(p)}\b", flat))
    local = {p: sp.Symbol(p) for p in referenced}
    try:
        sym = parse_expr(flat, local_dict=local, evaluate=True)
        subs = {local[p]: values[param_idx[p]] for p in referenced}
        return float(sym.subs(subs).evalf())
    except Exception:
        return None
