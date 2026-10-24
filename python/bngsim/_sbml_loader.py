"""bngsim._sbml_loader — SBML model loader via libsbml.

Core loader that converts SBML (from file, string, or via Antimony)
into a BNGsim NetworkModel using ModelBuilder.  Handles all SBML
semantics correctly: initialAmount vs initialConcentration,
hasOnlySubstanceUnits, compartment volumes, assignment/rate rules,
initialAssignment evaluation, MathML→ExprTk expression translation.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import io
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO

import libsbml

from bngsim import _sbml_unsupported
from bngsim._switch_sensitivity import _ATOMIC_TOKEN, _condition_key, _fully_parenthesised

logger = logging.getLogger("bngsim")

# ExprTk reserved words — same set as _antimony_loader.py
EXPRTK_RESERVED_WORDS = frozenset(
    {
        "if",
        "else",
        "switch",
        "case",
        "default",
        "for",
        "while",
        "repeat",
        "until",
        "break",
        "continue",
        "return",
        "true",
        "false",
        "and",
        "or",
        "not",
        "xor",
        "nand",
        "nor",
        "xnor",
        "in",
        "like",
        "ilike",
        "null",
        "nan",
        "inf",
        "var",
        "avg",
        "sum",
        "mul",
        "min",
        "max",
        "mand",
        "mor",
    }
)

EXPRTK_TIME = {"t", "time"}


@functools.lru_cache(maxsize=1)
def _exprtk_builtin_function_names() -> frozenset[str]:
    """ExprTk + bngsim built-in function names (``log``, ``sin``, ``exp``, …).

    Sourced from the compiled core (``reserved_names()["functions"]``) so this
    set cannot drift from the symbol table that actually rejects the collision.
    A model symbol sharing a name with a built-in must be renamed: ExprTk
    reserves the built-in and a single flat namespace cannot hold both meanings
    (GH #90 — COPASI exports a parameter literally named ``log`` whose rule is
    ``log = ln(V)``; the ``<ln/>`` builtin renders to ExprTk ``log(...)`` and
    collides with the symbol ``log``). Comparison is case-sensitive because the
    core builds with ``exprtk_disable_caseinsensitivity``.
    """
    try:
        from bngsim._bngsim_core import reserved_names as _reserved_names

        return frozenset(_reserved_names()["functions"])
    except Exception:  # pragma: no cover — core is importable wherever SBML loads
        return frozenset()


def _safe_name(name: str) -> str:
    """Rename identifiers that clash with ExprTk reserved words or built-ins."""
    if (
        name.lower() in EXPRTK_RESERVED_WORDS
        or name in EXPRTK_TIME
        or name in _exprtk_builtin_function_names()
    ):
        return f"_ant_{name}"
    return name


def _register_ic_param_ref(
    builder,
    species_idx0: int,
    sym: str,
    ia_single_param_ref: dict[str, str],
    ia_expr_param: dict[str, str],
    declared_params: set[str],
) -> None:
    """Link a state's initial condition to the parameter(s) that define it.

    Called wherever this loader adds an integrator state whose IC came from an
    ``<initialAssignment>`` over parameters: §3 for an SBML ``<species>``, and
    §8/§10 for a parameter or compartment promoted to a species by a rate rule
    or an event assignment. One function for all three because they must not
    drift — a state that gets the IC seed at one site and not another is a
    silently-zero sensitivity column at the other (issue #146).

    A bare ``<ci>`` links straight to the named parameter and is seeded with
    coefficient 1 by the legacy identity path; a compound expression links to
    the synthetic derived parameter §3 lowered it to, which reaches issue #43's
    sympy chain rule to primaries. ``add_species_param_ref`` resolves by
    parameter *name* at build time and refuses a name that is not a builder
    parameter (issue #863), so a bare ``<ci>`` naming a symbol this loader
    turned into a species (not in ``declared_params``) registers nothing
    rather than mis-seeding.
    """
    if sym in ia_single_param_ref:
        if ia_single_param_ref[sym] in declared_params:
            builder.add_species_param_ref(species_idx0, _safe_name(ia_single_param_ref[sym]))
    elif sym in ia_expr_param:
        builder.add_species_param_ref(species_idx0, ia_expr_param[sym])


# ── MathML AST → ExprTk string ───────────────────────────────────────────


def _real_literal(v: float) -> str:
    """Render an ``AST_REAL`` value as an ExprTk-parseable numeric literal.

    A finite value round-trips through ``repr``. A non-finite value has no
    bare ExprTk token — ``init_builtins`` registers neither ``nan`` nor
    ``inf``, and ExprTk's ``<n>#nan`` lexer form does not survive being
    embedded inside a parenthesised subexpression (``(0#nan*c)`` fails to
    compile) — so emit pure arithmetic that constant-folds to the IEEE value:
    ``(0.0/0.0)`` for NaN, ``(1.0/0.0)`` / ``(-1.0/0.0)`` for ±inf. The folded
    NaN then propagates through any referencing expression; e.g. the
    MODEL1910030001 event trigger ``nan*c > 0`` becomes a permanently-false
    comparison (NaN compares false), so the event never fires — matching
    libRoadRunner (GH #92).
    """
    if _math.isfinite(v):
        # A negative literal is parenthesised, like the -inf form below: the
        # parent templates splice it in as-is, and bare `-3.0` as a power base
        # reads as -(3^2) and after a minus as `a--3.0` (GH #744).
        return f"({v!r})" if _math.copysign(1.0, v) < 0.0 else repr(v)
    if _math.isnan(v):
        return "(0.0/0.0)"
    return "(1.0/0.0)" if v > 0.0 else "(-1.0/0.0)"


def _atomic(expr: str) -> str:
    """Parenthesise substituted expression text unless it is one token.

    A local parameter, a function-definition argument, a species-IC lift or a
    constant assignment-rule lift is spliced into operator templates as text.
    Bare `-3` or `a+b` there changes meaning as a power base or after a minus
    (GH #744), so anything but a name, an unsigned number or an already fully
    parenthesised group gets parentheses.
    """
    if _ATOMIC_TOKEN.fullmatch(expr) or _fully_parenthesised(expr):
        return expr
    return f"({expr})"


def _ia_value_changed(new: float, old: float | None) -> bool:
    """Whether an initialAssignment / assignment-rule fold has moved a symbol's
    value — the section-0 fixpoint loop's continue-iterating test, made
    non-finite-safe.

    The naive ``abs(new - old) > 1e-30`` silently DROPS a NaN result: when a
    ``<notanumber/>`` initialAssignment folds to NaN, ``abs(NaN - old)`` is NaN
    and ``NaN > 1e-30`` is False, so the loop treats it as "no change" and the
    symbol keeps its stale raw value (GH #247, case 00950's ``R`` stayed 0.0
    instead of NaN). Compare with NaN-awareness: two NaNs are "no change" so the
    loop still converges, NaN-vs-number is a change, and ±inf needs no special
    case (``abs(inf - old)`` is ``inf > 1e-30``, already True).
    """
    if old is None:
        return True
    new_nan = _math.isnan(new)
    old_nan = _math.isnan(old)
    if new_nan or old_nan:
        return new_nan != old_nan
    return abs(new - old) > 1e-30


# ── SBML rateOf csymbol (GH #106) ─────────────────────────────────────────
#
# rateOf(x) is the instantaneous time-derivative dx/dt — a value only the
# integrator knows. Two SBML encodings reach the loader:
#   * the official csymbol — libsbml AST_FUNCTION_RATE_OF (type 323);
#   * the COPASI functionDefinition idiom — a funcDef named ``rateOf`` whose
#     body is ``<notanumber/>`` (sometimes ``NaN + 0*a``) carrying a
#     ``definition="…/Derivative"`` annotation.
# Both normalize to one emission: a per-species accessor token
# ``rate_of__<species>``. The C++ builder (enable_rateof) binds each token to
# the live dx/dt the simulator publishes before every RHS / trigger eval. See
# dev/notes/gh106_rateof_kickoff.md.

_RATEOF_PREFIX = "rate_of__"

# Stored as the body of a COPASI rateOf-idiom functionDefinition in
# ``func_defs`` so the inliner emits an accessor for the *call's* argument
# instead of inlining the funcDef's NaN body.
_RATEOF_FUNCDEF = object()

# The parameters, compartments and speciesReference ids of the model being
# built that no rule sets.
# Their rate of change is 0 at every time, and the builder binds a
# ``rate_of__`` accessor only for a state, so :func:`_rateof_call` emits the 0
# for them. Set by the build, which runs in a context of its own, so a load
# never sees another load's names.
_RATEOF_CONSTANT_SYMBOLS: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_RATEOF_CONSTANT_SYMBOLS", default=frozenset()
)


def _rateof_constant_symbols(sbml_model) -> frozenset[str]:
    """Parameters, compartments and speciesReference ids (stoichiometries)
    that no rate or assignment rule sets.

    One an event assigns is among them: an event's assignment is a jump, not a
    rate, so between events it holds and its rate is 0. (bngsim makes it a
    state, whose accessor reads that same 0.)
    """
    ruled = set()
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isRate() or rule.isAssignment():
            ruled.add(rule.getVariable())
    names = {sbml_model.getParameter(i).getId() for i in range(sbml_model.getNumParameters())}
    names |= {sbml_model.getCompartment(i).getId() for i in range(sbml_model.getNumCompartments())}
    names |= {
        sr.getId()
        for sr in _iter_species_references(sbml_model)
        if hasattr(sr, "getId") and sr.getId()
    }
    return frozenset(names - ruled)


def _rateof_token(species_name: str) -> str:
    """Accessor token for ``rateOf(<species_name>)``. Must match the C++
    accessor name ``register_rateof_accessors`` binds — the ``rate_of__``
    prefix plus the species' ``_safe_name`` (the name it is registered under)."""
    return _RATEOF_PREFIX + _safe_name(species_name)


def _rateof_call(node: libsbml.ASTNode, local_params: dict | None = None) -> str:
    """Emit the accessor for a ``rateOf(...)`` call (csymbol or COPASI idiom).

    Only rateOf of a bare symbol — a species, or a rate-rule-promoted
    parameter/compartment that becomes a species — is representable as an
    accessor. A non-name argument has no species index to bind to; the load
    refuses one first (``_check_rateof_targets``, rule 10223), so the warning
    and ``0`` below are only a backstop.

    A kinetic-law ``<localParameter>`` is constant by SBML definition, so the
    rate of change of a local-parameter argument is ``0`` (suite 01459/01460).
    It also has no species index — emitting the accessor would either fail to
    compile (no global of that name) or silently bind to a same-named global,
    so the local-scope check both fixes the value and avoids the mis-bind.

    So is a parameter, compartment or speciesReference id that no rule sets
    (``_RATEOF_CONSTANT_SYMBOLS``): its rate is ``0``, and for one no event
    assigns either, its accessor, bound to nothing, failed to compile (#1045's
    case, outside an initial value).
    """
    if node.getNumChildren() >= 1:
        arg = node.getChild(0)
        if arg.getType() == libsbml.AST_NAME and arg.getName():
            if local_params and arg.getName() in local_params:
                return "0"
            if arg.getName() in _RATEOF_CONSTANT_SYMBOLS.get():
                return "0"
            return _rateof_token(arg.getName())
    logger.warning("rateOf() with a non-symbol argument is unsupported; emitting 0")
    return "0"


def _ast_at_time_zero(node: libsbml.ASTNode) -> libsbml.ASTNode:
    """A copy of *node* with every ``time`` csymbol replaced by the number 0.

    An initialAssignment is evaluated at the initial time, so this is its exact
    meaning there. Used when the load-time fold could not evaluate one that
    reads ``time`` (issue #732), so the engine can evaluate it instead.
    """
    copy = node.deepCopy()
    for n in _iter_ast_subtree(copy):
        if n.getType() == libsbml.AST_NAME_TIME:
            n.setType(libsbml.AST_REAL)
            n.setValue(0.0)
    return copy


def _iter_ast_subtree(node: libsbml.ASTNode):
    """Yield every node in the AST subtree rooted at ``node``, pre-order,
    iteratively (O(1) Python stack depth regardless of subtree shape).

    libsbml's MathML reader binarizes an n-ary ``<plus/>``/``<times/>`` into a
    deep left-leaning chain ``(((a + b) + c) + d)``; a recursive scan of such a
    chain costs one Python frame per operand and trips
    ``sys.getrecursionlimit()`` on BNG-emitted observables that sum hundreds of
    species (GH #111). Every full-subtree scanner here walks via this generator
    for the same reason :func:`_flatten_assoc_chain` exists on the eval path.
    """
    if node is None:
        return
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for i in range(n.getNumChildren() - 1, -1, -1):
            stack.append(n.getChild(i))


def _ast_has_rateof(node: libsbml.ASTNode, rateof_funcdef_names: set[str]) -> bool:
    """True iff the AST subtree references rateOf — the official csymbol or a
    call to a COPASI rateOf-idiom funcDef (by name)."""
    for n in _iter_ast_subtree(node):
        t = n.getType()
        if t == libsbml.AST_FUNCTION_RATE_OF:
            return True
        if t == libsbml.AST_FUNCTION and n.getName() in rateof_funcdef_names:
            return True
    return False


def _model_uses_rateof(sbml_model, func_defs: dict, rateof_funcdef_names: set[str]) -> bool:
    """True iff any math container references rateOf (csymbol or a *called*
    COPASI idiom). Walks rules, reactions, every event math, and initial
    assignments — plus non-idiom funcDef bodies, which still count here,
    although the load writes a call to one that reads rateOf out first
    (``_expand_rated_function_calls``)."""

    def _scan(n):
        return _ast_has_rateof(n, rateof_funcdef_names)

    for i in range(sbml_model.getNumRules()):
        if _scan(sbml_model.getRule(i).getMath()):
            return True
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl and kl.isSetMath() and _scan(kl.getMath()):
            return True
    for i in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(i)
        if ev.isSetTrigger() and _scan(ev.getTrigger().getMath()):
            return True
        if ev.isSetDelay() and _scan(ev.getDelay().getMath()):
            return True
        if ev.isSetPriority() and _scan(ev.getPriority().getMath()):
            return True
        for j in range(ev.getNumEventAssignments()):
            if _scan(ev.getEventAssignment(j).getMath()):
                return True
    for i in range(sbml_model.getNumInitialAssignments()):
        if _scan(sbml_model.getInitialAssignment(i).getMath()):
            return True
    # A rateOf csymbol buried in a (non-idiom) funcDef body that is later
    # inlined also enables the feature. The idiom funcDefs hold the sentinel
    # (not an AST), so skip them here.
    for _name, (_params, body) in func_defs.items():
        if body is not _RATEOF_FUNCDEF and _scan(body):
            return True
    return False


def _rated_functions(func_defs: dict) -> dict[str, set[str]]:
    """The functionDefinitions whose body reads ``rateOf``, through the COPASI
    idiom or the csymbol, directly or through another they call, each with the
    ones of them it calls."""
    idiom_names = {name for name, (_p, body) in func_defs.items() if body is _RATEOF_FUNCDEF}
    rated: set[str] = set()
    calls: dict[str, set[str]] = {}
    for fid, (_p, body) in func_defs.items():
        if body is _RATEOF_FUNCDEF:
            continue
        calls[fid] = set()
        for n in _iter_ast_subtree(body):
            t = n.getType()
            if t == libsbml.AST_FUNCTION_RATE_OF or (
                t == libsbml.AST_FUNCTION and n.getName() in idiom_names
            ):
                rated.add(fid)
            elif t == libsbml.AST_FUNCTION:
                calls[fid].add(n.getName())
    grew = True
    while grew:
        grew = False
        for fid, called in calls.items():
            if fid not in rated and not called.isdisjoint(rated):
                rated.add(fid)
                grew = True
    return {fid: calls[fid] & rated for fid in rated}


def _expand_rated_function_calls(sbml_model, func_defs: dict, rated: dict[str, set[str]]) -> None:
    """Write each call to a functionDefinition whose body reads ``rateOf`` out
    in place, with the caller's arguments put in for the parameters (issue
    #1124, item 8). ``rated`` is :func:`_rated_functions` of ``func_defs``.

    ``f(y) = 2*rateOf(y)`` called as ``f(S)`` then reads ``2*rateOf(S)``, the
    rate of what was passed, and the checks, the initial-value fold and the
    emitter take it as they take a call written there. Inside the body the
    argument is the function's own parameter, which none of them follows into
    the caller: the load refused every such call. A rate of an argument that
    is not a single symbol is refused by the check after this (rule 10223).

    Two calls are left as they are, for :func:`_check_rateof_targets` to refuse
    by name: one whose arguments do not match the parameters one for one, where
    a parameter would be left unbound or an argument dropped, and one to a
    function that calls itself, directly or through another (invalid SBML, rule
    20512), which has no end to write out.
    """
    if not rated:
        return
    recursive = set()
    for fid in rated:
        todo, seen = list(rated[fid]), set()
        while todo:
            callee = todo.pop()
            if callee == fid:
                recursive.add(fid)
                break
            if callee not in seen:
                seen.add(callee)
                todo.extend(rated[callee])

    def is_rated_call(n) -> bool:
        return n.getType() == libsbml.AST_FUNCTION and n.getName() in rated

    def can_write_out(n) -> bool:
        return (
            is_rated_call(n)
            and n.getName() not in recursive
            and n.getNumChildren() == len(func_defs[n.getName()][0])
        )

    def written_out(call):
        # Every argument is put in at once, so one that names another parameter
        # is not replaced again: sw(x, S) = rateOf(x)*S called as sw(S, k) is
        # rateOf(S)*k, where one parameter at a time gave rateOf(k)*k. Only a
        # <ci> is a parameter, as for the inliner (_substitute_ast); the time
        # csymbol is not, whatever its text. The body reads rateOf, so it is not
        # a bare parameter, and the root is never replaced.
        params, body = func_defs[call.getName()]
        args = {p: call.getChild(j) for j, p in enumerate(params)}
        out = body.deepCopy()
        reads = [
            (n, j)
            for n in _iter_ast_subtree(out)
            for j in range(n.getNumChildren())
            if n.getChild(j).getType() == libsbml.AST_NAME and n.getChild(j).getName() in args
        ]
        for n, j in reads:
            n.replaceChild(j, args[n.getChild(j).getName()].deepCopy(), True)
        return out

    def expanded(root):
        # One pass, top down: a call is written out where it stands, then the
        # pass goes on into what was put there, the arguments with it. A
        # function left out of `recursive` cannot reach itself, so a run of
        # write-outs at one place ends.
        while can_write_out(root):
            root = written_out(root)
        stack = [root]
        while stack:
            parent = stack.pop()
            for j in range(parent.getNumChildren()):
                while can_write_out(parent.getChild(j)):
                    parent.replaceChild(j, written_out(parent.getChild(j)), True)
                stack.append(parent.getChild(j))
        return root

    for holder, where in _iter_math_holders(sbml_model):
        math = holder.getMath()
        if not any(is_rated_call(n) for n in _iter_ast_subtree(math)):
            continue
        out = expanded(math.deepCopy())
        if holder.setMath(out) == libsbml.LIBSBML_OPERATION_SUCCESS:
            continue
        # libsbml holds no math it finds ill-formed, and the csymbol rateOf of
        # anything but a single symbol is: f(2*S), for f(x) = rateOf(x), reads
        # rateOf(2*S). The element would keep the call, so it is refused here,
        # as the check refuses one written there (rule 10223).
        offenders = {
            (libsbml.formulaToL3String(n), where): (
                "its argument is not a single symbol (validation rule 10223)"
            )
            for n in _iter_ast_subtree(out)
            if n.getType() == libsbml.AST_FUNCTION_RATE_OF
            and (n.getNumChildren() != 1 or n.getChild(0).getType() != libsbml.AST_NAME)
        }
        _refuse_rateof_reads(
            offenders
            or {(libsbml.formulaToL3String(out), where): "libsbml refuses it written out"}
        )


def _check_rateof_targets(sbml_model, func_defs: dict, rated_funcs: dict[str, set[str]]) -> None:
    """Refuse, by name, a ``rateOf`` that the model has no value for.

    One check before either the initial-value fold or the emitter runs, for the
    csymbol and for a call to COPASI's rateOf functionDefinition idiom alike:

    * ``rateOf(x)`` of a variable an assignmentRule sets (issue #1046). SBML
      defines no rate of change for one (L3V2 §3.4.6, validation rule 10224).
      Unchecked, the initial-value fold took it for a constant and gave 0, a
      value the model does not have. A rule, kinetic law or event emitted a
      ``rate_of__x`` accessor: for a parameter nothing binds it, and ExprTk
      refused it without naming anything in the model; for a species the
      builder binds it, and the model ran with the rate as 0.
    * ``rateOf(S)`` of a species held as a concentration in a compartment an
      assignmentRule sizes (rule 10225): its rate would need the rule's own
      rate, which SBML does not define either. Unchecked, the initial-value
      fold took the amount's rate over the size, with no term for the size's
      change; a rule had the term where the size reads only the time, and left
      it out (the amount's rate alone) where the size reads a species. A
      species with a rate rule of its own has its rate stated, whatever its
      compartment does, and is left alone (libsbml reports 10225 for it all
      the same). A constant or boundary one is not: SBML holds its amount, so
      its concentration moves with the size (suite 01117/01118).
    * ``rateOf`` of anything but a single symbol (rule 10223 for the
      csymbol). Unchecked, the fold refused the csymbol's but took the idiom's
      for NaN, and a rule read either as 0 with a warning.
    * ``rateOf(q)`` of a ``q`` the model declares nowhere. An initialAssignment
      or rule is refused before this (``_check_undefined_symbols``); in a
      kinetic law or an event, ExprTk refused the ``rate_of__q`` accessor,
      which names nothing in the model.
    * a call to a functionDefinition whose body reads ``rateOf``, directly or
      through another it calls (``rated_funcs``, :func:`_rated_functions`),
      that is still there. :func:`_expand_rated_function_calls` writes every
      such call out first (issue #1124, item 8), and leaves one only where its
      arguments do not match the parameters one for one, or where the function
      calls itself (rule 20512). Inside the body the argument is the function's
      own parameter: both paths took the rate of that name, 0 unless a model
      symbol shares it, rather than of what the caller passed. A function
      nothing calls is left alone.

    A kinetic law's own local parameter shadows a global of its id, so
    ``rateOf(q)`` in that law reads the constant local (0, suites 01459/01460)
    and is allowed; libsbml reports 10224 for it regardless of scope.
    """
    ar_rules = {}
    rate_ruled = set()
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isAssignment():
            ar_rules[rule.getVariable()] = rule
        elif rule.isRate():
            rate_ruled.add(rule.getVariable())
    idiom_names = {name for name, (_p, body) in func_defs.items() if body is _RATEOF_FUNCDEF}

    def _reads_rateof(n) -> bool:
        t = n.getType()
        return t == libsbml.AST_FUNCTION_RATE_OF or (
            t == libsbml.AST_FUNCTION and n.getName() in idiom_names
        )

    # A species held as a concentration, by its compartment (rule 10225), but
    # not one whose rate a rate rule of its own states.
    conc_compartment = {
        sp.getId(): sp.getCompartment()
        for sp in (sbml_model.getSpecies(i) for i in range(sbml_model.getNumSpecies()))
        if not sp.getHasOnlySubstanceUnits() and sp.getId() not in rate_ruled
    }
    law_locals = {}
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl is not None:
            law_locals[f"reaction:{rxn.getId()}"] = {
                lp.getId() for lp in _kinetic_law_local_params(kl)
            }

    def _rule_text(name: str) -> str:
        m = ar_rules[name].getMath()
        return f"{name} := {libsbml.formulaToL3String(m)}" if m is not None else name

    # (what is read, location) → why the model has no value for it, in model order
    offenders: dict[tuple[str, str], str] = {}
    declared: dict[str, bool] = {}  # an argument's name → whether the model has it
    for math, where in _iter_model_math(sbml_model):
        shadowed = law_locals.get(where, ())
        for n in _iter_ast_subtree(math):
            t = n.getType()
            if t == libsbml.AST_FUNCTION and n.getName() in rated_funcs:
                fid = n.getName()
                nparams = len(func_defs[fid][0])
                if n.getNumChildren() != nparams:
                    why = f"it takes {nparams} argument(s), not {n.getNumChildren()}"
                else:
                    why = "it calls itself, directly or through another (validation rule 20512)"
                offenders[(libsbml.formulaToL3String(n), where)] = (
                    f"the functionDefinition {fid} reads rateOf, and bngsim cannot "
                    f"write this call to it out with what is passed: {why}"
                )
                continue
            if not _reads_rateof(n):
                continue
            arg = n.getChild(0) if n.getNumChildren() >= 1 else None
            if arg is None or arg.getType() != libsbml.AST_NAME:
                offenders[(libsbml.formulaToL3String(n), where)] = (
                    "its argument is not a single symbol (validation rule 10223)"
                )
                continue
            name = arg.getName()
            if name in shadowed:
                continue
            if name not in declared:
                declared[name] = _model_declares_symbol(sbml_model, name)
            if not declared[name]:
                offenders[(f"rateOf({name})", where)] = f"{name} is declared nowhere in the model"
            elif name in ar_rules:
                offenders[(f"rateOf({name})", where)] = (
                    f"{name} is set by the assignmentRule {_rule_text(name)} "
                    "(validation rule 10224)"
                )
            elif conc_compartment.get(name) in ar_rules:
                cid = conc_compartment[name]
                offenders[(f"rateOf({name})", where)] = (
                    f"{name} is a concentration in compartment {cid}, which the "
                    f"assignmentRule {_rule_text(cid)} sizes (validation rule 10225)"
                )
    if offenders:
        _refuse_rateof_reads(offenders)


def _refuse_rateof_reads(offenders: dict[tuple[str, str], str]) -> None:
    """Raise the load's refusal of each ``(what is read, location) → why``."""
    from bngsim._exceptions import ModelError

    bullets = "\n".join(
        f"  - {shown} [{where}]: {why}" for (shown, where), why in offenders.items()
    )
    raise ModelError(
        "Model reads a rateOf that it has no value for:\n"
        f"{bullets}\n"
        "SBML (L3V2 §3.4.6) defines no rate of change for an assignmentRule "
        "variable, nor for a concentration in a compartment an assignmentRule "
        "sizes, and rateOf takes a single symbol; libsbml's consistency check "
        "reports each, and RoadRunner refuses them. Earlier bngsim releases loaded "
        "most of these with a wrong rate, or failed with an ExprTk error that "
        "named none of it. Make such a variable a rateRule target, or write its "
        "rate from the rule's own inputs; pass rateOf a single symbol, and a "
        "functionDefinition that reads it one argument per parameter."
    )


# ── Unsupported-construct refusal (GH #113) ──────────────────────────────
#
# bngsim integrates ODEs; it has no DDE or DAE solver and no fast-equilibrium
# constraint solver. Three SBML constructs were previously *dropped silently*
# under ODE, producing a confident finite trajectory for a different
# mathematical system with no warning:
#
#   * ``delay(x, τ)``   — turns the system into a delay-differential equation;
#                         the AST handler returned ``x`` (the zero-delay ODE).
#   * ``AlgebraicRule`` — a DAE constraint defining a variable implicitly;
#                         no rule loop dispatched it, so it was ignored.
#   * ``fast="true"``   — a fast-equilibrium constraint; under ODE the kinetic
#                         law integrated as an ordinary reaction.
#
# We refuse all three loud-by-default, mirroring the internal#94 unset-parameter
# gate (hard ``ModelError`` + ``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`` opt-out
# that restores the legacy silent-approximation behavior for deliberate triage,
# e.g. bngsim↔RoadRunner comparison). ``delay`` and ``AlgebraicRule`` are
# refused here at load (unsupported under every method); ``fast`` stays a
# loadable SSA issue and is refused at Simulator construction under ODE
# (_simulator.py) so the existing SSA validate/override contract is preserved.

_ALLOW_UNSUPPORTED_ENV = "BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS"


def _ast_is_literal_zero(node) -> bool:
    """True iff *node* is a numeric literal equal to 0."""
    if node is None:
        return False
    t = node.getType()
    if t == libsbml.AST_INTEGER:
        return node.getInteger() == 0
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
        return node.getReal() == 0.0
    return False


def _math_has_unsupported_delay(node, func_defs: dict, _seen: set | None = None) -> bool:
    """True iff *node* contains a ``delay()`` operator that is not the trivial
    ``delay(x, 0)``.

    ``delay(x, 0)`` equals ``x`` exactly, so dropping the delay is sound; any
    non-zero (or non-literal) delay makes the model a DDE bngsim cannot
    integrate. *Called* user-defined functions are expanded (so a delay buried
    in a helper actually used by the RHS is caught), while a delay in an
    uncalled funcDef body does not trip the gate. The ``_seen`` set guards
    against pathological cycles (SBML forbids recursive funcDefs).
    """
    if node is None:
        return False
    if _seen is None:
        _seen = set()
    for n in _iter_ast_subtree(node):
        t = n.getType()
        if t == libsbml.AST_FUNCTION_DELAY:
            if not (n.getNumChildren() >= 2 and _ast_is_literal_zero(n.getChild(1))):
                return True
        elif t == libsbml.AST_FUNCTION:
            fname = n.getName()
            if fname in func_defs and fname not in _seen:
                _seen.add(fname)
                _params, body = func_defs[fname]
                if body is not _RATEOF_FUNCDEF and _math_has_unsupported_delay(
                    body, func_defs, _seen
                ):
                    return True
    return False


def _law_is_nonnegative(node, lookup, env: dict | None = None, _seen: set | None = None) -> bool:
    """True iff *node* is non-negative by its form: a sum, product, quotient,
    power of a non-negative base, ``exp``, ``abs``, a ``piecewise`` of such
    pieces, over species, compartments, the time and non-negative constants.

    A reversible law that is not a difference: ``Vm*A/(Km + A)*c`` has no
    reverse flux to lose, the reversible flag on it is a label (COPASI sets it
    by default), and SSA runs it as the one channel it would run the same law as
    when flagged irreversible. Anything else is taken for a difference, however
    it is written: a minus, a negative constant, a parameter a rule or an event
    can make negative, ``ln``, ``sin``, an assignment rule ``v := kf*A - kr*B``
    the law reads, a reaction's rate. *lookup* names the model: see its use in
    the reaction loop. Called function definitions and assignment rules are
    expanded.
    """
    if node is None:
        return False
    env = env or {}
    _seen = set() if _seen is None else _seen
    t = node.getType()
    kids = [node.getChild(i) for i in range(node.getNumChildren())]

    def each(nodes) -> bool:
        return all(_law_is_nonnegative(k, lookup, env, _seen) for k in nodes)

    if t == libsbml.AST_INTEGER:
        return node.getInteger() >= 0
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E):
        v = node.getReal()
        return v == v and v >= 0
    if t == libsbml.AST_RATIONAL:
        return node.getDenominator() != 0 and node.getNumerator() * node.getDenominator() >= 0
    if t in (
        libsbml.AST_CONSTANT_E,
        libsbml.AST_CONSTANT_PI,
        libsbml.AST_CONSTANT_TRUE,
        libsbml.AST_CONSTANT_FALSE,
        libsbml.AST_NAME_TIME,
        libsbml.AST_NAME_AVOGADRO,
    ):
        return True
    if t == libsbml.AST_NAME:
        name = node.getName()
        if name in env:
            return env[name]
        return lookup(name, _seen)
    if t in (libsbml.AST_PLUS, libsbml.AST_TIMES, libsbml.AST_DIVIDE):
        return each(kids)
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        return len(kids) == 2 and _law_is_nonnegative(kids[0], lookup, env, _seen)
    if t in (libsbml.AST_FUNCTION_EXP, libsbml.AST_FUNCTION_ABS):
        return True
    if t in (libsbml.AST_FUNCTION_FLOOR, libsbml.AST_FUNCTION_CEILING, libsbml.AST_FUNCTION_MIN):
        return each(kids)
    if t == libsbml.AST_FUNCTION_ROOT:
        return bool(kids) and _law_is_nonnegative(kids[-1], lookup, env, _seen)
    if t == libsbml.AST_FUNCTION_DELAY:
        return bool(kids) and _law_is_nonnegative(kids[0], lookup, env, _seen)
    if t == libsbml.AST_FUNCTION_PIECEWISE:
        # piece, condition, piece, condition, ..., [otherwise]
        pieces = kids[0::2] if len(kids) % 2 == 0 else kids[0:-1:2] + [kids[-1]]
        return each(pieces)
    if t == libsbml.AST_FUNCTION:
        called = lookup.func_defs.get(node.getName())
        if called is None or called[1] is _RATEOF_FUNCDEF or len(called[0]) != len(kids):
            return False
        formals, body = called
        bound = {
            f: _law_is_nonnegative(k, lookup, env, _seen)
            for f, k in zip(formals, kids, strict=False)
        }
        return _law_is_nonnegative(body, lookup, bound, _seen)
    return False


def _check_unsupported_constructs(sbml_model, func_defs: dict) -> None:
    """Refuse ``delay()`` and ``AlgebraicRule`` models loud-by-default (GH #113).

    Scans every math container that feeds the integrated system — rate /
    assignment rules, reaction kinetic laws, initial assignments, and event
    trigger / delay / priority / assignment math — for a non-trivial
    ``delay()``, and flags every ``AlgebraicRule``. Raises a hard
    :class:`ModelError` naming each construct and element unless
    ``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`` is set, in which case it logs a
    warning per offender and restores the legacy silent approximation.

    ``fast="true"`` is intentionally *not* handled here — it stays a loadable
    SSA issue and is refused at Simulator construction under ODE.
    """
    import os

    offenders: list[tuple[str, str]] = []  # (construct, location)

    def _scan_delay(math, where: str) -> None:
        if _math_has_unsupported_delay(math, func_defs):
            offenders.append((_sbml_unsupported.DELAY, where))

    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isAlgebraic():
            m = rule.getMath()
            if m is None:
                # An AlgebraicRule with no MathML (``<algebraicRule/>``) states
                # ``0 = ∅`` — no constraint at all, so it is a no-op, not a DAE.
                # Refusing it declines a model bngsim can trivially simulate (SBML
                # suite 01244: a lone variable parameter that just holds its initial
                # value). Skip, mirroring the no-math event-assignment / rate-rule
                # handling. A non-empty algebraic rule is still a real DAE constraint
                # and is refused below.
                continue
            formula = libsbml.formulaToL3String(m)
            offenders.append((_sbml_unsupported.ALGEBRAIC_RULE, f"0 = {formula}"))
            continue
        if rule.isRate():
            _scan_delay(rule.getMath(), f"rateRule:{rule.getVariable()}")
        elif rule.isAssignment():
            _scan_delay(rule.getMath(), f"assignmentRule:{rule.getVariable()}")

    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl is not None and kl.isSetMath():
            _scan_delay(kl.getMath(), f"reaction:{rxn.getId()}")
        # L2 <stoichiometryMath> feeds the integrated system's stoichiometry just
        # as a kinetic law does; a delay() there (suite 01481) is as much a DDE as
        # one in the rate law. libSBML exposes it only via the SpeciesReference, so
        # scan every reactant/product (modifiers carry no stoichiometry).
        for getter, n in (
            (rxn.getReactant, rxn.getNumReactants()),
            (rxn.getProduct, rxn.getNumProducts()),
        ):
            for k in range(n):
                sr = getter(k)
                if sr.isSetStoichiometryMath() and sr.getStoichiometryMath() is not None:
                    sm = sr.getStoichiometryMath().getMath()
                    if sm is not None:
                        _scan_delay(
                            sm, f"reaction:{rxn.getId()}:stoichiometryMath:{sr.getSpecies()}"
                        )

    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        _scan_delay(ia.getMath(), f"initialAssignment:{ia.getSymbol()}")

    for i in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(i)
        eid = ev.getId() or f"event{i}"
        trig = ev.getTrigger()
        if trig is not None and trig.isSetMath():
            _scan_delay(trig.getMath(), f"event:{eid}:trigger")
        delay = ev.getDelay()
        if delay is not None and delay.isSetMath():
            _scan_delay(delay.getMath(), f"event:{eid}:delay")
        prio = ev.getPriority()
        if prio is not None and prio.isSetMath():
            _scan_delay(prio.getMath(), f"event:{eid}:priority")
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            if ea.isSetMath():
                _scan_delay(ea.getMath(), f"event:{eid}:assignment:{ea.getVariable()}")

    if not offenders:
        return

    from bngsim._exceptions import ModelError

    if os.environ.get(_ALLOW_UNSUPPORTED_ENV) == "1":
        for construct, where in offenders:
            logger.warning(
                "Unsupported construct %s [%s] is present; bngsim has no DDE/DAE "
                "solver and silently approximates it (%s=1). The integrated "
                "trajectory does not honor the model's stated semantics.",
                construct,
                where,
                _ALLOW_UNSUPPORTED_ENV,
            )
        return

    bullets = "\n".join(f"  - {construct} [{where}]" for construct, where in offenders)
    raise ModelError(
        "Model contains constructs bngsim cannot faithfully simulate under ODE "
        "(no DDE/DAE solver):\n"
        f"{bullets}\n"
        "Previously these were dropped silently, integrating a different "
        "mathematical system with no warning (RoadRunner refuses the same "
        "models). bngsim now refuses rather than return a wrong-but-plausible "
        f"trajectory. To restore the legacy silent-approximation behavior, set "
        f"{_ALLOW_UNSUPPORTED_ENV}=1."
    )


# ── Bare <ci>time</ci> diagnosis (GH #96) ────────────────────────────────
#
# libsbml parses the proper simulation-time csymbol
# (``<csymbol definitionURL=".../symbols/time">``) as AST_NAME_TIME. A bare
# ``<ci>time</ci>`` — the malformed idiom some COPASI / CellDesigner exports
# emit — parses instead as a plain AST_NAME whose name is "time". When the
# model declares no symbol named "time", that reference is undefined: the
# loader renames it to ``_ant_time`` (``time`` is an ExprTk reserved word; see
# _safe_name) and the core later fails to compile with an opaque
# ``ERR239 - Undefined symbol: u_ant_time``. RoadRunner refuses the same models.
#
# bngsim deliberately does NOT infer the csymbol from a bare ``<ci>time</ci>``
# (the lenient-resolution feature was declined in GH #96: it cuts against the
# loader's fail-closed/RR-matching posture and has no parity oracle). Instead
# we replace the opaque compile failure with a targeted, actionable error that
# names each offending location and tells the modeler to use the csymbol.


def _iter_species_references(sbml_model):
    """Yield every reactant/product ``SpeciesReference`` in model order."""
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        for getter, n_refs in (
            (rxn.getReactant, rxn.getNumReactants()),
            (rxn.getProduct, rxn.getNumProducts()),
        ):
            for j in range(n_refs):
                yield getter(j)


def _model_declares_symbol(sbml_model, name: str) -> bool:
    """True iff *name* binds to a declared model symbol.

    Checks the global SId namespace (species, parameters, compartments,
    reactions, events, function definitions, …) via ``getElementBySId`` plus
    every reaction's local (kinetic-law-scoped) parameters. Counting a local
    parameter as "declared" can only ever *suppress* the bare-``time``
    diagnostic, so a model that would otherwise load is never broken by a
    false positive — the worst case falls back to the legacy compile error.
    """
    if sbml_model.getElementBySId(name) is not None:
        return True
    for sr in _iter_species_references(sbml_model):
        if hasattr(sr, "getId") and sr.getId() == name:
            return True
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl is None:
            continue
        for j in range(kl.getNumLocalParameters()):
            if kl.getLocalParameter(j).getId() == name:
                return True
        for j in range(kl.getNumParameters()):
            if kl.getParameter(j).getId() == name:
                return True
    return False


def _iter_model_math(sbml_model):
    """Yield ``(math_node, location)`` for every MathML container that feeds the
    integrated system: rules, kinetic laws, Level 2 stoichiometryMath, initial
    assignments, and every event trigger / delay / priority / assignment.
    ``location`` is a short human-readable tag used in diagnostics."""
    for holder, where in _iter_math_holders(sbml_model):
        yield holder.getMath(), where


def _iter_math_holders(sbml_model):
    """Yield ``(element, location)`` for each element whose math
    :func:`_iter_model_math` yields, for a caller that sets it."""
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.getMath() is None:
            continue
        if rule.isRate():
            yield rule, f"rateRule:{rule.getVariable()}"
        elif rule.isAssignment():
            yield rule, f"assignmentRule:{rule.getVariable()}"
        elif rule.isAlgebraic():
            yield rule, "algebraicRule"
        else:
            yield rule, "rule"
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl is not None and kl.isSetMath():
            yield kl, f"reaction:{rxn.getId()}"
        # Level 2's <stoichiometryMath>, which libSBML keeps on the
        # speciesReference (modifiers carry no stoichiometry).
        for getter, n_refs in (
            (rxn.getReactant, rxn.getNumReactants()),
            (rxn.getProduct, rxn.getNumProducts()),
        ):
            for j in range(n_refs):
                sr = getter(j)
                sm = sr.getStoichiometryMath() if sr.isSetStoichiometryMath() else None
                if sm is not None and sm.isSetMath():
                    yield sm, f"reaction:{rxn.getId()}:stoichiometryMath:{sr.getSpecies()}"
    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        if ia.isSetMath():
            yield ia, f"initialAssignment:{ia.getSymbol()}"
    for i in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(i)
        eid = ev.getId() or f"event{i}"
        trig = ev.getTrigger()
        if trig is not None and trig.isSetMath():
            yield trig, f"event:{eid}:trigger"
        delay = ev.getDelay()
        if delay is not None and delay.isSetMath():
            yield delay, f"event:{eid}:delay"
        prio = ev.getPriority()
        if prio is not None and prio.isSetMath():
            yield prio, f"event:{eid}:priority"
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            if ea.isSetMath():
                yield ea, f"event:{eid}:assignment:{ea.getVariable()}"


def _check_bare_time_symbol(sbml_model) -> None:
    """Diagnose a bare ``<ci>time</ci>`` that references no declared symbol
    (GH #96) — fail closed with an actionable error instead of the opaque
    ``ERR239 - Undefined symbol: u_ant_time`` the core would later raise.

    Gated: fires ONLY when the model declares no symbol named ``time``. If a
    ``time`` symbol exists, the bare ``<ci>time</ci>`` legitimately binds to it
    and this is a no-op (the proper time csymbol parses as AST_NAME_TIME, never
    AST_NAME, so it is never mistaken for the bare idiom).
    """
    if _model_declares_symbol(sbml_model, "time"):
        return

    locations: list[str] = []
    for math, where in _iter_model_math(sbml_model):
        for n in _iter_ast_subtree(math):
            if n.getType() == libsbml.AST_NAME and n.getName() == "time":
                locations.append(where)
                break

    if not locations:
        return

    from bngsim._exceptions import ModelError

    bullets = "\n".join(f"  - {where}" for where in locations)
    raise ModelError(
        "Model references a bare <ci>time</ci> but declares no symbol named "
        "'time':\n"
        f"{bullets}\n"
        "In SBML a bare <ci> is a reference to a declared component; simulation "
        "time must be written as the time csymbol "
        '(<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">). '
        "Some tools (COPASI, CellDesigner) emit time as a bare <ci>time</ci> "
        "instead — rewrite it as the csymbol. RoadRunner refuses these models "
        "for the same reason; bngsim does not guess the intended meaning."
    )


# ── Undefined-symbol diagnosis in IAs and rules (GH #119) ────────────────
#
# bngsim's initialAssignment / assignmentRule evaluator (_eval_ast_numeric)
# returns ``None`` for an ``AST_NAME`` that binds to no declared symbol; the
# caller then leaves the target at its declared value and integrates — silently
# dropping the offending IA/rule with zero warnings and zero log lines.
# RoadRunner and COPASI both REFUSE to load such a model ("Could not find
# requested symbol 'X'" / ``CCopasiException``). This fail-open can turn a
# malformed model into a plausible-but-meaningless trajectory: MODEL2205030001
# (the GH #119 repro) had its entire class of ``Kd_*`` dissociation constants
# dropped on export, leaving all 23 initial assignments referencing undefined
# symbols — bngsim kept every declared initial condition and integrated anyway.
#
# Consistent with the loader's existing fail-closed posture — the bare
# ``<ci>time</ci>`` refusal (GH #96) and the fail-closed AST translator
# (GH #97) — we refuse, naming each undefined symbol and its location.
# ``BNGSIM_ALLOW_UNDEFINED_SYMBOLS=1`` restores the legacy silent-drop behavior
# with a loud per-offender warning (for deliberate bngsim↔RoadRunner triage).

_ALLOW_UNDEFINED_SYMBOLS_ENV = "BNGSIM_ALLOW_UNDEFINED_SYMBOLS"


def _iter_ia_and_rule_math(sbml_model):
    """Yield ``(math_node, location)`` for every initialAssignment and rule.

    Scoped deliberately narrower than :func:`_iter_model_math`: reaction
    kinetic laws are excluded (their local, kinetic-law-scoped parameters are
    not global symbols, so a global-namespace resolution check would mis-flag
    them) and so are events. The surfaced defect class (GH #119) is undefined
    symbols in the constructs that set up the integrated system's initial
    conditions and algebraic / rate structure — exactly IAs and rules.
    """
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        m = rule.getMath()
        if m is None:
            continue
        if rule.isRate():
            yield m, f"rateRule:{rule.getVariable()}"
        elif rule.isAssignment():
            yield m, f"assignmentRule:{rule.getVariable()}"
        elif rule.isAlgebraic():
            yield m, "algebraicRule"
        else:
            yield m, "rule"
    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        if ia.isSetMath():
            yield ia.getMath(), f"initialAssignment:{ia.getSymbol()}"


def _check_undefined_symbols(sbml_model) -> None:
    """Refuse a model that references an undefined symbol in an
    initialAssignment or rule (GH #119) instead of silently dropping it.

    Walks every ``AST_NAME`` in IAs and rules and verifies it binds to a
    declared model symbol (global SId, kinetic-law-scoped parameter, or — for a
    declared ``time`` — that symbol) via :func:`_model_declares_symbol`. The
    proper ``time`` csymbol parses as ``AST_NAME_TIME`` (never ``AST_NAME``) and
    user-function calls parse as ``AST_FUNCTION`` (their *name* is not an
    ``AST_NAME`` node, only their arguments are), so neither is mis-flagged.

    A bare ``<ci>time</ci>`` with no declared ``time`` symbol is diagnosed
    earlier and more specifically by :func:`_check_bare_time_symbol`, which has
    already raised by the time this runs; if a ``time`` symbol *is* declared it
    resolves here like any other name.

    Default is a hard :class:`ModelError` naming each offender (matching
    RoadRunner / COPASI, which refuse the same models).
    ``BNGSIM_ALLOW_UNDEFINED_SYMBOLS=1`` restores the legacy silent-drop with a
    loud per-offender warning.
    """
    import os

    # Distinct referenced name → one representative location. Dedupe first so
    # _model_declares_symbol (an O(model) lookup) runs once per name, not once
    # per occurrence.
    referenced: dict[str, str] = {}
    for math, where in _iter_ia_and_rule_math(sbml_model):
        for n in _iter_ast_subtree(math):
            if n.getType() == libsbml.AST_NAME and n.getName():
                referenced.setdefault(n.getName(), where)

    offenders = {
        name: where
        for name, where in referenced.items()
        if not _model_declares_symbol(sbml_model, name)
    }
    if not offenders:
        return

    from bngsim._exceptions import ModelError

    if os.environ.get(_ALLOW_UNDEFINED_SYMBOLS_ENV) == "1":
        for name, where in offenders.items():
            logger.warning(
                "Symbol '%s' referenced by %s binds to no declared model symbol; "
                "bngsim drops that initialAssignment / rule and keeps the target's "
                "declared value (%s=1). RoadRunner and COPASI refuse the model. "
                "The integrated initial conditions / rule structure are silently "
                "incomplete.",
                name,
                where,
                _ALLOW_UNDEFINED_SYMBOLS_ENV,
            )
        return

    bullets = "\n".join(f"  - {name!r} [{where}]" for name, where in offenders.items())
    raise ModelError(
        "Model references symbols that bind to no declared component "
        "(no global SId, kinetic-law parameter, or time csymbol) in an "
        "initialAssignment or rule:\n"
        f"{bullets}\n"
        "bngsim's evaluator returns no value for an undefined symbol, so it "
        "would silently DROP each offending initialAssignment / rule and keep "
        "the target's declared value — turning a malformed model (e.g. one whose "
        "constants were lost on export) into a plausible-but-meaningless "
        "trajectory. RoadRunner and COPASI both refuse to load these models. "
        "bngsim refuses rather than fail open; define the missing symbol(s) or "
        "remove the references. To restore the legacy silent-drop behavior, set "
        f"{_ALLOW_UNDEFINED_SYMBOLS_ENV}=1."
    )


# ── Time-dependent discontinuity detection (GH #72) ──────────────────────
#
# Piecewise expressions that switch on the `time` csymbol (drug-dosing
# windows, scheduled stimuli) introduce discontinuities in the ODE RHS that an
# adaptive integrator has no way to anticipate: with the default interpolated
# output it takes large internal steps and can jump straight over a narrow
# "on" window, silently dropping the dose (the BIOMD0000000879 chemo-pulse bug
# — 6 of 7 infusions missed). We detect each `time` threshold and register it
# as a CVODE root so the integrator is forced to stop at the crossing.

_INEQ_RELATIONAL_OPS = None  # lazily built {AST type: op string} (needs libsbml)


def _ast_references_time(node: libsbml.ASTNode) -> bool:
    """True iff the AST subtree references the SBML ``time`` csymbol."""
    return any(n.getType() == libsbml.AST_NAME_TIME for n in _iter_ast_subtree(node))


def _ast_side_is_time(
    node: libsbml.ASTNode, time_names: frozenset[str] | set[str] = frozenset()
) -> bool:
    """True iff one side of a relational moves with time, in *its own* frame.

    The ``time`` csymbol is not always what appears. An SBML model routinely
    aliases it — ``<assignmentRule variable="model_time"> time`` — and inside a
    function definition's body the threshold is written against a *formal
    parameter* the call site bound to something time-dependent (``stepfunc(t,
    t_start, …)`` with ``t`` ← ``model_time``). ``time_names`` carries both:
    the assignment-rule targets that transitively read time, plus the formals
    of the enclosing call frames. Reading them as time-dependent is what makes
    this judge the relational the way the inlined expression the RHS actually
    evaluates would.
    """
    return _ast_references_time(node) or bool(_ast_name_set(node) & time_names)


def _make_time_arg_predicate(
    time_names: frozenset[str] | set[str],
) -> Callable[[libsbml.ASTNode, frozenset[str], frozenset[str]], bool]:
    """:func:`_iter_funcdef_scan_frames`' ``arg_is_dynamic`` for the time scan.

    ``shadowed`` is unused here, unlike the state twin: a formal parameter can
    shadow a model symbol, but the ``time`` csymbol is not a name, so the
    direct-reference half can never be shadowed.
    """

    def arg_is_dynamic(
        arg: libsbml.ASTNode, scope_time: frozenset[str], shadowed: frozenset[str]
    ) -> bool:
        return _ast_side_is_time(arg, (time_names - shadowed) | scope_time)

    return arg_is_dynamic


def _make_time_relational_filter(
    time_names: frozenset[str] | set[str] = frozenset(),
) -> Callable[[libsbml.ASTNode, libsbml.ASTNode], bool]:
    """A ``keep`` predicate admitting every relational that moves with time.

    *Either* side time-dependent is enough (GH #259). The original test was
    *exactly* one side, written to exclude shapes like ``time < 2*time`` that
    compare time against itself — but it excludes real crossings too, because
    "moves with time" is transitive once an alias counts: ``2*time >= time+0.7``
    is the ``t >= 0.7`` edge with no constant side, and BIOMD0000000589's
    ``time >= i*24`` (``i := floor(time/24)``) is a genuine sawtooth crossing.
    Both were refused, and a pulse whose *opening* edge is refused is missed
    however well the closing edge is rooted.

    Admitting a relational that never actually flips costs one ExprTk evaluation
    per root-function call and nothing else: a discontinuity root is the boolean
    condition itself (``gout = evaluate(cond) - 0.5``), so its value is ±0.5 and
    it can neither vanish identically nor be bracketed where it does not change.
    That asymmetry — a missed crossing is a wrong trajectory, a spurious
    candidate is an evaluation — is why this errs toward admitting.
    """

    def keep(a: libsbml.ASTNode, b: libsbml.ASTNode) -> bool:
        return _ast_side_is_time(a, time_names) or _ast_side_is_time(b, time_names)

    return keep


def _collect_time_discontinuity_conditions(
    node: libsbml.ASTNode,
    func_defs: dict,
    out: list,
    local_params: dict[str, str] | None = None,
    time_names: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Append an ExprTk condition string for every inequality comparing ``time``
    against a non-time expression (a discontinuity edge), including the ones
    inside the function definitions *node* calls (GH #231).

    n-ary relationals (``a < b < c``) are split into the consecutive pairs
    MathML defines them as, so each emitted condition is a single threshold.
    Not necessarily a monotonic one: ``sin(10*time) > 0.99`` is true on windows
    narrower than a step, which a root on the boolean cannot bracket when a
    whole window lies inside one. Its crossings are placed as stops instead
    (:func:`bngsim._switch_sensitivity.crossings_with_periodic`, issue #714).
    piecewise / and / or / nested structure is reached at any depth.

    ``time_names`` are the assignment-rule targets that transitively read the
    ``time`` csymbol. A model routinely aliases it
    (``<assignmentRule variable="model_time"> time``) and then writes its whole
    schedule against the alias, so ``model_time > 1800`` names no csymbol on
    either side and was not a threshold to a csymbol-only test (GH #259). A
    formal parameter bound to such an alias is time-dependent for the same
    reason (GH #231). The C++ root function refreshes every assignment rule at
    the trial ``(t, y)`` before evaluating a trigger, so a condition written
    against an alias is evaluated with a live value.

    A name is read in the innermost frame that binds it: a formal shadows a
    model symbol of the same name, and is time-dependent only if *this* call
    bound it to something that is.
    """
    for body, scope, scope_time in _iter_funcdef_scan_frames(
        node, func_defs, _make_time_arg_predicate(time_names), local_params
    ):
        _collect_relational_edge_conditions(
            body,
            func_defs,
            out,
            keep=_make_time_relational_filter(
                (frozenset(time_names) - frozenset(scope or ())) | scope_time
            ),
            local_params=scope,
        )


def _collect_relational_edge_conditions(
    node: libsbml.ASTNode,
    func_defs: dict,
    out: list,
    *,
    keep: Callable[[libsbml.ASTNode, libsbml.ASTNode], bool] | None = None,
    local_params: dict[str, str] | None = None,
    equalities: list | None = None,
) -> None:
    """Append one root condition for each relational sub-expression in *node*.

    Event trigger roots are registered as the full Boolean trigger. That misses
    narrow compound pulses such as ``(S > lo) && (S < hi)`` when one integrator
    step sees the trigger false at both endpoints. Registering each relational
    edge separately forces a stop at the component threshold, where the normal
    event re-check can queue the event.

    ``keep(a, b)`` filters the emitted atoms by the two sides of the relational.
    Event triggers pass none — every trigger atom is a crossing worth stopping
    at by construction — while the rule / kinetic-law scan (GH #194) passes one
    that admits only the atoms reading integrated state.

    ``local_params`` is the lexical scope the emitted text is written in: the
    formal-parameter → call-site-expression binding when *node* is a function
    definition's body (see :func:`_iter_funcdef_scan_frames`), which the
    event-trigger scan walks as the time and state scans do. ``None`` is
    model scope.

    ``equalities``, a list, receives each two-sided ``==`` and ``!=`` the same
    way. Not a root: an equality on the clock holds (or fails) at one double,
    which no root finder brackets. The event-trigger scan asks for them so the
    run can stop on that double (``at (time == 2)``).

    A relational of a lone ``abs()``, or of a lone square against a number,
    is appended as the two edges it turns at (issue #904):
    ``abs(A - 5) < 0.01`` as ``A - 5`` against ``0.01`` and ``-0.01``. Its own
    truth is false at both ends of a step that passes A = 5, so a root on it
    reported nothing, and an event on it never fired. Each edge flips once
    there. The rate-law scan reads the same pieces
    (:func:`bngsim._switch_sensitivity._monotone_pieces`).
    """
    from bngsim._switch_sensitivity import _monotone_pieces

    global _INEQ_RELATIONAL_OPS
    if node is None:
        return
    if _INEQ_RELATIONAL_OPS is None:
        _INEQ_RELATIONAL_OPS = {
            libsbml.AST_RELATIONAL_LT: "<",
            libsbml.AST_RELATIONAL_LEQ: "<=",
            libsbml.AST_RELATIONAL_GT: ">",
            libsbml.AST_RELATIONAL_GEQ: ">=",
        }
    for n in _iter_ast_subtree(node):
        t = n.getType()
        nc = n.getNumChildren()
        if t in _INEQ_RELATIONAL_OPS and nc >= 2:
            op = _INEQ_RELATIONAL_OPS[t]
            for i in range(nc - 1):
                a, b = n.getChild(i), n.getChild(i + 1)
                if keep is not None and not keep(a, b):
                    continue
                lhs = _ast_to_exprtk_with_funcdefs(a, func_defs, local_params)
                rhs = _ast_to_exprtk_with_funcdefs(b, func_defs, local_params)
                atom = f"{lhs}{op}{rhs}"
                out.extend(f"({piece})" for piece in _monotone_pieces(atom) or [atom])
        elif (
            equalities is not None
            and nc == 2
            and t in (libsbml.AST_RELATIONAL_EQ, libsbml.AST_RELATIONAL_NEQ)
        ):
            lhs = _ast_to_exprtk_with_funcdefs(n.getChild(0), func_defs, local_params)
            rhs = _ast_to_exprtk_with_funcdefs(n.getChild(1), func_defs, local_params)
            op = "==" if t == libsbml.AST_RELATIONAL_EQ else "!="
            equalities.append(f"({lhs}{op}{rhs})")


def _iter_funcdef_scan_frames(
    node: libsbml.ASTNode,
    func_defs: dict,
    arg_is_dynamic: Callable[[libsbml.ASTNode, frozenset[str], frozenset[str]], bool],
    local_params: dict[str, str] | None = None,
) -> Iterator[tuple[libsbml.ASTNode, dict[str, str] | None, frozenset[str]]]:
    """Yield ``(subtree, local_params, local_dynamic)`` for *node* and for every
    function-definition body reachable from it.

    A threshold is just as often written one level down — ``MAX(a,b) :=
    piecewise(a, a >= b, b)`` on the state side (GH #194), ``stepfunc(t, t_start,
    …)`` on the time side (GH #231) — and a scan of the call site alone sees the
    *arguments* but never the threshold. Each callee is visited under a scope
    binding its formals to the call site's already-translated argument text, so
    the condition emitted from the body is the same string the RHS itself
    evaluates. ``seen`` bounds a self- or mutually-recursive definition to one
    expansion, and the walk is iterative for the GH #111 reason every other
    scanner here is.

    ``arg_is_dynamic(arg, scope_dynamic, shadowed)`` decides whether a call-site
    argument makes the formal it binds "dynamic" in the callee's frame — reading
    integrated state for the state scan, reading ``time`` for the time one. The
    yielded ``local_dynamic`` is that set, which each collector's ``keep`` filter
    reads to judge a relational written in the callee's own names.

    ``local_params`` seeds the outermost scope — for a kinetic law, its
    ``<localParameter>`` id → mangled-id map, without which the emitted
    condition names a symbol the evaluator does not have.
    """
    stack: list[tuple[libsbml.ASTNode, dict[str, str] | None, frozenset[str], frozenset[str]]] = [
        (node, local_params, frozenset(), frozenset())
    ]
    while stack:
        body, scope, scope_dynamic, seen = stack.pop()
        yield body, scope, scope_dynamic
        shadowed = frozenset(scope or ())
        for n in _iter_ast_subtree(body):
            if n.getType() != libsbml.AST_FUNCTION:
                continue
            fname = n.getName()
            entry = func_defs.get(fname)
            # A rateOf-idiom funcDef has no real body to descend into (GH #106).
            if entry is None or fname in seen or entry[1] is _RATEOF_FUNCDEF:
                continue
            param_names, callee_body = entry
            next_scope = dict(scope or {})
            next_dynamic = set(scope_dynamic)
            for j in range(min(n.getNumChildren(), len(param_names))):
                arg = n.getChild(j)
                pname = param_names[j]
                next_scope[pname] = _ast_to_exprtk_with_funcdefs(arg, func_defs, scope)
                if arg_is_dynamic(arg, scope_dynamic, shadowed):
                    next_dynamic.add(pname)
                else:
                    # The callee's own formal shadows anything of that name the
                    # caller's scope carried; it is dynamic only if THIS call
                    # bound it to something dynamic.
                    next_dynamic.discard(pname)
            stack.append((callee_body, next_scope, frozenset(next_dynamic), seen | {fname}))


# ── State-threshold discontinuity detection (GH #194) ────────────────────
#
# The time collector above roots `piecewise(dose, time < 6, 0)`. Its state twin
# — `piecewise(k_boost, X < hi and X > lo, k_base)` in an assignment rule or a
# kinetic law — matched none of the three collectors and registered nothing, so
# a window narrow in TIME (because the boosted rate is large) was stepped
# straight over and the trajectory came back bit-identical to "the window was
# never entered". The error does not move with rtol/atol, which is what
# separates it from an accuracy shortfall.
#
# What varies continuously is what can be bracketed: a species concentration, a
# rate-rule target, and anything an assignment rule builds out of those. A
# threshold against a genuine constant never crosses, and one against a
# parameter only an event writes changes only AT the event, where the
# integrator already restarts — neither earns a root, and leaving them out is
# what keeps this off the models that have nothing to gain.


def _state_dependent_names(sbml_model) -> set[str]:
    """Model symbols whose value moves continuously with the integrated state.

    Non-constant species and rate-rule targets are the primitives; assignment
    rules propagate the property to their targets, transitively (an assignment
    rule may be written in terms of another). Deliberately excludes the ``time``
    csymbol: a pure-time threshold is :func:`_collect_time_discontinuity_conditions`'s
    job, and a mixed one (``X < 2*time``) is emitted identically by both, so the
    builder's de-duplication makes it a single root either way.
    """
    dynamic: set[str] = set()
    for i in range(sbml_model.getNumSpecies()):
        sp = sbml_model.getSpecies(i)
        if not sp.getConstant() and sp.getId():
            dynamic.add(sp.getId())

    asg_math: dict[str, libsbml.ASTNode] = {}
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.getMath() is None:
            continue
        if rule.isRate():
            dynamic.add(rule.getVariable())
        elif rule.isAssignment():
            asg_math[rule.getVariable()] = rule.getMath()

    changed = True
    while changed:
        changed = False
        for var, math in asg_math.items():
            if var not in dynamic and (_ast_name_set(math) & dynamic):
                dynamic.add(var)
                changed = True
    return dynamic


def _make_state_relational_filter(
    dynamic_names: set[str],
    local_params: dict[str, str] | None = None,
    local_dynamic: frozenset[str] = frozenset(),
) -> Callable[[libsbml.ASTNode, libsbml.ASTNode], bool]:
    """A ``keep`` predicate for :func:`_collect_relational_edge_conditions` that
    admits exactly the relationals one of whose sides reads integrated state.

    Inside a function definition's body a name is a *formal parameter* first: it
    is state-dependent iff the call site bound it to something that is, which is
    what ``local_dynamic`` carries. A formal that shadows a model symbol
    (a parameter literally named after a species) therefore reads as its
    binding, not as the species — the emitted text substitutes it the same way.
    """
    shadowed = frozenset(local_params or ())

    def keep(a: libsbml.ASTNode, b: libsbml.ASTNode) -> bool:
        names = _ast_name_set(a) | _ast_name_set(b)
        return bool((names & local_dynamic) or ((names & dynamic_names) - shadowed))

    return keep


def _make_state_arg_predicate(
    dynamic_names: set[str],
) -> Callable[[libsbml.ASTNode, frozenset[str], frozenset[str]], bool]:
    """:func:`_iter_funcdef_scan_frames`' ``arg_is_dynamic`` for the state scan:
    a call-site argument that reads integrated state (directly, or through a
    formal of the caller's own frame that does)."""

    def arg_is_dynamic(
        arg: libsbml.ASTNode, scope_dynamic: frozenset[str], shadowed: frozenset[str]
    ) -> bool:
        names = _ast_name_set(arg)
        return bool((names & scope_dynamic) or ((names & dynamic_names) - shadowed))

    return arg_is_dynamic


def _collect_state_relational_conditions(
    node: libsbml.ASTNode,
    func_defs: dict,
    dynamic_names: set[str],
    out: list,
    local_params: dict[str, str] | None = None,
) -> None:
    """Append a root condition for every state-threshold relational feeding *node*,
    including the ones inside the function definitions it calls (GH #194)."""
    for body, scope, scope_dynamic in _iter_funcdef_scan_frames(
        node, func_defs, _make_state_arg_predicate(dynamic_names), local_params
    ):
        _collect_relational_edge_conditions(
            body,
            func_defs,
            out,
            keep=_make_state_relational_filter(dynamic_names, scope, scope_dynamic),
            local_params=scope,
        )


# ── State-dependent floor/ceiling discontinuity detection (GH #244) ──────
#
# Cases such as SBML semantic-suite 00028/00173/00269 use
# ``factorial(ceiling(S1 * 4))`` in a kinetic law or rate rule. The RHS is
# piecewise constant with jumps when the state-dependent argument crosses an
# integer. With no root registered, CVODE can step/interpolate across a jump and
# returns the same slightly shifted trajectory as RoadRunner. For the bounded,
# numeric cases we can identify safely at load time, register those integer
# crossings as no-op discontinuity roots so integration restarts exactly at each
# floor/ceiling boundary.
_STATE_DISC_ROOT_LIMIT = 64


def _assignment_time_dependent_names(sbml_model) -> set[str]:
    """Assignment-rule targets whose values depend on the SBML time csymbol."""
    asg_math: dict[str, libsbml.ASTNode] = {}
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isAssignment() and rule.getMath() is not None:
            asg_math[rule.getVariable()] = rule.getMath()

    td: set[str] = set()
    changed = True
    while changed:
        changed = False
        for var, math in asg_math.items():
            if var in td:
                continue
            if _ast_references_time(math) or (_ast_name_set(math) & td):
                td.add(var)
                changed = True
    return td


def _integer_thresholds_toward_zero(x0: float) -> list[int]:
    """Integer floor/ceiling boundaries between the initial value and zero."""
    if not _math.isfinite(x0):
        return []
    lo, hi = sorted((0.0, x0))
    first = int(_math.ceil(lo))
    last = int(_math.floor(hi))
    if last < first:
        return []
    scale = max(abs(x0), 1.0)
    eps = 1e-12 * scale
    vals = [k for k in range(first, last + 1) if abs(float(k) - x0) > eps]
    return vals if len(vals) <= _STATE_DISC_ROOT_LIMIT else []


def _collect_state_discontinuity_conditions(
    node: libsbml.ASTNode,
    func_defs: dict,
    base_ctx: dict,
    time_dependent_names: set[str],
    out: list[str],
    local_params: dict[str, str] | None = None,
    local_values: dict[str, float | None] | None = None,
) -> None:
    """Append root conditions for bounded state-dependent floor/ceiling jumps.

    ``local_params`` / ``local_values`` are a kinetic law's local parameters (id ->
    registered name, id -> value): the scope its names are written in. Without
    them a local `k` read as a global `k`, or as nothing (issue #580).
    """
    if node is None:
        return
    disc_funcs = {
        libsbml.AST_FUNCTION_FLOOR,
        libsbml.AST_FUNCTION_CEILING,
    }

    # Iterative pre-order walk (explicit stack). A recursive descent overflows
    # Python's frame limit on the deep left-leaning <plus/> chains that
    # BNG-roundtripped observables produce (one frame per operand, GH #111) — the
    # same reason every other AST scanner here is iterative. Each stack frame
    # carries the lexical scope threaded through user-defined-function inlining.
    stack: list[tuple[libsbml.ASTNode, dict, dict, set, set]] = [
        (node, dict(local_params or {}), dict(local_values or {}), set(), set())
    ]
    while stack:
        n, local_expr, local_values, local_time_dependent, seen_funcs = stack.pop()
        ntype = n.getType()

        if ntype in disc_funcs and n.getNumChildren() >= 1:
            arg = n.getChild(0)
            names = _ast_name_set(arg)
            if names and not (
                _ast_references_time(arg)
                or bool((names - local_expr.keys()) & time_dependent_names)
                or bool(names & local_time_dependent)
            ):
                ctx = dict(base_ctx)
                ctx.update(local_values)
                x0 = _eval_ast_numeric(arg, ctx, func_defs, time_value=None)
                if x0 is not None:
                    arg_expr = _ast_to_exprtk_recursive(arg, func_defs, local_expr)
                    for threshold in _integer_thresholds_toward_zero(float(x0)):
                        out.append(f"(({arg_expr})<{threshold:.17g})")

        # Preserve the recursive visit order: the callee body (visited under an
        # extended scope) comes before the call node's siblings, and children run
        # in source order. Push siblings reversed, then the body last so it is on
        # top of the stack and pops first — matching the old recursion exactly.
        for i in reversed(range(n.getNumChildren())):
            stack.append(
                (n.getChild(i), local_expr, local_values, local_time_dependent, seen_funcs)
            )

        if ntype == libsbml.AST_FUNCTION:
            fname = n.getName()
            if fname in func_defs and fname not in seen_funcs:
                param_names, body = func_defs[fname]
                if body is not _RATEOF_FUNCDEF:
                    ctx = dict(base_ctx)
                    ctx.update(local_values)
                    next_expr = dict(local_expr)
                    next_values = dict(local_values)
                    next_td = set(local_time_dependent)
                    for j in range(min(n.getNumChildren(), len(param_names))):
                        child = n.getChild(j)
                        pname = param_names[j]
                        child_names = _ast_name_set(child)
                        next_expr[pname] = _ast_to_exprtk_recursive(child, func_defs, local_expr)
                        val = _eval_ast_numeric(child, ctx, func_defs, time_value=None)
                        # An argument with no load-time value binds the formal to
                        # None: the evaluator reads that as unknown, so nothing that
                        # reads the formal folds (NaN did, through a comparison in
                        # a piecewise), and it still shadows the global of the
                        # formal's name that dropping it let through (#580).
                        next_values[pname] = val
                        if (
                            _ast_references_time(child)
                            or bool((child_names - local_expr.keys()) & time_dependent_names)
                            or bool(child_names & local_time_dependent)
                        ):
                            next_td.add(pname)
                        else:
                            # Rebinding clears what an outer formal of this name
                            # carried in.
                            next_td.discard(pname)
                    stack.append((body, next_expr, next_values, next_td, seen_funcs | {fname}))


# ── Numeric AST evaluator for initial assignments ────────────────────────

import math as _math  # noqa: E402  (kept near its only consumer below)

# Value of Avogadro's number. bngsim uses the current SI-exact constant
# 6.02214076e23 mol⁻¹ everywhere — both the SBML ``avogadro`` csymbol here and
# the BNG-language ``_NA`` built-in (src/expression.cpp). NOTE: the SBML L3 spec
# (§3.4.6) and the semantic test suite predate the 2019 SI redefinition and bake
# the older 6.02214179e23 into their references, so the cases that probe the
# avogadro magnitude to full precision (00960/00961/01323) sit 1.7e-7 off our
# value and stay failing — a deliberate physical-correctness choice, not a bug.
_AVOGADRO = 6.02214076e23


def _flatten_assoc_chain(node: libsbml.ASTNode, op_type: int) -> list:
    """Collect all operands of an associative chain rooted at ``node``.

    libsbml's MathML reader binarizes n-ary ``<plus/>``/``<times/>`` into a
    left-leaning chain ``(((a + b) + c) + d)``. Recursive AST evaluation on
    such a chain consumes one stack frame per operand; for BNG-emitted
    observables that sum hundreds of species, that exceeds Python's default
    recursion limit.

    This helper walks the chain iteratively, returning operands in
    left-to-right source order. It also handles already-flat n-ary nodes
    and mixed shapes.
    """
    operands: list = []
    work = [node]
    while work:
        n = work.pop()
        if n.getType() == op_type:
            nc = n.getNumChildren()
            # push children in reverse so the leftmost is processed first
            for i in range(nc - 1, -1, -1):
                work.append(n.getChild(i))
        else:
            operands.append(n)
    return operands


def _eval_ast_numeric(
    node: libsbml.ASTNode,
    ctx: dict,
    func_defs: dict | None = None,
    *,
    time_value: float | None = 0.0,
    avogadro_value: float | None = _AVOGADRO,
    rateof_resolver: Callable[[str, dict], float | None] | None = None,
) -> float | None:
    """Evaluate a MathML AST node numerically.

    Returns the computed float value, or None if evaluation fails
    (e.g., references an unknown symbol or unsupported construct).

    ``func_defs`` is the optional ``{name: (param_names, body_ast)}``
    dictionary collected from ``<listOfFunctionDefinitions>``. When
    supplied, calls to user-defined functions are inlined by binding
    each parameter to the numeric value of its argument and recursively
    evaluating the body in the extended context.

    ``time_value`` is what an ``AST_NAME_TIME`` node (the SBML ``time``
    csymbol) folds to. The default ``0.0`` is correct for initialAssignment
    evaluation (which runs at t=0). Pass ``time_value=None`` from contexts
    that fire later, e.g. event delay/priority expressions, so the eval
    bails to ``None`` and the caller falls back to the dynamic-expression
    path instead of treating ``time`` as the literal 0.

    ``avogadro_value`` is what an ``AST_NAME_AVOGADRO`` csymbol folds to.
    It defaults to Avogadro's number so an initialAssignment / assignment rule
    that reads ``avogadro`` resolves at load instead of staying at its default.
    Pass ``avogadro_value=None`` from the event delay/priority fold so the
    constant stays on the dynamic-expression path — folding it to a static
    priority int can flip simultaneous-event ordering (01662).

    ``rateof_resolver`` resolves a ``rateOf(<symbol>)`` csymbol (or COPASI
    idiom) to the instantaneous dx/dt at the *initial* state — the value an
    initialAssignment such as ``p2 = rateOf(S1)`` must take (SBML test
    01250/01251/01252/01254, GH #231). Signature ``resolver(name, ctx) ->
    float | None``; it reads the live ``ctx`` so the derivative reflects every
    initial value resolved so far. When ``None`` (every caller except the
    section-0 IA/AR fold) a rateOf csymbol stays non-foldable and returns
    ``None``, preserving the pre-#231 behavior on every other path.
    """
    t = node.getType()

    # Leaf: number
    if t == libsbml.AST_INTEGER:
        return float(node.getInteger())
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E):
        return node.getReal()
    if t == libsbml.AST_RATIONAL:
        d = node.getDenominator()
        return node.getNumerator() / d if d != 0 else None

    # Leaf: name
    if t == libsbml.AST_NAME:
        name = node.getName()
        return ctx.get(name)
    if t == libsbml.AST_NAME_TIME:
        return time_value
    if t == libsbml.AST_NAME_AVOGADRO:
        return avogadro_value

    # Constants
    if t == libsbml.AST_CONSTANT_PI:
        return _math.pi
    if t == libsbml.AST_CONSTANT_E:
        return _math.e
    if t == libsbml.AST_CONSTANT_TRUE:
        return 1.0
    if t == libsbml.AST_CONSTANT_FALSE:
        return 0.0

    nc = node.getNumChildren()

    def _c(i):
        return _eval_ast_numeric(
            node.getChild(i),
            ctx,
            func_defs,
            time_value=time_value,
            avogadro_value=avogadro_value,
            rateof_resolver=rateof_resolver,
        )

    # Unary minus
    if t == libsbml.AST_MINUS and nc == 1:
        a = _c(0)
        return -a if a is not None else None

    # Arithmetic. PLUS and TIMES are flattened iteratively because libsbml's
    # MathML reader represents an n-ary <plus/>/<times/> as a left-leaning
    # binary chain (((a+b)+c)+d), which costs O(N) Python frames per chain
    # and trips sys.getrecursionlimit() on BNG-emitted observables that sum
    # hundreds of species.
    if t == libsbml.AST_PLUS:
        operands = _flatten_assoc_chain(node, libsbml.AST_PLUS)
        total = 0.0
        for op in operands:
            v = _eval_ast_numeric(
                op,
                ctx,
                func_defs,
                time_value=time_value,
                avogadro_value=avogadro_value,
                rateof_resolver=rateof_resolver,
            )
            if v is None:
                return None
            total += v
        return total
    if t == libsbml.AST_MINUS and nc == 2:
        a, b = _c(0), _c(1)
        return a - b if a is not None and b is not None else None
    if t == libsbml.AST_TIMES:
        operands = _flatten_assoc_chain(node, libsbml.AST_TIMES)
        total = 1.0
        for op in operands:
            v = _eval_ast_numeric(
                op,
                ctx,
                func_defs,
                time_value=time_value,
                avogadro_value=avogadro_value,
                rateof_resolver=rateof_resolver,
            )
            if v is None:
                return None
            total *= v
        return total
    if t == libsbml.AST_DIVIDE:
        a, b = _c(0), _c(1)
        if a is None or b is None or b == 0:
            return None
        return a / b
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        a, b = _c(0), _c(1)
        if a is None or b is None:
            return None
        try:
            r = a**b
        except (ValueError, OverflowError):
            return None
        # Python returns a complex for a negative base raised to a non-integer
        # power; the real C engine yields NaN. Treat as non-foldable so the value
        # never leaks into a downstream comparison as a complex.
        return None if isinstance(r, complex) else r

    # Comparisons. MathML allows n-ary relationals: (x1 op x2 op ... op xn)
    # means (x1 op x2) AND (x2 op x3) AND ...; neq uniquely means
    # "all pairwise distinct".
    _cmp_ops = {
        libsbml.AST_RELATIONAL_EQ: lambda a, b: float(a == b),
        libsbml.AST_RELATIONAL_NEQ: lambda a, b: float(a != b),
        libsbml.AST_RELATIONAL_LT: lambda a, b: float(a < b),
        libsbml.AST_RELATIONAL_GT: lambda a, b: float(a > b),
        libsbml.AST_RELATIONAL_LEQ: lambda a, b: float(a <= b),
        libsbml.AST_RELATIONAL_GEQ: lambda a, b: float(a >= b),
    }
    if t in _cmp_ops:
        if nc < 2:
            return 1.0
        vals = [_c(i) for i in range(nc)]
        if any(v is None for v in vals):
            return None
        op = _cmp_ops[t]
        if t == libsbml.AST_RELATIONAL_NEQ:
            for i in range(nc):
                for j in range(i + 1, nc):
                    if vals[i] == vals[j]:
                        return 0.0
            return 1.0
        for i in range(nc - 1):
            if op(vals[i], vals[i + 1]) == 0.0:
                return 0.0
        return 1.0

    # Logical
    if t == libsbml.AST_LOGICAL_AND:
        vals = [_c(i) for i in range(nc)]
        if any(v is None for v in vals):
            return None
        return float(all(v != 0 for v in vals))
    if t == libsbml.AST_LOGICAL_OR:
        vals = [_c(i) for i in range(nc)]
        if any(v is None for v in vals):
            return None
        return float(any(v != 0 for v in vals))
    if t == libsbml.AST_LOGICAL_NOT:
        a = _c(0)
        return float(a == 0) if a is not None else None
    if t == libsbml.AST_LOGICAL_XOR:
        # n-ary xor = odd parity of the true operands (1-child ⇒ that operand's
        # truth value). Mirrors the ExprTk left-fold of boolean ``!=``.
        vals = [_c(i) for i in range(nc)]
        if any(v is None for v in vals):
            return None
        return float(sum(1 for v in vals if v != 0) % 2 == 1)
    if t == libsbml.AST_LOGICAL_IMPLIES:
        # implies(a, b) ≡ (not a) or b.
        a, b = _c(0), _c(1)
        if a is None or b is None:
            return None
        return float(a == 0 or b != 0)

    # Math functions
    _unary_funcs: dict[int, Callable[[float], float]] = {
        libsbml.AST_FUNCTION_ABS: abs,
        libsbml.AST_FUNCTION_CEILING: _math.ceil,
        libsbml.AST_FUNCTION_EXP: _math.exp,
        libsbml.AST_FUNCTION_FLOOR: _math.floor,
        libsbml.AST_FUNCTION_LN: _math.log,
        libsbml.AST_FUNCTION_SIN: _math.sin,
        libsbml.AST_FUNCTION_COS: _math.cos,
        libsbml.AST_FUNCTION_TAN: _math.tan,
        libsbml.AST_FUNCTION_ARCSIN: _math.asin,
        libsbml.AST_FUNCTION_ARCCOS: _math.acos,
        libsbml.AST_FUNCTION_ARCTAN: _math.atan,
        libsbml.AST_FUNCTION_SINH: _math.sinh,
        libsbml.AST_FUNCTION_COSH: _math.cosh,
        libsbml.AST_FUNCTION_TANH: _math.tanh,
        # Inverse hyperbolic (native in math) + reciprocal/inverse-reciprocal
        # trig as closed-form identities — mirrors the ExprTk emit so an
        # initialAssignment / stoichiometryMath fold over these functions
        # resolves (00958/01561/01562) instead of bailing to a 0 default.
        libsbml.AST_FUNCTION_ARCSINH: _math.asinh,
        libsbml.AST_FUNCTION_ARCCOSH: _math.acosh,
        libsbml.AST_FUNCTION_ARCTANH: _math.atanh,
        libsbml.AST_FUNCTION_SEC: lambda x: 1.0 / _math.cos(x),
        libsbml.AST_FUNCTION_CSC: lambda x: 1.0 / _math.sin(x),
        libsbml.AST_FUNCTION_COT: lambda x: _math.cos(x) / _math.sin(x),
        libsbml.AST_FUNCTION_SECH: lambda x: 1.0 / _math.cosh(x),
        libsbml.AST_FUNCTION_CSCH: lambda x: 1.0 / _math.sinh(x),
        libsbml.AST_FUNCTION_COTH: lambda x: _math.cosh(x) / _math.sinh(x),
        libsbml.AST_FUNCTION_ARCSEC: lambda x: _math.acos(1.0 / x),
        libsbml.AST_FUNCTION_ARCCSC: lambda x: _math.asin(1.0 / x),
        libsbml.AST_FUNCTION_ARCCOT: lambda x: _math.atan(1.0 / x),
        libsbml.AST_FUNCTION_ARCSECH: lambda x: _math.acosh(1.0 / x),
        libsbml.AST_FUNCTION_ARCCSCH: lambda x: _math.asinh(1.0 / x),
        libsbml.AST_FUNCTION_ARCCOTH: lambda x: _math.atanh(1.0 / x),
    }
    if t in _unary_funcs:
        a = _c(0)
        if a is None:
            return None
        try:
            return float(_unary_funcs[t](a))
        except (ValueError, OverflowError, ZeroDivisionError):
            return None

    if t == libsbml.AST_FUNCTION_LOG:
        if nc == 1:
            a = _c(0)
            return _math.log10(a) if a is not None and a > 0 else None
        if nc == 2:
            base, x = _c(0), _c(1)
            if base is None or x is None or base <= 0 or x <= 0:
                return None
            return _math.log(x) / _math.log(base)
        return None

    if t == libsbml.AST_FUNCTION_ROOT:
        if nc == 2:
            n, x = _c(0), _c(1)
            if n is None or x is None or n == 0:
                return None
            try:
                r = x ** (1.0 / n)
            except (ValueError, OverflowError):
                return None
            return None if isinstance(r, complex) else r
        if nc == 1:
            a = _c(0)
            return _math.sqrt(a) if a is not None and a >= 0 else None
        return None

    # L3v2 MathML: quotient, rem, max, min (mirror the ExprTk emit so a folded
    # constant param equals its runtime expression column).
    if t == libsbml.AST_FUNCTION_QUOTIENT:
        a, b = _c(0), _c(1)
        if a is None or b is None or b == 0:
            return None
        return float(_math.floor(a / b))
    if t == libsbml.AST_FUNCTION_REM:
        a, b = _c(0), _c(1)
        if a is None or b is None or b == 0:
            return None
        return a - b * _math.floor(a / b)
    if t == libsbml.AST_FUNCTION_MAX:
        vals = [_c(i) for i in range(nc)]
        if not vals or any(v is None for v in vals):
            return None
        return max(vals)
    if t == libsbml.AST_FUNCTION_MIN:
        vals = [_c(i) for i in range(nc)]
        if not vals or any(v is None for v in vals):
            return None
        return min(vals)

    # factorial(n) = Γ(n+1); the C engine emits ``tgamma`` for the same value.
    if t == libsbml.AST_FUNCTION_FACTORIAL:
        a = _c(0)
        if a is None:
            return None
        try:
            return _math.gamma(a + 1)
        except (ValueError, OverflowError):
            return None

    # Piecewise — evaluate lazily: test each condition first and only evaluate
    # the *taken* branch's value. The runtime engine evaluates only the selected
    # arm, so an un-taken arm may be undefined at fold time (e.g. a guarded real
    # cube root `if(x>0, x^(1/3), -(abs(x))^(1/3))` whose first arm is complex
    # for x<0 under Python's `**`). Eagerly folding both arms would surface that
    # as a spurious complex/NaN; lazy evaluation matches the engine exactly.
    if t == libsbml.AST_FUNCTION_PIECEWISE:
        i = 0
        while i < nc - 1:
            cond = _c(i + 1)
            if cond is None:
                # Undecided, so no branch is known to be taken. Reading it as
                # false returned a later branch's finite value for a condition
                # the engine decides (`1/0 > 0` is true), and a caller trusts a
                # fold that answered (issue #871).
                return None
            if cond != 0:
                return _c(i)
            i += 2
        if nc % 2 == 1:
            return _c(nc - 1)
        return 0.0

    # Lambda (return body)
    if t == libsbml.AST_LAMBDA:
        return _c(nc - 1)

    # rateOf(x) csymbol (GH #231): the instantaneous dx/dt. Foldable only when a
    # resolver is supplied (the section-0 IA/AR fold passes one); every other
    # caller leaves it None, so a rateOf in a kineticLaw/rule/trigger stays on
    # the dynamic accessor path (_rateof_call) rather than being mis-folded here.
    if t == libsbml.AST_FUNCTION_RATE_OF:
        if rateof_resolver is not None and nc >= 1:
            arg = node.getChild(0)
            if arg.getType() == libsbml.AST_NAME and arg.getName():
                return rateof_resolver(arg.getName(), ctx)
        return None

    # User-defined function call. Inline by binding param names to argument
    # values in a fresh context, then evaluate the body. Without this, an
    # initialAssignment / assignmentRule that calls a functionDefinition
    # (e.g. `S3 := multiply(k1, S2)`, SBML test 00635/00755) cannot be
    # constant-folded at load time and the species's t=0 raw value is used
    # instead of the AR-resolved value.
    if t == libsbml.AST_FUNCTION and func_defs is not None:
        fname = node.getName()
        if fname in func_defs:
            param_names, body = func_defs[fname]
            # COPASI rateOf idiom (GH #106): the instantaneous dx/dt. When a
            # resolver is supplied (the section-0 IA/AR fold), resolve the call's
            # bare-symbol argument to its initial derivative; otherwise the value
            # is unknown at fold time, so None, as for the csymbol above. (It was
            # the funcDef's NaN body, which an event delay folded to "delay nan"
            # and a priority emitted as an undefined `nan`, issue #1124.)
            if body is _RATEOF_FUNCDEF:
                if rateof_resolver is not None and nc >= 1:
                    arg = node.getChild(0)
                    if arg.getType() == libsbml.AST_NAME and arg.getName():
                        return rateof_resolver(arg.getName(), ctx)
                return None
            args = [_c(i) for i in range(nc)]
            if any(a is None for a in args):
                return None
            sub_ctx = dict(ctx)
            for j in range(min(len(param_names), len(args))):
                sub_ctx[param_names[j]] = args[j]
            return _eval_ast_numeric(
                body,
                sub_ctx,
                func_defs,
                time_value=time_value,
                avogadro_value=avogadro_value,
                rateof_resolver=rateof_resolver,
            )

    return None  # unsupported


def _make_rateof_initial_resolver(sbml_model, func_defs):
    """Build a ``resolver(name, ctx) -> float | None`` for ``rateOf(<name>)`` in
    an initialAssignment / assignment rule (GH #231 sub-cluster 1).

    ``rateOf(x)`` is the instantaneous dx/dt; in an initialAssignment it takes
    the value at the *initial* state. We compute it from the model the same way
    the runtime fills ``current_derivs``, but evaluated against ``ctx`` (the live
    t=0 numeric context), so a fold like ``p2 = rateOf(S1)`` resolves at load:

      * **rate-rule target** (parameter / species / compartment): the rate is the
        rule's RHS — ``dx/dt`` is stated directly (01250: ``rateOf(p1)`` where
        ``p1`` has rate rule ``1`` ⇒ ``1``).
      * **reaction-driven species**: ``dAmount/dt = cf·Σ net_stoich·kineticLaw``
        over every reaction, with ``cf`` the species' conversion factor or the
        model's (1 with neither), in the species' reporting units — amount-rate
        for a ``hasOnlySubstanceUnits`` species, else concentration-rate
        (``÷V``, less ``[S]·V'/V`` where a rate rule sizes the compartment). This
        mirrors ``current_derivs`` (which holds d[conc]/dt) and the sub-cluster-3
        ``report_rateof_amount`` ×V convention (01251/01252/01254, all V=1).
      * **boundary / constant species** with no rate rule: not changed by
        reactions ⇒ ``0``, or ``-[S]·V'/V`` for a concentration in a compartment
        a rate rule sizes (its amount holds).

    A rate rule or kinetic law can itself read a rateOf: ``p' = 0.5*rateOf(S3)``
    makes ``rateOf(p)`` half the rate of ``S3``. Such a rateOf is resolved
    through this resolver as well, on the same initial state.

    Returns ``None`` (fold fails, target keeps its declared value) when the
    derivative can't be evaluated — an unknown symbol, an unsupported kinetic
    law, a rateOf of something with no rate rule and no reaction, or rates that
    are written in terms of each other in a ring (an algebraic loop, which the
    run refuses too).
    """
    rate_rule_math: dict[str, libsbml.ASTNode] = {}
    for i in range(sbml_model.getNumRules()):
        r = sbml_model.getRule(i)
        if r.isRate() and r.isSetMath():
            rate_rule_math[r.getVariable()] = r.getMath()

    species_by_id = {
        sbml_model.getSpecies(i).getId(): sbml_model.getSpecies(i)
        for i in range(sbml_model.getNumSpecies())
    }
    idiom_names = {
        name for name, (_p, body) in (func_defs or {}).items() if body is _RATEOF_FUNCDEF
    }

    def _rateof_args(m) -> set[str]:
        """The symbols whose rateOf ``m`` reads, by the csymbol or by a call to
        COPASI's rateOf idiom."""
        return {
            n.getChild(0).getName()
            for n in _iter_ast_subtree(m)
            if (
                n.getType() == libsbml.AST_FUNCTION_RATE_OF
                or (n.getType() == libsbml.AST_FUNCTION and n.getName() in idiom_names)
            )
            and n.getNumChildren() >= 1
            and n.getChild(0).getType() == libsbml.AST_NAME
        }

    law_rates_read: dict[int, set[str]] = {}

    def _rates_read(name, ctx) -> set[str]:
        """The symbols whose rate ``_rate`` reads in working out the rate of
        ``name``: those a rate rule or a kinetic law of a reaction that changes
        it reads the rateOf of, and the size of the compartment that dilutes a
        concentration."""
        if name in rate_rule_math:
            return _rateof_args(rate_rule_math[name])
        sp = species_by_id.get(name)
        if sp is None:
            return set()
        out = set() if sp.getHasOnlySubstanceUnits() else {sp.getCompartment()}
        if sp.getBoundaryCondition() or sp.getConstant():
            return out
        for j in range(sbml_model.getNumReactions()):
            rxn = sbml_model.getReaction(j)
            kl = rxn.getKineticLaw()
            if kl is None or not kl.isSetMath() or not _net_stoich(rxn, name, ctx):
                continue
            if j not in law_rates_read:
                # A local parameter shadows the global of the same id, and
                # nothing changes it.
                law_rates_read[j] = (
                    _rateof_args(kl.getMath()) - _kinetic_law_local_values(kl).keys()
                )
            out |= law_rates_read[j]
        return out

    def _rates_first(name, ctx) -> list[str] | None:
        """``name`` and every symbol whose rate its rate reads, each after the
        ones its own rate reads; ``None`` where they read each other in a ring.
        The walk keeps its own stack, so a long chain of rates costs no depth."""
        order: list[str] = []
        on_path, finished = {name}, set()
        stack = [(name, iter(_rates_read(name, ctx)))]
        while stack:
            n, reads = stack[-1]
            nxt = next(reads, None)
            if nxt is None:
                stack.pop()
                on_path.discard(n)
                finished.add(n)
                order.append(n)
            elif nxt in on_path:
                return None
            elif nxt not in finished:
                on_path.add(nxt)
                stack.append((nxt, iter(_rates_read(nxt, ctx))))
        return order

    def _net_stoich(rxn, sid, ctx):
        """Signed product−reactant stoichiometry of ``sid`` in ``rxn`` (0 if absent)."""
        net = 0.0
        for getter, n, sign in (
            (rxn.getReactant, rxn.getNumReactants(), -1.0),
            (rxn.getProduct, rxn.getNumProducts(), +1.0),
        ):
            for k in range(n):
                sr = getter(k)
                if sr.getSpecies() != sid:
                    continue
                sval = None
                if sr.isSetStoichiometryMath() and sr.getStoichiometryMath() is not None:
                    sm = sr.getStoichiometryMath().getMath()
                    if sm is not None:
                        sval = _eval_ast_numeric(sm, ctx, func_defs)
                if sval is None:
                    srid = sr.getId() if hasattr(sr, "getId") else ""
                    if srid and srid in ctx:
                        sval = ctx[srid]
                    else:
                        s = sr.getStoichiometry()
                        sval = s if _math.isfinite(s) else None
                if sval is None:
                    return None
                net += sign * sval
        return net

    def _concentration_rate(name, sp, amt_rate, ctx):
        """d[S]/dt = (d amount/dt)/V - [S]·(dV/dt)/V. dV/dt is the compartment's
        rate rule, or 0 (issue #738: 01822 folded 1 for 0.675 without it)."""
        vol = ctx.get(sp.getCompartment(), 1.0)
        if not vol:
            return None
        dvol = resolver(sp.getCompartment(), ctx)
        if dvol is None:
            return None
        if dvol == 0.0:
            return amt_rate / vol
        conc = ctx.get(name)
        if conc is None:
            return None
        return amt_rate / vol - conc * dvol / vol

    def _rate(name, ctx):
        # 1) rate-rule target — dx/dt is the rule's RHS, in the target's units.
        # A rateOf the rule reads is the rate of its argument at the same
        # instant: `rateOf(p)` under `p' = 0.5*rateOf(S3)` was refused at
        # load, where the run and libRoadRunner read 0.2.
        if name in rate_rule_math:
            return _eval_ast_numeric(
                rate_rule_math[name], ctx, func_defs, rateof_resolver=resolver
            )
        sp = species_by_id.get(name)
        if sp is None:
            # A parameter / compartment with no rate rule is constant ⇒ 0. One
            # an assignmentRule sets is not; the load refuses a rateOf of it
            # first (_check_rateof_targets, issue #1046). A call to a
            # functionDefinition whose body reads rateOf reaches here with the
            # caller's argument, written out at load (issue #1124, item 8), or
            # is refused there.
            return 0.0
        # 2) species — reactions do not change a boundary or constant species, so
        # its amount holds. Held as a concentration in a compartment a rate rule
        # sizes, it is diluted all the same (suite 01117/01118 hold the amount as
        # the size grows), and the run's rateOf reads -[S]·V'/V; the fold gave 0
        # (issue #738).
        if sp.getBoundaryCondition() or sp.getConstant():
            if sp.getHasOnlySubstanceUnits() or sp.getCompartment() not in rate_rule_math:
                return 0.0
            return _concentration_rate(name, sp, 0.0, ctx)
        amt_rate = 0.0
        for j in range(sbml_model.getNumReactions()):
            rxn = sbml_model.getReaction(j)
            net = _net_stoich(rxn, name, ctx)
            if net is None:
                return None
            if net == 0.0:
                continue
            kl = rxn.getKineticLaw()
            if kl is None or not kl.isSetMath():
                return None
            _lv = _kinetic_law_local_values(kl)
            law_ctx = {**ctx, **_lv} if _lv else ctx
            v = _eval_ast_numeric(
                kl.getMath(),
                law_ctx,
                func_defs,
                # The rate of a local parameter is 0; the rate of any other
                # symbol is read on the model's state, not the law's.
                rateof_resolver=lambda n, _c, _lv=_lv: 0.0 if n in _lv else resolver(n, ctx),
            )
            if v is None:
                return None
            amt_rate += net * v
        # A conversion factor scales what the reactions do to the species: its
        # own, or the model's where it has none (issue #738). Left out, the fold
        # was 1/cf of what the run reads at t = 0 (01405: -0.5 for -1).
        cf_id = (
            sp.getConversionFactor()
            if sp.isSetConversionFactor()
            else (sbml_model.getConversionFactor() if sbml_model.isSetConversionFactor() else "")
        )
        # (Not where the reactions do nothing to it: 0 is 0 at any factor, and
        # a factor that reads this rate, `cf := 1 + rateOf(S3)` with S3 in no
        # reaction, had no value for it, issue #1124 item 9. `!= 0.0` on
        # purpose: a NaN sum, an unevaluable law, is not 0 and still reads the
        # factor and fails as before; do not tighten this to `> 0`.)
        if cf_id and amt_rate != 0.0:
            # A factor with no value yet fails the fold: the initialAssignment
            # or rule that gives it one refolds this on a later pass, and C2c
            # refuses one the model never gives.
            cf = ctx.get(cf_id)
            if cf is None:
                return None
            amt_rate *= cf
        # ``current_derivs`` is d[conc]/dt; rateOf reports amount-rate for a hOSU
        # species (sub-cluster 3) and concentration-rate otherwise.
        if sp.getHasOnlySubstanceUnits():
            return amt_rate
        return _concentration_rate(name, sp, amt_rate, ctx)

    # The rates worked out for the outermost rateOf being resolved, by symbol.
    # Every rate it reads is worked out first, in the order `_rates_first`
    # gives, so a rateOf read on the way finds its value here rather than
    # working it out again (a diamond of rates on rates would be read once per
    # path through it) or on a deeper stack.
    known_rates: list[dict | None] = [None]

    def resolver(name, ctx):
        known = known_rates[0]
        if known is not None:
            if name not in known:
                # Read through the body of a function, which the walk does not
                # follow. It is marked first so that a ring through it reads no
                # value rather than recursing.
                known[name] = None
                known[name] = _rate(name, ctx)
            return known[name]
        order = _rates_first(name, ctx)
        if order is None:
            return None  # rates written in terms of each other: no value
        known_rates[0] = known = {}
        try:
            for n in order:
                known[n] = _rate(n, ctx)
        finally:
            known_rates[0] = None
        return known[name]

    return resolver


# Past this many nodes, the texts of a size's jump (its condition both ways and
# the size on either side) read what more than one path reaches through
# functions of their own (issue #1124, item 6). About 20,000 characters.
_JUMP_TEXT_NODES = 5_000


# ── Periodic floor()/modulo dosing-discontinuity resolution (GH #88) ──────
#
# The #72 machinery above registers a CVODE root for every inequality that
# compares the `time` csymbol DIRECTLY against a fixed threshold (`time < 21`).
# That nails a *monotonic* edge — CVODE brackets the single crossing regardless
# of step size. A periodic chemo schedule, however, encodes its dose edges
# through floor()/modulo time arithmetic routed via intermediate assignment-rule
# parameters (MODEL1708310001 / Claret2009: `exposure` switches on
# `rem_time = time mod cycle`, then `frac = rem_time - floor(rem_time)` against
# Dose/dose-length thresholds). Those edges are (a) invisible to the direct-`time`
# scan, and (b) even if registered, a single boolean root for a *periodic* pulse
# is non-monotonic — CVODE can step straight over a narrow "on" window (both step
# endpoints read "off"), the very BIOMD0000000879 failure #72 fixed for explicit
# thresholds. With exponential growth (`dy/dt = (K_L - K_D·exposure)·y`) the
# stepped-over dose-decay compounds into a persistent offset: at the rr_parity
# sweep tol bngsim reads y(100)=1603 (analytical) / 1578 (fd) and RoadRunner
# 1570, while the exact segmented answer is 953.07 (every engine converges there
# only as its tol is tightened enough to resolve the 0.0625-day pulses).
#
# The robust resolution that DOES generalize to periodic floor/modulo edges is to
# bound the integrator step below the narrowest dose window, so no step can span
# a pulse — then every window gets an interior sample and the forcing is captured
# (oracle-exact 953.07 for any jacobian, tol-stable). We detect the periodic
# structure and numerically measure the narrowest discontinuity feature here,
# returning a recommended `max_step_size`; Simulator.run applies it unless the
# caller overrides. Returns None for any model with no time-dependent floor/ceil/
# modulo feeding the ODE RHS, so the integrator is byte-identical otherwise.

_DISC_FUNCS = None  # lazily built {floor, ceiling, rem} AST-type set (needs libsbml)


def _ast_name_set(node) -> set:
    """Set of every ``<ci>`` name referenced in the subtree rooted at ``node``."""
    return {n.getName() for n in _iter_ast_subtree(node) if n.getType() == libsbml.AST_NAME}


def _replace_exprtk_symbols(expr: str, replacements: dict[str, str]) -> str:
    """Replace safe ExprTk identifiers in *expr* without touching inserted text."""
    if not replacements:
        return expr
    keys = sorted(replacements, key=len, reverse=True)
    pat = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Za-z0-9_])"
    )
    return pat.sub(lambda m: f"({replacements[m.group(1)]})", expr)


# Past this many characters, a new size written out through its rules reads a
# rule more than one path reaches by a function of its own (issue #1124, item 4).
_RESIZE_INLINE_LIMIT = 20_000

# How many levels of rules ``inlined`` writes out; past them it reads a rule by
# name, which a cycle of rules needs.
_RESIZE_INLINE_DEPTH = 65

# A name in ExprTk text, as ``_replace_exprtk_symbols`` reads one: not the
# exponent of a number such as 1e-06.
_EXPRTK_NAME = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*")


def _rule_compartments_an_event_resizes(
    rule_math: dict,
    ar_comp_targets,
    resized_comps: list[tuple[str, str]],
    assigned: dict[str, str],
    rescaled_species: dict[str, list[str]],
    to_exprtk,
    cache: dict | None = None,
    define_helper=None,
) -> list[tuple[str, str]]:
    """``(compartment, new size)`` for each assignment-rule compartment an event
    resizes that ``resized_comps`` does not already hold, each after the ones
    whose resize it reads (issue #1055).

    A compartment is resized by an event that changes anything its size rule
    reads: a variable the event assigns, another compartment the event resizes,
    or a species whose concentration that resize rescales (a concentration in a
    resized compartment keeps its amount, so it moves). And the rule is read
    through other assignment rules, not only its own names: ``cv := 1 + q``,
    ``q := 0.5*p`` is resized by ``p := 4``. Only the rule's own names were read
    before, so both kinds of indirection left the compartment's species
    unrescaled.

    The order matters because the engine reads each new size after the rescales
    listed before it. So the set is found first and ordered after: a
    compartment listed when it first reaches a change could still read a
    species a later one rescales (``cw := p + 0.1*Sn`` with Sn in ``cv :=
    1 + 0.1*Kw``, and an event setting p and Kw's compartment), and its size
    was then read with that species unmoved.

    The new size is the rule with the rules it reaches written out and the
    event's values put in: an assigned variable's value, a resized
    compartment's new size, and a rescaled species' value after the rescale.
    The event's sensitivity support reads that text.

    A piecewise size's jump (issue #1054) asks the same of the concentrations
    it rescales, given as the values it assigns them.

    ``rule_math`` maps each assignment rule's variable to its math, in document
    order; ``assigned`` maps each variable the event assigns to its value;
    ``rescaled_species`` maps a compartment to the species a resize of it
    rescales; ``to_exprtk`` writes math as ExprTk text. ``cache``, kept by the
    caller from one event to the next, holds what is read from the rules
    alone, so that each rule is walked once per model rather than once per
    event.

    Written out, a rule that more than one path reaches is written once per
    path, and a diamond of rules (``r_i := r_{i-1} + r_{i-2}``) made the text
    exponential in its depth: 278 KB at 22 rules, 1.9 MB at 26 (issue #1124,
    item 4). Where the text would pass ``_RESIZE_INLINE_LIMIT``, and the caller
    gives ``define_helper(key, text) -> name``, each such rule's value after
    the event is a function of its own instead, which the rules reaching it
    read by name: the text is then as long as the rules are. A function reads
    the state the event fires from, as the text does. The length counts the
    event's values put in, not only the rules: under 16 rules, an assigned
    value of 60 terms made the new size 720 KB. Below the limit the text is
    written out as before, except where the rules go deeper than ``inlined``
    writes them out: a chain of 70 rules read its deepest by name, at the
    value before the event.
    """
    cache = {} if cache is None else cache
    reads_memo: dict[str, frozenset] = cache.setdefault("reads", {})

    def reads(var: str) -> frozenset:
        """Every name the rule for ``var`` reads, itself or through the rules
        it reads. Each rule is walked once, which a diamond of rules needs."""
        if var not in reads_memo:
            reads_memo[var] = frozenset()  # a cycle of rules reads no further
            names: set[str] = set()
            for n in _ast_name_set(rule_math[var]):
                names.add(n)
                if n in rule_math:
                    names |= reads(n)
            reads_memo[var] = frozenset(names)
        return reads_memo[var]

    def moves(comp: str) -> set[str]:
        """The compartment and the species a resize of it rescales."""
        return {comp, *rescaled_species.get(comp, ())}

    changed = set(assigned)
    for comp, _ in resized_comps:
        changed |= moves(comp)
    resized: list[str] = []
    progress = True
    while progress:
        progress = False
        for comp in rule_math:
            if comp in ar_comp_targets and comp not in changed and reads(comp) & changed:
                resized.append(comp)
                changed |= moves(comp)
                progress = True

    # Each one after every other whose size or species it reads. Two that read
    # each other's species have no such order and keep the document's.
    ordered: list[str] = []
    pending = list(resized)
    while pending:
        comp = next(
            (c for c in pending if not any(reads(c) & moves(d) for d in pending if d != c)),
            pending[0],
        )
        pending.remove(comp)
        ordered.append(comp)

    replacements = {_safe_name(var): expr for var, expr in assigned.items()}

    def note_rescaled(comp: str, new_size: str) -> None:
        # A species the event also assigns is given that value in the old size,
        # and keeps the amount it makes there (as the rescale the caller
        # injects for it does).
        for sid in rescaled_species.get(comp, ()):
            amount = f"({assigned[sid]})" if sid in assigned else _safe_name(sid)
            replacements[_safe_name(sid)] = f"{amount} * ({_safe_name(comp)}) / ({new_size})"

    for comp, new_size in resized_comps:
        note_rescaled(comp, new_size)

    if "rule_of" not in cache:
        cache["rule_of"] = {_safe_name(var): var for var in rule_math}
    rule_of: dict[str, str] = cache["rule_of"]
    rule_text: dict[str, str] = cache.setdefault("text", {})

    def inlined(expr: str, depth: int = 0) -> str:
        """The text with each rule it reads that reaches a change written out,
        so that the event's values can be put in."""
        if depth >= _RESIZE_INLINE_DEPTH:
            return expr
        inner = {}
        for name in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)):
            var = rule_of.get(name)
            if var is not None and var not in changed and reads(var) & changed:
                if var not in rule_text:
                    rule_text[var] = to_exprtk(rule_math[var])
                inner[name] = rule_text[var]
        if not inner:
            return expr
        return inlined(_replace_exprtk_symbols(expr, inner), depth + 1)

    rule_names: dict[str, list[str]] = cache.setdefault("names", {})

    def names_in(var: str) -> list[str]:
        """The names ``var``'s rule reads, once for each reading."""
        if var not in rule_names:
            if var not in rule_text:
                rule_text[var] = to_exprtk(rule_math[var])
            rule_names[var] = _EXPRTK_NAME.findall(rule_text[var])
        return rule_names[var]

    def reached_in(var: str) -> dict[str, int]:
        """The rules ``var``'s rule reads that reach a change, and how many
        times it reads each."""
        counts: dict[str, int] = {}
        for name in names_in(var):
            inner = rule_of.get(name)
            if inner is not None and inner not in changed and reads(inner) & changed:
                counts[inner] = counts.get(inner, 0) + 1
        return counts

    def through_helpers(comp: str) -> str | None:
        """The new size of ``comp`` written out rule by rule, each rule more
        than one path reaches read through a function of its own where the
        text would pass the limit. None where ``inlined`` writes it in full
        and short enough, or the rules read each other in a cycle."""
        children: dict[str, dict[str, int]] = {}
        written: dict[str, int] = {}  # length written out; -1 while under way
        depth: dict[str, int] = {}  # levels of rules written out below it

        def length(var: str) -> int | None:
            if var in written:
                return None if written[var] < 0 else written[var]
            written[var] = -1
            children[var] = reached_in(var)
            # Each name read becomes its text in parentheses.
            n = len(rule_text[var]) + sum(
                len(replacements[name]) + 2 - len(name)
                for name in names_in(var)
                if name in replacements
            )
            d = 0
            for inner, k in children[var].items():
                m = length(inner)
                if m is None:
                    return None
                n += k * (m + 2 - len(_safe_name(inner)))
                d = max(d, depth[inner] + 1)
            written[var] = n
            depth[var] = d
            return n

        total = length(comp)
        if total is None:
            return None
        share = define_helper is not None and total > _RESIZE_INLINE_LIMIT
        if not share and depth[comp] <= _RESIZE_INLINE_DEPTH:
            return None
        paths: dict[str, int] = {}
        for reads_of in children.values():
            for inner, k in reads_of.items():
                paths[inner] = paths.get(inner, 0) + k
        after: dict[str, str] = {}

        def value_after(var: str) -> str:
            # The rule's text with what it reads after the event put in.
            if var not in after:
                subs = {name: replacements[name] for name in names_in(var) if name in replacements}
                for inner in children[var]:
                    subs[_safe_name(inner)] = (
                        helper_of(inner) if share and paths[inner] > 1 else value_after(inner)
                    )
                after[var] = _replace_exprtk_symbols(rule_text[var], subs)
            return after[var]

        helpers: dict[str, str] = {}

        def helper_of(var: str) -> str:
            if var not in helpers:
                helpers[var] = define_helper(
                    f"{_safe_name(comp)}_{_safe_name(var)}", value_after(var)
                )
            return helpers[var]

        return value_after(comp)

    found: list[tuple[str, str]] = []
    for comp in ordered:
        through = through_helpers(comp)
        new_size = (
            through
            if through is not None
            else _replace_exprtk_symbols(inlined(to_exprtk(rule_math[comp])), replacements)
        )
        found.append((comp, new_size))
        replacements[_safe_name(comp)] = new_size
        note_rescaled(comp, new_size)
    return found


def _piecewise_branches(node) -> tuple[list, list]:
    """``(condition_children, value_children)`` of a MathML ``piecewise``.

    Each ``<piece>`` contributes a value at an even index and its condition at
    the next odd one; a trailing ``<otherwise>`` is a value with no condition.
    """
    nc = node.getNumChildren()
    conds, vals = [], []
    i = 0
    while i < nc:
        if i + 1 < nc:
            vals.append(node.getChild(i))
            conds.append(node.getChild(i + 1))
            i += 2
        else:
            vals.append(node.getChild(i))
            i += 1
    return conds, vals


def _periodic_disc_escapes_roots(
    rhs_maths: list,
    asg_math: dict,
    td: set,
    func_defs: dict,
    registered_conditions: set,
) -> bool:
    """Does a periodic discontinuity reach the RHS by a path no root stops at?

    GH #274. The GH #88 step bound and the GH #72/#231/#259 roots address the
    same hazard — an integrator step spanning a schedule edge — by different
    means, and where a root already forces the stop the bound is pure cost
    (``max_step`` shortens every step over the whole horizon, not just the ones
    near an edge: up to 2.5x the internal steps on the corpus, for an answer
    that does not move). So the bound is kept only for what the roots do not
    reach, which is what this decides.

    A time-dependent ``floor``/``ceiling``/``modulo`` influences the RHS two
    ways, and only one of them is rooted:

    * through a piecewise CONDITION — ``exposure = piecewise(D, frac < w, 0)``.
      A registered root on that condition forces a stop exactly at the edge, so
      the branch cannot be stepped over and the bound adds nothing. That path is
      pruned here.
    * as a VALUE — ``dose = D * frac``, or ``k = k0 * i`` with
      ``i = floor(time/24)``. The sawtooth jump is then an RHS discontinuity in
      its own right, at an instant no relational brackets. Rooting some other
      condition in the same model does not help, and the bound is what keeps a
      step from spanning it.

    So this is value-position reachability from the ODE RHS, followed through
    the assignment rules and pruned at every rooted condition; ``True`` means
    some periodic disc node survives that pruning. ``rhs_maths`` holds
    ``(math, local_map)`` pairs, ``local_map`` being a kinetic law's local
    parameters (``{}`` for a rate rule).

    A note for anyone extending this: it is deliberately ONE pass that carries
    context down, rather than two passes compared afterwards. libsbml returns a
    *fresh* Python wrapper from every ``getChild()`` call, so ``id(node)`` is not
    stable across traversals and any "was this node also seen over there" test
    silently answers no — which reads as full root coverage and drops the bound
    everywhere.
    """
    disc_types = {
        libsbml.AST_FUNCTION_FLOOR,
        libsbml.AST_FUNCTION_CEILING,
        libsbml.AST_FUNCTION_REM,
    }

    def _scan(node, names_out: list, local_map: dict | None = None) -> bool:
        """True as soon as an unrooted time-dependent disc node is reached.

        ``local_map`` is a kinetic law's local parameters (id -> registered
        name): they shadow the globals, and a root on a condition that reads one
        was registered under its registered name (#580)."""
        local = local_map or {}
        stack = [node]
        while stack:
            n = stack.pop()
            ntype = n.getType()
            if ntype in disc_types and (
                _ast_references_time(n) or ((_ast_name_set(n) - local.keys()) & td)
            ):
                return True
            if ntype == libsbml.AST_FUNCTION_PIECEWISE:
                conds, vals = _piecewise_branches(n)
                stack.extend(vals)
                for cond in conds:
                    emitted: list[str] = []
                    _collect_time_discontinuity_conditions(
                        cond, func_defs, emitted, local_map or None, td
                    )
                    # Emitted AND registered ⇒ the integrator stops at this edge.
                    if emitted and all(e in registered_conditions for e in emitted):
                        continue
                    stack.append(cond)
                continue
            if ntype == libsbml.AST_NAME and n.getName() not in local:
                names_out.append(n.getName())
            stack.extend(n.getChild(k) for k in range(n.getNumChildren()))
        return False

    pending: list[str] = []
    for math, local_map in rhs_maths:
        if _scan(math, pending, local_map):
            return True
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in asg_math and _scan(asg_math[name], pending):
            return True
    return False


def _periodic_time_disc_max_step(
    sbml_model, func_defs, base_ctx, registered_conditions=None, reaction_local_maps=None
):
    """Recommend a ``max_step_size`` that resolves periodic floor/modulo dosing
    discontinuities in the ODE RHS, or ``None`` when the model has none.

    See the module comment above. The value is the narrowest discontinuity
    feature (dose-window width) over a representative horizon, divided by a
    safety factor so at least one integrator step lands inside every window.
    The narrowest feature is found numerically from the schedule's *structural*
    signature — the integer value of each floor/ceil/modulo and the active branch
    of each piecewise — which changes ONLY at a true edge (continuous values like
    `rem_time` move every sample and would mask the spacing).

    ``registered_conditions`` is the set of discontinuity conditions the caller
    just registered as CVODE roots. Given it, GH #274 returns ``None`` for a
    schedule those roots already cover, since a root forces the same stop the
    bound would and costs nothing between edges — see
    :func:`_periodic_disc_escapes_roots`. Omitting it keeps the pre-#274
    behaviour: a caller that has not run the root scan has no basis for that
    judgement, and the safe default is to bound.

    ``reaction_local_maps`` (reaction index -> {local id: registered name}) is
    each kinetic law's local-parameter scope. A law's floor/modulo nodes are
    evaluated with its locals over the globals they shadow; without them a local
    `period` read the global `period`, or nothing, and the bound came out wrong
    or not at all (#580).
    """
    global _DISC_FUNCS
    if _DISC_FUNCS is None:
        _DISC_FUNCS = {
            libsbml.AST_FUNCTION_FLOOR,
            libsbml.AST_FUNCTION_CEILING,
            libsbml.AST_FUNCTION_REM,
        }

    # Assignment rules (var → math) and the maths that ARE the ODE RHS, each RHS
    # math with its local scope: (math, local id -> value, local id -> name).
    asg_math: dict[str, libsbml.ASTNode] = {}
    rhs_maths: list[libsbml.ASTNode] = []
    rhs_scopes: list[tuple[dict, dict]] = []
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        m = rule.getMath()
        if m is None:
            continue
        if rule.isAssignment():
            asg_math[rule.getVariable()] = m
        elif rule.isRate():
            rhs_maths.append(m)
            rhs_scopes.append(({}, {}))
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl and kl.getMath():
            rhs_maths.append(kl.getMath())
            rhs_scopes.append(
                (_kinetic_law_local_values(kl), (reaction_local_maps or {}).get(i) or {})
            )
    if not asg_math or not rhs_maths:
        return None

    # Transitive time-dependence of assignment-rule variables (fixed point):
    # a var is time-dependent if its rule references the `time` csymbol or any
    # already-time-dependent var.
    td: set[str] = set()
    changed = True
    while changed:
        changed = False
        for var, m in asg_math.items():
            if var in td:
                continue
            if _ast_references_time(m) or (_ast_name_set(m) & td):
                td.add(var)
                changed = True

    # Symbols feeding the ODE RHS, closed transitively through assignment rules.
    feed: set[str] = set()
    work = [
        n
        for m, (lv, _) in zip(rhs_maths, rhs_scopes, strict=True)
        for n in _ast_name_set(m) - lv.keys()
    ]
    while work:
        name = work.pop()
        if name in feed:
            continue
        feed.add(name)
        if name in asg_math:
            work.extend(_ast_name_set(asg_math[name]))

    # Periodic iff some floor/ceil/modulo with a time-dependent argument actually
    # feeds the RHS. Plain `time < const` schedules (BIOMD879) have no such node
    # and fall through to None — their monotonic roots already resolve them.
    relevant = [(m, lv) for m, (lv, _) in zip(rhs_maths, rhs_scopes, strict=True)] + [
        (asg_math[v], {}) for v in feed if v in asg_math
    ]
    disc_nodes: list[tuple[int, libsbml.ASTNode, dict]] = []  # (kind, node, locals)
    piecewise_nodes: list[tuple[libsbml.ASTNode, dict]] = []
    has_periodic = False
    for m, lv in relevant:
        for n in _iter_ast_subtree(m):
            ntype = n.getType()
            if ntype in _DISC_FUNCS:
                arg_td = _ast_references_time(n) or bool((_ast_name_set(n) - lv.keys()) & td)
                disc_nodes.append((ntype, n, lv))
                if arg_td:
                    has_periodic = True
            elif ntype == libsbml.AST_FUNCTION_PIECEWISE:
                piecewise_nodes.append((n, lv))
    if not has_periodic:
        return None

    # GH #274 — the roots have overtaken the bound on most of the shapes it was
    # written for. Derive one only where they provably cannot reach; see
    # :func:`_periodic_disc_escapes_roots`. Checked before the numeric edge scan
    # below, which is the expensive part and is pure waste when the answer is
    # None. ``registered_conditions=None`` means the caller did not run the root
    # scan and cannot make this judgement, so the bound stands.
    if registered_conditions is not None and not _periodic_disc_escapes_roots(
        [(m, lm) for m, (_, lm) in zip(rhs_maths, rhs_scopes, strict=True)],
        asg_math,
        td,
        func_defs,
        registered_conditions,
    ):
        return None

    # Order the time-dependent assignment rules by dependency so each can be
    # evaluated as a pure function of t into a fresh ctx.
    td_rules: list[tuple[str, libsbml.ASTNode]] = []
    placed = set(base_ctx) | {"time"}
    pending = [(v, asg_math[v]) for v in td if v in asg_math]
    td_names = {v for v, _ in pending}
    while pending:
        progressed = False
        for v, m in list(pending):
            if (_ast_name_set(m) & td_names) <= placed:
                td_rules.append((v, m))
                placed.add(v)
                pending.remove((v, m))
                progressed = True
        if not progressed:  # cyclic AR (shouldn't happen) — append as-is
            td_rules.extend(pending)
            break

    def _structural_signature(t: float) -> tuple:
        ctx = dict(base_ctx)
        for v, m in td_rules:
            val = _eval_ast_numeric(m, ctx, func_defs, time_value=t)
            if val is not None:
                ctx[v] = val
        sig = []
        for kind, n, lv in disc_nodes:
            nctx = {**ctx, **lv} if lv else ctx
            if kind == libsbml.AST_FUNCTION_REM:
                a = _eval_ast_numeric(n.getChild(0), nctx, func_defs, time_value=t)
                b = _eval_ast_numeric(n.getChild(1), nctx, func_defs, time_value=t)
                sig.append(None if (a is None or b is None or b == 0) else _math.floor(a / b))
            else:
                a = _eval_ast_numeric(n.getChild(0), nctx, func_defs, time_value=t)
                fn = _math.floor if kind == libsbml.AST_FUNCTION_FLOOR else _math.ceil
                sig.append(None if a is None else fn(a))
        for n, lv in piecewise_nodes:
            nctx = {**ctx, **lv} if lv else ctx
            nc = n.getNumChildren()
            branch = nc  # default: "otherwise"
            i = 0
            idx = 0
            while i < nc - 1:
                cond = _eval_ast_numeric(n.getChild(i + 1), nctx, func_defs, time_value=t)
                if cond is not None and cond != 0:
                    branch = idx
                    break
                i += 2
                idx += 1
            sig.append(branch)
        return tuple(sig)

    # Median |d(arg)/dt| over a short probe — robust to the sawtooth jumps a
    # nested modulo argument takes, so it reads the underlying ramp rate (how
    # fast the floor argument climbs toward its next integer crossing).
    def _slope(arg_node, lv: dict) -> float:
        pts: list[tuple[float, float]] = []
        for k in range(1, 9):
            tt = 0.05 * k
            v = _eval_ast_numeric(arg_node, {**base_ctx, **lv}, func_defs, time_value=tt)
            if v is not None:
                pts.append((tt, v))
        diffs = sorted(
            abs((pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0]))
            for i in range(len(pts) - 1)
        )
        return diffs[len(diffs) // 2] if diffs else 0.0

    # Integer-crossing periods of every floor/ceil/modulo argument (= 1/slope):
    # the SLOWEST (largest period) sets how wide a window must be scanned to see
    # every branch phase; the FASTEST (smallest period) sets how fine the grid
    # must be to resolve the narrowest dose pulse within it.
    periods = []
    for kind, n, lv in disc_nodes:
        if kind == libsbml.AST_FUNCTION_REM:
            # rem(a,b) jumps when a/b crosses an integer; slope ≈ |a'|/|b|.
            a_s = _slope(n.getChild(0), lv)
            b = _eval_ast_numeric(n.getChild(1), {**base_ctx, **lv}, func_defs, time_value=0.0)
            s = a_s / abs(b) if b else 0.0
        else:
            s = _slope(n.getChild(0), lv)
        if s > 0.0:
            periods.append(1.0 / s)
    longest = max(periods) if periods else 1.0
    shortest = min(periods) if periods else 1.0
    # A few of the longest period covers every phase. The window is capped so a
    # model with one very long period beside a fast one stays cheap and finely
    # sampled: the cap is 128, or four of the FASTEST period when that is longer.
    # A flat 128 (issue #712) held less than one cycle of any schedule slower
    # than about 51 time units -- weekly dosing in hours, daily in minutes, or a
    # first pulse after t = 127 -- so the scan saw one edge or only the off
    # interval, returned None or a bound far wider than the pulse, and CVODE
    # stepped over every pulse after the first. Scaling with the fastest period
    # keeps several complete cycles of the schedule in view whatever its units,
    # and leaves a model with any period under 32 exactly as before.
    horizon = min(max(2.5 * longest, 4.0 * shortest, 4.0), max(128.0, 4.0 * shortest))

    # Scan the structural signature for edges. Samples at cell centres so the
    # scan never lands exactly on an integer edge — a measure-zero `0 < frac` flip
    # would forge spurious adjacent edges. Resolution targets ~1/100 of the
    # shortest period so a dose window down to a few percent of a cycle is caught
    # in a coarse cell; npts is bounded to keep this pure-Python scan sub-second.
    #
    # Each coarse sign change is bisected to its true edge time so the measured
    # spacing is the real window width, not the grid pitch. Without this, several
    # floor / piecewise components flipping in adjacent cells read as edges a
    # fraction of a cell apart and would shrink max_step needlessly (or,
    # pathologically, toward zero). Edges nearer than a tight epsilon are the same
    # physical instant (two components switching together) and collapse to one.
    #
    # The bisection runs per COMPONENT that changed, not on the signature as a
    # whole: the two edges of a pulse narrower than the grid pitch fall in one
    # cell, and bisecting the whole signature finds only the first of them. The
    # pulse then reads as a gap of a whole period, and the bound lets CVODE step
    # over every pulse. A floor/ceiling/modulo component flips once a period, so
    # per component nothing is lost to the grid; a piecewise branch that opens and
    # closes inside one cell still is.
    def _scan_edges(horizon: float) -> list[float]:
        npts = int(min(max(horizon / (shortest / 100.0), 2000.0), 16000.0))
        h = horizon / npts
        centres = [(k + 0.5) * h for k in range(npts)]
        sigs = [_structural_signature(t) for t in centres]
        found = []
        for k in range(1, npts):
            s_lo, s_hi = sigs[k - 1], sigs[k]
            if s_lo == s_hi:
                continue
            for j in range(len(s_lo)):
                if s_lo[j] == s_hi[j]:
                    continue
                lo, hi = centres[k - 1], centres[k]
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if _structural_signature(mid)[j] == s_lo[j]:
                        lo = mid
                    else:
                        hi = mid
                found.append(hi)
        return found

    edges = _scan_edges(horizon)
    if horizon > 128.0:
        # The widened window (issue #712) is sampled more coarsely than the
        # 128-unit one it replaces. Scan that one too, at its own pitch, so the
        # edges are a superset of what it found and the bound is never looser
        # than it was; that covers a piecewise branch too narrow for the wide grid.
        edges.extend(_scan_edges(128.0))
    edges.sort()
    if len(edges) < 2:
        return None  # no resolvable periodic feature
    eps = 1e-7 * max(horizon, 1.0)
    gaps = [g for g in (edges[i + 1] - edges[i] for i in range(len(edges) - 1)) if g > eps]
    if not gaps:
        return None
    gap = min(gaps)
    if gap >= horizon or gap <= 1e-9:
        return None
    return gap / 3.0


# ── SBML → ModelBuilder ──────────────────────────────────────────────────


#: Every SBML Level 3 package namespace starts here — ``<prefix>/<package>/<version>``.
_L3_PACKAGE_URI_PREFIX = "http://www.sbml.org/sbml/level3/"


def _package_name_from_uri(uri: str) -> str:
    """The package name in an SBML L3 package namespace URI, else the URI itself.

    ``…/sbml/level3/version1/arrays/version1`` → ``arrays``. Only needed for a
    package libSBML has no extension for: a registered one is asked its name
    directly (``SBMLDocumentPlugin.getPackageName``). Anything not shaped like
    an SBML L3 package URI is reported as itself rather than guessed at.
    """
    parts = uri.rstrip("/").split("/")
    if not uri.startswith(_L3_PACKAGE_URI_PREFIX) or len(parts) < 2 or not parts[-2]:
        return uri
    return parts[-2]


def _raw_required_namespaces(stream: IO) -> dict[str, str]:
    """``{uri: name}`` for every namespace the root ``<sbml>`` element marks
    ``required="true"`` (issue #613), read from *stream* rather than libSBML.

    This is the safety net for namespaces outside
    ``http://www.sbml.org/sbml/level3/…``: libSBML drops them entirely — no
    plugin, no unknown-package entry, no error — so its API cannot see them.
    ``iterparse`` stops at the first start tag, so only the opening ``<sbml>``
    element is read however large the document is, and it hands the attributes
    back in Clark notation (``{uri}required``). The name is
    :func:`_package_name_from_uri`'s, which for a URI outside the L3 hierarchy
    is the URI itself.
    """
    import xml.etree.ElementTree as ET

    required: dict[str, str] = {}
    try:
        for _event, root in ET.iterparse(stream, events=("start",)):
            for clark, value in root.attrib.items():
                is_required = clark.startswith("{") and clark.endswith("}required")
                if is_required and value.strip().lower() == "true":
                    uri = clark[1 : clark.index("}")]
                    required[uri] = _package_name_from_uri(uri)
            break  # the root element only
    except ET.ParseError:
        pass  # malformed XML — libSBML reports it separately
    return required


def _unhandled_required_packages(doc, raw_xml: IO | None = None) -> list[tuple[str, str]]:
    """``(name, namespace URI)`` for every ``required="true"`` package bngsim
    does not account for, sorted by name (issues #592, #613).

    Three sources, because libSBML splits them and silently drops one class:

    * A package libSBML has an extension for is a *plugin* on the document.
    * One it does not (``arrays`` in 5.21, anything newer than the installed
      libSBML) is an *unknown package*, which it still records with its
      ``required`` flag.
    * A namespace outside ``http://www.sbml.org/sbml/level3/…`` is neither —
      libSBML drops it without a word. When *raw_xml* (a readable stream over
      the document) is given, its root element is parsed with ``ElementTree``
      to catch these (issue #613).

    Only ``required="true"`` is collected. That is the whole design: SBML
    defines it to mean the package changes the model's mathematical meaning, so
    it is exactly the set whose content cannot be read past, while ``layout`` /
    ``render`` / ``fbc`` declare ``required="false"`` and go through untouched.

    Two things libSBML reports as required packages that are not, and the guard
    would refuse half the corpus without them:

    * **Level 2 and below.** ``required`` is an SBML *Level 3* attribute. On an
      L2 document libSBML still attaches its annotation-based ``layout`` and
      ``render`` plugins and answers ``getPackageRequired`` with ``True`` for
      both — for every L2 model, BIOMD0000000003 included. L2 packages are
      annotations and carry no mathematical meaning, so there is nothing to
      refuse.
    * **``l3v2extendedmath``.** libSBML models the extended math L3V2 folded
      into core as a plugin, under the *core* namespace itself, and reports it
      required. Every plain L3V2 document carries it. A plugin whose URI is the
      document's own core namespace is core, not a package.
    """
    if doc.getLevel() < 3:
        return []
    handled = _sbml_unsupported.handled_package_names()
    core_uri = doc.getSBMLNamespaces().getURI()
    found: dict[str, str] = {}
    for i in range(doc.getNumPlugins()):
        plugin = doc.getPlugin(i)
        name = plugin.getPackageName()
        uri = plugin.getURI()
        if uri == core_uri or name in handled:
            continue
        if doc.getPackageRequired(uri):
            found[name] = uri
    for i in range(doc.getNumUnknownPackages()):
        uri = doc.getUnknownPackageURI(i)
        name = _package_name_from_uri(uri)
        if name not in handled and doc.getPackageRequired(uri):
            found[name] = uri
    # GH #613: namespaces outside the standard L3 hierarchy are invisible to
    # libSBML's plugin and unknown-package APIs.  Parse the root element
    # directly and add any that libSBML missed.
    if raw_xml is not None:
        already_seen = set(found.values()) | {core_uri}
        for uri, name in _raw_required_namespaces(raw_xml).items():
            if uri not in already_seen and name not in handled:
                found[name] = uri
    return sorted(found.items())


def _check_required_packages(doc, source: str, raw_xml: IO | None = None) -> None:
    """Refuse a document whose declared ``required="true"`` package bngsim does
    not interpret, instead of silently building a model from the core layer
    (issue #592).

    The motivating case is ``multi``, the package rule-based models are encoded
    in: its species are templates carrying feature values and its reactions are
    parameterized by component maps, so reading only the core
    ``<listOfSpecies>`` / ``<listOfReactions>`` gives a model with the wrong
    species count, the wrong stoichiometry and no combinatorial expansion — or
    an empty one. The loader consulted exactly one package (``comp``) and read
    past every other as if it were not there, with no exception and no warning.

    Same posture as :func:`_check_unsupported_constructs`: a hard
    :class:`ModelError` unless ``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1``, which
    logs one warning per package and loads the core layer anyway.
    """
    import os

    offenders = _unhandled_required_packages(doc, raw_xml=raw_xml)
    if not offenders:
        return

    from bngsim._exceptions import ModelError

    if os.environ.get(_ALLOW_UNSUPPORTED_ENV) == "1":
        for name, uri in offenders:
            logger.warning(
                'SBML package %s [%s] is declared required="true" and bngsim does '
                "not interpret it; loading the core layer anyway (%s=1). The model "
                "being integrated is not the one in %s.",
                name,
                uri,
                _ALLOW_UNSUPPORTED_ENV,
                source,
            )
        return

    bullets = "\n".join(f"  - {name} [{uri}]" for name, uri in offenders)
    raise ModelError(
        "Model declares SBML Level 3 package(s) bngsim does not interpret, "
        'with required="true":\n'
        f"{bullets}\n"
        'SBML defines required="true" to mean the package changes the mathematical '
        "meaning of the model, so the core <listOfSpecies>/<listOfReactions> is not "
        "the model: reading past the package would integrate a different system and "
        "report it as this one. Packages bngsim does account for:\n"
        f"{_sbml_unsupported.handled_packages_note()}\n"
        "A presentation-only package (layout, render, fbc) declares "
        'required="false" and loads untouched. To read the core layer anyway — '
        f"knowing the result is not the model on disk — set {_ALLOW_UNSUPPORTED_ENV}=1."
    )


def _doc_uses_comp(doc) -> bool:
    """Whether *doc* actually composes models via the SBML ``comp`` package
    (hierarchical model composition, GH #230).

    True only when the document carries ModelDefinitions / ExternalModel-
    Definitions or its model carries Submodels — i.e. there is something to
    flatten. A bare ``comp`` namespace declaration with no composition is left
    untouched.
    """
    plugin = doc.getPlugin("comp")
    if plugin is None:
        return False
    if plugin.getNumModelDefinitions() > 0 or plugin.getNumExternalModelDefinitions() > 0:
        return True
    model = doc.getModel()
    if model is not None:
        mplugin = model.getPlugin("comp")
        if mplugin is not None and mplugin.getNumSubmodels() > 0:
            return True
    return False


def _flatten_comp(doc, base_path: str | None = None):
    """Flatten an SBML ``comp`` composed model into a plain single-model
    document (GH #230).

    bngsim has no native ``comp`` interpreter, so submodel-scoped variables
    cannot be resolved (they surface as ``var_missing`` downstream). libSBML's
    :class:`CompFlatteningConverter` inlines every submodel — renaming scoped
    ids, applying ReplacedElements / ReplacedBy / SBaseRef substitutions — so
    the existing flat-model pipeline handles the result unchanged. This mirrors
    how RoadRunner ingests ``comp`` models.

    Mutates *doc* in place. ``base_path`` is the directory used to resolve
    ExternalModelDefinitions (the SBML file's own directory); pass ``None`` for
    string loads, where external references cannot be resolved.

    Raises :class:`RuntimeError` if the converter does not return success — a
    composed model we cannot flatten is a hard failure, not a silent pass-
    through to a misleading ``var_missing``.
    """
    props = libsbml.ConversionProperties()
    props.addOption("flatten comp", True, "flatten comp package")
    # Be permissive: keep unflattenable non-comp packages stripped rather than
    # aborting, and skip the converter's own validation pass (the downstream
    # loader / suite grading is the authority on correctness).
    props.addOption("abortIfUnflattenable", "none")
    props.addOption("stripUnflattenablePackages", True)
    props.addOption("performValidation", False)
    if base_path is not None:
        props.addOption("basePath", base_path)
    status = doc.convert(props)
    if status != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(
            f"comp flattening failed (libSBML status {status}); "
            "hierarchical comp model could not be inlined"
        )
    return doc


def _apply_compartment_size_overrides(doc, compartment_sizes: dict, source: str) -> None:
    """Rewrite compartment sizes on the parsed document, before interpretation.

    Issue #164 — a compartment size is folded at load into constants no later
    write can reach (``Species.volume_factor``, an amount-declared
    ``initial_conc``, the Elementary scalar rate's ``Π V^n / V_storage``,
    ``Reaction.ssa_volume_factor``, the emitted ``inv_vf`` table), so
    ``set_param`` refuses it. This is the supported way to get the same model at
    a different volume: the size moves *before* any of those folds happen, so
    every one of them reads the new value — identical to editing the ``size=``
    attribute in the file, which is the oracle the tests use. A volume scan or a
    fit over a volume is a loop over loads through here.

    Applied after ``comp`` flattening, so the ids are the ones the flattened
    document actually carries.
    """
    model = doc.getModel()
    if model is None:  # pragma: no cover - _check_sbml_errors fires first
        raise ValueError(f"compartment_sizes= given but {source} has no model to apply it to")

    known = {model.getCompartment(i).getId() for i in range(model.getNumCompartments())}
    unknown = sorted(set(compartment_sizes) - known)
    if unknown:
        raise ValueError(
            f"compartment_sizes: unknown compartment(s) {unknown} in {source}. "
            f"Known compartments: {sorted(known)}. (These are SBML compartment ids as they "
            f"appear in the document — before bngsim's identifier mangling, and after any "
            f"``comp`` flattening.)"
        )

    # An ASSIGNMENT RULE defines its compartment's size at every point in time
    # (``tV := mV + dV``), so the ``size=`` attribute is not the model's volume
    # and overwriting it changes nothing. Refuse rather than accept a write that
    # the rule silently discards — the same reason set_param refuses. A RATE
    # RULE or an event assignment target is different: the size is that state's
    # *initial value*, which the override legitimately sets.
    ar_targets = {
        model.getRule(j).getVariable()
        for j in range(model.getNumRules())
        if model.getRule(j).isAssignment()
    }
    blocked = sorted(set(compartment_sizes) & ar_targets & known)
    if blocked:
        raise ValueError(
            f"compartment_sizes: {blocked} is set by an SBML assignment rule, so its size "
            f"attribute does not determine its volume — the rule recomputes it at every "
            f"time point and the override would have no effect. Change the rule's inputs "
            f"instead."
        )

    for cid, value in compartment_sizes.items():
        try:
            size = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"compartment_sizes['{cid}']: expected a number, got {value!r}"
            ) from None
        if not _math.isfinite(size) or size <= 0.0:
            raise ValueError(
                f"compartment_sizes['{cid}']: a compartment size must be finite and "
                f"positive, got {size!r}. (Zero would make the storage divide singular; "
                f"the loader's amount→concentration conversion divides by it.)"
            )
        # An initialAssignment for the compartment overrides the size attribute
        # at load (§1 reads ``ia_values`` in preference to ``getSize()``), so an
        # override that left it in place would be silently discarded. Dropping it
        # is what the caller asked for: they are naming the volume outright.
        if model.getInitialAssignment(cid) is not None:
            model.removeInitialAssignment(cid)
        comp = model.getCompartment(cid)
        if comp.setSize(size) != libsbml.LIBSBML_OPERATION_SUCCESS:  # pragma: no cover
            raise ValueError(f"compartment_sizes['{cid}']: libsbml rejected size {size!r}")


def load_sbml(path: str | Path, compartment_sizes: dict | None = None):
    """Load an SBML .xml file into a BNGsim NetworkModel.

    Uses libsbml directly to parse the SBML document.  Previous versions
    routed through libantimony for "symbol normalization", but antimony
    has process-global mutable state that corrupts under repeated use
    (segfault in getSBMLString after ~200 loads in a single process).
    The direct libsbml path handles all SBML semantics correctly and
    is both faster and safer.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SBML file not found: {path}")

    t0 = time.perf_counter()
    doc = libsbml.readSBMLFromFile(str(path))
    _check_sbml_errors(doc, str(path))
    with open(path, "rb") as stream:  # read as bytes, so the XML declaration names the encoding
        _check_required_packages(doc, str(path), raw_xml=stream)
    if _doc_uses_comp(doc):
        _flatten_comp(doc, base_path=str(path.parent))
    if compartment_sizes:
        _apply_compartment_size_overrides(doc, compartment_sizes, str(path))
    parse_sec = time.perf_counter() - t0
    model = _build_model_from_sbml_doc(doc)
    model._libsbml_parse_sec = parse_sec
    return model


def load_sbml_string(text: str, compartment_sizes: dict | None = None):
    """Load an SBML model from an XML string."""
    # libSBML parse phase (shared C++ core) — readSBML* + doc-level error check.
    # Timed here and stamped on the model; the doc → _core interpretation and the
    # analytical-Jacobian derivation are timed separately inside _build (T-instr).
    t0 = time.perf_counter()
    doc = libsbml.readSBMLFromString(text)
    _check_sbml_errors(doc, "<string>")
    _check_required_packages(doc, "<string>", raw_xml=io.StringIO(text))
    if _doc_uses_comp(doc):
        # No file context for a string load, so ExternalModelDefinitions cannot
        # be resolved; in-document ModelDefinitions/Submodels flatten fine.
        _flatten_comp(doc, base_path=None)
    if compartment_sizes:
        _apply_compartment_size_overrides(doc, compartment_sizes, "<string>")
    parse_sec = time.perf_counter() - t0
    model = _build_model_from_sbml_doc(doc)
    model._libsbml_parse_sec = parse_sec
    return model


def _import_antimony():
    """Import libantimony, or raise an ImportError that says how to install it.

    ``antimony`` is an optional dependency (it ships no sdist and its PyPI
    wheels cover fewer platforms than bngsim's own), so a bare ``import`` here
    surfaces as ``ModuleNotFoundError: No module named 'antimony'`` — which does
    not tell the caller that ``.ant`` support is an extra. Mirrors
    ``bngsim.missing_features()["antimony"]``.
    """
    try:
        import antimony as ant
    except ImportError:
        raise ImportError(
            "The 'antimony' package is required to load Antimony models. "
            "Install with: pip install 'bngsim[antimony]'"
        ) from None
    return ant


def load_antimony_via_sbml(path: str | Path, compartment_sizes: dict | None = None):
    """Load an Antimony .ant file by converting to SBML first."""
    ant = _import_antimony()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Antimony file not found: {path}")

    ant.clearPreviousLoads()
    ret = ant.loadFile(str(path))
    if ret == -1:
        err = ant.getLastError()
        raise RuntimeError(f"Failed to parse Antimony file {path}: {err}")

    mod = ant.getModuleNames()[-1]
    sbml_str = ant.getSBMLString(mod)
    if not sbml_str or len(sbml_str) < 10:
        raise RuntimeError(f"Antimony produced empty SBML for {path}")

    return load_sbml_string(sbml_str, compartment_sizes)


def load_antimony_string_via_sbml(text: str, compartment_sizes: dict | None = None):
    """Load an Antimony model string by converting to SBML first."""
    ant = _import_antimony()

    ant.clearPreviousLoads()
    ret = ant.loadString(text)
    if ret == -1:
        err = ant.getLastError()
        raise RuntimeError(f"Failed to parse Antimony string: {err}")

    mod = ant.getModuleNames()[-1]
    sbml_str = ant.getSBMLString(mod)
    if not sbml_str or len(sbml_str) < 10:
        raise RuntimeError("Antimony produced empty SBML")

    return load_sbml_string(sbml_str, compartment_sizes)


# The largest species power the SSA falling factorial takes (Reaction stores m
# as a C int); the mass-action classifier expands a power into that many
# factors, so it stops far sooner, at a power no stoichiometry reaches.
_SSA_FF_MAX_POWER = 2**31 - 1
_MASS_ACTION_MAX_POWER = 1000


def _math_names(node) -> tuple[set[str], bool]:
    """The names a math AST reads, and whether it reads the clock (the time
    csymbol, ``delay``). Iterative: a rule over a thousand terms is a
    thousand-deep tree.

    ``rateOf(x)`` reads ``x``: its rate moves where ``x`` does, and is 0 where
    nothing changes ``x``. Counted as the clock, `cf := 1 + rateOf(S3)` with S3
    in no reaction was refused as a conversion factor that moves, where
    libRoadRunner runs it with cf = 1 (issue #1124, item 9)."""
    names: set[str] = set()
    clock = False
    stack = [node] if node is not None else []
    while stack:
        n = stack.pop()
        t = n.getType()
        if t in (libsbml.AST_NAME_TIME, libsbml.AST_FUNCTION_DELAY):
            clock = True
        elif t == libsbml.AST_NAME:
            names.add(n.getName())
        stack.extend(n.getChild(i) for i in range(n.getNumChildren()))
    return names, clock


def _moving_assignment_rule_vars(sbml_model) -> tuple[dict, set[str]]:
    """The names each assignment rule reads, by variable, and the variables whose
    value moves during a run: one that reads the clock, a species that a
    reaction, a rule or an event changes, a rate-rule or event target, a reaction
    whose rate moves, or another moving rule's variable. A rule over constants
    (``C := 2``, ``C := p``) holds its value. A concentration moves with its
    compartment's size, so a constant species is a constant only in a
    compartment whose size holds."""
    reads: dict[str, set[str]] = {}
    moving: set[str] = set()
    rate_vars = set()
    algebraic_names: set[str] = set()
    for j in range(sbml_model.getNumRules()):
        r = sbml_model.getRule(j)
        if r.isAssignment() and r.getMath() is not None:
            names, clock = _math_names(r.getMath())
            reads[r.getVariable()] = names
            if clock:
                moving.add(r.getVariable())
        elif r.isRate():
            rate_vars.add(r.getVariable())
        elif r.isAlgebraic() and r.getMath() is not None:
            algebraic_names |= _math_names(r.getMath())[0]
    if not reads:
        return reads, moving  # no assignment rule, so none moves
    sources = set(rate_vars)
    for j in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(j)
        for k in range(ev.getNumEventAssignments()):
            sources.add(ev.getEventAssignment(k).getVariable())
    species = [sbml_model.getSpecies(j) for j in range(sbml_model.getNumSpecies())]
    # A reaction's id in math is that reaction's rate, its kinetic law's value,
    # so a rule that reads it moves when the law does. `cf := J0`, with J0 at
    # 0.5*S1, was taken for still, and the conversion factor was folded at its
    # load value of 0.5: S1(1) was 0.7788 where libRoadRunner, which follows the
    # rate, gives 0.8000. A law's local parameters shadow the model's names.
    # Only the laws a rule reads, directly or through other laws, are walked: a
    # name that is no species, parameter, compartment or rule variable.
    known = (
        {sp.getId() for sp in species}
        | {sbml_model.getParameter(j).getId() for j in range(sbml_model.getNumParameters())}
        | {sbml_model.getCompartment(j).getId() for j in range(sbml_model.getNumCompartments())}
        | set(reads)
    )
    law_reads: dict[str, set[str]] = {}
    pending = set().union(*reads.values()) - known
    while pending:
        rid = pending.pop()
        if rid in law_reads or rid in sources:
            continue
        rxn = sbml_model.getReaction(rid)
        kl = rxn.getKineticLaw() if rxn is not None else None
        if kl is None or kl.getMath() is None:
            continue
        names, clock = _math_names(kl.getMath())
        if clock:
            sources.add(rid)
            continue
        law_reads[rid] = names - {lp.getId() for lp in _kinetic_law_local_params(kl)}
        pending |= law_reads[rid] - known
    # A species moves when a reaction changes it, as a reactant or a product
    # that is not a boundary species; a rate rule, an event or an assignment
    # rule that moves it is counted with the rules and events. A species that
    # nothing changes holds its value, as a constant one does: `C := 1 +
    # 0.1*S`, with S in no reaction, holds C at its load-time size, where SSA
    # refused C's reactions as if it moved. An algebraic rule can set any name
    # it reads, so its species move. Only the species a rule or a law reads
    # matter here, and the reactions are scanned only for those.
    read = set().union(*reads.values(), *law_reads.values())
    candidates = {sp.getId() for sp in species if not sp.getConstant()} & read
    # A species a reaction changes: listed with a net stoichiometry that is not
    # 0. A catalyst written as reactant and product, `S3 -> S3`, is left as it
    # was, and its rate is 0 (the load-time fold skips such a reaction too);
    # `cf := 1 + rateOf(S3)` beside one was refused as moving, where
    # libRoadRunner runs it with cf = 1 (#1138's review). A stoichiometry that
    # is not a plain finite number (set by math, by a rule, or not constant) is
    # taken as moving the species, as before.
    changed_by_reaction: set[str] = set()
    for j in range(sbml_model.getNumReactions() if candidates - sources else 0):
        rxn = sbml_model.getReaction(j)
        net: dict[str, float] = {}
        for lst, sign in ((rxn.getListOfReactants(), -1.0), (rxn.getListOfProducts(), 1.0)):
            for k in range(lst.size()):
                sr = lst.get(k)
                sid = sr.getSpecies()
                stoich = sr.getStoichiometry()
                if (
                    sr.isSetStoichiometryMath()
                    or not sr.getConstant()
                    or not _math.isfinite(stoich)
                ):
                    changed_by_reaction.add(sid)
                    continue
                net[sid] = net.get(sid, 0.0) + sign * stoich
        changed_by_reaction |= {sid for sid, v in net.items() if v != 0.0}
    sources |= {
        sp.getId()
        for sp in species
        if sp.getId() in candidates
        and (
            (sp.getId() in changed_by_reaction and not sp.getBoundaryCondition())
            or sp.getId() in algebraic_names
        )
    }
    # A resize keeps a species' amount, so the concentration of a constant or
    # boundary species moves with its compartment's size, though nothing names
    # it. `cv := 1 + 0.1*Kw` with Kw constant in `ce` was taken for still across
    # an event `ce = 2`, and SSA ran cv's reactions at its load-time size.
    conc_comp = {
        sp.getId(): sp.getCompartment() for sp in species if not sp.getHasOnlySubstanceUnits()
    }
    changed = True
    while changed:  # to the fixed point: a rule over a moving rule moves
        changed = False
        for sid, comp in conc_comp.items():
            if sid not in sources and (comp in sources or comp in moving):
                sources.add(sid)
                changed = True
        for rid, names in law_reads.items():
            if rid not in sources and (names & sources or names & moving):
                sources.add(rid)
                changed = True
        for var, names in reads.items():
            if var not in moving and (names & sources or names & moving):
                moving.add(var)
                changed = True
    return reads, moving


def _ssa_rule_event_conc_reads(
    sbml_model, rules, moving_comps, species_comp, species_hosu, rule_targets=frozenset()
) -> list[tuple[str, list[str]]]:
    """Rate rules and events that read a concentration-valued species in a
    compartment whose size moves (``moving_comps``: assignment-rule, event- or
    rate-rule-resized), directly or through assignment rules: the SSA stores a
    count over the load-time size, so such a read is stale. ``x' = S`` with S in
    ``C := 1 + 0.5*time`` integrated 40 against the ODE's 22, and ``at S < 6``
    never fired. ``rules`` maps each assignment rule's variable, and each
    reaction id a rule or a law reads, to the names it reads. Returns
    ``[(location, [compartments])]``."""
    if not moving_comps:
        return []

    def hits(math) -> list[str]:
        out: set[str] = set()
        work = list(_math_names(math)[0]) if math is not None else []
        seen: set[str] = set()
        while work:
            name = work.pop()
            comp = species_comp.get(name)
            if (
                comp in moving_comps
                and not species_hosu.get(name, False)
                and name not in rule_targets
            ):
                out.add(comp)
            if name in rules and name not in seen:
                seen.add(name)
                work.extend(rules[name])
        return sorted(out)

    found = []
    for j in range(sbml_model.getNumRules()):
        r = sbml_model.getRule(j)
        if r.isRate():
            h = hits(r.getMath())
            if h:
                found.append((f"rule:{r.getVariable()}", h))
    for j in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(j)
        maths = []
        if ev.isSetTrigger():
            maths.append(ev.getTrigger().getMath())
        if ev.isSetDelay():
            maths.append(ev.getDelay().getMath())
        if ev.isSetPriority():
            maths.append(ev.getPriority().getMath())
        maths.extend(
            ev.getEventAssignment(k).getMath() for k in range(ev.getNumEventAssignments())
        )
        h = sorted({c for m in maths for c in hits(m)})
        if h:
            found.append((f"event:{ev.getId() or j}", h))
    return found


def _ssa_moving_ar_comp_reads(
    rxn,
    rules,
    moving_vars,
    moving_comps,
    species_comp,
    species_hosu,
    varvol_comps=frozenset(),
    rule_targets=frozenset(),
) -> tuple[list[str], list[str]]:
    """The moving assignment-rule compartments holding a concentration-valued
    species that reaction ``rxn`` changes or reads: one of its reactants,
    products or modifiers, or one its kinetic law reads, directly or through the
    assignment rules it reads. The size itself is read live, and an amount-valued
    species is stored and read as an amount.

    Second, the event-resized or rate-rule compartments (``varvol_comps``) whose
    concentration-valued species the rate reads from outside them, or through an
    assignment rule: the SSA's live-volume correction covers the species of the
    compartment a reaction acts in, and no others.

    ``rules`` maps each assignment rule's variable, and each reaction id a rule
    or a law reads, to the names it reads (a reaction id reads its law). A
    species a rule sets (``rule_targets``) holds the value its rule gives it,
    not a count over a size, so reading it is not stale."""
    if not moving_comps and not varvol_comps:
        return [], []
    kl = rxn.getKineticLaw()
    local = set()
    if kl is not None:
        for j in range(kl.getNumLocalParameters()):
            local.add(kl.getLocalParameter(j).getId())
        for j in range(kl.getNumParameters()):
            local.add(kl.getParameter(j).getId())
    hit: set[str] = set()
    vv_hit: set[str] = set()
    acts_in = {
        species_comp.get(lst.get(j).getSpecies())
        for lst in (rxn.getListOfReactants(), rxn.getListOfProducts())
        for j in range(lst.size())
    }
    law_names = _math_names(kl.getMath())[0] - local if kl is not None and kl.getMath() else set()
    # A reaction across compartments, one of them resized: the live-volume
    # correction is a power of V_static/V_live per compartment, exact only when
    # the law reads each concentration there as a reactant, once per unit of its
    # stoichiometry. A modifier (`k*X*S*C`), a reactant read again (`k*X*X*C`
    # for X -> P) or a power is not that shape, and ran 16-24 sigma off the ODE.
    if len(acts_in - {None}) >= 2 and acts_in & set(varvol_comps) and kl is not None:
        stoich: dict[str, float] = {}
        for j in range(rxn.getNumReactants()):
            r = rxn.getReactant(j)
            stoich[r.getSpecies()] = stoich.get(r.getSpecies(), 0.0) + r.getStoichiometry()
        counts: dict[str, int] = {}
        powered: set[str] = set()
        stack = [kl.getMath()] if kl.getMath() is not None else []
        while stack:
            nd = stack.pop()
            t = nd.getType()
            if t == libsbml.AST_NAME and nd.getName() not in local:
                counts[nd.getName()] = counts.get(nd.getName(), 0) + 1
            if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
                powered |= _math_names(nd)[0]
            stack.extend(nd.getChild(i) for i in range(nd.getNumChildren()))
        for sid, cnt in counts.items():
            comp = species_comp.get(sid)
            if comp not in varvol_comps or species_hosu.get(sid, False) or sid in rule_targets:
                continue
            if sid not in stoich or sid in powered or cnt != stoich[sid]:
                vv_hit.add(comp)
    # (name, read through a rule); the touched species are read directly.
    work = [(n, False) for n in law_names | _ssa_species_touched(rxn)]
    seen_rules: set[str] = set()
    while work:
        name, via_rule = work.pop()
        comp = species_comp.get(name)
        conc = comp is not None and not species_hosu.get(name, False) and name not in rule_targets
        if conc and comp in moving_comps:
            hit.add(comp)
        if conc and comp in varvol_comps and (via_rule or comp not in acts_in):
            vv_hit.add(comp)
        if name in rules and name not in seen_rules:
            seen_rules.add(name)
            work.extend((n, True) for n in rules[name])
    return sorted(hit), sorted(vv_hit)


def _ssa_species_touched(rxn) -> set[str]:
    """Every species a reaction names: reactants, products and modifiers."""
    out = set()
    for lst in (rxn.getListOfReactants(), rxn.getListOfProducts(), rxn.getListOfModifiers()):
        for j in range(lst.size()):
            out.add(lst.get(j).getSpecies())
    return out


def _ssa_falling_factorial_terms(rxn, species_idx, continuous_ids) -> list[tuple[int, int]]:
    """The SSA falling factorial a kinetic law evaluated as written needs.

    A reaction the mass-action classifier declines (a boundary reactant,
    reactants in compartments of different sizes, an assignment-rule
    compartment, a law divided by a volume) is emitted Functional and its law
    is evaluated literally, so under SSA a species it reads m times contributes
    n^m where the number of distinct m-tuples of its molecules is
    n(n-1)...(n-m+1). ``2B -> P`` at ``k*B*B`` kept firing at k with a single B
    left, where the same reaction from a `.net` (``$B + $B``) or a law the
    classifier lifts fires at 0.

    Returns ``[(species index, m), ...]`` for every count species the law holds
    to a power m >= 2 when the law is a product: species only as factors of the
    numerator (bare, or to a non-negative integer power), anything else
    (numbers, parameters, compartments, functions of those) free to appear
    anywhere. A piecewise counts when every branch that is not a literal 0 is
    such a product of the same species (its conditions may read anything). A
    law that puts a species anywhere else (a sum such as ``X*(X-1)``, a
    function call, a denominator, piecewise branches that disagree) is its own
    statement of the combinatorics and gets nothing. A continuous slot (an
    assignment- or rate-rule target) is not a count and gets nothing, nor is a
    local parameter that shadows a species id.
    """
    kl = rxn.getKineticLaw()
    math = kl.getMath() if kl is not None else None
    if math is None:
        return []
    shadowed = set()
    for j in range(kl.getNumLocalParameters()):
        shadowed.add(kl.getLocalParameter(j).getId())
    for j in range(kl.getNumParameters()):
        shadowed.add(kl.getParameter(j).getId())

    def is_species(node) -> bool:
        return (
            node.getType() == libsbml.AST_NAME
            and node.getName() in species_idx
            and node.getName() not in shadowed
        )

    def mentions_species(node) -> bool:
        if is_species(node):
            return True
        return any(mentions_species(node.getChild(i)) for i in range(node.getNumChildren()))

    def integer_exponent(node):
        """A finite, integer-valued numeric literal, as an int; else None."""
        et = node.getType()
        if et == libsbml.AST_INTEGER:
            return node.getInteger()
        if et in (libsbml.AST_REAL, libsbml.AST_REAL_E):
            r = node.getReal()
            return int(r) if _math.isfinite(r) and r == int(r) else None
        if et == libsbml.AST_RATIONAL:
            num, den = node.getNumerator(), node.getDenominator()
            return num // den if den != 0 and num % den == 0 else None
        return None

    def is_zero(node) -> bool:
        et = node.getType()
        return (et == libsbml.AST_INTEGER and node.getInteger() == 0) or (
            et in (libsbml.AST_REAL, libsbml.AST_REAL_E) and node.getReal() == 0.0
        )

    def walk(node, numerator: bool, times: int, acc: Counter) -> bool:
        """Add each species factor of ``node`` (raised ``times``) to ``acc``."""
        t = node.getType()
        if t == libsbml.AST_TIMES:
            return all(
                walk(node.getChild(i), numerator, times, acc) for i in range(node.getNumChildren())
            )
        if t == libsbml.AST_DIVIDE and node.getNumChildren() == 2:
            return walk(node.getChild(0), numerator, times, acc) and walk(
                node.getChild(1), not numerator, times, acc
            )
        if is_species(node):
            if not numerator:
                return False
            acc[node.getName()] += times
            return True
        if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER) and node.getNumChildren() == 2:
            n = integer_exponent(node.getChild(1))
            if n is not None and n >= 0:
                # x^0 contributes nothing; (B*C)^2 contributes B twice.
                return n == 0 or walk(node.getChild(0), numerator, times * n, acc)
            return not mentions_species(node)
        if t == libsbml.AST_FUNCTION_PIECEWISE:
            # piecewise(k*B*B*C, cond, 0): the value in force is a product of the
            # same species wherever it is not 0, so the factorial is the same
            # whichever branch applies. Conditions only select, and may read
            # species freely. Branches that disagree are left as written.
            maps = []
            for i in range(0, node.getNumChildren(), 2):  # values; a trailing otherwise
                val = node.getChild(i)
                if is_zero(val):
                    continue
                sub: Counter = Counter()
                if not walk(val, numerator, times, sub):
                    return False
                maps.append(sub)
            if any(mp != maps[0] for mp in maps[1:]):
                return False
            if maps:
                acc.update(maps[0])
            return True
        return not mentions_species(node)

    mult: Counter = Counter()
    if not walk(math, True, 1, mult):
        return []
    if any(m > _SSA_FF_MAX_POWER for m in mult.values()):
        return []  # B^3e9: evaluated as written (the engine stores m as an int)
    return sorted(
        (species_idx[sid], m) for sid, m in mult.items() if m >= 2 and sid not in continuous_ids
    )


def _flatten_product_for_mass_action(node, out):
    """Walk a kinetic-law AST as a flat product, appending leaves to ``out``.

    Returns True iff the AST is a product of accepted leaves: ``AST_TIMES``
    (n-ary), ``AST_POWER`` with positive integer exponent (expanded into
    that many copies of the base), and pure leaves (numeric, name).
    Returns False on anything that breaks the mass-action shape — sums,
    differences, divisions, function calls, csymbols, unary minus, etc.
    """
    t = node.getType()
    if t == libsbml.AST_TIMES:
        for i in range(node.getNumChildren()):
            if not _flatten_product_for_mass_action(node.getChild(i), out):
                return False
        return True
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        if node.getNumChildren() != 2:
            return False
        base = node.getChild(0)
        exp = node.getChild(1)
        n_exp: int | None = None
        et = exp.getType()
        if et == libsbml.AST_INTEGER:
            n_exp = exp.getInteger()
        elif et in (libsbml.AST_REAL, libsbml.AST_REAL_E):
            r = exp.getReal()
            if _math.isfinite(r) and r == int(r):  # inf/nan: not an integer
                n_exp = int(r)
        if n_exp is None or n_exp < 1 or n_exp > _MASS_ACTION_MAX_POWER:
            return False
        for _ in range(n_exp):
            out.append(base)
        return True
    if t in (
        libsbml.AST_INTEGER,
        libsbml.AST_REAL,
        libsbml.AST_REAL_E,
        libsbml.AST_RATIONAL,
        libsbml.AST_NAME,
    ):
        out.append(node)
        return True
    return False


def _factor_minus_subtree(node):
    """Phase 7 reversible-split detection.

    Treat ``node`` as a product of mass-action-shape factors with exactly
    one embedded binary ``AST_MINUS`` subtree. On success returns
    ``(wrapper_leaves, minus_left, minus_right)`` — the wrapper leaves
    are AST nodes (numeric/name/expanded-power) that the splitter passes
    as ``prefix_factors`` to the per-side classifier, distributing them
    onto each operand of the MINUS. Returns ``None`` when the AST has
    no MINUS, more than one MINUS, or any non-mass-action-shape node
    (sums, divisions, function calls, csymbols, unary minus).

    Recognizes COPASI/Antimony-idiomatic reversible kineticLaws like
    ``compartment * (kf*A - kr*B)`` whose top-level AST is ``AST_TIMES``,
    not ``AST_MINUS`` — a shape the plan's "binary MINUS at top level"
    rule misses.
    """
    wrapper: list = []
    found: list = []  # at most one (left, right) pair

    def _walk(n) -> bool:
        t = n.getType()
        if t == libsbml.AST_TIMES:
            return all(_walk(n.getChild(i)) for i in range(n.getNumChildren()))
        if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
            if n.getNumChildren() != 2:
                return False
            base = n.getChild(0)
            exp = n.getChild(1)
            n_exp: int | None = None
            et = exp.getType()
            if et == libsbml.AST_INTEGER:
                n_exp = exp.getInteger()
            elif et in (libsbml.AST_REAL, libsbml.AST_REAL_E):
                r = exp.getReal()
                if _math.isfinite(r) and r == int(r):  # inf/nan: not an integer
                    n_exp = int(r)
            if n_exp is None or n_exp < 1 or n_exp > _MASS_ACTION_MAX_POWER:
                return False
            for _ in range(n_exp):
                wrapper.append(base)
            return True
        if t == libsbml.AST_MINUS:
            if n.getNumChildren() != 2:
                return False  # unary minus is rejected here
            if found:
                return False  # second MINUS — ambiguous, reject
            found.append((n.getChild(0), n.getChild(1)))
            return True
        if t in (
            libsbml.AST_INTEGER,
            libsbml.AST_REAL,
            libsbml.AST_REAL_E,
            libsbml.AST_RATIONAL,
            libsbml.AST_NAME,
        ):
            wrapper.append(n)
            return True
        return False

    if not _walk(node):
        return None
    if not found:
        return None
    minus_left, minus_right = found[0]
    return wrapper, minus_left, minus_right


def _classify_linear_species_rule(math, species_idx):
    """Try to express ``math`` as a linear combination of species.

    Returns a list of ``(species_idx_0based, factor)`` pairs when ``math``
    matches the form ``c_1 * s_1 [± c_2 * s_2 ...]`` where each ``c_i`` is
    a numeric literal (integer / real / rational, possibly absent) and each
    ``s_i`` is a species id. The factor for each species is allowed to be
    negative (subtraction folds into the sign).

    Returns ``None`` for any expression that is not a pure linear sum of
    species — products of species, divisions, parameter references,
    transcendentals, conditionals, and additive constants all reject.

    Used by the SBML loader to convert ``<assignmentRule variable="y">y =
    2*X</assignmentRule>`` into a BNGsim observable with weighted entries
    instead of a fixed species + function-defined synthetic parameter,
    which is the only thing the assignment-rule path can offer when the
    expression isn't linear. Observable values are recomputed at every
    SSA output sample, so the resulting trajectory tracks the rule's
    semantic intent without the round-tripping that the synthetic-param
    path would require.
    """

    def _walk(node, sign):
        t = node.getType()
        if t == libsbml.AST_PLUS:
            # Flatten the (left-leaning, possibly hundreds-deep) plus chain
            # before walking each operand, so a big observable-sum rule can't
            # overflow the stack (GH #111). Each operand carries the same sign.
            terms: list[tuple[int, float]] = []
            for operand in _flatten_assoc_chain(node, libsbml.AST_PLUS):
                child = _walk(operand, sign)
                if child is None:
                    return None
                terms.extend(child)
            return terms
        if t == libsbml.AST_MINUS:
            n = node.getNumChildren()
            if n == 1:
                return _walk(node.getChild(0), -sign)
            if n == 2:
                left = _walk(node.getChild(0), sign)
                right = _walk(node.getChild(1), -sign)
                if left is None or right is None:
                    return None
                return left + right
            return None
        if t == libsbml.AST_TIMES:
            coeff = float(sign)
            sp_idx: int | None = None
            for j in range(node.getNumChildren()):
                child = node.getChild(j)
                ct = child.getType()
                if ct == libsbml.AST_INTEGER:
                    coeff *= float(child.getInteger())
                elif ct in (libsbml.AST_REAL, libsbml.AST_REAL_E):
                    coeff *= child.getReal()
                elif ct == libsbml.AST_RATIONAL:
                    d = child.getDenominator()
                    if d == 0:
                        return None
                    coeff *= child.getNumerator() / d
                elif ct == libsbml.AST_NAME:
                    name = child.getName()
                    if name in species_idx and sp_idx is None:
                        sp_idx = species_idx[name]
                    else:
                        # parameter reference, or two species multiplied
                        return None
                else:
                    return None
            if sp_idx is None:
                return None
            return [(sp_idx, coeff)]
        if t == libsbml.AST_NAME:
            name = node.getName()
            if name in species_idx:
                return [(species_idx[name], float(sign))]
            return None
        # Bare numeric literals would be a constant offset — not expressible
        # as a BNGsim observable (which is a pure weighted sum of species).
        return None

    return _walk(math, 1)


def _classify_mass_action(
    rxn,
    kl_math,
    sbml_model,
    species_idx,
    species_hosu,
    species_comp,
    comp_volumes,
    rate_rule_targets,
    event_promoted_params,
    assignment_targets,
    moving_ar_comps,
    local_param_map,
    local_param_values,
    resolve_stoich,
    varvol_reject: dict | None = None,
):
    """Test whether a kinetic law is trivially mass-action (issue #16).

    Returns
    ``(rate_param_components, stat_factor, reactant_indices, product_indices)``
    when the kinetic law matches::

        [c *] [V_1 * V_2 * ...] k_1 * k_2 * ... * x_1^{m_1} * x_2^{m_2} * ...

    where ``c`` is a numeric literal (folded into stat_factor), ``V_i``
    are compartment-volume factors, ``k_j`` are SBML or kinetic-law
    parameters (a single parameter is the typical case; multiple are
    composed into a synthesized derived rate constant by the caller),
    and each species ``x_i`` has kinetic-law multiplicity ``a`` that
    combines with SBML reactant/product multiplicities ``b`` and ``c``
    such that the BNGsim product list (``c - b + a`` copies of each
    species) has all non-negative entries. The reactant list is the
    kinetic-law multiset, so ``rate = k * sf * Π y[reactant]`` and
    BNGsim's ydot accumulation reproduces SBML's ``ydot[s] = (c - b) *
    kinetic_law / V_s_factor`` exactly.

    ``rate_param_components`` is a list of ``(param_name, value)`` tuples
    in the order the symbols appeared in the AST (mostly cosmetic since
    multiplication commutes). Single-element list → caller passes the
    name straight to ``add_reaction(..., "elementary", name, ...)``.
    Multi-element list → caller synthesizes a derived parameter via
    ``add_parameter(..., is_expression=True)`` and uses its name.

    Volume / multi-compartment handling: the loader's per-species ``/V``
    step at the Functional path divides by the target species's
    ``V_s_factor`` (= compartment volume for hosu=False species, 1 for
    hosu=True species). For the Elementary substitution to be exact for
    every involved species, ``V_s_factor`` must be uniform across them.
    That's automatically true at V=1 across all involved compartments
    (BioModels' common shape, e.g. BIOMD0000000271's medium / cellsurface
    / cell) and lets multi-compartment reactions like
    ``kon * Epo * EpoR * cell`` lift to Elementary even though Epo,
    EpoR, and cell are different IDs.

    Returns ``None`` and falls back to the per-species Functional path on
    sums/differences/divisions (MM, Hill), SBML function-definition calls
    (``myrate(A, k)``), boundary-species reactants, non-integer
    stoichiometry, rate-constant references that resolve to assignment-
    rule functions (Elementary's codegen path emits ``p[k_idx]`` and
    only true parameters carry a param index), SBML reactants that don't
    appear in the kinetic law (rate doesn't depend on them — can't be
    Elementary), and reactions whose involved species don't share a
    common ``V_s_factor`` (per-species ``/V`` would diverge from a single
    Elementary rate).
    """
    multisets = _build_rxn_multisets(rxn, sbml_model, species_idx, resolve_stoich)
    if multisets is None:
        return None
    reactants_by_id, products_by_id = multisets
    return _classify_mass_action_ast(
        kl_math,
        reactants_by_id,
        products_by_id,
        sbml_model,
        species_idx,
        species_hosu,
        species_comp,
        comp_volumes,
        rate_rule_targets,
        event_promoted_params,
        assignment_targets,
        moving_ar_comps,
        local_param_map,
        local_param_values,
        varvol_reject=varvol_reject,
    )


def _build_rxn_multisets(rxn, sbml_model, species_idx, resolve_stoich):
    """Build (reactants_by_id, products_by_id) Counters from a reaction.

    Rejects boundary reactants and non-integer / non-positive stoichiometry
    (matching the original ``_classify_mass_action`` step 1-2 behavior).
    Returns ``None`` on rejection so callers can fall through to the
    Functional emission path. Boundary products are absorbed silently
    (they don't appear in ``products_by_id``).
    """
    reactants_by_id: Counter = Counter()
    for j in range(rxn.getNumReactants()):
        sr = rxn.getReactant(j)
        sid = sr.getSpecies()
        if sid not in species_idx:
            return None
        if sbml_model.getSpecies(sid).getBoundaryCondition():
            return None
        coeff = resolve_stoich(sr)
        if coeff is None or coeff <= 0 or coeff != int(coeff):
            return None
        reactants_by_id[sid] += int(coeff)

    products_by_id: Counter = Counter()
    for j in range(rxn.getNumProducts()):
        sr = rxn.getProduct(j)
        sid = sr.getSpecies()
        if sid not in species_idx:
            return None
        if sbml_model.getSpecies(sid).getBoundaryCondition():
            continue
        coeff = resolve_stoich(sr)
        if coeff is None or coeff <= 0 or coeff != int(coeff):
            return None
        products_by_id[sid] += int(coeff)
    return reactants_by_id, products_by_id


def _split_reversible_kinetic_law(
    rxn,
    kl_math,
    sbml_model,
    species_idx,
    species_hosu,
    species_comp,
    comp_volumes,
    rate_rule_targets,
    event_promoted_params,
    assignment_targets,
    moving_ar_comps,
    local_param_map,
    local_param_values,
    resolve_stoich,
):
    """Phase 7: split a reversible mass-action kineticLaw into two channels.

    Recognizes ``[wrapper *] (forward_rate - reverse_rate)`` shapes where
    each operand is independently mass-action. Returns
    ``(forward_classification, reverse_classification)`` (each in the same
    shape as ``_classify_mass_action`` returns) or ``None`` if the
    kineticLaw is not safely splittable. Callers fall through to the
    Phase 6 ``reversible_non_mass_action`` validation gate on ``None``.

    The reverse-side classifier runs with ``reactants_by_id`` and
    ``products_by_id`` swapped — the reverse rate consumes the original
    products and produces the original reactants. Per-species
    multiplicity reconciliation in ``_classify_mass_action_ast`` step 5
    enforces that each side's kinetic-law species multiset is consistent
    with its stoichiometric direction; misordered operands (reverse
    written first) auto-fail there.
    """
    factored = _factor_minus_subtree(kl_math)
    if factored is None:
        return None
    wrapper_leaves, fwd_math, rev_math = factored

    multisets = _build_rxn_multisets(rxn, sbml_model, species_idx, resolve_stoich)
    if multisets is None:
        return None
    reactants_by_id, products_by_id = multisets

    prefix = tuple(wrapper_leaves)
    fwd = _classify_mass_action_ast(
        fwd_math,
        reactants_by_id,
        products_by_id,
        sbml_model,
        species_idx,
        species_hosu,
        species_comp,
        comp_volumes,
        rate_rule_targets,
        event_promoted_params,
        assignment_targets,
        moving_ar_comps,
        local_param_map,
        local_param_values,
        prefix_factors=prefix,
    )
    if fwd is None:
        return None
    rev = _classify_mass_action_ast(
        rev_math,
        products_by_id,  # swapped
        reactants_by_id,  # swapped
        sbml_model,
        species_idx,
        species_hosu,
        species_comp,
        comp_volumes,
        rate_rule_targets,
        event_promoted_params,
        assignment_targets,
        moving_ar_comps,
        local_param_map,
        local_param_values,
        prefix_factors=prefix,
    )
    if rev is None:
        return None
    return fwd, rev


def _rule_reads_no_species(
    name: str,
    sbml_model,
    species_idx,
    assignment_targets,
    rate_rule_targets,
    event_promoted_params,
    _seen: set | None = None,
) -> bool:
    """True iff the assignment rule for *name* reads, through every rule it
    names, only constant parameters, compartments and the time: a coefficient
    whose value no species count moves. A species, a reaction rate, a rate-rule
    or event-assigned variable, or a name it cannot place makes it False.
    """
    if _seen is None:
        _seen = set()
    if name in _seen:
        return True
    _seen.add(name)
    rule = sbml_model.getAssignmentRuleByVariable(name)
    if rule is None or rule.getMath() is None:
        return False
    for n in _iter_ast_subtree(rule.getMath()):
        if n.getType() != libsbml.AST_NAME:
            continue
        ref = n.getName()
        if ref in species_idx or ref in rate_rule_targets or ref in event_promoted_params:
            return False
        if ref in assignment_targets:
            if not _rule_reads_no_species(
                ref,
                sbml_model,
                species_idx,
                assignment_targets,
                rate_rule_targets,
                event_promoted_params,
                _seen,
            ):
                return False
            continue
        if sbml_model.getParameter(ref) is None and sbml_model.getCompartment(ref) is None:
            return False
    return True


def _classify_mass_action_ast(
    kl_math,
    reactants_by_id: Counter,
    products_by_id: Counter,
    sbml_model,
    species_idx,
    species_hosu,
    species_comp,
    comp_volumes,
    rate_rule_targets,
    event_promoted_params,
    assignment_targets,
    moving_ar_comps,
    local_param_map,
    local_param_values,
    prefix_factors: tuple = (),
    varvol_reject: dict | None = None,
):
    """AST-only classifier core (issue #16 + Phase 7 reversible-split).

    Equivalent to ``_classify_mass_action`` minus its steps 1-2. Callers
    pass already-validated ``reactants_by_id`` / ``products_by_id``
    Counters. The Phase 7 reversible splitter calls this twice with the
    multisets swapped on the reverse side (consumed-side gets the
    products multiset, produced-side gets the reactants multiset).

    ``prefix_factors`` is an optional sequence of AST leaves to prepend
    to the flatten output before classification — used by the reversible
    splitter to distribute outer-product wrappers (e.g. the ``compartment``
    factor in COPASI-style ``compartment * (kf*A - kr*B)``) onto each
    side of the MINUS.

    ``moving_ar_comps`` are the assignment-rule compartments whose size moves
    during a run (``_moving_assignment_rule_vars``); the variable-volume guards
    below treat them as they treat a rate-rule or event-resized compartment.
    """
    # 3. Flatten the AST into a list of accepted leaves.
    factors: list = list(prefix_factors)
    if not _flatten_product_for_mass_action(kl_math, factors):
        return None

    # 4. Classify each leaf factor.
    numeric_const = 1.0
    rate_param_components: list[tuple[str, float]] = []
    compartment_factors: Counter = Counter()  # comp_id → multiplicity
    species_multiset: Counter = Counter()
    comp_ids = set(comp_volumes.keys())
    # A factor that reads no species but is not a constant parameter: an
    # assignment rule over time and parameters (``k2 := k1*f``). It cannot ride
    # an Elementary rate, but the law stays a monomial in the species.
    rule_coefficient = False

    for f in factors:
        ftype = f.getType()
        if ftype == libsbml.AST_INTEGER:
            numeric_const *= float(f.getInteger())
            continue
        if ftype in (libsbml.AST_REAL, libsbml.AST_REAL_E):
            numeric_const *= f.getReal()
            continue
        if ftype == libsbml.AST_RATIONAL:
            d = f.getDenominator()
            if d == 0:
                return None
            numeric_const *= f.getNumerator() / d
            continue
        if ftype != libsbml.AST_NAME:
            return None

        name = f.getName()
        if name in species_idx:
            if name in assignment_targets:
                # AR-driven species in the rate law must read their computed
                # assignment-rule value, not the frozen species concentration.
                # Folding such a species into the mass-action reactant factor
                # would make compute_rxn_rate multiply by the stale ``conc[]``
                # entry (assignment-rule targets are emitted ``fixed`` and never
                # integrate). Refuse mass-action classification so the reaction
                # takes the Functional path, where the name binds to the AR
                # observable (the linear-rule total, or the rule function).
                # Without this guard, e.g. a modifier ``k*A*B`` where ``B`` is an
                # AssignmentRule species reads B≡B(0) forever (BIOMD0000000104).
                return None
            species_multiset[name] += 1
            continue
        if name in comp_ids:
            if name in assignment_targets:
                # (#98) An ASSIGNMENT-RULE (variable-volume) compartment factor —
                # e.g. the leading ``tC`` of a COPASI-style ``tC * (kf·A·B − …)``
                # law where ``tC := mC + …``. Folding ``comp_volumes[tC]`` into the
                # scalar Elementary ``sf`` (step 6) bakes the load-time STATIC
                # volume, but the compartment's live value V(t) diverges, so the
                # amount-rate is wrong by V_static / V_live(t) as it grows. A
                # scalar rate cannot carry the live symbol; refuse mass-action
                # classification so the reaction takes the Functional path, where
                # the live ``tC`` symbol stays inside the law and the #87
                # numeric-V_static storage divide integrates it correctly.
                return None
            compartment_factors[name] += 1
            continue

        # Otherwise the leaf must resolve to a parameter.
        if name in local_param_map:
            mangled = local_param_map[name]
            param_value = local_param_values.get(mangled, 0.0)
        else:
            sbml_p = sbml_model.getParameter(name)
            if sbml_p is None:
                return None
            if name in assignment_targets and _rule_reads_no_species(
                name,
                sbml_model,
                species_idx,
                assignment_targets,
                rate_rule_targets,
                event_promoted_params,
            ):
                rule_coefficient = True
                continue
            if (
                name in rate_rule_targets
                or name in event_promoted_params
                or name in assignment_targets
            ):
                # These resolve to a species or function in the BNGsim
                # model, not a parameter — Elementary's rate_law_param_indices
                # would be unresolved, breaking codegen.
                return None
            mangled = _safe_name(name)
            param_value = sbml_p.getValue() if sbml_p.isSetValue() else 0.0
        rate_param_components.append((mangled, param_value))

    if numeric_const <= 0:
        return None
    # An Elementary rate needs a constant parameter to carry it. Without one
    # (``0.3*A*c``, ``3*c``, ``k2*A*c`` with ``k2`` a rule) the reaction takes
    # the Functional path, which is right for the ODE; what is left to decide
    # below is only whether it is a monomial the SSA's variable-volume
    # correction (§9) covers, which reads the species factors alone.
    elementary = bool(rate_param_components) and not rule_coefficient

    # 5. Per-species multiplicity reconciliation. For each species s in the
    #    union of (kinetic-law / SBML-reactants / SBML-products) multisets,
    #    let a = kinetic-law multiplicity, b = SBML reactant multiplicity,
    #    c = SBML product multiplicity. The BNGsim Elementary path computes
    #    ydot[s] = (P_count(s) - R_count(s)) * rate where R_count = a (so
    #    rate has the right factor structure). Setting P_count = c - b + a
    #    reproduces SBML's ydot[s] = (c - b) * rate exactly. Reject if any
    #    target P_count is negative (e.g., ``2 A -> B; k*A`` — stoich 2,
    #    rate first-order — can't be expressed as a single Elementary
    #    reaction) or if a species is consumed by SBML but absent from
    #    the rate (``S -> P; k`` — constant-rate, not mass-action).
    target_products: dict[str, int] = {}
    for sid in set(species_multiset) | set(reactants_by_id) | set(products_by_id):
        a = species_multiset.get(sid, 0)
        b = reactants_by_id.get(sid, 0)
        c = products_by_id.get(sid, 0)
        if a == 0 and b > c:
            return None  # consumption without rate dependence
        p_count = c - b + a
        if p_count < 0:
            return None
        target_products[sid] = p_count

    # 6. Volume reconciliation. bngsim stores every species as ``amount/V_c``
    #    (concentration), so the per-species ODE accumulation is always
    #        ydot[s] = (c_s - b_s) * kinetic_law_value / V_c(s)
    #    where ``kinetic_law_value`` is what RR/SBML integrate. For the
    #    Elementary substitution to reproduce this for every involved species
    #    at one shared ``sf``, all V_c(s) must agree (``v_common``).
    #
    #    The kinetic law multiplies *amounts* for hOSU=true species and
    #    *concentrations* (= stored values) for hOSU=false species, while the
    #    Elementary species factor multiplies the *stored* concentration of
    #    every reactant. So each hOSU=true reactant in the law needs its amount
    #    restored: ``amount_i = stored_i * V_c(i)``. As of GH #75 this amount
    #    restoration is performed by the *engine* — each species carries an
    #    ``amount_valued`` flag (set from hasOnlySubstanceUnits below), and
    #    ``compute_rxn_rate``/``ssa_propensity`` read an amount-valued reactant
    #    as ``stored_i * V_c(i)`` in the species factor. The classifier no longer
    #    folds the former ``Π_{i hOSU} V_c(i)^mult_i`` numerator into ``sf``;
    #    only the compartment-id leaves still contribute ``kl_volume_product``
    #    and the shared ``/v_common`` storage divide remain. Net:
    #        sf = numeric_const * kl_volume_product / v_common
    #    Invariance: V_c=1 ⇒ unchanged; hOSU=false ⇒ amount_valued off ⇒
    #    unchanged. Only hOSU=true V≠1 species move, and ODE remains byte-
    #    identical because the same Π V_c reaches the accumulation via the
    #    species factor instead of via sf.
    rxn_species = set(target_products.keys())
    if not rxn_species:
        return None  # degenerate (no species touch this reaction)

    # (#144 case 4, #172) Cross-compartment variable-volume monomial. A clean
    # mass-action monomial spanning ≥2 STORAGE compartments, at least one of which
    # changes size at runtime, cannot lift to a single Elementary rate: the
    # per-species ``/V`` differs across compartments (so v_factors below would reject
    # it anyway when the load-time volumes differ), and a variable-volume
    # compartment's V_live(t) diverges from the baked V_static even when the volumes
    # happen to coincide. Route it to the Functional per-species emission, which
    # divides each species by its LIVE compartment volume (ODE) and applies a
    # per-compartment SSA propensity correction (§9).
    #
    # (#172) The law may carry an EXPLICIT variable-volume compartment factor
    # (``cell·k·A·B``, the p≠1 cross-compartment shape — analogue of #130's single-
    # compartment p≠1). It is admitted on the SAME footing as the bare law because,
    # for an all-hOSU=false monomial, the corrections are INDEPENDENT of the explicit
    # factor power f_c. With law = K·∏_c V_c^{f_c}·∏_s [s]^{a_s} and m_c the
    # hOSU=false reactant species-factor count in compartment c:
    #   • ODE: ``base_func`` carries V_c,live^{f_c}, and the per-species ÷V_live(s)
    #     reproduces d[s]/dt = stoich·law/V_live(s) exactly for any f_c (the explicit
    #     factor needs no special handling — it is just part of the rate).
    #   • SSA: the true rate has V_c,live^{f_c−m_c} while the base propensity (stored
    #     conc = count/V_static, explicit factor read as the LIVE symbol) carries
    #     V_c,live^{f_c}·V_c,static^{−m_c}; the ratio is ∏_c (V_c,static/V_c,live)^{m_c}
    #     — the f_c power cancels. So §9's species-only m_c derivation already folds
    #     the explicit factor (it lives in ``base_func``, not in m_c). Validated vs
    #     RR+COPASI (ODE) and Extrande (SSA).
    # Scoped to all-hOSU=false species; a mix with an amount that changes such a
    # concentration is certified below, where the load-time sizes agree (#787),
    # and other hOSU=true cross-compartment shapes stay refused
    # (varvol_non_mass_action). The §9 certification is guarded to IRREVERSIBLE
    # only. Record the variable-volume storage compartments so §9 can certify +
    # correct the emitted reaction.
    #
    # (#745) An assignment-rule compartment whose size moves during the run is
    # routed the same way; it is left out of what §9 certifies for the SSA,
    # which does not run such a compartment. One whose rule reads constants
    # alone (``moving_ar_comps`` leaves it out) holds its load-time size, as a
    # write to what the rule reads is refused, so the Elementary rate is right.
    _storage_comps = {species_comp[s] for s in rxn_species}
    _varvol_storage = _storage_comps & (set(rate_rule_targets) | set(event_promoted_params))
    _ar_storage = _storage_comps & set(moving_ar_comps)
    if (
        len(_storage_comps) > 1
        and (_varvol_storage or _ar_storage)
        and all(not species_hosu.get(s, False) for s in rxn_species)
    ):
        if varvol_reject is not None and _varvol_storage:
            varvol_reject["xcomp_varvol_comps"] = sorted(_varvol_storage)
        return None

    def _v_s_factor(s: str) -> float:
        # The /V_c storage conversion applies to every species regardless of
        # hOSU (amount = stored * V_c always); the amount-vs-concentration
        # distinction for hOSU species moves into the numerator below.
        return comp_volumes.get(species_comp[s], 1.0)

    v_factors = {_v_s_factor(s) for s in rxn_species}
    if len(v_factors) > 1:
        return None  # different ``/V`` per species; one Elementary rate
        #              cannot satisfy all of them simultaneously
    v_common = next(iter(v_factors))
    if v_common == 0:
        return None

    kl_volume_product = 1.0
    for cid, count in compartment_factors.items():
        kl_volume_product *= comp_volumes.get(cid, 1.0) ** count

    # (#130) The Elementary scalar ``sf`` bakes each compartment's LOAD-TIME
    # volume:  sf = numeric · Π_c V_static[c]^compartment_factors[c] / V_static[storage].
    # That is exact only while every involved volume is *constant*. If any
    # VARIABLE-volume compartment (rate-rule or event-resize) survives in ``sf``
    # with a non-zero net power, the baked V_static diverges from V_live(t) once
    # the volume moves: bngsim's ODE was off by V_static/V_live for the bare
    # ``k·A·B`` law (no compartment factor, net power −1) after a resize. Route
    # such a reaction to the Functional path instead, where the live compartment
    # symbol stays inside the law (``law / V_live``) and reproduces SBML's
    # d[s]/dt = law / V_live(t) exactly for any power. The BNG ``compartment·k·A·B``
    # convention (storage compartment to power 1, net power 0) cancels and
    # correctly stays Elementary. Scoped to all-hOSU=false single-storage-
    # compartment monomials, matching the validated SSA varvol subset (§9).
    #
    # (#131 finding 4) The hOSU=true case needs a *different* criterion. The
    # engine restores each amount-valued reactant as ``stored·V_c`` when it reads
    # the species factor, so the shared ``/v_common`` storage divide is undone and
    # does NOT cancel a compartment that appears as an explicit law factor. Any
    # VARIABLE-volume compartment surviving in ``kl_volume_product`` therefore
    # bakes its load-time V_static where V_live(t) is required — an hOSU=true
    # ``cell·k`` synthesis stays flat instead of stepping with V; a ``cell·k·H``
    # decay follows ``exp(-k·V_static·t)`` rather than ``exp(-k·∫cell dt)``. Route
    # such a reaction to the Functional path (§9), which now divides the live law
    # by the numeric V_static for hOSU=true varvol species (#131 finding 3) and so
    # reproduces SBML's dynamics exactly. A bare amount law (``k·H``, no
    # compartment factor) is volume-independent and stays Elementary.
    # (#144) When a reaction is a single-compartment MONOMIAL that the guards
    # below route to the Functional path purely for variable-volume reasons (a
    # non-cancelling compartment power, case 2; or an hOSU=true law factor over a
    # variable-volume compartment, case 1), the Functional ODE is exact but the
    # SSA propensity still needs a scalar live-volume correction. Record the
    # hOSU=false law-factor count `n_f` and the single storage compartment in
    # ``varvol_reject`` so §9 can tag the emitted Functional reaction (exponent
    # n_f - 1 for the live-symbol divide, 0 for the numeric-V_static divide). The
    # reaction is still rejected (returns None) — it MUST take the Functional path
    # for the ODE to be right; this only carries the SSA correction metadata.
    def _record_varvol_reject(comp: str) -> None:
        if varvol_reject is not None and comp in (
            set(rate_rule_targets) | set(event_promoted_params)
        ):
            varvol_reject["comp"] = comp
            varvol_reject["n_f"] = sum(
                m for sid, m in species_multiset.items() if not species_hosu.get(sid, False)
            )

    # (#745) An assignment-rule compartment whose size moves is a variable
    # volume here as well: left out, a bare ``k*A`` in ``C := 2 + time`` stayed
    # Elementary with V_static baked in and ran 36% off (BIOMD0000000342's
    # TGF_beta_ex, 37%). One over constants keeps its load-time size, and its
    # Elementary rate, which a reversible split needs to run under the SSA.
    # _record_varvol_reject keeps to rate-rule and event compartments, which
    # are the ones the SSA correction it feeds is for.
    varvol_comp_ids = (
        set(rate_rule_targets) | set(event_promoted_params) | set(moving_ar_comps)
    ) & comp_ids
    if varvol_comp_ids:
        if all(not species_hosu.get(s, False) for s in rxn_species):
            storage_comps = {species_comp[s] for s in rxn_species}
            if len(storage_comps) == 1:
                storage_comp = next(iter(storage_comps))
                for c in varvol_comp_ids & (set(compartment_factors) | {storage_comp}):
                    net_power = compartment_factors.get(c, 0) - (1 if c == storage_comp else 0)
                    if net_power != 0:
                        _record_varvol_reject(storage_comp)
                        return None
        elif any(c in varvol_comp_ids for c in compartment_factors):
            # hOSU=true (or mixed) with a variable-volume compartment as an
            # explicit law factor — the baked V_static is wrong once V moves.
            storage_comps = {species_comp[s] for s in rxn_species}
            if len(storage_comps) == 1:
                _record_varvol_reject(next(iter(storage_comps)))
            return None
        elif any(
            products_by_id.get(s, 0) != reactants_by_id.get(s, 0)
            and not species_hosu.get(s, False)
            and species_comp[s] in varvol_comp_ids
            for s in rxn_species
        ):
            # Some species amount-valued, and the reaction changes a concentration
            # in a variable-volume compartment, with no compartment factor to
            # cancel: its row is law/V_live, and the one Elementary rate divides
            # every row by the baked V_static. `H => B; k*H` with H an amount and
            # B a concentration in a growing compartment made B(6) 40.55 molecules
            # out of 40; RoadRunner 33.39.
            storage_comps = {species_comp[s] for s in rxn_species}
            if len(storage_comps) == 1:
                _record_varvol_reject(next(iter(storage_comps)))
            elif varvol_reject is not None:
                # Across compartments it takes the per-species emission, each row
                # over its own size, and the SSA correction §9 certifies for that
                # emission counts the concentration factors alone: an amount is
                # read exactly. So it serves this mix as it serves a monomial over
                # concentrations. #787's `A => B; k*A`, A an amount and B a
                # concentration in a compartment a rate rule or an event resizes,
                # ran under the SSA while its ODE was wrong (B over the load-time
                # size), and was refused once the ODE was right.
                _vv = storage_comps & (set(rate_rule_targets) | set(event_promoted_params))
                if _vv:
                    varvol_reject["xcomp_varvol_comps"] = sorted(_vv)
            return None

    if not elementary:
        storage_comps = {species_comp[s] for s in rxn_species}
        if len(storage_comps) == 1:
            _record_varvol_reject(next(iter(storage_comps)))
        return None

    # Amount restoration for hOSU=true reactants is no longer folded into ``sf``
    # here (GH #75). Each species carries an ``amount_valued`` flag (set by the
    # loader from hasOnlySubstanceUnits); the engine's Elementary species factor
    # reads such a species as its amount (stored × V_c) directly, so the former
    # ``hosu_numerator = Π_{i hOSU} V_c(i)^mult_i`` term is supplied uniformly in
    # ``compute_rxn_rate``/``ssa_propensity`` instead of being pre-baked three
    # different ways in the loader. ODE is byte-identical (the same Π V_c reaches
    # the accumulation); SSA is byte-identical for first-order hOSU and uses the
    # population falling factorial for higher order, as it does for an hOSU=false
    # species (issue #692).
    sf = numeric_const * kl_volume_product / v_common

    # (#170) ``sf`` is the ONE place a compartment volume stops being a symbol.
    # Record the net power each compartment carries into it so the caller can put
    # the symbol back — as a ratio factor on the reaction's rate parameter, which
    # is a live ``p[]`` read every consumer already honors. The power is the
    # explicit law-factor multiplicity minus one for the storage compartment
    # (whose ``/v_common`` divide is the storage convention, not a law factor);
    # the BNG ``compartment·k·A·B`` convention nets to zero and records nothing,
    # which is why that row of #170's table was the one already correct.
    #
    # ``v_common`` is chosen by VALUE — ``v_factors`` above is a set of volumes,
    # not of compartment ids — so a reaction whose species span two compartments
    # that merely happen to share a load-time size is admitted here with no way to
    # say WHICH parameter the divide belongs to. Under a live volume the two
    # diverge and no single scalar rate can serve both, so report those
    # compartments as unresolvable instead of guessing; the caller keeps the #164
    # refusal for exactly them. 978 of the corpus's 8,796 reactions are shaped
    # this way, which is why they are refused rather than re-routed to the
    # per-species path (that would move the answer on a fifth of the corpus at the
    # nominal point).
    #
    # Only STATIC compartment parameters are recorded. A rate-rule or
    # event-resized compartment is promoted to a species (not a parameter, so
    # unwritable by definition) and an assignment-rule compartment is recomputed
    # from its rule every step, so a write to it would not survive the next
    # evaluation.
    _promoted = set(rate_rule_targets) | set(event_promoted_params)

    def _writable(c: str) -> bool:
        return c not in _promoted and c not in assignment_targets

    v_storage_comps: set[str] = {species_comp[s] for s in rxn_species}
    v_law_comps: set[str] = {str(c) for c in compartment_factors}
    volume_powers: dict[str, int] = {}
    volume_unresolvable: set[str] = set()
    if len(v_storage_comps) == 1:
        v_storage_comp = next(iter(v_storage_comps))
        for cid in v_law_comps | {v_storage_comp}:
            n = int(compartment_factors.get(cid, 0)) - (1 if cid == v_storage_comp else 0)
            if n and _writable(cid):
                volume_powers[cid] = n
    else:
        volume_unresolvable = {c for c in v_storage_comps | v_law_comps if _writable(c)}

    # 7. Build reactant/product index lists with multiplicity.
    reactant_indices: list[int] = []
    for sid, mult in species_multiset.items():
        for _ in range(mult):
            reactant_indices.append(species_idx[sid])
    product_indices: list[int] = []
    for sid, mult in target_products.items():
        for _ in range(mult):
            product_indices.append(species_idx[sid])
    # ``target_products`` is keyed off a set union of species IDs (step 5), whose
    # iteration order is PYTHONHASHSEED-randomized, so the emitted product order —
    # and hence the codegen C text and its SHA-256 cache key — varied per process.
    # That defeated the compiled-.so cache entirely (every load recompiled) and
    # made builds non-reproducible. Sort for a stable order. Order-neutral for the
    # ODE/SSA accumulation (independent ``ydot[p] += rate`` per index; products do
    # not enter the rate expression), so this changes nothing numerically — only
    # the emission order — while making the cache key deterministic (GH #111
    # follow-up). Reactants already come from a deterministic Counter, so they are
    # left as-is to avoid perturbing the rate-expression factor order.
    product_indices.sort()

    # (#81) The 5th element exposes the law's compartment-factor multiset
    # (comp_id → power) so the SSA variable-volume tagger can read the power `p`
    # of the reaction's compartment as an explicit law factor. The net V-power of
    # the propensity is `p - n_h`; bngsim's variable-volume ODE is correct only
    # when the compartment cancels (`p == 1`, the BNG ``compartment*k*A*B``
    # convention), and the SSA live-volume correction exponent is then `n_h - 1`.
    # (#170) The 6th/7th elements carry the volume bookkeeping above.
    return (
        rate_param_components,
        float(sf),
        reactant_indices,
        product_indices,
        dict(compartment_factors),
        volume_powers,
        volume_unresolvable,
    )


def _capped_names(names: list[str], cap: int = 10) -> str:
    """``'a', 'b', ... (N more)``: a load warning that names a genome-scale
    model's symbols one by one would run to thousands of entries."""
    shown = ", ".join(repr(s) for s in names[:cap])
    if len(names) > cap:
        shown += f", ... ({len(names) - cap} more)"
    return shown


def _kinetic_law_local_params(kl) -> list:
    """A kinetic law's local parameters, on either SBML level.

    L3 lists them under ``getLocalParameter``. libsbml lists an L2 law's under
    ``getParameter`` only: its ``getNumLocalParameters()`` is 0, so a reader that
    asks just that sees no local parameter on 963/978 BioModels and resolves the
    law's names to any global of the same id instead (issue #580).
    """
    n = kl.getNumLocalParameters()
    if n > 0:
        return [kl.getLocalParameter(j) for j in range(n)]
    return [kl.getParameter(j) for j in range(kl.getNumParameters())]


def _law_math_in_model_scope(rxn):
    """A reaction's kinetic law as the reaction's id reads in a rule or an event:
    a copy with each local parameter renamed to ``_lp_<rid>_<id>``, the name §9
    registers it by, so that a global of the same id does not take its place.
    None when the reaction has no law."""
    kl = rxn.getKineticLaw()
    if kl is None or kl.getMath() is None:
        return None
    math = kl.getMath().deepCopy()
    for lp in _kinetic_law_local_params(kl):
        math.renameSIdRefs(lp.getId(), f"_lp_{rxn.getId()}_{lp.getId()}")
    return math


def _local_param_value(lp) -> float:
    """A local parameter's value. An unset one has none: NaN, so nothing passes
    it off as a number, or the 0.0 ``BNGSIM_ALLOW_UNSET_PARAMS=1`` asks for. A law
    that reads an unset one is refused without that (C2b, issue #580)."""
    if lp.isSetValue():
        return lp.getValue()
    import os

    return 0.0 if os.environ.get("BNGSIM_ALLOW_UNSET_PARAMS") == "1" else float("nan")


def _kinetic_law_local_values(kl) -> dict[str, float]:
    """id → value of a kinetic law's local parameters, unset ones included so
    that each shadows a global of the same id, as SBML scoping requires."""
    return {lp.getId(): _local_param_value(lp) for lp in _kinetic_law_local_params(kl)}


def _unsized_compartments_read(sbml_model, ar_targets) -> list[str]:
    """Compartments whose size SBML leaves undefined and something can read, in
    declaration order (issue #580).

    Undefined: Level 2 or 3, no ``size``, more than zero dimensions, and no
    initialAssignment, assignmentRule or algebraic rule that could define it.
    A comp model is checked after flattening, which resolves a placeholder
    compartment ``replacedBy`` a sized one to that size.
    Read: it holds any species (bngsim reports every species as amount over its
    compartment's size, whatever ``hasOnlySubstanceUnits`` says, so this can warn
    where the size happens to cancel, e.g. a constant species given as a
    concentration); a rate rule or event sets it (it becomes a state that starts
    at the size); or a rule, initialAssignment, event, stoichiometryMath or kinetic
    law names it (a law's own local of that id shadows it).
    """
    if sbml_model.getLevel() < 2:
        return []
    defined = set(ar_targets)
    for j in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(j)
        if ia.getMath() is not None:
            defined.add(ia.getSymbol())
    maths: list = []
    for j in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(j)
        if rule.getMath() is None:
            continue
        if rule.isAlgebraic():
            # An algebraic rule may be what determines the size. Which name it
            # determines is not recorded, so every name it mentions counts; that
            # can silence one it only reads, but bngsim refuses algebraic rules
            # unless BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1 drops them.
            defined |= _ast_name_set(rule.getMath())
        maths.append(rule.getMath())
    cands = []
    for j in range(sbml_model.getNumCompartments()):
        c = sbml_model.getCompartment(j)
        if c.isSetSize() or c.getSpatialDimensionsAsDouble() == 0 or c.getId() in defined:
            continue
        cands.append(c.getId())
    if not cands:
        return []
    targets = set(cands)
    read: set[str] = set()
    for j in range(sbml_model.getNumSpecies()):
        if sbml_model.getSpecies(j).getCompartment() in targets:
            read.add(sbml_model.getSpecies(j).getCompartment())
    for j in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(j)
        if rule.isRate() and rule.getVariable() in targets:
            read.add(rule.getVariable())
    for j in range(sbml_model.getNumInitialAssignments()):
        maths.append(sbml_model.getInitialAssignment(j).getMath())
    for j in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(j)
        for part in (ev.getTrigger(), ev.getDelay(), ev.getPriority()):
            if part is not None:
                maths.append(part.getMath())
        for k in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(k)
            maths.append(ea.getMath())
            if ea.getVariable() in targets:
                read.add(ea.getVariable())
    for sr in _iter_species_references(sbml_model):
        if hasattr(sr, "isSetStoichiometryMath") and sr.isSetStoichiometryMath():
            maths.append(sr.getStoichiometryMath().getMath())
    for j in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(j).getKineticLaw()
        if kl is not None and kl.getMath() is not None:
            shadow = {lp.getId() for lp in _kinetic_law_local_params(kl)}
            _walk_name_refs(kl.getMath(), targets - shadow - read, read)
    for m in maths:
        if m is not None:
            _walk_name_refs(m, targets - read, read)
    return [cid for cid in cands if cid in read]


def _collect_local_params(kl, rid, builder):
    """Register a KineticLaw's local parameters, on either SBML level.

    Returns ``(local_param_map, local_param_values)`` — the first maps
    original_id → mangled_id for expression rewriting, the second maps
    mangled_id → numeric value (used by the mass-action classifier when
    a kinetic-law local parameter participates in a synthesized derived
    rate constant).
    """
    import os

    hatch = os.environ.get("BNGSIM_ALLOW_UNSET_PARAMS") == "1"
    local_param_map = {}
    local_param_values: dict[str, float] = {}
    for lp in _kinetic_law_local_params(kl):
        if not lp.isSetValue() and not hatch:
            # C2b refused it if the law (or a function body the law calls) names
            # it, so nothing can read it, and registering its NaN would put one
            # into every writer's output. An explicit value="NaN" is the model's
            # own and stays.
            continue
        orig_id = lp.getId()
        val = _local_param_value(lp)
        lpid = f"_lp_{rid}_{orig_id}"
        builder.add_parameter(lpid, val)
        local_param_map[orig_id] = lpid
        local_param_values[lpid] = val
    return local_param_map, local_param_values


def _check_sbml_errors(doc, source):
    """Check for fatal SBML parsing errors."""
    if doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL) > 0:
        msg = doc.getError(0).getMessage()
        raise RuntimeError(f"Fatal SBML error in {source}: {msg}")


def _walk_species_refs(node, species_id_set, out):
    """Collect species ids referenced by AST_NAME nodes anywhere in *node*."""
    _walk_name_refs(node, species_id_set, out)


def _walk_name_refs(node, target_id_set, out):
    """Collect ids in *target_id_set* referenced by AST_NAME nodes anywhere in *node*."""
    for n in _iter_ast_subtree(node):
        if n.getType() == libsbml.AST_NAME:
            name = n.getName()
            if name in target_id_set:
                out.add(name)


class _FastSbmlModel:
    """O(1) id-lookup wrapper around a libsbml ``Model`` (GH #164).

    libsbml resolves a *string* id via ``ListOf::get(const std::string&)``, a
    linear scan over the corresponding list. So ``Model.getSpecies(sid)`` /
    ``getParameter(sid)`` / ``getCompartment(sid)`` are each O(n), and calling
    them inside the per-reaction / per-parameter interpret loops below is O(n²).
    On genome-scale models (100k+ species/parameters) that turns the SBML load
    into a multi-minute hang — the ``isSetValue`` parameter sweep and the
    per-species reaction-emission boundary checks alone dominate everything else.

    This proxy builds id→object dicts once (O(n) total) and serves the
    string-keyed lookups from them in O(1). Integer-index lookups and every other
    ``Model`` method forward unchanged to the wrapped object, so the returned
    libsbml objects — and thus all downstream behavior — are byte-for-byte
    identical; only the lookup cost changes.
    """

    __slots__ = ("_m", "_species", "_params", "_comps")

    def __init__(self, model):
        self._m = model
        self._species = {
            (s := model.getSpecies(i)).getId(): s for i in range(model.getNumSpecies())
        }
        self._params = {
            (p := model.getParameter(i)).getId(): p for i in range(model.getNumParameters())
        }
        self._comps = {
            (c := model.getCompartment(i)).getId(): c for i in range(model.getNumCompartments())
        }

    def getSpecies(self, key):
        if isinstance(key, str):
            return self._species.get(key)
        return self._m.getSpecies(key)

    def getParameter(self, key):
        if isinstance(key, str):
            return self._params.get(key)
        return self._m.getParameter(key)

    def getCompartment(self, key):
        if isinstance(key, str):
            return self._comps.get(key)
        return self._m.getCompartment(key)

    def __getattr__(self, name):
        # Everything not overridden above (getNumReactions, getReaction,
        # getListOf*, getRule, getElementBySId, …) forwards to the real model.
        return getattr(self._m, name)


def _build_model_from_sbml_doc(doc):
    """Build a BNGsim Model from a parsed libsbml Document."""
    # In a context of its own: what the build sets for the emitter
    # (_RATEOF_CONSTANT_SYMBOLS) ends with it, however it ends.
    return contextvars.copy_context().run(_build_model_from_sbml_doc_in_context, doc)


def _build_model_from_sbml_doc_in_context(doc):
    from bngsim._bngsim_core import ModelBuilder
    from bngsim._model import Model
    from bngsim._ssa_validation import SsaIssue

    # Interpretation phase clock (doc → internal _core). Captured here at entry
    # and stamped on the model right before the step-11 Jacobian derivation, so it
    # covers steps 0–11-build (incl. builder.build()) and excludes jac + codegen,
    # which carry their own timers. libSBML parse is timed by the caller (T-instr).
    _interpret_t0 = time.perf_counter()

    sbml_model = doc.getModel()
    if sbml_model is None:
        raise RuntimeError("SBML document contains no model")
    # O(1) id→object lookups for getSpecies/getParameter/getCompartment by id;
    # without this the per-reaction/per-parameter interpret loops are O(n²) and
    # genome-scale models hang for minutes (GH #164).
    sbml_model = _FastSbmlModel(sbml_model)

    builder = ModelBuilder()

    # SSA-validation findings, populated as we walk the SBML constructs.
    # Stashed on the returned Model so Simulator(..., method="ssa") and
    # bngsim.validate_for_ssa(model) can inspect them. Empty list means
    # the loader saw no SSA-incompatible constructs.
    ssa_issues: list[SsaIssue] = []

    # SBML <listOfFunctionDefinitions> — collect early so that
    # _eval_ast_numeric can inline user-defined function calls in
    # initialAssignments and assignmentRules during section 0.
    func_defs = {}  # name → (param_names, body_AST | _RATEOF_FUNCDEF)
    for i in range(sbml_model.getNumFunctionDefinitions()):
        fd = sbml_model.getFunctionDefinition(i)
        fid = fd.getId()
        math = fd.getMath()
        if math is None:
            continue
        nargs = math.getNumChildren() - 1
        param_names = [math.getChild(j).getName() for j in range(nargs)]
        body = math.getChild(nargs)
        # COPASI rateOf idiom (GH #106): a funcDef standing in for the rateOf
        # csymbol — body is <notanumber/> (or NaN+0*a) and it carries a
        # "Derivative" annotation. Detect by the annotation or the conventional
        # id `rateOf`, and store a sentinel body so the inliner emits the
        # per-species accessor for each *call's* argument instead of inlining
        # the NaN. (MODEL2403070001 defines such a funcDef but never calls it —
        # the sentinel is then simply never consumed.)
        annotation = fd.getAnnotationString() or ""
        if fid == "rateOf" or "Derivative" in annotation:
            func_defs[fid] = (param_names, _RATEOF_FUNCDEF)
        else:
            func_defs[fid] = (param_names, body)

    # A call to a functionDefinition that reads rateOf is written out in place,
    # with the caller's arguments, so its rate is of what was passed (issue
    # #1124, item 8). First, so every check below sees what will run: a name the
    # body reads is then checked where it is put.
    rated_funcs = _rated_functions(func_defs)
    _expand_rated_function_calls(sbml_model, func_defs, rated_funcs)

    # ── 0a. Refuse unsupported ODE constructs (GH #113) ────────────────
    # Fail fast — before any further processing — on delay() / AlgebraicRule,
    # which bngsim would otherwise drop silently and integrate as a different
    # mathematical system. (fast="true" is handled at Simulator construction.)
    _check_unsupported_constructs(sbml_model, func_defs)

    # ── 0b. Diagnose a bare <ci>time</ci> with no declared `time` symbol ──
    # (GH #96) — replace the opaque core compile failure (ERR239) with a
    # targeted error pointing at the time csymbol. bngsim does not infer the
    # csymbol; lenient resolution was declined.
    _check_bare_time_symbol(sbml_model)

    # ── 0c. Refuse undefined symbols in IAs / rules (GH #119) ──────────
    # The IA / rule evaluator returns no value for a symbol that binds to no
    # declared component, so section 0 below would silently DROP the offending
    # initialAssignment / rule and keep the target's declared value. RoadRunner
    # and COPASI refuse such models; bngsim fails closed too (naming each
    # offender) rather than integrate a malformed model's incomplete setup.
    _check_undefined_symbols(sbml_model)

    # ── 0d. Refuse a rateOf the model has no value for ─────────────────
    # rateOf of an assignmentRule variable (issue #1046, rule 10224), of a
    # concentration in a rule-sized compartment (10225), of anything but a
    # symbol (10223), and through a call to a functionDefinition that could
    # not be written out above. Checked once, before the fold below or the
    # emitter would give any of them a wrong rate.
    _check_rateof_targets(sbml_model, func_defs, rated_funcs)
    # What no rule sets has rate 0 wherever it is read (see _rateof_call).
    _RATEOF_CONSTANT_SYMBOLS.set(_rateof_constant_symbols(sbml_model))

    # ── 0. Evaluate initialAssignments ────────────────────────────────
    # SBML initialAssignment elements compute ICs from expressions.
    # We evaluate them BEFORE adding species/parameters to the builder,
    # so we can pass the correct values directly.
    #
    # eval_ctx is built unconditionally (even when no IAs exist) so that
    # later sections — notably stoichiometry resolution in section 9 —
    # can constant-fold L2 <stoichiometryMath> expressions and look up
    # L3 speciesReference IDs that participate in initialAssignments.
    #
    # rateof_resolver lets an initialAssignment / assignment rule fold a
    # ``rateOf(<symbol>)`` csymbol to the initial dx/dt (GH #231 sub-cluster 1,
    # 01250/01251/01252/01254). Harmless when the model has no rateOf — the
    # resolver only fires on a rateOf node, so every other fold is byte-identical.
    rateof_resolver = _make_rateof_initial_resolver(sbml_model, func_defs)
    ia_values = {}  # symbol → computed value
    eval_ctx = {}
    # Gather all raw values for numeric context
    for j in range(sbml_model.getNumCompartments()):
        c = sbml_model.getCompartment(j)
        _sz = c.getSize() if c.isSetSize() else 1.0
        # (#353) A non-finite size cannot divide an initialAmount into a
        # concentration; the compartment loop further below substitutes 1.0 and
        # warns. Mirror that substitution here so an initialAssignment that reads
        # the size resolves against the same value instead of seeding a nan into
        # the IC graph. A compartment whose own initialAssignment supplies a
        # finite size still overrides this seed in the IA loop below.
        eval_ctx[c.getId()] = _sz if _math.isfinite(_sz) else 1.0
    for j in range(sbml_model.getNumParameters()):
        p = sbml_model.getParameter(j)
        if p.isSetValue():
            eval_ctx[p.getId()] = p.getValue()
    for j in range(sbml_model.getNumSpecies()):
        sp = sbml_model.getSpecies(j)
        sid = sp.getId()
        cid = sp.getCompartment()
        vol = eval_ctx.get(cid, 1.0)
        # A species with neither initialConcentration nor initialAmount is
        # left OUT of eval_ctx (its value comes from an initialAssignment /
        # assignmentRule resolved in the loop below). Seeding it 0.0 would let
        # a rule evaluate against the placeholder before the IA fires — e.g.
        # MODEL1112110004's ``GCM1 = (GE1/Gss)^GPRG`` would hit ``0^-2.79``
        # and raise before GE1's IA sets it.
        if sp.isSetInitialConcentration():
            conc = sp.getInitialConcentration()
            amt = conc * vol
        elif sp.isSetInitialAmount():
            amt = sp.getInitialAmount()
            conc = amt / vol if vol else 0.0
        else:
            continue
        # A hasOnlySubstanceUnits=true species's symbol denotes its *amount*
        # in every MathML expression (initialAssignment / assignmentRule /
        # kineticLaw); a hOSU=false species's symbol denotes concentration.
        # Seed eval_ctx with the matching quantity so initialAssignment and
        # t=0 assignment-rule evaluation reads what the literal MathML means.
        # Byte-identical for V=1 (amount==conc) and hOSU=false species; only
        # hOSU=true species in V≠1 compartments change. Example: BIOMD547
        # species_14 IA divides referenced hOSU species by their compartment
        # (``species_11/compartment_3``) expecting the amount; seeding conc
        # made every such term off by 1/V (tiny V → 1e10 blow-up).
        eval_ctx[sid] = amt if sp.getHasOnlySubstanceUnits() else conc

    # Seed named species-reference stoichiometries into the numeric context so an
    # initialAssignment / assignment rule that reads a stoich id resolves (GH
    # #237, testTag SpeciesReferenceInMath). The IA loop below overrides any sr
    # id that is itself an initialAssignment target. L2 stoichiometryMath is
    # constant-folded; otherwise the plain stoichiometry attribute is used.
    for j in range(sbml_model.getNumReactions()):
        _rxn = sbml_model.getReaction(j)
        for _getter, _n in (
            (_rxn.getReactant, _rxn.getNumReactants()),
            (_rxn.getProduct, _rxn.getNumProducts()),
        ):
            for k in range(_n):
                _sr = _getter(k)
                _srid = _sr.getId() if hasattr(_sr, "getId") else ""
                if not _srid or _srid in eval_ctx:
                    continue
                _v = None
                if hasattr(_sr, "getStoichiometryMath"):
                    _sm = _sr.getStoichiometryMath()
                    if _sm is not None and _sm.getMath() is not None:
                        _v = _eval_ast_numeric(_sm.getMath(), eval_ctx, func_defs)
                if _v is None:
                    _s = _sr.getStoichiometry()
                    _v = _s if _math.isfinite(_s) else None
                if _v is not None and _math.isfinite(_v):
                    eval_ctx[_srid] = _v

    # Seed each reaction id into the numeric context as its INITIAL kinetic-law
    # value (GH #239). In SBML L3 a reaction's id is a first-class symbol whose
    # value is that reaction's current rate (the kinetic-law extent, in
    # substance/time, NOT ÷V), analogous to a species id denoting its amount.
    # An initialAssignment that reads a reaction id — ``p1 = J0`` (01224/01300)
    # or ``p1 = addone(J0)`` (01233) — then folds via the AST_NAME leaf below.
    # Only the INITIAL rate is bound here (the live runtime reaction-rate buffer
    # is out of scope, #239); these cases are initialAssignment-only. Local
    # parameters shadow eval_ctx for the law evaluation, mirroring the rateOf
    # initial resolver. Guard ``not in eval_ctx`` — reaction ids are a disjoint
    # namespace, so this never clobbers a species/param/compartment.
    #
    # The seed reads the DECLARED values, so the loops below refold it as the
    # initialAssignments and rules resolve (`_refresh_reaction_ids`): `p1 = J0`
    # with J0's law `k*S` and `S = 10` over a declared 5 folded 5, not 10.
    _rid_laws: list[tuple[str, object, dict]] = []
    for j in range(sbml_model.getNumReactions()):
        _rxn = sbml_model.getReaction(j)
        _rid = _rxn.getId()
        if not _rid or _rid in eval_ctx:
            continue
        _kl = _rxn.getKineticLaw()
        if _kl is None or not _kl.isSetMath():
            continue
        _lps = _kinetic_law_local_values(_kl)
        _rid_laws.append((_rid, _kl.getMath(), _lps))
        _rv = _eval_ast_numeric(_kl.getMath(), {**eval_ctx, **_lps}, func_defs)
        if _rv is not None and _math.isfinite(_rv):
            eval_ctx[_rid] = _rv

    def _refresh_reaction_ids() -> bool:
        """Refold each reaction id at the current initial values; True if one moved.
        One that no longer folds to a finite rate leaves the context, so a fold
        that reads it fails (and is lifted or refused) rather than reading a
        stale number."""
        moved = False
        for _rid, _law, _lps in _rid_laws:
            _rv = _eval_ast_numeric(_law, {**eval_ctx, **_lps} if _lps else eval_ctx, func_defs)
            if _rv is not None and _math.isfinite(_rv):
                if _ia_value_changed(_rv, eval_ctx.get(_rid)):
                    eval_ctx[_rid] = _rv
                    moved = True
            elif eval_ctx.pop(_rid, None) is not None:
                moved = True
        return moved

    if sbml_model.getNumInitialAssignments() > 0:
        # Evaluate SBML assignment rules AND initialAssignments together
        # in a single iterative loop.
        #
        # Assignment rules (e.g. B := ModelValue_1) and initialAssignments
        # (e.g. ModelValue_1 = A, ModelValue_2 = B) can have circular
        # cross-dependencies: an IA may reference an AR target, and the
        # AR may reference another IA target.  Evaluating them in separate
        # passes fails because neither can fully resolve alone.
        #
        # GUARD: AR values are only added to eval_ctx for symbols that
        # do NOT already have a raw value.  This prevents ARs from
        # overwriting raw parameter values with runtime-dynamic
        # computed values that corrupt IA evaluation.
        #
        # This handles chains like: A(raw) → ModelValue_1(IA=A) →
        # B(AR:=ModelValue_1, B has no raw value) → ModelValue_2(IA=B).
        for _pass in range(10):
            changed = False
            # Try assignment rules — only for symbols NOT already in ctx
            for j in range(sbml_model.getNumRules()):
                rule = sbml_model.getRule(j)
                if not rule.isAssignment():
                    continue
                var = rule.getVariable()
                if var in eval_ctx:
                    continue  # Don't override raw values
                math = rule.getMath()
                if math is None:
                    continue
                val = _eval_ast_numeric(math, eval_ctx, func_defs, rateof_resolver=rateof_resolver)
                if val is not None:
                    eval_ctx[var] = val
                    changed = True
            # Try initialAssignments (these DO override raw values)
            for j in range(sbml_model.getNumInitialAssignments()):
                ia = sbml_model.getInitialAssignment(j)
                sym = ia.getSymbol()
                math = ia.getMath()
                if math is None:
                    continue
                val = _eval_ast_numeric(math, eval_ctx, func_defs, rateof_resolver=rateof_resolver)
                if val is not None:
                    old = eval_ctx.get(sym)
                    if _ia_value_changed(val, old):
                        eval_ctx[sym] = val
                        changed = True
            if _refresh_reaction_ids():
                changed = True
            if not changed:
                break

        # Collect computed values
        for j in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(j)
            sym = ia.getSymbol()
            if sym in eval_ctx:
                ia_values[sym] = eval_ctx[sym]

    # Apply assignment rules to override raw initial values for AR-targeted
    # species/parameters/compartments, regardless of whether any IAs exist.
    # Done AFTER the IA loop because the loop's GUARD prevents AR-over-raw
    # overrides during IA evaluation. This is required when a species's
    # raw initial value is inconsistent with its assignment rule (SBML L3
    # spec: AR holds at all times including t=0; the InitialValueReassigned
    # tag flags exactly this case — e.g. 00621 declares S3 initialAmount=3.75
    # while S3 := k1*S2 = 0.375).
    for j in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(j)
        if not rule.isAssignment():
            continue
        var = rule.getVariable()
        math = rule.getMath()
        if math is None:
            continue
        val = _eval_ast_numeric(math, eval_ctx, func_defs, rateof_resolver=rateof_resolver)
        if val is not None:
            eval_ctx[var] = val
            ia_values[var] = val

    # Re-propagate finalized assignment-rule values into any initialAssignment
    # that references them. The guarded IA loop above evaluated each IA against
    # the *raw* ``value=`` of its dependencies; the AR override just finalized
    # any AR-target value whose raw ``value=`` was stale. An IA like
    # ``ModelValue_19 := cin0`` — where ``cin0`` is an assignment-rule target
    # whose raw ``value=`` is stale (COPASI exports a quantity twice: once as a
    # raw-valued parameter, once as an AR target) — therefore still holds the
    # wrong number. Re-run the IAs (and the AR override) to convergence so each
    # IA target tracks its AR-target dependency. Per SBML an assignment rule
    # holds at all times including t=0, so the AR value is the correct IA input.
    # Pure no-op unless an IA value actually moves: for a model with no such
    # cross-dependency the first pass recomputes identical values and breaks
    # (MODEL1606100000 / issue #73: a stale ``ModelValue_19`` = +322026 instead
    # of ``cin0`` = -4807974 flips the boundary species ``Osmin`` to the wrong
    # sign — bngsim disagreed with both RoadRunner and COPASI until this fix).
    #
    # Run it with no initialAssignment too: the override above is one pass in
    # document order, so a rule that reads a later rule's target (``c2 := v``
    # before ``v := 2*p``) reads it unset or stale, and only this loop settles
    # it. A compartment sized that way loaded with volume factor 1 and divided
    # every reaction across compartments by 1 instead of its size (issue #994).
    #
    # A species whose symbol the seed above bound through a conversion by its
    # compartment's size (an amount declared for a concentration symbol, or a
    # concentration declared for an amount symbol) was converted with the
    # DECLARED size. Where an initialAssignment or a rule gives the size, the
    # size at the start is that one, so the loop converts again as the size
    # settles: `q = B`, with B declared as amount 0.8 in `c2 := 2*p` (declared
    # 1), folded 0.8 where the concentration at the start is 0.2. A species with
    # its own initialAssignment or rule takes that value instead.
    _set_by_math = {
        sbml_model.getInitialAssignment(j).getSymbol()
        for j in range(sbml_model.getNumInitialAssignments())
    } | {
        sbml_model.getRule(j).getVariable()
        for j in range(sbml_model.getNumRules())
        if sbml_model.getRule(j).isAssignment()
    }
    _size_converted: list[tuple[str, str, float, bool]] = []  # sid, cid, declared, divide
    for _j in range(sbml_model.getNumSpecies()):
        _sp = sbml_model.getSpecies(_j)
        _spid, _spc = _sp.getId(), _sp.getCompartment()
        if _spc not in _set_by_math or _spid in _set_by_math:
            continue
        if _sp.getHasOnlySubstanceUnits():
            if _sp.isSetInitialConcentration():
                _size_converted.append((_spid, _spc, _sp.getInitialConcentration(), False))
        elif _sp.isSetInitialAmount() and not _sp.isSetInitialConcentration():
            _size_converted.append((_spid, _spc, _sp.getInitialAmount(), True))

    def _reconvert_species() -> bool:
        """Convert each size-converted species with its compartment's current
        size, as the seed did with the declared one; True if one moved."""
        moved = False
        for _sid, _cid, _declared, _divide in _size_converted:
            _vol = eval_ctx.get(_cid, 1.0)
            if not _math.isfinite(_vol):
                _vol = 1.0  # what the seed and the compartment loop use (#353)
            _v = (_declared / _vol if _vol else 0.0) if _divide else _declared * _vol
            if _ia_value_changed(_v, eval_ctx.get(_sid)):
                eval_ctx[_sid] = _v
                moved = True
        return moved

    # Each pass settles every symbol whose inputs had settled when it began. A
    # chain needs another pass only where a symbol reads one the pass moves after
    # it. Within a pass the initialAssignments go first, then the rules, then the
    # converted species, then the reaction ids, so that reader is an
    # initialAssignment or rule that reads something the loop moves (directly,
    # or through rateOf, which reads the kinetic laws), or a reaction id whose
    # law reads another reaction's id; a converted species reads only sizes,
    # which move before it. One pass per such reader, plus one, reaches the
    # fixed point of any acyclic model, and a model that needs fewer breaks
    # early. Counting every symbol instead would let a circular definition
    # (illegal SBML, which never settles) cost a pass per rule or reaction,
    # each pass re-evaluating all of them.
    _moved = (
        _set_by_math | {_rid for _rid, _law, _lps in _rid_laws} | {s[0] for s in _size_converted}
    )
    _rid_names = {_rid for _rid, _law, _lps in _rid_laws}
    _rateof_fds = {name for name, (_p, _b) in func_defs.items() if _b is _RATEOF_FUNCDEF}
    _fold_maths = [
        sbml_model.getInitialAssignment(j).getMath()
        for j in range(sbml_model.getNumInitialAssignments())
    ] + [
        sbml_model.getRule(j).getMath()
        for j in range(sbml_model.getNumRules())
        if sbml_model.getRule(j).isAssignment()
    ]
    _readers = sum(
        1
        for _m in _fold_maths
        if _m is not None and (_ast_name_set(_m) & _moved or _ast_has_rateof(_m, _rateof_fds))
    ) + sum(1 for _rid, _law, _lps in _rid_laws if _ast_name_set(_law) & _rid_names)
    _settle_passes = max(10, _readers + 1)
    for _pass in range(_settle_passes):
        changed = False
        for j in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(j)
            sym = ia.getSymbol()
            math = ia.getMath()
            if math is None:
                continue
            val = _eval_ast_numeric(math, eval_ctx, func_defs, rateof_resolver=rateof_resolver)
            if val is not None and _ia_value_changed(val, ia_values.get(sym)):
                eval_ctx[sym] = val
                ia_values[sym] = val
                changed = True
        for j in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(j)
            if not rule.isAssignment():
                continue
            var = rule.getVariable()
            math = rule.getMath()
            if math is None:
                continue
            val = _eval_ast_numeric(math, eval_ctx, func_defs, rateof_resolver=rateof_resolver)
            if val is not None and _ia_value_changed(val, ia_values.get(var)):
                eval_ctx[var] = val
                ia_values[var] = val
                changed = True
        if _reconvert_species():
            changed = True
        if _refresh_reaction_ids():
            changed = True
        if not changed:
            break

    # ── C2: reject missing-value parameters that are actually consumed (internal#94) ──
    # SBML allows <parameter id="X"/> with no `value=` attribute. Defaulting it
    # to 0.0 is harmless when X is never consumed (extension-package
    # placeholders, doc-only params) but silently produces wrong dynamics when X
    # appears in a kineticLaw, rule, event, or initialAssignment — the silent 0.0
    # zeroes out whatever expression consumes it, so bngsim would return a
    # wrong-but-plausible trajectory instead of declining. RoadRunner refuses all
    # unset-value parameters at load. We scope precisely: walk the model's
    # expressions and act only on parameters whose value is unset, has no IA
    # fallback, AND is referenced somewhere — the cases where the silent 0.0
    # would actually flow into the simulation. The default is a hard ModelError
    # (matching RoadRunner's strictness); BNGSIM_ALLOW_UNSET_PARAMS=1 restores the
    # legacy lenient warn-and-default-to-0 behavior (e.g. for bngsim↔rr triage).
    # An unset-value parameter that nothing references stays silent either way.
    _all_param_ids = {
        sbml_model.getParameter(j).getId() for j in range(sbml_model.getNumParameters())
    }
    # An assignmentRule defines its target at all times (including t=0), so an
    # AR-target parameter is fully specified even with no `value=` and no IA —
    # its value comes from the rule, not a default. (RoadRunner agrees: it
    # refuses a *genuinely* valueless parameter but not an AR target.) This is
    # not subsumed by ``ia_values``: the IA-convergence loop above only captures
    # an AR target whose RHS is numerically evaluable at load time; a rule like
    # ``a := b`` / ``b := 0.5*S`` (S a species) cannot fold to a number yet but
    # is still well-defined. Exclude all AR targets so only parameters with no
    # value, no IA, and no defining rule remain candidates.
    _ar_targets = {
        sbml_model.getRule(j).getVariable()
        for j in range(sbml_model.getNumRules())
        if sbml_model.getRule(j).isAssignment() and sbml_model.getRule(j).getMath() is not None
    }
    # An initialAssignment gives its target a value as surely as a rule does,
    # whether or not it folds here. One that does not fold is evaluated by the
    # engine at the start, as it is for a parameter that declares a value the
    # assignment replaces (#732), or refused by name (#871). `k = 0.5 + S3`,
    # with S3 given no initial value, does not fold, as S3 is only started at
    # 0 (#580) further on; C2 refused k as having "no initialAssignment",
    # where libRoadRunner, and bngsim with any declared value for k, give
    # k = 0.5.
    _ia_targets = {
        sbml_model.getInitialAssignment(j).getSymbol()
        for j in range(sbml_model.getNumInitialAssignments())
        if sbml_model.getInitialAssignment(j).getMath() is not None
    }
    _unset_param_ids = {
        pid
        for pid in _all_param_ids
        if not sbml_model.getParameter(pid).isSetValue()
        and pid not in ia_values
        and pid not in _ia_targets
        and pid not in _ar_targets
    }
    if _unset_param_ids:
        _consumed: set[str] = set()

        def _scan(_math):
            if _math is not None:
                _walk_name_refs(_math, _unset_param_ids, _consumed)

        for j in range(sbml_model.getNumRules()):
            _scan(sbml_model.getRule(j).getMath())
        for j in range(sbml_model.getNumReactions()):
            kl = sbml_model.getReaction(j).getKineticLaw()
            if kl is not None and kl.getMath() is not None:
                # A law's own local parameter shadows a global of the same id, so
                # the law does not read that global (#580).
                _shadow = {lp.getId() for lp in _kinetic_law_local_params(kl)}
                _walk_name_refs(kl.getMath(), _unset_param_ids - _shadow, _consumed)
        for j in range(sbml_model.getNumEvents()):
            ev = sbml_model.getEvent(j)
            if ev.isSetTrigger():
                _scan(ev.getTrigger().getMath())
            if ev.isSetDelay():
                _scan(ev.getDelay().getMath())
            if ev.isSetPriority():
                _scan(ev.getPriority().getMath())
            for k in range(ev.getNumEventAssignments()):
                _scan(ev.getEventAssignment(k).getMath())
        for j in range(sbml_model.getNumInitialAssignments()):
            _scan(sbml_model.getInitialAssignment(j).getMath())

        _offenders = sorted(_unset_param_ids & _consumed)
        if _offenders:
            import os

            from bngsim._exceptions import UnderSpecifiedModelError

            if os.environ.get("BNGSIM_ALLOW_UNSET_PARAMS") == "1":
                for pid in _offenders:
                    logger.warning(
                        "Parameter '%s' has no value attribute and no "
                        "initialAssignment, but is referenced by a kineticLaw / "
                        "rule / event / initialAssignment expression in the "
                        "model. bngsim defaults the value to 0.0 "
                        "(BNGSIM_ALLOW_UNSET_PARAMS=1); this will silently zero "
                        "out any expression that consumes it. Set the parameter "
                        "value or add an initialAssignment.",
                        pid,
                    )
            else:
                names = ", ".join(repr(p) for p in _offenders)
                if len(_offenders) == 1:
                    lead = (
                        f"Parameter {names} has no value attribute and no "
                        "initialAssignment, but is referenced by a"
                    )
                else:
                    lead = (
                        f"Parameters {names} have no value attribute and no "
                        "initialAssignment, but are referenced by a"
                    )
                raise UnderSpecifiedModelError(
                    f"{lead} kineticLaw / rule / event / initialAssignment "
                    "expression in the model. The model is under-specified: "
                    "bngsim refuses to default the value to 0.0, which would "
                    "silently zero out any expression that consumes it and "
                    "return a wrong-but-plausible trajectory (RoadRunner refuses "
                    "the same model). Set the parameter value or add an "
                    "initialAssignment. To restore the legacy lenient "
                    "default-to-0 behavior, set BNGSIM_ALLOW_UNSET_PARAMS=1."
                )

    # ── C2b: a kineticLaw's valueless LOCAL parameter (issue #580) ──
    # C2 above walks only the model's global <parameter>s. A local one with no
    # `value=` used to load as 0.0 on L2 (the reaction silently switched off) and
    # as libsbml's NaN on L3 (a CVODE failure blaming negative states). A local
    # parameter is in scope only in its own law (and, through the translator, in
    # the bodies of functions the law calls) and nothing can assign it, so "unset
    # and named there" is the whole test. Like C2 it goes by what the
    # math names, not what it uses, so `piecewise(k, false, 1)` or a function
    # that ignores the argument still refuses; no corpus model has either. Same
    # refusal and hatch as C2. RoadRunner is no stricter here (it runs the L2
    # shape with the zero), but no model in the BioModels corpus or the SBML
    # semantic suite has a valueless local parameter its law reads, so refusing
    # costs nothing a reference engine runs.
    _unset_local: list[tuple[str, str]] = []
    for j in range(sbml_model.getNumReactions()):
        _rx = sbml_model.getReaction(j)
        _kl = _rx.getKineticLaw()
        if _kl is None or not _kl.isSetMath():
            continue
        _lps = _kinetic_law_local_params(_kl)
        _ids = [lp.getId() for lp in _lps]
        _dup = sorted({i for i in _ids if _ids.count(i) > 1})
        if _dup:
            # Invalid SBML (libsbml consistency error 10303), and there is no right
            # value to pick: the builder's duplicate-name check used to refuse it,
            # and a name -> value map would keep one of the two without a word.
            from bngsim._exceptions import ModelError

            raise ModelError(
                f"Reaction {_rx.getId()!r} declares local parameter "
                f"{', '.join(repr(i) for i in _dup)} more than once; a kineticLaw's "
                "local parameter ids must be unique."
            )
        _unset_ids = {lp.getId() for lp in _lps if not lp.isSetValue()}
        if not _unset_ids:
            continue
        _read: set[str] = set()
        # A name in a called function's body that is not a formal of it or of any
        # function on the call path resolves to the law's local (the translator
        # merges that scope in), so it is read as surely as one in the law itself.
        for _body, _scope, _ in _iter_funcdef_scan_frames(
            _kl.getMath(), func_defs, lambda *_: False
        ):
            _walk_name_refs(_body, _unset_ids - set(_scope or ()), _read)
        _unset_local.extend((_rx.getId(), lpid) for lpid in sorted(_read))
    if _unset_local:
        import os

        from bngsim._exceptions import UnderSpecifiedModelError

        names = ", ".join(f"{lpid!r} (reaction {rid!r})" for rid, lpid in _unset_local)
        if len(_unset_local) == 1:
            lead = f"Local parameter {names} has no value attribute, but its kineticLaw reads it"
        else:
            lead = (
                f"Local parameters {names} have no value attribute, but their "
                "kineticLaws read them"
            )
        if os.environ.get("BNGSIM_ALLOW_UNSET_PARAMS") == "1":
            one = len(_unset_local) == 1
            logger.warning(
                "%s. bngsim defaults %s to 0.0 (BNGSIM_ALLOW_UNSET_PARAMS=1), which "
                "silently zeroes out what %s. Set the %s.",
                lead,
                "it" if one else "them",
                "reads it" if one else "reads them",
                "value" if one else "values",
            )
        else:
            raise UnderSpecifiedModelError(
                f"{lead}. "
                "The model is under-specified: bngsim refuses to default the value "
                "to 0.0, which would silently zero out the rate law and return a "
                "wrong-but-plausible trajectory. Set the parameter value. To "
                "restore the legacy lenient default-to-0 behavior, set "
                "BNGSIM_ALLOW_UNSET_PARAMS=1."
            )

    # ── C2c: a conversionFactor that names a valueless parameter ──
    # A conversionFactor attribute, the model's or a species' own, reads a global
    # parameter as surely as a rate law does, but it is not math, so C2's walk
    # does not see it. SBML leaves such a parameter's value undefined, and the
    # factor scales every change a reaction makes to the species' amount. §3b
    # used to read it as 1, so the model ran as if it declared no factor:
    # `S1 -> S2` at 0.5*S1 with S1's factor `cf` left without a value gave
    # S1(1) = e^-0.5 = 0.607 and no warning. RoadRunner refuses the model. A
    # factor that an initialAssignment or an assignmentRule gives a value is not
    # in `_unset_param_ids` and loads as before (SBML suite 01642-01644 and
    # 01651-01653 give theirs by an initialAssignment). Nor is one whose
    # initialAssignment does not fold at load: the model does assign it, so it
    # is left to §3b, which names the assignment it could not evaluate. An
    # initialAssignment with no math assigns nothing, so its factor is refused
    # here; it used to be left to §3b too, which read it as 0.0 without a word
    # (S1 held at 1). Like C2 this goes by what the attribute names, so a
    # model-wide factor that every species overrides still refuses. Same
    # refusal and hatch as C2; under the hatch the factor is 0.0, the value the
    # parameter itself is given everywhere else.
    _cf_unset = _unset_param_ids
    _unset_cf: list[tuple[str, str]] = []
    if sbml_model.isSetConversionFactor() and sbml_model.getConversionFactor() in _cf_unset:
        _unset_cf.append((sbml_model.getConversionFactor(), "the model's conversionFactor"))
    for j in range(sbml_model.getNumSpecies()):
        _sp = sbml_model.getSpecies(j)
        if _sp.isSetConversionFactor() and _sp.getConversionFactor() in _cf_unset:
            _unset_cf.append(
                (_sp.getConversionFactor(), f"the conversionFactor of species {_sp.getId()!r}")
            )
    if _unset_cf:
        import os

        from bngsim._exceptions import UnderSpecifiedModelError

        one = len(_unset_cf) == 1
        names = ", ".join(f"{pid!r} ({where})" for pid, where in _unset_cf)
        lead = (f"Parameter {names} has" if one else f"Parameters {names} have") + (
            " no value attribute, initialAssignment or assignmentRule"
        )
        if os.environ.get("BNGSIM_ALLOW_UNSET_PARAMS") == "1":
            logger.warning(
                "%s, so SBML leaves the conversion %s undefined. bngsim takes %s as 0.0 "
                "(BNGSIM_ALLOW_UNSET_PARAMS=1), so no reaction changes the amount of a "
                "species %s converts. Set the %s.",
                lead,
                "factor" if one else "factors",
                "it" if one else "them",
                "it" if one else "they",
                "value" if one else "values",
            )
        else:
            raise UnderSpecifiedModelError(
                f"{lead}, so SBML leaves the conversion "
                f"{'factor' if one else 'factors'} undefined. The model is "
                "under-specified: bngsim refuses to read a factor as 1, which would "
                "silently drop it from every reaction that changes the species and "
                "return a wrong-but-plausible trajectory (RoadRunner refuses the same "
                "model). Set the parameter value or add an initialAssignment. To load "
                "with the factor at 0.0, the legacy lenient default for a parameter "
                "with no value, set BNGSIM_ALLOW_UNSET_PARAMS=1."
            )

    # Pre-scan rate-rule targets and event-assignment targets up front so
    # both step 1 (compartments) and step 2 (parameters) can skip IDs that
    # will be promoted to species later in step 8 / event handling. Without
    # this, an event that assigns to a compartment double-registers the
    # compartment id (once as a parameter here, once as a species at event
    # promotion time), which fails ExprTk's symbol-table register.
    rate_rule_targets = set()
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isRate():
            rate_rule_targets.add(rule.getVariable())

    _sp_ids_set = {sbml_model.getSpecies(ii).getId() for ii in range(sbml_model.getNumSpecies())}
    event_promoted_params = set()  # param/compartment IDs promoted to species
    for ii in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(ii)
        # An event whose <trigger> has no <math> child — or no <trigger> element
        # at all — can never transition false→true, so it never fires (SBML L3v2
        # §4.11.2; NoMathML / absent-trigger cases 01238/01239). §10 skips such an
        # event when building the C++ Event (see "has no trigger, skipping"), so
        # promoting its assignment targets to event-driven species would strand
        # them (the promotion never completes → column dropped, var_missing).
        # Leave the targets as plain parameters at their IC; a target assigned by
        # some *other* event with a real trigger is still promoted by that pass.
        _trig = ev.getTrigger()
        if _trig is None or _trig.getMath() is None:
            continue
        for jj in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(jj)
            var = ea.getVariable()
            # A <eventAssignment> with no <math> child is a no-op (SBML L3v2
            # §4.11.5 / NoMathML test cases 01600-01603): it never writes the
            # target, so the target keeps its value and stays a plain parameter
            # rather than being promoted to event-driven species state. Promoting
            # it would strand the symbol (the §10 emit loop skips no-math
            # assignments, so the promotion never happens), dropping the column
            # entirely — a non-constant parameter must still be reported (RR emits
            # it constant). A target assigned with math by some *other* event is
            # still promoted by that event's pass.
            if ea.getMath() is None:
                continue
            if var not in _sp_ids_set and var not in rate_rule_targets:
                event_promoted_params.add(var)

    # ── 1. Compartments → parameters (or species for rate rules) ─────
    comp_volumes = {}  # id → volume
    _comp_id_set = {
        sbml_model.getCompartment(j).getId() for j in range(sbml_model.getNumCompartments())
    }

    # (#74) Compartments whose size changes at runtime — by an event assignment
    # target or a rate rule. A reaction touching a variable-volume compartment
    # must divide its rate by the *live* compartment symbol (not the load-time
    # volume) to recover d[conc]/dt = law / V_live. The Functional emission's
    # common_vs==1.0 branch (section 9) otherwise reuses the raw law and lets a
    # post-resize compartment factor leak into the derivative (the BIOMD338
    # 3×-too-fast-after-dilution-event bug). For a V_c≠1 variable compartment
    # the divide already uses the live symbol, and mass action cancels the
    # compartment analytically, so only the common_vs==1.0 Functional path
    # needs this set.
    variable_comps = (event_promoted_params | rate_rule_targets) & _comp_id_set

    # (#87) Compartments whose size is set by an ASSIGNMENT RULE — e.g.
    # ``tV := mV + dV`` (BIOMD0000000856 budding-yeast cell cycle). Their live
    # value flows through the assignment-rule function bound to the same-named
    # parameter (``evaluate_functions`` writes ``mV+dV`` into the ``tV`` param
    # each RHS), so the symbol IS live. But for an amount-valued (hOSU=true)
    # species stored as ``amount/V_static``, the Functional storage-conversion
    # divide must use V_static (the load-time numeric volume the amount was
    # divided by), NOT the live symbol: dividing the amount-rate by V_live(t)
    # instead of V_static throttles every reaction by V_static/V_live(t) as the
    # compartment grows, which silently suppresses the cascade (#87: the cell
    # cycle never ignites — CLN/tV never reaches StartThr, no event fires). This
    # set drives the numeric-divide fix in section 9 and the report-time
    # concentration rescale (Simulator._apply_varvol_conc_map, extended to AR
    # compartments). Distinct from rate_rule_comps because an AR compartment has
    # no rate rule (no V̇), so the #86 dilution path does not apply, and its live
    # volume is sourced from the AR function rather than a promoted-species ODE.
    ar_comp_targets = {
        sbml_model.getRule(j).getVariable()
        for j in range(sbml_model.getNumRules())
        if sbml_model.getRule(j).isAssignment() and sbml_model.getRule(j).getMath() is not None
    } & _comp_id_set

    # (#745) The assignment-rule compartments whose size can move during a run:
    # their rule reads the clock, a species that changes, a rate-rule or event
    # target, a reaction's rate that moves, or another rule that does. One that
    # reads constants alone holds its load-time size for the whole run, as a
    # write to what it reads is refused. The same set §9 keeps from the SSA.
    # Which rules move is found once here, for §3b and §9 as well.
    _ar_rules, _ar_moving_vars = _moving_assignment_rule_vars(sbml_model)
    ar_comp_moving = ar_comp_targets & _ar_moving_vars

    # (#86) Compartments driven by a RATE RULE — continuously variable volume
    # V(t). A concentration-valued (hOSU=false) species in such a compartment
    # needs an explicit dilution term ``-[S]·V̇/V`` in its concentration ODE
    # (section 8b below): bngsim integrates the stored *concentration* while the
    # SBML/RR semantics conserve the *amount*, so a growing/shrinking V dilutes
    # /concentrates the species even with no reaction flux. Event-resized
    # compartments are excluded — they change V discontinuously at the event and
    # the #74 event injects a per-species ``V_old/V_new`` rescale there, so no
    # continuous dilution term applies between events.
    rate_rule_comps = rate_rule_targets & _comp_id_set
    event_resize_comps = event_promoted_params & _comp_id_set

    # (#114, #131) Variable-volume compartments in which an hOSU=true (amount-
    # valued) species's amount→storage divide must use the load-time numeric
    # V_static, NOT the live compartment symbol. Every variable-volume class —
    # AR (V(t) via an assignment rule, #87), rate-rule (continuous V(t), #114),
    # and event-resize (piecewise-constant V, #131) — stores such a species as
    # ``amount/V_static``, so dividing its amount-rate by V_live(t) throttles
    # the flux by V_static/V_live(t) and silently suppresses the dynamics. The
    # storage divide must therefore cancel V_static, not V_live, for ALL three.
    # (Event-resize was previously excluded — the #131 review found that wrong
    # for hOSU=true: the #74 per-event ``V_old/V_new`` rescale conserves the
    # *amount* across the resize but does not fix the live-symbol throttle on the
    # reaction rate; an hOSU=true ``cell·k`` synthesis stayed flat instead of
    # stepping with V.) hOSU=false species are excluded at each use site — they
    # divide by V_live and carry the #86 ``-[S]·V̇/V`` dilution / #74 event
    # rescale — so adding event-resize here changes only the hOSU=true path.
    vstatic_divide_comps = ar_comp_targets | rate_rule_comps | event_resize_comps

    # Compartments whose live symbol must divide Functional reaction laws even
    # when their load-time size is 1.0. ``variable_comps`` intentionally excludes
    # assignment-rule compartments for state-volume paths, but reaction
    # storage conversion still needs ``law / V_live`` for hOSU=false species in
    # AR compartments (suite 00946/00948: ``C := fakeC``, event changes fakeC).
    live_symbol_divide_comps = variable_comps | ar_comp_targets

    # (#81) Variable-volume compartments the SSA engine can follow: a rate-rule
    # compartment (continuous V(t), Tier 2) or an event-resized compartment
    # (piecewise-constant V, Tier 1). Both promote the compartment to a species
    # whose stored value IS the live volume V_live(t) — the rate rule integrates
    # it (is_rate_rule_ode), the event writes it. A mass-action reaction in such
    # a compartment runs under SSA with the live-volume propensity correction
    # (the reaction is tagged ssa_live_volume_* in §9). AR compartments (#87/#98)
    # are excluded — SSA support for them is out of scope for #81.
    #
    # The old blanket SSA rejections (``compartment_rate_rule`` /
    # ``compartment_event_resize``) are now per-reaction: a mass-action reaction
    # in a varvol compartment is tagged and runs (§9 Elementary branch); a
    # non-mass-action / hOSU=true / cross-compartment varvol reaction is refused
    # with ``varvol_non_mass_action`` (§9), because the scalar live-volume
    # correction is exact only for an all-hOSU=false single-compartment monomial.
    varvol_ssa_comps = rate_rule_comps | event_resize_comps
    # (rxn_idx0, comp_id, exp) tags applied after §10 promotes event compartments
    # to species (see _apply at the end of §10).
    ssa_varvol_fixups: list[tuple[int, str, float]] = []
    # (#144) SBML reaction index → hOSU=false law-factor count for single-
    # compartment MONOMIALS routed to the Functional path purely for variable-
    # volume reasons (case 1 hOSU=true law factor, case 2 non-cancelling power).
    # §9 emits the Functional reaction (correct ODE) and adds the matching SSA
    # live-volume correction instead of refusing it (varvol_non_mass_action).
    ssa_varvol_functional: dict[int, int] = {}
    # (#144 case 4) SBML reaction index → the sorted variable-volume STORAGE
    # compartments of a CROSS-COMPARTMENT mass-action monomial (≥2 compartments,
    # ≥1 variable-volume, bare law, all-hOSU=false) that the classifier routed to
    # the Functional path. §9's per-species emission then (a) divides each varvol
    # species by its LIVE compartment volume (ODE correctness) and (b) adds a
    # per-compartment SSA propensity correction, lifting the varvol_non_mass_action
    # gate. Cross-compartment is the only #144 case needing per-species/per-
    # compartment live-volume terms (the single scalar can't hold the product).
    ssa_varvol_xcompartment: dict[int, list[str]] = {}
    # (#144 case 4) deferred fixups applied after §10 promotes event compartments
    # to species. ODE: (species_idx0, comp_id) → set_species_ode_live_volume.
    # SSA: (rxn_idx0, comp_id, V_static, exp) → add_reaction_live_volume_term.
    ode_xcomp_species_fixups: list[tuple[int, str]] = []
    ssa_xcomp_term_fixups: list[tuple[int, str, float, float]] = []
    # (#170) Compartments whose size is still NOT writable after this loader has
    # put the volume back into every fold it can reach — the residue of the #164
    # refusal, kept per-compartment instead of blanket. §7 adds a compartment
    # whose storage divide it cannot attribute to one parameter (a mass-action
    # reaction whose species span two compartments that merely share a load-time
    # size); §3 adds one whose species carry an IC this loader cannot re-resolve.
    compartment_write_refused: set[str] = set()
    # (#170) compartment id → its 0-based parameter index (absent for a
    # compartment promoted to a species, which is state rather than a parameter).
    comp_param_idx: dict[str, int] = {}

    for i in range(sbml_model.getNumCompartments()):
        c = sbml_model.getCompartment(i)
        cid = c.getId()
        vol = c.getSize() if c.isSetSize() else 1.0
        # Override with initialAssignment value if available
        if cid in ia_values:
            vol = ia_values[cid]
        # (#353) A non-finite declared size — MODEL2002070001 declares
        # ``size="NaN"`` on both compartments — is unusable as the
        # amount↔concentration divisor: the stored value ``amount/V`` becomes nan
        # and poisons the whole state before a single step, so a well-specified
        # ``initialAmount=10`` amount-only species loads as nan (issue #353). The
        # size is not a legitimate divisor here — RoadRunner and AMICI both
        # integrate this model in *amount* units, with V absent from every rate
        # law — so substitute a unit volume: each amount-only species then loads
        # as its declared amount and the model integrates in amounts, matching
        # both oracles. Warn rather than refuse; refusing a model the reference
        # engines run would be a strict regression, and #170's writability guard
        # keeps a subsequent ``set_param`` on the size honest. A rate law that
        # actually reads the size (none does in the #353 model) would see 1.0.
        if not _math.isfinite(vol):
            logger.warning(
                "compartment '%s' declares a non-finite size (%s); substituting "
                "1.0 for the amount/concentration conversion so amount-only "
                "species load as their declared amounts rather than nan. A rate "
                "law that reads this compartment's size would see the substituted "
                "value.",
                cid,
                vol,
            )
            vol = 1.0
        comp_volumes[cid] = vol
        # Skip adding as a builder parameter if the compartment will be
        # promoted to a species — by a rate rule (step 8) or by an event
        # assignment target (step 10). Otherwise both adds collide on the
        # same ExprTk symbol name.
        if cid in rate_rule_targets or cid in event_promoted_params:
            continue
        # (#164) Mark it a compartment size, so a later ``set_param`` refuses a
        # value-changing write instead of moving this ``p[]`` entry while every
        # constant ``vol`` is about to be folded into stays at its load-time
        # value — ``volume_factor`` (§3), an amount-declared ``initial_conc``
        # (§3), the Elementary scalar rate's ``Π V^n / V_storage`` (§6), and
        # ``ssa_volume_factor`` (§9). Marked for EVERY static compartment, not
        # only the ones some fold actually reaches: which halves of a write land
        # varies per reaction inside one model (a mass-action law folds the
        # volume away entirely; a Functional law loaded at V≠1 divides by the
        # live symbol; the same law loaded at V=1 had that divide normalized
        # out), so "this particular write happens to be honored" is not a
        # property a caller can see or a loader can cheaply prove. Compartments
        # promoted to species above are not parameters and need no flag; their
        # live volume is genuinely a state variable. Issue #170 makes the volume
        # live and retires the flag.
        # (#170) Keep the parameter index: it is what binds each species' storage
        # convention and each reaction's SSA propensity volume back to the size,
        # so a write re-derives them instead of leaving them at ``vol``.
        comp_param_idx[cid] = builder.add_parameter(_safe_name(cid), vol, is_compartment_size=True)
        # An assignment rule recomputes the size every step, so a write to it does
        # not survive the next evaluation — refuse rather than appear to take.
        if cid in ar_comp_targets:
            compartment_write_refused.add(cid)

    # (#580) A compartment with no size and nothing that defines one: SBML Level
    # 2 and 3 leave its size undefined (Level 1 defaults it to 1.0). bngsim uses
    # 1.0, as RoadRunner does, and refusing would turn away the 8 BioModels that
    # have one, so warn rather than refuse, once per load, as for a species with
    # no initial value. Only where the 1.0 can reach a number: a species in it,
    # a rate rule or event that sets it, or math that names it. A size nothing
    # reads stays silent, as an unread valueless parameter does (C2).
    _unsized = _unsized_compartments_read(sbml_model, _ar_targets)
    if _unsized:
        logger.warning(
            "%d %s no size, initialAssignment or assignmentRule, so SBML leaves the "
            "size undefined; bngsim uses 1.0, as RoadRunner does: %s. Set the size "
            "if 1 is not what the model means.",
            len(_unsized),
            "compartment has" if len(_unsized) == 1 else "compartments have",
            _capped_names(_unsized),
        )

    # (#170) The compartments whose size is a parameter that ONLY a `set_param`
    # can move. Everything downstream — the storage convention bound to each
    # species, the SSA propensity volume, the Functional storage divide emitted
    # against the symbol — keys off THIS set, not off `comp_param_idx`.
    #
    # The exclusion is the #87/#114 trap, and BIOMD0000000856 is the case that
    # proves it: `tV` is an *assignment-rule* compartment, so it is a parameter
    # (not promoted to a species) whose value the rule recomputes every step.
    # Storage there is `amount/V_static` by construction — that is what
    # `vstatic_divide_comps` exists to say — so binding `volume_factor` to the
    # parameter makes the amount restoration read V(t) where V_static is required,
    # and the model's analytical Jacobian starts failing its own FD self-check.
    # Its size is refused above for the matching reason: a write would not survive
    # the rule's next evaluation.
    live_volume_param_comps = {c for c in comp_param_idx if c not in vstatic_divide_comps}

    # (#170) An ``<initialAssignment>`` is a load-time FOLD of the compartment
    # size. Section 0 evaluates it numerically against `eval_ctx` — which is
    # seeded from the sizes — and hands the number to the builder as a plain
    # constant, so a later `set_param` on the size cannot reach it. That is a
    # sixth fold of the family the #164 refusal named, and the one stage 2's
    # enumeration missed. The PBPK cluster BIOMD0000001027/1028/1029/1039 is
    # what it costs: each copies every compartment size into its own parameter
    # (``Compartment_18 = StomachLumen``) and weights observables by the copy, so
    # a write moved the size, left the copy at its load-time value, and returned
    # a trajectory 77% away from the rebuild it is contracted to reproduce —
    # silently, because nothing else in the model disagreed.
    #
    # `volume_dependent_ia` maps each folded symbol to the live sizes its value
    # depends on. Everything in it has to be either put back on the size (below
    # and in §3) or refused by name.
    #
    # The dependency is NOT just "this initialAssignment names a compartment".
    # `eval_ctx` is seeded from the sizes twice over: directly for a compartment,
    # and through the amount↔concentration conversion for every species, since
    # §0 binds a species symbol to the quantity its MathML means (the AMOUNT for
    # hasOnlySubstanceUnits, the CONCENTRATION otherwise) and computes whichever
    # one the declaration did not give. An initialAssignment over such a species
    # folds the size exactly as directly as one that names it, and an assignment
    # rule, a reaction id (§0 binds it to its kinetic law's initial value, GH
    # #239) or a stoichiometryMath in between carries it the rest of the way.
    _ia_math = {}
    for _j in range(sbml_model.getNumInitialAssignments()):
        _ia = sbml_model.getInitialAssignment(_j)
        if _ia.getMath() is not None:
            _ia_math[_ia.getSymbol()] = _ia.getMath()

    volume_taint: dict[str, set[str]] = {c: {c} for c in live_volume_param_comps}
    for _j in range(sbml_model.getNumSpecies()):
        _sp = sbml_model.getSpecies(_j)
        _cid = _sp.getCompartment()
        # A species with an initialAssignment takes its value from that IA, which
        # §0 evaluates *over* this seed — the taint then flows through the IA math.
        if _cid not in live_volume_param_comps or _sp.getId() in _ia_math:
            continue
        if (
            _sp.isSetInitialConcentration()
            if _sp.getHasOnlySubstanceUnits()
            else (_sp.isSetInitialAmount() and not _sp.isSetInitialConcentration())
        ):
            volume_taint[_sp.getId()] = {_cid}

    _taint_edges = list(_ia_math.items())
    for _j in range(sbml_model.getNumRules()):
        _r = sbml_model.getRule(_j)
        if _r.isAssignment() and _r.getMath() is not None:
            _taint_edges.append((_r.getVariable(), _r.getMath()))
    for _j in range(sbml_model.getNumReactions()):
        _rx = sbml_model.getReaction(_j)
        _kl = _rx.getKineticLaw()
        if _rx.getId() and _kl is not None and _kl.isSetMath():
            _taint_edges.append((_rx.getId(), _kl.getMath()))
        for _get, _n in (
            (_rx.getReactant, _rx.getNumReactants()),
            (_rx.getProduct, _rx.getNumProducts()),
        ):
            for _k in range(_n):
                _sr = _get(_k)
                _srid = _sr.getId() if hasattr(_sr, "getId") else ""
                _sm = _sr.getStoichiometryMath() if hasattr(_sr, "getStoichiometryMath") else None
                if _srid and _sm is not None and _sm.getMath() is not None:
                    _taint_edges.append((_srid, _sm.getMath()))
    for _ in range(len(_taint_edges) + 1):
        _grew = False
        for _sym, _m in _taint_edges:
            _deps: set[str] = set()
            for _n in _ast_name_set(_m):
                _deps |= volume_taint.get(_n, set())
            if _deps and not _deps <= volume_taint.get(_sym, set()):
                volume_taint[_sym] = volume_taint.get(_sym, set()) | _deps
                _grew = True
        if not _grew:
            break
    # A size is not a fold of ITSELF (that is just §1 reading `ia_values`), but a
    # size defined from ANOTHER size — ``<initialAssignment symbol="c1">c2``,
    # legal SBML and how a nested compartment gets written — is: §1 folds `c1` to
    # a number, so a write to `c2` moves `c2` and leaves `c1` behind where a
    # rebuild moves both. Nothing holds `c1`'s expression, so `c2` is refused.
    volume_dependent_ia: dict[str, set[str]] = {}
    for _s in _ia_math:
        _d = volume_taint.get(_s, set()) - {_s}
        if _d:
            volume_dependent_ia[_s] = _d
    for _s in live_volume_param_comps & set(volume_dependent_ia):
        compartment_write_refused.update(volume_dependent_ia.pop(_s))

    # ── 2. Parameters ─────────────────────────────────────────────────
    # (#170) A volume-dependent initialAssignment onto a PARAMETER is lifted back
    # onto the size: the target becomes a *derived* parameter, so issue #43's
    # chain rule re-evaluates it on the write exactly as it already does for
    # `_rateLaw_<rid>`. Nominal values do not move — the builder is still seeded
    # with §0's folded number and nothing re-evaluates a derived parameter until
    # something is written.
    #
    # (#313) The lift is not about volumes. §0 folds EVERY initialAssignment to a
    # number, so any parameter that reaches the model only by feeding one is a
    # dangling constant: `set_param` on it moves nothing, and — the loud half —
    # its forward sensitivity is identically zero at every species and time,
    # because the symbol is absent from the RHS the chain rule differentiates.
    # BIOMD0000000569 is the case that proves it: a chain
    # ``BSk0 = BSk1*BSc^p``, ``BSk1 = BSk2*BSc^p``, … puts the `BSk*` constants in
    # the rate laws and `BSc` nowhere else, so a 50% write to `BSc` left all four
    # derived constants where they were and `dx/dBSc` came back a confident zero
    # against AMICI's nonzero column. 113 of the 1324 vendored BioModels (8.5%,
    # 678 parameters) have at least one such parameter; they are the ordinary
    # COPASI spelling for a derived rate constant. So every parameter
    # initialAssignment that CAN be lifted is, and only the residue is refused.
    #
    # Liftable means every symbol the expression reads is itself a parameter of
    # the built model: the compartment sizes from §1 and the parameters this loop
    # declares. An expression reading a species amount, a reaction rate, a
    # stoichiometry id or `time` is not liftable — those are not parameters at
    # all — nor is one whose target a rate rule or an event promotes to a species.
    #
    # The lifted targets are held back and declared after the plain parameters,
    # topologically sorted, and a reference cycle (illegal SBML) simply drops out
    # of the lift and stays folded. SBML's own order is not a dependency order —
    # 569 declares `BSk0` before the `BSk1` it reads — and this sort predates
    # issue #568, when re-evaluation was a single declaration-order pass over the
    # parameter list and lifting 569 in place would have left `BSk0` reading a
    # stale `BSk1` for one write. The engine sorts for itself now
    # (`SharedModelData::derived_param_order`), so this no longer has to be
    # right for the values to come out right; it stays because the declared
    # order it produces is the one the `.net` round trip and every index-keyed
    # report read back.
    _param_pool = set(comp_param_idx)
    _param_decl_index: dict[str, int] = {}
    for i in range(sbml_model.getNumParameters()):
        _pid = sbml_model.getParameter(i).getId()
        if _pid in rate_rule_targets or _pid in event_promoted_params:
            continue
        _param_decl_index[_pid] = i
        _param_pool.add(_pid)
    # An assignmentRule target is not a symbol the lift can rest on, on EITHER
    # side of the assignment. The candidate filter below already declines one as
    # a lift *target* (§4's rule recomputes it, so the fold is not what a write
    # faces); the same fact disqualifies it as a lift *dependency*. Its slot is
    # function-backed — the engine rewrites it from the rule's own expression
    # before every derivative evaluation — so a lifted expression reading it
    # re-derives from whatever that slot holds at write time, which after a run
    # is the rule's value at the last integrated point, not the t=0 fold the
    # initialAssignment means. BIOMD0000000570 is the case: `ModelValue_60 =
    # O2c_bar` would go 5.68 → 7.87 on the next write after a run, including the
    # identity write `set_params(dict(zip(param_names, vec)))` round-trips
    # through. Refused instead, which leaves the fold where §0 put it.
    _param_pool -= _ar_targets

    # ...but an AR target whose rule is a constant expression over the pool is a
    # symbol the lift CAN rest on, provided it rests on the rule's BODY and not
    # on the target's slot (GH #385).
    #
    # The paragraph above is about the slot: it is function-backed, the engine
    # rewrites it from the rule before every derivative evaluation, so a lifted
    # expression *reading* it re-derives from the rule's value at the last
    # integrated point. Substituting the rule's own expression sidesteps that
    # entirely — the emitted text never mentions the target — and when the
    # substituted body reaches nothing but constant parameters, the two readings
    # coincide anyway: such a rule cannot move, so its value at the last
    # integrated point IS its t=0 fold. `BIOMD0000000570`'s `O2c_bar` is
    # excluded by exactly that test, its rule reading state.
    #
    # Without this, `BIOMD0000000587` classifies `ModelValue_1` and
    # `ModelValue_3` as PRIMARY — independent knobs — when the SBML says
    # `ModelValue_1 := Alpha := ModelValue_0/(24*3.344) := Theta/80.256` and
    # `ModelValue_3 := rho_b := ModelValue_2/ModelValue_18 := rho_f/f`. The
    # values are right either way; what is lost is every sensitivity term routed
    # through them, which on that model is most of `Theta`'s column and half of
    # `rho_f`'s — enough to flip two of the three signs.
    #
    # Resolved to a fixpoint so a rule may rest on another constant rule, with
    # the pass count bounding a reference cycle (illegal SBML, but a cycle here
    # would otherwise not terminate).
    _ar_math = {
        sbml_model.getRule(j).getVariable(): sbml_model.getRule(j).getMath()
        for j in range(sbml_model.getNumRules())
        if sbml_model.getRule(j).isAssignment() and sbml_model.getRule(j).getMath() is not None
    }
    _ar_const_names: dict[str, set[str]] = {}  # AR target → pool symbols it reaches
    for _ in range(len(_ar_math) + 1):
        _grew = False
        for _var, _m in _ar_math.items():
            if _var in _ar_const_names or _m is None or _ast_references_time(_m):
                continue
            _n = _ast_name_set(_m)
            # Every symbol is a pool parameter, or another rule already known
            # constant. A species, a compartment, or an unresolved rule stops it.
            if not _n <= (_param_pool | _ar_const_names.keys()):
                continue
            _closure: set[str] = set()
            for _s in _n:
                _closure |= {_s} & _param_pool
                _closure |= _ar_const_names.get(_s, set())
            _ar_const_names[_var] = _closure
            _grew = True
        if not _grew:
            break
    # Bottom-up, so a substitution string carries no assignment-rule name of its
    # own — the whole point being that the lifted text never mentions a
    # function-backed slot. `_inner` is taken against every AR target rather than
    # against the survivors, so a rule whose dependency fails to convert is never
    # emitted with that dependency left as a bare reference: it simply never
    # becomes ready, and the sweep below drops it. A rule is attempted at most
    # once — success or failure — and `_grew` counts only rules newly attempted,
    # so the pass loop stops as soon as nothing new becomes ready rather than
    # re-running a failed conversion on every remaining pass.
    _ar_const_expr: dict[str, str] = {}
    _ar_const_tried: set[str] = set()  # ready and attempted, whether or not it converted
    for _ in range(len(_ar_const_names) + 1):
        _grew = False
        for _var in list(_ar_const_names):
            if _var in _ar_const_tried:
                continue
            _inner = _ast_name_set(_ar_math[_var]) & _ar_targets
            if not _inner <= _ar_const_expr.keys():
                continue
            _ar_const_tried.add(_var)
            _grew = True
            try:
                _ar_const_expr[_var] = _ast_to_exprtk_with_funcdefs(
                    _ar_math[_var],
                    func_defs,
                    local_params={k: _ar_const_expr[k] for k in _inner} or None,
                )
            except Exception as e:  # noqa: BLE001 - the symbol simply stays unliftable
                logger.debug("assignmentRule for %s not usable as a lift body: %s", _var, e)
        if not _grew:
            break
    for _var in list(_ar_const_names):
        if _var not in _ar_const_expr:
            _ar_const_names.pop(_var)

    # Candidates: a parameter initialAssignment over parameters alone. An
    # assignment rule recomputes its target every step, so the IA fold is only a
    # t=0 value and the rule — not this expression — is what a write has to
    # survive; those are left to §4 and not refused over, since `ia_values` for
    # an AR target is overwritten by the rule anyway.
    # A parameter initialAssignment may also read a SPECIES, and it means that
    # species' initial value: `k_MET_expression = MET*k_MET_degradation +
    # MET*k_phospho_MET_basal` (BIOMD0000001102), whose target is a rate
    # constant. An initialAssignment is a t=0 evaluation, so a species whose own
    # IC is a declared constant contributes a constant here — substitute it and
    # the lift keeps the parameters the expression also reads, instead of
    # freezing all of them behind the fold.
    #
    # Restricted to a species with no initialAssignment and no assignment rule
    # of its own. One with an IA would need §3's `_ic_<sym>` parameter, declared
    # *after* these, so reading it here would be a forward reference; one with a
    # rule has its t=0 value defined by §4, not by its declaration.
    # What a species symbol is worth at t=0, as ExprTk over parameters.
    #
    # Section 0 binds the symbol to `amt if hasOnlySubstanceUnits else conc`, and
    # exactly one of those two is a CONVERSION by the compartment size — which is
    # what `volume_taint` marks. Substituting the converted *number* would bake a
    # writable size into the lifted expression as a literal: the fold #164
    # refuses a size over, reproduced one layer down and no longer visible to the
    # refusal. So emit the conversion instead, keeping the size symbolic, and a
    # later `set_param` on it still reaches this initial condition.
    #
    # A species tainted by anything OTHER than its own conversion is skipped —
    # the taint then came through a rule or another symbol, and this local
    # rewrite cannot stand for it.
    #
    # `rateOf(x)` is the exception in both sections: it needs the species SYMBOL,
    # not the symbol's value, so a substitution turns it into `rateOf(<number>)`
    # and §10.6's accessor emits 0. Expressions carrying one keep the old
    # parameters-only predicate.
    _ic_rateof_names = {name for name, (_p, _b) in func_defs.items() if _b is _RATEOF_FUNCDEF}

    _species_ic_expr: dict[str, str] = {}
    for _j in range(sbml_model.getNumSpecies()):
        _sp_ic = sbml_model.getSpecies(_j)
        _sid = _sp_ic.getId()
        if _sid in _ia_math or _sid in _ar_targets:
            continue
        _taint = volume_taint.get(_sid, set())
        if not _taint:
            _species_ic_expr[_sid] = _real_literal(float(eval_ctx.get(_sid, 0.0)))
            continue
        _cid_ic = _sp_ic.getCompartment()
        if _taint != {_cid_ic} or _cid_ic not in live_volume_param_comps:
            continue
        _vol_sym = _safe_name(_cid_ic)
        if _sp_ic.getHasOnlySubstanceUnits() and _sp_ic.isSetInitialConcentration():
            # The symbol means an amount; the declaration is a concentration.
            _species_ic_expr[_sid] = (
                f"({_real_literal(float(_sp_ic.getInitialConcentration()))}*{_vol_sym})"
            )
        elif not _sp_ic.getHasOnlySubstanceUnits() and _sp_ic.isSetInitialAmount():
            # The symbol means a concentration; the declaration is an amount.
            _species_ic_expr[_sid] = (
                f"({_real_literal(float(_sp_ic.getInitialAmount()))}/{_vol_sym})"
            )

    # Issue #732: an initialAssignment the section-0 fold could not evaluate was
    # dropped silently, and its target kept its declared value. `ia_values` does
    # not show it: the fold's context is seeded with every declared value, so a
    # failed IA still "collects" one. Asked directly instead.
    def _ia_folds(m) -> bool:
        return (
            _eval_ast_numeric(m, eval_ctx, func_defs, rateof_resolver=rateof_resolver) is not None
        )

    # What the translator refused for an IA the fold could not evaluate either
    # (a distrib draw, GH #97). Raised after both lowering passes below.
    _ia_refusals: dict[str, str] = {}

    _lift_expr: dict[str, str] = {}
    _lift_deps: dict[str, set[str]] = {}
    for _pid, _m in _ia_math.items():
        if _pid in comp_param_idx or _pid not in _param_decl_index or _pid in _ar_targets:
            continue
        _names = _ast_name_set(_m)
        if not _names or _ast_references_time(_m):
            # Folded at load: that value is the answer. Otherwise the engine
            # evaluates it, with IEEE semantics (1/0 -> inf, 0/0 -> nan, as SBML
            # suite 00950 expects), at the initial time (issue #732).
            if _ia_folds(_m):
                continue
            _m = _ast_at_time_zero(_m)
        # A constant assignment rule is substituted by its body (GH #385), so it
        # counts as whatever pool symbols that body reaches — never as itself.
        _ar_subs = {_n: _ar_const_expr[_n] for _n in _names & _ar_const_expr.keys()}
        _expanded = set()
        for _n in _names:
            _expanded |= _ar_const_names[_n] if _n in _ar_subs else {_n}
        if not _expanded <= (_param_pool | _species_ic_expr.keys()):
            continue
        _sp_subs = {_n: _species_ic_expr[_n] for _n in _expanded & _species_ic_expr.keys()}
        # rateOf is never lifted, whatever else the assignment reads: a derived
        # parameter is evaluated only when a parameter is written, and the
        # builder refuses one that reads a rate accessor (13c9a0e). The fold
        # above has its value at the start, which is what an initialAssignment
        # means (issue #1045: `p2 = rateOf(p1)` of a parameter no rule changes
        # is 0, and used to be refused); one the fold cannot evaluate is refused
        # by name below (#871), never kept at its declared value.
        if _ast_has_rateof(_m, _ic_rateof_names):
            continue
        try:
            _lift_expr[_pid] = _ast_to_exprtk_with_funcdefs(
                _m, func_defs, local_params={**_sp_subs, **_ar_subs} or None
            )
        except Exception as e:  # noqa: BLE001 - fall through to the refusal
            logger.debug("initialAssignment for %s not lifted: %s", _pid, e)
            if not _ia_folds(_m):
                _ia_refusals[_pid] = str(e)
            continue
        # Species are substituted, not referenced, so they are not lift deps; nor
        # are the assignment-rule targets the substitution just erased.
        _lift_deps[_pid] = _expanded - _species_ic_expr.keys()

    # Kahn, with SBML declaration order as the tie-break so the emitted parameter
    # list is deterministic. Whatever is still pending when nothing is ready sits
    # in a cycle and cannot be lifted.
    _lift_order: list[str] = []
    _pending = dict(_lift_deps)
    while _pending:
        _ready = sorted(
            (k for k, d in _pending.items() if not (d & _pending.keys())),
            key=lambda k: _param_decl_index[k],
        )
        if not _ready:
            for k in _pending:
                logger.debug("initialAssignment for %s not lifted: reference cycle", k)
                _lift_expr.pop(k, None)
            break
        _lift_order.extend(_ready)
        for k in _ready:
            del _pending[k]

    _declared_param_ids = set(comp_param_idx)
    lifted_ia_param_expr: dict[str, str] = {}
    _deferred_lift_val: dict[str, float] = {}
    for i in range(sbml_model.getNumParameters()):
        p = sbml_model.getParameter(i)
        pid = p.getId()
        if pid in rate_rule_targets:
            continue  # Will be promoted to species
        if pid in event_promoted_params:
            continue  # Will be promoted to species for events
        val = p.getValue() if p.isSetValue() else 0.0
        # Override with initialAssignment value if available
        if pid in ia_values:
            val = ia_values[pid]
        _vdeps = volume_dependent_ia.get(pid)
        if pid in _lift_expr:
            _deferred_lift_val[pid] = val
        else:
            if _vdeps and pid not in _ar_targets:
                compartment_write_refused.update(_vdeps)
            builder.add_parameter(_safe_name(pid), val)
        _declared_param_ids.add(pid)

    for pid in _lift_order:
        _expr = _lift_expr[pid]
        builder.add_parameter(
            _safe_name(pid), _deferred_lift_val[pid], expression=_expr, is_expression=True
        )
        lifted_ia_param_expr[pid] = _expr

    # The residue, a parameter initialAssignment §0 folded that the lift above
    # could not put back on its symbols, is named in a warning at §10.8 (#313),
    # from the same closure that refuses a write to what it read (issue #1048).

    # ── 3. Species ────────────────────────────────────────────────────
    # Collect assignment rule targets for step 4. linear_assignment_rules
    # collects the subset whose RHS is a linear combination of species
    # (e.g., ``y = 2*X``); these are emitted in step 4 as observables with
    # weighted entries so they track their species at every output sample,
    # rather than hanging off a synthetic parameter that no SSA fire ever
    # writes back to the species count vector. The classification is
    # deferred to a second pass below because it needs species_idx, which
    # this section is what builds.
    assignment_targets = set()
    _ar_math: dict[str, object] = {}
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        # A no-MathML assignmentRule is a no-op: it does not define its target
        # at runtime, so the target should keep its declared/initial-assigned
        # value and stay bound through the ordinary species/parameter/
        # speciesReference path (GH #243).
        if rule.isAssignment() and rule.getMath() is not None:
            assignment_targets.add(rule.getVariable())
            _ar_math[rule.getVariable()] = rule.getMath()

    # Detect species whose initialAssignment makes the IC a function of model
    # PARAMETERS: these get registered as species_ic_param_refs so CVODES
    # forward-sensitivity can seed ∂x_i(0)/∂p_k when p_k is requested via
    # sensitivity_params. Mirrors what the .net loader does for "begin species"
    # entries with a parameter-name IC.
    #
    # Two shapes qualify:
    #
    #   * a bare <ci> naming a parameter (``u = b``) — registered directly, and
    #     seeded with coefficient 1 by the legacy identity path;
    #   * any other expression over constant parameters (``u = b*v0``,
    #     ``2*init_X + offset``) — lowered to a synthetic *derived* parameter
    #     here, because `compute_ic_param_sens_seed` (issue #43) already
    #     differentiates a derived IC to its primaries with the sympy chain
    #     rule. Nothing new has to know how to differentiate; the expression
    #     just has to reach that code, and a bare-<ci> test is what stopped it.
    #
    # This was a silent zero, not a refusal: AMICI's `neuron` fixture sets
    # ``u(0) = b*v0`` with `b` a fitted parameter, and bngsim returned a `b`
    # column missing the whole ∂u(0)/∂b term — 1602 where AMICI says 7898, with
    # nothing on stderr. A trajectory finite difference does not catch it
    # either, because `set_param` does not re-resolve an initialAssignment, so
    # the oracle holds x(0) fixed in exactly the same way the seed does.
    #
    # The referenced symbols must be parameters nothing in the model can MOVE.
    # A wrong IC is far worse than a missing sensitivity seed — BIOMD0000000856
    # is the case that proves it: `WHISBF = 0.66*NSt`, whose synthetic derived
    # parameter evaluated to 0 against a symbol that is a species in the built
    # model, and the build-time IC resolution then wrote that 0 back over the
    # species' real initial condition, moving a plain trajectory. A rejected
    # model keeps the pre-existing behaviour of no seed.
    #
    # What makes a symbol unsafe here is being *promoted to a species*, and this
    # loader promotes exactly two kinds: a rate-rule target and an event
    # assignment target (§8/§10 — see the parameter loop above, which skips
    # those two and no others). The three subtractions below are therefore the
    # whole predicate. `NSt` is caught by them: it is an event assignment
    # target, and every `constant="false"` parameter in BIOMD0000000856 is
    # written by an event or an assignment rule.
    #
    # An additional `getConstant()` filter used to sit on top, and it was not
    # doing the work its comment claimed (#379). SBML's `constant="false"` is a
    # declaration that a symbol *may* vary, not that anything varies it, and
    # COPASI emits it routinely on parameters no rule and no event ever writes.
    # Such a parameter is a constant of the built model — §2 adds it with
    # `add_parameter` like any other — so excluding it bought no safety and cost
    # every IC expression that reads one. BIOMD0000000611 is the case: `Dilution`
    # is `constant="false"` and unwritten, it divides all 17 species
    # initialAssignments, and one symbol failing this predicate withheld the seed
    # from every one of them — 18 of 106 sensitivity columns identically zero,
    # including the largest in the tensor, silently.
    _param_ids = {sbml_model.getParameter(j).getId() for j in range(sbml_model.getNumParameters())}
    _const_param_ids = _param_ids - assignment_targets - rate_rule_targets - event_promoted_params
    _species_ids_set = {
        sbml_model.getSpecies(j).getId() for j in range(sbml_model.getNumSpecies())
    }
    # An SBML *species* is not the only thing that becomes an integrator state
    # here (issue #146). A parameter or compartment carrying a rate rule, or
    # written by an event assignment, is promoted to a species by §8/§10 — which
    # is how Antimony's pure-ODE spelling (``Virus' = ...`` emitted as
    # ``<parameter constant="false">``) arrives. Its initialAssignment defines an
    # initial condition exactly as a species' does, so it earns the same seed;
    # keying this scan on the SBML species list alone dropped ∂x(0)/∂p for every
    # one of them. AMICI's `nested_events` is the case that proves it: the whole
    # ``V_0`` column read 0 against AMICI's 1, and — worse than a missing seed —
    # ``set_param("V_0", …)`` did not move ``Virus(0)`` on a plain trajectory
    # either, because the IC link that re-resolves it (model.cpp set_param, #79)
    # is the same registration.
    _promoted_state_ids = (rate_rule_targets | event_promoted_params) - _species_ids_set
    _ic_seed_symbols = _species_ids_set | _promoted_state_ids
    # (#170) A live compartment size is a constant parameter of the built model
    # (§1) that only a `set_param` moves, so an IC expression may read one on the
    # same terms as any other constant: `S21 = 393.927*0.055*cell`
    # (MODEL1710030000) is a parameter expression once the size is a parameter.
    # It has to be, too — an IC this loader leaves folded is an IC the write
    # cannot move, and §2's comment says what that costs. The set is disjoint
    # from `_const_param_ids` (a compartment is not an SBML <parameter>) and
    # carries the same guarantee the strict predicate exists to enforce: a
    # compartment the model can move is an AR/rate-rule/event target, which
    # `vstatic_divide_comps` already excluded from `live_volume_param_comps`.
    _ic_expr_symbols = _const_param_ids | live_volume_param_comps

    # An assignmentRule target is normally off limits here — §4's rule owns the
    # slot and recomputes it every step, so it is not a symbol a parameter
    # expression may rest on. But a rule that reads NO state is a constant of
    # the built model, and an IC that reads one is still a parameter expression.
    # BIOMD0000000807 is the case: `n(0) = n0`, `n0 = N0/K`, `N0 = r_N/mu_N - 1`
    # — a two-level rule chain over five constants, which left `mu_N`, `r_N`,
    # `K`, `G0` and `A0` with no seed at all. Inlined rather than referenced,
    # because the target itself is function-backed in the built model; the
    # inlined text is over real parameters only.
    _ic_const_ar: dict[str, str] = {}

    def _resolve_const_ar(_name: str, _stack: frozenset) -> str | None:
        _m = _ar_math.get(_name)
        if _m is None or _ast_references_time(_m):
            return None
        _names = _ast_name_set(_m)
        if _names & _ic_seed_symbols:
            return None  # reads state, so it is not constant
        _subs: dict[str, str] = {}
        for _n in _names:
            if _n in _ar_math:
                if _n in _stack:
                    return None  # illegal SBML; refuse rather than recurse
                _inner = _resolve_const_ar(_n, _stack | {_n})
                if _inner is None:
                    return None
                _subs[_n] = f"({_inner})"
            elif _n not in _ic_expr_symbols:
                return None
        try:
            return _ast_to_exprtk_with_funcdefs(_m, func_defs, local_params=_subs or None)
        except Exception:  # noqa: BLE001 - pragma: no cover, defensive
            return None

    for _ar_name in _ar_math:
        _resolved = _resolve_const_ar(_ar_name, frozenset({_ar_name}))
        if _resolved is not None:
            _ic_const_ar[_ar_name] = _resolved

    ia_single_param_ref: dict[str, str] = {}
    ia_param_expr: dict[str, str] = {}  # species id → ExprTk expression over params
    # An IC expression may also read ANOTHER STATE, and it means that state's
    # initial value — `A = M*(1 - mu_a/alpha_A)` (BIOMD0000000838),
    # `A88 = A46 + ARPPtot` (BIOMD0000000643), eleven of BIOMD0000001102's
    # twelve. Rejecting those cost the whole seed: 24 of 1102's 27 columns read
    # zero at t=0 where both other engines report a number. The reference is
    # resolved below to whatever expresses that state's own IC, so the chain
    # rule composes through the existing derived-parameter DAG (#43) instead of
    # needing a second differentiator.
    _ia_state_targets: set[str] = set()
    _ic_cand: dict[str, object] = {}  # sym → math, for the compound candidates
    _ic_cand_deps: dict[str, set[str]] = {}  # sym → the states it reads
    _ic_cand_ar: dict[str, set[str]] = {}  # sym → the constant rule targets it reads
    for j in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(j)
        sym = ia.getSymbol()
        if sym not in _ic_seed_symbols:
            continue
        math = ia.getMath()
        if math is None:
            continue
        _ia_state_targets.add(sym)
        # libsbml: AST_NAME is the only node type for a bare <ci>. Reject
        # operators, numbers, function applications, and AST_NAME_TIME etc.
        if math.getNumChildren() == 0 and math.getType() == libsbml.AST_NAME:
            ref = math.getName()
            if ref in _ic_const_ar:
                # A rule owns the named slot, so the IC is not linked to it.
                # Its constant expansion is a parameter expression, though.
                ia_param_expr[sym] = _ic_const_ar[ref]
            elif ref in _param_ids:
                ia_single_param_ref[sym] = ref
            elif ref in live_volume_param_comps:
                ia_param_expr[sym] = _safe_name(ref)
            continue
        names = _ast_name_set(math)
        if not names or _ast_references_time(math):
            # As the parameter lift above (issue #732): a fold that answered
            # stands; one that did not is evaluated by the engine at t = 0.
            if _ia_folds(math):
                continue
            math = _ast_at_time_zero(math)
        if not names <= (_ic_expr_symbols | _ic_seed_symbols | _ic_const_ar.keys()):
            continue
        # As the parameter lift above: an initial value that reads rateOf keeps
        # the fold's value at the start, whatever else it reads (issue #1045).
        if _ast_has_rateof(math, _ic_rateof_names):
            continue
        _ic_cand[sym] = math
        _ic_cand_deps[sym] = names & _ic_seed_symbols
        _ic_cand_ar[sym] = names & _ic_const_ar.keys()

    # Lower the compound candidates in dependency order, so a state reference
    # resolves to a symbol that is already declared. Kahn with SBML declaration
    # order as the tie-break, matching §2's lift; a reference cycle (illegal
    # SBML) drops out and stays folded.
    _ic_pending = {k: set(v) for k, v in _ic_cand_deps.items()}
    _ic_decl_index = {s: i for i, s in enumerate(_ic_cand)}
    while _ic_pending:
        _ic_ready = sorted(
            (k for k, d in _ic_pending.items() if not (d & _ic_pending.keys())),
            key=lambda k: _ic_decl_index[k],
        )
        if not _ic_ready:
            for k in _ic_pending:
                logger.debug("initialAssignment for %s not lowered: IC reference cycle", k)
            break
        for sym in _ic_ready:
            del _ic_pending[sym]
            subs: dict[str, str] = {
                _ar: f"({_ic_const_ar[_ar]})" for _ar in _ic_cand_ar.get(sym, ())
            }
            for dep in _ic_cand_deps[sym]:
                if dep in ia_single_param_ref:
                    # That state's IC *is* a parameter; read it straight.
                    subs[dep] = _safe_name(ia_single_param_ref[dep])
                elif dep in ia_param_expr:
                    subs[dep] = _safe_name(f"_ic_{dep}")
                elif dep in _ia_state_targets:
                    # It HAS an initialAssignment that did not lower, so its own
                    # ∂x(0)/∂p is unavailable. Folding it to a number here would
                    # silently drop that term and answer a partial derivative as
                    # if it were the whole one — the failure this issue is about.
                    logger.debug(
                        "initialAssignment for %s not lowered: it reads %s, "
                        "whose own initial condition did not lower",
                        sym,
                        dep,
                    )
                    break
                elif dep in _species_ic_expr:
                    # No initialAssignment of its own, so its t=0 value is what
                    # its declaration says -- a constant, or its conversion by a
                    # still-symbolic compartment size (see `_species_ic_expr`).
                    subs[dep] = _species_ic_expr[dep]
                else:
                    logger.debug(
                        "initialAssignment for %s not lowered: it reads %s, "
                        "whose value depends on a writable compartment size",
                        sym,
                        dep,
                    )
                    break
            else:
                try:
                    ia_param_expr[sym] = _ast_to_exprtk_with_funcdefs(
                        _ic_cand[sym], func_defs, local_params=subs or None
                    )
                except Exception as e:  # noqa: BLE001 - refused below if nothing else set it
                    logger.debug(
                        "initialAssignment for %s not lowered for IC sensitivity: %s", sym, e
                    )
                    if not _ia_folds(_ic_cand[sym]):
                        _ia_refusals[sym] = str(e)

    # An initialAssignment neither the fold nor the engine can evaluate is
    # refused, as the same construct is in any other rule (GH #97), instead of
    # leaving its target at the declared value (issue #732).
    if _ia_refusals:
        from bngsim._exceptions import ModelError

        _sym, _why = next(iter(_ia_refusals.items()))
        raise ModelError(
            f"initialAssignment for '{_sym}' cannot be evaluated: {_why}. bngsim will not "
            "use the declared value in its place, since an initialAssignment overrides it."
        )

    # Issue #871: the two passes above put a failed fold back on the engine only
    # for a parameter or a species. Every other §0 fold read `eval_ctx`, where a
    # failed target still holds its declared value: an initialAssignment or
    # assignment rule over it (`Q = S + 1` answered 6 for `S = 1/0`), a reaction
    # id §0 bound to its kinetic law's initial value, a constant
    # stoichiometryMath. An assignment rule whose own fold fails leaves its
    # target's declared value there the same way, and so does the piecewise
    # fold now that it declines an undecidable condition. So take every failed
    # target out of the context and fold again: whatever no longer answers read
    # one, however indirectly (a rateOf reads through the kinetic laws). Then
    # refuse each such value that a load-time number stands for: an
    # initialAssignment target the engine does not evaluate, a compartment an
    # assignment rule sizes (the load converts its species by the folded size),
    # a constant stoichiometry. A target the engine evaluates reads its inputs
    # live, and each of those is evaluated too or refused here.
    import os

    def _escape_hatch_drops(m) -> bool:
        # GH #119's escape hatch drops an expression that reads an undefined
        # symbol and keeps the declared value, having warned about it. That is
        # the documented behaviour, not a fold failure to refuse.
        return os.environ.get(_ALLOW_UNDEFINED_SYMBOLS_ENV) == "1" and not all(
            _model_declares_symbol(sbml_model, _n) for _n in _ast_name_set(m)
        )

    # (kind, the eval_ctx key it defines or None, what a message calls it, math,
    # local parameters)
    _folds: list[tuple[str, str | None, str, object, dict | None]] = []
    for _sym, _m in _ia_math.items():
        _folds.append(("initialAssignment", _sym, _sym, _m, None))
    for _sym, _m in _ar_math.items():
        _folds.append(("assignment rule", _sym, _sym, _m, None))
    _rule_driven = rate_rule_targets | assignment_targets | event_promoted_params
    for _j in range(sbml_model.getNumReactions() if sbml_model.getLevel() < 3 else 0):
        _rx = sbml_model.getReaction(_j)
        for _get, _n in (
            (_rx.getReactant, _rx.getNumReactants()),
            (_rx.getProduct, _rx.getNumProducts()),
        ):
            for _k in range(_n):
                _sr = _get(_k)
                _sm = _sr.getStoichiometryMath()  # Level 2 only
                if _sm is None or _sm.getMath() is None:
                    continue
                _srid = _sr.getId()
                _smm = _sm.getMath()
                # A varying one stays symbolic (§6c) and is evaluated live.
                if (
                    _srid in _rule_driven
                    or _ast_references_time(_smm)
                    or _ast_name_set(_smm) & (_species_ids_set | _rule_driven)
                ):
                    continue
                _folds.append(
                    (
                        "stoichiometryMath",
                        _srid or None,
                        f"{_sr.getSpecies()}' in reaction '{_rx.getId()}",
                        _smm,
                        None,
                    )
                )

    _engine_evaluated = (
        lifted_ia_param_expr.keys()
        | ia_param_expr.keys()
        | {_s for _s, _p in ia_single_param_ref.items() if _p in _declared_param_ids}
    )
    _refusable = {
        _i
        for _i, (_kind, _key, _name, _m, _loc) in enumerate(_folds)
        if (_kind == "initialAssignment" and _key not in _engine_evaluated)
        or (_kind == "assignment rule" and _key in _comp_id_set)
        or _kind == "stoichiometryMath"
    }

    def _refold(i: int, ctx: dict):
        _kind, _key, _name, _m, _loc = _folds[i]
        return _eval_ast_numeric(
            _m,
            ctx if _loc is None else {**ctx, **_loc},
            func_defs,
            rateof_resolver=rateof_resolver,
        )

    _cause: dict[int, int] = {}  # fold -> the failed fold it traces to
    _via: dict[int, str] = {}  # fold -> the symbol it reads that carries the failure
    for _i, (_kind, _key, _name, _m, _loc) in enumerate(_folds):
        if not _refusable:
            break
        _v = _refold(_i, eval_ctx)
        # A stoichiometry is baked in as a number, so a non-finite one fails too;
        # _resolve_stoich would fall back to the declared attribute.
        if (_v is None or (_kind == "stoichiometryMath" and not _math.isfinite(_v))) and not (
            _escape_hatch_drops(_m)
        ):
            _cause[_i] = _i
    if _cause:
        # A reaction id is in eval_ctx only if its kinetic law folded, so it is
        # never a failure of its own; it only carries one.
        for _j in range(sbml_model.getNumReactions()):
            _rx = sbml_model.getReaction(_j)
            _kl = _rx.getKineticLaw()
            if _rx.getId() in eval_ctx and _kl is not None and _kl.isSetMath():
                _loc = _kinetic_law_local_values(_kl)
                _folds.append(("reaction", _rx.getId(), _rx.getId(), _kl.getMath(), _loc or None))
    _grew = bool(_cause)
    while _grew:
        _grew = False
        _gone = {_folds[_i][1] for _i in _cause} - {None}
        _ctx = {_k: _v for _k, _v in eval_ctx.items() if _k not in _gone}
        for _i in range(len(_folds)):
            if _i not in _cause and _refold(_i, _ctx) is None:
                _names = _ast_name_set(_folds[_i][3])
                _src = next((_j for _j in sorted(_cause) if _folds[_j][1] in _names), None)
                if _src is None:  # read through a rateOf
                    _src = min(_cause)
                else:
                    _via[_i] = _folds[_src][1]
                _cause[_i] = _cause[_src]
                _grew = True
        _grew = _grew and not (_refusable & _cause.keys())
    if _refusable & _cause.keys():
        from bngsim._exceptions import ModelError

        _i = min(_refusable & _cause.keys())
        _kind, _key, _name, _m, _loc = _folds[_i]

        _rkind, _rkey, _rname, _rm, _rloc = _folds[_cause[_i]]
        if _cause[_i] == _i:
            _why = "it cannot be evaluated at load"
        elif _via.get(_i) == _rkey:
            _why = f"it reads '{_rname}', whose {_rkind} cannot be evaluated at load"
        else:
            _through = f" through '{_via[_i]}'" if _i in _via else ""
            _why = (
                f"it depends{_through} on '{_rname}', whose {_rkind} cannot be evaluated at load"
            )
        raise ModelError(
            f"{_kind} for '{_name}' cannot be evaluated: {_why}, and bngsim does not "
            "evaluate it in the engine. bngsim will not use a declared value in its place."
        )

    # (#170) The residue: a section-0 fold of a live size that neither §2 nor the
    # loop above put back on the size. An initial condition still folded, a named
    # speciesReference stoichiometry — each is a constant a write cannot reach,
    # so the size it reads is refused BY NAME rather than half-honoured.
    # MODEL1710030000 is the case that needs it: `S21 = 393.927*0.055*cell`
    # lowers, but its three siblings also read `numConcFactor`, an SBML
    # `constant="false"` parameter the strict IC predicate rejects, so `cell`
    # stays refused until all four do.
    for _sym, _deps in volume_dependent_ia.items():
        if _sym in _declared_param_ids or _sym in _ar_targets:
            continue  # §2 already lifted it, refused over it, or left it to §4's rule
        if _sym in ia_param_expr:
            continue  # lowered to a derived `_ic_<sym>` above
        if _sym in ia_single_param_ref:
            # A bare <ci> onto a parameter, relinked to it by #79's IC path. The
            # parameter is volume-dependent too, so §2 already decided: it was
            # lifted (and this IC follows it) or its sizes are already refused.
            continue
        compartment_write_refused.update(_deps)

    # Lower each compound parameter-only IC to a derived parameter. Declared
    # here, after §2's plain parameters, so the declared order is already a
    # dependency order — which the engine no longer requires (issue #568 sorts
    # the derived parameters at build) but which keeps this section's output
    # stable against the `.net` round trip.
    ia_expr_param: dict[str, str] = {}  # species id → synthetic parameter name
    for sym, expr in ia_param_expr.items():
        pname = _safe_name(f"_ic_{sym}")
        builder.add_parameter(
            pname, float(ia_values.get(sym, 0.0)), expression=expr, is_expression=True
        )
        ia_expr_param[sym] = pname

    species_ids = []
    species_idx = {}  # id → 0-based index
    species_comp = {}  # id → compartment id
    species_hosu = {}  # id → hasOnlySubstanceUnits

    # (#234) Species that appear as a reactant or product in some reaction —
    # i.e. that can carry reaction flux. Used by the pure-dilution un-fix below to
    # admit only flux-free species (whose amount is conserved) into the dilution
    # treatment, so un-fixing never introduces flux that SBML/RR do not apply.
    reaction_species: set[str] = set()
    for _ri in range(sbml_model.getNumReactions()):
        _rxn = sbml_model.getReaction(_ri)
        for _sr in range(_rxn.getNumReactants()):
            reaction_species.add(_rxn.getReactant(_sr).getSpecies())
        for _sr in range(_rxn.getNumProducts()):
            reaction_species.add(_rxn.getProduct(_sr).getSpecies())

    # (#580) Species the model never gives a starting value; see the warning
    # after this loop.
    unset_ic_species: list[str] = []
    for i in range(sbml_model.getNumSpecies()):
        sp = sbml_model.getSpecies(i)
        sid = sp.getId()
        comp = sp.getCompartment()
        hosu = sp.getHasOnlySubstanceUnits()
        is_boundary = sp.getBoundaryCondition()
        is_const = sp.getConstant()

        # (#170) The declared *amount*, when the IC is one — the invariant a
        # compartment-size write has to hold fixed while the stored value
        # (`amount/V_c`) moves under it. NaN ⇒ declared as a concentration, which
        # is volume-independent and stays put.
        declared_amount = float("nan")
        # Determine initial value as concentration
        if sid in ia_values:
            # initialAssignment overrides everything. For a
            # hasOnlySubstanceUnits=true species the assignment math is in
            # *amount* units (RR sets the amount and reports amount/V), so
            # divide by the compartment volume to get bngsim's stored
            # concentration — symmetric with the initialAmount branch below.
            # hOSU=false species are assigned in concentration directly, so
            # V=1 / hOSU=false models stay byte-identical.
            if hosu:
                vol = comp_volumes.get(comp, 1.0)
                ic = ia_values[sid] / vol if vol != 0 else ia_values[sid]
                declared_amount = float(ia_values[sid])
            else:
                ic = ia_values[sid]
        elif sp.isSetInitialConcentration():
            ic = sp.getInitialConcentration()
        elif sp.isSetInitialAmount():
            amt = sp.getInitialAmount()
            vol = comp_volumes.get(comp, 1.0)
            ic = amt / vol if vol != 0 else amt
            declared_amount = float(amt)
        else:
            ic = 0.0
            # An assignment rule or a (not numerically folded) initialAssignment
            # gives the value later; only a species with neither starts at a 0.0
            # the model never wrote.
            if sid not in assignment_targets and sid not in _ia_math:
                unset_ic_species.append(sid)

        # Boundary species with rate rules should NOT be fixed —
        # the rate rule provides their dynamics via step 8.
        # Boundary species WITHOUT rate rules should be fixed
        # (constant boundary condition).
        # Assignment-rule species are always fixed (value from function).
        is_fixed = is_const or sid in assignment_targets
        if is_boundary and sid not in rate_rule_targets:
            is_fixed = True

        # (#86, extended #234) A hOSU=false species whose *amount* is conserved
        # but whose compartment volume V(t) varies is NOT truly fixed: SBML/RR keep
        # the amount and let the concentration dilute — ``[S](t) = A / V(t)``.
        # bngsim stores the concentration, so it must *integrate* the pure dilution
        # term ``-[S]·V̇/V`` (§8b for a rate-rule compartment, §8c for a time-varying
        # assignment-rule compartment) rather than hold a constant concentration.
        # Un-fix it so the dilution term — its sole derivative — takes effect.
        #
        # "Amount conserved with no own dynamics" = a boundary species (excluded
        # from every reaction's reactant/product list in §9, so no flux leaks in) OR
        # a constant=true species (its amount is immutable by definition), with no
        # rate/assignment rule of its own. The reaction-flux guard
        # (``sid not in reaction_species``) keeps the #86 caveat honest: a constant
        # species that *does* appear in a reaction is left fixed, so un-fixing never
        # admits flux RR does not apply. #234 extends the original #86 gate (boundary
        # ∧ rate-rule comp) to constant species and to assignment-rule compartments.
        if (
            not hosu
            and (comp in rate_rule_comps or comp in ar_comp_targets)
            and (is_boundary or is_const)
            and (is_boundary or sid not in reaction_species)
            and sid not in assignment_targets
            and sid not in rate_rule_targets
        ):
            is_fixed = False

        # volume_factor = compartment volume V_c. Storage is `amount/V_c`
        # uniformly (loader §3 always divides initialAmount by V), so each
        # ±1 amount fire under SSA is `±1/V_c` in storage units. Pass V_c
        # for every species regardless of hOSU; default 1.0 leaves V=1
        # SBML and the .net loader path unchanged.
        # amount_valued = hasOnlySubstanceUnits (GH #75). When set, the engine
        # reads this species as its amount (stored × V_c) in every reaction
        # species factor. This is the single core capability that replaces the
        # mass-action classifier's `hosu_numerator` product (removed above):
        # the Elementary path now restores the amount uniformly in the engine
        # instead of the loader folding Π V_c into the scalar rate. amount =
        # stored when V_c = 1.0, so V=1 hOSU and all hOSU=false species stay
        # byte-identical. (The Functional §9 and observable-weight paths still
        # carry their own restoration for now — GH #75 step 2 folds those in.)
        idx = builder.add_species(
            _safe_name(sid),
            ic,
            fixed=is_fixed,
            volume_factor=comp_volumes.get(comp, 1.0),
            amount_valued=hosu,
        )
        species_ids.append(sid)
        species_idx[sid] = idx
        species_comp[sid] = comp
        species_hosu[sid] = hosu
        # Issue #718: a rule sets this species' value; it is not a molecule
        # count, and SSA/PSA must not round it to one.
        if sid in assignment_targets or sid in rate_rule_targets:
            builder.set_species_continuous(idx)

        # (#170) Bind the storage convention to the compartment SIZE PARAMETER, so
        # a later write re-derives `volume_factor` (and an amount-declared IC)
        # rather than leaving both at the size this load happened to see — issue
        # #164's defect. Skipped for a compartment promoted to a species: its
        # volume is genuine integrator state, not a parameter, and the engine
        # already reads it live through `ode_live_volume_idx0`.
        if comp in live_volume_param_comps:
            builder.set_species_volume_param(idx, comp_param_idx[comp], declared_amount)

        # (GH #231 sub-cluster 3 / 01463) A hasOnlySubstanceUnits=true species's
        # symbol denotes its amount, so its rateOf csymbol is the amount-rate. The
        # engine stores amount/V_static (rescaling to live volume separately), so the
        # rateOf buffer holds d(amount)/dt / V_static and ×volume_factor recovers the
        # amount-rate — for a CONSTANT-volume compartment (01455/01457) AND a
        # VARIABLE-volume compartment promoted to a state (rate-rule / event-resized,
        # 01463), with no conc·V̇ correction. Flag the species so the engine (and
        # codegen) scale its rate_of__ accessor by volume_factor.
        #
        # An assignment-rule compartment too (issue #736). Its volume is not an
        # integrator state, but what decides the scaling is how the species is
        # stored, and it is stored as amount/V_static there as well: left out,
        # its rateOf read d(amount)/dt / V_static, a third of the answer in a
        # compartment of size 3, and a rate rule on it integrated that. Such a
        # compartment is not in live_volume_param_comps, so codegen scales by
        # the number V_static, which is what the storage divided by.
        if hosu:
            builder.set_species_rateof_amount(idx)

        # If the species's IC was set by an initialAssignment over parameters,
        # register the link so CVODES forward sensitivity seeds yS for them —
        # directly for a bare <ci>, or through the synthetic derived parameter
        # that carries a compound expression to issue #43's chain rule.
        _register_ic_param_ref(
            builder, idx, sid, ia_single_param_ref, ia_expr_param, _declared_param_ids
        )

    # (#580) SBML leaves such a species' initial value undefined, so the 0.0 above
    # is bngsim's choice, not the model's, and it must not pass for a value the
    # model wrote. Warn rather than refuse, as #353 does for a non-finite
    # compartment size: RoadRunner starts these species at 0 too, and 7 of the 8
    # BioModels that have one pass rr_parity with it (5 are genome-scale models
    # with thousands each, hence one capped message rather than one per species).
    # C2/C2b refuse a valueless parameter instead because a zero rate constant
    # silently switches a reaction off, and RoadRunner refuses the global case.
    if unset_ic_species:
        shown = _capped_names(unset_ic_species)
        logger.warning(
            "%d species %s no initialAmount, initialConcentration, initialAssignment "
            "or assignmentRule, so SBML leaves the initial value undefined; bngsim "
            "starts %s at 0.0, as RoadRunner does: %s. Set the initial value if 0 is "
            "not what the model means.",
            len(unset_ic_species),
            "has" if len(unset_ic_species) == 1 else "have",
            "it" if len(unset_ic_species) == 1 else "them",
            shown,
        )

    # ── 3b. Model / species conversionFactor (GH #232) ────────────────
    # SBML's conversionFactor scales how a species' AMOUNT changes per unit
    # reaction extent — d(amount_i)/dt = cf_i · Σ_r (stoich_{i,r} · rate_r) —
    # WITHOUT altering how a species appears in rate laws. The attribute must
    # reference a *constant* Parameter (SBML L3 §4.5 / §4.6), so cf is a
    # compile-time numeric: a species' own conversionFactor if set, else the
    # model's, else 1.
    #
    # We realize it by scaling the reaction RATE: when every species a reaction
    # changes shares one effective cf, multiplying that reaction's rate by cf
    # gives d(amount_i)/dt = Σ_r stoich·(cf·rate) = cf·Σ_r stoich·rate for every
    # involved species — exact for the ODE and for the SSA mean, but not the SSA
    # noise: each firing should move the species by cf·stoich, not fire cf times
    # as often in unit jumps, so SSA/PSA refuse a cf other than 1 (SsaIssue
    # ``conversion_factor``, in the reaction loop below). In section 7
    # this folds into the mass-action statistical factor ``sf``. Genuinely
    # mixed-cf-per-reaction and conversion-factored non-mass-action laws can't be
    # expressed as a single rate scale and are refused loudly there (never
    # silently mis-integrated). Models with no conversionFactor leave
    # ``_has_conversion_factor`` False and this is a complete no-op.
    #
    # A factor whose parameter changes during the run is therefore one the rates
    # cannot follow: they keep the value it has at load. SBML forbids it (rule
    # 20705), but bngsim used to load it without a word: `cf = 2` with the rate
    # rule `cf' = 1` gave S1(1) = 0.36788 for `S1 -> S2` at 0.5*S1, where
    # RoadRunner, which follows the parameter, gives 0.28651. What changes one:
    # a rate rule (a rate of 0 leaves it where it is), an event that assigns it
    # (one with a trigger, as the event pre-scan counts them), or an assignment
    # rule that moves (`_moving_assignment_rule_vars`: it reads the time, a
    # species that changes, a reaction's rate that does, or something a rate
    # rule or an event changes). A rule over constants, or over a species that
    # nothing changes, holds its value and is folded like an initialAssignment.
    _cf_ids = {
        _sp.getConversionFactor()
        for _sp in (sbml_model.getSpecies(_j) for _j in range(sbml_model.getNumSpecies()))
        if _sp.isSetConversionFactor()
    }
    if sbml_model.isSetConversionFactor():
        _cf_ids.add(sbml_model.getConversionFactor())
    _cf_changed_by: dict[str, list[str]] = {}
    if _cf_ids:
        for _j in range(sbml_model.getNumRules()):
            _r = sbml_model.getRule(_j)
            _v = _r.getVariable()
            if _v not in _cf_ids or _r.getMath() is None:
                continue
            _txt = libsbml.formulaToL3String(_r.getMath())
            if _r.isRate():
                _names, _clock = _math_names(_r.getMath())
                if (
                    not _names
                    and not _clock
                    and _eval_ast_numeric(_r.getMath(), {}, func_defs) == 0
                ):
                    continue
                _cf_changed_by.setdefault(_v, []).append(f"the rate rule d({_v})/dt = {_txt}")
            elif _r.isAssignment() and _v in _ar_moving_vars:
                _cf_changed_by.setdefault(_v, []).append(
                    f"the assignment rule {_v} := {_txt}, which reads the time or a value "
                    "that changes,"
                )
        for _j in range(sbml_model.getNumEvents()):
            _ev = sbml_model.getEvent(_j)
            if _ev.getTrigger() is None or _ev.getTrigger().getMath() is None:
                continue  # it never fires
            for _k in range(_ev.getNumEventAssignments()):
                _ea = _ev.getEventAssignment(_k)
                if _ea.getVariable() in _cf_ids and _ea.getMath() is not None:
                    _cf_changed_by.setdefault(_ea.getVariable(), []).append(
                        f"event {_ev.getId()!r}" if _ev.getId() else "an event"
                    )

    def _resolve_const_param_value(pid, what):
        # conversionFactor parameters are constant, and the factor is folded into
        # the rates as a number: the value its initialAssignment (or an
        # assignmentRule over constants) gives it at load, else the declared one.
        # A parameter that the run changes is refused first. Nothing falls back
        # to a number the model did not give. A factor read as 1 is a factor
        # dropped without a word, and a declared value that an assignment
        # replaces is not the factor either: `cf = 5` with the initialAssignment
        # `cf = 2 + S3`, S3 given no initial value, scaled the rates by 5 while
        # the parameter read 2 (RoadRunner: 2). C2c refused a factor the model
        # gives no value at all; under its hatch the factor is 0.0, as the
        # parameter is.
        from bngsim._exceptions import ModelError

        p = sbml_model.getParameter(pid)
        if p is None:
            # Invalid SBML (libsbml validation rules 20216 and 20617).
            raise ModelError(
                f"{what} is {pid!r}, which is not a parameter of the model. SBML "
                "requires a conversionFactor to name a constant parameter, and bngsim "
                "will not read it as 1."
            )
        if pid in _cf_changed_by:
            raise ModelError(
                f"{what} is {pid!r}, whose value {' and '.join(_cf_changed_by[pid])} "
                "changes during the run. bngsim folds a conversion factor into the "
                "rates as a number at load, so the run would keep the value it has at "
                "the start. SBML requires a conversionFactor to name a constant "
                "parameter (validation rule 20705)."
            )
        if pid in _ia_math or pid in _ar_targets:
            kind = "initialAssignment" if pid in _ia_math else "assignmentRule"
            # `ia_values` holds the declared value for an initialAssignment that
            # did not fold (issue #732), so whether it folds is asked directly.
            folded = _ia_folds(_ia_math[pid]) if pid in _ia_math else pid in ia_values
            if folded and pid in ia_values:
                return ia_values[pid]
            _m = _ia_math[pid] if pid in _ia_math else sbml_model.getRuleByVariable(pid).getMath()
            raise ModelError(
                f"{what} is {pid!r}, and bngsim could not evaluate the {kind} "
                f"{pid} = {libsbml.formulaToL3String(_m)} at load, so it has no "
                "number to scale the rates by: it reads something with no value "
                "there. A conversionFactor must name a constant parameter; give it a "
                "value, or an initialAssignment that evaluates at load."
            )
        if p.isSetValue():
            return p.getValue()
        return 0.0  # BNGSIM_ALLOW_UNSET_PARAMS=1: C2c refused it otherwise, and warned

    _model_cf_val = (
        _resolve_const_param_value(
            sbml_model.getConversionFactor(), "The model's conversionFactor"
        )
        if sbml_model.isSetConversionFactor()
        else None
    )
    species_cf: dict[str, float] = {}
    _any_species_cf = False
    for _sid in species_ids:
        _sp = sbml_model.getSpecies(_sid)
        if _sp is not None and _sp.isSetConversionFactor():
            species_cf[_sid] = _resolve_const_param_value(
                _sp.getConversionFactor(), f"The conversionFactor of species {_sid!r}"
            )
            _any_species_cf = True
        elif _model_cf_val is not None:
            species_cf[_sid] = _model_cf_val
        else:
            species_cf[_sid] = 1.0
    _has_conversion_factor = _model_cf_val is not None or _any_species_cf

    def _reaction_conversion_factor(rxn):
        """Effective conversion factor shared by every species this reaction
        *changes*. Returns ``(cf, mixed)``: ``cf`` is the shared value (1.0 if
        the reaction changes no species), ``mixed`` is True when the changed
        species carry more than one distinct cf — which a single rate scale
        cannot represent, so the caller refuses."""
        net: dict[str, float] = {}
        # The stoichiometry the emission uses (an initialAssignment or
        # stoichiometryMath included): read off the attribute, `A -> 2A` with
        # the 2 set by an initialAssignment read as a catalyst, and the cf was
        # dropped, for the ODE and the SSA refusal alike.
        for _j in range(rxn.getNumReactants()):
            _sr = rxn.getReactant(_j)
            _st = _resolve_stoich(_sr)
            _st = 1.0 if _st is None else _st
            net[_sr.getSpecies()] = net.get(_sr.getSpecies(), 0.0) - _st
        for _j in range(rxn.getNumProducts()):
            _sr = rxn.getProduct(_j)
            _st = _resolve_stoich(_sr)
            _st = 1.0 if _st is None else _st
            net[_sr.getSpecies()] = net.get(_sr.getSpecies(), 0.0) + _st
        cfs = set()
        for _sid, _n in net.items():
            if abs(_n) < 1e-12:
                continue  # catalyst / net-zero: its cf never enters the ODE
            _sp = sbml_model.getSpecies(_sid)
            if _sp is not None and (_sp.getBoundaryCondition() or _sp.getConstant()):
                continue  # amount is fixed; the reaction does not change it
            cfs.add(species_cf.get(_sid, 1.0))
        if not cfs:
            return 1.0, False
        if len(cfs) == 1:
            return next(iter(cfs)), False
        return None, True

    # Second pass over rules now that species_idx is populated: classify
    # each species-targeted AssignmentRule as linear-on-species or other.
    # Linear ones get emitted as observables with weighted entries below;
    # non-linear ones keep the synthetic-param-via-add_function path.
    # GH #75: amount-restoration for hOSU=true species reads is now a single
    # engine capability. ``Species::amount_valued`` makes update_observables
    # read such a species as its amount (stored × V_c), and because an SBML
    # rate law / observable-expression / assignment RHS references a species
    # through its same-named observable (which shadows the species variable in
    # the evaluator), every such reference picks up the amount automatically —
    # no per-emission ``s → s*V_c`` rewrite. The one residual loader concern is
    # the *target* side of an assignment whose target is itself an hOSU=true
    # V≠1 species:
    #   • A direct EVENT assignment writes the species's stored concentration
    #     slot, so the assigned value (an amount) must be ÷V_c(target) here.
    #   • An AssignmentRule target stays amount-valued (its function/observable
    #     yields the amount, which other expressions read correctly); the
    #     ÷V_c(target) for the reported concentration is applied at report time
    #     (Simulator._apply_ar_report_map), not in the emitted math.
    # Returns the node unchanged for V=1 / hOSU=false targets (byte-identical).
    def _divide_by_target_vc(math, target_var):
        # (#170) The `V_c != 1` shortcut is the V=1 normalisation again: at V=1 the
        # divide was dropped as a ÷1 no-op, so an event that assigns an amount wrote
        # it straight into the stored slot — and a later `set_param` on the volume
        # could not move it, leaving the species exactly V_new/V_load too large
        # (BIOMD0000000241's dose events, 2.5x on a 2.5x write). Emit it whenever the
        # target's compartment size is a writable PARAMETER, whatever this load's
        # value: `x/1.0 == x` exactly, so the nominal point is byte-identical.
        _tcomp = species_comp.get(target_var, "")
        if (
            target_var in species_idx
            and species_hosu.get(target_var, False)
            and (comp_volumes.get(_tcomp, 1.0) != 1.0 or _tcomp in live_volume_param_comps)
        ):
            target_comp = species_comp[target_var]
            div = libsbml.ASTNode(libsbml.AST_DIVIDE)
            div.addChild(math.deepCopy())
            # (#87, #114) For a target in an assignment-rule or rate-rule
            # (variable-volume) compartment, the stored slot is ``amount/V_static``
            # so the event-assigned amount must be ÷ the load-time numeric
            # V_static, not the live compartment symbol — dividing by V_live(t)
            # would corrupt the post-event stored value as the compartment grows.
            # Static compartments keep the symbol divide (byte-identical: symbol
            # value equals the numeric).
            if target_comp in vstatic_divide_comps:
                ctgt = libsbml.ASTNode(libsbml.AST_REAL)
                ctgt.setValue(float(comp_volumes.get(target_comp, 1.0)))
            else:
                ctgt = libsbml.ASTNode(libsbml.AST_NAME)
                ctgt.setName(target_comp)
            div.addChild(ctgt)
            return div
        return math

    _idx_to_sid = {idx: sid for sid, idx in species_idx.items()}

    linear_assignment_rules: dict[str, list[tuple[int, float]]] = {}
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if not rule.isAssignment():
            continue
        var = rule.getVariable()
        if var not in species_idx:
            continue
        math = rule.getMath()
        if math is None:
            continue
        entries = _classify_linear_species_rule(math, species_idx)
        if entries and any(_idx_to_sid[idx] in assignment_targets for idx, _ in entries):
            # A linear rule whose summands are themselves AssignmentRule-target
            # species (a chained AR, e.g. MODEL1112260002
            # ``Foxo1_all = cytoplasm_Foxo1_tot + nucleus_Foxo1_tot + …``). An
            # observable can only read raw species conc slots, and AR-target
            # species are emitted ``fixed`` (frozen at t=0), so the observable
            # would sum stale values. Route to the §6 function path instead,
            # where the bare summand names resolve to the live observable /
            # function values (observables refresh before functions each RHS).
            entries = None
        if entries:
            # GH #75: summand amount-restoration (×V_c for each hOSU=true
            # summand) is now done by the engine — update_observables reads an
            # amount_valued species as its amount, and this observable's entries
            # reference those species, so the weighted sum already accumulates
            # amounts. The weights are therefore the literal rule coefficients.
            # The ÷V_c(target) needed when the target is itself an hOSU=true V≠1
            # species (to report stored concentration) is applied at report
            # time (Simulator._apply_ar_report_map). Byte-identical for V=1 /
            # hOSU=false targets and summands.
            linear_assignment_rules[var] = entries

    # ── 4. Observables (one per species) ──────────────────────────────
    # Assignment-rule species: the function (step 6) uses the species
    # name directly. To avoid ExprTk duplicate-variable errors, create
    # the observable with an `_obs_` prefix. The harness maps it back.
    #
    # Linear-on-species AssignmentRule species are the exception: their
    # observable IS the rule (e.g., ``y = 2*X`` becomes observable ``y``
    # with one entry ``(idx_X, 2.0)``), so the bare name is bound to a
    # time-varying observable rather than a stuck synthetic parameter.
    # Section 6 below also skips the add_function call for these.
    for sid in species_ids:
        if sid in linear_assignment_rules:
            builder.add_observable(_safe_name(sid), linear_assignment_rules[sid])
        elif sid in assignment_targets:
            builder.add_observable(f"_obs_{_safe_name(sid)}", [(species_idx[sid], 1.0)])
        else:
            builder.add_observable(_safe_name(sid), [(species_idx[sid], 1.0)])

    # ── 5. SBML Function Definitions ──────────────────────────────────
    # These are like macros: functionDef(a, b) = a + b
    # We inline them during AST translation by substituting arguments.
    # The dict was already collected near the top of this function
    # (so _eval_ast_numeric can inline user calls during section 0); the
    # ExprTk translator below reuses the same mapping.

    # ── 6. Assignment rules → functions ───────────────────────────────
    # Linear-on-species rules are already wired as observables in step 4;
    # don't also bind a synthetic parameter — that would create a duplicate
    # ExprTk variable name (the observable already claims it).
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isAssignment():
            var = rule.getVariable()
            if var in linear_assignment_rules:
                continue
            math = rule.getMath()
            if math:
                # GH #75: the function yields the target's value reading any
                # referenced species as amounts (engine amount_valued path). For
                # an hOSU=true target the function value IS the target's amount;
                # the ÷V_c(target) to report stored concentration is applied at
                # report time (Simulator._apply_ar_report_map), so the bare math
                # is emitted here. Byte-identical for V=1 / hOSU=false models.
                expr = _ast_to_exprtk_with_funcdefs(math, func_defs)
                builder.add_function(_safe_name(var), expr)

    # _resolve_stoich must be available to both the kinetic-law-classification
    # loop (section 7) and the per-species emission loop (section 9), so it's
    # hoisted here.
    def _resolve_stoich(sr):
        """Return the constant stoichiometry of a speciesReference, or None.

        Resolution order, matching SBML semantics:
          1. L2 <stoichiometryMath> child: constant-fold the math AST.
          2. L3 speciesReference id matching an initialAssignment symbol:
             use the IA-computed value.
          3. The plain ``stoichiometry`` attribute, if it is a finite
             number (libsbml returns NaN when the attribute is absent
             and no fallback applies).
        Returns None when none of the above yields a finite value.
        """
        # 1) L2 <stoichiometryMath>
        sm_math = None
        if hasattr(sr, "getStoichiometryMath"):
            sm = sr.getStoichiometryMath()
            if sm is not None and sm.getMath() is not None:
                sm_math = sm.getMath()
        if sm_math is not None:
            val = _eval_ast_numeric(sm_math, eval_ctx, func_defs)
            if val is not None and _math.isfinite(val):
                return val
        # 2) L3 speciesReference id with initialAssignment
        sr_id = sr.getId() if hasattr(sr, "getId") else ""
        if sr_id and sr_id in ia_values:
            val = ia_values[sr_id]
            if _math.isfinite(val):
                return val
        # 3) Plain stoichiometry attribute
        s = sr.getStoichiometry()
        if _math.isfinite(s):
            return s
        return None

    # ── 6b. Named species-reference stoichiometry as a symbol (GH #237) ─────
    # An SBML <speciesReference> may carry an id; that id is a first-class
    # symbol whose value is the reactant/product stoichiometry and may be read
    # in a kinetic law / rule (testTag SpeciesReferenceInMath). bngsim bakes the
    # stoichiometry into the reaction coefficient (via _resolve_stoich in §9) and
    # otherwise never registers the id, so a rate law like ``S1_stoich * k`` is
    # an undefined symbol and fails to compile. Register each *static* stoich id
    # as a constant parameter equal to its resolved coefficient so it resolves
    # everywhere a global parameter would. SBML SId uniqueness guarantees no
    # clash with a species/parameter/compartment id; a local parameter that
    # shadows the id stays scoped to its kinetic law via the local_params remap.
    #
    # Static ids become constant parameters here. A stoich id targeted by an
    # assignment/rate rule or event is variable; §6c keeps it symbolic and §8/§10
    # promote it to live state when needed. Registering a frozen parameter for
    # those would silently mis-integrate.
    _reserved_ids = set(_param_ids) | set(_species_ids_set) | set(comp_volumes)
    _variable_stoich_ids = rate_rule_targets | assignment_targets | event_promoted_params
    _species_ref_initial_values: dict[str, float] = {}
    for _sr in _iter_species_references(sbml_model):
        _srid = _sr.getId() if hasattr(_sr, "getId") else ""
        if not _srid or _srid in _species_ref_initial_values:
            continue
        _stoich = _resolve_stoich(_sr)
        if _stoich is None or not _math.isfinite(_stoich):
            continue
        _species_ref_initial_values[_srid] = float(_stoich)

    for _srid, _stoich in _species_ref_initial_values.items():
        if _srid in _reserved_ids or _srid in _variable_stoich_ids:
            continue
        builder.add_parameter(_safe_name(_srid), float(_stoich))

    # ── 6c. VARIABLE species-reference stoichiometry (GH #237 Phase 2) ──────
    # The reaction-extent contribution of a species is ``stoich_{s,r} · v_r``;
    # when the stoichiometric coefficient is itself time-varying (an L2
    # <stoichiometryMath> that reads time / a species / a rate-rule- or
    # assignment- or event-driven parameter, or an L3 speciesReference id
    # directly targeted by such a rule), ``dS/dt`` must use the LIVE coefficient
    # rather than its frozen load-time value. Phase 1 (§6b) bakes only static
    # coefficients; Phase 2 keeps the coefficient symbolic.
    #
    # The engine already accumulates ``dS/dt += stat_factor · funcvalue`` for a
    # per-species Functional reaction (the non-integer-stoich fallback in §9), so
    # a time-varying coefficient needs no kernel change: emit the affected
    # species as its own Functional reaction whose rate function is
    # ``law · stoich_expr`` (the §9 block below). This helper returns that
    # ``stoich_expr`` for a variable reference, or None for a constant one (which
    # _resolve_stoich folds as before). A reaction with any variable reference is
    # forced off the Elementary/mass-action path onto §9.
    #
    # Covered: L2 <stoichiometryMath> whose value varies with time / a species / a
    # rate-rule-, assignment-, or event-driven parameter, and L3
    # <speciesReference> ids directly targeted by a rule/event (00972/01583).
    def _variable_stoich_expr(_sr):
        _srid = _sr.getId() if hasattr(_sr, "getId") else ""
        if _srid and _srid in _variable_stoich_ids:
            return _safe_name(_srid)
        if hasattr(_sr, "getStoichiometryMath"):
            _sm = _sr.getStoichiometryMath()
            if _sm is not None and _sm.getMath() is not None:
                _sm_math = _sm.getMath()
                _names = {
                    _n.getName()
                    for _n in _iter_ast_subtree(_sm_math)
                    if _n.getType() == libsbml.AST_NAME and _n.getName()
                }
                _varying = (
                    _ast_references_time(_sm_math)
                    or bool(_names & _species_ids_set)
                    or bool(_names & _variable_stoich_ids)
                )
                if _varying:
                    return _ast_to_exprtk_with_funcdefs(_sm_math, func_defs)
        return None  # constant stoichiometryMath → _resolve_stoich folds it

    _variable_stoich_rxns: set[int] = set()
    for _ri in range(sbml_model.getNumReactions()):
        _rxn = sbml_model.getReaction(_ri)
        for _getter, _n in (
            (_rxn.getReactant, _rxn.getNumReactants()),
            (_rxn.getProduct, _rxn.getNumProducts()),
        ):
            for _j in range(_n):
                if _variable_stoich_expr(_getter(_j)) is not None:
                    _variable_stoich_rxns.add(_ri)
                    break
            if _ri in _variable_stoich_rxns:
                break

    # SSA validation: AssignmentRule-bound reactants. Walks all reactions
    # (mass-action and not). The warning fires once per (reaction,
    # species) pair so users hearing about the same rule on multiple
    # reactions still see each binding called out.
    for i in range(sbml_model.getNumReactions()):
        _rxn = sbml_model.getReaction(i)
        _rid = _rxn.getId()
        _seen: set[str] = set()
        for j in range(_rxn.getNumReactants()):
            _sid = _rxn.getReactant(j).getSpecies()
            if _sid in _seen or _sid not in species_idx:
                continue
            _seen.add(_sid)
            if _sid in assignment_targets:
                ssa_issues.append(
                    SsaIssue(
                        severity="warning",
                        code="assignment_rule_on_reactant",
                        message=(
                            f"Reaction '{_rid}' consumes species '{_sid}', "
                            "whose value is set by an AssignmentRule. SSA "
                            "fires will not update the rule-bound species "
                            "directly; its trajectory tracks the rule's "
                            "right-hand side. Verify this is intentional."
                        ),
                        location=f"reaction:{_rid}:species:{_sid}",
                    )
                )

    # ── 7. Reaction rate functions ────────────────────────────────────
    # Mass-action kinetic laws (issue #16) take the Elementary path: we
    # skip registering the kinetic law as a Function and instead emit a
    # single Elementary reaction in section 9. Phase 7 reversible-split
    # kinetic laws emit TWO Elementary reactions for one SBML reaction id;
    # the cache value is therefore a list (1-element for plain mass-action,
    # 2-element for splits).
    mass_action_rxns: dict[int, list[tuple]] = {}

    _v0_param_names: dict[str, str] = {}
    # (#170) Names already spoken for, so a synthesized parameter cannot collide
    # with one the SBML declares. `_rateLaw_<rid>` used to be synthesized only for
    # a MULTI-component rate, which is rare enough that nobody had hit the clash;
    # a volume power now synthesizes one for ordinary single-constant laws too,
    # and bngsim's own `.net` → SBML export writes `_rateLaw_<rid>` out as a real
    # parameter — so re-importing that file collided on the first reaction and
    # `ModelBuilder::validate` rejected the model outright. Uniquifying covers the
    # pre-existing hazard as well. A species' id is a symbol of its own too, and
    # a reaction's id the function its rate law is read by: a function the
    # loader names after either overwrote it or failed to register (#1124).
    _taken_param_names: set[str] = (
        {
            _safe_name(sbml_model.getParameter(j).getId())
            for j in range(sbml_model.getNumParameters())
        }
        | {
            _safe_name(sbml_model.getCompartment(j).getId())
            for j in range(sbml_model.getNumCompartments())
        }
        | {_safe_name(sp.getId()) for sp in sbml_model.getListOfSpecies()}
        | {
            _safe_name(element.getId())
            for rxn in sbml_model.getListOfReactions()
            for element in (
                rxn,
                *rxn.getListOfReactants(),
                *rxn.getListOfProducts(),
                *rxn.getListOfModifiers(),
            )
            if element.getId()
        }
    )

    def _unique_param_name(base: str) -> str:
        name = base
        n = 0
        while name in _taken_param_names:
            n += 1
            name = f"{base}_{n}"
        _taken_param_names.add(name)
        return name

    def _load_time_volume_param(cid: str) -> str:
        """Name of a constant parameter holding compartment ``cid``'s load-time
        size, added on first use (#170).

        The volume ratio below has to evaluate to *exactly* 1.0 at the nominal
        point or every mass-action rate constant in the model shifts by an ulp,
        so the denominator must be the same binary double the compartment
        parameter holds. Writing it as a decimal literal is not enough: ExprTk's
        own literal parser is not correctly rounded, and BIOMD0000000706 is the
        case that proves it — ``cytoplasm / 1.65e-11`` (a repr round-trip that is
        exact in Python) comes back as 0.9999999999999998, turning a 5e-05 rate
        constant into 4.999999999999999e-05 and moving the model's RHS. Passing
        the value through ``add_parameter`` hands the engine the double itself,
        with no decimal round trip in between.
        """
        name = _v0_param_names.get(cid)
        if name is None:
            name = _unique_param_name(_safe_name(f"_V0_{cid}"))
            builder.add_parameter(name, float(comp_volumes.get(cid, 1.0)), is_internal=True)
            _v0_param_names[cid] = name
        return name

    def _classification_to_cache_tuple(classification, rid_safe, derived_suffix=""):
        """Turn ``_classify_mass_action`` output into the
        ``(rate_param_name, sf, reactants, products)`` cache tuple.
        Synthesizes a derived parameter named
        ``_rateLaw_<rid_safe><derived_suffix>`` when the rate has multiple
        constant-parameter components (so the codegen sensitivity path can
        chain-rule through a single rate constant). The initial value is
        the product of the constituent values; the ExprTk expression
        reuses each parameter's mangled name so ``set_param`` updates
        propagate correctly. Empty ``derived_suffix`` preserves the
        pre-Phase-7 derived-name convention for non-split callers.

        (#170) When ``sf`` folded a non-zero net compartment power, the same
        derived parameter carries the volume back as a live symbol:
        ``k * (C/V_base)^n``. Two properties make that safe to do corpus-wide:

        * ``(C/V_base)`` is *exactly* 1.0 while ``C`` holds its load-time value
          (``x/x == 1.0`` for every finite non-zero x, and ``k*1.0 == k``), so the
          rate constant is bit-identical until something is actually written —
          ``sf`` itself is left alone, still carrying the whole numeric fold; and
        * an expression-valued parameter is re-evaluated by ``set_param`` already
          (issue #43's chain rule), so every consumer that reads the rate through
          ``params[rate_param_idx0]`` — the interpreted RHS, the analytical
          Jacobian, the emitted C, ``∂f/∂p`` — becomes live with no change of its
          own. That is why the volume rides the rate parameter rather than
          ``Reaction::stat_factor``: reactions are SHARED across model clones, so
          rescaling the scalar in place would write through every sibling of the
          model being scanned, while the parameter vector is per-instance.
        """
        (
            rate_param_components,
            sf,
            reactants,
            products,
            comp_factors,
            volume_powers,
            volume_unresolvable,
        ) = classification
        compartment_write_refused.update(volume_unresolvable)
        vol_terms = ""
        for cid, power in sorted(volume_powers.items()):
            ratio = f"({_safe_name(cid)} / {_load_time_volume_param(cid)})"
            op = " * " if power > 0 else " / "
            vol_terms += (op + ratio) * abs(power)
        if len(rate_param_components) == 1 and not vol_terms:
            rate_param_name = rate_param_components[0][0]
        else:
            derived_name = _unique_param_name(f"_rateLaw_{rid_safe}{derived_suffix}")
            expr = " * ".join(name for name, _ in rate_param_components) + vol_terms
            init_value = 1.0
            for _, val in rate_param_components:
                init_value *= val
            builder.add_parameter(derived_name, init_value, expression=expr, is_expression=True)
            rate_param_name = derived_name
        # 5th element (#81): comp_id → law-factor power, for the SSA varvol tagger.
        return (rate_param_name, sf, reactants, products, comp_factors)

    # SBML L3: a <ci> referencing a reaction id evaluates to that reaction's
    # kineticLaw value (the reaction's rate of progress, in extent/time).
    # Most reaction kineticLaws are emitted as Elementary mass-action and
    # never need a function under the rid name, but if some OTHER expression
    # names a reaction id, we must register the rid as an ExprTk function so
    # that expression compiles. Two referencing contexts (GH #91):
    #   • a rule (rate or assignment) whose RHS names a reaction id, and
    #   • another reaction's kineticLaw that names a reaction id, e.g.
    #     MODEL2306170002 r0 = … * ((r9b - r10a) + r6n_c) where r9b/r10a/
    #     r6n_c are themselves reactions emitted as mass-action.
    # Pre-scan both to find which reaction ids are referenced this way; the
    # reaction loop below emits those functions (via the rate_expr_emitted
    # branch) before falling into the mass-action / split / unified-Functional
    # paths. A reaction's own id is excluded — a kineticLaw referencing its
    # own rate is a self-referential fixed point, not a resolvable symbol.
    _reaction_ids_set = {
        sbml_model.getReaction(j).getId() for j in range(sbml_model.getNumReactions())
    }
    referenced_reaction_ids: set[str] = set()
    for j in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(j)
        rmath = rule.getMath()
        if rmath is None:
            continue
        _walk_name_refs(rmath, _reaction_ids_set, referenced_reaction_ids)
    for j in range(sbml_model.getNumReactions()):
        rxn_j = sbml_model.getReaction(j)
        kl_j = rxn_j.getKineticLaw()
        if kl_j is None or kl_j.getMath() is None:
            continue
        refs_j: set[str] = set()
        _walk_name_refs(kl_j.getMath(), _reaction_ids_set, refs_j)
        refs_j.discard(rxn_j.getId())  # drop self-reference
        referenced_reaction_ids |= refs_j
    # An event reads a reaction's id as a rule does: `at J0 < 15` failed to
    # compile, J0 an undefined symbol, where libRoadRunner fires it at J0's
    # crossing.
    for j in range(sbml_model.getNumEvents()):
        ev_j = sbml_model.getEvent(j)
        ev_maths = [
            x.getMath()
            for x in (ev_j.getTrigger(), ev_j.getDelay(), ev_j.getPriority())
            if x is not None
        ]
        ev_maths += [
            ev_j.getEventAssignment(k).getMath() for k in range(ev_j.getNumEventAssignments())
        ]
        for m_j in ev_maths:
            if m_j is not None:
                _walk_name_refs(m_j, _reaction_ids_set, referenced_reaction_ids)

    # What a name in a kinetic law is, for _law_is_nonnegative: a species or a
    # compartment is, a constant parameter is by its value after initial
    # assignments, a rule's target is by its rule, anything a rate rule or an
    # event writes is not (it can be made negative), nor is anything else.
    _sign_memo: dict[str, bool] = {}
    # What an event writes, by target: an event can write any value, so its
    # target is non-negative only where every value it writes is.
    _event_writes: dict[str, list] = {}
    for _e in range(sbml_model.getNumEvents()):
        _ev = sbml_model.getEvent(_e)
        for _a in range(_ev.getNumEventAssignments()):
            _ea = _ev.getEventAssignment(_a)
            _event_writes.setdefault(_ea.getVariable(), []).append(_ea.getMath())

    class _SignLookup:
        def __init__(self, local_map, local_values):
            self.func_defs = func_defs
            self.local_map = local_map or {}
            self.local_values = local_values or {}

        def __call__(self, name, _seen) -> bool:
            if name in self.local_map:
                v = self.local_values.get(self.local_map[name])
                return v is not None and v == v and v >= 0
            if name in _sign_memo:
                return _sign_memo[name]
            _sign_memo[name] = False  # a cycle reads as not proven
            _sign_memo[name] = _global_is_nonnegative(name, _seen)
            return _sign_memo[name]

    def _global_is_nonnegative(name, _seen) -> bool:
        # Rules, rates and event values are read in the model's scope, not the
        # reaction's: a kinetic law's local of the same name does not reach them.
        if name in assignment_targets:
            rule = sbml_model.getAssignmentRuleByVariable(name)
            return (
                rule is not None
                and rule.getMath() is not None
                and _law_is_nonnegative(rule.getMath(), _global_sign, {}, _seen)
            )
        ia = sbml_model.getInitialAssignmentBySymbol(name)
        if ia is not None and ia.getMath() is not None:
            # Its start is the assignment's value. Where the load cannot fold it
            # (`delay(-3, 0)`), ia_values keeps the declared one and the math
            # decides.
            v0 = _eval_ast_numeric(ia.getMath(), eval_ctx, func_defs)
            if v0 is None or v0 != v0:
                if not _law_is_nonnegative(ia.getMath(), _global_sign, {}, _seen):
                    return False
            elif v0 < 0:
                return False
        else:
            if name in species_idx:
                sp = sbml_model.getSpecies(name)
                v = ia_values.get(name, float("nan"))
                if v != v and sp is not None:
                    if sp.isSetInitialAmount():
                        v = sp.getInitialAmount()
                    elif sp.isSetInitialConcentration():
                        v = sp.getInitialConcentration()
            elif name in comp_volumes:
                v = ia_values.get(name, comp_volumes[name])
            else:
                par = sbml_model.getParameter(name)
                if par is None:
                    return False
                v = ia_values.get(name, par.getValue() if par.isSetValue() else float("nan"))
            if not (v == v and v >= 0):
                return False
        # It starts non-negative. A rate rule keeps it so if its rate is
        # non-negative; an event, if every value it writes is; each read with
        # the variable itself non-negative, which is what holds them so by
        # induction (`c' = 0.2*c`, `S = S/2`).
        itself = {name: True}
        if name in rate_rule_targets:
            rate = sbml_model.getRateRuleByVariable(name)
            if rate is None or not _law_is_nonnegative(
                rate.getMath(), _global_sign, itself, _seen
            ):
                return False
        for written in _event_writes.get(name, ()):
            if written is not None and not _law_is_nonnegative(
                written, _global_sign, itself, _seen
            ):
                return False
        return True

    _global_sign = _SignLookup({}, {})

    reaction_local_param_maps: dict[int, dict[str, str]] = {}
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        rid = rxn.getId()

        # SSA validation: fast="true" reactions are out of scope (the SSA
        # backend does not solve fast-equilibrium constraints between
        # firing events). Detect on the SBML reaction directly so the
        # diagnostic surfaces even when there is no kinetic law.
        if rxn.isSetFast() and rxn.getFast():
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="fast_reaction",
                    message=(
                        f"Reaction '{rid}' has fast='true', which is not "
                        "supported under SSA (fast-equilibrium constraints "
                        "would have to be enforced between every fire). "
                        "Replace the fast-flag reaction with explicit forward "
                        "and reverse reactions or use method='ode'."
                    ),
                    location=f"reaction:{rid}",
                )
            )

        # SSA validation: a conversionFactor other than 1 scales how far each
        # firing moves a species. bngsim folds it into the rate instead (§3b),
        # which keeps the ODE and the SSA mean but gives the wrong noise under
        # SSA: the reaction fires cf times as often in unit jumps (variance k·T
        # where cf²·k·T is right). Refused until the jump itself is scaled.
        if _has_conversion_factor:
            _rcf, _rmixed = _reaction_conversion_factor(rxn)
            if _rmixed or _rcf != 1.0:
                ssa_issues.append(
                    SsaIssue(
                        severity="error",
                        code="conversion_factor",
                        message=(
                            f"Reaction '{rid}' changes species whose SBML "
                            f"conversionFactor is not 1 ({'mixed' if _rmixed else _rcf}). "
                            "Under SSA each firing would have to move them by "
                            "conversionFactor x stoichiometry; bngsim scales the rate "
                            "instead, which keeps the mean but not the noise, so the "
                            "model is refused for SSA/PSA. Use method='ode', or fold "
                            "the factor into the stoichiometry."
                        ),
                        location=f"reaction:{rid}",
                    )
                )

        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        math = kl.getMath()
        if math is None:
            continue

        # conversionFactor (GH #232): does this reaction change species with
        # DIFFERING factors? A single rate scale can't represent that, so a mixed
        # reaction is forced off the Elementary path onto the §9 Functional path,
        # where it is split per cf-group. Uniform-cf reactions (incl. cf-free)
        # keep their Elementary/Functional route and are scaled at emission.
        _cf_mixed = False
        if _has_conversion_factor:
            _, _cf_mixed = _reaction_conversion_factor(rxn)

        # KineticLaw local parameters — handle BOTH L2 and L3 APIs.
        # SBML L2: kl.getNumParameters() / kl.getParameter(j)
        # SBML L3: kl.getNumLocalParameters() / kl.getLocalParameter(j)
        local_param_map, local_param_values = _collect_local_params(kl, rid, builder)
        # Keep the id→mangled map for §10.5's discontinuity scan, which reads
        # the same kinetic-law math and must emit the same symbol names the RHS
        # does. Captured here rather than recomputed there: `_collect_local_params`
        # owns the mangling AND registers the parameter, so a second call would
        # double-register and a re-derived name is one more site to keep in step.
        reaction_local_param_maps[i] = local_param_map

        # If a rule or another reaction's kineticLaw references this reaction
        # id, emit `add_function(rid, ...)` up-front so any downstream
        # expression compiles regardless of which reaction-emission path
        # (mass-action / split / unified) we take below. The unified-Functional
        # path (last branch) only emits the function when we have NOT already
        # emitted it here.
        rate_expr_emitted = False
        if rid in referenced_reaction_ids:
            rate_expr_for_ref = _ast_to_exprtk_with_funcdefs(
                math,
                func_defs,
                local_params=local_param_map,
            )
            builder.add_function(rid, rate_expr_for_ref)
            rate_expr_emitted = True

        # A reversible law that is not a difference runs as an irreversible one.
        _rev_difference = rxn.getReversible() and not _law_is_nonnegative(
            math, _SignLookup(local_param_map, local_param_values)
        )

        _varvol_reject: dict = {}
        classification = _classify_mass_action(
            rxn,
            math,
            sbml_model,
            species_idx,
            species_hosu,
            species_comp,
            comp_volumes,
            rate_rule_targets,
            event_promoted_params,
            assignment_targets,
            ar_comp_moving,
            local_param_map,
            local_param_values,
            _resolve_stoich,
            varvol_reject=_varvol_reject,
        )
        if classification is not None and not _cf_mixed and i not in _variable_stoich_rxns:
            # (GH #232) A mixed-cf reaction skips the Elementary cache so it falls
            # through to the §9 Functional path, where the cf-group split applies
            # a per-species factor the single Elementary stat_factor cannot.
            # (GH #237 Phase 2) A variable-stoichiometry reaction likewise skips
            # the Elementary cache: the Elementary stat_factor bakes the load-time
            # coefficient, but §9 keeps it symbolic (``law · stoich_expr``).
            mass_action_rxns[i] = [_classification_to_cache_tuple(classification, _safe_name(rid))]
            continue

        # (#144) A single-compartment monomial that did NOT classify as
        # mass-action only because a variable-volume compartment's power doesn't
        # cancel (case 2) or it carries an hOSU=true law factor (case 1). It must
        # take the Functional path so the ODE is correct, but the SSA propensity
        # admits the exact scalar live-volume correction §9 will apply. Skip a
        # reversible law written as a forward-minus-reverse difference, which is
        # not a monomial — keep refusing those.
        if _varvol_reject.get("comp") in varvol_ssa_comps and not _rev_difference:
            ssa_varvol_functional[i] = int(_varvol_reject["n_f"])

        # (#144 case 4, #172) A cross-compartment variable-volume mass-action
        # monomial (≥2 storage compartments, ≥1 variable-volume, all-hOSU=false) the
        # classifier routed to the Functional path — bare law (#144) or carrying an
        # explicit variable-volume compartment factor (#172, `cell·k·A·B`). Certify
        # it so §9 lifts the varvol_non_mass_action gate and applies the
        # per-compartment ODE live divide + SSA propensity correction (both keyed on
        # species factors only; an explicit compartment factor lives in base_func and
        # cancels — see _classify_mass_action_ast). Not a reversible law written
        # as a forward-minus-reverse difference, which is not a monomial.
        #
        # Nor one the per-species divide cannot serve, which §9 writes out one
        # species at a time instead and SSA refuses: a reaction that changes a
        # species in an assignment-rule compartment whose size moves, which is no
        # state to divide by (#745), or that mixes conversion factors. Certified,
        # those took the per-species divide anyway: an irreversible `A + E => B + E`
        # with A and B in `C1 := 2 + 0.5*time` gave [A](5) = 0.287 for 0.326, and a
        # mixed factor refused to load, where the same law flagged reversible was
        # right. A rule over constants (`C := 2`) holds its load-time size, as a
        # write to what it reads is refused, so the per-species divide by that
        # size serves its species as it serves a static compartment's, and the
        # SSA reads their counts over it exactly.
        _xcomp_varvol = _varvol_reject.get("xcomp_varvol_comps")
        if _xcomp_varvol and not _rev_difference and not _cf_mixed:
            _net: dict[str, float] = {}
            for _sign, _refs in ((-1.0, rxn.getListOfReactants()), (1.0, rxn.getListOfProducts())):
                for _sr in _refs:
                    _st = _resolve_stoich(_sr)
                    # An unresolved stoichiometry (NaN) is taken for a change.
                    _net[_sr.getSpecies()] = _net.get(_sr.getSpecies(), 0.0) + _sign * (
                        float("nan") if _st is None else _st
                    )
            if not any(
                not abs(_n) <= 1e-12 and species_comp[_sid] in ar_comp_moving
                for _sid, _n in _net.items()
            ):
                ssa_varvol_xcompartment[i] = _xcomp_varvol

        # Phase 7: try splitting a reversible kineticLaw of the form
        # ``[wrapper *] (forward_rate - reverse_rate)`` into two mass-action
        # channels. Only attempted when the SBML reaction is flagged
        # reversible='true'; non-reversible kineticLaws written as a
        # difference are out of scope (and the per-side classifier would
        # accept them in ways the user likely didn't intend).
        if rxn.getReversible() and not _cf_mixed:
            split = _split_reversible_kinetic_law(
                rxn,
                math,
                sbml_model,
                species_idx,
                species_hosu,
                species_comp,
                comp_volumes,
                rate_rule_targets,
                event_promoted_params,
                assignment_targets,
                ar_comp_moving,
                local_param_map,
                local_param_values,
                _resolve_stoich,
            )
            if split is not None:
                fwd, rev = split
                rid_safe = _safe_name(rid)
                mass_action_rxns[i] = [
                    _classification_to_cache_tuple(fwd, rid_safe, "_fwd"),
                    _classification_to_cache_tuple(rev, rid_safe, "_rev"),
                ]
                continue

        # GH #75: the §9 unified Functional emission now reads hOSU=true
        # species as amounts (Species::amount_valued ⇒ update_observables
        # restores the amount, and the kinetic-law function references each
        # species through its same-named observable). The former
        # ``non_mass_action_volumetric_species`` SSA gate — which rejected
        # non-mass-action laws over hOSU=true V≠1 species because the rate was
        # evaluated on concentration instead of amount — is therefore obsolete.
        # The single-compartment SSA propensity (law(amount)/V_c · ssa_volume_
        # factor V_c = law(amount)) and the cross-compartment propensity
        # (law(amount), with per-species /V_c applied at the fire step) are both
        # the correct amount/time rate. This closes the latent
        # multi-compartment-hOSU-Functional-under-SSA gap (was Phase 2.7).

        if _rev_difference:
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="reversible_non_mass_action",
                    message=(
                        f"Reaction '{rid}' is reversible='true' with a "
                        "kinetic law that is a difference (forward minus "
                        "reverse) and did not split into two mass-action "
                        "channels. Under SSA the loader emits a single net-rate "
                        "channel (forward - reverse), so the propensity "
                        "drops to zero at the deterministic equilibrium "
                        "and the trajectory locks at the fixed point "
                        "instead of fluctuating. The full fix (split into "
                        "forward and reverse SSA channels) is tracked as a "
                        "future SBML SSA loader phase; workarounds: "
                        "rewrite the SBML with two non-reversible "
                        "reactions (one forward, one reverse), or use "
                        "method='ode'."
                    ),
                    location=f"reaction:{rid}",
                )
            )

        if not rate_expr_emitted:
            rate_expr = _ast_to_exprtk_with_funcdefs(
                math,
                func_defs,
                local_params=local_param_map,
            )
            builder.add_function(rid, rate_expr)

        # GH #75: no per-reaction ``_amt_<rid>`` amount-substituted variant is
        # needed anymore. The §9 Functional emission below routes through the
        # raw ``rid`` law, whose hOSU=true species references resolve (via their
        # same-named observables) to amounts through Species::amount_valued. The
        # /V_c accumulation (single-compartment `_vd_` wrapper and the
        # cross-compartment per_species_volume_scaling divide) is unchanged.

    # ── 8. Rate rules → ODE reactions ─────────────────────────────────
    # Rate rules can target species OR parameters. Pure ODE models
    # from Antimony (e.g., S' = -k*S) emit S as an SBML parameter
    # with a rate rule. We promote such parameters to species.
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isRate():
            var = rule.getVariable()
            math = rule.getMath()
            # A rate rule with no <math> means dvar/dt = 0 (var holds its initial
            # value). Promote var to a species with a zero RHS anyway, so a
            # rateOf(var) accessor still binds to it (suite 01461) — skipping it
            # would leave var a plain parameter with no rate_of__var to reference.

            # If var is a species, use it directly
            if var in species_idx:
                sp_i = species_idx[var]
            else:
                # Promote parameter (or compartment) to species. A rate rule
                # can target a compartment (variable volume) as well as a
                # parameter (pure ODE pattern), and step 1 deliberately skips
                # adding a builder parameter for such a compartment to avoid an
                # ExprTk symbol collision — so its initial size lives only in
                # ``comp_volumes`` here, NOT as an SBML parameter. A rate rule
                # can also target an L3 speciesReference id (GH #237); seed that
                # live stoichiometry from its declared/initial-assigned value.
                # Check ia_values first (e.g. ``W = Wf*exp(-b*a^(-C))``), then
                # the compartment size (GH #85), then the named
                # species-reference stoichiometry, then a plain parameter value.
                if var in ia_values:
                    ic = ia_values[var]
                elif var in comp_volumes:
                    ic = comp_volumes[var]
                elif var in _species_ref_initial_values:
                    ic = _species_ref_initial_values[var]
                else:
                    p = sbml_model.getParameter(var)
                    ic = p.getValue() if (p and p.isSetValue()) else 0.0
                sp_i = builder.add_species(_safe_name(var), ic)
                builder.set_species_continuous(sp_i)  # issue #718: not a count
                species_idx[var] = sp_i
                species_ids.append(var)
                builder.add_observable(_safe_name(var), [(sp_i, 1.0)])
                _register_ic_param_ref(
                    builder, sp_i, var, ia_single_param_ref, ia_expr_param, _declared_param_ids
                )

            expr = "0" if math is None else _ast_to_exprtk_with_funcdefs(math, func_defs)
            fname = f"_rhs_{_safe_name(var)}"
            builder.add_function(fname, expr)
            # GH #75: a rate rule on an hOSU=true species defines d(amount)/dt,
            # and the emitted law reads referenced species as amounts (the
            # same-named-observable shadow). Storage is amount/V_c, so the ODE
            # accumulation must divide by V_c(target). per_species_volume_scaling
            # divides the ydot accumulation by the product's volume_factor (=V_c)
            # while leaving the SSA propensity in amount/time (ssa_volume_factor
            # stays 1.0; the SSA fire step's 1/V_c divide completes it) — the
            # same shape as the §9 cross-compartment functional emission. Gated
            # on the hOSU target: a non-hOSU rate-rule target denotes a
            # concentration directly, and EVERY species carries volume_factor=V_c
            # (only amount_valued differs), so unconditional scaling would wrongly
            # divide a hOSU=false target. Byte-identical for hOSU=false / V_c=1.
            builder.add_reaction(
                [],
                [sp_i],
                "functional",
                fname,
                per_species_volume_scaling=species_hosu.get(var, False),
                # GH #81: mark this as a rate-rule ODE so the SSA/PSA loop
                # integrates the target deterministically (forward Euler)
                # rather than firing it as a stochastic birth/death channel.
                # The ODE path ignores this flag. A target that feeds a
                # reaction propensity (a species/parameter appearing in a
                # kinetic law) thus drives a time-varying-but-deterministic
                # rate, the same construct the variable-volume case needs.
                is_rate_rule_ode=True,
            )

    # ── 8b. Dilution term for hOSU=false species in rate-rule compartments ──
    # (#86) For a concentration-valued (hasOnlySubstanceUnits=false) species S of
    # amount A = [S]·V in a compartment whose volume V(t) is driven by a rate
    # rule, the concentration ODE is
    #
    #     d[S]/dt = d(A/V)/dt = (1/V)·dA/dt − [S]·(V̇/V)
    #               └ reaction term (§9 ``_vd_<rid>_varvol`` ÷V_live) ┘ └ dilution ┘
    #
    # §9 emits only the reaction term; the DILUTION term ``−[S]·V̇/V`` — the
    # concentration change caused by the volume itself moving — was never added,
    # so a species in a growing/shrinking compartment was integrated as if its
    # compartment were static (both [S] and the implied amount diverged from
    # RoadRunner). Emit it here as a Functional reaction producing S at rate
    # ``−1·S·V̇/V``, where V̇ is the compartment's rate-rule RHS and V is the
    # LIVE promoted-volume symbol. It is additive to the §9 reaction term and
    # also covers species with no reaction flux (pure dilution, ``[S]=A/V(t)``).
    #
    # Amount-valued (hOSU=true) species are untouched: their stored quantity is
    # the amount, whose ODE has no dilution term (#85 handles their report-time
    # concentration rescale). Rate-rule / assignment-rule target species are
    # excluded — their own rule fully defines the derivative. Gated on a
    # rate-rule compartment (NOT event-resized: those rescale at the event, #74),
    # so every static / V=1 / corpus model emits nothing here (byte-identical).
    if rate_rule_comps:
        comp_rate_rule_expr: dict[str, str] = {}
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.isRate() and rule.getVariable() in rate_rule_comps:
                rmath = rule.getMath()
                if rmath is not None:
                    comp_rate_rule_expr[rule.getVariable()] = _ast_to_exprtk_with_funcdefs(
                        rmath, func_defs
                    )
        for sid, comp in species_comp.items():
            if (
                comp in comp_rate_rule_expr
                and sid in species_idx
                and not species_hosu.get(sid, False)
                and sid not in rate_rule_targets
                and sid not in assignment_targets
            ):
                s_safe = _safe_name(sid)
                c_safe = _safe_name(comp)
                # −[S]·V̇/V. S resolves to the stored concentration (its
                # same-named observable; hOSU=false ⇒ not amount-valued), c_safe
                # to the live compartment volume, and the rate-rule RHS to V̇.
                dil_name = f"_dil_{s_safe}_{c_safe}"
                builder.add_function(
                    dil_name,
                    f"-1.0*({s_safe})*({comp_rate_rule_expr[comp]})/({c_safe})",
                )
                builder.add_reaction(
                    [],
                    [species_idx[sid]],
                    "functional",
                    dil_name,
                    apply_species_factor=False,
                    # (#81 Tier 2) ODE-only: the dilution term is excluded from
                    # SSA entirely. Under SSA the molecule count is conserved
                    # across the volume change — dilution lowers concentration,
                    # not molecule number — and the volume's effect on reaction
                    # rates is carried by the ssa_live_volume_* correction. Firing
                    # this negative-rate channel (GH #110 reverse-fire) would
                    # spuriously consume molecules; integrating it would shrink
                    # the count. ODE path is unaffected.
                    ode_only=True,
                )

    # ── 8c. Dilution term for hOSU=false species in ASSIGNMENT-RULE compartments
    #        whose volume is time-varying through a rate-ruled dependency (#234) ──
    # An assignment-rule compartment V := g(...) is NOT a rate-rule compartment, so
    # section 8b never emits its dilution term — yet V is just as time-varying when
    # g depends on a rate-ruled parameter (e.g. ``C := p1*p2`` with ``p2' = 0.1``,
    # suite 00310-00318) or on ``time()``. A concentration-valued (hOSU=false)
    # species there obeys the same concentration ODE as in 8b,
    #
    #     d[S]/dt = (1/V)·dA/dt − [S]·(V̇/V),
    #
    # and bngsim stores the concentration, so the dilution term ``−[S]·V̇/V`` must
    # be integrated or the implied amount drifts (00310: ``A = [S]·V`` grew like V
    # instead of decaying — max_err 0.667, exactly the dropped V̇/V at t=0).
    #
    # V̇ is the *time derivative of the assignment-rule RHS*, which 8b's rate-rule
    # RHS gives directly but an AR compartment does not — it needs the chain rule
    # V̇ = Σ_k (∂g/∂x_k)·ẋ_k over the time-varying quantities g references. We build
    # it symbolically (sympy), with ẋ_k for each leaf: ``time()`` (=1), a rate-ruled
    # non-species variable (its rate-rule RHS), another assignment target, species
    # or not (recursively), and any other species: its ``rate_of__`` accessor, the
    # engine's live dx/dt, which is SBML's rateOf of exactly the value the symbol
    # denotes (issue #1052: such a leaf used to make V̇ unknown, and the species
    # then kept its concentration where SBML keeps its amount, with nothing said).
    # Reading a rate turns the engine's rateOf machinery on (``_dilution_reads_rate``,
    # §10.6), which takes a probe of the derivatives before each RHS.
    #
    # Two leaves are refused by name rather than read. A concentration the size
    # reads in its own compartment, directly or through another such compartment:
    # the concentration is the amount over the size, so the two define each other
    # (libRoadRunner refuses it as a recursive assignment rule). And an amount
    # (hOSU=true) in an assignment-rule compartment, its own included, whose rate
    # accessor is still wrong there (#736). An event-promoted parameter holds
    # between its events (rate 0; the #74 rescale owns the jump), and so does a
    # species that is constant or a boundary with no rate rule, where its value
    # is an amount or its compartment's size holds; such a leaf needs no rate
    # accessor. An unparseable RHS still makes V̇ unknown, and we emit nothing,
    # so a static-RHS AR compartment (V̇ ≡ 0) and every non-AR model stay
    # byte-identical. hOSU=true species and rule-target species are excluded exactly
    # as in 8b. The emitted term is additive to §9's reaction term and also covers a
    # flux-free species (pure dilution ``[S] = A/V(t)``); ODE-only for the same #81
    # reason (SSA conserves molecule count across a volume change).
    #
    # hOSU=false species that received a dilution term — their stored concentration
    # tracks amount/V_live, so the report-time amount selector reads V_live from the
    # AR expression column (``_varvol_ar_amount_map`` below). Populated by 8c.
    ar_dilution_species: set[str] = set()
    _dilution_reads_rate = False  # a dilution term reads a rate_of__ accessor (#1052)
    # (compartment, trigger, size before, size after), all ExprTk (#1054).
    _size_jumps: list[tuple[str, str, str, str]] = []
    _jump_species: dict[str, list[str]] = {}  # compartment -> species it rescales
    if ar_comp_targets:
        import re as _re

        import sympy as _sp
        from sympy.functions.elementary.piecewise import ExprCondPair as _ExprCondPair

        from bngsim._codegen import _PY_KEYWORD_PARAM_NAMES, _alias_keyword_param
        from bngsim._jacobian import _TIME_SYM, _exprtk_to_sympy, sympy_to_exprtk

        # Python keywords only. This round trip prints back to *ExprTk* through
        # `sympy_to_exprtk`, not to C through `sp.ccode`, so a C reserved word is
        # not a hazard here at all — `_unalias` below reverses exactly this map,
        # and widening one without the other would leave `_BNG_KW_const` in the
        # emitted expression (GH #108; the rule lives in
        # `bngsim._codegen._alias_keyword_param`).
        def _alias(n: str) -> str:
            return _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n

        # ExprTk RHS strings (already in safe names, functionDefinitions inlined).
        # AR map covers ALL assignment targets so the chain rule can recurse through
        # an intermediate AR parameter; rate-rule map covers only non-species
        # targets (a rate-ruled var's stored value IS its value, so its rate-rule
        # RHS is exactly ẋ — for a species the amount/concentration split makes that
        # untrue, so species are bailed below rather than followed here).
        _sbml_species_ids = {
            sbml_model.getSpecies(k).getId() for k in range(sbml_model.getNumSpecies())
        }
        _ar_rhs_ex: dict[str, str] = {}  # safe(var) -> ExprTk
        _rr_rhs_ex: dict[str, str] = {}  # safe(var) -> ExprTk
        for j in range(sbml_model.getNumRules()):
            _rule = sbml_model.getRule(j)
            _v = _rule.getVariable()
            _m = _rule.getMath()
            if _m is None:
                continue
            if _rule.isAssignment() and _v in assignment_targets:
                _ar_rhs_ex[_safe_name(_v)] = _ast_to_exprtk_with_funcdefs(_m, func_defs)
            elif _rule.isRate() and _v not in _sbml_species_ids:
                _rr_rhs_ex[_safe_name(_v)] = _ast_to_exprtk_with_funcdefs(_m, func_defs)
        # A reaction's id in math is its kinetic law's value, so a size is
        # followed through the law as through an assignment rule: for its rate
        # here, and for its jumps below. Read as a constant, `C := 1 + 0.5*J0`
        # with J0 at k*A gave C no rate, and [X] in C was not diluted: [X](5)
        # was 0.677 where libRoadRunner gives 1.267.
        _law_math: dict[str, object] = {}
        for j in range(sbml_model.getNumReactions()):
            _rx = sbml_model.getReaction(j)
            if _rx.getId() not in referenced_reaction_ids:
                continue  # no rule or law reads its rate
            _lm = _law_math_in_model_scope(_rx)
            if _lm is not None:
                _law_math[_rx.getId()] = _lm
                _ar_rhs_ex[_safe_name(_rx.getId())] = _ast_to_exprtk_with_funcdefs(_lm, func_defs)

        # Classification in aliased-safe-name space (matches str(free_symbol)).
        _species_alias = {_alias(_safe_name(s)): s for s in _sbml_species_ids}  # alias -> id
        _rate_token_species = {_rateof_token(s): s for s in _sbml_species_ids}
        # A species whose value never moves continuously, so its rate is 0 and
        # reading it needs no rate accessor: nothing changes its amount (constant,
        # boundary, or in no reaction, with no rate rule), and the value is that
        # amount or a concentration in a compartment whose size holds between
        # events. `C := 1 + 0.1*S`, S in no reaction, read rateOf(S), which SSA
        # cannot run, for a rate of 0.
        _still_species = {
            s
            for s in _sbml_species_ids
            if s not in rate_rule_targets
            and s not in assignment_targets
            and (
                sbml_model.getSpecies(s).getConstant()
                or sbml_model.getSpecies(s).getBoundaryCondition()
                or s not in reaction_species
            )
            and (
                species_hosu.get(s, False)
                or species_comp.get(s) not in (ar_comp_targets | rate_rule_comps)
            )
        }
        _rr_alias = {_alias(s): s for s in _rr_rhs_ex}  # alias -> safe name
        _ar_alias = {_alias(s): s for s in _ar_rhs_ex}  # alias -> safe name
        _rr_sym_cache: dict[str, object] = {}

        def _rr_deriv(safe: str):
            if safe not in _rr_sym_cache:
                _rr_sym_cache[safe] = _exprtk_to_sympy(_rr_rhs_ex[safe])
            return _rr_sym_cache[safe]  # sympy expr, or None on parse failure

        # Each rule is parsed once and its rate derived once, however many
        # rules read it. The chain rule and the inlined size below both walk
        # the rules a size reads, and they used to start each rule afresh on
        # every path that reached it, so a diamond of rules (r_i := r_{i-1} +
        # r_{i-2}, each read by the next two) cost a parse and a derivation per
        # path — exponential in its depth: over 17 s to load at 22 rules, and
        # ×6.9 for every four more.
        _ar_sym_cache: dict[str, object] = {}

        def _ar_sym(safe: str):
            if safe not in _ar_sym_cache:
                _ar_sym_cache[safe] = _exprtk_to_sympy(_ar_rhs_ex[safe])
            return _ar_sym_cache[safe]  # sympy expr, or None on parse failure

        _on_path = object()  # a rule whose walk is under way: reaching it again is a cycle
        # safe(var) -> its rate, or None ⇒ unknown
        _rule_rate: dict[str, object] = {}
        # safe(var) -> the nonzero terms of its rate: (∂g/∂s, s, ṡ)
        _rule_terms: dict[str, list] = {}

        def _leaf_deriv(nm: str):
            # d(symbol)/dt: None ⇒ unknown (bail the whole V̇); 0 ⇒ constant.
            if nm == _TIME_SYM:
                return _sp.Integer(1)
            if nm in _ar_alias:  # an assignment target, a species among them
                return _vdot_sym(_ar_alias[nm])
            if nm in _species_alias:  # its live rate, rateOf(S) (#1052)
                sid = _species_alias[nm]
                if sid in _still_species:
                    return _sp.Integer(0)
                return _sp.Symbol(_rateof_token(sid))
            if nm in _rr_alias:
                return _rr_deriv(_rr_alias[nm])
            # An event-promoted parameter jumps at its events, which rescale the
            # concentrations there (#74), and holds between them (#1052: this
            # used to make V̇ unknown and drop the whole term).
            return _sp.Integer(0)

        def _vdot_sym(safe_var: str):
            # A rule on a cycle of rules, or one reading such a rule, has no
            # rate (None), as before: the walk that meets a rule still under way
            # unwinds with None through every rule on the cycle.
            if safe_var in _rule_rate:
                got = _rule_rate[safe_var]
                return None if got is _on_path else got
            if safe_var not in _ar_rhs_ex:
                return None
            _rule_rate[safe_var] = _on_path
            _rule_rate[safe_var], _rule_terms[safe_var] = _chain_rule(_ar_sym(safe_var))
            return _rule_rate[safe_var]

        def _chain_rule(g):
            # (Σ ∂g/∂s·ṡ over g's symbols, its nonzero terms), or (None, []).
            if g is None:
                return None, []
            total, terms = _sp.Integer(0), []
            for s in g.free_symbols:
                d = _leaf_deriv(str(s))
                if d is None:
                    return None, []
                if d != 0:
                    p = _sp.diff(g, s)
                    terms.append((p, s, d))
                    total = total + p * d
            return total, terms

        # Reverse the keyword aliasing _exprtk_to_sympy applies, so the emitted
        # term references the engine's real safe names.
        _all_safe = (
            {
                _safe_name(sbml_model.getParameter(k).getId())
                for k in range(sbml_model.getNumParameters())
            }
            | {_safe_name(s) for s in _sbml_species_ids}
            | {
                _safe_name(sbml_model.getCompartment(k).getId())
                for k in range(sbml_model.getNumCompartments())
            }
        )
        _kw_safe = [sn for sn in _all_safe if sn in _PY_KEYWORD_PARAM_NAMES]

        def _unalias(expr_str: str) -> str:
            for sn in _kw_safe:
                a = _alias_keyword_param(sn)
                if a != sn:
                    expr_str = _re.sub(rf"\b{_re.escape(a)}\b", sn, expr_str)
            return expr_str

        def _diluted(sid: str) -> bool:
            return (
                species_comp.get(sid) in ar_comp_targets
                and sid in species_idx
                and not species_hosu.get(sid, False)
                and sid not in rate_rule_targets
                and sid not in assignment_targets
            )

        def _subexpressions(expr) -> list:
            # Each distinct node of a sympy expression, once. A rate or size
            # built from shared rules' is a DAG, and its tree — what
            # `free_symbols` and `atoms` walk, and what printing spells out —
            # can be exponentially larger than the DAG.
            seen: set[int] = set()
            out, stack = [], [expr]
            while stack:
                e = stack.pop()
                if id(e) not in seen:
                    seen.add(id(e))
                    out.append(e)
                    stack.extend(e.args)
            return out

        def _tree_size(expr) -> int:
            # Nodes in the tree the expression prints as, a shared node counted
            # once per use, found without walking the tree.
            size: dict[int, int] = {}
            stack = [(expr, False)]
            while stack:
                e, expanded = stack.pop()
                if id(e) in size:
                    continue
                if expanded:
                    size[id(e)] = 1 + sum(size[id(a)] for a in e.args)
                else:
                    stack.append((e, True))
                    stack.extend((a, False) for a in e.args if id(a) not in size)
            return size[id(expr)]

        def _rate_tokens(expr) -> set[str]:
            return {
                _rate_token_species[str(s)]
                for s in _subexpressions(expr)
                if s.is_Symbol and str(s) in _rate_token_species
            }

        # V̇ of each compartment that holds a species to dilute, and the species
        # whose rates it reads.
        _comp_vdot: dict[str, object] = {}
        _comp_rated: dict[str, set[str]] = {}
        for sid, comp in species_comp.items():
            if comp not in _comp_vdot and _diluted(sid):
                vdot = _vdot_sym(_safe_name(comp))
                _comp_vdot[comp] = vdot
                _comp_rated[comp] = _rate_tokens(vdot) if vdot is not None else set()

        # Written out in full, a rate that more than one rule reads is written
        # once per path to it, and the term is as exponential as the walk was
        # (r_i := sin(r_{i-1}) + cos(r_{i-2}): 22,122 characters at 16 rules,
        # ×2.6 for every two more). So such a rate becomes a function of its own,
        # `_ddt_<rule>`, that the rules reading it read by name, wherever that is
        # shorter than writing V̇ out; elsewhere — a rate that is a number or a
        # symbol, a rule only one rule reads, a sum the chain rule collapses —
        # V̇ is written out exactly as before.
        _readers: dict[str, int] = {}
        _reached: set[str] = set()
        _walk = [_safe_name(c) for c, v in _comp_vdot.items() if v is not None]
        while _walk:
            r = _walk.pop()
            if r in _reached:
                continue
            _reached.add(r)
            for _, s, _ in _rule_terms[r]:
                child = _ar_alias.get(str(s))
                if child is not None:
                    _readers[child] = _readers.get(child, 0) + 1
                    _walk.append(child)
        _rate_fn = {
            r: f"_ddt_{r}"
            for r, n in sorted(_readers.items())
            if n > 1 and not _rule_rate[r].is_Atom
        }
        _named_rate: dict[str, object] = {}

        def _rate_by_name(safe_var: str):
            # The rule's rate with each shared rate it reads read by name.
            if safe_var not in _named_rate:
                total = _sp.Integer(0)
                for p, s, d in _rule_terms[safe_var]:
                    child = _ar_alias.get(str(s))
                    if child in _rate_fn:
                        d = _sp.Symbol(_rate_fn[child])
                    elif child is not None and not d.is_Atom:
                        d = _rate_by_name(child)
                    total = total + p * d
                _named_rate[safe_var] = total
            return _named_rate[safe_var]

        def _functions_read(expr) -> list[str]:
            # The `_ddt_` rules expr reads, directly or through each other,
            # each after the ones it reads.
            fn_rule = {fn: r for r, fn in _rate_fn.items()}
            order: list[str] = []
            done: set[str] = set()

            def visit(e):
                for s in _subexpressions(e):
                    r = fn_rule.get(str(s)) if s.is_Symbol else None
                    if r is not None and r not in done:
                        done.add(r)
                        visit(_rate_by_name(r))
                        order.append(r)

            visit(expr)
            return order

        # compartment -> (V̇ by name, the `_ddt_` rules it reads), where shorter
        _comp_by_name: dict[str, tuple[object, list[str]]] = {}
        if _rate_fn:
            for comp, vdot in _comp_vdot.items():
                if vdot is None or vdot == 0:
                    continue
                expr = _rate_by_name(_safe_name(comp))
                rules = _functions_read(expr)
                if _tree_size(expr) + sum(
                    _tree_size(_rate_by_name(r)) for r in rules
                ) < _tree_size(vdot):
                    _comp_by_name[comp] = (expr, rules)

        def _size_text(cid: str) -> str:
            for j in range(sbml_model.getNumRules()):
                r = sbml_model.getRule(j)
                if r.isAssignment() and r.getVariable() == cid and r.getMath() is not None:
                    return f"{cid} := {libsbml.formulaToL3String(r.getMath())}"
            return cid

        _refusals: list[str] = []

        # A piecewise size jumps where one of its conditions flips (#1054). The
        # chain rule above differentiates each piece, which is right between
        # the jumps, but a jump is a delta no ODE term carries, so it gets what
        # an event's resize gets (#74): each concentration × V_before/V_after,
        # at the condition's root. The size is inlined through the rules it
        # reads, and the relations in its conditions are set as they read on
        # either side of each root for the size there. Relations that compare
        # one difference (or a multiple of it) with 0 change at one root, as
        # `t < 2` and `t >= 2` do in a piecewise with no otherwise, so they are
        # set together. Set one at a time, the event of each read the other
        # after its flip: two rescales at the instant took [S] in
        # `piecewise(1, t < 2, 5, t >= 2)` to 0 or left it right, as the hash
        # seed ordered them, and with `t < 2` and `t > 2` the size between the
        # two, which no piece gives, was read as 0 and divided by. A condition
        # over constants and event targets alone moves only at an event, whose
        # own rescale owns that jump; one that reads an event target and the
        # state or time could flip at the event and then again by the jump
        # event, so it is refused.
        #
        # An event moves more than what it assigns: a concentration in a
        # compartment it resizes, directly or through the compartment's rule,
        # is rescaled there, and that resize's own rescale carries any jump of
        # a size that reads it (#1055). Left out, both rescaled at the event:
        # `cv := piecewise(2, Kw > 7, 1)` with Kw in a compartment the event
        # doubles took [Sn] to 20 for 10. And a species no reaction or rule
        # moves, in a compartment that holds its size between events, moves
        # only at an event, though it is not constant: `cv := piecewise(1, Kw >
        # 7, 2)` beside `at (t > 1): Kw = 2` was refused, where the event's
        # resize alone is right.
        _event_moved = {
            sbml_model.getEvent(k).getEventAssignment(j).getVariable()
            for k in range(sbml_model.getNumEvents())
            for j in range(sbml_model.getEvent(k).getNumEventAssignments())
        }
        _rescaled_in: dict[str, set[str]] = {}
        for _sid, _scomp in species_comp.items():
            if not species_hosu.get(_sid, False) and _sid not in assignment_targets:
                _rescaled_in.setdefault(_scomp, set()).add(_sid)
        # A rate a size reads is read through its law, as a rule is.
        _size_rule_math = {
            sbml_model.getRule(j).getVariable(): sbml_model.getRule(j).getMath()
            for j in range(sbml_model.getNumRules())
            if sbml_model.getRule(j).isAssignment() and sbml_model.getRule(j).getMath() is not None
        } | _law_math
        _rule_reach: dict[str, frozenset] = {}

        def _reaches(var: str) -> frozenset:
            if var not in _rule_reach:
                _rule_reach[var] = frozenset()  # a cycle of rules reads no further
                names: set[str] = set()
                for n in _ast_name_set(_size_rule_math[var]):
                    names.add(n)
                    if n in _size_rule_math:
                        names |= _reaches(n)
                _rule_reach[var] = frozenset(names)
            return _rule_reach[var]

        for _comp in _event_moved & _comp_id_set:
            _event_moved |= _rescaled_in.get(_comp, set())
        _grew = True
        while _grew:
            _grew = False
            for _comp in sorted(ar_comp_targets - _event_moved):
                if _reaches(_comp) & _event_moved:
                    _event_moved |= {_comp, *_rescaled_in.get(_comp, ())}
                    _grew = True
        _event_targets = {_alias(_safe_name(n)) for n in _event_moved}
        _flow_free = {
            s
            for s in _sbml_species_ids
            if s not in reaction_species
            and s not in rate_rule_targets
            and s not in assignment_targets
            and (
                species_hosu.get(s, False)
                or species_comp.get(s) not in (ar_comp_targets | rate_rule_comps)
            )
        }
        _moving = (
            {_TIME_SYM}
            | {
                a
                for a, s in _species_alias.items()
                if s not in _still_species and s not in _flow_free
            }
            | set(_rr_alias)
            | set(_rate_token_species)
        )

        # safe(var) -> the size written through the rules it reads, once a rule
        # (None ⇒ unknown, _on_path while under way, as for the rates)
        _inlined: dict[str, object] = {}

        def _inlined_size(safe_var: str):
            if safe_var in _inlined:
                got = _inlined[safe_var]
                return None if got is _on_path else got
            if safe_var not in _ar_rhs_ex:
                return None
            _inlined[safe_var] = _on_path
            _inlined[safe_var] = _inline_rules(_ar_sym(safe_var))
            return _inlined[safe_var]

        def _inline_rules(g):
            if g is None:
                return None
            subs = {}
            for s in g.free_symbols:
                if str(s) in _ar_alias:
                    inner = _inlined_size(_ar_alias[str(s)])
                    if inner is None:
                        return None
                    subs[s] = inner
            return _xreplace_once(g, subs)

        def _lattice(func, args):
            # And(*args) or Or(*args), put together as they are: the
            # constructor sorts its arguments by their size as trees
            # (`ordered`). An argument that decides it decides it, and one
            # that cannot is dropped, as there.
            kept: dict = {}
            for a in args:
                if a is func.zero:
                    return a
                if a is not func.identity:
                    kept.update(dict.fromkeys(a.args if a.func is func else (a,)))
            if len(kept) < 2:
                return next(iter(kept), func.identity)
            obj = _sp.Basic.__new__(func, *kept)
            obj._argset = frozenset(kept)
            return obj

        def _rebuilt(e, args):
            # e.func(*args), with a Piecewise and its pieces put together
            # without the constructors' walk of each condition (`replace`,
            # `has`): a condition over a diamond of rules (`r21 > 0.5`) is
            # exponentially longer as a tree than in memory (issue #1124, item
            # 6). As the constructor has them, a piece whose condition is false
            # is dropped, one whose condition is true ends it, and pieces in a
            # row that read alike are one, on either condition: `piecewise(r0,
            # S > 3, r1)` with r0 and r1 both S is S, and its S > 3 no root.
            # Except a condition that reads a piecewise: the pair's own
            # constructor folds it into relations of their own (`r > 5` under
            # `r := piecewise(S, S > 3, 2*S)` into S > 5 and 2*S > 5), each
            # continuous. Left as it was, `r > 5` jumped at S = 3 itself, and
            # its root fired there beside S > 3's: two rescales at one instant,
            # the one drawn last kept.
            if isinstance(e, (_sp.And, _sp.Or)):
                return _lattice(e.func, args)
            if isinstance(e, _ExprCondPair):
                if any(isinstance(x, _sp.Piecewise) for x in _subexpressions(args[1])):
                    return e.func(*args)
                return _sp.Basic.__new__(_ExprCondPair, *args)
            if isinstance(e, _sp.Piecewise):
                pieces = []
                for p in args:
                    if p.cond is _sp.false:
                        continue
                    last = pieces[-1].expr if pieces else None
                    if last is not None and (
                        last is p.expr or (hash(last) == hash(p.expr) and last == p.expr)
                    ):
                        p = _rebuilt(p, (last, _lattice(_sp.Or, (pieces[-1].cond, p.cond))))
                        pieces[-1] = p
                    else:
                        pieces.append(p)
                    if p.cond is _sp.true:
                        break
                if pieces and pieces[0].cond is _sp.true:
                    return pieces[0].expr
                if pieces:
                    return _sp.Basic.__new__(_sp.Piecewise, *pieces)
            return e.func(*args)

        def _xreplace_once(expr, rule: dict):
            # expr.xreplace(rule), each distinct node rebuilt once: sympy's own
            # walks the tree, which a size read through shared rules makes
            # exponentially larger than the expression.
            done: dict[int, object] = {}

            def go(e):
                if id(e) not in done:
                    if e in rule:
                        done[id(e)] = rule[e]
                    else:
                        args = tuple(go(a) for a in e.args)
                        changed = any(a is not b for a, b in zip(args, e.args, strict=True))
                        done[id(e)] = _rebuilt(e, args) if changed else e
                return done[id(e)]

            return go(expr)

        def _rules_by_name() -> dict:
            # A rule's size, written through the rules it reads, -> the rule's
            # own symbol, for each rule whose size holds no condition: it is the
            # same on either side of a jump, so the sizes there may read it by
            # name, as everything else in the model does.
            return {
                full: _sp.Symbol(_alias(safe))
                for safe, full in _inlined.items()
                if isinstance(full, _sp.Basic)  # not None, not under way
                and not full.is_Atom
                and not any(
                    isinstance(e, (_sp.Piecewise, _sp.core.relational.Relational))
                    for e in _subexpressions(full)
                )
            }

        def _names_in(expr) -> set[str]:
            # free_symbols, each distinct node visited once: sympy's walks the
            # tree, which a size read through shared rules makes exponential.
            return {str(e) for e in _subexpressions(expr) if e.is_Symbol}

        def _through_functions(exprs: list, prefix: str) -> tuple[list, list[tuple[str, object]]]:
            # ``exprs`` with each large node that more than one path reaches read
            # through a function of its own, and those functions, each after the
            # ones it reads, as (name, body) (issue #1124, item 6). A jump's
            # condition and sizes are written from the size through its rules,
            # and a diamond of rules under a condition, or a condition under a
            # diamond, was written out once per path: exponential in the depth.
            # Only where the texts would be long; only numbers, for a function
            # returns one, so a condition stays where it is read.
            size: dict[int, int] = {}
            refs: dict[int, int] = {}
            order: list = []
            seen: set[int] = set()
            for root in exprs:
                refs[id(root)] = refs.get(id(root), 0) + 1
                stack = [(root, False)]
                while stack:
                    e, expanded = stack.pop()
                    if expanded:
                        size[id(e)] = 1 + sum(size[id(a)] for a in e.args)
                        order.append(e)
                        continue
                    if id(e) in seen:
                        continue
                    seen.add(id(e))
                    for a in e.args:
                        refs[id(a)] = refs.get(id(a), 0) + 1
                    stack.append((e, True))
                    stack.extend((a, False) for a in e.args if id(a) not in seen)
            if sum(size[id(e)] for e in exprs) <= _JUMP_TEXT_NODES:
                return list(exprs), []
            helpers: list[tuple[str, object]] = []
            built: dict[int, object] = {}
            for e in order:  # each node after its arguments
                args = tuple(built[id(a)] for a in e.args)
                changed = any(a is not b for a, b in zip(args, e.args, strict=True))
                new = _rebuilt(e, args) if changed else e
                if (
                    refs.get(id(e), 0) > 1
                    and size[id(e)] >= 16
                    and isinstance(e, _sp.Expr)  # a number; a relation or an And is not one
                ):
                    name = f"{prefix}{len(helpers)}"
                    helpers.append((name, new))
                    new = _sp.Symbol(name)
                built[id(e)] = new
            return [built[id(e)] for e in exprs], helpers

        def _ratio_of(d, d0):
            # d / d0 where that is a nonzero number, else None. `cancel` walks
            # the tree, which a relation over a diamond of rules makes
            # exponential (issue #1124, item 6): `r17 > 0.5` beside `r17 <
            # -0.5`, or beside `S > 3`, took a second to load at 18 rules, x2.6
            # for every two more. So two long differences are first compared
            # term by term, as relations that share a root are written (`r21 >
            # 1.4` beside `2*r21 < 2.8`), then read at two points, to 30
            # digits, and only a pair whose ratio is the same at both is
            # cancelled; a read that is not a number, or raises, or a d0 near 0
            # there, leaves the pair to `cancel` too.
            if _tree_size(d) + _tree_size(d0) > _JUMP_TEXT_NODES:
                terms: list[dict] = [{}, {}]
                for x, out in zip((d, d0), terms, strict=True):
                    for t in _sp.Add.make_args(x):
                        k, m = t.as_coeff_Mul()
                        out[m] = out.get(m, 0) + k
                if terms[0].keys() == terms[1].keys() and all(terms[1].values()):
                    qs = [terms[0][m] / terms[1][m] for m in terms[1]]
                    if qs[0] != 0 and all(abs(q - qs[0]) <= 1e-12 * abs(qs[0]) for q in qs):
                        return qs[0]
                names = sorted(
                    {e for x in (d, d0) for e in _subexpressions(x) if e.is_Symbol}, key=str
                )
                seen = []
                for step in (0.6180339887, 0.4142135624):
                    at = {
                        s: _sp.Float(0.25 + ((k + 1) * step) % 1, 30) for k, s in enumerate(names)
                    }
                    try:
                        a, b = _xreplace_once(d, at), _xreplace_once(d0, at)
                    except Exception:  # Max of a complex number raises
                        break
                    if not all(v.is_Float or v.is_Rational for v in (a, b)) or abs(b) < 1e-8:
                        break
                    seen.append(a / b)
                if len(seen) == 2 and abs(seen[0] - seen[1]) > 1e-12 * max(1, abs(seen[0])):
                    return None
            ratio = _sp.cancel(d / d0)
            return ratio if ratio.is_number and ratio != 0 else None

        # compartment -> the concentrations its size's jump conditions read
        _cond_reads: dict[str, set[str]] = {}
        for comp, vdot in _comp_vdot.items():
            if vdot is None:
                continue
            g = _inlined_size(_safe_name(comp))
            if g is None:
                continue
            # In a fixed order, so the events do not move with the hash seed.
            conds = sorted(
                {
                    c
                    for pw in _subexpressions(g)
                    if isinstance(pw, _sp.Piecewise)
                    for _, c in pw.args
                    if c not in (_sp.true,)
                },
                key=_sp.default_sort_key,
            )
            relations: set = set()
            for c in conds:
                names = _names_in(c)
                if not names & _moving:
                    continue
                _cond_reads.setdefault(comp, set()).update(
                    _species_alias[n]
                    for n in names
                    if n in _species_alias and _diluted(_species_alias[n])
                )
                if names & _event_targets:
                    _refusals.append(
                        f"compartment {comp} ({_size_text(comp)}) jumps where "
                        f"{c} flips, which reads "
                        f"{', '.join(sorted(names & _event_targets))}, which an event "
                        "changes, and moves with the run as well; bngsim would rescale its "
                        "concentrations twice when an event flips it"
                    )
                    continue
                relations |= {
                    r
                    for r in _subexpressions(c)
                    if isinstance(r, _sp.core.relational.Relational) and _names_in(r) & _moving
                }
            # (difference, [(relation, its difference over that one)])
            roots: list[tuple[object, list]] = []
            for r in sorted(relations, key=_sp.default_sort_key):
                d = r.lhs - r.rhs
                for d0, rels in roots:
                    ratio = _ratio_of(d, d0)
                    if ratio is not None:
                        rels.append((r, ratio))
                        break
                else:
                    roots.append((d, [(r, _sp.Integer(1))]))
            by_name = _rules_by_name() if roots else {}

            # Roots that a run constant puts on one instant (#1124): `time < t1`
            # and `time < t2` with t1 = t2, or `S > a` and `S > b` with a = b. The
            # two jump events read the state from their trigger time, so the last
            # one's rescale is what stays, and each read the other's condition
            # as already flipped: [Sn] kept 2/3 of itself for 1/3. Each now
            # rescales by the whole instant, every coinciding condition read on
            # the side it leaves and the side it reaches, and a sibling whose
            # root is elsewhere is read as it stands, as before. A sibling's
            # difference is this one's times a number plus a run constant, which
            # is 0 where they coincide (`2*time < t2` sits on `time < t1` where t2
            # = 2·t1); read at the run, so a set_param moves it.
            def _moving_part(d):
                syms = [s for s in d.free_symbols if str(s) in _moving]
                return _sp.expand(d).as_independent(*syms, as_Add=True)[1]

            def _moving_terms(d):
                # d's terms that read the run, {term over its coefficient:
                # coefficient}, and the rest of d, a run constant
                moving: dict = {}
                rest = []
                for t in _sp.Add.make_args(d):
                    if _names_in(t) & _moving:
                        k, m = t.as_coeff_Mul()
                        moving[m] = moving.get(m, 0) + k
                    else:
                        rest.append(t)
                return moving, _sp.Add(*rest)

            def _sibling(d1, d0):
                # (scale, rest) where d1 = scale*d0 + rest, with scale a nonzero
                # number, else None. Past _JUMP_TEXT_NODES, `expand`, `cancel`
                # and `free_symbols` walk the tree, which a relation over a
                # diamond of rules makes exponential (issue #1124, item 6), so
                # the two are compared term by term, as `_ratio_of` does.
                if _tree_size(d0) + _tree_size(d1) > _JUMP_TEXT_NODES:
                    (m1, c1), (m0, c0) = _moving_terms(d1), _moving_terms(d0)
                    if not m0 or m1.keys() != m0.keys() or not all(m0.values()):
                        return None
                    qs = [m1[m] / m0[m] for m in m0]
                    if qs[0] == 0 or any(abs(q - qs[0]) > 1e-12 * abs(qs[0]) for q in qs):
                        return None
                    return qs[0], c1 - qs[0] * c0
                scale = _sp.cancel(_moving_part(d1) / _moving_part(d0))
                if not scale.is_number or scale == 0:
                    return None
                return scale, _sp.expand(d1 - scale * d0)

            def _siblings(d0, roots=roots) -> list[tuple[list, object, object]]:
                out = []
                for d1, rels1 in roots:
                    if d1 is d0:
                        continue
                    got = _sibling(d1, d0)
                    if (
                        got is not None
                        and not got[1].is_number
                        and not _names_in(got[1]) & _moving
                    ):
                        out.append((rels1, *got))
                return out

            for k_root, (d0, rels) in enumerate(roots):
                # A side is the sign of the difference there, and an inequality
                # in the set, where it has one, says which side is true: an
                # equality holds at the root alone, so it reads false on both.
                c, q = next(
                    ((r, q) for r, q in rels if not isinstance(r, (_sp.Eq, _sp.Ne))), rels[0]
                )
                if isinstance(c, _sp.Eq):
                    side_true, side_false = 0, 1
                elif isinstance(c, _sp.Ne):
                    side_true, side_false = 1, 0
                else:
                    side_true = -1 if c.func(-q, 0) else 1
                    side_false = -side_true
                siblings = _siblings(d0)

                def _at(side, siblings=siblings, rels=rels, g=g):
                    subs = {r: r.func(rq * side, 0) for r, rq in rels}
                    for rels1, scale, rest in siblings:
                        for r, rq in rels1:
                            # its side where rest = 0, else as it stands; as a
                            # Piecewise this costs sympy a canonicalisation per
                            # sibling, which a size of many roots pays N² times
                            subs[r] = (
                                _lattice(_sp.Or, (_sp.Eq(rest, 0), r))
                                if r.func(rq * scale * side, 0)
                                else _lattice(_sp.And, (_sp.Ne(rest, 0), r))
                            )
                    return _xreplace_once(g, subs)

                v_false, v_true = (_at(side) for side in (side_false, side_true))
                # Written out in full, a size that reads shared rules is as
                # exponential as the rates were; it reads those of its rules
                # that hold no condition by name where that is shorter.
                v_false, v_true = (
                    min((v, _xreplace_once(v, by_name)), key=_tree_size) for v in (v_false, v_true)
                )
                # And so does the condition (issue #1124, item 6): `r21 > 0.5`
                # under a diamond of rules was printed written out in full.
                c_read = min((c, _xreplace_once(c, by_name)), key=_tree_size)
                written, helpers = _through_functions(
                    [c_read, _sp.Not(c_read), v_false, v_true],
                    f"_szj_{_safe_name(comp)}_{k_root}_",
                )
                helper_texts = [(name, sympy_to_exprtk(body)) for name, body in helpers]
                texts = [sympy_to_exprtk(e) for e in written]
                # Under a name none of the model's parameters has: a function
                # of a parameter's name is bound to it (#266) and overwrote it.
                renamed = {name: _unique_param_name(name) for name, _ in helpers}
                renamed = {old: new for old, new in renamed.items() if new != old}
                if renamed and all(t is not None for _, t in helper_texts):
                    texts = [t and _replace_exprtk_symbols(t, renamed) for t in texts]
                    helper_texts = [
                        (renamed.get(name, name), _replace_exprtk_symbols(t, renamed))
                        for name, t in helper_texts
                    ]
                if any(t is None for t in texts) or any(t is None for _, t in helper_texts):
                    _refusals.append(
                        f"compartment {comp} ({_size_text(comp)}) jumps where {c_read} "
                        "flips, and bngsim cannot write the size on either side"
                    )
                    continue
                # A helper the two triggers read is read at every root; the
                # others only by the jump's assignments, when it executes, and
                # so are deferred: evaluated at every step, they cost more than
                # the model (issue #1186). Each reads only helpers before it.
                bodies = dict(helpers)
                in_trigger: set[str] = set()
                stack = [n for e in written[:2] for n in _names_in(e) if n in bodies]
                while stack:
                    n = stack.pop()
                    if n not in in_trigger:
                        in_trigger.add(n)
                        stack.extend(m for m in _names_in(bodies[n]) if m in bodies)
                deferred = {renamed.get(n, n) for n in bodies if n not in in_trigger}
                for name, text in helper_texts:
                    builder.add_function(name, _unalias(text), deferred=name in deferred)
                rises, falls, v_false, v_true = (_unalias(t) for t in texts)
                _jump_species[comp] = [
                    s for s, sc in species_comp.items() if sc == comp and _diluted(s)
                ]
                _size_jumps.append((comp, rises, v_false, v_true))
                _size_jumps.append((comp, falls, v_true, v_false))

        for comp, rated in _comp_rated.items():
            for leaf in sorted(rated):
                if species_hosu.get(leaf, False) and species_comp.get(leaf) in ar_comp_targets:
                    _refusals.append(
                        f"compartment {comp} ({_size_text(comp)}) reads {leaf}, an amount in "
                        f"compartment {species_comp[leaf]}, which an assignment rule sizes; "
                        "bngsim's rate of such a species is not yet right there (#736)"
                    )

        # A concentration the size reads in a compartment whose own size reads a
        # rate: each compartment's rate needs the next one's, so a ring of them
        # (a compartment reading its own concentration is one) has no first.
        def _reads_in(a: str, b: str, reads: dict[str, set[str]] | None = None) -> list[str]:
            # The diluted concentrations in compartment b that a's rate (or, over
            # `reads`, a's conditions) reads.
            source = _comp_rated if reads is None else reads
            return sorted(leaf for leaf in source[a] if _diluted(leaf) and species_comp[leaf] == b)

        def _first_ring(needs: dict[str, list[str]]) -> list[str]:
            # The first ring of compartments, each needing the next, as a path
            # that ends where it starts; [] where there is none.
            state: dict[str, int] = {}  # 1 on the walk, 2 finished
            for start in sorted(needs):
                if state.get(start):
                    continue
                path, walk = [start], [iter(needs[start])]
                state[start] = 1
                while walk:
                    nxt = next(walk[-1], None)
                    if nxt is None:
                        state[path.pop()] = 2
                        walk.pop()
                    elif state.get(nxt) == 1:
                        return path[path.index(nxt) :] + [nxt]
                    elif not state.get(nxt) and nxt in needs:
                        state[nxt] = 1
                        path.append(nxt)
                        walk.append(iter(needs[nxt]))
            return []

        _ring = _first_ring(
            {
                comp: sorted({species_comp[leaf] for leaf in rated if _diluted(leaf)})
                for comp, rated in _comp_rated.items()
                if rated
            }
        )
        if len(_ring) == 2:  # a compartment reading its own concentration
            c = _ring[0]
            _refusals.append(
                f"compartment {c} ({_size_text(c)}) reads {', '.join(_reads_in(c, c))}, a "
                f"concentration in {c} itself: a concentration is the amount over the size, "
                "so the two define each other, and bngsim does not solve that"
            )
        elif _ring:
            _refusals.append(
                "the sizes of compartments "
                + ", ".join(f"{c} ({_size_text(c)})" for c in _ring[:-1])
                + " each read a concentration in the next ("
                + "; ".join(
                    f"{a} reads {', '.join(_reads_in(a, b))} in {b}"
                    for a, b in zip(_ring, _ring[1:], strict=False)
                )
                + "), so the sizes define each other, and bngsim does not solve that"
            )
        # The same ring through the sizes' jump conditions alone (issue #1124,
        # item 3). Each size's condition reads the next one's concentration,
        # which that size's own jump moves: where a jump flips them, the sizes
        # after it may be read more than one way (all jumped, or none), each
        # consistent with the concentrations it leaves, and nothing in the
        # model says which. libRoadRunner refuses the model.
        # bngsim kept the sizes from before the instant, with no word of it.
        # (A ring with a link through a size's value as well is ordered by
        # that link, and runs.)
        _cond_ring = _first_ring(
            {
                comp: sorted({species_comp[leaf] for leaf in read} - {comp})
                for comp, read in _cond_reads.items()
                if {species_comp[leaf] for leaf in read} - {comp}
            }
        )

        if _cond_ring:
            _refusals.append(
                "the sizes of compartments "
                + ", ".join(f"{c} ({_size_text(c)})" for c in _cond_ring[:-1])
                + " each jump on a condition that reads a concentration in the next ("
                + "; ".join(
                    f"{a} reads {', '.join(_reads_in(a, b, _cond_reads))} in {b}"
                    for a, b in zip(_cond_ring, _cond_ring[1:], strict=False)
                )
                + "), so where a jump flips those conditions the sizes after it can be read "
                "more than one way, and bngsim does not choose"
            )
        if _refusals:
            from bngsim._exceptions import ModelError

            raise ModelError(
                "Model sizes a compartment by an assignment rule that bngsim cannot "
                "follow to its rate of change:\n"
                + "\n".join(f"  - {r}" for r in dict.fromkeys(_refusals))
                + "\nA concentration species in such a compartment changes with its "
                "size, so bngsim needs the size's rate; libRoadRunner refuses a size "
                "that reads its own concentration too (a recursive assignment rule)."
            )

        # compartment -> (V̇ as ExprTk, [(`_ddt_` function, its ExprTk)], whether
        # they read a rate), or None where one of them does not print
        _comp_vdot_ex: dict[str, tuple[str, list[tuple[str, str]], bool] | None] = {}
        _rate_fns_added: set[str] = set()
        for sid, comp in species_comp.items():
            if _diluted(sid):
                vdot = _comp_vdot[comp]
                if vdot is None or vdot == 0:
                    continue
                if comp not in _comp_vdot_ex:
                    expr, rules = _comp_by_name.get(comp, (vdot, []))
                    parts = [expr, *map(_rate_by_name, rules)]
                    texts = [sympy_to_exprtk(e) for e in parts]
                    _comp_vdot_ex[comp] = (
                        None
                        if any(t is None for t in texts)
                        else (
                            _unalias(texts[0]),
                            [
                                (_rate_fn[r], _unalias(t))
                                for r, t in zip(rules, texts[1:], strict=True)
                            ],
                            # By name, a rate the full V̇ cancels can still be read.
                            bool(_comp_rated[comp])
                            or (bool(rules) and any(_rate_tokens(e) for e in parts)),
                        )
                    )
                if _comp_vdot_ex[comp] is None:
                    continue
                vdot_ex, rate_fns, reads_rate = _comp_vdot_ex[comp]
                for fn, fn_ex in rate_fns:  # each after the ones it reads
                    if fn not in _rate_fns_added:
                        _rate_fns_added.add(fn)
                        builder.add_function(fn, fn_ex)
                s_safe = _safe_name(sid)
                c_safe = _safe_name(comp)
                dil_name = f"_dil_{s_safe}_{c_safe}"
                builder.add_function(
                    dil_name,
                    f"-1.0*({s_safe})*({vdot_ex})/({c_safe})",
                )
                builder.add_reaction(
                    [],
                    [species_idx[sid]],
                    "functional",
                    dil_name,
                    apply_species_factor=False,
                    ode_only=True,
                )
                # The dilution makes the stored concentration track amount/V_live,
                # so the bare-id amount selector must report conc·V_live(t) (read
                # from the AR compartment's expression column), not the stale
                # conc·V_static. Recorded for the report-time amount remap below.
                ar_dilution_species.add(sid)
                if reads_rate:
                    _dilution_reads_rate = True

    # ── 9. Reactions → BNGsim reaction emission ──────────────────────
    # Two emission strategies for non-mass-action kinetic laws:
    #
    # (A) Unified (preferred): emit one BNGsim Functional reaction per
    #     SBML reaction with full reactant/product multisets, marked
    #     apply_species_factor=False so the engine does NOT multiply the
    #     rate by the reactant count again — BNG's writeSBML emits
    #     kinetic laws like `_rateLaw * S` that already include the
    #     species factor explicitly. This is correct for both ODE and
    #     SSA, including non-trivial stoichiometry (`2P→P2`, `X→2X`,
    #     `→5X`). Requires integer stoichiometry and a uniform per-
    #     species V_s factor across the involved species.
    #
    # (B) Per-species fallback: emit one BNGsim reaction per affected
    #     species with `stat_factor=±net_coeff`. Mathematically equivalent
    #     to (A) under ODE; INCORRECT under SSA — a negative stat_factor
    #     produces a negative propensity that the SSA loop clamps to
    #     zero, so the consumption side of the reaction never fires.
    #     Reserved for the cross-compartment / non-integer cases that
    #     Phase 2 will replace.
    # Continuous slots never take a falling factorial (see
    # _ssa_falling_factorial_terms); built once, not per reaction.
    _ssa_ff_continuous = set(assignment_targets) | set(rate_rule_targets)
    _ar_moving_comps = ar_comp_targets & _ar_moving_vars
    # The SSA's stale-read walks below follow a reaction id that a rule or a law
    # reads into its kinetic law, as they follow a rule: the id is the law's
    # value, evaluated on the stored counts. `x' = J0`, J0 at k*S*C with C
    # doubled by an event, ran with no refusal, x(3) 72.3 for libRoadRunner's
    # 45.1, where `x' = S` is refused.
    _ssa_reads = dict(_ar_rules)
    for _j in range(sbml_model.getNumReactions()):
        _rx = sbml_model.getReaction(_j)
        _kl = _rx.getKineticLaw()
        if (
            _rx.getId() in referenced_reaction_ids
            and _kl is not None
            and _kl.getMath() is not None
        ):
            _ssa_reads[_rx.getId()] = _math_names(_kl.getMath())[0] - {
                lp.getId() for lp in _kinetic_law_local_params(_kl)
            }
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        rid = rxn.getId()

        # A compartment whose size an assignment rule sets moves with the run,
        # and the SSA does not follow it (#81 left it out of scope): a count is
        # stored over the load-time size, so a concentration the law reads is
        # stale by V_static/V(t), and the live divide on the law is never undone
        # by a live volume factor. `J: A => ; C*k*A` with `C := 1 + 0.5*time` ran
        # at 2x the ODE's A(2), and a mass-action `k*A` at 3x its A(4), silently.
        # (Its reported concentration is the count over the load-time size too,
        # #741.) A rule over constants holds its size, and an amount-valued
        # species is stored exactly and read as an amount; the size itself, read
        # by a law, is read live.
        _ar_conc, _vv_read = _ssa_moving_ar_comp_reads(
            rxn,
            _ssa_reads,
            _ar_moving_vars,
            _ar_moving_comps,
            species_comp,
            species_hosu,
            varvol_ssa_comps,
            _ssa_ff_continuous,
        )
        if _vv_read:
            # (g): `J: => A; k*S*D` with S in C, C resized by an event, read
            # [S] at C's load-time size and over-fired after the resize.
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="variable_compartment_read",
                    message=(
                        f"Reaction '{rid}' reads a concentration in a compartment "
                        f"whose size changes during the run ({', '.join(_vv_read)}), "
                        "from outside that compartment or through an assignment "
                        "rule. SSA corrects a propensity for a changing volume only "
                        "for the species of the compartment the reaction acts in, so "
                        "this one would be wrong. Use method='ode', or declare the "
                        'species hasOnlySubstanceUnits="true".'
                    ),
                    location=f"reaction:{rid}",
                )
            )
        if _ar_conc:
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="assignment_rule_compartment",
                    message=(
                        f"Reaction '{rid}' changes or reads a concentration in a "
                        f"compartment whose size an assignment rule changes "
                        f"({', '.join(_ar_conc)}). "
                        "SSA does not follow such a compartment's size, so this "
                        "reaction's propensity would be wrong. Use method='ode', "
                        "make the compartment's size constant or a rate rule, or "
                        'declare the species hasOnlySubstanceUnits="true".'
                    ),
                    location=f"reaction:{rid}",
                )
            )

        # conversionFactor (GH #232): the factor for this reaction. For a
        # uniform reaction it multiplies the emitted stat_factor so the rate —
        # hence each changed species' amount derivative and the SSA propensity —
        # scales by cf. ``_cf_mixed_i`` is True when the changed species carry
        # different factors; such reactions are scaled per-species at the unified
        # / per-species emission instead (``_cf_i`` is left 1.0 and unused there).
        if _has_conversion_factor:
            _cf_val, _cf_mixed_i = _reaction_conversion_factor(rxn)
            _cf_i = _cf_val if _cf_val is not None else 1.0
        else:
            _cf_i, _cf_mixed_i = 1.0, False

        # SBML reactions without a kineticLaw have no defined rate. Step 7
        # already skipped them (no Functional emission, no mass-action
        # classification). Refusing to load is the right call — silently
        # treating these as rate=0 would produce wrong dynamics that a user
        # could miss, and the existing fall-through emits a Functional
        # reaction whose `function_name=rid` references a function that was
        # never added (which fails validate() with a confusing
        # "reaction N (Functional) references unknown function 'rid'"
        # message). Reject loudly with an actionable message instead.
        if rxn.getKineticLaw() is None:
            raise RuntimeError(
                f"Reaction '{rid}' has no kineticLaw. bngsim cannot load "
                "models with rate-less reactions because treating them as "
                "rate=0 would silently produce wrong dynamics. This pattern "
                "typically comes from SBML extension packages (e.g. FBC) "
                "that bngsim does not implement; rewrite the SBML with an "
                "explicit kineticLaw on each reaction, or simulate with a "
                "different engine."
            )

        # Trivially-mass-action kinetic laws (#16) take the Elementary
        # path so analytical sensitivity codegen can chain-rule through
        # the single rate constant. Section 7 cached the classifier result.
        # Phase 7 reversible-split reactions emit two Elementary records
        # for one SBML reaction id; the cache value is a list with one
        # entry per emitted channel.
        if i in mass_action_rxns:
            for param_name, sf, reactants, products, comp_factors in mass_action_rxns[i]:
                # ssa_volume_factor = V_c (compartment volume) of the reaction's
                # involved species, which equals the classifier's uniform
                # _v_s_factor. SSA propensity = k·sf·∏(stored_i)·V_c; with the
                # hOSU-aware sf (which carries Π_{hOSU}V_c/V_c = V^(order−1) for
                # an all-hOSU single-V reaction) this nets to k·∏amount_i, the
                # correct amount/time propensity — for both hOSU=false (sf has
                # no numerator term) and hOSU=true. Look up V_c via the first
                # involved species; reactants[0] for consumption-bearing
                # reactions, products[0] for pure synthesis (e.g. `→ X`).
                _ref_idx0 = reactants[0] if reactants else products[0]
                _ref_comp = species_comp[species_ids[_ref_idx0]]
                _v_c = comp_volumes.get(_ref_comp, 1.0)
                _rxn_idx = builder.add_reaction(
                    reactants,
                    products,
                    "elementary",
                    param_name,
                    stat_factor=sf * _cf_i,  # GH #232 conversionFactor scale
                    ssa_volume_factor=_v_c,
                )
                # (#170) …and where that V_c came from, so the SSA propensity
                # reads the size live instead of the number this load baked.
                if _ref_comp in live_volume_param_comps:
                    builder.set_reaction_ssa_volume_param(_rxn_idx, comp_param_idx[_ref_comp])
                # (#81) SSA live-volume tag for a mass-action reaction in a
                # variable-volume (rate-rule / event-resize) compartment. The
                # propensity reads the live volume via the exact monomial
                # correction (V_static/V_live)^(n_h-p), where n_h = the hOSU=false
                # concentration-factor count and p = the power of the reaction's
                # compartment as an explicit law factor. bngsim's variable-volume
                # ODE is correct ONLY when the compartment cancels (p == 1, the
                # BNG ``compartment*k*A*B`` convention) — then sf is V-independent
                # and the post-resize concentration rescale gives the right amount
                # evolution (verified vs RoadRunner). For p != 1 (e.g. a bare
                # ``k*A*B`` concentration-rate law) the ODE itself bakes the static
                # volume and diverges from RoadRunner after a resize, so SSA must
                # refuse rather than run a value the ODE can't reproduce. With
                # p == 1 the exponent is n_h - 1: unimolecular (n_h=1) is a no-op,
                # bimolecular scales 1/V. Each species is stored as amount/V_static
                # and stale by V_static/V_live; an hOSU=true (amount) reference is
                # V-independent and not supported here. The compartment species
                # index is resolved after §10 (event compartments promote there),
                # so record a deferred fixup.
                if _ref_comp in varvol_ssa_comps:
                    _law_species = [species_ids[ri] for ri in reactants]
                    _p = comp_factors.get(_ref_comp, 0)
                    # Supported variable-volume mass-action shapes (p == 1, the
                    # cancelling BNG ``compartment*k*…`` convention, so sf is
                    # V-independent and the ODE matches RoadRunner):
                    #  (a) an all-hOSU=false single-compartment monomial with
                    #      ≥1 reactant — exponent n_h - 1 (unimolecular no-op,
                    #      bimolecular ∝1/V).
                    #  (b) (#144) zeroth-order synthesis (no reactants, n_h = 0)
                    #      whose products are all hOSU=false and in the reaction's
                    #      compartment — exponent -1, propensity ∝ V_live. The
                    #      product gate keeps this in the validated all-hOSU=false
                    #      single-compartment lane; cross-compartment / hOSU=true
                    #      synthesis stays refused (#144 cases 1/4).
                    #  (c) (#170) an all-hOSU=true single-compartment monomial
                    #      with NO surviving compartment power (p == 0) — e.g.
                    #      ``k*A`` with A hOSU=true in a rate-rule / event-resized
                    #      compartment. The engine reads each hOSU=true reactant
                    #      as its amount (molecule count), so the base propensity
                    #      is k·∏n_i with the static V powers in sf/
                    #      ssa_volume_factor cancelling — provably V-independent,
                    #      d(n_i)/dt carries no live-volume term. The correct
                    #      live-volume exponent is therefore 0 (run uncorrected).
                    #      This is the Elementary-path twin of #144 case 1's
                    #      Functional ``cell*k*H`` (which carries the compartment
                    #      as an explicit factor and reaches exponent 0 via the
                    #      numeric-V_static divide). Scoped tightly to the
                    #      no-compartment-factor, all-hOSU=true, single-storage-
                    #      compartment sub-case so it does NOT re-open the general
                    #      hOSU=true volume handling that #131 finding 4 routes to
                    #      the Functional path: a surviving compartment power, a
                    #      mixed/hOSU=false factor, or a cross-compartment shape
                    #      all stay refused.
                    _hosu_volume_independent = (
                        _p == 0
                        and bool(_law_species)
                        and all(
                            species_hosu.get(sid, False) and species_comp[sid] == _ref_comp
                            for sid in _law_species
                        )
                    )
                    if _law_species:
                        _supported = _hosu_volume_independent or (
                            _p == 1
                            and all(
                                not species_hosu.get(sid, False) and species_comp[sid] == _ref_comp
                                for sid in _law_species
                            )
                        )
                    else:
                        # (b), or (d): a zeroth-order synthesis of hOSU=true
                        # products with no compartment power, ``=> H; k``,
                        # whose propensity is k, V-independent as in (c).
                        _product_ids = [species_ids[pi] for pi in products]
                        _supported = bool(_product_ids) and (
                            (
                                _p == 1
                                and all(
                                    not species_hosu.get(sid, False)
                                    and species_comp[sid] == _ref_comp
                                    for sid in _product_ids
                                )
                            )
                            or (
                                _p == 0
                                and all(
                                    species_hosu.get(sid, False) and species_comp[sid] == _ref_comp
                                    for sid in _product_ids
                                )
                            )
                        )
                    if _supported:
                        # hOSU=false concentration-factor count: each such
                        # reactant is stored as amount/V_static and stale by
                        # V_static/V_live, so it drives the correction exponent
                        # n_h - p. hOSU=true (amount-valued) factors carry no
                        # staleness and are excluded — for the #170 all-hOSU=true
                        # case n_h = 0 and p = 0, giving exponent 0 (no
                        # correction); for the all-hOSU=false cases this equals
                        # len(reactants) (the validated #81/#144 correction).
                        n_h = sum(1 for sid in _law_species if not species_hosu.get(sid, False))
                        ssa_varvol_fixups.append((_rxn_idx, _ref_comp, float(n_h - _p)))
                    else:
                        ssa_issues.append(
                            SsaIssue(
                                severity="error",
                                code="varvol_non_mass_action",
                                message=(
                                    f"Reaction '{rid}' acts in variable-volume "
                                    f"compartment '{_ref_comp}' but is not a "
                                    "supported variable-volume mass-action "
                                    "reaction (the supported single-compartment "
                                    "shapes are: an all-hOSU=false monomial whose "
                                    "rate law carries the compartment as a factor "
                                    "exactly once — the BNG `compartment*k*A*B` "
                                    "convention; or a volume-independent "
                                    "all-hOSU=true monomial with no compartment "
                                    "factor, e.g. `k*A`; bngsim's variable-volume "
                                    "ODE — which SSA must match — is correct only "
                                    "then). Use method='ode', keep the compartment "
                                    "volume constant, or write the law as "
                                    "`compartment*<mass-action>`."
                                ),
                                location=f"reaction:{rid}",
                            )
                        )
            continue

        # Walk reactants/products: collect (a) net stoich for the fallback,
        # (b) reactant/product multisets for unified emission, (c) the set
        # of V_s factors involved (1 if hosu else V_comp), (d) whether any
        # stoichiometry is non-integer.
        net: Counter = Counter()
        reactant_mult: list[int] = []  # 0-based species indices, repeated
        product_mult: list[int] = []
        involved_vs: set[float] = set()
        non_integer = False
        non_integer_species: list[str] = []
        # (GH #237 Phase 2) (sid, sign, stoich_expr) per variable-stoichiometry
        # reference; sign is -1 for a reactant, +1 for a product. These are pulled
        # OUT of `net`/the multiset (no baked constant) and emitted below as
        # per-species Functional reactions whose rate carries the live coefficient.
        variable_stoich_terms: list[tuple[str, float, str]] = []

        for j in range(rxn.getNumReactants()):
            sr = rxn.getReactant(j)
            sid = sr.getSpecies()
            if sid not in species_idx:
                continue
            sp = sbml_model.getSpecies(sid)
            if sp.getBoundaryCondition():
                continue
            # (GH #237 Phase 2) Time-varying stoichiometry: keep the coefficient
            # symbolic and emit a dedicated per-species Functional reaction below
            # rather than baking it into `net`.
            _vexpr = _variable_stoich_expr(sr)
            if _vexpr is not None:
                variable_stoich_terms.append((sid, -1.0, _vexpr))
                continue
            coeff = _resolve_stoich(sr)
            if coeff is None:
                logger.warning(
                    f"Reaction '{rid}': cannot resolve reactant "
                    f"stoichiometry for '{sid}'; assuming 1"
                )
                coeff = 1.0
            net[sid] -= coeff
            if coeff != int(coeff):
                non_integer = True
                non_integer_species.append(sid)
            else:
                # Multiset emission (apply_species_factor=False ⇒ the engine uses
                # product_count − reactant_count as the net coefficient). A
                # <speciesReference> may carry a SIGNED stoichiometry, and the same
                # species may appear in several references on either side (SBML L3
                # §4.11.3 sums them). Route by sign: a positive reactant coeff adds
                # to the reactant multiset; a NEGATIVE reactant coeff behaves like a
                # product and adds to the product multiset (the old per-reference
                # ``[idx] * int(-1) == []`` silently dropped it — cases
                # 01422/01426/01427/01432/01433). Positive integer coeffs stay
                # byte-identical, so ordinary reactions and catalysts (a species on
                # both sides) are unchanged.
                _ic = int(coeff)
                if _ic > 0:
                    reactant_mult.extend([species_idx[sid]] * _ic)
                elif _ic < 0:
                    product_mult.extend([species_idx[sid]] * (-_ic))
            comp = species_comp[sid]
            # Storage is always concentration (amount/V_c), so the per-species
            # ODE accumulation is dStorage[s]/dt = (c-b)*law/V_c(s) for *every*
            # species regardless of hOSU — the amount-vs-concentration reading
            # of the law is handled by the engine (hOSU species read as amounts
            # via Species::amount_valued), not here. (Pre-fix this used 1.0 for
            # hOSU species, collapsing a cross-compartment reaction's V_s set to
            # {1.0} and skipping the divide — the BIOMD0000000019 bug.) V_c=1
            # leaves this unchanged.
            vs = comp_volumes.get(comp, 1.0)
            involved_vs.add(vs)

        for j in range(rxn.getNumProducts()):
            sr = rxn.getProduct(j)
            sid = sr.getSpecies()
            if sid not in species_idx:
                continue
            sp = sbml_model.getSpecies(sid)
            if sp.getBoundaryCondition():
                continue
            # (GH #237 Phase 2) Time-varying stoichiometry — see the reactant loop.
            _vexpr = _variable_stoich_expr(sr)
            if _vexpr is not None:
                variable_stoich_terms.append((sid, 1.0, _vexpr))
                continue
            coeff = _resolve_stoich(sr)
            if coeff is None:
                logger.warning(
                    f"Reaction '{rid}': cannot resolve product "
                    f"stoichiometry for '{sid}'; assuming 1"
                )
                coeff = 1.0
            net[sid] += coeff
            if coeff != int(coeff):
                non_integer = True
                non_integer_species.append(sid)
            else:
                # Sign-routed multiset emission — see the reactant loop. A positive
                # product coeff adds to the product multiset; a NEGATIVE product
                # coeff behaves like a reactant and adds to the reactant multiset.
                _ic = int(coeff)
                if _ic > 0:
                    product_mult.extend([species_idx[sid]] * _ic)
                elif _ic < 0:
                    reactant_mult.extend([species_idx[sid]] * (-_ic))
            comp = species_comp[sid]
            # Storage is always concentration (amount/V_c), so the per-species
            # ODE accumulation is dStorage[s]/dt = (c-b)*law/V_c(s) for *every*
            # species regardless of hOSU — the amount-vs-concentration reading
            # of the law is handled by the engine (hOSU species read as amounts
            # via Species::amount_valued), not here. (Pre-fix this used 1.0 for
            # hOSU species, collapsing a cross-compartment reaction's V_s set to
            # {1.0} and skipping the divide — the BIOMD0000000019 bug.) V_c=1
            # leaves this unchanged.
            vs = comp_volumes.get(comp, 1.0)
            involved_vs.add(vs)

        # SSA validation: non-integer stoichiometry routes to the
        # per-species fallback emission (line below), which is SSA-broken —
        # the consumption side gets clamped to zero propensity. Out of
        # scope for the SSA backend.
        if non_integer:
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="non_integer_stoichiometry",
                    message=(
                        f"Reaction '{rid}' has non-integer stoichiometry "
                        f"on {', '.join(sorted(set(non_integer_species)))}; "
                        "SSA requires integer ±1 fire deltas. Round to "
                        "integer stoichiometry, restructure the reaction, "
                        "or use method='ode'."
                    ),
                    location=f"reaction:{rid}",
                )
            )

        # (#81) SSA validation: a reaction that reaches the Functional emission
        # (i.e. did NOT classify as mass-action) but acts in a variable-volume
        # compartment is refused under SSA. The propensity reads stored
        # concentrations that are stale by V_static/V_live; a scalar correction
        # can undo that only for a monomial (mass-action) law — an arbitrary
        # Functional law's V-dependence is not a single power. Mass-action varvol
        # reactions take the Elementary branch above (tagged + corrected); this
        # catches the rest (non-mass-action, reversible-difference, hOSU-only).
        # (#144) EXCEPT a single-compartment monomial routed here only for a
        # variable-volume reason (case 1 hOSU=true law factor, case 2 p != 1):
        # the propensity correction IS exact (the §9 emission tags it below), so
        # admit it. A multi-compartment one (len > 1) is genuinely cross-
        # compartment (#144 case 4) and stays refused.
        _rxn_comps = {species_comp[sid] for sid in net}
        _varvol_ssa_ok = (i in ssa_varvol_functional and len(_rxn_comps) == 1) or (
            i in ssa_varvol_xcompartment  # (#144 case 4) cross-compartment monomial
        )
        if (_rxn_comps & varvol_ssa_comps) and not _varvol_ssa_ok:
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="varvol_non_mass_action",
                    message=(
                        f"Reaction '{rid}' acts in a variable-volume compartment "
                        f"({', '.join(sorted(_rxn_comps & varvol_ssa_comps))}) "
                        "with a non-mass-action kinetic law. The SSA live-volume "
                        "propensity correction is exact only for a mass-action "
                        "monomial, so this reaction is not supported under SSA. "
                        "Use method='ode', keep the compartment volume constant, "
                        "or express the reaction as mass-action."
                    ),
                    location=f"reaction:{rid}",
                )
            )

        # Rate-law function for the Functional emission: the raw ``rid`` law.
        # hOSU=true species references in it read amounts via their same-named
        # observables (Species::amount_valued), so no amount-substituted
        # variant is needed (GH #75).
        base_func = rid

        # (GH #237 Phase 2) Emit each time-varying-stoichiometry reference as its
        # own per-species Functional reaction. The species' amount derivative gets
        # ``sign · stoich_expr(t) · law / V_c`` — exactly the SBML extent law
        # ``stoich_{s,r}·v_r`` with the coefficient kept live instead of baked.
        # The coefficient is folded into the rate FUNCTION (``law · stoich_expr``)
        # rather than the constant ``stat_factor`` the static path uses, so the
        # engine reads it fresh every RHS evaluation — no kernel change. ``sign``
        # (±1, reactant/product) and the conversion factor ride the scalar
        # stat_factor; ``[] → [sp]`` keeps the sign in stat_factor (a reactant is a
        # negative-rate product), mirroring the non-integer per-species fallback.
        if variable_stoich_terms:
            for _k, (_vsid, _vsign, _vexpr) in enumerate(variable_stoich_terms):
                _vsp_i = species_idx[_vsid]
                _vcomp = species_comp[_vsid]
                _vvol = comp_volumes.get(_vcomp, 1.0)
                _vfunc_expr = f"({base_func})*({_vexpr})"
                if (
                    _vvol != 1.0
                    or _vcomp in live_symbol_divide_comps
                    or _vcomp in live_volume_param_comps
                ):
                    # Storage is amount/V_c; divide the law by V_c (V_static for an
                    # amount-valued species in a variable-volume compartment), as
                    # the non-integer per-species fallback does. A live-symbol
                    # compartment with V_static=1 still needs the divide after its
                    # volume changes — and (#170) so does a static V_c=1 whose size
                    # is a writable parameter, for the same reason. ÷1.0 is exact,
                    # so emitting it changes no number at the nominal point.
                    if _vcomp in vstatic_divide_comps and species_hosu.get(_vsid, False):
                        _vdivisor = repr(float(_vvol))
                    else:
                        _vdivisor = _safe_name(_vcomp)
                    _vfunc_expr = f"({_vfunc_expr})/{_vdivisor}"
                _vfunc_name = f"_vs_{rid}_{_safe_name(_vsid)}_{_k}"
                with contextlib.suppress(RuntimeError):
                    builder.add_function(_vfunc_name, _vfunc_expr)
                builder.add_reaction(
                    [],
                    [_vsp_i],
                    "functional",
                    _vfunc_name,
                    stat_factor=_vsign * species_cf.get(_vsid, 1.0),
                    apply_species_factor=False,
                )
            # SSA: a continuously-varying stoichiometric coefficient cannot be
            # expressed as integer ±1 fire deltas (the same reason non-integer
            # stoichiometry is refused). ODE TimeCourse only.
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="variable_stoichiometry",
                    message=(
                        f"Reaction '{rid}' has a time-varying stoichiometry "
                        f"(on {', '.join(sorted({s for s, _, _ in variable_stoich_terms}))}); "
                        "SSA requires integer ±1 fire deltas. Use method='ode'."
                    ),
                    location=f"reaction:{rid}",
                )
            )

        unified_ok = (not non_integer) and len(involved_vs) <= 1
        # (#144 case 4) A cross-compartment variable-volume monomial must take the
        # per-species branch — each species divides by its OWN live compartment
        # volume — never the single-rate unified shortcut, even when the load-time
        # volumes coincide (involved_vs collapses to one value). The unified
        # branch's single representative-compartment divide is the bug source.
        if i in ssa_varvol_xcompartment:
            unified_ok = False
        # The same holds for every cross-compartment reaction that changes an
        # hOSU=false species in a compartment whose size is integrated or reset
        # (a rate rule, an event), whatever its law: a reversible difference, a
        # saturating law, or a monomial on a reaction flagged reversible, which
        # SBML L2 does by default and Antimony's `->` does. The classifier
        # certifies only the irreversible monomial, because only that one has an
        # exact SSA propensity correction. The ODE row is `stoich·law/V_live` for
        # all of them. The single divide is by one representative compartment,
        # which is right only for the species in it, and the per-species divide
        # below is by the load-time size.
        #
        # A species the reaction leaves as it found it (a catalyst, net 0) has no
        # row to divide, and does not count: a reaction whose changed species all
        # sit in one compartment keeps the single divide it is right with.
        # Within one compartment too, where the reaction changes both a
        # concentration there and an amount: the concentration's row is
        # law/V_live and the amount's law/V_static, and no single divide is both.
        # `H => B; 0.3*H` divided H's by the live size: H(6) was 9.77 for 6.61.
        _changes_an_amount = any(net[_sid] != 0 and species_hosu.get(_sid, False) for _sid in net)
        _xc_varvol_species = (
            [
                _sid
                for _sid in net
                if net[_sid] != 0
                and species_comp[_sid] in varvol_ssa_comps
                and not species_hosu.get(_sid, False)
            ]
            if len(_rxn_comps) > 1 or _changes_an_amount
            else []
        )
        if _xc_varvol_species:
            unified_ok = False
        # (#745) The same holds for a concentration in a compartment an
        # assignment rule moves: its row is law/V_live, which the single divide
        # gives only to the species of the representative compartment.
        # `H => A; k*H` with H an amount and A a concentration in
        # `C := 3 + 0.5*time` divided H's row by the live size: [H](5) was 0.527
        # for 0.401. Two such compartments of one load-time size shared the
        # first one's divide, and at a size of 1 had none. A catalyst elsewhere
        # counts here, as the representative compartment can be the catalyst's.
        # The per-species branch below adds the live size to these rows.
        _xc_ar_species = (
            [
                _sid
                for _sid in net
                if net[_sid] != 0
                and species_comp[_sid] in ar_comp_moving
                and not species_hosu.get(_sid, False)
            ]
            if len(_rxn_comps) > 1 or _changes_an_amount
            else []
        )
        if _xc_ar_species:
            unified_ok = False
        # Two kinds of such a reaction cannot take the per-species divide either.
        # One that also changes a species in an assignment-rule compartment: that
        # compartment is no state, so there is nothing to point its divide at
        # (#745). One that mixes conversion factors, which that emission refuses.
        # Both are written out one species at a time instead, each over its own
        # compartment's live symbol and with its own factor, as a non-integer
        # stoichiometry is. That form has no SSA reading, and these are refused
        # under SSA already. The certified monomial keeps the per-species divide,
        # which its SSA correction is written for. A reaction routed off the
        # single divide for a moving assignment-rule compartment alone goes the
        # same way when it mixes conversion factors.
        _xc_by_function = (
            bool(_xc_varvol_species)
            and i not in ssa_varvol_xcompartment
            and (
                _cf_mixed_i
                or any(net[_sid] != 0 and species_comp[_sid] in ar_comp_targets for _sid in net)
            )
        ) or (bool(_xc_ar_species) and _cf_mixed_i)
        # (#192) The same by-value hazard for a WRITABLE STATIC volume, which is
        # #170's territory rather than #144's and was never done. `involved_vs`
        # holds volume VALUES, so a reaction whose species span several
        # compartments that merely share a load-time size passes the gate and
        # takes the single representative-compartment divide below. Write one of
        # those sizes and the representative stops being the right divisor for
        # the other compartments' species — but the shortcut was baked in at
        # load, so `set_param(C, v)` builds a structurally DIFFERENT model than
        # `from_sbml(..., compartment_sizes={C: v})`, which sees unequal volumes
        # and takes the per-species branch. Route it there up front so the two
        # builds agree at every size. Free at the nominal point: the per-species
        # divisors all equal the representative's there.
        # Guarded on `unified_ok` so this only speaks where it is the deciding
        # vote: the two branches below that a not-yet-unified reaction already
        # takes are both per-species (the Phase 2.5 emission and the non-integer
        # fallback), so there would be nothing to re-route and nothing to refuse.
        _xc_live = _rxn_comps & live_volume_param_comps
        if unified_ok and len(_rxn_comps) > 1 and _xc_live and (reactant_mult or product_mult):
            if _cf_mixed_i:
                # The Phase 2.5 per-species emission refuses a reaction whose
                # changed species carry DIFFERENT conversionFactors (GH #232), so
                # there is nowhere to re-route this one to — its rebuild at an
                # unequal size is not a model bngsim can build. Refuse the write
                # rather than ship a representative divide no rebuild would make.
                # (The non-integer fallback further down needs no such refusal: it
                # emits one reaction per species with that species' own volume
                # divide and its own factor, so it is already per-species.)
                compartment_write_refused.update(_xc_live)
            else:
                unified_ok = False

        if unified_ok:
            if not reactant_mult and not product_mult:
                # All non-boundary species canceled or absent — no effect.
                continue
            common_vs = next(iter(involved_vs)) if involved_vs else 1.0
            if common_vs == 1.0:
                # (#74) If this reaction's species live in a single
                # variable-volume compartment (event-, rate-, or assignment-rule
                # resized),
                # divide the law by the LIVE compartment symbol so the
                # derivative is d[conc]/dt = law / V_live. At load V==1 this is
                # a no-op (÷1), so static / V=1 models stay byte-identical; it
                # only bites once an event mutates the volume.
                rxn_comps = {species_comp[s] for s in net}
                if len(rxn_comps) == 1 and (rxn_comps & live_symbol_divide_comps):
                    rep_comp = next(iter(rxn_comps))
                    # (#114) For an hOSU=true (amount-valued) species in a
                    # rate-rule (continuously variable-volume) compartment the
                    # storage divide must use the load-time numeric V_static,
                    # not the live symbol — see vstatic_divide_comps. Dividing
                    # the amount-rate by V_live(t) throttles the flux by
                    # V_static/V_live(t) as V grows (the MODEL1904020001 INTF
                    # bug). The live-symbol divide is kept for hOSU=false species
                    # (#86 dilution) and for event-resize compartments (#74).
                    if rep_comp in vstatic_divide_comps and all(
                        species_hosu.get(s, False) for s in net if net[s] != 0
                    ):
                        divisor = repr(float(comp_volumes.get(rep_comp, 1.0)))
                    else:
                        divisor = _safe_name(rep_comp)
                    vf_name = f"_vd_{rid}_varvol"
                    with contextlib.suppress(RuntimeError):
                        builder.add_function(vf_name, f"{base_func}/{divisor}")
                    use_func = vf_name
                else:
                    if len(rxn_comps) > 1 and (rxn_comps & live_symbol_divide_comps):
                        logger.warning(
                            "Reaction '%s' spans multiple compartments, one of "
                            "which (%s) changes size at runtime. bngsim divides "
                            "a Functional rate by a single representative "
                            "compartment volume, so the post-resize derivative "
                            "for cross-compartment species may be off by the "
                            "volume ratio. Single-compartment resize events are "
                            "handled correctly.",
                            rid,
                            ", ".join(sorted(rxn_comps & live_symbol_divide_comps)),
                        )
                    # (#170) The V=1 normalisation is what made issue #170's rows 6
                    # and 7 — the same law, the same model — disagree: loaded at
                    # V≠1 the `else` branch below emits `law/C` against the live
                    # symbol and a write to C reproduces a rebuild exactly, while
                    # loaded at V=1 the divide was dropped as a ÷1 no-op and the
                    # same write did nothing. Emit it here too when the storage
                    # compartment is a static PARAMETER, so the two builds of one
                    # model behave the same. Numerically free at the nominal point:
                    # `x/1.0 == x` exactly for every finite x, so the trajectory is
                    # bit-identical until C is written.
                    #
                    # The representative compartment is chosen by the SAME rule the
                    # `else` branch below uses, and deliberately without restricting
                    # this to single-compartment reactions: a reaction whose species
                    # span several compartments of equal load-time size takes this
                    # unified path too, and at V≠1 it gets a representative divide.
                    # Requiring len(rxn_comps) == 1 here (stage 1 did) left those
                    # reactions undivided at V=1 only — 7 corpus models, 68
                    # reactions, where a write then moved the trajectory to
                    # something no rebuild produces (BIOMD0000000570: 1.8 relative
                    # in the RHS).
                    #
                    # (#192) Stage 2 added here that "whether a single representative
                    # divide is the right reading of such a reaction is a separate
                    # question that the rebuild answers the same way". It does not:
                    # the rebuild at a DIFFERENT size sees unequal volumes and takes
                    # the per-species branch, which is why a multi-compartment
                    # reaction now only reaches this line when none of its
                    # compartments is writable (the gate above re-routes the rest).
                    # For those, the two builds of one model do agree — nothing can
                    # move the sizes apart.
                    rep_comp = species_comp[next(iter(net))]
                    _rep_is_vstatic = rep_comp in vstatic_divide_comps and all(
                        species_hosu.get(s, False) for s in net if net[s] != 0
                    )
                    if not _rep_is_vstatic and rep_comp in live_volume_param_comps:
                        vf_name = f"_vd_{rid}_unified"
                        with contextlib.suppress(RuntimeError):
                            builder.add_function(vf_name, f"{base_func}/{_safe_name(rep_comp)}")
                        use_func = vf_name
                    else:
                        use_func = base_func
            else:
                # Single shared compartment; divide rate by its volume once.
                rep_comp = species_comp[next(iter(net))]
                # (#87, #114) For amount-valued species in a variable-volume
                # compartment — assignment-rule (V(t) via e.g. ``tV := mV + dV``,
                # #87) or rate-rule (continuous V(t), #114) — storage is
                # ``amount/V_static`` so the storage divide must use the load-time
                # numeric V_static, NOT the live symbol. Dividing the amount-rate
                # by V_live(t) would throttle every reaction by V_static/V_live(t)
                # as the compartment grows, silently suppressing the dynamics.
                # Byte-identical for static compartments (the symbol value equals
                # the numeric); only AR/rate-rule-compartment reactions change.
                if rep_comp in vstatic_divide_comps and all(
                    species_hosu.get(s, False) for s in net if net[s] != 0
                ):
                    divisor = repr(float(comp_volumes.get(rep_comp, 1.0)))
                else:
                    divisor = _safe_name(rep_comp)
                vf_name = f"_vd_{rid}_unified"
                with contextlib.suppress(RuntimeError):
                    builder.add_function(vf_name, f"{base_func}/{divisor}")
                use_func = vf_name
            # ssa_volume_factor=common_vs converts the ODE-units rate
            # (storage/time) to amount/time propensity for SSA. For all-
            # hOSU=false single-compartment reactions, common_vs == V_c, the
            # right value. For a single-compartment hOSU=true V≠1 reaction
            # common_vs == V_c too (the hOSU→1.0 collapse was removed), and the
            # law reads amounts (engine amount_valued), so use_func = law/V_c
            # and the SSA propensity = (law(amount)/V_c)·V_c = law(amount), the
            # correct amount/time rate (GH #75 — was the gated Phase 2.7 case).
            # GH #232: partition the reactant/product multiset by each species'
            # conversion factor and emit one Functional reaction per cf-group,
            # sharing the rate ``use_func`` but carrying that group's factor as
            # its stat_factor. d(amount_s)/dt = cf_s · net_stoich_s · rate for
            # every changed species, including reactions whose species carry
            # DIFFERENT factors. Uniform-cf (and every cf-free model) collapses to
            # a single group ⇒ one emission, byte-identical to the prior behavior.
            _cf_groups: dict[float, tuple[list, list]] = {}
            for _idx in reactant_mult:
                _cf_groups.setdefault(species_cf.get(species_ids[_idx], 1.0), ([], []))[0].append(
                    _idx
                )
            for _idx in product_mult:
                _cf_groups.setdefault(species_cf.get(species_ids[_idx], 1.0), ([], []))[1].append(
                    _idx
                )
            if not _cf_groups:  # no changed species (e.g. all-modifier) — keep one emission
                _cf_groups[_cf_i] = (reactant_mult, product_mult)
            _func_rxn_idx = None
            _ssa_ff = _ssa_falling_factorial_terms(rxn, species_idx, _ssa_ff_continuous)
            for _cf_v, (_rm, _pm) in _cf_groups.items():
                _func_rxn_idx = builder.add_reaction(
                    _rm,
                    _pm,
                    "functional",
                    use_func,
                    stat_factor=_cf_v,
                    apply_species_factor=False,
                    ssa_volume_factor=common_vs,
                )
                if _ssa_ff:
                    builder.set_reaction_ssa_falling_factorial(_func_rxn_idx, _ssa_ff)
                # (#170) `common_vs` is the storage compartment's size; bind it to
                # that parameter so a write moves the propensity too. Only for a
                # single-compartment reaction — a mixed-V unified emission has no
                # one parameter this scalar belongs to (and those take the
                # per-species branch below, where the factor is 1.0 anyway).
                _uni_comps = {species_comp[s] for s in net}
                if len(_uni_comps) == 1 and next(iter(_uni_comps)) in live_volume_param_comps:
                    builder.set_reaction_ssa_volume_param(
                        _func_rxn_idx, comp_param_idx[next(iter(_uni_comps))]
                    )
            # (#144) Tag the SSA live-volume correction for a single-compartment
            # monomial routed here for a variable-volume reason (case 1 / case 2).
            # The propensity = (base_func / divisor) · ssa_volume_factor(=V_static).
            # Working the algebra for a monomial with `n_f` hOSU=false law factors:
            #   • divisor = live compartment symbol (NOT all net species hOSU=true)
            #     → propensity is off by (V_static/V_live)^(n_f-1); correct exp = n_f-1.
            #   • divisor = numeric V_static (all net species hOSU=true) → already
            #     exact (the live volume enters only via base_func), exp = 0.
            # Independent of the compartment power `p`. Guard on single-compartment
            # (the divide above is one factor); cross-compartment stays refused.
            _fix_comps = {species_comp[s] for s in net}
            if i in ssa_varvol_functional and len(_fix_comps) == 1:
                _rep_comp = next(iter(_fix_comps))
                _n_f = ssa_varvol_functional[i]
                # A concentration the reaction only reads (a catalyst) is stale
                # too, under either divide: with the numeric V_static divide the
                # exponent is n_f, with the live-symbol divide n_f - 1.
                if _rep_comp in vstatic_divide_comps and all(
                    species_hosu.get(s, False) for s in net if net[s] != 0
                ):
                    _vv_exp = float(_n_f)
                else:
                    _vv_exp = float(_n_f - 1)
                ssa_varvol_fixups.append((_func_rxn_idx, _rep_comp, _vv_exp))
            continue

        # Cross-compartment / mixed-V_s unified emission (Phase 2.5). Use the
        # raw ``rid`` law (hOSU species read as amounts via the engine
        # amount_valued path) with no /V_c wrapper; compute_derivs divides by
        # each affected species's volume_factor at accumulation time. The
        # function evaluates to amount/time, so SSA propensity is correct with
        # ssa_volume_factor=1.0; the SSA fire step's per-species
        # 1/volume_factor divide already handles per-species amount→storage
        # scaling.
        if not non_integer and not _xc_by_function:
            # GH #232: a cross-compartment reaction whose species carry DIFFERENT
            # conversion factors would need a per-cf-group split interleaved with
            # the per-species volume scaling below — not yet implemented. Refuse
            # loudly rather than silently drop the factor (uniform cf is fine —
            # _cf_i scales the whole rate). Single-compartment conversion-factor
            # models, the common case, never reach here.
            if _cf_mixed_i:
                raise RuntimeError(
                    f"Reaction '{rid}' mixes per-species SBML conversionFactors "
                    "across compartments. bngsim supports differing conversion "
                    "factors only within a single compartment; rewrite as "
                    "single-compartment, use a uniform factor, or use another "
                    "engine."
                )
            _xrxn_idx = builder.add_reaction(
                reactant_mult,
                product_mult,
                "functional",
                base_func,
                stat_factor=_cf_i,  # GH #232 conversionFactor scale (1.0 if none)
                apply_species_factor=False,
                ssa_volume_factor=1.0,
                per_species_volume_scaling=True,
            )
            _ssa_ff = _ssa_falling_factorial_terms(rxn, species_idx, _ssa_ff_continuous)
            if _ssa_ff:
                builder.set_reaction_ssa_falling_factorial(_xrxn_idx, _ssa_ff)
            # (#745) The per-species divide above is by each species' load-time
            # size. For a concentration in a compartment an assignment rule sizes
            # the row is law/V_live; the engine has no state column for that size,
            # so the difference is added as its own reaction on that species, as
            # the §8c dilution is: net·law·(1/C - 1/V_static), with C the
            # compartment's live symbol. The two rows sum to net·law/C, and the
            # Jacobian and the sensitivities reach C through its rule as they do
            # in any rate law. ODE only, as the SSA does not run such a
            # compartment. BIOMD0000000342's re9 was 37% off without it. Only for
            # a size that can move (_xc_ar_species keeps to ar_comp_moving): for
            # one that cannot, the term is 0, and a reaction on one species would
            # hide a conservation law the right-hand side keeps.
            for _sid in _xc_ar_species:
                _n = net[_sid]
                _c = species_comp[_sid]
                _corr = f"_vd_{_safe_name(rid)}_{_safe_name(_sid)}_live"
                builder.add_function(
                    _corr,
                    f"({float(_n) * _cf_i!r})*({base_func})"
                    f"*(1.0/({_safe_name(_c)}) - 1.0/({float(comp_volumes.get(_c, 1.0))!r}))",
                )
                builder.add_reaction(
                    [],
                    [species_idx[_sid]],
                    "functional",
                    _corr,
                    apply_species_factor=False,
                    ode_only=True,
                )
            # (#144 case 4) Cross-compartment variable-volume monomial certified by
            # the classifier (§7). The per-species emission above divides each
            # species's storage derivative by its *static* volume_factor, which is
            # wrong for an hOSU=false species in a variable-volume compartment
            # (storage tracks amount/V_live). Record two deferred fixups, resolved
            # once §10 has promoted every compartment to a species:
            #   • ODE: set each varvol hOSU=false species's ode_live_volume_idx0 so
            #     compute_derivs divides it by V_live instead of V_static.
            #   • SSA: add one live-volume term per varvol compartment c — exponent
            #     m_c = the hOSU=false reactant law-factor count in c — so the
            #     propensity is multiplied by ∏_c (V_c,static / V_c,live)^m_c. (The
            #     bare-law emission uses ssa_volume_factor=1.0 and no /V_live in the
            #     function, so exp = m_c with no −1, unlike the single-compartment
            #     scalar's n_f − 1.)
            # The ODE divide does not depend on the law, so it is recorded for
            # every such reaction, certified or not. The SSA term is exact only
            # for the certified monomial; the rest stay refused under SSA
            # (varvol_non_mass_action, above).
            for _sid in _xc_varvol_species:
                ode_xcomp_species_fixups.append((species_idx[_sid], species_comp[_sid]))
            if i in ssa_varvol_functional and len(_rxn_comps) == 1:
                # A one-compartment monomial certified for the scalar correction
                # and written out per species because it changes an amount and a
                # concentration together. This emission has no /V_live in the
                # function, so its exponent is the hOSU=false law-factor count
                # n_f, not n_f - 1.
                _one = next(iter(_rxn_comps))
                ssa_xcomp_term_fixups.append(
                    (
                        _xrxn_idx,
                        _one,
                        float(comp_volumes.get(_one, 1.0)),
                        float(ssa_varvol_functional[i]),
                    )
                )
            if i in ssa_varvol_xcompartment:
                _varvol_comps = set(ssa_varvol_xcompartment[i])
                for _sid in net:
                    if (
                        _sid not in _xc_varvol_species
                        and species_comp[_sid] in _varvol_comps
                        and not species_hosu.get(_sid, False)
                    ):
                        ode_xcomp_species_fixups.append((species_idx[_sid], species_comp[_sid]))
                _comp_exp: Counter = Counter()
                for _ridx in reactant_mult:  # 0-based species idx, repeated by mult
                    _rsid = species_ids[_ridx]
                    _rc = species_comp[_rsid]
                    if _rc in _varvol_comps and not species_hosu.get(_rsid, False):
                        _comp_exp[_rc] += 1
                for _c, _m in _comp_exp.items():
                    ssa_xcomp_term_fixups.append(
                        (_xrxn_idx, _c, float(comp_volumes.get(_c, 1.0)), float(_m))
                    )
            continue

        # Non-integer stoichiometry fallback. Mathematically equivalent to
        # the unified branch under ODE; SSA-broken because consumption-side
        # reactions get clamped to zero propensity (negative stat_factor).
        # Not exercised by DSMTS or the SSA roundtrip corpus; Phase 3's
        # validate_for_ssa() will reject non-integer stoichiometry under SSA
        # with a clear error.
        for sid, coeff in net.items():
            if coeff == 0.0:
                continue
            sp_i = species_idx[sid]
            comp = species_comp[sid]
            vol = comp_volumes.get(comp, 1.0)

            # Storage is amount/V_c for every species, so divide the law by
            # V_c whenever the compartment is non-unit or can vary at runtime
            # (hOSU-independent — the amount reading of the law is already in
            # base_func). hOSU=false keeps base_func == rid, so static V=1
            # remains byte-identical.
            if vol != 1.0 or comp in live_symbol_divide_comps or comp in live_volume_param_comps:
                # (#87, #114) For an amount-valued species in a variable-volume
                # (assignment-rule or rate-rule) compartment, the storage divide
                # is V_static (the load-time numeric), NOT the live symbol — see
                # section 9's ``else`` branch. Byte-identical for static comps.
                # (#170) A static V=1 compartment that is a writable PARAMETER
                # also gets the divide, so the write reaches this path; ÷1.0 is
                # exact, so no number moves at the nominal point.
                # One function per divisor. A reaction that changes a
                # concentration and an amount-valued species in the same
                # changing compartment needs both, and under one name the
                # second registration was dropped: both rows then divided by
                # whichever came first.
                if comp in vstatic_divide_comps and species_hosu.get(sid, False):
                    divisor = repr(float(vol))
                    vf_name = f"_vds_{rid}_{comp}"
                else:
                    divisor = _safe_name(comp)
                    vf_name = f"_vd_{rid}_{comp}"
                with contextlib.suppress(RuntimeError):
                    builder.add_function(vf_name, f"{base_func}/{divisor}")
                use_func = vf_name
            else:
                use_func = base_func

            builder.add_reaction(
                [],
                [sp_i],
                "functional",
                use_func,
                # GH #232: this per-species path already isolates one species, so
                # use that species' own factor — correct even when the reaction
                # mixes conversion factors across species.
                stat_factor=float(coeff) * species_cf.get(sid, 1.0),
            )

    for _loc, _comps in _ssa_rule_event_conc_reads(
        sbml_model,
        _ssa_reads,
        _ar_moving_comps | varvol_ssa_comps,
        species_comp,
        species_hosu,
        _ssa_ff_continuous,
    ):
        ssa_issues.append(
            SsaIssue(
                severity="error",
                code="variable_compartment_read",
                message=(
                    f"{_loc.split(':', 1)[0].capitalize()} '{_loc.split(':', 1)[1]}' reads a "
                    "concentration in a compartment whose size changes during the run "
                    f"({', '.join(_comps)}). SSA stores a count over the compartment's "
                    "load-time size, so the value it reads would be wrong. Use "
                    "method='ode', or declare the species hasOnlySubstanceUnits=\"true\"."
                ),
                location=_loc,
            )
        )

    # ── 10. SBML Events ──────────────────────────────────────────────
    # Parse <listOfEvents> and call builder.add_event() for each.
    # SBML events fire when trigger transitions false→true.
    #
    # Key challenge: 67.6% of BioModels event assignments target
    # parameters (not species). The C++ Event struct only supports
    # species assignments (species_idx, value_expr). For parameter
    # targets, we promote the parameter to a species (like rate rules).
    #
    # Event assignments to parameters that are already parameters work
    # by promoting them to species with initial value = param value,
    # adding an observable, and re-targeting the event assignment.
    n_events = sbml_model.getNumEvents()

    # A rule's value after an event's resize, as a function of its own, where
    # the new size written out through its rules would be too long (issue
    # #1124, item 4). One function for each text, under a name nothing else
    # has: the ids it is named by are joined by "_", and two can meet (event
    # e and compartment c_x, event e_c and compartment x), where an event read
    # the value of the other's function of that name. Each is added once an
    # assignment the model keeps reads it, since every function is evaluated
    # wherever an event executes (and was, before issue #1186, at every step).
    _resize_helper_of: dict[str, str] = {}  # text -> name
    _resize_helper_text: dict[str, str] = {}  # name -> text
    _resize_helpers_added: set[str] = set()

    def _resize_helper(key: str, text: str) -> str:
        name = _resize_helper_of.get(text)
        if name is None:
            # A function named as a parameter is bound to it (#266), and wrote
            # its value over the parameter's at every step.
            name = _unique_param_name(f"_rsz_{_safe_name(key)}")
            _resize_helper_of[text] = name
            _resize_helper_text[name] = text
        return name

    def _add_resize_helpers_read_by(texts) -> None:
        # Each helper the texts read, after the ones it reads.
        def add(name: str) -> None:
            if name in _resize_helpers_added:
                return
            _resize_helpers_added.add(name)
            body = _resize_helper_text[name]
            for inner in _EXPRTK_NAME.findall(body):
                if inner in _resize_helper_text:
                    add(inner)
            # Read only by event assignments, so not at every step (#1186).
            builder.add_function(name, body, deferred=True)

        if _resize_helper_text:
            for text in texts:
                for name in _EXPRTK_NAME.findall(text):
                    if name in _resize_helper_text:
                        add(name)

    if n_events > 0:
        # Track which parameters need promotion for event assignments
        event_param_promotions = {}  # param_id → species_idx

        # What a <delay> or <priority> may be constant-folded against: nothing
        # but literals. Both are evaluated when the trigger fires (SBML L3v2
        # §4.11.3/§4.11.4). GH #558 stopped folding a parameter the model writes
        # (a rule or event target), since it can change between t=0 and the
        # fire, but still folded every other parameter to its load-time value.
        # Those are exactly the parameters a caller changes with set_param, so a
        # delay `d` or a priority `p` read the value in the file forever: a
        # set_param('d', 0.5) left the event firing at t_trigger + 1, and a
        # swapped pair of priorities left the old order, with no warning. An
        # expression that reads any parameter now goes to the C++ dispatcher,
        # which evaluates it at trigger time against the live value; a delay
        # or priority that is a plain number still folds.
        event_fold_ctx: dict[str, float] = {}

        # What an event's resize of an assignment-rule compartment is found
        # from (issue #1055): each assignment rule's math, in document order,
        # and the species a resize of each compartment rescales. Both are the
        # same for every event, so they are read once, not once per event. A
        # reaction id a rule reads is its kinetic law, so the law is read with
        # the rules: `C := 1 + 0.5*J0`, J0 at k*A, is resized by `k = 0.6`.
        event_rule_math: dict[str, object] = {}
        event_rescaled_species: dict[str, list[str]] = {}
        event_rule_cache: dict = {}
        if ar_comp_targets:
            for j in range(sbml_model.getNumRules()):
                rule = sbml_model.getRule(j)
                if rule.isAssignment() and rule.getMath() is not None:
                    event_rule_math[rule.getVariable()] = rule.getMath()
            for j in range(sbml_model.getNumReactions()):
                _rx = sbml_model.getReaction(j)
                if _rx.getId() in referenced_reaction_ids:
                    _lm = _law_math_in_model_scope(_rx)
                    if _lm is not None:
                        event_rule_math[_rx.getId()] = _lm
            for sid, scomp in species_comp.items():
                if not species_hosu.get(sid, False) and sid not in assignment_targets:
                    event_rescaled_species.setdefault(scomp, []).append(sid)

        for i in range(n_events):
            event = sbml_model.getEvent(i)
            eid = event.getId() or f"_event_{i}"

            # ── Trigger ───────────────────────────────────────────────
            trigger = event.getTrigger()
            if trigger is None or trigger.getMath() is None:
                logger.warning(f"Event '{eid}' has no trigger, skipping")
                continue

            trigger_expr = _ast_to_exprtk_with_funcdefs(trigger.getMath(), func_defs)

            # persistent / initialValue (SBML L3 attributes)
            persistent = True
            initial_value = True
            if trigger.isSetPersistent():
                persistent = trigger.getPersistent()
            if trigger.isSetInitialValue():
                initial_value = trigger.getInitialValue()

            # useValuesFromTriggerTime (SBML L3, default true): when true,
            # event-assignment RHS values are evaluated at trigger time and
            # held until firing time. Has effect only when delay > 0.
            use_values_from_trigger_time = True
            if (
                hasattr(event, "isSetUseValuesFromTriggerTime")
                and (event.isSetUseValuesFromTriggerTime())
                or hasattr(event, "getUseValuesFromTriggerTime")
            ):
                use_values_from_trigger_time = event.getUseValuesFromTriggerTime()

            # ── Delay: prefer expression form; constant-fold opportunistically ─
            delay_val = 0.0
            delay_expr = ""
            if event.isSetDelay():
                delay_obj = event.getDelay()
                if delay_obj is not None and delay_obj.getMath() is not None:
                    delay_ast = delay_obj.getMath()
                    delay_numeric = _eval_ast_numeric(
                        delay_ast,
                        event_fold_ctx,
                        func_defs,
                        time_value=None,
                        avogadro_value=None,
                    )
                    if delay_numeric is not None:
                        delay_val = delay_numeric
                    else:
                        # Translate the AST to an ExprTk expression that the
                        # CVODE event dispatcher will evaluate at trigger time.
                        delay_expr = _ast_to_exprtk_with_funcdefs(delay_ast, func_defs)

            # ── Priority: optional in SBML L3; constant-fold or expression ────
            priority_val = 0
            priority_expr = ""
            if event.isSetPriority():
                pri_obj = event.getPriority()
                if pri_obj is not None and pri_obj.getMath() is not None:
                    pri_ast = pri_obj.getMath()
                    pri_numeric = _eval_ast_numeric(
                        pri_ast,
                        event_fold_ctx,
                        func_defs,
                        time_value=None,
                        avogadro_value=None,
                    )
                    # The int field holds a C int: an integer past it, 1e10 or
                    # 1e308, failed the load with pybind11's "incompatible
                    # function arguments". It takes the double path below.
                    if (
                        pri_numeric is not None
                        and float(pri_numeric).is_integer()
                        and abs(float(pri_numeric)) <= 2**31 - 1
                    ):
                        priority_val = int(pri_numeric)
                    elif pri_numeric is not None:
                        # SBML L3 §4.11.3 priorities are REAL numbers, but the
                        # C++ constant-priority field is an int — truncating a
                        # fractional priority (2.5 → 2) collapses distinct
                        # priorities into ties, so simultaneous events fire in
                        # the wrong order (01533/01714). Route a non-integer
                        # constant through the double-valued priority_expr path
                        # (evaluated exactly at trigger time), which the C++ event
                        # dispatcher already compares as a double. Integer
                        # priorities keep the fast int field (byte-identical for
                        # every existing model).
                        priority_expr = repr(float(pri_numeric))
                    else:
                        priority_expr = _ast_to_exprtk_with_funcdefs(pri_ast, func_defs)

            # ── Assignments ───────────────────────────────────────────
            assignments = []
            assignment_value_expr_by_var: dict[str, str] = {}
            explicit_species_assignment_expr: dict[int, str] = {}
            # (#74) compartment-size events: (comp_id, new_size_expr) for each
            # assignment whose target is a compartment. Used after this loop to
            # inject the species concentration rescaling SBML semantics require.
            resized_comps = []
            directly_resized_comps: set[str] = set()
            n_ea = event.getNumEventAssignments()
            for j in range(n_ea):
                ea = event.getEventAssignment(j)
                var = ea.getVariable()
                ea_math = ea.getMath()
                if ea_math is None:
                    continue

                semantic_value_expr = _ast_to_exprtk_with_funcdefs(ea_math, func_defs)
                assignment_value_expr_by_var[var] = semantic_value_expr

                # hOSU=true V≠1 amount restoration (GH #75): the RHS reads any
                # referenced species as amounts via the engine amount_valued
                # path (no per-ref rewrite). An event assignment writes the
                # target's stored *concentration* slot directly, so when the
                # target is itself an hOSU=true V≠1 species the assigned amount
                # is divided by V_c(target) here. Byte-identical for
                # all-V=1 / hOSU=false models.
                value_expr = _ast_to_exprtk_with_funcdefs(
                    _divide_by_target_vc(ea_math, var), func_defs
                )

                # (#74) A compartment-size assignment needs its contained
                # species' concentrations rescaled to preserve amounts. Record
                # the new-size expression; the rescale assignments are injected
                # after this loop. (_divide_by_target_vc is a no-op for a
                # compartment target, so value_expr is the plain RHS here.)
                if var in _comp_id_set:
                    resized_comps.append((var, value_expr))
                    directly_resized_comps.add(var)

                if var in species_idx:
                    # Direct species assignment
                    sp_i = species_idx[var]
                    assignments.append((sp_i, value_expr))
                    if var in species_comp:
                        explicit_species_assignment_expr[sp_i] = value_expr
                elif var in event_param_promotions:
                    # Already promoted parameter
                    sp_i = event_param_promotions[var]
                    assignments.append((sp_i, value_expr))
                else:
                    # Parameter, compartment, or species-reference stoich id —
                    # promote to species. Check if it's a known model symbol.
                    p = sbml_model.getParameter(var)
                    c = sbml_model.getCompartment(var)
                    if p is not None:
                        ic = ia_values.get(var, p.getValue() if p.isSetValue() else 0.0)
                    elif c is not None:
                        ic = ia_values.get(var, c.getSize() if c.isSetSize() else 1.0)
                    elif var in _species_ref_initial_values:
                        ic = ia_values.get(var, _species_ref_initial_values[var])
                    else:
                        logger.warning(
                            f"Event '{eid}': unknown assignment target '{var}', skipping"
                        )
                        continue

                    # Promote: add as a new species (non-fixed, since
                    # events will change it; no reactions affect it so
                    # it stays constant between events).
                    #
                    # reported=False (GH #71 / #237): an SBML parameter,
                    # compartment, or species-reference stoichiometry symbol is
                    # not a floating species. We promote it to a species only
                    # because the engine's event assignments write species
                    # slots, so it must carry per-trajectory state — but it must
                    # NOT appear as a trajectory column (RoadRunner doesn't emit
                    # it as a floating species either). The promoted species
                    # stays full integrator state with an observable (so other
                    # expressions that reference the symbol resolve to its live
                    # value); the Result layer projects it out of
                    # species/species_names.
                    # (Contrast rate-rule-promoted parameters in §8, which are
                    # genuine ODE variables RoadRunner reports — those stay
                    # reported.)
                    sp_i = builder.add_species(_safe_name(var), ic, fixed=False, reported=False)
                    builder.set_species_continuous(sp_i)  # issue #718: not a count
                    event_param_promotions[var] = sp_i
                    species_idx[var] = sp_i
                    species_ids.append(var)
                    builder.add_observable(_safe_name(var), [(sp_i, 1.0)])
                    _register_ic_param_ref(
                        builder, sp_i, var, ia_single_param_ref, ia_expr_param, _declared_param_ids
                    )
                    assignments.append((sp_i, value_expr))

            if ar_comp_targets and (assignment_value_expr_by_var or resized_comps):
                resized_comps.extend(
                    _rule_compartments_an_event_resizes(
                        event_rule_math,
                        ar_comp_targets,
                        resized_comps,
                        assignment_value_expr_by_var,
                        event_rescaled_species,
                        lambda node: _ast_to_exprtk_with_funcdefs(node, func_defs),
                        event_rule_cache,
                        lambda key, text, _eid=eid: _resize_helper(f"{_eid}_{key}", text),
                    )
                )

            # (#74) Inject per-species concentration rescaling for any
            # compartment whose size this event changes. bngsim stores every
            # species as a concentration (amount / V_c); SBML keeps the *amount*
            # unchanged when a compartment resizes, so each contained species'
            # stored concentration is multiplied by V_old / V_new. V_old is the
            # compartment symbol's pre-fire value and V_new is the assignment
            # RHS — both read against pre-fire state because a single event
            # evaluates every assignment RHS before applying any (simultaneous
            # semantics; cvode_simulator process_firing_batch). Skipped:
            #   • hOSU=true (amount_valued) species — the engine already reads
            #     them as amount = stored × V_c (a load-time constant), so the
            #     amount is preserved with no rescale (rescaling would corrupt
            #     it);
            #   • AssignmentRule-target species — their value is dictated by the
            #     rule each step, not a conserved amount;
            #   • species the event assigns explicitly — under ODE, append a
            #     second ODE-only assignment based on the explicit RHS so the
            #     amount implied by that pre-resize concentration is preserved.
            # Byte-identical for V=1 / hOSU-only / resize-free models (no
            # compartment target ⇒ resized_comps empty).
            # (#81 Tier 1) Parallel to ``assignments``: True marks an assignment
            # that applies under ODE only and is skipped under SSA. The injected
            # compartment-resize concentration rescale below is the only such
            # case — under SSA the stored value is amount/V_static and the count
            # (conc·V_static) must be left unchanged across the resize, so the
            # rescale (which preserves the *amount* by changing the stored
            # *concentration*) must not run. Every real event assignment applies
            # in both modes. Empty/all-False ⇒ byte-identical for non-resize
            # events. A rate-rule target is the exception among the rescales: it
            # is no count (issue #718), and its slot holds the rule's
            # concentration under SSA as under ODE, so its rescale applies in
            # both. Skipped under SSA, its amount changed with the size: [S]
            # 0.733 for 0.867 a step after `c = 3`.
            assignment_ode_only = [False] * len(assignments)
            # (#936) Each rescale's (assignment index, compartment size, index of
            # the event's own assignment to the species or -1): the engine stores
            # base·size before/size after the event's other assignments, so the
            # amount is the one the species has when the resize executes and the
            # new size the one the compartment takes.
            rescale_meta: list[tuple[int, str, int]] = []
            if resized_comps:
                explicitly_assigned = {sp_i for sp_i, _ in assignments}
                n_own = len(assignments)
                for comp_id, new_size_expr in resized_comps:
                    comp_safe = _safe_name(comp_id)
                    for sid, scomp in species_comp.items():
                        if scomp != comp_id:
                            continue
                        if species_hosu.get(sid, False):
                            continue
                        if sid in assignment_targets:
                            continue
                        sp_i = species_idx[sid]
                        ode_only = sid not in rate_rule_targets
                        # A rate-rule target's amount is reported through the
                        # rule-target map (#728), which reads its column as a
                        # concentration under every method; the dilution map
                        # reads it as a count under SSA.
                        if comp_id in ar_comp_targets and ode_only:
                            ar_dilution_species.add(sid)
                        if sp_i in explicit_species_assignment_expr:
                            # Simultaneous event semantics evaluate all RHS
                            # values against the pre-fire state. A hOSU=false
                            # species assignment gives a concentration in the
                            # old compartment; after a same-event resize, the
                            # stored concentration must be adjusted so that the
                            # amount from that explicit assignment is conserved.
                            value_expr = explicit_species_assignment_expr[sp_i]
                            rescale_expr = f"({value_expr}) * ({comp_safe}) / ({new_size_expr})"
                            base_idx = next(
                                i for i, (j, _) in enumerate(assignments[:n_own]) if j == sp_i
                            )
                            rescale_meta.append((len(assignments), comp_safe, base_idx))
                            assignments.append((sp_i, rescale_expr))
                            assignment_ode_only.append(ode_only)
                            continue
                        if sp_i in explicitly_assigned:
                            continue
                        rescale_expr = f"{_safe_name(sid)} * ({comp_safe}) / ({new_size_expr})"
                        rescale_meta.append((len(assignments), comp_safe, -1))
                        assignments.append((sp_i, rescale_expr))
                        assignment_ode_only.append(ode_only)

            if not assignments:
                logger.warning(f"Event '{eid}': no valid assignments, skipping")
                continue
            _add_resize_helpers_read_by(expr for _, expr in assignments)

            builder.add_event(
                eid,
                trigger_expr,
                assignments,
                delay=delay_val,
                priority=priority_val,
                persistent=persistent,
                initial_value=initial_value,
                use_values_from_trigger_time=use_values_from_trigger_time,
                delay_expr=delay_expr,
                priority_expr=priority_expr,
                assignment_ode_only=assignment_ode_only,
            )
            for _ai, _size, _base in rescale_meta:
                builder.set_last_event_assignment_rescale(_ai, _size, _base)

    # (#1054) A piecewise size's jumps, found in §8c: an event at each
    # condition's root, each way, rescales the compartment's concentrations so
    # their amounts hold. ODE only, as every resize rescale is (#81); it is
    # added after the model's own events so their indices stay as they were.
    # The jump belongs to the size at that instant. Marked as a jump, it
    # executes before anything else there, and the model's events triggered at
    # the instant read their trigger, values and delay after it: read before
    # it, `at (time >= 2): X = Sn` beside `cv := piecewise(1, time < 2, 5)`
    # gave X the concentration from before the jump. Executed after an event
    # there that sets a concentration in the compartment, its rescale of the
    # value from before overwrote that setting (`at (time >= 2): Sn = 10` left
    # Sn at 1 or 10 as the same-priority draw went, #242). The priority keeps
    # it out of the draw against the model's own events under SSA, where it
    # writes nothing.
    #
    # A jump moves more than its own compartment's concentrations. A size rule
    # that reads one of them, directly or through other rules, jumps with it,
    # whether or not a condition of its own flips there: beside `cv :=
    # piecewise(1, t < 2, 5)` and [Sn] = 5, `cw := 1 + Sn` goes from 6 to 2 at
    # t = 2, and its concentrations were held where their amounts are. So the
    # jump rescales the concentrations in each such compartment too, each after
    # the ones whose species it reads, as an event that resizes a compartment
    # does (#1055): from the size there when the jump executes to the size its
    # own rescale leaves, both read as the rule gives them then (#936). Where
    # the compartment jumps at the instant itself, `cw := piecewise(1, t < 2, 1
    # + Sn)`, its own jump goes from 1 to 1 + [Sn] at the [Sn] from before, 6,
    # and executes first; this rescale takes it from 6 to 1 + 1, where its own
    # new size had read [Sn] from before cv's jump and left [Sw] = 3 at 3/6 for
    # 3/2.
    _jump_rule_math: dict[str, object] = {}
    _jump_moves: dict[str, list[str]] = {}
    _jump_rule_cache: dict[str, tuple[dict, dict]] = {}
    if _size_jumps:
        for j in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(j)
            if rule.isAssignment() and rule.getMath() is not None:
                _jump_rule_math[rule.getVariable()] = rule.getMath()
        for sid, scomp in species_comp.items():
            if _diluted(sid):
                _jump_moves.setdefault(scomp, []).append(sid)
    for k, (comp, trigger, before, after) in enumerate(_size_jumps):
        assignments = [
            (species_idx[sid], f"{_safe_name(sid)} * ({before}) / ({after})")
            for sid in _jump_species[comp]
        ]
        # Its own rule is left out of what is read. A size that reads it by name
        # is written out through it (§8c), so it jumps at its roots by events of
        # its own, and a condition of its own on its concentrations has a root,
        # and an event, of its own too.
        if comp not in _jump_rule_cache:
            _jump_rule_cache[comp] = (
                {var: m for var, m in _jump_rule_math.items() if var != comp},
                {},
            )
        rules_beside, cache = _jump_rule_cache[comp]
        followers = _rule_compartments_an_event_resizes(
            rules_beside,
            ar_comp_targets,
            [],
            {sid: f"{_safe_name(sid)} * ({before}) / ({after})" for sid in _jump_species[comp]},
            _jump_moves,
            lambda node: _ast_to_exprtk_with_funcdefs(node, func_defs),
            cache,
            lambda key, text, _k=k: _resize_helper(f"_size_jump_{_k}_{key}", text),
        )
        rescales: list[tuple[int, str]] = []
        for follower, new_size in followers:
            for sid in _jump_moves.get(follower, ()):
                rescales.append((len(assignments), _safe_name(follower)))
                assignments.append(
                    (
                        species_idx[sid],
                        f"{_safe_name(sid)} * ({_safe_name(follower)}) / ({new_size})",
                    )
                )
                ar_dilution_species.add(sid)
        _add_resize_helpers_read_by(expr for _, expr in assignments)
        builder.add_event(
            f"_size_jump_{_safe_name(comp)}_{k}",
            trigger,
            assignments,
            persistent=True,
            initial_value=True,
            priority_expr="1e308",
            assignment_ode_only=[True] * len(assignments),
        )
        builder.set_last_event_size_jump(comp)
        for ai, size in rescales:
            builder.set_last_event_assignment_rescale(ai, size, -1)
        for sid in _jump_species[comp]:
            ar_dilution_species.add(sid)

    # (#81) Apply the deferred SSA live-volume tags. §9 recorded each mass-action
    # varvol reaction as (rxn_idx, comp_id, exp); the compartment's promoted
    # species index is only resolvable now (an event-resized compartment is
    # promoted in §10, after reaction emission). conc[species_idx[comp_id]] is
    # the live volume V_live(t) — the rate rule integrates it / the event writes
    # it (volume_factor 1.0, so its stored value IS the size).
    for _rxn_idx, _comp_id, _exp in ssa_varvol_fixups:
        _live_idx0 = species_idx.get(_comp_id)
        if _live_idx0 is None:
            logger.warning(
                f"GH #81: variable-volume compartment '{_comp_id}' was not "
                "promoted to a species; SSA live-volume correction skipped for "
                f"reaction index {_rxn_idx}."
            )
            continue
        builder.set_reaction_live_volume(_rxn_idx, _live_idx0, _exp)

    # (#144 case 4) Apply the deferred cross-compartment variable-volume fixups.
    # The promoted compartment species index (conc[idx] == V_live(t)) is only
    # resolvable here, after §10. ODE: each varvol hOSU=false species divides by its
    # live compartment volume in compute_derivs. SSA: a per-compartment propensity
    # correction (V_static / V_live)^m_c.
    # (#741) SSA: an event's assignment of a concentration to an hOSU=false
    # species in a rate-rule or event-resized compartment, stored as
    # amount/V_static under SSA, is stored as value·V_live/V_static.
    for _sid, _cid in species_comp.items():
        if (
            _cid in (rate_rule_comps | event_resize_comps)
            and _sid in species_idx
            and not species_hosu.get(_sid, False)
            and _sid not in rate_rule_targets
            and _sid not in assignment_targets
            and species_idx.get(_cid) is not None
        ):
            builder.set_species_ssa_live_volume(species_idx[_sid], species_idx[_cid])
    for _sp_idx, _comp_id in ode_xcomp_species_fixups:
        _live_idx0 = species_idx.get(_comp_id)
        if _live_idx0 is None:
            logger.warning(
                f"GH #144: variable-volume compartment '{_comp_id}' was not "
                "promoted to a species; cross-compartment ODE live-volume divide "
                f"skipped for species index {_sp_idx}."
            )
            continue
        builder.set_species_ode_live_volume(_sp_idx, _live_idx0)
    for _rxn_idx, _comp_id, _vstat, _exp in ssa_xcomp_term_fixups:
        _live_idx0 = species_idx.get(_comp_id)
        if _live_idx0 is None:
            logger.warning(
                f"GH #144: variable-volume compartment '{_comp_id}' was not "
                "promoted to a species; cross-compartment SSA live-volume term "
                f"skipped for reaction index {_rxn_idx}."
            )
            continue
        builder.add_reaction_live_volume_term(_rxn_idx, _live_idx0, _vstat, _exp)

    # ── 10.5. Discontinuity triggers (GH #72, GH #194, GH #246) ────────
    # Scan every expression that feeds the ODE RHS — assignment-rule and
    # rate-rule math, plus kinetic laws — for inequalities switching on the
    # `time` csymbol, and register each distinct threshold as a CVODE root.
    # This stops the integrator from stepping over a narrow piecewise pulse
    # (e.g. a 0.125-time-unit chemo infusion window).
    #
    # Event trigger roots are already registered by the C++ event dispatcher as
    # the full Boolean trigger. That is insufficient for a compound pulse like
    # ``(S > lo) && (S < hi)``: the full trigger can be false on both sides of a
    # wide integrator step, so no sign change is visible. Register each
    # relational trigger subcondition as a no-op root too; at the forced stop the
    # existing event re-check sees the actual trigger state and queues the event
    # with the normal delay / persistence / priority semantics.
    #
    # Builder de-duplicates, so a threshold shared by several expressions yields
    # one root. Models with no such discontinuities register nothing and
    # integrate exactly as before.
    disc_conditions: list[str] = []
    time_dependent_names = _assignment_time_dependent_names(sbml_model)
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if (rule.isAssignment() or rule.isRate()) and rule.getMath():
            _collect_time_discontinuity_conditions(
                rule.getMath(), func_defs, disc_conditions, None, time_dependent_names
            )
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl and kl.getMath():
            # The reaction's <localParameter> map is the lexical scope the
            # condition is written in; without it a threshold naming a local
            # parameter emits a symbol the evaluator does not have (GH #194's
            # four ExprTk load failures, on the state side of the same scan).
            _collect_time_discontinuity_conditions(
                kl.getMath(),
                func_defs,
                disc_conditions,
                reaction_local_param_maps.get(i),
                time_dependent_names,
            )
    # Which of them an event trigger reads: a trigger that only touches its
    # threshold (`sin(time) >= 1`) still fires there, and the stop on the touch
    # that makes it fire would only hurt a rate law (issue #714).
    event_disc: set[str] = set()
    # An equality in a trigger, `at (time == 2)`, is true at one double. It is
    # no root (the boolean does not change sign across a step that passes it)
    # and no ordering, so nothing made the run land there, and the event fired
    # only where an output sample happened to. The run stops on the instant
    # instead, where the equality reads the clock and constants alone; the
    # resolver declines any other.
    trigger_equalities: list[str] = []
    for i in range(sbml_model.getNumEvents()):
        event = sbml_model.getEvent(i)
        trigger = event.getTrigger()
        if trigger is not None and trigger.getMath() is not None:
            _from_trigger: list[str] = []
            # Into the function definitions the trigger calls as well, as the
            # time and state scans go: `near(A, 5, 0.01)` with `near :=
            # abs(x - c) < w` holds its window in the body, and the trigger's
            # own math has no relational in it. Rooted whole, the window was
            # stepped over and the event never fired (issue #904). Every
            # trigger atom is kept, so no formal needs judging dynamic.
            for _body, _scope, _ in _iter_funcdef_scan_frames(
                trigger.getMath(), func_defs, lambda *_: False
            ):
                _collect_relational_edge_conditions(
                    _body,
                    func_defs,
                    _from_trigger,
                    local_params=_scope,
                    equalities=trigger_equalities,
                )
            disc_conditions.extend(_from_trigger)
            event_disc.update(_from_trigger)

    # The same per-atom edge routine over the RHS-feeding math (GH #194), for
    # the relationals that read integrated state rather than time. Splitting the
    # Boolean into its atoms is what catches a narrow ``(X > lo) && (X < hi)``
    # window: the conjunction reads false at both ends of one wide step, so the
    # compound condition never changes sign, while each half does.
    state_names = _state_dependent_names(sbml_model)
    if state_names:
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if (rule.isAssignment() or rule.isRate()) and rule.getMath():
                _collect_state_relational_conditions(
                    rule.getMath(), func_defs, state_names, disc_conditions
                )
        for i in range(sbml_model.getNumReactions()):
            kl = sbml_model.getReaction(i).getKineticLaw()
            if kl and kl.getMath():
                _collect_state_relational_conditions(
                    kl.getMath(),
                    func_defs,
                    state_names,
                    disc_conditions,
                    reaction_local_param_maps.get(i),
                )

    # State-dependent floor()/ceiling() thresholds (GH #244): small numeric
    # integer boundaries in expressions like ``ceiling(S1 * p1)`` are also
    # discontinuities in the RHS, but the crossing is a state threshold rather
    # than a fixed time threshold. Register the bounded cases we can infer at
    # load time; models outside that conservative envelope are left unchanged.
    # (`time_dependent_names` is the same set the time scan above reads.)
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if (rule.isAssignment() or rule.isRate()) and rule.getMath():
            _collect_state_discontinuity_conditions(
                rule.getMath(), func_defs, eval_ctx, time_dependent_names, disc_conditions
            )
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl and kl.getMath():
            _collect_state_discontinuity_conditions(
                kl.getMath(),
                func_defs,
                eval_ctx,
                time_dependent_names,
                disc_conditions,
                reaction_local_param_maps.get(i),
                _kinetic_law_local_values(kl),
            )

    # Every spelling met (GH #274's check below asks after the text it emits),
    # and one root per crossing: a condition reached through a function
    # definition is spelled `(time())` where the same one written directly is
    # `time()`, and both were registered (issue #1082). The first spelling met
    # is the one registered, so a model with no such pair registers what it did.
    seen_disc: set[str] = set()
    registered_disc: dict[str, str] = {}
    for cond in disc_conditions:
        seen_disc.add(cond)
        key = _condition_key(cond)
        if key not in registered_disc:
            registered_disc[key] = cond
            builder.add_discontinuity_trigger(cond)

    # Periodic floor()/modulo dosing schedules (GH #88) that the monotonic roots
    # above can't resolve — register a step-size bound that keeps the adaptive
    # integrator from stepping over a narrow dose pulse. None for any model with
    # no time-dependent floor/ceil/modulo feeding the RHS (the common case), and
    # since GH #274 also None when every such node's influence on the RHS is
    # already gated by one of the roots just registered — hence `seen_disc`,
    # which is what makes that judgement possible rather than a guess.
    periodic_disc_max_step = _periodic_time_disc_max_step(
        sbml_model, func_defs, eval_ctx, seen_disc, reaction_local_param_maps
    )

    # ── 10.6. rateOf csymbol support (GH #106) ─────────────────────────
    # Enable the live-derivative path iff the model actually *references*
    # rateOf — the official csymbol or a *called* COPASI rateOf-idiom funcDef.
    # A funcDef defined but never called (MODEL2403070001) leaves the model on
    # the byte-identical non-rateOf path.
    rateof_funcdef_names = {
        name for name, (_p, body) in func_defs.items() if body is _RATEOF_FUNCDEF
    }
    _uses_rateof = _model_uses_rateof(sbml_model, func_defs, rateof_funcdef_names)
    # A §8c dilution term that reads a species' rate needs the machinery too,
    # though the model itself names no rateOf (#1052).
    if _uses_rateof or _dilution_reads_rate:
        builder.enable_rateof()

    # ── 10.7. Unresolvable compartment sizes (#170) ───────────────────
    # Every other compartment size is now a live parameter: §7 put the volume
    # back into the mass-action scalar as a ratio on the rate parameter, §9 emits
    # the storage divide against the compartment symbol, §3 bound each species'
    # storage convention to the size, and the SSA propensity reads it from
    # ``params``. What is left is the residue this loader could not resolve — an
    # assignment-rule compartment, and a compartment a single mass-action scalar
    # divides two compartments' species by — which keeps issue #164's refusal so
    # a write is never silently half-applied.
    for _cid in sorted(compartment_write_refused):
        builder.set_param_volume_write_refused(_safe_name(_cid))

    # ── 10.8. Parameters folded at load (issues #313, #695, #696) ─────────
    # A handful of things an SBML document writes over its parameters are
    # evaluated once, when the model is built, and the number is what the model
    # holds from then on: the size of a compartment that an initialAssignment
    # sets (`c = 2*p`), a stoichiometry (a <stoichiometryMath>, or an
    # initialAssignment onto a speciesReference id, or the id itself), a
    # conversionFactor, and an initialAssignment the loader could not keep
    # symbolic, onto a parameter or onto a species. A parameter that reaches
    # the model through one of them does not move it afterwards: `set_param` on
    # it took the value and left the number (P(1) stayed 12.64 where a rebuild
    # gives 37.93), and its forward sensitivity lacked everything that goes
    # through the number, an exact 0 where nothing else reads the parameter.
    # Such a parameter is named here with what it was folded into, and
    # `Model.set_param` and the sensitivity request refuse it by that.
    #
    # What a fold reads is followed all the way down. The number came from
    # every symbol the expression names and from whatever gave each of those
    # its value at load: an initialAssignment, an assignment rule, a kinetic
    # law for a reaction id (with the law's own parameters, which the built
    # model has as `_lp_<reaction>_<id>`), a <stoichiometryMath> for a
    # speciesReference id, and the size of a species' compartment where the
    # species is declared in the unit it is not read in. (A lifted parameter is
    # read through the initialAssignment it was lifted from.) `c = J0`, `c = q`
    # under `q := 3*p`, and `c = S0` under `S0 = 2*p` each leave `p` as stale as
    # `c = 2*p` does.
    # `rateOf(B)` reads more than `B`: the rate of `B` at load is every kinetic
    # law of a reaction that makes or takes it, or its rate rule. `c =
    # rateOf(B)` under `J0: A -> B; k1*A` left `k1` writable and the size where
    # it was (0.172 off). And it does not read `B`'s value, so its argument is
    # not a name read unless the expression names it elsewhere: `p2 =
    # rateOf(p1)` of a `p1` nothing changes is 0 whatever `p1` is, and used to
    # freeze `p1` (issue #1048). The same for a call to COPASI's rateOf idiom:
    # its body is never evaluated, the call is the same rate as the csymbol.
    def _names_read(_m) -> set[str]:
        if not _uses_rateof:
            return _ast_name_set(_m)
        _out: set[str] = set()
        stack = [_m] if _m is not None else []
        while stack:
            _node = stack.pop()
            _t = _node.getType()
            if _t == libsbml.AST_NAME:
                _out.add(_node.getName())
            if _t == libsbml.AST_FUNCTION_RATE_OF or (
                _t == libsbml.AST_FUNCTION and _node.getName() in rateof_funcdef_names
            ):
                for _k in range(_node.getNumChildren()):
                    _out |= {f"rateOf({_n})" for _n in _ast_name_set(_node.getChild(_k))}
                continue
            stack.extend(_node.getChild(_k) for _k in range(_node.getNumChildren()))
        return _out

    _reads: dict[str, set[str]] = {}
    for _sym, _m in _taint_edges:
        _reads.setdefault(_sym, set()).update(_names_read(_m))
    for _rxn_i in range(sbml_model.getNumReactions() if _uses_rateof else 0):
        _rxn_f = sbml_model.getReaction(_rxn_i)
        for _getter, _n_refs in (
            (_rxn_f.getReactant, _rxn_f.getNumReactants()),
            (_rxn_f.getProduct, _rxn_f.getNumProducts()),
        ):
            for _j in range(_n_refs):
                _sr = _getter(_j)
                _sp_r = sbml_model.getSpecies(_sr.getSpecies())
                if _sp_r is None or _sp_r.getBoundaryCondition() or _sp_r.getConstant():
                    continue  # no reaction changes it
                _rate = _reads.setdefault(f"rateOf({_sr.getSpecies()})", set())
                _rate.add(_rxn_f.getId())
                if hasattr(_sr, "getId") and _sr.getId():
                    _rate.add(_sr.getId())
                if not _sp_r.getHasOnlySubstanceUnits():
                    # The rate of a concentration is the amount's over the
                    # size (`c = 1 + rateOf(B)`: a write to B's `d` 0.33 off).
                    _rate.add(_sp_r.getCompartment())
                # And its conversionFactor scales the amount's rate.
                if _sp_r.isSetConversionFactor():
                    _rate.add(_sp_r.getConversionFactor())
                elif sbml_model.isSetConversionFactor():
                    _rate.add(sbml_model.getConversionFactor())
    for _j in range(sbml_model.getNumRules() if _uses_rateof else 0):
        _r = sbml_model.getRule(_j)
        if _r.isRate() and _r.getMath() is not None:
            _reads.setdefault(f"rateOf({_r.getVariable()})", set()).update(
                _names_read(_r.getMath())
            )
    # A concentration in a compartment a rate rule sizes is diluted as the size
    # changes, whether reactions move it or not, so its rate also reads its own
    # value, the size and the size's rate (issue #738): `p = rateOf(B)` under
    # `c' = g` left `g` writable, and `p` 0.675 where a rebuild after `g = 1`
    # gives 0.35. One with a rate rule of its own has its rate stated.
    for _j in range(sbml_model.getNumSpecies() if _uses_rateof else 0):
        _sp_d = sbml_model.getSpecies(_j)
        _sid_d, _cid_d = _sp_d.getId(), _sp_d.getCompartment()
        if (
            _cid_d in rate_rule_comps
            and not _sp_d.getHasOnlySubstanceUnits()
            and _sid_d not in rate_rule_targets
        ):
            _reads.setdefault(f"rateOf({_sid_d})", set()).update(
                {_sid_d, _cid_d, f"rateOf({_cid_d})"}
            )
    for _rxn_i in range(sbml_model.getNumReactions()):
        _rxn_f = sbml_model.getReaction(_rxn_i)
        _kl_f = _rxn_f.getKineticLaw()
        _rid_f = _rxn_f.getId()
        if not _rid_f or _kl_f is None or _rid_f not in _reads:
            continue
        _own = {_lp.getId() for _lp in _kinetic_law_local_params(_kl_f)}
        if _own:
            _reads[_rid_f] = {f"_lp_{_rid_f}_{_n}" if _n in _own else _n for _n in _reads[_rid_f]}
    for _j in range(sbml_model.getNumSpecies()):
        _sp_f = sbml_model.getSpecies(_j)
        if _sp_f.getId() in _ia_math:
            continue  # its value is the assignment's, in the unit it is read in
        if (
            _sp_f.isSetInitialConcentration()
            if _sp_f.getHasOnlySubstanceUnits()
            else (_sp_f.isSetInitialAmount() and not _sp_f.isSetInitialConcentration())
        ):
            _reads.setdefault(_sp_f.getId(), set()).add(_sp_f.getCompartment())

    _folded_into: dict[str, str] = {}

    def _fold(names, what: str) -> set[str]:
        """Freeze what ``names`` reach behind ``what``; return what it froze."""
        frozen: set[str] = set()
        stack = list(names)
        seen: set[str] = set()
        while stack:
            _n = stack.pop()
            if _n in seen:
                continue
            seen.add(_n)
            stack.extend(_reads.get(_n, ()))
            if _n in compartment_write_refused:
                continue  # a size §10.7 refuses by name already, with its own reason
            if (
                (_n in _param_decl_index and _n not in _ar_targets)
                or _n in comp_param_idx
                or _n.startswith("_lp_")
            ):
                _folded_into.setdefault(_safe_name(_n), what)
                frozen.add(_safe_name(_n))
        return frozen

    # (#313) the residue of the lift: a parameter initialAssignment §0 folded
    # that §2 could not put back on its symbols. It is also named in a warning
    # once the model is built, from exactly what this freezes, so the warning
    # and the refusal cannot disagree. (Issue #1048: the warning was computed in
    # §2 from the expression's own names, so it said nothing of `k1` for
    # `p2 = rateOf(B)` under `J0: A -> B; k1*A`, which this refuses.)
    _frozen_upstream: set[str] = set()
    for _pid, _m in _ia_math.items():
        if _pid in _lift_expr or _pid in comp_param_idx or _pid not in _param_decl_index:
            continue
        if _pid in _ar_targets:
            continue  # §4's rule recomputes it; the fold is not what a write faces
        _frozen_upstream |= _fold(
            _names_read(_m), f"the initial value of {_pid!r}, which an initialAssignment sets"
        )
    # The same for a species, and for a parameter a rate rule or an event makes
    # a state of: an initialAssignment that is not lowered to an expression of
    # parameters (it reads a reaction's rate, the time, a species under a rule)
    # leaves the state's initial value a number.
    for _sym in sorted(_ia_state_targets):
        if _sym in ia_param_expr:
            continue
        if (
            _sym in ia_single_param_ref
            and ia_single_param_ref[_sym] in _declared_param_ids
            and ia_single_param_ref[_sym] not in _ar_targets
        ):
            # (A rule's target is declared too, and its slot is the rule's:
            # `S = q` under `q := p*(1 + A)` is a number at load.)
            continue
        if _sym in _ia_math:
            _fold(
                _names_read(_ia_math[_sym]),
                f"the initial value of {_sym!r}, which an initialAssignment sets",
            )
    # (#696) a compartment's size.
    for _cid in comp_param_idx:
        if _cid in _ia_math:
            _fold(
                _names_read(_ia_math[_cid]),
                f"the size of compartment {_cid!r}, which an initialAssignment sets",
            )
    # The same where an assignment rule sizes it from parameters alone and it
    # holds a species: the rule's value at load is the size the species were
    # converted by (`c := 2*p`). A rate law that names such a compartment reads
    # the rule itself, so one that holds no species has nothing folded.
    _holds_species = {
        sbml_model.getSpecies(_j).getCompartment() for _j in range(sbml_model.getNumSpecies())
    }
    for _j in range(sbml_model.getNumCompartments()):
        _cid = sbml_model.getCompartment(_j).getId()
        if _cid in _ic_const_ar and _cid in _ar_math and _cid in _holds_species:
            _fold(
                _names_read(_ar_math[_cid]),
                f"the size of compartment {_cid!r}, which an assignment rule sets",
            )
    # A compartment that a rate rule or an event makes a state of keeps its
    # initialAssignment as the state's initial value, and that follows a write.
    # A species in it that is declared in the unit it is not held in does not:
    # its value was converted by the size at load (`c = 2*p; c' = 0.1` with S
    # an amount in c: [S] 0.5 off).
    #
    # The same in a compartment that an assignment rule sizes from the time or
    # a state: the size at load is the rule's value there, and what the rule
    # reads set it (`V := v0 + g*time` with a concentration held as an amount:
    # dX/dv0 was 0 for 0.2262, and a write to `v0` left the amount where it
    # was, issue #742). Every name the rule reads is listed, the ones that do
    # not move its value at the start among them (`g` here). (An initial value
    # assigned from such a size, `X = 2*V`, is not kept as an expression and
    # is listed by the loop above. A rule that is a constant over parameters
    # is listed whole just above; it comes through here too, to no effect.)
    for _j in range(sbml_model.getNumSpecies()):
        _sp_f = sbml_model.getSpecies(_j)
        _cid = _sp_f.getCompartment()
        if _cid in (rate_rule_comps | event_resize_comps) and _cid in _ia_math:
            _sized_by, _listed = _ia_math[_cid], ""
        elif _cid in _ar_math:
            _sized_by = _ar_math[_cid]
            _listed = (
                ", the value of its assignment rule there (every name the rule reads is listed)"
            )
        else:
            continue
        if _sp_f.getId() in _ia_math or _sp_f.getId() in _ar_targets:
            # Its value is the assignment's, in the unit it is read in; or its
            # rule's, whatever value it declares.
            continue
        if (
            _sp_f.getInitialConcentration() != 0.0
            if _sp_f.getHasOnlySubstanceUnits() and _sp_f.isSetInitialConcentration()
            else (
                not _sp_f.getHasOnlySubstanceUnits()
                and _sp_f.isSetInitialAmount()
                and not _sp_f.isSetInitialConcentration()
                and _sp_f.getInitialAmount() != 0.0  # 0 is 0 at any size
            )
        ):
            _fold(
                _names_read(_sized_by),
                f"the initial value of species {_sp_f.getId()!r}, which was converted by the "
                f"size of compartment {_cid!r} at load{_listed}",
            )
    # (#695) a stoichiometry, and a conversionFactor.
    for _rxn_i in range(sbml_model.getNumReactions()):
        _rxn_f = sbml_model.getReaction(_rxn_i)
        for _getter, _n_refs in (
            (_rxn_f.getReactant, _rxn_f.getNumReactants()),
            (_rxn_f.getProduct, _rxn_f.getNumProducts()),
        ):
            for _j in range(_n_refs):
                _sr = _getter(_j)
                _srid = _sr.getId() if hasattr(_sr, "getId") else ""
                _named = bool(
                    _srid and _srid in _species_ref_initial_values and _srid not in _reserved_ids
                )
                _sm = _sr.getStoichiometryMath() if hasattr(_sr, "getStoichiometryMath") else None
                _has_math = _sm is not None and _sm.getMath() is not None
                if not _named and not _has_math:
                    continue  # a plain number: nothing to name
                _kept = _variable_stoich_expr(_sr) is not None
                if not _named and _kept:
                    continue  # kept symbolic (§6c): a write to what it reads moves it
                _what = f"the stoichiometry of {_sr.getSpecies()!r} in reaction {_rxn_f.getId()!r}"
                if _named and _srid not in _ar_targets:
                    # §6b registered the id as a parameter that holds the number
                    # at load. Kept symbolic or not, the stoichiometry does not
                    # read that parameter back. (An id a rule sets is the rule's
                    # slot, which no write reaches.)
                    _folded_into.setdefault(_safe_name(_srid), _what)
                if _kept:
                    continue  # a write to what it reads moves it
                if _has_math:
                    _fold(_names_read(_sm.getMath()), _what)
                if _named and _srid in _ia_math:
                    _fold(_names_read(_ia_math[_srid]), _what)
    if sbml_model.isSetConversionFactor():
        _fold([sbml_model.getConversionFactor()], "the model's conversionFactor")
    for _sid in species_ids:
        _sp = sbml_model.getSpecies(_sid)
        if _sp is not None and _sp.isSetConversionFactor():
            _fold([_sp.getConversionFactor()], f"the conversionFactor of species {_sid!r}")

    # ── 11. Build ─────────────────────────────────────────────────────
    core = builder.build()
    model = Model(_core=core)
    # Only what the built model has as a parameter a caller could write.
    _built_params = set(core.param_names)
    model._frozen_params = {n: w for n, w in _folded_into.items() if n in _built_params}
    # (#313) Anything the residue's fold read that a caller can still write is
    # frozen behind it: `set_param` refuses it, and so does the sensitivity
    # request. Say so once, at load, by the names they refuse. (An
    # initialAssignment that reads nothing writable stays quiet: it is frozen,
    # but no parameter is frozen behind it.)
    _residue_frozen = sorted(n for n in model._frozen_params if n in _frozen_upstream)
    if _residue_frozen:
        logger.warning(
            "initialAssignment(s) that bngsim could not keep symbolic read the "
            "parameter(s) %s. Their contribution through that initialAssignment is "
            "frozen at its load-time value, so set_param refuses to change them and "
            "a forward-sensitivity column for one is refused (Model.frozen_params).",
            ", ".join(_residue_frozen),
        )
    model._ssa_issues = ssa_issues
    model._periodic_disc_max_step = periodic_disc_max_step
    # The registered discontinuity conditions, verbatim (issue #305). A root at
    # a crossing is only half of reaching it: CVODE tests for one solely on a
    # step it ACCEPTS, so where the jump is large enough that the error test
    # rejects every step containing the crossing, the root never fires and the
    # run wedges at t + h == t just below it. Simulator.run resolves each of
    # these to a crossing *time* it can stop the step on
    # (_switch_sensitivity.fixed_time_crossings). Handing over the registered
    # set itself, rather than re-deriving one from the rule text, is what keeps
    # the stops a subset of the roots.
    # ...and, beside the roots, the trigger equalities, which are stops alone.
    # Both in the spelling registered (issue #1082).
    # An equality reached through a function definition beside the same one
    # written directly is one stop too, in the first spelling met (#1106's review).
    registered_eq: dict[str, str] = {}
    for eq in trigger_equalities:
        registered_eq.setdefault(_condition_key(eq), eq)
    model._time_disc_conditions = tuple(
        sorted(set(registered_disc.values()) | set(registered_eq.values()))
    )
    model._event_disc_conditions = frozenset(
        {registered_disc[k] for k in map(_condition_key, event_disc) if k in registered_disc}
        | set(registered_eq.values())
    )

    # AR-target species report map. An AssignmentRule target species is
    # emitted ``fixed`` (its ODE derivative is zeroed), so the integrator
    # leaves it frozen at its initial value and ``Result.species`` would
    # report that stale value. Its algebraically-correct value (what RR
    # reports, re-evaluating the rule each step) lives under the same bare
    # name as either an observable (linear-on-species rules — e.g.
    # ``Pt = P0+P1+P2+Pn``) or a function/expression (everything else —
    # e.g. ``S = floor(time/tau)``). Record the source so Simulator.run can
    # overwrite the species column with the live value. Names are mangled
    # to match the runtime arrays.
    ar_report_map: dict[str, tuple] = {}
    for sid in assignment_targets:
        if sid not in species_idx:
            continue
        safe = _safe_name(sid)
        kind = "observable" if sid in linear_assignment_rules else "expression"
        # GH #75: the rule's observable/expression now yields the target's
        # *amount* when the target is hOSU=true (referenced species are read as
        # amounts by the engine, and the loader no longer divides by V_c at
        # emission). bngsim stores/reports concentration, so divide the reported
        # value by V_c(target) for an hOSU=true V≠1 target. 1.0 ⇒ no-op for V=1
        # and hOSU=false targets (byte-identical reporting). NOTE: gated on
        # hOSU, not merely V≠1 — a hOSU=false AR-target in a V≠1 compartment is
        # assigned in concentration directly and must NOT be divided.
        vdiv = (
            comp_volumes.get(species_comp.get(sid, ""), 1.0)
            if species_hosu.get(sid, False)
            else 1.0
        )
        # (#170) …and WHICH compartment that divisor is, when the size is a
        # writable parameter, so the reported column follows a `set_param` instead
        # of staying at the volume this load saw. This is the one place the
        # amount→concentration conversion lives entirely on the Python side (a
        # report-time rescale, not emitted math), so it is the one place
        # `refresh_compartment_volume_state` cannot reach — and it was the last
        # site where a write did not reproduce a rebuild: the reported column came
        # out scaled by exactly V_new/V_load (0.4 on a 2.5x write, on 15 corpus
        # models). Empty for a vstatic compartment, whose divide must stay at the
        # load-time value and whose size is refused anyway.
        _vdiv_comp = species_comp.get(sid, "")
        vdiv_param = (
            _safe_name(_vdiv_comp)
            if species_hosu.get(sid, False) and _vdiv_comp in live_volume_param_comps
            else ""
        )
        # The 4th element is APPENDED only when there is a live size to name, so a
        # model with no writable compartment keeps the 3-tuple its consumers (and
        # their tests) already pin — the same widen-by-appending the 2-tuple → 3-tuple
        # step used, and why the reader below is length-guarded rather than unpacking.
        ar_report_map[safe] = (kind, safe, vdiv, vdiv_param) if vdiv_param else (kind, safe, vdiv)
    model._ar_report_map = ar_report_map

    # Variable-volume concentration report map (GH #85). bngsim stores every
    # species as ``amount / V_static`` using the compartment size at load
    # (``volume_factor``), which equals the true concentration ONLY while the
    # compartment is static. For a species in a compartment driven by a RATE
    # RULE the live size V(t) diverges continuously from V_static, so the stored
    # ``amount / V_static`` is off by exactly ``V_static / V_live(t)`` — the
    # integrated amounts are right (the dynamics already divide Functional rates
    # by the live compartment symbol, the #74 ``_vd_<rid>_varvol`` path), only
    # the reported concentration is stale. Record species_safe → compartment_safe
    # so Simulator._apply_varvol_conc_map can rescale the reported concentration
    # by V_static / V_live(t) at report time, reading V_live(t) from the
    # compartment's own promoted-species column. Empty (⇒ no-op, byte-identical
    # reporting) for every model with no rate-rule compartment.
    #
    # Scoped to rate-rule compartments ONLY, NOT the full ``variable_comps``
    # (which also includes event-resized compartments): an event resize injects
    # an explicit per-species ``V_old/V_new`` concentration rescale at the event
    # (step 10 / #74), so those species are ALREADY stored as ``amount/V_live``
    # and a second rescale here would double-correct them (regressing
    # BIOMD0000000338/339, whose amount-conservation invariant the
    # test_sbml_compartment_resize_event suite locks). A rate rule has no such
    # injected rescale, so its species need this report-time divide.
    #
    # Scoped further to hasOnlySubstanceUnits=true (amount-valued) species: the
    # rescale ``conc = stored·V_static/V_live`` is exact iff the amount invariant
    # ``stored·V_static == amount`` holds, which is guaranteed by construction
    # for an amount-valued species (the engine reads it as ``stored·V_static``).
    # A hOSU=false species is integrated in *concentration* space, so its
    # concentration column needs no rescale — but its bare-id *amount* report
    # does (``conc·V_live``, not ``conc·V_static``); that goes through the
    # separate ``_varvol_amount_map`` below (GH #86). ``rate_rule_comps`` is
    # defined once at section 1.
    varvol_conc_map: dict[str, str] = {}
    if rate_rule_comps:
        for sid, cid in species_comp.items():
            if cid in rate_rule_comps and sid in species_idx and species_hosu.get(sid, False):
                varvol_conc_map[_safe_name(sid)] = _safe_name(cid)
    model._varvol_conc_map = varvol_conc_map

    # Variable-volume *amount* report map (GH #86) — the hOSU=false counterpart.
    # A hOSU=false species in a rate-rule compartment is integrated in
    # concentration space WITH its dilution term (section 8b), so its stored
    # concentration ``[S](t)`` is already correct and must NOT be rescaled. Only
    # the bare-id amount selector in Result.as_roadrunner needs the live volume:
    # ``amount = [S]·V_live(t)``, not the stale ``[S]·V_static``. Simulator's
    # _apply_varvol_conc_map records the live-volume column for these species
    # without touching the concentration column. Same gate as the dilution
    # emission (hOSU=false, rate-rule compartment, not an AR/rate-rule target);
    # empty for every corpus / static / amount-valued-only model.
    varvol_amount_map: dict[str, str] = {}
    if rate_rule_comps:
        for sid, cid in species_comp.items():
            if (
                cid in rate_rule_comps
                and sid in species_idx
                and not species_hosu.get(sid, False)
                and sid not in rate_rule_targets
                and sid not in assignment_targets
            ):
                varvol_amount_map[_safe_name(sid)] = _safe_name(cid)
    model._varvol_amount_map = varvol_amount_map

    # (#87) Variable-volume concentration report map for ASSIGNMENT-RULE
    # compartments (e.g. ``tV := mV + dV``). An amount-valued (hOSU=true) species
    # in such a compartment is stored as ``amount / V_static`` and — after the
    # dynamics fix (section 1 / 9 / 10 use the numeric V_static) — its integrated
    # amount is correct. But the reported *concentration* must be ``amount /
    # V_live(t)``, and V_live = the AR compartment's live value, which diverges
    # from V_static as the compartment grows. The live value is recorded as the
    # compartment's own ASSIGNMENT-RULE EXPRESSION column (step 6 emits a function
    # named after the compartment), so unlike the rate-rule map (#85) — which
    # reads a promoted-species column — this one reads an *expression* column.
    # Covers BOTH the plain amount-valued species (column holds ``amount/V_static``
    # = stored) AND the AR-target species (``_apply_ar_report_map`` already set
    # their column to ``amount/V_static`` via vdiv=V_static): after those passes
    # every species in the AR compartment holds ``amount/V_static``, and a single
    # uniform ``× V_static / V_live(t)`` yields the correct concentration.
    # Maps species_safe → (comp_expr_name, V_static). Empty (no-op) for every
    # model with no assignment-rule compartment.
    varvol_ar_conc_map: dict[str, tuple[str, float]] = {}
    if ar_comp_targets:
        for sid, cid in species_comp.items():
            if cid in ar_comp_targets and sid in species_idx and species_hosu.get(sid, False):
                varvol_ar_conc_map[_safe_name(sid)] = (
                    _safe_name(cid),
                    float(comp_volumes.get(cid, 1.0)),
                )
    model._varvol_ar_conc_map = varvol_ar_conc_map

    # (#234) Variable-volume *amount* report map for ASSIGNMENT-RULE compartments —
    # the hOSU=false counterpart of varvol_ar_conc_map. A hOSU=false species that
    # received the §8c dilution term is integrated in concentration space, so its
    # stored column ``[S](t)`` is already the live ``amount/V_live(t)`` and must NOT
    # be rescaled. Only the bare-id amount selector needs the live volume:
    # ``amount = [S]·V_live(t)``, read from the AR compartment's EXPRESSION column
    # (an AR compartment has no ODE-state column). Mirrors varvol_amount_map (#86)
    # but sources V_live from an expression rather than a promoted species. Keyed by
    # the exact species §8c emitted a dilution term for; empty for every model with
    # no time-varying AR compartment.
    varvol_ar_amount_map: dict[str, str] = {}
    for sid in ar_dilution_species:
        varvol_ar_amount_map[_safe_name(sid)] = _safe_name(species_comp[sid])
    model._varvol_ar_amount_map = varvol_ar_amount_map

    # (#728) The amount report of a concentration species that a rule sets. The
    # two maps above take their species from the dilution gate, which leaves
    # out a rule target, rightly: its rule states its derivative or its value.
    # Its amount is still [S]·V_live(t), and as_roadrunner reported [S]·V_static
    # (01198: S1 = 0.2 for 0.28). Its column holds the rule's concentration
    # under every method, so V_live(t) is all the report needs, read from the
    # compartment's promoted-species column (a rate rule), its expression
    # column (an assignment rule) or its observable column (an event resize; the
    # event-resize map below leaves such a species to this one).
    varvol_target_amount_map: dict[str, tuple[str, str]] = {}
    for sid, cid in species_comp.items():
        if (
            sid not in species_idx
            or species_hosu.get(sid, False)
            or (sid not in rate_rule_targets and sid not in assignment_targets)
        ):
            continue
        if cid in rate_rule_comps:
            varvol_target_amount_map[_safe_name(sid)] = ("species", _safe_name(cid))
        elif cid in ar_comp_targets:
            varvol_target_amount_map[_safe_name(sid)] = ("expression", _safe_name(cid))
        elif cid in event_resize_comps:
            varvol_target_amount_map[_safe_name(sid)] = ("observable", _safe_name(cid))
    model._varvol_target_amount_map = varvol_target_amount_map

    # (#131) Variable-volume concentration report map for EVENT-RESIZED
    # compartments (an event assignment writes the compartment's size). Unlike a
    # rate-rule compartment, an event resize injects a per-species ``V_old/V_new``
    # *concentration* rescale at the event (step 10 / #74) that is ``ode_only``,
    # so the reported quantity depends on both hOSU and method:
    #
    #   • hOSU=true species are stored as ``amount/V_static`` and are NOT touched
    #     by that conc rescale (their amount is conserved across the resize), so
    #     their reported concentration ``[H] = amount/V_live`` is stale under
    #     BOTH ODE and SSA — the bare amount ``H = stored·V_static`` is already
    #     correct via the volume factor.
    #   • hOSU=false species get the ODE conc rescale (stored becomes the live
    #     ``amount/V_live``, so ODE ``[S]`` is already right), but under SSA the
    #     rescale is skipped to preserve counts, leaving ``[S]`` stale.
    #
    # So the report-time concentration correction (``× V_static/V_live``) is
    # needed for every hOSU=true species (both methods) and for hOSU=false
    # species under SSA only. Simulator._apply_varvol_event_resize_map reads the
    # live volume from the compartment's same-named OBSERVABLE column (the
    # event-promoted compartment is hidden from species output per GH #71 but is
    # emitted as an observable) and applies the factor method-by-method. Maps
    # species_safe → (comp_obs_name, V_static, hOSU). Empty (no-op) for every
    # model with no event-resized compartment.
    #
    # A concentration species that a rule sets is no count, and its column holds
    # the rule's concentration under SSA too (its rescale is not ODE-only), so
    # this map leaves it to the rule-target map above. Read as a count, `S := 3`
    # reported [S] = 1 and S = 3 under SSA after `c = 3`, for 3 and 9.
    varvol_event_resize_map: dict[str, tuple[str, float, bool]] = {}
    if event_resize_comps:
        for sid, cid in species_comp.items():
            if _safe_name(sid) in varvol_target_amount_map:
                continue
            if cid in event_resize_comps and sid in species_idx:
                varvol_event_resize_map[_safe_name(sid)] = (
                    _safe_name(cid),
                    float(comp_volumes.get(cid, 1.0)),
                    bool(species_hosu.get(sid, False)),
                )
    model._varvol_event_resize_map = varvol_event_resize_map

    # End of the interpretation phase (everything from entry through the build);
    # jac derivation + codegen below are timed separately so the per-model
    # breakdown is non-overlapping (T-instr).
    model._interpret_sec = time.perf_counter() - _interpret_t0

    # ── 11. Analytical Functional Jacobian (GH #76) — DEFERRED (GH #145) ──
    # The analytical Jacobian is consumed only by ODE solves (CVODE's dense
    # Jacobian, the steady-state Newton solver, codegen's analytical-Jacobian
    # emitter), so its SymPy derivation is no longer run here at load. It is
    # deferred to the first ODE-solve setup (Simulator.__init__ →
    # Model.prepare_analytical_jacobian), so an SBML model run under SSA/PSA/
    # NFsim/RuleMonkey, or merely inspected, never pays for it. (All-Elementary
    # and Michaelis–Menten models carry the closed-form analytical Jacobian from
    # the C++ build regardless; only genuinely-Functional rate laws need this
    # SymPy step.) Eager escape hatch for A/B and safety: BNGSIM_EAGER_JACOBIAN=1
    # (or from_sbml(..., defer_jacobian=False)) restores derive-at-load — and
    # because the ≥256-species auto-codegen (step 12, formerly here) also moved
    # to ODE-solve setup, ordered after the attach, the "attach before codegen"
    # invariant holds in both the lazy and the eager path.
    from bngsim._jacobian import eager_jacobian_requested

    if eager_jacobian_requested():
        model.prepare_analytical_jacobian()

    return model


def _ast_to_exprtk_with_funcdefs(node, func_defs, local_params=None):
    """Convert AST to ExprTk, inlining SBML function definitions."""
    t = node.getType()

    # Check for user function call that matches a functionDefinition
    if t == libsbml.AST_FUNCTION:
        fname = node.getName()
        if fname in func_defs:
            param_names, body = func_defs[fname]
            # COPASI rateOf idiom (GH #106): emit the accessor for the call's
            # argument rather than inlining the funcDef's NaN body (which #92
            # would render as (0.0/0.0), permanently falsifying any trigger).
            if body is _RATEOF_FUNCDEF:
                return _rateof_call(node, local_params)
            nc = node.getNumChildren()
            # Substitute arguments into body
            subs = {}
            for j in range(min(nc, len(param_names))):
                arg_expr = _ast_to_exprtk_with_funcdefs(node.getChild(j), func_defs, local_params)
                subs[param_names[j]] = arg_expr
            return _substitute_ast(body, subs, func_defs, local_params)

    # Local parameter remapping
    if t == libsbml.AST_NAME and local_params:
        name = node.getName()
        if name and name in local_params:
            return _atomic(local_params[name])

    # Default: recurse through the general translator.
    return _ast_to_exprtk_recursive(node, func_defs, local_params)


def _ast_type_name(t):
    """Best-effort reverse lookup of a libsbml AST type int -> ``AST_*`` name.

    Only called on the error path (a translation failure), so the linear scan of
    libsbml's namespace is never a hot-path cost.
    """
    for nm in dir(libsbml):
        if nm.startswith("AST_") and getattr(libsbml, nm) == t:
            return nm
    return None


def _unsupported_ast_error(node, t):
    """Build the fail-closed error for an untranslatable math node (GH #97)."""
    from bngsim._exceptions import ModelError

    type_name = _ast_type_name(t)
    try:
        node_str = libsbml.formulaToL3String(node)
    except Exception:
        node_str = ""

    hint = ""
    if type_name and "DISTRIB" in type_name:
        hint = (
            " This is an SBML 'distrib' package random-draw csymbol, which has no "
            "deterministic translation; bngsim's ODE/SSA engine cannot evaluate it."
        )

    return ModelError(
        "Cannot translate SBML math to an ExprTk expression: unsupported AST node "
        f"type {t}"
        + (f" ({type_name})" if type_name else "")
        + (f" in '{node_str}'" if node_str else "")
        + "."
        + hint
    )


def _ast_to_exprtk_recursive(node, func_defs, local_params):
    """Convert a libsbml MathML AST node to an ExprTk expression string,
    supporting inlined function definitions and local-parameter remapping.

    This is the single live MathML→ExprTk translator: every rule, event,
    kinetic law, parameter, and initial-assignment expression reaches it via
    :func:`_ast_to_exprtk_with_funcdefs`. An unrecognised node fails closed
    (GH #97) rather than silently degrading to ``0``.
    """
    t = node.getType()

    # Leaf: number
    if t == libsbml.AST_INTEGER:
        v = node.getInteger()
        return f"({v})" if v < 0 else str(v)  # see _real_literal (GH #744)
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E):
        return _real_literal(node.getReal())
    if t == libsbml.AST_RATIONAL:
        return f"({node.getNumerator()}/{node.getDenominator()})"

    # Leaf: name
    if t == libsbml.AST_NAME:
        name = node.getName()
        if local_params and name in local_params:
            return _atomic(local_params[name])
        return _safe_name(name) if name else "0"
    if t == libsbml.AST_NAME_TIME:
        return "time()"

    # Constants
    if t == libsbml.AST_CONSTANT_PI:
        return "_pi"
    if t == libsbml.AST_CONSTANT_E:
        return "exp(1)"
    if t == libsbml.AST_CONSTANT_TRUE:
        return "1"
    if t == libsbml.AST_CONSTANT_FALSE:
        return "0"

    # Avogadro's number (AST type 307 in libsbml). SI-exact value — see the
    # _AVOGADRO module constant and the C++ ``_NA`` registration.
    if t == libsbml.AST_NAME_AVOGADRO:
        return _real_literal(_AVOGADRO)

    nc = node.getNumChildren()

    def _r(i):
        return _ast_to_exprtk_with_funcdefs(node.getChild(i), func_defs, local_params)

    # delay(x, d) → x  (ODE approximation: ignore delay)
    if t == libsbml.AST_FUNCTION_DELAY:
        if nc >= 1:
            return _r(0)
        return "0"

    # Unary minus
    if t == libsbml.AST_MINUS and nc == 1:
        return f"(-{_r(0)})"

    # Binary arithmetic. PLUS/TIMES iteratively flattened — see
    # _flatten_assoc_chain for the rationale.
    if t == libsbml.AST_PLUS:
        ops = _flatten_assoc_chain(node, libsbml.AST_PLUS)
        if not ops:  # empty <plus/> = additive identity (MathML L3v2)
            return "0"
        return (
            "("
            + "+".join(_ast_to_exprtk_with_funcdefs(o, func_defs, local_params) for o in ops)
            + ")"
        )
    if t == libsbml.AST_MINUS:
        return f"({_r(0)}-{_r(1)})"
    if t == libsbml.AST_TIMES:
        ops = _flatten_assoc_chain(node, libsbml.AST_TIMES)
        if not ops:  # empty <times/> = multiplicative identity (MathML L3v2)
            return "1"
        return (
            "("
            + "*".join(_ast_to_exprtk_with_funcdefs(o, func_defs, local_params) for o in ops)
            + ")"
        )
    if t == libsbml.AST_DIVIDE:
        return f"({_r(0)}/{_r(1)})"
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        return f"({_r(0)}^{_r(1)})"

    # Comparisons
    _cmp = {
        libsbml.AST_RELATIONAL_EQ: "==",
        libsbml.AST_RELATIONAL_NEQ: "!=",
        libsbml.AST_RELATIONAL_LT: "<",
        libsbml.AST_RELATIONAL_GT: ">",
        libsbml.AST_RELATIONAL_LEQ: "<=",
        libsbml.AST_RELATIONAL_GEQ: ">=",
    }
    if t in _cmp:
        op = _cmp[t]
        if nc < 2:
            # A 0/1-arg relational is vacuously true (matches _eval_ast_numeric).
            return "1"
        if nc == 2:
            return f"({_r(0)}{op}{_r(1)})"
        # N-ary relational (MathML allows 3+ args, GH #229): chained-pairwise.
        # (x1 op x2 op ... op xn) == (x1 op x2) and (x2 op x3) and …; neq
        # uniquely means "all pairwise distinct". Mirrors the numeric folder so
        # the ExprTk expression and the t=0 constant fold agree (the loader emits
        # a non-time AR target as both an expression column and a folded param).
        parts = [_r(i) for i in range(nc)]
        if t == libsbml.AST_RELATIONAL_NEQ:
            terms = [f"({parts[i]}!={parts[j]})" for i in range(nc) for j in range(i + 1, nc)]
        else:
            terms = [f"({parts[i]}{op}{parts[i + 1]})" for i in range(nc - 1)]
        return "(" + " and ".join(terms) + ")"

    # Logical (empty <and/> = true, empty <or/> = false — MathML identities)
    if t == libsbml.AST_LOGICAL_AND:
        if nc == 0:
            return "1"
        return "(" + " and ".join(_r(i) for i in range(nc)) + ")"
    if t == libsbml.AST_LOGICAL_OR:
        if nc == 0:
            return "0"
        return "(" + " or ".join(_r(i) for i in range(nc)) + ")"
    if t == libsbml.AST_LOGICAL_NOT:
        return f"(not({_r(0)}))"
    if t == libsbml.AST_LOGICAL_XOR:
        # n-ary xor = odd parity of the true operands. Boolean-normalise each
        # operand (``x != 0``) then left-fold ``!=``; a 1-child xor is just the
        # single operand's truth value. Mirrors the numeric folder.
        if nc == 0:
            return "0"
        terms = [f"(({_r(i)})!=0)" for i in range(nc)]
        result = terms[0]
        for term in terms[1:]:
            result = f"({result}!={term})"
        return result

    # Math functions — direct ExprTk equivalents
    _funcs = {
        libsbml.AST_FUNCTION_ABS: "abs",
        libsbml.AST_FUNCTION_CEILING: "ceil",
        libsbml.AST_FUNCTION_EXP: "exp",
        libsbml.AST_FUNCTION_FLOOR: "floor",
        libsbml.AST_FUNCTION_LN: "log",
        libsbml.AST_FUNCTION_LOG: "log10",
        libsbml.AST_FUNCTION_ROOT: "sqrt",
        libsbml.AST_FUNCTION_SIN: "sin",
        libsbml.AST_FUNCTION_COS: "cos",
        libsbml.AST_FUNCTION_TAN: "tan",
        libsbml.AST_FUNCTION_ARCSIN: "asin",
        libsbml.AST_FUNCTION_ARCCOS: "acos",
        libsbml.AST_FUNCTION_ARCTAN: "atan",
        libsbml.AST_FUNCTION_SINH: "sinh",
        libsbml.AST_FUNCTION_COSH: "cosh",
        libsbml.AST_FUNCTION_TANH: "tanh",
    }
    if t in _funcs:
        fname = _funcs[t]
        if t == libsbml.AST_FUNCTION_ROOT and nc == 2:
            return f"({_r(1)}^(1/{_r(0)}))"
        if t == libsbml.AST_FUNCTION_LOG and nc == 2:
            return f"(log({_r(1)})/log({_r(0)}))"
        args = ",".join(_r(i) for i in range(nc))
        return f"{fname}({args})"

    # Trig functions not directly in ExprTk — rewrite as identities
    if t == libsbml.AST_FUNCTION_SEC:
        return f"(1/cos({_r(0)}))"
    if t == libsbml.AST_FUNCTION_CSC:
        return f"(1/sin({_r(0)}))"
    if t == libsbml.AST_FUNCTION_COT:
        return f"(cos({_r(0)})/sin({_r(0)}))"
    if t == libsbml.AST_FUNCTION_SECH:
        return f"(1/cosh({_r(0)}))"
    if t == libsbml.AST_FUNCTION_CSCH:
        return f"(1/sinh({_r(0)}))"
    if t == libsbml.AST_FUNCTION_COTH:
        return f"(cosh({_r(0)})/sinh({_r(0)}))"

    # Inverse hyperbolic — log identities
    if t == libsbml.AST_FUNCTION_ARCSINH:
        x = _r(0)
        return f"(log(({x})+sqrt(({x})^2+1)))"
    if t == libsbml.AST_FUNCTION_ARCCOSH:
        x = _r(0)
        return f"(log(({x})+sqrt(({x})^2-1)))"
    if t == libsbml.AST_FUNCTION_ARCTANH:
        x = _r(0)
        return f"(0.5*log((1+({x}))/(1-({x}))))"
    # Inverse trig for cot/csc/sec
    if t == libsbml.AST_FUNCTION_ARCCOT:
        return f"(atan(1/({_r(0)})))"
    if t == libsbml.AST_FUNCTION_ARCCOTH:
        x = _r(0)
        return f"(0.5*log((({x})+1)/(({x})-1)))"
    if t == libsbml.AST_FUNCTION_ARCCSC:
        return f"(asin(1/({_r(0)})))"
    if t == libsbml.AST_FUNCTION_ARCCSCH:
        x = _r(0)
        return f"(log(1/({x})+sqrt(1/({x})^2+1)))"
    if t == libsbml.AST_FUNCTION_ARCSEC:
        return f"(acos(1/({_r(0)})))"
    if t == libsbml.AST_FUNCTION_ARCSECH:
        x = _r(0)
        return f"(log(1/({x})+sqrt(1/({x})^2-1)))"

    # Factorial — factorial(n) = Γ(n+1). ExprTk has no gamma; the C++ evaluator
    # registers ``tgamma`` (std::tgamma) and native codegen emits the C library
    # ``tgamma`` directly. (The old ``exp(lgamma(...))`` form failed: ExprTk has
    # no ``lgamma`` either, so every factorial model load-failed.)
    if t == libsbml.AST_FUNCTION_FACTORIAL:
        x = _r(0)
        return f"(tgamma(({x})+1))"

    # L3v2 MathML: max, min, quotient, rem, implies
    if t == libsbml.AST_FUNCTION_MAX:
        if nc == 2:
            return f"max({_r(0)},{_r(1)})"
        parts = [_r(i) for i in range(nc)]
        result = parts[0]
        for p in parts[1:]:
            result = f"max({result},{p})"
        return result
    if t == libsbml.AST_FUNCTION_MIN:
        if nc == 2:
            return f"min({_r(0)},{_r(1)})"
        parts = [_r(i) for i in range(nc)]
        result = parts[0]
        for p in parts[1:]:
            result = f"min({result},{p})"
        return result
    if t == libsbml.AST_FUNCTION_QUOTIENT:
        return f"(floor(({_r(0)})/({_r(1)})))"
    if t == libsbml.AST_FUNCTION_REM:
        a, b = _r(0), _r(1)
        return f"(({a})-({b})*floor(({a})/({b})))"
    if t == libsbml.AST_LOGICAL_IMPLIES:
        return f"((not({_r(0)})) or ({_r(1)}))"

    # Piecewise
    if t == libsbml.AST_FUNCTION_PIECEWISE:
        pieces = []
        otherwise = "0"
        j = 0
        while j < nc - 1:
            val = _r(j)
            cond = _r(j + 1)
            pieces.append((cond, val))
            j += 2
        if nc % 2 == 1:
            otherwise = _r(nc - 1)
        result = otherwise
        for cond, val in reversed(pieces):
            result = f"if({cond},{val},{result})"
        return result

    # User function → inline if defined
    if t == libsbml.AST_FUNCTION:
        fname = node.getName()
        if fname in func_defs:
            param_names, body = func_defs[fname]
            # COPASI rateOf idiom (GH #106): emit the accessor for the call's
            # argument rather than inlining the funcDef's NaN body.
            if body is _RATEOF_FUNCDEF:
                return _rateof_call(node, local_params)
            subs = {}
            for j in range(min(nc, len(param_names))):
                subs[param_names[j]] = _r(j)
            return _substitute_ast(body, subs, func_defs, local_params)
        # Unknown function — emit as variable reference
        return _safe_name(fname)

    if t == libsbml.AST_LAMBDA:
        return _r(nc - 1)

    # rateOf(x) csymbol (GH #106): instantaneous dx/dt → per-species accessor.
    if t == libsbml.AST_FUNCTION_RATE_OF:
        return _rateof_call(node, local_params)

    # Fail closed (GH #97). An unrecognised math node previously fell through to
    # `return "0"` with only a warning — a silent, wrong RHS that loaded fine and
    # mis-simulated. Raise a clear, actionable load error instead. The corpus
    # survey (BioModels + benchmarks: 0 hits; SBML Test Suite: only `distrib`
    # random-draw csymbols) shows every node reaching here is a genuinely
    # unsupported construct, so this is a loud reject, not a regression.
    raise _unsupported_ast_error(node, t)


def _substitute_ast(node, subs, func_defs, local_params):
    """Substitute names in an AST body with expression strings."""
    t = node.getType()
    if t == libsbml.AST_NAME:
        name = node.getName()
        if name in subs:
            return f"({subs[name]})"
        if local_params and name in local_params:
            return _atomic(local_params[name])
        return _safe_name(name) if name else "0"

    # For non-name nodes, convert normally but with subs as local_params
    merged = dict(local_params or {})
    merged.update(subs)
    return _ast_to_exprtk_recursive(node, func_defs, merged)
