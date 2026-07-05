"""bngsim._sbml_loader — SBML model loader via libsbml.

Core loader that converts SBML (from file, string, or via Antimony)
into a BNGsim NetworkModel using ModelBuilder.  Handles all SBML
semantics correctly: initialAmount vs initialConcentration,
hasOnlySubstanceUnits, compartment volumes, assignment/rate rules,
initialAssignment evaluation, MathML→ExprTk expression translation.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import libsbml

from bngsim import _sbml_unsupported

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
        return repr(v)
    if _math.isnan(v):
        return "(0.0/0.0)"
    return "(1.0/0.0)" if v > 0.0 else "(-1.0/0.0)"


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


def _rateof_token(species_name: str) -> str:
    """Accessor token for ``rateOf(<species_name>)``. Must match the C++
    accessor name ``register_rateof_accessors`` binds — the ``rate_of__``
    prefix plus the species' ``_safe_name`` (the name it is registered under)."""
    return _RATEOF_PREFIX + _safe_name(species_name)


def _rateof_call(node: libsbml.ASTNode, local_params: dict | None = None) -> str:
    """Emit the accessor for a ``rateOf(...)`` call (csymbol or COPASI idiom).

    Only rateOf of a bare symbol — a species, or a rate-rule-promoted
    parameter/compartment that becomes a species — is representable as an
    accessor. A non-name argument has no species index to bind to, so warn and
    emit ``0`` rather than a silently-wrong derivative (no corpus model hits
    this; every rateOf argument across the BioModels SBML corpus is a species).

    A kinetic-law ``<localParameter>`` is constant by SBML definition, so the
    rate of change of a local-parameter argument is ``0`` (suite 01459/01460).
    It also has no species index — emitting the accessor would either fail to
    compile (no global of that name) or silently bind to a same-named global,
    so the local-scope check both fixes the value and avoids the mis-bind.
    """
    if node.getNumChildren() >= 1:
        arg = node.getChild(0)
        if arg.getType() == libsbml.AST_NAME and arg.getName():
            if local_params and arg.getName() in local_params:
                return "0"
            return _rateof_token(arg.getName())
    logger.warning("rateOf() with a non-symbol argument is unsupported; emitting 0")
    return "0"


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
    assignments — plus non-idiom funcDef bodies, since a rateOf csymbol nested
    in a funcDef body is emitted as an accessor when that funcDef is inlined."""

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
# We refuse all three loud-by-default, mirroring the GH #94 unset-parameter
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
    integrated system: rules, kinetic laws, initial assignments, and every
    event trigger / delay / priority / assignment. ``location`` is a short
    human-readable tag used in diagnostics."""
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
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl is not None and kl.isSetMath():
            yield kl.getMath(), f"reaction:{rxn.getId()}"
    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        if ia.isSetMath():
            yield ia.getMath(), f"initialAssignment:{ia.getSymbol()}"
    for i in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(i)
        eid = ev.getId() or f"event{i}"
        trig = ev.getTrigger()
        if trig is not None and trig.isSetMath():
            yield trig.getMath(), f"event:{eid}:trigger"
        delay = ev.getDelay()
        if delay is not None and delay.isSetMath():
            yield delay.getMath(), f"event:{eid}:delay"
        prio = ev.getPriority()
        if prio is not None and prio.isSetMath():
            yield prio.getMath(), f"event:{eid}:priority"
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            if ea.isSetMath():
                yield ea.getMath(), f"event:{eid}:assignment:{ea.getVariable()}"


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


def _collect_time_discontinuity_conditions(
    node: libsbml.ASTNode, func_defs: dict, out: list
) -> None:
    """Walk an AST and append ExprTk condition strings for every inequality
    comparing ``time`` against a non-time expression (a discontinuity edge).

    n-ary relationals (``a < b < c``) are split into the consecutive pairs
    MathML defines them as, so each emitted condition is a single
    monotonic-in-time threshold — exactly what CVODE root-finding brackets
    reliably regardless of step size. Recurses through piecewise / and / or /
    nested structure to reach conditions at any depth.
    """
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
    # Walk the whole subtree iteratively (GH #111) so a deep observable-sum
    # rule can't overflow the stack; emit one condition per inequality found.
    for n in _iter_ast_subtree(node):
        t = n.getType()
        nc = n.getNumChildren()
        if t in _INEQ_RELATIONAL_OPS and nc >= 2:
            op = _INEQ_RELATIONAL_OPS[t]
            for i in range(nc - 1):
                a, b = n.getChild(i), n.getChild(i + 1)
                # Exactly one side time-dependent ⇒ a clean `time vs threshold`
                # edge. (Both-sides-time, e.g. `time < 2*time`, is not a fixed
                # crossing and is skipped; neither-side is not time-dependent.)
                if _ast_references_time(a) != _ast_references_time(b):
                    lhs = _ast_to_exprtk_with_funcdefs(a, func_defs)
                    rhs = _ast_to_exprtk_with_funcdefs(b, func_defs)
                    out.append(f"({lhs}{op}{rhs})")


def _collect_relational_edge_conditions(node: libsbml.ASTNode, func_defs: dict, out: list) -> None:
    """Append one root condition for each relational sub-expression in *node*.

    Event trigger roots are registered as the full Boolean trigger. That misses
    narrow compound pulses such as ``(S > lo) && (S < hi)`` when one integrator
    step sees the trigger false at both endpoints. Registering each relational
    edge separately forces a stop at the component threshold, where the normal
    event re-check can queue the event.
    """
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
                lhs = _ast_to_exprtk_with_funcdefs(n.getChild(i), func_defs)
                rhs = _ast_to_exprtk_with_funcdefs(n.getChild(i + 1), func_defs)
                out.append(f"({lhs}{op}{rhs})")


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
) -> None:
    """Append root conditions for bounded state-dependent floor/ceiling jumps."""
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
    stack: list[tuple[libsbml.ASTNode, dict, dict, set, set]] = [(node, {}, {}, set(), set())]
    while stack:
        n, local_expr, local_values, local_time_dependent, seen_funcs = stack.pop()
        ntype = n.getType()

        if ntype in disc_funcs and n.getNumChildren() >= 1:
            arg = n.getChild(0)
            names = _ast_name_set(arg)
            if names and not (
                _ast_references_time(arg)
                or bool(names & time_dependent_names)
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
                        if val is None:
                            next_values.pop(pname, None)
                        else:
                            next_values[pname] = val
                        if (
                            _ast_references_time(child)
                            or bool(child_names & time_dependent_names)
                            or bool(child_names & local_time_dependent)
                        ):
                            next_td.add(pname)
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
            if cond is not None and cond != 0:
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
            # is unknown at fold time, so return NaN as it did pre-#106 (matches
            # the funcDef's NaN body — no numerically-folded context relied on it).
            if body is _RATEOF_FUNCDEF:
                if rateof_resolver is not None and nc >= 1:
                    arg = node.getChild(0)
                    if arg.getType() == libsbml.AST_NAME and arg.getName():
                        r = rateof_resolver(arg.getName(), ctx)
                        if r is not None:
                            return r
                return float("nan")
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
      * **reaction-driven species**: ``dAmount/dt = Σ net_stoich·kineticLaw``
        over every reaction, in the species' reporting units — amount-rate for a
        ``hasOnlySubstanceUnits`` species, else concentration-rate (``÷V``). This
        mirrors ``current_derivs`` (which holds d[conc]/dt) and the sub-cluster-3
        ``report_rateof_amount`` ×V convention (01251/01252/01254, all V=1).
      * **boundary / constant species** with no rate rule: not changed by
        reactions ⇒ ``0``.

    Returns ``None`` (fold fails, target keeps its declared value) when the
    derivative can't be evaluated — an unknown symbol, an unsupported kinetic
    law, or a rateOf of something with no rate rule and no reaction.
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

    def resolver(name, ctx):
        # 1) rate-rule target — dx/dt is the rule's RHS, in the target's units.
        if name in rate_rule_math:
            return _eval_ast_numeric(rate_rule_math[name], ctx, func_defs)
        sp = species_by_id.get(name)
        if sp is None:
            # A parameter / compartment with no rate rule is constant ⇒ 0.
            return 0.0
        # 2) species — boundary or constant species is held by reactions ⇒ 0.
        if sp.getBoundaryCondition() or sp.getConstant():
            return 0.0
        vol = ctx.get(sp.getCompartment(), 1.0)
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
            law_ctx = ctx
            n_lp = kl.getNumLocalParameters() if hasattr(kl, "getNumLocalParameters") else 0
            if n_lp:
                law_ctx = dict(ctx)
                for k in range(n_lp):
                    lp = kl.getLocalParameter(k)
                    if lp.isSetValue():
                        law_ctx[lp.getId()] = lp.getValue()
            v = _eval_ast_numeric(kl.getMath(), law_ctx, func_defs)
            if v is None:
                return None
            amt_rate += net * v
        # ``current_derivs`` is d[conc]/dt; rateOf reports amount-rate for a hOSU
        # species (sub-cluster 3) and concentration-rate otherwise.
        if sp.getHasOnlySubstanceUnits():
            return amt_rate
        return amt_rate / vol if vol else None

    return resolver


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


def _periodic_time_disc_max_step(sbml_model, func_defs, base_ctx):
    """Recommend a ``max_step_size`` that resolves periodic floor/modulo dosing
    discontinuities in the ODE RHS, or ``None`` when the model has none.

    See the module comment above. The value is the narrowest discontinuity
    feature (dose-window width) over a representative horizon, divided by a
    safety factor so at least one integrator step lands inside every window.
    The narrowest feature is found numerically from the schedule's *structural*
    signature — the integer value of each floor/ceil/modulo and the active branch
    of each piecewise — which changes ONLY at a true edge (continuous values like
    `rem_time` move every sample and would mask the spacing).
    """
    global _DISC_FUNCS
    if _DISC_FUNCS is None:
        _DISC_FUNCS = {
            libsbml.AST_FUNCTION_FLOOR,
            libsbml.AST_FUNCTION_CEILING,
            libsbml.AST_FUNCTION_REM,
        }

    # Assignment rules (var → math) and the maths that ARE the ODE RHS.
    asg_math: dict[str, libsbml.ASTNode] = {}
    rhs_maths: list[libsbml.ASTNode] = []
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        m = rule.getMath()
        if m is None:
            continue
        if rule.isAssignment():
            asg_math[rule.getVariable()] = m
        elif rule.isRate():
            rhs_maths.append(m)
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl and kl.getMath():
            rhs_maths.append(kl.getMath())
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
    work = [n for m in rhs_maths for n in _ast_name_set(m)]
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
    relevant = list(rhs_maths) + [asg_math[v] for v in feed if v in asg_math]
    disc_nodes: list[tuple[int, libsbml.ASTNode]] = []  # (kind, node)
    piecewise_nodes: list[libsbml.ASTNode] = []
    has_periodic = False
    for m in relevant:
        for n in _iter_ast_subtree(m):
            ntype = n.getType()
            if ntype in _DISC_FUNCS:
                arg_td = _ast_references_time(n) or bool(_ast_name_set(n) & td)
                disc_nodes.append((ntype, n))
                if arg_td:
                    has_periodic = True
            elif ntype == libsbml.AST_FUNCTION_PIECEWISE:
                piecewise_nodes.append(n)
    if not has_periodic:
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
        for kind, n in disc_nodes:
            if kind == libsbml.AST_FUNCTION_REM:
                a = _eval_ast_numeric(n.getChild(0), ctx, func_defs, time_value=t)
                b = _eval_ast_numeric(n.getChild(1), ctx, func_defs, time_value=t)
                sig.append(None if (a is None or b is None or b == 0) else _math.floor(a / b))
            else:
                a = _eval_ast_numeric(n.getChild(0), ctx, func_defs, time_value=t)
                fn = _math.floor if kind == libsbml.AST_FUNCTION_FLOOR else _math.ceil
                sig.append(None if a is None else fn(a))
        for n in piecewise_nodes:
            nc = n.getNumChildren()
            branch = nc  # default: "otherwise"
            i = 0
            idx = 0
            while i < nc - 1:
                cond = _eval_ast_numeric(n.getChild(i + 1), ctx, func_defs, time_value=t)
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
    def _slope(arg_node) -> float:
        pts: list[tuple[float, float]] = []
        for k in range(1, 9):
            tt = 0.05 * k
            v = _eval_ast_numeric(arg_node, dict(base_ctx), func_defs, time_value=tt)
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
    for kind, n in disc_nodes:
        if kind == libsbml.AST_FUNCTION_REM:
            # rem(a,b) jumps when a/b crosses an integer; slope ≈ |a'|/|b|.
            a_s = _slope(n.getChild(0))
            b = _eval_ast_numeric(n.getChild(1), dict(base_ctx), func_defs, time_value=0.0)
            s = a_s / abs(b) if b else 0.0
        else:
            s = _slope(n.getChild(0))
        if s > 0.0:
            periods.append(1.0 / s)
    longest = max(periods) if periods else 1.0
    shortest = min(periods) if periods else 1.0
    # A few of the longest period covers every phase; cap the window so a model
    # with a very long period stays cheap (the narrowest feature is periodic, so
    # a few cycles still contain it).
    horizon = min(max(2.5 * longest, 4.0 * shortest, 4.0), 128.0)

    # Scan the structural signature for edges. Samples at cell centres so the
    # scan never lands exactly on an integer edge — a measure-zero `0 < frac` flip
    # would forge spurious adjacent edges. Resolution targets ~1/100 of the
    # shortest period so a dose window down to a few percent of a cycle is caught
    # in a coarse cell; npts is bounded to keep this pure-Python scan sub-second.
    npts = int(min(max(horizon / (shortest / 100.0), 2000.0), 16000.0))
    h = horizon / npts
    centres = [(k + 0.5) * h for k in range(npts)]
    sigs = [_structural_signature(t) for t in centres]

    # Bisect each coarse sign change to its true edge time so the measured spacing
    # is the real window width, not the grid pitch. Without this, several floor /
    # piecewise components flipping in adjacent cells read as edges a fraction of a
    # cell apart and would shrink max_step needlessly (or, pathologically, toward
    # zero). Edges nearer than a tight epsilon are the same physical instant (two
    # components switching together) and collapse to one.
    edges = []
    for k in range(1, npts):
        if sigs[k] == sigs[k - 1]:
            continue
        lo, hi = centres[k - 1], centres[k]
        s_lo = sigs[k - 1]
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if _structural_signature(mid) == s_lo:
                lo = mid
            else:
                hi = mid
        edges.append(hi)
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


def load_sbml(path: str | Path):
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
    if _doc_uses_comp(doc):
        _flatten_comp(doc, base_path=str(path.parent))
    parse_sec = time.perf_counter() - t0
    model = _build_model_from_sbml_doc(doc)
    model._libsbml_parse_sec = parse_sec
    return model


def load_sbml_string(text: str):
    """Load an SBML model from an XML string."""
    # libSBML parse phase (shared C++ core) — readSBML* + doc-level error check.
    # Timed here and stamped on the model; the doc → _core interpretation and the
    # analytical-Jacobian derivation are timed separately inside _build (T-instr).
    t0 = time.perf_counter()
    doc = libsbml.readSBMLFromString(text)
    _check_sbml_errors(doc, "<string>")
    if _doc_uses_comp(doc):
        # No file context for a string load, so ExternalModelDefinitions cannot
        # be resolved; in-document ModelDefinitions/Submodels flatten fine.
        _flatten_comp(doc, base_path=None)
    parse_sec = time.perf_counter() - t0
    model = _build_model_from_sbml_doc(doc)
    model._libsbml_parse_sec = parse_sec
    return model


def load_antimony_via_sbml(path: str | Path):
    """Load an Antimony .ant file by converting to SBML first."""
    import antimony as ant

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

    return load_sbml_string(sbml_str)


def load_antimony_string_via_sbml(text: str):
    """Load an Antimony model string by converting to SBML first."""
    import antimony as ant

    ant.clearPreviousLoads()
    ret = ant.loadString(text)
    if ret == -1:
        err = ant.getLastError()
        raise RuntimeError(f"Failed to parse Antimony string: {err}")

    mod = ant.getModuleNames()[-1]
    sbml_str = ant.getSBMLString(mod)
    if not sbml_str or len(sbml_str) < 10:
        raise RuntimeError("Antimony produced empty SBML")

    return load_sbml_string(sbml_str)


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
            if r == int(r):
                n_exp = int(r)
        if n_exp is None or n_exp < 1:
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
                if r == int(r):
                    n_exp = int(r)
            if n_exp is None or n_exp < 1:
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
        local_param_map,
        local_param_values,
        prefix_factors=prefix,
    )
    if rev is None:
        return None
    return fwd, rev


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

    if not rate_param_components or numeric_const <= 0:
        return None

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
    # Scoped to all-hOSU=false species; hOSU=true cross-compartment shapes (whose
    # amount bookkeeping differs) stay refused (varvol_non_mass_action), and the §9
    # certification is guarded to IRREVERSIBLE only. Record the variable-volume
    # storage compartments so §9 can certify + correct the emitted reaction.
    _storage_comps = {species_comp[s] for s in rxn_species}
    _varvol_storage = _storage_comps & (set(rate_rule_targets) | set(event_promoted_params))
    if (
        len(_storage_comps) > 1
        and _varvol_storage
        and all(not species_hosu.get(s, False) for s in rxn_species)
    ):
        if varvol_reject is not None:
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

    varvol_comp_ids = (set(rate_rule_targets) | set(event_promoted_params)) & comp_ids
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

    # Amount restoration for hOSU=true reactants is no longer folded into ``sf``
    # here (GH #75). Each species carries an ``amount_valued`` flag (set by the
    # loader from hasOnlySubstanceUnits); the engine's Elementary species factor
    # reads such a species as its amount (stored × V_c) directly, so the former
    # ``hosu_numerator = Π_{i hOSU} V_c(i)^mult_i`` term is supplied uniformly in
    # ``compute_rxn_rate``/``ssa_propensity`` instead of being pre-baked three
    # different ways in the loader. ODE is byte-identical (the same Π V_c reaches
    # the accumulation); SSA is byte-identical for first-order hOSU and uses the
    # physically correct population falling factorial for higher order.
    sf = numeric_const * kl_volume_product / v_common

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
    return (
        rate_param_components,
        float(sf),
        reactant_indices,
        product_indices,
        dict(compartment_factors),
    )


def _collect_local_params(kl, rid, builder):
    """Collect local parameters from a KineticLaw, handling both L2 and L3.

    SBML L2 stores local parameters under kl.getParameter(j) (accessed
    via getNumParameters()), while L3 uses kl.getLocalParameter(j)
    (accessed via getNumLocalParameters()).  963/978 BioModels are L2.

    Returns ``(local_param_map, local_param_values)`` — the first maps
    original_id → mangled_id for expression rewriting, the second maps
    mangled_id → numeric value (used by the mass-action classifier when
    a kinetic-law local parameter participates in a synthesized derived
    rate constant).
    """
    local_param_map = {}
    local_param_values: dict[str, float] = {}
    n_lp_l3 = kl.getNumLocalParameters()
    if n_lp_l3 > 0:
        # SBML L3 path (or libsbml's L2 backwards-compat wrapper)
        for j in range(n_lp_l3):
            lp = kl.getLocalParameter(j)
            orig_id = lp.getId()
            lpid = f"_lp_{rid}_{orig_id}"
            val = lp.getValue()
            builder.add_parameter(lpid, val)
            local_param_map[orig_id] = lpid
            local_param_values[lpid] = val
    else:
        # SBML L2 path: local params stored under KineticLaw directly
        n_lp_l2 = kl.getNumParameters()
        for j in range(n_lp_l2):
            lp = kl.getParameter(j)
            orig_id = lp.getId()
            lpid = f"_lp_{rid}_{orig_id}"
            val = lp.getValue() if lp.isSetValue() else 0.0
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
    # the loader saw no SSA-incompatible constructs. See
    # dev/plans/SBML_SSA_SUPPORT_PLAN.md Phase 3.
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
        eval_ctx[c.getId()] = c.getSize() if c.isSetSize() else 1.0
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
    for j in range(sbml_model.getNumReactions()):
        _rxn = sbml_model.getReaction(j)
        _rid = _rxn.getId()
        if not _rid or _rid in eval_ctx:
            continue
        _kl = _rxn.getKineticLaw()
        if _kl is None or not _kl.isSetMath():
            continue
        _law_ctx = eval_ctx
        _n_lp = _kl.getNumLocalParameters() if hasattr(_kl, "getNumLocalParameters") else 0
        if _n_lp:
            _law_ctx = dict(eval_ctx)
            for k in range(_n_lp):
                _lp = _kl.getLocalParameter(k)
                if _lp.isSetValue():
                    _law_ctx[_lp.getId()] = _lp.getValue()
        _rv = _eval_ast_numeric(_kl.getMath(), _law_ctx, func_defs)
        if _rv is not None and _math.isfinite(_rv):
            eval_ctx[_rid] = _rv

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
    if sbml_model.getNumInitialAssignments() > 0:
        for _pass in range(10):
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
            if not changed:
                break

    # ── C2: reject missing-value parameters that are actually consumed (#94) ──
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
    _unset_param_ids = {
        pid
        for pid in _all_param_ids
        if not sbml_model.getParameter(pid).isSetValue()
        and pid not in ia_values
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
            if kl is not None:
                _scan(kl.getMath())
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

            from bngsim._exceptions import ModelError

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
                raise ModelError(
                    f"{lead} kineticLaw / rule / event / initialAssignment "
                    "expression in the model. The model is under-specified: "
                    "bngsim refuses to default the value to 0.0, which would "
                    "silently zero out any expression that consumes it and "
                    "return a wrong-but-plausible trajectory (RoadRunner refuses "
                    "the same model). Set the parameter value or add an "
                    "initialAssignment. To restore the legacy lenient "
                    "default-to-0 behavior, set BNGSIM_ALLOW_UNSET_PARAMS=1."
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
    # assignment-rule compartments for state-volume/rateOf paths, but reaction
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

    for i in range(sbml_model.getNumCompartments()):
        c = sbml_model.getCompartment(i)
        cid = c.getId()
        vol = c.getSize() if c.isSetSize() else 1.0
        # Override with initialAssignment value if available
        if cid in ia_values:
            vol = ia_values[cid]
        comp_volumes[cid] = vol
        # Skip adding as a builder parameter if the compartment will be
        # promoted to a species — by a rate rule (step 8) or by an event
        # assignment target (step 10). Otherwise both adds collide on the
        # same ExprTk symbol name.
        if cid in rate_rule_targets or cid in event_promoted_params:
            continue
        builder.add_parameter(_safe_name(cid), vol)

    # ── 2. Parameters ─────────────────────────────────────────────────
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
        builder.add_parameter(_safe_name(pid), val)

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
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        # A no-MathML assignmentRule is a no-op: it does not define its target
        # at runtime, so the target should keep its declared/initial-assigned
        # value and stay bound through the ordinary species/parameter/
        # speciesReference path (GH #243).
        if rule.isAssignment() and rule.getMath() is not None:
            assignment_targets.add(rule.getVariable())

    # Detect species whose initialAssignment is a single <ci> referencing a
    # model parameter: these get registered as species_ic_param_refs so
    # CVODES forward-sensitivity can seed dY_i(0)/dp_k = 1 when p_k is
    # requested via sensitivity_params. Mirrors what the .net loader does
    # for "begin species" entries with a parameter-name IC. We only
    # recognize the trivial single-symbol case; compound expressions like
    # ``2*init_X + offset`` would need a chain rule that the codegen sens
    # path doesn't currently support for IC params.
    _param_ids = {sbml_model.getParameter(j).getId() for j in range(sbml_model.getNumParameters())}
    _species_ids_set = {
        sbml_model.getSpecies(j).getId() for j in range(sbml_model.getNumSpecies())
    }
    ia_single_param_ref: dict[str, str] = {}
    for j in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(j)
        sym = ia.getSymbol()
        if sym not in _species_ids_set:
            continue
        math = ia.getMath()
        if math is None:
            continue
        # libsbml: AST_NAME is the only node type for a bare <ci>. Reject
        # operators, numbers, function applications, and AST_NAME_TIME etc.
        if math.getNumChildren() == 0 and math.getType() == libsbml.AST_NAME:
            ref = math.getName()
            if ref in _param_ids:
                ia_single_param_ref[sym] = ref

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

    for i in range(sbml_model.getNumSpecies()):
        sp = sbml_model.getSpecies(i)
        sid = sp.getId()
        comp = sp.getCompartment()
        hosu = sp.getHasOnlySubstanceUnits()
        is_boundary = sp.getBoundaryCondition()
        is_const = sp.getConstant()

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
            else:
                ic = ia_values[sid]
        elif sp.isSetInitialConcentration():
            ic = sp.getInitialConcentration()
        elif sp.isSetInitialAmount():
            amt = sp.getInitialAmount()
            vol = comp_volumes.get(comp, 1.0)
            ic = amt / vol if vol != 0 else amt
        else:
            ic = 0.0

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

        # (GH #231 sub-cluster 3 / 01463) A hasOnlySubstanceUnits=true species's
        # symbol denotes its amount, so its rateOf csymbol is the amount-rate. The
        # engine stores amount/V_static (rescaling to live volume separately), so the
        # rateOf buffer holds d(amount)/dt / V_static and ×volume_factor recovers the
        # amount-rate — for a CONSTANT-volume compartment (01455/01457) AND a
        # VARIABLE-volume compartment promoted to a state (rate-rule / event-resized,
        # 01463), with no conc·V̇ correction. Flag the species so the engine (and
        # codegen) scale its rate_of__ accessor by volume_factor. AR-compartment
        # volumes are not integrator states, so a hOSU species there stays on the
        # unscaled d(conc)/dt (excluded — no rateOf case there is in the suite).
        if hosu and (comp not in vstatic_divide_comps or comp in variable_comps):
            builder.set_species_rateof_amount(idx)

        # If the species's IC was set by an initialAssignment that's a
        # single <ci> referencing a parameter, register the link so CVODES
        # forward sensitivity will seed yS for that parameter.
        if sid in ia_single_param_ref:
            builder.add_species_param_ref(idx, _safe_name(ia_single_param_ref[sid]))

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
    # involved species — exact, and correct for ODE and SSA alike. In section 7
    # this folds into the mass-action statistical factor ``sf``. Genuinely
    # mixed-cf-per-reaction and conversion-factored non-mass-action laws can't be
    # expressed as a single rate scale and are refused loudly there (never
    # silently mis-integrated). Models with no conversionFactor leave
    # ``_has_conversion_factor`` False and this is a complete no-op.
    def _resolve_const_param_value(pid):
        # conversionFactor parameters are constant: value = initialAssignment if
        # any, else the declared value.
        if pid in ia_values:
            return ia_values[pid]
        p = sbml_model.getParameter(pid)
        return p.getValue() if (p is not None and p.isSetValue()) else 1.0

    _model_cf_val = (
        _resolve_const_param_value(sbml_model.getConversionFactor())
        if sbml_model.isSetConversionFactor()
        else None
    )
    species_cf: dict[str, float] = {}
    _any_species_cf = False
    for _sid in species_ids:
        _sp = sbml_model.getSpecies(_sid)
        if _sp is not None and _sp.isSetConversionFactor():
            species_cf[_sid] = _resolve_const_param_value(_sp.getConversionFactor())
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
        for _j in range(rxn.getNumReactants()):
            _sr = rxn.getReactant(_j)
            _st = _sr.getStoichiometry() if _sr.isSetStoichiometry() else 1.0
            net[_sr.getSpecies()] = net.get(_sr.getSpecies(), 0.0) - _st
        for _j in range(rxn.getNumProducts()):
            _sr = rxn.getProduct(_j)
            _st = _sr.getStoichiometry() if _sr.isSetStoichiometry() else 1.0
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
        if (
            target_var in species_idx
            and species_hosu.get(target_var, False)
            and comp_volumes.get(species_comp.get(target_var, ""), 1.0) != 1.0
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
        """
        rate_param_components, sf, reactants, products, comp_factors = classification
        if len(rate_param_components) == 1:
            rate_param_name = rate_param_components[0][0]
        else:
            derived_name = f"_rateLaw_{rid_safe}{derived_suffix}"
            expr = " * ".join(name for name, _ in rate_param_components)
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
        # admits the exact scalar live-volume correction §9 will apply. Skip
        # reversible reactions (their Functional law is a forward-minus-reverse
        # difference, not a monomial — keep refusing those).
        if _varvol_reject.get("comp") in varvol_ssa_comps and not rxn.getReversible():
            ssa_varvol_functional[i] = int(_varvol_reject["n_f"])

        # (#144 case 4, #172) A cross-compartment variable-volume mass-action
        # monomial (≥2 storage compartments, ≥1 variable-volume, all-hOSU=false) the
        # classifier routed to the Functional path — bare law (#144) or carrying an
        # explicit variable-volume compartment factor (#172, `cell·k·A·B`). Certify
        # it so §9 lifts the varvol_non_mass_action gate and applies the
        # per-compartment ODE live divide + SSA propensity correction (both keyed on
        # species factors only; an explicit compartment factor lives in base_func and
        # cancels — see _classify_mass_action_ast). Irreversible only (a reversible
        # Functional law is a forward-minus-reverse difference, not a monomial).
        _xcomp_varvol = _varvol_reject.get("xcomp_varvol_comps")
        if _xcomp_varvol and not rxn.getReversible():
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

        if rxn.getReversible():
            ssa_issues.append(
                SsaIssue(
                    severity="error",
                    code="reversible_non_mass_action",
                    message=(
                        f"Reaction '{rid}' is reversible='true' with a "
                        "kinetic law that did not classify as mass-action. "
                        "Under SSA the loader emits a single net-rate "
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
                species_idx[var] = sp_i
                species_ids.append(var)
                builder.add_observable(_safe_name(var), [(sp_i, 1.0)])

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
    # it symbolically (sympy), with ẋ_k known only for safe leaves: ``time()`` (=1),
    # a rate-ruled non-species variable (its rate-rule RHS), and another assignment
    # target (recursively). Any reference to a reaction/rule-driven SPECIES, an
    # event-promoted parameter (discontinuous; the #74 rescale owns those), or an
    # unparseable RHS makes V̇ unknown and we emit nothing — identical to today's
    # behavior, so a static-RHS AR compartment (V̇ ≡ 0) and every non-AR model stay
    # byte-identical. hOSU=true species and rule-target species are excluded exactly
    # as in 8b. The emitted term is additive to §9's reaction term and also covers a
    # flux-free species (pure dilution ``[S] = A/V(t)``); ODE-only for the same #81
    # reason (SSA conserves molecule count across a volume change).
    #
    # hOSU=false species that received a dilution term — their stored concentration
    # tracks amount/V_live, so the report-time amount selector reads V_live from the
    # AR expression column (``_varvol_ar_amount_map`` below). Populated by 8c.
    ar_dilution_species: set[str] = set()
    if ar_comp_targets:
        import re as _re

        import sympy as _sp

        from bngsim._codegen import _PY_KEYWORD_PARAM_NAMES, _alias_keyword_param
        from bngsim._jacobian import _TIME_SYM, _exprtk_to_sympy, sympy_to_exprtk

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

        # Classification in aliased-safe-name space (matches str(free_symbol)).
        _species_alias = {_alias(_safe_name(s)) for s in _sbml_species_ids}
        _event_alias = {_alias(_safe_name(p)) for p in event_promoted_params}
        _rr_alias = {_alias(s): s for s in _rr_rhs_ex}  # alias -> safe name
        _ar_alias = {_alias(s): s for s in _ar_rhs_ex}  # alias -> safe name
        _rr_sym_cache: dict[str, object] = {}

        def _rr_deriv(safe: str):
            if safe not in _rr_sym_cache:
                _rr_sym_cache[safe] = _exprtk_to_sympy(_rr_rhs_ex[safe])
            return _rr_sym_cache[safe]  # sympy expr, or None on parse failure

        def _leaf_deriv(nm: str, seen: frozenset):
            # d(symbol)/dt: None ⇒ unknown (bail the whole V̇); 0 ⇒ constant.
            if nm == _TIME_SYM:
                return _sp.Integer(1)
            if nm in _event_alias or nm in _species_alias:
                return None
            if nm in _rr_alias:
                return _rr_deriv(_rr_alias[nm])
            if nm in _ar_alias:
                return _vdot_sym(_ar_alias[nm], seen)
            return _sp.Integer(0)

        def _vdot_sym(safe_var: str, seen: frozenset):
            ex = _ar_rhs_ex.get(safe_var)
            if ex is None or safe_var in seen:
                return None
            g = _exprtk_to_sympy(ex)
            if g is None:
                return None
            seen2 = seen | {safe_var}
            total = _sp.Integer(0)
            for s in g.free_symbols:
                d = _leaf_deriv(str(s), seen2)
                if d is None:
                    return None
                if d != 0:
                    total = total + _sp.diff(g, s) * d
            return total

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

        for sid, comp in species_comp.items():
            if (
                comp in ar_comp_targets
                and sid in species_idx
                and not species_hosu.get(sid, False)
                and sid not in rate_rule_targets
                and sid not in assignment_targets
            ):
                vdot = _vdot_sym(_safe_name(comp), frozenset())
                if vdot is None or vdot == 0:
                    continue
                vdot_ex = sympy_to_exprtk(vdot)
                if vdot_ex is None:
                    continue
                vdot_ex = _unalias(vdot_ex)
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
    for i in range(sbml_model.getNumReactions()):
        rxn = sbml_model.getReaction(i)
        rid = rxn.getId()

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
                        _product_ids = [species_ids[pi] for pi in products]
                        _supported = (
                            _p == 1
                            and bool(_product_ids)
                            and all(
                                not species_hosu.get(sid, False) and species_comp[sid] == _ref_comp
                                for sid in _product_ids
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
                if _vvol != 1.0 or _vcomp in live_symbol_divide_comps:
                    # Storage is amount/V_c; divide the law by V_c (V_static for an
                    # amount-valued species in a variable-volume compartment), as
                    # the non-integer per-species fallback does. A static V_c=1
                    # skips this; a live-symbol compartment with V_static=1 still
                    # needs the divide after its volume changes.
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
                        species_hosu.get(s, False) for s in net
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
                    species_hosu.get(s, False) for s in net
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
                if _rep_comp in vstatic_divide_comps and all(
                    species_hosu.get(s, False) for s in net
                ):
                    _vv_exp = 0.0
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
        if not non_integer:
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
            if i in ssa_varvol_xcompartment:
                _varvol_comps = set(ssa_varvol_xcompartment[i])
                for _sid in net:
                    if species_comp[_sid] in _varvol_comps and not species_hosu.get(_sid, False):
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
            if vol != 1.0 or comp in live_symbol_divide_comps:
                # (#87, #114) For an amount-valued species in a variable-volume
                # (assignment-rule or rate-rule) compartment, the storage divide
                # is V_static (the load-time numeric), NOT the live symbol — see
                # section 9's ``else`` branch. Byte-identical for static comps.
                if comp in vstatic_divide_comps and species_hosu.get(sid, False):
                    divisor = repr(float(vol))
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
    if n_events > 0:
        # Track which parameters need promotion for event assignments
        event_param_promotions = {}  # param_id → species_idx

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
                        {
                            **{
                                p.getId(): (p.getValue() if p.isSetValue() else 0.0)
                                for j2 in range(sbml_model.getNumParameters())
                                for p in [sbml_model.getParameter(j2)]
                            },
                            **ia_values,
                        },
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
                        {
                            **{
                                p.getId(): (p.getValue() if p.isSetValue() else 0.0)
                                for j2 in range(sbml_model.getNumParameters())
                                for p in [sbml_model.getParameter(j2)]
                            },
                            **ia_values,
                        },
                        func_defs,
                        time_value=None,
                        avogadro_value=None,
                    )
                    if pri_numeric is not None and float(pri_numeric).is_integer():
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
                    event_param_promotions[var] = sp_i
                    species_idx[var] = sp_i
                    species_ids.append(var)
                    builder.add_observable(_safe_name(var), [(sp_i, 1.0)])
                    assignments.append((sp_i, value_expr))

            if assignment_value_expr_by_var and ar_comp_targets:
                event_assignment_replacements = {
                    _safe_name(var): expr for var, expr in assignment_value_expr_by_var.items()
                }
                assigned_vars = set(assignment_value_expr_by_var)
                for j in range(sbml_model.getNumRules()):
                    rule = sbml_model.getRule(j)
                    comp_id = rule.getVariable()
                    rmath = rule.getMath()
                    if (
                        not rule.isAssignment()
                        or comp_id not in ar_comp_targets
                        or comp_id in directly_resized_comps
                        or rmath is None
                        or not (_ast_name_set(rmath) & assigned_vars)
                    ):
                        continue
                    old_size_expr = _ast_to_exprtk_with_funcdefs(rmath, func_defs)
                    new_size_expr = _replace_exprtk_symbols(
                        old_size_expr, event_assignment_replacements
                    )
                    if new_size_expr != old_size_expr:
                        resized_comps.append((comp_id, new_size_expr))

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
            # events.
            assignment_ode_only = [False] * len(assignments)
            if resized_comps:
                explicitly_assigned = {sp_i for sp_i, _ in assignments}
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
                        if comp_id in ar_comp_targets:
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
                            assignments.append((sp_i, rescale_expr))
                            assignment_ode_only.append(True)
                            continue
                        if sp_i in explicitly_assigned:
                            continue
                        rescale_expr = f"{_safe_name(sid)} * ({comp_safe}) / ({new_size_expr})"
                        assignments.append((sp_i, rescale_expr))
                        assignment_ode_only.append(True)

            if not assignments:
                logger.warning(f"Event '{eid}': no valid assignments, skipping")
                continue

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

    # ── 10.5. Discontinuity triggers (GH #72, GH #246) ─────────────────
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
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if (rule.isAssignment() or rule.isRate()) and rule.getMath():
            _collect_time_discontinuity_conditions(rule.getMath(), func_defs, disc_conditions)
    for i in range(sbml_model.getNumReactions()):
        kl = sbml_model.getReaction(i).getKineticLaw()
        if kl and kl.getMath():
            _collect_time_discontinuity_conditions(kl.getMath(), func_defs, disc_conditions)
    for i in range(sbml_model.getNumEvents()):
        event = sbml_model.getEvent(i)
        trigger = event.getTrigger()
        if trigger is not None and trigger.getMath() is not None:
            _collect_relational_edge_conditions(trigger.getMath(), func_defs, disc_conditions)

    # State-dependent floor()/ceiling() thresholds (GH #244): small numeric
    # integer boundaries in expressions like ``ceiling(S1 * p1)`` are also
    # discontinuities in the RHS, but the crossing is a state threshold rather
    # than a fixed time threshold. Register the bounded cases we can infer at
    # load time; models outside that conservative envelope are left unchanged.
    time_dependent_names = _assignment_time_dependent_names(sbml_model)
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
                kl.getMath(), func_defs, eval_ctx, time_dependent_names, disc_conditions
            )

    seen_disc: set[str] = set()
    for cond in disc_conditions:
        if cond not in seen_disc:
            seen_disc.add(cond)
            builder.add_discontinuity_trigger(cond)

    # Periodic floor()/modulo dosing schedules (GH #88) can't be resolved by the
    # monotonic roots above — register a step-size bound that keeps the adaptive
    # integrator from stepping over a narrow dose pulse. None for any model with
    # no time-dependent floor/ceil/modulo feeding the RHS (the common case).
    periodic_disc_max_step = _periodic_time_disc_max_step(sbml_model, func_defs, eval_ctx)

    # ── 10.6. rateOf csymbol support (GH #106) ─────────────────────────
    # Enable the live-derivative path iff the model actually *references*
    # rateOf — the official csymbol or a *called* COPASI rateOf-idiom funcDef.
    # A funcDef defined but never called (MODEL2403070001) leaves the model on
    # the byte-identical non-rateOf path.
    rateof_funcdef_names = {
        name for name, (_p, body) in func_defs.items() if body is _RATEOF_FUNCDEF
    }
    if _model_uses_rateof(sbml_model, func_defs, rateof_funcdef_names):
        builder.enable_rateof()

    # ── 11. Build ─────────────────────────────────────────────────────
    core = builder.build()
    model = Model(_core=core)
    model._ssa_issues = ssa_issues
    model._periodic_disc_max_step = periodic_disc_max_step

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
    ar_report_map: dict[str, tuple[str, str, float]] = {}
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
        ar_report_map[safe] = (kind, safe, vdiv)
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
    varvol_event_resize_map: dict[str, tuple[str, float, bool]] = {}
    if event_resize_comps:
        for sid, cid in species_comp.items():
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
            return local_params[name]

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
        return str(node.getInteger())
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E):
        return _real_literal(node.getReal())
    if t == libsbml.AST_RATIONAL:
        return f"({node.getNumerator()}/{node.getDenominator()})"

    # Leaf: name
    if t == libsbml.AST_NAME:
        name = node.getName()
        if local_params and name in local_params:
            return local_params[name]
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
            return local_params[name]
        return _safe_name(name) if name else "0"

    # For non-name nodes, convert normally but with subs as local_params
    merged = dict(local_params or {})
    merged.update(subs)
    return _ast_to_exprtk_recursive(node, func_defs, merged)
