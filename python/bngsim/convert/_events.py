"""bngsim.convert._events — SBML events → BNGL actions (GH #224 phase 2).

A loaded :class:`bngsim.Model` exposes only ``core.n_events`` (a count) — the
event details (trigger, delay, assignments) live in the source SBML and are
compiled into the C++ event dispatcher. To translate events into a BNGL
**actions** block the converter re-reads the SBML here, classifies each event,
and maps the tractable ones onto a :class:`~bngsim.convert.ProtocolSpec`.

A BNGL action sequence is a scheduled-state-change protocol: ``simulate`` phases
with ``setConcentration``/``setParameter`` between them. So a **fixed-time**
SBML event — a ``time >= T`` trigger (constant ``T``, no species reference) that
assigns constants — becomes ``simulate(→T)`` → ``setConcentration(target, v)`` →
``simulate(→…)``. A **state-triggered** event (the trigger depends on a species
/ state crossing) has no actions equivalent and is refused fail-loud, as is any
event whose trigger time, delay, or assignment value is not a constant (the
actions channel carries only scheduled, pre-known changes).

The translation is intentionally conservative: anything it cannot represent
*faithfully* it refuses (returns a plain-English note) rather than emit a
wrong-but-plausible protocol. The accepted set is exactly the
concentration-space models the cBNGL writer already accepts (amount-valued
non-unit species are refused upstream), so an assignment value is emitted
verbatim.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

if TYPE_CHECKING:
    from bngsim._model import Model


# ─── numeric MathML folding ──────────────────────────────────────────────────


def _const_env(sbml_model) -> dict[str, float]:
    """The names that fold to a constant in a trigger/delay/assignment: global
    parameters and compartment sizes (species are *not* constants — a reference
    to one makes the expression state-dependent, i.e. non-fixed-time)."""
    env: dict[str, float] = {}
    for i in range(sbml_model.getNumParameters()):
        p = sbml_model.getParameter(i)
        if p.isSetValue():
            env[p.getId()] = p.getValue()
    for i in range(sbml_model.getNumCompartments()):
        c = sbml_model.getCompartment(i)
        if c.isSetSize():
            env[c.getId()] = c.getSize()
    return env


def _eval_const(node, env: dict[str, float]) -> float | None:
    """Evaluate a MathML AST to a constant, or ``None`` if it is not constant.

    Folds numbers, π/e, the global parameters / compartment sizes in ``env``, and
    ``+ - * / ^`` (incl. unary minus). A reference to ``time``, a species, or any
    unknown name — or an unsupported operator — yields ``None`` (→ "not a
    constant", which the caller treats as a refusal for the fixed-time path).
    """
    import libsbml

    if node is None:
        return None
    t = node.getType()
    if t == libsbml.AST_INTEGER:
        return float(node.getInteger())
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
        return float(node.getReal())
    if t == libsbml.AST_CONSTANT_PI:
        return math.pi
    if t == libsbml.AST_CONSTANT_E:
        return math.e
    if t == libsbml.AST_NAME:
        return env.get(node.getName())
    if t == libsbml.AST_NAME_TIME:
        return None
    n = node.getNumChildren()
    if t == libsbml.AST_PLUS:
        acc = 0.0
        for i in range(n):
            v = _eval_const(node.getChild(i), env)
            if v is None:
                return None
            acc += v
        return acc
    if t == libsbml.AST_TIMES:
        acc = 1.0
        for i in range(n):
            v = _eval_const(node.getChild(i), env)
            if v is None:
                return None
            acc *= v
        return acc
    if t == libsbml.AST_MINUS:
        if n == 1:
            v = _eval_const(node.getChild(0), env)
            return None if v is None else -v
        if n == 2:
            a = _eval_const(node.getChild(0), env)
            b = _eval_const(node.getChild(1), env)
            return None if a is None or b is None else a - b
        return None
    if t == libsbml.AST_DIVIDE and n == 2:
        a = _eval_const(node.getChild(0), env)
        b = _eval_const(node.getChild(1), env)
        if a is None or b is None or b == 0.0:
            return None
        return a / b
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER) and n == 2:
        a = _eval_const(node.getChild(0), env)
        b = _eval_const(node.getChild(1), env)
        if a is None or b is None:
            return None
        try:
            return float(a**b)
        except (ValueError, OverflowError):
            return None
    return None


# ─── trigger classification ──────────────────────────────────────────────────

# Relational op → (operator-with-time-on-the-left means rising,
#                  operator-with-time-on-the-right means rising)
# A fixed-time event fires when ``time`` crosses ``T`` upward, which is
# ``time >= T`` / ``time > T`` (time on the left) or the mirror ``T <= time`` /
# ``T < time`` (time on the right). Any other shape (``time <= T``,
# equality, a conjunction) is refused — it is not a single rising schedule.


def _is_time(node) -> bool:
    import libsbml

    if node is None:
        return False
    if node.getType() == libsbml.AST_NAME_TIME:
        return True
    # A bare <ci>time</ci> (malformed COPASI idiom) parses as a plain name.
    return node.getType() == libsbml.AST_NAME and node.getName() == "time"


def _dynamic_names(node, env: dict[str, float]) -> set[str]:
    """Names the AST references that are *not* constants — anything that moves in
    time: a species, a rate-rule / assignment-rule parameter, etc. (``time`` is
    excluded; constants in ``env`` are excluded). A trigger that references any of
    these is state-dependent, not a fixed schedule.
    """
    import libsbml

    if node is None:
        return set()
    out: set[str] = set()
    if node.getType() == libsbml.AST_NAME:
        nm = node.getName()
        if nm != "time" and nm not in env:
            out.add(nm)
    for i in range(node.getNumChildren()):
        out |= _dynamic_names(node.getChild(i), env)
    return out


def _trigger_fire_time(
    trigger_ast, env: dict[str, float], species_ids: set[str]
) -> tuple[float | None, str]:
    """Classify a trigger; return ``(fire_time, reason)``.

    ``fire_time`` is the constant ``T`` at which the event fires for a fixed-time
    (``time >= T``) trigger, else ``None`` with ``reason`` a plain-English note
    naming why it is not fixed-time (state-triggered, non-constant threshold,
    unsupported shape).
    """
    import libsbml

    if trigger_ast is None:
        return None, "has no trigger"
    dyn = _dynamic_names(trigger_ast, env)
    if dyn:
        kind = "a species" if dyn & species_ids else "a state variable"
        return None, (
            f"is state-triggered (the trigger depends on {kind}: "
            f"{', '.join(sorted(dyn)[:3])}) — no scheduled fire time"
        )

    t = trigger_ast.getType()
    rising_left = (libsbml.AST_RELATIONAL_GEQ, libsbml.AST_RELATIONAL_GT)
    rising_right = (libsbml.AST_RELATIONAL_LEQ, libsbml.AST_RELATIONAL_LT)
    if t not in rising_left + rising_right or trigger_ast.getNumChildren() != 2:
        return None, (
            "is not a simple rising time threshold (time >= T) — its trigger has "
            "no actions form"
        )
    left, right = trigger_ast.getChild(0), trigger_ast.getChild(1)
    if t in rising_left and _is_time(left):
        thr = _eval_const(right, env)
    elif t in rising_right and _is_time(right):
        thr = _eval_const(left, env)
    else:
        return None, (
            "is not a rising time threshold (time on the wrong side of the "
            "comparison) — no single scheduled fire time"
        )
    if thr is None:
        return None, "has a non-constant trigger time (cannot be scheduled)"
    return float(thr), ""


# ─── species/parameter target → codegen species name ─────────────────────────


def _codegen_species_names(model: Model) -> dict[str, int]:
    """Map each codegen species name to its index (for target resolution)."""
    species = model._core.codegen_data()["species"]
    return {s["name"]: i for i, s in enumerate(species)}


def _resolve_target(var: str, name_to_idx: dict[str, int]) -> int | None:
    """Resolve an SBML event-assignment target id to a codegen species index.

    Both an SBML *species* and an SBML *parameter/compartment* event target end
    up as a codegen species (the loader promotes assigned parameters to species),
    so every tractable target resolves here. ``_safe_name`` only rewrites ExprTk
    reserved words, so the id matches directly or as ``_ant_<id>``.
    """
    if var in name_to_idx:
        return name_to_idx[var]
    alt = f"_ant_{var}"
    return name_to_idx.get(alt)


# ─── public entry ────────────────────────────────────────────────────────────


def sbml_events_to_protocol(
    sbml_path: str | Path,
    model: Model,
    *,
    t_span: tuple[float, float] = (0.0, 100.0),
    n_points: int = 101,
) -> tuple[ProtocolSpec | None, list[str]]:
    """Translate the source SBML's events into a BNGL-actions :class:`ProtocolSpec`.

    Returns ``(protocol, lossy)``. When every event is a tractable fixed-time
    event, ``protocol`` is a sequence of ``simulate`` phases with
    ``setConcentration`` state changes at each fire time (targets are *codegen
    species names*; the model-block writer translates them to compartment-qualified
    patterns) and ``lossy`` is empty. When any event cannot be represented
    faithfully (state-triggered, non-constant trigger/delay/value, unresolvable
    target), ``protocol`` is ``None`` and ``lossy`` holds one plain-English note
    per offending event — the caller refuses (strict) or warns (best-effort).

    The horizon for the trailing ``simulate`` phase comes from ``t_span``; if an
    event fires past it the horizon is extended so no scheduled change is dropped.
    """
    import libsbml

    sbml_model = libsbml.readSBMLFromFile(str(sbml_path)).getModel()
    if sbml_model is None:  # pragma: no cover — Model.from_sbml already parsed it
        return None, ["could not re-read the source SBML to translate events"]

    n_events = sbml_model.getNumEvents()
    if n_events == 0:
        return None, []

    env = _const_env(sbml_model)
    species_ids = {
        sbml_model.getSpecies(i).getId() for i in range(sbml_model.getNumSpecies())
    }
    name_to_idx = _codegen_species_names(model)

    lossy: list[str] = []
    # (fire_time, event_order, [(species_idx, value), …])
    fires: list[tuple[float, int, list[tuple[int, float]]]] = []

    for i in range(n_events):
        ev = sbml_model.getEvent(i)
        eid = ev.getId() or f"event #{i + 1}"
        trig = ev.getTrigger().getMath() if ev.isSetTrigger() else None
        fire_t, reason = _trigger_fire_time(trig, env, species_ids)
        if fire_t is None:
            lossy.append(f"event {eid!r} {reason}")
            continue

        # A constant delay shifts the fire time; a non-constant one is unschedulable.
        if ev.isSetDelay() and ev.getDelay() is not None and ev.getDelay().getMath() is not None:
            d = _eval_const(ev.getDelay().getMath(), env)
            if d is None:
                lossy.append(f"event {eid!r} has a non-constant delay (cannot be scheduled)")
                continue
            fire_t += d

        assigns: list[tuple[int, float]] = []
        ok = True
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            var = ea.getVariable()
            val = _eval_const(ea.getMath(), env) if ea.isSetMath() else None
            if val is None:
                lossy.append(
                    f"event {eid!r} assigns {var!r} a non-constant value — the "
                    "actions channel carries only scheduled, pre-known changes"
                )
                ok = False
                break
            idx = _resolve_target(var, name_to_idx)
            if idx is None:
                lossy.append(
                    f"event {eid!r} assigns {var!r}, which has no species in the "
                    "converted model (target unresolved)"
                )
                ok = False
                break
            assigns.append((idx, val))
        if ok:
            fires.append((fire_t, i, assigns))

    if lossy:
        return None, lossy
    if not fires:
        return None, []

    return _build_protocol(fires, model, t_span, n_points), []


def _build_protocol(
    fires: list[tuple[float, int, list[tuple[int, float]]]],
    model: Model,
    t_span: tuple[float, float],
    n_points: int,
) -> ProtocolSpec:
    """Assemble ``simulate`` phases + ``setConcentration`` state changes.

    Phases run from ``t_start`` through each distinct fire time (in order) to the
    horizon; the events at a fire time apply their assignments (in SBML order)
    after the phase that reaches it. ``continue=>1`` chains each phase onto the
    prior end-state. Output steps per phase are apportioned by duration.
    """
    species = model._core.codegen_data()["species"]
    t_start, t_end = float(t_span[0]), float(t_span[1])
    max_fire = max(f[0] for f in fires)
    if max_fire > t_end:
        t_end = max_fire
    total = max(t_end - t_start, 1.0)
    base_steps = max(n_points - 1, 1)

    # group assignments by fire time, preserving event order within a time
    by_time: dict[float, list[tuple[int, float]]] = {}
    for ft, _order, assigns in sorted(fires, key=lambda x: (x[0], x[1])):
        by_time.setdefault(ft, []).extend(assigns)

    steps: list[object] = []
    prev = t_start
    first = True
    for ft in sorted(by_time):
        if ft > prev:
            steps.append(_phase(prev, ft, total, base_steps, first))
            first = False
            prev = ft
        for sp_idx, val in by_time[ft]:
            steps.append(
                StateChange(
                    kind="set_concentration",
                    target=species[sp_idx]["name"],
                    value=float(val),
                )
            )
    if t_end > prev:
        steps.append(_phase(prev, t_end, total, base_steps, first))

    return ProtocolSpec(steps=tuple(steps), source=None)


def _phase(
    a: float, b: float, total: float, base_steps: int, first: bool
) -> Experiment:
    n_steps = max(1, round(base_steps * (b - a) / total))
    extra: dict[str, Any] = {} if first else {"continue": 1}
    return Experiment(
        kind="simulate",
        method="ode",
        t_span=(a, b),
        n_points=n_steps + 1,
        extra=extra,
    )
