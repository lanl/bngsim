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
actions channel carries only scheduled, pre-known changes). A trigger already
true at the start fires there only if its ``initialValue`` is false, and the
events at one instant apply in the order the run executes them (see
:class:`_Fire`); where the run's random draw among equal priorities, or its
rounding and its stops, can decide what a species is left at, they are
refused (:func:`_unordered_writes`).

The translation is intentionally conservative: anything it cannot represent
*faithfully* it refuses (returns a plain-English note) rather than emit a
wrong-but-plausible protocol. The accepted set is exactly the
concentration-space models the cBNGL writer already accepts (amount-valued
non-unit species are refused upstream), so an assignment value is emitted
verbatim.
"""

from __future__ import annotations

import bisect
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

if TYPE_CHECKING:
    from bngsim._model import Model


# ─── numeric MathML folding ──────────────────────────────────────────────────


def _const_env(
    sbml_model,
) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[str, Any]]:
    """The names that fold to a constant in a trigger/delay/assignment: global
    parameters and compartment sizes (species are *not* constants — a reference
    to one makes the expression state-dependent, i.e. non-fixed-time).

    Returns ``(env, own, unfolded)``. ``env`` holds each name that keeps one
    value through the run, at that value: the declared one, or what its
    initialAssignment folds to (the assignment overrides the declared value), or
    what its assignment rule folds to over the others. A name a rate rule or an event
    changes is left out, as is one whose math does not fold (it reads a species
    or the time): a reference to it is refused.

    ``unfolded`` maps each name that keeps one value but whose initialAssignment
    does not fold (it reads a species' initial value, say) to that math.

    ``own[i]`` holds the targets event ``i`` alone writes, at their start values,
    for its assigned values to read: nothing changes such a target before the
    event executes, and a time-threshold trigger executes it once, so
    ``vDiet = vDiet*fDiet`` reads ``vDiet``'s start value. A trigger never reads
    them, as the event re-arms on what it writes.
    """

    def kept(sid: str) -> bool:
        return (
            sbml_model.getParameter(sid) is not None or sbml_model.getCompartment(sid) is not None
        )

    assigned: dict[str, Any] = {}
    rated: set[str] = set()
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isAssignment() and kept(rule.getVariable()):
            assigned[rule.getVariable()] = rule.getMath()
        elif rule.isRate():
            rated.add(rule.getVariable())
    writers: dict[str, set[int]] = {}
    for i in range(sbml_model.getNumEvents()):
        ev = sbml_model.getEvent(i)
        for j in range(ev.getNumEventAssignments()):
            writers.setdefault(ev.getEventAssignment(j).getVariable(), set()).add(i)
    initial: dict[str, Any] = {}
    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        if kept(ia.getSymbol()):
            initial[ia.getSymbol()] = ia.getMath()

    # What each holds at the start. An initialAssignment reads the others
    # there, those a rule or an event changes later included (COPASI writes
    # ``X_0 = X`` for X's start value).
    start: dict[str, float] = {}
    for i in range(sbml_model.getNumParameters()):
        p = sbml_model.getParameter(i)
        if p.isSetValue():
            start[p.getId()] = p.getValue()
    for i in range(sbml_model.getNumCompartments()):
        c = sbml_model.getCompartment(i)
        if c.isSetSize():
            start[c.getId()] = c.getSize()
    for sid in (*initial, *assigned):
        start.pop(sid, None)
    _fold_all(start, {**initial, **assigned})

    # What keeps it. An assignment rule keeps one value only where it folds
    # over those that keep theirs.
    moving = rated | set(writers) | set(assigned)
    env = {sid: v for sid, v in start.items() if sid not in moving}
    _fold_all(env, assigned)

    own: dict[int, dict[str, float]] = {}
    for sid, events in writers.items():
        if len(events) == 1 and sid in start and sid not in rated and sid not in assigned:
            own.setdefault(next(iter(events)), {})[sid] = start[sid]
    # Kept through the run, but from an initialAssignment that does not fold.
    unfolded = {sid: m for sid, m in initial.items() if sid not in moving and sid not in env}
    return env, own, unfolded


def _fold_all(env: dict[str, float], pending: dict[str, Any]) -> None:
    """Add each name of ``pending`` to ``env`` at what its math folds to over
    ``env``, again and again until no more fold: one may read another, in any
    document order. One that never folds is left out."""
    pending = dict(pending)
    while pending:
        folded = {s: v for s, m in pending.items() if (v := _eval_const(m, env)) is not None}
        if not folded:
            return
        env.update(folded)
        for s in folded:
            del pending[s]


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
        # A complex root (TypeError), 0 to a negative power (ZeroDivisionError).
        except (ValueError, ArithmeticError, TypeError):
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
            "is not a simple rising time threshold (time >= T) — its trigger has no actions form"
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
        # Reading nothing that moves, it is a constant this module cannot fold
        # (`abs(-1)`): nan, for the caller to read from the run.
        side = right if t in rising_left else left
        if not _reads_clock(side):
            return math.nan, ""
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


# ─── what executes when ──────────────────────────────────────────────────────


class _Fire(NamedTuple):
    """One event's execution: when, in what order, and what it sets."""

    time: float
    # A strict trigger is true only past its threshold, and the run reads it
    # there, a little past it (see _landing), so neither it nor its delayed
    # execution is on one double for certain. Written after the rest of its
    # instant, where it sets nothing they set differently.
    past: bool
    priority: float | None  # None: not a constant
    # No priority as the loader writes it: none, or a plain number that is 0.
    # Where every event of a tie has none, the run executes the delayed ones
    # after the rest.
    unprioritised: bool
    # Where it triggered, before its delay: delayed executions of one priority
    # run in the order they were triggered. None where it has no delay.
    triggered: float | None
    order: int  # document order, which breaks the remaining ties
    eid: str
    assigns: list[tuple[int, float, str]]  # (species index, value, SBML id)
    read: float = 0.0  # where the trigger turns true: its threshold, or the start
    delay: float = 0.0
    # Read at the start of the run, where it was true already: there exactly.
    pinned: bool = False

    def sort_key(self) -> tuple[float, bool, float, bool, float, int]:
        # Higher priority first; among equal ones, those not delayed, then the
        # delayed ones in the order they were triggered, one order the run can
        # take. Where it can take another that leaves a different value,
        # _unordered_writes refuses.
        return (
            self.time,
            self.past,
            -(self.priority or 0.0),
            self.triggered is not None,
            self.triggered or 0.0,
            self.order,
        )


def _instant_reach(t: float, span: float) -> float:
    """How far apart two times near ``t`` can be and still be read by the run
    as one instant, or in either order: the widest of the tolerances it lands
    on a time with, on a run of length ``span``.

    * Two stops on literal time closer than (64 + 4)·ε·max(|a|, |b|) are
      merged into the later one (``merge_crossing_stops`` and
      ``_same_instant``, in ``_switch_sensitivity.py``), so a trigger on the
      earlier is read on the later, with what turns true there, and a delay
      counts from there.
    * A root within 256·ε·max(|t|, 1) after a stop is taken as on it
      (``event_root_reach``, ``cvode_simulator.cpp``).
    * CVODE places a root within 100·ε·(|t| + |h|) of where it is, h the step,
      at most the run's length. That is where the run reads ``time > T``
      true: measured, 1 to 27,417 doubles past T (51 to 27,417 at T = 0.001,
      80 to 508 at 0.1, 1 to 167 at 0.3, 58 to 108 at 37), never past 0.39
      of this bound in some 1,350 runs.
    """
    eps = sys.float_info.epsilon
    return max(256.0 * eps * max(abs(t), 1.0), 100.0 * eps * (abs(t) + span))


def _chain_end(points: list[float], t: float, span: float) -> float:
    """The latest of the run's stops that ``t`` can be merged into: one within
    reach after it, or within reach after that one, and so on."""
    while True:
        j = bisect.bisect_right(points, t + _instant_reach(t, span)) - 1
        if j < 0 or points[j] <= t:
            return t
        t = points[j]


def _reading(f: _Fire, points: list[float], span: float) -> tuple[float, float]:
    """The earliest and latest times the run can read ``f``'s trigger at.

    On its own time, or on a stop it is merged into. A strict one is read past
    its threshold, up to a root's reach. So is any other that has a stop, a
    row or an execution within that reach before it: having just executed or
    stopped there, the run can step over the trigger's own stop and find it by
    its root (measured, up to 0.26 of the reach late). Read at the start, it is
    read there exactly.
    """
    if f.pinned:
        return f.read, f.read
    hi = _chain_end(points, f.read, span)
    reach = _instant_reach(f.read, span)
    j = bisect.bisect_left(points, f.read)
    if f.past or (j > 0 and points[j - 1] >= f.read - reach):
        hi = max(hi, f.read + reach)
    return f.read, hi


def _landing(f: _Fire, points: list[float], span: float) -> tuple[float, float]:
    """The earliest and latest times the run can execute ``f`` at: where its
    trigger is read (:func:`_reading`), and a delay after that.

    A delay too small to move the time off the double it was read on still
    makes the execution wait: the run queues it, finds it due, and executes
    it after the rest of the instant, whatever its priority. So it is not on
    that double for certain, and lands up to the reach after it.
    """
    lo, hi = _reading(f, points, span)
    if f.triggered is not None:
        moved = lo + f.delay != lo
        lo, hi = lo + f.delay, hi + f.delay
        if not moved:
            hi = max(hi, lo + _instant_reach(lo, span))
    return lo, hi


def _run_stops(model: Model, t_start: float, t_end: float) -> list[float]:
    """The times on literal time the run of ``model`` stops on, before they
    are merged, as ``Simulator._apply_crossing_stops`` finds them: every time
    discontinuity a condition, an event trigger, a table function or a smooth
    pulse of the clock has. Best effort, as there: none where it cannot tell."""
    from bngsim._smooth_pulses import smooth_pulse_stops
    from bngsim._switch_sensitivity import crossings_with_periodic, table_function_stops

    times: list[float] = []
    try:
        conditions = model.time_discontinuity_conditions()
        if conditions:
            crossings, periodic, _ = crossings_with_periodic(
                model._core,
                t_start,
                t_end,
                conditions,
                touches=getattr(model, "_event_disc_conditions", frozenset()),
                over_budget=[],
            )
            times += [stop.time for stop in (*crossings, *periodic)]
        breaks = model.table_function_breaks()
        if breaks:
            times += [
                stop.time for stop in table_function_stops(model._core, t_start, t_end, breaks)
            ]
        times += [stop.time for stop in smooth_pulse_stops(model._core, t_start, t_end)]
    except Exception:  # the run steps on without them where it cannot place them
        pass
    return [float(t) for t in times]


def _unordered_writes(
    fires: list[_Fire], span: float, stops: list[float], rows: list[float]
) -> list[str]:
    """A note for each species that events the run can execute together leave
    at a value their order decides, when that order is not known ahead.

    Executions whose landings (:func:`_landing`, over the run's ``stops`` and
    its output ``rows``) come within reach of each other (:func:`_instant_reach`)
    are one cluster, which the run can execute as one instant, by priority, or
    apart, as its rounding and its stops fall. Each species is decided by its
    own writers in the cluster (:func:`_species_note`).
    """
    points = sorted({*stops, *rows, *(f.read for f in fires), *(f.time for f in fires)})
    landed = sorted(((*_landing(f, points, span), f) for f in fires), key=lambda x: x[0])
    # Where each delayed execution's trigger is read: the run draws it there,
    # with what executes there, before it waits.
    readings = [(*_reading(f, points, span), f) for f in fires if f.triggered is not None]
    clusters: list[list[tuple[float, float, _Fire]]] = []
    reach_to = -math.inf
    for lo, hi, f in landed:
        if not clusters or lo > reach_to:
            clusters.append([])
            reach_to = -math.inf
        clusters[-1].append((lo, hi, f))
        reach_to = max(reach_to, hi + _instant_reach(hi, span))
    notes: list[str] = []
    for members in clusters:
        if len(members) < 2:
            continue
        writes: dict[int, list[tuple[_Fire, float]]] = {}
        var: dict[int, str] = {}
        for _, _, f in members:
            for idx, val, sid in f.assigns:
                writes.setdefault(idx, []).append((f, val))
                var[idx] = sid
        for idx, ws in writes.items():
            note = _species_note(var[idx], ws, members, span, readings)
            if note:
                notes.append(note)
    return notes


def _species_note(
    var: str,
    ws: list[tuple[_Fire, float]],
    members: list[tuple[float, float, _Fire]],
    span: float,
    readings: list[tuple[float, float, _Fire]] = (),  # type: ignore[assignment]
) -> str | None:
    """Why the value the writers ``ws`` of ``var``, in the cluster ``members``,
    leave is not known ahead, or None where it is.

    The last to execute sets it, so it is one that no other writer surely
    executes after: none lands later by more than the reach. Where those
    agree, it is known. Where they are one instant for certain, all landing on
    one double, or all triggered together on one, as the run reads it, by
    triggers that are not strict, with nothing delayed among them, the run
    orders them as it orders an instant (:func:`_order_note`), in a tie with
    whatever else in the cluster can land with them, and every delayed
    execution whose trigger is read there (``readings``), which the run draws
    with them before it waits. Otherwise it is not known.
    """
    landing = {f.order: (lo, hi) for lo, hi, f in members}

    def surely_after(g: _Fire, f: _Fire) -> bool:
        f_hi = landing[f.order][1]
        return landing[g.order][0] > f_hi + _instant_reach(f_hi, span)

    last = [(f, v) for f, v in ws if not any(surely_after(g, f) for g, _ in ws if g is not f)]
    if len({v for _, v in last}) < 2:
        return None
    fs = [f for f, _ in last]
    t = fs[0].time
    together = all(f.time == t for f in fs) and (
        all(landing[f.order] == (t, t) for f in fs)
        or all(not f.past and f.triggered is None and f.read == fs[0].read for f in fs)
    )
    if together:
        lo = min(landing[f.order][0] for f in fs)
        hi = max(landing[f.order][1] for f in fs)
        reach = _instant_reach(hi, span)
        # The run's tie is drawn among what executes there and what triggers
        # there to wait, so a priority among the latter counts too.
        batch = [g for a, b, g in members if a <= hi + reach and b >= lo - reach]
        batch += [g for a, b, g in readings if a <= hi + reach and b >= lo - reach]
        return _order_note(t, var, last, batch)

    # Name what keeps them from being one instant: a writer that does not land
    # on their double for certain, beside one that sets a different value.
    def apart(g: _Fire) -> bool:
        return landing[g.order] != (t, t)

    f, val = next(((g, v) for g, v in last if apart(g)), last[0])
    other = next(g for g, v in last if v != val)
    when = f"t = {f.time!r}" if f.time == other.time else f"t = {f.time!r} and t = {other.time!r}"
    return (
        f"events {f.eid!r} and {other.eid!r} set {var!r} to different values at {when}, "
        "close enough together that the translation cannot know which the run "
        "executes last — the value they leave has no actions form"
    )


def _order_note(
    t: float, var: str, ws: list[tuple[_Fire, float]], group: list[_Fire]
) -> str | None:
    """Why the value the executions ``ws``, on the double ``t`` for certain,
    leave in ``var`` is not known ahead, or None where it is. ``group`` is
    every execution there.

    The last of them to execute sets it: one of the lowest priority, and of the
    ones among those that may execute last (:func:`_may_execute_last`), the run
    picks at random. Where those set different values, the species has none an
    action can set. A priority that is not a constant leaves the order unknown
    wherever the values differ.
    """
    if len({val for _, val in ws}) < 2:
        return None

    def apart(f: _Fire, val: float, among: list[tuple[_Fire, float]]) -> str:
        other = next(g for g, v in among if v != val)
        return f"events {f.eid!r} and {other.eid!r} set {var!r} to different values at t = {t:g}"

    unknown = [(f, val) for f, val in ws if f.priority is None]
    if unknown:
        f, val = unknown[0]
        return (
            f"{apart(f, val, ws)}, and {f.eid!r} has a non-constant priority — "
            "which executes last cannot be scheduled"
        )
    low = min(f.priority for f, _ in ws if f.priority is not None)
    # The run's tie: what has that priority there, and may, a priority that is
    # not a constant.
    tie = [f for f in group if f.priority is None or f.priority == low]
    last = _may_execute_last([(f, val) for f, val in ws if f.priority == low], tie)
    if len({val for _, val in last}) > 1:
        f, val = last[0]
        return (
            f"{apart(f, val, last)} with equal priority, and the run executes "
            "them in a random order — the value they leave has no actions form"
        )
    return None


def _may_execute_last(
    ws: list[tuple[_Fire, float]], tie: list[_Fire]
) -> list[tuple[_Fire, float]]:
    """Of the executions ``ws``, all of one priority in one part of an instant,
    those the run may execute last. ``tie`` is every execution there of that
    priority, ``ws`` among them.

    The run draws among a tie at random, with two exceptions
    (src/cvode_simulator.cpp). Delayed executions keep the order they were
    triggered in, so only the last triggered may be last; two triggered at one
    instant were ordered by a random draw there. And where no event of the tie
    has a priority, the delayed ones execute after the rest.
    """
    delayed = [(f, val) for f, val in ws if f.triggered is not None]
    if delayed:
        latest = max(f.triggered for f, _ in delayed if f.triggered is not None)
        delayed = [(f, val) for f, val in delayed if f.triggered == latest]
        if all(f.unprioritised for f in tie):
            return delayed
    return [(f, val) for f, val in ws if f.triggered is None] + delayed


# ─── what the run reads ──────────────────────────────────────────────────────
# The run evaluates a trigger's threshold, a delay and a priority with its own
# evaluator, which folds constants in its own order (`d*3*7` as `d*21`), and
# holds a parameter an initialAssignment or a rule sets as an expression it
# evaluates the same way. So the doubles it compares are read from it, not
# folded again here, which can differ in the last places.


def _run_value(core, text: str, at: float) -> float | None:
    """``text`` as the run's evaluator reads it at time ``at``, or None where
    it does not compile."""
    try:
        return float(core.evaluate_over_times(text, [at])[0])
    except Exception:
        return None


def _run_param(model: Model, sid: str) -> float | None:
    """The value the loaded model holds for SBML parameter or compartment
    ``sid`` (renamed where it is an evaluator's word), or None."""
    for name in (sid, f"_ant_{sid}"):
        try:
            value = float(model.get_param(name))
        except Exception:
            continue
        if math.isfinite(value):
            return value
    return None


def _function_definitions(sbml_model) -> dict[str, tuple[list[str], Any]]:
    """The document's functionDefinitions, as the loader inlines them."""
    out: dict[str, tuple[list[str], Any]] = {}
    for i in range(sbml_model.getNumFunctionDefinitions()):
        fd = sbml_model.getFunctionDefinition(i)
        lam = fd.getMath()
        if lam is None:
            continue
        nargs = lam.getNumChildren() - 1
        out[fd.getId()] = ([lam.getChild(j).getName() for j in range(nargs)], lam.getChild(nargs))
    return out


def _inlined(node, sbml_model):
    """A copy of ``node`` with each call of a functionDefinition written out,
    so what it reads is seen: `f() = B` reads B. None for None."""
    import libsbml

    if node is None:
        return None
    copy = node.deepCopy()
    if sbml_model.getNumFunctionDefinitions():
        libsbml.SBMLTransforms.replaceFD(copy, sbml_model.getListOfFunctionDefinitions())
    return copy


def _reads_rate(node) -> bool:
    import libsbml

    if node is None:
        return False
    if node.getType() == libsbml.AST_FUNCTION_RATE_OF:
        return True
    return any(_reads_rate(node.getChild(i)) for i in range(node.getNumChildren()))


def _run_double(held: float | None, folded: float | None) -> str:
    """Why ``held``, the run's double for a quantity, cannot be taken, or ''
    where it can: a finite number its evaluator read, the same as this module's
    fold, ``folded``, where that folds at all."""
    if held is None:
        return "that the run's evaluator cannot read"
    if not math.isfinite(held):
        return f"of {held:g}, which is not a finite number"
    if folded is not None and folded != held:
        return (
            f"that the run computes as {held!r} and the translation folds to {folded!r}, "
            "in other arithmetic"
        )
    return ""


def _one_operation(node) -> bool:
    """Whether ``node`` is a number, a name, or one arithmetic operation on
    those: one rounding, which no evaluator can order otherwise."""
    import libsbml

    leaf = (
        libsbml.AST_INTEGER,
        libsbml.AST_REAL,
        libsbml.AST_REAL_E,
        libsbml.AST_NAME,
        libsbml.AST_CONSTANT_PI,
        libsbml.AST_CONSTANT_E,
    )
    if node.getType() in leaf:
        return True
    two = (libsbml.AST_PLUS, libsbml.AST_MINUS, libsbml.AST_TIMES, libsbml.AST_DIVIDE)
    kids = [node.getChild(k) for k in range(node.getNumChildren())]
    return node.getType() in two and 1 <= len(kids) <= 2 and all(k.getType() in leaf for k in kids)


def _run_start(core, sid: str, name_to_idx: dict[str, int]) -> float | None:
    """The start value the run holds for ``sid``, a parameter or compartment an
    event sets (held as a species), or None."""
    idx = _resolve_target(sid, name_to_idx)
    if idx is None:
        return None
    try:
        value = float(core.get_concentration(next(n for n, j in name_to_idx.items() if j == idx)))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _run_text(node, func_defs: dict[str, tuple[list[str], Any]]) -> str | None:
    """``node`` spelled for the run's evaluator, as the loader spells it, or
    None where it cannot be."""
    from bngsim._sbml_loader import _ast_to_exprtk_with_funcdefs

    try:
        return _ast_to_exprtk_with_funcdefs(node, func_defs)
    except Exception:
        return None


def _threshold_text(trigger: str) -> str | None:
    """The threshold side of a trigger as the loader writes it,
    ``(time()>=X)`` or ``(X<=time())`` and their strict twins, or None."""
    for op in (">=", ">"):
        head = f"(time(){op}"
        if trigger.startswith(head) and trigger.endswith(")"):
            return trigger[len(head) : -1]
    for op in ("<=", "<"):
        tail = f"{op}time())"
        if trigger.startswith("(") and trigger.endswith(tail):
            return trigger[1 : -len(tail)]
    return None


def _reads_clock(node) -> bool:
    import libsbml

    if node is None:
        return False
    if _is_time(node) or node.getType() == libsbml.AST_NAME_TIME:
        return True
    return any(_reads_clock(node.getChild(i)) for i in range(node.getNumChildren()))


def _unfolded_note(
    eid: str,
    what: str,
    node,
    env: dict[str, float],
    unfolded: dict[str, Any],
    species_ids: set[str],
    unruled: set[str] = frozenset(),  # type: ignore[assignment]
) -> str | None:
    """A note for ``what`` of event ``eid`` not folding because it reads a name
    that keeps one value but is set by an initialAssignment that does not fold,
    or by an assignment rule the run computes otherwise, or None."""
    if _reads_rate(node):
        return f"event {eid!r} {what} that reads rateOf, which the translation does not read"
    ruled = sorted(set(_dynamic_names(node, env)) & set(unruled))
    if ruled:
        return (
            f"event {eid!r} {what} that reads {ruled[0]!r}, which an assignment rule "
            "sets, and which the run computes in other arithmetic than the value the "
            "model holds before it runs"
        )
    names = sorted(set(_dynamic_names(node, env)) & set(unfolded))
    if not names:
        return None
    why = (
        "reads a species' initial value, which the translation does not fold"
        if _dynamic_names(unfolded[names[0]], {}) & species_ids
        else "does not fold to a number"
    )
    return (
        f"event {eid!r} {what} that reads {names[0]!r}, which keeps one value through "
        f"the run, but its initialAssignment {why}"
    )


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
    patterns) and ``lossy`` is empty. Otherwise ``protocol`` is ``None`` and
    ``lossy`` holds plain-English notes — the caller refuses (strict) or warns
    (best-effort): one per event that cannot be represented faithfully
    (state-triggered, a non-constant trigger time, delay or value, a delay or
    priority the run refuses, an unresolvable target), and one per species that
    events the run can execute together leave at a value their order decides
    (:func:`_unordered_writes`). A comp document is flattened first, as
    :meth:`Model.from_sbml` flattens it, and a model that runs an event the
    document does not declare is refused.

    The horizon for the trailing ``simulate`` phase comes from ``t_span``; if an
    event fires past it the horizon is extended so no scheduled change is dropped.
    """
    import libsbml

    from bngsim._sbml_loader import _doc_uses_comp, _flatten_comp

    sbml_path = Path(sbml_path)
    doc = libsbml.readSBMLFromFile(str(sbml_path))
    # Model.from_sbml flattens a comp document before it reads it, so a
    # submodel's events are the model's (``sub__E1``, setting ``sub__A``).
    # Unflattened, the top model has none of them to translate.
    if _doc_uses_comp(doc):
        _flatten_comp(doc, base_path=str(sbml_path.parent))
    sbml_model = doc.getModel()
    if sbml_model is None:  # pragma: no cover — Model.from_sbml already parsed it
        return None, ["could not re-read the source SBML to translate events"]

    n_events = sbml_model.getNumEvents()
    # Every event the model runs must be one read here, or it would be dropped
    # from the actions without a word. The loader adds an event of its own
    # where a piecewise compartment size jumps (``_size_jump_*``), which the
    # document does not declare; the rest are the document's. It skips an
    # event with no trigger or nothing to assign, so it can have fewer.
    n_model = sum(1 for comp in model._core.event_size_jump_compartments if not comp)
    if n_model > n_events:
        return None, [
            f"the source SBML read to translate the events declares {n_events}, but "
            f"the loaded model has {n_model}; an event not read cannot be carried "
            "into actions"
        ]
    if n_events == 0:
        return None, []

    env, own, unfolded = _const_env(sbml_model)
    # Each constant at the double the run holds it at (see _run_value).
    for sid in env:
        held = _run_param(model, sid)
        if held is not None:
            env[sid] = held
    species_ids = {sbml_model.getSpecies(i).getId() for i in range(sbml_model.getNumSpecies())}
    name_to_idx = _codegen_species_names(model)
    core = model._core
    func_defs = _function_definitions(sbml_model)
    # The run computes an assignment rule's name from the rule at every step,
    # where the model holds a fold of it before the run. Kept only where the
    # two agree, and where the rule reads nothing whose two differ.
    ruled = {
        sbml_model.getRule(j).getVariable(): sbml_model.getRule(j).getMath()
        for j in range(sbml_model.getNumRules())
        if sbml_model.getRule(j).isAssignment() and sbml_model.getRule(j).getVariable() in env
    }
    unruled: set[str] = set()
    changed = True
    while changed:
        changed = False
        for sid, rmath in ruled.items():
            if sid in unruled:
                continue
            body = _inlined(rmath, sbml_model)
            text = _run_text(body, func_defs) if body is not None else None
            held = _run_value(core, text, float(t_span[0])) if text else None
            if _dynamic_names(body, {}) & unruled or held != env[sid]:
                unruled.add(sid)
                changed = True
    for sid in unruled:
        env.pop(sid)
    # The loader names an event its id, or `_event_<i>` where it has none.
    run_events = {e["id"]: e for e in core.events_as_written()}

    t_start = float(t_span[0])
    lossy: list[str] = []
    fires: list[_Fire] = []

    for i in range(n_events):
        ev = sbml_model.getEvent(i)
        eid = ev.getId() or f"event #{i + 1}"
        run_ev = run_events.get(ev.getId() or f"_event_{i}")
        if run_ev is None:
            # The loader leaves out an event with nothing to assign; the run
            # has no such event, and neither have the actions.
            if not any(
                ev.getEventAssignment(j).isSetMath() for j in range(ev.getNumEventAssignments())
            ):
                continue
            lossy.append(f"event {eid!r} is not one the loaded model runs")
            continue
        trig = _inlined(ev.getTrigger().getMath() if ev.isSetTrigger() else None, sbml_model)
        threshold, reason = _trigger_fire_time(trig, env, species_ids)
        if threshold is None or _reads_rate(trig):
            note = _unfolded_note(eid, "has a trigger", trig, env, unfolded, species_ids, unruled)
            lossy.append(note or f"event {eid!r} {reason}")
            continue
        # Each time, priority and value is the double the run computes: its
        # evaluator's, from the doubles the model holds. Where this module's
        # fold gives another, it is refused (see _run_double).
        text = _threshold_text(run_ev["trigger"])
        held = _run_value(core, text, t_start) if text else None
        why = _run_double(held, None if math.isnan(threshold) else threshold)
        if why:
            lossy.append(f"event {eid!r} has a trigger time {why}")
            continue
        assert held is not None
        threshold = held
        # `time > T` (or `T < time`) is true only past T, not at it.
        strict = trig is not None and trig.getType() in (
            libsbml.AST_RELATIONAL_GT,
            libsbml.AST_RELATIONAL_LT,
        )

        # A trigger already true at the start fires there only if it was false
        # just before, which is what its initialValue says (true where unset,
        # as the loader reads it). Otherwise it never turns true, as time only
        # moves on, and the event never fires.
        fire_t, past, pinned = threshold, strict, False
        triggered: float | None = None
        if threshold < t_start or (threshold == t_start and not strict):
            trigger = ev.getTrigger()
            if not trigger.isSetInitialValue() or trigger.getInitialValue():
                continue
            fire_t, past, pinned = t_start, False, True
        read, delay = fire_t, 0.0

        # A constant delay shifts the fire time; a non-constant one is unschedulable.
        if ev.isSetDelay() and ev.getDelay() is not None and ev.getDelay().getMath() is not None:
            dmath = _inlined(ev.getDelay().getMath(), sbml_model)
            if _dynamic_names(dmath, env) or _reads_clock(dmath) or _reads_rate(dmath):
                note = _unfolded_note(
                    eid, "has a delay", dmath, env, unfolded, species_ids, unruled
                )
                lossy.append(
                    note or f"event {eid!r} has a non-constant delay (cannot be scheduled)"
                )
                continue
            # As the run reads it where it triggers.
            d = (
                _run_value(core, run_ev["delay_expr"], fire_t)
                if run_ev["delay_expr"]
                else float(run_ev["delay"])
            )
            # The run reads a delay within rounding of 0 below it as 0, and
            # refuses one further below, or not finite, where it fires.
            if d is not None and (not math.isfinite(d) or d < -1e-9 * max(1.0, abs(fire_t))):
                lossy.append(
                    f"event {eid!r} has a delay of {d:g}, which the run refuses: an event "
                    "delay must be a finite, non-negative number"
                )
                continue
            why = _run_double(d, _eval_const(dmath, env))
            if why:
                lossy.append(f"event {eid!r} has a delay {why}")
                continue
            assert d is not None
            d = max(d, 0.0)
            if d > 0:
                triggered = fire_t
            fire_t += d
            delay = d
            if not math.isfinite(fire_t):
                lossy.append(f"event {eid!r} is due at a time that is not a finite number")
                continue

        # An unset priority is 0, as the loader writes it, and so is one that
        # is a plain number 0: the run tells neither from no priority at all.
        # A priority that reads something that moves is not a constant (None).
        priority: float | None = 0.0
        unprioritised = not run_ev["priority_expr"] and run_ev["priority"] == 0
        pmath = _inlined(ev.getPriority().getMath() if ev.isSetPriority() else None, sbml_model)
        if pmath is not None and (
            set(_dynamic_names(pmath, env)) - set(unfolded)
            or _reads_clock(pmath)
            or _reads_rate(pmath)
        ):
            priority, unprioritised = None, False
        else:
            priority = (
                _run_value(core, run_ev["priority_expr"], fire_t)
                if run_ev["priority_expr"]
                else float(run_ev["priority"])
            )
            if priority is not None and math.isnan(priority):
                lossy.append(
                    f"event {eid!r} has a priority that is NaN, which the run refuses: an "
                    "event priority must be a number"
                )
                continue
            why = _run_double(priority, _eval_const(pmath, env) if pmath is not None else 0.0)
            if why:
                lossy.append(f"event {eid!r} has a priority {why}")
                continue

        assigns: list[tuple[int, float, str]] = []
        ok = True
        value_env = {**env, **own[i]} if i in own else env
        for j in range(ev.getNumEventAssignments()):
            ea = ev.getEventAssignment(j)
            var = ea.getVariable()
            vmath = _inlined(ea.getMath() if ea.isSetMath() else None, sbml_model)
            if (
                vmath is None
                or _reads_clock(vmath)
                or _reads_rate(vmath)
                or set(_dynamic_names(vmath, env)) - set(own.get(i, {}))
            ):
                val, why = None, ""
            elif _dynamic_names(vmath, env):
                # It reads the event's own target, at its start value, which the
                # run's evaluator reads as the target stands, so it cannot be
                # asked. Folded here, it is the run's double only as one
                # operation on the doubles the run holds.
                starts = {
                    sid: _run_start(core, sid, name_to_idx) for sid in _dynamic_names(vmath, env)
                }
                val = _eval_const(vmath, value_env)
                why = ""
                if not _one_operation(vmath) or any(
                    v != value_env[sid] for sid, v in starts.items()
                ):
                    why = (
                        "that reads its own start value in arithmetic the run may round otherwise"
                    )
                elif val is not None and not math.isfinite(val):
                    why = "that is not a finite number"
            else:
                text = _run_text(vmath, func_defs)
                val = _run_value(core, text, fire_t) if text else None
                why = _run_double(val, _eval_const(vmath, env))
            if why:
                lossy.append(f"event {eid!r} assigns {var!r} a value {why}")
                ok = False
                break
            if val is None:
                note = _unfolded_note(
                    eid,
                    f"assigns {var!r} a value",
                    vmath,
                    value_env,
                    unfolded,
                    species_ids,
                    unruled,
                )
                lossy.append(
                    note
                    or f"event {eid!r} assigns {var!r} a non-constant value — the "
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
            assigns.append((idx, val, var))
        if ok:
            fires.append(
                _Fire(
                    fire_t,
                    past,
                    priority,
                    unprioritised,
                    triggered,
                    i,
                    eid,
                    assigns,
                    read,
                    delay,
                    pinned,
                )
            )

    t_end = max([float(t_span[1]), *(f.time for f in fires)])
    if len(fires) > 1:
        import numpy as np

        stops = _run_stops(model, t_start, t_end)
        rows = [float(t) for t in np.linspace(t_start, float(t_span[1]), max(int(n_points), 2))]
        lossy.extend(_unordered_writes(fires, t_end - t_start, stops, rows))
    if lossy:
        return None, lossy
    if not fires:
        return None, []

    return _build_protocol(fires, model, t_span, n_points), []


def _build_protocol(
    fires: list[_Fire],
    model: Model,
    t_span: tuple[float, float],
    n_points: int,
) -> ProtocolSpec:
    """Assemble ``simulate`` phases + ``setConcentration`` state changes.

    Phases run from ``t_start`` through each distinct fire time (in order) to the
    horizon; the events at a fire time apply their assignments (in the order
    the run executes them: see :class:`_Fire`) after the phase that reaches it.
    ``continue=>1`` chains each phase onto the prior end-state. Output steps per
    phase are apportioned by duration.

    A change at the start, before the first phase, is in the first output
    row. The run's first row has it too for a trigger already true there,
    and not for a strict one on the start, `time > t0`, which it reads just
    past t0; no phase of no length can hold that row, so it is the one row
    the actions report otherwise.
    """
    species = model._core.codegen_data()["species"]
    t_start, t_end = float(t_span[0]), float(t_span[1])
    max_fire = max(f.time for f in fires)
    if max_fire > t_end:
        t_end = max_fire
    total = max(t_end - t_start, 1.0)
    base_steps = max(n_points - 1, 1)

    # group assignments by fire time, in execution order within a time
    by_time: dict[float, list[tuple[int, float, str]]] = {}
    for f in sorted(fires, key=_Fire.sort_key):
        by_time.setdefault(f.time, []).extend(f.assigns)

    steps: list[object] = []
    prev = t_start
    first = True
    for ft in sorted(by_time):
        if ft > prev:
            steps.append(_phase(prev, ft, total, base_steps, first))
            first = False
            prev = ft
        for sp_idx, val, _var in by_time[ft]:
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


def _phase(a: float, b: float, total: float, base_steps: int, first: bool) -> Experiment:
    n_steps = max(1, round(base_steps * (b - a) / total))
    extra: dict[str, Any] = {} if first else {"continue": 1}
    return Experiment(
        kind="simulate",
        method="ode",
        t_span=(a, b),
        n_points=n_steps + 1,
        extra=extra,
    )
