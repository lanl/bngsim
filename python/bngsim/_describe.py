"""What a model is, as one dict and its text (issue #1705).

:meth:`bngsim.Model.describe` builds the dict here from what the loader recorded
and the built model holds. It runs nothing and changes nothing.

The method verdicts read the checks the runs themselves make before they start,
so the two cannot disagree: :func:`ode_refusals` is the one the ODE Simulator
raises from, :func:`bngsim.validate_for_ssa` the one the SSA and PSA Simulators
raise from, and the two SSA refusals made in C++ (a model that reads ``rateOf``,
a model with no species) are read from the flags that C++ reads.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bngsim._model import Model

#: The version of the dict's layout. A change that removes or renames a key, or
#: changes what one holds, increments it.
SCHEMA_VERSION = 1

#: What the C++ SSA engine raises for a model that reads rateOf (ssa_simulator.cpp,
#: GH #106) and for one with no species: the run's own words.
_SSA_RATEOF = (
    "This model uses the SBML rateOf csymbol (instantaneous dx/dt), which "
    'is only supported for ODE simulation (method="ode"); rateOf has no '
    "well-defined value in a stochastic (SSA) trajectory."
)
_NO_SPECIES = "Cannot simulate: model has no species"


class ModelDescription(dict):
    """The dict :meth:`bngsim.Model.describe` returns. A plain ``dict`` in every
    respect (``json.dumps`` writes it, ``==`` compares it), except that ``str()``
    gives the text form, so ``print(model.describe())`` reads as a report."""

    def __str__(self) -> str:
        return _text(self)


# ─── Method verdicts ─────────────────────────────────────────────────────────


def ode_refusals(model: Model) -> list[tuple[str, str]]:
    """``(code, message)`` for each construct ``Simulator(model, method="ode")``
    refuses when it is made, in the order it checks them; the Simulator raises
    the first as a :class:`~bngsim.ModelError`.

    * ``fast_reaction``: an SBML reaction marked ``fast="true"``, a
      fast-equilibrium constraint no solver here honours (GH #113), unless
      ``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`` asks for the old approximation.
    * ``rateof_ring``: rates written in terms of each other through ``rateOf``,
      an algebraic loop with no right-hand side (lanl/bngsim#991).
    """
    out: list[tuple[str, str]] = []
    fast = [i for i in getattr(model, "_ssa_issues", None) or [] if i.code == "fast_reaction"]
    if fast and os.environ.get("BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS") != "1":
        locs = ", ".join(i.location for i in fast if i.location)
        out.append(
            (
                "fast_reaction",
                'Model contains fast="true" reaction(s) '
                f"[{locs}], a fast-equilibrium constraint bngsim cannot "
                "honor under ODE — the kinetic law would integrate as an "
                "ordinary reaction, silently ignoring the constraint "
                "(RoadRunner refuses the same model). To restore the legacy "
                "silent-approximation behavior, set "
                "BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1.",
            )
        )
    # Asked of a core that can say. The C++ simulator refuses the model too,
    # for a caller that comes by another way.
    ring = getattr(model._core, "rateof_ring_refusal", None)
    if ring is not None:
        message = ring()
        if message:
            out.append(("rateof_ring", message))
    return out


def steady_state_refusals(model: Model, where: str = "steady_state()") -> list[tuple[str, str]]:
    """``(code, message)`` for each reason a steady-state solve of ``model`` is
    refused whatever it asks, every one that holds, in the order
    :meth:`Simulator.steady_state` and :meth:`Simulator.steady_state_batch`
    check them; each raises the first as a
    :class:`~bngsim.SimulationError` (issues #710, #961, #1054). ``where`` names
    the call in the message.

    * ``event``: the model has an event, which a root of ``f(y)`` never fires.
    * ``size_jump``: a compartment an assignment rule sizes jumps where a
      piecewise the rule reads switches, and the solvers make no rescale there.
    * ``reads_clock``: the time reaches the state
      (``NetworkModel.state_reads_clock``), so there is no ``f(y) = 0`` to
      solve without a time being chosen.

    Not here: a conservation law the right-hand side does not keep (issue #758),
    which the call checks by finding the laws, gigabytes on a large network.
    """
    core = model._core
    # A compartment size that jumps keeps the amounts in it through an event
    # the loader adds (issue #1054), which is no event of the model's own.
    jumps = list(core.event_size_jump_compartments)
    n_events = sum(1 for comp in jumps if not comp)
    out: list[tuple[str, str]] = []
    if n_events:
        out.append(
            (
                "event",
                f"{where} is not supported for this model: it has {n_events} "
                f"event{'s' if n_events != 1 else ''}. The steady-state solvers look for a "
                "root of the right-hand side and fire no event, so an event's assignment is "
                "never made, and the state they return is one the model's dynamics may never "
                "reach (issue #710). Integrate the model with run() over a span long enough "
                "for the trajectory to settle, and read the state where it ends.",
            )
        )
    comps = sorted({comp for comp in jumps if comp})
    if comps:
        what = (
            f"the size of compartment {comps[0]}, which an assignment rule sets, jumps "
            "where a piecewise the rule reads switches, and the amounts in it are"
            if len(comps) == 1
            else f"the sizes of compartments {', '.join(comps)}, which assignment rules "
            "set, jump where a piecewise a rule reads switches, and the amounts in them are"
        )
        out.append(
            (
                "size_jump",
                f"{where} is not supported for this model: {what} kept across a jump by a "
                "rescale of their concentrations there (issue #1054). The steady-state "
                "solvers look for a root of the right-hand side and make no such rescale, so "
                "the state they return can hold the concentrations from the wrong side of a "
                "jump (issue #710). Integrate the model with run() over a span long enough for "
                "the trajectory to settle, and read the state where it ends.",
            )
        )
    if core.state_reads_clock:
        out.append(
            (
                "reads_clock",
                f"{where} is not supported for this model: it reads the time, through time() "
                "or a table function indexed by time, in a rate law, a rule a species or a "
                "compartment size is reported through, or what one of those reads. The "
                "steady-state solvers look for a root of f(y) with everything read at t = 0, "
                "which is the model's steady state only where nothing changes with the time "
                "after that: a rate that is 0 at t = 0 returned the initial state as "
                "converged, and a species set by a rule on the time its value at t = 0 "
                "(issue #710). Integrate the model with run() over a span long enough for the "
                "trajectory to settle, and read the state where it ends.",
            )
        )
    return out


#: What steady_state() raises for a model with no species, from the engine.
_SS_NO_SPECIES = "Steady-state computation failed: Cannot find steady state: model has no species"


def _steady_state_reasons(model: Model) -> list[dict[str, Any]]:
    """Why ``Simulator(model, method="ode").steady_state()`` is refused, in the
    order the refusals come: the ODE Simulator's own (a steady state is only
    asked of one), the call's checks, and the engine's for a model with no
    species."""
    reasons = [{"code": c, "message": m} for c, m in ode_refusals(model)]
    reasons += [{"code": c, "message": m} for c, m in steady_state_refusals(model)]
    if model._core.n_species == 0:
        reasons.append({"code": "no_species", "message": _SS_NO_SPECIES})
    return reasons


def _verdict(reasons: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"status": "refused" if reasons else "ok", "reasons": reasons, **extra}


def _by_code(issues: list, limit: int | None) -> list[dict[str, Any]]:
    """SsaIssues as one reason per code: its first message, how many there are,
    and where, cut at ``limit`` locations. A model can carry one per reaction,
    each a paragraph long (BIOMD0000000574: 830 KB of them)."""
    out: dict[str, dict[str, Any]] = {}
    for i in issues:
        entry = out.get(i.code)
        if entry is None:
            entry = out[i.code] = {"code": i.code, "message": i.message, "count": 0}
        entry["count"] += 1
        if i.location:
            entry.setdefault("locations", []).append(i.location)
    for entry in out.values():
        locations = entry.get("locations")
        if locations and limit is not None and len(locations) > limit:
            entry["locations_omitted"] = len(locations) - limit
            entry["locations"] = locations[:limit]
    return list(out.values())


def _stochastic(model: Model, limit: int | None) -> dict[str, Any]:
    """The SSA verdict, which PSA shares: the Simulator's validate_for_ssa
    errors, and the C++ engine's own two refusals."""
    from bngsim._ssa_validation import validate_for_ssa

    issues = validate_for_ssa(model)
    reasons = _by_code([i for i in issues if i.severity == "error"], limit)
    if model._core.n_species == 0:
        reasons.append({"code": "no_species", "message": _NO_SPECIES})
    if bool(getattr(model._core, "uses_rateof", False)):
        reasons.append({"code": "rateof", "message": _SSA_RATEOF})
    warnings = _by_code([i for i in issues if i.severity != "error"], limit)
    return _verdict(reasons, **({"warnings": warnings} if warnings else {}))


def _network_free(model: Model, canonical: str) -> dict[str, Any]:
    """A network-free method reads its rules from a BNG-XML file, never from the
    model, so the verdict is what it needs, or what this install lacks."""
    from bngsim._simulator import _UNAVAILABLE_NF_METHODS

    unavailable = _UNAVAILABLE_NF_METHODS.get(canonical)
    if unavailable is None and canonical == "nf_reject":
        from bngsim._bngsim_core import HAS_NFSIM

        if not HAS_NFSIM:
            unavailable = "NFsim support is not present in this bngsim install."
    if unavailable is not None:
        return {
            "status": "unavailable",
            "reasons": [{"code": "not_in_this_install", "message": unavailable}],
        }
    fmt = (model._source or {}).get("format")
    if fmt == "bngl":
        where = "BNG2.pl writes one for this model's .bngl with the writeXML() action."
    elif fmt in ("sbml", "antimony"):
        name = "SBML" if fmt == "sbml" else "Antimony"
        where = f"This model was loaded from {name}, which has no rules to give it."
    else:
        where = (
            "BNG2.pl writes one from the .bngl a network came from, with the writeXML() action."
        )
    return {
        "status": "needs_input",
        "reasons": [
            {
                "code": "needs_bng_xml",
                "message": "The network-free methods read the model's rules from a BNG-XML "
                "file, passed as xml_path=..., not from the loaded network. " + where,
            }
        ],
    }


#: What a steady-state verdict does not cover.
_STEADY_STATE_NOTE = (
    "From the checks steady_state() makes of the model alone. The call also refuses a "
    "conservation law the model's own right-hand side does not keep (issue #758), which "
    "it finds by computing the laws, not done here; and a solve can still fail to "
    "converge, or find a root the run does not reach."
)

#: Why a sensitivity verdict that nothing refuses is not "ok".
_UNDECIDED = (
    "Nothing in the model or this install refuses forward sensitivities before a "
    "run, but whether every rate law differentiates to closed form is decided when a "
    "sensitivity run builds its code, and that run refuses a model whose rate laws do "
    "not (a floor() in one, for instance). Other refusals depend on the parameters and "
    "span a run asks for, and the run raises them."
)


def _forward_sensitivities(model: Model) -> dict[str, Any]:
    """What a sensitivity run needs of this install (a codegen backend: a C
    compiler, or the MIR JIT), the events that refuse every request (the
    core's own check, asked as an initial-condition-only run asks it), and the
    parameters refused whatever the run: those folded at load.

    The status is ``"unavailable"``, ``"refused"``, or ``"undecided"``: never
    ``"ok"``, since whether the rate laws differentiate is decided only when a
    run builds the code (:data:`_UNDECIDED`)."""
    from bngsim import _codegen_toolchain
    from bngsim._bngsim_core import HAS_MIR
    from bngsim._codegen import _codegen_jit_backend

    install: list[dict[str, Any]] = []
    backend: str | None = None
    if os.environ.get("BNGSIM_NO_CODEGEN"):
        install.append(
            {
                "code": "codegen_disabled",
                "message": "Forward sensitivity analysis requires code generation, but "
                "BNGSIM_NO_CODEGEN is set.",
            }
        )
    elif _codegen_jit_backend() == "mir" and HAS_MIR:
        backend = "mir"
    else:
        # What a build would use and the verdict kept on it, with no probe.
        compiler, broken = _codegen_toolchain.peek()
        if compiler is None or broken is not None:
            install.append(
                {
                    "code": "no_compiler",
                    "message": (broken or "No C compiler found.")
                    + " Forward sensitivities compile the model, with a C compiler or, "
                    "where bngsim is built with it, the MIR JIT (BNGSIM_CODEGEN_JIT=mir).",
                }
            )
        else:
            backend = " ".join(compiler)
    model_reasons: list[dict[str, Any]] = []
    if model._core.n_events > 0:
        why = model._core.event_sensitivity_unsupported_reason([], [])
        if why:
            model_reasons.append({"code": "event", "message": str(why)})
    if install:
        status = "unavailable"
    elif model_reasons:
        status = "refused"
    else:
        status = "undecided"
    out: dict[str, Any] = {"status": status, "reasons": install + model_reasons}
    if status == "undecided":
        out["note"] = _UNDECIDED
    if backend is not None:
        out["backend"] = backend
    # Refused whatever the run asks: a parameter folded at load (what it went
    # into), and a compartment size set_param cannot write, whose column is the
    # derivative of that write (Simulator._raise_if_compartment_size_params).
    frozen = model._frozen_params
    unwritable = set(model.unwritable_compartment_size_params)
    # And a parameter a function owns, which no write moves
    # (Simulator._raise_if_function_backed_params, issue #329): for SBML the
    # file's own parameters an assignment rule sets, not the rates the loader
    # names after reactions.
    core = model._core
    backed = {
        n for n, internal in zip(core.param_names, core.param_is_internal, strict=True) if internal
    } & set(core.function_names)
    if model._sbml_record is not None:
        backed &= model._sbml_record["parameters"]

    def reason_for(n: str) -> str:
        if n in frozen:
            return f"folded at load into {frozen[n]}"
        if n in unwritable:
            return (
                "a compartment size set_param cannot write "
                "(Model.unwritable_compartment_size_params)"
            )
        return "set by a rule or function, which no write can move (issue #329)"

    refused = {
        n: reason_for(n)
        for n in model.param_names
        if n in frozen or n in unwritable or n in backed
    }
    if refused:
        out["refused_parameters"] = refused
    return out


# ─── The description ─────────────────────────────────────────────────────────

#: The most characters an expression keeps while ``limit`` is set.
TEXT_CAP = 200


def _check_limit(limit: Any) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(f"limit must be a non-negative int or None, not {limit!r}")
    return limit


def _cut(items: list, limit: int | None, key: str, omitted: dict[str, int]) -> list:
    if limit is None or len(items) <= limit:
        return items
    omitted[key] = len(items) - limit
    return items[:limit]


def _clip(text: str, limit: int | None) -> str:
    """``text``, or its first :data:`TEXT_CAP` characters and how many it lost."""
    if limit is None or len(text) <= TEXT_CAP:
        return text
    return f"{text[:TEXT_CAP]} ... ({len(text) - TEXT_CAP} more characters)"


def _term(name: str, coeff: float) -> tuple[str, str]:
    """A sign and a term of a linear combination: ``("+", "2*B")``."""
    sign = "-" if coeff < 0 else "+"
    mag = abs(coeff)
    return sign, name if mag == 1.0 else f"{_num(mag)}*{name}"


def _linear(entries: list, names: list[str], limit: int | None) -> str:
    """``A + 2*B - C`` for an observable's entries, cut after ``limit`` terms."""
    if not entries:
        return "0"
    shown = entries if limit is None else entries[:limit]
    parts: list[str] = []
    for i, (idx, coeff) in enumerate(shown):
        sign, term = _term(names[int(idx)], float(coeff))
        parts.append((f"-{term}" if sign == "-" else term) if i == 0 else f"{sign} {term}")
    text = " ".join(parts)
    if len(shown) < len(entries):
        text += f" + ... ({len(entries) - len(shown)} more terms)"
    return text


def _num(x: float) -> str:
    return f"{x:.6g}"


def _side(terms: list[tuple[str, Any]]) -> str:
    """One side of a reaction: ``2*A + B``, ``0`` when empty."""
    out = []
    for name, coeff in terms:
        if coeff is None or (isinstance(coeff, float) and coeff == 1.0):
            out.append(name)
        elif isinstance(coeff, float):
            out.append(f"{_num(coeff)}*{name}")
        else:
            out.append(f"({coeff})*{name}")
    return " + ".join(out) or "0"


def _counted(indices: list) -> list[tuple[int, float]]:
    """A ``.net`` reaction's species indices as ``(index, count)``, in order."""
    counts: dict[int, int] = {}
    for i in indices:
        counts[int(i)] = counts.get(int(i), 0) + 1
    return [(i, float(n)) for i, n in counts.items()]


def _parameters(model: Model, record: dict | None, limit: int | None) -> list[dict[str, Any]]:
    from bngsim._model import _names_in

    core = model._core
    names = list(core.param_names)
    is_expr = list(core.param_is_expression)
    exprs = list(core.param_expressions)
    internal = list(core.param_is_internal)
    comp = list(getattr(core, "param_is_compartment_size", [False] * len(names)))
    known = set(names) | set(core.species_names) | set(core.observable_names)
    known |= set(core.function_names)
    # An SBML model's own parameters, compartments and reaction-local ones: not
    # the ones its loader made (a reversible law's _rateLaw_*, an _ic_ carrier).
    declared = None if record is None else record["parameters"]
    frozen = model._frozen_params
    out: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        if internal[i]:
            continue
        if declared is not None and name not in declared and not name.startswith("_lp_"):
            continue
        entry: dict[str, Any] = {"name": name, "value": float(core.get_param(name))}
        if is_expr[i]:
            entry["kind"] = "derived"
            entry["expression"] = _clip(exprs[i], limit)
        elif comp[i]:
            entry["kind"] = "compartment_size"
        elif exprs[i] and any(n in known and n != name for n in _names_in(exprs[i])):
            # A derived parameter a set_param pinned to a number (issue #188):
            # its expression stays on record, and it no longer follows it.
            entry["kind"] = "derived"
            entry["expression"] = _clip(exprs[i], limit)
            entry["overridden"] = True
        else:
            entry["kind"] = "primary"
        if name in frozen:
            entry["frozen"] = frozen[name]
        out.append(entry)
    return out


def _species(
    model: Model, cd: dict, record: dict | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The species, and for an SBML model the other states: a compartment size
    or a parameter a rule or an event moves, which the loader holds as a state
    (GH #85, #131) and the file does not declare as a species."""
    core = model._core
    names = list(core.species_names)
    initial = [float(v) for v in core.get_initial_state()]
    meta = list(cd["species"])
    ar = model._ar_report_map
    rate_rule = {
        int(p) for r in cd["reactions"] if r.get("is_rate_rule_ode") for p in r.get("products", ())
    }
    by_event = {int(i) for i in core.event_assigned_species}
    in_comp = None if record is None else record["species"]
    species: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        sp = meta[i]
        entry: dict[str, Any] = {"name": name, "initial": initial[i]}
        set_by = [
            kind
            for kind, hit in (
                ("assignment_rule", name in ar),
                ("rate_rule", i in rate_rule),
                ("event", i in by_event),
            )
            if hit
        ]
        if in_comp is not None and name not in in_comp:
            if name in record["compartments"]:  # type: ignore[index]
                entry["holds"] = "compartment_size"
            elif name in record["parameters"]:  # type: ignore[index]
                entry["holds"] = "parameter"
            else:
                entry["holds"] = "other"
            if set_by:
                entry["set_by"] = set_by
            if not sp.get("reported", True):
                entry["reported"] = False
            other.append(entry)
            continue
        entry["fixed"] = bool(sp["fixed"])
        if set_by:
            entry["set_by"] = set_by
        if not sp.get("reported", True):
            # Integrator state with no column in Result.species (GH #71).
            entry["reported"] = False
        if in_comp is not None:
            entry["compartment"] = in_comp[name]
            if sp.get("amount_valued"):
                # The stored value is the amount over the size (loader
                # section 3); the factor follows a write of the size (#170).
                entry["quantity"] = "amount"
                entry["initial_amount"] = initial[i] * float(sp["volume_factor"])
            else:
                entry["quantity"] = "concentration"
        species.append(entry)
    return species, other


_RATE_LAWS = {"elementary": "elementary", "mm": "michaelis_menten", "functional": "functional"}


def _reactions(
    model: Model, cd: dict, record: dict | None, limit: int | None
) -> tuple[list[dict], int]:
    """The reactions as the file writes them, cut at ``limit``, and how many
    there are: an SBML file's from the loader's record (each shown law put in
    the model's names here, not at load), and a network's lines from the
    model, which holds them as written."""
    if record is not None:
        from bngsim._sbml_loader import sbml_rates_in_model_names

        out = []
        shown = record["reactions"] if limit is None else record["reactions"][:limit]
        rates = sbml_rates_in_model_names(shown, record["renamed"])
        for r, rate in zip(shown, rates, strict=True):
            entry: dict[str, Any] = {
                "id": r["id"],
                "reactants": [n for n, _ in r["reactants"]],
                "products": [n for n, _ in r["products"]],
                "equation": _clip(f"{_side(r['reactants'])} -> {_side(r['products'])}", limit),
                "rate_law": "kinetic_law",
                "rate": _clip(rate, limit),
                "reversible": r["reversible"],
            }
            if r["modifiers"]:
                entry["modifiers"] = list(r["modifiers"])
            if r["fast"]:
                entry["fast"] = True
            out.append(entry)
        return out, len(record["reactions"])
    core = model._core
    species = list(core.species_names)
    params = list(core.param_names)
    functions = dict(zip(core.function_names, core.function_expressions, strict=True))
    out = []
    for r in cd["reactions"]:
        lhs = [(species[i], n) for i, n in _counted(r["reactants"])]
        rhs = [(species[i], n) for i, n in _counted(r["products"])]
        entry = {
            "reactants": [n for n, _ in lhs],
            "products": [n for n, _ in rhs],
            "equation": _clip(f"{_side(lhs)} -> {_side(rhs)}", limit),
        }
        if r.get("is_rate_rule_ode"):
            entry["rate_law"] = "rate_rule"
        else:
            entry["rate_law"] = _RATE_LAWS.get(str(r["type"]), str(r["type"]))
        fn = str(r.get("function_name") or "")
        if r["type"] == "mm":
            entry["rate"] = ", ".join(params[int(i)] for i in r["rate_param_indices"])
        else:
            entry["rate"] = fn
            if fn in functions:
                entry["expression"] = _clip(functions[fn], limit)
        factor = float(r.get("stat_factor", 1.0))
        if factor != 1.0:
            entry["multiplier"] = factor
        out.append(entry)
    return out if limit is None else out[:limit], len(out)


def _compartments(model: Model, record: dict) -> list[dict[str, Any]]:
    """Each compartment's size as a run from the initial state starts with it:
    the parameter's value, which a ``set_param`` moves (issue #170); a state's
    initial value, for a size a rate rule or an event moves; the size at load,
    for one an assignment rule sets during the run."""
    core = model._core
    params = set(core.param_names)
    species = list(core.species_names)
    initial = core.get_initial_state()
    out = []
    for name, info in record["compartments"].items():
        entry: dict[str, Any] = {"name": name}
        if info["id"] != name:
            entry["id"] = info["id"]
        if name in species:
            entry["size"] = float(initial[species.index(name)])
        elif name in params and "assignment_rule" not in info["size_set_by"]:
            entry["size"] = float(core.get_param(name))
        else:
            entry["size"] = float(info["size"])
        entry["size_declared"] = info["size_declared"]
        entry["size_set_by"] = list(info["size_set_by"])
        out.append(entry)
    return out


def _events(model: Model, limit: int | None) -> list[dict[str, Any]]:
    species = list(model._core.species_names)
    out: list[dict[str, Any]] = []
    for ev in model._core.events_as_written():
        entry: dict[str, Any] = {
            "id": str(ev["id"]),
            "trigger": _clip(str(ev["trigger"]), limit),
            "assigns": [species[int(sp)] for sp, _ in ev["assignments"]],
        }
        delay_expr = str(ev.get("delay_expr") or "")
        delay = float(ev.get("delay") or 0.0)
        if delay_expr:
            entry["delay"] = _clip(delay_expr, limit)
        elif delay != 0.0:
            entry["delay"] = delay
        out.append(entry)
    return out


def describe_model(model: Model, *, limit: int | None = 50) -> ModelDescription:
    """Build :meth:`bngsim.Model.describe`'s dict. See that method."""
    from bngsim._simulator import _start_record

    limit = _check_limit(limit)
    core = model._core
    cd = core.codegen_data()
    record = model._sbml_record
    omitted: dict[str, int] = {}

    source = dict(model._source) if model._source is not None else None
    if source is None:
        source = {"format": None, "path": None, "sha256": None}

    parameters = _parameters(model, record, limit)
    species, other_states = _species(model, cd, record)
    species_names = list(core.species_names)  # once: each read converts them all
    observables = [
        {
            "name": o["name"],
            "reads": _clip(_linear(list(o["entries"]), species_names, limit), limit),
        }
        for o in cd["observables"]
    ]
    # An SBML model's functions are its assignment rules and its reactions'
    # rates, by the file's ids; the loader's own (a rate rule's _rhs_, a
    # dilution's _dil_, a volume-scaled _vd_ rate) are left out.
    ids = None if record is None else record["ids"]
    functions = [
        {"name": n, "expression": _clip(e, limit)}
        for n, e in zip(core.function_names, core.function_expressions, strict=True)
        if ids is None or n in ids
    ]
    reactions, n_reactions = _reactions(model, cd, record, limit)
    events = _events(model, limit)
    compartments = None if record is None else _compartments(model, record)
    time_tables = [
        str(t["name"]) for t in core.table_function_tables if t.get("index_kind") == "time"
    ]

    # Each count is that of its list, before the cut.
    counts: dict[str, int] = {
        "species": len(species),
        "reactions": n_reactions,
        "observables": len(observables),
        "parameters": len(parameters),
        "functions": len(functions),
        "events": len(events),
        "table_functions": int(core.n_table_functions),
    }
    if compartments is not None:
        counts["compartments"] = len(compartments)
    if other_states:
        counts["other_states"] = len(other_states)
    if record is not None and record["function_definitions"]:
        counts["function_definitions"] = len(record["function_definitions"])

    stochastic = _stochastic(model, limit)
    decline = model.analytical_jacobian_decline
    out = ModelDescription(
        schema_version=SCHEMA_VERSION,
        source=source,
        counts=counts,
        next_run_starts_from=_start_record(core)[0],
        species_values="concentration" if record is not None else None,
        methods={
            "ode": _verdict(
                [{"code": c, "message": m} for c, m in ode_refusals(model)],
            ),
            "ssa": stochastic,
            "psa": dict(stochastic),
            "nf": _network_free(model, "nf_reject"),
            "rm": _network_free(model, "nf_exact"),
            "forward_sensitivities": _forward_sensitivities(model),
            "steady_state": _verdict(_steady_state_reasons(model), note=_STEADY_STATE_NOTE),
        },
        time_dependence={
            "rate_laws_read_clock": bool(core.rate_laws_read_clock),
            "functions_reading_clock": list(core.functions_reading_clock),
            "state_reads_clock": bool(core.state_reads_clock),
            "time_tables": time_tables,
            "events": int(core.n_events),
        },
        jacobian={
            "analytical": model.analytical_jacobian_status.split(":", 1)[0],
            **({"decline": decline} if decline is not None else {}),
        },
    )
    out["parameters"] = _cut(parameters, limit, "parameters", omitted)
    if compartments is not None:
        out["compartments"] = _cut(compartments, limit, "compartments", omitted)
    out["species"] = _cut(species, limit, "species", omitted)
    if other_states:
        out["other_states"] = _cut(other_states, limit, "other_states", omitted)
    out["observables"] = _cut(observables, limit, "observables", omitted)
    out["functions"] = _cut(functions, limit, "functions", omitted)
    out["reactions"] = reactions  # cut by _reactions
    if n_reactions > len(reactions):
        omitted["reactions"] = n_reactions - len(reactions)
    out["events"] = _cut(events, limit, "events", omitted)
    definitions = [] if record is None else record["function_definitions"]
    if definitions:
        out["function_definitions"] = _cut(
            [
                {"name": f["name"], "definition": _clip(f["definition"], limit)}
                for f in definitions
            ],
            limit,
            "function_definitions",
            omitted,
        )
    if omitted:
        out["omitted"] = omitted
    return out


# ─── The text form ───────────────────────────────────────────────────────────


_SINGULAR = {
    "species": "species",
    "table_functions": "table function",
    "other_states": "other state",
    "function_definitions": "function definition",
}


def _count(n: int, key: str) -> str:
    """``"1 table function"``, ``"3 species"``, ``"2 reactions"``."""
    if n == 1:
        return f"1 {_SINGULAR.get(key, key[:-1])}"
    return f"{n} {key.replace('_', ' ')}"


def _status_line(verdict: dict[str, Any]) -> str:
    status = verdict["status"]
    reasons = verdict.get("reasons") or []
    if status == "ok":
        extra = []
        if verdict.get("backend"):
            extra.append(f"backend: {verdict['backend']}")
        if verdict.get("warnings"):
            codes = ", ".join(w["code"] for w in verdict["warnings"])
            extra.append(f"warnings: {codes}")
        return "ok" + (f" ({'; '.join(extra)})" if extra else "")
    if status == "undecided":
        via = f" (backend: {verdict['backend']})" if verdict.get("backend") else ""
        return (
            f"undecided{via}: nothing refuses them before a run; whether the rate laws "
            "differentiate is decided when a sensitivity run builds its code"
        )
    head = {"refused": "refused", "needs_input": "needs", "unavailable": "unavailable"}.get(
        status, status
    )

    def one(r: dict) -> str:
        more = f" ({r['count']} like it)" if r.get("count", 1) > 1 else ""
        return f"[{r['code']}] {r['message']}{more}"

    return f"{head}: " + "; ".join(one(r) for r in reasons)


def _text(d: dict[str, Any]) -> str:
    lines: list[str] = []
    src = d["source"]
    if src.get("format") is None:
        lines.append("Model built in Python (no source file recorded)")
    else:
        where = src.get("path") or "a string"
        sha = src.get("sha256")
        lines.append(
            f"Model from {src['format']}: {where}" + (f" (sha256 {sha[:12]})" if sha else "")
        )
        if src.get("network"):
            lines.append(f"  network: {src['network']}")
        if src.get("compartment_sizes"):
            sizes = ", ".join(f"{k}={_num(v)}" for k, v in src["compartment_sizes"].items())
            lines.append(f"  compartment sizes set at load: {sizes}")
    c = d["counts"]
    order = (
        "species",
        "reactions",
        "observables",
        "parameters",
        "functions",
        "events",
        "compartments",
        "other_states",
        "function_definitions",
        "table_functions",
    )
    # The first four always; the rest where the model has any.
    shown = [k for i, k in enumerate(order) if k in c and (i < 4 or c[k])]
    lines.append("Size: " + ", ".join(_count(c[k], k) for k in shown))
    starts = d["next_run_starts_from"]
    lines.append(
        "Next run starts from: "
        + (
            "the initial state"
            if starts == "initial"
            else "the model's current state, which is not its initial state (a run or "
            "set_concentration moved it; reset() restores the initial state)"
        )
    )
    if d.get("species_values") == "concentration":
        lines.append(
            "Species values are concentrations (amount / compartment size), for every "
            "species the file declares."
        )
    elif src.get("format") in ("net", "bngl"):
        lines.append(
            f"Species values are as the {src['format']} file gives them; it states no units."
        )
    else:
        lines.append("Species values carry no stated units.")

    lines.append("Methods:")
    labels = {
        "ode": "ode",
        "ssa": "ssa",
        "psa": "psa",
        "nf": "nf",
        "rm": "rm",
        "forward_sensitivities": "forward sensitivities",
        "steady_state": "steady state",
    }
    # Methods with the same verdict share a line: nf and rm, ssa and psa.
    grouped: dict[str, list[str]] = {}
    for key, label in labels.items():
        status = _status_line(d["methods"][key])
        if key == "steady_state" and status == "ok":
            # What the verdict leaves to the call (its note), said where it is read.
            status = "ok, if a solve converges (the call also checks the conservation laws)"
        grouped.setdefault(status, []).append(label)
    for status, names in grouped.items():
        lines.append(f"  {', '.join(names)}: {status}")
    refused = d["methods"]["forward_sensitivities"].get("refused_parameters")
    if refused:
        lines.append("  forward sensitivities refuse the parameters: " + ", ".join(refused))

    j = d["jacobian"]
    lines.append(
        f"Analytical Jacobian: {j['analytical']}"
        + (f" ({j['decline'].get('kind')})" if j.get("decline") else "")
    )
    t = d["time_dependence"]
    clock = []
    if t["rate_laws_read_clock"]:
        clock.append("rate laws read the clock")
    if t["functions_reading_clock"]:
        clock.append("functions that read it: " + ", ".join(t["functions_reading_clock"]))
    if t["state_reads_clock"]:
        clock.append("the state reads it (no steady state)")
    if t["time_tables"]:
        clock.append("tables over time: " + ", ".join(t["time_tables"]))
    lines.append("Time: " + ("; ".join(clock) if clock else "nothing reads the clock"))

    omitted = d.get("omitted", {})

    def section(key: str, title: str, render) -> None:
        items = d.get(key)
        if items is None:
            return
        total = len(items) + omitted.get(key, 0)
        if total == 0:
            return
        lines.append(f"{title} ({total}):")
        for item in items:
            lines.append("  " + render(item))
        if omitted.get(key):
            lines.append(f"  ... {omitted[key]} more not shown (describe(limit=None) lists all)")

    def param(p: dict) -> str:
        s = f"{p['name']} = {_num(p['value'])}  {p['kind'].replace('_', ' ')}"
        if p.get("expression"):
            s += f": {p['expression']}"
        if p.get("overridden"):
            s += " (overridden by set_param)"
        if p.get("frozen"):
            s += f"  [folded at load into {p['frozen']}]"
        return s

    def compartment(cp: dict) -> str:
        s = f"{cp['name']}  size {_num(cp['size'])}"
        if cp.get("id"):
            s += f" (the file's {cp['id']})"
        if not cp.get("size_declared", True):
            s += " (not given in the file; 1 used)"
        if cp.get("size_set_by"):
            s += ", set by " + ", ".join(k.replace("_", " ") for k in cp["size_set_by"])
        return s

    def species(sp: dict) -> str:
        s = f"{sp['name']}  {_num(sp['initial'])}"
        bits = []
        if "compartment" in sp:
            bits.append(f"in {sp['compartment']}")
        if sp.get("quantity") == "amount":
            bits.append(f"declared as an amount: initial amount {_num(sp['initial_amount'])}")
        if sp.get("holds"):
            bits.append(f"a {sp['holds'].replace('_', ' ')} held as a state")
        if sp.get("reported") is False:
            bits.append("no column in Result.species")
        if sp.get("fixed"):
            bits.append("fixed")
        if sp.get("set_by"):
            bits.append("set by " + ", ".join(k.replace("_", " ") for k in sp["set_by"]))
        return s + (f"  ({'; '.join(bits)})" if bits else "")

    def reaction(r: dict) -> str:
        rate = r["rate"]
        if r.get("expression"):
            rate = f"{rate} = {r['expression']}" if rate else r["expression"]
        if r.get("multiplier"):
            rate = f"{_num(r['multiplier'])}*{rate}"
        head = f"{r['id']}: " if r.get("id") else ""
        tail = "".join(
            f"  [{w}]"
            for w, on in (("reversible", r.get("reversible")), ("fast", r.get("fast")))
            if on
        )
        return f"{head}{r['equation']}  {r['rate_law'].replace('_', ' ')}  {rate}{tail}"

    def event(e: dict) -> str:
        s = f"{e['id'] or '(unnamed)'}: when {e['trigger']}"
        if e.get("delay") is not None:
            s += f", after {e['delay']}"
        if e["assigns"]:
            s += " sets " + ", ".join(e["assigns"])
        return s

    section("compartments", "Compartments", compartment)
    section("species", "Species", species)
    section("other_states", "Other states", species)
    section("parameters", "Parameters", param)
    section("observables", "Observables", lambda o: f"{o['name']} = {o['reads']}")
    section("functions", "Functions", lambda f: f"{f['name']} = {f['expression']}")
    section("reactions", "Reactions", reaction)
    section("events", "Events", event)
    section(
        "function_definitions",
        "Function definitions (a rate calls them by name)",
        lambda f: f"{f['name']} = {f['definition']}",
    )
    return "\n".join(lines)


__all__ = [
    "SCHEMA_VERSION",
    "ModelDescription",
    "describe_model",
    "ode_refusals",
    "steady_state_refusals",
]
