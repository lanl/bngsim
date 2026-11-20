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


def _verdict(reasons: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"status": "refused" if reasons else "ok", "reasons": reasons, **extra}


def _stochastic(model: Model) -> dict[str, Any]:
    """The SSA verdict, which PSA shares: the Simulator's validate_for_ssa
    errors, and the C++ engine's own two refusals."""
    from bngsim._ssa_validation import validate_for_ssa

    issues = validate_for_ssa(model)
    reasons = [
        {"code": i.code, "message": i.message, **({"location": i.location} if i.location else {})}
        for i in issues
        if i.severity == "error"
    ]
    if model._core.n_species == 0:
        reasons.append({"code": "no_species", "message": _NO_SPECIES})
    if bool(getattr(model._core, "uses_rateof", False)):
        reasons.append({"code": "rateof", "message": _SSA_RATEOF})
    warnings = [
        {"code": i.code, "message": i.message, **({"location": i.location} if i.location else {})}
        for i in issues
        if i.severity != "error"
    ]
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


def _forward_sensitivities(model: Model) -> dict[str, Any]:
    """What a sensitivity run needs of this install (a codegen backend: a C
    compiler, or the MIR JIT), and the parameters it refuses whatever the run:
    those folded at load. Its other refusals depend on the run's parameters and
    span, and :meth:`Simulator.run` raises them."""
    from bngsim import _codegen_toolchain
    from bngsim._bngsim_core import HAS_MIR
    from bngsim._codegen import _codegen_jit_backend

    reasons: list[dict[str, Any]] = []
    backend: str | None = None
    if os.environ.get("BNGSIM_NO_CODEGEN"):
        reasons.append(
            {
                "code": "codegen_disabled",
                "message": "Forward sensitivity analysis requires code generation, but "
                "BNGSIM_NO_CODEGEN is set.",
            }
        )
    elif _codegen_jit_backend() == "mir" and HAS_MIR:
        backend = "mir"
    else:
        compiler, broken = _codegen_toolchain.check()
        if compiler is None or broken is not None:
            reasons.append(
                {
                    "code": "no_compiler",
                    "message": (broken or "No C compiler found.")
                    + " Forward sensitivities compile the model, with a C compiler or, "
                    "where bngsim is built with it, the MIR JIT (BNGSIM_CODEGEN_JIT=mir).",
                }
            )
        elif (missing := _codegen_toolchain.record_missing(compiler)) is not None:
            reasons.append({"code": "no_compiler", "message": missing})
        else:
            backend = " ".join(compiler)
    out: dict[str, Any] = {
        "status": "unavailable" if reasons else "ok",
        "reasons": reasons,
    }
    if backend is not None:
        out["backend"] = backend
    frozen = model._frozen_params
    if frozen:
        out["refused_parameters"] = {n: frozen[n] for n in model.param_names if n in frozen}
    return out


# ─── The description ─────────────────────────────────────────────────────────


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


def _parameters(model: Model, cd: dict) -> list[dict[str, Any]]:
    from bngsim._model import _names_in

    core = model._core
    names = list(core.param_names)
    is_expr = list(core.param_is_expression)
    exprs = list(core.param_expressions)
    internal = list(core.param_is_internal)
    comp = list(getattr(core, "param_is_compartment_size", [False] * len(names)))
    known = set(names) | set(core.species_names) | set(core.observable_names)
    known |= set(core.function_names)
    frozen = model._frozen_params
    out: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        if internal[i]:
            continue
        entry: dict[str, Any] = {"name": name, "value": float(core.get_param(name))}
        if is_expr[i]:
            entry["kind"] = "derived"
            entry["expression"] = exprs[i]
        elif comp[i]:
            entry["kind"] = "compartment_size"
        elif exprs[i] and any(n in known and n != name for n in _names_in(exprs[i])):
            # A derived parameter a set_param pinned to a number (issue #188):
            # its expression stays on record, and it no longer follows it.
            entry["kind"] = "derived"
            entry["expression"] = exprs[i]
            entry["overridden"] = True
        else:
            entry["kind"] = "primary"
        if name in frozen:
            entry["frozen"] = frozen[name]
        out.append(entry)
    return out


def _species(model: Model, cd: dict, record: dict | None) -> list[dict[str, Any]]:
    core = model._core
    names = list(core.species_names)
    initial = [float(v) for v in core.get_initial_state()]
    meta = list(cd["species"])
    ar = model._ar_report_map
    rate_rule = {
        int(p) for r in cd["reactions"] if r.get("is_rate_rule_ode") for p in r.get("products", ())
    }
    by_event = {int(i) for i in core.event_assigned_species}
    in_comp = (record or {}).get("species", {})
    out: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        sp = meta[i]
        entry: dict[str, Any] = {"name": name, "initial": initial[i], "fixed": bool(sp["fixed"])}
        if not sp.get("reported", True):
            # Integrator state with no column in Result.species (GH #71): an
            # event-assigned parameter or compartment the loader promoted.
            entry["reported"] = False
        set_by = [
            kind
            for kind, hit in (
                ("assignment_rule", name in ar),
                ("rate_rule", i in rate_rule),
                ("event", i in by_event),
            )
            if hit
        ]
        if set_by:
            entry["set_by"] = set_by
        if record is not None:
            comp = in_comp.get(name)
            if comp is None:
                # A parameter or compartment size a rule or an event moves, held
                # as a state (GH #85, #131); not a species of the file.
                entry["sbml_species"] = False
            else:
                entry["compartment"] = comp
                if sp.get("amount_valued"):
                    # The stored value is the amount over the size (loader
                    # section 3); the factor follows a write of the size (#170).
                    entry["quantity"] = "amount"
                    entry["initial_amount"] = initial[i] * float(sp["volume_factor"])
                else:
                    entry["quantity"] = "concentration"
        out.append(entry)
    return out


_RATE_LAWS = {"elementary": "elementary", "mm": "michaelis_menten", "functional": "functional"}


def _reactions(model: Model, cd: dict) -> list[dict[str, Any]]:
    core = model._core
    species = list(core.species_names)
    params = list(core.param_names)
    functions = dict(zip(core.function_names, core.function_expressions, strict=True))
    out: list[dict[str, Any]] = []
    for r in cd["reactions"]:
        entry: dict[str, Any] = {
            "reactants": [species[int(i)] for i in r["reactants"]],
            "products": [species[int(i)] for i in r["products"]],
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
                entry["expression"] = functions[fn]
        factor = float(r.get("stat_factor", 1.0))
        if factor != 1.0:
            entry["multiplier"] = factor
        out.append(entry)
    return out


def _events(model: Model) -> list[dict[str, Any]]:
    species = list(model._core.species_names)
    out: list[dict[str, Any]] = []
    for ev in model._core.events_as_written():
        entry: dict[str, Any] = {
            "id": str(ev["id"]),
            "trigger": str(ev["trigger"]),
            "assigns": [species[int(sp)] for sp, _ in ev["assignments"]],
        }
        delay_expr = str(ev.get("delay_expr") or "")
        delay = float(ev.get("delay") or 0.0)
        if delay_expr:
            entry["delay"] = delay_expr
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
    record = model._compartments
    omitted: dict[str, int] = {}

    source = dict(model._source) if model._source is not None else None
    if source is None:
        source = {"format": None, "path": None, "sha256": None}

    parameters = _parameters(model, cd)
    species = _species(model, cd, record)
    species_names = list(core.species_names)  # once: each read converts them all
    observables = [
        {"name": o["name"], "reads": _linear(list(o["entries"]), species_names, limit)}
        for o in cd["observables"]
    ]
    functions = [
        {"name": n, "expression": e}
        for n, e in zip(core.function_names, core.function_expressions, strict=True)
    ]
    reactions = _reactions(model, cd)
    events = _events(model)
    compartments = (
        None
        if record is None
        else [{"name": name, **dict(info)} for name, info in record["compartments"].items()]
    )
    laws = model.conservation_laws
    time_tables = [
        str(t["name"]) for t in core.table_function_tables if t.get("index_kind") == "time"
    ]

    counts: dict[str, int] = {
        "species": int(core.n_species),
        "reactions": int(core.n_reactions),
        "observables": int(core.n_observables),
        "parameters": len(parameters),
        "functions": int(core.n_functions),
        "events": int(core.n_events),
        "table_functions": int(core.n_table_functions),
        "conservation_laws": int(laws["n_laws"]),
    }
    if compartments is not None:
        counts["compartments"] = len(compartments)

    stochastic = _stochastic(model)
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
    out["observables"] = _cut(observables, limit, "observables", omitted)
    out["functions"] = _cut(functions, limit, "functions", omitted)
    out["reactions"] = _cut(reactions, limit, "reactions", omitted)
    out["events"] = _cut(events, limit, "events", omitted)
    warnings = list(core.load_warnings)
    if warnings:
        out["load_warnings"] = warnings
    if omitted:
        out["omitted"] = omitted
    return out


# ─── The text form ───────────────────────────────────────────────────────────


_SINGULAR = {
    "species": "species",
    "conservation_laws": "conservation law",
    "table_functions": "table function",
}


def _count(n: int, key: str) -> str:
    """``"1 conservation law"``, ``"3 species"``, ``"2 reactions"``."""
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
    head = {"refused": "refused", "needs_input": "needs", "unavailable": "unavailable"}.get(
        status, status
    )
    return f"{head}: " + "; ".join(f"[{r['code']}] {r['message']}" for r in reasons)


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
        "table_functions",
        "conservation_laws",
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
    }
    # Methods with the same verdict share a line: nf and rm, ssa and psa.
    grouped: dict[str, list[str]] = {}
    for key, label in labels.items():
        grouped.setdefault(_status_line(d["methods"][key]), []).append(label)
    for status, names in grouped.items():
        lines.append(f"  {', '.join(names)}: {status}")
    refused = d["methods"]["forward_sensitivities"].get("refused_parameters")
    if refused:
        lines.append(
            "  forward sensitivities refuse the parameters folded at load: " + ", ".join(refused)
        )

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
        if sp.get("sbml_species") is False:
            bits.append("a size or parameter held as a state")
        if sp.get("reported") is False:
            bits.append("no column in Result.species")
        if sp.get("fixed"):
            bits.append("fixed")
        if sp.get("set_by"):
            bits.append("set by " + ", ".join(k.replace("_", " ") for k in sp["set_by"]))
        return s + (f"  ({'; '.join(bits)})" if bits else "")

    def reaction(r: dict) -> str:
        lhs = " + ".join(r["reactants"]) or "0"
        rhs = " + ".join(r["products"]) or "0"
        rate = r["rate"]
        if r.get("expression"):
            rate = f"{rate} = {r['expression']}" if rate else r["expression"]
        if r.get("multiplier"):
            rate = f"{_num(r['multiplier'])}*{rate}"
        return f"{lhs} -> {rhs}  {r['rate_law'].replace('_', '-')}  {rate}"

    def event(e: dict) -> str:
        s = f"{e['id'] or '(unnamed)'}: when {e['trigger']}"
        if e.get("delay") is not None:
            s += f", after {e['delay']}"
        if e["assigns"]:
            s += " sets " + ", ".join(e["assigns"])
        return s

    section("compartments", "Compartments", compartment)
    section("species", "Species", species)
    section("parameters", "Parameters", param)
    section("observables", "Observables", lambda o: f"{o['name']} = {o['reads']}")
    section("functions", "Functions", lambda f: f"{f['name']} = {f['expression']}")
    section("reactions", "Reactions", reaction)
    section("events", "Events", event)
    if d.get("load_warnings"):
        lines.append("Load warnings:")
        lines.extend(f"  {w}" for w in d["load_warnings"])
    return "\n".join(lines)


__all__ = ["SCHEMA_VERSION", "ModelDescription", "describe_model", "ode_refusals"]
