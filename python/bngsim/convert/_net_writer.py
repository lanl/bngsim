"""bngsim.convert._net_writer — serialize an in-memory model to BioNetGen ``.net``.

The reverse of :mod:`bngsim._net_reader`: take a loaded :class:`bngsim.Model`
(typically produced by ``Model.from_sbml``) and emit the text ``.net`` network
format that ``Model.from_net`` reads back. Drives off
``model._core.codegen_data()`` (the authoritative in-memory schema — see
``bngsim/src/_bngsim_core.cpp``) plus ``model.get_state()`` (species initial
values, which ``codegen_data`` omits).

Scope is the **network channel** only — species, reactions, parameters,
observables, functions. The plain ``.net`` text format is strictly less
expressive than the in-memory ``NetworkModel`` (no compartment-volume table, no
per-species ``volume_factor``/``amount_valued``, no events). Constructs that
cannot be represented faithfully are surfaced via :func:`capability_report` and
raise :class:`bngsim.ConversionError` (or warn under ``strict=False``); ``<event>``
elements are dropped with a :class:`bngsim.ConversionWarning` because they belong
to the simulation-protocol channel (a SED-ML sidecar), not the network.
"""

from __future__ import annotations

import contextlib
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bngsim._exceptions import ConversionError, ConversionWarning

if TYPE_CHECKING:
    from bngsim._model import Model

# Names we synthesize must not collide with model symbols. The .net builder
# auto-creates a shadow parameter per function, so every synthesized helper is
# both a function and a parameter on reload — the L1 check filters this prefix
# out of its counts (the helpers are encoding overhead, not model structure).
SYNTHETIC_PREFIX = "__s2n_"
_WRAPPER_PREFIX = f"{SYNTHETIC_PREFIX}rxn"


def _fmt(x: float) -> str:
    """Render a float so ``float()`` round-trips it exactly (net_reader parses
    every value field with ``float()``)."""
    return repr(float(x))


# ─── Capability analysis ──────────────────────────────────────────────────


def _rateof_refs(data: dict[str, Any]) -> list[str]:
    """Return sorted ``rate_of_*`` symbols a function references but nothing defines.

    A ``rateOf`` csymbol (SBML ``<csymbol ...rateOf>``) is wired to a reaction's
    derivative at run time; the SBML loader names it ``rate_of__<species>`` and it
    is **not** a species/parameter/observable/function — so neither the flat
    ``.net`` text nor cBNGL can carry it. ExprTk fails to compile the reloaded
    function (``Undefined symbol: 'rate_of__x8'``). Detect it up front so both
    capability reports can refuse fail-loud instead of crashing on reload.
    """
    funcs = data.get("functions", [])
    defined = {f["name"] for f in funcs}
    refs: set[str] = set()
    for f in funcs:
        for tok in re.findall(r"\brate_of_\w+", f.get("expression", "")):
            if tok not in defined:
                refs.add(tok)
    return sorted(refs)


def capability_report(model: Model) -> dict[str, list[str]]:
    """Inspect a model for constructs the plain ``.net`` text format cannot carry.

    Returns ``{"dropped": [...], "lossy": [...]}`` of plain-English notes.
    ``dropped`` is non-fatal (the network is still emitted); ``lossy`` is fatal
    under ``strict=True``. Pure; performs no I/O and emits no warnings.
    """
    core = model._core
    data = core.codegen_data()
    species = data["species"]
    reactions = data["reactions"]

    dropped: list[str] = []
    lossy: list[str] = []

    def _eg(names: list[str]) -> str:
        head = ", ".join(repr(n) for n in names[:3])
        return head + (", …" if len(names) > 3 else "")

    n_events = int(getattr(core, "n_events", 0) or 0)
    if n_events:
        dropped.append(
            f"{n_events} event(s) — events define simulation protocol, not the "
            "network channel; emit them as a SED-ML sidecar instead"
        )

    # Aggregate per category so a 200-species model yields one readable note, not 200.
    amt_vol = [
        s["name"]
        for s in species
        if s.get("amount_valued", False) and float(s.get("volume_factor", 1.0)) != 1.0
    ]
    if amt_vol:
        lossy.append(
            f"{len(amt_vol)} amount-valued species in a compartment whose volume is "
            f"not 1 (e.g. {_eg(amt_vol)}) — plain .net cannot store a per-species "
            "volume factor; the reloaded network would mis-scale these amounts"
        )

    live_vol = [s["name"] for s in species if int(s.get("ode_live_volume_idx0", -1)) >= 0]
    if live_vol:
        lossy.append(
            f"{len(live_vol)} species use a live (time-varying) compartment volume "
            f"(e.g. {_eg(live_vol)}) — not representable without a volume table"
        )

    n_mm = sum(1 for r in reactions if r.get("type") == "mm")
    if n_mm:
        lossy.append(
            f"{n_mm} reaction(s) use a Michaelis–Menten rate-law type, which the .net "
            "text format does not carry"
        )

    n_xcomp = sum(1 for r in reactions if r.get("per_species_volume_scaling", False))
    if n_xcomp:
        lossy.append(
            f"{n_xcomp} cross-compartment reaction(s) with differing volumes need "
            "per-species volume scaling — not representable in plain .net"
        )

    # Species whose *reported* value is corrected at run time for a live or
    # assignment-rule compartment volume (the SBML loader records these in its
    # _varvol_* maps; see _model.py). The correction is a Python-side report
    # transform Simulator.run applies — it is not stored in the model graph, so a
    # reloaded .net silently loses it and mis-scales those species. High-precision
    # signal: every model carrying such a map diverges numerically when round-
    # tripped through plain .net.
    n_varvol = (
        len(getattr(model, "_varvol_conc_map", {}) or {})
        + len(getattr(model, "_varvol_amount_map", {}) or {})
        + len(getattr(model, "_varvol_ar_conc_map", {}) or {})
        + len(getattr(model, "_varvol_event_resize_map", {}) or {})
    )
    if n_varvol:
        lossy.append(
            f"{n_varvol} species report a concentration corrected for a live or "
            "assignment-rule compartment volume — this run-time report transform "
            "is not stored in the .net graph; the reloaded network would mis-scale "
            "these species"
        )

    rateof = _rateof_refs(data)
    if rateof:
        lossy.append(
            f"{len(rateof)} function(s) reference a rateOf csymbol (e.g. "
            f"{_eg(rateof)}) — a reaction-derivative wired at run time, not a "
            "species/parameter the flat .net can carry; the reloaded function "
            "would fail to compile (undefined symbol)"
        )

    return {"dropped": dropped, "lossy": lossy}


def _apply_capability_policy(
    report: dict[str, list[str]], *, strict: bool, model_name: str
) -> None:
    """Warn on dropped constructs; raise (strict) or warn (lenient) on lossy ones."""
    for note in report["dropped"]:
        warnings.warn(f"{model_name}: dropping {note}", ConversionWarning, stacklevel=3)
    if report["lossy"]:
        joined = "; ".join(report["lossy"])
        if strict:
            raise ConversionError(
                f"{model_name}: cannot convert faithfully to .net — {joined}. "
                "Pass strict=False (--allow-lossy) to emit a best-effort network anyway."
            )
        for note in report["lossy"]:
            warnings.warn(
                f"{model_name}: lossy conversion — {note}", ConversionWarning, stacklevel=3
            )


# ─── Rate-token construction ──────────────────────────────────────────────


def _rate_token(
    rxn: dict[str, Any],
    idx: int | str,
    species_names: list[str],
    func_names: set[str],
    wrappers: list[tuple[str, str]],
) -> str:
    """Return the ``.net`` rate token for one reaction, appending any synthesized
    wrapper function to ``wrappers``.

    The text ``.net`` reader always applies the reactant species factor to a
    functional reaction (there is no ``apply_species_factor=False`` knob), and
    ignores per-reaction ``stat_factor``. We reproduce the model's exact
    propensity by emitting a tiny wrapper function when either of those needs
    compensating — otherwise we hand back a bare parameter/function name.
    """
    typ = rxn["type"]
    fname = rxn["function_name"]
    sf = float(rxn["stat_factor"])
    asf = bool(rxn["apply_species_factor"])
    reactants = list(rxn["reactants"])

    if typ == "elementary":
        if sf == 1.0:
            return fname  # truly Elementary: net_reader applies the species factor
        # Bake the statistical factor; net_reader still multiplies reactant amounts.
        wname = f"{_WRAPPER_PREFIX}{idx}"
        wrappers.append((wname, f"({_fmt(sf)} * {fname})"))
        return wname

    # Functional (Functions win name lookups over params/species, so referencing
    # ``fname`` is unambiguous even when a placeholder parameter shares the name).
    expr = fname
    needs_wrapper = False

    if not asf and reactants:
        # Divide the reactant amounts back out so net_reader's re-multiplication
        # cancels, leaving the complete propensity ``fname``.
        guard_names = [species_names[i] for i in reactants]
        collide = [g for g in guard_names if g in func_names]
        if collide:
            loc = idx + 1 if isinstance(idx, int) else idx
            raise ConversionError(
                f"reaction #{loc}: reactant species name(s) {collide} also name "
                "a function, so the .net reactant-factor cancellation would be "
                "ambiguous; rename the species or function in the source model"
            )
        guard = " * ".join(guard_names)
        expr = f"if(({guard}) > 1e-300, ({fname}) / ({guard}), 0)"
        needs_wrapper = True

    if sf != 1.0:
        expr = f"({_fmt(sf)} * ({expr}))"
        needs_wrapper = True

    if not needs_wrapper:
        return fname

    wname = f"{_WRAPPER_PREFIX}{idx}"
    wrappers.append((wname, expr))
    return wname


# ─── Per-species signed-flux expansion ────────────────────────────────────


def _flux_expandable(rxn: dict[str, Any]) -> bool:
    """True iff a reaction is a *functional, no-applied-species-factor* law whose
    stored function value *is* the whole propensity ``P`` (a unit statistical
    factor, and either no applied reactant factor or no reactants at all).

    These cannot be carried by the ordinary ``reactants -> products`` emission:
    the ``.net`` reader re-multiplies a functional rate by the reactant amounts,
    so :func:`_rate_token` divides them back out with ``if(r>1e-300, P/r, 0)`` —
    which silently **zeroes** the propensity whenever a reactant is momentarily
    zero. That is wrong for any ``P`` not proportional to the reactants (a
    constant influx, a saturable/Hill law, a reversible flux): the source RHS is
    nonzero there. Such reactions are instead emitted as per-species signed-flux
    reactions (:func:`_expand_functional_reactions`). Elementary / ``asf=True``
    laws are genuine mass-action (rate ∝ reactant amounts) and keep the ordinary
    path, where the re-multiplication is exactly what's wanted.
    """
    return (
        rxn["type"] == "functional"
        and float(rxn["stat_factor"]) == 1.0
        and ((not bool(rxn["apply_species_factor"])) or not rxn["reactants"])
    )


def _guard_unsafe_reactions(model) -> set[int]:
    """Indices of within-compartment ``asf=False`` functional reactions the
    reactant-division guard (``if(r>1e-300, P/r, 0)``) would wrongly zero **at the
    model's own state** — i.e. a reactant is zero at the initial state while the
    reaction's propensity ``P`` is non-negligible there (a constant influx, a
    saturable/Hill activation, a reversible flux — ``P`` not a multiple of the
    reactant). The guard reports rate 0 for such a reaction, dropping a real flux;
    only these need the topology-changing per-species signed-flux rewrite.

    Most functional reactions are mass-action in disguise (``P = rate·∏reactants``)
    or have nonzero reactants, so ``P`` vanishes exactly when the guard does and
    the topology-preserving ordinary emission is faithful. We therefore probe at
    the **initial state** — the state the faithfulness check itself evaluates —
    rather than over-flagging every reactant-independent law whose reactant never
    actually reaches zero. ``P`` is evaluated at ``t=0`` and ``t=1`` (a constant or
    time-ramped influx onto an initially-empty species still counts). A miss cannot
    ship silently: :func:`sbml_to_net`'s default ``"L2"`` RHS self-check measures
    any residual loss.
    """
    import numpy as np

    core = model._core
    reactions = core.codegen_data()["reactions"]
    cands = {
        n: r
        for n, r in enumerate(reactions)
        if _flux_expandable(r)
        and not r.get("per_species_volume_scaling", False)
        and r["reactants"]
    }
    if not cands:
        return set()

    y0 = np.asarray(model.get_state(), dtype=float)
    fn_vals: dict[str, float] = {}
    for t in (0.0, 1.0):
        try:
            for name, v in core._eval_functions(t, y0).items():
                with contextlib.suppress(TypeError, ValueError):
                    fn_vals[name] = max(fn_vals.get(name, 0.0), abs(float(v)))
        except Exception:  # out-of-domain at the initial state → leave undecided
            pass

    return {
        n
        for n, r in cands.items()
        if any(y0[i] <= 1e-12 for i in r["reactants"])
        and fn_vals.get(r["function_name"], 0.0) > 1e-12
    }


def _expand_functional_reactions(
    reactions: list[dict[str, Any]],
    species: list[dict[str, Any]],
    *,
    expand_psvs: bool = True,
    only_indices: set[int] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, str]]]:
    """Rewrite ``asf=False`` functional reactions as per-species signed-flux
    reactions, so the flat ``.net`` round-trip reproduces the source ODE exactly.

    For such a reaction the source contributes ``d[cᵢ]/dt += sᵢ·P/Vᵢ`` to each
    affected species — ``P`` the stored function value (the *whole* propensity),
    ``sᵢ`` species *i*'s net stoichiometry, ``Vᵢ`` its compartment volume when the
    reaction is per-species-volume-scaled (transport), else 1 (a within-
    compartment law already folds volume into ``P``). We emit one **pure-flux**
    reaction per affected species, ``0 -> Molᵢ()`` with rate function ``(sᵢ/Vᵢ)·P``.

    The zero-reactant form is the crux: it carries the whole flux faithfully for
    any ``P`` and any reactant value (the ``.net`` reader multiplies a zero-
    reactant functional rate by an empty product = 1), where the old reactant-
    guard zeroed reactant-independent / saturable / reversible laws at a zero
    reactant.

    ``expand_psvs`` controls per-species-volume-scaled (cross-compartment
    transport) reactions: cBNGL expands them (``True``); the flat ``.net`` writer
    refuses them at the capability gate, so it passes ``False`` to leave them on
    the ordinary path. ``only_indices``, when given, restricts the rewrite to that
    set of reaction indices (the flat ``.net`` writer passes the
    :func:`_guard_unsafe_reactions` set so it rewrites *only* the reactions the
    reactant guard would break, preserving topology elsewhere); ``None`` rewrites
    every expandable reaction (cBNGL, whose volume split needs all of them).
    Returns ``(labeled_reactions, helper_functions)``; a reaction kept on the
    ordinary path passes through as ``(str(index), rxn)``.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    helpers: list[tuple[str, str]] = []

    def _vol(i: int) -> float:
        return float(species[i].get("volume_factor", 1.0))

    for n, r in enumerate(reactions):
        psvs = bool(r.get("per_species_volume_scaling", False))
        if (
            not _flux_expandable(r)
            or (psvs and not expand_psvs)
            or (only_indices is not None and not psvs and n not in only_indices)
        ):
            out.append((str(n), r))
            continue

        base = r["function_name"]  # asf=False functional ⇒ this value IS P
        net: dict[int, int] = {}
        for i in r["reactants"]:
            net[i] = net.get(i, 0) - 1
        for i in r["products"]:
            net[i] = net.get(i, 0) + 1

        for k, (i, s_i) in enumerate(net.items()):
            if s_i == 0:  # a catalyst nets to zero — no ODE contribution
                continue
            factor = (s_i / _vol(i)) if psvs else float(s_i)
            fname = f"__flux{n}_{k}"
            helpers.append((fname, f"{_fmt(factor)} * ({base})"))
            out.append(
                (
                    f"{n}_{k}",
                    {
                        "type": "functional",
                        "function_name": fname,
                        "stat_factor": 1.0,
                        "apply_species_factor": False,
                        "per_species_volume_scaling": False,
                        "reactants": [],
                        "products": [i],
                    },
                )
            )

    return out, helpers


# ─── Writer ───────────────────────────────────────────────────────────────


def write_net(
    model: Model,
    out_path: str | Path | None = None,
    *,
    strict: bool = True,
    header: str | None = None,
    model_name: str | None = None,
) -> str:
    """Serialize ``model`` to BioNetGen ``.net`` text.

    Parameters
    ----------
    model : bngsim.Model
        A loaded model (e.g. from ``Model.from_sbml``).
    out_path : str | Path | None
        If given, the ``.net`` text is also written here.
    strict : bool
        When True (default), raise :class:`bngsim.ConversionError` on constructs
        that cannot be represented faithfully. When False, downgrade to a
        :class:`bngsim.ConversionWarning` and emit a best-effort network.
    header : str | None
        Optional comment text placed at the top of the file (each line is
        prefixed with ``# ``).

    Returns
    -------
    str
        The ``.net`` text.
    """
    core = model._core
    data = core.codegen_data()
    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    functions = data["functions"]
    reactions = data["reactions"]

    name = model_name or getattr(model, "name", None) or "model"
    _apply_capability_policy(capability_report(model), strict=strict, model_name=str(name))

    species_names = [s["name"] for s in species]
    func_names = {f["name"] for f in functions}
    init_state = list(model.get_state())

    lines: list[str] = []
    if header:
        for hl in str(header).splitlines():
            lines.append(f"# {hl}")
    else:
        lines.append("# Generated by bngsim.convert.sbml_to_net")
        lines.append(
            "# Network channel only (species/reactions/parameters/observables/functions)."
        )
    lines.append("")

    # parameters — emit evaluated literal values (derived params are constant
    # expressions; their numeric value is exact and avoids re-evaluation hazards).
    lines.append("begin parameters")
    for i, p in enumerate(params, 1):
        lines.append(f"    {i} {p['name']} {_fmt(p['value'])}")
    lines.append("end parameters")
    lines.append("")

    # species — value is the stored initial state (concentration space)
    lines.append("begin species")
    for i, s in enumerate(species, 1):
        nm = s["name"]
        marker = "$" if s.get("fixed", False) else ""
        val = init_state[i - 1] if i - 1 < len(init_state) else 0.0
        lines.append(f"    {i} {marker}{nm} {_fmt(val)}")
    lines.append("end species")
    lines.append("")

    # groups (observables)
    lines.append("begin groups")
    for j, o in enumerate(observables, 1):
        toks = []
        for idx0, factor in o["entries"]:
            idx1 = int(idx0) + 1
            toks.append(f"{idx1}" if float(factor) == 1.0 else f"{_fmt(factor)}*{idx1}")
        entry_str = (",".join(toks)) if toks else ""
        lines.append(f"    {j} {o['name']} {entry_str}".rstrip())
    lines.append("end groups")
    lines.append("")

    # Rewrite *only* the within-compartment asf=False functional reactions whose
    # propensity the reactant-division guard would wrongly zero (a reactant-
    # independent / saturable / reversible law) as per-species signed-flux
    # reactions, so they survive the round-trip; the topology of every other
    # reaction is preserved. Cross-compartment (per-species-volume-scaled)
    # reactions are refused at the capability gate (expand_psvs=False).
    labeled_reactions, flux_helpers = _expand_functional_reactions(
        reactions,
        species,
        expand_psvs=False,
        only_indices=_guard_unsafe_reactions(model),
    )
    func_names_all = func_names | {h for h, _ in flux_helpers}

    # reactions — build first so any synthesized wrapper functions are known
    wrappers: list[tuple[str, str]] = []
    rxn_lines: list[str] = []
    for n, (label, r) in enumerate(labeled_reactions):
        token = _rate_token(r, label, species_names, func_names_all, wrappers)
        rs = ",".join(str(i + 1) for i in r["reactants"]) or "0"
        ps = ",".join(str(i + 1) for i in r["products"]) or "0"
        rxn_lines.append(f"    {n + 1} {rs} {ps} {token}")

    # functions — originals (verbatim), then per-species flux helpers, then any
    # synthesized reaction wrappers
    if functions or flux_helpers or wrappers:
        lines.append("begin functions")
        k = 0
        for f in functions:
            k += 1
            lines.append(f"    {k} {f['name']}() {f['expression']}")
        for hname, hexpr in flux_helpers:
            k += 1
            lines.append(f"    {k} {hname}() {hexpr}")
        for wname, wexpr in wrappers:
            k += 1
            lines.append(f"    {k} {wname}() {wexpr}")
        lines.append("end functions")
        lines.append("")

    lines.append("begin reactions")
    lines.extend(rxn_lines)
    lines.append("end reactions")
    lines.append("")

    text = "\n".join(lines)
    if out_path is not None:
        Path(out_path).write_text(text)
    return text
