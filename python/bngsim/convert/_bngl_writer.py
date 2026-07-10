"""bngsim.convert._bngl_writer — serialize an in-memory model to cBNGL (GH #224).

Emit a **compartmental BNGL** (``.bngl``) model block from a loaded
:class:`bngsim.Model` (typically one produced by ``Model.from_sbml``). Unlike the
flat, unit-volume ``.net`` text format (:mod:`bngsim.convert._net_writer`), cBNGL
carries **static compartment volumes** natively, so a model whose species sit in
non-unit compartments — the dominant SBML→net refusal class — round-trips
faithfully through ``BNG2.pl generate_network`` → ``bngsim.Model.from_net``.

This is **deliverable 1** of the #224 epic: recover the *static* compartment set.
Events→actions (the protocol channel, #222) and the cross-compartment /
amount-valued / time-varying-volume cases are deferred; they are refused
fail-loud here (see :func:`bngl_capability_report`).

The round-trip math (why the rates look pre-scaled)
---------------------------------------------------
The validation pipeline is ``.bngl → BNG2.pl → .net → from_net``. Two transforms
sit between the cBNGL rate we write and the propensity ``from_net`` finally
computes, and the writer composes both so the *source* propensity is reproduced:

* **BNG bake.** For a rule with ``n`` reactant molecules in a compartment of
  volume ``V``, ``BNG2.pl`` multiplies the rate by ``unit_conversion = V**(1-n)``
  (synthesis ``n=0`` → ``×V``; unimolecular ``n=1`` → ``×1``; bimolecular ``n=2``
  → ``×1/V`` …) and by a statistical factor ``1/∏(mⱼ!)`` for identical-reactant
  multiplicities ``mⱼ``.
* **``.net`` reader quirk.** ``from_net`` always re-multiplies a *functional*
  reaction's rate by the reactant-species product (it treats any ``coeff*name``
  rate as a synthesized functional). :func:`bngsim.convert._net_writer._rate_token`
  already targets this quirk exactly (baking ``stat_factor`` and dividing the
  reactants back out for ``apply_species_factor=False`` rate laws), so we *reuse*
  it to get the ``.net``-correct rate token ``R``.

We then emit the BNGL reaction rate as ``R × V**(n-1) × ∏(mⱼ!)``. BNG's bake
cancels that compensation exactly, leaving ``R`` in the generated ``.net`` — which
``from_net`` reads back to the source propensity. Compartment topology is flat
(one ``name 3 V`` per distinct volume, no synthesized surfaces); rule patterns are
fully compartment-qualified so BNG maps them to the declared species rather than
spawning phantoms.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bngsim._exceptions import ConversionError, ConversionWarning
from bngsim.convert._net_writer import (
    _expand_functional_reactions,
    _flux_expandable,
    _rate_token,
    _rateof_refs,
)
from bngsim.convert._sbml_writer import (
    _LOG1P_TOKEN,
    _LOG2_TOKEN,
    _LOG_TOKEN,
    _dedup_functions,
    _rewrite_unary_call,
)

if TYPE_CHECKING:
    from bngsim._model import Model


# ─── BNGL identifier sanitization ──────────────────────────────────────────

_ID_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ID_BAD = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(name: str, used: set[str], *, fallback: str) -> str:
    """Return a valid, unique BNGL identifier derived from ``name``.

    BNGL molecule/parameter/observable/function names are
    ``[A-Za-z_][A-Za-z0-9_]*``. Pattern characters (``() ~ , . !`` …) become
    ``_``; a name starting with a digit (or emptying out) is prefixed with
    ``fallback``; collisions get ``_2``, ``_3``, … appended.
    """
    base = _ID_BAD.sub("_", name).strip("_")
    if not base or (not base[0].isalpha() and base[0] != "_"):
        base = f"{fallback}{base}" if base else fallback
    if not _ID_OK.match(base):
        base = fallback
    out = base
    n = 2
    while out in used:
        out = f"{base}_{n}"
        n += 1
    used.add(out)
    return out


def _normalize_bngl_expr(expr: str) -> str:
    """ExprTk function body → BNGL-parseable infix (pure string transforms).

    BNG's function parser is ExprTk-like but stricter on a few spellings the engine
    stores: the boolean keyword operators ``and``/``or`` must be the symbolic
    ``&&``/``||`` (BNG rejects the barewords with a misleading "missing parenthesis"),
    natural log ``log(`` must be ``ln(`` (BNG has no bareword ``log``), and base-2 /
    ``log1p`` (no BNG primitive) reduce to their exact ``ln`` identities. The π
    symbol ``_pi`` is reserved-but-undefined in BNG2.pl (it aborts at network
    generation whether the name is left bare *or* declared as a parameter), so it is
    folded to its numeric literal. ``if(…)``, ``time()``, ``^`` and ``log10`` are
    accepted verbatim. Word-boundary anchored so ``ligand``/``factor``/``nand`` are
    untouched.
    """
    expr = re.sub(r"(?<![A-Za-z0-9_])_pi(?![A-Za-z0-9_])", repr(math.pi), expr)
    out = re.sub(r"(?<![A-Za-z0-9_])and(?![A-Za-z0-9_])", "&&", expr)
    out = re.sub(r"(?<![A-Za-z0-9_])or(?![A-Za-z0-9_])", "||", out)
    out = _rewrite_unary_call(out, _LOG2_TOKEN, lambda a: f"(ln({a}) / ln(2))")
    out = _rewrite_unary_call(out, _LOG1P_TOKEN, lambda a: f"ln(1 + ({a}))")
    out = re.sub(r"(?<![A-Za-z0-9_])log10\s*\(", "__LOG10__(", out)
    out = _LOG_TOKEN.sub("ln(", out)
    out = out.replace("__LOG10__(", "log10(")
    return out


def _molecule_names(
    species: list[dict[str, Any]],
    params: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    shadow: set[str],
) -> tuple[set[str], list[str]]:
    """Allocate one structureless-molecule name per species.

    Molecule names are sanitized from the species name so ``from_net``'s reloaded
    ``@C::Mol()`` strips back to the source species name for name-aligned
    trajectory comparison. The live parameter/function namespace is reserved first
    so a molecule never shadows a scalar a rate references; *observable* names are
    deliberately **not** reserved — BNG allows a molecule and an observable to
    share a name, which is exactly the per-species observable case (and keeps the
    molecule name equal to the species name). Returns ``(used_ids, mol_names)``;
    the test harness imports this to reproduce the mapping exactly.
    """
    used: set[str] = {p["name"] for p in params if p["name"] not in shadow}
    used |= {f["name"] for f in functions}
    mol_names = [_sanitize(s["name"], used, fallback=f"S{i + 1}") for i, s in enumerate(species)]
    return used, mol_names


# ─── Capability analysis ───────────────────────────────────────────────────


def _reactant_volumes(rxn: dict[str, Any], species: list[dict[str, Any]]) -> set[float]:
    """Distinct compartment volumes of a reaction's reactant species."""
    return {float(species[i].get("volume_factor", 1.0)) for i in rxn["reactants"]}


def _dynamic_volumes(rxn: dict[str, Any], species: list[dict[str, Any]]) -> set[float]:
    """Distinct volumes of a reaction's *dynamic* (non-fixed) species (reactants ∪
    products).

    A reaction is single-compartment iff this set has ≤1 element. A fixed (``$``)
    species carries no ODE derivative, so it does not constrain the volume scaling
    (it enters the law as a constant). Used to refuse cross-compartment reactions
    — including **transport** (reactant and product in different volumes), where
    the source ODE's per-species ``1/Vᵢ`` factor makes the two sides asymmetric in
    a way the flat, unit-volume ``.net`` round-trip cannot reproduce (deferred to
    #224 deliverable 1b).
    """
    return {
        float(species[i].get("volume_factor", 1.0))
        for i in list(rxn["reactants"]) + list(rxn["products"])
        if not bool(species[i].get("fixed", False))
    }


_EVENT_GENERIC = object()  # sentinel: caller did not classify the events


def bngl_capability_report(
    model: Model, *, event_override: Any = _EVENT_GENERIC
) -> dict[str, list[str]]:
    """Inspect a model for constructs the cBNGL writer (deliverable 1) cannot carry.

    Returns ``{"dropped": [...], "lossy": [...]}`` of plain-English notes.
    ``dropped`` is non-fatal; ``lossy`` is fatal under ``strict=True``. The
    recovered class is **static compartment volumes** (incl. the unit-volume
    baseline). Refused fail-loud (each a ``lossy`` note):

    * **events** — by default a generic "events define the protocol, not the
      network" refusal (the no-SBML, direct-``write_bngl`` path). The
      SBML-aware orchestrator classifies them first (see
      :func:`bngsim.convert._events.sbml_events_to_protocol`) and passes
      ``event_override``: an empty list when every event is a fixed-time event it
      translated into a BNGL actions block (#224 phase 2), or specific notes for
      the state-triggered / non-constant events that have no actions form;
    * **live / time-varying compartment volumes** (the ``_varvol_*`` maps and any
      ``ode_live_volume`` species) — cBNGL compartments are static;
    * **cross-compartment reactions** (dynamic species span more than one
      compartment volume — including *transport*, reactant and product in
      different volumes) — the source ODE's per-species ``1/Vᵢ`` scaling makes the
      two sides asymmetric, which the flat unit-volume ``.net`` round-trip cannot
      reproduce (deferred to deliverable 1b);
    * **Michaelis–Menten reactions** — the ``mm`` closed form cannot cross the
      ``.net`` channel the round-trip validates through;
    * **amount-valued species in a non-unit compartment** — the writer keeps the
      concentration-space semantics ``from_net`` reads back, so a per-species
      amount factor would mis-scale.

    Pure; no I/O, no warnings.
    """
    core = model._core
    data = core.codegen_data()
    species = data["species"]
    reactions = data["reactions"]

    dropped: list[str] = []
    lossy: list[str] = []

    def _eg(names: list[Any]) -> str:
        head = ", ".join(repr(n) for n in names[:3])
        return head + (", …" if len(names) > 3 else "")

    n_events = int(getattr(core, "n_events", 0) or 0)
    if n_events:
        if event_override is _EVENT_GENERIC:
            lossy.append(
                f"{n_events} event(s) — fixed-time events translate to a BNGL actions "
                "block (#224 phase 2); state-triggered events have no actions form. "
                "Convert via sbml_to_bngl (which reads the source SBML to classify "
                "them) or emit a SED-ML sidecar"
            )
        else:
            # The orchestrator classified the events: empty → all fixed-time and
            # carried into the actions block; otherwise the specific refusals.
            lossy.extend(event_override)

    # Assignment-rule report species (GH #205, ``_ar_report_map``): a species
    # marked fixed whose *reported* value is an assignment expression Simulator.run
    # applies as a Python-side report transform — it is not in the model graph, so
    # the flat .net round-trip drops it and the species clamps at its initial value
    # (the assignment-rule forcing #223 owns the general fix).
    ar = sorted((getattr(model, "_ar_report_map", None) or {}).keys())
    if ar:
        lossy.append(
            f"{len(ar)} assignment-rule species (e.g. {_eg(ar)}) report an algebraic "
            "expression, not a clamped constant — the flat .net round-trip cannot "
            "carry the rule, so the reloaded species would freeze at its initial "
            "value (GH #205/#223; deferred)"
        )

    # Live / time-varying compartment volumes — cBNGL volumes are static.
    for attr, why in (
        ("_varvol_conc_map", "report a concentration corrected for a time-varying volume"),
        ("_varvol_amount_map", "report an amount corrected for a time-varying volume"),
        ("_varvol_ar_conc_map", "sit in an assignment-rule compartment"),
        ("_varvol_event_resize_map", "sit in an event-resized compartment"),
    ):
        names = sorted((getattr(model, attr, None) or {}).keys())
        if names:
            lossy.append(
                f"{len(names)} species {why} (e.g. {_eg(names)}) — cBNGL carries "
                "static compartment volumes only, not a volume that moves in time"
            )
    live_vol = [s["name"] for s in species if int(s.get("ode_live_volume_idx0", -1)) >= 0]
    if live_vol and not any("time-varying" in n or "static compartment" in n for n in lossy):
        lossy.append(
            f"{len(live_vol)} species use a live (time-varying) compartment volume "
            f"(e.g. {_eg(live_vol)}) — cBNGL emits static compartments only"
        )

    # Cross-compartment reactions. Any reaction whose dynamic species span more
    # than one compartment volume — product-cross-compartment transport AND
    # reactant-cross-compartment (reactants themselves in different volumes) — is
    # RECOVERED via the per-species signed-flux split (deliverable 1b; see
    # _expand_functional_reactions): each affected species gets a 0 -> Molᵢ() flux
    # at rate (sᵢ/Vᵢ)·P, dividing by that species' *own* volume, so an arbitrary
    # spread of reactant/product compartments round-trips faithfully. The split
    # requires a plain functional propensity whose value *is* the amount propensity
    # P (asf=False, unit statistical factor — _handled_transport). Still refused:
    # the rare exotic transport rate kinds (elementary, an applied reactant factor
    # with reactants, or a non-unit statistical factor) the split cannot carry.
    exotic = [
        n + 1
        for n, r in enumerate(reactions)
        if len(_dynamic_volumes(r, species)) > 1 and not _handled_transport(r)
    ]
    if exotic:
        lossy.append(
            f"{len(exotic)} cross-compartment reaction(s) (e.g. #{exotic[0]}) use a "
            "transport rate the per-species flux split cannot carry faithfully "
            "(non-functional, an applied reactant factor with reactants, or a "
            "non-unit statistical factor)"
        )

    # Functions using a call BNG2.pl's parser does not provide (e.g. ceil/floor):
    # BNG silently reads `ceil(x)` as the product `ceil*(x)` and then aborts on the
    # undefined `ceil`, so the emitted .bngl would not build. Refuse fail-loud.
    _UNSUPPORTED_FUNCS = ("ceil", "floor")
    bad_funcs = sorted(
        {
            fn
            for f in data["functions"]
            for fn in _UNSUPPORTED_FUNCS
            if re.search(rf"(?<![A-Za-z0-9_]){fn}\s*\(", f["expression"])
        }
    )
    if bad_funcs:
        lossy.append(
            f"{len(bad_funcs)} function call(s) BNG2.pl does not provide "
            f"({', '.join(bad_funcs)}) — the generated .bngl would not build "
            "(BNG has no ceil/floor primitive)"
        )

    rateof = _rateof_refs(data)
    if rateof:
        lossy.append(
            f"{len(rateof)} function(s) reference a rateOf csymbol (e.g. "
            f"{_eg(rateof)}) — a reaction-derivative wired at run time, not a "
            "species/parameter cBNGL can carry; the generated .bngl would not build"
        )

    # The not-equal operator `!=`: BNG2.pl's function parser does not provide it
    # (it reads the bare `!` as logical-not and aborts on the trailing `=…`), so a
    # function carrying `!=` would not build. The .net/ExprTk channel accepts it, so
    # this is a cBNGL-only refusal. (`==`, `>=`, `<=`, `>`, `<` and `&&`/`||` —
    # the latter via _normalize_bngl_expr — are all fine.)
    ne_funcs = sorted({f["name"] for f in data["functions"] if "!=" in f["expression"]})
    if ne_funcs:
        lossy.append(
            f"{len(ne_funcs)} function(s) (e.g. {_eg(ne_funcs)}) use the not-equal "
            "operator '!=', which BNG2.pl's parser does not provide — the generated "
            ".bngl would not build"
        )

    # Non-finite parameter values (±inf / nan — e.g. FBA flux bounds `_lp_*`): BNG2.pl
    # has no `inf` literal and reads the token as an undefined parameter, aborting at
    # network generation. The .net/ExprTk channel carries them, so cBNGL-only.
    nonfinite = sorted(
        p["name"]
        for p in data["parameters"]
        if not p.get("expression") and not math.isfinite(float(p["value"]))
    )
    if nonfinite:
        lossy.append(
            f"{len(nonfinite)} parameter(s) (e.g. {_eg(nonfinite)}) have a non-finite "
            "value (±inf / nan) — BNG2.pl has no such literal and aborts on it"
        )

    n_mm = sum(1 for r in reactions if r.get("type") == "mm")
    if n_mm:
        lossy.append(
            f"{n_mm} reaction(s) use a Michaelis–Menten rate-law type, which cannot "
            "cross the .net channel the BNG2.pl round-trip validates through"
        )

    amt_vol = [
        s["name"]
        for s in species
        if s.get("amount_valued", False) and float(s.get("volume_factor", 1.0)) != 1.0
    ]
    if amt_vol:
        lossy.append(
            f"{len(amt_vol)} amount-valued species in a non-unit compartment "
            f"(e.g. {_eg(amt_vol)}) — the writer keeps the concentration-space "
            "semantics from_net reads back; a per-species amount factor would mis-scale"
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
                f"{model_name}: cannot convert faithfully to cBNGL — {joined}. "
                "Pass strict=False (--allow-lossy) to emit a best-effort model anyway."
            )
        for note in report["lossy"]:
            warnings.warn(
                f"{model_name}: lossy conversion — {note}", ConversionWarning, stacklevel=3
            )


# ─── Rate compensation ─────────────────────────────────────────────────────


def _fmt(x: float) -> str:
    """Render a float so ``float()`` round-trips it (BNG re-parses every value)."""
    return repr(float(x))


def _factorial(n: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


def _bng_stat_factor(reactants: list[int], products: list[int]) -> int:
    """BNG's reaction-symmetry divisor — ``∏ⱼ pⱼ!·(mⱼ−pⱼ)!`` over each reactant
    species *j*, where ``mⱼ`` is its reactant multiplicity and ``pⱼ = min(mⱼ,
    productⱼ)`` the count BNG carries through unchanged. We multiply the emitted
    rate by it so BNG2.pl's division cancels.

    The divisor is **not** ``∏(mⱼ!)``: BNG only treats two identical reactant
    molecules as interchangeable when the reaction acts on them the same way.
    For a catalytic transform like ``X + X -> X + Y`` BNG preserves one X (mapped
    to the product X) and consumes the other, so the two X's are *distinguishable*
    and the factor is ``1!·1! = 1``, not ``2!``. A symmetric ``X + X -> Y`` (both
    consumed) keeps ``0!·2! = 2``; ``X + X -> X + X`` (both preserved) keeps
    ``2!·0! = 2``. Using ``∏mⱼ!`` here doubled the rate of every species-preserving
    identical-reactant reaction (the GH #224 BIOMD0000000233 mismatch). Verified
    against BNG2.pl 2.9.3 over 1-, 2- and 3-body reactant groups.
    """
    rcount: dict[int, int] = {}
    for i in reactants:
        rcount[i] = rcount.get(i, 0) + 1
    pcount: dict[int, int] = {}
    for i in products:
        pcount[i] = pcount.get(i, 0) + 1
    out = 1
    for i, m in rcount.items():
        p = min(m, pcount.get(i, 0))
        out *= _factorial(p) * _factorial(m - p)
    return out


def _reaction_volume(rxn: dict[str, Any], species: list[dict[str, Any]]) -> float:
    """The volume BNG uses for a reaction's ``unit_conversion`` (its compartment).

    For ``n ≥ 1`` reactants (all in one compartment — cross-compartment is refused
    up front) that is the reactant volume. For a synthesis (``n = 0``) the reaction
    occurs in the product compartment, so we use the product volume (a non-unit
    synthesis target still needs the ``×V`` compensation). Falls back to ``1.0``.
    """
    if rxn["reactants"]:
        return next(iter(_reactant_volumes(rxn, species)))
    for i in rxn["products"]:
        return float(species[i].get("volume_factor", 1.0))
    return 1.0


# ─── Cross-compartment transport (deliverable 1b) ──────────────────────────


def _handled_transport(rxn: dict[str, Any]) -> bool:
    """True iff a transport (``per_species_volume_scaling``) reaction is one the
    per-species flux split can carry — i.e. it is :func:`_flux_expandable` (a plain
    functional propensity whose value *is* the amount propensity ``P``, unit
    statistical factor, no applied reactant factor unless it has no reactants). The
    dominant transport shape (functional, asf=False, stat=1) qualifies; exotic
    combinations (elementary, an applied factor with reactants, non-unit stat) are
    refused by the capability report.
    """
    return bool(rxn.get("per_species_volume_scaling", False)) and _flux_expandable(rxn)


# ─── Writer ────────────────────────────────────────────────────────────────


def write_bngl(
    model: Model,
    out_path: str | Path | None = None,
    *,
    strict: bool = True,
    model_name: str | None = None,
    protocol: Any = None,
    event_override: Any = _EVENT_GENERIC,
) -> str:
    """Serialize ``model`` to a compartmental BNGL (``.bngl``) model block.

    Parameters
    ----------
    model : bngsim.Model
        A loaded model (e.g. from ``Model.from_sbml``).
    out_path : str | Path | None
        If given, the BNGL text is also written here.
    strict : bool
        When True (default), raise :class:`bngsim.ConversionError` on constructs
        deliverable 1 cannot carry (see :func:`bngl_capability_report`). When
        False, downgrade to a :class:`bngsim.ConversionWarning` and emit anyway.
    model_name : str | None
        Optional label placed in the header comment.
    protocol : ProtocolSpec | None
        When given (the SBML-event→actions translation, #224 phase 2), a
        ``begin actions`` block is appended after ``end model``:
        ``generate_network`` then the protocol's ``simulate`` phases and
        ``setConcentration`` state changes. Its ``StateChange.target`` is a
        *codegen species name*, which this writer translates to the
        compartment-qualified pattern (``@comp:Mol()``) the model declares.
    event_override : list[str] | sentinel
        Forwarded to :func:`bngl_capability_report` — the orchestrator's event
        classification (empty list = all fixed-time, carried into ``protocol``).

    Returns
    -------
    str
        The BNGL text — a ``begin model`` … ``end model`` block, optionally
        followed by a ``begin actions`` block when ``protocol`` is given. Without
        actions, append a ``generate_network`` action to run it through ``BNG2.pl``.
    """
    core = model._core
    data = core.codegen_data()
    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    # A cross-compartment volume-scaled .net can carry the same helper function
    # (e.g. ``_vd_reaction_1_compartment_1``) many times; BNG keys functions by
    # name, so emitting each repeat would "match a previously defined variable".
    functions = _dedup_functions(data["functions"])
    reactions = data["reactions"]

    name = model_name or getattr(model, "name", None) or "model"
    _apply_capability_policy(
        bngl_capability_report(model, event_override=event_override),
        strict=strict,
        model_name=str(name),
    )

    init_state = list(model.get_state())
    func_names = {f["name"] for f in functions}
    obs_names = {o["name"] for o in observables}
    # The .net builder adds a constant *shadow parameter* per function/observable;
    # drop those so the BNGL symbol namespace stays single-valued (the function /
    # observable definition is the live one).
    shadow = func_names | obs_names

    # ── molecule names (one structureless molecule per species) ────────────
    used_ids, mol_names = _molecule_names(species, params, functions, shadow)
    # Per-species observable used inside rate-law guards (apply_species_factor=False
    # divides the reactant amounts back out). Distinct prefix → never collides.
    sp_obs = [f"__sp{i + 1}" for i in range(len(species))]

    # ── compartments (one per distinct static volume; flat, no surfaces) ───
    vol_to_comp: dict[float, str] = {}
    comp_lines: list[str] = []

    def _comp_for(vol: float) -> str:
        key = float(vol)
        if key not in vol_to_comp:
            cid = _sanitize(f"comp{len(vol_to_comp) + 1}", used_ids, fallback="comp")
            vol_to_comp[key] = cid
            comp_lines.append(f"    {cid} 3 {_fmt(key)}")
        return vol_to_comp[key]

    species_comp = [_comp_for(float(s.get("volume_factor", 1.0))) for s in species]

    # ── expand asf=False functional reactions into per-species flux (1b) ───
    # Each emitted flux reaction is an ordinary reaction the machinery below
    # handles; its helper function ((sᵢ/Vᵢ)·P) joins the functions block.
    # func_names is extended so _rate_token treats a helper (used as a rate) as a
    # function. This also fixes transport (per-species 1/V asymmetry, deliverable
    # 1b) and any reactant-independent / reversible functional law.
    labeled_reactions, flux_helpers = _expand_functional_reactions(reactions, species)
    func_names = func_names | {h for h, _ in flux_helpers}

    # ── reaction rates (reuse _net_writer token, then add BNG compensation) ─
    # _rate_token references species by the names we hand it; pass the per-species
    # observables so its apply_species_factor=False guards resolve in BNGL.
    wrappers: list[tuple[str, str]] = []
    rxn_rate_funcs: list[tuple[str, str]] = []  # (funcname, expr) compensation wrappers
    rxn_rate_tokens: list[str] = []  # rate spelling used in each rule
    for label, r in labeled_reactions:
        token = _rate_token(r, label, sp_obs, func_names, wrappers)
        n_react = len(r["reactants"])
        vol = _reaction_volume(r, species)
        comp = (vol ** (n_react - 1)) * _bng_stat_factor(r["reactants"], r["products"])
        if comp == 1.0:
            rxn_rate_tokens.append(token)
        else:
            fn = f"__rxnrate{label}"
            rxn_rate_funcs.append((fn, f"({token}) * {_fmt(comp)}"))
            rxn_rate_tokens.append(fn)

    # ── assemble text ──────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# Generated by bngsim.convert.sbml_to_bngl from {name}")
    lines.append("# Compartmental BNGL (cBNGL): static compartment volumes recovered (GH #224).")
    lines.append("begin model")
    lines.append("")

    lines.append("begin parameters")
    k = 0
    for p in params:
        if p["name"] in shadow:
            continue
        k += 1
        lines.append(f"    {k} {p['name']} {_fmt(p['value'])}")
    lines.append("end parameters")
    lines.append("")

    lines.append("begin compartments")
    lines.extend(comp_lines)
    lines.append("end compartments")
    lines.append("")

    lines.append("begin molecule types")
    for mn in mol_names:
        lines.append(f"    {mn}()")
    lines.append("end molecule types")
    lines.append("")

    # observables: per-species probes (for rate guards + name-aligned reporting)
    # then the source observables (single namespace; referenced by function bodies).
    # A ``Molecules`` observable can only *sum* unit-weight patterns, so a weighted
    # or empty observable is emitted as a function instead (over the per-species
    # probes) — exact, and still referenced by bareword from function bodies.
    obs_as_funcs: list[tuple[str, str]] = []
    obs_lines: list[str] = []
    for o in observables:
        entries = o["entries"]
        if entries and all(float(f) == 1.0 for _, f in entries):
            pats = " ".join(mol_names[int(idx0)] + "()" for idx0, _ in entries)
            obs_lines.append(f"    Molecules {o['name']} {pats}")
        else:
            terms = [f"{_fmt(float(f))}*{sp_obs[int(idx0)]}" for idx0, f in entries]
            obs_as_funcs.append((o["name"], " + ".join(terms) if terms else "0"))

    lines.append("begin observables")
    for i, mn in enumerate(mol_names):
        lines.append(f"    Molecules {sp_obs[i]} {mn}()")
    lines.extend(obs_lines)
    lines.append("end observables")
    lines.append("")

    # functions: weighted/empty observables, then source functions (incl. _vd_
    # volume wrappers) verbatim, then the per-species flux helpers (referencing the
    # source functions), then the per-reaction rate-compensation wrappers and
    # _rate_token's reactant wrappers.
    if functions or wrappers or rxn_rate_funcs or obs_as_funcs or flux_helpers:
        lines.append("begin functions")
        for oname, oexpr in obs_as_funcs:
            lines.append(f"    {oname}() {oexpr}")
        for f in functions:
            lines.append(f"    {f['name']}() {_normalize_bngl_expr(f['expression'])}")
        for hname, hexpr in flux_helpers:
            lines.append(f"    {hname}() {hexpr}")
        for wname, wexpr in wrappers:
            lines.append(f"    {wname}() {wexpr}")
        for fn, expr in rxn_rate_funcs:
            lines.append(f"    {fn}() {expr}")
        lines.append("end functions")
        lines.append("")

    lines.append("begin species")
    for i, s in enumerate(species):
        marker = "$" if s.get("fixed", False) else ""
        val = init_state[i] if i < len(init_state) else 0.0
        lines.append(f"    @{species_comp[i]}:{marker}{mol_names[i]}() {_fmt(val)}")
    lines.append("end species")
    lines.append("")

    # reaction rules: fully compartment-qualified ground patterns (one reaction
    # each; transport reactions appear as their volume-scaled consumption/synthesis
    # halves).
    lines.append("begin reaction rules")
    for i, (_label, r) in enumerate(labeled_reactions):
        rs = _pattern_side(r["reactants"], mol_names, species_comp)
        ps = _pattern_side(r["products"], mol_names, species_comp)
        lines.append(f"    {rs} -> {ps} {rxn_rate_tokens[i]}")
    lines.append("end reaction rules")
    lines.append("")
    lines.append("end model")

    # ── actions (SBML events → BNGL protocol, #224 phase 2) ────────────────
    if protocol is not None and not protocol.is_empty:
        lines.append("")
        lines.append(_actions_block(protocol, mol_names, species_comp, species))

    text = "\n".join(lines) + "\n"
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


def _actions_block(
    protocol: Any,
    mol_names: list[str],
    species_comp: list[str],
    species: list[dict[str, Any]],
) -> str:
    """Serialize ``protocol`` to a ``begin actions`` block with ``setConcentration``
    targets translated from codegen species names to ``@comp:Mol()`` patterns.

    ``generate_network`` is prepended so the action sequence materializes the
    network before the ``simulate`` phases run.
    """
    from bngsim.convert._protocol import StateChange, write_bngl_protocol

    name_to_idx = {s["name"]: i for i, s in enumerate(species)}

    def _retarget(step: object) -> object:
        if isinstance(step, StateChange) and step.kind == "set_concentration":
            idx = name_to_idx.get(step.target or "")
            if idx is not None:
                pattern = f"@{species_comp[idx]}:{mol_names[idx]}()"
                return StateChange(
                    kind=step.kind,
                    target=pattern,
                    value=step.value,
                    value_expr=step.value_expr,
                )
        return step

    translated = type(protocol)(
        steps=tuple(_retarget(s) for s in protocol.steps),
        source=protocol.source,
        dropped=protocol.dropped,
        lossy=protocol.lossy,
    )
    block = write_bngl_protocol(translated)
    # Materialize the network before simulating (generate_network is a build
    # directive parse_bngl_protocol drops, so the serializer never emits it).
    return block.replace(
        "begin actions\n",
        "begin actions\n\ngenerate_network({overwrite=>1})\n",
        1,
    )


def _pattern_side(indices: list[int], mol_names: list[str], species_comp: list[str]) -> str:
    """Render one side of a reaction rule as compartment-qualified ground patterns.

    An empty side (synthesis / degradation) is BNG's ``0``. Repeated indices
    become repeated patterns (stoichiometry > 1).
    """
    if not indices:
        return "0"
    return " + ".join(f"@{species_comp[i]}:{mol_names[i]}()" for i in indices)
