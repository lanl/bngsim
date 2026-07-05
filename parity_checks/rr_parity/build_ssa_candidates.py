#!/usr/bin/env python3
"""Derive the SSA-compatible BioModels subset for cross-engine parity testing.

Scans a directory of BioModels SBML downloads and selects the models that
are genuine exact-SSA candidates, writing ``ssa_candidates.json`` next to
this script. A model is admitted only if it is, in order:

  1. non-empty and well-formed XML with no libSBML ERROR-severity diagnostics;
  2. carrying at least one ``<reaction>`` (SSA needs discrete channels --
     pure rule/assignment ODE models are excluded by design);
  3. loadable by ``bngsim.Model.from_sbml``; and
  4. clean under ``Model.validate_for_ssa()`` (integer populations + stoich,
     mass-action / split-reversible kinetics, no fast reactions, no
     compartment rate rules, etc.).

A further three structural gates reject models that pass ``validate_for_ssa``
but are not valid *exact-SSA* models (they surfaced as cross-engine "fails" in
the parity screen, all attributable to corpus quality rather than a bngsim
defect -- see ``dev/notes/SBML_VS_ROADRUNNER.md`` and GH #30):

  5. a reacting species whose initial molecule count (``conc x V``) rounds to 0
     (``0 < count < 0.5``) -- bngsim rounds non-integer populations, so the
     species starts at 0 molecules under exact SSA while a continuous-state
     engine keeps the fraction; the species is a continuous-ODE quantity, not a
     discrete population (``_subparticle_species``);
  6. an irreversible reaction whose rate law evaluates **negative** at the
     initial state (e.g. logistic / Allee growth above carrying capacity, or a
     constant-flux consumption) -- a sign-indefinite propensity is outside the
     exact-SSA framework, which requires non-negative propensities. bngsim
     handles it correctly (GH #110: the reaction fires in reverse with
     propensity ``|rate|``, mean-faithful to the ODE -- the SSA mean tracks
     bngsim's own CVODE), but a mean-faithful reverse-firing channel is an
     *extension* of exact SSA, not exact SSA, so it does not belong in an
     exact-SSA parity corpus (``_negative_initial_propensity``). RR's gillespie
     mishandles the same class (sys-bio/roadrunner#1318/#1320), so these read as
     cross-engine fails for an RR-side reason regardless; and
  7. hand-verified ``MANUAL_SSA_EXCLUDE`` entries for cases the structural gates
     miss (a sign-indefinite rate hidden behind local parameters, or an
     exact-SSA-intractable molecule count).

We deliberately do NOT repurpose continuous ODE models as SSA models: a
fractional initial population, a non-mass-action reversible law, a
sign-indefinite rate, or an intractable molecule count disqualifies the model
rather than being rounded or rescaled.

``validate_for_ssa`` is a pure-Python structural check, so a per-model
``SIGALRM`` reliably bounds it. We deliberately do NOT run a probe SSA
replicate here: a population-heavy model's ``sim.run`` is a long C++ call
that ``SIGALRM`` cannot interrupt (the signal only raises once control
returns to Python), so a single bomb would wedge the whole build. Cost
tiering is therefore left to the parity harness, which isolates each model
in a killable child process. We tag a coarse effort tier from species
count as a cheap static proxy.

The SBML download directory is resolved from ``$BIOMODELS_SBML_DIR`` and
defaults to ``~/Code/ssys/biomodels_batch/data/sbml_downloads`` (273M, not
vendored). The committed manifest stores only model IDs + metadata, never
the SBML itself.

KNOWN LIMITATION: ``Model.from_sbml`` is itself a C++ call that ``SIGALRM``
cannot interrupt, so a single pathological *load* (a huge network) can wedge
this script past ``--load-timeout``. The harvested ``ssa_candidates.json``
was therefore produced from an equivalent ``validate_for_ssa`` pass with
libSBML species counts; to make this script reliably re-runnable on the full
corpus, the per-model ``from_sbml`` + ``validate`` step needs the same
killable-subprocess isolation the parity harness uses. The admitted set is
identical either way.

Usage:
    cd bngsim && uv run python benchmarks/biomodels_ssa/build_ssa_candidates.py
                     [--sbml-dir DIR] [--load-timeout 60] [--limit N]

    # Re-apply only the libSBML-level structural gates (5/6/7 above) to the
    # already-committed manifest, without the from_sbml rebuild (which can wedge
    # on huge networks per the KNOWN LIMITATION below). Moves newly-excluded
    # models into ``_meta.structural_excludes``:
    cd bngsim && uv run python benchmarks/biomodels_ssa/build_ssa_candidates.py --prune
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import signal
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SBML_DIR = Path(
    os.environ.get(
        "BIOMODELS_SBML_DIR",
        os.path.expanduser("~/Code/ssys/biomodels_batch/data/sbml_downloads"),
    )
)
OUT_PATH = HERE / "ssa_candidates.json"

# Manually-confirmed SSA exclusions: ODE models that drive a non-boundary
# reacting species negative because it is consumed by a reaction whose rate law
# does not reference it (zeroth-order / other-species-driven consumption -- the
# propensity does not vanish when the pool empties). Post-GH #110 bngsim
# reproduces this *literally* (the count goes negative, matching bngsim's own
# CVODE -- non-negativity is the modeler's job), so it is NOT a bngsim defect;
# the species should have been declared boundaryCondition="true". We exclude it
# anyway on the same definitional basis as the sign-indefinite gate: a negative
# molecule pool is outside the exact-SSA framework and does not belong in an
# exact-SSA parity corpus. The broad structural signature (reactant absent from
# its own rate law) flags ~36 admitted models, but most are genuine catalysts
# written on both sides that never deplete (valid); we exclude only the
# hand-verified offenders and deliberately do NOT sweep the whole class -- the
# post-#110 review (GH #109) found no bngsim concern there, only model bugs both
# engines now handle identically.
MANUAL_SSA_EXCLUDE = {
    "BIOMD0000000484": "ES consumed by re2 at constant rate k1 (rate omits ES)",
    "BIOMD0000000485": "ES consumed by Reaction1/Reaction6 (rates omit ES)",
    "MODEL1504160000": "s7/s8 consumed at rates driven by other species, not themselves",
    # Sign-indefinite rate the t0 gate (_negative_initial_propensity) misses
    # because the kinetic law's sign depends on a local kineticLaw parameter
    # that evaluateASTNode does not resolve. Verified by hand: the
    # tumor/endothelial-volume growth term goes negative, so bngsim reverse-fires
    # it (GH #110, mean-faithful -- the SSA mean tracks bngsim's CVODE). Excluded
    # as outside the exact-SSA framework, not as a bngsim defect.
    "BIOMD0000000821": "tumor/endothelial-volume growth term is sign-indefinite (negative); bngsim reverse-fires it (GH #110), outside exact-SSA",
    # Sign-indefinite rates that the t0 gate (_negative_initial_propensity)
    # cannot catch: they are positive at t=0 and only go negative later in the
    # trajectory. There is no clean structural signature -- a subtraction
    # appears in an irreversible rate law in 112/310 admitted models, most of
    # them always-positive net rates -- so a "contains subtraction" gate would
    # over-prune. The hand-verified instances below are excluded directly
    # (matching the zeroth-order policy above). All are handled correctly by
    # bngsim post-GH #110 (the channel fires in reverse, mean-faithful -- the SSA
    # mean tracks bngsim's CVODE); they are excluded because a sign-indefinite
    # propensity is outside the exact-SSA framework, NOT because bngsim diverges.
    # (251 and 143 collapse a kf*A - kr*B reversible into one channel, so they are
    # additionally mean-faithful-but-not-fluctuation-faithful under reverse
    # firing -- a second reason a z-gate parity test is not meaningful for them.)
    # The live screen still surfaces this class via the harness's runtime
    # reverse-fire annotation (GH #109) for any model the t0/manual gates miss
    # (e.g. BIOMD0000000863, kept in corpus).
    "BIOMD0000000751": "immune_growth is logistic y*I*(1-I/K); rate < 0 once I exceeds capacity (positive at t0)",
    "BIOMD0000000752": "Wilkie2013r: same logistic immune_growth as BIOMD0000000751",
    "BIOMD0000000251": "reaction_7 cFOSm synthesis is a net forward-reverse rate (k7*cFOSp - k8*cFOSm); < 0 when degradation dominates (local params hide it from the t0 gate)",
    "BIOMD0000000143": "Olsen2003 PO oscillator: forward-reverse net-rate transport (R1, R14-R18) on irreversible channels + signed proton species (H_p/H_c go negative); not a valid exact-SSA model",
    # Exact-SSA-intractable: Ligand_EGF starts at 1.99e14 molecules, so the
    # Receptor_Ligand binding propensity is ~1e27/s. Exact SSA would need ~1e28
    # events to reach the horizon, so it cannot advance the slow downstream
    # reactions within max_steps (Internalised_Ligand frozen at 0 vs ODE 5.3e4).
    # A continuum/ODE model, not an exact-SSA model. Not a count threshold
    # because the count distribution has no clean cutoff (see GH #30); the
    # harness's wall-clock cap prunes the rest dynamically as `too_slow`.
    "BIOMD0000000985": "Ligand_EGF=1.99e14 -> binding propensity ~1e27; exact SSA intractable (continuum model)",
}

# Coarse effort tiers by species count (a cheap static proxy for SSA cost;
# the harness does the real wall-clock pruning). Small networks land in
# "low" so an --effort low run is a fast, broad smoke screen.
TIER_LOW_MAX_SPECIES = 10
TIER_MEDIUM_MAX_SPECIES = 40


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout()


def _has_negative_population(path: str) -> bool:
    """True if a reacting species declares a negative initial amount/conc.

    ``validate_for_ssa`` does not catch this: a negative initial value that
    happens to be integer-valued (e.g. MODEL2003060002's reacting species
    ``IAA`` at concentration -350) passes the integer-population check, yet a
    negative molecule pool is meaningless under SSA. Mirrors the
    ``negative_population`` drop in ``suites/biomodels/filter.py``; restricted
    to species that appear in a reaction so a signed rate-rule state variable
    is left alone.
    """
    import libsbml

    model = libsbml.readSBML(path).getModel()
    if model is None:
        return False
    reacting: set[str] = set()
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        for j in range(rxn.getNumReactants()):
            reacting.add(rxn.getReactant(j).getSpecies())
        for j in range(rxn.getNumProducts()):
            reacting.add(rxn.getProduct(j).getSpecies())
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if sp.getId() not in reacting:
            continue
        if sp.isSetInitialAmount() and sp.getInitialAmount() < 0:
            return True
        if sp.isSetInitialConcentration() and sp.getInitialConcentration() < 0:
            return True
    return False


def _reacting_species(model) -> set[str]:
    """IDs of species that appear as a reactant or product of some reaction."""
    s: set[str] = set()
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        for j in range(rxn.getNumReactants()):
            s.add(rxn.getReactant(j).getSpecies())
        for j in range(rxn.getNumProducts()):
            s.add(rxn.getProduct(j).getSpecies())
    return s


def _initial_count(species, model) -> float | None:
    """Molecule count seen by the SSA at t=0: initialAmount if set, else
    initialConcentration x compartment size. ``None`` if neither is set."""
    if species.isSetInitialAmount():
        return species.getInitialAmount()
    if species.isSetInitialConcentration():
        comp = model.getCompartment(species.getCompartment())
        vol = comp.getSize() if (comp is not None and comp.isSetSize()) else 1.0
        return species.getInitialConcentration() * vol
    return None


def _subparticle_species(path: str) -> list[tuple[str, float]]:
    """Reacting species whose initial molecule count rounds to 0 (0 < count <
    0.5). bngsim rounds non-integer SSA populations (0.4 -> 0, 0.5 -> 1), so
    such a species starts at 0 molecules under exact SSA and stays there while a
    continuous-state engine keeps the fraction -- it is a continuous-ODE
    quantity, not a discrete population. ``validate_for_ssa`` only warns on
    these; here they disqualify the whole model from the exact-SSA corpus.
    """
    import libsbml

    model = libsbml.readSBML(path).getModel()
    if model is None:
        return []
    reacting = _reacting_species(model)
    hits: list[tuple[str, float]] = []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if sp.getId() not in reacting:
            continue
        c = _initial_count(sp, model)
        if c is not None and 0.0 < c < 0.5:
            hits.append((sp.getId(), c))
    return hits


def _negative_initial_propensity(path: str, tol: float = 1e-9) -> list[tuple[str, float]]:
    """Irreversible reactions whose rate law evaluates **negative** at the
    initial state. A sign-indefinite propensity is outside the exact-SSA
    framework (which requires non-negative propensities): bngsim handles it by
    firing the reaction in reverse with propensity ``|rate|`` (GH #110,
    mean-faithful -- the SSA mean tracks the ODE), but a reverse-firing channel
    is an extension of exact SSA, not exact SSA, so the model is dropped from the
    exact-SSA corpus rather than admitted. The pattern is a logistic/Allee growth
    term above carrying capacity, or a constant-flux consumption written as a
    single reaction.

    Function definitions are inlined (``replaceFD``) and the law is evaluated
    against the model's declared initial values (``evaluateASTNode``). Reactions
    whose symbols do not fully resolve (some local ``kineticLaw`` parameters or
    ``initialAssignment`` chains) evaluate to NaN and are skipped, so this is a
    sound but not exhaustive gate -- hand-verified misses go in
    ``MANUAL_SSA_EXCLUDE``.
    """
    import math

    import libsbml

    model = libsbml.readSBML(path).getModel()
    if model is None:
        return []
    fds = model.getListOfFunctionDefinitions()
    hits: list[tuple[str, float]] = []
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn.getReversible():
            continue
        kl = rxn.getKineticLaw()
        if kl is None or kl.getMath() is None:
            continue
        math_ast = kl.getMath().deepCopy()
        libsbml.SBMLTransforms.replaceFD(math_ast, fds)
        val = libsbml.SBMLTransforms.evaluateASTNode(math_ast, model)
        if val is not None and not math.isnan(val) and val < -tol:
            hits.append((rxn.getId(), float(val)))
    return hits


def _tier_for(n_species: int) -> str:
    if n_species <= TIER_LOW_MAX_SPECIES:
        return "low"
    if n_species <= TIER_MEDIUM_MAX_SPECIES:
        return "medium"
    return "high"


def _prefilter(sbml_dir: Path) -> tuple[list[str], dict]:
    """libSBML-level prefilter. Returns (candidate paths, reject counts)."""
    import libsbml

    rdr = libsbml.SBMLReader()
    files = sorted(glob.glob(str(sbml_dir / "*.xml")))
    counts = {"total": len(files), "empty": 0, "sbml_error": 0, "no_model": 0, "zero_reactions": 0}
    candidates: list[str] = []
    for f in files:
        if os.path.getsize(f) == 0:
            counts["empty"] += 1
            continue
        doc = rdr.readSBML(f)
        if doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) > 0:
            counts["sbml_error"] += 1
            continue
        model = doc.getModel()
        if model is None:
            counts["no_model"] += 1
            continue
        if model.getNumReactions() == 0:
            counts["zero_reactions"] += 1
            continue
        candidates.append(f)
    return candidates, counts


def _prune_existing(sbml_dir: Path) -> int:
    """Re-apply only the libSBML-level structural gates (manual / sub-particle /
    sign-indefinite) to the already-committed manifest, without the from_sbml
    rebuild (which can wedge -- see KNOWN LIMITATION). Moves newly-excluded
    models into ``_meta.structural_excludes``. Idempotent: excluded models are
    removed from ``models``, so a second run is a no-op.
    """
    from collections import Counter

    if not OUT_PATH.exists():
        raise SystemExit(f"manifest not found: {OUT_PATH}")
    payload = json.loads(OUT_PATH.read_text())
    models = payload.get("models", [])

    kept: list[dict] = []
    excluded: list[dict] = []
    for m in models:
        name = m["name"]
        path = str(sbml_dir / f"{name}.xml")
        if not os.path.exists(path):
            kept.append(m)  # cannot re-check without the SBML; leave untouched
            continue
        if name in MANUAL_SSA_EXCLUDE:
            excluded.append(
                {"name": name, "reason": "manual_exclude", "detail": MANUAL_SSA_EXCLUDE[name]}
            )
            continue
        sp = _subparticle_species(path)
        if sp:
            detail = ", ".join(f"{sid}={c:.3g}" for sid, c in sp[:4])
            excluded.append({"name": name, "reason": "sub_particle", "detail": detail})
            continue
        neg = _negative_initial_propensity(path)
        if neg:
            detail = ", ".join(f"{rid} rate@t0={v:.3g}" for rid, v in neg[:4])
            excluded.append({"name": name, "reason": "sign_indefinite", "detail": detail})
            continue
        kept.append(m)

    try:
        import bngsim

        ver = getattr(bngsim, "__version__", "unknown")
    except Exception:
        ver = "unknown"

    # Accumulate: merge this run's exclusions with any already recorded, so
    # re-running (after the models are already pruned) preserves the history
    # rather than clobbering it with an empty set.
    prev = payload.get("_meta", {}).get("structural_excludes", {}).get("models", [])
    merged = {e["name"]: e for e in prev}
    for e in excluded:
        merged[e["name"]] = e
    all_excluded = sorted(merged.values(), key=lambda e: e["name"])

    payload["models"] = kept
    meta = payload.setdefault("_meta", {})
    meta["n_admitted"] = len(kept)
    meta["tier_counts"] = {
        t: sum(1 for a in kept if a.get("effort") == t) for t in ("low", "medium", "high")
    }
    meta["structural_excludes"] = {
        "generated": _dt.date.today().isoformat(),
        "bngsim_version": ver,
        "n_excluded": len(all_excluded),
        "counts": dict(Counter(e["reason"] for e in all_excluded)),
        "criteria": (
            "Exact-SSA structural gates re-applied to the validate_for_ssa-clean "
            "manifest (build_ssa_candidates.py gates 5/6/7): sub_particle "
            "(0<count<0.5 rounds to 0 molecules), sign_indefinite (irreversible "
            "reaction rate@t0 < 0), manual_exclude (hand-verified misses). These "
            "models pass validate_for_ssa but are not valid exact-SSA models; "
            "they surfaced as parity-screen fails attributable to corpus quality, "
            "not a bngsim defect (GH #30). Post-GH #110 bngsim handles the "
            "sign-indefinite class correctly (reverse-firing, mean-faithful); the "
            "gate drops them as outside the exact-SSA framework, not as a defect."
        ),
        "models": all_excluded,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"prune: {len(models)} -> {len(kept)} kept; "
        f"{len(excluded)} excluded this run {dict(Counter(e['reason'] for e in excluded))}; "
        f"{len(all_excluded)} excluded total {dict(Counter(e['reason'] for e in all_excluded))}"
    )
    print(f"wrote {OUT_PATH.relative_to(HERE.parents[2])}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sbml-dir", type=Path, default=DEFAULT_SBML_DIR)
    p.add_argument("--load-timeout", type=int, default=60)
    p.add_argument("--limit", type=int, default=0, help="Cap candidates scanned (0 = all).")
    p.add_argument(
        "--prune",
        action="store_true",
        help="Re-apply structural gates to the existing manifest (no rebuild).",
    )
    args = p.parse_args()

    sbml_dir = args.sbml_dir.expanduser()
    if not sbml_dir.is_dir():
        raise SystemExit(f"SBML download dir not found: {sbml_dir}\nSet $BIOMODELS_SBML_DIR.")

    if args.prune:
        return _prune_existing(sbml_dir)

    import bngsim

    signal.signal(signal.SIGALRM, _alarm)

    candidates, pre = _prefilter(sbml_dir)
    if args.limit:
        candidates = candidates[: args.limit]
    print(
        f"prefilter: {pre['total']} files -> {len(candidates)} reaction-bearing "
        f"(empty={pre['empty']} sbml_error={pre['sbml_error']} "
        f"zero_reactions={pre['zero_reactions']})",
        flush=True,
    )

    admitted: list[dict] = []
    rej = {
        "ssa_issues": 0,
        "load_error": 0,
        "timeout": 0,
        "negative_population": 0,
        "manual_exclude": 0,
        "sub_particle": 0,
        "sign_indefinite": 0,
    }
    issue_freq: dict[str, int] = {}
    t0 = time.perf_counter()

    for i, f in enumerate(candidates, 1):
        name = Path(f).stem
        try:
            signal.alarm(args.load_timeout)
            model = bngsim.Model.from_sbml(f)
            issues = model.validate_for_ssa()
            signal.alarm(0)
        except _Timeout:
            signal.alarm(0)
            rej["timeout"] += 1
            continue
        except Exception:
            signal.alarm(0)
            rej["load_error"] += 1
            continue

        if issues:
            rej["ssa_issues"] += 1
            for it in issues:
                code = getattr(it, "code", str(it))
                issue_freq[code] = issue_freq.get(code, 0) + 1
            continue

        # validate_for_ssa passes integer-valued negative initial populations
        # (e.g. -350); a negative molecule pool is still meaningless under SSA.
        if _has_negative_population(f):
            rej["negative_population"] += 1
            continue

        if name in MANUAL_SSA_EXCLUDE:
            rej["manual_exclude"] += 1
            continue

        # Structural exact-SSA gates (libSBML-only): a species that rounds to 0
        # molecules, or an irreversible reaction with a negative initial
        # propensity. Both pass validate_for_ssa but are not valid exact-SSA
        # models. (Kept consistent with the --prune path.)
        if _subparticle_species(f):
            rej["sub_particle"] += 1
            continue
        if _negative_initial_propensity(f):
            rej["sign_indefinite"] += 1
            continue

        # Clean for SSA. Tier by declared species count (static proxy);
        # no probe replicate -- the harness does the killable cost screen.
        n_species = len(getattr(model, "species_names", []) or [])
        admitted.append(
            {
                "name": name,
                "n_species": int(n_species),
                "effort": _tier_for(n_species),
            }
        )
        if i % 100 == 0:
            print(f"  scanned {i}/{len(candidates)}  admitted={len(admitted)}", flush=True)

    admitted.sort(key=lambda d: (d["n_species"], d["name"]))  # smallest first
    tier_counts = {
        t: sum(1 for a in admitted if a["effort"] == t) for t in ("low", "medium", "high")
    }

    payload = {
        "_meta": {
            "generated": _dt.date.today().isoformat(),
            # Redacted (not str(sbml_dir)): the corpus dir is environment-specific and
            # would bake a developer home dir into the committed manifest. The `note`
            # below records how a run resolves it ($BIOMODELS_SBML_DIR).
            "sbml_dir": "$BIOMODELS_SBML_DIR",
            "bngsim_version": getattr(bngsim, "__version__", "unknown"),
            "prefilter": pre,
            "rejected": rej,
            "ssa_issue_freq": dict(sorted(issue_freq.items(), key=lambda kv: -kv[1])),
            "n_admitted": len(admitted),
            "tier_counts": tier_counts,
            "note": (
                "SSA-compatible BioModels: well-formed + reaction-bearing + "
                "from_sbml-loadable + validate_for_ssa-clean. ODE models are "
                "NOT repurposed. SBML resolved from $BIOMODELS_SBML_DIR."
            ),
        },
        "models": admitted,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"\nadmitted {len(admitted)} SSA candidates "
        f"(low={tier_counts['low']} medium={tier_counts['medium']} high={tier_counts['high']}); "
        f"rejected {rej}; {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    print(f"wrote {OUT_PATH.relative_to(HERE.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
