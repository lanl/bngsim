#!/usr/bin/env python3
"""Build ``amici_ode_jobs.json`` — a curated subset of the rr_parity ODE corpus.

AMICI compiles a bespoke C++ extension per model (~20s cold), so the full
1323-model corpus is impractical for a first sweep. This selects a representative
subset that (a) spans model size — stratified over species-count bins — and (b)
covers the structural SBML features (events, rate rules, assignment rules,
variable volume, function definitions), preferring models the rr_parity sweep
shows both engines already handle (outcome PASS/DIFF) so the subset exercises the
comparable core rather than models AMICI simply can't import.

The output is a ``_core`` manifest (same schema as ``rr_parity/ode_jobs.json``)
with ``reference_engine`` set to ``amici`` and suite-relative ``model`` paths that
resolve under ``rr_parity/`` (where the SBML corpus lives) — amici_run.py reads it
by default. Very large models (>VERY_LARGE species) are excluded from the first
subset (their cold compile is minutes); the count of skipped giants is recorded in
``_meta`` so the omission is explicit, never silent.

Usage:
    cd bngsim && .venv/bin/python parity_checks/amici_parity/build_amici_jobs.py
    .venv/bin/python parity_checks/amici_parity/build_amici_jobs.py --per-bin 12 --feature-min 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RR_PARITY = HERE.parent / "rr_parity"
sys.path.insert(0, str(RR_PARITY))

import _sbml_features as sf  # noqa: E402

RR_ODE_JOBS = RR_PARITY / "ode_jobs.json"
RR_REPORT = RR_PARITY / "runs" / "report_ode.json"
OUT = HERE / "amici_ode_jobs.json"

# Species-count bins (upper-inclusive). Giants (> the last edge) are excluded.
SPECIES_BINS = [(0, 5), (6, 20), (21, 50), (51, 150), (151, 500)]
VERY_LARGE = 500

# Structural feature tags we want represented (substring-matched against the
# _sbml_features feature list, which renders e.g. "events (5)", "rate rules").
FEATURE_KEYS = [
    "events",
    "rate rules",
    "assignment rules",
    "variable volume",
    "function definitions",
]


def _comparable_models() -> set[str] | None:
    """model_ids the rr_parity report shows both engines ran (PASS/DIFF), or None
    if no report is present (then no preference is applied)."""
    if not RR_REPORT.exists():
        return None
    with open(RR_REPORT) as fh:
        rep = json.load(fh)
    out = set()
    for r in rep.get("results", []):
        if r.get("outcome") in ("PASS", "DIFF"):
            out.add(r["model_id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--per-bin", type=int, default=10, help="Models per species-count bin.")
    ap.add_argument(
        "--feature-min",
        type=int,
        default=3,
        help="Minimum models guaranteed to carry each structural feature tag.",
    )
    ap.add_argument(
        "--no-prefer-comparable",
        action="store_true",
        help="Do not prefer models the rr_parity report shows both engines ran.",
    )
    args = ap.parse_args()

    if not RR_ODE_JOBS.exists():
        sys.exit(f"missing {RR_ODE_JOBS}; run rr_parity/build_ode_jobs.py first.")
    with open(RR_ODE_JOBS) as fh:
        manifest = json.load(fh)
    jobs = manifest["jobs"]

    comparable = None if args.no_prefer_comparable else _comparable_models()

    # Tag every candidate job with n_species + feature tags (parse once each).
    tagged = []
    for j in jobs:
        path = (RR_PARITY / j["model"]).resolve()
        if not path.exists():
            continue
        feats = sf.extract_sbml_features(path)
        n_sp = feats.get("n_species")
        if n_sp is None:  # unparseable — skip (it would fail both engines anyway)
            continue
        tags = feats.get("features", [])
        tagged.append((j, int(n_sp), tags))

    # Preference key: comparable models first, then small models (cheaper compile),
    # then model_id for determinism.
    def pref(item):
        j, n_sp, _ = item
        is_comp = 0 if (comparable is None or j["model_id"] in comparable) else 1
        return (is_comp, n_sp, j["model_id"])

    tagged.sort(key=pref)

    chosen: dict[str, tuple] = {}  # model_id -> (job, n_sp, tags)
    n_giants = 0

    # 1. Stratified pick over species bins.
    for lo, hi in SPECIES_BINS:
        picked = 0
        for item in tagged:
            j, n_sp, _ = item
            if j["model_id"] in chosen:
                continue
            if lo <= n_sp <= hi:
                chosen[j["model_id"]] = item
                picked += 1
                if picked >= args.per_bin:
                    break

    # 2. Guarantee feature coverage.
    for key in FEATURE_KEYS:
        have = sum(1 for (_, _, tags) in chosen.values() if any(key in t for t in tags))
        if have >= args.feature_min:
            continue
        for item in tagged:
            j, n_sp, tags = item
            if j["model_id"] in chosen or n_sp > VERY_LARGE:
                continue
            if any(key in t for t in tags):
                chosen[j["model_id"]] = item
                have += 1
                if have >= args.feature_min:
                    break

    # Count giants we deliberately left out (for the _meta record).
    n_giants = sum(1 for (_, n_sp, _) in tagged if n_sp > VERY_LARGE)

    out_jobs = []
    for mid in sorted(chosen):
        j, _, _ = chosen[mid]
        jj = dict(j)
        jj["reference_engine"] = "amici"
        out_jobs.append(jj)

    # Feature/size coverage summary for the manifest header.
    bins_summary = {f"{lo}-{hi}": 0 for lo, hi in SPECIES_BINS}
    feat_summary = {k: 0 for k in FEATURE_KEYS}
    for _, n_sp, tags in chosen.values():
        for lo, hi in SPECIES_BINS:
            if lo <= n_sp <= hi:
                bins_summary[f"{lo}-{hi}"] += 1
                break
        for k in FEATURE_KEYS:
            if any(k in t for t in tags):
                feat_summary[k] += 1

    out = {
        "_meta": {
            "description": (
                "Curated bngsim-vs-AMICI ODE parity subset of rr_parity/ode_jobs.json, "
                "stratified by species count + structural feature coverage. AMICI "
                "compiles a C++ extension per model, so this is a representative "
                "subset, not the full corpus."
            ),
            "reference_engine": "amici",
            "source_manifest": "rr_parity/ode_jobs.json",
            "selection": {
                "per_bin": args.per_bin,
                "feature_min": args.feature_min,
                "prefer_comparable": comparable is not None,
                "species_bins_chosen": bins_summary,
                "feature_coverage": feat_summary,
                "very_large_excluded": {
                    "threshold_species": VERY_LARGE,
                    "count_in_corpus": n_giants,
                },
            },
            "n_jobs": len(out_jobs),
            "notes": (
                "model paths are suite-relative and resolve under rr_parity/ (the "
                "SBML corpus). reference_engine=amici. Regenerate with build_amici_jobs.py."
            ),
        },
        "jobs": out_jobs,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}  ({len(out_jobs)} jobs)")
    print(f"  species bins: {bins_summary}")
    print(f"  feature coverage: {feat_summary}")
    print(f"  giants (>{VERY_LARGE} sp) excluded from subset: {n_giants} in corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
