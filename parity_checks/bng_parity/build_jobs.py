#!/usr/bin/env python3
"""Emit the bng_parity _core manifest from the vendored corpus + overrides.

Reads ``manifest.json`` (written by vendor_corpus.py: one record per vendored
model with tier/source/relpath/sha256/methods/stochastic/companions) and emits
``jobs.json``, a _core manifest of one Job per model:

  * model_id          = the model's relpath under models/ (unique; basenames
                        collide across tiers, relpaths don't)
  * input_format      = "bngl"
  * method            = "ode" for deterministic models, "stochastic" otherwise
                        (the model's own actions decide; the regime picks the
                        oracle)
  * reference_engine  = "bng"  (legacy BNG2.pl / run_network / NFsim)
  * oracle            = max_rel_err (deterministic) | mean_zscore (stochastic)
  * params            = tier, source, methods, sha256, companions, has_free,
                        + seed/n_rep for stochastic
  * overrides         = the per-model fixtures from overrides.py, resolved onto
                        this model by basename (warns on basename collisions and
                        stale override keys)

Usage:
    python build_jobs.py [--manifest manifest.json] [--out jobs.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so `import overrides` resolves
import overrides as OV  # noqa: E402
from _core import Job, Oracle, Override, write_manifest  # noqa: E402

# Default oracles. Deterministic uses the combined abs/rel cell error; the
# stochastic gate matches the legacy parity harness's ensemble z-score.
ODE_TOL = 1e-4
SSA_TOL = 5.0
DEFAULT_SEED = 1
DEFAULT_N_REP = 10  # base ensemble; the runner escalates noisy DIFFs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--out", type=Path, default=HERE / "jobs.json")
    args = ap.parse_args()

    manifest_doc = json.loads(args.manifest.read_text())
    # manifest.json is the schema-wrapped {..., "records": [...]}; tolerate the
    # legacy bare-array shape too.
    records = manifest_doc["records"] if isinstance(manifest_doc, dict) else manifest_doc

    jobs = []
    matched_keys: set[str] = set()
    rehab_matched: set[str] = set()
    collisions: dict[str, list[str]] = {}
    regime_counts = {"ode": 0, "stochastic": 0}

    for r in records:
        basename = Path(r["relpath"]).name
        # `vendored` is already the path under models/ (tier/source/relpath),
        # which is the unique key. Do NOT strip a "models/" segment: RuleMonkey
        # models live under an internal `tests/models/corpus/...`, so splitting
        # on "models/" mangled 506 ids to a non-resolving suffix.
        model_id = r["vendored"]
        stochastic = r["stochastic"]

        ov_dicts = OV.overrides_for(basename)
        if ov_dicts:
            matched_keys.add(basename)
            collisions.setdefault(basename, []).append(model_id)

        # A model_id-keyed REHAB protocol replaces a dud model's actions with a
        # short runnable fixture (see overrides.REHAB). Attach it as an override
        # so the manifest carries it to the --jobs sweep path.
        rehab = OV.REHAB.get(model_id)
        if rehab is not None:
            rehab_matched.add(model_id)
            ov_dicts = ov_dicts + [
                {
                    "field": "rehab",
                    "value": rehab,
                    "reason": (
                        "rehab: model shipped without a runnable simulation protocol; "
                        "inject a short fixture so it yields a fingerprintable artifact"
                    ),
                }
            ]

        # An injected run action (action_inject or rehab) is authoritative for
        # the regime: such a model ships with no active action (its vendor
        # `stochastic` flag is a bare default), and parity_sweep injects the
        # action *before* its own stochastic classification. Mirror that here so
        # method (and the det/seedK output dir the golden emitter reads) matches
        # the run.
        inj = next((d for d in ov_dicts if d["field"] in ("action_inject", "rehab")), None)
        injected = None
        if inj is not None:
            m = re.search(r'method\s*=>\s*"(\w+)"', inj["value"])
            injected = m.group(1).lower() if m else None
            if injected in ("ode", "cvode"):
                stochastic = False
            elif injected in ("ssa", "nf", "pla"):
                stochastic = True

        method = "stochastic" if stochastic else "ode"
        regime_counts[method] += 1

        # An injected action REPLACES the model's own actions, so it is also the
        # authoritative simulate METHOD. Reflect it into params.methods (the
        # stochastic track-selection key in bng_stoch_run) — otherwise an injected
        # nf/ssa model whose source ships no action keeps the vendor's empty methods
        # list and is silently dropped by both the ssa and nf track filters.
        methods = [injected] if injected else r["methods"]
        params = {
            "tier": r["tier"],
            "source": r["source"],
            "methods": methods,
            "sha256": r["sha256"],
            "has_free": r["has_free"],
            "companions": r.get("companions", []),
        }
        if stochastic:
            params["seed"] = DEFAULT_SEED
            params["n_rep"] = DEFAULT_N_REP

        jobs.append(
            Job(
                model_id=model_id,
                input_format="bngl",
                method=method,
                reference_engine="bng",
                model=r["vendored"],
                oracle=Oracle(
                    metric="mean_zscore" if stochastic else "max_rel_err",
                    tol=SSA_TOL if stochastic else ODE_TOL,
                ),
                params=params,
                overrides=[Override(**d) for d in ov_dicts],
            )
        )

    # Surface override hygiene: stale keys (no vendored model) and collisions
    # (one basename → several vendored models, so the override fans out).
    stale = sorted(OV.ALL_KEYS - matched_keys)
    rehab_stale = sorted(set(OV.REHAB) - rehab_matched)
    fanned = {k: v for k, v in collisions.items() if len(v) > 1}

    meta = {
        "description": "bngsim-vs-legacy-BNG parity jobs (BNGL/.net corpus).",
        "reference_engine": "bng",
        "regime_counts": regime_counts,
        "defaults": {
            "ode_tol": ODE_TOL,
            "ssa_tol": SSA_TOL,
            "seed": DEFAULT_SEED,
            "n_rep": DEFAULT_N_REP,
        },
        "override_stale_keys": stale,
        "rehab_stale_model_ids": rehab_stale,
        "override_basename_collisions": fanned,
        "notes": (
            "One job per vendored model; model_id is the relpath under models/. Overrides are "
            "resolved by basename onto the matching model(s); a basename hitting >1 model is "
            "recorded in override_basename_collisions (fans out to all). seed/n_rep are the base "
            "stochastic ensemble; the runner escalates noisy DIFFs."
        ),
    }
    write_manifest(args.out, jobs, meta=meta)
    print(f"wrote {args.out} : {len(jobs)} jobs  regimes={regime_counts}")
    if stale:
        print(f"  WARN {len(stale)} stale override key(s) match no vendored model: {stale}")
    if rehab_stale:
        print(
            f"  WARN {len(rehab_stale)} rehab model_id(s) match no vendored model: {rehab_stale}"
        )
    if fanned:
        print(f"  WARN {len(fanned)} override basename(s) fan out to multiple models:")
        for k, v in fanned.items():
            print(f"    {k} -> {len(v)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
