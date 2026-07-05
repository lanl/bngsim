#!/usr/bin/env python3
"""Validation: BNGsim SBML/Antimony loader vs libRoadRunner.

Tests BNGsim's Antimony/SBML loading path against libRoadRunner
for Pools A (ssys) and B (BioModels). Criterion: max_rel_err < 1e-3.

Usage:
    python validate_sbml.py [--pool a|b|ab] [--quick N]

Output:
    results/validate_sbml.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    cross_validate_by_name,
    discover_pool_a,
    discover_pool_b,
    get_machine_info,
    run_bngsim_ode_antimony,
    run_roadrunner_ode,
    save_results,
)

RTOL = 1e-3
T_END = 1.0
N_STEPS = 200


def validate_one(ant_path, pool, t_end, n_steps):
    """Validate one Antimony model."""
    name = ant_path.stem
    entry = {"model": name, "pool": pool}

    # BNGsim
    bng = run_bngsim_ode_antimony(str(ant_path), t_end, n_steps)
    if "error" in bng:
        entry["status"] = "bngsim_fail"
        entry["error"] = bng["error"]
        return entry

    # RoadRunner
    rr = run_roadrunner_ode(str(ant_path), t_end, n_steps)
    if "error" in rr:
        entry["status"] = "rr_fail"
        entry["error"] = rr["error"]
        return entry

    # Cross-validate by species name
    passed, max_err, n_matched, detail = cross_validate_by_name(
        bng.get("species_names", []),
        bng["species"],
        rr.get("species_names", []),
        rr["species"],
        rtol=RTOL,
    )

    entry["max_rel_err"] = max_err
    entry["n_matched"] = n_matched
    entry["status"] = "pass" if passed else "fail"
    entry["detail"] = detail

    return entry


def main():
    parser = argparse.ArgumentParser(description="Validate BNGsim SBML loader vs RoadRunner")
    parser.add_argument(
        "--pool",
        default="ab",
        choices=["a", "b", "ab"],
    )
    parser.add_argument("--quick", type=int, default=0)
    args = parser.parse_args()

    print("=" * 70)
    print("  Validation: BNGsim Antimony vs libRoadRunner")
    print(f"  Criterion: max_rel_err < {RTOL}")
    print("=" * 70)

    info = get_machine_info()
    models = []

    if "a" in args.pool:
        pa = discover_pool_a()
        print(f"  Pool A: {len(pa)} models")
        if args.quick > 0:
            pa = pa[: args.quick]
        models.extend([(p, "A") for p in pa])

    if "b" in args.pool:
        pb = discover_pool_b(require_xval=True)
        print(f"  Pool B: {len(pb)} models")
        if args.quick > 0:
            pb = pb[: args.quick]
        models.extend([(p, "B") for p in pb])

    print(f"  Total: {len(models)} models\n")

    results = []
    counts = {"pass": 0, "fail": 0, "bngsim_fail": 0, "rr_fail": 0}

    for i, (path, pool) in enumerate(models):
        name = path.stem
        print(
            f"  [{i + 1}/{len(models)}] {pool}/{name}...",
            end=" ",
            flush=True,
        )

        entry = validate_one(path, pool, T_END, N_STEPS)
        results.append(entry)
        st = entry["status"]
        counts[st] = counts.get(st, 0) + 1

        if st == "pass":
            err = entry.get("max_rel_err", 0)
            n = entry.get("n_matched", 0)
            print(f"PASS (err={err:.2e}, {n} sp)")
        else:
            detail = entry.get("error", entry.get("detail", ""))
            print(f"{st.upper()} ({detail[:60]})")

    # Summary
    print(f"\n{'=' * 70}")
    for k, v in sorted(counts.items()):
        if v > 0:
            print(f"  {k}: {v}")
    print(f"  Total: {len(models)}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "criterion": f"max_rel_err < {RTOL}",
        "t_end": T_END,
        "n_steps": N_STEPS,
        "summary": {
            "total": len(models),
            **counts,
        },
        "results": results,
    }
    save_results(output, "validate_sbml")


if __name__ == "__main__":
    main()
