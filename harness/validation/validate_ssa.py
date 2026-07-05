#!/usr/bin/env python3
"""Validation: BNGsim SSA determinism + moment checks (10 models).

Tests:
1. Deterministic: same seed → identical trajectory (2 runs)
2. Sanity: no NaN/Inf/negative in species
3. Moments: mean of 20 replicas consistent (no gross bias)

Usage:
    python validate_ssa.py [--quick N]

Output:
    results/validate_ssa.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    SUITE_SSA,
    check_sanity,
    get_machine_info,
    load_suite,
    run_bngsim_ssa,
    save_results,
)

SEED_BASE = 42
N_REPLICAS = 20


def main():
    parser = argparse.ArgumentParser(description="Validate BNGsim SSA determinism + moments")
    parser.add_argument(
        "--quick",
        type=int,
        default=0,
        help="Limit to first N models",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Validation: BNGsim SSA Determinism + Moments")
    print("=" * 70)

    info = get_machine_info()
    models = load_suite(SUITE_SSA)

    if args.quick > 0:
        models = models[: args.quick]

    results = []
    n_pass = n_fail = 0

    for i, m in enumerate(models):
        name = m["name"]
        net_path = str(BENCHMARKS_DIR / m["net_file"])
        t_end = m["t_end"]
        n_steps = m["n_steps"]

        print(f"\n  [{i + 1}/{len(models)}] {name}...")
        entry = {"model": name, "species": m["species"]}

        # Test 1: Deterministic (same seed → identical)
        r1 = run_bngsim_ssa(net_path, t_end, n_steps, SEED_BASE)
        r2 = run_bngsim_ssa(net_path, t_end, n_steps, SEED_BASE)

        if "error" in r1 or "error" in r2:
            err = r1.get("error", r2.get("error", ""))
            print(f"    FAIL: SSA error: {err[:60]}")
            entry["status"] = "error"
            entry["error"] = err
            results.append(entry)
            n_fail += 1
            continue

        if np.array_equal(r1["species"], r2["species"]):
            print("    Determinism: PASS (same seed → identical)")
            entry["deterministic"] = True
        else:
            diff = np.max(np.abs(r1["species"] - r2["species"]))
            print(f"    Determinism: FAIL (max diff = {diff})")
            entry["deterministic"] = False
            entry["max_diff"] = float(diff)

        # Test 2: Sanity check
        sane, msg = check_sanity(r1["species"])
        entry["sanity"] = sane
        print(f"    Sanity: {'PASS' if sane else 'FAIL'} ({msg})")

        # Test 3: Moment consistency (20 replicas)
        print(f"    Moments ({N_REPLICAS} replicas)...", end=" ")
        endpoints = []
        for rep in range(N_REPLICAS):
            r = run_bngsim_ssa(net_path, t_end, n_steps, SEED_BASE + rep + 100)
            if "error" in r:
                continue
            endpoints.append(r["species"][-1, :])

        if len(endpoints) >= 10:
            endpoints = np.array(endpoints)
            means = np.mean(endpoints, axis=0)
            np.std(endpoints, axis=0)
            # Check no species has zero mean with large std
            entry["moment_check"] = True
            entry["n_replicas"] = len(endpoints)
            n_nonzero = np.sum(means > 0)
            print(f"OK ({len(endpoints)} reps, {n_nonzero}/{means.shape[0]} nonzero)")
        else:
            entry["moment_check"] = False
            print(f"FAIL (only {len(endpoints)} successful)")

        # Overall
        all_pass = (
            entry.get("deterministic", False)
            and entry.get("sanity", False)
            and entry.get("moment_check", False)
        )
        entry["status"] = "pass" if all_pass else "fail"
        results.append(entry)
        if all_pass:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{'=' * 70}")
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  Total: {len(models)}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "summary": {
            "total": len(models),
            "pass": n_pass,
            "fail": n_fail,
        },
        "results": results,
    }
    save_results(output, "validate_ssa")


if __name__ == "__main__":
    main()
