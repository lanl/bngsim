#!/usr/bin/env python3
"""Performance: BNGsim SSA vs run_network SSA (10 models).

Protocol: 2 warmup + 5 timed runs, median wall time.

Usage:
    python bench_ssa_vs_runnetwork.py [--quick N] [--runs R]

Output:
    results/bench_ssa_vs_runnetwork.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    SUITE_SSA,
    geometric_mean,
    get_machine_info,
    load_suite,
    run_bngsim_ssa,
    run_rn_ssa,
    save_results,
    timed_runs,
)

SEED_BASE = 1000
# erk_activation has populations up to 3M → SSA infeasible
SSA_SKIP = {"erk_activation"}


def main():
    parser = argparse.ArgumentParser(description="Benchmark BNGsim SSA vs run_network")
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = parser.parse_args()

    print("=" * 70)
    print("  Performance: BNGsim SSA vs run_network")
    print(f"  Protocol: {args.warmup} warmup + {args.runs} timed")
    print("=" * 70)

    info = get_machine_info()
    models = load_suite(SUITE_SSA)
    if args.quick > 0:
        models = models[: args.quick]

    results = []
    speedups = []

    for i, m in enumerate(models):
        name = m["name"]
        net = str(BENCHMARKS_DIR / m["net_file"])
        t_end = m["t_end"]
        n_steps = m["n_steps"]

        print(f"\n  [{i + 1}/{len(models)}] {name} ({m['species']} sp, {m['reactions']} rxn)")

        if name in SSA_SKIP:
            print("    SKIP (SSA infeasible)")
            results.append(
                {
                    "model": name,
                    "skipped": True,
                    "reason": "infeasible",
                }
            )
            continue

        entry = {
            "model": name,
            "species": m["species"],
            "reactions": m["reactions"],
        }

        # BNGsim SSA
        seed_i = [0]

        def bng_run(n=net, t=t_end, s=n_steps, _seed=seed_i):
            _seed[0] += 1
            return run_bngsim_ssa(n, t, s, SEED_BASE + _seed[0])

        bng = timed_runs(
            bng_run,
            n_warmup=args.warmup,
            n_runs=args.runs,
            verbose=True,
        )
        if "error" in bng:
            print(f"    BNGsim ERROR: {bng['error'][:60]}")
            entry["bngsim_time"] = -1
            results.append(entry)
            continue
        entry["bngsim_time"] = bng["median_time"]
        print(f"    BNGsim median: {bng['median_time']:.4f}s")

        # run_network SSA
        seed_j = [0]

        def rn_run(n=net, t=t_end, s=n_steps, _seed=seed_j):
            _seed[0] += 1
            return run_rn_ssa(n, t, s, SEED_BASE + _seed[0])

        rn = timed_runs(
            rn_run,
            n_warmup=args.warmup,
            n_runs=args.runs,
            verbose=True,
        )
        if "error" in rn:
            print(f"    run_network ERROR: {rn['error'][:60]}")
            entry["rn_time"] = -1
            results.append(entry)
            continue
        entry["rn_time"] = rn["median_time"]
        print(f"    run_network median: {rn['median_time']:.4f}s")

        # Speedup
        if bng["median_time"] > 0 and rn["median_time"] > 0:
            su = rn["median_time"] / bng["median_time"]
            entry["speedup"] = su
            speedups.append(su)
            print(f"    Speedup: {su:.1f}x")

        results.append(entry)

    # Summary
    print(f"\n{'=' * 70}")
    if speedups:
        gm = geometric_mean(speedups)
        print(f"  Geometric mean speedup: {gm:.1f}x")
    print(f"  Models benchmarked: {len(speedups)}/{len(models)}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "summary": {
            "n_models": len(speedups),
            "geometric_mean_speedup": (geometric_mean(speedups) if speedups else None),
        },
        "results": results,
    }
    save_results(output, "bench_ssa_vs_runnetwork")


if __name__ == "__main__":
    main()
