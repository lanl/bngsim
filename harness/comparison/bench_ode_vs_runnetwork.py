#!/usr/bin/env python3
"""Performance: BNGsim ODE vs run_network ODE (27 models).

Protocol: 2 warmup + 5 timed runs, median wall time.
BNGsim timing excludes model loading (amortized in PyBNF).
run_network timing includes full subprocess overhead.

Usage:
    python bench_ode_vs_runnetwork.py [--quick N] [--runs R] [--warmup W]

Output:
    results/bench_ode_vs_runnetwork.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    SUITE_ODE,
    geometric_mean,
    get_machine_info,
    load_suite,
    run_bngsim_ode,
    run_rn_ode,
    save_results,
    timed_runs,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark BNGsim ODE vs run_network")
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = parser.parse_args()

    print("=" * 70)
    print("  Performance: BNGsim ODE vs run_network")
    print(f"  Protocol: {args.warmup} warmup + {args.runs} timed")
    print("=" * 70)

    info = get_machine_info()
    models = load_suite(SUITE_ODE)
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

        entry = {
            "model": name,
            "species": m["species"],
            "reactions": m["reactions"],
            "t_end": t_end,
            "n_steps": n_steps,
        }

        # BNGsim
        bng = timed_runs(
            lambda n=net, t=t_end, s=n_steps: run_bngsim_ode(n, t, s),
            n_warmup=args.warmup,
            n_runs=args.runs,
            verbose=True,
        )
        if "error" in bng:
            print(f"    BNGsim ERROR: {bng['error'][:60]}")
            entry["bngsim_time"] = -1
            entry["error"] = bng["error"]
            results.append(entry)
            continue
        entry["bngsim_time"] = bng["median_time"]
        print(f"    BNGsim median: {bng['median_time']:.4f}s")

        # run_network
        rn = timed_runs(
            lambda n=net, t=t_end, s=n_steps: run_rn_ode(n, t, s),
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
        print(f"  Range: {min(speedups):.1f}x – {max(speedups):.1f}x")
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
    save_results(output, "bench_ode_vs_runnetwork")


if __name__ == "__main__":
    main()
