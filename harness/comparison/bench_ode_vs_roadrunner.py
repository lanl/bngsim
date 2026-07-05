#!/usr/bin/env python3
"""Performance: BNGsim ODE vs libRoadRunner (Antimony models).

Uses calibrated time horizons from calibrate_horizons.py.
Falls back to t_end=1.0 if calibration data unavailable.

Usage:
    python bench_ode_vs_roadrunner.py [--pool a|b|ab] [--quick N]

Output:
    results/bench_ode_vs_roadrunner.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    RESULTS_DIR,
    discover_pool_a,
    discover_pool_b,
    geometric_mean,
    get_machine_info,
    run_bngsim_ode_antimony,
    run_roadrunner_ode,
    save_results,
    timed_runs,
)

FALLBACK_T_END = 1.0
FALLBACK_N_STEPS = 200


def load_horizons():
    """Load calibrated horizons if available."""
    path = RESULTS_DIR / "calibrated_horizons.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    horizons = {}
    for m in data.get("models", []):
        if m.get("status") == "ok":
            horizons[m["model"]] = {
                "t_end": m["t_end"],
                "n_steps": m.get("n_steps", 200),
            }
    return horizons


def main():
    parser = argparse.ArgumentParser(description="Benchmark BNGsim vs libRoadRunner")
    parser.add_argument(
        "--pool",
        default="ab",
        choices=["a", "b", "ab"],
    )
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Performance: BNGsim ODE vs libRoadRunner")
    print(f"  Protocol: {args.warmup} warmup + {args.runs} timed")
    print("=" * 70)

    info = get_machine_info()
    horizons = load_horizons()
    if horizons:
        print(f"  Using {len(horizons)} calibrated horizons")
    else:
        print(f"  No calibration data — using t_end={FALLBACK_T_END}")

    models = []
    if "a" in args.pool:
        pa = discover_pool_a()
        if args.quick > 0:
            pa = pa[: args.quick]
        models.extend([(p, "A") for p in pa])
        print(f"  Pool A: {len(pa)} models")
    if "b" in args.pool:
        pb = discover_pool_b(require_xval=True)
        if args.quick > 0:
            pb = pb[: args.quick]
        models.extend([(p, "B") for p in pb])
        print(f"  Pool B: {len(pb)} models")

    print(f"  Total: {len(models)}\n")

    results = []
    speedups = []
    n_skip = 0

    for i, (path, pool) in enumerate(models):
        name = path.stem
        h = horizons.get(name, {})
        t_end = h.get("t_end", FALLBACK_T_END)
        n_steps = h.get("n_steps", FALLBACK_N_STEPS)

        print(
            f"  [{i + 1}/{len(models)}] {pool}/{name} (t={t_end:.1g})...",
            end=" ",
            flush=True,
        )

        entry = {
            "model": name,
            "pool": pool,
            "t_end": t_end,
            "n_steps": n_steps,
        }

        # BNGsim
        bng = timed_runs(
            lambda p=str(path), t=t_end, s=n_steps: run_bngsim_ode_antimony(p, t, s),
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
        if "error" in bng:
            entry["status"] = "bngsim_fail"
            entry["error"] = bng["error"]
            results.append(entry)
            n_skip += 1
            print("BNG_FAIL")
            continue
        entry["bngsim_time"] = bng["median_time"]

        # RoadRunner
        rr = timed_runs(
            lambda p=str(path), t=t_end, s=n_steps: run_roadrunner_ode(p, t, s),
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
        if "error" in rr:
            entry["status"] = "rr_fail"
            entry["error"] = rr["error"]
            results.append(entry)
            n_skip += 1
            print("RR_FAIL")
            continue

        entry["rr_time"] = rr["median_time"]
        entry["status"] = "ok"

        bt = bng["median_time"]
        rt = rr["median_time"]
        if bt > 0 and rt > 0:
            ratio = bt / rt  # <1 means BNGsim faster
            entry["ratio_bng_rr"] = ratio
            speedups.append(ratio)
            faster = "BNG" if ratio < 1 else "RR"
            factor = 1 / ratio if ratio < 1 else ratio
            print(f"BNG={bt:.4f}s RR={rt:.4f}s ({faster} {factor:.1f}x)")
        else:
            print(f"BNG={bt:.4f}s RR={rt:.4f}s")

        results.append(entry)

    # Summary
    print(f"\n{'=' * 70}")
    n_ok = len(speedups)
    if speedups:
        gm = geometric_mean(speedups)
        n_bng_wins = sum(1 for s in speedups if s < 1)
        n_rr_wins = sum(1 for s in speedups if s >= 1)
        print(f"  BNGsim/RR ratio geometric mean: {gm:.2f}")
        print(f"  BNGsim faster: {n_bng_wins}  RR faster: {n_rr_wins}")
    print(f"  Benchmarked: {n_ok}  Skipped: {n_skip}  Total: {len(models)}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "summary": {
            "n_benchmarked": n_ok,
            "n_skipped": n_skip,
            "ratio_geometric_mean": (geometric_mean(speedups) if speedups else None),
        },
        "results": results,
    }
    save_results(output, "bench_ode_vs_roadrunner")


if __name__ == "__main__":
    main()
