#!/usr/bin/env python3
"""Adaptive time horizon calibration for Antimony model pools.

For each model in Pool A (ssys) and Pool B (BioModels), finds a t_end
where libRoadRunner takes meaningful wall time (~0.05–1.0s), enabling
performance comparisons that measure simulation cost, not overhead.

Usage:
    python calibrate_horizons.py [--pool a|b|ab] [--quick N]
    python calibrate_horizons.py --pool a          # ssys models only
    python calibrate_horizons.py --pool b          # BioModels only
    python calibrate_horizons.py --pool ab         # both (default)
    python calibrate_horizons.py --quick 20        # first 20 models only

Output:
    results/calibrated_horizons.json

Each entry: {model, pool, t_end, rr_time, bngsim_time, n_species, status}
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure harness directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    discover_pool_a,
    discover_pool_b,
    get_machine_info,
    save_results,
)

# Calibration parameters
TARGET_MIN_TIME = 0.05  # seconds — minimum meaningful timing
TARGET_MAX_TIME = 2.0  # seconds — don't go too high
MAX_T_END = 1e6  # absolute ceiling for t_end
N_STEPS = 200  # fixed output points
RR_TIMEOUT = 30  # seconds per RR attempt


def calibrate_one_model(ant_path: Path, pool: str) -> dict:
    """Find t_end where libRoadRunner takes TARGET_MIN_TIME to TARGET_MAX_TIME.

    Strategy: start at t_end=1.0, multiply by 10 until wall time >= TARGET_MIN_TIME.
    Also verify BNGsim succeeds at the final horizon.
    """
    import roadrunner

    name = ant_path.stem
    result = {"model": name, "pool": pool, "path": str(ant_path)}

    # Load model in RoadRunner via ant→SBML conversion
    try:
        from common import _ant_to_sbml

        sbml_str = _ant_to_sbml(str(ant_path))
        rr = roadrunner.RoadRunner(sbml_str)
        rr.integrator.absolute_tolerance = 1e-12
        rr.integrator.relative_tolerance = 1e-8
        n_species = rr.model.getNumFloatingSpecies()
        result["n_species"] = n_species
    except Exception as e:
        result["status"] = "rr_load_fail"
        result["error"] = str(e)[:200]
        return result

    # Binary search for appropriate t_end
    t_end = 1.0
    best_t_end = None
    best_rr_time = None

    while t_end <= MAX_T_END:
        try:
            rr.reset()
            t0 = time.perf_counter()
            rr.simulate(0, t_end, N_STEPS + 1)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            # Simulation failed at this horizon — stiffness/divergence
            if best_t_end is not None:
                # Use the last successful horizon
                break
            result["status"] = "rr_sim_fail"
            result["error"] = f"t_end={t_end}: {str(e)[:200]}"
            return result

        if elapsed >= TARGET_MIN_TIME:
            best_t_end = t_end
            best_rr_time = elapsed
            break

        best_t_end = t_end
        best_rr_time = elapsed

        if elapsed >= TARGET_MIN_TIME / 10:
            # Getting close — multiply by 5 instead of 10
            t_end *= 5
        else:
            t_end *= 10

    if best_t_end is None:
        result["status"] = "no_valid_horizon"
        return result

    result["t_end"] = best_t_end
    result["rr_time"] = best_rr_time
    result["n_steps"] = N_STEPS

    # Verify BNGsim also works at this horizon
    try:
        import bngsim

        model = bngsim.Model.from_antimony(str(ant_path))
        sim = bngsim.Simulator(model, method="ode")
        t0 = time.perf_counter()
        bng_result = sim.run(t_span=(0, best_t_end), n_points=N_STEPS + 1)
        bng_time = time.perf_counter() - t0
        species = np.asarray(bng_result.species)

        # Sanity check
        if np.any(np.isnan(species)) or np.any(np.isinf(species)):
            result["status"] = "bngsim_nan"
            result["bngsim_time"] = bng_time
            return result

        result["bngsim_time"] = bng_time
        result["status"] = "ok"

    except Exception as e:
        result["status"] = "bngsim_fail"
        result["error"] = str(e)[:200]

    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate time horizons for Antimony models")
    parser.add_argument(
        "--pool", default="ab", choices=["a", "b", "ab"], help="Which model pool(s) to calibrate"
    )
    parser.add_argument(
        "--quick", type=int, default=0, help="Limit to first N models per pool (0=all)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Adaptive Time Horizon Calibration")
    print("=" * 70)

    info = get_machine_info()
    models = []

    if "a" in args.pool:
        pool_a = discover_pool_a()
        print(f"\n  Pool A (ssys): {len(pool_a)} Antimony models")
        if args.quick > 0:
            pool_a = pool_a[: args.quick]
            print(f"  (limited to first {args.quick})")
        models.extend([(p, "A") for p in pool_a])

    if "b" in args.pool:
        pool_b = discover_pool_b(require_xval=True)
        print(f"  Pool B (BioModels): {len(pool_b)} cross-validated Antimony models")
        if args.quick > 0:
            pool_b = pool_b[: args.quick]
            print(f"  (limited to first {args.quick})")
        models.extend([(p, "B") for p in pool_b])

    print(f"\n  Total models to calibrate: {len(models)}")
    print()

    results = []
    n_ok = 0
    n_fail = 0

    for i, (ant_path, pool) in enumerate(models):
        name = ant_path.stem
        print(f"  [{i + 1}/{len(models)}] {pool}/{name}...", end=" ", flush=True)

        r = calibrate_one_model(ant_path, pool)
        results.append(r)

        status = r.get("status", "unknown")
        if status == "ok":
            n_ok += 1
            t_end = r["t_end"]
            rr_t = r["rr_time"]
            bng_t = r.get("bngsim_time", -1)
            n_sp = r.get("n_species", "?")
            print(f"OK  t_end={t_end:.1g}  RR={rr_t:.3f}s  BNG={bng_t:.3f}s  ({n_sp} sp)")
        else:
            n_fail += 1
            print(f"FAIL ({status}: {r.get('error', '')[:60]})")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Calibration complete: {n_ok} OK, {n_fail} failed, {len(models)} total")
    print(f"{'=' * 70}")

    # Separate ok results by pool
    for pool_label in ["A", "B"]:
        pool_ok = [r for r in results if r.get("pool") == pool_label and r.get("status") == "ok"]
        if pool_ok:
            times = [r["rr_time"] for r in pool_ok]
            print(f"\n  Pool {pool_label}: {len(pool_ok)} calibrated models")
            print(f"    RR time range: {min(times):.3f}s – {max(times):.3f}s")
            print(f"    Median RR time: {np.median(times):.3f}s")

    # Save
    output = {
        "machine_info": info,
        "config": {
            "target_min_time": TARGET_MIN_TIME,
            "target_max_time": TARGET_MAX_TIME,
            "max_t_end": MAX_T_END,
            "n_steps": N_STEPS,
        },
        "summary": {
            "total": len(models),
            "ok": n_ok,
            "failed": n_fail,
        },
        "models": results,
    }
    save_results(output, "calibrated_horizons")


if __name__ == "__main__":
    main()
