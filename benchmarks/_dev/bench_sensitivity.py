#!/usr/bin/env python3
"""Benchmark: CVODES sensitivity overhead vs plain ODE (Session 27).

Measures the cost overhead of computing dY/dp via CVODES forward
sensitivity analysis across the ODE benchmark suite.

For each model:
  1. Run plain ODE (no sensitivities) — baseline wall time + RHS evals
  2. Run with sensitivity_params = [all params] — worst-case overhead
  3. Run with sensitivity_params = [2 params] — typical use case
  4. Report wall time ratio and extra RHS evaluations

This answers: "How much does computing dY/dp cost?"
"""

import json
import math
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]  # bngsim/benchmarks
SUITE_FILE = BENCH_DIR / "_dev" / "suite_ode.json"


def bench_one(net_path, t_end, n_points, n_runs, sens_params=None):
    """Benchmark a single configuration.

    Returns (median_time_s, n_steps, n_rhs_evals).
    """
    from bngsim._bngsim_core import (
        CvodeSimulator,
        NetworkModel,
        SolverOptions,
        TimeSpec,
    )

    times_list = []
    for _ in range(n_runs):
        model = NetworkModel.from_net(str(net_path))
        sim = CvodeSimulator(model)
        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = t_end
        ts.n_points = n_points
        opts = SolverOptions()
        opts.max_steps = 100000

        if sens_params:
            opts.set_sensitivity_params(sens_params)

        t0 = time.perf_counter()
        result = sim.run(ts, opts)
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)

    med = sorted(times_list)[len(times_list) // 2]
    return med, result.solver_stats.n_steps, result.solver_stats.n_rhs_evals


def get_param_names(net_path):
    """Get all parameter names from a .net file."""
    from bngsim._bngsim_core import NetworkModel

    model = NetworkModel.from_net(str(net_path))
    return list(model.param_names)


def main():
    with open(SUITE_FILE) as f:
        suite = json.load(f)

    print("=" * 100)
    print("  CVODES Forward Sensitivity Overhead Benchmark (Session 27)")
    print("=" * 100)
    print()
    print("Protocol: 2 warmup + 3 timed runs, median wall time.")
    print("Sensitivity modes: plain ODE, 2-param, all-param (worst case)")
    print()

    hdr = (
        f"{'Model':<20} {'Sp':>5} {'Np':>4}  "
        f"{'Plain(ms)':>10} {'2p(ms)':>10} {'All-p(ms)':>11} "
        f"{'2p/base':>8} {'All/base':>9} "
        f"{'RHS_base':>9} {'RHS_2p':>9} {'RHS_all':>9}"
    )
    print(hdr)
    print("-" * 100)

    geo_2p = []
    geo_all = []
    results = []

    n_warmup = 2
    n_timed = 3
    n_runs = n_warmup + n_timed

    for entry in suite["models"]:
        name = entry["name"]
        net_path = BENCH_DIR / entry["net_file"]
        t_end = entry["t_end"]
        n_pts = entry.get("n_steps", 200)

        if not net_path.exists():
            print(f"{name:<20}  SKIP (net file not found)")
            continue

        try:
            all_params = get_param_names(net_path)
            n_params = len(all_params)

            # Pick 2 params for typical case (first 2)
            sens_2 = all_params[: min(2, n_params)]

            # Plain ODE (no sensitivity)
            base_time, base_steps, base_rhs = bench_one(net_path, t_end, n_pts, n_runs)

            # 2-param sensitivity
            s2_time, s2_steps, s2_rhs = bench_one(
                net_path, t_end, n_pts, n_runs, sens_params=sens_2
            )

            # All-param sensitivity (worst case).
            # Skip when Np > 10: CVODES overhead is O(Np) per step, so
            # all-param with Np=50+ takes thousands of times longer.
            if n_params <= 10:
                sa_time, sa_steps, sa_rhs = bench_one(
                    net_path, t_end, n_pts, n_runs, sens_params=all_params
                )
            else:
                # Skip all-param for models with many params
                sa_time, _sa_steps, sa_rhs = float("nan"), 0, 0

            ratio_2p = s2_time / base_time if base_time > 0 else 0
            ratio_all = sa_time / base_time if base_time > 0 else 0

            n_sp = entry.get("species", "?")
            print(
                f"{name:<20} {n_sp:>5} {n_params:>4}  "
                f"{base_time * 1000:>10.2f} {s2_time * 1000:>10.2f} "
                f"{sa_time * 1000:>11.2f} "
                f"{ratio_2p:>8.2f}x {ratio_all:>9.2f}x "
                f"{base_rhs:>9} {s2_rhs:>9} {sa_rhs:>9}"
            )

            if ratio_2p > 0 and not math.isnan(ratio_2p):
                geo_2p.append(math.log(ratio_2p))
            if ratio_all > 0 and not math.isnan(ratio_all):
                geo_all.append(math.log(ratio_all))

            results.append(
                {
                    "name": name,
                    "n_species": entry.get("species", 0),
                    "n_params": n_params,
                    "base_time_ms": base_time * 1000,
                    "sens_2p_time_ms": s2_time * 1000,
                    "sens_all_time_ms": sa_time * 1000,
                    "ratio_2p": ratio_2p,
                    "ratio_all": ratio_all,
                    "rhs_base": base_rhs,
                    "rhs_2p": s2_rhs,
                    "rhs_all": sa_rhs,
                }
            )

        except Exception as e:
            print(f"{name:<20}  ERROR: {str(e)[:60]}")

    print()
    print("─" * 100)
    if geo_2p:
        gm_2p = math.exp(sum(geo_2p) / len(geo_2p))
        print(f"  Geometric mean overhead (2 params):     {gm_2p:.2f}x")
    if geo_all:
        gm_all = math.exp(sum(geo_all) / len(geo_all))
        print(f"  Geometric mean overhead (all params):    {gm_all:.2f}x")
    print()
    print("Legend:")
    print("  Sp    = number of species")
    print("  Np    = number of parameters")
    print("  Plain = ODE without sensitivities (baseline)")
    print("  2p    = sensitivity w.r.t. 2 parameters")
    print("  All-p = sensitivity w.r.t. ALL parameters (worst case)")
    print("  Ratio = time with sensitivity / time without")
    print("  RHS   = total RHS function evaluations")
    print()
    print("Note: CVODES forward sensitivity solves Ns extra linear systems per step,")
    print("where Ns = number of sensitivity parameters. The overhead scales ~linearly")
    print("with Ns when using CV_STAGGERED (default).")

    # Save results as JSON
    out_path = BENCH_DIR / "sensitivity_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
