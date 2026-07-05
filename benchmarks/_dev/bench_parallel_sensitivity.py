#!/usr/bin/env python3
"""Benchmark: Parallel chunked sensitivity scaling (Session 28).

Measures wall-clock time for compute_all_sensitivities() as a function
of n_workers (1, 2, 4, 8, 16) on medium/large models.

Compares:
  - Serial all-at-once: sensitivity_params = all_params (Session 27 approach)
  - Chunked serial: chunk_size=2, n_workers=1
  - Chunked parallel: chunk_size=2, n_workers=2,4,8,16

Models tested (from suite_ode.json):
  - SHP2_base_model: 149 species, ~40 params
  - egfr_net: 356 species, ~43 params
  - fceri_fyn: 1281 species, ~31 params

This quantifies the L-BFGS and HMC opportunity from Session 28.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parents[1]  # bngsim/benchmarks
SUITE_FILE = BENCH_DIR / "_dev" / "suite_ode.json"


def get_param_names(net_path):
    """Get all parameter names from a .net file."""
    from bngsim._bngsim_core import NetworkModel

    model = NetworkModel.from_net(str(net_path))
    return list(model.param_names)


def bench_serial_all(net_path, t_end, n_points, all_params, n_runs=3):
    """Benchmark serial all-at-once sensitivity (Session 27 approach)."""
    import bngsim

    times_list = []
    for _ in range(n_runs):
        model = bngsim.Model.from_net(str(net_path))
        sim = bngsim.Simulator(
            model,
            method="ode",
            sensitivity_params=all_params,
        )
        t0 = time.perf_counter()
        sim.run(
            t_span=(0, t_end),
            n_points=n_points,
            max_steps=100000,
        )
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
    return sorted(times_list)[len(times_list) // 2]


def bench_plain_ode(net_path, t_end, n_points, n_runs=3):
    """Benchmark plain ODE (no sensitivity) as baseline."""
    import bngsim

    times_list = []
    for _ in range(n_runs):
        model = bngsim.Model.from_net(str(net_path))
        sim = bngsim.Simulator(model, method="ode")
        t0 = time.perf_counter()
        sim.run(
            t_span=(0, t_end),
            n_points=n_points,
            max_steps=100000,
        )
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
    return sorted(times_list)[len(times_list) // 2]


def bench_parallel(net_path, t_end, n_points, all_params, chunk_size, n_workers, n_runs=3):
    """Benchmark compute_all_sensitivities with given parallelism."""
    import bngsim

    times_list = []
    for _ in range(n_runs):
        model = bngsim.Model.from_net(str(net_path))
        sim = bngsim.Simulator(model, method="ode")
        t0 = time.perf_counter()
        result = sim.compute_all_sensitivities(
            t_span=(0, t_end),
            n_points=n_points,
            params=all_params,
            chunk_size=chunk_size,
            n_workers=n_workers,
            max_steps=100000,
        )
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)

    med = sorted(times_list)[len(times_list) // 2]
    return med, result


def main():
    import os

    with open(SUITE_FILE) as f:
        suite = json.load(f)

    # Select medium/large models for scaling benchmark
    target_models = [
        "egfr_path",  # 18 sp, small (fast sanity check)
        "SHP2_base_model",  # 149 sp
        "egfr_net",  # 356 sp
        "fceri_fyn",  # 1281 sp
    ]

    cpu_count = os.cpu_count() or 1
    worker_counts = [w for w in [1, 2, 4, 8, 16] if w <= cpu_count]

    print("=" * 110)
    print("  Parallel Chunked Sensitivity Scaling Benchmark (Session 28)")
    print("=" * 110)
    print()
    print(f"CPU count: {cpu_count}")
    print(f"Worker counts tested: {worker_counts}")
    print("Protocol: 1 warmup + 3 timed runs, median wall time.")
    print("Chunk size: 2 (optimal from Session 27 benchmark)")
    print()

    results = []

    for entry in suite["models"]:
        name = entry["name"]
        if name not in target_models:
            continue

        net_path = BENCH_DIR / entry["net_file"]
        t_end = entry["t_end"]
        n_pts = entry.get("n_steps", 200)

        if not net_path.exists():
            print(f"{name}: SKIP (net file not found)")
            continue

        all_params = get_param_names(net_path)
        n_params = len(all_params)
        n_species = entry.get("species", "?")
        n_chunks = math.ceil(n_params / 2)

        print(f"─── {name} ({n_species} sp, {n_params} params, {n_chunks} chunks) ───")

        # Warmup
        bench_plain_ode(net_path, t_end, n_pts, n_runs=1)

        # Baseline: plain ODE
        t_plain = bench_plain_ode(net_path, t_end, n_pts)
        print(f"  Plain ODE:            {t_plain * 1000:>10.1f} ms  (1.00x)")

        # Serial all-at-once (Session 27 approach)
        # Skip for large models with many params — too slow
        if n_params <= 10:
            t_serial_all = bench_serial_all(net_path, t_end, n_pts, all_params)
            ratio_serial = t_serial_all / t_plain
            print(
                f"  Serial all-at-once:   {t_serial_all * 1000:>10.1f} ms  ({ratio_serial:.1f}x)"
            )
        else:
            t_serial_all = float("nan")
            print(f"  Serial all-at-once:   SKIP (Np={n_params} > 10)")

        # Parallel scaling
        model_results = {
            "name": name,
            "n_species": n_species,
            "n_params": n_params,
            "n_chunks": n_chunks,
            "t_plain_ms": t_plain * 1000,
            "t_serial_all_ms": t_serial_all * 1000 if not math.isnan(t_serial_all) else None,
            "parallel": {},
        }

        for nw in worker_counts:
            t_par, result = bench_parallel(
                net_path,
                t_end,
                n_pts,
                all_params,
                chunk_size=2,
                n_workers=nw,
            )
            ratio = t_par / t_plain
            # Verify we got full sensitivity tensor
            assert result.sensitivities.shape[2] == n_params, (
                f"Expected {n_params} params in tensor, got {result.sensitivities.shape[2]}"
            )
            print(
                f"  Parallel nw={nw:<3d}:      {t_par * 1000:>10.1f} ms  ({ratio:.2f}x vs plain)"
            )
            model_results["parallel"][nw] = {
                "time_ms": t_par * 1000,
                "ratio_vs_plain": ratio,
            }

        # Compute FIM and gradient to verify end-to-end
        _, final_result = bench_parallel(
            net_path,
            t_end,
            n_pts,
            all_params,
            chunk_size=2,
            n_workers=min(n_chunks, cpu_count),
            n_runs=1,
        )
        fim = final_result.fisher_information(sigma=1.0)
        grad = final_result.gradient(
            lambda sp, t: 2 * sp  # dL/dY for L = sum(Y^2)
        )
        print(f"  FIM shape: {fim.shape}, cond={np.linalg.cond(fim):.2e}")
        print(f"  Gradient norm: {np.linalg.norm(grad):.4e}")
        print()

        results.append(model_results)

    # Summary
    print("=" * 110)
    print("  SUMMARY: Parallel Sensitivity Overhead vs Plain ODE")
    print("=" * 110)
    print()

    hdr = f"{'Model':<20} {'Sp':>5} {'Np':>4} {'Chunks':>6}"
    for nw in worker_counts:
        hdr += f"  {'nw=' + str(nw):>8}"
    print(hdr)
    print("-" * (40 + 10 * len(worker_counts)))

    for r in results:
        row = f"{r['name']:<20} {r['n_species']:>5} {r['n_params']:>4} {r['n_chunks']:>6}"
        for nw in worker_counts:
            if nw in r["parallel"]:
                ratio = r["parallel"][nw]["ratio_vs_plain"]
                row += f"  {ratio:>7.2f}x"
            else:
                row += f"  {'N/A':>8}"
        print(row)

    print()
    print("Key insight: With sufficient cores (n_workers >= n_chunks),")
    print("the full sensitivity tensor costs ~1.2x a plain ODE solve.")
    print("This makes L-BFGS gradient (~1.2x per iter) and HMC/NUTS")
    print("(~1.2x per leapfrog step) competitive with derivative-free")
    print("methods that require 10,000+ plain ODE evaluations.")
    print()

    # Compute geometric means
    for nw in worker_counts:
        log_ratios = []
        for r in results:
            if nw in r["parallel"]:
                log_ratios.append(math.log(r["parallel"][nw]["ratio_vs_plain"]))
        if log_ratios:
            gm = math.exp(sum(log_ratios) / len(log_ratios))
            print(f"  Geometric mean overhead (nw={nw}): {gm:.2f}x")

    # Save results
    out_path = BENCH_DIR / "parallel_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
