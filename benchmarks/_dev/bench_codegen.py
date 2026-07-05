#!/usr/bin/env python3
"""Benchmark: Code-generated RHS vs ExprTk-based RHS (Session 21).

Compares wall time and correctness for ODE simulation using:
  - ExprTk bytecode interpreter (default)
  - Code-generated native C (-O3) via dlopen

The codegen RHS reads parameters from a runtime array, so the .so is
compiled ONCE per model structure and reused across parameter evaluations.
"""

import math
import time

import numpy as np
from bngsim._bngsim_core import (
    CvodeSimulator,
    NetworkModel,
    SolverOptions,
    TimeSpec,
)
from bngsim._codegen import prepare_codegen

MODELS = [
    # (name, net_path, t_end, n_points, n_runs)
    ("simple_decay", "bngsim/tests/data/simple_decay.net", 50, 51, 20),
    ("two_sp_rev", "bngsim/tests/data/two_species_reversible.net", 1000, 101, 20),
    ("LV", "bngsim/benchmarks/models/net/ode/LV.net", 100, 101, 20),
    ("egfr_path", "bngsim/benchmarks/models/net/ode/egfr_path.net", 100, 101, 10),
    ("SHP2", "bngsim/benchmarks/models/net/ode/SHP2_base_model.net", 100, 200, 5),
    ("fceri_ji", "bngsim/benchmarks/models/net/ode/fceri_ji.net", 200, 200, 5),
    ("egfr_net", "bngsim/benchmarks/models/net/ode/egfr_net.net", 120, 120, 5),
]


def bench_one(net_path, t_end, n_points, n_runs, codegen_so_path=None):
    """Benchmark a single model. Returns (median_time_s, n_steps, final_species)."""
    times_list = []
    for _ in range(n_runs):
        model = NetworkModel.from_net(net_path)
        sim = CvodeSimulator(model)
        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = t_end
        ts.n_points = n_points
        opts = SolverOptions()
        opts.max_steps = 100000
        if codegen_so_path:
            opts.codegen_so_path = codegen_so_path

        t0 = time.perf_counter()
        result = sim.run(ts, opts)
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)

    med = sorted(times_list)[len(times_list) // 2]
    sp = np.array(result.species_data)
    return med, result.solver_stats.n_steps, sp


def main():
    print("=" * 80)
    print("  Code-Generated RHS vs ExprTk — Benchmark (Session 21)")
    print("=" * 80)
    print()

    hdr = (
        f"{'Model':<16} {'Sp':>5} {'Rxn':>6}  "
        f"{'ExprTk(ms)':>11} {'Codegen(ms)':>12} {'Ratio':>7} "
        f"{'MaxDiff':>10} {'Steps':>7}"
    )
    print(hdr)
    print("-" * 80)

    geo_sum = 0.0
    geo_n = 0

    for name, path, t_end, n_pts, n_runs in MODELS:
        try:
            model = NetworkModel.from_net(path)
            n_sp = model.n_species
            n_rxn = model.n_reactions

            # Prepare codegen .so
            so_path = str(prepare_codegen(path))

            # Benchmark ExprTk
            et_time, et_steps, et_sp = bench_one(path, t_end, n_pts, n_runs)

            # Benchmark codegen
            cg_time, cg_steps, cg_sp = bench_one(
                path, t_end, n_pts, n_runs, codegen_so_path=so_path
            )

            # Compare correctness
            max_diff = np.max(np.abs(et_sp - cg_sp))

            ratio = cg_time / et_time if et_time > 0 else 0

            print(
                f"{name:<16} {n_sp:>5} {n_rxn:>6}  "
                f"{et_time * 1000:>11.2f} {cg_time * 1000:>12.2f} {ratio:>7.3f}x "
                f"{max_diff:>10.2e} {cg_steps:>7}"
            )

            if ratio > 0:
                geo_sum += math.log(ratio)
                geo_n += 1

        except Exception as e:
            print(f"{name:<16}  ERROR: {str(e)[:55]}")

    if geo_n > 0:
        geo_mean = math.exp(geo_sum / geo_n)
        print()
        print(
            f"  Geometric mean ratio: {geo_mean:.3f}x "
            f"({'faster' if geo_mean < 1 else 'slower'} with codegen)"
        )

    print()
    print("Legend:")
    print("  ExprTk  = ExprTk bytecode interpreter (baseline)")
    print("  Codegen = Native C code (-O3), dlopen'd into CVODE")
    print("  Ratio   = Codegen/ExprTk (< 1 = codegen faster)")
    print("  MaxDiff = max |species_exptk - species_codegen|")


if __name__ == "__main__":
    main()
