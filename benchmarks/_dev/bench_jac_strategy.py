#!/usr/bin/env python3
"""A/B comparison: jacobian="analytical" vs jacobian="fd" using Session 20 flag.

For all-Elementary models, compares wall time and step counts for:
  - jacobian="fd"         — CVODE's internal difference-quotient Jacobian
  - jacobian="analytical" — pre-computed analytical Jacobian (O(nnz))

Also includes Session 15 baselines (dense DQ, pre-analytical-Jac code).
"""

import time

from bngsim._bngsim_core import (
    CvodeSimulator,
    NetworkModel,
    SolverOptions,
    TimeSpec,
)

# (name, net_path, species, rxns, t_end, n_steps, session15_time_s)
MODELS = [
    # Small models (N < 50) — dense solver
    ("simple_decay", "bngsim/tests/data/simple_decay.net", 2, 1, 50, 51, None),
    ("two_species_rev", "bngsim/tests/data/two_species_reversible.net", 3, 2, 1000, 101, None),
    ("LV", "bngsim/benchmarks/models/net/ode/LV.net", 3, 4, 100, 101, 0.0004),
    (
        "CaOscillate_func",
        "bngsim/benchmarks/models/net/ode/CaOscillate_functional.net",
        12,
        8,
        200,
        201,
        0.0010,
    ),
    ("catalysis", "bngsim/benchmarks/models/net/ode/catalysis.net", 22, 204, 50, 101, 0.0007),
    ("egfr_path", "bngsim/benchmarks/models/net/ode/egfr_path.net", 42, 103, 100, 101, 0.0020),
    # Medium models (50 ≤ N < 200)
    (
        "Scaff_22_ground",
        "bngsim/benchmarks/models/net/ode/Scaff_22_ground.net",
        85,
        487,
        1000,
        200,
        0.0099,
    ),
    (
        "SHP2_base_model",
        "bngsim/benchmarks/models/net/ode/SHP2_base_model.net",
        149,
        1032,
        100,
        200,
        0.0107,
    ),
    # Large models (N ≥ 200)
    (
        "metapop_sir_100",
        "bngsim/benchmarks/models/net/ode/metapop_sir_100.net",
        300,
        600,
        100,
        200,
        None,
    ),
    ("fceri_ji", "bngsim/benchmarks/models/net/ode/fceri_ji.net", 354, 3680, 200, 200, 0.1655),
    ("egfr_net", "bngsim/benchmarks/models/net/ode/egfr_net.net", 356, 3749, 120, 120, 0.2523),
    (
        "before_bunching",
        "bngsim/benchmarks/models/net/ode/before_bunching.net",
        593,
        6400,
        100,
        200,
        0.7532,
    ),
    (
        "multisite_phos",
        "bngsim/benchmarks/models/net/ode/multisite_phos.net",
        1026,
        7680,
        100,
        200,
        None,
    ),
    ("fceri_fyn", "bngsim/benchmarks/models/net/ode/fceri_fyn.net", 1281, 15256, 200, 200, 5.9795),
]

N_WARMUP = 1
N_RUNS = 3


def bench_one(net_path, t_end, n_steps, jacobian_strategy):
    """Run ODE with given jacobian strategy, return (median_time, n_steps)."""
    model = NetworkModel.from_net(net_path)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_steps + 1
    opts = SolverOptions()
    opts.max_steps = 100000
    opts.jacobian = jacobian_strategy

    # Warmup
    for _ in range(N_WARMUP):
        model.reset()
        sim = CvodeSimulator(model)
        result = sim.run(ts, opts)

    # Timed runs
    times_list = []
    final_steps = 0
    for _ in range(N_RUNS):
        model.reset()
        sim = CvodeSimulator(model)
        t0 = time.perf_counter()
        result = sim.run(ts, opts)
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
        final_steps = result.solver_stats.n_steps

    median_time = sorted(times_list)[len(times_list) // 2]
    return median_time, final_steps


def main():
    print("=" * 110)
    print("  Analytical Jacobian vs DQ Jacobian — A/B Benchmark (Session 20)")
    print(f"  Protocol: {N_WARMUP} warmup + {N_RUNS} timed runs, median wall time")
    print("=" * 110)
    hdr = (
        f"{'Model':<22} {'Sp':>5} {'Rxn':>6}  "
        f"{'DQ(s)':>9} {'DQ_st':>7}  "
        f"{'AJ(s)':>9} {'AJ_st':>7}  "
        f"{'AJ/DQ':>7} {'S15(s)':>8} {'vs_S15':>7}"
    )
    print(hdr)
    print("-" * 110)

    geo_sum_log = 0.0
    geo_count = 0

    for name, path, n_sp, n_rxn, t_end, n_steps, s15_time in MODELS:
        try:
            # Check if model has all-Elementary reactions
            NetworkModel.from_net(path)
            has_aj = True  # we'll see if "analytical" works

            dq_time, dq_steps = bench_one(path, t_end, n_steps, "fd")

            try:
                aj_time, aj_steps = bench_one(path, t_end, n_steps, "analytical")
            except RuntimeError:
                # Model has Functional/MM rates → no analytical Jac
                has_aj = False
                aj_time, aj_steps = bench_one(path, t_end, n_steps, "auto")

            if has_aj:
                ratio = aj_time / dq_time if dq_time > 0 else float("inf")
                ratio_str = f"{ratio:.2f}x"
            else:
                ratio = 1.0
                ratio_str = "N/A(F)"

            if s15_time is not None and has_aj:
                vs_s15 = f"{s15_time / aj_time:.1f}x"
            elif s15_time is not None:
                vs_s15 = f"{s15_time / dq_time:.1f}x"
            else:
                vs_s15 = "—"

            print(
                f"{name:<22} {n_sp:>5} {n_rxn:>6}  "
                f"{dq_time:>9.4f} {dq_steps:>7}  "
                f"{aj_time:>9.4f} {aj_steps:>7}  "
                f"{ratio_str:>7} {(f'{s15_time:.4f}' if s15_time else '—'):>8} {vs_s15:>7}"
            )

            if has_aj and dq_time > 0:
                import math

                geo_sum_log += math.log(ratio)
                geo_count += 1

        except Exception as e:
            print(f"{name:<22} ERROR: {str(e)[:60]}")

    if geo_count > 0:
        import math

        geo_mean = math.exp(geo_sum_log / geo_count)
        print()
        print(
            f"  Geometric mean AJ/DQ ratio: {geo_mean:.3f}x "
            f"({'faster' if geo_mean < 1 else 'slower'} with analytical Jac)"
        )
        print(f"  ({geo_count} all-Elementary models)")
    print()
    print("Legend:")
    print("  DQ     = CVODE difference-quotient Jacobian (baseline)")
    print("  AJ     = Analytical Jacobian (Session 19/20)")
    print("  AJ/DQ  = ratio (< 1 = AJ faster)")
    print("  S15    = Session 15 baseline (dense DQ, no sparse/analytical)")
    print("  vs_S15 = speedup vs Session 15")
    print("  N/A(F) = model has Functional rates, no analytical Jac available")


if __name__ == "__main__":
    main()
