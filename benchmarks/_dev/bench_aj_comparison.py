#!/usr/bin/env python3
"""A/B comparison: Session 15 (dense DQ Jac) vs Session 19 (analytical Jac + KLU).

For all-Elementary models with N >= 50 species, compares:
  - "Before" = Session 15 dense Jacobian baselines (from ODE_BENCHMARK_RESULTS.md)
  - "After"  = Current BNGsim with analytical Jacobian + KLU sparse solver
"""

import time

from bngsim._bngsim_core import (
    CvodeSimulator,
    NetworkModel,
    SolverOptions,
    TimeSpec,
)

# Session 15 baselines: (name, net_path, species, rxns, t_end, n_steps, session15_time_s)
# These used dense Jacobian (LAPACK LU), BDF+Newton, rtol=atol=1e-8
MODELS = [
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
        "Motivating_example",
        "bngsim/benchmarks/models/net/ode/Motivating_example.net",
        78,
        354,
        100,
        200,
        None,
    ),  # not in Session 15
    (
        "SHP2_base_model",
        "bngsim/benchmarks/models/net/ode/SHP2_base_model.net",
        149,
        1032,
        100,
        200,
        0.0107,
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
    ("fceri_fyn", "bngsim/benchmarks/models/net/ode/fceri_fyn.net", 1281, 15256, 200, 200, 5.9795),
    (
        "metapop_sir_100",
        "bngsim/benchmarks/models/net/ode/metapop_sir_100.net",
        300,
        600,
        100,
        200,
        None,
    ),  # new model, no Session 15 baseline
]


def bench(net_path, t_end, n_steps, n_runs=3):
    """Run ODE n_runs times, return median wall time."""
    model = NetworkModel.from_net(net_path)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_steps + 1
    opts = SolverOptions()
    opts.max_steps = 100000

    times_list = []
    for _ in range(n_runs):
        model.reset()
        sim = CvodeSimulator(model)
        t0 = time.perf_counter()
        sim.run(ts, opts)
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)

    return sorted(times_list)[len(times_list) // 2]


def main():
    print("=" * 90)
    print("  Analytical Jacobian + KLU vs Dense DQ Jacobian (Session 15 baseline)")
    print("=" * 90)
    hdr = f"{'Model':<25} {'Sp':>5} {'Rxn':>6} {'Dense(s)':>9} {'AnaJac(s)':>10} {'Speedup':>8}"
    print(hdr)
    print("-" * len(hdr))

    for name, path, n_sp, n_rxn, t_end, n_steps, s15_time in MODELS:
        try:
            aj_time = bench(path, t_end, n_steps)
            if s15_time is not None:
                speedup = s15_time / aj_time
                print(
                    f"{name:<25} {n_sp:>5} {n_rxn:>6} "
                    f"{s15_time:>9.4f} {aj_time:>10.4f} "
                    f"{speedup:>7.1f}x"
                )
            else:
                print(f"{name:<25} {n_sp:>5} {n_rxn:>6} {'N/A':>9} {aj_time:>10.4f} {'N/A':>8}")
        except Exception as e:
            print(f"{name:<25} ERROR: {str(e)[:50]}")

    print()
    print("Notes:")
    print("  Dense = Session 15 baseline (dense DQ Jac, LAPACK LU)")
    print("  AnaJac = analytical Jac + KLU sparse (Session 19)")
    print("  Same t_end, n_steps, rtol=atol=1e-8 for both")


if __name__ == "__main__":
    main()
