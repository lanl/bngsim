#!/usr/bin/env python3
"""Diagnose: is the problem sparse solver or analytical Jac?

Compares step counts for the SAME model using:
  1. Dense solver (force N < SPARSE_THRESHOLD by not using KLU)
  2. Sparse solver + analytical Jac (current default for N >= 50)

If both have high step counts, the issue is the sparse solver/pattern.
If only analytical has high steps, the analytical Jac values are wrong.

Approach: We can't easily switch dense/sparse from Python, but we CAN
check step counts and compare with Session 15 (known dense) baselines.
"""

import time

from bngsim._bngsim_core import (
    CvodeSimulator,
    NetworkModel,
    SolverOptions,
    TimeSpec,
)


def run_model(net_path, t_end, n_steps):
    model = NetworkModel.from_net(net_path)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_steps + 1
    opts = SolverOptions()
    opts.max_steps = 1000000
    sim = CvodeSimulator(model)
    t0 = time.perf_counter()
    r = sim.run(ts, opts)
    elapsed = time.perf_counter() - t0
    st = r.solver_stats
    return elapsed, st.n_steps, st.n_rhs_evals, st.n_jac_evals


# Session 15 baselines (DENSE solver)
# name, path, species, t_end, n_steps, session15_steps, session15_time
MODELS = [
    ("blbr", "bngsim/benchmarks/models/net/ode/blbr.net", 20, 100, 200, None, None),
    ("egfr_path", "bngsim/benchmarks/models/net/ode/egfr_path.net", 18, 120, 120, 600, 0.0008),
    (
        "Scaff_22",
        "bngsim/benchmarks/models/net/ode/Scaff_22_ground.net",
        85,
        1000,
        200,
        557,
        0.0099,
    ),
    ("SHP2", "bngsim/benchmarks/models/net/ode/SHP2_base_model.net", 149, 100, 200, 151, 0.0107),
    ("fceri_ji", "bngsim/benchmarks/models/net/ode/fceri_ji.net", 354, 200, 200, 291, 0.1655),
    ("egfr_net", "bngsim/benchmarks/models/net/ode/egfr_net.net", 356, 120, 120, 448, 0.2523),
]

print("Comparing sparse+analyticalJac vs Session 15 dense baselines")
print("=" * 85)
hdr = (
    f"{'Model':<15} {'Sp':>5} "
    f"{'DenseSteps':>11} {'DenseTime':>10} "
    f"{'SparseSteps':>12} {'SparseTime':>11} "
    f"{'StepRatio':>10}"
)
print(hdr)
print("-" * 85)

for name, path, ns, t_end, n_pts, s15_steps, s15_time in MODELS:
    try:
        elapsed, steps, rhs, jacs = run_model(path, t_end, n_pts)
        if s15_steps:
            ratio = steps / s15_steps
            print(
                f"{name:<15} {ns:>5} "
                f"{s15_steps:>11} {s15_time:>10.4f} "
                f"{steps:>12} {elapsed:>11.4f} "
                f"{ratio:>9.1f}x"
            )
        else:
            print(
                f"{name:<15} {ns:>5} "
                f"{'N/A':>11} {'N/A':>10} "
                f"{steps:>12} {elapsed:>11.4f} "
                f"{'N/A':>10}"
            )
    except Exception as e:
        print(f"{name:<15} ERROR: {str(e)[:50]}")
