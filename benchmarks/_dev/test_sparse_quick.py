#!/usr/bin/env python3
"""Quick test: verify sparse Jacobian doesn't hang on large models."""

import os
import time

import bngsim

models = [
    ("bngsim/benchmarks/models/net/ssa/multisite_phos.net", 1),
    ("bngsim/benchmarks/models/net/ssa/fceri_gamma.net", 1),
    ("bngsim/benchmarks/models/net/ssa/egfr_net.net", 120),
    ("bngsim/benchmarks/models/net/ssa/prion_aggregation.net", 10),
]

for net, t_end in models:
    if not os.path.exists(net):
        print(f"SKIP: {net}")
        continue
    t0 = time.time()
    m = bngsim.Model.from_net(net)
    load_t = time.time() - t0
    print(
        f"{os.path.basename(net):30s}  {m.n_species:5d} sp  {m.n_reactions:6d} rxn  load={load_t:.2f}s"
    )

    t0 = time.time()
    sim = bngsim.Simulator(m, method="ode")
    r = sim.run(t_span=(0, t_end), n_points=3, max_steps=50000)
    ode_t = time.time() - t0
    print(f"  ODE: {ode_t:.2f}s  steps={r.solver_stats['n_steps']}")

print("\nAll models completed without hanging!")
