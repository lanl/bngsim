#!/usr/bin/env python3
"""Quick benchmark: verify graph coloring speedup on egfr_net (356 sp).

Before coloring (Session 17): egfr_net ODE took 12.6s
Expected with coloring: <1s
"""

import os
import time

import bngsim

net = "bngsim/benchmarks/models/net/ssa/egfr_net.net"
if not os.path.exists(net):
    print(f"SKIP: {net} not found")
    exit(1)

m = bngsim.Model.from_net(net)
print(f"egfr_net: {m.n_species} sp, {m.n_reactions} rxn")

t0 = time.time()
sim = bngsim.Simulator(m, method="ode")
r = sim.run(t_span=(0, 120), n_points=11, max_steps=50000)
elapsed = time.time() - t0
steps = r.solver_stats["n_steps"]
rhs = r.solver_stats["n_rhs_evals"]
print(f"  ODE: {elapsed:.2f}s  steps={steps}  rhs={rhs}")
print("  (Before coloring: 12.6s)")

# Also test prion_aggregation (104 sp)
net2 = "bngsim/benchmarks/models/net/ssa/prion_aggregation.net"
if os.path.exists(net2):
    m2 = bngsim.Model.from_net(net2)
    print(f"\nprion: {m2.n_species} sp, {m2.n_reactions} rxn")
    t0 = time.time()
    sim2 = bngsim.Simulator(m2, method="ode")
    r2 = sim2.run(t_span=(0, 10), n_points=11, max_steps=50000)
    elapsed2 = time.time() - t0
    print(f"  ODE: {elapsed2:.2f}s  steps={r2.solver_stats['n_steps']}")

print("\nDone!")
