#!/usr/bin/env python3
"""Worked example: driving bngsim as a reaction kernel (GH #102 Stage 0).

This is the documented direct-call example for :class:`bngsim.ReactionKernel`
— the framework-agnostic way an external orchestrator (a hand-rolled hybrid
SSA/ODE splitting loop, or anything else) drives a bngsim network per step. It
is a runnable tutorial, not a benchmark; for timing see ``warm_path_bench.py``
and ``mvp_kernel_demo.py``.

Three parts, each prints what it demonstrates:

1. **The acceptance invariant.** With no external coupling, advancing a model
   step-wise through the kernel reproduces a single standalone ``Simulator.run``
   over the same horizon, to integrator tolerance. This is the contract that
   lets an orchestrator trust the kernel: stepping changes *when* you can inject
   state, never the dynamics.

2. **The per-step drive loop.** The canonical pattern an orchestrator uses:

       state = kernel.get_state()      # pull the coupling species out
       state[idx] += influx            # the orchestrator's contribution
       kernel.set_state(state)         # inject it back
       kernel.advance(dt)              # integrate the subset by one step

   Here a toy "upstream source" injects a fixed amount into the first species
   every coupling step — standing in for the other half of a hybrid split
   feeding this (ODE) half through the shared state vector.

3. **Observables.** Reading derived quantities (here a conserved total) at each
   coupling step, the way an orchestrator would monitor or couple on them.

Run:
    python bngsim/benchmarks/kernel/kernel_example.py
"""

from __future__ import annotations

import bngsim
import numpy as np
from bngsim._bngsim_core import ModelBuilder


def build_demo_model() -> bngsim.Model:
    """A small linear transduction cascade S0 -> S1 -> S2 -> S3 -> S4.

    species[0] starts with all the mass; each step transfers it downstream at a
    distinct rate. A ``Total`` observable sums all species (conserved under this
    pure-transfer chain), and ``Downstream`` sums the last three — the kind of
    coarse readout an orchestrator might couple on.
    """
    n = 5
    b = ModelBuilder()
    sp = []
    for i in range(n):
        b.add_parameter(f"k{i}", 0.2 * (i + 1))
        sp.append(b.add_species(f"S{i}", 100.0 if i == 0 else 0.0))
    for i in range(n - 1):
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    b.add_observable("Total", [(i, 1.0) for i in range(n)])
    b.add_observable("Downstream", [(i, 1.0) for i in range(2, n)])
    b.set_compute_conservation_laws(False)
    return bngsim.Model(_core=b.build())


def part1_acceptance_invariant() -> float:
    print("\n== 1. Acceptance: step-wise kernel == standalone run ==")
    horizon, n_steps = 40.0, 8
    dt = horizon / n_steps

    # Standalone reference: one run over the whole horizon.
    standalone = bngsim.Simulator(build_demo_model())
    ref_final = standalone.run(t_span=(0.0, horizon), n_points=2).species[-1]

    # The same model, advanced step-wise through the kernel with no coupling.
    kernel = bngsim.ReactionKernel(build_demo_model())
    for _ in range(n_steps):
        kernel.advance(dt)

    rel_err = float(np.max(np.abs(kernel.get_state() - ref_final))) / float(
        np.max(np.abs(ref_final)) or 1.0
    )
    print(f"   horizon={horizon}, {n_steps} kernel steps of dt={dt}")
    print(f"   max relative error vs standalone run: {rel_err:.2e}")
    print(f"   -> {'PASS' if rel_err < 1e-6 else 'FAIL'}  (stepping does not change the dynamics)")
    return rel_err


def part2_per_step_drive_loop() -> None:
    print("\n== 2. Per-step drive loop with external coupling ==")
    kernel = bngsim.ReactionKernel(build_demo_model())
    dt, n_steps, influx = 2.0, 10, 20.0
    s0 = kernel.state_names.index("S0")

    print(f"   injecting {influx} into S0 every dt={dt} (a toy upstream source)")
    print(f"   {'t':>6} {'S0':>9} {'Total':>9}")
    for _ in range(n_steps):
        state = kernel.get_state()  # pull the coupling species out
        state[s0] += influx  # the orchestrator's contribution this step
        kernel.set_state(state)  # inject it back
        kernel.advance(dt)  # integrate one coupling step
        total = kernel.observables()[kernel.observable_names.index("Total")]
        print(f"   {kernel.time:>6.1f} {kernel.get_state()[s0]:>9.3f} {total:>9.3f}")

    injected = influx * n_steps
    print(f"   total injected over the run: {injected:.1f}  (Total tracks accumulated mass)")


def part3_observables_readout() -> None:
    print("\n== 3. Observables at each coupling step ==")
    kernel = bngsim.ReactionKernel(build_demo_model())
    dt, n_steps = 5.0, 5
    names = kernel.observable_names
    print(f"   observables: {names}")
    print(f"   {'t':>6} " + " ".join(f"{n:>11}" for n in names))
    # Pre-advance values come from the side-effect-free clone probe.
    print(f"   {kernel.time:>6.1f} " + " ".join(f"{v:>11.4f}" for v in kernel.observables()))
    for _ in range(n_steps):
        kernel.advance(dt)
        print(f"   {kernel.time:>6.1f} " + " ".join(f"{v:>11.4f}" for v in kernel.observables()))


def main() -> None:
    print("bngsim reaction-kernel worked example (GH #102 Stage 0)")
    print(f"  version: {bngsim.__version__}")
    part1_acceptance_invariant()
    part2_per_step_drive_loop()
    part3_observables_readout()
    print(
        "\nThe kernel is method-agnostic: pass method='ssa' or 'psa' to ReactionKernel for a\n"
        "stochastic subset. For a Vivarium composite, wrap it with\n"
        "bngsim.vivarium.BngsimProcess (optional extra: pip install 'bngsim[vivarium]')."
    )


if __name__ == "__main__":
    main()
