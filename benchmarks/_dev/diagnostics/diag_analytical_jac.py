#!/usr/bin/env python3
"""Diagnose analytical Jacobian: compare vs finite-difference Jacobian."""

import numpy as np
from bngsim._bngsim_core import CvodeSimulator, NetworkModel, SolverOptions, TimeSpec


def compute_fd_jacobian(model, conc, t=0.0, eps=1e-7):
    """Compute Jacobian via central finite differences."""
    ns = model.n_species
    conc = np.array(conc, dtype=np.float64)
    J = np.zeros((ns, ns))

    # Base RHS
    f0 = np.zeros(ns)
    model.compute_derivs(t, conc, f0)

    for j in range(ns):
        h = eps * max(abs(conc[j]), 1.0)
        conc_p = conc.copy()
        conc_p[j] += h
        fp = np.zeros(ns)
        model.compute_derivs(t, conc_p, fp)
        J[:, j] = (fp - f0) / h

    return J


def main():
    models = [
        ("egfr_net", "bngsim/benchmarks/models/net/ode/egfr_net.net"),
        ("Scaff_22_ground", "bngsim/benchmarks/models/net/ode/Scaff_22_ground.net"),
        ("SHP2_base_model", "bngsim/benchmarks/models/net/ode/SHP2_base_model.net"),
        ("blbr", "bngsim/benchmarks/models/net/ode/blbr.net"),
    ]

    for name, path in models:
        print(f"\n{'=' * 60}")
        print(f"Model: {name}")
        print(f"{'=' * 60}")

        try:
            model = NetworkModel.from_net(path)
        except Exception as e:
            print(f"  Load error: {e}")
            continue

        ns = model.n_species
        nr = model.n_reactions
        nf = model.n_functions
        print(f"  Species: {ns}, Reactions: {nr}, Functions: {nf}")

        # Get initial concentrations
        conc = np.array([s.concentration for s in model.species()])
        print(f"  Conc range: [{conc.min():.3g}, {conc.max():.3g}]")

        # Count fixed species
        fixed = [s for s in model.species() if s.fixed]
        print(f"  Fixed species: {len(fixed)}")

        # Compute FD Jacobian
        J_fd = compute_fd_jacobian(model, conc)

        # Now check: does the model have analytical Jac?
        # We infer from n_functions (not exposed to Python)
        if nf > 0:
            print("  Has functions -> analytical Jac NOT available (Functional model)")
            continue

        print("  Analytical Jac should be available (all Elementary)")

        # Check Jacobian properties
        nnz_fd = np.count_nonzero(np.abs(J_fd) > 1e-15)
        print(
            f"  FD Jacobian nnz: {nnz_fd} / {ns * ns} (density {100.0 * nnz_fd / (ns * ns):.1f}%)"
        )

        # Now simulate with t_end=1.0 and compare step counts
        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = 1.0
        ts.n_points = 11
        opts = SolverOptions()
        opts.max_steps = 100000

        model.reset()
        sim = CvodeSimulator(model)
        try:
            r = sim.run(ts, opts)
            st = r.solver_stats
            print(
                f"  t_end=1: steps={st.n_steps}, rhs={st.n_rhs_evals}, jac_setups={st.n_jac_evals}"
            )
        except Exception as e:
            print(f"  Simulation error: {e}")

        # Check: how many fixed species have nonzero Jac rows?
        for s in model.species():
            if s.fixed:
                row = s.index - 1
                row_max = np.max(np.abs(J_fd[row, :]))
                if row_max > 1e-15:
                    print(
                        f"  WARNING: Fixed species {s.name} "
                        f"(idx {row}) has nonzero FD Jac row "
                        f"(max={row_max:.3g})"
                    )


if __name__ == "__main__":
    main()
