#!/usr/bin/env python3
"""Benchmark all ODE .net files with analytical Jacobian (Session 19)."""

import glob
import os
import time

from bngsim._bngsim_core import (
    CvodeSimulator,
    NetworkModel,
    SolverOptions,
    TimeSpec,
)

TIMEOUT = 60  # skip models that take longer
SKIP_LARGE = {"multisite_phos", "fceri_gamma"}  # too slow for quick run


def main():
    net_files = sorted(glob.glob("bngsim/benchmarks/models/net/ode/*.net"))
    print(f"Found {len(net_files)} .net files\n")

    header = f"{'Model':<40} {'Sp':>5} {'Rxn':>6} {'AJ':>3} {'Time(s)':>9} {'Steps':>6} {'RHS':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for net_file in net_files:
        name = os.path.basename(net_file).replace(".net", "")
        if name in SKIP_LARGE:
            print(f"{name:<40} SKIP (too large for quick run)")
            results.append((name, 0, 0, "?", -1, 0, 0))
            continue
        try:
            model = NetworkModel.from_net(net_file)
            ns = model.n_species
            nr = model.n_reactions
            # Infer AJ availability: models with functions block it
            try:
                nf = model.n_functions
            except Exception:
                nf = -1
            aj = "?" if nf < 0 else ("N" if nf > 0 else "Y")

            ts = TimeSpec()
            ts.t_start = 0.0
            ts.t_end = 120.0
            ts.n_points = 121

            opts = SolverOptions()
            opts.max_steps = 50000

            sim = CvodeSimulator(model)
            t0 = time.perf_counter()
            result = sim.run(ts, opts)
            elapsed = time.perf_counter() - t0

            # solver_stats is a property in pybind11
            try:
                stats = result.solver_stats()
            except TypeError:
                stats = result.solver_stats
            n_steps = stats.n_steps
            n_rhs = stats.n_rhs_evals

            print(f"{name:<40} {ns:>5} {nr:>6} {aj:>3} {elapsed:>9.4f} {n_steps:>6} {n_rhs:>8}")
            results.append((name, ns, nr, aj, elapsed, n_steps, n_rhs))

        except Exception as e:
            err_msg = str(e)[:55]
            print(f"{name:<40} ERROR: {err_msg}")
            results.append((name, 0, 0, "?", -1, 0, 0))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    ok = [r for r in results if r[4] >= 0]
    err = [r for r in results if r[4] < 0]
    aj_yes = [r for r in ok if r[3] == "Y"]
    aj_no = [r for r in ok if r[3] == "N"]

    print(f"Total .net files:     {len(results)}")
    print(f"Successful runs:      {len(ok)}")
    print(f"  Analytical Jac:     {len(aj_yes)}")
    print(f"  Colored FD / DQ:    {len(aj_no)}")
    print(f"Errors:               {len(err)}")

    if aj_yes:
        total_time = sum(r[4] for r in aj_yes)
        print(f"\nAnalytical Jac models total time: {total_time:.3f}s")
        largest = max(aj_yes, key=lambda r: r[1])
        print(
            f"Largest model: {largest[0]} ({largest[1]} sp, {largest[2]} rxn) in {largest[4]:.3f}s"
        )


if __name__ == "__main__":
    main()
