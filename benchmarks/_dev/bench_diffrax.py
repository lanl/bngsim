#!/usr/bin/env python
"""A/B benchmark: Diffrax (pure JAX) vs CVODE (Session 22b).

Compares method="diffrax" (Kvaerno5, JIT-compiled, AD Jacobian)
vs CVODE (BDF, C++, FD or analytical Jacobian).

Usage:
    python _dev/bench_diffrax.py
"""

import os
import time

import numpy as np

ODE_DIR = os.path.join(os.path.dirname(__file__), "ode")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")


def run_cvode(net_path, t_end, n_points, n_runs=5):
    """Run CVODE baseline (auto Jacobian)."""
    from bngsim._bngsim_core import (
        CvodeSimulator,
        NetworkModel,
        SolverOptions,
        TimeSpec,
    )

    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_points

    times_list = []
    for _ in range(n_runs):
        m = NetworkModel.from_net(net_path)
        s = CvodeSimulator(m)
        opts = SolverOptions()
        t0 = time.perf_counter()
        r = s.run(ts, opts)
        times_list.append(time.perf_counter() - t0)
    species = np.array(r.species_data)
    steps = r.solver_stats.n_steps
    return np.median(times_list), species, steps


def run_diffrax_bench(net_path, t_end, n_points, n_runs=5):
    """Run Diffrax (pure JAX)."""
    from bngsim._codegen import _parse_net_file
    from bngsim._diffrax_solver import run_diffrax

    model = _parse_net_file(net_path)

    # Build param dict with evaluated values
    # We need to evaluate expressions in order, with math funcs
    import math

    math_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    math_ns["__builtins__"] = {}
    param_dict = {}
    for _, name, expr, _ in model["parameters"]:
        try:
            val = float(expr)
        except (ValueError, TypeError):
            try:
                ns = {**math_ns, **param_dict}
                val = eval(expr, ns)  # noqa: S307
            except Exception:
                val = 0.0
        param_dict[name] = val

    # Warmup (JIT compilation)
    print("    JIT compiling...", end="", flush=True)
    t_jit_start = time.perf_counter()
    r = run_diffrax(
        net_path,
        param_dict,
        t_start=0.0,
        t_end=t_end,
        n_points=n_points,
    )
    jit_time = time.perf_counter() - t_jit_start
    print(f" {jit_time:.1f}s")

    # Timed runs (post-JIT)
    times_list = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        r = run_diffrax(
            net_path,
            param_dict,
            t_start=0.0,
            t_end=t_end,
            n_points=n_points,
        )
        times_list.append(time.perf_counter() - t0)

    species = r["species"]
    steps = r["n_steps"]
    return (np.median(times_list), species, steps, jit_time)


def compare(net_path, label, t_end=100.0, n_points=101):
    """Compare CVODE vs Diffrax on one model."""
    print(f"\n--- {label} ---")

    # CVODE
    cvode_t, cvode_sp, cvode_steps = run_cvode(net_path, t_end, n_points)

    # Diffrax
    dfx_t, dfx_sp, dfx_steps, jit_t = run_diffrax_bench(net_path, t_end, n_points)

    # Compare species (may differ in shape if
    # observables vs species)
    n_sp = min(cvode_sp.shape[1], dfx_sp.shape[1])
    n_t = min(cvode_sp.shape[0], dfx_sp.shape[0])
    max_diff = np.max(np.abs(cvode_sp[:n_t, :n_sp] - dfx_sp[:n_t, :n_sp]))
    ratio = dfx_t / cvode_t if cvode_t > 0 else float("inf")

    print(f"  CVODE:   {cvode_t * 1000:8.2f} ms ({cvode_steps} steps)")
    print(f"  Diffrax: {dfx_t * 1000:8.2f} ms ({dfx_steps} steps) [JIT: {jit_t:.1f}s]")
    print(f"  Ratio (Diffrax/CVODE): {ratio:.3f}x")
    print(f"  Max species diff: {max_diff:.2e}")

    return {
        "label": label,
        "n_species": n_sp,
        "cvode_ms": cvode_t * 1000,
        "diffrax_ms": dfx_t * 1000,
        "ratio": ratio,
        "jit_s": jit_t,
        "max_diff": max_diff,
        "cvode_steps": cvode_steps,
        "dfx_steps": dfx_steps,
    }


def main():
    print("=" * 60)
    print("Diffrax vs CVODE — A/B Benchmark")
    print("=" * 60)

    results = []

    models = [
        (DATA_DIR, "simple_decay.net", "simple_decay (2sp)", 50.0),
        (ODE_DIR, "Kholodenko_2000.net", "Kholodenko (8sp FUNC)", 100.0),
        (ODE_DIR, "nfkb.net", "nfkb (22sp FUNC)", 100.0),
        (ODE_DIR, "egfr_net_red.net", "egfr_net_red (40sp FUNC)", 100.0),
        (ODE_DIR, "Motivating_example.net", "Motivating (78sp)", 100.0),
        (ODE_DIR, "SHP2_base_model.net", "SHP2 (149sp)", 100.0),
        (ODE_DIR, "metapop_sir_100.net", "metapop (300sp)", 100.0),
    ]

    for base, fname, label, t_end in models:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {fname}")
            continue
        try:
            r = compare(path, label, t_end=t_end)
            results.append(r)
        except Exception as e:
            print(f"  FAIL: {label}: {e}")

    # Summary
    if results:
        print(f"\n{'=' * 65}")
        print("SUMMARY (post-JIT times)")
        print(f"{'=' * 65}")
        hdr = f"{'Model':<28} {'CVODE':>8} {'Diffrax':>8} {'Ratio':>7} {'JIT':>6} {'Correct':>8}"
        print(hdr)
        print("-" * 65)
        for r in results:
            ok = "✓" if r["max_diff"] < 1e-3 else f"{r['max_diff']:.1e}"
            print(
                f"{r['label']:<28} "
                f"{r['cvode_ms']:8.2f} "
                f"{r['diffrax_ms']:8.2f} "
                f"{r['ratio']:7.3f} "
                f"{r['jit_s']:5.1f}s "
                f"{ok:>8}"
            )


if __name__ == "__main__":
    main()
