#!/usr/bin/env python
"""A/B benchmark: JAX AD Jacobian vs FD Jacobian (Session 22).

Compares jacobian="jax" vs jacobian="fd" on models with Functional rates,
where the JAX AD Jacobian provides exact derivatives vs CVODE's O(N) FD.

Also tests Elementary models to verify JAX matches analytical Jacobian results.

Usage:
    python _dev/bench_jax_jacobian.py
"""

import os
import time

import numpy as np

ODE_DIR = os.path.join(os.path.dirname(__file__), "ode")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")


def run_comparison(net_path, label, t_end=100.0, n_points=101, n_runs=5):
    """Run FD vs JAX comparison on a single model."""
    from bngsim._bngsim_core import (
        CvodeSimulator,
        NetworkModel,
        SolverOptions,
        TimeSpec,
    )
    from bngsim._jax_rhs import prepare_jax_jacobian

    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_points

    # ── FD baseline ──────────────────────────────────────────────
    fd_times = []
    fd_steps = None
    for _ in range(n_runs):
        m = NetworkModel.from_net(net_path)
        s = CvodeSimulator(m)
        opts = SolverOptions()
        opts.jacobian = "fd"
        t0 = time.perf_counter()
        r = s.run(ts, opts)
        fd_times.append(time.perf_counter() - t0)
        fd_steps = r.solver_stats.n_steps
    fd_species = np.array(r.species_data)

    # ── JAX Jacobian ─────────────────────────────────────────────
    eval_fn, n_sp = prepare_jax_jacobian(net_path)

    jax_times = []
    jax_steps = None
    for _ in range(n_runs):
        m = NetworkModel.from_net(net_path)
        s = CvodeSimulator(m)
        opts = SolverOptions()
        opts.jacobian = "jax"

        # Build param array
        param_names = m.param_names
        param_vals = np.array([m.get_param(n) for n in param_names], dtype=np.float64)

        def _jax_cb(t, y_arr, param_vals=param_vals):
            return eval_fn(np.asarray(y_arr), t, param_vals)

        opts.set_jax_jac_fn(_jax_cb)

        t0 = time.perf_counter()
        r = s.run(ts, opts)
        jax_times.append(time.perf_counter() - t0)
        jax_steps = r.solver_stats.n_steps
    jax_species = np.array(r.species_data)

    # ── Results ──────────────────────────────────────────────────
    fd_med = np.median(fd_times)
    jax_med = np.median(jax_times)
    ratio = jax_med / fd_med if fd_med > 0 else float("inf")
    max_diff = np.max(np.abs(fd_species - jax_species))

    print(f"\n{'=' * 60}")
    print(f"Model: {label}")
    print(f"  Species: {n_sp}, t_end: {t_end}")
    print(f"  FD:  {fd_med * 1000:8.2f} ms  ({fd_steps} steps)")
    print(f"  JAX: {jax_med * 1000:8.2f} ms  ({jax_steps} steps)")
    print(f"  Ratio (JAX/FD): {ratio:.3f}x")
    print(f"  Max species diff: {max_diff:.2e}")
    print(f"  Step count match: {'YES' if fd_steps == jax_steps else 'NO'}")
    print(f"{'=' * 60}")

    return {
        "label": label,
        "n_species": n_sp,
        "fd_ms": fd_med * 1000,
        "jax_ms": jax_med * 1000,
        "ratio": ratio,
        "fd_steps": fd_steps,
        "jax_steps": jax_steps,
        "max_diff": max_diff,
    }


def main():
    print("=" * 60)
    print("JAX AD Jacobian vs FD Jacobian — A/B Benchmark")
    print("=" * 60)

    results = []

    # All models sorted by species count (mix of Functional + Elementary)
    models = [
        # Functional rate models
        (ODE_DIR, "CaM_Ca_interaction_v1.net", "CaM_Ca (3sp FUNC)", 100.0),
        (ODE_DIR, "CaOscillate_functional.net", "CaOscillate (4sp FUNC)", 20.0),
        (ODE_DIR, "wofsy_goldstein.net", "wofsy_goldstein (6sp FUNC)", 100.0),
        (ODE_DIR, "Kholodenko_2000.net", "Kholodenko (8sp FUNC)", 100.0),
        (ODE_DIR, "nfkb.net", "nfkb (22sp FUNC)", 100.0),
        (ODE_DIR, "egfr_net_red.net", "egfr_net_red (40sp FUNC)", 100.0),
        # Elementary models (for size scaling)
        (DATA_DIR, "simple_decay.net", "simple_decay (2sp elem)", 50.0),
        (DATA_DIR, "two_species_reversible.net", "reversible (3sp elem)", 1000.0),
        (ODE_DIR, "Motivating_example.net", "Motivating (78sp elem)", 100.0),
        (ODE_DIR, "SHP2_base_model.net", "SHP2 (149sp elem)", 100.0),
        (ODE_DIR, "metapop_sir_100.net", "metapop (300sp elem)", 100.0),
    ]

    for base_dir, filename, label, t_end in models:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"  SKIP: {filename} not found")
            continue
        try:
            r = run_comparison(path, label, t_end=t_end)
            results.append(r)
        except Exception as e:
            print(f"  FAIL: {label}: {e}")

    # ── Summary ──────────────────────────────────────────────────
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Model':<35} {'FD ms':>8} {'JAX ms':>8} {'Ratio':>7} {'Correct':>8}")
        print("-" * 70)
        for r in results:
            correct = "✓" if r["max_diff"] < 1e-6 else f"{r['max_diff']:.1e}"
            print(
                f"{r['label']:<35} "
                f"{r['fd_ms']:8.2f} "
                f"{r['jax_ms']:8.2f} "
                f"{r['ratio']:7.3f} "
                f"{correct:>8}"
            )


if __name__ == "__main__":
    main()
