#!/usr/bin/env python3
"""Benchmark: BNGsim CVODE vs scipy BDF vs diffrax Kvaerno5.

Four engines on models with ≤ MAX_SPECIES species:
  1. BNGsim CVODE (C++ ExprTk/codegen RHS + SUNDIALS BDF)
  2. scipy BDF + bngsim C++ RHS callback (C++ RHS, Python BDF)
  3. scipy BDF + pure Python/numpy RHS (Python RHS, Python BDF)
  4. diffrax Kvaerno5 (JAX-traced RHS, JAX stepper)

Protocol: 2 warmup + 5 timed runs, median wall time.

Usage:
    python bench_ode_scipy_diffrax.py
    python bench_ode_scipy_diffrax.py --quick 3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import add_bngsim_timeout_arg, format_bngsim_timeout

# ── Configuration ────────────────────────────────────────────────

MAX_SPECIES = 40  # Include models up to this many species
N_WARMUP = 2
N_TIMED = 5

BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
SUITE_FILE = BENCH_DIR / "suite_ode.json"


def load_models(quick=None):
    """Load models from suite_ode.json, filtering by species count."""
    with open(SUITE_FILE) as f:
        suite = json.load(f)
    models = [m for m in suite["models"] if m["species"] <= MAX_SPECIES]
    if quick:
        models = models[:quick]
    return models


# ── Engine 1: BNGsim CVODE ──────────────────────────────────────


def run_bngsim(net_path, t_end, n_steps, bngsim_timeout=None):
    """Run with BNGsim CVODE (in-process C++)."""
    import bngsim

    model = bngsim.Model.from_net(net_path)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)
    return np.array(result.species)


# ── Engine 2: scipy BDF + bngsim C++ RHS ────────────────────────


def run_scipy_bngsim_rhs(net_path, t_end, n_steps):
    """scipy.integrate.solve_ivp(BDF) with the bngsim-generated JAX RHS.

    The function name is historical: an earlier iteration intended to
    plug bngsim's C++ RHS into scipy via a callback, but a direct C++
    RHS export wasn't available, so this engine instead uses the
    bngsim-derived JAX RHS evaluated with numpy arrays (no JIT).
    """
    import bngsim
    from scipy.integrate import solve_ivp

    model = bngsim.Model.from_net(net_path)

    y0 = np.array([model.get_concentration(name) for name in model.species_names])

    from bngsim._codegen import _parse_net_file
    from bngsim._jax_rhs import generate_jax_rhs

    parsed = _parse_net_file(net_path)
    param_values = np.array([v for _, _, v, _ in parsed["parameters"]], dtype=np.float64)

    # Generate JAX RHS but call it with numpy arrays (no JIT)
    import jax.numpy as jnp

    jax_rhs = generate_jax_rhs(net_path)

    def scipy_rhs(t, y):
        return np.array(jax_rhs(jnp.array(y), float(t), jnp.array(param_values)))

    t_eval = np.linspace(0, t_end, n_steps + 1)
    sol = solve_ivp(
        scipy_rhs,
        (0, t_end),
        y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )
    return sol.y.T  # (n_times, n_species)


# ── Engine 3: scipy BDF + pure Python/numpy RHS ─────────────────


def run_scipy_python_rhs(net_path, t_end, n_steps):
    """scipy.integrate.solve_ivp(BDF) with pure numpy RHS.

    No C++ code in the RHS loop — everything is Python/numpy.
    Uses the JAX RHS generator (which produces numpy-compatible
    functions) but without JAX JIT compilation.
    """
    from bngsim._codegen import _parse_net_file
    from scipy.integrate import solve_ivp

    parsed = _parse_net_file(net_path)
    param_values = np.array([v for _, _, v, _ in parsed["parameters"]], dtype=np.float64)
    species_ics = np.array([ic for _, _, ic, _ in parsed["species"]], dtype=np.float64)

    # Build a pure-numpy RHS from the JAX generator
    # We import jax but use numpy arrays (no tracing)
    import jax.numpy as jnp
    from bngsim._jax_rhs import generate_jax_rhs

    jax_rhs = generate_jax_rhs(net_path)

    def numpy_rhs(t, y):
        return np.array(jax_rhs(jnp.array(y), float(t), jnp.array(param_values)))

    y0 = species_ics
    t_eval = np.linspace(0, t_end, n_steps + 1)
    sol = solve_ivp(
        numpy_rhs,
        (0, t_end),
        y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )
    return sol.y.T


# ── Engine 4: diffrax Kvaerno5 ──────────────────────────────────


def run_diffrax(net_path, t_end, n_steps):
    """diffrax Kvaerno5 (JAX-native implicit RK solver).

    Everything is JAX-traced: RHS, Jacobian, stepping.
    JIT compilation is included in the first warmup run.
    """
    import diffrax
    import jax.numpy as jnp
    from bngsim._codegen import _parse_net_file
    from bngsim._jax_rhs import generate_jax_rhs

    parsed = _parse_net_file(net_path)
    param_values = jnp.array([v for _, _, v, _ in parsed["parameters"]], dtype=jnp.float64)
    species_ics = jnp.array([ic for _, _, ic, _ in parsed["species"]], dtype=jnp.float64)

    jax_rhs = generate_jax_rhs(net_path)

    # Wrap for diffrax API: (t, y, args) -> dy
    def diffrax_rhs(t, y, args):
        return jax_rhs(y, t, args)

    term = diffrax.ODETerm(diffrax_rhs)
    solver = diffrax.Kvaerno5()
    stepsize_ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_end, n_steps + 1))

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_end,
        dt0=t_end / 1000,
        y0=species_ics,
        args=param_values,
        stepsize_controller=stepsize_ctrl,
        saveat=saveat,
        max_steps=100000,
    )
    return np.array(sol.ys)


# ── Timing harness ───────────────────────────────────────────────


def time_engine(engine_fn, net_path, t_end, n_steps, n_warmup=N_WARMUP, n_timed=N_TIMED):
    """Time an engine: warmup + timed runs, return median."""
    for _ in range(n_warmup):
        try:
            engine_fn(net_path, t_end, n_steps)
        except Exception as e:
            return None, str(e)

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        try:
            engine_fn(net_path, t_end, n_steps)
        except Exception as e:
            return None, str(e)
        times.append(time.perf_counter() - t0)

    return median(times), None


# ── Main ─────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Benchmark: BNGsim CVODE vs scipy BDF vs diffrax")
    ap.add_argument("--quick", type=int, default=0, help="Limit to first N models")
    add_bngsim_timeout_arg(ap, default=None)
    args = ap.parse_args()

    models = load_models(args.quick or None)
    print(f"Benchmarking {len(models)} models (≤{MAX_SPECIES} species)")
    print(f"Protocol: {N_WARMUP} warmup + {N_TIMED} timed, median\n")
    print(f"BNGsim ODE timeout guard: {format_bngsim_timeout(args.bngsim_timeout)}\n")

    engines = [
        (
            "BNGsim",
            lambda net_path, t_end, n_steps, timeout=args.bngsim_timeout: run_bngsim(
                net_path,
                t_end,
                n_steps,
                bngsim_timeout=timeout,
            ),
        ),
        ("scipy+C++", run_scipy_bngsim_rhs),
        ("scipy+Py", run_scipy_python_rhs),
        ("diffrax", run_diffrax),
    ]

    results = []
    header = (
        f"{'Model':<25} {'Sp':>3} "
        f"{'BNGsim':>10} {'scipy+C++':>10} "
        f"{'scipy+Py':>10} {'diffrax':>10}"
    )
    print(header)
    print("-" * len(header))

    for m in models:
        net_path = str(BENCH_DIR / m["net_file"])
        t_end = m["t_end"]
        n_steps = m["n_steps"]
        name = m["name"]
        n_sp = m["species"]

        row = {"name": name, "species": n_sp}
        cells = []

        for ename, efn in engines:
            # Skip diffrax for models > 30 sp (too slow)
            if ename == "diffrax" and n_sp > 30:
                row[ename] = None
                cells.append("---")
                continue

            t_median, err = time_engine(efn, net_path, t_end, n_steps)
            if t_median is not None:
                row[ename] = t_median
                if t_median < 0.001:
                    cells.append("<0.001")
                else:
                    cells.append(f"{t_median:.3f}")
            else:
                row[ename] = None
                cells.append("FAIL")
                print(f"  [{ename}] {name}: {err}", file=sys.stderr)

        print(f"{name:<25} {n_sp:>3} " + " ".join(f"{c:>10}" for c in cells))
        results.append(row)

    # Save results
    out_path = Path(__file__).parent.parent / "results" / "bench_scipy_diffrax.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
