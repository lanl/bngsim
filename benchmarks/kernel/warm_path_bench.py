#!/usr/bin/env python3
"""Warm CVODE hot-path microbenchmark (GH #102 Stage 0).

Quantifies the win from the warm simulator path: ``CvodeSimulator::run()`` reused
across per-step ``run_until`` calls re-enters CVODE via ``CVodeReInit`` instead of
rebuilding all SUNDIALS state (SUNContext, N_Vector, CVODE memory, and — most
expensively — the KLU sparse solver with a fresh symbolic factorization) every
call. The reused linear solver keeps its symbolic factorization, so a warm step
pays only a cheap numeric refactor.

The warm path is transparent: every ODE ``run``/``run_until`` on a model with no
events, no forward sensitivities, and a non-JAX Jacobian takes it automatically.
This bench measures it by toggling the ``BNGSIM_NO_WARM_CVODE`` escape hatch,
which forces the old cold rebuild path, and comparing per-step wall time on the
*same* model.

What it shows:

1. **Parity** — warm and cold produce byte-identical trajectories (the warm path
   must change only performance, never results).
2. **Per-step win vs N** — the saved time per step is the fixed re-init cost the
   warm path eliminates; it grows super-linearly with N (the KLU symbolic
   factorization being skipped), reaching several ms/step at ~100K.
3. **Win vs coupling dt** — that saved time is fixed per call, so as the coupling
   step shrinks (more, smaller steps over a horizon) it becomes a larger
   fraction of each step and the speedup ratio grows. This is exactly the
   hybrid-splitting regime the issue targets: many small coupling steps.

Run:
    python bngsim/benchmarks/kernel/warm_path_bench.py            # default sweep
    python bngsim/benchmarks/kernel/warm_path_bench.py --quick    # fast smoke
    python bngsim/benchmarks/kernel/warm_path_bench.py \
        --sizes 10000 50000 100000 --steps 200 --t-end 1.0

Synthetic model: a random linear (first-order) network — N species ≈ N reactions,
~2 nonzeros per Jacobian column, log-uniform rates over ~3 decades. The same
class the MVP demo uses; trivial to generate reproducibly and representative of
the ~100K-first-order target. KLU is required to reach 100K (sparse solve); the
bench reports whether it is present.
"""

from __future__ import annotations

import argparse
import os
import time

import bngsim
import numpy as np
from bngsim._bngsim_core import ModelBuilder

_WARM_OFF = "BNGSIM_NO_WARM_CVODE"


def build_random_linear_network(n: int, *, seed: int = 0):
    """Synthetic random first-order network of ``n`` species (conservation off)."""
    rng = np.random.default_rng(seed)
    b = ModelBuilder()
    rates = 10.0 ** rng.uniform(-2.0, 1.0, size=n)
    sp = [b.add_species(f"S{i}", 100.0 if i % 50 == 0 else 0.0) for i in range(n)]
    for i in range(n):
        b.add_parameter(f"k{i}", float(rates[i]))
    targets = rng.integers(0, n, size=n)
    for i in range(n):
        j = int(targets[i])
        if j == i:
            j = (i + 1) % n
        b.add_reaction([sp[i]], [sp[j]], "elementary", f"k{i}")
    b.set_compute_conservation_laws(False)
    return b.build()


def has_klu() -> bool:
    probe = ModelBuilder()
    probe.add_parameter("k", 1.0)
    a = probe.add_species("A", 1.0)
    c = probe.add_species("B", 0.0)
    probe.add_reaction([a], [c], "elementary", "k")
    return bool(probe.build().codegen_jacobian_plan()["has_klu"])


def _set_warm(enabled: bool) -> None:
    """Toggle the warm path for subsequent run() calls in this process."""
    if enabled:
        os.environ.pop(_WARM_OFF, None)
    else:
        os.environ[_WARM_OFF] = "1"


def _final_state(n: int, n_steps: int, t_end: float, *, warm: bool, seed: int = 0) -> np.ndarray:
    _set_warm(warm)
    try:
        sim = bngsim.Simulator(bngsim.Model(_core=build_random_linear_network(n, seed=seed)))
        dt = t_end / n_steps
        for s in range(n_steps):
            sim.run_until(t=(s + 1) * dt, n_points=2)
        return sim.get_state()
    finally:
        _set_warm(True)


def _per_step_ms(n: int, n_steps: int, t_end: float, *, warm: bool, seed: int = 0) -> float:
    """Mean per-step wall time (ms), excluding step 0 (one-time first build)."""
    _set_warm(warm)
    try:
        sim = bngsim.Simulator(bngsim.Model(_core=build_random_linear_network(n, seed=seed)))
        dt = t_end / n_steps
        sim.run_until(t=dt, n_points=2)  # warm-up: first call builds either way
        t0 = time.perf_counter()
        for s in range(1, n_steps):
            sim.run_until(t=(s + 1) * dt, n_points=2)
        elapsed = time.perf_counter() - t0
        return elapsed / max(1, n_steps - 1) * 1e3
    finally:
        _set_warm(True)


def parity(sizes: list[int], n_steps: int, t_end: float, seed: int) -> bool:
    print("\n== 1. Parity: warm trajectory == cold trajectory (must be identical) ==")
    print(f"{'N':>8} {'max|Δ|':>12} {'result':>10}")
    ok = True
    for n in sizes:
        warm = _final_state(n, n_steps, t_end, warm=True, seed=seed)
        cold = _final_state(n, n_steps, t_end, warm=False, seed=seed)
        max_abs = float(np.max(np.abs(warm - cold)))
        passed = max_abs == 0.0
        ok = ok and passed
        print(f"{n:>8} {max_abs:>12.3e} {'IDENTICAL' if passed else 'DIFFER':>10}")
    return ok


def per_step_vs_n(sizes: list[int], n_steps: int, t_end: float, seed: int) -> None:
    print(f"\n== 2. Per-step win vs N (n_steps={n_steps}, horizon={t_end}) ==")
    print(f"{'N':>8} {'warm ms':>10} {'cold ms':>10} {'speedup':>8} {'saved ms':>10}")
    for n in sizes:
        w = _per_step_ms(n, n_steps, t_end, warm=True, seed=seed)
        c = _per_step_ms(n, n_steps, t_end, warm=False, seed=seed)
        ratio = c / w if w else float("nan")
        print(f"{n:>8} {w:>10.3f} {c:>10.3f} {ratio:>7.2f}x {c - w:>10.3f}")


def win_vs_dt(n: int, step_counts: list[int], t_end: float, seed: int) -> None:
    print(f"\n== 3. Win vs coupling dt (N={n}, horizon={t_end}) ==")
    print(
        f"{'n_steps':>8} {'dt':>10} {'warm ms':>10} {'cold ms':>10} {'speedup':>8} {'saved ms':>10}"
    )
    for ns in step_counts:
        w = _per_step_ms(n, ns, t_end, warm=True, seed=seed)
        c = _per_step_ms(n, ns, t_end, warm=False, seed=seed)
        ratio = c / w if w else float("nan")
        print(f"{ns:>8} {t_end / ns:>10.2e} {w:>10.3f} {c:>10.3f} {ratio:>7.2f}x {c - w:>10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="species counts")
    ap.add_argument("--steps", type=int, default=200, help="coupling steps for the N sweep")
    ap.add_argument("--t-end", type=float, default=1.0, help="integration horizon")
    ap.add_argument("--dt-size", type=int, default=10000, help="N for the dt sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small, fast sweep")
    args = ap.parse_args()

    klu = has_klu()
    print("bngsim warm CVODE hot-path microbench (GH #102)")
    print(
        f"  version: {bngsim.__version__}   sparse linear solver (KLU): "
        f"{'YES' if klu else 'NO (dense fallback — caps N)'}"
    )

    if args.quick:
        sizes = args.sizes or ([1000, 10000] if klu else [500, 1000])
        dt_steps = [20, 100, 500]
    else:
        sizes = args.sizes or ([1000, 10000, 50000, 100000] if klu else [500, 1000, 2000])
        dt_steps = [20, 100, 500, 2000]

    ok = parity(sizes, n_steps=min(args.steps, 50), t_end=args.t_end, seed=args.seed)
    per_step_vs_n(sizes, args.steps, args.t_end, args.seed)
    win_vs_dt(args.dt_size, dt_steps, args.t_end, args.seed)

    print(
        "\nSummary: the warm path is byte-identical to the cold rebuild and removes the "
        "fixed\nper-call SUNDIALS/KLU re-init cost. That saved time grows with N (skipped "
        "symbolic\nfactorization) and, being fixed per call, dominates as the coupling dt "
        "shrinks — the\nmany-small-steps regime of a hybrid SSA/ODE splitting loop."
    )
    if not ok:
        raise SystemExit("PARITY FAILED: warm and cold trajectories differ")


if __name__ == "__main__":
    main()
