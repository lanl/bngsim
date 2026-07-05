#!/usr/bin/env python3
"""``python_ode`` suite runner — pure-Python ODE workflow comparison.

Promoted from the ODE half of ``harness/comparison/bench_pythonic_workflows.py``
(paper Table S6). Compares three ODE engines on models defined **entirely in
Python** — no files, no readers:

  * BNGsim CVODE — built via BNGsim's ``ModelBuilder`` API (the reference),
  * scipy ``solve_ivp`` LSODA — a hand-coded numpy RHS,
  * Diffrax ``Kvaerno5`` — a hand-coded JAX RHS, JIT-compiled.

Every model is hand-coded three ways from the same reaction network, so all
three engines integrate an *exact* representation — there is no parsed-file
RHS approximation in the loop. The predecessor also drove scipy/Diffrax from
a hand-rolled ``_net_reader``→RHS translation; that path was dropped because
it could not faithfully reproduce BNG's network rate-law semantics (it gave
wrong trajectories for scipy *and* Diffrax on functional-rate models).

The model set is curated to span Diffrax's working range: classic
non-stiff systems it handles cleanly (``lotka_volterra``, ``sir``,
``m1_ground``) plus one stiff high-count system (``simple_system``) that
exposes its adaptive-stepper limit — an honest data point, not a hidden
failure.

Two gates per (model, engine):

1. correctness — each engine's trajectory is cross-validated against
   BNGsim CVODE on the shared time grid (numpy-allclose, additive atol
   scaled from the per-species peak). BNGsim is the reference.
2. timing — warmup + timed-run median wall time. Per the suite design
   rule, timing is only reported for an engine that passed correctness.

Usage:
    python run.py                     # both gates, full sweep
    python run.py --mode correctness  # cross-validation only
    python run.py --mode timing       # timing only
    python run.py --effort low        # cheap subset (cumulative tiers)
    python run.py --model sir         # substring filter on model name

Output (git-ignored ``results/``):
    python_ode_results.json + python_ode_results.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg, filter_by_effort  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_WARMUP = nb.DEFAULT_WARMUP
DEFAULT_RUNS = nb.DEFAULT_RUNS

# Cross-validation tolerances: a non-BNGsim engine's trajectory passes when
# |a - bng| <= XVAL_ATOL_FRAC*peak + XVAL_RTOL*|bng| for every cell, where
# peak is the per-species max magnitude. These compare distinct stiff
# solvers (CVODE vs LSODA vs Kvaerno5), so the floor is set above
# cross-method integration noise rather than at machine epsilon.
XVAL_RTOL = 1e-4
XVAL_ATOL_FRAC = 1e-6

# Diffrax's adaptive integrator caps total steps; the default (4096) is too
# small for stiff mass-action, so the budget is raised well past what
# Kvaerno5 needs on the non-stiff models. The lone stiff model
# (simple_system) still exhausts it — reported honestly as a failure.
DIFFRAX_MAX_STEPS = 1_000_000


# ── Timing / cross-validation helpers ─────────────────────────────────────


def time_fn(fn, *, warmup, runs):
    """Time fn (warmup + runs). Returns (median_time_or_None, error_or_None)."""
    for _ in range(warmup):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            return None, str(e)[:200]
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            return None, str(e)[:200]
        times.append(time.perf_counter() - t0)
    return median(times), None


def fmt(t):
    if t is None:
        return "---"
    return "<0.001" if t < 0.001 else f"{t:.4f}"


def max_rel_err(approx: np.ndarray, ref: np.ndarray) -> float:
    """Worst cross-validation error of ``approx`` vs the BNGsim ``ref``.

    Returns a plain relative number; <= XVAL_RTOL means every cell passes
    ``|a - r| <= XVAL_ATOL_FRAC*peak + XVAL_RTOL*|r|``.
    """
    approx = np.asarray(approx, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if approx.shape != ref.shape:
        return float("inf")
    peak = np.maximum(np.abs(ref).max(axis=0, keepdims=True), 1e-300)
    denom = XVAL_ATOL_FRAC * peak + XVAL_RTOL * np.abs(ref)
    return float(np.max(np.abs(approx - ref) / denom) * XVAL_RTOL)


# ══════════════════════════════════════════════════════════════════════════
# Models — each is hand-coded three ways (BNGsim ModelBuilder / numpy RHS /
# JAX RHS) from the same reaction network, with species in a single shared
# order so the trajectory arrays line up column-for-column.
# ══════════════════════════════════════════════════════════════════════════


def _bngsim_model(params, species, reactions):
    """Assemble a BNGsim model via ModelBuilder.

    ``params``   — list of (name, value).
    ``species``  — list of (name, initial_value), defining the state order.
    ``reactions``— list of (reactant_idx_list, product_idx_list, rate_param).
    """
    from bngsim._bngsim_core import ModelBuilder
    from bngsim._model import Model

    b = ModelBuilder()
    for name, val in params:
        b.add_parameter(name, val)
    handles = [b.add_species(name, float(ic)) for name, ic in species]
    for reactants, products, rate in reactions:
        b.add_reaction(
            [handles[i] for i in reactants],
            [handles[i] for i in products],
            "elementary",
            rate,
        )
    return Model(_core=b.build())


# ── lotka_volterra — classic non-stiff predator/prey oscillator ───────────
# X -> 2X (prey birth k1); X + Y -> 2Y (predation k2); Y -> 0 (death k3).
_LV_P = {"k1": 1.0, "k2": 0.01, "k3": 1.0}
_LV_Y0 = [120.0, 80.0]  # X (prey), Y (predator) — off the (100, 100) fixed point


def lotka_volterra_bngsim():
    return _bngsim_model(
        list(_LV_P.items()),
        [("X", _LV_Y0[0]), ("Y", _LV_Y0[1])],
        [([0], [0, 0], "k1"), ([0, 1], [1, 1], "k2"), ([1], [], "k3")],
    )


def _lv_terms(X, Y):
    k1, k2, k3 = _LV_P["k1"], _LV_P["k2"], _LV_P["k3"]
    return (k1 * X - k2 * X * Y, k2 * X * Y - k3 * Y)


# ── sir — epidemic model, non-stiff ───────────────────────────────────────
# S + I -> 2I (infection beta); I -> R (recovery gamma). The infection is the
# bimolecular mass-action form of the .net's functional rate beta*S*I.
_SIR_P = {"beta": 5e-8, "gamma": 1.0 / 7.0}
_SIR_Y0 = [2.0e7, 1.0, 0.0]  # S, I, R


def sir_bngsim():
    return _bngsim_model(
        list(_SIR_P.items()),
        [("S", _SIR_Y0[0]), ("I", _SIR_Y0[1]), ("R", _SIR_Y0[2])],
        [([0, 1], [1, 1], "beta"), ([1], [2], "gamma")],
    )


def _sir_terms(s, i, r):
    inf = _SIR_P["beta"] * s * i
    rec = _SIR_P["gamma"] * i
    return (-inf, inf - rec, rec)


# ── m1_ground — linear decay chain A->B->C->D, non-stiff ──────────────────
_M1_P = {"k1": 0.03, "k2": 0.02, "k3": 0.01}
_M1_Y0 = [100.0, 0.0, 0.0, 0.0]  # A, B, C, D


def m1_ground_bngsim():
    return _bngsim_model(
        list(_M1_P.items()),
        [("A", _M1_Y0[0]), ("B", _M1_Y0[1]), ("C", _M1_Y0[2]), ("D", _M1_Y0[3])],
        [([0], [1], "k1"), ([1], [2], "k2"), ([2], [3], "k3")],
    )


def _m1_terms(A, B, C, D):
    k1, k2, k3 = _M1_P["k1"], _M1_P["k2"], _M1_P["k3"]
    return (-k1 * A, k1 * A - k2 * B, k2 * B - k3 * C, k3 * C)


# ── simple_system — 4-species phosphorylation, stiff (high counts) ────────
# Retained as the honest data point: Diffrax's adaptive Kvaerno5 cannot
# integrate it within DIFFRAX_MAX_STEPS at the shared atol=1e-8.
_SS_P = {"kon": 10.0, "koff": 5.0, "kcat": 0.7, "dephos": 0.5}
_SS_Y0 = [5000.0, 0.0, 500.0, 0.0]  # X_u, Xp, Y, XY


def simple_system_bngsim():
    return _bngsim_model(
        list(_SS_P.items()),
        [("X_u", _SS_Y0[0]), ("Xp", _SS_Y0[1]), ("Y", _SS_Y0[2]), ("XY", _SS_Y0[3])],
        [([0, 2], [3], "kon"), ([1], [0], "dephos"), ([3], [0, 2], "koff"), ([3], [1, 2], "kcat")],
    )


def _ss_terms(Xu, Xp, Y, XY):
    kon, koff, kcat, dephos = _SS_P["kon"], _SS_P["koff"], _SS_P["kcat"], _SS_P["dephos"]
    r1 = kon * Xu * Y
    r2 = dephos * Xp
    r3 = koff * XY
    r4 = kcat * XY
    return (-r1 + r2 + r3, -r2 + r4, -r1 + r3 + r4, r1 - r3 - r4)


MODELS = [
    {
        "name": "m1_ground",
        "species": 4,
        "t_end": 100.0,
        "n_steps": 200,
        "effort": "low",
        "bngsim": m1_ground_bngsim,
        "terms": _m1_terms,
        "y0": _M1_Y0,
    },
    {
        "name": "sir",
        "species": 3,
        "t_end": 100.0,
        "n_steps": 200,
        "effort": "medium",
        "bngsim": sir_bngsim,
        "terms": _sir_terms,
        "y0": _SIR_Y0,
    },
    {
        "name": "lotka_volterra",
        "species": 2,
        "t_end": 20.0,
        "n_steps": 400,
        "effort": "medium",
        "bngsim": lotka_volterra_bngsim,
        "terms": _lv_terms,
        "y0": _LV_Y0,
    },
    {
        "name": "simple_system",
        "species": 4,
        "t_end": 5.0,
        "n_steps": 200,
        "effort": "high",
        "bngsim": simple_system_bngsim,
        "terms": _ss_terms,
        "y0": _SS_Y0,
    },
]


# ── Engine runners — each returns the species trajectory (n_points, n_sp) ──


def run_bngsim(mdef):
    import bngsim

    sim = bngsim.Simulator(mdef["bngsim"](), method="ode")
    result = sim.run(t_span=(0, mdef["t_end"]), n_points=mdef["n_steps"] + 1)
    return np.asarray(result.species, dtype=float)


def run_scipy(mdef):
    from scipy.integrate import solve_ivp

    terms = mdef["terms"]
    y0 = np.array(mdef["y0"], dtype=float)
    t_end, n_steps = mdef["t_end"], mdef["n_steps"]

    def rhs(t, y):
        return np.array(terms(*y))

    sol = solve_ivp(
        rhs,
        (0, t_end),
        y0,
        method="LSODA",
        t_eval=np.linspace(0, t_end, n_steps + 1),
        rtol=1e-8,
        atol=1e-8,
    )
    return sol.y.T


def run_diffrax(mdef):
    import diffrax
    import jax.numpy as jnp

    terms = mdef["terms"]
    y0 = jnp.array(mdef["y0"], dtype=float)
    t_end, n_steps = mdef["t_end"], mdef["n_steps"]

    def rhs(t, y, args):
        return jnp.array(terms(*y))

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Kvaerno5(),
        t0=0,
        t1=t_end,
        dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, t_end, n_steps + 1)),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=DIFFRAX_MAX_STEPS,
    )
    return np.asarray(sol.ys, dtype=float)


# ── Per-model orchestration ───────────────────────────────────────────────


def run_model(mdef, runners, *, mode, warmup, runs):
    """Run one model: correctness vs BNGsim + timing for each engine.

    ``runners`` is an ordered dict ``{engine: callable(mdef)}`` whose first
    entry is BNGsim (the reference). Returns a per-engine result dict.
    """
    out = {}
    ref = None
    for engine, fn in runners.items():
        entry: dict = {"engine": engine}
        if mode in ("correctness", "both"):
            try:
                traj = fn(mdef)
            except Exception as e:  # noqa: BLE001
                entry["correctness_ok"] = False
                entry["error"] = str(e)[:200]
                out[engine] = entry
                continue
            if engine == "bngsim":
                ref = traj
                entry["correctness_ok"] = True
                entry["xval_err"] = 0.0
            elif ref is None:
                entry["correctness_ok"] = False
                entry["error"] = "no BNGsim reference trajectory"
            else:
                err = max_rel_err(traj, ref)
                entry["xval_err"] = err
                entry["correctness_ok"] = err <= XVAL_RTOL
        else:
            entry["correctness_ok"] = True  # timing-only mode does not gate

        if mode in ("timing", "both") and entry.get("correctness_ok"):
            t, err = time_fn(lambda f=fn: f(mdef), warmup=warmup, runs=runs)
            entry["time_s"] = t
            if err:
                entry["error"] = err
        out[engine] = entry
    return out


def generate_markdown(payload: dict, outpath: Path) -> None:
    info = payload["machine_info"]
    lines = [
        "# `python_ode` suite results",
        "",
        f"- mode: `{payload['mode']}`  effort: `{payload['effort']}`",
        f"- host: {info.get('platform', '?')}",
        "",
        "ODE workflow comparison — BNGsim CVODE (reference) vs scipy LSODA",
        "vs Diffrax Kvaerno5, on models hand-coded entirely in Python.",
        "Timing is reported only for an engine that passed cross-validation.",
        "",
        "| Model | engine | xval err | correct | time (s) |",
        "|-------|--------|----------|---------|----------|",
    ]
    for m in payload["models"]:
        for engine, e in m["engines"].items():
            xv = e.get("xval_err")
            xv_s = "—" if xv is None else f"{xv:.2e}"
            ok = "ok" if e.get("correctness_ok") else "FAIL"
            lines.append(f"| {m['name']} | {engine} | {xv_s} | {ok} | {fmt(e.get('time_s'))} |")
    lines.append("")
    outpath.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(
        description="python_ode suite — pure-Python ODE workflow comparison"
    )
    ap.add_argument(
        "--mode",
        choices=("correctness", "timing", "both"),
        default="both",
        help="Which gates to run (default: both).",
    )
    ap.add_argument("--model", type=str, default="", help="Substring filter on model name.")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    add_effort_arg(ap)
    args = ap.parse_args()

    models = filter_by_effort(MODELS, args.effort, key=lambda m: m["effort"])
    if args.model:
        models = [m for m in models if args.model.lower() in m["name"].lower()]

    have_diffrax = _have("diffrax")
    if not have_diffrax:
        print("  WARNING: diffrax not installed — Diffrax engine skipped")

    print("=" * 72)
    print("  python_ode suite — pure-Python ODE workflow comparison (Table S6)")
    print(f"  mode={args.mode}  effort={args.effort}: {len(models)} of {len(MODELS)} models")
    print(f"  Protocol: {args.warmup}w + {args.runs}t, median wall time")
    print("=" * 72)

    all_models = []
    for mdef in models:
        name = mdef["name"]
        print(f"\n  {name} ({mdef['species']} sp, t_end={mdef['t_end']:g})")

        runners = {"bngsim": run_bngsim, "scipy": run_scipy}
        if have_diffrax:
            runners["diffrax"] = run_diffrax
        engines = run_model(mdef, runners, mode=args.mode, warmup=args.warmup, runs=args.runs)

        for engine, e in engines.items():
            xv = e.get("xval_err")
            xv_s = "" if xv is None else f"  xval={xv:.2e}"
            ok = "ok" if e.get("correctness_ok") else "FAIL"
            extra = f"  ({e['error']})" if e.get("error") else ""
            print(f"    {engine:<10} {ok:<5} time={fmt(e.get('time_s')):>10}{xv_s}{extra}")

        all_models.append({"name": name, "species": mdef["species"], "engines": engines})

    payload = {
        "machine_info": nb.machine_info(),
        "mode": args.mode,
        "effort": args.effort,
        "protocol": {"warmup": args.warmup, "runs": args.runs},
        "xval": {"rtol": XVAL_RTOL, "atol_frac": XVAL_ATOL_FRAC},
        "models": all_models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "python_ode_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    generate_markdown(payload, RESULTS_DIR / "python_ode_results.md")
    print(f"\nResults: {json_path}")
    print(f"Report:  {RESULTS_DIR / 'python_ode_results.md'}")


def _have(mod_name) -> bool:
    try:
        __import__(mod_name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
