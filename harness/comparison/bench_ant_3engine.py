#!/usr/bin/env python3
"""3-engine ODE comparison on Pool A Antimony models.

Uses @SIM comment tags baked into each .ant file for time horizons.
Compares BNGsim vs libRoadRunner vs AMICI.

Usage:
    python bench_ant_3engine.py [--quick N]
"""

import argparse
import hashlib
import re
import shutil
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    HARNESS_DIR,
    AmiciApiError,
    _ant_to_sbml,
    add_bngsim_timeout_arg,
    amici_prepare_model,
    amici_simulate,
    discover_pool_a,
    format_bngsim_timeout,
    get_machine_info,
    save_results,
)

AMICI_CACHE = HARNESS_DIR / "models" / "amici_cache"
WARMUP = 2
RUNS = 5


def parse_sim_tag(ant_path):
    """Parse @SIM comment from .ant file."""
    with open(ant_path) as f:
        for line in f:
            if "@SIM" in line:
                d = {}
                for m in re.finditer(r"(\w+)=([\d.eE+\-]+)", line):
                    d[m.group(1)] = float(m.group(2))
                return d
    return None


def run_bngsim(ant_path, t_end, n_steps, bngsim_timeout=None):
    """Run BNGsim ODE."""
    import bngsim

    try:
        model = bngsim.Model.from_antimony(str(ant_path))
        sim = bngsim.Simulator(model, method="ode")
        t0 = time.perf_counter()
        sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)
        return time.perf_counter() - t0
    except Exception:
        return -1


def run_rr(sbml_str, t_end, n_steps):
    """Run libRoadRunner ODE."""
    import roadrunner

    try:
        rr = roadrunner.RoadRunner(sbml_str)
        rr.integrator.absolute_tolerance = 1e-12
        rr.integrator.relative_tolerance = 1e-8
        t0 = time.perf_counter()
        rr.simulate(0, t_end, n_steps + 1)
        return time.perf_counter() - t0
    except Exception:
        return -1


def compile_amici(sbml_str, name):
    """Compile AMICI model (cached)."""
    import amici

    h = hashlib.sha256(sbml_str.encode()).hexdigest()[:12]
    safe = f"amici_ant_{name}_{h}"
    mdir = AMICI_CACHE / safe
    if mdir.exists():
        try:
            return amici.import_model_module(safe, str(mdir))
        except Exception:
            shutil.rmtree(mdir, ignore_errors=True)
    AMICI_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        imp = amici.SbmlImporter(sbml_str, from_file=False)
        imp.sbml2amici(safe, str(mdir), verbose=False)
        return amici.import_model_module(safe, str(mdir))
    except Exception:
        shutil.rmtree(mdir, ignore_errors=True)
        return None


def run_amici(mod, t_end, n_steps):
    """Run AMICI ODE (AMICI >= 1.0 snake_case API); return elapsed sec or -1.

    Re-raises AmiciApiError (API drift) so it crashes loudly instead of degrading
    to a silent -1 (GH #227); a genuine per-model failure returns -1.
    """
    try:
        model, solver = amici_prepare_model(mod)
        tspan = np.linspace(0, t_end, n_steps + 1)
        _, elapsed = amici_simulate(model, solver, tspan)
        return elapsed
    except AmiciApiError:
        raise
    except Exception:
        return -1


def timed(fn, w=WARMUP, r=RUNS):
    """Run fn w+r times, return median of last r."""
    times = []
    for i in range(w + r):
        t = fn()
        if t < 0:
            return -1
        if i >= w:
            times.append(t)
    return median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", type=int, default=0)
    add_bngsim_timeout_arg(parser, default=None)
    args = parser.parse_args()

    models = discover_pool_a()
    if args.quick > 0:
        models = models[: args.quick]

    info = get_machine_info()

    print("=" * 65)
    print("  3-Engine ODE: BNGsim vs RR vs AMICI (Pool A)")
    print(f"  Protocol: {WARMUP}w + {RUNS}t, median")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(args.bngsim_timeout)}")
    print(f"  Models: {len(models)}")
    print("=" * 65)

    results = []
    bng_times = []
    rr_ratios = []
    am_ratios = []
    n_rr_ok = 0
    n_am_ok = 0
    n_am_compile_fail = 0

    for i, ant_path in enumerate(models):
        name = ant_path.stem
        sim = parse_sim_tag(ant_path)
        if not sim:
            print(f"  [{i + 1}] {name}: no @SIM tag, skip")
            continue

        t_end = sim.get("T_END", 1.0)
        n_steps = int(sim.get("N_STEPS", 200))

        # Convert to SBML once
        try:
            sbml_str = _ant_to_sbml(str(ant_path))
        except Exception:
            print(f"  [{i + 1}] {name}: ant->SBML fail")
            continue

        # BNGsim
        bt = timed(
            lambda ant_path=ant_path, t_end=t_end, n_steps=n_steps, timeout=args.bngsim_timeout: run_bngsim(
                ant_path, t_end, n_steps, bngsim_timeout=timeout
            )
        )

        # RoadRunner
        rt = timed(
            lambda sbml_str=sbml_str, t_end=t_end, n_steps=n_steps: run_rr(
                sbml_str, t_end, n_steps
            )
        )

        # AMICI (compile once, time simulation)
        amici_mod = compile_amici(sbml_str, name)
        if amici_mod is not None:
            at = timed(
                lambda amici_mod=amici_mod, t_end=t_end, n_steps=n_steps: run_amici(
                    amici_mod, t_end, n_steps
                )
            )
        else:
            at = -1
            n_am_compile_fail += 1

        entry = {
            "model": name,
            "t_end": t_end,
            "n_steps": n_steps,
            "bngsim": bt,
            "rr": rt,
            "amici": at,
        }
        results.append(entry)

        # Stats
        parts = []
        if bt > 0:
            parts.append(f"BNG={bt:.4f}")
            bng_times.append(bt)
        if rt > 0:
            parts.append(f"RR={rt:.4f}")
            n_rr_ok += 1
            if bt > 0:
                rr_ratios.append(rt / bt)
        if at > 0:
            parts.append(f"AM={at:.4f}")
            n_am_ok += 1
            if bt > 0:
                am_ratios.append(at / bt)

        print(f"  [{i + 1}/{len(models)}] {name} (t={t_end:.0f}) {' '.join(parts)}")

    # Summary
    import math

    def gm(v):
        return math.exp(sum(math.log(x) for x in v) / len(v))

    print(f"\n{'=' * 65}")
    if rr_ratios:
        print(
            f"  RR/BNG ratio geomean: {gm(rr_ratios):.2f} "
            f"({n_rr_ok} models) "
            f"-> BNG {1 / gm(rr_ratios):.1f}x faster"
        )
    if am_ratios:
        print(
            f"  AMICI/BNG ratio geomean: {gm(am_ratios):.2f} "
            f"({n_am_ok} models) "
            f"-> BNG {1 / gm(am_ratios):.1f}x faster"
        )
    print(f"  AMICI compile failures: {n_am_compile_fail}")
    print(f"{'=' * 65}")

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": WARMUP,
            "runs": RUNS,
            "bngsim_timeout": args.bngsim_timeout,
        },
        "summary": {
            "n_models": len(results),
            "n_rr_ok": n_rr_ok,
            "n_amici_ok": n_am_ok,
            "n_amici_compile_fail": n_am_compile_fail,
            "rr_bng_ratio_geomean": gm(rr_ratios) if rr_ratios else None,
            "amici_bng_ratio_geomean": gm(am_ratios) if am_ratios else None,
        },
        "results": results,
    }
    save_results(output, "bench_ant_3engine")


if __name__ == "__main__":
    main()
