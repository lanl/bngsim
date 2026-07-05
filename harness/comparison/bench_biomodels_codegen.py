#!/usr/bin/env python3
"""BioModels 3-way benchmark: BNGsim ExprTk vs BNGsim Codegen vs libRoadRunner.

Reads the Phase 2 checkpoint from bench_biomodels_sbml.py (506 cross-validated
models) and runs timing for all three configurations:
  1. BNGsim with ExprTk bytecode (BNGSIM_NO_CODEGEN=1)
  2. BNGsim with auto-codegen (compiled C RHS, default)
  3. libRoadRunner (LLVM-compiled RHS)

Protocol: 2 warmup + 5 timed runs, median wall time per model.

Usage:
    python bench_biomodels_codegen.py                    # full run
    python bench_biomodels_codegen.py --quick 50         # first 50 models
    python bench_biomodels_codegen.py --engine exprTk    # ExprTk only
    python bench_biomodels_codegen.py --engine codegen   # codegen only
    python bench_biomodels_codegen.py --engine rr        # RR only
    python bench_biomodels_codegen.py --analyze          # analysis only
    python bench_biomodels_codegen.py --resume           # resume from checkpoint

Output:
    results/bench_biomodels_codegen.json
    results/bench_biomodels_codegen_analysis.json

Session 53.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    RESULTS_DIR,
    _ant_to_sbml,
    add_bngsim_timeout_arg,
    format_bngsim_timeout,
    get_machine_info,
    save_results,
)

# ── Constants ─────────────────────────────────────────────────────────────

N_STEPS = 200
_WARMUP = 2
_RUNS = 5

SIZE_BINS = [
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101+", 101, 999999),
]


# ── Helpers ───────────────────────────────────────────────────────────────


def geometric_mean(values):
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in values) / len(values)))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_checkpoint(data, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Checkpoint: {path}")
    return path


def load_checkpoint(name):
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Timing functions ──────────────────────────────────────────────────────
#
# IMPORTANT: We time SIMULATION ONLY, not model loading. The model is loaded
# once (outside timing), then sim.run() is timed repeatedly. This measures
# the RHS evaluation speed, which is what matters for fitting loops (1000s
# of evaluations per model). Model loading is amortized.
#
# For the codegen path, the .so is compiled at model load time (cached by
# SHA-256 hash). The simulation uses the dlopen'd .so via function pointer.
# For the ExprTk path, the simulation uses bytecode interpretation.
# For RR, the simulation uses LLVM-compiled RHS.


def _load_bngsim_exprTk(ant_path):
    """Load model with ExprTk bytecode (codegen disabled)."""
    old_val = os.environ.get("BNGSIM_NO_CODEGEN")
    os.environ["BNGSIM_NO_CODEGEN"] = "1"
    try:
        import bngsim

        model = bngsim.Model.from_antimony(str(ant_path))
        return model
    finally:
        if old_val is None:
            os.environ.pop("BNGSIM_NO_CODEGEN", None)
        else:
            os.environ["BNGSIM_NO_CODEGEN"] = old_val


def _load_bngsim_codegen(ant_path):
    """Load model with auto-codegen (compiled C RHS)."""
    os.environ.pop("BNGSIM_NO_CODEGEN", None)
    import bngsim

    model = bngsim.Model.from_antimony(str(ant_path))
    return model


def _load_rr(ant_path):
    """Load RoadRunner model."""
    import roadrunner

    sbml = _ant_to_sbml(str(ant_path))
    rr = roadrunner.RoadRunner(sbml)
    rr.integrator.absolute_tolerance = 1e-12
    rr.integrator.relative_tolerance = 1e-8
    return rr


def _run_bngsim(model, t_end, n_steps, bngsim_timeout=None):
    """Time a single BNGsim sim.run() on a pre-loaded model.

    Preserves codegen .so path through clone (clone() doesn't
    copy _codegen_so_path, so we do it explicitly).
    """
    import bngsim

    m = model.clone()
    m.reset()
    # Propagate codegen .so path (clone doesn't copy it)
    cg_path = getattr(model, "_codegen_so_path", "")
    if cg_path:
        m._codegen_so_path = cg_path
    sim = bngsim.Simulator(m, method="ode")
    t0 = time.perf_counter()
    # Optional wall-clock guard; timed_median() converts SimulationTimeout
    # into -1 (DNF) like any other engine failure.
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)
    return time.perf_counter() - t0


def _run_rr(rr, t_end, n_steps):
    """Time a single RoadRunner simulate() on a pre-loaded model."""
    rr.reset()
    t0 = time.perf_counter()
    rr.simulate(0, t_end, n_steps + 1)
    return time.perf_counter() - t0


def timed_median(fn, warmup=_WARMUP, runs=_RUNS):
    times = []
    for i in range(warmup + runs):
        try:
            t = fn()
            if t < 0:
                return -1
        except Exception:
            return -1
        if i >= warmup:
            times.append(t)
    return median(times)


# ── Phase: Timing ─────────────────────────────────────────────────────────


def run_timing(xval_models, engines, warmup, runs, bngsim_timeout=None, resume_data=None):
    """Run timing for specified engines on cross-validated models.

    Models are loaded ONCE outside the timing loop; only sim.run()
    is timed. This measures RHS evaluation speed, not load time.

    engines: list of strings from {"exprTk", "codegen", "rr"}
    """
    print("\n" + "=" * 70)
    print(f"  Timing: {len(xval_models)} models x engines={engines}")
    print(f"  Protocol: {warmup}w + {runs}t, median (sim only)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(bngsim_timeout)}")
    print("=" * 70)

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r
        print(f"  Resuming: {len(done)} already done")

    results = []

    for i, m in enumerate(xval_models):
        mid = m["model"]
        ant_path = m["ant_path"]
        t_end = m["t_end"]
        nsp = m.get("n_species", 0)

        if mid in done:
            entry = done[mid]
        else:
            entry = {
                "model": mid,
                "ant_path": ant_path,
                "n_species": nsp,
                "t_end": t_end,
            }

        # ExprTk: load once, time sim.run()
        if "exprTk" in engines and "exprTk_time" not in entry:
            try:
                mdl = _load_bngsim_exprTk(ant_path)
                bt = timed_median(
                    lambda mdl=mdl, te=t_end, timeout=bngsim_timeout: _run_bngsim(
                        mdl, te, N_STEPS, bngsim_timeout=timeout
                    ),
                    warmup=warmup,
                    runs=runs,
                )
                entry["exprTk_time"] = bt
            except Exception as e:
                entry["exprTk_time"] = -1
                entry["exprTk_error"] = str(e)[:200]

        # Codegen: load once, time sim.run()
        if "codegen" in engines and "codegen_time" not in entry:
            try:
                mdl = _load_bngsim_codegen(ant_path)
                has_cg = bool(getattr(mdl, "_codegen_so_path", ""))
                entry["codegen_active"] = has_cg
                bt = timed_median(
                    lambda mdl=mdl, te=t_end, timeout=bngsim_timeout: _run_bngsim(
                        mdl, te, N_STEPS, bngsim_timeout=timeout
                    ),
                    warmup=warmup,
                    runs=runs,
                )
                entry["codegen_time"] = bt
            except Exception as e:
                entry["codegen_time"] = -1
                entry["codegen_error"] = str(e)[:200]

        # RR: load once, time simulate()
        if "rr" in engines and "rr_time" not in entry:
            try:
                rr_obj = _load_rr(ant_path)
                rt = timed_median(
                    lambda rr=rr_obj, te=t_end: _run_rr(rr, te, N_STEPS), warmup=warmup, runs=runs
                )
                entry["rr_time"] = rt
            except Exception as e:
                entry["rr_time"] = -1
                entry["rr_error"] = str(e)[:200]

        results.append(entry)

        # Progress
        et = entry.get("exprTk_time", -1)
        ct = entry.get("codegen_time", -1)
        rt = entry.get("rr_time", -1)
        parts = []
        if et > 0:
            parts.append(f"ET={et:.4f}")
        if ct > 0:
            parts.append(f"CG={ct:.4f}")
        if rt > 0:
            parts.append(f"RR={rt:.4f}")
        if ct > 0 and rt > 0:
            parts.append(f"CG/RR={rt / ct:.1f}x")
        if ct > 0 and et > 0:
            parts.append(f"CG/ET={et / ct:.2f}x")

        print(f"  [{i + 1}/{len(xval_models)}] {mid} ({nsp}sp) {' '.join(parts)}")

        if (i + 1) % 50 == 0:
            save_checkpoint(results, "bench_biomodels_codegen_timing")

    save_checkpoint(results, "bench_biomodels_codegen_timing")
    return results


# ── Analysis ──────────────────────────────────────────────────────────────


def analyze_results(timing_results):
    """Compute bin-stratified 3-way comparison."""
    print("\n" + "=" * 70)
    print("  3-WAY ANALYSIS: ExprTk vs Codegen vs libRoadRunner")
    print("=" * 70)

    # Filter to models with all 3 timings
    complete = [
        r
        for r in timing_results
        if r.get("exprTk_time", -1) > 0
        and r.get("codegen_time", -1) > 0
        and r.get("rr_time", -1) > 0
    ]

    print(f"\n  Models with all 3 engines: {len(complete)}")

    # Overall stats
    et_rr_ratios = [r["rr_time"] / r["exprTk_time"] for r in complete]
    cg_rr_ratios = [r["rr_time"] / r["codegen_time"] for r in complete]
    cg_et_ratios = [r["exprTk_time"] / r["codegen_time"] for r in complete]

    overall = {
        "n_models": len(complete),
        "exprTk_vs_rr_geomean": geometric_mean(et_rr_ratios),
        "codegen_vs_rr_geomean": geometric_mean(cg_rr_ratios),
        "codegen_vs_exprTk_geomean": geometric_mean(cg_et_ratios),
        "exprTk_wins_vs_rr": sum(1 for r in et_rr_ratios if r > 1),
        "codegen_wins_vs_rr": sum(1 for r in cg_rr_ratios if r > 1),
        "codegen_wins_vs_exprTk": sum(1 for r in cg_et_ratios if r > 1),
    }

    print(f"\n  OVERALL ({len(complete)} models):")
    print(
        f"    ExprTk/RR  geomean: {overall['exprTk_vs_rr_geomean']:.2f}x "
        f"(BNGsim ExprTk wins {overall['exprTk_wins_vs_rr']}/{len(complete)})"
    )
    print(
        f"    Codegen/RR geomean: {overall['codegen_vs_rr_geomean']:.2f}x "
        f"(BNGsim Codegen wins {overall['codegen_wins_vs_rr']}/{len(complete)})"
    )
    print(
        f"    Codegen/ET geomean: {overall['codegen_vs_exprTk_geomean']:.2f}x "
        f"(Codegen wins {overall['codegen_wins_vs_exprTk']}/{len(complete)})"
    )

    # Bin-stratified analysis
    bins = []
    print(f"\n  {'Bin':>8s} {'N':>5s} {'ET/RR':>8s} {'CG/RR':>8s} {'CG/ET':>8s} {'CG wins':>8s}")
    print("  " + "-" * 55)

    for label, lo, hi in SIZE_BINS:
        bin_models = [r for r in complete if lo <= r.get("n_species", 0) <= hi]
        if not bin_models:
            bins.append(
                {
                    "bin": label,
                    "n": 0,
                    "exprTk_vs_rr": 0,
                    "codegen_vs_rr": 0,
                    "codegen_vs_exprTk": 0,
                }
            )
            continue

        et_rr = [r["rr_time"] / r["exprTk_time"] for r in bin_models]
        cg_rr = [r["rr_time"] / r["codegen_time"] for r in bin_models]
        cg_et = [r["exprTk_time"] / r["codegen_time"] for r in bin_models]

        gm_et_rr = geometric_mean(et_rr)
        gm_cg_rr = geometric_mean(cg_rr)
        gm_cg_et = geometric_mean(cg_et)
        cg_wins_rr = sum(1 for r in cg_rr if r > 1)

        bins.append(
            {
                "bin": label,
                "n": len(bin_models),
                "exprTk_vs_rr": gm_et_rr,
                "codegen_vs_rr": gm_cg_rr,
                "codegen_vs_exprTk": gm_cg_et,
                "codegen_wins_vs_rr": cg_wins_rr,
            }
        )

        print(
            f"  {label:>8s} {len(bin_models):>5d} {gm_et_rr:>8.2f}x"
            f" {gm_cg_rr:>8.2f}x {gm_cg_et:>8.2f}x"
            f" {cg_wins_rr:>4d}/{len(bin_models)}"
        )

    # Codegen failure rate
    all_with_et = [r for r in timing_results if r.get("exprTk_time", -1) > 0]
    all_with_cg = [r for r in timing_results if r.get("codegen_time", -1) > 0]
    cg_fail = len(all_with_et) - len(all_with_cg)

    print(f"\n  Codegen success rate: {len(all_with_cg)}/{len(all_with_et)} ({cg_fail} failures)")

    # Summary stats for exprTk vs codegen by model size
    print("\n  Codegen speedup vs ExprTk by model size:")
    for label, lo, hi in SIZE_BINS:
        bin_models = [r for r in complete if lo <= r.get("n_species", 0) <= hi]
        if not bin_models:
            continue
        cg_et = [r["exprTk_time"] / r["codegen_time"] for r in bin_models]
        gm = geometric_mean(cg_et)
        med = sorted(cg_et)[len(cg_et) // 2]
        mn = min(cg_et)
        mx = max(cg_et)
        print(f"    {label:>8s}: geomean={gm:.3f}x  median={med:.3f}x  range=[{mn:.3f}, {mx:.3f}]")

    analysis = {
        "overall": overall,
        "bins": bins,
        "codegen_failures": cg_fail,
    }

    return analysis


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="BioModels 3-way benchmark: ExprTk vs Codegen vs RR")
    ap.add_argument("--quick", type=int, default=0, help="Limit to first N models")
    ap.add_argument(
        "--engine",
        type=str,
        default="all",
        choices=["all", "exprTk", "codegen", "rr"],
        help="Run only one engine (for parallel runs)",
    )
    ap.add_argument("--analyze", action="store_true", help="Analysis only (no timing)")
    ap.add_argument("--resume", action="store_true", help="Resume from timing checkpoint")
    ap.add_argument("--warmup", type=int, default=_WARMUP)
    ap.add_argument("--runs", type=int, default=_RUNS)
    add_bngsim_timeout_arg(ap, default=None)
    args = ap.parse_args()

    info = get_machine_info()

    # Load Phase 2 cross-validation results
    p2 = load_checkpoint("bench_biomodels_sbml_phase2")
    if p2 is None:
        print("ERROR: No Phase 2 checkpoint found.")
        print("Run bench_biomodels_sbml.py first to generate cross-validated models.")
        sys.exit(1)

    xval_models = [r for r in p2 if r.get("xval")]
    print(f"Cross-validated models from Phase 2: {len(xval_models)}")

    if args.quick > 0:
        xval_models = xval_models[: args.quick]
        print(f"Limited to first {args.quick} models")

    if args.analyze:
        # Analysis only — load timing checkpoint
        timing = load_checkpoint("bench_biomodels_codegen_timing")
        if timing is None:
            print("ERROR: No timing checkpoint found.")
            sys.exit(1)
        analysis = analyze_results(timing)
        save_results({"analysis": analysis, "timing": timing}, "bench_biomodels_codegen")
        return

    # Determine engines to run
    engines = ["exprTk", "codegen", "rr"] if args.engine == "all" else [args.engine]

    # Load existing timing checkpoint for resume
    resume = load_checkpoint("bench_biomodels_codegen_timing") if args.resume else None

    # Run timing
    timing = run_timing(
        xval_models,
        engines,
        warmup=args.warmup,
        runs=args.runs,
        bngsim_timeout=args.bngsim_timeout,
        resume_data=resume,
    )

    # Analysis
    analysis = analyze_results(timing)

    # Save final results
    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "n_steps": N_STEPS,
            "engines": engines,
            "bngsim_timeout": args.bngsim_timeout,
        },
        "analysis": analysis,
        "timing": timing,
    }
    save_results(output, "bench_biomodels_codegen")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
