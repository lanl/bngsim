#!/usr/bin/env python3
"""BioModels benchmark with fitting-loop protocol (load once, warmup, time).

Session 54b.  Re-times the 506 cross-validated BioModels from Session 42
using the amortized protocol that matches PyBNF fitting loops:

Protocol per model:
  1. Load model ONCE  (outside timing)
  2. Create Simulator / RR ONCE  (outside timing)
  3. 1 warmup:  model.reset() + sim.run()  (outside timing)
  4. 5 timed:   model.reset() + sim.run()  → median

This eliminates per-model loading overhead (ExprTk compilation, LLVM JIT,
antimony→SBML conversion, codegen dlopen) and measures pure RHS evaluation
+ CVODE integration speed — the cost that matters for 10,000+ evaluations
in parameter fitting.

Reads Phase 2 checkpoint from bench_biomodels_sbml.py (506 models).

Usage:
    python bench_biomodels_warmup.py                 # full run
    python bench_biomodels_warmup.py --quick 50      # first 50 models
    python bench_biomodels_warmup.py --plot-only      # regenerate figure

Output:
    results/bench_biomodels_warmup.json
    bngsim/dev/paper/fig_biomodels_scatter_warmup.png
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

# ── Constants ─────────────────────────────────────────────────────

N_STEPS = 200
N_WARMUP = 1  # warmup sims (amortize first-call CVODE setup)
N_TIMED = 5  # timed sims

SIZE_BINS = [
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101+", 101, 999999),
]


# ── Helpers ───────────────────────────────────────────────────────


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
    return path


def load_checkpoint(name):
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── BNGsim ExprTk: load once, warmup, time ───────────────────────


def time_bngsim_exprTk(ant_path, t_end, bngsim_timeout=None):
    """Load with ExprTk (no codegen), warmup, time. Return median."""
    import bngsim

    old = os.environ.get("BNGSIM_NO_CODEGEN")
    os.environ["BNGSIM_NO_CODEGEN"] = "1"
    try:
        model = bngsim.Model.from_antimony(str(ant_path))
    finally:
        if old is None:
            os.environ.pop("BNGSIM_NO_CODEGEN", None)
        else:
            os.environ["BNGSIM_NO_CODEGEN"] = old

    try:
        sim = bngsim.Simulator(model, method="ode")

        for _ in range(N_WARMUP):
            model.reset()
            sim.run(t_span=(0, t_end), n_points=N_STEPS + 1, timeout=bngsim_timeout)

        times = []
        for _ in range(N_TIMED):
            model.reset()
            t0 = time.perf_counter()
            sim.run(t_span=(0, t_end), n_points=N_STEPS + 1, timeout=bngsim_timeout)
            times.append(time.perf_counter() - t0)

        return median(times)
    except Exception:
        return -1


# ── BNGsim Codegen: load once, warmup, time ──────────────────────


def time_bngsim_codegen(ant_path, t_end, bngsim_timeout=None):
    """Load with codegen (default), warmup, time. Return median."""
    import bngsim

    os.environ.pop("BNGSIM_NO_CODEGEN", None)
    try:
        model = bngsim.Model.from_antimony(str(ant_path))
        sim = bngsim.Simulator(model, method="ode")

        for _ in range(N_WARMUP):
            model.reset()
            sim.run(t_span=(0, t_end), n_points=N_STEPS + 1, timeout=bngsim_timeout)

        times = []
        for _ in range(N_TIMED):
            model.reset()
            t0 = time.perf_counter()
            sim.run(t_span=(0, t_end), n_points=N_STEPS + 1, timeout=bngsim_timeout)
            times.append(time.perf_counter() - t0)

        return median(times)
    except Exception:
        return -1


# ── libRoadRunner: load once, warmup, time ────────────────────────


def time_rr(ant_path, t_end):
    """Load once, warmup once, time N_TIMED runs. Return median."""
    import roadrunner

    try:
        sbml = _ant_to_sbml(str(ant_path))
        rr = roadrunner.RoadRunner(sbml)
        rr.integrator.absolute_tolerance = 1e-12
        rr.integrator.relative_tolerance = 1e-8

        # Warmup
        for _ in range(N_WARMUP):
            rr.reset()
            rr.simulate(0, t_end, N_STEPS + 1)

        # Timed runs
        times = []
        for _ in range(N_TIMED):
            rr.reset()
            t0 = time.perf_counter()
            rr.simulate(0, t_end, N_STEPS + 1)
            times.append(time.perf_counter() - t0)

        return median(times)
    except Exception:
        return -1


# ── Main timing loop ─────────────────────────────────────────────


def run_timing(xval_models, bngsim_timeout=None, resume_data=None):
    """Time all cross-validated models with warmup protocol."""
    print("\n" + "=" * 70)
    print(f"  Timing: {len(xval_models)} models")
    print(f"  Protocol: load once, {N_WARMUP} warmup, {N_TIMED} timed (median)")
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
            results.append(done[mid])
            continue

        et = time_bngsim_exprTk(ant_path, t_end, bngsim_timeout=bngsim_timeout)
        cg = time_bngsim_codegen(ant_path, t_end, bngsim_timeout=bngsim_timeout)
        rt = time_rr(ant_path, t_end)

        entry = {
            "model": mid,
            "ant_path": ant_path,
            "n_species": nsp,
            "t_end": t_end,
            "exprTk_time": et,
            "codegen_time": cg,
            "rr_time": rt,
        }

        parts = []
        if et > 0:
            parts.append(f"ET={et * 1e3:.3f}")
        if cg > 0:
            parts.append(f"CG={cg * 1e3:.3f}")
        if rt > 0:
            parts.append(f"RR={rt * 1e3:.3f}")
        if cg > 0 and rt > 0:
            parts.append(f"CG/RR={rt / cg:.2f}x")
        if et > 0 and rt > 0:
            parts.append(f"ET/RR={rt / et:.2f}x")
        print(f"  [{i + 1}/{len(xval_models)}] {mid} ({nsp}sp) {' '.join(parts)}")

        results.append(entry)

        if (i + 1) % 50 == 0:
            save_checkpoint(results, "bench_biomodels_warmup_ckpt")

    save_checkpoint(results, "bench_biomodels_warmup_ckpt")
    return results


# ── Analysis ──────────────────────────────────────────────────────


def analyze(results):
    """Compute bin-stratified 3-engine analysis."""
    print("\n" + "=" * 70)
    print("  ANALYSIS: ExprTk vs Codegen vs libRoadRunner (warmup protocol)")
    print("=" * 70)

    # Models with all 3 engines
    complete = [
        r
        for r in results
        if r.get("exprTk_time", -1) > 0
        and r.get("codegen_time", -1) > 0
        and r.get("rr_time", -1) > 0
    ]

    print(f"\n  Models with all 3 engines: {len(complete)}")

    if not complete:
        print("  No valid results!")
        return {}

    et_rr = [r["rr_time"] / r["exprTk_time"] for r in complete]
    cg_rr = [r["rr_time"] / r["codegen_time"] for r in complete]
    cg_et = [r["exprTk_time"] / r["codegen_time"] for r in complete]

    print("\n  OVERALL:")
    print(
        f"    ET/RR geomean: {geometric_mean(et_rr):.2f}x "
        f"(ET wins {sum(1 for s in et_rr if s > 1)}/{len(complete)})"
    )
    print(
        f"    CG/RR geomean: {geometric_mean(cg_rr):.2f}x "
        f"(CG wins {sum(1 for s in cg_rr if s > 1)}/{len(complete)})"
    )
    print(
        f"    CG/ET geomean: {geometric_mean(cg_et):.2f}x "
        f"(CG wins {sum(1 for s in cg_et if s > 1)}/{len(complete)})"
    )

    # Bin analysis
    bins = []
    print(f"\n  {'Bin':>8s} {'N':>5s} {'ET/RR':>8s} {'CG/RR':>8s} {'CG/ET':>8s} {'CG>RR':>6s}")
    print("  " + "-" * 50)

    for label, lo, hi in SIZE_BINS:
        bm = [r for r in complete if lo <= r.get("n_species", 0) <= hi]
        if not bm:
            bins.append({"bin": label, "n": 0})
            continue
        b_et_rr = geometric_mean([r["rr_time"] / r["exprTk_time"] for r in bm])
        b_cg_rr = geometric_mean([r["rr_time"] / r["codegen_time"] for r in bm])
        b_cg_et = geometric_mean([r["exprTk_time"] / r["codegen_time"] for r in bm])
        cg_wins = sum(1 for r in bm if r["codegen_time"] < r["rr_time"])
        bins.append(
            {
                "bin": label,
                "n": len(bm),
                "et_vs_rr": b_et_rr,
                "cg_vs_rr": b_cg_rr,
                "cg_vs_et": b_cg_et,
                "cg_wins_rr": cg_wins,
            }
        )
        print(
            f"  {label:>8s} {len(bm):>5d} {b_et_rr:>8.2f}x "
            f"{b_cg_rr:>8.2f}x {b_cg_et:>8.2f}x "
            f"{cg_wins:>3d}/{len(bm)}"
        )

    analysis = {
        "n_models": len(complete),
        "et_vs_rr_geomean": geometric_mean(et_rr),
        "cg_vs_rr_geomean": geometric_mean(cg_rr),
        "cg_vs_et_geomean": geometric_mean(cg_et),
        "bins": bins,
    }
    return analysis


# ── Plotting ──────────────────────────────────────────────────────


def generate_figure(results, output_path):
    """Generate 2-panel scatter plot: ExprTk vs RR and Codegen vs RR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Filter to models with all 3 engines
    valid = [
        r
        for r in results
        if r.get("exprTk_time", -1) > 0
        and r.get("codegen_time", -1) > 0
        and r.get("rr_time", -1) > 0
    ]

    if not valid:
        print("  WARNING: No valid 3-engine results for figure!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2196F3",
            markersize=6,
            label="1–10 species",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF9800",
            markersize=6,
            label="11–50 species",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#F44336",
            markersize=6,
            label=">50 species",
        ),
    ]

    for ax, bng_key, title in [
        (ax1, "exprTk_time", "BNGsim ExprTk vs libRoadRunner"),
        (ax2, "codegen_time", "BNGsim Codegen vs libRoadRunner"),
    ]:
        for r in valid:
            nsp = r.get("n_species", 0)
            bt = r[bng_key] * 1e3  # ms
            rt = r["rr_time"] * 1e3
            if nsp <= 10:
                c = "#2196F3"
            elif nsp <= 50:
                c = "#FF9800"
            else:
                c = "#F44336"
            ax.scatter(bt, rt, c=c, s=15, alpha=0.6, edgecolors="none")

        # Diagonal
        all_vals = [r[bng_key] * 1e3 for r in valid] + [r["rr_time"] * 1e3 for r in valid]
        lo = max(min(all_vals) * 0.5, 0.01)
        hi = max(all_vals) * 2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("BNGsim wall time (ms)")
        ax.set_ylabel("libRoadRunner wall time (ms)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2, which="both")
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.suptitle(
        "BNGsim vs libRoadRunner on 472 BioModels\n(load-once + warmup protocol)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n  Figure saved: {output_path}")

    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  PDF saved: {pdf_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="BioModels benchmark with warmup protocol (Session 54b)"
    )
    ap.add_argument("--quick", type=int, default=0, help="Limit to first N models")
    ap.add_argument(
        "--plot-only", action="store_true", help="Regenerate figure from existing JSON"
    )
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    add_bngsim_timeout_arg(ap, default=None)
    args = ap.parse_args()

    paper_dir = Path(__file__).resolve().parent.parent.parent.parent / "paper"
    fig_path = paper_dir / "fig_biomodels_scatter_warmup.png"

    if args.plot_only:
        data = load_checkpoint("bench_biomodels_warmup")
        if data is None:
            # Try loading from results file
            rpath = RESULTS_DIR / "bench_biomodels_warmup.json"
            if rpath.exists():
                with open(rpath) as f:
                    data = json.load(f)
            else:
                print("ERROR: No results found. Run benchmark first.")
                sys.exit(1)
        timing = data.get("timing", data)
        generate_figure(timing, str(fig_path))
        return

    info = get_machine_info()

    # Load Phase 2 cross-validation results
    p2 = load_checkpoint("bench_biomodels_sbml_phase2")
    if p2 is None:
        print("ERROR: No Phase 2 checkpoint found.")
        print("Run bench_biomodels_sbml.py first.")
        sys.exit(1)

    xval_models = [r for r in p2 if r.get("xval")]
    print(f"Cross-validated models from Phase 2: {len(xval_models)}")

    if args.quick > 0:
        xval_models = xval_models[: args.quick]
        print(f"Limited to first {args.quick} models")

    # Resume
    resume = load_checkpoint("bench_biomodels_warmup_ckpt") if args.resume else None

    # Run timing
    timing = run_timing(
        xval_models,
        bngsim_timeout=args.bngsim_timeout,
        resume_data=resume,
    )

    # Analysis
    analysis = analyze(timing)

    # Save results
    output = {
        "machine_info": info,
        "protocol": {
            "n_warmup": N_WARMUP,
            "n_timed": N_TIMED,
            "n_steps": N_STEPS,
            "bngsim_timeout": args.bngsim_timeout,
            "description": (
                "Load model ONCE, create Simulator ONCE, "
                f"{N_WARMUP} warmup sim(s), then {N_TIMED} timed sims "
                "(model.reset() + sim.run()). Report median."
            ),
        },
        "analysis": analysis,
        "timing": timing,
    }
    save_results(output, "bench_biomodels_warmup")

    # Generate figure
    paper_dir.mkdir(parents=True, exist_ok=True)
    generate_figure(timing, str(fig_path))

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
