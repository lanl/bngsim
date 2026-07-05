#!/usr/bin/env python3
"""Generate 4-panel scatter plot from bench_1013_4engine.py results.

Reads results/bench_1013_4engine.json (manifest union Pool B) and produces a 2x2 figure:
  Row 1: ExprTk     | vs RR  | vs AMICI
  Row 2: Codegen    | vs RR  | vs AMICI

Each panel: log-scale scatter, colored by species count,
diagonal y=x reference line, geometric mean speedup annotation.

Usage:
    python plot_1013_4panel.py                         # default paths
    python plot_1013_4panel.py --json results/bench_1013_4engine.json
    python plot_1013_4panel.py --output bngsim/dev/paper/fig_1013_4panel.png

Output:
    bngsim/dev/paper/fig_1013_4panel.png  (300 DPI)
    bngsim/dev/paper/fig_1013_4panel.pdf
"""

import argparse
import json
import math
import sys
from pathlib import Path


def geometric_mean(values):
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in values) / len(values)))


def species_color(n_species):
    """Map species count to color."""
    if n_species <= 10:
        return "#2196F3"  # blue
    elif n_species <= 50:
        return "#FF9800"  # orange
    else:
        return "#F44336"  # red


def make_panel(
    ax, results, bng_key, other_key, title, rr_xval_set=None, am_xval_set=None, filter_set=None
):
    """Draw one scatter panel."""
    from matplotlib.lines import Line2D

    xs, ys, colors = [], [], []
    ratios = []

    for r in results:
        bt = r.get(bng_key, -1)
        ot = r.get(other_key, -1)
        if bt <= 0 or ot <= 0:
            continue
        if filter_set is not None and r["model"] not in filter_set:
            continue

        nsp = r.get("n_species", 1)
        bt_ms = bt * 1e3
        ot_ms = ot * 1e3
        xs.append(bt_ms)
        ys.append(ot_ms)
        colors.append(species_color(nsp))
        ratios.append(ot / bt)

    if not xs:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        ax.set_title(title, fontsize=10)
        return

    ax.scatter(xs, ys, c=colors, s=12, alpha=0.5, edgecolors="none")

    # Diagonal reference
    all_vals = xs + ys
    lo = max(min(all_vals) * 0.3, 0.005)
    hi = max(all_vals) * 3
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, which="both")

    # Geometric mean annotation
    gm = geometric_mean(ratios)
    nw = sum(1 for r in ratios if r > 1)
    ax.text(
        0.03,
        0.97,
        f"N={len(ratios)}\ngeomean={gm:.2f}x\nBNG wins {nw}/{len(ratios)}",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_title(title, fontsize=10)

    # Legend (only on first panel)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2196F3",
            markersize=5,
            label="1-10 sp",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF9800",
            markersize=5,
            label="11-50 sp",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#F44336",
            markersize=5,
            label=">50 sp",
        ),
    ]
    return legend_elements


def main():
    ap = argparse.ArgumentParser(description="4-panel scatter plot from Pool B 4-engine benchmark")
    ap.add_argument("--json", type=str, default=None, help="Path to bench_1013_4engine.json")
    ap.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = ap.parse_args()

    # Find results JSON
    if args.json:
        json_path = Path(args.json)
    else:
        json_path = Path(__file__).resolve().parent.parent / "results" / "bench_1013_4engine.json"
    if not json_path.exists():
        print(f"ERROR: Results not found: {json_path}")
        print("Run bench_1013_4engine.py first.")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    p2 = data.get("phase2_xval", [])
    p3 = data.get("phase3_timing", [])

    if not p3:
        print("ERROR: No Phase 3 timing data in JSON.")
        sys.exit(1)

    # Build xval filter sets
    rr_set = {r["model"] for r in p2 if r.get("rr_xval")}
    am_set = {r["model"] for r in p2 if r.get("amici_xval")}

    # Output path
    if args.output:
        out_png = Path(args.output)
    else:
        # Default matches paper layout: bngsim/dev/paper/ (see harness jobs.yaml).
        out_png = (
            Path(__file__).resolve().parent.parent.parent
            / "dev"
            / "paper"
            / "fig_1013_4panel.png"
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # ── Plot ──
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    panels = [
        (axes[0, 0], "exprTk_time", "rr_time", "BNGsim ExprTk vs libRoadRunner", rr_set),
        (axes[0, 1], "exprTk_time", "amici_time", "BNGsim ExprTk vs AMICI", am_set),
        (axes[1, 0], "codegen_time", "rr_time", "BNGsim Codegen vs libRoadRunner", rr_set),
        (axes[1, 1], "codegen_time", "amici_time", "BNGsim Codegen vs AMICI", am_set),
    ]

    legend_els = None
    for ax, bng_key, other_key, title, fset in panels:
        els = make_panel(ax, p3, bng_key, other_key, title, filter_set=fset)
        if els and legend_els is None:
            legend_els = els

    # Axis labels
    for ax in axes[1, :]:
        ax.set_xlabel("BNGsim wall time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Competitor wall time (ms)")

    # Legend on first panel
    if legend_els:
        axes[0, 0].legend(handles=legend_els, loc="lower right", fontsize=7)

    n_models = len(p3)
    fig.suptitle(
        f"BNGsim vs libRoadRunner vs AMICI on {n_models} BioModels\n"
        "(fitting-loop protocol: load once, warmup, 5x timed)",
        fontsize=11,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_pdf = str(out_png).replace(".png", ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()
