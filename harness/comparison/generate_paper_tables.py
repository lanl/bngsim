#!/usr/bin/env python3
"""Generate paper-ready tables from benchmark JSON results.

Reads all JSON files from results/ and generates:
- Markdown summary tables
- LaTeX tables for publication

Usage:
    python generate_paper_tables.py

Output:
    results/paper_tables.md
    results/paper_tables.tex
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RESULTS_DIR, geometric_mean


def load_json(name):
    """Load a results JSON, return None if missing."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, prec=4):
    """Format a number for display."""
    if x is None or x < 0:
        return "—"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.{prec}f}"


def gen_ode_table(data):
    """Generate ODE vs run_network table."""
    if not data:
        return []
    lines = [
        "## ODE: BNGsim vs run_network\n",
        "| Model | Species | Rxns | BNGsim (s) | run_network (s) | Speedup |",
        "|-------|---------|------|------------|----------------|---------|",
    ]
    speedups = []
    for r in data.get("results", []):
        name = r.get("model", "?")
        sp = r.get("species", "?")
        rxn = r.get("reactions", "?")
        bt = r.get("bngsim_time", -1)
        rt = r.get("rn_time", -1)
        su = r.get("speedup")
        su_str = f"**{su:.1f}x**" if su else "—"
        if su:
            speedups.append(su)
        lines.append(f"| {name} | {sp} | {rxn} | {fmt(bt)} | {fmt(rt)} | {su_str} |")
    if speedups:
        gm = geometric_mean(speedups)
        lines.append("")
        lines.append(f"**Geometric mean speedup: {gm:.1f}x** (N={len(speedups)})")
    lines.append("")
    return lines


def gen_ssa_table(data):
    """Generate SSA vs run_network table."""
    if not data:
        return []
    lines = [
        "## SSA: BNGsim vs run_network\n",
        "| Model | Species | Rxns | BNGsim (s) | run_network (s) | Speedup |",
        "|-------|---------|------|------------|----------------|---------|",
    ]
    speedups = []
    for r in data.get("results", []):
        if r.get("skipped"):
            continue
        name = r.get("model", "?")
        sp = r.get("species", "?")
        rxn = r.get("reactions", "?")
        bt = r.get("bngsim_time", -1)
        rt = r.get("rn_time", -1)
        su = r.get("speedup")
        su_str = f"**{su:.1f}x**" if su else "—"
        if su:
            speedups.append(su)
        lines.append(f"| {name} | {sp} | {rxn} | {fmt(bt)} | {fmt(rt)} | {su_str} |")
    if speedups:
        gm = geometric_mean(speedups)
        lines.append("")
        lines.append(f"**Geometric mean speedup: {gm:.1f}x** (N={len(speedups)})")
    lines.append("")
    return lines


def gen_rr_table(data):
    """Generate BNGsim vs RoadRunner table."""
    if not data:
        return []
    lines = [
        "## ODE: BNGsim vs libRoadRunner\n",
        "| Pool | Models | BNG wins | RR wins | Ratio (BNG/RR) |",
        "|------|--------|----------|---------|----------------|",
    ]
    for pool in ["A", "B"]:
        pool_r = [
            r for r in data.get("results", []) if r.get("pool") == pool and r.get("status") == "ok"
        ]
        if not pool_r:
            continue
        ratios = [r["ratio_bng_rr"] for r in pool_r if "ratio_bng_rr" in r]
        if ratios:
            n_bng = sum(1 for r in ratios if r < 1)
            n_rr = sum(1 for r in ratios if r >= 1)
            gm = geometric_mean(ratios)
            lines.append(f"| {pool} | {len(ratios)} | {n_bng} | {n_rr} | {gm:.2f} |")
    lines.append("")
    return lines


def gen_validation_summary(data, label):
    """Generate a validation summary line."""
    if not data:
        return []
    s = data.get("summary", {})
    total = s.get("total", 0)
    passed = s.get("pass", 0)
    return [f"- **{label}**: {passed}/{total} pass"]


def gen_latex_ode(data):
    """Generate LaTeX table for ODE results."""
    if not data:
        return ""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{BNGsim vs run\_network ODE performance}",
        r"\label{tab:ode_performance}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Species & Rxns & BNGsim (s) & "
        r"run\_network (s) & Speedup \\",
        r"\midrule",
    ]
    speedups = []
    for r in data.get("results", []):
        name = r.get("model", "?").replace("_", r"\_")
        sp = r.get("species", "?")
        rxn = r.get("reactions", "?")
        bt = r.get("bngsim_time", -1)
        rt = r.get("rn_time", -1)
        su = r.get("speedup")
        su_s = f"{su:.1f}$\\times$" if su else "---"
        if su:
            speedups.append(su)
        lines.append(f"{name} & {sp} & {rxn} & {fmt(bt)} & {fmt(rt)} & {su_s} \\\\")
    if speedups:
        gm = geometric_mean(speedups)
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{5}{r}{"
            r"\textbf{Geometric mean}} & "
            f"\\textbf{{{gm:.1f}$\\times$}} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  Generating paper tables from results/")
    print("=" * 70)

    # Markdown
    md_lines = ["# BNGsim Benchmark Results\n"]

    # Validation summary
    md_lines.append("## Validation Summary\n")
    for name, label in [
        ("validate_ode", "ODE vs run_network (27 models)"),
        ("validate_ssa", "SSA determinism + moments (10 models)"),
        ("validate_nf", "NFsim vs BNG2.pl (5 models)"),
        ("validate_sbml", "Antimony vs libRoadRunner"),
    ]:
        data = load_json(name)
        md_lines.extend(gen_validation_summary(data, label))
    md_lines.append("")

    # Performance tables
    ode_data = load_json("bench_ode_vs_runnetwork")
    md_lines.extend(gen_ode_table(ode_data))

    ssa_data = load_json("bench_ssa_vs_runnetwork")
    md_lines.extend(gen_ssa_table(ssa_data))

    rr_data = load_json("bench_ode_vs_roadrunner")
    md_lines.extend(gen_rr_table(rr_data))

    md_path = RESULTS_DIR / "paper_tables.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines))
    print(f"  Markdown: {md_path}")

    # LaTeX
    latex = gen_latex_ode(ode_data)
    if latex:
        tex_path = RESULTS_DIR / "paper_tables.tex"
        tex_path.write_text(latex)
        print(f"  LaTeX: {tex_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
