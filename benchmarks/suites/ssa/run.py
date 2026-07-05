#!/usr/bin/env python3
"""``ssa`` suite runner -- SSA correctness + timing: BNGsim vs run_network.

Two gates per model (see ``benchmarks/_netbench.py`` for the machinery):

1. **correctness** -- an ensemble of replicate trajectories is simulated by
   each engine and the two ensemble means are compared with a *z*-test.
   Both engines run exact SSA on the same ``.net``, so their means must
   agree within stochastic error.
2. **timing** -- a warmup + timed-run wall-clock comparison.  Per the suite
   design rule, timing is only reported for a model that passed
   correctness (timing is meaningless if the result is wrong).

Usage::

    python run.py                       # both gates, full 12-model sweep
    python run.py --mode correctness     # correctness gate only
    python run.py --mode timing          # timing gate only (skips the z-test)
    python run.py --effort low           # cheap subset (cumulative tiers)
    python run.py --replicates 40        # larger correctness ensemble
"""

import argparse
import json
import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg, filter_by_effort  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
NET_DIR = _BENCH_ROOT / "models" / "net" / "ssa"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# The 12 SSA/PSA stochastic models, vendored at models/net/ssa/.  Each carries
# an "effort" tier (low/medium/high) driving --effort; the cost driver is the
# SSA event count -- network size combined with simulated horizon.  "ssa_skip"
# marks erk_activation: populations up to 3e6 make exact SSA O(billions of
# events) per replicate, so it is exercised by the psa suite instead.
MODELS = [
    {
        "name": "gene_expression_hill",
        "species": 2,
        "reactions": 4,
        "t_end": 3600,
        "n_steps": 360,
        "effort": "low",
        "notes": "Positive autoregulation with Hill function; bimodal protein distribution",
    },
    {
        "name": "simple_system",
        "species": 4,
        "reactions": 4,
        "t_end": 5,
        "n_steps": 200,
        "effort": "low",
        "notes": "Enzyme-substrate binding with phosphorylation/dephosphorylation",
    },
    {
        "name": "flagellar_motor",
        "species": 4,
        "reactions": 4,
        "t_end": 0.5,
        "n_steps": 200,
        "effort": "low",
        "notes": "CheY-driven motor switching via global functions",
    },
    {
        "name": "oscillatory_system",
        "species": 5,
        "reactions": 8,
        "t_end": 50,
        "n_steps": 400,
        "effort": "low",
        "notes": "Negative-feedback oscillations with functional rates",
    },
    {
        "name": "gene_expression",
        "species": 10,
        "reactions": 14,
        "t_end": 60000,
        "n_steps": 1000,
        "effort": "medium",
        "notes": "Two-state gene model with constitutive and regulated transcription",
    },
    {
        "name": "tcr_signaling",
        "species": 37,
        "reactions": 97,
        "t_end": 300,
        "n_steps": 1000,
        "effort": "medium",
        "notes": "Bistable stochastic switching in TCR activation",
    },
    {
        "name": "erk_activation",
        "species": 34,
        "reactions": 65,
        "t_end": 8640,
        "n_steps": 1000,
        "effort": "medium",
        "ssa_skip": True,
        "notes": "Populations up to 3e6; exact SSA infeasible -- see the psa suite",
    },
    {
        "name": "gene_expr_3stage",
        "species": 6,
        "reactions": 6,
        "t_end": 200000000,
        "n_steps": 10000,
        "effort": "high",
        "notes": "Bursty promoter switching; long horizon makes the SSA event count large",
    },
    {
        "name": "prion_aggregation",
        "species": 104,
        "reactions": 2809,
        "t_end": 10,
        "n_steps": 1000,
        "effort": "high",
        "notes": "Nucleated polymerization; 2809 reactions make SSA slow",
    },
    {
        "name": "egfr_net",
        "species": 356,
        "reactions": 3749,
        "t_end": 30,
        "n_steps": 120,
        "effort": "high",
        "notes": "EGFR signaling cascade (Blinov 2006)",
    },
    {
        "name": "multisite_phos",
        "species": 1026,
        "reactions": 7680,
        "t_end": 1.5,
        "n_steps": 300,
        "effort": "high",
        "notes": "5-site distributive phosphorylation; combinatorial explosion",
    },
    {
        "name": "fceri_gamma",
        "species": 3744,
        "reactions": 58276,
        "t_end": 100,
        "n_steps": 100,
        "effort": "high",
        "notes": "FceRI receptor aggregation and Syk activation",
    },
]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_markdown(payload, outpath):
    """Render the SSA correctness + timing report from collected results."""
    info = payload["machine_info"]
    lines = ["# SSA suite -- BNGsim vs run_network\n", "## Machine\n"]
    for label, key in (
        ("Platform", "platform"),
        ("Processor", "processor"),
        ("Python", "python"),
        ("BNGsim", "bngsim_version"),
        ("run_network", "run_network_version"),
        ("Git commit", "git_commit"),
        ("Date", "date"),
    ):
        lines.append(f"- **{label}**: {info.get(key, 'n/a')}")
    lines.append(
        f"- **Protocol**: mode={payload['mode']}, correctness ensemble="
        f"{payload['replicates']} replicates (z-tol {nb.DEFAULT_Z_TOL}), "
        f"timing={payload['warmup']} warmup + {payload['runs']} timed runs, median"
    )
    lines.append("")

    lines.append("## Results\n")
    lines.append(
        "| Model | Species | Rxns | Correctness | BNGsim time (s) | BNGsim steps | "
        "RN time (s) | RN steps | Speedup |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    speedups = []
    for r in payload["results"]:
        name, sp, rxn = r["name"], r["species"], r["reactions"]
        if r["status"] == "skip":
            lines.append(
                f"| {name} | {sp} | {rxn} | *(skipped: {r['reason']})* | — | — | — | — | — |"
            )
            continue

        corr = r.get("correctness")
        if corr is None:
            corr_str = "*(not run)*"
        elif corr.get("error"):
            corr_str = f"ERROR: {corr['error'][:40]}"
        elif corr["passed"]:
            corr_str = f"PASS (max\\|z\\|={corr['stat']:.2f})"
        else:
            corr_str = f"**FAIL** (max\\|z\\|={corr['stat']:.2f})"

        tim = r.get("timing")
        if tim is None:
            bt = bs = rt = rs = su = "—"
        else:
            bng, rn = tim["bngsim"], tim["run_network"]
            bt = f"{bng['median_wall_time']:.4f}" if bng["median_wall_time"] > 0 else "error"
            bs = f"{bng.get('median_steps', -1):,}" if bng.get("median_steps", -1) >= 0 else "n/a"
            rt = f"{rn['median_wall_time']:.4f}" if rn["median_wall_time"] > 0 else "error"
            rs = f"{rn.get('median_steps', -1):,}" if rn.get("median_steps", -1) >= 0 else "n/a"
            if tim["speedup"]:
                su = f"**{tim['speedup']:.1f}×**"
                speedups.append(tim["speedup"])
            else:
                su = "n/a"
        lines.append(f"| {name} | {sp} | {rxn} | {corr_str} | {bt} | {bs} | {rt} | {rs} | {su} |")

    lines.append("")
    geo = nb.geometric_mean(speedups)
    if geo:
        lines.append(f"**Geometric-mean speedup (correctness-passing models): {geo:.1f}×**\n")
    outpath.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SSA suite: BNGsim vs run_network")
    parser.add_argument(
        "--mode",
        choices=["correctness", "timing", "both"],
        default="both",
        help="Which gates to run (default: both -- timing only for correctness-passing models).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=nb.DEFAULT_REPLICATES,
        help=f"Correctness-ensemble size per engine (default: {nb.DEFAULT_REPLICATES}).",
    )
    parser.add_argument(
        "--warmup", type=int, default=nb.DEFAULT_WARMUP, help="Timing warmup runs."
    )
    parser.add_argument("--runs", type=int, default=nb.DEFAULT_RUNS, help="Timing timed runs.")
    add_effort_arg(parser)
    args = parser.parse_args()

    models = filter_by_effort(MODELS, args.effort, key=lambda m: m["effort"])

    print("=" * 70)
    print("  SSA suite -- BNGsim vs run_network")
    print("=" * 70)
    print(f"  mode={args.mode}, effort={args.effort}: {len(models)} of {len(MODELS)} models")

    results = []
    for cfg in models:
        name = cfg["name"]
        net_path = NET_DIR / f"{name}.net"
        print(f"\n--- {name} ({cfg['species']} sp, {cfg['reactions']} rxn) ---")

        row = {
            "name": name,
            "species": cfg["species"],
            "reactions": cfg["reactions"],
            "t_end": cfg["t_end"],
            "n_steps": cfg["n_steps"],
            "status": "ok",
        }

        if cfg.get("ssa_skip"):
            print("  SKIP: exact SSA infeasible (populations too large)")
            row["status"] = "skip"
            row["reason"] = "exact SSA infeasible"
            results.append(row)
            continue

        if not net_path.exists():
            print(f"  SKIP: .net not found: {net_path}")
            row["status"] = "skip"
            row["reason"] = ".net not found"
            results.append(row)
            continue

        correctness_ok = True
        if args.mode in ("correctness", "both"):
            print(f"  correctness: {args.replicates}-replicate ensemble z-test ...")
            corr = nb.correctness_gate(
                net_path, "ssa", cfg["t_end"], cfg["n_steps"], replicates=args.replicates
            )
            row["correctness"] = corr
            if corr["error"]:
                print(f"    ERROR: {corr['error'][:120]}")
                correctness_ok = False
            elif corr["passed"]:
                print(f"    PASS — {corr['detail']}")
            else:
                print(f"    FAIL — {corr['detail']}")
                correctness_ok = False

        if args.mode == "timing" or (args.mode == "both" and correctness_ok):
            print(f"  timing: {args.warmup} warmup + {args.runs} timed runs ...")
            tim = nb.timing_compare(
                net_path, "ssa", cfg["t_end"], cfg["n_steps"], warmup=args.warmup, runs=args.runs
            )
            row["timing"] = tim
            bng_t = tim["bngsim"]["median_wall_time"]
            rn_t = tim["run_network"]["median_wall_time"]
            su = f"{tim['speedup']:.1f}×" if tim["speedup"] else "n/a"
            print(f"    BNGsim {bng_t:.4f}s | run_network {rn_t:.4f}s | speedup {su}")
        elif args.mode == "both" and not correctness_ok:
            print("  timing: skipped (correctness gate did not pass)")

        results.append(row)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for r in results if r.get("correctness", {}).get("passed"))
    n_fail = sum(1 for r in results if r.get("correctness") and not r["correctness"].get("passed"))
    n_skip = sum(1 for r in results if r["status"] == "skip")
    print(f"  models: {len(results)}  correctness PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")

    payload = {
        "machine_info": nb.machine_info(),
        "mode": args.mode,
        "replicates": args.replicates,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "ssa_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    generate_markdown(payload, RESULTS_DIR / "ssa_results.md")
    print(f"\nResults: {json_path}")
    print(f"Report:  {RESULTS_DIR / 'ssa_results.md'}")


if __name__ == "__main__":
    main()
