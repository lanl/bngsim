#!/usr/bin/env python3
"""``psa`` suite runner -- PSA correctness + timing: BNGsim vs run_network.

PSA -- the partial-scaling approximation (Lin, Feng & Hlavacek 2019) --
trades exactness for speed on models whose populations make exact SSA
infeasible.  Each model is swept over a list of ``Nc`` population levels.

Two gates per ``(model, Nc)`` (machinery in ``benchmarks/_netbench.py``):

1. **correctness** -- BNGsim PSA and ``run_network`` PSA each simulate a
   replicate ensemble; the two ensemble means are compared with a
   *standardized-mean-difference* gate.  The two engines run *different*
   PSA scalings (BNGsim partial-scaling vs run_network heterogeneous
   adaptive scaling), so a *z*-test would eventually flag their bounded
   method bias as a failure -- the effect-size statistic, measured in
   process-sigma units, is replicate-count-stable and instead catches a
   gross discrepancy.  It is a cross-engine *consistency* check.
2. **timing** -- a warmup + timed-run wall-clock comparison, reported only
   for a ``(model, Nc)`` that passed correctness.

Usage::

    python run.py                       # both gates, all models x Nc
    python run.py --mode correctness     # correctness gate only
    python run.py --mode timing          # timing gate only
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
NET_DIR = _BENCH_ROOT / "models" / "net" / "psa"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# The 3 PSA benchmark models, vendored at models/net/psa/.  Each is swept
# over a list of Nc population levels and carries an "effort" tier driving
# --effort.  Exact SSA is impractical for these models (large populations or
# many reactions) -- that is the motivation for PSA.
MODELS = [
    {
        "name": "tcr_signaling",
        "species": 37,
        "reactions": 97,
        "t_end": 300,
        "n_steps": 1000,
        "poplevels": [10, 30, 100, 300],
        "effort": "low",
        "notes": "Bistable stochastic switching; PSA Nc=300 ~8x faster than SSA",
    },
    {
        "name": "erk_activation",
        "species": 34,
        "reactions": 65,
        "t_end": 8640,
        "n_steps": 1000,
        "poplevels": [10, 30, 100, 300],
        "effort": "medium",
        "notes": "Populations up to 3e6; exact SSA infeasible (billions of events)",
    },
    {
        "name": "prion_aggregation",
        "species": 104,
        "reactions": 2809,
        "t_end": 10,
        "n_steps": 1000,
        "poplevels": [10, 30, 100, 300],
        "effort": "high",
        "notes": "Nucleated polymerization; 2809 reactions make SSA slow",
    },
]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_markdown(payload, outpath):
    """Render the PSA correctness + timing report from collected results."""
    info = payload["machine_info"]
    lines = ["# PSA suite -- BNGsim vs run_network\n", "## Machine\n"]
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
        f"{payload['replicates']} replicates, effect-size gate (d-tol "
        f"{nb.DEFAULT_D_TOL}), timing={payload['warmup']} warmup + "
        f"{payload['runs']} timed runs, median"
    )
    lines.append("")
    lines.append("PSA reference: Lin YT, Feng S, Hlavacek WS (2019) J Chem Phys 150: 244101.\n")

    lines.append("## Results\n")
    lines.append(
        "| Model | Nc | Species | Rxns | Correctness | BNGsim time (s) | "
        "BNGsim steps | RN time (s) | Speedup |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    speedups = []
    for r in payload["results"]:
        name, nc, sp, rxn = r["name"], r["poplevel"], r["species"], r["reactions"]
        if r["status"] == "skip":
            lines.append(
                f"| {name} | {nc} | {sp} | {rxn} | *(skipped: {r['reason']})* | — | — | — | — |"
            )
            continue

        corr = r.get("correctness")
        if corr is None:
            corr_str = "*(not run)*"
        elif corr.get("error"):
            corr_str = f"ERROR: {corr['error'][:40]}"
        elif corr["passed"]:
            corr_str = f"PASS (d99.9={corr['stat']:.2f})"
        else:
            corr_str = f"**FAIL** (d99.9={corr['stat']:.2f})"

        tim = r.get("timing")
        if tim is None:
            bt = bs = rt = su = "—"
        else:
            bng, rn = tim["bngsim"], tim["run_network"]
            bt = f"{bng['median_wall_time']:.4f}" if bng["median_wall_time"] > 0 else "error"
            bs = f"{bng.get('median_steps', -1):,}" if bng.get("median_steps", -1) >= 0 else "n/a"
            rt = f"{rn['median_wall_time']:.4f}" if rn["median_wall_time"] > 0 else "error"
            if tim["speedup"]:
                su = f"**{tim['speedup']:.1f}×**"
                speedups.append(tim["speedup"])
            else:
                su = "n/a"
        lines.append(f"| {name} | {nc} | {sp} | {rxn} | {corr_str} | {bt} | {bs} | {rt} | {su} |")

    lines.append("")
    geo = nb.geometric_mean(speedups)
    if geo:
        lines.append(f"**Geometric-mean speedup (correctness-passing jobs): {geo:.1f}×**\n")
    outpath.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PSA suite: BNGsim vs run_network")
    parser.add_argument(
        "--mode",
        choices=["correctness", "timing", "both"],
        default="both",
        help="Which gates to run (default: both -- timing only for correctness-passing jobs).",
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
    n_jobs = sum(len(m["poplevels"]) for m in models)

    print("=" * 70)
    print("  PSA suite -- BNGsim vs run_network")
    print("=" * 70)
    print(
        f"  mode={args.mode}, effort={args.effort}: {len(models)} models, {n_jobs} (model,Nc) jobs"
    )

    results = []
    for cfg in models:
        name = cfg["name"]
        net_path = NET_DIR / f"{name}.net"
        print(f"\n=== {name} ({cfg['species']} sp, {cfg['reactions']} rxn) ===")

        for nc in cfg["poplevels"]:
            print(f"\n--- {name} @ Nc={nc} ---")
            row = {
                "name": name,
                "poplevel": nc,
                "species": cfg["species"],
                "reactions": cfg["reactions"],
                "t_end": cfg["t_end"],
                "n_steps": cfg["n_steps"],
                "status": "ok",
            }

            if not net_path.exists():
                print(f"  SKIP: .net not found: {net_path}")
                row["status"] = "skip"
                row["reason"] = ".net not found"
                results.append(row)
                continue

            correctness_ok = True
            if args.mode in ("correctness", "both"):
                print(f"  correctness: {args.replicates}-replicate PSA ensemble ...")
                corr = nb.correctness_gate(
                    net_path,
                    "psa",
                    cfg["t_end"],
                    cfg["n_steps"],
                    replicates=args.replicates,
                    poplevel=nc,
                    statistic="effect_size",
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
                    net_path,
                    "psa",
                    cfg["t_end"],
                    cfg["n_steps"],
                    warmup=args.warmup,
                    runs=args.runs,
                    poplevel=nc,
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
    print(f"  jobs: {len(results)}  correctness PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")

    payload = {
        "machine_info": nb.machine_info(),
        "mode": args.mode,
        "replicates": args.replicates,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "psa_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    generate_markdown(payload, RESULTS_DIR / "psa_results.md")
    print(f"\nResults: {json_path}")
    print(f"Report:  {RESULTS_DIR / 'psa_results.md'}")


if __name__ == "__main__":
    main()
