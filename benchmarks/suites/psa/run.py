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

Plus an anchor-free **diagnostics** pass (GH #15, on by default): one BNGsim
PSA run per ``(model, Nc)`` whose ``psa_diagnostics`` are digested into the
report -- the measured event-rate speedup ``ŝ``, which channels scaled and how
hard (``m̄``), the peak-population activation signal, and the top variance-
inflation channels by ``q̄``. It needs no ``run_network`` and no exact-SSA
anchor, so it reports even where neither gate can (exact SSA infeasible).

Usage::

    python run.py                       # both gates + diagnostics, all models x Nc
    python run.py --mode correctness     # correctness gate only (+ diagnostics)
    python run.py --mode timing          # timing gate only (+ diagnostics)
    python run.py --no-diagnostics       # skip the diagnostics pass
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

# Fixed seed + top-K channels for the anchor-free PSA diagnostics pass (GH #15).
DIAG_SEED = 20260716
DIAG_TOP_K = 3


# ---------------------------------------------------------------------------
# PSA scaling diagnostics (GH #15) -- anchor-free
# ---------------------------------------------------------------------------
#
# These models are the reason PSA exists: exact SSA is infeasible, so there is
# no per-Nc ground truth and the correctness gate can only check cross-engine
# *consistency*, not accuracy. The per-run partial-scaling diagnostics are the
# one signal computable from the scaled run *alone* (no anchor): the measured
# event-rate speedup s-hat, which channels actually scaled and how hard (m-bar),
# and the propensity-weighted excess q-bar attributing variance inflation to
# individual reactions. This pass runs one BNGsim PSA simulation per (model, Nc)
# and digests result.psa_diagnostics into a compact, JSON-serializable row.


def collect_diagnostics(net_path, t_end, n_steps, poplevel, seed):
    """Run one BNGsim PSA simulation and digest its ``psa_diagnostics`` (GH #15).

    Anchor-free -- needs no ``run_network``, so it reports even where the gates
    cannot (exact SSA infeasible). Returns a compact dict of scalars plus the
    top ``DIAG_TOP_K`` variance-inflation channels; ``error`` is a string on
    failure, ``None`` on success. The full per-reaction arrays are intentionally
    NOT stored -- prion has 2809 reactions -- only the digest goes into the JSON.
    """
    import time

    import bngsim
    import numpy as np

    try:
        model = bngsim.Model.from_net(str(net_path))
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="psa", poplevel=poplevel)
        t0 = time.perf_counter()
        r = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
        wall = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return {"error": f"{e}"[:200]}

    p = r.psa_diagnostics
    mbar = np.asarray(p["mbar"], dtype=float)
    qexc = np.asarray(p["qexc"], dtype=float)
    ridx = p["reaction_index"]
    scaled_mask = mbar > 1.0 + 1e-9
    # Channels carrying the most propensity-weighted excess q-bar (the reactions
    # responsible for the variance inflation), largest first.
    order = np.argsort(qexc)[::-1] if qexc.size else []
    top = [
        {"reaction_index": int(ridx[i]), "qexc": float(qexc[i]), "mbar": float(mbar[i])}
        for i in order[:DIAG_TOP_K]
        if qexc[i] > 0.0
    ]
    return {
        "error": None,
        "active": bool(p["active"]),
        "speedup": float(p["speedup"]),
        "scaled_event_integral": float(p["scaled_event_integral"]),
        "exact_event_integral": float(p["exact_event_integral"]),
        "n_reactions": int(mbar.size),
        "n_scaled": int(scaled_mask.sum()),
        "max_mbar": float(mbar.max()) if mbar.size else 1.0,
        "peak_population": float(p["peak_population"]),
        "activation_crossed": bool(p["activation_crossed"]),
        "sim_time": float(p["time"]),
        "wall_time": float(wall),
        "top_channels": top,
    }


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

    _append_diagnostics_section(lines, payload)
    outpath.write_text("\n".join(lines))


def _append_diagnostics_section(lines, payload):
    """Append the anchor-free PSA scaling-diagnostics section (GH #15)."""
    diag_rows = [
        r
        for r in payload["results"]
        if r.get("diagnostics")
        and not r["diagnostics"].get("error")
        and r["diagnostics"].get("active")
    ]
    if not diag_rows:
        return

    lines.append("## PSA scaling diagnostics (BNGsim, anchor-free)\n")
    lines.append(
        "Per-run partial-scaling statistics measured from the scaled run **alone** "
        "-- no exact-SSA anchor, which these models cannot provide. `ŝ` is a per-path "
        "*event-rate* ratio (would-be-exact events ÷ scaled events), an **upper bound** "
        "on wall-clock speedup (per-event overhead erodes it); `m̄` is the dwell-time-"
        "averaged leap multiplier and a channel counts as *scaled* when `m̄_r > 1`; "
        "*activated* means the peak population crossed the `2·Nc` threshold below which "
        "no channel can scale. See GH #15.\n"
    )
    lines.append(
        "| Model | Nc | ŝ (event-rate) | scaled events | would-be exact | scaled rxns | "
        "max m̄ | peak pop | activated |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in payload["results"]:
        d = r.get("diagnostics")
        if not d:
            continue
        if d.get("error"):
            lines.append(
                f"| {r['name']} | {r['poplevel']} | *(error: {d['error'][:40]})* | — | — | — | — | — | — |"
            )
            continue
        if not d.get("active"):
            continue
        lines.append(
            f"| {r['name']} | {r['poplevel']} | {d['speedup']:.1f}× | "
            f"{d['scaled_event_integral']:,.0f} | {d['exact_event_integral']:,.0f} | "
            f"{d['n_scaled']}/{d['n_reactions']} | {d['max_mbar']:,.0f} | "
            f"{d['peak_population']:,.0f} | {'yes' if d['activation_crossed'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "**Top variance-inflation channels** — the reactions with the largest "
        "propensity-weighted excess `q̄_r = (1/T)∫(m_r−1)·a_r dt`, the correct "
        "statistic for attributing inflation to a channel:\n"
    )
    any_top = False
    for r in payload["results"]:
        d = r.get("diagnostics")
        if not d or d.get("error") or not d.get("top_channels"):
            continue
        any_top = True
        chans = ", ".join(
            f"R{c['reaction_index']} (q̄={c['qexc']:.2g}, m̄={c['mbar']:.0f})"
            for c in d["top_channels"]
        )
        lines.append(f"- **{r['name']}** @ Nc={r['poplevel']}: {chans}")
    if not any_top:
        lines.append("- *(no channel scaled at any tested Nc)*")
    lines.append("")


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
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect anchor-free PSA scaling diagnostics (ŝ, m̄, q̄, activation) per "
        "(model, Nc). Needs no run_network. (default: on)",
    )
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

            if args.diagnostics:
                d = collect_diagnostics(net_path, cfg["t_end"], cfg["n_steps"], nc, DIAG_SEED)
                row["diagnostics"] = d
                if d.get("error"):
                    print(f"  diagnostics: ERROR {d['error'][:80]}")
                else:
                    print(
                        f"  diagnostics: ŝ={d['speedup']:.1f}× (event-rate) | "
                        f"scaled {d['n_scaled']}/{d['n_reactions']} rxns | "
                        f"max m̄={d['max_mbar']:,.0f} | activated={d['activation_crossed']}"
                    )

            results.append(row)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for r in results if r.get("correctness", {}).get("passed"))
    n_fail = sum(1 for r in results if r.get("correctness") and not r["correctness"].get("passed"))
    n_skip = sum(1 for r in results if r["status"] == "skip")
    print(f"  jobs: {len(results)}  correctness PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")
    if args.diagnostics:
        shats = [
            r["diagnostics"]["speedup"]
            for r in results
            if r.get("diagnostics")
            and not r["diagnostics"].get("error")
            and r["diagnostics"].get("active")
        ]
        if shats:
            geo_shat = nb.geometric_mean(shats)
            print(
                f"  PSA diagnostics: {len(shats)} jobs measured; "
                f"geometric-mean ŝ (event-rate) {geo_shat:.1f}×"
            )

    payload = {
        "machine_info": nb.machine_info(),
        "mode": args.mode,
        "replicates": args.replicates,
        "warmup": args.warmup,
        "runs": args.runs,
        "diagnostics": args.diagnostics,
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
