#!/usr/bin/env python3
"""BNGsim-only warm PSA timing for the ``psa`` suite -- a companion to ``run.py``.

``run.py`` reports a cross-engine timing only for a ``(model, Nc)`` whose
correctness gate passed (its design rule). When the *competitor* engine cannot
produce a faithful ensemble -- e.g. ``run_network``'s heterogeneous adaptive
scaling crashes intermittently on the 2809-reaction ``prion_aggregation`` network,
so a clean 20-replicate ensemble is essentially unobtainable -- that gate never
passes and ``run.py`` records no timing, even though BNGsim itself runs the model
fine on every ``Nc``.

This script measures BNGsim's *own* warm PSA cost for every ``(model, Nc)``, using
the identical ``run_bngsim`` protocol (``warmup`` discarded passes + ``runs`` timed
passes, median) that ``run.py``'s cross-engine timing uses for its BNGsim column.
It needs no ``run_network`` and is independent of the correctness gate, so it covers
the cells ``run.py`` leaves blank. The paper's ``generate_psa_table.py`` uses this as
the BNGsim(s) source for those cells (Table 7).

    python run_bngsim_timing.py            # all models x Nc (BNGsim only)

Writes ``results/psa_bngsim_timing.json``.
"""

import json
import sys
from pathlib import Path
from statistics import median

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run import MODELS, NET_DIR, RESULTS_DIR  # noqa: E402  (shared registry)


def main():
    warmup, runs = nb.DEFAULT_WARMUP, nb.DEFAULT_RUNS
    print("=" * 70)
    print("  PSA suite -- BNGsim-only warm timing (companion to run.py)")
    print("=" * 70)
    print(f"  protocol: {warmup} warmup + {runs} timed runs, median; BNGsim only")

    results = []
    for cfg in MODELS:
        name = cfg["name"]
        net_path = NET_DIR / f"{name}.net"
        print(f"\n=== {name} ({cfg['species']} sp, {cfg['reactions']} rxn) ===")
        for nc in cfg["poplevels"]:
            row = {
                "name": name,
                "poplevel": nc,
                "species": cfg["species"],
                "reactions": cfg["reactions"],
                "t_end": cfg["t_end"],
                "n_steps": cfg["n_steps"],
            }
            if not net_path.exists():
                row["error"] = ".net not found"
                results.append(row)
                print(f"  Nc={nc:>4}: SKIP (.net not found)")
                continue

            times, steps, err = [], [], None
            for i in range(warmup + runs):
                r = nb.run_bngsim(net_path, "psa", cfg["t_end"], cfg["n_steps"], 1000 + i, nc)
                if r["error"]:
                    err = r["error"]
                    break
                if i >= warmup:
                    times.append(r["wall_time"])
                    steps.append(r["steps"])
            if err:
                row["error"] = err
                print(f"  Nc={nc:>4}: ERROR {err[:80]}")
            else:
                row["error"] = None
                row["bngsim_median_sec"] = median(times)
                row["bngsim_median_steps"] = int(median(steps)) if steps else -1
                row["all_times"] = times
                print(f"  Nc={nc:>4}: bngsim warm median = {row['bngsim_median_sec']:.4f}s")
            results.append(row)

    payload = {
        "machine_info": nb.machine_info(),
        "metric": "bngsim warm PSA cost = median wall (s) over warmup+runs timed passes",
        "warmup": warmup,
        "runs": runs,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "psa_bngsim_timing.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
