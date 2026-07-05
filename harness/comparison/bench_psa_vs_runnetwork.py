#!/usr/bin/env python3
"""Performance: BNGsim PSA vs run_network PSA (3 models x 2 Nc).

When a model entry in suite_psa.json carries ``sbml_file`` (cross-referenced
into ``benchmarks/suites/sbml_roundtrip/``) a third timed arm runs PSA via the SBML loader,
recording ``bngsim_sbml_time`` and the ``sbml_vs_net_ratio`` overhead vs. the
.net path (additive schema; .net + run_network arms unchanged).

Protocol: 2 warmup + 5 timed runs, median wall time.

Usage:
    python bench_psa_vs_runnetwork.py [--quick N] [--runs R]

Output:
    results/bench_psa_vs_runnetwork.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    SUITE_PSA,
    geometric_mean,
    get_machine_info,
    load_suite,
    run_bngsim_psa,
    run_bngsim_psa_sbml,
    run_rn_psa,
    save_results,
    timed_runs,
)

SEED_BASE = 2000


def main():
    parser = argparse.ArgumentParser(description="Benchmark BNGsim PSA vs run_network")
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = parser.parse_args()

    print("=" * 70)
    print("  Performance: BNGsim PSA vs run_network")
    print(f"  Protocol: {args.warmup} warmup + {args.runs} timed")
    print("=" * 70)

    info = get_machine_info()
    models = load_suite(SUITE_PSA)
    if args.quick > 0:
        models = models[: args.quick]

    results = []
    speedups = []

    for _i, m in enumerate(models):
        name = m["name"]
        net = str(BENCHMARKS_DIR / m["net_file"])
        sbml = str(BENCHMARKS_DIR / m["sbml_file"]) if m.get("sbml_file") else None
        t_end = m["t_end"]
        n_steps = m["n_steps"]
        poplevels = m.get("poplevel", [10, 100])

        for nc in poplevels:
            label = f"{name} Nc={nc}"
            print(f"\n  [{label}] ({m['species']} sp, {m['reactions']} rxn)")

            entry = {
                "model": name,
                "poplevel": nc,
                "species": m["species"],
                "reactions": m["reactions"],
            }

            # BNGsim PSA (.net path)
            seed_i = [0]

            def bng_run(n=net, t=t_end, s=n_steps, pl=nc, _seed=seed_i):
                _seed[0] += 1
                return run_bngsim_psa(
                    n,
                    t,
                    s,
                    SEED_BASE + _seed[0],
                    pl,
                )

            bng = timed_runs(
                bng_run,
                n_warmup=args.warmup,
                n_runs=args.runs,
                verbose=True,
            )
            if "error" in bng:
                print(f"    BNGsim ERROR: {bng['error'][:60]}")
                entry["bngsim_time"] = -1
                results.append(entry)
                continue
            entry["bngsim_time"] = bng["median_time"]
            print(f"    BNGsim median: {bng['median_time']:.4f}s")

            # BNGsim PSA via SBML (additive arm; only when sbml_file is present)
            if sbml is not None:
                seed_k = [0]

                def bng_sbml_run(x=sbml, t=t_end, s=n_steps, pl=nc, _seed=seed_k):
                    _seed[0] += 1
                    return run_bngsim_psa_sbml(
                        x,
                        t,
                        s,
                        SEED_BASE + _seed[0],
                        pl,
                    )

                bng_sbml = timed_runs(
                    bng_sbml_run,
                    n_warmup=args.warmup,
                    n_runs=args.runs,
                    verbose=True,
                )
                if "error" in bng_sbml:
                    print(f"    BNGsim SBML ERROR: {bng_sbml['error'][:60]}")
                    entry["bngsim_sbml_time"] = -1
                else:
                    entry["bngsim_sbml_time"] = bng_sbml["median_time"]
                    if bng["median_time"] > 0:
                        entry["sbml_vs_net_ratio"] = bng_sbml["median_time"] / bng["median_time"]
                    print(
                        f"    BNGsim SBML median: {bng_sbml['median_time']:.4f}s"
                        + (
                            f" (sbml/net={entry['sbml_vs_net_ratio']:.2f}x)"
                            if "sbml_vs_net_ratio" in entry
                            else ""
                        )
                    )

            # run_network PSA
            seed_j = [0]

            def rn_run(n=net, t=t_end, s=n_steps, pl=nc, _seed=seed_j):
                _seed[0] += 1
                return run_rn_psa(
                    n,
                    t,
                    s,
                    SEED_BASE + _seed[0],
                    pl,
                )

            rn = timed_runs(
                rn_run,
                n_warmup=args.warmup,
                n_runs=args.runs,
                verbose=True,
            )
            if "error" in rn:
                print(f"    run_network ERROR: {rn['error'][:60]}")
                entry["rn_time"] = -1
                results.append(entry)
                continue
            entry["rn_time"] = rn["median_time"]
            print(f"    run_network median: {rn['median_time']:.4f}s")

            # Speedup
            bt = bng["median_time"]
            rt = rn["median_time"]
            if bt > 0 and rt > 0:
                su = rt / bt
                entry["speedup"] = su
                speedups.append(su)
                print(f"    Speedup: {su:.1f}x")

            results.append(entry)

    # Summary
    print(f"\n{'=' * 70}")
    if speedups:
        gm = geometric_mean(speedups)
        print(f"  Geometric mean speedup: {gm:.1f}x")
    n_cfg = sum(len(m.get("poplevel", [10, 100])) for m in models)
    print(f"  Configs benchmarked: {len(speedups)}/{n_cfg}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "summary": {
            "n_configs": len(speedups),
            "geometric_mean_speedup": (geometric_mean(speedups) if speedups else None),
        },
        "results": results,
    }
    save_results(output, "bench_psa_vs_runnetwork")


if __name__ == "__main__":
    main()
