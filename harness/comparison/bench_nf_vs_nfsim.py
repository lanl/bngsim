#!/usr/bin/env python3
"""Performance: BNGsim NFsim vs standalone BNG2.pl NFsim (5 models).

Protocol: 2 warmup + 5 timed runs, median wall time.

Usage:
    python bench_nf_vs_nfsim.py [--quick N] [--runs R]

Output:
    results/bench_nf_vs_nfsim.json
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BNG2_PL,
    BNG_TIMEOUT,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    NF_DIR,
    geometric_mean,
    get_machine_info,
    run_bngsim_nfsim,
    save_results,
    timed_runs,
)

SEED = 42

MODELS = {
    "simple_nfsim": {"t_end": 100, "n_steps": 50},
    "basicTLBR": {"t_end": 20, "n_steps": 100},
    "receptor_nf_iter36p0h3": {"t_end": 60, "n_steps": 30},
    "egfr_nf_iter5p12h10": {
        "t_end": 120,
        "n_steps": 60,
        "gml": 1000000,
    },
    "tcr": {"t_end": 60, "n_steps": 30},
}


def strip_actions(content):
    """Remove BNG action lines."""
    pats = [
        r"^\s*(generate_network|simulate|simulate_nf|"
        r"writeXML|writeNetwork|writeSBML|writeMDL|"
        r"resetConcentrations|resetParameters|"
        r"saveConcentrations|setConcentration|setParameter|"
        r"parameter_scan|bifurcate|"
        r"begin\s+actions|end\s+actions)\b.*$",
    ]
    lines = content.split("\n")
    out = []
    for line in lines:
        skip = any(re.match(p, line.strip(), re.MULTILINE) for p in pats)
        if not skip:
            out.append(line)
    return "\n".join(out)


def prepare_xml(bngl_path, work_dir):
    """Generate XML via BNG2.pl writeXML()."""
    stem = bngl_path.stem
    with open(bngl_path) as f:
        content = f.read()
    clean = strip_actions(content).rstrip()
    clean += "\n\nwriteXML()\n"
    mod = work_dir / f"{stem}_xml.bngl"
    mod.write_text(clean)
    subprocess.run(
        ["perl", BNG2_PL, str(mod)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=BNG_TIMEOUT,
    )
    xml = work_dir / f"{stem}_xml.xml"
    if not xml.exists():
        raise FileNotFoundError(f"writeXML failed: {stem}")
    return xml


def run_bng_nfsim_timed(bngl_path, work_dir, t_end, n_steps, seed):
    """Run BNG2.pl NFsim and return timing dict."""
    stem = bngl_path.stem
    with open(bngl_path) as f:
        content = f.read()
    clean = strip_actions(content).rstrip()
    action = (
        f'simulate({{method=>"nf",t_end=>{t_end},n_steps=>{n_steps},seed=>{seed},gml=>1000000}})\n'
    )
    clean += "\n\n" + action
    mod = work_dir / f"{stem}_sim.bngl"
    mod.write_text(clean)

    t0 = time.perf_counter()
    subprocess.run(
        ["perl", BNG2_PL, str(mod)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=BNG_TIMEOUT,
    )
    elapsed = time.perf_counter() - t0

    gdat = work_dir / f"{stem}_sim.gdat"
    if not gdat.exists():
        return {"wall_time": elapsed, "error": "no gdat"}
    return {"wall_time": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Benchmark BNGsim NFsim vs BNG2.pl")
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Performance: BNGsim NFsim vs BNG2.pl NFsim")
    print(f"  Protocol: {args.warmup} warmup + {args.runs} timed")
    print("=" * 70)

    info = get_machine_info()
    model_list = list(MODELS.items())
    if args.quick > 0:
        model_list = model_list[: args.quick]

    results = []
    speedups = []

    for name, cfg in model_list:
        bngl = NF_DIR / f"{name}.bngl"
        t_end = cfg["t_end"]
        n_steps = cfg["n_steps"]
        gml = cfg.get("gml", 1000000)

        print(f"\n  {name}...")

        if not bngl.exists():
            print("    SKIP: not found")
            results.append({"model": name, "status": "skip"})
            continue

        entry = {"model": name}

        with tempfile.TemporaryDirectory(prefix=f"nf_{name}_") as tmpdir:
            wd = Path(tmpdir)
            try:
                xml = prepare_xml(bngl, wd)
            except Exception as e:
                print(f"    ERROR: XML gen: {e}")
                entry["status"] = "xml_fail"
                results.append(entry)
                continue

            # BNGsim NFsim timing
            bng = timed_runs(
                lambda x=str(xml), t=t_end, s=n_steps, g=gml: run_bngsim_nfsim(
                    x, t, s, SEED, gml=g
                ),
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
            print(f"    BNGsim median: {bng['median_time']:.3f}s")

            # BNG2.pl NFsim timing
            rn = timed_runs(
                lambda b=bngl, w=wd, t=t_end, s=n_steps: run_bng_nfsim_timed(b, w, t, s, SEED),
                n_warmup=args.warmup,
                n_runs=args.runs,
                verbose=True,
            )
            if "error" in rn:
                print(f"    BNG2.pl ERROR: {rn['error'][:60]}")
                entry["bng_time"] = -1
                results.append(entry)
                continue
            entry["bng_time"] = rn["median_time"]
            print(f"    BNG2.pl median: {rn['median_time']:.3f}s")

            bt = bng["median_time"]
            rt = rn["median_time"]
            if bt > 0 and rt > 0:
                su = rt / bt
                entry["speedup"] = su
                speedups.append(su)
                print(f"    Speedup: {su:.1f}x")

        entry["status"] = "ok"
        results.append(entry)

    # Summary
    print(f"\n{'=' * 70}")
    if speedups:
        gm = geometric_mean(speedups)
        print(f"  Geometric mean speedup: {gm:.1f}x")
    print(f"  Models benchmarked: {len(speedups)}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "summary": {
            "n_models": len(speedups),
            "geometric_mean_speedup": (geometric_mean(speedups) if speedups else None),
        },
        "results": results,
    }
    save_results(output, "bench_nf_vs_nfsim")


if __name__ == "__main__":
    main()
