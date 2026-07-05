#!/usr/bin/env python3
"""Performance: Four-engine ODE comparison on Pool C (27 BNG models).

Compares BNGsim vs run_network vs libRoadRunner vs AMICI on the same models.
BNGsim and run_network use .net files; RR and AMICI use SBML converted via
BNG2.pl writeSBML().

Prerequisites:
    1. Run convert_bngl_to_sbml.py first to produce models/bng_sbml/*.xml
    2. AMICI models are compiled on first run and cached in models/amici_cache/

Protocol: 2 warmup + 5 timed runs, median wall time.

Usage:
    python bench_ode_4engine.py [--quick N] [--runs R] [--warmup W]

Output:
    results/bench_ode_4engine.json
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    HARNESS_DIR,
    SUITE_ODE,
    AmiciApiError,
    amici_prepare_model,
    amici_simulate,
    geometric_mean,
    get_machine_info,
    load_suite,
    run_bngsim_ode,
    run_rn_ode,
    run_roadrunner_ode_sbml,
    save_results,
    timed_runs,
)

SBML_DIR = HARNESS_DIR / "models" / "bng_sbml"
AMICI_CACHE = HARNESS_DIR / "models" / "amici_cache"


# ---------------------------------------------------------------------------
# AMICI helpers
# ---------------------------------------------------------------------------


def compile_amici_model(sbml_path, model_name):
    """Compile SBML file to AMICI model module. Returns module or None."""
    import amici

    with open(sbml_path) as f:
        sbml_str = f.read()

    h = hashlib.sha256(sbml_str.encode()).hexdigest()[:12]
    safe_name = f"amici_{model_name}_{h}"
    model_dir = AMICI_CACHE / safe_name

    if model_dir.exists():
        try:
            mod = amici.import_model_module(safe_name, str(model_dir))
            return mod
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)

    AMICI_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        sbml_importer = amici.SbmlImporter(sbml_str, from_file=False)
        sbml_importer.sbml2amici(safe_name, str(model_dir), verbose=False)
        mod = amici.import_model_module(safe_name, str(model_dir))
        return mod
    except Exception as e:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise RuntimeError(f"AMICI compile failed: {e}") from e


def run_amici_ode(amici_module, t_end, n_steps):
    """Run AMICI ODE simulation (AMICI >= 1.0 snake_case API). Returns timing dict.

    ``wall_time`` times only ``model.simulate()``. ``AmiciApiError`` (API drift)
    is re-raised to crash loudly rather than degrade to a silent error row
    (GH #227); a genuine per-model integration failure becomes an error row.
    """
    try:
        model, solver = amici_prepare_model(amici_module)
        tspan = np.linspace(0, t_end, n_steps + 1)
        rdata, elapsed = amici_simulate(model, solver, tspan)
        return {
            "wall_time": elapsed,
            "species": np.array(rdata.x) if rdata.x is not None else None,
        }
    except AmiciApiError:
        raise
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Four-engine ODE benchmark (Pool C)")
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--skip-amici", action="store_true", help="Skip AMICI (slow compilation)")
    args = parser.parse_args()

    # Check prerequisites
    have_amici = False
    if not args.skip_amici:
        try:
            import amici

            have_amici = True
        except ImportError:
            print("  WARNING: AMICI not installed, skipping AMICI column")

    try:
        import roadrunner

        have_rr = True
    except ImportError:
        have_rr = False
        print("  WARNING: libRoadRunner not installed, skipping RR column")

    print("=" * 70)
    print("  Four-Engine ODE Benchmark (Pool C)")
    print(f"  Protocol: {args.warmup} warmup + {args.runs} timed")
    print(
        f"  Engines: BNGsim, run_network"
        f"{', RoadRunner' if have_rr else ''}"
        f"{', AMICI' if have_amici else ''}"
    )
    print("=" * 70)

    info = get_machine_info()
    if have_rr:
        import roadrunner

        info["roadrunner_version"] = roadrunner.__version__
    if have_amici:
        import amici

        info["amici_version"] = amici.__version__

    models = load_suite(SUITE_ODE)
    if args.quick > 0:
        models = models[: args.quick]

    # Discover SBML files directly from disk (produced by convert_bngl_to_sbml.py)
    sbml_manifest = {}
    for m in models:
        sbml_path = SBML_DIR / f"{m['name']}.xml"
        if sbml_path.exists():
            sbml_manifest[m["name"]] = str(sbml_path)
    print(f"  SBML available: {len(sbml_manifest)}/{len(models)} models")

    # Pre-compile AMICI models (excluded from timing)
    amici_modules = {}
    if have_amici:
        print("\n  Pre-compiling AMICI models...")
        for m in models:
            name = m["name"]
            sbml_path = sbml_manifest.get(name)
            if not sbml_path or not Path(sbml_path).exists():
                continue
            try:
                amici_modules[name] = compile_amici_model(sbml_path, name)
                print(f"    {name}: compiled")
            except Exception as e:
                print(f"    {name}: FAIL ({str(e)[:60]})")
        print(f"  AMICI compiled: {len(amici_modules)}/{len(sbml_manifest)}")

    print()

    results = []
    speedups_rn = []
    speedups_rr = []
    speedups_amici = []

    for i, m in enumerate(models):
        name = m["name"]
        net = str(BENCHMARKS_DIR / m["net_file"])
        t_end = m["t_end"]
        n_steps = m["n_steps"]
        sbml_path = sbml_manifest.get(name)

        print(f"\n  [{i + 1}/{len(models)}] {name} ({m['species']} sp, {m['reactions']} rxn)")

        entry = {
            "model": name,
            "species": m["species"],
            "reactions": m["reactions"],
            "t_end": t_end,
            "n_steps": n_steps,
            "has_sbml": sbml_path is not None,
        }

        # --- BNGsim ---
        print("    BNGsim...", end=" ", flush=True)
        bng = timed_runs(
            lambda n=net, t=t_end, s=n_steps: run_bngsim_ode(n, t, s),
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
        if "error" in bng:
            print(f"ERROR: {bng['error'][:60]}")
            entry["bngsim_time"] = -1
            entry["bngsim_error"] = bng["error"]
            results.append(entry)
            continue
        entry["bngsim_time"] = bng["median_time"]
        print(f"{bng['median_time']:.4f}s")

        # --- run_network ---
        print("    run_network...", end=" ", flush=True)
        rn = timed_runs(
            lambda n=net, t=t_end, s=n_steps: run_rn_ode(n, t, s),
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
        if "error" in rn:
            print(f"ERROR: {rn['error'][:60]}")
            entry["rn_time"] = -1
        else:
            entry["rn_time"] = rn["median_time"]
            print(f"{rn['median_time']:.4f}s")
            if bng["median_time"] > 0 and rn["median_time"] > 0:
                su = rn["median_time"] / bng["median_time"]
                entry["speedup_vs_rn"] = su
                speedups_rn.append(su)

        # --- libRoadRunner ---
        # Skip SBML files > 5MB (RoadRunner XML parser chokes on huge files)
        sbml_too_large = (
            sbml_path and Path(sbml_path).exists() and Path(sbml_path).stat().st_size > 5_000_000
        )
        if sbml_too_large:
            entry["rr_time"] = None
            entry["rr_note"] = f"SBML too large ({Path(sbml_path).stat().st_size // 1_000_000}MB)"
            entry["amici_time"] = None
            entry["amici_note"] = "SBML too large"
            print("    RoadRunner... SKIP (SBML too large)")
            print("    AMICI... SKIP (SBML too large)")
            results.append(entry)
            continue

        if have_rr and sbml_path and Path(sbml_path).exists():
            print("    RoadRunner...", end=" ", flush=True)
            rr = timed_runs(
                lambda p=sbml_path, t=t_end, s=n_steps: run_roadrunner_ode_sbml(p, t, s),
                n_warmup=args.warmup,
                n_runs=args.runs,
            )
            if "error" in rr:
                print(f"ERROR: {rr['error'][:60]}")
                entry["rr_time"] = -1
                entry["rr_error"] = rr["error"]
            else:
                entry["rr_time"] = rr["median_time"]
                print(f"{rr['median_time']:.4f}s")
                if bng["median_time"] > 0 and rr["median_time"] > 0:
                    ratio = bng["median_time"] / rr["median_time"]
                    entry["ratio_bng_rr"] = ratio
                    speedups_rr.append(ratio)
        else:
            entry["rr_time"] = None
            entry["rr_note"] = "no SBML" if not sbml_path else "RR not installed"

        # --- AMICI ---
        if have_amici and name in amici_modules:
            print("    AMICI...", end=" ", flush=True)
            amici_mod = amici_modules[name]
            am = timed_runs(
                lambda mod=amici_mod, t=t_end, s=n_steps: run_amici_ode(mod, t, s),
                n_warmup=args.warmup,
                n_runs=args.runs,
            )
            if "error" in am:
                print(f"ERROR: {am['error'][:60]}")
                entry["amici_time"] = -1
                entry["amici_error"] = am["error"]
            else:
                entry["amici_time"] = am["median_time"]
                print(f"{am['median_time']:.4f}s")
                if bng["median_time"] > 0 and am["median_time"] > 0:
                    ratio = bng["median_time"] / am["median_time"]
                    entry["ratio_bng_amici"] = ratio
                    speedups_amici.append(ratio)
        else:
            entry["amici_time"] = None
            entry["amici_note"] = (
                "no SBML"
                if not sbml_path
                else "compile failed"
                if name not in amici_modules
                else "AMICI not installed"
            )

        # Summary line
        parts = [f"BNG={bng['median_time']:.4f}s"]
        if entry.get("rn_time") and entry["rn_time"] > 0:
            parts.append(f"RN={entry['rn_time']:.4f}s")
        if entry.get("rr_time") and entry["rr_time"] > 0:
            parts.append(f"RR={entry['rr_time']:.4f}s")
        if entry.get("amici_time") and entry["amici_time"] > 0:
            parts.append(f"AMICI={entry['amici_time']:.4f}s")
        su_rn = entry.get("speedup_vs_rn")
        if su_rn:
            parts.append(f"vs_RN={su_rn:.1f}x")
        print(f"    → {' | '.join(parts)}")

        results.append(entry)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    if speedups_rn:
        gm = geometric_mean(speedups_rn)
        print(f"  BNGsim vs run_network: {gm:.1f}x geomean ({len(speedups_rn)} models)")

    if speedups_rr:
        gm = geometric_mean(speedups_rr)
        n_bng = sum(1 for s in speedups_rr if s < 1)
        n_rr = sum(1 for s in speedups_rr if s >= 1)
        print(f"  BNGsim/RR ratio: {gm:.2f} geomean (BNG wins {n_bng}, RR wins {n_rr})")

    if speedups_amici:
        gm = geometric_mean(speedups_amici)
        n_bng = sum(1 for s in speedups_amici if s < 1)
        n_amici = sum(1 for s in speedups_amici if s >= 1)
        print(f"  BNGsim/AMICI ratio: {gm:.2f} geomean (BNG wins {n_bng}, AMICI wins {n_amici})")

    print(f"  Models benchmarked: {len(results)}/{len(models)}")
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "summary": {
            "n_models": len(results),
            "n_with_sbml": sum(1 for r in results if r.get("has_sbml")),
            "speedup_vs_rn_geomean": (geometric_mean(speedups_rn) if speedups_rn else None),
            "ratio_bng_rr_geomean": (geometric_mean(speedups_rr) if speedups_rr else None),
            "ratio_bng_amici_geomean": (
                geometric_mean(speedups_amici) if speedups_amici else None
            ),
        },
        "results": results,
    }
    save_results(output, "bench_ode_4engine")


if __name__ == "__main__":
    main()
