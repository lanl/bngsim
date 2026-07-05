#!/usr/bin/env python3
"""Performance: BNGsim ODE vs AMICI (Antimony models via SBML).

AMICI workflow per model:
1. Antimony → SBML string (via libantimony)
2. SBML → C code generation → compile shared library (one-time)
3. Timed: repeated ODE simulations using compiled model

Compilation is excluded from timing (analogous to BNGsim model loading).

Usage:
    python bench_ode_vs_amici.py [--pool a|b|ab] [--quick N]

Output:
    results/bench_ode_vs_amici.json
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    RESULTS_DIR,
    AmiciApiError,
    _ant_to_sbml,
    amici_prepare_model,
    amici_simulate,
    discover_pool_a,
    discover_pool_b,
    geometric_mean,
    get_machine_info,
    run_bngsim_ode_antimony,
    save_results,
    timed_runs,
)

AMICI_CACHE = Path(__file__).resolve().parent.parent / "models" / "amici_cache"
FALLBACK_T_END = 1.0
FALLBACK_N_STEPS = 200


def load_horizons():
    """Load calibrated horizons if available."""
    path = RESULTS_DIR / "calibrated_horizons.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    horizons = {}
    for m in data.get("models", []):
        if m.get("status") == "ok":
            horizons[m["model"]] = {
                "t_end": m["t_end"],
                "n_steps": m.get("n_steps", 200),
            }
    return horizons


def compile_amici_model(sbml_str, model_name):
    """Compile SBML to AMICI model module. Returns module or None."""
    import amici

    # Use content hash for caching
    h = hashlib.sha256(sbml_str.encode()).hexdigest()[:12]
    safe_name = f"amici_{model_name}_{h}"
    model_dir = AMICI_CACHE / safe_name

    if model_dir.exists():
        # Try loading cached model
        try:
            mod = amici.import_model_module(
                safe_name,
                str(model_dir),
            )
            return mod
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)

    # Generate and compile
    AMICI_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        sbml_importer = amici.SbmlImporter(
            sbml_str,
            from_file=False,
        )
        sbml_importer.sbml2amici(
            safe_name,
            str(model_dir),
            verbose=False,
        )
        mod = amici.import_model_module(
            safe_name,
            str(model_dir),
        )
        return mod
    except Exception as e:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise RuntimeError(f"AMICI compile failed: {e}") from e


def run_amici_ode(amici_module, t_end, n_steps):
    """Run AMICI ODE simulation (AMICI >= 1.0 snake_case API). Returns timing dict.

    ``wall_time`` times only the ``model.simulate()`` call, excluding the
    get_model/create_solver setup — matching the old camelCase driver's timed
    region. An ``AmiciApiError`` (API drift) is re-raised so it crashes loudly
    instead of degrading to a silent error row (GH #227); a genuine per-model
    integration failure is caught and reported as an error row.
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark BNGsim vs AMICI")
    parser.add_argument(
        "--pool",
        default="b",
        choices=["a", "b", "ab"],
    )
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )
    args = parser.parse_args()

    try:
        import amici

        print(f"  AMICI version: {amici.__version__}")
    except ImportError:
        print("ERROR: AMICI not installed")
        print("  Install with: uv pip install amici")
        sys.exit(1)

    print("=" * 70)
    print("  Performance: BNGsim ODE vs AMICI")
    print(f"  Protocol: {args.warmup} warmup + {args.runs} timed")
    print("=" * 70)

    info = get_machine_info()
    info["amici_version"] = amici.__version__
    horizons = load_horizons()

    models = []
    if "a" in args.pool:
        pa = discover_pool_a()
        if args.quick > 0:
            pa = pa[: args.quick]
        models.extend([(p, "A") for p in pa])
        print(f"  Pool A: {len(pa)} models")
    if "b" in args.pool:
        pb = discover_pool_b(require_xval=True)
        if args.quick > 0:
            pb = pb[: args.quick]
        models.extend([(p, "B") for p in pb])
        print(f"  Pool B: {len(pb)} models")

    print(f"  Total: {len(models)}\n")

    results = []
    ratios = []
    n_skip = 0
    n_compile_fail = 0

    for i, (path, pool) in enumerate(models):
        name = path.stem
        h = horizons.get(name, {})
        t_end = h.get("t_end", FALLBACK_T_END)
        n_steps = h.get("n_steps", FALLBACK_N_STEPS)

        print(
            f"  [{i + 1}/{len(models)}] {pool}/{name} (t={t_end:.1g})...",
            end=" ",
            flush=True,
        )

        entry = {
            "model": name,
            "pool": pool,
            "t_end": t_end,
            "n_steps": n_steps,
        }

        # Step 1: Antimony → SBML
        try:
            sbml = _ant_to_sbml(str(path))
        except Exception as e:
            entry["status"] = "ant_fail"
            entry["error"] = str(e)[:200]
            results.append(entry)
            n_skip += 1
            print("ANT_FAIL")
            continue

        # Step 2: Compile AMICI model (one-time, excluded from timing)
        try:
            amici_mod = compile_amici_model(sbml, name)
        except Exception as e:
            entry["status"] = "compile_fail"
            entry["error"] = str(e)[:200]
            results.append(entry)
            n_compile_fail += 1
            print("COMPILE_FAIL")
            continue

        # Step 3: Time AMICI
        amici_r = timed_runs(
            lambda m=amici_mod, t=t_end, s=n_steps: run_amici_ode(m, t, s),
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
        if "error" in amici_r:
            entry["status"] = "amici_fail"
            entry["error"] = amici_r["error"]
            results.append(entry)
            n_skip += 1
            print("AMICI_FAIL")
            continue
        entry["amici_time"] = amici_r["median_time"]

        # Step 4: Time BNGsim
        bng = timed_runs(
            lambda p=str(path), t=t_end, s=n_steps: run_bngsim_ode_antimony(p, t, s),
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
        if "error" in bng:
            entry["status"] = "bngsim_fail"
            entry["error"] = bng["error"]
            results.append(entry)
            n_skip += 1
            print("BNG_FAIL")
            continue
        entry["bngsim_time"] = bng["median_time"]
        entry["status"] = "ok"

        bt = bng["median_time"]
        at = amici_r["median_time"]
        if bt > 0 and at > 0:
            ratio = bt / at  # <1 means BNGsim faster
            entry["ratio_bng_amici"] = ratio
            ratios.append(ratio)
            faster = "BNG" if ratio < 1 else "AMICI"
            factor = 1 / ratio if ratio < 1 else ratio
            print(f"BNG={bt:.4f}s AMICI={at:.4f}s ({faster} {factor:.1f}x)")
        else:
            print(f"BNG={bt:.4f}s AMICI={at:.4f}s")

        results.append(entry)

    # Summary
    print(f"\n{'=' * 70}")
    n_ok = len(ratios)
    if ratios:
        gm = geometric_mean(ratios)
        n_bng = sum(1 for r in ratios if r < 1)
        n_amici = sum(1 for r in ratios if r >= 1)
        print(f"  BNGsim/AMICI ratio geometric mean: {gm:.2f}")
        print(f"  BNGsim faster: {n_bng}  AMICI faster: {n_amici}")
    print(
        f"  Benchmarked: {n_ok}  "
        f"Compile fail: {n_compile_fail}  "
        f"Other skip: {n_skip}  "
        f"Total: {len(models)}"
    )
    print(f"{'=' * 70}")

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "summary": {
            "n_benchmarked": n_ok,
            "n_compile_fail": n_compile_fail,
            "n_skipped": n_skip,
            "ratio_geometric_mean": (geometric_mean(ratios) if ratios else None),
        },
        "results": results,
    }
    save_results(output, "bench_ode_vs_amici")


if __name__ == "__main__":
    main()
