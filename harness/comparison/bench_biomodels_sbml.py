#!/usr/bin/env python3
"""BioModels large-scale benchmark: BNGsim vs libRoadRunner.

Uses BioModels Antimony files from ``BENCH_ANT_DIR`` (default: ``biomodels_ant/`` pool root).
BNGsim loads via from_antimony() (routes through SBML internally).
RR loads via ant->SBML conversion then roadrunner.RoadRunner(sbml).

Three phases (with checkpointing):
  Phase 1: Load — bngsim.Model.from_antimony() on each Antimony model (default: ~1013 manifest union)
  Phase 2: Cross-validate — run both engines at t=1.0, compare trajectories
  Phase 3: Time — 2 warmup + 5 timed runs, median wall time

Usage:
    python bench_biomodels_sbml.py                    # full run
    python bench_biomodels_sbml.py --quick 20         # first 20 models
    python bench_biomodels_sbml.py --phase 1          # load gate only
    python bench_biomodels_sbml.py --phase 2          # xval (needs phase 1)
    python bench_biomodels_sbml.py --phase 3          # timing (needs phase 2)
    python bench_biomodels_sbml.py --resume            # resume from checkpoint

Output:
    results/bench_biomodels_sbml.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    RESULTS_DIR,
    _ant_to_sbml,
    add_bngsim_timeout_arg,
    format_bngsim_timeout,
    get_machine_info,
    save_results,
)

# ── Paths ─────────────────────────────────────────────────────────────────

_DEFAULT_BIOMODELS_ANT = (
    Path(__file__).resolve().parent.parent.parent / "benchmarks" / "biomodels_ant"
)
ANT_DIR = Path(os.environ.get("BENCH_ANT_DIR", str(_DEFAULT_BIOMODELS_ANT))).expanduser().resolve()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _biomodels_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "benchmarks"
        / "biomodels_ssa"
        / "biomodels_ant_pool_manifest.json"
    )


# ── Constants ─────────────────────────────────────────────────────────────

XVAL_RTOL = 1e-3
XVAL_ATOL = 1e-8
NEAR_ZERO_THRESHOLD = 1e-8
ADAPTIVE_HORIZONS = [1.0, 0.1, 0.01, 0.001]
N_STEPS = 200
_WARMUP = 2
_RUNS = 5


# ── Helpers ───────────────────────────────────────────────────────────────


def geometric_mean(values):
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in values) / len(values)))


def _existing_manifest_paths(ids: list[str]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    missing: list[str] = []
    for mid in ids:
        candidates = (
            ANT_DIR / f"{mid}.ant",
            ANT_DIR / "review_extra" / f"{mid}.ant",
        )
        chosen = next((p for p in candidates if p.is_file()), None)
        if chosen is None:
            missing.append(mid)
        else:
            files.append(chosen)
    return files, missing


def discover_ant_files(*, quick_limit: int = 0, allow_partial_manifest: bool = False):
    limit = quick_limit if quick_limit > 0 else None
    if ANT_DIR.exists() and not ANT_DIR.is_dir():
        print(f"ERROR: BENCH_ANT_DIR is not a directory: {ANT_DIR}")
        sys.exit(1)
    if not ANT_DIR.exists() and not _biomodels_manifest_path().is_file():
        print(f"ERROR: ANT directory not found: {ANT_DIR}")
        sys.exit(1)
    mp = _biomodels_manifest_path()
    if mp.is_file():
        bm_root = mp.parent
        if str(bm_root) not in sys.path:
            sys.path.insert(0, str(bm_root))
        try:
            from biomodels_ant_pool import load_manifest as _load_manifest  # noqa: PLC0415
            from biomodels_ant_pool import required_ids as _required_ids  # noqa: PLC0415
            from biomodels_ant_pool import resolve_manifest_ant_paths  # noqa: PLC0415

            paths = resolve_manifest_ant_paths(
                ANT_DIR,
                include_review_extras=True,
                limit=limit,
            )
        except FileNotFoundError as e:
            if not allow_partial_manifest:
                print(f"ERROR: manifest pool resolution failed: {e}", file=sys.stderr)
                sys.exit(1)
            man = _load_manifest()
            ids = _required_ids(man, include_review_extras=True)
            if limit is not None:
                ids = ids[:limit]
            present_files, missing_ids = _existing_manifest_paths(ids)
            if not present_files:
                print(f"ERROR: manifest pool resolution failed: {e}", file=sys.stderr)
                sys.exit(1)
            print(
                "  WARNING: manifest pool is incomplete; proceeding with available subset "
                f"({len(present_files)}/{len(ids)} present, {len(missing_ids)} missing)."
            )
            paths = present_files
        except (ValueError, json.JSONDecodeError) as e:
            print(f"ERROR: manifest pool resolution failed: {e}", file=sys.stderr)
            sys.exit(1)
        if paths is not None:
            return paths
    top = sorted(ANT_DIR.glob("BIOMD*.ant")) + sorted(ANT_DIR.glob("MODEL*.ant"))
    rex = ANT_DIR / "review_extra"
    extra = sorted(rex.glob("BIOMD*.ant")) + sorted(rex.glob("MODEL*.ant")) if rex.is_dir() else []
    files = sorted({*top, *extra}, key=lambda p: p.name)
    if limit:
        files = files[:limit]
    return files


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_checkpoint(data, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"  Checkpoint: {path}")
    return path


def load_checkpoint(name):
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Phase 1: Load Gate ───────────────────────────────────────────────────


def phase1_load(ant_files, resume_data=None):
    import bngsim

    print("=" * 70)
    print("  Phase 1: Load Gate (BNGsim Model.from_antimony)")
    print(f"  Models: {len(ant_files)}")
    print("=" * 70)

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r
        print(f"  Resuming: {len(done)} already done")

    results = []
    n_ok = n_fail = 0

    for i, ant_path in enumerate(ant_files):
        model_id = ant_path.stem

        if model_id in done:
            results.append(done[model_id])
            if done[model_id]["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            continue

        entry = {"model": model_id, "ant_path": str(ant_path)}
        try:
            t0 = time.perf_counter()
            model = bngsim.Model.from_antimony(str(ant_path))
            load_time = time.perf_counter() - t0
            entry["status"] = "ok"
            entry["n_species"] = model.n_species
            entry["load_time"] = load_time
            n_ok += 1
        except Exception as e:
            entry["status"] = "fail"
            entry["error"] = str(e)[:200]
            n_fail += 1

        results.append(entry)

        if (i + 1) % 100 == 0 or i < 3:
            st = entry["status"]
            sp = entry.get("n_species", "?")
            print(f"  [{i + 1}/{len(ant_files)}] {model_id}: {st} ({sp}sp)")

        if (i + 1) % 200 == 0:
            save_checkpoint(results, "bench_biomodels_sbml_phase1")

    print(f"\n  Phase 1: {n_ok} loaded, {n_fail} failed / {len(ant_files)}")
    save_checkpoint(results, "bench_biomodels_sbml_phase1")
    return results


# ── Phase 2: Cross-Validation ────────────────────────────────────────────


def _run_bngsim_ant(ant_path, t_end, n_steps, bngsim_timeout=None):
    import bngsim

    model = bngsim.Model.from_antimony(str(ant_path))
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)
    return list(result.species_names), np.asarray(result.species)


def _run_rr_ant(ant_path, t_end, n_steps):
    import roadrunner

    sbml = _ant_to_sbml(str(ant_path))
    rr = roadrunner.RoadRunner(sbml)
    rr.integrator.absolute_tolerance = 1e-12
    rr.integrator.relative_tolerance = 1e-8
    result = rr.simulate(0, t_end, n_steps + 1)
    data = np.array(result)
    names = []
    for c in result.colnames[1:]:
        names.append(c[1:-1] if c.startswith("[") else c)
    return names, data[:, 1:]


def _xval_by_name(na, ta, nb, tb, rtol=XVAL_RTOL, atol=XVAL_ATOL):
    """Compare trajectories by name with near-zero masking.

    When both engine values are below NEAR_ZERO_THRESHOLD at a time
    point, absolute error is used instead of relative error to avoid
    inflating errors on effectively-zero trajectories.
    """
    ma = {n: i for i, n in enumerate(na)}
    mb = {n: i for i, n in enumerate(nb)}
    common = sorted(set(ma) & set(mb))
    if not common:
        return False, float("inf"), 0, ""
    nt = min(ta.shape[0], tb.shape[0])
    mre = 0.0
    worst = ""
    for name in common:
        ac = ta[:nt, ma[name]]
        bc = tb[:nt, mb[name]]
        abs_diff = np.abs(ac - bc)
        abs_ref = np.abs(bc)
        denom = np.maximum(abs_ref, atol)
        raw_rel = abs_diff / denom
        # Near-zero masking
        both_tiny = (np.abs(ac) < NEAR_ZERO_THRESHOLD) & (abs_ref < NEAR_ZERO_THRESHOLD)
        masked = np.where(both_tiny, abs_diff, raw_rel)
        re = float(np.max(masked))
        if re > mre:
            mre = re
            worst = name
    return mre <= rtol, mre, len(common), worst


def phase2_xval(phase1_results, bngsim_timeout=None, resume_data=None):
    print("\n" + "=" * 70)
    print("  Phase 2: Cross-Validation (BNGsim vs libRoadRunner)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(bngsim_timeout)}")
    print("=" * 70)

    loadable = [r for r in phase1_results if r["status"] == "ok"]
    print(f"  Loadable models: {len(loadable)}")

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r
        print(f"  Resuming: {len(done)} already done")

    results = []
    n_xval = n_bfail = n_rfail = n_xfail = 0

    for i, p1 in enumerate(loadable):
        mid = p1["model"]
        ant_path = p1["ant_path"]

        if mid in done:
            results.append(done[mid])
            if done[mid].get("xval"):
                n_xval += 1
            continue

        entry = {"model": mid, "ant_path": ant_path, "n_species": p1.get("n_species", 0)}
        best_err = float("inf")
        best_h = None
        passed = False

        for t_end in ADAPTIVE_HORIZONS:
            try:
                bn, bt = _run_bngsim_ant(
                    ant_path,
                    t_end,
                    N_STEPS,
                    bngsim_timeout=bngsim_timeout,
                )
            except Exception as e:
                entry.update(status="bngsim_fail", error=str(e)[:200], t_end=t_end)
                break
            if np.any(np.isnan(bt)) or np.any(np.isinf(bt)):
                if t_end == ADAPTIVE_HORIZONS[-1]:
                    entry.update(status="bngsim_nan", t_end=t_end)
                continue

            try:
                rn, rt = _run_rr_ant(ant_path, t_end, N_STEPS)
            except Exception as e:
                entry.update(status="rr_fail", error=str(e)[:200], t_end=t_end)
                break
            if np.any(np.isnan(rt)) or np.any(np.isinf(rt)):
                if t_end == ADAPTIVE_HORIZONS[-1]:
                    entry.update(status="rr_nan", t_end=t_end)
                continue

            ok, merr, nm, worst = _xval_by_name(bn, bt, rn, rt)
            if merr < best_err:
                best_err = merr
                best_h = t_end
            if ok:
                passed = True
                entry.update(
                    status="ok",
                    xval=True,
                    max_rel_err=merr,
                    n_matched=nm,
                    worst_species=worst,
                    t_end=t_end,
                )
                n_xval += 1
                break

        if not passed and "status" not in entry:
            entry.update(
                status="xval_fail",
                xval=False,
                max_rel_err=best_err,
                t_end=best_h or ADAPTIVE_HORIZONS[0],
            )
            n_xfail += 1
        if entry.get("status") == "bngsim_fail":
            n_bfail += 1
        elif entry.get("status") == "rr_fail":
            n_rfail += 1

        results.append(entry)

        if (i + 1) % 100 == 0 or i < 3:
            st = entry.get("status", "?")
            print(f"  [{i + 1}/{len(loadable)}] {mid}: {st} (err={best_err:.2e})")

        if (i + 1) % 200 == 0:
            save_checkpoint(results, "bench_biomodels_sbml_phase2")

    print(
        f"\n  Phase 2: {n_xval} xval, {n_bfail} BNG fail, {n_rfail} RR fail, {n_xfail} xval fail"
    )
    save_checkpoint(results, "bench_biomodels_sbml_phase2")
    return results


# ── Phase 3: Timing ──────────────────────────────────────────────────────


def _time_bngsim_ant(ant_path, t_end, n_steps, bngsim_timeout=None):
    import bngsim

    model = bngsim.Model.from_antimony(str(ant_path))
    sim = bngsim.Simulator(model, method="ode")
    t0 = time.perf_counter()
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)
    return time.perf_counter() - t0


def _time_rr_ant(ant_path, t_end, n_steps):
    import roadrunner

    sbml = _ant_to_sbml(str(ant_path))
    rr = roadrunner.RoadRunner(sbml)
    rr.integrator.absolute_tolerance = 1e-12
    rr.integrator.relative_tolerance = 1e-8
    t0 = time.perf_counter()
    rr.simulate(0, t_end, n_steps + 1)
    return time.perf_counter() - t0


def timed_median(fn, warmup=_WARMUP, runs=_RUNS):
    times = []
    for i in range(warmup + runs):
        try:
            t = fn()
            if t < 0:
                return -1
        except Exception:
            return -1
        if i >= warmup:
            times.append(t)
    return median(times)


def phase3_timing(phase2_results, warmup, runs, bngsim_timeout=None, resume_data=None):
    print("\n" + "=" * 70)
    print(f"  Phase 3: Timing ({warmup}w + {runs}t, median)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(bngsim_timeout)}")
    print("=" * 70)

    xval = [r for r in phase2_results if r.get("xval")]
    print(f"  Cross-validated models: {len(xval)}")

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r
        print(f"  Resuming: {len(done)} already done")

    results = []
    speedups = []

    for i, p2 in enumerate(xval):
        mid = p2["model"]
        ant_path = p2["ant_path"]
        t_end = p2["t_end"]
        nsp = p2.get("n_species", 0)

        if mid in done:
            entry = done[mid]
            results.append(entry)
            bt = entry.get("bngsim_time", -1)
            rt = entry.get("rr_time", -1)
            if bt > 0 and rt > 0:
                speedups.append(rt / bt)
            continue

        entry = {"model": mid, "ant_path": ant_path, "n_species": nsp, "t_end": t_end}

        bt = timed_median(
            lambda ap=ant_path, te=t_end, timeout=bngsim_timeout: _time_bngsim_ant(
                ap,
                te,
                N_STEPS,
                bngsim_timeout=timeout,
            ),
            warmup=warmup,
            runs=runs,
        )
        entry["bngsim_time"] = bt

        rt = timed_median(
            lambda ap=ant_path, te=t_end: _time_rr_ant(ap, te, N_STEPS), warmup=warmup, runs=runs
        )
        entry["rr_time"] = rt

        if bt > 0 and rt > 0:
            ratio = rt / bt
            entry["speedup"] = ratio
            speedups.append(ratio)
            faster = "BNG" if ratio > 1 else "RR"
            factor = ratio if ratio > 1 else 1 / ratio
            print(
                f"  [{i + 1}/{len(xval)}] {mid} ({nsp}sp) "
                f"BNG={bt:.4f}s RR={rt:.4f}s ({faster} {factor:.1f}x)"
            )
        else:
            entry["speedup"] = None
            print(f"  [{i + 1}/{len(xval)}] {mid}: timing fail")

        results.append(entry)

        if (i + 1) % 100 == 0:
            save_checkpoint(results, "bench_biomodels_sbml_phase3")

    if speedups:
        gm = geometric_mean(speedups)
        nbw = sum(1 for s in speedups if s > 1)
        nrw = sum(1 for s in speedups if s <= 1)
        print(f"\n  Phase 3: {len(speedups)} benchmarked")
        print(f"    Geomean speedup (RR/BNG): {gm:.2f}x")
        print(f"    BNGsim faster: {nbw}  RR faster: {nrw}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="BioModels benchmark: BNGsim vs libRoadRunner")
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--warmup", type=int, default=_WARMUP)
    ap.add_argument("--runs", type=int, default=_RUNS)
    ap.add_argument(
        "--ensure-pool",
        action="store_true",
        help="Fetch/convert missing .ant before discovery (opt-in; respects --quick)",
    )
    ap.add_argument(
        "--skip-pool-ensure",
        action="store_true",
        help="Never run pool ensure in the harness (even if BENCH_AUTO_ENSURE_POOL is set)",
    )
    ap.add_argument(
        "--pool-no-fetch",
        action="store_true",
        help="When ensuring: only convert from existing sbml_downloads (no download)",
    )
    add_bngsim_timeout_arg(ap, default=None)
    args = ap.parse_args()

    info = get_machine_info()

    skip_force = args.skip_pool_ensure or _env_truthy("BENCH_SKIP_POOL_ENSURE")
    want_ensure = (args.ensure_pool or _env_truthy("BENCH_AUTO_ENSURE_POOL")) and not skip_force
    allow_partial_manifest = False

    if _biomodels_manifest_path().is_file() and want_ensure:
        bm_root = _biomodels_manifest_path().parent
        if str(bm_root) not in sys.path:
            sys.path.insert(0, str(bm_root))
        from biomodels_ant_pool import (
            ensure_biomodels_ant_pool,  # noqa: PLC0415
            pool_ensure_succeeded,  # noqa: PLC0415
        )
        from biomodels_ant_pool import load_manifest as _load_manifest  # noqa: PLC0415
        from biomodels_ant_pool import required_ids as _required_ids  # noqa: PLC0415

        man = _load_manifest()
        base_ids = _required_ids(man, include_review_extras=True)
        ids_ensure = base_ids[: args.quick] if args.quick > 0 else base_ids
        summary = ensure_biomodels_ant_pool(
            ANT_DIR,
            fetch_network=not args.pool_no_fetch,
            include_review_extras=True,
            id_subset=ids_ensure,
        )
        if not pool_ensure_succeeded(summary):
            present_after = int(
                summary.get("present_after", summary.get("already_present", 0)) or 0
            )
            if present_after > 0:
                allow_partial_manifest = True
                n_fetch_failed = len(summary.get("fetch_failed") or [])
                n_convert_failed = len(summary.get("convert_failed") or [])
                print(
                    "WARNING: pool ensure incomplete "
                    f"(status={summary.get('status')!r}); proceeding with available materialized models "
                    f"({present_after}/{summary.get('required_count')}, "
                    f"fetch_failed={n_fetch_failed}, convert_failed={n_convert_failed}).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"ERROR: pool ensure incomplete (status={summary.get('status')!r}). "
                    "Fix deps/network, or run with a pre-populated BENCH_ANT_DIR:\n"
                    "  python bngsim/benchmarks/convert_sbml_to_ant.py\n"
                    "Or omit --ensure-pool / BENCH_AUTO_ENSURE_POOL if .ant files are already present.",
                    file=sys.stderr,
                )
                sys.exit(1)

    ant_files = discover_ant_files(
        quick_limit=args.quick,
        allow_partial_manifest=allow_partial_manifest,
    )
    if not ant_files:
        print(
            f"ERROR: No Antimony models under {ANT_DIR}.\n"
            "  On a fresh clone: python bngsim/benchmarks/convert_sbml_to_ant.py\n"
            "  Or run this harness with --ensure-pool (or BENCH_AUTO_ENSURE_POOL=1).\n"
            "  Or set BENCH_ANT_DIR to a directory that already contains the .ant files.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\nTotal Antimony files: {len(ant_files)}")

    # Phase 1
    p1r = load_checkpoint("bench_biomodels_sbml_phase1") if args.resume else None
    if args.phase in (0, 1):
        p1 = phase1_load(ant_files, resume_data=p1r)
    else:
        p1 = load_checkpoint("bench_biomodels_sbml_phase1")
        if p1 is None:
            print("ERROR: Phase 1 checkpoint needed")
            sys.exit(1)
    if args.phase == 1:
        return

    # Phase 2
    p2r = load_checkpoint("bench_biomodels_sbml_phase2") if args.resume else None
    if args.phase in (0, 2):
        p2 = phase2_xval(p1, bngsim_timeout=args.bngsim_timeout, resume_data=p2r)
    else:
        p2 = load_checkpoint("bench_biomodels_sbml_phase2")
        if p2 is None:
            print("ERROR: Phase 2 checkpoint needed")
            sys.exit(1)
    if args.phase == 2:
        return

    # Phase 3
    p3r = load_checkpoint("bench_biomodels_sbml_phase3") if args.resume else None
    if args.phase in (0, 3):
        p3 = phase3_timing(
            p2,
            args.warmup,
            args.runs,
            bngsim_timeout=args.bngsim_timeout,
            resume_data=p3r,
        )
    else:
        p3 = load_checkpoint("bench_biomodels_sbml_phase3")
        if p3 is None:
            print("ERROR: Phase 3 checkpoint needed")
            sys.exit(1)

    # Final report
    n_total = len(ant_files)
    n_loaded = sum(1 for r in p1 if r["status"] == "ok")
    n_xval = sum(1 for r in p2 if r.get("xval"))
    speedups = [
        r["rr_time"] / r["bngsim_time"]
        for r in p3
        if r.get("bngsim_time", -1) > 0 and r.get("rr_time", -1) > 0
    ]
    n_timed = len(speedups)

    summary = {
        "n_total": n_total,
        "n_loaded": n_loaded,
        "n_cross_validated": n_xval,
        "n_timed": n_timed,
    }
    if speedups:
        summary["speedup_geometric_mean"] = geometric_mean(speedups)
        summary["speedup_median"] = sorted(speedups)[len(speedups) // 2]
        summary["n_bngsim_faster"] = sum(1 for s in speedups if s > 1)
        summary["n_rr_faster"] = sum(1 for s in speedups if s <= 1)

    sizes = [r.get("n_species", 0) for r in p3 if r.get("bngsim_time", -1) > 0]
    if sizes:
        summary["species_min"] = min(sizes)
        summary["species_max"] = max(sizes)
        summary["species_median"] = sorted(sizes)[len(sizes) // 2]
        summary["species_mean"] = sum(sizes) / len(sizes)

    output = {
        "machine_info": info,
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "xval_rtol": XVAL_RTOL,
            "n_steps": N_STEPS,
            "adaptive_horizons": ADAPTIVE_HORIZONS,
            "bngsim_timeout": args.bngsim_timeout,
        },
        "summary": summary,
        "phase1_load": p1,
        "phase2_xval": p2,
        "phase3_timing": p3,
    }
    save_results(output, "bench_biomodels_sbml")

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Antimony files:        {n_total}")
    print(f"  BNGsim loaded:         {n_loaded} ({100 * n_loaded / n_total:.0f}%)")
    print(f"  Cross-validated:       {n_xval} ({100 * n_xval / n_total:.0f}%)")
    print(f"  Timed (both engines):  {n_timed}")
    if speedups:
        gm = geometric_mean(speedups)
        print(f"  Speedup (geomean):     {gm:.2f}x")
        print(f"  BNGsim faster:         {summary['n_bngsim_faster']}")
        print(f"  RR faster:             {summary['n_rr_faster']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
