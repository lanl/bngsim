#!/usr/bin/env python3
"""~1013-model (manifest union) 4-engine ODE benchmark: BNGsim (ExprTk/Codegen) vs RR vs AMICI.

Three-phase pipeline with checkpointing:
  Phase 1: Load Antimony models (manifest-driven when ``biomodels_ant_pool_manifest.json`` exists).
  **Default scope is the full manifest union** (~1013 IDs from ``all_required_ids``).
  Legacy subset IDs remain in the manifest for compatibility metadata, but this harness
  is named/scoped for the full union. Without a manifest, falls back to top-level ``*.ant``
  under ``BENCH_ANT_DIR``.
  Phase 2: Cross-validate BNGsim vs RR, BNGsim vs AMICI (max_rel_err < 1e-3)
  Phase 3: Fitting-loop timing (load once, warmup once, 5x reset+run, median)
    4 configs: ExprTk vs RR, Codegen vs RR, ExprTk vs AMICI, Codegen vs AMICI

Usage:
    python bench_1013_4engine.py                      # full union (~1013), all phases
    python bench_1013_4engine.py --quick 20           # first 20 models of the active slice
    python bench_1013_4engine.py --ensure-pool        # fetch/convert missing .ant before discovery
    python bench_1013_4engine.py --phase 1            # load gate only

Pool materialization is **opt-in**: pass ``--ensure-pool`` or set ``BENCH_AUTO_ENSURE_POOL=1``.
When enabled, ensure runs **only for the same IDs as this run** (respects ``--quick``).
``--pool-no-fetch`` uses cached SBML only. Override skip with ``--skip-pool-ensure`` /
``BENCH_SKIP_POOL_ENSURE=1``.

For codegen: set BNGSIM_CODEGEN_THRESHOLD=0 to force codegen on all models.

Default ``BENCH_ANT_DIR`` is ``bngsim/benchmarks/biomodels_ant`` (override with an absolute path).

Output:
    results/bench_1013_4engine.json
    results/bench_1013_4engine_phase{1,2,3}.json  (checkpoints)

"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import contextlib

from common import (
    RESULTS_DIR,
    AmiciApiError,
    _ant_to_sbml,
    add_bngsim_timeout_arg,
    amici_prepare_model,
    amici_simulate,
    amici_state_names,
    format_bngsim_timeout,
    get_machine_info,
    save_results,
)

# Stem for checkpoints and JSON outputs (matches script scope: ~1013 manifest union by default).
RUN_STEM = "bench_1013_4engine"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


# ── Paths ─────────────────────────────────────────────────────────────────

_DEFAULT_BIOMODELS_ANT = (
    Path(__file__).resolve().parent.parent.parent / "benchmarks" / "biomodels_ant"
)
ANT_DIR = Path(os.environ.get("BENCH_ANT_DIR", str(_DEFAULT_BIOMODELS_ANT))).expanduser().resolve()
AMICI_CACHE = Path(__file__).resolve().parent.parent / "models" / "amici_cache"

# ── Constants ─────────────────────────────────────────────────────────────

XVAL_RTOL = 1e-3
XVAL_ATOL = 1e-8
ADAPTIVE_HORIZONS = [1.0, 0.1, 10.0]
N_STEPS = 200
N_WARMUP = 1
N_TIMED = 5

SIZE_BINS = [
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101+", 101, 999999),
]

# ── JSON helpers ──────────────────────────────────────────────────────────


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
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(name):
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def geometric_mean(values):
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in values) / len(values)))


# ── Model discovery ───────────────────────────────────────────────────────


def _biomodels_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "benchmarks"
        / "biomodels_ssa"
        / "biomodels_ant_pool_manifest.json"
    )


def _maybe_ensure_ant_pool(
    ant_dir: Path,
    *,
    fetch_network: bool,
    id_subset: list[str],
) -> bool:
    """If repo manifest exists, optionally fetch/convert missing ``.ant`` under ``ant_dir``."""
    bm_root = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "biomodels_ssa"
    manifest = _biomodels_manifest_path()
    if not manifest.is_file():
        return False
    if str(bm_root) not in sys.path:
        sys.path.insert(0, str(bm_root))
    try:
        from biomodels_ant_pool import (  # noqa: PLC0415
            ensure_biomodels_ant_pool,
            pool_ensure_succeeded,
        )
    except ImportError as e:
        print(f"  ERROR: cannot import biomodels_ant_pool: {e}")
        sys.exit(1)

    print("\n  Pool ensure: checking biomodels_ant_pool manifest …")
    summary = ensure_biomodels_ant_pool(
        ant_dir,
        fetch_network=fetch_network,
        include_review_extras=True,
        id_subset=id_subset,
    )
    st = summary.get("status", "?")
    nreq = summary.get("required_count", "?")
    pres = summary.get("present_after", summary.get("already_present", "?"))
    print(f"  Pool ensure: status={st} present={pres}/{nreq}")
    if summary.get("error"):
        print(f"  Pool ensure: {summary['error']}")
    if summary.get("fetch_failed"):
        print(
            f"  WARNING: fetch_failed ({len(summary['fetch_failed'])}): {summary['fetch_failed'][:8]}"
        )
    if summary.get("convert_failed"):
        print(
            f"  WARNING: convert_failed ({len(summary['convert_failed'])}): "
            f"{summary['convert_failed'][:8]}",
        )
    if pool_ensure_succeeded(summary):
        return False
    present_after = int(summary.get("present_after", summary.get("already_present", 0)) or 0)
    if present_after > 0:
        print(
            "  WARNING: pool ensure incomplete, proceeding with available materialized models "
            f"({present_after}/{nreq}).",
        )
        return True
    if st == "incomplete_missing_sbml" and not fetch_network:
        print(
            "  ERROR: missing SBML under biomodels_ssa/data/sbml_downloads; omit --pool-no-fetch"
        )
    elif st == "incomplete_no_antimony":
        print("  ERROR: install antimony (pip install antimony) to convert downloaded SBML.")
    elif st in ("error_manifest", "error_invalid_ant_dir"):
        pass  # message already printed from summary['error']
    else:
        print(
            "  ERROR: pool incomplete after ensure (network, BioModels, or conversion). "
            "Try: python bngsim/benchmarks/convert_sbml_to_ant.py",
        )
    sys.exit(1)


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
    """Resolve ``.ant`` paths (manifest order) or legacy top-level glob."""
    if ANT_DIR.exists() and not ANT_DIR.is_dir():
        print(f"ERROR: BENCH_ANT_DIR is not a directory: {ANT_DIR}")
        sys.exit(1)
    if not ANT_DIR.exists() and not _biomodels_manifest_path().is_file():
        print(f"ERROR: ANT directory not found: {ANT_DIR}")
        print("Set BENCH_ANT_DIR env var or check the path.")
        sys.exit(1)
    manifest_files: list[Path] | None = None
    limit = quick_limit if quick_limit > 0 else None
    if _biomodels_manifest_path().is_file():
        bm_root = _biomodels_manifest_path().parent
        if str(bm_root) not in sys.path:
            sys.path.insert(0, str(bm_root))
        try:
            from biomodels_ant_pool import load_manifest as _load_manifest  # noqa: PLC0415
            from biomodels_ant_pool import required_ids as _required_ids  # noqa: PLC0415
            from biomodels_ant_pool import resolve_manifest_ant_paths  # noqa: PLC0415

            manifest_files = resolve_manifest_ant_paths(
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
            manifest_files = present_files
        except (ValueError, json.JSONDecodeError) as e:
            print(f"ERROR: manifest pool resolution failed: {e}", file=sys.stderr)
            sys.exit(1)
    if manifest_files is not None:
        files = manifest_files
        n_biomd = sum(1 for f in files if f.stem.startswith("BIOMD"))
        n_model = sum(1 for f in files if f.stem.startswith("MODEL"))
        n_other = len(files) - n_biomd - n_model
        scope = "full manifest union (~1013)"
        print(
            f"  Manifest pool ({scope}): {len(files)} .ant files ({n_biomd} BIOMD*, {n_model} MODEL*, "
            f"{n_other} other) from biomodels_ant_pool_manifest.json",
        )
        return files
    top = sorted(ANT_DIR.glob("*.ant"))
    rex = ANT_DIR / "review_extra"
    extra = sorted(rex.glob("*.ant")) if rex.is_dir() else []
    files = sorted({*top, *extra}, key=lambda p: p.name)
    n_biomd = sum(1 for f in files if f.stem.startswith("BIOMD"))
    n_model = sum(1 for f in files if f.stem.startswith("MODEL"))
    n_other = len(files) - n_biomd - n_model
    print(f"  Found {len(files)} .ant files: {n_biomd} BIOMD*, {n_model} MODEL*, {n_other} other")
    if limit is not None:
        files = files[:limit]
    return files


# ── AMICI helpers ─────────────────────────────────────────────────────────


def compile_amici_model(sbml_str, model_name):
    """Compile SBML to AMICI model module. Returns module or None."""
    import amici

    h = hashlib.sha256(sbml_str.encode()).hexdigest()[:12]
    safe_name = f"amici_{model_name}_{h}"
    model_dir = AMICI_CACHE / safe_name
    if model_dir.exists():
        try:
            return amici.import_model_module(safe_name, str(model_dir))
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
    AMICI_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        imp = amici.SbmlImporter(sbml_str, from_file=False)
        imp.sbml2amici(safe_name, str(model_dir), verbose=False)
        return amici.import_model_module(safe_name, str(model_dir))
    except Exception:
        shutil.rmtree(model_dir, ignore_errors=True)
        return None


# ── Phase 1: Load Gate ───────────────────────────────────────────────────


def phase1_load(ant_files, resume_data=None):
    """Load all .ant files in BNGsim, RR, and AMICI. Record success/fail."""
    import bngsim

    print("\n" + "=" * 70)
    print("  Phase 1: Load Gate (BNGsim + RR + AMICI)")
    print(f"  Models: {len(ant_files)}")
    print("=" * 70)

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r
        print(f"  Resuming: {len(done)} already done")

    # Check which engines are available
    have_rr = False
    try:
        import roadrunner  # noqa: F401

        have_rr = True
    except ImportError:
        print("  WARNING: roadrunner not installed, skipping RR load")

    have_amici = False
    try:
        import amici  # noqa: F401

        have_amici = True
    except ImportError:
        print("  WARNING: amici not installed, skipping AMICI load")

    results = []
    n_bng_ok = n_rr_ok = n_amici_ok = 0
    n_bng_fail = n_rr_fail = n_amici_fail = n_amici_compile_fail = 0

    for i, ant_path in enumerate(ant_files):
        model_id = ant_path.stem

        if model_id in done:
            results.append(done[model_id])
            if done[model_id].get("bngsim_status") == "ok":
                n_bng_ok += 1
            if done[model_id].get("rr_status") == "ok":
                n_rr_ok += 1
            if done[model_id].get("amici_status") == "ok":
                n_amici_ok += 1
            continue

        entry = {"model": model_id, "ant_path": str(ant_path)}

        # ── BNGsim load ──
        try:
            t0 = time.perf_counter()
            model = bngsim.Model.from_antimony(str(ant_path))
            entry["bngsim_status"] = "ok"
            entry["n_species"] = model.n_species
            entry["bngsim_load_time"] = time.perf_counter() - t0
            n_bng_ok += 1
        except Exception as e:
            entry["bngsim_status"] = "fail"
            entry["bngsim_error"] = str(e)[:200]
            n_bng_fail += 1

        # ── Antimony → SBML (shared by RR and AMICI) ──
        sbml_str = None
        try:
            sbml_str = _ant_to_sbml(str(ant_path))
            entry["sbml_ok"] = True
        except Exception as e:
            entry["sbml_ok"] = False
            entry["sbml_error"] = str(e)[:200]

        # ── RR load ──
        if have_rr and sbml_str:
            try:
                import roadrunner

                t0 = time.perf_counter()
                roadrunner.RoadRunner(sbml_str)
                entry["rr_status"] = "ok"
                entry["rr_load_time"] = time.perf_counter() - t0
                n_rr_ok += 1
            except Exception as e:
                entry["rr_status"] = "fail"
                entry["rr_error"] = str(e)[:200]
                n_rr_fail += 1
        elif not have_rr:
            entry["rr_status"] = "skip"
        else:
            entry["rr_status"] = "sbml_fail"
            n_rr_fail += 1

        # ── AMICI compile ──
        if have_amici and sbml_str:
            try:
                amici_mod = compile_amici_model(sbml_str, model_id)
                if amici_mod is not None:
                    entry["amici_status"] = "ok"
                    n_amici_ok += 1
                else:
                    entry["amici_status"] = "compile_fail"
                    n_amici_compile_fail += 1
            except Exception as e:
                entry["amici_status"] = "compile_fail"
                entry["amici_error"] = str(e)[:200]
                n_amici_compile_fail += 1
        elif not have_amici:
            entry["amici_status"] = "skip"
        else:
            entry["amici_status"] = "sbml_fail"
            n_amici_fail += 1

        results.append(entry)

        if (i + 1) % 100 == 0 or i < 3:
            bs = entry.get("bngsim_status", "?")
            sp = entry.get("n_species", "?")
            rs = entry.get("rr_status", "?")
            ams = entry.get("amici_status", "?")
            print(f"  [{i + 1}/{len(ant_files)}] {model_id}: BNG={bs}({sp}sp) RR={rs} AMICI={ams}")

        if (i + 1) % 200 == 0:
            save_checkpoint(results, f"{RUN_STEM}_phase1")

    print("\n  Phase 1 results:")
    print(f"    BNGsim:  {n_bng_ok} ok, {n_bng_fail} fail")
    print(f"    RR:      {n_rr_ok} ok, {n_rr_fail} fail")
    print(
        f"    AMICI:   {n_amici_ok} ok, {n_amici_compile_fail} compile_fail, {n_amici_fail} fail"
    )
    save_checkpoint(results, f"{RUN_STEM}_phase1")
    return results


# ── Phase 2: Cross-Validation ────────────────────────────────────────────


def _run_bngsim_ant(ant_path, t_end, n_steps, bngsim_timeout=None):
    """Run BNGsim ODE, return (species_names, species_array)."""
    import bngsim

    model = bngsim.Model.from_antimony(str(ant_path))
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)
    return list(result.species_names), np.asarray(result.species)


def _run_rr_sbml(sbml_str, t_end, n_steps):
    """Run RR ODE from SBML string, return (species_names, species_array)."""
    import roadrunner

    rr = roadrunner.RoadRunner(sbml_str)
    rr.integrator.absolute_tolerance = 1e-12
    rr.integrator.relative_tolerance = 1e-8
    result = rr.simulate(0, t_end, n_steps + 1)
    data = np.array(result)
    names = []
    for c in result.colnames[1:]:
        names.append(c[1:-1] if c.startswith("[") else c)
    return names, data[:, 1:]


def _run_amici_sbml(sbml_str, model_name, t_end, n_steps):
    """Run AMICI ODE (AMICI >= 1.0 snake_case API); return (names, states) or raise.

    Compile failure / non-success CVODES status raise RuntimeError; an AMICI API
    drift raises AmiciApiError, which propagates past the callers' error-row
    handlers to crash loudly (GH #227).
    """
    amici_mod = compile_amici_model(sbml_str, model_name)
    if amici_mod is None:
        raise RuntimeError("AMICI compile failed")
    model, solver = amici_prepare_model(amici_mod)
    tspan = np.linspace(0, t_end, n_steps + 1)
    rdata, _ = amici_simulate(model, solver, tspan)
    return amici_state_names(model), np.array(rdata.x)


def _xval_by_name(na, ta, nb, tb, rtol=XVAL_RTOL, atol=XVAL_ATOL):
    """Cross-validate two trajectories by matching species names."""
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
        denom = np.maximum(np.abs(bc), atol)
        re = float(np.max(np.abs(ac - bc) / denom))
        if re > mre:
            mre = re
            worst = name
    return mre <= rtol, mre, len(common), worst


def phase2_xval(phase1_results, bngsim_timeout=None, resume_data=None):
    """Cross-validate BNGsim vs RR and BNGsim vs AMICI."""
    print("\n" + "=" * 70)
    print("  Phase 2: Cross-Validation")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(bngsim_timeout)}")
    print("=" * 70)

    # Only consider models that loaded in BNGsim
    loadable = [r for r in phase1_results if r.get("bngsim_status") == "ok"]
    print(f"  BNGsim-loadable models: {len(loadable)}")

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r
        print(f"  Resuming: {len(done)} already done")

    results = []
    n_rr_xval = n_rr_fail = 0
    n_amici_xval = n_amici_fail = 0

    for i, p1 in enumerate(loadable):
        mid = p1["model"]
        ant_path = p1["ant_path"]

        if mid in done:
            results.append(done[mid])
            if done[mid].get("rr_xval"):
                n_rr_xval += 1
            if done[mid].get("amici_xval"):
                n_amici_xval += 1
            continue

        entry = {
            "model": mid,
            "ant_path": ant_path,
            "n_species": p1.get("n_species", 0),
        }

        # Get SBML string for RR and AMICI
        sbml_str = None
        with contextlib.suppress(Exception):
            sbml_str = _ant_to_sbml(str(ant_path))

        # Try adaptive time horizons for BNGsim
        bng_names = bng_traj = None
        best_t_end = None
        for t_end in ADAPTIVE_HORIZONS:
            try:
                bn, bt = _run_bngsim_ant(
                    ant_path,
                    t_end,
                    N_STEPS,
                    bngsim_timeout=bngsim_timeout,
                )
            except Exception:
                continue
            if np.any(np.isnan(bt)) or np.any(np.isinf(bt)):
                continue
            bng_names, bng_traj = bn, bt
            best_t_end = t_end
            break

        if bng_traj is None:
            entry["status"] = "bngsim_sim_fail"
            results.append(entry)
            continue

        entry["t_end"] = best_t_end

        # ── Cross-validate vs RR ──
        rr_ok = p1.get("rr_status") == "ok"
        if rr_ok and sbml_str:
            try:
                rn, rt = _run_rr_sbml(sbml_str, best_t_end, N_STEPS)
                if np.any(np.isnan(rt)) or np.any(np.isinf(rt)):
                    entry["rr_xval"] = False
                    entry["rr_xval_err"] = "rr_nan"
                    n_rr_fail += 1
                else:
                    ok, merr, nm, worst = _xval_by_name(bng_names, bng_traj, rn, rt)
                    entry["rr_xval"] = ok
                    entry["rr_max_rel_err"] = merr
                    entry["rr_n_matched"] = nm
                    if ok:
                        n_rr_xval += 1
                    else:
                        n_rr_fail += 1
            except Exception as e:
                entry["rr_xval"] = False
                entry["rr_xval_err"] = str(e)[:200]
                n_rr_fail += 1
        else:
            entry["rr_xval"] = False
            entry["rr_xval_err"] = "not_loaded"

        # ── Cross-validate vs AMICI ──
        amici_ok = p1.get("amici_status") == "ok"
        if amici_ok and sbml_str:
            try:
                an, at = _run_amici_sbml(sbml_str, mid, best_t_end, N_STEPS)
                if np.any(np.isnan(at)) or np.any(np.isinf(at)):
                    entry["amici_xval"] = False
                    entry["amici_xval_err"] = "amici_nan"
                    n_amici_fail += 1
                else:
                    ok, merr, nm, worst = _xval_by_name(bng_names, bng_traj, an, at)
                    entry["amici_xval"] = ok
                    entry["amici_max_rel_err"] = merr
                    entry["amici_n_matched"] = nm
                    if ok:
                        n_amici_xval += 1
                    else:
                        n_amici_fail += 1
            except AmiciApiError:
                raise  # API drift must crash loudly (GH #227)
            except Exception as e:
                entry["amici_xval"] = False
                entry["amici_xval_err"] = str(e)[:200]
                n_amici_fail += 1
        else:
            entry["amici_xval"] = False
            entry["amici_xval_err"] = "not_loaded"

        results.append(entry)

        if (i + 1) % 100 == 0 or i < 3:
            rv = "Y" if entry.get("rr_xval") else "N"
            av = "Y" if entry.get("amici_xval") else "N"
            print(f"  [{i + 1}/{len(loadable)}] {mid}: RR_xval={rv} AMICI_xval={av}")

        if (i + 1) % 200 == 0:
            save_checkpoint(results, f"{RUN_STEM}_phase2")

    print("\n  Phase 2 results:")
    print(f"    RR xval:    {n_rr_xval} pass, {n_rr_fail} fail")
    print(f"    AMICI xval: {n_amici_xval} pass, {n_amici_fail} fail")
    save_checkpoint(results, f"{RUN_STEM}_phase2")
    return results


# ── Phase 3: Fitting-Loop Timing ─────────────────────────────────────────
#
# Protocol per model per engine:
#   1. Load model ONCE (outside timing)
#   2. Create Simulator / RR / AMICI solver ONCE (outside timing)
#   3. N_WARMUP warmup: reset + run (outside timing)
#   4. N_TIMED timed:   reset + run → report median
#
# 4 timing configs:
#   - BNGsim ExprTk vs RR
#   - BNGsim Codegen vs RR
#   - BNGsim ExprTk vs AMICI
#   - BNGsim Codegen vs AMICI


def _load_bngsim_exprTk(ant_path):
    """Load BNGsim model with ExprTk (codegen disabled)."""
    old = os.environ.get("BNGSIM_NO_CODEGEN")
    os.environ["BNGSIM_NO_CODEGEN"] = "1"
    try:
        import bngsim

        return bngsim.Model.from_antimony(str(ant_path))
    finally:
        if old is None:
            os.environ.pop("BNGSIM_NO_CODEGEN", None)
        else:
            os.environ["BNGSIM_NO_CODEGEN"] = old


def _load_bngsim_codegen(ant_path):
    """Load BNGsim model with codegen (forced via threshold=0)."""
    old_no = os.environ.get("BNGSIM_NO_CODEGEN")
    old_th = os.environ.get("BNGSIM_CODEGEN_THRESHOLD")
    os.environ.pop("BNGSIM_NO_CODEGEN", None)
    os.environ["BNGSIM_CODEGEN_THRESHOLD"] = "0"
    try:
        import bngsim

        return bngsim.Model.from_antimony(str(ant_path))
    finally:
        if old_no is None:
            os.environ.pop("BNGSIM_NO_CODEGEN", None)
        else:
            os.environ["BNGSIM_NO_CODEGEN"] = old_no
        if old_th is None:
            os.environ.pop("BNGSIM_CODEGEN_THRESHOLD", None)
        else:
            os.environ["BNGSIM_CODEGEN_THRESHOLD"] = old_th


def _time_bngsim_run(model, t_end, bngsim_timeout=None):
    """Time a single BNGsim reset+run on a pre-loaded model."""
    import bngsim

    m = model.clone()
    m.reset()
    cg = getattr(model, "_codegen_so_path", "")
    if cg:
        m._codegen_so_path = cg
    sim = bngsim.Simulator(m, method="ode")
    t0 = time.perf_counter()
    sim.run(t_span=(0, t_end), n_points=N_STEPS + 1, timeout=bngsim_timeout)
    return time.perf_counter() - t0


def _load_rr(ant_path):
    """Load RR model from Antimony file."""
    import roadrunner

    sbml_str = _ant_to_sbml(str(ant_path))
    rr = roadrunner.RoadRunner(sbml_str)
    rr.integrator.absolute_tolerance = 1e-12
    rr.integrator.relative_tolerance = 1e-8
    return rr


def _time_rr_run(rr_obj, t_end):
    """Time a single RR reset+simulate on a pre-loaded model."""
    rr_obj.reset()
    t0 = time.perf_counter()
    rr_obj.simulate(0, t_end, N_STEPS + 1)
    return time.perf_counter() - t0


def _load_amici(ant_path, model_name):
    """Load AMICI model+solver from Antimony file (AMICI >= 1.0 snake_case API)."""
    sbml_str = _ant_to_sbml(str(ant_path))
    amici_mod = compile_amici_model(sbml_str, model_name)
    if amici_mod is None:
        raise RuntimeError("AMICI compile failed")
    return amici_prepare_model(amici_mod)


def _time_amici_run(amici_model, amici_solver, t_end):
    """Time a single AMICI simulation (AMICI >= 1.0 snake_case API).

    A non-success CVODES status raises RuntimeError (caught by _timed_median →
    -1, matching the old contract); an AMICI API drift raises AmiciApiError,
    which _timed_median re-raises so it crashes loudly (GH #227).
    """
    tspan = np.linspace(0, t_end, N_STEPS + 1)
    _, elapsed = amici_simulate(amici_model, amici_solver, tspan)
    return elapsed


def _timed_median(fn):
    """Run fn N_WARMUP+N_TIMED times, return median of timed runs."""
    times = []
    for i in range(N_WARMUP + N_TIMED):
        try:
            t = fn()
            if t < 0:
                return -1
        except AmiciApiError:
            raise  # API drift must crash loudly, never degrade to -1 (GH #227)
        except Exception:
            return -1
        if i >= N_WARMUP:
            times.append(t)
    return median(times)


def phase3_timing(phase2_results, bngsim_timeout=None, resume_data=None):
    """Fitting-loop timing: load once, warmup, 5x reset+run, median."""
    print("\n" + "=" * 70)
    print("  Phase 3: Fitting-Loop Timing")
    print(f"  Protocol: load once, {N_WARMUP} warmup, {N_TIMED} timed (median)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(bngsim_timeout)}")
    print("=" * 70)

    # Build sets of models validated per engine pair
    rr_xval = {r["model"] for r in phase2_results if r.get("rr_xval")}
    am_xval = {r["model"] for r in phase2_results if r.get("amici_xval")}
    # Models to time: union of both xval sets
    to_time = [r for r in phase2_results if r.get("rr_xval") or r.get("amici_xval")]
    print(
        f"  RR-validated: {len(rr_xval)}, AMICI-validated: {len(am_xval)}, Union: {len(to_time)}"
    )

    done = {}
    if resume_data:
        for r in resume_data:
            done[r["model"]] = r
        print(f"  Resuming: {len(done)} already done")

    results = []
    et_rr_speedups = []
    cg_rr_speedups = []
    et_am_speedups = []
    cg_am_speedups = []

    for i, p2 in enumerate(to_time):
        mid = p2["model"]
        ant_path = p2["ant_path"]
        t_end = p2.get("t_end", 1.0)
        nsp = p2.get("n_species", 0)

        if mid in done:
            entry = done[mid]
            results.append(entry)
            # Collect speedups from resumed data
            et = entry.get("exprTk_time", -1)
            cg = entry.get("codegen_time", -1)
            rt = entry.get("rr_time", -1)
            at = entry.get("amici_time", -1)
            if et > 0 and rt > 0 and mid in rr_xval:
                et_rr_speedups.append(rt / et)
            if cg > 0 and rt > 0 and mid in rr_xval:
                cg_rr_speedups.append(rt / cg)
            if et > 0 and at > 0 and mid in am_xval:
                et_am_speedups.append(at / et)
            if cg > 0 and at > 0 and mid in am_xval:
                cg_am_speedups.append(at / cg)
            continue

        entry = {
            "model": mid,
            "ant_path": ant_path,
            "n_species": nsp,
            "t_end": t_end,
            "rr_validated": mid in rr_xval,
            "amici_validated": mid in am_xval,
        }

        # ── BNGsim ExprTk ──
        try:
            mdl_et = _load_bngsim_exprTk(ant_path)
            et = _timed_median(
                lambda m=mdl_et, te=t_end, timeout=bngsim_timeout: _time_bngsim_run(
                    m,
                    te,
                    bngsim_timeout=timeout,
                )
            )
            entry["exprTk_time"] = et
        except Exception:
            entry["exprTk_time"] = -1

        # ── BNGsim Codegen ──
        try:
            mdl_cg = _load_bngsim_codegen(ant_path)
            cg = _timed_median(
                lambda m=mdl_cg, te=t_end, timeout=bngsim_timeout: _time_bngsim_run(
                    m,
                    te,
                    bngsim_timeout=timeout,
                )
            )
            entry["codegen_time"] = cg
        except Exception:
            entry["codegen_time"] = -1

        # ── libRoadRunner ──
        if mid in rr_xval:
            try:
                rr_obj = _load_rr(ant_path)
                rt = _timed_median(lambda r=rr_obj, te=t_end: _time_rr_run(r, te))
                entry["rr_time"] = rt
            except Exception:
                entry["rr_time"] = -1
        else:
            entry["rr_time"] = -1

        # ── AMICI ──
        if mid in am_xval:
            try:
                am_model, am_solver = _load_amici(ant_path, mid)
                at = _timed_median(
                    lambda m=am_model, s=am_solver, te=t_end: _time_amici_run(m, s, te)
                )
                entry["amici_time"] = at
            except AmiciApiError:
                raise  # API drift must crash loudly (GH #227)
            except Exception:
                entry["amici_time"] = -1
        else:
            entry["amici_time"] = -1

        results.append(entry)

        # Collect speedups
        et = entry.get("exprTk_time", -1)
        cg = entry.get("codegen_time", -1)
        rt = entry.get("rr_time", -1)
        at = entry.get("amici_time", -1)
        if et > 0 and rt > 0 and mid in rr_xval:
            et_rr_speedups.append(rt / et)
        if cg > 0 and rt > 0 and mid in rr_xval:
            cg_rr_speedups.append(rt / cg)
        if et > 0 and at > 0 and mid in am_xval:
            et_am_speedups.append(at / et)
        if cg > 0 and at > 0 and mid in am_xval:
            cg_am_speedups.append(at / cg)

        # Progress
        parts = []
        if et > 0:
            parts.append(f"ET={et * 1e3:.2f}ms")
        if cg > 0:
            parts.append(f"CG={cg * 1e3:.2f}ms")
        if rt > 0:
            parts.append(f"RR={rt * 1e3:.2f}ms")
        if at > 0:
            parts.append(f"AM={at * 1e3:.2f}ms")
        print(f"  [{i + 1}/{len(to_time)}] {mid} ({nsp}sp) {' '.join(parts)}")

        if (i + 1) % 100 == 0:
            save_checkpoint(results, f"{RUN_STEM}_phase3")

    # Print summary
    print("\n  Phase 3 timing summary:")
    for label, arr in [
        ("ExprTk vs RR", et_rr_speedups),
        ("Codegen vs RR", cg_rr_speedups),
        ("ExprTk vs AMICI", et_am_speedups),
        ("Codegen vs AMICI", cg_am_speedups),
    ]:
        if arr:
            gm = geometric_mean(arr)
            nw = sum(1 for s in arr if s > 1)
            print(f"    {label}: geomean {gm:.2f}x, BNG wins {nw}/{len(arr)}")

    save_checkpoint(results, f"{RUN_STEM}_phase3")
    return results


# ── Summary + Main ────────────────────────────────────────────────────────


def print_summary(p1, p2, p3, n_total):
    """Print final summary table."""
    n_loaded = sum(1 for r in p1 if r.get("bngsim_status") == "ok")
    n_rr_xval = sum(1 for r in p2 if r.get("rr_xval"))
    n_am_xval = sum(1 for r in p2 if r.get("amici_xval"))

    # Compute speedups from Phase 3
    rr_xval_set = {r["model"] for r in p2 if r.get("rr_xval")}
    am_xval_set = {r["model"] for r in p2 if r.get("amici_xval")}

    et_rr = [
        r["rr_time"] / r["exprTk_time"]
        for r in p3
        if r.get("exprTk_time", -1) > 0 and r.get("rr_time", -1) > 0 and r["model"] in rr_xval_set
    ]
    cg_rr = [
        r["rr_time"] / r["codegen_time"]
        for r in p3
        if r.get("codegen_time", -1) > 0 and r.get("rr_time", -1) > 0 and r["model"] in rr_xval_set
    ]
    et_am = [
        r["amici_time"] / r["exprTk_time"]
        for r in p3
        if r.get("exprTk_time", -1) > 0
        and r.get("amici_time", -1) > 0
        and r["model"] in am_xval_set
    ]
    cg_am = [
        r["amici_time"] / r["codegen_time"]
        for r in p3
        if r.get("codegen_time", -1) > 0
        and r.get("amici_time", -1) > 0
        and r["model"] in am_xval_set
    ]

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total .ant files:       {n_total}")
    print(f"  BNGsim loaded:          {n_loaded}")
    print(f"  RR cross-validated:     {n_rr_xval}")
    print(f"  AMICI cross-validated:  {n_am_xval}")
    print()
    for label, arr in [
        ("ExprTk vs RR", et_rr),
        ("Codegen vs RR", cg_rr),
        ("ExprTk vs AMICI", et_am),
        ("Codegen vs AMICI", cg_am),
    ]:
        if arr:
            gm = geometric_mean(arr)
            nw = sum(1 for s in arr if s > 1)
            print(f"  {label:20s}: geomean {gm:.2f}x, BNG wins {nw}/{len(arr)}")
        else:
            print(f"  {label:20s}: no data")

    # Bin analysis for Codegen vs RR
    if cg_rr:
        print("\n  Codegen vs RR by species bin:")
        print(f"  {'Bin':>8s} {'N':>5s} {'Geomean':>8s} {'BNG wins':>10s}")
        print("  " + "-" * 40)
        for label, lo, hi in SIZE_BINS:
            bm = [
                r
                for r in p3
                if lo <= r.get("n_species", 0) <= hi
                and r.get("codegen_time", -1) > 0
                and r.get("rr_time", -1) > 0
                and r["model"] in rr_xval_set
            ]
            if not bm:
                continue
            ratios = [r["rr_time"] / r["codegen_time"] for r in bm]
            gm = geometric_mean(ratios)
            nw = sum(1 for r in ratios if r > 1)
            print(f"  {label:>8s} {len(bm):>5d} {gm:>8.2f}x {nw:>5d}/{len(bm)}")

    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="~1013-model manifest-union 4-engine ODE benchmark")
    ap.add_argument("--quick", type=int, default=0, help="Limit to first N models")
    ap.add_argument("--phase", type=int, default=0, help="Run single phase (1/2/3), 0=all")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    ap.add_argument(
        "--ensure-pool",
        action="store_true",
        help="Fetch/convert missing .ant before discovery (opt-in; respects --quick and slice)",
    )
    ap.add_argument(
        "--skip-pool-ensure",
        action="store_true",
        help="Never fetch/convert via harness (even if BENCH_AUTO_ENSURE_POOL is set)",
    )
    ap.add_argument(
        "--pool-no-fetch",
        action="store_true",
        help="When ensuring: only convert from existing sbml_downloads (no BioModels download)",
    )
    add_bngsim_timeout_arg(ap)
    args = ap.parse_args()

    info = get_machine_info()

    skip_force = args.skip_pool_ensure or _env_truthy("BENCH_SKIP_POOL_ENSURE")
    want_ensure = (args.ensure_pool or _env_truthy("BENCH_AUTO_ENSURE_POOL")) and not skip_force

    allow_partial_manifest = False
    if _biomodels_manifest_path().is_file() and want_ensure:
        bm_root = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "biomodels_ssa"
        if str(bm_root) not in sys.path:
            sys.path.insert(0, str(bm_root))
        from biomodels_ant_pool import load_manifest as _load_manifest  # noqa: PLC0415
        from biomodels_ant_pool import required_ids as _required_ids  # noqa: PLC0415

        man = _load_manifest()
        base_ids = _required_ids(man, include_review_extras=True)
        ids_for_ensure = base_ids[: args.quick] if args.quick > 0 else base_ids
        allow_partial_manifest = _maybe_ensure_ant_pool(
            ANT_DIR,
            fetch_network=not args.pool_no_fetch,
            id_subset=ids_for_ensure,
        )

    ant_files = discover_ant_files(
        quick_limit=args.quick,
        allow_partial_manifest=allow_partial_manifest,
    )
    n_total = len(ant_files)
    print(f"\nTotal Antimony files: {n_total}")

    # ── Phase 1 ──
    p1r = load_checkpoint(f"{RUN_STEM}_phase1") if args.resume else None
    if args.phase in (0, 1):
        p1 = phase1_load(ant_files, resume_data=p1r)
    else:
        p1 = load_checkpoint(f"{RUN_STEM}_phase1")
        if p1 is None:
            print("ERROR: Phase 1 checkpoint needed")
            sys.exit(1)
    if args.phase == 1:
        return

    # ── Phase 2 ──
    p2r = load_checkpoint(f"{RUN_STEM}_phase2") if args.resume else None
    if args.phase in (0, 2):
        p2 = phase2_xval(p1, bngsim_timeout=args.bngsim_timeout, resume_data=p2r)
    else:
        p2 = load_checkpoint(f"{RUN_STEM}_phase2")
        if p2 is None:
            print("ERROR: Phase 2 checkpoint needed")
            sys.exit(1)
    if args.phase == 2:
        return

    # ── Phase 3 ──
    p3r = load_checkpoint(f"{RUN_STEM}_phase3") if args.resume else None
    if args.phase in (0, 3):
        p3 = phase3_timing(
            p2,
            bngsim_timeout=args.bngsim_timeout,
            resume_data=p3r,
        )
    else:
        p3 = load_checkpoint(f"{RUN_STEM}_phase3")
        if p3 is None:
            print("ERROR: Phase 3 checkpoint needed")
            sys.exit(1)

    # ── Final output ──
    print_summary(p1, p2, p3, n_total)

    output = {
        "machine_info": info,
        "protocol": {
            "n_warmup": N_WARMUP,
            "n_timed": N_TIMED,
            "n_steps": N_STEPS,
            "xval_rtol": XVAL_RTOL,
            "adaptive_horizons": ADAPTIVE_HORIZONS,
            "bngsim_timeout": args.bngsim_timeout,
        },
        "phase1_load": p1,
        "phase2_xval": p2,
        "phase3_timing": p3,
    }
    save_results(output, RUN_STEM)
    print("\nDone!")


if __name__ == "__main__":
    main()
