#!/usr/bin/env python3
"""Run the copasi_parity sweep (bngsim vs COPASI, SBML ODE) and emit a ``_core`` report.

The COPASI sibling of ``rr_parity/rr_run.py`` and ``amici_parity/amici_run.py``. It
reuses rr_parity's bngsim adapter (``bn_ode``), the shared ``_core.differ`` oracle,
the process scheduler, and the SBML corpus + job manifest under ``rr_parity/`` —
only the *reference* engine changes to COPASI (``_copasi_common.co_ode``). Each job
runs both engines in one disposable subprocess and compares them directly.

ODE-only by design. Verdict is derived from per-engine status, exactly as in
rr_parity / amici_parity:

    both ran, within tolerance ................. PASS
    both ran, metric over tolerance / non-finite DIFF
    both ran, species/time grids disjoint ...... DIFF (loud, value=inf)
    bngsim raised, COPASI ran .................. EXCEPTION (actionable bngsim bug)
    bngsim ran, COPASI raised ................. REFERENCE_FAILED (no oracle; non-scoring)
    both raised ................................ BAD_TEST (no signal; non-scoring)
    wall-clock cap exceeded .................... TIMEOUT
    child segfaulted without a result .......... EXCEPTION (unattributable)

COPASI has no per-model codegen/compile step, so a sweep is as cheap as rr_parity
(no AMICI-style cold-build tier).

OVERRIDES — as in amici_parity, only the engine-agnostic ``tol`` overrides
(ill-conditioned IVPs, applied identically to both engines) are honored. rr_parity's
``known_artifact`` / ``invalid_reference`` / ``no_oracle_adjudicated`` overrides are
calibrated against *RoadRunner* and are intentionally NOT applied, so COPASI
adjudicates every model independently.

Usage:
    cd bngsim && .venv/bin/python parity_checks/copasi_parity/copasi_run.py \\
        --limit 20 --workers 4
    .venv/bin/python parity_checks/copasi_parity/copasi_run.py \\
        --models BIOMD0000000012,BIOMD0000000010

Output: runs/report_ode.json (a _core report; runs/ is gitignored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RR_PARITY = HERE.parent / "rr_parity"  # corpus (models/) + default manifest live here
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` and `from rr_parity` resolve
sys.path.insert(0, str(HERE))  # so the spawned child can import _copasi_common / copasi_run

import _copasi_common as cc  # noqa: E402
from _core import (  # noqa: E402
    FAILING,
    JobResult,
    Outcome,
    differ,
    read_manifest,
    tally,
    versions,
    write_report,
)
from _core.versions import git_rev  # noqa: E402

RR_ODE_JOBS = RR_PARITY / "ode_jobs.json"
DEFAULT_ODE_TIMEOUT = 120.0  # COPASI has no compile phase, so rr_parity's 120s cap fits.


def _job_tol(job) -> dict | None:
    """Read the engine-agnostic ``tol`` override (ill-conditioned IVP) from
    ``Job.overrides``, or None. RR-calibrated overrides are ignored here (see the
    module docstring) — only ``tol`` is genuinely engine-independent."""
    for o in job.overrides:
        if o.field == "tol":
            return {"rtol": float(o.value["rtol"]), "atol": float(o.value["atol"])}
    return None


DEFAULT_RTOL = cc.DEFAULT_RTOL
DEFAULT_ATOL = cc.DEFAULT_ATOL


# --------------------------------------------------------------------------- #
# Comparison (the oracle, applied in the worker) — identical to rr_parity /
# amici_parity's, the shared _core.differ protocol over the common species.
# --------------------------------------------------------------------------- #
def _compare_ode(bn, co) -> tuple[str, float, str, str, float]:
    """(status, value, comment, metric, tol) for one deterministic job."""
    bn_t, bn_v, bn_n = bn
    co_t, co_v, co_n = co
    metric, tol = "max_rel_err", differ.REL_TOL
    if bn_t.shape != co_t.shape or not np.allclose(bn_t, co_t, rtol=0, atol=1e-9):
        return (
            "diff",
            float("inf"),
            f"time grid mismatch (bn n={bn_t.shape}, copasi n={co_t.shape})",
            metric,
            tol,
        )
    align = cc.align_common(bn_n, co_n)
    if align is None:
        return (
            "diff",
            float("inf"),
            f"disjoint species sets: bn={bn_n[:4]} copasi={co_n[:4]}",
            metric,
            tol,
        )
    bn_idx, co_idx, common = align
    v = differ.deterministic_verdict(bn_v[:, bn_idx], co_v[:, co_idx])
    status = "pass" if v["passed"] else "diff"
    comment = (
        f"{len(common)} sp; fail {v['n_fail']}/{v['n_cells']} "
        f"(hard {v['n_hard_fail']}, soft {v['n_soft_fail']}, forgiven {v['budget_forgiven']})"
    )
    return status, v["max_rel"], comment, metric, tol


# --------------------------------------------------------------------------- #
# Worker (module-level so it is picklable under the 'spawn' start method)
# --------------------------------------------------------------------------- #
SETTLED_REFUSALS = frozenset({"adjudicated_confirm", "adjudicated_uncoverable"})


def _classify_failure(bn_exc: str, co_exc: str) -> tuple[str, str]:
    """Map per-engine raise status to ``(status, exception)``. COPASI is the
    existence proof, so the engine that failed determines the bucket."""
    if bn_exc and co_exc:  # neither ran -> the test/model is the problem
        return "bad_test", f"{co_exc} || {bn_exc}"
    if co_exc:  # bngsim ran but the reference couldn't -> nothing to compare against
        return "reference_failed", co_exc
    return "exception", bn_exc  # the reference ran but bngsim raised -> bngsim bug


def _worker(spec: dict, q) -> None:
    """Run BOTH engines for one job and put a result dict on ``q``."""
    warmup = cc.measure_warmup()
    cc.set_copasi_quiet()

    # Apply the selected combo's bngsim override env vars inside the worker, before
    # bngsim loads the model (it reads them lazily at load/codegen/run time).
    for _k, _v in spec.get("config_env", {}).items():
        os.environ[_k] = _v

    p = spec["params"]
    xml = spec["xml"]
    tol = spec["tol"]
    res = {k: spec[k] for k in ("key", "model_id", "method")}
    res.update(
        {"metric": spec["metric"], "tol": tol, "value": None, "comment": "", "exception": ""}
    )

    # --- bngsim side (record status; do NOT short-circuit on a generic raise) ---
    bn = None
    bn_exc = ""
    bn_timing = None
    bn_wall = 0.0
    try:
        t0 = time.perf_counter()
        bn = cc.bn_ode(
            xml,
            p["t_start"],
            p["t_end"],
            p["n_points"],
            p["rtol"],
            p["atol"],
            force_dense_linear_solver=spec.get("force_dense", False),
        )
        bn_wall = time.perf_counter() - t0
        if len(bn) == 4:
            bn_timing = bn[3]
            bn = bn[:3]
    except Exception as exc:
        bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]

    # --- COPASI side (always run; it's the reference / existence proof) ---
    co = None
    co_exc = ""
    co_timing = None
    co_wall = 0.0
    try:
        t0 = time.perf_counter()
        co = cc.co_ode(xml, p["t_start"], p["t_end"], p["n_points"], p["rtol"], p["atol"])
        co_wall = time.perf_counter() - t0
        if len(co) == 4:
            co_timing = co[3]
            co = co[:3]
    except Exception as exc:
        co_exc = f"copasi: {type(exc).__name__}: {exc}"[:400]

    res["wall_sec"] = round(bn_wall + co_wall, 3)
    if co is not None:
        res["co_finite"] = bool(np.isfinite(co[1]).all())

    # --- collect timing data ---
    timing = {}
    if bn_timing:
        timing["bngsim"] = bn_timing
    if co_timing:
        timing["copasi"] = co_timing
    if warmup:
        timing["warmup"] = warmup
    if timing:
        res["timing"] = timing

    # --- classify from per-engine status (the reference anchors the verdict) ---
    if bn_exc or co_exc:
        res["status"], res["exception"] = _classify_failure(bn_exc, co_exc)
        q.put(res)
        return

    # --- both ran: compare (shared _core.differ protocol) ---
    try:
        status, value, comment, metric, m_tol = _compare_ode(bn, co)
        res["status"], res["value"], res["comment"] = status, value, comment
        res["metric"], res["tol"] = metric, m_tol
    except Exception as exc:
        res["status"] = "exception"
        res["exception"] = f"compare: {type(exc).__name__}: {exc}"[:400]
    q.put(res)


_OUTCOME = {
    "pass": Outcome.PASS,
    "diff": Outcome.DIFF,
    "exception": Outcome.EXCEPTION,
    "unsupported": Outcome.UNSUPPORTED,
    "reference_failed": Outcome.REFERENCE_FAILED,
    "bad_test": Outcome.BAD_TEST,
    "timeout": Outcome.TIMEOUT,
    "dead": Outcome.EXCEPTION,
}


def _make_progress(checkpoint_path: Path | None = None):
    def _progress(finished: int, total: int, res: dict) -> None:
        if checkpoint_path is not None:
            try:
                with open(checkpoint_path, "a") as _ck:
                    _ck.write(json.dumps(res, default=str) + "\n")
            except Exception:
                pass
        st = res.get("status", "?")
        tag = {
            "pass": "PASS",
            "diff": "DIFF",
            "exception": "ERR ",
            "unsupported": "UNSUP",
            "reference_failed": "REFFAIL",
            "bad_test": "BADTEST",
            "timeout": "SLOW",
            "dead": "DEAD",
        }.get(st, st)
        extra = ""
        if st in ("pass", "diff") and res.get("value") is not None:
            extra = f" {res['metric']}={res['value']:.3g}"
        elif st in ("exception", "dead", "reference_failed", "bad_test"):
            extra = f" {(res.get('exception') or 'died')[:64]}"
        elif st == "timeout":
            extra = f" >{res.get('cap')}s"
        print(
            f"  [{finished}/{total}] {tag} {res['model_id']} ({res['method']}){extra}", flush=True
        )

    return _progress


_BNGSIM_CONFIG_ENV_VARS = (
    "BNGSIM_CODEGEN_JIT",
    "BNGSIM_LAPACK_DENSE",
    "BNGSIM_ANALYTICAL_FUNCTIONAL_JAC",
    "BNGSIM_NO_CODEGEN",
)

_CONFIG_COMBOS: dict[str, dict] = {
    "auto": {"env": {}, "force_dense": False},
    "mir": {"env": {"BNGSIM_CODEGEN_JIT": "mir"}, "force_dense": False},
    "lapack": {"env": {"BNGSIM_LAPACK_DENSE": "1"}, "force_dense": True},
    "fd-jac": {"env": {"BNGSIM_ANALYTICAL_FUNCTIONAL_JAC": "0"}, "force_dense": False},
    "force-dense": {"env": {}, "force_dense": True},
}


def _bngsim_config_meta(args) -> dict:
    combo = getattr(args, "config", "auto")
    spec = _CONFIG_COMBOS[combo]
    env = dict.fromkeys(_BNGSIM_CONFIG_ENV_VARS)
    env.update(spec["env"])
    return {
        "combo": combo,
        "bngsim_method": "ode",
        "codegen_threshold": int(os.environ.get("BNGSIM_CODEGEN_THRESHOLD", "256")),
        "env": env,
        "force_dense_linear_solver": bool(spec["force_dense"]),
        "rtol": args.rtol,
        "atol": args.atol,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="", help="_core report path (default runs/report_ode.json)")
    ap.add_argument(
        "--manifest",
        default="",
        help="Job manifest (default: rr_parity/ode_jobs.json). Model paths resolve under rr_parity/.",
    )
    ap.add_argument(
        "--config",
        choices=sorted(_CONFIG_COMBOS),
        default="auto",
        help="bngsim method combo (default: auto). Non-auto writes report_ode__<combo>.json.",
    )
    ap.add_argument("--workers", type=int, default=4, help="Concurrent job subprocesses.")
    ap.add_argument(
        "--checkpoint",
        default="",
        help="Optional JSONL sidecar: append each completed job's raw result as it "
        "finishes, so a run killed before the final report is still reconstructable.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Per-job wall-clock cap (s); defaults to {DEFAULT_ODE_TIMEOUT}s when unset.",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"ODE integration rtol forced on BOTH engines (default {DEFAULT_RTOL}).",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"ODE integration atol forced on BOTH engines (default {DEFAULT_ATOL}).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max jobs after filtering (0=all).")
    ap.add_argument("--models", default="", help="Comma-separated model_id filter.")
    ap.add_argument("--include", default="", help="Substring filter on the model path.")
    ap.add_argument("--exclude", default="", help="Substring filter — drop matching model paths.")
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve() if args.manifest else RR_ODE_JOBS
    if not manifest.exists():
        sys.exit(f"missing manifest {manifest}; run rr_parity/build_ode_jobs.py.")
    _meta, rjobs = read_manifest(manifest)
    jobs = cc.load_and_filter(rjobs, args, suite_dir=RR_PARITY)

    if not jobs:
        sys.exit("no jobs after filtering.")

    # The SBML corpus lives under rr_parity/models/ — resolve every job's path there.
    missing = [j.model_id for j in jobs if not cc.model_path(RR_PARITY, j).exists()]
    if missing:
        sys.exit(
            f"{len(missing)} job(s) have no vendored SBML (e.g. {missing[:3]}). "
            "Run `python rr_parity/materialize.py` to place the gitignored model tree."
        )

    specs = []
    n_tol_ov = 0
    combo_spec = _CONFIG_COMBOS[args.config]
    for j in jobs:
        params = dict(j.params)
        tol_ov = _job_tol(j)
        if tol_ov:
            params["rtol"], params["atol"] = tol_ov["rtol"], tol_ov["atol"]
            n_tol_ov += 1
        else:
            params["rtol"], params["atol"] = args.rtol, args.atol
        cap = (
            args.timeout
            if args.timeout is not None
            else params.get("max_wall_sec") or DEFAULT_ODE_TIMEOUT
        )
        key = f"{j.model_id}:{j.method}"
        specs.append(
            {
                "key": key,
                "model_id": j.model_id,
                "method": j.method,
                "metric": j.oracle.metric,
                "tol": j.oracle.tol,
                "xml": str(cc.model_path(RR_PARITY, j)),
                "params": params,
                "cap": float(cap),
                "config_env": dict(combo_spec["env"]),
                "force_dense": bool(combo_spec["force_dense"]),
            }
        )

    ver = versions.stamp("copasi")
    ver["copasi"] = cc.copasi_version()
    ver["sundials"] = cc.sundials_version()
    print("=" * 72)
    print("  bngsim vs COPASI — SBML ODE parity (copasi_parity)")
    print("=" * 72)
    print(f"  jobs: {len(specs)}   workers: {args.workers}   (ODE-only)")
    print(f"  bngsim {ver['bngsim']}   copasi {ver.get('copasi')}   manifest {manifest.name}")
    print(
        f"  ODE integration tol (both engines): rtol={args.rtol:g} atol={args.atol:g}   protocol: _core.differ"
    )
    print()

    t0 = time.perf_counter()
    raw = cc.schedule(
        specs,
        _worker,
        workers=args.workers,
        timeout_of=lambda s: s["cap"],
        on_done=_make_progress(Path(args.checkpoint).resolve() if args.checkpoint else None),
    )
    elapsed = time.perf_counter() - t0

    results = []
    now = _dt.datetime.now().isoformat(timespec="seconds")
    for r in raw:
        outcome = _OUTCOME.get(r.get("status"), Outcome.EXCEPTION)
        wall = r.get("wall_sec") or 0.0
        comment = r.get("comment", "")
        if r.get("status") == "timeout":
            comment = f"killed at {r.get('cap')}s wall cap"
        elif r.get("status") == "dead":
            comment = f"worker died (exit={r.get('exitcode')})"
        refusal = None
        if outcome == Outcome.REFERENCE_FAILED:
            refusal = cc.classify_reference_refusal(r.get("exception", ""))
            comment = (
                f"{comment} | copasi refusal={refusal}" if comment else f"copasi refusal={refusal}"
            )
        results.append(
            JobResult(
                model_id=r["model_id"],
                method=r["method"],
                reference_engine="copasi",
                outcome=str(outcome),
                metric=r.get("metric"),
                value=r.get("value"),
                tol=r.get("tol"),
                exception=r.get("exception", ""),
                wall_sec=round(wall, 3) if wall else None,
                timestamp=now,
                versions=ver,
                comment=comment,
                reference_refusal=refusal,
                timing=r.get("timing"),
            )
        )

    results.sort(key=lambda x: (x.model_id, x.method))
    counts = tally(r.outcome for r in results)
    from collections import Counter

    refusal_breakdown = dict(
        Counter(r.reference_refusal for r in results if r.reference_refusal).most_common()
    )
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        suffix = "" if args.config == "auto" else f"__{args.config}"
        out_path = HERE / "runs" / f"report_ode{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite": "copasi_parity",
        "reference_engine": "copasi",
        "regime": "ode",
        "git_rev": git_rev(str(HERE)),
        "versions": ver,
        "tally": counts,
        "n_jobs": len(results),
        "elapsed_sec": round(elapsed, 2),
        "hardware": cc.hardware_info(),
        "concurrency": {"workers": args.workers, "mode": "process-parallel"},
        "config": _bngsim_config_meta(args),
        "integration_tol": {"rtol": args.rtol, "atol": args.atol, "applied_to": "both engines"},
        "reference_refusal_breakdown": {
            "counts": refusal_breakdown,
            "settled": sorted(SETTLED_REFUSALS),
            "note": (
                "Sub-classification of the REFERENCE_FAILED bucket (bngsim ran, "
                "COPASI refused), auto-derived from the COPASI failure message. "
                "Buckets: feature_gap (COPASI could not import the SBML), integrator "
                "(LSODA failed), other. COPASI has no per-model compile step."
            ),
        },
        "overrides": {
            "tol_overridden_jobs": n_tol_ov,
            "note": (
                "Only engine-agnostic tol overrides (ill-conditioned IVPs, applied "
                "to BOTH engines) are honored. rr_parity's known_artifact / "
                "invalid_reference / no_oracle_adjudicated overrides are calibrated "
                "against RoadRunner and are intentionally NOT applied here, so COPASI "
                "adjudicates every model independently."
            ),
        },
        "oracle_basis": (
            "Cross-engine NUMERIC tolerance (never byte-identical), via the shared "
            "_core.differ protocol. Both engines are forced to a tight shared "
            f"integration tolerance (rtol={args.rtol:g}, atol={args.atol:g}); the "
            "SED-ML per-job rtol/atol in the manifest are provenance only. ODE: "
            "deterministic_verdict over the common (intersection) species in "
            "concentration units. Failure attributed per-engine (both always run): "
            "bngsim raised + COPASI ran -> EXCEPTION (bngsim bug); bngsim ran + "
            "COPASI raised -> REFERENCE_FAILED (non-scoring); both raised -> "
            "BAD_TEST; wall-cap overrun -> TIMEOUT. The reference engine is COPASI "
            "(deterministic LSODA integrator)."
        ),
    }
    write_report(out_path, results, meta=meta)

    print()
    print("=" * 72)
    print(
        "  "
        + "  ".join(f"{k}: {v}" for k, v in counts.items() if v)
        + f"   elapsed {elapsed:.1f}s"
    )
    if n_tol_ov:
        print(f"  overrides: {n_tol_ov} tol-overridden")
    if refusal_breakdown:
        parts = [f"{cls}={n}" for cls, n in refusal_breakdown.items()]
        print(f"  REFERENCE_FAILED by refusal: {', '.join(parts)}")
    print(f"  report: {out_path}")
    print("=" * 72)
    n_fail = sum(counts.get(o.value, 0) for o in FAILING)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
