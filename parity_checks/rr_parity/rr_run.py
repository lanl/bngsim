#!/usr/bin/env python3
"""Run the rr_parity sweep and emit a ``_core`` report.

The SBML analog of bng_parity's ``bng_ode_run.py``, but simpler: both engines are
in-process Python (bngsim + libRoadRunner), so there is no two-output-tree diff —
each job runs both engines in one disposable subprocess and compares them directly.

For every job in ``ode_jobs.json`` (the committed spec manifest):

  * load the vendored SBML into bngsim (``Model.from_sbml``) and RoadRunner;
  * run one deterministic run per engine (cvode vs bngsim ode) at the forced
    shared ``rtol``/``atol``;
  * apply the job's oracle — ``max_rel_err`` over the common species;
  * resolve to one ``_core`` ``Outcome`` and write a ``JobResult``.

The SSA regime was retired here (GH #108): the broad bngsim-vs-RoadRunner SSA
cross-engine screen now lives in ``ssa_screen.py`` (richer third-oracle
attribution that subsumes the bare ``ensemble_verdict``), so this runner is
ODE-only. ``rr_golden.py`` still emits SSA golden references from
``ssa_jobs.json``, which therefore stays.

Cross-engine comparison is **numeric tolerance only** — different engines never
go byte-identical (that's the consumer-regeneration layer, ``rr_golden.py``).
The SBML loader is on trial alongside the engine: an SBML-loader divergence is
a real ``EXCEPTION`` (bngsim refused a model the reference ran) or ``DIFF``
(disjoint species set / time grid), **never a silent pass**.

Because the reference engine (RoadRunner) is the existence proof, the verdict is
derived from *per-engine* status — both engines always run (no short-circuit on
the first raise), so a failure is attributed to the engine that caused it:

    both ran, within tolerance ................. PASS
    both ran, metric over tolerance / non-finite DIFF
    both ran, species/time grids disjoint ...... DIFF (loud, value=inf)
    bngsim raised, RoadRunner ran .............. EXCEPTION (actionable bngsim bug)
    bngsim ran, RoadRunner raised .............. REFERENCE_FAILED (no oracle; non-scoring)
    both raised ................................ BAD_TEST (no signal; non-scoring)
    wall-clock cap exceeded .................... TIMEOUT
    child segfaulted without a result .......... EXCEPTION (unattributable)

REFERENCE_FAILED and BAD_TEST are auto-derived (never a manual list), so a model
*leaving* either bucket — because RoadRunner gained support, or bngsim/the
loader was fixed — is a visible win on the next run. EXCEPTION now means *only*
"bngsim raised but the reference succeeded" = a real bngsim defect.

Each REFERENCE_FAILED row is further sub-classified by WHY RoadRunner refused
(``JobResult.reference_refusal``, via ``classify_reference_refusal``), because
"bngsim ran, RR refused" lumps together a *settled win* and a *triage lead*:
``overstrict_missing_value`` is RR refusing a valueless parameter that bngsim
provably-correctly accepts (post-#94 — the loader hard-rejects a *referenced*
genuinely-missing param, so a model bngsim still accepts had an unreferenced /
rule-defined one), a decided case needing no re-triage; ``feature_gap`` /
``integrator`` / ``recursive`` are oracle-unverified and still worth chasing.
The report's ``reference_refusal_breakdown`` is the regression tracker: the
settled count jumping back up means the loader gate regressed.

Each job runs in its own spawned subprocess that the parent kills on overrun,
so a RoadRunner segfault on a pathological model can't take down the screen (but
it also leaves no per-engine status, hence that row maps to EXCEPTION).

Usage:
    cd bngsim && .venv/bin/python parity_checks/rr_parity/rr_run.py \\
        --limit 20 --workers 4
    .venv/bin/python parity_checks/rr_parity/rr_run.py \\
        --models BIOMD0000000012,BIOMD0000000010

Output: runs/report_ode.json (a _core report; runs/ is gitignored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so the spawned child can import _rr_common / rr_run

import _rr_common as rc  # noqa: E402
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

ODE_JOBS = HERE / "ode_jobs.json"
DEFAULT_ODE_TIMEOUT = 120.0  # ODE jobs carry no max_wall_sec; this is their cap.


def _job_overrides(
    job,
) -> tuple[
    dict | None,
    tuple[str, str | None] | None,
    tuple[str, str | None] | None,
    tuple[str, str, str | None] | None,
]:
    """Read (tol, known_artifact, invalid_reference, adjudication) from ``Job.overrides``.

    Returns ``(tol, artifact, invalid_ref, adjudication)`` where ``tol`` is
    ``{"rtol":..,"atol":..}`` or None (applied identically to both engines,
    overriding the global default); ``artifact`` is ``(reason, issue)`` or None
    (a DIFF on this job is an accepted divergence, reclassified to PASS);
    ``invalid_ref`` is ``(reason, issue)`` or None (the reference engine ran but
    produced no usable trajectory, reclassified to BAD_TEST while the premise
    holds); and ``adjudication`` is ``(verdict, reason, issue)`` or None (a
    REFERENCE_FAILED row settled by an independent oracle — see ``overrides.py``
    ``NO_ORACLE_ADJUDICATED``). See ``overrides.py`` for the authored entries.
    """
    tol = None
    artifact = None
    invalid_ref = None
    adjudication = None
    for o in job.overrides:
        if o.field == "tol":
            tol = {"rtol": float(o.value["rtol"]), "atol": float(o.value["atol"])}
        elif o.field == "known_artifact":
            issue = (o.value or {}).get("issue") if isinstance(o.value, dict) else None
            artifact = (o.reason, issue)
        elif o.field == "invalid_reference":
            issue = (o.value or {}).get("issue") if isinstance(o.value, dict) else None
            invalid_ref = (o.reason, issue)
        elif o.field == "no_oracle_adjudicated":
            v = o.value if isinstance(o.value, dict) else {}
            adjudication = (v.get("verdict") or "confirm", o.reason, v.get("issue"))
    return tol, artifact, invalid_ref, adjudication


# The shared tight integration tolerance (rc.DEFAULT_RTOL/ATOL) is forced on BOTH
# engines; the per-job SED-ML rtol/atol in the manifest are provenance only.
DEFAULT_RTOL = rc.DEFAULT_RTOL
DEFAULT_ATOL = rc.DEFAULT_ATOL


# --------------------------------------------------------------------------- #
# Comparison (the oracle, applied in the worker)
# --------------------------------------------------------------------------- #
def _compare_ode(bn, rr) -> tuple[str, float, str, str, float]:
    """(status, value, comment, metric, tol) for one deterministic job.

    Applies the shared ``_core.differ.deterministic_verdict`` protocol (combined
    abs+rel per-cell tolerance + fail-fraction budget + hard ceilings) over the
    common species. The reported metric is ``max_rel_err`` = the post-budget
    worst remaining relative error (0.0 on PASS), gated at ``REL_TOL``.
    """
    bn_t, bn_v, bn_n = bn
    rr_t, rr_v, rr_n = rr
    metric, tol = "max_rel_err", differ.REL_TOL
    if bn_t.shape != rr_t.shape or not np.allclose(bn_t, rr_t, rtol=0, atol=1e-9):
        return (
            "diff",
            float("inf"),
            f"time grid mismatch (bn n={bn_t.shape}, rr n={rr_t.shape})",
            metric,
            tol,
        )
    align = rc.align_common(bn_n, rr_n)
    if align is None:
        return (
            "diff",
            float("inf"),
            f"disjoint species sets: bn={bn_n[:4]} rr={rr_n[:4]}",
            metric,
            tol,
        )
    bn_idx, rr_idx, common = align
    v = differ.deterministic_verdict(bn_v[:, bn_idx], rr_v[:, rr_idx])
    status = "pass" if v["passed"] else "diff"
    comment = (
        f"{len(common)} sp; fail {v['n_fail']}/{v['n_cells']} "
        f"(hard {v['n_hard_fail']}, soft {v['n_soft_fail']}, forgiven {v['budget_forgiven']})"
    )
    return status, v["max_rel"], comment, metric, tol


# --------------------------------------------------------------------------- #
# Worker (module-level so it is picklable under the 'spawn' start method)
# --------------------------------------------------------------------------- #
def classify_reference_refusal(rr_exc: str) -> str:
    """Sub-classify WHY RoadRunner refused, for a REFERENCE_FAILED row.

    REFERENCE_FAILED (bngsim ran, RR raised) is a single mechanical bucket but
    carries several distinct meanings. This splits it so a future run can skip
    the *settled* cases and only re-triage the unverified ones — the field is
    recorded on the report (``JobResult.reference_refusal``), never a manual list.

    Vocabulary:

    - ``overstrict_missing_value`` — RR refuses a valueless parameter (``Global
      parameter X is missing a value``). Reaching this row means bngsim *accepted*
      the same model, which (post-#94, where bngsim hard-rejects a *referenced*
      genuinely-missing param) provably means X was unreferenced or rule-defined —
      i.e. RR is over-strict and bngsim's acceptance is correct **by
      construction**. SETTLED: a future run need not re-investigate. (This marks
      the refusal as unjustified, not bngsim's trajectory as oracle-verified.)
    - ``feature_gap`` — RR genuinely lacks a feature (fast reactions, delay /
      DDE, an unsupported csymbol). bngsim ran but there is no oracle, so
      correctness is UNVERIFIED — triage-worthy (wants a third oracle).
    - ``integrator`` — RR's CVODE bailed. The model may be ill-posed for both
      engines; bngsim's solve is also suspect. Triage-worthy.
    - ``recursive`` — RR rejected a recursive rule/assignment. Triage-worthy.
    - ``other`` — unclassified; inspect the exception text.
    """
    e = (rr_exc or "").lower()
    if "missing a value" in e or "missing value" in e:
        return "overstrict_missing_value"
    if ("fast" in e and "reaction" in e) or "delay differential" in e or "dde" in e:
        return "feature_gap"
    if "not physically stored" in e or ("symbol" in e and "max" in e):
        return "feature_gap"
    if "cvode" in e:
        return "integrator"
    if "recursive" in e:
        return "recursive"
    return "other"


# Refusal classes that are SETTLED — a future screen should not re-triage them.
# ``overstrict_missing_value``: bngsim's run is justified by construction (RR is
# over-strict). ``adjudicated_confirm`` / ``adjudicated_uncoverable``: the row was
# settled by an INDEPENDENT oracle (a confirmed bngsim solve, or a model with no
# possible ground truth) via overrides.NO_ORACLE_ADJUDICATED (#117). The rest are
# unverified.
SETTLED_REFUSALS = frozenset(
    {"overstrict_missing_value", "adjudicated_confirm", "adjudicated_uncoverable"}
)


def _classify_failure(bn_exc: str, rr_exc: str) -> tuple[str, str]:
    """Map per-engine generic-raise status to a ``(status, exception)`` pair.

    The reference engine (RoadRunner) is the existence proof, so the engine that
    failed determines the bucket. Each argument is that engine's exception string
    (empty == the engine ran successfully). A declared bngsim SSA refusal is
    handled upstream as UNSUPPORTED and never reaches here, and the caller only
    invokes this when at least one engine raised — so the all-ran case is absent.
    """
    if bn_exc and rr_exc:  # neither engine ran -> the test/model is the problem
        return "bad_test", f"{rr_exc} || {bn_exc}"
    if rr_exc:  # bngsim ran but the reference couldn't -> nothing to compare against
        return "reference_failed", rr_exc
    return "exception", bn_exc  # the reference ran but bngsim raised -> bngsim bug


def _apply_invalid_reference(outcome, rr_finite, reason: str, issue, comment: str):
    """Resolve an INVALID_REFERENCE override against one job's natural result.

    Returns ``(outcome, comment, is_stale)``. The override holds only while its
    premise does — bngsim failed (natural EXCEPTION or BAD_TEST) AND the
    reference is unusable (RoadRunner raised, so ``rr_finite is None``, or ran
    with non-finite output, ``rr_finite is False``). Then the job is reclassified
    to ``BAD_TEST`` with the reason recorded. Otherwise the premise is broken — a
    finite reference (``rr_finite is True``) or a now-running bngsim (natural
    PASS/DIFF/REFERENCE_FAILED) — so the natural outcome is kept and ``is_stale``
    is True, surfacing both the recovery and any real bngsim bug the override
    would have masked. ``rr_finite`` is True/False when RR ran, None otherwise.
    """
    tag = f" ({issue})" if issue else ""
    if outcome == Outcome.BAD_TEST or (outcome == Outcome.EXCEPTION and rr_finite is False):
        if outcome == Outcome.BAD_TEST:  # already both-broken; just append context
            extra = f" | {comment}" if comment else ""
        else:  # record the natural verdict so "bngsim also failed" stays visible
            extra = f" | was {outcome}" + (f": {comment}" if comment else "")
        return Outcome.BAD_TEST, f"invalid reference{tag}: {reason}{extra}", False
    note = (f"{comment} | " if comment else "") + (
        f"STALE invalid-reference entry{tag}: natural={outcome}, rr_finite={rr_finite}"
        " — reference usable / bngsim runs now, re-triage or prune"
    )
    return outcome, note, True


def _worker(spec: dict, q) -> None:
    """Run BOTH engines for one job and put a result dict on ``q``.

    The reference engine is the existence proof, so the verdict is derived from
    *per-engine* status rather than short-circuiting on the first raise:

      * both engines ran -> compare (``pass``/``diff``);
      * bngsim ok, RoadRunner raised -> ``reference_failed`` (no trajectory to
        compare against — not a bngsim defect);
      * bngsim raised, RoadRunner ok -> ``exception`` (actionable bngsim bug);
      * both raised -> ``bad_test`` (no parity signal, the model is the problem).

    The parent synthesizes TIMEOUT (kill on overrun) and ``dead`` (a child that
    segfaulted without a result — unattributable, so bucketed EXCEPTION).
    """
    # Per-process warmup (SymPy import for bngsim, LLVM/JIT init for RR) measured
    # ONCE here, before any model load, so the one-time SymPy import is attributed
    # to warmup rather than charged to the first model's parse/interpret. Each job
    # is its own subprocess → one warmup sample per job; the matrix aggregates them
    # (mean/median/std). Runs before set_rr_quiet so both imports are timed cold.
    warmup = rc.measure_warmup()

    rc.set_rr_quiet()

    # Apply the selected combo's override env vars here, inside the spawned
    # worker and before bngsim loads the model — bngsim reads these lazily at
    # load/codegen/run time, so setting them now is what makes --config take
    # effect (a parent os.environ mutation would not reliably reach this worker).
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
        bn = rc.bn_ode(
            xml,
            p["t_start"],
            p["t_end"],
            p["n_points"],
            p["rtol"],
            p["atol"],
            force_dense_linear_solver=spec.get("force_dense", False),
        )
        bn_wall = time.perf_counter() - t0
        # bn_ode now returns (time, species, names, timing)
        if len(bn) == 4:
            bn_timing = bn[3]
            bn = bn[:3]  # keep backward compat for tuple unpacking below
    except Exception as exc:
        bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]

    # --- RoadRunner side (always run; it's the reference / existence proof) ---
    rr = None
    rr_exc = ""
    rr_timing = None
    rr_wall = 0.0
    try:
        t0 = time.perf_counter()
        rr = rc.rr_ode(xml, p["t_start"], p["t_end"], p["n_points"], p["rtol"], p["atol"])
        rr_wall = time.perf_counter() - t0
        # rr_ode now returns (time, species, names, timing)
        if len(rr) == 4:
            rr_timing = rr[3]
            rr = rr[:3]  # keep backward compat for tuple unpacking below
    except Exception as exc:
        rr_exc = f"roadrunner: {type(exc).__name__}: {exc}"[:400]

    # Total job wall — bngsim + reference. The per-phase `timing` dict carries the
    # fine-grained breakdown; this single number supersedes the retired
    # wall_bn/wall_rr split (T6.1).
    res["wall_sec"] = round(bn_wall + rr_wall, 3)
    if rr is not None:
        # Record whether the reference trajectory is usable as an oracle: a run
        # that emits NaN/Inf is no existence proof. Only metadata — the verdict
        # still keys on raised-vs-not — but it is what makes an INVALID_REFERENCE
        # override self-maintaining (a finite reference makes it stale at once).
        # Computed outside the try so it can never be miscaught as an RR raise.
        res["rr_finite"] = bool(np.isfinite(rr[1]).all())

    # --- collect timing data ---
    timing = {}
    if bn_timing:
        timing["bngsim"] = bn_timing
    if rr_timing:
        timing["roadrunner"] = rr_timing
    # Per-process warmup (job-level, not per-engine-block) — always recorded, even
    # when both engines raised, so the matrix can aggregate it over every job.
    if warmup:
        timing["warmup"] = warmup
    if timing:
        res["timing"] = timing

    # --- classify from per-engine status (the reference anchors the verdict) ---
    if bn_exc or rr_exc:
        res["status"], res["exception"] = _classify_failure(bn_exc, rr_exc)
        q.put(res)
        return

    # --- both ran: compare (shared _core.differ protocol; metric realized here) ---
    try:
        status, value, comment, metric, m_tol = _compare_ode(bn, rr)
        res["status"], res["value"], res["comment"] = status, value, comment
        res["metric"], res["tol"] = metric, m_tol
    except Exception as exc:
        res["status"] = "exception"
        res["exception"] = f"compare: {type(exc).__name__}: {exc}"[:400]
    q.put(res)


# Worker status -> _core Outcome. (timeout/dead are synthesized by the scheduler;
# dead = a child that segfaulted with no per-engine status, so it can't be
# attributed to an engine — bucketed EXCEPTION as the conservative "investigate".)
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


def _make_progress(artifacts: dict[str, tuple[str, str | None]]):
    """Progress printer that flags a known-artifact DIFF as it will be reclassified.

    The worker reports the raw cross-engine verdict (it knows nothing about the
    allow-list); the parent reclassifies a known-artifact DIFF to PASS. So the
    live line shows ``DIFF→PASS*`` for those, matching the final report instead
    of misleadingly printing a bare DIFF that the summary then counts as PASS.
    """

    def _progress(finished: int, total: int, res: dict) -> None:
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
        if st == "diff" and res.get("key") in artifacts:
            tag = "DIFF→PASS*"  # known artifact; reclassified in the report
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


# The bngsim override knobs that change what method="ode" auto-selects per
# problem. Recorded in _meta["config"] so the report/HTML can state how the
# sweep was run and the per-combo matrices (T3) can be told apart. A value of
# None means the var was unset — i.e. the engine's own default for that knob.
_BNGSIM_CONFIG_ENV_VARS = (
    "BNGSIM_CODEGEN_JIT",  # 'mir' → in-process MIR micro-JIT instead of cc+dlopen
    "BNGSIM_LAPACK_DENSE",  # '1' → Accelerate dgetrf dense factor (GH #84)
    "BNGSIM_ANALYTICAL_FUNCTIONAL_JAC",  # '0' → force finite-difference Jacobian
    "BNGSIM_NO_CODEGEN",  # set → never codegen (force ExprTk)
)

# The bngsim method combos, defined ONCE. --config NAME selects one. Its env vars
# are applied per-worker (set inside _worker from the spec, NOT via a parent
# os.environ mutation — under the spawn start method a late parent mutation does
# not reliably reach an already-launched worker; carrying it on the spec makes
# the sweep self-describing). force_dense is a Simulator kwarg (not an env var),
# threaded through the spec into bn_ode. "auto" = the engine's own per-problem
# heuristic with no overrides (the production default).
#
# LAPACK-dense subtlety: SUNLinSol_LapackDense (LinearSolverKind 2) is only ever
# chosen on the DENSE linear-solver path; a KLU-eligible model (N>=50, density
# <10%) stays on KLU regardless of BNGSIM_LAPACK_DENSE. So the `lapack` combo
# ALSO forces the dense path — otherwise it never reaches dgetrf and is a no-op
# vs `auto` for the sparse-majority corpus. It is therefore the LAPACK-dgetrf
# counterpart of `force-dense` (built-in dense LU): comparing the two isolates
# dgetrf vs the SUNDIALS dense factor on the same forced-dense models, which is
# exactly the data the graduation decision (Phase 5) needs.
_CONFIG_COMBOS: dict[str, dict] = {
    "auto": {"env": {}, "force_dense": False},
    "mir": {"env": {"BNGSIM_CODEGEN_JIT": "mir"}, "force_dense": False},
    "lapack": {"env": {"BNGSIM_LAPACK_DENSE": "1"}, "force_dense": True},
    "fd-jac": {"env": {"BNGSIM_ANALYTICAL_FUNCTIONAL_JAC": "0"}, "force_dense": False},
    "force-dense": {"env": {}, "force_dense": True},
}


def _bngsim_config_meta(args) -> dict:
    """The bngsim method combo this sweep ran under (for _meta["config"]).

    ``combo`` names the knob set (``--config``); ``env`` reports the override
    knobs the combo applies in each worker (None = unset = engine default),
    ``codegen_threshold`` is the species count at/above which auto compiles native
    C, ``force_dense_linear_solver`` is the Simulator kwarg, and rtol/atol are the
    shared integration tolerance. Built from the combo definition — the same
    single source the workers read — so the report can't drift from what ran.
    """
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
        "--config",
        choices=sorted(_CONFIG_COMBOS),
        default="auto",
        help=(
            "bngsim method combo (default: auto = the engine's per-problem heuristic). "
            "mir/fd-jac set an override env var in each worker; force-dense forces the "
            "dense linear solver (built-in LU); lapack forces dense AND the LAPACK dgetrf "
            "factor (it must force dense, else KLU-eligible models never reach dgetrf). "
            "Non-auto writes runs/report_ode__<combo>.json."
        ),
    )
    ap.add_argument("--workers", type=int, default=4, help="Concurrent job subprocesses.")
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
        help=f"ODE integration rtol forced on BOTH engines (default {DEFAULT_RTOL}; "
        "the SED-ML per-job rtol is provenance only, not used for parity).",
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
    ap.add_argument(
        "--jobs",
        type=Path,
        default=ODE_JOBS,
        help="Job spec manifest to run (default ode_jobs.json). Point at "
        "smoke/ode_jobs_smoke.json to run the committed hermetic subset (no materialize).",
    )
    args = ap.parse_args()

    jobs_manifest = args.jobs if args.jobs.is_absolute() else (HERE / args.jobs)
    if not jobs_manifest.exists():
        sys.exit(f"missing {jobs_manifest}; run build_ode_jobs.py first.")
    _meta, rjobs = read_manifest(jobs_manifest)
    jobs = rc.load_and_filter(rjobs, args, suite_dir=HERE)

    if not jobs:
        sys.exit("no jobs after filtering.")

    # Verify the vendored SBML is materialized before spawning anything.
    missing = [j.model_id for j in jobs if not rc.model_path(HERE, j).exists()]
    if missing:
        sys.exit(
            f"{len(missing)} job(s) have no vendored SBML (e.g. {missing[:3]}). "
            "Run `python materialize.py` to place the gitignored model tree."
        )

    specs = []
    artifacts: dict[str, tuple[str, str | None]] = {}  # key -> (reason, issue)
    invalid_refs: dict[str, tuple[str, str | None]] = {}  # key -> (reason, issue)
    adjudications: dict[str, tuple[str, str, str | None]] = {}  # key -> (verdict, reason, issue)
    n_tol_ov = 0
    # The selected combo's overrides, threaded onto every spec so each spawned
    # worker applies them itself (env + force_dense kwarg) — see _CONFIG_COMBOS.
    combo_spec = _CONFIG_COMBOS[args.config]
    for j in jobs:
        params = dict(j.params)
        tol_ov, artifact, invalid_ref, adjudication = _job_overrides(j)
        # Force the shared parity integration tolerance on BOTH engines; the
        # SED-ML rtol/atol in the manifest stay as horizon/figure provenance. A
        # per-model TOL_OVERRIDES entry (ill-conditioned IVP) wins over the
        # shared default — still applied identically to both engines.
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
        if artifact:
            artifacts[key] = artifact
        if invalid_ref:
            invalid_refs[key] = invalid_ref
        if adjudication:
            adjudications[key] = adjudication
        specs.append(
            {
                "key": key,
                "model_id": j.model_id,
                "method": j.method,
                "metric": j.oracle.metric,
                "tol": j.oracle.tol,
                "xml": str(rc.model_path(HERE, j)),
                "params": params,
                "cap": float(cap),
                "config_env": dict(combo_spec["env"]),
                "force_dense": bool(combo_spec["force_dense"]),
            }
        )

    ver = versions.stamp("roadrunner")
    # The SUNDIALS/CVODE build version (constant across the sweep), so the matrix
    # can name the exact integrator both engines share — see sundials_version().
    ver["sundials"] = rc.sundials_version()
    print("=" * 72)
    print("  bngsim vs RoadRunner — SBML parity (rr_parity)")
    print("=" * 72)
    print(f"  jobs: {len(specs)}   workers: {args.workers}   (ODE-only; SSA → ssa_screen.py)")
    print(f"  bngsim {ver['bngsim']}   roadrunner {ver['roadrunner']}")
    print(
        f"  ODE integration tol (both engines): rtol={args.rtol:g} atol={args.atol:g}   protocol: _core.differ"
    )
    print()

    t0 = time.perf_counter()
    raw = rc.schedule(
        specs,
        _worker,
        workers=args.workers,
        timeout_of=lambda s: s["cap"],
        on_done=_make_progress(artifacts),
    )
    elapsed = time.perf_counter() - t0

    results = []
    n_reclassified = 0
    n_stale_artifact = 0
    n_invalid_ref = 0
    n_stale_invalid_ref = 0
    n_adjudicated = 0
    n_stale_adjudication = 0
    now = _dt.datetime.now().isoformat(timespec="seconds")
    for r in raw:
        outcome = _OUTCOME.get(r.get("status"), Outcome.EXCEPTION)
        wall = r.get("wall_sec") or 0.0
        comment = r.get("comment", "")
        if r.get("status") == "timeout":
            comment = f"killed at {r.get('cap')}s wall cap"
        elif r.get("status") == "dead":
            comment = f"worker died (exit={r.get('exitcode')})"
        # Known-artifact allow-list: an accepted/genuine divergence (bngsim
        # correct, or a degenerate-model comparison) is reclassified DIFF -> PASS
        # with the reason in the comment — loud, never silent. If the key passed
        # on its own the entry is stale (bngsim was fixed); flag it so it gets
        # pruned rather than silently masking a future regression.
        artifact = artifacts.get(r.get("key"))
        if artifact:
            reason, issue = artifact
            tag = f" ({issue})" if issue else ""
            if outcome == Outcome.DIFF:
                outcome = Outcome.PASS
                comment = f"known artifact{tag}: {reason}" + (f" | {comment}" if comment else "")
                n_reclassified += 1
            elif outcome == Outcome.PASS:
                comment = (f"{comment} | " if comment else "") + (
                    f"STALE known-artifact entry{tag}: passes on its own now — prune it"
                )
                n_stale_artifact += 1
        # Invalid-reference disposition: the reference engine ran but produced no
        # usable trajectory, so a job that landed in EXCEPTION/BAD_TEST is really
        # "both broken" -> BAD_TEST with the reason in the comment. Applied ONLY
        # while the premise holds (bngsim failed AND no usable reference — RR
        # raised, or ran with non-finite output); a finite reference or a
        # now-running bngsim flags the entry STALE so recovery (and any real
        # bngsim bug the override was masking) resurfaces — never silent.
        invalid_ref = invalid_refs.get(r.get("key"))
        if invalid_ref:
            reason, issue = invalid_ref
            outcome, comment, stale = _apply_invalid_reference(
                outcome, r.get("rr_finite"), reason, issue, comment
            )
            n_stale_invalid_ref += int(stale)
            n_invalid_ref += int(not stale)
        # Sub-classify a REFERENCE_FAILED row by WHY the reference refused, so a
        # later run skips the settled cases (RR over-strict on a valueless param
        # bngsim correctly accepts) and only re-triages the unverified ones
        # (feature gaps / integrator bails). Recorded on the row; auto-derived
        # from the reference's exception, never a manual list.
        refusal = None
        if outcome == Outcome.REFERENCE_FAILED:
            refusal = classify_reference_refusal(r.get("exception", ""))
            tag = "settled" if refusal in SETTLED_REFUSALS else "triage"
            note = f"reference refusal={refusal} ({tag})"
            comment = f"{comment} | {note}" if comment else note
        # NO_ORACLE adjudication: a REFERENCE_FAILED row settled by an independent
        # oracle (#117). It does NOT change the (non-scoring) outcome — it flips
        # the auto-derived refusal class to a SETTLED ``adjudicated_<verdict>`` and
        # records the verdict + evidence, so a future sweep skips re-triage. Holds
        # ONLY while the row is genuinely REFERENCE_FAILED: if RR now runs it, or
        # bngsim's status changed, the entry is flagged STALE (the recovery / any
        # newly-surfaced bngsim defect can never be silently masked).
        adjudication = adjudications.get(r.get("key"))
        if adjudication:
            verdict, reason, issue = adjudication
            tag = f" ({issue})" if issue else ""
            if outcome == Outcome.REFERENCE_FAILED:
                rr_refusal = refusal  # preserve provenance of WHY RR refused
                refusal = f"adjudicated_{verdict}"
                comment = (f"{comment} | " if comment else "") + (
                    f"adjudicated {verdict}{tag} (was reference refusal={rr_refusal}): {reason}"
                )
                n_adjudicated += 1
            else:
                comment = (f"{comment} | " if comment else "") + (
                    f"STALE no-oracle-adjudication{tag}: natural={outcome} (no longer "
                    "REFERENCE_FAILED) — reference runs / bngsim status changed, re-triage or prune"
                )
                n_stale_adjudication += 1
        results.append(
            JobResult(
                model_id=r["model_id"],
                method=r["method"],
                reference_engine="roadrunner",
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
    # Breakdown of REFERENCE_FAILED by refusal class — the settled (RR over-strict,
    # bngsim correct by construction) vs triage-worthy (unverified) split.
    from collections import Counter

    refusal_breakdown = dict(
        Counter(r.reference_refusal for r in results if r.reference_refusal).most_common()
    )
    n_settled_reffail = sum(v for k, v in refusal_breakdown.items() if k in SETTLED_REFUSALS)
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        # Non-auto combos write report_ode__<combo>.json so per-combo sweeps
        # (T3.2/T3.3) don't clobber each other or the auto baseline.
        suffix = "" if args.config == "auto" else f"__{args.config}"
        out_path = HERE / "runs" / f"report_ode{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite": "rr_parity",
        "reference_engine": "roadrunner",
        "regime": "ode",
        "git_rev": git_rev(str(HERE)),
        "versions": ver,
        "tally": counts,
        "n_jobs": len(results),
        "elapsed_sec": round(elapsed, 2),
        # CPU/OS the timings were measured on, and the worker concurrency: each job
        # is its own process and `workers` of them run at once, so per-integration
        # timings were collected under up-to-`workers`-way CPU contention (the
        # renderer surfaces this caveat). Recorded so absolute costs are interpretable.
        "hardware": rc.hardware_info(),
        "concurrency": {"workers": args.workers, "mode": "process-parallel"},
        # The bngsim method combo this sweep ran under (T2.1) — combo name, the
        # override env vars / kwargs in effect, codegen threshold, integration tol.
        "config": _bngsim_config_meta(args),
        "integration_tol": {"rtol": args.rtol, "atol": args.atol, "applied_to": "both engines"},
        "reference_refusal_breakdown": {
            "counts": refusal_breakdown,
            "settled": sorted(SETTLED_REFUSALS),
            "n_settled": n_settled_reffail,
            "note": (
                "Sub-classification of the REFERENCE_FAILED bucket (bngsim ran, "
                "RoadRunner refused), auto-derived from the reference exception — "
                "never a manual list. 'settled' classes (overstrict_missing_value: "
                "RR refuses a valueless parameter that bngsim provably-correctly "
                "accepts, post-#94) are decided wins and need no re-triage; the "
                "rest (feature_gap / integrator / recursive / other) are "
                "oracle-unverified and triage-worthy. A jump in the settled count "
                "back toward the old lumped total means the #94 loader gate "
                "regressed; a new non-settled class is a fresh lead."
            ),
        },
        "overrides": {
            "tol_overridden_jobs": n_tol_ov,
            "known_artifacts_reclassified": n_reclassified,
            "stale_known_artifacts": n_stale_artifact,
            "invalid_reference_reclassified": n_invalid_ref,
            "stale_invalid_reference": n_stale_invalid_ref,
            "no_oracle_adjudicated": n_adjudicated,
            "stale_no_oracle_adjudication": n_stale_adjudication,
            "note": (
                "Overrides are read from Job.overrides (baked by build_*_jobs.py "
                "from overrides.py). A tol override applies a per-model rtol/atol "
                "to BOTH engines; a known_artifact reclassifies a DIFF to PASS, an "
                "invalid_reference reclassifies an EXCEPTION/BAD_TEST to BAD_TEST "
                "(reference ran but non-finite), and a no_oracle_adjudicated marks a "
                "REFERENCE_FAILED row's refusal class SETTLED (verdict confirm/"
                "uncoverable, independent oracle, #117) without changing the "
                "non-scoring outcome — all with the reason in JobResult.comment. "
                "Stale = the premise no longer holds (passes on its own / the "
                "reference is usable again / no longer REFERENCE_FAILED), flagged "
                "not silently kept."
            ),
        },
        "oracle_basis": (
            "Cross-engine NUMERIC tolerance (never byte-identical), via the shared "
            "_core.differ protocol — the same verdict bng_parity uses. Both engines "
            "are forced to a tight shared integration tolerance "
            f"(rtol={args.rtol:g}, atol={args.atol:g}); the SED-ML per-job rtol/atol "
            "in the manifest are horizon/figure provenance only. ODE: "
            "deterministic_verdict over the common (intersection) species in "
            "concentration units — combined abs+rel per-cell tol with a "
            f"fail-fraction budget (<={differ.FAIL_FRAC_BUDGET}) gated by hard "
            f"ceilings (rel<={differ.HARD_REL_CEILING}, abs<={differ.HARD_ABS_CEILING_FILE}"
            "*file_scale); metric=max_rel_err = post-budget worst remaining rel "
            f"(tol={differ.REL_TOL}). Disjoint species / time-grid "
            "mismatch -> DIFF. Failure is attributed per-engine (both always run): "
            "bngsim raised + RoadRunner ran -> "
            "EXCEPTION (actionable bngsim bug); bngsim ran + RoadRunner raised -> "
            "REFERENCE_FAILED (no reference trajectory; non-scoring); both raised -> "
            "BAD_TEST (no parity signal; non-scoring); wall-cap overrun -> TIMEOUT. "
            "REFERENCE_FAILED / BAD_TEST are auto-derived from per-engine status, "
            "never a manual list, so a model leaving either bucket is a visible win."
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
    if (
        n_tol_ov
        or n_reclassified
        or n_stale_artifact
        or n_invalid_ref
        or n_stale_invalid_ref
        or n_adjudicated
        or n_stale_adjudication
    ):
        bits = []
        if n_tol_ov:
            bits.append(f"{n_tol_ov} tol-overridden")
        if n_reclassified:
            bits.append(f"{n_reclassified} known-artifact DIFF→PASS")
        if n_stale_artifact:
            bits.append(
                f"{n_stale_artifact} STALE artifact entr{'y' if n_stale_artifact == 1 else 'ies'}"
            )
        if n_invalid_ref:
            bits.append(f"{n_invalid_ref} invalid-reference →BAD_TEST")
        if n_stale_invalid_ref:
            bits.append(
                f"{n_stale_invalid_ref} STALE invalid-reference "
                f"entr{'y' if n_stale_invalid_ref == 1 else 'ies'}"
            )
        if n_adjudicated:
            bits.append(f"{n_adjudicated} no-oracle adjudicated (settled)")
        if n_stale_adjudication:
            bits.append(
                f"{n_stale_adjudication} STALE no-oracle-adjudication "
                f"entr{'y' if n_stale_adjudication == 1 else 'ies'}"
            )
        print("  overrides: " + ", ".join(bits))
    if refusal_breakdown:
        parts = [
            f"{cls}={n}{'*' if cls in SETTLED_REFUSALS else ''}"
            for cls, n in refusal_breakdown.items()
        ]
        print(f"  REFERENCE_FAILED by refusal: {', '.join(parts)}   (*settled, no re-triage)")
    print(f"  report: {out_path}")
    print("=" * 72)
    # Only the scoring (FAILING) outcomes gate the exit code; REFERENCE_FAILED /
    # BAD_TEST / UNSUPPORTED carry no parity signal about bngsim.
    n_fail = sum(counts.get(o.value, 0) for o in FAILING)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
