#!/usr/bin/env python3
"""bngsim vs the legacy BNG2.pl/run_network stack — ODE timing + parity (bng_parity).

The bng sibling of ``rr_parity/rr_run.py`` / ``amici_parity/amici_run.py``: it
reuses the suite's ``jobs.json`` corpus (the 591 deterministic/ODE jobs) and the
shared ``_core.differ`` oracle, but the reference engine is the legacy CLI stack
rather than a Python library, so each job runs through a three-step pipeline in
its own spawned worker:

  1. **Network generation (BNG2.pl)** — the shared per-model build *prefix*. The
     BNGL model (actions stripped, a bare ``generate_network`` appended) is run
     through BNG2.pl once to produce the reaction-network ``.net``. The same
     ``.net`` feeds BOTH integrators, so this cost is attributed to neither.
  2. **BNGsim** integrates the ``.net`` in-process (``Model.from_net`` →
     ``Simulator(method="ode").run``) with per-phase build timing + a cold→warm
     integration split — exactly the taxonomy rr_parity/amici_parity use.
  3. **run_network** (the legacy CVODE binary) integrates the SAME ``.net`` as a
     direct subprocess, reporting its own init/propagation CPU split.

Both trajectories are compared over the network species (the ``.cdat`` columns,
in identical network order) via ``_core.differ.deterministic_verdict``. Failure
is attributed per-engine, exactly as in the SBML suites:

  * both ran, within tolerance  -> PASS
  * both ran, exceeds tolerance  -> DIFF
  * bngsim raised, run_network ran -> EXCEPTION (actionable bngsim bug)
  * bngsim ran, run_network raised -> REFERENCE_FAILED (no oracle; non-scoring)
  * netgen failed / both raised  -> BAD_TEST (the test/model is the problem)
  * no resolvable ODE horizon    -> SKIP (not a parity signal)

OVERRIDES — like amici_parity, only the engine-agnostic ``tol`` override is
honored (an ill-conditioned IVP both engines mis-solve at the shared default).
A small catalog of documented comparison artifacts (``KNOWN_DETERMINISTIC_ARTIFACTS``
below — integrator phase-wander, a staircase function sampled on a discontinuity,
a reference-side tolerance tail) is reclassified out of the scoring DIFF bucket as
KNOWN ARTIFACT, but only while the divergence stays within a recorded magnitude
bound (a regression blows past it and re-flags). Otherwise this track adjudicates
every model independently on the network BNG2.pl produced. (These dispositions were
folded here from the retired correctness suite, GH #69.)

Usage
-----
    export BNGPATH=/path/to/BioNetGen-2.9.3      # BNG2.pl + bin/run_network
    python bng_ode_run.py --workers 4 --limit 40
    python generate_bng_matrix.py
    open runs/bng_matrix_ode.html
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves

import _bng_common as bc  # noqa: E402
from _core import (  # noqa: E402
    JobResult,
    Outcome,
    differ,
    read_manifest,
    tally,
    versions,
    write_report,
)
from _core.versions import git_rev  # noqa: E402

JOBS = HERE / "jobs.json"
DEFAULT_ODE_TIMEOUT = 240.0  # per-job wall cap: netgen + both integrators + warm reps


# --------------------------------------------------------------------------- #
# Tool resolution
# --------------------------------------------------------------------------- #
def _resolve_bng_tools() -> tuple[str, str]:
    """(BNG2.pl, run_network) from $BNGPATH/$BNG2_PL; never hardcoded.

    $BNGPATH may be the BioNetGen folder (containing BNG2.pl + bin/run_network) or
    a direct BNG2.pl path; run_network lives at ``<root>/bin/run_network``.
    """
    cand = os.environ.get("BNGPATH") or os.environ.get("BNG2_PL")
    if not cand:
        sys.exit(
            "ABORT: set $BNGPATH (the BioNetGen folder with BNG2.pl) or $BNG2_PL "
            "(a direct BNG2.pl path) so the reference stack (BNG2.pl netgen + "
            "run_network ODE) is resolvable. No path is hardcoded."
        )
    p = Path(cand)
    if not p.exists():
        sys.exit(f"ABORT: $BNGPATH/$BNG2_PL points at a nonexistent path: {cand}")
    root = p.parent if p.is_file() else p
    bng2_pl = str(p if p.is_file() else root / "BNG2.pl")
    run_network = str(root / "bin" / "run_network")
    if not Path(bng2_pl).exists():
        sys.exit(f"ABORT: BNG2.pl not found at {bng2_pl}")
    if not Path(run_network).exists():
        sys.exit(f"ABORT: run_network not found at {run_network} (expected under <BNGPATH>/bin/)")
    return bng2_pl, run_network


# --------------------------------------------------------------------------- #
# Known comparison artifacts (folded from the retired correctness suite, GH #69)
# --------------------------------------------------------------------------- #
# A DIFF on one of these models is a comparison artifact — an integrator
# phase/timing effect or a reference-side tolerance tail — NOT a bngsim defect.
# Each entry carries a ``max_abs_bound``: the model is excused only while its
# worst genuine divergence stays within the recorded bound, so a future
# regression (which would blow past it) still flags a scoring DIFF. These were
# adjudicated under the legacy correctness suite (``parity_diff.py``); they now
# live here so the matrix is the single adjudication home (GH #69). Keyed by the
# BNGL stem; ``issue``/``reason`` annotate the row comment.
KNOWN_DETERMINISTIC_ARTIFACTS = {
    "proliferation": {
        "max_abs_bound": 1.0,
        "issue": "stiff-oscillator phase wander (verified 2026-05-17)",
        "reason": "stiff relaxation-oscillator phase wander across sharp tanh() "
        "switches (<=4e-3 time units over 9 cycles); period, amplitude and cycle "
        "count match both sides. Observed max_abs 0.43.",
    },
    "ATG_model_v16": {
        "max_abs_bound": 100.0,
        "issue": "staircase knife-edge (verified 2026-05-17)",
        "reason": "discontinuous staircase function (APdat_*) sampled exactly on "
        "its t=230 breakpoint; a ~1-ULP output-time difference flips one bucket "
        "on a single row. Species and all non-staircase columns agree to 12 sig "
        "figs; the .net files are byte-identical. Observed max_abs 24.55.",
    },
    "predator-prey-dynamics": {
        "max_abs_bound": 5.0,
        "issue": "Lotka-Volterra conserved-quantity drift (verified 2026-05-24)",
        "reason": "neutrally-stable Lotka-Volterra oscillator over ~112 cycles; "
        "the two integrators' conserved quantity drifts into a tiny phase/amplitude "
        "offset (max_abs 0.395 on a peak-3503 oscillation, rel 2.7e-4). Both stacks "
        "bounded and positive, agree to ~4 sig figs.",
    },
    "transport_v2": {
        "max_abs_bound": 5e-10,
        "issue": "reference-side tolerance-tail (verified 2026-06-08)",
        "reason": "bngsim-accurate tolerance-tail artifact: blood<->lymph transport "
        "relaxes to a near-zero equilibrium (analytical SS AB=AL=AB(0)*VB/(VB+VL)="
        "3.3333e-7); the two default-tol CVODE stacks diverge <=1e-10 abs / 3e-4 rel "
        "on the decaying tail. bngsim converges to the exact analytical SS at tight "
        "tol (Jacobian-independent); BNG run_network default-tol is the outlier.",
    },
}


def annotate_known_artifact(res: dict, status: str, stem: str, max_abs: float) -> dict:
    """Tag a known deterministic comparison artifact in place; return ``res``.

    A DIFF on a catalogued model whose worst genuine divergence stays within the
    recorded ``max_abs_bound`` is a comparison artifact (integrator phase-wander,
    staircase knife-edge, reference-side tolerance tail), not a bngsim defect: we
    keep the honest ``status="diff"`` but set ``res["subclass"]="known_artifact"``
    and prepend the reason so the matrix renders it KNOWN ARTIFACT (non-scoring).
    If the divergence EXCEEDS the bound (or is non-finite) the model is NOT excused
    — a real regression would blow past it — and stays a scoring DIFF. A no-op for
    any non-DIFF status or uncatalogued model.
    """
    if status != "diff":
        return res
    art = KNOWN_DETERMINISTIC_ARTIFACTS.get(stem)
    if art is None:
        return res
    if not (max_abs is not None and np.isfinite(max_abs) and max_abs <= art["max_abs_bound"]):
        return res
    res["subclass"] = "known_artifact"
    res["comment"] = (
        f"Known comparison artifact, not a bngsim error: {art['reason']} Within its checked "
        f"bound → non-scoring. (Raw: {res.get('comment', '')})"
    )
    return res


# --------------------------------------------------------------------------- #
# Comparison (shared _core.differ protocol over the network species)
# --------------------------------------------------------------------------- #
def _ode_comment(v: dict, n_sp: int, *, multiseg: bool) -> str:
    """One-line verdict for a deterministic (ODE) comparison, written for a modeller.

    Uses ordinary modelling terms (species, tolerance, trajectory) but no harness-internal
    vocabulary (the hard/soft/forgiven cell tallies).
    """
    scope = "main phase (multi-phase protocol)" if multiseg else "full trajectory"
    if v["passed"]:
        extra = ""
        if v["budget_forgiven"]:
            extra = (
                f" ({v['budget_forgiven']} points just outside strict tolerance, within "
                f"solver-to-solver slack)"
            )
        return f"Matches run_network within tolerance across all {n_sp} species, {scope}{extra}."
    n_bad = v["n_hard_fail"] or v["n_fail"]
    return (
        f"Differs from run_network beyond tolerance at {n_bad}/{v['n_cells']} points "
        f"({n_sp} species), {scope}."
    )


def _compare_ode(bn, rn) -> tuple[str, float, str, str, float, float]:
    """(status, value, comment, metric, tol) for one deterministic job.

    ``bn``/``rn`` are (time, values, names). Both come from the one ``.net`` so the
    species columns are in identical network order and the grids coincide; we align
    by index (:func:`_bng_common.align_net_species`) and apply
    ``deterministic_verdict``. metric = max_rel_err (post-budget worst), gated at
    REL_TOL.
    """
    bn_t, bn_v, _bn_n = bn
    rn_t, rn_v, _rn_n = rn
    metric, tol = "max_rel_err", differ.REL_TOL
    aligned = bc.align_net_species(bn_v, rn_v, bn_t, rn_t)
    if aligned is None:
        return "diff", float("inf"), "Disjoint species/time-points — nothing to compare.", metric, tol, float("inf")
    bn_a, rn_a, n_sp = aligned
    v = differ.deterministic_verdict(bn_a, rn_a)
    status = "pass" if v["passed"] else "diff"
    comment = _ode_comment(v, n_sp, multiseg=False)
    return status, v["max_rel"], comment, metric, tol, v["max_abs"]


def _compare_ode_multiseg(bn, rn) -> tuple[str, float, str, str, float, float]:
    """(status, value, comment, metric, tol) for a dirty_carryover ODE job (GH #179).

    ``bn`` is the bngsim representative segment (full-protocol replay); ``rn`` is the
    native BNG2.pl protocol output, which for a ``continue`` model is the FULL
    concatenated trajectory (segments append to one .cdat) — so the representative rows
    are aligned onto the bngsim grid by NEAREST TIME (the GH #180 methodology), not by
    index. Species columns are in identical deterministic-netgen order, compared via
    ``deterministic_verdict``; the shared continue-boundary row (t_start, where the
    native value is the prior segment's end and bngsim's is this segment's carried start)
    is excluded as a known output offset.
    """
    bn_t, bn_v, _bn_n = bn
    rn_t, rn_v, _rn_n = rn
    metric, tol = "max_rel_err", differ.REL_TOL
    if bn_v.size == 0 or rn_v.size == 0:
        return "diff", float("inf"), "Disjoint species/time-points — nothing to compare.", metric, tol, float("inf")
    idx = [int(np.argmin(np.abs(rn_t - t))) for t in bn_t]
    rn_on_grid = rn_v[idx]
    n_sp = min(bn_v.shape[1], rn_on_grid.shape[1])
    bn_a, rn_a = bn_v[1:, :n_sp], rn_on_grid[1:, :n_sp]  # drop the continue-boundary row
    if bn_a.shape[0] == 0:
        bn_a, rn_a = bn_v[:, :n_sp], rn_on_grid[:, :n_sp]
    v = differ.deterministic_verdict(bn_a, rn_a)
    status = "pass" if v["passed"] else "diff"
    comment = _ode_comment(v, n_sp, multiseg=True)
    return status, v["max_rel"], comment, metric, tol, v["max_abs"]


# --------------------------------------------------------------------------- #
# Worker (module-level so it is picklable under the 'spawn' start method)
# --------------------------------------------------------------------------- #
def _worker(spec: dict, q) -> None:
    """Run the netgen → BNGsim → run_network pipeline for one job; result on ``q``.

    Status is derived from per-engine outcome (the reference is the existence
    proof). Netgen failure is BAD_TEST (no input for either engine). A model with
    no resolvable ODE horizon is SKIP (not a parity signal).
    """
    warmup = bc.measure_warmup(spec.get("run_network_bin"))

    res = {k: spec[k] for k in ("key", "model_id", "method")}
    res.update(
        {
            "metric": spec["metric"],
            "tol": spec["tol"],
            "value": None,
            "comment": "",
            "exception": "",
        }
    )

    bngl_path = Path(spec["bngl"])
    try:
        bngl_text = bngl_path.read_text(errors="replace")
    except Exception as exc:
        res["status"] = "bad_test"
        res["comment"] = "BNGL file could not be read."
        res["exception"] = f"unreadable BNGL: {type(exc).__name__}: {exc}"[:300]
        q.put(res)
        return

    # An injected-action fixture (rehab/action_inject) REPLACES a dud model's
    # actions, so it — not the raw BNGL — is the source of both the netgen caps and
    # the simulate horizon. Non-injected jobs keep the bare netgen + raw-BNGL horizon
    # (so their .net and result are unchanged).
    inject = spec.get("inject")
    horizon_text = inject or bngl_text
    gen_network = bc.injected_gen_network(inject)
    # GH #177/#179: the pre-representative setParameter/setConcentration state. A
    # dirty_carryover model (representative inherits a prior simulate's end state) is
    # driven through the FULL protocol on BOTH engines below; a single-phase model bakes
    # this prefix into the .net (option-1) and runs the representative once. Sourced from
    # horizon_text so an injected fixture's actions (not the dud raw BNGL) win.
    state_prefix, prefix_info = bc.state_setup_prefix(horizon_text, track="ode")
    dirty = prefix_info["dirty_carryover"]

    workdir = Path(tempfile.mkdtemp(prefix="bng_ode_"))
    try:
        # 1. Network generation. A dirty_carryover model replays the protocol from the
        #    DECLARED IC (multi_segment_replay), so it needs a CLEAN .net; a single-phase
        #    model bakes the state prefix into the .net both engines then run.
        net_path, netgen_sec, netgen_err = bc.generate_network(
            bngl_text,
            spec["bng2_pl"],
            workdir,
            timeout=spec["sub_timeout"],
            gen_network=gen_network,
            state_prefix=("" if dirty else state_prefix),
        )
        if net_path is None:
            res["status"] = "bad_test"
            res["comment"] = (
                "BNG2.pl could not generate this model's network (failed or timed out) — a "
                "model problem, not bngsim."
            )
            res["exception"] = netgen_err
            res["timing"] = {"netgen": {"netgen_sec": round(netgen_sec, 6)}, "warmup": warmup}
            q.put(res)
            return

        # 2. Continuation-aware representative spec (GH #179: fixes the t_start=0 secondary
        #    bug that misreads a continue=>1 segment's span). Injected fixtures / bare-action
        #    text have no begin-actions region parse_protocol can walk, so fall back to the
        #    raw-text scanner (always single-segment). Unresolvable horizon -> SKIP.
        net_params = bc.read_net_parameters(net_path)
        ode = bc.representative_spec(
            horizon_text, track="ode", net_params=net_params, atol=spec["atol"], rtol=spec["rtol"]
        )
        if ode is None:
            o = bc.parse_ode_spec(horizon_text, net_params, atol=spec["atol"], rtol=spec["rtol"])
            if o is None:
                res["status"] = "skip"
                res["comment"] = "No runnable time-course (no simulate with a numeric t_end) — nothing to compare."
                res["timing"] = {"netgen": {"netgen_sec": round(netgen_sec, 6)}, "warmup": warmup}
                q.put(res)
                return
            ode = {**o, "dirty_carryover": False, "steps": None, "rep_index": None}
        dirty = bool(ode.get("dirty_carryover")) and ode.get("steps") is not None
        n_points = int(ode["n_steps"]) + 1

        bn = None
        bn_exc = ""
        bn_timing = None
        bn_wall = 0.0
        rn = None
        rn_exc = ""
        rn_timing = None
        rn_wall = 0.0
        if dirty:
            # 3a. bngsim: drive the FULL protocol in-process, capture the representative
            #     segment's species (the GH #179 carry-over the single-segment path misses).
            try:
                t0 = time.perf_counter()
                result, _info = bc.multi_segment_replay(
                    net_path, ode["steps"], ode["rep_index"],
                    track="ode", atol=ode["atol"], rtol=ode["rtol"], seed=None, poplevel=0.0,
                )
                bn_wall = time.perf_counter() - t0
                bn = (np.asarray(result.time), np.asarray(result.species), list(result.species_names))
            except Exception as exc:
                bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]
            # 3b. legacy: run the model's OWN protocol natively (BNG2.pl carries state across
            #     continue/save/reset), read the representative segment's species.
            try:
                t0 = time.perf_counter()
                rn = bc.native_protocol_oracle(
                    horizon_text, spec["bng2_pl"], workdir / "native",
                    track="ode", rep_stmt_idx=ode["rep_stmt_idx"], timeout=spec["sub_timeout"],
                )
                rn_wall = time.perf_counter() - t0
            except Exception as exc:
                rn_exc = f"native BNG2.pl protocol: {type(exc).__name__}: {exc}"[:400]
        else:
            # 3a. BNGsim single-segment (record status; do not short-circuit a generic raise).
            try:
                t0 = time.perf_counter()
                t, v, n, bn_timing = bc.bn_ode_net(
                    net_path, ode["t_start"], ode["t_end"], n_points, ode["rtol"], ode["atol"]
                )
                bn_wall = time.perf_counter() - t0
                bn = (t, v, n)
            except Exception as exc:
                bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]

            # 3b. run_network single-segment (the reference; always run).
            try:
                t0 = time.perf_counter()
                t, v, n, rn_timing = bc.run_network_ode(
                    net_path,
                    spec["run_network_bin"],
                    t_start=ode["t_start"],
                    t_end=ode["t_end"],
                    n_steps=int(ode["n_steps"]),
                    rtol=ode["rtol"],
                    atol=ode["atol"],
                    out_prefix=str(workdir / "rn"),
                    timeout=spec["sub_timeout"],
                )
                rn_wall = time.perf_counter() - t0
                rn = (t, v, n)
            except Exception as exc:
                rn_exc = f"run_network: {type(exc).__name__}: {exc}"[:400]

        res["wall_sec"] = round(netgen_sec + bn_wall + rn_wall, 3)
        if rn is not None:
            res["rn_finite"] = bool(np.isfinite(rn[1]).all())

        timing: dict = {"netgen": {"netgen_sec": round(netgen_sec, 6)}}
        timing["spec"] = {
            "t_start": ode["t_start"],
            "t_end": ode["t_end"],
            "n_steps": int(ode["n_steps"]),
            "rtol": ode["rtol"],
            "atol": ode["atol"],
            "n_species": int(bn[1].shape[1]) if bn else (int(rn[1].shape[1]) if rn else 0),
            # GH #179: dirty_carryover -> full-protocol replay (bngsim) vs native BNG2.pl
            # protocol (legacy); the single-segment per-engine timing split does not apply.
            "mode": "multi_segment" if dirty else "single_segment",
        }
        if bn_timing:
            timing["bngsim"] = bn_timing
        if rn_timing:
            timing["run_network"] = rn_timing
        timing["warmup"] = warmup
        res["timing"] = timing

        # 4. Classify from per-engine status. The plain-English ``comment`` is the
        #    headline a reader sees; the raw engine error stays in ``exception`` as a
        #    secondary detail the matrix renders muted underneath.
        if bn_exc or rn_exc:
            if bn_exc and rn_exc:
                res["status"], res["exception"] = "bad_test", f"{rn_exc} || {bn_exc}"
                res["comment"] = "Neither engine could run this model — a model/setup problem, not bngsim."
            elif rn_exc:
                res["status"], res["exception"] = "reference_failed", rn_exc
                res["comment"] = "run_network failed here — no reference to compare against. bngsim ran fine; unscored."
            else:
                res["status"], res["exception"] = "exception", bn_exc
                res["comment"] = "bngsim errored while run_network succeeded — an actionable bngsim issue."
            q.put(res)
            return

        try:
            comparer = _compare_ode_multiseg if dirty else _compare_ode
            status, value, comment, metric, m_tol, max_abs = comparer(bn, rn)
            res["status"], res["value"], res["comment"] = status, value, comment
            res["metric"], res["tol"] = metric, m_tol
            # GH #69: a DIFF on a catalogued comparison artifact (integrator
            # phase-wander / staircase knife-edge / reference-side tolerance tail)
            # within its recorded magnitude bound is tagged so the matrix renders it
            # KNOWN ARTIFACT (non-scoring); the honest verdict is unchanged.
            annotate_known_artifact(res, status, bngl_path.stem, max_abs)
        except Exception as exc:
            res["status"] = "exception"
            res["exception"] = f"compare: {type(exc).__name__}: {exc}"[:400]
        q.put(res)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# Worker status -> _core Outcome (timeout/dead synthesized by the scheduler).
_OUTCOME = {
    "pass": Outcome.PASS,
    "diff": Outcome.DIFF,
    "exception": Outcome.EXCEPTION,
    "reference_failed": Outcome.REFERENCE_FAILED,
    "bad_test": Outcome.BAD_TEST,
    "skip": Outcome.SKIP,
    "timeout": Outcome.TIMEOUT,
    "dead": Outcome.EXCEPTION,
}


def _job_tol_override(job) -> dict | None:
    """The per-job ``tol`` override (ill-conditioned IVP), applied to both engines."""
    for ov in job.overrides:
        if ov.field == "tol" and isinstance(ov.value, dict):
            try:
                return {"rtol": float(ov.value["rtol"]), "atol": float(ov.value["atol"])}
            except (KeyError, ValueError, TypeError):
                return None
    return None


def _filter_jobs(jobs, args):
    out = [j for j in jobs if j.method == "ode"]
    if args.models:
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        out = [j for j in out if j.model_id in wanted]
    if args.include:
        out = [j for j in out if args.include in j.model]
    if args.exclude:
        out = [j for j in out if args.exclude not in j.model]
    if args.limit:
        out = out[: args.limit]
    return out


def _make_progress():
    def _cb(done: int, total: int, res: dict) -> None:
        if done % 25 == 0 or done == total:
            st = res.get("status", "?")
            print(f"  [{done}/{total}] {st:16s} {res.get('model_id', '')}", flush=True)

    return _cb


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", default=str(JOBS), help="_core manifest (default jobs.json)")
    ap.add_argument("--out", default="", help="report path (default runs/report_ode.json)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent job subprocesses.")
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Per-job wall cap (s); defaults to {DEFAULT_ODE_TIMEOUT}s.",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=bc.DEFAULT_RTOL,
        help=f"ODE rtol forced on BOTH engines (default {bc.DEFAULT_RTOL:g}; a per-job "
        "tol override wins, applied identically to both).",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=bc.DEFAULT_ATOL,
        help=f"ODE atol forced on BOTH engines (default {bc.DEFAULT_ATOL:g}).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max ODE jobs after filtering (0=all).")
    ap.add_argument("--models", default="", help="Comma-separated model_id filter.")
    ap.add_argument("--include", default="", help="Substring filter on the model path.")
    ap.add_argument("--exclude", default="", help="Substring filter — drop matching model paths.")
    args = ap.parse_args()

    bng2_pl, run_network_bin = _resolve_bng_tools()

    jobs_path = Path(args.jobs).resolve()
    if not jobs_path.exists():
        sys.exit(f"missing {jobs_path}")
    models_root = (jobs_path.parent / "models").resolve()
    _meta, alljobs = read_manifest(jobs_path)
    jobs = _filter_jobs(alljobs, args)
    if not jobs:
        sys.exit("no ODE jobs after filtering.")

    missing = [j.model_id for j in jobs if not (models_root / j.model).exists()]
    if missing:
        sys.exit(
            f"{len(missing)} job(s) have no vendored BNGL (e.g. {missing[:3]}). "
            "Run `python vendor_corpus.py` to materialize the model tree."
        )

    # Per-job wall cap; the inner BNG2.pl/run_network subprocesses get a slightly
    # smaller cap so they self-terminate before the scheduler kills the worker
    # (avoids orphaned grandchildren).
    cap = args.timeout if args.timeout is not None else DEFAULT_ODE_TIMEOUT
    sub_timeout = max(10.0, cap - 20.0)

    n_tol_ov = 0
    specs = []
    for j in jobs:
        tol_ov = _job_tol_override(j)
        if tol_ov:
            rtol, atol = tol_ov["rtol"], tol_ov["atol"]
            n_tol_ov += 1
        else:
            rtol, atol = args.rtol, args.atol
        specs.append(
            {
                "key": f"{j.model_id}:{j.method}",
                "model_id": j.model_id,
                "method": j.method,
                "metric": j.oracle.metric,
                "tol": j.oracle.tol,
                "bngl": str(models_root / j.model),
                "rtol": rtol,
                "atol": atol,
                "cap": float(cap),
                "sub_timeout": float(sub_timeout),
                "bng2_pl": bng2_pl,
                "run_network_bin": run_network_bin,
                "inject": bc.injected_action_block(j.overrides),
            }
        )

    ver = versions.stamp("bng")
    ver["sundials"] = bc.sundials_version()
    print("=" * 72)
    print("  bngsim vs BNG2.pl/run_network — BNGL ODE parity + timing (bng_parity)")
    print("=" * 72)
    print(f"  jobs: {len(specs)}   workers: {args.workers}   (ODE/deterministic only)")
    print(f"  bngsim {ver['bngsim']}   BNG {ver.get('bng')}   run_network: {run_network_bin}")
    print(
        f"  ODE tol (both engines): rtol={args.rtol:g} atol={args.atol:g}   protocol: _core.differ"
    )
    print()

    t0 = time.perf_counter()
    raw = bc.schedule(
        specs,
        _worker,
        workers=args.workers,
        timeout_of=lambda s: s["cap"],
        on_done=_make_progress(),
    )
    elapsed = time.perf_counter() - t0

    now = _dt.datetime.now().isoformat(timespec="seconds")
    results = []
    for r in raw:
        outcome = _OUTCOME.get(r.get("status"), Outcome.EXCEPTION)
        comment = r.get("comment", "")
        if r.get("status") == "timeout":
            comment = f"Exceeded the {r.get('cap')}s wall limit; stopped."
        elif r.get("status") == "dead":
            comment = "Worker process crashed before completion."
        wall = r.get("wall_sec") or 0.0
        results.append(
            JobResult(
                model_id=r["model_id"],
                method=r.get("method", "ode"),
                reference_engine="bng",
                outcome=str(outcome),
                metric=r.get("metric"),
                value=r.get("value"),
                tol=r.get("tol"),
                exception=r.get("exception", ""),
                wall_sec=round(wall, 3) if wall else None,
                timestamp=now,
                versions=ver,
                comment=comment,
                subclass=r.get("subclass"),
                timing=r.get("timing"),
            )
        )

    results.sort(key=lambda x: x.model_id)
    counts = tally(r.outcome for r in results)
    out_path = Path(args.out).resolve() if args.out else HERE / "runs" / "report_ode.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite": "bng_parity",
        "reference_engine": "bng",
        "reference_label": "run_network",
        "regime": "ode",
        "git_rev": git_rev(str(HERE)),
        "bngpath": os.environ.get("BNGPATH"),
        "run_network": run_network_bin,
        "versions": ver,
        "tally": counts,
        "n_jobs": len(results),
        "elapsed_sec": round(elapsed, 2),
        "hardware": bc.hardware_info(),
        "concurrency": {"workers": args.workers, "mode": "process-parallel"},
        "config": {
            "bngsim_method": "ode",
            "rtol": args.rtol,
            "atol": args.atol,
            "tol_overridden_jobs": n_tol_ov,
        },
        "integration_tol": {"rtol": args.rtol, "atol": args.atol, "applied_to": "both engines"},
        "oracle_basis": (
            "Cross-engine NUMERIC tolerance (never byte-identical), via the shared "
            "_core.differ protocol — the same verdict the bng_parity correctness "
            "suite uses. BNG2.pl generates the reaction network .net ONCE per model; "
            "BNGsim integrates it in-process and the legacy run_network CVODE binary "
            "integrates the SAME .net as a subprocess, so the comparison is "
            "apples-to-apples on identical input. Both engines are forced to a shared "
            f"tolerance (rtol={args.rtol:g}, atol={args.atol:g}); a per-job tol "
            "override (ill-conditioned IVP) wins, applied identically to both. ODE: "
            "deterministic_verdict over the network species (the .cdat columns, in "
            "identical network order) — combined abs+rel per-cell tol with a "
            f"fail-fraction budget (<={differ.FAIL_FRAC_BUDGET}) gated by hard "
            f"ceilings (rel<={differ.HARD_REL_CEILING}); metric=max_rel_err "
            f"(tol={differ.REL_TOL}). Failure is attributed per-engine: bngsim "
            "raised + run_network ran -> EXCEPTION (actionable bngsim bug); bngsim "
            "ran + run_network raised -> REFERENCE_FAILED (non-scoring); netgen "
            "failed or both raised -> BAD_TEST; no resolvable ODE horizon -> SKIP; "
            "wall-cap overrun -> TIMEOUT. Only the engine-agnostic tol override is "
            "applied here (KNOWN_ARTIFACT/PASS_REF_BUG live in the correctness "
            "suite); this timing track adjudicates every model independently."
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
    print(f"  report: {out_path}")
    print("=" * 72)
    from _core.taxonomy import FAILING

    n_fail = sum(counts.get(o.value, 0) for o in FAILING)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
