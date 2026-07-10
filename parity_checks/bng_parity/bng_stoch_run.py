#!/usr/bin/env python3
"""bngsim vs the legacy BNG stack — stochastic (SSA / NF) timing + parity (bng_parity).

The stochastic sibling of ``bng_ode_run.py``: same ``jobs.json`` corpus, same
shared ``_core`` machinery and process-isolation scheduler, but the regime is an
N-replicate ensemble compared on the OBSERVABLES (by name) rather than a single
deterministic trajectory. One runner, two tracks (``--track ssa|nf``):

  * **SSA** — bngsim ``method="ssa"`` vs the legacy ``run_network -p ssa`` binary,
    both on the SAME BNG2.pl ``.net`` (``generate_network`` is the shared build
    prefix, attributed to neither engine).
  * **NF** (network-free) — bngsim's ``NfsimSession`` vs the legacy ``NFsim``
    application, both on the SAME BNG-XML (``writeXML`` is the shared build prefix;
    a network-free model has no ``.net``).

Each track runs every model through a three-step pipeline in its own spawned
worker: build the shared artifact, run both engines' N-replicate ensembles
(seeds ``seed_base, …, seed_base+n_rep-1``), then compare via the shared
``_core.differ.ensemble_verdict`` (a per-cell K-sigma frac-pass test on the
ensemble means). Failure is attributed per-engine, exactly as in the ODE track:

  * both ensembles agree (frac_pass >= PASS_FRAC) -> PASS
  * both ran, disagree                            -> DIFF
  * bngsim raised, legacy ran                      -> EXCEPTION (actionable bngsim bug)
  * bngsim ran, legacy raised                      -> REFERENCE_FAILED (no oracle)
  * bngsim cleanly refused (validate_for_ssa)      -> UNSUPPORTED
  * bngsim hit the per-replicate wall cap          -> TIMEOUT (a tight clock, not a bug)
  * build (netgen/writeXML) failed / both raised   -> BAD_TEST
  * no resolvable stochastic horizon               -> SKIP
  * scheduler per-job wall-cap overrun             -> TIMEOUT

NF parity gotcha: bngsim runs network-free at its CORRECT default
``block_same_complex_binding=True`` (the project's deliberate, correct default).
The legacy NFsim v1.14.3 has no ``-bscb`` flag and applies same-complex binding, so
on ring-forming / multivalent models the two diverge — a KNOWN, expected,
REFERENCE-side divergence (a legacy NFsim bug, not a bngsim defect). This track
adjudicates every model independently and surfaces them as honest DIFF (the
ensembles really do diverge), but a catalogued model (``NF_KNOWN_REF`` for NF,
``SSA_KNOWN_DISPOSITION`` for SSA) is tagged ``subclass="ref_bscb"`` /
``"known_artifact"`` so the matrix renders it KNOWN REF / KNOWN ARTIFACT —
non-scoring, not a bngsim bug — rather than a bare DIFF read as a bngsim defect.
These dispositions were folded here from the retired correctness suite (GH #69), so
the matrix is now their single home; each was confirmed against an independent
oracle (ODE / RuleMonkey / COPASI / analytic). See GH #183 / #118 / #69.

Usage
-----
    export BNGPATH=/path/to/BioNetGen-2.9.3   # BNG2.pl + bin/run_network + bin/NFsim
    python bng_stoch_run.py --track ssa --workers 4 --limit 20
    python bng_stoch_run.py --track nf  --workers 4 --limit 20
    python generate_bng_matrix.py runs/report_ssa.json
    open runs/bng_matrix_ssa.html
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
    oracles,
    read_manifest,
    tally,
    versions,
    write_report,
)
from _core.versions import git_rev  # noqa: E402

JOBS = HERE / "jobs.json"
# Per-job wall cap: build prefix + both engines' N-replicate ensembles. Stochastic
# ensembles (esp. NF / population-heavy SSA) are slower than a single ODE solve, so
# this is larger than the ODE track's cap.
DEFAULT_STOCH_TIMEOUT = 1800.0  # raised for 100-rep ensembles: the BNG2.pl reference
# (one run_network subprocess per seed) is the bottleneck on heavy rule-networks
# (~5s/rep => ~500s for 100 reps), so the per-job wall must clear that (GH #190).
DEFAULT_N_REP = 10  # manifest base ensemble; DIFFs can escalate to 30/100 below.
# The agreement gate auto-scales by sqrt(N/10) (differ.ensemble_k_for), so an explicit
# larger --n-rep sharpens precision without tightening the meaningful-difference
# threshold. Keep the default aligned with build_jobs.py/jobs.json.
DEFAULT_SEED_BASE = 1

# Seed-escalation oracle (GH #69, direction-of-change discriminator GH #185). A
# DIFF at the base replicate count can be finite-N sampling noise: at N=10 the
# per-cell std estimate that sets the K-sigma bar is itself unreliable, so a
# correct model borderline-fails the 0.99 frac_pass threshold. We ACCUMULATE
# replicates up a geometric ladder (base -> 30 -> 100, reusing the seeds already
# run — N=30 is a strict superset of N=10) and read the DIRECTION of change rather
# than gate on a fixed frac_pass floor. A fixed floor cannot separate noise from a
# real divergence because the two OVERLAP in frac_pass space (noise can sit at 0.75
# while a genuine -bscb divergence sits at 0.66 — GH #185), so no single threshold
# is correct: a 0.95 floor left converging-noise models (m02_logistic 0.75,
# S1987_B1_lotka_volterra 0.94 at N=10, both -> 1.0 at N=100) as false DIFFs.
# Instead escalate ANY DIFF one rung; if agreement IMPROVES it is finite-N noise
# (keep climbing to convergence), if it STALLS or worsens it is a real
# disagreement (stop early — more replicates only reconfirm it). This is both
# cheaper (clearly-real diffs early-exit instead of grinding to N=100) and strictly
# more correct (it catches noise regardless of its starting frac_pass). Geometric
# (~3x) rungs because precision scales as sqrt(N): additive rungs (50, 70) buy
# diminishing tightening.
ESCALATION_RUNGS = (30, 100)  # cumulative replicate counts above the base
ESCALATE_IMPROVE_EPS = 1e-9  # frac_pass must RISE by more than this for a rung to
#   count as "improving" (noise -> keep escalating). frac_pass moves in steps of
#   1/n_cells, far larger than this, so any real gain of even one cell continues;
#   an exactly-flat or falling frac_pass is a stall and stops escalation.

# Per-track reference engine: which legacy binary, where it lives under <BNGPATH>,
# and the regime/labels stamped into the report.
TRACKS = {
    "ssa": {"legacy_label": "run_network", "legacy_rel": ("bin", "run_network"), "regime": "ssa"},
    "nf": {"legacy_label": "NFsim", "legacy_rel": ("bin", "NFsim"), "regime": "nf"},
}


# --------------------------------------------------------------------------- #
# Tool resolution
# --------------------------------------------------------------------------- #
def _resolve_bng_tools(track: str) -> tuple[str, str]:
    """(BNG2.pl, legacy_bin) from $BNGPATH/$BNG2_PL; never hardcoded.

    ``$BNGPATH`` may be the BioNetGen folder (containing BNG2.pl + bin/) or a direct
    BNG2.pl path. The legacy binary is ``run_network`` (SSA) or ``NFsim`` (NF),
    under ``<root>/bin/``.
    """
    cand = os.environ.get("BNGPATH") or os.environ.get("BNG2_PL")
    if not cand:
        sys.exit(
            "ABORT: set $BNGPATH (the BioNetGen folder with BNG2.pl + bin/) or "
            "$BNG2_PL (a direct BNG2.pl path) so the reference stack is resolvable. "
            "No path is hardcoded."
        )
    p = Path(cand)
    if not p.exists():
        sys.exit(f"ABORT: $BNGPATH/$BNG2_PL points at a nonexistent path: {cand}")
    root = p.parent if p.is_file() else p
    bng2_pl = str(p if p.is_file() else root / "BNG2.pl")
    legacy = str(root.joinpath(*TRACKS[track]["legacy_rel"]))
    if not Path(bng2_pl).exists():
        sys.exit(f"ABORT: BNG2.pl not found at {bng2_pl}")
    if not Path(legacy).exists():
        sys.exit(
            f"ABORT: {TRACKS[track]['legacy_label']} not found at {legacy} "
            "(expected under <BNGPATH>/bin/)"
        )
    return bng2_pl, legacy


# --------------------------------------------------------------------------- #
# Known reference-side NF divergences (legacy NFsim v1.14.3 is the buggy oracle)
# --------------------------------------------------------------------------- #
# A network-free (NF) DIFF on one of these models is a known LEGACY-NFsim bug, not a
# bngsim defect. Most are the ``block_same_complex_binding`` (-bscb) case: bngsim runs
# at its CORRECT default ``bscb=True``, while the legacy NFsim v1.14.3 has no -bscb flag
# and wrongly re-binds inside ring/multivalent complexes; a few are other legacy-NFsim
# feature bugs (it ignores ``$``-clamped species; it mishandles a cooperative rate).
# Every entry was confirmed against an INDEPENDENT oracle — the ODE network result,
# bngsim's own RuleMonkey engine (an independent network-free implementation), or the
# model's documented analytic answer — which is what attributes the divergence to the
# reference, not to bngsim. This track keeps the honest DIFF but tags the row
# ``subclass="ref_bscb"`` so the matrix renders it KNOWN REF (non-scoring). Folded here
# from the retired ``parity_diff.py`` correctness suite (GH #69), restricted to the
# models that still DIFF in the current full sweep. Keyed by the model-id stem.
NF_KNOWN_REF = {
    "EGFR_oligo_v2": {
        "issue": "internal#183",
        "reason": "EGFR receptor oligomerization (multivalent ring-former). bngsim (bscb "
        "on) blocks same-complex re-binding; legacy NFsim (no -bscb) does not, so the "
        "oligomer-ladder observables diverge. The reference is the outlier.",
    },
    "ode_vs_nf_discrepancy": {
        "issue": "internal#54",
        "reason": "Multivalent ring complex. Legacy NFsim (no -bscb) wrongly breaks a bond "
        "inside a 5-molecule ring. bngsim-NF, bngsim-RuleMonkey, and the ODE network all "
        "agree; legacy is the outlier (confirmed against the ODE oracle).",
    },
    "debug": {
        "issue": "internal#54",
        "reason": "Same -bscb root cause (MTOR/RPTOR pre-assembled complex). bngsim-NF "
        "tracks the ODE network result; legacy NFsim is the outlier.",
    },
    "debug_v3": {
        "issue": "internal#54",
        "reason": "Same -bscb root cause (simplified MTOR/RPTOR complex). bngsim-NF tracks "
        "the ODE network result; legacy NFsim is the outlier.",
    },
    "overlap_rules2": {
        "issue": "internal#54/#55",
        "reason": "Same -bscb root cause (size-2 ring). The ODE keeps the rings "
        "(ring-opening would violate molecularity); bngsim-NF agrees, legacy NFsim wrongly "
        "breaks them.",
    },
    "testrings_wsh": {
        "issue": "internal#54",
        "reason": "Same -bscb root cause (ring complex). bngsim-NF tracks the ODE network "
        "result; legacy NFsim is the outlier.",
    },
    "ft_clamped_species_strict": {
        "issue": "legacy NFsim ignores $-clamped species",
        "reason": "Fixed-species ($A) feature test. Legacy NFsim v1.14.3 silently ignores "
        "the $ clamp; bngsim enforces it and matches the model's documented analytic answer "
        "(A_tot=100, B_tot≈500). bngsim is correct; legacy is the wrong oracle.",
    },
    "bench_blbr_rings_posner1995": {
        "issue": "legacy NFsim ring-closure (no -bscb)",
        "reason": "BLBR cyclic-aggregate model. Legacy NFsim disagrees catastrophically on "
        "the bond/ring observables (~37% agreement); bngsim-NF and bngsim-RuleMonkey (an "
        "independent network-free engine) agree to 99.9%, so legacy is the buggy reference.",
    },
    "ft_cooperative_binding": {
        "issue": "legacy NFsim cooperative-binding bug",
        "reason": "Cooperative-binding feature test. Legacy NFsim mishandles the cooperative "
        "rate (~14% agreement); bngsim-NF and bngsim-RuleMonkey agree (~99%). Legacy is the "
        "outlier.",
    },
    "Tutorial_Example": {
        "issue": "legacy NFsim NF-segment bug",
        "reason": "Legacy NFsim NF-segment divergence (~72% agreement); bngsim-NF and "
        "bngsim-RuleMonkey agree exactly (100%). Legacy is the outlier.",
    },
}


# SSA dispositions folded from parity_diff.py's KNOWN_STOCHASTIC_ARTIFACTS (GH #69),
# restricted to the models that still DIFF in the current full sweep. Two kinds:
#   ``ref_bscb``      -> KNOWN REF: an independent oracle shows bngsim is the accurate
#                        side and the legacy engine is the outlier (GH #118: a
#                        .net-faithful Gillespie sampler + COPASI's direct-method
#                        Gillespie both track bngsim).
#   ``known_artifact``-> KNOWN ARTIFACT: a benign sampling artifact both engines agree
#                        on (ensemble means match; the residual is finite-N phase scatter).
SSA_KNOWN_DISPOSITION = {
    "Kholodenko2000": {
        "subclass": "ref_bscb",
        "issue": "GH #118",
        "reason": "MAPK cascade with observable-dependent functional propensities. bngsim-SSA "
        "and legacy-SSA disagree on Obs_MAPK; two independent oracles — a .net-faithful "
        "Gillespie sampler and COPASI's direct-method Gillespie — both track bngsim, while "
        "legacy is the outlier. bngsim is the accurate side.",
    },
    "V2005_bistable_gene": {
        "subclass": "ref_bscb",
        "issue": "GH #118",
        "reason": "Bistable gene circuit (Hill induction + fractional-power kinetics). An "
        "independent .net-faithful Gillespie oracle finds bngsim at least as accurate as "
        "legacy (mean rel 0.10 vs 0.18); the residual is intrinsic bistable basin-occupancy "
        "variance, not a bngsim defect.",
    },
    "Krishna2006": {
        "subclass": "known_artifact",
        "issue": "benign finite-N phase scatter",
        "reason": "NF-κB spiky oscillator. bngsim-SSA and legacy-SSA ensemble means agree to "
        "~2%; the sub-threshold per-cell agreement is genuine phase scatter on a spiky "
        "oscillator — both engines agree, neither is wrong.",
    },
}


def annotate_ref_bscb(res: dict, status: str, track: str, stem: str) -> dict:
    """Tag a known reference-side NF DIFF in place and return ``res`` (GH #183/#69).

    A DIFF on a catalogued NF model (``NF_KNOWN_REF``) is a known legacy-NFsim bug, not a
    bngsim defect: we keep the honest ``status="diff"`` but set ``res["subclass"]="ref_bscb"``
    and rewrite the comment so the matrix renders it KNOWN REF (non-scoring). A no-op for any
    other (status, track, stem) — only an nf-track DIFF on a catalogued model is tagged.
    """
    if status != "diff" or track != "nf":
        return res
    art = NF_KNOWN_REF.get(stem)
    if art is None:
        return res
    res["subclass"] = "ref_bscb"
    res["comment"] = (
        f"Known legacy NFsim limitation, not a bngsim error: {art['reason']} bngsim is correct "
        f"here → non-scoring. (Raw: {res.get('comment', '')})"
    )
    return res


def annotate_ssa_known(res: dict, status: str, track: str, stem: str) -> dict:
    """Tag a known SSA disposition in place and return ``res`` (GH #69, from parity_diff).

    A no-op unless an ssa-track DIFF on a catalogued model. ``ref_bscb`` renders KNOWN REF
    (legacy is the outlier, bngsim confirmed by an independent oracle); ``known_artifact``
    renders KNOWN ARTIFACT (a benign sampling artifact both engines agree on). Both
    non-scoring.
    """
    if status != "diff" or track != "ssa":
        return res
    art = SSA_KNOWN_DISPOSITION.get(stem)
    if art is None:
        return res
    res["subclass"] = art["subclass"]
    if art["subclass"] == "ref_bscb":
        res["comment"] = (
            f"Known reference-engine divergence, not a bngsim error ({art['issue']}): "
            f"{art['reason']} (Raw: {res.get('comment', '')})"
        )
    else:
        res["comment"] = (
            f"Known benign stochastic artifact ({art['issue']}): {art['reason']} "
            f"(Raw: {res.get('comment', '')})"
        )
    return res


def reference_failure_comment(track: str, legacy_label: str, leg_exc: str) -> str:
    """Human attribution for known legacy-reference failure modes."""
    if track == "ssa" and "edgepop:" in leg_exc:
        return (
            "The reference engine failed while evaluating a molecule/edge-population "
            "observable (run_network 'edgepop' error) — no reference ensemble to compare. "
            "bngsim ran fine; unscored."
        )
    if track == "nf" and "NO_STATE" in leg_exc:
        return (
            "The reference NFsim engine failed while evaluating a ring/bond observable "
            "(NO_STATE in the legacy output) — no reference ensemble to compare. "
            "bngsim ran fine; unscored."
        )
    return (
        f"The reference engine ({legacy_label}) failed here — no reference ensemble to "
        "compare against. bngsim ran fine; unscored."
    )


# --------------------------------------------------------------------------- #
# Comparison (shared _core ensemble oracle over the observables, by name)
# --------------------------------------------------------------------------- #
def _compare_stoch(bn, leg) -> tuple[str, float, str, str, float]:
    """(status, value, comment, metric, tol) for one stochastic job.

    ``bn``/``leg`` are ``(time, obs[n_rep,n_time,n_obs], obs_names)``. We align the
    two ensembles on the shared observable names, reduce each to per-cell
    ``(mean, sem)`` (:func:`_core.oracles.ensemble_stats`), and apply
    ``ensemble_verdict`` (K-sigma frac-pass). metric = ``ensemble_frac_pass`` (higher
    is better; PASS iff >= ENSEMBLE_PASS_FRAC), the realized oracle the bng_parity
    correctness suite also records.
    """
    bn_t, bn_v, bn_n = bn
    leg_t, leg_v, leg_n = leg
    metric, tol = "ensemble_frac_pass", differ.ENSEMBLE_PASS_FRAC
    aligned = bc.align_observables_by_name(bn_n, bn_v, leg_n, leg_v)
    if aligned is None:
        return ("diff", 0.0, "Disjoint observables — nothing in common to compare.", metric, tol)
    bn_a, leg_a, common = aligned
    bn_mean, bn_sem = oracles.ensemble_stats(bn_a)
    leg_mean, leg_sem = oracles.ensemble_stats(leg_a)
    # Effect-size-preserving gate: scale the per-cell sigma threshold by sqrt(N/10)
    # so a larger ensemble tightens the statistical *precision* without mechanically
    # shrinking the *meaningful* difference the gate flags (GH #190; mirrors rr_parity).
    k = differ.ensemble_k_for(bn_a.shape[0])
    v = differ.ensemble_verdict(bn_mean, bn_sem, leg_mean, leg_sem, k=k)
    status = "pass" if v["passed"] else "diff"
    pct = round(v["frac_pass"] * 100, 1)
    comment = (
        f"Ensemble means ({bn_a.shape[0]} replicates/engine) agree within tolerance at {pct}% of "
        f"points across {len(common)} observables; worst point {v['max_z']:.0f}σ apart "
        f"(gate {k:.1f}σ, scaled √(N/10))."
    )
    return status, float(v["frac_pass"]), comment, metric, tol


def _run_escalation(
    bn,
    leg,
    status,
    value,
    comment,
    metric,
    m_tol,
    *,
    n_rep_base,
    seed_base,
    compare,
    escalate,
):
    """Direction-of-change seed-escalation (GH #69 oracle, GH #185 discriminator).

    Starting from the base verdict ``(status, value, comment, metric, m_tol)`` over
    ensembles ``bn``/``leg`` (each ``(time, obs[n_rep,n_time,n_obs], names)``),
    accumulate replicates up :data:`ESCALATION_RUNGS` while a DIFF keeps IMPROVING
    (finite-N noise converging) and stop as soon as it stalls or worsens (a real
    divergence). ``compare(bn, leg) -> (status, value, comment, metric, tol)``
    re-runs the verdict; ``escalate(delta, seed_base, rung) -> (add_bn, add_leg)``
    runs ``delta`` more replicates of each engine to pool on. The discriminator is
    pure w.r.t. those two callables, so it is unit-testable without an engine.

    Returns ``(status, value, comment, metric, m_tol, prog, escalation_stalled)``:
    ``prog`` is the ``[(n_rep, round(frac_pass, 4)), ...]`` rungs walked, and
    ``escalation_stalled`` is True iff escalation stopped on a non-improving DIFF
    (vs. clearing to PASS, running out of rungs, or a build/grid error). When more
    than the base rung was walked, the returned ``comment`` is prefixed with a
    plain-English lead describing the trajectory (noise-converged / real-divergence
    / narrowing-but-unresolved).
    """
    cur_n_rep = int(n_rep_base)
    prog = [(cur_n_rep, round(float(value), 4))]
    # GH #185 direction-of-change: escalate ANY DIFF (no fixed floor) and let the
    # trend decide. prev_value is the frac_pass at the start of the current rung;
    # escalation_stalled records that we stopped because agreement did not improve
    # (a real divergence) rather than because we ran out of rungs.
    prev_value = float(value)
    escalation_stalled = False
    for target in ESCALATION_RUNGS:
        if status != "diff" or target <= cur_n_rep:
            break
        try:
            add_bn, add_leg = escalate(target - cur_n_rep, seed_base + cur_n_rep, target)
        except Exception as exc:
            comment += f" | escalation to N={target} aborted: {type(exc).__name__}"
            break
        if add_bn[1].shape[1:] != bn[1].shape[1:] or add_leg[1].shape[1:] != leg[1].shape[1:]:
            comment += f" | escalation to N={target} skipped (grid mismatch)"
            break
        bn = (bn[0], np.concatenate([bn[1], add_bn[1]], axis=0), bn[2])
        leg = (leg[0], np.concatenate([leg[1], add_leg[1]], axis=0), leg[2])
        cur_n_rep = target
        status, value, comment, metric, m_tol = compare(bn, leg)
        prog.append((target, round(float(value), 4)))
        # Stop early on a real divergence: still a DIFF and agreement did not improve
        # over the previous rung. Noise instead climbs and falls through to the next
        # rung (or clears the bar to PASS).
        if status == "diff" and float(value) <= prev_value + ESCALATE_IMPROVE_EPS:
            escalation_stalled = True
            break
        prev_value = float(value)

    if len(prog) > 1:
        first_n, final_n = prog[0][0], prog[-1][0]
        if status == "pass":
            lead = (
                f"Borderline at {first_n} replicates (within finite-sample scatter); "
                f"agreement converged by {final_n} replicates/engine — sampling noise, not "
                f"a real divergence."
            )
        elif escalation_stalled:
            lead = (
                f"Disagreement does not close from {first_n} to {final_n} replicates/engine "
                f"(agreement stalled or worsened as replicates grew) — a real divergence, "
                f"not sampling noise."
            )
        else:
            lead = (
                f"Agreement narrowed from {first_n} to {final_n} replicates/engine but stayed "
                f"below tolerance at the replicate ceiling — still divergent, not positively "
                f"resolved as noise."
            )
        detail = ", ".join(f"{round(fp * 100)}% at {n}" for n, fp in prog)
        comment = f"{lead} (Agreement: {detail}.) " + comment

    return status, value, comment, metric, m_tol, prog, escalation_stalled


# --------------------------------------------------------------------------- #
# Worker (module-level so it is picklable under the 'spawn' start method)
# --------------------------------------------------------------------------- #
def _bn_stoch_ensemble(track, artifact, spec, n_rep, seed_base, rep_timeout):
    """Run the bngsim N-replicate ensemble for ``track`` over the built artifact.
    Returns ``(time, obs, names, timing)``. Raises on a genuine bngsim failure."""
    if track == "ssa":
        return bc.bn_ssa_net(
            artifact,
            spec["t_start"],
            spec["t_end"],
            int(spec["n_steps"]) + 1,
            n_rep,
            seed_base,
            rep_timeout=rep_timeout,
        )
    return bc.bn_nf_xml(
        artifact,
        spec["t_start"],
        spec["t_end"],
        int(spec["n_steps"]) + 1,
        n_rep,
        seed_base,
        block_same_complex_binding=spec["nf_bscb"],
        molecule_limit=spec.get("gml"),
        rep_timeout=rep_timeout,
    )


def _legacy_stoch_ensemble(
    track, artifact, spec, legacy_bin, n_rep, seed_base, out_prefix, timeout
):
    """Run the legacy N-replicate ensemble for ``track``. Returns
    ``(time, obs, names, timing)``. Raises RuntimeError on a legacy failure."""
    if track == "ssa":
        return bc.run_network_ssa(
            artifact,
            legacy_bin,
            t_start=spec["t_start"],
            t_end=spec["t_end"],
            n_steps=int(spec["n_steps"]),
            n_rep=n_rep,
            seed_base=seed_base,
            out_prefix=out_prefix,
            timeout=timeout,
        )
    return bc.nfsim_run(
        artifact,
        legacy_bin,
        t_end=spec["t_end"],
        n_steps=int(spec["n_steps"]),
        n_rep=n_rep,
        seed_base=seed_base,
        out_prefix=out_prefix,
        timeout=timeout,
        gml=spec.get("gml"),
        complex_bookkeeping=spec.get("nf_complex_bookkeeping", False),
    )


def _worker(spec: dict, q) -> None:
    """Run the build → bngsim ensemble → legacy ensemble pipeline for one job.

    Status is derived from per-engine outcome (the legacy stack is the existence
    proof). A build failure (netgen/writeXML) is BAD_TEST; no resolvable horizon is
    SKIP; a clean bngsim SSA refusal (validate_for_ssa) is UNSUPPORTED.
    """
    track = spec["track"]
    warmup = bc.measure_stoch_warmup(spec.get("legacy_bin"), legacy_label=spec["legacy_label"])

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

    def _with_settings(timing: dict) -> dict:
        settings = spec.get("settings")
        if settings:
            timing["settings"] = settings
        return timing

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
    # actions, so it — not the raw BNGL — is the source of both the SSA netgen caps
    # and the simulate horizon. Non-injected SSA jobs preserve the model's own
    # generate_network(...) call when it carries caps such as max_stoich/max_agg; NF
    # builds XML directly (network-free), so its fixture carries no generate_network.
    inject = spec.get("inject")
    horizon_text = inject or bngl_text
    gen_network = bc.injected_gen_network(inject) or (
        None if inject is not None else bc._model_gen_network(bngl_text)
    )
    # GH #177/#179: pre-representative state. A dirty_carryover model is driven through
    # the FULL protocol (multi_segment) on BOTH engines; a single-phase model bakes the
    # prefix into the shared artifact (option-1). multiseg is gated on the protocol being
    # session-replayable by ONE engine — a cross-engine ode/ssa->nf carry-over is not, so
    # it (and an injected fixture) keeps the single-segment ensemble. The replayability
    # probe uses simulate METHODS only (no .net needed yet), so it precedes the build.
    state_prefix, prefix_info = bc.state_setup_prefix(horizon_text, track=track)
    steps_probe, rep_probe = bc.parse_protocol(
        horizon_text, methods=bc._TRACK_METHODS[track], default_method=track, net_params={}
    )
    multiseg = (
        prefix_info["dirty_carryover"]
        and inject is None
        and rep_probe is not None
        and bc.protocol_session_replayable(steps_probe, track)
    )

    workdir = Path(tempfile.mkdtemp(prefix=f"bng_{track}_"))
    try:
        # 1. Build the shared artifact. A multiseg replay starts from the DECLARED IC
        #    (CLEAN artifact); a single-segment run bakes the state prefix.
        art_prefix = "" if multiseg else state_prefix
        if track == "ssa":
            artifact, build_sec, build_err = bc.generate_network(
                bngl_text,
                spec["bng2_pl"],
                workdir,
                timeout=spec["sub_timeout"],
                gen_network=gen_network,
                state_prefix=art_prefix,
            )
        else:
            artifact, build_sec, build_err = bc.generate_xml(
                bngl_text,
                spec["bng2_pl"],
                workdir,
                timeout=spec["sub_timeout"],
                state_prefix=art_prefix,
            )
        build_key = "netgen_sec" if track == "ssa" else "writexml_sec"
        if artifact is None:
            res["status"] = "bad_test"
            res["comment"] = (
                "Could not build this model to run (network/model generation failed) — a model "
                "problem, not bngsim."
            )
            res["exception"] = build_err
            res["timing"] = _with_settings(
                {"build": {build_key: round(build_sec, 6)}, "warmup": warmup}
            )
            q.put(res)
            return

        # 2. Continuation-aware representative spec (GH #179: fixes the t_start=0 secondary
        #    bug). For SSA tokens resolve against the .net parameter table; NF has no .net
        #    (numeric only). Injected fixtures / bare-action text fall back to the raw-text
        #    scanner (always single-segment). Unresolvable horizon -> SKIP.
        net_params = bc.read_net_parameters(artifact) if track == "ssa" else {}
        sspec = bc.representative_spec(horizon_text, track=track, net_params=net_params)
        if sspec is None:
            o = bc.parse_stoch_spec(horizon_text, net_params, track=track)
            if o is None:
                res["status"] = "skip"
                res["comment"] = (
                    f"No runnable {track.upper()} time-course (no simulate with a numeric t_end) "
                    "— nothing to compare."
                )
                res["timing"] = _with_settings(
                    {"build": {build_key: round(build_sec, 6)}, "warmup": warmup}
                )
                q.put(res)
                return
            sspec = {
                **o,
                "dirty_carryover": False,
                "steps": None,
                "rep_index": None,
                "rep_stmt_idx": None,
            }
            multiseg = False
        if sspec.get("steps") is None:
            multiseg = False
        if track == "nf":
            # NFsim has no t_start offset — it always simulates [0, t_end]. Force the
            # bngsim side to match so the two grids coincide (the corpus has no
            # t_start>0 NF action today; this only guards a future one).
            sspec["t_start"] = 0.0
            sspec["nf_bscb"] = spec["nf_bscb"]
            # A Species-type observable forces NFsim's complex book-keeping (-cb),
            # which bngsim enables automatically; detect it once so the legacy
            # ensemble matches.
            sspec["nf_complex_bookkeeping"] = bc.nf_needs_complex_bookkeeping(bngl_text)
        rep_timeout = spec["rep_timeout"]

        bn = bn_timing = None
        bn_exc = ""
        bn_unsupported = False
        bn_rep_timeout = False
        bn_wall = 0.0
        leg = leg_timing = None
        leg_exc = ""
        leg_wall = 0.0

        if multiseg:
            # GH #179: full-protocol carry-over ENSEMBLE — bngsim replay (per seed) vs
            # native BNG2.pl protocol (per seed), both reduced to the representative
            # segment. Any replay failure (a refusal, a SimulationTimeout, an
            # unresolvable species pattern, or a native-oracle error) FALLS BACK to the
            # single-segment ensemble below — the honest baseline; a genuine bngsim bug
            # still surfaces there, classified normally.
            try:
                t0 = time.perf_counter()
                bn = bc.bn_stoch_multiseg_ensemble(
                    artifact,
                    sspec,
                    track=track,
                    n_rep=spec["n_rep"],
                    seed_base=spec["seed_base"],
                    rep_timeout=rep_timeout,
                    block_same_complex_binding=(sspec["nf_bscb"] if track == "nf" else True),
                )
                bn_wall = time.perf_counter() - t0
                t0 = time.perf_counter()
                leg = bc.native_stoch_ensemble(
                    horizon_text,
                    spec["bng2_pl"],
                    workdir / "native",
                    track=track,
                    rep_stmt_idx=sspec["rep_stmt_idx"],
                    n_rep=spec["n_rep"],
                    seed_base=spec["seed_base"],
                    timeout=spec["sub_timeout"],
                    target_time=bn[0],
                )
                leg_wall = time.perf_counter() - t0
                # The multiseg ensemble runners return only (time, obs, names) — no
                # per-rep timing breakdown — but we DID measure the full N-replicate
                # WORKFLOW wall for each engine. Record it so multi-phase models appear
                # in the ensemble (WORKFLOW) figure instead of "produced no ensemble".
                # The harness-stripped ENGINE numbers (warm trajectory / propagation
                # CPU) are not available for a multi-segment protocol, so they stay
                # absent and these models simply don't enter the ENGINE comparison.
                _nrep = max(int(spec["n_rep"]), 1)
                bn_timing = {
                    "load_sec": None,
                    "ensemble_sec": round(bn_wall, 6),
                    "rep_median_sec": round(bn_wall / _nrep, 6),
                    "n_rep": int(spec["n_rep"]),
                    "multi_segment": True,
                    "config": {
                        "method": "Gillespie SSA (exact)" if track == "ssa" else "NFsim",
                        "rhs": "multi-phase protocol replay (per-segment)",
                        "codegen": None,
                    },
                }
                leg_timing = {
                    "ensemble_sec": round(leg_wall, 6),
                    "rep_median_sec": round(leg_wall / _nrep, 6),
                    "n_rep": int(spec["n_rep"]),
                    "n_calls": int(spec["n_rep"]),
                    "multi_segment": True,
                    "config": {
                        "method": f"{spec['legacy_label']} protocol",
                        "codegen": "multi-phase",
                    },
                }
            except Exception:
                multiseg = False
                bn = leg = None
                bn_wall = leg_wall = 0.0
                bn_timing = leg_timing = None

        if not multiseg:
            # 3a. bngsim single-segment ensemble (record status; do not short-circuit).
            try:
                t0 = time.perf_counter()
                t, v, n, bn_timing = _bn_stoch_ensemble(
                    track, artifact, sspec, spec["n_rep"], spec["seed_base"], rep_timeout
                )
                bn_wall = time.perf_counter() - t0
                bn = (t, v, n)
            except Exception as exc:
                # A clean validate_for_ssa refusal is UNSUPPORTED (an expected,
                # documented non-run), not an actionable EXCEPTION.
                if track == "ssa" and type(exc).__name__ == "SsaValidationError":
                    bn_unsupported = True
                    bn_exc = f"bngsim refused (validate_for_ssa): {exc}"[:400]
                elif type(exc).__name__ == "SimulationTimeout":
                    # A per-replicate wall-cap timeout is a TIMEOUT, not an actionable
                    # bngsim bug: bngsim is handed a tighter clock than the legacy
                    # reference (rep_timeout vs the full sub_timeout the legacy ensemble
                    # gets), so a heavy-but-correct model trips it while the reference
                    # finishes. Bucket it as TIMEOUT so it does not inflate EXCEPTION.
                    bn_rep_timeout = True
                    bn_exc = f"bngsim: SimulationTimeout: {exc}"[:400]
                else:
                    bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]

            # 3b. legacy single-segment ensemble (the reference; always run).
            try:
                t0 = time.perf_counter()
                t, v, n, leg_timing = _legacy_stoch_ensemble(
                    track,
                    artifact,
                    sspec,
                    spec["legacy_bin"],
                    spec["n_rep"],
                    spec["seed_base"],
                    str(workdir / "leg"),
                    spec["sub_timeout"],
                )
                leg_wall = time.perf_counter() - t0
                leg = (t, v, n)
            except Exception as exc:
                leg_exc = f"{spec['legacy_label']}: {type(exc).__name__}: {exc}"[:400]

        res["wall_sec"] = round(build_sec + bn_wall + leg_wall, 3)
        if leg is not None:
            res["leg_finite"] = bool(np.isfinite(leg[1]).all())

        n_obs = 0
        if bn is not None:
            n_obs = int(bn[1].shape[2]) if bn[1].ndim == 3 else 0
        elif leg is not None:
            n_obs = int(leg[1].shape[2]) if leg[1].ndim == 3 else 0
        timing: dict = {"build": {build_key: round(build_sec, 6)}}
        timing["spec"] = {
            "t_start": sspec["t_start"],
            "t_end": sspec["t_end"],
            "n_steps": int(sspec["n_steps"]),
            "n_rep": int(spec["n_rep"]),
            "seed_base": int(spec["seed_base"]),
            "n_obs": n_obs,
            "gml": sspec.get("gml"),
            # GH #179: dirty_carryover -> full-protocol replay vs native BNG2.pl protocol.
            "mode": "multi_segment" if multiseg else "single_segment",
        }
        if bn_timing:
            timing["bngsim"] = bn_timing
        if leg_timing:
            timing["legacy"] = leg_timing
        timing["warmup"] = warmup
        res["timing"] = _with_settings(timing)

        # 4. Classify from per-engine status. The plain-English ``comment`` is the
        #    headline; the raw engine error stays in ``exception`` as a muted detail.
        if bn_unsupported:
            # bngsim cleanly refused. UNSUPPORTED regardless of the legacy outcome
            # (no bngsim trajectory to compare); keep the legacy timing for the row.
            res["status"] = "unsupported"
            res["comment"] = (
                "bngsim declines this model under this method (documented limitation) — no "
                "result to compare."
            )
            res["exception"] = bn_exc
            q.put(res)
            return
        if bn_rep_timeout:
            # bngsim hit the per-replicate wall cap. TIMEOUT regardless of the legacy
            # outcome (no completed bngsim ensemble to compare); keep legacy timing.
            res["status"] = "rep_timeout"
            res["comment"] = (
                f"A bngsim replicate exceeded the {rep_timeout:g}s cap; comparison stopped."
            )
            res["exception"] = bn_exc
            q.put(res)
            return
        if bn_exc or leg_exc:
            if bn_exc and leg_exc:
                res["status"], res["exception"] = "bad_test", f"{leg_exc} || {bn_exc}"
                res["comment"] = (
                    "Neither engine could run this model — a model/setup problem, not bngsim."
                )
            elif leg_exc:
                res["status"], res["exception"] = "reference_failed", leg_exc
                res["comment"] = reference_failure_comment(track, spec["legacy_label"], leg_exc)
            else:
                res["status"], res["exception"] = "exception", bn_exc
                res["comment"] = (
                    "bngsim errored while the reference engine succeeded — an actionable bngsim issue."
                )
            q.put(res)
            return

        def _escalate_pair(delta, seed_base, rung):
            """Run ``delta`` MORE replicates of BOTH engines at ``seed_base`` (the same
            multiseg/single-segment path that produced the base ensembles), into
            rung-unique scratch dirs so successive rungs never collide. Returns the new
            ``(bn, leg)`` slices to pool onto the running ensembles."""
            if multiseg:
                b = bc.bn_stoch_multiseg_ensemble(
                    artifact,
                    sspec,
                    track=track,
                    n_rep=delta,
                    seed_base=seed_base,
                    rep_timeout=rep_timeout,
                    block_same_complex_binding=(sspec["nf_bscb"] if track == "nf" else True),
                )
                leg = bc.native_stoch_ensemble(
                    horizon_text,
                    spec["bng2_pl"],
                    workdir / f"native_n{rung}",
                    track=track,
                    rep_stmt_idx=sspec["rep_stmt_idx"],
                    n_rep=delta,
                    seed_base=seed_base,
                    timeout=spec["sub_timeout"],
                    target_time=b[0],
                )
            else:
                t, v, n, _tm = _bn_stoch_ensemble(
                    track, artifact, sspec, delta, seed_base, rep_timeout
                )
                b = (t, v, n)
                t, v, n, _tm = _legacy_stoch_ensemble(
                    track,
                    artifact,
                    sspec,
                    spec["legacy_bin"],
                    delta,
                    seed_base,
                    str(workdir / f"leg_n{rung}"),
                    spec["sub_timeout"],
                )
                leg = (t, v, n)
            return b, leg

        try:
            status, value, comment, metric, m_tol = _compare_stoch(bn, leg)
            # GH #69 seed-escalation oracle / GH #185 direction-of-change discriminator:
            # a BORDERLINE diff accumulates replicates up ESCALATION_RUNGS and is judged
            # by whether agreement IMPROVES (finite-N noise -> converges to PASS) or
            # stalls/worsens (a real disagreement -> stop early). The pooled ensemble at
            # each rung is the prior one plus the seeds in the next disjoint block, so
            # seeds are never recomputed. _run_escalation is pure w.r.t. its two
            # callables (unit-tested in tests/test_escalation.py).
            status, value, comment, metric, m_tol, prog, _stalled = _run_escalation(
                bn,
                leg,
                status,
                value,
                comment,
                metric,
                m_tol,
                n_rep_base=int(spec["n_rep"]),
                seed_base=spec["seed_base"],
                compare=_compare_stoch,
                escalate=_escalate_pair,
            )
            if len(prog) > 1:
                res["escalation"] = prog
            res["status"], res["value"], res["comment"] = status, value, comment
            res["metric"], res["tol"] = metric, m_tol
            # GH #183/#69: a DIFF on a known reference-side NF bug (legacy NFsim) or a
            # catalogued SSA disposition (legacy outlier / benign artifact) is not a bngsim
            # bug. Keep the honest DIFF but tag it so the matrix renders it non-scoring
            # (KNOWN REF / KNOWN ARTIFACT); the verdict is unchanged.
            annotate_ref_bscb(res, status, track, bngl_path.stem)
            annotate_ssa_known(res, status, track, bngl_path.stem)
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
    "unsupported": Outcome.UNSUPPORTED,
    "bad_test": Outcome.BAD_TEST,
    "skip": Outcome.SKIP,
    "timeout": Outcome.TIMEOUT,  # scheduler hard per-job wall cap
    "rep_timeout": Outcome.TIMEOUT,  # bngsim per-replicate wall cap (in-worker)
    "dead": Outcome.EXCEPTION,
}


def _filter_jobs(jobs, args, track):
    """Jobs of the stochastic regime whose ``params.methods`` includes the track."""
    out = [
        j for j in jobs if j.method == "stochastic" and track in (j.params.get("methods") or [])
    ]
    if args.models:
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        out = [j for j in out if j.model_id in wanted]
    if args.include:
        out = [j for j in out if args.include in j.model]
    if args.exclude:
        out = [j for j in out if args.exclude not in j.model]
    if args.tier:
        tiers = {x.strip() for x in args.tier.split(",") if x.strip()}
        out = [j for j in out if j.params.get("tier") in tiers]
    if args.limit:
        out = out[: args.limit]
    return out


def _job_timeout_override(job) -> dict | None:
    """The per-job wall-clock cap override, when no global ``--timeout`` is forced."""
    for ov in job.overrides:
        if ov.field == "timeout":
            try:
                return {"timeout": float(ov.value), "reason": ov.reason}
            except (ValueError, TypeError):
                return None
    return None


def _override_summary(jobs) -> tuple[list[dict], dict[str, int]]:
    records: list[dict] = []
    counts: dict[str, int] = {}
    for j in jobs:
        for ov in j.overrides:
            counts[ov.field] = counts.get(ov.field, 0) + 1
            records.append(
                {
                    "model_id": j.model_id,
                    "field": ov.field,
                    "value": ov.value,
                    "reason": ov.reason,
                }
            )
    return records, counts


def _make_progress():
    def _cb(done: int, total: int, res: dict) -> None:
        if done % 10 == 0 or done == total:
            st = res.get("status", "?")
            print(f"  [{done}/{total}] {st:16s} {res.get('model_id', '')}", flush=True)

    return _cb


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--track", choices=["ssa", "nf"], required=True, help="Stochastic track.")
    ap.add_argument("--jobs", default=str(JOBS), help="_core manifest (default jobs.json)")
    ap.add_argument("--out", default="", help="report path (default runs/report_<track>.json)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent job subprocesses.")
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Per-job wall cap (s); defaults to {DEFAULT_STOCH_TIMEOUT}s.",
    )
    ap.add_argument(
        "--n-rep",
        type=int,
        default=DEFAULT_N_REP,
        help=f"Replicates per engine (default {DEFAULT_N_REP}; the corpus' modeler n_rep).",
    )
    ap.add_argument(
        "--seed-base", type=int, default=DEFAULT_SEED_BASE, help="First ensemble seed."
    )
    ap.add_argument(
        "--rep-timeout",
        type=float,
        default=None,
        help="Per-replicate wall cap (s) bounding each bngsim trajectory (a runtime "
        "sign-indefinite SSA model can fire unboundedly). Default: sub_timeout/(2·n_rep).",
    )
    ap.add_argument(
        "--nf-bscb",
        choices=["on", "off"],
        default="on",
        help="NF only: bngsim block_same_complex_binding (default on — bngsim's "
        "CORRECT default; the legacy NFsim has no -bscb, so ring-formers DIFF as "
        "expected reference bugs). 'off' matches the legacy binary for apples-to-apples.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max jobs after filtering (0=all).")
    ap.add_argument("--models", default="", help="Comma-separated model_id filter.")
    ap.add_argument("--include", default="", help="Substring filter on the model path.")
    ap.add_argument("--exclude", default="", help="Substring filter — drop matching model paths.")
    ap.add_argument(
        "--tier", default="", help="Comma-separated tier filter (original/slow/glacial)."
    )
    args = ap.parse_args()

    track = args.track
    bng2_pl, legacy_bin = _resolve_bng_tools(track)
    legacy_label = TRACKS[track]["legacy_label"]

    jobs_path = Path(args.jobs).resolve()
    if not jobs_path.exists():
        sys.exit(f"missing {jobs_path}")
    models_root = (jobs_path.parent / "models").resolve()
    _meta, alljobs = read_manifest(jobs_path)
    jobs = _filter_jobs(alljobs, args, track)
    if not jobs:
        sys.exit(f"no {track} jobs after filtering.")

    missing = [j.model_id for j in jobs if not (models_root / j.model).exists()]
    if missing:
        sys.exit(
            f"{len(missing)} job(s) have no vendored BNGL (e.g. {missing[:3]}). "
            "Run `python vendor_corpus.py` to materialize the model tree."
        )

    default_cap = args.timeout if args.timeout is not None else DEFAULT_STOCH_TIMEOUT
    default_timeout_source = "cli" if args.timeout is not None else "default"
    nf_bscb = args.nf_bscb == "on"

    n_timeout_ov = 0
    override_records, override_counts = _override_summary(jobs)
    specs = []
    for j in jobs:
        timeout_ov = None if args.timeout is not None else _job_timeout_override(j)
        if timeout_ov:
            cap = timeout_ov["timeout"]
            timeout_source = "job_override"
            timeout_reason = timeout_ov["reason"]
            n_timeout_ov += 1
        else:
            cap = default_cap
            timeout_source = default_timeout_source
            timeout_reason = None
        sub_timeout = max(15.0, cap - 30.0)
        rep_timeout = (
            args.rep_timeout
            if args.rep_timeout is not None
            else sub_timeout / (2 * max(args.n_rep, 1))
        )
        settings = {
            "timeout_sec": float(cap),
            "subprocess_timeout_sec": float(sub_timeout),
            "rep_timeout_sec": float(rep_timeout),
            "timeout_source": timeout_source,
            "timeout_reason": timeout_reason,
            "rep_timeout_source": "cli" if args.rep_timeout is not None else "derived",
            "n_rep": int(args.n_rep),
            "seed_base": int(args.seed_base),
            "nf_block_same_complex_binding": nf_bscb if track == "nf" else None,
            "non_default": (
                timeout_source != "default"
                or args.rep_timeout is not None
                or args.n_rep != DEFAULT_N_REP
                or args.seed_base != DEFAULT_SEED_BASE
                or (track == "nf" and args.nf_bscb != "on")
            ),
        }
        specs.append(
            {
                "key": f"{j.model_id}:{track}",
                "model_id": j.model_id,
                "method": j.method,
                "track": track,
                "metric": j.oracle.metric,
                "tol": j.oracle.tol,
                "bngl": str(models_root / j.model),
                "n_rep": int(args.n_rep),
                "seed_base": int(args.seed_base),
                "rep_timeout": float(rep_timeout),
                "nf_bscb": nf_bscb,
                "cap": float(cap),
                "sub_timeout": float(sub_timeout),
                "settings": settings,
                "bng2_pl": bng2_pl,
                "legacy_bin": legacy_bin,
                "legacy_label": legacy_label,
                "inject": bc.injected_action_block(j.overrides),
            }
        )

    ver = versions.stamp("bng")
    print("=" * 72)
    print(f"  bngsim vs {legacy_label} — BNGL {track.upper()} parity + timing (bng_parity)")
    print("=" * 72)
    print(f"  jobs: {len(specs)}   workers: {args.workers}   track: {track}")
    print(f"  bngsim {ver['bngsim']}   BNG {ver.get('bng')}   {legacy_label}: {legacy_bin}")
    print(
        f"  ensemble: n_rep={args.n_rep} seed_base={args.seed_base}   "
        f"oracle: _core.differ.ensemble_verdict (K={differ.ensemble_k_for(args.n_rep):.2f} "
        f"= 3.0·√(N/10), frac>={differ.ENSEMBLE_PASS_FRAC})"
    )
    if track == "nf":
        print(f"  NF block_same_complex_binding (bngsim): {'on' if nf_bscb else 'off'}")
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
        timing = r.get("timing")
        if not timing and r.get("settings"):
            timing = {"settings": r.get("settings")}
        results.append(
            JobResult(
                model_id=r["model_id"],
                method=r.get("method", "stochastic"),
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
                timing=timing,
            )
        )

    results.sort(key=lambda x: x.model_id)
    counts = tally(r.outcome for r in results)
    default_out = HERE / "runs" / f"report_{track}.json"
    out_path = Path(args.out).resolve() if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite": "bng_parity",
        "reference_engine": "bng",
        "reference_label": legacy_label,
        "regime": TRACKS[track]["regime"],
        "git_rev": git_rev(str(HERE)),
        "bngpath": os.environ.get("BNGPATH"),
        "legacy_bin": legacy_bin,
        "versions": ver,
        "tally": counts,
        "n_jobs": len(results),
        "elapsed_sec": round(elapsed, 2),
        "hardware": bc.hardware_info(),
        "concurrency": {"workers": args.workers, "mode": "process-parallel"},
        "config": {
            "track": track,
            "bngsim_method": "ssa" if track == "ssa" else "nf",
            "n_rep": args.n_rep,
            "seed_base": args.seed_base,
            "rep_timeout_sec": None if args.rep_timeout is None else round(args.rep_timeout, 3),
            "rep_timeout_source": "derived per job" if args.rep_timeout is None else "cli",
            "nf_block_same_complex_binding": nf_bscb if track == "nf" else None,
            "timeout_overridden_jobs": n_timeout_ov,
        },
        "runtime": {
            "default_timeout_sec": DEFAULT_STOCH_TIMEOUT,
            "global_timeout_sec": default_cap,
            "global_timeout_source": default_timeout_source,
            "subprocess_timeout_rule": "max(15, timeout_sec - 30)",
            "rep_timeout_rule": "cli --rep-timeout, else subprocess_timeout_sec / (2 * n_rep)",
            "timeout_overridden_jobs": n_timeout_ov,
        },
        "overrides": {
            "counts_by_field": override_counts,
            "jobs": override_records,
            "note": (
                "Overrides are read from Job.overrides, baked by build_jobs.py from "
                "overrides.py. A timeout override only applies when no global --timeout "
                "is supplied. t_end_cap/n_scan_pts/action_inject overrides are applied "
                "while building the shared artifact/protocol. Resolved settings are echoed "
                "in timing.settings so regenerated matrices retain provenance."
            ),
        },
        "ensemble": {
            "k_sigma": round(differ.ensemble_k_for(args.n_rep), 4),
            "k_sigma_base": differ.ENSEMBLE_K,
            "k_sigma_scaling": "3.0·√(N/10) — effect-size-preserving (GH #190)",
            "pass_frac": differ.ENSEMBLE_PASS_FRAC,
            "metric": "ensemble_frac_pass",
        },
        "oracle_basis": (
            "Cross-engine STOCHASTIC tolerance via the shared _core ensemble oracle "
            "— the same protocol the bng_parity correctness suite uses. BNG2.pl "
            f"produces the shared artifact ONCE per model ({'reaction-network .net' if track == 'ssa' else 'BNG-XML'}); "
            f"BNGsim runs it in-process ({'method=ssa' if track == 'ssa' else 'NfsimSession'}) and the legacy "
            f"{legacy_label} runs the SAME artifact as a subprocess, each an "
            f"N={args.n_rep}-replicate ensemble on an identical seed schedule "
            f"(seed_base={args.seed_base}). The two ensembles are reduced to per-cell "
            "(mean, sem) over the observables (.gdat, aligned BY NAME — the uniform "
            "axis: NF has no species) and compared with ensemble_verdict: a per-cell "
            f"K={differ.ensemble_k_for(args.n_rep):.2f}-sigma (= 3.0·√(N/10), effect-size-"
            "preserving so a larger N sharpens precision without flagging tinier diffs) test "
            "on the difference of means with a "
            "relative-agreement escape hatch and a near-zero skip; PASS iff "
            f">= {differ.ENSEMBLE_PASS_FRAC} of cells pass. metric=ensemble_frac_pass. "
            "Failure is attributed per-engine: bngsim raised + legacy ran -> EXCEPTION "
            "(actionable bngsim bug); bngsim ran + legacy raised -> REFERENCE_FAILED "
            "(non-scoring); a clean validate_for_ssa refusal -> UNSUPPORTED; "
            "build (netgen/writeXML) failed or both raised -> BAD_TEST; no resolvable "
            "horizon -> SKIP; wall-cap overrun -> TIMEOUT. "
            + (
                "NF: bngsim runs at block_same_complex_binding="
                + (
                    "True (its correct default); the legacy NFsim v1.14.3 lacks -bscb, "
                    "so ring-forming/multivalent models DIFF as expected REFERENCE bugs "
                    "(not bngsim defects — the correctness suite reclassifies them "
                    "PASS_REF_BUG)."
                    if nf_bscb
                    else "False (matched to the legacy NFsim)."
                )
                if track == "nf"
                else ""
            )
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
    if n_timeout_ov:
        print(f"  overrides: {n_timeout_ov} timeout-overridden")
    print(f"  report: {out_path}")
    print("=" * 72)
    from _core.taxonomy import FAILING

    n_fail = sum(counts.get(o.value, 0) for o in FAILING)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
