#!/usr/bin/env python3
"""bngsim vs the legacy BNG2.pl/run_network stack — ODE timing + parity (bng_parity).

The bng sibling of ``rr_parity/rr_run.py`` / ``amici_parity/amici_run.py``: it
reuses the suite's ``jobs.json`` corpus (the 591 deterministic/ODE jobs) and the
shared ``_core.differ`` oracle, but the reference engine is the legacy CLI stack
rather than a Python library, so each job runs through a three-step pipeline in
its own spawned worker:

  1. **Network generation (BNG2.pl)** — the shared per-model build *prefix*. The
     BNGL model (simulation actions stripped, model-authored ``generate_network``
     caps preserved when present) is run through BNG2.pl once to produce the
     reaction-network ``.net``. The same
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
bound (a regression blows past it and re-flags). A DIFF that is run_network leaving
the model's exact solution (``KNOWN_DETERMINISTIC_REF``, GH #1548) renders KNOWN REF,
but only while bngsim's own run is checked against that exact solution and holds.
Otherwise this track adjudicates every model independently on the network BNG2.pl
produced. (The artifact dispositions were folded here from the retired correctness
suite, GH #69.)

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
import hashlib
import json
import math
import os
import re
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
    require_bng,
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
    """(BNG2.pl, run_network) via the shared resolver; never hardcoded.

    ``_core.require_bng`` takes ``$BNG2_PL`` / ``$BNGPATH`` first (either the
    BioNetGen folder or a direct BNG2.pl path) and falls back to the copy
    PyBioNetGen bundles, reporting every place it looked if none works.
    run_network lives at ``<root>/bin/run_network``.
    """
    r = require_bng(
        "the reference stack (BNG2.pl netgen + run_network ODE) must be "
        "resolvable to run this matrix."
    )
    bng2_pl = str(r.bng2_pl)
    run_network = str(r.root / "bin" / "run_network")
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
    # max_abs here is the phase offset in disguise: about 96 times the gap between
    # the two engines' division times, the largest difference sitting in V where
    # it falls. The gap comes from rounding, not from the model. Re-measured
    # 2026-10-11 over 3,362 pairs of default-tolerance runs (bngsim ExprTk and
    # compiled against run_network, each at 41 rtol = atol from 0.9e-8 to
    # 1.1e-8): max_abs 0.52 median, 1.73 at the 99th percentile, 2.38 worst, and
    # above the old bound of 1.0 in 12% of pairs, which is how CI lost the tag
    # after #996. A real error moves it further: +0.1% on the growth rate gives
    # 5.8, +0.1% on Vmax 7.1, +1% on the decay rate 5.5.
    "proliferation": {
        "max_abs_bound": 4.0,
        "issue": "stiff-oscillator phase wander (re-measured 2026-10-11)",
        "reason": "stiff relaxation-oscillator phase wander across sharp tanh() "
        "switches (division times within 0.03 time units over 8 cycles of 11.57); "
        "period, amplitude and cycle count match both sides.",
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


# Chaotic / sensitive-dependence systems, keyed by BNGL stem. Unlike the bounded
# artifacts above, these are NOT gated on a small ``max_abs``: for a chaotic IVP
# large pointwise divergence between two correct integrators is mathematically
# inevitable (positive Lyapunov exponent → exponential decorrelation), so magnitude
# is not a valid discriminator. The disposition is a deliberate, per-model catalogue
# decision — pointwise trajectory parity is simply the wrong test for these systems.
KNOWN_CHAOTIC_SYSTEMS = {
    "ph_lorenz_attractor": {
        "issue": "deterministic chaos — pointwise trajectory parity ill-posed",
        "reason": "The Lorenz attractor is the canonical chaotic system (largest Lyapunov "
        "exponent ~0.9): two correct integrators taking different internal step sequences "
        "decorrelate exponentially. Over t_end=50 (~45 Lyapunov times) full pointwise "
        "divergence is inevitable and says nothing about integrator correctness — both stacks "
        "integrate the identical .net at 1e-8 tol. max_rel_err is not a valid discriminator "
        "here; the correct test is a statistical/invariant one (bounded attractor extent, "
        "matching moments).",
    },
}


def annotate_chaotic_artifact(res: dict, status: str, stem: str) -> dict:
    """Tag a known chaotic-system DIFF in place; return ``res``.

    A DIFF on a catalogued chaotic model (``KNOWN_CHAOTIC_SYSTEMS``) is a comparison
    artifact, not a bngsim defect: sensitive dependence guarantees two correct integrators
    diverge pointwise. We keep the honest ``status="diff"`` but set
    ``res["subclass"]="known_artifact"`` and rewrite the comment so the matrix renders it
    KNOWN ARTIFACT (non-scoring). Deliberately NOT gated on a magnitude bound (see
    ``KNOWN_CHAOTIC_SYSTEMS``). A no-op for any non-DIFF status or uncatalogued model.
    """
    if status != "diff":
        return res
    art = KNOWN_CHAOTIC_SYSTEMS.get(stem)
    if art is None:
        return res
    res["subclass"] = "known_artifact"
    res["comment"] = (
        f"Known comparison artifact, not a bngsim error: {art['reason']} Pointwise "
        f"trajectory parity is ill-posed for a chaotic system → non-scoring. "
        f"(Raw: {res.get('comment', '')})"
    )
    return res


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

    Either way the comment states this run's ``max_abs`` against the bound, near
    its start, so the number survives the nightly's 200-character verdict note. A
    catalogued model that stops being excused has to say by how much: the CI
    nightlies after #996 dropped proliferation's tag and recorded no number.
    """
    if status != "diff":
        return res
    art = KNOWN_DETERMINISTIC_ARTIFACTS.get(stem)
    if art is None:
        return res
    bound = art["max_abs_bound"]
    raw = res.get("comment", "")
    observed = "not measured" if max_abs is None else f"{max_abs:.3g}"
    if not (max_abs is not None and np.isfinite(max_abs) and max_abs <= bound):
        res["comment"] = (
            f"Catalogued known artifact, but max_abs {observed} is past its bound "
            f"{bound:g}, so this DIFF scores. {raw}"
        )
        return res
    res["subclass"] = "known_artifact"
    res["comment"] = (
        f"Known comparison artifact, not a bngsim error (max_abs {observed}, within its "
        f"bound {bound:g}): {art['reason']} → non-scoring. (Raw: {raw})"
    )
    return res


def ltcc_steady_state(v_rec: float) -> dict[str, float]:
    """Every species of l-type-calcium-channel-dynamics at steady state, v_rec() = ``v_rec``.

    The closed form the GH #1548 entry below holds bngsim's run to. For every t > 0
    the model's two state switches have fixed values on the exact solution:
    Phospho_LTCC > 0, so v_pka_shift() = k_pka_shift = -5, and Voltage() =
    50 - 49e^(-t) stays below 50, so v_rec() = 0.5. As Voltage() -> 50, v_open() ->
    k_open_max/(1 + e^0) = 5. Gating and inactivation read nothing of p~, and p~ reads
    nothing else, so a channel is p~P with probability 20/(20 + 0.01) apart from its
    (g, s) state, and the four (g, s) states of the 500 channels balance among
    themselves. With c = k_close, a = v_open, r = v_rec and i = v_inact() =
    5 Ca/(200 + Ca), each channel in OR (open, conducting) has beside it i/(c + r) in
    OI, c i/((c + r) r) in CI and c (1 + i/(c + r))/a in CR, so OR = 500/(A + B i) with
    A = 1 + c/a and B = (1 + c/r + c/a)/(c + r). Calcium() balances
    k_ca_flux OR = k_ca_pump Ca. The two make one quadratic in Ca, whose positive root
    is Calcium(): 957.956 at v_rec = 0.5, and 187.682 at the other branch's 0.05.
    Keyed by the .net's species names.
    """
    k_close, v_open, total = 0.5, 5.0, 500.0
    k_inact_max, km_ca = 5.0, 200.0
    gain = 15.0 / 0.8  # k_ca_flux / k_ca_pump
    a = 1.0 + k_close / v_open
    b = (1.0 + k_close / v_rec + k_close / v_open) / (k_close + v_rec)
    # Ca (a (km_ca + Ca) + b k_inact_max Ca) = total gain (km_ca + Ca)
    qa = a + b * k_inact_max
    qb = a * km_ca - total * gain
    qc = -total * gain * km_ca
    ca = (-qb + math.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)
    i = k_inact_max * ca / (km_ca + ca)
    o_r = total / (a + b * i)
    o_i = i * o_r / (k_close + v_rec)
    c_i = k_close * o_i / v_rec
    c_r = k_close * (1.0 + i / (k_close + v_rec)) * o_r / v_open
    p = 20.0 / (20.0 + 0.01)
    u = 1.0 - p
    return {
        "LTCC(b,g~C,loc~mem,p~U,s~R)": c_r * u,
        "Calcium()": ca,
        "Voltage()": 50.0,
        "LTCC(b,g~O,loc~mem,p~U,s~R)": o_r * u,
        "LTCC(b,g~C,loc~mem,p~P,s~R)": c_r * p,
        "LTCC(b,g~O,loc~mem,p~P,s~R)": o_r * p,
        "LTCC(b,g~O,loc~mem,p~U,s~I)": o_i * u,
        "LTCC(b,g~C,loc~mem,p~U,s~I)": c_i * u,
        "LTCC(b,g~O,loc~mem,p~P,s~I)": o_i * p,
        "LTCC(b,g~C,loc~mem,p~P,s~I)": c_i * p,
    }


# Reference-side divergences on deterministic models, keyed by BNGL stem (GH #1548).
# The DIFF is run_network leaving the model's exact solution, and bngsim's run is on
# it. That is a claim about which engine is right, so it is checked on every run
# rather than bounded in size, and the row renders KNOWN REF (non-scoring) only while
#   - the two runs have the same outputs, so that no row goes unread;
#   - bngsim's ``knife_edge`` species is on the exact solution's ``side`` of
#     ``threshold`` at every output;
#   - run_network's ``knife_edge`` species reads as the threshold or past it at some
#     output (its .cdat has 13 significant digits, so this is where the printed value
#     gets there, at or before where the value it integrates does), and the two engines
#     pass the verdict on every output before the first that does;
#   - from that output to the end, bngsim's value of every ``exact_settled`` species
#     is its closed form, to ``KNOWN_REF_SETTLED_REL`` of it. That is the stretch
#     where run_network no longer stands for the exact solution, so bngsim is held to
#     the closed form instead, at every output and in every species.
# Any one failing leaves a scoring DIFF whose comment names it.
KNOWN_DETERMINISTIC_REF = {
    "l-type-calcium-channel-dynamics": {
        "issue": "GH #1548",
        "knife_edge": {"species": "Voltage()", "threshold": 50.0, "side": "below"},
        # From t = 20, where Voltage() is 1e-7 short of 50, every species of the exact
        # solution is within 6e-8 of its steady state. At the parity tolerance (1e-8)
        # neither engine's output reads Voltage() as 50 before t = 25; at a looser one
        # an output can read it sooner, but bngsim's then reaches 50 too and the row
        # scores.
        "exact_settled": ltcc_steady_state(0.5),
        "reason": "Voltage() relaxes as 50 - 49e^(-t) and never reaches 50, so on the "
        "exact solution -70 + Voltage_Level < -20 holds throughout, v_rec() stays 0.5 and "
        "Calcium() settles at 957.956 (closed form). The threshold is that steady state "
        "itself: once Voltage() is within an engine's tolerance of 50, rounding picks "
        "v_rec()'s branch. run_network's lands on 50.0, v_rec() drops to 0.05 and Calcium() "
        "settles at 187.682, that branch's closed form. bngsim's branch is set by rounding "
        "too (on 151 output points instead of 301 it takes run_network's), so each run is "
        "checked against the exact solution rather than trusted.",
    },
}
# How close bngsim has to be to an ``exact_settled`` closed form: far outside the
# runs' own error (the parity tolerance is 1e-8) and far inside the gap to the other
# branch (187.682 against 957.956).
KNOWN_REF_SETTLED_REL = 1e-6


def _check_known_ref(ref: dict, bn, rn, atol: float | None) -> tuple[str, str]:
    """(failure, evidence) for one catalogued reference divergence.

    ``failure`` is "" when every check of ``KNOWN_DETERMINISTIC_REF`` holds, else
    what failed; ``evidence`` says, for an excused row, where run_network left the
    exact solution and what bngsim's run was checked against.
    """
    bn_t, bn_a, bn_names = bn
    rn_t, rn_a, _rn_names = rn
    bn_a, rn_a = np.asarray(bn_a), np.asarray(rn_a)
    times = np.asarray(bn_t)
    # The verdict reads the two runs' overlap (align_net_species); this reads them
    # whole, or a row past run_network's last would never be asked about.
    if (
        bn_a.shape != rn_a.shape
        or times.shape != (bn_a.shape[0],)
        or not np.allclose(times, np.asarray(rn_t), rtol=1e-9, atol=0.0)
    ):
        return (
            f"the runs' outputs differ (bngsim {bn_a.shape}, run_network {rn_a.shape})",
            "",
        )
    names = list(bn_names)
    if len(names) != bn_a.shape[1]:
        return f"bngsim's run names {len(names)} species for {bn_a.shape[1]} columns", ""
    edge = ref["knife_edge"]
    sp, thr = edge["species"], float(edge["threshold"])
    if sp not in names:
        return f"bngsim's run has no species {sp}", ""
    col = names.index(sp)

    below = edge["side"] == "below"
    # A NaN is neither on the exact side nor at or past the threshold.
    off = ~(bn_a[:, col] < thr if below else bn_a[:, col] > thr)
    if off.any():
        k = int(np.argmax(off))
        return f"bngsim's {sp} reaches {thr:g} at t={times[k]:g}", ""
    past = rn_a[:, col] >= thr if below else rn_a[:, col] <= thr
    if not past.any():
        return f"run_network's {sp} never reads {thr:g}", ""
    k = int(np.argmax(past))
    if k == 0:
        return f"run_network's {sp} starts at or past {thr:g}", ""
    if not differ.deterministic_verdict(bn_a[:k], rn_a[:k], atol=atol)["passed"]:
        return (
            f"the engines already differ before run_network's {sp} reads {thr:g} "
            f"at t={times[k]:g}",
            "",
        )
    settled = ref["exact_settled"]
    missing = [name for name in settled if name not in names]
    if missing:
        return f"bngsim's run has no species {missing[0]}", ""
    for name, exact in settled.items():
        got = bn_a[k:, names.index(name)]
        bad = ~(np.abs(got - exact) <= KNOWN_REF_SETTLED_REL * abs(exact))  # NaN is bad
        if bad.any():
            j = k + int(np.argmax(bad))
            return (
                f"bngsim's {name} is {bn_a[j, names.index(name)]:.9g} at t={times[j]:g}, "
                f"not the exact {exact:.9g}",
                "",
            )
    evidence = (
        f"run_network's output reads {sp} as {thr:g} at t={times[k]:g} and it leaves the exact "
        f"solution, which bngsim's run keeps ({sp} {edge['side']} {thr:g} at every output; "
        f"from t={times[k]:g} on, all {len(settled)} species within "
        f"{KNOWN_REF_SETTLED_REL:g} of their closed forms)"
    )
    return "", evidence


def annotate_known_ref(
    res: dict, status: str, stem: str, bn, rn, atol=None, *, multi_segment: bool = False
) -> dict:
    """Tag a catalogued reference-side divergence in place; return ``res`` (GH #1548).

    A DIFF on a ``KNOWN_DETERMINISTIC_REF`` model keeps its honest ``status="diff"``
    and, while every check in that catalogue holds on this run's ``bn``/``rn``
    (``(time, values, names)`` each), gets ``res["subclass"]="ref_bscb"``, which the
    matrix renders bngsim PASS · run_network KNOWN REF (non-scoring). A failed check
    leaves a scoring DIFF whose comment says which, near its start, so it survives
    the nightly's 200-character verdict note. A no-op for any non-DIFF status, an
    uncatalogued model, or a ``multi_segment`` job, whose two runs are aligned by
    nearest time rather than row for row.
    """
    if status != "diff" or multi_segment:
        return res
    ref = KNOWN_DETERMINISTIC_REF.get(stem)
    if ref is None:
        return res
    raw = res.get("comment", "")
    failure, evidence = _check_known_ref(ref, bn, rn, atol)
    if failure:
        res["comment"] = (
            f"Catalogued known reference divergence ({ref['issue']}), but {failure}, so "
            f"this DIFF scores. {raw}"
        )
        return res
    res["subclass"] = "ref_bscb"
    res["comment"] = (
        f"Known reference divergence ({ref['issue']}), not a bngsim error: {evidence}. "
        f"{ref['reason']} → non-scoring. (Raw: {raw})"
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
        if v.get("no_signal"):
            # Issue #873: nothing moves past its own rounding in either trajectory,
            # so a frozen engine would match it too.
            return (
                f"NO SIGNAL: no species moves in either trajectory ({n_sp} species, "
                f"{scope}{extra}), so agreement says nothing about the dynamics."
            )
        return (
            f"Matches run_network within tolerance across all {n_sp} species, {scope}{extra}; "
            f"worst error {v.get('max_err_over_movement', 0.0):.1e} of a species' movement."
        )
    n_bad = v["n_hard_fail"] or v["n_fail"]
    return (
        f"Differs from run_network beyond tolerance at {n_bad}/{v['n_cells']} points "
        f"({n_sp} species), {scope}."
    )


def _compare_ode(bn, rn, *, atol=None) -> tuple[str, float, str, str, float, float]:
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
        return (
            "diff",
            float("inf"),
            "Disjoint species/time-points — nothing to compare.",
            metric,
            tol,
            float("inf"),
        )
    bn_a, rn_a, n_sp = aligned
    v = differ.deterministic_verdict(bn_a, rn_a, atol=atol)
    status = "pass" if v["passed"] else "diff"
    comment = _ode_comment(v, n_sp, multiseg=False)
    return status, v["max_rel"], comment, metric, tol, v["max_abs"]


def _compare_ode_multiseg(bn, rn, *, atol=None) -> tuple[str, float, str, str, float, float]:
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
        return (
            "diff",
            float("inf"),
            "Disjoint species/time-points — nothing to compare.",
            metric,
            tol,
            float("inf"),
        )
    idx = [int(np.argmin(np.abs(rn_t - t))) for t in bn_t]
    rn_on_grid = rn_v[idx]
    n_sp = min(bn_v.shape[1], rn_on_grid.shape[1])
    bn_a, rn_a = bn_v[1:, :n_sp], rn_on_grid[1:, :n_sp]  # drop the continue-boundary row
    if bn_a.shape[0] == 0:
        bn_a, rn_a = bn_v[:, :n_sp], rn_on_grid[:, :n_sp]
    v = differ.deterministic_verdict(bn_a, rn_a, atol=atol)
    status = "pass" if v["passed"] else "diff"
    comment = _ode_comment(v, n_sp, multiseg=True)
    return status, v["max_rel"], comment, metric, tol, v["max_abs"]


# --------------------------------------------------------------------------- #
# Worker (module-level so it is picklable under the 'spawn' start method)
# --------------------------------------------------------------------------- #
# ─── The reference archive (issue #805) ─────────────────────────────────────
#
# A live run can record, per job, the network BNG2.pl generated and what the
# reference (run_network, or the native BNG2.pl protocol for a carry-over model)
# returned; a later run replays from that record with neither BNG2.pl nor
# run_network installed. One ``.npz`` per job: the ``.net`` text (or the
# network-generation failure), the reference's time grid, values and names (or
# its exception), and a ``meta`` record of what the entry is the reference FOR.
# Identity is checked twice: the inputs before the network is read (the BNGL
# file's and any injected action block's sha256, the forced tolerances), and the
# spec derived from them after (interval, steps, tolerances, segment mode). An
# entry that matches neither the job nor its spec is stale and never compared:
# the row is REFERENCE_FAILED, ``archive_stale``; one that cannot be read at all
# is ``archive_unreadable``, the same row, rather than a worker death blamed on
# bngsim. Beside the identity, an entry records the reference engine's version
# (``reference_version``, #805's item 1) and the harness's git revision when it
# was recorded; a replayed row reports the archived version as the reference's.
# ARCHIVE_FORMAT is bumped where the harness's own network rules change (the
# state prefix, the netgen caps, an injected generate_network), since the
# archived ``.net`` was made under the rules of the recording run.
# 2: the file name carries a hash of the key, so two keys that sanitize alike
# no longer share a file.
ARCHIVE_FORMAT = 2


def _archive_file(directory: str | Path, key: str) -> Path:
    # The readable part is sanitized, so the hash keeps keys such as a/b and a_b apart.
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return Path(directory) / f"{re.sub(r'[^A-Za-z0-9._-]', '_', key)}-{digest}.npz"


def _sha256(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()


def archive_input_identity(spec: dict, bngl_text: str) -> dict:
    """The job's inputs an archived reference is valid for."""
    return {
        "format": ARCHIVE_FORMAT,
        "model_id": spec["model_id"],
        "method": spec["method"],
        "bngl_sha256": _sha256(bngl_text),
        "inject_sha256": _sha256(spec.get("inject")),
        "rtol": float(spec["rtol"]),
        "atol": float(spec["atol"]),
    }


def archive_spec_identity(ode: dict, dirty: bool) -> dict:
    """The run the reference was made for, as derived from the network."""
    return {
        "t_start": float(ode["t_start"]),
        "t_end": float(ode["t_end"]),
        "n_steps": int(ode["n_steps"]),
        "rtol": float(ode["rtol"]),
        "atol": float(ode["atol"]),
        "mode": "multi_segment" if dirty else "single_segment",
    }


def archive_save(directory, spec, record: dict, rn=None) -> Path:
    """Write one job's archived reference (atomically). ``record`` holds the
    identity, the network (``net_text`` or ``netgen_error``), the spec identity
    where one was derived, and the reference's exception and timing."""
    out = _archive_file(directory, spec["key"])
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"meta": np.frombuffer(json.dumps(record).encode(), dtype=np.uint8)}
    if rn is not None:
        arrays["time"] = np.asarray(rn[0], dtype=float)
        arrays["values"] = np.asarray(rn[1], dtype=float)
        arrays["names"] = np.frombuffer(json.dumps(list(rn[2])).encode(), dtype=np.uint8)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out)
    return out


def archive_load(directory, spec) -> tuple[dict | None, tuple | None]:
    """``(record, reference)`` for a job, or ``(None, None)`` with no entry."""
    path = _archive_file(directory, spec["key"])
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=False) as z:
        record = json.loads(bytes(z["meta"]).decode())
        rn = None
        if "time" in z:
            rn = (
                np.array(z["time"]),
                np.array(z["values"]),
                json.loads(bytes(z["names"]).decode()),
            )
    return record, rn


def _stale_fields(recorded: dict | None, want: dict) -> list[str]:
    return sorted(k for k in want if (recorded or {}).get(k) != want[k])


def _worker(spec: dict, q) -> None:
    """Run the netgen → BNGsim → run_network pipeline for one job; result on ``q``.

    Status is derived from per-engine outcome (the reference is the existence
    proof). Netgen failure is BAD_TEST (no input for either engine). A model with
    no resolvable ODE horizon is SKIP (not a parity signal).
    """
    replay = spec.get("reference_archive")
    record_to = spec.get("write_reference_archive")
    # A replay needs neither BNG2.pl nor run_network (issue #805).
    warmup = bc.measure_warmup(None if replay else spec.get("run_network_bin"))

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
    # actions, so it — not the raw BNGL — is the source of both the netgen caps and
    # the simulate horizon. Non-injected jobs preserve the model's own
    # generate_network(...) call when it carries caps such as max_stoich/max_agg; dropping
    # those caps turns deliberately bounded networks into unbounded expansions.
    inject = spec.get("inject")
    horizon_text = inject or bngl_text
    gen_network = bc.injected_gen_network(inject) or (
        None if inject is not None else bc._model_gen_network(bngl_text)
    )
    archived, archived_rn = (None, None)
    record: dict = {}
    if replay:
        try:
            archived, archived_rn = archive_load(replay, spec)
        except Exception as exc:  # a corrupt or foreign file: the archive's fault
            res["status"] = "reference_failed"
            res["exception"] = (
                f"run_network (archive): archive_unreadable: {type(exc).__name__}: {exc}"[:400]
            )
            res["comment"] = "The archived reference could not be read; re-record it."
            q.put(res)
            return
        want = archive_input_identity(spec, bngl_text)
        stale = ["entry"] if archived is None else _stale_fields(archived, want)
        if stale:
            what = (
                "no entry for this job"
                if archived is None
                else f"the job's {', '.join(stale)} changed"
            )
            res["status"] = "reference_failed"
            res["exception"] = f"run_network (archive): archive_stale: {what}"
            res["comment"] = (
                "The archived reference is no reference for this job any more; re-record it."
            )
            q.put(res)
            return
    elif record_to:
        record = archive_input_identity(spec, bngl_text)
        record["reference_version"] = versions.bng_version(spec.get("bng2_pl"))
        record["harness_git_rev"] = git_rev(str(HERE))

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
        if replay:
            # The network BNG2.pl generated when the archive was recorded.
            netgen_sec = float(archived.get("netgen_sec") or 0.0)
            netgen_err = archived.get("netgen_error") or ""
            net_path = None
            if archived.get("net_text") is not None:
                net_path = workdir / "model.net"
                net_path.write_text(archived["net_text"])
        else:
            net_path, netgen_sec, netgen_err = bc.generate_network(
                bngl_text,
                spec["bng2_pl"],
                workdir,
                timeout=spec["sub_timeout"],
                gen_network=gen_network,
                state_prefix=("" if dirty else state_prefix),
            )
            if record_to:
                record["netgen_sec"] = round(netgen_sec, 6)
                if net_path is None:
                    record["netgen_error"] = netgen_err
                    archive_save(record_to, spec, record)
                else:
                    record["net_text"] = Path(net_path).read_text()
        if net_path is None:
            res["status"] = "bad_test"
            res["comment"] = (
                "BNG2.pl could not generate this model's network (failed or timed out) — a "
                "model problem, not bngsim."
            )
            res["exception"] = netgen_err
            res["timing"] = _with_settings(
                {"netgen": {"netgen_sec": round(netgen_sec, 6)}, "warmup": warmup}
            )
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
                if record_to:
                    archive_save(record_to, spec, record)
                res["status"] = "skip"
                res["comment"] = (
                    "No runnable time-course (no simulate with a numeric t_end) — nothing to compare."
                )
                res["timing"] = _with_settings(
                    {"netgen": {"netgen_sec": round(netgen_sec, 6)}, "warmup": warmup}
                )
                q.put(res)
                return
            ode = {**o, "dirty_carryover": False, "steps": None, "rep_index": None}
        dirty = bool(ode.get("dirty_carryover")) and ode.get("steps") is not None
        n_points = int(ode["n_steps"]) + 1
        spec_id = archive_spec_identity(ode, dirty)
        if replay:
            stale = _stale_fields(archived.get("spec"), spec_id)
            if stale:
                res["status"] = "reference_failed"
                res["exception"] = (
                    f"run_network (archive): archive_stale: the run's {', '.join(stale)} changed"
                )
                res["comment"] = (
                    "The archived reference is no reference for this job any more; re-record it."
                )
                q.put(res)
                return

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
                    net_path,
                    ode["steps"],
                    ode["rep_index"],
                    track="ode",
                    atol=ode["atol"],
                    rtol=ode["rtol"],
                    seed=None,
                    poplevel=0.0,
                    codegen=spec.get("codegen", False),
                )
                bn_wall = time.perf_counter() - t0
                bn = (
                    np.asarray(result.time),
                    np.asarray(result.species),
                    list(result.species_names),
                )
                bn_timing = {
                    "load_sec": None,
                    "jac_derive_sec": None,
                    "codegen_sec": None,
                    "integrate_sec": round(bn_wall, 6),
                    "integrate_cold_sec": round(bn_wall, 6),
                    "integrate_n_warm": 0,
                    "multi_segment": True,
                    # config.codegen names the replay mode; backend is what actually
                    # ran the representative segment (the compiled arm checks it).
                    "backend": _info.get("backend"),
                    "work": _info.get("work") or {},
                    "config": {
                        "codegen": "full-protocol replay",
                        "jacobian": "per segment",
                        "linear_solver": "CVODE via bngsim protocol replay",
                        "cached": None,
                    },
                }
            except Exception as exc:
                bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]
            # 3b. legacy: run the model's OWN protocol natively (BNG2.pl carries state across
            #     continue/save/reset), read the representative segment's species.
            if replay:
                rn, rn_exc, rn_timing = (
                    archived_rn,
                    archived.get("exception", ""),
                    archived.get("timing"),
                )
            else:
                try:
                    t0 = time.perf_counter()
                    rn = bc.native_protocol_oracle(
                        horizon_text,
                        spec["bng2_pl"],
                        workdir / "native",
                        track="ode",
                        rep_stmt_idx=ode["rep_stmt_idx"],
                        timeout=spec["sub_timeout"],
                    )
                    rn_wall = time.perf_counter() - t0
                    rn_timing = {
                        "integrate_sec": round(rn_wall, 6),
                        "integrate_cold_sec": round(rn_wall, 6),
                        "integrate_n_warm": 0,
                        "init_cpu_sec": None,
                        "propagation_cpu_sec": None,
                        "total_cpu_sec": None,
                        "total_clock_sec": round(rn_wall, 6),
                        "n_calls": 1,
                        "multi_segment": True,
                        "config": {
                            "codegen": "native BNG2.pl protocol",
                            "jacobian": "CVODE default",
                            "linear_solver": "run_network / BNG2.pl default",
                            "cached": None,
                        },
                    }
                except Exception as exc:
                    rn_exc = f"native BNG2.pl protocol: {type(exc).__name__}: {exc}"[:400]
        else:
            # 3a. BNGsim single-segment (record status; do not short-circuit a generic raise).
            try:
                t0 = time.perf_counter()
                t, v, n, bn_timing = bc.bn_ode_net(
                    net_path,
                    ode["t_start"],
                    ode["t_end"],
                    n_points,
                    ode["rtol"],
                    ode["atol"],
                    codegen=spec.get("codegen", False),
                )
                bn_wall = time.perf_counter() - t0
                bn = (t, v, n)
            except Exception as exc:
                bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]

            # 3b. run_network single-segment (the reference; always run).
            if replay:
                rn, rn_exc, rn_timing = (
                    archived_rn,
                    archived.get("exception", ""),
                    archived.get("timing"),
                )
            else:
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

        if record_to:
            record.update(spec=spec_id, exception=rn_exc, timing=rn_timing)
            archive_save(record_to, spec, record, rn)
        if replay:
            res["reference_version"] = archived.get("reference_version")
            if rn_timing is not None:
                rn_timing = {
                    **rn_timing,
                    "replayed_from_archive": True,
                    "reference_version": archived.get("reference_version"),
                    "recorded_by_harness": archived.get("harness_git_rev"),
                }

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
        res["timing"] = _with_settings(timing)

        # 4. Classify from per-engine status. The plain-English ``comment`` is the
        #    headline a reader sees; the raw engine error stays in ``exception`` as a
        #    secondary detail the matrix renders muted underneath.
        if bn_exc or rn_exc:
            if bn_exc and rn_exc:
                res["status"], res["exception"] = "bad_test", f"{rn_exc} || {bn_exc}"
                res["comment"] = (
                    "Neither engine could run this model — a model/setup problem, not bngsim."
                )
            elif rn_exc:
                res["status"], res["exception"] = "reference_failed", rn_exc
                res["comment"] = (
                    "run_network failed here — no reference to compare against. bngsim ran fine; unscored."
                )
            else:
                res["status"], res["exception"] = "exception", bn_exc
                res["comment"] = (
                    "bngsim errored while run_network succeeded — an actionable bngsim issue."
                )
            q.put(res)
            return

        try:
            comparer = _compare_ode_multiseg if dirty else _compare_ode
            status, value, comment, metric, m_tol, max_abs = comparer(bn, rn, atol=ode["atol"])
            res["status"], res["value"], res["comment"] = status, value, comment
            res["metric"], res["tol"] = metric, m_tol
            # GH #69: a DIFF on a catalogued comparison artifact (integrator
            # phase-wander / staircase knife-edge / reference-side tolerance tail)
            # within its recorded magnitude bound is tagged so the matrix renders it
            # KNOWN ARTIFACT (non-scoring); the honest verdict is unchanged.
            annotate_known_artifact(res, status, bngl_path.stem, max_abs)
            # Chaotic systems: pointwise divergence is expected (not magnitude-bounded),
            # so a separate identity-keyed disposition renders them KNOWN ARTIFACT.
            annotate_chaotic_artifact(res, status, bngl_path.stem)
            # GH #1548: run_network leaving a model's exact solution, which bngsim's
            # run is checked against here, renders KNOWN REF.
            annotate_known_ref(
                res, status, bngl_path.stem, bn, rn, atol=ode["atol"], multi_segment=dirty
            )
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


def _job_timeout_override(job) -> dict | None:
    """The per-job wall-clock cap override, when no global ``--timeout`` is forced.

    An override extends the default cap and never cuts it: one at or below
    ``DEFAULT_ODE_TIMEOUT`` is ignored, as bng_stoch_run.py ignores one at or below
    its own default (#1549).
    """
    for ov in job.overrides:
        if ov.field == "timeout":
            try:
                cap = float(ov.value)
            except (ValueError, TypeError):
                return None
            return {"timeout": cap, "reason": ov.reason} if cap > DEFAULT_ODE_TIMEOUT else None
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
    ap.add_argument(
        "--codegen",
        action="store_true",
        help="Force bngsim's compiled C RHS (Simulator(codegen=True)). Without it every job "
        "is pinned to the ExprTk interpreter (codegen=False), including the >=256-species "
        "models bngsim would compile by default since #825. The compiled path is what every "
        "sensitivity run uses, and #689/#699 existed only there. Needs a C compiler (GH #702).",
    )
    ap.add_argument(
        "--write-reference-archive",
        default="",
        help="Also record each job's generated network and reference side here "
        "(one .npz per job; issue #805).",
    )
    ap.add_argument(
        "--reference-archive",
        default="",
        help="Replay the reference side, and the network BNG2.pl generated, from the archive "
        "recorded here: neither BNG2.pl nor run_network is needed. An entry that no longer "
        "matches its job is never compared (REFERENCE_FAILED, archive_stale).",
    )
    args = ap.parse_args()
    if args.reference_archive and args.write_reference_archive:
        ap.error("--reference-archive replays an archive; --write-reference-archive records one")

    if args.reference_archive:
        bng2_pl, run_network_bin = None, None
    else:
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

    n_tol_ov = 0
    n_timeout_ov = 0
    override_records, override_counts = _override_summary(jobs)
    default_cap = args.timeout if args.timeout is not None else DEFAULT_ODE_TIMEOUT
    default_timeout_source = "cli" if args.timeout is not None else "default"
    global_tol_source = (
        "cli" if (args.rtol != bc.DEFAULT_RTOL or args.atol != bc.DEFAULT_ATOL) else "default"
    )
    specs = []
    for j in jobs:
        tol_ov = _job_tol_override(j)
        if tol_ov:
            rtol, atol = tol_ov["rtol"], tol_ov["atol"]
            tol_source = "job_override"
            n_tol_ov += 1
        else:
            rtol, atol = args.rtol, args.atol
            tol_source = global_tol_source
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
        # Per-job wall cap; the inner BNG2.pl/run_network subprocesses get a slightly
        # smaller cap so they self-terminate before the scheduler kills the worker
        # (avoids orphaned grandchildren).
        sub_timeout = max(10.0, cap - 20.0)
        settings = {
            "timeout_sec": float(cap),
            "subprocess_timeout_sec": float(sub_timeout),
            "timeout_source": timeout_source,
            "timeout_reason": timeout_reason,
            "rtol": float(rtol),
            "atol": float(atol),
            "tol_source": tol_source,
            "non_default": timeout_source != "default" or tol_source != "default",
        }
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
                "settings": settings,
                "bng2_pl": bng2_pl,
                "run_network_bin": run_network_bin,
                "inject": bc.injected_action_block(j.overrides),
                "codegen": bool(args.codegen),
                "reference_archive": args.reference_archive or None,
                "write_reference_archive": args.write_reference_archive or None,
            }
        )

    ver = versions.stamp("bng")
    ver["sundials"] = bc.sundials_version()
    print("=" * 72)
    print("  bngsim vs BNG2.pl/run_network — BNGL ODE parity + timing (bng_parity)")
    print("=" * 72)
    print(f"  jobs: {len(specs)}   workers: {args.workers}   (ODE/deterministic only)")
    if args.codegen:
        print("  bngsim RHS: compiled C (--codegen), not the ExprTk interpreter")
    else:
        print("  bngsim RHS: ExprTk interpreter on every job (codegen=False)")
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
        timing = r.get("timing")
        if not timing and r.get("settings"):
            timing = {"settings": r.get("settings")}
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
                # A replayed row's reference is the archived engine, not a local
                # one: on a replay every row reports the archived version, None
                # where the entry recorded none or the row has no entry.
                versions={**ver, "bng": r.get("reference_version")}
                if args.reference_archive
                else ver,
                comment=comment,
                subclass=r.get("subclass"),
                timing=timing,
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
        # Issue #805: where the reference side came from.
        "reference_source": (
            f"archive:{args.reference_archive}"
            if args.reference_archive
            else "run_network / BNG2.pl (live)"
        ),
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
            "codegen_forced": bool(args.codegen),
            "rtol": args.rtol,
            "atol": args.atol,
            "tol_overridden_jobs": n_tol_ov,
            "timeout_overridden_jobs": n_timeout_ov,
        },
        "runtime": {
            "default_timeout_sec": DEFAULT_ODE_TIMEOUT,
            "global_timeout_sec": default_cap,
            "global_timeout_source": default_timeout_source,
            "subprocess_timeout_rule": "max(10, timeout_sec - 20)",
            "tol_source": global_tol_source,
            "tol_overridden_jobs": n_tol_ov,
            "timeout_overridden_jobs": n_timeout_ov,
        },
        "overrides": {
            "counts_by_field": override_counts,
            "jobs": override_records,
            "note": (
                "Overrides are read from Job.overrides, baked by build_jobs.py from "
                "overrides.py. A timeout override only applies when no global --timeout "
                "is supplied; a tol override applies per-model rtol/atol to both engines. "
                "All non-default runtime settings are echoed in timing.settings and in "
                "this summary so regenerated matrices retain the provenance."
            ),
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
            "applied here. A DIFF keeps its verdict but is tagged non-scoring on a "
            "catalogued model: KNOWN ARTIFACT within a recorded magnitude bound or for "
            "a chaotic system (GH #69), and KNOWN REF where run_network leaves the "
            "model's exact solution and bngsim's run, checked against it, keeps it "
            "(GH #1548). Every other model is adjudicated independently."
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
    if n_timeout_ov:
        print(f"  overrides: {n_timeout_ov} timeout-overridden")
    print(f"  report: {out_path}")
    print("=" * 72)
    from _core.taxonomy import FAILING

    n_fail = sum(counts.get(o.value, 0) for o in FAILING)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
