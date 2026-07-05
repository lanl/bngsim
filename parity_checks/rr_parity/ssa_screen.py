#!/usr/bin/env python3
"""bngsim SSA vs libroadrunner gillespie parity over a broad SBML corpus.

Extends ``run_rr_ssa_trajectory_parity.py`` (the sensitive N=200 check on
the 12 BNG-roundtripped SSA models) to a *breadth* screen across two
corpora:

  * ``roundtrip`` -- the 12 BNG-derived SBMLs in
    ``benchmarks/suites/sbml_roundtrip/`` (their modeler-specified t_end /
    n_steps from ``suite_ssa.json``); and
  * ``biomodels`` -- the SSA-compatible BioModels subset listed in
    ``benchmarks/biomodels_ssa/ssa_candidates.json`` (built by
    ``build_ssa_candidates.py``), SBML resolved from ``$BIOMODELS_SBML_DIR``.

For each model we run ``--n`` replicates per engine at a shared seed
schedule and check that every (t>0, species) cell's mean z-score
``|mu_bn - mu_rr| / sqrt(var_bn/N + var_rr/N)`` stays under ``--mean-z-tol``.

Replicate count (``--n``, default 30) -- two-tier by design, GH #107.1.
This is the *breadth* screen: a low N over the whole 306-model BioModels
corpus, where the goal is to surface gross loader/engine divergences
cheaply. The *sensitive* check is ``run_rr_ssa_trajectory_parity.py`` on
the 12 BNG-roundtripped models at N=200, where small mean shifts matter.
The two jointly satisfy #30 (breadth-at-N=30 + sensitivity-at-N=200);
bump ``--n`` here if you want a deeper pass over the BioModels set.

Low-count z-gate floor (``--min-mean-count``, default 2.0, GH #107.3).
The mean z-test blows up when both engines sit at a near-deterministic
sub-particle value: a species at <=1 molecule has ~zero replicate
variance, so a 1-molecule MC difference reads as a many-sigma "fail"
that is noise, not a real divergence. We therefore skip a cell when
*both* engines' means convert to <= ``--min-mean-count`` molecules
(amount = concentration x compartment volume). The guard is deliberately
two-sided (``max(bn_count, rr_count) <= floor``): a cell where one engine
holds a meaningful population while the other is near zero (e.g. an
RR-side population artifact, sys-bio/roadrunner#1320) stays tested, so the
floor reads green on noise without masking a real engine disagreement.
Skipped cells are counted (``n_skipped_lowcount``), never silently
dropped. NB the floor clears *MC/sub-particle noise* only -- a genuine
divergence at meaningful counts (e.g. a runtime sign-indefinite model,
GH #109, where one species holds tens of molecules) still fails, by design.

Simulation semantics for BioModels SBML: there are none in core SBML, so
we impose an arbitrary-but-shared horizon (``--t-end`` / ``--n-steps``).
The point is cross-engine *agreement* on an identical spec -- whatever we
pick, both engines see the same thing, so a divergence flags a real
loader/engine bug, not a horizon choice. Vacuous flat trajectories
(horizon far off the model's natural timescale) still exercise both
loaders and pass cheaply.

Runtime is bounded the only way it can be for SSA: a per-model wall-clock
cap, enforced by running each model in its own process and killing it on
overrun (``too_slow``). The model-time horizon does NOT bound cost --
event count does -- so a population-heavy model is pruned rather than
allowed to run for minutes. ``--jobs`` runs models concurrently; effort
tiers (``--effort``) and ``--max-models`` trim the corpus for quick runs.

``too_slow`` is a coverage gap, not a pass (GH #107.2): a model killed at
the wall cap is a population/event-count bomb that *exact* SSA cannot
churn through in budget, and it is never compared. These are out of scope
for exact SSA (a tau-leaping / continuum cross-check would be the way to
cover them, deliberately not attempted here). The dropped set is logged
explicitly -- in stdout and as ``coverage.too_slow`` in the JSON -- so it
reads as uncovered rather than silently green.

DIFF attribution (GH #107 follow-up). A cell that survives the low-count
floor and still trips the z-gate is a *real* divergence -- but "real" is
not "bngsim's fault". Each surviving fail is attributed against a third
oracle: each engine's OWN deterministic ODE. Over the failing cells we ask
(molecule-aware, so neither the sub-particle nor the stiff-floor regime
produces a false signal) whether each engine's SSA mean tracks its own ODE
and whether the two ODEs agree, then label:

  * ``diff_not_bngsim`` -- bngsim SSA tracks bngsim ODE, RR SSA does not,
    ODEs agree => the DIFF is RR-gillespie-side, not a bngsim defect.
  * ``rr_known``        -- a specifically-filed RR issue (``KNOWN_RR_ISSUES``),
    a more precise label that takes precedence over ``diff_not_bngsim``.
  * ``diff_ode_level``  -- the two ODEs themselves disagree => loader/ODE-level,
    route to the ODE triage, not the SSA z-gate.
  * ``fail``            -- bngsim SSA deviates from its own ODE (bngsim-suspect),
    or an unexplained residual. THIS is the only status that means "look at
    bngsim".

Only ``fail`` / ``diff_ode_level`` / ``error`` fail the exit code; the
explained-not-bngsim statuses (``diff_not_bngsim`` / ``rr_known``) are green.
``--self-z-tol`` tunes the per-engine SSA-vs-own-ODE z tolerance (default 7.30,
the #30 discriminator). The attribution runs each engine's ODE only for the
handful of fails, so its cost is negligible.

Usage:
    cd bngsim && uv run python parity_checks/rr_parity/ssa_screen.py
        [--corpus all|roundtrip|biomodels] [--effort low|medium|high]
        [--n 100] [--jobs 4] [--timeout 40] [--max-models N]
        [--t-end 100] [--n-steps 10] [--mean-z-tol 9.13] [--out PATH]

Output:
    runs/ssa_screen.json  (gitignored; --out to override)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so `import _rr_common` resolves
import _rr_common as rc  # noqa: E402
from _core import versions  # noqa: E402
from _rr_common import EFFORT_LEVELS, effort_allows  # noqa: E402

BNGSIM_DIR = HERE.parents[1]  # bngsim/

# The roundtrip corpus (BNGL-derived SBMLs) and its suite manifest still live
# under benchmarks/; resolved via env so this suite carries no hard benchmarks
# import (and degrades gracefully if that tree moves). The biomodels corpus is
# vendored locally: ssa_candidates.json sits beside this file.
ROUNDTRIP_DIR = Path(
    os.environ.get("SBML_ROUNDTRIP_DIR", BNGSIM_DIR / "benchmarks" / "suites" / "sbml_roundtrip")
)
SUITE_SSA = Path(
    os.environ.get("SUITE_SSA_JSON", BNGSIM_DIR / "benchmarks" / "_dev" / "suite_ssa.json")
)
BIOMODELS_MANIFEST = Path(os.environ.get("SSA_CANDIDATES_JSON", HERE / "ssa_candidates.json"))
BIOMODELS_SBML_DIR = Path(
    os.environ.get(
        "BIOMODELS_SBML_DIR",
        os.path.expanduser("~/Code/ssys/biomodels_batch/data/sbml_downloads"),
    )
)
# Fresh-run output lands in the gitignored runs/ dir; the committed baseline the
# regression diffs against is authored separately (GH #108 follow-up steps).
DEFAULT_OUT = HERE / "runs" / "ssa_screen.json"
# The same run, re-expressed in the shared _core report schema (one JobResult per
# model). The GH #108 regression diffs a committed baseline of THIS against a
# fresh run; the native ssa_screen.json keeps the richer coverage detail.
DEFAULT_CORE_OUT = HERE / "runs" / "ssa_report.json"
# Per-model activity-probe sidecars (gitignored). A worker writes its stochastic-
# activity estimate here BEFORE the (possibly long) ensemble, so a model the parent
# kills on wall-clock overrun — exactly the most active — still reports an
# events-per-unit-time figure the matrix can show and sort on (recovered in
# ``_schedule`` on the too_slow branch). Cleared at the start of each run.
PROBE_DIR = HERE / "runs" / "_probe"


def _activity_probe(
    xml: str,
    t_end: float,
    seed: int,
    *,
    wall: float = 0.4,
    shrink: float = 10.0,
    max_attempts: int = 9,
) -> tuple[float | None, float, bool]:
    """Probe a model's stochastic ACTIVITY and its ACHIEVABLE HORIZON.

    SSA's firing rate is measurable from a short window: simulate from ``t=0`` under
    a tight wall cap and, if the engine raises ``SimulationTimeout``, shrink the
    horizon (``×1/shrink``) and retry — a hyper-active model fires enough events in
    a tiny window to finish fast. Returns ``(rate, horizon, full_ok)`` where ``rate``
    = ``solver_stats["n_steps"] / (simulated time reached)`` (reaction events per
    unit time, ``None`` if even the smallest window can't complete or the model
    can't load), ``horizon`` = the largest ``[0,H]`` one replicate finished under the
    wall cap, and ``full_ok`` = whether that was the full requested ``t_end``. The
    worker uses ``full_ok`` to decide full-horizon vs partial-horizon adjudication,
    and ``rate`` to fill in a timed-out model's activity. Best-effort and
    side-effect-free on the parity verdict — separately seeded clones, never touches
    the ensemble's RNG, so trajectories/z-scores are unchanged.
    """
    import warnings as _w

    import bngsim

    try:
        model = bngsim.Model.from_sbml(xml)
    except Exception:
        return None, float(t_end), False
    th = float(t_end)
    for _ in range(max_attempts):
        try:
            m = model.clone()
            m.reset()
            sim = bngsim.Simulator(m, method="ssa")
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                r = sim.run(t_span=(0.0, th), n_points=2, seed=seed, timeout=wall)
            reached = float(np.asarray(r.time)[-1])
            nst = int((r.solver_stats or {}).get("n_steps", 0) or 0)
            if reached > 0:
                return round(nst / reached, 4), th, (th >= float(t_end))
            th /= shrink
        except bngsim.SimulationTimeout:
            th /= shrink
        except Exception:
            return None, float(t_end), False
    return None, th, False


def _load_suite(path: Path) -> list[dict]:
    """Load a suite manifest's model list (``{"models": [...]}``)."""
    return json.loads(Path(path).read_text())["models"]


def _machine_info() -> dict:
    """Reproducibility stamp for the run report (engine + platform + git)."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "versions": versions.stamp("roadrunner"),
        "git_commit": versions.git_rev(str(BNGSIM_DIR)),
    }


def _json_default(obj):
    """JSON encoder for numpy scalars/arrays and Paths."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _save(payload: dict, out: Path) -> Path:
    """Write the run report as pretty JSON, creating ``runs/`` if needed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    print(f"\nResults saved to {out}")
    return out


# Heavy roundtrip models: huge networks / populations. Tagged "high" so a
# low/medium effort run skips them (mirrors the sensitive harness's
# LARGE_MODELS exclusion).
ROUNDTRIP_LARGE = {
    "egfr_net",
    "fceri_gamma",
    "multisite_phos",
    "prion_aggregation",
    "erk_activation",
}

DEFAULT_N = 100
DEFAULT_JOBS = 4
DEFAULT_TIMEOUT = 120.0
DEFAULT_T_END = 100.0
DEFAULT_N_STEPS = 10
DEFAULT_SEED_BASE = 2000
# z-gate sensitivity scales with sqrt(N): the standard error of each ensemble mean
# is std/sqrt(N), so for a FIXED true cross-engine difference the z-score grows like
# sqrt(N). To keep the gate detecting the same minimum *meaningful* (effect-size)
# difference as the original 30-replicate calibration — rather than mechanically
# flagging ever-tinier differences as N rises — the tolerances are scaled by
# sqrt(DEFAULT_N / 30): mean 5.0 -> 9.13 and self 4.0 -> 7.30 at N=100. If you
# change DEFAULT_N, rescale these by sqrt(new_N / 30) (GH #190).
DEFAULT_MEAN_Z_TOL = 9.13
# Low-count z-gate floor (GH #107.3): a (t, species) cell is z-tested only
# if at least one engine's mean exceeds this many molecules (amount =
# concentration x compartment volume). At a handful of molecules the
# replicate variance is ~zero and a 1-molecule MC difference reads as a
# spurious many-sigma fail. 2.0 ("a couple of molecules") clears the issue's
# "count <= 1 or near zero" regime with margin -- a true-count-1 species'
# sample mean fluctuates slightly above 1 (e.g. 1.03 over 30 reps), so a 1.0
# floor would leave it tested. Kept low so a genuine low-count *two-sided*
# divergence still trips; the guard is two-sided (skip only when BOTH engines
# are at/below the floor) so it never masks a one-sided divergence at any floor.
DEFAULT_MIN_MEAN_COUNT = 2.0
MAX_FAILURES_RECORDED = 20

# DIFF attribution (GH #107 follow-up). A cell that survives the low-count
# floor and still trips the cross-engine z-gate is a *real* divergence -- but
# "real" does not mean "bngsim's fault". We attribute each surviving fail
# against a third oracle: each engine's OWN deterministic ODE. For the failing
# cells we ask, per engine, whether its SSA mean tracks its own ODE within
# SELF_Z_TOL standard errors (the #30 closeout's discriminator, 4.0 at N=30),
# and whether the two ODEs agree to ODE_REL_TOL (a loader/semantics check).
# The engine whose SSA does NOT track its own ODE is the outlier; if that is
# RR and the ODEs agree, the DIFF is RR-gillespie-side, not a bngsim defect.
# Scaled with sqrt(N) like DEFAULT_MEAN_Z_TOL above: 4.0 * sqrt(100/30) = 7.30.
DEFAULT_SELF_Z_TOL = 7.30
ODE_REL_TOL = 0.05

# Statuses that mean "explained, NOT a bngsim defect" -- they do not fail the
# screen's exit code. ``rr_known`` (a specifically-filed RR issue) and
# ``diff_not_bngsim`` (oracle-attributed RR-side) are both green; plain ``fail``
# (bngsim SSA deviates from its own ODE) and ``diff_ode_level`` (ODE-level
# divergence, route to the loader triage) stay red.
EXPLAINED_NOT_BNGSIM = ("rr_known", "diff_not_bngsim")

# Residual screen fails attributable to a filed roadrunner gillespie bug, not a
# bngsim defect. These are *valid* models (kept in the corpus) where bngsim's
# SSA mean tracks its own ODE while RR's does not; the cross-engine z-gate
# therefore trips on RR's side. ``_annotate_known_rr_issues`` reclassifies them
# from ``fail`` to ``rr_known`` so the screen's fail count reflects only
# unexplained divergences. The corpus-quality classes that RR also mishandles
# (negative-population / sign-indefinite, sys-bio/roadrunner#1320) are dropped
# upstream by build_ssa_candidates.py's structural gates, so they never reach
# the screen and are not listed here. See dev/notes/SBML_VS_ROADRUNNER.md, GH #30.
KNOWN_RR_ISSUES = {
    "BIOMD0000001026": {
        "issue": "sys-bio/roadrunner#1318",
        "note": "Mrbc: RR gillespie does not re-evaluate the time-dependent assignmentRule "
        "feeding the synthesis rate; bngsim SSA tracks its own ODE (affine system: SSA "
        "mean must equal ODE).",
    },
    "BIOMD0000001040": {
        "issue": "sys-bio/roadrunner#1318",
        "note": "Mrbc: same time-dependent assignmentRule lag as BIOMD0000001026 "
        "(the minimal reproducer on #1318).",
    },
}


def _annotate_known_rr_issues(cases: list[dict]) -> int:
    """Reclassify ``fail`` cases attributable to a known roadrunner bug
    (``KNOWN_RR_ISSUES``) to status ``rr_known``, tagging each with the issue.
    Mutates in place; returns the number reclassified. Idempotent."""
    n = 0
    for c in cases:
        info = KNOWN_RR_ISSUES.get(c.get("name"))
        if info and c.get("status") == "fail":
            c["status"] = "rr_known"
            c["known_rr_issue"] = info
            n += 1
    return n


def _species_volumes(xml_path: str) -> dict[str, float]:
    """Map each SBML species id to its compartment size (volume), for the
    concentration->molecule-count conversion the low-count z-gate floor uses.

    Both engines report concentration (amount / volume), so a cell's mean is a
    concentration; ``mean * V`` is the molecule count. A compartment with no
    declared (or non-positive) size defaults to 1.0 -- the same convention
    ``build_ssa_candidates._initial_count`` uses for the sub-particle gate, so
    the screen's floor and the corpus filter agree on what "1 molecule" means.
    """
    import libsbml

    model = libsbml.readSBML(xml_path).getModel()
    if model is None:
        return {}
    comp_size: dict[str, float] = {}
    for i in range(model.getNumCompartments()):
        c = model.getCompartment(i)
        v = c.getSize() if c.isSetSize() else 1.0
        comp_size[c.getId()] = v if v and v > 0 else 1.0
    return {
        model.getSpecies(i).getId(): comp_size.get(model.getSpecies(i).getCompartment(), 1.0)
        for i in range(model.getNumSpecies())
    }


def _compare(
    bn_times,
    bn_arr,
    bn_names,
    rr_times,
    rr_arr,
    rr_names,
    mean_z_tol,
    max_failures,
    species_volumes=None,
    min_mean_count=DEFAULT_MIN_MEAN_COUNT,
) -> dict:
    """Per-(time>0, common-species) mean z-score gate. Pure, picklable output.

    A cell is z-tested only if at least one engine's mean exceeds
    ``min_mean_count`` molecules (amount = concentration * compartment volume,
    from ``species_volumes``; missing -> volume 1.0). When both engines sit
    at/below that floor the cell is in the sub-particle / near-zero regime where
    the z-test is dominated by discreteness noise (GH #107.3): it is skipped and
    tallied in ``n_skipped_lowcount`` rather than counted as a comparison. The
    two-sided floor (``max(bn_count, rr_count) <= floor``) never masks a
    one-sided divergence -- if one engine holds a real population the cell is
    still tested.
    """
    if not np.allclose(bn_times, rr_times, rtol=0, atol=1e-9):
        return {"status": "error", "error": "time grid mismatch"}

    bn_map = {n: i for i, n in enumerate(bn_names)}
    rr_map = {n: i for i, n in enumerate(rr_names)}
    common = sorted(set(bn_map) & set(rr_map))
    if not common:
        return {
            "status": "error",
            "error": f"no common species: bn={bn_names[:4]} rr={rr_names[:4]}",
        }
    bn_idx = [bn_map[n] for n in common]
    rr_idx = [rr_map[n] for n in common]
    vols = species_volumes or {}

    n_time = bn_arr.shape[1]
    n = bn_arr.shape[0]
    failures: list[dict] = []
    max_z = 0.0
    n_compared = 0
    n_skipped_lowcount = 0
    for ti in range(1, n_time):
        for k, sp in enumerate(common):
            bn_col = bn_arr[:, ti, bn_idx[k]]
            rr_col = rr_arr[:, ti, rr_idx[k]]
            bn_mu, rr_mu = float(bn_col.mean()), float(rr_col.mean())
            vol = float(vols.get(sp, 1.0))
            bn_cnt, rr_cnt = abs(bn_mu) * vol, abs(rr_mu) * vol
            if max(bn_cnt, rr_cnt) <= min_mean_count:
                n_skipped_lowcount += 1
                continue
            bn_var, rr_var = float(bn_col.var(ddof=1)), float(rr_col.var(ddof=1))
            se = float(np.sqrt(max(bn_var / n + rr_var / n, 1e-18)))
            z = 0.0 if se <= 1e-9 and bn_mu == rr_mu else abs(bn_mu - rr_mu) / max(se, 1e-9)
            n_compared += 1
            if z > max_z:
                max_z = z
            if z > mean_z_tol:
                failures.append(
                    {
                        "t": float(bn_times[ti]),
                        "species": sp,
                        "mean_z": float(z),
                        "bn_mu": bn_mu,
                        "rr_mu": rr_mu,
                        "bn_count": bn_cnt,
                        "rr_count": rr_cnt,
                    }
                )
    failures.sort(key=lambda f: -f["mean_z"])
    return {
        "status": "pass" if not failures else "fail",
        "n_species_compared": len(common),
        "n_compared": n_compared,
        "n_skipped_lowcount": n_skipped_lowcount,
        "min_mean_count": float(min_mean_count),
        "max_mean_z": float(max_z),
        "n_z_failures": len(failures),
        "z_failures": failures[:max_failures],
    }


def _cell_tracks_own_ode(ssa_mu, ode_val, ssa_se, vol, self_z_tol, min_mean_count):
    """True if an engine's SSA mean is consistent with its OWN ODE on a cell.

    Two ways to be consistent, OR-ed so neither stiff-floor nor sub-particle
    cells produce false deviations:
      * within ``self_z_tol`` standard errors of the ODE (the #30 z-discriminator,
        valid where there is replicate variance), or
      * within ``min_mean_count`` *molecules* of the ODE (``|mu-ode| * V``) -- the
        regime where exact SSA legitimately floors a <1-molecule ODE value to 0
        (bngsim SSA=0 vs bngsim ODE=0.1 molecule is NOT a deviation), or a
        deterministic cell (SE=0) differs only at the sub-particle level.
    """
    diff = abs(ssa_mu - ode_val)
    return diff <= self_z_tol * ssa_se or diff * vol <= min_mean_count


def _odes_agree_on_cell(bode, rode, vol, ode_rel_tol, min_mean_count):
    """True unless bngsim-ODE and RR-ODE genuinely diverge on a cell -- which
    requires BOTH an absolute gap above the molecule floor AND a relative gap
    above ``ode_rel_tol``. The two-condition gate kills the stiff-floor artifact
    (two ~1e-11 numbers off by a factor are 'relatively' far but absolutely nil)
    that the ode_partition investigation flagged as a false ODE_LEVEL."""
    abs_gap = abs(bode - rode)
    rel_gap = abs_gap / max(abs(bode), abs(rode), 1e-300)
    return abs_gap * vol <= min_mean_count or rel_gap <= ode_rel_tol


def _classify_diff(bn_tracks_own_ode, rr_tracks_own_ode, odes_agree):
    """Attribute a surviving cross-engine fail from three booleans, each
    aggregated over the failing cells. Pure (no engines) -> unit-testable.
    Returns ``(status, reason)``.

    * ODEs disagree            -> ``diff_ode_level`` (loader/ODE-level, not SSA).
    * bngsim tracks, RR doesn't -> ``diff_not_bngsim`` (RR-gillespie-side).
    * bngsim does NOT track     -> ``fail`` (bngsim-suspect).
    * both track yet differ     -> ``fail`` (unexplained).
    """
    if not odes_agree:
        return (
            "diff_ode_level",
            "bngsim-ODE and RR-ODE disagree on a failing cell (beyond the molecule floor "
            "AND ODE_REL_TOL); the divergence is ODE/loader-level, not an SSA-method "
            "difference -- route to the ODE triage, not the SSA z-gate.",
        )
    if bn_tracks_own_ode and not rr_tracks_own_ode:
        return (
            "diff_not_bngsim",
            "bngsim SSA mean tracks bngsim ODE on every failing cell; RR SSA does not track "
            "RR ODE; both ODEs agree. DIFF is RR-gillespie-side, not a bngsim defect.",
        )
    if not bn_tracks_own_ode:
        return (
            "fail",
            "bngsim SSA mean deviates from bngsim ODE beyond MC error and the molecule floor "
            "-- UNEXPLAINED, bngsim-suspect; investigate.",
        )
    return (
        "fail",
        "both SSA means track their own ODEs yet a cross-engine DIFF persists -- unexplained.",
    )


def _sign_indefinite_annotation(bn_ssa_diag: dict) -> dict | None:
    """Runtime sign-indefinite annotation from bngsim's GH #110 SSA boundary
    diagnostics (GH #109). ``None`` when no reaction fired in reverse.

    A non-None result means a rate law evaluated **negative** somewhere on the
    trajectory and bngsim ran that irreversible channel backward with propensity
    ``|rate|`` (mean-faithful to the ODE -- the SSA mean tracks bngsim's own
    CVODE). This is the *runtime* multi-state signal the static t0 corpus gate
    (``build_ssa_candidates._negative_initial_propensity``) cannot see, so it
    surfaces the models that slip past the t0/manual gates (e.g. BIOMD863).

    It is **context, not a verdict**: a model can be sign-indefinite *and* have a
    real bngsim issue, so conflating the two would mask the latter. The caller
    attaches this without touching ``status`` -- the cross-engine DIFF is still
    attributed on its own merits by ``_classify_diff`` (these land as
    ``diff_not_bngsim`` because bngsim's mean-faithful reverse-firing tracks its
    own ODE while RR's gillespie does not).
    """
    n = int((bn_ssa_diag or {}).get("n_reverse_fires", 0))
    if n <= 0:
        return None
    return {
        "reverse_fires": n,
        "reaction": (bn_ssa_diag.get("first_reverse_reaction") or "(unknown)"),
        "note": (
            "bngsim fired this irreversible reaction in reverse because its rate "
            "law evaluated negative (GH #110, mean-faithful); the model is runtime "
            "sign-indefinite (GH #109). Sign-indefinite models are normally dropped "
            "by build_ssa_candidates.py's t0/manual gates; this one survived them. "
            "Annotation only -- the cross-engine verdict is set independently."
        ),
    }


def _attribute_fail(
    xml,
    t_end,
    n_steps,
    bn_arr,
    bn_names,
    rr_arr,
    rr_names,
    species_volumes,
    mean_z_tol,
    min_mean_count,
    self_z_tol,
) -> dict:
    """Run each engine's deterministic ODE and attribute a fail against it.

    For every cell that survived the low-count floor and tripped the z-gate,
    test (molecule-aware) whether each engine's SSA mean tracks its OWN ODE and
    whether the two ODEs agree. Returns the chosen ``status``, a human-readable
    ``reason``, and supporting numbers. Raises on ODE error (the caller leaves
    the model as an unattributed ``fail``)."""
    import bngsim
    import numpy as np
    import roadrunner

    m = bngsim.Model.from_sbml(xml)
    bo = bngsim.Simulator(m, method="ode").run(t_span=(0, t_end), n_points=n_steps + 1, timeout=60)
    bo_arr, bo_names = np.asarray(bo.species), list(bo.species_names)

    rr = roadrunner.RoadRunner(xml)
    rr.setIntegrator("cvode")
    sn = list(rr.model.getFloatingSpeciesIds())
    rr.timeCourseSelections = ["time"] + ["[" + s + "]" for s in sn]
    ro_arr, ro_names = np.asarray(rr.simulate(0, t_end, n_steps + 1))[:, 1:], sn

    bn_map = {n: i for i, n in enumerate(bn_names)}
    rr_map = {n: i for i, n in enumerate(rr_names)}
    bo_map = {n: i for i, n in enumerate(bo_names)}
    ro_map = {n: i for i, n in enumerate(ro_names)}
    common = sorted(set(bn_map) & set(rr_map) & set(bo_map) & set(ro_map))
    vols = species_volumes or {}
    n_rep = bn_arr.shape[0]

    bn_all_track = rr_all_track = odes_agree = True
    n_fail_cells = 0
    worst = None  # the RR-deviating cell with the largest molecule gap, for context
    for ti in range(1, bn_arr.shape[1]):
        for sp in common:
            bn_col, rr_col = bn_arr[:, ti, bn_map[sp]], rr_arr[:, ti, rr_map[sp]]
            bmu, rmu = float(bn_col.mean()), float(rr_col.mean())
            vol = float(vols.get(sp, 1.0))
            if max(abs(bmu) * vol, abs(rmu) * vol) <= min_mean_count:
                continue
            bse = float(bn_col.std(ddof=1) / np.sqrt(n_rep))
            rse = float(rr_col.std(ddof=1) / np.sqrt(n_rep))
            se = float(np.sqrt(max(bse**2 + rse**2, 1e-18)))
            if abs(bmu - rmu) / max(se, 1e-9) <= mean_z_tol:
                continue
            n_fail_cells += 1
            bode, rode = float(bo_arr[ti, bo_map[sp]]), float(ro_arr[ti, ro_map[sp]])
            bn_tracks = _cell_tracks_own_ode(bmu, bode, bse, vol, self_z_tol, min_mean_count)
            rr_tracks = _cell_tracks_own_ode(rmu, rode, rse, vol, self_z_tol, min_mean_count)
            bn_all_track = bn_all_track and bn_tracks
            rr_all_track = rr_all_track and rr_tracks
            odes_agree = odes_agree and _odes_agree_on_cell(
                bode, rode, vol, ODE_REL_TOL, min_mean_count
            )
            rr_gap = abs(rmu - rode) * vol
            if not rr_tracks and (worst is None or rr_gap > worst["rr_ode_gap_molecules"]):
                worst = {
                    "species": sp,
                    "t": float(ti * (t_end / n_steps)),
                    "bn_ssa": bmu,
                    "bn_ode": bode,
                    "rr_ssa": rmu,
                    "rr_ode": rode,
                    "vol": vol,
                    "rr_ode_gap_molecules": round(rr_gap, 3),
                }

    if not n_fail_cells:
        # Every failing cell was ODE-unmappable (e.g. RR drops the species from
        # its floating set); cannot attribute -- leave it red.
        return {"status": "fail", "reason": "no ODE-mappable failing cell to attribute"}

    status, reason = _classify_diff(bn_all_track, rr_all_track, odes_agree)
    return {
        "status": status,
        "reason": reason,
        "bn_tracks_own_ode": bn_all_track,
        "rr_tracks_own_ode": rr_all_track,
        "odes_agree": odes_agree,
        "n_fail_cells": n_fail_cells,
        "worst_cell": worst,
    }


# Per-replicate wall cap for the adjudicator (seconds). A trajectory that does not
# finish within this is a blowup/too-slow seed; the full ensemble then falls back to
# the largest horizon at which EVERY replicate finishes (partial-horizon
# adjudication). Sized at the per-replicate budget — the per-model wall cap divided
# by (N replicates × 2 engines), ≈ 40 / 60 — so a model whose full-horizon run fits
# the wall budget stays a FULL verdict and only genuinely-slower ones fall to
# partial. (Too small a cap demotes completable models to provisional partials.)
REP_WALL_CAP = 0.6
# Smallest fraction of the requested horizon the shrink will try before giving up.
# Pushed low (GH #190 follow-up) so event-count-bomb models — which fire millions
# of reactions in <1% of the requested horizon and so never finish a 1% window —
# still get a partial comparison over the largest window that DOES finish. The
# informative-window guard is no longer this fraction (a tiny window of a
# high-activity model is rich, not vacuous); it is the per-window signal check in
# _partial_adjudicate (n_compared > 0), so a genuinely vacuous finishable window
# (a delayed-onset model that fires ~nothing before it explodes) still falls to a
# gray coverage gap rather than a meaningless "agreement".
PARTIAL_MIN_FRACTION = 1e-5
# Largest multiple of the requested horizon the EXPANSION will try (GH #190). The
# symmetric counterpart of PARTIAL_MIN_FRACTION: when a model is too QUIET at the
# requested horizon (vacuous — every cell below the low-count floor), grow the
# window ×10 up to this cap so a slow / delayed-onset model can accumulate signal.
# Count-limited models (0/1-copy switch species, static systems) never cross the
# floor at any horizon, so the per-window signal guard (n_compared>0) leaves them
# a gray coverage gap; a model that turns into an event-count bomb at a larger
# horizon trips the per-replicate wall cap and is likewise left vacuous.
EXTENDED_MAX_MULTIPLE = 1000.0


def _partial_adjudicate(
    xml: str,
    t_end: float,
    n_steps: int,
    n: int,
    seed_base: int,
    mean_z_tol: float,
    max_failures: int,
    min_mean_count: float,
    rep_timeout: float,
) -> dict:
    """Adjudicate an SSA-slow model over the largest achievable ``[0, H]`` window.

    Find the largest ``H`` (shrinking ``×1/10`` from ``t_end`` past each blowup) at
    which EVERY replicate of BOTH engines finishes under ``rep_timeout`` — a model
    can be slow uniformly OR only on some seeds (a runtime sign-indefinite rate
    explosion), and this handles both. bngsim runs first (bounded → raises
    SimulationTimeout on a blowup, caught here to shrink); RoadRunner is run only
    once bngsim completes at ``H`` (so at a window short enough for bngsim it is fast
    too). Returns the :func:`_compare` result + a ``timing`` block + the ``horizon``
    used; the caller stamps it ``partial`` (a provisional / yellow verdict, not a
    full-horizon judgment — and no ODE attribution, which would not finish either).
    Raises ``SimulationTimeout`` if even ``t_end × PARTIAL_MIN_FRACTION`` blows up
    (genuinely intractable)."""
    import bngsim

    h = float(t_end)
    floor = float(t_end) * PARTIAL_MIN_FRACTION
    vacuous_h = None  # largest finishable window that had no above-floor signal
    while h >= floor:
        try:
            bn_times, bn_arr, bn_names, _diag, bn_timing = rc.bn_ssa_replicates(
                xml, 0.0, h, n_steps + 1, n, seed_base, rep_timeout=rep_timeout
            )
            try:
                rr_times, rr_arr, rr_names, rr_timing = rc.rr_ssa(
                    xml, 0.0, h, n_steps + 1, n, seed_base
                )
            except bngsim.SimulationTimeout:
                raise
            except Exception as e:
                # RoadRunner refused/failed (e.g. its gillespie cannot do rate rules)
                # — re-raise RR-side so the worker maps it to REFERENCE_FAILED (a
                # capability gap), not a red bngsim EXCEPTION.
                raise RuntimeError(f"roadrunner: {type(e).__name__}: {e}") from e
            try:
                species_volumes = _species_volumes(xml)
            except Exception:
                species_volumes = {}
            res = _compare(
                bn_times,
                bn_arr,
                bn_names,
                rr_times,
                rr_arr,
                rr_names,
                mean_z_tol,
                max_failures,
                species_volumes=species_volumes,
                min_mean_count=min_mean_count,
            )
            # Signal guard (GH #190 follow-up): the largest finishable window is
            # the best available (most events of any window that completes). If
            # even it cleared no species above the low-count floor (n_compared==0),
            # the model fires ~nothing before it explodes (delayed onset) — a
            # "partial" verdict would be a vacuous "both engines agree on a flat
            # window". Smaller windows are quieter still, so stop and fall to a
            # gray coverage gap (too_slow), the honest classification: no window
            # both finishes AND has comparable signal.
            if res.get("status") == "pass" and res.get("n_compared", 0) == 0:
                vacuous_h = h
                break
            res["timing"] = {"bngsim": bn_timing, "roadrunner": rr_timing}
            res["horizon"] = h
            return res
        except bngsim.SimulationTimeout:
            h /= 10.0
    if vacuous_h is not None:
        raise bngsim.SimulationTimeout(
            f"partial adjudication: largest finishable window [0,{vacuous_h:g}] has no "
            f"above-low-count-floor signal (n_compared=0) — too active to reach a comparable horizon"
        )
    raise bngsim.SimulationTimeout(
        f"partial adjudication: a replicate never finishes even by t={floor:g}"
    )


def _extended_adjudicate(
    xml: str,
    t_end: float,
    n_steps: int,
    n: int,
    seed_base: int,
    mean_z_tol: float,
    max_failures: int,
    min_mean_count: float,
    rep_timeout: float,
) -> dict | None:
    """Adjudicate a too-QUIET model over a LARGER horizon (symmetric to
    :func:`_partial_adjudicate`).

    The requested horizon was vacuous — every (t>0, species) cell fell below the
    low-count z-gate floor, so nothing was comparable. Grow the window ×10 up to
    ``EXTENDED_MAX_MULTIPLE`` and return the :func:`_compare` result at the first
    horizon that yields a CONFIDENT comparison — above-floor signal
    (``n_compared > 0``) AND the ensembles agree (``status == "pass"``) — plus a
    ``timing`` block and the ``horizon`` used; the caller stamps it ``extended``.
    Returns ``None`` when no horizon within the cap gives a confident comparison:
    a count-limited model (0/1-copy switch species, static system) never crosses
    the floor at any horizon; a model that turns into an event-count bomb at a
    larger horizon trips ``rep_timeout``; and a model whose extended window
    *diverges* is rejected too (the extended regime is beyond the requested
    dynamics and prone to explosive/heavy-tailed behavior where the mean-z is
    unreliable — not a confident verdict). All of those stay a gray vacuous gap."""
    import bngsim  # for SimulationTimeout in the ×10-ladder loop below

    h = float(t_end)
    cap = float(t_end) * EXTENDED_MAX_MULTIPLE
    while h < cap:
        h *= 10.0
        try:
            bn_times, bn_arr, bn_names, _diag, bn_timing = rc.bn_ssa_replicates(
                xml, 0.0, h, n_steps + 1, n, seed_base, rep_timeout=rep_timeout
            )
            rr_times, rr_arr, rr_names, rr_timing = rc.rr_ssa(
                xml, 0.0, h, n_steps + 1, n, seed_base
            )
        except bngsim.SimulationTimeout:
            # Grown into intractable (event-count-bomb) territory — stop; the model
            # stays a vacuous gray gap (no window both quiet-enough to finish AND
            # active-enough to clear the floor was found on the ×10 ladder).
            return None
        except Exception:
            return None
        try:
            species_volumes = _species_volumes(xml)
        except Exception:
            species_volumes = {}
        res = _compare(
            bn_times,
            bn_arr,
            bn_names,
            rr_times,
            rr_arr,
            rr_names,
            mean_z_tol,
            max_failures,
            species_volumes=species_volumes,
            min_mean_count=min_mean_count,
        )
        if int(res.get("n_compared", 0) or 0) > 0 and res.get("status") == "pass":
            # Only a CONFIDENT extended comparison (the ensembles AGREE) counts as a
            # recovery. A divergence at an extended horizon is NOT reliable signal:
            # growing the window past the requested dynamics pushes these slow models
            # into explosive / heavy-tailed regimes where the 30-replicate ensemble
            # mean (and hence the mean-z) is dominated by a few rare large
            # trajectories — a z of 1e11 there is a statistic breaking down, not a
            # bngsim defect. So we do not surface it; keep expanding (a larger window
            # may clear more species and agree), and if none within the cap agree the
            # model stays an honest vacuous gray gap.
            res["timing"] = {"bngsim": bn_timing, "roadrunner": rr_timing}
            res["horizon"] = h
            return res
    return None


def _has_time_event(xml: str) -> bool:
    """True if the model has an event whose TRIGGER references the SBML ``time``
    csymbol. RoadRunner's gillespie silently does NOT fire such events — it warns
    ("time is not treated continuously in a gillespie simulation … will not be
    precise") and freezes the affected dynamics at the initial state — so RR cannot
    serve as a reference for these models. Detected here so the screen routes them
    to bngsim-vs-its-own-ODE self-validation instead (GH #190)."""
    import libsbml  # heavy; imported lazily so workers don't pull it unless needed

    def _ast_has_time(node) -> bool:
        if node is None:
            return False
        if node.getType() == libsbml.AST_NAME_TIME:
            return True
        return any(_ast_has_time(node.getChild(i)) for i in range(node.getNumChildren()))

    model = libsbml.readSBML(str(xml)).getModel()
    if model is None:
        return False
    for i in range(model.getNumEvents()):
        trig = model.getEvent(i).getTrigger()
        if trig is not None and _ast_has_time(trig.getMath()):
            return True
    return False


def _time_event_verdict(
    xml: str,
    t_end: float,
    n_steps: int,
    n: int,
    seed_base: int,
    min_mean_count: float,
    self_z_tol: float,
    rep_timeout: float,
) -> dict:
    """Adjudicate a time-triggered-event model WITHOUT RoadRunner (its gillespie
    can't fire time events — see :func:`_has_time_event`). Validate bngsim against
    its OWN deterministic ODE instead, over the smallest horizon (requested, grown
    ×10 up to ``EXTENDED_MAX_MULTIPLE``) at which the SSA shows above-floor signal.

    Status (both yellow — a documented RR capability gap, never scored against RR):
      * ``rr_time_event`` — bngsim SSA tracks its own ODE on every above-floor cell
        (bngsim positively validated against the deterministic oracle);
      * ``rr_time_event_unverified`` — bngsim could not be validated against its ODE,
        either because the SSA mean diverges from it (the ODE is NOT the SSA mean for
        nonlinear / heavy-tailed models — e.g. rare explosive seeds — so this is not a
        bngsim fault, merely unconfirmable here) or because bngsim is quiet at every
        tried horizon (``n_compared==0``). A mean-vs-ODE gap is deliberately NOT
        called a bngsim defect: it false-positives on legitimately nonlinear models."""
    import bngsim
    import numpy as np

    species_volumes = {}
    try:
        species_volumes = _species_volumes(xml)
    except Exception:
        species_volumes = {}

    # The validation grows the horizon (×10) to reach bngsim signal, so its
    # replicates run much longer than the partial-horizon screening cap (0.6s) and,
    # under parallel-worker contention, blow past it (validation then falsely bails
    # to "no signal"). Use a generous per-rep cap and fewer replicates so 3 horizon
    # tries × n_val reps still fit the per-model wall; the SSA-vs-ODE tracking check
    # (_cell_tracks_own_ode) is lenient and does not need a 30-replicate ensemble.
    n_val = min(n, 15)
    rep_cap = max(rep_timeout, 2.0)

    h = float(t_end)
    cap = float(t_end) * EXTENDED_MAX_MULTIPLE
    last_timing = None
    while True:
        try:
            _bt, barr, bn_names, _diag, btiming = rc.bn_ssa_replicates(
                xml, 0.0, h, n_steps + 1, n_val, seed_base, rep_timeout=rep_cap
            )
        except bngsim.SimulationTimeout:
            # bngsim itself grew intractable while hunting for signal — stop.
            return {
                "status": "rr_time_event",
                "n_compared": 0,
                "extended_horizon": round(h, 6),
                "bn_tracks_own_ode": None,
                "timing": {"bngsim": last_timing} if last_timing else {},
            }
        last_timing = btiming
        m = bngsim.Model.from_sbml(xml)
        bo = bngsim.Simulator(m, method="ode").run(t_span=(0, h), n_points=n_steps + 1, timeout=60)
        bo_arr, bo_names = np.asarray(bo.species), list(bo.species_names)
        bn_map = {nm: i for i, nm in enumerate(bn_names)}
        bo_map = {nm: i for i, nm in enumerate(bo_names)}
        common = sorted(set(bn_map) & set(bo_map))
        n_rep = barr.shape[0]
        n_compared = 0
        all_track = True
        worst = None
        for ti in range(1, barr.shape[1]):
            for sp in common:
                col = barr[:, ti, bn_map[sp]]
                mu = float(col.mean())
                vol = float(species_volumes.get(sp, 1.0))
                if abs(mu) * vol <= min_mean_count:
                    continue
                n_compared += 1
                se = float(col.std(ddof=1) / np.sqrt(n_rep))
                ode = float(bo_arr[ti, bo_map[sp]])
                if not _cell_tracks_own_ode(mu, ode, se, vol, self_z_tol, min_mean_count):
                    all_track = False
                    gap = abs(mu - ode) * vol
                    if worst is None or gap > worst["gap"]:
                        worst = {"species": sp, "ssa": mu, "ode": ode, "gap": round(gap, 3)}
        if n_compared > 0:
            out = {
                "status": "rr_time_event" if all_track else "rr_time_event_unverified",
                "n_compared": n_compared,
                "extended_horizon": round(h, 6) if h > float(t_end) else None,
                "bn_tracks_own_ode": all_track,
                "timing": {"bngsim": btiming},
            }
            if worst is not None:
                out["bngsim_ode_worst"] = worst
            return out
        if h >= cap:
            return {
                "status": "rr_time_event_unverified",
                "n_compared": 0,
                "bn_tracks_own_ode": None,
                "timing": {"bngsim": btiming},
            }
        h *= 10.0


def _revalidate_time_events(cases: list, specs: list, args) -> None:
    """SERIAL bngsim-vs-ODE self-validation for the time-event models the parallel
    workers tagged ``rr_time_event`` (RR can't anchor them). Run here, single-file,
    with a generous per-rep budget so the extended-horizon ensemble and the cold
    propensity-codegen compile complete (under parallel contention they overrun a
    per-rep cap and falsely report no signal). Mutates each case in place: fills
    ``bn_tracks_own_ode`` / ``n_compared`` / ``extended_horizon`` and sets the status
    to ``rr_time_event`` (bngsim validated against its own ODE) or
    ``rr_time_event_unverified`` (mean-vs-ODE gap from nonlinearity/heavy tails, or
    no signal — explicitly NOT a bngsim defect). GH #190."""
    spec_by_name = {s["name"]: s for s in specs}
    todo = [
        c
        for c in cases
        if c.get("status") == "rr_time_event" and c.get("bn_tracks_own_ode") is None
    ]
    if not todo:
        return
    print(
        f"  re-validating {len(todo)} time-event model(s) bngsim-vs-own-ODE "
        "(serial; RoadRunner cannot anchor time-triggered events)…"
    )
    for c in todo:
        spec = spec_by_name.get(c["name"])
        if not spec:
            continue
        try:
            tev = _time_event_verdict(
                spec["sbml_path"],
                float(spec["t_end"]),
                int(spec["n_steps"]),
                args.n,
                args.seed_base,
                args.min_mean_count,
                args.self_z_tol,
                # Serial (no parallel contention, no per-model wall): give the
                # extended-horizon ensemble + cold codegen compile ample headroom so
                # residual post-screen system load can't trip a false "no signal".
                rep_timeout=30.0,
            )
        except Exception:
            continue  # keep the worker's rr_time_event / no-signal classification
        c["status"] = tev["status"]
        for k in ("n_compared", "bn_tracks_own_ode", "extended_horizon", "bngsim_ode_worst"):
            if k in tev:
                c[k] = tev[k]


def _worker(
    spec: dict,
    n: int,
    seed_base: int,
    mean_z_tol: float,
    max_failures: int,
    min_mean_count: float,
    self_z_tol: float,
    q,
) -> None:
    """Run both engines for one model and put a result dict on ``q``.

    Top-level so it is picklable under the 'spawn' start method. Any
    overrun is handled by the parent killing this process, so there is no
    internal timeout here.
    """
    # Per-process warmup (#135), measured ONCE here before any engine call so the
    # _bngsim_core extension load (bngsim's SSA warmup — no SymPy on this path) and
    # RoadRunner's LLVM/JIT init are timed cold, not charged to the first model's
    # load. Each model is its own subprocess → one warmup sample per result; the
    # matrix aggregates them. Measurement-only: it never touches the verdict.
    warmup = rc.measure_ssa_warmup()

    # Silence RoadRunner's stderr chatter in the disposable child. The bngsim /
    # RoadRunner C extensions are imported lazily inside rc's adapters, so the
    # spawn import here stays light and confined to this killable worker.
    rc.set_rr_quiet()

    xml = spec["sbml_path"]
    t_end = float(spec["t_end"])
    n_steps = int(spec["n_steps"])
    entry = dict(spec)
    entry.pop("sbml_path", None)
    # Accumulate per-engine timing alongside the warmup as each engine runs, so
    # every exit path (including an engine-error early return) carries whatever was
    # measured: an RR-side error still keeps bngsim's load+ensemble timing, and the
    # warmup is present on every row for the matrix's per-process aggregate (#135).
    entry["timing"] = {"warmup": warmup}

    # Activity probe (single fast seed) FIRST → an activity sidecar, so a model the
    # parent ultimately kills still reports a reaction-events/unit-time estimate.
    # Best-effort; separately seeded clones, never perturbs the verdict.
    import bngsim  # for SimulationTimeout below

    try:
        rate, _h, _full_ok = _activity_probe(xml, t_end, seed_base)
    except Exception:
        rate = None
    if rate is not None:
        try:
            PROBE_DIR.mkdir(parents=True, exist_ok=True)
            (PROBE_DIR / f"{spec['name']}.json").write_text(json.dumps({"events_per_time": rate}))
        except Exception:
            pass

    def _activity_block():
        return (
            {
                "events_per_time": rate,
                "events_per_rep": round(rate * t_end, 1),
                "events_probe": True,
            }
            if rate is not None
            else {}
        )

    # ── RoadRunner cannot anchor time-triggered-event models: its gillespie
    # silently does NOT fire time events (it warns and freezes at the IC). So do
    # not compare against RR — validate bngsim against its OWN ODE oracle instead
    # and report a documented RR capability gap (GH #190).
    try:
        _is_time_event = _has_time_event(xml)
    except Exception:
        _is_time_event = False
    if _is_time_event:
        # Classify only — cheaply. The bngsim-vs-ODE self-validation runs to an
        # extended horizon and triggers a cold propensity-codegen compile, which
        # overruns any per-rep cap under parallel-worker contention (it then falsely
        # bails to "no signal"). So just confirm the model is SSA-compatible here and
        # tag it ``rr_time_event``; a SERIAL post-pass (no contention, warm .so
        # cache) fills bn_tracks_own_ode via _revalidate_time_events.
        incompatible = False
        try:
            _m = bngsim.Model.from_sbml(xml)
            bngsim.Simulator(_m, method="ssa")  # raises SsaValidationError if not SSA-able
        except bngsim.SsaValidationError:
            incompatible = True  # fall through to the normal UNSUPPORTED path
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = f"bngsim: {type(exc).__name__}: {exc}"[:300]
            q.put(entry)
            return
        if not incompatible:
            entry["status"] = "rr_time_event"
            entry["bn_tracks_own_ode"] = None
            entry["time_event"] = True
            if rate is not None:
                entry.setdefault("timing", {})
                entry["timing"].setdefault("bngsim", {}).update(_activity_block())
            q.put(entry)
            return

    # ── Try the FULL-horizon bngsim ensemble, BOUNDED per replicate. A blowup seed
    # (a runtime sign-indefinite rate explosion that never finishes on some seeds —
    # the common cause of a 'too_slow' whose other seeds are instant, e.g.
    # MODEL1909300003: seed 2007 explodes while every other seed fires 62 events)
    # raises SimulationTimeout here → fall back to partial-horizon adjudication.
    bn_arr = bn_names = bn_timing = bn_ssa_diag = None
    try:
        t0 = time.perf_counter()
        bn_times, bn_arr, bn_names, bn_ssa_diag, bn_timing = rc.bn_ssa_replicates(
            xml, 0.0, t_end, n_steps + 1, n, seed_base, rep_timeout=REP_WALL_CAP
        )
        entry["elapsed_sec_bn"] = round(time.perf_counter() - t0, 3)
        full = True
    except bngsim.SimulationTimeout:
        full = False
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"bngsim: {type(exc).__name__}: {exc}"[:300]
        q.put(entry)
        return

    # ── Partial-horizon fallback: the full ensemble has a replicate that never
    # finishes. Adjudicate over the largest window where ALL replicates do — down to
    # PARTIAL_MIN_FRACTION of the horizon, so even event-count-bomb models get a
    # comparison over a tiny but event-rich window; only report gray if NO window
    # both finishes and carries above-floor signal. A provisional (yellow) verdict;
    # no ODE attribution (the full-horizon oracle would not finish either).
    if not full:
        try:
            part = _partial_adjudicate(
                xml,
                t_end,
                n_steps,
                n,
                seed_base,
                mean_z_tol,
                max_failures,
                min_mean_count,
                REP_WALL_CAP,
            )
            h = part.pop("horizon")
            ptiming = part.pop("timing", {})
            entry["partial_pass"] = part.get("status") == "pass"
            entry.update(part)
            entry["timing"].update(ptiming)
            entry["partial_horizon"] = round(h, 6)
            entry["status"] = "partial"
        except bngsim.SimulationTimeout:
            entry["status"] = "too_slow"
            entry["error"] = (
                "exact SSA intractable — a replicate never finishes even over a 1% window"
            )
            if rate is not None:
                entry["timing"]["bngsim"] = _activity_block()
        except Exception as exc:
            entry["status"] = "error"
            # A RoadRunner-side refusal is re-raised "roadrunner:"-prefixed by
            # _partial_adjudicate → REFERENCE_FAILED via _classify_error; anything
            # else is a bngsim-side error.
            msg = str(exc)
            entry["error"] = (
                msg[:300]
                if msg.startswith("roadrunner:")
                else f"bngsim: {type(exc).__name__}: {exc}"[:300]
            )
        q.put(entry)
        return

    # ── Full-horizon flow: bngsim completed. Run RoadRunner (unbounded — at a horizon
    # bngsim handled it is generally fast too), compare, then attribute a fail.
    entry["timing"]["bngsim"] = bn_timing
    try:
        t0 = time.perf_counter()
        rr_times, rr_arr, rr_names, rr_timing = rc.rr_ssa(
            xml, 0.0, t_end, n_steps + 1, n, seed_base
        )
        entry["elapsed_sec_rr"] = round(time.perf_counter() - t0, 3)
        entry["timing"]["roadrunner"] = rr_timing
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = f"roadrunner: {type(exc).__name__}: {exc}"[:300]
        q.put(entry)
        return

    try:
        species_volumes = _species_volumes(xml)
    except Exception:
        species_volumes = {}

    entry.update(
        _compare(
            bn_times,
            bn_arr,
            bn_names,
            rr_times,
            rr_arr,
            rr_names,
            mean_z_tol,
            max_failures,
            species_volumes=species_volumes,
            min_mean_count=min_mean_count,
        )
    )

    # ── Extended-horizon fallback (GH #190): the requested horizon was too QUIET —
    # every (t>0, species) cell fell below the low-count floor, so nothing was
    # compared (a vacuous pass). Symmetric to the partial-horizon contraction for
    # too-active models: grow the window ×10 (up to EXTENDED_MAX_MULTIPLE) and, if a
    # slow / delayed-onset model accumulates above-floor signal there, adjudicate
    # over that larger window instead. Count-limited models (0/1-copy switch
    # species, static systems) never cross the floor at any horizon → stay vacuous.
    if (
        entry.get("status") == "pass"
        and int(entry.get("n_compared", 0) or 0) == 0
        and int(entry.get("n_skipped_lowcount", 0) or 0) > 0
    ):
        try:
            ext = _extended_adjudicate(
                xml,
                t_end,
                n_steps,
                n,
                seed_base,
                mean_z_tol,
                max_failures,
                min_mean_count,
                REP_WALL_CAP,
            )
        except Exception:
            ext = None
        if ext is not None:
            eh = ext.pop("horizon")
            etiming = ext.pop("timing", {})
            entry["extended_pass"] = ext.get("status") == "pass"
            entry.update(ext)
            entry["timing"].update(etiming)
            entry["extended_horizon"] = round(eh, 6)
            entry["status"] = "extended"

    # Runtime sign-indefinite annotation (GH #109): if any replicate fired a
    # reaction in reverse (rate law went negative; GH #110 mean-faithful firing),
    # tag the model. Additive context for the DIFF attribution below -- never a
    # verdict and never gates the status.
    sign_indef = _sign_indefinite_annotation(bn_ssa_diag)
    if sign_indef is not None:
        entry["sign_indefinite"] = sign_indef

    # PRELIMINARY persist: the SSA verdict is complete; the ODE attribution that
    # follows can hang on a stiff deterministic oracle (a common false 'too_slow' —
    # the SSA was instant, e.g. MODEL1909300003 fires 62 events). Write the verdict
    # to a sidecar NOW so a kill during attribution recovers it. A sidecar (not a
    # queue put) because a multiprocessing feeder thread cannot flush a put while the
    # C++ solve holds the GIL — the put would be lost on terminate. ``attribution_
    # pending`` flags a fail whose fault is undetermined → the scheduler maps a
    # recovered one to a provisional 'attribution_incomplete' (yellow), not red.
    needs_attrib = entry.get("status") == "fail" and spec["name"] not in KNOWN_RR_ISSUES
    if needs_attrib:
        entry["attribution_pending"] = True
        with contextlib.suppress(Exception):
            (PROBE_DIR / f"{spec['name']}.verdict.json").write_text(
                json.dumps(entry, default=_json_default)
            )

    # Attribute a surviving fail against each engine's own ODE (GH #107 follow-up).
    # KNOWN_RR_ISSUES models are left as ``fail`` here so the parent reclassifies
    # them to the more specific ``rr_known`` (with a filed issue #).
    if needs_attrib:
        try:
            attrib = _attribute_fail(
                xml,
                t_end,
                n_steps,
                bn_arr,
                bn_names,
                rr_arr,
                rr_names,
                species_volumes,
                mean_z_tol,
                min_mean_count,
                self_z_tol,
            )
            entry["diff_attribution"] = attrib
            entry["status"] = attrib["status"]
        except Exception as exc:
            entry["diff_attribution"] = {
                "status": "fail",
                "reason": f"attribution failed: {type(exc).__name__}: {exc}"[:200],
            }
    entry["attribution_pending"] = False
    q.put(entry)


def _build_corpus(args) -> list[dict]:
    specs: list[dict] = []

    if args.corpus in ("all", "roundtrip"):
        if SUITE_SSA.exists():
            roundtrip_models = _load_suite(SUITE_SSA)
        elif args.corpus == "roundtrip":
            sys.exit(f"missing roundtrip suite manifest: {SUITE_SSA}")
        else:
            roundtrip_models = []
            print(f"  (roundtrip corpus skipped: {SUITE_SSA} not found)")
        for m in roundtrip_models:
            name = m["name"]
            xml = ROUNDTRIP_DIR / f"{name}.xml"
            specs.append(
                {
                    "name": name,
                    "source": "roundtrip",
                    "sbml_path": str(xml),
                    "exists": xml.exists(),
                    "t_end": float(m["t_end"]),
                    "n_steps": int(m["n_steps"]),
                    "effort": "high" if name in ROUNDTRIP_LARGE else "low",
                }
            )

    if args.corpus in ("all", "biomodels"):
        if not BIOMODELS_MANIFEST.exists():
            sys.exit(
                f"missing {BIOMODELS_MANIFEST}; run "
                "benchmarks/biomodels_ssa/build_ssa_candidates.py first."
            )
        import json

        manifest = json.loads(BIOMODELS_MANIFEST.read_text())
        for m in manifest["models"]:
            name = m["name"]
            xml = BIOMODELS_SBML_DIR / f"{name}.xml"
            specs.append(
                {
                    "name": name,
                    "source": "biomodels",
                    "sbml_path": str(xml),
                    "exists": xml.exists(),
                    "t_end": args.t_end,
                    "n_steps": args.n_steps,
                    "effort": m.get("effort", "high"),
                    "n_species": m.get("n_species"),
                }
            )

    # effort filter (cumulative) + name filter + max-models cap.
    specs = [s for s in specs if effort_allows(args.effort, s["effort"])]
    if args.models:
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        specs = [s for s in specs if s["name"] in wanted]
    if args.max_models:
        # Keep all roundtrip, then cheapest biomodels up to the cap.
        rt = [s for s in specs if s["source"] == "roundtrip"]
        bm = [s for s in specs if s["source"] == "biomodels"][: args.max_models]
        specs = rt + bm
    return specs


def _schedule(specs, args) -> list[dict]:
    """Run model specs in up to ``--jobs`` child processes with a per-model
    wall-clock cap; kill and mark ``too_slow`` on overrun."""
    import multiprocessing as mp
    import shutil

    # Fresh activity-probe sidecar dir for this run (a killed worker leaves its
    # estimate here; we recover it on the too_slow branch below).
    shutil.rmtree(PROBE_DIR, ignore_errors=True)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    pending = list(specs)
    running: list[dict] = []
    done: dict[str, dict] = {}
    total = len(specs)
    finished = 0

    while pending or running:
        while pending and len(running) < args.jobs:
            spec = pending.pop(0)
            if not spec.get("exists", True):
                entry = dict(spec)
                entry.pop("sbml_path", None)
                entry["status"] = "skip"
                entry["error"] = "sbml file not found"
                done[spec["name"]] = entry
                finished += 1
                continue
            q = ctx.Queue()
            proc = ctx.Process(
                target=_worker,
                args=(
                    spec,
                    args.n,
                    args.seed_base,
                    args.mean_z_tol,
                    MAX_FAILURES_RECORDED,
                    args.min_mean_count,
                    args.self_z_tol,
                    q,
                ),
            )
            proc.start()
            running.append(
                {"proc": proc, "q": q, "spec": spec, "start": time.time(), "latest": None}
            )

        still = []
        for r in running:
            proc = r["proc"]
            spec = r["spec"]
            # Drain every queued message each tick, keeping the LATEST. The worker
            # puts a PRELIMINARY verdict before the (hang-prone) ODE attribution and
            # a FINAL one after, so a kill during attribution still leaves the
            # verdict here instead of being discarded as 'too_slow'.
            while True:
                try:
                    r["latest"] = r["q"].get_nowait()
                except Exception:
                    break
            if not proc.is_alive():
                proc.join()
                while True:  # final drain — the FINAL put may have landed post-tick
                    try:
                        r["latest"] = r["q"].get_nowait()
                    except Exception:
                        break
                res = r["latest"]
                if res is None:
                    try:
                        res = r["q"].get(timeout=2.0)
                    except Exception:
                        res = dict(spec)
                        res.pop("sbml_path", None)
                        res["status"] = "error"
                        res["error"] = f"worker died without result (exit={proc.exitcode})"
                res.pop("attribution_pending", None)
                done[spec["name"]] = res
                finished += 1
                _print_progress(finished, total, res)
            elif time.time() - r["start"] > args.timeout:
                proc.terminate()
                proc.join()
                while True:  # one last drain — the worker may have FINISHED right at the cap
                    try:
                        r["latest"] = r["q"].get_nowait()
                    except Exception:
                        break
                vf = PROBE_DIR / f"{spec['name']}.verdict.json"
                if r["latest"] is not None:
                    entry = r["latest"]  # finished just before the kill — a full result
                    entry.pop("attribution_pending", None)
                elif vf.exists():
                    # Recovered a completed SSA verdict despite the kill — the slow step
                    # was the ODE attribution, not the SSA (e.g. MODEL1909300003: SSA
                    # fires 62 events, the deterministic oracle hangs). A fail we never
                    # finished attributing is provisional, not a bngsim defect →
                    # 'attribution_incomplete' (yellow).
                    try:
                        entry = json.loads(vf.read_text())
                        if entry.get("attribution_pending") and entry.get("status") == "fail":
                            entry["status"] = "attribution_incomplete"
                        entry.pop("attribution_pending", None)
                    except Exception:
                        entry = None
                else:
                    entry = None
                if entry is None:
                    entry = dict(spec)
                    entry.pop("sbml_path", None)
                    entry["status"] = "too_slow"
                    entry["error"] = f"exceeded {args.timeout}s wall cap"
                    entry["elapsed_sec_wall"] = round(time.time() - r["start"], 1)
                    # Recover the activity probe the killed worker wrote before its
                    # ensemble, so a timed-out (= most-active) model still reports its
                    # stochastic activity (minimal bngsim timing block; probe estimate).
                    pf = PROBE_DIR / f"{spec['name']}.json"
                    if pf.exists():
                        try:
                            v = float(json.loads(pf.read_text())["events_per_time"])
                            entry["timing"] = {
                                "bngsim": {
                                    "events_per_time": v,
                                    "events_per_rep": round(v * float(spec["t_end"]), 1),
                                    "events_probe": True,
                                }
                            }
                        except Exception:
                            pass
                done[spec["name"]] = entry
                finished += 1
                _print_progress(finished, total, entry)
            else:
                still.append(r)
        running = still
        if running:
            time.sleep(0.05)

    return [done[s["name"]] for s in specs]


def _print_progress(finished: int, total: int, res: dict) -> None:
    st = res["status"]
    tag = {
        "pass": "PASS",
        "fail": "FAIL",
        "error": "ERR ",
        "too_slow": "SLOW",
        "skip": "SKIP",
        "diff_not_bngsim": "DIFF~",
        "diff_ode_level": "DIFFo",
        "partial": "PART ",
        "attribution_incomplete": "ATTR?",
    }.get(st, st)
    extra = ""
    if st == "partial":
        extra = (
            f" partial t=0→{res.get('partial_horizon', '?')} max_z={res.get('max_mean_z', 0):.2f}"
        )
    elif st in ("pass", "fail", "diff_not_bngsim", "diff_ode_level", "attribution_incomplete"):
        extra = (
            f" max_z={res.get('max_mean_z', 0):.2f} zf={res.get('n_z_failures', 0)}"
            f" lowcnt_skip={res.get('n_skipped_lowcount', 0)}"
        )
        attrib = res.get("diff_attribution") or {}
        if st in ("diff_not_bngsim", "diff_ode_level"):
            w = attrib.get("worst_cell") or {}
            extra += (
                f" bn_tracks_ode={attrib.get('bn_tracks_own_ode')}"
                f" rr_tracks_ode={attrib.get('rr_tracks_own_ode')}"
                f" odes_agree={attrib.get('odes_agree')}"
            )
            if w:
                extra += f" worst={w.get('species')}"
    elif st in ("error", "too_slow", "skip"):
        extra = f" {res.get('error', '')[:70]}"
    print(f"  [{finished}/{total}] {tag} {res['name']} ({res['source']}){extra}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["all", "roundtrip", "biomodels"], default="all")
    p.add_argument("--effort", choices=list(EFFORT_LEVELS), default="high")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="Replicates per engine.")
    p.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="Concurrent model processes.")
    p.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-model wall-clock cap (s)."
    )
    p.add_argument("--max-models", type=int, default=0, help="Cap biomodels count (0 = all).")
    p.add_argument("--t-end", type=float, default=DEFAULT_T_END, help="BioModels horizon.")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="BioModels output steps.")
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--mean-z-tol", type=float, default=DEFAULT_MEAN_Z_TOL)
    p.add_argument(
        "--min-mean-count",
        type=float,
        default=DEFAULT_MIN_MEAN_COUNT,
        help="Low-count z-gate floor (molecules): skip a cell when both engines "
        "are at/below this many molecules (GH #107.3). Default 2.0.",
    )
    p.add_argument(
        "--self-z-tol",
        type=float,
        default=DEFAULT_SELF_Z_TOL,
        help="DIFF attribution: a surviving fail is reclassified diff_not_bngsim "
        "when bngsim SSA tracks its own ODE within this many SE while RR does not. "
        "Default 7.30 (4.0 scaled by sqrt(N/30) at N=100).",
    )
    p.add_argument("--models", type=str, default="", help="Comma-separated name filter.")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Native run report path (default runs/ssa_screen.json, gitignored).",
    )
    p.add_argument(
        "--core-out",
        type=Path,
        default=DEFAULT_CORE_OUT,
        help="Shared _core-schema report path (default runs/ssa_report.json, gitignored).",
    )
    args = p.parse_args()

    specs = _build_corpus(args)
    n_rt = sum(1 for s in specs if s["source"] == "roundtrip")
    n_bm = sum(1 for s in specs if s["source"] == "biomodels")

    print("=" * 72)
    print("  bngsim SSA vs roadrunner gillespie -- broad-corpus parity screen")
    print("=" * 72)
    print(
        f"  corpus:     {args.corpus} ({len(specs)} models: {n_rt} roundtrip + {n_bm} biomodels)"
    )
    print(f"  effort:     {args.effort}    replicates: {args.n}    jobs: {args.jobs}")
    print(
        f"  per-model cap: {args.timeout}s    biomodels horizon: t_end={args.t_end} n_steps={args.n_steps}"
    )
    print(f"  mean_z_tol: {args.mean_z_tol}    seed_base: {args.seed_base}")
    print(f"  min_mean_count (low-count z-gate floor): {args.min_mean_count} molecule(s)")
    print(f"  self_z_tol (DIFF attribution vs own ODE): {args.self_z_tol}")
    print()

    info = _machine_info()
    t_start = time.perf_counter()
    cases = _schedule(specs, args)
    elapsed = time.perf_counter() - t_start

    n_rr_known = _annotate_known_rr_issues(cases)
    if n_rr_known:
        print(f"  ({n_rr_known} fail(s) reclassified rr_known -- see KNOWN_RR_ISSUES)")

    _revalidate_time_events(cases, specs, args)

    by_status: dict[str, int] = {}
    for c in cases:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1

    # Coverage accounting (GH #107.2): a too_slow/error/skip model is never
    # compared, so it is an explicit coverage gap -- log the dropped set rather
    # than letting an unflagged absence read as green.
    too_slow_models = sorted(c["name"] for c in cases if c["status"] == "too_slow")
    error_models = sorted(c["name"] for c in cases if c["status"] == "error")
    skip_models = sorted(c["name"] for c in cases if c["status"] == "skip")
    n_compared_models = sum(1 for c in cases if c["status"] not in ("too_slow", "error", "skip"))
    total_lowcount_skips = sum(int(c.get("n_skipped_lowcount", 0)) for c in cases)
    # DIFF attribution (GH #107 follow-up): each surviving cross-engine fail is
    # attributed against each engine's own ODE. ``diff_not_bngsim`` / ``rr_known``
    # are explained-as-not-bngsim; plain ``fail`` / ``diff_ode_level`` are not.
    attributed = sorted(
        (c["name"], c["status"], (c.get("diff_attribution") or {}).get("reason", ""))
        for c in cases
        if c["status"] in ("diff_not_bngsim", "diff_ode_level", "rr_known", "fail")
    )
    # Runtime sign-indefinite annotation (GH #109): models bngsim reverse-fired
    # because a rate law went negative (GH #110). Context, not a verdict -- these
    # are the sign-indefinite cases the static t0/manual corpus gates miss.
    sign_indefinite = sorted(
        (c["name"], c["status"], c["sign_indefinite"]["reaction"])
        for c in cases
        if c.get("sign_indefinite")
    )
    # A model whose every (t, species) cell fell under the low-count floor
    # "passes" vacuously -- no cell was actually z-tested. Surface it so the
    # green is honest rather than a hidden no-op (GH #107.3).
    vacuous_lowcount = sorted(
        c["name"]
        for c in cases
        if c["status"] == "pass"
        and int(c.get("n_compared", 0)) == 0
        and int(c.get("n_skipped_lowcount", 0)) > 0
    )

    print()
    print("=" * 72)
    print(
        "  "
        + "  ".join(f"{k.upper()}: {v}" for k, v in sorted(by_status.items()))
        + f"   elapsed: {elapsed:.1f}s"
    )
    print("=" * 72)
    print(
        f"  compared: {n_compared_models}/{len(cases)} models    "
        f"low-count cells skipped (z-gate floor): {total_lowcount_skips}"
    )
    if too_slow_models:
        print(
            f"  too_slow (DROPPED, exact-SSA-intractable at the {args.timeout}s cap; "
            f"out of scope for exact SSA): {len(too_slow_models)}"
        )
        print(f"    {', '.join(too_slow_models)}")
    if error_models:
        print(f"  error (DROPPED, engine refused/failed): {len(error_models)}")
        print(f"    {', '.join(error_models)}")
    if skip_models:
        print(f"  skip (DROPPED, sbml missing): {len(skip_models)}")
    if vacuous_lowcount:
        print(
            f"  pass-but-uncompared (every cell below the low-count floor): "
            f"{len(vacuous_lowcount)}"
        )
        print(f"    {', '.join(vacuous_lowcount)}")
    if attributed:
        print("  DIFF attribution (surviving cross-engine fails, vs each engine's own ODE):")
        for name, st, reason in attributed:
            print(f"    [{st}] {name}: {reason}")
    if sign_indefinite:
        print(
            "  runtime sign-indefinite (GH #109; bngsim reverse-fired a negative-rate "
            f"reaction, GH #110): {len(sign_indefinite)}"
        )
        for name, st, rxn in sign_indefinite:
            print(f"    [{st}] {name}: reverse-fired {rxn}")
    print("=" * 72)

    payload = {
        "machine_info": info,
        "config": {
            "corpus": args.corpus,
            "effort": args.effort,
            "n_replicates": args.n,
            "jobs": args.jobs,
            "per_model_timeout_sec": args.timeout,
            "biomodels_t_end": args.t_end,
            "biomodels_n_steps": args.n_steps,
            "seed_base": args.seed_base,
            "mean_z_tol": args.mean_z_tol,
            "min_mean_count": args.min_mean_count,
            "biomodels_sbml_dir": str(BIOMODELS_SBML_DIR),
        },
        "summary": {"total": len(cases), **by_status, "elapsed_sec": round(elapsed, 2)},
        "coverage": {
            "n_compared_models": n_compared_models,
            "n_lowcount_cells_skipped": total_lowcount_skips,
            "too_slow": too_slow_models,
            "error": error_models,
            "skip": skip_models,
            "pass_but_uncompared_lowcount": vacuous_lowcount,
            "diff_not_bngsim": sorted(
                c["name"] for c in cases if c["status"] == "diff_not_bngsim"
            ),
            "diff_ode_level": sorted(c["name"] for c in cases if c["status"] == "diff_ode_level"),
            "sign_indefinite": [
                {"name": name, "status": st, "reaction": rxn} for name, st, rxn in sign_indefinite
            ],
        },
        "cases": {c["name"]: c for c in cases},
    }
    _save(payload, args.out)

    # Also emit the run in the shared _core report schema (GH #108): the SSA
    # screen as a tracked regression in the suite's taxonomy. Imported here (not
    # at module top) so the spawned workers never pull it in. The committed
    # baseline the regression diffs against is authored from this in a follow-up
    # step; here it lands in the gitignored runs/ dir alongside the native report.
    import ssa_attribution  # noqa: E402
    from _core import write_report  # noqa: E402

    core_meta, core_results = ssa_attribution.payload_to_report(payload)
    args.core_out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.core_out, core_results, meta=core_meta)
    print(f"Core report saved to {args.core_out}")

    # Explained-as-not-bngsim statuses (rr_known, diff_not_bngsim) do not fail
    # the screen. Unexplained fails, ODE-level divergences, and engine errors do.
    n_bad = (
        by_status.get("fail", 0) + by_status.get("diff_ode_level", 0) + by_status.get("error", 0)
    )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
