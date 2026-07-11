#!/usr/bin/env python3
"""RoadRunner-via-SBML SSA oracle — the bng_parity SECOND *engine* (companion to
``net_gillespie.py``).

When an SSA-track job's LEGACY reference (``run_network``) fails — most often the
``edgepop`` molecule/edge-population observable crash (GH: reference-side bug) —
``net_gillespie.py`` is the PRIMARY independent oracle. This module adds a SECOND
independent SSA engine: libRoadRunner's Gillespie, reached by round-tripping the
SAME BNG ``.net`` through bngsim's own faithful ``.net`` → SBML converter
(``bngsim.convert.net_to_sbml``) and running the emitted SBML under RoadRunner's
``gillespie`` integrator.

Two jobs
    * EXTENDS coverage. ``net_gillespie`` refuses functional / time-dependent /
      concentration-unit nets (its supported gate is elementary constant-rate
      mass-action, count-based only). RoadRunner is an assignment-rule-aware SBML
      engine, so it can score some of those refused nets — turning an unscored
      REFERENCE_FAILED row into a real verdict.
    * CORROBORATES. Where ``net_gillespie`` also runs, agreement of a SECOND
      independent engine on the same ``.net`` raises confidence past a single oracle.

Independence — what this validates and what it does NOT
    RoadRunner consumes SBML that BNGSIM ITSELF wrote, so this validates bngsim's SSA
    *engine* against an independent solver — it does NOT independently validate the
    ``.net`` → SBML conversion. That conversion is guarded separately: this oracle
    refuses unless :func:`bngsim.convert.net_to_sbml` both converts under ``strict``
    (no construct SBML cannot carry faithfully) AND reports an RHS-faithful round-trip
    (scale-relative ``max_rhs_delta`` ≈ 0). A conversion defect that slips that gate
    surfaces here as a DIFF (bngsim's SSA vs RoadRunner disagree), never a false PASS.
    ``net_gillespie.py`` — its own ``.net`` parser, sharing no bngsim code — remains
    the FULLY independent primary oracle; treat RoadRunner as the second engine.

    One class of ``.net`` is deterministically faithful yet stochastically UNfaithful,
    so the RHS gate alone is not enough: a homo-oligomerization (repeated reactant, e.g.
    ``2A -> C``). Its exact-SSA propensity is the falling factorial ``k*A*(A-1)`` (what
    bngsim and ``net_gillespie`` compute), but SBML mass-action carries the ODE rate law
    ``k*A*A`` — the correct RHS, a different function of the count — which RoadRunner's
    Gillespie takes literally as the propensity (firing even at ``A == 1``). The two
    laws cannot be reconciled in one SBML kinetic law, so this oracle screens such nets
    out (:func:`_has_repeated_reactant`) and stays unscored rather than emit a DIFF that
    would be wrongly charged to bngsim's (correct) SSA. GH #9.

Why it sidesteps ``edgepop``
    ``net_to_sbml`` lowers each ``.net`` observable — the network generator's
    pre-resolved ``groups`` block — to an SBML ``<parameter>`` + ``<assignmentRule>``
    that is a weighted sum over species indices, with NO pattern/bond matching. So
    RoadRunner computes exactly the observables that crash run_network's ``edgepop``
    path, the same way ``net_gillespie`` reads the resolved ``groups`` block.

Optional dependency
    ``libroadrunner`` is a DEVELOPER extra (pyproject ``roadrunner``). When it is
    absent every entry point returns ``None`` and the caller keeps the honest
    REFERENCE_FAILED — this is never a hard bng_parity dependency.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter as _now

import numpy as np

# Scale-relative RHS-faithfulness gate on the .net → SBML conversion. net_to_sbml's
# L1 ``max_rhs_delta`` is ``max|Δrhs| / max(1, max|rhs|)`` over the initial state and a
# nonlinear-exercising perturbation; a faithful conversion is ~0. Above this the SBML
# RoadRunner would run is not the model bngsim simulated → refuse (stay unscored)
# rather than emit a misattributed DIFF.
RHS_FAITHFUL_TOL = 1e-6

# Conversion + RoadRunner JIT + n_rep replicates can be costly for a large network;
# past this wall the oracle bails (returns None) so a slow model stays an honest
# REFERENCE_FAILED instead of dominating the run. The bng_parity caller caps the RR
# comparison horizon (RR_ORACLE_MAX_POINTS) to keep a large net like prion's 2809-rxn
# network tractable (~90s for a 10-rep t≤10 ensemble); this wall gives that capped run
# headroom (JIT + slower machines) while still bounding a runaway.
DEFAULT_WALL_BUDGET_SEC = 180.0


def roadrunner_available() -> bool:
    """True iff ``libroadrunner`` can be imported (the optional dev extra is present)."""
    try:
        import roadrunner  # noqa: F401
    except Exception:
        return False
    return True


def _has_repeated_reactant(sbml_text: str) -> bool:
    """True iff some reaction consumes one species with total reactant multiplicity ≥2
    (a homo-oligomerization such as ``2A -> C``) — the one construct whose SBML is
    ODE-faithful yet NOT stochastically faithful under RoadRunner's Gillespie (GH #9).

    bngsim's SSA (``NetworkModel::compute_propensity``) and the ``net_gillespie``
    oracle both use the exact falling-factorial propensity ``k·A·(A-1)`` for such a
    reaction, but ``net_to_sbml`` emits the *deterministic* mass-action kinetic law
    ``k·A·A`` — the correct ODE RHS, yet a different function of the integer count.
    RoadRunner's ``gillespie`` integrator evaluates that law literally as the
    propensity, so it fires ``2A -> C`` even at ``A == 1`` (propensity ``0.5·k`` instead
    of ``0``), inflating BOTH the mean and the variance. The two SBML kinetic laws
    cannot be reconciled — ``A·(A-1) ≠ A·A`` as functions — so no converter change fixes
    this for a stochastic engine; the RHS-faithfulness gate is blind to it (``A·A`` IS
    the right RHS). We therefore screen for it structurally and refuse, keeping an
    honest REFERENCE_FAILED rather than emitting a DIFF that would be misattributed to
    bngsim's (correct) SSA.
    """
    import libsbml

    doc = libsbml.readSBMLFromString(sbml_text)
    model = doc.getModel()
    if model is None:
        return False
    for i in range(model.getNumReactions()):
        r = model.getReaction(i)
        tally: dict[str, float] = {}
        for j in range(r.getNumReactants()):
            sr = r.getReactant(j)
            tally[sr.getSpecies()] = tally.get(sr.getSpecies(), 0.0) + sr.getStoichiometry()
        if any(v >= 1.5 for v in tally.values()):
            return True
    return False


def _faithful_sbml_text(net_path: str | Path) -> str | None:
    """``.net`` → SBML text via bngsim's converter, gated for faithfulness.

    Returns the SBML string, or ``None`` when the model is not faithfully
    representable as SBML for a stochastic engine: ``net_to_sbml`` raised under
    ``strict`` (a construct SBML cannot carry); the round-trip ODE-RHS delta exceeds
    :data:`RHS_FAITHFUL_TOL` (the emitted SBML is not the model bngsim simulated); or a
    reaction is a homo-oligomerization (repeated reactant) whose SBML mass-action law is
    ODE-faithful but not propensity-faithful under RoadRunner's Gillespie
    (:func:`_has_repeated_reactant`, GH #9). ``None`` → the caller keeps REFERENCE_FAILED.
    """
    try:
        from bngsim.convert import net_to_sbml

        report = net_to_sbml(net_path, out_path=None, validate="L1", strict=True)
    except Exception:
        return None
    delta = report.max_rhs_delta
    if delta is None or not np.isfinite(delta) or delta > RHS_FAITHFUL_TOL:
        return None
    if _has_repeated_reactant(report.output_text):
        return None
    return report.output_text


def _observable_id_map(sbml_text: str) -> dict[str, str]:
    """Raw observable/function name → SBML SId for every assignment-rule parameter.

    ``net_to_sbml`` lowers each ``.net`` observable (and function) to a global
    ``<parameter>`` driven by an ``<assignmentRule>``; the raw ``.net`` name is the
    parameter's display *name* (or its *id*, when the name was already a valid unique
    SId). Returns ``raw name → SId`` so the caller can select bngsim's own observable
    names in RoadRunner and align by name. Assignment-rule *species* (not parameters)
    are excluded — only observable/function scalars are reportable targets here.
    """
    import libsbml

    doc = libsbml.readSBMLFromString(sbml_text)
    model = doc.getModel()
    if model is None:
        return {}
    ar_targets = {
        model.getRule(i).getVariable()
        for i in range(model.getNumRules())
        if model.getRule(i).isAssignment()
    }
    out: dict[str, str] = {}
    for i in range(model.getNumParameters()):
        p = model.getParameter(i)
        pid = p.getId()
        if pid in ar_targets:
            out[p.getName() or pid] = pid
    return out


def net_roadrunner_ensemble(
    net_path: str | Path,
    t_grid,
    n_rep: int,
    seed_base: int,
    *,
    obs_names=None,
    wall_budget_sec: float = DEFAULT_WALL_BUDGET_SEC,
):
    """Independent RoadRunner-Gillespie SSA ensemble on ``net_path`` at ``t_grid``.

    Returns ``(t_grid, values, names)`` with ``values`` shape (n_rep, n_time, n_obs) —
    the same layout the harness's bngsim ensemble uses, so ``_compare_stoch`` scores it
    directly — or ``None`` when the oracle is unavailable/inapplicable: ``libroadrunner``
    is not installed; the ``.net`` is not faithfully representable as SBML
    (:func:`_faithful_sbml_text`); the sample grid is non-uniform (RoadRunner's uniform
    ``simulate`` grid would misalign row-for-row); RoadRunner raises; or the ensemble
    exceeds ``wall_budget_sec``. On ``None`` the caller keeps the honest REFERENCE_FAILED.

    Replicates use seeds ``seed_base, seed_base+1, …`` — the SAME schedule as bngsim and
    ``net_gillespie`` — so a comparison is like-for-like. ``obs_names`` (bngsim's own
    observable names, i.e. ``bn[2]``) restricts the reported set to exactly those
    observables and in that order; ``None`` reports every SBML observable/function and
    lets the caller's name-alignment take the intersection.
    """
    try:
        import roadrunner
    except Exception:
        return None

    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_time = int(t_grid.size)
    if n_time < 2:
        return None
    # A uniform time course is required: RoadRunner.simulate(t0, t_end, n) samples a
    # uniform grid, so a non-uniform bn grid would misalign cell-for-cell (and
    # _compare_stoch reduces per-cell, assuming a shared grid).
    step = t_grid[1] - t_grid[0]
    if not (step > 0 and np.allclose(np.diff(t_grid), step, rtol=1e-6, atol=1e-12)):
        return None

    sbml_text = _faithful_sbml_text(net_path)
    if sbml_text is None:
        return None
    id_map = _observable_id_map(sbml_text)
    if not id_map:
        return None

    if obs_names is not None:
        pairs = [(nm, id_map[nm]) for nm in obs_names if nm in id_map]
    else:
        pairs = list(id_map.items())
    if not pairs:
        return None
    names = [nm for nm, _ in pairs]
    ids = [pid for _, pid in pairs]

    try:
        rr = roadrunner.RoadRunner(sbml_text)
        rr.integrator = "gillespie"
        rr.integrator.variable_step_size = False
        # Report the observable parameters (not the default floating-species set) so the
        # columns align by name with bngsim's observable ensemble.
        rr.timeCourseSelections = ["time"] + ids
    except Exception:
        return None

    t0, t_end = float(t_grid[0]), float(t_grid[-1])
    vals = np.empty((n_rep, n_time, len(ids)), dtype=np.float64)
    deadline = _now() + wall_budget_sec
    try:
        for rep in range(n_rep):
            if _now() > deadline:
                return None
            rr.reset()
            rr.integrator.seed = int(seed_base + rep)
            res = np.asarray(rr.simulate(t0, t_end, n_time), dtype=np.float64)
            if res.shape[0] != n_time or res.shape[1] != len(ids) + 1:
                return None
            vals[rep] = res[:, 1:]  # drop the leading time column
    except Exception:
        return None
    if not np.isfinite(vals).all():
        return None
    return t_grid, vals, names


if __name__ == "__main__":
    import sys

    if not roadrunner_available():
        print("UNAVAILABLE (libroadrunner not installed — pip install 'bngsim[roadrunner]')")
        sys.exit(2)
    net = Path(sys.argv[1])
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    n_rep = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    tg = np.linspace(0.0, t_end, 51)
    out = net_roadrunner_ensemble(net, tg, n_rep=n_rep, seed_base=1)
    if out is None:
        print("UNSCORED (.net not faithfully SBML-representable, non-uniform grid, or RR refused)")
        sys.exit(2)
    _, v, names = out
    print(f"supported: obs={names}")
    print(
        "final-time ensemble mean:",
        dict(zip(names, v[:, -1, :].mean(axis=0).round(3), strict=True)),
    )
