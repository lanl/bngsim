#!/usr/bin/env python3
"""bngsim's analytic forward sensitivities against central differences of its
own trajectories -> a ``_core`` report (issue #804).

The nightly checks forward sensitivities only against AMICI, and only on SBML.
A ``.net`` model has no sensitivity oracle at all (BNG2.pl computes none), and
when bngsim and AMICI agree nothing adjudicates. A finite difference of bngsim's
own state runs is an oracle independent of both, for any model bngsim can
integrate.

One job per (model, sensitivity method), both CVODES correctors (``staggered``
and ``simultaneous``), as ``amici_sens`` runs them:

    analytic and FD tensors within tolerance ...... PASS
    over tolerance, or a non-finite cell .......... DIFF
    bngsim DECLARED it cannot differentiate ........ UNSUPPORTED (by type)
    bngsim raised .................................. EXCEPTION
    a plain run of the difference raised, or the
      runs would not fit in the cap ................ REFERENCE_FAILED
    no parameter to differentiate, or the FD
      reference did not converge with its step .... BAD_TEST (nothing compared)
    wall-clock cap exceeded ........................ TIMEOUT

The reference
-------------
For each parameter ``p`` (value ``v``), central differences at two steps
``h`` and ``h/2`` (``h = FD_REL_STEP·|v|``, or ``FD_REL_STEP`` at ``v = 0``),
each from two plain runs at ``FD_RTOL`` / ``FD_ATOL``, far tighter than the
analytic run's, combined by Richardson extrapolation: ``(4·D(h/2) - D(h))/3``,
error O(h⁴). The two traps ``test_time_switch_sens_fd_reference.py`` (#368)
documents are what that guards against:

* **An unresolvable step.** A parameter that barely moves a species puts the
  difference under the solver's own floor, and the quotient is noise over 2h.
  The step is relative (``FD_REL_STEP``, coarse on purpose) and the runs are at
  ``FD_RTOL`` = 1e-12. A column is compared only where its analytic value or
  its difference clears both the verdict's ``atol/|p|`` floor and the
  difference's own noise (:func:`judgeable_columns`); the rest are left out and
  named on the row. Wherever the verdict looks, the two steps' quotients must
  agree to ``--fd-tol`` on each (species, parameter) column's own scale
  (:func:`step_disagreement`): one whose steps disagree is not a reference, and
  the row is BAD_TEST with the disagreement recorded rather than a DIFF blamed
  on the analytic side. A row with no column to compare is BAD_TEST too.
* **A switch node.** A central difference taken exactly at a time where a
  switch fires straddles it and reads half the one-sided derivative. The
  manifest keeps such models' sample grids off their switch times.

The two sides do not share the code under test. bngsim computes forward
sensitivities only through generated code (the interpreted path was retired,
GH #214), so every analytic run is compiled, which is the path #689 broke; the
plain runs of the difference are pinned to the ExprTk interpreter.

The verdict is ``_amici_sens.sens_verdict``, ``_core.differ`` with the
``atol/|p|`` noise floor, the same oracle and the same floor ``amici_sens``
uses, so a tolerance change lands in both. Two things differ, both for the
reference's sake (:func:`column_verdicts`): each parameter column is judged on
its own scale, not against the tensor's largest, and the cells inside the
difference's own noise or step error are forgiven. ``timing.bngsim`` carries the
analytic run's work counters for the nightly efficiency check, and its
``backend`` reads ``cc+dq`` where bngsim declined the analytic sensitivity RHS.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "amici_parity"))

import _amici_sens as asens  # noqa: E402
from _core import (  # noqa: E402
    JobResult,
    Outcome,
    capture_exception,
    tally,
    versions,
    write_report,
)
from _core.versions import git_rev  # noqa: E402

JOBS = HERE / "fd_sens_jobs.json"
SENS_METHODS = asens.SENS_METHODS

# The analytic run's tolerances: the suite's (amici_sens's) pair.
DEFAULT_RTOL = asens.DEFAULT_RTOL
DEFAULT_ATOL = asens.DEFAULT_ATOL
# The finite-difference runs: tight enough that a relative step of 1e-3 leaves
# the difference six or more orders above their own noise.
FD_RTOL = 1e-12
FD_ATOL = 1e-16
FD_REL_STEP = 1e-3
# Most parameters differenced per model: each costs four plain runs.
DEFAULT_PARAM_CAP = 6
DEFAULT_TIMEOUT = 300.0
REFERENCE_ENGINE = "finite_difference"
# The share of a job's cap its plain runs may run up to: the scheduler's clock
# starts at the child's spawn, before the worker's.
CAP_SHARE = 0.9


def load_model(path: Path):
    import bngsim

    if path.suffix == ".net":
        return bngsim.Model.from_net(str(path))
    return bngsim.Model.from_sbml(str(path))


def candidate_params(model) -> list[str]:
    """The parameters a column can be taken for: the model's knobs, less those a
    sensitivity is refused for by construction (compartment sizes, frozen
    parameters)."""
    excluded = set(getattr(model, "compartment_size_params", None) or ())
    excluded |= set(getattr(model, "frozen_params", None) or ())
    return [p for p in model.primary_param_names if p not in excluded]


def fd_step(value: float, rel: float = FD_REL_STEP) -> float:
    """The larger of the two central-difference steps for a parameter at ``value``."""
    return rel * abs(value) if value != 0.0 else rel


def richardson(d_h: np.ndarray, d_h2: np.ndarray) -> np.ndarray:
    """Central differences at ``h`` and ``h/2`` combined to O(h⁴)."""
    return (4.0 * d_h2 - d_h) / 3.0


def step_disagreement(
    analytic: np.ndarray, reference: np.ndarray, gap: np.ndarray, settled=None
) -> float:
    """How far the two steps' quotients are apart in one parameter's column, on
    the scale that column's verdict judges.

    The arrays are ``(n_t, n_sp)``, one parameter's column of each tensor, and
    ``gap`` is ``|D(h) - D(h/2)|`` cell by cell. :func:`column_verdicts` judges
    each species against its own peak over time, the peak being the larger of the
    two sides', and counts a species where that peak is past
    ``_core.differ.SIGNIF_FLOOR`` of the column's, or where ``differ``'s movement
    gate (issue #873) counts it. This reads ``max |D(h) - D(h/2)| / peak`` over
    exactly those species. ``settled`` cells, the ones inside a noise floor, which
    the verdict forgives whatever the gap, are left out, so where the two sides
    agree within the floors everywhere this reads 0: the gate guards a verdict
    from blaming an unsure reference, and there is nothing to blame. (The row's
    ``fd_step_disagreement_raw`` reads every cell.) A converged reference reads
    O(h²); a step lost in the solver's noise reads O(1).

    Per species, not per parameter: a small species whose difference is noise
    sits beside a large one in the same column, so the column's peak hides it.
    Peak-relative, not per cell: a cell near zero in an otherwise large column
    read 1e-2 on references that converge (nested_derived_rate_const,
    expr_sens_chain). The movement gate's species matter where the largest one
    holds still (mm_tqssa_stiff's dE/dE_0 = 1): that gate then takes its scale
    from the species that move, which can be a difference's noise.

    A reference with a non-finite cell is no reference, and reads ``inf``; a
    non-finite analytic cell is the verdict's to fail and sets no scale here.
    """
    from _core import differ

    a = np.asarray(analytic, dtype=float)
    r = np.asarray(reference, dtype=float)
    g = np.asarray(gap, dtype=float)
    if not (np.isfinite(r).all() and np.isfinite(g).all()):
        return float("inf")
    if r.size == 0:
        return 0.0
    peak = np.maximum(np.where(np.isfinite(a), np.abs(a), 0.0), np.abs(r)).max(axis=0)
    scale = float(peak.max())
    if scale == 0.0:
        return 0.0 if not np.any(g) else float("inf")
    live = peak > differ.SIGNIF_FLOOR * scale
    ref_range, ref_peak = differ._range_and_peak(r)
    eng_range, eng_peak = differ._range_and_peak(a)
    col_range = np.maximum(ref_range, eng_range)
    moves = col_range > differ.REL_TOL * np.maximum(ref_peak, eng_peak)
    if moves.any():
        live |= moves & (col_range > differ.SIGNIF_FLOOR * col_range[moves].max())
    looked = g if settled is None else np.where(settled, 0.0, g)
    return float(np.max(looked[:, live].max(axis=0) / peak[live]))


def _plain_species(path: Path, param: str, value: float, job: dict) -> np.ndarray:
    import bngsim

    model = load_model(path)
    model.set_param(param, value)
    model.reset()
    run = bngsim.Simulator(model, method="ode", codegen=False).run(
        t_span=(job["t_start"], job["t_end"]),
        n_points=job["n_points"],
        rtol=FD_RTOL,
        atol=FD_ATOL,
        max_steps=job.get("max_steps", 100_000),
    )
    return np.asarray(run.species, dtype=float)


class ReferenceOverBudget(RuntimeError):
    """The difference's plain runs would not finish inside the job's cap."""


def fd_tensor(path: Path, params: list[str], values: list[float], job: dict, deadline=None):
    """The Richardson-extrapolated FD tensor ``(n_t, n_sp, n_p)`` and, shaped
    like it, ``|D(h) - D(h/2)|``, the gap :func:`step_disagreement` reads.

    ``deadline`` (a ``time.perf_counter()`` reading) stops the plain runs before
    one that the longest so far says would end past it, so a reference too slow
    for the job's cap is that reference's failure, not a TIMEOUT scored against
    bngsim. A single plain run longer than the whole cap still is one.
    """
    cols, gaps = [], []
    n_runs, n_done, longest = 4 * len(params), 0, 0.0
    for p, v in zip(params, values, strict=True):
        h = fd_step(v)
        quotients = []
        for step in (h, h / 2.0):
            pair = []
            for value in (v + step, v - step):
                if deadline is not None and time.perf_counter() + longest > deadline:
                    raise ReferenceOverBudget(
                        f"the difference's plain runs would run past the job's cap: "
                        f"{n_done} of {n_runs} done, the longest {longest:.1f} s"
                    )
                t = time.perf_counter()
                pair.append(_plain_species(path, p, value, job))
                longest = max(longest, time.perf_counter() - t)
                n_done += 1
            quotients.append((pair[0] - pair[1]) / (2.0 * step))
        d_h, d_h2 = quotients
        gaps.append(np.abs(d_h - d_h2))
        cols.append(richardson(d_h, d_h2))
    return np.stack(cols, axis=-1), np.stack(gaps, axis=-1)


# How many multiples of a finite difference's own noise a column's peak must
# exceed before the difference is a reference for it. A cell whose two sides are
# closer than that noise is one the difference cannot adjudicate.
FD_NOISE_FACTOR = 10.0
# A cell no further from the difference than this many times its (species,
# parameter) column's largest step gap |D(h) - D(h/2)| is inside the error that
# gap estimates for the reference, and is not the analytic side's to fail.
FD_UNCERTAINTY_FACTOR = 3.0
# How far a column's peak must clear each floor to be judged: 1 / differ's
# HARD_REL_CEILING (5%). The verdict forgives every cell inside a floor, so a
# column only just over one would have all of its disagreement forgiven, and a
# PASS on it would say nothing; at 20x, what a floor forgives stays under what
# the verdict calls real. wofsy_goldstein's K11 (peak 2.2x its difference's
# floor) passed a 30% error, and nanomolar_dimer (2.5x the atol/|p| floor) 40%.
JUDGE_FACTOR = 20.0


def fd_noise_floor(x_max: float, value: float) -> float:
    """The smallest sensitivity a central difference resolves for a parameter at
    ``value``: the plain runs resolve a state to ``FD_RTOL·max|x| + FD_ATOL``, so
    a quotient over the smaller step ``h/2`` resolves ``FD_NOISE_FACTOR`` times
    that over ``h/2``."""
    return FD_NOISE_FACTOR * (FD_RTOL * x_max + FD_ATOL) / (fd_step(value) / 2.0)


def _column_peaks(tensor: np.ndarray) -> np.ndarray:
    """Per parameter, the peak ``|value|`` over (time, species); a column holding
    a non-finite cell peaks at ``inf``."""
    t = np.asarray(tensor, dtype=float)
    if t.size == 0:
        return np.zeros(t.shape[-1] if t.ndim else 0)
    return np.where(np.isfinite(t), np.abs(t), np.inf).max(axis=(0, 1))


def judgeable_columns(
    analytic: np.ndarray, reference: np.ndarray, x_max: float, values, atol: float
) -> list[int]:
    """The parameter columns a finite difference can judge.

    Two floors, and a column's peak, the larger of the analytic one and the
    difference's, has to clear both by ``JUDGE_FACTOR``:

    * the verdict's own, ``factor·atol/|p|`` (``_amici_sens.sens_resolution_floors``):
      below it the analytic run itself does not resolve the column, and the
      verdict forgives its cells anyway;
    * the difference's, :func:`fd_noise_floor`.

    A column under either is one neither side can speak for, and is left out
    of the comparison (counted on the row). Measured: erk_activation's q3 (value
    3e-4, step 1.5e-7) and mm_tqssa_stiff's Km (1e-8) have analytic peaks of
    5e-8 and 8e-4 under difference noise of order 1e-1 and 1e3, and their
    differences' two steps were 50% and 80% apart.

    Either side's peak counts, so a column one side gets wrong is judged
    whichever way it is wrong. One wrongly LARGE analytically clears the floors
    on the analytic peak; one wrongly ZERO analytically, the silent zero, clears
    them on the difference's, since a difference above its own noise floor is
    signal. A column with a non-finite cell on either side peaks at ``inf`` and
    is always judged: the verdict fails a non-finite analytic cell, and the gate
    refuses a non-finite reference.
    """
    floors = asens.sens_resolution_floors(values, atol)
    peaks = np.maximum(_column_peaks(analytic), _column_peaks(reference))
    return [
        j
        for j, v in enumerate(values)
        if peaks[j] > JUDGE_FACTOR * max(floors[j], fd_noise_floor(x_max, v))
    ]


def reference_settled(analytic, reference, x_max: float, values) -> np.ndarray:
    """The cells inside the difference's noise floor: ``|analytic - FD|`` within
    :func:`fd_noise_floor` for the cell's parameter. ``(n_t, n_sp, n_p)``."""
    floors = np.array([fd_noise_floor(x_max, v) for v in values])
    return np.abs(analytic - reference) <= floors[np.newaxis, np.newaxis, :]


def reference_uncertain(analytic, reference, gap) -> np.ndarray:
    """The cells inside the reference's own error: ``|analytic - FD|`` within
    ``FD_UNCERTAINTY_FACTOR`` times the largest step gap of the cell's (species,
    parameter) column. ``(n_t, n_sp, n_p)``."""
    return np.abs(analytic - reference) <= FD_UNCERTAINTY_FACTOR * gap.max(axis=0, keepdims=True)


def column_verdicts(analytic, reference, gap, x_max: float, values, atol: float, judged):
    """``sens_verdict`` for each judged parameter column on its own.

    A parameter column's sensitivities are in that parameter's units, so one
    tensor-wide scale judges every column against the largest one: ``differ``'s
    significance gate counts a (species, parameter) column only past 1e-3 of the
    tensor's peak, and in 18 of 44 manifest models 49 of 159 judged
    parameters sat under that, where zeroing or negating an analytic column still
    read PASS.
    Each column is therefore judged alone, scaled to a peak of 1 (so ``differ``'s
    absolute 1e-12 underflow floor is relative too: nanomolar_dimer's tensor
    peaks at 6e-20), with ``atol`` scaled alike for the ``atol/|p|`` floor.

    On top of that floor, the cells the difference cannot adjudicate are
    forgiven: inside its noise (:func:`reference_settled`) or its step error
    (:func:`reference_uncertain`). A DIFF therefore rests only on cells where the
    reference is sure of itself.
    """
    unsure = reference_settled(analytic, reference, x_max, values) | reference_uncertain(
        analytic, reference, gap
    )
    out = {}
    for j in judged:
        a, r = analytic[..., [j]], reference[..., [j]]
        mags = np.abs(np.concatenate([a.ravel(), r.ravel()]))
        mags = mags[np.isfinite(mags)]
        peak = float(mags.max()) if mags.size else 0.0
        c = 1.0 / peak if peak > 0.0 else 1.0
        out[j] = asens.sens_verdict(
            a * c, r * c, param_values=[values[j]], atol=atol * c, forgive_mask=unsure[..., [j]]
        )
    return out


def _judge(res: dict, analytic, reference, gap, species, values, spec: dict) -> None:
    """The verdict on two same-shaped tensors, into ``res``: which columns are
    judged, whether the reference converged where the verdict looks, then
    :func:`column_verdicts` over the judged columns."""
    x = np.abs(species)
    finite = x[np.isfinite(x)]
    x_max = float(finite.max()) if finite.size else 0.0
    atol = spec["atol"]
    judged = judgeable_columns(analytic, reference, x_max, values, atol)
    settled = asens.sensitivity_noise_mask(analytic, reference, values, atol) | reference_settled(
        analytic, reference, x_max, values
    )
    disagreement = max(
        (
            step_disagreement(analytic[..., j], reference[..., j], gap[..., j], settled[..., j])
            for j in judged
        ),
        default=0.0,
    )
    params = res["params"]
    res["fd_step_disagreement"] = disagreement
    # The same over every cell, settled ones too: what the step did, whatever
    # the verdict forgives.
    res["fd_step_disagreement_raw"] = max(
        (step_disagreement(analytic[..., j], reference[..., j], gap[..., j]) for j in judged),
        default=0.0,
    )
    res["n_columns_judged"] = len(judged)
    res["columns_not_judged"] = [p for j, p in enumerate(params) if j not in judged]
    if not judged:
        res.update(
            status="bad_test",
            why="no column carries a sensitivity above the solver's resolution: "
            "the comparison would establish nothing",
        )
    elif not np.isfinite(disagreement):
        res.update(
            status="bad_test",
            why="the FD reference did not converge with its step: it is not finite",
        )
    elif not disagreement <= spec["fd_tol"]:
        res.update(
            status="bad_test",
            why=(
                f"the FD reference did not converge with its step: the two steps' "
                f"quotients differ by {disagreement:.2e} relative (> {spec['fd_tol']:g})"
            ),
        )
    else:
        verdicts = column_verdicts(analytic, reference, gap, x_max, values, atol, judged)
        failing = [params[j] for j, v in verdicts.items() if not v["passed"]]
        res.update(
            status="diff" if failing else "pass",
            value=max(float(v["max_rel"]) for v in verdicts.values()),
            verdict={
                "n_below_noise_floor": sum(v["n_below_noise_floor"] for v in verdicts.values()),
                "n_noise_rescued": sum(v["n_noise_rescued"] for v in verdicts.values()),
                "noise_decisive": not failing
                and any(v["noise_decisive"] for v in verdicts.values()),
            },
        )
        if failing:
            res["columns_diff"] = failing


def _worker(spec: dict, q) -> None:
    """One (model, method) job in a child process; puts one result dict on ``q``."""
    t0 = time.perf_counter()
    res: dict = {"key": spec["key"]}
    try:
        import bngsim

        path = REPO / spec["path"]
        model = load_model(path)
        cands = candidate_params(model)
        params = asens.select_params(cands, spec["param_cap"])
        res["n_param_candidates"] = len(cands)
        if not params:
            res.update(status="bad_test", why="no parameter to differentiate")
            q.put(res)
            return
        values = [float(model.get_param(p)) for p in params]
        try:
            sim = bngsim.Simulator(
                model,
                method="ode",
                sensitivity_params=list(params),
                sensitivity_method=spec["sens_method"],
            )
            t1 = time.perf_counter()
            run = sim.run(
                t_span=(spec["t_start"], spec["t_end"]),
                n_points=spec["n_points"],
                rtol=spec["rtol"],
                atol=spec["atol"],
                max_steps=spec.get("max_steps", 100_000),
            )
            integrate_sec = time.perf_counter() - t1
        except bngsim.SensitivityUnsupportedError as e:
            res.update(status="unsupported", why=str(e)[:400], exception_class=type(e).__name__)
            q.put(res)
            return
        analytic = np.asarray(run.sensitivities, dtype=float)
        stats = run.solver_stats if hasattr(run, "solver_stats") else {}
        from _core.work import work_counters

        # A model bngsim declines the analytic sensitivity RHS for runs on CVODES'
        # difference quotient (issue #438): then this is a difference against a
        # difference, and the code under test never ran. The backend says so,
        # and --expect-backend cc alerts on it.
        analytic_rhs = bool(sim.has_analytic_sens_rhs)
        reason = None if analytic_rhs else sim.sens_rhs_decline_reason
        if reason is not None:
            res["sens_rhs_decline_reason"] = str(reason)[:400]
        res["timing"] = {
            "bngsim": {
                # One cold solve: the phase name verdicts.py sums into bngsim_sec.
                "integrate_cold_sec": round(integrate_sec, 6),
                "backend": sim.codegen_backend if analytic_rhs else f"{sim.codegen_backend}+dq",
                "config": {
                    "codegen": sim.codegen_backend,
                    "jacobian": sim.jacobian_strategy,
                    "sens_method": spec["sens_method"],
                    "sens_rhs": "analytic" if analytic_rhs else "difference_quotient",
                },
                "work": work_counters(stats),
            }
        }
        res["params"] = params
        res["max_abs_sx"] = float(np.nanmax(np.abs(analytic))) if analytic.size else 0.0
        res["resolvable_columns"] = asens.resolvable_param_columns(analytic, values, spec["atol"])
        try:
            deadline = t0 + CAP_SHARE * spec["timeout"]
            reference, gap = fd_tensor(path, params, values, spec, deadline=deadline)
        except Exception as e:  # noqa: BLE001 - the reference failed, not the code under test
            captured = capture_exception(REFERENCE_ENGINE, e)
            res.update(
                status="reference_failed",
                exception=captured.text,
                exception_class=captured.cls,
                wall_sec=round(time.perf_counter() - t0, 3),
            )
            q.put(res)
            return
        if analytic.shape != reference.shape:
            res.update(
                status="diff",
                value=float("inf"),
                why=f"tensor shapes differ: analytic {analytic.shape}, FD {reference.shape}",
            )
        else:
            species = np.asarray(run.species, dtype=float)
            _judge(res, analytic, reference, gap, species, values, spec)
    except Exception as e:  # noqa: BLE001 - one model's failure is that row's EXCEPTION
        captured = capture_exception("bngsim", e)
        res.update(status="exception", exception=captured.text, exception_class=captured.cls)
    res["wall_sec"] = round(time.perf_counter() - t0, 3)
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


def make_specs(jobs: list[dict], args) -> list[dict]:
    specs = []
    for job in jobs:
        for method in SENS_METHODS:
            specs.append(
                {
                    "key": f"{job['model_id']}|sens/{method}",
                    "model_id": job["model_id"],
                    "path": job["path"],
                    "t_start": float(job.get("t_start", 0.0)),
                    "t_end": float(job["t_end"]),
                    "n_points": int(job.get("n_points", 51)),
                    "max_steps": int(job.get("max_steps", 100_000)),
                    "rtol": args.rtol,
                    # A model whose states sit far from the suite's scale says so:
                    # atol is absolute, and the atol/|p| floor with it.
                    "atol": float(job.get("atol", args.atol)),
                    "fd_tol": args.fd_tol,
                    "param_cap": int(job.get("param_cap", args.param_cap)),
                    "sens_method": method,
                    "timeout": float(job.get("timeout", args.timeout)),
                }
            )
    return specs


def _note(r: dict) -> str:
    """What a row's note says. The scheduler's own results carry no ``why``: a
    timeout says its cap, a child that died without a result its exit code."""
    if r.get("exception") or r.get("why"):
        return r.get("exception") or r.get("why")
    if r.get("status") == "timeout":
        return f"over the {r.get('cap')} s per-job cap"
    if r.get("status") == "dead":
        return f"the job's process died without a result (exit code {r.get('exitcode')})"
    return ""


def to_result(spec: dict, r: dict, stamp: dict) -> JobResult:
    outcome = _OUTCOME.get(r.get("status"), Outcome.EXCEPTION)
    extra = {
        "sens_method": spec["sens_method"],
        "params": r.get("params"),
        "n_param_candidates": r.get("n_param_candidates"),
        "fd_step_disagreement": r.get("fd_step_disagreement"),
        "fd_step_disagreement_raw": r.get("fd_step_disagreement_raw"),
        "atol": spec.get("atol"),
        "n_columns_judged": r.get("n_columns_judged"),
        "columns_not_judged": r.get("columns_not_judged"),
        "max_abs_sx": r.get("max_abs_sx"),
        "resolvable_columns": r.get("resolvable_columns"),
        "columns_diff": r.get("columns_diff"),
        "sens_rhs_decline_reason": r.get("sens_rhs_decline_reason"),
        **(r.get("verdict") or {}),
    }
    return JobResult(
        model_id=spec["model_id"],
        method=f"sens/{spec['sens_method']}",
        reference_engine=REFERENCE_ENGINE,
        outcome=outcome.value,
        metric="max_rel_err",
        value=r.get("value"),
        tol=None,
        exception=_note(r),
        exception_class=r.get("exception_class"),
        wall_sec=r.get("wall_sec", r.get("wall")),
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        versions=stamp,
        comment="",
        timing=r.get("timing"),
        extra={k: v for k, v in extra.items() if v is not None},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", default=str(JOBS), help="the committed manifest")
    ap.add_argument("--out", default=str(HERE / "runs" / "report_fd_sens.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per job (s)")
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help="the analytic run's")
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL, help="the analytic run's")
    ap.add_argument(
        "--fd-tol",
        type=float,
        default=1e-3,
        help="how far apart the two steps' quotients may be before the reference "
        "is declared unconverged (BAD_TEST)",
    )
    ap.add_argument("--param-cap", type=int, default=DEFAULT_PARAM_CAP)
    ap.add_argument("--models", default="", help="comma-separated model_id substrings")
    args = ap.parse_args(argv)

    jobs = json.loads(Path(args.jobs).read_text())["jobs"]
    if args.models:
        subs = [s.strip() for s in args.models.split(",") if s.strip()]
        jobs = [j for j in jobs if any(s in j["model_id"] for s in subs)]
    specs = make_specs(jobs, args)
    print(f"fd_sens: {len(jobs)} models x {len(SENS_METHODS)} methods = {len(specs)} jobs")
    t0 = time.perf_counter()

    def on_done(finished, total, res):
        print(f"  [{finished}/{total}] {res.get('status', '?'):11} {res['key']}", flush=True)

    raw = asens.schedule(
        specs, _worker, workers=args.workers, timeout_of=lambda s: s["timeout"], on_done=on_done
    )
    stamp = versions.stamp(REFERENCE_ENGINE)
    results = [to_result(spec, r, stamp) for spec, r in zip(specs, raw, strict=False)]
    meta = {
        "suite": "fd_sens",
        "reference_engine": REFERENCE_ENGINE,
        "regime": "forward_sensitivity",
        "git_rev": git_rev(str(HERE)),
        "versions": stamp,
        "tally": tally(r.outcome for r in results),
        "n_jobs": len(specs),
        "n_models": len(jobs),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "config": {
            "rtol": args.rtol,
            "atol": args.atol,
            "fd_rtol": FD_RTOL,
            "fd_atol": FD_ATOL,
            "fd_rel_step": FD_REL_STEP,
            "fd_tol": args.fd_tol,
            "param_cap": args.param_cap,
        },
        "hardware": asens.hardware_info(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, results, meta=meta)
    print(f"report: {out}  {meta['tally']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
