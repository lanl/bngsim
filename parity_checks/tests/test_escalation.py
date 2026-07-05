"""GH #185: direction-of-change seed-escalation for the stochastic parity verdict.

A borderline SSA/NF DIFF at the base replicate count can be finite-N sampling noise.
The runner accumulates replicates up ``ESCALATION_RUNGS`` and decides by the DIRECTION
of change, not a fixed ``frac_pass`` floor: if agreement improves with more replicates
it is noise (keep climbing to convergence → PASS); if it stalls or worsens it is a real
divergence (stop early). The old fixed ``ESCALATE_FLOOR = 0.95`` could not separate the
two because noise and real divergences overlap in ``frac_pass`` space — it left
converging-noise models (``m02_logistic`` ~0.75, ``S1987_B1_lotka_volterra`` ~0.94 at
N=10, both → 1.0 at N=100) as false DIFFs.

These tests pin the discriminator (:func:`bng_stoch_run._run_escalation`) directly,
driving it with scripted ``compare``/``escalate`` callables so no engine runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bng_stoch_run as bsr  # noqa: E402

PASS_FRAC = bsr.differ.ENSEMBLE_PASS_FRAC  # 0.99


def _ens(n_rep, n_time=3, n_obs=2):
    """A minimal ``(time, obs[n_rep,n_time,n_obs], names)`` ensemble triple."""
    t = np.arange(n_time, dtype=float)
    v = np.zeros((n_rep, n_time, n_obs), dtype=float)
    return (t, v, [f"Obs{i}" for i in range(n_obs)])


class _ScriptedCompare:
    """A ``compare(bn, leg)`` that returns a pre-scripted frac_pass per call.

    ``trajectory`` is the sequence of frac_pass values for the rungs AFTER the base
    (one per escalation step). Status is derived from the verdict bar, matching the
    real ``_compare_stoch``. Records how many times it was called.
    """

    def __init__(self, trajectory):
        self.traj = list(trajectory)
        self.calls = 0

    def __call__(self, bn, leg):
        fp = float(self.traj[self.calls])
        self.calls += 1
        status = "pass" if fp >= PASS_FRAC else "diff"
        return status, fp, f"frac_pass {fp:.3f}", "ensemble_frac_pass", PASS_FRAC


def _escalate_ok(delta, seed_base, rung):
    """A shape-matched ``escalate`` that returns ``delta`` more zero replicates."""
    return _ens(delta), _ens(delta)


def _drive(base_fp, trajectory, escalate=_escalate_ok):
    """Run ``_run_escalation`` from a base DIFF/PASS through a scripted trajectory."""
    bn, leg = _ens(10), _ens(10)
    base_status = "pass" if base_fp >= PASS_FRAC else "diff"
    cmp = _ScriptedCompare(trajectory)
    out = bsr._run_escalation(
        bn, leg, base_status, base_fp, "base comment", "ensemble_frac_pass", PASS_FRAC,
        n_rep_base=10, seed_base=1, compare=cmp, escalate=escalate,
    )
    status, value, comment, metric, m_tol, prog, stalled = out
    return {
        "status": status, "value": value, "comment": comment, "prog": prog,
        "stalled": stalled, "compare_calls": cmp.calls,
    }


# --------------------------------------------------------------------------- #
# Noise → converges (the GH #185 fix)
# --------------------------------------------------------------------------- #
def test_converging_noise_escalates_to_pass():
    # Improves 0.75 → 0.90 → 1.0: climbs both rungs and clears the bar.
    r = _drive(0.75, [0.90, 1.0])
    assert r["status"] == "pass"
    assert r["prog"] == [(10, 0.75), (30, 0.9), (100, 1.0)]
    assert r["stalled"] is False
    assert r["compare_calls"] == 2  # escalated to 30 then 100
    assert "sampling noise" in r["comment"] and "converged" in r["comment"]
    assert "frac_pass 1.000" in r["comment"]  # final rung's verdict comment threads through


def test_below_old_floor_still_escalates():
    # frac_pass 0.75 sits below the retired ESCALATE_FLOOR=0.95, yet a converging
    # DIFF must still escalate now — the exact false-DIFF the old floor produced.
    r = _drive(0.75, [0.95, 1.0])
    assert r["status"] == "pass"
    assert len(r["prog"]) == 3
    # And the retired constant is gone (so nothing can re-introduce the floor gate).
    assert not hasattr(bsr, "ESCALATE_FLOOR")


# --------------------------------------------------------------------------- #
# Real divergence → stalls/worsens, stops early
# --------------------------------------------------------------------------- #
def test_stalled_divergence_stops_after_one_rung():
    # 0.66 → 0.66: no improvement at N=30, so stop without burning N=100.
    r = _drive(0.66, [0.66])
    assert r["status"] == "diff"
    assert r["stalled"] is True
    assert r["prog"] == [(10, 0.66), (30, 0.66)]
    assert r["compare_calls"] == 1  # did NOT escalate to 100
    assert "real divergence" in r["comment"]


def test_worsening_divergence_stops_early():
    # 0.70 → 0.60: agreement drops with more replicates → real divergence, stop.
    r = _drive(0.70, [0.60])
    assert r["status"] == "diff"
    assert r["stalled"] is True
    assert r["compare_calls"] == 1


def test_stall_on_second_rung_tracks_prev_value():
    # Improves 0.70 → 0.85 (continue), then stalls 0.85 → 0.85 at N=100 (stop).
    r = _drive(0.70, [0.85, 0.85])
    assert r["status"] == "diff"
    assert r["stalled"] is True
    assert r["prog"] == [(10, 0.7), (30, 0.85), (100, 0.85)]
    assert r["compare_calls"] == 2


# --------------------------------------------------------------------------- #
# Improving but unresolved at the ceiling, and the no-op cases
# --------------------------------------------------------------------------- #
def test_improving_but_unresolved_at_ceiling_is_narrowing():
    # 0.70 → 0.80 → 0.90: keeps improving but never reaches PASS by the last rung.
    r = _drive(0.70, [0.80, 0.90])
    assert r["status"] == "diff"
    assert r["stalled"] is False  # ran out of rungs, did not stall
    assert len(r["prog"]) == 3
    assert "narrowed" in r["comment"] and "still divergent" in r["comment"]


def test_base_pass_does_not_escalate():
    r = _drive(1.0, [])
    assert r["status"] == "pass"
    assert r["prog"] == [(10, 1.0)]
    assert r["compare_calls"] == 0
    assert r["comment"] == "base comment"  # no lead prepended for a single rung


def test_grid_mismatch_breaks_without_crashing():
    # An escalate() that returns a differently-shaped ensemble is skipped, not fatal.
    def _bad_grid(delta, seed_base, rung):
        return _ens(delta, n_time=5), _ens(delta, n_time=5)  # n_time 5 != base 3

    r = _drive(0.75, [0.90], escalate=_bad_grid)
    assert r["status"] == "diff"  # never re-compared
    assert r["prog"] == [(10, 0.75)]
    assert r["compare_calls"] == 0
    assert "grid mismatch" in r["comment"]


def test_escalate_exception_is_caught():
    def _boom(delta, seed_base, rung):
        raise RuntimeError("network build failed")

    r = _drive(0.75, [0.90], escalate=_boom)
    assert r["status"] == "diff"
    assert r["compare_calls"] == 0
    assert "aborted" in r["comment"]
