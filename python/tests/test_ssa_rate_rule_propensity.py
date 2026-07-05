"""SSA with a rate rule that feeds a reaction propensity (GH #81).

An SBML *rate rule* ``dX/dt = f`` makes ``X`` a deterministic continuous
quantity. bngsim compiles it into the Functional reaction ``[] → [X]``; under
ODE the engine accumulates ``+f`` into ``derivs[X]``. Under *exact* SSA, the old
behavior fired this synthetic reaction as a stochastic birth/death channel — so
``X`` jumped by integer ±1 at Poisson times instead of moving smoothly. When ``X``
feeds another reaction's kinetic law (a species/parameter appearing in a rate
law), that mis-modeled ``X`` corrupts the dependent propensity. Concretely, a
time-varying decay ``dk/dt = c`` driving ``A → ∅`` at rate ``k·A`` reported
``A(10) ≈ 625`` instead of the correct ``≈ 82`` (``z ≈ 24``): ``k`` was ``0`` in
most replicates because its birth reaction had fired ~Poisson(0.5) times.

The fix (``ssa_simulator.cpp``, ``is_rate_rule_ode``) integrates rate-rule targets
deterministically (forward Euler) at each sub-step and keeps them out of the
stochastic selection — exactly as the variable-volume Tier-2 case needs.

Oracles
-------
* **Linear/affine propensity** ⇒ SSA mean == ODE == closed form. Used directly.
* **Nonlinear propensity** ⇒ SSA mean ≠ ODE; validated against an independent
  hand-rolled **Extrande** sampler (``_extrande_reference``), itself checked
  against the closed form. RoadRunner ``gillespie`` and COPASI's stochastic
  Time-Course both refuse rate rules, so neither is usable as a reference here.
"""

import math
import os
import sys

import bngsim
import numpy as np
import pytest

# Every model here is built via Model.from_antimony_string, which needs the
# optional `antimony` dependency (GH #153). It is excluded from the
# cibuildwheel test env, so skip the whole module when it is unavailable.
pytest.importorskip("antimony")

# Sibling helper import under pytest's importlib mode.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Synthetic model: time-varying first-order decay ───────────────────────────
# dk/dt = c  ⇒  k(t) = k0 + c·t ;  A → ∅ at mass-action rate k·A.
# Linear death ⇒ E[A(t)] solves dA/dt = -k(t)·A exactly, so the SSA mean equals
# the ODE / closed form  A(t) = A0·exp(-(k0·t + c·t²/2)).
_DECAY = """
model tv_decay
  compartment C = 1;
  species A in C = 1000;
  k = 0.0;
  c = 0.05;
  k' = c;
  J1: A => ; k*A;
end
"""
_A0, _C, _KDECAY = 1000.0, 0.05, 0.0
_T_END, _N_POINTS = 10.0, 11


def _closed_form_A(t):
    return _A0 * np.exp(-(_KDECAY * t + _C * t**2 / 2.0))


def _run_ssa_batch(antimony, reps, seed, t_end=_T_END, n_points=_N_POINTS):
    model = bngsim.Model.from_antimony_string(antimony)
    sim = bngsim.Simulator(model, method="ssa")
    results = sim.run_batch(
        t_span=(0.0, t_end), n_points=n_points, params=[{} for _ in range(reps)], seed=seed
    )
    names = list(results[0].species_names) if hasattr(results[0], "species_names") else None
    arr = np.stack([np.asarray(r.species) for r in results], axis=0)  # reps × N × nsp
    if names is None:
        names = list(bngsim.Model.from_antimony_string(antimony).species_names)
    return arr, names


def _ode(antimony, t_end=_T_END, n_points=_N_POINTS):
    model = bngsim.Model.from_antimony_string(antimony)
    return np.asarray(model.species_names), np.asarray(
        bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=n_points).species
    )


def test_decay_ssa_mean_tracks_closed_form():
    """SSA mean of A tracks the closed-form within 5·SE at every sample t>0."""
    reps, seed = 300, 20260610
    arr, names = _run_ssa_batch(_DECAY, reps, seed)
    ai = names.index("A")
    mean = arr[:, :, ai].mean(0)
    se = arr[:, :, ai].std(0, ddof=1) / math.sqrt(reps)
    t = np.linspace(0.0, _T_END, _N_POINTS)
    closed = _closed_form_A(t)
    for i in range(1, _N_POINTS):
        z = abs(mean[i] - closed[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"A SSA mean at t={t[i]:.1f} = {mean[i]:.2f} vs closed {closed[i]:.2f} "
            f"(z={z:.2f}); rate-rule-driven decay not tracked."
        )


def test_rate_rule_target_integrated_deterministically():
    """The parameter target k is deterministic: ~zero variance, equals k0 + c·t.

    A rate rule whose RHS reads only constants/time must produce the *same*
    trajectory in every replicate — the signature that it is integrated as an ODE,
    not sampled as a birth/death channel (the pre-fix behavior gave integer k with
    large variance, and recording at the sub-step left endpoint gave dt_max-scale
    replicate jitter; the exact-at-sample-time recording removes both). For a
    constant derivative forward Euler is exact, so k matches k0 + c·t to ~FP.
    """
    reps, seed = 64, 11
    arr, names = _run_ssa_batch(_DECAY, reps, seed)
    ki = names.index("k")
    t = np.linspace(0.0, _T_END, _N_POINTS)
    k = arr[:, :, ki]
    assert k.std(0).max() < 1e-9, "rate-rule target k is not deterministic across replicates"
    np.testing.assert_allclose(k.mean(0), _KDECAY + _C * t, atol=1e-9)


def test_pre_fix_regression_guard():
    """A frozen-k regression would leave A near A0; assert it actually decays.

    Pre-fix, k stayed 0 in most replicates so A(10) ≈ 625 (≈ 0.62·A0). The fixed
    engine gives A(10) ≈ 82. Guard with a wide margin so only a true regression
    (k frozen / fired as a channel) trips it, not stochastic noise.
    """
    reps, seed = 200, 5
    arr, names = _run_ssa_batch(_DECAY, reps, seed)
    ai = names.index("A")
    a_end = arr[:, -1, ai].mean()
    closed_end = _closed_form_A(_T_END)  # ≈ 82
    assert a_end < 0.25 * _A0, (
        f"A(T) mean = {a_end:.1f} (> 0.25·A0); the rate-rule decay looks frozen — "
        "k is likely being fired as a stochastic channel again."
    )
    assert a_end == pytest.approx(closed_end, rel=0.25)


# ── Species (not parameter) rate rule feeding a propensity ────────────────────
# Mirrors the corpus shape (BIOMD207/556/1025: a rate-rule *species* feeds a
# kinetic law). dS/dt = a - b·S drives S to a moving setpoint; S then sets the
# birth rate of A (A → ∅ death). Both linear ⇒ SSA means == ODE.
_SPECIES_RR = """
model species_rr_feed
  compartment C = 1;
  species S in C = 0;
  species A in C = 0;
  a = 2.0; b = 0.4; kd = 0.3;
  S' = a - b*S;
  J_birth: => A; S;
  J_death: A => ; kd*A;
end
"""


def test_species_rate_rule_feeds_propensity_tracks_ode():
    """A rate-rule *species* feeding a propensity: stochastic mean tracks ODE.

    S' = a - b·S is a rate rule whose target then sets A's birth rate. S has no
    stochastic input, so it is *essentially* deterministic and tracks the ODE
    within the forward-Euler truncation error O(dt_max) — it is integrated by
    Euler at dt_max = horizon/1000, not by CVODE, so a *tolerance* (not a z-test)
    is the right check. Because the rate rule is state-dependent (b·S) and the
    Euler grid is the fire-time grid (replicate-dependent), S carries a *tiny*
    cross-replicate variance ~O(dt_max · ∂); we assert it is ≪ the birth-death
    noise the pre-fix path produced (O(√S)). The stochastic A mean tracks the ODE
    within 5·SE.
    """
    t_end, npts = 20.0, 11
    reps, seed = 300, 99
    arr, names = _run_ssa_batch(_SPECIES_RR, reps, seed, t_end=t_end, n_points=npts)
    onames, ode = _ode(_SPECIES_RR, t_end=t_end, n_points=npts)
    onames = list(onames)

    # Target S: essentially deterministic (≪ birth-death noise), tracks ODE within
    # the Euler tolerance. A birth-death S would have std ~√S ≈ 2 here.
    sj, soj = names.index("S"), onames.index("S")
    assert arr[:, :, sj].std(0).max() < 1e-3, "rate-rule species S carries birth-death-scale noise"
    np.testing.assert_allclose(arr[:, :, sj].mean(0), ode[:, soj], rtol=0.02, atol=0.03)

    # Stochastic A: mean tracks ODE within 5·SE.
    aj, aoj = names.index("A"), onames.index("A")
    mean = arr[:, :, aj].mean(0)
    se = arr[:, :, aj].std(0, ddof=1) / math.sqrt(reps)
    for i in range(1, npts):
        z = abs(mean[i] - ode[i, aoj]) / (se[i] + 1e-12)
        assert z < 5.0, f"A at sample {i}: SSA {mean[i]:.3f} vs ODE {ode[i, aoj]:.3f} (z={z:.2f})"


# ── Nonlinear propensity: agree with the independent Extrande oracle ──────────
_NL = """
model nl_quadratic
  compartment C = 1;
  species A in C = 25;
  k = 0.0; c = 0.0015;
  k' = c;
  J1: A => ; k*A*(A-1);
end
"""


def test_nonlinear_propensity_matches_extrande():
    """Rate rule k(t) feeding a nonlinear propensity k·A·(A-1).

    SSA mean ≠ ODE mean here (the rate is quadratic in the fluctuating discrete
    A). bngsim's sub-step SSA must agree with the independent Extrande sampler,
    which uses a different algorithm (rejection thinning) and is validated against
    the closed form elsewhere. This is the load-bearing correctness check for the
    case where no ODE oracle exists.
    """
    pytest.importorskip("numpy")
    from _extrande_reference import ReactionSpec, RefModel, simulate_batch

    reps, seed = 250, 31
    t = np.linspace(0.0, 10.0, 11)
    arr, names = _run_ssa_batch(_NL, reps, seed, t_end=10.0, n_points=11)
    ai = names.index("A")
    bs = arr[:, :, ai]
    bs_mean = bs.mean(0)
    bs_se = bs.std(0, ddof=1) / math.sqrt(reps)

    c = 0.0015
    ref = RefModel(
        species=["A"],
        x0={"A": 25.0},
        reactions=[ReactionSpec({"A": -1}, lambda s: s["k"] * s["A"] * (s["A"] - 1.0))],
        cont={"k": lambda s: c},
        c0={"k": 0.0},
    )
    ex = simulate_batch(ref, t, reps, seed + 1, look_ahead=10.0 / 600.0, bound_grid=6)[:, :, 0]
    ex_mean = ex.mean(0)
    ex_se = ex.std(0, ddof=1) / math.sqrt(reps)

    for i in range(1, 11):
        se = math.sqrt(bs_se[i] ** 2 + ex_se[i] ** 2)
        z = abs(bs_mean[i] - ex_mean[i]) / (se + 1e-12)
        assert z < 5.0, (
            f"sample {i}: bngsim-SSA {bs_mean[i]:.3f} vs Extrande {ex_mean[i]:.3f} (z={z:.2f}); "
            "the two independent stochastic engines disagree."
        )
