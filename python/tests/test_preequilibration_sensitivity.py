"""Pre-equilibration / steady-state output sensitivities (GH #210).

A two-phase pre-equilibration (ADR-0052) equilibrates to steady state under a
pre-condition (unmeasured), then perturbs and measures — running the same
persistent ``Simulator`` across two ``run()`` calls with NO reset between them,
so the equilibration steady state x_ss(θ) is the measurement phase's initial
condition. The measurement phase's forward-sensitivity seed must therefore be
the steady-state sensitivity dx_ss/dθ, NOT the fresh-start zero.

This module verifies the two halves of the definition of done:

  * **Correct** — with ``carry_sensitivities=True`` the measurement-phase
    output sensitivity matches central finite differences taken over the FULL
    two-phase run (and the t=0 seed equals the closed-form dx_ss/dθ).
  * **Loudly unsupported** — requesting sensitivities on a carried-over state
    *without* the opt-in raises (no silent wrong derivatives), as do the
    seed-missing, param-mismatch, IC-axis, and wrong-method cases.
"""

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# Production / degradation with a condition switch (extra_deg). See the .net
# header: dA/dt = k_prod - (k_deg + extra_deg)*A, so the phase-1 (extra_deg=0)
# steady state is A_ss = k_prod/k_deg, with closed-form sensitivities
#   dA_ss/dk_prod = 1/k_deg,   dA_ss/dk_deg = -k_prod/k_deg^2.
PREEQUIL_NET = str(DATA_DIR / "preequil_prod_deg.net")
# Issue #43's fixture: species R's initial condition IS the parameter R0, so the
# parameter-graph seeding gives ∂R(0)/∂R0 = 1 — the row a hand-assigned initial
# condition has to be able to override (issue #111).
IC_PARAM_NET = str(DATA_DIR / "ic_direct.net")

K_PROD = 5.0
K_DEG = 0.5
EXTRA_DEG = 2.0  # measurement-phase perturbation (a non-sensitivity condition param)

# Tight tolerances so the carried-over seed is the dominant error term, not the
# integrator. The FD step is chosen well inside the central-difference sweet
# spot for these O(1)-O(10) quantities.
_TOL = dict(rtol=1e-11, atol=1e-13)
_FD_H = 1e-5


def _iA() -> int:
    return bngsim.Model.from_net(PREEQUIL_NET).species_names.index("A()")


def _equilibrated(params=("k_prod", "k_deg")):
    """Phase 1 only: a persistent Simulator sitting on x_ss(θ) with dx_ss/dθ pending."""
    m = bngsim.Model.from_net(PREEQUIL_NET)
    m.set_param("k_prod", K_PROD)
    m.set_param("k_deg", K_DEG)
    m.set_param("extra_deg", 0.0)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(params))
    sim.run(t_span=(0, 200), n_points=3, steady_state=True, steady_state_tol=1e-12, **_TOL)
    return m, sim


def _phase2_exact(t, dose, a0=None, da0=(0.0, 0.0), kp=K_PROD, kd=K_DEG):
    """Exact (dA/dk_prod, dA/dk_deg) for the phase-2 relaxation at ``extra_deg=dose``.

    dA/dt = kp - (kd + dose)*A from A(0) = a0 gives A = P + (a0-P)e^{-λt} with
    λ = kd + dose and P = kp/λ, so both columns are closed form:

        dA/dθ = (∂P/∂θ)(1 - e^{-λt}) + (∂a0/∂θ) e^{-λt} - (a0-P) t e^{-λt} [θ=k_deg]

    ``a0=None`` means the carried pre-equilibrated start A(0) = kp/kd — a
    *function of θ*, which is what makes the carried seed dx_ss/dθ rather than
    zero. A float is an ``on_point`` override's assigned start, with ``da0`` its
    ``(∂a0/∂k_prod, ∂a0/∂k_deg)``: ``(0, 0)`` for a literal dose, nonzero for a
    dose computed from a differentiated parameter (issue #111).
    """
    t = np.asarray(t, dtype=float)
    lam = kd + dose
    p = kp / lam
    e = np.exp(-lam * t)
    if a0 is None:
        q = kp / kd - p
        d_kp = 1.0 / lam + (1.0 / kd - 1.0 / lam) * e
        d_kd = -kp / lam**2 + (-kp / kd**2 + kp / lam**2) * e - q * t * e
    else:
        q = a0 - p
        d_kp = (1.0 - e) / lam + da0[0] * e
        d_kd = -kp / lam**2 * (1.0 - e) + da0[1] * e - q * t * e
    return np.stack([d_kp, d_kd], axis=-1)


def _two_phase(k_prod, k_deg, *, carry, params=("k_prod", "k_deg")):
    """Run equilibrate→perturb→measure on one persistent Simulator.

    Returns (A_ss, measurement_Result).
    """
    iA = _iA()
    m = bngsim.Model.from_net(PREEQUIL_NET)
    m.set_param("k_prod", k_prod)
    m.set_param("k_deg", k_deg)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(params))

    # Phase 1: equilibrate under the pre-condition (extra_deg = 0), unmeasured.
    m.set_param("extra_deg", 0.0)
    r1 = sim.run(t_span=(0, 200), n_points=3, steady_state=True, steady_state_tol=1e-12, **_TOL)
    a_ss = float(np.asarray(r1.species)[-1, iA])

    # Phase 2: perturb (turn on extra degradation) and measure, state carried over.
    m.set_param("extra_deg", EXTRA_DEG)
    r2 = sim.run(t_span=(0, 3), n_points=31, carry_sensitivities=carry, **_TOL)
    return a_ss, r2


# ── Correctness: carry-over seeding vs finite differences ────────────────────


class TestCarryOverCorrectness:
    def test_steady_state_value(self):
        a_ss, _ = _two_phase(K_PROD, K_DEG, carry=True)
        assert a_ss == pytest.approx(K_PROD / K_DEG, rel=1e-7)  # 10.0

    def test_t0_seed_is_closed_form_steady_state_sensitivity(self):
        """Measurement-phase dA/dθ at t=0 == dx_ss/dθ (the carried seed)."""
        iA = _iA()
        _, r2 = _two_phase(K_PROD, K_DEG, carry=True)
        s = r2.sensitivities  # (n_times, n_species, n_params)
        assert s[0, iA, 0] == pytest.approx(1.0 / K_DEG, rel=1e-6)  # dA_ss/dk_prod = 2.0
        assert s[0, iA, 1] == pytest.approx(-K_PROD / K_DEG**2, rel=1e-6)  # dA_ss/dk_deg = -20

    def test_measurement_sensitivity_matches_full_run_fd(self):
        """Phase-2 dA/dθ matches central FD over the *entire* two-phase run."""
        iA = _iA()
        _, r2 = _two_phase(K_PROD, K_DEG, carry=True)
        analytic = r2.sensitivities[:, iA, :]  # (n_times, 2)

        def measure(k_prod, k_deg):
            _, r = _two_phase(k_prod, k_deg, carry=True)
            return np.asarray(r.species)[:, iA]

        h = _FD_H
        fd_kprod = (measure(K_PROD + h, K_DEG) - measure(K_PROD - h, K_DEG)) / (2 * h)
        fd_kdeg = (measure(K_PROD, K_DEG + h) - measure(K_PROD, K_DEG - h)) / (2 * h)

        assert np.allclose(analytic[:, 0], fd_kprod, rtol=1e-5, atol=1e-6)
        assert np.allclose(analytic[:, 1], fd_kdeg, rtol=1e-5, atol=1e-6)

    def test_observable_output_sensitivity_carries(self):
        """A_tot observable output sensitivity also rides the carried seed."""
        _, r2 = _two_phase(K_PROD, K_DEG, carry=True)
        # A_tot == A (group of the single species A), so its dθ equals dA/dθ.
        obs = r2.output_sensitivities("observable:A_tot")  # (n_times, 1, n_params)
        iA = _iA()
        assert np.allclose(obs[:, 0, :], r2.sensitivities[:, iA, :], rtol=1e-9, atol=1e-12)

    def test_without_carry_would_be_wrong_at_t0(self):
        """The carried seed is non-trivial: a fresh seed (0) would be far off."""
        iA = _iA()
        _, r2 = _two_phase(K_PROD, K_DEG, carry=True)
        # The correct t=0 seed is 2.0 / -20.0; a fresh start would give 0 / 0.
        assert abs(r2.sensitivities[0, iA, 0]) > 1.0
        assert abs(r2.sensitivities[0, iA, 1]) > 1.0


# ── Loudly-unsupported: the raise/warn policy ────────────────────────────────


class TestRaisePolicy:
    def test_carryover_state_without_flag_raises(self):
        """Sensitivities on a carried-over state without the opt-in must raise."""
        with pytest.raises(bngsim.SimulationError, match=r"carried-over.*GH #210"):
            _two_phase(K_PROD, K_DEG, carry=False)

    def test_carry_without_prior_seed_raises(self):
        """carry_sensitivities=True with no equilibration phase run raises."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        with pytest.raises(bngsim.SimulationError, match=r"no matching forward-sensitivity seed"):
            sim.run(t_span=(0, 3), n_points=5, carry_sensitivities=True)

    def test_reset_between_phases_then_carry_raises(self):
        """A reset() between phases drops the seed → carry then raises (SBML/reset
        path analogue: a backend that resets every action cannot carry over)."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        sim.run(t_span=(0, 200), n_points=3, steady_state=True)
        m.reset()  # wipe the carry-over (as an every-action-reset backend would)
        with pytest.raises(bngsim.SimulationError, match=r"no matching forward-sensitivity seed"):
            sim.run(t_span=(0, 3), n_points=5, carry_sensitivities=True)

    def test_param_name_mismatch_raises(self):
        """Seed columns must match the requested sensitivity_params exactly."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        # Phase 1 captures a seed for [k_prod, k_deg]...
        sim1 = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        sim1.run(t_span=(0, 200), n_points=3, steady_state=True)
        # ...but a second Simulator over the SAME model asks for only [k_prod].
        sim2 = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod"])
        with pytest.raises(bngsim.SimulationError, match=r"no matching forward-sensitivity seed"):
            sim2.run(t_span=(0, 3), n_points=5, carry_sensitivities=True)

    def test_ic_axis_sensitivity_across_boundary_raises(self):
        """IC (∂y/∂y_k(0)) sensitivities across a carry-over boundary are out of
        scope: the carried state is no longer the model's initial condition."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(
            m, method="ode", sensitivity_params=["k_prod"], sensitivity_ic=["A()"]
        )
        sim.run(t_span=(0, 200), n_points=3, steady_state=True)
        with pytest.raises(bngsim.SimulationError, match=r"sensitivity_ic.*not supported"):
            sim.run(t_span=(0, 3), n_points=5, carry_sensitivities=True)

    def test_carry_without_sensitivity_params_raises(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(m, method="ode")
        with pytest.raises(ValueError, match=r"requires sensitivity_params"):
            sim.run(t_span=(0, 3), n_points=5, carry_sensitivities=True)

    def test_carry_non_ode_raises(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(m, method="ssa")
        with pytest.raises(ValueError, match=r"only supported for method='ode'"):
            sim.run(t_span=(0, 3), n_points=5, carry_sensitivities=True)


# ── State lifecycle and no-regression on fresh runs ──────────────────────────


class TestStateLifecycle:
    def test_dirty_and_seed_set_after_sensitivity_run(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        assert core.ic_state_dirty is False
        assert core.has_pending_sensitivity_seed is False
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        sim.run(t_span=(0, 5), n_points=6)
        assert core.ic_state_dirty is True
        assert core.has_pending_sensitivity_seed is True
        assert core.pending_sensitivity_seed().shape == (core.n_species, 2)
        assert list(core.pending_sensitivity_seed_param_names) == ["k_prod", "k_deg"]

    def test_reset_clears_carryover(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        sim.run(t_span=(0, 5), n_points=6)
        m.reset()
        assert core.ic_state_dirty is False
        assert core.has_pending_sensitivity_seed is False
        # A fresh sensitivity run after reset just works (no raise, fresh seeding).
        r = sim.run(t_span=(0, 5), n_points=6)
        assert r.has_sensitivities

    def test_save_concentrations_hands_carryover_to_the_baseline(self):
        """save_concentrations() redefines the IC baseline to the *current* state,
        so the new baseline inherits that state's dx/dθ instead of dropping it
        (issue #81) — the state did not change, so neither did its derivative."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        sim.run(t_span=(0, 5), n_points=6)
        seed = np.array(core.pending_sensitivity_seed())
        m.save_concentrations()
        assert core.has_baseline_sensitivity_seed is True
        assert core.has_pending_sensitivity_seed is True
        # Still "not fresh-start seedable": a sensitivity run must opt in.
        assert core.ic_state_dirty is True
        # ...and reset() returns to that θ-dependent baseline *with* its dx/dθ.
        m.reset()
        assert core.ic_state_dirty is True
        np.testing.assert_allclose(core.pending_sensitivity_seed(), seed)

    def test_save_concentrations_with_no_carried_derivative_is_a_fresh_start(self):
        """A baseline saved without a carried dx/dθ is θ-independent literal ICs:
        pre-#81 behavior (fresh-start seeding, no baseline seed)."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        m.set_concentration("A()", 4.0)
        m.save_concentrations()
        assert core.has_baseline_sensitivity_seed is False
        assert core.has_pending_sensitivity_seed is False
        assert core.ic_state_dirty is False

    def test_non_sensitivity_run_drops_stale_seed(self):
        """A plain run advances state without tracking dx/dθ, so any seed is
        invalidated → a later carry raises rather than using a stale seed."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        sim.run(t_span=(0, 200), n_points=3, steady_state=True)  # captures seed
        assert core.has_pending_sensitivity_seed is True
        plain = bngsim.Simulator(m, method="ode")  # no sensitivity_params
        plain.run(t_span=(0, 1), n_points=3)  # advances state, drops seed
        assert core.has_pending_sensitivity_seed is False
        assert core.ic_state_dirty is True

    def test_fresh_single_run_unaffected(self):
        """The common case — one sensitivity run on a fresh model — is unchanged."""
        iA = _iA()
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        r = sim.run(t_span=(0, 5), n_points=11, **_TOL)
        # Fresh start: A(0)=0, dA/dθ(0)=0 (no carry seed applied).
        assert r.sensitivities[0, iA, 0] == pytest.approx(0.0, abs=1e-9)
        assert r.sensitivities[0, iA, 1] == pytest.approx(0.0, abs=1e-9)

    def test_pending_seed_write_round_trip(self):
        """The write half of the seed accessor — the primitive that lets a
        protocol restore a state together with its dx/dθ (issue #81)."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        seed = np.arange(core.n_species * 2, dtype=np.float64).reshape(core.n_species, 2)
        core.set_pending_sensitivity_seed(seed, ["k_prod", "k_deg"])
        np.testing.assert_array_equal(core.pending_sensitivity_seed(), seed)
        assert list(core.pending_sensitivity_seed_param_names) == ["k_prod", "k_deg"]
        # set_state drops it (an externally supplied state has no known dx/dθ)...
        m.set_state(m.get_state())
        assert core.has_pending_sensitivity_seed is False
        # ...and an empty write clears it explicitly.
        core.set_pending_sensitivity_seed(seed, ["k_prod", "k_deg"])
        core.set_pending_sensitivity_seed(np.zeros((0, 0)), [])
        assert core.has_pending_sensitivity_seed is False

    def test_pending_seed_write_validates_shape(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        with pytest.raises(ValueError, match="rows"):
            core.set_pending_sensitivity_seed(np.zeros((2, 2)), ["k_prod", "k_deg"])
        with pytest.raises(ValueError, match="param_names"):
            core.set_pending_sensitivity_seed(np.zeros((core.n_species, 2)), ["k_prod"])


# ── Carrying the equilibration's dx/dθ into a parameter scan (issue #81) ──────


class TestScanCarryOver:
    """A pre-equilibration → scored dose scan is one differentiable protocol.

    Each scan point's initial condition is the equilibrated snapshot x_ss(θ), so
    its forward-sensitivity seed is dx_ss/dθ. The oracle is closed form
    (``_phase2_exact``), so these compare against the exact answer rather than a
    finite difference: the whole point of #81 is that re-seeding each point fresh
    is *wrong*, not noisy, and only an exact reference distinguishes the two.
    """

    DOSES = [0.5, 1.0, 2.0, 4.0]
    SPAN = (0.0, 3.0)
    NPTS = 13

    def _times(self):
        return np.linspace(*self.SPAN, self.NPTS)

    def _scan(self, sim, **kw):
        return sim.parameter_scan(
            "extra_deg", self.DOSES, t_span=self.SPAN, n_points=self.NPTS, **_TOL, **kw
        )

    def test_scan_points_match_the_closed_form(self):
        iA = _iA()
        m, sim = _equilibrated()
        results = self._scan(sim)
        assert len(results) == len(self.DOSES)
        for dose, r in zip(self.DOSES, results, strict=True):
            got = np.asarray(r.sensitivities)[:, iA, :]
            np.testing.assert_allclose(
                got, _phase2_exact(self._times(), dose), rtol=1e-6, atol=1e-8
            )
            assert r.custom_attrs["scan_value"] == dose

    def test_each_point_seeds_from_the_equilibration(self):
        """The t=0 seed of *every* point is dx_ss/dθ — not zero, and not the
        previous point's end-state derivative."""
        iA = _iA()
        m, sim = _equilibrated()
        for r in self._scan(sim):
            s0 = np.asarray(r.sensitivities)[0, iA, :]
            assert s0[0] == pytest.approx(1.0 / K_DEG, rel=1e-6)  # 2.0
            assert s0[1] == pytest.approx(-K_PROD / K_DEG**2, rel=1e-6)  # -20.0

    def test_observable_output_sensitivities_per_point(self):
        """What a gradient fit actually consumes: d(observable)/dθ per point."""
        m, sim = _equilibrated()
        for dose, r in zip(self.DOSES, self._scan(sim), strict=True):
            grad = r.output_sensitivities("observable:A_tot")[:, 0, :]
            np.testing.assert_allclose(
                grad, _phase2_exact(self._times(), dose), rtol=1e-6, atol=1e-8
            )

    def test_scored_steady_state_dose_response(self):
        """The #81 protocol shape: pre-equilibrate once, then score each dose at
        its own steady state. dA_ss/dθ = (1/λ, -k_prod/λ²) with λ = k_deg+dose."""
        iA = _iA()
        m, sim = _equilibrated()
        results = sim.parameter_scan(
            "extra_deg",
            self.DOSES,
            t_span=(0, 400),
            n_points=3,
            steady_state=True,
            steady_state_tol=1e-12,
            **_TOL,
        )
        for dose, r in zip(self.DOSES, results, strict=True):
            lam = K_DEG + dose
            exact = np.array([1.0 / lam, -K_PROD / lam**2])
            np.testing.assert_allclose(
                np.asarray(r.sensitivities)[-1, iA, :], exact, rtol=1e-6, atol=1e-9
            )

    def test_bng_save_concentrations_protocol(self):
        """equilibrate → saveConcentrations() → scan, the BNG action ordering:
        the baseline inherits dx/dθ (issue #81), so the scan still carries."""
        iA = _iA()
        m, sim = _equilibrated()
        m.save_concentrations()
        for dose, r in zip(self.DOSES, self._scan(sim), strict=True):
            np.testing.assert_allclose(
                np.asarray(r.sensitivities)[:, iA, :],
                _phase2_exact(self._times(), dose),
                rtol=1e-6,
                atol=1e-8,
            )

    def test_reset_to_named_snapshot_carries_its_derivative(self):
        """A named snapshot remembers its dx/dθ, so reset_to= is seeded correctly
        even after the live state has moved on."""
        iA = _iA()
        m, sim = _equilibrated()
        m.save_concentrations(label="preeq")
        # Move the live state (and its seed) well away from the snapshot.
        m.set_param("extra_deg", 99.0)
        sim.run(t_span=(0, 1), n_points=3, carry_sensitivities=True, **_TOL)
        results = sim.parameter_scan(
            "extra_deg",
            self.DOSES,
            t_span=self.SPAN,
            n_points=self.NPTS,
            reset_to="preeq",
            **_TOL,
        )
        for dose, r in zip(self.DOSES, results, strict=True):
            assert np.asarray(r.species)[0, iA] == pytest.approx(K_PROD / K_DEG, rel=1e-7)
            np.testing.assert_allclose(
                np.asarray(r.sensitivities)[:, iA, :],
                _phase2_exact(self._times(), dose),
                rtol=1e-6,
                atol=1e-8,
            )

    def test_named_snapshot_without_a_derivative_refuses(self):
        """A snapshot saved before any sensitivity run has no dx/dθ to restore —
        refuse rather than seed each point fresh off it."""
        m, sim = _equilibrated()
        m.set_state(m.get_state())  # drops the pending dx/dθ (external assignment)
        m.save_concentrations(label="cold")
        with pytest.raises(ValueError, match=r"the saved state 'cold' carries a matching"):
            sim.parameter_scan("extra_deg", [1.0], t_span=self.SPAN, n_points=3, reset_to="cold")

    def test_continuation_equals_sequential_carry_runs(self):
        """bifurcate carries state *and* dx/dθ point to point, so it is exactly a
        chain of carry_sensitivities=True runs — the already-verified primitive."""
        m, sim = _equilibrated()
        cont = sim.bifurcate("extra_deg", self.DOSES, t_span=self.SPAN, n_points=self.NPTS, **_TOL)

        m2, sim2 = _equilibrated()
        manual = []
        for dose in self.DOSES:
            m2.set_param("extra_deg", dose)
            manual.append(
                sim2.run(t_span=self.SPAN, n_points=self.NPTS, carry_sensitivities=True, **_TOL)
            )
        for a, b in zip(cont, manual, strict=True):
            np.testing.assert_allclose(a.species, b.species, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(a.sensitivities, b.sensitivities, rtol=1e-9, atol=1e-12)

    def test_on_point_literal_dose_measures_a_zero_row(self):
        """A literal on_point setConcentration override is a θ-independent initial
        condition, so its row measures ∂x_k(0)/∂θ = 0 (issue #111 measures what
        #81 assumed) while the rest keep the carried derivative."""
        iA = _iA()
        a0 = 3.0
        m, sim = _equilibrated()
        results = self._scan(sim, on_point=lambda model, v: model.set_concentration("A()", a0))
        for dose, r in zip(self.DOSES, results, strict=True):
            got = np.asarray(r.sensitivities)[:, iA, :]
            assert got[0, 0] == pytest.approx(0.0, abs=1e-9)
            assert got[0, 1] == pytest.approx(0.0, abs=1e-9)
            np.testing.assert_allclose(
                got, _phase2_exact(self._times(), dose, a0=a0), rtol=1e-6, atol=1e-8
            )

    def test_on_point_may_install_its_own_seed(self):
        """The escape hatch for a θ-dependent override: a seed the hook leaves
        pending after its concentration writes is taken verbatim."""
        iA = _iA()
        m, sim = _equilibrated()
        core = m._core
        declared = np.zeros((core.n_species, 2))
        declared[iA, 0] = 0.25  # a made-up but distinctive ∂A(0)/∂k_prod

        def hook(model, value):
            model.set_concentration("A()", 3.0)  # clears the pending seed
            model._core.set_pending_sensitivity_seed(declared, ["k_prod", "k_deg"])

        for r in self._scan(sim, on_point=hook):
            s0 = np.asarray(r.sensitivities)[0, iA, :]
            assert s0[0] == pytest.approx(0.25, rel=1e-9)
            assert s0[1] == pytest.approx(0.0, abs=1e-12)

    def test_scan_leaves_the_carry_over_state_as_found(self):
        """The scan's "left as we found it" promise now covers dx/dθ: a carry run
        after the scan still works."""
        iA = _iA()
        m, sim = _equilibrated()
        state, seed = m.get_state().copy(), np.array(m._core.pending_sensitivity_seed())
        self._scan(sim)
        np.testing.assert_array_equal(m.get_state(), state)
        np.testing.assert_array_equal(m._core.pending_sensitivity_seed(), seed)
        assert m._core.ic_state_dirty is True
        assert m.get_param("extra_deg") == 0.0
        m.set_param("extra_deg", EXTRA_DEG)
        r = sim.run(t_span=self.SPAN, n_points=self.NPTS, carry_sensitivities=True, **_TOL)
        np.testing.assert_allclose(
            np.asarray(r.sensitivities)[:, iA, :],
            _phase2_exact(self._times(), EXTRA_DEG),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_plain_scan_leaves_a_clean_model_clean(self):
        """The same restore on the no-sensitivity path: a scan no longer leaves the
        model marked as carried-over dynamics (it rewinds the state it advanced)."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(m, method="ode")
        sim.parameter_scan("extra_deg", [1.0, 2.0], t_span=(0, 3), n_points=4)
        assert m._core.ic_state_dirty is False
        assert m._core.has_pending_sensitivity_seed is False

    # ── Refusals: no silently re-seeded gradient ────────────────────────────

    def test_no_carried_derivative_refuses(self):
        """A fresh (never equilibrated) sensitivity Simulator still refuses: there
        is no carried dx/dθ, so the old blanket refusal's reasoning still holds."""
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        with pytest.raises(ValueError, match=r"carries a matching forward-sensitivity"):
            self._scan(sim)

    def test_plain_run_between_the_phases_refuses(self):
        """A non-sensitivity run drops the carry (it advanced x without dx/dθ), so
        the scan refuses instead of seeding off a stale/absent derivative."""
        m, sim = _equilibrated()
        bngsim.Simulator(m, method="ode").run(t_span=(0, 1), n_points=3)
        with pytest.raises(ValueError, match=r"carries a matching forward-sensitivity"):
            self._scan(sim)

    def test_param_mismatch_refuses(self):
        m, _ = _equilibrated(params=("k_prod", "k_deg"))
        other = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod"])
        with pytest.raises(ValueError, match=r"carried columns: k_prod, k_deg"):
            other.parameter_scan("extra_deg", [1.0], t_span=self.SPAN, n_points=3)

    def test_scanning_a_differentiated_parameter_refuses(self):
        m, sim = _equilibrated()
        with pytest.raises(ValueError, match=r"cannot scan 'k_prod'"):
            sim.parameter_scan("k_prod", [4.0, 6.0], t_span=self.SPAN, n_points=3)

    def test_sensitivity_ic_refuses(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        sim = bngsim.Simulator(
            m, method="ode", sensitivity_params=["k_prod"], sensitivity_ic=["A()"]
        )
        with pytest.raises(ValueError, match=r"do not support sensitivity_ic"):
            sim.parameter_scan("extra_deg", [1.0], t_span=self.SPAN, n_points=3)

    def test_on_point_moving_a_differentiated_parameter_refuses(self):
        m, sim = _equilibrated()
        with pytest.raises(ValueError, match=r"on_point changed the sensitivity parameter"):
            self._scan(sim, on_point=lambda model, v: model.set_param("k_deg", 0.6))


# ── The IC an on_point hook assigns has its own ∂x(0)/∂θ (issue #111) ─────────


class TestOnPointIcSensitivity:
    """A hook assigns the state its point integrates from, so that state's
    θ-derivative is the hook's own arithmetic — measured through the hook, or
    declared. Oracles are closed form (``_phase2_exact`` with ``da0``), because the
    failure this fixes is a *wrong* row, not a noisy one.
    """

    DOSES = [0.5, 1.0, 2.0]
    SPAN = (0.0, 3.0)
    NPTS = 11

    def _times(self):
        return np.linspace(*self.SPAN, self.NPTS)

    def _scan(self, sim, hook):
        return sim.parameter_scan(
            "extra_deg", self.DOSES, t_span=self.SPAN, n_points=self.NPTS, on_point=hook, **_TOL
        )

    def _check(self, results, a0_of, da0_of):
        iA = _iA()
        for dose, r in zip(self.DOSES, results, strict=True):
            np.testing.assert_allclose(
                np.asarray(r.sensitivities)[:, iA, :],
                _phase2_exact(self._times(), dose, a0=a0_of(dose), da0=da0_of(dose)),
                rtol=1e-5,
                atol=1e-8,
            )

    def test_dose_computed_from_a_differentiated_parameter(self):
        """The issue: A(0) = dose·k_prod has ∂A(0)/∂k_prod = dose, not 0."""
        iA = _iA()
        m, sim = _equilibrated()
        results = self._scan(
            sim, lambda model, v: model.set_concentration("A()", v * model.get_param("k_prod"))
        )
        for dose, r in zip(self.DOSES, results, strict=True):
            s0 = np.asarray(r.sensitivities)[0, iA, :]
            assert s0[0] == pytest.approx(dose, rel=1e-6)  # ∂A(0)/∂k_prod = dose
            assert s0[1] == pytest.approx(0.0, abs=1e-9)
        self._check(results, lambda d: d * K_PROD, lambda d: (d, 0.0))

    def test_nonlinear_dose_formula(self):
        """A smooth but nonlinear formula is measured to difference-quotient
        accuracy, not exactly — the only row that is not exact."""
        m, sim = _equilibrated()
        results = self._scan(
            sim,
            lambda model, v: model.set_concentration(
                "A()", v * np.sqrt(model.get_param("k_prod") * model.get_param("k_deg"))
            ),
        )
        self._check(
            results,
            lambda d: d * np.sqrt(K_PROD * K_DEG),
            lambda d: (d * np.sqrt(K_DEG / K_PROD) / 2, d * np.sqrt(K_PROD / K_DEG) / 2),
        )

    def test_increment_of_the_carried_pool_keeps_the_carried_row(self):
        """A(0) = A_ss + dose depends on θ *through the carried state*, so the row
        is the carried derivative plus the dose's — which is why the measurement
        differences along the carried column, not just in θ."""
        iA = _iA()
        m, sim = _equilibrated()
        results = self._scan(
            sim,
            lambda model, v: model.set_concentration("A()", model.get_concentration("A()") + v),
        )
        for r in results:
            s0 = np.asarray(r.sensitivities)[0, iA, :]
            assert s0[0] == pytest.approx(1.0 / K_DEG, rel=1e-6)  # dA_ss/dk_prod
            assert s0[1] == pytest.approx(-K_PROD / K_DEG**2, rel=1e-6)  # dA_ss/dk_deg
        self._check(
            results,
            lambda d: K_PROD / K_DEG + d,
            lambda d: (1.0 / K_DEG, -K_PROD / K_DEG**2),
        )

    def test_declared_row_is_used_verbatim_and_skips_probing(self):
        """A declared row wins over the measurement — and the hook is then called
        once per point, not once plus the probes."""
        m, sim = _equilibrated()
        calls = []

        def hook(model, v):
            calls.append(v)
            model.set_concentration("A()", v * model.get_param("k_prod"))
            model.declare_ic_sensitivity({"A()": {"k_prod": v}})

        results = self._scan(sim, hook)
        self._check(results, lambda d: d * K_PROD, lambda d: (d, 0.0))
        assert len(calls) == len(self.DOSES)  # no probe invocations

    def test_declaring_an_empty_row_pins_it_to_zero(self):
        """An intentionally θ-independent assignment declares ``{}`` — which is
        also how a non-differentiable dose opts out of measurement."""
        iA = _iA()
        m, sim = _equilibrated()

        def hook(model, v):
            model.set_concentration("A()", float(round(v * model.get_param("k_prod"))))
            model.declare_ic_sensitivity({"A()": {}})

        for r in self._scan(sim, hook):
            np.testing.assert_allclose(
                np.asarray(r.sensitivities)[0, iA, :], [0.0, 0.0], atol=1e-12
            )

    def test_species_the_hook_leaves_alone_keep_the_carried_row_bit_exact(self):
        """An untouched row is never routed through the measurement, so it is
        identical to the same scan with no hook at all."""
        iA = _iA()
        m, sim = _equilibrated()
        plain = sim.parameter_scan(
            "extra_deg", self.DOSES, t_span=self.SPAN, n_points=self.NPTS, **_TOL
        )
        m2, sim2 = _equilibrated()
        hooked = self._scan(sim2, lambda model, v: model.set_concentration("Ad()", 0.0))
        for a, b in zip(plain, hooked, strict=True):
            np.testing.assert_array_equal(
                np.asarray(a.sensitivities)[0, iA, :], np.asarray(b.sensitivities)[0, iA, :]
            )

    def test_non_differentiable_dose_refuses(self):
        """A dose rounded to whole molecules has no derivative at the crossing;
        the two step sizes disagree and bngsim refuses instead of reporting the
        difference quotient of a jump."""
        m, sim = _equilibrated()
        with pytest.raises(ValueError, match=r"not differentiable in 'k_prod'"):
            self._scan(
                sim,
                lambda model, v: model.set_concentration(
                    "A()", float(round(v * model.get_param("k_prod")))
                ),
            )

    def test_hook_that_raises_at_a_perturbed_input_refuses(self):
        m, sim = _equilibrated()
        nominal = m.get_param("k_prod")

        def brittle(model, v):
            if model.get_param("k_prod") != nominal:
                raise RuntimeError("only the nominal value, please")
            model.set_concentration("A()", 3.0)

        with pytest.raises(ValueError, match=r"on_point raised while bngsim measured"):
            self._scan(sim, brittle)

    def test_non_deterministic_hook_refuses(self):
        m, sim = _equilibrated()
        state = {"n": 0}

        def drifting(model, v):
            state["n"] += 1
            model.set_concentration("A()", 3.0 + state["n"])

        with pytest.raises(ValueError, match=r"not a deterministic function"):
            self._scan(sim, drifting)

    def test_probe_leaves_the_point_and_the_model_as_the_hook_left_them(self):
        """The measurement runs the hook at perturbed inputs on the live model, so
        the point must integrate from the nominal assignment and the scan must
        still leave the model as it found it."""
        iA = _iA()
        m, sim = _equilibrated()
        state, seed = m.get_state().copy(), np.array(m._core.pending_sensitivity_seed())
        results = self._scan(
            sim, lambda model, v: model.set_concentration("A()", v * model.get_param("k_prod"))
        )
        for dose, r in zip(self.DOSES, results, strict=True):
            # The integrated trajectory starts from the *nominal* assignment.
            assert np.asarray(r.species)[0, iA] == pytest.approx(dose * K_PROD, rel=1e-12)
        np.testing.assert_array_equal(m.get_state(), state)
        np.testing.assert_array_equal(m._core.pending_sensitivity_seed(), seed)
        assert m.get_param("k_prod") == K_PROD
        assert m._declared_ic_sens == {}
        assert m._ic_write_log is None


# ── Declaring the derivative of a hand-assigned initial condition (issue #111) ─


class TestDeclaredIcSensitivityOnAFreshRun:
    """``set_concentration`` replaces the ``.net`` IC expression that the
    parameter-graph seeding (issue #43) differentiates, so a *hand-assigned*
    θ-dependent initial condition has no derivative the engine can infer — outside
    a hook there is nothing to probe. ``declare_ic_sensitivity`` supplies it.
    """

    SPAN = (0.0, 3.0)
    NPTS = 9
    DOSE = 1.0  # extra_deg for the measurement run

    def _fresh(self, *, declare):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        m.set_param("k_prod", K_PROD)
        m.set_param("k_deg", K_DEG)
        m.set_param("extra_deg", self.DOSE)
        a0 = 3.0 * m.get_param("k_prod")  # a θ-dependent hand-assigned IC
        m.set_concentration("A()", a0)
        if declare:
            m.declare_ic_sensitivity({"A()": {"k_prod": 3.0}})
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        return m, sim.run(t_span=self.SPAN, n_points=self.NPTS, **_TOL), a0

    def test_declared_row_matches_the_closed_form(self):
        iA = _iA()
        _, r, a0 = self._fresh(declare=True)
        np.testing.assert_allclose(
            np.asarray(r.sensitivities)[:, iA, :],
            _phase2_exact(np.linspace(*self.SPAN, self.NPTS), self.DOSE, a0=a0, da0=(3.0, 0.0)),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_without_the_declaration_the_row_is_the_literal_zero(self):
        """The undeclared reading — ∂A(0)/∂k_prod = 0 — is what a literal
        assignment means, and is why a θ-dependent one must say so."""
        iA = _iA()
        _, r, _ = self._fresh(declare=False)
        assert np.asarray(r.sensitivities)[0, iA, 0] == pytest.approx(0.0, abs=1e-12)

    def test_declaration_replaces_the_parameter_graph_row(self):
        """A species whose .net IC *is* a parameter reference gets its declared row
        instead of the expression's (the assignment replaced the expression)."""
        m = bngsim.Model.from_net(IC_PARAM_NET)
        i_r = m.species_names.index("R()")
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["R0"])
        # Without a declaration: ∂R(0)/∂R0 = 1 from the IC expression `R() R0`.
        r = sim.run(t_span=(0, 1), n_points=3, **_TOL)
        assert np.asarray(r.sensitivities)[0, i_r, 0] == pytest.approx(1.0, rel=1e-9)
        # Pin R to a literal and say so: the row becomes 0.
        m2 = bngsim.Model.from_net(IC_PARAM_NET)
        m2.set_concentration("R()", 7.0)
        m2.declare_ic_sensitivity({"R()": {}})
        sim2 = bngsim.Simulator(m2, method="ode", sensitivity_params=["R0"])
        r2 = sim2.run(t_span=(0, 1), n_points=3, **_TOL)
        assert np.asarray(r2.sensitivities)[0, i_r, 0] == pytest.approx(0.0, abs=1e-12)

    def test_lifecycle(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        m.set_concentration("A()", 1.0)
        m.declare_ic_sensitivity({"A()": {"k_prod": 2.0}})
        assert m._declared_ic_sens == {"A()": {"k_prod": 2.0}}
        # Re-assigning the species supersedes its declaration...
        m.set_concentration("A()", 2.0)
        assert m._declared_ic_sens == {}
        # ...as does a wholesale state change.
        m.declare_ic_sensitivity({"A()": {"k_prod": 2.0}})
        m.reset()
        assert m._declared_ic_sens == {}
        m.declare_ic_sensitivity({"A()": {"k_prod": 2.0}})
        m.set_state(m.get_state())
        assert m._declared_ic_sens == {}

    def test_unknown_names_raise(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        with pytest.raises(bngsim.ModelError, match=r"species 'NoSuch\(\)' not found"):
            m.declare_ic_sensitivity({"NoSuch()": {"k_prod": 1.0}})
        with pytest.raises(bngsim.ModelError, match=r"parameter 'nope'"):
            m.declare_ic_sensitivity({"A()": {"nope": 1.0}})
        assert m._declared_ic_sens == {}  # validated before anything is staged


# ── Event path: fixed-time events now run (GH #205 → #212 Phase 1) ───────────


class TestEventSensitivity:
    """GH #212: fixed-time events propagate sensitivities; state-dependent
    triggers still raise (the GH #205 correctness posture for the long tail)."""

    def _event_sim(self, trigger="time() >= 1000", assign="A"):
        from bngsim._bngsim_core import ModelBuilder

        b = ModelBuilder()
        b.add_parameter("k_prod", 5.0)
        b.add_parameter("k_deg", 0.5)
        s = b.add_species("A", 0.0)
        b.add_reaction([], [s], "elementary", "k_prod")  # 0-order synthesis
        b.add_reaction([s], [], "elementary", "k_deg")  # degradation
        b.add_event("evt", trigger, [(s, assign)])
        m = bngsim.Model(b.build())
        assert m._core.n_events == 1
        return bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])

    def test_single_phase_fixed_time_runs(self):
        sim = self._event_sim()
        r = sim.run(t_span=(0, 5), n_points=6)
        assert np.all(np.isfinite(r.sensitivities))

    def test_state_dependent_event_raises(self):
        sim = self._event_sim(trigger="A > 1000", assign="0")
        with pytest.raises(ValueError, match=r"state-dependent|205"):
            sim.run(t_span=(0, 5), n_points=6)
