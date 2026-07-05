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

    def test_save_concentrations_clears_carryover(self):
        m = bngsim.Model.from_net(PREEQUIL_NET)
        core = m._core
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])
        sim.run(t_span=(0, 5), n_points=6)
        m.save_concentrations()
        assert core.ic_state_dirty is False
        assert core.has_pending_sensitivity_seed is False

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
        return bngsim.Simulator(
            m, method="ode", sensitivity_params=["k_prod", "k_deg"]
        )

    def test_single_phase_fixed_time_runs(self):
        sim = self._event_sim()
        r = sim.run(t_span=(0, 5), n_points=6)
        assert np.all(np.isfinite(r.sensitivities))

    def test_state_dependent_event_raises(self):
        sim = self._event_sim(trigger="A > 1000", assign="0")
        with pytest.raises(ValueError, match=r"state-dependent|205"):
            sim.run(t_span=(0, 5), n_points=6)
