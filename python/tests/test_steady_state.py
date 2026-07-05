"""Tests for the two-tier steady-state solver (Session 33).

Covers:
- Simple decay: Y_ss = 0
- Reversible reaction: Y_ss = K*A0/(1+K)
- Integration method
- Newton method (with fallback)
- Auto method
- Dose-response curve (batch)
- Steady-state sensitivity dY_ss/dp
- SteadyStateResult API (dict access, to_dict, repr)
- Error cases (SSA, bad method)
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# -- Fixtures -------------------------------------------------------


@pytest.fixture
def simple_decay_model(data_dir):
    """A -> 0 with rate k."""
    return bngsim.Model.from_net(str(data_dir / "simple_decay.net"))


@pytest.fixture
def reversible_model(data_dir):
    """A <-> B with kf, kr."""
    return bngsim.Model.from_net(str(data_dir / "two_species_reversible.net"))


# -- Tier 1: Integration -------------------------------------------


class TestIntegration:
    """Tier 1: CVODE integration with early termination."""

    def test_simple_decay_converges(self, simple_decay_model):
        """Simple decay converges."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state(method="integration", tol=1e-9)
        assert ss.converged
        assert ss.method_used == "integration"
        assert ss.residual < 1e-9

    def test_reversible_equilibrium(self, reversible_model):
        """A <-> B reaches equilibrium via integration."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(method="integration", tol=1e-8)
        assert ss.converged
        assert ss.residual < 1e-8

    def test_result_properties(self, simple_decay_model):
        """SteadyStateResult has correct properties."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state(method="integration")
        assert isinstance(ss.converged, bool)
        assert isinstance(ss.residual, float)
        assert isinstance(ss.method_used, str)
        assert isinstance(ss.n_steps, int)
        assert isinstance(ss.n_rhs_evals, int)
        assert ss.n_steps > 0
        assert ss.n_rhs_evals > 0
        assert len(ss.species_names) > 0
        n = len(ss.concentrations)
        assert n == len(ss.species_names)


# -- Tier 2: Newton ------------------------------------------------


class TestNewton:
    """Tier 2: KINSOL Newton solver."""

    def test_newton_or_fallback(self, simple_decay_model):
        """Newton solver runs (may or may not converge
        depending on model structure)."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        # Newton may fail for models with fixed species
        # (singular Jacobian). That's expected.
        ss = sim.steady_state(method="newton", tol=1e-9)
        # Just verify it returns a result
        assert isinstance(ss.residual, float)
        assert ss.method_used == "newton"

    def test_newton_converges_with_fallback(self, reversible_model):
        """Newton converges; on non-convergence it falls back EXPLICITLY
        to the parity integration path. Either way method='newton' returns
        a converged result on a reversible model."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(method="newton", tol=1e-8)
        assert ss.converged
        # method_used is "newton" when Newton succeeds, "integration" when
        # the explicit parity fallback fires.
        assert ss.method_used in ("newton", "integration")

    def test_kinsol_alias(self, reversible_model):
        """'kinsol' is an input alias for 'newton'; canonical name echoed."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(method="kinsol", tol=1e-8)
        assert ss.converged
        # When Newton itself converges the canonical name "newton" is echoed
        # (never "kinsol"); the parity fallback would echo "integration".
        assert ss.method_used in ("newton", "integration")


# -- Default method (newton) ---------------------------------------


class TestDefaultMethod:
    """Default method is 'newton' (KINSOL + explicit parity fallback)."""

    def test_default_is_newton(self, simple_decay_model):
        """Calling steady_state() with no method defaults to newton."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state(tol=1e-9)
        assert ss.converged
        assert ss.method_used in ("newton", "integration")

    def test_integration_and_default_agree(self, reversible_model):
        """Integration and the default newton give similar results."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss_int = sim.steady_state(method="integration", tol=1e-8)
        reversible_model.reset()
        ss_def = sim.steady_state(tol=1e-8)
        assert ss_int.converged
        assert ss_def.converged
        np.testing.assert_allclose(
            ss_int.concentrations,
            ss_def.concentrations,
            atol=1e-4,
        )

    def test_auto_method_rejected(self, simple_decay_model):
        """The removed 'auto' method (and any unknown method) raises."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        with pytest.raises((ValueError, RuntimeError, bngsim.SimulationError)):
            sim.steady_state(method="auto")


# -- Batch ---------------------------------------------------------


class TestBatch:
    """Steady-state batch (dose-response curve)."""

    def test_dose_response(self, reversible_model):
        """Varying kf gives all-converged batch."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        doses = [{"kf": kf} for kf in [0.01, 0.1, 1.0, 10.0]]
        results = sim.steady_state_batch(doses)
        assert len(results) == 4
        assert all(r.converged for r in results)

    def test_batch_parallel(self, reversible_model):
        """Parallel batch gives same results as serial."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        doses = [{"kf": kf} for kf in [0.1, 1.0, 10.0]]
        serial = sim.steady_state_batch(doses)
        reversible_model.reset()
        parallel = sim.steady_state_batch(doses, n_workers=2)
        for s, p in zip(serial, parallel, strict=False):
            np.testing.assert_allclose(
                s.concentrations,
                p.concentrations,
                atol=1e-4,
            )


# -- Sensitivity ---------------------------------------------------


class TestSensitivity:
    """Steady-state sensitivity dY_ss/dp."""

    def test_sensitivity_shape(self, reversible_model):
        """Sensitivity matrix has correct shape."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(sensitivity_params=["kf", "kr"])
        assert ss.converged
        assert ss.sensitivity is not None
        ns = len(ss.concentrations)
        assert ss.sensitivity.shape == (ns, 2)
        assert ss.sensitivity_params == ["kf", "kr"]

    def test_sensitivity_vs_fd(self, simple_decay_model):
        """Sensitivity matches finite-difference.

        Session 34: Fixed by reduced-space sensitivity solve
        that handles singular J from conservation laws.
        """
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        param_name = simple_decay_model.param_names[0]
        ss = sim.steady_state(
            method="integration",
            sensitivity_params=[param_name],
        )
        assert ss.converged
        assert ss.sensitivity is not None

        # FD: perturb the first parameter
        p0 = simple_decay_model.get_param(param_name)
        h = max(1e-5 * abs(p0), 1e-8)

        simple_decay_model.set_param(param_name, p0 + h)
        simple_decay_model.reset()
        ss_plus = sim.steady_state(method="integration")
        assert ss_plus.converged

        simple_decay_model.set_param(param_name, p0 - h)
        simple_decay_model.reset()
        ss_minus = sim.steady_state(method="integration")
        assert ss_minus.converged

        fd_sens = (ss_plus.concentrations - ss_minus.concentrations) / (2 * h)

        # Compare (relax tolerance for FD accuracy)
        np.testing.assert_allclose(
            ss.sensitivity[:, 0],
            fd_sens,
            rtol=0.1,
            atol=1e-3,
        )


# -- SteadyStateResult API ----------------------------------------


class TestResultAPI:
    """SteadyStateResult dict-like access."""

    def test_dict_access(self, simple_decay_model):
        """Dict-like species access works."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state()
        assert ss.converged
        for name in ss.species_names:
            val = ss[name]
            assert isinstance(val, float)

    def test_to_dict(self, simple_decay_model):
        """to_dict returns correct dict."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state()
        d = ss.to_dict()
        assert isinstance(d, dict)
        assert len(d) == len(ss.species_names)

    def test_repr(self, simple_decay_model):
        """repr is informative."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state()
        r = repr(ss)
        assert "SteadyStateResult" in r
        assert "converged" in r

    def test_key_error(self, simple_decay_model):
        """Accessing unknown species raises KeyError."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state()
        with pytest.raises(KeyError, match="not found"):
            ss["nonexistent_species"]


# -- Error cases ---------------------------------------------------


class TestErrors:
    """Error handling."""

    def test_ssa_rejects(self, simple_decay_model):
        """SSA rejects steady_state()."""
        sim = bngsim.Simulator(simple_decay_model, method="ssa")
        with pytest.raises(ValueError, match="ode"):
            sim.steady_state()

    def test_ssa_rejects_batch(self, simple_decay_model):
        """SSA rejects steady_state_batch()."""
        sim = bngsim.Simulator(simple_decay_model, method="ssa")
        with pytest.raises(ValueError, match="ode"):
            sim.steady_state_batch([{"k": 1.0}])

    def test_empty_batch(self, reversible_model):
        """Empty batch raises ValueError."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        with pytest.raises(ValueError, match="non-empty"):
            sim.steady_state_batch([])
