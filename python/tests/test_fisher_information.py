"""Tests for Result.fisher_information() (Session 27).

Verifies the Fisher Information Matrix computation from CVODES sensitivity data.
"""

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _get_simple_decay_net() -> str:
    p = DATA_DIR / "simple_decay.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_reversible_net() -> str:
    p = DATA_DIR / "two_species_reversible.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


class TestFisherInformationShape:
    """Shape and basic properties of the FIM."""

    def test_single_param_shape(self):
        """FIM for 1 param should be (1, 1)."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        fim = result.fisher_information(sigma=1.0)
        assert fim.shape == (1, 1)

    def test_two_param_shape(self):
        """FIM for 2 params should be (2, 2)."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
        result = sim.run(t_span=(0, 10), n_points=101)

        fim = result.fisher_information(sigma=1.0)
        assert fim.shape == (2, 2)

    def test_symmetric(self):
        """FIM should be symmetric."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
        result = sim.run(t_span=(0, 10), n_points=101)

        fim = result.fisher_information(sigma=1.0)
        np.testing.assert_allclose(fim, fim.T, atol=1e-12)

    def test_positive_semidefinite(self):
        """FIM should be positive semi-definite (all eigenvalues >= 0)."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
        result = sim.run(t_span=(0, 10), n_points=101)

        fim = result.fisher_information(sigma=1.0)
        eigvals = np.linalg.eigvalsh(fim)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals}"


class TestFisherInformationSigma:
    """Test sigma (noise) parameter handling."""

    def test_scalar_sigma(self):
        """Scalar sigma should work."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        fim1 = result.fisher_information(sigma=1.0)
        fim01 = result.fisher_information(sigma=0.1)

        # Smaller sigma → larger FIM (more information per measurement)
        assert fim01[0, 0] > fim1[0, 0]

    def test_sigma_scaling(self):
        """FIM scales as 1/sigma^2."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        fim1 = result.fisher_information(sigma=1.0)
        fim2 = result.fisher_information(sigma=2.0)

        # FIM(sigma=2) should be FIM(sigma=1) / 4
        np.testing.assert_allclose(fim2, fim1 / 4.0, rtol=1e-10)

    def test_per_species_sigma(self):
        """Per-species sigma array should work."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        n_sp = result.n_species
        sigma_arr = np.ones(n_sp) * 1.0
        fim_arr = result.fisher_information(sigma=sigma_arr)
        fim_scalar = result.fisher_information(sigma=1.0)

        np.testing.assert_allclose(fim_arr, fim_scalar, rtol=1e-10)

    def test_per_species_sigma_different(self):
        """Different per-species sigmas should change the result."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        n_sp = result.n_species
        sigma_uniform = np.ones(n_sp)
        sigma_weighted = np.ones(n_sp) * 10.0
        sigma_weighted[0] = 1.0  # only species 0 has low noise

        fim_uniform = result.fisher_information(sigma=sigma_uniform)
        fim_weighted = result.fisher_information(sigma=sigma_weighted)

        # They should be different (species 0 dominates in weighted case)
        assert not np.allclose(fim_uniform, fim_weighted)


class TestFisherInformationErrors:
    """Error handling."""

    def test_no_sensitivity_raises(self):
        """Should raise ValueError if no sensitivity data."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        with pytest.raises(ValueError, match="No sensitivity data"):
            result.fisher_information(sigma=1.0)

    def test_wrong_sigma_length(self):
        """Should raise ValueError for wrong sigma array length."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        with pytest.raises(ValueError, match="must match n_species"):
            result.fisher_information(sigma=np.ones(999))

    def test_negative_sigma_raises(self):
        """Should raise ValueError for non-positive sigma."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        n_sp = result.n_species
        sigma_bad = np.ones(n_sp)
        sigma_bad[0] = -1.0
        with pytest.raises(ValueError, match="must be > 0"):
            result.fisher_information(sigma=sigma_bad)

    def test_2d_sigma_raises(self):
        """Should raise ValueError for 2D sigma."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        with pytest.raises(ValueError, match="scalar or 1-D"):
            result.fisher_information(sigma=np.ones((2, 2)))


class TestFisherInformationAnalytical:
    """Analytical verification of FIM for simple decay.

    Model: dA/dt = -k1*A, A(0)=100, k1=0.1
    Sensitivity: dA/dk1 = -t * A(0) * exp(-k1*t)

    FIM = sum_t (dY/dp)^T (1/sigma^2) (dY/dp)
    For 1 param, 2 species (A and B=100-A):
      dA/dk1 = -t * 100 * exp(-0.1*t)
      dB/dk1 = +t * 100 * exp(-0.1*t)
    FIM_11 = (1/sigma^2) * sum_t [(dA/dk1)^2 + (dB/dk1)^2]
           = (1/sigma^2) * sum_t [2 * (t * 100 * exp(-0.1*t))^2]
    """

    def test_analytical_fim(self):
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        sigma = 1.0
        fim = result.fisher_information(sigma=sigma)

        # Compute expected FIM analytically
        t = result.time
        k1 = 0.1
        A0 = 100.0
        dA_dk1 = -t * A0 * np.exp(-k1 * t)
        dB_dk1 = t * A0 * np.exp(-k1 * t)

        expected_fim = np.sum(dA_dk1**2 + dB_dk1**2) / sigma**2

        # Should be close (CVODES internal FD introduces small error)
        np.testing.assert_allclose(
            fim[0, 0], expected_fim, rtol=0.01, err_msg="FIM analytical check failed"
        )

    def test_nonzero_fim(self):
        """FIM should be nonzero for a non-trivial system."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        fim = result.fisher_information(sigma=1.0)
        assert fim[0, 0] > 0.0, "FIM should be positive for identifiable param"
