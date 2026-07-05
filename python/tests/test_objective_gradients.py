"""Tests for built-in objective convenience methods (Session 32).

Verifies sse_gradient, chi2_gradient, neg_log_likelihood_gradient
on Result objects with CVODES sensitivity data.
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


def _get_result_with_sens(net_name, params=None):
    """Load model, run with all-param sensitivity, return result."""
    p = DATA_DIR / net_name
    assert p.exists(), f"Test data not found: {p}"
    model = bngsim.Model.from_net(str(p))
    param_names = params or model.param_names
    sim = bngsim.Simulator(
        model,
        method="ode",
        sensitivity_params=list(param_names),
    )
    result = sim.run(t_span=(0, 10), n_points=51)
    return result


class TestSSEGradient:
    """Tests for result.sse_gradient(data)."""

    def test_shape(self):
        """Returns (float, ndarray(n_params,))."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        data = np.zeros_like(r.species)
        loss, grad = r.sse_gradient(data)
        assert isinstance(loss, float)
        assert grad.shape == (1,)

    def test_zero_residual_zero_gradient(self):
        """When data == species, loss=0 and gradient=0."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        loss, grad = r.sse_gradient(r.species.copy())
        assert loss == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_matches_manual_gradient(self):
        """SSE gradient matches Result.gradient() with dL/dY = 2*(Y-D)."""
        r = _get_result_with_sens("two_species_reversible.net")
        data = np.zeros_like(r.species)

        loss, grad = r.sse_gradient(data)

        # Manual
        manual_grad = r.gradient(lambda species, time: 2 * (species - data))
        np.testing.assert_allclose(
            grad,
            manual_grad,
            atol=1e-12,
            err_msg="sse_gradient != manual gradient",
        )

        # Verify loss value
        expected_loss = np.sum((r.species - data) ** 2)
        assert loss == pytest.approx(expected_loss, rel=1e-12)

    def test_loss_positive(self):
        """Loss should be positive for non-matching data."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        data = np.ones_like(r.species) * 999.0
        loss, grad = r.sse_gradient(data)
        assert loss > 0

    def test_wrong_shape_raises(self):
        """Should raise if data shape doesn't match."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        with pytest.raises(ValueError, match="shape"):
            r.sse_gradient(np.zeros((5, 5)))

    def test_no_sensitivity_raises(self):
        """Should raise if no sensitivity data."""
        model = bngsim.Model.from_net(str(DATA_DIR / "simple_decay.net"))
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)
        with pytest.raises(ValueError, match="sensitivity"):
            result.sse_gradient(np.zeros_like(result.species))


class TestChi2Gradient:
    """Tests for result.chi2_gradient(data, sigma)."""

    def test_scalar_sigma(self):
        """chi2 with scalar sigma gives correct shape."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        data = np.zeros_like(r.species)
        chi2, grad = r.chi2_gradient(data, sigma=1.0)
        assert isinstance(chi2, float)
        assert grad.shape == (1,)

    def test_sigma_1_equals_sse(self):
        """chi2 with sigma=1 should equal SSE."""
        r = _get_result_with_sens("two_species_reversible.net")
        data = np.zeros_like(r.species)
        sse_loss, sse_grad = r.sse_gradient(data)
        chi2_loss, chi2_grad = r.chi2_gradient(data, sigma=1.0)
        assert chi2_loss == pytest.approx(sse_loss, rel=1e-12)
        np.testing.assert_allclose(
            chi2_grad,
            sse_grad,
            atol=1e-12,
        )

    def test_sigma_scaling(self):
        """Doubling sigma should quarter the chi2."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        data = np.zeros_like(r.species)
        chi2_1, _ = r.chi2_gradient(data, sigma=1.0)
        chi2_2, _ = r.chi2_gradient(data, sigma=2.0)
        assert chi2_2 == pytest.approx(chi2_1 / 4.0, rel=1e-10)

    def test_per_species_sigma(self):
        """Per-species sigma works."""
        r = _get_result_with_sens("two_species_reversible.net")
        data = np.zeros_like(r.species)
        sigma = np.array([1.0] * r.n_species)
        chi2, grad = r.chi2_gradient(data, sigma=sigma)
        # Should match scalar sigma=1
        chi2_s, grad_s = r.chi2_gradient(data, sigma=1.0)
        assert chi2 == pytest.approx(chi2_s, rel=1e-12)
        np.testing.assert_allclose(grad, grad_s, atol=1e-12)


class TestNegLogLikelihoodGradient:
    """Tests for result.neg_log_likelihood_gradient(data, sigma)."""

    def test_shape(self):
        """Returns (float, ndarray)."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        data = np.zeros_like(r.species)
        nll, grad = r.neg_log_likelihood_gradient(data, sigma=1.0)
        assert isinstance(nll, float)
        assert grad.shape == (1,)

    def test_gradient_equals_half_chi2_gradient(self):
        """NLL gradient should be 0.5 * chi2 gradient.

        NLL = 0.5 * chi2 + const, so d(NLL)/dp = 0.5 * d(chi2)/dp.
        """
        r = _get_result_with_sens("two_species_reversible.net")
        data = np.zeros_like(r.species)
        _, chi2_grad = r.chi2_gradient(data, sigma=0.5)
        _, nll_grad = r.neg_log_likelihood_gradient(data, sigma=0.5)
        np.testing.assert_allclose(
            nll_grad,
            0.5 * chi2_grad,
            atol=1e-12,
        )

    def test_zero_residual(self):
        """When data == species, gradient is zero but NLL is nonzero
        (constant term from log(2*pi*sigma^2))."""
        r = _get_result_with_sens("simple_decay.net", ["k1"])
        nll, grad = r.neg_log_likelihood_gradient(r.species.copy(), sigma=1.0)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)
        # NLL has constant term: 0.5 * n_data * log(2*pi*1^2)
        n_data = r.species.size
        expected = 0.5 * n_data * np.log(2 * np.pi)
        assert nll == pytest.approx(expected, rel=1e-10)


class TestSpeciesIndices:
    """Test species_indices kwarg on all methods."""

    def test_sse_species_indices(self):
        """SSE with species_indices=[0] only uses first species."""
        r = _get_result_with_sens("two_species_reversible.net")
        data_sub = np.zeros((r.n_times, 1))

        loss_sub, grad_sub = r.sse_gradient(data_sub, species_indices=[0])
        # Should only have residual from species 0
        expected = np.sum(r.species[:, 0] ** 2)
        assert loss_sub == pytest.approx(expected, rel=1e-10)
