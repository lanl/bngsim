"""Tests for JAX custom_jvp bridge — differentiable ODE solving (Session 32).

Verifies:
1. Primal correctness: differentiable_solve matches Simulator.run()
2. Gradient shape and accuracy (vs Result.gradient and analytical)
3. value_and_grad: both loss value and gradient correct
4. jacfwd: full Jacobian matches CVODES sensitivity tensor
5. Composition: jax.grad of lambda wrapping differentiable_solve
6. Functional rates: CVODES FD fallback
7. Fixed species: correct gradient
8. Error cases: wrong param length, SSA model detection
9. Solver options propagation
10. Thread safety via cloning
"""

import os
from pathlib import Path

import numpy as np
import pytest

# Skip entire module if JAX not available
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import bngsim  # noqa: E402  (after pytest.importorskip above)
from bngsim.jax import differentiable_solve  # noqa: E402

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _get_net(name: str) -> str:
    p = DATA_DIR / name
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_model_and_params(net_name: str):
    """Load model and extract default parameter vector."""
    model = bngsim.Model.from_net(_get_net(net_name))
    param_names = model.param_names
    p0 = jnp.array(
        [model.get_param(n) for n in param_names],
        dtype=jnp.float64,
    )
    return model, p0, param_names


# ── Primal correctness ──────────────────────────────────────────────


class TestPrimalCorrectness:
    """differentiable_solve primal output matches Simulator.run()."""

    def test_simple_decay_matches_simulator(self):
        """Simple decay: Y from differentiable_solve == Y from Simulator."""
        model, p0, _ = _get_model_and_params("simple_decay.net")
        Y_jax = differentiable_solve(model, p0, (0, 10), 11)

        # Reference via Simulator
        model2 = bngsim.Model.from_net(_get_net("simple_decay.net"))
        sim = bngsim.Simulator(model2, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        np.testing.assert_allclose(
            np.asarray(Y_jax),
            result.species,
            atol=1e-10,
            err_msg="Primal mismatch vs Simulator",
        )

    def test_reversible_matches_simulator(self):
        """Two-species reversible reaction."""
        model, p0, _ = _get_model_and_params("two_species_reversible.net")
        Y_jax = differentiable_solve(model, p0, (0, 10), 51)

        model2 = bngsim.Model.from_net(_get_net("two_species_reversible.net"))
        sim = bngsim.Simulator(model2, method="ode")
        result = sim.run(t_span=(0, 10), n_points=51)

        np.testing.assert_allclose(
            np.asarray(Y_jax),
            result.species,
            atol=1e-10,
        )

    def test_output_shape(self):
        """Output shape is (n_points, n_species)."""
        model, p0, _ = _get_model_and_params("simple_decay.net")
        Y = differentiable_solve(model, p0, (0, 10), 21)
        assert Y.shape == (21, model.n_species)

    def test_output_is_jnp_array(self):
        """Output should be a JAX array, not numpy."""
        model, p0, _ = _get_model_and_params("simple_decay.net")
        Y = differentiable_solve(model, p0, (0, 10), 11)
        assert isinstance(Y, jnp.ndarray)


# ── Gradient (jax.grad) ─────────────────────────────────────────────


class TestGradient:
    """jax.grad through differentiable_solve."""

    def test_gradient_shape(self):
        """Gradient has shape (n_params,)."""
        model, p0, _ = _get_model_and_params("simple_decay.net")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 11)
            return jnp.sum(Y**2)

        grad = jax.grad(loss)(p0)
        assert grad.shape == p0.shape

    def test_gradient_nonzero(self):
        """Gradient should be nonzero for a non-trivial loss."""
        model, p0, _ = _get_model_and_params("simple_decay.net")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 101)
            return jnp.sum(Y**2)

        grad = jax.grad(loss)(p0)
        assert jnp.any(jnp.abs(grad) > 1e-10)

    def test_gradient_analytical_simple_decay(self):
        """Compare gradient to analytical formula for simple decay.

        Model: dA/dt = -k1*A, A(0)=100, k1=0.1
        Loss: L = sum_t A(t)^2
        dL/dk1 = sum_t 2*A(t) * dA/dk1
               = sum_t 2*(A0*exp(-k1*t)) * (-t*A0*exp(-k1*t))
               = -2 * A0^2 * sum_t t * exp(-2*k1*t)
        """
        model, p0, _ = _get_model_and_params("simple_decay.net")
        n_points = 101

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), n_points)
            # Only sum A (species 0) squared
            return jnp.sum(Y[:, 0] ** 2)

        grad = jax.grad(loss)(p0)

        # Analytical
        k1 = 0.1
        A0 = 100.0
        t = np.linspace(0, 10, n_points)
        expected_dk1 = np.sum(-2 * A0**2 * t * np.exp(-2 * k1 * t))

        # grad[0] corresponds to k1
        np.testing.assert_allclose(
            float(grad[0]),
            expected_dk1,
            rtol=0.01,
            err_msg="Analytical gradient check failed",
        )

    def test_gradient_matches_result_gradient(self):
        """JAX gradient matches Result.gradient() (manual chain rule).

        Both should produce the same parameter gradient for the same
        loss function, since they both use CVODES sensitivities.
        """
        net_path = _get_net("two_species_reversible.net")
        model, p0, param_names = _get_model_and_params("two_species_reversible.net")
        n_points = 51

        # Target data (zeros for simplicity)
        target = np.zeros((n_points, model.n_species))

        # JAX gradient
        def jax_loss(p):
            Y = differentiable_solve(model, p, (0, 10), n_points)
            return jnp.sum((Y - target) ** 2)

        grad_jax = np.asarray(jax.grad(jax_loss)(p0))

        # Result.gradient (Session 28 API)
        model2 = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(
            model2,
            method="ode",
            sensitivity_params=list(param_names),
        )
        result = sim.run(t_span=(0, 10), n_points=n_points)

        def loss_fn(species, time):
            return 2 * (species - target)

        grad_result = result.gradient(loss_fn)

        np.testing.assert_allclose(
            grad_jax,
            grad_result,
            atol=1e-5,
            rtol=1e-4,
            err_msg="JAX grad != Result.gradient()",
        )


# ── value_and_grad ───────────────────────────────────────────────────


class TestValueAndGrad:
    """jax.value_and_grad through differentiable_solve."""

    def test_value_and_grad(self):
        """Both loss value and gradient are correct."""
        model, p0, _ = _get_model_and_params("simple_decay.net")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 11)
            return jnp.sum(Y**2)

        val, grad = jax.value_and_grad(loss)(p0)

        # val should be positive (sum of squares)
        assert float(val) > 0
        # grad should match jax.grad
        grad_only = jax.grad(loss)(p0)
        np.testing.assert_allclose(
            np.asarray(grad),
            np.asarray(grad_only),
            atol=1e-12,
        )


# ── jacfwd ───────────────────────────────────────────────────────────


class TestJacfwd:
    """jax.jacfwd to get full sensitivity matrix."""

    def test_jacfwd_shape(self):
        """jacfwd should return (n_points*n_species, n_params)."""
        model, p0, _ = _get_model_and_params("simple_decay.net")
        n_points = 11

        def solve_flat(p):
            return differentiable_solve(model, p, (0, 10), n_points).ravel()

        J = jax.jacfwd(solve_flat)(p0)
        n_sp = model.n_species
        n_p = len(p0)
        assert J.shape == (n_points * n_sp, n_p)

    def test_jacfwd_matches_cvodes_sensitivity(self):
        """jacfwd output matches CVODES sensitivity tensor.

        jacfwd returns d(Y.ravel())/dp which is the sensitivity
        tensor reshaped to (n_points*n_species, n_params).
        """
        net_path = _get_net("simple_decay.net")
        model, p0, param_names = _get_model_and_params("simple_decay.net")
        n_points = 51

        # jacfwd
        def solve_flat(p):
            return differentiable_solve(model, p, (0, 10), n_points).ravel()

        J_jax = np.asarray(jax.jacfwd(solve_flat)(p0))

        # CVODES direct
        model2 = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(
            model2,
            method="ode",
            sensitivity_params=list(param_names),
        )
        result = sim.run(t_span=(0, 10), n_points=n_points)
        # sens: (n_times, n_species, n_params) → reshape to match
        sens = result.sensitivities
        n_sp = model.n_species
        J_cvodes = sens.reshape(n_points * n_sp, len(param_names))

        np.testing.assert_allclose(
            J_jax,
            J_cvodes,
            atol=1e-5,
            rtol=1e-4,
            err_msg="jacfwd != CVODES sensitivity tensor",
        )


# ── Composition ──────────────────────────────────────────────────────


class TestComposition:
    """Composition with other JAX operations."""

    def test_grad_of_sum(self):
        """jax.grad(lambda p: sum(solve(m, p, ...)[:, 0])) works.

        Note: sum over ALL species is zero for simple_decay because
        A + B is conserved. So we sum only species 0 (A).
        """
        model, p0, _ = _get_model_and_params("simple_decay.net")

        grad = jax.grad(lambda p: jnp.sum(differentiable_solve(model, p, (0, 10), 11)[:, 0]))(p0)
        assert grad.shape == p0.shape
        assert jnp.any(jnp.abs(grad) > 1e-10)

    def test_grad_of_final_state(self):
        """Gradient of a function of the final species state."""
        model, p0, _ = _get_model_and_params("simple_decay.net")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 11)
            return Y[-1, 0]  # final value of species 0

        grad = jax.grad(loss)(p0)
        assert grad.shape == p0.shape
        # For decay, dA_final/dk1 should be negative
        assert float(grad[0]) < 0


# ── Fixed species ────────────────────────────────────────────────────


class TestFixedSpecies:
    """Models with fixed (boundary) species."""

    def test_fixed_species_gradient(self):
        """Gradient should still work when model has fixed species."""
        try:
            model, p0, _ = _get_model_and_params("fixed_species.net")
        except (AssertionError, FileNotFoundError):
            pytest.skip("fixed_species.net not available")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 11)
            return jnp.sum(Y**2)

        grad = jax.grad(loss)(p0)
        assert grad.shape == p0.shape


# ── Solver options ───────────────────────────────────────────────────


class TestSolverOptions:
    """Solver options (rtol, atol, max_steps) propagation."""

    def test_solver_options_propagate(self):
        """Custom rtol/atol/max_steps are used by the solver."""
        model, p0, _ = _get_model_and_params("simple_decay.net")

        # Should not raise with reasonable options
        Y = differentiable_solve(
            model,
            p0,
            (0, 10),
            11,
            rtol=1e-6,
            atol=1e-6,
            max_steps=5000,
        )
        assert Y.shape == (11, model.n_species)

    def test_tight_tolerances(self):
        """Tighter tolerances should give similar but not identical result."""
        model, p0, _ = _get_model_and_params("simple_decay.net")

        Y_loose = differentiable_solve(
            model,
            p0,
            (0, 10),
            101,
            rtol=1e-4,
            atol=1e-4,
        )
        Y_tight = differentiable_solve(
            model,
            p0,
            (0, 10),
            101,
            rtol=1e-12,
            atol=1e-12,
        )

        # Both should be close but not identical
        max_diff = float(jnp.max(jnp.abs(Y_loose - Y_tight)))
        assert max_diff < 1e-2  # loose tolerance → small diff
        assert max_diff > 1e-12  # not exactly the same


# ── Error cases ──────────────────────────────────────────────────────


class TestErrors:
    """Error handling."""

    def test_wrong_param_length_raises(self):
        """Passing wrong number of parameters should raise."""
        model, p0, _ = _get_model_and_params("simple_decay.net")
        wrong_p = jnp.array([1.0, 2.0, 3.0])  # wrong length

        with pytest.raises(ValueError, match="params length"):
            differentiable_solve(model, wrong_p, (0, 10), 11)

    def test_import_guard(self):
        """bngsim.jax submodule should be importable when JAX is present."""
        # If we got this far, JAX is available
        from bngsim.jax import differentiable_solve as ds

        assert callable(ds)


# ── Thread safety ────────────────────────────────────────────────────


class TestThreadSafety:
    """Model cloning ensures thread safety."""

    def test_concurrent_grad_calls(self):
        """Multiple concurrent jax.grad calls don't interfere."""
        from concurrent.futures import ThreadPoolExecutor

        model, p0, _ = _get_model_and_params("simple_decay.net")

        def compute_grad(scale):
            p = p0 * scale

            def loss(p):
                Y = differentiable_solve(model, p, (0, 10), 11)
                return jnp.sum(Y**2)

            return np.asarray(jax.grad(loss)(p))

        # Run 3 gradient computations with different param scales
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(compute_grad, s) for s in [0.5, 1.0, 2.0]]
            grads = [f.result() for f in futures]

        # All should complete without error and be nonzero
        for g in grads:
            assert g.shape == (len(p0),)
            assert np.any(np.abs(g) > 1e-10)

        # Different scales should give different gradients
        assert not np.allclose(grads[0], grads[1])
        assert not np.allclose(grads[1], grads[2])


# ── Chunked parallel sensitivity ────────────────────────────────────


class TestChunkedSensitivity:
    """Tests for chunk_size/n_workers parallel sensitivity path."""

    def test_chunked_gradient_matches_serial(self):
        """Gradient with chunk_size=1 matches serial (chunk_size=0)."""
        model, p0, _ = _get_model_and_params("two_species_reversible.net")

        def loss_serial(p):
            Y = differentiable_solve(model, p, (0, 10), 51)
            return jnp.sum(Y**2)

        def loss_chunked(p):
            Y = differentiable_solve(
                model,
                p,
                (0, 10),
                51,
                chunk_size=1,
                n_workers=2,
            )
            return jnp.sum(Y**2)

        grad_serial = np.asarray(jax.grad(loss_serial)(p0))
        grad_chunked = np.asarray(jax.grad(loss_chunked)(p0))

        np.testing.assert_allclose(
            grad_serial,
            grad_chunked,
            atol=1e-5,
            rtol=1e-4,
            err_msg="Chunked gradient != serial gradient",
        )

    def test_chunked_primal_matches(self):
        """Primal output identical regardless of chunk_size."""
        model, p0, _ = _get_model_and_params("simple_decay.net")

        Y_serial = differentiable_solve(model, p0, (0, 10), 11)
        Y_chunked = differentiable_solve(
            model,
            p0,
            (0, 10),
            11,
            chunk_size=1,
            n_workers=1,
        )

        np.testing.assert_allclose(
            np.asarray(Y_serial),
            np.asarray(Y_chunked),
            atol=1e-10,
        )

    def test_chunk_size_2(self):
        """chunk_size=2 (optimal from Session 27) works correctly."""
        model, p0, _ = _get_model_and_params("two_species_reversible.net")

        def loss(p):
            Y = differentiable_solve(
                model,
                p,
                (0, 10),
                51,
                chunk_size=2,
                n_workers=2,
            )
            return jnp.sum(Y[:, 0] ** 2)

        grad = jax.grad(loss)(p0)
        assert grad.shape == p0.shape
        assert jnp.any(jnp.abs(grad) > 1e-10)


# ── Finite-difference gradient verification ──────────────────────────


class TestFiniteDifferenceVerification:
    """Gold standard: compare jax.grad against brute-force FD."""

    def test_fd_gradient_simple_decay(self):
        """FD gradient matches jax.grad for simple decay."""
        model, p0, _ = _get_model_and_params("simple_decay.net")
        n_points = 51

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), n_points)
            return jnp.sum(Y[:, 0] ** 2)

        grad_jax = np.asarray(jax.grad(loss)(p0))

        # Brute-force central FD
        eps = 1e-5
        grad_fd = np.zeros(len(p0))
        for i in range(len(p0)):
            p_plus = np.array(p0, dtype=np.float64)
            p_minus = np.array(p0, dtype=np.float64)
            p_plus[i] += eps
            p_minus[i] -= eps
            l_plus = float(loss(jnp.array(p_plus)))
            l_minus = float(loss(jnp.array(p_minus)))
            grad_fd[i] = (l_plus - l_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-4,
            err_msg="jax.grad != FD gradient",
        )

    def test_fd_gradient_reversible(self):
        """FD gradient matches jax.grad for 2-param model."""
        model, p0, _ = _get_model_and_params("two_species_reversible.net")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 51)
            return jnp.sum(Y[:, 0] ** 2)

        grad_jax = np.asarray(jax.grad(loss)(p0))

        eps = 1e-5
        grad_fd = np.zeros(len(p0))
        for i in range(len(p0)):
            p_plus = np.array(p0, dtype=np.float64)
            p_minus = np.array(p0, dtype=np.float64)
            p_plus[i] += eps
            p_minus[i] -= eps
            l_plus = float(loss(jnp.array(p_plus)))
            l_minus = float(loss(jnp.array(p_minus)))
            grad_fd[i] = (l_plus - l_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="jax.grad != FD gradient (reversible)",
        )


# ── End-to-end L-BFGS-B optimization ────────────────────────────────


class TestLBFGSBOptimization:
    """Actually converge scipy L-BFGS-B to a known optimum."""

    def test_lbfgsb_recovers_true_params(self):
        """Generate synthetic data at k1=0.1, fit from k1=0.5.

        simple_decay: A(0)=100, dA/dt=-k1*A, B=100-A.
        True k1=0.1. Start from k1=0.5. L-BFGS-B should recover.
        """
        from scipy.optimize import minimize

        model = bngsim.Model.from_net(_get_net("simple_decay.net"))
        n_points = 101

        # Generate "observed" data at true params
        p_true = jnp.array([0.1])
        data = np.asarray(
            differentiable_solve(
                model,
                p_true,
                (0, 10),
                n_points,
            )
        )

        # Objective using JAX bridge
        def objective(p_vec):
            p = jnp.array(p_vec)

            def jax_loss(p):
                Y = differentiable_solve(model, p, (0, 10), n_points)
                return jnp.sum((Y - data) ** 2)

            val, grad = jax.value_and_grad(jax_loss)(p)
            return float(val), np.asarray(grad)

        result = minimize(
            objective,
            x0=[0.5],  # wrong initial guess
            method="L-BFGS-B",
            jac=True,
            bounds=[(1e-4, 10.0)],
        )
        assert result.success, f"L-BFGS-B failed: {result.message}"
        assert result.x[0] == pytest.approx(0.1, abs=1e-4), (
            f"Recovered k1={result.x[0]}, expected 0.1"
        )


# ── Functional rate model ────────────────────────────────────────────


class TestFunctionalRates:
    """Test with Functional rate models (CVODES FD fallback)."""

    def test_functional_rate_gradient(self):
        """Gradient should work for models with Functional rates.

        time_dependent_func.net has Functional rate laws. CVODES
        uses internal FD for sensitivity (not analytical).
        """
        try:
            model, p0, _ = _get_model_and_params("time_dependent_func.net")
        except (AssertionError, FileNotFoundError):
            pytest.skip("time_dependent_func.net not available")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 51)
            return jnp.sum(Y**2)

        grad = jax.grad(loss)(p0)
        assert grad.shape == p0.shape
        # Gradient should be finite (no NaN from FD sensitivity)
        assert jnp.all(jnp.isfinite(grad))


# ── Built-in objectives + JAX consistency ────────────────────────────


class TestObjectiveJAXConsistency:
    """Built-in objective gradients match JAX bridge gradients."""

    def test_sse_matches_jax_grad(self):
        """result.sse_gradient matches jax.grad(sse_loss)."""
        net_path = _get_net("two_species_reversible.net")
        model, p0, param_names = _get_model_and_params("two_species_reversible.net")
        n_points = 51
        data = np.zeros((n_points, model.n_species))

        # JAX path
        def jax_loss(p):
            Y = differentiable_solve(model, p, (0, 10), n_points)
            return jnp.sum((Y - data) ** 2)

        grad_jax = np.asarray(jax.grad(jax_loss)(p0))

        # Built-in path (via CVODES sensitivity)
        model2 = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(
            model2,
            method="ode",
            sensitivity_params=list(param_names),
        )
        result = sim.run(t_span=(0, 10), n_points=n_points)
        _, grad_builtin = result.sse_gradient(data)

        np.testing.assert_allclose(
            grad_jax,
            grad_builtin,
            atol=1e-5,
            rtol=1e-4,
            err_msg="JAX grad != sse_gradient",
        )


# ── Two-parameter model ─────────────────────────────────────────────


class TestTwoParamModel:
    """Tests with two-species reversible reaction (kf, kr)."""

    def test_both_gradients_nonzero(self):
        """Both kf and kr should have nonzero gradient."""
        model, p0, _ = _get_model_and_params("two_species_reversible.net")

        def loss(p):
            Y = differentiable_solve(model, p, (0, 10), 51)
            return jnp.sum(Y**2)

        grad = jax.grad(loss)(p0)
        # Both parameters should contribute to the gradient
        assert jnp.abs(grad[0]) > 1e-10, "kf gradient is zero"
        assert jnp.abs(grad[1]) > 1e-10, "kr gradient is zero"


# ── Derived ConstantExpression parameters (issue #2 follow-up) ───────


class TestPrimaryParamsDefault:
    """Default ``flat=False`` mode: differentiate over primary params only.

    Models exported by BNG2.pl carry derived ``_rateLaw{N}`` parameters
    (e.g., ``chi*kon``). The default JAX bridge should:

    1. Take a vector sized to ``model.primary_param_names`` (excludes
       ``_rateLaw{N}`` entries).
    2. Re-evaluate derived parameters from primaries on every call so
       ``jax.grad`` returns gradients with the chain rule applied.
    3. Match centred finite differences on the primary itself.
    """

    NET = "derived_rate_const.net"

    def _model_and_p0(self):
        model = bngsim.Model.from_net(_get_net(self.NET))
        # New default: use primaries only.
        names = model.primary_param_names
        p0 = jnp.array([model.get_param(n) for n in names], dtype=jnp.float64)
        return model, p0, names

    def test_primary_only_vector_size(self):
        model, p0, names = self._model_and_p0()
        assert "_rateLaw1" not in names, (
            "primary_param_names should exclude derived _rateLaw* entries"
        )
        assert p0.shape == (len(names),)
        assert p0.shape[0] < len(model.param_names)

    def test_primary_grad_kon_matches_fd(self):
        """jax.grad w.r.t. kon must reflect the chi*kon chain rule."""
        model, p0, names = self._model_and_p0()
        i_kon = names.index("kon")
        n_pts = 51
        SOLVER = dict(rtol=1e-10, atol=1e-12, max_steps=10**7)

        def loss(p):
            Y = differentiable_solve(model, p, (0, 5.0), n_pts, **SOLVER)
            return Y[-1, 1]  # final B

        grad = jax.grad(loss)(p0)

        # Brute-force centred FD on the primary kon (set_param re-evaluates
        # _rateLaw1 = chi*kon automatically inside _run_primal).
        eps = 1e-5

        def L(kon_val):
            p = p0.at[i_kon].set(kon_val)
            return float(loss(p))

        kon0 = float(p0[i_kon])
        fd = (L(kon0 * (1 + eps)) - L(kon0 * (1 - eps))) / (2 * eps * kon0)

        # Sanity: FD should be non-trivial (chi=10, so the indirect term
        # dominates and kon visibly affects B at the final time).
        assert abs(fd) > 1e-3, f"FD reference for d(B_final)/dkon is suspiciously small ({fd})"
        np.testing.assert_allclose(
            float(grad[i_kon]),
            fd,
            rtol=5e-3,
            err_msg="jax.grad[kon] does not include chain rule via _rateLaw1",
        )
        # Sign check is the original issue #2 diagnostic.
        assert np.sign(float(grad[i_kon])) == np.sign(fd), (
            "jax.grad[kon] has the wrong sign (issue #2 regression)"
        )

    def test_wrong_param_length_raises_for_default(self):
        """Passing a flat-length vector when flat=False should raise."""
        model = bngsim.Model.from_net(_get_net(self.NET))
        # Flat-length vector (includes _rateLaw1) is wrong for the default.
        p_flat = jnp.array(
            [model.get_param(n) for n in model.param_names],
            dtype=jnp.float64,
        )
        with pytest.raises(ValueError, match="primary parameter set"):
            differentiable_solve(model, p_flat, (0, 5.0), 11)


class TestFlatLegacyMode:
    """``flat=True`` reproduces the legacy independent-vector semantics."""

    NET = "derived_rate_const.net"

    def _model_and_flat_p0(self):
        model = bngsim.Model.from_net(_get_net(self.NET))
        names = model.param_names
        p0 = jnp.array([model.get_param(n) for n in names], dtype=jnp.float64)
        return model, p0, names

    def test_flat_vector_size(self):
        model, p0, names = self._model_and_flat_p0()
        Y = differentiable_solve(
            model,
            p0,
            (0, 5.0),
            11,
            flat=True,
            rtol=1e-10,
            atol=1e-12,
            max_steps=10**7,
        )
        assert Y.shape == (11, model.n_species)

    def test_flat_grad_kon_is_zero(self):
        """In flat mode, varying kon while holding _rateLaw1 fixed leaves B unchanged."""
        model, p0, names = self._model_and_flat_p0()
        i_kon = names.index("kon")
        i_rl = names.index("_rateLaw1")
        SOLVER = dict(rtol=1e-10, atol=1e-12, max_steps=10**7)

        def loss(p):
            Y = differentiable_solve(model, p, (0, 5.0), 51, flat=True, **SOLVER)
            return Y[-1, 1]

        grad = jax.grad(loss)(p0)
        # _rateLaw1 should carry the rate-constant gradient.
        assert abs(float(grad[i_rl])) > 1e-3
        # kon, treated as independent of _rateLaw1, should be ~zero.
        assert abs(float(grad[i_kon])) < 1e-9, (
            "flat=True should give grad[kon] ≈ 0 — kon does not appear "
            "in any reaction once _rateLaw1 is treated as independent"
        )

    def test_wrong_param_length_raises_for_flat(self):
        model = bngsim.Model.from_net(_get_net(self.NET))
        # Primary-length vector is wrong for flat=True.
        p_primary = jnp.array(
            [model.get_param(n) for n in model.primary_param_names],
            dtype=jnp.float64,
        )
        with pytest.raises(ValueError, match="flat parameter set"):
            differentiable_solve(model, p_primary, (0, 5.0), 11, flat=True)
