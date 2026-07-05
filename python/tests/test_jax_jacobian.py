"""Tests for JAX AD Jacobian (Session 22).

Tests the JAX RHS generator, expression translator, discontinuity
screening, and Jacobian correctness vs finite differences.

These tests require JAX: ``pip install jax jaxlib``
If JAX is not installed, all tests are skipped.
"""

import os

import numpy as np
import pytest

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)
ODE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "models", "net", "ode")

# Skip entire module if JAX is not available
try:
    import jax  # noqa: F401
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


class TestJaxAvailability:
    """Test JAX availability detection."""

    def test_jax_available(self):
        from bngsim._jax_rhs import jax_available

        assert jax_available() is True


class TestDiscontinuityScreening:
    """Test screening for discontinuous functions."""

    def test_clean_model(self):
        from bngsim._jax_rhs import screen_for_discontinuities

        path = os.path.join(DATA, "simple_decay.net")
        problems = screen_for_discontinuities(path)
        assert problems == []

    def test_clean_functional_model(self):
        from bngsim._jax_rhs import screen_for_discontinuities

        path = os.path.join(ODE_DIR, "CaOscillate_functional.net")
        if not os.path.exists(path):
            pytest.skip("CaOscillate_functional.net not found")
        problems = screen_for_discontinuities(path)
        assert problems == []


class TestJaxRhsGeneration:
    """Test JAX RHS function generation."""

    def test_simple_decay_rhs(self):
        """JAX RHS matches analytical: A -> B, k=0.1, A0=100."""
        from bngsim._jax_rhs import generate_jax_rhs

        path = os.path.join(DATA, "simple_decay.net")
        rhs = generate_jax_rhs(path)

        assert rhs.n_species == 2
        assert rhs.n_params == 1

        # Initial state
        y = jnp.array([100.0, 0.0])
        params = jnp.array([0.1])  # k1 = 0.1

        dydt = rhs(y, 0.0, params)
        dydt_np = np.asarray(dydt)

        # dA/dt = -k*A = -0.1*100 = -10
        # dB/dt = +k*A = +0.1*100 = +10
        assert abs(dydt_np[0] - (-10.0)) < 1e-10
        assert abs(dydt_np[1] - 10.0) < 1e-10

    def test_reversible_rhs(self):
        """JAX RHS for A + B <-> C."""
        from bngsim._jax_rhs import generate_jax_rhs

        path = os.path.join(DATA, "two_species_reversible.net")
        rhs = generate_jax_rhs(path)

        assert rhs.n_species == 3
        assert rhs.n_params == 2


class TestJaxJacobian:
    """Test JAX AD Jacobian computation."""

    def test_simple_decay_jacobian(self):
        """Jacobian of simple decay: J = [[-k, 0], [k, 0]]."""
        from bngsim._jax_rhs import generate_jax_jacobian

        path = os.path.join(DATA, "simple_decay.net")
        jac_fn = generate_jax_jacobian(path)

        y = jnp.array([100.0, 0.0])
        params = jnp.array([0.1])

        J = jac_fn(y, 0.0, params)
        J_np = np.asarray(J)

        # dA/dt = -k*A → ∂(dA/dt)/∂A = -k = -0.1
        # dB/dt = +k*A → ∂(dB/dt)/∂A = +k = +0.1
        assert abs(J_np[0, 0] - (-0.1)) < 1e-10
        assert abs(J_np[1, 0] - 0.1) < 1e-10
        # No dependence on B
        assert abs(J_np[0, 1]) < 1e-10
        assert abs(J_np[1, 1]) < 1e-10

    def test_reversible_jacobian_vs_fd(self):
        """Compare JAX Jacobian vs finite differences for A+B<->C."""
        from bngsim._jax_rhs import generate_jax_jacobian

        path = os.path.join(DATA, "two_species_reversible.net")
        jac_fn = generate_jax_jacobian(path)
        rhs = jac_fn.rhs

        y = jnp.array([50.0, 30.0, 20.0])
        params = jnp.array([0.001, 0.1])

        J_jax = np.asarray(jac_fn(y, 0.0, params))

        # Finite difference Jacobian
        n = len(y)
        J_fd = np.zeros((n, n))
        h = 1e-7
        f0 = np.asarray(rhs(y, 0.0, params))
        for j in range(n):
            y_pert = np.array(y)
            y_pert[j] += h
            f1 = np.asarray(rhs(jnp.array(y_pert), 0.0, params))
            J_fd[:, j] = (f1 - f0) / h

        # Compare
        max_err = np.max(np.abs(J_jax - J_fd))
        assert max_err < 1e-5, f"JAX vs FD max error: {max_err}"

    def test_functional_rate_jacobian(self):
        """JAX Jacobian for CaOscillate (Functional rates)."""
        from bngsim._jax_rhs import generate_jax_jacobian

        path = os.path.join(ODE_DIR, "CaOscillate_functional.net")
        if not os.path.exists(path):
            pytest.skip("CaOscillate_functional.net not found")

        jac_fn = generate_jax_jacobian(path)

        # Use model's initial conditions
        from bngsim._codegen import _parse_net_file

        model = _parse_net_file(path)
        n_sp = len(model["species"])

        # Evaluate parameter values
        param_vals = []
        for _, _name, expr, _is_const in model["parameters"]:
            try:
                val = float(expr)
            except (ValueError, TypeError):
                val = 1.0  # placeholder for expressions
            param_vals.append(val)
        params = jnp.array(param_vals, dtype=jnp.float64)

        # Initial species: use simple values
        y = jnp.ones(n_sp, dtype=jnp.float64) * 1000.0

        J_jax = np.asarray(jac_fn(y, 0.0, params))
        assert J_jax.shape == (n_sp, n_sp)
        # Should have non-zero entries (it's a coupled system)
        assert np.any(J_jax != 0.0)


class TestPrepareJaxJacobian:
    """Test the prepare_jax_jacobian entry point."""

    def test_prepare_returns_callable(self):
        from bngsim._jax_rhs import prepare_jax_jacobian

        path = os.path.join(DATA, "simple_decay.net")
        eval_fn, n_sp = prepare_jax_jacobian(path)

        assert n_sp == 2
        assert callable(eval_fn)

    def test_prepare_column_major_output(self):
        """Verify output is column-major flat array."""
        from bngsim._jax_rhs import prepare_jax_jacobian

        path = os.path.join(DATA, "simple_decay.net")
        eval_fn, n_sp = prepare_jax_jacobian(path)

        y = np.array([100.0, 0.0])
        params = np.array([0.1])

        flat_jac = eval_fn(y, 0.0, params)
        assert flat_jac.shape == (4,)  # 2x2 flattened

        # Column-major: col 0 = [J[0,0], J[1,0]], col 1 = [J[0,1], J[1,1]]
        # J[0,0] = -k = -0.1, J[1,0] = +k = +0.1
        assert abs(flat_jac[0] - (-0.1)) < 1e-10  # col 0, row 0
        assert abs(flat_jac[1] - 0.1) < 1e-10  # col 0, row 1
        assert abs(flat_jac[2]) < 1e-10  # col 1, row 0
        assert abs(flat_jac[3]) < 1e-10  # col 1, row 1


class TestExpressionTranslation:
    """Test .net expression -> JAX translation."""

    def test_translate_simple(self):
        from bngsim._jax_rhs import _translate_expr_jax

        expr = "k3/(K4+G)"
        param_names = {"k3": 0, "K4": 1}
        obs_names = {"G": 0}

        result = _translate_expr_jax(expr, param_names, obs_names, set(), [])
        assert "params[0]" in result
        assert "params[1]" in result
        assert "obs[0]" in result

    def test_translate_time(self):
        from bngsim._jax_rhs import _translate_expr_jax

        expr = "time()+1"
        result = _translate_expr_jax(expr, {}, {}, set(), [])
        assert "t+1" in result or "t +1" in result

    def test_translate_exponentiation(self):
        from bngsim._jax_rhs import _translate_expr_jax

        expr = "k*(x^2)"
        result = _translate_expr_jax(expr, {"k": 0, "x": 1}, {}, set(), [])
        assert "**" in result
