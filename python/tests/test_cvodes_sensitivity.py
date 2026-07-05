"""Tests for CVODES forward sensitivity analysis (Session 26).

Verifies that the CVODES integration produces correct dY/dp sensitivities
alongside the ODE solution, with zero regressions to existing behavior.
"""

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

# Use the .net files from the existing test data. Honor BNGSIM_TEST_DATA so
# this module works under run_tests.sh, which copies tests to a temp dir
# (breaking __file__-relative resolution).
_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_simple_decay_net() -> str:
    """Return path to simple_decay.net (A --k1--> B, k1=0.1, A0=100, B0=0)."""
    p = DATA_DIR / "simple_decay.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_reversible_net() -> str:
    """Return path to two_species_reversible.net (A <--kf/kr--> B)."""
    p = DATA_DIR / "two_species_reversible.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


# ── Test: no sensitivities by default ────────────────────────────────────────


class TestNoSensitivities:
    """When sensitivity_params is empty/None, no sensitivity data."""

    def test_default_no_sensitivity(self):
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        assert not result.has_sensitivities
        assert result.sensitivities.shape == (0, 0, 0)
        assert result.sensitivity_params == []

    def test_empty_list_no_sensitivity(self):
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=[])
        result = sim.run(t_span=(0, 10), n_points=11)

        assert not result.has_sensitivities


# ── Test: simple decay analytical check ──────────────────────────────────────


class TestSimpleDecaySensitivity:
    """dA/dt = -k1*A, A(0)=100, k1=0.1.

    Analytical: A(t) = A0*exp(-k1*t)
    dA/dk1 = -t * A0 * exp(-k1*t) = -t * A(t)
    """

    def test_shape(self):
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        assert result.has_sensitivities
        # (11 times, 2 species [A, B], 1 param)
        assert result.sensitivities.shape == (11, 2, 1)
        assert result.sensitivity_params == ["k1"]

    def test_analytical_accuracy(self):
        """Compare CVODES dA/dk1 against analytical -t*A(t)."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=101)

        k1 = 0.1
        A0 = 100.0
        t = result.time
        # Analytical sensitivity: dA/dk1 = -t * A0 * exp(-k1*t)
        analytical_dA_dk1 = -t * A0 * np.exp(-k1 * t)

        # CVODES sensitivity for species A (index 0), param k1 (index 0)
        cvodes_sens = result.sensitivities[:, 0, 0]

        # Should match to high precision (CVODES internal FD is accurate)
        max_err = np.max(np.abs(cvodes_sens - analytical_dA_dk1))
        assert max_err < 1.0, f"Max error {max_err} too large"

    def test_zero_at_t0(self):
        """Sensitivities should be zero at t=0 (zero IC)."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        # All sensitivities zero at t=0
        assert np.all(result.sensitivities[0] == 0.0)


# ── Test: two-parameter reversible model ─────────────────────────────────────


class TestReversibleSensitivity:
    """A <--kf/kr--> B with sensitivity w.r.t. both kf and kr."""

    def test_two_params(self):
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
        result = sim.run(t_span=(0, 10), n_points=11)

        ns = model.n_species
        assert result.sensitivities.shape == (11, ns, 2)
        assert result.sensitivity_params == ["kf", "kr"]

    def test_sensitivities_nonzero(self):
        """At t>0, sensitivities should be nonzero."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
        result = sim.run(t_span=(0, 10), n_points=11)

        # At the last time point, sensitivities should be nonzero
        sens_final = result.sensitivities[-1]
        assert np.any(np.abs(sens_final) > 1e-10)


# ── Test: invalid parameter name ─────────────────────────────────────────────


class TestInvalidParam:
    def test_invalid_param_raises(self):
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["nonexistent_param"])
        with pytest.raises(Exception, match="nonexistent_param"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_sensitivity_ssa_raises(self):
        model = bngsim.Model.from_net(_get_simple_decay_net())
        with pytest.raises(ValueError, match="only supported"):
            bngsim.Simulator(model, method="ssa", sensitivity_params=["k1"])


# ── Test: ODE solution unchanged when sensitivities enabled ──────────────────


class TestSolutionUnchanged:
    """ODE trajectories should be identical with and without sensitivities."""

    def test_species_identical(self):
        net_path = _get_simple_decay_net()

        # Without sensitivities
        m1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(m1, method="ode")
        r1 = sim1.run(t_span=(0, 10), n_points=101)

        # With sensitivities
        m2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(m2, method="ode", sensitivity_params=["k1"])
        r2 = sim2.run(t_span=(0, 10), n_points=101)

        # Species should be identical
        np.testing.assert_allclose(r1.species, r2.species, atol=1e-10)


# ── Test: sensitivity method selection ───────────────────────────────────────


class TestSensitivityMethod:
    """Test that sensitivity_method parameter works correctly."""

    def test_staggered_default(self):
        """Default method should be staggered."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.has_sensitivities
        # Both methods should produce sensitivities
        assert result.sensitivities.shape == (11, 2, 1)

    def test_simultaneous_method(self):
        """Test explicit simultaneous method."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(
            model, method="ode", sensitivity_params=["k1"], sensitivity_method="simultaneous"
        )
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.has_sensitivities
        assert result.sensitivities.shape == (11, 2, 1)

    def test_invalid_method(self):
        """Invalid method should raise error."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        with pytest.raises(ValueError, match="sensitivity_method"):
            bngsim.Simulator(
                model, method="ode", sensitivity_params=["k1"], sensitivity_method="invalid"
            )

    def test_results_similar(self):
        """Both methods should produce similar results."""
        # Separate Model objects: CvodeSimulator.run() writes the final
        # species state back to the model (BNG-style), so reusing one
        # Model across two runs would start the second run from t=t_end.
        net_path = _get_simple_decay_net()
        model_stag = bngsim.Model.from_net(net_path)
        model_simul = bngsim.Model.from_net(net_path)

        sim_stag = bngsim.Simulator(
            model_stag, method="ode", sensitivity_params=["k1"], sensitivity_method="staggered"
        )
        result_stag = sim_stag.run(t_span=(0, 10), n_points=51)

        sim_simul = bngsim.Simulator(
            model_simul, method="ode", sensitivity_params=["k1"], sensitivity_method="simultaneous"
        )
        result_simul = sim_simul.run(t_span=(0, 10), n_points=51)

        # Results should be close (both are accurate CVODES methods)
        max_diff = np.max(np.abs(result_stag.sensitivities - result_simul.sensitivities))
        assert max_diff < 1e-3, f"Methods differ by {max_diff}"
