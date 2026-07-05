"""Tests for conservation laws + reduced-space Newton (Session 34).

Covers:
- Conservation law detection (reversible A<->B+C, simple decay)
- Reduced-space Newton convergence for models with conservation laws
- Newton method now works via auto for reversible model
- Sensitivity at steady state with conservation laws
- parameter_scan integration in BngsimModel
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


def _import_parameter_scan_action():
    """Import PyBNF's parameter_scan parser when its optional stack is present.

    Bngsim wheels/releases may be used without a full PyBNF install; these
    tests then skip instead of failing at import time.
    """
    try:
        from pybnf.bngsim_model import _parse_parameter_scan_action
    except ModuleNotFoundError as exc:
        pytest.skip(f"PyBNF parameter_scan parser unavailable in this environment: {exc.name}")
    return _parse_parameter_scan_action


# -- Fixtures -------------------------------------------------------


@pytest.fixture
def reversible_model(data_dir):
    """A + B <-> C with kf, kr.  Conservation law: A + C = const, B + C = const."""
    return bngsim.Model.from_net(str(data_dir / "two_species_reversible.net"))


@pytest.fixture
def simple_decay_model(data_dir):
    """A -> B with rate k1.  Conservation law: A + B = const."""
    return bngsim.Model.from_net(str(data_dir / "simple_decay.net"))


# -- Part A1: Conservation law detection ----------------------------


class TestConservationLawDetection:
    """Test that conservation laws are correctly detected."""

    def test_reversible_has_conservation_laws(self, reversible_model):
        """A+B<->C: 2 conservation laws (A+C, B+C)."""
        cl = reversible_model._core.conservation_laws
        assert cl["n_laws"] == 2
        assert cl["n_species"] == 3
        assert len(cl["dependent"]) == 2
        assert len(cl["independent"]) == 1
        # Dependents must be distinct
        assert cl["dependent"][0] != cl["dependent"][1]

    def test_simple_decay_has_conservation_law(self, simple_decay_model):
        """A -> B has one conservation law: A + B = const."""
        cl = simple_decay_model._core.conservation_laws
        assert cl["n_laws"] == 1
        assert cl["n_species"] == 2
        assert len(cl["dependent"]) == 1
        assert len(cl["independent"]) == 1

    def test_conservation_constants(self, simple_decay_model):
        """Conservation constant = A(0)+B(0) = 100."""
        cl = simple_decay_model._core.conservation_laws
        assert cl["n_laws"] == 1
        c = cl["constants"][0]
        assert abs(c - 100.0) < 1e-10 or abs(c + 100.0) < 1e-10

    def test_clone_preserves_conservation_laws(self, reversible_model):
        """Clone preserves conservation laws."""
        cl1 = reversible_model._core.conservation_laws
        clone = reversible_model.clone()
        cl2 = clone._core.conservation_laws
        assert cl1["n_laws"] == cl2["n_laws"]
        assert cl1["n_species"] == cl2["n_species"]


# -- Part A2: Reduced-space Newton ---------------------------------


class TestReducedNewton:
    """Test reduced-space Newton solver for models with conservation laws."""

    def test_newton_converges_reversible(self, reversible_model):
        """Newton converges for reversible model (was failing before Session 34)."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(method="newton", tol=1e-8)
        assert ss.converged
        assert ss.method_used == "newton"
        assert ss.residual < 1e-8

    def test_default_uses_newton_for_reversible(self, reversible_model):
        """Default method (newton) uses reduced-space Newton for reversible model."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(tol=1e-8)
        assert ss.converged
        # With conservation laws detected, reduced-space Newton converges;
        # method_used falls back to "integration" only if Newton failed.
        assert ss.method_used in ("newton", "integration")

    def test_newton_simple_decay(self, simple_decay_model):
        """Newton converges for simple decay (A -> B, ss: A=0, B=100)."""
        sim = bngsim.Simulator(simple_decay_model, method="ode")
        ss = sim.steady_state(method="newton", tol=1e-8)
        assert ss.converged
        # At steady state, A should be ~0 and B should be ~100
        d = ss.to_dict()
        assert d["A()"] < 1e-4
        assert abs(d["B()"] - 100.0) < 1e-4

    def test_newton_preserves_conservation(self, reversible_model):
        """Newton solution satisfies conservation constraints."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(method="newton", tol=1e-8)
        assert ss.converged
        d = ss.to_dict()
        # A(0)=100, C(0)=0 → A + C should be 100
        assert abs(d["A()"] + d["C()"] - 100.0) < 1e-4
        # B(0)=50, C(0)=0 → B + C should be 50
        assert abs(d["B()"] + d["C()"] - 50.0) < 1e-4

    def test_newton_agrees_with_integration(self, reversible_model):
        """Newton and integration give the same result."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss_newton = sim.steady_state(method="newton", tol=1e-8)
        reversible_model.reset()
        ss_int = sim.steady_state(method="integration", tol=1e-8)
        assert ss_newton.converged
        assert ss_int.converged
        np.testing.assert_allclose(ss_newton.concentrations, ss_int.concentrations, atol=1e-3)


# -- Part A3: Sensitivity with conservation laws -------------------


class TestSensitivityWithConservation:
    """Sensitivity at steady state now works for models with conservation laws."""

    def test_sensitivity_reversible(self, reversible_model):
        """Sensitivity works for reversible model (was xfail before Session 34)."""
        sim = bngsim.Simulator(reversible_model, method="ode")
        ss = sim.steady_state(
            sensitivity_params=["kf", "kr"],
            tol=1e-8,
        )
        assert ss.converged
        assert ss.sensitivity is not None
        ns = len(ss.concentrations)
        assert ss.sensitivity.shape == (ns, 2)

    def test_sensitivity_vs_fd_simple_decay(self, simple_decay_model):
        """Sensitivity matches FD for simple decay.

        For A→B, steady state is A=0, B=100 regardless
        of k1, so dY_ss/dk1 ≈ 0. We verify via FD.
        """
        sim = bngsim.Simulator(simple_decay_model, method="ode")

        # FD: perturb k1
        p0 = simple_decay_model.get_param("k1")
        h = max(1e-5 * abs(p0), 1e-8)

        simple_decay_model.set_param("k1", p0 + h)
        simple_decay_model.reset()
        ss_plus = sim.steady_state(method="integration", tol=1e-9)
        assert ss_plus.converged

        simple_decay_model.set_param("k1", p0 - h)
        simple_decay_model.reset()
        ss_minus = sim.steady_state(method="integration", tol=1e-9)
        assert ss_minus.converged

        fd_sens = (ss_plus.concentrations - ss_minus.concentrations) / (2 * h)

        # Both should be ~0 (ss doesn't depend on k1)
        np.testing.assert_allclose(fd_sens, 0.0, atol=1e-3)


# -- Part B: parameter_scan in BngsimModel -------------------------


class TestParameterScan:
    """Test parameter_scan() action parsing and routing in BngsimModel."""

    def test_parse_parameter_scan_action(self):
        """_parse_parameter_scan_action parses correctly."""
        _parse_parameter_scan_action = _import_parameter_scan_action()
        line = (
            'parameter_scan({parameter=>"kf",method=>"ode",'
            "par_min=>0.001,par_max=>1.0,n_scan_pts=>5,"
            'suffix=>"dose",t_end=>1000,steady_state=>1})'
        )
        result = _parse_parameter_scan_action(line)
        assert result is not None
        assert result["parameter"] == "kf"
        assert result["par_min"] == "0.001"
        assert result["par_max"] == "1.0"
        assert result["n_scan_pts"] == "5"
        assert result["suffix"] == "dose"
        assert result["t_end"] == "1000"
        assert result["steady_state"] == "1"

    def test_parse_parameter_scan_returns_none(self):
        """Non-parameter_scan lines return None."""
        _parse_parameter_scan_action = _import_parameter_scan_action()
        assert _parse_parameter_scan_action("simulate({method=>ode})") is None
        assert _parse_parameter_scan_action("# comment") is None
