"""Tests for conservation laws + reduced-space Newton (Session 34).

Covers:
- Conservation law detection (reversible A<->B+C, simple decay)
- Reduced-space Newton convergence for models with conservation laws
- Newton method now works via auto for reversible model
- Sensitivity at steady state with conservation laws
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# -- Fixtures -------------------------------------------------------


@pytest.fixture
def reversible_model(data_dir):
    """A + B <-> C with kf, kr.  Conservation law: A + C = const, B + C = const."""
    return bngsim.Model.from_net(str(data_dir / "two_species_reversible.net"))


@pytest.fixture
def simple_decay_model(data_dir):
    """A -> B with rate k1.  Conservation law: A + B = const."""
    return bngsim.Model.from_net(str(data_dir / "simple_decay.net"))


@pytest.fixture
def singular_dep_net(data_dir):
    """Enzyme-substrate motif whose two laws both constrain the complex.

    See the header of the .net file. Choosing each law's dependent independently
    used to land on ``L[:, dependent] = [[-1,-1],[1,1]]`` — rank 1, so the
    elimination it sets up has no solution.
    """
    return str(data_dir / "conservation_singular_dep.net")


# -- Part A1: Conservation law detection ----------------------------


class TestConservationLawDetection:
    """Test that conservation laws are correctly detected."""

    def test_reversible_has_conservation_laws(self, reversible_model):
        """A+B<->C: 2 conservation laws (A+C, B+C)."""
        cl = reversible_model.conservation_laws
        assert cl["n_laws"] == 2
        assert cl["n_species"] == 3
        assert len(cl["dependent"]) == 2
        assert len(cl["independent"]) == 1
        # Dependents must be distinct
        assert cl["dependent"][0] != cl["dependent"][1]

    def test_simple_decay_has_conservation_law(self, simple_decay_model):
        """A -> B has one conservation law: A + B = const."""
        cl = simple_decay_model.conservation_laws
        assert cl["n_laws"] == 1
        assert cl["n_species"] == 2
        assert len(cl["dependent"]) == 1
        assert len(cl["independent"]) == 1

    def test_conservation_constants(self, simple_decay_model):
        """Conservation constant = A(0)+B(0) = 100."""
        cl = simple_decay_model.conservation_laws
        assert cl["n_laws"] == 1
        c = cl["constants"][0]
        assert abs(c - 100.0) < 1e-10 or abs(c + 100.0) < 1e-10

    def test_clone_preserves_conservation_laws(self, reversible_model):
        """Clone preserves conservation laws."""
        cl1 = reversible_model.conservation_laws
        clone = reversible_model.clone()
        cl2 = clone.conservation_laws
        assert cl1["n_laws"] == cl2["n_laws"]
        assert cl1["n_species"] == cl2["n_species"]


# -- Part A1b: the reduction has to be SOLVABLE, not just present ---


class TestDependentSpeciesChoice:
    """``L[:, dependent]`` must be the identity (issue #63 follow-up).

    Every consumer of these laws eliminates one species per law in a single pass:
    ``reconstruct_full`` solves law k for ``y[dep_k]`` from the current value of
    every other species, and ``compute_ss_sensitivity`` forward-substitutes the
    same walk to get ∂y_dep/∂y_ind. Both are exact only when no law constrains
    another law's dependent — i.e. when the dependent block is the identity.

    Picking each law's dependent independently guaranteed neither, and on 52 of
    the 374 ode_fullnet corpus models that have conservation laws it chose a set
    for which the block is outright SINGULAR. The failures were silent: the
    constraints came back violated, the reduced Jacobian picked up a null space of
    the reduction's own making, and the reduced Newton solve quietly fell back to
    integration.
    """

    @pytest.mark.parametrize(
        "net_name",
        [
            "conservation_singular_dep.net",  # singular block before the fix
            "two_species_reversible.net",
            "ssa_abc.net",
            "per_observable_jac.net",
            "simple_decay.net",
        ],
    )
    def test_dependent_block_is_identity(self, data_dir, net_name):
        model = bngsim.Model.from_net(str(data_dir / net_name))
        cl = model.conservation_laws
        if cl["n_laws"] == 0:
            pytest.skip(f"{net_name} has no conservation laws")
        coeffs = np.array([np.asarray(c, dtype=float) for c in cl["coefficients"]])
        dep = np.asarray(cl["dependent"], dtype=int)
        np.testing.assert_allclose(coeffs[:, dep], np.eye(dep.size), atol=1e-12)

    def test_reduction_satisfies_its_own_constraints(self, singular_dep_net):
        """D = ∂y_dep/∂y_ind must annihilate the constraints: L_ind + L_dep·D = 0.

        This is the quantity that came back as 1.0 instead of 0 — the reduction
        reporting sensitivities that break the conservation laws it enforces.
        """
        model = bngsim.Model.from_net(singular_dep_net)
        cl = model.conservation_laws
        coeffs = np.array([np.asarray(c, dtype=float) for c in cl["coefficients"]])
        dep = np.asarray(cl["dependent"], dtype=int)
        ind = np.asarray(cl["independent"], dtype=int)

        # D by the same forward substitution compute_ss_sensitivity uses.
        n_laws, n_ind = cl["n_laws"], ind.size
        D = np.zeros((n_laws, n_ind))
        for k in range(n_laws):
            d = dep[k]
            c_dep = coeffs[k][d]
            for j in range(n_ind):
                acc = coeffs[k][ind[j]] + sum(
                    coeffs[k][dep[kp]] * D[kp, j] for kp in range(k) if dep[kp] != d
                )
                D[k, j] = -acc / c_dep
        np.testing.assert_allclose(coeffs[:, ind] + coeffs[:, dep] @ D, 0.0, atol=1e-12)

    def test_reduced_newton_converges_and_conserves(self, singular_dep_net):
        """An unsolvable elimination made KINSOL fail and drop back to integration."""
        model = bngsim.Model.from_net(singular_dep_net)
        totals = model.conservation_laws["constants"]
        sim = bngsim.Simulator(model, method="ode")
        ss = sim.steady_state(method="newton", tol=1e-8)
        assert ss.converged
        assert ss.method_used == "newton"

        coeffs = np.array(
            [np.asarray(c, dtype=float) for c in model.conservation_laws["coefficients"]]
        )
        np.testing.assert_allclose(coeffs @ np.asarray(ss.concentrations), totals, rtol=1e-8)

    def test_steady_state_sensitivity_matches_finite_difference(self, singular_dep_net):
        """dY_ss/dp is a real gradient again, not the output of a singular solve.

        Before the fix the reduced Jacobian here was rank 1 of 2, so
        ``-J_red⁻¹·∂f/∂p`` was whatever the LU made of a singular system. The
        central difference of the steady state itself owes that solve nothing.
        """
        pname, h_frac = "kcat", 1e-4

        model = bngsim.Model.from_net(singular_dep_net)
        ss = bngsim.Simulator(model, method="ode").steady_state(
            sensitivity_params=[pname], tol=1e-12
        )
        assert ss.converged
        assert ss.sens_jacobian_rcond > 1e-6
        analytic = np.asarray(ss.sensitivity)[:, 0]

        p0 = bngsim.Model.from_net(singular_dep_net).get_param(pname)
        h = h_frac * abs(p0)
        roots = []
        for value in (p0 + h, p0 - h):
            m = bngsim.Model.from_net(singular_dep_net)
            m.set_param(pname, value)
            m.reset()
            r = bngsim.Simulator(m, method="ode").steady_state(tol=1e-12)
            assert r.converged
            roots.append(np.asarray(r.concentrations))
        fd = (roots[0] - roots[1]) / (2 * h)

        np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())


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


# -- Part B: parameter_scan parsing — tested in PyBNF ---------------
#
# The `_parse_parameter_scan_action` unit tests that lived here imported a private
# PyBNF helper across the repo boundary; they now sit with the code they cover, in
# PyBNF tests/test_bngsim_parsing.py (canonical keys + the decline-foreign-lines
# parametrization). Removed as part of lanl/bngsim#45 — see the longer note at the
# foot of test_sample_times.py for why this direction of coupling kept going stale
# without anyone's CI noticing.
#
# The prose above (`parameter_scan` routing through BNGsim's batch steady-state
# and time-course paths) still describes real behavior; what it does NOT need is
# for us to assert PyBNF's parser is correct.
