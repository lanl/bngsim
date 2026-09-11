"""The public NumPy evaluators on ``Model`` (issue #523).

``rhs``, ``jacobian``, ``propensities`` and ``stoichiometry_matrix`` are the
provider every analysis built outside bngsim assumes — a continuation needs the
RHS and Jacobian at states of its own choosing, a root classifier needs the
Jacobian at a root found elsewhere, a moment or CME generator needs the
stoichiometry and the propensity vector. The C++ evaluators behind them had
been there all along; none reached Python.

What these tests pin is the contract, not the arithmetic of any one model:
each evaluator is the one the corresponding engine runs (the CVODE callback's
RHS with its observable refresh, the analytical-or-differenced Jacobian rule
the solvers apply, the SSA loop's propensity pass), the state passed in is the
state evaluated — never the stored one — and the model's own concentrations
are left alone.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import JacobianMatrix

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def decay(data_dir):
    """A -> B at k1 = 0.1, A(0) = 100. One law: A + B."""
    return bngsim.Model.from_net(str(data_dir / "simple_decay.net"))


@pytest.fixture
def reversible(data_dir):
    """A + B <-> C, kf = 0.001, kr = 0.1. Two laws over three species."""
    return bngsim.Model.from_net(str(data_dir / "two_species_reversible.net"))


@pytest.fixture
def homotrimer(data_dir):
    """A + A + A -> C at (1/6)·k1: the reactant repeats, so the two rate
    conventions differ — k·x³/6 for the ODE, k·x(x−1)(x−2)/6 for the SSA."""
    return bngsim.Model.from_net(str(data_dir / "ssa_aaa.net"))


@pytest.fixture
def boundary(data_dir):
    """$A -> B: A is a fixed boundary species."""
    return bngsim.Model.from_net(str(data_dir / "fixed_species.net"))


@pytest.fixture
def functional(data_dir):
    """Three functional rate laws reading the A_tot / C_tot observables (see the
    fixture header). A BNGL functional rate multiplies the reactant count in,
    so A -> B at fRate0 = k1·A_tot runs at k1·A²; the two zeroth-order
    reactions run at their bare function values: f_A = -k1·A² - k1·A,
    f_B = k1·A², f_C = k2·C while A_tot <= 200 and 0 above it."""
    return bngsim.Model.from_net(str(data_dir / "func_composition.net"))


@pytest.fixture
def time_dependent(data_dir):
    """A -> A + B at rate time()+1, A(0) = 1: dB/dt = t + 1."""
    return bngsim.Model.from_net(str(data_dir / "time_dependent_func.net"))


@pytest.fixture
def table_function(data_dir):
    """A -> A + B at the tabulated rate cumNcases(t) — 22 at t = 7 — which the
    symbolic differentiator declines, so the analytical Jacobian is incomplete."""
    return bngsim.Model.from_net(str(data_dir / "tfun_time_indexed.net"))


# ── stoichiometry_matrix ────────────────────────────────────────────────────


class TestStoichiometryMatrix:
    def test_shape_orientation_and_sign(self, reversible):
        S = reversible.stoichiometry_matrix()
        assert S.shape == (reversible.n_species, reversible.n_reactions) == (3, 2)
        assert S.dtype == np.float64
        # Reaction 1: A + B -> C; reaction 2: C -> A + B.
        np.testing.assert_array_equal(S, [[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])

    def test_repeated_reactant_is_a_net_coefficient(self, homotrimer):
        """A + A + A -> C puts -3 in A's row, not three -1 entries."""
        np.testing.assert_array_equal(homotrimer.stoichiometry_matrix(), [[-3.0], [1.0]])

    def test_fixed_species_row_is_zero(self, boundary):
        """$A -> B: the RHS zeroes A's derivative and an SSA firing never
        updates A, so the matrix that describes the engines has no entry there."""
        np.testing.assert_array_equal(boundary.stoichiometry_matrix(), [[0.0], [1.0]])

    def test_conservation_laws_annihilate_it(self, reversible):
        """This is the matrix the laws were row-reduced from: L @ S == 0."""
        laws = reversible.conservation_laws
        L = np.asarray(laws["coefficients"], dtype=float)
        assert laws["n_laws"] == 2
        np.testing.assert_allclose(L @ reversible.stoichiometry_matrix(), 0.0, atol=1e-12)

    def test_rhs_is_S_times_the_ode_rates(self, reversible):
        """For a mass-action .net model f(y) = S · v(y) exactly."""
        y = np.array([100.0, 50.0, 3.0])
        kf, kr = reversible.get_param("kf"), reversible.get_param("kr")
        v = np.array([kf * y[0] * y[1], kr * y[2]])
        np.testing.assert_allclose(reversible.rhs(y), reversible.stoichiometry_matrix() @ v)

    def test_sparse_form_is_the_same_matrix(self, reversible):
        sparse = pytest.importorskip("scipy.sparse")
        S = reversible.stoichiometry_matrix(sparse=True)
        assert isinstance(S, sparse.csc_array)
        assert S.shape == (3, 2)
        np.testing.assert_array_equal(S.toarray(), reversible.stoichiometry_matrix())

    def test_each_call_is_a_fresh_matrix(self, decay):
        S = decay.stoichiometry_matrix()
        S[:] = 42.0
        np.testing.assert_array_equal(decay.stoichiometry_matrix(), [[-1.0], [1.0]])


# ── rhs ─────────────────────────────────────────────────────────────────────


class TestRhs:
    def test_simple_decay(self, decay):
        np.testing.assert_allclose(decay.rhs([100.0, 0.0]), [-10.0, 10.0])
        np.testing.assert_allclose(decay.rhs(np.array([20.0, 5.0])), [-2.0, 2.0])

    def test_reads_the_live_parameters(self, decay):
        decay.set_param("k1", 0.25)
        np.testing.assert_allclose(decay.rhs([100.0, 0.0]), [-25.0, 25.0])

    def test_evaluates_the_state_passed_in_not_the_stored_one(self, functional):
        """Observables are refreshed AT y, exactly as the CVODE callback does.

        The stored state is [100, 0, 50]; at A = 300 the A_tot > 200 branch of
        fRate_cond switches C's production off, which only happens if A_tot was
        recomputed from the argument."""
        np.testing.assert_allclose(functional.get_state(), [100.0, 0.0, 50.0])
        np.testing.assert_allclose(functional.rhs([100.0, 0.0, 50.0]), [-1010.0, 1000.0, 2.5])
        np.testing.assert_allclose(functional.rhs([300.0, 0.0, 50.0]), [-9030.0, 9000.0, 0.0])

    def test_leaves_the_stored_concentrations_alone(self, functional):
        before = functional.get_state().copy()
        functional.rhs([300.0, 7.0, 1.0])
        np.testing.assert_array_equal(functional.get_state(), before)

    def test_runs_the_observable_refresh_under_the_rhs_gate(self, decay, functional):
        """The same gate the CVODE callback takes: a pure mass-action model
        skips the observable/function passes, a functional one runs them."""
        for model, expect_refresh in ((decay, False), (functional, True)):
            model._core.reset_rhs_counters()
            model.rhs(model.get_state())
            assert model._core.rhs_eval_count == 1
            assert (model._core.rhs_observable_eval_count == 1) is expect_refresh

    def test_time_reaches_the_rate_law(self, time_dependent):
        y = [1.0, 0.0]
        np.testing.assert_allclose(time_dependent.rhs(y), [0.0, 1.0])
        np.testing.assert_allclose(time_dependent.rhs(y, t=5.0), [0.0, 6.0])

    def test_fixed_species_derivative_is_zero(self, boundary):
        np.testing.assert_allclose(boundary.rhs([100.0, 0.0]), [0.0, 10.0])

    def test_is_the_callback_evaluator(self, functional):
        """One implementation: the public method and the RHS test hook the
        Jacobian self-check has always used read the same function."""
        y = [250.0, 1.0, 9.0]
        np.testing.assert_array_equal(functional.rhs(y), functional._core._eval_rhs(0.0, y))

    @pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], [[1.0, 2.0]]])
    def test_rejects_a_wrong_shaped_state(self, decay, bad):
        with pytest.raises(ValueError, match="expected"):
            decay.rhs(bad)


# ── jacobian ────────────────────────────────────────────────────────────────


class TestJacobian:
    def test_simple_decay_closed_form(self, decay):
        J = decay.jacobian([100.0, 0.0])
        assert isinstance(J, JacobianMatrix)
        assert J.source == "analytical"
        assert J.shape == (2, 2)
        # Row i is ∂f_i, column j is ∂/∂x_j: f_B = k1·A, so J[B, A] = k1.
        np.testing.assert_allclose(J, [[-0.1, 0.0], [0.1, 0.0]])

    def test_bimolecular_closed_form(self, reversible):
        y = np.array([100.0, 50.0, 3.0])
        kf, kr = reversible.get_param("kf"), reversible.get_param("kr")
        row = [-kf * y[1], -kf * y[0], kr]  # ∂f_A/∂(A, B, C)
        expect = np.array([row, row, [-v for v in row]])
        J = reversible.jacobian(y)
        assert J.source == "analytical"
        np.testing.assert_allclose(J, expect, rtol=1e-12)

    def test_functional_closed_form(self, functional):
        """f_A = -k1·A² - k1·A, f_B = k1·A², f_C = k2·C inside the if(): the
        branch condition contributes no derivative."""
        J = functional.jacobian([100.0, 0.0, 50.0])
        assert J.source == "analytical"
        expect = np.zeros((3, 3))
        expect[0, 0], expect[1, 0], expect[2, 2] = -20.1, 20.0, 0.05
        np.testing.assert_allclose(J, expect, atol=1e-12)

    def test_reads_the_live_parameters(self, decay):
        decay.set_param("k1", 0.5)
        np.testing.assert_allclose(decay.jacobian([1.0, 1.0]), [[-0.5, 0.0], [0.5, 0.0]])

    def test_fixed_species_row_is_zero(self, boundary):
        np.testing.assert_allclose(boundary.jacobian([100.0, 0.0]), [[0.0, 0.0], [0.1, 0.0]])

    def test_difference_quotient_agrees_with_the_closed_form(self, reversible, functional):
        """The fallback is a faithful stand-in: the solver's own FD rule,
        applied to rhs, reproduces the analytical matrix to FD accuracy."""
        for model, y in ((reversible, [100.0, 50.0, 3.0]), (functional, [100.0, 0.0, 50.0])):
            analytical = model.jacobian(y)
            assert analytical.source == "analytical"
            fd = model._core.fill_dense_fd_jacobian(0.0, np.asarray(y))
            np.testing.assert_allclose(fd, analytical, rtol=1e-6, atol=1e-9)

    def test_incomplete_analytical_jacobian_is_differenced_and_says_so(self, table_function):
        """A tabulated rate law has no symbolic derivative. The partial closed
        form is never returned as if it were whole: the matrix is the
        difference quotient, and `source` reports it — the same label
        SteadyStateResult.solver_jacobian_source uses for the same fallback."""
        assert not table_function.prepare_analytical_jacobian()
        J = table_function.jacobian([1.0, 0.0], t=7.0)
        assert J.source == "finite-difference"
        # dB/dt = cumNcases(t)·A = 22·A at t = 7, and nothing else moves.
        np.testing.assert_allclose(J, [[0.0, 0.0], [22.0, 0.0]], rtol=1e-6, atol=1e-9)

    def test_sparse_form_is_the_same_matrix(self, reversible, table_function):
        sparse = pytest.importorskip("scipy.sparse")
        for model, y, t, source in (
            (reversible, [100.0, 50.0, 3.0], 0.0, "analytical"),
            (table_function, [1.0, 0.0], 7.0, "finite-difference"),
        ):
            dense = model.jacobian(y, t=t)
            M = model.jacobian(y, t=t, sparse=True)
            assert isinstance(M, sparse.csc_array)
            assert M.source == source == dense.source
            assert M.shape == dense.shape
            np.testing.assert_allclose(M.toarray(), dense, rtol=1e-12)
            assert M.nnz == model._core.jacobian_sparsity["nnz"]

    def test_source_survives_array_operations(self, decay):
        J = decay.jacobian([1.0, 0.0])
        assert J[:1].source == "analytical"
        assert (2.0 * J).source == "analytical"
        assert type(np.asarray(J)) is JacobianMatrix or np.asarray(J).shape == (2, 2)
        np.testing.assert_allclose(np.linalg.eigvals(J), [0.0, -0.1], atol=1e-15)

    @pytest.mark.parametrize("bad", [[1.0], [[1.0, 2.0]]])
    def test_rejects_a_wrong_shaped_state(self, decay, bad):
        with pytest.raises(ValueError, match="expected"):
            decay.jacobian(bad)


# ── propensities ────────────────────────────────────────────────────────────


class TestPropensities:
    def test_unimolecular_net_model_agrees_with_the_ode_rate(self, decay):
        y = np.array([100.0, 0.0])
        a = decay.propensities(y)
        assert a.shape == (1,)
        np.testing.assert_allclose(a, [10.0])
        np.testing.assert_allclose(decay.stoichiometry_matrix() @ a, decay.rhs(y))

    def test_repeated_reactant_takes_the_falling_factorial(self, homotrimer):
        """SSA convention: (1/6)·k·x(x−1)(x−2), where the ODE rate is k·x³/6."""
        np.testing.assert_allclose(homotrimer.propensities([3.0, 0.0]), [1.0])
        np.testing.assert_allclose(homotrimer.propensities([2.0, 0.0]), [0.0])
        np.testing.assert_allclose(homotrimer.rhs([3.0, 0.0]), [-13.5, 4.5])  # ODE: 3³/6 = 4.5

    def test_compartment_volume_multiplies_the_ode_rate(self):
        """The SSA volume convention the docstring promises: in a compartment of
        volume V the propensity is V times the ODE (storage-units) rate."""
        pytest.importorskip("antimony")
        model = bngsim.Model.from_antimony_string(
            "compartment C = 2; species S in C; S = 10; k = 0.1; J1: S -> ; k*S;"
        )
        y = model.get_state()
        np.testing.assert_allclose(model.rhs(y), [-0.5])  # -(k·S)/V
        np.testing.assert_allclose(model.propensities(y), [1.0])  # k·S = V·|rhs|

    def test_refreshes_observables_at_the_state_passed_in(self, functional):
        """The SSA loop's own refresh before its propensity pass, at y."""
        np.testing.assert_allclose(
            functional.propensities([100.0, 0.0, 50.0]), [1000.0, -10.0, 2.5]
        )
        np.testing.assert_allclose(
            functional.propensities([300.0, 0.0, 50.0]), [9000.0, -30.0, 0.0]
        )

    def test_leaves_the_stored_concentrations_alone(self, functional):
        before = functional.get_state().copy()
        functional.propensities([300.0, 0.0, 50.0])
        np.testing.assert_array_equal(functional.get_state(), before)

    def test_time_reaches_the_rate_law(self, time_dependent):
        np.testing.assert_allclose(time_dependent.propensities([1.0, 0.0], t=5.0), [6.0])

    def test_rejects_a_wrong_shaped_state(self, decay):
        with pytest.raises(ValueError, match="expected 2 species values"):
            decay.propensities([1.0, 2.0, 3.0])


# ── The issue's own reproducer ───────────────────────────────────────────────


def test_issue_523_reproducer(decay):
    """Every name the issue found missing is now present, on Model and on the core."""
    for name in ("rhs", "jacobian", "stoichiometry_matrix", "propensities"):
        assert hasattr(decay, name), name
    for name in (
        "compute_derivs",
        "fill_dense_analytical_jacobian",
        "compute_propensity",
        "stoichiometry",
    ):
        assert hasattr(decay._core, name), name
    ss = bngsim.Simulator(decay, method="ode").steady_state(method="newton")
    assert ss.root_stability == "stable"
    assert hasattr(ss, "eigenvalues")
