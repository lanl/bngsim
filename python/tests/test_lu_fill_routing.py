"""The LU-fill test in the linear-solver routing (bngsim/sparse_jacobian.hpp).

The size and density rule sends a model to KLU when its *Jacobian* is sparse,
but KLU factors the Newton matrix ``I - gamma*J``, and on rule-derived networks
that factor fills in: the Jacobian of ``egfr_net`` is 6.7% dense while KLU's own
symbolic analysis puts its LU at 30% of ``n²``. KLU has no BLAS, so such a model
spent most of every run refactoring, and the BLAS dense factor is several times
faster end to end. A model of 256 to 5,000 species whose estimated fill is 10% or
more now goes to that dense factor, when the build links one and the analytical
Jacobian is in use.

These tests pin both sides of the rule on the vendored networks — a fill-heavy
one that moves, a genuinely sparse one and an under-size one that stay — each
escape from it, that the route is only a cost (the trajectory and the
steady-state root do not move), and the two places that must agree with the
solver about it: the codegen emitter, which builds one Jacobian shape per model,
and ``steady_state()``, which routes its march by the same rule.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _codegen_backend import selected_codegen_backend  # noqa: E402

_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"

# LinearSolverKind codes — mirror include/bngsim/result.hpp.
LS_DENSE, LS_KLU, LS_LAPACK = 0, 1, 2

# The constants the rule uses (bngsim/sparse_jacobian.hpp).
LU_FILL_DENSE_MIN = 0.10
LU_FILL_DENSE_MIN_N = 256

# 356 species, Jacobian 6.7% dense, estimated LU fill 0.30: moves to dense.
_FILL_MODEL = "egfr_net.net"
# 300 species, 1% dense, fill 0.02: a genuinely sparse factor, stays on KLU.
_SPARSE_MODEL = "metapop_sir_100.net"
# 149 species, fill 0.26 but under LU_FILL_DENSE_MIN_N, where KLU and the dense
# factor tie: stays on KLU.
_SMALL_MODEL = "SHP2_base_model.net"

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

pytestmark = [
    pytest.mark.skipif(not bngsim.HAS_KLU, reason="KLU not compiled"),
    pytest.mark.skipif(
        not bngsim.HAS_LAPACK_DENSE,
        reason="no BLAS dense backend, so the fill test never moves a model",
    ),
]


@pytest.fixture(autouse=True)
def _default_solver_env(monkeypatch):
    """The fill route must not depend on the GH #84 opt-in, nor its gate knobs."""
    for var in (
        "BNGSIM_LAPACK_DENSE",
        "BNGSIM_LAPACK_DENSE_K",
        "BNGSIM_LAPACK_DENSE_MIN_N",
        "BNGSIM_NO_WARM_CVODE",
    ):
        monkeypatch.delenv(var, raising=False)


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    return str(path)


def _sim(name: str, **kwargs) -> bngsim.Simulator:
    return bngsim.Simulator(bngsim.Model.from_net(_net(name)), method="ode", **kwargs)


def _run(sim: bngsim.Simulator):
    return sim.run(t_span=(0, 100), n_points=21, rtol=1e-8, atol=1e-10)


def _assert_same_trajectory(a, b, msg: str = "") -> None:
    """KLU and a dense LU are different factorizations of the same matrix, so the
    two runs take slightly different steps; compare on the state's own scale."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7 * scale, err_msg=msg)


class TestFixturesStraddleTheRule:
    """Guard the premise, so a retuned constant fails here rather than turning a
    routing assertion below into a tautology."""

    def test_fill_model_is_klu_by_density_and_dense_by_fill(self):
        sp = bngsim.Model.from_net(_net(_FILL_MODEL))._core.jacobian_sparsity
        assert sp["n"] >= LU_FILL_DENSE_MIN_N
        assert sp["density"] < 0.10  # the size/density rule alone says KLU
        assert sp["lu_fill_estimate"] >= LU_FILL_DENSE_MIN

    def test_sparse_model_factor_stays_sparse(self):
        sp = bngsim.Model.from_net(_net(_SPARSE_MODEL))._core.jacobian_sparsity
        assert sp["n"] >= LU_FILL_DENSE_MIN_N
        assert 0.0 <= sp["lu_fill_estimate"] < LU_FILL_DENSE_MIN

    def test_small_model_fills_but_is_under_the_size_window(self):
        sp = bngsim.Model.from_net(_net(_SMALL_MODEL))._core.jacobian_sparsity
        assert sp["n"] < LU_FILL_DENSE_MIN_N
        assert sp["lu_fill_estimate"] >= LU_FILL_DENSE_MIN

    def test_estimate_is_a_fraction_of_n_squared(self):
        """nnz(L+U) counts the diagonal once, so a fill estimate can be no
        smaller than 1/n and no larger than 1."""
        sp = bngsim.Model.from_net(_net(_FILL_MODEL))._core.jacobian_sparsity
        assert 1.0 / sp["n"] <= sp["lu_fill_estimate"] <= 1.0


class TestRunRouting:
    def test_fill_heavy_model_takes_the_blas_dense_factor(self):
        r = _run(_sim(_FILL_MODEL))
        st = r.solver_stats
        assert st["linear_solver"] == LS_LAPACK
        # BLAS from the first factorization: no GH #132 count gate, whose first
        # K factorizations would be built-in. Every Jacobian evaluation is
        # followed by a factorization, so there are at least that many.
        assert st["n_jac_evals"] > 0
        assert st["n_dense_blas_factorizations"] >= st["n_jac_evals"]

    @pytest.mark.parametrize("name", [_SPARSE_MODEL, _SMALL_MODEL])
    def test_sparse_factor_and_small_model_stay_on_klu(self, name):
        assert _run(_sim(name)).solver_stats["linear_solver"] == LS_KLU

    def test_force_sparse_keeps_klu(self):
        assert (
            _run(_sim(_FILL_MODEL, force_sparse_linear_solver=True)).solver_stats["linear_solver"]
            == LS_KLU
        )

    def test_fd_jacobian_keeps_klu(self):
        """The dense route's only other Jacobian is CVODE's n-evaluation
        difference quotient, where KLU colors its differences — so the fill test
        applies only with the analytical Jacobian in use."""
        assert _run(_sim(_FILL_MODEL, jacobian="fd")).solver_stats["linear_solver"] == LS_KLU

    def test_force_dense_is_unchanged(self):
        """force_dense without the GH #84 opt-in stays on the built-in dense LU,
        as before: the fill route is the auto rule's, not the flag's."""
        st = _run(_sim(_FILL_MODEL, force_dense_linear_solver=True)).solver_stats
        assert st["linear_solver"] == LS_DENSE
        assert st["n_dense_blas_factorizations"] == 0

    def test_route_changes_the_cost_not_the_answer(self):
        auto = _run(_sim(_FILL_MODEL))
        klu = _run(_sim(_FILL_MODEL, force_sparse_linear_solver=True))
        assert auto.solver_stats["linear_solver"] == LS_LAPACK
        assert klu.solver_stats["linear_solver"] == LS_KLU
        _assert_same_trajectory(auto.species, klu.species)

    def test_warm_rerun_keeps_the_route(self):
        """The warm path (a second run on the same Simulator) builds its solver
        from the same decision."""
        sim = _sim(_FILL_MODEL)
        first = _run(sim)
        sim.model.reset()  # a run leaves the model at its final state
        second = _run(sim)
        assert second.solver_stats["linear_solver"] == LS_LAPACK
        assert (
            second.solver_stats["n_dense_blas_factorizations"]
            >= (second.solver_stats["n_jac_evals"])
        )
        np.testing.assert_array_equal(first.species, second.species)


class TestCodegenAgreesWithTheSolver:
    """The emitter builds one Jacobian layout per model and reads the route from
    ``codegen_jacobian_plan()["routes_sparse"]``, the engine's own answer — a
    compiled sparse Jacobian on a model the solver factors densely would never be
    called."""

    @pytest.mark.parametrize(("name", "sparse"), [(_FILL_MODEL, False), (_SPARSE_MODEL, True)])
    def test_plan_reports_the_route(self, name, sparse):
        sim = _sim(name)  # attaches the analytical Jacobian the route requires
        assert sim.model._core.codegen_jacobian_plan()["routes_sparse"] is sparse

    @needs_cc
    def test_fill_routed_model_compiles_a_dense_jacobian(self):
        sim = _sim(_FILL_MODEL, codegen=True)
        assert sim.codegen_backend == selected_codegen_backend()
        if sim._codegen_c_source:
            # The codegen JIT compiles this source in process (issue #1217); it
            # names each Jacobian entry point only where it defines it.
            src = sim._codegen_c_source
            assert "bngsim_codegen_jac(" in src
            assert "bngsim_codegen_jac_sparse(" not in src
        else:
            lib = ctypes.CDLL(sim._codegen_so_path)
            assert hasattr(lib, "bngsim_codegen_jac")
            assert not hasattr(lib, "bngsim_codegen_jac_sparse")
        r = _run(sim)
        assert r.solver_stats["linear_solver"] == LS_LAPACK
        _assert_same_trajectory(r.species, _run(_sim(_FILL_MODEL, codegen=False)).species)


class TestSteadyStateRouting:
    def _solve(self, **kwargs):
        sim = _sim(_FILL_MODEL, **kwargs)
        return sim.steady_state(method="integration", tol=1e-9, max_time=1e6)

    def test_march_takes_the_blas_dense_factor_and_the_same_root(self):
        auto = self._solve()
        klu = self._solve(force_sparse_linear_solver=True)
        assert auto.converged and klu.converged
        assert auto.linear_solver == "lapack-dense"
        assert klu.linear_solver == "klu"
        _assert_same_trajectory(auto.concentrations, klu.concentrations)
