"""``steady_state()``'s linear-solver route on real models (issue #128).

The routing *rule* is pinned on synthetic gate-straddling fixtures in
``test_force_sparse_linear_solver.py``, next to the ``run()`` half it now shares
a decision with. What is left here is what a ``ModelBuilder`` model cannot ask:

* the rule on published networks, on both sides of its own density ceiling;
* the compiled Jacobian, which the codegen emits in exactly ONE layout per model
  — so forcing a model onto the other route asks for a shape its artifact does
  not have, in both directions;
* the two entry points that build their own ``SteadyStateOptions``
  (``steady_state`` and ``steady_state_batch``), which is where the flags were
  missing in the first place.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"

# 149 species at 7.2% density: over SPARSE_THRESHOLD (50) and under
# SPARSE_DENSITY_MAX (10%), so the auto rule routes it to KLU.
_SPARSE_MODEL = "SHP2_base_model.net"
# 85 species at 15.7% density: over the size threshold, over the density ceiling
# — the other side of the same rule, and the reason the ceiling exists (KLU
# loses on an effectively-dense matrix).
_DENSE_MODEL = "Kocieniewski_2012.net"

pytestmark = pytest.mark.skipif(not bngsim.HAS_KLU, reason="KLU not compiled")


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    return str(path)


def _solve(name: str, *, method: str = "integration", **sim_kwargs):
    sim = bngsim.Simulator(bngsim.Model.from_net(_net(name)), method="ode", **sim_kwargs)
    return sim.steady_state(method=method, tol=1e-9, max_time=1e6)


def _assert_same_root(a, b, msg: str = "") -> None:
    """The two solves landed on the same steady state.

    Scale-relative, not per-component relative. KLU and a dense LU are different
    factorizations of the same matrix, so they take slightly different step
    sequences and stop at slightly different points on the same trajectory; a
    species sitting at 1e-12 of the state's own magnitude can differ by 100%
    relatively while the state is identical for every purpose. The tolerance is
    on the state's scale — over the 585-model corpus the worst converged model
    moves by 2.0e-8 of its own scale.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7 * scale, err_msg=msg)


class TestAutoRoutingOnPublishedNetworks:
    def test_large_sparse_model_routes_to_klu(self):
        ss = _solve(_SPARSE_MODEL)
        assert ss.converged
        assert ss.linear_solver == "klu"

    def test_model_over_the_density_ceiling_stays_dense(self):
        ss = _solve(_DENSE_MODEL)
        assert ss.converged
        assert ss.linear_solver != "klu"

    @pytest.mark.parametrize("method", ["integration", "newton"])
    def test_both_methods_route_and_agree(self, method):
        """``method="newton"`` marches before it polishes, so tier 1 routes too.

        The polish itself does not: it factors the conservation-law reduction,
        whose sparsity pattern is a different object from the model's. The root
        is the same either way.
        """
        sparse = _solve(_SPARSE_MODEL, method=method, force_sparse_linear_solver=True)
        dense = _solve(_SPARSE_MODEL, method=method, force_dense_linear_solver=True)
        assert sparse.converged and dense.converged
        assert sparse.linear_solver == "klu"
        assert dense.linear_solver != "klu"
        _assert_same_root(sparse.concentrations, dense.concentrations)


class TestCompiledJacobianInEitherLayout:
    """The codegen emits ONE Jacobian shape per model; both fills convert.

    ``bngsim_codegen_jac_sparse`` is emitted for models the routing sends to KLU
    and ``bngsim_codegen_jac`` for the rest (GH #162), so forcing a model onto
    the other route asks for a shape its artifact does not have. Both directions
    are converted rather than dropped — a forced route that silently demoted the
    compiled Jacobian to the interpreted one would be the same class of quiet
    downgrade this issue is itself a case of.
    """

    @staticmethod
    def _codegen_solve(name: str, **sim_kwargs):
        sim = bngsim.Simulator(
            bngsim.Model.from_net(_net(name)), method="ode", codegen=True, **sim_kwargs
        )
        if sim.codegen_backend == "none":
            pytest.skip("no codegen backend available in this environment")
        return sim.steady_state(tol=1e-9, max_time=1e6)

    def test_forced_sparse_still_uses_the_compiled_dense_jacobian(self):
        """A dense-emitted artifact, gathered into the CSC values."""
        ss = self._codegen_solve(_DENSE_MODEL, force_sparse_linear_solver=True)
        assert ss.converged
        assert ss.linear_solver == "klu"
        assert ss.rhs_backend.startswith("codegen")
        assert ss.solver_jacobian_source == "codegen"
        _assert_same_root(
            ss.concentrations,
            self._codegen_solve(_DENSE_MODEL, force_dense_linear_solver=True).concentrations,
        )

    def test_forced_dense_still_uses_the_compiled_sparse_jacobian(self):
        """A CSC-emitted artifact, scattered into the dense matrix."""
        ss = self._codegen_solve(_SPARSE_MODEL, force_dense_linear_solver=True)
        assert ss.converged
        assert ss.linear_solver != "klu"
        assert ss.solver_jacobian_source == "codegen"


class TestBatch:
    def test_steady_state_batch_routes_too(self):
        """The batch entry point builds its own options; it must set them too."""
        sim = bngsim.Simulator(
            bngsim.Model.from_net(_net(_DENSE_MODEL)),
            method="ode",
            force_sparse_linear_solver=True,
        )
        results = sim.steady_state_batch([{}, {}], tol=1e-9, max_time=1e6)
        assert [r.linear_solver for r in results] == ["klu", "klu"]
