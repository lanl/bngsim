"""GH #29 — ``force_sparse_linear_solver``, the counterpart to ``force_dense``.

The ODE linear-solver kind is auto-selected: sparse KLU when the model is both
large (``n_species >= 50``) and sparse (Jacobian density ``< 0.10``), dense
otherwise. ``force_dense_linear_solver`` (GH #102) can push that decision toward
dense; this flag is the missing other direction, so the rule can be measured
against its own alternative on the models it sends to dense.

These tests pin the two halves of that: the flag reaches KLU on models the auto
rule would not (below the size threshold, and above the density ceiling), and it
does so without changing the trajectory — a forced run that integrates to a
different answer would be measuring something other than the same problem.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import HAS_KLU, ModelBuilder, SolverOptions

# LinearSolverKind codes — mirror include/bngsim/result.hpp.
LS_DENSE, LS_KLU, LS_LAPACK = 0, 1, 2

requires_klu = pytest.mark.skipif(not HAS_KLU, reason="build has no SuiteSparse/KLU")


def _chain(n: int):
    """A → B → C → ... first-order chain: N=n, density ≈ 2/n.

    n=10 is below SPARSE_THRESHOLD (auto → dense); n=60 clears both auto gates
    (density ≈ 0.033 < 0.10, auto → KLU) and serves as the control.
    """
    b = ModelBuilder()
    sp = []
    for i in range(n):
        b.add_parameter(f"k{i}", 0.1 * (i + 1))
        sp.append(b.add_species(f"S{i}", 100.0 if i == 0 else 0.0))
    for i in range(n - 1):
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    return b.build()


def _mesh(n: int, span: int):
    """Bimolecular web ``S_i + S_{i+d} -> S_{i+2d}`` for d in 1..span.

    Large *and* dense — the other half of the corpus the auto rule sends to the
    dense solver. span=4 at n=60 gives density ≈ 0.18 (≥ the 0.10 ceiling, so
    auto → dense) while staying under 0.5. span=12 crosses 0.5, the density that
    used to be left uncolored and so unreachable under the flag without an
    analytical Jacobian.
    """
    b = ModelBuilder()
    b.add_parameter("k", 1e-4)
    sp = [b.add_species(f"S{i}", 10.0 if i % 3 == 0 else 1.0) for i in range(n)]
    for i in range(n):
        for d in range(1, span + 1):
            j, k = (i + d) % n, (i + 2 * d) % n
            if len({i, j, k}) == 3:
                b.add_reaction([sp[i], sp[j]], [sp[k]], "elementary", "k")
    return b.build()


def _run(core, **kwargs):
    sim = bngsim.Simulator(bngsim.Model(_core=core), method="ode", **kwargs)
    return sim.run(t_span=(0, 20), n_points=21, rtol=1e-10, atol=1e-12)


def _density(core) -> float:
    return float(core.codegen_jacobian_plan()["density"])


class TestFixturesStraddleTheAutoGates:
    """Guard the premise: these models must sit where the tests assume.

    Without this, a shift in SPARSE_THRESHOLD / SPARSE_DENSITY_MAX would turn
    the tests below into tautologies (asserting KLU on a model the auto rule
    already sends to KLU) instead of failures.
    """

    def test_small_chain_is_below_the_size_threshold(self):
        core = _chain(10)
        assert core.n_species < 50

    def test_mesh_is_large_but_over_the_density_ceiling(self):
        core = _mesh(60, 4)
        assert core.n_species >= 50
        assert 0.10 <= _density(core) < 0.5

    def test_long_chain_clears_both_auto_gates(self):
        core = _chain(60)
        assert core.n_species >= 50
        assert _density(core) < 0.10


@requires_klu
class TestForcedSparseReachesKLU:
    def test_small_model_auto_is_dense(self):
        """Baseline: without the flag, a sub-threshold model is dense-only."""
        assert _run(_chain(10)).solver_stats["linear_solver"] != LS_KLU

    def test_small_model_forced_is_klu(self):
        r = _run(_chain(10), force_sparse_linear_solver=True)
        assert r.solver_stats["linear_solver"] == LS_KLU

    def test_small_model_forced_matches_dense_trajectory(self):
        sparse = _run(_chain(10), force_sparse_linear_solver=True)
        dense = _run(_chain(10), force_dense_linear_solver=True)
        assert sparse.solver_stats["linear_solver"] == LS_KLU
        assert dense.solver_stats["linear_solver"] != LS_KLU
        np.testing.assert_allclose(sparse.species, dense.species, rtol=1e-8, atol=1e-10)

    def test_large_dense_model_auto_is_dense(self):
        """Baseline: past the size threshold, density alone still selects dense."""
        assert _run(_mesh(60, 4)).solver_stats["linear_solver"] != LS_KLU

    def test_large_dense_model_forced_is_klu(self):
        r = _run(_mesh(60, 4), force_sparse_linear_solver=True)
        assert r.solver_stats["linear_solver"] == LS_KLU

    def test_large_dense_model_forced_matches_dense_trajectory(self):
        sparse = _run(_mesh(60, 4), force_sparse_linear_solver=True)
        dense = _run(_mesh(60, 4), force_dense_linear_solver=True)
        assert sparse.solver_stats["linear_solver"] == LS_KLU
        np.testing.assert_allclose(sparse.species, dense.species, rtol=1e-8, atol=1e-10)

    def test_fd_jacobian_forced_sparse_uses_the_coloring(self):
        """``jacobian="fd"`` takes the colored-FD sparse callback, not analytical."""
        r = _run(_mesh(60, 4), force_sparse_linear_solver=True, jacobian="fd")
        assert r.solver_stats["linear_solver"] == LS_KLU

    def test_near_dense_fd_model_forced_is_klu(self):
        """Density ≥ 0.5 with no analytical Jacobian still reaches KLU.

        This is the case that used to refuse: coloring was computed at build
        time only below density 0.5, so a near-dense pattern with the analytical
        path suppressed had nothing to fill the CSC matrix KLU factorizes.
        Coloring is now materialized on demand at any density (GH #29) — for a
        fully dense pattern it degenerates to one column per color, i.e. plain
        FD, which is the honest cost to measure here rather than a reason to
        refuse the run.
        """
        core = _mesh(60, 12)
        assert _density(core) >= 0.5
        r = _run(core, force_sparse_linear_solver=True, jacobian="fd")
        assert r.solver_stats["linear_solver"] == LS_KLU

    def test_near_dense_fd_model_forced_matches_dense_trajectory(self):
        """Degenerate coloring must still produce the same Jacobian, hence answer."""
        sparse = _run(_mesh(60, 12), force_sparse_linear_solver=True, jacobian="fd")
        dense = _run(_mesh(60, 12), force_dense_linear_solver=True, jacobian="fd")
        assert sparse.solver_stats["linear_solver"] == LS_KLU
        assert dense.solver_stats["linear_solver"] != LS_KLU
        np.testing.assert_allclose(sparse.species, dense.species, rtol=1e-8, atol=1e-10)

    def test_already_sparse_model_is_unchanged(self):
        """On a model the auto rule already sends to KLU, the flag is inert."""
        auto = _run(_chain(60))
        forced = _run(_chain(60), force_sparse_linear_solver=True)
        assert auto.solver_stats["linear_solver"] == LS_KLU
        assert forced.solver_stats["linear_solver"] == LS_KLU
        np.testing.assert_allclose(auto.species, forced.species, rtol=1e-8, atol=1e-10)


class TestHardRequirementsStillHold:
    """The flag bypasses the size/density gates — and nothing else."""

    def test_default_is_off(self):
        assert SolverOptions().force_sparse_linear_solver is False
        assert bngsim.Simulator(bngsim.Model(_core=_chain(4)))._force_sparse_linear_solver is False

    def test_both_force_flags_raise(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            bngsim.Simulator(
                bngsim.Model(_core=_chain(10)),
                method="ode",
                force_dense_linear_solver=True,
                force_sparse_linear_solver=True,
            )

    def test_both_force_flags_raise_at_the_core_too(self):
        """The C++ gate refuses the pair independently of the Python kwarg check.

        SolverOptions is reachable without going through Simulator; letting one
        flag quietly win there would hand a caller auto-selected numbers under a
        "forced" label.
        """
        from bngsim._bngsim_core import CvodeSimulator, TimeSpec

        opts = SolverOptions()
        opts.force_dense_linear_solver = True
        opts.force_sparse_linear_solver = True
        times = TimeSpec()
        times.t_start, times.t_end, times.n_points = 0.0, 20.0, 21
        with pytest.raises(ValueError, match="mutually exclusive"):
            CvodeSimulator(_chain(10)).run(times, opts)

    @pytest.mark.skipif(HAS_KLU, reason="requires a build without SuiteSparse/KLU")
    def test_no_op_without_klu(self):
        """Documented as a no-op in a KLU-less build, exactly like force_dense."""
        r = _run(_chain(10), force_sparse_linear_solver=True)
        assert r.solver_stats["linear_solver"] in (LS_DENSE, LS_LAPACK)


@requires_klu
class TestWarmCacheHonorsTheFlip:
    """The warm path reuses CVODE memory across ``run()`` calls; a solver-kind
    change has to invalidate that, or the second run silently reports one kind
    while integrating with the other."""

    def test_flipping_the_flag_rebuilds_the_linear_solver(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_NO_WARM_CVODE", raising=False)
        sim = bngsim.Simulator(bngsim.Model(_core=_chain(10)), method="ode")
        ic = sim.get_state()

        first = sim.run(t_span=(0, 20), n_points=21, rtol=1e-10, atol=1e-12)
        sim.set_state(ic)
        sim._force_sparse_linear_solver = True
        second = sim.run(t_span=(0, 20), n_points=21, rtol=1e-10, atol=1e-12)
        sim.set_state(ic)
        sim._force_sparse_linear_solver = False
        third = sim.run(t_span=(0, 20), n_points=21, rtol=1e-10, atol=1e-12)

        assert first.solver_stats["linear_solver"] != LS_KLU
        assert second.solver_stats["linear_solver"] == LS_KLU
        assert third.solver_stats["linear_solver"] != LS_KLU
        # Same initial condition each time, so the rebuilt solver must reproduce
        # the trajectory — a stale reused solver would not.
        np.testing.assert_allclose(second.species, first.species, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(third.species, first.species, rtol=1e-8, atol=1e-10)
