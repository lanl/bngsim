"""GH #84 — optimized BLAS dense linear solver: integration parity + gate.

The C++ test (tests/test_lapack_dense_linsol.cpp) pins the kernel (dgetrf factor
+ built-in denseGETRS solve) to the built-in dense LU at the Ax=b level. These
tests cover the *integration* path: a full CVODE run with the BLAS factor must
produce the same trajectory as the built-in factor, and the density-aware gate
must report the backend it actually used.
"""

from __future__ import annotations

import numpy as np
import pytest
from bngsim import Model, Simulator
from bngsim import _bngsim_core as _core

HAS_LAPACK = bool(getattr(_core, "HAS_LAPACK_DENSE", False))

# LinearSolverKind codes — mirror include/bngsim/result.hpp.
LS_DENSE, LS_KLU, LS_LAPACK = 0, 1, 2

requires_lapack = pytest.mark.skipif(
    not HAS_LAPACK, reason="build links no BLAS dense backend (Accelerate / LAPACK)"
)


def _run(net, env_value, monkeypatch):
    """Run an ODE sim with BNGSIM_LAPACK_DENSE set to env_value (None = unset).

    A fresh Simulator per call gives a fresh warm cache, so the env override is
    re-read at solver setup rather than reusing a previously-built solver.
    force_dense_linear_solver keeps the dense path even for tiny models (no KLU
    detour), so the backend choice is built-in-dense vs BLAS-dense.
    """
    if env_value is None:
        monkeypatch.delenv("BNGSIM_LAPACK_DENSE", raising=False)
    else:
        monkeypatch.setenv("BNGSIM_LAPACK_DENSE", env_value)
    model = Model.from_net(str(net))
    sim = Simulator(model, method="ode", force_dense_linear_solver=True)
    return sim.run(t_span=(0, 50), n_points=51, rtol=1e-9, atol=1e-12)


@requires_lapack
def test_force_on_off_trajectory_parity(simple_decay_net, monkeypatch):
    """Forcing the BLAS factor must not perturb the trajectory vs the built-in."""
    on = _run(simple_decay_net, "force", monkeypatch)
    off = _run(simple_decay_net, "off", monkeypatch)
    assert on.solver_stats["linear_solver"] == LS_LAPACK
    assert off.solver_stats["linear_solver"] == LS_DENSE
    np.testing.assert_allclose(on.species, off.species, rtol=1e-9, atol=1e-12)


@requires_lapack
def test_reversible_trajectory_parity(reversible_net, monkeypatch):
    """Same parity on a reversible (stiffer) two-species system."""
    on = _run(reversible_net, "force", monkeypatch)
    off = _run(reversible_net, "off", monkeypatch)
    np.testing.assert_allclose(on.species, off.species, rtol=1e-9, atol=1e-12)


@requires_lapack
def test_warm_cache_rebuilds_when_env_gate_changes(simple_decay_net, monkeypatch):
    """Changing the env gate on a reused Simulator must rebuild the dense solver."""
    monkeypatch.delenv("BNGSIM_NO_WARM_CVODE", raising=False)
    monkeypatch.setenv("BNGSIM_LAPACK_DENSE", "off")
    model = Model.from_net(str(simple_decay_net))
    sim = Simulator(model, method="ode", force_dense_linear_solver=True)

    off = sim.run(t_span=(0, 50), n_points=51, rtol=1e-9, atol=1e-12)
    monkeypatch.setenv("BNGSIM_LAPACK_DENSE", "1")
    on = sim.run(t_span=(0, 50), n_points=51, rtol=1e-9, atol=1e-12)

    assert off.solver_stats["linear_solver"] == LS_DENSE
    assert on.solver_stats["linear_solver"] == LS_LAPACK


@requires_lapack
def test_adaptive_gate_dgetrf_in_cvode_parity(reversible_net, monkeypatch):
    """Drive the BLAS dgetrf factor through a real CVODE integration (GH #132).

    With the default adaptive gate (K=5, MIN_N=256) a tiny model never reaches
    the dgetrf path, so force it: K=0 switches on the first factorization and
    MIN_N=0 removes the small-N guard. The resulting trajectory must match the
    built-in factor, confirming the in-integrator dgetrf path is correct (the
    C++ test covers only the bare Ax=b kernel).
    """
    monkeypatch.setenv("BNGSIM_LAPACK_DENSE_K", "0")
    monkeypatch.setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0")
    on = _run(reversible_net, "force", monkeypatch)
    off = _run(reversible_net, "off", monkeypatch)
    assert on.solver_stats["linear_solver"] == LS_LAPACK
    np.testing.assert_allclose(on.species, off.species, rtol=1e-9, atol=1e-12)


@requires_lapack
def test_adaptive_gate_K_does_not_change_trajectory(reversible_net, monkeypatch):
    """Switching the dgetrf threshold K must never change the answer (GH #132).

    Built-in GETRF and dgetrf are bit-equivalent (modulo pivot base), so where
    the adaptive gate flips backends is a pure performance knob: K=0 (all dgetrf)
    and a K past the run's factorization count (all built-in) must agree.
    """
    monkeypatch.setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0")
    monkeypatch.setenv("BNGSIM_LAPACK_DENSE_K", "0")
    all_blas = _run(reversible_net, "force", monkeypatch)
    monkeypatch.setenv("BNGSIM_LAPACK_DENSE_K", "100000")
    all_builtin = _run(reversible_net, "force", monkeypatch)
    np.testing.assert_allclose(all_blas.species, all_builtin.species, rtol=1e-12, atol=0.0)


def test_small_model_default_stays_builtin(simple_decay_net, monkeypatch):
    """The density-aware gate never engages the BLAS factor on a tiny model."""
    monkeypatch.delenv("BNGSIM_LAPACK_DENSE", raising=False)
    model = Model.from_net(str(simple_decay_net))
    sim = Simulator(model, method="ode")
    r = sim.run(t_span=(0, 50), n_points=51)
    assert r.solver_stats["linear_solver"] != LS_LAPACK


def test_linear_solver_stat_present(simple_decay_net):
    """solver_stats always carries the linear_solver key (benchmark observability)."""
    model = Model.from_net(str(simple_decay_net))
    sim = Simulator(model, method="ode")
    r = sim.run(t_span=(0, 50), n_points=51)
    assert "linear_solver" in r.solver_stats
    assert r.solver_stats["linear_solver"] in (LS_DENSE, LS_KLU, LS_LAPACK)


def test_has_lapack_dense_attr_is_bool():
    assert isinstance(HAS_LAPACK, bool)
