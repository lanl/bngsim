"""Runtime guard: the dense-solver-for-lack-of-KLU warning (GH #209).

When bngsim is built without SuiteSparse/KLU the ODE backend can only use the
dense linear solver, which factorizes the full N×N Jacobian at O(N³). For a
large/sparse model that is silently catastrophic (minutes → hours). The
:class:`bngsim.DenseSolverFallbackWarning` surfaces it once at ``run()`` — but
only when the dense solver is forced by the *missing build*, not by a deliberate
``force_dense_linear_solver`` / ``jacobian="jax"`` choice, and only for models
large enough for sparsity to matter.

These tests drive the branch by patching the module-level KLU flag and the size
threshold so a tiny test model trips the guard; ``monkeypatch`` restores both,
and the process-wide one-shot flag, after each test.
"""

from __future__ import annotations

import warnings

import bngsim
import bngsim._simulator as _simulator
import pytest


@pytest.fixture
def force_dense_fallback(monkeypatch):
    """Pretend KLU is absent, drop the size threshold, reset the one-shot flag."""
    monkeypatch.setattr(_simulator, "_HAS_KLU", False)
    monkeypatch.setattr(_simulator, "_DENSE_FALLBACK_WARN_NSPECIES", 1)
    monkeypatch.setattr(_simulator, "_dense_fallback_warned", False)


def _run(model, **kwargs):
    bngsim.Simulator(model, method="ode", **kwargs).run(t_span=(0, 1), n_points=3)


def test_warns_once_when_klu_missing(simple_decay_net, force_dense_fallback):
    m = bngsim.Model.from_net(str(simple_decay_net))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(m)
        _run(m)  # second run must NOT re-warn (one-shot per process)
    dense = [w for w in caught if w.category is bngsim.DenseSolverFallbackWarning]
    assert len(dense) == 1
    msg = str(dense[0].message)
    # Names the cause and a concrete fix.
    assert "KLU" in msg
    assert "O(N³)" in msg or "O(N^3)" in msg
    assert "capabilities()" in msg


def test_silent_for_user_forced_dense(simple_decay_net, force_dense_fallback):
    m = bngsim.Model.from_net(str(simple_decay_net))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(m, force_dense_linear_solver=True)
    dense = [w for w in caught if w.category is bngsim.DenseSolverFallbackWarning]
    assert dense == []


def test_silent_below_size_threshold(simple_decay_net, monkeypatch):
    # KLU missing, but the model is below the warn threshold → no notice.
    monkeypatch.setattr(_simulator, "_HAS_KLU", False)
    monkeypatch.setattr(_simulator, "_DENSE_FALLBACK_WARN_NSPECIES", 10_000)
    monkeypatch.setattr(_simulator, "_dense_fallback_warned", False)
    m = bngsim.Model.from_net(str(simple_decay_net))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(m)
    dense = [w for w in caught if w.category is bngsim.DenseSolverFallbackWarning]
    assert dense == []


def test_silent_when_klu_available(simple_decay_net, monkeypatch):
    # A KLU-enabled install never reaches the warning, even for a large model.
    monkeypatch.setattr(_simulator, "_HAS_KLU", True)
    monkeypatch.setattr(_simulator, "_DENSE_FALLBACK_WARN_NSPECIES", 1)
    monkeypatch.setattr(_simulator, "_dense_fallback_warned", False)
    m = bngsim.Model.from_net(str(simple_decay_net))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(m)
    dense = [w for w in caught if w.category is bngsim.DenseSolverFallbackWarning]
    assert dense == []


def test_non_ode_method_does_not_warn(simple_decay_net, force_dense_fallback):
    # The guard is ODE-only; SSA never selects a linear solver.
    m = bngsim.Model.from_net(str(simple_decay_net))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bngsim.Simulator(m, method="ssa").run(t_span=(0, 1), n_points=3, seed=1)
    dense = [w for w in caught if w.category is bngsim.DenseSolverFallbackWarning]
    assert dense == []
