"""Tests for ``Simulator.run_replicates`` (fix 1b).

``run_replicates`` runs N stochastic replicates of the *same* model by reusing
one simulator and ``reset()``-ing state between replicates, instead of cloning +
reconstructing a simulator per replicate (the old ensemble pattern). The
contract under test:

- It is **answer-invariant**: replicate *i* (seed ``base+i``) is byte-identical
  to the clone-per-replicate reference run with the same seed, because reusing
  the simulator only reuses topology (the cached dependency graph) and reset
  restores initial state. Verified on a mass-action model and a functional /
  time-dependent model (the latter exercises the per-step ExprTk path).
- The parallel path (clone once per worker thread) matches the sequential path.
- Seeds are ``base_seed + i`` and exposed via ``Result.seed``.
- It is stochastic-only and validates its arguments.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

T_SPAN = (0.0, 50.0)
N_POINTS = 51
N_REP = 10


def _clone_per_rep_reference(net_path: Path, n: int, seed_base: int) -> list[np.ndarray]:
    """The pre-fix ensemble pattern: clone + new Simulator per replicate."""
    base = bngsim.Model.from_net(str(net_path))
    out: list[np.ndarray] = []
    for i in range(n):
        m = base.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ssa")
        r = sim.run(t_span=T_SPAN, n_points=N_POINTS, seed=seed_base + i)
        out.append(np.asarray(r.species))
    return out


@pytest.fixture(params=["simple_decay_net", "time_func_net"])
def stochastic_net(request: pytest.FixtureRequest) -> Path:
    """A mass-action model and a functional/time-dependent one (per-step ExprTk)."""
    return request.getfixturevalue(request.param)


def test_run_replicates_matches_clone_per_rep(stochastic_net: Path) -> None:
    ref = _clone_per_rep_reference(stochastic_net, N_REP, seed_base=0)

    sim = bngsim.Simulator(bngsim.Model.from_net(str(stochastic_net)), method="ssa")
    reps = sim.run_replicates(N_REP, t_span=T_SPAN, n_points=N_POINTS, seed=0)

    assert len(reps) == N_REP
    for i, r in enumerate(reps):
        np.testing.assert_array_equal(np.asarray(r.species), ref[i])


def test_run_replicates_parallel_matches_sequential(stochastic_net: Path) -> None:
    ref = _clone_per_rep_reference(stochastic_net, N_REP, seed_base=3)

    sim = bngsim.Simulator(bngsim.Model.from_net(str(stochastic_net)), method="ssa")
    reps = sim.run_replicates(N_REP, t_span=T_SPAN, n_points=N_POINTS, seed=3, num_processors=4)

    for i, r in enumerate(reps):
        np.testing.assert_array_equal(np.asarray(r.species), ref[i])


def test_run_replicates_exposes_per_replicate_seeds(simple_decay_net: Path) -> None:
    sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ssa")
    reps = sim.run_replicates(N_REP, t_span=T_SPAN, n_points=N_POINTS, seed=100)
    assert [r.seed for r in reps] == list(range(100, 100 + N_REP))


def test_run_replicates_squeeze_shape(simple_decay_net: Path) -> None:
    model = bngsim.Model.from_net(str(simple_decay_net))
    sim = bngsim.Simulator(model, method="ssa")
    squeezed = sim.run_replicates(N_REP, t_span=T_SPAN, n_points=N_POINTS, seed=0, squeeze=True)
    assert squeezed.species.shape == (N_REP, N_POINTS, model.n_species)


def test_run_replicates_rejects_ode(simple_decay_net: Path) -> None:
    sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ode")
    with pytest.raises(ValueError, match="stochastic"):
        sim.run_replicates(4, t_span=T_SPAN, n_points=N_POINTS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_replicates": 0},
        {"n_replicates": 4, "t_span": (5.0, 5.0)},
        {"n_replicates": 4, "n_points": 1},
        {"n_replicates": 4, "timeout": -1.0},
    ],
)
def test_run_replicates_validates_arguments(simple_decay_net: Path, kwargs: dict) -> None:
    sim = bngsim.Simulator(bngsim.Model.from_net(str(simple_decay_net)), method="ssa")
    call = {"t_span": T_SPAN, "n_points": N_POINTS, **kwargs}
    with pytest.raises(ValueError):
        sim.run_replicates(**call)
