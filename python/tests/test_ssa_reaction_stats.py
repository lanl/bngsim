"""Per-reaction firing counts and propensity integrals from the exact SSA (GH #616).

``Simulator(model, method="ssa", reaction_stats=True)`` records, at every output
time, each reaction's cumulative firing count ``N_r(t)`` and integrated
propensity ``∫₀ᵗ a_r ds``. The contract under test:

- Off by default, and refused for methods that do not run an exact SSA.
- Recording changes no trajectory, on either SSA loop: the compiled
  recompute-all fast loop and the interpreted incremental one
  (``BNGSIM_SSA_NO_CODEGEN`` selects the latter).
- The blocks are internally consistent: zero at ``t_start``, monotone, integer
  counts, ``A(t) = A(0) + N_birth(t) − N_death(t)`` exactly, and the total count
  is the run's step count.
- ``N_r(t) − ∫₀ᵗ a_r ds`` has zero mean across replicates, against its exact
  variance ``E[∫₀ᵗ a_r ds]``: the compensator identity, which is the
  instrumentation's correctness test — it fails the moment an integral is
  banked against the wrong intensity.
- The likelihood-ratio gradient built from them (``bngsim.girsanov``) agrees
  with CVODES forward sensitivities on a model whose mean obeys the ODE exactly.
- The blocks survive ``run_replicates`` (sequential and parallel, list and
  stacked), ``save``/``load`` and ``to_xarray``.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import girsanov

# Birth-death: 0 -> A at k1, A -> 0 at k2. Linear, so E[A(t)] obeys
# dA/dt = k1 - k2*A exactly and CVODES sensitivities are the exact gradient.
BIRTH_DEATH_NET = """\
begin parameters
    1 k1  5.0
    2 k2  0.2
end parameters
begin species
    1 A() 10
end species
begin reactions
    1 0 1 k1 #_R1
    2 1 0 k2 #_R2
end reactions
begin groups
    1 A_tot  1
end groups
"""

T_SPAN = (0.0, 20.0)
N_POINTS = 21
N_REP = 400
LN10 = math.log(10.0)
DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"


@pytest.fixture
def net_path(tmp_path: Path) -> Path:
    p = tmp_path / "birth_death.net"
    p.write_text(BIRTH_DEATH_NET)
    return p


def _model(net_path: Path) -> bngsim.Model:
    return bngsim.Model.from_net(str(net_path))


@pytest.fixture(params=["default", "interpreted"])
def loop(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Each test runs on the default (compiled recompute-all) loop and on the
    interpreted incremental loop, which is the one the general machinery
    (dependency graph, events, rate rules) runs through."""
    if request.param == "interpreted":
        monkeypatch.setenv("BNGSIM_SSA_NO_CODEGEN", "1")
    else:
        monkeypatch.delenv("BNGSIM_SSA_NO_CODEGEN", raising=False)
    return request.param


def _ensemble(net_path: Path, n: int, seed: int) -> tuple[bngsim.Model, bngsim.Result]:
    m = _model(net_path)
    sim = bngsim.Simulator(m, method="ssa", reaction_stats=True)
    batch = sim.run_replicates(n, t_span=T_SPAN, n_points=N_POINTS, seed=seed, squeeze=True)
    return m, batch


def test_off_by_default(net_path: Path) -> None:
    r = bngsim.Simulator(_model(net_path), method="ssa").run(
        t_span=T_SPAN, n_points=N_POINTS, seed=3
    )
    assert not r.has_reaction_stats
    assert r.reaction_firing_counts.shape == (0, 0)
    assert r.reaction_propensity_integrals.shape == (0, 0)
    assert r.reaction_labels == []


@pytest.mark.parametrize(("method", "kwargs"), [("ode", {}), ("psa", {"poplevel": 10})])
def test_refused_off_the_exact_ssa(net_path: Path, method: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="reaction_stats"):
        bngsim.Simulator(_model(net_path), method=method, reaction_stats=True, **kwargs)


def test_trajectory_unchanged_and_blocks_consistent(net_path: Path, loop: str) -> None:
    plain = bngsim.Simulator(_model(net_path), method="ssa").run(
        t_span=T_SPAN, n_points=N_POINTS, seed=11
    )
    r = bngsim.Simulator(_model(net_path), method="ssa", reaction_stats=True).run(
        t_span=T_SPAN, n_points=N_POINTS, seed=11
    )
    if loop == "interpreted":
        assert r.ssa_diagnostics["propensity_backend"] == "interpreted"
    np.testing.assert_array_equal(np.asarray(r.species), np.asarray(plain.species))

    counts, integrals = r.reaction_firing_counts, r.reaction_propensity_integrals
    assert counts.shape == integrals.shape == (N_POINTS, 2)
    assert r.reaction_labels == ["R1 (0 -> A())", "R2 (A() -> 0)"]
    assert counts[0].tolist() == [0.0, 0.0]
    assert integrals[0].tolist() == [0.0, 0.0]
    assert np.array_equal(counts, np.round(counts))
    assert (np.diff(counts, axis=0) >= 0).all()
    assert (np.diff(integrals, axis=0) >= 0).all()
    assert (integrals[1:] > 0).all()
    a = np.asarray(r.species)[:, 0]
    np.testing.assert_array_equal(a, a[0] + counts[:, 0] - counts[:, 1])
    assert counts[-1].sum() == r.solver_stats["n_steps"]


def test_compensator_has_zero_mean(net_path: Path, loop: str) -> None:
    _, batch = _ensemble(net_path, N_REP, seed=7)
    counts, integrals = batch.reaction_firing_counts, batch.reaction_propensity_integrals
    assert counts.shape == integrals.shape == (N_REP, N_POINTS, 2)
    d = counts - integrals
    mean = d.mean(axis=0)
    # Var(N - ∫a) = E[∫a] exactly (the compensated process's quadratic variation);
    # the sample variance is not used because it is wrong for a rare channel.
    se = np.sqrt(integrals.mean(axis=0) / N_REP)
    assert (np.abs(mean[1:]) <= 4.0 * se[1:]).all(), (mean, se)


def test_gradient_matches_cvodes(net_path: Path, loop: str) -> None:
    m, batch = _ensemble(net_path, N_REP, seed=19)
    ode = bngsim.Simulator(_model(net_path), method="ode", sensitivity_params=["k1", "k2"])
    truth = np.asarray(ode.run(t_span=T_SPAN, n_points=N_POINTS).sensitivities)[:, 0, :]

    w = girsanov.scores(m, batch, ["k1", "k2"])
    assert w.shape == (N_REP, N_POINTS, 2)
    grad, se = girsanov.gradient(np.asarray(batch.species)[:, :, 0], w)
    assert grad.shape == se.shape == (N_POINTS, 2)
    assert (np.abs(grad[1:] - truth[1:]) <= 4.0 * se[1:] + 1e-9).all(), (grad, truth, se)
    # The signal is resolved, not merely bracketed: the error bar is well
    # inside the gradient at the end of the time course.
    assert (se[-1] < 0.5 * np.abs(truth[-1])).all()

    # The observables path gives the same numbers with an output axis.
    grad3, se3 = girsanov.gradient(np.asarray(batch.observables), w)
    assert grad3.shape == (N_POINTS, 1, 2)
    np.testing.assert_allclose(grad3[:, 0, :], grad)
    np.testing.assert_allclose(se3[:, 0, :], se)

    # log10 scores carry the chain rule ln(10)·k through exactly.
    w10 = girsanov.scores(m, batch, ["k1", "k2"], log10=True)
    np.testing.assert_allclose(w10, w * (LN10 * np.array([5.0, 0.2])), rtol=1e-12, atol=1e-12)


DERIVED_NET = """\
begin parameters
    1 k1__FREE  2.5
    2 k1        2*k1__FREE
    3 k2        0.2
    4 unused    1.0
end parameters
begin species
    1 A() 10
end species
begin reactions
    1 0 1 k1 #_R1
    2 1 0 k2 #_R2
end reactions
begin groups
    1 A_tot  1
end groups
"""


def test_scores_through_derived_rate_constants(tmp_path: Path) -> None:
    """A fitting tool's ``k = 2*k__FREE`` alias: the score with respect to the
    free parameter is the chain rule through the model's parameter expression,
    and it agrees with CVODES differentiating through the same expression."""
    net = tmp_path / "derived.net"
    net.write_text(DERIVED_NET)
    m, batch = _ensemble(net, N_REP, seed=23)
    assert girsanov.rate_constant_reactions(m) == {"k1": [0], "k2": [1]}

    w_free = girsanov.scores(m, batch, ["k1__FREE"])
    w_k1 = girsanov.scores(m, batch, ["k1"])
    np.testing.assert_allclose(w_free, 2.0 * w_k1, rtol=1e-8)
    # In log space the two are the same score: d/dlog(k1__FREE) = d/dlog(k1).
    np.testing.assert_allclose(
        girsanov.scores(m, batch, ["k1__FREE"], log10=True),
        girsanov.scores(m, batch, ["k1"], log10=True),
        rtol=1e-8,
    )
    assert m.get_param("k1__FREE") == 2.5  # perturbed for the derivative, restored
    assert m.get_param("k1") == 5.0

    ode = bngsim.Simulator(_model(net), method="ode", sensitivity_params=["k1__FREE"])
    truth = np.asarray(ode.run(t_span=T_SPAN, n_points=N_POINTS).sensitivities)[:, 0, 0]
    grad, se = girsanov.gradient(np.asarray(batch.species)[:, :, 0], w_free)
    assert (np.abs(grad[1:, 0] - truth[1:]) <= 4.0 * se[1:, 0] + 1e-9).all()

    with pytest.raises(ValueError, match="sets no elementary reaction"):
        girsanov.scores(m, batch, ["unused"])


def test_scores_refusals(net_path: Path) -> None:
    m, batch = _ensemble(net_path, 4, seed=1)
    with pytest.raises(ValueError, match="not a parameter of the model"):
        girsanov.scores(m, batch, ["nope"])
    plain = bngsim.Simulator(_model(net_path), method="ssa").run(
        t_span=T_SPAN, n_points=N_POINTS, seed=1
    )
    with pytest.raises(ValueError, match="reaction_stats=True"):
        girsanov.scores(m, plain, ["k1"])
    mm = bngsim.Model.from_net(str(DATA_DIR / "mm_tqssa.net"))
    with pytest.raises(ValueError, match="mass-action"):
        girsanov.scores(mm, batch, ["k1"])
    assert girsanov.rate_constant_reactions(m) == {"k1": [0], "k2": [1]}
    with pytest.raises(ValueError, match="at least two"):
        girsanov.gradient(np.zeros((1, N_POINTS)), np.zeros((1, N_POINTS, 1)))


def test_replicates_list_parallel_and_roundtrip(net_path: Path, tmp_path: Path) -> None:
    sim = bngsim.Simulator(_model(net_path), method="ssa", reaction_stats=True)
    reps = sim.run_replicates(3, t_span=T_SPAN, n_points=N_POINTS, seed=5)
    assert all(r.reaction_firing_counts.shape == (N_POINTS, 2) for r in reps)
    # The parallel path builds its own simulator per worker; the flag and the
    # propensity library follow it, so each worker runs the loop the sequential
    # path runs and the answer is the same bit for bit. Before the library
    # followed, the workers took the incremental loop, whose dwell increments
    # sum in another order.
    par = sim.run_replicates(3, t_span=T_SPAN, n_points=N_POINTS, seed=5, num_processors=2)
    for a, b in zip(reps, par, strict=True):
        np.testing.assert_array_equal(a.reaction_firing_counts, b.reaction_firing_counts)
        np.testing.assert_array_equal(
            a.reaction_propensity_integrals, b.reaction_propensity_integrals
        )

    pytest.importorskip("h5py")
    path = tmp_path / "r.h5"
    reps[0].save(path)
    back = bngsim.Result.load(path)
    np.testing.assert_array_equal(back.reaction_firing_counts, reps[0].reaction_firing_counts)
    np.testing.assert_array_equal(
        back.reaction_propensity_integrals, reps[0].reaction_propensity_integrals
    )
    assert back.reaction_labels == reps[0].reaction_labels
    plain = bngsim.Simulator(_model(net_path), method="ssa").run(
        t_span=T_SPAN, n_points=N_POINTS, seed=5
    )
    plain.save(path)
    back = bngsim.Result.load(path)
    assert not back.has_reaction_stats
    assert back.reaction_labels == []


def test_batch_rows_carry_the_blocks(net_path: Path) -> None:
    """run_batch built its per-row engine without the flag, so every row came
    back with no blocks. A row at the model's own values is the run at the
    row's seed, blocks included."""
    sim = bngsim.Simulator(_model(net_path), method="ssa", reaction_stats=True)
    rows = sim.run_batch(T_SPAN, N_POINTS, params=[{"k1": 5.0}, {"k1": 2.0}], seed=5)
    # The row's own seed, derived from the base (issue #1365).
    one = bngsim.Simulator(_model(net_path), method="ssa", reaction_stats=True).run(
        t_span=T_SPAN, n_points=N_POINTS, seed=rows[0].seed
    )
    assert [r.has_reaction_stats for r in rows] == [True, True]
    assert rows[0].reaction_labels == one.reaction_labels
    np.testing.assert_array_equal(rows[0].reaction_firing_counts, one.reaction_firing_counts)
    np.testing.assert_array_equal(
        rows[0].reaction_propensity_integrals, one.reaction_propensity_integrals
    )


def test_squeezed_batch_reports_its_backend(net_path: Path) -> None:
    """A stacked batch used to report the raw-array default's backend
    ("interpreted") whatever its replicates ran; it now carries the aggregate."""
    sim = bngsim.Simulator(_model(net_path), method="ssa", reaction_stats=True)
    one = sim.run(t_span=T_SPAN, n_points=N_POINTS, seed=5)
    batch = sim.run_replicates(3, t_span=T_SPAN, n_points=N_POINTS, seed=5, squeeze=True)
    assert batch.ssa_diagnostics["propensity_backend"] == one.ssa_diagnostics["propensity_backend"]
    assert batch.ssa_diagnostics["n_negative_crossings"] == 0
    assert batch.ssa_diagnostics["n_reverse_fires"] == 0
    loaded = bngsim.Result(
        core=None, _time=np.zeros(1), _species=np.zeros((1, 1)), _observables=np.zeros((1, 0))
    )
    assert loaded.ssa_diagnostics["propensity_backend"] == "unknown"


def test_to_xarray_carries_the_blocks(net_path: Path) -> None:
    pytest.importorskip("xarray")
    r = bngsim.Simulator(_model(net_path), method="ssa", reaction_stats=True).run(
        t_span=T_SPAN, n_points=N_POINTS, seed=2
    )
    ds = r.to_xarray()
    assert ds["reaction_firing_counts"].dims == ("time", "reaction")
    assert ds["reaction_propensity_integrals"].dims == ("time", "reaction")
    assert list(ds.coords["reaction"].values) == r.reaction_labels
    plain = bngsim.Simulator(_model(net_path), method="ssa").run(
        t_span=T_SPAN, n_points=N_POINTS, seed=2
    )
    assert "reaction_firing_counts" not in plain.to_xarray()
