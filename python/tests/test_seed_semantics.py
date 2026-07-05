"""Tests for the RNG seed contract (#10).

The contract:

- Reproducibility unit = **same starting model state + same `seed=N`**.
  The C++ SSA/PSA backends construct a fresh `std::mt19937_64(seed)`
  on every `.run()` call, so passing the same seed always seeds the
  RNG identically. What persists across `.run()` calls on the same
  ``Simulator`` is the *model state* — that's what makes multi-segment
  SSA protocols (``simulate(...); simulate({continue=>1, ...})``) work.
- ``seed=None`` (or omitted) draws a fresh seed from system entropy on
  each call, so two fresh ``Simulator`` instances default to independent
  trajectories.
- ``Result.seed`` exposes the integer that was passed down to the
  backend (``None`` for ODE).
- ``Simulator.run_batch`` resolves a single ``base_seed`` per call;
  per-sim seeds are ``base_seed + i`` and stamped onto each ``Result``.
- Sessions: ``initialize(seed=...)`` follows the same contract; the
  session's ``seed`` property and every ``Result.seed`` from
  ``simulate()`` carry the resolved integer.
- HDF5 round-trips ``Result.seed``.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


@pytest.fixture
def model(simple_decay_net: Path) -> bngsim.Model:
    return bngsim.Model.from_net(str(simple_decay_net))


# ─── Resolver ─────────────────────────────────────────────────────


class TestResolveSeed:
    def test_explicit_int_passes_through(self):
        from bngsim._seed import _resolve_seed

        assert _resolve_seed(42) == 42
        assert _resolve_seed(0) == 0
        assert _resolve_seed(123456789) == 123456789

    def test_none_draws_fresh_each_call(self):
        from bngsim._seed import _resolve_seed

        seeds = {_resolve_seed(None) for _ in range(20)}
        assert len(seeds) == 20

    def test_resolved_seed_fits_in_int32(self):
        from bngsim._seed import _resolve_seed

        for _ in range(50):
            s = _resolve_seed(None)
            assert 0 <= s < 2**31


# ─── Simulator.run() default ──────────────────────────────────────


class TestSimulatorRunSeedDefault:
    """Default seed=None: fresh Simulator instances must default to
    independent stochastic trajectories."""

    def test_fresh_sims_default_to_independent_trajectories(self, model):
        s1 = bngsim.Simulator(model.clone(), method="ssa")
        s2 = bngsim.Simulator(model.clone(), method="ssa")
        r1 = s1.run(t_span=(0, 10), n_points=11)
        r2 = s2.run(t_span=(0, 10), n_points=11)
        assert r1.seed != r2.seed
        assert not np.array_equal(np.asarray(r1.species), np.asarray(r2.species))

    def test_explicit_seed_on_fresh_sims_reproduces(self, model):
        s1 = bngsim.Simulator(model.clone(), method="ssa")
        s2 = bngsim.Simulator(model.clone(), method="ssa")
        r1 = s1.run(t_span=(0, 10), n_points=11, seed=42)
        r2 = s2.run(t_span=(0, 10), n_points=11, seed=42)
        assert r1.seed == 42
        assert r2.seed == 42
        np.testing.assert_array_equal(np.asarray(r1.species), np.asarray(r2.species))

    def test_resolved_seed_fits_int32(self, model):
        sim = bngsim.Simulator(model, method="ssa")
        r = sim.run(t_span=(0, 10), n_points=11)
        assert isinstance(r.seed, int)
        assert 0 <= r.seed < 2**31


class TestReproducibilityUnit:
    """Reproducibility = same starting model state + same seed.

    The C++ SSA/PSA backends construct a fresh mt19937_64 on every run,
    so the seed re-seeds identically each call; trajectory differences
    across consecutive ``.run(seed=N)`` calls on the same Simulator
    come from the model state continuing — not from any RNG quirk.
    Calling ``model.reset()`` between runs returns to initial state and
    reproduces.
    """

    def test_reset_then_reseed_reproduces_on_same_simulator(self, model):
        sim = bngsim.Simulator(model, method="ssa")
        r1 = sim.run(t_span=(0, 10), n_points=11, seed=42)
        model.reset()
        r2 = sim.run(t_span=(0, 10), n_points=11, seed=42)
        np.testing.assert_array_equal(np.asarray(r1.species), np.asarray(r2.species))

    def test_no_reset_means_state_continues(self, model):
        sim = bngsim.Simulator(model, method="ssa")
        r1 = sim.run(t_span=(0, 10), n_points=11, seed=42)
        r2 = sim.run(t_span=(0, 10), n_points=11, seed=42)
        # r2 starts from r1's end state; that's what makes the
        # trajectories differ — not an RNG state continuation.
        np.testing.assert_array_equal(np.asarray(r2.species)[0], np.asarray(r1.species)[-1])

    def test_fresh_simulator_matches_reset_simulator(self, model):
        # A fresh Simulator with seed=42 produces the same trajectory
        # as model.reset() + sim.run(seed=42) on a used Simulator —
        # because both start from the model's initial state.
        sim_used = bngsim.Simulator(model.clone(), method="ssa")
        sim_used.run(t_span=(0, 10), n_points=11, seed=999)  # warm it up
        sim_used._model.reset()
        r_reset = sim_used.run(t_span=(0, 10), n_points=11, seed=42)

        sim_fresh = bngsim.Simulator(model.clone(), method="ssa")
        r_fresh = sim_fresh.run(t_span=(0, 10), n_points=11, seed=42)

        np.testing.assert_array_equal(np.asarray(r_reset.species), np.asarray(r_fresh.species))


# ─── Result.seed exposure ────────────────────────────────────────


class TestResultSeed:
    def test_explicit_seed_round_trip(self, model):
        sim = bngsim.Simulator(model, method="ssa")
        r = sim.run(t_span=(0, 10), n_points=11, seed=12345)
        assert r.seed == 12345

    def test_ode_seed_is_none(self, model):
        sim = bngsim.Simulator(model, method="ode")
        r = sim.run(t_span=(0, 10), n_points=11)
        assert r.seed is None

    def test_psa_seed_exposed(self, model):
        sim = bngsim.Simulator(model, method="psa", poplevel=10)
        r = sim.run(t_span=(0, 10), n_points=11, seed=999)
        assert r.seed == 999

    def test_psa_default_seed_is_drawn(self, model):
        s1 = bngsim.Simulator(model.clone(), method="psa", poplevel=10)
        s2 = bngsim.Simulator(model.clone(), method="psa", poplevel=10)
        r1 = s1.run(t_span=(0, 10), n_points=11)
        r2 = s2.run(t_span=(0, 10), n_points=11)
        assert r1.seed is not None
        assert r2.seed is not None
        assert r1.seed != r2.seed


# ─── run_batch ───────────────────────────────────────────────────


class TestRunBatchSeed:
    def test_per_sim_seeds_are_consecutive(self, model):
        sim = bngsim.Simulator(model, method="ssa")
        psets = [{"k1": v} for v in [0.1, 0.2, 0.3, 0.4]]
        results = sim.run_batch(t_span=(0, 10), n_points=11, params=psets, seed=1000)
        assert [r.seed for r in results] == [1000, 1001, 1002, 1003]

    def test_default_seed_draws_fresh_base_per_call(self, model):
        sim = bngsim.Simulator(model, method="ssa")
        psets = [{"k1": 0.1}, {"k1": 0.2}]
        rs1 = sim.run_batch(t_span=(0, 10), n_points=11, params=psets)
        rs2 = sim.run_batch(t_span=(0, 10), n_points=11, params=psets)
        # Different base seeds → different per-sim seeds
        assert rs1[0].seed != rs2[0].seed
        # Per-sim seeds within each batch are consecutive
        assert rs1[1].seed == rs1[0].seed + 1
        assert rs2[1].seed == rs2[0].seed + 1

    def test_explicit_base_reproduces_batch(self, model):
        sim1 = bngsim.Simulator(model.clone(), method="ssa")
        sim2 = bngsim.Simulator(model.clone(), method="ssa")
        psets = [{"k1": v} for v in [0.5, 0.5]]
        r1 = sim1.run_batch(t_span=(0, 10), n_points=11, params=psets, seed=777)
        r2 = sim2.run_batch(t_span=(0, 10), n_points=11, params=psets, seed=777)
        for a, b in zip(r1, r2, strict=False):
            assert a.seed == b.seed
            np.testing.assert_array_equal(np.asarray(a.species), np.asarray(b.species))

    def test_squeeze_preserves_uniform_seed_when_present(self, model):
        # Under default seed=None each per-sim seed is unique, so the
        # squeezed Result.seed is None.
        sim = bngsim.Simulator(model, method="ssa")
        psets = [{"k1": 0.1}, {"k1": 0.2}]
        squeezed = sim.run_batch(t_span=(0, 10), n_points=11, params=psets, squeeze=True)
        assert squeezed.seed is None


# ─── Sessions ─────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
class TestNfsimSessionSeed:
    def test_initialize_default_draws_fresh(self, nfsim_xml):
        from bngsim import NfsimSession

        with NfsimSession(str(nfsim_xml)) as nf1, NfsimSession(str(nfsim_xml)) as nf2:
            nf1.initialize()
            nf2.initialize()
            assert nf1.seed is not None
            assert nf2.seed is not None
            assert nf1.seed != nf2.seed

    def test_initialize_explicit_int(self, nfsim_xml):
        from bngsim import NfsimSession

        with NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=2024)
            assert nf.seed == 2024

    def test_simulate_result_carries_session_seed(self, nfsim_xml):
        from bngsim import NfsimSession

        with NfsimSession(str(nfsim_xml)) as nf:
            nf.initialize(seed=99)
            r = nf.simulate(0, 1, n_points=11)
            assert r.seed == 99

    def test_seed_none_before_initialize(self, nfsim_xml):
        from bngsim import NfsimSession

        with NfsimSession(str(nfsim_xml)) as nf:
            assert nf.seed is None


@pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in")
class TestRuleMonkeySessionSeed:
    def test_initialize_default_draws_fresh(self, nfsim_xml):
        from bngsim import RuleMonkeySession

        with RuleMonkeySession(str(nfsim_xml)) as rm1, RuleMonkeySession(str(nfsim_xml)) as rm2:
            rm1.initialize()
            rm2.initialize()
            assert rm1.seed is not None
            assert rm2.seed is not None
            assert rm1.seed != rm2.seed

    def test_initialize_explicit_int(self, nfsim_xml):
        from bngsim import RuleMonkeySession

        with RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=2024)
            assert rm.seed == 2024

    def test_simulate_result_carries_session_seed(self, nfsim_xml):
        from bngsim import RuleMonkeySession

        with RuleMonkeySession(str(nfsim_xml)) as rm:
            rm.initialize(seed=99)
            r = rm.simulate(0, 1, n_points=11)
            assert r.seed == 99


# ─── HDF5 round-trip ─────────────────────────────────────────────


class TestHdf5RoundTrip:
    def test_seed_persists_through_save_load(self, model, tmp_path):
        pytest.importorskip("h5py")
        sim = bngsim.Simulator(model, method="ssa")
        r = sim.run(t_span=(0, 10), n_points=11, seed=314159)
        path = tmp_path / "result.h5"
        r.save(path)
        loaded = bngsim.Result.load(path)
        assert loaded.seed == 314159

    def test_ode_no_seed_attr(self, model, tmp_path):
        pytest.importorskip("h5py")
        sim = bngsim.Simulator(model, method="ode")
        r = sim.run(t_span=(0, 10), n_points=11)
        path = tmp_path / "ode.h5"
        r.save(path)
        loaded = bngsim.Result.load(path)
        assert loaded.seed is None


# ─── Backwards compatibility ─────────────────────────────────────


class TestExplicitSeedStillWorks:
    """The contract for the *explicit* seed=N path is unchanged: it
    passes through to the backend verbatim."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345, 2**30])
    def test_passthrough(self, model, seed):
        sim = bngsim.Simulator(model, method="ssa")
        r = sim.run(t_span=(0, 10), n_points=11, seed=seed)
        assert r.seed == seed
