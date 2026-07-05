"""Tests for ``Simulator.run(steady_state=True)`` early termination (issue #47).

BNG2.pl's ``simulate({steady_state=>1})`` runs ``run_network -c``: the
integrator stops once ``||f(t,y)||_2 / n_species`` falls below the tolerance
(its integration ``atol``) and writes only the rows up to that point. These
tests cover the bngsim equivalent on ``Simulator.run``:

- the criterion fires and the Result is truncated to the integrated rows;
- truncated rows are byte-identical to the same rows of a full run;
- a tighter tolerance keeps more rows; a looser one keeps fewer;
- ``steady_state_tol`` defaults to ``atol`` when unset;
- a window too short to equilibrate returns every row, reached=0;
- non-ODE methods reject the flag.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


@pytest.fixture
def decay_model(data_dir):
    """A -> B, k=0.1, A(0)=100. ||f||/n decays monotonically toward 0."""
    return bngsim.Model.from_net(str(data_dir / "simple_decay.net"))


@pytest.fixture
def reversible_model(data_dir):
    """A + B <-> C; settles to a nonzero equilibrium."""
    return bngsim.Model.from_net(str(data_dir / "two_species_reversible.net"))


def _full_run(model_path, **kw):
    m = bngsim.Model.from_net(str(model_path))
    return bngsim.Simulator(m, method="ode").run(**kw)


class TestEarlyStop:
    def test_criterion_fires_and_truncates(self, decay_model, data_dir):
        sim = bngsim.Simulator(decay_model, method="ode")
        r = sim.run(t_span=(0.0, 200.0), n_points=21, steady_state=True, steady_state_tol=1e-4)

        # ||f||/n crosses 1e-4 between t=110 (1.18e-4) and t=120 (4.34e-5),
        # so the run stops at t=120 — 13 of the 21 requested rows.
        assert r.solver_stats["steady_state_reached"] == 1
        assert len(r.time) < 21
        assert r.time[-1] == pytest.approx(120.0)
        # Time/species/observable arrays are all truncated consistently.
        assert r.species.shape[0] == len(r.time)
        assert r.observables.shape[0] == len(r.time)

    def test_truncated_rows_match_full_run(self, data_dir):
        net = data_dir / "simple_decay.net"
        full = _full_run(net, t_span=(0.0, 200.0), n_points=21)
        ss = _full_run(
            net, t_span=(0.0, 200.0), n_points=21, steady_state=True, steady_state_tol=1e-4
        )
        n = len(ss.time)
        assert n < len(full.time)
        # Overlap is exact: the early-stop only drops trailing rows, it does
        # not perturb the integration of the rows it keeps.
        np.testing.assert_array_equal(ss.time, full.time[:n])
        np.testing.assert_array_equal(ss.species, full.species[:n])
        np.testing.assert_array_equal(ss.observables, full.observables[:n])

    def test_tighter_tol_keeps_more_rows(self, data_dir):
        net = data_dir / "simple_decay.net"
        loose = _full_run(
            net, t_span=(0.0, 200.0), n_points=21, steady_state=True, steady_state_tol=1e-3
        )
        tight = _full_run(
            net, t_span=(0.0, 200.0), n_points=21, steady_state=True, steady_state_tol=1e-6
        )
        assert len(loose.time) < len(tight.time)
        assert loose.solver_stats["steady_state_reached"] == 1
        assert tight.solver_stats["steady_state_reached"] == 1

    def test_window_too_short_returns_all_rows(self, decay_model):
        sim = bngsim.Simulator(decay_model, method="ode")
        r = sim.run(t_span=(0.0, 5.0), n_points=11, steady_state=True, steady_state_tol=1e-8)
        # ||f||/n at t=5 is ~4.3 — nowhere near 1e-8, so no early stop.
        assert r.solver_stats["steady_state_reached"] == 0
        assert len(r.time) == 11
        assert r.time[-1] == pytest.approx(5.0)

    def test_disabled_by_default(self, decay_model):
        sim = bngsim.Simulator(decay_model, method="ode")
        r = sim.run(t_span=(0.0, 200.0), n_points=21)
        assert r.solver_stats["steady_state_reached"] == 0
        assert len(r.time) == 21

    def test_tol_defaults_to_atol(self, data_dir):
        # With steady_state_tol unset, the cutoff is atol. A loose atol stops
        # earlier than a tight one (BNG2.pl reuses its integration atol).
        net = data_dir / "simple_decay.net"
        loose_atol = _full_run(net, t_span=(0.0, 200.0), n_points=41, steady_state=True, atol=1e-4)
        tight_atol = _full_run(net, t_span=(0.0, 200.0), n_points=41, steady_state=True, atol=1e-7)
        assert loose_atol.solver_stats["steady_state_reached"] == 1
        assert len(loose_atol.time) < len(tight_atol.time)

    def test_reversible_settles_to_equilibrium(self, reversible_model):
        # Nonzero equilibrium: at steady state ||f||/n -> 0 because the net
        # flux (kf*A*B - kr*C) vanishes, even though concentrations are large.
        sim = bngsim.Simulator(reversible_model, method="ode")
        r = sim.run(t_span=(0.0, 5000.0), n_points=101, steady_state=True, steady_state_tol=1e-6)
        assert r.solver_stats["steady_state_reached"] == 1
        assert len(r.time) < 101
        # Equilibrium identity: kf*A*B == kr*C (Keq = kf/kr = 0.01).
        A, B, C = r.species[-1]
        assert pytest.approx(0.1 * C, rel=1e-4) == 0.001 * A * B


class TestBatchEarlyStop:
    """``run_batch(steady_state=True)`` applies the parity early-stop per point."""

    def test_batch_truncates_per_point(self, decay_model, data_dir):
        sim = bngsim.Simulator(decay_model, method="ode")
        # Same model under three k1 values, all of which equilibrate within
        # the t=200 window (||f||/n crosses 1e-4 at t≈112/85/60 for
        # k1=0.1/0.15/0.2); each clone truncates to its own row count.
        psets = [{"k1": 0.1}, {"k1": 0.15}, {"k1": 0.2}]
        results = sim.run_batch(
            t_span=(0.0, 200.0),
            n_points=21,
            params=psets,
            steady_state=True,
            steady_state_tol=1e-4,
        )
        assert len(results) == 3
        for r in results:
            assert r.solver_stats["steady_state_reached"] == 1
            assert len(r.time) <= 21
            assert r.species.shape[0] == len(r.time)
        # Faster decay (larger k1) reaches the cutoff in fewer rows.
        assert len(results[2].time) <= len(results[0].time)

    def test_batch_matches_single_run(self, decay_model, data_dir):
        sim = bngsim.Simulator(decay_model, method="ode")
        batch = sim.run_batch(
            t_span=(0.0, 200.0),
            n_points=21,
            params=[{"k1": 0.1}],
            steady_state=True,
            steady_state_tol=1e-4,
        )
        single = sim.run(
            t_span=(0.0, 200.0), n_points=21, steady_state=True, steady_state_tol=1e-4
        )
        np.testing.assert_array_equal(batch[0].time, single.time)
        np.testing.assert_array_equal(batch[0].species, single.species)

    def test_batch_squeeze_rejected_with_steady_state(self, decay_model):
        sim = bngsim.Simulator(decay_model, method="ode")
        with pytest.raises(ValueError, match="squeeze"):
            sim.run_batch(
                t_span=(0.0, 200.0),
                n_points=21,
                params=[{"k1": 0.1}],
                steady_state=True,
                squeeze=True,
            )


class TestValidation:
    def test_steady_state_rejected_for_ssa(self, decay_model):
        sim = bngsim.Simulator(decay_model, method="ssa")
        with pytest.raises(ValueError, match="only supported for method='ode'"):
            sim.run(t_span=(0.0, 10.0), n_points=11, steady_state=True)

    def test_negative_tol_rejected(self, decay_model):
        sim = bngsim.Simulator(decay_model, method="ode")
        with pytest.raises(ValueError, match="steady_state_tol must be non-negative"):
            sim.run(t_span=(0.0, 10.0), n_points=11, steady_state=True, steady_state_tol=-1.0)

    def test_batch_steady_state_rejected_for_ssa(self, decay_model):
        sim = bngsim.Simulator(decay_model, method="ssa")
        with pytest.raises(ValueError, match="only supported for method='ode'"):
            sim.run_batch(t_span=(0.0, 10.0), n_points=11, params=[{"k1": 0.1}], steady_state=True)
