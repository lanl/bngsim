"""Tests for sample_times support in Simulator.run() and BNGL action parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Simulator

# ═══════════════════════════════════════════════════════════════════════════════
# Simulator.run() with sample_times — ODE
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampleTimesODE:
    """ODE simulations with explicit sample_times."""

    def test_exact_output_times(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(sample_times=[0, 1, 5, 10])
        times = np.asarray(result.time)
        np.testing.assert_allclose(times, [0, 1, 5, 10], atol=1e-12)
        assert result.n_times == 4

    def test_non_uniform_spacing(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        sample = [0.0, 0.1, 0.5, 2.0, 10.0, 100.0]
        result = sim.run(sample_times=sample)
        np.testing.assert_allclose(result.time, sample, atol=1e-12)

    def test_unsorted_auto_sorts(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(sample_times=[10, 0, 5, 1])
        np.testing.assert_allclose(result.time, [0, 1, 5, 10], atol=1e-12)

    def test_observables_shape(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(sample_times=[0, 2, 4, 6, 8, 10])
        obs = np.asarray(result.observables)
        assert obs.shape == (6, result.n_observables)

    def test_values_match_uniform(self, simple_decay_net: Path):
        """sample_times at uniform points should give same results as n_points."""
        model1 = Model.from_net(simple_decay_net)
        sim1 = Simulator(model1, method="ode")
        r_uniform = sim1.run(t_span=(0, 10), n_points=6)

        model2 = Model.from_net(simple_decay_net)
        sim2 = Simulator(model2, method="ode")
        r_sample = sim2.run(sample_times=[0, 2, 4, 6, 8, 10])

        np.testing.assert_allclose(
            np.asarray(r_sample.observables),
            np.asarray(r_uniform.observables),
            rtol=1e-6,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator.run() with sample_times — SSA
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampleTimesSSA:
    """SSA simulations with explicit sample_times."""

    def test_exact_output_times(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")
        result = sim.run(sample_times=[0, 1, 5, 10])
        np.testing.assert_allclose(result.time, [0, 1, 5, 10], atol=1e-12)
        assert result.n_times == 4

    def test_observables_shape(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")
        result = sim.run(sample_times=[0, 3, 7, 10])
        obs = np.asarray(result.observables)
        assert obs.shape == (4, result.n_observables)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator.run() with sample_times — network-free backends (GH #169)
#
# Before GH #169 the NFsim (nf_reject) and RuleMonkey (nf_exact) network-free
# backends ignored an explicit sample_times list and always emitted a uniform
# t_start..t_end grid sized by len(sample_times): requesting [0, 2, 5, 9]
# returned output at [0, 3, 6, 9]. These tests pin the fix — both backends now
# record observable/function output at exactly the requested instants — and
# guard against a regression to the uniform-grid behavior. Both sample in-engine
# in a single run() pass: NFsim steps a single live System; RuleMonkey honors
# TimeSpec::sample_times in the vendored engine's run_ssa recorder (upstream
# RuleMonkey #16). See TestSampleTimesNetworkFreeNonInvasive below for the
# non-invasive (bit-identical) property that follows.
# ═══════════════════════════════════════════════════════════════════════════════


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# One row per network-free backend, each skipped independently if that engine
# was not compiled in. "nfsim" == nf_reject (NFsim); "rm" == nf_exact
# (RuleMonkey). Both route through the same Simulator.run(times, seed) path.
_NF_BACKENDS = [
    pytest.param(
        "nfsim",
        marks=pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in"),
    ),
    pytest.param(
        "rm",
        marks=pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in"),
    ),
]


@pytest.mark.parametrize("method", _NF_BACKENDS)
class TestSampleTimesNetworkFree:
    """NFsim / RuleMonkey simulations with explicit sample_times (GH #169)."""

    def _sim(self, simple_decay_net: Path, nfsim_xml: Path, method: str) -> Simulator:
        # The network-free backends read their model from the BNG XML; the
        # Model arg is required by the Simulator API but otherwise unused, so a
        # trivial .net stands in for it (mirrors the test_nfsim dummy_model).
        model = Model.from_net(str(simple_decay_net))
        return Simulator(model, method=method, xml_path=str(nfsim_xml))

    def test_exact_output_times(self, simple_decay_net, nfsim_xml, method):
        """Output lands at exactly the requested instants, not a uniform grid."""
        sim = self._sim(simple_decay_net, nfsim_xml, method)
        result = sim.run(sample_times=[0, 2, 5, 9], seed=42)
        times = np.asarray(result.time)
        np.testing.assert_allclose(times, [0, 2, 5, 9], atol=1e-12)
        assert result.n_times == 4
        # Regression guard: the pre-fix uniform grid for 4 points over [0, 9]
        # would have been [0, 3, 6, 9]. Make the wrong answer fail loudly.
        assert not np.allclose(times, [0, 3, 6, 9])

    def test_non_uniform_spacing(self, simple_decay_net, nfsim_xml, method):
        sim = self._sim(simple_decay_net, nfsim_xml, method)
        sample = [0.0, 0.05, 0.1, 0.5, 2.0, 9.9]
        result = sim.run(sample_times=sample, seed=42)
        np.testing.assert_allclose(result.time, sample, atol=1e-12)
        assert result.n_times == len(sample)

    def test_unsorted_auto_sorts(self, simple_decay_net, nfsim_xml, method):
        sim = self._sim(simple_decay_net, nfsim_xml, method)
        result = sim.run(sample_times=[9, 0, 5, 2], seed=42)
        np.testing.assert_allclose(result.time, [0, 2, 5, 9], atol=1e-12)

    def test_observables_shape(self, simple_decay_net, nfsim_xml, method):
        sim = self._sim(simple_decay_net, nfsim_xml, method)
        result = sim.run(sample_times=[0, 3, 7, 10], seed=42)
        obs = np.asarray(result.observables)
        assert obs.shape == (4, result.n_observables)

    def test_two_points_works(self, simple_decay_net, nfsim_xml, method):
        sim = self._sim(simple_decay_net, nfsim_xml, method)
        result = sim.run(sample_times=[0, 10], seed=42)
        np.testing.assert_allclose(result.time, [0, 10], atol=1e-12)
        assert result.n_times == 2

    def test_reproducible_same_seed(self, simple_decay_net, nfsim_xml, method):
        """A fixed seed yields a deterministic trajectory at the requested times."""
        sample = [0, 1.3, 2.7, 4.1, 6.6, 9.0]
        r1 = self._sim(simple_decay_net, nfsim_xml, method).run(sample_times=sample, seed=11)
        r2 = self._sim(simple_decay_net, nfsim_xml, method).run(sample_times=sample, seed=11)
        np.testing.assert_array_equal(np.asarray(r1.observables), np.asarray(r2.observables))

    def test_single_segment_matches_uniform(self, simple_decay_net, nfsim_xml, method):
        """sample_times=[t_start, t_end] reproduces the n_points=2 uniform run.

        Both backends record explicit times non-invasively, so a fixed seed
        yields a bit-identical trajectory to the equivalent uniform run. See
        TestSampleTimesNetworkFreeNonInvasive for the general multi-point
        property.
        """
        r_uniform = self._sim(simple_decay_net, nfsim_xml, method).run(
            t_span=(0, 10), n_points=2, seed=5
        )
        r_sample = self._sim(simple_decay_net, nfsim_xml, method).run(sample_times=[0, 10], seed=5)
        np.testing.assert_array_equal(
            np.asarray(r_sample.observables), np.asarray(r_uniform.observables)
        )


# Both network-free backends record explicit times non-invasively in a single
# run() pass: NFsim steps one live System (src/nfsim_simulator.cpp), and
# RuleMonkey honors TimeSpec::sample_times inside the vendored engine's run_ssa
# recorder (upstream RuleMonkey #16). For a fixed seed the recorded trajectory
# is therefore independent of *which* instants are requested — a run with
# explicit sample_times is bit-identical to the uniform-grid run at any instants
# the two schedules share. (Before RuleMonkey #16 the bngsim wrapper drove the
# session API per segment, which re-entered the SSA loop and perturbed reaction
# selection across segment boundaries; that realization was reproducible and
# unbiased but not bit-identical. The in-engine path retired the workaround and
# regained bit-identical sampling plus event_count/n_steps reporting.)
@pytest.mark.parametrize("method", _NF_BACKENDS)
class TestSampleTimesNetworkFreeNonInvasive:
    """Network-free sampling does not perturb the trajectory (GH #169)."""

    def _sim(self, simple_decay_net: Path, nfsim_xml: Path, method: str) -> Simulator:
        model = Model.from_net(str(simple_decay_net))
        return Simulator(model, method=method, xml_path=str(nfsim_xml))

    def test_matches_uniform_grid_same_seed(self, simple_decay_net, nfsim_xml, method):
        """sample_times at the uniform instants reproduce the n_points path."""
        r_uniform = self._sim(simple_decay_net, nfsim_xml, method).run(
            t_span=(0, 10), n_points=11, seed=7
        )
        r_sample = self._sim(simple_decay_net, nfsim_xml, method).run(
            sample_times=list(range(11)), seed=7
        )
        np.testing.assert_allclose(r_sample.time, r_uniform.time, atol=1e-12)
        np.testing.assert_array_equal(
            np.asarray(r_sample.observables), np.asarray(r_uniform.observables)
        )

    def test_subset_matches_uniform_at_those_instants(self, simple_decay_net, nfsim_xml, method):
        """A subset of the uniform instants returns the true state there."""
        r_uniform = self._sim(simple_decay_net, nfsim_xml, method).run(
            t_span=(0, 10), n_points=11, seed=7
        )
        uniform_times = list(np.asarray(r_uniform.time))

        subset = [0, 3, 7, 10]
        r_sub = self._sim(simple_decay_net, nfsim_xml, method).run(sample_times=subset, seed=7)

        ref_rows = [uniform_times.index(t) for t in subset]
        np.testing.assert_array_equal(
            np.asarray(r_sub.observables),
            np.asarray(r_uniform.observables)[ref_rows],
        )

    def test_event_count_matches_uniform(self, simple_decay_net, nfsim_xml, method):
        """Explicit sample_times report the same SSA event count as the uniform
        run: non-invasive sampling never perturbs the realization, so the event
        counter is identical for a fixed seed. Regression guard for the
        RuleMonkey in-engine path (GH #169) — the retired session-stepping
        workaround could not report n_steps at all (it stayed at 0).
        """
        r_uniform = self._sim(simple_decay_net, nfsim_xml, method).run(
            t_span=(0, 10), n_points=11, seed=7
        )
        r_sample = self._sim(simple_decay_net, nfsim_xml, method).run(
            sample_times=list(range(11)), seed=7
        )
        n_sample = r_sample.solver_stats["n_steps"]
        assert n_sample > 0
        assert n_sample == r_uniform.solver_stats["n_steps"]


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampleTimesValidation:
    """Validation and edge cases for sample_times."""

    def test_two_points_works(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(sample_times=[0, 10])
        assert result.n_times == 2

    def test_single_point_raises(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        with pytest.raises(ValueError, match="at least 2 points"):
            sim.run(sample_times=[5])

    def test_empty_list_raises(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        with pytest.raises(ValueError, match="at least 2 points"):
            sim.run(sample_times=[])

    def test_none_falls_back_to_uniform(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11, sample_times=None)
        assert result.n_times == 11

    def test_three_points_minimum(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(sample_times=[0, 5, 10])
        assert result.n_times == 3


# ═══════════════════════════════════════════════════════════════════════════════
# PyBNF's _resolve_sample_times / _parse_simulate_action — tested in PyBNF
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file used to unit-test those two private PyBNF helpers by importing them
# across the repo boundary. That coverage now lives with the code it covers:
#
#   PyBNF tests/test_bngsim_bridge.py::TestResolveSampleTimes
#   PyBNF tests/test_bngsim_parsing.py (action-dict + value parsing)
#
# Removed rather than fixed in place (lanl/bngsim#45). PyBNF already had its own
# TestResolveSampleTimes — same class name, broader coverage — so ours was a fork
# that drifted: when PyBNF deliberately lowered the sample_times minimum from 3
# points to 2, their copy was updated and ours kept asserting the old rule. It sat
# red for months because the class skipped wherever roadrunner was absent, which
# is every CI leg we have, so only a dev box with the full PyBNF stack ever ran it
# — and there it looked like a pre-existing failure to wave off.
#
# What belongs here instead is our own half of the contract, which needs no PyBNF:
# TestSampleTimesValidation above pins `resolve_sample_times` (>= 2 points, sorted,
# finite), and TestSampleTimesODE/SSA/NetworkFree pin that a request is honored
# exactly. If PyBNF changes how it *derives* sample_times, that is their test's
# job; if we change what we *accept*, it is ours.
