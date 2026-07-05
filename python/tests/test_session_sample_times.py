"""Explicit session output times (GH #184), unified across NF backends.

The stateful network-free sessions (:class:`bngsim.NfsimSession`,
:class:`bngsim.RuleMonkeySession`) accept ``simulate(sample_times=[...])`` so a
host such as PyBNF can record observables at a dataset's exact time points
instead of a uniform grid. The contract must be *identical* across NFsim and
RuleMonkey so the two backends stay interchangeable — these tests are
parametrized over both to enforce that.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


def _has_nfsim() -> bool:
    return bool(getattr(bngsim, "HAS_NFSIM", False))


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# Each backend is skipped independently if its support was not compiled in, so
# the shared contract still runs for whichever backend is available.
BACKENDS = [
    pytest.param(
        "NfsimSession",
        marks=pytest.mark.skipif(not _has_nfsim(), reason="no NFsim support"),
        id="nfsim",
    ),
    pytest.param(
        "RuleMonkeySession",
        marks=pytest.mark.skipif(not _has_rulemonkey(), reason="no RuleMonkey support"),
        id="rulemonkey",
    ),
]


@pytest.fixture
def session_cls(request) -> type:
    return getattr(bngsim, request.param)


def _xml(nfsim_xml: Path) -> str:
    # simple_system.xml is a valid BNG XML for both NF backends.
    return str(nfsim_xml)


@pytest.mark.parametrize("session_cls", BACKENDS, indirect=True)
class TestSessionSampleTimes:
    SAMPLES = [0.0, 1.0, 2.5, 5.0, 10.0]

    def test_time_column_equals_sample_times_exactly(self, session_cls, nfsim_xml):
        """The returned time axis is exactly sample_times, not a uniform grid."""
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=7)
            r = s.simulate(sample_times=self.SAMPLES)
        assert np.array_equal(r.time, np.asarray(self.SAMPLES))
        assert r.n_times == len(self.SAMPLES)

    def test_unsorted_input_is_sorted(self, session_cls, nfsim_xml):
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=7)
            r = s.simulate(sample_times=[10.0, 0.0, 2.5, 1.0, 5.0])
        assert np.array_equal(r.time, np.asarray(self.SAMPLES))

    def test_bit_identical_where_schedules_coincide(self, session_cls, nfsim_xml):
        """Sampling at explicit times must not perturb the SSA trajectory:
        observable values at times shared with the uniform grid are identical
        under the same seed. This is the core correctness guarantee."""
        coincident = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        rows = [0, 2, 4, 6, 8, 10]  # those times in a 0..10 / 11-point grid

        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=99)
            uniform = s.simulate(0.0, 10.0, n_points=11)
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=99)
            explicit = s.simulate(sample_times=coincident)

        assert np.array_equal(
            np.asarray(uniform.observables)[rows, :], np.asarray(explicit.observables)
        )
        if uniform.n_expressions:
            assert np.array_equal(
                np.asarray(uniform.expressions)[rows, :], np.asarray(explicit.expressions)
            )

    def test_works_mid_protocol(self, session_cls, nfsim_xml):
        """Explicit times are honored after a prior segment / mutation, and the
        labels remain the requested absolute times (continuing trajectory)."""
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=7)
            first = s.simulate(0.0, 3.0, n_points=4)
            assert first.time[-1] == pytest.approx(3.0)
            second = s.simulate(sample_times=[3.0, 4.0, 7.0, 10.0])
        assert np.array_equal(second.time, np.asarray([3.0, 4.0, 7.0, 10.0]))

    def test_mid_protocol_continues_same_trajectory(self, session_cls, nfsim_xml):
        """A two-segment run and a single run sampled at the same instants
        trace the same continuing trajectory (same seed, no re-seed)."""
        # One continuous run sampled at 0,2,4,6.
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=123)
            whole = s.simulate(sample_times=[0.0, 2.0, 4.0, 6.0])
        # Split: segment to t=2, then continue sampling at 2,4,6.
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=123)
            s.simulate(sample_times=[0.0, 2.0])
            tail = s.simulate(sample_times=[2.0, 4.0, 6.0])
        # Rows for t=2,4,6 in the whole run are indices 1,2,3; in tail 0,1,2.
        assert np.array_equal(
            np.asarray(whole.observables)[[1, 2, 3], :], np.asarray(tail.observables)
        )

    def test_rejects_fewer_than_two_points(self, session_cls, nfsim_xml):
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=1)
            with pytest.raises(ValueError, match="at least 2 points"):
                s.simulate(sample_times=[5.0])

    def test_rejects_non_finite(self, session_cls, nfsim_xml):
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=1)
            with pytest.raises(ValueError, match="finite"):
                s.simulate(sample_times=[0.0, float("inf")])

    def test_uniform_path_still_requires_endpoints(self, session_cls, nfsim_xml):
        with session_cls(_xml(nfsim_xml)) as s:
            s.initialize(seed=1)
            with pytest.raises(ValueError, match="required"):
                s.simulate(0.0, 10.0)  # missing n_points, no sample_times


@pytest.mark.skipif(not _has_nfsim(), reason="no NFsim support")
def test_nfsim_relative_time_and_sample_times_mutually_exclusive(nfsim_xml):
    with bngsim.NfsimSession(str(nfsim_xml)) as s:
        s.initialize(seed=1)
        with pytest.raises(ValueError, match="mutually exclusive"):
            s.simulate(sample_times=[0.0, 1.0], relative_time=True)
