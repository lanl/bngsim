"""Tests for NFsim t_start contract — Decision D2 (Session 37).

BNGsim contract: run(TimeSpec{t_start, t_end, n_points}) returns
the CURRENT system state labeled as t_start at index 0, then
advances the system via stepTo() for indices 1..n_points-1.

See: bngsim/dev/adr/ADR-002-function-semantics.md §2 for rationale.
"""

import numpy as np
import pytest


@pytest.fixture
def nfsim_xml():
    """Path to a simple NFsim test XML."""
    from pathlib import Path

    xml = Path(__file__).parent.parent.parent / "tests" / "data" / "nfsim" / "simple_system.xml"
    if xml.exists():
        return str(xml)
    pytest.skip("simple_system.xml not available")


def _has_nfsim():
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return HAS_NFSIM
    except ImportError:
        return False


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
class TestNfsimTstart:
    """Test NFsim run() with various t_start values."""

    def test_tstart_zero_basic(self, nfsim_xml):
        """t_start=0: first sample at t=0."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(nfsim_xml)
        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 11
        result = sim.run(times, 42)

        assert result.n_times == 11
        t = np.array(result.time)
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(1.0)

    def test_tstart_nonzero_labels(self, nfsim_xml):
        """t_start=5: time axis starts at 5, ends at 10."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(nfsim_xml)
        times = TimeSpec()
        times.t_start = 5.0
        times.t_end = 10.0
        times.n_points = 6
        result = sim.run(times, 42)

        assert result.n_times == 6
        t = np.array(result.time)
        assert t[0] == pytest.approx(5.0)
        assert t[-1] == pytest.approx(10.0)

    def test_tstart_nonzero_first_sample_is_initial(self, nfsim_xml):
        """First sample with t_start>0 = initial state.

        For a fresh system, the first sample is always the
        unadvanced initial condition regardless of t_start.
        """
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        # Run with t_start=0
        sim0 = NfsimSimulator(nfsim_xml)
        ts0 = TimeSpec()
        ts0.t_start = 0.0
        ts0.t_end = 1.0
        ts0.n_points = 2
        r0 = sim0.run(ts0, 42)
        obs0 = np.array(r0.observable_data)
        ic0 = obs0[0, :]  # first time point

        # Run with t_start=5
        sim5 = NfsimSimulator(nfsim_xml)
        ts5 = TimeSpec()
        ts5.t_start = 5.0
        ts5.t_end = 6.0
        ts5.n_points = 2
        r5 = sim5.run(ts5, 42)
        obs5 = np.array(r5.observable_data)
        ic5 = obs5[0, :]  # first time point

        # Both should have same initial observable values
        np.testing.assert_array_equal(
            ic0, ic5, err_msg="Initial observables differ between t_start=0 and t_start=5"
        )

    def test_tstart_nonzero_simulation_progresses(self, nfsim_xml):
        """Simulation advances after the first sample."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(nfsim_xml)
        times = TimeSpec()
        times.t_start = 5.0
        times.t_end = 15.0
        times.n_points = 11
        result = sim.run(times, 42)

        assert result.solver_stats.n_steps > 0

    def test_session_tstart_uses_internal_clock(self, nfsim_xml):
        """Session API: simulate() handles time offsets."""
        from bngsim._bngsim_core import NfsimSimulator

        sim = NfsimSimulator(nfsim_xml)
        sim.initialize(42)

        # First segment
        r1 = sim.simulate(0.0, 5.0, 6)
        t1 = np.array(r1.time)
        assert t1[0] == pytest.approx(0.0)
        assert t1[-1] == pytest.approx(5.0)

        # Second segment continues
        r2 = sim.simulate(5.0, 10.0, 6)
        t2 = np.array(r2.time)
        assert t2[0] == pytest.approx(5.0)
        assert t2[-1] == pytest.approx(10.0)

    def test_session_repeated_small_segments_match_run(self, nfsim_xml):
        """Repeated session stepping should match one-shot run()."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        seed = 42
        t_end = 10.0
        n_steps = 100

        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = t_end
        times.n_points = n_steps + 1

        sim_run = NfsimSimulator(nfsim_xml)
        expected = sim_run.run(times, seed)
        expected_obs = np.array(expected.observable_data)

        sim = NfsimSimulator(nfsim_xml)
        sim.initialize(seed)

        try:
            grid = np.linspace(0.0, t_end, n_steps + 1)
            observed = np.empty_like(expected_obs)
            observed[0, :] = np.array(sim.get_observable_values())

            last_steps = 0
            for i in range(1, grid.size):
                seg = sim.simulate(float(grid[i - 1]), float(grid[i]), 2)
                observed[i, :] = np.array(seg.observable_data)[-1, :]
                last_steps = seg.solver_stats.n_steps
        finally:
            if sim.has_session():
                sim.destroy_session()

        np.testing.assert_array_equal(observed, expected_obs)
        assert last_steps == expected.solver_stats.n_steps
