"""Tests for fixes from the 2026-04-05 compatibility review.

Covers:
- #6:  HAS_NFSIM exported from bngsim.__init__
- #14: sample_times minimum relaxed from 3 to 2
- #15: Stop condition eval failures logged instead of silently swallowed
"""

from __future__ import annotations

import logging
from pathlib import Path

import bngsim
import numpy as np
import pytest

# ─── #6: HAS_NFSIM in public API ────────────────────────────────────────


class TestHasNfsimExport:
    def test_has_nfsim_in_module(self):
        """bngsim.HAS_NFSIM is accessible without reaching into _bngsim_core."""
        assert hasattr(bngsim, "HAS_NFSIM")
        assert isinstance(bngsim.HAS_NFSIM, bool)

    def test_has_nfsim_in_all(self):
        """HAS_NFSIM is listed in __all__."""
        assert "HAS_NFSIM" in bngsim.__all__


# ─── #14: sample_times minimum relaxed to 2 ─────────────────────────────


class TestSampleTimesMinimum:
    def test_sample_times_two_points_works(self, simple_decay_net: Path):
        """sample_times with exactly 2 points should succeed."""
        model = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(sample_times=[0.0, 100.0])

        assert result.n_times == 2
        np.testing.assert_almost_equal(result.time[0], 0.0)
        np.testing.assert_almost_equal(result.time[-1], 100.0)

    def test_sample_times_one_point_raises(self, simple_decay_net: Path):
        """sample_times with 1 point should still be rejected."""
        model = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(model, method="ode")
        with pytest.raises(ValueError, match="at least 2"):
            sim.run(sample_times=[50.0])


# ─── #15: Stop condition eval logging ────────────────────────────────────


class TestStopConditionLogging:
    def test_bad_expression_logs_debug(self, simple_decay_net: Path, caplog):
        """A stop condition with an undefined variable logs at DEBUG level."""
        model = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(model, method="ode")
        sim.add_stop_condition("nonexistent_var > 999", label="bad_cond")

        with caplog.at_level(logging.DEBUG, logger="bngsim"):
            # Should complete without raising — the bad condition
            # just never triggers.
            result = sim.run(t_span=(0, 10), n_points=11)

        assert result.n_times == 11
        # Verify that a debug message was logged about the eval failure
        debug_messages = [
            r.message
            for r in caplog.records
            if r.levelno == logging.DEBUG and "eval failed" in r.message
        ]
        assert len(debug_messages) > 0
        assert "nonexistent_var" in debug_messages[0]
