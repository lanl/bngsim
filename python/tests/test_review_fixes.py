"""Tests for fixes from the 2026-04-05 compatibility review.

Covers:
- #6:  HAS_NFSIM exported from bngsim.__init__
- #14: sample_times minimum relaxed from 3 to 2
- #15: Stop condition eval failures logged instead of silently swallowed (now
  refused where the condition is added, issue #1509)
"""

from __future__ import annotations

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


# ─── #15: Stop condition with a name it cannot read ─────────────────────


class TestStopConditionUnreadableName:
    def test_bad_expression_is_refused_where_it_is_added(self, simple_decay_net: Path):
        """A stop condition with an undefined variable is refused when it is added.

        The 2026-04-05 fix logged each failed sample at DEBUG and let the run
        finish, so the condition read as one never met; issue #1509 replaced that
        with a refusal naming the variable.
        """
        model = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(model, method="ode")
        with pytest.raises(ValueError, match="nonexistent_var is not a name the model defines"):
            sim.add_stop_condition("nonexistent_var > 999", label="bad_cond")
