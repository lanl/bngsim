"""An SSA/PSA run that raises leaves the model as it found it.

The loop writes its running state into the model as it goes: the species it
syncs for observables, functions and event triggers, and the clock. A run
stopped part way, by a timeout or a refusal, used to leave that mid-run state
behind, under a clock and an event state still at the run's start, and the next
run continued from it with no error.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import SimulationTimeout

pytest.importorskip("antimony")

# Fast enough that the long run cannot finish in the timeout; the event makes
# the loop sync the model's species.
TEXT = (
    "species A = 100000; species B = 0; species n = 0; k1 = 1;"
    " R1: A => B; k1*A; R2: B => A; k1*B; E: at (time >= 5), t0 = false: n = n + 1;"
)


def _sim(text, method):
    kw = {"poplevel": 100} if method == "psa" else {}
    return bngsim.Simulator(bngsim.Model.from_antimony_string(text), method=method, **kw)


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_timeout(method):
    s = _sim(TEXT, method)
    s.run_until(10, seed=1)
    before = s.get_state().copy()
    with pytest.raises(SimulationTimeout):
        s.run(t_span=(10, 1e6), n_points=2, timeout=0.05, seed=2)
    np.testing.assert_array_equal(s.get_state(), before)
    assert s.current_time == 10
    # The next leg is the one a run that never saw the timeout takes.
    ref = _sim(TEXT, method)
    ref.run_until(10, seed=1)
    np.testing.assert_array_equal(
        np.asarray(s.run_until(12, seed=3).species), np.asarray(ref.run_until(12, seed=3).species)
    )


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_refusal_part_way(method):
    """``sqrt(3 - time)`` is NaN past t = 3: the run is refused there."""
    text = (
        "species A = 0; species n = 0; J: => A; 50*sqrt(3 - time);"
        " E: at (time >= 1), t0 = false: n = n + 1;"
    )
    s = _sim(text, method)
    before = s.get_state().copy()
    with pytest.raises(bngsim.SimulationError):
        s.run(t_span=(0, 10), n_points=11, seed=1)
    np.testing.assert_array_equal(s.get_state(), before)
