"""Issue #1658: an SSA/PSA row on a state trigger's crossing holds the state after it.

``q' = 0.1`` from 0 meets ``q >= 2.5`` at t = 25. A row on an event's instant
reports the state after it (#1124), as the ODE run and libRoadRunner do, so
[S](25) = 1 after ``S = 1``. The SSA integrates q over panels, and q(25) landed
a few ulps either side of 2.5 as the grid's end moved: the row read the trigger
false at t_end = 25, 27, 30 and 50 and held S = 3, with no error. The rows now
read the rate-rule state a bisection tolerance ahead, the clock at the row, as
the ODE run reads them half a root tolerance ahead (#1133).

No reaction moves S or q, so a run is deterministic.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

STATE = "species S = 3; q = 0; q' = 0.1; E1: at (q >= 2.5): S = 1;"


def _s_at_25(text, method, t_end, seed=1):
    kw = {"poplevel": 100} if method == "psa" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method=method, **kw).run(
            t_span=(0, t_end), n_points=t_end + 1, seed=seed
        )
    row = list(r.time).index(25.0)
    return float(np.asarray(r.species)[row, list(r.species_names).index("S")])


@pytest.mark.parametrize("method", ["ode", "ssa", "psa"])
@pytest.mark.parametrize("t_end", [25, 26, 27, 30, 40, 50])
def test_a_state_trigger_met_on_a_row_is_after_the_event(method, t_end):
    assert _s_at_25(STATE, method, t_end) == 1.0


@pytest.mark.parametrize("method", ["ssa", "psa"])
@pytest.mark.parametrize("t_end", [25, 27, 50])
def test_a_strict_clock_trigger_on_a_row_stays_before_it(method, t_end):
    """The read-ahead moves the state only: `time > 25` is false at 25, so the
    row holds the state before the event, as ODE and libRoadRunner hold it."""
    assert _s_at_25("species S = 3; E1: at (time > 25): S = 1;", method, t_end) == 3.0


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_crossing_after_the_row_is_not_moved_onto_it(method):
    """q reaches 2.5 at 25.0001 (q' = 0.1 from -1e-5): the row at 25 holds the
    state before the event, and the next row after it."""
    text = "species S = 3; q = -1e-5; q' = 0.1; E1: at (q >= 2.5): S = 1;"
    kw = {"poplevel": 100} if method == "psa" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method=method, **kw).run(
            sample_times=[0.0, 25.0, 25.001], seed=1
        )
    np.testing.assert_array_equal(np.asarray(r.species)[:, 0], [3.0, 3.0, 1.0])


def test_with_a_reaction_every_replicate_fires_on_the_row():
    """The issue's last check: a decay beside the event, 50 seeds, t_end = 50.
    Every one read [S](25) = 3."""
    text = "species A = 100; J: A -> ; 0.05*A; " + STATE
    assert {_s_at_25(text, "ssa", 50, seed=k) for k in range(1, 51)} == {1.0}
