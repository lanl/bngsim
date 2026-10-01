"""A run that continues the previous one continues its events too (issue #693).

``run_until`` legs, and a ``run`` whose span starts where the last one ended,
used to restart the events at every leg: each trigger's baseline was reseeded
from its ``initialValue``, which describes the trigger before the simulation
starts, so an ``initialValue=false`` event still true at the boundary fired
again; and a delayed execution pending across the boundary was dropped. The
model now carries each trigger's truth and the pending executions beside the
species and the clock, so N legs give the answer one run gives.

Oracles are closed forms: A decays at k1 = 0.1 from A(0) = 10.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

E = np.exp
# One dose of 5 at t = 5.
DOSE = "species A = 10; species n = 0; k1 = 0.1; R1: A => ; k1*A;" + (
    " E: at (time >= 5), t0 = false: A = A + 5, n = n + 1;"
)
DOSE_A20 = (10 * E(-0.5) + 5) * E(-1.5)
# Fires at t = 9, freezes A(9) + 5, applies it at t = 11.
DELAYED = "species A = 10; k1 = 0.1; R1: A => ; k1*A; D: at 2 after (time >= 9): A = A + 5;"
DELAYED_A20 = (10 * E(-0.9) + 5) * E(-0.9)


def _sim(text, method="ode"):
    kw = {"poplevel": 100} if method == "psa" else {}
    return bngsim.Simulator(bngsim.Model.from_antimony_string(text), method=method, **kw)


def _at(r, name, row=-1):
    return float(r.species[row, list(r.species_names).index(name)])


def test_two_legs_give_one_run():
    s = _sim(DOSE)
    a = s.run_until(10)
    b = s.run_until(20)
    assert _at(b, "n") == 1
    assert _at(b, "A") == pytest.approx(DOSE_A20, rel=1e-5)
    # The legs agree about the time point they share.
    assert _at(b, "A", 0) == _at(a, "A")


def test_many_legs_past_the_trigger():
    s = _sim(DOSE)
    for t in (6, 7, 8, 9, 10):
        r = s.run_until(t)
    assert _at(r, "n") == 1


def test_a_run_whose_span_starts_where_the_last_ended():
    s = _sim(DOSE)
    s.run(t_span=(0, 10), n_points=11)
    r = s.run(t_span=(10, 20), n_points=11)
    assert _at(r, "n") == 1
    assert _at(r, "A") == pytest.approx(DOSE_A20, rel=1e-5)


def test_a_fresh_start_still_fires_an_initial_value_false_event():
    """Not a continuation: after reset(), or from another t_start, an event true
    at the start fires there, as before."""
    text = "species n = 0; E: at (time >= 0), t0 = false: n = n + 1;"
    s = _sim(text)
    assert _at(s.run(t_span=(0, 1), n_points=2), "n") == 1
    s.model.reset()
    assert _at(s.run(t_span=(0, 1), n_points=2), "n") == 1


def test_intervene_between_legs():
    s = _sim(DOSE)
    s.run_until(10)
    s.intervene({"k1": 0.1})
    assert _at(s.run_until(20), "n") == 1


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_stochastic_legs(method):
    text = "species n = 0; species B = 0; J: => B; 1; E: at (time >= 5), t0 = false: n = n + 1;"
    s = _sim(text, method)
    s.run_until(10, seed=1)
    assert _at(s.run_until(20, seed=2), "n") == 1


def test_a_delayed_execution_pending_across_the_boundary():
    s = _sim(DELAYED)
    s.run_until(10)
    assert _at(s.run_until(20), "A") == pytest.approx(DELAYED_A20, rel=1e-5)


def test_a_rewind_rewinds_the_events():
    s = _sim(DOSE)
    s.run_until(10)
    snap = s.snapshot()
    s.run_until(20)
    s.restore(snap)
    r = s.run_until(20)
    assert _at(r, "n") == 1
    assert _at(r, "A") == pytest.approx(DOSE_A20, rel=1e-5)


def test_a_rewind_restores_a_pending_execution():
    """Snapshot while the delayed dose is pending: the rerun applies it once."""
    s = _sim(DELAYED)
    s.run_until(10)
    snap = s.snapshot()
    s.run_until(20)
    s.restore(snap)
    assert _at(s.run_until(20), "A") == pytest.approx(DELAYED_A20, rel=1e-5)


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_an_intervention_that_raises_a_trigger_fires_at_the_boundary(method):
    """A trigger is an edge in time: ``A < 3`` was false when leg 1 ended, and
    setting A = 1 between the legs makes it true, so the event fires at the start
    of leg 2 even with ``initialValue=true`` (a fresh start would not fire it)."""
    text = "species A = 10; species n = 0; E: at (A < 3), t0 = true: n = n + 1;"
    s = _sim(text, method)
    s.run_until(5)
    s.model.set_concentration("A", 1.0)
    assert _at(s.run_until(10), "n") == 1


def test_a_scan_from_the_carried_time_starts_every_point_alike():
    """Each point of a scan starting where the last run ended continues that
    run's events, not the previous point's."""
    s = _sim(DOSE)
    s.run_until(10)
    res = s.parameter_scan("k1", [0.1, 0.2, 0.3], t_span=(10, 20), n_points=3)
    assert [_at(r, "n") for r in res] == [1, 1, 1]
    # ...and leaves the interactive session where it was.
    assert _at(s.run_until(20), "n") == 1


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_a_step_rolled_back_and_redone(method):
    """A predictor-corrector loop redoes every step from its start
    (``set_state(x, time=t0)``): the redone step continues the events the step
    started from, not the ones the predictor left, however often it is redone."""
    s = _sim(DOSE, method)
    kw = {"seed": 1} if method != "ode" else {}
    s.run_until(4, **kw)
    for t0 in (4.0, 6.0, 8.0):
        saved = s.get_state().copy()
        s.run_until(t0 + 2, **kw)
        for _ in range(2):
            s.set_state(saved, time=t0)
            r = s.run_until(t0 + 2, **kw)
    assert _at(r, "n") == 1
    if method == "ode":
        assert _at(r, "A") == pytest.approx((10 * E(-0.5) + 5) * E(-0.5), rel=1e-5)
