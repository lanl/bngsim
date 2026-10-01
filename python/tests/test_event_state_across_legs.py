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

import warnings

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


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_a_leg_ending_off_the_grid(method):
    """0 + 73·0.1 is 7.300000000000001: the last output time is t_end itself, so
    the next leg starts where the carry was left."""
    s = _sim(DOSE, method)
    kw = {"seed": 1} if method != "ode" else {}
    r = s.run_until(7.3, **kw)
    assert r.time[-1] == 7.3 == s.current_time
    r = s.run_until(20, **kw)
    assert _at(r, "n") == 1
    if method == "ode":
        assert _at(r, "A") == pytest.approx(DOSE_A20, rel=1e-5)


def test_a_rerun_of_the_same_span_is_a_fresh_start():
    """After ``run_until(10)``, ``run(t_span=(10, 20))`` three times: the second and
    third start at 10 from the state the first left at 20, which is no state the
    events were carried for, so each is a fresh start, as on main."""
    text = "species A = 10; species n = 0; R1: A => ; 0.1*A; E: at (A < 3): n = n + 1;"
    s = _sim(text)
    s.run_until(10)
    for _ in range(3):
        assert _at(s.run(t_span=(10, 20), n_points=3), "n") == 1


@pytest.mark.parametrize(("T", "d", "L"), [(15.3, 1.1, 16.4), (6.9, 1.3, 8.2)])
@pytest.mark.parametrize("n_points", [None, 2])
def test_a_carried_execution_due_an_ulp_into_the_next_leg(T, d, L, n_points):
    """T + d is one ulp past the leg end L: the execution is carried, due an ulp
    after the next leg starts, and is applied there, not refused (CV_TOO_CLOSE)
    or deferred to the next output point."""
    text = (
        "species A = 10; species n = 0; k1 = 0.1; R1: A => ; k1*A;"
        f" D: at {d} after (time >= {T}): A = A + 5, n = n + 1;"
    )
    s = _sim(text)
    s.run_until(L)
    r = s.run_until(L + 5, n_points=n_points) if n_points else s.run_until(L + 5)
    assert _at(r, "n") == 1
    exact = (10 * E(-0.1 * T) + 5) * E(-0.1 * (L + 5 - (T + d)))
    assert _at(r, "A") == pytest.approx(exact, rel=1e-5)


def test_a_rollback_past_the_last_leg():
    """Back two legs (6 → 8 → 10, then to 6): the events go back to where the
    leg ending at 6 left them."""
    s = _sim(DOSE)
    s.run_until(6)
    saved = s.get_state().copy()
    s.run_until(8)
    s.run_until(10)
    s.set_state(saved, time=6)
    r = s.run_until(10)
    assert _at(r, "n") == 1
    assert _at(r, "A") == pytest.approx((10 * E(-0.5) + 5) * E(-0.5), rel=1e-5)


def test_a_carry_is_checked_against_the_model():
    s = _sim(DELAYED)
    s.run_until(10)
    t, trig, pend = s.model._core.event_carry()
    core = s.model._core
    with pytest.raises(ValueError, match="frozen values"):
        core.set_event_carry((t, trig, [(0, 11.0, [1.0, 2.0, 3.0])]))
    with pytest.raises(ValueError, match="no event"):
        core.set_event_carry((t, trig, [(5, 11.0, [])]))
    with pytest.raises(ValueError, match="trigger values"):
        core.set_event_carry((t, [*trig, True], pend))
    core.set_event_carry((t, trig, pend))
    assert core.event_carry() == (t, trig, pend)


def test_a_scan_leaves_no_leg_end_to_roll_back_to():
    """The scan's points publish their own leg ends at 10; a rollback to 10 after
    it must find the interactive run's, with its pending dose."""
    s = _sim(DELAYED)
    s.run_until(10)
    s.parameter_scan("k1", [0.1, 0.2], t_span=(0, 10), n_points=2)
    s.set_time(s.current_time)
    assert _at(s.run_until(20), "A") == pytest.approx(DELAYED_A20, rel=1e-5)


def test_replicates_leave_no_leg_end_to_roll_back_to():
    s = _sim(DOSE, "ssa")
    s.run_until(10, seed=1)
    s.run_replicates(5, t_span=(0, 10), n_points=2, seed=3)
    s.set_time(s.current_time)
    assert _at(s.run_until(20, seed=2), "n") == 1


def test_a_rollback_onto_an_abandoned_branch_is_a_fresh_start():
    """Branch A queues a dose at 6 (applied at 7); restored to 4, branch B turns
    the dose off. Rolling branch B back to 6 must not pick up A's queued dose."""
    text = (
        "species A = 10; species n = 0; k1 = 0.1; tf = 5; R1: A => ; k1*A;"
        " D: at 2 after (time >= tf): A = A + 5, n = n + 1;"
    )
    s = _sim(text)
    s.run_until(4)
    s.snapshot()
    s.run_until(6)
    s.run_until(10)
    s.restore()
    s.intervene({"tf": 100.0})
    r = s.run_until(10, n_points=7)
    x6 = np.asarray(s.model.get_state()).copy()
    x6[list(s.model.species_names).index("A")] = _at(r, "A", 2)
    s.set_state(x6, time=6.0)
    assert _at(s.run_until(10), "n") == 0


def test_an_intervention_cancels_a_carried_non_persistent_execution():
    """``A > 0.5`` queues an execution 1.5 later; setting A = 0 between the legs
    makes the trigger fall, which cancels a non-persistent one, and its next rise
    (at 1.5) queues the only execution that runs, at 3."""
    text = (
        "species A = 0; species n = 0; J: => A; 1;"
        " D: at 1.5 after (A > 0.5), persistent = false, fromTrigger = false: n = n + 1;"
    )
    for n_points in (2, 5):
        s2 = _sim(text)
        s2.run_until(1)
        s2.model.set_concentration("A", 0.0)
        assert _at(s2.run_until(5, n_points=n_points), "n") == 1


def test_a_rollback_past_the_kept_legs_warns():
    s = _sim(DOSE)
    s.run_until(2)
    saved = s.get_state().copy()
    for t in range(3, 3 + 70):
        s.run_until(t)
    with pytest.warns(UserWarning, match="rolls back past the last 64 legs"):
        s.set_state(saved, time=2.0)
    s2 = _sim(DOSE)
    s2.run_until(2)
    for t in range(3, 10):
        s2.run_until(t)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s2.set_state(saved, time=2.0)


def test_a_carry_must_be_finite():
    s = _sim(DELAYED)
    s.run_until(10)
    t, trig, pend = s.model._core.event_carry()
    with pytest.raises(ValueError, match="not finite"):
        s.model._core.set_event_carry((t, trig, [(0, float("nan"), [])]))
    with pytest.raises(ValueError, match="not finite"):
        s.model._core.set_event_carry((float("inf"), trig, pend))


@pytest.mark.parametrize("n_points", [2, 5])
def test_an_event_at_the_leg_start_cancels_a_carried_non_persistent_execution(n_points):
    """As the intervention test, but A is zeroed by an event that fires at the
    leg start: the carried execution (due at 2) is cancelled by it, before D's
    trigger rises again at 1.5."""
    text = (
        "species A = 0; species n = 0; tE = 100; J: => A; 1;"
        " Z: at (time >= tE), t0 = false: A = 0;"
        " D: at 1.5 after (A > 0.5), persistent = false, fromTrigger = false: n = n + 1;"
    )
    s = _sim(text)
    s.run_until(1)
    s.intervene({"tE": 1.0})
    assert _at(s.run_until(5, n_points=n_points), "n") == 1


def test_a_look_back_then_forward_again():
    """Back to 10 and forward to 10.5 again, touching nothing: the run continues
    from 10.5 with the dose still pending and the counter not refired."""
    text = (
        "species A = 10; species n = 0; species m = 0; k1 = 0.1; R1: A => ; k1*A;"
        " D: at 2 after (time >= 9): A = A + 5, n = n + 1;"
        " E: at (time >= 5), t0 = false: m = m + 1;"
    )
    s = _sim(text)
    s.run_until(10)
    s.run_until(10.5)
    s.set_time(10)
    s.set_time(10.5)
    r = s.run_until(20)
    assert (_at(r, "n"), _at(r, "m")) == (1, 1)
