"""An event whose trigger sits beside a state-dependent rate-law switch.

A forward-sensitivity run stops at a rate-law switch's crossing and restarts a
probe step PAST the surface (issues #82, #150), about 1e-13 of the time scale.
An event trigger on the same threshold, or one within that step of it, is
carried across by the restart. CVODE then starts with the event's root already
on its far side and never reports it. Found while fixing issue #910.

Two things followed, in a sensitivity run only and with nothing said:

* a trigger that ROSE across the restart never fired. ``at (A < thr)`` beside
  ``piecewise(kb, A < thr, 0)``, one threshold written twice, lost its event:
  the trajectory itself was wrong, where the plain run is right;
* a trigger that FELL across the restart was still recorded as true, so its next
  rise, however far from any switch, was not a rising edge and did not fire.

The switch's saltation jump and the event's jump cannot be composed at one
instant (issue #150 refuses the exact coincidence), so the first is refused the
same way. The second needs no refusal: the fall is recorded.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SimulationError

EPS = float(np.finfo(float).eps)


def _final(text, params=None, times=(0.0, 4.0, 8.0, 12.0)):
    model = bngsim.Model.from_antimony_string(text)
    kw = {"sensitivity_params": list(params)} if params else {}
    run = bngsim.Simulator(model, method="ode", **kw).run(
        sample_times=list(times), rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = dict(zip(names, np.asarray(run.species)[-1], strict=True))
    s = np.asarray(run.sensitivities)[-1, names.index("B")] if params else None
    return x, s


RISE = (
    "species A, Y, W, B; A = 10; Y = 0; W = 0; B = 0; a = 0.5; thr1 = 2; thr2 = {thr2:.17g};"
    " kb = 3\n"
    "J0: A -> ; a*A\nJ1: -> Y; piecewise(kb, A < thr1, 0)\nJ2: -> B; W\n"
    "E: at (A < thr2): W = 1\n"
)


@pytest.mark.parametrize("k", [0, 50, 100, 300, 600, 1000, -100, -300])
def test_an_event_on_a_switch_threshold_fires_or_is_refused(k):
    """A decays past thr1, where Y's source switches, and past thr2 = thr1·(1 − k·ε),
    where the event sets W = 1 and B starts to grow: B(T) = T − ln(A0/thr2)/a.
    For k from 0 to 300 the sensitivity run returned W = 0 and B = 0. Those may
    be refused; at 600, 1000 and −300 the event is its own crossing and the run
    has to be right. (−100 is refused on main too: there CVODE reports the two
    roots together, the exact coincidence of issue #150.)"""
    thr2 = float(2.0 * (1 - k * EPS))
    text = RISE.format(thr2=thr2)
    plain, _ = _final(text, times=(0.0, 4.0, 8.0))
    want_b = 8.0 - np.log(10.0 / thr2) / 0.5
    assert plain["W"] == 1.0 and plain["B"] == pytest.approx(want_b, rel=1e-8)
    try:
        sens, s = _final(text, ["a", "thr1", "kb"], times=(0.0, 4.0, 8.0))
    except SimulationError as e:
        # Well clear of the restart step the two are separate crossings, and
        # the run must not be refused.
        assert k in (0, 50, 100, 300, -100), str(e)
        assert "state-dependent rate-law switch" in str(e) and "event" in str(e)
        return
    assert sens["W"] == 1.0 and sens["B"] == pytest.approx(want_b, rel=1e-8)
    # dB/da = −dt*/da = ln(A0/thr2)/a²; thr1 and kb do not move the event.
    np.testing.assert_allclose(s, [np.log(10.0 / thr2) / 0.25, 0.0, 0.0], rtol=1e-6, atol=1e-9)


def test_the_same_threshold_written_twice_is_refused_not_skipped():
    """k = 0, the natural spelling. The message names the switch and the event."""
    with pytest.raises(SimulationError, match="event 'E'"):
        _final(RISE.format(thr2=2.0), ["a", "thr1", "kb"], times=(0.0, 4.0, 8.0))


@pytest.mark.parametrize("trigger", ["A < 10", "10 > A"])
def test_a_trigger_that_starts_on_its_threshold_fires_whichever_way_it_is_written(trigger):
    """A(0) = 10 and A decays, so the trigger is false at the start and true
    from the first instant on: the event fires at once, W = 1 and B(8) = 8.

    A trigger that starts ON its threshold fires only if the trajectory leaves
    into its true side (issue #340). That was read off the sign of d/dt of
    ``lhs - rhs``, taking positive for true, which is right for ``>`` and wrong
    for ``<``: ``at (A < 10)`` never fired, and W and B stayed 0, where
    ``at (10 > A)`` fired. That is in a run without sensitivities too. The
    review of the #910 fix found it."""
    text = (
        "species A, W, B; A = 10; W = 0; B = 0; a = 0.5\n"
        f"J0: A -> ; a*A\nJ1: -> B; W\nE: at ({trigger}): W = 1\n"
    )
    plain, _ = _final(text, times=(0.0, 4.0, 8.0))
    assert plain["W"] == 1.0 and plain["B"] == pytest.approx(8.0, rel=1e-8)
    sens, s = _final(text, ["a"], times=(0.0, 4.0, 8.0))
    assert sens["W"] == 1.0 and sens["B"] == pytest.approx(8.0, rel=1e-8)
    assert s[0] == pytest.approx(0.0, abs=1e-9)  # the fire time does not move with a


@pytest.mark.parametrize("trigger", ["A <= 10", "A > 10"])
def test_a_trigger_that_does_not_leave_its_threshold_into_its_true_side_does_not_fire(trigger):
    """``A <= 10`` is true at the start, so there is no rising edge. ``A > 10``
    is false at the start and A moves away from it."""
    text = (
        "species A, W, B; A = 10; W = 0; B = 0; a = 0.5\n"
        f"J0: A -> ; a*A\nJ1: -> B; W\nE: at ({trigger}): W = 1\n"
    )
    plain, _ = _final(text, times=(0.0, 4.0, 8.0))
    assert plain["W"] == 0.0 and plain["B"] == 0.0


# C is fed through a smooth gate on a clock species T, about 1e-17 at the switch
# and kin per unit time past t = 6, so nothing stops the run between the fall
# and the rise.
REARM = (
    "species A, C, T, Y, W, B; A = 10; C = 0; T = 0; Y = 0; W = 0; B = 0; a = 0.5; thr1 = 2;"
    " thr2 = {thr2:.17g}; kb = 3; kin = {kin!r}\n"
    "J0: A -> ; a*A\nJ1: -> Y; piecewise(kb, A < thr1, 0)\nJ2: -> B; W\n"
    "J3: -> C; kin*T^60/(6^60 + T^60)\nJ4: -> T; 1\n"
    "E: at (A + C > thr2): W = 1\n"
)


@pytest.mark.parametrize("k", [100, 300, 1000, -100])
def test_a_trigger_that_falls_beside_a_switch_fires_on_its_next_rise(k):
    """``A + C > thr2`` is true at the start, falls as A decays past thr2 (a hair
    past the switch at thr1 for k = 100 and 300), and rises again near t = 7.8
    through C alone. The fall was carried across the restart unrecorded, so the
    rise was not an edge: W stayed 0 and every column of B was 0."""
    thr2 = float(2.0 * (1 - k * EPS))
    plain, _ = _final(REARM.format(thr2=thr2, kin=1.0))
    sens, s = _final(REARM.format(thr2=thr2, kin=1.0), ["a", "thr1", "kb", "kin"])
    assert plain["W"] == 1.0 and sens["W"] == 1.0
    assert sens["B"] == pytest.approx(plain["B"], rel=1e-8)
    h = 1e-5
    up, _ = _final(REARM.format(thr2=thr2, kin=1.0 + h))
    dn, _ = _final(REARM.format(thr2=thr2, kin=1.0 - h))
    assert s[3] == pytest.approx((up["B"] - dn["B"]) / (2 * h), rel=1e-3)
    assert s[1] == pytest.approx(0.0, abs=1e-9) and s[2] == 0.0
