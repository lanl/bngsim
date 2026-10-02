"""A rate law that switches on the state, on the instant of a switch on a
clock or of an event (issues #946 and #945).

The jump of a state-dependent switch is read as the whole right-hand side a
probe step before the crossing less the same a step after it, with the time
moved as the state is. A condition on a clock that switches between the two
is in that difference, and its jump was given the state switch's shift: twice
over where the clock switch is fitted, since its own record adds it too, and
once where it is fixed and should have none.

Where the two are composed, in one rate law or through an event that changes
what the switched law reads, their order is what the result turns on, and a
state crossing's time is known only to the tolerances of the run. The run
returned one side of a kink.

Both are refused. Every expected value is a closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

TIMES = [0.0, 1.0, 2.5, 3.5, 5.0]
# S = 0.5·t reaches 0.5·thr at t = thr.
HEAD = "species S, X, Y; S = 0; X = 0; Y = 0; k = 0.5; thr = 3\nJs: -> S; 0.5\n"


def _columns(text, params, times=TIMES):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
        sample_times=times, rtol=1e-10, atol=1e-12, timeout=60
    )
    return np.asarray(run.sensitivities)[-1]


def _refused(text, params, issue, times=TIMES):
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
    with pytest.raises(bngsim.SimulationError, match=f"#{issue}"):
        sim.run(sample_times=times, rtol=1e-10, atol=1e-12, timeout=60)


ON_ONE_INSTANT = {
    # dX/dthr = −1 for −0.5: the time switch's jump, twice.
    "a-fitted-time-switch-in-another-law": (
        "Jx: -> X; piecewise(k, time >= thr, 0)\nJy: -> Y; piecewise(k, S >= 0.5*thr, 0)\n"
    ),
    # dX/dthr = −0.5 for 0: a fixed gate given the state switch's shift.
    "a-fixed-time-switch-in-another-law": (
        "Jx: -> X; piecewise(k, time >= 3, 0)\nJy: -> Y; piecewise(k, S >= 0.5*thr, 0)\n"
    ),
    # A kink, −0.5 | −0.25: the two in one rate law.
    "a-fixed-time-switch-in-the-same-law": (
        "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)*piecewise(1, time >= 3, 0.5)\n"
    ),
}


@pytest.mark.parametrize("case", sorted(ON_ONE_INSTANT))
def test_a_state_switch_on_the_instant_of_a_time_switch_is_refused(case):
    _refused(HEAD + ON_ONE_INSTANT[case], ["thr"], 946)


def test_a_state_switch_a_twentieth_after_the_time_switch_runs():
    """Control. The state threshold 0.05 higher: the two are 0.1 apart in
    time, and each takes its own jump."""
    text = HEAD + (
        "Jx: -> X; piecewise(k, time >= thr, 0)\nJy: -> Y; piecewise(k, S >= 0.5*thr + 0.05, 0)\n"
    )
    np.testing.assert_allclose(_columns(text, ["thr"]), [[0.0], [-0.5], [-0.5]], atol=1e-8)


@pytest.mark.parametrize(
    "law",
    ["Jx: -> X; piecewise(k, time >= thr, 0)\n", "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)\n"],
    ids=["the-time-switch", "the-state-switch"],
)
def test_each_switch_alone_runs(law):
    """Control."""
    got = _columns(HEAD + law, ["thr"])
    want = [[0.0], [-0.5], [0.0]] if "Jx" in law else [[0.0], [0.0], [-0.5]]
    np.testing.assert_allclose(got, want, atol=1e-8)


def test_a_state_switch_that_only_bends_is_refused_there_too():
    """``piecewise(k·(S − 0.5·thr), S >= 0.5·thr, 0)`` is continuous where it
    switches and has no jump of its own. The probes across it read the time
    switch's all the same, and what they read is not told from a jump of the
    state switch's: dX/dthr came back −1 for −0.5."""
    text = HEAD + (
        "Jx: -> X; piecewise(k, time >= thr, 0)\n"
        "Jy: -> Y; piecewise(k*(S - 0.5*thr), S >= 0.5*thr, 0)\n"
    )
    _refused(text, ["thr"], 946)


def test_a_state_switch_that_only_bends_runs_a_tenth_after_the_time_switch():
    """Control. The same bend 0.1 later. Y = k·(T − thr − 0.1)²/4."""
    text = HEAD + (
        "Jx: -> X; piecewise(k, time >= thr, 0)\n"
        "Jy: -> Y; piecewise(k*(S - 0.5*thr - 0.05), S >= 0.5*thr + 0.05, 0)\n"
    )
    np.testing.assert_allclose(_columns(text, ["thr"]), [[0.0], [-0.5], [-0.475]], atol=1e-8)


def test_a_falling_species_on_the_instant_of_a_fixed_gate_is_refused():
    """u falls at rate 1 from 10 and crosses the fitted ua = 7 at t = 3, where
    Z's law switches on the time. Z's gate is fixed: dZ/dua = 0. It came back
    1.5, the gate's jump with u's shift."""
    text = (
        "species Y, Z, u; Y = 0; Z = 0; u = 10; r = 1; ua = 7\nJd: u -> ; 1\n"
        "J1: -> Y; piecewise(r, u <= ua, 0)\nJ9: -> Z; piecewise(2, time >= 3, 0.5)\n"
    )
    _refused(text, ["ua"], 946, [0.0, 1.0, 2.0, 4.0, 5.0, 6.0])


EVENT = (
    "species X, Y, W, V; X = 0; Y = 0; W = 0; V = 0; a = 2; k = 0.5; q = 0.7; tau = 3\n"
    "J0: -> X; a\nJv: -> V; 2\nJ1: -> Y; k\nJ2: -> W; piecewise(q*X, V >= {at}, 0)\n"
    "E1: at (time >= tau): X = 0.5*X\n"
)
EVENT_TIMES = [float(t) for t in np.linspace(0.0, 7.5, 16)]


def test_a_state_switch_on_the_instant_of_an_event_is_refused():
    """V = 2·t reaches 6 at t = 3, where the event halves X, and the switched
    law reads X: dW/dtau is −1.05 with the event first and −3.15 with the
    switch first. V's crossing is known to the tolerance of the run and not to
    the last bit, so the two came as two stops, each handled as if the other
    were not there, and the run returned −3.15."""
    _refused(EVENT.format(at=6), ["tau"], 945, EVENT_TIMES)


def test_a_state_switch_well_after_the_event_runs():
    """Control. V reaches 6.4 at t = 3.2, after the event at 3. W is
    q·X integrated from 3.2, with X halved at tau: dW/dtau = −q·a·(T − 3.2)/2."""
    got = _columns(EVENT.format(at=6.4), ["tau"], EVENT_TIMES)
    assert got[2, 0] == pytest.approx(-0.7 * 2.0 * (7.5 - 3.2) / 2.0, rel=1e-7)
