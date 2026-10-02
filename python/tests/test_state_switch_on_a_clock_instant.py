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


@pytest.mark.parametrize(
    ("param", "want"),
    [("q", [0.0, 0.0, 33.75, 0.0]), ("a", [6.0, 0.0, 11.8125, 0.0])],
    ids=["the-switched-rate", "what-the-event-scales"],
)
def test_an_event_on_the_switch_runs_where_no_column_moves_either(param, want):
    """Control. The event at the fixed tau = 3 and V's crossing at t = 3, with
    neither tau nor anything V reads requested: no column moves either, there
    is no shift to give and no kink to be on one side of. X is 1.5·a at 3+ and
    W is q·X integrated from there."""
    got = _columns(EVENT.format(at=6), [param], EVENT_TIMES)
    np.testing.assert_allclose(got[:, 0], want, rtol=1e-7, atol=1e-9)


def test_a_fixed_gate_on_the_switch_runs_where_no_column_moves_either():
    """Control. The gate at t = 3 and S's crossing of 0.5·thr at t = 3, with
    k requested and thr not: X and Y are both k·(T − 3)."""
    text = HEAD + (
        "Jx: -> X; piecewise(k, time >= 3, 0)\nJy: -> Y; piecewise(k, S >= 0.5*thr, 0)\n"
    )
    np.testing.assert_allclose(_columns(text, ["k"]), [[0.0], [2.0], [2.0]], atol=1e-8)


def test_a_fitted_gate_on_a_state_switch_nothing_moves_is_refused():
    """The state crossing is fixed, S = 1.5 at t = 3, and the gate in the same
    law is at the fitted thr = 3: Y is k·(T − 3) for thr below 3 and
    k·(T − 3) − 0.5·k·(thr − 3) above it. dY/dthr is 0 | −0.25."""
    text = HEAD + "Jy: -> Y; piecewise(k, S >= 1.5, 0)*piecewise(1, time >= thr, 0.5)\n"
    _refused(text, ["thr"], 946)


EVENT_ON_STATE = (
    "species X, Y, W, V, U; X = 0; Y = 0; W = 0; V = 0; U = 0; a = 2; k = 0.5; q = 0.7; c = 1\n"
    "J0: -> X; a\nJv: -> V; 2\nJu: -> U; c\nJ1: -> Y; k\nJ2: -> W; piecewise(q*X, V >= 6, 0)\n"
    "E1: at (U > {at}): X = 0.5*X\n"
)


def test_an_event_on_another_species_on_the_instant_of_a_state_switch_is_refused():
    """Control. The event fires where U = c·t reaches 3, at t = 3/c, and V
    reaches 6 at t = 3: two crossings on two species, on one instant, with the
    event's time moved by c. dW/dc has a kink there. Refused before this, as a
    switch on the instant of an event."""
    model = bngsim.Model.from_antimony_string(EVENT_ON_STATE.format(at=3))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["c"])
    with pytest.raises(bngsim.SimulationError, match="Forward sensitivity"):
        sim.run(sample_times=EVENT_TIMES, rtol=1e-10, atol=1e-12, timeout=60)


def test_an_event_on_another_species_after_the_state_switch_runs():
    """Control. The event at t_e = 3.3/c, the switch at 3. X(T) = a·T − 0.5·a·t_e
    and dW/dt_e = 0.5·q·a·(2·t_e − T), with dt_e/dc = −3.3."""
    got = _columns(EVENT_ON_STATE.format(at=3.3), ["c"], EVENT_TIMES)[:, 0]
    want = [3.3, 0.0, 0.5 * 0.7 * 2.0 * (6.6 - 7.5) * -3.3, 0.0, 7.5]
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


def test_the_reach_is_the_tolerance_of_the_run():
    """At rtol 1e-6 the state S is known to 1.5e-6 where it crosses and its
    crossing time to 3e-6. A gate in the same law 1e-7 later is not told from
    one on the instant, and dY/dthr has a kink between the two orders: −0.25
    with the switch first, −0.5 with the gate first. (S is linear and the run
    has it exactly, so the order the run found here was the true one; a state
    that carries its integration error has no such luck.) At rtol 1e-10 the
    same pair is hundreds of reaches apart and runs."""
    text = HEAD + "Jy: -> Y; piecewise(k, S >= 0.5*thr - 5e-8, 0)*piecewise(1, time >= 3, 0.5)\n"
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr"])
    with pytest.raises(bngsim.SimulationError, match="#946"):
        sim.run(sample_times=TIMES, rtol=1e-6, atol=1e-8, timeout=60)
    assert _columns(text, ["thr"])[2, 0] == pytest.approx(-0.25, rel=1e-6)


NET = """begin parameters
    1 k 0.5
    2 thr 3
    3 one 1
    4 half 0.5
end parameters
begin functions
    1 fX() if(t>=thr,k,0)
    2 fY() if(Sobs>={level},k,0)
end functions
begin species
    1 S() 0
    2 X() 0
    3 Y() 0
    4 Tc() 0
end species
begin reactions
    1 0 1 half
    2 0 2 fX
    3 0 3 fY
    4 0 4 one
end reactions
begin groups
    1 Sobs 1
    2 t 4
end groups
"""


def _net_columns(tmp_path, level):
    path = tmp_path / "m.net"
    path.write_text(NET.format(level=level))
    model = bngsim.Model.from_net(str(path))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr"])
    return np.asarray(
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12, timeout=60).sensitivities
    )[-1]


def test_a_state_switch_on_the_instant_of_a_counter_s_switch_is_refused(tmp_path):
    """The clock is a counter species, ``Tc`` with rate 1, and X's law
    switches where it reaches the fitted thr, on the instant S reaches
    0.5·thr. dX/dthr came back −1 for −0.5."""
    with pytest.raises(bngsim.SimulationError, match="#946"):
        _net_columns(tmp_path, "0.5*thr")


def test_a_state_switch_after_a_counter_s_switch_runs(tmp_path):
    """Control. S reaches 0.5·thr + 0.05 a tenth after the counter reaches thr."""
    np.testing.assert_allclose(
        _net_columns(tmp_path, "0.5*thr+0.05"), [[0.0], [-0.5], [-0.5], [0.0]], atol=1e-8
    )
