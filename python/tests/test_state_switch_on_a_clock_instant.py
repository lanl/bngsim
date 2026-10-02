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

import math

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


def _refused(text, params, issue, times=TIMES, rtol=1e-10, atol=1e-12, ic=None):
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params, sensitivity_ic=ic)
    with pytest.raises(bngsim.SimulationError, match=f"#{issue}"):
        sim.run(sample_times=times, rtol=rtol, atol=atol, timeout=60)


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
    "species X, Y, W, V; X = 0; Y = 0; W = 0; V = 0; a = 2; k = 0.5; q = 0.7; tau = 3; kv = 2\n"
    "J0: -> X; a\nJv: -> V; kv\nJ1: -> Y; k\nJ2: -> W; piecewise(q*X, V >= {at}, 0)\n"
    "E1: at (time >= tau): X = 0.5*X\n"
)
EVENT_TIMES = [float(t) for t in np.linspace(0.0, 7.5, 16)]


def test_a_state_switch_on_the_instant_of_an_event_is_refused():
    """V = 2·t reaches 6 at t = 3, where the event halves X, and the switched
    law reads X: dW/dtau is −1.05 with the switch first and −3.15 with the
    event first. V's crossing is known to the tolerance of the run and not to
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


@pytest.mark.parametrize("off", [-4e-10, 4e-10], ids=["the-switch-first", "the-event-first"])
def test_a_state_switch_within_reach_of_an_event_is_refused_in_either_order(off):
    """V reaches its threshold 2e-10 before the event and 2e-10 after it, at
    rtol 1e-10: inside what the run knows V's crossing time to, 3e-10. The
    refusal is made at the second of the two to come, the switch or the fire.
    (V is linear and the run has it exactly, so the side it would have
    returned here was the true one.)"""
    _refused(EVENT.format(at=repr(6.0 + off)), ["tau"], 945, EVENT_TIMES)


@pytest.mark.parametrize("off", [-4e-9, 4e-9], ids=["the-switch-first", "the-event-first"])
def test_a_state_switch_outside_the_reach_of_an_event_runs(off):
    """Control. 2e-9 apart at rtol 1e-10: −1.05 with the switch first, −3.15
    with the event first."""
    got = _columns(EVENT.format(at=repr(6.0 + off)), ["tau"], EVENT_TIMES)
    assert got[2, 0] == pytest.approx(-1.05 if off < 0 else -3.15, rel=1e-7)


@pytest.mark.parametrize("off", [-4e-10, 0.0, 4e-10], ids=["before", "on", "after"])
def test_a_state_switch_a_column_moves_on_a_fixed_event_is_refused(off):
    """The event's time is fixed and V's rate kv is requested: the switch
    moves through the event. dW/dkv is 6.3 with the switch first, where X is
    not yet halved, and 3.15 with the event first. On the instant the run
    returned 3.15."""
    _refused(EVENT.format(at=repr(6.0 + off)), ["kv"], 945, EVENT_TIMES)


@pytest.mark.parametrize("off", [-2e-10, 2e-10], ids=["the-event-first", "the-switch-first"])
def test_an_event_on_another_species_within_reach_of_a_state_switch_is_refused(off):
    """The event fires where U = c·t reaches its threshold, 2e-10 before V's
    crossing and 2e-10 after it: dW/dc is 9.45 on one side and 3.15 on the
    other, and the two crossings are on two species, each known to 3e-10."""
    _refused(EVENT_ON_STATE.format(at=repr(3.0 + off)), ["c"], 945, EVENT_TIMES)


FALLING = (
    "species Y, u; Y = 0; u = 10; r = 1; ua = 0\nJd: u -> ; 1\n"
    "J1: -> Y; piecewise(r, u <= ua, 0)*piecewise(2, time >= {gate}, 0.5)\n"
)
FALLING_TIMES = [0.0, 5.0, 9.0, 11.0, 12.0]


def _falling(off):
    model = bngsim.Model.from_antimony_string(FALLING.format(gate=repr(10.0 + off)))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["ua"])
    return sim.run(sample_times=FALLING_TIMES, rtol=1e-10, atol=1e-8, timeout=60)


@pytest.mark.parametrize("off", [-1e-9, 1e-9], ids=["the-gate-first", "the-switch-first"])
def test_the_reach_at_a_state_of_zero_is_the_absolute_tolerance(off):
    """u falls through 0 at t = 10, where the relative tolerance allows
    nothing: its crossing time is known to atol over its rate, 1e-8. A gate in
    the same law 1e-9 away is not told from one on the instant, and dY/dua is
    2 with the gate first and 0.5 with the switch first."""
    with pytest.raises(bngsim.SimulationError, match="#946"):
        _falling(off)


def test_a_gate_outside_the_absolute_tolerance_s_reach_runs():
    """Control. The gate 1e-6 after the crossing: dY/dua = 0.5."""
    got = np.asarray(_falling(1e-6).sensitivities)[-1]
    assert got[0, 0] == pytest.approx(0.5, rel=1e-7)


def test_a_time_switch_between_the_probes_is_refused_at_any_tolerance():
    """The jump is read two probe steps either side of the crossing, whatever
    the tolerances: at rtol 1e-14 they put the crossing time within 3e-14,
    and a fitted time switch 2e-13 before it is still between the probes,
    where its jump is read as this switch's. dX/dthr came back 3e-14 for
    −0.5."""
    text = HEAD + (
        "Jx: -> X; piecewise(k, time >= thr, 0)\nJy: -> Y; piecewise(k, S >= 0.5*thr + 1e-13, 0)\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr"])
    with pytest.raises(bngsim.SimulationError, match="#946"):
        sim.run(sample_times=TIMES, rtol=1e-14, atol=1e-16, timeout=60)


ROOTED = (
    "species X, Y, W, V; X = 0; Y = 0; W = 0; V = 0; a = 2; k = 0.5; q = 0.7; kv = 2\n"
    "J0: -> X; a\nJv: -> V; kv\nJ1: -> Y; k\nJ2: -> W; piecewise(q*X, V >= {at}, 0)\n"
    "E1: at (time + 0.1*sin(time) >= 3.0141120008059867): X = 0.5*X\n"
)


@pytest.mark.parametrize("off", [-4e-10, 0.0, 4e-10], ids=["before", "on", "after"])
def test_a_moved_switch_on_an_event_whose_time_is_found_as_a_root_is_refused(off):
    """The trigger ``time + 0.1·sin(time) >= 3 + 0.1·sin(3)`` is true from
    t = 3, a time the run finds as a root and takes no stop for. V's rate kv
    is requested and moves the switch through the fire: dW/dkv is 6.3 with
    the switch first and 3.15 with the event first. With the switch first
    the refusal is made at the fire, which no column moves, for the switch
    that one does."""
    _refused(ROOTED.format(at=repr(6.0 + off)), ["kv"], 945, EVENT_TIMES)


def test_an_event_whose_time_is_found_as_a_root_runs_where_no_column_moves_either():
    """Control. The same with q requested: W = q·X integrated from 3."""
    got = _columns(ROOTED.format(at=6), ["q"], EVENT_TIMES)
    np.testing.assert_allclose(got[:, 0], [0.0, 0.0, 33.75, 0.0], rtol=1e-7, atol=1e-9)


# ── What the review of this change found ────────────────────────────────────

SWITCH = "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)\n"
GATES_ON_A_POLYNOMIAL = {
    # dX/dthr = −0.5 for 0, each: the gate's jump with the state switch's shift.
    "a-square-in-another-law": HEAD + "Jx: -> X; piecewise(k, time^2 >= 9, 0)\n" + SWITCH,
    "a-quadratic-in-another-law": (
        HEAD + "Jx: -> X; piecewise(k, time^2 + time >= 12, 0)\n" + SWITCH
    ),
    # −0.5 on the kink −0.25 | −0.5.
    "a-square-in-the-same-law": (
        HEAD + "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)*piecewise(1, time^2 >= 9, 0.5)\n"
    ),
    # 0.5 on the kink 0.25 | 0.5. The gate shows only with S held before its crossing.
    "a-square-in-a-law-the-switch-closes": (
        HEAD + "Jy: -> Y; piecewise(0, S >= 0.5*thr, k)*piecewise(1, time^2 >= 9, 0.5)\n"
    ),
}


@pytest.mark.parametrize("case", sorted(GATES_ON_A_POLYNOMIAL))
def test_a_gate_on_a_polynomial_in_time_is_refused_on_the_instant(case):
    """``time^2 >= 9`` switches at t = 3, and the run stops a few ulp past it
    (issue #714): a fixed crossing like any other, with the stop a little
    after where the condition flips."""
    _refused(GATES_ON_A_POLYNOMIAL[case], ["thr"], 946)


def test_a_gate_on_a_polynomial_in_a_counter_is_refused_on_the_instant(tmp_path):
    """The same on a counter species, ``if(t^2 >= 9, k, 0)``:
    dX/dthr = −0.5 for 0."""
    path = tmp_path / "m.net"
    path.write_text(NET.replace("if(t>=thr,k,0)", "if(t^2>=9,k,0)").format(level="0.5*thr"))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["thr"]
    )
    with pytest.raises(bngsim.SimulationError, match="#946"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12, timeout=60)


def test_a_gate_that_closes_the_window_on_the_instant_is_refused():
    """``piecewise(k, S >= 0.5*thr, 0)*piecewise(0, time >= 3, 1)``: the gate
    closes the window on the instant the switch would open it. With thr under
    3 the window is open for 3 − thr, and dY/dthr is −0.5; above 3 it never
    opens. The law is 0 after the crossing either way, so the switch reads as
    one that does not jump, and nothing was asked: the run returned 0."""
    text = HEAD + "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)*piecewise(0, time >= 3, 1)\n"
    _refused(text, ["thr"], 946)


def test_an_event_that_empties_what_the_law_reads_is_refused():
    """The event sets X to 0 on the instant V reaches 6: the switched law
    q·X is 0 past it and the switch reads as one that does not jump. dW/dtau
    is −6.3 with the switch first and −2.1 with the event first, and the run
    returned −6.3."""
    _refused(EVENT.replace("X = 0.5*X", "X = 0").format(at=6), ["tau"], 945, EVENT_TIMES)


def test_a_gate_that_opens_inside_the_reach_is_refused():
    """S = e^t reaches its threshold 1.86e-5 after the gate at 3 opens the
    law, and at rtol 1e-6 the run finds that crossing before the gate, where
    the law is 0: dY/dsth came back 0 for −0.0249."""
    text = (
        f"species S, Y; S = 1; Y = 0; k = 0.5; sth = {math.exp(3 + 1.86e-5)!r}\nJs: -> S; S\n"
        "Jy: -> Y; piecewise(k, S >= sth, 0)*piecewise(1, time >= 3, 0)\n"
    )
    _refused(text, ["sth"], 946, rtol=1e-6, atol=1e-9)


def test_a_gate_that_closes_inside_the_reach_is_refused():
    """The same pair with a gate that closes the law at 3, 1.86e-5 before S
    reaches its threshold: the window never opens and dY/dsth is 0. The run
    found the crossing before the gate, and dY/dsth came back −0.0249."""
    text = (
        f"species S, Y; S = 1; Y = 0; k = 0.5; sth = {math.exp(3 + 1.86e-5)!r}\nJs: -> S; S\n"
        "Jy: -> Y; piecewise(k, S >= sth, 0)*piecewise(0, time >= 3, 1)\n"
    )
    _refused(text, ["sth"], 946, rtol=1e-6, atol=1e-9)


def test_a_crossing_found_after_the_gate_that_closed_its_law_is_refused():
    """Refused here, where main is right. S is linear and crosses 1e-7 after
    the gate closes the law, inside the 3e-6 the crossing is known to at rtol
    1e-6. The law is 0 on both sides of the crossing as the run finds it, and
    the jump shows only with the time put before the gate. dY/dthr is
    −0.5 | 0 between the two orders. At rtol 1e-10 the pair runs."""
    text = HEAD + "Jy: -> Y; piecewise(k, S >= 0.5*thr + 5e-8, 0)*piecewise(0, time >= 3, 1)\n"
    _refused(text, ["thr"], 946, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(_columns(text, ["thr"])[:, 0], [0.0, 0.0, 0.0], atol=1e-9)


COUNTER = "species S, X, Y, Tc; S = 0; X = 0; Y = 0; Tc = 0; k = 0.5; t2 = 3\nJs: -> S; 0.5\n"


def test_a_counter_moved_by_its_own_initial_value_is_refused():
    """The gate is on a counter at the literal 3 and the switch is fixed, with
    the counter's initial value requested: the gate moves with its own
    column and no record says so. dY/dTc(0) is 0.25 | 0, and the run
    returned 0."""
    text = COUNTER + (
        "Jc: -> Tc; 1\nJy: -> Y; piecewise(k, S >= 1.5, 0)*piecewise(1, Tc >= 3, 0.5)\n"
    )
    _refused(text, None, 946, ic=["Tc"])


def test_a_counter_moved_by_its_own_rate_is_refused():
    """The same with the counter's rate requested and the switch, on S = e^t,
    3.17e-5 after the gate: inside the reach at rtol 1e-6. dY/d(one) came
    back 0.75 for 0."""
    text = (
        "species S, Y, Tc; S = 1; Y = 0; Tc = 0; k = 0.5; one = 1\nJs: -> S; S\nJc: -> Tc; one\n"
        f"Jy: -> Y; piecewise(k, S >= {math.exp(3 + 3.17e-5)!r}, 0)*piecewise(1, Tc >= 3, 0.5)\n"
    )
    _refused(text, ["one"], 946, rtol=1e-6, atol=1e-9)


def test_an_event_between_the_two_does_not_hide_the_first():
    """The event on U = e^(c·t) is 5e-7 after the switch on V, inside the
    reach at rtol 1e-6, and an event that does nothing fires on V between
    them. Only the last fire was remembered: dW/dc came back 9.45 for 3.15."""
    text = (
        "species X, W, V, U; X = 0; W = 0; V = 0; U = 1; a = 2; q = 0.7; c = 1\n"
        "J0: -> X; a\nJv: -> V; 2\nJu: -> U; c*U\nJ2: -> W; piecewise(q*X, V >= 6, 0)\n"
        f"E1: at (U > {math.exp(3 + 5e-7)!r}): X = 0.5*X\nE2: at (V >= 5.9999998): W = W + 0\n"
    )
    _refused(text, ["c"], 945, EVENT_TIMES, rtol=1e-6, atol=1e-9)


def test_a_switch_between_the_two_does_not_hide_the_first():
    """The switch on V = e^(2t) is 1.99e-5 after the event at tau, found
    before it at rtol 1e-6, and a second switch that jumps, on Z through 0,
    crosses between them. Only the last crossing was remembered: dW/dtau came
    back −1.05 for −3.15."""
    text = (
        "species X, W, V, Z, Q; X = 0; W = 0; V = 1; Z = 5.9999996; Q = 0; a = 2; q = 0.7;"
        " tau = 3; kv = 2\n"
        "J0: -> X; a\nJv: -> V; kv*V\nJz: Z -> ; 2\nJq: -> Q; piecewise(1, Z <= 0, 0)\n"
        f"J2: -> W; piecewise(q*X, V >= {math.exp(6 + 2 * 1.99e-5)!r}, 0)\n"
        "E1: at (time >= tau): X = 0.5*X\n"
    )
    _refused(text, ["tau"], 945, EVENT_TIMES, rtol=1e-6, atol=1e-9)


def test_a_trigger_that_reads_the_switch_s_species_and_another_is_refused():
    """``at (V > U·e³)`` reads V, which the switch reads, and U. It fires
    where U says, not where V is on its way through 6: the two are not on one
    trajectory. dW/dc came back −9.45 for −3.15."""
    text = (
        "species X, W, V, U; X = 0; W = 0; V = 1; U = 1; a = 2; q = 0.7; c = 1; kv = 2\n"
        "J0: -> X; a\nJv: -> V; kv*V\nJu: -> U; c*U\n"
        f"J2: -> W; piecewise(q*X, V >= {math.exp(6 - 2 * 2.4329e-5)!r}, 0)\n"
        f"E1: at (V > U*{math.exp(3.0)!r}): X = 0.5*X\n"
    )
    _refused(text, ["c"], 945, EVENT_TIMES, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    ("rate", "column", "want"),
    [("50", 1, 0.0), ("piecewise(0.005, S >= 50*thr, 50)", 0, 49.995)],
    ids=["a-steady-rate", "a-rate-the-switch-slows"],
)
def test_the_reach_goes_by_the_rate_the_species_arrives_at(rate, column, want):
    """Control. S comes to its threshold at a rate of 50, with a gate in
    another law 0.02 later: 7,000 reaches away at rtol 1e-6. A switch that
    slows S to 0.005 past the crossing says nothing of how fast it got there;
    read from the flow after the crossing the reach was 0.03, and the run was
    refused."""
    law = "" if "piecewise" in rate else "Jy: -> Y; piecewise(k, S >= 50*thr, 0)\n"
    text = (
        "species S, X, Y; S = 0; X = 0; Y = 0; k = 0.5; thr = 3\n"
        f"Js: -> S; {rate}\n{law}Jx: -> X; piecewise(k, time >= 3.02, 0)\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr"])
    run = sim.run(sample_times=TIMES, rtol=1e-6, atol=1e-9, timeout=60)
    assert np.asarray(run.sensitivities)[-1][column, 0] == pytest.approx(want, rel=1e-5, abs=1e-8)


def test_a_switch_that_speeds_its_own_species_is_within_the_reach_it_arrived_with():
    """S comes to its threshold at 0.005 and leaves at 50, with a gate in the
    same law 1e-7 later: inside the 3e-6 that rtol 1e-6 allows at the rate of
    arrival. Read from the flow after the crossing the reach was 3e-10, and
    the pair ran. (S is linear, so the order it ran in was the true one.)"""
    text = (
        "species S, Y; S = 0; Y = 0; k = 0.5; thr = 3\n"
        "Js: -> S; piecewise(50, S >= 0.005*thr, 0.005)\n"
        "Jy: -> Y; piecewise(k, S >= 0.005*thr, 0)*piecewise(1, time >= 3.0000001, 0.5)\n"
    )
    _refused(text, ["thr"], 946, rtol=1e-6, atol=1e-9)


def test_a_slow_crossing_is_within_reach_as_far_as_the_tolerances_leave_its_time_open():
    """Refused here, where main is right. S comes to its threshold of 10 at
    0.001, and at rtol 1e-6 it is known there to 1e-5: its crossing time, 3,
    to 0.01. A gate in the same law 0.001 later is inside that, and dY/dthr
    is −250 with the switch first and −500 with the gate first. (S is linear,
    so the order the run found was the true one.) At rtol 1e-10 the crossing
    is known to 1e-6 and the pair runs."""
    text = (
        "species S, Y; S = 9.997; Y = 0; k = 0.5; thr = 10\nJs: -> S; 0.001\n"
        "Jy: -> Y; piecewise(k, S >= thr, 0)*piecewise(1, time >= 3.001, 0.5)\n"
    )
    _refused(text, ["thr"], 946, rtol=1e-6, atol=1e-8)
    assert _columns(text, ["thr"])[1, 0] == pytest.approx(-250.0, rel=1e-6)


SPEEDS = (
    "species S, Y, Tc; S = 0; Y = 0; Tc = 0; k = 0.5; tg = 3.0000001; one = 1\n"
    "Js: -> S; piecewise(50, S >= 0.015, 0.005)\nJc: -> Tc; one\n"
    "Jy: -> Y; piecewise(k, S >= 0.015, 0)*piecewise(1, {gate}, 0.5)\n"
)


@pytest.mark.parametrize(
    ("gate", "param", "want"),
    [("time >= tg", "tg", -0.25), ("Tc >= 3.0000001", "one", 0.75)],
    ids=["a-fitted-gate", "a-counter-s-gate"],
)
def test_a_moved_gate_within_the_reach_a_fixed_switch_arrived_with_is_refused(gate, param, want):
    """Refused here, where main is right. The switch is fixed, at t = 3, and
    the gate in the same law is 1e-7 after it and moved by the one column
    requested: its fitted time, or the rate of the counter it is on. At the
    gate S is 5e-6 past its threshold, far outside what the tolerances allow
    it, so nothing is asked from the clock's side; the crossing's time is
    known to the 3e-6 it arrived with. dY/dtg is 0 | −0.25. At rtol 1e-10
    the pair runs."""
    _refused(SPEEDS.format(gate=gate), [param], 946, rtol=1e-6, atol=1e-9)
    got = _columns(SPEEDS.format(gate=gate), [param])
    assert got[1, 0] == pytest.approx(want, rel=1e-6)


def test_a_fitted_gate_on_a_fixed_switch_in_another_law_is_refused():
    """Refused here, where main is right. The gate moves and the switch does
    not, on one instant, in different rate laws: they commute, and each has
    its own jump. Whether the two are composed is not asked."""
    text = HEAD + ("Jx: -> X; piecewise(k, time >= thr, 0)\nJy: -> Y; piecewise(k, S >= 1.5, 0)\n")
    _refused(text, ["thr"], 946)


@pytest.mark.parametrize(
    "gate", ["", "Jx: -> X; piecewise(k, time >= 2, 0)\n"], ids=["alone", "a-gate"]
)
def test_a_switch_on_a_species_in_a_fast_exchange_runs(gate):
    """Control. S and P exchange at 1e9 and S crosses its threshold at t = 6,
    with a gate in another law at 2 or none. Put a probe step to one side
    while P is put to the other, S is off the exchange's balance, and the
    right-hand side changes by 2e-4 of itself for smooth reasons. That
    change halves with the step, and is not taken for a jump.
    dY/dthr = −k·dt*/dthr = −1."""
    text = (
        "species S, P, X, Y; S = 0; P = 0; X = 0; Y = 0; k = 0.5; thr = 3; kf = 1e9\n"
        "Js: -> S; 0.5\nJf: S -> P; kf*S\nJr: P -> S; kf*P\n"
        "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)\n" + gate
    )
    got = _columns(text, ["thr"], [0.0, 2.0, 5.0, 7.0, 9.0])
    assert got[3, 0] == pytest.approx(-1.0, rel=1e-5)


STARTED = (
    "species X, Y; X = 2; Y = 0; r = 1.5; k = 0.5; tau = 3; X0 = 2\n"
    "Jx: -> X; piecewise(r, time >= tau, 0)\nJy: -> Y; piecewise({law}, X > X0, 0)\n"
)


def test_a_bend_that_a_fitted_gate_starts_runs():
    """Control. X sits on X0 until the gate at the fitted tau starts it, and
    the law is a ramp from X0: a bend, 0 at its surface whichever of the two
    comes first, a few ulp after the gate where the run finds it.
    Y = k·r·(T − tau)²/2. (The stimulus of BIOMD0000000161, which a cut that
    asked at every crossing refused.)"""
    got = _columns(STARTED.format(law="k*(X - X0)"), ["tau", "r"])
    np.testing.assert_allclose(got[1], [-0.5 * 1.5 * 2.0, 0.5 * 2.0**2 / 2.0], rtol=1e-7)


def test_a_jump_that_a_fitted_gate_starts_is_refused():
    """The same with a law that jumps where X leaves X0, ``k`` from there:
    Y = k·(T − tau). X's residual is exactly 0 until the gate, and leaves 0
    without coming through it: no root is reported, the state switch's jump
    was never made, and dY/dtau came back 0 for −0.5."""
    _refused(STARTED.format(law="k"), ["tau", "r"], 946)


def test_a_jump_that_a_fitted_gate_starts_runs_for_what_does_not_move_the_gate():
    """Control. The same with only r requested: the gate does not move, the
    crossing is on it whatever r is, and dY/dr = 0."""
    got = _columns(STARTED.format(law="k"), ["r"])
    np.testing.assert_allclose(got[:, 0], [2.0 * 1.0, 0.0], atol=1e-9)


COUNTER_STARTED = (
    "species X, Y, Tc; X = 2; Y = 0; Tc = 0; r = 1.5; k = 0.5; X0 = 2; one = 1\n"
    "Jc: -> Tc; one\nJx: -> X; piecewise(r, Tc >= 3, 0)\nJy: -> Y; piecewise(k, X > X0, 0)\n"
)


def test_a_jump_that_a_counter_s_gate_starts_is_refused():
    """The gate is on a counter whose rate is requested, at t = 3/one:
    Y = k·(T − 3/one) and dY/d(one) = 1.5. It came back 0."""
    _refused(COUNTER_STARTED, ["one"], 946)


def test_a_jump_that_a_counter_s_gate_starts_runs_where_nothing_moves_the_counter():
    """Control. With k and r requested the gate is at 3 whatever they are."""
    got = _columns(COUNTER_STARTED, ["k", "r"])
    np.testing.assert_allclose(got, [[0.0, 2.0], [2.0, 0.0], [0.0, 0.0]], atol=1e-9)


NESTED = """begin parameters
    1 k 0.5
    2 a 1
    3 b 10
    4 t1 3
    5 one 1
end parameters
begin functions
    1 sched() if(t<t1,a,b)
    2 fX() if(t>=sched(),k,0)
end functions
begin species
    1 X() 0
    2 Tc() 0
end species
begin reactions
    1 0 1 fX
    2 0 2 one
end reactions
begin groups
    1 t 2
end groups
"""
NESTED_TIMES = [0.0, 2.0, 5.0, 8.0, 12.0]


def _nested(tmp_path, params, law="k"):
    path = tmp_path / "nested.net"
    assert "if(t>=sched(),k,0)" in NESTED
    path.write_text(NESTED.replace("if(t>=sched(),k,0)", f"if(t>=sched(),{law},0)"))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=params
    )
    return sim.run(sample_times=NESTED_TIMES, rtol=1e-10, atol=1e-12, timeout=60)


def test_a_threshold_that_is_itself_a_condition_on_the_clock_is_refused(tmp_path):
    """``t >= if(t < t1, a, b)``: on from a to t1 and again from b, so
    X = k·((t1 − a) + (T − b)) and dX/dt1 = k. At t1 the threshold jumps from
    a to b and the residual jumps across 0 with it: the clock's own record
    makes the law's jump there, and the state switch made it again.
    dX/dt1 came back 1 for 0.5."""
    with pytest.raises(bngsim.SimulationError, match="#946"):
        _nested(tmp_path, ["t1"])


def test_a_threshold_that_is_a_condition_runs_where_no_column_moves_a_crossing(tmp_path):
    """Control. With k requested nothing moves any of the three crossings."""
    got = np.asarray(_nested(tmp_path, ["k"]).sensitivities)[-1]
    assert got[0, 0] == pytest.approx(4.0, rel=1e-7)


def test_a_threshold_that_is_a_condition_runs_under_a_law_that_is_continuous_there(tmp_path):
    """Control. The same threshold under ``k·(t1 − t)²``, which is 0 where
    the threshold jumps: the residual jumps across 0 and the law does not,
    so there is no jump to make twice.
    dX/dt1 = k·((t1 − a)² + (b − t1)² − (T − t1)²) = −14."""
    got = np.asarray(_nested(tmp_path, ["t1"], law="k*(t1-t)^2").sensitivities)[-1]
    assert got[0, 0] == pytest.approx(-14.0, rel=1e-7)


def test_a_column_that_moves_the_crossing_by_less_than_the_reach_does_not_move_it():
    """Control. S is made at 0.5 + 1e-14·q and crosses 1.5 at 3 − 6e-14,
    with a fixed gate in another law at 3. q moves the crossing by 6e-14 for
    the whole of itself, far inside the 3e-10 the crossing's time is known
    to: nothing is on one side of anything. dS/dq = 5e-14 and the rest are 0
    to 1e-12."""
    text = (
        "species S, X, Y; S = 0; X = 0; Y = 0; k = 0.5; q = 1\n"
        "Js: -> S; 0.5 + 1e-14*q\nJx: -> X; piecewise(k, time >= 3, 0)\n"
        "Jy: -> Y; piecewise(k, S >= 1.5, 0)\n"
    )
    np.testing.assert_allclose(_columns(text, ["q"])[:, 0], [5e-14, 0.0, 0.0], atol=1e-12)


SET = (
    "species X, Y, W, V; X = {x0}; Y = 0; W = 0; V = 0; k = 0.5; q = 0.7; tau = 3\n"
    "Jv: -> V; 2\nJ1: -> Y; k\nJ2: -> W; piecewise({law}, V >= {at}, 0)\n"
    "E1: at (time >= tau): X = {to}\n"
)


@pytest.mark.parametrize(
    "off", [-4e-10, 0.0, 4e-10], ids=["the-switch-first", "on", "the-event-first"]
)
def test_an_event_that_fills_what_the_law_reads_is_refused(off):
    """X is 0 until the event sets it to 5, and the switched law is q·X: with
    the switch first it crosses where the law is 0 on both sides and reads as
    continuous, and the jump shows once the event has fired. dW/dtau is 0
    with the switch first and −3.5 with the event first."""
    text = SET.format(x0=0, law="q*X", at=repr(6.0 + off), to="5")
    _refused(text, ["tau"], 945, EVENT_TIMES)


@pytest.mark.parametrize("off", [-4e-10, 4e-10], ids=["the-switch-first", "the-event-first"])
def test_a_bend_beside_an_event_runs(off):
    """Control. The switched law is a ramp from the threshold, q·(V − at)·X,
    and the event doubles X within the reach of the crossing: 0 at the
    surface before the event and after it. dW/dtau is 0 to 1e-8."""
    at = 6.0 + off
    text = SET.format(x0=1, law=f"q*(V - {at!r})*X", at=repr(at), to="2*X")
    got = _columns(text, ["tau"], EVENT_TIMES)
    assert abs(got[2, 0]) < 1e-8


# ── From the second review ──────────────────────────────────────────────────

BYSTANDER = "species B; B = 0; Jb: -> B; {rate}\n"
HIDDEN_BY_A_BYSTANDER = {
    # The window closes on the switch's instant: 0 on the kink −0.5 | 0.
    "a-gate-that-closes-the-window": (
        HEAD + "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)*piecewise(0, time >= 3, 1)\n",
        "thr",
        946,
        TIMES,
        "1e6",
    ),
    # A fitted gate starts X off the X0 it sat on: dY/dtau = 0 for −0.5.
    "a-jump-that-a-fitted-gate-starts": (STARTED.format(law="k"), "tau", 946, TIMES, "1e6"),
    # An event empties what the law reads: −6.3 on the kink −6.3 | −2.1.
    "an-event-that-empties-what-the-law-reads": (
        EVENT.replace("X = 0.5*X", "X = 0").format(at=6),
        "tau",
        945,
        EVENT_TIMES,
        "1e7",
    ),
}


@pytest.mark.parametrize("case", sorted(HIDDEN_BY_A_BYSTANDER))
def test_a_bystander_made_fast_does_not_hide_a_jump(case):
    """A species the switch never touches, made at 1e6, set the scale every
    jump was read against: a millionth of the largest net rate in the model.
    A jump of 0.5 read as none, and each of these ran. A jump is read
    against the rate that drives the crossing, as the crossing's own is."""
    text, param, issue, times, rate = HIDDEN_BY_A_BYSTANDER[case]
    _refused(text + BYSTANDER.format(rate=rate), [param], issue, times)


EVENT_STARTS = {
    # dY/dtau = 0 for −0.5.
    "sets-the-rate": (
        "species X, Y; X = 2; Y = 0; r = 0; k = 0.5; tau = 3; X0 = 2\nJx: -> X; r\n"
        "Jy: -> Y; piecewise(k, X > X0, 0)\nE1: at (time >= tau): r = 1.5\n",
        "tau",
    ),
    # The trigger is on the state, U = c·t > 3: dY/dc = 0 for 1.5.
    "on-a-state-trigger": (
        "species X, Y, U; X = 2; Y = 0; U = 0; r = 0; k = 0.5; c = 1; X0 = 2\nJu: -> U; c\n"
        "Jx: -> X; r\nJy: -> Y; piecewise(k, X > X0, 0)\nE1: at (U > 3): r = 1.5\n",
        "c",
    ),
    # The event puts a rising X on X0 exactly: dY/dtau = 0 for −0.5.
    "puts-the-species-on-its-threshold": (
        "species X, Y; X = -10; Y = 0; k = 0.5; tau = 3; X0 = 2\n"
        "Jx: -> X; piecewise(1.5, time >= 2.9, 0)\nJy: -> Y; piecewise(k, X > X0, 0)\n"
        "E1: at (time >= tau): X = X0\n",
        "tau",
    ),
}


@pytest.mark.parametrize("case", sorted(EVENT_STARTS))
def test_an_event_that_starts_the_state_off_its_surface_is_refused(case):
    """As a fitted gate does: the state is on the switch's surface in the
    state the event leaves, and leaves it without coming through 0. No root
    is reported and the switch's jump was never made."""
    text, param = EVENT_STARTS[case]
    _refused(text, [param], 945)


EVENT_SETS_THE_RATE = (
    "species X, Y; X = 2; Y = 0; r = 0; k = 0.5; tau = 3; X0 = 2\nJx: {made}; r\n"
    "Jy: -> Y; piecewise({law}, X > X0, 0)\nE1: at (time >= tau): r = 1.5\n"
)


@pytest.mark.parametrize(
    ("made", "law", "param", "want"),
    [
        ("-> X", "k", "k", [0.0, 2.0]),  # nothing moves the event
        ("-> X", "k*(X - X0)", "tau", [-1.5, -1.5]),  # a bend: Y = k·r·(T − tau)²/2
        ("X ->", "k", "tau", [1.5, 0.0]),  # pushed away from X > X0: the law stays off
    ],
    ids=["a-column-that-does-not-move-it", "a-bend", "pushed-away"],
)
def test_an_event_that_starts_the_state_runs_where_no_jump_is_missed(made, law, param, want):
    """Control."""
    got = _columns(EVENT_SETS_THE_RATE.format(made=made, law=law), [param])
    np.testing.assert_allclose(got[:, 0], want, rtol=1e-7, atol=1e-9)


READS_THE_TIME = "species S, X, Y; S = 0; X = 0; Y = 0; k = 0.5; thr = 4.5\nJs: -> S; 0.5\n"


@pytest.mark.parametrize(
    "text",
    [
        READS_THE_TIME + "Jy: -> Y; piecewise(k, S + time >= thr, 0)*piecewise(0, time >= 3, 1)\n",
        "species S, Y, Tc; S = 0; Y = 0; Tc = 0; k = 0.5; thr = 4.5\nJs: -> S; 0.5\n"
        "Jc: -> Tc; 1\nJy: -> Y; piecewise(k, S + Tc >= thr, 0)*piecewise(0, Tc >= 3, 1)\n",
    ],
    ids=["the-time", "a-counter"],
)
def test_a_switch_that_reads_the_clock_as_well_is_refused_where_a_gate_closes_its_window(text):
    """``S + time >= thr`` crosses at t = 3, where the gate closes the law:
    0 on the kink −1/3 | 0. The surface moves with the clock, so with the
    gate put on its other side the species were a step either side of
    nothing, and the hidden jump was not seen. They are put as far as it
    takes to cross the surface there; and on a counter the switch reads, the
    surface is crossed by the other species."""
    _refused(text, ["thr"], 946)


def test_a_switch_that_reads_the_time_as_well_runs_alone_and_apart_from_a_gate():
    """Control. S + t reaches thr at t = thr/1.5: dY/dthr = −k/1.5, and
    dY/dk = T − 3. The same with a gate that closes the law at 3.4."""
    alone = READS_THE_TIME + "Jy: -> Y; piecewise(k, S + time >= thr, 0)\n"
    np.testing.assert_allclose(_columns(alone, ["thr", "k"])[2], [-1 / 3, 2.0], rtol=1e-7)
    apart = READS_THE_TIME + (
        "Jy: -> Y; piecewise(k, S + time >= thr, 0)*piecewise(0, time >= 3.4, 1)\n"
    )
    assert _columns(apart, ["thr"])[2, 0] == pytest.approx(-1 / 3, rel=1e-7)


def _two_clocks(opens, other):
    return (
        "species S, Y, X, Tc, Tb; S = 1; Y = 0; X = 0; Tc = 0; Tb = 0; k = 0.5; "
        f"sth = {math.exp(3 + 2.95e-5)!r}\nJs: -> S; S\nJc: -> Tc; 1\nJb: -> Tb; 1\n"
        f"Jy: -> Y; piecewise(k, S >= sth, 0)*piecewise(1, {opens}, 0)\n"
        f"Jx: -> X; piecewise(k, {other}, 0)\n"
    )


@pytest.mark.parametrize(
    ("opens", "other"),
    [("time >= 3", "Tc >= 3"), ("Tc >= 3", "Tb >= 3")],
    ids=["the-time-behind-a-counter", "a-counter-behind-another"],
)
def test_two_clocks_on_one_instant_are_both_put_on_their_other_side(opens, other):
    """The gate that opens the switched law and a gate in another law switch
    on one instant, and are one stop: the counter's, with the time's behind
    it. Only the kept one was put on its other side, and the hidden jump was
    not seen: dY/dsth = 0 for −0.0249. The clock is put across whole, the
    time and every counter with it."""
    _refused(_two_clocks(opens, other), ["sth"], 946, rtol=1e-6, atol=1e-9)


def test_a_trigger_that_reads_the_switch_s_species_and_the_time_is_refused():
    """``at (V + 1e5*time > e^6 + 1e5*tau)`` reads V, the one species the
    switch reads, and crosses where the time says: it is on no species'
    trajectory. dW/dtau came back −1.0416 for −3.1248."""
    text = (
        "species X, W, V; X = 0; W = 0; V = 1; a = 2; q = 0.7; tau = 3; kv = 2\n"
        "J0: -> X; a\nJv: -> V; kv*V\n"
        f"J2: -> W; piecewise(q*X, V >= {math.exp(6 + 6.16e-5)!r}, 0)\n"
        f"E1: at (V + 1e5*time > {math.exp(6.0)!r} + 1e5*tau): X = 0.5*X\n"
    )
    _refused(text, ["tau"], 945, EVENT_TIMES, rtol=1e-6, atol=1e-9)


FITTED_GATE = (
    "species X, Y, Z; X = 0; Y = 0; Z = 0; r = 1.5; k = 0.5; tau = 3\n"
    "Jx: -> X; piecewise(r, time >= tau, 0)\n"
)
ON_A_SURFACE_AND_STAYING = {
    # A guard on a species that is not there.
    "a-guard-on-an-absent-species": (
        FITTED_GATE + "Jy: -> Y; piecewise(k, Z > 0, 0)\nJz: Z -> ; 0.1*Z\n",
        TIMES,
        1e-10,
        1e-12,
        [-1.5, 0.0, 0.0],
    ),
    "a-division-guard": (
        FITTED_GATE + "Jy: -> Y; piecewise(k/Z, Z > 0, 0)\n",
        TIMES,
        1e-10,
        1e-12,
        [-1.5, 0.0, 0.0],
    ),
    # X is on X0 and the gate pushes it down, away from X > X0.
    "pushed-away": (
        "species X, Y; X = 2; Y = 0; r = 1.5; k = 0.5; tau = 3; X0 = 2\n"
        "Jx: X -> ; piecewise(r, time >= tau, 0)\nJy: -> Y; piecewise(k, X > X0, 0)\n",
        TIMES,
        1e-10,
        1e-12,
        [1.5, 0.0],
    ),
    # B = e^(−5t) is long under its guard, and under the absolute tolerance.
    "a-guard-on-a-species-that-has-decayed": (
        "species B, X, Y; B = 1; X = 0; Y = 0; lam = 5; k = 0.5; r = 1.5; tau = 8\n"
        "Jb: B -> ; lam*B\nJy: -> Y; piecewise(k, B > 1e-9, 0)\n"
        "Jx: -> X; piecewise(r, time >= tau, 0)\n",
        [0.0, 2.0, 4.0, 8.0, 12.0],
        1e-8,
        1e-8,
        [0.0, -1.5, 0.0],
    ),
}


@pytest.mark.parametrize("case", sorted(ON_A_SURFACE_AND_STAYING))
def test_a_fitted_gate_beside_a_state_that_stays_on_its_surface_runs(case):
    """Control. A fitted gate, and a switch whose residual is 0 there, or
    under the tolerance, and is not started off it: nothing flips. An
    earlier cut asked of every residual the tolerances allow to be 0, and
    refused each of these."""
    text, times, rtol, atol, want = ON_A_SURFACE_AND_STAYING[case]
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"]).run(
        sample_times=times, rtol=rtol, atol=atol, timeout=60
    )
    got = np.asarray(run.sensitivities)[-1][:, 0]
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)


def _nested_law(tmp_path, law, params):
    path = tmp_path / "nested.net"
    assert "if(t>=sched(),k,0)" in NESTED
    path.write_text(NESTED.replace("if(t>=sched(),k,0)", law))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=params
    )
    return sim.run(sample_times=NESTED_TIMES, rtol=1e-10, atol=1e-12, timeout=60)


@pytest.mark.parametrize("at", ["1", "1.00000000002"], ids=["on-it", "beside-it"])
def test_a_fixed_gate_on_the_counter_at_the_crossing_is_refused(tmp_path, at):
    """A fixed gate in the same law on the value the counter has at the
    crossing, or 2e-11 past it: one stop for the two. dX/da came back −0.5
    on the kink −0.25 | −0.5 with the gate on it."""
    with pytest.raises(bngsim.SimulationError, match="#946"):
        _nested_law(tmp_path, f"if(t>=sched(),k,0)*if(t>={at},1,0.5)", ["a"])


def test_a_fitted_gate_on_the_crossing_s_own_threshold_is_refused(tmp_path):
    """``if(t >= a, 1, 0.5)`` beside ``t >= if(t < t1, a, b)``: a switch time
    fitted to a, on the crossing's own surface. Its record makes the whole
    jump there, both conditions flipping with the counter, and the state
    switch made it again: dX/da came back −1 for −0.5."""
    with pytest.raises(bngsim.SimulationError, match="#946"):
        _nested_law(tmp_path, "if(t>=sched(),k,0)*if(t>=a,1,0.5)", ["a"])


# A condition on a counter that the resolver places no stop for, `exp(t) >= e³`,
# is a state-dependent switch on the counter: a root, at t = 3.
EXP_GATE = "exp(t)>=20.085536923187668"
PROBE_STEP = 256 * 2.220446049250313e-16 * 3


def _two_roots(tmp_path, gate, law, level, params=("thr",)):
    path = tmp_path / "m.net"
    text = NET.replace("if(t>=thr,k,0)", gate).replace("if(Sobs>={level},k,0)", law)
    assert text != NET
    path.write_text(text.format(level=level))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=list(params)
    )
    return np.asarray(
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12, timeout=60).sensitivities
    )[-1]


def test_two_switches_on_one_instant_that_hide_each_other_s_jump_are_refused(tmp_path):
    """``if(Sobs >= 0.5*thr, k, 0)*if(exp(t) >= e³, 0, 1)``: the law is 0
    before the two and 0 after them, and k between them where S comes first.
    Read across both it is continuous, and dY/dthr came back 0 on the kink
    −0.5 | 0."""
    with pytest.raises(bngsim.SimulationError, match="#946"):
        _two_roots(tmp_path, "0", f"if(Sobs>={{level}},k,0)*if({EXP_GATE},0,1)", "0.5*thr")


def test_two_switches_well_apart_in_one_law_run(tmp_path):
    """Control. The same with S crossing eight probe steps before the gate
    closes the law: Y = k·(3 − t_c) and dY/dthr = −0.5."""
    level = f"0.5*thr-{4 * PROBE_STEP!r}"
    got = _two_roots(tmp_path, "0", f"if(Sobs>={{level}},k,0)*if({EXP_GATE},0,1)", level)
    assert got[2, 0] == pytest.approx(-0.5, rel=1e-6)


def test_two_switches_on_one_instant_run_where_no_column_moves_them(tmp_path):
    """Control. The same pair with k requested: the window is empty whatever
    k is."""
    law = f"if(Sobs>={{level}},k,0)*if({EXP_GATE},0,1)"
    got = _two_roots(tmp_path, "0", law, "0.5*thr", params=("k",))
    np.testing.assert_allclose(got[:, 0], [0.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_a_bend_that_reads_the_time_runs_beside_a_ramp_that_starts_on_its_instant():
    """Control. ``k·(S + time − thr)`` from where S + t reaches thr, at
    t = 3, and a ramp in another law that starts at the fixed 3. With the
    clock put on its other side the bend's surface is 4 probe steps off,
    and the species are put far enough to cross it there: nothing jumps.
    Y = 0.75·k·(T − thr/1.5)² and dY/dthr = −k·(T − 3)."""
    text = READS_THE_TIME + (
        "Jx: -> X; piecewise(k*(time - 3), time >= 3, 0)\n"
        "Jy: -> Y; piecewise(k*(S + time - thr), S + time >= thr, 0)\n"
    )
    np.testing.assert_allclose(_columns(text, ["thr"])[:, 0], [0.0, 0.0, -1.0], atol=1e-8)


def test_a_crossing_the_time_alone_carries_is_refused_where_a_gate_closes_its_window():
    """S stands at 1.5 and ``S + time >= thr`` crosses where the time says.
    There is no flow to put S across its surface by with the gate on its
    other side, and what cannot be asked is refused: 0 on the kink −0.5 | 0.
    Alone the switch runs, dY/dthr = −k."""
    head = "species S, Y; S = 1.5; Y = 0; k = 0.5; thr = 4.5\nJs: -> S; 0\n"
    alone = head + "Jy: -> Y; piecewise(k, S + time >= thr, 0)\n"
    assert _columns(alone, ["thr"])[1, 0] == pytest.approx(-0.5, rel=1e-7)
    closed = head + "Jy: -> Y; piecewise(k, S + time >= thr, 0)*piecewise(0, time >= 3, 1)\n"
    _refused(closed, ["thr"], 946)


def test_a_fixed_gate_between_the_probes_of_a_switch_that_reads_the_time_is_refused():
    """At rtol 1e-14 the crossing's time is known to 3e-14, and a fixed gate
    in another law 1e-13 before it is between the probes all the same: a
    stop is asked about two probe steps either way, whatever the
    tolerances. dX/dthr came back −1/3 for 0."""
    text = READS_THE_TIME + (
        "Jx: -> X; piecewise(k, time >= 3, 0)\n"
        "Jy: -> Y; piecewise(k, S + time >= thr + 1.5e-13, 0)\n"
    )
    _refused(text, ["thr"], 946, rtol=1e-14, atol=1e-16)


READS_V_AND_THE_TIME = (
    "species X, W, V; X = 0; W = 0; V = 0; a = 2; q = 0.7; kv = 2\n"
    "J0: -> X; a\nJv: -> V; kv\nJ2: -> W; piecewise(q*X, V + time >= 9, 0)\n"
    "E1: at (V > {at}): X = 0.5*X\n"
)


@pytest.mark.parametrize("off", [-4e-10, 4e-10], ids=["the-event-first", "the-switch-first"])
def test_a_switch_that_reads_the_time_is_on_no_trajectory_with_a_trigger_on_its_species(off):
    """The trigger reads V and the switch reads V and the time: one species
    between them, and two crossings that kv moves differently. dW/dkv is
    6.825 with the event first and 5.775 with the switch first."""
    _refused(READS_V_AND_THE_TIME.format(at=repr(6.0 + off)), ["kv"], 945, EVENT_TIMES)


def test_a_switch_that_reads_the_time_runs_apart_from_a_trigger_on_its_species():
    """Control. The event at V = 7, t = 3.5, half a unit after the switch."""
    got = _columns(READS_V_AND_THE_TIME.format(at="7"), ["kv"], EVENT_TIMES)
    np.testing.assert_allclose(got[:, 0], [1.75, 4.8125, 7.5], rtol=1e-7)


FILLED_AFTER = (
    "species X, W, V; X = 0; W = 0; V = 0; q = 0.7; tau = {tau}\n"
    "Jv: -> V; 2\nJ2: -> W; piecewise(q*X, V + time >= 9, 0)\nE1: at (time >= tau): X = 5\n"
)


def test_an_event_that_fills_what_a_switch_that_reads_the_time_reads_is_refused():
    """Refused here, where main is right. The crossing is continuous as it is
    found, q·X at an X of 0, and the event 1e-10 later fills X: dW/dtau is
    −3.5 | 0. The surface has moved with the time by then, and the crossing
    cannot be read again where it was."""
    _refused(FILLED_AFTER.format(tau="3.0000000001"), ["tau"], 945, EVENT_TIMES)
    got = _columns(FILLED_AFTER.format(tau="3.5"), ["tau"], EVENT_TIMES)
    np.testing.assert_allclose(got[:, 0], [0.0, -3.5, 0.0], atol=1e-8)


@pytest.mark.parametrize("param", ["a", "b"])
def test_a_threshold_that_is_a_condition_is_refused_for_a_column_that_moves_a_crossing(
    tmp_path, param
):
    """Refused here, where main is right (to 7e-7 with b). The condition is
    on a counter alone, so each crossing of it is a root here and, to the
    resolver that places stops (issue #714), a fixed crossing the run stops
    a few ulp past. With a requested the crossing at t = a is within reach
    of that stop, and whether the stop is this crossing seen again or
    another gate on the same value of the counter cannot be told by moving
    the counter, which flips both. With b requested the residual moves with
    the column where it jumps across 0, at t1."""
    with pytest.raises(bngsim.SimulationError, match="#946"):
        _nested(tmp_path, [param])


# ── From the third review ───────────────────────────────────────────────────

ON_ITS_THRESHOLD = (
    "species X, Y, Z; X = {x0}; Y = 0; Z = 0; r = {r}; k = 0.5; tau = 3; X0 = {x0}\n"
    "{made}Jy: -> Y; piecewise(k, X > X0, 0)\n{event}"
)
GATE_STARTS = "Jx: -> X; piecewise(r, time >= tau, 0)\n"
LEAVES_ITS_SURFACE = {
    # The gate starts Z, and X is made at Z: X leaves X0 at second order.
    "through-another-species": dict(
        x0=2, r=1.5, made="Jz: -> Z; piecewise(r, time >= tau, 0)\nJx: -> X; Z\n", event=""
    ),
    "on-a-ramp": dict(
        x0=2, r=1.5, made="Jx: -> X; piecewise(r*(time - tau), time >= tau, 0)\n", event=""
    ),
    "after-a-fast-follower": dict(
        x0=2,
        r=1.5,
        made="Jz: -> Z; piecewise(r, time >= tau, 0)\nJx: -> X; 1e6*(Z - X + 2)\n",
        event="",
    ),
    "through-another-species-at-an-event": dict(
        x0=2, r=0, made="Jz: -> Z; r\nJx: -> X; Z\n", event="E1: at (time >= tau): r = 1.5\n"
    ),
    # A step of the species along its flow that is under an ulp of 1e5.
    "from-a-large-value": dict(x0="1e5", r=1.5, made=GATE_STARTS, event=""),
    "from-a-large-value-at-an-event": dict(
        x0="1e5", r=0, made="Jx: -> X; r\n", event="E1: at (time >= tau): r = 1.5\n"
    ),
    # A jump of 0.5 beside the 1e6 the switch's own species is made at.
    "at-a-great-rate": dict(x0=2, r="1e6", made=GATE_STARTS, event=""),
}


@pytest.mark.parametrize("case", sorted(LEAVES_ITS_SURFACE))
def test_a_state_that_leaves_its_surface_at_a_moved_instant_is_refused(case):
    """X sits on X0 until the gate or the event at the fitted tau, and the
    law is k from where X > X0: dY/dtau = −0.5, and 0 came back. The way it
    leaves is read from the flow a moment on where the flow there is 0, the
    step is taken past the rounding of X itself, and the jump is read
    against Y's own rate."""
    spec = LEAVES_ITS_SURFACE[case]
    _refused(ON_ITS_THRESHOLD.format(**spec), ["tau"], 945 if spec["event"] else 946)


TOUCHES = (
    "species X, Y; X = 0; Y = 0; k = 0.5; tau = 3; X0 = 1.5\n"
    "Jx: -> X; piecewise({after}, time >= tau, 0.5)\nJy: -> Y; piecewise(k, X {cmp} X0, 0)\n"
)


@pytest.mark.parametrize(
    ("after", "cmp"), [("-0.5", ">"), ("0", ">=")], ids=["and-turns-back", "and-stops"]
)
def test_a_state_that_comes_to_its_surface_as_a_fitted_gate_turns_it_is_refused(after, cmp):
    """X rises to X0 at t = 3, where the fitted gate turns it back or stops
    it. With the gate later X is over X0 for a while: dY/dtau is 0 | 1 where
    it turns back, and Y itself jumps where it stops. No root is reported
    for a residual that comes to 0 and does not go through, and 0 came
    back."""
    _refused(TOUCHES.format(after=after, cmp=cmp), ["tau"], 946)


FAST_SPECIES = (
    "species S, X, Y; S = 0; X = 0; Y = 0; k = 0.5; thr = 3; tau = 3; v = 1e6\nJs: -> S; v\n"
    "Jy: -> Y; piecewise(k, S >= v*thr, 0)*piecewise(0, time >= {gate}, 1)\n"
)


@pytest.mark.parametrize(("gate", "param"), [("tau", "tau"), ("3", "thr")])
def test_a_jump_hidden_beside_a_species_made_fast_is_refused(gate, param):
    """The switch's own species is made at 1e6 and the hidden jump is 0.5:
    read against a millionth of the rate that drives the crossing it was no
    jump. 0 came back on the kinks 0 | 0.5 and −0.5 | 0."""
    _refused(FAST_SPECIES.format(gate=gate), [param], 946)


ON_ONE_SPECIES = (
    "species S, Y; S = 0; Y = 0; k = 0.5; thr = 3; zt = 1.5\nJs: -> S; 0.5\n"
    "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)*piecewise(0, S >= zt, 1)\n"
)


@pytest.mark.parametrize("param", ["thr", "zt"])
def test_two_switches_on_one_species_that_a_column_moves_apart_are_refused(param):
    """The window opens where S reaches 0.5·thr and closes where it reaches
    zt, on one instant: 0 came back on the kinks −0.5 | 0 and 0 | 1. Two
    conditions on one species cannot be asked apart by moving the species,
    so two that a column moves apart are refused."""
    _refused(ON_ONE_SPECIES, [param], 946)


def test_two_switches_on_one_species_run_where_no_column_moves_either():
    """Control. With k requested the window is empty whatever k is."""
    np.testing.assert_allclose(_columns(ON_ONE_SPECIES, ["k"])[:, 0], [0.0, 0.0], atol=1e-9)


RESET = (
    "species V, Y; V = 0.5; Y = 0; kv = 1; k = 0.5\nJv: -> V; kv\n"
    "Jy: -> Y; piecewise(k, V > 0, 0)\nE1: at (V > 1): V = 0\n"
)
RESET_TIMES = [0.0, 1.2, 2.6, 3.7, 4.9]


def test_an_event_that_resets_a_species_onto_its_guard_runs_for_a_column_that_moves_nothing():
    """Control. V is reset to 0 under ``piecewise(k, V > 0, 0)`` and leaves 0
    at once. With k requested no fire moves: dY/dk = T."""
    got = _columns(RESET, ["k"], RESET_TIMES)
    assert got[1, 0] == pytest.approx(4.9, rel=1e-6)


def test_an_event_that_resets_a_species_onto_its_guard_is_refused_for_a_column_that_moves_it():
    """With kv requested each fire moves, and the guard flips where V leaves
    the 0 the event put it on: dY/dkv = 0, and −6.25 came back."""
    _refused(RESET, ["kv"], 945, RESET_TIMES)


def test_a_switch_beside_a_term_that_rounds_as_a_staircase_runs():
    """Control. No gate and no event: a switch on S, and a law that
    differences two pools of 1e8. A probe step moves P by an ulp, the
    difference moves in steps that do not halve, and an earlier cut read
    that as another jump between the probes and refused the run."""
    text = (
        "species S, Y, P, Q, U; S = 0; Y = 0; P = 1e8; Q = 1e8; U = 0; k = 0.5; thr = 3\n"
        "Js: -> S; 0.5\nJp: -> P; 6.1e4\nJu: -> U; 100*(P - Q)\n"
        "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)\n"
    )
    assert _columns(text, ["thr"])[1, 0] == pytest.approx(-0.5, rel=1e-7)


def test_two_bends_in_different_rate_laws_on_one_instant_run():
    """Control. Two ramps, each from its own species' threshold, that start
    on one instant, with a column that moves one and not the other: no rate
    law reads both conditions, so neither has a jump of the other's to hide.
    Y = k·(T − thr)²/4 and dY/dthr = −k·(T − thr)/2."""
    text = (
        "species S, Z, X, Y; S = 0; Z = 0; X = 0; Y = 0; k = 0.5; thr = 3\n"
        "Js: -> S; 0.5\nJz: -> Z; 0.5\n"
        "Jx: -> X; piecewise(k*(Z - 1.5), Z >= 1.5, 0)\n"
        "Jy: -> Y; piecewise(k*(S - 0.5*thr), S >= 0.5*thr, 0)\n"
    )
    np.testing.assert_allclose(
        _columns(text, ["thr"])[:, 0], [0.0, 0.0, 0.0, -0.5], rtol=1e-7, atol=1e-9
    )


EMPTIED_FOR_GOOD = (
    "species X, W, V, U; X = 5; W = 0; V = 0; U = 0; q = 0.7; kv = 2; c = 0.5\n"
    "Jv: -> V; kv\nJu: -> U; c\nJ2: -> W; piecewise(q*X, V >= {at}, 0)\n"
    "E1: at (U > 1.5): X = 0\n"
)


@pytest.mark.parametrize("off", [4e-10, -4e-10], ids=["the-event-first", "the-switch-first"])
def test_an_event_no_column_moves_that_empties_what_the_law_reads_is_refused_beside_a_moved_switch(
    off,
):
    """The event fires where U = 0.5·t passes 1.5 and empties X for good; the
    switch is where V = kv·t reaches 6, with kv requested. W = 5·q·(3 − t_c)
    with the switch first and 0 with the event first: dW/dkv is 5.25 | 0.
    Found after the event the crossing is continuous, q·X at an X of 0, and
    the jump shows in the state as it was before the event."""
    _refused(EMPTIED_FOR_GOOD.format(at=repr(6.0 + off)), ["kv"], 945, EVENT_TIMES)


def test_an_event_that_empties_what_the_law_reads_runs_apart_from_the_switch():
    """Control. The switch at t = 2.5, half a unit before the event:
    dW/dkv = 5·q·5/kv²."""
    got = _columns(EMPTIED_FOR_GOOD.format(at="5"), ["kv"], EVENT_TIMES)
    np.testing.assert_allclose(got[:, 0], [0.0, 4.375, 7.5, 0.0], rtol=1e-7, atol=1e-9)


EVENT_ON_U_STARTS = (
    "species X, Y, U; X = {x}; Y = 0; U = 0; r = 0; k = 0.5; c = 1; X0 = 2\nJu: -> U; c\n"
    "Jx: -> X; r\nJy: -> Y; piecewise(k, X > X0, 0)\nE1: at (U > 3): r = 1.5\n"
)


def test_a_state_just_under_its_surface_that_a_moved_event_starts_is_refused():
    """Refused here, where main is right. X is 1e-11 under X0 and at rest
    until an event on U = c·t starts it: it crosses 7e-12 later, a root like
    any other, within reach of a fire that c moves. From well under X0 the
    pair runs: dY/dc = 1.5."""
    _refused(EVENT_ON_U_STARTS.format(x=repr(2 - 1e-11)), ["c"], 945)
    got = _columns(EVENT_ON_U_STARTS.format(x="1.7"), ["c"])
    np.testing.assert_allclose(got[:, 0], [4.5, 1.5, 5.0], rtol=1e-7)


@pytest.mark.parametrize(("after", "before"), [(0, 1), (1, 0)], ids=["closes", "opens"])
def test_a_state_that_touches_its_surface_as_the_gate_that_turns_it_gates_the_law_is_refused(
    after, before
):
    """X rises to X0 as the fitted gate turns it back, and the same gate
    closes the switched law, or opens it: the law jumps across X's surface
    on one side of the gate only. With the gate later X is over X0 for a
    while on either side of it. 0 came back on a kink."""
    text = (
        "species X, Y; X = 0; Y = 0; k = 0.5; tau = 3; X0 = 1.5\n"
        "Jx: -> X; piecewise(-0.5, time >= tau, 0.5)\n"
        f"Jy: -> Y; piecewise(k, X > X0, 0)*piecewise({after}, time >= tau, {before})\n"
    )
    _refused(text, ["tau"], 946)


def test_an_event_on_a_state_no_column_moves_runs_beside_a_switch_no_column_moves():
    """Control. The event fires where U = c·t passes 3, 2e-10 after the
    switch, and q is requested: no column moves either. A trigger on the
    state is moved where a column moves the state it reads, not wherever
    it reads one. dW/dq = 33.75."""
    text = EVENT_ON_STATE.replace("U > {at}", "U > 3.0000000002")
    got = _columns(text, ["q"], EVENT_TIMES)
    np.testing.assert_allclose(got[:, 0], [0.0, 0.0, 33.75, 0.0, 0.0], rtol=1e-7, atol=1e-9)


# ── From the fourth review ──────────────────────────────────────────────────

BESIDE_ANOTHER_SOURCE = {
    # A fitted gate starts X off X0, and Y has another source: dY/dtau = 0 for −0.5.
    "in-the-same-law": (
        STARTED.replace("piecewise({law}, X > X0, 0)", "1e6 + piecewise(k, X > X0, 0)"),
        "tau",
        946,
        TIMES,
    ),
    "in-another-reaction": (STARTED.format(law="k") + "Jb: -> Y; 1e6\n", "tau", 946, TIMES),
    # What is read is the flux of the reactions that read the switch.
    "in-another-reaction-at-1e10": (
        STARTED.format(law="k") + "Jb: -> Y; 1e10\n",
        "tau",
        946,
        TIMES,
    ),
    # The window closes on the switch's instant: 0 on the kink −0.5 | 0.
    "a-gate-that-closes-the-window": (
        HEAD
        + "Jb: -> Y; 1e6\n"
        + "Jy: -> Y; piecewise(k, S >= 0.5*thr, 0)*piecewise(0, time >= 3, 1)\n",
        "thr",
        946,
        TIMES,
    ),
    # An event empties what the law reads: −6.3 on the kink −6.3 | −2.1.
    "an-event-that-empties-what-the-law-reads": (
        EVENT.replace("X = 0.5*X", "X = 0").format(at=6) + "Jb: -> W; 1e7\n",
        "tau",
        945,
        EVENT_TIMES,
    ),
}


@pytest.mark.parametrize("case", sorted(BESIDE_ANOTHER_SOURCE))
def test_a_jump_in_a_species_that_has_another_source_is_seen(case):
    """The species the switched law makes is also made at 1e6, and the jump
    is 0.5: read against a millionth of the species' own rate it was no
    jump. A jump is a change above the rounding of the flux that going
    further from the surface does not grow, whatever is made beside it."""
    text, param, issue, times = BESIDE_ANOTHER_SOURCE[case]
    _refused(text, [param], issue, times)


TWO_SPECIES = (
    "species S, Z, Y; S = 0; Z = 0; Y = 0; k = 0.5; thr = 3; zt = 1.5\n"
    "Js: -> S; 0.5\nJz: -> Z; 0.5\n"
)


@pytest.mark.parametrize(
    ("law", "want"),
    [
        ("piecewise(S, S < 0.5*thr, 0.5*thr) + piecewise(Z, Z < zt, zt)", 1.0),
        ("piecewise(S - 0.5*thr, S >= 0.5*thr, 0) + piecewise(Z - 1.5, Z >= 1.5, 0)", -1.0),
        ("piecewise(S - 0.5*thr, S >= 0.5*thr, 0)*piecewise(0, Z >= 1.5, 1)", 0.0),
    ],
    ids=["two-clamps", "two-ramps", "a-ramp-under-a-gate-that-closes-on-its-onset"],
)
def test_two_bends_added_in_one_rate_law_on_one_instant_run(law, want):
    """Control. One rate law reads both conditions and thr moves one of
    them, but each is continuous with the other held to either side: there
    is no jump for either to hide. An earlier cut refused any two in one law
    that a column moves apart."""
    got = _columns(TWO_SPECIES + f"Jy: -> Y; {law}\n", ["thr"])
    assert got[2, 0] == pytest.approx(want, rel=1e-7, abs=1e-9)


# ── From the fifth review ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "law",
    [
        # Two closers: with one put before, the other still keeps the window shut.
        "piecewise(k, S >= 0.5*thr, 0)*piecewise(0, Z >= 1.5, 1)*piecewise(0, Q >= 1.5, 1)",
        # A closer that reads the opener's species: neither can be moved alone.
        "piecewise(k, S >= 0.5*thr, 0)*piecewise(0, Z + S >= 3, 1)",
    ],
    ids=["three-on-the-instant", "a-species-both-read"],
)
def test_switches_on_one_instant_that_cannot_be_asked_apart_are_refused(law):
    """The window opens where S reaches 0.5·thr and is closed on the same
    instant: 0 came back on the kink −0.5 | 0. A pair on different species
    is asked, each with the other held before; these cannot be."""
    text = (
        "species S, Z, Q, Y; S = 0; Z = 0; Q = 0; Y = 0; k = 0.5; thr = 3\n"
        "Js: -> S; 0.5\nJz: -> Z; 0.5\nJq: -> Q; 0.5\n"
        f"Jy: -> Y; {law}\n"
    )
    _refused(text, ["thr"], 946)


def test_two_bends_on_one_species_in_different_rate_laws_run():
    """Control. Two ramps from thresholds of one species, in two rate laws,
    with a column that moves one: no law reads both conditions.
    dY/dthr = −k·(T − thr)/2."""
    text = HEAD + (
        "Jx: -> X; piecewise(k*(S - 1.5), S >= 1.5, 0)\n"
        "Jy: -> Y; piecewise(k*(S - 0.5*thr), S >= 0.5*thr, 0)\n"
    )
    np.testing.assert_allclose(_columns(text, ["thr"])[:, 0], [0.0, 0.0, -0.5], atol=1e-9)


def test_branches_that_meet_to_the_digits_their_constants_are_written_to_are_no_jump():
    """Control. ``piecewise(0.333333333*X, X < 1.5, 0.5)`` with X on 1.5 until
    a fitted gate takes it down: the two branches differ by 5e-10 there, a
    clamp with its slope written to nine digits. Read against nothing but
    rounding that was a jump, and the run was refused. Y loses nothing until
    tau, and dY/dtau = 1 to the gap."""
    text = (
        "species X, Y; X = 1.5; Y = 0; r = 1.5; tau = 3\n"
        "Jx: X -> ; piecewise(r, time >= tau, 0)\n"
        "Jy: -> Y; piecewise(0.333333333*X, X < 1.5, 0.5)\n"
    )
    got = _columns(text, ["tau"])
    assert got[1, 0] == pytest.approx(1.0, rel=1e-6)
