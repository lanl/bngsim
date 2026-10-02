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
GATES_WITH_NO_STOP = {
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


@pytest.mark.parametrize("case", sorted(GATES_WITH_NO_STOP))
def test_a_gate_the_run_takes_no_stop_for_is_refused_on_the_instant(case):
    """``time^2 >= 9`` switches at t = 3, a time the run does not know ahead:
    it is no stop and no record. What else jumps between the probes is asked
    of the right-hand side itself, with this switch's species held to one
    side of the crossing and everything else put to the other."""
    _refused(GATES_WITH_NO_STOP[case], ["thr"], 946)


def test_a_counter_s_gate_the_run_has_no_root_for_is_refused_on_the_instant(tmp_path):
    """The same on a counter species, ``if(t^2 >= 9, k, 0)``, which a run with
    sensitivities has no root for at all: dX/dthr = −0.5 for 0."""
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


@pytest.mark.parametrize(("param", "want"), [("k", 4.0), ("a", -0.5)])
def test_a_threshold_that_is_a_condition_runs_for_what_does_not_move_it(tmp_path, param, want):
    """Control. With k requested, or the first threshold a, nothing moves
    the crossing at t1."""
    got = np.asarray(_nested(tmp_path, [param]).sensitivities)[-1]
    assert got[0, 0] == pytest.approx(want, rel=1e-7)


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
