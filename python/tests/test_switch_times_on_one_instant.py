"""Rate-law conditions on ``time`` that share an instant (issues #951 and #944).

A fitted switch time's jump is read by nudging the clock across it. Where another
condition sits on the same instant it flips with the nudge, so the crossing is
read with its own threshold raised a hair instead (issue #375), which leaves the
other condition already switched. Two things were wrong with that.

**#951.** The jump read that way is the crossing's with the other condition
past. If the two conditions are composed, a product of two gates, the later of
two times, a window of no width, the jump with the other still to come is a
different one, and a parameter that moves one crossing and not the other has a
kink there: its derivative from above and from below differ. The run returned
one of them. What the other conditions do is now read with this crossing still
to come and with it made, and the run is refused where the two differ.

**#944.** A step call on a clock outside any condition, ``floor(time/3)``, was
placed as a stop and not counted among the conditions on its instant, so a
fitted switch that landed on it was read with the step inside its bracket and
took the step's jump into its own column: dW/dtau = −4.2 for 0. A step that
sits on a fitted switch's instant is counted now, and the switch is read apart
from it.

Every expected value is a closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

TIMES = [0.0, 1.5, 3.0, 4.5, 6.0]


def _columns(text, params, times=TIMES):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=list(params)).run(
        sample_times=list(times), rtol=1e-10, atol=1e-12
    )
    return np.asarray(run.sensitivities)[-1]


COMPOSED = {
    # −0.75 from above, −0.375 from below; the run returned −0.75.
    "a-product-of-a-fitted-and-a-fixed-gate": (
        "species Y; Y = 0; r = 1; tau = 3\n"
        "J1: -> Y; r*piecewise(1, time >= tau, 0.25)*piecewise(1, time >= 3, 0.5)\n",
        ["tau"],
    ),
    # (−1, −1) from above and (0, 0) from below; the run returned (−1, −1),
    # which says moving both by δ moves Y by −2δ where it moves it by −δ.
    "the-later-of-two-times": (
        "species Y; Y = 0; r = 1; tau = 3; t2 = 3\n"
        "J1: -> Y; piecewise(r, time >= tau && time >= t2, 0)\n",
        ["tau", "t2"],
    ),
    # (0, 1) from above and (−1, 0) from below; the run returned (0, 1).
    "a-window-of-no-width": (
        "species Y; Y = 0; r = 1; tau = 3; t2 = 3\n"
        "J1: -> Y; piecewise(r, time >= tau && time < t2, 0)\n",
        ["tau", "t2"],
    ),
    # Two fitted gates with one of them asked for: −0.5 from above and −0.125
    # from below in t2.
    "a-product-of-two-fitted-gates-one-requested": (
        "species Y; Y = 0; r = 1; tau = 3; t2 = 3\n"
        "J1: -> Y; r*piecewise(1, time >= tau, 0.25)*piecewise(1, time >= t2, 0.5)\n",
        ["t2"],
    ),
}


@pytest.mark.parametrize("case", sorted(COMPOSED))
def test_conditions_on_one_instant_that_do_not_commute_are_refused(case):
    """Each has a kink in the requested parameter at the value it is run at."""
    text, params = COMPOSED[case]
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
    with pytest.raises(bngsim.SimulationError, match="do not commute.*issue #951"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


def test_two_rate_laws_that_switch_together_run():
    """Control. Y = r·(T − tau) and Z = 2r·(T − t2): each crossing's jump is
    its own whichever side of the other it is read from."""
    text = (
        "species Y, Z; Y = 0; Z = 0; r = 1; tau = 3; t2 = 3\n"
        "J1: -> Y; piecewise(r, time >= tau, 0)\n"
        "J2: -> Z; piecewise(2*r, time >= t2, 0)\n"
    )
    np.testing.assert_allclose(
        _columns(text, ["tau", "t2"]), [[-1.0, 0.0], [0.0, -2.0]], atol=1e-9
    )


def test_two_gates_summed_in_one_rate_law_run():
    """Control. One rate law, and the two gates add: dY/dtau = −r and
    dY/dt2 = −2r."""
    text = (
        "species Y; Y = 0; r = 1; tau = 3; t2 = 3\n"
        "J1: -> Y; piecewise(r, time >= tau, 0) + piecewise(2*r, time >= t2, 0)\n"
    )
    np.testing.assert_allclose(_columns(text, ["tau", "t2"]), [[-1.0, -2.0]], atol=1e-9)


@pytest.mark.parametrize("fixed", [4.0, 2.0], ids=["later", "earlier"])
def test_a_product_of_two_gates_on_different_instants_runs(fixed):
    """Control. The same product with the fixed gate at another time: the fitted
    gate's jump is 0.75·r times the fixed gate's value at t = 3."""
    text = (
        "species Y; Y = 0; r = 1; tau = 3\n"
        f"J1: -> Y; r*piecewise(1, time >= tau, 0.25)*piecewise(1, time >= {fixed}, 0.5)\n"
    )
    want = -0.75 * (0.5 if fixed > 3.0 else 1.0)
    np.testing.assert_allclose(_columns(text, ["tau"]), [[want]], rtol=1e-9)


def test_a_composed_pair_no_requested_parameter_moves_runs():
    """Control. The product of two gates on one instant, with only r asked for:
    nothing moves either crossing, and Y = r·(0.125·3 + 3) at t = 6."""
    text = (
        "species Y; Y = 0; r = 1; tau = 3\n"
        "J1: -> Y; r*piecewise(1, time >= tau, 0.25)*piecewise(1, time >= 3, 0.5)\n"
    )
    np.testing.assert_allclose(_columns(text, ["r"]), [[0.125 * 3 + 3.0]], rtol=1e-9)


FLOOR = (
    "species X, Y, W; X = 0; Y = 0; W = 0; a = 2; k = 0.5; q = 0.7; tau = {tau}\n"
    "J0: -> X; a\n"
    "J1: -> Y; piecewise(k*X, time >= tau, 0)\n"
    "J2: -> W; q*X*{step}\n"
)
FLOOR_TIMES = [float(t) for t in np.linspace(0.0, 7.5, 16)]


@pytest.mark.parametrize("step", ["floor(time/3)", "ceil(time/3)"])
def test_a_fitted_switch_on_a_step_of_time_does_not_take_the_step(step):
    """W's rate law steps at t = 3 whatever tau is, so dW/dtau = 0. Y's switches
    at tau = 3 with X = 6: dY/dtau = −k·X(tau) = −3. The step's jump, q·X(3),
    came back in the tau column of W: −4.2."""
    got = _columns(FLOOR.format(tau=3, step=step), ["tau"], FLOOR_TIMES)
    np.testing.assert_allclose(got, [[0.0], [-3.0], [0.0]], atol=1e-8)


def test_a_fitted_switch_away_from_the_step():
    """Control. tau = 3.3: dY/dtau = −k·a·tau."""
    got = _columns(FLOOR.format(tau=3.3, step="floor(time/3)"), ["tau"], FLOOR_TIMES)
    np.testing.assert_allclose(got, [[0.0], [-3.3], [0.0]], atol=1e-8)


def test_an_event_on_a_fitted_switch_and_a_step_is_refused():
    """The event halves X at tau, on the step: W's column has a kink in tau,
    −2.1 from one side and −4.2 from the other. It came back −4.2, or −6.3. The
    step is counted on the instant now, so the event's own test for a fixed
    switch it does not commute with sees it."""
    text = FLOOR.format(tau=3, step="floor(time/3)") + "E1: at (time >= tau): X = 0.5*X\n"
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(bngsim.SimulationError, match="Forward sensitivity"):
        sim.run(sample_times=FLOOR_TIMES, rtol=1e-10, atol=1e-12)


# ─── What a first cut of this refused, or let through ───────────────────────

BESIDE = "J9: -> Z; piecewise(2, time >= 3, 0.5)\n"
OWN_TIME = {
    # Y = r·(T − tau)²/2
    "a-ramp-from-the-switch": ("piecewise(r*(time - tau), time >= tau, 0)", -3.0),
    # Y = (r/k)·(1 − e^(−k·(T − tau)))
    "a-decay-from-the-switch": (
        "piecewise(r*exp(-k*(time - tau)), time >= tau, 0)",
        -float(np.exp(-0.5 * 3.0)),
    ),
    # Y = r·tau·(T − tau)
    "the-switch-time-as-a-rate": ("piecewise(r*tau, time >= tau, 0)", 0.0),
}


@pytest.mark.parametrize("case", sorted(OWN_TIME))
def test_a_rate_law_that_reads_its_own_switch_time_beside_another_switch(case):
    """Control. The law of Y reads tau outside its condition, and a law of Z
    switches at the literal 3, which is tau. The two commute. Read as this
    crossing's own jump from the two sides, its after-branch is a nudge past
    its onset on one and a whole hair past it on the other, and a first cut of
    this refused every such model, two of the corpus among them. What the other
    condition does is the same with this one before and after."""
    law, want = OWN_TIME[case]
    text = "species Y, Z; Y = 0; Z = 0; r = 1; k = 0.5; tau = 3\n" + f"J1: -> Y; {law}\n" + BESIDE
    np.testing.assert_allclose(_columns(text, ["tau"]), [[want], [0.0]], rtol=1e-7, atol=1e-9)


TWO_CLOCKS = """begin parameters
    1 r 1.0
    2 tau 3
    3 t2 3
    4 _rateLaw1 1
    5 k 0.5
end parameters
begin functions
    1 fY() {law}
end functions
begin species
    1 Y() 0
    2 Tc() 0
end species
begin reactions
    1 0 1 fY
    2 0 2 _rateLaw1
end reactions
begin groups
    1 t 2
end groups
"""
ACROSS_CLOCKS = {
    # −0.75 from above and −0.375 from below; the run returned −0.375.
    "a-product-of-a-gate-on-time-and-one-on-a-counter": (
        "r*if(time()>=tau,1,0.25)*if(t>=3,1,0.5)",
        ["tau"],
    ),
    # (−1, −1) from above and (0, 0) from below; the run returned (0, −1).
    "the-later-of-a-time-and-a-counter": ("if((time()>=tau)&&(t>=t2),r,0)", ["tau", "t2"]),
    # −0.5 from above and 0 from below; the run returned 0.
    "a-gate-on-time-and-a-step-of-a-counter": ("if(time()>=tau,k,0)*floor(t/3)", ["tau"]),
}


@pytest.mark.parametrize("case", sorted(ACROSS_CLOCKS))
def test_conditions_on_two_clocks_that_do_not_commute_are_refused(tmp_path, case):
    """One condition on the time and one on a counter that reads the same
    instant. A nudge of one clock does not flip the condition on the other, so
    each was read with the other where it stood, and the two were never
    compared. Every clock on the instant is nudged now."""
    law, params = ACROSS_CLOCKS[case]
    path = tmp_path / "m.net"
    path.write_text(TWO_CLOCKS.format(law=law))
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=params)
    with pytest.raises(bngsim.SimulationError, match="Forward sensitivity"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "step", ["floor(time*time/9)", "floor(sqrt(3*time))", "floor(exp(time - 3))"]
)
def test_a_fitted_switch_on_a_step_whose_edges_are_not_listed(step):
    """These step at t = 3 too, and are not steps whose edges bngsim lists. The
    step is found by evaluating the call a nudge either side of the fitted
    switch's instant. dW/dtau came back −4.2 for 0."""
    got = _columns(FLOOR.format(tau=3, step=step), ["tau"], FLOOR_TIMES)
    np.testing.assert_allclose(got, [[0.0], [-3.0], [0.0]], atol=1e-8)


def test_a_fitted_switch_on_a_step_at_the_end_of_the_run():
    """The run ends on the instant. A step's edge at the end of the window was
    not listed, and W's column came back −4.2 for 0."""
    got = _columns(FLOOR.format(tau=3, step="floor(time/3)"), ["tau"], [0.0, 1.5, 3.0])
    assert got[2, 0] == pytest.approx(0.0, abs=1e-8)


def test_a_fitted_switch_on_one_of_twelve_thousand_steps():
    """``floor(time/0.0005)`` has more edges in the window than are listed, and
    was then not counted at all: −4.2 for 0."""
    got = _columns(FLOOR.format(tau=3, step="floor(time/0.0005)"), ["tau"], [0.0, 1.5, 4.5, 6.0])
    assert got[2, 0] == pytest.approx(0.0, abs=1e-6)


TIED = {
    # Both gates move with p: Y = r·(0.125·p + (T − p)), dY/dp = −0.875.
    "a-product": "r*piecewise(1, time >= tA, 0.25)*piecewise(1, time >= tB, 0.5)",
    # dY/dp = −1.
    "the-later-of-the-two": "piecewise(r, time >= tA && time >= tB, 0)",
}


@pytest.mark.parametrize("case", sorted(TIED))
def test_two_composed_switches_that_one_parameter_moves_together_are_refused(case):
    """``tA := p + a`` and ``tB := p + b`` with a = b = 0: one parameter moves
    both crossings at one rate, so the result has a derivative in p, −0.875 and
    −1. It came back −1.25 and −2: each crossing's jump was read with the other
    made, and the two were added. The two are not taken as one crossing, and
    the run is refused."""
    text = (
        "species Y; Y = 0; r = 1; p = 3; a = 0; b = 0; tA := p + a; tB := p + b\n"
        f"J1: -> Y; {TIED[case]}\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["p"])
    with pytest.raises(bngsim.SimulationError, match="do not commute.*issue #951"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


def test_a_neighbour_whose_rate_law_reads_the_moved_switch_time():
    """Control. Z's law switches at the literal 3 and is scaled by tau, which is
    the threshold raised a hair to read Y's switch apart: what Z does across
    the instant is 1.5·tau, and differs between the two readings by the hair.
    That part is in proportion to the hair and is taken out. Z = tau·(0.5·3 +
    2·(T − 3)), so dZ/dtau = 7.5, and dY/dtau = −r. To a part in a million:
    the jump that is applied is still read with the threshold a hair up, as
    it was."""
    text = (
        "species Y, Z; Y = 0; Z = 0; r = 1; tau = 3\n"
        "J1: -> Y; piecewise(r, time >= tau, 0)\n"
        "J9: -> Z; tau*piecewise(2, time >= 3, 0.5)\n"
    )
    np.testing.assert_allclose(_columns(text, ["tau"]), [[-1.0], [7.5]], rtol=1e-5)


def test_a_gate_a_hundred_ulp_past_the_instant_is_not_taken_for_a_slope():
    """``r·gate(tau)·(gate(3) + gate(3·(1 + 100·ε)))``: dY/dtau is −2 from
    above and 0 from below, and came back −1. The second fixed gate is not on
    the instant, and is inside twice the nudge. Read again over twice the
    nudge, what it adds doubled with the nudge, as a slope would, and was
    taken out with it. The second reading is over half the nudge."""
    late = repr(float(3.0 * (1 + 100 * np.finfo(float).eps)))
    text = (
        "species Y; Y = 0; r = 1; tau = 3\n"
        "J1: -> Y; r*piecewise(1, time >= tau, 0)"
        f"*(piecewise(1, time >= 3, 0) + piecewise(1, time >= {late}, 0))\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(bngsim.SimulationError, match="do not commute.*issue #951"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


COUNTERS = """begin parameters
    1 a 2
    2 k 0.5
    3 q 0.7
    4 tau {tau}
    5 _rateLaw1 1
end parameters
begin functions
    1 fY() if(t>=tau,k*Xo,0)
    2 fW() {step}
end functions
begin species
    1 X() 0
    2 Y() 0
    3 W() 0
    4 Tc() {start}
    5 Uc() 0
end species
begin reactions
    1 0 1 a
    2 0 2 fY
    3 0 3 fW
    4 0 4 _rateLaw1
    5 0 5 _rateLaw1
end reactions
begin groups
    1 Xo 1
    2 t 4
    3 u 5
end groups
"""


@pytest.mark.parametrize(
    ("step", "tau", "start"),
    [("q*Xo*floor((t-1e6)/3)", "1000003", "1e6"), ("q*Xo*floor((t+u)/6)", "3", "0")],
    ids=["a-counter-that-starts-at-a-million", "a-step-of-two-counters"],
)
def test_a_fitted_switch_on_a_step_of_a_counter(tmp_path, step, tau, start):
    """The switch and the step are on a counter clock. A counter at 1e6 is
    nudged about its own value, not about the time, and a call that reads two
    counters is on no one clock and is taken to step. dW/dtau came back −4.2
    for 0 in both."""
    path = tmp_path / "m.net"
    path.write_text(COUNTERS.format(step=step, tau=tau, start=start))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["tau"]
    ).run(sample_times=[0.0, 1.0, 2.0, 4.0, 5.0, 7.5], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, :3, 0]
    np.testing.assert_allclose(got, [0.0, -3.0, 0.0], atol=1e-7)


# ─── What a second cut refused, or let through ──────────────────────────────

RAMPS = {
    # Y = r·(T − tau)²/2 with T = tau + 3.
    "up-from-the-switch": ("piecewise(r*(time - tau), time >= tau, 0)", lambda tau: -3.0),
    # Y = r·tau²/2.
    "down-to-the-switch": ("piecewise(0, time >= tau, r*(tau - time))", lambda tau: tau),
}


@pytest.mark.parametrize("tau", [2.6, 3.7, 5.3, 17.3, 123.456, 1000.1])
@pytest.mark.parametrize("shape", sorted(RAMPS))
def test_a_ramp_beside_another_switch_at_a_time_that_is_not_a_round_number(shape, tau):
    """Control. A ramp from its own switch time, beside a law of Z that
    switches at the same number. The ramp's slope is in what the other
    condition is read to do, in proportion to how far the clock is moved, and
    is taken out by a second reading over half as far. At tau = 3 half of 64
    ulp is a double. At 2.6 it is not: the two lengths were not as 1 to a
    half, the slope times an ulp was left, and against a rate that is 0 at its
    own switch that read as a kink. About half of all switch times were
    refused. The lengths the clock was in fact moved over are used now."""
    law, want = RAMPS[shape]
    text = (
        f"species Y, Z; Y = 0; Z = 0; r = 1; tau = {tau!r}\n"
        f"J1: -> Y; {law}\n"
        f"J9: -> Z; piecewise(2, time >= {tau!r}, 0.5)\n"
    )
    got = _columns(text, ["tau"], [0.0, tau / 2, tau + 1.0, tau + 3.0])
    np.testing.assert_allclose(got, [[want(tau)], [0.0]], rtol=1e-6, atol=1e-8)


BILLION = """begin parameters
    1 r 1.0
    2 tau 1000000003
    3 _rateLaw1 1
end parameters
begin functions
    1 fY() if(t>=tau,r*(t-tau),0)
    2 fZ() {gate}
end functions
begin species
    1 Y() 0
    2 Tc() 1e9
    3 Z() 0
    4 Uc() 0
end species
begin reactions
    1 0 1 fY
    2 0 2 _rateLaw1
    3 0 3 fZ
    4 0 4 _rateLaw1
end reactions
begin groups
    1 t 2
    2 u 4
end groups
"""


@pytest.mark.parametrize(
    "gate",
    ["if(t>=1000000003,2,0.5)", "if(time()>=3,2,0.5)", "if(u>=3,2,0.5)"],
    ids=["a-gate-on-the-same-counter", "a-gate-on-the-time", "a-gate-on-another-counter"],
)
def test_a_ramp_on_a_counter_that_starts_at_a_billion(tmp_path, gate):
    """Control. The same ramp on a counter clock that starts at 1e9, beside a
    gate on that counter, on the time, and on a second counter. A nudge of a
    counter at 1e9 is 1.4e-5, and half of it is not half as far in doubles.
    dY/dtau = −r·(T − tau) = −3, to what a counter at 1e9 resolves."""
    path = tmp_path / "m.net"
    path.write_text(BILLION.format(gate=gate))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["tau"]
    ).run(sample_times=[0.0, 1.0, 2.0, 4.0, 5.0, 6.0], rtol=1e-10, atol=1e-12, timeout=60)
    got = np.asarray(run.sensitivities)[-1, :, 0]
    np.testing.assert_allclose(got[[0, 2]], [-3.0, 0.0], rtol=1e-4, atol=1e-8)


def _gate(ulps):
    at = repr(float(3.0 * (1 + ulps * np.finfo(float).eps)))
    return f"piecewise(1, time >= {at}, 0)"


NEAR = {
    # −2 from above and 0 from below; the run returned −2. The second gate is
    # inside the nudge and outside half of it, with the weight of the first.
    "two-gates-of-one-weight-forty-ulp-apart": [0, 40],
    "two-gates-of-one-weight-sixty-ulp-apart": [0, 60],
    # Neither fixed gate on the fitted switch's own time.
    "neither-gate-on-the-switch": [10, 40],
    "three-gates": [0, 40, 100],
}


@pytest.mark.parametrize("case", sorted(NEAR))
def test_gates_a_few_ulp_from_the_instant_are_refused(case):
    """``r·gate(tau)·(gate(3 + a) + gate(3 + b) + …)`` with the fixed gates a
    few ulp from the fitted switch at 3: dY/dtau has a kink that close. A
    gate that is inside the nudge and outside half of it adds its jump to the
    long reading and not to the short one, which is what a slope does, and one
    with the weight of the gate on the instant cancelled that one's part
    exactly. A third reading, over a quarter of the nudge, does not agree with
    the other two unless the difference is a jump and a slope."""
    text = (
        "species Y; Y = 0; r = 1; tau = 3\n"
        "J1: -> Y; r*piecewise(1, time >= tau, 0)*("
        + " + ".join(_gate(n) for n in NEAR[case])
        + ")\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(bngsim.SimulationError, match="issue #951"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("ulps", [100, -100])
def test_a_gate_composed_with_the_switch_a_hundred_ulp_away_is_refused(ulps):
    """``r·gate(tau)·g`` with g going from 0.5 to 1 a hundred ulp from tau,
    beside a law of Z that switches on tau's own instant: dY/dtau is −1 on one
    side and −0.5 on the other, and the run returned one of them. Every clock
    is now put outside the whole instant, before its earliest threshold and
    past its latest, so the gate is in every reading."""
    at = repr(float(3.0 * (1 + ulps * np.finfo(float).eps)))
    text = (
        "species Y, Z; Y = 0; Z = 0; r = 1; tau = 3\n"
        f"J1: -> Y; r*piecewise(1, time >= tau, 0)*piecewise(1, time >= {at}, 0.5)\n"
        "J9: -> Z; piecewise(2, time >= 3, 0.5)\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(bngsim.SimulationError, match="issue #951"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)
