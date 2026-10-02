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
one of them. Both jumps are read now, and the run is refused where they differ.

**#944.** A step of ``floor(time/3)`` outside any condition was placed as a stop
and not counted among the conditions on its instant, so a fitted switch that
landed on it was read with the step inside its bracket and took the step's jump
into its own column: dW/dtau = −4.2 for 0. It is counted now, and the switch is
read apart from it.

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
    # Only the fixed gate's partner is asked for: the kink is in tau alone.
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
