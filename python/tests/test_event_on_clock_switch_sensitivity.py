"""An event and a time-clock rate-law switch at one instant (issue #767).

A dosing or reset time is often one parameter that both fires an event and
gates a rate law: ``at (time >= tau): X = 0`` beside
``piecewise(k*X, time >= tau, 0)``. The event jump shifts the sensitivities
along the flow just before and just after the event by how far the event time
moves, and the switch jump adds the gap between the rate law's two branches.

Both were applied, but the event jump read its flows at the instant itself,
where ``time >= tau`` is already true: on the switch's after-branch. The switch
jump then added its gap at the post-event state. With H the event's Jacobian
and Δ = f_after − f_before, the column that moves the shared time was off by
``(H·Δ(x⁻) − Δ(x⁺))·∂t*/∂p``, with no warning and the trajectory right.

The event jump now reads its flows a nudge before the instant, on the
before-branch, and the two jumps compose. Where the event and the switch do not
move together with a requested parameter, the run is refused: which of them
comes first would then depend on the parameter.

Every expected value is a closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

TIMES = [float(t) for t in np.linspace(0.0, 6.0, 13)]
T = TIMES[-1]


def _sens(text, params):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
        sample_times=TIMES, rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = dict(zip(names, np.asarray(run.species)[-1], strict=True))
    s = dict(zip(names, np.asarray(run.sensitivities)[-1], strict=True))
    return x, s


RESET = (
    "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = 3\n"
    "J0: -> X; a\n"
    "J1: -> Y; piecewise(k*X, time {cmp} tau, 0)\n"
    "E1: at (time {trig} tau): X = {reset}\n"
)


@pytest.mark.parametrize("cmp", [">=", ">"])
@pytest.mark.parametrize("trig", [">=", ">"])
@pytest.mark.parametrize("keep", [0.0, 0.5])
def test_a_reset_and_a_switched_rate_law_on_one_time(cmp, trig, keep):
    """X grows at a, is reset to keep·X at tau, and feeds Y at k·X from tau on.
    With c = keep and u = T − tau:

        X(T) = c·a·tau + a·u        Y(T) = k·(c·a·tau·u + a·u²/2)

    With ``>=`` in the rate law, dY/dtau came out 0 for −3 at keep = 0 and −1.5
    for −3 at keep = 0.5. With ``>`` the flows happened to be read on the right
    branch, and that spelling was already right."""
    a, k, tau, c = 2.0, 0.5, 3.0, keep
    u = T - tau
    reset = "0" if keep == 0.0 else f"{keep!r}*X"
    x, s = _sens(RESET.format(cmp=cmp, trig=trig, reset=reset), ["tau", "a", "k"])
    assert x["X"] == pytest.approx(c * a * tau + a * u, rel=1e-8)
    assert x["Y"] == pytest.approx(k * (c * a * tau * u + a * u * u / 2), rel=1e-8)
    want_y = [
        k * (c * a * u - c * a * tau - a * u),
        k * (c * tau * u + u * u / 2),
        c * a * tau * u + a * u * u / 2,
    ]
    np.testing.assert_allclose(s["Y"], want_y, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(s["X"], [c * a - a, c * tau + u, 0.0], rtol=1e-6, atol=1e-8)


def test_the_switch_on_the_assigned_species_itself():
    """X' is 2 before tau and 5 after, and the event halves X at tau:
    X(T) = tau + 5·(T − tau), so dX/dtau = −4. It came out −5.5: the switched
    branch does not have to read the assigned species for the two jumps to
    interfere."""
    text = (
        "species X; X = 0; tau = 3\n"
        "J0: -> X; piecewise(5, time >= tau, 2)\n"
        "E1: at (time >= tau): X = 0.5*X\n"
    )
    x, s = _sens(text, ["tau"])
    assert x["X"] == pytest.approx(18.0, rel=1e-9)
    assert s["X"][0] == pytest.approx(-4.0, rel=1e-7)


DOSE = (
    "species X, Y; X = 1; Y = 0; kd = 0.4; k = 0.5; D = 2; tau = 3; lag = {lag}\n"
    "J0: X -> ; kd*X\n"
    "J1: -> Y; piecewise(k*X, time >= tau + lag, 0)\n"
    "E1: at (time >= tau): X = X + D\n"
)


def _dose_y(tau, lag, kd=0.4, k=0.5, dose=2.0):
    """Y(T) = k·∫ X over (tau + lag, T), X = e^(−kd·t) + D·e^(−kd·(t − tau))."""
    return (k / kd) * (
        np.exp(-kd * (tau + lag))
        - np.exp(-kd * T)
        + dose * (np.exp(-kd * lag) - np.exp(-kd * (T - tau)))
    )


@pytest.mark.parametrize("lag", [0.0, 0.05])
def test_a_dose_and_the_rate_law_it_switches_on(lag):
    """A dose D at tau, and a readout that integrates X from tau + lag on. With
    lag = 0 the two share an instant, and dY/dtau was off by exactly −k·D. With
    lag = 0.05 they are apart, which was right before and is the control."""
    x, s = _sens(DOSE.format(lag=lag), ["tau", "D", "kd"])
    assert x["Y"] == pytest.approx(_dose_y(3.0, lag), rel=1e-8)
    h = 1e-5
    want = [
        (_dose_y(3.0 + h, lag) - _dose_y(3.0 - h, lag)) / (2 * h),
        (_dose_y(3.0, lag, dose=2.0 + h) - _dose_y(3.0, lag, dose=2.0 - h)) / (2 * h),
        (_dose_y(3.0, lag, kd=0.4 + h) - _dose_y(3.0, lag, kd=0.4 - h)) / (2 * h),
    ]
    np.testing.assert_allclose(s["Y"], want, rtol=1e-6, atol=1e-8)


APART = (
    "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = 3\n"
    "J0: -> X; a\n"
    "J1: -> Y; piecewise(k*X, time >= {switch}, 0)\n"
    "E1: at (time >= {event}): X = 0\n"
)


@pytest.mark.parametrize(
    ("event", "switch"), [("3", "tau"), ("tau", "3")], ids=["fixed-event", "fixed-switch"]
)
def test_a_pair_that_tau_pulls_apart_is_refused(event, switch):
    """One of the two is at the literal time 3 and the other at tau = 3. Under
    tau they come apart, and Y's column depends on which is then first:
    X is reset before the rate law turns on, or after. That is a kink in tau,
    not a derivative, and the run is refused."""
    model = bngsim.Model.from_antimony_string(APART.format(event=event, switch=switch))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(Exception, match="issue #767"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("event", "switch"), [("3", "tau"), ("tau", "3")], ids=["fixed-event", "fixed-switch"]
)
def test_the_same_pair_runs_for_a_column_that_moves_neither(event, switch):
    """With only `a` requested, nothing moves either time, and
    Y(T) = k·a·(T − tau)²/2."""
    x, s = _sens(APART.format(event=event, switch=switch), ["a"])
    assert x["Y"] == pytest.approx(0.5 * 2.0 * 9.0 / 2, rel=1e-8)
    assert s["Y"][0] == pytest.approx(0.5 * 9.0 / 2, rel=1e-6)
