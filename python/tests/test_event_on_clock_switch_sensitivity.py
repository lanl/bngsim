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
before-branch, and the two jumps compose.

Where a requested parameter moves the event and not the switch, or the reverse,
the two come apart under it. The two orders differ by
``(H·Δ(x⁻) − Δ(x⁺))·(∂t_switch/∂p − ∂t_event/∂p)``, which is a kink in the
parameter unless the event and the switch commute. A pair that commutes runs:
a bolus beside an infusion that starts then. One that does not is refused.

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

    With ``>=`` in both, dY/dtau came out 0 for −3 at keep = 0 and −1.5 for −3
    at keep = 0.5. The other three spellings were already right: with ``>`` in
    the rate law the flows happened to be read on the before-branch, and with
    ``>`` in the trigger the switch's stop is taken first and the event fires a
    few ulp later, past it."""
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
    """Control. With only `a` requested, nothing moves either time, and
    Y(T) = k·a·(T − tau)²/2."""
    x, s = _sens(APART.format(event=event, switch=switch), ["a"])
    assert x["Y"] == pytest.approx(0.5 * 2.0 * 9.0 / 2, rel=1e-8)
    assert s["Y"][0] == pytest.approx(0.5 * 9.0 / 2, rel=1e-6)


def test_an_event_between_two_switch_times_one_instant_apart_is_refused():
    """Switch times at tau and 140 ulp later, and an event 50 ulp after tau.
    The stop at tau has taken the first switch's jump, the second's is still
    to come, and the event is within one instant of both. Its flows would have
    to be read after the one and before the other, which no nudge of the clock
    selects.

    The three times are set as doubles: written into the model text they keep
    15 significant digits."""
    tau = 3.0
    step = float(np.spacing(tau))
    text = (
        "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = 3; tau2 = 3; te = 3\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*X, time >= tau, 0) + piecewise(k, time >= tau2, 0)\n"
        "E1: at (time >= te): X = 0\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("tau2", tau + 140 * step)
    model.set_param("te", tau + 50 * step)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau", "tau2", "te"])
    with pytest.raises(Exception, match="between two rate-law switch times"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


PK = (
    "species X; X = 1; kd = 0.4; R = 1.5; D = 2; tau = 3\n"
    "J0: X -> ; kd*X\n"
    "J1: -> X; piecewise(R, time >= {infusion}, 0)\n"
    "E1: at (time >= {bolus}): X = X + D\n"
)


@pytest.mark.parametrize(
    ("bolus", "infusion", "want"),
    [("3", "tau", -1.5 * np.exp(-0.4 * 3.0)), ("tau", "3", 2.0 * 0.4 * np.exp(-0.4 * 3.0))],
    ids=["fixed-bolus", "fixed-infusion-start"],
)
def test_a_pair_that_commutes_runs(bolus, infusion, want):
    """Control. A bolus at one time and an infusion that starts at the other, one
    of them at tau = 3 and one at the literal 3. The bolus adds to X and the
    infusion's rate does not read X, so the order of the two does not matter and
    the derivative exists: with u = T − 3,

        dX/d(infusion start) = −R·e^(−kd·u)      dX/d(bolus time) = D·kd·e^(−kd·u)

    An earlier cut of this fix refused every pair one parameter pulls apart."""
    _x, s = _sens(PK.format(bolus=bolus, infusion=infusion), ["tau"])
    assert s["X"][0] == pytest.approx(want, rel=1e-6)


TROUGH = (
    "species X; X = {x0}; kd = 0.004; R = 1.5; D = 100; tau = 3\n"
    "J0: X -> ; kd*X\n"
    "J1: -> X; piecewise(R, time >= {infusion}, 0)\n"
    "E1: at (time >= {bolus}): X = X + D\n"
)


@pytest.mark.parametrize("x0", ["0", "0.001"])
@pytest.mark.parametrize(
    ("bolus", "infusion", "want"),
    [
        ("3", "tau", -1.5 * np.exp(-0.004 * 3.0)),
        ("tau", "3", 100.0 * 0.004 * np.exp(-0.004 * 3.0)),
    ],
    ids=["fixed-bolus", "fixed-infusion-start"],
)
def test_a_bolus_into_an_empty_compartment_commutes_too(x0, bolus, infusion, want):
    """The same pair with a dose of 100 into a compartment holding nothing, or
    0.001. An earlier cut carried the switch's jump through the event by a
    Jacobian differenced over a millionth of the state, which at a state near
    zero beside a dose of 100 left 1e-5 of the dose in it, and refused.

    The fixed bolus is a control. The moving one was off by 1e-5 before: the
    event's own jump differenced the dose the same way (see
    test_event_assignment_derivative_step.py)."""
    _x, s = _sens(TROUGH.format(x0=x0, bolus=bolus, infusion=infusion), ["tau"])
    assert s["X"][0] == pytest.approx(want, rel=1e-6)


SMOOTH = (
    "species X, Y; X = 1; Y = 0; k = 0.5; tau = 3\n"
    "J1: -> Y; {law}\n"
    "E1: at (time >= tau): X = 2*X\n"
)


@pytest.mark.parametrize(
    ("law", "want"),
    [
        ("k*X*(time - 3)", 0.0),
        ("k*X*sin(time)", -0.5 * np.sin(3.0)),
        ("k*X*(1 + sin(200*time))", -0.5 * (1.0 + np.sin(600.0))),
        ("piecewise(k*(time - 3), time >= 3, 0)", 0.0),
        ("piecewise(k*X*(time - 3)^2, time >= 3, 0)", 0.0),
        ("k*X*max(0, time - 3)^2", 0.0),
    ],
    ids=["through-zero", "sine", "fast-sine", "continuous-ramp", "square-onset", "max-onset"],
)
def test_a_smooth_rate_law_is_not_taken_for_a_switch(law, want):
    """Control. Nothing jumps at the event's instant: a rate law that passes
    through zero there, an oscillation, and a ramp that starts there and is
    continuous. X doubles at tau, so dY/dtau = −(rate law per unit X at tau).

    An earlier cut read the right-hand side a nudge either side of the instant
    and refused where it changed by more than 1e-6 of its size, which a rate
    near zero or a fast one does. A later one compared against the flows it
    had read, which an onset of second order leaves at 1e-26: rounding beside
    rounding."""
    _x, s = _sens(SMOOTH.format(law=law), ["tau"])
    assert s["Y"][0] == pytest.approx(want, rel=1e-5, abs=1e-7)


def test_a_smooth_rate_law_late_in_time():
    """Control. The nudge is 64 ulp of the time, so at t = 1e6 it is 1e-8 and a
    modest frequency moves the rate law by more than rounding across it."""
    t0 = 1.0e6
    text = (
        f"species X, Y; X = 1; Y = 0; k = 0.5; tau = {t0 + 3.0!r}\n"
        "J1: -> Y; k*X*(1 + sin(30*time))\n"
        "E1: at (time >= tau): X = 2*X\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"]).run(
        sample_times=[t0, t0 + 2.0, t0 + 5.0], rtol=1e-10, atol=1e-12
    )
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y"), 0]
    assert got == pytest.approx(-0.5 * (1.0 + np.sin(30.0 * (t0 + 3.0))), rel=1e-5)


def test_an_assignment_that_reads_a_row_late_in_time():
    """Control. The same model with W set to 1e6·Y at the event. What the
    clock's rounding leaves in Y's row at t = 1e6 goes through the assignment
    at a gain of 1e6 into a row whose own rate is 0, and has to be allowed for
    there too: dW/dtau = 1e6·Y'(tau)."""
    t0 = 1.0e6
    text = (
        f"species X, Y, W; X = 1; Y = 0; W = 0; k = 0.5; tau = {t0 + 3.0!r}\n"
        "J1: -> Y; k*X*(1 + sin(30*time))\n"
        "E1: at (time >= tau): X = 2*X, W = 1e6*Y\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"]).run(
        sample_times=[t0, t0 + 2.0, t0 + 5.0], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    got = np.asarray(run.sensitivities)[-1, :, 0]
    rate = 0.5 * (1.0 + np.sin(30.0 * (t0 + 3.0)))
    assert got[names.index("W")] == pytest.approx(1e6 * rate, rel=1e-5)
    assert got[names.index("Y")] == pytest.approx(-rate, rel=1e-5)


def test_a_shift_that_is_rounding_is_not_a_disagreement():
    """Control. U = b·t reaches b·tau at tau whatever b is, so the trigger's
    time moves with tau and not with b. Its ∂t*/∂b is a finite difference and
    comes out 1e-17 where the switch's is an exact 0. That is the same shift,
    and an earlier cut refused it as a different one."""
    text = (
        "species X, Y, U; X = 0; Y = 0; U = 0; a = 2; k = 0.5; b = 1.5; tau = 3\n"
        "J0: -> X; a\n"
        "JU: -> U; b\n"
        "J1: -> Y; piecewise(k*X, time >= tau, 0)\n"
        "E1: at (U >= b*tau): X = 0.5*X\n"
    )
    _x, s = _sens(text, ["tau", "b"])
    a, k, tau, c = 2.0, 0.5, 3.0, 0.5
    u = T - tau
    assert s["Y"][0] == pytest.approx(k * (c * a * u - c * a * tau - a * u), rel=1e-6)
    assert s["Y"][1] == pytest.approx(0.0, abs=1e-7)


def test_a_trigger_time_no_requested_parameter_moves():
    """Control. The same trigger beside a rate law that switches at the literal
    3, with only b requested. Nothing requested moves either time, the event's
    1e-17 is no shift, and Y does not depend on b. An earlier cut took any
    nonzero shift for a moving event and refused."""
    text = (
        "species X, Y, U; X = 0; Y = 0; U = 0; a = 2; k = 0.5; b = 1.5; tau = 3\n"
        "J0: -> X; a\n"
        "JU: -> U; b\n"
        "J1: -> Y; piecewise(k*X, time >= 3, 0)\n"
        "E1: at (U >= b*tau): X = 0.5*X\n"
    )
    x, s = _sens(text, ["b"])
    a, k, tau, c = 2.0, 0.5, 3.0, 0.5
    u = T - tau
    assert x["Y"] == pytest.approx(k * (c * a * tau * u + a * u * u / 2), rel=1e-8)
    assert s["Y"][0] == pytest.approx(0.0, abs=1e-7)


def test_a_trigger_time_differenced_beside_values_near_a_thousand():
    """Control. U starts at 1000 and the trigger is U >= b·tau + 1000, so the
    event is at tau and its ∂t*/∂tau is a difference of values near 1000: 1 to
    about nine digits, beside the switch's exact 1. An earlier cut held the
    two to 1e-9 of each other and refused."""
    text = (
        "species X, Y, U; X = 0; Y = 0; U = 1000; a = 2; k = 0.5; b = 1.5; tau = 3\n"
        "J0: -> X; a\n"
        "JU: -> U; b\n"
        "J1: -> Y; piecewise(k*X, time >= tau, 0)\n"
        "E1: at (U >= b*tau + 1000): X = 0.5*X\n"
    )
    _x, s = _sens(text, ["tau"])
    a, k, tau, c = 2.0, 0.5, 3.0, 0.5
    u = T - tau
    assert s["Y"][0] == pytest.approx(k * (c * a * u - c * a * tau - a * u), rel=1e-5)


KINKS = {
    # The rate law reads X and switches at the literal 3. X is 0 before the
    # dose, so the switch's jump is 0 at the pre-event state and k·D after.
    "dose-from-zero": (
        "species X, Y; X = 0; Y = 0; k = 0.5; tau = 3\n"
        "J1: -> Y; piecewise(k*X, time >= 3, 0)\n"
        "E1: at (time >= tau): X = X + 2\n",
        "tau",
    ),
    # A flux of 1e14 in another species: the jump is small beside it, not small.
    # Each row is held to the rounding of its own flows.
    "beside-a-large-flux": (
        "species X, Y, W; X = 0; Y = 0; W = 0; a = 2; k = 0.5; tau = 3\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*X, time >= 3, 0)\n"
        "J2: -> W; 1e14\n"
        "E1: at (time >= tau): X = 0\n",
        "tau",
    ),
    # The same flux in the switched law's own row.
    "beside-a-large-flux-in-its-row": (
        "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = 3\n"
        "J0: -> X; a\n"
        "J1: -> Y; 1e8 + piecewise(k*X, time >= 3, 0)\n"
        "E1: at (time >= tau): X = 0\n",
        "tau",
    ),
    # Two counters cross their thresholds together. The run stops there once,
    # for the first; the second is the one the event does not commute with.
    "second-of-two-fixed-counters": (
        "species X, Y, W, C1, C2; X = 0; Y = 0; W = 0; C1 = 0; C2 = 1\n"
        "a = 2; k = 0.5; q = 0.7; tau = 3\n"
        "J0: -> X; a\n"
        "Jc1: -> C1; 1\n"
        "Jc2: -> C2; 1\n"
        "J1: -> Y; piecewise(k, C1 >= 3, 0)\n"
        "J2: -> W; piecewise(q*X, C2 >= 4, 0)\n"
        "E1: at (time >= tau): X = 0.5*X\n",
        "tau",
    ),
    "strict-trigger": (
        "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = 3\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*X, time >= 3, 0)\n"
        "E1: at (time > tau): X = 0\n",
        "tau",
    ),
    # A fixed threshold on a counter: a nudge of the time does not move it.
    "fixed-counter-threshold": (
        "species X, Y, C; X = 0; Y = 0; C = 0; a = 2; k = 0.5; tau = 3\n"
        "J0: -> X; a\n"
        "JC: -> C; 1\n"
        "J1: -> Y; piecewise(k*X, C >= 3, 0)\n"
        "E1: at (time >= tau): X = 0.5*X\n",
        "tau",
    ),
    # The event overwrites the species the switched rate law feeds. The jump is
    # the same at both states, and the event's Jacobian is what kills it.
    "overwrite-of-the-fed-species": (
        "species Z; Z = 0; r = 2; tau = 3\n"
        "J1: -> Z; piecewise(r, time >= 3, 0)\n"
        "E1: at (time >= tau): Z = 5\n",
        "tau",
    ),
    # X = a·t reaches 6 at t = 3, where the rate law switches. The root is
    # reported a few ulp from the switch's own stop.
    "state-trigger-on-a-literal-time": (
        "species X, Y, Z; X = 0; Y = 0; Z = 1; a = 2; k = 0.5\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*Z, time >= 3, 0)\n"
        "E1: at (X >= 6): Z = 2*Z\n",
        "a",
    ),
}


@pytest.mark.parametrize("kink", sorted(KINKS))
def test_a_fixed_switch_the_event_does_not_commute_with_is_refused(kink):
    """Each returned a number, the derivative from one side of the kink. The
    first cut of the refusal missed them: it looked for the jump at the
    pre-event state only, against the largest flux in the model, by moving the
    time, within one nudge."""
    text, param = KINKS[kink]
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param])
    with pytest.raises(Exception, match="at a fixed time.*issue #767"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("cmp", [">=", ">"])
def test_a_fixed_switch_an_ulp_before_the_shared_time_is_refused(cmp):
    """t1 + t2 = 0.1 + 0.2 is one ulp above the literal 0.3. A fixed switch at
    0.3, and an event and a fitted switch at t1 + t2: three crossings on one
    instant. The event's flows have to be read after the fixed switch and before
    the fitted one, which moving the clock cannot do. Read a nudge before the
    instant they were before both, and dW/dt1 came back −0.315 for −0.105, the
    derivative from the other side of a kink one ulp away."""
    text = (
        "species X, Y, W; X = 0; Y = 0; W = 0; a = 2; k = 0.5; q = 0.7; t1 = 0.1; t2 = 0.2\n"
        "J0: -> X; a\n"
        f"J1: -> Y; piecewise(k*X, time {cmp} t1 + t2, 0)\n"
        "J2: -> W; piecewise(q*X, time >= 0.3, 0)\n"
        "E1: at (time >= t1 + t2): X = 0.5*X\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["t1"])
    with pytest.raises(Exception, match="shares its instant with another switch"):
        sim.run(sample_times=[0.0, 0.2, 0.5, 0.75], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("ulps", [-700, -300, 150, 300, 700])
def test_a_fitted_switch_near_the_event_does_not_hide_a_fixed_one(ulps):
    """A fixed switch at the literal 3 on W, the event at tau = 3, and a fitted
    switch on Y a few hundred ulp away that moves with the event. The fixed one
    is a kink in tau: X is halved before W's law turns on, or after. An earlier
    cut skipped the search for a fixed switch wherever a fitted one was in
    reach, and returned dW/dtau from one side."""
    text = (
        "species X, Y, W; X = 0; Y = 0; W = 0; a = 2; k = 0.5; q = 0.7; tau = 3; off = 0\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*X, time >= tau + off, 0)\n"
        "J2: -> W; piecewise(q*X, time >= 3, 0)\n"
        "E1: at (time >= tau): X = 0.5*X\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("off", ulps * float(np.spacing(3.0)))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(Exception, match="shares its instant with another switch"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("switch", "event"),
    [("tau + 1e10*z", "3 + 1e10*z"), ("3 + 1e10*z", "tau + 1e10*z")],
    ids=["tau-moves-the-switch", "tau-moves-the-event"],
)
def test_one_column_with_a_large_shift_does_not_hide_another(switch, event):
    """z = 0 moves both times by 1e10 per unit and tau moves one of them by 1.
    Under tau they come apart, and that is a kink. An earlier cut compared two
    shifts against the largest shift of any column, where 1 beside 0 is nothing
    beside 1e10, and returned a number."""
    text = (
        "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = 3; z = 0\n"
        "J0: -> X; a\n"
        f"J1: -> Y; piecewise(k*X, time >= {switch}, 0)\n"
        f"E1: at (time >= {event}): X = 0\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau", "z"])
    with pytest.raises(Exception, match="issue #767"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("tau", [1e-3, 1e-9])
def test_a_fixed_switch_under_an_event_early_in_time_is_refused(tau):
    """The same kink at t = 1e-3 and 1e-9. Whether a time moves is asked against
    the time itself: per unit relative change of tau the event moves by tau,
    which is small only beside a time unit."""
    text = (
        f"species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = {tau!r}\n"
        "J0: -> X; a\n"
        f"J1: -> Y; piecewise(k*X, time >= {tau!r}, 0)\n"
        "E1: at (time >= tau): X = 0\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(Exception, match="at a fixed time.*issue #767"):
        sim.run(sample_times=[0.0, 0.5 * tau, 2.0 * tau, 3.0 * tau], rtol=1e-10, atol=1e-30)


@pytest.mark.parametrize("atol", [1e-8, "tracking"])
def test_a_jump_beside_a_species_at_1e13_whatever_the_tolerance(atol):
    """A reset at tau beside a rate law that switches at the literal 3, and a
    species at 1e13 that does nothing. A floor under the jump that read the
    absolute tolerance let it through where the tolerance tracks the state: the
    ceiling it read was 1000. The jump is carried to a bracket of zero width
    now, and has no floor."""
    text = (
        "species X, Y, W; X = 0; Y = 0; W = 1e13; a = 2; k = 0.5; kw = 0; tau = 3\n"
        "J0: -> X; a\n"
        "JW: W -> ; kw*W\n"
        "J1: -> Y; piecewise(k*X, time >= 3, 0)\n"
        "E1: at (time >= tau): X = 0\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(Exception, match="at a fixed time.*issue #767"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=atol)


@pytest.mark.parametrize(("t0", "off"), [(0.05, 5e-14), (0.007, 2e-14)])
def test_a_fitted_switch_near_an_early_event_does_not_hide_a_fixed_one(t0, off):
    """Below t = 1 the detector's twelve digits are narrower than the reach of
    the event's root, so a fitted switch can be within that reach of the event
    and not grouped with the fixed switch on the event's instant. The search
    for a fixed switch was skipped wherever a record was in reach. It now skips
    only a crossing within one instant of a record on the same clock."""
    text = (
        f"species X, Y, W; X = 0; Y = 0; W = 0; a = 2; k = 0.5; q = 0.7; tau = {t0!r}; off = 0\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*X, time >= tau + off, 0)\n"
        f"J2: -> W; piecewise(q*X, time >= {t0!r}, 0)\n"
        "E1: at (time >= tau): X = 0.5*X\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("off", off)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    with pytest.raises(Exception, match="issue #767"):
        sim.run(sample_times=[0.0, 0.5 * t0, 2 * t0, 3 * t0], rtol=1e-10, atol=1e-12)


BACK_TO_BACK = (
    "species X; X = 0; kd = 0.3; R1 = 1.5; R2 = 0.7; D = 20; t1 = 3; dur = 2; t2 = 5\n"
    "J0: X -> ; kd*X\n"
    "J1: -> X; piecewise(R1, time >= t1 && time < t1 + dur, 0)"
    " + piecewise(R2, time >= t2 && time < t2 + dur, 0)\n"
    "E1: at (time >= t2): X = X + D\n"
)


@pytest.mark.parametrize("params", [["dur"], ["t1"], ["t2"], ["t1", "t2", "dur"]])
def test_back_to_back_infusions_with_a_bolus_on_the_boundary(params):
    """Control. One infusion ends at t1 + dur = 5, where the next begins and a
    bolus is given: three crossings on one instant, two of them grouped by the
    detector. The bolus adds to X and neither rate reads X, so everything
    commutes and each derivative exists. With u = T − 5 and v = T − 3:

        dX/d(dur) = R1·e^(−kd·u)
        dX/d(t1)  = R1·(e^(−kd·u) − e^(−kd·v))
        dX/d(t2)  = (D·kd − R2)·e^(−kd·u)

    An earlier cut refused every event on a group of switches. The group is
    asked for its jump now, like any other switch."""
    kd, r1, r2, dose = 0.3, 1.5, 0.7, 20.0
    end = 6.0
    u, v = end - 5.0, end - 3.0
    want = {
        "dur": r1 * np.exp(-kd * u),
        "t1": r1 * (np.exp(-kd * u) - np.exp(-kd * v)),
        "t2": (dose * kd - r2) * np.exp(-kd * u),
    }
    _x, s = _sens(BACK_TO_BACK, params)
    np.testing.assert_allclose(s["X"], [want[p] for p in params], rtol=1e-6)


def test_an_onset_of_second_order_under_a_dose_from_zero():
    """Control. The rate law turns on at the literal 3 as k·X·(time − 3)², and
    X is 0 until a dose at tau = 3. Nothing steps: dY/dtau = −k·D·(tau − 3)² = 0.
    The second difference of the law is 0 before the dose and 2·k·D·q² after,
    all of it the onset's curvature, and it has to stay under what the outer
    reads allow."""
    text = (
        "species X, Y; X = 0; Y = 0; k = 0.5; tau = 3\n"
        "J1: -> Y; piecewise(k*X*(time - 3)^2, time >= 3, 0)\n"
        "E1: at (time >= tau): X = X + 2\n"
    )
    _x, s = _sens(text, ["tau"])
    assert s["Y"][0] == pytest.approx(0.0, abs=1e-7)


def test_an_onset_of_second_order_at_a_tolerance_of_1e_30():
    """Control. The square onset of the smooth cases, with an absolute tolerance
    far under anything in the model. A floor that read the tolerance refused
    it."""
    text = SMOOTH.format(law="piecewise(k*X*(time - 3)^2, time >= 3, 0)")
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"]).run(
        sample_times=TIMES, rtol=1e-10, atol=1e-30
    )
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y"), 0]
    assert got == pytest.approx(0.0, abs=1e-7)


COUNTER = (
    "species X, Y, Cl; X = 0; Y = 0; Cl = 0; a = 2; k = 0.5; tau = {tau!r}\n"
    "J0: -> X; a\n"
    "Jc: -> Cl; 1\n"
    "J1: -> Y; piecewise(k*X, Cl >= tau, 0)\n"
    "E1: at ({trigger}): X = 0.5*X\n"
)


@pytest.mark.parametrize(
    ("trigger", "tau"),
    [
        ("time >= tau", 1.8695001483274925),
        ("Cl >= tau", 1.8695001483274925),
        ("time > tau", 53.83796736812605),
    ],
    ids=["time-trigger", "counter-trigger", "strict-time-trigger"],
)
def test_a_switch_on_a_counter_clock(trigger, tau):
    """The rate law is gated on a counter, Cl = t. X = a·t is halved at tau and
    feeds Y from tau on: with u = T − tau,

        dX/dtau = −a/2        dY/dtau = −k·a·tau/2 − k·a·u/2

    At the stop the integrated counter is within rounding of its threshold, on
    either side, and the stop lands it on the threshold only where that does not
    step over a root, which the event's own is. So the branch the flows were
    read on was a matter of rounding: at the first tau the switch was still to
    come and read as passed, at the second it had been taken and read as not."""
    text = COUNTER.format(trigger=trigger, tau=tau)
    times = [0.0, tau / 2, 1.4 * tau, 1.9 * tau]
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"]).run(
        sample_times=times, rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    s = np.asarray(run.sensitivities)[-1, :, 0]
    u = times[-1] - tau
    assert s[names.index("X")] == pytest.approx(-1.0, rel=1e-7)
    assert s[names.index("Y")] == pytest.approx(-0.5 * tau - 0.5 * u, rel=1e-6)


def test_a_second_fitted_switch_on_the_instant_that_does_not_move_with_the_event():
    """Two fitted switch times with the same value, and an event on the first.
    Under tb the second switch comes apart from the event. Its own jump cannot
    be read apart from the first's by moving the clock, so whether the pair
    commutes cannot be measured, and the run is refused."""
    text = (
        "species X, Y, W; X = 0; Y = 0; W = 0; a = 2; k = 0.5; q = 0.7; ta = 3; tb = 3\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*X, time >= ta, 0)\n"
        "J2: -> W; piecewise(q*X, time >= tb, 0)\n"
        "E1: at (time > ta): X = 0.5*X\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["ta", "tb"])
    with pytest.raises(Exception, match="shares its instant with another switch"):
        sim.run(sample_times=TIMES, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("ulps", [-300, -150, 150, 200])
def test_a_fitted_switch_a_hair_from_the_event_is_not_a_fixed_one(ulps):
    """Control. The switch is at tau + off, a few hundred ulp from the event at
    tau: not one instant, and both move with tau. The event's own trigger time
    is a stop the run knows, and the jump read across it reaches that far, so
    an earlier cut took the fitted switch for a fixed one there and refused.
    The offset is set as a double: written into the model text it would be 0."""
    text = (
        "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = 3; off = 0\n"
        "J0: -> X; a\n"
        "J1: -> Y; piecewise(k*X, time >= tau + off, 0)\n"
        "E1: at (time >= tau): X = 0.5*X\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("off", ulps * float(np.spacing(3.0)))
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"]).run(
        sample_times=TIMES, rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    s = np.asarray(run.sensitivities)[-1, :, 0]
    a, k, tau, c = 2.0, 0.5, 3.0, 0.5
    u = T - tau
    assert s[names.index("X")] == pytest.approx(c * a - a, rel=1e-7)
    assert s[names.index("Y")] == pytest.approx(k * (c * a * u - c * a * tau - a * u), rel=1e-6)


@pytest.mark.parametrize(
    ("text", "param", "want"),
    [
        (
            "species X, Y; X = 0; Y = 0; a = 2; k = 0.5; T0 = 1e9\n"
            "J0: -> X; a\n"
            "J1: -> Y; piecewise(k, time >= T0, 0)\n"
            "E1: at (X >= 5): T0 = time\n",
            "a",
            0.5 * 5.0 / 4.0,
        ),
        (
            "species Y; Y = 0; k = 0.5; T0 = 1e9; tau = 2.5\n"
            "J1: -> Y; piecewise(k, time >= T0, 0)\n"
            "E1: at (time >= tau): T0 = time\n",
            "tau",
            -0.5,
        ),
    ],
    ids=["state-trigger", "time-trigger"],
)
def test_an_event_that_records_its_own_time_is_not_a_fixed_switch(text, param, want):
    """Control. The event writes T0 = time, and a rate law is gated on
    `time >= T0`. At the post-event state that threshold sits exactly on the
    event's instant, but it moves with the event and is no switch at all on the
    trajectory: Y grows at k from the event on, and dY/d(event time) = −k.

    An earlier cut read the right-hand side either side of the instant at the
    post-event state, found the step and refused. A corpus model does this
    (BIOMD0000000675 records the time of START)."""
    _x, s = _sens(text, [param])
    assert s["Y"][0] == pytest.approx(want, rel=1e-6)
