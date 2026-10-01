"""An event assignment whose value steps in time at its own fire instant (issue #915).

``u := piecewise(5, time >= T0 + 1, 0)`` assigned by ``at (time >= T0 + 1): B = u``
is 5 whatever T0 is: the step moves with the event, so dB/dT0 = 0.

The jump differentiated the value by ∂h/∂p and by ∂h/∂t·∂t*/∂p separately. With
a step at the fire instant each is a difference across the step, −5/(2·δp) and
+5/(2·δt), and they cancel only when the two steps are the same size, which they
are when the fire time is the parameter itself and not when it is the parameter
plus one: dB/dT0 came back in the millions, with no warning.

The value is now differentiated along the column's own direction, the parameter
and the fire time moving together, which a step the fire time carries never
crosses. A step it does not carry is a kink in the parameter, and the run is
refused.

Every expected value is a closed form: nothing changes B after the event.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


def _sens(text, params, t_end):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=list(params)).run(
        sample_times=[0.0, 0.5 * t_end, t_end], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    return np.asarray(run.sensitivities)[-1, names.index("B"), :]


CARRIED = (
    "species B; B = 0; T0 = {T0}; q = 0.3\n"
    "u := piecewise({value}, time {cmp} T0 + {lag}, 0)\n"
    "E1: at (time {trig} T0 + {lag}): B = u\n"
)


@pytest.mark.parametrize("lag", [1.0, 10.0])
@pytest.mark.parametrize("T0", [0.5, 1.3])
@pytest.mark.parametrize(("cmp", "trig"), [(">=", ">="), (">", ">="), (">=", ">")])
def test_a_step_the_fire_time_carries_is_not_crossed(T0, lag, cmp, trig):
    """The event fires at T0 + lag and assigns a value that steps there. B is 5
    after it (0 where the value's own comparison is strict and the trigger is
    not), for every T0. At T0 = 0.5 and lag = 1 the column was 3.3e6."""
    text = CARRIED.format(T0=T0, lag=lag, cmp=cmp, trig=trig, value="5")
    assert _sens(text, ["T0"], 2 * (T0 + lag))[0] == pytest.approx(0.0, abs=1e-6)


def test_a_value_that_also_moves_smoothly_with_the_parameter():
    """Past its step the value is 3·(time − T0 − 1) + 1 + q·T0, which the event
    reads at time = T0 + 1: B = 1 + q·T0, so dB/dT0 = q and dB/dq = T0."""
    text = CARRIED.format(
        T0=1.3, lag=1.0, cmp=">=", trig=">=", value="3*(time - T0 - 1) + 1 + q*T0"
    )
    got = _sens(text, ["T0", "q"], 5.0)
    assert got[0] == pytest.approx(0.3, rel=1e-6)
    assert got[1] == pytest.approx(1.3, rel=1e-6)


KINKS = {
    # The step is at the literal 2.3 and the event at T0 + 1 = 2.3.
    "fixed-step": (
        "species B; B = 0; T0 = 1.3\n"
        "u := piecewise(5, time >= 2.3, 0)\n"
        "E1: at (time >= T0 + 1): B = u\n",
        "T0",
    ),
    # The step is at 2·T0 − 0.3, which is T0 + 1 at T0 = 1.3 and moves twice as fast.
    "step-at-another-rate": (
        "species B; B = 0; T0 = 1.3\n"
        "u := piecewise(5, time >= 2*T0 - 0.3, 0)\n"
        "E1: at (time >= T0 + 1): B = u\n",
        "T0",
    ),
    # The event is at the literal 2.3 and T1 moves the step across it.
    "step-moved-across-a-fixed-event": (
        "species B; B = 0; T1 = 2.3\n"
        "u := piecewise(5, time >= T1, 0)\n"
        "E1: at (time >= 2.3): B = u\n",
        "T1",
    ),
}


@pytest.mark.parametrize("kink", sorted(KINKS))
def test_a_step_the_fire_time_does_not_carry_is_refused(kink):
    """B is 5 on one side of the parameter's value and 0 on the other, so there
    is no derivative. The first two came back as −5/(2·δ) + 5/(2·δ') and the
    third as −5/(2·δ), each a difference across the step."""
    text, param = KINKS[kink]
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param])
    with pytest.raises(Exception, match="issue #915"):
        sim.run(sample_times=[0.0, 2.0, 5.0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("2*time + sin(time)", 2.0 + np.cos(2.3)),
        ("time", 1.0),
        ("piecewise(5, time >= T0 + 3, 0)", 0.0),
        ("piecewise(5, time >= 0.2, 0)", 0.0),
    ],
    ids=["smooth", "the-time-itself", "a-step-later", "a-step-long-past"],
)
def test_a_value_with_no_step_at_the_fire_instant_is_as_it_was(value, want):
    """Control. A smooth value moves with the fire time by ∂h/∂t (issue #735),
    and a step somewhere else is not at the fire instant."""
    text = f"species B; B = 0; T0 = 1.3\nu := {value}\nE1: at (time >= T0 + 1): B = u\n"
    assert _sens(text, ["T0"], 5.0)[0] == pytest.approx(want, rel=1e-6, abs=1e-8)


def test_a_fire_time_that_moves_much_faster_than_its_parameter():
    """Control. The event fires at 100·(T0 − 0.99), so one part in a million of
    T0 moves it by a hundred parts in a million of its own time. The step along
    the direction is sized by the time it moves, not by the parameter, or the
    difference of sin(50·time) would be taken over too wide a span."""
    text = (
        "species B; B = 0; T0 = 1.0\n"
        "u := sin(50*time) + T0\n"
        "E1: at (time >= 100*(T0 - 0.99)): B = u\n"
    )
    t_fire = 100.0 * (1.0 - 0.99)
    want = 50.0 * np.cos(50.0 * t_fire) * 100.0 + 1.0
    assert _sens(text, ["T0"], 3.0)[0] == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("q", [1.0, 1e-6, 1e-9, 1e-12])
def test_a_parameter_far_smaller_than_the_time_it_moves(q):
    """Control. The event fires at 3 + q and assigns D + q·Y + 2·time, so
    dB/dq = Y + 2 whatever q is. A step along the column's direction is sized
    by the parameter, and at q = 1e-9 it moves the time by 1e-15, under one ulp
    of 3: the difference along it is rounding. It is kept only where it differs
    from the two separate derivatives by more than its own rounding, which a
    step in the value does and a smooth value does not. A first cut of this fix
    returned 7.1 at q = 1e-9 and 0 at 1e-12."""
    text = (
        "species B, Y; B = 0; Y = 3; q = 1; D = 100\n"
        "J0: -> B; 0\n"
        "E1: at (time >= 3 + q): B = D + q*Y + 2*time\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("q", q)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["q"]).run(
        sample_times=[0.0, 2.0, 5.0], rtol=1e-10, atol=1e-12
    )
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("B"), 0]
    assert got == pytest.approx(5.0, rel=1e-6)
