"""An event assignment whose value is not smooth where the event reads it (issue #915).

``u := piecewise(5, time >= T0 + 1, 0)`` assigned by ``at (time >= T0 + 1): B = u``
is 5 whatever T0 is, so dB/dT0 = 0.

The jump differentiates the assigned value by central differences: ∂h/∂p, and
∂h/∂t·∂t*/∂p for a fire time that moves. With a step at the fire instant each is
a difference across the step, −5/(2·δp) and +5/(2·δt), and they cancel only when
the two steps are the same size, which they are when the fire time is the
parameter itself and not when it is the parameter plus one: dB/dT0 came back in
the millions, with no warning. A bend there came back as the mean of its two
slopes, and a value that turns inside the difference as whatever the difference
caught of it.

A central difference across a step, a bend or a turn is not a derivative, and
the run is now refused there. Across a smooth value the difference over half
the step is half as large and the second difference about the point a quarter
as large. A value is refused where either is out by more than a part in 1e3 of
what the value moves by across the step, and by more than 16 ulp of the largest
thing the value reads. Where the value is smooth, the derivative is taken
exactly as it was.

Every expected value is a closed form: nothing changes B after the event.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


def _sens(text, params, t_end, species="B"):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=list(params)).run(
        sample_times=[0.0, 0.5 * t_end, t_end], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    return np.asarray(run.sensitivities)[-1, names.index(species), :]


def _refused(text, params, t_end, where):
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    with pytest.raises(Exception, match="issue #915") as caught:
        sim.run(sample_times=[0.0, 0.5 * t_end, t_end], rtol=1e-10, atol=1e-12)
    assert f"is not smooth in {where}" in str(caught.value)


CARRIED = (
    "species B; B = 0; T0 = {T0}; q = 0.3\n"
    "u := piecewise({value}, time {cmp} T0 + {lag}, {other})\n"
    "E1: at (time {trig} T0 + {lag}): B = u\n"
)


@pytest.mark.parametrize("lag", [1.0, 10.0])
@pytest.mark.parametrize("T0", [0.5, 1.3])
@pytest.mark.parametrize(("cmp", "trig"), [(">=", ">="), (">", ">="), (">=", ">")])
def test_a_step_at_the_fire_instant_is_refused(T0, lag, cmp, trig):
    """The event fires at T0 + lag and assigns a value that steps there. At
    T0 = 0.5 and lag = 1 the column was −3.3e6 for 0."""
    text = CARRIED.format(T0=T0, lag=lag, cmp=cmp, trig=trig, value="5", other="0")
    _refused(text, ["T0"], 2 * (T0 + lag), "the parameter 'T0'")


def test_a_step_between_two_values_that_move_with_the_parameter():
    """``piecewise(5·T0, time > T0 + 1, 3·T0)`` read at T0 + 1 is 3·T0, so
    dB/dT0 = 3. It came back −434779."""
    text = CARRIED.format(T0=1.3, lag=1.0, cmp=">", trig=">=", value="5*T0", other="3*T0")
    _refused(text, ["T0"], 5.0, "the parameter 'T0'")


KINKS = {
    # The step is at the literal 2.3 and the event at T0 + 1 = 2.3: 1.09e6.
    "fixed-step": (
        "species B; B = 0; T0 = 1.3\n"
        "u := piecewise(5, time >= 2.3, 0)\n"
        "E1: at (time >= T0 + 1): B = u\n",
        "T0",
        "the time",
    ),
    # The step is at 2·T0 − 0.3, which is T0 + 1 at T0 = 1.3 and moves twice as
    # fast: −836120.
    "step-at-another-rate": (
        "species B; B = 0; T0 = 1.3\n"
        "u := piecewise(5, time >= 2*T0 - 0.3, 0)\n"
        "E1: at (time >= T0 + 1): B = u\n",
        "T0",
        "the parameter 'T0'",
    ),
    # The event is at the literal 2.3 and T1 moves the step across it.
    "step-moved-across-a-fixed-event": (
        "species B; B = 0; T1 = 2.3\n"
        "u := piecewise(5, time >= T1, 0)\n"
        "E1: at (time >= 2.3): B = u\n",
        "T1",
        "the parameter 'T1'",
    ),
    # A step in a parameter, at the parameter's own value: 1.67e6.
    "step-in-a-parameter": (
        "species B; B = 0; q = 0.3\nE1: at (time >= 1): B = piecewise(1, q >= 0.3, 0)\n",
        "q",
        "the parameter 'q'",
    ),
    # A bend at the fire instant: 0.5, the mean of 0 and 1.
    "bend-in-time": (
        "species B; B = 0; T0 = 1.3\nE1: at (time >= T0 + 1): B = max(time - 2.3, 0)\n",
        "T0",
        "the time",
    ),
}


@pytest.mark.parametrize("kink", sorted(KINKS))
def test_a_value_with_no_derivative_where_it_is_read_is_refused(kink):
    """B is one thing on one side of the parameter's value and another on the
    other, or bends there, so there is no derivative. Each came back as a
    difference across it."""
    text, param, where = KINKS[kink]
    _refused(text, [param], 5.0, where)


@pytest.mark.parametrize("value", ["max(A - 1, 0)", "abs(A - 1)"])
def test_a_bend_in_a_species_the_column_moves_is_refused(value):
    """A decays through 1 at t = 1, where the event reads the value. For
    ``max(A − 1, 0)`` dB/dkd is 0 from one side and −1 from the other, and it
    came back −0.5. For ``abs(A − 1)`` it is +1 and −1, and it came back 0: the
    difference across the bend is 0, which is not a reason to skip it."""
    text = (
        "species B, A; B = 0; A = 2.718281828459045; kd = 1\n"
        "J0: A -> ; kd*A\n"
        f"E1: at (time >= 1): B = {value}\n"
    )
    _refused(text, ["kd"], 5.0, "the species 'A'")


def test_a_bend_a_species_sits_exactly_on_is_refused():
    """X is held at 3 by its initial value x0, and the event reads
    ``abs(X − 3)``: dB/dx0 is +1 from one side and −1 from the other. The
    difference across the bend is exactly 0, and the column came back 0."""
    text = (
        "species B, X; B = 0; x0 = 3; X = x0\nJ0: -> B; 0*x0\nE1: at (time >= 1): B = abs(X - 3)\n"
    )
    _refused(text, ["x0"], 5.0, "the species 'X'")


TURNS = {
    # 98273.2 for 1e5: the difference is over 2.3e-6 either side, a quarter of
    # the width of the turn.
    "tanh": ("tanh((time - 2.3)/1e-5)", "time >= T0 + 1", 1.3, 5.0),
    # 6.7e-13 for 8.0e-13: the difference is over a whole unit of time either
    # side of 1e6.
    "sin-at-a-late-time": ("1e-12*sin(time)", "time >= T0 + 1e6", 1.0, 2e6),
}


@pytest.mark.parametrize("turn", sorted(TURNS))
def test_a_value_that_turns_inside_the_difference_is_refused(turn):
    """A smooth value, but not across a millionth of the fire time: the central
    difference was 1.7% and 16% off."""
    value, trigger, T0, t_end = TURNS[turn]
    text = f"species B; B = 0; T0 = {T0}\nE1: at ({trigger}): B = {value}\n"
    _refused(text, ["T0"], t_end, "the time")


AT_THE_TRIGGER = {
    "step": ("piecewise(5, X > thr, 0)", "the parameter 'thr'"),
    "bend": ("max(X - thr, 0)", "the parameter 'thr'"),
}


@pytest.mark.parametrize("shape", sorted(AT_THE_TRIGGER))
def test_a_value_that_steps_or_bends_on_its_own_state_trigger_is_refused(shape):
    """The event fires as X passes thr and assigns a value that steps or bends
    in X at thr. The columns were 0, which is right: ∂h/∂X·dX/dthr and ∂h/∂thr
    are differences across the same step, over the same fraction of X and of
    thr, and cancel. With X read as ``2·X`` they would not. Refused now, where
    it was right by that."""
    value, where = AT_THE_TRIGGER[shape]
    text = (
        f"species B, X; B = 0; X = 0; k = 1; thr = 2\nJ0: -> X; k\nE1: at (X > thr): B = {value}\n"
    )
    _refused(text, ["k", "thr"], 5.0, where)


REFUSED_WHERE_IT_WAS_RIGHT = {
    # 3·max(time − T0 − 1, 0) read at T0 + 1 is 0 for every T0: the bend moves
    # with the event. The two differences cancelled, to −3e-10.
    "a-bend-the-fire-time-carries": (
        "species B; B = 0; T0 = 1.0\n"
        "u := 3*max(time - T0 - 1, 0)\n"
        "E1: at (time >= T0 + 1): B = u\n",
        "the parameter 'T0'",
    ),
    # (time − 2.3)³ read at 2.3: smooth, with slope and curvature both 0 there.
    # The difference is h² over the whole step and h²/4 over half of it, which
    # is what a value that turns inside the step shows. 5e-12 for 0. It reads
    # nothing but the time, so there is no larger thing to round against.
    "a-flat-inflection": (
        "species B; B = 0; T0 = 1.3\nE1: at (time >= T0 + 1): B = (time - 2.3)^3\n",
        "the time",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED_WHERE_IT_WAS_RIGHT))
def test_what_is_refused_where_the_differences_happened_to_cancel(case):
    """Each has a derivative, 0, and the run returned it. A bend or a turn
    exactly at the fire instant is not told from one that makes the difference
    wrong, so these are refused with the rest."""
    text, where = REFUSED_WHERE_IT_WAS_RIGHT[case]
    _refused(text, ["T0"], 5.0, where)


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("2*time + sin(time)", 2.0 + np.cos(2.3)),
        ("time", 1.0),
        ("piecewise(5, time >= T0 + 3, 0)", 0.0),
        ("piecewise(5, time >= 0.2, 0)", 0.0),
        ("(time - 2.3)^2", 0.0),
        ("q*time", 0.3),
    ],
    ids=["smooth", "the-time-itself", "a-step-later", "a-step-long-past", "a-minimum", "linear"],
)
def test_a_value_that_is_smooth_at_the_fire_instant_is_as_it_was(value, want):
    """Control. A smooth value moves with the fire time by ∂h/∂t (issue #735),
    and a step somewhere else is not at the fire instant."""
    text = f"species B; B = 0; T0 = 1.3; q = 0.3\nu := {value}\nE1: at (time >= T0 + 1): B = u\n"
    assert _sens(text, ["T0"], 5.0)[0] == pytest.approx(want, rel=1e-6, abs=1e-8)


def test_a_step_no_requested_parameter_reaches_is_not_asked():
    """Control. The value steps at the fire instant, in the time and in T0. Only
    q is requested, which moves neither: dB/dq = 1."""
    text = (
        "species B; B = 0; T0 = 1.3; q = 0.3\n"
        "u := piecewise(5, time >= T0 + 1, 0)\n"
        "E1: at (time >= T0 + 1): B = u + q\n"
    )
    assert _sens(text, ["q"], 5.0)[0] == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("value", ["floor(X) + k", "abs(X - 3) + k", "max(X - 3, 0) + k"])
def test_a_step_in_a_species_no_column_moves_is_not_asked(value):
    """Control. X sits at 3, on a step of floor(X) and on a bend of the other
    two, and no parameter moves it: dB/dk = 1."""
    text = f"species B, X; B = 0; X = 3; k = 1\nJ0: -> X; 0*k\nE1: at (time >= 1): B = {value}\n"
    assert _sens(text, ["k"], 5.0)[0] == pytest.approx(1.0, rel=1e-9)


def test_a_fire_time_that_moves_much_faster_than_its_parameter():
    """Control. The event fires at 100·(T0 − 0.99), and the value turns fifty
    times in a unit of time: smooth across a millionth of the fire time."""
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
    dB/dq = Y + 2 whatever q is."""
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


@pytest.mark.parametrize("T0", [0.5, 1.0, 1.3, 2.0, 4.0])
def test_the_time_since_the_parameter_is_straight(T0):
    """Control. ``time − T0`` read at T0 + 0.001 is 0.001 for every T0. It is a
    difference of two numbers a thousand times its size, straight to their
    rounding and to no more, at a power of two as anywhere else."""
    text = f"species B; B = 0; T0 = {T0}\nE1: at (time >= T0 + 0.001): B = time - T0\n"
    assert _sens(text, ["T0"], T0 + 3.0)[0] == pytest.approx(0.0, abs=1e-8)


SMOOTH_ROWS = [
    # value, trigger, T0, t_end, want, rel: values the value's size or the fire
    # time makes hard to read, each right as it was.
    ("sin(time) + T0", "time >= T0 + 10", 1e-9, 20.0, 1.0 + np.cos(10.0), 1e-4),
    ("sin(time) + T0", "time >= T0 + 100", 1e-6, 200.0, 1.0 + np.cos(100.0), 1e-4),
    ("(time - 1e8)*(1 + T0) + q", "time >= T0 + 1e8", 0.0, 2e8, 1.0, 1e-6),
    ("sin(3*time/T0) + q*time", "time >= T0", 1e-6, 1e-5, 0.3, 1e-4),
]


@pytest.mark.parametrize("row", range(len(SMOOTH_ROWS)))
def test_a_smooth_value_at_a_small_parameter_or_a_late_time(row):
    """Control. A parameter of 1e-9 beside a fire time of 10, a fire time of
    1e8, and one of 1e-6."""
    value, trigger, T0, t_end, want, rel = SMOOTH_ROWS[row]
    text = f"species B; B = 0; T0 = {T0!r}; q = 0.3\nE1: at ({trigger}): B = {value}\n"
    assert _sens(text, ["T0"], t_end)[0] == pytest.approx(want, rel=rel)


def test_a_value_read_at_a_state_trigger():
    """Control. ``time − thr/k`` read as X = k·t passes thr is 0 for every k and
    thr."""
    text = (
        "species B, X; B = 0; X = 0; k = 1; thr = 2\n"
        "J0: -> X; k\n"
        "E1: at (X >= thr): B = time - thr/k\n"
    )
    np.testing.assert_allclose(_sens(text, ["k", "thr"], 5.0), [0.0, 0.0], atol=1e-8)


@pytest.mark.parametrize("value", ["piecewise(1, X > 3, -1, X < 3, 0)", "tanh((X - 3)/3e-6)"])
def test_a_step_or_a_turn_a_species_sits_exactly_on_is_refused(value):
    """X is held at 3 by its initial value x0. The step is odd about 3, so the
    value at 3 lies midway between its neighbours and the three readings are in
    line: dB/dx0 came back 333333 for the step, which has no derivative, and
    253865 for the turn, whose is 333333."""
    text = (
        f"species B, X; B = 0; x0 = 3; X = x0\nJ0: -> B; 0*x0\nE1: at (time >= 1): B = {value}\n"
    )
    _refused(text, ["x0"], 5.0, "the species 'X'")


@pytest.mark.parametrize("x0", [0.1, 0.4, 0.7, 1.0, 1.9, 2.8, 4.0])
def test_a_value_that_is_a_difference_of_larger_things(x0):
    """Control. ``Atot − X − Abound`` with Atot at 1000 is 1.5 − X and rounds
    by an ulp of 1000, which is a thousand times what its own size says: the
    readings are not in line to that. dB/dx0 = −1. A first cut of this fix
    refused 12 of 40 such values."""
    text = (
        f"species B, X; B = 0; x0 = {x0}; X = x0; Atot = 1000; Abound = 998.5\n"
        "J0: -> B; 0*x0\n"
        "E1: at (time >= 1): B = Atot - X - Abound\n"
    )
    assert _sens(text, ["x0"], 5.0)[0] == pytest.approx(-1.0, rel=1e-6)


@pytest.mark.parametrize(
    ("value", "param", "want", "rel"),
    [
        ("1 - exp(-k1*time)", "k1", 2.3 * np.exp(-0.001 * 2.3), 1e-6),
        # 1e-10, read from a value that is 1 and rounds by 1e-16: three digits.
        ("X^3/(8 + X^3)", "x0", 3 * 8 * 700.0**2 / (8 + 700.0**3) ** 2, 2e-3),
        ("exp(-320000*k1*time)", "k1", 0.0, 0.0),
    ],
    ids=["slow-first-order", "saturated", "vanishing"],
)
def test_a_value_that_barely_moves_across_the_difference(value, param, want, rel):
    """Control. A slow first-order term, 0.0023 beside the 1 it is taken from;
    a Hill function at X = 700, where it is 1 to eight digits; and a value that
    has decayed to 1e-320, where a double has four digits left. Each moves
    across the difference by little more than it rounds by, and the derivative
    is that small."""
    text = (
        "species B, X; B = 0; x0 = 700; X = x0; k1 = 0.001; T0 = 1.3\n"
        "J0: -> B; 0*x0\n"
        f"E1: at (time >= T0 + 1): B = {value}\n"
    )
    assert _sens(text, [param], 5.0)[0] == pytest.approx(want, rel=rel, abs=1e-300)


@pytest.mark.parametrize("T0", [1.0, 3.0, 7.0])
def test_a_minimum_at_a_fire_time_that_is_a_power_of_two(T0):
    """Control. ``(time − c)²`` read at its minimum, with c = T0 + 1 at 2, 4 and
    8, where the time rounds differently on either side: the difference is the
    rounding of the two readings, and it is as far out over half the step."""
    text = f"species B; B = 0; T0 = {T0}\nE1: at (time >= T0 + 1): B = (time - {T0 + 1.0})^2\n"
    assert _sens(text, ["T0"], 2 * (T0 + 1.0))[0] == pytest.approx(0.0, abs=1e-8)


def test_a_bend_in_time_under_a_state_trigger_no_column_moves():
    """Control. The event fires as X = k·t passes thr, at t = 2, and assigns a
    value that steps in time there. Only z is requested, which moves nothing
    the trigger reads: the fire time does not move, and the step is not asked
    about. A state trigger takes the path a moving fire time takes, with every
    shift 0."""
    text = (
        "species B, X, W; B = 0; X = 0; W = 1; k = 1; thr = 2; z = 0.5\n"
        "J0: -> X; k\n"
        "J1: W -> ; z*W\n"
        "E1: at (X >= thr): B = piecewise(5, time >= 2.0, 0)\n"
    )
    assert _sens(text, ["z"], 5.0)[0] == pytest.approx(0.0, abs=1e-12)
