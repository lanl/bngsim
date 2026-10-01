"""Forward sensitivities across events that fire at one instant (issue #722).

A same-instant batch is not one simultaneous map. It executes one fire at a
time: highest priority first, a tie broken at random, a
``fromTrigger = false`` assignment reading the state the earlier fires left,
and a non-persistent event cancelled when an earlier fire makes its trigger
false. The state has always followed that. The sensitivity jump walked the
root-detected events in declaration order and took every derivative at the
pre-batch state, so:

- two events assigning one species gave it the derivative of the one declared
  last, not the one that executed last;
- a ``fromTrigger = false`` value was differentiated at a state it never read;
- a cancelled event still wrote its row.

All silent, and reordering the declarations changed the gradient. The jump now
composes the batch in the order it executed.

Every expected value here is a closed form from SBML L3 event semantics. The
dynamics are ``X' = p``, so ``X(t) = p·t`` until an event writes X.
"""

from __future__ import annotations

import itertools

import bngsim
import numpy as np
import pytest

TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]


def _run(text, params, **kw):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
        sample_times=TIMES, rtol=1e-10, atol=1e-12, **kw
    )
    names = list(run.species_names)
    x = dict(zip(names, np.asarray(run.species)[-1], strict=True))
    s = dict(zip(names, np.asarray(run.sensitivities)[-1], strict=True))
    return x, s


def _model(head, events, order):
    """`events` in the declaration order `order`."""
    return head + "".join(events[i] + "\n" for i in order)


HEAD = "species X, Y, Z; X = 0; Y = 0; Z = 0; p = 1; q = 0.5\nJ0: -> X; p\n"


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["survivor-first", "survivor-last"])
def test_two_events_on_one_species_follow_the_execution_order(order):
    """Ehi (priority 2) executes first and Elo last, so X(2) = 3p and
    X(4) = 3p + 2p whatever the order they are declared in. Declared with Ehi
    last, dX/dp came out 7: Ehi's 5 + 2."""
    events = [
        "Elo: at (time >= 2), priority = 1: X = 3*p",
        "Ehi: at (time >= 2), priority = 2: X = 5*p",
    ]
    x, s = _run(_model(HEAD, events, order), ["p"])
    assert x["X"] == pytest.approx(5.0, rel=1e-9)
    assert s["X"][0] == pytest.approx(5.0, rel=1e-8)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["reader-first", "reader-last"])
@pytest.mark.parametrize("reset", ["0", "3*p"])
def test_a_value_read_at_execution_sees_the_earlier_fire(order, reset):
    """E1 (priority 2) writes X, then E0 reads X with ``fromTrigger = false``:
    Y = X_new·p + q. With X := 0 that is q, and dY/dp = 0 where the old jump
    gave X⁻ + p·dX⁻/dp = 4. With X := 3p it is 3p² + q, and dY/dp = 6."""
    events = [
        "E0: at (time >= 2), priority = 1, fromTrigger = false: Y = X*p + q",
        f"E1: at (time >= 2), priority = 2: X = {reset}",
    ]
    x, s = _run(_model(HEAD, events, order), ["p", "q"])
    x_new, dx_new = (0.0, 0.0) if reset == "0" else (3.0, 3.0)
    assert x["Y"] == pytest.approx(x_new + 0.5, rel=1e-9)
    np.testing.assert_allclose(s["Y"], [x_new + dx_new, 1.0], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(s["X"], [dx_new + 2.0, 0.0], rtol=1e-6, atol=1e-8)


def test_a_value_frozen_at_the_trigger_time_still_reads_the_pre_batch_state():
    """The same pair with the default ``fromTrigger = true``: Y = X⁻·p + q with
    X⁻ = 2p, so dY/dp = 4p. This was right before and has to stay right."""
    events = [
        "E0: at (time >= 2), priority = 1: Y = X*p + q",
        "E1: at (time >= 2), priority = 2: X = 0",
    ]
    for order in ((0, 1), (1, 0)):
        x, s = _run(_model(HEAD, events, order), ["p", "q"])
        assert x["Y"] == pytest.approx(2.5, rel=1e-9)
        np.testing.assert_allclose(s["Y"], [4.0, 1.0], rtol=1e-6)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["cancelled-first", "cancelled-last"])
@pytest.mark.parametrize("trigger", ["X >= 3*p", "X >= thr"])
def test_a_cancelled_event_writes_no_row(order, trigger):
    """Both events have one trigger. EX (priority 2) resets X to 0, which makes
    the trigger false again, and EY is not persistent, so it never executes:
    Y stays 0 and so does every Y column. The old jump wrote EY's 5 into it.

    ``X >= 3*p`` is crossed at t = 3 for every p, and X(4) = p. ``X >= thr``
    is crossed at thr/p, which moves, and X(4) = 4p - thr."""
    events = [
        f"EY: at ({trigger}), priority = 1, persistent = false: Y = 5*p",
        f"EX: at ({trigger}), priority = 2: X = 0",
    ]
    x, s = _run(_model(HEAD + "thr = 3\n", events, order), ["p", "thr"])
    assert x["Y"] == 0.0
    np.testing.assert_allclose(s["Y"], [0.0, 0.0], atol=1e-9)
    assert x["X"] == pytest.approx(1.0, rel=1e-8)
    want = [1.0, 0.0] if trigger == "X >= 3*p" else [4.0, -1.0]
    np.testing.assert_allclose(s["X"], want, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_a_chain_of_three_fires_is_composed_in_order(order):
    """E1 writes X = 2p, E2 then reads it for Y = X + q·X, and E3 reads both for
    Z = Y·X. So Y = 2p(1 + q) and Z = 4p²(1 + q), in any declaration order."""
    events = [
        "E3: at (time >= 2), priority = 1, fromTrigger = false: Z = Y*X",
        "E1: at (time >= 2), priority = 3: X = 2*p",
        "E2: at (time >= 2), priority = 2, fromTrigger = false: Y = X + q*X",
    ]
    x, s = _run(_model(HEAD, events, order), ["p", "q"])
    assert x["Z"] == pytest.approx(6.0, rel=1e-9)
    np.testing.assert_allclose(s["Y"], [3.0, 2.0], rtol=1e-6)
    np.testing.assert_allclose(s["Z"], [12.0, 4.0], rtol=1e-6)


def test_one_events_assignments_do_not_read_each_other():
    """Within ONE event the assignments are simultaneous, at execution time
    too. X = Y·p and Y = X·p each read the other from before the event: with
    X⁻ = 2p and Y⁻ = 0.5, X becomes 0.5p and Y becomes 2p². So dY/dp = 4 and
    dX/dp(4) = 0.5 + 2. Right before, and the composition must not break it
    whichever of the two rows is formed first."""
    text = (
        "species X, Y; X = 0; Y = 0.5; p = 1\nJ0: -> X; p\n"
        "E: at (time >= 2), fromTrigger = false: X = Y*p, Y = X*p\n"
    )
    x, s = _run(text, ["p"])
    assert x["Y"] == pytest.approx(2.0, rel=1e-9)
    assert x["X"] == pytest.approx(2.5, rel=1e-9)
    assert s["Y"][0] == pytest.approx(4.0, rel=1e-6)
    assert s["X"][0] == pytest.approx(2.5, rel=1e-6)


def test_a_random_tie_takes_the_gradient_of_the_order_that_was_drawn():
    """Two events of equal priority write X at one instant, and the one drawn
    last wins (SBML L3 §4.11.6). X(4) = c·p + 2p with c = 3 or 5 by the draw,
    so at p = 1 the gradient is X(4) itself. The old jump gave 7, the event
    declared last, whichever was drawn."""
    text = HEAD + "Ea: at (time >= 2): X = 3*p\nEb: at (time >= 2): X = 5*p\n"
    seen = set()
    for seed in range(12):
        x, s = _run(text, ["p"], seed=seed)
        assert x["X"] in (pytest.approx(5.0, rel=1e-9), pytest.approx(7.0, rel=1e-9))
        assert s["X"][0] == pytest.approx(x["X"], rel=1e-8)
        seen.add(round(x["X"]))
    assert seen == {5, 7}, "the seeds drew only one order, so the test shows nothing"


def test_a_moving_crossing_is_composed_too():
    """Both events trigger on ``X >= thr``, at t* = thr/p, which moves with p
    and thr. E1 writes X = a·X + c·time, which reads the clock. E2 then reads
    the new X and the clock: Y = X·b + time. Afterwards X grows on from its
    new value and does not reach thr again. With u = a + c/p:

        Y(T) = b·thr·u + thr/p          X(T) = thr·u + p·T − thr

    The old jump differentiated E2 at the X from before E1."""
    p, thr, a, b, c, T = 1.0, 3.0, 0.1, 3.0, 0.1, TIMES[-1]
    u = a + c / p
    text = (
        "species X, Y; X = 0; Y = 0; p = 1; thr = 3; a = 0.1; b = 3; c = 0.1\n"
        "J0: -> X; p\n"
        "E2: at (X >= thr), priority = 1, fromTrigger = false: Y = X*b + time\n"
        "E1: at (X >= thr), priority = 2: X = a*X + c*time\n"
    )
    x, s = _run(text, ["p", "thr", "a", "b", "c"])
    assert x["Y"] == pytest.approx(b * thr * u + thr / p, rel=1e-8)
    assert x["X"] == pytest.approx(thr * u + p * T - thr, rel=1e-8)
    want_y = [-(b * c + 1.0) * thr / p**2, b * u + 1 / p, b * thr, thr * u, b * thr / p]
    want_x = [T - thr * c / p**2, u - 1.0, thr, 0.0, thr / p]
    np.testing.assert_allclose(s["Y"], want_y, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(s["X"], want_x, rtol=1e-5, atol=1e-7)


def test_an_ordered_pair_at_the_start_of_the_run():
    """Two ``t0 = false`` events fire at t_start: E1 sets A = 8, then E2 copies
    the new A into B. Neither value depends on A0 any more, so every A0 column
    is 0 from the first sample on. The old jump gave dB/dA0 = 1 (issue #717's
    review)."""
    text = (
        "species A, B; A0 = 10; A = A0; B = 0; k1 = 0.3\n"
        "J0: A -> B; k1*A\n"
        "E1: at (time >= 0), t0 = false, priority = 2, fromTrigger = false: A = 8\n"
        "E2: at (time >= 0), t0 = false, priority = 1, fromTrigger = false: B = A\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["A0", "k1"]).run(
        sample_times=TIMES, rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)
    s = np.asarray(run.sensitivities)
    t = np.asarray(TIMES)
    a, b = names.index("A"), names.index("B")
    np.testing.assert_allclose(x[:, a], 8.0 * np.exp(-0.3 * t), rtol=1e-8)
    np.testing.assert_allclose(x[:, b], 8.0 + 8.0 * (1.0 - np.exp(-0.3 * t)), rtol=1e-8)
    np.testing.assert_allclose(s[:, a, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(s[:, b, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(s[:, b, 1], 8.0 * t * np.exp(-0.3 * t), rtol=1e-6, atol=1e-9)
