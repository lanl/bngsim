"""A strict comparison and the non-strict one on the same residual, turning true
at one read, execute in sequence (issue #947).

``E1: at (lhs > tau), priority 3: X = 0.3*X`` beside ``E2: at (lhs >= tau),
priority 2: X = 0.5*X``, X from 1. In exact arithmetic ``lhs >= tau`` is true
on the crossing and ``lhs > tau`` only past it, so E2 executes first and E1
after it, reading the X it left: X = 0.15. Read as one batch, E1 executes
first by priority and E2 then overwrites it with the value it froze at the
trigger: X = 0.50.

Which of the two a run gave depended on where the read landed. The stop for a
clock trigger and CVODE's located root both land on the crossing or up to
about a hundred doubles past it, and past it both triggers read true. On main
at aedadd85, over 307 values of tau, the batch came out for 26 with
``0.1*time``, 95 with ``time/3``, 103 with ``time - 1`` and 37 with
``3*time``; only the bare ``time`` gave the sequence everywhere. The SSA
drain did the same. A strict trigger seeded beside its non-strict twin on a
crossing is now held back to a second sub-instant: triggered, its values and
delay read, and executed once everything else of the instant has settled. A
pair that a jump turns true, with no instant between, is one batch as before.

Oracles are closed forms: no reaction moves X, so X after the pair is
0.3·0.5 in sequence. A sensitivity run cannot compose a second jump at one
instant that reads or writes what the first wrote (issue #144), so it refuses
such a held pair by name rather than return the batch's trajectory, which the
plain run no longer gives.
"""

from __future__ import annotations

import random
import re

import bngsim
import numpy as np
import pytest

_RNG = random.Random(947)
# The issue's seven, then values the rounding decides both ways.
TAUS = [3.0, 7.3, 14.0, 14.145775408033089, 5.1, 9.9, 123.456] + [
    10 ** _RNG.uniform(-2, 4) for _ in range(53)
]
SEQUENCE = 0.3 * 0.5
# No reaction moves X (J0's rate is 0), so X after the pair is the events' own
# arithmetic, two IEEE multiplications: all 300 runs of the first test give
# 0.15 to the bit (measured). 1e-12 leaves room for an evaluator that
# reassociates them, and the batch's 0.5 is 2.3 away.
EXACT = 1e-12


def _pair(lhs_gt, lhs_ge, extra=""):
    return (
        "species X; X = 1; tau = 3\nJ0: -> X; 0\n"
        f"E1: at ({lhs_gt}){extra}, priority = 3: X = 0.3*X\n"
        f"E2: at ({lhs_ge}){extra}, priority = 2: X = 0.5*X\n"
    )


def _final(text, tau, t_cross, method="ode", **kw):
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("tau", tau)
    times = [0.0, t_cross / 2, 1.4 * t_cross, 2.1 * t_cross]
    sim = bngsim.Simulator(model, method=method, **kw)
    if method == "ode":
        run = sim.run(sample_times=times, rtol=1e-10, atol=1e-12)
    else:
        run = sim.run(sample_times=times, seed=1)
    return np.asarray(run.species)[-1]


@pytest.mark.parametrize(
    "lhs, scale",
    [("time", 1.0), ("0.1*time", 10.0), ("time/3", 3.0), ("time - 1", None), ("3*time", 1 / 3)],
)
def test_the_pair_executes_in_sequence_whatever_the_rounding(lhs, scale):
    """Every tau gives the sequence, in every spelling of the clock."""
    text = _pair(f"{lhs} > tau", f"{lhs} >= tau")
    got = []
    for tau in TAUS:
        t_cross = tau + 1 if scale is None else scale * tau
        got.append(_final(text, tau, t_cross)[0])
    batched = [
        tau for tau, x in zip(TAUS, got, strict=True) if x != pytest.approx(SEQUENCE, rel=EXACT)
    ]
    assert batched == []


@pytest.mark.parametrize(
    "gt, ge, cross",
    [
        ("time > tau/7", "time >= tau/7", lambda tau: tau / 7),
        ("time > tau + 0.1", "time >= tau + 0.1", lambda tau: tau + 0.1),
        ("tau < time/3", "tau <= time/3", lambda tau: 3 * tau),
        ("time/3 > tau", "tau <= time/3", lambda tau: 3 * tau),
    ],
    ids=["tau-over-7", "tau-plus-0.1", "reversed", "mixed-orientation"],
)
def test_the_threshold_and_orientation_are_read_as_one_comparison(gt, ge, cross):
    """`tau <= time/3` and `time/3 >= tau` are one residual, oriented to rise
    where the trigger turns true."""
    text = _pair(gt, ge)
    for tau in TAUS:
        assert _final(text, tau, cross(tau))[0] == pytest.approx(SEQUENCE, rel=EXACT), tau


@pytest.mark.parametrize(
    "extra", ["", ", fromTrigger=false"], ids=["values-at-trigger", "values-at-execution"]
)
@pytest.mark.parametrize("p_gt, p_ge", [(3, 2), (2, 3), (0, 0)])
def test_priorities_do_not_reorder_the_two_sub_instants(extra, p_gt, p_ge):
    text = (
        "species X; X = 1; tau = 3\nJ0: -> X; 0\n"
        f"E1: at (time/3 > tau){extra}, priority = {p_gt}: X = 0.3*X\n"
        f"E2: at (time/3 >= tau){extra}, priority = {p_ge}: X = 0.5*X\n"
    )
    for tau in TAUS:
        assert _final(text, tau, 3 * tau)[0] == pytest.approx(SEQUENCE, rel=EXACT), tau


def test_the_held_event_reads_the_state_the_first_left():
    """E1 copies X, frozen at its trigger. Held, its trigger is after E2's
    write: Y = 0.5. Frozen with E2 in one batch, it was 1."""
    text = (
        "species X, Y; X = 1; Y = 0; tau = 3\nJ0: -> X; 0\n"
        "E1: at (time/3 > tau), priority = 3: Y = X\n"
        "E2: at (time/3 >= tau), priority = 2: X = 0.5*X\n"
    )
    for tau in TAUS:
        assert _final(text, tau, 3 * tau)[1] == pytest.approx(0.5, rel=EXACT), tau


def test_priority_orders_the_events_within_each_sub_instant():
    """Two non-strict triggers of priorities 1 and 2 and the strict one of
    priority 3: the order is 2, 1, then the strict one, recorded digit by
    digit. One batch gave 3, 2, 1."""
    text = (
        "species X; X = 0; tau = 3\nJ0: -> X; 0\n"
        "Es: at (time/3 > tau), priority = 3, fromTrigger=false: X = 10*X + 3\n"
        "Eb: at (time/3 >= tau), priority = 2, fromTrigger=false: X = 10*X + 2\n"
        "Ea: at (time/3 >= tau), priority = 1, fromTrigger=false: X = 10*X + 1\n"
    )
    # X = 10*X + d from 0: Eb gives 2, Ea 21, then Es 213. One batch by
    # priority, 3, 2, 1, gave 321.
    for tau in TAUS:
        assert _final(text, tau, 3 * tau)[0] == pytest.approx(213.0, rel=EXACT), tau


def test_two_strict_triggers_on_one_residual_stay_one_batch():
    """Only a strict trigger beside a non-strict one is held. Two strict ones
    rise together on every read and execute as one batch by priority: E2's
    frozen 0.5 is written last."""
    text = _pair("time/3 > tau", "time/3 > tau")
    for tau in TAUS:
        assert _final(text, tau, 3 * tau)[0] == pytest.approx(0.5, rel=EXACT), tau


def test_two_strict_triggers_on_one_residual_keep_their_priority_beside_a_third():
    """The two strict triggers, and Ec, which reads the same clock comparison
    inside an `&&` and is not one comparison itself, rise at one read. Not held,
    the three execute as one batch by priority: X = 10*X + d from 0 gives 3,
    32, 321. Holding the two strict ones behind Ec gives 1, 13, 132. Alone, two
    held strict triggers are released together and look like one batch, so only
    a third event shows the order."""
    text = (
        "species X; X = 0; tau = 3\nJ0: -> X; 0\n"
        "Es1: at (time/3 > tau), priority = 3, fromTrigger=false: X = 10*X + 3\n"
        "Es2: at (time/3 > tau), priority = 2, fromTrigger=false: X = 10*X + 2\n"
        "Ec: at ((time/3 > tau) && (tau > -1)), priority = 1, fromTrigger=false: X = 10*X + 1\n"
    )
    for tau in TAUS:
        assert _final(text, tau, 3 * tau)[0] == pytest.approx(321.0, rel=EXACT), tau


def test_a_state_trigger_pair_executes_in_sequence():
    """A' = 1, so A crosses tau at t = tau, located by CVODE's root finder."""
    text = (
        "species X, A; X = 1; A = 0; tau = 3\nJ0: -> A; 1\n"
        "E1: at (A > tau), priority = 3: X = 0.3*X\n"
        "E2: at (A >= tau), priority = 2: X = 0.5*X\n"
    )
    for tau in TAUS:
        assert _final(text, tau, tau)[0] == pytest.approx(SEQUENCE, rel=EXACT), tau


def test_a_delayed_strict_event_freezes_its_values_after_the_non_strict_one():
    """Held, E1 is triggered after E2 has halved X, so the value it freezes
    and writes a delay of tau later is 0.3·0.5. One batch froze 0.3·1."""
    text = (
        "species X; X = 1; tau = 3\nJ0: -> X; 0\n"
        "E1: at tau after (time/3 > tau), priority = 3: X = 0.3*X\n"
        "E2: at (time/3 >= tau), priority = 2: X = 0.5*X\n"
    )
    for tau in TAUS:
        assert _final(text, tau, 3 * tau)[0] == pytest.approx(SEQUENCE, rel=EXACT), tau


def test_a_pair_true_at_the_start_is_one_batch():
    """At the run's start both triggers read true with no crossing between
    them, so they execute as one batch, as before: E2's frozen 0.5 is written
    last."""
    text = (
        "species X, A; X = 1; A = 5; tau = 3\nJ0: -> A; 0\n"
        "E1: at (A > tau), t0=false, priority = 3: X = 0.3*X\n"
        "E2: at (A >= tau), t0=false, priority = 2: X = 0.5*X\n"
    )
    assert _final(text, 3.0, 1.0)[0] == pytest.approx(0.5, rel=EXACT)
    assert _final(text, 3.0, 1.0, method="ssa")[0] == pytest.approx(0.5, rel=EXACT)


def test_a_pair_a_jump_turns_true_is_one_batch():
    """f jumps from 0 to 2 at t = 5, so `f > 1` and `f >= 1` turn true at one
    instant with none between: one batch by priority, E2's frozen 0.5 last."""
    text = (
        "species X; X = 1; tau = 3\nJ0: -> X; 0\nf := piecewise(0, time < 5, 2)\n"
        "E1: at (f > 1), priority = 3: X = 0.3*X\n"
        "E2: at (f >= 1), priority = 2: X = 0.5*X\n"
    )
    assert _final(text, 3.0, 5.0)[0] == pytest.approx(0.5, rel=EXACT)


def test_ssa_a_pair_a_firing_turns_true_is_one_batch():
    """A firing takes X from 2 to 3, past 2.5 at once."""
    text = (
        "species X, Z; X = 0; Z = 1; tau = 3\nJ0: -> X; 1\n"
        "E1: at (X > 2.5), priority = 3: Z = 0.3*Z\n"
        "E2: at (X >= 2.5), priority = 2: Z = 0.5*Z\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    for seed in range(5):
        run = bngsim.Simulator(model, method="ssa").run(t_span=(0.0, 40.0), n_points=3, seed=seed)
        x, z = np.asarray(run.species)[-1]
        # X(40) is Poisson(40): it stays below 3 with probability
        # e^-40·(1 + 40 + 800) = 3.6e-15, so the crossing happened, for any seed.
        assert x >= 3
        assert z == pytest.approx(0.5, rel=EXACT), seed
        model.reset()


def test_a_held_trigger_the_first_event_turns_false_never_fires():
    """E2 sets A back to 0 at the crossing, so A > tau never holds: in sequence
    E1 never fires. One batch fired it, Y = 1."""
    text = (
        "species Y, A; Y = 0; A = 0; tau = 3\nJ0: -> A; 1\n"
        "E1: at (A > tau), priority = 3: Y = 1\n"
        "E2: at (A >= tau), priority = 2: A = 0\n"
    )
    for tau in TAUS:
        # Only E1 writes Y, so Y is exactly 0 unfired and 1 fired.
        assert _final(text, tau, tau)[0] == pytest.approx(0.0, abs=EXACT), tau


@pytest.mark.parametrize(
    "lhs, inc, scale, method",
    [("A", "A", 1.0, "ode"), ("time/3 + B", "B", 3.0, "ode"), ("time/3 + B", "B", 3.0, "ssa")],
    ids=["state", "clock", "ssa-clock"],
)
def test_an_event_that_moves_the_held_comparison_raises_it_there(lhs, inc, scale, method):
    """On the crossing `lhs > tau` is false, and Ehi's write moves lhs past
    tau: that raises Es there, a cascade of priority 5, ahead of Elo's 1.
    X = 10*X + d from 0 gives 9, 95, 951. Held to the end of the instant, Es
    ran after Elo, 915, for 46 of the 60 tau in the state run, 40 in the clock
    run and 51 in SSA. One batch gave 951 too, as main does."""
    text = (
        "species X, A, B; X = 0; A = 0; B = 0; tau = 3\nJ0: -> A; 1\n"
        f"Ehi: at ({lhs} >= tau), priority = 10, fromTrigger=false: "
        f"{inc} = {inc} + 1, X = 10*X + 9\n"
        f"Es: at ({lhs} > tau), priority = 5, fromTrigger=false: X = 10*X + 5\n"
        f"Elo: at ({lhs} >= tau), priority = 1, fromTrigger=false: X = 10*X + 1\n"
    )
    for tau in TAUS:
        assert _final(text, tau, scale * tau, method=method)[0] == pytest.approx(
            951.0, rel=EXACT
        ), tau


def test_a_sensitivity_run_through_a_moved_comparison():
    """Ehi writes A, which Es's comparison reads, so Es is raised there. Where
    Es reads X, which Ehi wrote, the run refuses it by name rather than return
    the batch's 0.3 (main did, for 46 of 60 tau). Where it writes Y alone, the
    jump composes it as one of the batch: dA/dk = T, dY/dk = dZ/dk = 0. A read
    that lands on the crossing finds Es false and raised by Ehi's write, which
    the run refuses as any cascade (issue #144), as main does."""
    reads = (
        "species X, A; X = 1; A = 0; tau = 3; k = 1\nJ0: -> A; k\n"
        "Ehi: at (A >= tau), priority = 10: A = A + 1, X = 0.5*X\n"
        "Es: at (A > tau), priority = 5: X = 0.3*X\n"
        "Elo: at (A >= tau), priority = 1: A = A\n"
    )
    keeps = (
        "species Y, Z, A; Y = 0; Z = 0; A = 0; tau = 3; k = 1\nJ0: -> A; k\n"
        "Ehi: at (A >= tau), priority = 10: A = A + 1\n"
        "Es: at (A > tau), priority = 5: Y = 1\n"
        "Elo: at (A >= tau), priority = 1: Z = 2\n"
    )
    by_name = kept = 0
    for tau in TAUS:
        times = [0.0, tau / 2, 1.4 * tau, 2.1 * tau]
        for text in (reads, keeps):
            model = bngsim.Model.from_antimony_string(text)
            model.set_param("tau", tau)
            sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
            try:
                run = sim.run(sample_times=times, rtol=1e-10, atol=1e-12)
            except bngsim.SimulationError as e:
                assert "'Es'" in str(e), str(e)
                if text is reads and "issue #947" in str(e):
                    by_name += 1
                continue
            assert text is keeps, (tau, np.asarray(run.species)[-1])
            names = list(run.species_names)
            x = dict(zip(names, np.asarray(run.species)[-1], strict=True))
            s = dict(zip(names, np.asarray(run.sensitivities)[-1, :, 0], strict=True))
            assert (x["Y"], x["Z"]) == (
                pytest.approx(1.0, rel=EXACT),
                pytest.approx(2.0, rel=EXACT),
            ), tau
            # As in the test below: A = k·t + 1 is linear in t past the
            # crossing, and dA/dk is 3.5e-16 off T at worst (measured).
            assert s["A"] == pytest.approx(times[-1], rel=1e-8), tau
            assert (s["Y"], s["Z"]) == (
                pytest.approx(0.0, abs=EXACT),
                pytest.approx(0.0, abs=EXACT),
            ), tau
            kept += 1
    # 46 of 60 each on macOS arm64 (measured); the other 14 land on the
    # crossing. A release that always refused would keep none.
    assert by_name > 0 and kept > 0


def test_a_sensitivity_run_refuses_a_held_pair_that_reads_the_first_write():
    """E1 reads the X that E2 wrote, so the plain run's E1 is a second jump at
    the instant, which the sensitivity jump cannot compose (issue #144). A run
    whose read lands on the crossing fires E1 at its own root, and gives the
    sequence. Either way it never returns the batch's 0.5."""
    text = (
        "species X, A; X = 1; A = 0; tau = 3; k = 1\nJ0: -> A; k\n"
        "E1: at (A > tau), priority = 3: X = 0.3*X\n"
        "E2: at (A >= tau), priority = 2: X = 0.5*X\n"
    )
    refused = 0
    for tau in TAUS:
        try:
            x = _final(text, tau, tau, sensitivity_params=["k"])[0]
        except bngsim.SimulationError as e:
            assert re.search(r"'E1'.*'E2'.*issue #947", str(e)), str(e)
            refused += 1
        else:
            assert x == pytest.approx(SEQUENCE, rel=EXACT), tau
    # CVODE's located root lands past the crossing for most tau: 57 of the 60
    # refuse on macOS arm64 (measured). A run that never refused would be
    # returning the plain run's sequence with sensitivities it cannot compose.
    assert refused > 0


def test_a_sensitivity_run_keeps_a_pair_that_touches_nothing_of_the_other():
    """E1 writes Y and E2 writes Z, and neither reads the other's: batch and
    sequence agree, so the run keeps its sensitivities. A = k·t and the events
    write constants, so dA/dk(T) = T and dY/dk = dZ/dk = 0 past the crossing."""
    text = (
        "species Y, Z, A; Y = 0; Z = 0; A = 0; tau = 3; k = 1\nJ0: -> A; k\n"
        "E1: at (A > tau), priority = 3: Y = 1\n"
        "E2: at (A >= tau), priority = 2: Z = 2\n"
    )
    for tau in TAUS:
        model = bngsim.Model.from_antimony_string(text)
        model.set_param("tau", tau)
        times = [0.0, tau / 2, 1.4 * tau, 2.1 * tau]
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
            sample_times=times, rtol=1e-10, atol=1e-12
        )
        names = list(run.species_names)
        x = dict(zip(names, np.asarray(run.species)[-1], strict=True))
        s = dict(zip(names, np.asarray(run.sensitivities)[-1, :, 0], strict=True))
        assert (x["Y"], x["Z"]) == (
            pytest.approx(1.0, rel=EXACT),
            pytest.approx(2.0, rel=EXACT),
        ), tau
        # A = k·t is linear in t, which BDF integrates exactly, and the run is at
        # rtol 1e-10: dA/dk is 6e-16 off T at worst (measured); 1e-8 is 100x the
        # solver's tolerance.
        assert s["A"] == pytest.approx(times[-1], rel=1e-8), tau
        # Y and Z are written constants with Y' = Z' = 0 on both sides of the
        # crossing, so the event-time jump adds nothing: 0 in every run
        # (measured). A jump that moved them by dt*/dk = -tau/k would be off by
        # tau, at least 0.01 here.
        assert (s["Y"], s["Z"]) == (
            pytest.approx(0.0, abs=EXACT),
            pytest.approx(0.0, abs=EXACT),
        ), tau


@pytest.mark.parametrize("lhs, scale", [("0.1*time", 10.0), ("time/3", 3.0)])
def test_ssa_executes_the_pair_in_sequence(lhs, scale):
    text = (
        "species X; X = 100; tau = 3\nJ0: -> X; 0\n"
        f"E1: at ({lhs} > tau), priority = 3: X = 0.3*X\n"
        f"E2: at ({lhs} >= tau), priority = 2: X = 0.5*X\n"
    )
    for tau in TAUS:
        assert _final(text, tau, scale * tau, method="ssa")[0] == pytest.approx(15.0, rel=EXACT), (
            tau
        )
