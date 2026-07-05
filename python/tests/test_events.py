# bngsim/python/tests/test_events.py — Tests for SBML event support (Session 66, P0)
#
# Tests the end-to-end event pipeline: ModelBuilder → add_event → CVODE → discontinuity.

import math

import numpy as np
from bngsim._bngsim_core import CvodeSimulator, ModelBuilder, TimeSpec


def test_event_bolus_dose_ode():
    """Exponential decay + bolus dose event at t=10.

    Model: dS/dt = -k*S, S(0) = 100, k = 0.1
    Event: at time() >= 10, set S = S + 50

    Before event: S(t) = 100*exp(-0.1*t)
    At t=10: S jumps from ~36.8 to ~86.8
    After event: S(t) = 86.8*exp(-0.1*(t-10))
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.1)
    s_idx = b.add_species("S", 100.0)

    # dS/dt = -k*S via elementary rate law: S → ∅ with rate k
    b.add_reaction([s_idx], [], "elementary", "k")

    # Event: at time() >= 10, S := S + 50
    b.add_event("dose", "time() >= 10", [(s_idx, "S + 50")])

    model = b.build()
    assert model.n_species == 1
    assert model.n_events == 1

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 20.0
    ts.n_points = 201

    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    # Before event (t=5): S ≈ 100*exp(-0.5) ≈ 60.65
    idx_5 = np.argmin(np.abs(t - 5.0))
    S_5_expected = 100.0 * math.exp(-0.5)
    assert abs(S[idx_5] - S_5_expected) < 1.0, f"S(5) = {S[idx_5]}, expected ~{S_5_expected}"

    # Find the discontinuity around t=10
    # S should jump UP (bolus adds 50)
    idx_before = np.argmin(np.abs(t - 9.9))
    idx_after = np.argmin(np.abs(t - 10.1))
    S_before = S[idx_before]
    S_after = S[idx_after]

    assert S_after > S_before, f"S should increase after event: {S_before} -> {S_after}"
    assert S_after > 70.0, f"S after event should be > 70, got {S_after}"

    # At t=20: should be much higher than without event
    S_20_no_event = 100.0 * math.exp(-2.0)  # ~13.5
    assert S[-1] > S_20_no_event * 1.5, (
        f"S(20)={S[-1]} should be much higher than no-event S(20)={S_20_no_event}"
    )


def test_event_simple_time_trigger():
    """Simple event: at time() >= 5, set S = 200."""
    b = ModelBuilder()
    b.add_parameter("k", 0.0)  # no decay
    s_idx = b.add_species("S", 100.0)

    # No reactions (S stays at 100 until event)
    # Event: at time() >= 5, S := 200
    b.add_event("jump", "time() >= 5", [(s_idx, "200")])

    model = b.build()
    assert model.n_events == 1

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 10.0
    ts.n_points = 101

    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    # Before event: S = 100 (no reactions)
    # After event (t >= 5): S = 200
    for i in range(len(t)):
        if t[i] < 4.9:
            assert abs(S[i] - 100.0) < 1.0, f"Before event: S({t[i]}) = {S[i]}, expected 100"
        elif t[i] > 5.1:
            assert abs(S[i] - 200.0) < 1.0, f"After event: S({t[i]}) = {S[i]}, expected 200"


def test_event_assigns_promoted_parameter():
    """Promoted-parameter event: event changes a 'parameter' that's used in a rate law.

    Session 68 (P0.next) bug fix: When an SBML event assigns to a parameter,
    the SBML loader promotes the parameter to a species. The event writes to
    species[i].concentration and y_data[i], but the observable backing the
    promoted parameter wasn't updated until the next compute_derivs() call.
    This meant rate laws using that parameter saw the STALE value.

    Model:
      Species S, S(0) = 100
      Promoted parameter k_sp, k_sp(0) = 0.1 (also a species)
      Observable k_obs = k_sp
      Function rate_fn = k_obs   (rate law reads the observable)
      Reaction: S → ∅ with functional rate law rate_fn (rate = k_obs * S)
      Event: at time() >= 5, k_sp := 0.5  (increase decay rate 5×)

    Before event (t < 5): dS/dt = -0.1*S → S(t) = 100*exp(-0.1*t)
    After event (t >= 5): dS/dt = -0.5*S → S(t) = S(5)*exp(-0.5*(t-5))

    If the bug is present, k_obs stays at 0.1 after the event, and
    S decays slowly. If the fix works, k_obs jumps to 0.5, and S decays
    rapidly after t=5.
    """
    b = ModelBuilder()

    # The promoted parameter: a species with initial value 0.1
    k_idx = b.add_species("k_sp", 0.1)

    # The real species
    s_idx = b.add_species("S", 100.0)

    # Observable tracking the promoted parameter species
    b.add_observable("k_obs", [(k_idx, 1.0)])

    # A function that reads the observable (like a variable parameter).
    # The function name matches the parameter name below — the builder
    # auto-creates a var_param_binding so evaluate_functions() copies
    # the function value into the parameter before each RHS evaluation.
    b.add_function("rate_fn", "k_obs")

    # Parameter with same name as function: builder auto-binds them.
    b.add_parameter("rate_fn", 0.1)

    # Reaction: S → ∅ with rate = rate_fn * S (elementary rate law)
    b.add_reaction([s_idx], [], "elementary", "rate_fn")

    # Event: at time() >= 5, set k_sp := 0.5
    b.add_event("increase_k", "time() >= 5", [(k_idx, "0.5")])

    model = b.build()
    assert model.n_species == 2  # S + k_sp
    assert model.n_events == 1

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 10.0
    ts.n_points = 101

    result = sim.run(ts)
    t = np.array(result.time)
    data = np.array(result.species_data)
    S = data[:, 1]  # S is species index 1 (k_sp is 0)
    k = data[:, 0]  # k_sp is species index 0

    # Before event: S(4) ≈ 100*exp(-0.4) ≈ 67.0
    idx_4 = np.argmin(np.abs(t - 4.0))
    S_4_expected = 100.0 * math.exp(-0.4)
    assert abs(S[idx_4] - S_4_expected) / S_4_expected < 0.05, (
        f"S(4) = {S[idx_4]:.3f}, expected ~{S_4_expected:.3f}"
    )

    # k_sp should be 0.1 before event, 0.5 after
    assert abs(k[idx_4] - 0.1) < 0.01, f"k(4) should be 0.1, got {k[idx_4]}"
    idx_6 = np.argmin(np.abs(t - 6.0))
    assert abs(k[idx_6] - 0.5) < 0.01, f"k(6) should be 0.5, got {k[idx_6]}"

    # After event: S should decay MUCH faster (5× rate)
    # S(5) ≈ 100*exp(-0.5) ≈ 60.65
    # S(10) = S(5)*exp(-0.5*5) = 60.65*exp(-2.5) ≈ 4.98
    S_5_expected = 100.0 * math.exp(-0.5)
    S_10_expected = S_5_expected * math.exp(-0.5 * 5.0)

    # If the bug is present, S(10) ≈ 100*exp(-1.0) ≈ 36.8 (slow decay throughout)
    S_10_no_fix = 100.0 * math.exp(-1.0)

    # The fixed result should be much closer to the fast-decay expectation
    assert S[-1] < S_10_no_fix * 0.5, (
        f"S(10) = {S[-1]:.3f} — too high, suggests promoted-param event "
        f"didn't propagate to rate law. Expected ~{S_10_expected:.3f}, "
        f"no-fix would give ~{S_10_no_fix:.3f}"
    )

    # More precise: S(10) should be within 20% of analytical expectation
    assert abs(S[-1] - S_10_expected) / S_10_expected < 0.20, (
        f"S(10) = {S[-1]:.3f}, expected ~{S_10_expected:.3f}"
    )


def test_delayed_event():
    """Delayed event: trigger fires at t=5, but assignment
    applies at t=5+delay=8.

    Model: S stays constant (no reactions), S(0)=100.
    Event: at time() >= 5, after delay=3, set S := 200.

    S should stay 100 until t=8, then jump to 200.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 100.0)

    # Event with delay=3: trigger at t>=5, apply at t=8
    b.add_event("delayed_jump", "time() >= 5", [(s_idx, "200")], delay=3.0)

    model = b.build()
    assert model.n_events == 1

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 15.0
    ts.n_points = 151

    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    # Before delay expires (t=7): S should be 100
    idx_7 = np.argmin(np.abs(t - 7.0))
    assert abs(S[idx_7] - 100.0) < 1.0, f"S(7) = {S[idx_7]}, should be 100 (delay not expired)"

    # After delay expires (t=9): S should be 200
    idx_9 = np.argmin(np.abs(t - 9.0))
    assert abs(S[idx_9] - 200.0) < 1.0, f"S(9) = {S[idx_9]}, should be 200 (delayed event applied)"

    # At end: still 200
    assert abs(S[-1] - 200.0) < 1.0, f"S(15) = {S[-1]}, should still be 200"


def test_event_initialvalue_false_fires_at_t0():
    """SBML L3 §3.4.5: initialValue=false ⇒ trigger presumed false before t=0,
    so a t=0 expression that is true is a rising-edge fire AT t=0.

    Model: S(0)=100, no reactions. Event with trigger ``time() >= 0`` (true at
    t=0) and ``initial_value=False`` sets S := 200. Recorded value at t=0
    must already be 200, since the event fires before the initial state is
    recorded.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 100.0)
    b.add_event(
        "fire_at_zero",
        "time() >= 0",
        [(s_idx, "200")],
        initial_value=False,  # presumed-prior false → t=0 is rising edge
    )
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 5.0
    ts.n_points = 11
    result = sim.run(ts)
    S = np.array(result.species_data)[:, 0]

    assert abs(S[0] - 200.0) < 1e-6, (
        f"S(0)={S[0]} — event with initial_value=False must fire at t=0"
    )
    assert abs(S[-1] - 200.0) < 1e-6


def test_event_initialvalue_true_does_not_fire_at_t0():
    """initialValue=true ⇒ trigger presumed already true before t=0, so a
    t=0 expression that is true is NOT a transition and the event must NOT
    fire at t=0. Once the trigger goes false then back to true later, it
    fires.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 100.0)
    # Trigger that is true at t=0, false on (1,2), true again at t>=2.
    # The only acceptable rising edge is at t=2 (false→true).
    b.add_event(
        "edge_at_two",
        "(time() < 1) or (time() >= 2)",
        [(s_idx, "200")],
        initial_value=True,  # presumed-prior true → no fire at t=0
    )
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 3.0
    ts.n_points = 31
    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    assert abs(S[0] - 100.0) < 1e-6, (
        f"S(0)={S[0]} should be 100 (initial_value=True suppresses t=0 fire)"
    )
    # Between t=1 and t=2 the trigger is false; S still 100.
    idx_15 = np.argmin(np.abs(t - 1.5))
    assert abs(S[idx_15] - 100.0) < 1e-6
    # After t=2 the trigger becomes true again — rising edge fires.
    assert abs(S[-1] - 200.0) < 1.0, f"S(end)={S[-1]} should be 200 after rising edge at t=2"


def test_event_priority_orders_simultaneous_fires():
    """Two events with the same trigger time fire simultaneously; the
    one with higher priority must apply LAST so its assignment wins.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 0.0)
    # Both fire at t>=1. Without priority, ordering is implementation-defined.
    # With priority, the higher-priority event applies LAST (sorted descending,
    # processed in order, so highest applies first… BUT we want the *winner*
    # to be the last write — therefore the WINNER has the LOWEST priority in
    # our sort? No: SBML defines higher priority = fires FIRST. The earlier
    # write is then overwritten by the lower-priority later write. That is
    # the opposite of what most users expect, so the test pins the spec.)
    b.add_event("low_prio", "time() >= 1", [(s_idx, "1")], priority=1)
    b.add_event("high_prio", "time() >= 1", [(s_idx, "2")], priority=10)
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 2.0
    ts.n_points = 21
    result = sim.run(ts)
    S = np.array(result.species_data)[:, 0]

    # Per SBML L3: higher priority fires first → low_prio fires after →
    # final value is 1 (low_prio's assignment, written last).
    assert abs(S[-1] - 1.0) < 1e-6, (
        f"S(end)={S[-1]}: high-priority event must fire first, low-priority "
        "writes last → expected 1.0"
    )


def test_event_state_dependent_delay():
    """Event delay computed at trigger time from a species value.

    Model: S(0)=4, no reactions. Event triggers at t>=1 with delay=S → 4.
    Apply time = 1 + 4 = 5. After t=5 the assignment fires.
    """
    b = ModelBuilder()
    b.add_parameter("k", 0.0)
    s_idx = b.add_species("S", 4.0)
    b.add_event(
        "state_delay",
        "time() >= 1",
        [(s_idx, "999")],
        delay_expr="S",
    )
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 7.0
    ts.n_points = 71
    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    # Before apply time (t=4): still 4
    idx_4 = np.argmin(np.abs(t - 4.0))
    assert abs(S[idx_4] - 4.0) < 1e-6, f"S(4)={S[idx_4]} should still be 4 (delay not expired)"
    # After apply time (t=6): 999
    idx_6 = np.argmin(np.abs(t - 6.0))
    assert abs(S[idx_6] - 999.0) < 1e-3, f"S(6)={S[idx_6]} should be 999 (delay expired at t=5)"


def test_event_use_values_from_trigger_time_true_default():
    """useValuesFromTriggerTime=True (SBML L3 default): assignment RHS is
    evaluated at trigger time, frozen, and applied verbatim at firing time.

    Model: S(0)=10, dS/dt = -S (decay). Event triggers at t=1 with delay=2,
    assigns ``S := S * 10``. With UVFTT=true, the RHS S*10 is evaluated at
    t=1 (S≈10*exp(-1)≈3.679 → RHS≈36.79) and that frozen value is written
    at t=3. Without UVFTT, the RHS would evaluate against the t=3 state
    (S≈10*exp(-3)≈0.498 → RHS≈4.98).
    """
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 10.0)
    b.add_reaction([s_idx], [], "elementary", "k")  # dS/dt = -S
    b.add_event(
        "freeze_rhs",
        "time() >= 1",
        [(s_idx, "S * 10")],
        delay=2.0,
        use_values_from_trigger_time=True,
    )
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 4.0
    ts.n_points = 81
    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    expected_frozen = 10.0 * math.exp(-1.0) * 10.0  # ~36.79
    # Sample exactly at the apply time t=3.0 (post-event, pre-further-decay).
    idx_3 = np.argmin(np.abs(t - 3.0))
    assert abs(S[idx_3] - expected_frozen) / expected_frozen < 0.01, (
        f"S(3)={S[idx_3]} — RHS should be frozen at trigger time (~{expected_frozen:.3f})"
    )


def test_event_use_values_from_trigger_time_false():
    """useValuesFromTriggerTime=False: assignment RHS is evaluated at
    firing time against the current state. Same model as the UVFTT=true
    test, but with UVFTT=false the apply-time RHS uses S(t=3)≈0.498 →
    target ≈ 4.98.
    """
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 10.0)
    b.add_reaction([s_idx], [], "elementary", "k")  # dS/dt = -S
    b.add_event(
        "fresh_rhs",
        "time() >= 1",
        [(s_idx, "S * 10")],
        delay=2.0,
        use_values_from_trigger_time=False,
    )
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 4.0
    ts.n_points = 81
    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    expected_fresh = 10.0 * math.exp(-3.0) * 10.0  # ~4.98
    idx_3 = np.argmin(np.abs(t - 3.0))
    assert abs(S[idx_3] - expected_fresh) / expected_fresh < 0.01, (
        f"S(3)={S[idx_3]} — RHS should be evaluated at apply time (~{expected_fresh:.3f})"
    )


def test_event_priority_cancels_non_persistent_when_trigger_falsified():
    """SBML L3: when a high-priority event fires and changes state such
    that a lower-priority non-persistent event's trigger goes false, the
    non-persistent event must be cancelled (not fire). Mirrors SBML test
    suite case 00935.

    Three events with the SAME trigger (S1 < 0.5) firing simultaneously:
      A (priority=10, persistent=true,  S1=1, S2=0)  ← fires first
      C (priority=9,  persistent=true,  S1=3, S2=2)  ← fires second
      B (priority=8,  persistent=false, S1=2, S2=1)  ← cancelled

    After A fires: S1=1, so the trigger (S1<0.5) becomes false. C is
    persistent so still fires (writes S1=3, S2=2). B is non-persistent
    and the trigger is now false, so B is cancelled and does NOT fire.

    Final: S1=3, S2=2 (C wins).
    """
    b = ModelBuilder()
    s1 = b.add_species("S1", 0.0)
    s2 = b.add_species("S2", 1.0)
    b.add_event("A", "S1 < 0.5", [(s1, "1"), (s2, "0")], priority=10, persistent=True)
    b.add_event("C", "S1 < 0.5", [(s1, "3"), (s2, "2")], priority=9, persistent=True)
    b.add_event(
        "B", "S1 < 0.5", [(s1, "2"), (s2, "1")], priority=8, persistent=False, initial_value=False
    )
    # initial_value=False on B forces a t=0 fire if trigger is true (it is, S1=0<0.5).
    # Same for the others — they all need initial_value=False to fire at t=0.
    # (We re-add A and C with the right setting; ModelBuilder doesn't expose
    # update so we rebuild from scratch.)
    b = ModelBuilder()
    s1 = b.add_species("S1", 0.0)
    s2 = b.add_species("S2", 1.0)
    b.add_event(
        "A", "S1 < 0.5", [(s1, "1"), (s2, "0")], priority=10, persistent=True, initial_value=False
    )
    b.add_event(
        "C", "S1 < 0.5", [(s1, "3"), (s2, "2")], priority=9, persistent=True, initial_value=False
    )
    b.add_event(
        "B", "S1 < 0.5", [(s1, "2"), (s2, "1")], priority=8, persistent=False, initial_value=False
    )

    model = b.build()
    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 1.0
    ts.n_points = 11
    result = sim.run(ts)
    S1 = np.array(result.species_data)[:, 0]
    S2 = np.array(result.species_data)[:, 1]

    assert abs(S1[0] - 3.0) < 1e-6, (
        f"S1(0)={S1[0]} — expected 3 (event C wins after A's fire cancels B)"
    )
    assert abs(S2[0] - 2.0) < 1e-6, f"S2(0)={S2[0]} — expected 2"


def test_event_re_arms_when_assignment_falsifies_own_trigger():
    """Mirrors SBML test suite case 00026.

    Event whose own assignment falsifies its trigger must re-arm: after
    firing, the trigger goes false (because the assignment changed
    state); the next time state evolves so the trigger goes false→true,
    the event must fire again.

    Model: dS/dt = -S, S(0) = 1. Event: when ``S < 0.1``, set S := 1.
    S decays to 0.1 at t = ln(10) ≈ 2.303. The event fires there,
    resetting S to 1; S decays again and crosses 0.1 at t ≈ 4.605, where
    it must fire a second time.
    """
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 1.0)
    b.add_reaction([s_idx], [], "elementary", "k")  # dS/dt = -S
    b.add_event("rearm", "S < 0.1", [(s_idx, "1")])

    model = b.build()
    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 5.0
    ts.n_points = 51
    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    # First crossing: ln(10) ≈ 2.303. Just after, S should be near 1.
    idx_24 = np.argmin(np.abs(t - 2.4))
    assert S[idx_24] > 0.85, f"S(2.4)={S[idx_24]} — expected ~exp(-0.1) ≈ 0.905 after first fire"

    # Second crossing: ~4.605. Just after, S should be near 1 again.
    idx_47 = np.argmin(np.abs(t - 4.7))
    assert S[idx_47] > 0.85, f"S(4.7)={S[idx_47]} — expected ~0.905 after second fire (re-arm)"


def test_event_apply_time_precision_between_samples():
    """Mirrors SBML test suite case 00071. Delayed event must apply at
    the precise apply_time, not at the next output sample point.

    Model: dS/dt = -S, S(0) = 1. Event triggers at S < 0.1 (which the
    decay reaches at t ≈ 2.303); delay = 1, assignment S := 1. The event
    must apply at t ≈ 3.303 — *between* the regularly spaced output
    samples at t = 3.3 and t = 3.4. The recorded value at t = 3.4 must
    therefore reflect S = 1 * exp(-(3.4 - 3.303)) ≈ 0.9075, NOT S = 1
    (which is what we'd see if the apply was deferred to the next
    sample). The diff between those is ~0.09 in absolute terms.
    """
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 1.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    b.add_event("delayed_rearm", "S < 0.1", [(s_idx, "1")], delay=1.0)

    model = b.build()
    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 5.0
    ts.n_points = 51
    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    apply_time = math.log(10.0) + 1.0  # 2.302585... + 1 = 3.302585...
    idx_34 = np.argmin(np.abs(t - 3.4))
    expected = math.exp(-(3.4 - apply_time))  # ≈ 0.9075
    assert abs(S[idx_34] - expected) < 0.005, (
        f"S(3.4)={S[idx_34]} — expected ~{expected:.4f}; the event must "
        "apply at apply_time, not at the next sample point"
    )


def test_event_priority_refresh_between_simultaneous_fires():
    """SBML L3 §3.4.6: state-dependent priorities must be re-evaluated
    AFTER each immediate fire. Three events trigger simultaneously; the
    initial priority order picks A, but after A fires, B's and C's
    priorities flip — so C must fire next, not B as initial sort would
    suggest. Mirrors SBML test 00934.

    Initial: S1=0, S2=1.
    Events all trigger when time>=0.99:
      A: priority 10              ; assigns S1=1, S2=0
      B: priority 2*S2            ; assigns S1=2, S2=1
      C: priority 2*S1            ; assigns S1=3, S2=2

    Initial priorities: A=10, B=2, C=0  → A fires first.
    After A: S1=1, S2=0 → priorities recomputed: B=0, C=2 → C fires next.
    After C: S1=3, S2=2 → B's priority = 4 → B fires last.
    Final state: S1=2, S2=1 (B's writes win, because B fired last).
    """
    b = ModelBuilder()
    s1 = b.add_species("S1", 0.0)
    s2 = b.add_species("S2", 1.0)
    b.add_event("A", "time() >= 0.99", [(s1, "1"), (s2, "0")], priority=10)
    b.add_event(
        "B",
        "time() >= 0.99",
        [(s1, "2"), (s2, "1")],
        priority_expr="2*S2",
    )
    b.add_event(
        "C",
        "time() >= 0.99",
        [(s1, "3"), (s2, "2")],
        priority_expr="2*S1",
    )
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 2.0
    ts.n_points = 21
    result = sim.run(ts)
    S = np.array(result.species_data)
    # Final values after all three fires:
    assert abs(S[-1, 0] - 2.0) < 1e-6, f"S1(end)={S[-1, 0]}, expected 2.0"
    assert abs(S[-1, 1] - 1.0) < 1e-6, f"S2(end)={S[-1, 1]}, expected 1.0"


def test_event_csymbol_time_in_delay_expression():
    """SBML csymbol time as the event delay. With trigger ``time>=1`` and
    delay ``time``, the apply_time = trigger_time + trigger_time = 2.0.
    The loader had been folding ``time()`` to 0 inside the constant-fold
    pass and emitting delay=0 instead of an expression — squashing the
    delay entirely. Mirrors SBML test 00886.
    """
    b = ModelBuilder()
    s_idx = b.add_species("S", 0.0)
    b.add_event(
        "csymbol_delay",
        "time() >= 1",
        [(s_idx, "5")],
        delay_expr="time()",
    )
    model = b.build()

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 3.0
    ts.n_points = 31
    result = sim.run(ts)
    t = np.array(result.time)
    S = np.array(result.species_data)[:, 0]

    # Before apply_time = 2.0: S = 0
    idx_19 = np.argmin(np.abs(t - 1.9))
    assert S[idx_19] < 1e-6, f"S({t[idx_19]})={S[idx_19]} should still be 0"
    # After apply_time = 2.0: S = 5
    idx_22 = np.argmin(np.abs(t - 2.2))
    assert abs(S[idx_22] - 5.0) < 1e-6, (
        f"S({t[idx_22]})={S[idx_22]} should be 5 once delayed assignment fires"
    )


def test_no_events_zero_overhead():
    """Model without events should work exactly as before."""
    b = ModelBuilder()
    b.add_parameter("k", 0.1)
    s_idx = b.add_species("S", 100.0)

    # Use elementary rate law (no functional needed)
    b.add_reaction([s_idx], [], "elementary", "k")

    model = b.build()
    assert model.n_events == 0

    sim = CvodeSimulator(model)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = 10.0
    ts.n_points = 11

    result = sim.run(ts)
    S_end = np.array(result.species_data)[-1, 0]
    S_expected = 100.0 * math.exp(-1.0)
    assert abs(S_end - S_expected) < 0.01, f"S(10) = {S_end}, expected {S_expected}"
