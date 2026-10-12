"""What executes at one instant is ordered by priority, read when each
executes, whether it was triggered just now or a delay ago (issue #752).

SBML L3 calls events simultaneous when their execution times (trigger time
plus delay) are equal, and has the one of highest priority execute next, the
priorities read again after each execution. bngsim read a priority only when
an event was triggered. A delayed execution was queued, and at its time the
queue was run in the order it was filled, after everything triggered on that
instant had already executed:

* two delayed events triggered at different times and due together ran in
  trigger order whatever their priorities (case A);
* a priority that changed during the delay was never read again (case B);
* an immediate event always ran before a delayed one due on the same instant
  (case C).

In each the last writer of x, and so the state from there on, was the wrong
one, with no warning. The due delayed executions now join the batch of the
instant they execute at.

Oracle: the SBML rule, by hand. Each model writes x from two or three events
on one instant; the last to execute, which is the one of lowest priority,
leaves its value. The immediate-event twins of case B, which were right
before, are the control that the readout is the model's.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


def _end(text: str, name: str = "x", t_end: float = 4.0, **kw) -> float:
    model = bngsim.Model.from_antimony_string(text)
    kw.setdefault("n_points", 5)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), **kw)
    return float(np.asarray(r.observables)[-1, list(r.observable_names).index(name)])


# Case A: E1 is triggered at 0 and waits 2, E2 at 1 and waits 1.
_A = (
    "x = 0;\n"
    "E1: at 2 after (time >= 0), t0=false, priority = {p1}: x = 1;\n"
    "E2: at 1 after (time >= 1), priority = {p2}: x = 2;\n"
)
# Case B: both triggered at 1 and due at 2; E0 moves E1's priority at 1.5.
_B = (
    "x = 0; p = {p0};\n"
    "E0: at (time >= 1.5): p = {p1};\n"
    "E1: at 1 after (time >= 1), priority = p: x = 1;\n"
    "E2: at 1 after (time >= 1), priority = 5: x = 2;\n"
)
_B_IMMEDIATE = (
    "x = 0; p = {p0};\n"
    "E0: at (time >= 1.5): p = {p1};\n"
    "E1: at (time >= 2), priority = p: x = 1;\n"
    "E2: at (time >= 2), priority = 5: x = 2;\n"
)
# Case C: one triggered at 3, one triggered at 0 and due at 3.
_C = (
    "x = 0;\n"
    "Eimm: at (time >= 3), priority = {pi}: x = 1;\n"
    "Edel: at 3 after (time >= 0), t0=false, priority = {pd}: x = 2;\n"
)


@pytest.mark.parametrize("p1, p2, last", [(1, 10, 1.0), (10, 1, 2.0)])
def test_delayed_events_due_together_execute_by_priority(p1, p2, last):
    """The one of lower priority executes last and its value stays. Before,
    x = 2 for both orders: E2 was queued second."""
    assert _end(_A.format(p1=p1, p2=p2)) == last


@pytest.mark.parametrize("p0, p1, last", [(10, 0, 1.0), (0, 10, 2.0)])
def test_a_priority_is_read_when_the_event_executes(p0, p1, last):
    """E1's priority is p. At the trigger it outranks E2 in the first row and
    not in the second; at execution it is the other way round, which is what
    decides. Before, each row gave the other's answer."""
    assert _end(_B.format(p0=p0, p1=p1)) == last
    assert _end(_B_IMMEDIATE.format(p0=p0, p1=p1)) == last


@pytest.mark.parametrize("pi, pd, last", [(1, 10, 1.0), (10, 1, 2.0)])
def test_a_delayed_event_and_an_immediate_one_on_an_instant_execute_by_priority(pi, pd, last):
    """Before, x = 2 for both: the immediate one always ran first."""
    assert _end(_C.format(pi=pi, pd=pd)) == last


def test_three_delayed_events_triggered_at_three_times():
    text = (
        "x = 0;\n"
        "E1: at 2 after (time >= 0), t0=false, priority = 1: x = 1;\n"
        "E2: at 1.5 after (time >= 0.5), priority = 3: x = 2;\n"
        "E3: at 1 after (time >= 1), priority = 2: x = 3;\n"
    )
    assert _end(text) == 1.0


def test_a_later_execution_reads_what_an_earlier_one_wrote():
    """E2, of priority 10, sets x = 2 first; E1, reading at execution, then
    copies it. Before, E1 ran first and y stayed 0."""
    text = (
        "x = 0; y = 0;\n"
        "E1: at 2 after (time >= 0), t0=false, priority = 1, fromTrigger=false: y = x;\n"
        "E2: at 1 after (time >= 1), priority = 10: x = 2;\n"
    )
    assert _end(text, "y") == 2.0


def test_values_frozen_at_the_trigger_are_still_the_ones_written():
    """E1 freezes y := x at its trigger (x = 0), and executes after E2 has set
    x = 2: the order is by priority, the value is the trigger's."""
    text = (
        "x = 0; y = 7;\n"
        "E1: at 2 after (time >= 0), t0=false, priority = 1: y = x;\n"
        "E2: at 1 after (time >= 1), priority = 10: x = 2;\n"
    )
    assert _end(text, "y") == 0.0


def test_events_due_at_different_times_are_not_reordered():
    """Priority orders one instant. E1 is due at 2 and E2 at 2.5: E2 is last,
    whatever its priority."""
    text = (
        "x = 0;\n"
        "E1: at 2 after (time >= 0), t0=false, priority = 1: x = 1;\n"
        "E2: at 1.5 after (time >= 1), priority = 10: x = 2;\n"
    )
    assert _end(text) == 2.0


def test_a_non_persistent_delayed_event_another_disables_does_not_execute():
    """Both are due at 2. E2, of higher priority, executes first and makes
    E1's trigger false; E1 is not persistent, so it is cancelled (SBML
    4.11.3) and x keeps E2's 2. With E1 persistent it executes and x = 1."""
    text = (
        "x = 0; g = 1;\n"
        "E1: at 2 after (time >= 0 && g > 0.5), t0=false, persistent={persistent}, "
        "priority = 1: x = 1;\n"
        "E2: at 1 after (time >= 1), priority = 10: x = 2, g = 0;\n"
    )
    assert _end(text.format(persistent="false")) == 2.0
    assert _end(text.format(persistent="true")) == 1.0


@pytest.mark.parametrize("from_trigger, n", [("false", 3.0), ("true", 1.0)])
def test_an_event_a_delayed_execution_triggers_fires_on_the_same_instant(from_trigger, n):
    """E1's delayed assignment makes E3's trigger true at 2, and E3 has no
    delay: it executes at 2 as well, after E2, which was already due and
    outranks it. Reading at execution E3 sees E2's x = 2; with its value
    frozen at its own trigger, between E1 and E2, it sees x = 0."""
    text = (
        "x = 0; g = 0; n = 0;\n"
        "E1: at 2 after (time >= 0), t0=false, priority = 10: g = 1;\n"
        "E2: at 1 after (time >= 1), priority = 5: x = 2;\n"
        f"E3: at (g > 0.5), priority = 1, fromTrigger={from_trigger}: n = x + 1;\n"
    )
    assert _end(text, "n") == n


@pytest.mark.parametrize("n_points", [2, 5, 41, 401])
def test_the_order_does_not_depend_on_the_output_grid(n_points):
    assert _end(_A.format(p1=1, p2=10), n_points=n_points) == 1.0
    assert _end(_C.format(pi=1, pd=10), n_points=n_points) == 1.0


def test_the_trajectory_after_the_instant_is_the_last_writers():
    """X decays from what the instant left. E2 (priority 10) sets X = 8 and
    E1 (priority 1) then halves it: X(4) = 4 exp(-0.5 * 2). In the other order
    it would be 8 exp(-1)."""
    text = (
        "species X = 1; J: X -> ; 0.5*X;\n"
        "E1: at 2 after (time >= 0), t0=false, priority = 1, fromTrigger=false: X = X/2;\n"
        "E2: at 1 after (time >= 1), priority = 10: X = 8;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 4.0), n_points=5, rtol=1e-10, atol=1e-12
    )
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(4.0 * np.exp(-1.0), rel=1e-8)


# ── What the first review found ─────────────────────────────────────────────
# A's trigger is true from 0.5, so an execution of it waits on 3.5. At t = 2
# one event drops the trigger and another raises it again, within one instant.
# The fall cancels the waiting execution; the rise starts a new wait, to 5.
_FALL_AND_RISE = {
    "both_delayed": (
        "B: at 1 after (time >= 1), priority = 10: g = 0;\n"
        "C: at 1 after (time >= 1), priority = 5: g = 1;\n"
    ),
    "a_cascade": "B: at 1 after (time >= 1): g = 0;\nC: at (g < 0.5): g = 1;\n",
    "both_immediate": (
        "B: at (time >= 2), priority = 10: g = 0;\nC: at (time >= 2), priority = 5: g = 1;\n"
    ),
}
_TIMES = [0.0, 1.0, 3.0, 4.0, 4.9, 5.5, 6.0]


def _x_at(text: str) -> list[float]:
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(sample_times=_TIMES)
    return [float(v) for v in np.asarray(r.observables)[:, list(r.observable_names).index("x")]]


@pytest.mark.parametrize("others", _FALL_AND_RISE.values(), ids=_FALL_AND_RISE.keys())
def test_a_fall_of_the_trigger_inside_a_batch_cancels_a_waiting_execution(others):
    """Read only once the batch was over, the trigger was true again and the
    execution it had lost at 2 ran at 3.5: two executions for one (SBML
    section 4.11.3; libRoadRunner cancels it too)."""
    a = (
        "x = 0; g = 1;\n"
        "A: at 3 after (time >= 0.5 && g > 0.5), persistent=false, fromTrigger=false: "
        "x = x + 1;\n"
    )
    assert _x_at(a + others) == [0, 0, 0, 0, 0, 1, 1]


def test_a_persistent_event_keeps_the_execution_through_the_fall():
    a = (
        "x = 0; g = 1;\n"
        "A: at 3 after (time >= 0.5 && g > 0.5), persistent=true, fromTrigger=false: x = x + 1;\n"
    )
    assert _x_at(a + _FALL_AND_RISE["both_delayed"]) == [0, 0, 0, 1, 1, 2, 2]


# ── What the second review found ────────────────────────────────────────────
def test_a_fall_read_by_the_root_search_cancels_a_waiting_execution():
    """The clock half of E's trigger falls at 3, on the instant an event
    raises the other half. The root search read the fall and moved the
    baseline the batch compares against, so the batch saw no fall, and after
    it the trigger was true again: E executed at 6 and at 8, for once at 8."""
    text = (
        "x = 0; q = 0;\n"
        "E: at 5 after (((time >= 1) && (time < 3)) || (q > 0.5)), persistent=false, "
        "fromTrigger=false: x = x + 1;\n"
        "F: at (time >= 3): q = 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(model, method="ode").run(sample_times=[0.0, 5.5, 6.5, 7.5, 8.5, 9.5])
    x = [float(v) for v in np.asarray(r.observables)[:, list(r.observable_names).index("x")]]
    assert x == [0, 0, 0, 0, 1, 1]


@pytest.mark.parametrize("seed", [None, 0, 1, 2, 3, 4, 5])
def test_events_with_no_priority_keep_the_order_they_had(seed):
    """An immediate event and a delayed one due on the same instant, neither
    with a priority: SBML leaves the order open, and the immediate one goes
    first, as before. Drawn, the answer of every model without priorities
    would have changed with the seed."""
    text = "x = 0;\nD: at 1 after (time >= 1): x = 1;\nI: at (time >= 2): x = 2;\n"
    kw = {} if seed is None else {"seed": seed}
    assert _end(text, **kw) == 1.0


# ── What the third review found ─────────────────────────────────────────────
def test_many_idle_non_persistent_events_cost_one_pass_over_what_waits():
    """300 non-persistent events that never trigger, beside a delayed event
    with hundreds of executions waiting: each read of the triggers made one
    pass over the waiting executions per idle event (seven times the run
    time). The count is the same and the run is short."""
    import time

    idle = "".join(f"G{j}: at (x > {5 + j}), persistent=false: x = 0;\n" for j in range(300))
    text = (
        "n = 0; x = 0;\n"
        "E: at 200 after (time - floor(time) >= 0.5), fromTrigger=false: n = n + 1;\n" + idle
    )
    model = bngsim.Model.from_antimony_string(text)
    t0 = time.perf_counter()
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 600.0), n_points=3)
    elapsed = time.perf_counter() - t0
    n = float(np.asarray(r.observables)[-1, list(r.observable_names).index("n")])
    assert n == 400.0
    assert elapsed < 5.0


# ── What the fourth review found ────────────────────────────────────────────
def test_many_waiting_executions_are_not_passed_over_at_every_stop():
    """Two thousand executions waiting and a stop every quarter of a time
    unit. Each stop made several passes over what waited (for what was due,
    for what had lapsed, for the next target), and a batch copied the whole
    queue: twice the time of the run before, where the earliest time anything
    waits for, read once after each change to the queue, is all a stop needs.
    """
    import time

    text = (
        "n = 0; q = 0;\n"
        "E: at 2000 after (time - floor(time) >= 0.5), fromTrigger=false: n = n + 1;\n"
        "G: at (time - 0.25*floor(time/0.25) >= 0.125), fromTrigger=false: q = q + 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    t0 = time.perf_counter()
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 8000.0), n_points=3)
    elapsed = time.perf_counter() - t0
    names = list(r.observable_names)
    assert float(np.asarray(r.observables)[-1, names.index("n")]) == 6000.0
    assert float(np.asarray(r.observables)[-1, names.index("q")]) == 32000.0
    assert elapsed < 20.0


def test_a_chain_of_executions_inside_one_stretch_runs_link_by_link():
    """An event that re-arms itself with a delay of a few doubles, the whole
    chain inside the stretch CVODE returns a stop in place of. Each link is
    read at its own time, and there are more links than executions waiting
    when the stop is read."""
    step = 16 * 2.220446049250313e-16 * 2.0
    text = (
        "n = 0; go = 0; dly = 1;\n"
        "S: at (time >= 2), fromTrigger=false: go = 1;\n"
        "C: at dly after (go > 0.5 && n < 40), fromTrigger=false, persistent=true: "
        "n = n + 1, go = 0;\n"
        "R: at (go < 0.5 && n > 0 && n < 40), fromTrigger=false: go = 1;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("dly", step)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 4.0), n_points=3)
    names = list(r.observable_names)
    assert float(np.asarray(r.observables)[-1, names.index("n")]) == 40.0
