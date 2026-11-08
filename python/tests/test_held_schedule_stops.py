"""A schedule that reads a species only events write is stopped at stretch by
stretch (issue #1244).

``X' = piecewise(1, time - 24*floor(time/24) < w, 0)`` is a pulse of width w
at the start of each day, and an event at ``time >= 50`` sets w to 2. w is no
run constant, since an event writes it, so the resolvers placed no stop for
the condition at all: CVODE stepped over every pulse but the one the run
starts on, before the event as well as after, and X(240) came to 1 for 17 at
every tolerance, with nothing said.

Between two of its events w holds. Where the events fire at fixed times and
set values known before the run, each stretch between them is resolved with w
as its number there. Where they do not, a condition that changes more than
once is refused by name, and one that changes once is read as before.

Oracle: the pulse train integrated exactly, piece by piece between every edge
and every change of w.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

PULSE = "piecewise(1, (time - 24*floor(time/24)) < w, 0)"


def _model(events: str, law: str = PULSE, extra: str = "") -> bngsim.Model:
    return bngsim.Model.from_antimony_string(
        f"""
model m
  compartment C = 1
  species X in C
  X = 0
  w = 1
  a = 1
  w2 = 2
  {extra}
  J0: -> X; {law}
  {events}
end"""
    )


def _exact(changes, t_end: float, t_start: float = 0.0, w0: float = 1.0) -> float:
    """X(t_end) for a pulse of width w(t) at the start of each day, w set to
    each (t, w) of *changes* from t on."""
    ws = [(t_start, w0), *sorted(changes)]
    edges = {t_start, t_end} | {t for t, _ in ws}
    for day in range(int(t_start // 24), int(t_end // 24) + 2):
        edges |= {24.0 * day} | {24.0 * day + w for _, w in ws}
    edges = sorted(e for e in edges if t_start <= e <= t_end)
    total = 0.0
    for lo, hi in zip(edges, edges[1:], strict=False):
        mid = 0.5 * (lo + hi)
        w = [v for t, v in ws if t <= mid][-1]
        total += (hi - lo) * float(mid - 24 * np.floor(mid / 24) < w)
    return total


def _x(model, times, **kw) -> np.ndarray:
    result = bngsim.Simulator(model, method="ode", **kw.pop("sim", {})).run(
        sample_times=list(times), **kw
    )
    return np.asarray(result.species)[:, list(result.species_names).index("X")]


@pytest.mark.parametrize("rtol", [1e-2, 1e-6, 1e-10])
def test_the_issue_s_schedule_is_stopped_at(rtol):
    """3 pulses of width 1 by t = 50, then width 2 from 72 on: 3, 5 and 17,
    where main gave 1, 1 and 1."""
    x = _x(_model("E1: at (time >= 50): w = 2"), [0, 49, 74, 240], rtol=rtol, atol=rtol / 100)
    np.testing.assert_allclose(x, [0, 3, 5, 17], rtol=10 * rtol, atol=10 * rtol)


@pytest.mark.parametrize(
    ("events", "changes"),
    [
        ("E1: at 3 after (time >= 50): w = 2", [(53, 2)]),
        # inside the pulse at 72: it runs on to 74
        ("E1: at (time >= 72.5): w = 2", [(72.5, 2)]),
        ("E1: at (time > 50): w = 2", [(50, 2)]),
        ("E1: at (time >= 50): w = 2\n  E2: at (time >= 130): w = 0.5", [(50, 2), (130, 0.5)]),
        # a value that reads the species itself, as it holds then
        ("E1: at (time >= 50): w = w + 1\n  E2: at (time >= 130): w = w + 1", [(50, 2), (130, 3)]),
        # a run constant
        ("E1: at (time >= 50): w = w2", [(50, 2)]),
    ],
    ids=["delayed", "inside-a-pulse", "strict", "two-events", "reads-itself", "a-parameter"],
)
def test_each_stretch_is_stopped_at(events, changes):
    x = _x(_model(events), [0, 240], rtol=1e-8, atol=1e-10)
    assert x[-1] == pytest.approx(_exact(changes, 240), rel=1e-6)


@pytest.mark.parametrize(("initial", "w"), [("", 1.0), (", t0=false", 2.0)])
def test_a_run_that_starts_after_the_trigger(initial, w):
    """From t = 60 the trigger time >= 50 holds already: with initialValue
    true it never rises, and w stays 1; with false it fires at the start."""
    x = _x(_model(f"E1: at (time >= 50){initial}: w = 2"), [60, 240], rtol=1e-8, atol=1e-10)
    assert x[-1] - x[0] == pytest.approx(_exact([], 240, t_start=60, w0=w), rel=1e-6)


def test_a_sensitivity_run_is_stopped_at_too():
    """dX/da = X/a with the rate a times the pulse: 17 and 17, where main gave
    1 and 1."""
    for events in ("E1: at (time >= 50): w = 2", "E1: at (time >= 50): w = w2"):
        result = bngsim.Simulator(
            _model(events, law=f"a*{PULSE}"), method="ode", sensitivity_params=["a"]
        ).run(sample_times=[0, 240], rtol=1e-8, atol=1e-10)
        assert np.asarray(result.species)[-1, 0] == pytest.approx(17, rel=1e-6)
        assert np.asarray(result.sensitivities)[-1, 0, 0] == pytest.approx(17, rel=1e-5)


@pytest.mark.parametrize(
    ("events", "sens", "why"),
    [
        ("E1: at (X > 2): w = 2", None, "a trigger that is no fixed time"),
        ("E1: at (time >= 50): w = X", None, "sets it to X, which is not known before the run"),
        ("E1: at (time >= 50): w = w2", ["w2"], "which a requested sensitivity parameter moves"),
    ],
    ids=["a-state-trigger", "a-value-of-the-state", "a-requested-parameter"],
)
def test_a_schedule_not_known_before_the_run_is_refused(events, sens, why):
    """Main stepped over these as over the issue's: X(240) was 1, and dX/dw2
    2e-5 for 7."""
    sim = {"sensitivity_params": sens} if sens else {}
    with pytest.raises(bngsim.SimulationError, match="issue #1244") as e:
        _x(_model(events), [0, 240], rtol=1e-6, atol=1e-8, sim=sim)
    assert why in str(e.value)
    assert "reads w, and only events set w" in str(e.value)


def test_a_condition_that_changes_once_is_read_as_before():
    """``time < toff``, with toff set from 10 to 30 when X reaches 0.5: one
    change as the run starts, which the condition's root sees. Not refused,
    and as main: X(240) = 30."""
    x = _x(
        _model(
            "E1: at (X > 0.5): toff = 30", law="piecewise(1, time < toff, 0)", extra="toff = 10"
        ),
        [0, 240],
        rtol=1e-8,
        atol=1e-10,
    )
    assert x[-1] == pytest.approx(30, rel=1e-6)


def test_a_schedule_no_event_writes_is_as_before():
    x = _x(_model(""), [0, 49, 74, 240], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(x, [0, 3, 4, 10], rtol=1e-5)


# ── A run that continues the last one ───────────────────────────────────────
#
# A run that starts where the last one ended continues its events
# (NetworkModel.event_carry, issue #693): each trigger's truth there, and the
# delayed executions not yet applied. The schedule was read as a fresh start's:
# a trigger true at the start with initialValue false was taken to fire again,
# and an execution delayed into this run was not seen, so the stops were placed
# for a w the run never holds. Leg 2 of `w = w + 1`, t0=false, came to 17.8 for
# 20 at rtol 1e-2, and a delayed execution carried into it to 16.6 for 19.


def _legs(events, rtol):
    model = _model(events)
    sim = bngsim.Simulator(model, method="ode")
    sim.run(sample_times=[0, 240], rtol=rtol, atol=rtol / 100)
    x = np.asarray(sim.run(sample_times=[240, 480], rtol=rtol, atol=rtol / 100).species)[:, 0]
    return x[-1] - x[0], model


@pytest.mark.parametrize("rtol", [1e-2, 1e-3, 1e-8])
@pytest.mark.parametrize("initial", ["", ", t0=false"])
def test_a_continued_run_does_not_fire_a_trigger_that_stays_true(rtol, initial):
    """w = 2 from 50 on, and the trigger is still true at 240: no second fire.
    Leg 2 is ten pulses of width 2."""
    gain, model = _legs(f"E1: at (time >= 50){initial}: w = w + 1", rtol)
    assert model.get_concentration("w") == 2.0
    assert gain == pytest.approx(_exact([], 480, t_start=240, w0=2.0), rel=10 * rtol)


@pytest.mark.parametrize("rtol", [1e-2, 1e-3, 1e-8])
def test_a_continued_run_takes_the_execution_the_last_one_delayed(rtol):
    """Triggered at 230 with its value frozen there (w + 1 = 2), executed at
    250, in leg 2: w = 1 until 250, then 2."""
    gain, model = _legs("E1: at 20 after (time >= 230): w = w + 1", rtol)
    assert model.get_concentration("w") == 2.0
    assert gain == pytest.approx(_exact([(250, 2.0)], 480, t_start=240, w0=1.0), rel=10 * rtol)


@pytest.mark.parametrize("rtol", [1e-2, 1e-3, 1e-8])
def test_a_carried_execution_sets_the_value_it_froze(rtol):
    """E1's value is frozen at its trigger, 230 (w + 1 = 2). E2 sets w to 5 at
    245, before E1 executes at 250, so read then w + 1 would be 6: the run
    holds 1, 5 from 245, and 2 from 250."""
    events = "E1: at 20 after (time >= 230): w = w + 1\n  E2: at (time >= 245): w = 5"
    gain, model = _legs(events, rtol)
    assert model.get_concentration("w") == 2.0
    exact = _exact([(245, 5.0), (250, 2.0)], 480, t_start=240, w0=1.0)
    assert gain == pytest.approx(exact, rel=10 * rtol)


# ── Review of the first two commits ─────────────────────────────────────────


def test_a_schedule_quiet_at_its_start_is_refused():
    """w = 0 as the run starts, so `mod(time, 24) < w` never holds then, and a
    state trigger sets w = 1 at t = 20. Read only at its start the condition
    changed no times and was let through: Y(240) came to 0 for 9, the issue
    again with nothing said. Read with w = 1 too, the number E1 sets, it
    changes 20 times, and the run is refused."""
    model = bngsim.Model.from_antimony_string(
        f"""
model m
  compartment C = 1
  species X in C, Y in C
  X = 0; Y = 0
  w = 0
  J0: -> X; 0.1
  J1: -> Y; {PULSE}
  E1: at (X > 2): w = 1
end"""
    )
    with pytest.raises(bngsim.SimulationError, match="issue #1244") as e:
        bngsim.Simulator(model, method="ode").run(sample_times=[0, 240], rtol=1e-6, atol=1e-8)
    assert "with w at 1.0, which an event sets it to" in str(e.value)


def test_a_condition_only_an_output_reads_is_not_refused():
    """The pulse is an output's alone: no rate law reads it, and a step across
    one of its windows moves nothing. Main ran it; the first commits refused
    it as a rate law's switch."""
    model = bngsim.Model.from_antimony_string(
        f"""
model m
  compartment C = 1
  species X in C
  X = 0
  w = 1
  y := {PULSE}
  J0: -> X; 0.05
  E1: at (X > 2): w = 2
end"""
    )
    x = _x(model, [0, 240], rtol=1e-8, atol=1e-10)
    assert x[-1] == pytest.approx(12, rel=1e-6)


def test_a_value_read_through_a_derived_parameter_is_known():
    """k2 = 2*w2 is a constant of the run, so `w = k2` sets a known value:
    w = 4 from 50 on. It was refused as not known before the run."""
    x = _x(_model("E1: at (time >= 50): w = k2", extra="k2 = 2*w2"), [0, 240], rtol=1e-8)
    assert x[-1] == pytest.approx(_exact([(50, 4.0)], 240), rel=1e-6)


def test_a_held_species_that_holds_over_the_run_reads_as_a_run_constant():
    """w's one event is past the end, so w is 1 throughout, and the run must be
    the one where w is a parameter, with an event elsewhere. A stretch's
    condition, with w written as its number, was not found in the rate law,
    so a ramp times the pulse was taken to hold between its changes and its
    stops moved a double early: 1088.15 for that run's 1058.25 at rtol 1e-2."""
    law = f"time*{PULSE}"
    held = _x(_model("E1: at (time >= 1000): w = 2", law=law), [0, 240], rtol=1e-2, atol=1e-4)
    constant = _x(
        _model("E1: at (time >= 1000): Z = 2", law=law, extra="Z = 1"),
        [0, 240],
        rtol=1e-2,
        atol=1e-4,
    )
    assert held[-1] == constant[-1]


def _stretches_past(budget, monkeypatch, events, t_end):
    from bngsim import _switch_sensitivity

    # The resolvers' memo keeps an answer made against the budget, a refusal
    # included, and outlives the test: under the lowered budget, 0..240 at
    # w = 1 is remembered as past it, and a later run of that window on the
    # same worker was refused at the real budget. So this test has its own,
    # and monkeypatch hands back the untouched one (test_crossing_stop_budget's
    # fixture empties it on both sides for the same reason).
    monkeypatch.setattr(_switch_sensitivity, "_SCHEDULE_CACHE", {})
    monkeypatch.setattr(_switch_sensitivity, "_RUN_STOP_BUDGET", budget)
    model = _model(events)
    past: list = []
    fixed, stops, _ = _switch_sensitivity.crossings_with_periodic(
        model._core,
        0.0,
        t_end,
        model.time_discontinuity_conditions(),
        touches=model._event_disc_conditions,
        over_budget=past,
    )
    return len(fixed) + len(stops), past


def test_the_stretches_share_the_run_s_one_budget(monkeypatch):
    """Three stretches of 13 or 14 changes each, against a budget of 30: each
    stretch fit its own, so all 40 were placed, past the budget the run has.
    (At the real budget, 1,200,002 stops were placed for 1,000,000.)"""
    placed, past = _stretches_past(
        30, monkeypatch, "E1: at (time >= 160): w = 2\n  E2: at (time >= 320): w = 1", 480.0
    )
    assert placed <= 30
    assert [item.condition for item in past] == ["((time()-(24*floor((time()/24))))<w)"]


def test_a_condition_past_the_budget_in_two_stretches_keeps_its_shortest_gap(monkeypatch):
    """Past a budget of 10 before t = 240, where its windows are 1 wide, and
    after, where they are 3: the step bound is made from the shortest gap, 1.
    The last stretch's alone, 3, let a step span a window before 240."""
    _, past = _stretches_past(10, monkeypatch, "E1: at (time >= 240): w = 3", 480.0)
    assert len(past) == 1
    assert past[0].changes == 40
    assert past[0].min_gap == pytest.approx(1.0)
