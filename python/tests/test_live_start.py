"""Where a run starts, recorded on its Result and warned of (issue #1701).

The state lives in the Model, and every ODE or stochastic run writes its end
state back there, so a second ``run()`` continues from where the first one
ended. That is BNG2.pl's behaviour and stays. What #1701 adds is that a Result
says which start it had (``started_from``, ``start_state_sha256``), that
``run(start=...)`` chooses it, and that a run restarting the clock on exactly
the state the last run left raises :class:`bngsim.LiveStartWarning` first.

The oracle is the decay ``A -> B`` at ``k = 1`` from ``A = 10``: A(t) = 10 e^-t.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert._protocol import Experiment, ProtocolSpec

DECAY_NET = (
    "begin parameters\n 1 k 1\nend parameters\n"
    "begin species\n 1 A() 10\n 2 B() 0\nend species\n"
    "begin reactions\n 1 1 2 k\nend reactions\n"
    "begin groups\n 1 A 1\nend groups\n"
)
# A(10) is 4.5e-4: the absolute tolerance (1e-8) bounds its error, not rtol.
ATOL_BOUND = 1e-7
# Counts, for the stochastic methods: 1000 molecules decay visibly in t = 1.
COUNTS_NET = DECAY_NET.replace(" 1 A() 10\n", " 1 A() 1000\n")


def _model(tmp_path: Path, text: str = DECAY_NET) -> bngsim.Model:
    path = tmp_path / "decay.net"
    path.write_text(text, encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _sha256(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _live_start_warnings(caught) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, bngsim.LiveStartWarning)]


def _run(sim, **kwargs):
    """``sim.run(**kwargs)``, and the LiveStartWarnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(**kwargs)
    return result, _live_start_warnings(caught)


# ─── The issue's case ───────────────────────────────────────────────────────


def test_a_second_run_over_the_same_span_is_recorded_and_warned_of(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, w1 = _run(sim, t_span=(0, 5), n_points=6)
    r2, w2 = _run(sim, t_span=(0, 5), n_points=6)

    a1, a2 = r1.observables["A"], r2.observables["A"]
    assert a1[0] == 10.0
    assert a1[-1] == pytest.approx(10 * math.exp(-5), rel=1e-6)
    # The second run continued the first, on the same time axis: the behaviour.
    assert a2[0] == a1[-1]
    assert a2[-1] == pytest.approx(10 * math.exp(-10), abs=ATOL_BOUND)
    np.testing.assert_array_equal(r1.time, r2.time)

    # What is new: the Results say so, and the second run was warned of.
    assert (r1.started_from, r2.started_from) == ("initial", "live")
    assert r1.start_state_sha256 == _sha256([10.0, 0.0])
    assert r2.start_state_sha256 == _sha256(r1.species[-1])
    assert w1 == []
    assert len(w2) == 1
    text = str(w2[0].message)
    assert "t = 0 from the state the last run left at t = 5" in text
    assert "start='initial'" in text and "start='live'" in text
    # A first-time caller sees this warning more than any other; it says what to
    # do in plain words, with no issue number to look up.
    assert "#" not in text
    # It names the caller's line, not one of bngsim's.
    assert w2[0].filename == __file__

    s1, s2 = r1.summary(), r2.summary()
    differ = {k for k in s1 if s1[k] != s2[k]}
    assert {"started_from", "start_state_sha256"} <= differ
    json.dumps(s2)  # still JSON-serializable


def test_start_initial_reruns_the_first_run_exactly(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    r2, w2 = _run(sim, t_span=(0, 5), n_points=6, start="initial")
    assert w2 == []
    np.testing.assert_array_equal(r2.species, r1.species)
    assert r2.started_from == "initial"
    assert r2.start_state_sha256 == r1.start_state_sha256
    # The parameters stay as they are: only the state is reset.
    model.set_param("k", 2.0)
    r3, _ = _run(sim, t_span=(0, 5), n_points=6, start="initial")
    assert r3.observables["A"][-1] == pytest.approx(10 * math.exp(-10), abs=ATOL_BOUND)


def test_start_live_continues_with_no_warning(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    r2, w2 = _run(sim, t_span=(0, 5), n_points=6, start="live")
    assert w2 == []
    assert r2.observables["A"][0] == r1.observables["A"][-1]
    assert r2.started_from == "live"


@pytest.mark.parametrize(
    ("method", "options"),
    [("ode", {}), ("ssa", {}), ("psa", {"poplevel": 100.0})],
)
def test_every_method_that_writes_its_state_back_is_warned_of(tmp_path, method, options):
    model = _model(tmp_path, COUNTS_NET)
    sim = bngsim.Simulator(model, method=method, **options)
    r1, w1 = _run(sim, t_span=(0, 1), n_points=3, seed=1)
    r2, w2 = _run(sim, t_span=(0, 1), n_points=3, seed=1)
    assert w1 == [] and len(w2) == 1
    assert r2.observables["A"][0] == r1.observables["A"][-1] < 1000
    assert (r1.started_from, r2.started_from) == ("initial", "live")
    # From the same state with the same seed, a stochastic run is the same run.
    r3, w3 = _run(sim, t_span=(0, 1), n_points=3, seed=1, start="initial")
    assert w3 == []
    np.testing.assert_array_equal(r3.species, r1.species)


# ─── What is not warned of ──────────────────────────────────────────────────


def test_a_run_that_continues_the_clock_is_not_warned_of(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    _run(sim, t_span=(0, 5), n_points=6)
    r2, w2 = _run(sim, t_span=(5, 10), n_points=6)
    assert w2 == []
    assert r2.started_from == "live"
    assert r2.observables["A"][-1] == pytest.approx(10 * math.exp(-10), abs=ATOL_BOUND)
    # sample_times from the time the last run ended at continue it too.
    r3, w3 = _run(sim, sample_times=[10.0, 12.0])
    assert w3 == []
    assert r3.observables["A"][-1] == pytest.approx(10 * math.exp(-12), abs=ATOL_BOUND)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.set_concentration("A()", 3.0),
        lambda m: m.set_state(np.array([2.0, 8.0])),
    ],
    ids=["set_concentration", "set_state"],
)
def test_a_state_changed_since_the_run_is_not_warned_of(tmp_path, change):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    _run(sim, t_span=(0, 5), n_points=6)
    change(model)
    r2, w2 = _run(sim, t_span=(0, 5), n_points=6)
    assert w2 == []
    assert r2.started_from == "live"


def test_a_reset_between_runs_starts_from_the_initial_conditions(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    model.reset()
    r2, w2 = _run(sim, t_span=(0, 5), n_points=6)
    assert w2 == []
    assert r2.started_from == "initial"
    np.testing.assert_array_equal(r2.species, r1.species)


def test_a_saved_baseline_is_the_initial_state(tmp_path):
    # After an unlabeled save_concentrations the reset target is the saved
    # state, so a run from it starts from the "initial" conditions.
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    _run(sim, t_span=(0, 5), n_points=6)
    model.save_concentrations()
    r2, w2 = _run(sim, t_span=(0, 5), n_points=6)
    assert w2 == []
    assert r2.started_from == "initial"


def test_run_until_steps_are_not_warned_of(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        steps = [sim.run_until(t) for t in (1.0, 2.0, 3.0)]
        # A rollback to an earlier time is a step the caller chose.
        sim.set_state(steps[0].species[-1], time=1.0)
        again = sim.run_until(2.0)
    assert _live_start_warnings(caught) == []
    assert [s.started_from for s in steps] == ["initial", "live", "live"]
    assert again.observables["A"][-1] == pytest.approx(steps[1].observables["A"][-1], rel=1e-6)


def test_scans_choose_their_starts_and_put_the_record_back(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forward = sim.parameter_scan("k", [0.5, 1.0, 2.0], t_span=(0, 5), n_points=6)
        onward = sim.bifurcate("k", [0.5, 1.0, 2.0], t_span=(0, 5), n_points=6)
    assert _live_start_warnings(caught) == []
    assert [r.started_from for r in forward] == ["initial"] * 3
    # A continuation scan continues from each point's end on purpose.
    assert [r.started_from for r in onward] == ["initial", "live", "live"]


@pytest.mark.parametrize("scan", ["parameter_scan", "bifurcate"])
def test_a_scan_after_a_run_is_warned_of_once(tmp_path, scan):
    # Every point (or the first, continuing) starts from the run's end at t = 0:
    # the issue's mistake, once per call, not once per point.
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        points = getattr(sim, scan)("k", [0.5, 1.0, 2.0], t_span=(0, 5), n_points=6)
    hits = _live_start_warnings(caught)
    assert len(hits) == 1 and hits[0].filename == __file__
    assert "point of this scan starts at t = 0" in str(hits[0].message)
    assert points[0].observables["A"][0] == r1.observables["A"][-1]
    # The scan leaves the model as it found it, the record of the run before it
    # included, so the run again is warned of too.
    _, w = _run(sim, t_span=(0, 5), n_points=6)
    assert len(w) == 1


def test_a_named_reset_target_is_a_start_the_caller_chose(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    _run(sim, t_span=(0, 5), n_points=6)
    model.save_concentrations("equilibrated")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.parameter_scan("k", [0.5, 2.0], t_span=(0, 5), n_points=6, reset_to="equilibrated")
    assert _live_start_warnings(caught) == []


def test_replicates_after_a_run_are_warned_of_and_recorded(tmp_path):
    model = _model(tmp_path, COUNTS_NET)
    sim = bngsim.Simulator(model, method="ssa")
    r1, _ = _run(sim, t_span=(0, 1), n_points=3, seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reps = sim.run_replicates(3, t_span=(0, 1), n_points=3, seed=2)
    hits = _live_start_warnings(caught)
    assert len(hits) == 1 and hits[0].filename == __file__
    assert [r.observables["A"][0] for r in reps] == [r1.observables["A"][-1]] * 3
    assert [r.started_from for r in reps] == ["live"] * 3
    assert {r.start_state_sha256 for r in reps} == {_sha256(r1.species[-1])}
    # From the initial conditions: recorded so, and not warned of.
    model.reset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh = sim.run_replicates(2, t_span=(0, 1), n_points=3, seed=2)
    assert _live_start_warnings(caught) == []
    assert [r.started_from for r in fresh] == ["initial"] * 2
    assert {r.start_state_sha256 for r in fresh} == {_sha256([1000.0, 0.0])}


def test_a_protocols_runs_continue_without_a_warning(tmp_path):
    # BNG2.pl's successive simulate actions continue the state, from whatever
    # time each names: what run_protocol replays.
    protocol = ProtocolSpec(
        steps=(
            Experiment("simulate", "ode", (0.0, 5.0), 6),
            Experiment("simulate", "ode", (0.0, 5.0), 6),
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay = bngsim.run_protocol(_model(tmp_path), protocol)
    assert _live_start_warnings(caught) == []
    first, second = (run.result for run in replay)
    assert (first.started_from, second.started_from) == ("initial", "live")
    assert second.observables["A"][-1] == pytest.approx(10 * math.exp(-10), abs=ATOL_BOUND)


def test_a_protocol_on_a_model_a_run_advanced_is_warned_of_once(tmp_path):
    # Its first run starts from the run's end at t = 0; its second continues
    # its own first, as BNG2.pl's does, and is not warned of.
    model = _model(tmp_path)
    r1, _ = _run(bngsim.Simulator(model, method="ode"), t_span=(0, 5), n_points=6)
    protocol = ProtocolSpec(
        steps=(
            Experiment("simulate", "ode", (0.0, 5.0), 6),
            Experiment("simulate", "ode", (0.0, 5.0), 6),
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay = bngsim.run_protocol(model, protocol)
    hits = _live_start_warnings(caught)
    assert len(hits) == 1 and hits[0].filename == __file__
    assert "protocol's first run" in str(hits[0].message)
    first = replay[0].result
    assert first.started_from == "live"
    assert first.observables["A"][0] == r1.observables["A"][-1]


def test_a_zero_length_protocol_run_records_its_start(tmp_path):
    # t_end not past t_start: BNG2.pl writes the one row at t_start, and warns.
    protocol = ProtocolSpec(steps=(Experiment("simulate", "ode", (2.0, 2.0), 2),))
    with pytest.warns(bngsim.ConversionWarning, match="not past t_start"):
        replay = bngsim.run_protocol(_model(tmp_path), protocol)
    only = replay[0].result
    assert only.n_times == 1
    assert (only.started_from, only.start_state_sha256) == ("initial", _sha256([10.0, 0.0]))


def test_a_run_from_float_noise_on_the_end_time_continues_it(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    _run(sim, t_span=(0, 0.1 + 0.2), n_points=4)  # ends at 0.30000000000000004
    _, w = _run(sim, t_span=(0.3, 1.0), n_points=4)
    assert w == []


def test_a_run_from_after_the_end_time_continues_it(tmp_path):
    # A steady-state run stops before its span's end; the next run from the
    # planned end continues it, the clock moving on, not back.
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    r1, _ = _run(sim, t_span=(0, 1000), n_points=101, steady_state=True)
    assert r1.time[-1] < 1000
    r2, w = _run(sim, t_span=(1000, 2000), n_points=3)
    assert w == []
    assert r2.started_from == "live"


def test_set_time_says_when_the_state_is(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    _run(sim, t_span=(0, 5), n_points=6)
    sim.set_time(0.0)
    _, w = _run(sim, t_span=(0, 5), n_points=6)
    assert w == []


def test_a_second_simulators_first_step_is_asked_against_its_clock(tmp_path):
    model = _model(tmp_path)
    _run(bngsim.Simulator(model, method="ode"), t_span=(0, 5), n_points=6)
    other = bngsim.Simulator(model, method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step = other.run_until(1.0, n_points=2)
    hits = _live_start_warnings(caught)
    assert len(hits) == 1 and hits[0].filename == __file__
    assert "run_until step" in str(hits[0].message) and "set_time" in str(hits[0].message)
    assert step.started_from == "live"
    # set_time puts the clock where the state is, and the step continues it.
    third = bngsim.Simulator(model, method="ode")
    third.set_time(step.time[-1] + 4.0)  # the first run ended at 5, then 1 more
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        third.run_until(7.0, n_points=2)
    assert _live_start_warnings(caught) == []


# ─── Other simulators and copies of the model ───────────────────────────────


def test_a_second_simulator_on_the_model_is_warned_of(tmp_path):
    model = _model(tmp_path)
    _run(bngsim.Simulator(model, method="ode"), t_span=(0, 5), n_points=6)
    other = bngsim.Simulator(model, method="ode")
    r2, w2 = _run(other, t_span=(0, 5), n_points=6)
    assert len(w2) == 1
    assert r2.started_from == "live"


def test_a_clone_holds_the_state_and_the_record(tmp_path):
    model = _model(tmp_path)
    before = model.clone()
    _run(bngsim.Simulator(model, method="ode"), t_span=(0, 5), n_points=6)
    after = model.clone()
    r_before, w_before = _run(bngsim.Simulator(before, method="ode"), t_span=(0, 5), n_points=6)
    r_after, w_after = _run(bngsim.Simulator(after, method="ode"), t_span=(0, 5), n_points=6)
    assert w_before == [] and r_before.started_from == "initial"
    assert len(w_after) == 1 and r_after.started_from == "live"


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


NFSIM = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
def test_a_network_free_run_starts_from_the_model_each_time():
    # NFsim starts each run from the model's state and writes nothing back, so
    # two runs over one span are two runs from the same start, as the same seed
    # shows: not a continuation, and not warned of. The .net and the XML are
    # one model (L0 = 300, R0 = 200), so the record describes the start.
    model = bngsim.Model.from_net(str(NFSIM / "model_parameters.net"))
    sim = bngsim.Simulator(model, method="nf_reject", xml_path=str(NFSIM / "model_parameters.xml"))
    r1, w1 = _run(sim, t_span=(0, 5), n_points=3, seed=1)
    r2, w2 = _run(sim, t_span=(0, 5), n_points=3, seed=1)
    assert w1 == [] and w2 == []
    np.testing.assert_array_equal(r2.observables, r1.observables)
    assert r1.observables["Lfree"][0] == 300.0
    assert (r1.started_from, r2.started_from) == ("initial", "initial")
    assert r1.start_state_sha256 == _sha256(model.get_state())
    # A start the caller set is the one the backend is handed, and recorded.
    model.set_concentration("L(r)", 100.0)
    r3, w3 = _run(sim, t_span=(0, 5), n_points=3, seed=1)
    assert w3 == []
    assert r3.observables["Lfree"][0] == 100.0
    assert r3.started_from == "live"


# ─── The warning's contract ─────────────────────────────────────────────────


def test_an_escalated_warning_stops_the_run_before_it_integrates(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.LiveStartWarning)
        with pytest.raises(bngsim.LiveStartWarning):
            sim.run(t_span=(0, 5), n_points=6)
    np.testing.assert_array_equal(model.get_state(), r1.species[-1])
    assert sim.current_time == 5.0


def test_a_sensitivity_run_the_engine_refuses_is_left_to_its_own_error(tmp_path):
    # From a state a run advanced, a sensitivity run without
    # carry_sensitivities=True is refused with that remedy (GH #210, #1528). A
    # LiveStartWarning first would, under -W error, stop it with the wrong one.
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    sim.run(t_span=(0, 5), n_points=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.LiveStartWarning)
        with pytest.raises(Exception, match="carry_sensitivities=True") as refused:
            sim.run(t_span=(0, 5), n_points=6)
    assert not isinstance(refused.value, bngsim.LiveStartWarning)
    # The way it names works, and the way #1701 adds does too.
    sim.run(t_span=(0, 5), n_points=6, carry_sensitivities=True)
    rerun = sim.run(t_span=(0, 5), n_points=6, start="initial")
    assert rerun.started_from == "initial"


def test_start_is_validated_before_anything_changes(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    with pytest.raises(ValueError, match="start must be None, 'live' or 'initial'"):
        sim.run(t_span=(0, 5), start="fresh")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="carry_sensitivities"):
        sim.run(t_span=(0, 5), start="initial", carry_sensitivities=True)
    # Neither refusal reset the model.
    np.testing.assert_array_equal(model.get_state(), r1.species[-1])


@pytest.mark.parametrize(
    "bad",
    [{"seed": -1}, {"atol": [1e-8, 1e-8, 1e-8]}, {"atol": "nope"}],
    ids=["seed", "atol-length", "atol-token"],
)
def test_a_refused_start_initial_leaves_the_model_and_clock_as_they_were(tmp_path, bad):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    with pytest.raises(ValueError):
        sim.run(t_span=(0, 5), n_points=6, start="initial", **bad)
    np.testing.assert_array_equal(model.get_state(), r1.species[-1])
    assert sim.current_time == 5.0
    # So a step continues from where the state and the clock both are.
    step = sim.run_until(6.0, n_points=2)
    assert step.observables["A"][-1] == pytest.approx(10 * math.exp(-6), abs=ATOL_BOUND)


# ─── The record on derived Results ──────────────────────────────────────────


def test_a_stopped_run_keeps_its_record(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    sim.add_stop_condition("A < 5", label="half")
    with pytest.raises(bngsim.StopConditionMet) as stop:
        sim.run(t_span=(0, 5), n_points=51)
    assert stop.value.result.started_from == "initial"
    assert stop.value.result.start_state_sha256 == _sha256([10.0, 0.0])


def test_a_stack_keeps_what_its_rows_agree_on(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    r2, _ = _run(sim, t_span=(0, 5), n_points=6, start="live")
    r3, _ = _run(sim, t_span=(0, 5), n_points=6, start="initial")
    same = bngsim.Result.squeeze([r1, r3])
    assert (same.started_from, same.start_state_sha256) == ("initial", r1.start_state_sha256)
    mixed = bngsim.Result.squeeze([r1, r2])
    assert (mixed.started_from, mixed.start_state_sha256) == (None, None)
    # Rows that all start from the initial conditions, each from its own state.
    model.reset()
    model.set_concentration("A()", 4.0)
    model.save_concentrations()
    r4, _ = _run(sim, t_span=(0, 5), n_points=6)
    own = bngsim.Result.squeeze([r1, r4])
    assert (own.started_from, own.start_state_sha256) == ("initial", None)


def test_a_refused_start_initial_puts_the_carried_state_back(tmp_path):
    # The reset clears the pending dx/dθ and the carried mark; a refusal must
    # put them back with the state, or the measurement run that follows loses
    # its seed (GH #210).
    def equilibrated():
        model = _model(tmp_path)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
        sim.run(t_span=(0, 5), n_points=6)
        return model, sim

    model, sim = equilibrated()
    core = model._core
    assert core.has_pending_sensitivity_seed and core.ic_state_dirty
    with pytest.raises(ValueError):
        sim.run(t_span=(5, 10), n_points=6, start="initial", atol=[1e-8, 1e-8, 1e-8])
    assert core.has_pending_sensitivity_seed and core.ic_state_dirty
    assert core.advanced_since_baseline
    measured = sim.run(t_span=(5, 10), n_points=6, carry_sensitivities=True)
    _, twin = equilibrated()
    expected = twin.run(t_span=(5, 10), n_points=6, carry_sensitivities=True)
    np.testing.assert_array_equal(measured.sensitivities, expected.sensitivities)
    np.testing.assert_array_equal(measured.species, expected.species)


@pytest.mark.parametrize("n", [2, 32, 33, 500])
def test_both_comparisons_agree_with_numpy(n):
    # _same_values compares in Python up to 32 entries and with numpy above;
    # either is np.array_equal(equal_nan=True), and equal vectors share a digest.
    from bngsim._result import _state_digest
    from bngsim._simulator import _same_values

    rng = np.random.default_rng(n)
    base = rng.random(n)
    variants = {"same": base.copy()}
    signed = base.copy()
    signed[0] = 0.0
    neg = signed.copy()
    neg[0] = -0.0
    variants["zero-sign"] = neg
    nan_a = base.copy()
    nan_a[-1] = np.nan
    variants["nan-both"] = nan_a
    moved = base.copy()
    moved[n // 2] += 1e-12
    variants["moved"] = moved
    pairs = [
        (base, variants["same"]),
        (signed, variants["zero-sign"]),
        (nan_a, nan_a.copy()),
        (base, nan_a),
        (nan_a, base),
        (base, moved),
        (base, base[:-1]),
    ]
    for a, b in pairs:
        expected = a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
        assert _same_values(a, b) == expected
        if expected:
            assert _state_digest(a) == _state_digest(b)


def test_equal_values_have_one_digest(tmp_path):
    # -0.0 equals 0.0, so the start is the initial one and its digest is too.
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    model.set_state(np.array([10.0, -0.0]))
    r2, _ = _run(sim, t_span=(0, 5), n_points=6)
    assert r2.started_from == "initial"
    assert r2.start_state_sha256 == r1.start_state_sha256


def test_save_and_load_keep_the_record(tmp_path):
    pytest.importorskip("h5py")
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    _run(sim, t_span=(0, 5), n_points=6)
    r2, _ = _run(sim, t_span=(0, 5), n_points=6, start="live")
    path = tmp_path / "r2.h5"
    r2.save(path)
    back = bngsim.Result.load(path)
    assert back.started_from == "live"
    assert back.start_state_sha256 == r2.start_state_sha256


def test_to_xarray_carries_the_record(tmp_path):
    pytest.importorskip("xarray")
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    r1, _ = _run(sim, t_span=(0, 5), n_points=6)
    attrs = r1.to_xarray().attrs
    assert attrs["started_from"] == "initial"
    assert attrs["start_state_sha256"] == r1.start_state_sha256


def test_a_scan_whose_hook_sets_the_start_is_not_warned_of(tmp_path):
    # An on_point hook that writes the species sets each point's start, as a
    # named reset target does: a start the caller chose. One that only sets a
    # parameter leaves the run's end as the start, and is warned of.
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    _run(sim, t_span=(0, 5), n_points=6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        points = sim.parameter_scan(
            "k",
            [1.0, 2.0],
            t_span=(0, 5),
            n_points=6,
            on_point=lambda m, v: m.set_concentration("A()", 10.0 * v),
        )
    assert _live_start_warnings(caught) == []
    assert [p.observables["A"][0] for p in points] == [10.0, 20.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.parameter_scan("k", [1.0, 2.0], t_span=(0, 5), n_points=6, on_point=lambda m, v: None)
    assert len(_live_start_warnings(caught)) == 1


def test_another_simulators_clock_does_not_move_the_record(tmp_path):
    # The second Simulator's clock at 0 says nothing of where the first one's
    # run left the state: a rerun of the first one's span is still warned of.
    model = _model(tmp_path)
    first = bngsim.Simulator(model, method="ode")
    _run(first, t_span=(0, 5), n_points=6)
    bngsim.Simulator(model, method="ode").set_time(0.0)
    r, w = _run(first, t_span=(0, 5), n_points=6)
    assert len(w) == 1
    assert r.started_from == "live"


def test_compute_all_sensitivities_records_its_start(tmp_path):
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    fresh = sim.compute_all_sensitivities(t_span=(0, 5), n_points=6, params=["k"])
    assert (fresh.started_from, fresh.start_state_sha256) == ("initial", _sha256([10.0, 0.0]))
    model.set_concentration("A()", 3.0)
    dosed = sim.compute_all_sensitivities(t_span=(0, 5), n_points=6, params=["k"])
    assert dosed.started_from == "live"
    assert dosed.observables["A"][0] == 3.0
