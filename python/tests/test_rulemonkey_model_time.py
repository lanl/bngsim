"""Issue #1022: method='rm' runs rate laws that read the clock, on model time.

RuleMonkey 3.10.1 treated a total propensity of 0 as final, so a rate that is 0
at the start and turns on later never fired: ``if(time()>=6, k, 0)`` returned
20000 on every row (#862), and a table function indexed by time stayed flat
(#892). bngsim refused both on rm. RuleMonkey 3.11.0 integrates a rate that
reads the clock between events (richardposner/RuleMonkey#86), so the refusals
are gone, and it puts the seed state at t_start on the model's clock
(richardposner/RuleMonkey#98), so rm now runs on absolute model time, as ode
and ssa do, instead of on time elapsed since t_start.

Oracles: closed forms (every model here is a set of independent molecules, so
the exact mean and variance of each count are known), network SSA on the
BNG-generated .net, and one-shot runs against sessions under the same seed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._rulemonkey_clock import OFFSET_PARAM as OFFSET

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
NF = DATA / "nfsim"
SWITCH = NF / "time_switch.xml"  # A(s~0) -> A(s~1) at if(time()>=6, 0.5, 0), 20000 A(s~0)
RAMP = NF / "time_ramp"  # A(s~0) -> A(s~1) at 0.02*time(), 1000 A(s~0)
TFUN = NF / "tfun_new_format" / "valid_inline_step_time.xml"


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in")


def _rm(xml: Path | str, seed: int, **times) -> bngsim.Result:
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    return bngsim.Simulator(m, method="rm", xml_path=str(xml)).run(seed=seed, **times)


def _assert_mean(samples: np.ndarray, mean: np.ndarray, var: np.ndarray) -> None:
    """Each column of *samples* (replicates x rows) has the exact *mean*: equal to
    it where the count is certain, within 4.5 standard errors elsewhere."""
    samples = np.asarray(samples, dtype=float)
    certain = var <= 0.0
    np.testing.assert_array_equal(
        samples[:, certain], np.broadcast_to(mean[certain], samples[:, certain].shape)
    )
    se = np.sqrt(var[~certain] / samples.shape[0])
    z = (samples[:, ~certain].mean(axis=0) - mean[~certain]) / se
    assert np.all(np.abs(z) < 4.5), z


# ── #862: a rate that turns on at t = 6 ──────────────────────────────────────


def _switch_survival(t: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.clip(t - 6.0, 0.0, None))


@pytest.mark.parametrize(
    "times",
    [
        {"t_span": (0.0, 10.0), "n_points": 11},
        # Absolute model time: the rate turns on at t = 6 inside the run. On
        # elapsed time this run never reached it and returned 20000 throughout.
        {"t_span": (5.0, 7.0), "n_points": 3},
        {"sample_times": [5.0, 5.5, 6.0, 6.5, 7.0, 8.0]},
    ],
)
def test_a_rate_that_turns_on_later_fires_on_model_time(times):
    rows = [_rm(SWITCH, seed, **times) for seed in range(1, 6)]
    t = np.asarray(rows[0].time)
    q = _switch_survival(t)
    _assert_mean([r.observables["A0"] for r in rows], 20000 * q, 20000 * q * (1 - q))
    # The rows before t = 6 hold the initial state exactly: nothing can fire.
    assert all(np.all(np.asarray(r.observables["A0"])[t < 6.0] == 20000) for r in rows)


def test_the_late_start_of_issue_1022():
    # [20000, 20000, 12131] by ode; rm lands within SSA noise of the last.
    a = np.asarray(_rm(SWITCH, 3, t_span=(5.0, 7.0), n_points=3).observables["A0"])
    assert a[0] == a[1] == 20000
    assert abs(a[2] - 20000 * math.exp(-0.5)) < 4.5 * 70


# ── A rate that reads time() throughout, from t_start > 0 ───────────────────


def _ramp_survival(t: np.ndarray, t0: float) -> np.ndarray:
    return np.exp(-0.02 * (t**2 - t0**2) / 2)


def test_a_ramp_from_a_late_start_matches_ssa_and_the_closed_form():
    # Elapsed time would read 0.02*(t - 5) instead of 0.02*t: at t = 10 the
    # survival is exp(-0.25) = 0.78 against exp(-0.75) = 0.47 on model time.
    T = [5.0, 7.5, 10.0, 15.0]
    rm = np.array(
        [_rm(f"{RAMP}.xml", seed, sample_times=T).observables["A0"] for seed in range(1, 41)]
    )
    ssa = []
    for seed in range(1, 41):
        m = bngsim.Model.from_net(f"{RAMP}.net")
        ssa.append(
            bngsim.Simulator(m, method="ssa").run(sample_times=T, seed=seed).observables["A0"]
        )
    ssa = np.asarray(ssa, dtype=float)
    q = _ramp_survival(np.asarray(T), 5.0)
    _assert_mean(rm, 1000 * q, 1000 * q * (1 - q))
    _assert_mean(ssa, 1000 * q, 1000 * q * (1 - q))
    se = np.sqrt(rm[:, 1:].var(axis=0, ddof=1) / 40 + ssa[:, 1:].var(axis=0, ddof=1) / 40)
    assert np.all(np.abs(rm[:, 1:].mean(axis=0) - ssa[:, 1:].mean(axis=0)) < 4.5 * se)


# ── #892: a table function indexed by time ──────────────────────────────────


def _tfun_variant(tmp_path: Path, n0: int, n1: int, dephos: float) -> Path:
    xml = TFUN.read_text(encoding="utf-8")
    xml = xml.replace('concentration="100" name="X(p~0)"', f'concentration="{n0}" name="X(p~0)"')
    xml = xml.replace('concentration="0" name="X(p~1)"', f'concentration="{n1}" name="X(p~1)"')
    xml = xml.replace(
        'id="dephos" type="Constant" value="0.1"', f'id="dephos" type="Constant" value="{dephos}"'
    )
    out = tmp_path / f"tfun_{n0}_{n1}_{dephos}.xml"
    out.write_text(xml, encoding="utf-8")
    return out


def _tfun_phos(start: int, dephos: float, t: float) -> float:
    """P(X(p~1) at t) for one molecule: X(p~0) -> X(p~1) at dephos, and back at
    the step table 0 on [0, 1), 10 on [1, 2), 20 after, each piece solved."""
    p = float(start)
    for lo, hi, back in ((0.0, 1.0, 0.0), (1.0, 2.0, 10.0), (2.0, math.inf, 20.0)):
        if t <= lo:
            break
        rate = dephos + back
        eq = dephos / rate
        p = eq + (p - eq) * math.exp(-rate * (min(t, hi) - lo))
    return p


@pytest.mark.parametrize(
    "n0, n1, dephos",
    [
        (0, 100, 0.1),  # nothing can fire until the table turns on at t = 1
        (1, 99, 0.001),  # next to nothing can fire until then
        (1, 99, 1.0),
        (100, 0, 0.1),  # the fixture as written
    ],
)
def test_a_time_indexed_table_function_matches_the_ode(tmp_path, n0, n1, dephos):
    xml = _tfun_variant(tmp_path, n0, n1, dephos)
    T = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    runs = np.array(
        [_rm(xml, seed, sample_times=T).observables["X_phos"] for seed in range(1, 201)]
    )
    q1 = np.array([_tfun_phos(1, dephos, t) for t in T])
    q0 = np.array([_tfun_phos(0, dephos, t) for t in T])
    mean = n1 * q1 + n0 * q0
    var = n1 * q1 * (1 - q1) + n0 * q0 * (1 - q0)
    _assert_mean(runs, mean, var)


def test_a_time_indexed_table_function_from_a_late_start(tmp_path):
    # From t = 1.5 the table already reads 10, and at 2 it steps to 20: the
    # elapsed clock would read 0 until t = 2.5.
    xml = _tfun_variant(tmp_path, 0, 100, 0.1)
    T = [1.5, 2.0, 2.5, 3.0]
    runs = np.array(
        [_rm(xml, seed, sample_times=T).observables["X_phos"] for seed in range(1, 201)]
    )
    # Seeded all phosphorylated at t = 1.5: the pieces from there on.
    q = []
    for t in T:
        p = 1.0
        for lo, hi, back in ((1.5, 2.0, 10.0), (2.0, math.inf, 20.0)):
            if t <= lo:
                break
            rate = 0.1 + back
            p = 0.1 / rate + (p - 0.1 / rate) * math.exp(-rate * (min(t, hi) - lo))
        q.append(p)
    q = np.asarray(q)
    _assert_mean(runs, 100 * q, 100 * q * (1 - q))


# ── #1063: a time-indexed table far from t = 0 ───────────────────────────────


@pytest.mark.parametrize("t0", [1e12, 1e13])
@pytest.mark.parametrize("path", ["run", "session"])
def test_a_time_indexed_table_keeps_its_waiting_times_far_from_t0(tmp_path, t0, path):
    """From t0 >= 2 the step table reads 20, so 10^4 X(p~1) leave at 20 and come
    back at 0.001: a total propensity of 2e5, whose waiting times are below the
    spacing of the doubles near t0. A table indexed by time is not offset, so the
    run is on model time, and bngsim refused a span this short there until
    RuleMonkey 3.12.0 put its events and its clock panels on elapsed time
    (richardposner/RuleMonkey#110)."""
    n, dephos = 10_000, 0.001
    xml = _tfun_variant(tmp_path, 0, n, dephos)
    span = (t0 + 0.05) - t0  # the span the doubles near t0 hold
    T = [t0, t0 + span]
    runs = []
    for seed in range(1, 41):
        if path == "run":
            runs.append(_rm(xml, seed, sample_times=T).observables["X_phos"])
        else:
            with bngsim.RuleMonkeySession(str(xml)) as s:
                s.initialize(seed=seed, t_start=t0)
                runs.append(s.simulate(sample_times=T).observables["X_phos"])
    rate = dephos + 20.0
    q = dephos / rate + (1 - dephos / rate) * np.exp(-rate * np.array([0.0, span]))
    _assert_mean(runs, n * q, n * q * (1 - q))


# ── Sessions on model time ───────────────────────────────────────────────────


def test_a_session_started_late_is_the_one_shot_run():
    """initialize(seed) then simulate(5, 15): the seed state moves to t = 5, as
    Simulator.run(t_span=(5, 15)) puts it, and the two are the same run."""
    one_shot = _rm(f"{RAMP}.xml", 11, t_span=(5.0, 15.0), n_points=5)
    with bngsim.RuleMonkeySession(f"{RAMP}.xml") as s:
        s.initialize(seed=11)
        r = s.simulate(5.0, 15.0, n_points=5)
        assert s.current_time == 15.0
    np.testing.assert_array_equal(r.time, one_shot.time)
    np.testing.assert_array_equal(np.asarray(r.observables), np.asarray(one_shot.observables))


def test_a_session_given_its_start_is_the_one_shot_run():
    one_shot = _rm(f"{RAMP}.xml", 11, sample_times=[5.0, 6.0, 9.0, 15.0])
    with bngsim.RuleMonkeySession(f"{RAMP}.xml") as s:
        s.initialize(seed=11, t_start=5.0)
        assert s.current_time == 5.0
        assert s.evaluate("time()") == 5.0
        r = s.simulate(sample_times=[5.0, 6.0, 9.0, 15.0])
    np.testing.assert_array_equal(np.asarray(r.observables), np.asarray(one_shot.observables))


def test_a_session_in_segments_reads_the_clock_across_them():
    T1, T2 = [5.0, 7.5, 10.0], [11.0, 12.5, 15.0]
    first, second = [], []
    for seed in range(1, 41):
        with bngsim.RuleMonkeySession(f"{RAMP}.xml") as s:
            s.initialize(seed=seed, t_start=5.0)
            first.append(s.simulate(sample_times=T1).observables["A0"])
            s.step_to(11.0)
            second.append(s.simulate(sample_times=T2).observables["A0"])
    q1 = _ramp_survival(np.asarray(T1), 5.0)
    q2 = _ramp_survival(np.asarray(T2), 5.0)
    _assert_mean(first, 1000 * q1, 1000 * q1 * (1 - q1))
    _assert_mean(second, 1000 * q2, 1000 * q2 * (1 - q2))


@pytest.mark.parametrize("move", ["segment", "explicit start", "add_molecules", "step_to"])
def test_a_session_refuses_a_segment_off_the_clock_of_a_model_that_reads_it(move):
    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        if move == "segment":
            s.initialize(seed=1)
            s.simulate(0.0, 1.0, n_points=2)
        elif move == "explicit start":
            s.initialize(seed=1, t_start=0.0)
        elif move == "add_molecules":
            s.initialize(seed=1)
            s.add_molecules("A", 1)
        else:
            s.initialize(seed=1)
            s.step_to(1.0)
        with pytest.raises(bngsim.SimulationError, match=r"reads the clock in 'f'"):
            s.simulate(5.0, 7.0, n_points=3)
        with pytest.raises(bngsim.SimulationError, match=r"reads the clock in 'f'"):
            s.simulate(sample_times=[5.0, 7.0])
        # On the clock it runs.
        now = s.current_time
        assert s.simulate(now, now + 1.0, n_points=2).n_times == 2


def test_a_session_refuses_a_time_indexed_table_off_its_clock():
    with bngsim.RuleMonkeySession(str(TFUN)) as s:
        s.initialize(seed=1)
        s.simulate(0.0, 1.0, n_points=2)
        with pytest.raises(bngsim.SimulationError, match=r"reads the clock in 'tfun_rate'"):
            s.simulate(0.0, 1.0, n_points=2)


def test_a_session_of_a_model_that_does_not_read_the_clock_still_relabels():
    """Unchanged: a segment off the clock continues from it, labelled as asked."""
    with bngsim.RuleMonkeySession(str(NF / "first_order_switch.xml")) as s:
        s.initialize(seed=1)
        s.simulate(0.0, 1.0, n_points=2)
        r = s.simulate(0.0, 1.0, n_points=2)
        assert s.current_time == 2.0
    np.testing.assert_array_equal(r.time, [0.0, 1.0])


@pytest.mark.parametrize(
    "bad_segment",
    [
        {"t_start": 5.0, "t_end": 4.0, "n_points": 3},
        {"t_start": 5.0, "t_end": 7.0, "n_points": 0},
        {"t_start": math.inf, "t_end": math.inf, "n_points": 3},
    ],
)
def test_a_failed_segment_leaves_the_start_unset(bad_segment):
    """A segment that raises does not move the seed state: the next one still
    places it, and the session is the one-shot run."""
    one_shot = _rm(SWITCH, 7, t_span=(5.0, 7.0), n_points=3)
    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        s.initialize(seed=7)
        with pytest.raises(bngsim.SimulationError):
            s.simulate(**bad_segment)
        assert s.current_time == 0.0
        r = s.simulate(5.0, 7.0, n_points=3)
    np.testing.assert_array_equal(np.asarray(r.observables), np.asarray(one_shot.observables))


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_session_refuses_a_start_that_is_not_finite(bad):
    with bngsim.RuleMonkeySession(str(SWITCH)) as s, pytest.raises(ValueError, match="finite"):
        s.initialize(seed=1, t_start=bad)


def test_a_session_counts_its_events_since_initialize():
    """Every segment's n_steps is the session's running total, as an NFsim
    session's is: the first matches a one-shot run of the same seed and span."""
    xml = str(NF / "first_order_switch.xml")
    one_shot = _rm(xml, 5, t_span=(0.0, 1.0), n_points=2)
    with bngsim.RuleMonkeySession(xml) as s:
        s.initialize(seed=5)
        first = s.simulate(0.0, 1.0, n_points=2)
        second = s.simulate(1.0, 2.0, n_points=2)
    assert first.solver_stats["n_steps"] == one_shot.solver_stats["n_steps"] > 0
    a0 = np.asarray(second.observables["A0"])
    assert second.solver_stats["n_steps"] == first.solver_stats["n_steps"] + (a0[0] - a0[-1])


@pytest.mark.parametrize("xml", [f"{RAMP}.xml", str(NF / "first_order_switch.xml")])
def test_a_span_that_is_not_finite_is_refused(xml):
    # On a model that reads the clock it used to integrate for ever, past any
    # timeout; on one that does not, it returned nan and inf for times.
    # Simulator.run refuses the span itself now, the same ValueError on every
    # method (issue #1504); the session below takes bare times, which the C++
    # refuses.
    for times in (
        {"t_span": (0.0, math.inf), "n_points": 3},
        {"t_span": (math.nan, 1.0), "n_points": 3},
    ):
        with pytest.raises(ValueError, match="finite"):
            _rm(xml, 1, **times)
    with bngsim.RuleMonkeySession(xml) as s:
        s.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="finite"):
            s.simulate(0.0, math.inf, n_points=3)
        with pytest.raises(bngsim.SimulationError, match="finite"):
            s.step_to(math.inf)
        assert s.simulate(0.0, 1.0, n_points=2).n_times == 2


def test_a_short_span_far_from_t0_runs():
    """At t = 1e13 the doubles are 0.002 apart, and a span of 2 has integration
    panels of 0.002: on model time RuleMonkey 3.11.0 fired nothing where a
    constant rate written through time() should have left 135 of 1000.

    A model that reads the clock through its functions integrates on elapsed
    time since issue #1063, one-shot and in a session, and reads the clock as
    time() + t_start: 0.02 * 1e13 per molecule clears all 1000 in the first
    instant, as the closed form exp(-0.02 * (t^2 - t0^2) / 2) does.

    A model with a table indexed by time runs on model time, where bngsim
    refused such a span until RuleMonkey 3.12.0 placed its events and panels on
    elapsed time (richardposner/RuleMonkey#110). It now runs one-shot, as a
    session segment and as a step; its values are checked against the closed
    form in test_a_time_indexed_table_keeps_its_waiting_times_far_from_t0."""
    span = (1e13, 1e13 + 2.0)
    a = np.asarray(_rm(f"{RAMP}.xml", 1, t_span=span, n_points=3).observables["A0"])
    np.testing.assert_array_equal(a, [1000, 0, 0])
    with bngsim.RuleMonkeySession(f"{RAMP}.xml") as s:
        s.initialize(seed=1, t_start=span[0])
        a = np.asarray(s.simulate(*span, n_points=3).observables["A0"])
    np.testing.assert_array_equal(a, [1000, 0, 0])

    assert _rm(TFUN, 1, t_span=span, n_points=3).n_times == 3
    with bngsim.RuleMonkeySession(str(TFUN)) as s:
        s.initialize(seed=1, t_start=span[0])
        assert s.simulate(*span, n_points=3).n_times == 3
    with bngsim.RuleMonkeySession(str(TFUN)) as s:
        s.initialize(seed=1, t_start=span[0])
        s.step_to(span[1])
        assert s.current_time == span[1]
    # A model that reads no clock runs on elapsed time there (test_network_free_t_start).
    assert _rm(str(NF / "first_order_switch.xml"), 1, t_span=span, n_points=3).n_times == 3


def test_a_session_steps_finely_on_a_clock_far_from_zero():
    """1000 segments of 0.01 from t = 86400, a day in seconds: panels of 1e-5
    against doubles 1.5e-11 apart, which RuleMonkey integrates over correctly
    and the resolution check lets through. f() is 0.5 from t = 6 on."""
    a0 = []
    for seed in (1, 2):
        with bngsim.RuleMonkeySession(str(SWITCH)) as s:
            s.initialize(seed=seed, t_start=86400.0)
            t = 86400.0
            for _ in range(1000):
                r = s.simulate(t, t + 0.01, n_points=2)
                t += 0.01
            a0.append(r.observables["A0"][-1])
    q = math.exp(-0.5 * 10.0)
    _assert_mean(np.asarray(a0)[:, None], np.array([20000 * q]), np.array([20000 * q * (1 - q)]))


def test_a_session_of_a_clock_free_model_starts_at_zero_whatever_its_start():
    """A model that reads no clock cannot tell model time from elapsed time, so
    its session runs from 0, as it always has, and keeps event times small: on
    a clock at 1e12 every waiting time of 1e-4 would round away. A segment from
    t_start is the one-shot run, labelled with its own times."""
    xml = str(NF / "first_order_switch.xml")
    one_shot = _rm(xml, 3, t_span=(0.0, 2.0), n_points=3)
    with bngsim.RuleMonkeySession(xml) as s:
        s.initialize(seed=3, t_start=1e12)
        assert s.current_time == 0.0
        r = s.simulate(1e12, 1e12 + 2.0, n_points=3)
    np.testing.assert_array_equal(r.time, [1e12, 1e12 + 1.0, 1e12 + 2.0])
    np.testing.assert_array_equal(np.asarray(r.observables), np.asarray(one_shot.observables))


# ── #1063: a model that reads the clock, from a large t_start ───────────────


def test_a_late_start_keeps_its_waiting_times():
    """RuleMonkey places each event at t + tau. From t = 1e12, where the doubles
    are 1.2e-4 apart, the waiting times of 20000 molecules at 0.5 (1e-4 each)
    rounded away, and the survival at 1e12 + 2 came back 10.8 standard errors
    under exp(-1) over 10 seeds (z = -291 at 1e5 molecules in the issue). A model
    that reads the clock runs on elapsed time now and reads the clock as
    time() + t_start. The span is long enough for model time's own refusal."""
    T = [1e12, 1e12 + 2.0, 1e12 + 1.3e5]
    rows = [_rm(SWITCH, seed, sample_times=T) for seed in range(1, 11)]
    np.testing.assert_array_equal(rows[0].time, T)
    q = np.exp(-0.5 * (np.asarray(T) - T[0]))
    _assert_mean([r.observables["A0"] for r in rows], 20000 * q, 20000 * q * (1 - q))


def test_an_output_function_reports_on_model_time():
    """The function the ramp's rate reads, f() = 0.02 * time(), at t = 5, 6, 7:
    the rewritten clock is the model's, not the run's elapsed time."""
    r = _rm(f"{RAMP}.xml", 1, t_span=(5.0, 7.0), n_points=3)
    np.testing.assert_allclose(r.expressions["f"], 0.02 * np.array([5.0, 6.0, 7.0]), rtol=1e-12)


def _variant(tmp_path: Path, edits: dict[str, str], *extra: str) -> str:
    """SWITCH with each of *edits* made and the *extra* functions added."""
    xml = SWITCH.read_text(encoding="utf-8")
    for old, new in edits.items():
        assert old in xml
        xml = xml.replace(old, new)
    xml = xml.replace("</ListOfFunctions>", "".join(extra) + "</ListOfFunctions>")
    path = tmp_path / "v.xml"
    path.write_text(xml, encoding="utf-8")
    return str(path)


def _output(fid: str, body: str) -> str:
    return f'<Function id="{fid}"><Expression> {body} </Expression></Function>\n    '


@pytest.mark.parametrize("spelling", ["time", "time( )", "t", "t()"])
@pytest.mark.parametrize("beside", [False, True], ids=["alone", "beside_time_call"])
def test_every_spelling_of_the_clock_reads_model_time(tmp_path, spelling, beside):
    """RuleMonkey reads the clock as time(), a bare or spaced time, and its alias
    t or t() (engine.cpp). Each is offset: an output h() of the spelling reports
    5, 6, 7 from t_start = 5, and f() = if(<spelling> >= 6, k, 0) switches on at
    6. Before, t was missed: alone, the model ran as if it read no clock, and
    beside a time() it read elapsed time once the run moved there (issue #1063);
    either way h was 0, 1, 2 and nothing fired by 7."""
    extra = [_output("h", spelling)] + ([_output("g", "time()")] if beside else [])
    xml = _variant(tmp_path, {"if(time()>=6,k,0)": f"if({spelling}>=6,k,0)"}, *extra)
    rows = [_rm(xml, seed, t_span=(5.0, 7.0), n_points=3) for seed in range(1, 6)]
    np.testing.assert_array_equal(rows[0].expressions["h"], [5.0, 6.0, 7.0])
    q = _switch_survival(np.array([5.0, 6.0, 7.0]))
    _assert_mean([r.observables["A0"] for r in rows], 20000 * q, 20000 * q * (1 - q))


@pytest.mark.parametrize("spelling", ["t", "t()"])
def test_a_session_reads_the_alias_t_on_model_time(tmp_path, spelling):
    """A session decides its clock by the same reading: a model that reads the
    clock only through t starts at t_start, so h() = t is the model time."""
    xml = _variant(tmp_path, {"if(time()>=6,k,0)": f"if({spelling}>=6,k,0)"}, _output("h", "t"))
    with bngsim.RuleMonkeySession(xml) as s:
        s.initialize(seed=1, t_start=5.0)
        r = s.simulate(5.0, 7.0, n_points=3)
    np.testing.assert_array_equal(r.expressions["h"], [5.0, 6.0, 7.0])


@pytest.mark.parametrize("symbol", ["time", "t"])
@pytest.mark.parametrize("beside", [False, True], ids=["alone", "beside_time_call"])
def test_a_symbol_named_like_the_clock_is_read_as_the_symbol(tmp_path, symbol, beside):
    """BNG2.pl accepts an observable named time. RuleMonkey reads a bare time,
    or t, that names a symbol as the symbol, and only time() as the clock. Here
    the observable counts A(s~0) and f() = k * <symbol> / 20000: f reports
    k * count / 20000 on every row, and the count follows the closed form
    20000 / (1 + k t) of dN/dt = -k N^2 / 20000. The rewrite once turned the
    observable into the clock, and f read 0.5 * 0 / 20000 at t = 0 while the
    count stayed at 20000."""
    extra = [_output("g", "time()")] if beside else []
    edits = {'name="A0"': f'name="{symbol}"', "if(time()>=6,k,0)": f"k*{symbol}/20000"}
    xml = _variant(tmp_path, edits, *extra)
    rows = [_rm(xml, seed, t_span=(0.0, 2.0), n_points=3) for seed in range(1, 6)]
    for r in rows:
        np.testing.assert_allclose(r.expressions["f"], 0.5 * r.observables[symbol] / 20000)
    if beside:
        np.testing.assert_array_equal(rows[0].expressions["g"], [0.0, 1.0, 2.0])
    mean = np.mean([r.observables[symbol][-1] for r in rows])
    assert abs(mean - 20000 / (1 + 0.5 * 2.0)) < 300, mean


def test_the_rewrite_reads_the_clock_where_rulemonkey_does():
    """The text of the copy: every clock read is offset, a bare time that names
    an observable is not, and a self-closing function is left alone."""
    from bngsim._rulemonkey_clock import offset_clock_text

    t0 = f"(time()+{OFFSET})"
    xml = SWITCH.read_text(encoding="utf-8")
    out = offset_clock_text(xml, SWITCH.parent)
    assert f"if({t0}>=6,k,0)" in out
    assert f'<Parameter id="{OFFSET}" type="Constant" value="0" expr="0"/>' in out

    named = xml.replace('name="A0"', 'name="time"').replace(
        "if(time()>=6,k,0)", "if(time()>=6,k,0)*time + time ( )"
    )
    named = named.replace("<ListOfFunctions>", '<ListOfFunctions>\n      <Function id="z"/>')
    out = offset_clock_text(named, SWITCH.parent)
    assert f"if({t0}>=6,k,0)*time + time ( )" in out
    assert '<Function id="z"/>' in out


@pytest.mark.parametrize(
    "old, new",
    [('id="k"', f'id="{OFFSET}"'), ('name="A0"', f'name="{OFFSET}"')],
    ids=["parameter", "observable"],
)
def test_a_model_that_names_the_offset_is_refused(old, new):
    """An observable is a RuleMonkey symbol by its name, and would take the
    offset's place in every rewritten function."""
    from bngsim._rulemonkey_clock import offset_clock_text

    xml = SWITCH.read_text(encoding="utf-8").replace(old, new)
    with pytest.raises(bngsim.ModelError, match=OFFSET):
        offset_clock_text(xml, SWITCH.parent)


@pytest.mark.parametrize("quote", ['"', "'"])
def test_a_table_indexed_by_the_clock_is_not_offset(tmp_path, quote):
    """Its counter would have to be a function, and RuleMonkey 3.11.0 recurses
    without end evaluating such a table. The model runs on model time, as the
    late-start test of #892 above checks, however its attributes are quoted: in
    single quotes the table was once missed, and the run went to elapsed time,
    where the table reads the elapsed clock."""
    text = TFUN.read_text(encoding="utf-8")
    tag = re.search(r'<Function [^>]*type="TFUN"[^>]*>', text).group(0)
    xml = tmp_path / "v.xml"
    xml.write_text(text.replace(tag, tag.replace('"', quote)), encoding="utf-8")
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    sim = bngsim.Simulator(m, method="rm", xml_path=str(xml))
    assert not sim._rm_clock_offset
    r = sim.run(t_span=(1.0, 1.5), n_points=3, seed=3)
    np.testing.assert_array_equal(
        np.asarray(r.observables),
        np.asarray(_rm(TFUN, 3, t_span=(1.0, 1.5), n_points=3).observables),
    )


def test_a_table_file_is_found_from_the_copy(tmp_path):
    """The copy is written elsewhere, so a table's data file named relative to
    the original XML is made absolute, and escaped: here the directory's name
    has characters XML must escape. g() = 1 + A0 / 10000 by its table."""
    here = tmp_path / "a&b'c"
    here.mkdir()
    (here / "x.tfun").write_text("0 1\n30000 4\n")
    table = (
        '<Function id="g" type="TFUN" ctrName="A0" file="x.tfun" method="linear">'
        "<Expression> __TFUN_VAL__ </Expression></Function>\n    "
    )
    xml = _variant(here, {}, table)
    r = _rm(xml, 1, t_span=(5.0, 7.0), n_points=3)
    np.testing.assert_allclose(r.expressions["g"], 1 + r.observables["A0"] / 10000)


# ── #1063: a session of a model that reads the clock, from a large start ────


def test_a_late_session_keeps_its_waiting_times():
    """The session half of test_a_late_start_keeps_its_waiting_times: on model
    time a session from t = 1e12 came back 10.8 standard errors under exp(-1) at
    1e12 + 2 over these 10 seeds. It runs on elapsed time now, and is the
    one-shot run seed for seed."""
    T = [1e12, 1e12 + 2.0, 1e12 + 1.3e5]
    rows = []
    for seed in range(1, 11):
        with bngsim.RuleMonkeySession(str(SWITCH)) as s:
            s.initialize(seed=seed, t_start=T[0])
            r = s.simulate(sample_times=T)
        np.testing.assert_array_equal(r.time, T)
        one_shot = _rm(SWITCH, seed, sample_times=T)
        np.testing.assert_array_equal(r.observables["A0"], one_shot.observables["A0"])
        rows.append(r.observables["A0"])
    q = np.exp(-0.5 * (np.asarray(T) - T[0]))
    _assert_mean(rows, 20000 * q, 20000 * q * (1 - q))


def test_a_late_session_reads_and_reports_model_time():
    """Every time a session takes or gives is the model's, though its events are
    placed on elapsed time: current_time, the clock and f() = 0.02 * time() read
    through evaluate, a segment's rows and its functions, step_to, and a second
    initialize. An override of t is the override."""
    t0 = 1e12
    with bngsim.RuleMonkeySession(f"{RAMP}.xml") as s:
        s.initialize(seed=4, t_start=t0)
        assert s.current_time == s.evaluate("time()") == s.evaluate("t") == t0
        assert s.evaluate("f") == pytest.approx(0.02 * t0, rel=1e-12)
        assert s.evaluate("t", {"t": 3.0}) == 3.0
        r = s.simulate(t0, t0 + 2.0, n_points=3)
        np.testing.assert_array_equal(r.time, [t0, t0 + 1.0, t0 + 2.0])
        np.testing.assert_allclose(r.expressions["f"], 0.02 * (t0 + np.arange(3.0)), rtol=1e-12)
        assert s.current_time == t0 + 2.0
        s.step_to(t0 + 3.0)
        assert s.current_time == s.evaluate("time()") == t0 + 3.0
        s.initialize(seed=4, t_start=5.0)
        assert s.current_time == s.evaluate("time()") == 5.0


def test_a_late_session_saves_and_loads_on_model_time(tmp_path):
    """A snapshot holds the model's time, as one saved on model time does, and a
    session that loads it restarts its elapsed clock there: it continues as the
    session that saved it does, seed for seed."""
    t0, path = 1e12, tmp_path / "state"
    with bngsim.RuleMonkeySession(str(SWITCH)) as a:
        a.initialize(seed=9, t_start=t0)
        a.simulate(t0, t0 + 1.0, n_points=2)
        a.save_state(path)
        kept = a.simulate(t0 + 1.0, t0 + 2.0, n_points=3)
    assert float(path.read_text().split("\n")[2]) == t0 + 1.0
    with bngsim.RuleMonkeySession(str(SWITCH)) as b:
        b.load_state(path)
        assert b.current_time == b.evaluate("time()") == t0 + 1.0
        loaded = b.simulate(t0 + 1.0, t0 + 2.0, n_points=3)
    np.testing.assert_array_equal(loaded.observables["A0"], kept.observables["A0"])
