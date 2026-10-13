"""GH #1022: method='rm' runs a rate that reads the clock, on absolute model time.

Until RuleMonkey 3.11.0 a total propensity of 0 was final, so a rate that is 0
at the start and turns on later never fired, and bngsim refused every model
whose rate laws read ``time()`` or a table function indexed by time (issues
#862 and #892). 3.11.0 integrates such a rate between events
(richardposner/RuleMonkey#86) and puts the seed state at ``t_start`` on the
model's clock (richardposner/RuleMonkey#98), so the refusals are gone and a run
over ``(5, 7)`` reads ``time()`` from 5 to 7, as ode and ssa do.

Every expected value here is a closed form. A count is compared within five
standard deviations of its own distribution, never to one machine's digits, and
a row the model fixes exactly (nothing can fire yet) is compared exactly.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import SimulationError


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_rulemonkey(),
    reason="RuleMonkey not compiled in",
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
# A(s~0) -> A(s~1) at f() = if(time()>=6, k, 0), k = 0.5, from 20000 A(s~0).
SWITCH = DATA / "nfsim" / "time_switch.xml"
# The same rule at the constant k: a model that does not read the clock.
PLAIN = DATA / "nfsim" / "first_order_switch.xml"
# X(p~0) -> X(p~1) at dephos, X(p~1) -> X(p~0) at k_base*tfun(time), the table
# a step: 0 until t = 1, 10 until t = 2, then 20. 100 X in all.
TFUN = DATA / "nfsim" / "tfun_new_format" / "valid_inline_step_time.xml"
N = 20000.0
K = 0.5


def _model():
    return bngsim.Model.from_net(str(DATA / "simple_decay.net"))


def _run(xml, seed=3, **times):
    r = bngsim.Simulator(_model(), method="rm", xml_path=str(xml)).run(seed=seed, **times)
    return r, np.asarray(r.time), np.asarray(r.observables["A0"])


def _within(count, mean, sd, n_sd=5.0):
    """*count* within *n_sd* standard deviations of *mean*, row by row. A row
    with no spread has to match exactly."""
    np.testing.assert_array_less(np.abs(count - mean), n_sd * sd + 1e-9)


def _binomial(p):
    return N * p, np.sqrt(N * p * (1.0 - p))


# ── #862: a rate law that reads time() ────────────────────────────────────


def test_a_rate_that_turns_on_later_fires_when_it_does():
    """A0 = 20000 until t = 6, then 20000*exp(-k (t - 6)). On 3.10.1 the run
    was flat at 20000, and bngsim refused it."""
    r, t, a = _run(SWITCH, t_span=(0.0, 10.0), n_points=11)
    np.testing.assert_array_equal(a[t <= 6.0], N)
    mean, sd = _binomial(np.exp(-K * np.clip(t - 6.0, 0.0, None)))
    _within(a, mean, sd)
    assert a[-1] < 0.2 * N  # 2707 expected: the rate did turn on
    # The function column reads the clock at each row, not the last event's time.
    np.testing.assert_array_equal(np.asarray(r.expressions["f"]), np.where(t >= 6.0, K, 0.0))


@pytest.mark.parametrize(
    "times", [{"t_span": (5.0, 7.0), "n_points": 3}, {"sample_times": [5.0, 6.0, 7.0]}]
)
def test_a_late_start_reads_absolute_time(times):
    """Over (5, 7) the clock runs from 5 to 7, so the rate turns on at the
    second row: [20000, 20000, 12131], as ode gives. On elapsed time (the clock
    from 0 to 2) nothing would fire and the last row would be 20000 too."""
    r, t, a = _run(SWITCH, **times)
    np.testing.assert_array_equal(t, [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(a[:2], N)
    mean, sd = _binomial(np.exp(-K * 1.0))
    _within(a[2:], mean, sd)
    np.testing.assert_array_equal(np.asarray(r.expressions["f"]), [0.0, K, K])


def _ramp(tmp_path):
    """A(s~0) -> A(s~1) at f() = k*time(), k = 0.05, as a BNG XML for rm and as
    a .net for ssa: A0(t) = 20000*exp(-k (t^2 - t0^2)/2) from a start at t0."""
    xml = tmp_path / "ramp.xml"
    text = SWITCH.read_text(encoding="utf-8")
    assert " if(time()>=6,k,0) " in text and 'value="0.5" expr="0.5"' in text
    text = text.replace(" if(time()>=6,k,0) ", " k*time() ")
    xml.write_text(text.replace('value="0.5" expr="0.5"', 'value="0.05" expr="0.05"'), "utf-8")
    net = tmp_path / "ramp.net"
    net.write_text(
        "begin parameters\n    1 k  0.05\nend parameters\n"
        "begin functions\n    1 f() k*time()\nend functions\n"
        "begin species\n    1 A(s~0) 20000\n    2 A(s~1) 0\nend species\n"
        "begin reactions\n    1 1 2 f #_R1\nend reactions\n"
        "begin groups\n    1 A0  1\nend groups\n",
        encoding="utf-8",
    )
    return xml, net


def test_a_ramp_from_a_late_start_agrees_with_ssa(tmp_path):
    """k*time() over (5, 7): rm and ssa both follow the closed form on absolute
    time, [20000, 15191, 10976]. On elapsed time the last row would be 18097,
    more than a hundred standard deviations away."""
    xml, net = _ramp(tmp_path)
    t = np.array([5.0, 6.0, 7.0])
    mean, sd = _binomial(np.exp(-0.05 * (t**2 - 25.0) / 2.0))

    _, t_rm, rm = _run(xml, t_span=(5.0, 7.0), n_points=3)
    np.testing.assert_array_equal(t_rm, t)
    _within(rm, mean, sd)

    r = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ssa").run(
        t_span=(5.0, 7.0), n_points=3, seed=3
    )
    ssa = np.asarray(r.observables["A0"])
    _within(ssa, mean, sd)
    # And with each other: the difference of two such counts.
    _within(rm - ssa, 0.0, np.sqrt(2.0) * sd)


# ── #892: a table function indexed by time ────────────────────────────────


def _tfun_variant(tmp_path, x0, x1, dephos):
    """The #892 fixture started from *x0* X(p~0) and *x1* X(p~1)."""
    text = TFUN.read_text(encoding="utf-8")
    for old, new in (
        ('concentration="100" name="X(p~0)"', f'concentration="{x0}" name="X(p~0)"'),
        ('concentration="0" name="X(p~1)"', f'concentration="{x1}" name="X(p~1)"'),
        (
            'id="dephos" type="Constant" value="0.1"',
            f'id="dephos" type="Constant" value="{dephos}"',
        ),
    ):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    xml = tmp_path / f"tfun_{x0}_{x1}_{dephos}.xml"
    xml.write_text(text, encoding="utf-8")
    return xml


def _tfun_closed_form(t, x0, x1, a):
    """Mean and standard deviation of X_phos at each *t*. Every molecule is its
    own two-state chain, up at *a* and down at 0, 10, 20 on [0,1), [1,2), [2,∞),
    so p(up) relaxes to a/(a + r) at rate a + r on each stretch and the count is
    a sum of independent Bernoulli draws."""

    def p_up(t_end, p):
        for lo, hi, r in ((0.0, 1.0, 0.0), (1.0, 2.0, 10.0), (2.0, np.inf, 20.0)):
            dt = min(t_end, hi) - lo
            if dt <= 0.0:
                break
            p = a / (a + r) + (p - a / (a + r)) * np.exp(-(a + r) * dt)
        return p

    from_down = np.array([p_up(x, 0.0) for x in t])
    from_up = np.array([p_up(x, 1.0) for x in t])
    mean = x0 * from_down + x1 * from_up
    var = x0 * from_down * (1.0 - from_down) + x1 * from_up * (1.0 - from_up)
    return mean, np.sqrt(var)


@pytest.mark.parametrize(
    ("x0", "x1", "dephos"),
    [
        # The fixture as written. Events are frequent, so 3.10.1 tracked this too.
        (100, 0, 0.1),
        # Nothing can fire until the table turns the rate on at t = 1. On 3.10.1
        # X_phos stayed at 100 on every row.
        (0, 100, 0.1),
        # One molecule can fire, at 0.001: X_phos stayed at 99 on every row.
        (1, 99, 0.001),
        # At dephos = 1 it stayed at 99 until t = 2.5.
        (1, 99, 1.0),
    ],
)
def test_a_time_indexed_table_function_follows_the_table(tmp_path, x0, x1, dephos):
    """Mean X_phos over 200 seeds against the closed form, at t = 0, 0.5, ..., 3."""
    xml = _tfun_variant(tmp_path, x0, x1, dephos)
    seeds = range(1, 201)
    sim = bngsim.Simulator(_model(), method="rm", xml_path=str(xml))
    runs = np.array(
        [
            np.asarray(sim.run(t_span=(0.0, 3.0), n_points=7, seed=s).observables["X_phos"])
            for s in seeds
        ]
    )
    t = np.linspace(0.0, 3.0, 7)
    mean, sd = _tfun_closed_form(t, x0, x1, dephos)
    np.testing.assert_array_equal(runs[:, 0], x1)
    _within(runs.mean(axis=0), mean, sd / np.sqrt(len(seeds)))
    if x1:
        # The drain after t = 1 is what 3.10.1 missed: by t = 1.5 the 99 or 100
        # are down to 9.5 at most (1.6, 0.7 and 9.5 in the three cases).
        assert runs.mean(axis=0)[3] < 0.15 * x1


# ── Sessions run on the model's clock ─────────────────────────────────────


def _a0(result):
    return np.asarray(result.observables["A0"])


def test_a_session_started_at_t_start_is_the_one_shot_run():
    """initialize(seed, t_start=5) then simulate(5, 7) spends the same draws as
    run(t_span=(5, 7)): the two are the same run."""
    _, _, one_shot = _run(SWITCH, seed=11, t_span=(5.0, 7.0), n_points=5)
    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        s.initialize(seed=11, t_start=5.0)
        assert s.current_time == 5.0
        r = s.simulate(5.0, 7.0, 5)
        assert s.current_time == 7.0
    np.testing.assert_array_equal(np.asarray(r.time), [5.0, 5.5, 6.0, 6.5, 7.0])
    np.testing.assert_array_equal(_a0(r), one_shot)
    np.testing.assert_array_equal(_a0(r)[:3], N)
    assert _a0(r)[-1] < 0.7 * N  # 12131 expected


def test_a_session_continued_on_its_clock_crosses_the_switch():
    """0 -> 5 -> 7 in two segments: nothing fires in the first, and the second
    turns on at t = 6."""
    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        s.initialize(seed=5)
        first = s.simulate(0.0, 5.0, 2)
        second = s.simulate(sample_times=[5.0, 6.0, 7.0])
    np.testing.assert_array_equal(_a0(first), N)
    np.testing.assert_array_equal(_a0(second)[:2], N)
    mean, sd = _binomial(np.exp(-K * 1.0))
    _within(_a0(second)[2:], mean, sd)


@pytest.mark.parametrize("times", [{"args": (5.0, 7.0, 3)}, {"sample_times": [5.0, 6.0, 7.0]}])
def test_a_segment_off_the_clock_is_refused_when_the_model_reads_it(times):
    """The session's clock reads 0. Run from there under labels 5..7, the rate
    would read time() from 0 to 2 and never turn on, in rows that say 5 to 7."""
    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        s.initialize(seed=5)
        with pytest.raises(SimulationError, match=r"segment starts at t=5\.0 .* clock reads 0\.0"):
            s.simulate(*times.get("args", ()), sample_times=times.get("sample_times"))
        # Refused before anything ran: the session is where it was.
        assert s.current_time == 0.0
        np.testing.assert_array_equal(_a0(s.simulate(0.0, 1.0, 2)), N)


def test_the_refusal_names_the_functions_and_the_way_out():
    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        s.initialize(seed=5)
        s.simulate(0.0, 7.0, 2)
        with pytest.raises(SimulationError) as err:
            s.simulate(0.0, 7.0, 2)  # a second segment that restarts its labels at 0
    message = str(err.value)
    assert "functions: 'f'" in message
    assert "initialize(seed, t_start=...)" in message
    assert "#1022" in message


def test_a_table_function_indexed_by_time_counts_as_reading_the_clock():
    with bngsim.RuleMonkeySession(str(TFUN)) as s:
        s.initialize(seed=5)
        with pytest.raises(SimulationError, match="functions: 'tfun_rate'"):
            s.simulate(1.0, 2.0, 2)


@pytest.mark.parametrize("times", [{"args": (5.0, 7.0, 3)}, {"sample_times": [5.0, 6.0, 7.0]}])
def test_a_segment_off_the_clock_still_runs_as_elapsed_time_without_a_clock(times):
    """Control. A model that does not read the clock sees only elapsed time, so
    simulate(5, 7) on a session at 0 is the run from 0 to 2 under labels 5..7,
    as before, and a segment that restarts its labels continues the state."""
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=3)
        shifted = s.simulate(*times.get("args", ()), sample_times=times.get("sample_times"))
        assert s.current_time == 2.0
        again = s.simulate(sample_times=[0.0, 1.0])
        assert s.current_time == 3.0
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=3)
        plain = s.simulate(0.0, 2.0, 3)
    np.testing.assert_array_equal(np.asarray(shifted.time), [5.0, 6.0, 7.0])
    np.testing.assert_array_equal(_a0(shifted), _a0(plain))
    np.testing.assert_array_equal(np.asarray(again.time), [0.0, 1.0])
    assert _a0(again)[0] == _a0(shifted)[-1]
    mean, sd = _binomial(np.exp(-K * 3.0))
    _within(_a0(again)[1:], mean, sd)


def test_a_model_this_cannot_read_is_treated_as_reading_the_clock(monkeypatch):
    """If the XML cannot be scanned here, an off-clock segment is refused rather
    than run on the guess that the model has no clock."""
    monkeypatch.setattr("bngsim._network_free_checks.clock_reading_functions", lambda _path: None)
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=3)
        with pytest.raises(SimulationError, match="could not be read from its XML"):
            s.simulate(5.0, 7.0, 3)
        assert _a0(s.simulate(0.0, 1.0, 2))[0] == N


def test_the_core_refuses_an_off_clock_segment_unless_told_it_is_safe():
    """The C++ adapter does not take the guess either: elapsed time is opt-in."""
    from bngsim._bngsim_core import RuleMonkeySimulator

    core = RuleMonkeySimulator(str(PLAIN))
    core.initialize(3)
    assert core.starts_on_clock(0.0) and not core.starts_on_clock(5.0)
    with pytest.raises(RuntimeError, match=r"segment starts at t=5 .* clock reads 0"):
        core.simulate(5.0, 7.0, 3)
    with pytest.raises(RuntimeError, match="segment starts at t=5"):
        core.simulate(0.0, 0.0, 0, 0.0, [5.0, 7.0])
    assert core.current_time() == 0.0
    shifted = core.simulate(5.0, 7.0, 3, 0.0, [], True)
    assert core.current_time() == 2.0
    assert list(shifted.time) == [5.0, 6.0, 7.0]
    core.destroy_session()


def test_a_start_a_rounding_error_off_the_clock_is_on_it():
    """A segment's start is compared with the clock as RuleMonkey's own
    simulate() compares it, to 1e-9 of the clock. A start that differs by
    rounding continues on the clock; one that differs by more is another time."""
    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        s.initialize(seed=5, t_start=5.0)
        r = s.simulate(5.0 + 1e-12, 7.0, 3)
        assert s.current_time == 7.0
        with pytest.raises(SimulationError, match=r"clock reads 7\.0"):
            s.simulate(7.0 + 1e-6, 8.0, 2)
        with pytest.raises(SimulationError, match=r"clock reads 7\.0"):
            s.simulate(7.0 - 1e-6, 8.0, 2)
    assert _a0(r)[0] == N
    mean, sd = _binomial(np.exp(-K * 1.0))
    _within(_a0(r)[2:], mean, sd)


def test_the_last_row_carries_t_end_itself():
    """RuleMonkey builds a uniform grid as t_start + i*dt, which over (0.3, 0.9)
    in three steps ends at 0.9000000000000001. The rows carry the instants
    asked for, on a one-shot run and on a session."""
    assert 0.3 + 3 * ((0.9 - 0.3) / 3) != 0.9
    _, t, _ = _run(PLAIN, t_span=(0.3, 0.9), n_points=4)
    assert t[0] == 0.3 and t[-1] == 0.9
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=3, t_start=0.3)
        r = s.simulate(0.3, 0.9, 4)
        assert s.current_time == 0.9
    assert r.time[0] == 0.3 and r.time[-1] == 0.9


def test_initialize_refuses_a_start_that_is_not_a_time():
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        with pytest.raises(SimulationError, match="t_start must be finite"):
            s.initialize(seed=1, t_start=float("nan"))
        assert not s.initialized


# ── richardposner/RuleMonkey#89: a session advanced in short calls ────────


@pytest.mark.parametrize("advance", ["simulate", "step_to"])
def test_a_session_advanced_in_short_calls_keeps_firing(advance):
    """A0(2) = 20000*exp(-1) = 7358 however the two time units are cut. 3.10.1
    rolled the generator back at the end of every call and spent the same draw
    again, so 2000 calls of 0.001 read 15636 here, and 200000 calls stayed at
    20000."""
    grid = np.linspace(0.0, 2.0, 2001)
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=7)
        for lo, hi in zip(grid[:-1], grid[1:], strict=True):
            if advance == "simulate":
                s.simulate(float(lo), float(hi), 2)
            else:
                s.step_to(float(hi))
        final = dict(zip(s.get_observable_names(), s.get_observable_values(), strict=True))["A0"]
    mean, sd = _binomial(np.exp(-K * 2.0))
    _within(np.array([final]), mean, sd)


def test_a_session_result_counts_the_events_of_the_whole_session():
    """Control. n_steps is the session's running total, as NfsimSession's is:
    the second segment reports the first segment's events too."""
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=7)
        first = s.simulate(0.0, 1.0, 2)
        second = s.simulate(1.0, 2.0, 2)
    # Each event turns one A(s~0) into A(s~1), so events = the drop in A0.
    assert first.solver_stats["n_steps"] == N - _a0(first)[-1]
    assert second.solver_stats["n_steps"] == N - _a0(second)[-1]
    assert second.solver_stats["n_steps"] > first.solver_stats["n_steps"] > 0
