"""GH #1022: method='rm' runs a rate that reads the clock, on absolute model time.

Until RuleMonkey 3.11.0 a total propensity of 0 was final, so a rate that is 0
at the start and turns on later never fired, and bngsim refused every model
whose rate laws read ``time()`` or a table function indexed by time (issues
#862 and #892). 3.11.0 integrates such a rate between events
(richardposner/RuleMonkey#86) and puts the seed state at ``t_start`` on the
model's clock (richardposner/RuleMonkey#98), so a run over ``(5, 7)`` reads
``time()`` from 5 to 7, as ode and ssa do.

The engine integrates in steps of up to a thousandth of a call and can step
over a short window in a rate, so bngsim cuts every engine call at the instants
the rates change at (test_rulemonkey_clock_profile.py is how it finds them). A
pulse is then seen however short it is and however long the run.

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


def test_the_switch_is_placed_where_it_is_written():
    """The mean over 200 seeds, to a standard error of 5: a rate 0.5% off, or a
    switch placed 0.01 late, would show at t = 7 and t = 8."""
    sim = bngsim.Simulator(_model(), method="rm", xml_path=str(SWITCH))
    seeds = range(1, 201)
    runs = np.array(
        [
            np.asarray(sim.run(t_span=(5.0, 8.0), n_points=4, seed=s).observables["A0"])
            for s in seeds
        ]
    )
    t = np.array([5.0, 6.0, 7.0, 8.0])
    mean, sd = _binomial(np.exp(-K * np.clip(t - 6.0, 0.0, None)))
    np.testing.assert_array_equal(runs[:, :2], N)
    _within(runs.mean(axis=0), mean, sd / np.sqrt(len(seeds)))


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


def test_the_core_takes_a_model_to_read_the_clock_until_told_otherwise():
    """The C++ adapter does not guess: elapsed time is for a model declared not
    to read the clock."""
    from bngsim._bngsim_core import RuleMonkeySimulator

    core = RuleMonkeySimulator(str(PLAIN))
    core.initialize(3)
    assert core.starts_on_clock(0.0) and not core.starts_on_clock(5.0)
    with pytest.raises(RuntimeError, match=r"segment starts at t=5 .* clock reads 0"):
        core.simulate(5.0, 7.0, 3)
    with pytest.raises(RuntimeError, match="segment starts at t=5"):
        core.simulate(0.0, 0.0, 2, 0.0, [5.0, 7.0])
    assert core.current_time() == 0.0
    core.destroy_session()

    core.set_clock_profile(False)
    core.initialize(3)
    shifted = core.simulate(5.0, 7.0, 3)
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
    again, so 2000 calls of 0.001 read about 15600 (the mean of 20 seeds; 17187
    on this one), and 200000 calls stayed at 20000."""
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


# ── A window shorter than the engine's steps ──────────────────────────────
#
# A(s~0) -> A(s~1) at a rate that is 1/w for a stretch w starting at t = 6 and
# 0 otherwise. The pulse has area 1 whatever w is, so A0 after it is 20000/e.
# RuleMonkey integrates a clock-reading rate in panels of up to a thousandth of
# a call and accepts a panel whose nodes all read 0, so without the cuts a pulse
# 0.1 wide fired nothing in a call of 1000 and A0 stayed at 20000, with no error.

PULSE_START = 6.0


def _escaped(expression: str) -> str:
    return expression.replace("&", "&amp;").replace("<", "&lt;")


def _pulse_expression(tmp_path, width):
    """The window written with numbers: if(time()>=6, if(time()<6+w, 1/w, 0), 0)."""
    text = SWITCH.read_text(encoding="utf-8")
    expression = f"if(time()>=6, if(time()<{PULSE_START + width!r}, {1.0 / width!r}, 0), 0)"
    xml = tmp_path / f"pulse_expression_{width}.xml"
    xml.write_text(text.replace(" if(time()>=6,k,0) ", f" {_escaped(expression)} "), "utf-8")
    return xml


def _pulse_parameters(tmp_path, width):
    """The window written with parameters, in the parentheses BNG2.pl writes."""
    text = SWITCH.read_text(encoding="utf-8")
    declared = (
        f'      <Parameter id="t_on" type="Constant" value="{PULSE_START!r}"/>\n'
        f'      <Parameter id="t_off" type="Constant" value="{PULSE_START + width!r}"/>\n'
        f'      <Parameter id="height" type="Constant" value="{1.0 / width!r}"/>\n'
    )
    text = text.replace("    </ListOfParameters>\n", declared + "    </ListOfParameters>\n")
    expression = "if((time()>=t_on),if((time()<t_off),height,0),0)"
    xml = tmp_path / f"pulse_parameters_{width}.xml"
    xml.write_text(text.replace(" if(time()>=6,k,0) ", f" {_escaped(expression)} "), "utf-8")
    return xml


def _pulse_table(tmp_path, width):
    """The window as a step table indexed by time: 0, then 1/w from 6, then 0."""
    text = SWITCH.read_text(encoding="utf-8")
    function = text[text.index('      <Function id="f">') : text.index("    </ListOfFunctions>")]
    table = (
        f'      <Function id="f" type="TFUN" mode="inline" ctrName="time" '
        f'xData="0,{PULSE_START!r},{PULSE_START + width!r}" yData="0,{1.0 / width!r},0" '
        'method="step">\n        <Expression>__TFUN_VAL__</Expression>\n      </Function>\n'
    )
    xml = tmp_path / f"pulse_table_{width}.xml"
    xml.write_text(text.replace(function, table), encoding="utf-8")
    return xml


PULSES = [_pulse_expression, _pulse_parameters, _pulse_table]
AFTER_PULSE = _binomial(np.exp(-1.0))


@pytest.mark.parametrize("build", PULSES)
@pytest.mark.parametrize("width", [0.5, 0.02, 1e-5])
@pytest.mark.parametrize("t_end", [10.0, 1000.0, 1e5])
def test_a_pulse_is_seen_however_short_and_however_long_the_run(tmp_path, build, width, t_end):
    xml = build(tmp_path, width)
    r = bngsim.Simulator(_model(), method="rm", xml_path=str(xml)).run(
        t_span=(0.0, t_end), n_points=3, seed=3
    )
    a = np.asarray(r.observables["A0"])
    assert a[0] == N
    _within(a[-1:], *AFTER_PULSE)
    # One event per molecule that left A(s~0), across every stretch of the run.
    assert r.solver_stats["n_steps"] == N - a[-1]


@pytest.mark.parametrize("build", PULSES)
def test_rows_on_and_around_a_pulse(tmp_path, build):
    """Output times on the instants the rate changes at, and inside the pulse.
    Halfway through, half the hazard is spent: 20000*exp(-0.5) = 12131."""
    xml = build(tmp_path, 0.02)
    r = bngsim.Simulator(_model(), method="rm", xml_path=str(xml)).run(
        sample_times=[0.0, 5.0, 6.0, 6.01, 6.02, 500.0], seed=9
    )
    np.testing.assert_array_equal(np.asarray(r.time), [0.0, 5.0, 6.0, 6.01, 6.02, 500.0])
    a = np.asarray(r.observables["A0"])
    np.testing.assert_array_equal(a[:3], N)
    _within(a[3:4], *_binomial(np.exp(-0.5)))
    _within(a[4:], *AFTER_PULSE)
    assert a[4] == a[5]  # nothing fires after the pulse
    # The function column reads the branch that starts at each row's instant.
    np.testing.assert_allclose(np.asarray(r.expressions["f"]), [0, 0, 50, 50, 0, 0], rtol=1e-12)


@pytest.mark.parametrize("build", PULSES)
@pytest.mark.parametrize("cut", ["one call", "two calls", "step_to"])
def test_a_session_sees_the_pulse_however_it_is_cut(tmp_path, build, cut):
    """One simulate(0, 1000) read 20000; cut at 10 it read 7319. The answer no
    longer depends on where the calls end."""
    xml = build(tmp_path, 0.02)
    with bngsim.RuleMonkeySession(str(xml)) as s:
        s.initialize(seed=4)
        if cut == "one call":
            final = _a0(s.simulate(0.0, 1000.0, 3))[-1]
        elif cut == "two calls":
            s.simulate(0.0, 500.0, 2)
            final = _a0(s.simulate(sample_times=[500.0, 1000.0]))[-1]
        else:
            s.step_to(1000.0)
            final = dict(zip(s.get_observable_names(), s.get_observable_values(), strict=True))[
                "A0"
            ]
        assert s.current_time == 1000.0
    _within(np.array([final]), *AFTER_PULSE)


def test_a_parameter_threshold_follows_an_override(tmp_path):
    """The pulse moved to t = 20 by set_param is cut at 20: a threshold that
    names a parameter is read when the session starts, not when the XML is."""
    xml = _pulse_parameters(tmp_path, 0.02)
    with bngsim.RuleMonkeySession(str(xml)) as s:
        s.set_param("t_on", 20.0)
        s.set_param("t_off", 20.02)
        s.initialize(seed=4)
        r = s.simulate(sample_times=[0.0, 19.0, 20.01, 1000.0])
    a = _a0(r)
    np.testing.assert_array_equal(a[:2], N)
    _within(a[2:3], *_binomial(np.exp(-0.5)))
    _within(a[3:], *AFTER_PULSE)


def test_a_one_shot_run_across_a_switch_leaves_a_live_session_alone():
    """Cutting a run at a threshold takes a session of the run's own. The core
    refuses to take one over rather than end it."""
    from bngsim._bngsim_core import RuleMonkeySimulator, TimeSpec

    core = RuleMonkeySimulator(str(SWITCH))
    core.set_clock_profile(True, [6.0])
    ts = TimeSpec()
    ts.t_start, ts.t_end, ts.n_points = 0.0, 10.0, 3
    core.initialize(1)
    with pytest.raises(RuntimeError, match="one is already active"):
        core.run(ts, 1)
    assert core.has_session() and core.current_time() == 0.0
    core.destroy_session()
    assert list(core.run(ts, 1).time) == [0.0, 5.0, 10.0]
    assert not core.has_session()


def test_a_saved_session_is_cut_where_the_rates_change_when_it_is_loaded(tmp_path):
    """load_state starts a session without initialize(): it has to hand the
    engine the thresholds too."""
    xml = _pulse_parameters(tmp_path, 0.02)
    snapshot = tmp_path / "before_the_pulse.state"
    with bngsim.RuleMonkeySession(str(xml)) as s:
        s.initialize(seed=4)
        s.simulate(0.0, 5.0, 2)
        s.save_state(snapshot)
    with bngsim.RuleMonkeySession(str(xml)) as s:
        s.load_state(snapshot)
        assert s.current_time == 5.0
        final = _a0(s.simulate(5.0, 1000.0, 2))[-1]
    _within(np.array([final]), *AFTER_PULSE)


def test_the_core_orders_the_instants_it_is_given():
    """A cut spends a draw, so where the cuts fall decides the random stream:
    the same instants in another order, or twice, have to give the same run."""
    from bngsim._bngsim_core import RuleMonkeySimulator, TimeSpec

    ts = TimeSpec()
    ts.t_start, ts.t_end, ts.n_points = 0.0, 4.0, 5
    core = RuleMonkeySimulator(str(PLAIN))

    def run(breakpoints):
        core.set_clock_profile(True, breakpoints)
        return [list(column) for column in core.run(ts, 5).observable_data]

    ordered = run([0.5, 1.5, 2.5])
    assert run([2.5, 0.5, 1.5]) == ordered
    assert run([1.5, 0.5, 0.5, 2.5, 2.5, 1.5]) == ordered
    assert run([0.5, 1.5]) != ordered  # and a cut does move the stream
    with pytest.raises(RuntimeError, match="not a finite number"):
        core.set_clock_profile(True, [6.0, float("nan")])


def test_the_core_refuses_times_it_cannot_run():
    from bngsim._bngsim_core import RuleMonkeySimulator, TimeSpec

    def spec(t_start, t_end):
        ts = TimeSpec()
        ts.t_start, ts.t_end, ts.n_points = t_start, t_end, 3
        return ts

    for reads_clock in (True, False):
        core = RuleMonkeySimulator(str(PLAIN))
        core.set_clock_profile(reads_clock)
        for bad in (spec(0.0, float("inf")), spec(float("nan"), 1.0)):
            with pytest.raises(RuntimeError, match="not a finite number"):
                core.run(bad, 1)
        with pytest.raises(RuntimeError, match="earlier than"):
            core.run(spec(5.0, 3.0), 1)
        assert not core.has_session()
        core.initialize(1, 5.0)
        for bad in ((5.0, float("inf"), 3), (5.0, 3.0, 3)):
            with pytest.raises(RuntimeError):
                core.simulate(*bad)
        with pytest.raises(RuntimeError, match="not a finite number"):
            core.simulate(5.0, 6.0, 2, 0.0, [5.0, float("nan")])
        with pytest.raises(RuntimeError, match="not a finite number"):
            core.step_to(float("inf"))
        assert core.current_time() == 5.0
        core.destroy_session()


def test_a_timeout_inside_a_cut_run_is_a_timeout(tmp_path):
    from bngsim import SimulationTimeout

    xml = _pulse_expression(tmp_path, 0.02)
    sim = bngsim.Simulator(_model(), method="rm", xml_path=str(xml))
    with pytest.raises(SimulationTimeout):
        sim.run(t_span=(0.0, 1000.0), n_points=3, seed=3, timeout=1e-9)
    # And the simulator runs again afterwards.
    _within(
        np.asarray(sim.run(t_span=(0.0, 10.0), n_points=3, seed=3).observables["A0"])[-1:],
        *AFTER_PULSE,
    )


# ── A model without a clock keeps a small clock ───────────────────────────


@pytest.mark.parametrize("t0", [5.0, 1e9, 1e12, 1e13])
def test_a_late_start_without_a_clock_is_the_run_from_zero(t0):
    """Only elapsed time matters to this model, so it is run on elapsed time and
    is the run from 0, count for count, however late it starts. On the engine's
    absolute clock a start at 1e12 read 20000 at the second row (12131
    expected): the clock there cannot resolve the waits between events."""
    _, _, from_zero = _run(PLAIN, t_span=(0.0, 2.0), n_points=3)
    _, t, late = _run(PLAIN, t_span=(t0, t0 + 2.0), n_points=3)
    assert t[0] == t0 and t[-1] == t0 + 2.0
    np.testing.assert_array_equal(late, from_zero)


def test_output_times_closer_than_the_engine_tells_apart_are_still_read():
    """A model that reads the clock runs on absolute time, and the engine takes
    an output time within 1e-12 of t_end of a call's start for the start. At
    t = 1e9 that is 0.001, in which this model fires 10 of its 20000: the row
    read 20000. Each such row is now the first row of a call of its own."""
    t = np.array([1e9, 1e9 + 1e-3, 1e9 + 2e-3])
    mean, sd = _binomial(np.exp(-K * (t - 1e9)))  # the switch has been on since t = 6
    seeds = range(1, 101)
    sim = bngsim.Simulator(_model(), method="rm", xml_path=str(SWITCH))
    runs = np.array(
        [np.asarray(sim.run(sample_times=list(t), seed=s).observables["A0"]) for s in seeds]
    )
    np.testing.assert_array_equal(runs[:, 0], N)
    _within(runs.mean(axis=0), mean, sd / np.sqrt(len(seeds)))
    assert runs.mean(axis=0)[1] < N - 5.0  # 19990 expected, not the start state

    with bngsim.RuleMonkeySession(str(SWITCH)) as s:
        s.initialize(seed=1, t_start=1e9)
        session = _a0(s.simulate(sample_times=list(t)))
        assert s.current_time == t[-1]
    np.testing.assert_array_equal(session, runs[0])


# ── An engine that stops before the end of a call ─────────────────────────


def _printed_switch(tmp_path):
    """The plain decay at k, plus a function that is only printed and reads the
    clock: g() = if(time()>=50, 1, 0). No rate reads the clock, so the engine
    stops for good once every A(s~0) has gone, near t = 21."""
    text = SWITCH.read_text(encoding="utf-8")
    assert text.count(" if(time()>=6,k,0) ") == 1 and text.count("      </Function>\n") == 1
    text = text.replace(" if(time()>=6,k,0) ", " k ")
    printed = (
        '      <Function id="g">\n        <Expression> if(time()>=50,1,0) </Expression>\n'
        "      </Function>\n"
    )
    xml = tmp_path / "printed_switch.xml"
    xml.write_text(text.replace("      </Function>\n", "      </Function>\n" + printed), "utf-8")
    return xml


def test_a_run_that_stops_firing_before_a_cut_still_returns_every_row(tmp_path):
    """The run is cut at t = 50 for g, and the engine has nothing left to fire
    long before. The rows after that hold the state it stopped in, and g reads
    each row's own time."""
    xml = _printed_switch(tmp_path)
    times = [0.0, 1.0, 40.0, 50.0, 60.0, 100.0]
    r = bngsim.Simulator(_model(), method="rm", xml_path=str(xml)).run(sample_times=times, seed=2)
    np.testing.assert_array_equal(np.asarray(r.time), times)
    a = np.asarray(r.observables["A0"])
    _within(a[1:2], *_binomial(np.exp(-K * 1.0)))
    np.testing.assert_array_equal(a[2:], 0.0)
    np.testing.assert_array_equal(np.asarray(r.expressions["g"]), [0, 0, 0, 1, 1, 1])
    assert r.solver_stats["n_steps"] == N


def test_a_session_clock_stays_where_the_engine_stopped(tmp_path):
    """As after any RuleMonkey call that runs out of things to fire: the clock
    is the time of the last event, and the next segment starts there."""
    xml = _printed_switch(tmp_path)
    with bngsim.RuleMonkeySession(str(xml)) as s:
        s.initialize(seed=2)
        first = s.simulate(sample_times=[0.0, 40.0, 60.0, 100.0])
        stopped = s.current_time
        assert 5.0 < stopped < 40.0
        with pytest.raises(SimulationError, match="clock stops short of a segment's end"):
            s.simulate(100.0, 200.0, 3)
        second = s.simulate(sample_times=[stopped, 150.0])
    np.testing.assert_array_equal(_a0(first)[1:], 0.0)
    np.testing.assert_array_equal(np.asarray(first.expressions["g"]), [0, 0, 1, 1])
    np.testing.assert_array_equal(_a0(second), 0.0)
    np.testing.assert_array_equal(np.asarray(second.expressions["g"]), [0, 1])


# ── Calls that used to go wrong quietly ───────────────────────────────────


def test_a_session_segment_of_one_point_is_refused_before_it_runs():
    """It raised "mismatched time labels" after the session had advanced."""
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=1)
        with pytest.raises(SimulationError, match="n_points must be at least 2"):
            s.simulate(0.0, 1.0, 1)
        assert s.current_time == 0.0
        assert _a0(s.simulate(0.0, 1.0, 2))[0] == N


def test_step_to_does_not_go_back():
    """A target before the clock ran nothing and said nothing."""
    with bngsim.RuleMonkeySession(str(PLAIN)) as s:
        s.initialize(seed=1)
        s.step_to(2.0)
        with pytest.raises(SimulationError, match="earlier than the session's clock"):
            s.step_to(1.0)
        s.step_to(2.0)  # where it already is: nothing to do
        assert s.current_time == 2.0


@pytest.mark.parametrize(
    "times",
    [
        {"t_span": (0.0, float("inf")), "n_points": 3},
        {"t_span": (float("nan"), 1.0), "n_points": 3},
    ],
)
def test_a_run_over_times_that_are_not_numbers_is_refused(times):
    """t_span=(0, inf) returned rows at [nan, inf, inf] holding zeros."""
    sim = bngsim.Simulator(_model(), method="rm", xml_path=str(PLAIN))
    with pytest.raises((SimulationError, ValueError)):
        sim.run(seed=1, **times)
