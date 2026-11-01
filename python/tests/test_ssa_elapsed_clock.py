"""Issue #1059: an SSA run from a large t_start keeps its waiting times.

Far from the origin the doubles are coarse, 1.2e-4 apart at t = 1e12, and a
waiting time placed at ``t + tau`` loses what is below that spacing. #1075 made
the two discrete loops carry what rounding drops. The continuous loop, which
runs a model whose rate reads the clock, places its firings on panels in
absolute time, and kept the loss: a decay ``k*A*(1 + 1e-30*time())`` from 1e5
copies, whose exact mean at t0 + 2 is 1e5/e = 36788, came back 22829 (z = -289)
from t0 = 1e12. The run now counts its own clock from t_start, and hands the
model t_start + elapsed where the model reads the clock; rows report the times
asked for, and the state and events carried to a next run their absolute times.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

T0 = 1e12
EXACT = 1e5 * np.exp(-1.0)


def _ssa(text, **run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_antimony_string(text)
        return bngsim.Simulator(model, method="ssa").run(**run)


def _z(values):
    values = np.asarray(values, dtype=float)
    se = np.sqrt(EXACT * (1 - np.exp(-1.0)) / len(values))
    return (values.mean() - EXACT) / se


@pytest.mark.parametrize(
    "law",
    [
        pytest.param("k*A*(1 + 1e-30*time())", id="continuous loop (reads the clock)"),
        pytest.param("k*A", id="discrete loop (#1075)"),
    ],
)
def test_a_decay_from_a_far_start_keeps_its_rate(law):
    text = f"species A = 100000; k = 0.5; R: A -> ; {law};"
    ends = [
        np.asarray(_ssa(text, sample_times=[T0, T0 + 1, T0 + 2], seed=s + 1).species)[-1, 0]
        for s in range(10)
    ]
    assert abs(_z(ends)) < 5.0, _z(ends)


def test_the_rows_are_the_times_asked_for():
    times = [T0, T0 + 0.5, T0 + 2]
    r = _ssa(
        "species A = 1000; k = 0.5; R: A -> ; k*A*(1 + 1e-30*time());", sample_times=times, seed=1
    )
    assert list(np.asarray(r.time)) == times


def test_an_event_at_an_absolute_time_fires_there():
    """The clock the trigger reads is absolute: halving A at t0 + 1 shows
    between the rows at t0 + 0.9 and t0 + 1.1, nowhere else."""
    text = (
        "species A = 1000000; k = 1e-12; R: A -> ; k*A*(1 + 1e-30*time());\n"
        f"E: at (time > {T0!r} + 1): A = A/2;\n"
    )
    r = _ssa(text, sample_times=[T0, T0 + 0.9, T0 + 1.1, T0 + 2], seed=1)
    a = np.asarray(r.species)[:, 0]
    assert a[1] > 9.9e5 and 4.9e5 < a[2] < 5.1e5 and abs(a[3] - a[2]) < 10


def test_a_rate_that_switches_at_an_absolute_time_switches_there():
    """A rate on from t0 + 1 (a breakpoint, shifted onto the run's clock):
    nothing decays before it, and after it the decay is e^-k(t - t0 - 1)."""
    text = f"species A = 100000; k = 1; R: A -> ; k*A*piecewise(1, time() > {T0!r} + 1, 0);"
    ends = []
    for s in range(10):
        r = _ssa(text, sample_times=[T0, T0 + 1, T0 + 2], seed=s + 1)
        a = np.asarray(r.species)[:, 0]
        assert a[1] == 100000
        ends.append(a[2])
    assert abs(_z(ends)) < 5.0, _z(ends)


def test_a_run_continues_from_where_the_last_one_ended():
    """The state, the time and an execution still waiting are carried to the
    next run at their absolute times: a delayed event that fires at t0 + 1 and
    applies at t0 + 3 applies in the second leg, at t0 + 3."""
    text = (
        "species A = 1000; k = 1e-12; R: A -> ; k*A*(1 + 1e-30*time());\n"
        f"E: at 2 after (time > {T0!r} + 1): A = 7;\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_antimony_string(text)
        sim = bngsim.Simulator(model, method="ssa")
        first = sim.run(sample_times=[T0, T0 + 1, T0 + 2], seed=1)
        second = sim.run(sample_times=[T0 + 2, T0 + 2.9, T0 + 3.1], seed=2)
    a1, a2 = np.asarray(first.species)[:, 0], np.asarray(second.species)[:, 0]
    assert a1[-1] == 1000 and a2[0] == 1000 and a2[1] == 1000 and a2[2] == 7
    assert list(np.asarray(second.time)) == [T0 + 2, T0 + 2.9, T0 + 3.1]


def test_a_long_run_hands_on_its_exact_end():
    """A run longer than its start, t_end > 2*t_start, where t_start + (t_end -
    t_start) is not t_end in doubles. The next run continues from the exact
    t_end, and so finds the execution still waiting: one that fires at t0 + 1
    with a delay of t_end - t0 applies a second into the second leg."""
    t0, te = 1146994.9100613801, 4086777.6982969143
    assert t0 + (te - t0) != te
    text = (
        "species A = 1000; k = 1e-12; R: A -> ; k*A*(1 + 1e-30*time());\n"
        f"E: at {te - t0!r} after (time > {t0!r} + 1): A = 7;\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ssa")
        sim.run(sample_times=[t0, t0 + 2, te], seed=1)
        second = sim.run(sample_times=[te, te + 0.5, te + 1.5], seed=2)
    np.testing.assert_array_equal(np.asarray(second.species)[:, 0], [1000, 1000, 7])


def test_psa_from_a_far_start_is_the_run_from_zero():
    """PSA shares the run's body: from t0 = 1e12 the continuous-loop decay is
    the run from 0, firing for firing (the rate's 1e-30*time() rounds away at
    both), where an absolute clock lost its waiting times."""
    text = "species A = 100000; k = 0.5; R: A -> ; k*A*(1 + 1e-30*time());"

    def ends(t0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = bngsim.Simulator(
                bngsim.Model.from_antimony_string(text), method="psa", poplevel=100
            )
            return [
                np.asarray(sim.run(sample_times=[t0, t0 + 1, t0 + 2], seed=s + 1).species)[-1, 0]
                for s in range(5)
            ]

    np.testing.assert_array_equal(ends(T0), ends(0.0))


def test_an_event_just_past_a_switch_leaves_the_rate_switched_on():
    """A rate held between breakpoints, re-read after an event: the event fires
    on the first firing past t0 + 1, within half an ulp of t0 + 1 at 1e12, where
    the clock reads t0 + 1 and `time() > t0 + 1` is still false. Read there, the
    rate went back to 0 for good: 99999 of 100000 left where e^-1 should be."""
    text = (
        f"species A = 100000; species B = 0; k = 1;\n"
        f"R: A => ; k*A*piecewise(1, time() > {T0!r} + 1, 0);\n"
        "E: at (A < 99999.5): B = 1;\n"
    )
    ends = []
    for s in range(10):
        a = np.asarray(_ssa(text, sample_times=[T0, T0 + 1, T0 + 2], seed=s + 1).species)
        assert a[1, 0] == 100000 and a[-1, 1] == 1
        ends.append(a[-1, 0])
    assert abs(_z(ends)) < 5.0, _z(ends)


def test_a_row_just_past_a_switch_is_not_refused():
    """A row at t0 + 1.00005 reads the clock at t0 + 1 (the doubles there are
    1.2e-4 apart), on the breakpoint, where the held rate's guard must not
    take the condition's old side for a rate that moved."""
    text = f"species A = 100000; k = 1; R: A => ; k*A*piecewise(1, time() > {T0!r} + 1, 0);"
    r = _ssa(text, t_span=(T0, T0 + 2.0001), n_points=3, seed=1)
    a = np.asarray(r.species)[:, 0]
    exact = 1e5 * np.exp(-1.0001)
    assert a[0] == 100000 and a[1] > 99900 and abs(a[2] - exact) < 5 * np.sqrt(exact), a


@pytest.mark.parametrize("tau", [3.0, 7.3, 14.0, 5.1, 9.9, 123.456, 0.37, 41.2])
def test_a_strict_trigger_runs_after_its_non_strict_twin(tau):
    """#947 from a far start: `lhs > tau` beside `lhs >= tau` turn true at one
    read on the crossing, and the strict one runs after its twin, X = 0.3*0.5*X.
    The probe that tells a crossing from a jump steps back in the time the model
    reads, t0 + elapsed: a step sized on the elapsed clock alone was lost in its
    rounding, and the pair ran as one batch, X = 0.5*X."""
    text = (
        f"species X; X = 100; T0 = {T0!r}; tau = {tau!r}\nJ0: -> X; 0\n"
        "E1: at (time - T0 > tau), priority = 3: X = 0.3*X\n"
        "E2: at (time - T0 >= tau), priority = 2: X = 0.5*X\n"
    )
    times = [T0, T0 + tau / 2, T0 + 1.4 * tau, T0 + 2.1 * tau]
    assert np.asarray(_ssa(text, sample_times=times, seed=1).species)[-1, 0] == pytest.approx(15.0)


@pytest.mark.parametrize(
    "rhs, exact",
    [("1e5*(time() - T0)", 5e4), ("1e5*(time() - T0)^2", 1e5 / 3)],
    ids=["linear", "quadratic"],
)
def test_a_rate_rule_that_reads_the_clock(rhs, exact):
    """The rate rule's dX/dt at a far start, beside a held rate and a firing
    reaction. The panel's d/dt of the rule steps where the clock the model
    reads, t0 + elapsed, moves: a step below its ulp read no change or a whole
    ulp, 0 or a spike. And a firing leaves the model at its own time, where
    the next panel starts, not inside the held rate's interval."""
    text = (
        f"T0 = {T0!r}; species A = 1000; species B = 1000; k = 0.5;\n"
        "R1: A => ; k*A*piecewise(1, time() > T0 + 5, 0);\n"
        "R2: B => ; k*B;\n"
        f"species X = 0; X' = {rhs};\n"
    )
    r = _ssa(text, sample_times=[T0, T0 + 0.5, T0 + 1, T0 + 10], seed=1)
    x = np.asarray(r.species)[2, 2]
    # The clock reads 1.2e-4 apart at 1e12: X is right to about that, relative.
    assert abs(x - exact) < 1e-4 * exact, x


@pytest.mark.parametrize(
    "law", ["k*A*(1 + 1e-30*time())", "k*A"], ids=["continuous loop", "discrete loop"]
)
def test_a_t_span_row_is_taken_at_the_time_it_reports(law):
    """`t_span` and `n_points`: each row is taken where the clock the model reads
    is the time the row reports, to the bit. A grid drawn again on the run's
    clock rounded apart from the reported one by an ulp of t0, and a row
    reported just past an event's time was taken before it."""
    text = f"species A = 1000000; k = 1e-12; R: A => ; {law}; clock := time();"
    for span, n in [(2.0001, 7), (3.3, 11), (1.7, 13), (10.0 / 3, 9), (0.77, 23)]:
        r = _ssa(text, t_span=(T0, T0 + span), n_points=n, seed=1)
        np.testing.assert_array_equal(np.asarray(r.expressions["clock"]), np.asarray(r.time))
    event = text + f"\nE: at (time > {T0!r} + 1): A = A/2;"
    r = _ssa(event, t_span=(T0, T0 + 2.0001), n_points=7, seed=1)
    a, t = np.asarray(r.species)[:, 0], np.asarray(r.time)
    np.testing.assert_array_equal(a > 7.5e5, t <= T0 + 1)


@pytest.mark.parametrize(
    "start, times",
    [
        pytest.param("", [-T0, -T0 + 1, -T0 + 2], id="near -1e12"),
        pytest.param("E: at (time > -2): A = 100000;", [-T0, -2.0, -1.0, 0.0], id="-1e12 to 0"),
    ],
)
def test_a_decay_at_far_negative_times_keeps_its_rate(start, times):
    """The run's clock counts from the end of the run nearer 0. From -1e12 to
    -1e12 + 2 that is t_end, as fine as a far start's; from -1e12 to 0 it is
    0, the clock as it was. Counted from t_start, the half of the run toward 0
    was coarser than the absolute clock: z = -289 over the last two units."""
    initial = 100000 if not start else 0
    text = f"species A = {initial}; k = 0.5; R: A => ; k*A*(1 + 1e-30*time());\n{start}\n"
    ends = [
        np.asarray(_ssa(text, sample_times=times, seed=s + 1).species)[-1, 0] for s in range(10)
    ]
    assert abs(_z(ends)) < 5.0, _z(ends)


@pytest.mark.parametrize(
    "t_span, n",
    [((1146994.9100613801, 4086777.6982969143), 777), ((-(2.0**21), 1.0), 1001)],
    ids=["t_end > 2*t_start", "across 0"],
)
def test_a_row_is_read_at_its_time_where_the_shift_is_inexact(t_span, n):
    """Past 2*t_start, t - t_start rounds, and t_start + it misses t by an ulp:
    each time is moved onto the run's clock where the clock the model reads is
    the time itself, so a `time()` column is r.time to the bit."""
    text = "species A = 1000000; k = 1e-12; R: A => ; k*A*(1 + 1e-30*time()); clock := time();"
    r = _ssa(text, t_span=t_span, n_points=n, seed=1)
    np.testing.assert_array_equal(np.asarray(r.expressions["clock"]), np.asarray(r.time))


def _burst(t_busy, law="k*A*(1 + 1e-30*time())"):
    """Quiet from 0, then 1e5 copies decaying at 0.5 from t_busy - 2."""
    return (
        f"species A = 0; k = 0.5; R: A => ; {law};\nE: at (time > {t_busy!r} - 2): A = 100000;\n"
    )


def _run(text, method="ssa", poplevel=None, **run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_antimony_string(text)
        kw = {} if poplevel is None else {"poplevel": poplevel}
        return bngsim.Simulator(model, method=method, **kw).run(**run)


_STEADY = "species A = 0; x = 0; x' = 0; R: => A; 5e4*x;\nE: at (time > {t!r} - 2): x = 1;\n"


@pytest.mark.parametrize(
    "text, method, poplevel",
    [
        pytest.param(_burst(3.2e10), "ssa", None, id="ssa decay"),
        pytest.param(_burst(3.2e10), "psa", 1e6, id="psa decay"),
        pytest.param(_STEADY.format(t=3.2e10), "ssa", None, id="ssa steady, through a rate rule"),
    ],
)
def test_a_run_too_long_for_its_clock_is_refused(text, method, poplevel):
    """One run from 0 that turns busy at 3.2e10: 5e4 firings per unit there,
    where the doubles are 3.8e-6 apart, 0.19 per spacing. Each waiting time
    after a firing that restarts the panel rounds onto them, and the run fired
    too fast with no error: a decay busy at 1e12 kept 22831 of an exact 36788.
    PSA at a poplevel above every population fires one molecule at a time, as
    SSA does."""
    t_busy = 3.2e10
    with pytest.raises(bngsim.SimulationError, match="fires faster than its clock"):
        _run(text, method, poplevel, sample_times=[0.0, t_busy - 2, t_busy - 1, t_busy], seed=1)


def test_firings_that_keep_their_panel_are_not_refused():
    """A synthesis whose rate reads the clock and nothing a firing moves: each
    firing keeps its panel and the next is placed inside it, exactly, however
    coarse the doubles. Busy at 1e12, 6.1 firings per spacing, it is right."""
    t_busy = 1e12
    text = (
        f"species A = 0; R: => A; 5e4*(1 + 1e-30*time())*piecewise(1, time > {t_busy!r} - 2, 0);"
    )
    times = [0.0, t_busy - 2, t_busy - 1, t_busy]
    ends = [
        np.asarray(_run(text, sample_times=times, seed=s + 1).species)[-1, 0] for s in range(10)
    ]
    z = (np.mean(ends) - 1e5) / np.sqrt(1e5 / len(ends))
    assert abs(z) < 5.0, z


def test_fast_firings_between_rare_restarts_are_not_refused():
    """A fast exchange whose firings keep the panel, 5e4 a unit at 1e12 (6.1 per
    spacing), beside a slow decay that restarts it 8 times a unit. Only the
    restarts round onto the clock, and their gaps are long, so the run is
    right; the rate judged is the restarting one."""
    t_busy = 1e12
    text = (
        "species A = 0; species B = 0; species C = 0; species D = 8000; kf = 500;\n"
        "R1: A => B + C; kf*A; R2: B => A; kf*B;\n"
        f"F: => A; 1e-3*(1 + 1e-30*time())*piecewise(1, time > {t_busy!r} - 2, 0);\n"
        f"S: D => ; 1e-3*D*(1 + 1e-30*time())*piecewise(1, time > {t_busy!r} - 2, 0);\n"
        f"E: at (time > {t_busy!r} - 2): A = 100;\n"
    )
    times = [0.0, t_busy - 2, t_busy - 1, t_busy]
    # C counts R1's firings: 500*50 a unit at equilibrium, and 25 on the way there.
    ends = [
        np.asarray(_run(text, sample_times=times, seed=s + 1).species)[-1, 2] for s in range(3)
    ]
    np.testing.assert_allclose(np.mean(ends), 50025, rtol=0.01)


def test_one_fast_firing_is_not_refused():
    """The rate the next waiting time runs at is judged, not the one before the
    firing: one molecule at 1e6 a unit at 1e12 fires once, and nothing rounds
    after it."""
    t_busy = 1e12
    text = (
        "species A = 0; k = 1e6; R: A => ; k*A*(1 + 1e-30*time());\n"
        f"E: at (time > {t_busy!r} - 2): A = 1;\n"
    )
    r = _run(text, sample_times=[0.0, t_busy - 2, t_busy - 1, t_busy], seed=1)
    assert np.asarray(r.species)[-1, 0] == 0


@pytest.mark.parametrize(
    "t_busy, law",
    [
        pytest.param(1.6e10, "k*A*(1 + 1e-30*time())", id="continuous loop, 0.095 per spacing"),
        pytest.param(1e12, "k*A", id="discrete loop (#1075 carries the rounding)"),
    ],
)
def test_a_long_run_its_clock_can_place_runs_and_keeps_its_rate(t_busy, law):
    """Just under the limit, the continuous loop runs, and its rate is right;
    the discrete loop carries what rounds away, so it is never refused."""
    times = [0.0, t_busy - 2, t_busy - 1, t_busy]
    ends = [
        np.asarray(_run(_burst(t_busy, law), sample_times=times, seed=s + 1).species)[-1, 0]
        for s in range(10)
    ]
    assert abs(_z(ends)) < 5.0, _z(ends)


def test_the_remedy_is_a_leg_that_starts_nearer_the_busy_stretch():
    """What the refusal says to do: run_until from a time near the busy
    stretch, which counts its clock from there."""
    t_busy = 1e12
    ends = []
    for s in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = bngsim.Simulator(bngsim.Model.from_antimony_string(_burst(t_busy)), method="ssa")
            sim.run(sample_times=[0.0, t_busy / 2, t_busy - 2.5], seed=s + 1)
            ends.append(np.asarray(sim.run_until(t_busy, n_points=3, seed=s + 101).species)[-1, 0])
    assert abs(_z(ends)) < 5.0, _z(ends)


def test_psa_whose_leaps_the_clock_can_place_is_not_refused():
    """At poplevel 100 PSA leaps about 1000 molecules a firing: 50 firings per
    unit at 1e12, 0.006 per spacing. It runs, and busy at 1e12 its mean is the
    one busy at 1e6 (a leap rounding past a row moves a run by 1000)."""

    def mean(t_busy):
        times = [0.0, t_busy - 2, t_busy - 1, t_busy]
        return np.mean(
            [
                np.asarray(
                    _run(_burst(t_busy), "psa", 100, sample_times=times, seed=s + 1).species
                )[-1, 0]
                for s in range(5)
            ]
        )

    np.testing.assert_allclose(mean(1e12), mean(1e6), rtol=0.02)
