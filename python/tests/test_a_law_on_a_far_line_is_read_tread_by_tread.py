"""A rate law on a line far from zero is read tread by tread (issue #1018).

The evaluator's value of `1e12 + time()` holds each of its doubles for a stretch
of time, a tread (1.2e-4), and a law that reads the clock only through such a
line is a staircase. The scan read it every quarter instant and ran out of
reads, or refused a sawtooth whose whole number of periods is beyond 2^48, and
the step resolver read a sawtooth of such a line as within its rounding of zero
on both sides of every wrap and placed none: refused, or 21% over by the sixth
of six legs.

Now a law the scan refuses or leaves without stops is read once on each tread,
and stopped on the last double before each change. Between two stops the rate
is then exactly constant. The oracle is the law summed tread by tread: where
`c + time()` is far from zero, the sum is `c + k*u` for `t` within half a double
`u` of `k*u`, and numpy's `fmod` and `floor` are C's.
"""

from __future__ import annotations

import math
import re

import bngsim
import numpy as np
import pytest
from bngsim import _clock_scan, _switch_sensitivity

_NET = """begin parameters
    1 one 1
end parameters
begin species
    1 X() {x0!r}
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""

_U12 = float(np.spacing(1e12))


def _model(tmp_path, dose: str, x0: float = 0.0) -> bngsim.Model:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(dose=dose, x0=x0), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _gain(tmp_path, dose: str, t0: float, t1: float, x0: float = 0.0, **kw) -> float:
    sim = bngsim.Simulator(_model(tmp_path, dose, x0), method="ode", **kw)
    return float(np.asarray(sim.run(t_span=(t0, t1), n_points=3).species)[-1, 0]) - x0


def _tread_sum(f, c: float, t0: float, t1: float) -> float:
    """``f(c + time())`` summed tread by tread over ``(t0, t1)``, for times
    small against ``c``."""
    u = float(np.spacing(c))
    k = np.arange(math.floor(t0 / u) - 1, math.ceil(t1 / u) + 2)
    lo = np.clip((k - 0.5) * u, t0, t1)
    hi = np.clip((k + 0.5) * u, t0, t1)
    return float(np.sum(f(c + k * u) * (hi - lo)))


def _fresh() -> None:
    """Forget what was read and said: a reading is remembered, and a warning
    said once a session."""
    _switch_sensitivity._SCHEDULE_CACHE.clear()
    _switch_sensitivity._SAID.clear()


def _said(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "#1018" in r.getMessage()]


class _Options:
    """What `_apply_crossing_stops` reads and writes, kept where it can be read."""

    max_step_size = 0.0

    def __init__(self, rtol: float = 1e-8):
        self.rtol = rtol
        self.stops = []

    def set_crossing_stops(self, stops, restarts=()):
        self.stops = [time for time, _clock, _value in stops]

    def set_crossing_probes(self, probes):
        pass


# (rate law, c, the law as numpy reads it of c + time(), run). The rows of the
# issue the scan refused.
_REFUSED = [
    pytest.param(
        "floor(mod(10000000000.0 + time(), 0.01)/0.01 + 0.25)",
        1e10,
        lambda x: np.floor(np.fmod(x, 0.01) / 0.01 + 0.25),
        (0.0, 1.0),
        id="more-of-the-run-than-can-be-read",
    ),
    pytest.param(
        "floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)",
        1e12,
        lambda x: np.floor(np.fmod(x, 0.3) / 0.3 + 0.0005),
        (1.3992, 241.3992),
        id="two-edges-inside-the-step-resolvers-rounding",
    ),
    *(
        pytest.param(
            f"100*mod(1000000000000.0 + time(), {n * _U12!r})",
            1e12,
            lambda x, n=n: 100 * np.fmod(x, n * _U12),
            (0.0, 100 * n * _U12),
            id=f"{n}-doubles-to-the-period",
        )
        for n in (5, 8, 12)
    ),
]


@pytest.mark.parametrize("dose, c, law, run", _REFUSED)
def test_a_law_the_scan_refused_is_read_tread_by_tread(tmp_path, dose, c, law, run):
    """Refused, with every row: 0.25 for 0.25, 0.1171875 for 0.12, the three
    remainders exact to 1e-14. A stop on the first double of each new value
    had them 5.6e-7 to 2.1e-3 off at these tolerances, in five to twelve times
    the steps: the step that ends on a stop reads the rate at its end."""
    exact = _tread_sum(law, c, *run)
    assert _gain(tmp_path, dose, *run) == pytest.approx(exact, rel=1e-9)


def test_a_sawtooth_beyond_2_48_says_what_it_resolves(tmp_path, caplog):
    dose = f"100*mod(1000000000000.0 + time(), {12 * _U12!r})"
    _fresh()
    with caplog.at_level("WARNING", logger="bngsim"):
        _gain(tmp_path, dose, 0.0, 1200 * _U12)
    said = [r.getMessage() for r in caplog.records if "2^48" in r.getMessage()]
    assert len(said) == 1
    assert "`mod(time(), P)`" in said[0] and "#1018" in said[0]


_COARSE = "100*mod(1100000000000.0 + time(), 0.0048828125)"  # 20 doubles of the sum


def _coarse(x):
    return 100 * np.fmod(x, 0.0048828125)


@pytest.mark.parametrize("n_points", [2, 3])
@pytest.mark.parametrize("x0", [0.0, 1e4])
def test_a_sawtooth_the_resolver_left_alone_adds_every_leg(tmp_path, x0, n_points):
    """The issue's last-but-one row: with no stop, six legs of 0.2 with two
    outputs to the leg were 1.073, 1.049, 1.037, 1.110 and 1.206 of the
    integral from the second on. (From X(0) = 1e4 the error test allows
    rtol*X = 1e-4 a step, and the first leg is 6e-5 under. With three outputs
    to the leg from X(0) = 0, main has them right too: the control.)"""
    sim = bngsim.Simulator(_model(tmp_path, _COARSE, x0), method="ode")
    for leg in range(6):
        result = sim.run(t_span=(0.2 * leg, 0.2 * (leg + 1)), n_points=n_points)
        gain = float(np.asarray(result.species)[-1, 0]) - x0
        exact = _tread_sum(_coarse, 1.1e12, 0.0, 0.2 * (leg + 1))
        assert gain == pytest.approx(exact, rel=1e-4, abs=1e-8 * x0 * (leg + 1))


@pytest.mark.parametrize("scale, x0", [(100.0, 1e4), (1.0, 0.0)])
def test_a_sawtooth_the_resolver_left_alone_follows_the_tolerance(tmp_path, scale, x0):
    """With no stop, 5.5% over from X(0) = 1e4, and 1.3% over at a scale of
    1: the error test finds each wrap or does not."""
    dose = f"{scale!r}*mod(1100000000000.0 + time(), 0.0048828125)"
    exact = _tread_sum(lambda x: scale * np.fmod(x, 0.0048828125), 1.1e12, 0.0, 20.2)
    assert _gain(tmp_path, dose, 0.0, 20.2, x0=x0) == pytest.approx(exact, rel=2e-5)


def test_each_stop_is_the_last_double_before_a_change(tmp_path):
    model = _model(tmp_path, _COARSE)
    opts = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 0.0, 0.1)
    stops = np.asarray(opts.stops)
    # A change on every tread of the sum in the run: from (k - 1/2)u to (k + 1/2)u.
    assert stops.size == math.ceil(0.1 / float(np.spacing(1.1e12)) - 0.5)
    text = "mod(1100000000000.0 + time(), 0.0048828125)"
    on = np.asarray(model._core.evaluate_over_times(text, stops))
    after = np.asarray(model._core.evaluate_over_times(text, np.nextafter(stops, np.inf)))
    before = np.asarray(model._core.evaluate_over_times(text, np.nextafter(stops, -np.inf)))
    assert np.all(on != after)
    assert np.all(on == before)


def test_a_sawtooth_whose_line_skips_doubles_is_read_by_the_sum_inside(tmp_path):
    """`(1.1e12 + time())/0.0048828125` steps 0.05 at a time among doubles
    0.03125 apart: no tread to a double of it. The sum inside it has one."""
    dose = "100*frac((1100000000000.0 + time())/0.0048828125)"
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(opts, 0.0, 0.1)
    assert len(opts.stops) == math.ceil(0.1 / float(np.spacing(1.1e12)) - 0.5)
    exact = _tread_sum(lambda x: 100 * np.fmod(x / 0.0048828125, 1.0), 1.1e12, 0.0, 0.1)
    assert _gain(tmp_path, dose, 0.0, 0.1) == pytest.approx(exact, rel=1e-9)


@pytest.mark.parametrize(
    "dose, run",
    [
        # 4.5e9 doubles of the argument to a period, and 6e10 treads in the run
        ("100*frac(1e6 + time())", (3.0, 10.0)),
        # a time of day on an epoch clock: 3.6e11 to a period
        ("100*mod(1700000000.0 + time(), 86400.0)/86400", (0.0, 0.2)),
        # 2,048 to a period, and 2 million changes in the run
        ("100*frac((3000000000.0 + time())/0.0009765625)", (1000000.023, 1000001.023)),
    ],
)
def test_a_fine_sawtooth_is_left_as_main_leaves_it(tmp_path, caplog, dose, run):
    """A staircase of steps under a thousandth of its period is the line it
    nearly is, and main integrates it as one: the time of day over 0.2 in 14
    steps, right to 4e-10, where a stop on each of its 838,861 treads took 1.7
    million steps and 5.6 s. Read only up to 1,024 doubles to a period."""
    _fresh()
    opts = _Options()
    with caplog.at_level("WARNING", logger="bngsim"):
        bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(opts, *run)
    assert opts.stops == []
    assert _said(caplog) == []


def test_a_fine_sawtooth_is_integrated_as_on_main(tmp_path):
    """Without stops, as on main, the error test finds each wrap: 4.6e-9 of
    the answer on arm64, 3.4e-8 on x86-64."""
    assert _gain(tmp_path, "100*frac(1e6 + time())", 3.0, 10.0) == pytest.approx(350.0, rel=1e-6)


def test_a_coarse_sawtooth_past_the_stop_budget_is_left_as_main_and_said(tmp_path, caplog):
    """1,024,000 changes over 250: no stops, as on main, and a warning with a
    leg that fits, rounded down (244.14 to 244)."""
    _fresh()
    opts = _Options()
    with caplog.at_level("WARNING", logger="bngsim"):
        bngsim.Simulator(_model(tmp_path, _COARSE), method="ode")._apply_crossing_stops(
            opts, 0.0, 250.0
        )
        bngsim.Simulator(_model(tmp_path, _COARSE), method="ode")._apply_crossing_stops(
            opts, 250.0, 500.0
        )
    assert opts.stops == []
    said = _said(caplog)
    assert len(said) == 1  # once, for two legs
    assert "changes 1024000 times" in said[0] and "legs of 244 " in said[0]


def test_the_suggested_leg_is_rounded_down():
    """A leg of 0.477 where 0.476837 fits held 1,000,341 changes."""
    assert _switch_sensitivity._three_places_down(0.476837) == "0.476"
    assert _switch_sensitivity._three_places_down(1.0) == "1"
    assert _switch_sensitivity._three_places_down(123456.0) == "1.23e+05"


def test_a_law_beside_a_schedule_that_takes_the_budget_is_left_as_main(tmp_path, caplog):
    """990,099 stops of a schedule and 20,480 changes of the sawtooth are more
    than a run's budget. The sawtooth's are the ones left out, as main leaves
    them, and said: taken first, they put the schedule over and the run was
    refused."""
    _fresh()
    dose = f"{_COARSE} + if(mod(time(), 0.0000101) < 0.00000505, 1, 0)"
    opts = _Options(rtol=1e-3)
    with caplog.at_level("WARNING", logger="bngsim"):
        bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
            opts, 0.0, 5.0
        )
    assert len(opts.stops) == 990099
    said = _said(caplog)
    assert len(said) == 1 and "changes 20480 times" in said[0] and "990,099 stops" in said[0]


@pytest.mark.parametrize("t0", [3e7, 1.2e7])
def test_changes_closer_than_cvode_stops_apart_are_left_as_main(tmp_path, t0):
    """At t = 3e7 a tread of `3e9 + time()` is 128 doubles of the clock, and
    CVODE takes a stop as reached within 100*uround*t of it: stopped on each,
    `1e4*mod(3e9 + time(), 40u)` lost 63% of X, where main is 1% over."""
    period = 40 * float(np.spacing(3e9))
    dose = f"10000.0*mod(3000000000.0 + time(), {period!r})"
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
        opts, t0 + 0.023, t0 + 0.0231
    )
    assert opts.stops == []


def test_a_sawtooth_beyond_2_48_that_no_resolver_stops_is_read(tmp_path, caplog):
    """`100*mod(3e9 + time(), 12u)`: a resolver's "no change", no stop and no
    word on main, 11% low. (At a scale of 1e8, for an X the tolerance reads.)"""
    period = 12 * float(np.spacing(3e9))
    dose = f"100000000.0*mod(3000000000.0 + time(), {period!r})"
    run = (0.0, 100 * period)
    _fresh()
    with caplog.at_level("WARNING", logger="bngsim"):
        gain = _gain(tmp_path, dose, *run)
    exact = _tread_sum(lambda x: 1e8 * np.fmod(x, period), 3e9, *run)
    assert gain == pytest.approx(exact, rel=1e-9)
    assert [m for m in _said(caplog) if "2^48" in m]


def test_a_sawtooth_beyond_2_48_that_cannot_be_read_is_left_as_main(tmp_path):
    """At t = 1e6 the stops of `100*mod(1e8 + time(), 20u)` are closer than
    CVODE can stop apart. Main has a resolver's "no change" and no stop, and
    is within a tenth of a local tolerance: refused, the run was refused where
    main is right."""
    period = 20 * float(np.spacing(1e8))
    dose = f"100*mod(100000000.0 + time(), {period!r})"
    opts = _Options()
    _fresh()
    bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
        opts, 1e6, 1e6 + 1e-4
    )
    assert opts.stops == []
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    sim.run(t_span=(1e6, 1e6 + 1e-4), n_points=2)


def test_a_sawtooth_beyond_2_48_past_the_budget_is_left_as_main(tmp_path, caplog):
    """`100*mod(1.1e12 + time(), 12u)` over 300, a resolver's "no change" on
    main: 1.2 million changes. Main runs it without stops, 2.7 local
    tolerances off from X(0) = 1e4, and so does this, with a warning and a leg
    that fits; it was refused."""
    period = 12 * float(np.spacing(1.1e12))
    dose = f"100*mod(1100000000000.0 + time(), {period!r})"
    opts = _Options()
    _fresh()
    with caplog.at_level("WARNING", logger="bngsim"):
        bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
            opts, 0.0, 300.0
        )
    assert opts.stops == []
    said = _said(caplog)
    assert len(said) == 1 and "legs of 244 " in said[0] and "2^48" not in said[0]


def test_a_sawtooth_beyond_2_48_beside_a_step_main_places_leaves_it_its_stops(tmp_path, caplog):
    """`floor(time()/0.00005)` has its 899,999 stops, as on main, and the
    sawtooth's 184,320 are what is left out: counted first, they put the floor
    over the budget and the run was refused, whatever max_step was passed."""
    period = 12 * float(np.spacing(1.1e12))
    dose = f"100*mod(1100000000000.0 + time(), {period!r}) + floor(time()/0.00005)"
    opts = _Options(rtol=1e-3)
    _fresh()
    with caplog.at_level("WARNING", logger="bngsim"):
        bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
            opts, 0.0, 45.0
        )
    assert len(opts.stops) == 899999
    said = _said(caplog)
    assert len(said) == 1 and "changes 184320 times" in said[0]


@pytest.mark.parametrize(
    "dose",
    ["floor((10000000000.0 + time())/(0.01))", "floor(((10000000000.0+time()))/(0.01))"],
    ids=["parenthesized-divisor", "sbml-quotient"],
)
def test_a_line_inside_a_step_whose_argument_is_no_line_is_read(tmp_path, dose):
    """A number in parentheses is no piece of a line, so `(1e10 + time())/(0.01)`
    is none, and SBML's `quotient` loads so: refused, as on main, though the
    clock is read once, inside `1e10 + time()`."""
    exact = _tread_sum(lambda x: np.floor(x / 0.01), 1e10, 0.0, 1.0)
    assert _gain(tmp_path, dose, 0.0, 1.0) == pytest.approx(exact, rel=1e-9)


def test_a_law_that_reads_the_far_line_outside_the_condition_too_is_refused_as_on_main(tmp_path):
    """`rem(1.1e12 + time, P)` written out as `x - P*floor(x/P)`: a stop on
    each change of the `floor` is not a stop on each step of `x`, and the run
    was 9e-5 off at rtol 1e-8, where main refuses it. (SBML's own `<rem/>` by a
    parameter now loads with a truncating sign test, and is refused too: see
    test_sbml_quotient_rem_truncate.py.)"""
    x = "((1100000000000.0+time()))"
    dose = f"100*({x} - (0.0048828125)*floor({x}/(0.0048828125)))"
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match="cannot be placed"):
        sim.run(t_span=(0.0, 0.2), n_points=2)


def test_whether_a_reading_clears_cvodes_stops_is_not_where_the_run_ends():
    """28.6 of CVODE's stop tolerances between treads: read. The last tread's
    stretch was measured to the run's end, and a run ending 100 doubles into
    a tread was declined (main's error, 1.2e-2, where read it is 1.9e-4)."""
    period = 40 * float(np.spacing(3e9))
    table = {"(x)": lambda t: 3e9 + t, "s": lambda t: np.fmod(3e9 + t, period)}

    def ev(text, at):
        return table[text](np.asarray(at, dtype=float))

    lo = 7.5e5 + 0.023
    u = float(np.spacing(3e9 + lo))
    # A change of `3e9 + time()` some 4,000 treads in: half way between two of
    # its doubles, exactly, at this clock.
    x = 3e9 + lo + 4000 * u
    change = (x - 3e9) + 0.5 * u
    for hi in (change + 0.5 * u, change + 100 * float(np.spacing(change))):
        read = _clock_scan.tread_changes(ev, "s", "x", lo, hi, 10**6)
        assert read is not None and read.stops is not None
        assert read.changes == 4001  # 4,000 boundaries before the change, and it


def test_the_leg_offered_fits_where_the_treads_double(tmp_path):
    """`(2^40 - 150) + time()` crosses 2^40 at 150, where its doubles double:
    the leg offered for a run of 300, the run times the budget over the count,
    held 1,277,952 changes."""
    c = 2.0**40 - 150
    dose = f"100*mod({c!r} + time(), {12 * float(np.spacing(c))!r})"
    _fresh()
    with pytest.raises(bngsim.SimulationError, match=r"legs of ([0-9.]+) or shorter") as refused:
        bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
            _Options(), 0.0, 300.0
        )
    leg = float(re.search(r"legs of ([0-9.]+) or shorter", str(refused.value)).group(1))
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(opts, 0.0, leg)
    assert 0 < len(opts.stops) <= 1_000_000


def test_the_leg_offered_past_the_reads_fits_the_stops_too(tmp_path, caplog):
    """4.5 million treads over 1,100: past the reads. The leg was the run times
    the reads over the treads, 1,020, which held 4 million changes."""
    _fresh()
    with caplog.at_level("WARNING", logger="bngsim"):
        bngsim.Simulator(_model(tmp_path, _COARSE), method="ode")._apply_crossing_stops(
            _Options(), 0.0, 1100.0
        )
    said = _said(caplog)
    assert len(said) == 1 and "legs of 244 " in said[0]


def test_a_refusal_for_the_budget_is_not_told_of_a_staircase(tmp_path, caplog):
    """The 2^48 warning, that the run is made as a staircase with a stop on
    each step, was said before the run was refused for the budget."""
    dose = f"100*mod(1000000000000.0 + time(), {12 * _U12!r})"
    _fresh()
    with caplog.at_level("WARNING", logger="bngsim"), pytest.raises(bngsim.SimulationError):
        bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
            _Options(), 0.0, 160.0
        )
    assert not [m for m in _said(caplog) if "2^48" in m]


def test_a_law_no_rate_law_reads_is_not_read(tmp_path):
    """An output: a stop on each tread of it took 165,555 steps where the run
    took 9."""
    path = tmp_path / "out.net"
    path.write_text(
        _NET.format(dose="one", x0=0.0).replace(
            "    1 dose() one", f"    1 dose() one\n    2 out() {_COARSE}"
        ),
        encoding="utf-8",
    )
    model = bngsim.Model.from_net(str(path))
    assert model.time_discontinuity_conditions()
    opts = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 0.0, 20.2)
    assert opts.stops == []


def test_what_was_read_for_an_output_is_not_the_answer_for_a_rate_law(tmp_path):
    """A reading is remembered by the condition and the window, and which model
    asked is no part of that: read for an output and remembered as no stops,
    the same law in a rate law next had none."""
    _fresh()
    path = tmp_path / "out.net"
    path.write_text(
        _NET.format(dose="one", x0=0.0).replace(
            "    1 dose() one", f"    1 dose() one\n    2 out() {_COARSE}"
        ),
        encoding="utf-8",
    )
    opts = _Options()
    bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")._apply_crossing_stops(
        opts, 0.0, 0.2
    )
    assert opts.stops == []
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, _COARSE), method="ode")._apply_crossing_stops(opts, 0.0, 0.2)
    assert len(opts.stops) == math.ceil(0.2 / float(np.spacing(1.1e12)) - 0.5)


@pytest.mark.parametrize(
    "dose, c",
    [
        ("floor(mod(1e12 + (time() - 1), 0.3)/0.3 + 0.0005)", 1e12 - 1),
        ("floor(mod((time() - 1) + 1e12, 0.3)/0.3 + 0.0005)", 1e12 - 1),
        ("floor(mod(1e12 + (time()), 0.3)/0.3 + 0.0005)", 1e12),
    ],
)
def test_a_part_of_the_line_too_fine_to_read_is_passed_over(tmp_path, dose, c):
    """`time() - 1` is a double of the clock to a tread, and was taken for the
    line: refused, as on main."""
    run = (1.3992, 241.3992)
    exact = _tread_sum(lambda x: np.floor(np.fmod(x, 0.3) / 0.3 + 0.0005), c, *run)
    assert _gain(tmp_path, dose, *run) == pytest.approx(exact, rel=1e-9)


_COUNTER_NET = """begin parameters
    1 one 1
end parameters
begin species
    1 X() 0
    2 T() 0
end species
begin functions
    1 dose() floor(mod(1e12 + t, 0.3)/0.3 + 0.0005)
end functions
begin reactions
    1 0 1 dose #_R1
    2 0 2 one #clock
end reactions
begin groups
    1 t 2
end groups
"""


def test_a_law_on_a_counter_clock_is_read_as_before(tmp_path):
    """The integrated counter lands a few of its doubles past a stop's value,
    and a stop a double before a change is not a stop before it there."""
    path = tmp_path / "counter.net"
    path.write_text(_COUNTER_NET, encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    conds = model.time_discontinuity_conditions()
    answers = []
    for by_tread in (False, True):
        _fresh()
        over: list = []
        found, periodic, _ = _switch_sensitivity.crossings_with_periodic(
            model._core, 1.3992, 241.3992, conds, over_budget=over, by_tread=by_tread
        )
        answers.append(([s.time for s in found], [s.time for s in periodic], over))
    assert answers[0] == answers[1]


@pytest.mark.parametrize("max_step", [0.0, 1e-5])
def test_a_law_the_scan_refused_past_the_stop_budget_is_refused_by_name(tmp_path, max_step):
    """1,310,720 changes over 160. A step bound of half a tread in place of
    the stops left it 0.4% off over 130 at the default tolerances, and 4.5%
    from X(0) = 1e4, where over 100, a stop on each change, it is exact to
    rounding. So it is refused whatever bound is passed, with a leg that fits
    (160 * 1e6 / 1,310,720 = 122.07)."""
    dose = f"100*mod(1000000000000.0 + time(), {12 * _U12!r})"
    opts = _Options()
    opts.max_step_size = max_step
    _fresh()
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"#1018.*legs of 122 or shorter") as refused:
        sim._apply_crossing_stops(opts, 0.0, 160.0)
    assert "changes 1310720 times" in str(refused.value)


def test_a_law_the_scan_refused_past_the_stop_budget_beside_another_is_refused(tmp_path):
    dose = f"100*mod(1000000000000.0 + time(), {12 * _U12!r}) + if(time() > 5, 1, 0)"
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match="1,000,000 stops"):
        sim.run(t_span=(0.0, 160.0), n_points=2)


def test_a_run_with_sensitivity_columns_is_refused_as_before(tmp_path):
    """Its stops are its switch times, and those stay the resolvers' own."""
    path = tmp_path / "s.net"
    path.write_text(
        _NET.format(dose="k*floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)", x0=0.0).replace(
            "1 one 1", "1 one 1\n    2 k 1"
        ),
        encoding="utf-8",
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["k"]
    )
    with pytest.raises(bngsim.SimulationError, match="cannot be placed"):
        sim.run(t_span=(1.3992, 241.3992), n_points=2)


@pytest.mark.parametrize(
    "by_tread, touches", [(False, False), (True, True)], ids=["ssa-path", "event-trigger"]
)
def test_the_ssa_and_an_event_trigger_are_not_read_tread_by_tread(tmp_path, by_tread, touches):
    """The SSA asks for none (a stop a double before a change holds the old
    propensity to it), and an event trigger's condition keeps its stops."""
    model = _model(tmp_path, "floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)")
    conds = model.time_discontinuity_conditions()
    over: list = []
    _switch_sensitivity._SCHEDULE_CACHE.clear()
    found, periodic, _ = _switch_sensitivity.crossings_with_periodic(
        model._core,
        1.3992,
        241.3992,
        conds,
        touches=frozenset(conds) if touches else frozenset(),
        over_budget=over,
        by_tread=by_tread,
    )
    assert not found and not periodic
    assert len(over) == 1 and over[0].flicker is not None


def _ev(table):
    return lambda text, at: table[text](np.asarray(at, dtype=float))


def test_a_line_that_skips_doubles_or_crosses_zero_is_not_read():
    skips = _ev({"(x)": lambda t: (1.9e12 + t) * 1.1, "s": lambda t: t})
    assert _clock_scan.tread_changes(skips, "s", "x", 0.0, 1.0, 10**6) is None
    crosses = _ev({"(x)": lambda t: t - 0.5, "s": lambda t: t})
    assert _clock_scan.tread_changes(crosses, "s", "x", 0.0, 1.0, 10**6) is None


def test_the_parts_of_a_line_that_hold_the_clock():
    assert _clock_scan.lines_holding_the_clock("((2e9 + time())/3 + 5)*7") == [
        "2e9 + time()",
        "(2e9 + time())/3 + 5",
        "((2e9 + time())/3 + 5)*7",
    ]
    assert _clock_scan.lines_holding_the_clock("1e12 + time()") == ["1e12 + time()"]


# ── Round 3: where it is read with another such law, or beside one past the budget ──
def test_two_laws_read_so_in_one_run_are_as_on_main(tmp_path):
    """`1e12 + time()` and `1e12 + 3*time()` end treads on the same instants:
    two stops a double or two apart, which a run merges onto the later, on the
    earlier's change. A third of the changes had no stop before them (+977
    local tolerances). Each alone is read; the two together are main's: the
    remainder pair refused, the coarse pair without stops."""
    period = 12 * _U12
    one = f"100*mod(1000000000000.0 + time(), {period!r})"
    three = f"100*mod(1000000000000.0 + 3*time(), {period!r})"
    _fresh()
    sim = bngsim.Simulator(_model(tmp_path, f"{one} + {three}"), method="ode")
    with pytest.raises(bngsim.SimulationError, match="cannot be placed"):
        sim.run(t_span=(0.0, 0.15), n_points=2)
    opts = _Options()
    coarse = f"{_COARSE} + 100*mod(1100000000000.0 + 3*time(), 0.0048828125)"
    bngsim.Simulator(_model(tmp_path, coarse), method="ode")._apply_crossing_stops(opts, 0.0, 0.2)
    assert opts.stops == []


def test_stops_read_so_beside_a_condition_past_the_budget_are_as_on_main(tmp_path):
    """Two floors of 600,000 changes each are past the budget, and the step is
    bounded at half the shortest stretch between their changes. Placed, the
    sawtooth's stops cut that stretch from 5e-6 to 3.9e-8, and a run main
    makes under `max_step=2e-6` was refused for a bound of 3e8 steps."""
    dose = f"{_COARSE} + floor(time()/0.00001) + floor(time()/0.00001 + 0.5)"
    opts = _Options()
    opts.max_step_size = 2e-6
    _fresh()
    bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(opts, 0.0, 6.0)
    assert opts.max_step_size == 2e-6


def test_an_ordinary_schedule_does_not_read_the_rate_laws_each_leg(tmp_path, monkeypatch):
    """`mod(time(), 24) < 12` starts like a sawtooth call, and over a leg with
    no change of it every rate law was flattened, at three times main's cost
    a leg."""
    calls = []
    real = _switch_sensitivity._rate_law_flats
    monkeypatch.setattr(
        _switch_sensitivity, "_rate_law_flats", lambda core: calls.append(1) or real(core)
    )
    _fresh()
    sim = bngsim.Simulator(_model(tmp_path, "if(mod(time(), 24) < 12, 1, 0)"), method="ode")
    for leg in range(5):
        sim.run(t_span=(0.2 * leg + 0.1, 0.2 * leg + 0.3), n_points=2)
    assert calls == []
    assert _switch_sensitivity._a_whole_sawtooth_call("mod(1e12 + time(), 0.3)")
    assert not _switch_sensitivity._a_whole_sawtooth_call("mod(time(), 24) < 12")
    assert not _switch_sensitivity._a_whole_sawtooth_call("mod(time(), 2) + mod(time(), 3)")


def test_what_a_reading_found_is_not_said_of_a_run_another_condition_refuses(tmp_path, caplog):
    """Said as `floor(time()/1e-7)` refused the run, the 2^48 warning was
    spent, and the legs run next had none."""
    dose = f"100*mod(1000000000000.0 + time(), {12 * _U12!r}) + floor(time()/0.0000001)"
    _fresh()
    with caplog.at_level("WARNING", logger="bngsim"):
        with pytest.raises(bngsim.SimulationError):
            bngsim.Simulator(_model(tmp_path, dose), method="ode").run(
                t_span=(0.0, 1200 * _U12), n_points=2
            )
        assert not [m for m in _said(caplog) if "2^48" in m]
        bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
            _Options(), 0.0, 0.05
        )
    assert len([m for m in _said(caplog) if "2^48" in m]) == 1
