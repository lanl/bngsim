"""A run with stops looks inside the stretches #998's look did not reach (issue
#1192).

Issue #998 made a run with stops watch CVODE's steps until a stretch has been
seen, and read inside a step about to take most of what is left of it before
the step is taken: after a flat restart a step can grow over a whole stretch
and pass over a pulse inside it, with no error. That look did not reach:

1. the steps CVODE takes inside a call to an output row: a step that could
   cross the row was taken in CV_NORMAL to it, and where it failed, its retries
   and the steps after them were taken inside the call, unseen;
2. a stretch while a delayed event waits to be executed: the call to the
   event's time was left to CVODE;
3. the stretch after a stop that restarts nothing (a linear table's
   breakpoint): CVODE does not report the step it takes after one;
4. a model with a stiff species anywhere in it: on the state CVODE's history
   predicts, a stiff rate is off by |lambda|*h times what CVODE's corrector
   leaves of it, so the look took a step CVODE passed for one it fails;

and it read only steps that grew with nothing measured, from half of what is
left.

`0 -> X` at the law, so X gains exactly the law's integral, which is the oracle
throughout (by quadrature, never a run). Every case is right on main with
`max_step` no larger than the pulse.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

_NET = """begin parameters
    1 one 1
{params}end parameters
begin species
    1 X() 0
{species}end species
begin functions
    1 dose() {law}
{functions}end functions
begin reactions
    1 0 1 dose #_R1
{reactions}end reactions
"""


def _model(tmp_path, law, params="", species="", functions="", reactions=""):
    path = tmp_path / "m.net"
    path.write_text(
        _NET.format(
            law=law, params=params, species=species, functions=functions, reactions=reactions
        )
    )
    return bngsim.Model.from_net(str(path))


def _gain(model, t0, t1, **kw):
    """What a run over (t0, t1) adds to X, and its solver statistics."""
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(t0, t1), n_points=kw.pop("n_points", 3), **kw)
    x = np.asarray(result.species)[:, 0]
    return float(x[-1] - x[0]), dict(result.solver_stats)


def _quadrature(f, a, b, n=2_000_001):
    grid = np.linspace(a, b, n)
    v = f(grid)
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(grid)))


_LATE = "mod(time(), 1)^40/(0.9^40 + mod(time(), 1)^40)"
_LATE_A_PERIOD = _quadrature(lambda x: x**40 / (0.9**40 + x**40), 0.0, 1.0)
_NOON = "5*exp(-((mod(time(), 24) - 12)/0.75)^2)"
_NOON_A_DAY = _quadrature(lambda u: 5 * np.exp(-(((u - 12) / 0.75) ** 2)), 0.0, 24.0)


@pytest.mark.parametrize(
    "rows_a_day", [pytest.param(2, id="every-12-h"), pytest.param(3, id="every-8-h")]
)
def test_a_daily_pulse_with_rows_through_the_day(tmp_path, rows_a_day):
    """A dose at noon each day for 100 days, with the usual output grid. Where
    the step that could cross a row failed, CVODE's retries and the steps after
    them were taken inside the call to the row, and one grew over the dose. On
    main 0.28 and 0.40 of the integral."""
    x, _ = _gain(_model(tmp_path, _NOON), 0.0, 2400.0, n_points=100 * rows_a_day + 1)
    assert x == pytest.approx(100 * _NOON_A_DAY, rel=2e-4)


def test_a_late_pulse_with_rows_inside_each_period_from_a_late_start(tmp_path):
    """The late pulse over periods 1,000 to 1,200 with 4 rows a period. On main
    none of the integral."""
    x, _ = _gain(_model(tmp_path, _LATE), 1000.0, 1200.0, n_points=801)
    assert x == pytest.approx(200 * _LATE_A_PERIOD, rel=2e-4)


_U = "(time - 24*floor(time/24))"


@pytest.mark.parametrize(
    "rows_a_day", [pytest.param(2, id="every-12-h"), pytest.param(3, id="every-8-h")]
)
def test_rows_where_roots_are_searched(rows_a_day):
    """The noon dose beside an event that never fires, so CVODE searches a root
    all run: a tout inside a step would split that search, so the call to the
    row is cut off after each step instead. On main 0.27 and 0.39."""
    text = (
        f"model m\n X = 0; Y = 0\n J1: -> X; 5*exp(-(({_U} - 12)/0.75)^2)\n J2: -> Y; 0\n"
        " E1: at (X > 1e9): Y = Y + 1\nend"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    result = sim.run(t_span=(0.0, 2400.0), n_points=100 * rows_a_day + 1)
    x = np.asarray(result.species)[:, 0]
    assert x[-1] == pytest.approx(100 * _NOON_A_DAY, rel=2e-4)
    assert np.asarray(result.species)[-1, 1] == 0.0


_FRAC = "(time - floor(time))"
_MIDDLE_A_PERIOD = _quadrature(lambda u: np.exp(-(((u - 0.5) / 0.05) ** 2)), 0.0, 1.0)


@pytest.mark.parametrize(
    "event, t_end, executions",
    [
        pytest.param(f"at 0.8 after ({_FRAC} > 0.1): Y = Y + 1", 100.0, 100, id="each-period"),
        pytest.param("at 150 after (time > 0.1): Y = Y + 1", 200.0, 1, id="for-150-periods"),
    ],
)
def test_a_stretch_while_a_delayed_event_waits(event, t_end, executions):
    """A pulse in the middle of each period while an event waits to be
    executed, at 0.9 of each period or once at t = 150.1. The call to the
    event's time was left to CVODE. On main 0.08 and 0.75 of the integral.
    The events run as they did."""
    text = (
        f"model m\n X = 0; Y = 0\n J1: -> X; exp(-(({_FRAC} - 0.5)/0.05)^2)\n J2: -> Y; 0\n"
        f" E1: {event}\nend"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    species = np.asarray(sim.run(t_span=(0.0, t_end), n_points=3).species)
    assert species[-1, 0] == pytest.approx(t_end * _MIDDLE_A_PERIOD, rel=2e-4)
    assert species[-1, 1] == executions


def _sine_table(n, period, start=0.3):
    xs = [start + k for k in range(n)]
    ys = [float(np.sin(2 * np.pi * x / period)) for x in xs]
    return (
        f"tfun([{','.join(repr(v) for v in xs)}],[{','.join(repr(v) for v in ys)}], "
        'time, method=>"linear")'
    )


class _Options:
    max_step_size = 0.0
    rtol = 1e-6
    max_steps = 10000
    stops: list = []
    restarts: list = []

    def set_crossing_stops(self, stops, restarts=()):
        self.stops = list(stops)
        self.restarts = list(restarts)

    def set_crossing_probes(self, probes):
        pass


def test_the_stretch_after_a_stop_that_restarts_nothing(tmp_path):
    """The late pulse plus 1e-30 of a linear table of a sine sampled 0.3 into
    each period: the table's small bends are stops the run ends a step on and
    does not restart at. The stretch after one was closed with the one before
    it, and the first step after the stop, sized from CVODE's history, took
    the rest of the period over the pulse. On main 0.7833, every period lost
    one that holds such a stop."""
    model = _model(
        tmp_path,
        f"{_LATE} + 1e-30*tf()",
        functions=f"    2 tf() {_sine_table(60, 30.0)}\n",
    )
    options = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(options, 0.0, 60.0)
    held = [
        t for (t, *_), restart in zip(options.stops, options.restarts, strict=True) if not restart
    ]
    assert len(held) >= 10  # or the run below proves nothing

    x, _ = _gain(model, 0.0, 60.0)
    assert x == pytest.approx(60 * _LATE_A_PERIOD, rel=2e-4)


@pytest.mark.parametrize("k", [1e3, 1e4, 1e6])
def test_a_stiff_species_elsewhere_in_the_model(tmp_path, k):
    """The late pulse beside an unrelated `S + B <-> C` at K both ways, S
    decaying at 1e-4. On main 0.57 at K = 1e3 and 0.04 at 1e4 and 1e6: the look
    left to CVODE a step CVODE passed. Read through I - gamma*J, as CVODE's
    corrector reads it, the step passes and is read inside."""
    model = _model(
        tmp_path,
        _LATE,
        params=f"    2 K {k!r}\n    3 kd 1e-4\n",
        species="    2 S() 1\n    3 B() 1\n    4 C() 0\n",
        reactions="    2 2,3 4 K #_R2\n    3 4 2,3 K #_R3\n    4 2 0 kd #_R4\n",
    )
    x, _ = _gain(model, 0.0, 100.0)
    assert x == pytest.approx(100 * _LATE_A_PERIOD, rel=2e-4)


def _two_pulses(u):
    return 9.95 * np.exp(-(((u - 0.18) / 0.031) ** 2)) + 0.985 * np.exp(
        -(((u - 0.186) / 0.0198) ** 2)
    )


@pytest.mark.parametrize(
    "law, period, periods, f",
    [
        pytest.param(
            "9.95*exp(-((mod(time(), 1) - 0.18)/0.031)^2)"
            " + 0.985*exp(-((mod(time(), 1) - 0.186)/0.0198)^2)",
            1.0,
            30,
            _two_pulses,
            id="early-pulse",
        ),
        pytest.param(
            "2.39*exp(-((mod(time(), 24) - 8.98)/1.006)^2)",
            24.0,
            72,
            lambda u: 2.39 * np.exp(-(((u - 8.98) / 1.006) ** 2)),
            id="hour-wide-dose-at-9-h",
        ),
    ],
)
def test_a_step_of_a_quarter_of_what_is_left_is_read(tmp_path, law, period, periods, f):
    """A pulse early in each period, where a step over it took less than half
    of what was left and was not read. On main 0.8667 and 0.9861 of the
    integral (four periods of 30, one day of 72)."""
    x, _ = _gain(_model(tmp_path, law), 0.0, periods * period)
    assert x == pytest.approx(periods * _quadrature(f, 0.0, period), rel=2e-4)


def test_a_step_whose_end_reads_one_to_ten_times_stays_right_at_its_cost(tmp_path):
    """Issue #1192's cost case: a narrow pulse and a late rise, with rows. Its
    look's stops are all in steps whose ends read as passing, so it costs what
    main's does: 947 steps where they were measured, with room for a
    platform's rounding."""
    law = "0.826*exp(-((mod(time(), 1) - 0.478)/0.027)^2) + 0.0055*mod(time(), 1)^40"
    a_period = _quadrature(
        lambda u: 0.826 * np.exp(-(((u - 0.478) / 0.027) ** 2)) + 0.0055 * u**40, 0.0, 1.0
    )
    x, stats = _gain(_model(tmp_path, law), 0.0, 5.0, n_points=21)
    assert x == pytest.approx(5 * a_period, rel=1e-4)
    assert stats["n_steps"] <= 1.02 * 947


@pytest.mark.parametrize(
    "rows, main_steps",
    [(3, 39_071), (201, 39_071), (801, 35_069)],
    ids=["no-rows", "rows", "4-a-period"],
)
def test_a_square_wave_beside_a_root_steps_as_before(rows, main_steps):
    """A square wave beside an event that never fires: each stretch is read and
    nothing found, and the calls cut off after a step take the steps the calls
    to the rows took, roots searched over the same brackets. Main's step counts,
    with room for a platform's rounding (bitwise main's where measured)."""
    text = (
        f"model m\n X = 0; Y = 0\n J1: -> X; piecewise(3, {_FRAC} < 0.5, 1)\n J2: -> Y; 0\n"
        " E1: at (X > 1e9): Y = Y + 1\nend"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    result = sim.run(t_span=(0.0, 200.0), n_points=rows)
    assert np.asarray(result.species)[-1, 0] == pytest.approx(400.0, rel=1e-7)
    assert result.solver_stats["n_steps"] <= 1.02 * main_steps
