"""A run with stops looks inside a stretch before it takes the stretch in one
step after a flat restart (issue #998).

After a start or a restart where the right-hand side is flat, CVODE's first
step has no error to speak of, and the step after it may be 10,000 times as
long. From about the 72nd of evenly spaced stops on, whatever the spacing, that
step reached the next stop and was cut to end on it, where a rate law that
repeats reads what it read at the restart. The step read the right-hand side at
its end alone, found nothing changed and was accepted with nothing added for
the stretch. A dose train that begins each period flat gave 0.72 of its
integral over 100 periods and 0.365 over 200; a leg of ten periods from
t = 1,000 periods gave a tenth; the remainder of a far clock,
`100*mod(1e12 + time(), 0.125)`, 0.72. With no error.

Now, until a stretch has been looked at, the run watches CVODE's steps. Before
one that grew with nothing measured and would take half or more of what is
left of the stretch, it reads the right-hand side where that step ends, on
the state CVODE's history predicts there and against the rate it predicts.
Where CVODE's error test would fail the step there, the step is left to CVODE,
as on main. Where it would pass, the step is read at a quarter, a half and
three quarters of its length, and the first place where a step would fail
becomes a stop that restarts the run.

Oracles: the integral of each law in closed form or by quadrature (`0 -> X`
adds exactly that to X); for a sawtooth of a far clock the law summed tread by
tread of its argument, which is what the model's evaluator computes; where
events reset the state, a run of the same model under a small `max_step`, which
cannot step over a period. None of them is the run under test.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _switch_sensitivity

_NET = """begin parameters
    1 one 1
end parameters
begin species
    1 X() {x0!r}
end species
begin functions
    1 dose() {law}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""


def _model(tmp_path, law, x0=0.0, name="m.net"):
    path = tmp_path / name
    path.write_text(_NET.format(law=law, x0=x0))
    return bngsim.Model.from_net(str(path))


def _gain(tmp_path, law, t0, t1, x0=0.0, codegen=None, **kw):
    """What a run over (t0, t1) adds to X, and its solver statistics."""
    sim = bngsim.Simulator(_model(tmp_path, law, x0), method="ode", codegen=codegen)
    result = sim.run(t_span=(t0, t1), n_points=kw.pop("n_points", 3), **kw)
    return float(np.asarray(result.species)[-1, 0]) - x0, dict(result.solver_stats)


def _quadrature(f, a, b, n=2_000_001):
    grid = np.linspace(a, b, n)
    v = f(grid)
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(grid)))


def _tread_sum(scale, offset, modulus, t0, t1, kind="mod"):
    """scale*mod(offset + t, modulus), or scale*frac((offset + t)/modulus),
    integrated as the evaluator computes it: constant on each double of
    offset + t, which holds from half a double before to half a double after."""
    u = float(np.spacing(abs(offset)))
    k = np.arange(int(np.floor(t0 / u)) - 1, int(np.ceil(t1 / u)) + 2)
    lo = np.clip((k - 0.5) * u, t0, t1)
    hi = np.clip((k + 0.5) * u, t0, t1)
    x = offset + k * u
    value = np.fmod(x, modulus) if kind == "mod" else np.fmod(x / modulus, 1.0)
    return float(np.sum(scale * value * (hi - lo)))


_HILL = "mod(time(), 24)^10/(12^10 + mod(time(), 24)^10)"
_HILL_A_PERIOD = _quadrature(lambda x: x**10 / (12.0**10 + x**10), 0.0, 24.0)
_LATE = "mod(time(), 1)^40/(0.9^40 + mod(time(), 1)^40)"
_LATE_A_PERIOD = _quadrature(lambda x: x**40 / (0.9**40 + x**40), 0.0, 1.0)
_BUMP = "218790*(mod(time(), 1)*(1 - mod(time(), 1)))^8"
_BUMP_A_PERIOD = _quadrature(lambda x: 218790 * (x * (1 - x)) ** 8, 0.0, 1.0)
# law, period, its integral over a period
_TRAINS = [
    pytest.param(_HILL, 24.0, _HILL_A_PERIOD, id="hill-of-a-24h-clock"),
    pytest.param("9*mod(time(), 1)^8", 1.0, 1.0, id="eighth-power"),
    # Still flat nine tenths of the way through its period.
    pytest.param(_LATE, 1.0, _LATE_A_PERIOD, id="late-pulse"),
    # Flat at both ends of its period, so the stop sees nothing either side.
    pytest.param(_BUMP, 1.0, _BUMP_A_PERIOD, id="bump"),
]


@pytest.mark.parametrize("law, period, a_period", _TRAINS)
@pytest.mark.parametrize("periods", [60, 100, 200])
def test_a_dose_train_that_begins_each_period_flat_adds_every_period(
    tmp_path, law, period, a_period, periods
):
    """On main the first two: 1.0000 at 60 periods, 0.7200 at 100, 0.3650 at
    200. The late pulse: 0.0667, 0.0400, 0.0200. The bump: 1.0, 0.61, 0.305.
    Each run ends on a wrap."""
    x, _ = _gain(tmp_path, law, 0.0, periods * period)
    assert x == pytest.approx(periods * a_period, rel=2e-4)


@pytest.mark.parametrize("law, period, a_period", _TRAINS)
@pytest.mark.parametrize("periods", [2, 10])
def test_a_leg_that_begins_late_adds_every_period(tmp_path, law, period, a_period, periods):
    """The first step of a run is the same step: from t = 1,000 periods its
    reach covers a period at once. On main: 0.5 of two periods, 0.1 of ten."""
    x, _ = _gain(tmp_path, law, 1000 * period, (1000 + periods) * period)
    assert x == pytest.approx(periods * a_period, rel=2e-4)


@pytest.mark.parametrize(
    "law, t0, t1, x0, exact, rel",
    [
        ("9*mod(time(), 1)^8", 0.0, 100.0, 1000.0, 100.0, 2e-3),
        ("9*mod(time(), 1)^8", 90.0, 100.0, 1000.0, 10.0, 2e-3),
        ("0.001*mod(time(), 1)^8", 0.0, 100.0, 0.0, 100 * 0.001 / 9, 5e-3),
        (_HILL, 0.0, 200 * 24.0, 1e6, 200 * _HILL_A_PERIOD, 1e-2),
    ],
    ids=["from-1000", "ten-periods-from-1000", "a-small-dose", "hill-from-1e6"],
)
def test_a_dose_small_against_the_tolerance_is_added_too(tmp_path, law, t0, t1, x0, exact, rel):
    """Where rtol*|X| + atol is large against a dose, the law is flat to the
    tolerance for longer into each period. On main 0.72, 0.0, 0.72 and 0.36 of
    the integral. (Held to the tolerance of the run, which for X = 1e6 is 1e-2
    a step.)"""
    x, _ = _gain(tmp_path, law, t0, t1, x0=x0)
    assert x == pytest.approx(exact, rel=rel)


@pytest.mark.parametrize(
    "modulus, periods",
    [(0.125, 100), (0.125, 200), (0.25, 100), (0.5, 100)],
)
def test_the_remainder_of_a_far_clock_adds_every_period(tmp_path, modulus, periods):
    """The issue's table: `100*mod(1e12 + time(), P)`, a stop on every wrap.
    On main 0.7195 and 0.3597 of the sum at P = 0.125, 0.7198 at 0.25 and 0.6299
    at 0.5."""
    law = f"100*mod(1e12 + time(), {modulus!r})"
    t1 = periods * modulus
    x, _ = _gain(tmp_path, law, 0.0, t1)
    assert x == pytest.approx(_tread_sum(100.0, 1e12, modulus, 0.0, t1), rel=1e-3)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_a_compiled_right_hand_side_is_looked_inside_alike(tmp_path, codegen):
    """The look reads the interpreter's right-hand side whichever one the run
    integrates with. On main 0.72 either way."""
    x, _ = _gain(tmp_path, "9*mod(time(), 1)^8", 0.0, 100.0, codegen=codegen)
    assert x == pytest.approx(100.0, rel=2e-4)


def test_rows_inside_the_stretches_change_nothing(tmp_path):
    """A row on every period and rows at times of no particular kind. On main
    0.72 and 0.56 of the integral."""
    law = "9*mod(time(), 1)^8"
    x, _ = _gain(tmp_path, law, 0.0, 100.0, n_points=101)
    assert x == pytest.approx(100.0, rel=2e-4)
    sim = bngsim.Simulator(_model(tmp_path, law), method="ode")
    times = [0.0] + [k + 0.37 for k in range(0, 150, 7)] + [150.0]
    result = sim.run(sample_times=times)
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(150.0, rel=2e-4)


_U = "(time - floor(time))"


@pytest.mark.parametrize(
    "event, t_end",
    [
        ("at (time >= 50.5): X = 0", 100.0),
        ("at (X >= 29.5): X = 0", 100.0),
        ("at (Y >= 39.5): X = X + 5, Y = 0", 200.0),
    ],
    ids=["at-a-time", "on-the-state", "on-another-species"],
)
def test_events_beside_the_stops_see_every_period(event, t_end):
    """An SBML dose train into X and Y, with an event that resets one of them.
    The oracle is the same model under `max_step=0.01`, which cannot step over
    a period. On main X ends at 22.0 for 50.0, 13.0 for 11.5, and 78.0 for 225.0.

    The state thresholds are crossed inside a period. A period adds exactly 1,
    so a threshold of 40 is reached on a period's last instant, and when the
    event fires there (and what Y reads at t = 200) is decided by rounding:
    the oracle itself read 1.4e-3 on macOS and 1.6e-5 on Linux."""
    text = f"model m\n X = 0; Y = 0\n J1: -> X; 9*{_U}^8\n J2: -> Y; 9*{_U}^8\n E1: {event}\nend"

    def run(**kw):
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
        return np.asarray(sim.run(t_span=(0.0, t_end), n_points=3, **kw).species)[-1]

    np.testing.assert_allclose(run(), run(max_step=0.01), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize(
    "offset, t0, span",
    [(3e9, 1e6, 1.0), (3e9, 1e6, 10.0), (2e9, 1e6, 1.0)],
)
def test_a_stop_on_every_wrap_of_a_coarse_sawtooth_adds_every_period(
    tmp_path, monkeypatch, offset, t0, span
):
    """The scan does not place stops on the wraps of a sawtooth whose argument
    is coarser than the clock (`_coarse_sawtooth`), because a run that
    restarted on each of them lost the stretches behind them: 0.0010, 0.0001
    and 0.5029 of the sum over treads on main. The run is what is fixed here;
    placing the stops again is issue #1018. Placed here by hand."""
    monkeypatch.setattr(_switch_sensitivity, "_coarse_sawtooth", lambda *a, **k: False)
    law = f"100*frac(({offset!r} + time())/0.0009765625)"

    class _Options:
        max_step_size = 0.0
        rtol = 1e-6
        max_steps = 10000
        stops: list = []

        def set_crossing_stops(self, stops, restarts=()):
            self.stops = list(stops)

        def set_crossing_probes(self, probes):
            pass

    options = _Options()
    sim = bngsim.Simulator(_model(tmp_path, law), method="ode")
    sim._apply_crossing_stops(options, t0, t0 + span)
    assert len(options.stops) >= 1000  # or the run below proves nothing

    x, _ = _gain(tmp_path, law, t0, t0 + span)
    exact = _tread_sum(100.0, offset, 0.0009765625, t0, t0 + span, kind="frac")
    assert x == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize(
    "law, t1, main_steps",
    [
        ("if(mod(time(), 1) < 0.5, 3, 1)", 1000.0, 180_315),
        ("100*mod(time(), 0.125)", 125.0, 75_459),
    ],
    ids=["square-wave", "sawtooth"],
)
def test_a_run_whose_stretches_are_not_flat_steps_as_before(tmp_path, law, t1, main_steps):
    """A square wave is flat behind each restart and is looked at, but the look
    finds nothing the history does not predict, and a sawtooth is a straight
    line the history predicts exactly: neither run places a stop, and each is
    main's step for step. The step counts are main's on the machine they were
    measured on, with room for a platform's rounding."""
    x, stats = _gain(tmp_path, law, 0.0, t1)
    exact = 2.0 * t1 if law.startswith("if") else 100 * 0.125**2 / 2 * (t1 / 0.125)
    assert x == pytest.approx(exact, rel=1e-5)
    assert stats["n_steps"] <= 1.02 * main_steps


_LATE_ROWS = [pytest.param(4, id="4-rows-a-period"), pytest.param(10, id="10-rows-a-period")]


@pytest.mark.parametrize("rows_a_period", _LATE_ROWS)
def test_rows_inside_each_period_do_not_hide_the_stretch(tmp_path, rows_a_period):
    """With rows inside each period the first step after a restart is sized
    from the next row, and it is the third or fourth step that takes the rest
    of the stretch, not the second. On main 0.020 of the integral with either.
    (Where a step that crosses a row fails inside the call to it: issue
    #1192's tests.)"""
    x, _ = _gain(tmp_path, _LATE, 0.0, 200.0, n_points=200 * rows_a_period + 1)
    assert x == pytest.approx(200 * _LATE_A_PERIOD, rel=2e-4)


@pytest.mark.parametrize(
    "rows", [pytest.param(4 * 100 + 1, id="6-hourly"), pytest.param(24 * 100 + 1, id="hourly")]
)
def test_a_late_daily_dose_with_rows_through_the_day(tmp_path, rows):
    """A dose that peaks at 21 h of each day, over 100 days, with the usual
    output grid. On main 0.02 of the integral."""
    law = "5*exp(-((mod(time(), 24) - 21)/0.75)^2)"
    a_day = _quadrature(lambda u: 5 * np.exp(-(((u - 21) / 0.75) ** 2)), 0.0, 24.0)
    x, _ = _gain(tmp_path, law, 0.0, 2400.0, n_points=rows)
    assert x == pytest.approx(100 * a_day, rel=2e-4)


def test_a_narrow_pulse_in_each_period_is_added_from_the_first(tmp_path):
    """A Gaussian pulse at the middle of each period, dead at both ends of it:
    every period kept, the early ones too, where the steps have not yet grown
    to the stretch. On main 0.04 of the integral."""
    law = "exp(-((mod(time(), 1) - 0.5)/0.05)^2)"
    a_period = _quadrature(lambda u: np.exp(-(((u - 0.5) / 0.05) ** 2)), 0.0, 1.0)
    sim = bngsim.Simulator(_model(tmp_path, law), method="ode")
    x = np.asarray(sim.run(t_span=(0.0, 100.0), n_points=101).species)[:, 0]
    np.testing.assert_allclose(np.diff(x), a_period, rtol=1e-3)


_RAMP_PULSE = "10.0*exp(-((mod(time(), 1) - 0.64)/0.03)^2) + {eps!r}*mod(time(), 1)^4"


def _ramp_pulse_period(eps):
    return _quadrature(lambda u: 10 * np.exp(-(((u - 0.64) / 0.03) ** 2)) + eps * u**4, 0.0, 1.0)


@pytest.mark.parametrize(
    "eps, kw, rel",
    [
        (0.001, {"rtol": 1e-6, "atol": 1e-6}, 1e-3),
        (0.001, {"rtol": 1e-6, "atol": 1e-6, "n_points": 101}, 1e-3),
        (1e-7, {}, 1e-3),
        (3e-7, {}, 1e-3),
        (1e-6, {}, 1e-3),
    ],
    ids=["ramp-1e-3", "ramp-1e-3-rows", "ramp-1e-7", "ramp-3e-7", "ramp-1e-6"],
)
def test_a_pulse_beyond_a_small_change_is_not_stepped_over(tmp_path, eps, kw, rel):
    """A pulse at 0.64 of each period on a small ramp, over 72 periods, where
    the ramp is what CVODE's steps see. A look that placed stops wherever the
    ramp showed changed where CVODE's steps ended, and one cut to end on such a
    stop passed over the pulse that CVODE's own step would have ended in: 0.44
    of the integral where main has 1.0002. Now the look leaves a step to CVODE
    wherever CVODE's error test would fail it at its own end. On main 1.0002,
    0.83, 0.07, 0.22 and 0.65."""
    x, _ = _gain(tmp_path, _RAMP_PULSE.format(eps=eps), 0.0, 72.0, **kw)
    assert x == pytest.approx(72 * _ramp_pulse_period(eps), rel=rel)


def test_a_narrow_pulse_on_a_trend_is_left_to_cvode(tmp_path):
    """Round 4 of the review: a narrow pulse on a small trend, at the default
    tolerances. A look that chased the trend with stops at a quarter of what
    was left cut CVODE's steps to end on them, and one passed over the pulse:
    0.368 of the integral where main has 1.000057. CVODE's own steps end where
    the trend fails them; the look leaves those alone, and the run is main's."""
    law = "10.0*exp(-((mod(time(), 1) - 0.543)/0.01)^2) + 0.001*mod(time(), 1)^2"
    a_period = _quadrature(
        lambda u: 10 * np.exp(-(((u - 0.543) / 0.01) ** 2)) + 0.001 * u**2, 0.0, 1.0
    )
    x, _ = _gain(tmp_path, law, 0.0, 30.0, x0=100.0)
    assert x == pytest.approx(30 * a_period, rel=2e-4)


@pytest.mark.parametrize(
    "rtol, atol",
    [(1e-8, 1e-10), (1e-8, 1e-12), (1e-6, 1e-12)],
)
def test_a_late_pulse_at_a_tight_absolute_tolerance(tmp_path, rtol, atol):
    """A tight `atol` makes the ramp before a late pulse count: a look that
    stopped at the half let the next step take the pulse. On main 0.04."""
    x, _ = _gain(tmp_path, _LATE, 0.0, 100.0, rtol=rtol, atol=atol)
    assert x == pytest.approx(100 * _LATE_A_PERIOD, rel=2e-4)


def test_a_row_after_a_root_stepped_on_through_is_read_at_its_time():
    """After a root CVODE steps on through without a restart (a bend whose
    if() branch is off, #1147), it reports the step it took as its next, not
    the one it takes. A step taken alone on that report crossed the next row,
    and the row was read where the step ended: S at t = 32.25 was 0.0012066
    for 0.0011703. On main 2.1e-7 at worst, against 0.0008 + 0.0005*sin(t)."""
    text = (
        "species S = 0.0008; k = 0.5\nJ1: -> S; 0.0005*cos(time)\n"
        "J2: S -> ; piecewise(k*S*S/max(S, 0.001), time > 50, 0)\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    s = np.asarray(sim.run(t_span=(0.0, 60.0), n_points=241).species)[:, 0]
    t = np.linspace(0.0, 60.0, 241)
    before = t < 50.0
    assert np.max(np.abs(s[before] - (0.0008 + 0.0005 * np.sin(t[before])))) < 2e-6
