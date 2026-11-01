"""Issue #1286: a smooth narrow bump of the clock in a rate law is not stepped over.

X -> Y at a Gaussian of area 0.5 in a counter T fed at rate 2, centred at T =
50 (t = 25): the exposure is 0.25, so Y(30) = 10·(1 - e^(-1/4)) = 2.211992
whatever the width. Away from the bump the rate is exactly 0 in doubles
(exp(-10^4) underflows), CVODE's error estimate is 0, and its step grew until
one spanned the bump: Y(30) came back 0 at every tolerance, on dense output and
in sensitivity runs, with no word. A stop at the bump's centre now lands the
step on its peak.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest
from bngsim._smooth_pulses import smooth_pulse_stops

EXACT = 10.0 * (1.0 - math.exp(-0.25))
GAUSS = "kx*0.5/(s*sqrt(_pi))*exp(-((CLK-c)/s)^2)"
SECH2_COSH = "kx*0.25/s/cosh((CLK-c)/s)^2"
SECH2_TANH = "kx*0.25/s*(1-tanh((CLK-c)/s)^2)"


@pytest.fixture
def model(tmp_path):
    def build(
        law: str,
        s: float,
        *,
        clock: str = "Tobs",
        c: float = 50.0,
        feed: str = "k_clock",
        width: str = "s",
    ) -> bngsim.Model:
        path = tmp_path / "pulse.net"
        path.write_text(
            f"begin parameters\n 1 kx 1\n 2 k_clock 2\n 3 c {c!r}\n 4 {width} {s!r}\n"
            "end parameters\n"
            "begin species\n 1 T() 0\n 2 X() 10\n 3 Y() 0\nend species\n"
            f"begin functions\n 1 f() {law.replace('CLK', clock)}\nend functions\n"
            f"begin reactions\n 1 0 1 {feed}\n 2 2 3 f #_FUNC\nend reactions\n"
            "begin groups\n 1 Tobs 1\nend groups\n"
        )
        return bngsim.Model.from_net(str(path))

    return build


def _y_end(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(m, method="ode").run(t_span=(0, 30), **kw)
    return float(np.asarray(r.species)[-1, 2])


@pytest.mark.parametrize("law", [GAUSS, SECH2_COSH, SECH2_TANH], ids=["gauss", "sech2", "1-tanh2"])
@pytest.mark.parametrize("s", [0.005, 0.5])
@pytest.mark.parametrize("rtol", [1e-6, 1e-12])
def test_a_bump_in_a_counter_is_integrated(model, law, s, rtol):
    """On main, Y(30) = 0 in every one of these."""
    y = _y_end(model(law, s), n_points=3, rtol=rtol, atol=rtol * 1e-2)
    assert y == pytest.approx(EXACT, rel=max(1e2 * rtol, 1e-8))


def test_a_bump_in_literal_time_is_integrated(model):
    """The same bump in time(), centred at t = 25 (width 0.25 in time)."""
    m = model("kx*0.25/(s*sqrt(_pi))*exp(-((CLK-c)/s)^2)", 0.25, clock="time()", c=25.0)
    assert _y_end(m, n_points=3, rtol=1e-9, atol=1e-11) == pytest.approx(EXACT, rel=1e-6)


def test_dense_output_does_not_step_over_it_either(model):
    """CV_NORMAL interpolates output points and does not bound the step."""
    assert _y_end(model(GAUSS, 0.005), n_points=301, rtol=1e-9, atol=1e-11) == pytest.approx(
        EXACT, rel=1e-6
    )


def test_a_sensitivity_run_integrates_the_bump(model):
    """Y(30) = 10(1 - e^(-kx/4)), so dY/dkx = 2.5·e^(-kx/4) = 1.947 at kx = 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(model(GAUSS, 0.5), method="ode", sensitivity_params=["kx"])
        r = sim.run(t_span=(0, 30), n_points=3, rtol=1e-9, atol=1e-11)
    assert float(np.asarray(r.species)[-1, 2]) == pytest.approx(EXACT, rel=1e-6)
    assert float(np.asarray(r.sensitivities)[-1, 2, 0]) == pytest.approx(
        2.5 * math.exp(-0.25), rel=1e-5
    )


def test_the_stop_is_at_the_centre(model):
    """Centre T = 50 on a counter fed at rate 2 from 0: t = 25."""
    stops = smooth_pulse_stops(model(GAUSS, 0.5)._core, 0.0, 30.0)
    assert [s.time for s in stops] == pytest.approx([25.0])


def test_a_stop_outside_the_run_is_not_placed(model):
    assert smooth_pulse_stops(model(GAUSS, 0.5)._core, 0.0, 20.0) == []


@pytest.mark.parametrize(
    "law",
    [
        pytest.param("kx*0.5/(s*_pi)/(1+((CLK-c)/s)^2)", id="lorentzian (never underflows)"),
        pytest.param("kx*exp(-0.1*CLK)", id="an exponential decay of the clock"),
        pytest.param("kx*exp(-((Xo-c)/s)^2)", id="a bump of the state, not the clock"),
    ],
)
def test_no_stop_where_there_is_no_bump_of_a_clock(tmp_path, law):
    """Control: nothing placed, so the run steps as it did."""
    path = tmp_path / "none.net"
    path.write_text(
        "begin parameters\n 1 kx 1\n 2 k_clock 2\n 3 c 50\n 4 s 0.5\nend parameters\n"
        "begin species\n 1 T() 0\n 2 X() 10\n 3 Y() 0\nend species\n"
        f"begin functions\n 1 f() {law.replace('CLK', 'Tobs')}\nend functions\n"
        "begin reactions\n 1 0 1 k_clock\n 2 2 3 f #_FUNC\nend reactions\n"
        "begin groups\n 1 Tobs 1\n 2 Xo 2\nend groups\n"
    )
    assert smooth_pulse_stops(bngsim.Model.from_net(str(path))._core, 0.0, 30.0) == []


def test_models_alike_but_for_the_feed_do_not_share_stops(model):
    """The same functions, values and state, with the counter fed at 1 (kx) in
    place of 2: the centre T = 50 is at t = 50. Asked after the model fed at 2,
    it was handed that model's stop at t = 25, and stepped over its own bump."""
    assert [s.time for s in smooth_pulse_stops(model(GAUSS, 0.05)._core, 0.0, 60.0)] == [25.0]
    fed_at_1 = model(GAUSS, 0.05, feed="kx")
    assert [s.time for s in smooth_pulse_stops(fed_at_1._core, 0.0, 60.0)] == [50.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(fed_at_1, method="ode").run(
            t_span=(0, 60), n_points=3, rtol=1e-9, atol=1e-11
        )
    # Area 0.5 in T, fed at 1: the exposure is 0.5.
    assert float(np.asarray(r.species)[-1, 2]) == pytest.approx(
        10.0 * (1.0 - math.exp(-0.5)), rel=1e-6
    )


def test_a_width_named_for_a_python_keyword(model):
    """`lambda` is parsed under an alias (GH #108), and was left in the argument
    as a second symbol: no stop, and Y(30) = 0."""
    law = "kx*0.5/(lambda*sqrt(_pi))*exp(-((CLK-c)/lambda)^2)"
    m = model(law, 0.005, width="lambda")
    assert [s.time for s in smooth_pulse_stops(m._core, 0.0, 30.0)] == [25.0]
    assert _y_end(model(law, 0.005, width="lambda"), n_points=3, rtol=1e-9, atol=1e-11) == (
        pytest.approx(EXACT, rel=1e-6)
    )


def test_e_to_a_power_is_a_gaussian(model):
    """`_e^q` is parsed as exp(q), and was not read for a bump: Y(30) = 0."""
    law = "kx*0.5/(s*sqrt(_pi))*_e^(-((CLK-c)/s)^2)"
    assert _y_end(model(law, 0.005), n_points=3, rtol=1e-9, atol=1e-11) == pytest.approx(
        EXACT, rel=1e-6
    )


def test_a_width_of_zero_costs_no_other_bump_its_stop(tmp_path):
    """A width of 0 at a scan point puts zoo in its argument. It raised out of
    the whole model's stops, on every run, and the other bump's went with it."""
    path = tmp_path / "zero.net"
    path.write_text(
        "begin parameters\n 1 kx 1\n 2 k_clock 2\n 3 c 50\n 4 s 0.5\n 5 s0 0\n"
        "end parameters\n"
        "begin species\n 1 T() 0\n 2 X() 10\n 3 Y() 0\nend species\n"
        "begin functions\n 1 f() kx*exp(-((Tobs-c)/s)^2)\n"
        " 2 g() kx*exp(-((Tobs-20)/s0)^2)\nend functions\n"
        "begin reactions\n 1 0 1 k_clock\n 2 2 3 f #_FUNC\n 3 2 3 g #_FUNC\nend reactions\n"
        "begin groups\n 1 Tobs 1\nend groups\n"
    )
    core = bngsim.Model.from_net(str(path))._core
    assert [s.time for s in smooth_pulse_stops(core, 0.0, 30.0)] == [25.0]


def test_a_bump_centred_past_the_end_of_the_run(model):
    """Centred at t = 30.05, a width (0.05 in time) past the end: a fifth of
    its area is inside the run. With no stop there, the last step was taken
    over the rise and the peak, and Y(30) came back 0."""
    stops = smooth_pulse_stops(model(GAUSS, 0.1, c=60.1)._core, 0.0, 30.0)
    assert [s.time for s in stops] == [30.0]
    reference = _y_end(model(GAUSS, 0.1, c=60.1), n_points=3, rtol=1e-9, atol=1e-11, max_step=1e-3)
    assert reference == pytest.approx(10.0 * (1.0 - math.exp(-0.25 * 0.5 * math.erfc(1.0))))
    got = _y_end(model(GAUSS, 0.1, c=60.1), n_points=3, rtol=1e-9, atol=1e-11)
    assert got == pytest.approx(reference, rel=1e-6)


def test_a_counter_in_substance_units_is_read_as_its_amount():
    """In a compartment of size 2, a substance-units counter fed at 2 has a
    stored value moving at 1 and an amount, which the rate law reads, moving
    at 2. Read at the stored rate, the stop came at t = 50 for a bump at 25."""
    pytest.importorskip("antimony")
    src = """
      compartment C = 2
      substanceOnly species T in C; T = 0
      species X in C = 10; species Y in C = 0
      kx = 1; c = 50; s = 0.05; k_clock = 2
      J0: -> T; k_clock
      J1: X -> Y; kx*0.5/(s*sqrt(pi))*exp(-((T-c)/s)^2)*X
    """
    m = bngsim.Model.from_antimony_string(src)
    assert [s.time for s in smooth_pulse_stops(m._core, 0.0, 30.0)] == pytest.approx([25.0])
    y = list(m._core.species_names).index("Y")

    def y_end(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode").run(
                t_span=(0, 30), n_points=3, rtol=1e-9, atol=1e-11, **kw
            )
        return float(np.asarray(r.species)[-1, y])

    reference = y_end(max_step=2e-3)
    assert reference > 1.0
    assert y_end() == pytest.approx(reference, rel=1e-6)


def test_a_bump_elsewhere_keeps_the_refusal_of_an_event_on_two_switches():
    """An event moved by `tau` fires at t = 5, where a rate law reading X and
    one that does not switch: the first does not commute with the event, so
    the sensitivity does not exist there (#767). The probes that ask both are
    handed over where an instant has more than one crossing, and the bump's
    stop at t = 20, counted as one more switch instant, dropped them: the run
    returned a dX/dtau in place of the refusal."""
    pytest.importorskip("antimony")
    m = bngsim.Model.from_antimony_string(
        """
        species T = 0; species X = 10; species Y = 0; species Z = 0
        tau = 5; k1 = 1; k2 = 2; kz = 1; cc = 20; s = 0.05
        J0: -> T; 1
        J1: X -> Y; piecewise(k1, time > 5, 0)*X
        J2: -> Y; piecewise(k2, T > 5, 0)
        J3: -> Z; kz*exp(-((time-cc)/s)^2)
        E1: at time > tau: X = X + 5
        """
    )
    assert [s.time for s in smooth_pulse_stops(m._core, 0.0, 30.0)] == [20.0]
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["tau"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(bngsim.SimulationError, match="767"):
            sim.run(t_span=(0, 30), n_points=4, rtol=1e-10, atol=1e-12)


def test_e_to_a_power_written_as_pow_is_a_gaussian(model):
    """`pow(_e, q)` is parsed as exp(q), and was not read for a bump."""
    law = "kx*0.5/(s*sqrt(_pi))*pow(_e, -((CLK-c)/s)^2)"
    assert _y_end(model(law, 0.005), n_points=3, rtol=1e-9, atol=1e-11) == pytest.approx(
        EXACT, rel=1e-6
    )


def test_a_bump_on_a_switch_keeps_the_switch_s_stop(tmp_path):
    """A Gaussian centred on `time() >= 10`. At rtol 1e-2 the switch's stop is
    the double before its flip (#1226); merged as the later of two on literal
    time, the bump's stop at 10 took its place, and Y(20) was 10.0005."""
    path = tmp_path / "on_switch.net"
    path.write_text(
        "begin parameters\n 1 k 1\nend parameters\n"
        "begin species\n 1 Y() 0\n 2 Z() 0\nend species\n"
        "begin functions\n 1 f() if(time()>=10,k,0)\n"
        " 2 g() k*exp(-((time()-10)/0.3)^2)\nend functions\n"
        "begin reactions\n 1 0 1 f #_FUNC\n 2 0 2 g #_FUNC\nend reactions\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
            t_span=(0, 20), n_points=3, rtol=1e-2, atol=1e-4
        )
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(10.0, rel=1e-12)


def test_a_counter_an_event_resets_is_stopped_on_once_and_says_so():
    """T fed at 2 and reset to 0 at 100: its bump at 50 comes at t = 25, 75 and
    125. The stop is placed from T's value at the start, so only the first is
    stopped on; the warning says the passes after a reset are not."""
    pytest.importorskip("antimony")
    m = bngsim.Model.from_antimony_string(
        """
        species T = 0; species X = 10; species Y = 0
        kx = 1; c = 50; s = 0.05; k_clock = 2
        J0: -> T; k_clock
        J1: X -> Y; kx*0.5/(s*sqrt(pi))*exp(-((T-c)/s)^2)*X
        E1: at T >= 100: T = 0
        """
    )
    with pytest.warns(UserWarning, match="an event assigns T"):
        stops = smooth_pulse_stops(m._core, 0.0, 40.0)
    assert [s.time for s in stops] == [25.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(m, method="ode").run(
            t_span=(0, 40), n_points=3, rtol=1e-9, atol=1e-11
        )
    y = list(m._core.species_names).index("Y")
    assert float(np.asarray(r.species)[-1, y]) == pytest.approx(EXACT, rel=1e-6)
