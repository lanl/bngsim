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
    def build(law: str, s: float, *, clock: str = "Tobs", c: float = 50.0) -> bngsim.Model:
        path = tmp_path / "pulse.net"
        path.write_text(
            f"begin parameters\n 1 kx 1\n 2 k_clock 2\n 3 c {c!r}\n 4 s {s!r}\nend parameters\n"
            "begin species\n 1 T() 0\n 2 X() 10\n 3 Y() 0\nend species\n"
            f"begin functions\n 1 f() {law.replace('CLK', clock)}\nend functions\n"
            "begin reactions\n 1 0 1 k_clock\n 2 2 3 f #_FUNC\nend reactions\n"
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
