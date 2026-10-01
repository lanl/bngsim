"""The closing edge of a window, in the column that moves it (issue #760).

A window ``k1·s·(1-s)^(a-1)`` with ``s = (t-on)/D`` closes at ``on + D`` as a
power. The forcing of the D column goes as ``(1-s)^(a-2)`` before the crossing,
which no step resolves for a < 2. A comoving column ``V = S + c·f`` has no such
term (issue #545), but a column entered its frame at the crossing it moves,
after the approach had been integrated: dX/dD came back 0.37% low at a = 1.1,
flat in the tolerance, and the run raised at a = 1.05.

The column for ``on`` was right all along. It enters at the opening edge, which
it moves at the same rate as the closing one, and is still in its frame when
the window closes.

A column whose next switch time is approached through such a power now enters
before it: at the start of the run, or at a stop the run takes halfway from the
clock crossing behind it. Not at that crossing. A window that opens as a power
too, ``s^(a-1)·(1-s)^(a-1)``, has an f whose slope is unbounded just past the
opening.

X is linear in the pulse, so every expected value is a quadrature of the pulse
against ``e^(-kdeg·(T-t))``, differenced in the parameter at two steps and
extrapolated.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from scipy.integrate import quad

ON, WIDTH, K1, KDEG = 3.0, 4.0, 2.0, 0.3
# 5 is the stop the D column enters at. Nothing is sampled on the onset (3) or
# the close (7): the window's own edge is a kink in D there.
T = [0.0, 1.0, 2.0, 4.0, 5.0, 5.5, 6.5, 6.99, 7.01, 8.0, 10.0]

NET = """begin parameters
    1 k0    0.1
    2 k1    2.0
    3 a     {a}
    4 on    3.0
    5 D     4.0
    6 kdeg  0.3
    7 _rateLaw1 1
    8 tmid  {tmid}
end parameters
begin functions
    1 s() (t-on)/D
    2 prod() k0+if(t>=on,if(t{close}(on+D),k1*{shape},0),0){extra}
end functions
begin species
    1 X() 0
    2 Tc() 0
end species
begin reactions
    1 0 1 prod
    2 1 0 kdeg
    3 0 2 _rateLaw1
end reactions
begin groups
    1 t 2
end groups
"""

SHAPES = {
    "closing": ("s()*((1-s())^(a-1))", lambda u, a: u * (1 - u) ** (a - 1)),
    "both": ("(s()^(a-1))*((1-s())^(a-1))", lambda u, a: u ** (a - 1) * (1 - u) ** (a - 1)),
    "opening": ("(s()^(a-1))*(1-s())", lambda u, a: u ** (a - 1) * (1 - u)),
}


def _model(tmp_path, shape, a, close="<=", extra="", tmid=5.0):
    text = NET.format(a=a, close=close, shape=SHAPES[shape][0], extra=extra, tmid=tmid)
    path = tmp_path / "m.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def _exact(shape, a, param):
    """dX(t)/dparam at every sample time."""
    pulse = SHAPES[shape][1]

    def x_of(t_end, on, width):
        hi = min(on + width, t_end)
        if hi <= on:
            return 0.0
        value, _err = quad(
            lambda u: K1 * pulse(u, a) * np.exp(-KDEG * (t_end - on - width * u)) * width,
            0.0,
            (hi - on) / width,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=400,
        )
        return value

    def moved(t_end, by):
        return x_of(t_end, ON + by, WIDTH) if param == "on" else x_of(t_end, ON, WIDTH + by)

    out = []
    for t_end in T:
        h = 1e-4
        coarse = (moved(t_end, h) - moved(t_end, -h)) / (2 * h)
        fine = (moved(t_end, h / 2) - moved(t_end, -h / 2)) / h
        out.append((4 * fine - coarse) / 3)
    return np.array(out)


def _column(model, param, rtol=1e-8, atol=1e-10):
    run = bngsim.Simulator(model, method="ode", sensitivity_params=[param]).run(
        sample_times=T, rtol=rtol, atol=atol
    )
    return np.asarray(run.sensitivities)[:, 0, 0]


def _worst(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.mark.parametrize("close", ["<=", "<"])
@pytest.mark.parametrize("a", [1.05, 1.1, 1.2, 1.3])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_the_column_that_moves_the_closing_edge(tmp_path, shape, a, close):
    """dX/dD at every sample: one on the stop the column enters at (5), and one
    0.01 either side of the close. It was 0.5% off at a = 1.05 and 1.1e-5 at
    a = 1.3, at rtol 1e-8, and the closing shape raised at a = 1.05."""
    got = _column(_model(tmp_path, shape, a, close), "D")
    assert _worst(got, _exact(shape, a, "D")) < 5e-6


@pytest.mark.parametrize("a", [1.05, 1.1, 1.2, 1.3])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_the_same_column_at_a_looser_tolerance(tmp_path, shape, a):
    """At rtol 1e-6 the column is as good as the onset column, 3e-5. It was
    1.2e-4 to 7e-3."""
    got = _column(_model(tmp_path, shape, a), "D", rtol=1e-6, atol=1e-8)
    assert _worst(got, _exact(shape, a, "D")) < 6e-5


@pytest.mark.parametrize("tmid", [4.0, 5.0, 6.9])
@pytest.mark.parametrize("shape", ["closing", "both"])
def test_a_crossing_inside_the_window_asks_again(tmp_path, shape, tmid):
    """A fixed crossing at tmid, inside the window, restarts the run and leaves
    every frame. The column asks for a new stop halfway from there to the close.
    At 5 the crossing is on the stop the run had planned; at 6.9 the new stop is
    0.05 before the close."""
    extra = "+if(t>=tmid,0.0,0.0)*0"
    got = _column(_model(tmp_path, shape, 1.1, extra=extra, tmid=tmid), "D")
    assert _worst(got, _exact(shape, 1.1, "D")) < 5e-6


@pytest.mark.parametrize("a", [1.1, 1.5])
@pytest.mark.parametrize("shape", ["closing", "both", "opening"])
def test_the_onset_column_is_as_it_was(tmp_path, shape, a):
    """Control. The onset moves both edges at 1, so its column was in its frame
    across the whole window already. It enters ahead now at the start of the
    run, where the first sample has to read 0 and not c·f."""
    got = _column(_model(tmp_path, shape, a), "on")
    assert got[0] == 0.0
    assert _worst(got, _exact(shape, a, "on")) < 5e-6


@pytest.mark.parametrize("a", [1.1, 1.5])
def test_a_window_that_only_opens_as_a_power_is_as_it_was(tmp_path, a):
    """Control. ``s^(a-1)·(1-s)`` closes smoothly: nothing enters ahead for D,
    whose column at the opening edge has no shift at all."""
    got = _column(_model(tmp_path, "opening", a), "D")
    assert _worst(got, _exact("opening", a, "D")) < 5e-6


def test_the_generator_marks_the_case_a_closing_edge_approaches(tmp_path):
    """The D column has a case only where the edge it moves is singular, and
    that case is marked as one to enter ahead of."""
    from bngsim import _codegen

    def source(shape):
        core = _model(tmp_path, shape, 1.1)._core
        return _codegen.generate_sens_from_model(core, functional=True, emit_term_scale=True)

    def approach(src):
        body = src.split("int bngsim_codegen_comoving_approach(int case_idx)")[1].split("\n}")[0]
        return body.count("return 1;")

    assert "bngsim_codegen_comoving_approach" in source("closing")
    assert approach(source("closing")) == 2  # on and D
    assert approach(source("both")) == 2
    assert approach(source("opening")) == 0
