"""Issue #1776: a crossing the clock carries against the state.

A counter ``0 -> T`` at ``k_clock = 2``, read as ``Tobs``, and ``X -> Y`` at a
rate that switches on ``u = a*time() - b*Tobs``, which runs at
r = a - b·k_clock. With b > 0 the clock carries u across each threshold while
Tobs, moving along its own flow, carries it back. A sensitivity run of such a
pulse stalled at its near foot at every tolerance (the plain run was right):

* the restart past a crossing keeps the time and moves the state δt along its
  flow. That lowered u, and put the state back on the side it came from, a
  hair short of a corner whose root had been taken;
* the run carried over a corner it stalls at (issue #1229) only where the
  species' part of the residual's flow points the way the whole of it does,
  so the far foot, which no step resolves, was never carried;
* once both were mended, the clock-rate column of the restart that moves the
  state to the far side counted the residual's lag behind the clock twice,
  4e-4 off at rtol 1e-10 to 1e-11 on some output grids.

Whatever the pulse's shape, the exposure is kx·(area)/r, so the columns at
t_end have a closed form. The rows with b <= 0 are controls: the species carry
those crossings, and nothing here changes them.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 kx 1
    2 k_clock 2.0
    3 c {c!r}
    4 w 0.003
end parameters
begin functions
    1 pulse() {law}
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 pulse
end reactions
begin groups
    1 Tobs 1
end groups
"""

K_CLOCK = 2.0
T_END = 30.0


def _u(a: float, b: float) -> str:
    return f"{a!r}*time()-{b!r}*Tobs"


def _triangle(a: float, b: float) -> str:
    return f"kx*(1/w)*max(0,1-abs({_u(a, b)}-c)/(w/2))"


def _step(a: float, b: float) -> str:
    return f"kx*if({_u(a, b)}>c,1,0)"


def _sawtooth(a: float, b: float) -> str:
    # A jump up at c, then down to 0 over w: a jump the restart takes, and a
    # far corner, continuous but read by the rate law, that the run stalls at.
    u = _u(a, b)
    return f"kx*(2/w)*if({u}>c,if({u}<c+w,1-({u}-c)/w,0),0)"


def _model(tmp_path, law: str, c: float) -> bngsim.Model:
    path = tmp_path / f"m_{abs(hash((law, c)))}.net"
    path.write_text(NET.format(law=law, c=c))
    return bngsim.Model.from_net(str(path))


def _exact(shape: str, a: float, b: float, c: float) -> tuple[float, float, float]:
    """Y, dY/dkx and dY/dk_clock at T_END, with kx = 1."""
    r = a - b * K_CLOCK
    if shape == "triangle":
        exposure, d_kx, d_k = 1.0 / (2 * r), 1.0 / (2 * r), b / (2 * r * r)
    elif shape == "sawtooth":
        exposure, d_kx, d_k = 1.0 / r, 1.0 / r, b / (r * r)
    else:
        exposure = T_END - c / r
        d_kx, d_k = exposure, -c * b / (r * r)
    x = 10 * math.exp(-exposure)
    return 10 - x, x * d_kx, x * d_k


def _run(model: bngsim.Model, rtol: float, n_points: int):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kx", "k_clock"])
    res = sim.run(t_span=(0.0, T_END), n_points=n_points, rtol=rtol, atol=rtol * 1e-2, timeout=60)
    s = np.asarray(res.sensitivities)
    y = res.species_names.index("Y()")
    return float(res.species[-1][y]), float(s[-1, y, 0]), float(s[-1, y, 1])


def test_the_issues_reproducer_runs_and_is_right(tmp_path):
    """The issue's model, `3*time() - Tobs`, stalled at t = 24.9985."""
    a, b = 3.0, 1.0
    c = 25.0 * (a - b * K_CLOCK)
    model = _model(tmp_path, _triangle(a, b), c)
    got = _run(model, 1e-8, 4)
    want = _exact("triangle", a, b, c)
    assert got[0] == pytest.approx(want[0], rel=1e-7)
    assert got[1] == pytest.approx(want[1], rel=1e-6)
    # As close as the same pulse with the clock behind it comes at 1e-8.
    assert got[2] == pytest.approx(want[2], rel=1e-2)


# (a, b): b > 0 has the state run against the clock. The pulse sits at t = 25.
AGAINST = [(3.0, 1.0), (2.5, 0.5), (2.0, 0.25), (3.5, 1.0), (5.5, 2.0)]
WITH = [(1.0, -1.0), (0.5, -0.5)]


@pytest.mark.parametrize("n_points", [4, 7, 31])
@pytest.mark.parametrize("a,b", AGAINST + WITH)
def test_a_triangle_column_converges(tmp_path, a, b, n_points):
    """At rtol 1e-11 every row comes within 6e-6 of the closed form on these
    grids (macOS arm64). Before the kept-time reading was mended, rows against
    the clock came out up to 4.2e-4 off on some of them."""
    c = 25.0 * (a - b * K_CLOCK)
    model = _model(tmp_path, _triangle(a, b), c)
    got = _run(model, 1e-11, n_points)
    want = _exact("triangle", a, b, c)
    assert got[0] == pytest.approx(want[0], rel=1e-9)
    assert got[1] == pytest.approx(want[1], rel=1e-7)
    assert got[2] == pytest.approx(want[2], rel=6e-5)


@pytest.mark.parametrize("rtol", [1e-6, 1e-8, 1e-10])
@pytest.mark.parametrize("a,b", [(3.0, 1.0), (2.5, 0.5), (1.0, -1.0)])
def test_a_step_the_clock_carries_is_right(tmp_path, a, b, rtol):
    """The same crossing as a jump in the rate. Against the clock it ran past
    the 60 s timeout on main, at every tolerance."""
    c = 25.0 * (a - b * K_CLOCK)
    model = _model(tmp_path, _step(a, b), c)
    got = _run(model, rtol, 4)
    want = _exact("step", a, b, c)
    for g, w in zip(got, want, strict=False):
        assert g == pytest.approx(w, rel=200 * rtol)


@pytest.mark.parametrize("n_points", [4, 7, 31])
@pytest.mark.parametrize("a,b", [(3.0, 1.0), (2.5, 0.5), (5.5, 2.0)])
def test_a_sawtooth_the_clock_carries_is_right(tmp_path, a, b, n_points):
    """Against the clock it ran to the timeout on main. Its far corner is
    reached by the stall carry, whose probe for a jump reads both sides of the
    surface. Within 2.3e-5 of the closed form at rtol 1e-10 to 1e-12 on these
    grids (macOS arm64), as close as the same sawtooth with the clock behind
    the state comes; with the restart's kept-time term read at the moved state
    on the jump path, up to 2.5e-4."""
    c = 25.0 * (a - b * K_CLOCK)
    want = _exact("sawtooth", a, b, c)
    for rtol in (1e-10, 1e-11, 1e-12):
        got = _run(_model(tmp_path, _sawtooth(a, b), c), rtol, n_points)
        assert got[0] == pytest.approx(want[0], rel=1e-8)
        assert got[1] == pytest.approx(want[1], rel=1e-6)
        assert got[2] == pytest.approx(want[2], rel=6e-5)


def test_the_plain_run_is_unchanged(tmp_path):
    """A run without sensitivities was right before, and is the same."""
    a, b = 3.0, 1.0
    c = 25.0 * (a - b * K_CLOCK)
    model = _model(tmp_path, _triangle(a, b), c)
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, T_END), n_points=4, rtol=1e-8, atol=1e-10, timeout=60
    )
    y = res.species_names.index("Y()")
    assert float(res.species[-1][y]) == pytest.approx(_exact("triangle", a, b, c)[0], rel=1e-7)
