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
t_end have a closed form. The rows with b < 0 are controls: the species carry
those crossings with the clock, and their columns are as on main (their state
can differ from main's in the last bits, where the stall carry's reach now
counts the time's ulp).
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


def _exact(
    shape: str, a: float, b: float, c: float, k_clock: float = K_CLOCK
) -> tuple[float, float, float]:
    """Y, dY/dkx and dY/dk_clock at T_END, with kx = 1."""
    r = a - b * k_clock
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
    assert got[0] == pytest.approx(want[0], rel=1e-8)
    assert got[1] == pytest.approx(want[1], rel=1e-7)
    assert got[2] == pytest.approx(want[2], rel=6e-5)


@pytest.mark.parametrize("shape", ["triangle", "step"])
@pytest.mark.parametrize("a,b", [(3.0, 1e-3), (3.0, 1e-6), (1.0, -1e-4), (1.0, -1e-10)])
def test_a_clock_that_outruns_its_species_leaves_the_state_alone(tmp_path, shape, a, b):
    """Where the clock carries the residual tens of thousands of times faster
    than its species, the restart is not moved: the move would be δt·∂g/∂t
    over a gradient of 1e-10, and moved without a bound, T(30) came out
    1.3e-2 off at b = -1e-10, where the run was right as it was. Main and this
    branch give T(30) = 60 to 3e-11 on these rows."""
    c = 25.0 * (a - b * K_CLOCK)
    law = _triangle(a, b) if shape == "triangle" else _step(a, b)
    sim = bngsim.Simulator(
        _model(tmp_path, law, c), method="ode", sensitivity_params=["kx", "k_clock"]
    )
    res = sim.run(t_span=(0.0, T_END), n_points=4, rtol=1e-9, atol=1e-11, timeout=60)
    t = res.species_names.index("T()")
    assert float(res.species[-1][t]) == pytest.approx(K_CLOCK * T_END, rel=1e-11)


@pytest.mark.parametrize("a,b", [(3.0, 1e-4), (3.0, 1e-5), (3.0, 1e-6), (1.0, -1e-5)])
def test_a_stall_carry_never_pushes_a_weakly_coupled_species_far(tmp_path, a, b):
    """The stall carry's reach counts the time's ulp only up to 16 times the
    species' own, as the restart's move is bounded. Counted without that cap,
    at rtol 1e-12 these rows returned T(30) off by up to 3.6e-7, where main
    stalls with "CVODE made no progress". A stall is still what such a run may
    do (issue #1779); a run that returns has the clock species right."""
    c = 25.0 * (a - b * K_CLOCK)
    sim = bngsim.Simulator(
        _model(tmp_path, _triangle(a, b), c), method="ode", sensitivity_params=["kx", "k_clock"]
    )
    try:
        res = sim.run(t_span=(0.0, T_END), n_points=4, rtol=1e-12, atol=1e-14, timeout=60)
    except bngsim.SimulationError:
        return
    t = res.species_names.index("T()")
    assert float(res.species[-1][t]) == pytest.approx(K_CLOCK * T_END, rel=1e-11)


Z_NET = """begin parameters
    1 kx 1
    2 k_clock 2.0
    3 c {c!r}
    4 w 0.003
    5 kz 0.1
end parameters
begin functions
    1 pulse() {law}
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
    4 {z}
end species
begin reactions
    1 0 1 k_clock
    2 2 3 pulse
    3 4 0 kz
end reactions
begin groups
    1 Tobs 1
    2 Zobs 4
end groups
"""


@pytest.mark.parametrize("z,zval", [("$Z() 5", 5.0), ("Z() 0", 0.0)])
@pytest.mark.parametrize(
    "law",
    [
        "kx*if(3*time()-Tobs-Zobs>c,1,0)",
        "kx*(1/w)*max(0,1-abs(3*time()-Tobs-Zobs-c)/(w/2))",
    ],
)
def test_the_move_leaves_a_species_the_flow_does_not_move(tmp_path, z, zval, law):
    """The restart's move takes up the clock's share with the species the flow
    moves, and with no other: a fixed species that a reaction consumes, or one
    at rest at 0, read by the residual stays where it is. Moved with the rest, $Z ended at
    4.999999999997868 and Z at -2.1e-12."""
    path = tmp_path / "z.net"
    # The pulse sits at t = 25: 3t - 2t - Z = c.
    path.write_text(Z_NET.format(c=25.0 - zval, law=law, z=z))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["kx", "k_clock"]
    )
    res = sim.run(t_span=(0.0, T_END), n_points=4, rtol=1e-9, atol=1e-11, timeout=60)
    zi = res.species_names.index("Z()")
    assert float(res.species[-1][zi]) == zval
    # The switch fired: a test whose pulse lay past T_END read Z untouched.
    assert float(res.species[-1][res.species_names.index("Y()")]) > 1.0


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


@pytest.mark.parametrize("rtol", [1e-10, 1e-11])
@pytest.mark.parametrize("b", [0.1, 0.12])
def test_a_weakly_coupled_sawtooth_is_carried_over_its_far_corner(tmp_path, b, rtol):
    """At `3*time() - 0.1*Tobs` the far corner sat 1.4e-14 short of its
    surface, an ulp of the time's 3·t, where the stall carry's reach, 16 ulp
    of Tobs at 0.1 a unit, was 1.1e-14: the run stalled there at rtol 1e-10 and
    1e-11. The reach now counts the time's ulp too."""
    a = 3.0
    c = 25.0 * (a - b * K_CLOCK)
    got = _run(_model(tmp_path, _sawtooth(a, b), c), rtol, 4)
    want = _exact("sawtooth", a, b, c)
    assert got[0] == pytest.approx(want[0], rel=1e-8)
    assert got[1] == pytest.approx(want[1], rel=1e-6)
    assert got[2] == pytest.approx(want[2], rel=3e-4)


@pytest.mark.parametrize("k_clock,b", [(1.2, 0.17), (0.6, 0.34)])
def test_the_stall_reach_and_the_restart_share_one_bound(tmp_path, k_clock, b):
    """The clock outruns Tobs 14.7 times here, inside the restart's bound of
    16. Bounded by the ratio of ulps instead, which is that rate times
    k_clock·ulp(t)/ulp(T), 17.6 here, the stall carry's reach left these out,
    and the sawtooth stalled at its far corner at rtol 1e-10."""
    a = 3.0
    c = 25.0 * (a - b * k_clock)
    path = tmp_path / "k.net"
    path.write_text(
        NET.format(law=_sawtooth(a, b), c=c).replace(
            "    2 k_clock 2.0\n", f"    2 k_clock {k_clock!r}\n"
        )
    )
    got = _run(bngsim.Model.from_net(str(path)), 1e-10, 4)
    want = _exact("sawtooth", a, b, c, k_clock)
    assert got[0] == pytest.approx(want[0], rel=1e-8)
    assert got[1] == pytest.approx(want[1], rel=1e-6)
    assert got[2] == pytest.approx(want[2], rel=3e-4)


MULTI_NET = """begin parameters
    1 kx 1
    2 k_clock 2.0
    3 w 0.003
    4 kw {kw!r}
    5 kd {kd!r}
end parameters
begin functions
    1 pulse() {law}
end functions
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
    4 W() 0
    5 Z() {z0!r}
end species
begin reactions
    1 0 1 k_clock
    2 2 3 pulse
    3 0 4 kw
    4 5 0 kd
end reactions
begin groups
    1 Tobs 1
    2 Wobs 4
    3 Zobs 5
end groups
"""

MULTI = {
    # A second species W the residual reads, rising 400 times slower than T.
    "step-W": ("step", "3*time()-Tobs-Wobs", 25 * 0.995, dict(kw=5e-3, kd=0.0, z0=0.0)),
    "triangle-W": ("triangle", "3*time()-Tobs-Wobs", 25 * 0.995, dict(kw=5e-3, kd=0.0, z0=0.0)),
    # Z decaying to nearly 0 by the crossing, and slowly.
    "step-Z-fast": ("step", "3*time()-Tobs-Zobs", 25.0, dict(kw=0.0, kd=1.0, z0=1.0)),
    "triangle-Z-slow": (
        "triangle",
        "3*time()-Tobs-Zobs",
        25.0 - math.exp(-0.25),
        dict(kw=0.0, kd=0.01, z0=1.0),
    ),
    "step-half-W": (
        "step",
        "3*time()-0.5*Tobs-0.5*Wobs",
        25 * 1.9975,
        dict(kw=5e-3, kd=0.0, z0=0.0),
    ),
}


@pytest.mark.parametrize("case", sorted(MULTI))
def test_a_residual_reading_a_slow_second_species(tmp_path, case):
    """The move is shared out by each species' own flow, so a second species
    the residual reads steeply and the flow moves slowly takes little of it.
    Shared by the gradient alone, that species took most of the move, the
    bound refused it, and all five stalled or timed out, as on main. The
    columns are checked against central differences of plain runs, which were
    right on main."""
    shape, u, c, kv = MULTI[case]
    law = (
        f"kx*(1/w)*max(0,1-abs({u}-{c!r})/(w/2))"
        if shape == "triangle"
        else f"kx*if({u}>{c!r},1,0)"
    )

    def model(kx=1.0, k_clock=K_CLOCK):
        text = MULTI_NET.format(law=law, **kv)
        text = text.replace("    1 kx 1\n", f"    1 kx {kx!r}\n").replace(
            "    2 k_clock 2.0\n", f"    2 k_clock {k_clock!r}\n"
        )
        path = tmp_path / f"multi_{abs(hash((kx, k_clock)))}.net"
        path.write_text(text)
        return bngsim.Model.from_net(str(path))

    def y_plain(**kw):
        res = bngsim.Simulator(model(**kw), method="ode").run(
            t_span=(0.0, T_END), n_points=4, rtol=1e-12, atol=1e-14, timeout=60
        )
        return float(res.species[-1][res.species_names.index("Y()")])

    h = 1e-5
    fd = (
        (y_plain(kx=1 + h) - y_plain(kx=1 - h)) / (2 * h),
        (y_plain(k_clock=K_CLOCK + h) - y_plain(k_clock=K_CLOCK - h)) / (2 * h),
    )
    got = _run(model(), 1e-10, 4)
    assert got[1] == pytest.approx(fd[0], rel=1e-5)
    assert got[2] == pytest.approx(fd[1], rel=1e-4)


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
