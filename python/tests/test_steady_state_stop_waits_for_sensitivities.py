"""Issue #1516: run(steady_state=True) with sensitivities stops once dx/dθ has settled.

The early stop tested ``||f(t, y)||_2 / n`` alone. Y is made at ky = 1 and lost at
dy = 0.01 and starts at its steady state 100; X is made at kx*Y, lost at dx = 5,
from 0. f is small by t = 20, but dY/dky relaxes at dy, with time constant 100:
the run stopped at t = 20 with steady_state_reached = 1 and a last row of
dX/dky = 17.96 where dx_ss/dθ = kx/(dx*dy) = 100, and carry_sensitivities=True
started the next phase from it (dXo/dky at t = 2: 16.35 for 83.33). With
sensitivities the stop now also asks each column to stand still, by the same
test with θ·dS/dt in f's place.

The bug needs the check to run at output points before the slow mode settles,
so phase 1 is sampled every 10 time units (n_points = 1001 over 1e4).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 ky 1
    2 dy 0.01
    3 kx 5
    4 dx 5
    5 L  0
end parameters
begin species
    1 Y() 100
    2 X() 0
end species
begin reactions
    1 0 1   ky
    2 1 0   dy
    3 1 1,2 kx
    4 2 0   dx
    5 2 0   L
end reactions
begin groups
    1 Xo 2
end groups
"""
PHASE1 = dict(t_span=(0, 1e4), n_points=1001, steady_state=True)
DXSS_DKY = 100.0  # kx/(dx*dy)


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "m.net"
    path.write_text(NET)
    return str(path)


def _sim(net, **kw) -> bngsim.Simulator:
    return bngsim.Simulator(bngsim.Model.from_net(net), method="ode", **kw)


def _last(result, species: int, column: int) -> float:
    return float(np.asarray(result.sensitivities)[-1, species, column])


def test_the_last_row_of_a_steady_state_run_is_dx_ss_dtheta(net):
    """The stop came at t = 20 with dX/dky = 17.96."""
    r = _sim(net, sensitivity_params=["ky"]).run(**PHASE1)
    assert r.solver_stats["steady_state_reached"]
    assert _last(r, 1, 0) == pytest.approx(DXSS_DKY, rel=1e-6)
    assert float(r.time[-1]) > 20.0


def test_the_carried_seed_is_dx_ss_dtheta(net):
    """The documented pre-equilibration protocol: phase 2 switches on a loss L of X
    and carries phase 1's sensitivities. dXo/dky at t = 2 read 16.35 for 83.33."""
    m = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["ky"])
    sim.run(**PHASE1)
    m.set_param("L", 1.0)
    r = sim.run(t_span=(0, 2), n_points=3, carry_sensitivities=True)
    # X(0) = 100 relaxes from rate 5 to rate 6: X(t) = 100*(5 + exp(-6t))/6, and
    # dX/dky(t) = 100*(5 + exp(-6t))/6 likewise, Y being at its steady state.
    exact = 100.0 * (5.0 + math.exp(-12.0)) / 6.0
    dxo = np.asarray(r.output_sensitivities("observable:Xo")).reshape(len(r.time), -1)
    assert float(dxo[-1, 0]) == pytest.approx(exact, rel=1e-6)


def test_an_initial_condition_column_settles_too(net):
    """dY/dY(0) = exp(-dy t) is 0 at the steady state; the stop at t = 20 left 0.82.
    The column is held to dS/dt, the scale CVODES resolves it on."""
    r = _sim(net, sensitivity_ic=["Y()"]).run(**PHASE1)
    assert abs(float(np.asarray(r.sensitivities_ic)[-1, 0, 0])) < 1e-5


def test_an_initial_condition_column_of_a_large_amount_settles(tmp_path):
    """A0 = 1e6 decays at 0.1 to its steady state, with dA/dA0 = exp(-0.1 t) -> 0.
    Scaled by the starting amount the column had to pass 1e6*dS/dt < atol, finer
    than CVODES resolves it, and the run never stopped."""
    path = tmp_path / "a.net"
    path.write_text(
        "begin parameters\n 1 ks 1e5\n 2 kd 0.1\nend parameters\n"
        "begin species\n 1 A() 1e6\nend species\n"
        "begin reactions\n 1 0 1 ks\n 2 1 0 kd\nend reactions\n"
        "begin groups\n 1 At 1\nend groups\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", sensitivity_ic=["A()"])
    r = sim.run(t_span=(0, 1e4), n_points=1001, steady_state=True)
    assert r.solver_stats["steady_state_reached"]
    assert abs(float(np.asarray(r.sensitivities_ic)[-1, 0, 0])) < 1e-4


def test_a_slow_column_that_is_not_the_first_holds_the_stop(net):
    """kx's column settles with X, at rate 5; ky's with Y, at rate 0.01."""
    r = _sim(net, sensitivity_params=["kx", "ky"]).run(**PHASE1)
    assert _last(r, 1, 1) == pytest.approx(DXSS_DKY, rel=1e-6)


def test_a_column_is_held_to_theta_times_its_rate(tmp_path):
    """Scaled up: ky = 1e4, so Y(0) = Y_ss = 1e6, and a tolerance that suits those
    amounts. dY/dky still relaxes at dy, and its rate is 1e4 times smaller than
    θ·dS/dt. Held to dS/dt alone the column would pass at t ~ 460, 1% short of
    dY_ss/dky = 1/dy = 100."""
    path = tmp_path / "big.net"
    path.write_text(NET.replace("1 ky 1", "1 ky 1e4").replace("1 Y() 100", "1 Y() 1e6"))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["ky"]
    )
    r = sim.run(**PHASE1, steady_state_tol=1e-2)
    assert r.solver_stats["steady_state_reached"]
    assert _last(r, 0, 0) == pytest.approx(100.0, rel=1e-5)


def test_a_column_of_a_parameter_at_zero_is_held_to_its_rate(tmp_path):
    """ly = 0 is a second loss of Y. θ falls back to 1 for a zero parameter; held to
    0·dS/dt the column would pass at once. dY/dly = -Y/dy (1 - exp(-dy t)) -> -1e4,
    and the stop at t = 20 left -1813."""
    path = tmp_path / "ly.net"
    path.write_text(
        NET.replace("    5 L  0\n", "    5 L  0\n    6 ly 0\n").replace(
            "    5 2 0   L\n", "    5 2 0   L\n    6 1 0   ly\n"
        )
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["ly"]
    )
    r = sim.run(**PHASE1)
    assert r.solver_stats["steady_state_reached"]
    assert _last(r, 0, 0) == pytest.approx(-1e4, rel=1e-6)


def test_a_batch_row_stops_where_its_columns_have_settled(net):
    """run_batch rows took the same stop: the ky = 1 row read 17.96."""
    rows = _sim(net, sensitivity_params=["ky"]).run_batch(
        params=[{"ky": 1.0}, {"ky": 2.0}], **PHASE1
    )
    for row in rows:
        assert _last(row, 1, 0) == pytest.approx(DXSS_DKY, rel=1e-6)


def test_each_scan_point_carries_a_settled_seed(net):
    """A scan after the equilibration started each point from the unsettled seed:
    L = 0 read 25.77 for 100 and L = 1 27.38 for 83.33."""
    sim = _sim(net, sensitivity_params=["ky"])
    sim.run(**PHASE1)
    points = sim.parameter_scan("L", [0.0, 1.0], **PHASE1)
    for point, L in zip(points, (0.0, 1.0), strict=True):
        assert _last(point, 1, 0) == pytest.approx(5.0 / (0.01 * (5.0 + L)), rel=1e-6)


def test_without_sensitivities_the_stop_is_where_it_was(net):
    """Control: with no sensitivity columns the test is the state's alone, and the
    run stops at t = 40 as it did."""
    r = _sim(net).run(**PHASE1)
    assert r.solver_stats["steady_state_reached"]
    assert float(r.time[-1]) == 40.0


def test_a_check_at_the_end_alone_was_right(net):
    """Control: with two output points the only check is at t_end, by which time
    every column has settled; the guide's example reads this way."""
    r = _sim(net, sensitivity_params=["ky"]).run(t_span=(0, 1e4), n_points=2, steady_state=True)
    assert _last(r, 1, 0) == pytest.approx(DXSS_DKY, rel=1e-6)
