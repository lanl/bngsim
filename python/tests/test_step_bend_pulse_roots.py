"""Issue #1068: a pulse written with step or bend calls over a counter that is not a clock.

``100*(floor((Tobs-50)/100) - floor((Tobs-50.01)/100))`` is a pulse of width
0.01 in T at T = 50, written with no comparison, and so is the triangle
``100*max(0, 1 - abs(Tobs-50.005)/0.005)``. With T counting at 1 it is a
clock, and #869/#904 stop at each edge. At any other rate T is state: no stop
could be placed, no comparison was there to root on, the rate law is flat on
both sides of the pulse, and one step spanned it. Y came back 0 with no error.

Each call is now rooted where it jumps or bends: a rounding call where
``sin(pi*u)`` changes sign, a bend where its arguments meet, and an ``abs()``
or ``min()`` inside such a turn unfolded into its branches, which cross at the
pulse's feet.

T = k*t, so the pulse lasts 0.01/k and Y = 10(1 - exp(-kx/k)) for the
rectangle, 10(1 - exp(-kx/(2k))) for the triangle.

A run with sensitivities roots the bends as stops: it jumps nothing there, as
the pulse is continuous, and without them the triangle gave Y = 0 and dY/dkx
= 0 beside a plain run that was right.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _find_close_paren_strict
from bngsim._switch_sensitivity import _bend_branches, _turn_atoms, state_switch_root_conditions

NET = """begin parameters
    1 k_clock {k}
    2 kx 1
end parameters
begin functions
    1 pulse() kx*({law})
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

# (rate law, its area in T)
PULSES = {
    "floor": ("100*(floor((Tobs-50)/100)-floor((Tobs-50.01)/100))", 1.0),
    "ceil": ("100*(ceil((Tobs-50)/100)-ceil((Tobs-50.01)/100))", 1.0),
    "sign": ("50*(sign(Tobs-50)-sign(Tobs-50.01))", 1.0),
    "inrange": ("100*inrange(50,Tobs,50.01)", 1.0),
    "max-abs triangle": ("100*max(0,1-abs(Tobs-50.005)/0.005)", 0.5),
    "min-max triangle": ("100*max(0,min((Tobs-50)/0.005,(50.01-Tobs)/0.005))", 0.5),
    "mod": ("mod(Tobs+49.99,100)-mod(Tobs+50,100)+0.01", 1.0),
    "round": ("100*(round((Tobs-50)/100+0.5)-round((Tobs-50.01)/100+0.5))", 1.0),
    "trunc": ("100*(trunc((Tobs-50)/100+1)-trunc((Tobs-50.01)/100+1))", 1.0),
    "clamp triangle": ("100*(clamp(0,(Tobs-50)/0.01,1)-clamp(0,(Tobs-50.01)/0.01,1))", 1.0),
    # x outside [50, 50.01], a bound inside it: a jump at 50.005 and bends at both ends.
    "iclamp": ("1e6*(iclamp(50,Tobs,50.01)-Tobs)^2", 2e6 * 0.005**3 / 3),
}
# The pulses made of bends alone, which a run with sensitivities accepts. Since
# issue #1126 min, max and abs have an analytic sensitivity RHS; clamp is still
# on CVODES' difference quotient.
BENDS = ["max-abs triangle", "min-max triangle", "clamp triangle"]


def _model(tmp_path, law: str, k: float) -> bngsim.Model:
    path = tmp_path / "pulse.net"
    path.write_text(NET.format(k=k, law=law))
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("k", [2.0, 0.5])
@pytest.mark.parametrize("name", list(PULSES))
def test_the_pulse_is_not_stepped_over(tmp_path, name, k):
    law, area = PULSES[name]
    run = bngsim.Simulator(_model(tmp_path, law, k), method="ode").run(
        t_span=(0.0, 60.0 / k), n_points=3
    )
    y = float(np.asarray(run.species)[-1, 2])
    assert y == pytest.approx(10 * (1 - np.exp(-area / k)), rel=1e-5)


@pytest.mark.parametrize("name", ["floor", "max-abs triangle"])
def test_a_unit_rate_clock_keeps_its_stops_and_gets_no_roots(tmp_path, name):
    """At k = 1 the stops of #869/#904 cover every jump and bend, the
    abs() inside the max() included; a root there would be a second restart."""
    model = _model(tmp_path, PULSES[name][0], 1.0)
    assert model.time_discontinuity_conditions()
    assert model.state_switch_root_conditions() == ()


def test_the_turns_of_each_call():
    pi = repr(np.pi)
    assert _turn_atoms("floor(u)") == [f"sin({pi}*(u))<0"]
    assert _turn_atoms("round(u)") == [f"cos({pi}*(u))<0"]
    assert _turn_atoms("mod(a, b)") == [f"sin({pi}*(a)/(b))<0"]
    assert _turn_atoms("sgn(u)") == ["(u)<0"]
    assert _turn_atoms("clamp(lo,x,hi)") == ["(x)<(lo)", "(x)<(hi)"]
    assert _turn_atoms("iclamp(lo,x,hi)") == ["(x)<(lo)", "(x)<(hi)", "(x)<((lo)+(hi))/2"]
    assert _turn_atoms("max(a,b,c)") == ["(a)<(b)", "(a)<(c)", "(b)<(c)"]
    assert _turn_atoms("floor(a,b)") == []


def test_a_bend_in_a_turn_unfolds_into_its_branches():
    assert _bend_branches("0<1-abs(T-5)/w") == ["0<1-(T-5)/w", "0<1-(-(T-5))/w"]
    assert _bend_branches("(0)<(min(p, q))") == ["(0)<((p))", "(0)<((q))"]
    assert len(_bend_branches("abs(a)<abs(b)")) == 4
    assert _bend_branches("abs(a)<max(b,c,d)") == []  # six branches
    assert _bend_branches("a<b") == []
    assert _bend_branches("abs(abs(a)-1)<0.1") == []


@pytest.mark.parametrize("k", [2.0, 0.5])
@pytest.mark.parametrize("name", BENDS)
def test_a_sensitivity_run_does_not_step_over_a_bend_pulse(tmp_path, name, k):
    """The plain run's roots for the bends are this run's stops; before, the
    triangle gave Y = 0 and dY/dkx = 0 here while the plain run was right."""
    law, area = PULSES[name]
    run = bngsim.Simulator(_model(tmp_path, law, k), method="ode", sensitivity_params=["kx"]).run(
        t_span=(0.0, 60.0 / k), n_points=3
    )
    y = float(np.asarray(run.species)[-1, 2])
    dy = float(np.asarray(run.sensitivities)[-1, 2, 0])
    assert y == pytest.approx(10 * (1 - np.exp(-area / k)), rel=1e-5)
    assert dy == pytest.approx(10 * (area / k) * np.exp(-area / k), rel=1e-4)


@pytest.mark.parametrize("name", ["max-abs triangle", "min-max triangle"])
def test_the_clock_rate_column_of_a_bend_pulse(tmp_path, name):
    """Issue #1126: a bend has an analytic sensitivity RHS. On the difference
    quotient this column was -1.18 for -0.97 at the default tolerances and
    -5.97 at rtol 1e-6, and did not converge."""
    law, area = PULSES[name]
    run = bngsim.Simulator(
        _model(tmp_path, law, 2.0), method="ode", sensitivity_params=["kx", "k_clock"]
    ).run(t_span=(0.0, 30.0), n_points=3, rtol=1e-8, atol=1e-10)
    dy = float(np.asarray(run.sensitivities)[-1, 2, 1])
    assert dy == pytest.approx(-10 * (area / 4) * np.exp(-area / 2), rel=1e-3)


def test_the_clock_rate_column_of_a_clamp_pulse(tmp_path):
    """Issue #1227: clamp() is a bend with an analytic sensitivity RHS too. On
    the difference quotient this column was -1.53 for -1.52 at rtol 1e-8, 7% off
    at the defaults, and did not converge."""
    law, area = PULSES["clamp triangle"]
    run = bngsim.Simulator(
        _model(tmp_path, law, 2.0), method="ode", sensitivity_params=["kx", "k_clock"]
    ).run(t_span=(0.0, 30.0), n_points=3, rtol=1e-8, atol=1e-10)
    dy = float(np.asarray(run.sensitivities)[-1, 2, 1])
    assert dy == pytest.approx(-10 * (area / 4) * np.exp(-area / 2), rel=1e-3)


@pytest.mark.parametrize("sens", [None, ["kx"]])
def test_a_ramp_rooted_where_it_bends_keeps_its_answer(tmp_path, sens):
    """``max(0, T - 50)`` bends without making a pulse, and its root at T = 50
    must leave the run as right as it was."""
    model = _model(tmp_path, "0.001*max(0,Tobs-50)", 2.0)
    assert model.state_switch_root_conditions() == ("(0)<(Tobs-50)",)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=sens).run(
        t_span=(0.0, 30.0), n_points=3, rtol=1e-10, atol=1e-12
    )
    # T = 2t reaches 60, so the integral of the rate over t is 0.001*10^2/(2*2).
    a = 0.001 * 10**2 / 4
    assert float(np.asarray(run.species)[-1, 2]) == pytest.approx(10 * (1 - np.exp(-a)), rel=1e-8)
    if sens:
        dy = float(np.asarray(run.sensitivities)[-1, 2, 0])
        assert dy == pytest.approx(10 * a * np.exp(-a), rel=1e-6)


def test_calls_only_leaves_out_the_comparisons(tmp_path):
    model = _model(tmp_path, "if(Tobs>55,max(0,Tobs-50),0)", 2.0)
    covered = model.time_discontinuity_conditions()
    assert state_switch_root_conditions(model._core, covered) == ("Tobs>55", "(0)<(Tobs-50)")
    assert state_switch_root_conditions(model._core, covered, calls_only=True) == (
        "(0)<(Tobs-50)",
    )


def _close_paren_by_char(expr: str, open_pos: int) -> int:
    depth = 1
    for i in range(open_pos + 1, len(expr)):
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def test_the_close_paren_scan_matches_a_scan_by_character():
    rng = np.random.default_rng(1068)
    for _ in range(2000):
        expr = "".join(rng.choice(list("(()))ab+"), size=int(rng.integers(1, 40))))
        for pos in [i for i, c in enumerate(expr) if c == "("]:
            assert _find_close_paren_strict(expr, pos) == _close_paren_by_char(expr, pos)
