"""Issue #1285: the clock-rate column of a narrow kinked pulse.

A counter ``0 -> T`` at ``k_clock = 2``, read as ``Tobs``, and ``X -> Y`` at a
triangle of area 0.5 in T and width w, centred on c. Whatever the pulse's
shape, the exposure is 0.5/k_clock, so Y = 10(1 - exp(-0.25)) and
dY/dk_clock = -10·exp(-1/4)/8 = -0.973501.

On the analytic path that column stopped converging for w <= 0.005: 2e-5 to
9e-5 off from rtol 1e-10 on, where dY/dkx was right to 1e-9. Inside the pulse
the column runs at 1/w² times the clock's column, which is t, and it ends 4e4
times smaller than it peaks. Three things moved the state at a corner and left
the columns where they were:

* a crossing restarts δt >= 256·ε·t along the flow past its surface and keeps
  the time. δt behind the state at 5.5e7 is 8e-5 (``columns_past_surface``);
* a root CVODE places 1e-12 past a corner, after a step over the whole pulse,
  has its columns still on the branch before it (the same, where
  ``root_step_ran_past`` says the step ran on over the pulse; no test here
  pins that reading, whose effect at rtol 1e-10 is within what moves between
  platforms);
* a run carried over a corner it stalls at (issue #1229) moves the clock a few
  ulp across it (``carry_columns``).

And the run read a corner's change of slope across its probe pair as a jump,
and gave it a saltation term instead (``unlisted_bend``).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

EXACT = -10 * math.exp(-0.25) / 8  # dY/dk_clock
EXACT_KX = 10 * math.exp(-0.25) / 4  # dY/dkx

NET = """begin parameters
    1 kx 1
    2 k_clock 2.0
    3 c {c!r}
    4 w {w!r}
    5 w1 {w1!r}
    6 w2 {w2!r}
end parameters
begin functions
    1 pulse() {law}
end functions
begin species
    1 T() {t0!r}
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

LAWS = {
    "max-abs": "kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))",
    "if": (
        "kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c,(Tobs-c+w/2)/(w/2),"
        "if(Tobs<c+w/2,(c+w/2-Tobs)/(w/2),0)))"
    ),
    # Rising over w1 and falling over w2, so the corners do not cancel at the
    # peak as a symmetric triangle's do.
    "asymmetric": "kx*(1/(w1+w2))*max(0,min((Tobs-c+w1)/w1,(c+w2-Tobs)/w2))",
}


def _model(tmp_path, law: str, *, c: float, t0: float = 0.0, w: float = 0.003, w1=None, w2=None):
    w1 = w / 3 if w1 is None else w1
    w2 = 2 * w / 3 if w2 is None else w2
    path = tmp_path / f"pulse_{abs(hash((law, c, t0, w, w1, w2)))}.net"
    path.write_text(NET.format(law=law, c=c, t0=t0, w=w, w1=w1, w2=w2))
    return bngsim.Model.from_net(str(path))


def _columns(model, n_points: int, rtol: float) -> np.ndarray:
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["kx", "k_clock"]).run(
        t_span=(0.0, 30.0), n_points=n_points, rtol=rtol, atol=rtol / 100
    )
    return np.asarray(run.sensitivities)[-1, 2, :]


@pytest.mark.parametrize("n_points", [3, 31])
@pytest.mark.parametrize("law", list(LAWS))
def test_the_clock_rate_column_converges(tmp_path, law, n_points):
    """The pulse at t = 25 as in the issue, with the counter started at
    -49.95 so that it is 0.05 there and not 50: an ulp of it is then 7e-18,
    and the clock's own rounding (below) is out of the reading. On main the
    column was 1.5e-5 to 7.9e-5 off for max/abs, and 5e-4 for if(). The
    31-point grid is the one whose corner was read as a jump."""
    model = _model(tmp_path, LAWS[law], c=0.05, t0=-49.95)
    dkx, dk = _columns(model, n_points, 1e-12)
    assert dk == pytest.approx(EXACT, rel=5e-6)
    assert dkx == pytest.approx(EXACT_KX, rel=1e-8)


@pytest.mark.parametrize(
    ("law", "n_points"), [("max-abs", 3), ("max-abs", 16), ("if", 11), ("if", 16)]
)
def test_the_issues_own_pulse(tmp_path, law, n_points):
    """The issue's model, the counter at 50 across the pulse. A step there
    rounds it by up to an ulp, 7e-15, which a column moving at 5.5e7 turns
    into up to 2e-7 of itself, so the column of a pulse this narrow stays near
    1e-5 at any tolerance. So does that of a Gaussian of the same width, which
    has no corner. On these grids main was 8e-5 off for max/abs, and 7e-4 for
    if()."""
    _, dk = _columns(_model(tmp_path, LAWS[law], c=50.0), n_points, 1e-12)
    assert dk == pytest.approx(EXACT, rel=4e-5)


def test_a_carry_over_a_steep_corner_takes_the_columns_along(tmp_path):
    """A pulse that rises over 0.03 and falls over 3e-5: the run stalls at its
    foot, where the column moves at 1/w2 times its rise's rate, and is carried
    a few ulp of the clock across it. Left behind there, the column came out
    1.4e-5 to 2.9e-5 short on every grid, against a scatter of 8e-6 either way
    from the clock's rounding, so the mean over the grids is what is read."""
    errors = [
        _columns(_model(tmp_path, LAWS["asymmetric"], c=50.0, w1=0.03, w2=3e-5), n, 1e-12)[1]
        / EXACT
        - 1
        for n in (3, 4, 5, 6, 8, 11, 16, 31)
    ]
    assert abs(float(np.mean(errors))) < 8e-6, errors


# ── Issue #1454: the same restart at a jump ─────────────────────────────────
#
# A rectangle of height 0.5/w, and a jump into a slope, written with if() so
# that the rate law jumps at each edge. The crossing applies its saltation
# term and then restarts δt along the flow, as a continuous one does, and the
# columns were left δt behind the state there: 4.6e-5 at w = 0.003 for the
# rectangle and 6.1e-5 for the slope, at every tolerance.

JUMPS = {
    # Area 0.5 in T: dY/dk_clock as above.
    "rectangle": ("kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c+w/2,0.5,0))", 0.5),
    # Area 1.5 in T.
    "jump-into-slope": ("kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c+w/2,1.5+(Tobs-c)/w,0))", 1.5),
}


def _exact_clock_column(area: float, rate: float = 2.0) -> float:
    """dY/dk for Y = 10(1 - exp(-area/rate)), the exposure area/rate."""
    return -10 * math.exp(-area / rate) * area / rate**2


@pytest.mark.parametrize("n_points", [3, 4, 5, 6, 8, 11, 16, 31])
@pytest.mark.parametrize("law", list(JUMPS))
def test_a_jump_moves_the_columns_with_the_state(tmp_path, law, n_points):
    """The counter at 0.05 across the pulse, as above. Main was 4e-6 to 6e-5
    off on every grid; the branch is within 1.1e-7. The 8-point grid of the
    rectangle is also the one whose crossing re-take stopped on the far edge
    itself (see the next test)."""
    text, area = JUMPS[law]
    _, dk = _columns(_model(tmp_path, text, c=0.05, t0=-49.95), n_points, 1e-12)
    assert dk == pytest.approx(_exact_clock_column(area), rel=1e-6)


def test_a_re_take_does_not_stop_on_the_crossing(tmp_path):
    """A re-take of the step a crossing was found in integrates again from
    the step's start to a stop short of the crossing. At the end of a step of
    5e-14 that crept up on a jump, the stop's shortfall was below an ulp of t:
    it stopped on the crossing itself, past the surface, where CVODE never
    reports it, and the edge's jump in the columns was lost. dY/dk_clock came
    out 1.6e4 for -0.97, with no error. A stop not on the near side is not
    taken, and the crossing is restarted at as any other is."""
    text, area = JUMPS["rectangle"]
    for n_points in (8, 9, 10, 12):
        dkx, dk = _columns(_model(tmp_path, text, c=0.05, t0=-49.95), n_points, 1e-12)
        assert dk == pytest.approx(_exact_clock_column(area), rel=1e-6), n_points
        assert dkx == pytest.approx(EXACT_KX, rel=1e-8), n_points


# ── Issue #1455: a condition that reads the time ─────────────────────────────
#
# The triangle in u = Tobs + time(), the counter started at -74.925 so that u
# crosses the pulse at t = 25, where it is small. The exposure is
# 0.5/(k_clock + 1). The restart past a crossing moves the state δt along the
# flow and keeps the time, and where the rate law reads the time that restart
# moves with the crossing time, which moves with the columns. Left out, three
# grids in seven were 3.3e-4 off at rtol 1e-12.

TIME_LAW = "kx*(1/w)*max(0,1-abs(Tobs+time()-c)/(w/2))"
EXACT_TIME = -10 * math.exp(-1 / 6) * 0.5 / 9


@pytest.mark.parametrize("n_points", [3, 4, 5, 6, 8, 11, 16])
def test_a_condition_that_reads_the_time(tmp_path, n_points):
    """At t = 25 an ulp of the time is 3.6e-15, and a column moving at 5e7
    turns the clock's rounding into a few 1e-6 of itself, as the counter's
    does at T = 50 above: up to 3e-5 on grids and pulse positions near these.
    Main was 3.3e-4 off on the 4-, 8- and 16-point grids."""
    dkx, dk = _columns(_model(tmp_path, TIME_LAW, c=0.075, t0=-74.925), n_points, 1e-12)
    assert dk == pytest.approx(EXACT_TIME, rel=6e-5)
    assert dkx == pytest.approx(10 * math.exp(-1 / 6) / 6, rel=1e-8)


@pytest.mark.parametrize("w", [0.003, 0.03])
def test_a_condition_that_reads_the_time_near_t0(tmp_path, w):
    """The same pulse at t = 0.25, where the time rounds by 3e-17 and the
    restart's δt is small: right to 1e-8 on main and on the branch, which is
    what tells the clock's rounding above from an error in the columns."""
    for n_points in (3, 4, 8, 16):
        _, dk = _columns(_model(tmp_path, TIME_LAW, c=0.075, t0=-0.675, w=w), n_points, 1e-12)
        assert dk == pytest.approx(EXACT_TIME, rel=2e-8), n_points


# ── Issue #1456: a root CVODE finds inside a step past a corner ──────────────
#
# CVODE finds a corner's root inside the step that went past it, and reads
# the columns there from that step. Across the corner the column's rate jumps,
# and the step's corrector takes the rate past the corner for the whole step:
# a step of 1.1e-11 that ended 1.1e-13 past a peak passed its error test with
# the column 5.9e-4 off. The columns at the root are now integrated again over
# that step from its start (columns_over_root_step).


@pytest.mark.parametrize("law", ["max-abs", "if"])
def test_the_triangle_at_a_loose_tolerance(tmp_path, law):
    """The issue's triangle, w = 0.1 at c = 50 and rtol 1e-8. Main was 1e-4
    off on these grids, its worst over 38; every grid is now within 1.6e-5."""
    for n_points in (5, 6, 18, 21, 28, 30):
        _, dk = _columns(_model(tmp_path, LAWS[law], c=50.0, w=0.1), n_points, 1e-8)
        assert dk == pytest.approx(EXACT, rel=4e-5), n_points


def test_the_issues_traced_grid(tmp_path):
    """w = 0.005, rtol 1e-9: main 3.7e-4 off on the 9-point grid; every grid
    is now near 5.2e-5, the same on each."""
    for n_points in (9, 16, 23):
        _, dk = _columns(_model(tmp_path, LAWS["if"], c=50.0, w=0.005), n_points, 1e-9)
        assert dk == pytest.approx(EXACT, rel=1.2e-4), n_points


RELAX_U = "(Aobs+a*time())"
RELAX_LAW = (
    f"kx*(1/w)*if({RELAX_U}<c-w/2,0,if({RELAX_U}<c,({RELAX_U}-c+w/2)/(w/2),"
    f"if({RELAX_U}<c+w/2,(c+w/2-{RELAX_U})/(w/2),0)))"
)
RELAX_NET = """begin parameters
    1 kx 1.0
    2 ka 2.0
    3 kd 0.4
    4 c 2.5
    5 w 0.0005
    6 a 0.0
    7 kf 1e4
    8 kr 1e4
    9 A0 0.0
end parameters
begin functions
    1 law() {law}
end functions
begin species
    1 A() A0
    2 X() 10
    3 Y() 0
    4 P() 1e7
    5 Q() 1e7
end species
begin reactions
    1 0 1 ka
    2 1 0 kd
    3 2 3 law
    4 4 5 kf
    5 5 4 kr
end reactions
begin groups
    1 Aobs 1
    2 Pobs 4
    3 Qobs 5
end groups
"""
# E = ∫ p(a) / (ka - kd·a) da over the triangle, to second order in w.
EXACT_RELAX = 1.21306141756


def test_a_triangle_in_a_relaxing_species(tmp_path):
    """A triangle 5e-4 wide in A, which relaxes towards 5, beside a stiff
    pair of 1e7 pools that sets the steps (``a = 0``: the law reads the time
    with a factor 0, which leaves its value and moves the steps it takes as
    the issue's do). At rtol 1e-11 main was 3.8e-4 off
    on the 37-point grid, against a median of 8e-7 over grids 3 to 40: the
    step that found the peak's root ended 1.1e-13 past it. Every grid is now
    within 3e-6. Which grid lands so is a matter of the steps: an atol of
    1e-13 rather than 1e-11 * 1e-2, an ulp apart, misses it on main, and
    another platform's arithmetic may too. The two tests above are the ones
    that pin it everywhere."""
    path = tmp_path / "relax.net"
    path.write_text(RELAX_NET.format(law=RELAX_LAW))
    params = ["kx", "ka", "kd", "c"]
    for n_points in (35, 36, 37, 38, 40):
        run = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=params
        ).run(t_span=(0.0, 20.0), n_points=n_points, rtol=1e-11, atol=1e-11 * 1e-2)
        dc = float(np.asarray(run.sensitivities)[-1, 2, params.index("c")])
        assert dc == pytest.approx(EXACT_RELAX, rel=2e-5), n_points


# A model beside the pulse: a species W fed through a second switch.
SIDE_NET = """begin parameters
    1 kx 1
    2 k_clock 2.0
    3 kz 0.3
    4 c {c!r}
    5 w {w!r}
end parameters
begin functions
    1 pulse() {law}
    2 side() {side}
end functions
begin species
    1 T() {t0!r}
    2 X() 10
    3 Y() 0
    4 Z() {z0!r}
    5 W() 0
end species
begin reactions
    1 0 1 k_clock
    2 2 3 pulse
    3 4 5 side
end reactions
begin groups
    1 Tobs 1
    2 Zobs 4
end groups
"""


def _side_model(tmp_path, law, side, *, c, t0, w=0.003, z0=7.0):
    path = tmp_path / f"side_{abs(hash((law, side, c, t0, w, z0)))}.net"
    path.write_text(SIDE_NET.format(law=law, side=side, c=c, t0=t0, w=w, z0=z0))
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("side", ["kz*if(time()>5,1,0)", "kz*if(time()<5,1,0)"])
@pytest.mark.parametrize("j", [-8, -6, -5])
def test_the_clocks_share_is_not_read_across_a_clock_switch(tmp_path, side, j):
    """A crossing a step or two of δt short of a switch on the time alone,
    which W reads and the counter never reaches: dW/dk_clock is 0. Reading
    the clock's share past t* + δt took that switch's jump in, -2.1 or +1.2
    (review of this change; main is 0)."""
    dt = 256 * 2.220446049250313e-16 * 5
    law = "kx*if(Tobs>c,Tobs-c,0)"
    model = _side_model(tmp_path, law, side, c=2 * (5 + j * 0.25 * dt), t0=0.0)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["kx", "k_clock", "kz"]).run(
        t_span=(0.0, 8.0), n_points=4, rtol=1e-9, atol=1e-11
    )
    w_col = np.asarray(run.sensitivities)[-1, 4, 1]
    assert abs(w_col) < 1e-9


@pytest.mark.parametrize("n_points", [4, 8, 16])
def test_a_switch_parked_at_zero_beside_the_pulse(tmp_path, n_points):
    """A species at 0 under ``if(Zobs>0, ...)``: its residual is 0 at every
    crossing, which read as changing sides and left the clock's share out of
    the whole run, 3.3e-4 off as on main."""
    law = (
        "kx*(1/w)*if(Tobs+time()<c-w/2,0,if(Tobs+time()<c,(Tobs+time()-c+w/2)/(w/2),"
        "if(Tobs+time()<c+w/2,(c+w/2-Tobs-time())/(w/2),0)))"
    )
    model = _side_model(tmp_path, law, "0.1*if(Zobs>0,Zobs,0)", c=0.075, t0=-74.925, z0=0.0)
    _, dk = _columns(model, n_points, 1e-12)
    assert dk == pytest.approx(EXACT_TIME, rel=6e-5)
