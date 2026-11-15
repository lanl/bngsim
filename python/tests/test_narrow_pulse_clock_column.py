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
    does at T = 50 above: the branch reads 5.5e-6 at worst there."""
    dkx, dk = _columns(_model(tmp_path, TIME_LAW, c=0.075, t0=-74.925), n_points, 1e-12)
    assert dk == pytest.approx(EXACT_TIME, rel=3e-5)
    assert dkx == pytest.approx(10 * math.exp(-1 / 6) / 6, rel=1e-8)


@pytest.mark.parametrize("w", [0.003, 0.03])
def test_a_condition_that_reads_the_time_near_t0(tmp_path, w):
    """The same pulse at t = 0.25, where the time rounds by 3e-17 and the
    restart's δt is small: right to 1e-8 on main and on the branch, which is
    what tells the clock's rounding above from an error in the columns."""
    for n_points in (3, 4, 8, 16):
        _, dk = _columns(_model(tmp_path, TIME_LAW, c=0.075, t0=-0.675, w=w), n_points, 1e-12)
        assert dk == pytest.approx(EXACT_TIME, rel=2e-8), n_points


# ── Issue #1456: a root CVODE places inside a step past a corner ──────────────
#
# Filed with the column at the peak's root 8.6e-3 off as CVODE interpolated
# it, after a step that ran past the corner, and dY/dk_clock 1e-4 to 1e-2 off
# on some grids. Measured against the exact column at every peak crossing of
# 456 runs (both spellings, w = 0.005 and 0.1, rtol 1e-8 to 1e-10, 38 grids
# each), the interpolated column is right to 1.3e-7 at worst, and what is left
# of the final error falls with the tolerance: the column falls from 2e4 at
# the peak to 1 at the end, so it carries rtol times its peak. These pin that.


@pytest.mark.parametrize("law", ["max-abs", "if"])
def test_the_triangle_converges_with_the_tolerance(tmp_path, law):
    """The issue's triangle, w = 0.1 at c = 50. At rtol 1e-8 its worst over
    38 grids was 1.1e-4 to 1.4e-4; at 1e-10 it is 1.5e-6 and at 1e-11
    1.8e-7, on the grids that were worst at 1e-8."""
    for n_points in (5, 6, 9, 16, 23, 26, 30, 35):
        _, dk = _columns(_model(tmp_path, LAWS[law], c=50.0, w=0.1), n_points, 1e-11)
        assert dk == pytest.approx(EXACT, rel=2e-6), n_points


def test_the_issues_traced_grid(tmp_path):
    """w = 0.005, rtol 1e-9, 16 points: 7.9e-3 off when filed, 1.5e-4 now,
    of the size every grid has there (median 7e-5, worst 3.7e-4 over 38)."""
    _, dk = _columns(_model(tmp_path, LAWS["if"], c=50.0, w=0.005), 16, 1e-9)
    assert dk == pytest.approx(EXACT, rel=1e-3)
