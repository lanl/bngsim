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
