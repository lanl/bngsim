"""A sensitivity run is carried over a bend it stalls at, not refused (issue #1229).

The issue #952 hold check ends a run with sensitivities whose state is held
within a few ulp of a state-switch surface, because at a jump there is a
saltation term to apply and no crossing time to take it from. It asked that
of every surface, and a bend has neither: where a triangle pulse returns to 0
the rate law is continuous, and only its slope changes, by ~1/w². At a tight
tolerance that slope change fails the error test on the step to it, and the
run was refused for a jump that does not exist. The same pulse written with
``if()`` was refused the same way.

A stalled surface is now read a step either side along the flow (the #150
probe), and one the right-hand side is continuous across, or that no rate law
reads (a bend stop), is carried over as a plain run's is, the columns
unchanged. A stalled surface it jumps across is refused as before; see
``test_pinned_state_switch_sensitivity.py``.

The model is ``test_bend_sensitivity_rhs.py``'s: a counter T at k_clock = 2, and
X -> Y at a triangle of area 0.5 in T, of width w, centred on 50. Exact
dY/dk_clock = -0.973501 and Y = 10(1 - e^(-1/4)).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

EXACT = -10 * 0.25 * math.exp(-0.25) / 2  # -0.973501
Y_EXACT = 10 * (1 - math.exp(-0.25))
LAWS = {
    "max/abs": "kx*(1/w)*max(0,1-abs(Tobs-c)/(w/2))",
    "if()": (
        "kx*(1/w)*if(Tobs<c-w/2,0,if(Tobs<c,(Tobs-c+w/2)/(w/2),"
        "if(Tobs<c+w/2,(c+w/2-Tobs)/(w/2),0)))"
    ),
}


def _run(tmp_path, law, w, rtol):
    path = tmp_path / "m.net"
    path.write_text(
        f"begin parameters\n 1 kx 1\n 2 k_clock 2\n 3 c 50\n 4 w {w!r}\nend parameters\n"
        "begin species\n 1 T() 0\n 2 X() 10\n 3 Y() 0\nend species\n"
        f"begin functions\n 1 f() {law}\nend functions\n"
        "begin reactions\n 1 0 1 k_clock\n 2 2 3 f #_FUNC\nend reactions\n"
        "begin groups\n 1 Tobs 1\nend groups\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["kx", "k_clock"]
    )
    run = sim.run(t_span=(0, 30), n_points=3, rtol=rtol, atol=rtol / 100)
    return float(np.asarray(run.species)[-1, 2]), float(np.asarray(run.sensitivities)[-1, 2, 1])


# The cells the issue lists as refused on main.
@pytest.mark.parametrize("spelling", list(LAWS))
@pytest.mark.parametrize(
    ("w", "rtol", "rel"),
    [(0.01, 1e-10, 2e-5), (0.01, 1e-11, 1e-5), (0.03, 1e-11, 1e-6)],
)
def test_a_run_that_stalls_at_a_bend_returns_the_exact_column(tmp_path, spelling, w, rtol, rel):
    y, dy = _run(tmp_path, LAWS[spelling], w, rtol)
    assert y == pytest.approx(Y_EXACT, rel=1e-8)
    assert dy == pytest.approx(EXACT, rel=rel)


@pytest.mark.parametrize("spelling", list(LAWS))
@pytest.mark.parametrize("rtol", [1e-9, 1e-10, 1e-11])
def test_the_narrowest_pulse_runs_too(tmp_path, spelling, rtol):
    """w = 0.003 was refused at every tolerance. Its column is now within what
    the analytic path gives a pulse that narrow where main runs it (9e-5 at w =
    0.005, rtol 1e-9; 9e-4 at w = 0.003, rtol 1e-8), not to the tolerance."""
    y, dy = _run(tmp_path, LAWS[spelling], 0.003, rtol)
    assert y == pytest.approx(Y_EXACT, rel=1e-8)
    assert dy == pytest.approx(EXACT, rel=5e-4)
