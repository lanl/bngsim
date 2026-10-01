"""A state that slides along a rate law's switching surface (issue #926).

``f() = if(Sobs < 1, amp, -amp)`` on ``0 -> S``: S rises at amp until S = 1 at
t = 1/amp, and then both branches point into the surface. S stays at 1, on
neither branch, so dS/damp = 0 from there on.

A forward-sensitivity run had nothing right to apply. At a tight tolerance the
root at S = 1 was found and the jump from one branch to the other applied, over
and over, until the wall clock ended the run. At a loose one no root was ever
reported: the state crept along just short of the surface, inside the tolerance
of it, and the column integrated the near branch's ∂f/∂amp = 1 all the way:
dS/damp = t, 1.5 for 0 at t = 1.5, with no warning.

The slide is now refused, at the root where the flow on the far side points
back, and at the first batch of steps spent beside the surface where no root
is reported. The trajectory, without sensitivities, is as it was.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 amp {amp!r}
    2 lvl 1.0
    3 k 0.5
end parameters
begin functions
    1 f() {law}
end functions
begin species
    1 S() {s0!r}
    2 Y() 0
end species
begin reactions
    1 0 1 f
    2 0 2 k
end reactions
begin groups
    1 Sobs 1
end groups
"""
FROM_BELOW = "if(Sobs<lvl,amp,-amp)"
FROM_ABOVE = "if(Sobs>lvl,-amp,amp)"


def _model(tmp_path, law, s0=0.0, amp=1.0):
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=law, s0=s0, amp=amp))
    return bngsim.Model.from_net(path)


@pytest.mark.parametrize(
    ("rtol", "atol"), [(1e-6, 1e-6), (1e-6, 1e-8), (1e-8, 1e-10)], ids=["loose", "mid", "tight"]
)
@pytest.mark.parametrize(
    ("law", "s0"), [(FROM_BELOW, 0.0), (FROM_ABOVE, 2.0)], ids=["from-below", "from-above"]
)
def test_a_slide_is_refused(tmp_path, law, s0, rtol, atol):
    """dS/damp came back 1.5 for 0 at the two looser tolerances, and the run
    timed out at the tight one."""
    sim = bngsim.Simulator(_model(tmp_path, law, s0), method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="slides along the switching surface.*issue #926"):
        sim.run(sample_times=[0.0, 0.5, 1.5], rtol=rtol, atol=atol, timeout=10.0)


def test_a_column_that_does_not_move_the_slide_is_refused_too(tmp_path):
    """`lvl` is where S is held, so dS/dlvl = 1 on the slide. The column cannot
    be told from the one for `amp` at the surface, and the run is refused
    whichever is requested."""
    sim = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["lvl"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=[0.0, 0.5, 1.5], rtol=1e-6, atol=1e-8, timeout=10.0)


def test_a_run_that_ends_before_the_slide(tmp_path):
    """Control. Up to t = 0.9 the state has not reached the surface:
    dS/damp = t."""
    run = bngsim.Simulator(
        _model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["amp"]
    ).run(sample_times=[0.0, 0.5, 0.9], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(run.sensitivities)[:, 0, 0], [0.0, 0.5, 0.9], rtol=1e-8)


def test_the_trajectory_of_a_slide_without_sensitivities(tmp_path):
    """Control. S is held at 1 from t = 1 on."""
    run = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode").run(
        sample_times=[0.0, 0.5, 1.5, 3.0], rtol=1e-6, atol=1e-8, timeout=60.0
    )
    np.testing.assert_allclose(np.asarray(run.species)[:, 0], [0.0, 0.5, 1.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("after", [0.25, 3.0], ids=["slower", "faster"])
def test_a_crossing_that_carries_on_is_not_a_slide(tmp_path, after):
    """Control. ``if(S < 1, amp, after·amp)``: the slope changes at S = 1 and
    the state goes on through. With t* = 1/amp, S = 1 + after·amp·(t − t*), so
    dS/damp = after·t past the crossing."""
    law = f"if(Sobs<lvl,amp,{after!r}*amp)"
    run = bngsim.Simulator(_model(tmp_path, law), method="ode", sensitivity_params=["amp"]).run(
        sample_times=[0.0, 0.5, 1.5, 2.0], rtol=1e-8, atol=1e-10
    )
    got = np.asarray(run.sensitivities)[:, 0, 0]
    np.testing.assert_allclose(got, [0.0, 0.5, after * 1.5, after * 2.0], rtol=1e-6)
