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
back, and where no root is reported: a run that carries sensitivities through
a state switch is taken in batches of 50 steps and asked after each. The
trajectory, without sensitivities, is as it was.
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
    """From below dS/damp came back 1.0 and 1.5 for 0 at the two looser
    tolerances, from above −1.0 and 0.5, and the run timed out at the tight
    one."""
    sim = bngsim.Simulator(_model(tmp_path, law, s0), method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="slides along the switching surface.*issue #926"):
        sim.run(sample_times=[0.0, 0.5, 1.5], rtol=rtol, atol=atol, timeout=10.0)


@pytest.mark.parametrize(
    ("rtol", "n_points"), [(1e-4, 101), (1e-5, 1001), (1e-6, 10001)], ids=["101", "1001", "10001"]
)
def test_a_slide_between_the_points_of_a_dense_grid_is_refused(tmp_path, rtol, n_points):
    """The same slide reported on a grid: no interval between two points uses a
    batch of 10000 steps up, which was the only place a first cut of this
    asked, and dS/damp came back 1.5 for 0 at each of these."""
    sim = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(t_span=(0.0, 1.5), n_points=n_points, rtol=rtol, atol=1e-2 * rtol, timeout=30.0)


def test_a_slide_under_a_batch_of_a_million_steps_is_refused(tmp_path):
    """``max_steps`` is the batch the caller asks for, and the run is asked
    about its switches after 50 steps whatever it is."""
    sim = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=[0.0, 0.5, 1.5], rtol=1e-6, atol=1e-8, max_steps=1000000, timeout=30)


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


@pytest.mark.parametrize("max_steps", [1, 2, 20])
def test_a_state_on_its_way_to_the_surface_is_not_a_slide(tmp_path, max_steps):
    """Control. In batches of a step or two the run is asked about a slide while
    S is still on its way to the surface it will slide along: at 0.5, and at
    0.97, with the surface at 1. At the surface the two flows already point
    into it. It is a slide only once the state is inside the tolerance's band
    of it."""
    run = bngsim.Simulator(
        _model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["amp"]
    ).run(sample_times=[0.0, 0.5, 0.97], rtol=1e-8, atol=1e-10, max_steps=max_steps)
    np.testing.assert_allclose(np.asarray(run.sensitivities)[:, 0, 0], [0.0, 0.5, 0.97], rtol=1e-8)


def test_the_trajectory_of_a_slide_without_sensitivities(tmp_path):
    """Control. S is held at 1 from t = 1 on."""
    run = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode").run(
        sample_times=[0.0, 0.5, 1.5, 3.0], rtol=1e-6, atol=1e-8, timeout=60.0
    )
    np.testing.assert_allclose(np.asarray(run.species)[:, 0], [0.0, 0.5, 1.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("max_steps", [1, 2])
def test_a_crossing_into_a_branch_a_million_times_faster(tmp_path, max_steps):
    """Control. ``if(S < 1, amp, 1e6·amp)``: past the crossing the steps are a
    millionth as long, and in batches of a step or two the run is asked while
    they are. dS/damp = 1e6·t there."""
    law = "if(Sobs<lvl,amp,1e6*amp)"
    run = bngsim.Simulator(_model(tmp_path, law), method="ode", sensitivity_params=["amp"]).run(
        sample_times=[0.0, 0.5, 1.0000005], rtol=1e-8, atol=1e-10, max_steps=max_steps
    )
    assert np.asarray(run.sensitivities)[-1, 0, 0] == pytest.approx(1e6 * 1.0000005, rel=1e-6)


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


@pytest.mark.parametrize("steeper", [1, 3], ids=["one-slope", "three-times-steeper-past-it"])
@pytest.mark.parametrize("max_steps", [None, 20, 1])
def test_a_state_that_comes_to_rest_on_a_continuous_switch_is_not_a_slide(
    tmp_path, max_steps, steeper
):
    """Control. ``if(S < lvl, k·(lvl − S), −k·(S − lvl))`` is one field written
    as two branches: S relaxes to lvl and rests on the switch, with a flow that
    points in from both sides and runs out at the surface. Nothing jumps there.
    S = lvl·(1 − e^(−k·t)), so dS/dlvl = 1 − e^(−k·t) and dS/dk = lvl·t·e^(−k·t).
    A cut of this fix that asked only which way the two flows point refused a
    corpus model that settles on its own switch this way. In batches of one
    step the run is asked after every step it takes inside the tolerance's
    band of the surface. The far branch may be three times as steep: the flux
    is continuous all the same."""
    model = _model(tmp_path, f"if(Sobs<lvl,k*(lvl-Sobs),(-{steeper}*k)*(Sobs-lvl))")
    extra = {} if max_steps is None else {"max_steps": max_steps}
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["lvl", "k"]).run(
        sample_times=[0.0, 2.0, 60.0, 200.0], rtol=1e-8, atol=1e-10, **extra
    )
    k, lvl = 0.5, 1.0
    t = np.array([0.0, 2.0, 60.0, 200.0])
    got = np.asarray(run.sensitivities)[:, 0, :]
    np.testing.assert_allclose(got[:, 0], 1 - np.exp(-k * t), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got[:, 1], lvl * t * np.exp(-k * t), rtol=1e-6, atol=1e-7)


TURNS_ON = """begin parameters
    1 k 1e3
    2 thr 1e4
    3 d 100
    4 kc 500
end parameters
begin functions
    1 fA() if(Aobs+Cobs>thr,-d,k)
    2 fC() if(Aobs+Cobs>thr,kc,0)
end functions
begin species
    1 A() 0
    2 C() 1e-10
end species
begin reactions
    1 0 1 fA
    2 0 2 fC
end reactions
begin groups
    1 Aobs 1
    2 Cobs 2
end groups
"""


def test_a_crossing_carried_on_by_a_species_far_smaller_than_the_threshold(tmp_path):
    """Control. A rises to thr = 1e4 and then falls at d, and C, 1e-10 until
    then, rises at kc from the crossing: A + C goes on through at kc − d. The
    way C moves the residual is read over a millionth of C, 1e-16, which is
    under an ulp of 1e4, so it read as not moving it at all, and the far side's
    flow as −d: a slide. A corpus model whose switch turns a species on from
    nothing was refused that way. The difference is retaken over wider steps
    where it is lost. With t* = thr/k, C = kc·(t − t*): dC/dk = kc·t*/k and
    dC/dthr = −kc/k."""
    path = tmp_path / "m.net"
    path.write_text(TURNS_ON)
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["k", "thr"]
    ).run(sample_times=[0.0, 5.0, 12.0, 20.0], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, 1, :]
    np.testing.assert_allclose(got, [500 * 10 / 1e3, -500 / 1e3], rtol=1e-6)
