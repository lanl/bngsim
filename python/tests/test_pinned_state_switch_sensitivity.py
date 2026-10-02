"""A sensitivity run pinned on a state-switch surface it approaches slowly (issue #952).

``if(Aobs > thr, kb, 0)`` with A rising from 1 at ``eps`` a unit of time and
``thr = 1 + 5·eps``: A reaches thr at t = 5, and Y = kb·(t − 5) from there.

A step that moves A by one ulp is 2.2e-16/eps long, and the jump in Y' lets pass
only a step of about atol/kb. Under a slow enough approach no step carries A
across: one that would fails its error test, and one that passes leaves A an
ulp short of thr. No root is reported, so the run's state-switch jump never
runs. At eps = 1e-8 the run ended in the wall-clock timeout. At 1e-10 it
returned, with the switch not taken: Y(10) = 0 for 15 and dY/dthr = 0, with no
warning.

A plain run is carried across such a surface (issue #928). A run that carries
sensitivities has a jump to apply there, taken from a trajectory that crosses,
and this one does not. It is refused.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 eps {eps!r}
    2 thr {thr!r}
    3 kb 3
    4 Ainf {Ainf!r}
end parameters
begin functions
    1 fY() if(Aobs>thr,kb,0)
    2 fA() {fA}
end functions
begin species
    1 A() 1
    2 Y() 0
end species
begin reactions
    1 0 1 fA
    2 0 2 fY
end reactions
begin groups
    1 Aobs 1
end groups
"""


def _model(tmp_path, eps, thr=None, fA="eps", Ainf=0.0):
    path = tmp_path / "m.net"
    path.write_text(NET.format(eps=eps, thr=1 + 5 * eps if thr is None else thr, fA=fA, Ainf=Ainf))
    return bngsim.Model.from_net(path)


def _run(model, **kwargs):
    return bngsim.Simulator(model, method="ode", **kwargs).run(
        t_span=(0.0, 10.0), n_points=2, rtol=1e-10, atol=1e-12, timeout=20
    )


@pytest.mark.parametrize("eps", [1e-8, 1e-10])
def test_a_surface_approached_too_slowly_to_cross_is_refused(tmp_path, eps):
    """At 1e-8 the run ended in the timeout. At 1e-10 it returned Y(10) = 0 for
    15 and a column of zeros."""
    with pytest.raises(bngsim.SimulationError, match="issue #952"):
        _run(_model(tmp_path, eps), sensitivity_params=["thr"])


@pytest.mark.parametrize("eps", [1e-10, 1e-12, 1e-13])
def test_a_slow_approach_reported_on_a_grid_is_refused(tmp_path, eps):
    """At the default tolerances and 101 points no interval uses a batch of
    10000 steps up, which was the only place a first cut of this asked: the run
    returned Y(10) = 0 for 15 and dY/dthr = 0."""
    sim = bngsim.Simulator(_model(tmp_path, eps), method="ode", sensitivity_params=["thr"])
    with pytest.raises(bngsim.SimulationError, match="issue #952"):
        sim.run(t_span=(0.0, 10.0), n_points=101, timeout=30)


@pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-7])
def test_a_surface_the_steps_cross_is_jumped(tmp_path, eps):
    """Control. Y(10) = 15 and dY/dthr = −kb/eps."""
    run = _run(_model(tmp_path, eps), sensitivity_params=["thr"])
    assert np.asarray(run.species)[-1, 1] == pytest.approx(15.0, rel=1e-6)
    assert np.asarray(run.sensitivities)[-1, 1, 0] == pytest.approx(-3.0 / eps, rel=1e-6)


@pytest.mark.parametrize("max_steps", [1, 2, 20])
@pytest.mark.parametrize("eps", [1e-3, 1e-7])
def test_a_surface_the_steps_cross_in_small_batches_is_jumped(tmp_path, eps, max_steps):
    """Control. At this tolerance the step that takes the rate law's jump is as
    short as a pinned one, for a step or two, and then the state is across. In
    batches of a step or two the run is asked while it is that short, and a
    first cut refused the crossing it was about to make. A state is pinned
    once it has stayed there for 200 steps."""
    run = bngsim.Simulator(_model(tmp_path, eps), method="ode", sensitivity_params=["thr"]).run(
        t_span=(0.0, 10.0), n_points=2, rtol=1e-10, atol=1e-12, timeout=20, max_steps=max_steps
    )
    assert np.asarray(run.species)[-1, 1] == pytest.approx(15.0, rel=1e-6)
    assert np.asarray(run.sensitivities)[-1, 1, 0] == pytest.approx(-3.0 / eps, rel=1e-6)


@pytest.mark.parametrize("eps", [1e-8, 1e-10])
def test_the_plain_run_is_carried_across(tmp_path, eps):
    """Control. Issue #928: Y(10) = 15 to what the tolerance on A allows the
    crossing time, rtol·A/eps."""
    run = _run(_model(tmp_path, eps))
    assert np.asarray(run.species)[-1, 1] == pytest.approx(15.0, abs=3.0 * (2e-10 / eps + 1e-6))


@pytest.mark.parametrize("max_steps", [1, 2, 5, 20])
def test_a_state_at_rest_within_the_tolerance_of_the_surface_is_not_a_slide(tmp_path, max_steps):
    """Control. A relaxes to Ainf, 1e-7 short of thr, which is inside the
    tolerance's band of the surface. It heads for the surface all the way and
    the flow past the surface points back, which is what a slide looks like
    from where the state is. At the surface the flow on this side points back
    too: the state never gets there. Y = 0 and both columns are 0."""
    model = _model(tmp_path, 0.5, thr=2.0, fA="eps*(Ainf-Aobs)", Ainf=2.0 - 1e-7)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["eps", "thr"]).run(
        t_span=(0.0, 200.0), n_points=3, rtol=1e-8, atol=1e-10, timeout=20, max_steps=max_steps
    )
    assert np.asarray(run.species)[-1, 1] == 0.0
    np.testing.assert_array_equal(np.asarray(run.sensitivities)[-1, 1, :], [0.0, 0.0])


@pytest.mark.parametrize("max_steps", [None, 50])
def test_a_state_that_comes_to_rest_short_of_the_surface_is_not_refused(tmp_path, max_steps):
    """Control. A relaxes to Ainf, 1024 ulp short of thr, and stays there: the
    switch is never taken, Y = 0, and nothing is pinned, whether the run takes
    its steps in batches of 50 or not. (Within about 64 ulp the root finder
    reports a crossing the state then does not make, which is refused on main
    too.)"""
    thr = 1.5
    Ainf = float(thr - 1024 * np.spacing(thr))
    model = _model(tmp_path, 1.0, thr=thr, fA="eps*(Ainf-Aobs)", Ainf=Ainf)
    extra = {} if max_steps is None else {"max_steps": max_steps}
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["eps", "thr"]).run(
        t_span=(0.0, 100.0), n_points=2, rtol=1e-10, atol=1e-12, timeout=20, **extra
    )
    assert np.asarray(run.species)[-1, 1] == 0.0
    np.testing.assert_array_equal(np.asarray(run.sensitivities)[-1, 1, :], [0.0, 0.0])
