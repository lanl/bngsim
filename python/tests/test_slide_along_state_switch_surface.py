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
a state switch is taken in batches of 8 steps and asked after each. The
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
    ("rtol", "n_points"),
    [(1e-4, 101), (1e-5, 1001), (1e-6, 10001), (1e-3, 1001), (1e-4, 10001)],
    ids=["101", "1001", "10001", "loose-1001", "loose-10001"],
)
def test_a_slide_between_the_points_of_a_dense_grid_is_refused(tmp_path, rtol, n_points):
    """The same slide reported on a grid: no interval between two points uses a
    batch of 10000 steps up, which was the only place a first cut of this
    asked, and dS/damp came back 1.5 for 0 at each of these. A later cut let
    the two loose ones through: −1.0 and 1.0 for 0."""
    sim = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(t_span=(0.0, 1.5), n_points=n_points, rtol=rtol, atol=1e-2 * rtol, timeout=30.0)


def test_a_slide_under_a_batch_of_a_million_steps_is_refused(tmp_path):
    """``max_steps`` is the batch the caller asks for, and the run is asked
    about its switches after 8 steps whatever it is."""
    sim = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=[0.0, 0.5, 1.5], rtol=1e-6, atol=1e-8, max_steps=1000000, timeout=30)


def test_a_slide_along_a_surface_that_is_not_a_level_is_refused(tmp_path):
    """``if(exp(Sobs) < lvl, amp, −amp)``: the surface is S = 0, and the
    residual is not linear in the state. The states either side of the surface
    are placed by the residual's own value there, not by a step in S:
    dS/damp came back 1.5 for 0."""
    model = _model(tmp_path, "if(exp(Sobs)<lvl,amp,-amp)", s0=-1.0)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=[0.0, 0.5, 1.5], rtol=1e-4, atol=1e-6, timeout=30.0)


def test_a_slide_reached_within_the_first_steps_is_refused(tmp_path):
    """At rtol = atol = 1e-3 the state is on the surface before the run has
    been asked once, so there is no earlier reading to compare the residual
    with: dS/damp came back 1.5 for 0."""
    sim = bngsim.Simulator(_model(tmp_path, FROM_BELOW), method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=[0.0, 0.5, 1.5], rtol=1e-3, atol=1e-3, timeout=30.0)


def test_a_run_that_ends_just_after_the_slide_begins_is_refused(tmp_path):
    """S starts a thousandth short of the surface and the run ends at
    t = 0.01, nine thousandths into the slide: dS/damp came back 0.01 for 0."""
    model = _model(tmp_path, FROM_BELOW, s0=1 - 1e-3)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["amp"])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=[0.0, 0.005, 0.01], rtol=1e-6, atol=1e-8, timeout=30.0)


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


def test_a_state_closing_on_the_surface_step_by_step_is_not_yet_a_slide(tmp_path):
    """Control. ``if(S < lvl, k·(lvl − S) + 0.05, −amp)``: S closes on the
    surface as 1.1·(1 − e^(−k·t)) and will slide along it once it is there. The
    run ends with S at 0.97, and in batches of one step it is asked at every
    step on the way. A slide is refused inside the tolerance's band of the
    surface, not a hundredth of the way from it: amp is not read yet, and
    dS/damp = 0."""
    model = _model(tmp_path, "if(Sobs<lvl,k*(lvl-Sobs)+0.05,-amp)")
    t_end = float(-2.0 * np.log(0.13 / 1.1))
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["amp"]).run(
        sample_times=[0.0, 2.0, t_end], rtol=1e-8, atol=1e-10, max_steps=1
    )
    assert np.asarray(run.species)[-1, 0] == pytest.approx(0.97, rel=1e-6)
    np.testing.assert_array_equal(np.asarray(run.sensitivities)[:, 0, 0], [0.0, 0.0, 0.0])


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


@pytest.mark.parametrize("steeper", [1, 3, 16, 1000])
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
    band of the surface. The far branch may be 3, 16 or 1000 times as steep:
    the flux is continuous all the same, and each side's flow runs out at the
    surface. A cut that compared the two sides' rates a fixed distance from
    the surface refused the steeper two."""
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


RESTS_ON = """begin parameters
    1 A0 {a0!r}
    2 k 1
    3 thr 2
    4 kb 3
end parameters
begin functions
    1 src() k*thr
    2 fW() if(Aobs<thr,kb,0.25*kb)
end functions
begin species
    1 A() A0
    2 W() 0
end species
begin reactions
    1 0 1 src
    2 1 0 k
    3 0 2 fW
end reactions
begin groups
    1 Aobs 1
end groups
"""


@pytest.mark.parametrize("max_steps", [None, 2, 20])
@pytest.mark.parametrize("a0", [10.0, 0.0], ids=["from-above", "from-below"])
def test_a_state_at_rest_on_a_threshold_that_gates_another_rate_is_not_a_slide(
    tmp_path, a0, max_steps
):
    """Control. A' = k·(thr − A): A relaxes to thr and rests on it, and
    ``if(A < thr, kb, kb/4)`` is the rate of W. The right-hand side jumps at
    the surface and the flow past it points back, but the flow on this side
    runs out there: the state never arrives, and W stays on the branch it
    started on. A corpus model whose voltage settles on the threshold of its
    recovery rate does this. A cut that asked whether the rate law jumps and
    the far side points back refused it."""
    path = tmp_path / "m.net"
    path.write_text(RESTS_ON.format(a0=a0))
    extra = {} if max_steps is None else {"max_steps": max_steps}
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["k", "thr", "kb"]
    ).run(sample_times=[0.0, 10.0, 30.0, 100.0], rtol=1e-8, atol=1e-10, timeout=60.0, **extra)
    rate = 0.75 if a0 > 2.0 else 3.0
    got = np.asarray(run.sensitivities)[-1]
    assert np.asarray(run.species)[-1, 1] == pytest.approx(100.0 * rate, rel=1e-9)
    np.testing.assert_allclose(got[0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(got[1], [0.0, 0.0, 100.0 * rate / 3.0], atol=1e-6)


BLOWS_UP = """begin parameters
    1 k 1.0
    2 kb 3.0
end parameters
begin functions
    1 fX() k*Xobs*Xobs
    2 fY() if(Xobs>2,kb,0)
end functions
begin species
    1 X() 1.0
    2 Y() 0.0
end species
begin reactions
    1 0 1 fX
    2 0 2 fY
end reactions
begin groups
    1 Xobs 1
end groups
"""


def test_a_blow_up_beside_a_state_switch_fails_as_it_did(tmp_path):
    """Control. X' = k·X² is infinite at t = 1, and the run ends in CVODE on a
    sensitivity right-hand side that is not finite, which the error names. A
    run that carries sensitivities through a state switch is taken in batches
    of 8 steps, and a stall is a whole ``max_steps`` of them that do not move
    the time. Counted from wherever the time stopped moving, and not in the
    windows a run in whole batches has, that was reached a few steps before
    CVODE failed, and the run was called stalled at a discontinuity."""
    path = tmp_path / "m.net"
    path.write_text(BLOWS_UP)
    sim = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["k", "kb"]
    )
    with pytest.raises(bngsim.SimulationError, match="CV_REPTD_SRHSFUNC_ERR.*non-finite"):
        sim.run(sample_times=[0.0, 2.0], rtol=1e-8, atol=1e-10, timeout=60.0)


@pytest.mark.parametrize(
    ("tol", "s0", "times"),
    [
        (1e-2, 0.999, [0.0, 0.05, 0.2]),
        (1e-2, 0.99, [0.0, 0.05, 0.2]),
        (1e-2, 0.9, [0.0, 0.05, 0.2]),
        (1e-3, 0.999, [0.0, 0.01, 0.05]),
    ],
    ids=["a-thousandth-short", "a-hundredth-short", "a-tenth-short", "tighter"],
)
def test_a_short_slide_at_a_loose_tolerance_is_refused(tmp_path, tol, s0, times):
    """S starts inside the tolerance's band of the surface, or a few steps
    from it, and slides for most of a short run: dS/damp came back the length
    of the run, 0.2 for 0.001 in the first. A state that starts inside the
    band is no nearer the surface than it has ever been, so every switch is
    read at the first asking, and a state inside what the tolerance allows
    the residual is held: the run cannot tell it from one on the surface."""
    sim = bngsim.Simulator(
        _model(tmp_path, FROM_BELOW, s0), method="ode", sensitivity_params=["amp"]
    )
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=times, rtol=tol, atol=tol, timeout=30.0)


@pytest.mark.parametrize("tol", [1e-2, 1e-3, 1e-4])
def test_a_state_inside_a_wide_band_that_has_not_arrived_is_not_a_slide(tmp_path, tol):
    """Control. S goes from 0.9 to 0.95, with the surface at 1. At a tolerance
    of a hundredth the band is a quarter of the state, S is inside it all the
    way, and the two flows at the surface are those of the slide that begins
    at t = 0.1. The state is not held there: its residual is closing on the
    surface at the rate its own flow closes at. dS/damp = t."""
    run = bngsim.Simulator(
        _model(tmp_path, FROM_BELOW, 0.9), method="ode", sensitivity_params=["amp"]
    ).run(sample_times=[0.0, 0.01, 0.05], rtol=tol, atol=tol)
    np.testing.assert_allclose(
        np.asarray(run.sensitivities)[:, 0, 0], [0.0, 0.01, 0.05], rtol=1e-6
    )


MOVING = """begin parameters
    1 a 1.0
    2 b 1.0
    3 lvl {lvl!r}
    4 r 1.0
    5 k 1.0
end parameters
begin functions
    1 f() {law}
    2 fZ() 1
    3 fX() 9*sin(10*Zobs)
end functions
begin species
    1 S() {s0!r}
    2 Z() 0
    3 X() 0.1
end species
begin reactions
    1 0 1 f
    2 0 2 fZ
    3 0 3 fX
end reactions
begin groups
    1 Sobs 1
    2 Zobs 2
    3 Xobs 3
end groups
"""


def _moving(tmp_path, law, s0, lvl=1.0):
    path = tmp_path / "m.net"
    path.write_text(MOVING.format(law=law, s0=s0, lvl=lvl))
    return bngsim.Model.from_net(path)


NOT_THERE = {
    # S' = a·t until t = 0.4 and −a after: S peaks at 0.98 and comes back.
    # S(1) = 0.9 + a·0.08 − a·0.6, so dS/da = −0.52.
    "a-flow-that-turns-back-short-of-the-surface": (
        "if(Sobs<lvl,if(Zobs<0.4,a*Zobs,-a),-b)",
        "a",
        1e-3,
        {"t_span": (0.0, 1.0), "n_points": 101},
        0.38,
        -0.52,
        5e-2,
    ),
    # S = 0.9 + r·t²/2: 0.9648 at t = 0.36, dS/dr = 0.0648.
    "a-flow-that-speeds-up": (
        "if(Sobs<lvl,r*Zobs,-b)",
        "r",
        1e-2,
        {"sample_times": [0.0, 0.36]},
        0.9648,
        0.0648,
        8e-2,
    ),
    # X = 1 − 0.9·cos(10·t): S = 0.9 + a·(t − 0.09·sin(10·t)), 0.970 at 0.16.
    "a-flow-that-comes-and-goes": (
        "if(Sobs<lvl,a*Xobs,-b)",
        "a",
        1e-3,
        {"sample_times": [0.0, 0.16]},
        0.9700,
        0.0700,
        2e-2,
    ),
}


@pytest.mark.parametrize("case", sorted(NOT_THERE))
def test_a_state_in_the_band_whose_flow_changes_on_the_way_is_not_a_slide(tmp_path, case):
    """Control. Each ends short of the surface, inside the band, with the flows
    of a slide at the surface. A cut that asked whether the flow at the
    surface had had the time to bring the state there refused all three: a
    flow that speeds up on the way, or comes and goes with another state,
    covers less ground than its value at the surface says. The state is asked
    whether its residual has stopped closing, which these have not."""
    law, par, tol, when, s_end, want, rel = NOT_THERE[case]
    sim = bngsim.Simulator(_moving(tmp_path, law, 0.9), method="ode", sensitivity_params=[par])
    run = sim.run(rtol=tol, atol=tol, timeout=30.0, **when)
    assert np.asarray(run.species)[-1, 0] == pytest.approx(s_end, rel=rel)
    assert np.asarray(run.sensitivities)[-1, 0, 0] == pytest.approx(want, rel=rel)


HELD = {
    # S closes on the surface as k·(lvl + 0.1 − S), a tenth of k where it gets
    # there at t = ln(11)/k, and is held: dS/dk = 0. It came back 0.216.
    "a-flow-that-slows-toward-the-surface": (
        "if(Sobs<lvl,k*(lvl+0.1-Sobs),-b)",
        "k",
        0.0,
        1.0,
        [0.0, 2.6],
        1e-3,
        1e-3,
    ),
    # The surface is sqrt(S) = 1.5: S rises from 1.25 to 2.25 at t = 1 and is
    # held. dS/da came back 1.0 for 0.
    "a-surface-that-is-a-square-root": (
        "if(sqrt(Sobs)<lvl,a,-b)",
        "a",
        1.25,
        1.5,
        [0.0, 0.5, 1.5],
        1e-4,
        1e-6,
    ),
    # S' = ±1 − 0.25·t: held on the surface from t = 0.87 to t = 4, where the
    # upward flow no longer reaches it. dS/da came back 6 for 2.
    "a-slide-that-ends": (
        "if(Sobs<lvl,a,-b)-0.25*Zobs",
        "a",
        0.0,
        1.0,
        [0.0, 2.0, 6.0],
        1e-4,
        1e-6,
    ),
}


@pytest.mark.parametrize("case", sorted(HELD))
def test_a_slide_whose_flow_is_not_what_it_is_at_the_surface_is_refused(tmp_path, case):
    """Three slides the same cut let through, or left to the wall clock at a
    tight tolerance: a flow that slows toward the surface needs longer to
    arrive than its value there says, and was never taken to have arrived."""
    law, par, s0, lvl, times, rtol, atol = HELD[case]
    sim = bngsim.Simulator(_moving(tmp_path, law, s0, lvl), method="ode", sensitivity_params=[par])
    with pytest.raises(Exception, match="issue #926"):
        sim.run(sample_times=times, rtol=rtol, atol=atol, timeout=30.0)
