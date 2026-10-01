"""A plain run pinned on a state-switch surface (issue #928).

``A <-> B`` conserves A + B, so ``Aobs < thr`` and ``Bobs > thrB`` with
``thrB = A0 + B0 − thr`` are one surface, spelled twice. Each switch turns on a
source of its own. thr sits 1e-5 of the way from A's asymptote to A0, so the
crossing is slow.

A run without sensitivities restarts at a state-switch root only where the
solver's own trajectory resolves the crossing (issue #897). Here the roots are
found in steps the discontinuity has already collapsed, none is resolved, and
the run steps on with A exactly on thr and B one ulp short of thrB. A step long
enough to move A by an ulp carries the jump in Y' into an error test it fails,
and one short enough to pass leaves A where it is. CVODE took a batch of steps
for 1e-4 of time until the wall clock ended the run.

Where a whole batch of steps has been spent that way, and the flow would have
carried the residual across in the time it has been seen there, the residual
is now put the few ulp to its far side, by moving each species it reads that
the flow moves by the same few ulp of its own, and the integrator restarts
there.

One switch does the same under a slow approach. ``if(A > thr, kb, 0)`` with A
rising from 1 at 1e-8 a unit of time: a step that moves A by an ulp is 2e-8
long, and the jump in Y' lets pass only a step of 3e-13.

Every expected value is a closed form: A = Aeq + (A0 − Aeq)·e^(−(a+kback)·t)
crosses thr at t*, and Y = kb·(T − t*), Z = kc·(T − t*).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

A0, RATE, KB, KC, T_END = 10.0, 0.5, 3.0, 2.0, 30.0
AEQ = A0 * 0.1 / 0.6
THR = AEQ + (A0 - AEQ) * 1e-5

NET = """begin parameters
    1 A0 10
    2 B0 0
    3 a 0.5
    4 kback 0.1
    5 thr {thr!r}
    6 kb 3
    7 kc 2
    8 thrB A0+B0-thr
end parameters
begin functions
    1 fY() {fY}
    2 fZ() {fZ}
end functions
begin species
    1 A() A0
    2 B() B0
    3 Y() 0
    4 Z() 0
end species
begin reactions
    1 1 2 a #_R1
    2 2 1 kback #_R2
    3 0 3 fY #_R3
    4 0 4 fZ #_R4
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
end groups
"""
ON_A = "if(Aobs<thr,kb,0)"
ON_B = "if(Bobs>thrB,kc,0)"


def _ends(tmp_path, f_y, f_z, nudge, timeout=5.0):
    path = tmp_path / "m.net"
    path.write_text(NET.format(thr=THR, fY=f_y, fZ=f_z))
    model = bngsim.Model.from_net(path)
    kback = 0.1 + nudge
    model.set_param("kback", kback)
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, T_END), n_points=2, rtol=1e-10, atol=1e-12, timeout=timeout
    )
    names = list(run.species_names)
    end = np.asarray(run.species)[-1]
    a_eq = A0 * kback / (RATE + kback)
    t_star = np.log((A0 - a_eq) / (THR - a_eq)) / (RATE + kback)
    return end[names.index("Y()")], end[names.index("Z()")], T_END - t_star


@pytest.mark.parametrize("nudge", [1e-7, -6.99e-7, -1.39e-7, -6.9e-8, 2.81e-7, 4.91e-7, 7.01e-7])
def test_one_surface_spelled_twice_at_a_slow_crossing(tmp_path, nudge):
    """The issue's run, at 1e-7, and six more of 25 rates scanned around it.
    Each ended in the wall-clock timeout, at t = 19.2."""
    y, z, after = _ends(tmp_path, ON_A, ON_B, nudge)
    assert y == pytest.approx(KB * after, rel=1e-5)
    assert z == pytest.approx(KC * after, rel=1e-5)


@pytest.mark.parametrize("nudge", [-1e-7, 0.0, 3e-7, -3e-7])
def test_the_same_surface_at_other_rates(tmp_path, nudge):
    """Control. Whether the run is pinned is a matter of where the last steps
    before the surface land, which the rate constant moves. These got across on
    main."""
    y, z, after = _ends(tmp_path, ON_A, ON_B, nudge)
    assert y == pytest.approx(KB * after, rel=1e-5)
    assert z == pytest.approx(KC * after, rel=1e-5)


@pytest.mark.parametrize(("f_y", "f_z"), [(ON_A, "kc"), ("kb", ON_B)], ids=["A-only", "B-only"])
def test_either_spelling_alone(tmp_path, f_y, f_z):
    """Control. With one of the two switches replaced by a constant the run
    finished on main, in 0.2 s."""
    y, z, after = _ends(tmp_path, f_y, f_z, 1e-7)
    assert y == pytest.approx(KB * (after if f_y == ON_A else T_END), rel=1e-5)
    assert z == pytest.approx(KC * (after if f_z == ON_B else T_END), rel=1e-5)


SLIDE = """begin parameters
    1 amp 1.0
end parameters
begin functions
    1 f() if(Sobs<1,amp,-amp)
end functions
begin species
    1 S() 0
end species
begin reactions
    1 0 1 f
end reactions
begin groups
    1 Sobs 1
end groups
"""


def test_a_slide_along_the_surface_is_not_carried_across(tmp_path):
    """Control. ``if(S < 1, amp, −amp)`` holds S at 1 from t = 1 on: both
    branches point into the surface, so there is no far side to carry the state
    to, and the run is left to step as it did (#908 is its cost)."""
    path = tmp_path / "slide.net"
    path.write_text(SLIDE)
    run = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        sample_times=[0.0, 0.5, 1.5, 3.0], rtol=1e-6, atol=1e-8, timeout=60.0
    )
    np.testing.assert_allclose(np.asarray(run.species)[:, 0], [0.0, 0.5, 1.0, 1.0], atol=1e-5)


SLOW = """begin parameters
    1 eps {eps!r}
    2 thr {thr!r}
    3 kb 3
    4 w 50
end parameters
begin functions
    1 fY() if(Aobs>thr,kb,0)
    2 fP() w*Qobs
    3 fQ() -w*Pobs
end functions
begin species
    1 A() 1
    2 Y() 0
    3 P() 0
    4 Q() 1
end species
begin reactions
    1 0 1 eps
    2 0 2 fY
    3 0 3 fP
    4 0 4 fQ
end reactions
begin groups
    1 Aobs 1
    2 Pobs 3
    3 Qobs 4
end groups
"""
SLOW_END, SLOW_RTOL = 10.0, 1e-10


def _slow(tmp_path, eps, thr, **run_options):
    """A rises from 1 at ``eps`` and turns Y on past ``thr``, beside an
    oscillator the switch does not read. Returns the state at the end."""
    path = tmp_path / "slow.net"
    path.write_text(SLOW.format(eps=eps, thr=thr))
    run = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, SLOW_END), n_points=2, rtol=SLOW_RTOL, atol=1e-12, timeout=5.0, **run_options
    )
    return np.asarray(run.species)[-1]


@pytest.mark.parametrize("eps", [1e-7, 1e-8, 1e-9])
def test_one_switch_under_a_slow_approach(tmp_path, eps):
    """A reaches thr = 1 + 5·eps at t = 5. Each of these ended in the
    wall-clock timeout, with A held on thr.

    The crossing time is known only as well as A is: an error of rtol·A in A
    moves it by rtol/eps, which is 1e-3 to 0.1 here. Y is held to a tenth of
    that. The oscillator is not what was pinned, and it is held to what the
    run's tolerances give it."""
    thr = 1.0 + 5.0 * eps
    a, y, p, q = _slow(tmp_path, eps, thr)
    assert a == pytest.approx(1.0 + eps * SLOW_END, rel=0.0, abs=1e-3 * eps * SLOW_END)
    assert y == pytest.approx(KB * (SLOW_END - (thr - 1.0) / eps), abs=0.1 * KB * SLOW_RTOL / eps)
    assert p == pytest.approx(np.sin(50.0 * SLOW_END), abs=1e-6)
    assert q == pytest.approx(np.cos(50.0 * SLOW_END), abs=1e-6)


def test_a_pinned_state_seen_over_several_short_batches(tmp_path):
    """With 20 steps to a batch, no one batch lasts as long as the flow needs
    to cross the few ulp, so the first batch that finds the state pinned does
    not move it. The time it has been seen there is counted from that batch
    on."""
    eps = 1e-9
    thr = 1.0 + 5.0 * eps
    a, y, p, q = _slow(tmp_path, eps, thr, max_steps=20)
    assert a == pytest.approx(1.0 + eps * SLOW_END, rel=0.0, abs=1e-3 * eps * SLOW_END)
    assert y == pytest.approx(KB * (SLOW_END - (thr - 1.0) / eps), abs=0.1 * KB * SLOW_RTOL / eps)
    assert p == pytest.approx(np.sin(50.0 * SLOW_END), abs=1e-6)
    assert q == pytest.approx(np.cos(50.0 * SLOW_END), abs=1e-6)


def test_a_state_that_is_close_to_a_surface_is_not_pinned_on_it(tmp_path):
    """Control. A starts four ulp short of thr and approaches it at 1e-20, so
    it does not cross in this run. The oscillator uses up every batch of 50
    steps, and each batch ends with the residual within a few ulp of zero and a
    step far too short for the flow to move it. The flow would not have carried
    it across in the time it has been seen there, so it is left where it is."""
    thr = float(1.0 + 4 * np.finfo(float).eps)
    a, y, p, q = _slow(tmp_path, 1e-20, thr, max_steps=50)
    assert a == 1.0
    assert y == 0.0
    assert p == pytest.approx(np.sin(50.0 * SLOW_END), abs=1e-6)
    assert q == pytest.approx(np.cos(50.0 * SLOW_END), abs=1e-6)


def test_a_threshold_on_a_counter_whose_crossing_stop_is_stood_down():
    """Issue #54's fixture: a rate law that turns on when a counter species
    reaches ``sigma``. As written the run stops exactly on the crossing (issue
    #443). With that stop stood down, which is how a crossing bngsim cannot
    resolve reaches the integrator, the step size collapsed with the counter an
    ulp short of ``sigma`` and the run ended in "CVODE made no progress". It is
    now put across, and matches the run as written."""
    from pathlib import Path

    net = Path(__file__).resolve().parent.parent.parent / "tests" / "data"
    net = net / "switch_discontinuity_stall.net"
    as_written = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode").run(
        t_span=(0.0, 648.0), n_points=649
    )
    model = bngsim.Model.from_net(str(net))
    assert model.time_discontinuity_conditions() == ("t>=sigma",)
    model._derived_time_disc_conditions = ()
    stood_down = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 648.0), n_points=649)
    want = np.asarray(as_written.species)
    np.testing.assert_allclose(
        np.asarray(stood_down.species), want, rtol=1e-9, atol=1e-9 * np.abs(want).max()
    )


SUM = """begin parameters
    1 eps {eps!r}
    2 thr {thr!r}
    3 kb 3
end parameters
begin functions
    1 fY() if(Sobs{op}thr,kb,0)
end functions
begin species
    1 A1() {a1!r}
    2 {fixed}A2() {a2!r}
    3 Y() 0
end species
begin reactions
    1 0 1 eps
    2 0 3 fY
end reactions
begin groups
    1 Sobs 1,2
end groups
"""


@pytest.mark.parametrize(
    ("a2", "rising", "fixed"),
    [(1e-3, True, ""), (0.0, False, ""), (1e-3, True, "$"), (0.0, False, "$")],
    ids=["small", "zero", "fixed-small", "fixed-zero"],
)
def test_a_threshold_on_a_sum_moves_only_the_species_the_flow_moves(tmp_path, a2, rising, fixed):
    """``if(A1 + A2 > thr, kb, 0)`` with A1 = 1e6 moving at 1e-8 of itself and
    A2 a species nothing moves, small or 0, free or fixed. Each ended in the
    wall-clock timeout.

    The residual reads both species, and an ulp of it is an ulp of A1: 1e-10.
    Spread over the two by the gradient, as a first cut did it, the move put
    A2 at 1e-3 + 1e-9, or at −1e-9 from 0, which a reaction that feeds on A2
    grows from. Only a species the flow moves is moved, so A2 is where it
    started, to the bit."""
    a1 = 1e6
    eps = (1.0 if rising else -1.0) * 1e-8 * a1
    thr = (a1 + a2) + 5.0 * eps
    path = tmp_path / "sum.net"
    path.write_text(
        SUM.format(eps=eps, thr=thr, a1=a1, a2=a2, op=">" if rising else "<", fixed=fixed)
    )
    run = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=1e-10, atol=1e-12, timeout=5.0
    )
    end = np.asarray(run.species)[-1]
    assert end[1] == a2
    assert end[0] == pytest.approx(a1 + eps * 10.0, rel=1e-12)
    assert end[2] == pytest.approx(KB * 5.0, abs=0.1 * KB * 1e-10 * a1 / abs(eps))


REST = """begin parameters
    1 k 1.0
    2 Ainf 1.0
    3 thr {thr!r}
    4 kf 1.0
    5 kb 3
    6 w 0.5
end parameters
begin functions
    1 fA() k*(Ainf-Aobs)+if(Aobs>thr,kf,0)
    2 fY() if(Aobs>thr,kb,0)
    3 fP() w*Qobs
    4 fQ() -w*Pobs
end functions
begin species
    1 A() 0.5
    2 Y() 0
    3 P() 0
    4 Q() 1
end species
begin reactions
    1 0 1 fA
    2 0 2 fY
    3 0 3 fP
    4 0 4 fQ
end reactions
begin groups
    1 Aobs 1
    2 Pobs 3
    3 Qobs 4
end groups
"""


@pytest.mark.parametrize("ulps", [0, 1, 8])
@pytest.mark.parametrize("rtol", [1e-8, 1e-10])
def test_a_state_that_comes_to_rest_short_of_a_surface_is_not_put_across(tmp_path, ulps, rtol):
    """Control. A relaxes to Ainf = 1 and thr is Ainf, or an ulp or eight past
    it, so A never reaches it and Y stays 0. Past thr a feedback switches on,
    which would carry A on to 2.

    A ends a few ulp short of Ainf, held there by rounding with a flow that is
    not 0, and over 1e5 units of time that flow would have covered the few ulp
    many times. A first cut put it across: Y = 3e5 and A = 2. The flow has to
    reach the surface, and this one runs out before it: read as far back again,
    it is larger in proportion to the distance from Ainf."""
    thr = 1.0
    for _ in range(ulps):
        thr = float(np.nextafter(thr, np.inf))
    path = tmp_path / "rest.net"
    path.write_text(REST.format(thr=thr))
    run = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, 1e5), n_points=2, rtol=rtol, atol=rtol * 1e-2, timeout=60.0
    )
    end = np.asarray(run.species)[-1]
    assert end[1] == 0.0
    assert end[0] <= thr


TWO_MOVING = """begin parameters
    1 thr {thr!r}
    2 kb 3
    3 fAv {fa!r}
    4 fBv {fb!r}
end parameters
begin functions
    1 fY() {cond}
    2 fA() fAv
    3 fB() fBv
end functions
begin species
    1 A() {a0!r}
    2 B() {b0!r}
    3 Y() 0
end species
begin reactions
    1 0 1 fA
    2 0 2 fB
    3 0 3 fY
end reactions
begin groups
    1 Aobs 1
    2 Bobs 2
    3 Sobs 1,2
end groups
"""


@pytest.mark.parametrize(
    ("cond", "a0", "b0", "fa", "fb", "thr"),
    [
        ("if(Sobs>thr,kb,0)", 1.0, 3.0, 0.1, -0.1 + 1e-8, 4.0 + 5e-8),
        ("if(Aobs>Bobs,kb,0)", 1.0, 1.0 + 5e-8, 0.1 + 1e-8, 0.1, 0.0),
        ("if(Aobs-Bobs>thr,kb,0)", 2.0, 1.0, 1e-3 + 1e-8, 1e-3, 1.0 + 5e-8),
    ],
    ids=["a-sum-of-two-that-cancel", "one-overtaking-another", "a-difference"],
)
def test_two_species_that_move_under_one_threshold(tmp_path, cond, a0, b0, fa, fb, thr):
    """A and B each move at 0.1 or 1e-3 a unit of time, and what the condition
    reads of them moves at 1e-8: it crosses at t = 5. Each ended in the
    wall-clock timeout.

    Each species is a line in time, so its value at the end is known to
    rounding. A cut that moved each species as its own flow would move it, in
    proportion, moved A and B by 1e7 times what the residual needed: 5e-8 of
    each. They are moved a few ulp of their own."""
    path = tmp_path / "two.net"
    path.write_text(TWO_MOVING.format(cond=cond, a0=a0, b0=b0, fa=fa, fb=fb, thr=thr))
    run = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=1e-10, atol=1e-12, timeout=5.0
    )
    a, b, y = np.asarray(run.species)[-1]
    assert a == pytest.approx(a0 + fa * 10.0, rel=1e-11)
    assert b == pytest.approx(b0 + fb * 10.0, rel=1e-11)
    assert y == pytest.approx(KB * 5.0, abs=0.1 * KB * 1e-10 * max(a0, b0) / 1e-8)
