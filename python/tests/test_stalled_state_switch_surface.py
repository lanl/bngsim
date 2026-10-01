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
and one short enough to pass leaves A where it is. CVODE took 500 steps per
batch for 1e-4 of time until the wall clock ended the run.

Where a whole batch of steps has been spent that way, the state is now carried
across the surface along the flow and the integrator restarts there.

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
