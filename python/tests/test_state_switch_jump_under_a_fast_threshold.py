"""A small state-switch jump under a threshold species that moves fast (issue #917).

``fY = if(Bobs < thr, kb, 0)`` with B decaying from 1e8 through thr = 5e7. The
crossing was judged continuous where the jump was under 1e-6 of the rate that
drives it, the largest |dx/dt| among the species the residual reads. That rate
is kdeg·B = 5e6 here, so any jump under 5 was dropped: every column that moves
the crossing time came back 0, with no warning.

The drive is there for a switched flux that vanishes on both branches, the BNGL
signed-rate idiom: across a pair of probes it differs by its slope times the
crossing's speed. Where both branches are extended to the root from two probes
each, that is out of the reading already, and what is left besides a step is
rounding. That is now what the extended reading is held to: what one ulp of
each species the residual reads moves the flux by, and a few ulp of the
readings themselves, never more than the drive tolerance.

Every expected value is a closed form. B = B0·e^(−kdeg·t) reaches thr at
t* = ln(B0/thr)/kdeg.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

KDEG, T_END = 0.1, 15.0
PARAMS = ["kdeg", "thr", "B0", "kb"]

NET = """begin parameters
    1 kb {kb!r}
    2 thr {thr!r}
    3 kdeg 0.1
    4 B0 {B0!r}
    5 ksyn {ksyn!r}
end parameters
begin functions
    1 fY() {law}
end functions
begin species
    1 B() B0
    2 Y() 0
end species
begin reactions
    1 1 0 kdeg
    2 0 2 fY
end reactions
begin groups
    1 Bobs 1
end groups
"""


def _columns(tmp_path, law, kb, pool, ksyn=0.0):
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=law, kb=kb, thr=pool / 2, B0=pool, ksyn=ksyn))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=PARAMS
    ).run(sample_times=[0.0, 5.0, 10.0, T_END], rtol=1e-10, atol=1e-12)
    return np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]


def _step(kb, pool):
    """Y = kb·(T − t*) past the crossing."""
    t_star = np.log(2.0) / KDEG
    return np.array(
        [kb * t_star / KDEG, kb / (KDEG * pool / 2), -kb / (KDEG * pool), T_END - t_star]
    )


@pytest.mark.parametrize("kb", [3.0, 3e-3])
@pytest.mark.parametrize("pool", [1e8, 1e10, 1e12])
def test_a_jump_under_a_fast_threshold_species(tmp_path, pool, kb):
    """dY/dkdeg, dY/dthr and dY/dB0 were 0: 207.9 for kb = 3."""
    got = _columns(tmp_path, "if(Bobs<thr,kb,0)", kb, pool)
    np.testing.assert_allclose(got, _step(kb, pool), rtol=1e-6)


def test_a_smaller_jump_under_a_slower_one(tmp_path):
    """At a pool of 1e7 the drive is 5e5 and a jump of 3 was read. One of 3e-3
    was not."""
    got = _columns(tmp_path, "if(Bobs<thr,kb,0)", 3e-3, 1e7)
    np.testing.assert_allclose(got, _step(3e-3, 1e7), rtol=1e-6)


def test_a_jump_beside_a_large_constant_in_the_same_rate_law(tmp_path):
    """``ksyn + if(…, kb, 0)`` with ksyn = 1e6: the flux does not move with the
    species the residual reads, so it is allowed no rounding, and the jump of 3
    is read under a drive of 5e6."""
    got = _columns(tmp_path, "ksyn+if(Bobs<thr,kb,0)", 3.0, 1e8, ksyn=1e6)
    want = _step(3.0, 1e8)
    np.testing.assert_allclose(got[:3], want[:3], rtol=1e-6)


def test_a_switched_law_that_reads_the_fast_species(tmp_path):
    """``if(B < thr, kb·B/thr, 0)`` jumps by kb at the crossing and falls with B
    after it: Y = (kb/kdeg)·(1 − (B0/thr)·e^(−kdeg·T))."""
    kb, pool = 3.0, 1e8
    thr = pool / 2
    got = _columns(tmp_path, "if(Bobs<thr,kb*Bobs/thr,0)", kb, pool)
    tail = np.exp(-KDEG * T_END)
    want = [
        -(kb / KDEG**2) * (1 - (pool / thr) * tail) + (kb / KDEG) * (pool / thr) * T_END * tail,
        (kb / KDEG) * (pool / thr**2) * tail,
        -(kb / KDEG) * tail / thr,
        (1 - (pool / thr) * tail) / KDEG,
    ]
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize("pool", [1e7, 1e8, 1e12])
def test_a_flux_that_vanishes_at_the_surface_is_still_continuous(tmp_path, pool):
    """Control. ``if(B < thr, kb·(thr − B)/thr, 0)`` is 0 at the crossing on both
    branches. Its value there is the rounding of thr − B, an ulp of 5e7 times
    kb/thr, which is what one ulp of B moves it by: no saltation term.
    Y = kb·(T − t*) − (kb/kdeg)·(1 − (B0/thr)·e^(−kdeg·T))."""
    kb = 3.0
    thr = pool / 2
    got = _columns(tmp_path, "if(Bobs<thr,kb*(thr-Bobs)/thr,0)", kb, pool)
    t_star = np.log(2.0) / KDEG
    tail = np.exp(-KDEG * T_END)
    ramp = 1 - (pool / thr) * tail
    want = [
        kb * t_star / KDEG + (kb / KDEG**2) * ramp - (kb / KDEG) * (pool / thr) * T_END * tail,
        kb / (KDEG * thr) - (kb / KDEG) * (pool / thr**2) * tail,
        -kb / (KDEG * pool) + (kb / KDEG) * tail / thr,
        (T_END - t_star) - ramp / KDEG,
    ]
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-12)


def test_a_jump_of_a_few_ulp_of_its_own_rate_law_under_a_slow_threshold(tmp_path):
    """Control. ``ksyn + if(…, kb, 0)`` with ksyn = 1e14 and kb = 0.125, which is
    8 ulp of it, under a threshold species moving at 500. The drive tolerance,
    5e-4, reads the jump, and the rounding allowed the extended reading is never
    more than that: 8 ulp of the flux alone would have passed it over."""
    got = _columns(tmp_path, "ksyn+if(Bobs<thr,kb,0)", 0.125, 1e4, ksyn=1e14)
    np.testing.assert_allclose(got[:3], _step(0.125, 1e4)[:3], rtol=1e-6)


BESIDE_A_BYSTANDER = """begin parameters
    1 kb 3.0
    2 thr 5e7
    3 kdeg 0.1
    4 B0 1e8
    5 kA {kA!r}
    6 kdA {kdA!r}
    7 A0 1e7
    8 kdC {kdC!r}
    9 C0 3e6
    10 KA 2e6
end parameters
begin functions
    1 ramp() kb*(thr-Bobs)/thr
    2 fY() kA*Aobs*Cobs/(1+Aobs/KA)+if(Bobs<thr,ramp(),0)
end functions
begin species
    1 B() B0
    2 Y() 0
    3 A() A0
    4 C() C0
end species
begin reactions
    1 1 0 kdeg
    2 0 2 fY
    3 3 0 kdA
    4 4 0 kdC
end reactions
begin groups
    1 Bobs 1
    2 Aobs 3
    3 Cobs 4
end groups
"""


@pytest.mark.parametrize(
    ("kA", "kdA", "kdC"),
    [
        (0.85, 0.17, 0.66),
        (1.02, 0.83, 0.14),
        (1.65, 0.71, 0.26),
        (0.5, 0.43, 0.76),
        (0.56, 0.5, 0.06),
        (1.73, 0.95, 0.48),
    ],
)
def test_a_continuous_switch_beside_a_fast_bystander_in_the_same_rate_law(tmp_path, kA, kdA, kdC):
    """Control. The switched term vanishes at the surface, and the rate law also
    carries ``kA·A·C/(1 + A/KA)``, two species the residual does not read, large
    and moving. The reading extended to the root from two probes a side rounds
    by a few ulp of that: 1.9 to 4.2 times ε·|flux| in these six. No ulp of B
    accounts for it, so the extension is allowed a few ulp of its own readings.
    Allowed none, or one, it took the rounding for a jump: up to 1e-4 of
    dY/dkdeg here, and on the corpus two residuals that cross together (SIR_v5)
    each read one and the run was refused."""
    path = tmp_path / "m.net"
    path.write_text(BESIDE_A_BYSTANDER.format(kA=kA, kdA=kdA, kdC=kdC))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=PARAMS
    ).run(sample_times=[0.0, 5.0, 10.0, T_END], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]
    kb, pool = 3.0, 1e8
    thr = pool / 2
    t_star = np.log(2.0) / KDEG
    tail = np.exp(-KDEG * T_END)
    ramp = 1 - (pool / thr) * tail
    want = [
        kb * t_star / KDEG + (kb / KDEG**2) * ramp - (kb / KDEG) * (pool / thr) * T_END * tail,
        kb / (KDEG * thr) - (kb / KDEG) * (pool / thr**2) * tail,
        -kb / (KDEG * pool) + (kb / KDEG) * tail / thr,
        (T_END - t_star) - ramp / KDEG,
    ]
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-12)
