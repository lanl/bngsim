"""A small state-switch jump under a threshold species that moves fast (issue #917).

``fY = if(Bobs < thr, kb, 0)`` with B decaying from 1e8 through thr = 5e7. The
crossing was judged continuous where the jump was under 1e-6 of the rate that
drives it, the largest |dx/dt| among the species the residual reads. That rate
is kdeg·B = 5e6 here, so any jump under 5 was dropped: every column that moves
the crossing time came back 0, with no warning.

The drive is there for a switched flux that vanishes on both branches, the BNGL
signed-rate idiom: across a pair of probes it differs by its slope times the
crossing's speed. Where both branches are extended to the root from two probes
each, that is out of the reading already. What is left besides a step is the
rounding of the readings, the flux bending on its own side of the surface, and
what an ulp of its inputs moves it by. The extended reading is now held to a
bound on those three, and to the drive tolerance where that is the smaller, so
it is never held to less than it was.

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
    """``ksyn + if(…, kb, 0)`` with ksyn = 1e6: the flux neither bends nor moves
    with an ulp of anything it reads, so it is allowed a few ulp of 1e6 and no
    more, and the jump of 3 is read under a drive of 5e6."""
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
    branches, and the reading extended to the root is the rounding of thr − B:
    no saltation term.
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
    5e-4, reads the jump, and the extended reading is never allowed more than
    that: 16 ulp of the flux alone would have passed it over."""
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
    by a few ulp of that: 1.9 to 4.2 times ε·|flux| in these six. Allowed none,
    or one, an earlier cut took the rounding for a jump: up to 1e-4 of dY/dkdeg
    here, and on the corpus two residuals that cross together (SIR_v5) each read
    one and the run was refused."""
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


RISING = """begin parameters
    1 kb 3.0
    2 thr {thr!r}
    3 ksyn {ksyn!r}
    4 kdeg {kdeg!r}
end parameters
begin functions
    1 fY() {law}
end functions
begin species
    1 B() 0
    2 Y() 0
end species
begin reactions
    1 0 1 ksyn
    2 1 0 kdeg
    3 0 2 fY
end reactions
begin groups
    1 Bobs 1
end groups
"""


def _rising(tmp_path, law, kdeg, ksyn, thr):
    """B rises to ksyn/kdeg through thr at t*, and Y = kb·(T − t*) with
    T = t* + 2. Returns (got, want) for dY/dthr and dY/dksyn."""
    path = tmp_path / "m.net"
    path.write_text(RISING.format(law=law, kdeg=kdeg, ksyn=ksyn, thr=thr))
    phi = thr * kdeg / ksyn
    t_star = -np.log1p(-phi) / kdeg
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["thr", "ksyn"]
    ).run(sample_times=[0.0, float(t_star + 2.0)], rtol=1e-10, atol=1e-12, timeout=60.0)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]
    want = [-3.0 / (kdeg * (ksyn / kdeg - thr)), 3.0 * phi / (ksyn * kdeg * (1 - phi))]
    return got, want


@pytest.mark.parametrize(
    ("law", "kdeg", "ksyn", "thr"),
    [
        ("if(Bobs<thr,0,kb)", 1.985, 176593524.3915372, 85348456.16892558),
        ("if(Bobs>thr,kb,0)", 0.931, 45237233.33769633, 44736457.716640085),
        ("if(Bobs>thr,kb,0)", 1.108, 83264853.95991284, 71193081.05844033),
    ],
)
def test_a_jump_under_a_fast_rising_threshold_species(tmp_path, law, kdeg, ksyn, thr):
    """A pool that rises through the threshold at 4e6 to 7e6 a unit of time:
    dY/dthr and dY/dksyn were 0."""
    got, want = _rising(tmp_path, law, kdeg, ksyn, thr)
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize(
    ("kdeg", "ksyn", "thr"),
    [
        (0.443, 176.71588075440624, 398.8213733894357),
        (0.289, 11296406.124902554, 39079640.608187035),
        (1.273, 25.750487701783452, 20.223754145104092),
    ],
)
def test_a_jump_at_a_slow_crossing_under_a_plateau(tmp_path, kdeg, ksyn, thr):
    """Control. thr is 2e-4 short of the plateau, so the crossing is slow and
    the probes are a few ulp of B from the surface. An earlier cut allowed the
    reading what one ulp of B moves the flux by, read at a probe that the ulp
    carried across the surface: the allowance was the jump itself, and both
    columns came back 0."""
    got, want = _rising(tmp_path, "if(Bobs<thr,0,kb)", kdeg, ksyn, thr)
    np.testing.assert_allclose(got, want, rtol=1e-6)


BESIDE_A_DIFFERENCE = """begin parameters
    1 kb 3.0
    2 thr 5e7
    3 kdeg 0.1
    4 B0 1e8
    5 kbig {kbig!r}
    6 kdP {kdP!r}
    7 P0 {P0!r}
    8 Q0 1e12
    9 kdQ 0.2
end parameters
begin functions
    1 ramp() kb*(thr-Bobs)/thr
    2 fY() kbig*(Pobs-Qobs)+if(Bobs<thr,ramp(),0)
end functions
begin species
    1 B() B0
    2 Y() 0
    3 P() P0
    4 Q() Q0
end species
begin reactions
    1 1 0 kdeg
    2 0 2 fY
    3 3 0 kdP
    4 4 0 kdQ
end reactions
begin groups
    1 Bobs 1
    2 Pobs 3
    3 Qobs 4
end groups
"""


@pytest.mark.parametrize("kbig", [1e2, 1e4])
@pytest.mark.parametrize("kdP", [0.3, 0.43])
def test_a_continuous_switch_beside_a_difference_of_two_large_pools(tmp_path, kbig, kdP):
    """Control. The rate law also carries ``kbig·(P − Q)``, with P and Q at
    2.5e11 and a part in 1e9 apart at the crossing. That term rounds by an ulp
    of P, not by an ulp of its own value, so a bound of a few ulp of the
    readings is far too tight for it. An earlier cut read the rounding as a
    jump: dY/dkdeg up to 60% off at kbig = 1e4. The bound now counts what one
    ulp of every species and parameter moves the flux by."""
    t_star = np.log(2.0) / KDEG
    p0 = 1e12 * float(np.exp((kdP - 0.2) * t_star)) * (1 + 1e-9)
    path = tmp_path / "m.net"
    path.write_text(BESIDE_A_DIFFERENCE.format(kbig=kbig, kdP=kdP, P0=p0))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=PARAMS
    ).run(sample_times=[0.0, 5.0, 10.0, T_END], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]
    kb, pool = 3.0, 1e8
    thr = pool / 2
    tail = np.exp(-KDEG * T_END)
    ramp = 1 - (pool / thr) * tail
    want = [
        kb * t_star / KDEG + (kb / KDEG**2) * ramp - (kb / KDEG) * (pool / thr) * T_END * tail,
        kb / (KDEG * thr) - (kb / KDEG) * (pool / thr**2) * tail,
        -kb / (KDEG * pool) + (kb / KDEG) * tail / thr,
        (T_END - t_star) - ramp / KDEG,
    ]
    np.testing.assert_allclose(got, want, rtol=1e-6)


PAIR = """begin parameters
    1 k {k!r}
    2 thr1 {thr!r}
    3 thr2 {thr!r}
    4 kdeg {kdeg!r}
    5 B0 {B0!r}
    6 off {off!r}
end parameters
begin functions
{funcs}
end functions
begin species
    1 B1() B0
    2 B2() B0
    3 Y1() 0
    4 Y2() 0
end species
begin reactions
    1 1 0 kdeg
    2 2 0 kdeg
    3 0 3 fY1
    4 0 4 fY2
end reactions
begin groups
    1 B1obs 1
    2 B2obs 2
end groups
"""
QUADRATIC = (
    "    1 fY1() if(B1obs>thr1,k*(B1obs-thr1)^2,0)\n    2 fY2() if(B2obs>thr2,k*(B2obs-thr2)^2,0)"
)
OFFSET = (
    "    1 s1() B1obs-off\n    2 s2() B2obs-off\n"
    "    3 g1() k*(s1()+off-thr1)\n    4 g2() k*(s2()+off-thr2)\n"
    "    5 fY1() if(g1()>0,g1(),0)\n    6 fY2() if(g2()>0,g2(),0)"
)


def _pair(tmp_path, funcs, pool, frac, kdeg, k):
    """Two copies of one switch on thresholds of their own, crossing together.
    Returns d(Y1, Y2)/d(thr1, thr2) at twice the crossing time."""
    thr = pool * frac
    path = tmp_path / "m.net"
    path.write_text(PAIR.format(funcs=funcs, k=k, thr=thr, kdeg=kdeg, B0=pool, off=2 * pool))
    t_star = np.log(1 / frac) / kdeg
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["thr1", "thr2"]
    ).run(sample_times=[0.0, float(2 * t_star)], rtol=1e-10, atol=1e-12 * pool)
    names = list(run.species_names)
    rows = [names.index("Y1()"), names.index("Y2()")]
    return np.asarray(run.sensitivities)[-1][rows, :], thr, t_star


@pytest.mark.parametrize(
    ("pool", "frac", "kdeg", "k"),
    [(1.0, 0.5, 0.1, 1.0), (1e6, 0.3, 2.0, 1e-2), (40.0, 0.7, 0.03, 50.0)],
)
def test_two_switches_that_turn_off_with_zero_slope_cross_together(tmp_path, pool, frac, kdeg, k):
    """Control. ``if(B > thr, k·(B − thr)², 0)`` meets 0 with no slope, so a
    line through two probes on its own side leaves the curve's bend at the
    root. An earlier cut read that as a jump in each of the two, and refused
    the pair as two jumps on one instant. The bend is now read from a third
    probe a side. dY/dthr = −2k·((B0 − thr)/kdeg − thr·t*)."""
    got, thr, t_star = _pair(tmp_path, QUADRATIC, pool, frac, kdeg, k)
    own = -2 * k * ((pool - thr) / kdeg - thr * t_star)
    np.testing.assert_allclose(got, [[own, 0.0], [0.0, own]], rtol=1e-6, atol=1e-6 * abs(own))


@pytest.mark.parametrize(
    ("pool", "frac", "kdeg", "k"),
    [
        (16409.31242378578, 0.21299300190962633, 4.206269876778064, 133.13651639489908),
        (42948.961778762176, 0.29815289739334216, 0.15301427251275476, 0.04655553007911675),
        (17771858.43306985, 0.33796235705279576, 0.017460679088385594, 1.6913124945519817),
    ],
)
def test_two_ramps_written_through_an_offset_cross_together(tmp_path, pool, frac, kdeg, k):
    """Control. The BNGL signed-quantity idiom: ``s() = B − off`` and
    ``g() = k·(s() + off − thr)`` under ``if(g() > 0, g(), 0)``. The ramp is 0
    at the surface to the rounding of ``off``, which is not an ulp of B and not
    an ulp of g. An earlier cut read it as a jump and refused these three as
    two jumps on one instant. dY/dthr = −k·t*."""
    got, _thr, t_star = _pair(tmp_path, OFFSET, pool, frac, kdeg, k)
    own = -k * t_star
    np.testing.assert_allclose(got, [[own, 0.0], [0.0, own]], rtol=1e-6, atol=1e-6 * abs(own))
