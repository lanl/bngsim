"""A small state-switch jump under a threshold species that moves fast (issue #917).

``fY = if(Bobs < thr, kb, 0)`` with B decaying from 1e8 through thr = 5e7. The
crossing was judged continuous where the jump was under 1e-6 of the rate that
drives it, the largest |dx/dt| among the species the residual reads. That rate
is kdeg·B = 5e6 here, so any jump under 5 was dropped: every column that moves
the crossing time came back 0, with no warning.

The drive is there for a switched flux that vanishes on both branches, the BNGL
signed-rate idiom: across a pair of probes it differs by its slope times the
crossing's speed. Under that tolerance the two branches are now read at one
state, the state on the surface with only the species the condition reads moved
a few ulp to either side. Every term that does not switch is the same in both
readings, so what is left is the step, or the rounding of the two readings and
what the flux does on its own over those few ulp.

The same reading takes back a jump that is not the switch's: a term beside a
continuous switch that rounds as a staircase steps between the probes.

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
    """``ksyn + if(…, kb, 0)`` with ksyn = 1e6: at one state the constant is the
    same on both sides of the surface, so the two readings differ by the jump of
    3 and are allowed a few ulp of 1e6. It is read under a drive of 5e6."""
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
    by a few ulp of that: 1.9 to 4.2 times ε·|flux| in these six, which is under
    the 16 it is allowed, so the crossing is continuous before the two branches
    are read at one state at all. Allowed one, an earlier cut took the rounding
    for a jump: up to 1e-4 of dY/dkdeg here, and on the corpus two residuals
    that cross together (SIR_v5) each read one and the run was refused."""
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
    the probes along the flow are a few ulp of B from the surface. The drive is
    small here, and the jump is past its tolerance: it is read as it was. An
    earlier cut measured its allowance at a probe that one ulp carried across
    the surface, so the allowance was the jump itself, and both columns came
    back 0."""
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
    of P, not by an ulp of its own value, so between two probes along the flow
    it moves by far more than a few ulp of the readings. An earlier cut read
    that as a jump: dY/dkdeg up to 60% off at kbig = 1e4. At one state the term
    reads the same on both sides of the surface, to the bit."""
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
    7 ksyn {ksyn!r}
end parameters
begin functions
{funcs}
end functions
begin species
    1 B1() B0
    2 B2() B0
    3 Y1() 0
    4 Y2() 0
    5 C() off
end species
begin reactions
    1 1 0 kdeg
    2 2 0 kdeg
    3 0 3 fY1
    4 0 4 fY2
    5 0 1 ksyn
    6 0 2 ksyn
end reactions
begin groups
    1 B1obs 1
    2 B2obs 2
    3 Cobs 5
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
# The same, with the offset a species that nothing moves.
OFFSET_SPECIES = OFFSET.replace("-off", "-Cobs").replace("+off", "+Cobs")


def _pair(tmp_path, funcs, pool, frac, kdeg, k, rising=False, off=2.0):
    """Two copies of one switch on thresholds of their own, crossing together.
    B falls from ``pool``, or rises to it from 0, and the offset is ``off``
    pools. Returns d(Y1, Y2)/d(thr1, thr2) at twice the crossing time, with the
    threshold and that time."""
    thr = pool * frac
    path = tmp_path / "m.net"
    path.write_text(
        PAIR.format(
            funcs=funcs,
            k=k,
            thr=thr,
            kdeg=kdeg,
            B0=0.0 if rising else pool,
            off=off * pool,
            ksyn=pool * kdeg if rising else 0.0,
        )
    )
    t_star = (-np.log1p(-frac) if rising else np.log(1 / frac)) / kdeg
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
    the pair as two jumps on one instant. A few ulp either side of the surface
    the law is a few ulp squared, and what it does as far again out is allowed.
    dY/dthr = −2k·((B0 − thr)/kdeg − thr·t*)."""
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


def _risen(pool, thr, kdeg, t_star):
    """∫(B − thr) dt from t* to 2·t*, for B = pool·(1 − e^(−kdeg·t))."""
    return (pool - thr) * t_star - (pool / kdeg) * (
        np.exp(-kdeg * t_star) - np.exp(-2 * kdeg * t_star)
    )


@pytest.mark.parametrize(
    ("pool", "frac", "kdeg", "k"), [(1.0, 0.5, 0.1, 1.0), (1e6, 0.3, 2.0, 1e-2)]
)
def test_two_switches_that_turn_on_with_zero_slope_cross_together(tmp_path, pool, frac, kdeg, k):
    """Control. B rises through thr, so the law that meets 0 with no slope is on
    the far side of the crossing, and what it does on its own is read on that
    side too. dY/dthr = −2k·∫(B − thr) dt from t* on."""
    got, thr, t_star = _pair(tmp_path, QUADRATIC, pool, frac, kdeg, k, rising=True)
    own = -2 * k * _risen(pool, thr, kdeg, t_star)
    np.testing.assert_allclose(got, [[own, 0.0], [0.0, own]], rtol=1e-6, atol=1e-6 * abs(own))


@pytest.mark.parametrize(
    ("pool", "frac", "kdeg", "k"),
    [
        (16409.31242378578, 0.21299300190962633, 4.206269876778064, 133.13651639489908),
        (42948.961778762176, 0.29815289739334216, 0.15301427251275476, 0.04655553007911675),
        (17771858.43306985, 0.33796235705279576, 0.017460679088385594, 1.6913124945519817),
    ],
)
@pytest.mark.parametrize("funcs", [OFFSET, OFFSET_SPECIES], ids=["parameter", "species"])
def test_two_ramps_through_an_offset_that_turn_on_cross_together(
    tmp_path, funcs, pool, frac, kdeg, k
):
    """Control. B rises, so the ramp that rounds by its offset is on the far
    side. The offset is a parameter, or a species that nothing moves. It is two
    million pools, so the ramp is a staircase with treads far wider than the
    probes are apart, and wider than the few ulp of B the two sides are read
    at: the side that is on reads one tread, which is followed out to where the
    ramp moves again. dY/dthr = −k·t*."""
    got, _thr, t_star = _pair(tmp_path, funcs, pool, frac, kdeg, k, rising=True, off=2e6)
    own = -k * t_star
    np.testing.assert_allclose(got, [[own, 0.0], [0.0, own]], rtol=1e-6, atol=1e-6 * abs(own))


@pytest.mark.parametrize(
    ("pool", "frac", "kdeg", "k"),
    [
        (16409.31242378578, 0.21299300190962633, 4.206269876778064, 133.13651639489908),
        (42948.961778762176, 0.29815289739334216, 0.15301427251275476, 0.04655553007911675),
        (17771858.43306985, 0.33796235705279576, 0.017460679088385594, 1.6913124945519817),
    ],
)
def test_two_ramps_through_a_species_offset_cross_together(tmp_path, pool, frac, kdeg, k):
    """Control. The falling pair, with the offset a species of two million
    pools."""
    got, _thr, t_star = _pair(tmp_path, OFFSET_SPECIES, pool, frac, kdeg, k, off=2e6)
    own = -k * t_star
    np.testing.assert_allclose(got, [[own, 0.0], [0.0, own]], rtol=1e-6, atol=1e-6 * abs(own))


JUMP_BESIDE_A_BYSTANDER = """begin parameters
    1 kb {kb!r}
    2 thr 5e7
    3 kdeg 0.1
    4 B0 1e8
    5 kA 3.0
    6 kdA 0.17
    7 A0 1e7
    8 kdC 0.66
    9 C0 3e6
end parameters
begin functions
    1 fY() kA*Aobs*Cobs+if(Bobs<thr,kb,0)
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


@pytest.mark.parametrize("kb", [3.0, 0.3])
def test_a_jump_beside_a_large_term_that_moves(tmp_path, kb):
    """The rate law also carries ``kA·A·C``, 2.9e11 at the crossing and falling
    by 3e-13 of itself from one probe to the next. The jump is 49,000 ulp of
    that, or 4,900, and both were dropped. It is read to the rounding of the
    readings it is a difference of, about two ulp of the term: 5e-5 of the
    jump, or 5e-4.

    The term moves from probe to probe by more than the smaller jump. It does
    not read the threshold species, so at one state it is the same on both
    sides of the surface."""
    path = tmp_path / "m.net"
    path.write_text(JUMP_BESIDE_A_BYSTANDER.format(kb=kb))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["kdeg", "thr", "B0"]
    ).run(sample_times=[0.0, 5.0, 10.0, T_END], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]
    t_star = np.log(2.0) / KDEG
    want = [kb * t_star / KDEG, kb / (KDEG * 5e7), -kb / (KDEG * 1e8)]
    np.testing.assert_allclose(got, want, rtol=6e-4 / kb)


BESIDE_A_STAIRCASE = """begin parameters
    1 kb {kb!r}
    2 thr 5e7
    3 kdeg 0.1
    4 B0 1e8
    5 kbig {kbig!r}
    6 kdP {kdP!r}
    7 P0 {P0!r}
    8 Q0 {Q0!r}
    9 kdQ 0.2
    10 off {off!r}
end parameters
begin functions
    1 ramp() kb*(thr-Bobs)/thr
    2 sP() Pobs-off
    3 fY() {beside}+if(Bobs<thr,ramp(),0)
end functions
begin species
    1 B() B0
    2 Y() 0
    3 P() P0
    4 Q() Q0
    5 D() off
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
    4 PD 3,5
    5 QD 4,5
end groups
"""

# kb, kbig, kdP, Q0, the offset over the pools, how far apart the pools are at
# the crossing, and the term beside the ramp.
STAIRCASES = {
    "through-a-function": (
        0.07654195096766644,
        3759.1379884560324,
        0.3536612938510687,
        77985043982.31714,
        1351.7330454198013,
        2.615418575883961e-07,
        "kbig*(sP()+off-Qobs)",
    ),
    "through-a-species": (
        1.2224545360710726,
        27218.08064422743,
        0.4,
        2.59e11,
        616.7022937579026,
        1.9245609074859466e-07,
        "kbig*(PD-QD)",
    ),
    "two-pools": (
        1.4811783897186779,
        740992.6140522596,
        0.3,
        9.13e10,
        1.0,
        1.7071886536920877e-07,
        "kbig*(Pobs-Qobs)",
    ),
}


def _beside_a_staircase(tmp_path, case, kb, switched, thr=5e7):
    """dY/d(kdeg, thr, B0) with `switched` as the law the condition turns on."""
    _, kbig, kdP, Q0, over, gap, beside = STAIRCASES[case]
    t_star = np.log(1e8 / thr) / KDEG
    pools = Q0 * np.exp(-0.2 * t_star)
    text = BESIDE_A_STAIRCASE.format(
        kb=kb,
        kbig=kbig,
        kdP=kdP,
        P0=Q0 * float(np.exp((kdP - 0.2) * t_star)) * (1 + gap),
        Q0=Q0,
        off=float(over * pools),
        beside=beside,
    ).replace("if(Bobs<thr,ramp(),0)", f"if(Bobs<thr,{switched},0)")
    path = tmp_path / "m.net"
    path.write_text(text)
    model = bngsim.Model.from_net(path)
    model.set_param("thr", thr)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["kdeg", "thr", "B0"]).run(
        sample_times=[0.0, 5.0, 10.0, T_END], rtol=1e-10, atol=1e-12
    )
    return np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]


@pytest.mark.parametrize("kb", [0.3, 3.0, 30.0])
@pytest.mark.parametrize("case", sorted(STAIRCASES))
def test_a_jump_beside_a_term_that_rounds_as_a_staircase(tmp_path, case, kb):
    """The same staircase beside a switch that does jump, by kb. The jump was
    read, and sized from everything the right-hand side did between the probes,
    the staircase's tread included: with kb = 3 and a tread of 15, dY/dkdeg came
    back −1826 for 207.9. Read at one state, the tread is the same on both sides
    and the jump is kb."""
    got = _beside_a_staircase(tmp_path, case, kb, "kb")
    t_star = np.log(2.0) / KDEG
    want = [kb * t_star / KDEG, kb / (KDEG * 5e7), -kb / (KDEG * 1e8)]
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize("case", sorted(STAIRCASES))
def test_a_continuous_switch_beside_a_term_that_rounds_as_a_staircase(tmp_path, case):
    """``kbig·(P − Q)`` with P and Q a part in 1e7 apart, written through an
    offset a thousand times their size, moves in treads of an ulp of the offset:
    15, 213 and 3 in these, under a drive of 5e6 and a tolerance of 5. It does not
    read the switch. Between two probes it stepped by a tread, which was read as
    the switch's jump: dY/dkdeg came back −2029.6 for 4.42. The switch is a ramp
    that vanishes at the surface, and at one state the term beside it is the
    same on both sides."""
    kb = STAIRCASES[case][0]
    t_star = np.log(2.0) / KDEG
    got = _beside_a_staircase(tmp_path, case, kb, "ramp()")
    pool, thr, tail = 1e8, 5e7, np.exp(-KDEG * T_END)
    want = [
        kb * t_star / KDEG
        + (kb / KDEG**2) * (1 - (pool / thr) * tail)
        - (kb / KDEG) * (pool / thr) * T_END * tail,
        kb / (KDEG * thr) - (kb / KDEG) * (pool / thr**2) * tail,
        -kb / (KDEG * pool) + (kb / KDEG) * tail / thr,
    ]
    np.testing.assert_allclose(got, want, rtol=1e-6)


STEEP = """begin parameters
    1 kb 0.06
    2 thr 5e7
    3 kdeg 1e-3
    4 B0 1e8
    5 kbig 1e5
end parameters
begin functions
    1 fY() kbig*Bobs+if(Bobs<thr,kb,0)
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


def test_a_jump_of_eighty_ulp_of_a_steep_term_that_reads_the_threshold_species(tmp_path):
    """Control. ``kbig·B + if(B < thr, kb, 0)`` with kb at 80 ulp of kbig·B and
    just over the drive tolerance, so it is a jump as it always was. The steep
    term moves between the two sides of the surface by 32 of its ulp and more,
    which takes most of the jump out of their difference as it stands: read that
    way alone, the jump was taken back and dY/dthr came back 0. Carried to the
    surface along each side's slope it is 80 ulp again. The column is the jump
    to the 3% that 80 ulp can be read to."""
    path = tmp_path / "m.net"
    path.write_text(STEEP)
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["thr"]
    ).run(sample_times=[0.0, 500.0, 1500.0], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), 0]
    # dY/dthr = kbig·dB-integral's part is 0 (the smooth term does not read thr)
    # plus the jump's kb/(kdeg·thr).
    assert got == pytest.approx(0.06 / (1e-3 * 5e7), rel=0.05)


SPECIES_THRESHOLD = """begin parameters
    1 kb {kb!r}
    2 C0 5e7
    3 kdeg 0.1
    4 B0 1e8
    5 off {off!r}
end parameters
begin functions
    1 sB() Bobs-off
    2 fY() {law}
end functions
begin species
    1 B() B0
    2 Y() 0
    3 Cst() C0
end species
begin reactions
    1 1 0 kdeg
    2 0 2 fY
end reactions
begin groups
    1 Bobs 1
    2 Cobs 3
end groups
"""


@pytest.mark.parametrize("kb", [3.0, 3e-3])
@pytest.mark.parametrize(
    ("law", "off"),
    [("if(Bobs<Cobs,kb,0)", 0.0), ("if(sB()+off<Cobs,kb,0)", 1e10)],
    ids=["a-species-threshold", "through-an-offset"],
)
def test_a_jump_under_a_residual_that_reads_two_species(tmp_path, law, off, kb):
    """The threshold is a species that does not move, so the residual reads two
    species, one on each side of it, and they move it opposite ways: to put the
    state on either side of the surface each is moved its own way. Written
    through an offset a hundred times the pool, the residual moves in treads of
    128 ulp of B, and the way each species moves it shows only over a step wider
    than a tread. dY/dkdeg and dY/dB0 were 0, as under a parameter threshold."""
    path = tmp_path / "m.net"
    path.write_text(SPECIES_THRESHOLD.format(kb=kb, law=law, off=off))
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["kdeg", "B0", "kb"]
    ).run(sample_times=[0.0, 5.0, 10.0, T_END], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]
    t_star = np.log(2.0) / KDEG
    np.testing.assert_allclose(
        got, [kb * t_star / KDEG, -kb / (KDEG * 1e8), T_END - t_star], rtol=1e-6
    )


STEEP_STAIRCASE = """begin parameters
    1 kb 0.18369935312979419
    2 thr 334184.8555261631
    3 kdeg 0.49422710341294795
    4 B0 469545.4943152115
    5 kbig 5746070.828948309
    6 off 17431975.458342686
    7 c0 333850.670670637
end parameters
begin functions
    1 fY() kbig*((Bobs-off)+off-c0)+if(Bobs<thr,kb,0)
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


def test_a_jump_just_past_the_tolerance_beside_a_staircase_that_reads_the_threshold(tmp_path):
    """Control. The term beside the jump is a staircase in B itself, with a
    tread of 0.13 of the drive tolerance, and the jump is 1.11 of it: a jump as
    it always was. A boundary between treads falls between the two sides of the
    surface, so their difference is the jump less a tread. A cut that allowed
    that reading a tread took the jump back, and dY/dthr came back 0 where it
    is 1.112e-6 and main is within 0.3% of that."""
    path = tmp_path / "m.net"
    path.write_text(STEEP_STAIRCASE)
    t_end = 2 * np.log(469545.4943152115 / 334184.8555261631) / 0.49422710341294795
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["thr"]
    ).run(sample_times=[0.0, 0.4 * t_end, t_end], rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), 0]
    want = 0.18369935312979419 / (0.49422710341294795 * 334184.8555261631)
    assert got == pytest.approx(want, rel=1e-2)


TWO_JUMPS = """begin parameters
    1 thr 5e7
    2 thr2 {thr2!r}
    3 kdeg 0.1
    4 B0 1e8
end parameters
begin functions
    1 fY() if(Bobs<thr,3,0)+if(Bobs<thr2,5,0)
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


@pytest.mark.parametrize("gap", [-30, 17, 30, 44, 48])
def test_two_jumps_of_one_rate_law_a_few_ulp_apart_are_still_refused(tmp_path, gap):
    """Control. Two thresholds 17 to 48 ulp of B apart, each with a jump under
    the drive tolerance and the two together over it. The pair is refused, as
    two crossings on one instant that the columns move apart. Read at one
    state, each switch has the other's jump just beyond its two sides, and a
    cut that allowed a reading what the flux does further out took both jumps
    back: (0, 0) for (6e-7, 1e-6)."""
    path = tmp_path / "m.net"
    path.write_text(TWO_JUMPS.format(thr2=float(5e7 + gap * np.spacing(5e7))))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["thr", "thr2"]
    )
    with pytest.raises(Exception, match="cross at the same instant"):
        sim.run(sample_times=[0.0, 5.0, 10.0, T_END], rtol=1e-10, atol=1e-12)
