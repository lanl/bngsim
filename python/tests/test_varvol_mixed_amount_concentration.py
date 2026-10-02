"""A reaction that changes an amount and a concentration in one compartment whose
size changes.

An hOSU=true species is stored as amount/V_static and an hOSU=false one as its
live concentration, so their rows are law/V_static and law/V_live: no single
divide is both. The Elementary rate divided every row by the size at load, and
the one-reaction Functional emission every row by the live size. `H => B; k*H`
with H an amount and B a concentration in a growing compartment made B(6) 40.55
molecules out of 40 (RoadRunner 33.39); `H => B; 0.3*H` made H(6) 9.77 for
6.61. Of 54 such shapes against RoadRunner, 45 were wrong on main.

Oracles: closed forms. Each law below reads amounts, or a concentration whose
amount it scales by a known V(t), so the amounts solve exactly.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

GROW = "compartment c = 1; c' = 0.1;"  # V(t) = 1 + t/10
RESIZE = "compartment c = 1; E1: at time >= 2: c = 2.5;"
K = 0.3
T = np.array([0.0, 2.0, 4.0, 6.0])
H_TO_B = "substanceOnly species H in c = 40; species B in c = 0; k = 0.3; J1: H => B; {law};"


def _int_inv_v(vol, t):
    """∫0^t ds / V(s)."""
    if vol is GROW:
        return 10.0 * np.log1p(t / 10.0)
    return np.where(t < 2.0, t, 2.0 + (t - 2.0) / 2.5)


def _int_v(vol, t):
    """∫0^t V(s) ds."""
    if vol is GROW:
        return t + t**2 / 20.0
    return np.where(t < 2.0, t, 2.0 + 2.5 * (t - 2.0))


def _amounts(text, names):
    m = bngsim.Model.from_antimony_string(text)
    r = bngsim.Simulator(m).run(sample_times=list(T), rtol=1e-11, atol=1e-13)
    return np.asarray(r.as_roadrunner(names))


@pytest.mark.parametrize("vol", [GROW, RESIZE], ids=["rate-rule", "event-resize"])
@pytest.mark.parametrize("law", ["k*H", "0.3*H"])
def test_an_amount_turned_into_a_concentration(vol, law):
    text = (
        vol + f"substanceOnly species H in c = 40; species B in c = 0; k = {K}; J1: H => B; {law};"
    )
    h = 40.0 * np.exp(-K * T)
    np.testing.assert_allclose(
        _amounts(text, ["H", "B"]), np.c_[h, 40.0 - h], rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize("vol", [GROW, RESIZE], ids=["rate-rule", "event-resize"])
def test_an_amount_lost_at_a_rate_scaled_by_the_size(vol):
    text = (
        vol + f"substanceOnly species H in c = 40; species B in c = 0; k = {K}; J1: H => B; k*H*c;"
    )
    h = 40.0 * np.exp(-K * _int_v(vol, T))
    np.testing.assert_allclose(
        _amounts(text, ["H", "B"]), np.c_[h, 40.0 - h], rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize("vol", [GROW, RESIZE], ids=["rate-rule", "event-resize"])
def test_a_concentration_turned_into_an_amount(vol):
    """k*B is a rate of k·[B] = k·n_B/V(t), so n_B = 20·exp(-k∫ds/V)."""
    text = (
        vol + f"species B in c = 20; substanceOnly species H in c = 0; k = {K}; J1: B => H; k*B;"
    )
    b = 20.0 * np.exp(-K * _int_inv_v(vol, T))
    np.testing.assert_allclose(
        _amounts(text, ["B", "H"]), np.c_[b, 20.0 - b], rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize("vol", [GROW, RESIZE], ids=["rate-rule", "event-resize"])
def test_a_concentration_changed_beside_an_amount_valued_catalyst(vol):
    """G (an amount, 5) catalyses A -> B: a rate of k·[A]·5, so
    n_A = 20·exp(-5k∫ds/V). Both rows are concentrations, divided by the live
    size, but the Elementary rate divided them by the size at load."""
    text = vol + (
        "substanceOnly species G in c = 5; species A in c = 20; species B in c = 0;"
        f" k = {K}; J1: A => B; k*A*G;"
    )
    a = 20.0 * np.exp(-5.0 * K * _int_inv_v(vol, T))
    np.testing.assert_allclose(
        _amounts(text, ["A", "B"]), np.c_[a, 20.0 - a], rtol=1e-7, atol=1e-9
    )


def test_the_ssa_mean_of_an_amount_turned_into_a_concentration():
    text = (
        GROW + f"substanceOnly species H in c = 40; species B in c = 0; k = {K}; J1: H => B; k*H;"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ssa")
    reps = 800
    runs = []
    for i in range(reps):
        sim.model.reset()
        r = sim.run(sample_times=list(T), seed=300 + i)
        runs.append(np.asarray(r.as_roadrunner(["H", "B"])))
    x = np.array(runs)
    h = 40.0 * np.exp(-K * T)
    se = x.std(0, ddof=1) / np.sqrt(reps)
    assert np.all(np.abs(x.mean(0) - np.c_[h, 40.0 - h]) <= 4.5 * se + 1e-9)
