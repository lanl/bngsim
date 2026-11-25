"""method="nfsim" and its total propensity (issues #1666, #1672).

NFsim's selector kept the total propensity as a running sum, ``a_tot - old +
new``, and summed it afresh only where a pick ran past the last reaction. A
propensity added to a large total is rounded to that total's spacing, and is
not given back when the large term is taken off again. With
``A() + B() -> C()`` at kf from 1000 of each beside ``0 -> D()`` at ks = 1e-6,
the total starts at kf*1e6 + ks. Once A and B were gone it held 0 at kf = 1e5,
so D never fired, and 1.907e-6 at kf = 1e4, so D fired at 1.9 times its rate.
D is born at a rate that reads nothing, so D(t) is Poisson with mean ks*t
whatever kf is. The total is now summed afresh once it falls far below the
largest value it has held since it was last summed, and every 4096 updates,
which bounds the rounding where the total never falls far.

``stepTo`` also fired nothing while the total was 1e-9 or less, a threshold in
the model's own time unit: ``0 -> D()`` at 5e-10 gave D = 0 for a mean of 20,
and 100 A decaying at 1e-10 stopped at exactly 10, where the total is 1e-9.
With the total kept close to the sum of the propensities, any total above 0
fires.

A total that is not a number is the model's own, and is still refused (issues
#1818, #1788): it is never summed away, nor read as a total of 0.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import NfsimSession


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


pytestmark = pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")

NFSIM_DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
SLOW_BESIDE_FAST = NFSIM_DATA / "slow_channel_beside_fast.xml"
TINY_TOTAL = NFSIM_DATA / "tiny_total_propensity.xml"
NAN_AT_EMPTYING = NFSIM_DATA / "nan_once_the_fast_channel_empties.xml"

SEEDS = range(1, 41)


def _run(xml: Path, params: dict[str, float], t_end: float, n_points: int, seed: int):
    with NfsimSession(str(xml)) as nf:
        for name, value in params.items():
            nf.set_param(name, value)
        nf.initialize(seed=seed)
        return nf.simulate(0.0, t_end, n_points)


def _finals(xml: Path, params: dict[str, float], t_end: float, observable: str) -> list[float]:
    """*observable* at t_end, for each of SEEDS."""
    return [
        float(np.asarray(_run(xml, params, t_end, 3, seed).observables[observable])[-1])
        for seed in SEEDS
    ]


# ─── A slow channel beside a fast one (#1666) ─────────────────────────────────


@pytest.mark.parametrize("kf", [1e5, 1e4, 1.0])
def test_a_slow_channel_keeps_its_rate_once_a_fast_one_empties(kf: float) -> None:
    # D(2e7) ~ Poisson(20) at every kf, a standard error of 0.71 over 40 seeds.
    # Before: 0.00 at kf = 1e5 and 38.8 at kf = 1e4. kf = 1 was right then too.
    finals = _finals(SLOW_BESIDE_FAST, {"kf": kf}, 2e7, "D")
    assert 15.0 < np.mean(finals) < 25.0, np.mean(finals)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_after_the_fast_channel_empties_a_seed_runs_the_same_at_every_kf(seed: int) -> None:
    # Whatever kf, the fast channel fires 1000 times, taking the same draws, and
    # leaves the same state. From there D is born at ks alone, so with ks as the
    # total a seed gives the same D at every row. Before, the total left for ks
    # = 1e-6 was 0 at kf = 1e5, 1.907e-6 at 1e4, 0.954e-6 at 1e3 and 0.998e-6
    # at 1e2.
    rows = {
        kf: np.asarray(_run(SLOW_BESIDE_FAST, {"kf": kf}, 2e7, 21, seed).observables["D"])
        for kf in (1e5, 1e4, 1e3, 1e2)
    }
    assert rows[1e2][-1] > 0
    for kf in (1e5, 1e4, 1e3):
        np.testing.assert_array_equal(rows[kf], rows[1e2], err_msg=f"kf = {kf}")


def test_the_fast_channel_has_emptied_when_the_slow_one_is_read() -> None:
    # The premise of the test above: by t = 1e3 every A has gone.
    result = _run(SLOW_BESIDE_FAST, {"kf": 1e4}, 1e3, 2, seed=1)
    assert np.asarray(result.observables["A"]).tolist() == [1000.0, 0.0]


# ─── A total of 1e-9 or less (#1672) ──────────────────────────────────────────


@pytest.mark.parametrize("ks", [5e-10, 1e-9])
def test_a_total_propensity_of_1e_9_or_less_still_fires(ks: float) -> None:
    # D(20/ks) ~ Poisson(20). Before: D = 0 in every seed.
    finals = _finals(TINY_TOTAL, {"ks": ks, "k": 0.0}, 20 / ks, "D")
    assert 15.0 < np.mean(finals) < 25.0, np.mean(finals)


def test_a_decay_goes_on_past_a_total_of_1e_9() -> None:
    # 100 A at k = 1e-10 to t = 5e10: A ~ Binomial(100, exp(-5)), mean 0.674,
    # a standard error of 0.13 over 40 seeds, and 10 or more in about 3e-9 of
    # runs. Before: exactly 10 in every seed, where the total is 1e-9.
    finals = _finals(TINY_TOTAL, {"ks": 0.0, "k": 1e-10}, 5e10, "A")
    assert max(finals) < 10, finals
    assert np.mean(finals) == pytest.approx(100 * math.exp(-5), abs=0.6)


def test_a_run_is_the_same_in_a_longer_time_unit() -> None:
    # Every rate scaled by 2**-40 and every time by 2**40 is the same run: a
    # power of two scales each propensity, their total, each waiting time and
    # the clock without rounding, so a seed fires the same reactions at the
    # scaled times. Scaled, the total starts at 9.3e-12; before, nothing fired
    # there, and A stayed at 100 on every row.
    observed = []
    for scale in (1.0, 2.0**-40):
        result = _run(TINY_TOTAL, {"ks": 0.2 * scale, "k": 0.1 * scale}, 30.0 / scale, 31, seed=7)
        observed.append({name: np.asarray(result.observables[name]) for name in ("A", "D")})
    unscaled, scaled = observed
    assert unscaled["A"][-1] < 10 and unscaled["D"][-1] > 0
    np.testing.assert_array_equal(scaled["A"], unscaled["A"])
    np.testing.assert_array_equal(scaled["D"], unscaled["D"])


# ─── A total that is not a number is still refused ────────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_nan_from_the_firing_that_empties_a_fast_channel_is_refused(seed: int) -> None:
    # D is born at ks*B/B, NaN from the firing that takes the last B. That
    # firing also takes the fast channel to 0, a fall that has the total summed
    # afresh where it is next read for a draw. The NaN is kept there, and
    # reaches the check that refuses it.
    with NfsimSession(str(NAN_AT_EMPTYING)) as nf:
        nf.initialize(seed=seed)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan") as err:
            nf.simulate(0.0, 2e7, 3)
    assert "Non-finite propensity in 1 reaction(s): '_R2'" in str(err.value)
    assert "NFsim fires nothing once it is" in str(err.value)


def test_the_one_shot_run_refuses_that_nan_too() -> None:
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 2e7, 3
    with pytest.raises(RuntimeError, match="NfsimSimulator::run: NFsim's total propensity is nan"):
        NfsimSimulator(str(NAN_AT_EMPTYING)).run(times, 1)


def test_the_nan_rate_is_the_slow_rate_until_then() -> None:
    # The control: before the last B goes, ks*B/B is ks, and nothing is refused.
    result = _run(NAN_AT_EMPTYING, {}, 1e-7, 2, seed=1)
    assert np.asarray(result.observables["B"])[-1] > 0
