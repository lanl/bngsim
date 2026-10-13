"""GH #703: method='nf' and method='rm' start from the initial state at t_start.

Both network-free backends ignored t_start: NFsim stepped to absolute times and
labelled the t = 0 state as t_start, and RuleMonkey's stateless run simulated
[0, t_start] before its first row. A bngsim run starts from the initial state at
t_start and never sees [0, t_start], as ode and ssa do and as BNG2.pl's
simulate_nf does (it runs NFsim for t_end - t_start).

The two backends keep different clocks under that (issue #1022). NFsim runs on
elapsed time, its clock from 0 to t_end - t_start. RuleMonkey runs on absolute
model time, its clock from t_start to t_end, as ode and ssa do. This model does
not read the clock, so the two give the same answer here; a model that does is
in test_rulemonkey_model_clock.py.

A(s~0) -> A(s~1) at k = 0.5 from 20000 copies, so A0 = 20000*exp(-0.5 (t - t0)).
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
XML = DATA / "nfsim" / "first_order_switch.xml"


def _available(method: str) -> bool:
    if method == "nf":
        return bool(getattr(bngsim, "HAS_NFSIM", False))
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# Literal reasons, the ones the other network-free tests give: the skip audit
# (test_skip_audit.py) reads each reason from the source and cannot read an
# f-string's.
METHODS = [
    pytest.param(
        "nf", marks=pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
    ),
    pytest.param(
        "rm",
        marks=pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in"),
    ),
]


def _run(method: str, seed: int = 3, **times):
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    sim = bngsim.Simulator(m, method=method, xml_path=str(XML))
    r = sim.run(seed=seed, **times)
    return np.asarray(r.time), np.asarray(r.observables["A0"])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "times", [{"t_span": (5.0, 7.0), "n_points": 3}, {"sample_times": [5.0, 6.0, 7.0]}]
)
def test_a_late_start_begins_at_the_initial_state(method, times):
    t, a = _run(method, **times)
    np.testing.assert_array_equal(t, [5.0, 6.0, 7.0])
    assert a[0] == 20000.0
    # Within SSA noise (sd ~ 100) of the closed form; before the fix the rows
    # were ~1000 and ~600 (a decay from t = 0).
    np.testing.assert_allclose(a, 20000.0 * np.exp(-0.5 * (t - 5.0)), rtol=0.05)


@pytest.mark.parametrize("method", METHODS)
def test_a_late_start_is_the_same_run_shifted(method):
    """Same seed, same elapsed schedule: the counts are those of a run from
    t = 0, only the labels move.

    NFsim's clock is elapsed time in both runs, so its counts are identical.
    RuleMonkey spends the same draws in both, but an event lands at 5 + S in
    one and at S in the other, and rounding can put one of them on the other
    side of a sample time. That moves a row by one molecule at most, as
    RuleMonkey's own clock_origin_test allows.
    """
    _, a0 = _run(method, t_span=(0.0, 2.0), n_points=3)
    t5, a5 = _run(method, t_span=(5.0, 7.0), n_points=3)
    np.testing.assert_array_equal(t5, [5.0, 6.0, 7.0])
    if method == "nf":
        np.testing.assert_array_equal(a5, a0)
    else:
        assert np.abs(a5 - a0).max() <= 1.0


TFUN = DATA / "nfsim" / "tfun_new_format" / "valid_inline_step_time.xml"


def _mean_x_phos(method: str):
    seeds = range(1, 101)
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    sim = bngsim.Simulator(m, method=method, xml_path=str(TFUN))
    return np.mean(
        [
            np.asarray(sim.run(t_span=(1.0, 2.0), n_points=3, seed=s).observables["X_phos"])
            for s in seeds
        ],
        axis=0,
    )


@pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in")
def test_rulemonkey_reads_a_table_on_absolute_time():
    """Issue #1022. X(p~0) -> X(p~1) at 0.1 and back at a table of time that is
    0 until t = 1 and 10 until t = 2, from 100 X(p~0), over (1, 2). On absolute
    time the table reads 10 throughout, so X_phos sits near its balance,
    100*0.1/10.1 = 0.99: [0, 0.98, 0.99]."""
    mean = _mean_x_phos("rm")
    p = np.array([0.0, 0.0984, 0.0990]) / 10.0
    assert mean[0] == 0.0
    np.testing.assert_array_less(np.abs(mean - 100.0 * p), 5.0 * np.sqrt(p * (1 - p)) + 1e-9)


@pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
def test_nfsim_reads_a_table_on_elapsed_time():
    """The same run on NFsim, whose clock goes from 0 to 1, where the table
    reads 0: nothing comes back and X_phos climbs to 100*(1 - exp(-0.1)) = 9.5.
    That is BNG2.pl's simulate_nf, and it is the one place the two backends'
    clocks show. The bound is loose because NFsim reads a table only when a
    reaction fires (issue #892)."""
    mean = _mean_x_phos("nf")
    assert mean[0] == 0.0
    assert 3.5 < mean[1] < 6.5  # 4.9 expected; 0.98 on absolute time
    assert 7.5 < mean[2] < 11.5  # 9.5 expected; 0.99 on absolute time
