"""SSA/PSA reaction selection keeps a slow channel's rate, and refuses a
non-finite propensity (issues #713, #809).

#713: the selection structure was a Fenwick tree whose nodes were running sums
adjusted by deltas and never re-summed. A large propensity that shared a node
with a small one kept its rounding in the node; once the large one decayed, the
small channel's slice was that residue (1.1e-16 instead of 1e-6), and the
channel never fired again, with no warning. Oracle: a zeroth-order birth
``0 -> D`` at rate ``ks`` is independent of the fast channel, so
``E[D(T)] = ks·T`` exactly, and D(T) is Poisson.

#809: a NaN propensity failed the ``a0 <= 0`` stuck test and gave a NaN waiting
time, so SSA spun forever and PSA returned the initial state as its trajectory.
Both now raise ``SimulationError`` naming the reaction and the time.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import SimulationError


def _net(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _drift_net(kf, ks, *, functional):
    """``A + B -> C`` at kf from 1e6 each (a propensity of kf·1e12 that decays
    to 0), beside ``0 -> D`` at ks. A functional rate for the birth keeps the
    run off the compiled mass-action fast loop, on the selection tree."""
    rate = "ksf" if functional else "ks"
    funcs = "begin functions\n    1 ksf() ks\nend functions\n" if functional else ""
    return (
        f"begin parameters\n    1 kf {kf}\n    2 ks {ks}\nend parameters\n{funcs}"
        "begin species\n    1 A() 1e6\n    2 B() 1e6\n    3 C() 0\n    4 D() 0\nend species\n"
        f"begin reactions\n    1 1,2 3 kf\n    2 0 4 {rate}\nend reactions\n"
        "begin groups\n    1 Dtot 4\nend groups\n"
    )


SELECTION = [
    pytest.param({"functional": True}, {}, id="tree-functional"),
    pytest.param({"functional": False}, {"BNGSIM_SSA_NO_CODEGEN": "1"}, id="tree-elementary"),
    pytest.param({"functional": True}, {"BNGSIM_SSA_SELECT": "flat"}, id="flat"),
]


@pytest.mark.parametrize(("shape", "env"), SELECTION)
def test_slow_channel_survives_a_decayed_fast_one(tmp_path, monkeypatch, shape, env):
    """kf·1e12 = 3.7e11 against ks = 1e-6: a ratio past 1e17. The old tree
    lost the slow channel outright; every replicate reported D = 0, where
    P(D(2e7) = 0) = e^-20 for one replicate."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    m = _net(tmp_path, "drift.net", _drift_net(0.37, 1e-6, **shape))
    reps = 20
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 2e7), n_points=3, seed=5, squeeze=True
    )
    d = np.asarray(r.observables)[:, :, 0]
    assert (d[:, -1] > 0).all(), d[:, -1]
    for col, t in ((1, 1e7), (2, 2e7)):
        mean, want = d[:, col].mean(), 1e-6 * t
        se = np.sqrt(want / reps)  # Poisson
        assert abs(mean - want) <= 4.5 * se, (t, mean, want)


def test_slow_channel_rate_is_not_biased(tmp_path):
    """The partial-loss regime: kf = 0.3 against ks = 1e-3 leaves the slow
    channel a quantised residue rather than none, and the old tree gave
    E[D(1e8)] = 97737 against an exact 1e5 (-2.3%, about 20 standard errors)."""
    m = _net(tmp_path, "partial.net", _drift_net(0.3, 1e-3, functional=True))
    reps = 8
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 1e8), n_points=2, seed=3, squeeze=True
    )
    d = np.asarray(r.observables)[:, -1, 0]
    want = 1e-3 * 1e8
    se = np.sqrt(want / reps)
    assert abs(d.mean() - want) <= 4.5 * se, (d.mean(), want, se)


# ── #809: a non-finite propensity is refused ────────────────────────────────

NAN_NET = """begin parameters
    1 k -1.0
    2 d sqrt(k)
end parameters
begin functions
    1 H() 0.1*d
end functions
begin species
    1 A() 10
end species
begin reactions
    1 1 0 H
end reactions
begin groups
    1 Atot 1
end groups
"""

MASS_ACTION_NET = """begin parameters
    1 k 1.0
    2 k2 1.0
end parameters
begin species
    1 A() 10
    2 B() 0
end species
begin reactions
    1 1 2 k
    2 2 1 k2
end reactions
begin groups
    1 Atot 1
end groups
"""

METHODS = [
    pytest.param("ssa", {}, id="ssa"),
    pytest.param("ssa", {"codegen": False}, id="ssa-interpreted"),
    pytest.param("psa", {"poplevel": 100}, id="psa"),
]


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_nan_rate_law_is_refused(tmp_path, method, kw):
    """``sqrt(-1)`` in a function rate: SSA used to hang and PSA to return
    ``A = 10`` throughout. The ODE path refuses the same model."""
    m = _net(tmp_path, "nan.net", NAN_NET)
    with pytest.raises(SimulationError, match=r"reaction R1 \(A\(\) -> 0\) is nan at t = 0"):
        bngsim.Simulator(m, method=method, **kw).run((0.0, 1.0), 3, seed=1, timeout=30)


@pytest.mark.parametrize(("method", "kw"), METHODS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_rate_constant_is_refused(tmp_path, method, kw, bad):
    """A mass-action model takes the compiled fast loop by default; a scan or a
    fit that writes a NaN or infinite rate constant must be refused there too."""
    m = _net(tmp_path, "ma.net", MASS_ACTION_NET)
    m.set_param("k", bad)
    with pytest.raises(SimulationError, match=r"reaction R1 \(A\(\) -> B\(\)\) is (nan|inf)"):
        bngsim.Simulator(m, method=method, **kw).run((0.0, 1.0), 3, seed=1, timeout=30)


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_an_overflowing_total_is_refused(tmp_path, method, kw):
    """Two finite propensities of 1e308 sum to infinity; the total is refused,
    not sampled from."""
    m = _net(
        tmp_path,
        "ma.net",
        MASS_ACTION_NET.replace("1 A() 10", "1 A() 1").replace("2 B() 0", "2 B() 1"),
    )
    m.set_param("k", 1.5e308)
    m.set_param("k2", 1.5e308)
    with pytest.raises(SimulationError, match="total propensity is inf"):
        bngsim.Simulator(m, method=method, **kw).run((0.0, 1.0), 3, seed=1, timeout=30)
