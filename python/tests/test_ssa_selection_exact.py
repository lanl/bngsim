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
    pytest.param({"functional": True, "method": "psa"}, {}, id="tree-psa"),
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
    shape = dict(shape)
    method = shape.pop("method", "ssa")
    kw = {"poplevel": 100} if method == "psa" else {}
    m = _net(tmp_path, "drift.net", _drift_net(0.37, 1e-6, **shape))
    reps = 20
    r = bngsim.Simulator(m, method=method, **kw).run_replicates(
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


def test_slow_channel_survives_deep_in_the_tree(tmp_path):
    """17 reactions make a three-level tree: the fast channel at leaf 1 and the
    slow birth at leaf 5 meet only at the root, so the recompute has to carry
    the decay through every level. (The two-reaction model above is a single
    internal node.)"""
    dummies = "".join(f"    {i + 5} Z{i}() 0\n" for i in range(15))
    net = (
        "begin parameters\n    1 kf 0.37\n    2 ks 1e-6\n    3 kz 1\nend parameters\n"
        "begin functions\n    1 ksf() ks\nend functions\n"
        "begin species\n    1 A() 1e6\n    2 B() 1e6\n    3 C() 0\n    4 D() 0\n"
        + dummies
        + "end species\n"
        # reaction order puts a zero-rate decay at leaf 0, A + B at leaf 1, and
        # the birth at leaf 5
        "begin reactions\n    1 5 0 kz\n    2 1,2 3 kf\n"
        + "".join(f"    {i + 3} {i + 6} 0 kz\n" for i in range(3))
        + "    6 0 4 ksf\n"
        + "".join(f"    {i + 7} {i + 9} 0 kz\n" for i in range(11))
        + "end reactions\nbegin groups\n    1 Dtot 4\nend groups\n"
    )
    m = _net(tmp_path, "deep.net", net)
    assert m.n_reactions == 17
    reps = 20
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 2e7), n_points=2, seed=9, squeeze=True
    )
    d = np.asarray(r.observables)[:, -1, 0]
    want = 1e-6 * 2e7
    assert abs(d.mean() - want) <= 4.5 * np.sqrt(want / reps), (d.mean(), want)


MM_NET = """begin parameters
    1 kcat 1.0
    2 Km 10.0
end parameters
begin species
    1 E() 10
    2 S() 100
    3 P() 0
end species
begin reactions
    1 1,2 1,3 MM kcat Km
end reactions
begin groups
    1 Ptot 3
end groups
"""


@pytest.mark.parametrize(
    ("method", "kw"),
    METHODS
    + [
        pytest.param("ode", {"codegen": False}, id="ode-interpreted"),
        pytest.param("ode", {"codegen": True}, id="ode-codegen"),
    ],
)
@pytest.mark.parametrize("param", ["Km", "kcat"])
def test_a_nan_michaelis_menten_parameter_is_refused(tmp_path, method, kw, param):
    """A NaN Km failed both of the tQSSA's guards and read as a rate of 0, so
    the reaction silently stopped (P stayed 0) under SSA, PSA and ODE."""
    m = _net(tmp_path, "mm.net", MM_NET)
    m.set_param(param, float("nan"))
    with pytest.raises(SimulationError):
        bngsim.Simulator(m, method=method, **kw).run((0.0, 1.0), 3, seed=1, timeout=30)


def test_a_finite_michaelis_menten_rate_is_unchanged(tmp_path):
    """The term that carries a NaN through adds exactly 0.0 to a finite rate."""
    m = _net(tmp_path, "mm.net", MM_NET)
    a = m.propensities(np.asarray([10.0, 100.0, 0.0]))[0]
    sfree = 0.5 * ((100 - 10 - 10) + np.sqrt((100 - 10 - 10) ** 2 + 4 * 10 * 100))
    assert a == pytest.approx(10 * sfree / (10 + sfree), rel=1e-15)


def test_an_overflow_mid_run_is_refused_in_the_fast_loop(tmp_path):
    """``A + A -> 3A`` at k = 1e300 grows until its propensity overflows. The
    compiled fast loop takes this model, and the overflow happens after
    t = 0, where every earlier check had already passed."""
    net = (
        "begin parameters\n    1 k 1e300\nend parameters\n"
        "begin species\n    1 A() 2\nend species\n"
        "begin reactions\n    1 1,1 1,1,1 k\nend reactions\n"
        "begin groups\n    1 Atot 1\nend groups\n"
    )
    m = _net(tmp_path, "grow.net", net)
    probe = bngsim.Model.from_net(str(tmp_path / "grow.net"))
    probe.set_param("k", 1e-3)
    r = bngsim.Simulator(probe, method="ssa").run((0.0, 1e-3), 2, seed=1)
    if r.ssa_diagnostics["propensity_backend"] != "cc":
        pytest.skip("no C compiler: the compiled fast loop is not in play")
    with pytest.raises(SimulationError, match=r"propensity .* is inf at t = (?!0 )"):
        bngsim.Simulator(m, method="ssa").run((0.0, 1.0), 3, seed=1, timeout=30)
