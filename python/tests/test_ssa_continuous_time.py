"""SSA/PSA with rates that move between firings: exact in time (issues #719,
#751, #753).

A rate that reads time, or a rate-rule target, used to be held constant over
sub-steps of (t_end − t_start)/1000, so the answer depended on the horizon: a
synthesis at 5·(1 + sin t) sampled once per 5 time units over (0, 5000)
reported E[N(995)] = 2707 against an exact 5002. Rate-rule targets were
integrated by forward Euler on that grid (x' = −30x over (0, 100) went to
1e33), and a model with rate rules and no firing reaction advanced them only
at output times.

The loop now integrates each dynamic propensity over adaptive panels and fires
at the root of the integrated hazard (the time-change theorem), with rate
rules on an L-stable Rosenbrock method. Oracles are closed forms: Poisson
means of integrated rates, and first moments of linear systems, which equal
the ODE.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")


def _ant(text):
    return bngsim.Model.from_antimony_string(text)


def _col(r, name):
    return np.asarray(r.species)[..., list(r.species_names).index(name)]


def _within(got, want, reps, z=4.5):
    """Poisson (or sum-of-Poisson) mean: SE √(want/reps)."""
    se = np.sqrt(max(want, 1e-12) / reps)
    assert abs(got - want) <= z * se, (got, want, se)


SINE = "species N = 0; J: => N; 5*(1 + sin(time));"


def _sine_mean(t):
    return 5.0 * (t + 1.0 - np.cos(t))


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_periodic_rate_over_a_long_horizon(method):
    """E[N(t)] = 5·(t + 1 − cos t). The old sub-step, 5 time units at
    T = 5000, sampled the sine once per cycle."""
    kw = {"poplevel": 100} if method == "psa" else {}
    reps = 40
    r = bngsim.Simulator(_ant(SINE), method=method, **kw).run_replicates(
        reps, t_span=(0, 5000), n_points=5001, seed=3, squeeze=True
    )
    n = _col(r, "N")
    for t in (50.0, 995.0, 4321.0):
        _within(n[:, int(t)].mean(), _sine_mean(t), reps)


def test_the_answer_does_not_depend_on_the_horizon():
    """The same seed, two horizons: the counts at t = 50 agree."""
    out = []
    for t_end in (100, 1000):
        r = bngsim.Simulator(_ant(SINE), method="ssa").run_replicates(
            50, t_span=(0, t_end), n_points=t_end + 1, seed=11, squeeze=True
        )
        out.append(_col(r, "N")[:, 50])
    _within(out[0].mean(), _sine_mean(50.0), 50)
    _within(out[1].mean(), _sine_mean(50.0), 50)


def test_a_step_in_the_rate():
    """``piecewise(10, t > 5, 0)``: E[N(20)] = 150."""
    reps = 200
    r = bngsim.Simulator(_ant("species N = 0; J: => N; piecewise(10, time > 5, 0);"), method="ssa")
    r = r.run_replicates(reps, t_span=(0, 20), n_points=3, seed=9, squeeze=True)
    _within(_col(r, "N")[:, -1].mean(), 150.0, reps)


NET_PULSE = """begin parameters
    1 k 1000
end parameters
begin functions
    1 pulse() if(time()>5 && time()<5.001, k, 0)
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 pulse
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.mark.parametrize(("method", "kw"), [("ssa", {}), ("psa", {"poplevel": 100})])
def test_a_pulse_narrower_than_a_step_is_not_stepped_over(tmp_path, method, kw):
    """1000 for a millisecond at t = 5 in a run to 100: E[A] = 1. The loop's
    step does not see a pulse that falls between its nodes; the pulse's edges
    are breakpoints, and main reported A = 0 in every replicate."""
    p = tmp_path / "pulse.net"
    p.write_text(NET_PULSE)
    reps = 400
    r = bngsim.Simulator(bngsim.Model.from_net(str(p)), method=method, **kw).run_replicates(
        reps, t_span=(0, 100), n_points=2, seed=3, squeeze=True
    )
    _within(np.asarray(r.observables)[:, -1, 0].mean(), 1.0, reps)


def test_a_rate_rule_drives_a_propensity():
    """X' = 1 from 0 and ``0 -> S`` at X: E[S(4)] = ∫₀⁴ t dt = 8."""
    reps = 400
    m = _ant("species S = 0; X = 0; X' = 1; J: => S; X;")
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 4), n_points=5, seed=5, squeeze=True
    )
    _within(_col(r, "S")[:, -1].mean(), 8.0, reps)
    np.testing.assert_allclose(_col(r, "X")[0], [0, 1, 2, 3, 4], rtol=1e-6, atol=1e-9)


def test_a_rate_rule_with_nothing_firing_is_integrated():
    """#751: x' = −x with no reaction that can fire. x(t) = e^−t."""
    m = _ant("species z = 0; x = 1; x' = -x; E: at (time >= 8): z = 1;")
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 10), n_points=11, seed=1)
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "x"), np.exp(-t), rtol=2e-3)
    assert _col(r, "z")[-1] == 1.0


@pytest.mark.parametrize("t_end", [100, 1000])
def test_a_stiff_rate_rule(t_end):
    """#753: x' = −30x beside a slow decay. Forward Euler on a step of
    t_end/1000 diverged (|x| to 1e33 at t_end = 100); x stays at 0."""
    m = _ant("species A = 10; J: A => ; 0.1*A; x = 1; x' = -30*x;")
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, t_end), n_points=11, seed=1)
    assert np.max(np.abs(_col(r, "x")[1:])) < 1e-8


def test_an_event_on_a_rate_rule_target_fires_at_the_crossing():
    """y' = 1, ``at (y > 0.5): b = time``: b = 0.5."""
    m = _ant(
        "y = 0; y' = 1; species b = 0; species a = 0; E2: at (y > 0.5): b = time; J: => a; 0.1;"
    )
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 3), n_points=4, seed=1)
    assert _col(r, "b")[-1] == pytest.approx(0.5, abs=1e-9)


# ── linear systems: the mean is the ODE ──────────────────────────────────────


def _mean_vs_ode(text, name, t_end, reps, method="ssa", **kw):
    m = _ant(text)
    ode = bngsim.Simulator(_ant(text), method="ode").run(t_span=(0, t_end), n_points=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaRoundingWarning)
        r = bngsim.Simulator(m, method=method, **kw).run_replicates(
            reps, t_span=(0, t_end), n_points=5, seed=17, squeeze=True
        )
    got = _col(r, name)
    want = _col(ode, name)
    for i in range(1, 5):
        se = got[:, i].std(ddof=1) / np.sqrt(reps)
        assert abs(got[:, i].mean() - want[i]) <= 4.5 * se + 1e-9, (
            i,
            got[:, i].mean(),
            want[i],
            se,
        )


def test_forcing_beside_a_fast_decay():
    """``0 -> M`` at 50·(1 + sin t) and ``M -> 0`` at 0.5·M. The forcing reads
    only time, so a decay firing keeps the loop's panel; the mean is the
    ODE's."""
    _mean_vs_ode("species M = 0; J1: => M; 50*(1 + sin(time)); J2: M => ; 0.5*M;", "M", 20.0, 200)


def test_a_time_dependent_rate_that_reads_the_state():
    """``M -> 2M`` at c(t)·M: each firing moves what the rate reads, so the
    panel is rebuilt. Linear, so the mean is the ODE's."""
    _mean_vs_ode(
        "species M = 20; J1: M => 2 M; 0.1*(1 + sin(time))*M; J2: M => ; 0.15*M;", "M", 10.0, 300
    )


def test_psa_with_a_dynamic_reaction():
    _mean_vs_ode(
        "species M = 0; J1: => M; 500*(1 + sin(time)); J2: M => ; 0.5*M;",
        "M",
        10.0,
        100,
        method="psa",
        poplevel=50,
    )


def test_the_integrated_propensity_of_a_dynamic_reaction():
    """Reaction stats (#616) bank ∫a dt for a dynamic reaction from the loop's
    quadrature: 5·(t + 1 − cos t) at every row, whatever fired."""
    m = _ant("species N = 0; species M = 0; J1: => N; 5*(1 + sin(time)); J2: => M; 3;")
    sim = bngsim.Simulator(m, method="ssa", reaction_stats=True)
    r = sim.run(t_span=(0, 30), n_points=7, seed=2)
    integ = np.asarray(r.reaction_propensity_integrals)
    t = np.asarray(r.time)
    np.testing.assert_allclose(integ[:, 0], _sine_mean(t) - _sine_mean(0.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(integ[:, 1], 3.0 * t, rtol=1e-12)
