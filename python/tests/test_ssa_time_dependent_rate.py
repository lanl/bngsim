"""SSA with a time-dependent assignment-rule rate (RC#2 regression).

The direct method assumes propensities are constant between fires, and the
dependency graph only refreshes a reaction's propensity when one of its
*species* populations changes. A rate law that reads a purely time-dependent
assignment rule therefore has no trigger to refresh it — so the SSA used to
freeze the rate at its t=0 value. When that t=0 value is zero (the synthesis
flux below is ``5*time``, i.e. 0 at t=0), the total propensity is 0 at the
start and the loop wedged in its "stuck" fast-forward, flat-lining the
trajectory at the initial state forever.

This mirrors the BioModels #30 cross-engine finding (BIOMD0000001040 /
BIOMD0000001026: ``Mrbc`` frozen at 0 under SSA while the ODE — and RR's ODE —
grow it). It is also the mirror image of roadrunner#1317 (RR's gillespie has
the same lag, undershooting the same trajectory by one output interval).

Oracle: for an immigration–death process with time-dependent immigration rate
λ(t) and per-capita death μ, E[N(t)] satisfies dN/dt = λ(t) − μ·N — exactly
the deterministic ODE. So the SSA mean over replicates must track the ODE
trajectory within sampling error. We use the *same-engine* ODE as the oracle,
which sidesteps the RR-side bug entirely.

See ``dev/notes/SBML_VS_ROADRUNNER.md`` (RC#2) and
``src/ssa_simulator.cpp`` (time-inhomogeneous sub-stepping).
"""

import bngsim
import numpy as np
import pytest

# Both models here are built via Model.from_antimony_string, which needs the
# optional `antimony` dependency (GH #153). It is excluded from the
# cibuildwheel test env, so skip the whole module when it is unavailable.
pytest.importorskip("antimony")

# Synthesis flux ``synth := 5*time`` is 0 at t=0 and rises linearly — exactly
# the RC#2 shape (zero initial propensity, time-dependent lift-off). Degradation
# ``kdeg*S`` makes it a proper immigration–death process with an ODE mean.
_ANTIMONY = """
model time_dep_birth_death
  compartment C = 1;
  species S in C = 0;
  kdeg = 0.3;
  synth := 5*time;
  J1: => S; synth;
  J2: S => ; kdeg*S;
end
"""

T_END = 10.0
N_POINTS = 11
N_REPS = 500
SEED = 20260524


def _ode_trajectory():
    model = bngsim.Model.from_antimony_string(_ANTIMONY)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0.0, T_END), n_points=N_POINTS)
    return np.asarray(result.species)[:, 0]


def _ssa_mean_sd():
    model = bngsim.Model.from_antimony_string(_ANTIMONY)
    sim = bngsim.Simulator(model, method="ssa")
    results = sim.run_batch(
        t_span=(0.0, T_END), n_points=N_POINTS, params=[{} for _ in range(N_REPS)], seed=SEED
    )
    arr = np.array([np.asarray(r.species)[:, 0] for r in results])
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)


def test_ssa_tracks_ode_with_time_dependent_rate():
    """SSA mean over replicates tracks the ODE at every t>0 within 5·SE."""
    ode = _ode_trajectory()
    mean, sd = _ssa_mean_sd()

    se = sd / np.sqrt(N_REPS)
    # Skip t=0 (deterministic 0) and avoid div-by-zero where sd==0.
    for i in range(1, N_POINTS):
        z = abs(mean[i] - ode[i]) / (se[i] + 1e-12)
        assert z < 5.0, (
            f"SSA mean at sample {i} = {mean[i]:.3f} vs ODE {ode[i]:.3f} "
            f"(z = {z:.2f}, 5·SE = {5 * se[i]:.3f}); time-dependent rate not "
            f"tracked."
        )


def test_ssa_does_not_freeze_at_initial_zero_propensity():
    """Pre-fix guard: with synth(0)=0 the old loop froze S at 0 forever.

    Assert the trajectory is meaningfully non-zero and monotone-ish growing,
    so a regression that reinstates the freeze fails loudly rather than
    sliding under the 5·SE band at small times.
    """
    ode = _ode_trajectory()
    mean, _ = _ssa_mean_sd()

    # The ODE reaches ~114 by t=10; a frozen SSA would report 0 everywhere.
    assert mean[-1] > 0.5 * ode[-1], (
        f"SSA mean at t={T_END} = {mean[-1]:.3f} but ODE = {ode[-1]:.3f}; "
        f"the time-dependent synthesis rate appears frozen."
    )
    # Growth must actually happen across the run, not just at the end.
    assert mean[N_POINTS // 2] > 1.0
