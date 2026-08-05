"""Issue #177 — a sensitivity error floor below the arithmetic's own noise.

**The bug is open.** This module pins the mechanism with a closed-form model, so
that whoever fixes it has a failing test to turn green and a statement of what
must NOT change while doing it.

``setup_forward_sensitivities`` hands CVODES a per-(state × column) absolute
tolerance ``atolS[iS][i] = atol·scale[i]/pbar[iS]``, where ``scale[i]`` is a
characteristic size of state *i* (GH #214). For column ``iS`` the variational
equation is ``ṡ = J·s + ∂f/∂p``, so row *i*'s value is assembled by summing over
row *i*, and its roundoff is not zero but ``ε·(the terms summed)``. A row whose
own ``|s_i|`` has decayed to ~0 contributes nothing to ``rtol·|s_i|``, leaving
``atolS_i`` as the only thing holding the error weight finite — and set below
that roundoff, the test can never pass at any step size. CVODES shrinks ``h``
chasing accuracy the arithmetic does not have.

The state solve has no equivalent: a species sitting at zero has a zero RHS, not
a difference of huge fluxes, and a species at 1e18 is covered many times over by
``rtol·|x|``. ``test_scalar_solve_is_fast_and_exact`` pins that asymmetry, which
is what makes this a sensitivity-only defect.

**What the fix may not do.** The sensitivities that *are* resolvable are correct
today — ``test_sensitivities_are_accurate_where_resolvable`` holds on the
unfixed engine — so the 183,219 steps are pure waste, not the price of the
answer. Any fix has to keep that column right while removing the steps.

``sens_scale_cancellation.net`` is the mechanism in two reactions and one
parameter; see its header. It has a closed form, so this module needs no FD
oracle:

    X(t)     = (c1/c2)·S·(1 − e^{−p·c2·t})
    dX/dp(t) = c1·S·t·e^{−p·c2·t}
"""

from __future__ import annotations

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# Must match sens_scale_cancellation.net.
S0, C1, C2, P0 = 1e18, 2.0, 1.0, 1.0

# The horizon matters. dX/dp peaks at t = 1/(p·c2) and decays as t·e^{−p·c2·t},
# so the unreachable floor only starts costing steps once the sensitivity has
# fallen below the roundoff of its own source term — around t ≈ 30 here. Over
# [0, 10] the step count is unremarkable. The early samples are for the accuracy
# check, which needs the decade where dX/dp is still above the noise.
SAMPLE_TIMES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0]
T_END = SAMPLE_TIMES[-1]

# Measured on the unfixed engine: 183,219 steps for this call, against 210 for
# the same model's scalar solve, and rising with the horizon because what
# collapses is the step size, not the interval count. A fix should land near the
# scalar cost; this budget sits far from both.
STEP_BUDGET = 5_000


def _net(name: str) -> str:
    p = DATA_DIR / name
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _exact_sens(t: np.ndarray) -> np.ndarray:
    return C1 * S0 * t * np.exp(-P0 * C2 * t)


def _exact_state(t: np.ndarray) -> np.ndarray:
    return (C1 / C2) * S0 * (1.0 - np.exp(-P0 * C2 * t))


def _run(sensitivity: bool):
    kwargs = {"sensitivity_params": ["p"]} if sensitivity else {}
    sim = bngsim.Simulator(
        bngsim.Model.load(_net("sens_scale_cancellation.net")), method="ode", **kwargs
    )
    return sim.run(
        t_span=(0.0, T_END),
        n_points=len(SAMPLE_TIMES),
        sample_times=SAMPLE_TIMES,
        timeout=300.0,
    )


@pytest.fixture(scope="module")
def sens_run():
    return _run(sensitivity=True)


@pytest.fixture(scope="module")
def scalar_run():
    return _run(sensitivity=False)


@pytest.mark.xfail(
    strict=True,
    reason="issue #177: the sensitivity error floor is below the roundoff of "
    "ṡ = J·s + ∂f/∂p, so CVODES micro-steps. Remove this marker with the fix.",
)
def test_sensitivity_solve_does_not_micro_step(sens_run):
    """The reported symptom, in miniature.

    On ``Smith_BMCSystBiol2013`` the same mechanism turns a 0.013 s scalar run
    into one that does not finish in 300 s with 16 sensitivity columns. Here the
    model is small enough that the run still returns — it just spends ~900x the
    steps of its own scalar solve to do it.
    """
    n_steps = sens_run.solver_stats["n_steps"]
    assert n_steps < STEP_BUDGET, (
        f"coupled state+sensitivity solve took {n_steps} steps over [0, {T_END}]; "
        "the sensitivity absolute floor is below the arithmetic's own noise and "
        "is driving h down (issue #177)"
    )


def test_scalar_solve_is_fast_and_exact(scalar_run):
    """The asymmetry that makes this a sensitivity-only defect.

    Same model, same state, same horizon: the state solve neither micro-steps
    nor loses accuracy, because a species at 2e18 is covered many times over by
    ``rtol·|x|`` and a species at zero has a zero RHS rather than a difference of
    huge fluxes.
    """
    n_steps = scalar_run.solver_stats["n_steps"]
    assert n_steps < 2_000, f"scalar solve took {n_steps} steps"
    t = np.asarray(scalar_run.time)
    x = np.asarray(scalar_run.species)[:, 1]
    exact = _exact_state(t)
    rel = np.abs(x - exact) / np.abs(exact).max()
    assert rel.max() < 1e-6, f"state trajectory off the closed form by {rel.max():.2e}"


def test_sensitivities_are_accurate_where_resolvable(sens_run):
    """The steps are waste, not the price of the answer.

    Over the decade where ``dX/dp`` is above 1e-8 of its own peak the column
    already matches the closed form to ~5e-6 — so a fix that removes the steps
    must leave this untouched, and one that "fixes" the run by giving up
    accuracy here has not fixed it.
    """
    t = np.asarray(sens_run.time)
    s = np.asarray(sens_run.sensitivities)[:, 1, 0]
    exact = _exact_sens(t)
    live = np.abs(exact) > 1e-8 * np.abs(exact).max()
    assert live.sum() >= 3
    rel = np.abs(s[live] - exact[live]) / np.abs(exact[live])
    assert rel.max() < 1e-4, f"analytic dX/dp off the closed form by {rel.max():.2e}"


def test_state_is_unaffected_by_requesting_sensitivities(sens_run, scalar_run):
    """Whatever the fix does to the sensitivity tolerances, x(t) must not move.

    Pinned here rather than assumed: both static floors that were tried for #177
    touched only ``atolS``, and a later fix that reaches the state tolerances
    would change every trajectory in the corpus.
    """
    xs = np.asarray(sens_run.species)[:, 1]
    xr = np.asarray(scalar_run.species)[:, 1]
    scale = max(np.abs(xr).max(), 1e-300)
    assert np.abs(xs - xr).max() / scale < 1e-6
