"""GH #176: ``jacobian="auto"`` falls back to the finite-difference Jacobian when
the analytical Jacobian de-stabilizes CVODE on a rate law that is discontinuous
in a state variable.

l-type-calcium-channel-dynamics has ``v_rec = if((-70+V)<-20, 0.5, 0.05)`` — a
genuine value jump in the state ``V``, which asymptotically approaches the
threshold 50 at t≈25. The exact analytical Jacobian's derivative of a step is 0,
so it cannot warn CVODE's implicit corrector about the impending jump: the BDF
predictor overshoots, the corrector meets an unanticipated jump, the local error
test fails repeatedly and the step collapses to hmin (flag=-3). The
finite-difference Jacobian straddles the step and supplies a regularizing slope,
so it — and legacy run_network, which is always FD — integrate the model cleanly.

The fix is at the Simulator: ``jacobian="auto"`` (the default) is a bet, so on a
solver failure it transparently retries once with the FD Jacobian. An explicit
``jacobian="analytical"`` is *not* second-guessed.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import SimulationError

FIXTURE = "ltype_calcium_discontinuous_jacobian.net"
T_SPAN = (0.0, 150.0)
N_POINTS = 301
TOL = 1e-8


def _net(data_dir: Path) -> str:
    return str(data_dir / FIXTURE)


def test_auto_falls_back_to_fd_and_integrates(data_dir: Path) -> None:
    """The default config integrates the full horizon (the analytical attempt
    fails internally and is retried with FD)."""
    m = bngsim.Model.from_net(_net(data_dir))
    sim = bngsim.Simulator(m, method="ode")
    result = sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL)
    assert result.n_times == N_POINTS
    # The fallback is observable post-hoc.
    assert sim.jacobian_strategy == "fd"


def test_auto_fallback_matches_explicit_fd(data_dir: Path) -> None:
    """The auto (fallen-back) trajectory is identical to the explicit-FD one —
    the retry simply selects the FD Jacobian, which is deterministic."""
    m_auto = bngsim.Model.from_net(_net(data_dir))
    r_auto = bngsim.Simulator(m_auto, method="ode").run(
        t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL
    )
    m_fd = bngsim.Model.from_net(_net(data_dir))
    r_fd = bngsim.Simulator(m_fd, method="ode", jacobian="fd").run(
        t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL
    )
    assert np.array_equal(np.asarray(r_auto.observables), np.asarray(r_fd.observables))


def test_explicit_analytical_is_not_second_guessed(data_dir: Path) -> None:
    """An explicit ``jacobian="analytical"`` surfaces the failure rather than
    silently falling back — the user asked for analytical."""
    m = bngsim.Model.from_net(_net(data_dir))
    sim = bngsim.Simulator(m, method="ode", jacobian="analytical")
    with pytest.raises(SimulationError):
        sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL)


def test_repeated_runs_skip_the_doomed_attempt(
    data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Once the analytical attempt has failed on a Simulator, subsequent runs go
    straight to FD (memoized): only the first run pays for (and logs) a failed
    analytical attempt."""
    m = bngsim.Model.from_net(_net(data_dir))
    sim = bngsim.Simulator(m, method="ode")
    with caplog.at_level("WARNING", logger="bngsim"):
        first = sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL)
        assert sim._ode_jacobian_fell_back is True
        second = sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL)
    assert second.n_times == first.n_times == N_POINTS
    # The fallback warning is logged exactly once — the second run did not retry.
    fallbacks = [r for r in caplog.records if "GH#176 analytical Jacobian" in r.getMessage()]
    assert len(fallbacks) == 1
