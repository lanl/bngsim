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

Issue #127 gave the steady-state solver the same bet — its march installs the
closed-form Jacobian too — and therefore needed the same way out. That half is
decided in C++ (a failed march is a flag on the result, not an exception) and is
covered by the second group of tests below, on this same fixture: it is the model
the failure is about, whichever entry point meets it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import SimulationError

FIXTURE = "ltype_calcium_discontinuous_jacobian.net"
T_SPAN = (0.0, 150.0)
N_POINTS = 301
TOL = 1e-8

# Quarantine for lanl/bngsim#176 — NOT the "GH #176" this file's header is about.
# The digits collide and the trackers do not: "GH #176" is the upstream issue that
# ADDED the finite-difference retry, lanl/bngsim#176 is the report that the retry
# does not save this fixture on Linux.
#
# What the first whole-suite Linux run (#169) showed: the retry machinery works.
# The analytical attempt dies at the t≈25 crossing the header documents, the
# warning fires, FD engages — and FD then dies at t≈34.6, a *second* crossing the
# header does not mention. So the header's premise ("the finite-difference
# Jacobian straddles the step ... so it integrates the model cleanly") holds under
# Accelerate and not under Linux's reference LAPACK.
#
# The four steady-state tests below are deliberately NOT marked: they pass on
# Linux, because #127's march never reaches t≈34.6. That contrast is the sharpest
# evidence in the report, so keep the marker per-test rather than module-wide.
#
# strict=True so this retires itself — the day FD carries the full 150 s horizon
# on Linux, these xpass and the run goes red until the marker is deleted.
fd_fallback_dies_on_linux = pytest.mark.xfail(
    sys.platform.startswith("linux"),
    reason="lanl/bngsim#176: the FD fallback dies at a second crossing (t≈34.6) on Linux",
    strict=True,
    raises=SimulationError,
)


def _net(data_dir: Path) -> str:
    return str(data_dir / FIXTURE)


@fd_fallback_dies_on_linux
def test_auto_falls_back_to_fd_and_integrates(data_dir: Path) -> None:
    """The default config integrates the full horizon (the analytical attempt
    fails internally and is retried with FD)."""
    m = bngsim.Model.from_net(_net(data_dir))
    sim = bngsim.Simulator(m, method="ode")
    result = sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL)
    assert result.n_times == N_POINTS
    # The fallback is observable post-hoc.
    assert sim.jacobian_strategy == "fd"


@fd_fallback_dies_on_linux
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


@fd_fallback_dies_on_linux
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


# ── The steady-state half of the same policy (issue #127) ────────────────────
#
# Since #127 the march installs the closed-form Jacobian, so it meets this
# model's discontinuity exactly as run() does — CVODE gives up at t≈24 and the
# march returns unconverged. The retry is decided in the solver rather than in
# Python, because a failed march is a flag, not an exception.


def test_steady_state_auto_falls_back_and_converges(data_dir: Path) -> None:
    """The default config converges, on the difference quotient, and says so."""
    m = bngsim.Model.from_net(_net(data_dir))
    sim = bngsim.Simulator(m, method="ode")
    ss = sim.steady_state(method="integration", rtol=TOL, atol=TOL)
    assert ss.converged
    assert ss.solver_jacobian_retried
    assert ss.solver_jacobian_source == "finite-difference"


def test_steady_state_fallback_matches_explicit_fd(data_dir: Path) -> None:
    """The retry selects the FD Jacobian and nothing else, so the retried solve
    is the explicit-FD solve — bit for bit, and to the same step count."""
    m_auto = bngsim.Model.from_net(_net(data_dir))
    auto = bngsim.Simulator(m_auto, method="ode").steady_state(
        method="integration", rtol=TOL, atol=TOL
    )
    m_fd = bngsim.Model.from_net(_net(data_dir))
    fd = bngsim.Simulator(m_fd, method="ode", jacobian="fd").steady_state(
        method="integration", rtol=TOL, atol=TOL
    )
    assert not fd.solver_jacobian_retried, "explicit fd never installed one to call off"
    assert auto.n_steps == fd.n_steps
    assert np.array_equal(np.asarray(auto.concentrations), np.asarray(fd.concentrations))


def test_steady_state_explicit_analytical_is_not_second_guessed(data_dir: Path) -> None:
    """``jacobian="analytical"`` surfaces the failure: the march does not retry,
    and reports the unconverged answer the closed form gave it."""
    m = bngsim.Model.from_net(_net(data_dir))
    sim = bngsim.Simulator(m, method="ode", jacobian="analytical")
    ss = sim.steady_state(method="integration", rtol=TOL, atol=TOL)
    assert not ss.converged
    assert not ss.solver_jacobian_retried
    assert ss.solver_jacobian_source == "analytical"


def test_steady_state_repeated_solves_skip_the_doomed_attempt(
    data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The memo, which is what keeps a dose-response scan from re-paying the
    failed march at every point: only the first solve retries, and only the
    first one warns."""
    m = bngsim.Model.from_net(_net(data_dir))
    sim = bngsim.Simulator(m, method="ode")
    with caplog.at_level("WARNING", logger="bngsim"):
        first = sim.steady_state(method="integration", rtol=TOL, atol=TOL)
        assert sim._ss_jacobian_fell_back is True
        second = sim.steady_state(method="integration", rtol=TOL, atol=TOL)
    assert first.converged and second.converged
    assert first.solver_jacobian_retried
    assert not second.solver_jacobian_retried, "the second solve went straight to fd"
    assert np.array_equal(np.asarray(first.concentrations), np.asarray(second.concentrations))
    fallbacks = [r for r in caplog.records if "GH#176 analytical Jacobian" in r.getMessage()]
    assert len(fallbacks) == 1
