"""GH #176: ``jacobian="auto"`` falls back to the finite-difference Jacobian when
the analytical Jacobian de-stabilizes CVODE on a rate law that is discontinuous
in a state variable.

l-type-calcium-channel-dynamics has ``v_rec = if((-70+V)<-20, 0.5, 0.05)`` — a
genuine value jump in the state ``V``, which relaxes towards the threshold 50 and
parks just below it. The exact analytical Jacobian's derivative of a step is 0,
so it cannot warn CVODE's implicit corrector about the jump the predictor keeps
landing on: the corrector meets an unanticipated jump, the local error test fails
repeatedly and the step collapses to hmin (flag=-3). The finite-difference
Jacobian perturbs ``V`` by ``srur*|V|`` = 7.4e-7, straddles the step and supplies
a regularizing slope, so it — and legacy run_network, which is always FD —
integrate this fixture.

**"Integrate this fixture", not "integrate such models."** The effect needs the
trajectory to PARK on the step: give the same model a threshold it crosses
transversally and the analytical Jacobian integrates it fine, because CVODE steps
over a lone value jump whatever the Jacobian says. And in the parked regime the
outcome is a rounding outcome — see lanl/bngsim#176, where the FD rescue held on
one platform and died at t≈34.6 on another with the same source. The fixture now
parks a defined 1e-11 below the step (its header derives the two margins) so that
particular coin flip is gone, but the regime is chaotic in general: sweeping the
parking gap over 23 values finds isolated gaps — 2e-12, 9e-11, 1e-10 — where the
steady-state march below stops converging while both neighbours converge in ~600
steps.

So the run-half tests assert the *policy*, against an explicit-FD reference taken
on the same host, rather than a hard-coded "this model integrates". The policy is
what bngsim controls; whether FD carries a given model to 150 s is the model's
business and the host's.

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

# This module, for the two tests that monkeypatch its own helper. pytest loads
# the file by path, so it is not importable under its bare filename.
_THIS = sys.modules[__name__]

FIXTURE = "ltype_calcium_discontinuous_jacobian.net"
T_SPAN = (0.0, 150.0)
N_POINTS = 301
TOL = 1e-8


def _net(data_dir: Path) -> str:
    return str(data_dir / FIXTURE)


def _run(net: str, jacobian: str | None = None) -> tuple[str, np.ndarray | None]:
    """Run ``net`` to the full horizon and report the OUTCOME rather than assume
    one: ``("ok", observables)`` or ``("raised", None)``.

    lanl/bngsim#176 is the standing reason this is a pair and not a bare call. In
    the parked-on-the-step regime the module docstring describes, whether FD
    carries a model to 150 s is settled by rounding, so a test that hard-codes
    "it integrates" is asserting something neither bngsim nor the model decides.
    Comparing two outcomes taken on the same host asserts only the part bngsim
    does decide.
    """
    kwargs = {} if jacobian is None else {"jacobian": jacobian}
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode", **kwargs)
    try:
        result = sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL)
    except SimulationError:
        return "raised", None
    return "ok", np.asarray(result.observables)


def _require_fd_reference(net: str) -> np.ndarray:
    """The explicit-FD trajectory on THIS host, or skip.

    The tests below that assert a *successful* rescue need FD to actually carry
    the fixture here. It does wherever the fixture's two margins hold, which is
    the point of parking it 1e-11 below the step — but the skip is keyed on the
    observable itself rather than on a guessed platform axis, which is the
    mistake lanl/bngsim#176 recorded (it read as Linux-vs-macOS; #228 then showed
    the causal axis was neither the OS nor the BLAS).
    """
    outcome, observables = _run(net, "fd")
    if outcome != "ok":
        pytest.skip(
            "the finite-difference Jacobian does not carry this fixture on this "
            "build, so there is no rescue to assert — see the module docstring"
        )
    assert observables is not None
    return observables


def test_auto_is_exactly_the_fd_run(data_dir: Path) -> None:
    """The contract, and the one assertion that holds on every host: ``auto`` is
    "try analytical, then FD", so whatever explicit FD does here ``auto`` does —
    bit for bit when FD integrates, and the same failure when it does not."""
    net = _net(data_dir)
    auto_outcome, auto_obs = _run(net)
    fd_outcome, fd_obs = _run(net, "fd")
    assert auto_outcome == fd_outcome
    if fd_outcome == "ok":
        assert auto_obs is not None and fd_obs is not None
        assert np.array_equal(auto_obs, fd_obs)


def test_auto_falls_back_to_fd_and_integrates(data_dir: Path) -> None:
    """The default config integrates the full horizon (the analytical attempt
    fails internally and is retried with FD)."""
    net = _net(data_dir)
    _require_fd_reference(net)
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    result = sim.run(t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL)
    assert result.n_times == N_POINTS
    # The fallback is observable post-hoc.
    assert sim.jacobian_strategy == "fd"


def test_auto_fallback_matches_explicit_fd(data_dir: Path) -> None:
    """The auto (fallen-back) trajectory is identical to the explicit-FD one —
    the retry simply selects the FD Jacobian, which is deterministic."""
    net = _net(data_dir)
    fd_obs = _require_fd_reference(net)
    r_auto = bngsim.Simulator(bngsim.Model.from_net(net), method="ode").run(
        t_span=T_SPAN, n_points=N_POINTS, rtol=TOL, atol=TOL
    )
    assert np.array_equal(np.asarray(r_auto.observables), fd_obs)


def test_fd_reference_skips_rather_than_fails_when_fd_cannot_carry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard's skip branch only ever fires on a host where FD gives up, i.e.
    never on the host that ordinarily runs these tests — the same one-platform
    blind spot lanl/bngsim#176 itself came from. So exercise it directly rather
    than let a typo in it sleep until some future Linux run trips over it."""
    monkeypatch.setattr(_THIS, "_run", lambda net, jacobian=None: ("raised", None))
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _require_fd_reference("unused.net")
    assert "does not carry this fixture" in str(excinfo.value)


def test_fd_reference_returns_the_trajectory_when_fd_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the other branch hands back the reference itself, so the callers that
    compare against it are comparing against FD and not against ``None``."""
    reference = np.arange(6.0).reshape(3, 2)
    monkeypatch.setattr(_THIS, "_run", lambda net, jacobian=None: ("ok", reference))
    assert _require_fd_reference("unused.net") is reference


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
    _require_fd_reference(_net(data_dir))
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
# model's discontinuity exactly as run() does — CVODE gives up and the march
# returns unconverged. The retry is decided in the solver rather than in Python,
# because a failed march is a flag, not an exception.
#
# These four are NOT written against an explicit-FD reference the way the run
# half is, because on this fixture the march converges in ~600 steps and has no
# platform history: they passed on the Linux leg that reported lanl/bngsim#176
# even while the run half failed there. That is not a guarantee — the sweep in
# the module docstring found parking gaps where the march instead burns >100k
# steps and reports unconverged — but a march that chatters at a state
# discontinuity is its own defect with its own blast radius, so it is tracked in
# issue #235 rather than papered over here.


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
