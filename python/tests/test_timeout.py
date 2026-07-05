"""Timeout / wall-clock cancellation tests (closes issue #9).

PyBNF's ``wall_time_sim`` setting needs an in-process way to cancel
long-running simulations across all backends. These tests verify the
``timeout`` kwarg on :meth:`bngsim.Simulator.run` (and :meth:`run_batch`)
raises :class:`bngsim.SimulationTimeout` and that the exception carries
the configured limit alongside the actual elapsed wall-clock time.
"""

from __future__ import annotations

import time

import bngsim
import numpy as np
import pytest

# ─── Helpers ──────────────────────────────────────────────────────────


def _reversible_ssa_sim(reversible_net) -> bngsim.Simulator:
    """Reversible A + B <-> C with high rates: SSA loop runs for many
    reactions per real second, making it easy to trip a sub-second budget.
    """
    m = bngsim.Model.from_net(str(reversible_net))
    # Bump rates so a0 ≈ kf*A*B + kr*C ≈ 5e4 — lots of reactions/second.
    m.set_param("kf", 10.0)
    m.set_param("kr", 10.0)
    return bngsim.Simulator(m, method="ssa")


# ─── Exception surface ────────────────────────────────────────────────


def test_simulation_timeout_in_exception_hierarchy() -> None:
    # PyBNF and other consumers should be able to catch wall-clock failures
    # via `except SimulationTimeout` (specific) or `except BngsimError`
    # (umbrella). Inheriting from RuntimeError preserves the looser
    # `except RuntimeError` path too.
    assert issubclass(bngsim.SimulationTimeout, bngsim.BngsimError)
    assert issubclass(bngsim.SimulationTimeout, RuntimeError)
    # SimulationTimeout is NOT a SimulationError — that distinction is the
    # whole point of the typed exception (issue #9).
    assert not issubclass(bngsim.SimulationTimeout, bngsim.SimulationError)


def test_simulation_timeout_exposed_in_public_api() -> None:
    # Public re-export so `bngsim.SimulationTimeout` is the canonical name.
    assert bngsim.SimulationTimeout is not None
    assert "SimulationTimeout" in bngsim.__all__


def test_simulation_timeout_attributes() -> None:
    # Carry both the configured limit and the actual elapsed time so the
    # caller can log / classify accurately.
    exc = bngsim.SimulationTimeout("msg", timeout=1.5, elapsed=1.7)
    assert exc.timeout == 1.5
    assert exc.elapsed == 1.7
    assert exc.partial_result is None
    assert "msg" in str(exc)


# ─── Argument validation ──────────────────────────────────────────────


def test_negative_timeout_rejected(simple_decay_net) -> None:
    m = bngsim.Model.from_net(simple_decay_net)
    sim = bngsim.Simulator(m, method="ode")
    with pytest.raises(ValueError, match="non-negative"):
        sim.run(t_span=(0, 10), n_points=11, timeout=-1.0)


def test_timeout_zero_treated_as_disabled(simple_decay_net) -> None:
    # 0 (or None) → no budget; the simulation runs to completion regardless.
    m = bngsim.Model.from_net(simple_decay_net)
    sim = bngsim.Simulator(m, method="ode")
    r = sim.run(t_span=(0, 50), n_points=11, timeout=0.0)
    assert len(r.time) == 11
    r = sim.run(t_span=(0, 50), n_points=11, timeout=None)
    assert len(r.time) == 11


# ─── Backend-by-backend timeout firing ────────────────────────────────


def test_ssa_timeout_raises_simulation_timeout(reversible_net) -> None:
    """SSA: with a tiny budget, a long simulation must raise the typed
    exception in roughly the configured time."""
    sim = _reversible_ssa_sim(reversible_net)
    budget = 0.3
    t0 = time.perf_counter()
    with pytest.raises(bngsim.SimulationTimeout) as info:
        sim.run(t_span=(0, 1e6), n_points=1001, seed=1, timeout=budget)
    wall = time.perf_counter() - t0

    # Wall-clock attribute must reflect actual elapsed at the time the
    # check tripped, not the configured limit.
    assert info.value.timeout == pytest.approx(budget)
    assert info.value.elapsed >= budget
    # Overall fire-and-translate latency should be within a generous bound
    # (heavy CI hosts can stall briefly between the check and the throw).
    assert wall < budget + 5.0, (
        f"Timeout took {wall:.3f}s wall to surface, budget was {budget:.3f}s"
    )


def test_ssa_short_run_unaffected_by_generous_timeout(simple_decay_net) -> None:
    """A generous budget on a fast simulation completes normally."""
    m = bngsim.Model.from_net(simple_decay_net)
    sim = bngsim.Simulator(m, method="ssa")
    r = sim.run(t_span=(0, 50), n_points=11, seed=1, timeout=10.0)
    assert len(r.time) == 11
    # Result is a full Result, not a partial.
    assert r.observables.shape[0] == 11


def test_ode_short_run_unaffected_by_generous_timeout(simple_decay_net) -> None:
    m = bngsim.Model.from_net(simple_decay_net)
    sim = bngsim.Simulator(m, method="ode")
    r = sim.run(t_span=(0, 100), n_points=101, timeout=10.0)
    assert len(r.time) == 101


def test_psa_run_passes_timeout_kwarg(simple_decay_net) -> None:
    """The PSA dispatch path threads timeout through to run_psa()."""
    m = bngsim.Model.from_net(simple_decay_net)
    sim = bngsim.Simulator(m, method="psa", poplevel=100.0)
    # Tight budget on a moderately long PSA run: either the run finishes
    # quickly (no failure) or it times out — both are valid outcomes for
    # this smoke test. What we're asserting is that the kwarg is accepted
    # AND any timeout that fires is the typed exception.
    try:
        sim.run(t_span=(0, 1000), n_points=11, seed=1, timeout=5.0)
    except bngsim.SimulationTimeout as e:
        assert e.timeout == pytest.approx(5.0)


# ─── RuleMonkey timeout (closes followup to issue #32) ────────────────


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_stateless_timeout_raises(simple_decay_net, nfsim_xml) -> None:
    """Simulator.run(method='nf_exact', timeout=...) honors the budget via
    the upstream RuleMonkey cancellation hook (richardposner/RuleMonkey#3)."""
    dummy_model = bngsim.Model.from_net(str(simple_decay_net))
    sim = bngsim.Simulator(dummy_model, method="nf_exact", xml_path=str(nfsim_xml))

    budget = 0.3
    t0 = time.perf_counter()
    with pytest.raises(bngsim.SimulationTimeout) as info:
        sim.run(t_span=(0, 1e6), n_points=1001, seed=1, timeout=budget)
    wall = time.perf_counter() - t0

    assert info.value.timeout == pytest.approx(budget)
    assert info.value.elapsed >= budget
    # Upstream polls every ~1024 SSA events; allow generous slack for the
    # in-flight stride that finishes after the budget trips.
    assert wall < budget + 30.0, (
        f"RuleMonkey timeout took {wall:.3f}s to surface, budget was {budget:.3f}s"
    )


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_sample_times_timeout_raises_and_recovers(simple_decay_net, nfsim_xml) -> None:
    """The explicit sample_times path (GH #169) honors the budget and stays usable.

    The sample_times branch drives the upstream session API per segment, so a
    mid-advance cancellation must (a) surface as SimulationTimeout and (b) tear
    the live session down — otherwise the next run() would see a stale session.
    """
    dummy_model = bngsim.Model.from_net(str(simple_decay_net))
    sim = bngsim.Simulator(dummy_model, method="nf_exact", xml_path=str(nfsim_xml))

    budget = 0.3
    with pytest.raises(bngsim.SimulationTimeout) as info:
        # A long horizon split into several explicit instants forces the SSA to
        # run long enough for the wall-clock budget to trip between segments.
        sim.run(sample_times=[0, 2.5e5, 5e5, 7.5e5, 1e6], seed=1, timeout=budget)
    assert info.value.timeout == pytest.approx(budget)
    assert info.value.elapsed >= budget

    # The session was cleaned up on the timeout, so a fresh quick run succeeds.
    recovered = sim.run(sample_times=[0, 0.05, 0.1], seed=1, timeout=30.0)
    assert recovered.n_times == 3
    np.testing.assert_allclose(recovered.time, [0, 0.05, 0.1], atol=1e-12)


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_stateless_generous_timeout_completes(simple_decay_net, nfsim_xml) -> None:
    """A generous budget on a quick segment completes with a full Result."""
    dummy_model = bngsim.Model.from_net(str(simple_decay_net))
    sim = bngsim.Simulator(dummy_model, method="nf_exact", xml_path=str(nfsim_xml))
    r = sim.run(t_span=(0, 0.1), n_points=11, seed=1, timeout=30.0)
    assert r.n_times == 11
    # timeout=None still works (no rejection now that RM supports it).
    r2 = sim.run(t_span=(0, 0.1), n_points=11, seed=1, timeout=None)
    assert r2.n_times == 11


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_session_simulate_timeout_raises(nfsim_xml) -> None:
    """RuleMonkeySession.simulate must honor a wall-clock budget."""
    budget = 0.3
    t0 = time.perf_counter()
    with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
        rm.initialize(seed=1)
        with pytest.raises(bngsim.SimulationTimeout) as info:
            rm.simulate(0.0, 1e6, n_points=1001, timeout=budget)
        wall = time.perf_counter() - t0

        assert info.value.timeout == pytest.approx(budget)
        assert info.value.elapsed >= budget
        assert wall < budget + 30.0, (
            f"Session timeout took {wall:.3f}s to surface, budget was {budget:.3f}s"
        )
        assert rm.initialized
        assert not rm.destroyed


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_session_simulate_destroys_cleanly_after_timeout(nfsim_xml) -> None:
    """After a timed-out simulate(), explicit destroy() must succeed."""
    with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
        rm.initialize(seed=1)
        with pytest.raises(bngsim.SimulationTimeout):
            rm.simulate(0.0, 1e6, n_points=1001, timeout=0.2)
        rm.destroy()
        assert rm.destroyed


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_session_simulate_generous_timeout_completes(nfsim_xml) -> None:
    """A generous budget on a quick segment finishes normally."""
    with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
        rm.initialize(seed=1)
        r = rm.simulate(0.0, 0.1, n_points=11, timeout=30.0)
        assert r.n_times == 11


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_session_simulate_negative_timeout_rejected(nfsim_xml) -> None:
    with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
        rm.initialize(seed=1)
        with pytest.raises(ValueError, match="non-negative"):
            rm.simulate(0.0, 1.0, n_points=11, timeout=-1.0)


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_session_simulate_zero_and_none_disable_budget(nfsim_xml) -> None:
    """timeout=0 and timeout=None both disable the wall-clock budget."""
    with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
        rm.initialize(seed=1)
        r1 = rm.simulate(0.0, 0.1, n_points=11, timeout=0.0)
        assert r1.n_times == 11
        r2 = rm.simulate(0.1, 0.2, n_points=11, timeout=None)
        assert r2.n_times == 11


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_session_step_to_timeout_raises(nfsim_xml) -> None:
    """step_to() also honors the cooperative-cancellation hook."""
    with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
        rm.initialize(seed=1)
        with pytest.raises(bngsim.SimulationTimeout) as info:
            rm.step_to(1e6, timeout=0.3)
        assert info.value.timeout == pytest.approx(0.3)
        assert info.value.elapsed >= 0.3


@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not built")
def test_rulemonkey_session_step_to_negative_timeout_rejected(nfsim_xml) -> None:
    with bngsim.RuleMonkeySession(str(nfsim_xml)) as rm:
        rm.initialize(seed=1)
        with pytest.raises(ValueError, match="non-negative"):
            rm.step_to(1.0, timeout=-1.0)


# ─── SolverOptions surface (for advanced callers) ─────────────────────


def test_solver_options_timeout_seconds_field() -> None:
    """Advanced callers using the low-level C++ binding can set
    timeout_seconds directly on SolverOptions."""
    from bngsim._bngsim_core import SolverOptions

    opts = SolverOptions()
    assert opts.timeout_seconds == 0.0
    opts.timeout_seconds = 2.5
    assert opts.timeout_seconds == 2.5


# ─── NfsimSession.simulate timeout (closes issue #32) ─────────────────


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not built")
def test_nfsim_session_simulate_timeout_raises(nfsim_xml) -> None:
    """NfsimSession.simulate must honor a wall-clock budget by raising
    SimulationTimeout between stepTo() output points."""
    # NFsim's sampler is opaque inside one stepTo() call, so the budget
    # can only be checked between output points. Pick a dt small enough
    # that each hop is short (so the surfaced overshoot is bounded) but a
    # total horizon long enough that many hops occur and the budget trips
    # well before the nominal end time.
    budget = 0.3
    t0 = time.perf_counter()
    with bngsim.NfsimSession(str(nfsim_xml)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationTimeout) as info:
            nf.simulate(0.0, 1000.0, n_points=1001, timeout=budget)
        wall = time.perf_counter() - t0

        assert info.value.timeout == pytest.approx(budget)
        assert info.value.elapsed >= budget
        # NFsim granularity = one stepTo() hop, so allow a generous slack
        # for the trailing hop that finishes after the budget expires.
        assert wall < budget + 30.0, (
            f"Session timeout took {wall:.3f}s to surface, budget was {budget:.3f}s"
        )

        # Session must remain usable for teardown after the timeout — the
        # context manager's __exit__ should destroy() cleanly.
        assert nf.initialized
        assert not nf.destroyed


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not built")
def test_nfsim_session_simulate_destroys_cleanly_after_timeout(nfsim_xml) -> None:
    """After a timed-out simulate(), explicit destroy() must succeed and
    the session must report destroyed=True."""
    with bngsim.NfsimSession(str(nfsim_xml)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationTimeout):
            nf.simulate(0.0, 1000.0, n_points=1001, timeout=0.2)

        # Explicit destroy() (not just __exit__) — verifies the C++ session
        # object is in a state that accepts destroy_session() cleanly.
        nf.destroy()
        assert nf.destroyed


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not built")
def test_nfsim_session_simulate_generous_timeout_completes(nfsim_xml) -> None:
    """A generous budget on a quick segment finishes normally with a full Result."""
    with bngsim.NfsimSession(str(nfsim_xml)) as nf:
        nf.initialize(seed=1)
        r = nf.simulate(0.0, 0.1, n_points=11, timeout=30.0)
        assert r.n_times == 11
        # Result is a full Result, not a partial.
        assert r.observables.shape[0] == 11


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not built")
def test_nfsim_session_simulate_negative_timeout_rejected(nfsim_xml) -> None:
    """Mirrors Simulator.run's argument-validation contract."""
    with bngsim.NfsimSession(str(nfsim_xml)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(ValueError, match="non-negative"):
            nf.simulate(0.0, 1.0, n_points=11, timeout=-1.0)


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not built")
def test_nfsim_session_simulate_zero_and_none_disable_budget(nfsim_xml) -> None:
    """timeout=0 and timeout=None both disable the wall-clock budget."""
    with bngsim.NfsimSession(str(nfsim_xml)) as nf:
        nf.initialize(seed=1)
        r1 = nf.simulate(0.0, 0.1, n_points=11, timeout=0.0)
        assert r1.n_times == 11
        r2 = nf.simulate(0.1, 0.2, n_points=11, timeout=None)
        assert r2.n_times == 11
