"""Tests for NFsim tech debt elimination (Session 29).

Task 1: set_block_same_complex() removed — no pybind11 binding exists.
Task 2: exit(1) → throw std::runtime_error() — malformed XML raises Python exception.
Task 3: StreamSuppressor thread-safe — concurrent NFsim runs don't crash.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bngsim
import numpy as np
import pytest


def _has_nfsim() -> bool:
    """Check if NFsim support is compiled in."""
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return HAS_NFSIM
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_nfsim(),
    reason="bngsim compiled without NFsim support",
)


# ─── Task 1: set_block_same_complex removed ──────────────────────────────────


class TestBlockSameComplexRemoved:
    """Verify the misleading set_block_same_complex API is gone."""

    def test_no_set_block_same_complex_on_core(self, nfsim_xml):
        """NfsimSimulator C++ binding has no set_block_same_complex method."""
        from bngsim._bngsim_core import NfsimSimulator

        sim = NfsimSimulator(str(nfsim_xml))
        assert not hasattr(sim, "set_block_same_complex"), (
            "set_block_same_complex should have been removed"
        )

    def test_normal_run_still_works(self, dummy_model, nfsim_xml):
        """NFsim run works correctly without set_block_same_complex."""
        sim = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim.run(t_span=(0, 1), n_points=11, seed=42)
        assert result.n_times == 11
        assert result.n_observables == 6

        # Conservation still holds (complex bookkeeping is always on)
        xtotal = result.observables["Xtotal"]
        np.testing.assert_allclose(xtotal, 5000.0, atol=0.5)


# ─── Task 2: exit(1) → exceptions ────────────────────────────────────────────


class TestExitToException:
    """Verify that NFsim errors raise Python exceptions, not process death."""

    def test_malformed_xml_raises_runtime_error(self, nfsim_malformed_xml):
        """Malformed XML triggers RuntimeError, not process exit."""
        from bngsim._bngsim_core import NfsimSimulator, TimeSpec

        sim = NfsimSimulator(str(nfsim_malformed_xml))
        times = TimeSpec()
        times.t_start = 0.0
        times.t_end = 1.0
        times.n_points = 2

        # NFsim should raise an exception, NOT kill the process
        with pytest.raises(RuntimeError):
            sim.run(times, 42)

    def test_malformed_xml_via_simulator(self, dummy_model, nfsim_malformed_xml):
        """High-level Simulator wraps the error as SimulationError."""
        sim = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_malformed_xml),
        )
        with pytest.raises(bngsim.SimulationError):
            sim.run(t_span=(0, 1), n_points=2, seed=42)

    def test_process_survives_after_error(self, dummy_model, nfsim_xml, nfsim_malformed_xml):
        """Process survives a malformed XML error and can run again."""
        # First: trigger error
        sim_bad = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_malformed_xml),
        )
        with pytest.raises(bngsim.SimulationError):
            sim_bad.run(t_span=(0, 1), n_points=2, seed=42)

        # Second: a valid simulation still works
        sim_good = bngsim.Simulator(dummy_model, method="nfsim", xml_path=str(nfsim_xml))
        result = sim_good.run(t_span=(0, 1), n_points=11, seed=42)
        assert result.n_times == 11

    def test_composite_with_local_deps_does_not_kill_process(self, nfsim_composite_local_deps_xml):
        """Initialize succeeds on a model whose rate law is a CompositeFunction
        with local-function deps; the composite is skipped from
        expression_names.

        Before Session 29's exit(1)->throw conversion (compositeFunction.cpp
        evaluateOn), the bngsim probe in resolve_output_functions hit a hard
        exit(1) for every composite with n_lfs>0, killing the Python process
        on essentially every BNGL model with a non-trivial rate law. The
        try/catch around the probe was structurally unable to catch
        exit(1); only the post-Session-29 std::runtime_error makes the probe
        actually skip-and-continue.
        """
        with bngsim.NfsimSession(str(nfsim_composite_local_deps_xml)) as s:
            s.initialize(2024)
            result = s.simulate(0, 1, 11)
        # The composite is skipped from expression_names because it has
        # local deps and cannot be evaluated scope-free. The simulation
        # itself still ran — i.e. the throw was caught by the host, not
        # propagated as a process kill or an init failure.
        assert result.n_times == 11
        assert "_rateLaw1" not in list(result.expression_names)

    def test_composite_with_reactant_count_dep_does_not_segfault(
        self, nfsim_composite_reactant_count_dep_xml
    ):
        """Initialize succeeds on a model whose rate law is a CompositeFunction
        with a reactant-count dependency (`reactant_1()*k`); the composite is
        skipped from expression_names.

        GH #116: a composite that references `reactant_N()` has n_lfs==0 (no
        local-function deps) but n_reactantCounts>0. The bngsim probe in
        resolve_output_functions evaluates every composite scope-free
        (evaluateOn(nullptr,nullptr,nullptr,0)) to find scalar-valued output
        functions. That call passed the local-function guard (n_lfs==0) and
        then dereferenced the NULL reactant-count array in
        CompositeFunction::evaluateOn — a native SIGSEGV that killed the Python
        process during initialize(), uncatchable from Python. evaluateOn now
        throws a catchable std::runtime_error when a reactant-count context is
        missing, so the probe skips the composite and the simulation runs.
        """
        with bngsim.NfsimSession(str(nfsim_composite_reactant_count_dep_xml)) as s:
            s.initialize(2024)
            result = s.simulate(0, 1, 11)
        # The composite rate law is skipped from expression_names — it has no
        # scope-free scalar value — but the simulation itself ran to completion.
        assert result.n_times == 11
        assert "_rateLaw1" not in list(result.expression_names)
        assert "reactant_1" not in list(result.expression_names)

    def test_unknown_molecule_type_raises_runtime_error(self, nfsim_xml):
        """Asking NFsim for an unknown molecule-type name raises RuntimeError,
        not exit(1).

        Exercises system.cpp::getMoleculeTypeByName, one of the 14 system.cpp
        sites Session 29 converted from exit(1) -> throw std::runtime_error.
        Before the conversion, a typo in add_molecules / get_molecule_count
        would kill the embedding Python process.
        """
        with bngsim.NfsimSession(str(nfsim_xml)) as s:
            s.initialize(42)
            with pytest.raises(RuntimeError):
                s.add_molecules("DoesNotExist", 1)


# ─── Task 3: Thread-safe StreamSuppressor ─────────────────────────────────────


class TestConcurrentNfsim:
    """Verify concurrent NFsim instances don't crash or deadlock."""

    def test_concurrent_runs_no_crash(self, dummy_model, nfsim_xml):
        """Two NFsim instances via ThreadPoolExecutor don't crash."""

        def _run_nfsim(seed: int) -> int:
            """Run a short NFsim simulation, return n_times."""
            sim = bngsim.Simulator(
                dummy_model,
                method="nfsim",
                xml_path=str(nfsim_xml),
            )
            result = sim.run(t_span=(0, 0.5), n_points=6, seed=seed)
            return result.n_times

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_run_nfsim, seed) for seed in [100, 200]]
            results = [f.result(timeout=30) for f in futures]

        # Both should complete successfully
        assert results == [6, 6]

    def test_concurrent_deterministic(self, dummy_model, nfsim_xml):
        """Serial and concurrent NFsim give same results for same seed."""

        def _run(seed: int) -> np.ndarray:
            sim = bngsim.Simulator(
                dummy_model,
                method="nfsim",
                xml_path=str(nfsim_xml),
            )
            result = sim.run(t_span=(0, 1), n_points=11, seed=seed)
            return np.asarray(result.observables)

        # Serial
        serial_1 = _run(42)
        serial_2 = _run(42)
        np.testing.assert_array_equal(serial_1, serial_2)

        # Concurrent (same seeds — mutex serializes, so results
        # should match serial)
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(_run, 42)
            f2 = executor.submit(_run, 42)
            concurrent_1 = f1.result(timeout=30)
            concurrent_2 = f2.result(timeout=30)

        np.testing.assert_array_equal(serial_1, concurrent_1)
        np.testing.assert_array_equal(serial_1, concurrent_2)


@pytest.fixture
def dummy_model(simple_decay_net: Path) -> bngsim.Model:
    """A dummy Model (NFsim ignores it; only xml_path matters)."""
    return bngsim.Model.from_net(str(simple_decay_net))
