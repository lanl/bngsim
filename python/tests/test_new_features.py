"""bngsim comprehensive tests for new Phase B / B.5 features.

Tests cover:
- num_processors argument for run_batch (ThreadPoolExecutor)
- Squeeze semantics (Result.squeeze, squeeze=True in run_batch)
- HDF5 save/load (Result.save / Result.load)
- Stop conditions (expression strings + Python callables)
- Logging integration (configure_logging)
- Interactive simulation (run_until, intervene, snapshot, restore)
- Expression/function output in Result
- Simulator properties and configuration
- ObservableAccessor edge cases
- Regression tests for additional model files
- Result construction from raw arrays
- File export on loaded (non-core) results
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import Model, Result, Simulator
from bngsim._exceptions import (
    BngsimError,
    ModelError,
    SimulationError,
    StopConditionMet,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Batch with num_processors (ThreadPoolExecutor)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumProcessors:
    """Test run_batch with num_processors for parallel execution."""

    def test_batch_parallel_ode(self, simple_decay_net: Path):
        """Parallel ODE batch produces same results as sequential."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        params = [{"k1": k} for k in [0.05, 0.1, 0.5, 1.0]]

        seq = sim.run_batch(t_span=(0, 10), n_points=11, params=params)
        par = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=params,
            num_processors=2,
        )

        assert len(seq) == len(par) == 4
        for s, p in zip(seq, par, strict=False):
            np.testing.assert_allclose(s.species, p.species, atol=1e-10)

    def test_batch_parallel_ssa(self, simple_decay_net: Path):
        """Parallel SSA batch with same base seed."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")

        params = [{"k1": 0.1}, {"k1": 0.5}]

        seq = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=params,
            seed=100,
        )
        par = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=params,
            seed=100,
            num_processors=2,
        )

        assert len(seq) == len(par) == 2
        for s, p in zip(seq, par, strict=False):
            np.testing.assert_array_equal(s.species, p.species)

    def test_batch_parallel_single_processor(self, simple_decay_net: Path):
        """num_processors=1 falls back to sequential."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}],
            num_processors=1,
        )
        assert len(results) == 1
        assert results[0].n_times == 11

    def test_batch_parallel_many(self, simple_decay_net: Path):
        """Parallel batch with more sims than processors."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        params = [{"k1": 0.1 * (i + 1)} for i in range(8)]
        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=params,
            num_processors=3,
        )
        assert len(results) == 8
        for r in results:
            assert r.n_times == 11


# ═══════════════════════════════════════════════════════════════════════════════
# Squeeze semantics
# ═══════════════════════════════════════════════════════════════════════════════


class TestSqueezeSemantics:
    """Test squeeze=True in run_batch and Result.squeeze."""

    def test_squeeze_3d_species(self, simple_decay_net: Path):
        """squeeze=True produces 3D species array."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        batch = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}, {"k1": 0.5}, {"k1": 1.0}],
            squeeze=True,
        )

        assert isinstance(batch, Result)
        assert batch.species.ndim == 3
        assert batch.species.shape == (3, 11, 2)

    def test_squeeze_3d_observables(self, simple_decay_net: Path):
        """squeeze=True produces 3D observables array."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        batch = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}, {"k1": 0.5}],
            squeeze=True,
        )

        obs_arr = np.asarray(batch.observables)
        assert obs_arr.ndim == 3
        assert obs_arr.shape == (2, 11, 2)

    def test_squeeze_time_shared(self, simple_decay_net: Path):
        """Squeezed result has 1D time (shared across sims)."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        batch = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}, {"k1": 0.5}],
            squeeze=True,
        )
        assert batch.time.shape == (11,)

    def test_squeeze_single_returns_original(self, simple_decay_net: Path):
        """squeeze with single result returns the result itself."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        results = [sim.run(t_span=(0, 10), n_points=11)]
        squeezed = Result.squeeze(results)
        assert squeezed is results[0]
        assert squeezed.species.ndim == 2

    def test_squeeze_empty_raises(self):
        """squeeze with empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Result.squeeze([])

    def test_squeeze_solver_stats_aggregated(self, simple_decay_net: Path):
        """Squeezed result aggregates solver stats."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}, {"k1": 0.5}],
        )

        total_steps = sum(r.solver_stats["n_steps"] for r in results)
        squeezed = Result.squeeze(results)
        assert squeezed.solver_stats["n_steps"] == total_steps
        assert squeezed.solver_stats["linear_solver"] == results[0].solver_stats["linear_solver"]

        alternate_solver = 1 if results[0].solver_stats["linear_solver"] != 1 else 2
        results[1].solver_stats["linear_solver"] = alternate_solver
        mixed = Result.squeeze(results)
        assert mixed.solver_stats["linear_solver"] == -1

    def test_squeeze_names_preserved(self, simple_decay_net: Path):
        """Squeezed result preserves species/observable names."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        batch = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}, {"k1": 0.5}],
            squeeze=True,
        )
        assert batch.species_names == ["A()", "B()"]
        assert batch.observable_names == ["A_tot", "B_tot"]


# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 save / load
# ═══════════════════════════════════════════════════════════════════════════════


class TestHdf5SaveLoad:
    """Test Result.save() and Result.load() for HDF5 serialization."""

    @pytest.fixture(autouse=True)
    def _check_h5py(self):
        pytest.importorskip("h5py")

    def test_round_trip(self, simple_decay_net: Path, tmp_path: Path):
        """Save and load preserves all data."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        path = tmp_path / "test.h5"
        result.save(path)
        assert path.exists()

        loaded = Result.load(path)
        np.testing.assert_allclose(loaded.time, result.time, atol=1e-12)
        np.testing.assert_allclose(loaded.species, result.species, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(loaded.observables),
            np.asarray(result.observables),
            atol=1e-12,
        )

    def test_round_trip_names(self, simple_decay_net: Path, tmp_path: Path):
        """Save/load preserves species and observable names."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        path = tmp_path / "names.h5"
        result.save(path)
        loaded = Result.load(path)

        assert loaded.species_names == result.species_names
        assert loaded.observable_names == result.observable_names

    def test_round_trip_solver_stats(self, simple_decay_net: Path, tmp_path: Path):
        """Save/load preserves solver stats."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        path = tmp_path / "stats.h5"
        result.save(path)
        loaded = Result.load(path)

        assert loaded.solver_stats["n_steps"] == (result.solver_stats["n_steps"])
        assert loaded.solver_stats["n_rhs_evals"] == (result.solver_stats["n_rhs_evals"])

    def test_round_trip_custom_attrs(self, simple_decay_net: Path, tmp_path: Path):
        """Save/load preserves custom_attrs (string + numeric)."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        result.custom_attrs["experiment"] = "test_hdf5"
        result.custom_attrs["run_id"] = 42

        path = tmp_path / "attrs.h5"
        result.save(path)
        loaded = Result.load(path)

        assert loaded.custom_attrs["experiment"] == "test_hdf5"
        assert loaded.custom_attrs["run_id"] == 42

    def test_load_nonexistent_raises(self, tmp_path: Path):
        """Load from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Result.load(tmp_path / "nonexistent.h5")

    def test_round_trip_dimensions(self, simple_decay_net: Path, tmp_path: Path):
        """Loaded result has correct dimensions."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        path = tmp_path / "dims.h5"
        result.save(path)
        loaded = Result.load(path)

        assert loaded.n_times == 11
        assert loaded.n_species == 2
        assert loaded.n_observables == 2

    def test_loaded_result_repr(self, simple_decay_net: Path, tmp_path: Path):
        """Loaded result has correct repr."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        path = tmp_path / "repr.h5"
        result.save(path)
        loaded = Result.load(path)

        assert "n_times=11" in repr(loaded)
        assert "n_species=2" in repr(loaded)


# ═══════════════════════════════════════════════════════════════════════════════
# Stop conditions
# ═══════════════════════════════════════════════════════════════════════════════


class TestStopConditions:
    """Test stop conditions (expression strings + callables)."""

    def test_callable_stop_triggers(self, simple_decay_net: Path):
        """Callable stop condition triggers on decayed species."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.add_stop_condition(
            lambda r: r.species[-1, 0] < 10.0,
            label="A_below_10",
        )

        with pytest.raises(StopConditionMet) as exc_info:
            sim.run(t_span=(0, 100), n_points=101)

        assert exc_info.value.result is not None
        assert exc_info.value.condition == "A_below_10"

    def test_expression_stop_triggers(self, simple_decay_net: Path):
        """Expression string stop condition triggers."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.add_stop_condition("A_tot < 50", label="half_decay")

        with pytest.raises(StopConditionMet) as exc_info:
            sim.run(t_span=(0, 100), n_points=101)

        result = exc_info.value.result
        assert result is not None
        # Result should be truncated
        assert result.n_times < 101

    def test_stop_condition_partial_result(self, simple_decay_net: Path):
        """Partial result from stop condition is valid."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.add_stop_condition(
            lambda r: r.species[-1, 0] < 50.0,
            label="half",
        )

        with pytest.raises(StopConditionMet) as exc_info:
            sim.run(t_span=(0, 100), n_points=101)

        result = exc_info.value.result
        assert result.time.shape[0] == result.species.shape[0]
        assert result.species.shape[1] == 2

    def test_no_stop_no_exception(self, simple_decay_net: Path):
        """If condition doesn't trigger, no exception."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.add_stop_condition(
            lambda r: r.species[-1, 0] < -1.0,
            label="never",
        )

        # Should complete normally
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.n_times == 11

    def test_clear_stop_conditions(self, simple_decay_net: Path):
        """clear_stop_conditions removes all conditions."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.add_stop_condition(lambda r: True, label="always")
        sim.clear_stop_conditions()

        # Should complete normally since conditions are cleared
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.n_times == 11

    def test_stop_condition_bad_type_raises(self, simple_decay_net: Path):
        """Non-string, non-callable raises TypeError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        with pytest.raises(TypeError, match="callable"):
            sim.add_stop_condition(42)

    def test_stop_condition_inherits_bngsim_error(self):
        """StopConditionMet is a BngsimError."""
        assert issubclass(StopConditionMet, BngsimError)

    def test_multiple_stop_conditions(self, simple_decay_net: Path):
        """First triggered condition wins."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.add_stop_condition("A_tot < 80", label="early")
        sim.add_stop_condition("A_tot < 20", label="late")

        with pytest.raises(StopConditionMet) as exc_info:
            sim.run(t_span=(0, 100), n_points=101)

        # The earlier condition should trigger first
        assert exc_info.value.condition == "early"


# ═══════════════════════════════════════════════════════════════════════════════
# Logging integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogging:
    """Test configure_logging and log output."""

    def test_configure_logging_returns_logger(self):
        """configure_logging returns a Logger."""
        log = bngsim.configure_logging(logging.DEBUG)
        assert isinstance(log, logging.Logger)
        assert log.name == "bngsim"
        # Clean up
        log.handlers.clear()

    def test_configure_logging_level(self):
        """Logger has the requested level."""
        log = bngsim.configure_logging(logging.WARNING)
        assert log.level == logging.WARNING
        log.handlers.clear()

    def test_configure_logging_custom_handler(self):
        """Custom handler is attached."""
        handler = logging.StreamHandler()
        log = bngsim.configure_logging(logging.INFO, handler=handler)
        assert handler in log.handlers
        log.handlers.clear()

    def test_sim_emits_log(self, simple_decay_net: Path, caplog):
        """Simulation emits log messages when logging enabled."""
        log = bngsim.configure_logging(logging.DEBUG)
        try:
            with caplog.at_level(logging.DEBUG, logger="bngsim"):
                model = Model.from_net(simple_decay_net)
                sim = Simulator(model, method="ode")
                sim.run(t_span=(0, 10), n_points=11)

            # Should have at least one log record from bngsim
            bngsim_records = [r for r in caplog.records if r.name == "bngsim"]
            assert len(bngsim_records) > 0
        finally:
            log.handlers.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive simulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestInteractiveSimulation:
    """Test run_until, intervene, snapshot, restore."""

    def test_run_until_basic(self, simple_decay_net: Path):
        """run_until advances simulation and updates current_time."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        assert sim.current_time == 0.0

        result = sim.run_until(t=10.0)
        assert sim.current_time == 10.0
        assert result.time[-1] == pytest.approx(10.0)
        assert result.n_times >= 2

    def test_run_until_continues(self, simple_decay_net: Path):
        """run_until can be called multiple times to continue."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.run_until(t=10.0)
        assert sim.current_time == 10.0

        r2 = sim.run_until(t=20.0)
        assert sim.current_time == 20.0
        assert r2.time[0] == pytest.approx(10.0)
        assert r2.time[-1] == pytest.approx(20.0)

    def test_run_until_past_raises(self, simple_decay_net: Path):
        """run_until to a past time raises ValueError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.run_until(t=10.0)
        with pytest.raises(ValueError, match="current time"):
            sim.run_until(t=5.0)

    def test_run_until_with_n_points(self, simple_decay_net: Path):
        """run_until with explicit n_points."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        result = sim.run_until(t=10.0, n_points=21)
        assert result.n_times == 21

    def test_intervene_changes_dynamics(self, simple_decay_net: Path):
        """intervene changes parameters mid-simulation."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        # Run with k1=0.1 until t=10
        sim.run_until(t=10.0)

        # Stop the reaction (k1=0)
        sim.intervene({"k1": 0.0})

        # Continue: A should stop decaying
        r2 = sim.run_until(t=20.0, n_points=11)
        A_at_10 = r2.species[0, 0]
        A_at_20 = r2.species[-1, 0]

        # With k1=0, A should be nearly constant
        assert A_at_20 == pytest.approx(A_at_10, abs=0.1)

    def test_snapshot_and_restore(self, simple_decay_net: Path):
        """snapshot captures state, restore reverts to it."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.run_until(t=10.0)
        snap = sim.snapshot()
        A_at_10 = model.get_concentration("A()")

        # Continue forward
        sim.run_until(t=50.0)
        assert sim.current_time == 50.0
        A_at_50 = model.get_concentration("A()")
        assert A_at_50 < A_at_10  # more decay

        # Restore to t=10
        sim.restore(snap)
        assert sim.current_time == 10.0
        A_restored = model.get_concentration("A()")
        assert A_restored == pytest.approx(A_at_10, abs=0.1)

    def test_restore_from_stack(self, simple_decay_net: Path):
        """restore() without arg uses internal stack."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.run_until(t=10.0)
        sim.snapshot()  # pushed to internal stack

        sim.run_until(t=50.0)
        sim.restore()  # pops from stack

        assert sim.current_time == 10.0

    def test_restore_empty_stack_raises(self, simple_decay_net: Path):
        """restore() with empty stack raises SimulationError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        with pytest.raises(SimulationError, match="snapshot"):
            sim.restore()

    def test_snapshot_preserves_params(self, simple_decay_net: Path):
        """Snapshot captures parameter values too."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.run_until(t=5.0)
        snap = sim.snapshot()

        sim.intervene({"k1": 99.0})
        assert model.get_param("k1") == pytest.approx(99.0)

        sim.restore(snap)
        assert model.get_param("k1") == pytest.approx(0.1)

    def test_interactive_ssa(self, simple_decay_net: Path):
        """Interactive simulation works with SSA."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")

        r1 = sim.run_until(t=10.0, seed=42)
        assert sim.current_time == 10.0
        assert r1.n_times >= 2

        sim.run_until(t=20.0, seed=42)
        assert sim.current_time == 20.0


# ═══════════════════════════════════════════════════════════════════════════════
# Expression / function output
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpressionOutput:
    """Test Result.expressions accessor for function output."""

    def test_rate_law_functions_filtered_from_expressions(self, time_func_net: Path):
        """Auto-generated _rateLawN functions are excluded from expression
        columns (BNG2.pl .gdat/.scan parity), but remain on the raw_* accessors.

        The model's only function is the synthetic `_rateLaw1() = time()+1`,
        which BNG2.pl never writes to .gdat. bngsim filters it identically.
        """
        model = Model.from_net(time_func_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        # _rateLaw1 is dropped from the BNG2.pl-parity expression columns.
        assert result.n_expressions == 0
        assert result.expression_names == []

        # …but the internal value stays available for debugging via raw_*.
        core = result._core
        assert list(core.raw_expression_names) == ["_rateLaw1"]
        assert core.raw_n_expressions == 1
        raw = np.asarray(core.raw_expression_data)
        assert raw.shape == (11, 1)
        # _rateLaw1() = time()+1, so the last point (t=10) is 11.
        assert raw[-1, 0] == pytest.approx(11.0)

    def test_expressions_empty_on_simple(self, simple_decay_net: Path):
        """Model without functions has empty expressions."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        assert result.n_expressions == 0
        assert result.expression_names == []

    def test_expressions_named_access(self, tfun_time_indexed_net: Path):
        """Named access to a user-named function column.

        ``expression_names`` are bare in-memory column keys (no "()" suffix);
        the BNG2.pl "()" header convention is applied only when serializing
        to .gdat/.scan, so named access uses the bare name.
        """
        model = Model.from_net(tfun_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 7), n_points=11)

        assert result.expression_names == ["cumNcases"]
        col = result.expressions["cumNcases"]
        assert col.shape == (11,)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator properties and configuration
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimulatorProperties:
    """Test Simulator properties and solver configuration."""

    def test_method_property(self, simple_decay_net: Path):
        """method property returns correct string."""
        model = Model.from_net(simple_decay_net)
        ode_sim = Simulator(model, method="ode")
        ssa_sim = Simulator(model, method="ssa")

        assert ode_sim.method == "ode"
        assert ssa_sim.method == "ssa"

    def test_model_property(self, simple_decay_net: Path):
        """model property returns the model."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        assert sim.model is model

    def test_current_time_property(self, simple_decay_net: Path):
        """current_time starts at 0."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        assert sim.current_time == 0.0

    def test_repr(self, simple_decay_net: Path):
        """repr includes method and model."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        r = repr(sim)
        assert "ode" in r
        assert "Model" in r

    def test_set_tolerances(self, simple_decay_net: Path):
        """set_tolerances updates internal state."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.set_tolerances(rtol=1e-12, atol=1e-14)
        # Run should work with tighter tolerances
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.n_times == 11

    def test_set_max_steps(self, simple_decay_net: Path):
        """set_max_steps updates internal state."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        sim.set_max_steps(50000)
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.n_times == 11

    def test_set_tolerances_ssa_noop(self, simple_decay_net: Path):
        """set_tolerances on SSA simulator is a no-op."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")

        # Should not raise
        sim.set_tolerances(rtol=1e-6, atol=1e-6)
        result = sim.run(t_span=(0, 10), n_points=11, seed=42)
        assert result.n_times == 11


# ═══════════════════════════════════════════════════════════════════════════════
# ObservableAccessor edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestObservableAccessor:
    """Test _ObservableAccessor dict-like and array-like access."""

    def test_int_indexing(self, simple_decay_net: Path):
        """Integer indexing returns a single time-point row."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        row = result.observables[0]
        assert row.shape == (2,)
        assert row[0] == pytest.approx(100.0, abs=1e-10)

    def test_slice_indexing(self, simple_decay_net: Path):
        """Slice indexing returns a sub-array."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        sliced = result.observables[:3]
        assert sliced.shape == (3, 2)

    def test_len(self, simple_decay_net: Path):
        """len(observables) returns number of time points."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        assert len(result.observables) == 11

    def test_ndim(self, simple_decay_net: Path):
        """ndim returns 2 for standard results."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        assert result.observables.ndim == 2

    def test_repr(self, simple_decay_net: Path):
        """repr includes shape and names."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        r = repr(result.observables)
        assert "ObservableAccessor" in r
        assert "shape" in r

    def test_array_copy_true(self, simple_decay_net: Path):
        """__array__(copy=True) returns a copy."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        arr = np.array(result.observables, copy=True)
        assert arr.shape == (11, 2)

    def test_array_default(self, simple_decay_net: Path):
        """np.asarray uses default copy semantics."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        arr = np.asarray(result.observables)
        assert arr.shape == (11, 2)
        assert arr.dtype == np.float64

    def test_negative_indexing(self, simple_decay_net: Path):
        """Negative index works for last row."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        last = result.observables[-1]
        assert last.shape == (2,)


# ═══════════════════════════════════════════════════════════════════════════════
# Regression: ConstantExpression set_param
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstExprSetParam:
    """Regression: set_param on ConstantExpression detaches expression."""

    def test_set_param_const_expr(self, const_expr_net: Path):
        """set_param('d', 27) makes d=27, not re-evaluated from d__FREE."""
        model = Model.from_net(const_expr_net)

        # d is a ConstantExpression: d = d__FREE = 70
        assert model.get_param("d") == pytest.approx(70.0)

        # Set d directly
        model.set_param("d", 27.0)
        assert model.get_param("d") == pytest.approx(27.0)

        # d__FREE unchanged
        assert model.get_param("d__FREE") == pytest.approx(70.0)

    def test_set_param_const_expr_survives_sim(self, const_expr_net: Path):
        """After set_param + simulate, the overridden value persists."""
        model = Model.from_net(const_expr_net)
        model.set_param("d", 27.0)

        sim = Simulator(model, method="ode")
        sim.run(t_span=(0, 10), n_points=11)

        # d should still be 27 after simulation
        assert model.get_param("d") == pytest.approx(27.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Regression: Expression-valued parameter species init
# ═══════════════════════════════════════════════════════════════════════════════


class TestExprParamSpecies:
    """Regression: species init from expression-valued parameter."""

    def test_species_init_from_expr(self, expr_param_net: Path):
        """A(0) = p = 2+3 = 5, not 2."""
        model = Model.from_net(expr_param_net)

        # A should start at p = 2+3 = 5
        assert model.get_concentration("A()") == pytest.approx(5.0)

    def test_species_init_analytical(self, expr_param_net: Path):
        """Simulate from expression-valued init."""
        model = Model.from_net(expr_param_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        # A(0) = 5, k1 = 0.1
        A_0 = result.species[0, 0]
        assert pytest.approx(5.0, abs=0.01) == A_0

        # A(10) = 5 * exp(-0.1 * 10) = 5 * exp(-1)
        A_10 = result.species[-1, 0]
        expected = 5.0 * math.exp(-1.0)
        assert pytest.approx(expected, abs=0.01) == A_10


# ═══════════════════════════════════════════════════════════════════════════════
# Regression: SSA homodimer propensity
# ═══════════════════════════════════════════════════════════════════════════════


class TestHomodimerSSA:
    """Regression: SSA propensity for repeated reactants (2A->C)."""

    def test_homodimer_single_molecule_no_reaction(self, homodimer_ssa_net: Path):
        """2A->C with A=1: propensity=0, no reaction occurs."""
        model = Model.from_net(homodimer_ssa_net)
        sim = Simulator(model, method="ssa")
        result = sim.run(t_span=(0, 100), n_points=101, seed=42)

        # A should stay at 1 (can't pick 2 from 1)
        np.testing.assert_allclose(result.species[:, 0], 1.0)
        # C should stay at 0
        np.testing.assert_allclose(result.species[:, 1], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Saturation-like functional rate law (formerly Sat, now Functional)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaturationRateLaw:
    """Test model with saturation-like functional rate law.

    The saturation.net test model was rewritten from 'Sat k3 K4' to a
    Functional rate law '_rateLaw1() = k3/(K4+G)'. This gives identical
    dynamics but is compatible with NFsim (which rejects Sat).
    """

    def test_saturation_model_loads(self, saturation_net: Path):
        """Saturation-like functional model loads without error."""
        model = Model.from_net(saturation_net)
        assert model.n_species == 3
        assert model.n_reactions == 2

    def test_saturation_runs_ode(self, saturation_net: Path):
        """ODE simulation with functional rate law completes."""
        model = Model.from_net(saturation_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 100), n_points=101)

        assert result.n_times == 101
        # Ga should be non-negative throughout
        assert np.all(result.species[:, 1] >= -1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy Sat/Hill .net rate-law rewrite
# ═══════════════════════════════════════════════════════════════════════════════


class TestSatHillNetRewrite:
    """Legacy Sat and Hill .net tokens are rewritten to Functional rates."""

    def test_sat_loads_with_rewrite_warning(self, sat_rewrite_net: Path):
        """Sat rate law loads and warns with correct Functional guidance."""
        with pytest.warns(UserWarning) as record:
            model = Model.from_net(sat_rewrite_net)

        assert model.n_reactions == 1
        messages = "\n".join(str(w.message) for w in record)
        assert "Legacy/deprecated" in messages
        assert "Sat" in messages
        assert "Functional" in messages
        assert "k/(K+S)" in messages
        assert "not k*S" in messages

    def test_hill_loads_with_rewrite_warning(self, hill_rewrite_net: Path):
        """Hill rate law loads and warns with correct Functional guidance."""
        with pytest.warns(UserWarning) as record:
            model = Model.from_net(hill_rewrite_net)

        assert model.n_reactions == 1
        messages = "\n".join(str(w.message) for w in record)
        assert "Legacy/deprecated" in messages
        assert "Hill" in messages
        assert "Functional" in messages
        assert "Vmax*S" in messages

    def test_already_functional_saturation_does_not_warn(
        self, saturation_net: Path, recwarn: pytest.WarningsRecorder
    ):
        """Already-rewritten functional fixture does not emit legacy warnings."""
        Model.from_net(saturation_net)
        assert not [
            w for w in recwarn if "Legacy/deprecated BioNetGen .net rate law" in str(w.message)
        ]

    def test_sat_rewrite_matches_explicit_functional(self, sat_rewrite_net: Path, tmp_path: Path):
        """Sat k K is numerically equivalent to f() = k/(K + S_obs)."""
        explicit = tmp_path / "sat_explicit.net"
        explicit.write_text(
            """begin parameters
    1 k3     1.52
    2 K4     114.418
    3 S0     100
end parameters
begin species
    1 S() S0
    2 P() 0
end species
begin functions
    1 sat_rate() k3/(K4+Stot)
end functions
begin reactions
    1 1 2 sat_rate
end reactions
begin groups
    1 Stot 1
end groups
"""
        )

        with pytest.warns(UserWarning, match="Sat"):
            legacy = Simulator(Model.from_net(sat_rewrite_net), method="ode").run(
                t_span=(0, 20), n_points=41
            )
        rewritten = Simulator(Model.from_net(explicit), method="ode").run(
            t_span=(0, 20), n_points=41
        )
        np.testing.assert_allclose(legacy.species, rewritten.species, rtol=1e-8, atol=1e-10)

    def test_sat_multi_rewrite_matches_explicit_functional(self, tmp_path: Path):
        """Sat k K1 K2 saturates the first two reactants."""
        legacy_net = tmp_path / "sat_multi.net"
        explicit_net = tmp_path / "sat_multi_explicit.net"
        common = """begin parameters
    1 k     0.5
    2 Ks    10.0
    3 Ke    3.0
    4 S0    25.0
    5 E0    7.0
end parameters
begin species
    1 S() S0
    2 E() E0
    3 P() 0
end species
"""
        legacy_net.write_text(
            common
            + """begin reactions
    1 1,2 2,3 Sat k Ks Ke
end reactions
begin groups
    1 Sobs 1
    2 Eobs 2
end groups
"""
        )
        explicit_net.write_text(
            common
            + """begin functions
    1 sat_rate() k/((Ks+Sobs)*(Ke+Eobs))
end functions
begin reactions
    1 1,2 2,3 sat_rate
end reactions
begin groups
    1 Sobs 1
    2 Eobs 2
end groups
"""
        )

        with pytest.warns(UserWarning, match="Sat"):
            legacy = Simulator(Model.from_net(legacy_net), method="ode").run(
                t_span=(0, 12), n_points=25
            )
        rewritten = Simulator(Model.from_net(explicit_net), method="ode").run(
            t_span=(0, 12), n_points=25
        )
        np.testing.assert_allclose(legacy.species, rewritten.species, rtol=1e-8, atol=1e-10)

    def test_hill_rewrite_matches_explicit_functional(
        self, hill_rewrite_net: Path, tmp_path: Path
    ):
        """Hill Vmax Kh h is rewritten as the per-reactant multiplier."""
        explicit = tmp_path / "hill_explicit.net"
        explicit.write_text(
            """begin parameters
    1 k     1.0
    2 K     50.0
    3 n     2.0
    4 S0    100
end parameters
begin species
    1 S() S0
    2 P() 0
end species
begin functions
    1 hill_rate() k*(Stot^(n-1))/((K^n)+(Stot^n))
end functions
begin reactions
    1 1 2 hill_rate
end reactions
begin groups
    1 Stot 1
end groups
"""
        )

        with pytest.warns(UserWarning, match="Hill"):
            legacy = Simulator(Model.from_net(hill_rewrite_net), method="ode").run(
                t_span=(0, 20), n_points=41
            )
        rewritten = Simulator(Model.from_net(explicit), method="ode").run(
            t_span=(0, 20), n_points=41
        )
        np.testing.assert_allclose(legacy.species, rewritten.species, rtol=1e-8, atol=1e-10)

    def test_rewrite_observables_not_leaked_to_result(self, sat_rewrite_net: Path, tmp_path: Path):
        """The internal `__bngsim_net_rewrite_obs_*` observables that back the
        rewritten Sat/Hill rate laws must stay inside the model and never reach
        the Result API or the .gdat output (issue #61). run_network emits only
        the user observables; bngsim must match that column set."""
        with pytest.warns(UserWarning, match="Sat"):
            model = Model.from_net(sat_rewrite_net)

        # The model keeps the scaffolding observable so the rewritten
        # functional rate law can reference the reactant count.
        raw = list(model._core.observable_names)
        assert any(n.startswith("__bngsim_net_rewrite_obs_") for n in raw), (
            "loader should still synthesize the internal observable: " + repr(raw)
        )

        result = Simulator(model, method="ode").run(t_span=(0, 20), n_points=41)

        # ...but it is filtered everywhere user-facing.
        assert result.observable_names == ["Stot"]
        assert result.n_observables == 1
        assert np.asarray(result.observables).shape == (41, 1)
        assert not any(n.startswith("__bngsim_") for n in result.observable_names)

        gdat = tmp_path / "sat.gdat"
        result.to_gdat(gdat)
        header = gdat.read_text().splitlines()[0].split()
        assert header == ["#", "time", "Stot"]
        assert not any("__bngsim_" in tok for tok in header)

    def test_multi_rewrite_all_internal_observables_filtered(self, tmp_path: Path):
        """A rule with multiple saturation constants synthesizes one internal
        observable per saturated reactant; all must be filtered (issue #61)."""
        net = tmp_path / "sat_multi.net"
        net.write_text(
            """begin parameters
    1 k     0.5
    2 Ks    10.0
    3 Ke    3.0
    4 S0    25.0
    5 E0    7.0
end parameters
begin species
    1 S() S0
    2 E() E0
    3 P() 0
end species
begin reactions
    1 1,2 2,3 Sat k Ks Ke
end reactions
begin groups
    1 Sobs 1
    2 Eobs 2
end groups
"""
        )
        with pytest.warns(UserWarning, match="Sat"):
            model = Model.from_net(net)

        raw = [n for n in model._core.observable_names if n.startswith("__bngsim_")]
        assert len(raw) == 2, "two saturated reactants → two internal observables: " + repr(raw)

        result = Simulator(model, method="ode").run(t_span=(0, 12), n_points=25)
        assert result.observable_names == ["Sobs", "Eobs"]
        assert result.n_observables == 2
        assert np.asarray(result.observables).shape == (25, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Michaelis-Menten tQSSA
# ═══════════════════════════════════════════════════════════════════════════════


class TestMMtQSSA:
    """Test Michaelis-Menten rate law with tQSSA formula.

    The tQSSA formula:
        sFree = 0.5 * ((S - Km - E) + sqrt((S - Km - E)^2 + 4*Km*S))
        rate  = kcat * sFree * E / (Km + sFree)
    is more accurate than sQSSA when E is comparable to S.
    """

    def test_mm_model_loads(self, mm_tqssa_net: Path):
        """MM tQSSA model loads correctly."""
        model = Model.from_net(mm_tqssa_net)
        assert model.n_species == 3
        assert model.n_reactions == 1

    def test_mm_enzyme_conserved(self, mm_tqssa_net: Path):
        """Enzyme (catalyst) concentration stays constant."""
        model = Model.from_net(mm_tqssa_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 200), n_points=201)

        # E should remain at 10
        np.testing.assert_allclose(result.species[:, 0], 10.0, atol=1e-6)

    def test_mm_substrate_product_conserved(self, mm_tqssa_net: Path):
        """S + P = S_0 throughout simulation."""
        model = Model.from_net(mm_tqssa_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 200), n_points=201)

        S_plus_P = result.species[:, 1] + result.species[:, 2]
        np.testing.assert_allclose(S_plus_P, 100.0, atol=1e-4)

    def test_mm_substrate_consumed(self, mm_tqssa_net: Path):
        """Substrate decreases substantially over time."""
        model = Model.from_net(mm_tqssa_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 200), n_points=201)

        S_end = result.species[-1, 1]
        P_end = result.species[-1, 2]
        assert S_end < 10.0
        assert P_end > 90.0

    def test_mm_initial_rate_tqssa(self, mm_tqssa_net: Path):
        """Initial rate matches tQSSA formula prediction."""
        model = Model.from_net(mm_tqssa_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 200), n_points=201)

        # At t=0: E=10, S=100, Km=50, kcat=1
        # tQSSA rate ≈ 6.515
        P_1 = result.species[1, 2]
        dt = result.time[1] - result.time[0]
        approx_rate = P_1 / dt
        assert 6.0 < approx_rate < 7.0


# ═══════════════════════════════════════════════════════════════════════════════
# SSA with zero-count reactant
# ═══════════════════════════════════════════════════════════════════════════════


class TestSSAZeroReactant:
    """Test SSA with zero-count reactant blocks reaction."""

    def test_abc_no_reaction_when_c_zero(self, ssa_abc_net: Path):
        """A+B+C->D with C=0: propensity=0, no reaction."""
        model = Model.from_net(ssa_abc_net)
        sim = Simulator(model, method="ssa")
        result = sim.run(t_span=(0, 100), n_points=101, seed=42)

        # A=5, B=3, C=0, D=0 — nothing should change
        np.testing.assert_allclose(result.species[:, 0], 5.0)
        np.testing.assert_allclose(result.species[:, 1], 3.0)
        np.testing.assert_allclose(result.species[:, 2], 0.0)
        np.testing.assert_allclose(result.species[:, 3], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Result from raw arrays
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultFromRawArrays:
    """Test constructing Result from raw arrays (for load/batch)."""

    def test_construct_from_arrays(self):
        """Result from raw arrays has correct properties."""
        time = np.linspace(0, 10, 11)
        species = np.random.rand(11, 3)
        observables = np.random.rand(11, 2)

        result = Result(
            core=None,
            _time=time,
            _species=species,
            _observables=observables,
            _species_names=["A()", "B()", "C()"],
            _observable_names=["A_tot", "B_tot"],
        )

        assert result.n_times == 11
        assert result.n_species == 3
        assert result.n_observables == 2
        assert result.species_names == ["A()", "B()", "C()"]
        assert result.observable_names == ["A_tot", "B_tot"]

    def test_construct_default_stats(self):
        """Default solver stats are zeros."""
        result = Result(core=None)
        assert result.solver_stats["n_steps"] == 0
        assert result.solver_stats["n_rhs_evals"] == 0

    def test_construct_custom_attrs(self):
        """Custom attrs passed to constructor."""
        result = Result(
            core=None,
            custom_attrs={"key": "value"},
        )
        assert result.custom_attrs["key"] == "value"

    def test_named_access_on_raw(self):
        """Named observable access works on raw-constructed Result."""
        time = np.arange(5.0)
        obs = np.arange(10.0).reshape(5, 2)

        result = Result(
            core=None,
            _time=time,
            _species=np.zeros((5, 1)),
            _observables=obs,
            _species_names=["X()"],
            _observable_names=["X_tot", "Y_tot"],
        )

        x_col = result.observables["X_tot"]
        assert x_col.shape == (5,)
        np.testing.assert_array_equal(x_col, obs[:, 0])


# ═══════════════════════════════════════════════════════════════════════════════
# File export on loaded (non-core) results
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultFileExportLoaded:
    """Test to_gdat / to_cdat on results without a C++ core."""

    def test_gdat_from_raw(self, tmp_path: Path):
        """to_gdat works on raw-constructed Result."""
        time = np.linspace(0, 10, 6)
        obs = np.column_stack(
            [
                100.0 * np.exp(-0.1 * time),
                100.0 * (1 - np.exp(-0.1 * time)),
            ]
        )

        result = Result(
            core=None,
            _time=time,
            _species=np.zeros((6, 2)),
            _observables=obs,
            _species_names=["A()", "B()"],
            _observable_names=["A_tot", "B_tot"],
        )

        out = tmp_path / "raw.gdat"
        result.to_gdat(out)
        assert out.exists()

        content = out.read_text()
        assert "time" in content
        assert "A_tot" in content
        assert "B_tot" in content
        # Should have header + 6 data lines
        lines = [ln for ln in content.strip().split("\n") if ln.strip()]
        assert len(lines) == 7  # 1 header + 6 data

    def test_cdat_from_raw(self, tmp_path: Path):
        """to_cdat works on raw-constructed Result."""
        time = np.linspace(0, 5, 3)
        species = np.array(
            [
                [100.0, 0.0],
                [60.0, 40.0],
                [36.0, 64.0],
            ]
        )

        result = Result(
            core=None,
            _time=time,
            _species=species,
            _observables=np.zeros((3, 2)),
            _species_names=["A()", "B()"],
            _observable_names=["X", "Y"],
        )

        out = tmp_path / "raw.cdat"
        result.to_cdat(out)
        assert out.exists()

        content = out.read_text()
        assert "time" in content
        assert "A()" in content

    def test_gdat_round_trip_via_hdf5(self, simple_decay_net: Path, tmp_path: Path):
        """HDF5 load → gdat export still works."""
        pytest.importorskip("h5py")

        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        h5_path = tmp_path / "result.h5"
        result.save(h5_path)
        loaded = Result.load(h5_path)

        gdat_path = tmp_path / "loaded.gdat"
        loaded.to_gdat(gdat_path)
        assert gdat_path.exists()

        content = gdat_path.read_text()
        assert "A_tot" in content


# ═══════════════════════════════════════════════════════════════════════════════
# to_csv — plain delimited text export (issue #11)
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultToCsv:
    """Plain delimited-text export via Result.to_csv."""

    def test_csv_observables_round_trips_through_numpy(
        self, simple_decay_net: Path, tmp_path: Path
    ):
        """to_csv default writes CSV that np.loadtxt reads back."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        out = tmp_path / "out.csv"
        result.to_csv(out)
        assert out.exists()

        first_line = out.read_text().splitlines()[0]
        # No BNG-style '#' prefix; plain header row.
        assert not first_line.startswith("#")
        assert first_line.split(",")[0] == "time"
        assert "A_tot" in first_line

        loaded = np.loadtxt(out, delimiter=",", skiprows=1)
        assert loaded.shape == (11, 1 + result.n_observables)
        np.testing.assert_allclose(loaded[:, 0], result.time)
        np.testing.assert_allclose(
            loaded[:, 1:],
            np.asarray(result.observables),
            rtol=1e-10,
        )

    def test_csv_species_kind(self, simple_decay_net: Path, tmp_path: Path):
        """kind='species' writes species columns instead of observables."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=6)

        out = tmp_path / "species.csv"
        result.to_csv(out, kind="species")

        first_line = out.read_text().splitlines()[0]
        cols = first_line.split(",")
        assert cols[0] == "time"
        assert cols[1:] == result.species_names

    def test_csv_tsv_separator(self, simple_decay_net: Path, tmp_path: Path):
        """sep='\\t' produces a tab-separated file readable by np.loadtxt."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=6)

        out = tmp_path / "out.tsv"
        result.to_csv(out, sep="\t")

        first_line = out.read_text().splitlines()[0]
        assert "\t" in first_line
        loaded = np.loadtxt(out, delimiter="\t", skiprows=1)
        assert loaded.shape[0] == 6
        np.testing.assert_allclose(loaded[:, 0], result.time)

    def test_csv_no_header(self, simple_decay_net: Path, tmp_path: Path):
        """header=False omits the header row."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=4)

        out = tmp_path / "no_header.csv"
        result.to_csv(out, header=False)

        loaded = np.loadtxt(out, delimiter=",")
        assert loaded.shape == (4, 1 + result.n_observables)

    def test_csv_no_time_column(self, simple_decay_net: Path, tmp_path: Path):
        """include_time=False drops the leading time column."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=5)

        out = tmp_path / "no_time.csv"
        result.to_csv(out, include_time=False)

        first_line = out.read_text().splitlines()[0]
        assert "time" not in first_line.split(",")
        loaded = np.loadtxt(out, delimiter=",", skiprows=1)
        assert loaded.shape == (5, result.n_observables)

    def test_csv_from_raw_loaded_result(self, tmp_path: Path):
        """to_csv works on a raw-constructed Result (no C++ core)."""
        time = np.linspace(0, 10, 6)
        obs = np.column_stack(
            [
                100.0 * np.exp(-0.1 * time),
                100.0 * (1 - np.exp(-0.1 * time)),
            ]
        )
        result = Result(
            core=None,
            _time=time,
            _species=np.zeros((6, 2)),
            _observables=obs,
            _species_names=["A()", "B()"],
            _observable_names=["A_tot", "B_tot"],
        )

        out = tmp_path / "raw.csv"
        result.to_csv(out)

        first_line = out.read_text().splitlines()[0]
        assert first_line == "time,A_tot,B_tot"
        loaded = np.loadtxt(out, delimiter=",", skiprows=1)
        np.testing.assert_allclose(loaded[:, 1:], obs, rtol=1e-10)

    def test_csv_pandas_compatibility(self, simple_decay_net: Path, tmp_path: Path):
        """pandas.read_csv loads the file with the bngsim column names."""
        pd = pytest.importorskip("pandas")

        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        out = tmp_path / "for_pandas.csv"
        result.to_csv(out)

        df = pd.read_csv(out)
        assert list(df.columns) == ["time"] + result.observable_names
        np.testing.assert_allclose(df["time"].to_numpy(), result.time)

    def test_csv_round_trip_through_hdf5(self, simple_decay_net: Path, tmp_path: Path):
        """HDF5 load → to_csv works for loaded results too."""
        pytest.importorskip("h5py")

        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        h5 = tmp_path / "r.h5"
        result.save(h5)
        loaded = Result.load(h5)

        csv = tmp_path / "loaded.csv"
        loaded.to_csv(csv)
        first_line = csv.read_text().splitlines()[0]
        assert first_line.split(",")[0] == "time"
        assert "A_tot" in first_line

    def test_csv_invalid_kind(self, simple_decay_net: Path, tmp_path: Path):
        """Unknown kind raises ValueError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=3)

        with pytest.raises(ValueError, match="kind"):
            result.to_csv(tmp_path / "x.csv", kind="expressions")

    def test_csv_invalid_separator(self, simple_decay_net: Path, tmp_path: Path):
        """Multi-char separator raises ValueError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=3)

        with pytest.raises(ValueError, match="single character"):
            result.to_csv(tmp_path / "x.csv", sep=", ")

    def test_csv_rejects_batch_3d(self, simple_decay_net: Path, tmp_path: Path):
        """Squeezed (3-D) batch result is rejected with a clear message."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        params = [{"k1": k} for k in (0.1, 0.5, 1.0)]
        batch = sim.run_batch(t_span=(0, 10), n_points=6, params=params, squeeze=True)
        assert batch.observables.ndim == 3

        with pytest.raises(ValueError, match="2-D single-sim"):
            batch.to_csv(tmp_path / "batch.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Result.xr / Result.to_xarray — AMICI-style labeled-array shim
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultXarray:
    """Labeled-array access via xarray (AMICI-flavored)."""

    def test_xr_species_dims_and_coords(self, simple_decay_net: Path):
        pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        sp = result.xr.species
        assert sp.dims == ("time", "state")
        assert list(sp.coords["state"].values) == result.species_names
        assert sp.shape == (11, result.n_species)
        np.testing.assert_allclose(sp.coords["time"].values, result.time)

    def test_xr_observables_select_by_name(self, simple_decay_net: Path):
        pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        obs = result.xr.observables
        assert obs.dims == ("time", "observable")
        a_tot = obs.sel(observable="A_tot").values
        np.testing.assert_allclose(a_tot, np.asarray(result.observables["A_tot"]))

    def test_xr_sensitivities_dims(self, simple_decay_net: Path):
        pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        sens = result.xr.sensitivities
        assert sens.dims == ("time", "state", "parameter")
        assert list(sens.coords["parameter"].values) == ["k1"]
        a_k1 = sens.sel(parameter="k1", state="A()").values
        np.testing.assert_allclose(a_k1, result.sensitivities[:, 0, 0])

    def test_xr_missing_sensitivities_raises(self, simple_decay_net: Path):
        pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=5)

        with pytest.raises(AttributeError, match="no sensitivities"):
            _ = result.xr.sensitivities

    def test_xr_unknown_field_raises(self, simple_decay_net: Path):
        pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=5)

        with pytest.raises(AttributeError, match="no field 'foo'"):
            _ = result.xr.foo

    def test_to_xarray_dataset_shape(self, simple_decay_net: Path):
        xr = pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        ds = result.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars) >= {"species", "observables", "sensitivities"}
        np.testing.assert_allclose(ds.coords["time"].values, result.time)
        assert ds.sensitivities.dims == ("time", "state", "parameter")

    def test_to_xarray_carries_seed_and_custom_attrs(self, simple_decay_net: Path):
        pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")
        result = sim.run(t_span=(0, 10), n_points=11, seed=12345)
        result.custom_attrs["experiment"] = "smoke"

        ds = result.to_xarray()
        assert ds.attrs.get("seed") == 12345
        assert ds.attrs.get("experiment") == "smoke"

    def test_to_xarray_rejects_batch_3d(self, simple_decay_net: Path):
        pytest.importorskip("xarray")
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        params = [{"k1": k} for k in (0.1, 0.5, 1.0)]
        batch = sim.run_batch(t_span=(0, 10), n_points=6, params=params, squeeze=True)
        with pytest.raises(RuntimeError, match="single-simulation"):
            batch.to_xarray()

    def test_xr_from_hdf5_loaded_result(self, simple_decay_net: Path, tmp_path: Path):
        pytest.importorskip("xarray")
        pytest.importorskip("h5py")

        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        h5 = tmp_path / "r.h5"
        result.save(h5)
        loaded = Result.load(h5)

        sp = loaded.xr.species
        assert sp.dims == ("time", "state")
        np.testing.assert_allclose(
            sp.sel(state="A()").values,
            result.species[:, 0],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Batch edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchEdgeCases:
    """Additional edge cases for run_batch."""

    def test_batch_single_param_set(self, simple_decay_net: Path):
        """Batch with exactly one param set."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}],
        )
        assert len(results) == 1
        assert results[0].species[0, 0] == pytest.approx(100.0)

    def test_batch_with_tolerances(self, simple_decay_net: Path):
        """Batch passes tolerance options through."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}],
            rtol=1e-12,
            atol=1e-12,
        )
        assert len(results) == 1

        # Tighter tolerance should give very accurate result
        A_end = results[0].species[-1, 0]
        expected = 100.0 * math.exp(-1.0)
        assert A_end == pytest.approx(expected, abs=1e-6)

    def test_batch_invalid_n_points(self, simple_decay_net: Path):
        """Batch with invalid n_points raises ValueError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        with pytest.raises(ValueError, match="n_points"):
            sim.run_batch(
                t_span=(0, 10),
                n_points=1,
                params=[{"k1": 0.1}],
            )

    def test_batch_squeeze_with_parallel(self, simple_decay_net: Path):
        """squeeze + num_processors together."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        batch = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}, {"k1": 0.5}],
            num_processors=2,
            squeeze=True,
        )

        assert isinstance(batch, Result)
        assert batch.species.ndim == 3
        assert batch.species.shape == (2, 11, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Model introspection edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelIntrospection:
    """Additional model introspection tests."""

    def test_n_functions_property(self, time_func_net: Path):
        """n_functions reports function count."""
        model = Model.from_net(time_func_net)
        assert model.n_functions >= 1

    def test_n_functions_zero(self, simple_decay_net: Path):
        """Model without functions has n_functions=0."""
        model = Model.from_net(simple_decay_net)
        assert model.n_functions == 0

    def test_reversible_model_species(self, reversible_net: Path):
        """Reversible model has 3 species and 2 reactions."""
        model = Model.from_net(reversible_net)
        assert model.n_species == 3
        assert model.n_reactions == 2
        assert "A()" in model.species_names
        assert "B()" in model.species_names
        assert "C()" in model.species_names

    def test_clone_preserves_concentrations(self, simple_decay_net: Path):
        """Clone preserves modified concentrations."""
        model = Model.from_net(simple_decay_net)
        model.set_concentration("A()", 42.0)

        clone = model.clone()
        assert clone.get_concentration("A()") == pytest.approx(42.0)

    def test_multiple_resets(self, simple_decay_net: Path):
        """Multiple resets are safe."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        for _ in range(5):
            sim.run(t_span=(0, 10), n_points=11)
            model.reset()
            assert model.get_concentration("A()") == (pytest.approx(100.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level API
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleAPI:
    """Test module-level attributes and functions."""

    def test_version(self):
        """__version__ is a string."""
        assert isinstance(bngsim.__version__, str)
        assert "." in bngsim.__version__

    def test_all_exports(self):
        """__all__ contains expected names."""
        assert "Model" in bngsim.__all__
        assert "Simulator" in bngsim.__all__
        assert "Result" in bngsim.__all__
        assert "reserved_names" in bngsim.__all__
        assert "configure_logging" in bngsim.__all__
        assert "BngsimError" in bngsim.__all__
        assert "StopConditionMet" in bngsim.__all__

    def test_reserved_names_callable(self):
        """reserved_names is callable from module."""
        names = bngsim.reserved_names()
        assert isinstance(names, dict)
        assert "constants" in names
        assert "functions" in names


# ═══════════════════════════════════════════════════════════════════════════════
# Table functions (tfun) — ADR-001 §3.12
# ═══════════════════════════════════════════════════════════════════════════════


class TestTfunFromFile:
    """Test table functions loaded from .net files with tfun() syntax."""

    def test_tfun_time_indexed_loads(self, tfun_time_indexed_net: Path):
        """Model with time-indexed tfun loads correctly."""
        model = Model.from_net(tfun_time_indexed_net)
        assert model.n_table_functions == 1
        assert model.table_function_names == ["cumNcases"]
        assert model.n_functions == 1

    def test_tfun_time_indexed_simulate(self, tfun_time_indexed_net: Path):
        """Time-indexed tfun drives production rate during simulation."""
        model = Model.from_net(tfun_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 7), n_points=71)

        # B(0) = 0
        assert result.species[0, 1] == pytest.approx(0.0, abs=1e-10)

        # cumNcases is 0 at t=0..1, so B stays near 0 until t=1
        # After t=2, cumNcases rises → B accumulates
        B_end = result.species[-1, 1]
        assert B_end > 10.0, "B(7) should be > 10 (integral of rising cumNcases)"

    def test_tfun_time_indexed_expression_output(self, tfun_time_indexed_net: Path):
        """tfun values appear in expression columns of result."""
        model = Model.from_net(tfun_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 7), n_points=71)

        # The function cumNcases() should appear in expressions
        assert result.n_expressions >= 1
        expr_names = result.expression_names
        assert "cumNcases" in expr_names

        # At t=0, cumNcases should be 0
        cumNcases_col = result.expressions["cumNcases"]
        assert cumNcases_col[0] == pytest.approx(0.0, abs=1e-6)

        # At t=7, cumNcases should be 22 (last data point)
        assert cumNcases_col[-1] == pytest.approx(22.0, abs=0.1)

    def test_tfun_time_indexed_gdat_export(self, tfun_time_indexed_net: Path, tmp_path: Path):
        """GDAT export includes observable columns (tfun expressions in Result)."""
        model = Model.from_net(tfun_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 7), n_points=71)

        gdat = tmp_path / "tfun.gdat"
        result.to_gdat(gdat)
        assert gdat.exists()

        content = gdat.read_text()
        assert "time" in content
        assert "A_tot" in content
        assert "B_tot" in content
        # tfun expression values are accessible via result.expressions
        assert result.n_expressions >= 1
        assert "cumNcases" in result.expression_names

    def test_to_gdat_default_omits_function_columns(
        self, tfun_time_indexed_net: Path, tmp_path: Path
    ):
        """Default to_gdat() writes observables only (matches BNG2.pl without
        print_functions=>1) — the function header is not present."""
        model = Model.from_net(tfun_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 7), n_points=71)

        gdat = tmp_path / "default.gdat"
        result.to_gdat(gdat)
        header = gdat.read_text().splitlines()[0]
        assert "cumNcases" not in header

    def test_to_gdat_print_functions_appends_bare_columns(
        self, tfun_time_indexed_net: Path, tmp_path: Path
    ):
        """to_gdat(print_functions=True) appends the user function column with a
        BARE header (no "()"), after the observables. Issue #58 normalised the
        per-method "()" behaviour #53 had introduced for nf."""
        model = Model.from_net(tfun_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 7), n_points=71)

        gdat = tmp_path / "funcs.gdat"
        result.to_gdat(gdat, print_functions=True)
        lines = gdat.read_text().splitlines()
        header_cols = lines[0].lstrip("#").split()

        # Function column appears bare, after the observables; "()" never appears.
        assert "cumNcases" in header_cols
        assert "cumNcases()" not in header_cols
        assert "()" not in lines[0]
        assert header_cols.index("cumNcases") > header_cols.index("A_tot")

        # Last data row's function value matches the in-memory column (=22 at t=7).
        last_vals = [float(x) for x in lines[-1].split()]
        assert last_vals[header_cols.index("cumNcases")] == pytest.approx(22.0, abs=0.1)

    def test_to_gdat_print_functions_omits_rate_laws(self, time_func_net: Path, tmp_path: Path):
        """A model whose only function is the synthetic _rateLaw1 gets no
        function column with print_functions alone (rate laws need
        print_rate_laws)."""
        model = Model.from_net(time_func_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        gdat = tmp_path / "ratelaw.gdat"
        result.to_gdat(gdat, print_functions=True)
        header = gdat.read_text().splitlines()[0]
        assert "_rateLaw1" not in header

    def test_to_gdat_print_rate_laws_includes_bare_rate_law(
        self, time_func_net: Path, tmp_path: Path
    ):
        """to_gdat(print_rate_laws=True) appends the synthetic _rateLawN column
        with a BARE header; print_functions=False keeps user funcs out."""
        model = Model.from_net(time_func_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        gdat = tmp_path / "ratelaw.gdat"
        result.to_gdat(gdat, print_rate_laws=True)
        lines = gdat.read_text().splitlines()
        header_cols = lines[0].lstrip("#").split()
        assert "_rateLaw1" in header_cols
        assert "_rateLaw1()" not in header_cols
        assert "()" not in lines[0]

        # _rateLaw1() = time()+1, so the last point (t=10) is 11.
        last_vals = [float(x) for x in lines[-1].split()]
        assert last_vals[header_cols.index("_rateLaw1")] == pytest.approx(11.0, abs=1e-6)

    def test_gdat_expression_names_property(self, tfun_time_indexed_net: Path):
        """gdat_expression_names is bare (no "()"), identical to
        expression_names — issue #58."""
        model = Model.from_net(tfun_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 7), n_points=11)

        assert result.expression_names == ["cumNcases"]
        assert result.gdat_expression_names == ["cumNcases"]
        # Same bare label is exposed at the core boundary for consumers reading _core.
        assert list(result._core.gdat_expression_names) == ["cumNcases"]

    def test_raw_expression_exposes_rate_laws(self, time_func_net: Path):
        """raw_expression_* recovers the _rateLawN columns that the default
        expression_* filter drops (the #58 recoverability fix)."""
        model = Model.from_net(time_func_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        # Default view filters the synthetic rate law out.
        assert result.expression_names == []
        assert result.n_expressions == 0
        # Raw view recovers it (bare name), with the integrated values.
        assert result.raw_expression_names == ["_rateLaw1"]
        assert result.raw_n_expressions == 1
        col = result.raw_expressions["_rateLaw1"]
        assert col[-1] == pytest.approx(11.0, abs=1e-6)

    def test_gdat_header_schema_identical_across_methods(self, data_dir: Path, tmp_path: Path):
        """The .gdat function-column header is byte-identical across simulation
        methods for the same model — the core #58 guarantee. func_composition
        carries both user functions (fRate0, fRate_cond) and synthetic rate
        laws (_rateLaw1, _rateLaw2), exercised here with both flags on."""
        model = Model.from_net(data_dir / "func_composition.net")

        headers = {}
        for method in ("ode", "ssa"):
            result = Simulator(model, method=method).run(t_span=(0, 5), n_points=6, seed=1)
            out = tmp_path / f"{method}.gdat"
            result.to_gdat(out, print_functions=True, print_rate_laws=True)
            headers[method] = out.read_text().splitlines()[0]

        assert headers["ode"] == headers["ssa"]
        # Bare names, in declared order, no "()".
        cols = headers["ode"].lstrip("#").split()
        assert cols == [
            "time",
            "A_tot",
            "B_tot",
            "C_tot",
            "fRate0",
            "_rateLaw1",
            "_rateLaw2",
            "fRate_cond",
        ]
        assert "()" not in headers["ode"]

    def test_tfun_step_method_from_net(self, tfun_step_time_indexed_net: Path):
        """Step-interpolated tfun from .net should use piecewise-constant values."""
        model = Model.from_net(tfun_step_time_indexed_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 2), n_points=201)

        # Integral of stepDrive over [0,2]: 0*1 + 10*1 = 10
        B_end = result.species[-1, 1]
        assert B_end == pytest.approx(10.0, abs=0.5)


class TestTfunParamIndexed:
    """Test parameter-indexed table functions."""

    def test_tfun_param_indexed_loads(self, tfun_param_indexed_net: Path):
        """Model with parameter-indexed tfun loads correctly."""
        model = Model.from_net(tfun_param_indexed_net)
        assert model.n_table_functions == 1
        assert model.table_function_names == ["response"]

    def test_tfun_param_indexed_simulate(self, tfun_param_indexed_net: Path):
        """Parameter-indexed tfun drives reaction rate."""
        model = Model.from_net(tfun_param_indexed_net)
        sim = Simulator(model, method="ode")

        # drug_conc=1.0 → response=50.0 → fast decay
        result = sim.run(t_span=(0, 1), n_points=11)
        A_end = result.species[-1, 0]
        assert A_end < 1.0, "A should be nearly 0 at t=1 with response=50"

    def test_tfun_param_indexed_zero_dose(self, tfun_param_indexed_net: Path):
        """Zero drug_conc → response=0 → no reaction."""
        model = Model.from_net(tfun_param_indexed_net)
        model.set_param("drug_conc", 0.0)
        model.reset()

        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 1), n_points=11)
        A_end = result.species[-1, 0]
        assert A_end == pytest.approx(100.0, abs=0.1), "A should stay ~100 when drug_conc=0"

    def test_tfun_param_indexed_dose_sweep(self, tfun_param_indexed_net: Path):
        """Higher drug_conc → faster decay (dose-response curve)."""
        model = Model.from_net(tfun_param_indexed_net)
        sim = Simulator(model, method="ode")

        A_finals = []
        for dose in [0.0, 0.1, 0.5, 1.0, 5.0]:
            model.set_param("drug_conc", dose)
            model.reset()
            result = sim.run(t_span=(0, 0.5), n_points=11)
            A_finals.append(result.species[-1, 0])

        # Each higher dose should give more decay
        for i in range(len(A_finals) - 1):
            assert A_finals[i] >= A_finals[i + 1] - 0.1, (
                f"A should decrease with increasing dose: {A_finals}"
            )


class TestTfunFromArrays:
    """Test add_table_function() with times/values arrays."""

    def test_tfun_arrays_basic(self, simple_decay_net: Path):
        """Add a time-indexed tfun from arrays."""
        model = Model.from_net(simple_decay_net)
        model.add_table_function("scale", times=[0, 5, 10, 50], values=[1, 1, 2, 2])
        assert model.n_table_functions == 1
        assert model.table_function_names == ["scale"]

    def test_tfun_arrays_validation(self, simple_decay_net: Path):
        """Invalid arrays raise errors."""
        model = Model.from_net(simple_decay_net)

        # Too few points
        with pytest.raises(ModelError):
            model.add_table_function("bad", times=[0], values=[1])

        # Mismatched sizes
        with pytest.raises(ModelError):
            model.add_table_function("bad", times=[0, 1], values=[1, 2, 3])

    def test_tfun_arrays_argument_errors(self, simple_decay_net: Path):
        """Invalid argument combinations raise ValueError."""
        model = Model.from_net(simple_decay_net)

        # Both file and times
        with pytest.raises(ValueError, match="Cannot specify both"):
            model.add_table_function("bad", file="x.tfun", times=[0, 1], values=[1, 2])

        # Neither file nor times
        with pytest.raises(ValueError, match="Must specify"):
            model.add_table_function("bad")

    def test_tfun_arrays_with_custom_index(self, simple_decay_net: Path):
        """Add tfun indexed by a parameter name."""
        model = Model.from_net(simple_decay_net)
        model.add_table_function(
            "scale",
            times=[0, 0.5, 1.0],
            values=[0, 50, 100],
            index="k1",
        )
        assert model.n_table_functions == 1

    def test_tfun_arrays_invalid_method(self, simple_decay_net: Path):
        """Unsupported interpolation method should raise ValueError."""
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ValueError, match="Expected 'linear' or 'step'"):
            model.add_table_function(
                "scale",
                times=[0, 1, 2],
                values=[0, 10, 20],
                method="cubic",
            )


class TestTfunClone:
    """Test that tfuns survive model.clone()."""

    def test_tfun_clone_preserves_count(self, tfun_time_indexed_net: Path):
        """Clone preserves table function count."""
        model = Model.from_net(tfun_time_indexed_net)
        assert model.n_table_functions == 1

        clone = model.clone()
        assert clone.n_table_functions == 1
        assert clone.table_function_names == ["cumNcases"]

    def test_tfun_clone_simulates_independently(self, tfun_time_indexed_net: Path):
        """Cloned model with tfun simulates correctly."""
        model = Model.from_net(tfun_time_indexed_net)
        clone = model.clone()

        sim = Simulator(clone, method="ode")
        result = sim.run(t_span=(0, 5), n_points=51)

        assert result.n_times == 51
        B_end = result.species[-1, 1]
        assert B_end > 0.0, "Clone: B should grow via tfun-driven rate"

    def test_tfun_clone_independent_params(self, tfun_param_indexed_net: Path):
        """Cloned model has independent tfun parameter bindings."""
        model = Model.from_net(tfun_param_indexed_net)
        clone = model.clone()

        # Set different drug_conc on clone
        clone.set_param("drug_conc", 10.0)

        # Original should still have drug_conc=1.0
        assert model.get_param("drug_conc") == pytest.approx(1.0)
        assert clone.get_param("drug_conc") == pytest.approx(10.0)

        # Simulate both — clone should decay much faster
        sim_orig = Simulator(model, method="ode")
        sim_clone = Simulator(clone, method="ode")

        # Compare at t=0.1, where both species are still well above the atol
        # noise floor. The original check was at t=0.5, by which point BOTH A's
        # have fully decayed to ~±1e-8 (atol-level noise) and their ordering is
        # meaningless — the faster decay of the higher dose is unambiguous only
        # while A is still meaningfully positive (orig≈0.67, clone≈0.0075 here).
        # (The RHS now evaluates rate laws on a nonnegative-clamped state, GH
        # #135, so a fully-decayed species settles at ~0 instead of drifting to a
        # small negative — which flipped the old noise-level comparison.)
        r_orig = sim_orig.run(t_span=(0, 0.1), n_points=11)
        r_clone = sim_clone.run(t_span=(0, 0.1), n_points=11)

        A_orig = r_orig.species[-1, 0]
        A_clone = r_clone.species[-1, 0]
        assert A_orig > 1e-3, "sanity: original is still meaningfully positive at t=0.1"
        assert A_clone < A_orig, "Clone with higher dose should decay faster"

    def test_tfun_from_arrays_survives_clone(self, simple_decay_net: Path):
        """tfun added from arrays survives clone."""
        model = Model.from_net(simple_decay_net)
        model.add_table_function(
            "boost",
            times=[0.0, 5.0, 10.0],
            values=[1.0, 2.0, 3.0],
        )
        assert model.n_table_functions == 1

        clone = model.clone()
        assert clone.n_table_functions == 1
        assert clone.table_function_names == ["boost"]

    def test_step_tfun_clone_preserves_method(self, tfun_step_time_indexed_net: Path):
        """Clone should preserve step interpolation mode for tfuns."""
        model = Model.from_net(tfun_step_time_indexed_net)
        clone = model.clone()

        sim = Simulator(clone, method="ode")
        result = sim.run(t_span=(0, 2), n_points=201)
        B_end = result.species[-1, 1]
        assert B_end == pytest.approx(10.0, abs=0.5)


class TestTfunIndexCanonicalization:
    """GH #35: BNG's TfunReader normalizes tfun index/header tokens before
    validation — case-insensitive `time`/`t`, and trailing `()` stripped on
    any column. bngsim must match those semantics.

    Reference: `bng2/Perl2/TfunReader.pm:84-93` on RuleWorld/bionetgen master.
    """

    def test_uppercase_time_header_loads(self, tfun_uppercase_time_net: Path):
        """`.tfun` header `# Time  cumNcases()` with tfun call passing
        `Time` as the index must resolve to the model's time index."""
        model = Model.from_net(tfun_uppercase_time_net)
        assert model.n_table_functions == 1
        assert model.table_function_names == ["cumNcases"]

    def test_uppercase_time_simulates_like_lowercase(
        self,
        tfun_uppercase_time_net: Path,
        tfun_time_indexed_net: Path,
    ):
        """Trajectory must match the canonical lowercase-`time` fixture
        — case + paren differences in the header are purely cosmetic."""
        ref = Simulator(Model.from_net(tfun_time_indexed_net), method="ode").run(
            t_span=(0, 7), n_points=71
        )
        got = Simulator(Model.from_net(tfun_uppercase_time_net), method="ode").run(
            t_span=(0, 7), n_points=71
        )
        assert got.species[-1, 1] == pytest.approx(ref.species[-1, 1], rel=1e-9)

    def test_paren_param_header_loads(self, tfun_paren_param_net: Path):
        """`.tfun` column-1 header `drug_conc()` must be accepted as the
        `drug_conc` parameter index after BNG-style `()` stripping."""
        model = Model.from_net(tfun_paren_param_net)
        assert model.n_table_functions == 1
        assert model.table_function_names == ["response"]

    def test_paren_param_simulates_like_bare(
        self,
        tfun_paren_param_net: Path,
        tfun_param_indexed_net: Path,
    ):
        """Trajectory must match the bare-`drug_conc` fixture."""
        ref = Simulator(Model.from_net(tfun_param_indexed_net), method="ode").run(
            t_span=(0, 1), n_points=11
        )
        got = Simulator(Model.from_net(tfun_paren_param_net), method="ode").run(
            t_span=(0, 1), n_points=11
        )
        assert got.species[-1, 0] == pytest.approx(ref.species[-1, 0], rel=1e-9)


class TestTfunWrapperForm:
    """GH #33: tfun(...) embedded inside arithmetic.

    The .tfun data file is the same step-rising series ``1.0, 2.0, 4.0`` at
    ``t = 0, 1, 2``. With piecewise-linear interpolation:

        ∫₀² tfun(t) dt = 4.5     (bare control — was already correct)
        ∫₀² (tfun(t) + 5)/10 dt = (4.5 + 10) / 10 = 1.45    (wrapper case)
    """

    def test_wrap_single_interp_preserves_wrapper(self, wrap_single_net: Path):
        """Interp path: wrapper math survives load + simulate."""
        model = Model.from_net(wrap_single_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0.0, 2.0), n_points=21)
        Xtot_end = result.observables["Xtot"][-1]
        assert Xtot_end == pytest.approx(1.45, abs=1e-3)

    def test_wrap_single_uses_synthetic_table_name(self, wrap_single_net: Path):
        """Wrapper-form registers the table under a synthetic name so the BNG
        function's wrapper math can call it without colliding."""
        model = Model.from_net(wrap_single_net)
        assert model.n_table_functions == 1
        assert model.table_function_names == ["f_complex__tfun0"]

    def test_wrap_single_codegen_runs_and_matches_interp(self, wrap_single_net: Path):
        """Codegen path: wrapper math + embedded tfun_eval callback gives the
        same numeric answer as the interpreter."""
        model = Model.from_net(wrap_single_net)
        sim = Simulator(model, method="ode", codegen=True, net_path=wrap_single_net)
        result = sim.run(t_span=(0.0, 2.0), n_points=21)
        Xtot_end = result.observables["Xtot"][-1]
        assert Xtot_end == pytest.approx(1.45, abs=1e-3)

    def test_wrap_single_codegen_emits_callback_inside_wrapper(self, wrap_single_net: Path):
        """The emitted C source for the wrapper function body must contain a
        tfun_eval callback nested inside the wrapper arithmetic, not a raw
        tfun(...) token that would fail to compile."""
        from bngsim._codegen import generate_rhs_c

        src = generate_rhs_c(str(wrap_single_net))
        func_lines = [line for line in src.splitlines() if "func_f_complex" in line]
        assert func_lines, "codegen did not emit func_f_complex assignment"
        rhs = func_lines[0]
        assert "tfun_eval(" in rhs, f"missing tfun_eval callback in: {rhs}"
        # Wrapper math must surround the callback (parentheses + the +5/p[0]
        # tail). Codegen v9 float-ifies integer literals, so the constant is
        # emitted as ``5.0`` (numerically identical) — match either form.
        assert "+5.0)/p[" in rhs.replace(" ", ""), f"wrapper math lost in: {rhs}"
        # Raw tfun(...) token must NOT survive — that's the pre-fix compile failure.
        assert "tfun('" not in rhs and 'tfun("' not in rhs, f"raw tfun token in: {rhs}"
