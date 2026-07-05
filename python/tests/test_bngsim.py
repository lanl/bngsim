"""bngsim Python binding tests (Phase B).

These tests mirror the C++ Phase A tests but exercise the Python API.
They verify:
- Model loading and introspection
- ODE simulation with analytical solution comparison
- SSA simulation with reproducibility
- Parameter update and reset
- Named observable access
- Exception handling
- Reserved names API
- File export
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import Model, Result, Simulator
from bngsim._exceptions import BngsimError, ModelError, ParameterError, SimulationError

# ═══════════════════════════════════════════════════════════════════════════════
# Model loading and introspection
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelLoading:
    """Test Model.from_net and model introspection."""

    def test_load_simple_decay(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        assert model.n_species == 2
        assert model.n_reactions == 1
        assert model.n_observables == 2
        assert model.n_parameters == 1

    def test_param_access(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        assert model.get_param("k1") == pytest.approx(0.1)
        assert "k1" in model.param_names

    def test_species_names(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        names = model.species_names
        assert len(names) == 2
        assert "A()" in names
        assert "B()" in names

    def test_observable_names(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        names = model.observable_names
        assert len(names) == 2
        assert "A_tot" in names
        assert "B_tot" in names

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Model.from_net("/nonexistent/path.net")

    def test_repr(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        r = repr(model)
        assert "species=2" in r
        assert "reactions=1" in r

    def test_clone(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        clone = model.clone()

        # Clone is independent
        clone.set_param("k1", 99.0)
        assert model.get_param("k1") == pytest.approx(0.1)
        assert clone.get_param("k1") == pytest.approx(99.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter access
# ═══════════════════════════════════════════════════════════════════════════════


class TestParameters:
    """Test parameter get/set/set_params."""

    def test_set_param(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        model.set_param("k1", 1.0)
        assert model.get_param("k1") == pytest.approx(1.0)

    def test_set_params_dict(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        model.set_params({"k1": 2.5})
        assert model.get_param("k1") == pytest.approx(2.5)

    def test_unknown_param_raises(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ParameterError):
            model.set_param("nonexistent", 1.0)

    def test_unknown_param_get_raises(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ParameterError):
            model.get_param("nonexistent")

    def test_set_params_unknown_raises(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ParameterError, match="Unknown parameter"):
            model.set_params({"k1": 1.0, "bogus": 2.0})

    def test_set_params_bad_value_no_partial_update(self, simple_decay_net: Path):
        """Regression: set_params must be atomic — bad value doesn't change any param."""
        model = Model.from_net(simple_decay_net)
        original_k1 = model.get_param("k1")
        with pytest.raises(ParameterError, match="Invalid value"):
            model.set_params({"k1": "not_a_number"})
        # k1 must be unchanged
        assert model.get_param("k1") == pytest.approx(original_k1)

    def test_set_params_bad_value_type(self, simple_decay_net: Path):
        """Regression: non-numeric values raise ParameterError, not ValueError."""
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ParameterError):
            model.set_params({"k1": None})


# ═══════════════════════════════════════════════════════════════════════════════
# ODE simulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestOdeSimulation:
    """Test ODE (CVODE) simulation via Simulator(method='ode')."""

    def test_simple_decay_analytical(self, simple_decay_net: Path):
        """Compare against analytical solution: A(t) = 100*exp(-0.1*t)."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 50), n_points=51)

        assert isinstance(result, Result)
        assert result.time.shape == (51,)
        assert result.species.shape == (51, 2)
        assert result.observables.shape == (51, 2)

        for i in range(result.n_times):
            t = result.time[i]
            A_exact = 100.0 * math.exp(-0.1 * t)
            B_exact = 100.0 - A_exact
            assert result.species[i, 0] == pytest.approx(A_exact, abs=1e-4)
            assert result.species[i, 1] == pytest.approx(B_exact, abs=1e-4)

    def test_param_update_resimulate(self, simple_decay_net: Path):
        """Change k1, resimulate, check new analytical solution."""
        model = Model.from_net(simple_decay_net)
        model.set_param("k1", 1.0)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 5), n_points=51)

        A_end = result.species[-1, 0]
        expected = 100.0 * math.exp(-5.0)
        assert A_end == pytest.approx(expected, abs=1e-3)

    def test_reversible_equilibrium(self, reversible_net: Path):
        """Reversible binding reaches equilibrium with conservation."""
        model = Model.from_net(reversible_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 1000), n_points=101)

        # At equilibrium, conservation laws hold
        A_eq = result.species[-1, 0]
        B_eq = result.species[-1, 1]
        C_eq = result.species[-1, 2]

        assert A_eq + C_eq == pytest.approx(100.0, abs=1e-3)
        assert B_eq + C_eq == pytest.approx(50.0, abs=1e-3)
        assert C_eq > 2.0  # some product formed

    def test_solver_stats(self, simple_decay_net: Path):
        """Solver diagnostics are populated."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        stats = result.solver_stats
        assert isinstance(stats, dict)
        assert stats["n_steps"] > 0
        assert stats["n_rhs_evals"] > 0

    def test_time_dependent_function(self, time_func_net: Path):
        """Time-dependent rate law: dB/dt = (t+1)*1, B(t) = t²/2 + t."""
        model = Model.from_net(time_func_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        B_10 = result.species[-1, 1]
        assert pytest.approx(60.0, abs=0.1) == B_10  # 10²/2 + 10

        B_5 = result.species[5, 1]
        assert pytest.approx(17.5, abs=0.1) == B_5  # 5²/2 + 5

    def test_tolerances(self, simple_decay_net: Path):
        """Custom tolerances don't crash."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11, rtol=1e-12, atol=1e-12)
        assert result.n_times == 11


# ═══════════════════════════════════════════════════════════════════════════════
# SSA simulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSsaSimulation:
    """Test SSA (Gillespie) simulation via Simulator(method='ssa')."""

    def test_simple_decay_decreases(self, simple_decay_net: Path):
        """SSA: A should decrease over time."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")
        result = sim.run(t_span=(0, 50), n_points=51, seed=42)

        assert result.time.shape == (51,)
        assert result.species[0, 0] == pytest.approx(100.0)
        assert result.species[-1, 0] < 10.0  # should be small

    def test_reproducibility(self, simple_decay_net: Path):
        """Same seed → identical SSA trajectories."""
        model1 = Model.from_net(simple_decay_net)
        model2 = Model.from_net(simple_decay_net)
        sim1 = Simulator(model1, method="ssa")
        sim2 = Simulator(model2, method="ssa")

        r1 = sim1.run(t_span=(0, 10), n_points=11, seed=12345)
        r2 = sim2.run(t_span=(0, 10), n_points=11, seed=12345)

        np.testing.assert_array_equal(r1.species, r2.species)

    def test_different_seeds_differ(self, simple_decay_net: Path):
        """Different seeds → different trajectories (probabilistic)."""
        model1 = Model.from_net(simple_decay_net)
        model2 = Model.from_net(simple_decay_net)
        sim1 = Simulator(model1, method="ssa")
        sim2 = Simulator(model2, method="ssa")

        r1 = sim1.run(t_span=(0, 50), n_points=51, seed=1)
        r2 = sim2.run(t_span=(0, 50), n_points=51, seed=999)

        # With 100 molecules decaying, different seeds should give different paths
        # (not guaranteed, but overwhelmingly likely)
        assert not np.array_equal(r1.species, r2.species)

    def test_fixed_species(self, fixed_species_net: Path):
        """$A stays constant in SSA."""
        model = Model.from_net(fixed_species_net)
        sim = Simulator(model, method="ssa")
        result = sim.run(t_span=(0, 50), n_points=51, seed=42)

        # $A should stay at 100 throughout
        np.testing.assert_allclose(result.species[:, 0], 100.0)
        # B should have grown
        assert result.species[-1, 1] > 0.0

    def test_fractional_initial_population_is_rounded(self, fractional_ssa_net: Path):
        """SSA starts from integer molecule counts when .net ICs are fractional.

        Load-bearing for the GH #118 disposition: the V2005_bistable_gene /
        Kholodenko2000 SSA adjudication assumes bngsim rounds the fractional
        ODE-warmed warm start to the same integer seed legacy uses. If this
        regresses, the SSA_KNOWN_DISPOSITION reasons in
        parity_checks/bng_parity/bng_stoch_run.py no longer hold — see
        dev/notes/issue118_adjudication.md.
        """
        model = Model.from_net(fractional_ssa_net)

        issues = model.validate_for_ssa()
        assert any(i.code == "non_integer_initial_population" for i in issues)

        sim = Simulator(model, method="ssa")
        result = sim.run(t_span=(0, 5), n_points=6, seed=1)

        species = np.asarray(result.species)[:, 0]
        assert species[0] == pytest.approx(6.0)
        np.testing.assert_allclose(species, np.rint(species), atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Model reset
# ═══════════════════════════════════════════════════════════════════════════════


class TestReset:
    """Test model reset after simulation."""

    def test_reset_after_ode(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        sim.run(t_span=(0, 10), n_points=11)

        model.reset()
        # After reset, re-run should give same result
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.species[0, 0] == pytest.approx(100.0)
        assert result.species[0, 1] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Result object
# ═══════════════════════════════════════════════════════════════════════════════


class TestResult:
    """Test Result object features."""

    def test_named_observable_access(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        obs = result.observables
        A_tot = obs["A_tot"]
        assert A_tot.shape == (11,)
        assert A_tot[0] == pytest.approx(100.0, abs=1e-10)

    def test_named_observable_missing(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        with pytest.raises(KeyError, match="not found"):
            result.observables["nonexistent"]

    def test_observables_as_array(self, simple_decay_net: Path):
        """np.asarray(result.observables) should work."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        arr = np.asarray(result.observables)
        assert arr.shape == (11, 2)

    def test_observable_shape(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        assert result.observables.shape == (11, 2)

    def test_species_names(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        assert result.species_names == ["A()", "B()"]

    def test_observable_names(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        assert result.observable_names == ["A_tot", "B_tot"]

    def test_custom_attrs(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        result.custom_attrs["experiment"] = "test_1"
        assert result.custom_attrs["experiment"] == "test_1"

    def test_repr(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        r = repr(result)
        assert "n_times=11" in r
        assert "n_species=2" in r
        assert "n_observables=2" in r


# ═══════════════════════════════════════════════════════════════════════════════
# File export
# ═══════════════════════════════════════════════════════════════════════════════


class TestFileExport:
    """Test .gdat and .cdat file export."""

    def test_gdat_export(self, simple_decay_net: Path, tmp_path: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        out = tmp_path / "test.gdat"
        result.to_gdat(out)
        assert out.exists()
        content = out.read_text()
        assert "time" in content
        assert "A_tot" in content

    def test_cdat_export(self, simple_decay_net: Path, tmp_path: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        out = tmp_path / "test.cdat"
        result.to_cdat(out)
        assert out.exists()
        content = out.read_text()
        assert "time" in content


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimulatorValidation:
    """Test Simulator input validation."""

    def test_invalid_method(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ValueError, match="Unknown method"):
            Simulator(model, method="euler")

    def test_invalid_t_span(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        with pytest.raises(ValueError, match="t_end"):
            sim.run(t_span=(100, 0), n_points=11)

    def test_invalid_n_points(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        with pytest.raises(ValueError, match="n_points"):
            sim.run(t_span=(0, 10), n_points=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Reserved names
# ═══════════════════════════════════════════════════════════════════════════════


class TestReservedNames:
    """Test the reserved_names() module function."""

    def test_reserved_names_structure(self):
        names = bngsim.reserved_names()
        assert "constants" in names
        assert "functions" in names
        assert isinstance(names["constants"], list)
        assert isinstance(names["functions"], list)

    def test_reserved_constants(self):
        names = bngsim.reserved_names()
        assert "_pi" in names["constants"]
        assert "_e" in names["constants"]
        assert "_kB" in names["constants"]
        assert len(names["constants"]) == 7

    def test_reserved_functions(self):
        names = bngsim.reserved_names()
        assert "time" in names["functions"]
        # Issue #24: `t` must NOT be reserved — BNG2.pl uses time() only,
        # leaving `t` free as a model identifier.
        assert "t" not in names["functions"]
        assert "sin" in names["functions"]
        assert len(names["functions"]) > 30


# ═══════════════════════════════════════════════════════════════════════════════
# Exception hierarchy
# ═══════════════════════════════════════════════════════════════════════════════


class TestExceptions:
    """Test that bngsim exceptions have correct inheritance."""

    def test_hierarchy(self):
        assert issubclass(ModelError, BngsimError)
        assert issubclass(SimulationError, BngsimError)
        assert issubclass(ParameterError, BngsimError)
        assert issubclass(BngsimError, RuntimeError)

    def test_catch_base(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        with pytest.raises(BngsimError):
            model.set_param("nonexistent", 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# DataFrame (optional pandas)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataFrame:
    """Test the optional .dataframe property."""

    def test_dataframe(self, simple_decay_net: Path):
        """If pandas is installed, .dataframe should work."""
        pytest.importorskip("pandas")

        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        df = result.dataframe
        assert "time" in df.columns
        assert "A_tot" in df.columns
        assert "B_tot" in df.columns
        assert len(df) == 11


# ═══════════════════════════════════════════════════════════════════════════════
# save_concentrations / set_concentration (Phase B.5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveConcentrations:
    """Test saveConcentrations() and setConcentration() actions."""

    def test_save_then_reset(self, simple_decay_net: Path):
        """save_concentrations() changes what reset() restores to."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        # Simulate half-way: A decays from 100
        result = sim.run(t_span=(0, 20), n_points=21)
        A_at_20 = result.species[-1, 0]
        assert A_at_20 < 100.0
        assert A_at_20 > 0.0

        # Save current concentrations as new initial state
        model.save_concentrations()

        # Reset should now restore to the saved state (not original 100)
        model.reset()

        # Re-simulate from the saved state
        result2 = sim.run(t_span=(0, 20), n_points=21)
        assert result2.species[0, 0] == pytest.approx(A_at_20, abs=1e-3)

    def test_set_concentration(self, simple_decay_net: Path):
        """set_concentration() changes a single species."""
        model = Model.from_net(simple_decay_net)

        # Initial A = 100, B = 0
        assert model.get_concentration("A()") == pytest.approx(100.0)
        assert model.get_concentration("B()") == pytest.approx(0.0)

        # Set A to 50
        model.set_concentration("A()", 50.0)
        assert model.get_concentration("A()") == pytest.approx(50.0)

        # B unchanged
        assert model.get_concentration("B()") == pytest.approx(0.0)

    def test_set_concentration_affects_sim(self, simple_decay_net: Path):
        """set_concentration() is picked up by subsequent simulation."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        # Set A to 50 (instead of 100)
        model.set_concentration("A()", 50.0)
        result = sim.run(t_span=(0, 50), n_points=51)

        # First time point should reflect the change
        assert result.species[0, 0] == pytest.approx(50.0, abs=0.1)

        # Analytical: A(t) = 50*exp(-0.1*t)
        A_50 = 50.0 * math.exp(-0.1 * 50)
        assert result.species[-1, 0] == pytest.approx(A_50, abs=0.1)

    def test_get_concentration_unknown_raises(self, simple_decay_net: Path):
        """get_concentration() with unknown species raises ModelError."""
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ModelError):
            model.get_concentration("nonexistent")

    def test_set_concentration_unknown_raises(self, simple_decay_net: Path):
        """set_concentration() with unknown species raises ModelError."""
        model = Model.from_net(simple_decay_net)
        with pytest.raises(ModelError):
            model.set_concentration("nonexistent", 42.0)

    def test_save_concentrations_and_set(self, simple_decay_net: Path):
        """Combined: set_concentration + save_concentrations + reset."""
        model = Model.from_net(simple_decay_net)

        # Change A, save, reset
        model.set_concentration("A()", 75.0)
        model.save_concentrations()
        model.set_concentration("A()", 0.0)  # temporary change
        model.reset()

        # After reset, should be back to 75 (the saved state)
        assert model.get_concentration("A()") == pytest.approx(75.0)


# ═══════════════════════════════════════════════════════════════════════════════
# run_batch (Phase B)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunBatch:
    """Test Simulator.run_batch() for batch parameter sweeps."""

    def test_batch_ode_basic(self, simple_decay_net: Path):
        """Basic batch with 3 parameter sets."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        param_sets = [
            {"k1": 0.01},
            {"k1": 0.1},
            {"k1": 1.0},
        ]
        results = sim.run_batch(t_span=(0, 10), n_points=11, params=param_sets)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, Result)
            assert r.time.shape == (11,)
            assert r.species.shape == (11, 2)

    def test_batch_results_differ(self, simple_decay_net: Path):
        """Different parameter values → different results."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.01}, {"k1": 1.0}],
        )

        # Slow decay vs fast decay: final A should differ
        A_slow = results[0].species[-1, 0]
        A_fast = results[1].species[-1, 0]
        assert A_slow > A_fast  # slow: more A remaining

    def test_batch_analytical_check(self, simple_decay_net: Path):
        """Verify batch results against analytical solution."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        k_values = [0.05, 0.1, 0.5]
        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": k} for k in k_values],
        )

        for k, result in zip(k_values, results, strict=False):
            A_end = result.species[-1, 0]
            expected = 100.0 * math.exp(-k * 10.0)
            assert A_end == pytest.approx(expected, abs=0.1)

    def test_batch_preserves_original_model(self, simple_decay_net: Path):
        """Batch does not modify the original model."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        original_k1 = model.get_param("k1")
        sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 99.0}],
        )
        assert model.get_param("k1") == pytest.approx(original_k1)

    def test_batch_ssa(self, simple_decay_net: Path):
        """Batch works with SSA method."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ssa")

        results = sim.run_batch(
            t_span=(0, 10),
            n_points=11,
            params=[{"k1": 0.1}, {"k1": 0.5}],
            seed=42,
        )

        assert len(results) == 2
        # Different seeds per batch element → different trajectories
        assert not np.array_equal(results[0].species, results[1].species)

    def test_batch_empty_raises(self, simple_decay_net: Path):
        """Empty params raises ValueError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        with pytest.raises(ValueError, match="non-empty"):
            sim.run_batch(t_span=(0, 10), n_points=11, params=[])

    def test_batch_none_raises(self, simple_decay_net: Path):
        """None params raises ValueError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        with pytest.raises(ValueError, match="non-empty"):
            sim.run_batch(t_span=(0, 10), n_points=11, params=None)

    def test_batch_invalid_t_span(self, simple_decay_net: Path):
        """Invalid t_span raises ValueError."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")

        with pytest.raises(ValueError, match="t_end"):
            sim.run_batch(
                t_span=(100, 0),
                n_points=11,
                params=[{"k1": 0.1}],
            )
