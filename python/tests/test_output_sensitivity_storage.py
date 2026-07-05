"""GH #196 — storage + Python API for observable / expression output sensitivities.

This stage adds the *storage and plumbing* for ``d observable/dθ`` and
``d expression/dθ`` (plus the IC variants) ahead of any computation. The blocks
are therefore empty on every current run; the tests assert the empty-state
contract, the round-trip / xarray plumbing using manually-injected blocks, and
that nothing about the existing **species** sensitivities changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Result, Simulator

# Shapes for the manually-constructed (raw-array) Results used below.
_NT, _NS, _NO, _NE, _NPAR, _NIC = 5, 2, 2, 3, 2, 1


def _make_populated_result() -> Result:
    """A raw-array Result with every sensitivity block populated with a
    distinct constant, so a round-trip / stack can be checked block-by-block."""
    nt, ns, no, ne, npar, nic = _NT, _NS, _NO, _NE, _NPAR, _NIC
    return Result(
        _time=np.linspace(0.0, 1.0, nt),
        _species=np.zeros((nt, ns)),
        _observables=np.zeros((nt, no)),
        _expressions=np.zeros((nt, ne)),
        _species_names=["A", "B"],
        _observable_names=["Atot", "Btot"],
        _expression_names=["f", "g", "h"],
        _sensitivities=np.arange(nt * ns * npar, dtype=float).reshape(nt, ns, npar),
        _sensitivity_params=["k1", "k2"],
        _sensitivities_ic=np.full((nt, ns, nic), 9.0),
        _sensitivity_ic_species=["A"],
        _observable_sensitivities=np.full((nt, no, npar), 1.0),
        _expression_sensitivities=np.full((nt, ne, npar), 2.0),
        _observable_sensitivities_ic=np.full((nt, no, nic), 3.0),
        _expression_sensitivities_ic=np.full((nt, ne, nic), 4.0),
    )


# ── Empty-state contract on a freshly-run result ──────────────────────────────


class TestEmptyState:
    """No computation path populates the output blocks yet, so every block on a
    real run is the empty ``(0, 0, 0)`` sentinel with ``has_* == False``."""

    def test_new_blocks_empty_on_plain_run(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        result = Simulator(model, method="ode").run(t_span=(0, 10), n_points=11)

        for has, arr in [
            (result.has_sensitivities_observables, result.sensitivities_observables),
            (result.has_sensitivities_expressions, result.sensitivities_expressions),
            (result.has_sensitivities_observables_ic, result.sensitivities_observables_ic),
            (result.has_sensitivities_expressions_ic, result.sensitivities_expressions_ic),
        ]:
            assert has is False
            assert arr.shape == (0, 0, 0)

    def test_observable_param_block_populated_with_species_sensitivities(
        self, simple_decay_net: Path
    ):
        """GH #197: observable parameter sensitivities are now computed at
        simulation time (runtime chain rule) whenever species parameter
        sensitivities are. simple_decay.net has no global functions, so the
        expression blocks stay empty (GH #198 only populates them for models
        with functions); no IC axis was requested, so those blocks stay empty
        too."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        assert result.has_sensitivities  # species block populated
        # Observable parameter sensitivities now populate (GH #197).
        assert result.has_sensitivities_observables
        assert result.sensitivities_observables.shape == (11, 2, 1)
        # Expression output sensitivities remain unimplemented (GH #198), and
        # no IC axis was requested, so those three blocks stay empty.
        assert not result.has_sensitivities_expressions
        assert not result.has_sensitivities_observables_ic
        assert not result.has_sensitivities_expressions_ic


# ── species alias ─────────────────────────────────────────────────────────────


class TestSpeciesAlias:
    def test_alias_equals_sensitivities_empty(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        result = Simulator(model, method="ode").run(t_span=(0, 10), n_points=11)
        assert np.array_equal(result.sensitivities_species, result.sensitivities)

    def test_alias_equals_sensitivities_populated(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)
        assert result.sensitivities.size > 0
        # Same object/array — the alias is species-only, never the new blocks.
        assert np.array_equal(result.sensitivities_species, result.sensitivities)


# ── Backward compat: species sensitivity API unchanged ────────────────────────


class TestBackwardCompat:
    def test_species_sensitivities_unchanged(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)

        # The existing species API behaves exactly as before.
        assert result.has_sensitivities
        assert result.sensitivities.shape == (11, 2, 1)
        assert result.sensitivity_params == ["k1"]

    def test_sensitivity_data_pybind_is_species_only(self, simple_decay_net: Path):
        """The C++ ``sensitivity_data`` property still returns the species block
        (shape over n_species), independent of the new output blocks."""
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode", sensitivity_params=["k1"])
        result = sim.run(t_span=(0, 10), n_points=11)
        core_block = result._core.sensitivity_data
        assert core_block.shape == (11, 2, 1)


# ── HDF5 round-trip ───────────────────────────────────────────────────────────


class TestHdf5RoundTrip:
    @pytest.fixture(autouse=True)
    def _check_h5py(self):
        pytest.importorskip("h5py")

    def test_empty_blocks_round_trip(self, simple_decay_net: Path, tmp_path: Path):
        """A result with no sensitivities saves no sensitivity datasets and
        loads back to the empty blocks."""
        model = Model.from_net(simple_decay_net)
        result = Simulator(model, method="ode").run(t_span=(0, 10), n_points=11)

        path = tmp_path / "empty_sens.h5"
        result.save(path)
        loaded = Result.load(path)

        assert loaded.sensitivities_observables.shape == (0, 0, 0)
        assert loaded.sensitivities_expressions.shape == (0, 0, 0)
        assert loaded.sensitivities_observables_ic.shape == (0, 0, 0)
        assert loaded.sensitivities_expressions_ic.shape == (0, 0, 0)
        # Species blocks too (also empty on a plain run).
        assert loaded.sensitivities.shape == (0, 0, 0)
        assert not loaded.has_sensitivities_observables

    def test_populated_blocks_round_trip(self, tmp_path: Path):
        result = _make_populated_result()
        path = tmp_path / "full_sens.h5"
        result.save(path)
        loaded = Result.load(path)

        np.testing.assert_array_equal(loaded.sensitivities, result.sensitivities)
        np.testing.assert_array_equal(loaded.sensitivities_ic, result.sensitivities_ic)
        np.testing.assert_array_equal(
            loaded.sensitivities_observables, result.sensitivities_observables
        )
        np.testing.assert_array_equal(
            loaded.sensitivities_expressions, result.sensitivities_expressions
        )
        np.testing.assert_array_equal(
            loaded.sensitivities_observables_ic, result.sensitivities_observables_ic
        )
        np.testing.assert_array_equal(
            loaded.sensitivities_expressions_ic, result.sensitivities_expressions_ic
        )
        assert loaded.sensitivity_params == ["k1", "k2"]
        assert loaded.sensitivity_ic_species == ["A"]


# ── xarray dims / coords ──────────────────────────────────────────────────────


class TestXarray:
    @pytest.fixture(autouse=True)
    def _check_xarray(self):
        pytest.importorskip("xarray")

    def test_per_field_dims_and_coords(self):
        result = _make_populated_result()

        obs = result.xr.sensitivities_observables
        assert obs.dims == ("time", "observable", "parameter")
        assert list(obs.coords["observable"].values) == ["Atot", "Btot"]
        assert list(obs.coords["parameter"].values) == ["k1", "k2"]

        expr = result.xr.sensitivities_expressions
        assert expr.dims == ("time", "expression", "parameter")
        assert list(expr.coords["expression"].values) == ["f", "g", "h"]

        obs_ic = result.xr.sensitivities_observables_ic
        assert obs_ic.dims == ("time", "observable", "ic_state")
        assert list(obs_ic.coords["ic_state"].values) == ["A"]

        expr_ic = result.xr.sensitivities_expressions_ic
        assert expr_ic.dims == ("time", "expression", "ic_state")

    def test_to_xarray_includes_new_blocks(self):
        ds = _make_populated_result().to_xarray()
        for var in (
            "sensitivities_observables",
            "sensitivities_expressions",
            "sensitivities_observables_ic",
            "sensitivities_expressions_ic",
        ):
            assert var in ds.data_vars
        assert ds["sensitivities_observables"].dims == ("time", "observable", "parameter")

    def test_empty_blocks_absent_from_xarray(self, simple_decay_net: Path):
        result = Simulator(Model.from_net(simple_decay_net), method="ode").run(
            t_span=(0, 10), n_points=11
        )
        ds = result.to_xarray()
        assert "sensitivities_observables" not in ds.data_vars
        with pytest.raises(AttributeError):
            _ = result.xr.sensitivities_observables


# ── Batch squeeze wiring ──────────────────────────────────────────────────────


class TestSqueeze:
    def test_squeeze_stacks_new_blocks(self):
        results = [_make_populated_result() for _ in range(3)]
        batch = Result.squeeze(results)

        assert batch.sensitivities_observables.shape == (3, _NT, _NO, _NPAR)
        assert batch.sensitivities_expressions.shape == (3, _NT, _NE, _NPAR)
        assert batch.sensitivities_observables_ic.shape == (3, _NT, _NO, _NIC)
        assert batch.sensitivities_expressions_ic.shape == (3, _NT, _NE, _NIC)
        # Names carried from the first result.
        assert batch.sensitivity_params == ["k1", "k2"]
        assert batch.sensitivity_ic_species == ["A"]

    def test_squeeze_empty_blocks_stay_empty(self, simple_decay_net: Path):
        model = Model.from_net(simple_decay_net)
        sim = Simulator(model, method="ode")
        results = [sim.run(t_span=(0, 10), n_points=11) for _ in range(2)]
        batch = Result.squeeze(results)
        assert batch.sensitivities_observables.shape == (0, 0, 0)
        assert batch.sensitivities.shape == (0, 0, 0)
