# bngsim/python/tests/test_model_validation.py
# Session 65: P9 — Model validation on construction
#
# Tests that ModelBuilder.build() rejects malformed models with clear errors.

import pytest
from bngsim._bngsim_core import ModelBuilder


def _builder():
    """Create a fresh ModelBuilder."""
    return ModelBuilder()


class TestDuplicateNames:
    """Duplicate species/parameter names are rejected."""

    def test_duplicate_species(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("A", 20.0)  # duplicate
        b.add_reaction([0], [1], "elementary", "k")
        with pytest.raises(RuntimeError, match="duplicate species"):
            b.build()

    def test_duplicate_parameter(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_parameter("k", 0.2)  # duplicate
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], "elementary", "k")
        with pytest.raises(RuntimeError, match="duplicate parameter"):
            b.build()


class TestSpeciesIndexRange:
    """Reaction species indices must be in range."""

    def test_bad_reactant_index(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_reaction([5], [0], "elementary", "k")
        with pytest.raises(RuntimeError, match="reactant species index"):
            b.build()

    def test_bad_product_index(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_reaction([0], [3], "elementary", "k")
        with pytest.raises(RuntimeError, match="product species index"):
            b.build()


class TestObservableIndexRange:
    """Observable group species indices must be in range."""

    def test_bad_observable_species(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_observable("bad", [(5, 1.0)])
        b.add_reaction([0], [1], "elementary", "k")
        with pytest.raises(RuntimeError, match="observable.*out of range"):
            b.build()


class TestRateLawResolution:
    """Rate law parameter/function refs must resolve."""

    def test_unknown_elementary_param(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], "elementary", "k_missing")
        with pytest.raises(RuntimeError, match="unknown parameter"):
            b.build()

    def test_unknown_functional_func(self):
        b = _builder()
        b.add_parameter("k", 0.1)
        b.add_species("A", 10.0)
        b.add_species("B", 0.0)
        b.add_reaction([0], [1], "functional", "ghost")
        with pytest.raises(RuntimeError, match="unknown function"):
            b.build()


class TestValidModelOK:
    """Well-formed models build and simulate."""

    def test_simple_model(self):
        from bngsim._bngsim_core import (
            CvodeSimulator,
            TimeSpec,
        )

        b = _builder()
        b.add_parameter("k", 0.1)
        a = b.add_species("A", 100.0)
        bb = b.add_species("B", 0.0)
        b.add_observable("A_tot", [(a, 1.0)])
        b.add_reaction([a], [bb], "elementary", "k")
        model = b.build()
        assert model.n_species == 2
        assert model.n_reactions == 1

        sim = CvodeSimulator(model)
        ts = TimeSpec()
        ts.t_start = 0.0
        ts.t_end = 10.0
        ts.n_points = 11
        result = sim.run(ts)
        assert result.n_times == 11
