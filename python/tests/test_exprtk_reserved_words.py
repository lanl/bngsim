"""Regression tests for issue #18: BNGL parameters that collide with ExprTk
reserved words (`const`, `true`, `false`, ...) used to crash Model.from_net
with `ExprTk: failed to register variable '<name>'`. bngsim now mangles
these names to `r_<name>` internally and rewrites references in compiled
expressions, so authors can keep BNGL names that round-trip through BNG2.pl.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from bngsim import Model, Simulator


class TestReservedWordParameters:
    """Parameters whose BNG names collide with ExprTk reserved tokens."""

    def test_loads_without_crash(self, exprtk_reserved_words_net: Path):
        """The .net loads — this is the primary regression."""
        model = Model.from_net(exprtk_reserved_words_net)
        assert model.n_species == 1
        assert model.n_reactions == 1

    def test_param_values_preserved(self, exprtk_reserved_words_net: Path):
        """The original BNG names round-trip through the public Python API."""
        model = Model.from_net(exprtk_reserved_words_net)
        assert model.get_param("const") == pytest.approx(2.0)
        assert model.get_param("true") == pytest.approx(0.0)
        assert model.get_param("false") == pytest.approx(0.0)
        assert model.get_param("k") == pytest.approx(0.5)
        assert "const" in model.param_names
        assert "true" in model.param_names
        assert "false" in model.param_names

    def test_simulation_uses_reserved_param_value(self, exprtk_reserved_words_net: Path):
        """The reserved-name parameters drive the rate law, not ExprTk literals.

        Effective rate = k*const + true - false = 0.5*2 + 0 - 0 = 1.0
        So A(t) = 100 * exp(-t). At t=10, A ≈ 100 * exp(-10) ≈ 4.54e-3.

        If `const` were silently dropped or `true`/`false` evaluated as the
        ExprTk literals 1/0, the rate would change and this check would fail.
        """
        model = Model.from_net(exprtk_reserved_words_net)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
        A_end = result.species[-1, 0]
        expected = 100.0 * math.exp(-1.0 * 10.0)
        assert A_end == pytest.approx(expected, rel=1e-4)

    def test_set_param_on_reserved_name(self, exprtk_reserved_words_net: Path):
        """set_param honors the user-facing BNG name, not the mangled key."""
        model = Model.from_net(exprtk_reserved_words_net)
        model.set_param("const", 10.0)
        assert model.get_param("const") == pytest.approx(10.0)

        # Now effective rate = 0.5*10 + 0 - 0 = 5.0
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0.0, 1.0), n_points=11)
        A_end = result.species[-1, 0]
        expected = 100.0 * math.exp(-5.0 * 1.0)
        assert A_end == pytest.approx(expected, rel=1e-4)

    def test_clone_preserves_mangling(self, exprtk_reserved_words_net: Path):
        """Cloning re-builds the symbol table; reserved-name params still work."""
        model = Model.from_net(exprtk_reserved_words_net)
        clone = model.clone()
        assert clone.get_param("const") == pytest.approx(2.0)

        # Mutating the clone does not affect the original.
        clone.set_param("const", 4.0)
        assert clone.get_param("const") == pytest.approx(4.0)
        assert model.get_param("const") == pytest.approx(2.0)

        # Clone simulates correctly with its independent value.
        sim = Simulator(clone, method="ode")
        result = sim.run(t_span=(0.0, 1.0), n_points=11)
        A_end = result.species[-1, 0]
        # rate = 0.5*4 + 0 - 0 = 2.0 → A(1) = 100 * exp(-2)
        expected = 100.0 * math.exp(-2.0 * 1.0)
        assert A_end == pytest.approx(expected, rel=1e-4)

    def test_builtin_function_names_still_work(
        self, exprtk_reserved_words_net: Path, tmp_path: Path
    ):
        """Mangling is conditional: built-in tokens like `sin`, `if`, `time`
        must NOT be rewritten when they appear in expressions but are not
        registered as user variables.
        """
        # Borrow the reserved-words .net but inject a function expression
        # that uses `sin` and `if` as ExprTk built-ins.
        net = (exprtk_reserved_words_net.read_text()).replace(
            "((k*const)+true)-false",
            "if(sin(0)==0,1,2)*((k*const)+true)-false",
        )
        out = tmp_path / "exprtk_reserved_with_builtins.net"
        out.write_text(net)

        model = Model.from_net(out)
        # if(sin(0)==0, 1, 2) → 1, so the effective rate is unchanged.
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
        A_end = result.species[-1, 0]
        expected = 100.0 * math.exp(-1.0 * 10.0)
        assert A_end == pytest.approx(expected, rel=1e-4)
