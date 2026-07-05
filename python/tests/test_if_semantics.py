"""Tests for if(cond, a, b) semantics — Decision D1 (Session 37).

DISCOVERY: ExprTk's built-in `if` keyword uses nonzero truthiness
and takes precedence over any custom IfFunction registered via
add_function("if", ...). The custom IfFunction with cond > 0.5
in expression.cpp and nfsim_funcparser.h is DEAD CODE.

Actual BNGsim semantics: cond != 0 -> true_val, cond == 0 -> false_val
This matches BioNetGen Perl truthiness (if($_[0])).

See: bngsim/dev/adr/ADR-002-function-semantics.md for full analysis.
"""

import math

import pytest


class TestIfSemanticsODE:
    """if() semantics in ODE (ExprTk) expressions.

    ExprTk built-in if() uses nonzero truthiness:
      cond != 0 -> true_val
      cond == 0 -> false_val
    """

    @pytest.fixture
    def make_if_decay_model(self, tmp_path):
        """Factory: A -> 0 with rate = if(cond_val, 10, 20).

        After dt, A ~ A0 * exp(-rate * dt).
        We infer rate from A(dt)/A(0).
        """

        def _make(cond_val: float):
            net = tmp_path / f"if_{cond_val}.net"
            net.write_text(
                f"begin parameters\n"
                f"  1 cond_val {cond_val}\n"
                f"end parameters\n"
                f"begin species\n"
                f"  1 A() 100\n"
                f"end species\n"
                f"begin functions\n"
                f"  1 branch() if(cond_val, 10, 20)\n"
                f"end functions\n"
                f"begin reactions\n"
                f"  1 1 0 branch\n"
                f"end reactions\n"
                f"begin groups\n"
                f"  1 A_obs 1\n"
                f"end groups\n"
            )
            return str(net)

        return _make

    def _infer_rate(self, net_path, dt=0.01):
        """Run ODE and infer the effective rate constant."""
        from bngsim import Model, Simulator

        model = Model.from_net(net_path)
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, dt), n_points=2)
        a0 = result.species[0, 0]
        a1 = result.species[1, 0]
        if a1 <= 0 or a0 <= 0:
            return float("inf")
        return -math.log(a1 / a0) / dt

    def test_cond_neg1_true(self, make_if_decay_model):
        """if(-1, 10, 20) -> 10 (nonzero -> true)."""
        rate = self._infer_rate(make_if_decay_model(-1.0))
        assert abs(rate - 10.0) < 0.1

    def test_cond_zero_false(self, make_if_decay_model):
        """if(0, 10, 20) -> 20 (zero -> false)."""
        rate = self._infer_rate(make_if_decay_model(0.0))
        assert abs(rate - 20.0) < 0.1

    def test_cond_0p001_true(self, make_if_decay_model):
        """if(0.001, 10, 20) -> 10 (nonzero -> true)."""
        rate = self._infer_rate(make_if_decay_model(0.001))
        assert abs(rate - 10.0) < 0.1

    def test_cond_0p2_true(self, make_if_decay_model):
        """if(0.2, 10, 20) -> 10 (nonzero -> true)."""
        rate = self._infer_rate(make_if_decay_model(0.2))
        assert abs(rate - 10.0) < 0.1

    def test_cond_0p5_true(self, make_if_decay_model):
        """if(0.5, 10, 20) -> 10 (nonzero -> true)."""
        rate = self._infer_rate(make_if_decay_model(0.5))
        assert abs(rate - 10.0) < 0.1

    def test_cond_0p51_true(self, make_if_decay_model):
        """if(0.51, 10, 20) -> 10 (nonzero -> true)."""
        rate = self._infer_rate(make_if_decay_model(0.51))
        assert abs(rate - 10.0) < 0.1

    def test_cond_one_true(self, make_if_decay_model):
        """if(1, 10, 20) -> 10 (nonzero -> true)."""
        rate = self._infer_rate(make_if_decay_model(1.0))
        assert abs(rate - 10.0) < 0.1

    def test_cond_large_true(self, make_if_decay_model):
        """if(100, 10, 20) -> 10 (nonzero -> true)."""
        rate = self._infer_rate(make_if_decay_model(100.0))
        assert abs(rate - 10.0) < 0.1

    def test_relational_expr(self, tmp_path):
        """if(A_obs > 50, 10, 20): relational -> 0/1."""
        from bngsim import Model, Simulator

        net = tmp_path / "if_relational.net"
        net.write_text(
            "begin parameters\n"
            "end parameters\n"
            "begin species\n"
            "  1 A() 100\n"
            "end species\n"
            "begin functions\n"
            "  1 branch() if(A_obs > 50, 10, 20)\n"
            "end functions\n"
            "begin reactions\n"
            "  1 1 0 branch\n"
            "end reactions\n"
            "begin groups\n"
            "  1 A_obs 1\n"
            "end groups\n"
        )
        model = Model.from_net(str(net))
        sim = Simulator(model, method="ode")
        result = sim.run(t_span=(0, 0.01), n_points=2)
        a0 = result.species[0, 0]
        a1 = result.species[1, 0]
        rate = -math.log(a1 / a0) / 0.01
        assert abs(rate - 10.0) < 0.5
