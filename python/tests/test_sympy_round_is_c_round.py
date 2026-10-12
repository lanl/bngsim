"""``round`` means C's round in the sympy paths too (follow-up to issue #684).

The engine's ``round`` is C's: a half away from zero, ``round(2.5)`` is 3. The
three places that read ExprTk text into sympy — the rate-law Jacobian
(``_exprtk_to_sympy``), the derived-parameter chain rule
(``_prepare_derived_expr``) and the clock-threshold solvers
(``_parse_clock_expr``) — left ``round`` unbound, and ``parse_expr`` resolves an
unbound call through Python's builtins. Python's ``round`` rounds a half to
EVEN and folded a literal on the spot, so ``round(2.5)`` became 2. Nothing
raised:

* a derived parameter ``q = round(2.5)*k`` got ``dq/dk = 2``, so a forward
  sensitivity through it came back 8 where the trajectory says 12;
* a clock condition ``floor(time()) >= round(2.5)`` got its stop at t = 2, not
  at the t = 3 where it switches;
* the analytic Jacobian of ``k*round(2.5)*Atot`` was built with 2, failed its
  finite-difference self-check, and the model lost it.

Two more ways to get it wrong are pinned too: folding the 60-bit Float that
``parse_expr`` makes of a 17-digit literal rather than the double the engine
reads, and evaluating a symbolic ``round`` through ``lambdify``'s numpy
namespace, where it is ``numpy.round`` — a half to even again.

Each is pinned against C's round, read exactly from a ``Decimal``.
"""

from __future__ import annotations

import contextlib
import io
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import bngsim
import numpy as np
import pytest
import sympy as sp
from bngsim import _switch_sensitivity as sw
from bngsim._jacobian import _exprtk_to_sympy, builtin_shadow_bindings

ROUND = builtin_shadow_bindings(sp)["round"]
CLOCK = frozenset({"time", "time()"})

#: Halves, which Python rounds to even and C away from zero; then the edges
#: where floor(x + 0.5) itself rounds, which C's round does not.
CASES = [2.5, -2.5, 0.5, -0.5, 1.5, 3.5, 2.4, -2.7, 0.0]
EDGES = [0.49999999999999994, -0.49999999999999994, 4503599627370497.0]


def _c_round(x: float) -> float:
    """C's round, exactly: a Decimal holds the double without error, and its
    ROUND_HALF_UP is a half away from zero."""
    return float(Decimal(x).quantize(Decimal(1), rounding=ROUND_HALF_UP))


@pytest.mark.parametrize("x", CASES + EDGES)
def test_a_float_folds_to_c_round(x):
    assert float(ROUND(sp.Float(x))) == _c_round(x)


@pytest.mark.parametrize("x", CASES)
def test_a_rational_folds_to_c_round(x):
    assert ROUND(sp.Rational(x)) == sp.Integer(int(_c_round(x)))


def test_a_symbol_stays_a_call_with_no_derivative():
    x = sp.Symbol("x")
    call = ROUND(x)
    assert call.func is ROUND
    assert float(call.subs(x, 2.5)) == 3.0
    assert isinstance(call.diff(x), sp.Derivative)  # a model through it declines
    assert sp.ccode(2 * call) == "2*round(x)"  # C's round, by name
    assert str(call) == "round(x)"  # and ExprTk's, which is C's since #684


def test_non_finite_arguments_stay_unevaluated():
    assert ROUND(sp.oo).func is ROUND
    assert ROUND(sp.nan).func is ROUND


def test_built_once():
    """A second class of the same name would compare unequal to the first."""
    assert builtin_shadow_bindings(sp)["round"] is ROUND


def test_the_rate_law_parser_folds_round_as_c_does():
    k, a = sp.symbols("k A")
    assert sp.simplify(_exprtk_to_sympy("k*round(2.5)*A") - 3 * k * a) == 0
    assert sp.simplify(_exprtk_to_sympy("k*round(-2.5)*A") + 3 * k * a) == 0


@pytest.mark.parametrize(
    "text, want",
    [
        # parse_expr holds these at 60 bits, a hair from the half the engine sees
        ("2.4999999999999999", 3.0),
        ("4503599627370496.5", 4503599627370496.0),
        # and these at 56, where rounding that once more to a double lands on
        # the half: float() of the parsed Float gave 8193 and 9218
        ("8192.499999999999", 8192.0),
        ("9217.499999999999", 9217.0),
    ],
)
def test_a_long_literal_folds_as_the_double_the_engine_reads(text, want):
    assert _c_round(float(text)) == want  # C's round of the double strtod reads
    assert float(_exprtk_to_sympy(f"round({text})")) == want


def test_a_wide_float_from_arithmetic_folds_as_itself():
    """h times a 16-digit literal is a 56-bit Float holding the double
    2.4999999999999996, whose 16 digits print as 2.500000000000000. Read back
    from them it rounded to 3; the engine computes 2."""
    from bngsim._codegen import _derived_expr_value_numeric

    expr = "round(h*1.000000000000000)"
    assert _derived_expr_value_numeric(expr, {"h"}, (), {"h": 0}, [2.4999999999999996]) == 2.0
    assert _derived_expr_value_numeric(expr, {"h"}, (), {"h": 0}, [2.5]) == 3.0


def test_a_literal_past_the_doubles_is_not_folded():
    """C's round of it is inf; sympy's Rational(inf) would have made it 0."""
    assert _exprtk_to_sympy("round(1.8e308)").func is ROUND


@pytest.mark.parametrize("x", CASES + EDGES)
def test_numpy_evaluates_a_symbolic_round_as_c_does(x):
    """lambdify(modules="numpy") is how the derived-parameter partials are
    evaluated (_codegen._partial_value). Unspelled, round is numpy.round."""
    h = sp.Symbol("h")
    fn = sp.lambdify(h, ROUND(h), modules="numpy")
    assert float(fn(np.float64(x))) == _c_round(x)


def test_a_parameter_named_round_is_still_a_symbol():
    expr = _exprtk_to_sympy("round + k")
    assert {str(s) for s in expr.free_symbols} == {"round", "k"}


def test_the_clock_solver_places_the_threshold_where_c_does():
    lhs, threshold = sw._clock_affine_threshold("time() >= round(2.5)", CLOCK)
    assert float(sp.sympify(threshold)) == 3.0
    expr, _ = sw._parse_clock_expr("time - round(-2.5)")
    assert float(expr.subs("time", 0)) == 3.0


def _scope():
    return sw.SwitchConditionScope(
        core=None,
        clocks={},
        clock_symbols=CLOCK,
        param_names=(),
        param_pats={},
        primary_names=frozenset(),
        param_idx={},
        values=(),
        derived_exprs={},
        run_constants=frozenset(),
        function_names=frozenset(),
    )


def test_a_step_edge_stop_lands_where_the_condition_switches():
    """floor(time()) reaches round(2.5) = 3 at t = 3. It was placed at 2."""
    got = sw._resolve_step_edge_stop_times("floor(time()) >= round(2.5)", _scope(), 0.0, 10.0)
    assert got == [3.0]


def test_a_step_edge_reader_declines_round_of_the_clock():
    """round(time()) jumps at every half, and the reader only knows floor's and
    trunc's jumps. Unevaluated, it is not affine in time, so the reader gives up
    rather than set the stops from the floor alone."""
    atom = "floor(time()) + round(time()) >= 5"
    assert sw._resolve_step_edge_stop_times(atom, _scope(), 0.0, 10.0) is None


def _net(tmp_path: Path, params: list[str], fn: str) -> str:
    """0 -> B at the rate `f`, so B(t) is the integral of f."""
    lines = ["begin parameters"]
    lines += [f"  {i} {p}" for i, p in enumerate(params, 1)]
    lines += ["end parameters", "begin species", "  1 B() 0", "end species"]
    lines += ["begin functions", f"  1 f() {fn}", "end functions"]
    lines += ["begin reactions", "  1 0 1 f", "end reactions"]
    lines += ["begin groups", "  1 Btot 1", "end groups"]
    path = tmp_path / "m.net"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _sensitivity(net: str, param: str) -> float:
    """dB(4)/d(param), from the forward sensitivities."""
    model = bngsim.Model.from_net(net)
    with contextlib.redirect_stderr(io.StringIO()):
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param])
        res = sim.run(t_span=(0, 4), n_points=5)
    return float(np.asarray(res.sensitivities)[-1, 0, 0])


@pytest.mark.parametrize("c", [2.5, -2.5])
def test_a_derived_parameter_through_round_has_the_right_sensitivity(tmp_path, c):
    """B(4) = 4*q with q = round(c)*k, so dB/dk = 4*round(c): 12 and -12. The
    half-to-even fold gave 8 and -8."""
    net = _net(tmp_path, ["k 1", f"q round({c!r})*k"], "q")
    assert _sensitivity(net, "k") == pytest.approx(4 * _c_round(c), rel=1e-6)


def test_a_constant_round_threshold_keeps_its_sensitivity(tmp_path):
    """The switch is at t = 3, so B(4) = 3k and dB/dk = 3. This was right before
    only because the threshold, though misplaced at 2, does not move with k; a
    round that sympy could not evaluate at all declined it instead."""
    net = _net(tmp_path, ["k 1"], "if(time() >= round(2.5), 0, k)")
    assert _sensitivity(net, "k") == pytest.approx(3.0, rel=1e-6)


def test_an_initial_condition_seeded_through_round_has_the_right_sensitivity(tmp_path):
    """B(0) = q = k*round(h) at h = 2.5, so dB(0)/dk = round(2.5) = 3. The seed
    is computed numerically; through numpy.round it came back 2."""
    path = tmp_path / "ic.net"
    path.write_text(
        "begin parameters\n  1 k 1\n  2 h 2.5\n  3 q k*round(h)\n  4 z 0\nend parameters\n"
        "begin species\n  1 B() q\nend species\n"
        "begin reactions\n  1 1 0 z\nend reactions\n"
        "begin groups\n  1 Btot 1\nend groups\n"
    )
    model = bngsim.Model.from_net(str(path))
    with contextlib.redirect_stderr(io.StringIO()):
        res = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
            t_span=(0, 1), n_points=2
        )
    assert float(res.observables["Btot"][0]) == 3.0
    assert float(np.asarray(res.sensitivities)[0, 0, 0]) == pytest.approx(3.0, rel=1e-9)
