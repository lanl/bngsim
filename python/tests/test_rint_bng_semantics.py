"""``rint(x)`` is BNG's ``floor(x + 0.5)`` in every backend (issue #771).

BioNetGen defines ``rint`` as ``floor(x + 0.5)``: that is what BNG2.pl's
``run_network`` evaluates (muParser's ``Rint``). bngsim implemented it as C's
``std::round``, which rounds a half *away from zero*. The two agree for
``x >= 0`` and differ by exactly 1 at every negative half: ``rint(-2.5)`` was -3
instead of -2, and ``rint(-0.5)`` was -1 instead of 0. The docs promised
"backward compatibility with BNG's ``rint()``".

It was wrong the same way in each place ``rint`` is evaluated, so each is pinned
here against the literal values of ``floor(x + 0.5)``:

* the ExprTk interpreter, which ODE and SSA share (``RintFunction`` in
  src/expression.cpp, now ``expr_compat::rint``);
* the C codegen, which renamed ``rint`` to C's ``round`` and now emits
  ``floor((x) + 0.5)``.

NFsim's shim forwards to the same ``expr_compat::rint``; it is pinned in
test_nfsim_exprtk_parity.py. The JAX leg is pinned against the engine by
test_jax_unmapped_functions.py.
"""

from __future__ import annotations

import math
import textwrap
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import bngsim
import pytest
from bngsim._codegen import _replace_engine_calls

#: Arguments, with BNG's floor(x + 0.5) for each. The negative halves are the
#: cases that discriminate; the others must not move. The last is the edge where
#: x + 0.5 itself rounds up to 1.0 in binary64, so BNG's rint gives 1 there.
CASES = [
    (-2.5, -2.0),
    (-1.5, -1.0),
    (-0.5, 0.0),
    (0.5, 1.0),
    (1.5, 2.0),
    (2.5, 3.0),
    (-0.3, 0.0),
    (-2.7, -3.0),
    (0.49999999999999994, 1.0),
]


def test_the_cases_are_floor_x_plus_half():
    """The oracle column is what it claims to be."""
    for x, want in CASES:
        assert math.floor(x + 0.5) == want


def _net(tmp_path: Path, c: float, body: str) -> str:
    """0 -> B at the rate `body`, with parameter c. B(t) = rate * t exactly."""
    path = tmp_path / f"rint_{abs(hash((c, body)))}.net"
    path.write_text(
        textwrap.dedent(
            f"""
            begin parameters
              1 c {c!r}
            end parameters
            begin species
              1 B() 0
            end species
            begin functions
              1 r() {body}
            end functions
            begin reactions
              1 0 1 r
            end reactions
            begin groups
              1 Btot 1
            end groups
            """
        ).strip()
        + "\n"
    )
    return str(path)


def _ode_rate(net: str, codegen: bool) -> float:
    model = bngsim.Model.from_net(net)
    res = bngsim.Simulator(model, method="ode", codegen=codegen).run(t_span=(0, 1), n_points=2)
    return float(res.observables["Btot"][-1])


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
@pytest.mark.parametrize("x, want", CASES)
def test_ode_rint_is_floor_x_plus_half(tmp_path, codegen, x, want):
    """A constant rate `rint(c) + 10` integrates to B(1) = rint(c) + 10."""
    got = _ode_rate(_net(tmp_path, x, "rint(c) + 10"), codegen)
    assert got == pytest.approx(want + 10.0, rel=1e-9, abs=1e-9)


def test_ssa_takes_the_bng_branch(tmp_path):
    """SSA shares the interpreter's evaluator. rint(-0.5) is 0 under BNG, so the
    rate is 10 and B fires about 100 times in t = 10; under std::round it was -1,
    the rate 0, and B never fired."""
    model = bngsim.Model.from_net(_net(tmp_path, -0.5, "if(rint(c) == 0, 10, 0)"))
    res = bngsim.Simulator(model, method="ssa").run(t_span=(0, 10), n_points=2, seed=1)
    assert res.observables["Btot"][-1] > 0


def test_codegen_emits_floor_not_round():
    """The C spelling, independent of any compiler: BNG's arithmetic, in the
    same order as expr_compat::rint, and never C's round()."""
    out = _replace_engine_calls("k*rint(A-2)")
    assert out == "k*floor((A-2) + 0.5)"
    assert "round" not in out


def test_codegen_rewrites_nested_rint():
    assert _replace_engine_calls("rint(rint(x)/2)") == "floor((floor((x) + 0.5)/2) + 0.5)"


def test_rint_differs_from_round_at_negative_halves(tmp_path):
    """`round` is C's (a half away from zero, issue #684). The two must
    disagree at -2.5 and agree at 2.5. They also disagree where rint's x + 0.5
    itself rounds, as the expression reference says: BNG's
    rint(0.49999999999999994) is floor(1.0) = 1, round of it 0."""
    at_neg = _ode_rate(_net(tmp_path, -2.5, "rint(c) - round(c)"), codegen=False)
    at_pos = _ode_rate(_net(tmp_path, 2.5, "rint(c) - round(c) + 1"), codegen=False)
    edge = _net(tmp_path, 0.49999999999999994, "rint(c) - round(c) + 1")
    at_edge = _ode_rate(edge, codegen=False)
    assert at_neg == pytest.approx(1.0)  # -2 - (-3)
    assert at_pos == pytest.approx(1.0)  # 3 - 3 + 1
    assert at_edge == pytest.approx(2.0)  # 1 - 0 + 1


#: round(x) arguments where ExprTk's builtin, floor(x + 0.5) or ceil(x - 0.5)
#: below zero, is off by one because that addition rounds (issue #684): the
#: largest double below 0.5, and an odd integer in [2^52, 2^53). Then halves,
#: which must not move.
ROUND_CASES = [
    0.49999999999999994,
    -0.49999999999999994,
    4503599627370497.0,
    -4503599627370497.0,
    2.5,
    -2.5,
    0.5,
    -0.5,
    -2.7,
]


def _c_round(x: float) -> float:
    """C's round, exactly: a Decimal holds the double without error, and its
    ROUND_HALF_UP is a half away from zero."""
    return float(Decimal(x).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _function_value(net: str, codegen: bool) -> float:
    """The function `r` itself at t = 0, read off rather than integrated, so a
    value near 2^52 is compared exactly."""
    model = bngsim.Model.from_net(net)
    res = bngsim.Simulator(model, method="ode", codegen=codegen).run(t_span=(0, 1e-9), n_points=2)
    return float(res.expressions["r"][0])


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
@pytest.mark.parametrize("x", ROUND_CASES)
def test_round_is_c_round_on_both_paths(tmp_path, codegen, x):
    """The interpreter evaluated ExprTk's round and codegen C's, so one model
    gave round(0.49999999999999994) = 1 or 0 depending on whether codegen ran
    (issue #684). Both are C's round now."""
    assert _function_value(_net(tmp_path, x, "round(c)"), codegen) == _c_round(x)
