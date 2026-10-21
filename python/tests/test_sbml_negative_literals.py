"""GH #744: a negative MathML literal keeps its meaning in the ExprTk text.

The translator used to splice ``-3`` bare into operator templates, so
``(-3)^2`` read as ``-(3^2)`` and ``a - (-2)`` became ``a--2.0``, which the C
compiler rejects. The same bare text reached a power base through a
function-definition argument and a constant assignment-rule lift.
"""

from __future__ import annotations

import bngsim
import pytest
from bngsim._sbml_loader import _atomic, _real_literal

pytest.importorskip("antimony")

_CASES = {
    "power_base": ("X' = (-2)^2", 4.0),
    "minus_negative": ("a = 1; X' = a - (-2)", 3.0),
    "unary_minus_negative": ("X' = -(-2)", 2.0),
    "funcdef_argument": ("X' = sq(-3)", 9.0),
    "assignment_rule_lift": ("b := -3; X' = b^2", 9.0),
}


def _x_at_one(body: str, codegen: bool) -> float:
    src = (
        f"function sq(x)\n  x^2\nend\nmodel m; compartment c = 1; species X in c = 0; {body}; end"
    )
    m = bngsim.Model.from_antimony_string(src)
    r = bngsim.Simulator(m, method="ode", codegen=codegen).run(t_span=(0.0, 1.0), n_points=2)
    return float(r.species[-1, 0])


@pytest.mark.parametrize("codegen", [False, True], ids=["interp", "codegen"])
@pytest.mark.parametrize("case", list(_CASES))
def test_negative_literal_keeps_its_meaning(case, codegen):
    body, rate = _CASES[case]
    # X' is a constant, so X(1) is that constant.
    assert _x_at_one(body, codegen) == pytest.approx(rate, rel=1e-9)


def test_real_literal_parenthesises_negatives():
    assert _real_literal(3.0) == "3.0"
    assert _real_literal(-3.0) == "(-3.0)"
    assert _real_literal(-0.0) == "(-0.0)"
    assert _real_literal(float("-inf")) == "(-1.0/0.0)"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("k", "k"),
        ("2.5e-3", "2.5e-3"),
        ("(-3)", "(-3)"),
        ("(a+b)", "(a+b)"),
        ("-3", "(-3)"),
        ("a+b", "(a+b)"),
        ("(a)+(b)", "((a)+(b))"),
        # A call binds tighter than any operator, so it stays bare (issue #1082).
        ("f(x)", "f(x)"),
        ("time()", "time()"),
        ("sin(time())", "sin(time())"),
        ("f(x)+g(y)", "(f(x)+g(y))"),
        ("f(x)^2", "(f(x)^2)"),
        ("-f(x)", "(-f(x))"),
        ("f(x)(y)", "(f(x)(y))"),
    ],
)
def test_atomic_wraps_everything_but_a_single_token_or_call(text, expected):
    assert _atomic(text) == expected
