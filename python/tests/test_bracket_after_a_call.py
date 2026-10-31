"""A bracket after a call is refused, not read as a second statement (issue
#1415).

ExprTk multiplies a variable, and a bracketed group, against the bracket after
it: ``A(B)`` and ``(A)(B)`` are A·B. After a function call it does not. The
bracket opens a second statement, and the expression's value is the last
statement's: ``k*exp(-A)(B)`` with A = 3, B = 5 came back 5, where the product
is 0.124, and a ``.net`` rate law written so ran as its last factor with no
word. Inside a bracket it was refused with ExprTk's ``ERR244 - Expected ')'``.

BNG2.pl rejects the same text as a syntax error, so a network it writes never
holds one. The engine now refuses it at load, naming the call; a variable or a
group before a bracket is read as the product, as before.
"""

from __future__ import annotations

import re

import bngsim
import pytest

NET = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 3
    2 B() 5
    3 Y() 0
end species
begin groups
    1 A 1
    2 B 2
end groups
begin functions
    1 f() {law}
end functions
begin reactions
    1 0 3 f
end reactions
"""


def _model(tmp_path, law):
    path = tmp_path / "m.net"
    path.write_text(NET.replace("{law}", law))
    return bngsim.Model.from_net(str(path))


def _rate(tmp_path, law):
    core = _model(tmp_path, law)._core
    return float(core.compute_derivs(0.0, core.get_state())[2])


@pytest.mark.parametrize(
    ("law", "call"),
    [
        ("k*exp(-A)(B)", "exp(-A)"),  # 5 on main, for 0.124
        ("sin(A)(B)", "sin(A)"),  # 5, for 0.706
        ("2*abs(A)(B) + 1", "abs(A)"),  # 6, for 31
        ("time()(B)", "time()"),
        ("rint(A)(B)", "rint(A)"),
        ("if(A>1,1,0)(B)", "if(A>1,1,0)"),
        ("sin(A)[B]", "sin(A)"),
        ("sin(A) (B)", "sin(A)"),
        ("sin (A)(B)", "sin (A)"),
        # The loader writes an inline table as a call of its own, which is named.
        ("tfun([0,1],[0,1],time)(B)", "tfun_f__tfun0()"),
        ("k*(1 + exp(-A)(B))", "exp(-A)"),  # ERR244 on main, with no word of why
    ],
)
def test_a_bracket_after_a_call_is_refused_by_name(tmp_path, law, call):
    with pytest.raises(bngsim.ModelError, match=rf"the call '{re.escape(call)}'.*#1415"):
        _model(tmp_path, law)


@pytest.mark.parametrize(
    ("law", "want"),
    [
        ("A(B)", 15.0),
        ("(A)(B)", 15.0),
        ("A (B)", 15.0),
        ("2(B)", 10.0),
        ("k(B)", 2.5),
        ("k*exp(-A)*(B)", 0.5 * 2.718281828459045**-3 * 5),
        ("exp(-A)+(B)", 2.718281828459045**-3 + 5),
        ("sin(A)", 0.1411200080598672),
        ("1e-1(B)", 0.5),
        ("1e1(B)", 50.0),
        # A bracket after the product's own bracket: still the product, A·B·A.
        ("A(B)(A)", 45.0),
        ("k(B)(A)", 7.5),  # a parameter
        ("1e1(B)(A)", 150.0),  # an exponent is no name: `e1(` is not a call
        # A word operator is no call: (A > 1) and B·A, which is 1.
        ("(A > 1) and (B)(A)", 1.0),
    ],
)
def test_a_bracket_after_a_variable_a_group_or_a_number_is_the_product(tmp_path, law, want):
    """Controls: ExprTk's implicit product, and calls no bracket follows."""
    assert _rate(tmp_path, law) == pytest.approx(want, rel=1e-12)


def test_a_body_given_after_load_is_refused_too(tmp_path):
    """The engine's hook for replacing how a function is evaluated compiles
    through the same path."""
    model = _model(tmp_path, "k")
    with pytest.raises(RuntimeError, match=r"the call 'exp\(-A\)'.*#1415"):
        model._core.set_function_eval_expression("f", "k*exp(-A)(B)")
