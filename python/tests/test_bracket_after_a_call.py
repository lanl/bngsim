"""A bracket after a call is refused, not read as a second statement (issue
#1415).

ExprTk multiplies a variable, and a bracketed group, against the bracket after
it: ``A(B)`` and ``(A)(B)`` are A·B. After a function call it does not. The
bracket opens a second statement, and the expression's value is the last
statement's: ``k*exp(-A)(B)`` with A = 3, B = 5 came back 5, where the product
is 0.124, and a ``.net`` rate law written so ran as its last factor with no
word. Inside a bracket it was refused with ExprTk's ``ERR244 - Expected ')'``.

BNG2.pl rejects the same text as a syntax error, so a network it writes never
holds one. The engine now refuses it at load, naming the call as written, and
so does NFsim's function parser; a variable or a group before a bracket is read
as the product, as before.
"""

from __future__ import annotations

import re

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 k 0.5
    2 k.g 2
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
        # ExprTk's vararg functions are calls, not the word operators they look
        # like: each was 5, for 2.5.
        ("k*mand(A > 1, B > 1)(B)", "mand(A > 1, B > 1)"),
        ("k*mor(A > 1, B > 1)(B)", "mor(A > 1, B > 1)"),
        # Named as written, not as the engine rewrites it (`&&` as ` and `).
        ("if(A > 1 && B > 1, 1, 0)(B)", "if(A > 1 && B > 1, 1, 0)"),
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
        # A dotted name is one variable, as ExprTk reads it: k.g·A·B, not `g(A)`.
        ("k.g(A)(B)", 30.0),
        # A literal the engine respells as a bracketed quotient after a call:
        # still the product the lexer reads, not the quotient alone.
        ("abs(B)0.49999999999999997", 5 * 0.49999999999999994),
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


def test_an_unknown_name_is_left_to_exprtk(tmp_path):
    """A name nothing registers is no call to refuse: ExprTk reports it, rather
    than advise writing ``nope(A)*(...)``, which would fail just the same."""
    with pytest.raises(bngsim.ModelError, match="nope") as caught:
        _model(tmp_path, "nope(A)(B)")
    assert "#1415" not in str(caught.value)


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not compiled in")
def test_an_nfsim_function_is_refused_too(data_dir, tmp_path):
    """NFsim compiles its XML's functions through a parser of its own, which
    read ``3*exp(0)(2)`` as 2, for 6."""
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    def column(body):
        fixture = (data_dir / "nfsim" / "exprtk_shim_parity.xml").read_text()
        function = (
            '<Function id="probe"><ListOfReferences></ListOfReferences>'
            f"<Expression> {body} </Expression></Function>\n"
        )
        path = tmp_path / "m.xml"
        # On a line of its own, as BNG2.pl writes each function.
        path.write_text(fixture.replace("<ListOfFunctions>", "<ListOfFunctions>\n" + function, 1))
        times = TimeSpec()
        times.t_start, times.t_end, times.n_points = 0.0, 1.0, 2
        result = NfsimSimulator(str(path)).run(times, 1)
        row = np.asarray(result.expression_data[0]).ravel()
        return float(row[list(result.expression_names).index("probe")])

    assert column("3*exp(0)*(2)") == 6.0
    with pytest.raises(RuntimeError, match=r"the call 'exp\(0\)'.*#1415"):
        column("3*exp(0)(2)")
