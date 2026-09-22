"""GH #556 — an n-ary ``max``/``min`` must reach the generated C as binary calls.

ExprTk's ``max`` and ``min`` take any number of arguments; C's ``fmax`` and
``fmin`` take exactly two. The identifier pass renamed ``max`` to ``fmax`` and
left the arguments alone, so ``max(A,2,3)`` became ``fmax(obs_A,2.0,3.0)``, which
the compiler rejects ("too many arguments to function call, expected 2, have
3"). A model loaded through the sympy round trip never showed this, because that
emitter already folds an n-ary call. A ``.net`` function body keeps the call as
written, so an explicit ``codegen=True`` run and every forward sensitivity run of
such a model failed to build while the interpreted run was fine.

The fix folds the call left to right, the way ExprTk reduces it:
``max(a,b,c)`` becomes ``max(max(a,b),c)``, and the rename then applies to each
binary call. A binary call is the only form that ever compiled, and it is left
byte-for-byte as written, so no existing model's generated source changes.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _expr_to_c, _replace_engine_calls

NET = """begin parameters
    1 k       0.1  # Constant
    2 n       2.5  # Constant
end parameters
begin functions
    1 law() {body}
end functions
begin species
    1 A() {a0}
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
end groups
"""


def _model(tmp_path, body: str, a0: float = 5.0):
    p = tmp_path / "m.net"
    p.write_text(NET.format(body=body, a0=a0))
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(str(p))


def _run(tmp_path, body: str, a0: float = 5.0, codegen: bool = False, t_end: float = 20.0):
    m = _model(tmp_path, body, a0)
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Simulator(m, method="ode", codegen=codegen).run(
            t_span=(0.0, t_end), n_points=21
        )


# ── The rewriter itself ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expr, expect",
    [
        ("max(a,b,c)", "max(max(a,b),c)"),
        ("min(a,b,c)", "min(min(a,b),c)"),
        ("max(a,b,c,d)", "max(max(max(a,b),c),d)"),
        ("max(a, b, c)", "max(max(a,b),c)"),
        # an n-ary call inside another n-ary one
        ("max(a,min(b,c,d),e)", "max(max(a,min(min(b,c),d)),e)"),
        # ...and inside a binary one, which is itself left as written
        ("max(a,min(b,c,d))", "max(a,min(min(b,c),d))"),
        # ...and inside the other engine calls
        ("sum(max(a,b,c))", "((max(max(a,b),c)))"),
        # arguments that are expressions, with commas of their own
        ("max(a+b, pow(b,c), 2)", "max(max(a+b,pow(b,c)),2)"),
    ],
)
def test_an_n_ary_call_folds_left_to_right(expr, expect):
    assert _replace_engine_calls(expr) == expect


@pytest.mark.parametrize(
    "expr",
    [
        "max(a,b)",
        "min(a,b)",
        # spacing inside a binary call survives exactly
        "max (a, b)",
        "min( a ,b )",
        "max(a,min(b,c))",
        # a longer name that merely ends in one
        "vmax(a,b,c)",
        "k*maxval",
    ],
)
def test_a_binary_call_is_left_byte_for_byte(expr):
    assert _replace_engine_calls(expr) == expr


def test_the_issue_expression_translates_to_binary_c():
    out = _expr_to_c("max(a,b,c)", ["a", "b", "c"], [], [], [])
    assert out == "fmax(fmax(p[0],p[1]),p[2])"
    out = _expr_to_c("min(a,b,c)", ["a", "b", "c"], [], [], [])
    assert out == "fmin(fmin(p[0],p[1]),p[2])"


# ── The models build, and compute what the interpreter computes ──────────────

# A starts at 5 and decays through every threshold below, so each case switches
# which argument wins at least once over the window.
CASES = [
    "k*max(A,2,3)",
    "k*min(A,2,3)",
    "k*A*min(A,4,3.5,n)",
    "k*max(1,min(A,4,3))",
    "k*A*max(min(A,4,3),n,1)",
]


@pytest.mark.parametrize("body", CASES)
def test_the_compiled_path_matches_the_interpreter(tmp_path, body):
    interpreted = np.asarray(_run(tmp_path, body, codegen=False).species)[:, 0]
    compiled = np.asarray(_run(tmp_path, body, codegen=True).species)[:, 0]
    # Same reduction in the same order, so this is an equality, not a tolerance.
    assert compiled == pytest.approx(interpreted, rel=0, abs=0)
    # ...and the model went somewhere, so the equality is not two flat lines.
    assert interpreted[-1] < interpreted[0]


def test_it_matches_the_nested_spelling(tmp_path):
    """The n-ary call and the nested form the loader writes are the same model."""
    nary = np.asarray(_run(tmp_path, "k*A*min(A,4,3.5,n)", codegen=True).species)[:, 0]
    nested = np.asarray(_run(tmp_path, "k*A*min(min(min(A,4),3.5),n)", codegen=True).species)[:, 0]
    assert nary == pytest.approx(nested, rel=0, abs=0)
