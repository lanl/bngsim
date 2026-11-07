"""Issue #1495 — a comparison or logical result that arithmetic reads is a double.

ExprTk computes ``A > thr`` as a double, 1 or 0, and so does the interpreter.
The C generator copied the comparison through as text, and in C it is an
``int``, as are ``==``, ``!=``, ``&&``, ``||`` and ``!``. Two ints divide as
ints, so ``((A>thr)+(B>thr))/((A>0)+(B>0))`` was 1/2 = 0 compiled where the
interpreter has 0.5: a species fed at that rate reached 0 instead of 5, with no
warning, under ``codegen=True`` and under the default once a model has 256
species. ``(A>thr)/(C>thr)`` with ``C`` below ``thr`` divided an int by zero,
which is undefined behaviour in C where the interpreter gives inf, and
``-(C>thr)`` was 0 where the engine has -0.0, so ``1/-(C>thr)`` was +inf
compiled and -inf interpreted.

Each comparison or logical group that arithmetic reads is now written
``((...) ? 1.0 : 0.0)``. A condition, an operand of ``&&`` or ``||``, or a
call's argument reads an int as it reads a double, and is left as it was, so
a model with no comparison in arithmetic emits the source it did.
"""

from __future__ import annotations

import contextlib
import io
import math

import bngsim
import bngsim._codegen as _cg
import numpy as np
import pytest
from bngsim._codegen import _double_valued_logic, _expr_to_c

# A and B are constant (no reaction touches them) and `0 -> X` runs at r(), so
# X(10) = 10*r() exactly.
NET = """begin parameters
    1 thr 1.0
    2 k 2.0
    3 a 3.0
    4 b 0.7
end parameters
begin species
    1 A() 3.0
    2 B() 0.7
    3 X() 0
{inert}end species
begin functions
    1 r() {law}
end functions
begin reactions
    1 0 3 r
end reactions
begin groups
    1 A_tot 1
    2 B_tot 2
end groups
"""


def _model(tmp_path, law: str, n_inert: int = 0):
    inert = "".join(f"    {i} D{i}() 1.0\n" for i in range(4, 4 + n_inert))
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=law, inert=inert))
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(str(path))


def _x10(tmp_path, law: str, **kwargs) -> float:
    model = _model(tmp_path, law, kwargs.pop("n_inert", 0))
    sim = bngsim.Simulator(model, method="ode", **kwargs)
    with contextlib.redirect_stderr(io.StringIO()):
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
    return float(np.asarray(result.species)[-1, list(model.species_names).index("X()")])


# (rate law, its value at A = 3, B = 0.7, thr = 1). Every quotient below has a
# comparison or a logical result on both sides, which is the case that divided
# as ints.
LAWS = [
    ("((A_tot>thr)+(B_tot>thr))/((A_tot>0)+(B_tot>0))", 0.5),
    ("sum(A_tot>thr, B_tot>thr)/sum(A_tot>0, B_tot>0)", 0.5),
    ("(A_tot>thr)/((A_tot>0)+(B_tot>0))", 0.5),
    # Left to right, `(A>thr)/(...)*k` divided as ints before k could promote it.
    ("(A_tot>thr)/((A_tot>0)+(B_tot>0))*k", 1.0),
    ("(A_tot>thr && B_tot>0)/((A_tot>0)+(B_tot>0))", 0.5),
    ("(A_tot>thr and B_tot>0)/((A_tot>0) or (B_tot>0)) - 0.5", 0.5),
    ("((A_tot>thr) || (B_tot>thr))/((A_tot>0)+(B_tot>0))", 0.5),
    # ExprTk's `&` and `|` are its `and` and `or`.
    ("((A_tot>thr) & (B_tot>0))/((A_tot>0)+(B_tot>0))", 0.5),
    ("((A_tot>thr) | (B_tot>thr))/((A_tot>0)+(B_tot>0))", 0.5),
    ("not(B_tot>thr)/((A_tot>0)+(B_tot>0))", 0.5),
    ("(not(B_tot>thr)+not(A_tot>thr))/((A_tot>0)+(B_tot>0))", 0.5),
    ("if(A_tot>thr, B_tot>thr, A_tot>0)/((A_tot>0)+(B_tot>0))", 0.0),
    ("if(A_tot<thr, B_tot>thr, A_tot>0)/((A_tot>0)+(B_tot>0))", 0.5),
    ("(A_tot>thr)/((A_tot>0)+(B_tot>0))^2", 0.25),
    # -(B>thr) is -0.0, so 1/-(B>thr) is -inf, and atan of it -pi/2. As an int
    # the minus was lost: +inf, and +pi/2.
    ("2 + atan(1/-(B_tot>thr))", 2.0 - math.pi / 2),
]


@pytest.mark.parametrize("law, value", LAWS, ids=[law for law, _ in LAWS])
def test_the_compiled_run_reaches_the_closed_form(tmp_path, law, value):
    interpreted = _x10(tmp_path, law, codegen=False)
    compiled = _x10(tmp_path, law, codegen=True)
    assert interpreted == pytest.approx(10.0 * value, rel=1e-9, abs=1e-12)
    # The same arithmetic in the same order, so this is an equality.
    assert compiled == interpreted


def test_the_auto_codegen_threshold_runs_it_compiled_too(tmp_path):
    """At 256 species the default runs compiled, and it got 0 for 5 there."""
    law = LAWS[0][0]
    model = _model(tmp_path, law, n_inert=253)
    assert model.n_species == 256
    sim = bngsim.Simulator(model, method="ode")
    with contextlib.redirect_stderr(io.StringIO()):
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
    assert sim.codegen_backend in ("cc", "mir")
    x = float(np.asarray(result.species)[-1, list(model.species_names).index("X()")])
    assert x == pytest.approx(5.0, rel=1e-9)


def test_a_division_by_a_false_comparison_fails_as_the_interpreter_does(tmp_path):
    """1/0 is inf to the engine and the run is refused. As an int division it was
    undefined behaviour, which on arm64 came back 1 at -O3 and 0 at -O0."""
    law = "(A_tot>thr)/(B_tot>thr)"
    for codegen in (False, True):
        with pytest.raises(bngsim.SimulationError, match="non-finite"):
            _x10(tmp_path, law, codegen=codegen)


def test_a_sensitivity_run_reads_the_quotient_in_a_condition(tmp_path):
    """A quotient of comparisons inside an if() condition is admitted on the
    forward-sensitivity path, which runs compiled at any model size. Both the
    trajectory and the column were 0: X(10) = 10*k*A = 60, dX/dk = 10*A = 30."""
    model = _model(tmp_path, "if(((a>thr)+(b>thr))/((a>0)+(b>0)) > 0.3, k*A_tot, 0)")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    with contextlib.redirect_stderr(io.StringIO()):
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
    x = list(model.species_names).index("X()")
    assert float(np.asarray(result.species)[-1, x]) == pytest.approx(60.0, rel=1e-9)
    assert float(np.asarray(result.sensitivities)[-1, x, 0]) == pytest.approx(30.0, rel=1e-6)


def test_a_recorded_function_of_parameters_reads_the_same(tmp_path):
    """With nothing that switches on the state, the compiled outputs evaluator
    records the functions, and it recorded 0 for 0.5 here."""
    model = _model(tmp_path, "((a>thr)+(b>thr))/((a>0)+(b>0))")
    for codegen in (False, True):
        sim = bngsim.Simulator(model, method="ode", codegen=codegen)
        with contextlib.redirect_stderr(io.StringIO()):
            result = sim.run(t_span=(0.0, 10.0), n_points=11)
        assert np.asarray(result.expressions["r"]) == pytest.approx(0.5, abs=0)


# ── The emitted text ─────────────────────────────────────────────────────────

PARAMS, OBSERVABLES = ["thr", "k"], ["A", "B", "C"]


@pytest.mark.parametrize(
    "expr, expect",
    [
        ("(A>thr)/(B>thr)", "((obs[0]>p[0]) ? 1.0 : 0.0)/((obs[1]>p[0]) ? 1.0 : 0.0)"),
        ("1/-(C>thr)", "1.0/-((obs[2]>p[0]) ? 1.0 : 0.0)"),
        # All of a group's content is read where the group is.
        ("((A>thr))/2", "(((obs[0]>p[0]) ? 1.0 : 0.0))/2.0"),
        ("(A>thr && B>0)*k", "((obs[0]>p[0] && obs[1]>0.0) ? 1.0 : 0.0)*p[1]"),
        ("((A>thr) & (B>0))*k", "(((obs[0]>p[0]) && (obs[1]>0.0)) ? 1.0 : 0.0)*p[1]"),
        # A ternary's branches are values.
        (
            "if(k, A>thr, B>thr)/k",
            "((p[1]) ? ((obs[0]>p[0]) ? 1.0 : 0.0) : ((obs[1]>p[0]) ? 1.0 : 0.0))/p[1]",
        ),
        # An operand of + is a value even beside a comparison.
        ("k + (A>thr) != 0", "p[1] + ((obs[0]>p[0]) ? 1.0 : 0.0) != 0.0"),
    ],
)
def test_arithmetic_reads_a_double(expr, expect):
    assert _expr_to_c(expr, PARAMS, [], OBSERVABLES, []) == expect


@pytest.mark.parametrize(
    "expr",
    [
        # The whole expression: it is assigned to a double.
        "A>thr",
        "(A>thr)",
        # Conditions, and the operands of && and ||.
        "if(A>thr, k, 0)",
        "if((A>thr)&&(B<thr),1,0)",
        "if(((A>thr)) && (B<thr), 1, 0)",
        "if(A>thr || B>thr && C>thr, k, 0)",
        # Operands of a comparison read 0/1 as 0.0/1.0.
        "(A>thr)==(B>thr)",
        # A call's argument is converted by the call.
        "max(A>thr, B>thr)",
        "(A>thr)^2",
        # The engine calls that were already doubles.
        "mand(A>1,B>1)*k",
        "mor(A>1,B>1)/k",
        "sign(A-thr)*k",
        "clamp(0, A, thr)*k",
    ],
)
def test_a_read_as_a_truth_value_is_left_as_it_was(expr, monkeypatch):
    """What emitted the same before is emitted the same now: the pass is the
    identity on it."""
    emitted = _expr_to_c(expr, PARAMS, [], OBSERVABLES, [])
    monkeypatch.setattr(_cg, "_double_valued_logic", lambda c: c)
    assert _expr_to_c(expr, PARAMS, [], OBSERVABLES, []) == emitted


def test_a_member_access_is_not_a_comparison():
    """`data->tfun_eval` holds a `>`; it is not a comparison."""
    text = "(data->tfun_eval(0, t, data->tfun_ctx) + 1.0)/2.0"
    assert _double_valued_logic(text) == text


def test_not_emits_no_c_int_operator():
    """`not` is ExprTk's notl_impl, a double, and never C's `!` (issue #1506)."""
    out = _expr_to_c("not(A>thr)/not(B>thr)", PARAMS, [], OBSERVABLES, [])
    assert "!" not in out.replace("!=", "")
    assert out == ("(((obs[0]>p[0]) != 0.0) ? 0.0 : 1.0)/(((obs[1]>p[0]) != 0.0) ? 0.0 : 1.0)")


# ── The JAX translation, which had the same fault in its own form ─────────────
#
# JAX computes a comparison as a boolean, and adds two booleans as a logical or:
# ((A>0)+(B>0)) was 1 where the engine has 2, and -(A>B) raised. The values are
# checked against the interpreter in test_expression_backend_equivalence.py;
# these pin which text the pass touches. They need no JAX: the pass is textual.


@pytest.mark.parametrize(
    "text",
    [
        "jnp.where(obs[0] > obs[1], params[0], params[1])",
        "jnp.where(jnp.logical_and(obs[0] > 1.0, obs[1] < 2.0), 1.0, 0.0)",
        "params[0] * obs[0] / (params[1] + obs[0])",
    ],
)
def test_jax_leaves_a_condition_as_it_was(text):
    from bngsim._jax_rhs import _booleans_as_doubles

    assert _booleans_as_doubles(text) == text


@pytest.mark.parametrize(
    "text, expect",
    [
        (
            "(obs[0] > 0.0) + (obs[1] > 0.0)",
            "jnp.where(obs[0] > 0.0, 1.0, 0.0) + jnp.where(obs[1] > 0.0, 1.0, 0.0)",
        ),
        ("-(obs[0] > obs[1])", "-jnp.where(obs[0] > obs[1], 1.0, 0.0)"),
        ("obs[0] > obs[1]", "jnp.where(obs[0] > obs[1], 1.0, 0.0)"),
        (
            "jnp.where(params[0], obs[0] > 1.0, obs[1] > 1.0) * 2.0",
            "jnp.where(params[0], jnp.where(obs[0] > 1.0, 1.0, 0.0), "
            "jnp.where(obs[1] > 1.0, 1.0, 0.0)) * 2.0",
        ),
        (
            "jnp.logical_not(obs[0] > 1.0) / 2.0",
            "jnp.where(jnp.logical_not(obs[0] > 1.0), 1.0, 0.0) / 2.0",
        ),
        # A comparison's operands are left as booleans; the comparison, the
        # whole expression, is a value.
        (
            "jnp.logical_not(obs[0] > obs[1]) == (obs[1] > obs[0])",
            "jnp.where(jnp.logical_not(obs[0] > obs[1]) == (obs[1] > obs[0]), 1.0, 0.0)",
        ),
    ],
)
def test_jax_reads_a_double_where_arithmetic_does(text, expect):
    from bngsim._jax_rhs import _booleans_as_doubles

    assert _booleans_as_doubles(text) == expect
