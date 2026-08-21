"""Issue #460 — the ExprTk emitter wrote sympy's spelling of Min and Max.

A rate law whose derivative still carries a ``max()`` or a ``min()`` lost its
analytic Jacobian. The emitter printed ``Max(k1, k2)``, the engine's expression
parser is case sensitive, and so it answered "Undefined symbol: 'Max'" and the
model quietly fell back to the finite-difference Jacobian.

Why it happened is worth keeping, because it is the shape of the bug rather than
the bug itself. ``_SYMPY_FUNC_TO_EXPRTK`` does map ``Min`` and ``Max`` to the
engine's names, and those two entries were never once read. sympy's ``Min`` and
``Max`` are not ``Function`` subclasses, so they never reach the printer's
``_print_Function``, and sympy's own ``StrPrinter._print_LatticeOp`` handled them
instead. That method prints the class name. The C emitter was never affected,
because it has carried its own ``_print_Min`` and ``_print_Max`` all along.

The same defect cost a second thing nobody was looking for. The GH #333
zero-logarithm guard rewrites a rate law by parsing it to sympy and printing it
back through this printer, so a law carrying both a guardable logarithm and a
``max()`` came back spelled ``Max(...)``, the engine refused to install it, and
the guard was silently dropped. That law then returns ``nan`` at zero
concentration where it should return zero.

A ``max()`` over a variable being differentiated is a separate matter and stays
refused on purpose, since its derivative is a step. This is only about one that
survives differentiation untouched, such as a ``max()`` over two parameters.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._jacobian import attach_functional_jacobian, guard_rate_law_text

sp = pytest.importorskip("sympy")

NET = """begin parameters
    1 k1 0.3  # Constant
    2 k2 0.5  # Constant
    3 n  2.0  # Constant
end parameters
begin observables
    1 Molecules Aobs A
end observables
begin functions
    1 law() {body}
end functions
begin species
    1 A() 10.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 Aobs                    1
end groups
"""


def _model(tmp_path, body: str, tag: str = ""):
    p = tmp_path / f"m{tag}.net"
    p.write_text(NET.format(body=body))
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(str(p))


def _core(model):
    return model._core if hasattr(model, "_core") else model


# ── The emitted spelling ─────────────────────────────────────────────────────


def test_min_and_max_emit_the_engines_spelling():
    from bngsim._jacobian import sympy_to_exprtk

    a, b = sp.Symbol("a"), sp.Symbol("b")
    assert sympy_to_exprtk(sp.Max(a, b)) == "max(a,b)"
    assert sympy_to_exprtk(sp.Min(a, b)) == "min(a,b)"


def test_an_n_ary_min_or_max_folds_into_binary_calls():
    """Matching the C twin, and the shape the loader already produces for an
    n-ary ``max()`` written in a model's own text."""
    from bngsim._jacobian import sympy_to_c, sympy_to_exprtk

    a, b, c = sp.symbols("a b c")
    assert sympy_to_exprtk(sp.Max(a, b, c)) == "max(max(a,b),c)"
    assert sympy_to_c(sp.Max(a, b, c), lambda n: n) == "fmax(fmax(a, b), c)"


def test_the_engine_accepts_what_the_emitter_writes(tmp_path):
    """The property that actually matters, asked of the engine rather than of a
    string comparison. ``Max(k1,k2)`` is in here as the negative half: it is
    what the emitter used to write, and the engine rejects it, which is the
    whole reason this is a bug and not a cosmetic difference."""
    from bngsim._jacobian import sympy_to_exprtk

    emitted = sympy_to_exprtk(sp.Max(sp.Symbol("k1"), sp.Symbol("k2")))
    model = _model(tmp_path, f"{emitted}*Aobs", tag="_ok")
    assert _core(model)._eval_functions(0.0, [10.0])["law"] == pytest.approx(0.5 * 10.0)

    with pytest.raises(Exception, match="Max"):
        _model(tmp_path, "Max(k1, k2)*Aobs", tag="_bad")


# ── The analytic Jacobian it was losing ──────────────────────────────────────


@pytest.mark.parametrize("body", ["max(k1,k2)*Aobs", "min(k1,k2)*Aobs"])
def test_the_analytic_jacobian_attaches(tmp_path, body):
    assert attach_functional_jacobian(_core(_model(tmp_path, body, tag="_jac")))


def test_a_min_or_max_over_a_differentiation_variable_is_still_refused(tmp_path):
    """Its derivative is a step, so it has no emittable form and the model keeps
    the finite-difference Jacobian. Unchanged by this, and the reason the fix is
    only about the spelling."""
    assert not attach_functional_jacobian(_core(_model(tmp_path, "max(Aobs,k1)", tag="_step")))


def _solve(model):
    with contextlib.redirect_stderr(io.StringIO()):
        r = bngsim.Simulator(model, method="ode", codegen=False).run(
            t_span=(0.0, 20.0), n_points=21, rtol=1e-10, atol=1e-12
        )
    return np.asarray(r.species)


def test_the_trajectory_does_not_move(tmp_path, monkeypatch):
    """The analytic Jacobian and the finite-difference one solve the same
    problem, so gaining the first must not change the answer.

    ``BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0`` forces the finite-difference
    Jacobian, which is what this model was silently getting before the fix.
    """
    analytic = _solve(_model(tmp_path, "max(k1,k2)*Aobs", tag="_traj"))
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", "0")
    quotient = _solve(_model(tmp_path, "max(k1,k2)*Aobs", tag="_traj_fd"))
    np.testing.assert_allclose(analytic, quotient, rtol=1e-8, atol=1e-12)


# ── The zero-logarithm guard it was also losing ──────────────────────────────


def test_the_zero_log_guard_survives_a_max(tmp_path):
    """GH #333's guard rewrites a rate law through this same printer, so the
    spelling took the guard down with it. Without the guard the law answers
    ``nan`` at zero concentration instead of zero, which is what #333 exists to
    prevent."""
    guarded = guard_rate_law_text("max(k1,k2)*Aobs^n*ln(Aobs)")
    assert guarded is not None
    assert "max(k1,k2)" in guarded
    assert "Max(" not in guarded

    model = _model(tmp_path, "max(k1,k2)*Aobs^n*ln(Aobs)", tag="_guard")
    core = _core(model)
    from bngsim._jacobian import guard_function_expressions

    with contextlib.redirect_stderr(io.StringIO()):
        changed = guard_function_expressions(core)
    assert [name for name, _before, _after in changed] == ["law"]
    with contextlib.redirect_stderr(io.StringIO()):
        assert core._eval_functions(0.0, [0.0])["law"] == 0.0


# ── The blind spot behind it, closed for the rest of the family ──────────────


def test_a_lattice_operation_with_no_spelling_is_refused():
    """``StrPrinter._print_LatticeOp`` printing a class name is what made
    ``Max`` look emittable. Anything else reaching it now declines instead."""
    from bngsim._jacobian import _ExprTkEmitError, _make_printer

    class _Unknown(sp.Max):  # a lattice operation with no method of its own
        pass

    printer = _make_printer()()
    with pytest.raises(_ExprTkEmitError):
        printer._print_LatticeOp(_Unknown(sp.Symbol("a"), sp.Symbol("b")))


@pytest.mark.parametrize("name", ["Xor", "Implies", "Equivalent"])
def test_a_boolean_node_with_no_spelling_is_refused(name):
    """Boolean nodes share ``Min`` and ``Max``'s blind spot: they are
    ``Application`` but not ``Function``, so ``_is_emittable``'s scan never saw
    them either.

    ``Xor`` is the one that matters. sympy prints it infix as ``^``, which is
    legal ExprTk and means EXPONENTIATION there, so it would have been a wrong
    number rather than a refusal. Nothing in the loader builds one today, and
    this is the guard for the day something does.
    """
    from bngsim._jacobian import _is_emittable, sympy_to_c, sympy_to_exprtk
    from sympy.logic import boolalg

    a, b = sp.Symbol("a"), sp.Symbol("b")
    node = getattr(boolalg, name)(a > 0, b > 0)
    assert type(node).__name__ == name, "sympy no longer builds this node"
    assert not _is_emittable(node)
    assert sympy_to_exprtk(node) is None
    assert sympy_to_c(node, lambda n: n) is None


def test_the_boolean_nodes_that_do_have_a_spelling_still_emit():
    """The guard above must not take out the three connectives every
    conditional rate law is built from, nor ``ITE``, which ``_is_emittable``
    sees before ``_normalize_booleans`` rewrites it away."""
    from bngsim._jacobian import sympy_to_c, sympy_to_exprtk
    from sympy.logic.boolalg import ITE

    a, b, c = sp.symbols("a b c")
    for node in (
        sp.And(a > 0, b > 0),
        sp.Or(a > 0, b > 0),
        sp.Not(a > 0),
        ITE(a > 0, b > 0, c > 0),
    ):
        assert sympy_to_exprtk(node) is not None, node
        assert sympy_to_c(node, lambda n: n) is not None, node
