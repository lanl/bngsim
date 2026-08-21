"""Issue #448 — the engine's own functions must reach the generated C as C.

``sign``, ``sgn``, ``clamp``, ``avg`` and ``sum`` are in the engine's reserved
function list (``reserved_names()`` in src/expression.cpp), so a model is
allowed to call them and the interpreter evaluates them. C has none of them.
Before the fix the name went into the generated source unchanged and the
compile failed with "call to undeclared function", which took down an explicit
``codegen=True`` run and every forward sensitivity run of the same model. The
plain interpreted run was fine throughout, so the failure only showed up once a
user asked for speed or for gradients.

Two claims are checked here, and the second is the one that is easy to lose.
The first is that these models now build. The second is that what they compute
is what the interpreter computes, down to the last bit, which is why the
crossed-bounds clamp has a test of its own: it is the one case where the
obvious ``fmin``/``fmax`` spelling gives a different answer, so without it a
later simplification could pass everything else here and still be wrong.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _replace_engine_calls

NET = """begin parameters
    1 k       0.1  # Constant
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


def _run(tmp_path, body: str, a0: float = 5.0, codegen: bool = False, t_end: float = 2.0):
    m = _model(tmp_path, body, a0)
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Simulator(m, method="ode", codegen=codegen).run(
            t_span=(0.0, t_end), n_points=11
        )


# ── The rewriter itself ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expr, expect",
    [
        ("k*sign(A)", "k*(((A) > 0.0) ? 1.0 : (((A) < 0.0) ? -1.0 : 0.0))"),
        ("sgn(A-3)", "(((A-3) > 0.0) ? 1.0 : (((A-3) < 0.0) ? -1.0 : 0.0))"),
        ("clamp(1,A,4)", "(((A) < (1)) ? (1) : (((A) > (4)) ? (4) : (A)))"),
        ("sum(A,2,3)", "((A) + (2) + (3))"),
        ("avg(A,2,3)", "(((A) + (2) + (3)) / 3.0)"),
    ],
)
def test_each_engine_call_becomes_c(expr, expect):
    assert _replace_engine_calls(expr) == expect


@pytest.mark.parametrize(
    "expr",
    [
        "law()",  # a .net calls a model function with empty parens
        "sum()",  # …including one that happens to be spelled like a built-in
        "mysum(A,2)",  # a longer name that merely ends in one
        "assign(x)",  # …or merely contains one
        "k*signal",  # a bare identifier, no call at all
        "sign(A,2)",  # an argument count the engine would have rejected
        "clamp(1,A)",
    ],
)
def test_what_the_rewriter_must_leave_alone(expr):
    assert _replace_engine_calls(expr) == expr


def test_nested_calls_are_rewritten_inside_out():
    """An engine call inside another one, and inside an if(), must be reached."""
    out = _replace_engine_calls("sum(sgn(A), avg(B,C))")
    assert "sgn(" not in out
    assert "avg(" not in out
    assert "sum(" not in out
    assert "? 1.0 :" in out and "/ 2.0" in out


# ── The models build, and compute what the interpreter computes ──────────────

# body, initial A, and whether A should grow over the window. The direction is
# asserted so that a case cannot pass by both paths standing still: for sign
# and sgn it is the branch under test that decides it.
CASES = [
    ("k*sign(A-3)", 5.0, False),
    ("k*sign(A-3)", 1.0, True),
    ("k*sgn(A-3)", 5.0, False),
    ("k*sgn(A-3)", 1.0, True),
    ("k*clamp(1,A,4)", 5.0, False),
    ("k*clamp(4,A,1)", 5.0, False),
    ("k*clamp(4,A,1)", 2.0, False),
    ("k*avg(A,2)", 5.0, False),
    ("k*avg(A,2,3,4,5,6)", 5.0, False),
    ("k*sum(A,2)", 5.0, False),
    ("k*sum(A,2,3,4)", 5.0, False),
    ("k*sgn(clamp(1,A,4)-2)*avg(A,sum(A,1))", 5.0, False),
    ("k*if(sum(A,2)>4, sgn(A-3), avg(A,1))", 5.0, False),
]


@pytest.mark.parametrize("body, a0, grows", CASES, ids=[c[0] for c in CASES])
def test_the_compiled_path_matches_the_interpreter(tmp_path, body, a0, grows):
    interpreted = np.asarray(_run(tmp_path, body, a0, codegen=False).species)[:, 0]
    compiled = np.asarray(_run(tmp_path, body, a0, codegen=True).species)[:, 0]
    # Same arithmetic in the same order, so this is an equality, not a tolerance.
    assert compiled == pytest.approx(interpreted, rel=0, abs=0)
    # …and the model went somewhere, so the equality above is not two flat lines.
    moved = interpreted[-1] - interpreted[0]
    assert (moved > 0) if grows else (moved < 0)


@pytest.mark.parametrize("a0, expect", [(2.0, 4.0), (6.0, 1.0)])
def test_the_crossed_bounds_clamp_follows_the_engine(tmp_path, a0, expect):
    """``clamp(lo, x, hi)`` with lo above hi, which pins the exact spelling.

    The engine returns ``lo`` when ``x < lo`` and ``hi`` when ``x > hi``, tested
    in that order, so with the bounds crossed a low ``x`` gives ``lo`` and a
    high one gives ``hi``. Neither of the two obvious two-call spellings
    reproduces both halves. ``fmax(lo, fmin(x, hi))`` gets the low ``x`` right
    and answers 4 instead of 1 for the high one; ``fmin(hi, fmax(x, lo))`` gets
    the high one right and answers 1 instead of 4 for the low one. So both rows
    below are needed, and both are read off the compiled model rather than off
    the interpreter, because the emitted C is what is under test.

    Crossed bounds are a nonsense model, but ``lo`` and ``hi`` can be fitted
    parameters, and a fit that walks them past each other should not change
    which of the two paths a user is on.
    """
    m = _model(tmp_path, "k*clamp(4,A,1)", a0)
    core = m._core if hasattr(m, "_core") else m
    assert core._eval_functions(0.0, [a0])["law"] / 0.1 == pytest.approx(expect)

    # dA/dt = -law * A at t = 0, so the opening slope reads the clamp back out.
    result = _run(tmp_path, "k*clamp(4,A,1)", a0, codegen=True, t_end=1e-3)
    species = np.asarray(result.species)[:, 0]
    times = np.asarray(result.time)
    slope = (species[1] - species[0]) / (times[1] - times[0])
    assert slope == pytest.approx(-0.1 * expect * a0, rel=1e-3)


# ── The sensitivity run, which is where the issue was reported ───────────────


@pytest.mark.parametrize(
    "body",
    ["k*sign(A+1)", "k*sgn(A+1)", "k*clamp(1,A,4)", "k*avg(A,2)", "k*sum(A,2)"],
)
def test_a_sensitivity_run_builds_and_falls_back(tmp_path, body):
    """The issue's own case, plus the fallback it is supposed to reach.

    None of these functions has a derivative the emitter can write, so the
    analytic sensitivity right-hand side is declined and CVODES' own difference
    quotient is used. That was always the intended answer. What went wrong was
    that the rate law itself failed to compile, so the run never got as far as
    the fallback. Both halves are checked: the run finishes, and it finished by
    declining rather than by emitting some derivative of a step function.
    """
    m = _model(tmp_path, body)
    with contextlib.redirect_stderr(io.StringIO()):
        result = bngsim.Simulator(m, method="ode", sensitivity_params=["k"]).run(
            t_span=(0.0, 4.0), n_points=9
        )
    sens = np.asarray(result.sensitivities)[:, 0, 0]
    assert np.all(np.isfinite(sens))
    assert np.any(sens != 0.0)
    assert m._codegen_sens_decline is not None
