"""GH #579 — the three expression backends must agree, or decline out loud.

Six places in the tree parse or translate model expressions, each with its own
grammar, and nothing asserted that they agree. Six issues came out of that gap
one at a time, each found by a user at runtime on a model that had worked the
day before: ``^`` handled three different ways (#554, #555, #573), an n-ary
``max`` that would not compile (#556), ``jnp.jnp.log`` (#564), a missing
``log10`` mapping (#565).

This is the differential test #579 asks for. Three of those grammars translate
the same rate-law text — the ExprTk evaluator the ODE and SSA engines run, the
C the codegen backend emits, and the JAX the ``jacobian="jax"`` path emits —
so each expression below is driven through all three at the same state and the
same time, and their answers are compared.

The comparison is made where the engines make it: at a derivative. Each
expression becomes the functional rate law of a synthetic ``0 -> PROBE``
reaction, so ``dPROBE/dt`` *is* the expression, evaluated by the same
``compute_derivs`` CVODE calls. The compiled-C and JAX legs translate that same
rate law and evaluate it against the same numbers.

Two things are asserted, and the second matters as much as the first:

* where all three backends accept a form, their values agree;
* where one cannot, it says so while the expression is still in hand, naming
  what it could not translate — rather than emitting C that will not compile or
  Python that will raise somewhere inside a solve.

Each leg is gated on its own: the C leg needs a compiler, the JAX leg needs
JAX, and either runs without the other.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder
from bngsim._codegen import _CODEGEN_PRELUDE_LINES, _expr_to_c
from bngsim._jax_rhs import _translate_expr_jax, jax_available

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH for the codegen leg")
needs_jax = pytest.mark.skipif(not jax_available(), reason="JAX not installed")

# The shared alphabet: every backend binds parameters, observables and the
# clock, so an expression written in those three is one all of them can read.
PARAMS: dict[str, float] = {"k": 3.0, "n": 2.0, "z": 0.5, "w": 1.5, "q": 4.0}

# Two probe points, chosen so every comparison in the corpus flips between
# them. A single point lets a mis-grouped conjunction agree by luck.
POINTS: tuple[tuple[dict[str, float], float], ...] = (
    ({"A": 2.5, "B": 0.75}, 1.25),
    ({"A": 0.4, "B": 3.2}, 0.0),
)

# ── The corpus ───────────────────────────────────────────────────────────────
#
# Every form the engine's own reserved_names() advertises that all three
# backends claim to handle, plus the shapes the six issues came in through.
# Domain-safe at both probe points: nothing here is NaN on one leg and NaN on
# another, which would compare equal to nothing.

CORPUS: tuple[str, ...] = (
    # Names, the clock, and plain arithmetic.
    "k",
    "A",
    "time()",
    "A()",
    "k*A + z",
    "1/2",
    "k/n/z",
    "1e-3*k",
    "1.5e3",
    "1e+3*z",
    "-k",
    "k*-n",
    # Powers — three of the six issues live here (#554, #555, #573).
    "A^n",
    "A^2*B^3",
    "2^3^2",
    "k^z^n",
    "k^n^z^q",
    "A^B^2",
    "A^-n",
    # #573, fixed by #680: a sign run or a plus ahead of the exponent.
    "A^+n",
    "A^--n",
    "k^-(z+n)",
    "-k^n",
    "-A^-n",
    "(A+B)^2",
    "A^(1/2)",
    "2^-3",
    "A^0",
    "(A+1)^n",
    "k^time()",
    "time()^2",
    # Variadic min/max — #556 emitted fmax(a,b,c), which does not compile.
    "max(k,n,z)",
    "min(k,n,z,q)",
    "max(A,B)",
    "min(A,B)",
    "max(A,B)^2",
    "max(k,min(n,z))",
    # The transcendental table — #564 and #565 are both in here.
    "ln(k)",
    "log(k)",
    "log10(q)",
    "log2(q)",
    "log(q)/log(n)",
    "exp(z)",
    "exp(-k*time())",
    "exp(A)^2",
    "sqrt(q)",
    "sqrt(A^2+B^2)",
    "abs(-k)",
    "abs(A-B)^z",
    "sin(z)+cos(z)+tan(z)",
    "sinh(z)+cosh(z)+tanh(z)",
    "asin(z)+acos(z)+atan(z)",
    "asinh(z)+atanh(z)",
    "acosh(k)",
    # Rounding and sign — the shapes whose half-way and zero cases differ
    # between C, ExprTk and numpy if nobody checks.
    "floor(k/n)",
    "ceil(z)",
    "trunc(k/n)",
    "round(z)",
    "round(1.5)",
    "round(2.5)",
    "round(-1.5)",
    "round(-z)",
    "rint(z)",
    "rint(-1.5)",
    "rint(-z)",
    "floor(-z)",
    "ceil(-z)",
    "sign(-k)",
    "sign(0)",
    "sgn(z)",
    # Branches and the conjunctions that gate them.
    "if(A>B, k, n)",
    "if(A>B, k, n)^2",
    "if(A>B, if(k>n, k, n), z)",
    "if(A>B, 1, 0) + if(B>A, 1, 0)",
    "if(A>=B, 1, 0)",
    "if(A<=B, 1, 0)",
    "if(A==B, 1, 0)",
    "if(A!=B, 1, 0)",
    "if(A>B && k>n, k, n)",
    "if(A>B || k<n, 10, 20)",
    "if(A>2 && 4<B, 10, 20)",
    "if(A>B && k>n && z<q, 1, 0)",
    "if(A>B || k>n && z<q, 1, 0)",
    "if(A>B && k>n || z>q, 1, 0)",
    "if(A>B && (k>n || z>q), 1, 0)",
    "if((A>B) && (k>n), 10, 20)",
    "if(A>B and k>n, 1, 0)",
    "if(A>B or k>n, 1, 0)",
    "if(time()>1 && A>1, 1, 0)",
    "k*if(A>B && n<q, 2, 3)",
    "max(if(A>B && k>n,1,0), 0)",
    # `and` binds tighter than `or` in ExprTk (`1 or 0 and 0` is 1), and any
    # nonzero number is true (`2 and 3` is 1). `not` is ExprTk's only negation
    # and must be written as a call. Python's and/or/not fail on these under
    # jax.jit, which is where the JAX leg evaluates them.
    "if(A>B or k>n and z>q, 1, 0)",
    "if(A>B || k>n && z>q, 1, 0)",
    "if(A && B, 1, 0)",
    "if(A>B && z, 1, 0)",
    "if(not(A>B), 1, 0)",
    "if(not(A>B) || k<n, 1, 0)",
    # A comparison or logical result is a double, 1 or 0, wherever arithmetic
    # reads it (issue #1495). C computes each as an int, so a quotient of two
    # divided as ints, 1/2 = 0; JAX adds two booleans as a logical or.
    "((A>B)+(k>n))/((A>0)+(B>0))",
    "(A>B)/((B>0)+(A>0))",
    "(A>B)/(k>n)*z",
    "(A>B && k>n)/((A>0)+(B>0))",
    "((A>B) || (B>A))/((k>n)+(q>n))",
    "((A>B) & (k>n))/((A>0)+(B>0))",
    "((A>B) | (k<n))/((A>0)+(B>0))",
    "(A>B and k>n)/((A>0) or (B>0))",
    "not(A>B)/((A>0)+(B>0))",
    # -(B>A) is -0.0 where B>A is false, so atan(1/-0.0) is -pi/2; a C int
    # lost the sign.
    "atan(1/-(B>A))",
    # `not` is a call, so not(k)^n is (not(k))^n (issue #1506). The two
    # readings differ at an exponent <= 0, and where k^n underflows.
    "not(k-3)^(n-2)",
    "not(k-3)^-n",
    "not (k-3)^-n",
    "not(not(k))^(n-2)",
    "z*not(k-3)^-n",
    "not(k*1e-201)^n",
    "2^not(k)",
    "A^not(k-3)",
    "if(not(k-3)^(n-2) > 0.5, 1, 2)",
    # Constants.
    "_pi*_e",
    "_pi*A",
    # Saturating shapes, which is most of what real rate laws are.
    "k*A^2/(n+A^2)",
    "A/(k+A)",
    "k*A*B/((n+A)*(w+B))",
)

# Forms the engine accepts that the JAX translator has no jnp spelling for. The
# contract is not that it handle them — it is that it say so, by name, while
# the expression is in hand.
JAX_DECLINES: tuple[str, ...] = (
    "clamp(z,k,q)",
    "avg(k,n,z)",
    "sum(k,n,z)",
    "erf(z)",
    "erfc(z)",
    "tgamma(k)",
    "mratio(k,n,z)",
)


# ── The three legs ───────────────────────────────────────────────────────────


def _interpreter(expr: str, obs: dict[str, float], t: float) -> float:
    """dPROBE/dt for ``0 -> PROBE`` at rate ``expr``, from the engine's own
    ``compute_derivs`` — the derivative CVODE and the SSA both read."""
    builder = ModelBuilder()
    for name, value in PARAMS.items():
        builder.add_parameter(name, value)
    for idx, (name, value) in enumerate(obs.items()):
        builder.add_species(f"S_{name}", value)
        builder.add_observable(name, [(idx, 1.0)])
    probe = builder.add_species("PROBE", 0.0)
    builder.add_function("probe_rate", expr)
    builder.add_reaction([], [probe], "functional", "probe_rate")
    model = builder.build()
    state = np.array([*obs.values(), 0.0], dtype=np.float64)
    return float(model.compute_derivs(t, state)[probe])


_C_PROGRAM = """#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>
{prelude}
int main(void) {{
    double p[] = {{{params}}};
    double obs[] = {{{observables}}};
    double t = {t!r};
    (void)p; (void)obs; (void)t;
    printf("%.17g\\n", (double)({body}));
    return 0;
}}
"""


def _compiled_c(expr: str, obs: dict[str, float], t: float) -> float:
    """The same expression through the codegen translator, compiled by the same
    compiler the backend uses and run.

    The emitted text is dropped into the codegen prelude verbatim, so the
    helpers it may call (``bngsim_mratio`` and friends) are the shipped ones
    rather than a test's idea of them.
    """
    body = _expr_to_c(expr, list(PARAMS), [], list(obs), [])
    source = _C_PROGRAM.format(
        prelude="\n".join(_CODEGEN_PRELUDE_LINES),
        params=",".join(repr(v) for v in PARAMS.values()),
        observables=",".join(repr(v) for v in obs.values()),
        t=t,
        body=body,
    )
    with tempfile.TemporaryDirectory() as tmp:
        c_path = Path(tmp) / "probe.c"
        exe = Path(tmp) / "probe"
        c_path.write_text(source)
        assert _CC is not None
        built = subprocess.run(
            [_CC, "-O0", str(c_path), "-o", str(exe), "-lm"],
            capture_output=True,
            text=True,
        )
        assert built.returncode == 0, (
            f"the codegen backend emitted C that does not compile\n"
            f"  expression: {expr}\n"
            f"  emitted:    {body}\n"
            f"  {built.stderr.strip().splitlines()[0] if built.stderr.strip() else ''}"
        )
        return float(subprocess.run([str(exe)], capture_output=True, text=True).stdout)


def _jax(expr: str, obs: dict[str, float], t: float) -> float:
    """The same expression through the JAX translator, evaluated in the
    namespace the generated RHS evaluates it in — under ``jax.jit``, because
    that is how the Jacobian is built (``jax.jit(jax.jacfwd(rhs))``). An
    untraced eval hands Python's ``and``/``or``/``not`` concrete values and
    lets them through; a traced one asks a tracer for ``bool()`` and fails,
    as the solve does."""
    jax_available()  # also switches JAX to float64, as the RHS does
    import jax
    import jax.numpy as jnp
    from bngsim._jax_rhs import _JAX_HELPERS

    code = _translate_expr_jax(
        expr,
        {name: i for i, name in enumerate(PARAMS)},
        {name: i for i, name in enumerate(obs)},
        set(),
        [],
    )

    @jax.jit
    def evaluate(obs_arr, t_arr):
        namespace = {
            "jnp": jnp,
            "params": jnp.array(list(PARAMS.values()), dtype=jnp.float64),
            "obs": obs_arr,
            "t": t_arr,
            **_JAX_HELPERS,
        }
        return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307

    return float(evaluate(jnp.array(list(obs.values()), dtype=jnp.float64), jnp.float64(t)))


# ── The differential ─────────────────────────────────────────────────────────


@needs_cc
@pytest.mark.parametrize("expr", CORPUS)
def test_the_interpreter_and_the_compiled_c_agree(expr: str):
    """What the ODE engine computes and what the codegen backend computes, at
    the same state and time. #555, #556 and #573 are all failures of this."""
    for obs, t in POINTS:
        reference = _interpreter(expr, obs, t)
        assert _compiled_c(expr, obs, t) == pytest.approx(reference, rel=1e-12, abs=1e-15), (
            f"{expr!r} at {obs} t={t}: the compiled C disagrees with the interpreter"
        )


@needs_jax
@pytest.mark.parametrize("expr", CORPUS)
def test_the_interpreter_and_the_jax_translation_agree(expr: str):
    """The same, for the translation ``jacobian='jax'`` differentiates. #564 and
    #565 are failures of this, and so is the conjunction grouping below."""
    for obs, t in POINTS:
        reference = _interpreter(expr, obs, t)
        assert _jax(expr, obs, t) == pytest.approx(reference, rel=1e-12, abs=1e-15), (
            f"{expr!r} at {obs} t={t}: the JAX translation disagrees with the interpreter"
        )


@pytest.mark.parametrize("expr", CORPUS)
def test_every_expression_in_the_corpus_is_one_the_engine_accepts(expr: str):
    """The corpus is only a reference if ExprTk compiles all of it — an
    expression the engine itself rejects would let the other two legs agree on
    a form no model can contain."""
    value = _interpreter(expr, *POINTS[0])
    assert math.isfinite(value), f"{expr!r} is not finite at the first probe point"


# ── Declining, rather than emitting something broken ─────────────────────────


@needs_jax
@pytest.mark.parametrize("expr", JAX_DECLINES)
def test_a_form_jax_cannot_translate_is_refused_by_name(expr: str):
    """A backend that cannot handle a model must decline explicitly. These
    seven have no jnp spelling; the translator has to say which, here, and not
    hand the solve a bare name to fail on (#565)."""
    with pytest.raises(ValueError) as caught:
        _jax(expr, POINTS[0][0], POINTS[0][1])
    name = expr.split("(")[0]
    assert name in str(caught.value), f"the refusal does not name {name!r}: {caught.value}"


@needs_cc
@pytest.mark.parametrize("expr", JAX_DECLINES)
def test_the_forms_jax_declines_are_ones_the_other_two_do_handle(expr: str):
    """Declining is only the right answer for a form the backend genuinely
    cannot express — not a licence to drop coverage. Each of these is live in
    the interpreter and in the compiled C."""
    for obs, t in POINTS:
        assert _compiled_c(expr, obs, t) == pytest.approx(
            _interpreter(expr, obs, t), rel=1e-12, abs=1e-15
        )


# ── The divergence this suite found ─────────────────────────────────────────────


@needs_jax
def test_a_conjunction_of_comparisons_keeps_the_source_grouping():
    """The seventh divergence, found by the corpus above on its first run.

    ``&&`` was rewritten to ``&`` as text. In ExprTk and in C ``&&`` binds
    *looser* than the comparisons it joins; in Python ``&`` binds *tighter*.
    So ``A > 2 && 4 < B`` became ``A > (2 & 4) < B`` — ``2 & 4`` is ``0``, and
    what was left was a Python comparison chain that evaluated happily and
    returned the wrong branch. On float operands the same shape raised
    ``TypeError: and does not accept dtype float64`` inside the solve instead.
    """
    obs = {"A": 2.5, "B": 0.75}
    # A > 2 is true, 4 < B is false: the conjunction is false, so 20.
    assert _interpreter("if(A>2 && 4<B, 10, 20)", obs, 0.0) == 20.0
    assert _jax("if(A>2 && 4<B, 10, 20)", obs, 0.0) == 20.0


@needs_jax
def test_a_comparison_run_is_read_left_to_right():
    """The same inversion in its other half: ``a < b < c`` is ``(a < b) < c``
    for ExprTk and for C, and a chain for Python."""
    obs = {"A": 2.5, "B": 0.75}
    # (A < B) is 0, and 0 < k, so 1 — where Python's chain would say 0.
    assert _interpreter("if(A<B<k, 1, 0)", obs, 0.0) == 1.0
    assert _jax("if(A<B<k, 1, 0)", obs, 0.0) == 1.0


def _is_plain(expr: str) -> bool:
    """No conjunction and at most one comparison — the shape the regrouping
    below promises not to touch."""
    from bngsim._jax_rhs import _LOGICAL_OPS

    return not any(op in expr for op in _LOGICAL_OPS) and (
        sum(expr.count(op) for op in ("<", ">", "==", "!=")) <= 1
    )


@pytest.mark.parametrize("expr", [e for e in CORPUS if _is_plain(e)])
def test_regrouping_leaves_an_expression_without_conjunctions_alone(expr: str):
    """The regrouping is a no-op on anything with no conjunction and at most
    one comparison, which is nearly every rate law ever written — 186,991 of
    the 186,993 expressions in the tracked and benchmark ``.net`` corpora come
    through byte-for-byte."""
    from bngsim._jax_rhs import _regroup_for_python

    assert _regroup_for_python(expr) == expr


@needs_jax
def test_a_gated_rate_law_reaches_the_jax_rhs_intact(tmp_path):
    """End to end, past the translator: a model whose rate is gated on a
    conjunction, run through the generated JAX RHS and the interpreter at three
    states — one inside the gate and two outside it, so a gate stuck open or
    shut cannot pass."""
    import bngsim
    import jax.numpy as jnp
    from bngsim._jax_rhs import generate_jax_rhs

    net = tmp_path / "gated.net"
    net.write_text(
        "begin parameters\n"
        "    1 k1  2\n"
        "    2 thr 1\n"
        "end parameters\n"
        "begin functions\n"
        "    1 gate() if(A_tot>thr && B_tot<thr, k1, 0)\n"
        "end functions\n"
        "begin species\n"
        "    1 A() 3\n"
        "    2 B() 0\n"
        "end species\n"
        "begin reactions\n"
        "    1 1 1,2 gate #Rule1\n"
        "end reactions\n"
        "begin groups\n"
        "    1 A_tot  1\n"
        "    2 B_tot  2\n"
        "end groups\n"
    )
    model = bngsim.Model.from_net(str(net))
    rhs = generate_jax_rhs(str(net))
    params = jnp.array([2.0, 1.0], dtype=jnp.float64)

    open_gate = np.array([3.0, 0.0])
    assert model.rhs(open_gate, 0.0)[1] != 0.0, "the open state must exercise the true branch"
    for state in (open_gate, np.array([0.5, 0.0]), np.array([3.0, 2.0])):
        expected = np.asarray(model.rhs(state, 0.0))
        got = np.asarray(rhs(jnp.array(state), 0.0, params))
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15), f"at {state}"
