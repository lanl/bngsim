"""Issue #457 — mratio has a derivative, so a fit over a rate constant can use
an analytic gradient.

A model calling ``mratio`` in a rate law used to lose its whole analytic
sensitivity right-hand side. The differentiation layer had never heard of the
function, so every derivative through it came back unevaluated and the run fell
back to CVODES' internal difference quotient. That is correct but slower, and
here it was also unnecessary.

``mratio(a, b, z)`` is ``M(a+1,b+1,z) / M(a,b,z)``, and Kummer's identity is
``dM(a,b,z)/dz = (a/b)*M(a+1,b+1,z)``. Putting that through the quotient rule
gives, writing ``R`` for mratio,

    dR/dz = R(a,b,z) * [ (a+1)/(b+1)*R(a+1,b+1,z) - (a/b)*R(a,b,z) ]

so the derivative in the third argument is mratio again. No new special function
and no new numerics.

The first two arguments get no derivative, because there is no comparable closed
form for them. That is not much of a loss: BNG builds ``a`` and ``b`` from
molecule counts and puts the rate constant in ``z``, so fitting a rate constant
moves ``z`` and nothing else. A model that does differentiate through ``a`` or
``b`` declines, which is what ``test_the_first_argument_is_declined`` holds.

The one awkward part is the second call. Issue #453 gave mratio a region it
trusts and made it refuse outside it, and ``(a+1, b+1, z)`` is not automatically
inside. Where it is refused, the emitted helper falls back to a second exact
expression for the same derivative, from the contiguous relation Kummer's
equation gives:

    (a+1)/(b+1)*R(a+1,b+1,z)*R(a,b,z) = ( b - (b-z)*R(a,b,z) ) / z

That form is second choice because the subtraction cancels, badly for a small
``|z|``. A small ``|z|`` is exactly where mratio trusts the shifted call
unconditionally, so the fallback only ever runs where it is accurate. Without
it, a model that ran before this change would fail after it, because the
compiled path has no retry: a NaN out of the derivative ends the run.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _CODEGEN_PRELUDE_LINES, generate_sens_from_model

# One species draining at a rate the parameter under study controls only through
# mratio. ``k`` scales it, so the sensitivity is not degenerate.
NET = """begin parameters
    1 Keq {keq:.17g}  # Constant
    2 k   0.05  # Constant
end parameters
begin functions
    1 law() k*mratio({a},{b},{z})
end functions
begin species
    1 A() 10.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
end groups
"""


def _model(tmp_path, keq, *, a="-3", b="5", z="-1/Keq", tag=""):
    p = tmp_path / f"m{tag}.net"
    p.write_text(NET.format(keq=keq, a=a, b=b, z=z))
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(str(p))


def _species(tmp_path, keq, **kw):
    """The trajectory at one parameter value, through the ordinary loader."""
    model = _model(tmp_path, keq, tag=f"_{keq!r}", **kw)
    with contextlib.redirect_stderr(io.StringIO()):
        r = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, 5.0), n_points=6, rtol=1e-12, atol=1e-14
        )
    return np.asarray(r.species)[:, 0]


def _sensitivity(tmp_path, keq, param="Keq", **kw):
    """``(analytic dA/dparam, the Simulator)`` at one parameter value."""
    model = _model(tmp_path, keq, tag="_sens", **kw)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param])
    with contextlib.redirect_stderr(io.StringIO()):
        r = sim.run(t_span=(0.0, 5.0), n_points=6, rtol=1e-12, atol=1e-14)
    return np.asarray(r.sensitivities)[:, 0, 0], sim


def _mratio(tmp_path, a, b, z):
    """mratio(a, b, z) straight out of the interpreter, at one point."""
    model = _model(tmp_path, 1.0, a=repr(float(a)), b=repr(float(b)), z=repr(float(z)), tag="_v")
    core = model._core if hasattr(model, "_core") else model
    return core._eval_functions(0.0, [10.0])["law"] / 0.05


# ── The rate law is differentiated now ───────────────────────────────────────


def test_the_rate_law_is_differentiated(tmp_path):
    """The model the issue was reported with keeps its analytic right-hand side."""
    _, sim = _sensitivity(tmp_path, 1e-3)
    assert sim.has_analytic_sens_rhs
    assert sim.sens_rhs_decline_reason is None


def test_the_sensitivity_matches_a_finite_difference(tmp_path):
    """Against a difference taken by editing the model text and reloading.

    The only reference worth having here: it goes through the same loader, the
    same rate-law parser and the same integrator the run does, so nothing about
    the answer is shared between the two sides except the model file.

    The step is swept because a single step proves nothing on its own — one that
    is too large measures curvature and one that is too small measures the
    solver's own noise. 1e-3 of the parameter is the bottom of that V here.
    """
    keq = 1e-3
    analytic, _ = _sensitivity(tmp_path, keq)
    h = keq * 1e-3
    fd = (_species(tmp_path, keq + h) - _species(tmp_path, keq - h)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-9)


def test_the_analytic_answer_is_the_one_the_difference_quotient_gave(tmp_path, monkeypatch):
    """Same numbers as before this change, only arrived at differently.

    ``BNGSIM_NO_FUNCTIONAL_SENS_RHS=1`` is the hatch that puts a Functional
    model back on CVODES' internal difference quotient, which is the path every
    model calling mratio was on until now.
    """
    analytic, sim = _sensitivity(tmp_path, 1e-3)
    assert sim.has_analytic_sens_rhs
    monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
    quotient, dq_sim = _sensitivity(tmp_path, 1e-3)
    assert not dq_sim.has_analytic_sens_rhs
    np.testing.assert_allclose(analytic, quotient, rtol=1e-6, atol=1e-9)


# ── What reaches the generated C ─────────────────────────────────────────────


def test_the_derivative_helper_is_in_every_generated_source():
    """One definition, carried by the prelude every emitter writes out."""
    text = "\n".join(_CODEGEN_PRELUDE_LINES)
    assert "static double bngsim_mratio_dz(double a, double b, double z)" in text
    # It calls the value helper, so it has to come after it in the same guard.
    value_at = text.index("static double bngsim_mratio(")
    assert value_at < text.index("static double bngsim_mratio_dz(")
    assert "#ifndef BNGSIM_MRATIO_DEFINED" in text


def test_the_sensitivity_source_calls_the_helper(tmp_path):
    """The emitted ``df/dp`` is one helper call, not the identity written out.

    One call rather than three, and one place for the fallback below to live.
    """
    model = _model(tmp_path, 1e-3, tag="_src")
    src = generate_sens_from_model(model, functional=True)
    assert src is not None
    calls = [
        line
        for line in src.splitlines()
        if "bngsim_mratio_dz(" in line and "static double" not in line
    ]
    assert calls, "the sensitivity right-hand side does not call the derivative helper"


# ── What is still refused, and why ───────────────────────────────────────────


def test_the_first_argument_is_declined(tmp_path):
    """No closed form for d/da, so the model keeps the difference quotient.

    The failure has to be a decline and not a wrong number: a missing partial
    reads downstream as an exact zero.
    """
    # ``aa`` is not in the template's parameter block, so write one that has it.
    p = tmp_path / "m_a.net"
    p.write_text(
        NET.format(keq=1e-3, a="aa", b="5", z="-1000").replace(
            "    2 k   0.05  # Constant", "    2 k   0.05  # Constant\n    3 aa  -3.0  # Constant"
        )
    )
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["aa"])
    with contextlib.redirect_stderr(io.StringIO()):
        r = sim.run(t_span=(0.0, 5.0), n_points=4)
    assert not sim.has_analytic_sens_rhs
    assert "aa" in sim.sens_rhs_decline_reason
    # The run still answers, by the route it was always on.
    assert np.isfinite(np.asarray(r.sensitivities)).all()


def test_the_exprtk_emitter_has_no_spelling_for_the_derivative():
    """So the interpreted Jacobian declines rather than emitting a call ExprTk
    cannot compile.

    The engine has ``mratio`` but nothing for its derivative, and writing the
    identity out for it instead would hand the interpreted evaluator an
    expression that throws exactly where the compiled helper falls back. The
    interpreted path answers with the finite-difference Jacobian instead, which
    is what it did before this change.
    """
    sympy = pytest.importorskip("sympy")
    from bngsim._jacobian import _exprtk_to_sympy, sympy_to_c, sympy_to_exprtk

    deriv = sympy.diff(_exprtk_to_sympy("mratio(-3,5,x)"), sympy.Symbol("x"))
    assert sympy_to_exprtk(deriv) is None
    assert sympy_to_c(deriv, lambda name: name) == "bngsim_mratio_dz(-3.0, 5.0, x)"


def test_a_second_derivative_is_declined():
    """Nothing here knows d2R/dz2, so sympy leaves it unevaluated and both
    emitters refuse it."""
    sympy = pytest.importorskip("sympy")
    from bngsim._jacobian import _exprtk_to_sympy, _is_emittable

    second = sympy.diff(_exprtk_to_sympy("mratio(-3,5,x)"), sympy.Symbol("x"), 2)
    assert second.has(sympy.Derivative)
    assert not _is_emittable(second)


# ── The identity itself ──────────────────────────────────────────────────────

# (a, b, z). The last is the one scipy.special.hyp1f1 cannot be used on at all:
# M there is about 1.3e318, so it overflows to nan and the ratio cannot be
# formed, while mratio answers because the continued fraction never builds M.
IDENTITY_ARGS = [
    (-3.0, 5.0, -1.0),
    (-3.0, 5.0, -1000.0),
    (-10.0, 20.0, -50.0),
    (-2.5, 7.0, -3.0),
    (-4.0, 9.0, 2.0),
    (-1000.0, 9001.0, -10000.0),
]


@pytest.mark.parametrize("a, b, z", IDENTITY_ARGS)
def test_the_closed_form_is_the_derivative(tmp_path, a, b, z):
    """The formula against a central difference of the engine's own mratio."""
    r = _mratio(tmp_path, a, b, z)
    closed = r * ((a + 1) / (b + 1) * _mratio(tmp_path, a + 1, b + 1, z) - (a / b) * r)
    h = abs(z) * 1e-5
    fd = (_mratio(tmp_path, a, b, z + h) - _mratio(tmp_path, a, b, z - h)) / (2 * h)
    assert closed == pytest.approx(fd, rel=1e-8)


@pytest.mark.parametrize("a, b, z", IDENTITY_ARGS)
def test_the_fallback_expression_agrees_with_the_first_one(tmp_path, a, b, z):
    """The contiguous relation gives the same derivative from one mratio call.

    Loosely, because it is the less accurate of the two — that is why it is the
    fallback and not the first choice.
    """
    r = _mratio(tmp_path, a, b, z)
    closed = r * ((a + 1) / (b + 1) * _mratio(tmp_path, a + 1, b + 1, z) - (a / b) * r)
    contiguous = (b - (b - z) * r) / z - (a / b) * r * r
    assert contiguous == pytest.approx(closed, rel=1e-6)


# ── The corner where the second call is refused ──────────────────────────────


def test_a_refused_second_call_does_not_end_the_run(tmp_path):
    """``a`` above -1 puts ``a+1`` where mratio will not compute.

    mratio answers for ``a = -0.5, z = -1000`` and refuses for ``a = 0.5``, so
    the derivative's shifted call is refused while the value's is not. Before
    the fallback this returned NaN, the compiled sensitivity right-hand side
    failed on its first call, and a model that had run happily on the difference
    quotient stopped running at all.

    No BNG-generated model is here: BNG builds ``a = -min(AT,BT)``, which is at
    or below -1.
    """
    z = -1000.0
    p = tmp_path / "m_refused.net"
    p.write_text(
        NET.format(keq=1e-3, a="-0.5", b="2.5", z="zz").replace(
            "    2 k   0.05  # Constant",
            f"    2 k   0.05  # Constant\n    3 zz  {z:.17g}  # Constant",
        )
    )
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["zz"])
    with contextlib.redirect_stderr(io.StringIO()):
        r = sim.run(t_span=(0.0, 5.0), n_points=6, rtol=1e-12, atol=1e-14)
    assert sim.has_analytic_sens_rhs
    analytic = np.asarray(r.sensitivities)[:, 0, 0]
    assert np.isfinite(analytic).all()

    # And it is the right derivative, not merely a finite one.
    def species(zz):
        q = tmp_path / f"m_refused_{zz!r}.net"
        q.write_text(
            NET.format(keq=1e-3, a="-0.5", b="2.5", z="zz").replace(
                "    2 k   0.05  # Constant",
                f"    2 k   0.05  # Constant\n    3 zz  {zz:.17g}  # Constant",
            )
        )
        with contextlib.redirect_stderr(io.StringIO()):
            m = bngsim.Model.from_net(str(q))
            out = bngsim.Simulator(m, method="ode").run(
                t_span=(0.0, 5.0), n_points=6, rtol=1e-12, atol=1e-14
            )
        return np.asarray(out.species)[:, 0]

    h = abs(z) * 1e-4
    fd = (species(z + h) - species(z - h)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-12)
