"""Issue #541: a seasonal pulse written as ``if(c, u, 0)^(a-1)`` keeps its
analytical Jacobian and its analytic sensitivity RHS.

``SIR_v5`` shapes its transmission rate with a Kumaraswamy pulse: ``u()`` maps the
season onto [0, 1] and is 0 outside it, the pulse is ``u()^(a()-1)·(1-u()^a())^(b-1)``,
and ``a()`` selects one shape parameter per year on the day counter ``t``. Three
things stood between that model and its derivatives, each of which the fixture
below reproduces in miniature.

**sympy's power rule at a zero base.** sympy distributes a power over the branches
of a Piecewise, so off-season the pulse holds ``0^(a()-1)`` and ``0^a()``. The rule
``d(b^e) = b^e·(e'·log(b) + b'·e/b)`` builds ``log(0)`` and ``e/0`` there and folds
them to ``nan``, although ``0^e`` is a step in ``e`` and constant on each side. The
exponent reads ``t`` (through the year selection) and every ``a_20xx``, so ``∂/∂t``
declined the Jacobian and ``∂/∂a_2021`` the sensitivity RHS.

**An ITE the zero-base logarithm guard built.** The in-season ``∂/∂a`` carries
``u^(a-1)·log(u)``, which GH #310's guard rewrites to its limit at ``u = 0``
behind ``Eq(u, 0) & (a - 1 > 0)``. With ``a`` a Piecewise sympy folds that
condition into an ``ITE``, and nothing normalized it after the guard ran: the C
compiler refused the call and a forward-sensitivity run raised.

**0/0 at the season onset.** The in-season ``∂/∂t`` is
``(a-1)·((t - t_start)·r)^(a-1)/(t - t_start)``: removable, but GH #96's rewrite
cancelled it only for a bare-symbol denominator, so at ``t == t_start`` it was
NaN — and a run whose counter started on an onset stopped at the first
sensitivity RHS call, where CVODES' difference quotient had run to the end.

``SIR_v4``, the issue's other model, divides by its shape parameters after its last
modelled year (``if(t<=1461, a_2024, 0)``), so its rate law is non-finite there
before anything is differentiated. It still declines; what changed is that both
reasons now say where the rate law itself is non-finite.
"""

from __future__ import annotations

import glob

import bngsim
import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from bngsim import _jacobian as J  # noqa: E402

# A two-year SIR_v5 in miniature. `C()` is the day counter (made at rate 1) and `t`
# its observable; `a()` selects the pulse shape per year, and `u()` is the season.
# Both shape exponents are at least 2, so the pulse's slope at an onset is 0 from
# either side and a central difference straddling the onset converges to it.
PULSE_NET = """\
begin parameters
    1 T         10  # Constant
    2 ton1      3  # Constant
    3 dur1      4  # Constant
    4 ton2      12  # Constant
    5 dur2      5  # Constant
    6 a1        3  # Constant
    7 a2        4  # Constant
    8 b         2  # Constant
    9 bmin      0.1  # Constant
   10 bmax      2.0  # Constant
   11 gamma     0.5  # Constant
   12 _rateLaw1 1  # Constant
end parameters
begin functions
    1 t_start() if((t<T),ton1,ton2)
    2 duration() if((t<T),dur1,dur2)
    3 u() if(((t>=t_start())&&(t<=(t_start()+duration()))),((t-t_start())/duration()),0)
    4 a() if((t<T),a1,a2)
    5 f() (u()^(a()-1))*((1-(u()^a()))^(b-1))
    6 beta() bmin+((bmax-bmin)*f())
    7 _rateLaw2() beta()*I
end functions
begin species
    1 S() 0.9
    2 I() 0.1
    3 C() {c0}
end species
begin reactions
    1 0 3 _rateLaw1 #clock
    2 1 2 _rateLaw2 #infect
    3 2 0 gamma #recover
end reactions
begin groups
    1 S 1
    2 I 2
    3 t 3
end groups
"""

# A Hill exponent chosen by an if() on the clock, under a logarithm: the smallest
# law whose zero-base guard condition sympy folds into an ITE.
ITE_NET = """\
begin parameters
    1 k   0.5  # Constant
    2 n1  2.0  # Constant
    3 n2  3.0  # Constant
    4 kd  0.1  # Constant
end parameters
begin functions
    1 grow() k*A^if(time()<5,n1,n2)*log(A+2)
end functions
begin species
    1 S() 1.0
    2 A() 0.5
end species
begin reactions
    1 1 2 grow #grow
    2 2 0 kd #decay
end reactions
begin groups
    1 A 2
end groups
"""

TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def _between_crossings(t_end: float) -> dict:
    """Output times at the half days. Every crossing in these fixtures falls on a
    whole day, and a sample on a crossing that moves with the parameter is where a
    central difference of the trajectory is only first order — its error there
    shrinks linearly with the step while the analytic column agrees with CVODES'
    difference quotient to ~2e-6 — so a sample there would test the check."""
    return {"sample_times": [0.0] + [k + 0.5 for k in range(int(t_end))], **TIGHT}


def _net(tmp_path, text, name, **fmt):
    path = tmp_path / name
    path.write_text(text.format(**fmt) if fmt else text)
    return str(path)


def _central_jacobian(m, y, h=1e-6):
    n = y.size
    D = np.zeros((n, n))
    for j in range(n):
        s = h * max(abs(y[j]), 1.0)
        yp, ym = y.copy(), y.copy()
        yp[j] += s
        ym[j] -= s
        D[:, j] = (np.asarray(m.rhs(yp)) - np.asarray(m.rhs(ym))) / (2.0 * s)
    return D


def _trajectory_fd(path, param, kw, rel=1e-3):
    """Central difference of the species trajectory with respect to ``param``.

    A wide step at a tighter tolerance than the run it checks: at a 1e-5 step and
    rtol 1e-10 the quotient's own error on this fixture is ~2e-4, while at these
    settings it agrees with CVODES' difference quotient to ~2e-6."""
    v = bngsim.Model.from_net(path).get_param(param)
    h = rel * abs(v)
    runs = []
    for sign in (1.0, -1.0):
        m = bngsim.Model.from_net(path)
        m.set_param(param, v + sign * h)
        run = {**kw, "rtol": 1e-12, "atol": 1e-14}
        runs.append(np.asarray(bngsim.Simulator(m, method="ode").run(**run).species))
    return (runs[0] - runs[1]) / (2.0 * h)


# ─── The symbolic core ───────────────────────────────────────────────────────


def test_the_issue_reproduction_differentiates_to_zero_off_season():
    """The issue's minimal reproduction, through the derivation entry point: the
    off-season branch of ``∂/∂t`` is the ``0`` it is, not ``nan``, and the
    in-season branch is still the ordinary derivative."""
    rate = "if((t>=0)&&(t<=30),t/30,0)^(if(t<1096,a3,a4)-1)"
    dd = J.differentiate_rate_law(rate, {}, {"t"}, {"a3", "a4"})
    assert dd is not None, J.last_decline_reason()
    d = dd["t"]
    assert not d.has(sp.nan, sp.zoo)
    t, a3 = sp.Symbol("t"), sp.Symbol("a3")
    assert d.subs({t: 40, a3: 3}) == 0
    in_season = float(d.subs({t: 12, a3: 3}))
    assert in_season == pytest.approx(2 * 12 / 30**2)  # d/dt (t/30)^2


def test_a_value_factor_prints_exactly_as_it_did():
    """The twin goes back to the power it stands for: a derivative that carries
    ``0^e`` as a factor — here with respect to ``k`` — prints it unchanged."""
    zeropow = J._zero_base_bindings(sp)["zeropow"]
    t, a, k = sp.symbols("t a k")
    value = k * sp.Piecewise((t, t > 1), (0, True)) ** (sp.Piecewise((a, t < 5), (2, True)) - 1)
    prepared = J._prepare_zero_bases(value, {"t", "k"}, {"a"})
    assert prepared.has(zeropow)
    assert J._finish_zero_bases(prepared) == value
    assert J._finish_zero_bases(sp.diff(prepared, k)) == sp.diff(value, k)


def test_nothing_is_rebuilt_when_nothing_qualifies():
    """The same object back, so a rate law with no zero-base power, or one whose
    exponent the differentiation never reaches, emits byte-identical text."""
    t, a, k = sp.symbols("t a k")
    plain = k * sp.exp(t) * sp.Piecewise((t, t > 1), (1, True)) ** a
    assert J._prepare_zero_bases(plain, {"t"}, {"a", "k"}) is plain
    untouched = k * t * sp.Piecewise((t, t > 1), (0, True)) ** a
    assert J._prepare_zero_bases(untouched, {"t"}, {"a", "k"}) is untouched


def test_an_exponent_that_reads_the_state_keeps_declining():
    """``0^(A-1)`` over a state is a step in that state which nothing locates, so
    a derivative of 0 through it would drop a jump the run can cross. The twin
    is reserved for exponents whose values are run-constants."""
    dd = J.differentiate_rate_law("k*if(B>1,B,0)^(A-1)", {}, {"A", "B"}, {"k"})
    assert dd is None
    assert "zoo / oo / nan" in J.last_decline_reason()


def test_a_float_zero_base_is_a_zero_base():
    """sympy >= 1.13 does not call ``Float(0.0)`` equal to ``Integer(0)``, and a
    model may write the zero branch either way."""
    dd = J.differentiate_rate_law("if(t>5,t,0.0)^(if(t<9,a1,a2)-1)", {}, {"t"}, {"a1", "a2"})
    assert dd is not None, J.last_decline_reason()
    assert not dd["t"].has(sp.nan, sp.zoo)


# ─── The fixture, end to end ─────────────────────────────────────────────────


def test_the_pulse_attaches_and_matches_finite_differences(tmp_path):
    """Off-season, in-season, and exactly on each onset — where the removable
    ``0/0`` used to sit — in both years."""
    m = bngsim.Model.from_net(_net(tmp_path, PULSE_NET, "pulse.net", c0=0))
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    for day in (1.0, 3.0, 5.5, 8.0, 12.0, 14.0, 18.5):
        y = np.array([0.6, 0.2, day])
        Jy = m.jacobian(y)
        assert Jy.source == "analytical"
        assert np.all(np.isfinite(np.asarray(Jy))), day
        np.testing.assert_allclose(
            np.asarray(Jy), _central_jacobian(m, y), rtol=1e-5, atol=1e-5, err_msg=f"t={day}"
        )


def test_the_pulse_runs_on_the_analytic_sensitivity_rhs(tmp_path):
    """Each column against a central difference of the trajectory: the exponent
    ``a1``, the onset ``ton1`` and the second season's ``dur2`` — whose crossings
    move — and ``gamma`` beside them."""
    path = _net(tmp_path, PULSE_NET, "pulse.net", c0=0)
    params = ["a1", "ton1", "dur2", "gamma"]
    kw = _between_crossings(20.0)
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=params)
    result = sim.run(**kw)
    assert sim.has_analytic_sens_rhs, sim.sens_rhs_decline_reason
    S = np.asarray(result.sensitivities)
    assert np.all(np.isfinite(S))
    for k, param in enumerate(params):
        np.testing.assert_allclose(
            S[:, :, k], _trajectory_fd(path, param, kw), rtol=1e-4, atol=1e-7, err_msg=param
        )


def test_a_run_that_starts_on_an_onset_completes(tmp_path):
    """The counter starts at ``ton1`` exactly, so CVODES evaluates the sensitivity
    RHS on the onset at its first call. Before the denominator was cancelled that
    call returned NaN and the run stopped with ``CV_FIRST_SRHSFUNC_ERR``."""
    path = _net(tmp_path, PULSE_NET, "onset.net", c0=3)
    kw = _between_crossings(5.0)
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=["ton1"])
    result = sim.run(**kw)
    assert sim.has_analytic_sens_rhs
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[:, :, 0],
        _trajectory_fd(path, "ton1", kw),
        rtol=1e-4,
        atol=1e-7,
    )


# ─── The ITE the zero-base logarithm guard builds ────────────────────────────


def test_the_guard_condition_reaches_the_emitters_without_an_ite():
    rate = "k*A^if(time()<5,n1,n2)*log(A+2)"
    terms = J.build_per_observable_terms(rate, {}, {"A"}, {"k", "n1", "n2"})
    assert terms is not None and "ITE" not in terms[0][1], terms
    value = J._exprtk_to_sympy(rate)
    deriv = sp.diff(value, sp.Symbol("n1"))
    names = {"A": "y[0]", "k": "p[0]", "n1": "p[1]", "n2": "p[2]", J._TIME_SYM: "t"}
    c = J.sympy_to_c(deriv, names.get)
    assert c is not None and "ITE" not in c, c


def test_an_if_selected_hill_exponent_under_a_logarithm_keeps_both_derivatives(tmp_path):
    """Before: the Jacobian declined because ExprTk would not compile ``ITE(``, and
    the forward-sensitivity run raised because the C compiler would not either."""
    path = _net(tmp_path, ITE_NET, "ite.net")
    m = bngsim.Model.from_net(path)
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    kw = {"t_span": (0.0, 10.0), "n_points": 11, **TIGHT}
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=["n1"])
    result = sim.run(**kw)
    assert sim.has_analytic_sens_rhs, sim.sens_rhs_decline_reason
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[:, :, 0],
        _trajectory_fd(path, "n1", kw),
        rtol=1e-4,
        atol=1e-8,
    )


# ─── A division by zero folded into a rewrite's condition ────────────────────
#
# SIR_v4's season window is bounded by if() chains that end in 0 after its last
# modelled year, so on that branch the pulse's base is (t - 0)/(0 - 0). The model
# never evaluates it there — the window's own condition is false — and sympy does
# not fold it while parsing. But once the zero-base power differentiates, the
# in-season ∂/∂a carries u^(a-1)·log(u), GH #310's guard builds Eq(u, 0), and
# sympy folds that condition branch by branch into zoo*t. The emitters checked for
# non-finite atoms only before their rewrites, so the zoo reached the printer and
# the forward-sensitivity build failed to compile.

WINDOW_BOUNDS_END_IN_ZERO = PULSE_NET.replace(
    "    1 t_start() if((t<T),ton1,ton2)\n    2 duration() if((t<T),dur1,dur2)\n"
    "    3 u() if(((t>=t_start())&&(t<=(t_start()+duration()))),"
    "((t-t_start())/duration()),0)\n",
    "    1 t_start() if((t<T),ton1,0)\n    2 t_end() if((t<T),(ton1+dur1),0)\n"
    "    3 u() if(((t>=t_start())&&(t<=t_end())),((t-t_start())/(t_end()-t_start())),0)\n",
)


def test_a_zero_folded_into_a_rewrite_condition_is_refused_not_printed():
    rate = (
        "k*I*if((t>=if(t<T,ton1,0))&&(t<=if(t<T,ton1+dur1,0)),"
        "(t-if(t<T,ton1,0))/(if(t<T,ton1+dur1,0)-if(t<T,ton1,0)),0)^(if(t<T,a1,a2)-1)"
    )
    value = J._exprtk_to_sympy(rate)
    params = {"k", "T", "ton1", "dur1", "a1", "a2"}
    prepared = J._prepare_zero_bases(value, params, params)
    deriv = J._finish_zero_bases(sp.diff(prepared, sp.Symbol("a1")))
    assert not deriv.has(sp.zoo, sp.nan), "the zoo must come from the rewrites, not before"
    names = {"k": "p[0]", "I": "obs[0]", "t": "obs[1]", "T": "p[1]", "ton1": "p[2]"}
    names.update({"dur1": "p[3]", "a1": "p[4]", "a2": "p[5]"})
    assert J.sympy_to_c(deriv, names.get) is None
    assert J.sympy_to_exprtk(deriv) is None
    assert "once the emitters' rewrites have run" in J.unemitted_derivative_reason(deriv, prepared)


def test_a_window_whose_bounds_end_in_zero_declines_the_sensitivity_rhs_cleanly(tmp_path):
    """The Jacobian attaches — ∂/∂t carries no logarithm — and the sensitivity run
    completes on CVODES' difference quotient, saying why, instead of raising."""
    assert WINDOW_BOUNDS_END_IN_ZERO != PULSE_NET, "the fixture edit did not apply"
    path = _net(tmp_path, WINDOW_BOUNDS_END_IN_ZERO, "window.net", c0=0)
    m = bngsim.Model.from_net(path)
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=["a1"])
    result = sim.run(**_between_crossings(20.0))
    assert not sim.has_analytic_sens_rhs
    assert "once the emitters' rewrites have run" in (sim.sens_rhs_decline_reason or "")
    assert np.all(np.isfinite(np.asarray(result.sensitivities)))


# ─── SIR_v4: the rate law itself is non-finite ───────────────────────────────


def test_a_rate_law_that_divides_by_zero_says_where():
    rate = "k*A/if(t<=10,a2,0)"
    assert J.differentiate_rate_law(rate, {}, {"A", "t"}, {"k", "a2"}) is None
    reason = J.last_decline_reason()
    assert "with respect to A carries a symbolic zoo" in reason, reason
    assert "which the rate law itself already has where (t > 10)" in reason, reason


def test_the_sensitivity_reason_names_the_singularity_not_a_function():
    value = J._exprtk_to_sympy("k*A/if(t<=10,a2,0)")
    reason = J.unemitted_derivative_reason(sp.diff(value, sp.Symbol("k")), value)
    assert "unsupported function" not in reason
    assert reason.startswith("carries a symbolic zoo")
    assert "where (t > 10)" in reason


def test_a_derivative_that_introduces_the_singularity_keeps_the_plain_reason():
    """The note is only for a rate law that already holds the non-finite atom; one
    the differentiation itself produced reads as it always did."""
    x = sp.Symbol("x")
    reason = J.unemitted_derivative_reason(sp.Piecewise((sp.zoo, sp.Eq(x, 0)), (x, True)), x)
    assert reason == J._NON_FINITE_REASON


# ─── The corpus models the issue names ───────────────────────────────────────

_NETS = "benchmarks/suites/ode_fullnet/nets/original__bngl_models__my_models__ode__{}.bngl.net"
_SIR_V4 = glob.glob(_NETS.format("SIR_v4"))
_SIR_V5 = glob.glob(_NETS.format("SIR_v5"))


@pytest.mark.skipif(not _SIR_V5, reason="benchmark ode_fullnet corpus not present")
def test_sir_v5_keeps_both_derivatives():
    m = bngsim.Model.from_net(_SIR_V5[0])
    assert m.prepare_analytical_jacobian() is True, m.analytical_jacobian_status
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=["a_2021", "dt_2021"])
    result = sim.run(t_span=(0.0, 1460.0), n_points=74)
    assert sim.has_analytic_sens_rhs, sim.sens_rhs_decline_reason
    assert np.all(np.isfinite(np.asarray(result.sensitivities)))


@pytest.mark.skipif(not _SIR_V4, reason="benchmark ode_fullnet corpus not present")
def test_sir_v4_declines_naming_its_last_year():
    m = bngsim.Model.from_net(_SIR_V4[0])
    assert m.prepare_analytical_jacobian() is False
    status = m.analytical_jacobian_status
    assert "which the rate law itself already has where" in status, status
    assert "(t > 1461)" in status, status
