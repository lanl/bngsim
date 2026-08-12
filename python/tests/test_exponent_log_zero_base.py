"""GH #310 — ``d/dn base^n = base^n·ln(base)`` at ``base == 0``.

Differentiating a Hill/power law with respect to its **exponent** produces
``base^n·ln(base)``. On the non-negative concentration domain BNGsim evaluates,
a species that is exactly zero — an unset initial condition, the common case at
``t = 0`` — makes that ``0·(-inf)`` = ``NaN`` in floating point, even though the
limit exists and is ``0`` for every ``n > 0``.

One NaN there is not a local blemish: it enters ``∂f/∂p`` on the first step and
either poisons that parameter's whole sensitivity column or, when it is the only
column, defeats the corrector and fails the solve outright. Both shapes are
under test below — the second is what the end-to-end case reproduces, and it is
how the issue was found (the AMICI forward-sensitivity parity job on
``BIOMD0000000012``).

The guard is symbolic and lives with the derivation
(:func:`bngsim._jacobian._guard_exponent_log_at_zero`), applied by both emitters
on their way out, so the ExprTk evaluator and every codegen backend (cc / MIR,
which compile the same C) get it from one place.

What it deliberately does **not** do is answer ``0`` unconditionally. The limit
is ``0`` only for a *positive* exponent; at ``n <= 0`` the expression has no
finite limit and a NaN is the honest report, so a numeric non-positive exponent
is left alone and a symbolic one is decided at run time against its current
value. A negative base is likewise untouched: that NaN is real.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("sympy")

from bngsim._jacobian import (  # noqa: E402
    _guard_exponent_log_at_zero,
    sympy_to_c,
    sympy_to_exprtk,
)

# A Hill activation whose activator starts at exactly zero. Reaction 1 is the
# basal B → A that lifts A off zero (without it A stays at zero forever and the
# whole trajectory is degenerate); reaction 2 is the Hill term carrying the
# exponent parameter, evaluated at Atot = 0 on the first step.
HILL_ZERO_IC = """\
begin parameters
    1 n      3.0  # Constant
    2 KM     2.0  # Constant
    3 vmax   1.5  # Constant
    4 kdeg   0.4  # Constant
end parameters
begin functions
    1 activate() vmax*Atot^n/(KM^n + Atot^n)
    2 basal()    kdeg*Btot
end functions
begin species
    1 A() 0.0
    2 B() 4.0
end species
begin reactions
    1 2 1 basal #_R1
    2 1 2 activate #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""


@pytest.fixture
def hill_net(tmp_path):
    net = tmp_path / "hill.net"
    net.write_text(HILL_ZERO_IC)
    return net


def _hill_derivative():
    """``∂/∂n`` of a Hill fraction, in sympy — the shape that produces the NaN."""
    import sympy as sp

    x, n, K = sp.symbols("x n K")
    return sp.diff(x**n / (K**n + x**n), n), (x, n, K)


# ─── the symbolic guard ────────────────────────────────────────────────────


class TestTheGuard:
    def test_the_limit_replaces_the_nan_at_a_zero_base(self):
        """The whole issue, at the level the fix is written: evaluate the raw
        derivative and the guarded one at ``x = 0``. The raw one is NaN; the
        guarded one is the limit."""
        import sympy as sp

        expr, (x, n, K) = _hill_derivative()
        raw = sp.lambdify((x, n, K), expr, "numpy")
        guarded = sp.lambdify((x, n, K), _guard_exponent_log_at_zero(expr), "numpy")

        with np.errstate(divide="ignore", invalid="ignore"):
            assert np.isnan(raw(0.0, 3.0, 2.0))
            assert guarded(0.0, 3.0, 2.0) == 0.0

    def test_a_nonzero_base_is_untouched(self):
        """The guard is a branch taken only at the singularity: everywhere else
        the emitted arithmetic must be the same number it always was."""
        import sympy as sp

        expr, (x, n, K) = _hill_derivative()
        raw = sp.lambdify((x, n, K), expr, "numpy")
        guarded = sp.lambdify((x, n, K), _guard_exponent_log_at_zero(expr), "numpy")

        for xv in (1e-8, 0.5, 2.0, 37.0):
            assert guarded(xv, 3.0, 2.0) == raw(xv, 3.0, 2.0)

    def test_a_nonpositive_numeric_exponent_keeps_its_nan(self):
        """``x^-2·ln(x)`` has no finite limit at ``x → 0+`` — it diverges. A
        blanket ``base == 0 → 0`` would paper over a genuine singularity, so the
        rewrite declines and the expression comes back untouched."""
        import sympy as sp

        x = sp.Symbol("x")
        expr = sp.Mul(sp.Pow(x, -2), sp.log(x), evaluate=False)
        assert _guard_exponent_log_at_zero(expr) == expr

    def test_a_symbolic_exponent_is_decided_at_run_time(self):
        """With the exponent a parameter, the sign is not known at build time,
        so the guard carries an ``exp > 0`` test alongside the ``base == 0`` one
        and both branches must be reachable."""
        import sympy as sp

        x, n = sp.symbols("x n")
        guarded = _guard_exponent_log_at_zero(sp.Mul(sp.Pow(x, n), sp.log(x), evaluate=False))
        f = sp.lambdify((x, n), guarded, "numpy")

        with np.errstate(divide="ignore", invalid="ignore"):
            assert f(np.float64(0.0), np.float64(3.0)) == 0.0  # limit exists
            # ...and no limit at a negative exponent, so the raw branch stands.
            assert not np.isfinite(f(np.float64(0.0), np.float64(-1.0)))

    def test_a_constant_base_is_not_guarded(self):
        """``log(2)`` is a finite constant. Guarding it would cost a branch per
        evaluation and buy nothing."""
        import sympy as sp

        x = sp.Symbol("x")
        expr = sp.Mul(sp.Pow(sp.Integer(2), x), sp.log(sp.Integer(2)), evaluate=False)
        assert _guard_exponent_log_at_zero(expr) == expr


# ─── the emitters ──────────────────────────────────────────────────────────


class TestBothEmitters:
    """One symbolic guard, applied on the way out of each emitter — so the
    interpreted ExprTk path and the compiled C path (cc and MIR alike, which
    compile the same source) cannot disagree about the value at zero."""

    def test_the_c_emitter_guards_every_log(self):
        expr, _ = _hill_derivative()
        c = sympy_to_c(expr, lambda s: {"x": "y[0]", "n": "p[0]", "K": "p[1]"}.get(s))
        assert c is not None
        # Each log() of a possibly-zero base sits inside a ternary that returns
        # 0.0 when the base is zero and the exponent positive.
        assert "((y[0] == 0.0) && (p[0] > 0.0))) ? (0.0) : (pow(y[0], p[0])*log(y[0]))" in c
        # sympy's StrPrinter spells equality as the function call `Eq(a, b)`,
        # which is neither C nor ExprTk; nothing may leak out in that form.
        assert "Eq(" not in c

    def test_the_exprtk_emitter_guards_every_log(self):
        expr, _ = _hill_derivative()
        s = sympy_to_exprtk(expr)
        assert s is not None
        assert "if(((x == 0) and (n > 0)),0,((x)^(n))*log(x))" in s
        assert "Eq(" not in s


# ─── end to end ────────────────────────────────────────────────────────────


class TestTheSolve:
    def test_the_solve_completes_and_matches_finite_differences(self, hill_net):
        """The reproducer's shape, with the exponent as the *only* sensitivity
        parameter: before the fix the NaN in ∂f/∂p defeated the corrector at the
        first step and the run raised. Now it completes, and the column it
        returns is checked against central finite differences of bngsim's own
        state trajectories — so the assertion does not rest on trusting the same
        analytic path it is testing."""
        model = bngsim.Model.from_net(hill_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["n"])
        result = sim.run(t_span=(0, 10), n_points=6, rtol=1e-10, atol=1e-12)

        sens = np.asarray(result.sensitivities)[:, :, 0]
        assert np.all(np.isfinite(sens))
        # Not a column of zeros dressed up as an answer.
        assert np.max(np.abs(sens)) > 1e-3

        # h = 1e-4, not the smaller step a truncation argument alone would pick:
        # the differenced quantity is a *solver output*, so each leg carries its
        # own integration error and the quotient divides that error by 2h. At
        # 1e-6 the amplification is what dominates the comparison, and it varies
        # with the platform's step sequence. This law is mild enough in n over
        # this window that the extra truncation costs less than the noise does.
        h = 1e-4
        legs = []
        for step in (+h, -h):
            perturbed = bngsim.Model.from_net(hill_net)
            perturbed.set_param("n", 3.0 + step)
            run = bngsim.Simulator(perturbed, method="ode").run(
                t_span=(0, 10), n_points=6, rtol=1e-12, atol=1e-14
            )
            legs.append(np.asarray(run.species))
        fd = (legs[0] - legs[1]) / (2 * h)

        # Loose, and deliberately so: what is under test is a NaN column against
        # a finite one, and three orders of margin over the observed agreement
        # still catches any answer that is merely plausible.
        np.testing.assert_allclose(sens, fd, rtol=1e-3, atol=1e-7)

    def test_the_column_survives_alongside_other_parameters(self, hill_net):
        """The other failure shape from the issue: with more columns sharing the
        solve, the corrector survives and the damage shows up as one all-NaN
        column while every other parameter reads fine. Every column must be
        finite, and the exponent's must still be the finite-difference one."""
        model = bngsim.Model.from_net(hill_net)
        names = ["KM", "kdeg", "n", "vmax"]
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=names)
        result = sim.run(t_span=(0, 10), n_points=6, rtol=1e-10, atol=1e-12)

        sens = np.asarray(result.sensitivities)
        assert np.all(np.isfinite(sens))

        solo = bngsim.Simulator(
            bngsim.Model.from_net(hill_net), method="ode", sensitivity_params=["n"]
        ).run(t_span=(0, 10), n_points=6, rtol=1e-10, atol=1e-12)
        # Not bit-for-bit: the columns in one solve share a CVODES error test, so
        # the column count moves the adaptive step sequence. Three orders over the
        # observed agreement, which is still far inside "the same column".
        np.testing.assert_allclose(
            sens[:, :, result.sensitivity_params.index("n")],
            np.asarray(solo.sensitivities)[:, :, 0],
            rtol=1e-6,
            atol=1e-9,
        )
