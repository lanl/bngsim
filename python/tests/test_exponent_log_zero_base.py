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

GH #317 is the same NaN reached by a differently-spelled divergence. The limit
argument never depended on the logarithm appearing exactly once — ``base^exp``
with ``exp > 0`` decays faster than *any* power of ``ln(base)`` diverges — but
the first implementation paired one ``log`` *node* with a ``Pow`` sibling, so
``ln(base)^2``, ``(a + ln base)`` and ``ln(base/K)`` all walked past it and
NaN'd with the guard sitting beside them. The guard now takes the whole
logarithmic sub-product of each ``Mul``; ``TestShapesBeyondOneSiblingLog``
covers each shape, and ``TestWhatStaysOutside`` pins what it still declines.

GH #388 is the third: one logarithm can be the carrier of *two* bases —
``ln(k·(t − T))`` diverges at either — and the scan handed it to whichever power
it met first and then moved on, leaving the other base unguarded with nothing
left to guard it. ``TestOneLogarithmCarriesTwoBases`` is that case, and
``MODEL2403070001`` is where it cost a whole sensitivity column.
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


# ─── GH #317: divergences the sibling pairing could not see ────────────────


def _at(expr, **values):
    """Evaluate ``expr`` numerically, with the ``0·inf`` warnings quieted."""
    import sympy as sp

    names = sorted(values, key=str)
    f = sp.lambdify([sp.Symbol(k) for k in names], expr, "numpy")
    with np.errstate(divide="ignore", invalid="ignore"):
        return f(*(values[k] for k in names))


class TestShapesBeyondOneSiblingLog:
    """Each of these is ``base^exp · (something logarithmic in base)`` at
    ``base == 0``, so each has limit ``0`` for ``exp > 0`` by exactly the
    argument #310 made. The first implementation matched only a bare ``log``
    node sitting beside the ``Pow`` in the same ``Mul``, and so answered NaN for
    every one of them.

    Every shape here is what a *first-order* ``sp.diff`` hands the emitter for
    an ordinary rate law — no ``factor()``/``collect()``/``simplify()`` involved.
    """

    def test_a_squared_logarithm(self):
        """``d/dn`` of a rate law that itself contains ``ln S``. The ``log`` is
        wrapped in a ``Pow``, so an ``isinstance(factor, sp.log)`` test never
        sees it."""
        import sympy as sp

        x, n, v = sp.symbols("x n v")
        expr = sp.diff(v * x**n * sp.log(x), n)
        assert expr == v * x**n * sp.log(x) ** 2  # the shape under test

        assert np.isnan(_at(expr, x=0.0, n=3.0, v=1.5))
        assert _at(_guard_exponent_log_at_zero(expr), x=0.0, n=3.0, v=1.5) == 0.0

    def test_a_logarithm_inside_a_sum_factor(self):
        """A hand-factored log difference: the ``log`` sits inside an ``Add``
        factor rather than beside the ``Pow``. ``bottom_up`` does descend into
        the ``Add``, but each summand is then a ``Mul`` with no ``Pow`` of the
        base in it, so the pairing had nothing to pair."""
        import sympy as sp

        x, n, K = sp.symbols("x n K")
        expr = sp.Mul(sp.Pow(x, n), sp.log(x) - sp.log(K), evaluate=False)

        assert np.isnan(_at(expr, x=0.0, n=3.0, K=2.0))
        assert _at(_guard_exponent_log_at_zero(expr), x=0.0, n=3.0, K=2.0) == 0.0

    def test_a_logarithm_whose_argument_is_not_structurally_the_base(self):
        """``ln(S/K)`` diverges at ``S = 0`` just as ``ln S`` does, but its
        argument is ``S/K``, which is not structurally equal to the ``Pow``'s
        base ``S``. The old equality test at the pairing step rejected it."""
        import sympy as sp

        x, n, K, v = sp.symbols("x n K v")
        expr = sp.diff(v * x**n * sp.log(x / K), n)

        assert np.isnan(_at(expr, x=0.0, n=3.0, K=5.0, v=1.5))
        assert _at(_guard_exponent_log_at_zero(expr), x=0.0, n=3.0, K=5.0, v=1.5) == 0.0

    def test_the_product_rule_leaves_a_sum_beside_the_pow(self):
        """The shape a plain first-order derivative actually produces for
        ``v·S^n·(a + ln S)`` — a ``log`` sibling *and* an ``Add`` sibling. The
        pairing guard fired here, absorbed the bare ``log``, and still returned
        NaN because the ``Add`` went on diverging next to the ``0``."""
        import sympy as sp

        x, n, v, a = sp.symbols("x n v a")
        expr = sp.diff(v * x**n * (a + sp.log(x)), n)

        assert np.isnan(_at(expr, x=0.0, n=3.0, v=1.5, a=0.7))
        assert _at(_guard_exponent_log_at_zero(expr), x=0.0, n=3.0, v=1.5, a=0.7) == 0.0

    def test_every_logarithmic_sibling_lands_in_one_piecewise(self):
        """Absorbing the siblings one at a time would nest a ``Piecewise`` per
        ``log`` and pay a branch for each. One product, one branch — and nothing
        logarithmic in the base may be left multiplying outside it, which is
        precisely the bug the previous test describes."""
        import sympy as sp

        x, n, v, a = sp.symbols("x n v a")
        guarded = _guard_exponent_log_at_zero(sp.diff(v * x**n * (a + sp.log(x)), n))

        branches = [e for e in sp.preorder_traversal(guarded) if isinstance(e, sp.Piecewise)]
        assert len(branches) == 1
        raw = branches[0].args[1][0]
        assert raw == x**n * (a + sp.log(x)) * sp.log(x)
        # ...and the only factor left outside is the base-free `v`.
        assert set(guarded.args) - {branches[0]} == {v}

    def test_a_second_base_in_the_same_product_is_guarded_too(self):
        """Two independent powers, each with its own logarithm, get their own
        branch: absorbing the first must not end the scan."""
        import sympy as sp

        x, y, n, m = sp.symbols("x y n m")
        expr = sp.Mul(sp.Pow(x, n), sp.Pow(y, m), sp.log(x), sp.log(y), evaluate=False)
        guarded = _guard_exponent_log_at_zero(expr)

        branches = [e for e in sp.preorder_traversal(guarded) if isinstance(e, sp.Piecewise)]
        assert len(branches) == 2
        assert _at(guarded, x=0.0, y=2.0, n=3.0, m=3.0) == 0.0
        assert _at(guarded, x=2.0, y=0.0, n=3.0, m=3.0) == 0.0


class TestOneLogarithmCarriesTwoBases:
    """GH #388. ``ln(x·y)`` diverges at ``x = 0`` and at ``y = 0`` alike, and
    blanking logarithms — the test :func:`_is_log_carrier` makes — leaves neither
    base behind, so it is a carrier for **both** powers beside it.

    The scan used to take the first candidate power, absorb its carriers into
    that power's ``Piecewise``, and move on. Every later base then found no
    carrier left and went unguarded, which is not a missed optimisation: the
    carrier it lost was the one that diverges there. Powers linked by a shared
    carrier are grouped instead, and the group's condition is the ``Or`` of the
    members' — each base going to zero still sends the whole product to zero,
    because that base's power decays while its siblings stay finite.
    """

    def test_a_shared_logarithm_guards_every_base_it_carries(self):
        import sympy as sp

        x, y, n, m = sp.symbols("x y n m")
        expr = sp.Mul(sp.Pow(x, n), sp.Pow(y, m), sp.log(x * y), evaluate=False)

        assert np.isnan(_at(expr, x=0.0, y=2.0, n=3.0, m=3.0))
        assert np.isnan(_at(expr, x=2.0, y=0.0, n=3.0, m=3.0))

        guarded = _guard_exponent_log_at_zero(expr)
        assert _at(guarded, x=0.0, y=2.0, n=3.0, m=3.0) == 0.0
        assert _at(guarded, x=2.0, y=0.0, n=3.0, m=3.0) == 0.0
        # and away from either zero it is still the same function
        assert _at(guarded, x=2.0, y=3.0, n=3.0, m=3.0) == pytest.approx(
            _at(expr, x=2.0, y=3.0, n=3.0, m=3.0), rel=1e-15
        )

    def test_the_shared_carrier_makes_one_branch_not_two(self):
        """One product, one branch — the two powers are in the same group, so
        the guard cannot leave half of it multiplying outside."""
        import sympy as sp

        x, y, n, m = sp.symbols("x y n m")
        guarded = _guard_exponent_log_at_zero(
            sp.Mul(sp.Pow(x, n), sp.Pow(y, m), sp.log(x * y), evaluate=False)
        )

        branches = [e for e in sp.preorder_traversal(guarded) if isinstance(e, sp.Piecewise)]
        assert len(branches) == 1
        assert branches[0].args[1][0] == x**n * y**m * sp.log(x * y)

    def test_the_meal_pulse_derivative_is_finite_at_its_own_onset(self):
        """``MODEL2403070001``'s shape, and the one that found this.

        ``G_meal = σ·k^σ·(t − T)^(σ−1)·exp(−(k(t − T))^σ)`` differentiated w.r.t.
        the shape parameter ``σ`` carries ``k^σ·(t − T)^(σ−1)·ln(k(t − T))``. The
        logarithm was absorbed by ``k^σ`` — a rate constant that is never zero,
        so the branch never fires — leaving ``(t − T)^(σ−1)·ln(k(t − T))`` to NaN
        at ``t = T``, which is exactly the instant the pulse starts."""
        import sympy as sp

        sigma, k, t, T = sp.symbols("sigma k t T")
        law = sigma * k**sigma * (t - T) ** (sigma - 1) * sp.exp(-((k * (t - T)) ** sigma))
        deriv = sp.diff(law, sigma)
        onset = {"sigma": 1.4, "k": 0.05, "t": 60.0, "T": 60.0}

        assert np.isnan(_at(deriv, **onset))
        assert _at(_guard_exponent_log_at_zero(deriv), **onset) == 0.0

        after = {"sigma": 1.4, "k": 0.05, "t": 61.0, "T": 60.0}
        assert _at(_guard_exponent_log_at_zero(deriv), **after) == pytest.approx(
            _at(deriv, **after), rel=1e-14
        )


class TestWhatStaysOutside:
    """The generalization widens what counts as "logarithmic in the base", and
    must not widen into swallowing singularities that are real. So a sibling
    qualifies only when blanking the logarithms removes ``base`` from it *and*
    leaves a polynomial in those logarithms — the second half because ``exp``
    inverts a logarithm, which
    :meth:`test_an_exponential_undoes_a_logarithm_and_is_not_a_carrier` is
    about."""

    def test_a_pole_beside_the_guard_is_still_reported(self):
        """``1/(S + K)`` mentions the base outside a logarithm. It is finite at
        ``S = 0`` and needs no help, so it stays outside the ``Piecewise`` and
        multiplies through — which means that when it is *not* finite (``K = 0``
        too), the NaN it produces still reaches the caller instead of being
        rewritten to a confident zero."""
        import sympy as sp

        x, n, K = sp.symbols("x n K")
        expr = sp.Mul(sp.Pow(x, n), sp.log(x), sp.Pow(x + K, -1), evaluate=False)
        guarded = _guard_exponent_log_at_zero(expr)

        assert _at(guarded, x=0.0, n=3.0, K=2.0) == 0.0  # this guard's own case
        assert np.isnan(_at(guarded, x=0.0, n=3.0, K=0.0))  # somebody else's pole

    def test_an_exponential_undoes_a_logarithm_and_is_not_a_carrier(self):
        """``exp(ln(u)/k)`` is ``u^(1/k)`` — a *power* of the base wearing a
        logarithm's clothes. So "the base appears only under a ``log``" is not
        on its own enough to call a factor logarithmic, and the test is that
        what remains after blanking the logarithms is a polynomial in them.

        The first expression is how BIOMD0000000613 writes a Hill exponent, and
        it is what caught this: reading it as a carrier is *answerable* there
        (the product really is ``OC0^(gam+1)·c``), so it passes unnoticed. The
        second is the same construction where the answer is not ``0`` at all —
        at ``k = -1`` it is ``x^(n-2)``, which diverges at ``x = 0`` for
        ``n < 2`` and must keep saying so.
        """
        import sympy as sp

        x, n, k, A, T = sp.symbols("x n k A T")

        hill = sp.Mul(
            sp.Pow(x, n), sp.exp(sp.log(A * sp.Pow(x, n) / T - sp.Pow(x, n)) / n), evaluate=False
        )
        assert _guard_exponent_log_at_zero(hill) == hill

        divergent = sp.Mul(sp.Pow(x, n), sp.exp(sp.log(sp.Pow(x, 2)) / k), evaluate=False)
        assert _guard_exponent_log_at_zero(divergent) == divergent

    def test_a_nonpositive_exponent_keeps_its_nan_under_a_power_of_log(self):
        """``ln(S)^2/S^2`` diverges; widening the ``log`` side must not weaken
        the exponent-sign test that #310 put on the ``Pow`` side."""
        import sympy as sp

        x = sp.Symbol("x")
        expr = sp.Mul(sp.Pow(x, -2), sp.Pow(sp.log(x), 2), evaluate=False)
        assert _guard_exponent_log_at_zero(expr) == expr

    def test_a_nonzero_base_is_still_untouched(self):
        """Off the singularity the guard is a branch not taken, and the raw
        branch holds the same factors it always did.

        Not asserted bit-for-bit: pulling the logarithmic factors into the
        ``Piecewise`` re-associates the product, so the last ulp can move. That
        is the whole of the numerical difference away from ``base == 0`` — a few
        parts in 1e16, against a guard that exists to replace a NaN.
        """
        import sympy as sp

        x, n, K, v, a = sp.symbols("x n K v a")
        for expr in (
            sp.diff(v * x**n * sp.log(x), n),
            sp.diff(v * x**n * sp.log(x / K), n),
            sp.diff(v * x**n * (a + sp.log(x)), n),
        ):
            guarded = _guard_exponent_log_at_zero(expr)
            for xv in (1e-8, 0.5, 2.0, 37.0):
                full = dict(x=xv, n=3.0, K=5.0, v=1.5, a=0.7)
                point = {k: full[k] for k in (str(s) for s in expr.free_symbols)}
                np.testing.assert_allclose(
                    _at(guarded, **point), _at(expr, **point), rtol=1e-15, atol=0
                )


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

    def test_both_emitters_carry_the_squared_logarithm_into_the_branch(self):
        """GH #317 through the emitters: the ``log`` is inside a ``Pow`` here,
        and what each backend must show is the *whole* product under one test,
        not a guarded factor multiplied by an unguarded divergence."""
        import sympy as sp

        x, n, v = sp.symbols("x n v")
        expr = sp.diff(v * x**n * sp.log(x), n)  # v*x**n*log(x)**2

        c = sympy_to_c(expr, lambda s: {"x": "y[0]", "n": "p[0]", "v": "p[1]"}.get(s))
        assert c is not None
        assert (
            "((y[0] == 0.0) && (p[0] > 0.0))) ? (0.0) : "
            "(pow(y[0], p[0])*((log(y[0]))*(log(y[0]))))" in c
        )
        assert "Eq(" not in c

        s = sympy_to_exprtk(expr)
        assert s is not None
        assert "if(((x == 0) and (n > 0)),0,((x)^(n))*((log(x))^(2)))" in s
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


# ─── #317's shape, through a solve ─────────────────────────────────────────

# Every #317 shape needs a logarithm in the rate law itself. A first-order
# derivative introduces at most one `ln(g)` per product-rule term — that is all
# `d/dn g^n = g^n·ln(g)` can contribute — so a log-free law cannot produce
# `ln(base)^2`, `(a + ln base)` or `ln(base/K)` beside the power no matter how
# sympy associates the result. (Measured: over 1644 corpus models and 117842
# first-order derivatives, the generalized guard and the old sibling pairing
# emit the same expression for every one.)
#
# Such a rate law used to fail in the RHS at `S = 0` before any derivative was
# consulted, because the value path never passed through the emitters that apply
# this guard. That was GH #333 and it is fixed — `test_rate_law_zero_base_log.py`
# holds its contract, and a solve now runs from `S = 0` on both engines.
#
# The forward-sensitivity run then stayed broken for one more issue's worth of
# time, and the cause was not a zero base, a negative one, or anything in this
# file: it was GH #151's native SymPy-free differentiator, which recognizes
# `^` and `ln` but reaches none of the emitters this guard lives in. It
# differentiated `vmax·S^n·ln(S)` term by term into
# `(n·S^(n-1))·ln(S) + S^n·(1/S)` — both halves `0·∞` at `S = 0` — and that
# expression is the Jacobian entry, so it entered the `J·yS` half of the analytic
# sensitivity RHS and made every column NaN on the first call, including columns
# for parameters the logarithm does not touch. GH #336 fixed that by deferring a
# state-dependent logarithm to SymPy, where this guard applies; the assertion
# below (a strict xfail until then) is the end-to-end case.
#
# It is worth saying why the earlier reading — "the solver evaluates the law at
# `S < 0`" — looked right. The evidence for it was that `ln(abs(S))` makes the
# same run complete. It does, but not because `abs` finitizes a negative
# argument: `abs` is not in the native differentiator's whitelist, so spelling it
# that way pushed the whole law onto SymPy and picked up the guard.
LOG_RATE_LAW = HILL_ZERO_IC.replace(
    "1 activate() vmax*Atot^n/(KM^n + Atot^n)",
    "1 activate() vmax*Atot^n*ln(Atot)",
)


class TestTheShapeThroughASolve:
    @pytest.fixture
    def log_net(self, tmp_path):
        net = tmp_path / "logpow.net"
        net.write_text(LOG_RATE_LAW)
        return net

    def test_the_value_of_that_rate_law_integrates_from_zero(self, log_net):
        """GH #333, from this file's side: the same model whose RHS used to be
        NaN at ``Atot = 0`` solves."""
        model = bngsim.Model.from_net(log_net)
        assert model._guarded_functions, "the rate law should have been guarded"
        species = np.asarray(
            bngsim.Simulator(model, method="ode")
            .run(t_span=(0, 5), n_points=4, rtol=1e-10, atol=1e-12)
            .species
        )
        assert np.all(np.isfinite(species))

    def test_the_derivative_of_that_rate_law_is_guarded(self, log_net):
        """Taken from the model's own rate-law text rather than a hand-built
        expression: `∂/∂n` of `vmax·S^n·ln S` is the ``ln(S)^2`` shape, and
        through bngsim's own converter it answers the limit at ``S = 0`` instead
        of NaN."""
        import sympy as sp
        from bngsim._jacobian import _exprtk_to_sympy

        expr = _exprtk_to_sympy("vmax*Atot^n*log(Atot)")
        deriv = sp.diff(expr, sp.Symbol("n"))
        assert (
            deriv
            == sp.Symbol("vmax")
            * sp.Symbol("Atot") ** sp.Symbol("n")
            * sp.log(sp.Symbol("Atot")) ** 2
        )

        point = {"Atot": 0.0, "n": 3.0, "vmax": 1.5}
        assert np.isnan(_at(deriv, **point))
        assert _at(_guard_exponent_log_at_zero(deriv), **point) == 0.0

    def test_the_solve_completes_and_the_exponent_column_is_finite(self, log_net):
        """The end-to-end case, and the only route to #317's ``ln(S)^2`` shape
        through a solve rather than through the emitter. Carried a strict xfail
        until GH #336 stopped the native differentiator from re-introducing the
        NaN this guard removes."""
        model = bngsim.Model.from_net(log_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["n"])
        result = sim.run(t_span=(0, 5), n_points=4, rtol=1e-10, atol=1e-12)

        sens = np.asarray(result.sensitivities)[:, :, 0]
        assert np.all(np.isfinite(sens))
        assert np.max(np.abs(sens)) > 1e-3

    def test_the_zero_start_agrees_with_the_limit_from_above(self, log_net, tmp_path):
        """The control on the case above: one floating-point value apart the run
        always worked, so starting *at* zero is only credible if it lands on what
        approaching zero gives."""
        net = tmp_path / "logpow_eps.net"
        net.write_text(LOG_RATE_LAW.replace("1 A() 0.0", "1 A() 1e-30"))

        def final(path):
            model = bngsim.Model.from_net(path)
            sim = bngsim.Simulator(model, method="ode", sensitivity_params=["n"])
            result = sim.run(t_span=(0, 5), n_points=4, rtol=1e-10, atol=1e-12)
            sens = np.asarray(result.sensitivities)
            assert np.all(np.isfinite(sens))
            return sens[-1, :, 0]

        np.testing.assert_allclose(final(log_net), final(net), rtol=1e-6, atol=1e-9)
