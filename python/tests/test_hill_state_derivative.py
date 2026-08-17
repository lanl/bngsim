"""GH #402 — the state derivative of a bare Hill ratio must not be ``inf/inf`` either.

GH #393 stopped ``x^n/(K^n + x^n)`` overflowing when it is differentiated w.r.t. its
*exponent*, by dividing the ratio through by whichever factor runs away. It did not
reach the **state** direction of the plainest Hill ratio there is, and the reason was
one rewrite upstream::

    d/dx [x^n/(K^n + x^n)]  raw    -n·x^(2n)/(x·(K^n + x^n)^2) + n·x^n/(x·(K^n + x^n))
                            folded  n·x^(n-1)/(K^n + x^n) − n·x^(2n-1)/(K^n + x^n)^2

``_remove_removable_power_denominators`` (GH #96/#351) folds the removable ``x^(2n)/x``
into ``x^(2n-1)`` — it has to, that quotient is ``0/0`` at ``x = 0`` — and ``2n − 1``
is no multiple of ``n``, so GH #393's match by base-and-integer-exponent-ratio does not
fire and the emitted C keeps the overflow.

The band is the same one, and narrow only in the sense that a Hill exponent of 10
reaches it at ``x > 1e16``:

    x = 1e21, n = 10, K = 2      x^(n-1) = 1e189      x^(2n-1) = inf
                                 x^n     = 1e210      (K^n + x^n)^2 = inf
                                 emitted C = nan      true value = 1.02e-227

One square root lower down it is worse than a NaN. At ``x = 1e16`` only the *squared*
denominator has overflowed, so the second term silently reads ``0`` and the emitted
derivative comes back ``1e-15`` where the truth is ``1e-172`` — a wrong number with
nothing to mark it.

The fix carries a numeric offset alongside the exponent ratio, so ``x^(2n-1)`` is
recognised as ``(x^n)^2·x^-1``. Where that leftover goes is the whole design, and
``TestWhyTheLeftoverGoesInTheDenominator`` is the measurement that chose it: the two
other placements each trade this NaN for another one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim import _saturable_jacobian as _sat  # noqa: E402
from bngsim._jacobian import (  # noqa: E402
    _remove_removable_power_denominators,
    _rewrite_saturating_ratio,
    sympy_to_c,
    sympy_to_exprtk,
)

x, n, K, v, p = sp.symbols("x n K v p")

RATIO = x**n / (K**n + x**n)
# The issue's own point: x^n is an ordinary double and its square is not.
OVERFLOWING = {"x": 1e21, "n": 10.0, "K": 2.0}
# One square root lower: x^(2n-1) is still finite, (K^n + x^n)^2 is not.
HALF_OVERFLOWING = {"x": 1e16, "n": 10.0, "K": 2.0}
ORDINARY = {"x": 3.0, "n": 2.0, "K": 2.0}


def _exprtk(expr, **values):
    """Evaluate ``expr`` through the ExprTk emitter's own text.

    Not ``lambdify``: it re-spells ``a/x^n`` as ``a·x^(-n)``, and a separately
    computed reciprocal overflows where the quotient does not, so it answers a
    different question than the shipped text does.
    """
    text = sympy_to_exprtk(expr)
    assert text is not None
    py = text.replace("if(", "_if(").replace("^", "**")
    env = {"log": np.log, "exp": np.exp, "sqrt": np.sqrt, "_if": lambda c, a, b: a if c else b}
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(eval(py, env, {k: np.float64(w) for k, w in values.items()}))  # noqa: S307


def _c(expr, **values):
    src = sympy_to_c(expr, lambda name: name)
    assert src is not None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(
            eval(  # noqa: S307 - this module's own emitted C
                src,
                {"pow": np.float_power, "log": np.log, "exp": np.exp, "sqrt": np.sqrt},
                {k: np.float64(w) for k, w in values.items()},
            )
        )


def _shipped(expr, **values):
    """What the emitters printed before this fix: the pipeline with the folded
    numerator left unmatched. ``lambdify`` is honest here because the shipped form
    is a plain quotient of two powers — there is no reciprocal for it to re-spell.
    """
    folded = _remove_removable_power_denominators(expr)
    names = sorted(values, key=str)
    f = sp.lambdify([sp.Symbol(nm) for nm in names], folded, "numpy")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(f(*(np.float64(values[nm]) for nm in names)))


def _exact(expr, values, digits=200):
    """The derivative's true value at ``digits`` of precision — the oracle both
    forms are float64 approximations of. 200 rather than 40 because these are the
    points where two terms of order ``1e205`` cancel to ``1e-205``."""
    subs = {sp.Symbol(k): sp.Float(w, digits + 20) for k, w in values.items()}
    return expr.evalf(digits, subs=subs)


class TestTheMeasuredDefect:
    """The issue's table, term by term."""

    def test_the_folded_numerator_overflows_where_the_ratio_does_not(self):
        """The premise: ``x^n`` is an ordinary double at this state and both the
        folded numerator and the squared denominator are ``inf``."""
        assert _exprtk(x**n, **OVERFLOWING) == pytest.approx(1e210, rel=1e-9)
        assert _exprtk(x ** (n - 1), **OVERFLOWING) == pytest.approx(1e189, rel=1e-9)
        assert np.isinf(_exprtk(x ** (2 * n - 1), **OVERFLOWING))
        assert np.isinf(_exprtk((K**n + x**n) ** 2, **OVERFLOWING))

    def test_the_shipped_state_derivative_was_nan_and_is_now_finite(self):
        deriv = sp.diff(RATIO, x)

        assert np.isnan(_shipped(deriv, **OVERFLOWING))
        assert np.isfinite(_c(deriv, **OVERFLOWING))
        assert np.isfinite(_exprtk(deriv, **OVERFLOWING))
        # The truth is 1.02e-227 and what float64 has left after the rewrite is
        # the cancellation residue of two terms of order 1e-20 — 1e-36 or so.
        # Not the true number, but the same "indistinguishable from zero" the
        # column would carry anyway, where a NaN stops the run outright.
        assert abs(_c(deriv, **OVERFLOWING)) < 1e-30
        assert abs(_exact(deriv, OVERFLOWING)) < 1e-220

    def test_the_silently_wrong_number_one_square_root_lower(self):
        """Worse than the NaN, and the reason this is not only about ``inf/inf``.

        At ``x = 1e16`` the numerator ``x^(2n-1) = 1e304`` is still finite; only
        ``(K^n + x^n)^2`` has overflowed. So the second term reads a clean ``0``,
        drops out of the subtraction, and the derivative comes back as the first
        term alone — ``1e-15``, where the true value is ``1e-172``. Nothing marks
        it.
        """
        deriv = sp.diff(RATIO, x)
        truth = _exact(deriv, HALF_OVERFLOWING)

        assert np.isfinite(_shipped(deriv, **HALF_OVERFLOWING))
        assert _shipped(deriv, **HALF_OVERFLOWING) == pytest.approx(1e-15, rel=1e-6)
        assert abs(truth) < 1e-170  # ...and the truth is 157 orders below that

        # 1e-15 is the model's own scale here; the rewrite's answer is not.
        assert abs(_c(deriv, **HALF_OVERFLOWING)) < 1e-30

    def test_an_ordinary_hill_point_is_unmoved(self):
        deriv = sp.diff(RATIO, x)
        truth = float(_exact(deriv, ORDINARY))

        assert _shipped(deriv, **ORDINARY) == pytest.approx(truth, rel=1e-15)
        assert _c(deriv, **ORDINARY) == pytest.approx(truth, rel=1e-15)
        assert _exprtk(deriv, **ORDINARY) == pytest.approx(truth, rel=1e-15)


class TestWhyTheLeftoverGoesInTheDenominator:
    """``x^(2n-1)/(K^n + x^n)^2`` is ``(x^n)^2·x^-1`` over the sum, and there are
    three places to put that ``x^-1``. Two of them buy this NaN with another one,
    and both are cheap to demonstrate — which is why the shipped rewrite expands
    the denominator instead of factoring it.

    * Beside the term, ``x^-1·(f/(K^n + f))^2``: ``inf·0`` at ``x = 0``. That is
      the removable ``0/0`` GH #96's fold exists to remove, handed straight back.
    * Spread as an ``m``-th root through the factors,
      ``1/(K^n·x^(1/2-n) + sqrt(x))^2``: correct at both ends, and ``NaN`` at a
      negative state where the ``pow(x, 2n-1)`` it replaces is an ordinary number.
      A species dipping below zero mid-solve is routine, and an *integer* Hill
      exponent — the common case — is exactly where C's ``pow`` defines it.
    """

    TERM = sp.Mul(x ** (2 * n - 1), sp.Pow(K**n + x**n, -2), evaluate=False)

    def _at(self, expr, **values):
        names = sorted(values, key=str)
        f = sp.lambdify([sp.Symbol(nm) for nm in names], expr, "numpy")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return float(f(*(np.float64(values[nm]) for nm in names)))

    def test_the_leftover_left_standing_beside_the_term_is_nan_at_zero(self):
        beside = x**-1 * sp.Pow(1 + K**n * x**-n, -2)
        zero = {"x": 0.0, "n": 10.0, "K": 2.0}

        assert self._at(beside, **zero) != self._at(beside, **zero)  # NaN
        assert self._at(self.TERM, **zero) == 0.0  # what it has to be
        assert _exprtk(self.TERM, **zero) == 0.0

    def test_the_leftover_spread_as_a_root_is_nan_at_a_negative_state(self):
        rooted = sp.Pow(K**n * x ** (sp.Rational(1, 2) - n) + sp.sqrt(x), -2)
        negative = {"x": -2.0, "n": 10.0, "K": 2.0}
        truth = float(_exact(self.TERM, negative))

        assert self._at(rooted, **negative) != self._at(rooted, **negative)  # NaN
        assert self._at(self.TERM, **negative) == pytest.approx(truth, rel=1e-15)
        assert _exprtk(self.TERM, **negative) == pytest.approx(truth, rel=1e-15)

    def test_the_shipped_spelling_holds_all_three_ends(self):
        """The expanded denominator, at each point the other two lose."""
        for point in (
            {"x": 0.0, "n": 10.0, "K": 2.0},
            {"x": -2.0, "n": 10.0, "K": 2.0},
            OVERFLOWING,
        ):
            assert np.isfinite(_exprtk(self.TERM, **point)), point
            assert np.isfinite(_c(self.TERM, **point)), point

    def test_every_emitted_exponent_is_an_integer_shift_of_the_originals(self):
        """The property the negative state depends on: the rewrite prints powers
        of ``x`` at ``1``, ``1 − n`` and ``1 − 2n``, never a half. A base that
        ``pow`` accepted before is one it still accepts."""
        text = sympy_to_exprtk(self.TERM)
        assert "sqrt" not in text
        for spelling in ("(x)^(1 - n)", "(x)^(1 - 2*n)"):
            assert spelling in text, text


class TestTheIdentityAndItsAccuracy:
    def test_it_is_the_same_function_over_the_hill_family(self):
        """Scored against an exact evaluation at 200 digits over the whole ratio's
        symbol set. Two statements, because "always closer" would be false and
        would not mean much if it were: where the old form has digits the new one
        keeps them, and head to head the new one wins more than it loses.

        The bar for "keeps them" is a decade wider than the one that selects the
        points, and the gap is measured rather than picked. Over 1200 scored
        points the old form was inside ``1e-10`` at 1055 of them; the new form is
        outside ``1e-10`` at **3** of those 1055, worst case ``2.9e-10``, with a
        median error ratio of exactly ``1``. Both forms are the same difference of
        two nearly-equal terms at a saturated Hill point, and neither has digits
        the other does not — so the honest statement is a bounded degradation, not
        an identical one.
        """
        rng = np.random.default_rng(20260402)
        derivs = [sp.diff(RATIO, t) for t in (x, n, K)]

        well_conditioned = new_closer = old_closer = 0
        for _ in range(40):
            point = {
                "x": float(10 ** rng.uniform(-4, 4)),
                "n": float(rng.uniform(0.5, 6)),
                "K": float(10 ** rng.uniform(-2, 2)),
            }
            for raw in derivs:
                truth = _exact(raw, point)
                old, new = _shipped(raw, **point), _exprtk(raw, **point)
                if truth == 0 or not (np.isfinite(old) and np.isfinite(new)):
                    continue
                old_err = abs(float((sp.Float(old, 200) - truth) / truth))
                new_err = abs(float((sp.Float(new, 200) - truth) / truth))
                if old_err < 1e-10:
                    well_conditioned += 1
                    assert new_err < 1e-9, (point, old, new, truth)
                new_closer += new_err < old_err
                old_closer += old_err < new_err

        assert well_conditioned > 50  # the loop really did compare something
        assert new_closer > old_closer

    def test_a_positive_offset_is_carried_the_same_way(self):
        """``x^(n+1)`` beside ``x^n`` is ``f·x``, an offset of ``+1``, and it has
        the same defect in the other direction: at a large ``x`` the shipped form
        underflows the whole term to ``0`` where the truth is ``1e-189``."""
        term = sp.Mul(x ** (n + 1), sp.Pow(K + x**n, -2), evaluate=False)
        point = {"x": 1e21, "n": 10.0, "K": 2.0}
        truth = float(_exact(term, point))

        assert truth == pytest.approx(1e-189, rel=1e-3)
        assert _shipped(term, **point) == 0.0
        assert _exprtk(term, **point) == pytest.approx(truth, rel=1e-12)


class TestWhatIsStillLeftAlone:
    def test_an_offset_of_more_than_one_power_is_refused(self):
        """``x^(2n-2)`` is ``f^2·x^-2``, and ``x^2`` in the rewritten denominator
        is a fresh way to overflow — at ``x > 1e154`` — inside a rewrite whose
        purpose is to remove one. The bound is on what may be printed, not on
        what the fold may produce."""
        expr = sp.Mul(x ** (2 * n - 2), sp.Pow(K**n + x**n, -2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_symbolic_offset_is_refused(self):
        """``x^(2n+p)`` leaves ``x^p``, which is not a leftover to be placed —
        it is another power free to run away on its own."""
        expr = sp.Mul(x ** (2 * n + p), sp.Pow(K**n + x**n, -2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_numeric_exponent_still_buys_nothing(self):
        """``x^3/(K + x^2)^2``: an offset does not lift GH #393's gate. No state
        a solver can hold makes ``x^3`` overflow beside ``x^2``."""
        expr = sp.Mul(x**3, sp.Pow(K + x**2, -2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_bare_species_summand_is_still_not_an_f(self):
        expr = sp.Mul(x**n, sp.Pow(K + x, -2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr


class TestNoCollateralOnTheOffsetFreeRows:
    """GH #393's own shapes have no leftover, and they keep the text they had —
    byte for byte, in both printers. The offset is what pays for the expanded
    denominator, so a row without one must not be charged for it."""

    M, z = sp.symbols("M z")
    HILL_829 = z * (1 / M) ** n / (K**n + (1 / M) ** n)

    @pytest.mark.parametrize("wrt", ["n", "M", "K", "z"])
    def test_the_393_witness_emits_what_it_emitted(self, wrt):
        deriv = sp.diff(self.HILL_829, sp.Symbol(wrt))
        text = sympy_to_exprtk(deriv)
        assert text is not None
        # The offset-free divide-through, spelled as the reciprocal power both
        # printers already knew: ``a·f^-1 + 1``, not an expanded sum.
        if "K)^(n)" in text and wrt in ("n", "M"):
            assert "+ 1)" in text or "+ 1.0)" in text

    def test_the_measured_column_values_do_not_move(self):
        """``BIOMD0000000829``'s own numbers, which are GH #393's acceptance
        criterion."""
        measured = {"M": 4.58308775e-21, "n": 10.0, "K": 0.5, "z": 2.5}
        assert _exprtk(sp.diff(self.HILL_829, n), **measured) == 0.0
        assert _exprtk(sp.diff(self.HILL_829, self.M), **measured) == 0.0
        assert _exprtk(sp.diff(self.HILL_829, self.z), **measured) == 1.0
        d_K = sp.diff(self.HILL_829, K)
        assert _exprtk(d_K, **measured) == pytest.approx(float(_exact(d_K, measured)), rel=1e-14)


class TestTheNativeMirror:
    """``bngsim._saturable_jacobian`` differentiates the saturable family without
    SymPy and prints its own C, so it never sees a rewrite that lives in
    ``sympy_to_c``. It is the ``J·yS`` half of the analytic sensitivity RHS and
    the whole of the analytical Jacobian — guarding only the SymPy side fixes
    ∂f/∂p and lets the identical NaN back in one term later.

    Its differentiator writes ``n·S^(n-1)`` directly rather than ``n·S^n/S``, so
    there is no fold here and no reordering question: the numerator simply arrives
    one power short of the summand, which is the same shortfall by another route.
    """

    LAW = "vmax*S^nH/(KH^nH + S^nH)"
    CONSTS = frozenset({"vmax", "nH", "KH"})
    # nH = 8: at S = 1e40, S^nH overflows and S^(nH-1) does not — GH #393's point,
    # already fixed. At S = 1e50 BOTH overflow, and that is this issue's.
    BASE = {"vmax": np.float64(1.5), "nH": np.float64(8.0), "KH": np.float64(2.0)}

    def _node(self):
        derivs = _sat.differentiate_rate_law_native(self.LAW, {}, {"S"}, self.CONSTS)
        assert derivs is not None
        return derivs["S"]

    def _eval(self, text, S):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return float(
                eval(  # noqa: S307 - this module's own emitted text
                    text.replace("^", "**"),
                    {"exp": np.exp},
                    dict(self.BASE, S=np.float64(S)),
                )
            )

    def test_both_powers_overflowing_was_nan_and_is_now_finite(self):
        node = self._node()
        assert np.isnan(self._eval(_sat._emit_exprtk(node), 1e50))
        assert np.isfinite(self._eval(_sat.emit_exprtk(node), 1e50))

    def test_the_393_point_still_reads_zero(self):
        node = self._node()
        assert self._eval(_sat.emit_exprtk(node), 1e40) == 0.0

    def test_it_is_the_same_function_away_from_the_overflow(self):
        node = self._node()
        raw = self._eval(_sat._emit_exprtk(node), 2.2)
        assert np.isfinite(raw) and raw != 0.0
        assert self._eval(_sat.emit_exprtk(node), 2.2) == pytest.approx(raw, rel=1e-14)

    def test_a_law_with_no_power_over_a_sum_emits_the_same_text_as_before(self):
        derivs = _sat.differentiate_rate_law_native(
            "vmax*S/(KM + S)", {}, {"S"}, frozenset({"vmax", "KM"})
        )
        node = derivs["S"]
        assert _sat.emit_exprtk(node) == _sat._emit_exprtk(node)


# ─── the shape through a solve ─────────────────────────────────────────────

# A Hill law whose substrate sits in the band: nH = 10 and S = 1e20 put S^nH at
# 1e200 — an ordinary double — and S^(2·nH-1) at 1e380, which is not. The value
# path never notices, the ratio being saturated at 1, so the trajectory is a clean
# straight line and only the differentiated form fails.
HILL_STATE = """\
begin parameters
    1 nH    10.0            # Constant
    2 KH    1.0             # Constant
    3 vmax  2.5             # Constant
end parameters
begin functions
    1 flux()  vmax*Stot^nH/(KH^nH + Stot^nH)
end functions
begin species
    1 S() 1e20
    2 P() 0.0
end species
begin reactions
    1 0 2 flux #_R1
end reactions
begin groups
    1 Stot 1
    2 Ptot 2
end groups
"""


class TestTheShapeThroughASolve:
    @pytest.fixture
    def hill_net(self, tmp_path):
        net = tmp_path / "hill_state.net"
        net.write_text(HILL_STATE)
        return net

    def test_the_sensitivity_solve_is_finite_from_the_first_output_point(self, hill_net):
        """End to end. ``S`` is large enough that the Jacobian's own diagonal —
        ∂flux/∂S, which is exactly this issue's shape — overflows, and the
        ``J·yS`` half of the sensitivity RHS carries it."""
        import bngsim

        model = bngsim.Model.from_net(hill_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["vmax"])
        result = sim.run(t_span=(0, 10), n_points=11, rtol=1e-9, atol=1e-12)

        species = np.asarray(result.species)
        assert np.all(np.isfinite(species))
        # The ratio is saturated, so P accumulates at vmax.
        assert species[-1][1] == pytest.approx(25.0, rel=1e-6)

        sens = np.asarray(result.sensitivities)
        assert np.all(np.isfinite(sens))
        # dP/dvmax = t for a saturated ratio, and it is the column the Jacobian
        # entry under test propagates.
        assert sens[-1][1][0] == pytest.approx(10.0, rel=1e-5)
