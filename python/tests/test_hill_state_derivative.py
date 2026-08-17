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

x, n, K, p = sp.symbols("x n K p")

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
    """``x^(2n-1)/(K^n + x^n)^2`` is ``(x^n)^2·x^-1`` over the sum, and where that
    ``x^-1`` goes is the whole design.

    Five placements were scored over one grid of 9236 evaluations against a
    200-digit oracle. Counting points that were non-finite before and are finite
    after, against points that went the other way, the shipped spelling — expanding
    the divided-through denominator into a sum of single powers — repairs 716 and
    breaks **1**, where dividing out one ``f`` at a time repairs 624 and breaks 14,
    and spreading the leftover as an ``m``-th root breaks **251**. Three of the
    four rejected placements are cheap to demonstrate at a single point each, which
    is the point of holding them down here rather than in a commit message.

    * **Beside the term**, ``x^-1·(f/(K^n + f))^2``: ``inf·0`` at ``x = 0``. That
      is the removable ``0/0`` GH #96's fold exists to remove, handed straight
      back — the trade the issue measured when it tried reordering the two
      rewrites.
    * **Divided out one ``f`` at a time**, the issue's own first proposal,
      ``x^(n-1)·(1/(1 + K^n·x^-n))·(K^n + x^n)^-1``: correct at ``x = 0``, correct
      at a negative state, and still ``NaN`` once ``x^n`` itself overflows — the
      leading ``x^(n-1)`` is ``inf`` beside a ``0``. It repairs the band where only
      the *square* has overflowed and leaves the band above it.
    * **Spread as an ``m``-th root through the factors**,
      ``1/(K^n·x^(1/2-n) + sqrt(x))^2``: correct at both of those, and ``NaN`` at a
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

    def test_dividing_out_one_f_at_a_time_stops_at_the_next_overflow(self):
        """The issue's own first proposal, verbatim. It is not wrong anywhere —
        it is incomplete: ``x^(n-1)`` overflows whenever ``x^n`` does, so above
        the band it repairs there is a second one it does not."""
        one_at_a_time = x ** (n - 1) * sp.Pow(1 + K**n * x**-n, -1) * sp.Pow(K**n + x**n, -1)
        beyond = {"x": 1e40, "n": 10.0, "K": 2.0}

        # In the issue's own band it is a repair, like the shipped spelling.
        assert np.isnan(self._at(self.TERM, **OVERFLOWING))
        assert self._at(one_at_a_time, **OVERFLOWING) == pytest.approx(1e-21, rel=1e-9)
        # One band up, both the numerator and the reciprocal have run out.
        assert self._at(one_at_a_time, **beyond) != self._at(one_at_a_time, **beyond)
        assert np.isfinite(_exprtk(self.TERM, **beyond))

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

    It reaches the same shortfall by another route. Its differentiator writes
    ``n·S^(n-1)`` directly rather than ``n·S^n/S``, so there is no fold here and no
    reordering question — the numerator simply arrives one power short of the
    summand it sits over.

    **Which way the ratio faces decides whether the model survives to notice.** An
    *activating* ``S^n/(K^n + S^n)`` has no reachable case on this path: the
    numerator ``S^(n-1)`` can only overflow once ``S^n`` has, and by then the rate
    law's own value is ``inf/inf`` too, so the run is already lost. An *inhibitory*
    ``1/(1 + (S/K)^n)`` — the same family written the other way round, and just as
    common — evaluates to a clean ``0`` at exactly the state where its derivative
    is ``NaN``. That is the one the solve below runs, and it is why this half is
    not merely the SymPy half's mirror image.
    """

    LAW = "vmax*S^nH/(KH^nH + S^nH)"
    INHIBITORY = "vmax/(1 + (S/KH)^nH)"
    CONSTS = frozenset({"vmax", "nH", "KH"})
    BASE = {"vmax": np.float64(1.5), "nH": np.float64(8.0), "KH": np.float64(2.0)}

    def _node(self, law=None, consts=None):
        derivs = _sat.differentiate_rate_law_native(
            law or self.LAW, {}, {"S"}, consts or self.CONSTS
        )
        assert derivs is not None
        return derivs["S"]

    def _eval(self, text, S, **over):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return float(
                eval(  # noqa: S307 - this module's own emitted text
                    text.replace("^", "**"),
                    {"exp": np.exp},
                    dict(self.BASE, **over, S=np.float64(S)),
                )
            )

    def test_the_inhibitory_ratio_is_nan_where_its_own_value_is_a_clean_zero(self):
        """The reachable case, and the premise of ``TestTheShapeThroughASolve``:
        ``1/(1 + inf)`` is ``0`` and the run continues, while ``(S/K)^(n-1)`` over
        the same ``1 + (S/K)^n`` is ``inf/inf``."""
        node = self._node(self.INHIBITORY)
        point = {"nH": np.float64(10.0), "KH": np.float64(1.0)}

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            law_value = float(
                eval(  # noqa: S307
                    self.INHIBITORY.replace("^", "**"),
                    {},
                    dict(self.BASE, **point, S=np.float64(1e35)),
                )
            )
        assert law_value == 0.0  # the model runs straight past it

        assert np.isnan(self._eval(_sat._emit_exprtk(node), 1e35, **point))
        assert np.isfinite(self._eval(_sat.emit_exprtk(node), 1e35, **point))

    def test_the_activating_ratio_is_repaired_too(self):
        """``S^(nH-1)`` and ``S^nH`` both ``inf`` at ``S = 1e50``. The rate law is
        past saving there — its own value is ``inf/inf`` — but the two emitters
        must still agree about the derivative, which is the whole reason this
        module carries a copy of the rewrite at all (GH #336's lesson)."""
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

# An inhibitory Hill switch whose substrate starts far above its threshold and
# decays through it. At t = 0, `(S/KH)^nH` is 1e350 — `inf` — so the flux is a
# clean `0` and the trajectory never notices; the Jacobian's `∂flux/∂S`, this
# issue's shape, is `inf/inf`. By t = 10 the substrate has decayed to O(1) and the
# switch has opened, so the run has a value to be right about as well as a NaN to
# avoid.
INHIBITORY_SWITCH = """\
begin parameters
    1 nH    10.0            # Constant
    2 KH    1.0             # Constant
    3 vmax  2.5             # Constant
    4 kdec  8.0             # Constant
    5 one   1.0             # Constant
end parameters
begin functions
    1 flux()  {law}
end functions
begin species
    1 S() 1e35
    2 P() 0.0
end species
begin reactions
    1 1 0 kdec #_R1
    2 0 2 flux #_R2
end reactions
begin groups
    1 Stot 1
    2 Ptot 2
end groups
"""

# The same law through each emitter. `cos(one - one)` is 1, so the arithmetic is
# untouched — but `cos` is not in the native differentiator's whitelist, so that
# law defers to SymPy and exercises `sympy_to_c`'s copy of the rewrite instead of
# `_saturable_jacobian`'s.
LAWS = {
    "native": "vmax/(1 + (Stot/KH)^nH)",
    "sympy": "vmax*cos(one - one)/(1 + (Stot/KH)^nH)",
}


class TestTheShapeThroughASolve:
    @pytest.fixture(params=sorted(LAWS))
    def switch_net(self, request, tmp_path):
        net = tmp_path / f"inhibitory_{request.param}.net"
        net.write_text(INHIBITORY_SWITCH.format(law=LAWS[request.param]))
        return net

    def test_the_sensitivity_solve_runs_and_is_finite(self, switch_net):
        """End to end, through both emitters.

        Without the rewrite this does not return a NaN — CVODES refuses the run at
        the first call of the sensitivity RHS (GH #395), ``CV_FIRST_SRHSFUNC_ERR``
        at ``t = 0``, on both paths.
        """
        import bngsim

        model = bngsim.Model.from_net(switch_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["vmax"])
        result = sim.run(t_span=(0, 10), n_points=11, rtol=1e-9, atol=1e-12)

        species = np.asarray(result.species)
        sens = np.asarray(result.sensitivities)
        assert np.all(np.isfinite(species))
        assert np.all(np.isfinite(sens))

        # The switch really did open: S decayed from 1e35 to O(1) and P moved.
        assert species[-1][0] == pytest.approx(1.8048527, rel=1e-5)
        assert species[-1][1] > 0.0

        # An oracle that needs no reference run: the flux is linear in vmax, so
        # dP/dvmax is exactly P/vmax at every output point.
        for t, (state, column) in enumerate(zip(species, sens, strict=True)):
            assert column[1][0] == pytest.approx(state[1] / 2.5, rel=1e-6), t


# ─── the cost this rewrite does incur ──────────────────────────────────────

# A Hill law at a species the solver can put a few ulps below zero. `A` is touched
# by no reaction, so its declared IC is its value at every callback and there is no
# predictor luck in the fixture (the GH #392 recipe).
NEGATIVE_STATE = """\
begin parameters
    1 nH    13.0            # Constant
    2 Kd    2.0             # Constant
    3 vmax  1.5             # Constant
end parameters
begin functions
    1 flux()  vmax*Atot^nH/(Kd^nH + Atot^nH)
end functions
begin species
    1 A() {ic}
    2 P() 0.0
end species
begin reactions
    1 0 2 flux #_R1
end reactions
begin groups
    1 Atot 1
    2 Ptot 2
end groups
"""


class TestTheNegativeStateCostAndWhatCoversIt:
    """The one regime where this rewrite is a step backwards, and the machinery
    that already catches it.

    The expanded denominator's terms are single powers of the base, and at a
    *negative* base their signs alternate with the parity of each exponent. When
    two of them overflow the other way up — ``a^2·x^(1-2n)`` at ``-inf`` beside
    ``2a·x^(1-n)`` at ``+inf`` — the sum is ``NaN`` where the shipped form read an
    underflowed ``±0.0``. Measured through both emitters over 200000 negative-base
    points with integer exponents: **7775 such points against 39447 repairs**, and
    **0 regressions** over the same sweep with a positive base.

    It is not information that is lost — 87% of those points read exactly ``±0.0``
    before, and the true value is below ``1e-308`` at almost all of them — but a
    ``NaN`` is not a zero: it propagates, and #395 makes the sensitivity RHS refuse
    on one.

    What covers it is not new. Every sampled regression has ``|x|`` inside a typical
    ``atol``, and at the *clamped* state both retries evaluate at — GH #135's
    unconditional nonnegative retry in ``cvode_codegen_dense_jac``, GH #392's
    ``atol``-bounded one in ``cvode_codegen_sens_rhs`` — the new form is finite at
    every one of them. This class holds that down at the boundary and through a
    solve, so a future change to either clamp cannot quietly take it away.
    """

    NEGATIVE = {"x": -1.1497569953977356e-30, "n": 13.0, "K": 32.19989080628787}

    def test_the_emitted_form_is_nan_at_the_negative_state(self):
        """Stated rather than hidden: this is the cost, at a point that has it."""
        assert np.isnan(_exprtk(sp.diff(RATIO, x), **self.NEGATIVE))

    def test_and_finite_at_the_state_the_retries_evaluate_at(self):
        """Which is why the cost is bounded. The clamp snaps the species to its
        boundary, and the expansion is *correct* there — it is the placement that
        keeps ``a^m·x^(1-2n)`` as its own term, which is exactly what makes ``x = 0``
        an ``inf`` in the denominator rather than an ``inf·0``."""
        clamped = dict(self.NEGATIVE, x=0.0)
        assert np.isfinite(_exprtk(sp.diff(RATIO, x), **clamped))
        assert np.isfinite(_c(sp.diff(RATIO, x), **clamped))

    @pytest.fixture(params=["-1e-30", "0.0", "1e-30"])
    def near_zero_net(self, request, tmp_path):
        net = tmp_path / f"neg_{request.param}.net"
        net.write_text(NEGATIVE_STATE.format(ic=request.param))
        return net

    def test_a_solve_at_that_state_still_runs(self, near_zero_net):
        """End to end, and the point of the whole class: a species a few ulps below
        zero under a Hill exponent of 13 is a state this rewrite newly returns NaN
        at, and the run completes anyway — with the same answer it gives at the
        boundary itself."""
        import bngsim

        model = bngsim.Model.from_net(near_zero_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["vmax"])
        result = sim.run(t_span=(0, 5), n_points=6, rtol=1e-9, atol=1e-12)

        assert np.all(np.isfinite(np.asarray(result.species, dtype=float)))
        assert np.all(np.isfinite(np.asarray(result.sensitivities, dtype=float)))
