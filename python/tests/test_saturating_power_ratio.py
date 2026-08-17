"""GH #393 — ``x^n/(K^n + x^n)`` must not be differentiated into ``inf/inf``.

A Hill ratio saturates at ``1``, so once ``x^n`` is large the fraction is flat
and every derivative of it is a very small number. Written out literally, though,
the quotient rule puts ``x^n`` in the numerator a second time:

    ∂/∂n [x^n/(K^n + x^n)]  =  x^n·ln x/(K^n + x^n)
                             + x^n·(−K^n·ln K − x^n·ln x)/(K^n + x^n)²

and the second term is ``inf/inf`` = **NaN** whenever ``x^n`` is big enough that
its *square* overflows — from ``1e154`` up, which is one square root short of the
``1e308`` the value path needs. ``BIOMD0000000829`` sits squarely in that band:

    mass_s := mass + zeta_1·(1/mTOR_R)^n_1/(K_m^n_1 + (1/mTOR_R)^n_1)
    zeta_1 = 2.5   K_m = 0.5   n_1 = 10   mTOR_R = 4.58308775e-21

``1/mTOR_R = 2.18e20``, so ``(1/mTOR_R)^n_1 = 2.45e203`` is an ordinary finite
double and ``(1/mTOR_R)^(2·n_1)`` is not. The value path never notices — the
fraction evaluates to a clean ``1`` — so the trajectory matches its reference and
only the ``n_1`` sensitivity column fails, at the first output point.

The rewrite is GH #388's, one family over: ``bngsim._jacobian``'s
:func:`_rewrite_saturating_ratio` divides through by the overflowing factor,

    f^m/(a + f)^k  ==  (1/(1 + a/f))^m · (a + f)^(m−k)

which is an identity wherever ``f`` is finite and nonzero and has no intermediate
that can overflow into a ratio of infinities — one factor saturates as ``f → ∞``
and the other as ``f → 0``. For the sigmoid ``f`` is ``exp(u)`` (GH #388, covered
in ``test_saturating_exp_ratio.py``); here it is ``x^n``.

``m > 1`` is the *state* direction of the same ratio. ``sp.diff(…, x)`` folds the
quotient rule's ``x^n·x^n`` into a single ``x^(2n)``, so the numerator is matched
by base and integer exponent ratio rather than by identity.

Both emitters now run the zero-base logarithm guard (GH #310/#317) *before* this
rewrite rather than after. The two want the same ``x^n``, at opposite ends of its
range, and taking it away from the guard turns a clean ``0`` at ``x = 0`` into a
NaN — ``TestTheOrderAgainstTheZeroBaseGuard`` holds that down.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim import _saturable_jacobian as _sat  # noqa: E402
from bngsim._jacobian import (  # noqa: E402
    _rewrite_saturating_ratio,
    sympy_to_c,
    sympy_to_exprtk,
)

M, n, K, z, x, y, S = sp.symbols("M n K z x y S")

# The model's own numbers, from the assignment rule quoted above.
HILL = z * (1 / M) ** n / (K**n + (1 / M) ** n)
MEASURED = {"M": 4.58308775e-21, "n": 10.0, "K": 0.5, "z": 2.5}


def _at(expr, **values):
    """Evaluate ``expr`` in float64, with the overflow warnings quieted."""
    names = sorted(values, key=str)
    f = sp.lambdify([sp.Symbol(nm) for nm in names], expr, "numpy")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(f(*(np.float64(values[nm]) for nm in names)))


def _exact(expr, values, digits=200):
    """The derivative's true value at ``digits`` of precision.

    ``sp.Float(v, digits + 20)`` is the *exact* binary double — float64 carries
    at most 17 significant decimal digits — so this is the same expression at the
    same inputs, without float64's rounding along the way. It is the oracle: both
    the old form and the new one are float64 approximations of this number, and
    the question a text diff cannot answer is which one is closer.

    200 digits, not 40, because the points that matter here are exactly the ones
    where two terms of order ``1e205`` cancel to ``1e-205``. At 40 digits sympy
    answers ``0.e-171`` — an honest "no significant digits", and a silent zero to
    anything that divides by it. A caller must still treat a zero as "no oracle"
    rather than as a value.
    """
    subs = {sp.Symbol(k): sp.Float(v, digits + 20) for k, v in values.items()}
    return expr.evalf(digits, subs=subs)


def _relative_error(value, truth):
    return abs(float((sp.Float(value, 200) - truth) / truth))


class TestTheMeasuredModel:
    """``BIOMD0000000829``'s arithmetic, symbol for symbol. Every derivative of
    its Hill ratio is an ordinary number there; two of the four are NaN without
    the rewrite."""

    @pytest.mark.parametrize("wrt", [n, M])
    def test_the_raw_form_is_nan_and_the_rewritten_one_is_finite(self, wrt):
        deriv = sp.diff(HILL, wrt)

        assert np.isnan(_at(deriv, **MEASURED))
        rewritten = _at(_rewrite_saturating_ratio(deriv), **MEASURED)
        assert np.isfinite(rewritten)
        # The true values are 4.7e-205 (w.r.t. n) and -2.2e-185 (w.r.t. M), and
        # what float64 has left after the rewrite is cancellation noise between
        # two terms of order 100 — around 1e-14. That is not the true number, but
        # it is the same "indistinguishable from zero" the column would carry
        # anyway, thirteen orders below the model's O(1) sensitivities, where a
        # NaN stops the run outright.
        assert abs(rewritten) < 1e-13
        assert abs(_exact(deriv, MEASURED)) < 1e-180  # ...and the truth really is ~0

    def test_the_half_saturation_column_gains_its_value_rather_than_keeping_it(self):
        """``∂/∂K`` was not NaN — it was ``-0.0``, an underflow of a numerator
        that the rewrite never forms. Dividing through leaves the true
        ``-1.996e-205`` standing to the last digit float64 has, so this column
        goes from a silent zero to a measured derivative."""
        deriv = sp.diff(HILL, K)
        truth = _exact(deriv, MEASURED)

        assert _at(deriv, **MEASURED) == 0.0
        assert _relative_error(_at(_rewrite_saturating_ratio(deriv), **MEASURED), truth) < 1e-15

    def test_a_symbol_outside_the_ratio_is_untouched(self):
        """``∂/∂zeta_1`` is the saturated fraction itself — ``1`` — and was
        always fine. Nothing about it moves."""
        deriv = sp.diff(HILL, z)
        assert _at(deriv, **MEASURED) == 1.0
        assert _at(_rewrite_saturating_ratio(deriv), **MEASURED) == 1.0


class TestTheIdentity:
    def test_the_ratio_itself_saturates_instead_of_going_nan(self):
        """``k = 1`` — the Hill ratio, not a derivative of it. CVODES reaches it
        through the Jacobian of any rate law that multiplies by one, so leaving
        it out would fix ∂f/∂p and let the same NaN back in one term later."""
        expr = x**n / (K**n + x**n)
        rewritten = _rewrite_saturating_ratio(expr)
        big = {"x": 1e40, "n": 10.0, "K": 2.0}  # x^n = 1e400 → inf

        assert np.isnan(_at(expr, **big))
        assert _at(rewritten, **big) == 1.0
        assert _at(rewritten, x=0.0, n=10.0, K=2.0) == 0.0  # the other end
        assert _at(rewritten, x=3.0, n=2.5, K=2.0) == pytest.approx(
            _at(expr, x=3.0, n=2.5, K=2.0), rel=1e-15
        )

    def test_the_state_direction_is_covered_through_the_folded_square(self):
        """``sp.diff`` writes the quotient rule's ``x^n·x^n`` as ``x^(2n)``, so
        the numerator overflows one square root before the fraction does — at
        ``x^n = 1e154``, not ``1e308``. It is matched by base and integer
        exponent ratio, not by identity, which is what recognises ``x^(2n)``
        beside ``x^n``.

        The measured model's own base, ``1/mTOR_R``, because that is the one that
        reaches the emitters: see
        :meth:`TestWhatIsLeftAlone.test_a_folded_removable_denominator_is_left_to_gh_402`.
        """
        deriv = sp.diff(HILL, M)
        assert deriv.has((1 / M) ** (2 * n))  # the shape this test is about

        assert np.isnan(_at(deriv, **MEASURED))
        assert _at(_rewrite_saturating_ratio(deriv), **MEASURED) == 0.0

    def test_two_bases_over_one_sum_are_each_divided_out(self):
        """``x^n·y^n/(x^n + y^n)^2`` is the product of two saturating terms, and
        each has its own overflowing factor. One numerator being spent does not
        spend the denominator."""
        expr = sp.Mul(x**n, y**n, sp.Pow(x**n + y**n, -2), evaluate=False)
        rewritten = _rewrite_saturating_ratio(expr)
        point = {"x": 1e40, "y": 2.0, "n": 10.0}  # x^n = inf, y^n = 1024

        assert np.isnan(_at(expr, **point))
        assert _at(rewritten, **point) == 0.0
        ordinary = {"x": 1.5, "y": 2.0, "n": 3.0}
        assert _at(rewritten, **ordinary) == pytest.approx(_at(expr, **ordinary), rel=1e-14)

    def test_it_does_not_lose_accuracy_where_the_old_form_was_finite(self):
        """The rewrite spends a division the old form did not — ``a/f`` — so the
        question is whether ordinary Hill points pay for it. Scored against an
        exact evaluation at 40 digits, over every symbol of the ratio, they do
        not.

        Two statements, because a bare "new is always closer" would be false and
        would not mean much if it were true. Away from saturation both forms are
        the *same* difference of two nearly-equal terms, and where that
        difference cancels neither has digits left — the point that used to fail
        this assertion had a true value of ``1e-23`` with the old form at ``0``
        and the new at ``-7e-15``, which is two spellings of "no information",
        not a regression. So:

        * **wherever the old form had digits, the new one keeps them.** A point
          the old form got to within ``1e-10`` relative is a point where the
          expression is well conditioned, and there the new form is never worse.
        * **head to head, the new form wins more often than it loses.**
        """
        rng = np.random.default_rng(20250817)
        pairs = [
            (sp.diff(HILL, t), _rewrite_saturating_ratio(sp.diff(HILL, t))) for t in (n, M, K)
        ]

        well_conditioned = new_closer = old_closer = 0
        for _ in range(40):
            point = {
                "M": float(10 ** rng.uniform(-3, 3)),
                "n": float(rng.uniform(0.5, 6)),
                "K": float(10 ** rng.uniform(-2, 2)),
                "z": float(rng.uniform(0.1, 10)),
            }
            for raw, rewritten in pairs:
                truth = _exact(raw, point)
                old, new = _at(raw, **point), _at(rewritten, **point)
                if truth == 0 or not (np.isfinite(old) and np.isfinite(new)):
                    continue
                old_err, new_err = _relative_error(old, truth), _relative_error(new, truth)
                if old_err < 1e-10:
                    well_conditioned += 1
                    assert new_err < 1e-10, (point, old, new, truth)
                new_closer += new_err < old_err
                old_closer += old_err < new_err

        assert well_conditioned > 50  # the loop really did compare something
        assert new_closer > old_closer


class TestTheOrderAgainstTheZeroBaseGuard:
    """The two rewrites want the same ``x^n``, at opposite ends of its range, so
    which runs first decides whether one of them gets it.

    :func:`_guard_exponent_log_at_zero` (GH #310/#317) replaces ``x^n·ln x`` with
    its limit at ``x = 0``; this one divides through at ``x^n → ∞``. Both
    emitters now guard first. Going the other way round hands the guard an
    ``ln x/(1 + K^n·x^-n)`` whose factors are ``−inf`` and ``0`` at ``x = 0``,
    which is a NaN where the guarded product is a clean ``0`` — and
    ``test_exponent_log_zero_base.py``'s reproducer is exactly that shape, with a
    species whose initial condition *is* zero.
    """

    DERIV = sp.diff(x**n / (K**n + x**n), n)
    ZERO_BASE = {"x": 0.0, "n": 3.0, "K": 2.0}
    OVERFLOWING = {"x": 2.1819350938676658e20, "n": 10.0, "K": 0.5}

    def _emitted(self, point):
        text = sympy_to_exprtk(self.DERIV)
        assert text is not None
        # ExprTk's `if(c,a,b)` is a call, not Python's conditional expression.
        py = text.replace("if(", "_if(").replace("^", "**")
        env = {"log": np.log, "_if": lambda c, a, b: a if c else b}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float(eval(py, env, {k: np.float64(v) for k, v in point.items()}))  # noqa: S307

    def test_a_base_of_exactly_zero_still_reads_zero(self):
        assert self._emitted(self.ZERO_BASE) == 0.0

    def test_and_the_overflowing_base_is_still_finite(self):
        assert self._emitted(self.OVERFLOWING) == 0.0

    def test_the_guard_keeps_the_power_it_claimed(self):
        """Structurally: the logarithm's own power is inside the ``Piecewise``,
        not divided out into a reciprocal beside it."""
        from bngsim._jacobian import (
            _guard_exponent_log_at_zero,
            _remove_removable_power_denominators,
        )

        guarded = _guard_exponent_log_at_zero(_remove_removable_power_denominators(self.DERIV))
        claimed = {p for p in guarded.atoms(sp.Piecewise) if p.has(sp.log(x))}
        assert claimed  # the guard really did claim the logarithm's power
        assert all(p.args[1][0].has(x**n) for p in claimed)

        after = _rewrite_saturating_ratio(guarded)
        assert claimed <= after.atoms(sp.Piecewise)  # carried through untouched


class TestWhatIsLeftAlone:
    def test_a_bare_saturable_term_is_not_touched(self):
        """``S/(K + S)`` is Michaelis-Menten, and no state a solver can hold
        makes it ``inf/inf``. Rewriting it would spend a division to buy
        nothing — and would divide by a species that is routinely ``0``."""
        expr = S / (K + S)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_bare_saturable_term_squared_is_not_touched_either(self):
        """The same, through the ``f^m`` branch: ``S`` is neither an exponential
        nor a power, so ``S^2`` beside ``K + S`` is not a whole power of an
        overflowing factor."""
        expr = sp.Mul(S**2, sp.Pow(K + S, -2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_numerator_that_is_not_a_whole_power_is_not_touched(self):
        """``x^(n+1)`` beside ``x^n`` is ``x^n·x``, not ``(x^n)^m`` — the
        exponent ratio ``(n + 1)/n`` is not an integer, so there is no ``f`` to
        divide through by."""
        expr = sp.Mul(x ** (n + 1), sp.Pow(K + x**n, -2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_numeric_exponent_is_not_worth_a_division(self):
        """``x^2/(K + x^2)`` needs ``x > 1e154`` to be ``inf/inf``, and no state
        a solver can hold gets there — where ``x^n`` with a Hill exponent of 10
        needs only ``x > 1e31``. So the numerator has to be an exponential or a
        power whose exponent is *not* a plain number; anything else would spend a
        division on a case that does not arise."""
        expr = sp.Mul(x**2, sp.Pow(K + x**2, -1), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_positive_power_is_not_touched(self):
        """``x^n·(K + x^n)^2`` really is ``+inf`` at a large ``x^n``. There is no
        cancellation to arrange and no limit to take."""
        expr = sp.Mul(x**n, sp.Pow(K + x**n, 2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_denominator_whose_rest_holds_the_same_power_is_refused(self):
        """``x^n/(y·x^n + x^n)^2``: the ``rest`` here is ``y·x^n``, so dividing
        through would trade one overflow for another."""
        expr = sp.Mul(x**n, sp.Pow(y * x**n + x**n, -1), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_different_base_is_not_a_carrier(self):
        """``K^n`` beside ``x^n`` shares the exponent, not the base. Dividing
        through by the wrong factor would not be an identity."""
        expr = sp.Mul(K**n, sp.Pow(y + x**n, -1), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_folded_removable_denominator_is_left_to_gh_402(self):
        """The state derivative of the *plainest* Hill ratio is not covered, and
        the reason is upstream of this rewrite.

        ``x^(2n)/x`` is a removable ``0/0`` at ``x = 0``, so GH #96/#351 folds it
        into ``x^(2n-1)`` before this rewrite ever runs — and ``x^(2n-1)`` is not
        a whole power of the ``x^n`` in the denominator's sum, so there is no
        ``f`` to divide through by. The emitted C is still NaN at ``x^(2n-1) =
        inf``, where the true value is ``1e-224``.

        Reordering the two rewrites matches the shape and reintroduces the
        ``0/0`` at ``x = 0`` that GH #96 removed, so it is a trade rather than a
        fix, measured in GH #402. This test pins the boundary rather than the
        defect: it fails the day the fold and the divide-through learn to
        compose, and that is the day GH #402 closes.
        """
        from bngsim._jacobian import _remove_removable_power_denominators

        folded = _remove_removable_power_denominators(sp.diff(x**n / (K**n + x**n), x))
        assert folded.has(x ** (2 * n - 1))  # the fold this test is about
        assert _rewrite_saturating_ratio(folded) == folded

    def test_an_expression_with_no_negative_power_of_a_sum_comes_back_identical(self):
        """The gate that keeps this off every derivative in the corpus that
        cannot need it — and keeps the object itself, not a rebuilt equal."""
        expr = z * x**n * sp.log(x)
        assert _rewrite_saturating_ratio(expr) is expr


class TestTheNativeMirror:
    """``bngsim._saturable_jacobian`` differentiates the saturable rate-law
    family without SymPy and emits its own C, so it does not see the rewrite
    above. It is the ``J·yS`` half of the analytic sensitivity RHS and the whole
    of the analytical Jacobian, so leaving it out fixes ∂f/∂p and lets the same
    NaN back in one term later — the GH #336 situation, one family over."""

    LAW = "vmax*S^nH/(KH^nH + S^nH)"
    CONSTS = frozenset({"vmax", "nH", "KH"})
    # `_diff` leaves `vmax*S^nH/(KH^nH + S^nH)` standing as its own factor of the
    # quotient rule, and that factor is inf/inf once S^nH overflows. nH = 8 puts
    # S^nH = 1e320 over the line while S^(nH-1) = 1e280 stays under it, so the
    # *only* non-finite thing in the derivative is the ratio under test.
    # np.float64 throughout: Python's own `**` raises OverflowError instead of
    # returning `inf`, so a bare float would never reach the arithmetic under
    # test. C and ExprTk both overflow silently, as numpy does.
    POINT = {
        "vmax": np.float64(1.5),
        "nH": np.float64(8.0),
        "KH": np.float64(2.0),
        "S": np.float64(1e40),
    }

    def _dS(self):
        return _sat.differentiate_rate_law_native(self.LAW, {}, {"S"}, self.CONSTS)

    def test_the_native_jacobian_entry_is_finite_where_the_power_overflows(self):
        derivs = self._dS()
        assert derivs is not None
        node = derivs["S"]

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = eval(  # noqa: S307 - this module's own emitted text
                _sat._emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, dict(self.POINT)
            )
            fixed = eval(  # noqa: S307
                _sat.emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, dict(self.POINT)
            )
        assert np.isnan(raw)
        assert np.isfinite(fixed)

    def test_the_native_rewrite_is_the_same_function_away_from_the_overflow(self):
        node = self._dS()["S"]
        near = dict(self.POINT, S=np.float64(2.2))  # just past half-saturation

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = eval(  # noqa: S307
                _sat._emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, near
            )
            fixed = eval(  # noqa: S307
                _sat.emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, near
            )
        assert np.isfinite(raw) and raw != 0.0
        assert fixed == pytest.approx(raw, rel=1e-14)

    def test_a_law_with_no_power_over_a_sum_emits_the_same_text_as_before(self):
        """The gate: nothing in the family that cannot overflow this way moves.
        ``S`` alone is not a factor this rewrite will divide through by."""
        derivs = _sat.differentiate_rate_law_native(
            "vmax*S/(KM + S)", {}, {"S"}, frozenset({"vmax", "KM"})
        )
        node = derivs["S"]
        assert _sat.emit_exprtk(node) == _sat._emit_exprtk(node)


class TestThroughTheEmitters:
    """The rewrite is applied on the way out of both printers, so what the
    engine and the compiler actually evaluate is the rewritten text.

    The derivative used here is the *state* one. It is the half that reaches the
    Jacobian and the ``J·yS`` term, and unlike the exponent's it carries no
    logarithm — so the emitted text holds no zero-base guard (GH #310/#317),
    whose C ternaries and ExprTk ``if()`` neither ``eval`` would parse. The
    exponent's derivative goes through both printers too, in
    ``TestTheShapeThroughASolve`` below, where the engine reads them.
    """

    POINT = {k: np.float64(v) for k, v in MEASURED.items()}

    def _deriv(self):
        return sp.diff(HILL, M)

    def test_the_emitted_c_is_finite_at_the_overflowing_argument(self):
        src = sympy_to_c(self._deriv(), lambda name: name)
        assert src is not None
        assert "pow" in src

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = eval(  # noqa: S307 - the input is this module's own emitted C
                src, {"pow": np.float_power, "log": np.log, "exp": np.exp}, dict(self.POINT)
            )
        assert float(value) == 0.0

    def test_the_emitted_exprtk_is_finite_at_the_overflowing_argument(self):
        text = sympy_to_exprtk(self._deriv())
        assert text is not None

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = eval(  # noqa: S307 - the input is this module's own emitted text
                text.replace("^", "**"), {"log": np.log, "exp": np.exp}, dict(self.POINT)
            )
        assert float(value) == 0.0


# ─── the shape through a solve ─────────────────────────────────────────────

# The measured model in miniature: a Hill ratio whose base is the reciprocal of a
# very small species, with the Hill exponent as the sensitivity parameter.
# `(1/Mtot)^n_1` is 2.45e203 — finite — and its square is not, which is the whole
# defect. `M` has no reactions, so it holds 4.58308775e-21 for the run and the
# ratio stays saturated at 1, exactly as in `BIOMD0000000829`.
HILL_DOSE = """\
begin parameters
    1 n_1   10.0            # Constant
    2 K_m   0.5             # Constant
    3 zeta  2.5             # Constant
    4 ks    1.0             # Constant
end parameters
begin functions
    1 mass_s()  zeta*(1/Mtot)^n_1/(K_m^n_1 + (1/Mtot)^n_1)
    2 grow()    ks*mass_s()
end functions
begin species
    1 A() 0.0
    2 M() 4.58308775e-21
end species
begin reactions
    1 0 1 grow #_R1
end reactions
begin groups
    1 Atot 1
    2 Mtot 2
end groups
"""


class TestTheShapeThroughASolve:
    @pytest.fixture
    def hill_net(self, tmp_path):
        net = tmp_path / "hill_dose.net"
        net.write_text(HILL_DOSE)
        return net

    def test_the_exponent_column_is_finite_from_the_first_output_point(self, hill_net):
        """The end-to-end case. The trajectory was always fine — the ratio is a
        clean ``1`` — so only the sensitivity solve shows the defect, and without
        the rewrite it does not merely return NaN: CVODES refuses the run at the
        first call of the sensitivity RHS (GH #395)."""
        import bngsim

        model = bngsim.Model.from_net(hill_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["n_1"])
        result = sim.run(t_span=(0, 10), n_points=11, rtol=1e-9, atol=1e-12)

        species = np.asarray(result.species)
        assert np.all(np.isfinite(species))
        # The ratio is saturated, so A grows at zeta*ks = 2.5 per unit time.
        assert species[-1][0] == pytest.approx(25.0, rel=1e-6)

        sens = np.asarray(result.sensitivities)
        assert np.all(np.isfinite(sens))
