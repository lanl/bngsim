"""GH #388 — ``exp(u)/(a + exp(u))^n`` must not be evaluated as ``inf/inf``.

A sigmoid is written ``1/(1 + exp(-k·(t - t0)))`` in essentially every dose
schedule in BioModels, and differentiating one with respect to *any* of the
parameters in its exponent produces ``c·exp(u)/(1 + exp(u))^2``. That function
is bounded by ``1/4`` everywhere on the real line and decays to ``0`` at both
ends. Evaluated literally it is **NaN** the moment ``u`` passes 709, because
``exp`` overflows to ``inf`` in both the numerator and the denominator and
``inf/inf`` is NaN — and a schedule reaches ``u = 709`` without trying, since
``u`` is a steepness times an onset time:

* ``BIOMD0000000636``   ``DAdip_steepness · DAdip_onset = 100 · 10``
* ``BIOMD0000000554``   ``sr_GLY · (to + to_GLY)       = 4 · 283``
* ``MODEL1701170000/1`` the same construct, from the same authors

The value path never notices. ``1/(1 + inf)`` is a clean ``0``, so the
trajectory is exact and only the differentiated form fails — which is why all
four models handed back a non-finite forward-sensitivity column at the very
first output point while their ODE runs matched the reference.

:func:`bngsim._jacobian._rewrite_saturating_ratio` divides the ratio
through by ``exp(u)``::

    exp(u)/(a + exp(u))^n  ==  1/(1 + a·exp(-u)) · (a + exp(u))^(1-n)

which is an identity for every finite ``u`` and has no intermediate that can
overflow into a ratio of infinities: one factor saturates as ``u → +∞`` and the
other as ``u → -∞``, so whichever end overflows contributes a ``0`` or a ``1``
instead of an ``inf`` that has to be cancelled against another.

It lives at the same chokepoint as the zero-base logarithm guard (GH #310/#317)
and the removable-power-denominator rewrite (GH #96/#351), so the ExprTk
evaluator and every codegen backend get it from one place.

GH #393 generalised the same rewrite to a *power* — ``x^n/(K^n + x^n)``, which
overflows the identical way and is every Hill function there is. This file keeps
the sigmoid half; ``test_saturating_power_ratio.py`` is the other, and between
them they hold the two shapes of the one ``f`` the rewrite divides through by.

Issue #549 found that ``BIOMD0000000554`` had never actually been fixed, and
``BIOMD0000000627`` alongside it. Two spellings of the same sigmoid reached the
match and were refused by it, and the tests above missed both because they write
the sigmoid the way a person would rather than the way a loader does:

* **A constant shift in the exponent.** ``exp(-4.59186·(t - (t_0 + t_1 - 3)))``
  multiplies out to an exponent with a number in it, and sympy folds that number
  into a factor out front — ``960856.16·exp(u)``. The denominator summand is then
  a ``Mul`` and not the bare ``exp`` the match looked for.
* **A ``Float`` where an ``Integer`` was expected.** libSBML writes ``1.0``, so an
  SBML-sourced exponent is ``-1.0·sr·(t - to)``; the exponent ratio that decides
  the match comes out ``Float(1.0)``, and ``sp.Float(1.0) == 1`` is False.

Either one alone was enough to hand back a NaN sensitivity column at ``t = 0``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim import _saturable_jacobian as _sat  # noqa: E402
from bngsim._jacobian import (  # noqa: E402
    _integer_at_least,
    _rewrite_saturating_ratio,
    sympy_to_c,
    sympy_to_exprtk,
)

k, t, t0, a, u, c = sp.symbols("k t t0 a u c")


def _at(expr, **values):
    """Evaluate ``expr`` numerically, with the overflow warnings quieted."""
    names = sorted(values, key=str)
    f = sp.lambdify([sp.Symbol(n) for n in names], expr, "numpy")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(f(*(np.float64(values[n]) for n in names)))


SIGMOID = 1 / (1 + sp.exp(-k * (t - t0)))


class TestTheDerivativeThatUsedToOverflow:
    """Each parameter of the sigmoid, differentiated, at an argument big enough
    to overflow ``exp``. The true value is a small number or zero in every case;
    the unrewritten form is NaN in every case."""

    @pytest.mark.parametrize("wrt", [k, t0])
    @pytest.mark.parametrize(
        ("kv", "tv", "t0v"),
        [
            (100.0, 0.0, 10.0),  # BIOMD0000000636's steepness · onset
            (4.0, 0.0, 283.0),  # BIOMD0000000554's sr_GLY · (to + to_GLY)
            (100.0, 5.0, 20.0),  # ...and a steepness deep inside the off state
        ],
    )
    def test_the_raw_form_is_nan_and_the_rewritten_one_is_the_limit(self, wrt, kv, tv, t0v):
        deriv = sp.diff(SIGMOID, wrt)
        point = {"k": kv, "t": tv, "t0": t0v}

        assert np.isnan(_at(deriv, **point))
        assert _at(_rewrite_saturating_ratio(deriv), **point) == 0.0

    @pytest.mark.parametrize("wrt", [k, t0])
    def test_it_is_the_same_function_where_the_raw_form_was_finite(self, wrt):
        """The rewrite is exact, not an approximation that happens to be
        well-behaved: away from the overflow the two agree to the last ulp the
        reassociation allows."""
        deriv = sp.diff(SIGMOID, wrt)
        rewritten = _rewrite_saturating_ratio(deriv)

        for kv, tv, t0v in [(1.0, 1.0, 0.5), (0.3, 7.0, 2.0), (2.0, 0.0, 1.5), (5.0, 3.2, 3.0)]:
            point = {"k": kv, "t": tv, "t0": t0v}
            raw = _at(deriv, **point)
            assert raw != 0.0  # the interesting comparison, not 0 == 0
            assert _at(rewritten, **point) == pytest.approx(raw, rel=1e-15)

    def test_the_sigmoid_itself_is_covered(self):
        """``n = 1`` — ``exp(u)/(a + exp(u))`` rather than the squared
        denominator. CVODES reaches this through the Jacobian of any rate law
        that multiplies by a sigmoid, so leaving it out would fix the ∂f/∂p half
        and not the J·yS half."""
        expr = sp.exp(u) / (1 + sp.exp(u))
        rewritten = _rewrite_saturating_ratio(expr)

        assert np.isnan(_at(expr, u=800.0))
        assert _at(rewritten, u=800.0) == 1.0
        assert _at(rewritten, u=-800.0) == 0.0
        assert _at(rewritten, u=0.5) == pytest.approx(_at(expr, u=0.5), rel=1e-15)

    def test_a_constant_other_than_one_is_carried_through(self):
        """``a`` is whatever the model wrote — the rewrite multiplies it by
        ``exp(-u)`` rather than assuming the textbook ``1``."""
        expr = sp.exp(u) / (a + sp.exp(u)) ** 2
        rewritten = _rewrite_saturating_ratio(expr)

        assert np.isnan(_at(expr, u=800.0, a=3.0))
        assert _at(rewritten, u=800.0, a=3.0) == 0.0
        assert _at(rewritten, u=1.25, a=3.0) == pytest.approx(_at(expr, u=1.25, a=3.0), rel=1e-15)


class TestASummandWithANumericCoefficient:
    """``B·exp(u) + 1`` rather than ``exp(u) + 1`` (issue #549).

    ``BIOMD0000000627`` shifts its schedule by a constant —
    ``exp(-4.59186·(t - (t_0 + t_1 - 3)))`` — and that constant multiplies out
    into the exponent, where sympy folds it into a factor out front:
    ``exp(u + 13.78)`` is ``960856.16·exp(u)``. So the summand standing under the
    numerator is a ``Mul``, the match skipped it, and the derivative went out in
    the very ``inf/inf`` form this rewrite exists to remove.
    """

    # exp(4.59186·3) — BIOMD0000000627's own shift, folded.
    B = 960856.1606434912

    def test_the_coefficient_form_used_to_be_nan_and_is_now_the_limit(self):
        expr = sp.exp(u) / (self.B * sp.exp(u) + 1) ** 2
        rewritten = _rewrite_saturating_ratio(expr)

        assert np.isnan(_at(expr, u=800.0))
        assert _at(rewritten, u=800.0) == 0.0
        assert _at(rewritten, u=-800.0) == 0.0

    def test_it_is_the_same_function_where_the_raw_form_was_finite(self):
        """Exact, not merely well-behaved: ``c^-m`` against ``rest/c`` is an
        identity for any nonzero ``c``, so away from the overflow the two forms
        agree to the last ulp the reassociation allows."""
        expr = sp.exp(u) / (self.B * sp.exp(u) + 1) ** 2
        rewritten = _rewrite_saturating_ratio(expr)

        for uv in (-9.0, -1.0, 0.0, 0.25, 4.0, 11.0):
            raw = _at(expr, u=uv)
            assert raw != 0.0  # the interesting comparison, not 0 == 0
            assert _at(rewritten, u=uv) == pytest.approx(raw, rel=1e-14)

    def test_the_model_law_itself_at_the_argument_the_run_starts_on(self):
        """The whole chain rather than the shape alone: one sigmoid of
        ``BIOMD0000000627``'s ``f_CBF_dyn``, differentiated w.r.t. the onset it
        shifts, at ``t = 0``. ``4.59186·199`` is past ``ln(DBL_MAX) ≈ 709.8``, and
        the term that used to be NaN there stands for ``1e-397``."""
        law = 1 / (1 + sp.exp(-sp.Float(4.59186) * (t - (t0 + 3))))
        deriv = sp.diff(law, t0)
        point = {"t": 0.0, "t0": 200.0}

        assert np.isnan(_at(deriv, **point))
        assert _at(_rewrite_saturating_ratio(deriv), **point) == 0.0

    def test_a_negative_coefficient_is_carried_the_same_way(self):
        """``c^-m`` is a sign as much as a scale. It moves no pole either: the
        denominator vanishes where it always did, since both sides of
        ``rest + c·f`` are divided by the same ``c``."""
        expr = sp.exp(u) / (1 - 3 * sp.exp(u)) ** 2
        rewritten = _rewrite_saturating_ratio(expr)

        assert np.isnan(_at(expr, u=800.0))
        assert _at(rewritten, u=800.0) == 0.0
        for uv in (-5.0, 0.0, 2.0):  # clear of the pole at u = -ln 3
            assert _at(rewritten, u=uv) == pytest.approx(_at(expr, u=uv), rel=1e-14)

    def test_a_symbolic_coefficient_is_refused(self):
        """``a·exp(u) + 1`` is left alone. The rewrite would divide by ``a``,
        which is a value the original expression is perfectly happy at, so the
        coefficient has to be one that is known nonzero when the C is emitted — a
        number, which a ``Mul`` cannot hold as a zero."""
        expr = sp.Mul(sp.exp(u), sp.Pow(a * sp.exp(u) + 1, -2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr


class TestAFloatWhereAnIntegerWasExpected:
    """``-1.0·k·(t - t0)`` rather than ``-k·(t - t0)`` (issue #549).

    libSBML spells its coefficients ``1.0``, so the same law carries a ``Float``
    when it is loaded from SBML and an ``Integer`` when it is hand-written in a
    ``.net``. The exponent ratio that decides the match is then ``Float(1.0)``,
    and ``sp.Float(1.0) == 1`` is **False** — sympy holds a Float and an Integer
    to be different numbers however equal their values. So
    :func:`~bngsim._jacobian._integer_at_least` answered ``None`` for every Float
    it was ever handed, the rewrite declined every SBML-sourced sigmoid there is,
    and nothing looked wrong because a refused match simply leaves the expression
    as it was.
    """

    @pytest.mark.parametrize("one", [sp.Integer(1), sp.Float(1.0)])
    def test_the_sigmoid_is_matched_however_its_coefficient_is_spelled(self, one):
        exponent = -one * k * (t - t0)
        expr = sp.exp(exponent) / (one + sp.exp(exponent)) ** 2
        rewritten = _rewrite_saturating_ratio(expr)
        # BIOMD0000000554's sr_GLY · (to + to_GLY) = 4 · 283, at the run's start.
        point = {"k": 4.0, "t": 0.0, "t0": 283.0}

        assert np.isnan(_at(expr, **point))
        assert _at(rewritten, **point) == 0.0
        assert _at(rewritten, k=4.0, t=283.5, t0=283.0) == pytest.approx(
            _at(expr, k=4.0, t=283.5, t0=283.0), rel=1e-14
        )

    def test_the_helper_counts_a_float_as_the_integer_it_is(self):
        """The contract its docstring has always stated, which ``==`` never
        delivered."""
        assert _integer_at_least(sp.Float(2.0), 1) == 2
        assert _integer_at_least(sp.Float(1.0), 1) == 1
        assert _integer_at_least(sp.Integer(2), 1) == 2
        assert _integer_at_least(sp.Integer(-1), -1) == -1

    def test_it_still_refuses_everything_that_is_not_a_count(self):
        """Widening the equality must not widen the acceptance: a half-integer is
        no more an exponent multiple than it was, a magnitude no double holds is
        refused rather than raised on, and anything without a value at all is
        still declined — an exception here aborts the whole codegen build."""
        assert _integer_at_least(sp.Rational(3, 2), 1) is None
        assert _integer_at_least(sp.Float(2.5), 1) is None
        assert _integer_at_least(sp.Float(2.0), 3) is None  # below the minimum
        assert _integer_at_least(sp.Symbol("n"), 1) is None
        assert _integer_at_least(2 * sp.I, 1) is None
        assert _integer_at_least(sp.Integer(10) ** 400, 1) is None


class TestWhatIsLeftAlone:
    def test_a_positive_power_is_not_touched(self):
        """``exp(u)·(a + exp(u))^2`` really is ``+inf`` at ``u = 800``. There is
        no cancellation to arrange and no limit to take."""
        expr = sp.Mul(sp.exp(u), sp.Pow(a + sp.exp(u), 2), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_denominator_that_does_not_hold_the_numerator_is_not_touched(self):
        """``exp(u)/(a + exp(2u))`` is not this shape: dividing through by
        ``exp(u)`` leaves ``exp(u)`` in the denominator and buys nothing."""
        expr = sp.Mul(sp.exp(u), sp.Pow(a + sp.exp(2 * u), -1), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_a_denominator_whose_rest_holds_the_same_exponential_is_refused(self):
        """``exp(u)/(x·exp(u) + exp(u))^2``: the ``rest`` here is ``x·exp(u)``,
        so dividing through would trade one overflow for another."""
        x = sp.Symbol("x")
        expr = sp.Mul(sp.exp(u), sp.Pow(x * sp.exp(u) + sp.exp(u), -1), evaluate=False)
        assert _rewrite_saturating_ratio(expr) == expr

    def test_an_expression_with_neither_an_exponential_nor_a_power_over_the_sum(self):
        """``c·t^2/(a + t)^2`` has the negative power of a sum this rewrite looks
        for, and nothing in that sum it will divide through by: ``t`` is neither
        an exponential nor a power, so no finite state can make the ratio
        ``inf/inf`` and dividing by it would buy nothing (GH #393). Returned
        identically — the same object, not a rebuilt equal."""
        expr = c * t**2 / (a + t) ** 2
        assert _rewrite_saturating_ratio(expr) is expr


class TestTheNativeMirror:
    """``bngsim._saturable_jacobian`` differentiates the saturable rate-law
    family without SymPy and emits its own C, so it does not see the rewrite
    above. It is the ``J·yS`` half of the analytic sensitivity RHS and the whole
    of the analytical Jacobian, so leaving it out fixes ∂f/∂p and lets the same
    NaN back in one term later — the GH #336 situation, one family over."""

    LAW = "vmax/(1 + exp(-kswitch*(S - tswitch)))"
    CONSTS = frozenset({"vmax", "kswitch", "tswitch"})
    POINT = {"vmax": 1.5, "kswitch": 100.0, "tswitch": 10.0, "S": 1.0}

    def _dS(self):
        return _sat.differentiate_rate_law_native(self.LAW, {}, {"S"}, self.CONSTS)

    def test_the_native_jacobian_entry_is_finite_where_exp_overflows(self):
        derivs = self._dS()
        assert derivs is not None
        node = derivs["S"]

        with np.errstate(over="ignore", invalid="ignore"):
            raw = eval(  # noqa: S307 - this module's own emitted text
                _sat._emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, dict(self.POINT)
            )
            fixed = eval(  # noqa: S307
                _sat.emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, dict(self.POINT)
            )
        assert np.isnan(raw)
        assert fixed == 0.0

    def test_the_native_rewrite_is_the_same_function_away_from_the_overflow(self):
        node = self._dS()["S"]
        near = dict(self.POINT, S=10.02)

        with np.errstate(over="ignore", invalid="ignore"):
            raw = eval(  # noqa: S307
                _sat._emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, near
            )
            fixed = eval(  # noqa: S307
                _sat.emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, near
            )
        assert np.isfinite(raw) and raw != 0.0
        assert fixed == pytest.approx(raw, rel=1e-14)

    def test_a_law_with_no_exponential_emits_the_same_text_as_before(self):
        """The gate: nothing in the family that cannot overflow this way moves."""
        derivs = _sat.differentiate_rate_law_native(
            "vmax*S/(KM + S)", {}, {"S"}, frozenset({"vmax", "KM"})
        )
        node = derivs["S"]
        assert _sat.emit_exprtk(node) == _sat._emit_exprtk(node)

    # ── a summand with a coefficient, on this side of the pair (issue #549) ──
    #
    # Here it takes a law that writes the coefficient itself. The sympy twin
    # reaches the same shape from a plain sigmoid, because sympy folds a constant
    # in the exponent into one; this module parses the text and folds nothing.
    COEFF_LAW = "vmax/(1 + 5*exp(-kswitch*(S - tswitch)))"

    def _coeff_dS(self):
        return _sat.differentiate_rate_law_native(self.COEFF_LAW, {}, {"S"}, self.CONSTS)["S"]

    def test_a_coefficient_on_the_summand_is_divided_through_too(self):
        node = self._coeff_dS()

        with np.errstate(over="ignore", invalid="ignore"):
            raw = eval(  # noqa: S307 - this module's own emitted text
                _sat._emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, dict(self.POINT)
            )
            fixed = eval(  # noqa: S307
                _sat.emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, dict(self.POINT)
            )
        assert np.isnan(raw)
        assert fixed == 0.0

    def test_the_coefficient_form_is_the_same_function_away_from_the_overflow(self):
        node = self._coeff_dS()
        near = dict(self.POINT, S=10.02)

        with np.errstate(over="ignore", invalid="ignore"):
            raw = eval(  # noqa: S307
                _sat._emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, near
            )
            fixed = eval(  # noqa: S307
                _sat.emit_exprtk(node).replace("^", "**"), {"exp": np.exp}, near
            )
        assert np.isfinite(raw) and raw != 0.0
        assert fixed == pytest.approx(raw, rel=1e-14)

    def test_what_the_coefficient_peeler_will_and_will_not_take(self):
        """Either side of the product, and a number only. A symbolic coefficient
        is a value the expression is happy at — zero included — so dividing
        through by it would trade a NaN at one argument for a division by zero at
        another; a literal ``0`` cannot be divided by at all. Both are handed back
        whole, which leaves the summand looking like the ``Mul`` it is and stops
        the rewrite there."""
        f = _sat._mk_call("exp", ("var", "w"))
        five = _sat._num(5.0)

        assert _sat._numeric_coefficient(f) == (_sat._ONE, f)
        assert _sat._numeric_coefficient(("*", five, f)) == (five, f)
        assert _sat._numeric_coefficient(("*", f, five)) == (five, f)
        for refused in (("*", ("var", "a"), f), ("*", _sat._num(0.0), f)):
            assert _sat._numeric_coefficient(refused) == (_sat._ONE, refused)

    def test_a_summand_without_a_coefficient_costs_nothing(self):
        """``1`` is what the peeler hands back for a bare summand, and the smart
        constructors fold a division by it away — so every expression the rewrite
        already handled keeps the text it had, not merely the value."""
        f = _sat._mk_call("exp", ("var", "w"))
        coeff, peeled = _sat._numeric_coefficient(f)
        assert peeled is f
        assert _sat._mk_div(f, coeff) is f
        assert _sat._mk_div(("var", "rest"), coeff) == ("var", "rest")


class TestThroughTheEmitters:
    """The rewrite is applied on the way out of both printers, so what the
    engine and the compiler actually evaluate is the rewritten text."""

    def test_the_emitted_c_is_finite_at_the_overflowing_argument(self):
        deriv = sp.diff(SIGMOID, k)
        src = sympy_to_c(deriv, lambda name: name)
        assert src is not None

        with np.errstate(over="ignore", invalid="ignore"):
            value = eval(  # noqa: S307 - the input is this module's own emitted C
                src, {"exp": np.exp, "pow": np.float_power}, {"k": 100.0, "t": 0.0, "t0": 10.0}
            )
        assert value == 0.0

    def test_the_emitted_exprtk_is_finite_at_the_overflowing_argument(self):
        deriv = sp.diff(SIGMOID, k)
        text = sympy_to_exprtk(deriv)
        assert text is not None
        assert "exp" in text

        with np.errstate(over="ignore", invalid="ignore"):
            value = eval(  # noqa: S307 - the input is this module's own emitted text
                text.replace("^", "**"),
                {"exp": np.exp},
                {"k": 100.0, "t": 0.0, "t0": 10.0},
            )
        assert float(value) == 0.0


# ─── the shape through a solve ─────────────────────────────────────────────

# A steep sigmoid dose, switched by a state variable that ramps through the
# switch point during the run. `kswitch*(Stot - tswitch)` starts at
# `100*(1 - 10)` = -900, so `exp(-kswitch*(Stot - tswitch))` is `inf` for the
# whole first half of the run — which is where all four corpus models were
# reported failing, at output index 1.
#
# The switch is over a species rather than over `time` on purpose: a `.net`
# rate law that mentions `time` is declined by the analytic sensitivity RHS
# outright ("references unrecognized symbol(s): time"), so it would fall back to
# CVODES' difference quotient and never evaluate the expression under test.
SIGMOID_DOSE = """\
begin parameters
    1 kswitch 100.0  # Constant
    2 tswitch 10.0   # Constant
    3 vmax    1.5    # Constant
    4 ks      1.0    # Constant
end parameters
begin functions
    1 dose()  vmax/(1 + exp(-kswitch*(Stot - tswitch)))
    2 grow()  ks
end functions
begin species
    1 A() 0.0
    2 S() 1.0
end species
begin reactions
    1 0 2 grow #_R1
    2 0 1 dose #_R2
end reactions
begin groups
    1 Atot 1
    2 Stot 2
end groups
"""


# The same dose, spelled the way the two corpus models of issue #549 spell theirs:
# a literal steepness and a constant shift inside the exponent. Both come out of
# sympy changed — `-100.0*(Stot - tswitch + 3)` multiplies out to an exponent with
# a `Float` coefficient AND a number in it, and the number folds into a factor:
#
#     exp(-100.0*Stot + 100.0*tswitch - 300.0) == 5.148e-131 * exp(-100.0*Stot + 100.0*tswitch)
#
# so the denominator summand is `5.148e-131*exp(u)`, a `Mul` with a `Float` in its
# exponent, and neither the summand nor the exponent ratio matched before. At the
# run's start `100*(20 - 1)` = 1900 is far past 709.8, so `exp` is `inf` in the
# numerator and in the denominator alike.
SHIFTED_SIGMOID_DOSE = """\
begin parameters
    1 tswitch 20.0   # Constant
    2 vmax    1.5    # Constant
    3 ks      1.0    # Constant
end parameters
begin functions
    1 dose()  vmax/(1 + exp(-100.0*(Stot - tswitch + 3)))
    2 grow()  ks
end functions
begin species
    1 A() 0.0
    2 S() 1.0
end species
begin reactions
    1 0 2 grow #_R1
    2 0 1 dose #_R2
end reactions
begin groups
    1 Atot 1
    2 Stot 2
end groups
"""


class TestTheShapeThroughASolve:
    @pytest.fixture
    def dose_net(self, tmp_path):
        net = tmp_path / "sigmoid_dose.net"
        net.write_text(SIGMOID_DOSE)
        return net

    def test_the_steepness_column_is_finite_from_the_first_output_point(self, dose_net):
        """The end-to-end case. The trajectory was always fine — ``1/(1 + inf)``
        is ``0`` — so only the sensitivity solve shows the defect, and it shows
        it before the switch, which is where the run starts.

        This exercises *both* emitters: the ∂f/∂p half is derived through SymPy
        and the J·yS half through the native saturable differentiator
        (``bngsim._saturable_jacobian``), which has to carry the same rewrite or
        the NaN comes back through the Jacobian."""
        import bngsim

        model = bngsim.Model.from_net(dose_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kswitch", "tswitch"])
        result = sim.run(t_span=(0, 20), n_points=21, rtol=1e-9, atol=1e-12)

        species = np.asarray(result.species)
        assert np.all(np.isfinite(species))

        sens = np.asarray(result.sensitivities)
        assert np.all(np.isfinite(sens))
        # ...and the columns are not trivially zero: the switch moves the dose.
        assert np.max(np.abs(sens)) > 1e-6

    @pytest.fixture
    def shifted_dose_net(self, tmp_path):
        net = tmp_path / "shifted_sigmoid_dose.net"
        net.write_text(SHIFTED_SIGMOID_DOSE)
        return net

    def test_a_shifted_dose_with_a_literal_steepness_is_finite_too(self, shifted_dose_net):
        """Issue #549's own spelling, end to end. This is the pair of misses the
        corpus models hit — a folded constant on the denominator summand and a
        ``Float`` in the exponent — and either one alone puts the NaN back."""
        import bngsim

        model = bngsim.Model.from_net(shifted_dose_net)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tswitch", "vmax"])
        result = sim.run(t_span=(0, 25), n_points=26, rtol=1e-9, atol=1e-12)

        assert np.all(np.isfinite(np.asarray(result.species)))
        sens = np.asarray(result.sensitivities)
        assert np.all(np.isfinite(sens))
        assert np.max(np.abs(sens[:, :, 0])) > 1e-6  # the onset column actually moves
