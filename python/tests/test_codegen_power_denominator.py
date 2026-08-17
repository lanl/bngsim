"""GH #96 — the removable-power-denominator rewrite must not cost more than the
derivative it is tidying.

``_remove_removable_power_denominators`` rewrites ``base(x)^n / x`` to
``(base/x)·base^(n-1)`` when ``base`` is a linear multiple of ``x``, so the
emitted C does not evaluate ``0/0`` at a zero-valued initial condition. It asked
"is ``base/x`` free of ``x``?" with ``sp.cancel``, which answers by normalising
the whole rational expression first — unbounded work on a large ``Add``, and work
thrown away whenever the answer is "no", which is the common case (a base that
merely *mentions* ``x``, like ``Km + x``, is not a multiple of it).

That made a shared helper the slowest thing on three code paths, since
``sympy_to_c`` is what the analytical Jacobian (#76), the sensitivity RHS (#90)
and the expression output-sensitivity evaluator (#198) all print through. On
BioModels ``BIOMD0000000217`` a single ``cancel`` inside one derivative's rewrite
ran for minutes; the model's whole ``_analyze_output_sens`` took ~900 s, which is
a build that reads as a hang.

The tests below pin both halves of the fix:

* **the answer is unchanged** — the structural test accepts exactly the bases
  ``cancel`` accepted (including the ``a·x + b·x`` sum that a purely local ``Mul``
  test would miss) and rejects exactly what it rejected, with a quotient equal in
  *value*. Equal in value, not always identical in printed form: over a rational
  field ``cancel`` returns a ``Float`` ``1.0`` where the structural test returns
  ``Integer`` ``1``, which the C printer parenthesises differently
  (``1.0/(d)`` vs ``(1.0/(d))``). That is the entire emitted-source difference
  across the corpora — 0 of 585 ``.net`` models, 5 of 456 BioModels SBML models,
  and on those five all 138 differing ``(base, sym)`` pairs were checked to
  differ by form only;
* **the cost is bounded** — the pathological shape now finishes in well under a
  second, asserted as a hard wall-clock ceiling rather than a benchmark, so a
  reintroduced ``cancel`` fails the suite instead of quietly costing 15 minutes.

Issue #351 then generalised the same rewrite: the GH #96 question above needs the
denominator to be a bare ``Symbol``, which left the commonest shape of all —
``base`` divided by *itself* — untouched wherever the base was anything else.
Those tests are the third section below, and they carry the cost ceiling forward
onto the new branch, which by construction meets denominators GH #96 rejected
before they were ever looked at.

GH #388 relaxed the other half of the GH #96 bar. Asking ``cancel`` whether a
quotient exists means asking whether normalisation removes ``x`` from the
*answer*, and the structural extraction inherited that: it refused
``(x/(x + K))^n / x`` because the quotient ``1/(x + K)`` still mentions ``x``.
The extraction returns ``q`` with ``q·x == base`` by construction, so the rewrite
is an identity whatever ``q`` contains, and refusing this one left ``0/0`` = NaN
in the emitted ∂/∂IP3 of ``BIOMD0000000374`` / ``375`` at ``IP3 = 0`` where the
derivative is an ordinary ``0``.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim._jacobian import (  # noqa: E402
    _power_denominator_quotient,
    _remove_removable_power_denominators,
    _symbol_multiple_quotient,
    sympy_to_c,
)

x, y, K, a, b = sp.symbols("x y K a b", positive=True)


def _eval_c(src: str, **names: float) -> float:
    """Evaluate emitted C under **C** floating-point semantics, not Python's.

    The distinction is the whole point of the zero-base tests: C's ``pow(0.0, -0.5)``
    is ``+inf`` and ``pow(0.0, 0.0)`` is ``1.0``, while Python's ``0.0 ** -0.5``
    raises ``ZeroDivisionError``. ``np.float_power`` follows C, so the assertions
    below describe what the compiled artifact actually computes rather than what
    the same text would mean in the test's own language.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(
            eval(  # noqa: S307 - the input is this module's own emitted C
                src,
                {"pow": lambda b, e: np.float_power(np.float64(b), np.float64(e))},
                dict(names),
            )
        )


# ─── the structural quotient answers what cancel answered ──────────────────


class TestLinearMultipleQuotient:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            (x, sp.S.One),  # plain x
            (x / K, 1 / K),  # the documented rate-law base
            (2 * x, sp.Integer(2)),
            (x * y, y),
            (a * x + b * x, a + b),  # the Add case a local Mul test would miss
            (x * (a + b), a + b),
            # GH #388: the quotient may keep x. Both of these are exact — the
            # extraction returns q with q·x == base by construction — and both
            # were refused while the bar was "cancel eliminates x outright".
            (x / (x + K), 1 / (x + K)),  # the saturating base of BIOMD374/375
            (x * (a + x), a + x),
        ],
    )
    def test_accepts_every_linear_multiple(self, base, expected):
        got = _symbol_multiple_quotient(base, x, sp)
        assert got is not None
        assert sp.simplify(got - expected) == 0

    @pytest.mark.parametrize("base", [K + x, x**2, sp.exp(x), y, sp.Integer(3)])
    def test_rejects_everything_else(self, base):
        """Rejection has to be exact, not merely cheap: accepting a base that
        ``x`` does not divide would rewrite ``base^n/x`` into something that is
        not equal to it."""
        assert _symbol_multiple_quotient(base, x, sp) is None

    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            (x / (x + K), 1 / (x + K)),
            (x * (a + x), a + x),
            (a * x + b * x * y, a + b * y),
        ],
    )
    def test_the_quotient_still_multiplies_back_to_the_base(self, base, expected):
        """The property the rewrite rests on, checked directly rather than
        through ``cancel``: ``q·x == base``, which is what makes
        ``base^n/x == q·base^(n-1)`` an identity whatever ``q`` contains."""
        quotient = _symbol_multiple_quotient(base, x, sp)
        assert quotient is not None
        assert sp.simplify(quotient * x - base) == 0
        assert sp.simplify(quotient - expected) == 0

    def test_a_float_coefficient_agrees_in_value_not_in_form(self):
        """The one way the two forms differ, pinned so it stays a known property
        rather than a surprise in the next corpus diff: over a rational field
        ``cancel`` normalises the numerator to a ``Float``, so this base yields
        ``1.0/(...)`` there and ``1/(...)`` here. Equal in value; the C printer
        parenthesises them differently, which is the whole of why five BioModels
        models' emitted source moved without anything semantic moving with it."""
        base = x / (sp.Float("0.5") * y + sp.Float("0.25"))
        cancelled = sp.cancel(base / x)
        structural = _symbol_multiple_quotient(base, x, sp)
        assert not cancelled.has(x)
        assert structural is not None
        assert sp.simplify(structural - cancelled) == 0

    @pytest.mark.parametrize(
        "base", [x, x / K, 2 * x, x * y, a * x + b * x, K + x, x**2, sp.exp(x)]
    )
    def test_it_agrees_with_cancel_wherever_cancel_answers(self, base):
        """The property the GH #96 replacement rested on, stated directly against
        the thing it replaced: same accept/reject verdict, same quotient value.

        Held over every base ``cancel`` can decide. GH #388 then made the
        structural test the *stricter* description of the two — it accepts every
        base ``x`` divides, where ``cancel`` reports one only when normalisation
        also removes ``x`` from the answer — so the shapes in
        ``test_the_quotient_still_multiplies_back_to_the_base`` are deliberately
        not in this list."""
        cancelled = sp.cancel(base / x)
        cancel_says = None if cancelled.has(x) else cancelled
        structural = _symbol_multiple_quotient(base, x, sp)
        if cancel_says is None:
            assert structural is None
        else:
            assert structural is not None
            assert sp.simplify(structural - cancel_says) == 0


# ─── the rewrite still removes the denominator it exists for ───────────────


class TestRewriteStillWorks:
    def test_the_removable_denominator_is_removed(self):
        """The whole point of the pass: ``(x/K)^n/x`` must not survive as a form
        that evaluates 0/0 at x = 0."""
        expr = (x / K) ** 3 / x
        out = _remove_removable_power_denominators(expr)
        assert sp.simplify(out - expr) == 0
        assert not out.has(sp.Pow(x, -1))

    def test_a_genuine_denominator_is_left_alone(self):
        expr = (K + x) ** 3 / x
        out = _remove_removable_power_denominators(expr)
        assert sp.simplify(out - expr) == 0

    def test_emitted_c_is_finite_at_zero(self):
        """The behavioural reason the rewrite exists, checked through the emitter
        rather than on the sympy tree."""
        c = sympy_to_c((x / K) ** 2 / x, lambda n: {"x": "y[0]", "K": "p[0]"}.get(n))
        assert c is not None
        assert eval(c.replace("y[0]", "0.0").replace("p[0]", "2.0")) == 0.0


# ─── issue #351: the base divided by ITSELF ────────────────────────────────
#
# GH #96 above answers "is `base` a linear multiple of the symbol `sym` I am
# dividing by?", and to ask it at all the denominator had to be a bare Symbol.
# That left the commonest shape of the lot untouched: `base` divided by *itself*.
#
# It is the shape `sp.diff` produces for every power law — differentiating `u^n`
# gives `n·u^n·u'/u`, and sympy leaves the two Pows uncombined because with a
# symbolic exponent and a base of unknown sign `u^a·u^b = u^(a+b)` crosses a
# branch cut it will not assume. Where `u` is a bare symbol GH #96 caught it by
# accident (`_symbol_multiple_quotient(x, x)` is 1). Where `u` is anything else —
# `A4 - A4_star` in BIOMD0000000703 — nothing did, and the emitted `pow(u,n)/u`
# is 0/0 at `u == 0`, at a point where the law's own value is finite and the true
# derivative is an ordinary number.
#
# Only a SYMBOLIC exponent can reach this: `sp.diff(u**3, x)` is evaluated to
# `3*u**2*u'` by sympy itself. That is why the failure is rare and why it is real
# — BIOMD0000000703 writes its Hill coefficient as the parameter `nA4`.


class TestSameBaseCancels:
    @pytest.mark.parametrize(
        "base",
        [
            x,  # the bare symbol GH #96 already reached
            x - y,  # BIOMD0000000703's shape: a difference that can vanish
            x + K,
            x * y,
            (x - y) / K,
            sp.exp(x) - 1,
        ],
        ids=["symbol", "difference", "sum", "product", "scaled-difference", "transcendental"],
    )
    def test_the_quotient_is_one_for_any_identical_base(self, base):
        """``u^n/u == u^(n-1)`` is an identity wherever both sides are defined, so
        this branch needs no reasoning about ``u``'s form at all — which is exactly
        what the GH #96 branch beside it cannot say."""
        assert _power_denominator_quotient(base, base, sp) == sp.S.One

    @pytest.mark.parametrize(
        "base", [x - y, x + K, sp.exp(x) - 1], ids=["difference", "sum", "transcendental"]
    )
    def test_the_shape_sympy_actually_produces_is_cancelled(self, base):
        """Driven off ``sp.diff`` rather than a hand-written ``u**n/u``: the point
        is that the emitter meets this form in the wild, not that the rewrite can
        recognise a literal someone typed."""
        n = sp.Symbol("n_hill")
        deriv = sp.diff(base**n, y if base.has(y) else x)
        assert deriv.has(sp.Pow(base, -1)), "fixture no longer reproduces the shape"

        out = _remove_removable_power_denominators(deriv)

        assert not out.has(sp.Pow(base, -1))
        assert sp.simplify(out - deriv) == 0

    def test_a_genuine_denominator_survives_beside_a_cancelling_one(self):
        """The rewrite must be surgical. ``(K+x)^n/(x·(K+x))`` has one removable
        denominator and one real pole; removing the pole too would invent a finite
        value where the function genuinely diverges."""
        n = sp.Symbol("n_hill")
        expr = sp.diff((K + x) ** n, x) / x

        out = _remove_removable_power_denominators(expr)

        assert not out.has(sp.Pow(K + x, -1))
        assert out.has(sp.Pow(x, -1))
        assert sp.simplify(out - expr) == 0

    @pytest.mark.parametrize("n_val", [3.0, 2.0, 1.5])
    def test_the_rewrite_does_not_move_the_value_away_from_zero(self, n_val):
        """Away from the singular point the two forms are the same number, so every
        model that is correct today stays correct. (Not bit-identical: ``pow(u,n)/u``
        and ``pow(u,n-1)`` round differently, which is what the corpus diff is for.)"""
        n = sp.Symbol("n_hill")
        deriv = sp.diff((x - y) ** n, y)
        subs = {x: 3.0, y: 1.25, n: n_val}
        assert float(_remove_removable_power_denominators(deriv).subs(subs)) == pytest.approx(
            float(deriv.subs(subs)), rel=1e-12
        )

    @pytest.mark.parametrize(
        ("n_val", "expected"),
        [
            # d/dy (x-y)^n at x == y, i.e. -n·0^(n-1), in each exponent regime.
            (3.0, 0.0),  # n > 1: the derivative vanishes
            (2.0, 0.0),
            (1.0, -1.0),  # n = 1: pow(0,0) is 1 by C99, so this is -1, not 0
            (0.5, -float("inf")),  # n < 1: genuinely infinite, and honestly reported
        ],
        ids=["n=3", "n=2", "n=1", "n=0.5"],
    )
    def test_emitted_c_is_the_true_derivative_at_a_zero_base(self, n_val, expected):
        """The claim that makes this fix need no exponent case-split.

        ``pow(u, n-1)`` in IEEE arithmetic *is* the right answer in all three
        regimes, including the infinite one — so unlike the log / fractional-power
        family (GH #310/#317/#333/#336) there is nothing here to guard or refuse.
        A blanket zero would have been wrong for ``n = 1`` and a lie for ``n < 1``.
        """
        n = sp.Symbol("n_hill")
        deriv = sp.diff((x - y) ** n, y)  # -n·(x-y)^n/(x-y)
        c = sympy_to_c(deriv, lambda s: {"x": "u", "y": "v", "n_hill": "n"}.get(s))
        assert c is not None

        got = _eval_c(c, u=1.0, v=1.0, n=n_val)  # base x - y is exactly 0

        assert not np.isnan(got), "the NaN this fix exists to remove"
        assert got == expected if np.isinf(expected) else got == pytest.approx(expected)

    def test_the_pre_fix_form_is_the_nan_this_removes(self):
        """The counterfactual, stated as a test so the previous one cannot pass
        vacuously: the uncancelled form really is NaN at the same point."""
        assert np.isnan(_eval_c("pow(u - v, n) / (u - v)", u=1.0, v=1.0, n=2.0))


# ─── ...and the whole solve, against an exact oracle ───────────────────────
#
#   dS/dt = 1,          S(0) = 0   =>  S(t) = t
#   dP/dt = (S - c)^n,  P(0) = 0   =>  P(t) = ((t-c)^3 + c^3)/3   at n = 2
#   dP/dc = c^2 - (t-c)^2          =>  -t^2  at c = 0
#
# The base `S - c` is exactly zero at t = 0, which is BIOMD0000000703's shape in
# four reactions instead of ninety. `n` is a *parameter*, because a literal
# exponent cannot reach the bug at all: sympy evaluates `diff(u**3, x)` to
# `3*u**2*u'` itself and no division is ever emitted.
#
# Synthetic rather than the real model on purpose. BIOMD0000000703 lives in
# parity_checks/, which a wheel/subtree checkout does not carry, so a test built
# on it would skip exactly where regressions are least likely to be noticed — and
# it would have to check against finite differences, where this checks against
# closed form.

_ZERO_BASE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="zero_base_power">
    <listOfCompartments>
      <compartment id="c1" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c1" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="c1" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="cpar" value="0" constant="true"/>
      <parameter id="nn" value="2" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfProducts><speciesReference species="S" stoichiometry="1"/></listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
        </kineticLaw>
      </reaction>
      <reaction id="R2" reversible="false">
        <listOfProducts><speciesReference species="P" stoichiometry="1"/></listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><power/>
              <apply><minus/><ci>S</ci><ci>cpar</ci></apply>
              <ci>nn</ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


@needs_cc
def test_a_zero_base_power_solves_to_the_exact_analytic_sensitivity(tmp_path, monkeypatch):
    """End to end: pre-fix this is CV_CONV_FAILURE at t=0, post-fix it is -t^2.

    **The cold cache is load-bearing, not hygiene.** ``_CODEGEN_CACHE_KEY`` folds a
    digest of the emitter sources, so a machine holding an artifact built before
    this fix would keep loading it and the test would pass against code that never
    ran — which is issue #51's silent inertness, and is exactly what happened when
    this test was first written against the live cache.
    """
    import bngsim
    import bngsim._codegen as cg
    import bngsim._sbml_loader as sbml_loader

    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "codegen")

    model = sbml_loader.load_sbml_string(_ZERO_BASE_SBML)
    result = bngsim.Simulator(model, method="ode", sensitivity_params=["cpar"]).run(
        t_span=(0.0, 2.0), n_points=11, rtol=1e-10, atol=1e-12
    )

    t = result.time
    names = list(model.species_names)
    dP_dc = result.sensitivities[:, names.index("P"), 0]
    dS_dc = result.sensitivities[:, names.index("S"), 0]

    assert np.all(np.isfinite(result.sensitivities))
    np.testing.assert_allclose(dP_dc, -(t**2), rtol=1e-7, atol=1e-9)
    # S never sees `cpar`, so its column is exactly zero — a column that came back
    # merely *finite* would not distinguish a fix from a blanket zero.
    np.testing.assert_allclose(dS_dc, 0.0, atol=1e-12)


# ─── and it is no longer the slowest thing on the build ────────────────────


_KS = sp.symbols("K1:9", positive=True)


def _binding_polynomial(ks):
    """``1 + Σ x^i / ∏(k_1..k_i)`` — the receptor binding polynomial whose shape
    made ``cancel`` explode on BIOMD0000000217."""
    return 1 + sum(sp.prod(ks[:i]) ** -1 * x**i for i in range(1, len(ks) + 1))


def _pathological(depth=4):
    """A product carrying ``x``'s reciprocal beside a power whose base is a
    rational sum with *nested* binding polynomials in its denominators.

    This is the fixture the cost tests need to be worth anything, and it was
    arrived at by measurement, not by eye: an earlier hand-written "big rational"
    took 0.02 s with the bug in place, so the tests it guarded would have passed
    against the very code they exist to reject. Rejecting ``base = Add`` costs
    ``cancel`` a full multivariate normalisation, so cost climbs with the number
    of distinct constants — measured on the pre-fix code at 0.26 s / 2.0 s /
    12.0 s for depth 2 / 3 / 4, against ~0.2 ms after. depth=4 leaves the ceiling
    below a ~6x margin while keeping a passing run instant.
    """
    b1, b2 = _binding_polynomial(_KS[:depth]), _binding_polynomial(_KS[depth : 2 * depth])
    term = x ** (depth + 1) / (sp.prod(_KS) * b1 * b2)
    big = sum(term.subs(x, x + i) for i in range(3))
    return sp.Mul(big**2, sp.Pow(x, -1), evaluate=False)


class TestCostIsBounded:
    """Ceilings, not benchmarks: generous enough that ordinary machine noise
    cannot trip them, tight enough that restoring ``cancel`` does (12 s measured
    against a 2 s ceiling)."""

    def test_the_rewrite_finishes_fast(self):
        expr = _pathological()
        t0 = time.perf_counter()
        _remove_removable_power_denominators(expr)
        assert time.perf_counter() - t0 < 2.0

    def test_a_real_derivative_prints_fast(self):
        """End to end through ``sympy_to_c``, differentiating first — the call
        shape ``_analyze_output_sens`` and the Jacobian emitters actually make."""
        deriv = sp.diff(_pathological(), _KS[0])
        t0 = time.perf_counter()
        out = sympy_to_c(deriv, lambda _n: "z")
        elapsed = time.perf_counter() - t0
        assert out is not None
        assert elapsed < 2.0

    def test_a_large_non_symbol_denominator_is_rejected_fast(self):
        """The cost shape issue #351 opened, which the fixture above cannot reach.

        GH #96 fast-rejected any denominator that was not a bare ``Symbol``, so a
        big rational reciprocal never entered the inner loop at all. #351 has to let
        it in — that is the whole point — and the check it meets is a structural
        ``base == denom``. This pins that the comparison stays a comparison: a
        future ``sp.simplify(base - denom) == 0``, or anything else that normalises
        before answering, would be correct and would reintroduce exactly the cost
        GH #96 removed. Measured ~0.5 ms; the ceiling is four orders above it.
        """
        b1 = _binding_polynomial(_KS[:4])
        b2 = _binding_polynomial(_KS[4:8])
        big = sum((x**5 / (sp.prod(_KS) * b1 * b2)).subs(x, x + i) for i in range(3))
        # The reciprocal is a large Add that does NOT match the power's base, so
        # every candidate pair is rejected — the worst case, not the lucky one.
        expr = sp.Mul(big**2, sp.Pow(b1 + b2 + x, -1), evaluate=False)

        t0 = time.perf_counter()
        _remove_removable_power_denominators(sp.diff(expr, _KS[0]))
        assert time.perf_counter() - t0 < 2.0
