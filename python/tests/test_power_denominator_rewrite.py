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
  test would miss) and rejects exactly what it rejected;
* **the cost is bounded** — the pathological shape now finishes in well under a
  second, asserted as a hard wall-clock ceiling rather than a benchmark, so a
  reintroduced ``cancel`` fails the suite instead of quietly costing 15 minutes.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim._jacobian import (  # noqa: E402
    _linear_multiple_quotient,
    _remove_removable_power_denominators,
    sympy_to_c,
)

x, y, K, a, b = sp.symbols("x y K a b", positive=True)


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
        ],
    )
    def test_accepts_every_linear_multiple(self, base, expected):
        got = _linear_multiple_quotient(base, x, sp)
        assert got is not None
        assert sp.simplify(got - expected) == 0

    @pytest.mark.parametrize("base", [K + x, x**2, x * (a + x), sp.exp(x), y, sp.Integer(3)])
    def test_rejects_everything_else(self, base):
        """Rejection has to be exact, not merely cheap: accepting a base that is
        not a linear multiple would rewrite ``base^n/x`` into something that is
        not equal to it."""
        assert _linear_multiple_quotient(base, x, sp) is None

    @pytest.mark.parametrize(
        "base", [x, x / K, 2 * x, x * y, a * x + b * x, K + x, x**2, sp.exp(x)]
    )
    def test_it_agrees_with_cancel(self, base):
        """The property the replacement rests on, stated directly against the
        thing it replaced: same accept/reject verdict, same quotient."""
        cancelled = sp.cancel(base / x)
        cancel_says = None if cancelled.has(x) else cancelled
        structural = _linear_multiple_quotient(base, x, sp)
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
