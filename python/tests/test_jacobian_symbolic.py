"""Unit tests for the symbolic rate-law differentiator (``bngsim._jacobian``),
the consumer-agnostic core of the GH #76 analytical Jacobian for Functional /
Michaelis–Menten rate laws.

The oracle is **finite differencing of the original rate law**: an emitted
analytic derivative is correct iff, evaluated at a point, it matches a central
finite difference of the original expression there. This catches both
differentiation bugs and ExprTk-emitter bugs in one shot (the emitted string is
re-parsed and evaluated). Fallback contracts assert ``None`` (→ FD Jacobian)
rather than a wrong answer for inputs the path cannot handle.
"""

from __future__ import annotations

import math
import time

import pytest

sympy = pytest.importorskip("sympy")
import sympy as sp  # noqa: E402
from bngsim import _jacobian as J  # noqa: E402

_TIME = J._TIME_SYM


def _lambdify(exprtk_str, varnames):
    """Parse an ExprTk string back to sympy and lambdify over ``varnames``."""
    e = J._exprtk_to_sympy(exprtk_str)
    assert e is not None, f"re-parse failed: {exprtk_str!r}"
    return sp.lambdify([sp.Symbol(v) for v in varnames], e, "numpy")


def _fd(f, base, var, h_rel=1e-6):
    h = h_rel * max(abs(base[var]), 1.0)
    bp = dict(base)
    bp[var] += h
    bm = dict(base)
    bm[var] -= h
    return (float(f(**bp)) - float(f(**bm))) / (2.0 * h)


# Rate laws spanning the constructs SBML/.net Functional laws use. Each: name,
# expression, state observables, constant params.
SMOOTH_CASES = [
    ("michaelis_menten", "(Vmax*A)/(Km+A)", {"A"}, {"Vmax", "Km"}),
    ("bisubstrate", "(k*A*B)/((Km+A)*(Km+B))", {"A", "B"}, {"k", "Km"}),
    ("hill", "(Vmax*(A^n))/((Km^n)+(A^n))", {"A"}, {"Vmax", "Km", "n"}),
    ("ratio_exp", "k*A*exp(-B/Km)", {"A", "B"}, {"k", "Km"}),
    ("linear_combo", "k*(A+2*B)/(C+1)", {"A", "B", "C"}, {"k"}),
    ("sqrt", "k*sqrt(A*B)", {"A", "B"}, {"k"}),
    ("log", "k*log(A/Km)", {"A"}, {"k", "Km"}),
    ("with_time", "k*A*time()", {"A"}, {"k"}),
]

_VALS = {"A": 1.7, "B": 0.9, "C": 2.3, "k": 0.5, "Km": 0.8, "Vmax": 3.0, "n": 2.0, _TIME: 4.2}


@pytest.mark.parametrize("name,expr,obs,const", SMOOTH_CASES, ids=[c[0] for c in SMOOTH_CASES])
def test_per_observable_matches_finite_difference(name, expr, obs, const):
    """Every emitted ∂rate/∂obs matches a central FD of the original."""
    terms = J.build_per_observable_terms(expr, {}, obs, const)
    assert terms is not None, f"{name}: differentiator unexpectedly fell back"
    varnames = sorted(obs | const) + [_TIME]
    f = _lambdify(expr, varnames)
    base = {v: _VALS[v] for v in varnames}
    seen = set()
    for obs_name, deriv_str in terms:
        seen.add(obs_name)
        df = _lambdify(deriv_str, varnames)
        analytic = float(df(**base))
        fd = _fd(f, base, obs_name)
        assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-9), (
            f"{name} d/d{obs_name}: analytic={analytic} fd={fd} expr={deriv_str}"
        )
    # An observable absent from the terms must have a genuinely zero derivative.
    for obs_name in obs - seen:
        assert _fd(f, base, obs_name) == pytest.approx(0.0, abs=1e-7)


def test_piecewise_derivative_tracks_active_branch():
    """if(A>Km, k*A, k*Km) differentiates per branch: k above the threshold,
    0 below — the FD oracle must agree on both sides."""
    expr = "if(A>Km,k*A,k*Km)"
    obs, const = {"A"}, {"k", "Km"}
    terms = J.build_per_observable_terms(expr, {}, obs, const)
    assert terms is not None
    varnames = sorted(obs | const) + [_TIME]
    f = _lambdify(expr, varnames)
    for region in (
        {"A": 1.7, "k": 0.5, "Km": 0.8, _TIME: 0.0},
        {"A": 0.3, "k": 0.5, "Km": 0.8, _TIME: 0.0},
    ):
        for obs_name, deriv_str in terms:
            df = _lambdify(deriv_str, varnames)
            assert float(df(**region)) == pytest.approx(_fd(f, region, obs_name), abs=1e-4)


def test_nested_function_is_inlined_before_diff():
    """A rate law referencing a derived quantity (assignment rule / nested
    function) is flattened, so d(k*helper)/dA with helper=A*A yields 2*k*A."""
    terms = J.build_per_observable_terms("k*helper", {"helper": "A*A"}, {"A"}, {"k"})
    assert terms is not None
    dmap = dict(terms)
    f = _lambdify(dmap["A"], ["A", "k"])
    assert float(f(A=3.0, k=0.5)) == pytest.approx(2 * 0.5 * 3.0)


class TestPerSpeciesChainRule:
    """SBML per-species packaging chain-rules ∂rate/∂obs through each
    observable's species group, keeping observable symbols (live-evaluable)."""

    def test_one_to_one_observable_is_plain_partial(self):
        terms = J.build_per_species_terms(
            "(Vmax*A)/(Km+A)", {}, {"A": [(0, 1.0)]}, {0: (False, 1.0)}, {"Vmax", "Km"}
        )
        assert terms is not None
        sp_idx, deriv = terms[0]
        assert sp_idx == 0
        f = _lambdify(deriv, ["A", "Vmax", "Km"])
        g = _lambdify("(Vmax*A)/(Km+A)", ["A", "Vmax", "Km"])
        base = {"A": 1.7, "Vmax": 3.0, "Km": 0.8}
        assert float(f(**base)) == pytest.approx(_fd(g, base, "A"))

    def test_aggregating_observable_distributes_group_factors(self):
        """obs = 1*sp0 + 2*sp1 ⇒ ∂rate/∂sp1 = 2·∂rate/∂sp0, and both stay in
        the observable symbol (not expanded to species)."""
        groups = {"obstot": [(0, 1.0), (1, 2.0)]}
        terms = J.build_per_species_terms(
            "k*obstot/(Km+obstot)",
            {},
            groups,
            {0: (False, 1.0), 1: (False, 1.0)},
            {"k", "Km"},
        )
        assert terms is not None
        d = dict(terms)
        assert "obstot" in d[0] and "obstot" in d[1]  # observable symbol retained
        f0 = _lambdify(d[0], ["obstot", "k", "Km"])
        f1 = _lambdify(d[1], ["obstot", "k", "Km"])
        base = {"obstot": 1.3, "k": 0.5, "Km": 0.8}
        assert float(f1(**base)) == pytest.approx(2.0 * float(f0(**base)))

    def test_amount_valued_species_scales_by_volume(self):
        """An amount-valued species reads as x·V in observables, so its column
        picks up the volume factor V."""
        groups = {"A": [(0, 1.0)]}
        plain = J.build_per_species_terms(
            "(Vmax*A)/(Km+A)", {}, groups, {0: (False, 1.0)}, {"Vmax", "Km"}
        )
        scaled = J.build_per_species_terms(
            "(Vmax*A)/(Km+A)", {}, groups, {0: (True, 4.0)}, {"Vmax", "Km"}
        )
        assert plain is not None and scaled is not None
        base = {"A": 1.7, "Vmax": 3.0, "Km": 0.8}
        fp = _lambdify(dict(plain)[0], ["A", "Vmax", "Km"])
        fs = _lambdify(dict(scaled)[0], ["A", "Vmax", "Km"])
        assert float(fs(**base)) == pytest.approx(4.0 * float(fp(**base)))


class TestLogicalOperators:
    """ExprTk spells logical AND/OR three ways each (``and``/``&&``/``&``,
    ``or``/``||``/``|``); hand-written BNGL ``.net`` conditions overwhelmingly use
    the symbolic form. All spellings must parse, and — critically — must keep
    ExprTk's precedence, where a comparison binds tighter than a logical. A naive
    ``&&`` → ``&`` substitution does not: Python binds ``&`` tighter, so
    ``a > b & c < d`` reassociates to ``a > (b & c) < d`` and raises.
    """

    # Four spellings of the same condition, all of which must parse identically.
    EQUIVALENT = [
        "if((A>Km)&&(A<2*Km),k*A,0)",
        "if(A>Km && A<2*Km,k*A,0)",  # unparenthesized operands
        "if((A>Km) and (A<2*Km),k*A,0)",
        "if(A>Km & A<2*Km,k*A,0)",
    ]

    @pytest.mark.parametrize("expr", EQUIVALENT)
    def test_and_spellings_agree(self, expr):
        e = J._exprtk_to_sympy(expr)
        assert e is not None, f"failed to parse {expr!r}"
        assert e == J._exprtk_to_sympy("if((A>Km) and (A<2*Km),k*A,0)")

    @pytest.mark.parametrize("expr", ["if((A>Km)||(B>Km),k,0)", "if(A>Km or B>Km,k,0)"])
    def test_or_spellings_agree(self, expr):
        e = J._exprtk_to_sympy(expr)
        assert e is not None, f"failed to parse {expr!r}"
        assert e == J._exprtk_to_sympy("if((A>Km)|(B>Km),k,0)")

    def test_and_binds_tighter_than_or(self):
        """``a && b || c`` is ``(a && b) || c``, matching ExprTk and BNGL."""
        got = J._exprtk_to_sympy("if(A>1 && B>2 || C>3,k,0)")
        want = J._exprtk_to_sympy("if((A>1 && B>2) || C>3,k,0)")
        assert got is not None and got == want
        assert got != J._exprtk_to_sympy("if(A>1 && (B>2 || C>3),k,0)")

    def test_comma_is_not_a_logical_boundary(self):
        """A depth-0 comma separates arguments; a logical inside one argument must
        not swallow it (``Piecewise((v, cond), …)``, ``max(a, b)``)."""
        A, B, k, Km = sp.symbols("A B k Km")
        e = J._exprtk_to_sympy("if(A>1 && B>2,k*A,k*Km)")
        assert e is not None
        assert e == sp.Piecewise((k * A, sp.And(A > 1, B > 2)), (k * Km, True))
        # a non-logical multi-arg call keeps its arguments too
        assert J._exprtk_to_sympy("max(A,B)+min(A,2)") == sp.Max(A, B) + sp.Min(A, 2)

    def test_word_operators_are_not_matched_inside_identifiers(self):
        """``land`` / ``orbit`` are identifiers, not operator occurrences."""
        e = J._exprtk_to_sympy("k*land + orbit")
        assert e is not None
        assert {str(s) for s in e.free_symbols} == {"k", "land", "orbit"}

    def test_derivative_of_compound_condition_matches_fd_in_each_branch(self):
        """The whole point: a compound-condition rate law differentiates per
        branch instead of falling back, and matches an FD of the original."""
        expr = "if((A>Km)&&(A<3*Km),k*A,0)"
        obs, const = {"A"}, {"k", "Km"}
        terms = J.build_per_observable_terms(expr, {}, obs, const)
        assert terms is not None, "compound condition unexpectedly fell back"
        varnames = sorted(obs | const) + [_TIME]
        f = _lambdify(expr, varnames)
        for region in (  # inside the window, below it, above it
            {"A": 1.7, "k": 0.5, "Km": 0.8, _TIME: 0.0},
            {"A": 0.3, "k": 0.5, "Km": 0.8, _TIME: 0.0},
            {"A": 9.9, "k": 0.5, "Km": 0.8, _TIME: 0.0},
        ):
            for obs_name, deriv_str in terms:
                df = _lambdify(deriv_str, varnames)
                assert float(df(**region)) == pytest.approx(_fd(f, region, obs_name), abs=1e-4)

    def test_emitted_derivative_round_trips(self):
        """The emitted ExprTk derivative must re-parse — it did not before, since
        the emitter writes word-form logicals with unparenthesized operands."""
        terms = J.build_per_observable_terms("if((A>Km)&&(A<3*Km),k*A,0)", {}, {"A"}, {"k", "Km"})
        assert terms
        for _obs, s in terms:
            assert J._exprtk_to_sympy(s) is not None, f"emitted {s!r} does not re-parse"

    def test_unbalanced_parens_still_fall_back(self):
        assert (
            J.build_per_observable_terms("if((A>Km)&&(A<2*Km,k*A,0)", {}, {"A"}, {"k", "Km"})
            is None
        )

    def test_deeply_nested_logical_free_expression_is_untouched(self):
        """A logical-free expression must not be walked at all. Real models nest
        `if()` hundreds deep — the mallela2024 COVID model has a 354-level daily
        lookup table — and a rewrite that descended per paren level blew Python's
        recursion limit on it."""
        expr = "0"
        for i in range(400):
            expr = f"if((t<={i}),{i}.0,{expr})"
        assert J._rewrite_logicals(expr) == expr  # no descent, no RecursionError

    def test_pathologically_nested_logical_falls_back_not_crashes(self):
        """Past the rewrite budget the string is returned untouched, so the caller
        gets the FD fallback rather than a RecursionError."""
        expr = "A>0"
        for _ in range(400):
            expr = f"({expr}) && (A>0)"
        J._rewrite_logicals(expr)  # must not raise
        assert J.build_per_observable_terms(f"if({expr},k*A,0)", {}, {"A"}, {"k"}) is None


class TestEqualityOperators:
    """GH #335. ``==`` / ``!=`` are the two relationals ``parse_expr`` evaluates
    with *Python* semantics — ``Symbol('x') == 0`` is the bool ``False`` — so
    before the fix they were silently discarded: ``if(x==0,1,2)`` parsed to a
    bare ``2`` (the condition gone, the else-branch left) and ``if(x!=0,1,2)`` to
    ``1``. They are rewritten to the ``Eq`` / ``Ne`` call form, exactly as the
    logical connectives are and at the same tighter-than-logical precedence, so
    the comparison reaches sympy as a real relational. The four ordering
    relationals already build sympy relationals and must stay untouched.
    """

    def test_equality_condition_is_preserved(self):
        x = sp.Symbol("x")
        assert J._exprtk_to_sympy("if(x==0,1,2)") == sp.Piecewise((1, sp.Eq(x, 0)), (2, True))

    def test_inequality_condition_is_preserved(self):
        x = sp.Symbol("x")
        assert J._exprtk_to_sympy("if(x!=0,1,2)") == sp.Piecewise((1, sp.Ne(x, 0)), (2, True))

    def test_condition_no_longer_collapses_to_a_bare_branch(self):
        """The exact defect in the report: the comparison must not vanish, leaving
        the else-branch (``2``) or the matched branch (``1``) as a constant."""
        assert J._exprtk_to_sympy("if(x==0,1,2)") != sp.Integer(2)
        assert J._exprtk_to_sympy("if(x!=0,1,2)") != sp.Integer(1)
        # and the ``k/x`` guard the report highlights is now a real guard
        x, k = sp.symbols("x k")
        assert J._exprtk_to_sympy("if(x==0,0,k/x)") == sp.Piecewise(
            (0, sp.Eq(x, 0)), (k / x, True)
        )

    def test_nested_equality_keeps_every_branch(self):
        """BIOMD0000000248's shape — a nested ``==`` selector collapsed to the
        innermost else-branch (``VmaxVH``), dropping the other two. Every branch
        must now be reachable."""
        sel, a, b, c = sp.symbols("sel a b c")
        g = J._exprtk_to_sympy("if((sel==1),a,if((sel==2),b,c))")
        assert g is not None
        assert g.subs(sel, 1) == a and g.subs(sel, 2) == b and g.subs(sel, 3) == c

    def test_equality_binds_tighter_than_logicals(self):
        """``flag==1 && x>0`` is ``(flag==1) && (x>0)``: the equality is an operand
        of the And, matching ExprTk/BNGL precedence."""
        flag, x, k = sp.symbols("flag x k")
        assert J._exprtk_to_sympy("if(flag==1 && x>0, k, 0)") == sp.Piecewise(
            (k, sp.And(sp.Eq(flag, 1), x > 0)), (0, True)
        )
        assert J._exprtk_to_sympy("if(flag!=1 || x>0, k, 0)") == sp.Piecewise(
            (k, sp.Or(sp.Ne(flag, 1), x > 0)), (0, True)
        )

    @pytest.mark.parametrize(
        "op,rel",
        [
            ("<", sp.StrictLessThan),
            ("<=", sp.LessThan),
            (">", sp.StrictGreaterThan),
            (">=", sp.GreaterThan),
        ],
    )
    def test_ordering_relationals_are_untouched(self, op, rel):
        """<, <=, >, >= already parse to sympy relationals; the equality rewrite
        must not disturb them (they carry no ``==`` / ``!=`` to match)."""
        x = sp.Symbol("x")
        assert J._exprtk_to_sympy(f"if(x{op}1,1,2)") == sp.Piecewise((1, rel(x, 1)), (2, True))

    def test_equality_present_but_logical_free_still_descends(self):
        """The cheap pre-filter must fire on ``==`` with no logical operator
        present — that case short-circuited before the fix and left it raw."""
        pre = J._rewrite_logicals("Piecewise((1, x==0), (2, True))")
        assert "Eq(x, 0)" in pre and "x==0" not in pre

    def test_derivative_tracks_the_active_branch(self):
        """The payoff: ``d/dA if(sel==1, k*A, k*A*A)`` selects ``k`` in the matched
        branch and ``2*k*A`` in the else, each matching an FD of the original.
        Before the fix the condition was gone and one branch answered for both."""
        expr = "if(sel==1, k*A, k*A*A)"
        obs, const = {"A"}, {"k", "sel"}
        terms = J.build_per_observable_terms(expr, {}, obs, const)
        assert terms is not None, "equality-guarded law unexpectedly fell back"
        varnames = sorted(obs | const) + [_TIME]
        f = _lambdify(expr, varnames)
        for sel in (1.0, 2.0):  # matched branch, then else
            region = {"A": 1.7, "k": 0.5, "sel": sel, _TIME: 0.0}
            for obs_name, deriv_str in terms:
                df = _lambdify(deriv_str, varnames)
                assert float(df(**region)) == pytest.approx(_fd(f, region, obs_name), abs=1e-4)

    def test_emitted_equality_derivative_round_trips(self):
        """The emitted derivative carries the ``Eq`` condition through; it must
        re-parse (the ExprTk printer spells it infix, GH #310)."""
        terms = J.build_per_observable_terms("if(sel==1, k*A, k*A*A)", {}, {"A"}, {"k", "sel"})
        assert terms
        for _obs, s in terms:
            assert J._exprtk_to_sympy(s) is not None, f"emitted {s!r} does not re-parse"


class TestFallbackContract:
    """Inputs the path cannot guarantee must return None (→ FD Jacobian), never
    a silently-wrong derivative."""

    def test_unclassified_symbol_falls_back(self):
        # Z is neither an observable nor a declared constant — could be a hidden
        # state (e.g. a rate-rule-target parameter). Must not be diffed as const.
        assert J.build_per_observable_terms("k*A*Z", {}, {"A"}, {"k"}) is None

    def test_function_cycle_falls_back(self):
        assert J.build_per_observable_terms("g", {"g": "g+1"}, {"A"}, {"k"}) is None

    def test_unparseable_falls_back(self):
        assert J.build_per_observable_terms("k*A*", {}, {"A"}, {"k"}) is None

    def test_constant_only_rate_is_covered_with_empty_column(self):
        # No observable dependence ⇒ a zero Jacobian column. This is *covered*
        # (empty list), NOT a fallback (None) — a constant-rate functional
        # reaction must not knock the whole model onto the FD path.
        assert J.build_per_observable_terms("k*Km", {}, {"A"}, {"k", "Km"}) == []


class TestNonDifferentiableConstructs:
    """GH #250 — rate laws the emitter accepts but cannot differentiate.

    ``abs``, ``min``, ``max``, ``floor``, ``ceil`` and ``sign`` are all ExprTk
    builtins, so they reach the symbolic core happily as *inputs*. None of them has
    a derivative the emitter can print: the first three produce ``re``/``im``/
    ``Heaviside`` and the last three an unevaluated ``Derivative``. The contract is
    that such a rate law falls back (``None``), and that it does so **before**
    paying the derivation — on ``BIOMD0000000385`` that is 138 s against a 20 s
    budget, unbounded by it because one ``Abs`` derivative is a single ``sp.diff``.
    """

    @pytest.mark.parametrize("fn", ["abs", "floor", "ceil", "sign"])
    def test_unary_nondifferentiable_over_observable_falls_back(self, fn):
        assert J.build_per_observable_terms(f"k*{fn}(A)", {}, {"A"}, {"k"}) is None

    @pytest.mark.parametrize("fn", ["min", "max"])
    def test_binary_nondifferentiable_over_observable_falls_back(self, fn):
        assert J.build_per_observable_terms(f"k*{fn}(A,B)", {}, {"A", "B"}, {"k"}) is None

    @pytest.mark.parametrize("fn", ["abs", "floor", "ceil", "sign"])
    def test_over_a_parameter_only_still_differentiates(self, fn):
        # d/dA of abs(k)*A is abs(k) — the construct is only a problem when a
        # *differentiation variable* sits under it, so this must NOT fall back.
        terms = J.build_per_observable_terms(f"{fn}(k)*A", {}, {"A"}, {"k"})
        assert terms is not None and len(terms) == 1

    def test_inside_a_piecewise_condition_still_differentiates(self):
        # d/dx Piecewise((f, c), ...) = Piecewise((df/dx, c), ...): conditions are
        # copied through, never differentiated, so a blocked function in one is
        # emitted verbatim and stays legal. MODEL1006230034 is this shape and keeps
        # a complete analytical Jacobian; a position-blind check would take it away.
        terms = J.build_per_observable_terms("if(abs(A)>Km, k*A, 0)", {}, {"A"}, {"k", "Km"})
        assert terms is not None and len(terms) == 1
        assert "abs(A)" in terms[0][1], f"condition lost its abs(): {terms[0][1]}"

    def test_reached_through_an_inlined_function(self):
        # The check runs on the *inlined* expression, so a construct hidden behind
        # a user function is caught the same way.
        assert J.build_per_observable_terms("g", {"g": "k*abs(A)"}, {"A"}, {"k"}) is None

    def test_declines_without_paying_the_derivation(self):
        # The point of the pre-check: the expensive case is expensive *because*
        # sympy expands Abs into re/im, so a rate law with several of them over
        # several observables must return in well under the time one such
        # derivative takes (measured: 62 s for a single dAbs on BIOMD0000000385).
        expr = "k*abs(A*B + A/B) + abs(B*B - A) + abs(A + B)"
        t0 = time.perf_counter()
        assert J.build_per_observable_terms(expr, {}, {"A", "B"}, {"k"}) is None
        assert time.perf_counter() - t0 < 1.0, "fell back only after differentiating"


class TestUnevaluatedDerivativeIsNotEmittable:
    """GH #250 — ``_is_emittable`` must reject an unevaluated ``Derivative``.

    It scanned ``atoms(sp.Function)`` only, and ``Derivative`` is not a ``Function``,
    so ``d/dx sign(x)`` — which sympy returns as ``Derivative(sign(x), x)`` — passed
    the gate and both printers then printed it verbatim. ``Derivative(sign(x), x)``
    is not an ExprTk builtin and ``Derivative(...)`` is not a declared C function,
    so the emitters produced a broken derivative instead of declining to one.
    """

    @pytest.mark.parametrize("fn", [sp.sign, sp.floor, sp.ceiling])
    def test_unevaluated_derivative_rejected(self, fn):
        x = sp.Symbol("A")
        d = sp.diff(fn(x), x)
        assert isinstance(d, sp.Derivative), "fixture no longer produces a Derivative"
        assert J._is_emittable(d) is False

    @pytest.mark.parametrize("fn", [sp.sign, sp.floor, sp.ceiling])
    def test_printers_decline_rather_than_emit_it(self, fn):
        x = sp.Symbol("A")
        d = sp.diff(fn(x), x)
        assert J.sympy_to_exprtk(d) is None
        assert J.sympy_to_c(d, lambda n: n) is None

    def test_ordinary_derivatives_still_emittable(self):
        x = sp.Symbol("A")
        for e in (x / (1 + x**2), sp.exp(x) * x, sp.sqrt(x), sp.sin(x) * sp.log(x)):
            d = sp.diff(e, x)
            assert J._is_emittable(d) is True, f"{e} regressed"
            assert J.sympy_to_exprtk(d) is not None


def test_nondifferentiable_set_matches_a_live_rederivation():
    """``_NONDIFFERENTIABLE_EMITTER_FUNCS`` is a cache; this runs the derivation.

    The constant was produced by differentiating every name in
    ``_SYMPY_FUNC_TO_EXPRTK`` and keeping the ones whose result the emitter refuses
    to print. That makes it a *fact about the emitter map*, not a judgement — so
    adding a function to the map without re-deriving would silently reintroduce
    GH #250 for that function, and dropping one would leave a stale name declining
    rate laws for no reason. Neither is caught by the cases above, which only test
    the six names already in the constant.

    Re-derived here through ``sympy_to_exprtk`` rather than ``_is_emittable``,
    because the printer is the gate that actually matters and it is what the #250
    fix had to change: ``_is_emittable`` passed ``Derivative(sign(x), x)`` while the
    printer happily emitted it.
    """
    a, b = sp.Symbol("A"), sp.Symbol("B")
    derived = set()
    for name in J._SYMPY_FUNC_TO_EXPRTK:
        fn = getattr(sp, name, None)
        assert fn is not None, f"{name} maps to no sympy object — emitter map is stale"
        # Both arities, unioned, rather than the first that builds: `Min(A)`
        # degenerates to `A` (a clean derivative) and only `Min(A, B)` shows the
        # Heaviside, while `sqrt(A, B)` does not exist at all. A function belongs
        # in the set if *any* arity it accepts has an underivable form.
        built = False
        for args in ((a,), (a, b)):
            try:
                deriv = sp.diff(fn(*args), a)
            except (TypeError, ValueError):
                continue
            built = True
            if J.sympy_to_exprtk(deriv) is None:
                derived.add(name)
        assert built, f"could not build {name} at arity 1 or 2"

    assert derived == set(J._NONDIFFERENTIABLE_EMITTER_FUNCS), (
        "the non-differentiable set no longer matches the emitter map: "
        f"re-derived {sorted(derived)}, constant holds "
        f"{sorted(J._NONDIFFERENTIABLE_EMITTER_FUNCS)}. If a function was added to "
        "_SYMPY_FUNC_TO_EXPRTK, re-derive the constant rather than editing this "
        "test — a name missing from it means rate laws using that function pay a "
        "derivation the emitter will refuse (GH #250)."
    )


class TestEmitterRoundTrip:
    """The ExprTk emitter must produce strings the engine grammar accepts and
    that re-parse to the same value."""

    @pytest.mark.parametrize(
        "src", ["A^2", "sqrt(A)", "1.0/A", "exp(A)*A", "A^(-2)", "k*A/(Km+A)"]
    )
    def test_roundtrip_value(self, src):
        e = J._exprtk_to_sympy(src)
        s = J.sympy_to_exprtk(e)
        assert s is not None
        back = J._exprtk_to_sympy(s)
        diff = sp.simplify(e - back)
        assert diff == 0, f"{src} -> {s} did not round-trip ({diff})"

    def test_power_uses_caret_not_double_star(self):
        s = J.sympy_to_exprtk(J._exprtk_to_sympy("A^3"))
        assert "**" not in s and "^" in s

    def test_time_symbol_emits_as_time_call(self):
        s = J.sympy_to_exprtk(J._exprtk_to_sympy("k*time()"))
        assert "time()" in s


# Saturating (Michaelis–Menten-like) degradation: a single Functional reaction
# rate = k*S/(Km+S). ∂rate/∂S = k*Km/(Km+S)^2 is non-trivial and smooth — the
# canonical end-to-end case for the C++ FD self-validation gate.
_SBML_SAT_DEG = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sat_deg">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="2" constant="true"/>
      <parameter id="Km" value="5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/>
            <apply><times/><ci>k</ci><ci>S</ci></apply>
            <apply><plus/><ci>Km</ci><ci>S</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# Hill-rate derivative at S=0 used to emit removable ``S^n/S`` terms. The
# C++ self-check evaluates the analytical Jacobian at the initial state, so that
# raw form produced NaN even though the mathematical derivative is finite.
_SBML_ZERO_AXIS_HILL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="zero_axis_hill">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="v" value="2" constant="true"/>
      <parameter id="K" value="5" constant="true"/>
      <parameter id="n" value="4" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="activation" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
            <ci>v</ci>
            <apply><divide/>
              <apply><power/>
                <apply><divide/><ci>S</ci><ci>K</ci></apply>
                <ci>n</ci>
              </apply>
              <apply><plus/>
                <cn>1</cn>
                <apply><power/>
                  <apply><divide/><ci>S</ci><ci>K</ci></apply>
                  <ci>n</ci>
                </apply>
              </apply>
            </apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


class TestSelfCheckEndToEnd:
    """End-to-end exercise of the C++ FD self-validation gate
    (``NetworkModel::set_functional_jacobian``): a correct analytical Jacobian is
    accepted; a deliberately wrong one is rejected (model stays on FD). This is
    the gate that the reliability-aware (two-step Richardson) rewrite hardened so
    it no longer false-fails correct stiff Jacobians (GH #76)."""

    @staticmethod
    def _ensure_no_bypass():
        import os

        # The diagnostic bypass must not be active — these tests assert the gate.
        os.environ.pop("BNGSIM_JAC_NO_SELFCHECK", None)

    @staticmethod
    def _build_terms(core):
        ctx = core.functional_jacobian_context()
        func_map = dict(ctx["function_map"])
        obs_groups = {n: [(int(s), float(f)) for s, f in g] for n, g in ctx["observables"]}
        smeta = {i: (bool(a), float(v)) for i, (a, v) in enumerate(ctx["species_meta"])}
        consts = set(ctx["constant_names"])
        terms = []
        for rxn in ctx["functional_reactions"]:
            t = J.build_per_species_terms(rxn["rate_expr"], func_map, obs_groups, smeta, consts)
            assert t is not None
            terms.append((rxn["rxn_idx"], False, [(int(j), e) for j, e in t]))
        return terms

    def test_correct_functional_jacobian_passes(self):
        self._ensure_no_bypass()
        import bngsim

        core = bngsim.Model.from_sbml_string(_SBML_SAT_DEG)._core
        terms = self._build_terms(core)
        assert terms and terms[0][2], "expected a non-empty Functional Jacobian term"
        assert core.set_functional_jacobian(terms) is True
        assert core.analytical_jacobian_complete is True

    def test_corrupted_term_is_rejected(self):
        self._ensure_no_bypass()
        import bngsim

        core = bngsim.Model.from_sbml_string(_SBML_SAT_DEG)._core
        terms = self._build_terms(core)
        ri, po, dl = terms[0]
        # Double the (correct) ∂rate/∂S — a genuine ~100% divergence the gate must
        # catch even though the FD reliability gating tolerates real FD noise.
        corrupted = [(ri, po, [(dl[0][0], f"2.0*({dl[0][1]})"), *dl[1:]])]
        assert core.set_functional_jacobian(corrupted) is False
        assert core.analytical_jacobian_complete is False

    def test_zero_axis_hill_derivative_passes_self_check(self):
        self._ensure_no_bypass()
        import bngsim

        model = bngsim.Model.from_sbml_string(_SBML_ZERO_AXIS_HILL)
        assert model.prepare_analytical_jacobian() is True
        flat = model._core._dense_analytical_jacobian(0.0, [0.0, 0.0])
        assert all(math.isfinite(float(v)) for v in flat)


class TestPerObservableEndToEnd:
    """`.net` Functional reactions carry a mass-action species factor
    (apply_species_factor=true): rate = func(observables)·∏reactants. The
    per-observable C++ path (GH #76 task 2) chain-rules ∂func/∂obs_k through each
    observable group and applies the product rule for the species factor. These
    tests drive the full ``attach_functional_jacobian`` round-trip end to end."""

    def _model(self, data_dir):
        import bngsim

        return bngsim.Model.from_net(str(data_dir / "per_observable_jac.net"))

    def test_attach_succeeds_and_is_complete(self, data_dir):
        import os

        os.environ.pop("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", None)
        core = self._model(data_dir)._core
        # The fixture's functional reaction has a non-empty reactant multiset, so
        # it routes to the per-observable path the C++ side previously rejected.
        ctx = core.functional_jacobian_context()
        rxn = ctx["functional_reactions"][0]
        assert rxn["apply_species_factor"] and rxn["reactant_idx0"], (
            "expected a .net species factor"
        )
        assert J.attach_functional_jacobian(core) is True
        assert core.analytical_jacobian_complete is True

    def test_analytical_matches_fd_trajectory(self, data_dir):
        import os

        import bngsim
        import numpy as np

        def run(force_fd: bool):
            if force_fd:
                os.environ["BNGSIM_ANALYTICAL_FUNCTIONAL_JAC"] = "0"
            else:
                os.environ.pop("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", None)
            sim = bngsim.Simulator(self._model(data_dir), method="ode", jacobian="auto")
            return np.asarray(sim.run((0.0, 200.0), 101).species)

        analytical, fd = run(False), run(True)
        os.environ.pop("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", None)
        peak = np.maximum(np.abs(fd).max(axis=0), 1.0)
        assert float((np.abs(analytical - fd) / peak).max()) < 1e-8

    def test_explicit_analytical_request_is_accepted(self, data_dir):
        # Before task 2 a .net Functional reaction with a species factor forced FD,
        # so jacobian="analytical" raised. It must now run.
        import os

        import bngsim

        os.environ.pop("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", None)
        sim = bngsim.Simulator(self._model(data_dir), method="ode", jacobian="analytical")
        result = sim.run((0.0, 50.0), 51)
        assert result.species is not None


class TestMichaelisMentenClosedForm:
    """Michaelis–Menten (tQSSA) reactions now carry a closed-form analytical
    Jacobian (GH #76 task 3) — previously they cleared the analytical-availability
    flag and the whole model ran on finite differences. The C++ closed form is
    validated here against the **generic sympy differentiation** of the exact
    tQSSA rate law (the reference the task specifies), evaluated through the real
    ``_dense_analytical_jacobian`` assembly."""

    def _mm_model(self, data_dir):
        import bngsim

        return bngsim.Model.from_net(str(data_dir / "mm_tqssa.net"))

    @staticmethod
    def _sympy_jacobian():
        """∂f/∂{E,S} of the tQSSA MM model E+S→E+P, by generic sympy
        differentiation of the same rate law the engine uses. Returns a callable
        (E,S,kcat,Km) -> 3×3 numpy Jacobian (species order E,S,P)."""
        E, S, kcat, Km = sp.symbols("E S kcat Km", positive=True)
        delta = S - Km - E
        D = sp.sqrt(delta**2 + 4 * Km * S)
        sFree = sp.Rational(1, 2) * (delta + D)
        rate = kcat * sFree * E / (Km + sFree)  # stat_factor = 1 in the fixture
        # f_E = 0 (enzyme is a catalyst: -1 reactant +1 product), f_S = -rate,
        # f_P = +rate.
        f = [sp.Integer(0), -rate, rate]
        Jmat = sp.Matrix([[sp.diff(fi, v) for v in (E, S, sp.Symbol("P"))] for fi in f])
        return sp.lambdify((E, S, kcat, Km), Jmat, "numpy")

    def test_available_and_complete(self, data_dir):
        core = self._mm_model(data_dir)._core
        assert core.analytical_jacobian_complete is True

    def test_dense_jacobian_matches_generic_sympy(self, data_dir):
        import numpy as np

        core = self._mm_model(data_dir)._core
        ns = 3
        kcat, Km = 1.0, 50.0  # fixture parameter values
        Jf = self._sympy_jacobian()
        # States with S>0 (so sFree>0, the unclamped smooth regime), spanning
        # E≷S; P is inert for the rate so its value is irrelevant.
        worst = 0.0
        for E, S, P in [(10.0, 100.0, 0.0), (80.0, 30.0, 12.0), (5.0, 5.0, 50.0), (1.0, 0.2, 3.0)]:
            flat = np.asarray(core._dense_analytical_jacobian(0.0, [E, S, P]), dtype=float)
            J_cpp = flat.reshape(ns, ns).T  # flat is column-major jac[j*ns+i]
            J_sym = np.asarray(Jf(E, S, kcat, Km), dtype=float)
            denom = np.maximum(np.maximum(np.abs(J_cpp), np.abs(J_sym)), 1e-9)
            worst = max(worst, float((np.abs(J_cpp - J_sym) / denom).max()))
        assert worst < 1e-10, f"MM closed form vs sympy worst rel err {worst}"

    def test_analytical_matches_fd_trajectory(self, data_dir):
        import bngsim
        import numpy as np

        def run(jac):
            sim = bngsim.Simulator(self._mm_model(data_dir), method="ode", jacobian=jac)
            return np.asarray(sim.run((0.0, 50.0), 101).species, dtype=float)

        analytical, fd = run("analytical"), run("fd")
        peak = np.maximum(np.abs(fd).max(axis=0), 1.0)
        assert float((np.abs(analytical - fd) / peak).max()) < 1e-8
