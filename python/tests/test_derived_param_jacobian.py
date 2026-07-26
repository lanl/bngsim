"""Regression tests for issues #26, #27 and #56 in
``bngsim._codegen._compute_derived_param_jacobian``.

#26 (minimum fix, already in place): widen the ``parse_expr`` except clause
to swallow ``tokenize.TokenError`` so a parameter literally named ``lambda``
no longer abandons the codegen .so build. With #26 alone, the derived param
is treated as having no analytic chain-rule contribution (``∂p_d/∂primary = 0``).

#27 (deeper fix): two preprocessing passes restore the analytic Jacobian
contribution that #26 silently zeroed out:

  Pass 1 — BNGL ``if(c, t, f)`` is rewritten to sympy
  ``Piecewise((t, c), (f, True))`` before parse_expr, so the conditional
  differentiates analytically.

  Pass 2 — Python-keyword-named primaries (``lambda``, ``if``, ``class``,
  ``for``, ...) are aliased to ``_BNG_KW_<name>`` before parse_expr, then
  round-tripped back to ``p[idx]`` when ``sp.ccode`` emits the partial.

#56: pass 1 only ever rewrote ``if()``, so a *compound* condition
(``if((sel>=1)&&(sel<10), kA, kB)``) — and equally ``^`` or ``not()`` — failed
to parse and the whole chain rule was silently zeroed. Pass 1 now runs the same
ExprTk-to-sympy pipeline the rate-law differentiator uses, and a failure that
zeroes a real contribution is logged instead of passing for a legitimate zero.
"""

from __future__ import annotations

import logging
import re

import pytest
from bngsim._codegen import (
    _compute_derived_param_jacobian,
    _derived_expr_partials_numeric,
    _derived_param_jacobian_checked,
)


class TestPythonKeywordParamNames:
    """Derived params that reference a parameter named with a Python keyword
    must not crash the codegen Jacobian path."""

    def test_lambda_param_does_not_raise_tokenerror(self):
        """#26 invariant: ``lambda*(1-phi)`` must not leak
        ``tokenize.TokenError`` out of ``_compute_derived_param_jacobian``.

        Pre-#26 it crashed; #26 widened the except to ``Exception`` so it
        returned ``None``; #27 then aliases the keyword and returns an
        analytic Jacobian. Either non-crashing outcome (None or a dict)
        satisfies the #26 contract — the value contract is asserted in
        ``TestIssue27CorpusShapes``."""
        result = _compute_derived_param_jacobian(
            "lambda*(1-phi)",
            primary_param_names={"lambda", "phi"},
            param_idx={"lambda": 0, "phi": 1},
        )
        assert result is None or isinstance(result, dict)

    @pytest.mark.parametrize(
        "expr,primary",
        [
            # Statement-keyword identifiers: parse_expr raises SyntaxError.
            # Pre-#26 these returned None; pre-#27 they kept returning None
            # because the keyword wasn't aliased. Post-#27 they alias to
            # ``_BNG_KW_<kw>`` before parse_expr and yield an analytic
            # contribution.
            ("if*2", {"if"}),
            ("class+1", {"class"}),
            ("for/2", {"for"}),
            ("lambda+1", {"lambda"}),
        ],
    )
    def test_keyword_named_params_return_jacobian_post_27(self, expr, primary):
        """Issue #27 deeper fix: Python-keyword-named primaries are aliased
        to safe placeholders before parse_expr and round-tripped back to
        ``p[idx]`` on the way out, so the analytic chain-rule term is
        recovered instead of silently zeroed."""
        param_idx = {p: i for i, p in enumerate(sorted(primary))}
        result = _compute_derived_param_jacobian(expr, primary, param_idx)
        assert result is not None
        # Single-primary expressions all have exactly one non-zero partial.
        assert set(result.keys()) == set(primary)
        # The emitted C must not leak the alias placeholder or the raw
        # keyword name (the latter would be a syntax error in C and is the
        # whole point of aliasing).
        for p_name, c_str in result.items():
            assert "_BNG_KW_" not in c_str
            assert not re.search(rf"\b{re.escape(p_name)}\b", c_str)

    def test_if_call_in_derived_param_returns_jacobian_post_27(self):
        """Issue #27: BNGL ``if(c, t, f)`` is translated to sympy Piecewise
        before parse_expr, so what used to be ``None`` is now a real analytic
        chain-rule contribution. ``∂(if(k1>0, k1, 1))/∂k1`` is the indicator
        ``[k1>0]`` (ignoring the boundary delta, per sympy's standard
        Piecewise convention)."""
        result = _compute_derived_param_jacobian(
            "if(k1>0, k1, 1)",
            primary_param_names={"k1"},
            param_idx={"k1": 0},
        )
        assert result is not None
        assert set(result.keys()) == {"k1"}
        c = result["k1"]
        # The C source is a ternary that switches on ``p[0] > 0``. We don't
        # assert exact whitespace because sympy's ccode formats Piecewise
        # multi-line.
        assert "p[0] > 0" in c
        assert "?" in c and ":" in c
        # Branch values: ∂/∂k1 of k1 == 1, ∂/∂k1 of 1 == 0.
        assert "1" in c and "0" in c

    def test_normal_expression_still_returns_jacobian(self):
        """Sanity check: the widened except didn't break the happy path."""
        result = _compute_derived_param_jacobian(
            "kf*kr",
            primary_param_names={"kf", "kr"},
            param_idx={"kf": 0, "kr": 1},
        )
        assert result is not None
        assert set(result.keys()) == {"kf", "kr"}
        # ∂(kf*kr)/∂kf = kr → references p[1]; ∂(kf*kr)/∂kr = kf → p[0].
        assert "p[1]" in result["kf"]
        assert "p[0]" in result["kr"]


class TestPrepareCodegenWithLambdaParam:
    """End-to-end: `prepare_codegen` must succeed on a model that names a
    parameter `lambda`. Pre-fix this path raised TokenError out of
    `_compute_derived_param_jacobian`, the bridge caught it broadly, and
    fell back to interpreted ODE."""

    def test_prepare_codegen_succeeds_on_lambda_named_param(self, tmp_path):
        # Smallest-possible reproducer: a derived param whose expression
        # references a primary param literally named `lambda`.
        net = tmp_path / "lambda_param.net"
        net.write_text(
            "# Reproducer for issue #26: derived param chained off `lambda`.\n"
            "begin parameters\n"
            "    1 lambda  0.5\n"
            "    2 phi     0.3\n"
            "    3 _rateLaw1  lambda*(1-phi)\n"
            "end parameters\n"
            "begin species\n"
            "    1 A() 100\n"
            "    2 B() 0\n"
            "end species\n"
            "begin reactions\n"
            "    1 1 2 _rateLaw1 #_R1\n"
            "end reactions\n"
            "begin groups\n"
            "    1 A_tot 1\n"
            "    2 B_tot 2\n"
            "end groups\n"
        )

        from bngsim._codegen import prepare_codegen

        so_path = prepare_codegen(str(net))
        assert so_path is not None
        assert so_path.exists()


class TestIssue27EndToEndForwardSens:
    """Issue #27 acceptance criterion: forward sensitivity on a
    ``scaling_example``-shaped model with ``sensitivity_params=['lambda']``
    must match CVODES internal FD. The shape is the corpus model that
    motivated the issue: primary param literally named ``lambda``, derived
    rate constant ``_rateLaw1 = lambda*(1-phi)`` driving the reaction.

    Pre-#27 the codegen Jacobian path silently zeroed ``∂_rateLaw1/∂lambda``
    (the alias-and-Piecewise passes weren't there), so the codegen analytic
    sens for ``lambda`` was wrong. Post-#27 the chain rule is re-established
    and codegen sens must match CVODES internal FD.
    """

    def _write_scaling_example_net(self, tmp_path):
        net = tmp_path / "scaling_example_repro.net"
        net.write_text(
            "# Reproducer for issue #27: derived rate constant chained off\n"
            "# a primary param literally named ``lambda``. Used to validate\n"
            "# the alias-and-round-trip path in _compute_derived_param_jacobian.\n"
            "begin parameters\n"
            "    1 lambda     0.5   # Constant\n"
            "    2 phi        0.3   # Constant\n"
            "    3 _rateLaw1  lambda*(1-phi)  # ConstantExpression\n"
            "end parameters\n"
            "begin species\n"
            "    1 A() 100\n"
            "    2 B() 0\n"
            "end species\n"
            "begin reactions\n"
            "    1 1 2 _rateLaw1 #_R1\n"
            "end reactions\n"
            "begin groups\n"
            "    1 A_tot 1\n"
            "    2 B_tot 2\n"
            "end groups\n"
        )
        return str(net)

    def test_codegen_sens_for_lambda_matches_cvodes_fd(self, tmp_path):
        import platform

        import bngsim
        import numpy as np
        from bngsim._codegen import CACHE_DIR, compute_model_hash, prepare_codegen

        net_path = self._write_scaling_example_net(tmp_path)

        # Force codegen .so re-generation so this test exercises the new
        # preprocessing passes rather than a stale cached artifact.
        h = compute_model_hash(net_path)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        cached = CACHE_DIR / f"rhs_{h}{suffix}"
        if cached.exists():
            cached.unlink()

        sample_times = list(np.linspace(0.0, 5.0, 51))

        # Reference: CVODES internal FD (codegen=False)
        m1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(m1, method="ode", sensitivity_params=["lambda"])
        r_fd = sim1.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)

        # Codegen analytic sens (codegen=True). prepare_codegen exercises the
        # full path: _compute_derived_param_jacobian for ``lambda*(1-phi)``
        # must return an analytic ∂_rateLaw1/∂lambda = 1-phi (#27 pass 2).
        prepare_codegen(net_path)
        m2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(
            m2, method="ode", sensitivity_params=["lambda"], codegen=True, net_path=net_path
        )
        r_cg = sim2.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, max_steps=10**6)

        # Species trajectories must agree first — same model, same params.
        np.testing.assert_allclose(r_fd.species, r_cg.species, atol=1e-8)

        # Sensitivities ∂y/∂lambda from both methods must match. Pre-#27 the
        # codegen path silently dropped the chain rule and produced (≈ 0)
        # contributions for ``lambda`` — they'd be way off.
        s_fd = r_fd.sensitivities[:, :, 0]
        s_cg = r_cg.sensitivities[:, :, 0]
        denom = np.maximum(np.abs(s_fd[1:]), np.abs(s_cg[1:]))
        mask = denom > 1e-9
        assert mask.any(), "FD reference sens is identically zero — bad test setup"
        rel = np.abs(s_fd[1:] - s_cg[1:])[mask] / denom[mask]
        assert rel.max() < 1e-3, (
            f"codegen analytic sens for ``lambda`` does not match CVODES FD "
            f"(max relerr={rel.max():.3e}); chain rule through _rateLaw1 = "
            f"lambda*(1-phi) likely dropped — see issue #27"
        )


class TestIssue27CorpusShapes:
    """Issue #27 deeper-fix coverage: the exact derived-param shapes from
    the three corpus models called out in the issue write-up. Pre-#27 each
    returned ``None`` and the codegen sensitivity RHS silently treated the
    derived param as an independent constant. Post-#27 each yields an
    analytic chain-rule contribution."""

    def test_scaling_example_lambda_times_one_minus_phi(self):
        # ode/scaling_example.bngl: _rateLaw1 = lambda*(1-phi)
        result = _compute_derived_param_jacobian(
            "lambda*(1-phi)",
            primary_param_names={"lambda", "phi"},
            param_idx={"lambda": 0, "phi": 1},
        )
        assert result is not None
        assert set(result.keys()) == {"lambda", "phi"}
        # ∂/∂lambda = 1 - phi → references p[1].
        assert "p[1]" in result["lambda"]
        # ∂/∂phi = -lambda → references p[0].
        assert "p[0]" in result["phi"]
        # Aliased name must not leak through to C output.
        for c_str in result.values():
            assert "_BNG_KW_" not in c_str
            assert "lambda" not in c_str

    def test_4var_model_T0_if_branch(self):
        # ode/4var_model.bngl: T0 = if(t<t_stim, T0_low, T0_high) — a model
        # parameter switched by an inequality. Verifies pass 1 (Piecewise)
        # handles a primary in every branch.
        result = _compute_derived_param_jacobian(
            "if(t_stim>0, T0_low, T0_high)",
            primary_param_names={"t_stim", "T0_low", "T0_high"},
            param_idx={"t_stim": 0, "T0_low": 1, "T0_high": 2},
        )
        assert result is not None
        # ∂/∂T0_low = [t_stim>0]; ∂/∂T0_high = [t_stim<=0]; ∂/∂t_stim = 0
        # (ignoring the boundary delta).
        assert "T0_low" in result and "T0_high" in result
        for c_str in result.values():
            assert "?" in c_str and ":" in c_str
            assert "p[0] > 0" in c_str

    def test_4var_model_with_FDC_combined_keyword_and_if(self):
        # Mixed shape covering both passes simultaneously: a keyword-named
        # primary inside an if(...) condition AND inside both branches.
        result = _compute_derived_param_jacobian(
            "if(lambda>0, lambda*phi, phi)",
            primary_param_names={"lambda", "phi"},
            param_idx={"lambda": 0, "phi": 1},
        )
        assert result is not None
        assert set(result.keys()) == {"lambda", "phi"}
        for c_str in result.values():
            assert "_BNG_KW_" not in c_str
            assert "lambda" not in c_str
            # The Piecewise switch must reference the keyword-aliased primary
            # as p[0], not as the raw alias or the keyword itself.
            assert "p[0] > 0" in c_str


class TestNestedDerivedParams:
    """Issue #41: a derived (ConstantExpression) parameter whose expression
    references ANOTHER derived parameter must be flattened to primaries before
    differentiation, so the forward-sensitivity chain rule reaches the
    underlying primary. Without ``derived_exprs`` the nested reference is a
    non-primary free symbol and the whole partial is silently dropped (``None``)
    — the pre-#41 behavior, preserved for callers that pass no map."""

    def test_nested_ref_dropped_without_map(self):
        # a2prime = 3*a1prime, a1prime = kcr. With no derived_exprs, a1prime is
        # an unknown (non-primary) free symbol → rejected, as before #41.
        result = _compute_derived_param_jacobian(
            "3*a1prime",
            primary_param_names={"kcr", "kf"},
            param_idx={"kcr": 0, "kf": 1, "a1prime": 2, "a2prime": 3},
        )
        assert result is None

    def test_nested_ref_resolved_with_map(self):
        # Same expression, now with the derived-expression map: a1prime inlines
        # to kcr, so ∂(3*a1prime)/∂kcr = 3.
        result = _compute_derived_param_jacobian(
            "3*a1prime",
            primary_param_names={"kcr", "kf"},
            param_idx={"kcr": 0, "kf": 1, "a1prime": 2, "a2prime": 3},
            derived_exprs={"a1prime": "kcr", "a2prime": "3*a1prime"},
        )
        assert result is not None
        assert set(result.keys()) == {"kcr"}
        # ∂/∂kcr = 3 (referencing the primary index for kcr, not a1prime).
        assert result["kcr"].replace(" ", "").lstrip("+") in {"3", "3.0", "3.0*1", "1*3.0"}
        assert "a1prime" not in result["kcr"]

    def test_nested_quotient_multiple_primaries(self):
        # igf1r-shaped: a2prime = (a2*a1prime*d1)/(a1*d2), a1prime = kcr. The
        # partial w.r.t. kcr must survive AND the directly-referenced primaries
        # (a1, a2, d1, d2) must all get their partials (all dropped pre-#41).
        primaries = {"kcr", "a1", "a2", "d1", "d2"}
        param_idx = {n: i for i, n in enumerate(sorted(primaries | {"a1prime", "a2prime"}))}
        result = _compute_derived_param_jacobian(
            "(a2*a1prime*d1)/(a1*d2)",
            primary_param_names=primaries,
            param_idx=param_idx,
            derived_exprs={"a1prime": "kcr", "a2prime": "(a2*a1prime*d1)/(a1*d2)"},
        )
        assert result is not None
        # kcr enters only through a1prime; a1/a2/d1/d2 enter directly.
        assert set(result.keys()) == {"kcr", "a1", "a2", "d1", "d2"}
        for c_str in result.values():
            assert "a1prime" not in c_str and "a2prime" not in c_str

    def test_three_level_nesting(self):
        # p3 -> p2 -> p1 -> base. All three inlined down to the primary ``base``.
        result = _compute_derived_param_jacobian(
            "2*p2",
            primary_param_names={"base"},
            param_idx={"base": 0, "p1": 1, "p2": 2, "p3": 3},
            derived_exprs={"p1": "base", "p2": "5*p1", "p3": "2*p2"},
        )
        assert result is not None
        assert set(result.keys()) == {"base"}  # ∂(2*(5*base))/∂base = 10
        assert result["base"].replace(" ", "").lstrip("+") in {"10", "10.0", "2*5.0", "10.0*1"}


# ─── Issue #56 ────────────────────────────────────────────────────────────

# The six spellings ExprTk accepts for logical AND / OR. Every one of them
# selects the ``kA`` branch at ``sel = 5``, so all six must produce the same
# chain rule as the simple single-comparison control.
_COMPOUND_CONDITIONS = [
    "if((sel>=1)&&(sel<10), kA, kB)",
    "if((sel>=1)&(sel<10), kA, kB)",
    "if((sel>=1) and (sel<10), kA, kB)",
    "if((sel>=1)||(sel>100), kA, kB)",
    "if((sel>=1)|(sel>100), kA, kB)",
    "if((sel>=1) or (sel>100), kA, kB)",
]


class TestIssue56CompoundConditions:
    """A compound condition in a derived parameter must yield the same analytic
    chain rule as the equivalent simple condition. Pre-#56 every one of these
    returned ``None`` and the caller read that as ``∂p_d/∂primary = 0`` — a
    gradient component that came back exactly zero rather than merely
    approximate."""

    _PRIMARIES = {"kA", "kB", "sel"}
    _IDX = {"kA": 0, "kB": 1, "sel": 2}

    @pytest.mark.parametrize("expr", _COMPOUND_CONDITIONS)
    def test_compound_condition_yields_chain_rule(self, expr):
        result = _compute_derived_param_jacobian(expr, self._PRIMARIES, self._IDX)
        assert result is not None, f"{expr!r} still falls back to the silent zero"
        # kA and kB each appear in one branch; sel only gates, so (ignoring the
        # boundary delta) it has no partial.
        assert set(result.keys()) == {"kA", "kB"}
        for c_str in result.values():
            # A C ternary switching on a C logical — never the word form, which
            # would not compile.
            assert "?" in c_str and ":" in c_str
            assert "&&" in c_str or "||" in c_str
            assert not re.search(r"\b(and|or|not)\b", c_str)

    @pytest.mark.parametrize("expr", _COMPOUND_CONDITIONS)
    def test_compound_partial_evaluates_like_the_simple_control(self, expr):
        """Not just "parses" — the emitted C must select the same branch the
        single-comparison control does at ``sel = 5``."""
        got = _compute_derived_param_jacobian(expr, self._PRIMARIES, self._IDX)
        want = _compute_derived_param_jacobian("if(sel>=1, kA, kB)", self._PRIMARIES, self._IDX)
        assert got is not None and want is not None
        assert _eval_c_partials(got, {0: 0.3, 1: 0.9, 2: 5.0}) == _eval_c_partials(
            want, {0: 0.3, 1: 0.9, 2: 5.0}
        )

    def test_and_binds_tighter_than_or(self):
        """``a && b || c`` is ``(a && b) || c`` in ExprTk and BNGL. Getting this
        backwards would silently pick the wrong branch's derivative."""
        got = _compute_derived_param_jacobian(
            "if(sel>1 && sel>2 || sel>3, kA, kB)", self._PRIMARIES, self._IDX
        )
        want = _compute_derived_param_jacobian(
            "if((sel>1 && sel>2) || sel>3, kA, kB)", self._PRIMARIES, self._IDX
        )
        other = _compute_derived_param_jacobian(
            "if(sel>1 && (sel>2 || sel>3), kA, kB)", self._PRIMARIES, self._IDX
        )
        assert got is not None and got == want
        assert got != other

    def test_caret_is_exponentiation_not_xor(self):
        """BNGL/ExprTk ``^`` is power. It reached ``parse_expr`` untranslated,
        where Python reads it as XOR — another silent zero."""
        result = _compute_derived_param_jacobian("kA^2*kB", self._PRIMARIES, self._IDX)
        assert result is not None
        assert set(result.keys()) == {"kA", "kB"}
        # ∂(kA²·kB)/∂kA = 2·kA·kB = 0.54, ∂/∂kB = kA² = 0.09
        vals = _eval_c_partials(result, {0: 0.3, 1: 0.9, 2: 5.0})
        assert vals["kA"] == pytest.approx(0.54)
        assert vals["kB"] == pytest.approx(0.09)

    def test_not_call_is_translated(self):
        """``not(x)`` is ExprTk's negation; untranslated it parsed as a call to
        an unknown function ``not``."""
        result = _compute_derived_param_jacobian(
            "if(not(sel<1), kA, kB)", self._PRIMARIES, self._IDX
        )
        assert result is not None
        # sel = 5 ⇒ ``not(sel<1)`` is true ⇒ the kA branch is live.
        vals = _eval_c_partials(result, {0: 0.3, 1: 0.9, 2: 5.0})
        assert vals["kA"] == pytest.approx(1.0)
        assert vals["kB"] == pytest.approx(0.0)


class TestIssue56NumericPartials:
    """The IC-seed / switch-threshold counterpart
    (``_derived_expr_partials_numeric``) had the identical hole: it returned
    ``{}`` on a compound condition and the caller left the seed at zero."""

    _PRIMARIES = {"R0", "scale", "sel"}
    _IDX = {"R0": 0, "scale": 1, "sel": 2}
    _VALUES = [100.0, 2.0, 5.0]

    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("if((sel>=1)&&(sel<10), R0*scale, R0)", {"R0": 2.0, "scale": 100.0}),
            ("if((sel>=1) and (sel<10), R0*scale, R0)", {"R0": 2.0, "scale": 100.0}),
            # The false branch is live here: sel = 5 is not > 100.
            ("if((sel>100)||(sel<0), R0*scale, R0)", {"R0": 1.0}),
            ("R0*scale^2", {"R0": 4.0, "scale": 400.0}),
        ],
    )
    def test_compound_condition_seeds_are_recovered(self, expr, expected):
        out = _derived_expr_partials_numeric(expr, self._PRIMARIES, self._IDX, self._VALUES, {})
        assert out == pytest.approx(expected)


class TestIssue56FailuresAreNotSilent:
    """Even with the logical operators handled, some expressions will still fail
    to differentiate — and ``None`` alone cannot be told apart from a primary
    that genuinely does not appear, because both mean zero downstream.
    ``_derived_param_jacobian_checked`` separates the two: a reason string only
    when a real contribution was lost."""

    _PRIMARIES = {"kA", "kB"}
    _IDX = {"kA": 0, "kB": 1}

    def test_unparseable_expression_reports_a_reason(self):
        jac, reason = _derived_param_jacobian_checked("kA +* kB", self._PRIMARIES, self._IDX)
        assert jac is None
        assert reason and "SyntaxError" in reason

    def test_underivable_expression_returns_none_instead_of_raising(self):
        """An un-inlined function call differentiates fine in sympy but cannot
        be rendered as C. That used to escape as ``PrintMethodNotImplementedError``
        and abort the whole codegen build, against this function's None-or-dict
        contract."""
        assert _compute_derived_param_jacobian("kA*foo(kB)", self._PRIMARIES, self._IDX) is None
        jac, reason = _derived_param_jacobian_checked("kA*foo(kB)", self._PRIMARIES, self._IDX)
        assert jac is None
        assert reason and "not expressible in C" in reason

    def test_genuine_absence_is_not_a_failure(self, caplog):
        """A derived parameter that references no primary is a real zero, not a
        failure — no reason, and no warning, or the signal is worthless."""
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert _derived_param_jacobian_checked("3*4", self._PRIMARIES, self._IDX) == (
                None,
                None,
            )
            assert (
                _derived_expr_partials_numeric("3*4", self._PRIMARIES, self._IDX, [1.0], {}) == {}
            )
        assert not caplog.records

    def test_primary_shadowing_a_sympy_name_is_refused(self):
        """A primary named ``And``/``Or``/``Piecewise`` would be shadowed by the
        sympy class we bind, and differentiate to a silent zero. Refuse it."""
        jac, reason = _derived_param_jacobian_checked(
            "And*2", primary_param_names={"And"}, param_idx={"And": 0}
        )
        assert jac is None
        assert reason == "a primary parameter shadows a sympy name"

    def test_ic_seed_path_warns_because_it_has_no_fallback(self, caplog):
        """The IC-seed path cannot decline the way the sensitivity RHS can — the
        seed is either computed or left at zero — so there a lost partial is
        reported as a warning."""
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert (
                _derived_expr_partials_numeric(
                    "R0*foo(scale)", {"R0", "scale"}, {"R0": 0, "scale": 1}, [1.0, 2.0], {}
                )
                == {}
            )
        assert any("could not differentiate" in r.message for r in caplog.records)


class TestIssue56SensRhsDeclinedNotWrong:
    """When a derived *rate constant* cannot be differentiated, the analytic
    sensitivity RHS must be declined outright so the run falls back to CVODES'
    internal difference quotient — correct, just slower. Emitting the RHS
    without that chain rule would report the gradient as exactly zero, which is
    strictly worse than being slow (issue #56, and the precedent set by #53).

    A derived parameter that is *not* a rate constant — a reporting quantity
    used only by an observable or a function — cannot affect this RHS, so it
    must not cost the model its analytic sensitivities."""

    _HEAD = (
        "begin parameters\n"
        "    1 kf    0.5   # Constant\n"
        "    2 scale 2.0   # Constant\n"
        "    3 {decl}\n"
        "end parameters\n"
    )
    _TAIL = (
        "begin species\n"
        "    1 A() 100\n"
        "    2 B() 0\n"
        "end species\n"
        "begin reactions\n"
        "    1 1 2 {rate}   #_R1\n"
        "end reactions\n"
        "begin groups\n"
        "    1 A_tot 1\n"
        "    2 B_tot 2\n"
        "end groups\n"
    )

    def _write(self, tmp_path, decl, rate, name="m.net"):
        p = tmp_path / name
        p.write_text(self._HEAD.format(decl=decl) + self._TAIL.format(rate=rate))
        return str(p)

    def test_undifferentiable_rate_constant_declines_the_rhs(self, tmp_path, caplog):
        from bngsim._codegen import generate_sens_rhs_c

        net = self._write(tmp_path, "kd  kf*foo(scale)  # ConstantExpression", "kd")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert generate_sens_rhs_c(net) is None, (
                "an undifferentiable derived rate constant must decline the analytic "
                "sens RHS, not emit one with a zeroed chain rule (issue #56)"
            )
        assert any("analytic sensitivity RHS is declined" in r.message for r in caplog.records)

    def test_differentiable_rate_constant_still_emits(self, tmp_path):
        from bngsim._codegen import generate_sens_rhs_c

        net = self._write(tmp_path, "kd  kf*scale  # ConstantExpression", "kd")
        assert generate_sens_rhs_c(net) is not None

    def test_compound_condition_rate_constant_still_emits(self, tmp_path):
        """The headline #56 case must now produce an RHS rather than decline."""
        from bngsim._codegen import generate_sens_rhs_c

        net = self._write(
            tmp_path, "kd  if((scale>=1)&&(scale<10), kf, 2*kf)  # ConstantExpression", "kd"
        )
        assert generate_sens_rhs_c(net) is not None

    def test_undifferentiable_non_rate_constant_does_not_decline(self, tmp_path, caplog):
        """``kd`` is never a rate constant here — only ``kf`` drives the
        reaction — so its undifferentiable expression is irrelevant to this RHS
        and must neither decline it nor warn."""
        from bngsim._codegen import generate_sens_rhs_c

        net = self._write(tmp_path, "kd  kf*foo(scale)  # ConstantExpression", "kf")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            assert generate_sens_rhs_c(net) is not None
        assert not caplog.records


class TestIssue56EndToEndForwardSens:
    """The reported symptom: a model whose rate constant is a derived parameter
    with a compound condition returns ``dB/dkA == 0.0`` exactly, while the
    forward trajectory is unaffected — so nothing looks broken and a fitting run
    simply never moves ``kA``.

    ``A -> B`` with rate constant ``k_eff`` has the closed form
    ``B(t) = A0·(1 - e^{-k·t})``, so ``∂B/∂kA = A0·t·e^{-k·t}`` exactly whenever
    the condition selects the ``kA`` branch. All conditions here do, at
    ``sel = 5``.
    """

    A0, KA, KB, SEL, T_END = 10.0, 0.3, 0.9, 5.0, 5.0

    def _build(self, cond_expr):
        import bngsim
        from bngsim._bngsim_core import ModelBuilder

        b = ModelBuilder()
        b.add_parameter("kA", self.KA, "", False)
        b.add_parameter("kB", self.KB, "", False)
        b.add_parameter("sel", self.SEL, "", False)
        b.add_parameter("k_eff", 0.0, cond_expr, True)  # ConstantExpression
        b.add_species("A", self.A0, False)
        b.add_species("B", 0.0, False)
        b.add_observable("Atot", [(0, 1.0)])
        b.add_observable("Btot", [(1, 1.0)])
        b.add_reaction([0], [1], "elementary", "k_eff", 1.0)
        return bngsim.Model(_core=b.build())

    @pytest.mark.parametrize("cond", ["if(sel>=1, kA, kB)", *_COMPOUND_CONDITIONS])
    def test_sens_matches_closed_form(self, cond):
        import bngsim
        import numpy as np

        model = self._build(cond)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kA"], codegen=True)
        model.reset()
        result = sim.run(
            sample_times=list(np.linspace(0.0, self.T_END, 11)),
            rtol=1e-10,
            atol=1e-12,
            max_steps=10**7,
        )

        # Forward trajectory: identical for every spelling (it was never the
        # broken part) and equal to the closed form.
        b_final = float(np.asarray(result.species)[-1][1])
        assert b_final == pytest.approx(self.A0 * (1.0 - np.exp(-self.KA * self.T_END)), rel=1e-6)

        expected = self.A0 * self.T_END * np.exp(-self.KA * self.T_END)
        got = float(np.asarray(result.sensitivities)[-1][1, 0])
        assert got == pytest.approx(expected, rel=1e-5), (
            f"dB/dkA = {got} for k_eff = {cond!r}; expected {expected}. A value of "
            f"exactly 0.0 means the derived-parameter chain rule was dropped — issue #56"
        )


def _eval_c_partials(result: dict[str, str], p_values: dict[int, float]) -> dict[str, float]:
    """Evaluate the emitted C partial-derivative sources at ``p_values``.

    The emitted strings are C ternaries over ``p[idx]``; Python's conditional
    expression has the same semantics once ``a ? b : c`` and the C logical
    operators are rewritten, which is enough to check which branch a Piecewise
    derivative selects.
    """
    p = [p_values[i] for i in sorted(p_values)]
    out: dict[str, float] = {}
    for name, c_str in result.items():
        py = c_str.replace("\n", " ").replace("&&", " and ").replace("||", " or ")
        # ``(cond ? (a) : (b))`` → ``((a) if cond else (b))``, innermost last.
        while "?" in py:
            m = re.search(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\?(.*)\)\Z", py, re.S)
            assert m, f"unparsed C ternary: {c_str!r}"
            cond, rest = m.group(1), m.group(2)
            colon = _depth0_colon(rest)
            py = f"(({rest[:colon]}) if ({cond}) else ({rest[colon + 1 :]}))"
        out[name] = float(eval(py, {"p": p}))  # noqa: S307
    return out


def _depth0_colon(s: str) -> int:
    """Index of the ternary's ``:``, i.e. the first one outside any parens."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
    raise AssertionError(f"no top-level ':' in C ternary tail: {s!r}")


class TestInlineDerivedParamRefs:
    """Direct coverage of the nested-reference flattening helper (issue #41)."""

    def test_single_level_is_noop(self):
        from bngsim._codegen import _inline_derived_param_refs

        # An expression already in primaries is returned untouched, so
        # single-level derived params stay byte-identical to pre-#41 output.
        assert _inline_derived_param_refs("chi*kon", {"a1prime": "kcr"}) == "chi*kon"

    def test_parenthesizes_to_preserve_precedence(self):
        from bngsim._codegen import _inline_derived_param_refs

        # a1prime = kcr + 1 must inline as (kcr + 1), not bare kcr + 1.
        out = _inline_derived_param_refs("2*a1prime", {"a1prime": "kcr + 1"})
        assert out == "2*(kcr + 1)"

    def test_whole_word_only(self):
        from bngsim._codegen import _inline_derived_param_refs

        # ``a1`` must not be substituted inside ``a1prime``.
        out = _inline_derived_param_refs("a1prime + a1", {"a1": "kcr"})
        assert out == "a1prime + (kcr)"

    def test_cycle_is_bounded(self):
        from bngsim._codegen import _inline_derived_param_refs

        # A pathological reference cycle must terminate (bounded passes) rather
        # than loop forever; the derived names simply remain in the output and
        # the caller's free-symbol check then rejects it.
        out = _inline_derived_param_refs("x", {"x": "y", "y": "x"}, max_passes=4)
        assert isinstance(out, str)
