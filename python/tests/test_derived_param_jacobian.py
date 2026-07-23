"""Regression tests for issues #26 and #27 in
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
"""

from __future__ import annotations

import re

import pytest
from bngsim._codegen import _compute_derived_param_jacobian


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
