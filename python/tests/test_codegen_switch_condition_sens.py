"""GH #68 — the analytic sensitivity RHS for *condition-bearing* Functional laws.

#67 declined every Functional rate law carrying an ``if()``. Some of those
declines were necessary and some were not, and telling them apart is the whole
of this stage:

    if(t >= sigma, beta, 0)     ->  admitted.  ∂f/∂sigma is a clean 0 in both
                                    branch interiors, and that IS the answer
                                    there — issue #48 supplies the rest as the
                                    crossing jump s⁺ = s⁻ + (f⁻−f⁺)·∂t*/∂sigma.

    if(I >= thresh, beta, 0)    ->  refused.   The same clean 0 comes back, but
                                    the crossing moves with the trajectory and
                                    nothing supplies that term. This is the
                                    rate-law twin of the state-dependent event
                                    trigger issue #52 refuses.

The danger is that the two are *indistinguishable downstream*. ``sympy.diff`` of
a ``Piecewise`` w.r.t. a condition-only parameter returns ``0`` with no Dirac
delta, so ``_is_emittable`` never rejects it, the C compiles, the solver
converges, and the gradient is silently zero. There is no oracle that separates
them either: the emitted derivative and a naive finite difference agree with each
other and both miss the crossing. So what is tested here is the **gate**, and —
because a gate that disagrees with the detector it is a proxy for is no gate at
all — that the gate and :func:`compute_switch_time_sens` classify the same
conditions the same way.

Both callers go through one recognizer (``_clock_threshold_split``) reached from
one scope (``switch_condition_scope``), which is issue #68's "do not add a third
spelling" requirement. ``TestTheGateAndTheDetectorAgree`` is what actually holds
them together: it asserts *behavioural* agreement, which is the property #56
lost when #53 fixed one copy of a predicate and not the other.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim import _switch_sensitivity as sw

pytest.importorskip("sympy")


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

# TestEndToEnd below carried an xfail(strict) quarantine for GH #85 — under the
# MIR JIT backend a Functional model *constructed with* ``sensitivity_params``
# did not compile, because the JIT prelude never supplied the ``size_t`` the
# GH #198 ``bngsim_codegen_output_sens`` block names. Fixed in mir_jit.hpp;
# it runs on both backends now, and test_codegen_jit_prelude.py owns the
# regression.


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# A counter-clock SIR: ``counter()`` is a species with a zeroth-order synthesis
# at rate 1, exposed as the observable ``t``. That is the BNGL idiom for making
# simulation time readable by a rate law, and it is what all eight corpus models
# this stage unblocks are written with — so the clock here is *detected*, not
# assumed, exactly as in the field.

SWITCHED = """\
begin parameters
    1 S0      1000  # Constant
    2 I0      1  # Constant
    3 beta    0.002  # Constant
    4 gamma   0.15  # Constant
    5 sigma   3.0  # Constant
    6 kclock  1  # Constant
    7 thresh  40.0  # Constant
end parameters
begin functions
    1 betaI() if(t>=sigma,beta,0)*I
end functions
begin species
    1 person(state~S) S0
    2 person(state~I) I0
    3 person(state~R) 0
    4 counter() 0
end species
begin reactions
    1 1 2 betaI #_R1
    2 2 3 gamma #_R2
    3 0 4 kclock #_R3
end reactions
begin groups
    1 S                    1
    2 I                    2
    3 R                    3
    4 t                    4
end groups
"""


def _with_law(body: str) -> str:
    return SWITCHED.replace("    1 betaI() if(t>=sigma,beta,0)*I\n", f"    1 betaI() {body}\n")


def _with_params(extra: str, body: str) -> str:
    """Same fixture with an extra parameter block line and a swapped rate law."""
    return _with_law(body).replace("end parameters\n", f"{extra}end parameters\n")


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _decline(tmp_path, text, name="m.net"):
    """``(terms, reason)`` from the real derivation path."""
    core = _model(tmp_path, text, name)._core
    return cg._functional_dfdp_terms(core, core.codegen_data())


# ─── the rule ──────────────────────────────────────────────────────────────


class TestTheRule:
    """A condition is admissible on exactly two grounds: it is a recognized
    clock threshold, or it names no symbol at all."""

    def test_a_clock_threshold_is_admitted(self, tmp_path):
        terms, reason = _decline(tmp_path, SWITCHED)
        assert reason is None
        assert terms  # the rate law does contribute ∂f/∂p columns

    def test_a_fitted_state_dependent_threshold_is_refused(self, tmp_path):
        """Issue #68's definition of done. ``thresh`` is a model parameter a
        caller could plausibly fit, and ``I`` is an observable — so the crossing
        moves with ``thresh`` and the Piecewise derivative's 0 is simply wrong.
        Answering it would be worse than declining: the answer looks converged.
        """
        terms, reason = _decline(tmp_path, _with_law("if(I>=thresh,beta,0)*I"))
        assert terms == {}
        assert reason is not None
        assert "'thresh'" in reason
        assert "I>=thresh" in reason
        assert "not a recognized clock threshold" in reason

    def test_the_piecewise_zero_is_what_would_have_been_shipped(self, tmp_path):
        """Why the gate has to be a *pre-scan*: nothing further down the pipeline
        objects. Differentiating the refused law by hand yields a clean 0 for
        ``thresh`` — no Dirac delta for ``_is_emittable`` to catch, no C the
        compiler would reject, no warning anyone would see."""
        import sympy as sp
        from bngsim._jacobian import _exprtk_to_sympy, _is_emittable

        expr = _exprtk_to_sympy("if(I>=thresh,beta,0)*I")
        d = sp.diff(expr, sp.Symbol("thresh"))
        assert d == 0
        assert _is_emittable(expr)

    def test_a_state_threshold_with_no_parameter_in_it_is_refused_too(self, tmp_path):
        """The rule is not "does a *fitted* parameter sit in the condition". A
        state condition's crossing moves with **every** parameter through the
        trajectory — that is precisely the ``neuron`` hazard issue #52 widened
        the event guard to catch — so a literal threshold over state is no
        safer."""
        terms, reason = _decline(tmp_path, _with_law("if(I>=40,beta,0)*I"))
        assert terms == {}
        assert reason is not None and "reads model state" in reason

    def test_a_literal_comparison_does_not_block(self, tmp_path):
        """``0>0`` is a compile-time constant with no crossing at all. Corpus
        models really do carry these (BNG2.pl emits them from a rule whose
        guard folded away), so refusing them would cost real models for no
        correctness gain."""
        terms, reason = _decline(tmp_path, _with_law("if(0>0,beta,beta)*I"))
        assert reason is None and terms

    def test_a_comparison_outside_an_if_is_refused(self, tmp_path):
        """``beta*(I>1)`` is the boolean-as-a-number idiom: a branch with no
        ``if()`` head, so the condition scan would not see it at all. Checked
        over the whole expression for that reason."""
        terms, reason = _decline(tmp_path, _with_law("beta*(I>1)"))
        assert terms == {}
        assert reason is not None and "not inside an if() condition" in reason

    def test_a_threshold_that_reads_the_clock_back_is_refused(self, tmp_path):
        """``t < 2*t`` has no fixed crossing time, so the detector skips it and
        the gate must not admit what the detector skips."""
        terms, reason = _decline(tmp_path, _with_law("if(t<2*t,beta,0)*I"))
        assert terms == {}
        assert reason is not None

    def test_a_compound_condition_is_admitted_only_if_every_atom_is(self, tmp_path):
        """``&&`` splits into atoms and each is judged on its own — one bad atom
        is enough. The Lin2021 idiom ``(t>=sigma)&&(t<tau1)`` is the good case;
        pairing it with a state atom must still refuse."""
        ok, reason = _decline(
            tmp_path,
            _with_params("    8 tau1 8.0  # Constant\n", "if((t>=sigma)&&(t<tau1),beta,0)*I"),
        )
        assert reason is None and ok
        _terms, reason = _decline(tmp_path, _with_law("if((t>=sigma)&&(I>=thresh),beta,0)*I"))
        assert reason is not None and "'thresh'" in reason

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            ("beta*abs(I)", "abs()"),
            ("beta*max(I,1)", "max()"),
            ("beta*min(I,1)", "min()"),
            ("beta*floor(I)", "floor()"),
        ],
    )
    def test_the_non_conditional_constructs_are_still_refused(self, tmp_path, body, fragment):
        """``allow_conditions`` waives the branch-selecting class and nothing
        else. A kink in the *state* has no crossing time, so there is no #48
        jump that could compensate it and no version of this gate that admits
        it."""
        terms, reason = _decline(tmp_path, _with_law(body))
        assert terms == {}
        assert reason is not None and fragment in reason


# ─── the derived-threshold chain ───────────────────────────────────────────


class TestDerivedThresholds:
    def test_a_derived_threshold_is_admitted(self, tmp_path):
        """``sigma = t0 + t_delta`` is how the corpus models actually spell a
        fitted onset. The gate must clear ``t0``/``t_delta`` too, because the
        detector chain-rules the jump through to them."""
        text = SWITCHED.replace(
            "    5 sigma   3.0  # Constant\n",
            "    5 t0      2.0  # Constant\n"
            "    8 t_delta 1.0  # Constant\n"
            "    9 sigma   t0+t_delta  # ConstantExpression\n",
        )
        terms, reason = _decline(tmp_path, text)
        assert reason is None and terms

    def test_a_threshold_that_does_not_reduce_to_primaries_is_refused(self, tmp_path):
        """A clock threshold the detector would *skip* is no better than a state
        threshold: its ∂t*/∂p never reaches the solver, so the Piecewise zero
        would be the whole gradient. Here the threshold reads an observable, so
        it is not a constant expression over the parameters."""
        terms, reason = _decline(tmp_path, _with_law("if(t>=sigma*I,beta,0)*I"))
        assert terms == {}
        assert reason is not None

    def test_the_switch_parameter_gets_no_dfdp_column(self, tmp_path):
        """``sigma`` enters f only through the condition, so its analytic ∂f/∂p
        is 0 and it must contribute **no** term — the jump is issue #48's to
        apply, and a spurious in-branch term would double-count it."""
        model = _model(tmp_path, SWITCHED)
        core = model._core
        data = core.codegen_data()
        names = [p["name"] for p in data["parameters"]]
        terms, reason = cg._functional_dfdp_terms(core, data)
        assert reason is None
        by_param = {names[k] for k, _c in terms[0]}
        assert "beta" in by_param
        assert "sigma" not in by_param

    def test_the_in_branch_derivative_keeps_its_condition(self, tmp_path):
        """∂f/∂beta is *not* constant — it is 0 before the switch, so the emitted
        term is ``I·1{t>=sigma}``, not ``I``. Losing the indicator would be wrong
        in the opposite direction from losing the jump, and just as quiet.

        ``sigma`` is ``p[4]`` and the clock observable ``t`` is ``obs[3]``, so the
        comparison must be between exactly those two — a ternary over the wrong
        pair would still compile and still converge."""
        model = _model(tmp_path, SWITCHED)
        core = model._core
        names = [p["name"] for p in core.codegen_data()["parameters"]]
        terms, _ = cg._functional_dfdp_terms(core, core.codegen_data())
        c_expr = dict(terms[0])[names.index("beta")]
        assert "?" in c_expr and f"p[{names.index('sigma')}]" in c_expr

        src = cg.generate_sens_from_model(model, functional=True)
        assert src is not None
        # Both halves must carry it: ∂f/∂p above, and ∂(rate)/∂I inside J·v.
        assert sum("?" in ln for ln in src.splitlines()) >= 2


def _kw_threshold(name: str) -> str:
    """``SWITCHED`` with the onset spelled ``<name>+gap`` instead of ``sigma``.

    Arithmetic on purpose: :func:`_evaluate_threshold` short-circuits on a bare
    parameter name before it ever reaches sympy, so only the compound form
    exercises the parse — which is exactly why issue #105 went unnoticed.
    """
    text = SWITCHED.replace(
        "    5 sigma   3.0  # Constant\n",
        f"    5 {name}   2.0  # Constant\n    8 gap     1.0  # Constant\n",
    )
    return text.replace("if(t>=sigma,beta,0)*I", f"if(t>={name}+gap,beta,0)*I")


class TestPythonKeywordThresholdParameters:
    """Issue #105. A threshold parameter whose *name* is a Python keyword —
    ``del``, ``lambda``, ``as`` — must be treated exactly like any other name.

    ``parse_expr("del+gap")`` raises at **tokenization**, so passing ``del`` in
    ``local_dict`` cannot rescue it; the name has to be rewritten to an alias
    first, which is what ``_sympy_symbol_alias_map`` (issue #27) already does on
    the emitting path. Before the fix ``_evaluate_threshold`` returned ``None``
    while ``_derived_expr_partials_numeric`` returned the correct partials for
    the *same* string, so the detector emitted no crossing record at all and the
    model dropped into issue #82's mxstep stall.

    43 of the 46 corpus models with a keyword-named parameter use ``lambda``,
    which is why this is parametrized over more than the one name that surfaced
    it.
    """

    @pytest.mark.parametrize("name", ["onset", "del", "lambda", "as"])
    def test_the_gate_and_the_detector_are_blind_to_the_name(self, tmp_path, name):
        """The whole assertion is an equality against the ordinary-name case: a
        keyword name is not a *feature*, it must simply not be visible anywhere
        downstream."""
        core = _model(tmp_path, _kw_threshold(name), name=f"{name}.net")._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None, reason
        records, pinned = sw.compute_switch_time_sens(core, [name, "gap"], 0.0, 100.0)
        # onset = <name> + gap = 2.0 + 1.0, and ∂t*/∂p is 1 for both.
        assert records == [(3.0, 3, 3.0, [1.0, 1.0])]
        assert pinned

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("del+gap", 2.5),
            ("del+(1.0*stepT)", 1801.0),  # the MODEL1710030000 trigger shape
            ("2*gap", 3.0),  # an ordinary name still evaluates as before
            ("del", 1.0),  # bare name: the short circuit, never sympy
            ("14", 14.0),
        ],
    )
    def test_evaluate_threshold_parses_keyword_arithmetic(self, expr, expected):
        param_idx = {"del": 0, "stepT": 1, "gap": 2}
        values = [1.0, 1800.0, 1.5]
        assert sw._evaluate_threshold(expr, param_idx, values, {}) == expected

    def test_the_value_and_its_partials_agree_on_what_they_can_handle(self):
        """The invariant the fix restores, and the one the docstring already
        claimed: both come from the same round trip. A threshold with partials
        but no value (or the reverse) makes the caller drop the crossing, which
        is a silent loss of the entire gradient for that parameter."""
        param_idx = {"del": 0, "gap": 1}
        values = [1.0, 1.5]
        for expr in ("del+gap", "del*2+gap", "gap-del", "del^2"):
            value = sw._evaluate_threshold(expr, param_idx, values, {})
            partials = cg._derived_expr_partials_numeric(
                expr, set(param_idx), param_idx, values, {}, warn_on_failure=False
            )
            assert (value is None) == (not partials), (
                f"{expr!r}: value={value} but partials={partials}"
            )

    def test_a_derived_keyword_threshold_chain_rules_through(self, tmp_path):
        """The keyword name need not appear in the condition at all: ``base =
        del+gap`` is a ConstantExpression, and inlining runs *before* the parse,
        so ``t >= base+lead`` still hands ``del+gap+lead`` to sympy. This is the
        ordering the fix depends on — alias after inlining, not before."""
        text = SWITCHED.replace(
            "    5 sigma   3.0  # Constant\n",
            "    5 del     1.0  # Constant\n"
            "    8 gap     1.0  # Constant\n"
            "    9 lead    1.0  # Constant\n"
            "   10 base    del+gap  # ConstantExpression\n",
        ).replace("if(t>=sigma,beta,0)*I", "if(t>=base+lead,beta,0)*I")
        core = _model(tmp_path, text)._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None, reason
        records, _pinned = sw.compute_switch_time_sens(core, ["del", "gap", "lead"], 0.0, 100.0)
        assert records == [(3.0, 3, 3.0, [1.0, 1.0, 1.0])]

    def test_colliding_aliases_bail_rather_than_merge(self):
        """``_sympy_symbol_alias_map`` returns ``None`` when two parameters would
        share one symbol. Evaluating anyway would silently substitute one
        parameter's value for the other's, so ``None`` (which makes the caller
        decline the crossing) is the only safe answer."""
        param_idx = {"del": 0, "_BNG_KW_del": 1}
        assert sw._evaluate_threshold("del+_BNG_KW_del", param_idx, [1.0, 2.0], {}) is None


# ─── one predicate, two callers ────────────────────────────────────────────


class TestTheGateAndTheDetectorAgree:
    """The anti-drift check, and the reason this stage did not write its own
    predicate. #56 happened because #53 fixed the logical-operator rewrite in
    one module and the identical hole survived in another — so what is asserted
    here is not "they share a function" (a refactor could undo that silently)
    but "they reach the same verdict on the same condition"."""

    @pytest.mark.parametrize(
        ("body", "admitted"),
        [
            ("if(t>=sigma,beta,0)*I", True),
            ("if(t<sigma,beta,0)*I", True),
            ("if(sigma<=t,beta,0)*I", True),
            ("if(I>=thresh,beta,0)*I", False),
            ("if(t>=sigma*I,beta,0)*I", False),
            ("if(t<2*t,beta,0)*I", False),
        ],
    )
    def test_admitted_iff_the_detector_compensates_it(self, tmp_path, body, admitted):
        core = _model(tmp_path, _with_law(body))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        gate_admits = reason is None

        # The detector, asked over a window that contains the crossing, with the
        # threshold parameter requested. A record means it will stop there and
        # apply the jump; nothing means the Piecewise zero stands alone.
        records, pinned = sw.compute_switch_time_sens(core, ["sigma", "thresh"], 0.0, 100.0)
        detector_compensates = bool(records)

        assert gate_admits is admitted
        assert gate_admits == detector_compensates, (
            f"gate {'admits' if gate_admits else 'refuses'} {body!r} but the detector "
            f"{'does' if detector_compensates else 'does not'} compensate it"
        )
        if detector_compensates:
            assert pinned, "a compensated crossing must also pin its parameter"

    def test_both_callers_reach_the_same_recognizer(self, tmp_path):
        """Structural companion to the behavioural check above: the scope the
        gate is built from is the one the detector uses, clocks included."""
        core = _model(tmp_path, SWITCHED)._core
        scope = sw.switch_condition_scope(core)
        assert "t" in scope.clocks, "the counter observable must be detected as a clock"
        assert "t" in scope.clock_symbols and "time" in scope.clock_symbols
        assert sw.uncompensated_condition_reason("if(t>=sigma,beta,0)*I", scope) is None
        assert sw.uncompensated_condition_reason("if(I>=thresh,beta,0)*I", scope) is not None

    def test_a_model_with_no_clock_refuses_a_time_looking_condition(self, tmp_path):
        """``t`` with no unit-rate counter behind it is just a name. The corpus
        has exactly this (an observable with an empty group), and the detector
        finds no crossing for it — so the gate must refuse rather than trust the
        spelling."""
        text = SWITCHED.replace("    3 0 4 kclock #_R3\n", "")
        core = _model(tmp_path, text)._core
        scope = sw.switch_condition_scope(core)
        assert "t" not in scope.clocks
        assert sw.uncompensated_condition_reason("if(t>=sigma,beta,0)*I", scope) is not None


# ─── end to end ────────────────────────────────────────────────────────────


def _run_sens(model, params, t_end, n=61):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    return sim.run(t_span=(0.0, t_end), n_points=n, rtol=1e-11, atol=1e-11)


@requires_cc
class TestEndToEnd:
    def test_the_analytic_rhs_reproduces_the_difference_quotient(self, tmp_path, monkeypatch):
        """The migration check, and the strongest statement available for this
        stage: switching a switch-bearing model onto the analytic RHS must not
        move the answer — including the ``sigma`` column, which is entirely the
        #48 jump on both paths.

        There is no tighter oracle here. A trajectory finite difference across a
        moving discontinuity is only O(h)-accurate near the crossing, and both
        paths miss it by the *same* amount (measured: 1.3e-3 relative for both),
        so it cannot separate them. What it can do is confirm neither path
        regressed, which the previous stage's oracles already cover for the
        smooth part."""
        params = ["beta", "gamma", "sigma"]
        analytic = _run_sens(_model(tmp_path, SWITCHED), params, t_end=12.0)
        with monkeypatch.context() as mp:
            mp.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
            dq = _run_sens(_model(tmp_path, SWITCHED, name="dq.net"), params, t_end=12.0)
        a, d = np.asarray(analytic.sensitivities), np.asarray(dq.sensitivities)
        assert a.shape == d.shape
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(d))), 1e-300)
        np.testing.assert_allclose(a, d, rtol=1e-4, atol=1e-5 * scale)
        # And the switch column is not trivially zero on either path — otherwise
        # the agreement above would be vacuous.
        assert float(np.max(np.abs(a[:, :, 2]))) > 1.0

    def test_the_switch_jump_still_fires_under_the_analytic_rhs(self, tmp_path, caplog):
        """#48 stops at the crossing and jumps across it independently of where
        ∂f/∂p came from. Losing that on the analytic path would zero the whole
        ``sigma`` column while every other column stayed right — the failure
        mode most likely to pass a casual review."""
        import logging

        with caplog.at_level(logging.INFO, logger="bngsim"):
            run = _run_sens(_model(tmp_path, SWITCHED), ["sigma"], t_end=12.0)
        assert any("Switch-time forward sensitivity" in r.getMessage() for r in caplog.records)
        assert float(np.max(np.abs(np.asarray(run.sensitivities)))) > 1.0

    def test_a_refused_model_still_runs_on_the_difference_quotient(self, tmp_path):
        """Declining is a fallback, not an error: the state-threshold model must
        still produce sensitivities, just via CVODES' internal DQ."""
        model = _model(tmp_path, _with_law("if(I>=thresh,beta,0)*I"))
        run = _run_sens(model, ["beta", "gamma"], t_end=12.0)
        assert np.asarray(run.sensitivities).shape[2] == 2
