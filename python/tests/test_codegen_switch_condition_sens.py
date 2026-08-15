"""GH #68 — the analytic sensitivity RHS for *condition-bearing* Functional laws.

#67 declined every Functional rate law carrying an ``if()``. Some of those
declines were necessary and some were not, and telling them apart is the whole
of this stage:

    if(t >= sigma, beta, 0)     ->  admitted.  ∂f/∂sigma is a clean 0 in both
                                    branch interiors, and that IS the answer
                                    there — issue #48 supplies the rest as the
                                    crossing jump s⁺ = s⁻ + (f⁻−f⁺)·∂t*/∂sigma.

    if(I >= thresh, beta, 0)    ->  refused, until issue #150. The same clean 0
                                    comes back and is equally right in-branch,
                                    but the crossing moves with the trajectory
                                    and nothing supplied that term. #150 roots
                                    on the condition's residual and applies
                                    s⁺ = s⁻ + (f⁻−f⁺)·dt*/dθ there, so this is
                                    now admitted too — the rate-law twin of the
                                    state-dependent event trigger issue #144
                                    differentiates.

    if(I == thresh, beta, 0)    ->  still refused. An equality on a continuous
                                    trajectory holds on a measure-zero set, so
                                    there is no transversal crossing for either
                                    machinery to bracket.

The danger is that these are *indistinguishable downstream*. ``sympy.diff`` of a
``Piecewise`` w.r.t. a condition-only parameter returns ``0`` with no Dirac
delta, so ``_is_emittable`` never rejects it, the C compiles, the solver
converges, and the gradient is silently zero. There is no oracle that separates
them either: the emitted derivative and a naive finite difference agree with each
other and both miss the crossing. So what is tested here is the **gate**, and —
because a gate that disagrees with the machinery it is a proxy for is no gate at
all — that the gate, :func:`compute_switch_time_sens` and
:func:`state_switch_conditions` classify the same conditions the same way.

All three go through recognizers reached from one scope
(``switch_condition_scope``), which is issue #68's "do not add a third spelling"
requirement — and since #150 there are two machineries to keep apart as well as
aligned, because a crossing claimed by both would have its jump applied twice.
``TestTheGateAndTheDetectorsAgree`` is what actually holds them together: it
asserts *behavioural* agreement, which is the property #56 lost when #53 fixed
one copy of a predicate and not the other.
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


def _only_atom(body: str) -> str:
    """The single relational atom of a one-condition rate law, as written."""
    atoms = [a for cond in sw._iter_if_conditions(body) for a in sw._split_logical_atoms(cond)]
    assert len(atoms) == 1, f"{body!r} has {len(atoms)} atoms, not 1"
    return atoms[0]


# ─── the rule ──────────────────────────────────────────────────────────────


class TestTheRule:
    """A condition is admissible on exactly three grounds: it is a recognized
    clock threshold, it is a single comparison over live state whose residual
    the solver can root on (issue #150), or it names no symbol at all."""

    def test_a_clock_threshold_is_admitted(self, tmp_path):
        terms, reason = _decline(tmp_path, SWITCHED)
        assert reason is None
        assert terms  # the rate law does contribute ∂f/∂p columns

    def test_a_fitted_state_dependent_threshold_is_admitted_and_rooted(self, tmp_path):
        """Issue #68 refused this; issue #150 supplies what was missing.

        ``thresh`` is a model parameter a caller could plausibly fit and ``I`` is
        an observable, so the crossing moves with ``thresh`` AND, through the
        trajectory, with every other parameter. The Piecewise derivative's 0 is
        still the whole *in-branch* story; what it is missing is the jump at the
        crossing, and that crossing is now located as a CVODE root and jumped
        there. So the gate admits — but only because the detector registers it,
        which is asserted in the same breath: an admission with nothing behind it
        is the silent zero this gate exists to prevent."""
        core = _model(tmp_path, _with_law("if(I>=thresh,beta,0)*I"))._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        assert terms
        assert sw.state_switch_conditions(core) == ["I>=thresh"]

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

    def test_a_state_threshold_with_no_parameter_in_it_is_rooted_too(self, tmp_path):
        """The rule is not "does a *fitted* parameter sit in the condition". A
        state condition's crossing moves with **every** parameter through the
        trajectory — that is precisely the ``neuron`` hazard issue #52 widened
        the event guard to catch — so a literal threshold over state needs the
        same root and the same jump, and gets them. ``∂g/∂p`` is simply 0 here
        and the whole of ``dt*/dθ`` comes from ``∂g/∂x·s⁻``."""
        core = _model(tmp_path, _with_law("if(I>=40,beta,0)*I"))._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms
        assert sw.state_switch_conditions(core) == ["I>=40"]

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

    def test_a_threshold_that_reads_the_clock_back_goes_to_the_state_path(self, tmp_path):
        """``t < 2*t`` has no fixed crossing time, so the issue #48 detector
        skips it — and the gate must not admit what a detector skips. What
        changed is which detector picks it up: a BNGL counter clock is a
        *species*, so the comparison is a comparison over live state, and issue
        #150 roots on its residual and jumps at the crossing wherever the
        trajectory puts it. Asserted through both halves so "admitted" can never
        mean "admitted by nobody"."""
        core = _model(tmp_path, _with_law("if(t<2*t,beta,0)*I"))._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms
        records, _pinned = sw.compute_switch_time_sens(core, ["sigma", "thresh"], 0.0, 100.0)
        assert records == []  # the clock detector does NOT claim it
        assert sw.state_switch_conditions(core) == ["t<2*t"]

    def test_a_compound_condition_is_admitted_only_if_every_atom_is(self, tmp_path):
        """``&&`` splits into atoms and each is judged on its own — one bad atom
        is enough. The Lin2021 idiom ``(t>=sigma)&&(t<tau1)`` is the good case;
        pairing it with an atom NOBODY compensates must still refuse — and an
        equality is that atom: a continuous trajectory satisfies ``I==thresh``
        on a measure-zero set, so there is no transversal crossing to root on
        either. The mixed clock/state pairing is admitted, and each half goes to
        its own machinery: ``t>=sigma`` to the issue #48 stop time, ``I>=thresh``
        to the issue #150 crossing root."""
        ok, reason = _decline(
            tmp_path,
            _with_params("    8 tau1 8.0  # Constant\n", "if((t>=sigma)&&(t<tau1),beta,0)*I"),
        )
        assert reason is None and ok

        core = _model(tmp_path, _with_law("if((t>=sigma)&&(I>=thresh),beta,0)*I"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        records, _pinned = sw.compute_switch_time_sens(core, ["sigma"], 0.0, 100.0)
        assert records and sw.state_switch_conditions(core) == ["I>=thresh"]

        _terms, reason = _decline(tmp_path, _with_law("if((t>=sigma)&&(I==thresh),beta,0)*I"))
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

    def test_a_threshold_that_does_not_reduce_to_primaries_falls_to_the_state_path(self, tmp_path):
        """A clock threshold the issue #48 detector would *skip* leaves its
        ∂t*/∂p reaching nobody — so the gate may not admit it on the clock
        ground. Here the threshold reads an observable, which is exactly why it
        is not a constant expression over the parameters; and it is also exactly
        what makes the comparison a comparison over live state, which issue #150
        roots on and differentiates by the implicit function theorem. So the
        crossing IS compensated, just not by the machinery whose recognizer
        declined it."""
        core = _model(tmp_path, _with_law("if(t>=sigma*I,beta,0)*I"))._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms
        scope = sw.switch_condition_scope(core)
        assert not sw.clock_crossing_compensated("t>=sigma*I", scope)
        assert sw.state_switch_conditions(core) == ["t>=sigma*I"]

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
        del+gap`` is a ConstantExpression, so ``t >= base+lead`` reaches ``del``
        only through the derived-parameter DAG. The value and the partials walk
        that DAG the same way (GH #108) — ``base`` is one symbol carrying its own
        current value on both sides — and the aliasing has to survive it."""
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


def _squaring_chain(depth: int) -> dict[str, str]:
    """``d0 = base``, ``d{k+1} = dk*dk`` — a derived-parameter chain whose
    flattened text doubles at every level while its DAG has ``depth`` nodes."""
    return {"d0": "base"} | {f"d{k + 1}": f"d{k}*d{k}" for k in range(depth)}


class TestIssue108ThresholdValueSharesOnePreparation:
    """GH #108. ``_evaluate_threshold`` and ``_derived_expr_partials_numeric``
    now run *literally* the same preparation
    (``_codegen._prepare_derived_expr``) rather than two copies of one sequence.

    #105 was the fourth time a fix landed on one of a pair of expression parsers
    and not the other, and #99 promptly made it five: it moved the partials off
    the exponential ``_inline_derived_param_refs`` flattening and onto a DAG walk,
    and this site — which the issue's scope note did not name — kept flattening.
    Nothing failed, because the two agree on the *number*; what diverged was the
    cost and the set of expressions each could read.
    """

    def test_the_value_never_sees_the_flattened_expression(self, monkeypatch):
        """The #99 invariant, applied to the value: assert on the size of what
        reaches sympy, not on the clock.

        ``d{k+1} = dk*dk`` doubles the flattened text at every level, so the
        pre-#108 threshold value handed ``sp.parse_expr`` a 2^depth-character
        string; substituting each derived name's own value hands it the
        expression as written. Spying on sympy's parser rather than on a bngsim
        helper keeps this true of *any* implementation that stops flattening.
        """
        import sympy.parsing.sympy_parser as spp

        depth = 12
        chain = _squaring_chain(depth)
        param_idx = {"base": 0} | {f"d{k}": k + 1 for k in range(depth + 1)}
        # base^(2^k), which is what the engine's own ConstantExpression pass
        # would have left in each derived parameter's slot.
        values = [1.0001]
        for _ in range(depth + 1):
            values.append(values[-1] ** 2)
        values[1] = values[0]  # d0 = base

        expr = f"2*d{depth}"
        flattened = cg._inline_derived_param_refs(expr, chain)
        assert len(flattened) > 2**depth, "the fixture no longer blows up when flattened"

        seen: list[str] = []
        real_parse = spp.parse_expr
        monkeypatch.setattr(
            spp, "parse_expr", lambda s, **kw: (seen.append(s), real_parse(s, **kw))[1]
        )
        value = sw._evaluate_threshold(expr, param_idx, values, chain)

        assert seen, "nothing reached sympy at all — the spy is in the wrong place"
        assert max(len(s) for s in seen) < 200, (
            f"sympy was handed {max(len(s) for s in seen)} characters for a "
            f"{len(expr)}-character threshold — that is the flattened DAG"
        )
        assert value == pytest.approx(2.0 * values[param_idx[f"d{depth}"]])

    def test_the_value_and_its_partials_agree_through_the_dag(self):
        """The property #105 restored, now over a *nested* threshold: both halves
        must succeed or both must decline, or the caller drops the crossing."""
        depth = 6
        chain = _squaring_chain(depth)
        param_idx = {"base": 0} | {f"d{k}": k + 1 for k in range(depth + 1)}
        values = [2.0]
        for _ in range(depth + 1):
            values.append(values[-1] ** 2)
        values[1] = values[0]

        for expr in (f"2*d{depth}", f"d{depth}+d1", f"d{depth}/d0"):
            value = sw._evaluate_threshold(expr, param_idx, values, chain)
            partials = cg._derived_expr_partials_numeric(
                expr, {"base"}, param_idx, values, chain, warn_on_failure=False
            )
            assert value is not None and partials, f"{expr!r}: value={value}, {partials=}"
            assert set(partials) == {"base"}

    def test_a_constant_threshold_still_evaluates(self):
        """``_prepare_derived_expr`` declines an expression that names no
        parameter — for the differentiating callers that is a genuine zero, but a
        threshold of ``2*3600`` still has a crossing time. The value path opts
        out of that early return rather than growing a second parse."""
        assert sw._evaluate_threshold("2*3600", {"k": 0}, [1.0], {}) == pytest.approx(7200.0)


# ─── one predicate, two callers ────────────────────────────────────────────


class TestTheGateAndTheDetectorsAgree:
    """The anti-drift check, and the reason this stage did not write its own
    predicate. #56 happened because #53 fixed the logical-operator rewrite in
    one module and the identical hole survived in another — so what is asserted
    here is not "they share a function" (a refactor could undo that silently)
    but "they reach the same verdict on the same condition".

    Issue #150 added a *second* detector, and with it a second way to drift: a
    crossing claimed by both would have its jump applied twice, and one claimed
    by neither is the silent zero all of this exists to stop. So the invariant
    is now a partition — the gate admits iff EXACTLY ONE machinery compensates
    the crossing (or the crossing does not move at all)."""

    @pytest.mark.parametrize(
        ("body", "admitted", "by"),
        [
            ("if(t>=sigma,beta,0)*I", True, "clock"),
            ("if(t<sigma,beta,0)*I", True, "clock"),
            ("if(sigma<=t,beta,0)*I", True, "clock"),
            ("if(I>=thresh,beta,0)*I", True, "state"),
            ("if(t>=sigma*I,beta,0)*I", True, "state"),
            ("if(t<2*t,beta,0)*I", True, "state"),
            ("if(I==thresh,beta,0)*I", False, None),
            # Negation names the same surface as the comparison under it — issue
            # #234 peels it, so the partition claims this exactly once, on the
            # state side, just like the un-negated `if(I<=1,...)` spelling.
            # (`not(...)` and not `!(...)`: ExprTk rejects the operator form, so
            # the two spellings can only be compared at the splitter.)
            ("if(not(I>1),beta,0)*I", True, "state"),
            ("beta*(I>1)", False, None),
        ],
    )
    def test_admitted_iff_exactly_one_detector_compensates_it(self, tmp_path, body, admitted, by):
        core = _model(tmp_path, _with_law(body))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        gate_admits = reason is None

        # Detector 1 (issue #48), asked over a window that contains the crossing
        # with the threshold parameter requested. A record means it will stop
        # there and apply the jump.
        records, pinned = sw.compute_switch_time_sens(core, ["sigma", "thresh"], 0.0, 100.0)
        # Detector 2 (issue #150): a registered condition means the solver roots
        # on the crossing and applies the saltation jump there.
        states = sw.state_switch_conditions(core)

        assert gate_admits is admitted
        assert gate_admits == (bool(records) or bool(states)), (
            f"gate {'admits' if gate_admits else 'refuses'} {body!r} but "
            f"{len(records)} clock crossing(s) and {len(states)} state crossing(s) are "
            "compensated"
        )
        assert not (records and states), (
            f"{body!r} is claimed by BOTH detectors — its jump would be applied twice"
        )
        if by == "clock":
            assert records and pinned, "a compensated clock crossing must also pin its parameter"
        elif by == "state":
            assert states == [_only_atom(body)]

    def test_both_callers_reach_the_same_recognizer(self, tmp_path):
        """Structural companion to the behavioural check above: the scope the
        gate is built from is the one the detectors use, clocks and core
        included."""
        core = _model(tmp_path, SWITCHED)._core
        scope = sw.switch_condition_scope(core)
        assert "t" in scope.clocks, "the counter observable must be detected as a clock"
        assert "t" in scope.clock_symbols and "time" in scope.clock_symbols
        assert scope.core is core, "the gate must be able to ask the model about a residual"
        assert sw.uncompensated_condition_reason("if(t>=sigma,beta,0)*I", scope) is None
        assert sw.uncompensated_condition_reason("if(I>=thresh,beta,0)*I", scope) is None
        assert sw.uncompensated_condition_reason("if(I==thresh,beta,0)*I", scope) is not None

    def test_a_model_with_no_clock_does_not_read_a_time_looking_condition_as_a_clock(
        self, tmp_path
    ):
        """``t`` with no unit-rate counter behind it is just a name. The corpus
        has exactly this (an observable with an empty group), and the issue #48
        detector finds no crossing for it — so the gate must not admit it on the
        clock ground merely because of how it is spelled.

        It is still an observable, i.e. live state, so issue #150 does root on
        it: the admission is real, and it comes from the machinery that will
        actually run. What this pins is that the *clock* path stays out of it."""
        text = SWITCHED.replace("    3 0 4 kclock #_R3\n", "")
        core = _model(tmp_path, text)._core
        scope = sw.switch_condition_scope(core)
        assert "t" not in scope.clocks
        assert not sw.clock_crossing_compensated("t>=sigma", scope)
        assert sw.compute_switch_time_sens(core, ["sigma"], 0.0, 100.0) == ([], [])
        assert sw.state_switch_conditions(core) == ["t>=sigma"]


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
        """Declining is a fallback, not an error: a model whose crossing nothing
        compensates must still produce sensitivities, just via CVODES'
        internal DQ — with the issue #146 warning attached."""
        model = _model(tmp_path, _with_law(UNCOMPENSATED))
        run = _run_sens(model, ["beta", "gamma"], t_end=12.0)
        assert np.asarray(run.sensitivities).shape[2] == 2


class TestTheThresholdIsRecognisedByItsCrossingNotItsSpelling:
    """Issue #355. ``_clock_threshold_split_bare`` matches on *where the clock
    sits*: exactly one side must be the clock symbol itself. Two shapes that are
    the same threshold fail that test, and the corpus carries both —
    ``(time()-Tdam)<0`` (the PEtab spelling) and
    ``0>=Dam0-krepair*(time()-Tdam)`` (affine, parameter-dependent slope).

    Declining them is not free. The gate is per model, so one of these takes the
    whole model onto CVODES' difference quotient, which integrates straight
    through the crossing and drops the ``(f⁻−f⁺)·∂t*/∂p`` jump — the failure
    issue #232 measured at 53%. Three of issue #326's five open models are
    written the second way and stall at exactly their crossing.

    The recogniser now solves the residual for the clock instead of matching on
    it, so a threshold is recognised by the crossing it has and not by how it was
    typed.
    """

    def test_the_shifted_spelling_resolves_to_the_same_threshold(self):
        """``(t-sigma)>=0`` is ``t>=sigma``, and must produce the same threshold
        *expression* — that expression is what ``∂t*/∂p`` is differentiated
        from, so an equal-but-differently-spelled one would still be right while
        a different one would be silently wrong."""
        clocks = frozenset({"t"})
        assert sw._clock_threshold_split("t>=sigma", clocks) == ("t", "sigma")
        assert sw._clock_threshold_split_bare("(t-sigma)>=0", clocks) is None
        clock, threshold = sw._clock_threshold_split("(t-sigma)>=0", clocks)
        assert clock == "t"
        assert threshold.replace(" ", "") == "sigma"

    def test_an_affine_slope_resolves_to_the_crossing_in_closed_form(self):
        """The shape that blocks issue #326's stalling family. ``0 >= D0 −
        krep*(t − sigma)`` crosses at ``t = sigma + D0/krep``, which is a
        differentiable expression in three parameters — exactly what issue #48's
        jump needs and precisely what a lexical test cannot see."""
        clock, threshold = sw._clock_threshold_split(
            "0.0>=(D0-(krep*(t-sigma)))", frozenset({"t"})
        )
        assert clock == "t"
        import sympy as sp

        got = sp.sympify(threshold.replace("^", "**"))
        want = sp.sympify("sigma + D0/krep")
        assert sp.simplify(got - want) == 0

    def test_the_bare_spelling_is_answered_by_the_bare_test(self):
        """The blast-radius property, asserted rather than assumed: the widened
        recogniser is only ever reached by an atom the bare one declined, so no
        threshold recognised before this issue changes path or text."""
        clocks = frozenset({"t"})
        for atom in ("t>=sigma", "t<14", "sigma<=t", "t>2*sigma"):
            bare = sw._clock_threshold_split_bare(atom, clocks)
            assert bare is not None
            assert sw._clock_threshold_split(atom, clocks) == bare

    def test_the_clock_on_both_sides_is_still_not_a_clock_threshold(self):
        """``t<2*t`` is affine and does solve — to ``t*=0`` — but the bare test
        rejects it deliberately and the state path claims it. Relaxing *which
        side* the clock may sit in must not relax *how many* sides it may sit
        on, or a crossing moves between two machineries for no gain."""
        assert sw._clock_threshold_split("t<2*t", frozenset({"t"})) is None

    def test_a_non_linear_condition_is_declined(self):
        """``−b/a`` is the crossing only when the residual is degree 1 in the
        clock. Anything else has to keep declining rather than stop in the wrong
        place."""
        assert sw._clock_threshold_split("(t*t-sigma)<0", frozenset({"t"})) is None

    def test_a_state_threshold_is_not_captured_by_the_solve(self):
        """The solve must not annex the issue #150 path. ``I-thresh<0`` reads no
        clock at all, so it is not a clock atom in either recogniser."""
        assert sw._clock_threshold_split("(I-thresh)<0", frozenset({"t"})) is None

    def test_the_two_spellings_integrate_to_the_same_sensitivities(self, tmp_path):
        """The end-to-end statement, and the one that would catch a threshold
        that resolved to the wrong expression: the same model written both ways
        must produce the same tensor, ``sigma`` column included — which is
        entirely the crossing jump.

        Before this issue the shifted spelling did not merely differ, it took a
        different code path (difference quotient, no jump at all)."""
        params = ["beta", "gamma", "sigma"]
        bare = _run_sens(_model(tmp_path, SWITCHED), params, t_end=12.0)
        shifted = _run_sens(
            _model(tmp_path, _with_law("if((t-sigma)>=0,beta,0)*I"), name="shift.net"),
            params,
            t_end=12.0,
        )
        a, b = np.asarray(bare.sensitivities), np.asarray(shifted.sensitivities)
        assert a.shape == b.shape
        scale = max(float(np.max(np.abs(a))), 1e-300)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9 * scale)
        # Not vacuous: the switch column carries the jump on both spellings.
        assert float(np.max(np.abs(b[:, :, 2]))) > 1.0

    def test_the_shifted_spelling_gets_the_jump_rather_than_the_fallback(self, tmp_path):
        """Agreement above would also be satisfied by both spellings quietly
        landing on the difference quotient. This asserts the mechanism: the
        shifted model is admitted by the gate and its crossing is registered."""
        core = _model(tmp_path, _with_law("if((t-sigma)>=0,beta,0)*I"))._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        assert terms
        # and it is the CLOCK path that claims it, not issue #150's state root
        assert sw.state_switch_conditions(core) == []


# A crossing NOTHING compensates, post-#150. An equality on a continuous
# trajectory is satisfied on a measure-zero set, so there is no transversal
# crossing for the issue #150 root to bracket and no rising edge for issue #48
# to stop at — which is exactly what makes it the right fixture for "the
# difference quotient is not a correct fallback either".
UNCOMPENSATED = "if(I==thresh,beta,0)*I"


class TestTheDeclineDoesNotPromiseACorrectFallback:
    """Issue #146. Running on the difference quotient is a *correct* fallback for
    an underivable rate law — the problem is smooth, CVODES just answers it more
    slowly. It is not correct for an uncompensated crossing: the difference
    quotient integrates the variational equation straight through the crossing
    and drops the same jump the analytic path was declined for. On AMICI's
    ``nested_events`` every column came back a factor of ``f⁺/f⁻ = 2`` low after
    the ``Virus < 1`` crossing.

    Issue #150 emptied most of this class by fixing the underlying gap rather
    than labelling it: a single comparison over live state is now rooted and
    jumped, so it is admitted outright. What survives is the crossing no
    machinery can bracket, and the warning has to keep telling the truth about
    exactly those."""

    def test_an_uncompensated_crossing_is_tagged_as_such(self, tmp_path):
        from bngsim._switch_sensitivity import UncompensatedCrossingReason

        _terms, reason = _decline(tmp_path, _with_law(UNCOMPENSATED))
        assert isinstance(reason, UncompensatedCrossingReason)

    def test_a_compensated_state_crossing_is_not_tagged_at_all(self, tmp_path):
        """The companion that makes the tag mean something. Before #150 this
        very condition carried the tag; now nothing is declined for it, so there
        is no reason to carry and no warning to emit."""
        core = _model(tmp_path, _with_law("if(I>=thresh,beta,0)*I"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        assert sw.state_switch_conditions(core) == ["I>=thresh"]

    def test_the_tag_survives_the_reaction_context_wrapper(self, tmp_path):
        """The reason is re-wrapped with the reaction's name before it reaches
        the warning. An f-string over a ``str`` subclass is a plain ``str``, so
        that wrap is exactly where the distinction goes missing."""
        _terms, reason = _decline(tmp_path, _with_law(UNCOMPENSATED))
        assert "the Functional rate law for reaction" in reason

    def test_the_warning_says_the_fallback_does_not_recover_it(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _decline(tmp_path, _with_law(UNCOMPENSATED))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("does NOT recover the missing term" in m for m in msgs)
        assert any("issue #150" in m for m in msgs)
        assert not any("correct, but slower" in m for m in msgs)

    def test_an_underivable_law_keeps_the_correct_fallback_wording(self, tmp_path, caplog):
        """The control. A rate law that simply cannot be differentiated (#56/#66)
        really does fall back to a correct difference quotient, and must keep
        saying so — otherwise this change just moves the dishonesty."""
        import logging

        from bngsim._switch_sensitivity import UncompensatedCrossingReason

        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _terms, reason = _decline(tmp_path, _with_law("erf(I)*beta*I"))
        assert reason is not None
        assert not isinstance(reason, UncompensatedCrossingReason)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("correct, but slower" in m for m in msgs)
        assert not any("does NOT recover" in m for m in msgs)
