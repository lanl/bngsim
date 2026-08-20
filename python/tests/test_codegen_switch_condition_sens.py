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

    if(I == thresh, beta, 0)    ->  refused, until issue #381. A continuous
                                    trajectory holds the equality for an
                                    instant, not an interval — but the surface
                                    bounding that instant is `I − thresh = 0`,
                                    which is where `I >= thresh` changes branch
                                    as well, so #150's root claims it too. What
                                    made refusing it expensive is that atoms are
                                    judged one at a time: MODEL2003190004 spells
                                    `APC <= 0.2` as `(APC == 0.2) or (APC < 0.2)`
                                    and the equality half declined the whole
                                    model, onto a difference quotient that then
                                    stalled at that very crossing.

    if(time()*time() >= thresh, beta, 0)
                                ->  admitted, since issue #418. A clock threshold
                                    #48's affine solver cannot invert for t*, but
                                    a single power of the clock still has one
                                    crossing in closed form — `time = sqrt(thresh)`
                                    — so `_clock_monomial_threshold` solves it and
                                    #48 jumps it like any affine clock threshold.

    if((time()-5)*(time()-5) >= thresh, beta, 0)
                                ->  admitted, since issue #421. A quadratic in
                                    the clock has TWO crossings, both still in
                                    closed form (the quadratic formula), so the
                                    recogniser answers with a list and #48 jumps
                                    each of them. A clock threshold cubic or
                                    higher stays refused.

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
    the solver can root on (issue #150), or it cannot cross at all because it is
    written over run-constants (issue #382 — ``0>0``, and equally a comparison
    between frozen parameters)."""

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

    def test_a_comparison_between_run_constants_does_not_block(self, tmp_path):
        """Issue #382 — the same compile-time constant, spelled with names.

        ``thresh`` is a plain ``# Constant`` parameter, so ``thresh>1e7`` is
        false at the first step and false at the last: no crossing, nothing to
        compensate, and the in-branch derivative is the whole story. Refusing it
        cost four ``MODEL09112*`` corpus models the analytic sensitivity RHS for
        the WHOLE model, and CVODES' difference quotient — the fallback the
        refusal hands the solve to — then failed CV_CONV_FAILURE at t=1 on every
        one of them.

        Asserted in the same breath that NEITHER detector claims it: unlike
        grounds 1 and 2, an admission here is meant to have nothing behind it,
        because there is no crossing for anything to be behind."""
        core = _model(tmp_path, _with_law("if(thresh>1e7,beta,0)*I"))._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms
        assert sw.state_switch_conditions(core) == []
        records, _pinned = sw.compute_switch_time_sens(core, ["thresh"], 0.0, 100.0)
        assert records == []

    def test_the_branch_taken_by_a_run_constant_condition_is_the_one_measured(self, tmp_path):
        """Admitting the condition is only right if the columns it then yields
        are the derivatives of the branch that actually runs. Asserted
        numerically, because the emitted term for the *untaken* arm is not
        absent — it is ``I·1{thresh>1e7}``, the same conditional form a clock
        threshold's in-branch derivative keeps — so reading the term list cannot
        tell a correct emission from an inverted one. Running it can.

        ``if(thresh>1e7, gamma, beta)`` takes the ``beta`` arm for the whole run
        (``thresh`` is 40). So ``∂x/∂beta`` is live and must match a finite
        difference, while ``∂x/∂gamma`` — through this rate law — and
        ``∂x/∂thresh`` are exactly zero: no perturbation short of 1e7 moves
        anything, which is what "the condition cannot cross" means.

        ``gamma`` is also the recovery rate of ``_R2``, so its column is not zero
        overall; ``I`` is what this compares, and the only route from ``gamma``
        to ``I`` that this rate law could open is the arm that never runs."""
        text = _with_law("if(thresh>1e7,gamma,beta)*I")
        model = _model(tmp_path, text)
        cols = ["beta", "gamma", "thresh"]
        run = bngsim.Simulator(model, method="ode", sensitivity_params=cols).run(
            t_span=(0.0, 6.0), n_points=13, rtol=1e-10, atol=1e-12
        )
        s = np.asarray(run.sensitivities)  # (n_t, n_species, n_param)
        assert np.all(np.isfinite(s))

        # `thresh` moves nothing at all: it enters f only through a condition
        # that is false throughout, and there is no crossing whose jump could
        # carry it.
        assert np.max(np.abs(s[:, :, cols.index("thresh")])) == 0.0

        # A finite difference of the model's own trajectory, for the arm that
        # does run and for the one that does not.
        def traj(name=None, value=None):
            m = _model(tmp_path, text, name="fd.net")
            if name is not None:
                m.set_param(name, value)
            r = bngsim.Simulator(m, method="ode").run(
                t_span=(0.0, 6.0), n_points=13, rtol=1e-10, atol=1e-12
            )
            return np.asarray(r.species)

        i_sp = list(model.species_names).index("person(state~I)")
        for name, h in (("beta", 1e-8), ("thresh", 1.0)):
            p0 = model.get_param(name)
            fd = (traj(name, p0 + h) - traj(name, p0 - h))[:, i_sp] / (2 * h)
            col = s[:, i_sp, cols.index(name)]
            scale = max(np.max(np.abs(fd)), np.max(np.abs(col)), 1e-30)
            assert np.max(np.abs(col - fd)) / scale < 1e-5, name

    def test_a_condition_over_a_functions_slot_is_not_a_run_constant(self, tmp_path):
        """The trap the run-constant test has to survive.

        A model *function* also names a parameter **slot** (issues #227, #266),
        and ``evaluate_functions()`` rewrites that slot from the function's own
        expression before every derivative evaluation — so its value moves with
        the trajectory even though its address is a parameter's. Reading
        ``param_names`` alone would call ``level>0.5`` a comparison between
        constants and admit a crossing that really does happen.

        Here ``level()`` is ``I/S0``, so the condition is a state condition in
        disguise. What must NOT happen is a silent admission on ground 3."""
        text = _with_law("if(level>0.5,beta,0)*I").replace(
            "    1 betaI() ", "    1 level() I/S0\n    2 betaI() "
        )
        core = _model(tmp_path, text)._core
        scope = sw.switch_condition_scope(core)
        assert "level" in scope.function_names
        assert "level" not in scope.run_constants
        assert not sw.condition_cannot_cross("level>0.5", scope)

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
        pairing it with an atom NOBODY compensates must still refuse, and
        ``(time()-5)*(time()-5)>=thresh`` is that atom: two clock terms, so it is
        not a bare power issue #418 can solve, not affine for issue #48, and over
        no live state for issue #150 to root on. The mixed clock/state pairing is
        admitted, and each half goes to its own machinery: ``t>=sigma`` to the
        issue #48 stop time, ``I>=thresh`` to the issue #150 crossing root."""
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

        _terms, reason = _decline(
            tmp_path,
            _with_law("if((t>=sigma)&&(time()*time()*time()+time()>=thresh),beta,0)*I"),
        )
        assert reason is not None and "'thresh'" in reason

    def test_an_equality_or_its_own_inequality_is_one_crossing(self, tmp_path):
        """Issue #381. MODEL2003190004 spells ``APC <= 0.2`` as an ``<or/>`` of
        ``<eq/>`` and ``<lt/>`` over one pair of operands, which the splitter
        hands over as two atoms. Judged apart, the equality half used to decline
        the whole model — and that decline is what put every column on CVODES'
        difference quotient, whose probe then stalled at the very crossing the
        ``<`` half had already earned a root for.

        Both atoms bound their true-sets with the same surface, so both resolve
        to the same residual and the detector registers ONE root: the condition
        is admitted, and the crossing behind the admission is the single one the
        ``<=`` spelling would have given."""
        core = _model(tmp_path, _with_law("if((I==thresh)||(I<thresh),beta,0)*I"))._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms
        assert sw.state_switch_conditions(core) == ["I<thresh"]
        assert (
            core.state_switch_residual("I==thresh")[0]
            == core.state_switch_residual("I<=thresh")[0]
        )

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
        assert records == [(3.0, 3, 3.0, [1.0, 1.0], [], [])]
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
        assert records == [(3.0, 3, 3.0, [1.0, 1.0, 1.0], [], [])]

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
            # An equality selects no branch over any INTERVAL, so it is admitted
            # with neither machinery compensating it — there is nothing to
            # compensate. Issue #381; the same shape as issue #382's
            # run-constant ground, arrived at from the other side.
            ("if(I==thresh,beta,0)*I", True, "no-crossing"),
            # Quadratic in the clock: issue #48's affine solver cannot invert it,
            # but issue #418 solves the single clock power to `time = sqrt(thresh)`
            # and #48 jumps that crossing like any clock threshold.
            ("if(time()*time()>=thresh,beta,0)*I", True, "clock"),
            # Quadratic with two clock terms: #418 declines it, but issue #421
            # writes both crossings down with the quadratic formula, so #48 jumps
            # each of them and the clock path claims it.
            ("if((time()-5)*(time()-5)>=thresh,beta,0)*I", True, "clock"),
            # Cubic in the clock: past what a closed form can be trusted for, so
            # it stays refused — the crossing nothing brackets.
            ("if(time()*time()*time()+time()>=thresh,beta,0)*I", False, None),
            # A repeating schedule: a crossing in every period rather than a
            # fixed number of them, enumerated from the period, the offset and
            # the duty since issue #436. `thresh` is the period (40) and `sigma`
            # the duty (3), so it turns over twice in every 40 time units.
            ("if(time()-thresh*floor(time()/thresh)>=sigma,beta,0)*I", True, "clock"),
            # A remainder OF a remainder — one level past what the schedule
            # recognizer reads, and the shape MODEL1708310001 writes. Refused.
            (
                "if(time()-thresh*floor(time()/thresh)"
                "-sigma*floor((time()-thresh*floor(time()/thresh))/sigma)>=1,beta,0)*I",
                False,
                None,
            ),
            # A clock compared against a SPECIES. It names no parameter, which
            # used to be the whole of the "nothing moves this crossing" test, so
            # the clock path claimed it and then registered nothing. Its
            # threshold does not evaluate to a number, so the crossing moves with
            # the trajectory and belongs to issue #150.
            ("if(time()<R,beta,0)*I", True, "state"),
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
        if by == "no-crossing":
            # The third way to be admissible, and the one that is NOT a
            # partition of the two detectors: nothing crosses, so nothing has to
            # compensate anything. Registering a crossing here is the error the
            # case guards against, not the requirement (see
            # test_equality_switch_surface.py).
            assert not records and not states, (
                f"{body!r} has no branch interval to compensate, so neither "
                "machinery may claim a crossing in it"
            )
            return
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
        assert sw.uncompensated_condition_reason("if(I==thresh,beta,0)*I", scope) is None
        # Issue #418: a single clock power is solved to sqrt(thresh) and admitted.
        assert (
            sw.uncompensated_condition_reason("if(time()*time()>=thresh,beta,0)*I", scope) is None
        )
        # Issue #421: a quadratic is solved at both of its crossings and admitted;
        # a cubic clock threshold stays refused.
        assert (
            sw.uncompensated_condition_reason("if((time()-5)*(time()-5)>=thresh,beta,0)*I", scope)
            is None
        )
        assert (
            sw.uncompensated_condition_reason(
                "if(time()*time()*time()+time()>=thresh,beta,0)*I", scope
            )
            is not None
        )

    def test_a_clock_threshold_over_a_species_is_not_a_fixed_crossing(self, tmp_path):
        """A crossing is fixed when its threshold evaluates to a number and no
        parameter moves it. Testing only the second half admits a threshold that
        is a *species*: ``time() < R`` names no parameter, so it was called fixed,
        the clock path claimed it on that ground, and the issue #48 detector then
        registered nothing because ``R`` is not a constant. The state root stands
        off whatever the clock path claims, so the crossing was compensated by
        neither and nothing warned.

        BIOMD0000000675 is the corpus case: three conditions
        (``time() >= START_S``, ``time() < END_M``, ``time() < END_M + 12``) over
        names that are not parameters, on a model the gate admitted."""
        core = _model(tmp_path, _with_law("if(time()<R,beta,0)*I"))._core
        scope = sw.switch_condition_scope(core)
        assert sw._clock_threshold_splits("time()<R", scope.clock_symbols) == ("time()", ["R"])
        assert not sw.fixed_clock_threshold("time()<R", scope)
        assert not sw.clock_crossing_compensated("time()<R", scope)
        # so issue #150 claims it, and the warning path calls it moving as well
        assert sw.state_switch_conditions(core) == ["time()<R"]
        assert sw.model_moving_crossings(core) == ("time()<R",)
        # a real literal threshold is still fixed, and still claimed by nobody
        assert sw.fixed_clock_threshold("time()<14", scope)

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

    def test_a_moving_crossing_is_refused_not_run_on_the_difference_quotient(self, tmp_path):
        """Issue #414. Declining the analytic sensitivity RHS over a crossing
        nothing compensates used to fall back to CVODES' difference quotient,
        which integrates smoothly through the moving crossing and returns a
        gradient wrong at and after it (issue #146). bngsim now refuses that run —
        a clean, typed non-run rather than a number it has already flagged as
        wrong — the same way it refuses an undifferentiable event trigger (GH
        #205) and a rate law it cannot differentiate at all (GH #214)."""
        model = _model(tmp_path, _with_law(UNCOMPENSATED))
        with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time moves"):
            _run_sens(model, ["beta", "gamma"], t_end=12.0)

    def test_the_refusal_is_typed_and_names_the_policy(self, tmp_path):
        """The refusal has to be actionable and distinguishable by type from the
        other things forward sensitivity raises: a ``SensitivityUnsupportedError``
        (still a ``ValueError``, for handlers that predate the typed class), which
        names the policy it enacts and explains the moving crossing. The exact
        condition it names is covered separately and backend-independently by
        ``test_the_scanner_names_the_uncompensated_condition`` — asserted off the
        detector rather than off a run, because the refusal string embeds
        whatever the run's model reports and this is an end-to-end run."""
        model = _model(tmp_path, _with_law(UNCOMPENSATED))
        with pytest.raises(bngsim.SensitivityUnsupportedError) as ei:
            _run_sens(model, ["beta"], t_end=12.0)
        message = str(ei.value)
        assert isinstance(ei.value, ValueError)
        assert "issue #414" in message
        assert "crossing time moves" in message

    def test_the_scanner_names_the_uncompensated_condition(self, tmp_path):
        """The detector the refusal reads names the offending atom — the same one
        the codegen decline is an ``UncompensatedCrossingReason`` for. Asserted on
        a fresh core (no Simulator, no JIT), so it pins the reason text
        deterministically and independently of the codegen backend."""
        from bngsim._switch_sensitivity import (
            UncompensatedCrossingReason,
            model_uncompensated_crossing_reason,
        )

        core = _model(tmp_path, _with_law(UNCOMPENSATED))._core
        reason = model_uncompensated_crossing_reason(core)
        assert isinstance(reason, UncompensatedCrossingReason)
        assert UNCOMPENSATED in reason
        assert "not inside an if() condition" in reason

    def test_a_compensated_state_crossing_is_not_refused(self, tmp_path):
        """The over-refusal guard on the issue #150 side. ``I>=thresh`` reads live
        state and its crossing moves, but the solver roots on it and applies the
        saltation jump — so the analytic RHS is admitted and its symbol is
        present, and the run must proceed rather than be swept up in the #414
        refusal."""
        sim = bngsim.Simulator(
            _model(tmp_path, _with_law("if(I>=thresh,beta,0)*I")),
            method="ode",
            sensitivity_params=["beta", "gamma"],
        )
        assert sim._codegen_provides_sens_rhs()
        sim._raise_if_uncompensated_crossing_sensitivities()  # must not raise
        run = sim.run(t_span=(0.0, 12.0), n_points=31, rtol=1e-11, atol=1e-11)
        assert np.asarray(run.sensitivities).shape[2] == 2

    def test_a_smooth_underivable_law_with_no_crossing_is_not_refused(self, tmp_path):
        """The over-refusal guard on the issue #56/#66 side, and the reason the
        gate keys on the crossing and not merely on 'the analytic RHS was
        declined'. ``erf(I)`` cannot be differentiated so the analytic sensitivity
        RHS is declined, but the law branches on nothing — the difference quotient
        is a correct, slower answer to the same smooth problem, so #414 must leave
        it be."""
        sim = bngsim.Simulator(
            _model(tmp_path, _with_law("erf(I)*beta*I")),
            method="ode",
            sensitivity_params=["beta", "gamma"],
        )
        assert not sim._codegen_provides_sens_rhs()  # erf declined the analytic RHS
        sim._raise_if_uncompensated_crossing_sensitivities()  # ... but no crossing, so no refusal

    def test_a_compensated_clock_crossing_is_not_refused(self, tmp_path):
        """The over-refusal guard on the issue #48 side. ``t>=sigma`` crosses at a
        time the switch-time detector knows a priori, so the analytic RHS is
        admitted and the run proceeds — the #414 gate must not fire on a crossing
        something already compensates."""
        sim = bngsim.Simulator(
            _model(tmp_path, SWITCHED),
            method="ode",
            sensitivity_params=["beta", "gamma", "sigma"],
        )
        assert sim._codegen_provides_sens_rhs()
        sim._raise_if_uncompensated_crossing_sensitivities()  # must not raise

    def test_compute_all_sensitivities_also_refuses_a_moving_crossing(self, tmp_path):
        """The refusal reaches the entry points that build codegen for themselves,
        not only run(). ``compute_all_sensitivities`` is routinely called on a
        Simulator constructed WITHOUT ``sensitivity_params``, so it gates on the
        artifact it prepares rather than on the constructor's."""
        sim = bngsim.Simulator(_model(tmp_path, _with_law(UNCOMPENSATED)), method="ode")
        with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time moves"):
            sim.compute_all_sensitivities(
                t_span=(0.0, 12.0), n_points=11, params=["beta", "gamma"]
            )

    def test_a_plain_run_of_the_same_model_is_unaffected(self, tmp_path):
        """The refusal is a forward-sensitivity policy, not a model verdict: the
        same model integrates as it always did when no sensitivities are asked
        for — the gate is behind the ``sensitivity_params or sensitivity_ic``
        guard, so a plain run never reaches it."""
        sim = bngsim.Simulator(_model(tmp_path, _with_law(UNCOMPENSATED)), method="ode")
        run = sim.run(t_span=(0.0, 12.0), n_points=11)
        assert np.asarray(run.species).shape[0] == 11


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
        assert sw._clock_threshold_splits("t>=sigma", clocks) == ("t", ["sigma"])
        assert sw._clock_threshold_split_bare("(t-sigma)>=0", clocks) is None
        clock, thresholds = sw._clock_threshold_splits("(t-sigma)>=0", clocks)
        assert clock == "t"
        assert [t.replace(" ", "") for t in thresholds] == ["sigma"]

    def test_an_affine_slope_resolves_to_the_crossing_in_closed_form(self):
        """The shape that blocks issue #326's stalling family. ``0 >= D0 −
        krep*(t − sigma)`` crosses at ``t = sigma + D0/krep``, which is a
        differentiable expression in three parameters — exactly what issue #48's
        jump needs and precisely what a lexical test cannot see."""
        clock, thresholds = sw._clock_threshold_splits(
            "0.0>=(D0-(krep*(t-sigma)))", frozenset({"t"})
        )
        assert clock == "t"
        import sympy as sp

        (threshold,) = thresholds
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
            assert sw._clock_threshold_splits(atom, clocks) == (bare[0], [bare[1]])

    def test_the_clock_on_both_sides_is_still_not_a_clock_threshold(self):
        """``t<2*t`` is affine and does solve — to ``t*=0`` — but the bare test
        rejects it deliberately and the state path claims it. Relaxing *which
        side* the clock may sit in must not relax *how many* sides it may sit
        on, or a crossing moves between two machineries for no gain."""
        assert sw._clock_threshold_splits("t<2*t", frozenset({"t"})) is None

    def test_a_non_linear_condition_is_declined(self):
        """``−b/a`` is the crossing only when the residual is degree 1 in the
        clock. Issue #418 solved one shape past that — a bare power ``t*t`` — and
        issue #421 the quadratic. A residual past both (``t*t*t + t``, cubic) still
        has no crossing any solver here can place, and has to keep declining
        rather than stop in the wrong place."""
        assert sw._clock_threshold_splits("(t*t*t+t-sigma)<0", frozenset({"t"})) is None
        # the bare power IS solved now — the boundary #418 moved
        assert sw._clock_threshold_splits("(t*t-sigma)<0", frozenset({"t"})) == (
            "t",
            ["sqrt(sigma)"],
        )

    def test_a_state_threshold_is_not_captured_by_the_solve(self):
        """The solve must not annex the issue #150 path. ``I-thresh<0`` reads no
        clock at all, so it is not a clock atom in either recogniser."""
        assert sw._clock_threshold_splits("(I-thresh)<0", frozenset({"t"})) is None

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


class TestANonAffineClockThresholdIsSolvedAndJumped:
    """Issue #418. ``_clock_affine_threshold`` (issue #355) solves a residual
    degree 1 in the clock; ``_clock_monomial_threshold`` solves the next shape up,
    a single power ``c·clock^n``. ``time()*time()>=thresh`` is the corpus case
    (issue #414 refused it): ``c·clock^n`` is strictly monotonic on ``clock ≥ 0``,
    so it has exactly one crossing there — ``time = (thresh/c)^(1/n)`` — and once
    that crossing-time expression is in hand the whole issue #48 machinery jumps it
    unchanged. A threshold that is not a bare clock power has no single crossing to
    name from the text alone and stays refused.
    """

    def test_the_recognizer_solves_the_single_power(self, tmp_path):
        core = _model(tmp_path, _with_law("if(time()*time()>=thresh,beta,0)*I"))._core
        scope = sw.switch_condition_scope(core)
        assert sw._clock_threshold_splits("time()*time()>=thresh", scope.clock_symbols) == (
            "time()",
            ["sqrt(thresh)"],
        )
        # cube and a scaled square, for the general (thresh/c)^(1/n) shape
        assert sw._clock_threshold_splits("time()*time()*time()>=thresh", scope.clock_symbols) == (
            "time()",
            ["thresh^(1/3)"],
        )

    def test_the_gate_admits_it_and_the_state_root_stands_off(self, tmp_path):
        core = _model(tmp_path, _with_law("if(time()*time()>=thresh,beta,0)*I"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None, "the monomial clock threshold must be admitted"
        # It reads no live state, so #150 must not also claim the crossing.
        assert sw.state_switch_conditions(core) == []

    def test_the_crossing_time_and_its_partial(self, tmp_path):
        """t* = sqrt(thresh) and ∂t*/∂thresh = 1/(2·sqrt(thresh)). With thresh=9
        the crossing is at 3.0 and the partial is 1/6."""
        text = _with_law("if(time()*time()>=thresh,beta,0)*I").replace(
            "7 thresh  40.0", "7 thresh  9.0"
        )
        core = _model(tmp_path, text)._core
        records, pinned = sw.compute_switch_time_sens(
            core, ["thresh", "beta"], 0.0, 12.0, has_analytic_sens_rhs=True
        )
        assert len(records) == 1
        t_star, _ci, _thr, dtstar = records[0][0], records[0][1], records[0][2], records[0][3]
        assert t_star == pytest.approx(3.0, abs=1e-9)
        assert dtstar[0] == pytest.approx(1.0 / 6.0, rel=1e-9)  # ∂t*/∂thresh
        assert dtstar[1] == 0.0  # beta does not move the crossing
        assert pinned == [list(core.param_names).index("thresh")]

    @pytest.mark.parametrize(
        "body",
        [
            "if(time()*time()*time()+time()>=thresh,beta,0)*I",  # cubic
            "if(time()*time()*time()*time()>=thresh*time(),beta,0)*I",  # quartic
        ],
    )
    def test_a_clock_threshold_past_the_quadratic_stays_refused(self, tmp_path, body):
        """The boundary after issue #421 moved it. ``(time()-5)^2`` and
        ``time()^2+time()`` used to live here and are solved now; degree 3 and up
        are not, and the reason is not that sympy declines to write their roots.
        A cubic with three real roots has none expressible in real radicals, so
        the closed forms route through complex intermediates that this would read
        as crossings that never happen — silently dropping real jumps."""
        core = _model(tmp_path, _with_law(body))._core
        scope = sw.switch_condition_scope(core)
        assert sw._clock_threshold_splits(_only_atom(body), scope.clock_symbols) is None
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert isinstance(reason, sw.UncompensatedCrossingReason)

    @requires_cc
    def test_the_sensitivity_matches_a_finite_difference_away_from_the_node(self, tmp_path):
        """The end-to-end oracle. ∂x/∂thresh from the analytic run must match a
        central difference of two plain trajectories — away from the crossing node
        at t*=3, where a central difference straddles the jump and converges to
        half the one-sided derivative bngsim reports (the same node behaviour issue
        #368 pins). thresh=9 puts the crossing at t=3 in a (0, 12) window."""

        def net(thresh):
            return _with_law("if(time()*time()>=thresh,beta,0)*I").replace(
                "7 thresh  40.0", f"7 thresh  {thresh}"
            )

        ts = (0.0, 12.0)
        n = 61
        times = np.linspace(*ts, n)
        analytic = bngsim.Simulator(
            _model(tmp_path, net(9.0)), method="ode", sensitivity_params=["thresh"]
        ).run(t_span=ts, n_points=n, rtol=1e-11, atol=1e-11)
        col = np.asarray(analytic.sensitivities)[:, :, 0]

        h = 9.0 * 1e-3

        def traj(thresh, name):
            return np.asarray(
                bngsim.Simulator(_model(tmp_path, net(thresh), name=name), method="ode")
                .run(t_span=ts, n_points=n, rtol=1e-12, atol=1e-14)
                .species
            )

        fd = (traj(9.0 + h, "hi.net") - traj(9.0 - h, "lo.net")) / (2.0 * h)

        # Away from the node the column is smooth in thresh, so the central
        # difference converges at O(h^2); the crossing itself is not vacuous.
        after = times >= 5.0
        assert float(np.max(np.abs(col[after]))) > 1.0
        scale = max(float(np.max(np.abs(col[after]))), 1e-300)
        np.testing.assert_allclose(col[after], fd[after], rtol=2e-4, atol=1e-4 * scale)

        # And at the node the central difference is ~half — a positive check that
        # the jump is really there rather than the column being trivially right.
        i_node = int(np.argmin(np.abs(times - 3.0)))
        node_a = float(np.max(np.abs(col[i_node])))
        node_fd = float(np.max(np.abs(fd[i_node])))
        assert node_fd == pytest.approx(0.5 * node_a, rel=0.1)


class TestAQuadraticClockThresholdIsSolvedAtBothCrossings:
    """Issue #421. Every recogniser before this one names at most ONE crossing:
    a residual of degree 1 has a single root, and ``c·clock^n`` is monotonic where
    the clock lives so it crosses once there. A quadratic is the first shape with
    genuinely two, and it is how the corpus writes a *window* —
    ``(time()-5)*(time()-5) >= thresh`` is true early, false through the middle
    and true again late.

    Both crossings are still closed form (the quadratic formula), so each one is
    an ordinary issue #48 record: evaluate the expression for ``t*``, differentiate
    it for ``∂t*/∂p``. Differentiating the root expression IS the implicit function
    theorem for this residual, so nothing here needs a numeric root find. What is
    new is that one atom now yields two thresholds, which is why the recogniser
    answers with a list.
    """

    def test_the_recognizer_answers_with_both_crossings(self, tmp_path):
        core = _model(tmp_path, _with_law("if((time()-5)*(time()-5)>=thresh,beta,0)*I"))._core
        scope = sw.switch_condition_scope(core)
        clock, thresholds = sw._clock_threshold_splits(
            "(time()-5)*(time()-5)>=thresh", scope.clock_symbols
        )
        assert clock == "time()"
        import sympy as sp

        got = [sp.sympify(t.replace("^", "**")) for t in thresholds]
        assert len(got) == 2
        for want in (sp.sympify("5 - sqrt(thresh)"), sp.sympify("5 + sqrt(thresh)")):
            assert any(sp.simplify(g - want) == 0 for g in got), f"{want} missing from {got}"

    def test_a_linear_term_no_longer_stops_the_solve(self, tmp_path):
        """``time^2 + time`` was the mixed-power shape issue #418 had to decline
        for want of a single crossing to name. The quadratic formula names both."""
        core = _model(tmp_path, _with_law("if(time()*time()+time()>=thresh,beta,0)*I"))._core
        scope = sw.switch_condition_scope(core)
        clock, thresholds = sw._clock_threshold_splits(
            "time()*time()+time()>=thresh", scope.clock_symbols
        )
        assert clock == "time()"
        assert len(thresholds) == 2

    def test_a_tangency_yields_one_repeated_root_not_two(self):
        """``(t-5)^2 >= 0`` touches its threshold instead of crossing it. The
        discriminant is a literal zero, so the solve answers with the single
        repeated root rather than two records for one instant."""
        assert sw._clock_threshold_splits("(t-5)*(t-5)>=0", frozenset({"t"})) == ("t", ["5"])

    def test_the_gate_admits_it_and_the_state_root_stands_off(self, tmp_path):
        core = _model(tmp_path, _with_law("if((time()-5)*(time()-5)>=thresh,beta,0)*I"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None, "the quadratic clock threshold must be admitted"
        # It reads no live state, so #150 must not also claim the crossings.
        assert sw.state_switch_conditions(core) == []

    def test_both_crossing_times_and_their_partials(self, tmp_path):
        """``(time()-5)^2 >= thresh`` at thresh=9 crosses at t=2 and t=8, and
        ``∂t*/∂thresh`` is ∓1/(2·sqrt(thresh)) = ∓1/6. Opposite signs: the branch
        turns off at the first crossing and back on at the second, and raising
        thresh widens the window from both ends."""
        text = _with_law("if((time()-5)*(time()-5)>=thresh,beta,0)*I").replace(
            "7 thresh  40.0", "7 thresh  9.0"
        )
        core = _model(tmp_path, text)._core
        records, pinned = sw.compute_switch_time_sens(
            core, ["thresh", "beta"], 0.0, 12.0, has_analytic_sens_rhs=True
        )
        assert [r.t_star for r in records] == pytest.approx([2.0, 8.0], abs=1e-9)
        assert [r.dtstar[0] for r in records] == pytest.approx([-1.0 / 6.0, 1.0 / 6.0], rel=1e-9)
        assert [r.dtstar[1] for r in records] == [0.0, 0.0]  # beta moves neither
        assert pinned == [list(core.param_names).index("thresh")]

    def test_a_crossing_outside_the_window_is_dropped_and_the_other_kept(self, tmp_path):
        """The two roots are independent records, so the window filter applies to
        each on its own. At the fixture's thresh=40 the roots are 5∓sqrt(40), and
        only the later one is inside (0, 12]."""
        core = _model(tmp_path, _with_law("if((time()-5)*(time()-5)>=thresh,beta,0)*I"))._core
        records, _pinned = sw.compute_switch_time_sens(
            core, ["thresh"], 0.0, 12.0, has_analytic_sens_rhs=True
        )
        assert len(records) == 1
        assert records[0].t_star == pytest.approx(5.0 + 40.0**0.5, abs=1e-9)

    @pytest.mark.parametrize(
        "body",
        [
            "if((time()-5)*(time()-5)>=thresh,beta,0)*I",
            "if(time()*time()>=thresh,beta,0)*I",
        ],
    )
    def test_a_threshold_with_no_real_root_is_admitted_rather_than_refused(self, tmp_path, body):
        """A crossing time that comes out non-real is not an unreadable threshold,
        it is the statement that the condition never crosses: at thresh=-4 both
        ``(time()-5)^2 >= thresh`` and ``time()^2 >= thresh`` are true for the whole
        run, the branch never flips, and ``∂f/∂thresh`` is a correct clean zero.
        Reading that as "does not reduce to a constant" refuses a model bngsim can
        answer, and the quadratic formula puts a whole region of parameter space
        there, since the discriminant goes negative as soon as thresh does.

        The single-power case is the same defect one recogniser earlier, and was
        refused before this issue."""
        text = _with_law(body).replace("7 thresh  40.0", "7 thresh  -4.0")
        core = _model(tmp_path, text)._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        assert sw.compute_switch_time_sens(
            core, ["thresh"], 0.0, 12.0, has_analytic_sens_rhs=True
        ) == ([], [])

    def test_a_leading_coefficient_that_is_zero_at_run_time_is_refused(self, tmp_path):
        """``a*time()^2 + time() >= thresh`` is a quadratic whose roots divide by
        ``a``, and a fit can set ``a`` to 0, where the condition is really the
        affine ``time() >= thresh``. Both roots then evaluate to a degenerate
        1/0 rather than to a number, and bngsim refuses instead of stopping in the
        wrong place. That is the safe direction: a refusal, not a gradient missing
        a jump."""
        core = _model(
            tmp_path,
            _with_params(
                "    8 a  0.0  # Constant\n", "if(a*time()*time()+time()>=thresh,beta,0)*I"
            ),
        )._core
        scope = sw.switch_condition_scope(core)
        atom = "a*time()*time()+time()>=thresh"
        assert len(sw._clock_threshold_splits(atom, scope.clock_symbols)[1]) == 2
        assert not sw.clock_crossing_compensated(atom, scope)
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert isinstance(reason, sw.UncompensatedCrossingReason)
        assert sw.compute_switch_time_sens(
            core, ["thresh", "a"], 0.0, 12.0, has_analytic_sens_rhs=True
        ) == ([], [])

    def test_an_atom_compensates_all_its_crossings_or_none(self, tmp_path):
        """``(time()-5)*(time()-thresh) >= 0`` has a fixed edge at t=5 and a fitted
        one at t=thresh, but the quadratic formula writes the fixed edge with
        ``thresh`` still inside it, and its partial cancels to exactly 0 — which
        the chain rule cannot tell apart from a chain rule it lost (issue #56). So
        one crossing resolves and the other does not, and the atom is refused
        rather than half compensated. Jumping the edge bngsim can place while the
        other flips the branch unjumped is a silent-zero half answer, and it would
        also split the gate from the detector, which is what issue #68 exists to
        prevent. Both must say no."""
        core = _model(tmp_path, _with_law("if((time()-5)*(time()-thresh)>=0,beta,0)*I"))._core
        scope = sw.switch_condition_scope(core)
        _clock, thresholds = sw._clock_threshold_splits(
            "(time()-5)*(time()-thresh)>=0", scope.clock_symbols
        )
        assert len(thresholds) == 2
        assert [sw._threshold_compensated(t, scope) for t in thresholds] == [False, True]
        assert not sw.clock_crossing_compensated("(time()-5)*(time()-thresh)>=0", scope)
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert isinstance(reason, sw.UncompensatedCrossingReason)
        assert sw.compute_switch_time_sens(
            core, ["thresh"], 0.0, 100.0, has_analytic_sens_rhs=True
        ) == ([], [])

    def test_an_imaginary_unit_is_not_mistaken_for_an_absent_crossing(self, tmp_path):
        """The trap the guard above has to survive. sympy reads a bare ``I`` as the
        imaginary unit, and an SIR model spells its infected observable exactly
        that — so ``t >= sigma*I`` evaluates to a non-real number without being a
        crossing that fails to happen. It is a threshold over live state, it
        belongs to issue #150, and it has to keep reaching it."""
        core = _model(tmp_path, _with_law("if(t>=sigma*I,beta,0)*I"))._core
        scope = sw.switch_condition_scope(core)
        assert not sw._threshold_is_non_real(
            "sigma*I", scope.param_idx, scope.values, scope.derived_exprs
        )
        assert not sw.clock_crossing_compensated("t>=sigma*I", scope)
        assert sw.state_switch_conditions(core) == ["t>=sigma*I"]

    @requires_cc
    def test_the_counter_clock_spelling_reaches_the_same_answer(self, tmp_path):
        """A BNGL counter clock is a *species*, so ``(t-5)*(t-5)>=thresh`` reads
        live state and issue #150 rooted on it before this change; now the clock
        path recognises it and claims it first, as it already did for the affine
        spellings. Two different machineries, so this is worth stating as a
        number: the counter spelling and the ``time()`` spelling must produce the
        same tensor, and exactly one of the two detectors may claim each."""
        params = ["thresh"]
        counter = _model(
            tmp_path,
            _with_law("if((t-5)*(t-5)>=thresh,beta,0)*I").replace(
                "7 thresh  40.0", "7 thresh  9.0"
            ),
            name="counter.net",
        )
        literal = _model(
            tmp_path,
            _with_law("if((time()-5)*(time()-5)>=thresh,beta,0)*I").replace(
                "7 thresh  40.0", "7 thresh  9.0"
            ),
            name="literal.net",
        )
        for model in (counter, literal):
            records, _pinned = sw.compute_switch_time_sens(
                model._core, params, 0.0, 12.0, has_analytic_sens_rhs=True
            )
            assert len(records) == 2
            assert sw.state_switch_conditions(model._core) == []

        a = np.asarray(_run_sens(counter, params, t_end=12.0).sensitivities)
        b = np.asarray(_run_sens(literal, params, t_end=12.0).sensitivities)
        scale = max(float(np.max(np.abs(a))), 1e-300)
        assert scale > 1.0  # not a comparison of two zero tensors
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9 * scale)

    @requires_cc
    def test_the_sensitivity_matches_a_finite_difference_at_both_crossings(self, tmp_path):
        """The end-to-end oracle, and the one assertion that would catch a run
        compensating only the *second* crossing. ``∂x/∂thresh`` is exactly 0 before
        t=2, so the whole plateau between the two crossings is the first jump and
        nothing else — a run that missed it would read 0 there and still look fine
        after t=8.

        Away from the two nodes a central difference of two plain trajectories
        converges to the analytic column; at a node it straddles the jump and
        converges to half of it, which is the switch-node behaviour issue #368
        pins."""

        def net(thresh):
            return _with_law("if((time()-5)*(time()-5)>=thresh,beta,0)*I").replace(
                "7 thresh  40.0", f"7 thresh  {thresh}"
            )

        ts = (0.0, 12.0)
        n = 121
        times = np.linspace(*ts, n)
        analytic = bngsim.Simulator(
            _model(tmp_path, net(9.0)), method="ode", sensitivity_params=["thresh"]
        ).run(t_span=ts, n_points=n, rtol=1e-11, atol=1e-11)
        col = np.asarray(analytic.sensitivities)[:, :, 0]

        h = 9.0 * 1e-3

        def traj(thresh, name):
            return np.asarray(
                bngsim.Simulator(_model(tmp_path, net(thresh), name=name), method="ode")
                .run(t_span=ts, n_points=n, rtol=1e-12, atol=1e-14)
                .species
            )

        fd = (traj(9.0 + h, "hi.net") - traj(9.0 - h, "lo.net")) / (2.0 * h)

        # The first crossing on its own: nothing before it, a plateau after it
        # that persists until the second crossing at t=8.
        before = times < 1.9
        plateau = (times > 2.1) & (times < 7.9)
        assert float(np.max(np.abs(col[before]))) == 0.0
        assert float(np.min(np.max(np.abs(col[plateau]), axis=1))) > 1.0

        # Away from both nodes the column is smooth in thresh, so the central
        # difference converges at O(h^2).
        away = (np.abs(times - 2.0) > 0.05) & (np.abs(times - 8.0) > 0.05)
        scale = max(float(np.max(np.abs(col[away]))), 1e-300)
        np.testing.assert_allclose(col[away], fd[away], rtol=2e-4, atol=1e-4 * scale)

        # And at the first node the central difference is half the analytic value
        # — a positive check that the jump is really applied there.
        i_node = int(np.argmin(np.abs(times - 2.0)))
        assert float(np.max(np.abs(fd[i_node]))) == pytest.approx(
            0.5 * float(np.max(np.abs(col[i_node]))), rel=0.1
        )


# A crossing NOTHING compensates, post-#150, post-#381, post-#418 and post-#421: the
# boolean-as-a-number idiom, a step at I=1 with no `if()` head for any threshold
# to be located in. Neither machinery brackets it, which is exactly what makes it
# the right fixture for "the difference quotient is not a correct fallback either".
# (`if(time()*time()>=thresh,...)` used to sit here too; issue #418 compensates it,
# so it moved to TestANonAffineClockThresholdIsSolvedAndJumped above, which also
# RUNS it — the MIR SIGABRT that once kept it out of run tests was #413's
# self-multiply overflow, fixed in #415.)
# ─── a repeating schedule (issue #436) ─────────────────────────────────────
#
# A separate fixture from ``SWITCHED`` because what a dose schedule does is
# accumulate: ``A`` fills while the schedule is on and drains all the time, so
# every edge in the window leaves a mark on the trajectory and a finite
# difference of it has something to compare against. The SIR fixture cannot serve
# — its epidemic burns out inside the first period, which makes ∂x/∂P five orders
# of magnitude smaller than ∂x/∂d and puts it under the difference's noise.

SCHEDULED = """\
begin parameters
    1 P       24.0  # Constant
    2 d        7.0  # Constant
    3 kin      0.1  # Constant
    4 kout     0.05  # Constant
    5 kclock   1  # Constant
end parameters
begin functions
    1 dose() if(time()-P*floor(time()/P)>=d,kin,0)
end functions
begin species
    1 A() 0
    2 counter() 0
end species
begin reactions
    1 0 1 dose #_R1
    2 1 0 kout #_R2
    3 0 2 kclock #_R3
end reactions
begin groups
    1 A                    1
    2 t                    2
end groups
"""

_SCHEDULED_LAW = "    1 dose() if(time()-P*floor(time()/P)>=d,kin,0)\n"


def _with_dose(law: str) -> str:
    """``SCHEDULED`` with its rate law swapped for *law*."""
    return SCHEDULED.replace(_SCHEDULED_LAW, f"    1 dose() {law}\n")


class TestAPeriodicClockScheduleIsEnumeratedAndJumped:
    """Issue #436. Every recogniser before this one names a *fixed* number of
    crossings, because its residual is a polynomial in the clock and a polynomial
    has finitely many roots. A schedule has one in every period for as long as the
    run lasts::

        if(time() - 24*floor(time()/24) >= 7, on, off)

    which is "on for the last 17 hours of every 24 hour day" — repeated dosing, a
    light and dark cycle, a train of stimulus pulses. Nineteen of the twenty-two
    corpus models with a rate-law crossing nothing compensated write this one
    shape, several of them spelled differently.

    The edges are still closed form, so nothing here roots on anything: for a
    period ``P``, an offset ``s`` and a duty ``d`` they fall at ``s + k*P + d``
    and ``s + (k+1)*P`` for every whole ``k`` in the window, and each is an
    ordinary issue #48 record — a value to stop at and an expression to
    differentiate for ``∂t*/∂p``. What is new is that the *number* of records
    depends on the run window, which is why the recogniser answers with the
    schedule rather than with a list of crossing times, and why there is a budget.
    """

    def test_the_recognizer_reads_the_schedule_not_the_spelling(self):
        """Issue #355's lesson, applied to a schedule. These five atoms are the
        same 24-hour cycle written the five ways the corpus writes it: the
        threshold on either side, a start time folded in, a remainder taken in
        seconds and divided back to hours (BIOMD0000000238), and ``ceil`` instead
        of ``floor``. The recogniser has to answer with the same schedule for all
        of them, because a fitted gradient that depends on which way the modeller
        typed the condition is not a gradient anyone can use."""
        import sympy as sp

        clocks = frozenset({"time", "time()"})
        for atom, period, offset, duty in [
            ("time()-24*floor(time()/24)>=7", "24", "0", "7"),
            ("7<=time()-24*floor(time()/24)", "24", "0", "7"),
            ("(time()-start)-24*floor((time()-start)/24)>=7", "24", "start", "7"),
            (
                "((time()*3600)-(floor((time()*3600)/day_length))*day_length)/3600>=7",
                "day_length/3600",
                "0",
                "7",
            ),
            ("time()-24*ceil(time()/24)>=-17", "-24", "0", "-17"),
        ]:
            sched = sw._clock_periodic_schedule(atom, clocks)
            assert sched is not None, f"{atom!r} was not read as a schedule"
            for got, want in zip(sched[1:], (period, offset, duty), strict=True):
                assert sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0, (
                    f"{atom!r}: got {sched}, wanted {(period, offset, duty)}"
                )

    def test_a_remainder_of_a_remainder_is_declined(self):
        """One step function, not two. A schedule of schedules — the shape
        MODEL1708310001 writes for "one dose a day for the first fourteen days of
        every twenty-one day cycle" — is enumerable in principle and is not
        enumerated here, so it stays refused rather than being read as the outer
        schedule alone."""
        clocks = frozenset({"time", "time()"})
        inner = "time()-24*floor(time()/24)"
        assert sw._clock_periodic_schedule(f"{inner}-6*floor(({inner})/6)>=2", clocks) is None

    def test_a_residual_that_does_not_repeat_is_declined(self):
        """What makes the three-number description true is that the residual reads
        the same in every period. ``rem(t, P) >= t/2`` has enumerable crossings
        too, but they are not one period apart and no (period, offset, duty)
        describes them, so the recogniser declines rather than answer with a
        schedule the condition does not follow."""
        clocks = frozenset({"time", "time()"})
        assert sw._clock_periodic_schedule("time()-24*floor(time()/24)>=time()/2", clocks) is None

    def test_the_gate_admits_it_and_neither_other_machinery_claims_it(self, tmp_path):
        core = _model(tmp_path, SCHEDULED)._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None, f"the schedule must be admitted, got: {reason}"
        assert terms
        # It reads no live state, so issue #150 must not also root on it — a
        # crossing claimed twice is jumped twice.
        assert sw.state_switch_conditions(core) == []

    def test_the_detector_places_both_edges_of_every_period(self, tmp_path):
        """The whole answer, in one assertion. Over ``(0, 100]`` a 24-hour cycle
        with a 7-hour duty turns on at 7, 31, 55 and 79 and off at 24, 48, 72 and
        96, and the partials say which parameter moves which edge: the duty moves
        only the on-edges and by one, while the period moves the k-th edge of
        either family by k — which is the whole reason a fitted period is worth
        compensating rather than approximating."""
        core = _model(tmp_path, SCHEDULED)._core
        records, pinned = sw.compute_switch_time_sens(
            core, ["P", "d"], 0.0, 100.0, has_analytic_sens_rhs=True
        )
        assert [r.t_star for r in records] == pytest.approx(
            [7.0, 24.0, 31.0, 48.0, 55.0, 72.0, 79.0, 96.0]
        )
        assert [r.dtstar[0] for r in records] == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
        assert [r.dtstar[1] for r in records] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert pinned == sorted(list(core.param_names).index(n) for n in ("P", "d"))

    def test_a_counter_clock_schedule_is_offset_by_the_counter(self, tmp_path):
        """The BNGL spelling. ``t`` is a species integrated at rate 1, so the
        schedule is written over a clock whose value is not the simulation time
        and whose edges have to be shifted back onto it."""
        core = _model(tmp_path, _with_dose("if(t-P*floor(t/P)>=d,kin,0)"))._core
        records, _pinned = sw.compute_switch_time_sens(
            core, ["P", "d"], 0.0, 50.0, has_analytic_sens_rhs=True
        )
        assert [r.t_star for r in records] == pytest.approx([7.0, 24.0, 31.0, 48.0])
        assert records[0].clock_idx0 == list(core.species_names).index("counter()")
        # A counter clock IS a species, so the issue #150 state root would happily
        # claim this condition as well. Exactly one machinery may have it, or the
        # jump is applied twice.
        assert sw.state_switch_conditions(core) == []

    def test_a_fixed_schedule_is_admitted_and_needs_no_records(self, tmp_path):
        """Six of the nineteen corpus models write their light and dark cycle with
        literals. Nothing moves any of those edges, so ``∂t*/∂p`` is exactly 0 and
        there is no jump to make — but the in-branch derivative is still the whole
        gradient, and refusing the condition took the analytic sensitivity RHS away
        from the whole model. The gate has to admit it with no record behind it,
        which is exactly what ``t<14`` has always done for a single crossing."""
        core = _model(tmp_path, _with_dose("if(time()-24*floor(time()/24)>=7,kin,0)"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        records, pinned = sw.compute_switch_time_sens(
            core, ["kin", "kout"], 0.0, 100.0, has_analytic_sens_rhs=True
        )
        assert records == [] and pinned == []
        # ... and it is not reported as a crossing the difference quotient would
        # miss either, because there is nothing there to miss (issue #232).
        assert sw.model_moving_crossings(core) == ()

    def test_a_schedule_that_never_crosses_registers_nothing(self, tmp_path):
        """``rem(time(), P) >= 0`` is true at every instant of the run: a
        remainder is never negative. MODEL0406793751 really does write this, and
        reading it as a schedule with an edge at the top of every period would put
        a stop time where the branch does not change. The duty has to fall
        strictly inside the period for there to be a crossing at all."""
        core = _model(tmp_path, _with_dose("if(time()-P*floor(time()/P)>=0,kin,0)"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        records, _pinned = sw.compute_switch_time_sens(
            core, ["P", "d"], 0.0, 100.0, has_analytic_sens_rhs=True
        )
        assert records == []

    def test_the_schedule_is_checked_against_the_condition_the_model_evaluates(self, tmp_path):
        """The recogniser reads the residual through sympy's parser, which binds
        ``I`` to the imaginary unit — so a model parameter spelled ``I`` is not the
        symbol the recogniser thinks it is. Most of the time that costs nothing,
        because the imaginary unit multiplies and divides like any other symbol and
        the answer comes back spelled with the same name. ``I*I`` is where it stops
        being harmless: sympy folds it to ``-1``, the schedule reads as one whose
        duty falls outside its period, and a condition that crosses eight times in
        the window would be admitted with no record behind it. Four evaluations of
        the model's own residual are what stop that."""
        text = _with_dose("if(time()-I*I*floor(time()/(I*I))>=d,kin,0)").replace(
            "    1 P       24.0", "    1 I       24.0"
        )
        core = _model(tmp_path, text)._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is not None
        assert "does not follow that schedule" in reason

    def test_a_parameter_named_I_is_not_refused_on_suspicion(self, tmp_path):
        """The companion that keeps the check above from being a blanket refusal.
        ``I`` alone survives the imaginary unit's arithmetic — ``1/I`` is ``-I``,
        and the period comes back spelled ``I`` again — so this schedule is read
        correctly and is compensated, which the residual check confirms rather than
        assumes."""
        text = _with_dose("if(time()-I*floor(time()/I)>=d,kin,0)").replace(
            "    1 P       24.0", "    1 I       24.0"
        )
        core = _model(tmp_path, text)._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        records, _pinned = sw.compute_switch_time_sens(
            core, ["I", "d"], 0.0, 50.0, has_analytic_sens_rhs=True
        )
        assert [r.t_star for r in records] == pytest.approx([7.0, 24.0, 31.0, 48.0])

    @staticmethod
    def _crowded(period: float = 0.01, duty: float = 0.005) -> str:
        """``SCHEDULED`` with a period short enough to overrun the edge budget.

        Asserted against the budget rather than hard-coded against a number, so
        raising the budget cannot quietly turn the two tests below into tests of
        nothing."""
        edges = 2 * 100.0 / period
        assert edges > sw._SCHEDULE_EDGE_BUDGET, (
            f"{edges:.0f} edges no longer overruns a budget of {sw._SCHEDULE_EDGE_BUDGET}"
        )
        return SCHEDULED.replace("    1 P       24.0", f"    1 P       {period}").replace(
            "    2 d        7.0", f"    2 d        {duty}"
        )

    def test_the_budget_refuses_a_window_it_cannot_enumerate(self, tmp_path):
        """A hundred days of hourly dosing is thousands of stop times, and the
        count is a property of the run rather than of the model, so it is the one
        thing here the value-free gate cannot decide. Compensating the edges that
        fit inside a budget and not the ones after it would give a gradient right
        at the start of the run and silently wrong at the end, so the run is
        refused instead — loudly, and naming what to change."""
        text = self._crowded()
        core = _model(tmp_path, text)._core
        with pytest.raises(bngsim.SensitivityUnsupportedError) as ei:
            sw.compute_switch_time_sens(core, ["P", "d"], 0.0, 100.0, has_analytic_sens_rhs=True)
        assert "repeating schedule" in str(ei.value)
        assert "'P'" in str(ei.value)

    def test_the_budget_is_not_charged_for_a_schedule_nobody_asked_about(self, tmp_path):
        """The same model, with the schedule's own parameters left out of the
        request. Every edge then carries ``∂t*/∂p = 0`` for every column asked
        for, so none of them would emit a record however many there are, and
        refusing the run over a budget for records nobody wanted would be a
        refusal with nothing behind it."""
        core = _model(tmp_path, self._crowded())._core
        records, pinned = sw.compute_switch_time_sens(
            core, ["kin", "kout"], 0.0, 100.0, has_analytic_sens_rhs=True
        )
        assert records == [] and pinned == []

    def test_only_the_edges_inside_the_window_are_recorded(self, tmp_path):
        """The window is half-open at the start and closed at the end, the same
        filter every other crossing in this module is judged by: an edge at
        ``t_start`` would precede the run's first recorded column, and one at
        ``t_end`` still jumps into its last."""
        core = _model(tmp_path, SCHEDULED)._core
        records, _pinned = sw.compute_switch_time_sens(
            core, ["P", "d"], 7.0, 31.0, has_analytic_sens_rhs=True
        )
        assert [r.t_star for r in records] == pytest.approx([24.0, 31.0])

    def test_a_floor_inside_a_condition_no_longer_declines_the_rate_law(self, tmp_path):
        """The second half of the refusal this issue lifts, and the one that has
        nothing to do with crossings. ``floor()`` is rejected wherever it is
        differentiated, and the pre-scan that rejects it did not care where in the
        rate law it sat. Inside an ``if()`` condition it is never differentiated —
        sympy copies a Piecewise's conditions through untouched — so what used to
        stop the model before the crossing gate ever ran is waived there, and
        nowhere else."""
        from bngsim._jacobian import unsupported_expr_construct

        law = "if(time()-P*floor(time()/P)>=d,kin,0)"
        assert unsupported_expr_construct(law) == "if() conditional"
        assert unsupported_expr_construct(law, allow_conditions=True) is None
        # In a branch, or outside an if() altogether, it really is differentiated.
        assert (
            unsupported_expr_construct(
                "if(time()-P*floor(time()/P)>=d,kin*floor(time()/P),0)", allow_conditions=True
            )
            == "floor()"
        )
        assert (
            unsupported_expr_construct("kin*floor(time()/P)", allow_conditions=True) == "floor()"
        )

    def test_a_floor_in_a_branch_still_declines_the_model(self, tmp_path):
        """The same boundary, asserted through the real derivation path rather
        than through the pre-scan alone."""
        core = _model(
            tmp_path, _with_dose("if(time()-P*floor(time()/P)>=d,kin*floor(time()),0)")
        )._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is not None
        assert "floor()" in reason

    @pytest.mark.parametrize(
        "threshold", ["floor(P)", "ceil(P)", "sign(P)", "floor(P)+1", "sign(P)*P"]
    )
    def test_a_step_function_in_a_threshold_declines_instead_of_crashing(
        self, tmp_path, threshold
    ):
        """A crossing time that steps as a parameter moves, rather than moving
        with it. There is no chain rule to the primaries for one, and asking sympy
        for it does not fail cleanly: ``d/dP floor(P)`` comes back as an
        unevaluated ``Derivative``, and evaluating that recurses until Python
        raises ``RecursionError`` — which is not an exception any caller here
        handles, so the whole codegen pass died with a stack overflow instead of
        declining the model.

        The ``sign`` spellings reach this on bngsim today, with no schedule and no
        ``floor`` anywhere: ``sign`` is not one of the constructs the pre-scan
        rejects, so ``if(time() >= sign(P), …)`` crashed. The ``floor`` and
        ``ceil`` spellings became reachable with this issue, which waives a
        ``floor()`` inside an ``if()`` condition. Both now decline."""
        core = _model(tmp_path, _with_dose(f"if(time()>={threshold},kin,0)"))._core
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is not None
        records, _pinned = sw.compute_switch_time_sens(
            core, ["P", "d"], 0.0, 100.0, has_analytic_sens_rhs=True
        )
        assert records == []


@requires_cc
class TestAPeriodicScheduleAgainstAFiniteDifference:
    """The oracle for issue #436, and the only one that separates a schedule
    compensated from a schedule merely admitted.

    ``A`` fills at ``kin`` while the schedule is on and drains at ``kout`` all the
    time, so its value at any instant carries every edge that has passed. Two
    trajectories at ``p ± h`` therefore differ by the accumulated effect of every
    jump, which is what the analytic column has to reproduce.

    Sample times are deliberately off the edges. A central difference taken at an
    instant that IS a crossing returns exactly half the analytic value — the two
    one-sided derivatives averaged — which is a property of the reference, not a
    defect in the column (issue #368)."""

    @staticmethod
    def _run(tmp_path, name, overrides, sens=None):
        import numpy as np

        model = _model(tmp_path, SCHEDULED, name=name)
        for key, value in overrides.items():
            model.set_param(key, value)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=sens)
        run = sim.run(
            sample_times=list(np.arange(0.0, 240.001, 4.0) + 0.37), rtol=1e-11, atol=1e-13
        )
        return run

    @pytest.mark.parametrize(
        ("name", "nominal", "step"),
        [("P", 24.0, 2.4e-4), ("d", 7.0, 7e-5), ("kin", 0.1, 1e-6), ("kout", 0.05, 5e-7)],
    )
    def test_every_column_matches_a_central_difference(self, tmp_path, name, nominal, step):
        params = ["P", "d", "kin", "kout"]
        analytic = np.asarray(self._run(tmp_path, "an.net", {}, sens=params).sensitivities)[
            :, :, params.index(name)
        ]
        up = np.asarray(self._run(tmp_path, "up.net", {name: nominal + step}).species)
        down = np.asarray(self._run(tmp_path, "dn.net", {name: nominal - step}).species)
        fd = (up - down) / (2.0 * step)
        scale = float(np.max(np.abs(fd)))
        assert scale > 1e-3, f"the {name} column is too small for the difference to test"
        np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-5 * scale)

    def test_the_period_column_is_not_the_duty_column_in_disguise(self, tmp_path):
        """Ten periods fit in this window, so the last edge moves ten times as far
        with the period as the first does. A column that had dropped the ``k``
        factor — treating every edge as if it were the first — would still look
        plausible and would still be wrong, so the two shapes are separated here
        rather than left to the tolerance above."""
        params = ["P", "d"]
        analytic = np.asarray(self._run(tmp_path, "an2.net", {}, sens=params).sensitivities)
        period_col = analytic[-1, 0, 0]
        duty_col = analytic[-1, 0, 1]
        assert abs(period_col) > 2.0 * abs(duty_col)


UNCOMPENSATED = "beta*(I>1)*I"


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
