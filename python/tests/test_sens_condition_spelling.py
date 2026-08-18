"""Issue #232 — a boolean connective in a *condition* must not decline the model.

``_functional_rate_law_partials`` ends its pre-scan by looking for call heads it
does not recognize, so a rate law that reaches out to a table function or an
un-inlined SBML helper declines with a message naming the call. The scan is
lexical: it matches ``identifier(``. In ``(X<hi) and (X>lo)`` the ``and`` is an
*infix operator* whose right operand merely happens to be parenthesized, and the
scan read it as a call to an unknown function ``and`` — so the whole model's
analytic sensitivity RHS was declined over a piece of syntax.

That is invisible on the ``.net`` side, where BNGL writes ``&&`` (not an
identifier, so the scan never sees it), and universal on the SBML side, where
``_ast_to_exprtk`` renders every ``<and/>`` as the word.

**Why it is not merely slow.** The decline message said the difference-quotient
fallback is "correct, but slower". At a state switch it is neither. CVODES'
internal DQ integrates the variational equation smoothly through the crossing,
dropping the saltation jump the issue #150 machinery exists to apply, and its
probe evaluates ``f`` at ``y + σ·s`` — which just past the surface lands on the
other branch. Forcing the *nested* spelling onto the same fallback reproduces the
``and`` spelling's number to every digit (``-6.197678503e-01`` against a closed
form of ``-1.3120451477e+00``, 273 steps against 179) and its failure to
integrate below ``rtol=1e-9``. So the second half of the fix is that the warning
stops promising a correct fallback whenever the declined model carries a crossing
whose time moves.

What this locks:

  1. one window, four SBML spellings (``and``, nested ``piecewise``, ``or`` over
     the complement, and ``not(... and ...)``), one gradient — matching the
     closed form, and each other, at every tolerance. That invariant is what
     removes the constant. The fourth arm is issue #234: ``_split_logical_atoms``
     dropped a leading ``!`` and split what was under it, but kept a ``not(...)``
     call whole, so the only negation spelling SBML can produce was the only one
     whose surfaces never reached the crossing machinery;
  2. the admitted head set is exactly the heads the ExprTk→sympy preprocessor
     rewrites into a construct, asserted by running them through it rather than
     by re-listing them, and the excluded ones really are untranslatable;
  3. the heads are admitted only behind the same gate ``if`` is — a model whose
     conditions were NOT cleared still reports them as unsupported;
  4. the decline warning's claim: honest when the model has a moving crossing,
     unchanged when it does not, and never downgrading the issue #146 class.
"""

from __future__ import annotations

import math
import textwrap

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


# ─── one window, four spellings ────────────────────────────────────────────
#
# X' = -k(X)·X, with k = K_BOOST while LO < X < HI and K_BASE outside it. Wide
# enough that the trajectory is easy for every arm (issue #194 owns the narrow
# window, where the question is whether the window is entered at all); the only
# thing that moves between the arms here is how the one condition is written.

X0, K_BASE, K_BOOST, T_END = 10.0, 0.2, 0.5, 6.0
LO, HI = 3.0, 8.0


def _closed_form() -> tuple[float, float]:
    """``(X(T_END), dX(T_END)/dK_BOOST)``.

    Three exact segments: decay at ``K_BASE`` down to ``HI``, at ``K_BOOST``
    down to ``LO``, then at ``K_BASE`` again. Only the *duration* of the middle
    segment depends on ``K_BOOST``, so the whole gradient is the shift it puts on
    the exit time ``t2``:  ``dX/dK_BOOST = X(T_END)·K_BASE·dt2/dK_BOOST``.
    """
    t1 = math.log(X0 / HI) / K_BASE
    t2 = t1 + math.log(HI / LO) / K_BOOST
    x_end = LO * math.exp(-K_BASE * (T_END - t2))
    dt2_dk = -math.log(HI / LO) / K_BOOST**2
    return x_end, x_end * K_BASE * dt2_dk


_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="state_window">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_base" value="0.2" constant="true"/>
      <parameter id="k_boost" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
{CONDITION}
            <ci>X</ci>
          </apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# The `and` spelling: one compound condition. This is the arm the issue is about.
_CONJUNCTIVE = """\
<piecewise>
  <piece><ci>k_boost</ci>
    <apply><and/>
      <apply><lt/><ci>X</ci><cn>8</cn></apply>
      <apply><gt/><ci>X</ci><cn>3</cn></apply>
    </apply>
  </piece>
  <otherwise><ci>k_base</ci></otherwise>
</piecewise>"""

# The same window with no connective at all: one comparison per `piecewise`.
# This arm was already correct, which is what makes it the control — it is the
# existence proof that the derivative the other two need is representable.
_NESTED = """\
<piecewise>
  <piece>
    <piecewise>
      <piece><ci>k_boost</ci>
        <apply><gt/><ci>X</ci><cn>3</cn></apply></piece>
      <otherwise><ci>k_base</ci></otherwise>
    </piecewise>
    <apply><lt/><ci>X</ci><cn>8</cn></apply>
  </piece>
  <otherwise><ci>k_base</ci></otherwise>
</piecewise>"""

# De Morgan's complement: `or` over the negated comparisons, with the branches
# swapped. Same window everywhere except on the measure-zero boundary itself,
# and it exercises the other connective.
_DISJUNCTIVE = """\
<piecewise>
  <piece><ci>k_base</ci>
    <apply><or/>
      <apply><geq/><ci>X</ci><cn>8</cn></apply>
      <apply><leq/><ci>X</ci><cn>3</cn></apply>
    </apply>
  </piece>
  <otherwise><ci>k_boost</ci></otherwise>
</piecewise>"""

# De Morgan again, but as `not(... and ...)` rather than as `or` over the
# complement — the fourth spelling, and issue #234's. This is the only one of
# the four a `<not/>` in SBML can produce, and until #234 it was the only one
# that did not reach the crossing machinery: `_split_logical_atoms` kept the
# whole `not(((X<8.0) and (X>3.0)))` as one atom, which is neither a clock
# threshold nor a rootable comparison, so the model was declined and ran on the
# difference quotient — 18 % wrong at rtol=1e-8, and no result at all at 1e-10.
_NEGATED = """\
<piecewise>
  <piece><ci>k_base</ci>
    <apply><not/>
      <apply><and/>
        <apply><lt/><ci>X</ci><cn>8</cn></apply>
        <apply><gt/><ci>X</ci><cn>3</cn></apply>
      </apply>
    </apply>
  </piece>
  <otherwise><ci>k_boost</ci></otherwise>
</piecewise>"""

SPELLINGS = {
    "and": _CONJUNCTIVE,
    "nested": _NESTED,
    "or": _DISJUNCTIVE,
    "not_and": _NEGATED,
}


def _sbml(tmp_path, name: str):
    path = tmp_path / f"window_{name}.xml"
    condition = textwrap.indent(SPELLINGS[name], " " * 12)
    path.write_text(_SBML.replace("{CONDITION}", condition))
    return bngsim.Model.from_sbml(path)


def _sens_at_end(tmp_path, name: str, rtol: float):
    """``(X(T_END), dX(T_END)/dk_boost, n_steps)`` for one spelling."""
    sim = bngsim.Simulator(_sbml(tmp_path, name), sensitivity_params=["k_boost"])
    res = sim.run(t_span=(0.0, T_END), n_points=7, rtol=rtol, atol=rtol * 1e-3)
    return (
        float(np.asarray(res.species)[-1, 0]),
        float(np.asarray(res.sensitivities)[-1, 0, 0]),
        int(res.solver_stats.get("n_steps", -1)),
    )


@requires_cc
class TestOneWindowFourSpellings:
    """The invariant that removes the constant. The mathematics is fixed and only
    the syntax varies, so any disagreement between the arms is a bug in how the
    syntax is read — there is nothing else left for it to be."""

    @pytest.mark.parametrize("name", sorted(SPELLINGS))
    @pytest.mark.parametrize("rtol", [1e-8, 1e-10])
    def test_every_spelling_matches_the_closed_form(self, tmp_path, name, rtol):
        """Before the fix the ``and`` and ``or`` arms came back 53 % low at
        ``rtol=1e-8`` and raised :class:`SimulationError` at ``1e-10``, and
        ``not_and`` came back 18 % low and raised the same (issue #234); the
        ``nested`` arm passed both throughout."""
        x_ref, dx_ref = _closed_form()
        x, dx, _steps = _sens_at_end(tmp_path, name, rtol)
        assert x == pytest.approx(x_ref, rel=1e-6)
        assert dx == pytest.approx(dx_ref, rel=1e-6)

    def test_the_spellings_agree_with_each_other_to_the_last_digit(self, tmp_path):
        """Stronger than agreeing with the closed form, and the point of the
        issue: one window written four ways has to produce one sensitivity RHS,
        so the runs are not merely close — they take the same number of steps and
        return the same doubles."""
        results = {name: _sens_at_end(tmp_path, name, 1e-10) for name in SPELLINGS}
        assert len({r[1] for r in results.values()}) == 1, results
        assert len({r[2] for r in results.values()}) == 1, results

    @pytest.mark.parametrize("name", sorted(SPELLINGS))
    def test_no_spelling_falls_back_to_the_difference_quotient(
        self, tmp_path, monkeypatch, caplog, name
    ):
        """The mechanism, asserted directly rather than through the number.

        The codegen cache is redirected at a fresh directory on purpose: the
        decline is emitted during *derivation*, which a cache hit skips, so on a
        warm cache a declined model looks exactly like one that took the analytic
        path.
        """
        import logging

        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "codegen")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _sens_at_end(tmp_path, name, 1e-8)
        assert not [r for r in caplog.records if "sensitivity RHS is declined" in r.getMessage()]


# ─── which call heads are admitted, and why ────────────────────────────────


class TestTheAdmittedHeads:
    """``_condition_call_heads`` is not a list of names someone thought were
    safe: it is the set the ExprTk→sympy preprocessor rewrites into a construct
    before ``parse_expr`` performs any function lookup, so none of them ever
    reaches one. These tests ask the preprocessor rather than the list."""

    PROBES = {
        "if": "if(X>1,a,b)",
        "and": "if((X>1) and (X<2),a,b)",
        "or": "if((X>1) or (X<2),a,b)",
        "not": "if(not(X>1),a,b)",
    }
    UNTRANSLATED = ["xor", "nand", "nor", "xnor"]

    def test_the_set_is_exactly_the_probed_heads(self):
        assert cg._condition_call_heads() == frozenset(self.PROBES)

    @pytest.mark.parametrize("head", sorted(PROBES))
    def test_each_admitted_head_survives_the_round_trip(self, head):
        """An admitted head must leave a fully bound sympy expression — no
        applied undefined function, which is the shape the free-symbol check
        below the scan cannot see and the scan exists to reject."""
        from bngsim._jacobian import _exprtk_to_sympy
        from sympy.core.function import AppliedUndef

        expr = _exprtk_to_sympy(self.PROBES[head])
        assert expr is not None, head
        assert not expr.atoms(AppliedUndef), (head, expr)

    @pytest.mark.parametrize("head", UNTRANSLATED)
    def test_an_unadmitted_connective_really_is_untranslatable(self, head):
        """The companion that makes the exclusion mean something. ExprTk has four
        more connectives and no rewrite maps any of them, so admitting them would
        hand ``parse_expr`` a call it cannot bind — the scan's message ("calls
        unsupported function(s)") is the better of the two failures."""
        from bngsim._jacobian import _exprtk_to_sympy

        assert head not in cg._condition_call_heads()
        assert _exprtk_to_sympy(f"if((X>1) {head} (X<2),a,b)") is None

    @pytest.mark.parametrize("head", sorted(PROBES))
    def test_the_heads_are_admitted_only_behind_the_condition_gate(self, head):
        """Same gate ``if`` has always been behind: ``switch_scope is not None``,
        which means the model's conditions cleared
        :func:`sw.uncompensated_condition_reason`. Without one, every probe must
        still be refused — otherwise the derivation would proceed on a branch
        whose crossing nobody compensates, and sympy differentiates that to a
        clean, wrong ``0``.

        The refusal comes from the *construct* pre-scan rather than the head
        scan, because both read the same flag and the construct scan runs first.
        That is the belt to the scan's braces, and it is why the assertion here
        is on the outcome rather than on which check produced it."""
        scope = cg._FunctionalDfdpScope(
            func_map={},
            c_ref={"X": "y[0]", "a": "p[0]", "b": "p[1]"},
            param_of_alias={"a": "a", "b": "b"},
            param_idx_by_name={"a": 0, "b": 1},
            primary_param_names={"a", "b"},
            derived_exprs={},
            switch_scope=None,
        )
        terms, reason = cg._functional_rate_law_partials(self.PROBES[head], scope)
        assert terms is None
        assert reason.startswith("uses unsupported construct: ")


# ─── the condition scan has to terminate ───────────────────────────────────


class TestTheConditionScanTerminates:
    """``_split_logical_atoms`` re-descended into any part carrying a logical,
    including one it had not reduced. A logical that is neither at depth 0 nor
    inside a strippable paren group left the strip step returning the string
    unchanged, and the function called itself on it forever.

    The blast radius is wider than the sensitivity path, because
    ``switch_gate_cache_digest`` reads the same atoms: on ``MODEL0911047946``
    the ``RecursionError`` propagated into ``compute_model_codegen_hash``, which
    caught it and silently fell back to the source-hash key — issue #216, whose
    own diagnosis blamed ``_canon_update``'s nesting depth. It is this loop.

    ``not((X<hi) and (X>lo))`` used to be the headline shape here; issue #234
    took it out of this class by *reducing* it (see below), which leaves the
    guard load-bearing for what is genuinely irreducible — a logical buried in a
    call argument — and for anything malformed enough that no peel applies."""

    # A logical the strip step cannot reach: neither at depth 0 nor under a
    # negation nor inside a strippable paren group.
    CALL_ARG = "max(a, b and c) > 1"
    # Malformed, so the `not(` peel finds no matching close paren and declines to
    # fire. Nothing downstream will compile this either; the requirement is only
    # that the scan hand it back rather than spin.
    UNBALANCED = "not((a>1) and (b>2)"

    @pytest.mark.parametrize("cond", [CALL_ARG, UNBALANCED])
    def test_an_irreducible_logical_is_returned_whole(self, cond):
        assert sw._split_logical_atoms(cond) == [cond]

    @pytest.mark.parametrize(
        ("cond", "atoms"),
        [
            ("(X<8.0) and (X>3.0)", ["X<8.0", "X>3.0"]),
            ("((t>=sigma)&&(t<tau1))", ["t>=sigma", "t<tau1"]),
            ("(a>1) || ((b>2) && (c>3))", ["a>1", "b>2", "c>3"]),
            ("0>0", ["0>0"]),
        ],
    )
    def test_every_unnegated_shape_is_unchanged(self, cond, atoms):
        """The guard only ever stops a call that could not have returned, so
        nothing that used to produce atoms may produce different ones."""
        assert sw._split_logical_atoms(cond) == atoms

    def test_the_codegen_cache_key_no_longer_falls_back(self, tmp_path):
        """Issue #216's symptom, on a model small enough to keep in the file."""
        model = _sbml(tmp_path, "not_and")
        assert isinstance(cg.compute_model_codegen_hash(model._core), str)


# ─── one operator, two spellings ───────────────────────────────────────────


class TestNegationIsPeeledNotInterpreted:
    """Issue #234. ``_split_logical_atoms`` dropped a leading ``!`` and split
    what was under it, but kept a ``not(...)`` call whole — two readings of one
    operator, and the whole one is the reading every SBML model got, because
    ``_ast_to_exprtk`` renders ``<not/>`` as the call form (ExprTk rejects ``!``
    outright, so the operator spelling is reachable only at this level).

    Peeling is sound for these callers and not merely convenient: they ask
    *where* the branch flips, never which side is true, because the core reads
    f⁻/f⁺ by evaluating the real RHS on each side of the located crossing. De
    Morgan supplies the rest — ∂(¬(A∧B)) ⊆ ∂A ∪ ∂B — so the peeled reading names
    no surface the condition does not have, and names exactly the pair the
    un-negated spelling already registers."""

    @pytest.mark.parametrize(
        ("cond", "atoms"),
        [
            # The pair, both spellings, with and without the loader's extra parens.
            ("not((X>1) and (X<2))", ["X>1", "X<2"]),
            ("not(((X<8.0) and (X>3.0)))", ["X<8.0", "X>3.0"]),
            ("!((a>1) && (b>2))", ["a>1", "b>2"]),
            # A single comparison: peeled AND unparenthesised, because
            # `_relational_split_op` stops looking at depth > 0 and so does not
            # read `(X>3)` as a comparison at all. That is what the `!` spelling
            # used to hand it, and why it was refused where `not(X>3)` is not.
            ("not(X>3.0)", ["X>3.0"]),
            ("!(X>3)", ["X>3"]),
            # Negation is an involution here, and it composes with a connective
            # at depth 0 on either side of it.
            ("not(not(a>1))", ["a>1"]),
            ("not((a>1) and (b>2)) or (c>3)", ["a>1", "b>2", "c>3"]),
            ("not(a>1) and (b>2)", ["a>1", "b>2"]),
            # A `not` that does not wrap the whole part is not a negation of it.
            ("not(a) > 1", ["not(a) > 1"]),
        ],
    )
    def test_the_atoms(self, cond, atoms):
        assert sw._split_logical_atoms(cond) == atoms

    def test_the_two_spellings_agree(self):
        """Stated as an identity rather than as two expected lists, because the
        defect was precisely that the two answers were allowed to differ."""
        assert sw._split_logical_atoms("not((a>1) and (b>2))") == sw._split_logical_atoms(
            "!((a>1) && (b>2))"
        )

    def test_the_model_is_admitted_and_both_crossings_rooted(self, tmp_path, caplog):
        """The decline this lifts, and — asserted in the same breath — the
        machinery behind the admission. An admission with no registered crossing
        is the silent zero the issue #68 gate exists to prevent, so the test that
        the reason is ``None`` is worth nothing without the second assertion."""
        import logging

        path = tmp_path / "negated.net"
        path.write_text(_SWITCHED.replace("{LAW}", "if(not((I>=thresh) and (I<900)),beta,0)*I"))
        core = bngsim.Model.from_net(path)._core
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None
        assert terms
        assert sw.state_switch_conditions(core) == ["I>=thresh", "I<900"]
        assert not [m for m in _messages(caplog) if "sensitivity RHS is declined" in m]

    def test_the_negated_pair_is_the_pair_the_plain_conjunction_registers(self, tmp_path):
        """Issue #153's hazard is two roots at one instant, so a change that
        hands the solver new roots owes an answer about collisions. This is the
        answer: the negated spelling registers no root the un-negated spelling
        does not, so it cannot collide anywhere ``(A and B)`` does not already."""
        plain = _core(tmp_path, "if((I>=thresh) and (I<900),beta,0)*I", name="plain.net")
        negated = _core(tmp_path, "if(not((I>=thresh) and (I<900)),beta,0)*I", name="neg.net")
        assert sw.state_switch_conditions(negated) == sw.state_switch_conditions(plain)

    def test_a_negated_clock_threshold_is_claimed_by_the_clock_path(self, tmp_path):
        """The other side of the partition: peeling must not hand a counter-clock
        threshold to the issue #150 detector, which would jump it twice. The
        crossing under a ``not()`` is a clock crossing exactly as the bare one
        is, so issue #48 claims it and #150 stands off."""
        core = _core(tmp_path, "if(not(t>=sigma),beta,0)*I")
        records, pinned = sw.compute_switch_time_sens(core, ["sigma"], 0.0, 100.0)
        assert records and pinned
        assert sw.state_switch_conditions(core) == []

    @pytest.mark.parametrize("law", ["not(I)", "if(I>=thresh,beta,not(I))*I"])
    def test_a_negation_outside_a_condition_is_still_refused(self, tmp_path, law):
        """The same two-spellings gap one level out, found while fixing this one
        and fixed with it. ``uncompensated_condition_reason`` rejects a
        comparison that is not inside an ``if()`` head — ``beta*(I>1)``, the
        boolean-as-a-number idiom — and watched ``!`` for it but not ``not()``.
        ``not(I)`` is a step at ``I=0`` carrying no operator either of the other
        patterns match, so it was admitted and sympy differentiated ``~I`` to a
        clean ``1``, with nothing warned.

        Peeling does not reach here: peeling is for a *condition*, where the
        branch has a crossing something can be made to locate. Outside one there
        is no ``if()`` head to root on, which is the whole objection."""
        core = _core(tmp_path, law)
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert isinstance(reason, sw.UncompensatedCrossingReason)
        assert "'not' in" in reason and "not inside an if() condition" in reason

    def test_the_event_path_still_refuses_a_negated_trigger(self, tmp_path):
        """The asymmetry the peel must not spread to. An event's ∂t*/∂p is
        derived for a false→true edge, so that reduction orients every atom into
        a lower or an upper bound and takes ``t* = max(lower)``; negation swaps
        those roles, and peeling there would return a confidently wrong number
        rather than a coarser one. The refusal is a whole-trigger text scan that
        runs *before* the splitter, so the peel is unreachable from it."""
        core = _core(tmp_path, "if(t>=sigma,beta,0)*I")
        ctx = core.functional_jacobian_context()
        scope = sw.switch_condition_scope(core, ctx)
        thresholds = sw._threshold_scope(scope, ctx)
        for trigger in ("not(time>=sigma)", "not((time>=sigma) and (time<200))"):
            verdict = sw._analyze_event_trigger(
                core, trigger, scope, thresholds, scope.clocks, set(scope.clock_symbols), 0.0
            )
            assert isinstance(verdict, str) and "is negated" in verdict


# ─── the decline warning's claim ───────────────────────────────────────────
#
# The counter-clock SIR ``test_codegen_switch_condition_sens`` uses, restated
# here rather than imported (a sibling import does not survive the out-of-tree
# test runner). ``counter()`` is a species synthesized at rate 1 and exposed as
# the observable ``t`` — the BNGL idiom for a rate law that reads simulation
# time, so the clock is *detected* rather than assumed. ``erf`` is the
# undifferentiable half below: a genuinely ExprTk-unknown call, so the decline
# it produces has nothing to do with the model's conditions.

_SWITCHED = """\
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
    1 betaI() {LAW}
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


def _core(tmp_path, law: str, name="m.net"):
    net = tmp_path / name
    net.write_text(_SWITCHED.replace("{LAW}", law))
    return bngsim.Model.from_net(net)._core


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


class TestMovingCrossings:
    """What :func:`sw.model_moving_crossings` counts. The question is only
    "can this condition cross at a time that moves", never "is the crossing
    compensated" — because once the model is on the difference quotient, no
    jump is applied either way."""

    def test_a_state_threshold_moves(self, tmp_path):
        core = _core(tmp_path, "if(I>=thresh,beta,0)*I")
        assert sw.model_moving_crossings(core) == ("I>=thresh",)

    def test_both_halves_of_a_compound_condition_are_reported(self, tmp_path):
        core = _core(tmp_path, "if((I>=thresh) and (I<900),beta,0)*I")
        assert sw.model_moving_crossings(core) == ("I>=thresh", "I<900")

    def test_a_parameter_clock_threshold_moves(self, tmp_path):
        core = _core(tmp_path, "if(t>=sigma,beta,0)*I")
        assert sw.model_moving_crossings(core) == ("t>=sigma",)

    def test_a_literal_clock_threshold_does_not(self, tmp_path):
        """``t>=3.0`` crosses at the same instant for every parameter, so
        ``∂t*/∂p`` is exactly 0 and the difference quotient misses nothing.
        Excluded through :func:`sw.fixed_clock_threshold`, the same predicate
        :func:`sw.clock_crossing_compensated` admits it on."""
        core = _core(tmp_path, "if(t>=3.0,beta,0)*I")
        assert sw.fixed_clock_threshold("t>=3.0", sw.switch_condition_scope(core))
        assert sw.model_moving_crossings(core) == ()

    def test_a_constant_comparison_does_not(self, tmp_path):
        """``0>0`` is decided at load — the same ground
        :func:`sw.uncompensated_condition_reason` admits it on."""
        core = _core(tmp_path, "if(0>0,beta,beta)*I")
        assert sw.model_moving_crossings(core) == ()

    def test_a_condition_free_model_does_not(self, tmp_path):
        core = _core(tmp_path, "beta*I")
        assert sw.model_moving_crossings(core) == ()


class TestTheWarningStopsPromisingACorrectFallback:
    """Issue #232's second, independent half. ``CVodeSensInit1`` takes one
    callback for every column, so a decline for a reason that has nothing to do
    with the conditions still puts the model's crossing on the difference
    quotient — and "correct, but slower" is then the sentence that makes a 53 %
    error silent."""

    UNDERIVABLE_AT_A_CROSSING = "if(I>=thresh,beta,0)*erf(I)*I"
    UNDERIVABLE_SMOOTH = "erf(I)*beta*I"

    def test_the_decline_is_tagged(self, tmp_path):
        core = _core(tmp_path, self.UNDERIVABLE_AT_A_CROSSING)
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert isinstance(reason, sw.DeclinedAtMovingCrossingReason)
        assert "unsupported function(s): erf" in reason
        assert "'I>=thresh'" in reason

    def test_the_warning_says_the_fallback_is_wrong(self, tmp_path, caplog):
        import logging

        core = _core(tmp_path, self.UNDERIVABLE_AT_A_CROSSING)
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            cg._functional_dfdp_terms(core, core.codegen_data())
        msgs = _messages(caplog)
        assert any("does NOT recover the missing term" in m for m in msgs)
        assert any("issue #232" in m for m in msgs)
        assert not any("correct, but slower" in m for m in msgs)

    def test_a_smooth_model_keeps_the_correct_fallback_wording(self, tmp_path, caplog):
        """The control, and the thing that keeps this change from just moving the
        dishonesty: the SAME undifferentiable call, on a model with no branch
        condition at all, really does fall back to a correct difference quotient
        and must keep saying so."""
        import logging

        core = _core(tmp_path, self.UNDERIVABLE_SMOOTH)
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert not isinstance(reason, sw.UncompensatedCrossingReason)
        msgs = _messages(caplog)
        assert any("correct, but slower" in m for m in msgs)
        assert not any("does NOT recover" in m for m in msgs)

    def test_a_literal_clock_switch_keeps_it_too(self, tmp_path, caplog):
        """The second control. A crossing at a fixed time contributes no jump to
        any column, so the fallback really is correct there — over-warning would
        cost the message its meaning."""
        import logging

        core = _core(tmp_path, "if(t>=3.0,beta,0)*erf(I)*I")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert not isinstance(reason, sw.UncompensatedCrossingReason)
        assert any("correct, but slower" in m for m in _messages(caplog))

    def test_an_uncompensated_crossing_keeps_its_own_class_and_pointer(self, tmp_path, caplog):
        """The issue #146 class must not be downgraded — nor re-tagged. It got
        its verdict by naming the crossing itself, which is the better message,
        and it still points at issue #150 rather than at a decline there is
        nothing to remove."""
        import logging

        # Two clock terms, so issue #418's single-power solve declines it and it
        # stays the crossing nothing brackets (a bare `time()*time()>=thresh` is
        # compensated now — see test_codegen_switch_condition_sens.py).
        core = _core(tmp_path, "if((time()-5)*(time()-5)>=thresh,beta,0)*I")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert isinstance(reason, sw.UncompensatedCrossingReason)
        assert not isinstance(reason, sw.DeclinedAtMovingCrossingReason)
        assert any("issue #150" in m for m in _messages(caplog))

    def test_the_reaction_wrapper_does_not_downgrade_the_subclass(self):
        """``_carry_reason_class`` rebuilds the reason as ``type(inner)``. Doing
        it as the base class would silently swap the closing sentence — the exact
        drift that helper exists to prevent."""
        inner = sw.DeclinedAtMovingCrossingReason("inner")
        assert isinstance(cg._carry_reason_class(inner, "wrapped"), type(inner))
        base = sw.UncompensatedCrossingReason("inner")
        wrapped = cg._carry_reason_class(base, "wrapped")
        assert isinstance(wrapped, sw.UncompensatedCrossingReason)
        assert not isinstance(wrapped, sw.DeclinedAtMovingCrossingReason)
