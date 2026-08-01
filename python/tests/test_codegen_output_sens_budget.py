"""GH #97 — a build-time derivation budget for the #198 ``d func/dθ`` path.

#90 bounded every symbolic-derivation site on the sensitivity-RHS build. One
sympy site on the **same build** was left unbounded: ``_analyze_output_sens`` →
``differentiate_expression_output_partials``, which runs one ``sp.diff`` per
(user function, directly-referenced symbol) pair for the #198 chain rule, plus
the derived-parameter chain rule that feeds it. So the "build appears to hang"
failure mode #90 removed was still reachable one emitter over.

Three things are under test, and the middle one is not a cost story:

* **The policy is keyed on the derivation-step count.** Every other derivation
  budget here scales with species count (#187), which on this path is the wrong
  axis: a model can carry thousands of global functions on a few hundred species
  (``MODEL1112100000``: 1265 species, 3633 functions, 14532 steps). Same
  ``max(base, slope × size)`` shape over the size that actually drives the cost,
  counted from the reference graph's own tokens before any sympy runs. It reads
  the same env var as ∂f/∂p — one knob for one build — but resolves its own
  deadline, because the two phases run on the same build and a shared one would
  let either starve the other (``BIOMD0000000497``: 8.1 s in ∂f/∂p, 19.2 s here,
  so one 20 s deadline cuts a model that today derives in full).

* **The analysis is memoized, for correctness.** It is genuinely evaluated twice
  per sensitivity workflow — once by the C emitter, once by the ``Result``'s
  support map — and a wall-clock bound makes it no longer a pure function of the
  model. Two independent evaluations can cut at different functions, and then the
  emitted C carries a NaN sentinel for a function the support map reports as
  supported. The memo is what makes the budget safe to add at all.

* **An expiry marks functions ``unsupported``, it does not decline.** Unlike
  ∂f/∂p — one callback for every column, so one bad rate law takes the model —
  output sensitivities are per function. Every function derived before the
  deadline keeps working and is still numerically right; the rest raise the
  reason (naming the override) when selected. Declining the symbol would take the
  derived ones down with it; refusing the build would take the sensitivity RHS.

What this budget does **not** buy, so no test here claims it: a wall-clock
deadline can only be checked *between* sympy calls, so a single pathological
expression still overshoots by one uninterruptible ``sp.diff``. The budget is a
backstop against the accumulating case (many moderately-priced functions), not a
fix for the outliers — those are printer/derivation defects (#96, #99) and are
fixed where they live.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim import _jacobian as jac
from bngsim._jacobian import (
    _BUDGET_PER_STEP_S,
    _DEFAULT_DERIVATION_BUDGET_S,
    _DerivationBudgetExceeded,
    _output_sens_derivation_budget_s,
    _sens_derivation_budget_s,
)

pytest.importorskip("sympy")

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# The #198 fixtures, reused so this budget is measured against the same models
# the feature's own suite validates against FD (test_expression_output_sensitivities).
#   chain:   four user functions — scaled/ratio/combo/tdep — so a cut has
#            somewhere to land partway through.
#   derived: kd = 2*kbase, so the derived-parameter phase has work to bound.
CHAIN_NET = str(DATA_DIR / "expr_sens_chain.net")
DERIVED_NET = str(DATA_DIR / "expr_sens_derived.net")

_SENS_ENV = "BNGSIM_SENS_DERIV_BUDGET_S"
_JAC_ENV = "BNGSIM_JAC_DERIV_BUDGET_S"

# Measured over the BioModels SBML corpus on the current emitters (see
# _output_sens_derivation_budget_s for the full table). Two numbers govern
# whether this budget is a guard against a hang or a behaviour change on real
# models: the worst per-step rate above the knee, where the slope decides, and
# the worst absolute time below it, where the base does.
_WORST_CORPUS_S_PER_STEP = 0.0055  # MODEL1603150001: 15568 steps, 85.7 s
_WORST_SMALL_MODEL_S = 1.42  # BIOMD0000000247: 249 steps, well under the knee

# Step count at which the slope overtakes the base — below it every model gets
# the same budget the other two derivation phases start from.
_KNEE_STEPS = int(_DEFAULT_DERIVATION_BUDGET_S / _BUDGET_PER_STEP_S)


@pytest.fixture(autouse=True)
def _clear_budget_env(monkeypatch):
    monkeypatch.delenv(_SENS_ENV, raising=False)
    monkeypatch.delenv(_JAC_ENV, raising=False)


class _Clock:
    """A ``time.perf_counter`` that advances exactly 1 s per reading.

    Timing this budget against the real clock would be flaky in both directions;
    a counted clock makes "the deadline expired after N checks" exact. Every
    deadline check on this path is one reading, so a budget of B seconds admits
    the first B checks and cuts at the next.

    :meth:`exhaust` jumps it past any budget, so that a *second* evaluation of
    the same analysis would cut at a different function than the first — which is
    how the memo's correctness claim is tested rather than asserted.
    """

    def __init__(self) -> None:
        self.n = 0
        self.offset = 0.0

    def __call__(self) -> float:
        self.n += 1
        return self.offset + self.n

    def exhaust(self) -> None:
        self.offset = 1e9


def _nan_functions(src: str) -> set[str]:
    """Function names the emitted evaluator fills with the NaN sentinel."""
    return set(re.findall(r"fs\[\d+\] = NAN;\s+/\* (\w+): unsupported \*/", src))


def _unsupported(support: dict[str, str | None]) -> set[str]:
    return {name for name, reason in support.items() if reason is not None}


def _statuses(analysis: dict) -> dict[str, str]:
    return {info["name"]: info["status"] for info in analysis["func_infos"]}


def _steps_for(model) -> int:
    """The step count the analysis sizes its budget from, captured where it is
    actually used rather than re-derived in the test."""
    seen: dict[str, int] = {}
    real = cg._output_sens_derivation_deadline

    def spy(n_steps=0):
        seen["n"] = n_steps
        return real(n_steps)

    cg._output_sens_derivation_deadline = spy
    try:
        cg._analyze_output_sens(model)
    finally:
        cg._output_sens_derivation_deadline = real
    return seen["n"]


# ─── the policy ────────────────────────────────────────────────────────────


class TestPolicy:
    @pytest.mark.parametrize("n_steps", [0, 1, 100, _KNEE_STEPS])
    def test_a_small_model_gets_the_shared_base(self, n_steps):
        assert _output_sens_derivation_budget_s(n_steps=n_steps) == _DEFAULT_DERIVATION_BUDGET_S

    def test_it_scales_with_the_step_count_not_the_species_count(self):
        """The one place this policy departs from #187's. The cost here is one
        parse per expression plus one derivative per referenced symbol, and a model
        can carry thousands of global functions on a few hundred species — so a
        per-species slope would be loose exactly where the work is."""
        n = 4 * _KNEE_STEPS
        assert _output_sens_derivation_budget_s(n_steps=n) == pytest.approx(_BUDGET_PER_STEP_S * n)
        # Species do not enter it at all: the ∂f/∂p budget on a genome-scale model
        # is generous where this one, on the same model with few functions, is not.
        assert _sens_derivation_budget_s(n_species=100_000) > _output_sens_derivation_budget_s()

    def test_the_policy_clears_what_the_corpus_actually_costs(self):
        """The regression guard for "0 corpus diffs": a budget a real model can hit
        is not a guard against a hang, it is a new failure mode. Above the knee the
        slope has to clear the worst measured per-step rate; below it the base has
        to clear the worst measured model outright. A flat 20 s budget — the other
        candidate — fails the first of these on two BioModels models."""
        assert _BUDGET_PER_STEP_S >= 8 * _WORST_CORPUS_S_PER_STEP
        assert _DEFAULT_DERIVATION_BUDGET_S >= 8 * _WORST_SMALL_MODEL_S

    @pytest.mark.parametrize("raw", ["inf", "none", "off", "0", "-1", " INF ", "nan"])
    def test_the_override_can_disable_it(self, monkeypatch, raw):
        monkeypatch.setenv(_SENS_ENV, raw)
        assert _output_sens_derivation_budget_s(n_steps=10_000) is None

    def test_the_override_wins_over_the_size_policy(self, monkeypatch):
        monkeypatch.setenv(_SENS_ENV, "2.5")
        assert _output_sens_derivation_budget_s(n_steps=10_000) == 2.5

    def test_a_malformed_override_degrades_to_the_policy(self, monkeypatch):
        """A typo must not read as "no budget at all" — that is the failure mode
        this issue is about."""
        monkeypatch.setenv(_SENS_ENV, "twenty")
        assert _output_sens_derivation_budget_s() == _DEFAULT_DERIVATION_BUDGET_S

    def test_the_jacobian_budget_does_not_reach_it(self, monkeypatch):
        """``BNGSIM_JAC_DERIV_BUDGET_S=inf`` is the documented genome-scale escape
        hatch for the analytical Jacobian. It buys a different thing and falls back
        to a different path, so it must not silently uncap this one."""
        monkeypatch.setenv(_JAC_ENV, "inf")
        assert _output_sens_derivation_budget_s() == _DEFAULT_DERIVATION_BUDGET_S

    def test_the_step_count_is_the_work_that_is_coming(self):
        """Counted from the reference graph's tokens, so the budget is sized before
        any sympy runs — and deliberately an upper bound, since a symbol that
        cancels out of the parsed expression takes no ``diff``."""
        # Four functions parsed, plus their referenced symbols: scaled(scale,
        # A_obs) + ratio(A_obs, BC2, eps) + combo(scaled, ratio) + tdep(k1, A_obs).
        # time() is not a differentiation variable, so it is not a step.
        assert _steps_for(bngsim.Model.from_net(CHAIN_NET)) == 4 + 9
        # kd = 2.0*kbase adds the derived phase's own parse and derivative.
        assert _steps_for(bngsim.Model.from_net(DERIVED_NET)) == 2 + 3

    def test_each_phase_gets_its_own_deadline(self, monkeypatch):
        """One knob, two clocks. ∂f/∂p and ``d func/dθ`` run on the same build and
        read the same env var, but a *shared* deadline would let a slow ∂f/∂p
        starve this phase — measured, not hypothetical: ``BIOMD0000000497`` spends
        8.1 s in ∂f/∂p and 19.2 s here, so one 20 s deadline cuts a model that
        today derives in full."""
        ticks = iter([0.0, 15.0])
        monkeypatch.setattr(cg.time, "perf_counter", lambda: next(ticks))
        dfdp = cg._sens_derivation_deadline(n_species=3)
        # ∂f/∂p has already spent 15 of its 20 s when this phase starts; it still
        # gets a full budget, where a shared deadline would have left it 5 s.
        assert cg._output_sens_derivation_deadline() == 15.0 + _DEFAULT_DERIVATION_BUDGET_S
        assert dfdp == _DEFAULT_DERIVATION_BUDGET_S


# ─── the deadline is checked *during* the derivation ───────────────────────


class TestDeadlinePlumbing:
    _CREF = dict(
        species_cref={"A": "y[0]"},
        observable_cref={"A_obs": "obs[0]"},
        param_cref={"k": "p[0]", "scale": "p[1]"},
        function_cref={},
    )

    def test_no_deadline_is_the_unbudgeted_path(self):
        """Every caller outside this analysis passes ``None``, so the default must
        be a no-op rather than a budget of zero."""
        partials, reason = jac.differentiate_expression_output_partials("k*A_obs", **self._CREF)
        assert reason is None and partials["param"]

    def test_an_expired_deadline_raises_rather_than_reporting_a_reason(self):
        """A wall-clock expiry is a property of the *build*, not of this function
        body — the caller has to tell the two apart, because one of them means
        "stop attempting the rest" and the other does not."""
        with pytest.raises(_DerivationBudgetExceeded):
            jac.differentiate_expression_output_partials(
                "k*A_obs", deadline=cg.time.perf_counter() - 1.0, **self._CREF
            )

    def test_the_deadline_is_checked_between_partials(self, monkeypatch):
        """Not merely on entry. A body referencing many symbols is one ``sp.diff``
        per symbol, so a check only at the top would bound nothing on exactly the
        body that needs bounding."""
        clock = _Clock()
        monkeypatch.setattr(cg.time, "perf_counter", clock)
        # Entry check reads 1, then one reading per partial: a deadline of 2 admits
        # the entry check and the first partial, and cuts on the second.
        with pytest.raises(_DerivationBudgetExceeded):
            jac.differentiate_expression_output_partials(
                "k*A_obs*scale", deadline=2.0, **self._CREF
            )
        assert clock.n == 3

    def test_the_derived_parameter_phase_is_bounded_too(self, monkeypatch):
        """The other sympy on this analysis: ∂p_d/∂primary for every derived
        parameter, run before any function. #99's 5-species hang lives here, and a
        model with thousands of ``# ConstantExpression`` parameters can accumulate
        the whole budget here without reaching a function at all."""
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        analysis = cg._analyze_output_sens(bngsim.Model.from_net(DERIVED_NET))
        assert analysis["derived_expansion"] == {}
        assert _statuses(analysis) == {"dfn": "unsupported"}
        assert "derived parameters" in analysis["func_infos"][0]["reason"]


# ─── what an expiry costs, and what it does not ────────────────────────────


class TestPartialCut:
    """The behaviour the issue argues for over the two alternatives: not a
    declined symbol (which would take the derived functions down too) and not a
    refused build (which would take the sensitivity RHS with it)."""

    def _cut(self, monkeypatch, budget="3"):
        clock = _Clock()
        monkeypatch.setattr(cg.time, "perf_counter", clock)
        monkeypatch.setenv(_SENS_ENV, budget)
        return clock, bngsim.Model.from_net(CHAIN_NET)

    def test_functions_derived_before_the_deadline_survive(self, monkeypatch):
        _clock, model = self._cut(monkeypatch)
        analysis = cg._analyze_output_sens(model)
        assert _statuses(analysis) == {
            "scaled": "ok",
            "ratio": "unsupported",
            "combo": "unsupported",
            "tdep": "unsupported",
        }
        assert analysis["func_infos"][0]["partials"]["param"]  # a real derivative

    def test_the_whole_analysis_is_not_declined(self, monkeypatch):
        """``decline`` is the whole-model refusal (rateOf, embedded tfun, no
        functions) and turns the evaluator into no symbol at all. A budget expiry
        is not that: the symbol is still emitted, with the derived functions
        intact."""
        _clock, model = self._cut(monkeypatch)
        assert cg._analyze_output_sens(model)["decline"] is None
        src = cg.generate_output_sens_from_model(model)
        assert src is not None
        assert _nan_functions(src) == {"ratio", "combo", "tdep"}
        assert "obs_sens_c" in src  # observable sensitivities are untouched

    def test_the_reason_is_specific_and_names_the_override(self, monkeypatch):
        _clock, model = self._cut(monkeypatch)
        reason = cg.output_sens_support(model)["ratio"]
        assert _SENS_ENV in reason
        assert "budget" in reason
        assert "derived 1 of 4 functions" in reason  # how far it got

    def test_the_cut_is_logged_once(self, monkeypatch, caplog):
        """A build that quietly turned every ``d func/dθ`` into a NaN is not
        something to discover at selection time. Once, not once per function, and
        not once per caller — the memo is what makes the second true."""
        _clock, model = self._cut(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            cg.generate_output_sens_from_model(model)
            cg.output_sens_support(model)
        hits = [r for r in caplog.records if "d(function)/d" in r.getMessage()]
        assert len(hits) == 1
        assert _SENS_ENV in hits[0].getMessage()

    def test_an_exhausted_budget_supports_nothing_but_still_emits(self, monkeypatch):
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        model = bngsim.Model.from_net(CHAIN_NET)
        src = cg.generate_output_sens_from_model(model)
        assert src is not None
        assert _nan_functions(src) == {"scaled", "ratio", "combo", "tdep"}


# ─── one analysis per model ────────────────────────────────────────────────


class TestMemo:
    def test_the_analysis_is_computed_once_per_model(self):
        model = bngsim.Model.from_net(CHAIN_NET)
        assert cg._analyze_output_sens(model) is cg._analyze_output_sens(model)

    def test_the_two_callers_share_one_evaluation(self, monkeypatch):
        """The emitter and the support map each ran the full analysis, so a
        sensitivity workflow paid it twice — 85.7 s twice on the worst BioModels
        model."""
        calls: list[str] = []
        real = jac.differentiate_expression_output_partials

        def counting(body, **kw):
            calls.append(body)
            return real(body, **kw)

        monkeypatch.setattr(jac, "differentiate_expression_output_partials", counting)
        model = bngsim.Model.from_net(CHAIN_NET)
        cg.generate_output_sens_from_model(model)
        cg.output_sens_support(model)
        assert len(calls) == 4  # one per user function, not two

    def test_a_cut_analysis_is_not_re_derived_for_the_support_map(self, monkeypatch):
        """The correctness half of the memo, and the reason it had to land with
        the budget rather than after it. Two independent evaluations under a
        wall-clock bound can cut at different functions; the emitted C would then
        carry a NaN sentinel for a function the support map calls supported (or,
        as here, the support map would condemn one the C computes correctly).

        The clock is exhausted between the two callers, so a second evaluation
        would cut at the very first function — which is exactly what a missing
        memo looks like from here."""
        clock = _Clock()
        monkeypatch.setattr(cg.time, "perf_counter", clock)
        monkeypatch.setenv(_SENS_ENV, "3")
        model = bngsim.Model.from_net(CHAIN_NET)

        src = cg.generate_output_sens_from_model(model)
        assert _nan_functions(src) == {"ratio", "combo", "tdep"}  # a real partial cut

        clock.exhaust()
        assert _unsupported(cg.output_sens_support(model)) == _nan_functions(src)

    def test_a_budget_change_re_derives(self, monkeypatch):
        """The memo is keyed on the override for the same reason the ``.net``
        ``.so`` cache is: an expiry changes which functions come back supported,
        so an analysis made under one budget must not be served to a caller that
        set another."""
        model = bngsim.Model.from_net(CHAIN_NET)
        assert _unsupported(cg.output_sens_support(model)) == set()
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        assert _unsupported(cg.output_sens_support(model)) == {"scaled", "ratio", "combo", "tdep"}
        monkeypatch.delenv(_SENS_ENV)
        assert _unsupported(cg.output_sens_support(model)) == set()

    def test_a_clone_inherits_the_analysis(self):
        """Parallel fitting clones a warmed model per worker; re-running this
        sympy N times is the cost the warm-clone invariant exists to avoid."""
        model = bngsim.Model.from_net(CHAIN_NET)
        analysis = cg._analyze_output_sens(model)
        assert cg._analyze_output_sens(model.clone()) is analysis


# ─── end to end ────────────────────────────────────────────────────────────


class TestEndToEnd:
    @pytest.fixture(autouse=True)
    def _force_codegen(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)

    def _run(self, model):
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1", "scale"])
        return sim.run(t_span=(0.0, 8.0), n_points=5)

    def test_a_derived_function_is_still_numerically_right(self, monkeypatch):
        """The whole argument for per-function ``unsupported`` over a decline:
        ``scaled`` was differentiated before the deadline, so it answers — and
        answers correctly. At t=0, d(scale·A_obs)/d scale = A_obs(0) = 100 and
        d(scale·A_obs)/d k1 = scale·dA/dk1 = 0."""
        clock = _Clock()
        monkeypatch.setattr(cg.time, "perf_counter", clock)
        monkeypatch.setenv(_SENS_ENV, "3")
        r = self._run(bngsim.Model.from_net(CHAIN_NET))
        row0 = np.asarray(r.output_sensitivities("expression:scaled"))[0].ravel()
        assert row0 == pytest.approx([0.0, 100.0])

    def test_a_cut_function_raises_the_budget_reason(self, monkeypatch):
        """And the rest fail loudly and specifically rather than returning a NaN
        row or a bare empty-block error."""
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        r = self._run(bngsim.Model.from_net(CHAIN_NET))
        with pytest.raises(ValueError, match=_SENS_ENV):
            r.output_sensitivities("expression:scaled")
        # The columns this budget never touched are unaffected.
        assert np.all(np.isfinite(np.asarray(r.output_sensitivities("observable:A_obs"))))

    def test_the_default_budget_changes_nothing(self, monkeypatch):
        """The regression guard for every model that already had output
        sensitivities: nothing on the corpus is near this budget, so under the
        default every function must still be supported and finite."""
        r = self._run(bngsim.Model.from_net(CHAIN_NET))
        for name in ("scaled", "ratio", "combo", "tdep"):
            col = np.asarray(r.output_sensitivities(f"expression:{name}"))
            assert np.all(np.isfinite(col))
