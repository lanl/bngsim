"""GH #90 — a build-time derivation budget for the sensitivity ∂f/∂p path.

#95/#187 bounded the symbolic derivation of the analytical *Jacobian*: a model
that does not derive in time falls back to the finite-difference Jacobian instead
of hanging the load. The ∂f/∂p path added by #55 (#65/#66/#67/#68) does the same
kind of sympy work — one ``sp.diff`` per (distinct rate law, parameter it reads)
pair, the one axis here that grows with the *product* of model size and parameter
count — and had no budget at all. The failure mode that removes is a build that
appears to hang rather than one that declines and says why.

Two things are under test, and the second is why the first is safe:

* **The policy.** The sensitivity budget shares the Jacobian's base, slope and
  override grammar (one spelling, so they cannot drift) but has its own env var
  and is **never unbounded by size**. That last difference is the design decision
  worth pinning: ``_FD_NONVIABLE_SPECIES`` exists because past that size an FD
  Jacobian does not converge, so there is nothing to fall back *to*; the
  sensitivity fallback is CVODES' internal difference quotient, which is what
  every Functional model used before #55 and stays viable at every scale.

* **The plumbing.** The deadline is checked *during* the derivation (on entry to
  each rate law, before each ``sp.diff``, and inside the derived-parameter chain
  rule), not merely before it — otherwise the budget bounds nothing on the single
  pathological model it exists for. An expiry declines through the same channel
  as every other decline, names the override, and leaves a model that still
  solves.

The one thing deliberately *not* budgeted here is ``_derived_expr_partials_numeric``
(the #43 IC-seed path). It runs at solve setup rather than on this build, and a
partial result there is a silently wrong ∂x₀/∂p rather than a slower-but-correct
fallback, so it needs its own all-or-nothing story and its own issue.
"""

from __future__ import annotations

import logging
import math

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim._jacobian import (
    _BUDGET_PER_SPECIES_S,
    _DEFAULT_DERIVATION_BUDGET_S,
    _FD_NONVIABLE_SPECIES,
    _derivation_budget_s,
    _DerivationBudgetExceeded,
    _sens_derivation_budget_s,
)

pytest.importorskip("sympy")

_SENS_ENV = "BNGSIM_SENS_DERIV_BUDGET_S"
_JAC_ENV = "BNGSIM_JAC_DERIV_BUDGET_S"


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")

# The one test here that builds a Simulator carried an xfail(strict) quarantine
# for GH #85 — under the MIR JIT backend a Functional model constructed *with*
# ``sensitivity_params`` did not compile, because the JIT prelude never supplied
# the ``size_t`` the GH #198 ``bngsim_codegen_output_sens`` block names. Fixed in
# mir_jit.hpp; test_codegen_jit_prelude.py owns the regression.


@pytest.fixture(autouse=True)
def _clear_budget_env(monkeypatch):
    """Both budgets resolve from their env var first, so every case here starts
    from the size-derived default unless it sets one explicitly."""
    monkeypatch.delenv(_SENS_ENV, raising=False)
    monkeypatch.delenv(_JAC_ENV, raising=False)


# ─── fixtures ──────────────────────────────────────────────────────────────
#
# Same shapes as test_codegen_functional_sens_rhs.py: a Functional law (which
# reaches _functional_rate_law_partials) and an Elementary model with a derived
# rate constant (which reaches only the chain-rule loop, the *other* budgeted
# site and the one an all-Elementary model can still hang on).

SIR = """\
begin parameters
    1 S0     2e7  # Constant
    2 I0     1  # Constant
    3 beta   1/S0  # ConstantExpression
    4 gamma  1/7  # Constant
end parameters
begin functions
    1 betaI() beta*I
end functions
begin species
    1 person(state~S) S0
    2 person(state~I) I0
    3 person(state~R) 0
end species
begin reactions
    1 1 2 betaI #_R1
    2 2 3 gamma #_R2
end reactions
begin groups
    1 S                    1
    2 I                    2
    3 R                    3
end groups
"""

# Smooth saturating algebra, and — unlike SIR — a model whose difference-quotient
# fallback actually resolves, so the cost of a decline can be measured against the
# analytic answer rather than against a step-budget failure.
HILL = """\
begin parameters
    1 kmax   3.5  # Constant
    2 Km     4.0  # Constant
    3 n      2.0  # Constant
    4 kdeg   0.2  # Constant
end parameters
begin functions
    1 vsat() kmax*Atot^n/(Km^n + Atot^n)
end functions
begin species
    1 A() 6.0
    2 B() 1.0
end species
begin reactions
    1 1 2 vsat #_R1
    2 2 1 kdeg #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""

# ``sin`` puts the ∂f/∂x half (J·yS) on the SymPy fallback: the #151 native
# saturable emitters cover mass-action, Hill and exponential shapes without ever
# importing SymPy, so a fixture from that family would exercise no derivation to
# budget. test_codegen_determinism.py needs ``sin()`` for the same reason.
TRANSCENDENTAL = """\
begin parameters
    1 kmax   3.5  # Constant
    2 Km     4.0  # Constant
    3 kdeg   0.2  # Constant
end parameters
begin functions
    1 vosc() kmax*sin(Atot/Km)
end functions
begin species
    1 A() 6.0
    2 B() 1.0
end species
begin reactions
    1 1 2 vosc #_R1
    2 2 1 kdeg #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""

ELEMENTARY = """\
begin parameters
    1 k1     0.3  # Constant
    2 scale  2.0  # Constant
    3 k2     k1*scale  # ConstantExpression
end parameters
begin species
    1 A() 10.0
    2 B() 0.0
end species
begin reactions
    1 1 2 k1 #_R1
    2 2 1 k2 #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
end groups
"""


def _model(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _net(tmp_path, text, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return str(net)


# ─── the policy ────────────────────────────────────────────────────────────


class TestPolicy:
    @pytest.mark.parametrize("n_species", [0, 1, 295, 1000])
    def test_small_models_share_the_jacobian_base(self, n_species):
        """Below the scaling knee the sensitivity budget is the #95 base, exactly
        as the Jacobian's is — the two policies are one policy with two dials."""
        assert _sens_derivation_budget_s(n_species=n_species) == _DEFAULT_DERIVATION_BUDGET_S
        assert _sens_derivation_budget_s(n_species=n_species) == _derivation_budget_s(
            n_species=n_species
        )

    def test_the_budget_scales_with_species_count(self):
        n = int(4 * _DEFAULT_DERIVATION_BUDGET_S / _BUDGET_PER_SPECIES_S)
        assert n < _FD_NONVIABLE_SPECIES
        assert _sens_derivation_budget_s(n_species=n) == pytest.approx(_BUDGET_PER_SPECIES_S * n)

    def test_it_stays_finite_where_the_jacobian_goes_unbounded(self):
        """The one substantive difference between the two budgets, and the reason
        it exists: past ``_FD_NONVIABLE_SPECIES`` an FD Jacobian does not converge,
        so the Jacobian derivation is mandatory and runs to completion. Declining a
        sensitivity RHS only hands the columns to CVODES' difference quotient,
        which is correct at any size — so there is never a reason to let *this*
        derivation run unbounded, and a genome-scale model gets a decline instead
        of a build that appears to hang."""
        n = _FD_NONVIABLE_SPECIES * 4
        assert _derivation_budget_s(n_species=n) is None
        budget = _sens_derivation_budget_s(n_species=n)
        assert budget is not None
        assert math.isfinite(budget)
        # Generous, though: ~10x the observed derivation rate, so a model that
        # scales like a real one is not cut off for being large.
        assert budget == pytest.approx(_BUDGET_PER_SPECIES_S * n)

    @pytest.mark.parametrize("raw", ["inf", "none", "off", "0", "-1", " INF ", "nan"])
    def test_the_override_can_disable_it(self, monkeypatch, raw):
        monkeypatch.setenv(_SENS_ENV, raw)
        assert _sens_derivation_budget_s(n_species=10) is None

    def test_the_override_wins_over_the_size_policy(self, monkeypatch):
        monkeypatch.setenv(_SENS_ENV, "2.5")
        assert _sens_derivation_budget_s(n_species=_FD_NONVIABLE_SPECIES * 4) == 2.5

    def test_a_malformed_override_degrades_to_the_policy(self, monkeypatch):
        """A typo must not read as "no budget at all" — that is the failure this
        whole issue is about."""
        monkeypatch.setenv(_SENS_ENV, "twenty")
        assert _sens_derivation_budget_s(n_species=10) == _DEFAULT_DERIVATION_BUDGET_S

    def test_the_two_budgets_are_independent(self, monkeypatch):
        """``BNGSIM_JAC_DERIV_BUDGET_S=inf`` is the documented way to keep a
        genome-scale model's analytical Jacobian. It buys a different thing and
        falls back to a different path, so it must not silently uncap this one —
        which would restore exactly the hang #90 removes."""
        monkeypatch.setenv(_JAC_ENV, "inf")
        assert _derivation_budget_s(n_species=10) is None
        assert _sens_derivation_budget_s(n_species=10) == _DEFAULT_DERIVATION_BUDGET_S

        monkeypatch.setenv(_SENS_ENV, "inf")
        monkeypatch.setenv(_JAC_ENV, "3")
        assert _derivation_budget_s(n_species=10) == 3.0
        assert _sens_derivation_budget_s(n_species=10) is None


# ─── the cache key ─────────────────────────────────────────────────────────


class TestCacheKey:
    """The ``.net`` path keys its ``.so`` on the model's *content*, not on the
    generated C, so anything that changes whether the sens RHS is emitted has to
    reach the key — the trap #67's A/B hatch already had to sidestep."""

    def test_unset_leaves_every_existing_key_untouched(self):
        assert cg._sens_budget_cache_tag() == ""

    def test_each_override_gets_its_own_namespace(self, monkeypatch):
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        tight = cg._sens_budget_cache_tag()
        monkeypatch.setenv(_SENS_ENV, "inf")
        loose = cg._sens_budget_cache_tag()
        assert tight and loose and tight != loose

    def test_the_tag_is_normalized(self, monkeypatch):
        monkeypatch.setenv(_SENS_ENV, " INF ")
        assert cg._sens_budget_cache_tag() == ":sens_budget=inf"

    @requires_cc
    def test_a_tight_budget_is_not_served_a_cached_analytic_so(self, tmp_path, monkeypatch):
        """End to end through the real cache: emit under the default budget, then
        re-request the same .net under a budget that cannot be met. Sharing a key
        would hand back the analytic .so and make the decline invisible."""
        model = _model(tmp_path, SIR)
        net = str(tmp_path / "m.net")
        analytic = cg.prepare_codegen(net, model=model)
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        declined = cg.prepare_codegen(net, model=model)
        assert analytic != declined


# ─── the deadline is checked *during* the derivation ───────────────────────


class TestDeadlinePlumbing:
    def test_an_expired_deadline_stops_the_derived_chain_rule(self):
        """``_derived_param_jacobian_checked`` raises rather than returning a
        reason: a wall-clock expiry is a property of the build, not of this
        expression, and the caller memoizes reasons per expression."""
        args = ("k1*scale", {"k1", "scale"}, {"k1": 0, "scale": 1})
        jac, reason = cg._derived_param_jacobian_checked(*args)
        assert reason is None and jac
        with pytest.raises(_DerivationBudgetExceeded):
            cg._derived_param_jacobian_checked(*args, deadline=cg.time.perf_counter() - 1.0)

    def test_no_deadline_is_the_unbudgeted_path(self):
        """Every caller outside the sensitivity build passes ``None``, so the
        default must be a no-op rather than a budget of zero."""
        cg._check_derivation_deadline(None)  # must not raise

    def test_the_jacobian_half_of_the_sens_rhs_is_bounded_too(self, tmp_path):
        """``ySdot = J·yS + ∂f/∂p`` has two derivations, and #90's issue text names
        only the second. The first is a *re*-derivation of what
        ``attach_functional_jacobian`` ran at load — but one that ignores its own
        clock, and on a model whose load-time attach was itself cut off by the #95
        budget there is no earlier bound to inherit. Bounding only ∂f/∂p would
        leave the same hang reachable one call later."""
        model = _model(tmp_path, TRANSCENDENTAL)
        data = model._core.codegen_data()
        assert cg._functional_jacobian_groups(model._core, data, cg._jacv_add) is not None
        with pytest.raises(_DerivationBudgetExceeded):
            cg._functional_jacobian_groups(
                model._core, data, cg._jacv_add, cg.time.perf_counter() - 1.0
            )

    def test_the_budget_is_enforced_mid_derivation(self, tmp_path, monkeypatch):
        """The point of checking per rate law and per parameter rather than once
        up front. A clock that advances a second per reading expires a 2 s budget
        only *after* the derivation has started, so a build that declines here
        cannot have been stopped by a pre-flight check."""
        model = _model(tmp_path, SIR)
        ticks = iter(range(0, 10_000))
        monkeypatch.setattr(cg.time, "perf_counter", lambda: float(next(ticks)))
        monkeypatch.setenv(_SENS_ENV, "2")
        assert cg.generate_sens_from_model(model, functional=True) is None

    def test_a_generous_budget_on_the_same_clock_still_emits(self, tmp_path, monkeypatch):
        """The control for the case above: same fake clock, same model, a budget
        the derivation fits inside. Without this, the test above would also pass
        if the deadline were simply always expired."""
        model = _model(tmp_path, SIR)
        ticks = iter(range(0, 10_000))
        monkeypatch.setattr(cg.time, "perf_counter", lambda: float(next(ticks)))
        monkeypatch.setenv(_SENS_ENV, "1e6")
        assert cg.generate_sens_from_model(model, functional=True) is not None


# ─── what a caller sees ────────────────────────────────────────────────────


class TestDecline:
    def test_the_default_budget_changes_nothing(self, tmp_path):
        """The regression guard for every model that already had an analytic sens
        RHS: the corpus derives in 2.6 s total, so nothing real is near the
        budget and emission must be untouched."""
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, SIR))
        assert has_sens is True

    def test_a_functional_model_declines_when_the_budget_expires(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, SIR))
        assert has_sens is False

    def test_an_elementary_model_declines_too(self, tmp_path, monkeypatch):
        """The chain-rule loop is the *only* sympy an all-Elementary model runs on
        this path, and a model with thousands of ``# ConstantExpression`` rate
        constants can hang on it without ever touching a Functional law."""
        model = _model(tmp_path, ELEMENTARY)
        assert cg.generate_sens_from_model(model) is not None
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        assert cg.generate_sens_from_model(model) is None

    def test_the_net_text_path_declines_too(self, tmp_path, monkeypatch):
        """``generate_sens_rhs_c`` reads the .net as text and never sees a model,
        so it is a separate entry point with its own build — and its own
        derived-rate-constant derivation to bound."""
        net = _net(tmp_path, ELEMENTARY)
        assert cg.generate_sens_rhs_c(net) is not None
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        assert cg.generate_sens_rhs_c(net) is None

    def test_the_escape_hatch_restores_the_unbudgeted_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_SENS_ENV, "inf")
        _src, has_sens = cg.generate_combined_from_model(_model(tmp_path, SIR))
        assert has_sens is True

    def test_the_decline_is_loud_and_names_the_override(self, tmp_path, monkeypatch, caplog):
        """A silent decline is a 9-37x slowdown nobody can attribute. It routes
        through the same warning as every other decline reason (#56's precedent),
        says how far the derivation got, and names the knob."""
        monkeypatch.setenv(_SENS_ENV, "1e-9")
        with caplog.at_level(logging.WARNING, logger="bngsim"):
            cg.generate_combined_from_model(_model(tmp_path, SIR))
        msg = "\n".join(r.getMessage() for r in caplog.records)
        assert _SENS_ENV in msg
        assert "∂f/∂p" in msg and "budget" in msg
        assert "difference quotient" in msg


class TestFallbackStillSolves:
    """What a decline actually costs the caller: the columns are still computed,
    by CVODES' internal difference quotient, and they agree with the analytic
    ones.

    Not a claim that the fallback is *cheap*. #55 measured 9-37x per column, and
    on a stiff model at a tight tolerance the DQ's ~sqrt(rtol) accuracy collapses
    the step size outright — the SIR fixture above burns its entire step budget
    and returns zeros, which is why it is not the model used here. That is the
    honest cost of this budget, and the reason the warning names the override:
    declining beats hanging, but it is not free.
    """

    def test_a_declined_model_still_returns_sensitivities(self, tmp_path, monkeypatch):
        analytic = np.asarray(_run(_model(tmp_path, HILL), ["kmax", "Km"]).sensitivities)

        monkeypatch.setenv(_SENS_ENV, "1e-9")
        declined = np.asarray(_run(_model(tmp_path, HILL, "m2.net"), ["kmax", "Km"]).sensitivities)

        assert np.all(np.isfinite(declined))
        assert np.any(declined != 0.0)
        scale = max(float(np.max(np.abs(analytic))), 1e-300)
        np.testing.assert_allclose(declined, analytic, rtol=1e-4, atol=1e-5 * scale)


def _run(model, params, t_end=40.0, n=21):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    return sim.run(t_span=(0.0, t_end), n_points=n)
