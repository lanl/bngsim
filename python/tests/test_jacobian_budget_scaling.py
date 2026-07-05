"""GH #187 — scale the build-time analytical-Jacobian derivation budget with model
size, and keep the finite-difference (FD) fallback safe at genome scale.

The #95 budget (`test_sbml_jacobian_budget_biomd496.py`) was a *fixed* 20 s
wall-clock cap. That correctly cuts off pathological *small* models, where the FD
Jacobian is a fine fallback, but it inverts at scale: an FD Jacobian needs
~n_species RHS evals per Newton setup, so on a genome-scale model (tens of
thousands of species) it is not a viable solver path. The fixed cap therefore
silently dropped a 74,795-species model (GS-SPARCED, ~33 s derivation) to an
unrunnable FD solve whenever the node finished derivation just over 20 s.

These are pure, machine-independent unit tests of the budget-resolution policy
(`bngsim._jacobian._derivation_budget_s`) — no corpus models or solver required.
They lock in:

  * small models keep the #95 base budget (the losers are <= 295 species);
  * the budget scales up with species count;
  * a genome-scale model gets an *unbounded* budget (FD is non-viable there);
  * an explicit ``BNGSIM_JAC_DERIV_BUDGET_S`` still wins over the size policy.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
from bngsim._jacobian import (
    _BUDGET_PER_SPECIES_S,
    _DEFAULT_DERIVATION_BUDGET_S,
    _FD_COSTLY_SPECIES,
    _FD_NONVIABLE_SPECIES,
    _derivation_budget_s,
)

_ENV = "BNGSIM_JAC_DERIV_BUDGET_S"


@pytest.fixture(autouse=True)
def _clear_budget_env(monkeypatch):
    """Every case here exercises the *size-derived default*, so the env override
    must be absent unless a test sets it explicitly."""
    monkeypatch.delenv(_ENV, raising=False)


# ── the size policy is internally consistent ────────────────────────────────────


def test_thresholds_are_ordered():
    """Both the scaling knee (base / slope) and the costly-warn threshold sit below
    the FD-non-viable gate: the budget must ramp up, and the warn band must be
    non-empty, before the gate makes derivation unbounded (past the gate the budget
    never expires, so it can never warn)."""
    knee = _DEFAULT_DERIVATION_BUDGET_S / _BUDGET_PER_SPECIES_S
    assert knee < _FD_NONVIABLE_SPECIES
    assert _FD_COSTLY_SPECIES < _FD_NONVIABLE_SPECIES


# ── small models keep the #95 base (the losers are <= 295 species) ──────────────


@pytest.mark.parametrize("n_species", [0, 1, 139, 295, 1000])
def test_small_models_pinned_to_base(n_species):
    """Below the scaling knee the budget is exactly the #95 base, so the
    small-but-slow-to-derive losers (BIOMD0000000628: 139 sp, BIOMD0000000496:
    295 sp) are still cut off at 20 s and fall back to FD as #95 intends."""
    assert _derivation_budget_s(n_species=n_species) == _DEFAULT_DERIVATION_BUDGET_S


def test_zero_arg_call_is_base():
    """The default (no model size known) is the base budget — backward compatible
    with the original zero-argument signature."""
    assert _derivation_budget_s() == _DEFAULT_DERIVATION_BUDGET_S


# ── the budget scales up between the knee and the non-viable gate ───────────────


def test_budget_scales_above_the_knee():
    """Past the knee the budget grows linearly with species count."""
    knee = _DEFAULT_DERIVATION_BUDGET_S / _BUDGET_PER_SPECIES_S
    n = int(knee * 3)
    budget = _derivation_budget_s(n_species=n)
    assert budget == pytest.approx(_BUDGET_PER_SPECIES_S * n)
    assert budget > _DEFAULT_DERIVATION_BUDGET_S


def test_budget_is_monotonic_in_species():
    """More species never buys *less* derivation time."""
    samples = [0, 500, 4000, 8000, 15000, _FD_NONVIABLE_SPECIES - 1]
    budgets = [_derivation_budget_s(n_species=n) for n in samples]
    assert all(a <= b for a, b in zip(budgets, budgets[1:]))


def test_scaled_budget_covers_gs_sparced_derivation():
    """At the GS-SPARCED scale (~74.8k species, ~33 s derivation) the budget must
    be unbounded — the fixed 20 s default was the bug. Even just *below* the
    non-viable gate the scaled budget already clears 33 s with margin, so the
    machine-dependent cliff is gone on both sides of the gate."""
    assert _derivation_budget_s(n_species=74795) is None  # unbounded (gated)
    near_gate = _derivation_budget_s(n_species=_FD_NONVIABLE_SPECIES - 1)
    assert near_gate is not None and near_gate > 33.0


# ── the FD-non-viable gate makes the analytical Jacobian mandatory ──────────────


@pytest.mark.parametrize("n_species", [_FD_NONVIABLE_SPECIES, _FD_NONVIABLE_SPECIES + 1, 74795])
def test_genome_scale_budget_is_unbounded(n_species):
    """At/above the gate FD is not a viable solver path, so the budget is unbounded
    (None) — the derivation is allowed to run to completion regardless of how slow
    the machine is."""
    assert _derivation_budget_s(n_species=n_species) is None


# ── an explicit env override still wins over the size policy ─────────────────────


def test_env_absolute_value_overrides_size_policy(monkeypatch):
    """A concrete BNGSIM_JAC_DERIV_BUDGET_S pins the budget regardless of size —
    even on a genome-scale model the user can force a finite cap (and own the
    consequences)."""
    monkeypatch.setenv(_ENV, "5")
    assert _derivation_budget_s(n_species=74795) == 5.0
    assert _derivation_budget_s(n_species=10) == 5.0


@pytest.mark.parametrize("raw", ["inf", "none", "off", "0", "-1", "nan"])
def test_env_disables_budget(monkeypatch, raw):
    """The documented genome-scale workaround (and its aliases) unbound the budget
    on any model."""
    monkeypatch.setenv(_ENV, raw)
    assert _derivation_budget_s(n_species=500) is None


def test_malformed_env_falls_back_to_size_policy(monkeypatch):
    """A junk value is ignored and the size-derived default applies — so a typo
    cannot silently re-impose the fixed cap on a genome-scale model."""
    monkeypatch.setenv(_ENV, "not-a-number")
    assert _derivation_budget_s(n_species=500) == _DEFAULT_DERIVATION_BUDGET_S
    assert _derivation_budget_s(n_species=_FD_NONVIABLE_SPECIES) is None


def test_env_finite_override_is_respected_exactly(monkeypatch):
    """A finite override is returned verbatim, including fractional seconds."""
    monkeypatch.setenv(_ENV, "12.5")
    assert _derivation_budget_s(n_species=500) == 12.5


def test_base_still_covers_needs_analytical_floor():
    """The base must clear the slowest needs-analytical derivation (BIOMD0000000457,
    ~12 s) with margin — the #95 invariant the scaling must not erode."""
    assert _DEFAULT_DERIVATION_BUDGET_S >= 15.0
    assert not math.isinf(_DEFAULT_DERIVATION_BUDGET_S)


# ── the loud-fallback path is wired end-to-end (GH #187 option 3) ───────────────

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"


def test_large_model_budget_expiry_warns_with_workaround(monkeypatch, caplog):
    """When the budget expires on a model large enough that FD is costly, the
    fallback is logged at WARNING and names the exact workaround.

    Driven end-to-end through ``attach_functional_jacobian`` on a real Functional
    model (BIOMD0000000496, 295 species): a sub-microsecond env budget forces the
    derivation to bail, and lowering the costly threshold below 295 puts the model
    in the loud band. The unit tests above already cover the size *policy*; this
    pins the message wiring (logger, level, env-var hint).
    """
    import bngsim

    xmls = sorted((_MODELS_DIR / "BIOMD0000000496").glob("*.xml"))
    if not xmls:
        pytest.skip(f"rr_parity corpus model not present: {_MODELS_DIR / 'BIOMD0000000496'}")

    # Force an immediate budget expiry, and make 295 species count as "FD-costly".
    monkeypatch.setenv(_ENV, "1e-9")
    monkeypatch.setattr("bngsim._jacobian._FD_COSTLY_SPECIES", 100)

    model = bngsim.Model.from_sbml(str(xmls[0]))
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        attached = model.prepare_analytical_jacobian()

    assert attached is False, "budget should have expired and dropped the analytical Jacobian"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "large-model budget expiry did not emit a WARNING"
    msg = warnings[-1].getMessage()
    assert "BNGSIM_JAC_DERIV_BUDGET_S=inf" in msg
    assert "finite-difference" in msg
