"""SBML+PSA support — closes internal#8.

The C++ PSA stepper (``ssa_simulator.cpp``) and the ``poplevel`` validation
in ``_simulator.py`` are model-type-agnostic: SBML/Antimony models flow
through the same dispatch as ``.net``. These tests lock in the user-facing
contract — that ``Simulator(sbml_model, method="psa", poplevel=N_c)`` works
end-to-end and shares the SSA validation gate (``dispatch in ("ssa", "psa")``
at ``_simulator.py:415``).

Companion tests at ``test_ssa_psa_volume.py`` exercise V≠1 PSA correctness;
this file fills the gap on user-API + shared-validation behavior.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

from .test_sbml_ssa_validation import NON_INTEGER_STOICH_SBML, REVERSIBLE_HILL_SBML
from .test_ssa_psa_volume import N0_AMOUNT, POPLEVEL, SBML_PSA_ISOMER_V1, T_END

# ── PSA option validation (model-type-agnostic; tested via SBML) ──────


def test_psa_missing_poplevel_raises():
    """``Simulator(method="psa")`` without ``poplevel`` raises ValueError
    pointing at the Lin/Feng/Hlavacek 2019 reference. PSA-options check
    fires before the SSA validation gate (``_simulator.py:386–393``)."""
    model = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V1)
    with pytest.raises(ValueError, match="poplevel"):
        bngsim.Simulator(model, method="psa")


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0])
def test_psa_poplevel_le_one_raises(bad):
    """``poplevel <= 1`` is rejected with a pointer to method='ssa' for
    exact stochastic sim (``_simulator.py:394–398``)."""
    model = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V1)
    with pytest.raises(ValueError, match="poplevel must be > 1"):
        bngsim.Simulator(model, method="psa", poplevel=bad)


# ── End-to-end smoke ──────────────────────────────────────────────────


def test_psa_smoke_small_sbml():
    """End-to-end smoke: SBML → Model → Simulator(method='psa') → run()
    returns a finite, correctly-shaped trajectory."""
    model = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V1)
    sim = bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    res = sim.run(t_span=(0.0, T_END), n_points=11, seed=20260510)
    assert res.species.shape == (11, 2)  # X, Y
    assert np.all(np.isfinite(res.species))
    # Conservation X+Y=N0_AMOUNT in V=1 storage units (storage==amount).
    totals = res.species.sum(axis=1)
    assert np.allclose(totals, N0_AMOUNT, atol=1e-9)


# ── Shared SSA validation: PSA must reuse the same gate ───────────────


def test_psa_shares_ssa_validation_reversible_non_mass_action():
    """``method="psa"`` flows through the same ``validate_for_ssa`` gate
    as SSA (``_simulator.py:415`` — ``dispatch in ("ssa", "psa")``).
    Reversible Hill kinetics fail the per-side mass-action splitter and
    the gate raises before the C++ simulator is constructed."""
    model = bngsim.Model.from_sbml_string(REVERSIBLE_HILL_SBML)
    with pytest.raises(bngsim.SsaValidationError, match="reversible"):
        bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)


def test_psa_shares_ssa_validation_non_integer_stoich():
    """``non_integer_stoichiometry`` is in NON_OVERRIDABLE_CODES — PSA
    cannot fire fractional ±N steps any more than SSA can. Must raise
    with ``strict_ssa`` at default and at ``False``."""
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH_SBML)
    with pytest.raises(bngsim.SsaValidationError, match="non_integer_stoichiometry"):
        bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    # Non-overridable: strict_ssa=False does not rescue this code.
    with pytest.raises(bngsim.SsaValidationError, match="non_integer_stoichiometry"):
        bngsim.Simulator(model, method="psa", poplevel=POPLEVEL, strict_ssa=False)


# ── Statistical parity: PSA mean tracks analytical (Step 3) ───────────


def test_psa_sbml_matches_analytical():
    """V=1 isomer X↔Y: mean(amount[X](t_end)) at PSA must track the
    analytical relaxation X(t) = N0·(kr + kf·exp(-(kf+kr)·t))/(kf+kr).
    Pattern matches ``test_ssa_psa_volume.py::test_psa_v_eq_1_unchanged``
    (n_reps=200, n_points=11, poplevel=100). 5·SE band."""
    model = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V1)
    sim = bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    n_reps = 200
    results = sim.run_batch(
        t_span=(0.0, T_END),
        n_points=11,
        params=[{} for _ in range(n_reps)],
        seed=20260510,
    )
    x_amounts_at_end = np.array([r.species[-1, 0] for r in results])  # V=1 storage==amount

    # kf=kr=0.01, T_END=50 → X_ss = N0·kr/(kf+kr) = 5000.
    expected_mean = N0_AMOUNT * (0.01 + 0.01 * math.exp(-0.02 * T_END)) / 0.02
    p_x = expected_mean / N0_AMOUNT
    var = N0_AMOUNT * p_x * (1.0 - p_x)
    sd_mean = math.sqrt(var / n_reps)

    mean_empirical = float(np.mean(x_amounts_at_end))
    delta = abs(mean_empirical - expected_mean)
    assert delta < 5.0 * sd_mean, (
        f"PSA-on-SBML mean(amount[X](t={T_END})) = {mean_empirical:.2f}, "
        f"analytical = {expected_mean:.2f} (delta = {delta:.2f}, "
        f"5·SE = {5.0 * sd_mean:.2f})."
    )
