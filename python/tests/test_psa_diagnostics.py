"""PSA partial-scaling diagnostics + cost/precision decision helper (GH #15).

Exercises the full path: C++ dwell-time accumulation in ssa_simulator.cpp →
pybind SsaDiagnostics fields → Result.psa_diagnostics dict → psa_cost_decision.

The reference model is a birth–death process ∅ → A (rate k), A → ∅ (rate d·A),
whose steady state is A ~ Poisson(k/d). With k/d large and N_c ≪ k/d, PSA scales
both channels by m ≈ ⌊(k/d)/N_c⌋, so the diagnostics have predictable values.
"""

import math

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder


def _birth_death(k: float, d: float, a0: float):
    """∅ -> A at rate k;  A -> ∅ at rate d*A.  Steady state A ~ Poisson(k/d)."""
    b = ModelBuilder()
    b.add_parameter("k", k)
    b.add_parameter("d", d)
    a = b.add_species("A", a0)
    b.add_reaction([], [a], "elementary", "k")  # reaction index 1: synthesis
    b.add_reaction([a], [], "elementary", "d")  # reaction index 2: death
    return bngsim.Model(_core=b.build())


def _psa_result(poplevel: float, *, k=2000.0, d=1.0, seed=20260716):
    model = _birth_death(k, d, k / d)
    sim = bngsim.Simulator(model, method="psa", poplevel=poplevel)
    return sim.run((0.0, 10.0), n_points=51, seed=seed)


# ─────────────────────────── diagnostics shape / activation ───────────────────


def test_psa_diagnostics_active_and_shaped():
    p = _psa_result(poplevel=100.0).psa_diagnostics
    assert p["active"] is True
    # 1-based reaction indices, parallel to every per-reaction list.
    assert p["reaction_index"] == [1, 2]
    for key in ("mbar", "qexc", "mbar_integral", "qexc_integral"):
        assert len(p[key]) == 2, key
    assert p["time"] > 0.0
    # Steady state ~2000, N_c=100 ⇒ m ≈ 20 on both channels.
    assert all(m > 5.0 for m in p["mbar"])
    assert all(q > 0.0 for q in p["qexc"])
    assert p["exact_event_integral"] > p["scaled_event_integral"] > 0.0
    assert p["speedup"] > 5.0
    # Scaling engaged: peak population crossed the 2·N_c activation threshold.
    assert p["peak_population"] >= 2 * 100.0
    assert p["activation_crossed"] is True


def test_psa_diagnostics_internal_consistency():
    p = _psa_result(poplevel=100.0).psa_diagnostics
    # ŝ is exactly the ratio of the two event integrals.
    assert math.isclose(
        p["speedup"], p["exact_event_integral"] / p["scaled_event_integral"], rel_tol=1e-12
    )
    # Averages are the integrals divided by the horizon T.
    t = p["time"]
    assert np.allclose(p["mbar"], np.array(p["mbar_integral"]) / t)
    assert np.allclose(p["qexc"], np.array(p["qexc_integral"]) / t)
    # For (near-)homogeneous scaling the measured speedup tracks the mean m̄.
    assert math.isclose(p["speedup"], float(np.mean(p["mbar"])), rel_tol=0.1)


def test_exact_ssa_leaves_psa_diagnostics_inactive():
    model = _birth_death(2000.0, 1.0, 2000.0)
    res = bngsim.Simulator(model, method="ssa").run((0.0, 10.0), n_points=51, seed=1)
    p = res.psa_diagnostics
    assert p["active"] is False
    assert p["reaction_index"] == []
    assert p["mbar"] == [] and p["qexc"] == []
    assert p["peak_population"] == 0.0
    assert p["activation_crossed"] is False
    assert math.isnan(p["speedup"])


def test_psa_with_huge_poplevel_never_scales():
    # N_c far above the population: ⌊N/N_c⌋ = 0 ⇒ m ≡ 1, so the run is pathwise
    # identical to exact SSA and the activation signal stays low.
    p = _psa_result(poplevel=1e9).psa_diagnostics
    assert p["active"] is True  # the run *used* method="psa"...
    assert all(math.isclose(m, 1.0) for m in p["mbar"])  # ...but nothing scaled
    assert all(math.isclose(q, 0.0, abs_tol=1e-9) for q in p["qexc"])
    assert math.isclose(p["speedup"], 1.0, rel_tol=1e-9)
    assert p["activation_crossed"] is False


def test_psa_instrumentation_preserves_mean():
    # Sanity: the accumulation does not perturb the trajectory — PSA mean of A
    # still matches the analytic steady state k/d = 2000.
    model = _birth_death(2000.0, 1.0, 2000.0)
    sim = bngsim.Simulator(model, method="psa", poplevel=100.0)
    a_ends = []
    for s in range(40):
        res = sim.run((0.0, 12.0), n_points=13, seed=1000 + s)
        a_ends.append(float(res.species[-1, 0]))
    mean_a = float(np.mean(a_ends))
    se = math.sqrt(2000.0 / len(a_ends))
    assert abs(mean_a - 2000.0) < 5 * se, f"PSA mean A={mean_a:.1f} vs 2000 (5·SE={5 * se:.1f})"


# ─────────────────────────── decision helper ─────────────────────────────────


def test_cost_decision_mean_rule():
    # Mean: net win iff ŝ > ρ; cost ratio ρ/ŝ.
    d = bngsim.psa_cost_decision(speedup=10.0, rho=4.0, kind="mean")
    assert d["net_win"] is True
    assert math.isclose(d["threshold"], 4.0)
    assert math.isclose(d["cost_ratio"], 0.4)
    assert bngsim.psa_cost_decision(speedup=3.0, rho=4.0, kind="mean")["net_win"] is False
    # Break-even at ŝ = ρ.
    assert math.isclose(
        bngsim.psa_cost_decision(speedup=4.0, rho=4.0, kind="mean")["cost_ratio"], 1.0
    )


def test_cost_decision_variance_rule_gaussian():
    # Gaussian variance: net win iff ŝ > ρ²; cost ratio ρ²/ŝ.
    d = bngsim.psa_cost_decision(speedup=20.0, rho=4.0, kind="variance")
    assert math.isclose(d["threshold"], 16.0)
    assert math.isclose(d["cost_ratio"], 0.8)
    assert d["net_win"] is True
    assert bngsim.psa_cost_decision(speedup=10.0, rho=4.0, kind="variance")["net_win"] is False


def test_cost_decision_variance_non_gaussian_kurtosis():
    # Non-Gaussian: the (κ_s−1)/(κ_0−1) factor multiplies ρ²/ŝ.
    d = bngsim.psa_cost_decision(
        speedup=20.0, rho=4.0, kind="variance", kappa_scaled=5.0, kappa_true=3.0
    )
    # factor = (5-1)/(3-1) = 2 ⇒ threshold = 2·16 = 32.
    assert math.isclose(d["threshold"], 32.0)
    assert math.isclose(d["cost_ratio"], 1.6)
    assert d["net_win"] is False


def test_cost_decision_ideal_homogeneous_matches_notes():
    # docs/notes.tex Cor. "Ideal homogeneous scaling": with v_s ≈ m·v0 and
    # ŝ ≈ m (so ρ = ŝ), mean cost is ≈ unchanged while variance cost ≈ ×m.
    m = 18.0
    mean = bngsim.psa_cost_decision(speedup=m, rho=m, kind="mean")
    var = bngsim.psa_cost_decision(speedup=m, rho=m, kind="variance")
    assert math.isclose(mean["cost_ratio"], 1.0)  # unchanged
    assert math.isclose(var["cost_ratio"], m)  # multiplied by ~m


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speedup": 0.0, "rho": 2.0},
        {"speedup": -1.0, "rho": 2.0},
        {"speedup": 2.0, "rho": 0.0},
        {"speedup": 2.0, "rho": -1.0},
        {"speedup": 2.0, "rho": 2.0, "kind": "bogus"},
        {"speedup": 2.0, "rho": 2.0, "kind": "variance", "kappa_scaled": 1.0},
        {"speedup": 2.0, "rho": 2.0, "kind": "variance", "kappa_true": 0.5},
    ],
)
def test_cost_decision_validation(kwargs):
    with pytest.raises(ValueError):
        bngsim.psa_cost_decision(**kwargs)


def test_cost_decision_rho_is_required():
    # ρ is not derivable from the scaled run alone, so the helper requires it.
    with pytest.raises(TypeError):
        bngsim.psa_cost_decision(10.0)  # type: ignore[call-arg]
