"""PSA + V≠1 floor-ratio correctness — Phase 2.5 item (1).

PSA computes a per-reaction scaling factor λ_r = 1 / max(1, ⌊N_min^r / N_c⌋)
where N_min^r is the minimum reactant population. Pre-Phase-2.5,
``ssa_simulator.cpp`` read ``conc[si]`` directly, which is in storage units
(= amount/V_c) for SBML loaders with V≠1. With Phase 2's per-species
``volume_factor``, we now multiply by it inside both PSA loops to recover
population. ``.net`` models are unaffected (volume_factor defaults to 1.0).

Discrimination strategy: an isomerization X↔Y with conservation X+Y=N0
prevents depletion, so PSA's λ is well-defined throughout the run and
n_steps variance is moderate. With matched populations between a V=1
model (amount=storage=N0) and a V=2 model (amount=N0, storage=N0/2),
post-fix should produce *identical* PSA trajectories at matched seeds —
the propensities are the same in amount/time and the per-fire amount
step is the same. Pre-fix would compute n_min in storage units on V=2,
giving ~half the aggregation factor and visibly different trajectories.
"""

import math

import bngsim
import numpy as np

# Isomerization X↔Y, conservation X+Y=10000. Single compartment V_c=2,
# initialConcentration[X]=5000 → amount=10000, initialConcentration[Y]=0.
SBML_PSA_ISOMER_V2 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="psa_isomer">
    <listOfCompartments>
      <compartment id="c" size="2" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="5000"
               hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="Y" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="0.01" constant="true"/>
      <parameter id="kr" value="0.01" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="fwd" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Y" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>c</ci><ci>kf</ci><ci>X</ci></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="rev" reversible="false">
        <listOfReactants>
          <speciesReference species="Y" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>c</ci><ci>kr</ci><ci>Y</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# V=1 mirror: same population N0=10000 lives directly in storage units.
SBML_PSA_ISOMER_V1 = SBML_PSA_ISOMER_V2.replace('size="2"', 'size="1"').replace(
    'initialConcentration="5000"', 'initialConcentration="10000"'
)

N0_AMOUNT = 10000.0
POPLEVEL = 100.0
T_END = 50.0


def _amounts_storage_to_amount(result, vc: float) -> np.ndarray:
    """Convert storage trajectory to amount trajectory."""
    return result.species * vc


def test_psa_v_neq_1_matches_v_eq_1_at_matched_seeds():
    """Matched-seed isomer trajectories must agree on amount-space when
    V=1 and V=2 represent the same population N0=10000.

    Post-fix: both runs see propensity in amount/time, n_min in
    population, λ identical → PSA selects the same reactions, fires the
    same amount steps, lands on identical amount trajectories.

    Pre-fix V=2 reads n_min from storage (=5000 vs. true 10000), giving
    λ_pre = 1/50 (vs. λ_post = 1/100). The pre-fix V=2 trajectory
    diverges from V=1 within a handful of events.
    """
    n_reps = 30
    seeds = np.arange(20260507, 20260507 + n_reps)

    model_v1 = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V1)
    sim_v1 = bngsim.Simulator(model_v1, method="psa", poplevel=POPLEVEL)
    model_v2 = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V2)
    sim_v2 = bngsim.Simulator(model_v2, method="psa", poplevel=POPLEVEL)

    # Compare end-of-run amount[X] across matched seeds.
    end_amounts_v1 = []
    end_amounts_v2 = []
    n_steps_v1 = []
    n_steps_v2 = []
    for s in seeds:
        r1 = sim_v1.run(t_span=(0.0, T_END), n_points=11, seed=int(s))
        r2 = sim_v2.run(t_span=(0.0, T_END), n_points=11, seed=int(s))
        end_amounts_v1.append(r1.species[-1, 0] * 1.0)  # V=1 storage == amount
        end_amounts_v2.append(r2.species[-1, 0] * 2.0)  # V=2 storage * V_c
        n_steps_v1.append(r1.solver_stats["n_steps"])
        n_steps_v2.append(r2.solver_stats["n_steps"])

    end_amounts_v1 = np.array(end_amounts_v1)
    end_amounts_v2 = np.array(end_amounts_v2)
    n_steps_v1 = np.array(n_steps_v1)
    n_steps_v2 = np.array(n_steps_v2)

    # The key invariant: per-rep mean(|delta|) should be small relative to
    # the RMS amount at steady state (~5000). Pre-fix's per-rep delta is
    # routinely 500+ at t=T_END.
    abs_delta = np.abs(end_amounts_v1 - end_amounts_v2)
    rms_population_at_ss = math.sqrt(0.25 * N0_AMOUNT)  # binomial sd ≈ √(N·p·q)
    assert np.mean(abs_delta) < 0.5 * rms_population_at_ss, (
        f"mean |amount_v1 − amount_v2| at t=T_END = {np.mean(abs_delta):.2f}; "
        f"rms population sd ≈ {rms_population_at_ss:.2f}. "
        "Wide divergence → V=2 likely on pre-fix path (n_min in storage units). "
        f"V=1 amounts: {end_amounts_v1[:5]}, V=2 amounts: {end_amounts_v2[:5]}."
    )

    # Stronger check: mean(n_steps) should match within a tight band.
    # Pre-fix V=2 has ~2× the events of V=1 because λ_pre = 2·λ_post.
    delta_steps = abs(np.mean(n_steps_v1) - np.mean(n_steps_v2))
    se_steps = math.sqrt(
        np.std(n_steps_v1, ddof=1) ** 2 / n_reps + np.std(n_steps_v2, ddof=1) ** 2 / n_reps
    )
    assert delta_steps < 3.0 * se_steps, (
        f"mean n_steps_v1 = {np.mean(n_steps_v1):.1f}, "
        f"mean n_steps_v2 = {np.mean(n_steps_v2):.1f}; "
        f"|delta| = {delta_steps:.1f}, 3·SE = {3.0 * se_steps:.1f}. "
        "Wide gap → V=2 likely on pre-fix path."
    )


def test_psa_v_neq_1_mean_matches_analytical():
    """PSA + V_c=2 isomer: mean(amount[X](t_end)) tracks the analytical
    relaxation to steady state X_ss = N0·kr/(kf+kr) = 5000.
    """
    model = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V2)
    sim = bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    n_reps = 200
    param_sets = [{} for _ in range(n_reps)]
    results = sim.run_batch(t_span=(0.0, T_END), n_points=11, params=param_sets, seed=20260507)
    x_amounts_at_end = np.array([r.species[-1, 0] * 2.0 for r in results])

    # X(t) = N0 · (kr + kf · exp(-(kf+kr)·t)) / (kf+kr)
    # At t=50, kf=kr=0.01: X(50) = 10000·(0.01 + 0.01·exp(-1)) / 0.02
    #                            = 5000·(1 + exp(-1)) ≈ 6839.4.
    expected_mean = N0_AMOUNT * (0.01 + 0.01 * math.exp(-0.02 * T_END)) / 0.02
    # Variance of binomial-like split: N·p·(1-p) at p ≈ 0.684.
    p_x = expected_mean / N0_AMOUNT
    var = N0_AMOUNT * p_x * (1.0 - p_x)
    sd_mean = math.sqrt(var / n_reps)

    mean_empirical = float(np.mean(x_amounts_at_end))
    delta = abs(mean_empirical - expected_mean)
    assert delta < 5.0 * sd_mean, (
        f"PSA mean(amount[X](t={T_END})) = {mean_empirical:.2f}, "
        f"analytical = {expected_mean:.2f} (delta = {delta:.2f}, "
        f"5·SE = {5.0 * sd_mean:.2f})."
    )


def test_psa_v_eq_1_unchanged():
    """Smoke test: V=1 PSA still tracks the analytical isomer steady
    state. .net models share this code path with volume_factor=1.0
    default; the multiply-by-volume_factor in n_min is a no-op (×1) for
    V=1, so trajectory + event counts must be unchanged.
    """
    model = bngsim.Model.from_sbml_string(SBML_PSA_ISOMER_V1)
    sim = bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    n_reps = 200
    param_sets = [{} for _ in range(n_reps)]
    results = sim.run_batch(t_span=(0.0, T_END), n_points=11, params=param_sets, seed=20260507)
    x_amounts_at_end = np.array([r.species[-1, 0] for r in results])  # storage=amount

    expected_mean = N0_AMOUNT * (0.01 + 0.01 * math.exp(-0.02 * T_END)) / 0.02
    p_x = expected_mean / N0_AMOUNT
    var = N0_AMOUNT * p_x * (1.0 - p_x)
    sd_mean = math.sqrt(var / n_reps)

    mean_empirical = float(np.mean(x_amounts_at_end))
    assert abs(mean_empirical - expected_mean) < 5.0 * sd_mean, (
        f"V=1 PSA mean(amount[X](t={T_END})) = {mean_empirical:.2f}, "
        f"analytical = {expected_mean:.2f}, 5·SE = {5.0 * sd_mean:.2f}."
    )
