"""PSA governing-population = min over reactants ∪ products (GH #14).

Pre-#14, bngsim's PSA leap factor iScaling = max(1, ⌊N_min/N_c⌋) took N_min over
*reactant* species only. That had two consequences:

  1. Synthesis reactions (∅ → A) have no reactants, so N_min defaulted to 0 and
     they were NEVER scaled — the source channel dominated the step budget even
     when A was large (the GH #14 report).
  2. A reaction with a huge reactant but a small product was over-scaled: the
     leap was bounded by the (large) reactant, so it dumped a big jump into a
     currently-small product, corrupting that product's trajectory. BioNetGen's
     run_network guards against this by default (rxn_rate_scaled, pScaleChecker=
     true), taking N_min over reactants ∪ products.

The fix takes N_min over the union of reactants and products for every reaction.
For synthesis this means the product population governs: A is scaled once it is
large and runs as exact SSA while it is small — which also intentionally departs
from run_network, whose synthesis path scales by a flat N_c regardless of A.

These tests assert the resulting behavior against exact SSA / analytic means.
"""

import math

import bngsim
import numpy as np


def _birth_death_sbml(k: float, d: float, a0: float) -> str:
    """∅ -> A at rate k; A -> ∅ at rate d*A. Steady state A ~ Poisson(k/d)."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="birth_death">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{a0}"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="{k}" constant="true"/>
      <parameter id="d" value="{d}" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="birth" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <ci>k</ci>
        </math></kineticLaw>
      </reaction>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>d</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# A held constant (boundary) at a large value; A -> B at rate k1*A (large
# propensity, large reactant); B -> ∅ at rate k2*B keeps B small. B ~ Poisson(k1*A/k2).
_SOURCE_SMALL_PRODUCT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="src_small_prod">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10000"
               hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>
      <species id="B" compartment="c" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.1" constant="true"/>
      <parameter id="k2" value="100" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="convert" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k1</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="sink" reversible="false">
        <listOfReactants>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k2</ci><ci>B</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

POPLEVEL = 100.0


def _batch_end(sim, t_end, n_reps, seed, col=0):
    params = [{} for _ in range(n_reps)]
    results = sim.run_batch(t_span=(0.0, t_end), n_points=6, params=params, seed=seed)
    vals = np.array([r.species[-1, col] for r in results])
    steps = np.array([r.solver_stats["n_steps"] for r in results])
    return vals, steps


def test_psa_synthesis_large_product_is_scaled():
    """∅ -> A with a large steady state (A_ss = 1000, ≫ 2·N_c): synthesis is now
    PSA-scaled by the product population. Mean tracks the analytic steady state,
    and the PSA step count is a large fraction below exact SSA (pre-#14 the
    synthesis channel never scaled, so every molecule made cost ~one step)."""
    k, d, a_ss = 1000.0, 1.0, 1000.0
    model = bngsim.Model.from_sbml_string(_birth_death_sbml(k, d, a_ss))
    sim_psa = bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    sim_ssa = bngsim.Simulator(model, method="ssa")

    n_reps = 300
    a_psa, steps_psa = _batch_end(sim_psa, 15.0, n_reps, seed=20260716)
    a_ssa, steps_ssa = _batch_end(sim_ssa, 15.0, n_reps, seed=20260716)

    # Poisson(a_ss): mean a_ss, sd of the batch mean = sqrt(a_ss / n_reps).
    se = math.sqrt(a_ss / n_reps)
    assert abs(a_psa.mean() - a_ss) < 5 * se, (
        f"PSA mean A = {a_psa.mean():.2f}, analytic = {a_ss} (5·SE = {5 * se:.2f})"
    )
    # Sanity: exact SSA agrees too.
    assert abs(a_ssa.mean() - a_ss) < 5 * se

    # Synthesis is now scaled ⇒ far fewer steps than exact SSA. (Observed ≈ 9×.)
    assert steps_psa.mean() < 0.3 * steps_ssa.mean(), (
        f"PSA steps {steps_psa.mean():.0f} not << SSA steps {steps_ssa.mean():.0f}; "
        "synthesis channel appears unscaled."
    )


def test_psa_synthesis_small_product_stays_exact():
    """∅ -> A with a small steady state (A_ss = 10, well below 2·N_c): the product
    population governs, so synthesis is NOT scaled and runs as exact SSA. Mean
    matches exact SSA and the PSA step count is essentially the SSA step count.
    (run_network would instead scale this by a flat N_c = 100, jumping A by ~100
    per fire — the behavior we deliberately do not reproduce.)"""
    k, d, a_ss = 1000.0, 100.0, 10.0
    model = bngsim.Model.from_sbml_string(_birth_death_sbml(k, d, a_ss))
    sim_psa = bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    sim_ssa = bngsim.Simulator(model, method="ssa")

    n_reps = 400
    a_psa, steps_psa = _batch_end(sim_psa, 2.0, n_reps, seed=20260716)
    a_ssa, steps_ssa = _batch_end(sim_ssa, 2.0, n_reps, seed=20260716)

    # Means agree (Poisson(a_ss)); compare as two independent batch means.
    se_diff = math.sqrt(2 * a_ss / n_reps)
    assert abs(a_psa.mean() - a_ssa.mean()) < 5 * se_diff, (
        f"PSA mean A = {a_psa.mean():.2f} vs SSA {a_ssa.mean():.2f} "
        f"(5·SE_diff = {5 * se_diff:.2f})"
    )
    assert abs(a_psa.mean() - a_ss) < 5 * math.sqrt(a_ss / n_reps)

    # No scaling happened ⇒ PSA and SSA take essentially the same number of steps.
    assert steps_psa.mean() > 0.8 * steps_ssa.mean(), (
        f"PSA steps {steps_psa.mean():.0f} materially below SSA {steps_ssa.mean():.0f}; "
        "a small-product synthesis reaction was scaled (should stay exact)."
    )


def test_psa_product_scale_check_prevents_overscaling():
    """A (held large) -> B -> ∅ with B small (B_ss = 10). The reactant A is huge,
    but the product B is small, so the union-min rule refuses to scale A -> B —
    matching run_network's default product-scale check. Pre-#14 (reactant-only)
    would scale by ⌊A/N_c⌋ = 100 and dump B in leaps of 100, wrecking B's mean.
    Assert PSA mean(B) matches exact SSA and no scaling occurs."""
    model = bngsim.Model.from_sbml_string(_SOURCE_SMALL_PRODUCT_SBML)
    sim_psa = bngsim.Simulator(model, method="psa", poplevel=POPLEVEL)
    sim_ssa = bngsim.Simulator(model, method="ssa")

    b_ss = 0.1 * 10000.0 / 100.0  # k1*A/k2 = 10
    n_reps = 300
    b_psa, steps_psa = _batch_end(sim_psa, 2.0, n_reps, seed=20260716, col=1)
    b_ssa, steps_ssa = _batch_end(sim_ssa, 2.0, n_reps, seed=20260716, col=1)

    se_diff = math.sqrt(2 * b_ss / n_reps)
    assert abs(b_psa.mean() - b_ssa.mean()) < 5 * se_diff, (
        f"PSA mean B = {b_psa.mean():.2f} vs SSA {b_ssa.mean():.2f} "
        f"(5·SE_diff = {5 * se_diff:.2f}). Over-scaling would inflate B."
    )
    assert abs(b_psa.mean() - b_ss) < 5 * math.sqrt(b_ss / n_reps), (
        f"PSA mean B = {b_psa.mean():.2f}, analytic = {b_ss}"
    )
    # The large-reactant/small-product channel must NOT be scaled ⇒ steps ≈ SSA.
    assert steps_psa.mean() > 0.8 * steps_ssa.mean(), (
        f"PSA steps {steps_psa.mean():.0f} << SSA {steps_ssa.mean():.0f}; "
        "A -> B was over-scaled despite the small product B."
    )
