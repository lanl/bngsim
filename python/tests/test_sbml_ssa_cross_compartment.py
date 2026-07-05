"""Cross-compartment SBML SSA correctness — Phase 2.5 item (3).

Hand-rolled SBML L3 model with two compartments at V_a=2, V_b=3 and a
mass-action transport reaction A→B. Without the Phase 2.5
``per_species_volume_scaling`` flag, the loader's per-species fallback
emits one BNGsim reaction per affected species with ``stat_factor=±1``,
which is SSA-broken (negative propensity clamps to zero on the consumption
side). With the flag, ``compute_derivs`` divides the rate by each
species's volume_factor at accumulation time and SSA fires correctly.

Three checks:
  1. Loader takes the unified branch (one reaction emitted, not one per
     affected species).
  2. ODE trajectory matches the analytical solution (storage[A] decays
     with rate constant k·c_A/V_a = k, storage[B] grows with rate constant
     k·c_A/V_b = (2/3)·k against the asymptote V_a/V_b·conc[A](0)).
  3. SSA mean(amount[A](t)) over N reps lands within tolerance of the
     analytical mean.

Reference: ADR-003 amendment + ``dev/investigations/sbml_ssa_phase2_5_*.md``.
"""

import math

import bngsim
import numpy as np

# Two compartments (V_a=2, V_b=3); A→B with kineticLaw = c_A * k * A.
# In SBML's amount semantics that's a mass-action propensity 0.05·amount[A].
SBML_CROSS_COMPARTMENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="cross_compartment_transport">
    <listOfCompartments>
      <compartment id="cA" size="2" constant="true" spatialDimensions="3"/>
      <compartment id="cB" size="3" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cA" initialConcentration="100"
               hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cB" initialConcentration="0"
               hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.05" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="transport" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/>
              <ci>cA</ci><ci>k</ci><ci>A</ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

K = 0.05
V_A = 2.0
V_B = 3.0
A0_AMOUNT = 200.0  # initialConcentration=100, V_a=2 → amount=200


def _load_model():
    return bngsim.Model.from_sbml_string(SBML_CROSS_COMPARTMENT)


def test_loader_emits_single_unified_reaction():
    """Cross-compartment reaction → exactly one BNGsim reaction (not split
    per affected species)."""
    model = _load_model()
    assert model.n_reactions == 1, (
        f"Expected unified emission (1 reaction), got {model.n_reactions} — "
        "per-species fallback is SSA-broken and must not fire here."
    )
    rxns = model._core.codegen_data()["reactions"]
    rxn = rxns[0]
    assert rxn["per_species_volume_scaling"] is True
    assert rxn["apply_species_factor"] is False
    # ssa_volume_factor=1.0: kinetic law evaluates to amount/time directly,
    # SSA fire-step's per-species 1/volume_factor handles amount→storage.
    assert rxn["ssa_volume_factor"] == 1.0
    assert sorted(rxn["reactants"]) == [0]  # A consumed once
    assert sorted(rxn["products"]) == [1]  # B produced once


def test_ode_matches_analytical():
    """ODE: dStorage[A]/dt = -k·c_A/V_a · Storage[A] = -k·Storage[A].
    dStorage[B]/dt = +k·c_A/V_b · Storage[A] = +(2/3)·k·Storage[A].

    Closed-form: Storage[A](t) = 100·exp(-k·t),
    Storage[B](t) = (V_a/V_b)·100·(1 - exp(-k·t)).
    """
    model = _load_model()
    sim = bngsim.Simulator(model, method="ode")
    t_end = 10.0
    result = sim.run(t_span=(0.0, t_end), n_points=11)

    times = result.time
    sa = result.species[:, 0]
    sb = result.species[:, 1]

    expected_a = 100.0 * np.exp(-K * times)
    expected_b = (V_A / V_B) * 100.0 * (1.0 - np.exp(-K * times))

    np.testing.assert_allclose(sa, expected_a, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(sb, expected_b, atol=1e-6, rtol=1e-6)

    # Amount conservation (ODE): amount[A] + amount[B] = A0_AMOUNT.
    amounts = sa * V_A + sb * V_B
    np.testing.assert_allclose(amounts, A0_AMOUNT, atol=1e-6, rtol=1e-6)


def test_ssa_mean_matches_analytical():
    """SSA: propensity = kinetic_law = c_A·k·conc[A] = k·amount[A].
    Each fire decrements amount[A] by 1, increments amount[B] by 1.
    Population trajectory is binomial: amount[A](t) ~ Binomial(A0, exp(-k·t)).

    Check sample mean over N reps lands within 5σ_mean of analytical.
    """
    n_reps = 1500
    t_end = 10.0

    # run_batch with no parameter changes: pass identity dicts so each
    # replicate uses base_seed + i.
    model = _load_model()
    sim = bngsim.Simulator(model, method="ssa")
    param_sets = [{} for _ in range(n_reps)]
    results = sim.run_batch(t_span=(0.0, t_end), n_points=11, params=param_sets, seed=20260507)

    # Storage = amount/V_c → amount = storage * V_c.
    a_amounts_at_end = np.array([r.species[-1, 0] * V_A for r in results])

    # Analytical: mean = A0·exp(-k·t), var = A0·p·(1-p), p = exp(-k·t).
    p_end = math.exp(-K * t_end)
    mean_analytical = A0_AMOUNT * p_end
    var_analytical = A0_AMOUNT * p_end * (1.0 - p_end)
    sd_mean = math.sqrt(var_analytical / n_reps)

    mean_empirical = float(np.mean(a_amounts_at_end))
    delta = abs(mean_empirical - mean_analytical)
    assert delta < 5.0 * sd_mean, (
        f"SSA mean(amount[A](t={t_end})) = {mean_empirical:.3f}, "
        f"analytical = {mean_analytical:.3f} (delta = {delta:.3f}, "
        f"5·SE = {5.0 * sd_mean:.3f})."
    )


def test_ssa_amount_conservation():
    """Single SSA replicate: amount[A] + amount[B] stays at A0_AMOUNT
    at every output sample (each fire is -1 on A, +1 on B in amount)."""
    model = _load_model()
    sim = bngsim.Simulator(model, method="ssa")
    result = sim.run(t_span=(0.0, 50.0), n_points=51, seed=20260507)

    amounts = result.species[:, 0] * V_A + result.species[:, 1] * V_B
    np.testing.assert_allclose(amounts, A0_AMOUNT, atol=1e-9)


# ── Multi-compartment hOSU=true V≠1 *Functional* reaction under SSA (GH #75) ──
#
# The headline "still-uncovered surface" named in #75: a non-mass-action
# (Functional) rate law over hasOnlySubstanceUnits=true species in V≠1
# compartments, under SSA. Formerly rejected by the
# ``non_mass_action_volumetric_species`` gate. S in cA (V=2), P in cB (V=3),
# both hOSU; saturating law kf·S/(Km+S) reads amount_S. The propensity must
# read S as an amount (via its same-named observable), so the SSA ensemble mean
# converges to the amount-correct ODE and the integer amount a_S+a_P=200 is
# conserved on every recorded step.
SBML_XCOMP_HOSU_FUNCTIONAL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp_hosu_func">
    <listOfCompartments>
      <compartment id="cA" size="2" constant="true" spatialDimensions="3"/>
      <compartment id="cB" size="3" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cA" initialAmount="200"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cB" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="8.0" constant="true"/>
      <parameter id="Km" value="100.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/>
            <apply><times/><ci>kf</ci><ci>S</ci></apply>
            <apply><plus/><ci>Km</ci><ci>S</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_xcomp_hosu_functional_ssa_matches_ode():
    """Multi-compartment hOSU=true Functional reaction under SSA (GH #75):
    no SSA validation error, integer amount conserved, ensemble mean → ODE."""
    model = bngsim.Model.from_sbml_string(SBML_XCOMP_HOSU_FUNCTIONAL)
    assert not any(
        i.code == "non_mass_action_volumetric_species" for i in bngsim.validate_for_ssa(model)
    )
    t_end, n_pts, vA, vB = 40.0, 21, 2.0, 3.0
    model.reset()
    ode = bngsim.Simulator(model, method="ode").run(t_span=(0, t_end), n_points=n_pts)
    P_ode = np.asarray(ode.species)[:, list(ode.species_names).index("P")]
    assert np.asarray(ode.species)[0, list(ode.species_names).index("S")] == 100.0  # 200/2

    reps = 200
    accP = np.zeros(n_pts)
    for rep in range(reps):
        model.reset()
        rs = bngsim.Simulator(model, method="ssa").run(
            t_span=(0, t_end), n_points=n_pts, seed=3000 + rep
        )
        sn = list(rs.species_names)
        Ss = np.asarray(rs.species)[:, sn.index("S")]
        Ps = np.asarray(rs.species)[:, sn.index("P")]
        np.testing.assert_allclose(Ss * vA + Ps * vB, 200.0, atol=1e-6)  # integer amount
        accP += Ps
    P_ssa = accP / reps
    assert P_ode[-1] > 5.0  # reaction progressed
    np.testing.assert_allclose(P_ssa[-1], P_ode[-1], rtol=0.08)
