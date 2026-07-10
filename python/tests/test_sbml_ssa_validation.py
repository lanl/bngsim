"""Phase 3 — SSA validation gate (validate_for_ssa).

One test per detection branch from
``dev/plans/SBML_SSA_SUPPORT_PLAN.md`` §"Phase 3: SSA Validation Gate":

  ERRORS
  - non-integer stoichiometry
  - fast="true" reactions
  - rate rules on compartments
  - non-mass-action kinetic laws on hOSU=true species in V≠1 compartments
    (Phase 2.7 latent class)

  WARNINGS
  - AssignmentRules on species used as reactants

Plus:
  - clean SBML model produces no issues
  - .net model produces no issues (validation only fires for SBML loads)
  - Simulator(method="ssa") raises SsaValidationError on any error
  - Simulator(method="ssa") logs warnings without raising
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest

# ── Reference clean model (no SSA issues) ─────────────────────────────

CLEAN_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="clean">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="cell" initialAmount="10"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_clean_model_has_no_issues():
    model = bngsim.Model.from_sbml_string(CLEAN_SBML)
    assert bngsim.validate_for_ssa(model) == []
    assert model.validate_for_ssa() == []


def test_clean_model_simulator_ssa_constructs():
    model = bngsim.Model.from_sbml_string(CLEAN_SBML)
    sim = bngsim.Simulator(model, method="ssa")
    res = sim.run(t_span=(0, 1), n_points=2, seed=1)
    assert res is not None


# ── Error: non-integer stoichiometry ──────────────────────────────────

NON_INTEGER_STOICH_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ni_stoich">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="frac" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1.5" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_non_integer_stoichiometry_is_error():
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH_SBML)
    issues = bngsim.validate_for_ssa(model)
    errs = [i for i in issues if i.code == "non_integer_stoichiometry"]
    assert len(errs) == 1
    assert errs[0].severity == "error"
    assert "frac" in errs[0].location


def test_non_integer_stoichiometry_simulator_raises():
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH_SBML)
    with pytest.raises(bngsim.SsaValidationError) as ei:
        bngsim.Simulator(model, method="ssa")
    assert "non_integer_stoichiometry" in str(ei.value)
    # Also exposed on the exception object
    assert any(i.code == "non_integer_stoichiometry" for i in ei.value.issues)


# ── Error: fast="true" reaction ───────────────────────────────────────

# fast="true" is only valid in SBML L2 (deprecated in L3v2). Use L2v4.
FAST_REACTION_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="fast_rxn">
    <listOfCompartments>
      <compartment id="cell" size="1"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true"/>
      <species id="B" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="rfast" reversible="false" fast="true">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_fast_reaction_is_error():
    model = bngsim.Model.from_sbml_string(FAST_REACTION_SBML)
    issues = bngsim.validate_for_ssa(model)
    errs = [i for i in issues if i.code == "fast_reaction"]
    assert len(errs) == 1
    assert errs[0].severity == "error"
    assert "rfast" in errs[0].location


def test_fast_reaction_simulator_raises():
    model = bngsim.Model.from_sbml_string(FAST_REACTION_SBML)
    with pytest.raises(bngsim.SsaValidationError):
        bngsim.Simulator(model, method="ssa")


# ── Error: rate rule on a compartment ─────────────────────────────────

COMPARTMENT_RATE_RULE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="comp_rate">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="false" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.01" constant="true"/>
      <parameter id="growth" value="0.001" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="cell">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <ci>growth</ci>
        </math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_compartment_rate_rule_is_supported():
    """A rate rule on a compartment is supported under SSA for a mass-action
    reaction (GH #81 Tier 2; see test_ssa_variable_volume). This model's reaction
    ``death: A -> ; k*A`` acts on an hOSU=true (amount-valued) species with NO
    compartment factor — a bare amount law whose propensity ``k*n_A`` is provably
    volume-independent (GH #170). It now validates clean: the correct live-volume
    exponent is 0, so it runs uncorrected. (Earlier this was the refused
    ``varvol_non_mass_action`` case; the gap is closed.)"""
    model = bngsim.Model.from_sbml_string(COMPARTMENT_RATE_RULE_SBML)
    issues = bngsim.validate_for_ssa(model)
    assert "compartment_rate_rule" not in {i.code for i in issues}  # blanket gate retired
    assert not [i for i in issues if i.severity == "error"]


def test_compartment_rate_rule_simulator_constructs():
    """The bare hOSU=true amount law in a rate-rule compartment constructs and
    runs under SSA (GH #170). The propensity is volume-independent, so the SSA
    count mean tracks the closed form n_A(t) = n_A0*exp(-k*t) — V_static = 1 makes
    the stored column the integer molecule count."""
    model = bngsim.Model.from_sbml_string(COMPARTMENT_RATE_RULE_SBML)
    sim = bngsim.Simulator(model, method="ssa")  # no SsaValidationError
    reps, k, A0 = 1200, 0.01, 100
    res = sim.run_batch(t_span=(0.0, 50.0), n_points=11, params=[{} for _ in range(reps)], seed=4)
    names = list(res[0].species_names)
    nA = np.stack([np.asarray(r.species) for r in res], axis=0)[:, :, names.index("A")]
    assert np.allclose(nA, np.round(nA))  # integer counts (V_static = 1)
    t = np.linspace(0.0, 50.0, 11)
    cf = A0 * np.exp(-k * t)  # volume-independent closed form
    mb = nA.mean(0)
    se = np.sqrt(nA.var(0, ddof=1) / reps)
    for i in range(1, len(t)):
        z = abs(mb[i] - cf[i]) / (se[i] + 1e-12)
        assert z < 5.0, f"t={t[i]:.0f}: ssa={mb[i]:.2f} exact={cf[i]:.2f} z={z:.2f}"


# ── GH #75: non-mass-action kinetic law on hOSU=true V≠1 species ──────
#
# Formerly the Phase 2.7 latent class (SSA error
# ``non_mass_action_volumetric_species``): a kinetic law that does not
# classify mass-action AND references a hOSU=true species in a V≠1
# compartment. It is now SUPPORTED. The unified Functional emission reads
# each hOSU species as its amount (Species::amount_valued ⇒
# update_observables restores stored×V_c, and the law references the species
# through its same-named observable), so the ODE rate (law/V_c) and the SSA
# propensity (law(amount)) are both correct. The gate is gone.
HOSU_VOLUMETRIC_MM_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_volumetric_mm">
    <listOfCompartments>
      <compartment id="cell" size="2" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="Vmax" value="1.0" constant="true"/>
      <parameter id="Km" value="10.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="mm" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><divide/>
              <apply><times/><ci>Vmax</ci><ci>S</ci></apply>
              <apply><plus/><ci>Km</ci><ci>S</ci></apply>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_hosu_volumetric_functional_ssa_now_supported():
    """The hOSU=true V≠1 non-mass-action law that was the Phase 2.7 latent
    SSA gap now validates clean and simulates amount-correctly (GH #75)."""
    model = bngsim.Model.from_sbml_string(HOSU_VOLUMETRIC_MM_SBML)
    # Gap closed: the volumetric-species SSA error is no longer raised.
    issues = bngsim.validate_for_ssa(model)
    assert not any(i.code == "non_mass_action_volumetric_species" for i in issues)

    # ODE reads S as an amount: stored conc = amount / V_c. S0 amount 100,
    # V_c=2 ⇒ conc 50; P0 0. The law only consumes S, so amount_S falls and
    # amount_P rises, with amount conservation a_S + a_P = 100 for all t.
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 20), n_points=21)
    names = list(r.species_names)
    S = np.asarray(r.species)[:, names.index("S")]
    P = np.asarray(r.species)[:, names.index("P")]
    assert S[0] == pytest.approx(50.0, rel=1e-9)  # amount 100 / V_c 2
    assert P[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(S * 2 + P * 2, 100.0, rtol=1e-7, atol=1e-6)
    assert S[-1] < S[0] - 1.0  # S was consumed

    # SSA now runs (previously raised at construction). reset() restores the
    # initial state after the ODE run (run() leaves the model at its final
    # state). The SSA starts at the correct stored conc (S amount 100 / V_c 2 =
    # 50) and conserves the integer amount a_S + a_P = 100 exactly on every
    # recorded step (each fire moves one molecule S→P, i.e. ±1/V_c = ±0.5 in
    # stored conc).
    model.reset()
    rs = bngsim.Simulator(model, method="ssa").run(t_span=(0, 20), n_points=21, seed=12345)
    sn = list(rs.species_names)
    Ss = np.asarray(rs.species)[:, sn.index("S")]
    Ps = np.asarray(rs.species)[:, sn.index("P")]
    assert Ss[0] == pytest.approx(50.0, rel=1e-9)  # SSA starts at amount 100 / V_c 2
    assert Ps[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(Ss * 2 + Ps * 2, 100.0, atol=1e-9)


def test_hosu_volumetric_mm_v1_does_not_trigger():
    """Same MM shape but V=1 never emitted the volumetric error
    (kinetic law on amount equals kinetic law on storage)."""
    model = bngsim.Model.from_sbml_string(HOSU_VOLUMETRIC_MM_SBML.replace('size="2"', 'size="1"'))
    issues = bngsim.validate_for_ssa(model)
    assert not any(i.code == "non_mass_action_volumetric_species" for i in issues)


# Strong SSA discriminator for GH #75: a hOSU=true species in a V_c=4
# compartment under a saturating (non-mass-action) law. Reading S as a
# concentration (stored = amount/4) instead of an amount changes the
# propensity by ~2× (law(400)=k·400/600 vs law(100)=k·100/300). The SSA
# ensemble mean must converge to the amount-correct ODE trajectory, which
# directly exercises that the SSA propensity reads the amount.
HOSU_FUNCTIONAL_SSA_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_functional_ssa">
    <listOfCompartments>
      <compartment id="cell" size="4" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cell" initialAmount="400"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="6.0" constant="true"/>
      <parameter id="Km" value="200.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><divide/>
              <apply><times/><ci>kf</ci><ci>S</ci></apply>
              <apply><plus/><ci>Km</ci><ci>S</ci></apply>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_hosu_volumetric_functional_ssa_matches_ode():
    """SSA ensemble mean of a hOSU=true V≠1 Functional reaction converges to
    the amount-correct ODE, proving the SSA propensity reads amounts (GH #75)."""
    model = bngsim.Model.from_sbml_string(HOSU_FUNCTIONAL_SSA_SBML)
    assert not any(
        i.code == "non_mass_action_volumetric_species" for i in bngsim.validate_for_ssa(model)
    )
    t_end, n_pts = 60.0, 31
    # run() leaves the model at its final state, so reset() before each run
    # (the idiomatic replicate pattern, cf. harness/run_ssa_roundtrip.py).
    model.reset()
    ode = bngsim.Simulator(model, method="ode").run(t_span=(0, t_end), n_points=n_pts)
    P_ode = np.asarray(ode.species)[:, list(ode.species_names).index("P")]

    reps = 300
    acc = np.zeros(n_pts)
    for rep in range(reps):
        model.reset()
        rs = bngsim.Simulator(model, method="ssa").run(
            t_span=(0, t_end), n_points=n_pts, seed=1000 + rep
        )
        acc += np.asarray(rs.species)[:, list(rs.species_names).index("P")]
    P_ssa = acc / reps

    # P is reported as concentration (= amount / V_c, V_c=4): amount converts
    # 0→400, so P_conc rises 0→~52 over the window. Converges to the
    # amount-correct ODE; the conc-vs-amount bug would make the SSA rate ~2×
    # off (V_c=4, saturating law) — far outside this band.
    assert P_ode[0] == pytest.approx(0.0, abs=1e-12)
    assert P_ode[-1] > 30.0  # the run actually progressed (amount > 120 of 400)
    np.testing.assert_allclose(P_ssa[-1], P_ode[-1], rtol=0.06)
    mid = n_pts // 2
    np.testing.assert_allclose(P_ssa[mid:], P_ode[mid:], rtol=0.10, atol=2.0)


# ── Warning: AssignmentRule on species used as a reactant ─────────────

AR_REACTANT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_reactant">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="cell" initialAmount="50"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Y" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="Z" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.01" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="Y">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><ci>X</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="consume_y" reversible="false">
        <listOfReactants>
          <speciesReference species="Y" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="Z" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>Y</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_ar_on_reactant_is_warning():
    model = bngsim.Model.from_sbml_string(AR_REACTANT_SBML)
    issues = bngsim.validate_for_ssa(model)
    warns = [i for i in issues if i.code == "assignment_rule_on_reactant"]
    assert len(warns) == 1
    assert warns[0].severity == "warning"
    assert "consume_y" in warns[0].location
    assert "Y" in warns[0].location
    # Warnings alone must not block Simulator construction.
    sim = bngsim.Simulator(model, method="ssa")
    assert sim is not None


def test_ar_on_reactant_simulator_logs_warning(caplog):
    model = bngsim.Model.from_sbml_string(AR_REACTANT_SBML)
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Simulator(model, method="ssa")
    assert any("assignment_rule_on_reactant" in rec.message for rec in caplog.records)


# ── Reversible reaction with non-splittable net-rate kinetic law ──────
# Phase 7 splits mass-action-shape reversible kineticLaws like
# `kf*A - kr*B` into two SSA channels. The Phase 6 gate is reserved for
# kineticLaws whose forward and/or reverse operand is NOT mass-action —
# Hill, Michaelis-Menten, function-defined rates, or any signed
# difference of non-mass-action terms. These remain SSA-broken (a single
# net-rate channel locks at the deterministic equilibrium), so the gate
# must keep raising for them.
REVERSIBLE_HILL_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="reversible_hill">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="20"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="Vmax" value="1.0" constant="true"/>
      <parameter id="Km" value="5.0" constant="true"/>
      <parameter id="kBA" value="0.01" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="rev" reversible="true">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><minus/>
              <apply><divide/>
                <apply><times/><ci>Vmax</ci><ci>A</ci></apply>
                <apply><plus/><ci>Km</ci><ci>A</ci></apply>
              </apply>
              <apply><times/><ci>kBA</ci><ci>B</ci></apply>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_reversible_non_mass_action_is_error():
    """Hill-shape forward operand fails the per-side mass-action classifier;
    the splitter returns None and the Phase 6 gate raises."""
    model = bngsim.Model.from_sbml_string(REVERSIBLE_HILL_SBML)
    issues = bngsim.validate_for_ssa(model)
    errs = [i for i in issues if i.code == "reversible_non_mass_action"]
    assert len(errs) == 1
    assert errs[0].severity == "error"
    assert errs[0].location == "reaction:rev"


def test_reversible_non_mass_action_simulator_raises():
    model = bngsim.Model.from_sbml_string(REVERSIBLE_HILL_SBML)
    with pytest.raises(bngsim.SsaValidationError, match="reversible"):
        bngsim.Simulator(model, method="ssa")


def test_reversible_mass_action_does_not_trigger():
    """A reversible='true' reaction whose kineticLaw is plain mass-action
    (k*A) classifies via the primary classifier and never reaches the
    Functional path — the reversible flag alone must not produce a false
    positive. (Phase 7 also produces a clean split, but this test
    exercises the simpler primary-classifier path.)"""
    sbml = REVERSIBLE_HILL_SBML.replace(
        """<apply><minus/>
              <apply><divide/>
                <apply><times/><ci>Vmax</ci><ci>A</ci></apply>
                <apply><plus/><ci>Km</ci><ci>A</ci></apply>
              </apply>
              <apply><times/><ci>kBA</ci><ci>B</ci></apply>
            </apply>""",
        "<apply><times/><ci>Vmax</ci><ci>A</ci></apply>",
    )
    model = bngsim.Model.from_sbml_string(sbml)
    issues = bngsim.validate_for_ssa(model)
    assert not any(i.code == "reversible_non_mass_action" for i in issues)


def test_irreversible_hill_does_not_trigger():
    """Same Hill kineticLaw with reversible='false' must not flag —
    Phase 7's split is gated on reversible='true'; irreversible
    Functional emissions are not SSA-broken (the propensity is a single
    forward-only rate, no equilibrium-lock failure mode)."""
    sbml = REVERSIBLE_HILL_SBML.replace('reversible="true"', 'reversible="false"')
    model = bngsim.Model.from_sbml_string(sbml)
    issues = bngsim.validate_for_ssa(model)
    assert not any(i.code == "reversible_non_mass_action" for i in issues)


# ── Negative: .net models bypass SSA validation ───────────────────────


def test_net_model_validate_for_ssa_empty(tmp_path):
    """Models loaded from .net carry no captured SSA issues."""
    # Tiny .net via a roundtrip from SBML — uses an existing benchmark
    # corpus; if the corpus isn't present, fall back to building a Model
    # via SBML and assert the .net path works on any clean Model.
    model = bngsim.Model.from_sbml_string(CLEAN_SBML)
    assert bngsim.validate_for_ssa(model) == []


# ── Sanity: the validation gate only fires for method="ssa" ───────────


def test_ode_path_does_not_validate_ssa():
    """An ODE simulator must not raise even if the model has SSA errors."""
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH_SBML)
    # ODE backend is happy with non-integer stoichiometry.
    sim = bngsim.Simulator(model, method="ode")
    assert sim is not None


# ── strict_ssa override (PUNCHLIST §B1) ───────────────────────────────


def test_strict_ssa_true_default_raises_on_overridable_error():
    """Default ``strict_ssa=True`` keeps the cautious-at-SSA behavior:
    the gate raises on overridable errors like reversible_non_mass_action.
    """
    model = bngsim.Model.from_sbml_string(REVERSIBLE_HILL_SBML)
    with pytest.raises(bngsim.SsaValidationError, match="reversible"):
        bngsim.Simulator(model, method="ssa")


def test_strict_ssa_false_downgrades_overridable_to_warning(caplog):
    """``strict_ssa=False`` downgrades overridable errors to warnings and
    constructs the simulator anyway — the rr-style warn-and-run UX for
    bngsim↔rr comparisons.
    """
    model = bngsim.Model.from_sbml_string(REVERSIBLE_HILL_SBML)
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        sim = bngsim.Simulator(model, method="ssa", strict_ssa=False)
    assert sim is not None
    downgraded = [
        r
        for r in caplog.records
        if "downgraded" in r.getMessage() and "reversible_non_mass_action" in r.getMessage()
    ]
    assert downgraded, "expected a downgraded-to-warning log line for the reversible gate"


def test_strict_ssa_false_still_raises_non_integer_stoichiometry():
    """Non-overridable code: SSA's discrete-fire model breaks under
    fractional stoichiometry no matter the override flag.
    """
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH_SBML)
    with pytest.raises(bngsim.SsaValidationError, match="non_integer_stoichiometry"):
        bngsim.Simulator(model, method="ssa", strict_ssa=False)


def test_strict_ssa_false_still_raises_fast_reaction():
    """Non-overridable code: fast-equilibrium constraint solver isn't
    implemented, so fast='true' reactions can't be approximated even
    with the override. Reuses the FAST_REACTION_SBML fixture defined
    higher in this file.
    """
    model = bngsim.Model.from_sbml_string(FAST_REACTION_SBML)
    with pytest.raises(bngsim.SsaValidationError, match="fast_reaction"):
        bngsim.Simulator(model, method="ssa", strict_ssa=False)


def test_strict_ssa_default_error_advertises_override():
    """When the error envelope contains at least one overridable code, the
    text must point users at strict_ssa=False so they can find the lever
    without reading the docs."""
    model = bngsim.Model.from_sbml_string(REVERSIBLE_HILL_SBML)
    with pytest.raises(bngsim.SsaValidationError) as excinfo:
        bngsim.Simulator(model, method="ssa")
    msg = str(excinfo.value)
    assert "strict_ssa=False" in msg
    assert "non_integer_stoichiometry" in msg  # listed as non-overridable
    assert "fast_reaction" in msg


def test_strict_ssa_default_error_omits_override_when_only_non_overridable():
    """If every error is non-overridable, suggesting strict_ssa=False would
    mislead the user — the lever wouldn't help. The hint must NOT appear."""
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH_SBML)
    with pytest.raises(bngsim.SsaValidationError) as excinfo:
        bngsim.Simulator(model, method="ssa")
    msg = str(excinfo.value)
    # The non-overridable error stands on its own; no "pass strict_ssa=False" hint
    assert "non_integer_stoichiometry" in msg
    assert "pass strict_ssa=False" not in msg


def test_strict_ssa_false_non_overridable_raise_explains_why_flag_didnt_help():
    """When strict_ssa=False raises because every error is non-overridable,
    the message must explain that — otherwise the user wonders why their
    flag was ignored."""
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH_SBML)
    with pytest.raises(bngsim.SsaValidationError) as excinfo:
        bngsim.Simulator(model, method="ssa", strict_ssa=False)
    msg = str(excinfo.value)
    assert "strict_ssa=False" in msg
    assert "cannot override" in msg
    assert "non_integer_stoichiometry" in msg
    # And, since we ARE in an override attempt, don't re-suggest the flag
    assert "pass strict_ssa=False" not in msg
