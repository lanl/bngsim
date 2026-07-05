"""ODE correctness for AssignmentRule-target species in the SBML loader.

Regression tests for the 2026-05-24 ODE-level divergence investigation
(``dev/notes/SBML_VS_ROADRUNNER.md`` → "ODE-level divergence — root cause").
The #30 BioModels cross-engine screen surfaced a handful of models where
bngsim-ODE diverged from libRoadRunner-ODE. Of the twelve flagged, the
genuine bngsim bugs were *not* the suspected tiny-volume bimolecular
volume-scaling — they were two AssignmentRule-on-species defects:

  1. **Dynamics** — a kinetic law that references an AssignmentRule-target
     species (typically as a modifier) was classified as mass-action and
     read the species's *frozen* ``conc[]`` slot (AR targets are emitted
     ``fixed``), instead of the rule's live value. e.g. BIOMD0000000104's
     ``reaction_1 = k2*species_1*species_3`` with ``species_3`` AR-driven
     never fired, freezing the whole downstream cascade.

  2. **Reporting** — ``Result.species`` reported the same frozen value for
     an AR-target species, even when the model dynamics were correct
     (BIOMD0000000016 ``Pt = P0+P1+P2+Pn``, BIOMD0000000199 ``FeIII_t``).

Both are exercised here against closed-form oracles. A fourth test locks in
that a V≠1 bimolecular mass-action law (the *suspected* but non-existent
volume bug) already reproduces the literal kinetic-law trajectory.
"""

import bngsim
import numpy as np
import pytest

# ── Model 1: AR-target species used in a (mass-action-shaped) rate law ──────
#
# A → ∅ with law kd·A   (ordinary first-order decay; A(t) = A0·e^(-kd·t))
# ∅ → B with law  k·S    where S is an AssignmentRule species, S = A.
#
# So dB/dt = k·S = k·A(t) = k·A0·e^(-kd·t)  ⇒  B(t) = (k·A0/kd)(1 - e^(-kd·t)).
#
# ``∅ → B; k*S`` is mass-action-shaped (rate = const · species), so the
# classifier would emit it Elementary with S folded into the reactant factor,
# reading the frozen conc[S] ≡ S(0). With the fix it falls to the Functional
# path and binds S to the live observable (= A(t)). The discriminator: the bug
# gives linear growth dB/dt = k·A0 (S stuck at A0); the fix gives saturating
# growth. S is also declared with a deliberately wrong initial concentration
# (0, not A0) to prove the AssignmentRule overrides it at t=0 as well.
SBML_AR_MODIFIER = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_modifier">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="kd" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>A</ci></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kd</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="synthB" reversible="false">
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <listOfModifiers>
          <modifierSpeciesReference species="S"/>
        </listOfModifiers>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

A0, K, KD = 10.0, 0.3, 0.5


def test_ar_target_species_feeds_live_value_into_rate_law():
    """B grows as the saturating oracle, proving the rate reads S = A(t),
    not the frozen S(0). The bug produced linear growth dB/dt = k·A0."""
    model = bngsim.Model.from_sbml_string(SBML_AR_MODIFIER)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 10), n_points=21)
    t = np.asarray(r.time)
    names = list(r.species_names)
    A = np.asarray(r.species)[:, names.index("A")]
    B = np.asarray(r.species)[:, names.index("B")]

    expected_A = A0 * np.exp(-KD * t)
    expected_B = (K * A0 / KD) * (1.0 - np.exp(-KD * t))
    np.testing.assert_allclose(A, expected_A, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(B, expected_B, rtol=1e-6, atol=1e-8)

    # Guard against the regression specifically: the buggy linear growth
    # B_lin = k·A0·t overshoots the saturating curve badly by t=10.
    B_buggy_linear = K * A0 * t
    assert B[-1] < 0.5 * B_buggy_linear[-1]


def test_ar_target_species_reported_at_live_value():
    """Result.species[:, S] reports the rule value S = A(t) (incl. the t=0
    override of the declared IC=0), not the frozen ``fixed`` slot."""
    model = bngsim.Model.from_sbml_string(SBML_AR_MODIFIER)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 10), n_points=21)
    names = list(r.species_names)
    assert "S" in names, "AR-target species should still appear in species output"
    A = np.asarray(r.species)[:, names.index("A")]
    S = np.asarray(r.species)[:, names.index("S")]
    np.testing.assert_allclose(S, A, rtol=1e-9, atol=1e-12)
    # The declared IC was 0; the AssignmentRule must override it at t=0 too.
    assert S[0] == A[0] == A0


# ── Model 2: non-linear AssignmentRule reporting ───────────────────────────
#
# A decays (A→∅, law kd·A); S2 = A*A is a *non-linear* AssignmentRule, which
# the loader emits as a function/expression rather than a linear observable.
# Result.species[:, S2] must still report A(t)².
SBML_AR_NONLINEAR = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_nonlinear">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="kd" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="S2">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>A</ci><ci>A</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kd</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="synthB" reversible="false">
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <listOfModifiers>
          <modifierSpeciesReference species="S2"/>
        </listOfModifiers>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S2</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_nonlinear_ar_target_species_reported_at_live_value():
    """A non-linear AssignmentRule (S2 = A²) target reports A(t)² in
    Result.species — exercises the expression (function) report path."""
    model = bngsim.Model.from_sbml_string(SBML_AR_NONLINEAR)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 10), n_points=21)
    names = list(r.species_names)
    A = np.asarray(r.species)[:, names.index("A")]
    S2 = np.asarray(r.species)[:, names.index("S2")]
    np.testing.assert_allclose(S2, A * A, rtol=1e-9, atol=1e-12)


# ── Model 3: V≠1 bimolecular mass-action reproduces the literal law ─────────
#
# The screen *suspected* a tiny-volume bimolecular volume-scaling bug. It does
# not exist: a V≠1 bimolecular mass-action law whose compartment factor is
# written into the law reproduces the literal kinetic-law dynamics. With
# A0=B0 and law = V·k·A·B (so d(amount A)/dt = -V·k·A·B, d[A]/dt = -k·A·B),
# A=B holds and A(t) = A0/(1 + A0·k·t).
SBML_V_BIMOL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="v_bimol">
    <listOfCompartments>
      <compartment id="V" size="2.5e-3" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="V" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="V" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="C" compartment="V" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.02" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="bind" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>V</ci><ci>k</ci><ci>A</ci><ci>B</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

BIMOL_A0, BIMOL_K = 100.0, 0.02


def test_v_neq_1_bimolecular_matches_literal_law():
    """Tiny-V bimolecular mass-action reproduces A(t) = A0/(1 + A0·k·t)."""
    model = bngsim.Model.from_sbml_string(SBML_V_BIMOL)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-14
    )
    t = np.asarray(r.time)
    names = list(r.species_names)
    A = np.asarray(r.species)[:, names.index("A")]
    C = np.asarray(r.species)[:, names.index("C")]
    expected_A = BIMOL_A0 / (1.0 + BIMOL_A0 * BIMOL_K * t)
    np.testing.assert_allclose(A, expected_A, rtol=1e-6, atol=1e-6)
    # Mass balance: A + C = A0 (concentration; single compartment).
    np.testing.assert_allclose(A + C, BIMOL_A0, rtol=1e-6, atol=1e-6)


# ── Model 4: hOSU=true V≠1 bimolecular — the amount-law (latent Phase-2.7) ──
#
# For ``hasOnlySubstanceUnits=true`` species the kinetic-law symbols are
# *amounts*, so the literal law RR integrates is ``d(amount_A)/dt =
# -k·amount_A·amount_B``. With amount_A0=amount_B0=N0 the closed form is
# ``amount_A(t) = N0/(1 + N0·k·t)`` and the engine reports storage
# ``[A] = amount_A / V``. This was the lone remaining ODE-level bngsim-vs-RR
# divergence from the #30 BioModels work (MODEL1102210001, all-hOSU V=1e-12):
# bngsim computed the rate on stored concentrations and accumulated ±rate with
# no compartment factor, draining A ~1/V too fast. The fix restores the amount
# substitution (Π_{i hOSU} V_c(i)^mult) in the mass-action classifier's ``sf``.
# Contrast with Model 3 above (hOSU=false): there the law's symbols are
# concentrations and no amount substitution applies.
SBML_HOSU_V_BIMOL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_v_bimol">
    <listOfCompartments>
      <compartment id="V" size="1e-3" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="V" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="V" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="C" compartment="V" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="0.02" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="bind" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kf</ci><ci>A</ci><ci>B</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

HOSU_V, HOSU_N0, HOSU_KF = 1e-3, 100.0, 0.02


def test_hosu_true_v_neq_1_bimolecular_matches_amount_law():
    """hOSU=true tiny-V bimolecular reports [A] = (N0/(1+N0·kf·t)) / V."""
    model = bngsim.Model.from_sbml_string(SBML_HOSU_V_BIMOL)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-16
    )
    t = np.asarray(r.time)
    names = list(r.species_names)
    A = np.asarray(r.species)[:, names.index("A")]
    C = np.asarray(r.species)[:, names.index("C")]
    # Amount oracle, then convert to the storage/concentration bngsim reports.
    amount_A = HOSU_N0 / (1.0 + HOSU_N0 * HOSU_KF * t)
    expected_A_storage = amount_A / HOSU_V
    np.testing.assert_allclose(A, expected_A_storage, rtol=1e-6, atol=1e-6)
    # Mass balance in storage units: A + C = N0/V.
    np.testing.assert_allclose(A + C, HOSU_N0 / HOSU_V, rtol=1e-6, atol=1e-3)


# ── Model 5: hOSU=true MULTI-compartment cross-compartment reactions ────────
#
# Regression for GH #70 (PyBNF-Private). The v0.9.10 hOSU/V≠1 amount
# substitution above only covered the *single-compartment* Elementary path.
# When a reaction's species span **two compartments with different V_c**, the
# mass-action classifier's ``len(v_factors)>1`` guard rejects it and the
# reaction is emitted as a Functional. Pre-fix, the §9 ``involved_vs`` set
# collapsed hOSU species to V_s=1.0, so an all-hOSU cross-compartment reaction
# saw a single {1.0} factor, took the unified path with ``common_vs=1`` and
# accumulated ±rate with **no** ``/V_c`` — and any hOSU V≠1 species *in the
# law* was read as concentration instead of amount. Surfaced by the rr_parity
# SBML suite on BIOMD0000000019 (Schoeberl EGFR): species in the V=4.3e-6
# compartment ``c3`` diverged from RoadRunner by ~1/V (x13: 1.0 vs 2.36e5).
#
# Two independent mechanisms, each with a closed-form amount oracle:
#
#   5a — unimolecular C(c3) → A(c2), law ``kdeg·C``. C lives in the V≠1
#        compartment and appears in the law, so its symbol must be restored to
#        an amount (×V_c3) AND the per-species accumulation must divide A by
#        V_c2=1 and C by V_c3. This isolates the amount-substitution: pre-fix,
#        A grew 1/V_c3 too fast.
#
#   5b — bimolecular A(c2)+B(c2) → P(c3), law ``kf·A·B``. The reactants share
#        V_c2=1 (no amount restoration needed in the law) but the product P
#        lives in V_c3, so the per-species ``/V_c`` must differ across species.
#        This isolates the cross-compartment per-species volume divide.
SBML_HOSU_XCOMP_UNI = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_xcomp_uni">
    <listOfCompartments>
      <compartment id="c2" size="1.0" constant="true" spatialDimensions="3"/>
      <compartment id="c3" size="2.5e-4" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c2" initialAmount="5"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c3" initialAmount="200"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kdeg" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="move" reversible="false">
        <listOfReactants>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kdeg</ci><ci>C</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

XCOMP_VC3, XCOMP_A0, XCOMP_C0, XCOMP_KDEG = 2.5e-4, 5.0, 200.0, 0.3


def test_hosu_true_cross_compartment_unimolecular_matches_amount_law():
    """C(c3)→A(c2), hOSU: amount_C decays exp; A gains the amounts (GH #70)."""
    model = bngsim.Model.from_sbml_string(SBML_HOSU_XCOMP_UNI)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-16
    )
    t = np.asarray(r.time)
    names = list(r.species_names)
    A = np.asarray(r.species)[:, names.index("A")]
    C = np.asarray(r.species)[:, names.index("C")]
    # Amount oracle (RR's literal MathML reading, C symbol = amount):
    #   d(amount_C)/dt = -kdeg·amount_C ; d(amount_A)/dt = +kdeg·amount_C.
    amount_C = XCOMP_C0 * np.exp(-XCOMP_KDEG * t)
    amount_A = XCOMP_A0 + XCOMP_C0 * (1.0 - np.exp(-XCOMP_KDEG * t))
    # bngsim reports storage = amount / V_c.
    np.testing.assert_allclose(C, amount_C / XCOMP_VC3, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(A, amount_A / 1.0, rtol=1e-6, atol=1e-6)
    # Amount conservation: amount_A + amount_C = A0 + C0.
    np.testing.assert_allclose(A * 1.0 + C * XCOMP_VC3, XCOMP_A0 + XCOMP_C0, rtol=1e-6, atol=1e-6)


SBML_HOSU_XCOMP_BIMOL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_xcomp_bimol">
    <listOfCompartments>
      <compartment id="c2" size="1.0" constant="true" spatialDimensions="3"/>
      <compartment id="c3" size="2.5e-4" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c2" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c2" initialAmount="100"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="c3" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="0.05" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="bind" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kf</ci><ci>A</ci><ci>B</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

XBIMOL_VC3, XBIMOL_N0, XBIMOL_KF = 2.5e-4, 100.0, 0.05


def test_hosu_true_cross_compartment_bimolecular_matches_amount_law():
    """A(c2)+B(c2)→P(c3), hOSU: amount-law a(t)=N0/(1+N0·kf·t), P in V≠1 (GH #70)."""
    model = bngsim.Model.from_sbml_string(SBML_HOSU_XCOMP_BIMOL)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-16
    )
    t = np.asarray(r.time)
    names = list(r.species_names)
    A = np.asarray(r.species)[:, names.index("A")]
    P = np.asarray(r.species)[:, names.index("P")]
    # amount_A = amount_B = a(t) = N0/(1+N0·kf·t); amount_P = N0 - a(t).
    a = XBIMOL_N0 / (1.0 + XBIMOL_N0 * XBIMOL_KF * t)
    amount_P = XBIMOL_N0 - a
    # A in c2 (V=1): stored == amount; P in c3 (V≠1): stored = amount / V_c3.
    np.testing.assert_allclose(A, a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(P, amount_P / XBIMOL_VC3, rtol=1e-6, atol=1e-3)


# ── Model 6: cross-compartment hOSU linear AssignmentRule (the obs path) ─────
#
# Also from GH #70 / BIOMD0000000019: the divergent EGF_EGFR_act species is a
# linear AssignmentRule summing species across compartments
# (``EGF_EGFR_act = x5 + ... + x15 + ...`` with x15 in the V=4.3e-6 c3). Such
# linear rules become BNGsim *observables* (weighted species sums), and the
# weights must restore amounts (×V_c per hOSU summand) and convert back to the
# target's stored units (÷V_c(target)). Pre-fix the observable summed stored
# concentrations, so the c3 terms were inflated ~1/V (EGF_EGFR_act: bngsim
# 1.86e8 vs RR 2.4e4). Here ``total = A + C`` over the same C(c3)→A(c2)
# transport as Model 5a: since the rule reads amounts, total is the conserved
# amount A0+C0 for all t (a clean invariant oracle).
SBML_HOSU_XCOMP_AR = SBML_HOSU_XCOMP_UNI.replace(
    "    </listOfSpecies>",
    """      <species id="total" compartment="c2" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfRules>
      <assignmentRule variable="total">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><ci>A</ci><ci>C</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>""",
).replace('id="hosu_xcomp_uni"', 'id="hosu_xcomp_ar"')


def test_hosu_true_cross_compartment_linear_assignment_rule_reports_amounts():
    """total = A + C over a cross-compartment hOSU sum reads amounts (GH #70)."""
    model = bngsim.Model.from_sbml_string(SBML_HOSU_XCOMP_AR)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-16
    )
    names = list(r.species_names)
    total = np.asarray(r.species)[:, names.index("total")]
    # amount_A + amount_C is conserved = A0 + C0; total (hOSU, c2 V=1) reports
    # that amount sum directly. Pre-fix this summed stored concentrations and
    # blew up by ~1/V_c3.
    np.testing.assert_allclose(total, XCOMP_A0 + XCOMP_C0, rtol=1e-6, atol=1e-6)


# ── Model 7: hOSU=true initialAssignment that references other hOSU species ──
#
# Fix D (2026-05-29 rr_parity triage). The initialAssignment evaluation context
# seeded every species symbol with its *concentration*. For a
# hasOnlySubstanceUnits=true species the MathML symbol denotes the *amount*, so
# an IA that divides a referenced hOSU species by its compartment (the SBML
# idiom for "convert amount→concentration", e.g. BIOMD0000000547
# ``species_14 = (species_11/compartment_3)·… · compartment_3``) read conc/V
# instead of amount/V — off by 1/V, a 1e10 blow-up at tiny V.
#
# Here: c has V=2. X is hOSU with initialAmount=8 ⇒ amount 8, conc 4. Y is hOSU
# with IA ``Y = (X / c) * 3 * c``. Read as the literal MathML wants (X = amount):
# (8/2)·3·2 = 24 amount ⇒ stored conc 24/V = 12 = 3·conc_X. The pre-fix
# concentration seeding gave (4/2)·3·2 = 12 amount ⇒ conc 6. Neither species
# reacts, so the reported t=0 value is exactly the IA result.
SBML_HOSU_IA_AMOUNT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_ia">
    <listOfCompartments>
      <compartment id="c" size="2" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialAmount="8"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="Y" compartment="c"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfInitialAssignments>
      <initialAssignment symbol="Y">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
            <apply><divide/><ci>X</ci><ci>c</ci></apply>
            <cn type="integer">3</cn>
            <ci>c</ci>
          </apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions/>
  </model>
</sbml>"""


def test_hosu_initial_assignment_reads_amount_not_concentration():
    """A hOSU species's IA that references another hOSU species via ``X/c``
    reads the amount (= conc·V), so Y(0) = 3·conc_X = 12, not the conc-seeded 6."""
    model = bngsim.Model.from_sbml_string(SBML_HOSU_IA_AMOUNT)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    names = list(r.species_names)
    X = np.asarray(r.species)[:, names.index("X")]
    Y = np.asarray(r.species)[:, names.index("Y")]
    assert X[0] == 4.0  # amount 8 / V 2
    np.testing.assert_allclose(Y[0], 12.0, rtol=1e-12, atol=1e-12)
    assert abs(Y[0] - 6.0) > 1.0  # guard against the concentration-seeded bug


# ── Model 8: chained linear AssignmentRule (summands are AR-target species) ──
#
# Fix B (the linear-AR observable can only read raw species conc slots, and
# AR-target species are emitted ``fixed`` — frozen at t=0 — so an observable
# total summed stale values). A linear rule whose summands are themselves
# AssignmentRule targets must route to the function path, where the bare names
# resolve to the live AR values. Pattern: MODEL1112260002
# ``Foxo1_all = cytoplasm_Foxo1_tot + nucleus_Foxo1_tot + …``.
#
# A decays (A(t)=A0·e^(-kd·t)); sub1:=A and sub2:=2·A are AR targets; the chained
# linear rule tot := sub1 + sub2 must report 3·A(t). The bug freezes tot at
# 3·A0 (or at the stale ICs). sub1/sub2 carry deliberately wrong ICs (0).
SBML_CHAINED_AR = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="chained_ar">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="sub1" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="sub2" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="tot" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kd" value="0.4" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="sub1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>A</ci></math>
      </assignmentRule>
      <assignmentRule variable="sub2">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn type="integer">2</cn><ci>A</ci></apply>
        </math>
      </assignmentRule>
      <assignmentRule variable="tot">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><ci>sub1</ci><ci>sub2</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kd</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_chained_linear_assignment_rule_reads_live_summands():
    """tot := sub1 + sub2 with sub1:=A, sub2:=2A tracks 3·A(t), not frozen ICs."""
    model = bngsim.Model.from_sbml_string(SBML_CHAINED_AR)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 5), n_points=11)
    t = np.asarray(r.time)
    names = list(r.species_names)
    A = np.asarray(r.species)[:, names.index("A")]
    tot = np.asarray(r.species)[:, names.index("tot")]
    np.testing.assert_allclose(A, 10.0 * np.exp(-0.4 * t), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(tot, 3.0 * A, rtol=1e-6, atol=1e-8)
    # The bug froze tot at 3·A0 = 30 (or at the stale IC 0); by t=5, 3·A ≈ 4.06.
    assert tot[-1] < 10.0


# ── Rate rule on an hOSU=true V≠1 species (GH #75 step-2 regression guard) ──
#
# A rateRule on a hasOnlySubstanceUnits=true species defines d(amount)/dt, and
# its RHS reads that species as an amount. bngsim stores concentration
# (amount/V_c), so d(storage)/dt = (d amount/dt)/V_c(target). Step 2 made the
# species read as an amount everywhere (the same-named-observable shadow) but
# the rate-rule lowering did not divide the accumulation by V_c, so a linear
# decay came out at rate k·V_c instead of k (a factor-of-V error confirmed
# against libRoadRunner). The fix marks the lowered synthesis reaction
# per_species_volume_scaling for an hOSU target. For dR/dt = -k·R with R an
# amount, amount_R(t) = N0·e^{-k t} and the reported storage is amount_R/V_c —
# the SAME exponential rate k regardless of V_c, which is exactly what the
# ×V bug violated. (Cross-checked: this matches RoadRunner's `[R]` column.)
SBML_RATERULE_HOSU = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="raterule_hosu">
    <listOfCompartments>
      <compartment id="V" size="2" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="R" compartment="V" initialAmount="50"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kdeg" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="R">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>kdeg</ci><ci>R</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


def test_rate_rule_on_hosu_v_neq_1_species_matches_amount_law():
    """dR/dt = -kdeg·R on an hOSU=true V=2 species: reported storage decays as
    (N0·e^{-kdeg·t})/V_c — at rate kdeg, NOT kdeg·V_c (the GH #75 step-2 bug)."""
    model = bngsim.Model.from_sbml_string(SBML_RATERULE_HOSU)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 5), n_points=11, rtol=1e-10, atol=1e-14
    )
    t = np.asarray(r.time)
    R = np.asarray(r.species)[:, list(r.species_names).index("R")]
    V, N0, K = 2.0, 50.0, 0.3
    assert R[0] == pytest.approx(N0 / V, rel=1e-9)  # 25 = amount 50 / V_c 2
    expected = (N0 * np.exp(-K * t)) / V  # rate kdeg (V cancels), NOT kdeg·V
    np.testing.assert_allclose(R, expected, rtol=1e-6, atol=1e-7)
    # The bug decayed at kdeg·V = 0.6, so R[-1] would be ~1.245 instead of ~5.578.
    assert R[-1] == pytest.approx((N0 * np.exp(-K * 5.0)) / V, rel=1e-4)
