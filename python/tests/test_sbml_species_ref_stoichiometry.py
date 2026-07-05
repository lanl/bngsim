"""GH #237 — named species-reference stoichiometry as a symbol.

An SBML ``<speciesReference>`` may carry an ``id``; that id is a first-class
symbol whose value is the reactant/product stoichiometry and may be read in a
kinetic law / rule (testTag ``SpeciesReferenceInMath``). bngsim bakes the
stoichiometry into the reaction coefficient and otherwise never registered the
id, so a rate law or initial assignment that referenced it failed to compile
(``undefined symbol``). Phase 1 registers each *static* stoich id as a constant
parameter equal to its resolved coefficient, so it resolves everywhere a global
parameter would.

Variable stoichiometry (the id targeted by a real assignment/rate rule) is
Phase 2. A no-MathML assignmentRule is a no-op and should leave the
species-reference id as its declared stoichiometry (GH #243).
"""

import bngsim

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'


def _model(body: str) -> str:
    return f"{_HDR}\n{body}\n</sbml>"


# Reaction 2 S1 -> S2; the S1 reactant carries id "S1_stoich" (= 2). A parameter
# k0 := S1_stoich * 2 reads it in an initial assignment ⇒ k0 == 4. The rate law
# also reads it (S1_stoich * 0.01) — both must resolve the named stoichiometry.
NAMED_STOICH = _model(
    """  <model id="m_named_stoich">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="C" initialAmount="10" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k0" constant="true"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="k0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>S1_stoich</ci><cn type="integer">2</cn></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference id="S1_stoich" species="S1" stoichiometry="2" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="S2" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>S1_stoich</ci><cn>0.01</cn></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)


NO_MATH_ASSIGNMENT_RULE_NAMED_STOICH = _model(
    """  <model id="m_no_math_assignment_rule_named_stoich">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="C" initialAmount="10" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k0" constant="true"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="k0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>S1_stoich</ci><cn type="integer">2</cn></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <assignmentRule variable="S1_stoich"/>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference id="S1_stoich" species="S1" stoichiometry="2" constant="false"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="S2" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>S1_stoich</ci><cn>0.01</cn></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)


def test_named_species_reference_resolves_in_math():
    # Previously a load failure (undefined symbol S1_stoich); now S1_stoich is a
    # constant parameter equal to the reactant stoichiometry (2).
    model = bngsim.Model.from_sbml_string(NAMED_STOICH)
    assert "S1_stoich" in model.param_names
    assert model.get_param("S1_stoich") == 2.0
    # The initial assignment that reads it folds correctly: k0 = 2 * 2 = 4.
    assert model.get_param("k0") == 4.0


def test_named_species_reference_model_simulates():
    # The kinetic law reads the named stoichiometry and must compile + run.
    model = bngsim.Model.from_sbml_string(NAMED_STOICH)
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)
    assert result.n_times == 3


def test_no_math_assignment_rule_targeting_species_reference_id_is_noop():
    # GH #243: a no-MathML assignmentRule targeting a speciesReference id is a
    # no-op. The id must remain bound to the declared stoichiometry, so both an
    # initialAssignment and kineticLaw can read it.
    model = bngsim.Model.from_sbml_string(NO_MATH_ASSIGNMENT_RULE_NAMED_STOICH)
    assert "S1_stoich" in model.param_names
    assert model.get_param("S1_stoich") == 2.0
    assert model.get_param("k0") == 4.0

    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)
    assert result.n_times == 3
