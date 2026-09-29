#  Copyright (c) 2026 Los Alamos National Laboratory. All Rights Reserved.
#  SPDX-License-Identifier: MIT
"""GH #239 — a reaction's ``id`` used as a rate symbol in an initialAssignment.

In SBML L3 a reaction's id is a first-class symbol whose value is that reaction's
current rate: the kinetic-law extent, in substance/time (NOT ÷V), analogous to a
species id denoting its amount. An initialAssignment that reads a reaction id must
fold to the reaction's INITIAL rate. bngsim seeds each reaction id into the numeric
context (its kinetic-law value at the initial state) and refolds it inside the IA
convergence loop, so ``p1 = J0`` and ``p1 = addone(J0)`` resolve at load, at the
initial values the other initialAssignments and rules resolve (issue #871). These
cases are initialAssignment-only; a live runtime reaction-rate binding is out of
scope (#239).
"""

from __future__ import annotations

import bngsim
import pytest


def _sbml(model_body: str) -> str:
    """Wrap a ``<model>`` body in a minimal SBML L3V2 document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
        'level="3" version="2">\n'
        f"{model_body}\n"
        "</sbml>\n"
    )


def test_reaction_id_in_initial_assignment():
    """01224: ``p1 = J0`` where J0 (``-> S1``) has kinetic law 1 ⇒ p1 folds to 1.

    The rate is the law value AS WRITTEN (substance/time), not divided by volume.
    """
    body = """
  <model id="case01224">
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>J0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer">1</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(1.0)


def test_reaction_id_through_funcdef_in_initial_assignment():
    """01233: ``p1 = addone(J0)`` with ``addone(x)=x+1`` and J0 rate 1 ⇒ p1 = 2.

    The reaction id resolves inside a function-definition argument — the funcDef
    folds its argument first, so seeding J0 into the context makes it work.
    """
    body = """
  <model id="case01233">
    <listOfFunctionDefinitions>
      <functionDefinition id="addone"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <lambda><bvar><ci>x</ci></bvar>
          <apply><plus/><ci>x</ci><cn type="integer">1</cn></apply>
        </lambda></math></functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><ci>addone</ci><ci>J0</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer">1</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(2.0)


def test_empty_reaction_id_in_initial_assignment():
    """01300: ``p1 = J0`` where J0 is an EMPTY reaction (no reactants/products)
    with kinetic law 3 ⇒ p1 = 3. The rate is still just the law value."""
    body = """
  <model id="case01300">
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>J0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="true">
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer">3</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(3.0)


def test_reaction_id_with_local_parameter():
    """A local parameter shadows the global context when the reaction id folds:
    ``J0`` law ``k`` with a local ``k=7`` ⇒ ``p1 = J0`` resolves to 7, exercising
    the local-parameter branch of the reaction-id seeding."""
    body = """
  <model id="m">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>J0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci>k</ci></math>
          <listOfLocalParameters>
            <localParameter id="k" value="7"/>
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(7.0)


def _rate_read_by_an_ia(ias: str, params: str = "", rules: str = "") -> str:
    """p1 = J0, J0: S -> at k*S, S declared 5, k = 1."""
    m = "http://www.w3.org/1998/Math/MathML"
    return _sbml(f"""
  <model id="m">
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="5"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="false"/>
      <parameter id="p1" value="0" constant="true"/>{params}
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="{m}"><ci>J0</ci></math></initialAssignment>
      {ias}
    </listOfInitialAssignments>
    <listOfRules>{rules}</listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="{m}"><apply><times/><ci>k</ci><ci>S</ci></apply></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>""")


def test_reaction_id_reads_the_resolved_initial_state():
    """The rate is taken at S's initialAssignment, 10, not its declared 5 (issue #871:
    the seed was folded once, before the initialAssignments resolved)."""
    m = "http://www.w3.org/1998/Math/MathML"
    ia = f'<initialAssignment symbol="S"><math xmlns="{m}"><cn>10</cn></math></initialAssignment>'
    model = bngsim.Model.from_sbml_string(_rate_read_by_an_ia(ia))
    assert model._core.get_param("p1") == pytest.approx(10.0)


def test_reaction_id_reads_a_rule_resolved_at_the_initial_state():
    """k := 2*q with q = 3 by initialAssignment: J0 = 6*5 = 30."""
    m = "http://www.w3.org/1998/Math/MathML"
    ia = f'<initialAssignment symbol="q"><math xmlns="{m}"><cn>3</cn></math></initialAssignment>'
    rule = (
        f'<assignmentRule variable="k"><math xmlns="{m}">'
        "<apply><times/><cn>2</cn><ci>q</ci></apply></math></assignmentRule>"
    )
    q = '\n      <parameter id="q" value="1" constant="true"/>'
    model = bngsim.Model.from_sbml_string(_rate_read_by_an_ia(ia, params=q, rules=rule))
    assert model._core.get_param("p1") == pytest.approx(30.0)
