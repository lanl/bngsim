"""AMICI's amount states are compared as concentrations.

AMICI keeps a species with ``hasOnlySubstanceUnits="true"`` as an amount in
``rdata.x``; bngsim and RoadRunner (``[id]``) report every species as a
concentration. ``amici_ode`` compared the two as they came, so such a species in
a compartment whose size is not 1 read as a DIFF off by exactly that size: 12 of
the 26 AMICI DIFFs in the BioModels coverage sweep (BIOMD0000000197, 305, 407,
MODEL1102210001, ...) were this alone. ``amount_states_as_concentrations`` divides
each amount state by its compartment's size: constant, integrated by AMICI (a
rate rule), or set by an assignment rule.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libsbml")

import _amici_common as ac  # noqa: E402

MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'
TIME = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>'

# A: amount species in a constant compartment of size 2, decaying at rate k.
#    concentration = 2 exp(-k t)
# B: concentration species in the same compartment: left as it is, exp(-k t).
# C: amount species (constant amount 3) in a compartment with dV/dt = 1, V(0) = 1.
#    concentration = 3 / (1 + t)
# D: amount species (constant amount 5) in a compartment W = 2 + time.
#    concentration = 5 / (2 + t)
# E: amount species (constant amount 6) in a constant compartment declared with
#    size 1 and sized by an initial assignment, cE = 4 k = 2. concentration = 3
SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="amount_species">
    <listOfCompartments>
      <compartment id="c" size="2" constant="true" spatialDimensions="3"/>
      <compartment id="V" size="1" constant="false" spatialDimensions="3"/>
      <compartment id="W" size="2" constant="false" spatialDimensions="3"/>
      <compartment id="cE" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="4" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="C" compartment="V" initialAmount="3" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="D" compartment="W" initialAmount="5" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="E" compartment="cE" initialAmount="6" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="cE"><math {MATH}><apply><times/><cn> 4 </cn><ci> k </ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="V"><math {MATH}><cn> 1 </cn></math></rateRule>
      <assignmentRule variable="W"><math {MATH}><apply><plus/><cn> 2 </cn>{TIME}</apply></math></assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="decayA" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math {MATH}><apply><times/><ci> k </ci><ci> A </ci></apply></math></kineticLaw>
      </reaction>
      <reaction id="decayB" reversible="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math {MATH}><apply><times/><ci> c </ci><ci> k </ci><ci> B </ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

GRID = (0.0, 4.0, 9)


def test_only_amount_species_are_listed():
    assert ac.amount_species_compartments(SBML) == {"A": "c", "C": "V", "D": "W", "E": "cE"}


def test_a_compartment_sized_by_an_initial_assignment_is_evaluated():
    import libsbml

    sbml = libsbml.readSBMLFromString(SBML).getModel()
    assert ac._initial_assignment_value("cE", sbml, {"k": 0.5}.get) == 2.0
    # Not a guess: an id it cannot value gives None.
    assert ac._initial_assignment_value("cE", sbml, {}.get) is None
    assert ac._initial_assignment_value("c", sbml, {"k": 0.5}.get) is None


def test_amount_states_are_divided_by_their_compartments_size():
    t = np.linspace(*GRID)
    x = np.column_stack(
        [4.0 * np.ones_like(t), np.ones_like(t), 3.0 * np.ones_like(t), 7.0 * np.ones_like(t)]
    )
    names = ["A", "B", "C", "V"]
    sizes = {"c": 2.0, "V": 1.0 + t}
    out, kept, dropped = ac.amount_states_as_concentrations(x, names, SBML, sizes.get)
    assert kept == names and dropped == []
    np.testing.assert_array_equal(out[:, 0], 2.0)  # A: amount 4 in size 2
    np.testing.assert_array_equal(out[:, 1], 1.0)  # B: a concentration, untouched
    np.testing.assert_array_equal(out[:, 2], 3.0 / (1.0 + t))  # C: per-time size
    np.testing.assert_array_equal(out[:, 3], 7.0)  # V: a compartment, not a species
    np.testing.assert_array_equal(x[:, 0], 4.0)  # the input is not modified


def test_an_amount_species_with_no_known_size_is_dropped_not_compared():
    x = np.ones((3, 2))
    out, kept, dropped = ac.amount_states_as_concentrations(x, ["A", "B"], SBML, lambda cid: None)
    assert kept == ["B"] and dropped == ["A"] and out.shape == (3, 1)


def test_amicis_renamed_ids_are_matched_to_their_sbml_species():
    sbml = (
        SBML.replace('id="A"', 'id="x"')
        .replace("<ci> A </ci>", "<ci> x </ci>")
        .replace('species="A"', 'species="x"')
    )
    out, kept, _ = ac.amount_states_as_concentrations(
        np.full((2, 1), 4.0), ["amici_x"], sbml, {"c": 2.0}.get
    )
    assert kept == ["amici_x"]
    np.testing.assert_array_equal(out[:, 0], 2.0)


def test_amici_ode_reports_every_species_as_a_concentration():
    """End to end through AMICI: fails on the adapter that compared amounts."""
    pytest.importorskip("amici")
    t, x, names, timing = ac.amici_ode(SBML, *GRID, 1e-10, 1e-12)
    col = {n: x[:, i] for i, n in enumerate(ac.normalize_amici_names(list(names)))}
    k = 0.5
    np.testing.assert_allclose(col["A"], 2.0 * np.exp(-k * t), rtol=1e-6)
    np.testing.assert_allclose(col["B"], np.exp(-k * t), rtol=1e-6)
    np.testing.assert_allclose(col["C"], 3.0 / (1.0 + t), rtol=1e-6)
    np.testing.assert_allclose(col["D"], 5.0 / (2.0 + t), rtol=1e-6)
    np.testing.assert_allclose(col["E"], 3.0, rtol=1e-12)
    assert timing["amount_states_dropped"] == []
