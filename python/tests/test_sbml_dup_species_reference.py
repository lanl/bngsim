"""Duplicate / signed ``<speciesReference>`` entries to the same species.

A reaction may list the same species in several ``<speciesReference>`` entries —
on the same side and/or on both sides — each carrying a SIGNED stoichiometry.
SBML L3 §4.11.3 says the net stoichiometric coefficient for that species is the
sum of the signed product references minus the sum of the signed reactant
references. The engine must therefore SUM the references, not treat each one
independently.

Regression: the §9 Functional emission used to build the reactant/product
multisets by extending them once per reference, which (a) silently dropped a
negative coefficient (``[idx] * int(-1) == []``) and (b) never aggregated
duplicates. A reactant listed ``+1`` and ``-1`` (net 0) was emitted as a net
``-1`` decay, etc. These are SBML Test Suite cases 01422 / 01426 / 01427 /
01432 / 01433 ("Multiple species references to the same species"). The fix
derives the multisets from the aggregated ``net`` dict.
"""

import numpy as np

import bngsim

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'


def _model(reactants: str, products: str, law: str) -> str:
    return f"""{_HDR}
  <model id="m_dup_ref">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="J0" reversible="true">
        <listOfReactants>{reactants}</listOfReactants>
        <listOfProducts>{products}</listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">{law}</math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _ref(species: str, stoich) -> str:
    return f'<speciesReference species="{species}" stoichiometry="{stoich}" constant="true"/>'


def _traj(res, sid):
    return np.asarray(res.species)[:, list(res.species_names).index(sid)]


def test_same_side_signed_refs_cancel():
    # Case 01422: A appears twice as a reactant, +1 and -1, with a constant law.
    # The signed references sum to net 0, so A must stay constant (the old code
    # dropped the -1 ref and emitted a net -1 decay).
    xml = _model(_ref("A", 1) + _ref("A", -1), "", "<cn type='integer'>-1</cn>")
    res = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        t_span=(0, 5), n_points=6
    )
    A = _traj(res, "A")
    assert np.allclose(A, 10.0), A


def test_duplicate_refs_sum_to_net():
    # Case 01433: A listed three times as reactant (+1, +2, -4 ⇒ reactant net -1)
    # and four times as product (+5, -2, +1, -1 ⇒ product net +3). Net for A is
    # -(-1) + 3 = +4. With law = 1, dA/dt = +4, so A(t) = 10 + 4t.
    reactants = _ref("A", 1) + _ref("A", 2) + _ref("A", -4)
    products = _ref("A", 5) + _ref("A", -2) + _ref("A", 1) + _ref("A", -1)
    xml = _model(reactants, products, "<cn type='integer'>1</cn>")
    res = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        t_span=(0, 5), n_points=6
    )
    A = _traj(res, "A")
    assert np.allclose(A, 10.0 + 4.0 * np.asarray(res.time)), A


def test_both_sides_signed_net_two_species():
    # Case 01432: A reactant +2 / product -1 ⇒ net -3; B reactant -1 / product
    # +2 ⇒ net +3. With law = 1, dA/dt = -3 and dB/dt = +3 (the old code dropped
    # both negative references, giving the wrong net for each species).
    reactants = _ref("A", 2) + _ref("B", -1)
    products = _ref("A", -1) + _ref("B", 2)
    xml = _model(reactants, products, "<cn type='integer'>1</cn>")
    res = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        t_span=(0, 2), n_points=5
    )
    t = np.asarray(res.time)
    assert np.allclose(_traj(res, "A"), 10.0 - 3.0 * t), _traj(res, "A")
    assert np.allclose(_traj(res, "B"), 10.0 + 3.0 * t), _traj(res, "B")


def test_plain_reaction_unchanged():
    # Sanity guard for the common one-reference-per-side path: a plain A -> B
    # with unit stoichiometry and a constant law must still move A down and B up
    # by exactly the rate (the net-based multiset rebuild is equivalent here).
    xml = _model(_ref("A", 1), _ref("B", 1), "<cn type='integer'>2</cn>")
    res = bngsim.Simulator(bngsim.Model.from_sbml_string(xml), method="ode").run(
        t_span=(0, 2), n_points=5
    )
    t = np.asarray(res.time)
    assert np.allclose(_traj(res, "A"), 10.0 - 2.0 * t), _traj(res, "A")
    assert np.allclose(_traj(res, "B"), 10.0 + 2.0 * t), _traj(res, "B")
