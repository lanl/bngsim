"""SBML ``comp`` hierarchical-composition flattening (GH #230).

bngsim has no native interpreter for the SBML Level 3 ``comp`` package
(ModelDefinition / Submodel / ReplacedElement / Port / SubmodelOutput). Before
GH #230 every composed model failed downstream as ``var_missing`` because
submodel-scoped variables could not be resolved. The loader now detects a
composed document and runs libSBML's ``CompFlatteningConverter`` to inline every
submodel — renaming scoped ids, applying the comp substitutions — so the
existing flat-model pipeline handles the result unchanged (the same path
RoadRunner takes).

These are self-contained models (no dependency on the external SBML Test Suite):
the scoped-rename and id-disambiguation behaviour is exactly what flattening
buys, and is checked against closed-form trajectories.
"""

import numpy as np

import bngsim
from bngsim._sbml_loader import _doc_uses_comp

import libsbml


def _modeldef(defid: str, k: float) -> str:
    """A trivial submodel: one species S obeying dS/dt = k, S(0) = 0."""
    return f"""    <comp:modelDefinition id="{defid}">
      <listOfCompartments>
        <compartment id="c" spatialDimensions="3" size="1" constant="true"/>
      </listOfCompartments>
      <listOfSpecies>
        <species id="S" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"
                 boundaryCondition="false" constant="false"/>
      </listOfSpecies>
      <listOfParameters>
        <parameter id="k" value="{k}" constant="true"/>
      </listOfParameters>
      <listOfRules>
        <rateRule variable="S">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> k </ci></math>
        </rateRule>
      </listOfRules>
    </comp:modelDefinition>"""


def _composed(submodels: str, modeldefs: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"
      xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1" comp:required="true">
  <comp:listOfModelDefinitions>
{modeldefs}
  </comp:listOfModelDefinitions>
  <model id="top">
    <comp:listOfSubmodels>
{submodels}
    </comp:listOfSubmodels>
  </model>
</sbml>"""


def _run(sbml: str, t_end: float = 5.0, n: int = 6):
    model = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, t_end), n_points=n, rtol=1e-10, atol=1e-12
    )
    return model, r


def test_comp_submodel_flattens_and_simulates():
    """A single ``comp`` submodel is inlined; its scoped species is renamed and
    integrates to the closed form S(t) = k·t."""
    sbml = _composed(
        '      <comp:submodel comp:id="sub" comp:modelRef="subdef"/>',
        _modeldef("subdef", 2.0),
    )
    _, r = _run(sbml)
    names = list(r.species_names)
    # Submodel-scoped rename: the bare id ``S`` is namespaced by the submodel id.
    assert names == ["sub__S"], names
    t = np.asarray(r.time)
    S = np.asarray(r.species)[:, 0]
    np.testing.assert_allclose(S, 2.0 * t, rtol=1e-7, atol=1e-9)


def test_comp_colliding_ids_disambiguated():
    """Two submodels that both define a species ``S`` flatten to distinct,
    non-colliding states with independent dynamics — the core thing comp
    flattening buys over a naive merge."""
    sbml = _composed(
        '      <comp:submodel comp:id="a" comp:modelRef="d1"/>\n'
        '      <comp:submodel comp:id="b" comp:modelRef="d2"/>',
        _modeldef("d1", 2.0) + "\n" + _modeldef("d2", 5.0),
    )
    _, r = _run(sbml, t_end=4.0, n=5)
    names = list(r.species_names)
    assert set(names) == {"a__S", "b__S"}, names
    t = np.asarray(r.time)
    sp = np.asarray(r.species)
    np.testing.assert_allclose(sp[:, names.index("a__S")], 2.0 * t, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(sp[:, names.index("b__S")], 5.0 * t, rtol=1e-7, atol=1e-9)


def test_plain_model_is_not_flattened():
    """The flattener is gated: a document that does not compose submodels is
    detected as non-comp and left untouched (no converter run)."""
    plain = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfParameters><parameter id="p" value="1" constant="true"/></listOfParameters>
  </model>
</sbml>"""
    doc = libsbml.readSBMLFromString(plain)
    assert _doc_uses_comp(doc) is False
