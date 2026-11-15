"""A document whose root is in no SBML namespace is refused, not loaded empty (issue #1712).

BioNetGen's XML, the file BNG2.pl's ``writeXML`` writes for NFsim and
RuleMonkey, has an ``<sbml>`` root in ``http://www.sbml.org/sbml/level3``, which
is no SBML namespace. libSBML reports error 20101 at severity Error and reads no
model from it. bngsim's loader refused only fatal errors, so ``Model.load`` on
such a file gave ``Model(species=0, reactions=0, observables=0, parameters=0)``,
which ran and returned nothing, with no error and no warning. Now it is refused,
and BioNetGen's XML is named for what it is, with the routes that run it.

A census of the 4,359 XML files in the repository and the BioModels corpus found
125 that give error 20101, every one BioNetGen XML: no SBML document is refused
that loaded before.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import bngsim
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

# Not a __file__ walk-up: run_tests.sh copies the suite elsewhere (issue #578).
DATA = (bngsim_source_root() or Path(__file__).resolve().parents[2]) / "tests" / "data"

# The issue's own file.
BNG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Created by BioNetGen 2.9.3  -->
<sbml xmlns="http://www.sbml.org/sbml/level3" level="3" version="1">
  <model id="model">
    <ListOfParameters>
      <Parameter id="k" type="Constant" value="1" expr="1"/>
    </ListOfParameters>
    <ListOfMoleculeTypes>
      <MoleculeType id="A"/>
    </ListOfMoleculeTypes>
    <ListOfSpecies>
      <Species id="S1" concentration="100" name="A()">
        <ListOfMolecules>
          <Molecule id="S1_M1" name="A"/>
        </ListOfMolecules>
      </Species>
    </ListOfSpecies>
    <ListOfReactionRules>
    </ListOfReactionRules>
    <ListOfObservables>
    </ListOfObservables>
  </model>
</sbml>
"""

SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
  </model>
</sbml>
"""


@pytest.fixture
def bng_xml(tmp_path):
    path = tmp_path / "model.xml"
    path.write_text(BNG_XML, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "load",
    [bngsim.Model.load, bngsim.Model.from_sbml],
    ids=["Model.load", "Model.from_sbml"],
)
def test_the_issues_file_is_refused_and_named(bng_xml, load):
    # Main loaded Model(species=0, reactions=0, observables=0, parameters=0).
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML") as raised:
        load(str(bng_xml))
    message = str(raised.value)
    assert "http://www.sbml.org/sbml/level3" in message
    assert "error 20101" in message
    assert "NfsimSession" in message and "xml_path" in message


def test_the_same_text_as_a_string_is_refused(bng_xml):
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.from_sbml_string(BNG_XML)


def test_a_tracked_bng_xml_file_is_refused():
    # tests/data/nfsim/compartment_bimolecular_size2.xml, which the issue names.
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.load(str(DATA / "nfsim" / "compartment_bimolecular_size2.xml"))


def test_other_xml_in_no_sbml_namespace_is_refused_as_not_sbml(tmp_path):
    path = tmp_path / "other.xml"
    path.write_text(
        '<sbml xmlns="http://example.org/not-sbml" level="3" version="1"><model/></sbml>'
    )
    with pytest.raises(bngsim.ModelError, match="is not SBML") as raised:
        bngsim.Model.load(str(path))
    assert "BioNetGen" not in str(raised.value)
    assert "'http://example.org/not-sbml'" in str(raised.value)


def test_sbml_still_loads(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(SBML, encoding="utf-8")
    model = bngsim.Model.load(str(path))
    assert list(model.species_names) == ["A"]
    assert bngsim.Model.from_sbml_string(SBML).species_names == model.species_names


# --- the routes the message names, run -------------------------------------------

PLAIN = DATA / "nfsim" / "fixed_seed.xml"


@pytest.mark.parametrize("session", ["NfsimSession", "RuleMonkeySession"])
def test_a_session_named_in_the_message_runs_the_file(session):
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.load(str(PLAIN))
    with getattr(bngsim, session)(str(PLAIN)) as s:
        s.initialize(seed=1)
        result = s.simulate(0, 1, n_points=3)
    assert "Afree" in result.observable_names
    assert result.species.shape[0] == 3


def test_xml_path_beside_the_net_runs_the_file():
    model = bngsim.Model.from_net(str(PLAIN.with_suffix(".net")))
    result = bngsim.Simulator(model, method="nfsim", xml_path=str(PLAIN)).run(
        t_span=(0, 1), n_points=3, seed=1
    )
    assert "Afree" in result.observable_names
