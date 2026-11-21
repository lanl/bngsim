"""BioNetGen's XML is refused by the SBML loader, not loaded empty (issue #1712).

BioNetGen's XML, the file BNG2.pl's ``writeXML`` writes for NFsim and
RuleMonkey, has an ``<sbml>`` root in ``http://www.sbml.org/sbml/level3``, which
is no SBML namespace, and a ``<model>`` whose elements are not SBML's. libSBML
reports error 20101 at severity Error, reads the ``<model>`` element and reads
none of the elements inside it. bngsim's loader refused only fatal errors, so
``Model.load`` on such a file gave
``Model(species=0, reactions=0, observables=0, parameters=0)``, which ran and
returned nothing, with no error and no warning. Now it is refused, and
BioNetGen's XML is named for what it is, with the routes that run it.

What is refused is an empty model that is BioNetGen's XML or whose root is in no
SBML namespace. Error 20101 alone refuses nothing: SBML with a missing or
slightly wrong root namespace gives it too, libSBML still reads its whole model,
and it loads as it did before.
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

SBML_ROOT = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
{SBML_ROOT}
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
    assert "load the .bngl or .net it came from" in message


def test_the_same_text_as_a_string_is_refused():
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.from_sbml_string(BNG_XML)


def test_a_tracked_bng_xml_file_is_refused():
    # tests/data/nfsim/compartment_bimolecular_size2.xml, which the issue names.
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.load(str(DATA / "nfsim" / "compartment_bimolecular_size2.xml"))


def test_other_xml_in_no_sbml_namespace_is_refused_as_not_sbml(tmp_path):
    path = tmp_path / "other.xml"
    path.write_text(
        '<sbml xmlns="http://example.org/not-sbml" level="3" version="1"><model/></sbml>',
        encoding="utf-8",
    )
    with pytest.raises(bngsim.ModelError, match="is not SBML") as raised:
        bngsim.Model.load(str(path))
    assert "BioNetGen" not in str(raised.value)
    assert "'http://example.org/not-sbml'" in str(raised.value)
    assert "error 20101" in str(raised.value)


def test_sbml_still_loads(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(SBML, encoding="utf-8")
    model = bngsim.Model.load(str(path))
    assert list(model.species_names) == ["A"]
    assert bngsim.Model.from_sbml_string(SBML).species_names == model.species_names


# --- error 20101 alone refuses nothing ---------------------------------------------

# Roots libSBML gives error 20101 for while it reads the whole model: no
# namespace, a trailing slash, https, and spaces about the "=".
WRONG_ROOTS = {
    "no xmlns": '<sbml level="3" version="2">',
    "trailing slash": SBML_ROOT.replace("/core", "/core/"),
    "https": SBML_ROOT.replace("http:", "https:"),
    "spaced =": SBML_ROOT.replace("xmlns=", "xmlns = ").replace("/core", "/kore"),
}


@pytest.mark.parametrize("root", list(WRONG_ROOTS.values()), ids=list(WRONG_ROOTS))
def test_sbml_with_a_wrong_root_namespace_still_loads(tmp_path, root):
    import libsbml

    text = SBML.replace(SBML_ROOT, root)
    doc = libsbml.readSBMLFromString(text)
    errors = [doc.getError(i).getErrorId() for i in range(doc.getNumErrors())]
    assert libsbml.InvalidNamespaceOnSBML in errors  # else the case tests nothing
    assert bngsim.Model.from_sbml_string(text).species_names == ["A"]
    path = tmp_path / "m.xml"
    path.write_text(text, encoding="utf-8")
    assert bngsim.Model.load(str(path)).species_names == ["A"]


def test_bionetgens_comment_does_not_name_sbml_as_bionetgen_xml(tmp_path):
    # BNG2.pl's writeSBML writes "Created by BioNetGen" at the top of real SBML.
    # With a root that gives error 20101 and a model, that file loads; with the
    # root and no model content, it is refused as not SBML, not as BioNetGen's XML.
    comment = "<!-- Created by BioNetGen 2.9.3  -->\n"
    wrong = SBML_ROOT.replace("http:", "https:")
    text = SBML.replace(SBML_ROOT, comment + wrong)
    assert bngsim.Model.from_sbml_string(text).species_names == ["A"]
    with pytest.raises(bngsim.ModelError, match="is not SBML") as raised:
        bngsim.Model.from_sbml_string(f"{comment}{wrong}<model/></sbml>")
    assert "BioNetGen" not in str(raised.value)


def test_an_empty_model_in_an_sbml_namespace_still_loads():
    # Valid SBML, and it loaded before.
    model = bngsim.Model.from_sbml_string(f"{SBML_ROOT}<model/></sbml>")
    assert list(model.species_names) == []


# --- what the message says ---------------------------------------------------------


def test_bionetgen_xml_in_an_sbml_namespace_is_refused_too():
    # No error 20101 here: the root has been given a real SBML namespace. The
    # model is as empty, and the file is as much BioNetGen's.
    text = BNG_XML.replace(
        "http://www.sbml.org/sbml/level3", "http://www.sbml.org/sbml/level3/version1/core"
    )
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML") as raised:
        bngsim.Model.from_sbml_string(text)
    assert "20101" not in str(raised.value)
    assert "no SBML namespace" not in str(raised.value)


def test_bionetgen_xml_is_told_past_the_first_64_kib(tmp_path):
    # A model with many parameters puts <ListOfMoleculeTypes> far into the file.
    many = "\n".join(
        f'      <Parameter id="k{i}" type="Constant" value="1" expr="1"/>' for i in range(1500)
    )
    text = BNG_XML.replace("    </ListOfParameters>", many + "\n    </ListOfParameters>")
    text = text.replace("<!-- Created by BioNetGen 2.9.3  -->\n", "")
    assert text.index("<ListOfMoleculeTypes>") > 65536
    path = tmp_path / "big.xml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.load(str(path))
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.from_sbml_string(text)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_bionetgen_xml_saved_as_utf16_is_named(tmp_path, encoding):
    bom = {"utf-16": b"", "utf-16-le": b"\xff\xfe", "utf-16-be": b"\xfe\xff"}[encoding]
    path = tmp_path / "model.xml"
    path.write_bytes(bom + BNG_XML.replace("UTF-8", "UTF-16").encode(encoding))
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML") as raised:
        bngsim.Model.load(str(path))
    assert "'http://www.sbml.org/sbml/level3'" in str(raised.value)


def test_a_root_with_no_namespace_is_not_said_to_name_one():
    with pytest.raises(bngsim.ModelError, match="is not SBML") as raised:
        bngsim.Model.from_sbml_string('<sbml level="3" version="1"><model/></sbml>')
    message = str(raised.value)
    assert "its root element is in no SBML namespace" in message
    assert "it names" not in message and "in the namespace" not in message


def test_the_namespace_quoted_is_the_roots_not_a_comments():
    text = (
        '<!-- <sbml xmlns="decoy"> -->\n'
        '<sbml xmlns = "http://example.org/x" level="3" version="1"><model/></sbml>'
    )
    with pytest.raises(bngsim.ModelError, match="is not SBML") as raised:
        bngsim.Model.from_sbml_string(text)
    assert "'http://example.org/x'" in str(raised.value)
    assert "decoy" not in str(raised.value)


def test_the_signature_inside_a_comment_does_not_name_bionetgen_xml():
    text = (
        '<sbml xmlns="http://example.org/x" level="3" version="1">'
        "<!-- <ListOfMoleculeTypes> --><model/></sbml>"
    )
    with pytest.raises(bngsim.ModelError, match="is not SBML") as raised:
        bngsim.Model.from_sbml_string(text)
    assert "BioNetGen" not in str(raised.value)


@pytest.mark.parametrize(
    ("nfsim", "rulemonkey", "named", "unnamed"),
    [
        (True, True, ["bngsim.NfsimSession(path)", "bngsim.RuleMonkeySession(path)"], []),
        (True, False, ["bngsim.NfsimSession(path)", "xml_path=path"], ["RuleMonkeySession"]),
        (False, True, ["bngsim.RuleMonkeySession(path)"], ["NfsimSession", "xml_path"]),
        (False, False, ["this build has neither"], ["Session", "xml_path"]),
    ],
    ids=["both", "nfsim only", "rulemonkey only", "neither"],
)
def test_the_message_names_only_the_routes_this_build_has(
    monkeypatch, nfsim, rulemonkey, named, unnamed
):
    monkeypatch.setattr(bngsim, "HAS_NFSIM", nfsim)
    monkeypatch.setattr(bngsim, "HAS_RULEMONKEY", rulemonkey)
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML") as raised:
        bngsim.Model.from_sbml_string(BNG_XML)
    message = str(raised.value)
    assert all(route in message for route in named), message
    assert not any(route in message for route in unnamed), message
    assert "load the .bngl or .net it came from" in message
    # Pasted as written, every name resolves: none is unqualified.
    assert "(Model.from_net" not in message


# --- the routes the message names, run -------------------------------------------

PLAIN = DATA / "nfsim" / "fixed_seed.xml"

# Literal reasons, the ones the other network-free tests give: the skip audit
# (test_skip_audit.py) reads each reason from the source.
needs_nfsim = pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not compiled in")
needs_rulemonkey = pytest.mark.skipif(
    not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not compiled in"
)


@pytest.mark.parametrize(
    "session",
    [
        pytest.param("NfsimSession", marks=needs_nfsim),
        pytest.param("RuleMonkeySession", marks=needs_rulemonkey),
    ],
)
def test_a_session_named_in_the_message_runs_the_file(session):
    with pytest.raises(bngsim.ModelError, match="is BioNetGen's XML"):
        bngsim.Model.load(str(PLAIN))
    with getattr(bngsim, session)(str(PLAIN)) as s:
        s.initialize(seed=1)
        result = s.simulate(0, 1, n_points=3)
    assert "Afree" in result.observable_names
    assert result.species.shape[0] == 3


@needs_nfsim
def test_xml_path_beside_the_net_runs_the_file():
    model = bngsim.Model.from_net(str(PLAIN.with_suffix(".net")))
    result = bngsim.Simulator(model, method="nfsim", xml_path=str(PLAIN)).run(
        t_span=(0, 1), n_points=3, seed=1
    )
    assert "Afree" in result.observable_names
