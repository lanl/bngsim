"""Every SED-ML document bngsim writes is valid against the SED-ML Level 1
Version 3 schema (issue #1734).

#1677 found by hand, with ``xmllint``, that every document
:func:`write_sedml_protocol` wrote failed the schema (``listOfModels`` before
``listOfSimulations``), so every archive ``net_to_omex(bngl=...)`` made did. A
test that held the order as written into the test passed the same way the
writer did. This one holds each document to the schema itself, with
``lxml.etree.XMLSchema``: the libxml2 validator ``xmllint --schema`` runs, in
process, so it runs on every leg. By hand it is

    xmllint --noout --schema tests/data/sedml_schema/sed-ml-L1-V3.xsd simulation.sedml

The schema is vendored under ``tests/data/sedml_schema`` (its README says from
where). It checks structure only: a valid document can still mean something
other than the protocol, which the tests that run the documents check
(``test_sedml_protocol_fidelity.py``).
"""

from __future__ import annotations

import os
import sys
import warnings
import zipfile
from pathlib import Path

import bngsim
import pytest
from bngsim import EvaluationSpec
from bngsim.convert import (
    default_protocol,
    net_to_omex,
    parse_bngl_protocol,
    write_sedml,
    write_sedml_protocol,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _source_root import bngsim_source_root  # noqa: E402

etree = pytest.importorskip("lxml.etree")

_ROOT = bngsim_source_root()
_SCHEMA_DIR = (_ROOT or Path(__file__).resolve().parents[2]) / "tests" / "data" / "sedml_schema"


@pytest.fixture(scope="module")
def schema():
    """The L1V3 schema, read with no network: its MathML part is the copy
    beside it, which it imports by a relative schemaLocation."""
    path = _SCHEMA_DIR / "sed-ml-L1-V3.xsd"
    assert path.is_file(), f"the vendored SED-ML schema is not at {path}"
    parser = etree.XMLParser(no_network=True)
    return etree.XMLSchema(etree.parse(str(path), parser))


def _assert_valid(schema, text: str | bytes, what: object) -> None:
    doc = etree.fromstring(text.encode() if isinstance(text, str) else text)
    if not schema.validate(doc):
        errors = "\n".join(str(e) for e in schema.error_log)
        pytest.fail(f"{what}: not valid SED-ML L1V3:\n{errors}")


# ─── the schema catches what #1677 was ──────────────────────────────────────


def test_the_schema_refuses_the_order_1677_wrote(schema):
    """A live check: the lists in #1677's order fail it."""
    protocol = parse_bngl_protocol('simulate({method=>"ode",t_end=>1,n_steps=>1})\n')
    doc = etree.fromstring(write_sedml_protocol(protocol).encode())
    sed = "{http://sed-ml.org/sed-ml/level1/version3}"
    models = doc.find(f"{sed}listOfModels")
    doc.remove(models)
    doc.insert(list(doc).index(doc.find(f"{sed}listOfSimulations")), models)
    assert not schema.validate(doc)
    assert "listOfSimulations" in str(schema.error_log.last_error)


# ─── write_sedml and the net2sbml sidecar ───────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [
        EvaluationSpec(
            model_source="model.xml",
            model_format="sbml",
            method="ode",
            t_span=(0.0, 50.0),
            n_points=26,
            outputs=("observable:Atot", "species:A", "expression:foo"),
            rtol=1e-7,
            atol=1e-9,
            max_steps=5000,
        ),
        EvaluationSpec(
            model_source="model.xml",
            model_format="sbml",
            method="ssa",
            t_span=(0.0, 10.0),
            n_points=11,
            outputs=("species:A",),
        ),
    ],
    ids=["ode-with-tolerances", "ssa"],
)
def test_write_sedml_writes_valid_sedml(schema, spec):
    with warnings.catch_warnings():
        # With no SBML, a species output has no target, and says so.
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        text = write_sedml(spec)
    _assert_valid(schema, text, spec.method)


def test_the_net2sbml_sidecar_is_valid_sedml(schema, tmp_path):
    pytest.importorskip("libsbml")
    from bngsim.convert._cli import net2sbml_main

    src = _SCHEMA_DIR.parent / "two_species_reversible.net"
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    out = tmp_path / "out.xml"
    assert net2sbml_main([str(src), "-o", str(out), "--sidecar", "--no-validate"]) == 0
    _assert_valid(schema, out.with_suffix(".sedml").read_bytes(), "net2sbml --sidecar")


def test_default_protocol_writes_valid_sedml(schema):
    src = _SCHEMA_DIR.parent / "two_species_reversible.net"
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    model = bngsim.Model.from_net(str(src))
    _assert_valid(schema, write_sedml(default_protocol(model)), "default_protocol")


# ─── write_sedml_protocol over the repository's .bngl corpus ────────────────


def _corpus() -> list[Path]:
    if _ROOT is None:
        return []
    return sorted((_ROOT / "benchmarks" / "models" / "bngl").rglob("*.bngl"))


def test_every_corpus_protocol_is_valid_sedml(schema):
    """Each protocol of the corpus' .bngl files (87 when #1734 was filed,
    every one invalid on the main of then)."""
    files = _corpus()
    if not files:
        pytest.skip("benchmark bngl corpus not present")
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        for f in files:
            protocol = parse_bngl_protocol(f, strict=False)
            if protocol.is_empty:
                continue
            _assert_valid(schema, write_sedml_protocol(protocol), f.name)
            checked += 1
    assert checked >= 80


# ─── the archives net_to_omex writes, in each shape the writer has ──────────

NET = """begin parameters
    1 k1 1
    2 k2 k1*3
    3 A0 100
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 k1
    2 2 1 k2
end reactions
begin groups
    1 Atot 1
end groups
"""
MODEL = """begin model
begin parameters
  k1 1
  k2 k1*3
  A0 100
end parameters
begin seed species
  A() A0
  B() 0
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() <-> B() k1, k2
end reaction rules
end model
"""
GEN = "generate_network({overwrite=>1})\n"
RUN = 'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
SCAN = (
    'parameter_scan({{method=>"ode",parameter=>"{p}",par_scan_vals=>[1,2],{x}'
    "t_end=>1,n_steps=>1}})\n"
)
SHAPES = {
    # a data generator and a report per experiment
    "a-run": GEN + RUN,
    # changeAttribute before the model loads, under its assignment
    "set-before-a-run": GEN + 'setParameter("k1",5)\n' + RUN,
    # removeXML of the assignment, then changeAttribute
    "replace-a-derived-parameter": GEN + 'setParameter("k2",7)\n' + RUN,
    # computeChange: an expression, and an addition
    "expressions-before-a-run": GEN
    + 'setParameter("k1","A0/50")\naddConcentration("A()",5)\n'
    + RUN,
    # setValue between chained runs, with what follows it
    "set-between-runs": GEN + RUN + 'setParameter("k1",5)\nsetConcentration("A()",7)\n' + RUN,
    "expression-between-runs": GEN
    + RUN
    + 'setParameter("k2","k1*4")\nsetParameter("k1",2)\n'
    + RUN,
    # setValue with a range, and what follows the swept parameter from it
    "scan-that-resets": GEN + SCAN.format(p="k1", x=""),
    "dose-scan": GEN + SCAN.format(p="A0", x=""),
    "scan-that-goes-on": GEN + SCAN.format(p="A0", x="reset_conc=>0,"),
    "run-after-a-scan-that-goes-on": GEN + SCAN.format(p="A0", x="reset_conc=>0,") + RUN,
    "bifurcate": GEN
    + 'bifurcate({method=>"ode",parameter=>"k1",par_min=>1,par_max=>3,n_scan_pts=>3,'
    "t_end=>1,n_steps=>1})\n",
    # a reset back to the network's expression, and to saved concentrations
    "resets": GEN
    + "saveParameters()\nsaveConcentrations()\n"
    + RUN
    + 'setParameter("k2",7)\n'
    + RUN
    + "resetParameters()\nresetConcentrations()\n"
    + RUN,
    # saves before the network was generated
    "saves-before-generation": 'setParameter("k1",2)\nsaveParameters()\nsetParameter("k1",1)\n'
    + GEN
    + "resetParameters()\n"
    + RUN,
    "ssa": GEN + 'simulate({method=>"ssa",t_end=>1,n_steps=>2})\n',
    # computeChange of what an assignment computes, before an expression that reads it
    "expression-of-a-derived-parameter": GEN
    + 'setParameter("k1",5)\nsetConcentration("B()","k2*2")\n'
    + RUN,
    # a species set to its seed before an addition, in a model and in a step
    "add-to-a-seed": GEN + 'setParameter("A0",50)\naddConcentration("A()",10)\n' + RUN + RUN,
    # setValue of a species from a range
    "scan-of-an-amount-set-to-an-expression": GEN
    + 'setConcentration("B()","A0*2")\n'
    + SCAN.format(p="A0", x=""),
}


def test_a_min_the_sbml_or_an_action_holds_is_valid_sedml(schema, tmp_path):
    """SED-ML L1V3's MathML is SBML Level 2's, which has no min or max, and
    SBML L3V2's, which the SBML writer uses, has: a derived k2 = min(k1*3, 6)
    said again beside a scan of k1, and an action's max, are each written as
    a piecewise."""
    pytest.importorskip("libsbml")
    (tmp_path / "m.net").write_text(NET.replace("k2 k1*3", "k2 min((k1*3),6)"))
    (tmp_path / "m.bngl").write_text(
        MODEL.replace("k2 k1*3", "k2 min(k1*3,6)")
        + GEN
        + 'setConcentration("B()","max(A0/50,k1,3)")\n'
        + SCAN.format(p="k1", x="")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(tmp_path / "m.net", tmp_path / "m.omex", bngl=tmp_path / "m.bngl", gate="none")
    text = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml")
    doc = etree.fromstring(text)
    mathml = "{http://www.w3.org/1998/Math/MathML}"
    assert len(list(doc.iter(f"{mathml}piecewise"))) >= 3  # k2's, and the max of three
    _assert_valid(schema, text, "min and max")


@pytest.mark.parametrize("actions", list(SHAPES.values()), ids=list(SHAPES))
def test_each_archive_holds_valid_sedml(schema, tmp_path, actions):
    pytest.importorskip("libsbml")
    (tmp_path / "m.net").write_text(NET)
    (tmp_path / "m.bngl").write_text(MODEL + actions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(tmp_path / "m.net", tmp_path / "m.omex", bngl=tmp_path / "m.bngl", gate="none")
    text = zipfile.ZipFile(tmp_path / "m.omex").read("simulation.sedml")
    _assert_valid(schema, text, actions)
