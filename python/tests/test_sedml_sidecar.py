"""Tests for bngsim.convert SED-ML sidecar protocol channel (GH #218, epic #211).

SBML/`.net` carry structure+math only; SED-ML supplies the simulation protocol
(time grid, outputs, solver/tolerances). These tests cover the
EvaluationSpec⇄SED-ML mapping in both directions: an exact write→read round-trip
of the uniform-time-course subset (t-grid, outputs, method, tolerances), the
n_points↔numberOfPoints convention, KiSAO method/tolerance terms,
namespace-agnostic reading of a foreign (L1V2) document, the no-sidecar
``default_protocol`` fallback, the unsupported-method and malformed-input
guards, the headline acceptance (``sbml2net`` + sidecar yields a runnable bngsim
job whose protocol matches the SED-ML), and the ``net2sbml --sidecar`` CLI.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim import EvaluationSpec
from bngsim.convert import (
    default_protocol,
    read_sedml,
    write_sedml,
)

_BNGSIM = Path(__file__).resolve().parents[2]
_DATA = _BNGSIM / "tests" / "data"


@pytest.fixture
def data_dir() -> Path:
    assert _DATA.is_dir(), f"Test data directory not found: {_DATA}"
    return _DATA


def _require(name: str, data_dir: Path) -> Path:
    src = data_dir / name
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    return src


# ─── write → read round-trip ────────────────────────────────────────────────


def test_roundtrip_preserves_protocol() -> None:
    spec = EvaluationSpec(
        model_source="model.xml",
        model_format="sbml",
        method="ode",
        t_span=(0.0, 50.0),
        n_points=26,
        outputs=("observable:Atot", "species:A", "expression:foo"),
        rtol=1e-7,
        atol=1e-9,
        max_steps=5000,
    )
    back = read_sedml(write_sedml(spec))
    assert back.t_span == spec.t_span
    assert back.n_points == spec.n_points
    assert back.outputs == spec.outputs
    assert back.method == spec.method
    assert (back.rtol, back.atol, back.max_steps) == (spec.rtol, spec.atol, spec.max_steps)
    assert back.model_source == "model.xml"


def test_numberofpoints_convention() -> None:
    # SED-ML numberOfPoints counts steps after the start → n_points - 1.
    spec = EvaluationSpec(model_source="m.xml", t_span=(0.0, 10.0), n_points=11)
    text = write_sedml(spec)
    assert 'numberOfPoints="10"' in text
    assert read_sedml(text).n_points == 11


def test_tolerances_optional() -> None:
    # With no tolerances set, no algorithmParameter block is emitted and the
    # read-back leaves them None (Simulator defaults).
    spec = EvaluationSpec(model_source="m.xml", t_span=(0, 5), n_points=6)
    text = write_sedml(spec)
    assert "algorithmParameter" not in text
    back = read_sedml(text)
    assert back.rtol is None and back.atol is None and back.max_steps is None


def test_ssa_method_maps_to_kisao() -> None:
    spec = EvaluationSpec(
        model_source="m.xml", method="ssa", t_span=(0, 10), n_points=11, outputs=("observable:X",)
    )
    text = write_sedml(spec)
    assert "KISAO:0000029" in text
    assert read_sedml(text).method == "ssa"


def test_unsupported_method_rejected() -> None:
    spec = EvaluationSpec(model_source="m.xml", method="nfsim", t_span=(0, 10), n_points=11)
    with pytest.raises(bngsim.ConversionError, match="no KiSAO"):
        write_sedml(spec)


def test_write_to_file(tmp_path: Path) -> None:
    spec = EvaluationSpec(model_source="m.xml", t_span=(0, 5), n_points=6)
    out = tmp_path / "proto.sedml"
    returned = write_sedml(spec, out)
    assert out.is_file()
    assert out.read_text() == returned
    assert read_sedml(out).n_points == 6


def test_model_source_override_on_write_and_read(tmp_path: Path) -> None:
    spec = EvaluationSpec(model_source="orig.xml", t_span=(0, 5), n_points=6)
    text = write_sedml(spec, model_source="converted.net")
    assert 'source="converted.net"' in text
    back = read_sedml(text, model_source="run.net", model_format="net")
    assert back.model_source == "run.net"
    assert back.model_format == "net"


# ─── namespace-agnostic reading of a foreign document ───────────────────────


def test_reads_foreign_l1v2_document() -> None:
    # A hand-authored L1V2-style document with default-namespaced MathML and a
    # different SED-ML namespace — the reader is namespace-agnostic.
    doc = """<?xml version="1.0" encoding="UTF-8"?>
<sedML xmlns="http://sed-ml.org/sed-ml/level1/version2" level="1" version="2">
  <listOfSimulations>
    <uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
                       outputEndTime="200" numberOfPoints="40">
      <algorithm kisaoID="KISAO:0000088">
        <listOfAlgorithmParameters>
          <algorithmParameter kisaoID="KISAO:0000209" value="1e-6"/>
        </listOfAlgorithmParameters>
      </algorithm>
    </uniformTimeCourse>
  </listOfSimulations>
  <listOfModels>
    <model id="m" language="urn:sedml:language:sbml" source="foo.xml"/>
  </listOfModels>
  <listOfTasks><task id="t" modelReference="m" simulationReference="s"/></listOfTasks>
  <listOfDataGenerators>
    <dataGenerator id="dt" name="time">
      <listOfVariables>
        <variable id="vt" symbol="urn:sedml:symbol:time" taskReference="t"/>
      </listOfVariables>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>vt</ci></math>
    </dataGenerator>
    <dataGenerator id="d0" name="observable:Y">
      <listOfVariables>
        <variable id="v0" name="observable:Y" taskReference="t"/>
      </listOfVariables>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>v0</ci></math>
    </dataGenerator>
  </listOfDataGenerators>
</sedML>"""
    spec = read_sedml(doc)
    assert spec.t_span == (0.0, 200.0)
    assert spec.n_points == 41
    assert spec.method == "ode"  # KISAO:0000088 (LSODA) → ode
    assert spec.rtol == 1e-6
    assert spec.outputs == ("observable:Y",)
    assert spec.model_source == "foo.xml"


def test_unmapped_kisao_warns_and_defaults_ode() -> None:
    doc = """<sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" level="1" version="3">
  <listOfSimulations>
    <uniformTimeCourse id="s" outputStartTime="0" outputEndTime="10" numberOfPoints="10">
      <algorithm kisaoID="KISAO:0009999"/>
    </uniformTimeCourse>
  </listOfSimulations>
</sedML>"""
    with pytest.warns(bngsim.ConversionWarning, match="unmapped algorithm KiSAO"):
        spec = read_sedml(doc)
    assert spec.method == "ode"


# ─── malformed input guards ─────────────────────────────────────────────────


def test_non_sedml_root_rejected() -> None:
    with pytest.raises(bngsim.ConversionError, match="not a SED-ML document"):
        read_sedml("<sbml></sbml>")


def test_missing_uniform_time_course_rejected() -> None:
    doc = '<sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" level="1" version="3"/>'
    with pytest.raises(bngsim.ConversionError, match="uniformTimeCourse"):
        read_sedml(doc)


def test_unparseable_rejected() -> None:
    with pytest.raises(bngsim.ConversionError, match="parseable"):
        read_sedml("<sedML this is not xml")


# ─── default_protocol (no-sidecar fallback) ─────────────────────────────────


def test_default_protocol_reports_observables(data_dir: Path) -> None:
    src = _require("two_species_reversible.net", data_dir)
    spec = default_protocol(src, t_span=(0, 20), n_points=21)
    assert spec.model_format == "net"
    assert spec.model_source == str(src)
    assert all(o.startswith("observable:") for o in spec.outputs)
    assert len(spec.outputs) == bngsim.Model.from_net(src).n_observables


def test_default_protocol_from_loaded_model(data_dir: Path) -> None:
    src = _require("simple_decay.net", data_dir)
    model = bngsim.Model.from_net(src)
    spec = default_protocol(model, model_source="x.net", model_format="net")
    assert spec.outputs  # non-empty
    assert spec.model_source == "x.net"


# ─── headline acceptance: sbml2net + sidecar → runnable job ─────────────────


def test_sbml2net_plus_sidecar_yields_runnable_job(data_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("libsbml")
    from bngsim.convert import sbml_to_net, write_sbml

    src = _require("two_species_reversible.net", data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_net(src)
        sbml = tmp_path / "m.xml"
        write_sbml(model, sbml, strict=True)
        # Emit the protocol sidecar, then convert the network the other way.
        proto = default_protocol(sbml, model_format="sbml", t_span=(0, 30), n_points=16)
        sidecar = write_sedml(proto, tmp_path / "m.sedml")
        net = tmp_path / "m.net"
        sbml_to_net(sbml, net, validate=None, strict=False)

    # The SED-ML protocol drives a runnable bngsim job against the converted .net.
    job = read_sedml(sidecar, model_source=str(net), model_format="net")
    assert job.t_span == (0.0, 30.0)
    assert job.n_points == 16
    result = job.evaluate()
    assert result.observables.shape[0] == job.n_points  # protocol matches


# ─── CLI ────────────────────────────────────────────────────────────────────


def test_cli_net2sbml_sidecar(data_dir: Path, tmp_path: Path, capsys) -> None:
    pytest.importorskip("libsbml")
    from bngsim.convert._cli import net2sbml_main

    src = _require("two_species_reversible.net", data_dir)
    out = tmp_path / "out.xml"
    rc = net2sbml_main([str(src), "-o", str(out), "--sidecar", "--no-validate"])
    assert rc == 0
    sidecar = out.with_suffix(".sedml")
    assert sidecar.is_file()
    assert "sidecar" in capsys.readouterr().out
    # The emitted sidecar parses back to a protocol over the SBML output.
    spec = read_sedml(sidecar)
    assert spec.outputs
    assert spec.model_source == str(out)
