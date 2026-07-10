"""Tests for the multi-experiment SED-ML protocol channel (GH #211, Option 3).

A whole BNGL actions block is a *sequence* of experiments (simulate /
parameter_scan) with interleaved model-state changes — more than the single
uniform time course the #218 sidecar carries. :func:`write_sedml_protocol` /
:func:`read_sedml_protocol` map a :class:`ProtocolSpec` to/from SED-ML, emitting
standards-compliant elements (a uniformTimeCourse + task per experiment, a
repeatedTask + range per scan, changeAttribute-derived models for accumulated
overrides) plus a bngsim annotation that makes the round-trip exact.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import bngsim
import pytest
from bngsim.convert import (
    parse_bngl_protocol,
    read_sedml_protocol,
    write_sedml_protocol,
)

_SEQUENTIAL = (
    "begin model\nend model\ngenerate_network()\n"
    'simulate({method=>"ode",t_end=>600,n_steps=>10})\n'
    'setParameter("L",1)\n'
    'simulate({method=>"ode",t_end=>60,n_steps=>60})\n'
)
_MULTISTAGE = (
    "begin model\nend model\ngenerate_network()\n"
    'setParameter("a",0)\n'
    'parameter_scan({method=>"ode",t_end=>10,n_steps=>10,par_min=>0,par_max=>1,'
    'n_scan_pts=>20,parameter=>"p",suffix=>"00"})\n'
    "resetConcentrations()\n"
    'setParameter("a",1)\n'
    'parameter_scan({method=>"ssa",t_end=>10,n_steps=>10,par_min=>0,par_max=>2,'
    'n_scan_pts=>5,parameter=>"p",suffix=>"10"})\n'
)
_SINGLE = 'begin model\nend model\nsimulate({method=>"ode",t_end=>100,n_steps=>100})\n'


def _locals(root: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == name]


# ─── Exact round-trip via the bngsim annotation ─────────────────────────────


@pytest.mark.parametrize("text", [_SINGLE, _SEQUENTIAL, _MULTISTAGE])
def test_exact_round_trip(text: str) -> None:
    p = parse_bngl_protocol(text)
    xml = write_sedml_protocol(p, model_source="model.xml")
    back = read_sedml_protocol(xml)
    assert back.to_dict() == p.to_dict()


# ─── Standards-compliant elements for interop ───────────────────────────────


def test_sequential_emits_task_per_experiment_and_derived_model() -> None:
    p = parse_bngl_protocol(_SEQUENTIAL)
    root = ET.fromstring(write_sedml_protocol(p))
    assert len(_locals(root, "uniformTimeCourse")) == 2
    assert len(_locals(root, "task")) == 2
    # The setParameter before the 2nd simulate → one changeAttribute-derived model.
    assert len(_locals(root, "changeAttribute")) == 1
    assert len(_locals(root, "repeatedTask")) == 0


def test_scan_emits_repeated_task_with_range_and_setvalue() -> None:
    p = parse_bngl_protocol(_MULTISTAGE)
    root = ET.fromstring(write_sedml_protocol(p))
    assert len(_locals(root, "repeatedTask")) == 2
    assert len(_locals(root, "setValue")) == 2
    rng = _locals(root, "uniformRange")[0]
    assert rng.get("start") == "0" and rng.get("end") == "1"
    assert rng.get("numberOfPoints") == "20"


def test_well_formed_xml_and_file_write(tmp_path: Path) -> None:
    p = parse_bngl_protocol(_MULTISTAGE)
    out = tmp_path / "protocol.sedml"
    text = write_sedml_protocol(p, out, model_source="m.xml")
    assert out.is_file() and out.read_text() == text
    ET.fromstring(text)  # parseable
    assert "model.xml" not in text  # honored the custom model_source
    assert 'source="m.xml"' in text


# ─── Best-effort reconstruction when the annotation is absent (foreign SED-ML)


def test_best_effort_without_annotation() -> None:
    p = parse_bngl_protocol(_MULTISTAGE)
    xml = write_sedml_protocol(p)
    # Strip the bngsim annotation payload → forces standard-element reconstruction.
    xml_foreign = xml.replace(p.to_json(), "")
    back = read_sedml_protocol(xml_foreign)
    assert len(back.experiments) == 2
    scans = [e for e in back.experiments if e.is_scan]
    assert len(scans) == 2
    assert scans[0].scan_parameter == "p"
    assert scans[0].scan_points == 20
    assert scans[0].method == "ode" and scans[1].method == "ssa"
    assert scans[1].scan_max == 2.0


def test_best_effort_recovers_sequential_horizons() -> None:
    p = parse_bngl_protocol(_SEQUENTIAL)
    xml = write_sedml_protocol(p)
    back = read_sedml_protocol(xml.replace(p.to_json(), ""))
    spans = [e.t_span for e in back.experiments]
    assert spans == [(0.0, 600.0), (0.0, 60.0)]


# ─── nf method: annotation preserves it, standard element approximates ──────


def test_nf_method_preserved_in_annotation() -> None:
    p = parse_bngl_protocol("begin model\nend model\nsimulate_nf({t_end=>10,n_steps=>10})\n")
    xml = write_sedml_protocol(p)
    assert read_sedml_protocol(xml).experiments[0].method == "nf"  # exact via annotation


# ─── Capability boundary ────────────────────────────────────────────────────


def test_empty_protocol_refused() -> None:
    p = parse_bngl_protocol("begin model\nend model\ngenerate_network()\n")
    with pytest.raises(bngsim.ConversionError, match="empty protocol"):
        write_sedml_protocol(p)


def test_not_a_sedml_document_raises() -> None:
    with pytest.raises(bngsim.ConversionError, match="not a SED-ML document"):
        read_sedml_protocol("<notSed/>")


# ─── Real corpus: every multi-experiment bngl round-trips ───────────────────


def test_real_corpus_protocols_round_trip() -> None:
    bngl_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "bngl"
    if not bngl_dir.is_dir():
        pytest.skip("benchmark bngl corpus not present")
    files = sorted(bngl_dir.rglob("*.bngl"))
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        for f in files:
            p = parse_bngl_protocol(f, strict=False)
            if p.is_empty:
                continue
            xml = write_sedml_protocol(p)
            assert read_sedml_protocol(xml).to_dict() == p.to_dict(), f
            checked += 1
    assert checked > 0
