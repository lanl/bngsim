"""Tests for bngsim.convert OMEX / COMBINE archive packaging (GH #219, epic #211).

A COMBINE archive (`.omex`) is the standard zip container bundling a model
(SBML), its simulation protocol (SED-ML), and a `manifest.xml` listing every
entry's format URI. These tests cover the packaging layer on top of the SBML
(#216) and SED-ML (#218) channels: writing an archive and round-tripping its
manifest (entry format URIs correct), reading a hand-authored BioModels-style
archive into a runnable network + protocol, the `net_to_omex` end-to-end
orchestrator, the no-SED-ML default-protocol fallback, the malformed-input and
zip-slip guards, and the `bngsim-omex` pack/unpack CLI.
"""

from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import bngsim
import pytest
from bngsim.convert import (
    OmexArchive,
    net_to_omex,
    read_omex,
    write_omex,
)

_BNGSIM = Path(__file__).resolve().parents[2]
_DATA = _BNGSIM / "tests" / "data"

_FMT_SBML = "http://identifiers.org/combine.specifications/sbml"
_FMT_SEDML = "http://identifiers.org/combine.specifications/sed-ml"
_FMT_MANIFEST = "http://identifiers.org/combine.specifications/omex-manifest"


@pytest.fixture
def data_dir() -> Path:
    assert _DATA.is_dir(), f"Test data directory not found: {_DATA}"
    return _DATA


def _require(name: str, data_dir: Path) -> Path:
    src = data_dir / name
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    return src


# ─── write_omex: manifest + format URIs ─────────────────────────────────────


def test_write_omex_manifest_and_format_uris(tmp_path: Path) -> None:
    out = tmp_path / "a.omex"
    write_omex(
        out,
        sbml="<sbml/>",
        sedml="<sedML/>",
        metadata="<rdf/>",
    )
    assert zipfile.is_zipfile(out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {"manifest.xml", "model.xml", "simulation.sedml", "metadata.rdf"} <= names
        manifest = zf.read("manifest.xml").decode()

    # The manifest lists the archive root, itself, and each content with the
    # correct COMBINE format URI; the model is the master.
    assert 'location="." format="http://identifiers.org/combine.specifications/omex"' in manifest
    assert f'location="./manifest.xml" format="{_FMT_MANIFEST}"' in manifest
    assert f'location="./model.xml" format="{_FMT_SBML}"' in manifest
    assert 'master="true"' in manifest
    assert f'location="./simulation.sedml" format="{_FMT_SEDML}"' in manifest


def test_write_omex_requires_a_model(tmp_path: Path) -> None:
    with pytest.raises(bngsim.ConversionError, match="at least a model"):
        write_omex(tmp_path / "a.omex", sedml="<sedML/>")


def test_write_omex_net_master(tmp_path: Path) -> None:
    out = tmp_path / "n.omex"
    write_omex(out, net="begin model\nend model\n")
    arch = read_omex(out, extract_dir=tmp_path / "ex")
    entry = arch.master_model_entry()
    assert entry is not None
    assert entry.kind == "net"
    assert entry.master


def test_write_omex_rejects_duplicate_locations(tmp_path: Path) -> None:
    with pytest.raises(bngsim.ConversionError, match="duplicate archive location"):
        write_omex(
            tmp_path / "d.omex",
            sbml="<sbml/>",
            net="x",
            sbml_location="./shared.xml",
            net_location="./shared.xml",
        )


# ─── read_omex: hand-authored BioModels-style archive → network + protocol ──


def _make_biomodels_style_omex(out: Path, sbml_text: str, sedml_text: str) -> None:
    manifest = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<omexManifest xmlns="http://identifiers.org/combine.specifications/omex-manifest">\n'
        '  <content location="." format="http://identifiers.org/combine.specifications/omex"/>\n'
        '  <content location="./manifest.xml" '
        'format="http://identifiers.org/combine.specifications/omex-manifest"/>\n'
        '  <content location="./model.xml" '
        'format="http://identifiers.org/combine.specifications/sbml.level-3.version-2" '
        'master="true"/>\n'
        '  <content location="./sim.sedml" '
        'format="http://identifiers.org/combine.specifications/sed-ml.level-1.version-3"/>\n'
        "</omexManifest>\n"
    )
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("manifest.xml", manifest)
        zf.writestr("model.xml", sbml_text)
        zf.writestr("sim.sedml", sedml_text)


def test_read_biomodels_style_archive(data_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("libsbml")
    from bngsim.convert import default_protocol, write_sbml, write_sedml

    src = _require("two_species_reversible.net", data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_net(src)
        sbml_text = write_sbml(model, None, strict=False)
        proto = default_protocol(
            model, model_source="model.xml", model_format="sbml", t_span=(0, 40), n_points=21
        )
        sedml_text = write_sedml(proto, model_source="model.xml")

    omex = tmp_path / "biomodels.omex"
    _make_biomodels_style_omex(omex, sbml_text, sedml_text)

    # Versioned format URIs (…/sbml.level-3.version-2) must still dispatch.
    with read_omex(omex) as arch:
        assert arch.master_model_entry().kind == "sbml"
        assert arch.master_sedml_entry().kind == "sedml"
        loaded = arch.load_model()
        assert loaded.n_species == model.n_species
        job = arch.load_protocol()
        assert job.t_span == (0.0, 40.0)
        assert job.n_points == 21
        result = job.evaluate()
        assert result.observables.shape[0] == 21


def test_read_omex_default_protocol_when_no_sedml(data_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("libsbml")
    from bngsim.convert import write_sbml

    src = _require("two_species_reversible.net", data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        model = bngsim.Model.from_net(src)
        sbml_text = write_sbml(model, None, strict=False)

    omex = tmp_path / "model_only.omex"
    write_omex(omex, sbml=sbml_text)  # no SED-ML

    with read_omex(omex) as arch:
        assert arch.master_sedml_entry() is None
        # Falls back to a default protocol over the model's observables.
        proto = arch.load_protocol()
        assert proto.outputs
        assert proto.model_format == "sbml"


# ─── read_omex guards ───────────────────────────────────────────────────────


def test_read_omex_rejects_non_zip(tmp_path: Path) -> None:
    bogus = tmp_path / "not.omex"
    bogus.write_text("i am not a zip")
    with pytest.raises(bngsim.ConversionError, match="not a zip archive"):
        read_omex(bogus)


def test_read_omex_rejects_missing_manifest(tmp_path: Path) -> None:
    out = tmp_path / "nomanifest.omex"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("model.xml", "<sbml/>")
    with pytest.raises(bngsim.ConversionError, match="no manifest.xml"):
        read_omex(out)


def test_read_omex_rejects_zip_slip(tmp_path: Path) -> None:
    out = tmp_path / "evil.omex"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("manifest.xml", "<omexManifest/>")
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(bngsim.ConversionError, match="escapes the archive root"):
        read_omex(out, extract_dir=tmp_path / "ex")


def test_load_model_errors_when_no_model(tmp_path: Path) -> None:
    out = tmp_path / "empty.omex"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(
            "manifest.xml",
            '<omexManifest xmlns="http://identifiers.org/combine.specifications/omex-manifest"/>',
        )
    with read_omex(out, extract_dir=tmp_path / "ex") as arch:
        assert arch.entries == []
        with pytest.raises(bngsim.ConversionError, match="no model entry"):
            arch.load_model()


# ─── net_to_omex orchestrator (acceptance: write the inverse) ───────────────


def test_net_to_omex_round_trip(data_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("libsbml")
    src = _require("two_species_reversible.net", data_dir)
    out = tmp_path / "m.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_omex(src, out, t_span=(0, 30), n_points=16)
    assert out.is_file()
    assert report.out_path == str(out)

    with read_omex(out) as arch:
        # Manifest round-trips: SBML master + SED-ML protocol + the source .net
        # bundled by default (include_source=True) as a secondary model entry, plus
        # provenance (metadata.rdf + bngsim-conversion.json) on by default.
        assert {"sbml", "sedml", "net"} <= {e.kind for e in arch.entries}
        model = arch.load_model()  # still dispatches the SBML master
        assert arch.master_model_entry().kind == "sbml"
        assert model.n_species == report.n_species
        job = arch.load_protocol()
        assert job.t_span == (0.0, 30.0)
        assert job.n_points == 16
        result = job.evaluate()
        assert result.observables.shape[0] == 16


def test_net_to_omex_embeds_source(data_dir: Path, tmp_path: Path) -> None:
    """include_source (default) bundles the original .net (secondary model entry)
    and the rule-based .bngl (provenance `source` entry) so a published archive
    carries the modeller's formulation, not just the flattened SBML. SBML stays the
    master/curated entry. include_source=False restores the SBML+SED-ML-only output."""
    pytest.importorskip("libsbml")
    src = _require("two_species_reversible.net", data_dir)
    bngl = tmp_path / "model.bngl"
    bngl.write_text('begin model\nend model\nsimulate({method=>"ode",t_end=>30,n_steps=>15})\n')
    out = tmp_path / "m.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, out, bngl=bngl, gate="none", provenance=False)

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "two_species_reversible.net" in names  # original .net, by its real name
    assert "model.bngl" in names  # rule-based source, by its real name
    with read_omex(out) as arch:
        assert {e.kind for e in arch.entries} == {"sbml", "sedml", "net", "source"}
        assert arch.master_model_entry().kind == "sbml"  # SBML still curated master

    # Opt-out → the lean SBML + SED-ML archive (no source files, no provenance).
    lean = tmp_path / "lean.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, lean, bngl=bngl, gate="none", include_source=False, provenance=False)
    with read_omex(lean) as arch:
        assert {e.kind for e in arch.entries} == {"sbml", "sedml"}


def test_net_to_omex_provenance(data_dir: Path, tmp_path: Path) -> None:
    """provenance (default) stamps how the archive was produced + the faithfulness
    verdict: a COMBINE metadata.rdf (creator=bngsim version, the given created date)
    and a bngsim-conversion.json (gate, ok, per-level L0–L4, version). The `created`
    timestamp is injectable for byte-reproducible archives; provenance=False omits
    both files."""
    pytest.importorskip("libsbml")
    import json as _json

    src = _require("two_species_reversible.net", data_dir)
    out = tmp_path / "m.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(
            src, out, gate="full", created="2026-06-28T00:00:00+00:00", t_span=(0, 30), n_points=16
        )

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {"metadata.rdf", "bngsim-conversion.json"} <= names
        rdf = zf.read("metadata.rdf").decode()
        rec = _json.loads(zf.read("bngsim-conversion.json"))
    assert f"bngsim {bngsim.__version__}" in rdf
    assert "2026-06-28T00:00:00+00:00" in rdf  # injected created date, byte-stable
    assert rec["version"] == bngsim.__version__
    assert rec["conversion"]["gate"] == "full"
    assert rec["conversion"]["ok"] is True
    assert {lv["level"] for lv in rec["conversion"]["validation"]} >= {"L0", "L3"}

    # Reproducible: same inputs + same created → byte-identical archive.
    out2 = tmp_path / "m2.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(
            src,
            out2,
            gate="full",
            created="2026-06-28T00:00:00+00:00",
            t_span=(0, 30),
            n_points=16,
        )
    assert out.read_bytes() == out2.read_bytes()

    # provenance=False omits both files.
    bare = tmp_path / "bare.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, bare, gate="none", include_source=False, provenance=False)
    with zipfile.ZipFile(bare) as zf:
        assert not ({"metadata.rdf", "bngsim-conversion.json"} & set(zf.namelist()))


def test_extract_dir_persists(data_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("libsbml")
    src = _require("two_species_reversible.net", data_dir)
    out = tmp_path / "m.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, out)
    ex = tmp_path / "extracted"
    arch = read_omex(out, extract_dir=ex)
    assert isinstance(arch, OmexArchive)
    # An explicit extract_dir is not cleaned up — files remain for inspection.
    assert (ex / "model.xml").is_file()
    assert (ex / "simulation.sedml").is_file()


# ─── CLI ────────────────────────────────────────────────────────────────────


def test_cli_pack_then_unpack(data_dir: Path, tmp_path: Path, capsys) -> None:
    pytest.importorskip("libsbml")
    from bngsim.convert._cli import omex_main

    src = _require("two_species_reversible.net", data_dir)
    out = tmp_path / "cli.omex"
    rc = omex_main(["pack", str(src), "-o", str(out), "--n-points", "11"])
    assert rc == 0
    assert out.is_file()
    assert "archive" in capsys.readouterr().out

    ex = tmp_path / "ex"
    rc = omex_main(["unpack", str(out), "-d", str(ex)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "model.xml" in captured
    assert "SED-ML" in captured
    assert (ex / "model.xml").is_file()


def test_cli_unpack_missing_file(tmp_path: Path) -> None:
    from bngsim.convert._cli import omex_main

    rc = omex_main(["unpack", str(tmp_path / "nope.omex")])
    assert rc == 2


# ─── net_to_omex carries the real .bngl protocol + gates (Option 3) ─────────


_BNGL_TWO_STAGE = (
    "begin model\nend model\ngenerate_network()\n"
    'simulate({method=>"ode",t_end=>500,n_steps=>50})\n'
    'setParameter("k",2)\n'
    'simulate({method=>"ode",t_end=>50,n_steps=>50})\n'
)


def test_net_to_omex_carries_bngl_protocol(data_dir: Path, tmp_path: Path) -> None:
    """A source .bngl makes the archive carry the modeller's WHOLE protocol
    (every simulate), not a synthesized single default — and it round-trips."""
    from bngsim.convert import read_sedml_protocol

    src = _require("simple_decay.net", data_dir)
    bngl = tmp_path / "decay.bngl"
    bngl.write_text(_BNGL_TWO_STAGE)
    out = tmp_path / "decay.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_omex(src, out, bngl=bngl, gate="full")

    assert report.ok, report.summary()
    assert report.protocol is not None and len(report.protocol.experiments) == 2
    # The gate ran over the bngl horizon.
    assert report.validation.level("L3").metrics["t_span"] == [0.0, 500.0]
    # The archived SED-ML is the multi-experiment protocol, recovered exactly.
    sedml = zipfile.ZipFile(out).read("simulation.sedml").decode()
    back = read_sedml_protocol(sedml)
    assert back.to_dict() == report.protocol.to_dict()


def test_net_to_omex_without_bngl_falls_back_to_default(data_dir: Path, tmp_path: Path) -> None:
    """No .bngl → a default uniform time course (back-compat), still gated."""
    src = _require("simple_decay.net", data_dir)
    out = tmp_path / "d.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = net_to_omex(src, out, gate="full")
    assert report.ok
    sedml = zipfile.ZipFile(out).read("simulation.sedml").decode()
    assert "<uniformTimeCourse" in sedml  # a single default course
    assert "repeatedTask" not in sedml


def test_net_to_omex_unknown_gate_raises(data_dir: Path, tmp_path: Path) -> None:
    src = _require("simple_decay.net", data_dir)
    with pytest.raises(ValueError, match="unknown gate"):
        net_to_omex(src, tmp_path / "x.omex", gate="L9")


# ─── omex_to_net: the reverse orchestrator (SBML+SED-ML → .net) ──────────────


def test_omex_to_net_round_trip(data_dir: Path, tmp_path: Path) -> None:
    """net_to_omex → omex_to_net recovers an equivalent network, and the unpack
    consumes the archive's SED-ML horizon for the gate's L3 (symmetry with the
    pack direction's bngl= horizon)."""
    from bngsim.convert import omex_to_net

    src = _require("simple_decay.net", data_dir)
    bngl = tmp_path / "decay.bngl"
    bngl.write_text(_BNGL_TWO_STAGE)
    omex = tmp_path / "decay.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, omex, bngl=bngl, gate="full")

    out = tmp_path / "recovered.net"
    report = omex_to_net(omex, out, gate="full")
    assert report.ok, report.summary()
    assert out.is_file()
    assert report.source == str(omex)
    # The SED-ML protocol carried into the archive drove the L3 horizon.
    assert report.protocol is not None
    assert report.validation.level("L3").metrics.get("horizon_source") == "sedml protocol"
    # The recovered network matches the original .net's structure.
    orig = bngsim.Model.from_net(src)
    recov = bngsim.Model.from_net(out)
    assert recov.n_species == orig.n_species
    assert recov.n_reactions == orig.n_reactions


def test_omex_to_net_unknown_gate_raises(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert import omex_to_net

    src = _require("simple_decay.net", data_dir)
    omex = tmp_path / "d.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, omex, gate="none")
    with pytest.raises(ValueError, match="unknown gate"):
        omex_to_net(omex, tmp_path / "x.net", gate="L9")


def test_cli_omex_to_net(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import omex_main

    src = _require("simple_decay.net", data_dir)
    omex = tmp_path / "decay.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, omex, gate="none")

    out = tmp_path / "decay.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        rc = omex_main(["to-net", str(omex), "-o", str(out), "--gate", "full"])
    assert rc == 0
    assert out.is_file()
    assert bngsim.Model.from_net(out).n_reactions > 0


def test_cli_omex_pack_bngl_sibling_and_gate(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import omex_main

    src = _require("simple_decay.net", data_dir)
    net_copy = tmp_path / "decay.net"
    net_copy.write_text(src.read_text())
    (tmp_path / "decay.bngl").write_text(_BNGL_TWO_STAGE)
    out = tmp_path / "decay.omex"
    rc = omex_main(["pack", str(net_copy), "-o", str(out), "--gate", "full", "--quiet"])
    assert rc == 0
    # The sibling .bngl was picked up → multi-experiment protocol in the archive.
    sedml = zipfile.ZipFile(out).read("simulation.sedml").decode()
    assert sedml.count("<uniformTimeCourse") == 2


def test_cli_omex_pack_missing_bngl_exits_two(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import omex_main

    src = _require("simple_decay.net", data_dir)
    rc = omex_main(
        ["pack", str(src), "-o", str(tmp_path / "x.omex"), "--bngl", str(tmp_path / "no.bngl")]
    )
    assert rc == 2


# ─── No real protocol → warn + label the synthesized default (GH #211) ──────


def test_net_to_omex_no_protocol_warns_and_labels(data_dir: Path, tmp_path: Path) -> None:
    """No .bngl → a default protocol is still bundled (runnable), but a
    ConversionWarning fires and the SED-ML is marked synthesized so a consumer
    can never mistake the placeholder for the modeller's protocol."""
    src = _require("simple_decay.net", data_dir)
    out = tmp_path / "d.omex"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net_to_omex(src, out, gate="none")
    msgs = [str(w.message) for w in caught if issubclass(w.category, bngsim.ConversionWarning)]
    assert any("NOT the modeller's protocol" in m for m in msgs), msgs
    sed = zipfile.ZipFile(out).read("simulation.sedml").decode()
    assert "synthesizedDefault" in sed
    assert "synthesized" in sed  # the distinguishing report name


def test_net_to_omex_empty_actions_bngl_warns_and_labels(data_dir: Path, tmp_path: Path) -> None:
    src = _require("simple_decay.net", data_dir)
    bngl = tmp_path / "x.bngl"
    bngl.write_text("begin model\nend model\ngenerate_network()\n")  # no simulate
    out = tmp_path / "e.omex"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net_to_omex(src, out, bngl=bngl, gate="none")
    msgs = [str(w.message) for w in caught if issubclass(w.category, bngsim.ConversionWarning)]
    assert any("no simulate action" in m for m in msgs), msgs
    assert "synthesizedDefault" in zipfile.ZipFile(out).read("simulation.sedml").decode()


def test_net_to_omex_real_protocol_not_labeled(data_dir: Path, tmp_path: Path) -> None:
    """A real .bngl protocol is carried verbatim — no synthesized marker, no
    fabrication warning."""
    src = _require("simple_decay.net", data_dir)
    bngl = tmp_path / "x.bngl"
    bngl.write_text('begin model\nend model\nsimulate({method=>"ode",t_end=>400,n_steps=>40})\n')
    out = tmp_path / "r.omex"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net_to_omex(src, out, bngl=bngl, gate="none")
    msgs = [str(w.message) for w in caught if issubclass(w.category, bngsim.ConversionWarning)]
    assert not any("synthesized" in m or "placeholder" in m for m in msgs), msgs
    assert "synthesizedDefault" not in zipfile.ZipFile(out).read("simulation.sedml").decode()


def test_synthesized_default_round_trips_warning_on_read(data_dir: Path) -> None:
    """The synthesized marker is actionable: read_sedml warns a bngsim consumer
    that the protocol is a fabricated placeholder."""
    from bngsim.convert import default_protocol, read_sedml, write_sedml

    src = _require("simple_decay.net", data_dir)
    spec = default_protocol(src, model_source="model.xml", model_format="sbml")
    text = write_sedml(spec, model_source="model.xml", synthesized_default=True)
    assert "synthesizedDefault" in text
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_sedml(text)
    assert any(
        "synthesized-DEFAULT" in str(w.message)
        for w in caught
        if issubclass(w.category, bngsim.ConversionWarning)
    )


# ─── omex_to_net: multi-experiment / multi-file protocol → .bngl actions (#222) ─


def test_omex_to_net_emits_actions_block(data_dir: Path, tmp_path: Path) -> None:
    """omex_to_net composes the archive's WHOLE protocol and emits it as a .bngl
    actions block alongside the .net (every experiment, re-parseable)."""
    from bngsim.convert import omex_to_net, parse_bngl_protocol

    src = _require("simple_decay.net", data_dir)
    bngl = tmp_path / "decay.bngl"
    bngl.write_text(_BNGL_TWO_STAGE)
    omex = tmp_path / "decay.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, omex, bngl=bngl, gate="none")

    out = tmp_path / "recovered.net"
    report = omex_to_net(omex, out, gate="none")
    # A .bngl actions sidecar landed next to the .net by default.
    assert report.bngl_out == str(out.with_suffix(".bngl"))
    actions = Path(report.bngl_out)
    assert actions.is_file()
    text = actions.read_text()
    assert text.count("simulate(") == 2
    assert "begin actions" in text and "end actions" in text
    # The emitted block re-parses to the same multi-experiment protocol.
    reparsed = parse_bngl_protocol(actions)
    assert len(reparsed.experiments) == 2
    assert reparsed.to_dict()["steps"] == report.protocol.to_dict()["steps"]


def test_omex_to_net_no_actions_suppresses(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert import omex_to_net

    src = _require("simple_decay.net", data_dir)
    bngl = tmp_path / "decay.bngl"
    bngl.write_text(_BNGL_TWO_STAGE)
    omex = tmp_path / "decay.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, omex, bngl=bngl, gate="none")

    out = tmp_path / "recovered.net"
    report = omex_to_net(omex, out, gate="none", write_actions=False)
    assert report.bngl_out is None
    assert not out.with_suffix(".bngl").exists()
    # The protocol is still composed and attached, just not written.
    assert report.protocol is not None and len(report.protocol.experiments) == 2


def test_omex_to_net_composes_multiple_sedml_files(data_dir: Path, tmp_path: Path) -> None:
    """An archive carrying more than one SED-ML file composes EVERY experiment
    (not just the master), warns that it did, and separates the files with a
    resetConcentrations boundary."""
    from bngsim.convert import (
        Experiment,
        ProtocolSpec,
        omex_to_net,
        write_omex,
        write_sedml_protocol,
    )

    src = _require("simple_decay.net", data_dir)
    # Borrow a valid SBML for the model channel from a packed archive.
    seed = tmp_path / "seed.omex"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        net_to_omex(src, seed, gate="none")
    sbml_text = zipfile.ZipFile(seed).read("model.xml").decode()

    s1 = write_sedml_protocol(
        ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 10.0), 11),)),
        model_source="model.xml",
    )
    s2 = write_sedml_protocol(
        ProtocolSpec(steps=(Experiment("simulate", "ode", (0.0, 50.0), 51),)),
        model_source="model.xml",
    )
    omex = tmp_path / "multi.omex"
    write_omex(
        omex,
        sbml=sbml_text,
        sedml=s1,
        extra=[("./sim2.sedml", s2, _FMT_SEDML)],
        sbml_location="./model.xml",
        master="sbml",
    )

    out = tmp_path / "multi.net"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = omex_to_net(omex, out, gate="none")
    msgs = [str(w.message) for w in caught if issubclass(w.category, bngsim.ConversionWarning)]
    assert any("2 SED-ML files" in m for m in msgs), msgs

    assert len(report.protocol.experiments) == 2
    actions = Path(report.bngl_out).read_text()
    assert actions.count("simulate(") == 2
    assert "resetConcentrations()" in actions  # independent-file boundary


def test_cli_omex_to_net_actions(data_dir: Path, tmp_path: Path) -> None:
    from bngsim.convert._cli import omex_main

    src = _require("simple_decay.net", data_dir)
    (tmp_path / "decay.bngl").write_text(_BNGL_TWO_STAGE)
    omex = tmp_path / "decay.omex"
    net_copy = tmp_path / "decay.net"
    net_copy.write_text(src.read_text())
    rc = omex_main(["pack", str(net_copy), "-o", str(omex), "--gate", "none", "--quiet"])
    assert rc == 0

    # Default: the actions block is emitted next to the .net.
    out = tmp_path / "decay_out.net"
    rc = omex_main(["to-net", str(omex), "-o", str(out), "--gate", "none", "--quiet"])
    assert rc == 0
    assert out.with_suffix(".bngl").is_file()

    # --no-actions suppresses it; --actions-out redirects it.
    out2 = tmp_path / "decay_out2.net"
    rc = omex_main(
        ["to-net", str(omex), "-o", str(out2), "--gate", "none", "--no-actions", "--quiet"]
    )
    assert rc == 0
    assert not out2.with_suffix(".bngl").exists()

    out3 = tmp_path / "decay_out3.net"
    redirect = tmp_path / "custom_actions.bngl"
    rc = omex_main(
        [
            "to-net",
            str(omex),
            "-o",
            str(out3),
            "--gate",
            "none",
            "--actions-out",
            str(redirect),
            "--quiet",
        ]
    )
    assert rc == 0
    assert redirect.is_file() and not out3.with_suffix(".bngl").exists()
