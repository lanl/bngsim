"""SBML Level 3 packages declared ``required="true"`` (issues #592, #613).

SBML defines ``required="true"`` on a package namespace to mean the package
changes the *mathematical meaning* of the model. bngsim interprets exactly one
package — ``comp``, which it flattens (GH #230) — and used to read past every
other one as if it were not there: a ``multi`` document (the package rule-based
models are encoded in) loaded as its bare core layer, with no exception and no
warning, and the caller got a simulation of a model that is not the one on disk.

The loader now refuses any ``required="true"`` package it does not account for,
by name. These tests pin both halves of that boundary: what is refused, and the
much larger set that must keep loading untouched — every SBML Level 2 document,
every plain L3V2 document, the presentation-only packages, and the two packages
bngsim does account for. The two traps in the middle are libSBML's doing and
each cost a corpus: on an L2 document it reports its annotation-based ``layout``
/ ``render`` plugins as *required*, and on any L3V2 document it reports the
``l3v2extendedmath`` plugin — under the core namespace — as required too.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import bngsim
import libsbml
import numpy as np
import pytest
from bngsim._exceptions import ModelError
from bngsim._sbml_loader import (
    _ALLOW_UNSUPPORTED_ENV,
    _check_submodel_required_packages,
    _comp_submodel_documents,
    _flatten_comp,
)

_CORE_L3V1 = "http://www.sbml.org/sbml/level3/version1/core"
_CORE_L3V2 = "http://www.sbml.org/sbml/level3/version2/core"

#: A decaying species, as core SBML — the *whole* model in a core-only document,
#: and the misleading remainder in a package document whose real content is the
#: layer above it.
_CORE_BODY = """
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialAmount="100" hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
"""


def _with_package(prefix: str, uri: str, required: str, core: str = _CORE_L3V1) -> str:
    """A loadable core model in a document that also declares one package."""
    level, version = ("3", "2") if core == _CORE_L3V2 else ("3", "1")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<sbml xmlns="{core}" xmlns:{prefix}="{uri}" '
        f'level="{level}" version="{version}" {prefix}:required="{required}">\n'
        f'  <model id="m">{_CORE_BODY}</model>\n'
        "</sbml>\n"
    )


_MULTI_URI = "http://www.sbml.org/sbml/level3/version1/multi/version1"
_QUAL_URI = "http://www.sbml.org/sbml/level3/version1/qual/version1"
# A package libSBML 5.21 ships no extension for: it lands in the document's
# "unknown packages" list rather than among its plugins, which is a second place
# the guard has to look.
_ARRAYS_URI = "http://www.sbml.org/sbml/level3/version1/arrays/version1"


# ─── Refused ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("prefix", "uri"),
    [("multi", _MULTI_URI), ("qual", _QUAL_URI), ("arrays", _ARRAYS_URI)],
)
def test_a_required_package_is_refused_by_name(prefix: str, uri: str) -> None:
    """Including ``arrays``, which this libSBML has no extension for — an
    unregistered package is still recorded with its ``required`` flag, and
    reading past it is the same mistake as reading past a known one."""
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml_string(_with_package(prefix, uri, "true"))
    msg = str(excinfo.value)
    assert prefix in msg and uri in msg
    assert 'required="true"' in msg
    assert _ALLOW_UNSUPPORTED_ENV in msg


def test_the_refusal_names_what_bngsim_does_account_for() -> None:
    """A refusal that only says no is a dead end: the message has to say which
    packages do load, and that a presentation-only one is not affected."""
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml_string(_with_package("multi", _MULTI_URI, "true"))
    msg = str(excinfo.value)
    assert "comp" in msg and "distrib" in msg
    assert "layout" in msg


def test_the_file_entry_point_guards_too(tmp_path) -> None:
    """``load_sbml`` and ``load_sbml_string`` are separate entry points, and the
    documented way in (``Model.from_sbml``) is the file one."""
    path = tmp_path / "multi.xml"
    path.write_text(_with_package("multi", _MULTI_URI, "true"))
    with pytest.raises(ModelError, match="multi"):
        bngsim.Model.from_sbml(path)


def test_the_core_layer_of_a_refused_document_is_not_the_model() -> None:
    """Why this is a refusal and not a warning.

    The core layer here is a *complete, valid, simulable* model — 1 species, a
    first-order decay — and it is not what the document says. Before the guard
    this returned a model and a trajectory with nothing to indicate that the
    ``multi`` layer above it (species types, feature values, component maps) had
    been dropped. The same shape as a real qual model, whose core
    ``<listOfSpecies>`` is empty and which loaded as a model with 0 species.
    """
    core_only = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<sbml xmlns="{_CORE_L3V1}" level="3" version="1">\n'
        f'  <model id="m">{_CORE_BODY}</model>\n'
        "</sbml>\n"
    )
    assert bngsim.Model.from_sbml_string(core_only).species_names == ["A"]
    with pytest.raises(ModelError):
        bngsim.Model.from_sbml_string(_with_package("multi", _MULTI_URI, "true"))


# ─── Loads untouched ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("prefix", "uri"),
    [
        ("layout", "http://www.sbml.org/sbml/level3/version1/layout/version1"),
        ("render", "http://www.sbml.org/sbml/level3/version1/render/version1"),
        ("fbc", "http://www.sbml.org/sbml/level3/version2/fbc/version2"),
    ],
)
def test_a_presentation_package_is_not_refused(prefix: str, uri: str) -> None:
    """``required="false"`` is the package saying it does not change the math,
    and keying on that attribute rather than on a list of package names is what
    makes the guard general without touching these."""
    model = bngsim.Model.from_sbml_string(_with_package(prefix, uri, "false"))
    assert model.species_names == ["A"]


def test_a_plain_l3v2_document_is_not_refused() -> None:
    """libSBML models the extended math that L3V2 folded into core as a plugin
    named ``l3v2extendedmath``, carrying the *core* namespace and reported as
    required. Every L3V2 document has it, so a guard that took it at its word
    would refuse the entire corpus."""
    core_only = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<sbml xmlns="{_CORE_L3V2}" level="3" version="2">\n'
        f'  <model id="m">{_CORE_BODY}</model>\n'
        "</sbml>\n"
    )
    assert bngsim.Model.from_sbml_string(core_only).species_names == ["A"]


def test_a_level_2_document_is_not_refused() -> None:
    """``required`` is a Level 3 attribute; L2 packages are annotations. libSBML
    attaches its L2 ``layout`` / ``render`` plugins to *every* L2 document and
    answers ``getPackageRequired`` True for both, so the guard has to be scoped
    to L3 or it refuses every L2 model — BIOMD0000000003 among them."""
    l2 = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">\n'
        f'  <model id="m">{_CORE_BODY}</model>\n'
        "</sbml>\n"
    )
    assert bngsim.Model.from_sbml_string(l2).species_names == ["A"]


def test_a_handled_package_still_loads() -> None:
    """``comp`` and ``distrib`` are the handled list, and both declare
    ``required="true"``. ``comp`` is flattened (GH #230); ``distrib``'s
    math-changing content — its random-draw csymbols — is refused by the MathML
    translator by name (GH #97), and the rest of it is uncertainty annotation
    that never enters the integrated system. Neither is read past in silence,
    which is the only thing this guard exists to stop."""
    for prefix, uri in (
        ("comp", "http://www.sbml.org/sbml/level3/version1/comp/version1"),
        ("distrib", "http://www.sbml.org/sbml/level3/version1/distrib/version1"),
    ):
        model = bngsim.Model.from_sbml_string(_with_package(prefix, uri, "true"))
        assert model.species_names == ["A"], prefix


# ─── Non-standard namespace blind spot (issue #613) ─────────────────────────

#: A namespace outside http://www.sbml.org/sbml/level3/… — libSBML drops it
#: entirely: no plugin, no unknown-package entry, no error.
_ZZZ_URI = "http://example.com/zzz/version1"


def test_a_nonstandard_required_namespace_is_refused(tmp_path) -> None:
    """A ``required="true"`` attribute on a namespace outside the standard SBML
    L3 hierarchy is invisible to libSBML's plugin and unknown-package APIs
    (issue #613). The loader must still refuse it rather than silently loading
    the core layer as if the namespace were not there."""
    xml = _with_package("zzz", _ZZZ_URI, "true")
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml_string(xml)
    msg = str(excinfo.value)
    assert _ZZZ_URI in msg
    assert 'required="true"' in msg
    assert _ALLOW_UNSUPPORTED_ENV in msg


def test_a_nonstandard_required_false_namespace_is_not_refused() -> None:
    """``required="false"`` on a non-standard namespace means the same thing it
    does on a standard one: the package does not change the math, and the core
    layer is the model."""
    xml = _with_package("zzz", _ZZZ_URI, "false")
    model = bngsim.Model.from_sbml_string(xml)
    assert model.species_names == ["A"]


def test_nonstandard_required_namespace_opt_out(monkeypatch, caplog) -> None:
    """``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`` opts out of the refusal for
    non-standard namespaces the same as for standard ones."""
    monkeypatch.setenv(_ALLOW_UNSUPPORTED_ENV, "1")
    xml = _with_package("zzz", _ZZZ_URI, "true")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        model = bngsim.Model.from_sbml_string(xml)
    assert model.species_names == ["A"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(_ZZZ_URI in w and _ALLOW_UNSUPPORTED_ENV in w for w in warnings), warnings


def test_the_file_entry_point_catches_nonstandard_namespace(tmp_path) -> None:
    """The file entry point (``Model.from_sbml``) must also catch non-standard
    required namespaces, not only the string entry point."""
    path = tmp_path / "zzz.xml"
    path.write_text(_with_package("zzz", _ZZZ_URI, "true"))
    with pytest.raises(ModelError, match=_ZZZ_URI):
        bngsim.Model.from_sbml(path)


def test_the_raw_scan_reads_only_the_root_element() -> None:
    """The #613 scan must cost the same on a 20 MB document as on a 2 kB one:
    it reads as far as the ``<sbml>`` start tag and stops, rather than taking a
    second full copy of a document libSBML has already read."""
    import io

    from bngsim._sbml_loader import _raw_required_namespaces

    class _Counting(io.BytesIO):
        consumed = 0

        def read(self, size=-1):
            data = super().read(size)
            _Counting.consumed += len(data)
            return data

    padding = b"<!-- " + b"x" * (20 * 1024 * 1024) + b" -->"
    stream = _Counting(_with_package("zzz", _ZZZ_URI, "true").encode() + padding)
    assert _raw_required_namespaces(stream) == {_ZZZ_URI: _ZZZ_URI}
    assert _Counting.consumed < 1024 * 1024


# ─── Opt-out ────────────────────────────────────────────────────────────────


def test_the_opt_out_loads_the_core_layer_and_says_so(monkeypatch, caplog) -> None:
    """Same escape hatch as the delay/AlgebraicRule refusals, and the same
    meaning: the caller is asking for a model that is not the one on disk, and
    the log says which package was dropped to produce it."""
    monkeypatch.setenv(_ALLOW_UNSUPPORTED_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        model = bngsim.Model.from_sbml_string(_with_package("multi", _MULTI_URI, "true"))
    assert model.species_names == ["A"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("multi" in w and _ALLOW_UNSUPPORTED_ENV in w for w in warnings), warnings


# ─── A package in a comp submodel's own file (issue #1786) ──────────────────
#
# A comp ExternalModelDefinition pulls a model in from another file, and the
# flattener strips any package it cannot flatten from it, so the guard above,
# run on the loaded document alone, let a ``required="true"`` package in that
# file through: the submodel's core layer loaded and ran with no warning.

_COMP_URI = "http://www.sbml.org/sbml/level3/version1/comp/version1"


def _leaf(k: float, prefix: str | None = None, uri: str | None = None) -> str:
    """The decaying species at rate *k* as a document of its own (model id
    ``m``), declaring the package *prefix* / *uri* ``required="true"`` if given."""
    body = _CORE_BODY.replace('value="0.1"', f'value="{k}"')
    if prefix is None:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<sbml xmlns="{_CORE_L3V1}" level="3" version="1">\n'
            f'  <model id="m">{body}</model>\n'
            "</sbml>\n"
        )
    return _with_package(prefix, uri, "true").replace(_CORE_BODY, body)


def _comp(external=(), submodels=(), definitions: str = "") -> str:
    """A comp document. *external* holds ``(id, source, modelRef)`` for each
    ExternalModelDefinition, *submodels* ``(id, modelRef)`` for each submodel of
    its main model (id ``top``), and *definitions* its ModelDefinitions."""
    emds = "".join(
        f'<comp:externalModelDefinition comp:id="{i}" comp:source="{s}" comp:modelRef="{r}"/>'
        for i, s, r in external
    )
    subs = "".join(f'<comp:submodel comp:id="{i}" comp:modelRef="{r}"/>' for i, r in submodels)

    def listed(tag: str, items: str) -> str:  # an L3V1 listOf may not be empty
        return f"<comp:listOf{tag}>{items}</comp:listOf{tag}>" if items else ""

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<sbml xmlns="{_CORE_L3V1}" xmlns:comp="{_COMP_URI}" level="3" version="1" '
        'comp:required="true">\n'
        f"  {listed('ExternalModelDefinitions', emds)}{listed('ModelDefinitions', definitions)}\n"
        f'  <model id="top">{listed("Submodels", subs)}</model>\n'
        "</sbml>\n"
    )


def _definition(def_id: str, k: float) -> str:
    """The decaying species at rate *k* as a ModelDefinition."""
    body = _CORE_BODY.replace('value="0.1"', f'value="{k}"')
    return f'<comp:modelDefinition id="{def_id}">{body}</comp:modelDefinition>'


def _assert_decays_at(model, k: float) -> None:
    """*model* is the one species decaying at rate *k*: A(t) = 100·e^(−k·t)."""
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=11, rtol=1e-10, atol=1e-10
    )
    t = np.asarray(res.time)
    np.testing.assert_allclose(np.asarray(res.species)[:, 0], 100 * np.exp(-k * t), rtol=1e-6)


def _submodel_of(source: str) -> str:
    """A comp document whose one submodel is the model ``m`` of *source*."""
    return _comp(external=[("ext", source, "m")], submodels=[("sub", "ext")])


def _found_by_the_last_resort(root, k: float, prefix=None, uri=None):
    """``top.xml``, whose ``models/mid.xml`` names an ``ext.xml`` that exists
    only beside ``top.xml``: the flattener finds it in the loaded file's
    directory, its last resort. The model's rate is *k*, parameter
    ``sub__sub__k``."""
    (root / "models").mkdir(parents=True)
    (root / "ext.xml").write_text(_leaf(k, prefix, uri))
    (root / "models" / "mid.xml").write_text(_submodel_of("ext.xml"))
    (root / "top.xml").write_text(
        _comp(external=[("mid", "models/mid.xml", "top")], submodels=[("sub", "mid")])
    )
    return root / "top.xml"


def test_a_required_package_in_a_submodel_s_file_is_refused(tmp_path) -> None:
    """The issue's case. ``ext.xml`` declaring ``multi`` is refused on its own,
    and ``top.xml``, whose one submodel is ``ext.xml``'s model, loaded that
    model's core layer and ran it. The refusal names the file and the
    ExternalModelDefinition that reads it, since neither is the file passed."""
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(_submodel_of("ext.xml"))
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    msg = str(excinfo.value)
    assert "multi" in msg and _MULTI_URI in msg
    assert str(tmp_path / "ext.xml") in msg and "'ext'" in msg
    assert 'required="true"' in msg and _ALLOW_UNSUPPORTED_ENV in msg
    assert excinfo.value.code == "model.unsupported_construct"


@pytest.mark.parametrize("prefix", [None, "layout"])
def test_a_submodel_s_file_with_no_required_package_still_loads(tmp_path, prefix) -> None:
    """The other side: a file that declares nothing, or a ``required="false"``
    presentation package, flattens into the composed model as before."""
    if prefix is None:
        ext = _leaf(0.1)
    else:
        uri = "http://www.sbml.org/sbml/level3/version1/layout/version1"
        ext = _with_package(prefix, uri, "false")
    (tmp_path / "ext.xml").write_text(ext)
    (tmp_path / "top.xml").write_text(_submodel_of("ext.xml"))
    model = bngsim.Model.from_sbml(tmp_path / "top.xml")
    assert model.species_names == ["sub__A"]
    _assert_decays_at(model, 0.1)


def test_a_nested_submodel_s_file_is_the_one_flattening_reads(tmp_path, monkeypatch) -> None:
    """A source resolves against the document that names it, as the flattener
    resolves it: ``models/mid.xml``'s ``ext.xml`` is ``models/ext.xml``, not the
    clean ``ext.xml`` beside ``top.xml``. The opt-out run shows which one the
    flattened model is made of."""
    (tmp_path / "models").mkdir()
    (tmp_path / "ext.xml").write_text(_leaf(0.5))
    (tmp_path / "models" / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "models" / "mid.xml").write_text(_submodel_of("ext.xml"))
    (tmp_path / "top.xml").write_text(
        _comp(external=[("mid", "models/mid.xml", "top")], submodels=[("sub", "mid")])
    )
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    assert str(tmp_path / "models" / "ext.xml") in str(excinfo.value)

    monkeypatch.setenv(_ALLOW_UNSUPPORTED_ENV, "1")
    _assert_decays_at(bngsim.Model.from_sbml(tmp_path / "top.xml"), 0.1)


def test_a_source_found_only_beside_the_loaded_file_is_checked(tmp_path, monkeypatch) -> None:
    """The flattener's last resort for a source is the loaded file's own
    directory: ``models/mid.xml`` names an ``ext.xml`` that exists only beside
    ``top.xml``, and that is the one it reads."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")  # nothing to find in the working directory
    top = _found_by_the_last_resort(tmp_path, 0.3, "multi", _MULTI_URI)
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(top)
    assert str(tmp_path / "ext.xml") in str(excinfo.value)

    monkeypatch.setenv(_ALLOW_UNSUPPORTED_ENV, "1")
    _assert_decays_at(bngsim.Model.from_sbml(top), 0.3)


def test_a_chain_of_external_model_definitions_is_followed(tmp_path) -> None:
    """An ExternalModelDefinition can name another one: ``top.xml``'s points at
    ``a.xml``'s, which points at ``ext.xml``'s model."""
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "a.xml").write_text(_comp(external=[("x", "ext.xml", "m")]))
    (tmp_path / "top.xml").write_text(
        _comp(external=[("a", "a.xml", "x")], submodels=[("sub", "a")])
    )
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    assert str(tmp_path / "ext.xml") in str(excinfo.value)


def test_a_relative_path_is_checked_like_an_absolute_one(tmp_path, monkeypatch) -> None:
    """libSBML records a relative path as a relative location, and resolves
    sources against it from the working directory."""
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(_submodel_of("ext.xml"))
    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(ModelError, match="multi"):
        bngsim.Model.from_sbml(f"{tmp_path.name}/top.xml")


def test_the_string_entry_point_checks_files_in_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    """A string has no location, so libSBML resolves its sources against the
    working directory: ``from_sbml_string`` flattens a file found there, and
    has to check it too."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ext.xml").write_text(_leaf(0.1))
    _assert_decays_at(bngsim.Model.from_sbml_string(_submodel_of("ext.xml")), 0.1)

    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    with pytest.raises(ModelError, match="multi"):
        bngsim.Model.from_sbml_string(_submodel_of("ext.xml"))


def test_a_nonstandard_required_namespace_in_a_submodel_s_file_is_refused(tmp_path) -> None:
    """The #613 scan of the root element runs on the submodel's file as well:
    libSBML keeps no trace of a namespace outside the L3 hierarchy."""
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "zzz", _ZZZ_URI))
    (tmp_path / "top.xml").write_text(_submodel_of("ext.xml"))
    with pytest.raises(ModelError, match=_ZZZ_URI):
        bngsim.Model.from_sbml(tmp_path / "top.xml")


def test_an_external_model_definition_no_submodel_uses_is_not_checked(tmp_path) -> None:
    """The flattener instantiates what the main model's submodels reach and
    reads no other file, so a ``multi`` file behind an unused
    ExternalModelDefinition is no part of the model and does not refuse it."""
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(
        _comp(
            external=[("unused", "ext.xml", "m")],
            submodels=[("sub", "d")],
            definitions=_definition("d", 0.2),
        )
    )
    _assert_decays_at(bngsim.Model.from_sbml(tmp_path / "top.xml"), 0.2)


def test_a_file_that_names_itself_loads(tmp_path) -> None:
    """An ExternalModelDefinition can point back at its own file, for one of
    its ModelDefinitions. That file was checked as the one loaded."""
    (tmp_path / "top.xml").write_text(
        _comp(
            external=[("me", "top.xml", "d")],
            submodels=[("sub", "me")],
            definitions=_definition("d", 0.2),
        )
    )
    _assert_decays_at(bngsim.Model.from_sbml(tmp_path / "top.xml"), 0.2)


def test_a_cycle_of_files_ends_in_the_flattener_s_refusal(tmp_path) -> None:
    """``a.xml``'s submodel is ``b.xml``'s model and ``b.xml``'s is ``a.xml``'s.
    The check stops going round, and the flattener refuses the cycle as before."""
    (tmp_path / "a.xml").write_text(
        _comp(external=[("b", "b.xml", "top")], submodels=[("sb", "b")])
    )
    (tmp_path / "b.xml").write_text(
        _comp(external=[("a", "a.xml", "top")], submodels=[("sa", "a")])
    )
    with pytest.raises(ModelError, match="comp flattening failed"):
        bngsim.Model.from_sbml(tmp_path / "a.xml")


def test_a_source_that_does_not_resolve_is_left_to_the_flattener(tmp_path) -> None:
    """Nothing to check, and the flattener's own refusal is unchanged."""
    (tmp_path / "top.xml").write_text(_submodel_of("missing.xml"))
    with pytest.raises(ModelError, match="comp flattening failed"):
        bngsim.Model.from_sbml(tmp_path / "top.xml")


def test_the_opt_out_covers_a_submodel_s_file(tmp_path, monkeypatch, caplog) -> None:
    """``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`` loads the submodel's core layer,
    as it does a loaded file's, and the warning names the submodel's file."""
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(_submodel_of("ext.xml"))
    monkeypatch.setenv(_ALLOW_UNSUPPORTED_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        model = bngsim.Model.from_sbml(tmp_path / "top.xml")
    _assert_decays_at(model, 0.1)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "multi" in w and _ALLOW_UNSUPPORTED_ENV in w and str(tmp_path / "ext.xml") in w
        for w in warnings
    ), warnings


def _holding(def_id: str, submodel_ref: str) -> str:
    """A ModelDefinition whose one submodel is *submodel_ref*."""
    return (
        f'<comp:modelDefinition id="{def_id}"><comp:listOfSubmodels>'
        f'<comp:submodel comp:id="inner" comp:modelRef="{submodel_ref}"/>'
        "</comp:listOfSubmodels></comp:modelDefinition>"
    )


@pytest.mark.parametrize("layout", ["in the loaded file", "in the submodel's file"])
def test_a_model_definition_whose_submodel_reads_a_file_is_followed(tmp_path, layout) -> None:
    """The commonest comp layout puts the submodel that reads another file in a
    ModelDefinition rather than in the main model. In the loaded file, the main
    model's submodel is ``d``, which holds a submodel of ``ext.xml``. In a
    submodel's file, ``sub/ext.xml``'s ModelDefinition holds one of
    ``inner.xml``, which is ``sub/inner.xml`` and not the clean one beside
    ``top.xml``."""
    if layout == "in the loaded file":
        offender = tmp_path / "ext.xml"
        top = _comp(
            external=[("ext", "ext.xml", "m")],
            submodels=[("sub", "d")],
            definitions=_holding("d", "ext"),
        )
    else:
        (tmp_path / "sub").mkdir()
        offender = tmp_path / "sub" / "inner.xml"
        (tmp_path / "inner.xml").write_text(_leaf(0.9))
        (tmp_path / "sub" / "ext.xml").write_text(
            _comp(external=[("x", "inner.xml", "m")], definitions=_holding("md", "x"))
        )
        top = _comp(external=[("e", "sub/ext.xml", "md")], submodels=[("sub", "e")])
    offender.write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(top)
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    assert str(offender) in str(excinfo.value)


symlinks_posix = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-specific: creating a symlink needs privileges"
)


@symlinks_posix
def test_a_file_reached_two_ways_is_walked_from_each(tmp_path) -> None:
    """A source resolves against the location its file was reached by, not
    where the file is. ``d1/ext.xml`` is a link to ``d2/ext.xml``, whose
    submodel reads ``inner.xml``: through ``d1`` that is the clean
    ``d1/inner.xml``, through ``d2`` the ``multi`` one in ``d2``. ``top.xml``
    has a submodel through each, so the flattener reads both."""
    for d in ("d1", "d2"):
        (tmp_path / d).mkdir()
    (tmp_path / "d2" / "ext.xml").write_text(
        _comp(external=[("x", "inner.xml", "m")], submodels=[("sx", "x")])
    )
    os.symlink(os.path.join("..", "d2", "ext.xml"), tmp_path / "d1" / "ext.xml")
    (tmp_path / "d1" / "inner.xml").write_text(_leaf(0.1))
    (tmp_path / "d2" / "inner.xml").write_text(_leaf(0.2, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(
        _comp(
            external=[("a", "d1/ext.xml", "top"), ("b", "d2/ext.xml", "top")],
            submodels=[("s1", "a"), ("s2", "b")],
        )
    )
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    assert str(tmp_path / "d2" / "inner.xml") in str(excinfo.value)


@symlinks_posix
@pytest.mark.parametrize("load", ["relative", "absolute"])
def test_a_source_through_a_linked_directory_is_the_file_the_os_opens(
    tmp_path, monkeypatch, load
) -> None:
    """``cwd/lnk`` is a link to ``real/proj``, and ``top.xml`` there names
    ``../shared/ext.xml``. The OS follows the link before it applies ``..``, so
    the file is ``real/shared/ext.xml``; cancelling ``..`` against the path's
    text gives ``cwd/shared/ext.xml``, which does not exist. Loaded as
    ``lnk/top.xml`` the source is found only by the last resort, the loaded
    file's directory, and the refusal names the file that was read."""
    for d in ("real/proj", "real/shared", "cwd"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "real" / "shared" / "ext.xml").write_text(_leaf(0.7, "multi", _MULTI_URI))
    (tmp_path / "real" / "proj" / "top.xml").write_text(_submodel_of("../shared/ext.xml"))
    os.symlink(tmp_path / "real" / "proj", tmp_path / "cwd" / "lnk")
    monkeypatch.chdir(tmp_path / "cwd")
    top = "lnk/top.xml" if load == "relative" else str(tmp_path / "cwd" / "lnk" / "top.xml")
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(top)
    assert os.path.realpath(tmp_path / "real" / "shared" / "ext.xml") in str(excinfo.value)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-specific: the decoy path is built from a POSIX absolute path",
)
def test_the_last_resort_s_file_is_read_even_where_the_plugin_finds_another(
    tmp_path, monkeypatch
) -> None:
    """Asked for an absolute path, libSBML's file resolver first tries it below
    the naming document's directory, joined with no separator: for
    ``models/mid.xml`` naming ``ext.xml`` that only the last resort finds as
    ``<root>/ext.xml``, the plugin would read ``models/<root>/ext.xml``. The
    flattener reads ``<root>/ext.xml``, so that is the file checked."""
    top = _found_by_the_last_resort(tmp_path, 0.7, "multi", _MULTI_URI)
    decoy = tmp_path / "models" / str(tmp_path).lstrip("/") / "ext.xml"
    decoy.parent.mkdir(parents=True)
    decoy.write_text(_leaf(0.1))
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(top)
    assert str(tmp_path / "ext.xml") in str(excinfo.value)


@pytest.mark.parametrize(
    "found",
    [
        "beside the file that names it",
        "by the last resort",
        "by the last resort, from a relative path",
    ],
)
def test_the_flattener_builds_from_the_documents_checked(tmp_path, monkeypatch, found) -> None:
    """The comp plugin keeps each document it reads, and the flattener's own
    lookups get those same documents back. So the model is made of what was
    checked: a file rewritten between the check and the flattening, here to
    declare ``multi``, is not read again. A relative load path is the case
    where the walk's lookup and the flattener's could key the same file
    differently."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    ext = tmp_path / "ext.xml"
    if found == "beside the file that names it":
        ext.write_text(_leaf(0.1))
        top, k = tmp_path / "top.xml", "sub__k"
        top.write_text(_submodel_of("ext.xml"))
    elif found == "by the last resort":
        top, k = _found_by_the_last_resort(tmp_path, 0.1), "sub__sub__k"
    else:
        _found_by_the_last_resort(tmp_path / "proj", 0.1)
        monkeypatch.chdir(tmp_path)
        ext, top, k = tmp_path / "proj" / "ext.xml", Path("proj", "top.xml"), "sub__sub__k"
    doc = libsbml.readSBMLFromFile(str(top))
    _check_submodel_required_packages(doc, top, str(top.parent))
    ext.write_text(_leaf(0.9, "multi", _MULTI_URI))
    _flatten_comp(doc, base_path=str(top.parent))
    assert doc.getModel().getParameter(k).getValue() == 0.1


def test_the_check_puts_nothing_in_libsbml_s_resolver_registry(tmp_path, monkeypatch) -> None:
    """The registry is process-wide. The flattener adds its last resort to it
    and takes it out inside one call, which holds the GIL; a resolver the check
    added between two calls would be there for another thread's load to find
    files through, the other thread's or its own. The registry is sampled at
    every call the check makes, on a file only the last resort finds."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    top = _found_by_the_last_resort(tmp_path, 0.1)
    registry = libsbml.SBMLResolverRegistry.getInstance()
    before = registry.getNumResolvers()
    counts = set()
    doc = libsbml.readSBMLFromFile(str(top))
    sys.setprofile(lambda *_: counts.add(registry.getNumResolvers()))
    try:
        names = [name for _, name, _, _ in _comp_submodel_documents(doc, str(top), str(tmp_path))]
    finally:
        sys.setprofile(None)
    assert names == [str(tmp_path / "models" / "mid.xml"), str(tmp_path / "ext.xml")]
    assert counts == {before}


def test_concurrent_loads_each_read_their_own_files(tmp_path, monkeypatch) -> None:
    """Two threads, switching as often as the interpreter allows, load two
    models whose files only the last resort finds: one at k = 0.3 and one whose
    file declares ``multi``. Each load must read its own directory, so every
    load of the first is k = 0.3 and every load of the second is refused."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    paths = {
        "clean": _found_by_the_last_resort(tmp_path / "clean", 0.3),
        "multi": _found_by_the_last_resort(tmp_path / "multi", 0.7, "multi", _MULTI_URI),
    }
    loads = 150
    outcomes: dict[str, list] = {name: [] for name in paths}
    start = threading.Barrier(len(paths))

    def load(name: str) -> None:
        start.wait()
        for _ in range(loads):
            try:
                outcomes[name].append(bngsim.Model.from_sbml(paths[name]).get_param("sub__sub__k"))
            except ModelError as e:
                outcomes[name].append("refused" if "multi" in str(e) else str(e))
            except Exception as e:  # anything else fails the comparison below
                outcomes[name].append(repr(e))

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=load, args=(name,)) for name in paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(interval)
    assert outcomes["clean"] == [0.3] * loads
    assert outcomes["multi"] == ["refused"] * loads


_ARRAYS = ("arrays", _ARRAYS_URI)


@pytest.mark.parametrize("package", [("multi", _MULTI_URI), _ARRAYS], ids=["multi", "arrays"])
@pytest.mark.parametrize("load", ["relative", "absolute, mirrored under the working directory"])
def test_two_copies_of_one_file_are_walked_from_each_location(
    tmp_path, monkeypatch, package, load
) -> None:
    """libSBML resolves a document's sources against its location URI, read
    as it reads one. Loaded as ``s1/top.xml``, the file is recorded as
    ``file:s1/top.xml``, read as plain ``top.xml``, so its ``inner.xml`` is the
    one in the working directory. Read again for the ExternalModelDefinition
    ``self``, which names ``top.xml``, the copy is ``file:///s1/top.xml`` and its
    ``inner.xml`` is ``s1/inner.xml``. Both copies' ModelDefinition ``d`` reads
    ``inner.xml``, so the flattener reads both. An absolute load is read
    without its leading ``/``, and a copy of the tree under the working
    directory takes the place of the working directory's ``inner.xml``."""
    cwd = tmp_path / "cwd"
    project = cwd / "s1"
    project.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    if load == "relative":
        top, clean = "s1/top.xml", cwd / "inner.xml"
    else:
        top = str(project / "top.xml")
        clean = cwd / str(project).lstrip(os.sep) / "inner.xml"
        clean.parent.mkdir(parents=True)
    clean.write_text(_leaf(0.1))
    (project / "inner.xml").write_text(_leaf(0.7, *package))
    (project / "top.xml").write_text(
        _comp(
            external=[("x", "inner.xml", "m"), ("self", "top.xml", "d")],
            submodels=[("a", "d"), ("b", "self")],
            definitions=_holding("d", "x"),
        )
    )
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(top)
    assert package[1] in str(excinfo.value)
    assert os.path.realpath(project / "inner.xml") in str(excinfo.value)


def test_a_chain_through_a_second_copy_of_the_loaded_file_is_followed(
    tmp_path, monkeypatch
) -> None:
    """``s1/top.xml``'s ``E0`` names ``a.xml``'s model ``Y``; loaded as
    ``s1/top.xml`` its ``a.xml`` is the working directory's, whose ``Y`` is an
    ExternalModelDefinition naming ``s1/top.xml``'s ``E0`` again. That copy's
    ``a.xml`` is ``s1/a.xml``, a ``multi`` file whose model is ``Y``. A chain's
    first link is not among the links libSBML stops a chain on, so it reads
    ``s1/a.xml``."""
    cwd = tmp_path / "cwd"
    (cwd / "s1").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    (cwd / "a.xml").write_text(_comp(external=[("Y", "s1/top.xml", "E0")]))
    (cwd / "s1" / "a.xml").write_text(_leaf(0.7, "multi", _MULTI_URI).replace('id="m"', 'id="Y"'))
    (cwd / "s1" / "top.xml").write_text(
        _comp(external=[("E0", "a.xml", "Y")], submodels=[("s", "E0")])
    )
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml("s1/top.xml")
    assert os.path.realpath(cwd / "s1" / "a.xml") in str(excinfo.value)


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-specific: Windows forbids '?' in a name"
)
def test_a_submodel_s_file_in_a_directory_named_with_a_question_mark_is_scanned(
    tmp_path,
) -> None:
    """libSBML reads a URI's path only up to a ``?``. The file a resolver
    read is the path it resolved, so the root-element scan reads that, not the
    path of the document's location URI read back."""
    folder = tmp_path / "q?d"
    folder.mkdir()
    (folder / "ext.xml").write_text(_leaf(0.7, "zzz", _ZZZ_URI))
    (folder / "top.xml").write_text(_submodel_of("ext.xml"))
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(folder / "top.xml")
    assert _ZZZ_URI in str(excinfo.value)
    assert str(folder / "ext.xml") in str(excinfo.value)


def test_the_flattened_model_is_checked_as_well(tmp_path, monkeypatch) -> None:
    """A backstop for a file the walk misses: flattening copies each
    instantiated file's packages, known ones with their ``required`` flags, onto
    the flattened model, so a ``multi`` the walk never saw still refuses the
    load. The walk is switched off here to show it."""
    import bngsim._sbml_loader as loader

    monkeypatch.setattr(loader, "_check_submodel_required_packages", lambda *a, **k: [])
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(_submodel_of("ext.xml"))
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    assert "flattened" in str(excinfo.value) and _MULTI_URI in str(excinfo.value)


def test_the_opt_out_warns_of_a_package_once(tmp_path, monkeypatch, caplog) -> None:
    """The walk and the check of the flattened model both see the submodel's
    ``multi``; the opt-out says so once."""
    (tmp_path / "ext.xml").write_text(_leaf(0.1, "multi", _MULTI_URI))
    (tmp_path / "top.xml").write_text(_submodel_of("ext.xml"))
    monkeypatch.setenv(_ALLOW_UNSUPPORTED_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    warned = [r.getMessage() for r in caplog.records if _MULTI_URI in r.getMessage()]
    assert len(warned) == 1, warned
