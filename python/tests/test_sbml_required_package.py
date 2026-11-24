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

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import ModelError
from bngsim._sbml_loader import _ALLOW_UNSUPPORTED_ENV

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
    (tmp_path / "models").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")  # nothing to find in the working directory
    (tmp_path / "ext.xml").write_text(_leaf(0.3, "multi", _MULTI_URI))
    (tmp_path / "models" / "mid.xml").write_text(_submodel_of("ext.xml"))
    (tmp_path / "top.xml").write_text(
        _comp(external=[("mid", "models/mid.xml", "top")], submodels=[("sub", "mid")])
    )
    with pytest.raises(ModelError) as excinfo:
        bngsim.Model.from_sbml(tmp_path / "top.xml")
    assert str(tmp_path / "ext.xml") in str(excinfo.value)

    monkeypatch.setenv(_ALLOW_UNSUPPORTED_ENV, "1")
    _assert_decays_at(bngsim.Model.from_sbml(tmp_path / "top.xml"), 0.3)


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
