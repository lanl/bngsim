"""SBML Level 3 packages declared ``required="true"`` (issue #592).

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
