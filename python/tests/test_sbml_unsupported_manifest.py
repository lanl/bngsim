"""GH #241 — the unsupported-construct SSOT and its committed test-runner manifest.

:mod:`bngsim._sbml_unsupported` is the single source of truth for the SBML
constructs bngsim refuses under ODE and their SBML Test Suite feature tags. Two
things must stay pinned to it:

1. the committed manifest
   ``benchmarks/suites/sbml_test_suite/testrunner/bngsim-unsupported-tags.txt``
   fed to the official grading — it must equal ``manifest_text()`` byte-for-byte
   (regenerate with ``testrunner/gen_manifest.py`` when the SSOT changes);
2. the loader / simulator refusal messages — they must contain the SSOT
   construct *labels*, so the tags declared unsupported and the strings the guard
   emits can never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

import bngsim
import pytest
from bngsim import _sbml_unsupported as u

ENV = "BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS"

EXPECTED_TAGS = {"AlgebraicRule", "CSymbolDelay", "FastReaction", "MultipleFastReactions", "fbc"}


def _source_root() -> Path | None:
    """Locate the ``bngsim/`` source tree (mirrors test_version_consistency)."""
    candidates: list[Path] = []
    env_data = os.environ.get("BNGSIM_TEST_DATA")
    if env_data:
        candidates.extend(Path(env_data).resolve().parents)
    env_root = os.environ.get("BNGSIM_SOURCE_ROOT")
    if env_root:
        candidates.append(Path(env_root).resolve())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        py = candidate / "pyproject.toml"
        if py.is_file() and 'name = "bngsim"' in py.read_text():
            return candidate
    return None


def _manifest_path() -> Path | None:
    root = _source_root()
    if root is None:
        return None
    p = root / "benchmarks/suites/sbml_test_suite/testrunner/bngsim-unsupported-tags.txt"
    return p if p.is_file() else None


# ── SSOT invariants (always runnable — need only the installed package) ────────


def test_construct_labels_are_load_bearing():
    # These substrings are what the refusal-message tests assert on; changing
    # them silently would break the loader/simulator contract.
    assert u.construct_labels() == ("delay", "AlgebraicRule", "fast")


def test_unsupported_tags_are_the_declared_set():
    assert set(u.unsupported_tags()) == EXPECTED_TAGS
    # sorted + de-duplicated
    assert u.unsupported_tags() == sorted(set(u.unsupported_tags()))


def test_avogadro_and_random_events_not_declared():
    # Documented honest deviations, NOT capability gaps — must stay OUT of the
    # manifest (GH #231 / GH #242).
    tags = set(u.unsupported_tags())
    assert "CSymbolAvogadro" not in tags
    assert "RandomEventExecution" not in tags


def test_manifest_text_roundtrips():
    assert u.parse_manifest(u.manifest_text()) == u.unsupported_tags()


# ── Committed manifest is pinned to the SSOT ──────────────────────────────────


def test_committed_manifest_matches_ssot():
    path = _manifest_path()
    if path is None:
        pytest.skip("committed manifest not found (benchmarks tree absent in this env)")
    assert path.read_text() == u.manifest_text(), (
        "bngsim-unsupported-tags.txt is stale; regenerate with "
        "benchmarks/suites/sbml_test_suite/testrunner/gen_manifest.py"
    )


# ── Loader / simulator refusal messages carry the SSOT labels ─────────────────

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">'
_DELAY_CSYMBOL = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/delay">delay</csymbol>'
)


def _model(body: str) -> str:
    return f"{_HDR}\n{body}\n</sbml>"


_DELAY_MODEL = _model(
    f"""  <model id="m">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="100" hasOnlySubstanceUnits="true"/>
      <species id="B" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1"/></listOfParameters>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci>
            <apply>{_DELAY_CSYMBOL}<ci>A</ci><cn>1</cn></apply>
          </apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)

_ALGEBRAIC_MODEL = _model(
    """  <model id="m">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="50" hasOnlySubstanceUnits="true"/>
      <species id="B" compartment="c" initialAmount="50" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="T0" value="100"/></listOfParameters>
    <listOfRules>
      <algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><ci>T0</ci><apply><plus/><ci>A</ci><ci>B</ci></apply></apply>
      </math></algebraicRule>
    </listOfRules>
  </model>"""
)

_FAST_MODEL = _model(
    """  <model id="m">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="100" hasOnlySubstanceUnits="true"/>
      <species id="B" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1"/></listOfParameters>
    <listOfReactions>
      <reaction id="rfast" reversible="false" fast="true">
        <listOfReactants><speciesReference species="A" stoichiometry="1"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)


@pytest.fixture(autouse=True)
def _clear_optout(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_delay_refusal_message_uses_ssot_label():
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(_DELAY_MODEL)
    assert u.DELAY in str(exc.value)


def test_algebraic_refusal_message_uses_ssot_label():
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(_ALGEBRAIC_MODEL)
    assert u.ALGEBRAIC_RULE in str(exc.value)


def test_fast_refusal_message_uses_ssot_label():
    model = bngsim.Model.from_sbml_string(_FAST_MODEL)  # fast loads; refused at Simulator
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Simulator(model, method="ode")
    assert u.FAST in str(exc.value)
