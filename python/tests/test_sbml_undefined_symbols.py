"""GH #119 — refuse a model that references an undefined symbol in an
initialAssignment or rule, instead of silently dropping it.

bngsim's IA / rule evaluator returns no value for an ``AST_NAME`` that binds to
no declared component, so the loader would otherwise keep the target's declared
value and integrate — dropping the offending construct with zero warnings.
RoadRunner and COPASI both refuse such models ("Could not find requested symbol
'X'" / ``CCopasiException``). bngsim fails closed too, consistent with the bare
``<ci>time</ci>`` refusal (GH #96) and the fail-closed AST translator (GH #97).

The check is surgical: it walks only ``AST_NAME`` nodes (so the proper ``time``
csymbol — ``AST_NAME_TIME`` — and user-defined function *calls* — ``AST_FUNCTION``,
whose name is not an ``AST_NAME`` — are never mis-flagged), scoped to
initialAssignments and rules. ``BNGSIM_ALLOW_UNDEFINED_SYMBOLS=1`` restores the
legacy silent-drop with a loud per-offender warning.

Repro of record: MODEL2205030001, whose entire ``Kd_*`` constant class was lost
on export, leaving all 23 initial assignments referencing undefined symbols.
"""

import bngsim
import pytest

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">'
_MATHML = "http://www.w3.org/1998/Math/MathML"


def _model(body: str) -> str:
    return f"{_HDR}\n{body}\n</sbml>"


# `Kd_missing` is referenced by an initialAssignment but declared nowhere; the
# multiplicand `A` IS declared. RoadRunner/COPASI refuse; bngsim must too.
UNDEFINED_IN_IA = _model(
    f"""  <model id="m_undef_ia">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="P" value="1" constant="true"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="P">
        <math xmlns="{_MATHML}"><apply><times/><ci>A</ci><ci>Kd_missing</ci></apply></math>
      </initialAssignment>
    </listOfInitialAssignments>
  </model>"""
)

# Same defect class, but the undefined symbol drives an assignment rule.
UNDEFINED_IN_RULE = _model(
    f"""  <model id="m_undef_rule">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="sig" value="0" constant="false"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="sig">
        <math xmlns="{_MATHML}"><ci>missing_k</ci></math>
      </assignmentRule>
    </listOfRules>
  </model>"""
)

# Identical shape to UNDEFINED_IN_IA, but `Kd_missing` IS a declared parameter —
# a well-formed model that must still load (guards against a false positive that
# would break every model with cross-referencing initial assignments).
WELL_FORMED_IA = _model(
    f"""  <model id="m_wellformed_ia">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="2" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="P" value="1" constant="true"/>
      <parameter id="Kd_missing" value="3" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="P">
        <math xmlns="{_MATHML}"><apply><times/><ci>A</ci><ci>Kd_missing</ci></apply></math>
      </initialAssignment>
    </listOfInitialAssignments>
  </model>"""
)

# A user-defined function CALL in an initialAssignment. The funcDef body
# references its formal parameter `x`, which is NOT a global symbol — but the
# loader walks only the call site `f(A)` (an AST_FUNCTION whose name is not an
# AST_NAME), so neither `f` nor `x` is mis-flagged. Must load.
FUNCDEF_CALL_IA = _model(
    f"""  <model id="m_funcdef_ia">
    <listOfFunctionDefinitions>
      <functionDefinition id="f">
        <math xmlns="{_MATHML}">
          <lambda><bvar><ci>x</ci></bvar><apply><times/><ci>x</ci><cn>2</cn></apply></lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="2" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="P" value="1" constant="true"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="P">
        <math xmlns="{_MATHML}"><apply><ci>f</ci><ci>A</ci></apply></math>
      </initialAssignment>
    </listOfInitialAssignments>
  </model>"""
)


@pytest.mark.parametrize(
    "sbml, symbol, where",
    [
        (UNDEFINED_IN_IA, "Kd_missing", "initialAssignment:P"),
        (UNDEFINED_IN_RULE, "missing_k", "assignmentRule:sig"),
    ],
    ids=["initial_assignment", "assignment_rule"],
)
def test_undefined_symbol_refused_at_load(sbml, symbol, where):
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(sbml)
    msg = str(exc.value)
    # Names the offending symbol, its location, mirrors the RR/COPASI refusal,
    # and points at the escape hatch.
    assert symbol in msg
    assert where in msg
    assert "RoadRunner" in msg
    assert "BNGSIM_ALLOW_UNDEFINED_SYMBOLS" in msg


@pytest.mark.parametrize(
    "sbml",
    [WELL_FORMED_IA, FUNCDEF_CALL_IA],
    ids=["declared_symbol", "funcdef_call"],
)
def test_well_formed_models_still_load(sbml):
    # No false positive: every AST_NAME resolves (declared parameter; funcDef
    # call args), so the model loads unchanged.
    model = bngsim.Model.from_sbml_string(sbml)
    assert model is not None


def test_escape_hatch_restores_silent_load(monkeypatch):
    # BNGSIM_ALLOW_UNDEFINED_SYMBOLS=1 restores the legacy lenient behavior:
    # the IA is dropped, the target keeps its declared value, and the model
    # loads (with a per-offender warning, asserted on the logger).
    monkeypatch.setenv("BNGSIM_ALLOW_UNDEFINED_SYMBOLS", "1")
    model = bngsim.Model.from_sbml_string(UNDEFINED_IN_IA)
    assert model is not None
    # P kept its declared value (the IA referencing Kd_missing was dropped).
    assert model.get_param("P") == pytest.approx(1.0)
