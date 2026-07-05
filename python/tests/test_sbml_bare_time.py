"""GH #96 — diagnose a bare ``<ci>time</ci>`` that references no declared
symbol, instead of failing later with an opaque ``ERR239 - Undefined symbol:
u_ant_time`` from the core.

libsbml parses the proper simulation-time csymbol
(``<csymbol definitionURL=".../symbols/time">``) as ``AST_NAME_TIME``; a bare
``<ci>time</ci>`` — the malformed idiom some COPASI / CellDesigner exports emit
— parses as a plain ``AST_NAME`` named "time". When the model declares no
symbol named "time", that reference is undefined and RoadRunner refuses it too.

bngsim does NOT infer the csymbol (lenient resolution was declined in GH #96):
it fails closed with a targeted, actionable error. The check is GATED — it
fires only when no ``time`` symbol is declared, so a model that legitimately
names a symbol "time" still binds the bare ``<ci>time</ci>`` to it, and a
proper time csymbol is never mistaken for the bare idiom.
"""

import bngsim
import pytest

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">'
_TIME_CSYMBOL = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)


def _model(body: str) -> str:
    return f"{_HDR}\n{body}\n</sbml>"


# A bare <ci>time</ci> driving an assignment rule, no `time` symbol declared.
BARE_TIME_RULE = _model(
    """  <model id="m_bare_time_rule">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="sig" value="0" constant="false"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="sig">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>time</ci></math>
      </assignmentRule>
    </listOfRules>
  </model>"""
)

# Same idiom, but in an EVENT TRIGGER — the second GH #96 model (MODEL2402030002)
# puts it here, a container the delay/AlgebraicRule scan does not even cover.
BARE_TIME_EVENT = _model(
    """  <model id="m_bare_time_event">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="p" value="0" constant="false"/></listOfParameters>
    <listOfEvents>
      <event>
        <trigger>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/><ci>time</ci><cn>70</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="p">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>"""
)

# A declared parameter literally named `time` must SHADOW the bare <ci>time</ci>
# (the gate): this is a well-formed model and must load unchanged.
DECLARED_TIME_SHADOWS = _model(
    """  <model id="m_declared_time">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="time" value="5" constant="true"/>
      <parameter id="sig" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="sig">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>time</ci></math>
      </assignmentRule>
    </listOfRules>
  </model>"""
)

# The PROPER time csymbol (AST_NAME_TIME) with no `time` symbol declared must
# load — it is never mistaken for the bare-<ci> idiom.
PROPER_CSYMBOL = _model(
    f"""  <model id="m_csymbol">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="sig" value="0" constant="false"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="sig">
        <math xmlns="http://www.w3.org/1998/Math/MathML">{_TIME_CSYMBOL}</math>
      </assignmentRule>
    </listOfRules>
  </model>"""
)


@pytest.mark.parametrize(
    "sbml, where",
    [(BARE_TIME_RULE, "assignmentRule:sig"), (BARE_TIME_EVENT, "event")],
    ids=["assignment_rule", "event_trigger"],
)
def test_bare_time_refused_at_load(sbml, where):
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(sbml)
    msg = str(exc.value)
    # Names the bare idiom, points at the csymbol, and locates the offender.
    assert "<ci>time</ci>" in msg
    assert "symbols/time" in msg
    assert where in msg


def test_declared_time_symbol_shadows_and_loads():
    # A model that legitimately declares `time` is well-formed: the gate must
    # not fire, and the bare <ci>time</ci> binds to the declared parameter.
    model = bngsim.Model.from_sbml_string(DECLARED_TIME_SHADOWS)
    assert model is not None


def test_proper_time_csymbol_still_loads():
    model = bngsim.Model.from_sbml_string(PROPER_CSYMBOL)
    assert model is not None
