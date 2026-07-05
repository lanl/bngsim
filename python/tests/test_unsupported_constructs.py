"""GH #113 — the SBML loader must REFUSE constructs it cannot faithfully
simulate under ODE rather than silently approximate them.

Three constructs were previously dropped with no warning, producing a
confident finite trajectory for a *different* mathematical system:

  * ``delay(x, τ)``   — DDE; the AST handler returned ``x`` (zero-delay ODE).
  * ``AlgebraicRule`` — DAE constraint; no rule loop dispatched it.
  * ``fast="true"``   — fast-equilibrium constraint; integrated as ordinary.

``delay`` and ``AlgebraicRule`` are refused at load (unsupported under every
method); ``fast`` stays a loadable SSA issue and is refused at Simulator
construction under ODE so the SSA validate/override contract is preserved.
All three honor ``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`` to restore the
legacy silent-approximation behavior (mirrors the GH #94 unset-param gate).
"""

import warnings

import bngsim
import pytest

ENV = "BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS"

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">'
_HDR_L3 = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
_DELAY_CSYMBOL = (
    '<csymbol encoding="text" '
    'definitionURL="http://www.sbml.org/sbml/symbols/delay">delay</csymbol>'
)


def _model(body: str) -> str:
    return f"{_HDR}\n{body}\n</sbml>"


def _model_l3(body: str) -> str:
    return f"{_HDR_L3}\n{body}\n</sbml>"


# ── delay() in a reaction kinetic law ─────────────────────────────────────

DELAY_KINETIC = _model(
    f"""  <model id="m_delay">
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

# delay(x, 0) == x exactly — the zero-delay carve-out must still load.
DELAY_ZERO = DELAY_KINETIC.replace("<cn>1</cn>", "<cn>0</cn>")

# delay buried in a *called* user-defined function — must be caught.
DELAY_IN_CALLED_FUNCDEF = _model(
    f"""  <model id="m_delay_fd">
    <listOfFunctionDefinitions>
      <functionDefinition id="lagged">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>
          <bvar><ci>x</ci></bvar>
          <apply>{_DELAY_CSYMBOL}<ci>x</ci><cn>2</cn></apply>
        </lambda></math>
      </functionDefinition>
    </listOfFunctionDefinitions>
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
          <apply><times/><ci>k</ci><apply><ci>lagged</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)

# delay defined in a funcDef that is NEVER called — must NOT trip the gate
# (precise scoping: only constructs that feed the integrated system count).
DELAY_IN_UNCALLED_FUNCDEF = _model(
    f"""  <model id="m_delay_uncalled">
    <listOfFunctionDefinitions>
      <functionDefinition id="unused">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>
          <bvar><ci>x</ci></bvar>
          <apply>{_DELAY_CSYMBOL}<ci>x</ci><cn>2</cn></apply>
        </lambda></math>
      </functionDefinition>
    </listOfFunctionDefinitions>
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
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)

# ── delay() in L2 <stoichiometryMath> (SBML suite 01481) ──────────────────
# A delay in a speciesReference's stoichiometry feeds the integrated system just
# as a rate law does, so it is as much a DDE. libSBML exposes it only via the
# SpeciesReference, so the guard must reach into every reactant/product.

DELAY_STOICHIOMETRY = _model(
    f"""  <model id="m_delay_stoich">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="100" hasOnlySubstanceUnits="true"/>
      <species id="B" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1"/></listOfParameters>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfReactants>
          <speciesReference species="A">
            <stoichiometryMath><math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>{_DELAY_CSYMBOL}<ci>A</ci><cn>1</cn></apply>
            </math></stoichiometryMath>
          </speciesReference>
        </listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)

# delay(A, 0) == A exactly — the zero-delay carve-out must still load here too.
DELAY_STOICHIOMETRY_ZERO = DELAY_STOICHIOMETRY.replace("<cn>1</cn>", "<cn>0</cn>")


# ── delay() in event math ─────────────────────────────────────────────────

_EVENT_DELAY_EXPR = f"""<apply>{_DELAY_CSYMBOL}<ci>P1</ci><cn>1</cn></apply>"""

_EVENT_TRUE_TRIGGER = """        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><true/></math>
        </trigger>"""

_EVENT_ASSIGN_ONE = """        <listOfEventAssignments>
          <eventAssignment variable="P2">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>"""


def _event_model(event_body: str) -> str:
    return _model_l3(
        f"""  <model id="m_event_delay">
    <listOfParameters>
      <parameter id="P1" value="1" constant="false"/>
      <parameter id="P2" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
{event_body}
      </event>
    </listOfEvents>
  </model>"""
    )


DELAY_EVENT_TRIGGER = _event_model(
    f"""        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/>{_EVENT_DELAY_EXPR}<cn>0</cn></apply>
          </math>
        </trigger>
{_EVENT_ASSIGN_ONE}"""
)

DELAY_EVENT_DELAY = _event_model(
    f"""{_EVENT_TRUE_TRIGGER}
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML">{_EVENT_DELAY_EXPR}</math>
        </delay>
{_EVENT_ASSIGN_ONE}"""
)

DELAY_EVENT_PRIORITY = _event_model(
    f"""{_EVENT_TRUE_TRIGGER}
        <priority>
          <math xmlns="http://www.w3.org/1998/Math/MathML">{_EVENT_DELAY_EXPR}</math>
        </priority>
{_EVENT_ASSIGN_ONE}"""
)

DELAY_EVENT_ASSIGNMENT = _event_model(
    f"""{_EVENT_TRUE_TRIGGER}
        <listOfEventAssignments>
          <eventAssignment variable="P2">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              {_EVENT_DELAY_EXPR}
            </math>
          </eventAssignment>
        </listOfEventAssignments>"""
)

# ── AlgebraicRule ─────────────────────────────────────────────────────────

ALGEBRAIC_RULE = _model(
    """  <model id="m_alg">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="50" hasOnlySubstanceUnits="true"/>
      <species id="B" compartment="c" initialAmount="50" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="T0" value="100"/><parameter id="k" value="0.1"/>
    </listOfParameters>
    <listOfRules>
      <algebraicRule><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><minus/><ci>T0</ci><apply><plus/><ci>A</ci><ci>B</ci></apply></apply>
      </math></algebraicRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
)

# ── fast="true" ───────────────────────────────────────────────────────────

FAST_REACTION = _model(
    """  <model id="m_fast">
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
    """Default-deny: every test starts with the opt-out env var unset."""
    monkeypatch.delenv(ENV, raising=False)


# ── delay refusal ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sbml",
    [DELAY_KINETIC, DELAY_IN_CALLED_FUNCDEF],
    ids=["kinetic_law", "called_funcdef"],
)
def test_delay_refused_at_load(sbml):
    with pytest.raises(bngsim.ModelError, match="delay"):
        bngsim.Model.from_sbml_string(sbml)


def test_delay_zero_is_allowed():
    """delay(x, 0) == x exactly; the silent drop is sound, so it must load."""
    model = bngsim.Model.from_sbml_string(DELAY_ZERO)
    # And it actually integrates.
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    assert res is not None


def test_delay_in_uncalled_funcdef_is_allowed():
    """A delay in a funcDef nothing calls does not feed the system — no refusal."""
    model = bngsim.Model.from_sbml_string(DELAY_IN_UNCALLED_FUNCDEF)
    assert model is not None


def test_delay_in_stoichiometry_math_refused_at_load():
    """SBML suite 01481: delay() in <stoichiometryMath> is a DDE the kinetic-law
    scan never reached, so bngsim silently zero-delayed it and returned a wrong
    trajectory. It must refuse at load, naming the stoichiometryMath location."""
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(DELAY_STOICHIOMETRY)
    msg = str(exc.value)
    assert "delay" in msg
    assert "stoichiometryMath" in msg


def test_delay_zero_in_stoichiometry_math_is_allowed():
    """delay(x, 0) == x exactly; the zero-delay carve-out holds here too."""
    model = bngsim.Model.from_sbml_string(DELAY_STOICHIOMETRY_ZERO)
    assert model is not None


@pytest.mark.parametrize(
    "sbml",
    [
        DELAY_EVENT_TRIGGER,
        DELAY_EVENT_DELAY,
        DELAY_EVENT_PRIORITY,
        DELAY_EVENT_ASSIGNMENT,
    ],
    ids=["event_trigger", "event_delay", "event_priority", "event_assignment"],
)
def test_delay_in_event_math_refused_at_load(sbml):
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(sbml)
    msg = str(exc.value)
    assert "delay" in msg
    assert "event:E0" in msg


# ── AlgebraicRule refusal ─────────────────────────────────────────────────


def test_algebraic_rule_refused_at_load():
    with pytest.raises(bngsim.ModelError, match="AlgebraicRule"):
        bngsim.Model.from_sbml_string(ALGEBRAIC_RULE)


def test_refusal_message_names_construct_and_element():
    with pytest.raises(bngsim.ModelError) as exc:
        bngsim.Model.from_sbml_string(ALGEBRAIC_RULE)
    msg = str(exc.value)
    assert "AlgebraicRule" in msg
    assert "T0" in msg  # the offending rule's formula is quoted
    assert ENV in msg  # the opt-out lever is advertised


# An AlgebraicRule with NO MathML (``<algebraicRule/>``) states ``0 = ∅`` — no
# constraint, so it is a no-op, not a DAE. bngsim must load and simulate it, not
# refuse (SBML suite 01244: a lone variable parameter that just holds its value).
EMPTY_ALGEBRAIC_RULE = _model(
    """  <model id="m_empty_alg">
    <listOfParameters><parameter id="p" value="3" constant="false"/></listOfParameters>
    <listOfRules>
      <algebraicRule/>
    </listOfRules>
  </model>"""
)


def test_empty_algebraic_rule_is_noop_not_refused():
    model = bngsim.Model.from_sbml_string(EMPTY_ALGEBRAIC_RULE)  # must not raise
    res = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=11)
    assert model.get_param("p") == pytest.approx(3.0)
    assert res is not None


# ── fast under ODE: refused at Simulator, SSA path intact ─────────────────


def test_fast_loads_and_is_recorded_as_ssa_issue():
    """fast stays loadable so validate_for_ssa can still report it."""
    model = bngsim.Model.from_sbml_string(FAST_REACTION)
    codes = [i.code for i in bngsim.validate_for_ssa(model)]
    assert "fast_reaction" in codes


def test_fast_refused_under_ode():
    model = bngsim.Model.from_sbml_string(FAST_REACTION)
    with pytest.raises(bngsim.ModelError, match="fast"):
        bngsim.Simulator(model, method="ode")


# ── opt-out restores the legacy silent approximation ──────────────────────


@pytest.mark.parametrize("sbml", [DELAY_KINETIC, ALGEBRAIC_RULE], ids=["delay", "algebraic"])
def test_optout_allows_load(monkeypatch, sbml):
    monkeypatch.setenv(ENV, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_sbml_string(sbml)
    assert model is not None


def test_optout_allows_fast_under_ode(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    model = bngsim.Model.from_sbml_string(FAST_REACTION)
    sim = bngsim.Simulator(model, method="ode")
    assert sim is not None
