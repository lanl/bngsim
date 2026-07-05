"""GH #237 Phase 2 — time-varying species-reference stoichiometry.

An SBML L2 ``<stoichiometryMath>`` may evaluate to a quantity that changes over
the simulation (it reads ``time``, a species, or a rate-rule-/assignment-/event-
driven parameter). The reaction's net stoichiometric coefficient is then
time-varying and ``dS/dt`` must use the LIVE value rather than the frozen
load-time value that Phase 1 (static stoichiometry) bakes.

bngsim emits each such reference as its own per-species Functional reaction whose
rate is ``law · stoich_expr`` — exactly the SBML extent law ``stoich_{s,r}·v_r``
with the coefficient kept symbolic — so no kernel change is needed. These mirror
SBML Test Suite cases 00972 / 00973 / 00989 / 00994 / 01583 / 01632.
"""

import bngsim
import numpy as np
import pytest

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level2/version5" level="2" version="5">'
_HDR_L3 = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'


def _traj(res, sid):
    return np.asarray(res.species)[:, list(res.species_names).index(sid)]


# dX/dt = p(t)·k1 where p is rate-ruled (p' = 0.01, p0 = 1) and k1 = 1, so
# dX/dt = 1 + 0.01t and X(t) = t + 0.005t² (case 00973).
RATE_RULED_STOICH = f"""{_HDR}
  <model id="m_rate_ruled">
    <listOfCompartments><compartment id="C" size="1"/></listOfCompartments>
    <listOfSpecies><species id="X" compartment="C" initialConcentration="0"/></listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="1" constant="false"/>
      <parameter id="p" value="1" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="p">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.01</cn></math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference id="Xref" species="X">
            <stoichiometryMath>
              <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>p</ci></math>
            </stoichiometryMath>
          </speciesReference>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k1</ci></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_rate_ruled_stoichiometry():
    model = bngsim.Model.from_sbml_string(RATE_RULED_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 10), n_points=6)
    t = np.asarray(res.time)
    assert np.allclose(_traj(res, "X"), t + 0.005 * t * t, rtol=1e-4, atol=1e-6)


# dX/dt = k1·X (stoichiometry equals the species itself); X0 = 1 ⇒ X(t) = e^{k1 t}
# (case 00989).
SPECIES_VALUED_STOICH = f"""{_HDR}
  <model id="m_species_valued">
    <listOfCompartments><compartment id="C" size="1"/></listOfCompartments>
    <listOfSpecies><species id="X" compartment="C" initialConcentration="1"/></listOfSpecies>
    <listOfParameters><parameter id="k1" value="1" constant="false"/></listOfParameters>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference id="Xref" species="X">
            <stoichiometryMath>
              <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>X</ci></math>
            </stoichiometryMath>
          </speciesReference>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k1</ci></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_species_valued_stoichiometry():
    model = bngsim.Model.from_sbml_string(SPECIES_VALUED_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 3), n_points=7)
    t = np.asarray(res.time)
    assert np.allclose(_traj(res, "X"), np.exp(t), rtol=1e-4, atol=1e-6)


# dX/dt = k1·time (stoichiometry equals the time csymbol); X(t) = k1·t²/2 (case
# 00994).
TIME_VALUED_STOICH = f"""{_HDR}
  <model id="m_time_valued">
    <listOfCompartments><compartment id="C" size="1"/></listOfCompartments>
    <listOfSpecies><species id="X" compartment="C" initialConcentration="0"/></listOfSpecies>
    <listOfParameters><parameter id="k1" value="1" constant="false"/></listOfParameters>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference id="Xref" species="X">
            <stoichiometryMath>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              </math>
            </stoichiometryMath>
          </speciesReference>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k1</ci></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_time_valued_stoichiometry():
    model = bngsim.Model.from_sbml_string(TIME_VALUED_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 5), n_points=6)
    t = np.asarray(res.time)
    assert np.allclose(_traj(res, "X"), 0.5 * t * t, rtol=1e-4, atol=1e-6)


# Two reactions on one species: J0 a constant stoichiometry (3, consumed) and J1
# a variable one (rate-ruled p, produced). dS1/dt = -3 + p(t) with p' = 1, p0 = 1,
# so dS1/dt = -2 + t and S1(t) = 2 - 2t + t²/2 (case 01632). Exercises the mixed
# constant-and-variable path: the constant part keeps the baked multiset.
MIXED_CONST_VAR_STOICH = f"""{_HDR}
  <model id="m_mixed">
    <listOfCompartments><compartment id="C" size="1"/></listOfCompartments>
    <listOfSpecies><species id="S1" compartment="C" initialAmount="2"/></listOfSpecies>
    <listOfParameters><parameter id="p" value="1" constant="false"/></listOfParameters>
    <listOfRules>
      <rateRule variable="p">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference id="S1_degrade" species="S1">
            <stoichiometryMath>
              <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">3</cn></math>
            </stoichiometryMath>
          </speciesReference>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
        </kineticLaw>
      </reaction>
      <reaction id="J1" reversible="false">
        <listOfProducts>
          <speciesReference id="S1_create" species="S1">
            <stoichiometryMath>
              <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>p</ci></math>
            </stoichiometryMath>
          </speciesReference>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_mixed_constant_and_variable_stoichiometry():
    model = bngsim.Model.from_sbml_string(MIXED_CONST_VAR_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 8), n_points=5)
    t = np.asarray(res.time)
    assert np.allclose(_traj(res, "S1"), 2 - 2 * t + 0.5 * t * t, rtol=1e-4, atol=1e-6)


def test_variable_stoichiometry_refused_under_ssa():
    # A continuously-varying coefficient cannot be a stochastic integer fire
    # delta, so SSA is refused (consistent with the non-integer gate).
    model = bngsim.Model.from_sbml_string(RATE_RULED_STOICH)
    issues = bngsim.validate_for_ssa(model)
    errs = [i for i in issues if i.code == "variable_stoichiometry"]
    assert len(errs) == 1 and errs[0].severity == "error"
    with pytest.raises(bngsim.SsaValidationError) as ei:
        bngsim.Simulator(model, method="ssa")
    assert "variable_stoichiometry" in str(ei.value)


# L3 direct speciesReference id target: Xref := 1 + time, so dX/dt = 1 + t.
ASSIGNMENT_RULED_L3_STOICH = f"""{_HDR_L3}
  <model id="m_l3_assignment_rule_stoich">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies><species id="X" compartment="C" initialConcentration="0"/></listOfSpecies>
    <listOfRules>
      <assignmentRule variable="Xref">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><cn type="integer">1</cn>
            <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
          </apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference id="Xref" species="X" stoichiometry="1" constant="false"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_l3_assignment_rule_targeted_species_reference_id_is_live():
    model = bngsim.Model.from_sbml_string(ASSIGNMENT_RULED_L3_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 4), n_points=5)
    t = np.asarray(res.time)
    np.testing.assert_allclose(_traj(res, "X"), t + 0.5 * t * t, rtol=1e-5, atol=1e-7)


# L3 direct speciesReference id target: Xref' = 1, Xref(0) is the declared
# stoichiometry 1, so dX/dt = 1 + t.
RATE_RULED_L3_STOICH = f"""{_HDR_L3}
  <model id="m_l3_rate_rule_stoich">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies><species id="X" compartment="C" initialConcentration="0"/></listOfSpecies>
    <listOfRules>
      <rateRule variable="Xref">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference id="Xref" species="X" stoichiometry="1" constant="false"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_l3_rate_rule_targeted_species_reference_id_uses_declared_initial_stoich():
    model = bngsim.Model.from_sbml_string(RATE_RULED_L3_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 4), n_points=5)
    t = np.asarray(res.time)
    np.testing.assert_allclose(_traj(res, "X"), t + 0.5 * t * t, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(res.observables["Xref"]), 1 + t, rtol=1e-5)


# Suite 00972 shape: an event changes the product stoichiometry from 1 to 3 at
# t=2. The reaction must use the live event-promoted Xref value after the event.
EVENT_TARGETED_L3_STOICH = f"""{_HDR_L3}
  <model id="m_l3_event_stoich">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies><species id="X" compartment="C" initialConcentration="0"/></listOfSpecies>
    <listOfParameters><parameter id="k1" value="1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="J0" reversible="true">
        <listOfProducts>
          <speciesReference id="Xref" species="X" stoichiometry="1" constant="false"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k1</ci></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn type="integer">2</cn>
            </apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="Xref">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">3</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_l3_event_targeted_species_reference_id_is_live():
    model = bngsim.Model.from_sbml_string(EVENT_TARGETED_L3_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 5), n_points=6)
    t = np.asarray(res.time)
    expected = np.where(t <= 2.0, t, 2.0 + 3.0 * (t - 2.0))
    np.testing.assert_allclose(_traj(res, "X"), expected, rtol=1e-5, atol=1e-7)
    assert "Xref" in res.observable_names
    np.testing.assert_allclose(np.asarray(res.observables["Xref"])[-1], 3.0, rtol=1e-12)


# Suite 01583 shape: two simultaneous events assign the same speciesReference id.
# Higher priority (10) fires first, lower priority (5) fires last and wins,
# leaving S1_stoich = 3 after k1 crosses 4.5.
PRIORITY_EVENT_L3_STOICH = f"""{_HDR_L3}
  <model id="m_l3_priority_event_stoich">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies><species id="S1" compartment="C" initialConcentration="1"/></listOfSpecies>
    <listOfParameters><parameter id="k1" value="0" constant="false"/></listOfParameters>
    <listOfRules>
      <rateRule variable="k1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="true">
        <listOfProducts>
          <speciesReference id="S1_stoich" species="S1" stoichiometry="1" constant="false"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.1</cn></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><ci>k1</ci><cn>4.5</cn></apply></math>
        </trigger>
        <priority>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn type="integer">10</cn>
          </math>
        </priority>
        <listOfEventAssignments>
          <eventAssignment variable="S1_stoich">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">2</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="E1" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><ci>k1</ci><cn>4.5</cn></apply></math>
        </trigger>
        <priority>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn type="integer">5</cn>
          </math>
        </priority>
        <listOfEventAssignments>
          <eventAssignment variable="S1_stoich">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">3</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_l3_priority_events_targeting_species_reference_id_match_suite_01583():
    model = bngsim.Model.from_sbml_string(PRIORITY_EVENT_L3_STOICH)
    res = bngsim.Simulator(model, method="ode").run(t_span=(0, 10), n_points=11)
    t = np.asarray(res.time)
    expected = np.where(t <= 4.5, 1.0 + 0.1 * t, 1.45 + 0.3 * (t - 4.5))
    np.testing.assert_allclose(_traj(res, "S1"), expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(res.observables["S1_stoich"])[-1], 3.0, rtol=1e-12)
