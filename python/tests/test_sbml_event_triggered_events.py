"""GH #233 — events triggered by another event's assignment ("events triggering
events"), plus T0 firing + persistence of delayed events.

An event assignment is a discrete jump in state. When that jump pushes another
(or the same) event's trigger from false to true, SBML L3 §3.4 requires the
newly-true event to fire — but the CVODE root finder only sees zero-crossings
during continuous integration, never the jump. bngsim re-checks every trigger
after each assignment batch and schedules the delayed events that just rose,
sustaining self-perpetuating delayed-event chains that the root finder alone
freezes after a single round.

Cases mirrored: 01759 (a time event seeds a P1>1 chain with no T0 firing) and
01758 (two persistent delayed events fire at t=0 via initialValue=false and keep
re-triggering each other; one would cancel the other were it not persistent).

Same-instant (delay-0) assignment-induced re-triggers are intentionally NOT
cascaded — see the DEFERRED note in cvode_simulator.cpp: matching the suite's
RandomEventExecution references needs random event selection, which a
deterministic engine does not implement. This test only exercises delayed chains.
"""

import bngsim
import numpy as np

# 01759: P1=0.5. E0(P1>1, delay 2 → P1=3), E1(P1>1, delay 1 → P1=0),
# E2(time>0.65, no delay → P1=2). E2 lifts P1 above 1, arming E0/E1; thereafter
# E0's delayed P1=3 re-arms both and the 0↔3 chain self-sustains (period 2).
SBML_01759_CHAIN = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="evt_chain">
    <listOfParameters>
      <parameter id="P1" value="0.5" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>P1</ci><cn type="integer">1</cn></apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">2</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="P1">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">3</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="E1" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>P1</ci><cn type="integer">1</cn></apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="P1">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">0</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="E2" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>0.65</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="P1">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">2</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

# 01758: P1=1.5 (>1) at t=0. E0(delay 2.3 → P1=3), E1(delay 1.5 → P1=0), both
# trigger P1>1 with initialValue=false (so both fire at t=0) and persistent=true
# (so E0 still fires at t=2.3 even though E1 set P1=0 at t=1.5, making the trigger
# false in the meantime). E0's P1=3 then re-arms both and the chain continues.
SBML_01758_T0 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="evt_t0">
    <listOfParameters>
      <parameter id="P1" value="1.5" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>P1</ci><cn type="integer">1</cn></apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2.3</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="P1">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">3</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="E1" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>P1</ci><cn type="integer">1</cn></apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1.5</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="P1">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">0</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def _p1(sbml):
    model = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=11, rtol=1e-8, atol=1e-10
    )
    assert "P1" in r.observable_names
    return np.asarray(r.observables["P1"])


def test_event_assignment_triggers_delayed_event_chain():
    """01759: a time event lifts P1>1, and the persistent delayed events then
    keep re-triggering each other into a sustained 0↔3 oscillation. Without
    cascading the assignment-induced rising edges, bngsim freezes at P1=2."""
    expected = [0.5, 2, 0, 3, 0, 3, 0, 3, 0, 3, 0]
    np.testing.assert_allclose(_p1(SBML_01759_CHAIN), expected, rtol=0, atol=1e-6)


def test_t0_firing_persistent_delayed_chain():
    """01758: two persistent delayed events fire at t=0 (initialValue=false),
    survive the trigger lapsing before their delay elapses (persistence), and
    re-trigger each other thereafter."""
    expected = [1.5, 1.5, 0, 3, 0, 3, 3, 3, 3, 0, 3]
    np.testing.assert_allclose(_p1(SBML_01758_T0), expected, rtol=0, atol=1e-6)
