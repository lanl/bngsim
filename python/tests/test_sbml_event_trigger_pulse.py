"""GH #246 — compound event-trigger pulses must not be stepped over.

The SBML semantic-suite case 01511 has a trigger band
``(S1 > 0.18) && (S1 < 0.19)``. With the full Boolean trigger as the only CVODE
root, one internal step can see false before and after the narrow true window,
so the persistent delayed event is never queued. The loader now registers the
relational trigger edges as no-op roots so the normal event re-check sees the
true pulse and queues the delayed assignment.
"""

import bngsim
import numpy as np

SBML_PERSISTENT_DELAYED_TRIGGER_PULSE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="pulse">
    <listOfCompartments>
      <compartment id="C1" size="0.5" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="C1" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="x" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="S1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.4</cn></math>
      </rateRule>
      <rateRule variable="C1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.2</cn></math>
      </rateRule>
      <assignmentRule variable="x">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>S1</ci></math>
      </assignmentRule>
    </listOfRules>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><and/>
              <apply><gt/><ci>S1</ci><cn>0.18</cn></apply>
              <apply><lt/><ci>S1</ci><cn>0.19</cn></apply>
            </apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.1</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="S1">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.2</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_persistent_delayed_event_queues_from_narrow_trigger_pulse():
    model = bngsim.Model.from_sbml_string(SBML_PERSISTENT_DELAYED_TRIGGER_PULSE)
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 1), n_points=11, rtol=1e-8, atol=1e-12
    )

    t = np.asarray(result.time)
    c1 = np.asarray(result.species)[:, list(result.species_names).index("C1")]
    s1_conc = np.asarray(result.expressions["x"])

    expected_c1 = 0.5 + 0.2 * t
    # The trigger band is entered at t=0.45; the persistent delayed event
    # executes at t=0.55 and sets the concentration to 0.2. The rate rule then
    # resumes with d[S1]/dt = 0.4.
    expected_conc = np.where(t <= 0.5, 0.4 * t, 0.2 + 0.4 * (t - 0.55))
    expected_amount = expected_conc * expected_c1

    np.testing.assert_allclose(c1, expected_c1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s1_conc, expected_conc, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        s1_conc * c1,
        expected_amount,
        rtol=1e-8,
        atol=1e-10,
    )
