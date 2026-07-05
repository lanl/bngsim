"""GH #233 — an ``<eventAssignment>`` with no ``<math>`` child is a no-op.

SBML L3v2 §4.11.5 allows an event assignment to omit its MathML: such an
assignment never writes its target, so the variable keeps its current value. The
loader previously treated *any* event-assignment target as event-driven state and
queued it for promotion to a species, but the §10 emit loop skips no-math
assignments — so the promotion never happened and the symbol was stranded,
dropping the (non-constant) parameter from the output entirely (``var_missing``).

The fix: a target whose only event assignment lacks ``<math>`` is left as a plain
parameter (still reported as a constant, the way RoadRunner emits it). A target
assigned with math by some other event is still promoted by that event's pass.

Suite cases this covers: 01600/01602 (lone no-math assignment — whole event is a
no-op) and 01601/01603/01243 (no-math assignment alongside a real one), plus the
no-delay 01237.
"""

import bngsim
import numpy as np
import pytest

# Case 01600 shape: a single non-constant parameter P1, mutated only by an event
# whose lone assignment has no <math>. The assignment is a no-op, so P1 holds its
# initial value (1) for the whole run — but it must still appear as output.
SBML_LONE_NO_MATH = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="lone_no_math">
    <listOfParameters>
      <parameter id="P1" value="1" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>4.5</cn></apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="P1"/>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

# Case 01601 shape: two non-constant parameters. The event assigns P2=3 (real,
# delayed) and P1 (no <math>). P1 must stay constant 1; P2 fires to 3 once the
# delayed trigger (time>=4.5, delay 1 -> t=5.5) elapses.
SBML_MIXED_NO_MATH = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="mixed_no_math">
    <listOfParameters>
      <parameter id="P1" value="1" constant="false"/>
      <parameter id="P2" value="1" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
              <cn>4.5</cn></apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="P2">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">3</cn></math>
          </eventAssignment>
          <eventAssignment variable="P1"/>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_lone_no_math_assignment_keeps_param_reported_and_constant():
    """The sole assignment has no <math>: the event is a complete no-op, but the
    non-constant parameter P1 must still be reported (constant at its IC)."""
    model = bngsim.Model.from_sbml_string(SBML_LONE_NO_MATH)

    # P1 is left as a plain parameter (not promoted to event-driven species
    # state), so it's queryable by name at its initial value.
    assert "P1" in model.param_names
    assert model.get_param("P1") == pytest.approx(1.0)

    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=11, rtol=1e-10, atol=1e-12
    )
    # No spurious promotion: P1 is neither a species nor an observable column.
    assert "P1" not in r.species_names
    assert "P1" not in r.observable_names


def test_mixed_no_math_and_real_assignment():
    """One no-math assignment (P1) next to a real one (P2): P1 stays constant,
    P2 fires through the delayed trigger."""
    model = bngsim.Model.from_sbml_string(SBML_MIXED_NO_MATH)

    # P1 (no-math target) remains a plain parameter; P2 (real target) is promoted.
    assert "P1" in model.param_names
    assert model.get_param("P1") == pytest.approx(1.0)
    assert "P2" not in model.param_names

    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=11, rtol=1e-10, atol=1e-12
    )
    assert "P2" in r.observable_names

    t = np.asarray(r.time)
    P2 = np.asarray(r.observables["P2"])
    # Trigger time>=4.5 fires at t=4.5, delay 1 -> assignment applies at t=5.5.
    np.testing.assert_allclose(P2[t <= 5.0], 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(P2[t >= 6.0], 3.0, rtol=0, atol=1e-9)
