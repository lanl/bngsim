"""GH #233 — an event whose ``<trigger>`` has no ``<math>`` (or no ``<trigger>``
at all) never fires.

An event fires on a false→true transition of its trigger. A trigger with no
MathML formula — or an event with no trigger element — has no condition that can
ever become true, so it never fires (SBML L3v2 §4.11.2; suite cases 01238/01239,
both expect the target to hold its initial value). The loader's §10 event build
already skips such an event, but the event-target pre-scan still queued the
event's assignment target for promotion to event-driven species state — so the
promotion never completed and the (non-constant) parameter was stranded, dropping
its column from the output entirely (``var_missing``).

The fix: an event whose trigger has no ``<math>`` child is skipped in the pre-scan
too, leaving its assignment targets as plain parameters reported at their IC (the
way RoadRunner emits them). A target assigned by some *other* event with a real
trigger is still promoted by that event's pass.
"""

import bngsim
import pytest

# Case 01238 shape: <trigger initialValue="true"/> with NO <math> child. p is a
# non-constant parameter; its only event assignment (p=2) never runs because the
# event never fires. p must hold its initial value 3 and still be reported.
SBML_NO_MATH_TRIGGER = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="no_math_trigger">
    <listOfParameters>
      <parameter id="p" value="3" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true"/>
        <listOfEventAssignments>
          <eventAssignment variable="p">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn type="integer">2</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

# Case 01239 shape: an event with NO <trigger> element at all.
SBML_ABSENT_TRIGGER = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="absent_trigger">
    <listOfParameters>
      <parameter id="p" value="3" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <listOfEventAssignments>
          <eventAssignment variable="p">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn type="integer">2</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


@pytest.mark.parametrize(
    "sbml", [SBML_NO_MATH_TRIGGER, SBML_ABSENT_TRIGGER], ids=["no_math", "absent"]
)
def test_no_math_or_absent_trigger_never_fires(sbml):
    """The event never fires, so its assignment (p=2) never runs: p stays a plain
    parameter reported at its initial value 3, not promoted to event-driven state."""
    model = bngsim.Model.from_sbml_string(sbml)

    # p is left as a plain parameter (not stranded / dropped): queryable by name.
    assert "p" in model.param_names
    assert model.get_param("p") == pytest.approx(3.0)

    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 10), n_points=11, rtol=1e-10, atol=1e-12
    )
    # No spurious promotion to a species / observable column.
    assert "p" not in r.species_names
    assert "p" not in r.observable_names
