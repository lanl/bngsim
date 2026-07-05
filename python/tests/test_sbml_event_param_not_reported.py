"""GH #71 — an SBML parameter/compartment mutated only by an event must NOT
leak into the trajectory output as a species column.

The engine writes event assignments into species slots, so a parameter or
compartment that an event mutates has to be promoted to a species to carry
per-trajectory state. But it is not a floating species, and RoadRunner does not
emit it as a trajectory column. The loader marks such a promotion
``reported=False``; the Result layer keeps it as full integrator state and as a
same-named observable (so other expressions resolve the live value) but projects
it out of ``species`` / ``species_names`` and the ``.cdat`` export.

Surfaced by MODEL1108260014, whose events assign to ``parameter_1`` (IIa*) and
``compartment_1`` — both appeared as spurious bngsim-only trajectory columns
against RoadRunner. (A rate-rule-promoted parameter, by contrast, IS a genuine
ODE variable RoadRunner reports, and stays reported — covered elsewhere.)
"""

import bngsim
import numpy as np
import pytest

# Floating species S degrading in a constant compartment, plus a parameter P
# that is constant between events and set to 5 by an event at t > 0.5. P never
# participates in a reaction or rule — it is pure event-driven state.
SBML_EVENT_PARAM = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="event_param">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
      <parameter id="P" value="0" constant="false"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="bump" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.5</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="P">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn>5</cn>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_event_promoted_parameter_is_not_a_species_column():
    """P is mutated only by an event: full integrator state + observable, but
    not a floating-species trajectory column."""
    model = bngsim.Model.from_sbml_string(SBML_EVENT_PARAM)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 1.0), n_points=21, rtol=1e-10, atol=1e-12
    )
    names = list(r.species_names)

    # Only the floating species S is a trajectory column — not P.
    assert names == ["S"]
    assert "P" not in names
    assert r.species.shape[1] == 1

    # P is still exposed as a same-named observable (so referencing expressions
    # resolve its live value), and the event actually drove it 0 → 5.
    assert "P" in r.observable_names
    P = np.asarray(r.observables["P"])
    t = np.asarray(r.time)
    assert P[t <= 0.5].max() == pytest.approx(0.0)
    assert P[t > 0.5].min() == pytest.approx(5.0)

    # S itself integrates correctly (S0·e^(-k·t)), unaffected by the hidden P.
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], np.exp(-t), rtol=1e-5, atol=1e-7)


def test_event_promoted_parameter_absent_from_cdat_export(tmp_path):
    """The C++ .cdat export projects to reported species too — P must not appear
    in its header or data columns."""
    model = bngsim.Model.from_sbml_string(SBML_EVENT_PARAM)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 1.0), n_points=6, rtol=1e-10, atol=1e-12
    )
    out = tmp_path / "traj.cdat"
    r.to_cdat(out)
    text = out.read_text()
    header = text.splitlines()[0]
    assert "S" in header
    assert "P" not in header.split()
    # One time column + one species column (S), nothing else.
    assert len(text.splitlines()[1].split()) == 2
