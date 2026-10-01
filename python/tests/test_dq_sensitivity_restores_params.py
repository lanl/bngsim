"""CVODES' difference-quotient probes leave no trace outside the RHS (issue #690).

When the analytic sensitivity RHS is declined (here by ``abs()``), CVODES
differences the RHS itself: it perturbs ``sens_p[which]`` to p ± delta, calls the
RHS, and restores ``sens_p``. The RHS callbacks mirror ``sens_p`` into the model's
parameters and the codegen buffer, but nothing mirrored the restore back, so
between probes the model held p - delta for the LAST sensitivity parameter.
Every reader outside the RHS saw it: a recorded function reading that parameter
was off by ~sqrt(rtol), ``get_param`` returned the probed value after the run,
repeated runs compounded it (2.0 → 1.998 → 1.996 → …), and an event assignment
reading it landed ~1e-3 off. The callbacks now put the nominal values back
after every probe.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

_NET = """
begin parameters
    1 k1     1.0
    2 k2     0.3
    3 scale  2.0
    4 A0     10
end parameters
begin functions
    1 rf() k1*abs(Atot)
    2 Yobs() scale*Btot
end functions
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 rf
    2 2 0 k2
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""

_T = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)
_MML = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_EVENT_SBML = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="ev">
  <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialConcentration="10" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="0.5" constant="true"/>
   <parameter id="reset" value="5" constant="true"/>
   <parameter id="ton" value="1" constant="true"/>
  </listOfParameters>
  <listOfReactions>
   <reaction id="deg" reversible="false">
    <listOfReactants>
     <speciesReference species="S" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <kineticLaw><math {_MML}><apply><times/><ci>k</ci><apply><abs/><ci>S</ci></apply></apply>
    </math></kineticLaw>
   </reaction>
  </listOfReactions>
  <listOfEvents>
   <event id="e1" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true">
     <math {_MML}><apply><geq/>{_T}<ci>ton</ci></apply></math>
    </trigger>
    <listOfEventAssignments>
     <eventAssignment variable="S"><math {_MML}><ci>reset</ci></math></eventAssignment>
    </listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>"""


@pytest.fixture(autouse=True)
def _quiet_decline():
    # abs() declines the analytic sensitivity RHS on purpose: that is the path
    # under test, and the decline notice is expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _net_model(tmp_path: Path) -> bngsim.Model:
    net = tmp_path / "m1.net"
    net.write_text(textwrap.dedent(_NET).strip() + "\n")
    return bngsim.Model.from_net(str(net))


def test_a_function_reading_the_last_sensitivity_parameter_is_exact(tmp_path):
    """Yobs = scale*Btot with scale last in sensitivity_params: it came back
    1e-3 low, the relative size of the probe."""
    m = _net_model(tmp_path)
    r = bngsim.Simulator(m, sensitivity_params=["k1", "scale"]).run((0, 5), 6, rtol=1e-6)
    y = np.asarray(r.expressions)[:, list(r.expression_names).index("Yobs")]
    b = np.asarray(r.observables)[:, list(r.observable_names).index("Btot")]
    np.testing.assert_array_equal(y, 2.0 * b)


def test_the_parameter_is_unchanged_after_the_run_and_does_not_drift(tmp_path):
    """get_param('scale') read 1.998 after one run and 1.992011992002 after four."""
    m = _net_model(tmp_path)
    sim = bngsim.Simulator(m, sensitivity_params=["k1", "scale"])
    for _ in range(4):
        m.reset()
        sim.run((0, 5), 6, rtol=1e-6)
        assert m.get_param("scale") == 2.0
        assert m.get_param("k1") == 1.0


def test_an_event_assignment_reading_a_sensitivity_parameter_is_exact():
    """S := reset at t = 1 landed 1e-3 low; the state and both sensitivity
    columns now match the closed form."""
    m = bngsim.Model.from_sbml_string(_EVENT_SBML)
    r = bngsim.Simulator(m, sensitivity_params=["k", "reset"]).run(
        (0, 2), 5, rtol=1e-8, atol=1e-10
    )
    t = np.asarray(r.time)
    s = np.asarray(r.species)[:, 0]
    sens = np.asarray(r.sensitivities)[:, 0, :]
    after = t >= 1
    exact = np.where(after, 5 * np.exp(-0.5 * (t - 1)), 10 * np.exp(-0.5 * t))
    dk = np.where(after, -5 * (t - 1) * np.exp(-0.5 * (t - 1)), -10 * t * np.exp(-0.5 * t))
    dreset = np.where(after, np.exp(-0.5 * (t - 1)), 0.0)
    np.testing.assert_allclose(s, exact, rtol=1e-6)
    np.testing.assert_allclose(sens[:, 0], dk, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(sens[:, 1], dreset, rtol=1e-5, atol=1e-7)
    assert m.get_param("reset") == 5.0


def test_a_derived_parameter_off_the_snapshot_path_still_gets_its_parameters_back():
    """The restore copies a snapshot of the nominal point only when re-deriving
    there is provably a function of the listed built-ins of the parameters.
    ``kt`` calls ``erf``, which that check does not list, so this model takes
    the re-deriving restore instead, and must end the run at nominal all the
    same (issue #690). (It read ``time()`` until the builder began refusing a
    parameter that does.)"""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    a = b.add_species("A", 10.0)
    bb = b.add_species("B", 0.0)
    b.add_parameter("k", 0.5)
    b.add_parameter("kt", 0.0, "k*(1 + 0.1*erf(k))", is_expression=True)
    b.add_observable("Atot", [(a, 1.0)])
    b.add_function("rf", "kt*abs(Atot)")
    b.add_reaction([a], [bb], "functional", "rf")
    m = bngsim.Model(_core=b.build())
    sim = bngsim.Simulator(m, sensitivity_params=["k"])
    for _ in range(3):
        m.reset()
        sim.run((0, 5), 6, rtol=1e-6)
        assert m.get_param("k") == 0.5
