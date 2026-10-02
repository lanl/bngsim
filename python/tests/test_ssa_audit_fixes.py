"""SSA/PSA defects found by a correctness audit.

- A run with a tiny total propensity and a trigger that reads the clock probed
  the trigger on a grid out to its next firing, far past t_end: it hung,
  ignoring ``timeout=``.
- An SBML conversionFactor is folded into the rate, which keeps the ODE and the
  SSA mean but not the SSA's noise (each firing should move the species by
  cf x stoichiometry): such a model is refused under SSA/PSA.
- (Kept as is: a run_until leg rounds a fractional count an event wrote in the
  previous leg, with the #718 warning.)
"""

from __future__ import annotations

import time

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")


def _sim(text, method="ssa"):
    kw = {"poplevel": 100} if method == "psa" else {}
    return bngsim.Simulator(bngsim.Model.from_antimony_string(text), method=method, **kw)


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_tiny_propensity_beside_a_clock_trigger_does_not_hang(method):
    text = "species A = 1; J0: A => ; 1e-8*A; E1: at time >= 2: A = A + 1;"
    t0 = time.perf_counter()
    r = _sim(text, method).run(t_span=(0, 4), n_points=5, seed=1, timeout=20)
    assert time.perf_counter() - t0 < 5
    assert np.asarray(r.species)[-1, 0] == 2


CF = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="A" compartment="c" initialAmount="4" hasOnlySubstanceUnits="true"
            boundaryCondition="false" constant="false" conversionFactor="cf"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="cf" value="{cf}" constant="true"/><parameter id="k" value="3" constant="true"/>
  </listOfParameters>
  <listOfReactions>
   <reaction id="R0" reversible="false">
    <listOfProducts>
     <speciesReference species="A" stoichiometry="1" constant="true"/>
    </listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_conversion_factor_is_refused_under_ssa(method):
    m = bngsim.Model.from_sbml_string(CF.format(cf=2))
    assert any(i.code == "conversion_factor" for i in m.validate_for_ssa())
    kw = {"poplevel": 100} if method == "psa" else {}
    with pytest.raises(bngsim.SsaValidationError, match="conversionFactor"):
        bngsim.Simulator(m, method=method, **kw)
    # The ODE keeps it: dA/dt = cf*k.
    r = bngsim.Simulator(bngsim.Model.from_sbml_string(CF.format(cf=2))).run(
        t_span=(0, 10), n_points=2
    )
    assert np.asarray(r.species)[-1, 0] == pytest.approx(4 + 2 * 3 * 10)


def test_a_unit_conversion_factor_is_not_refused():
    m = bngsim.Model.from_sbml_string(CF.format(cf=1))
    assert not any(i.code == "conversion_factor" for i in m.validate_for_ssa())
    bngsim.Simulator(m, method="ssa").run(t_span=(0, 1), n_points=2, seed=1)


def test_a_fractional_count_an_event_wrote_is_rounded_loudly_by_the_next_leg():
    """A = A/2 with A = 7 writes 3.5 molecules, which the run carries (#692). A
    run_until leg that continues it rounds it at its start, as every start
    rounds a fractional count (#718): the legs then differ from the run whole,
    but not silently."""
    text = "species A = 7; species B = 0; J: => B; 1; E: at time >= 1: A = A/2;"
    whole = _sim(text).run(t_span=(0, 4), n_points=5, seed=3)
    assert np.asarray(whole.species)[2, 0] == 3.5
    s = _sim(text)
    s.run_until(2, seed=3)
    with pytest.warns(bngsim.SsaRoundingWarning):
        s.run_until(4, seed=4)


def test_an_event_just_past_a_leg_end_fires_once():
    """``time > 4`` with legs ending at 4 and 8: the first leg looked past its
    end, fired it, and the second fired it again (B = 200)."""
    text = "species B = 0; species C = 0; J: => C; 1; E: at time > 4: B = B + 100;"
    s = _sim(text)
    s.run_until(4, seed=1)
    r = s.run_until(8, seed=2)
    assert np.asarray(r.species)[-1, 0] == 100


CF_IA = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="A" compartment="c" initialAmount="10" hasOnlySubstanceUnits="true"
            boundaryCondition="false" constant="false" conversionFactor="cf"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="cf" value="2" constant="true"/><parameter id="k" value="3" constant="true"/>
  </listOfParameters>
  <listOfInitialAssignments>
   <initialAssignment symbol="sr">
    <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math>
   </initialAssignment>
  </listOfInitialAssignments>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfReactants>
     <speciesReference species="A" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <listOfProducts>
     <speciesReference id="sr" species="A" stoichiometry="1" constant="true"/>
    </listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""


def test_a_conversion_factor_with_a_stoichiometry_from_an_initial_assignment():
    """A -> 2A with the 2 from an initialAssignment, cf = 2: dA/dt = cf*k, so
    A(10) = 70 (libRoadRunner agrees). The cf used to be read against the
    attribute's 1 -> 1, a catalyst, and dropped: 40 under the ODE, and SSA ran."""
    r = bngsim.Simulator(bngsim.Model.from_sbml_string(CF_IA)).run(t_span=(0, 10), n_points=2)
    assert np.asarray(r.species)[-1, 0] == pytest.approx(70.0)
    m = bngsim.Model.from_sbml_string(CF_IA)
    assert any(i.code == "conversion_factor" for i in m.validate_for_ssa())
