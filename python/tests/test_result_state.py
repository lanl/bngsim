"""``Result.state`` / ``Result.state_names`` (issue #508): the full integrator state.

``Result.species`` is the *reported* species block: on an SBML model whose event assigns
a parameter or a compartment, the promoted state entry is projected out (GH #71), and a
remapped column reports a value the integrator did not hold (GH #85, #131). A row of it
is therefore not a state ``Model.rhs`` / ``Model.jacobian`` accept — too short, or not the
point — and the private evaluator hooks used to read such a row past its end. ``state`` is
the trajectory the integrator actually held, and the hooks now refuse a wrong length.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import Simulator

# S degrades at k·S; an event at t >= 1 assigns the (non-constant) parameter k, which
# promotes k to a state entry the species projection hides.
SBML_PROMOTED_PARAMETER = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="promoted">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="false"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><apply><times/><ci>k</ci><ci>S</ci></apply></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="slow" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
              <cn>1</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="k">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.05</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>
"""


@pytest.fixture
def promoted(tmp_path):
    p = tmp_path / "promoted.xml"
    p.write_text(SBML_PROMOTED_PARAMETER)
    return bngsim.Model.from_sbml(str(p))


def _solve(m, t_end=2.0, n_points=21):
    return Simulator(m, method="ode").run(t_span=(0.0, t_end), n_points=n_points)


def test_net_model_state_is_the_species_block(data_dir):
    m = bngsim.Model.from_net(str(data_dir / "simple_decay.net"))
    res = _solve(m, 5.0, 6)
    np.testing.assert_array_equal(res.state, res.species)
    assert res.state_names == res.species_names == list(m.species_names)


def test_event_promoted_parameter_is_in_state_but_not_in_species(promoted):
    m = promoted
    assert "k" in m.species_names, "the event target should be promoted to a state entry"
    res = _solve(m)
    assert res.state_names == list(m.species_names)
    assert "k" in res.state_names and "k" not in res.species_names
    assert res.state.shape == (21, len(m.species_names))
    assert res.state.shape[1] == res.species.shape[1] + 1
    k = res.state[:, res.state_names.index("k")]
    assert k[0] == 0.5 and k[-1] == pytest.approx(0.05)  # the event fired
    for j, name in enumerate(res.species_names):  # the reported columns are the same numbers
        np.testing.assert_array_equal(res.state[:, res.state_names.index(name)], res.species[:, j])


def test_state_rows_are_what_the_evaluators_accept(promoted):
    m = promoted
    res = _solve(m)
    n = len(m.species_names)
    J = m.jacobian(res.state[-1], t=res.time[-1])
    assert J.shape == (n, n) and np.isfinite(J).all()
    assert np.isfinite(m.rhs(res.state[-1], t=res.time[-1])).all()
    with pytest.raises(ValueError, match=f"expected {n} species values, got {n - 1}"):
        m.jacobian(res.species[-1], t=res.time[-1])


def test_private_hooks_refuse_a_wrong_length_state(promoted):
    """The GH #76 test hooks read exactly n_species entries: a shorter list was read past
    its end (undefined behaviour, NaN on one run and a number on the next)."""
    core = promoted._core
    n = len(promoted.species_names)
    good = [1.0] * n
    assert len(core._eval_rhs(0.0, good)) == n
    assert len(core._dense_analytical_jacobian(0.0, good)) == n * n
    with pytest.raises(ValueError, match=f"_eval_rhs: expected {n} species values, got {n - 1}"):
        core._eval_rhs(0.0, good[:-1])
    with pytest.raises(
        ValueError, match=f"_dense_analytical_jacobian: expected {n} species values, got {n + 1}"
    ):
        core._dense_analytical_jacobian(0.0, good + [1.0])


def test_a_result_built_from_arrays_carries_no_state():
    res = bngsim.Result(
        _time=np.array([0.0, 1.0]),
        _species=np.ones((2, 1)),
        _species_names=["A"],
        _observables=np.zeros((2, 0)),
        _observable_names=[],
    )
    with pytest.raises(ValueError, match="returned by a solve"):
        _ = res.state
    with pytest.raises(ValueError, match="returned by a solve"):
        _ = res.state_names
