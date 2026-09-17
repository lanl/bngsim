"""The interactive clock must name the state the model holds (issue #553).

``run_until`` used to assign ``self._current_time = t`` *after* ``run()``
returned. Stop conditions are evaluated against a completed result, so
``run()`` raises ``StopConditionMet`` after the backend has integrated the whole
interval and written the end-of-span concentrations back into the model — and
that assignment was skipped. The simulator was left holding the t=100 state with
its clock still reading 0.0, and the next ``run_until`` built its time axis from
the stale clock: a trajectory whose first row is the t=100 state under a label
saying t=0, with no error or warning.

``run()`` now advances the clock itself, the moment the backend writes the state
back, so every post-solve raise leaves the two agreeing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import Model, Simulator, SsaBoundaryWarning, StopConditionMet

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

# X -> 0 at a constant rate: X must go negative, which trips SsaBoundaryWarning.
SBML_CONSTANT_SINK = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
  <listOfSpecies>
   <species id="X" compartment="c" initialAmount="20" hasOnlySubstanceUnits="true"/>
  </listOfSpecies>
  <listOfParameters><parameter id="k" value="10"/></listOfParameters>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfReactants><speciesReference species="X"/></listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""

DECAY_A0 = 100.0
DECAY_K = 0.1


def _decay(data_dir: Path) -> Simulator:
    return Simulator(Model.from_net(str(data_dir / "simple_decay.net")), method="ode")


def _analytic(t: float) -> float:
    return DECAY_A0 * np.exp(-DECAY_K * t)


class TestClockAfterStopCondition:
    def test_current_time_is_where_the_backend_left_the_state(self, data_dir: Path) -> None:
        """The reported stop time and the clock differ, and the clock is the honest one."""
        sim = _decay(data_dir)
        sim.add_stop_condition("A_tot < 50", label="half")
        with pytest.raises(StopConditionMet) as excinfo:
            sim.run_until(t=100)

        assert excinfo.value.result.time[-1] == pytest.approx(7.0)
        # The model holds the end-of-interval state, not the trigger-point one.
        assert sim.model.get_state()[0] == pytest.approx(_analytic(100.0), rel=1e-4)
        assert sim.current_time == 100.0

    def test_next_leg_time_axis_matches_its_own_first_row(self, data_dir: Path) -> None:
        """The headline symptom: a Result labelled t=0 whose first row was the t=100 state."""
        sim = _decay(data_dir)
        sim.add_stop_condition("A_tot < 50", label="half")
        with pytest.raises(StopConditionMet):
            sim.run_until(t=100)
        sim.clear_stop_conditions()

        result = sim.run_until(t=150)
        assert result.time[0] == pytest.approx(100.0)
        assert result.time[-1] == pytest.approx(150.0)
        # Row 0 is the t=100 state and the axis now says so.
        assert result.species[0, 0] == pytest.approx(_analytic(100.0), rel=1e-4)
        assert result.species[-1, 0] == pytest.approx(_analytic(150.0), rel=1e-3)
        assert sim.current_time == 150.0

    def test_a_target_behind_the_advanced_state_is_refused(self, data_dir: Path) -> None:
        """Refusing beats relabelling: the old clock made this leg look legitimate."""
        sim = _decay(data_dir)
        sim.add_stop_condition("A_tot < 50", label="half")
        with pytest.raises(StopConditionMet):
            sim.run_until(t=100)
        sim.clear_stop_conditions()

        with pytest.raises(ValueError, match=r"must be > current time \(100"):
            sim.run_until(t=50)

    def test_snapshot_restore_still_returns_to_the_leg_start(self, data_dir: Path) -> None:
        """The documented way to continue from before the stop, clock included."""
        sim = _decay(data_dir)
        sim.run_until(t=10)
        snap = sim.snapshot()
        sim.add_stop_condition("A_tot < 5", label="low")
        with pytest.raises(StopConditionMet):
            sim.run_until(t=100)
        assert sim.current_time == 100.0

        sim.restore(snap)
        assert sim.current_time == 10.0
        assert sim.model.get_state()[0] == pytest.approx(_analytic(10.0), rel=1e-4)


class TestClockAfterOtherPostSolveRaises:
    def test_ssa_boundary_warning_as_error_leaves_the_clock_current(self, tmp_path: Path) -> None:
        """StopConditionMet is not the only post-solve raise that stranded the clock.

        ``_warn_ssa_boundary`` runs after the backend has written the state back
        too, so under ``-W error`` it strands the clock the same way. X drains at
        a constant rate and goes negative, which is what trips the warning.
        """
        path = tmp_path / "sink.xml"
        path.write_text(SBML_CONSTANT_SINK)
        sim = Simulator(bngsim.Model.from_sbml(str(path)), method="ssa")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SsaBoundaryWarning)
            with pytest.raises(SsaBoundaryWarning):
                sim.run_until(t=5)
        assert sim.current_time == 5.0


class TestClockOnTheOrdinaryPaths:
    def test_run_until_endpoints_are_exact(self, data_dir: Path) -> None:
        """The clock is read off the result, and must still land exactly on the target."""
        sim = _decay(data_dir)
        sim.run_until(t=10.0)
        assert sim.current_time == 10.0
        sim.run_until(t=20.0)
        assert sim.current_time == 20.0

    def test_plain_run_advances_the_clock_too(self, data_dir: Path) -> None:
        """run() moves the state, so it owns the clock; run_until then continues correctly."""
        sim = _decay(data_dir)
        assert sim.current_time == 0.0
        sim.run(t_span=(0, 100), n_points=3)
        assert sim.current_time == 100.0

        result = sim.run_until(t=150)
        assert result.time[0] == pytest.approx(100.0)
        assert result.species[0, 0] == pytest.approx(_analytic(100.0), rel=1e-4)

    def test_parameter_scan_leaves_the_clock_as_it_found_it(self, data_dir: Path) -> None:
        """The scan rewinds the model when it finishes; the clock is part of that."""
        sim = _decay(data_dir)
        sim.run_until(t=10.0)
        before_time, before_state = sim.current_time, sim.model.get_state().copy()

        sim.parameter_scan("k1", [0.1, 0.2], t_span=(0, 5), n_points=3)

        assert sim.current_time == before_time
        np.testing.assert_allclose(sim.model.get_state(), before_state)

    def test_clock_starts_at_zero(self, data_dir: Path) -> None:
        assert _decay(data_dir).current_time == 0.0


def test_stop_condition_result_cannot_rewind_a_promoted_state_model(tmp_path: Path) -> None:
    """Why the fix moves the clock rather than rolling the model back (issue #553).

    The alternative — rewind the model to the trigger point — needs a full
    integrator state row, and the truncated result cannot supply one: it is built
    without a core, so ``Result.state`` refuses, and its species block omits any
    promoted entry. Pinning this keeps the rejected design from being reattempted
    on the assumption that ``species`` is the whole state.
    """
    path = tmp_path / "promoted.xml"
    path.write_text(SBML_PROMOTED_PARAMETER)
    model = bngsim.Model.from_sbml(str(path))
    assert "k" in model.species_names

    sim = Simulator(model, method="ode")
    sim.add_stop_condition(lambda r: bool(r.time[-1] >= 1.0), label="t1")
    with pytest.raises(StopConditionMet) as excinfo:
        sim.run(t_span=(0, 2), n_points=21)

    partial = excinfo.value.result
    assert partial.species.shape[1] < len(model.get_state())
    with pytest.raises(ValueError, match=r"only available on a Result returned by a solve"):
        _ = partial.state
