"""GH #212: forward sensitivity through fixed-time (Phase-1) events.

GH #205 originally refused output sensitivities on *any* model with events: the
integrator reinitialises state at an event (``CVodeReInit``) but the CVODES
forward-sensitivity vectors were never reinitialised, so the columns went
silently stale at and after the first fire.

GH #212 lifts that refusal for the **Phase-1 subclass** — fixed-time,
persistent, no-delay events (``g = time − T``, the dosing/stimulation pattern).
For that class the event-time sensitivity ``∂t*/∂p = 0`` and the core applies
the jump ``s⁺ = J_h·s⁻ + ∂h/∂p`` plus ``CVodeSensReInit`` at each fire. The
remaining subclasses keep raising: state-dependent triggers and parameter-valued
trigger times (Phase 2), delays and non-persistent triggers (Phase 3).

Three groups are asserted:

  * **Phase-1 allowed + correct** — fixed-time event models now run and the
    ``output_sensitivities`` match an independent central finite-difference
    across the event (constant reset, additive bolus, parameter-valued reset).
  * **Still unsupported** — state-dependent trigger, delay, non-persistent, and
    a parameter-valued trigger time all raise with a clear reason.
  * **No false positives** — plain (non-sensitivity) runs and discontinuity-
    trigger models (forcing pulses; ``n_events == 0``) are unaffected.
"""

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

# ── SBML event model: S degrades, an event resets S:=100 at t>=1 (fixed-time) ─
SBML_EVENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ev">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
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
      <event id="bump" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
              <cn>1</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="S">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>100</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

# ── Discontinuity-trigger model: a piecewise-time forcing pulse on parameter
# `inp` drives production of X. n_discontinuity_triggers > 0 but n_events == 0 —
# the pulse breaks the integrator step yet never jumps state, so forward
# sensitivities through it stay valid and must NOT be refused. ────────────────
SBML_PULSE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="pulse_train">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kin" value="100" constant="true"/>
      <parameter id="d" value="1" constant="true"/>
      <parameter id="inp" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="inp">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece>
              <ci>kin</ci>
              <apply><and/>
                <apply><geq/>
                  <csymbol encoding="text"
                    definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.7</cn></apply>
                <apply><leq/>
                  <csymbol encoding="text"
                    definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.75</cn></apply>
              </apply>
            </piece>
            <otherwise><cn>0</cn></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="prod" reversible="false">
        <listOfProducts>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><ci>inp</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><ci>d</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# ── Builders / FD helpers ───────────────────────────────────────────────────


def _decay_with_event(assign_expr, trigger="time() >= 5", extra_params=None, **event_kwargs):
    """dS/dt = -k·S, S(0)=100, plus one event ``S := assign_expr`` at ``trigger``."""
    b = ModelBuilder()
    b.add_parameter("k", 0.1)
    for name, val in extra_params or []:
        b.add_parameter(name, val)
    s = b.add_species("S", 100.0)
    b.add_reaction([s], [], "elementary", "k")
    b.add_observable("Sobs", [(s, 1.0)])
    b.add_event("evt", trigger, [(s, assign_expr)], **event_kwargs)
    return bngsim.Model(_core=b.build()), s


def _fd_output_sens(assign_expr, param, p0, extra_params=None, h=1e-6, **event_kwargs):
    """Central-difference ``d(Sobs)/dparam`` on bngsim's own trajectory."""

    def traj(pv):
        m, _ = _decay_with_event(assign_expr, extra_params=extra_params, **event_kwargs)
        m.set_param(param, pv)
        r = bngsim.Simulator(m, method="ode").run(t_span=(0, 10), n_points=101)
        return np.asarray(r.outputs(["observable:Sobs"]))[:, 0]

    step = h * abs(p0) if p0 else h
    return (traj(p0 + step) - traj(p0 - step)) / (2 * step)


def _analytic_output_sens(assign_expr, param, extra_params=None, **event_kwargs):
    m, _ = _decay_with_event(assign_expr, extra_params=extra_params, **event_kwargs)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=[param])
    r = sim.run(t_span=(0, 10), n_points=101)
    return np.asarray(r.time), np.asarray(r.output_sensitivities(["observable:Sobs"]))[:, 0, 0]


def _assert_matches_fd(assign_expr, param, p0, extra_params=None, rtol=1e-4, **event_kwargs):
    t, ana = _analytic_output_sens(assign_expr, param, extra_params, **event_kwargs)
    fd = _fd_output_sens(assign_expr, param, p0, extra_params, **event_kwargs)
    i_evt = int(np.argmin(np.abs(t - 5.0)))  # event fires at t=5
    assert np.isfinite(ana).all()
    scale = np.maximum(np.abs(fd), np.abs(ana))
    mask = scale > 1e-6
    relerr = np.abs(ana[mask] - fd[mask]) / scale[mask]
    assert relerr.max() < rtol, f"max relerr {relerr.max():.2e} >= {rtol}"
    # Specifically check the points straddling the discontinuity.
    for i in (i_evt - 1, i_evt + 1, -1):
        assert ana[i] == pytest.approx(fd[i], rel=1e-3, abs=1e-6)
    return t, ana, fd


# ── Phase-1 allowed + numerically correct ───────────────────────────────────


class TestPhase1Allowed:
    def test_constant_reset_matches_fd(self):
        # S := 2.0 at t=5: sensitivity is zeroed at the event (∂h/∂x=∂h/∂p=0),
        # then regrows; must match FD across the discontinuity.
        t, ana, fd = _assert_matches_fd("2.0", "k", 0.1)
        i = int(np.argmin(np.abs(t - 5.0)))
        assert abs(ana[i + 1]) < abs(ana[i - 1])  # dropped at the event

    def test_additive_bolus_is_continuous(self):
        # S := S + 50 at t=5: ∂h/∂x = 1, ∂h/∂p = 0 ⇒ s⁺ = s⁻ (continuous).
        _assert_matches_fd("S + 50", "k", 0.1)

    def test_parameter_valued_reset_matches_fd(self):
        # S := dose at t=5 with `dose` the sensitivity parameter: the jump must
        # pick up ∂h/∂dose = 1, so d(Sobs)/d(dose) jumps to ~1 then decays.
        t, ana, fd = _assert_matches_fd("dose", "dose", 7.0, extra_params=[("dose", 7.0)])
        i = int(np.argmin(np.abs(t - 5.0)))
        assert abs(ana[i - 1]) < 1e-6  # no dependence before the dose
        assert ana[i + 1] == pytest.approx(1.0, abs=0.05)  # unit jump after

    def test_sbml_event_model_runs_and_is_finite(self):
        m = bngsim.Model.from_sbml_string(SBML_EVENT)
        assert m._core.n_events == 1
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 3), n_points=31)
        assert r.sensitivities.shape == (31, m._core.n_species, 1)
        assert np.all(np.isfinite(r.sensitivities))


# ── Still unsupported subclasses (raise loudly) ─────────────────────────────


class TestStillUnsupported:
    def test_state_dependent_trigger_raises(self):
        m, _ = _decay_with_event("2.0", trigger="S < 50")
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match="state-dependent"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_delayed_event_raises(self):
        m, _ = _decay_with_event("2.0", delay=1.0)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match="delay"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_nonpersistent_event_raises(self):
        m, _ = _decay_with_event("2.0", persistent=False)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match="non-persistent"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_trigger_time_parameter_raises(self):
        # time() >= t_dose with t_dose requested as a sensitivity parameter:
        # the crossing time moves with the parameter (∂t*/∂p ≠ 0) — Phase 2.
        m, _ = _decay_with_event("2.0", trigger="time() >= t_dose", extra_params=[("t_dose", 5.0)])
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k", "t_dose"])
        with pytest.raises(ValueError, match="t_dose"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_trigger_time_parameter_ok_when_not_sensitized(self):
        # Same trigger but t_dose NOT among the requested params ⇒ ∂t*/∂p = 0.
        m, _ = _decay_with_event("2.0", trigger="time() >= t_dose", extra_params=[("t_dose", 5.0)])
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 10), n_points=11)
        assert np.all(np.isfinite(r.sensitivities))

    def test_compute_all_sensitivities_with_trigger_time_param_raises(self):
        # compute_all_sensitivities defaults to ALL params, which includes the
        # trigger-time parameter t_dose, so it must refuse.
        m, _ = _decay_with_event("2.0", trigger="time() >= t_dose", extra_params=[("t_dose", 5.0)])
        sim = bngsim.Simulator(m, method="ode")
        with pytest.raises(ValueError, match="t_dose"):
            sim.compute_all_sensitivities(t_span=(0, 10), n_points=11)


# ── No false positives ──────────────────────────────────────────────────────


class TestNoFalsePositives:
    def test_plain_run_on_event_model_still_works(self):
        m, _ = _decay_with_event("2.0")
        r = bngsim.Simulator(m, method="ode").run(t_span=(0, 10), n_points=11)
        assert r.species.shape[0] == 11

    def test_compute_all_sensitivities_fixed_time_allowed(self):
        # No trigger-time parameter ⇒ the full-tensor entry point is allowed.
        m, _ = _decay_with_event("2.0")
        res = bngsim.Simulator(m, method="ode").compute_all_sensitivities(
            t_span=(0, 10), n_points=11
        )
        assert res is not None

    def test_discontinuity_trigger_model_allows_sensitivities(self):
        m = bngsim.Model.from_sbml_string(SBML_PULSE)
        assert m._core.n_events == 0
        assert m._core.n_discontinuity_triggers > 0
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["kin", "d"])
        r = sim.run(t_span=(0, 2), n_points=21, max_step=0.01)
        assert r.sensitivities.shape == (21, m._core.n_species, 2)
        assert np.all(np.isfinite(r.sensitivities))
