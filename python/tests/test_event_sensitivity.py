"""GH #212 / issue #49: forward sensitivity through events.

GH #205 originally refused output sensitivities on *any* model with events: the
integrator reinitialises state at an event (``CVodeReInit``) but the CVODES
forward-sensitivity vectors were never reinitialised, so the columns went
silently stale at and after the first fire.

GH #212 lifted that refusal for the **Phase-1 subclass** — fixed-time events
(``g = time − T``, the dosing/stimulation pattern). For that class the
event-time sensitivity ``∂t*/∂p = 0`` and the core applies the jump
``s⁺ = J_h·s⁻ + ∂h/∂p`` plus ``CVodeSensReInit`` at each fire.

Issue #49 lifts it for the case where the crossing time ITSELF moves — an
onset written as ``time >= T0`` with ``T0`` fitted, which is the same modelling
intent as the ``piecewise(kin, time >= T0, 0)`` issue #48 already supported and
has the same gradient. The jump then carries all four terms::

    s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p

with ∂t*/∂p supplied by ``bngsim._switch_sensitivity.compute_event_time_sens``.
Same issue: a delay of literal ``0`` is not a delay, and ``persistent=false``
without a delay has no window to act in, so neither is refused any more.

Four groups are asserted:

  * **Phase-1 allowed + correct** — fixed-time event models run and the
    ``output_sensitivities`` match an independent central finite-difference
    across the event (constant reset, additive bolus, parameter-valued reset).
  * **Still unsupported** — state-dependent triggers and real delays raise with
    a clear reason.
  * **No false positives** — plain (non-sensitivity) runs and discontinuity-
    trigger models (forcing pulses; ``n_events == 0``) are unaffected.
  * **Event-time sensitivity (issue #49)** — the onset column matches the
    closed form, finite differences, and the ``piecewise`` encoding of the same
    dynamics; it does not drift with ``rtol``; and a threshold that is not
    arithmetic over the model's primary parameters is refused rather than
    attributed to the wrong column.
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


def _sbml_event_with_state_trigger() -> str:
    """``SBML_EVENT`` with its fixed-time trigger swapped for a state-dependent
    one (``S < 5``) — the shape of AMICI's ``neuron`` fixture, whose ``v > 30``
    reads a state variable while naming no parameter (issue #52)."""
    fixed_time = """<apply><geq/>
              <csymbol encoding="text"
                definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
              <cn>1</cn></apply>"""
    assert fixed_time in SBML_EVENT, "SBML_EVENT trigger changed; update this helper"
    return SBML_EVENT.replace(fixed_time, "<apply><lt/><ci>S</ci><cn>5</cn></apply>")


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

    def test_sbml_state_dependent_trigger_raises(self):
        """Issue #52: the same refusal, reached through SBML.

        The guard tests the trigger's *bound addresses*, and ModelBuilder
        registers a species as an ExprTk variable only when the name is free.
        SBML models routinely give each species an observable of the same name,
        so the species registration is skipped and the trigger's token binds to
        the observable total instead of ``&sp.concentration``. Checking
        concentrations alone therefore saw no state dependence here and answered
        the model — on AMICI's ``neuron`` fixture (Izhikevich, trigger
        ``v > 30``) the sensitivities came back 6x-135x off, uniformly in one
        direction, rather than being refused.
        """
        m = bngsim.Model.from_sbml_string(_sbml_event_with_state_trigger())
        assert m._core.n_events == 1
        # Precondition for the bug: species and observable share the name, which
        # is what pushed the trigger's binding onto the observable total.
        assert "S" in list(m._core.species_names)
        assert "S" in list(m._core.observable_names)

        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match="state-dependent"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_sbml_state_dependent_trigger_refused_for_every_entry_point(self):
        """``compute_all_sensitivities`` takes the same guard, so it must refuse
        the same model rather than quietly returning a tensor missing the event
        contributions."""
        m = bngsim.Model.from_sbml_string(_sbml_event_with_state_trigger())
        with pytest.raises(ValueError, match="state-dependent"):
            bngsim.Simulator(m, method="ode").compute_all_sensitivities(
                t_span=(0, 10), n_points=11
            )

    def test_delayed_event_raises(self):
        m, _ = _decay_with_event("2.0", delay=1.0)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match="delay"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_trigger_time_parameter_ok_when_not_sensitized(self):
        # Same trigger but t_dose NOT among the requested params ⇒ ∂t*/∂p = 0.
        m, _ = _decay_with_event("2.0", trigger="time() >= t_dose", extra_params=[("t_dose", 5.0)])
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 10), n_points=11)
        assert np.all(np.isfinite(r.sensitivities))

    def test_state_dependent_trigger_detail_names_the_atom(self):
        """Issue #49 routes the trigger through the same clock-threshold
        recognizer the rate-law path uses, and reports what it could not
        reduce — so a refusal says which comparison defeated it."""
        m, _ = _decay_with_event("2.0", trigger="S < 50")
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match="'S < 50'"):
            sim.run(t_span=(0, 10), n_points=11)


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


# ── Issue #49: the crossing time itself moves (∂t*/∂p ≠ 0) ──────────────────
#
# An onset written as an event — `time >= T0` firing `on := 1`, with the rate
# laws reading `on` — is the same modelling intent as the `piecewise(kin, time
# >= T0, 0)` issue #48 covers, and has the same gradient. Both corners of the
# general jump
#
#     s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p
#
# were already implemented (GH #212 at ∂t*/∂p = 0, issue #48 at h = identity);
# this exercises the middle, where both halves are live at once.

KIN, KOUT, T0 = 2.0, 0.3, 2.5


def _onset_model(T0v=T0, trigger="time() >= T0", **event_kwargs):
    """dX/dt = kin·on − kout·X, with an event switching ``on`` 0 → 1 at ``T0``.

    The issue's exemplar shape: the fitted parameter appears ONLY in the event
    trigger, so ∂f/∂T0 is a genuine zero and the whole gradient is the jump.
    """
    b = ModelBuilder()
    b.add_parameter("kin", KIN)
    b.add_parameter("kout", KOUT)
    b.add_parameter("T0", T0v)
    b.add_parameter("toff", 100.0)
    x = b.add_species("X", 0.0)
    on = b.add_species("on", 0.0)
    # `on` is reactant AND product, so d(on)/dt = 0 and the rate is kin·on.
    b.add_reaction([on], [on, x], "elementary", "kin")
    b.add_reaction([x], [], "elementary", "kout")
    b.add_observable("Xobs", [(x, 1.0)])
    b.add_event("onset", trigger, [(on, "1.0")], **event_kwargs)
    return bngsim.Model(_core=b.build())


def _onset_closed_form(t, T0v=T0):
    """X(t) and dX/dT0 for the onset model.

    X = (kin/kout)(1 − e^(−kout·(t−T0)))  for t ≥ T0, else 0
    dX/dT0 = −kin·e^(−kout·(t−T0))        for t ≥ T0, else 0
    """
    t = np.asarray(t, dtype=float)
    z = np.maximum(t - T0v, 0.0)
    x = (KIN / KOUT) * (1.0 - np.exp(-KOUT * z))
    dx = np.where(t >= T0v, -KIN * np.exp(-KOUT * z), 0.0)
    return x, dx


def _piecewise_onset_model(T0v=T0):
    """The SAME dynamics written as issue #48's `if()` rate law instead.

    dX/dt = if(time() >= T0, kin, 0) − kout·X. Nothing about the trajectory or
    the gradient differs — only the encoding — which is the asymmetry issue #49
    exists to remove.
    """
    b = ModelBuilder()
    b.add_parameter("kin", KIN)
    b.add_parameter("kout", KOUT)
    b.add_parameter("T0", T0v)
    x = b.add_species("X", 0.0)
    b.add_function("kin_gate", "if(time() >= T0, kin, 0)")
    b.add_parameter("kin_gate", 0.0)  # builder auto-binds function → parameter
    b.add_reaction([], [x], "elementary", "kin_gate")
    b.add_reaction([x], [], "elementary", "kout")
    b.add_observable("Xobs", [(x, 1.0)])
    return bngsim.Model(_core=b.build())


def _xobs_sens(model, params, t_span=(0, 10), n_points=101, **run_kwargs):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    r = sim.run(t_span=t_span, n_points=n_points, **run_kwargs)
    return (
        np.asarray(r.time),
        np.asarray(r.output_sensitivities(["observable:Xobs"]))[:, 0, :],
    )


def _fd_over_builds(build, selector, p0, h=1e-4, t_span=(0, 10), n_points=101):
    """Central difference of the trajectory itself.

    ``build(pv)`` must return the model rebuilt AT ``pv`` rather than a model
    with ``set_param`` applied afterwards: a trigger threshold is read at fire
    time, but rebuilding is what keeps this oracle honest for parameters that
    also seed initial conditions (GH #79).
    """

    def traj(pv):
        r = bngsim.Simulator(build(pv), method="ode").run(t_span=t_span, n_points=n_points)
        return np.asarray(r.outputs([selector]))[:, 0]

    step = h * abs(p0) if p0 else h
    return (traj(p0 + step) - traj(p0 - step)) / (2.0 * step)


class TestEventTimeSensitivity:
    def test_onset_matches_closed_form(self):
        t, sens = _xobs_sens(_onset_model(), ["T0", "kin"])
        _x, dx = _onset_closed_form(t)
        ana = sens[:, 0]
        assert np.all(np.isfinite(ana))
        # Skip the single sample that lands exactly ON the crossing: the
        # analytic column is right-continuous there (−kin) while any central
        # difference of the trajectory averages the two sides (−kin/2).
        mask = (np.abs(dx) > 1e-9) & (np.abs(t - T0) > 1e-12)
        relerr = np.abs(ana[mask] - dx[mask]) / np.abs(dx[mask])
        assert relerr.max() < 1e-6, f"max relerr {relerr.max():.3e}"

    def test_onset_matches_finite_differences(self):
        t, sens = _xobs_sens(_onset_model(), ["T0"])
        fd = _fd_over_builds(lambda p: _onset_model(T0v=p), "observable:Xobs", T0)
        mask = (np.abs(fd) > 1e-6) & (np.abs(t - T0) > 1e-12)
        relerr = np.abs(sens[mask, 0] - fd[mask]) / np.abs(fd[mask])
        assert relerr.max() < 1e-4, f"max relerr {relerr.max():.3e}"

    def test_event_and_piecewise_encodings_agree(self):
        """The acceptance criterion the issue leads with: identical dynamics,
        identical gradient, and now identical support."""
        t_ev, s_ev = _xobs_sens(_onset_model(), ["T0"])
        t_pw, s_pw = _xobs_sens(_piecewise_onset_model(), ["T0"])
        assert np.allclose(t_ev, t_pw)
        mask = np.abs(s_pw[:, 0]) > 1e-6
        relerr = np.abs(s_ev[mask, 0] - s_pw[mask, 0]) / np.abs(s_pw[mask, 0])
        assert relerr.max() < 1e-5, f"max relerr {relerr.max():.3e}"

    def test_answer_does_not_drift_with_rtol(self):
        """Issue #48's failure mode: a jump term read at whatever point CVODES'
        last finite-difference probe left the parameters at comes out scaled by
        1 ∓ √rtol, so the answer moves when rtol does. f⁻/f⁺ multiply ∂t*/∂p
        here, so this path has the same exposure."""
        _t, loose = _xobs_sens(_onset_model(), ["T0"], rtol=1e-6)
        _t, tight = _xobs_sens(_onset_model(), ["T0"], rtol=1e-10)
        mask = np.abs(tight[:, 0]) > 1e-6
        relerr = np.abs(loose[mask, 0] - tight[mask, 0]) / np.abs(tight[mask, 0])
        assert relerr.max() < 1e-4, f"rtol-driven drift {relerr.max():.3e}"

    def test_parameter_valued_reset_on_a_moving_trigger(self):
        """Both ∂h/∂p ≠ 0 and ∂t*/∂p ≠ 0 — the term neither existing corner
        exercises. ``S := dose`` at ``time() >= t_dose``, differentiated w.r.t.
        both."""
        p0 = {"t_dose": 5.0, "dose": 7.0}

        def build(t_dose=p0["t_dose"], dose=p0["dose"]):
            m, _ = _decay_with_event(
                "dose",
                trigger="time() >= t_dose",
                extra_params=[("t_dose", t_dose), ("dose", dose)],
            )
            return m

        sim = bngsim.Simulator(build(), method="ode", sensitivity_params=["t_dose", "dose"])
        r = sim.run(t_span=(0, 10), n_points=101)
        t = np.asarray(r.time)
        ana = np.asarray(r.output_sensitivities(["observable:Sobs"]))[:, 0, :]

        for col, name in enumerate(("t_dose", "dose")):
            rebuild = (
                (lambda v: build(t_dose=v)) if name == "t_dose" else (lambda v: build(dose=v))
            )
            fd = _fd_over_builds(rebuild, "observable:Sobs", p0[name])
            mask = (np.abs(fd) > 1e-5) & (np.abs(t - p0["t_dose"]) > 1e-12)
            relerr = np.abs(ana[mask, col] - fd[mask]) / np.abs(fd[mask])
            assert relerr.max() < 1e-3, f"{name}: max relerr {relerr.max():.3e}"

        # Sanity on the shape of the t_dose column: before the dose the reset
        # has not happened, so moving its time changes nothing.
        i = int(np.argmin(np.abs(t - p0["t_dose"])))
        assert abs(ana[i - 1, 0]) < 1e-8

    def test_conjunction_uses_the_later_lower_bound(self):
        """`ton <= time && time <= toff` rises at ton, so ∂t*/∂ton = 1 and
        ∂t*/∂toff = 0 — the upper bound moves the event's *end*, which is not a
        firing at all."""
        m = _onset_model(trigger="(time() >= T0) && (time() <= toff)")
        t, sens = _xobs_sens(m, ["T0", "toff"])
        _x, dx = _onset_closed_form(t)
        mask = (np.abs(dx) > 1e-9) & (np.abs(t - T0) > 1e-12)
        relerr = np.abs(sens[mask, 0] - dx[mask]) / np.abs(dx[mask])
        assert relerr.max() < 1e-6
        assert np.abs(sens[:, 1]).max() < 1e-9  # toff moves nothing

    def test_derived_threshold_chains_to_its_primaries(self):
        """`T0 = t_base + t_delta` puts the jump on both primaries, with the
        same chain rule the issue #48 detector uses."""
        b = ModelBuilder()
        b.add_parameter("kin", KIN)
        b.add_parameter("kout", KOUT)
        b.add_parameter("t_base", 1.5)
        b.add_parameter("t_delta", 1.0)
        b.add_parameter("T0", 0.0, "t_base + t_delta", True)
        x = b.add_species("X", 0.0)
        on = b.add_species("on", 0.0)
        b.add_reaction([on], [on, x], "elementary", "kin")
        b.add_reaction([x], [], "elementary", "kout")
        b.add_observable("Xobs", [(x, 1.0)])
        b.add_event("onset", "time() >= T0", [(on, "1.0")])
        m = bngsim.Model(_core=b.build())

        t, sens = _xobs_sens(m, ["t_base", "t_delta"])
        _x, dx = _onset_closed_form(t, T0v=2.5)
        mask = (np.abs(dx) > 1e-9) & (np.abs(t - 2.5) > 1e-12)
        for col in (0, 1):
            relerr = np.abs(sens[mask, col] - dx[mask]) / np.abs(dx[mask])
            assert relerr.max() < 1e-6, f"col {col}: {relerr.max():.3e}"

    def test_assignment_rule_threshold_chains_to_its_primaries(self):
        """A threshold that is an SBML ``<assignmentRule>`` parameter is NOT a
        constant, even though ``param_is_expression`` is false for it and
        reading its current value looks like reading a literal.

        BIOMD0000000301 writes its pulse schedule as ``pulse2_start =
        pulse1_start + pulse1_length + pulse_interval``. Attributing the whole
        ∂t*/∂p to ``pulse2_start`` put the gradient on a column the fitter does
        not move and left ``pulse1_start``'s at zero — a confidently wrong
        answer, which is worse than the refusal it replaced.
        """
        from bngsim._switch_sensitivity import compute_event_time_sens

        b = ModelBuilder()
        b.add_parameter("kin", KIN)
        b.add_parameter("kout", KOUT)
        b.add_parameter("t_first", 1.0)
        b.add_parameter("gap", 1.5)
        # The var_param_binding idiom the SBML loader uses for an assignment
        # rule on a parameter: a function whose name matches a parameter's.
        b.add_function("t_second", "t_first + gap")
        b.add_parameter("t_second", 0.0)
        x = b.add_species("X", 0.0)
        on = b.add_species("on", 0.0)
        b.add_reaction([on], [on, x], "elementary", "kin")
        b.add_reaction([x], [], "elementary", "kout")
        b.add_observable("Xobs", [(x, 1.0)])
        b.add_event("onset", "time() >= t_second", [(on, "1.0")])
        m = bngsim.Model(_core=b.build())

        res = compute_event_time_sens(m._core, ["t_first", "gap", "t_second"], 0.0, 10.0)
        assert res.compensated == [0]
        # ∂t*/∂t_first = ∂t*/∂gap = 1; the rule-bound name itself gets nothing.
        assert res.records == [(0, [1.0, 1.0, 0.0])]

        t, sens = _xobs_sens(m, ["t_first", "gap"])
        _x, dx = _onset_closed_form(t, T0v=2.5)
        mask = (np.abs(dx) > 1e-9) & (np.abs(t - 2.5) > 1e-12)
        for col in (0, 1):
            relerr = np.abs(sens[mask, col] - dx[mask]) / np.abs(dx[mask])
            assert relerr.max() < 1e-6, f"col {col}: {relerr.max():.3e}"

    @pytest.mark.parametrize("name", ["onset_t", "del", "lambda"])
    def test_a_python_keyword_threshold_parameter_is_just_a_name(self, name):
        """Issue #105. ``time() >= del + gap`` used to come back with correct
        partials but no *value*, because ``_evaluate_threshold`` parsed the
        threshold without the keyword alias map its sibling already applied. The
        trigger was then not compensated and the whole model was refused forward
        sensitivity — ``MODEL1710030000`` (``time >= del + N*stepT``) is the one
        corpus model that hit this.

        Parametrized over ``lambda`` too: it is 43 of the 46 corpus models with a
        keyword-named parameter, so it is the likelier next encounter.
        """
        from bngsim._switch_sensitivity import compute_event_time_sens

        b = ModelBuilder()
        b.add_parameter("kin", KIN)
        b.add_parameter("kout", KOUT)
        b.add_parameter(name, 1.5)
        b.add_parameter("gap", 1.0)  # onset at 2.5 = T0
        x = b.add_species("X", 0.0)
        on = b.add_species("on", 0.0)
        b.add_reaction([on], [on, x], "elementary", "kin")
        b.add_reaction([x], [], "elementary", "kout")
        b.add_observable("Xobs", [(x, 1.0)])
        b.add_event("onset", f"time() >= {name} + gap", [(on, "1.0")])
        m = bngsim.Model(_core=b.build())

        res = compute_event_time_sens(m._core, [name, "gap"], 0.0, 10.0)
        assert res.reasons == {}, res.reasons
        assert res.compensated == [0]
        assert res.records == [(0, [1.0, 1.0])]

        # And the gradient itself, against the closed form the ordinary-name
        # twin already matches — a compensated record that produced the wrong
        # number would pass the assertions above.
        t, sens = _xobs_sens(m, [name, "gap"])
        _x, dx = _onset_closed_form(t)
        mask = (np.abs(dx) > 1e-9) & (np.abs(t - T0) > 1e-12)
        for col in (0, 1):
            relerr = np.abs(sens[mask, col] - dx[mask]) / np.abs(dx[mask])
            assert relerr.max() < 1e-6, f"col {col}: {relerr.max():.3e}"

    def test_unresolvable_rule_threshold_is_refused_not_guessed(self):
        """The same shape, but the rule reads state. Its value at the current
        point still *looks* like a number; treating it as one would attribute
        ∂t*/∂p to a parameter that does not move the crossing."""
        b = ModelBuilder()
        b.add_parameter("kin", KIN)
        b.add_parameter("kout", KOUT)
        b.add_parameter("t_first", 1.0)
        x = b.add_species("X", 1.0)
        on = b.add_species("on", 0.0)
        b.add_observable("Xobs", [(x, 1.0)])
        b.add_function("t_rule", "t_first + Xobs")
        b.add_parameter("t_rule", 0.0)
        b.add_reaction([on], [on, x], "elementary", "kin")
        b.add_reaction([x], [], "elementary", "kout")
        b.add_event("onset", "time() >= t_rule", [(on, "1.0")])
        m = bngsim.Model(_core=b.build())

        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["t_first"])
        with pytest.raises(ValueError, match="does not reduce to arithmetic"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_compute_all_sensitivities_includes_the_trigger_parameter(self):
        """The full-tensor entry point defaults to every parameter, so it used
        to be refused outright on any model with a fitted onset."""
        m = _onset_model()
        res = bngsim.Simulator(m, method="ode").compute_all_sensitivities(
            t_span=(0, 10), n_points=51
        )
        assert res is not None
        assert np.all(np.isfinite(np.asarray(res.sensitivities)))

    def test_zero_delay_expression_is_not_a_delay(self):
        """Issue #49 sub-finding: 25 of the 88 blocked corpus events also
        tripped the delay check, and in every case the delay was a literal 0 —
        which process_firing_batch already treats as an immediate fire, so
        there is no trigger-time-to-execution-time window to worry about."""
        m = _onset_model(delay_expr="0")
        t, sens = _xobs_sens(m, ["T0"])
        _x, dx = _onset_closed_form(t)
        mask = (np.abs(dx) > 1e-9) & (np.abs(t - T0) > 1e-12)
        relerr = np.abs(sens[mask, 0] - dx[mask]) / np.abs(dx[mask])
        assert relerr.max() < 1e-6

    def test_nonpersistent_without_a_delay_is_vacuous(self):
        """Per SBML L3v2 §4.11.3 `persistent` can only cancel a fire during the
        window between trigger time and execution time. A zero-delay event has
        no such window, so the flag is a no-op — Ghanbari2020 and Zongo2020
        were blocked by exactly this."""
        m, _ = _decay_with_event("2.0", persistent=False)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 10), n_points=11)
        assert np.all(np.isfinite(r.sensitivities))

    @pytest.mark.parametrize(
        "trigger,match",
        [
            ("(time() >= T0) || (time() >= toff)", "conjunction"),
            ("not(time() >= T0)", "negated"),
            ("time() == T0", "not a comparison of"),
            ("(time() >= T0) && (X > 3)", "not a comparison of"),
        ],
    )
    def test_still_refused_with_a_clear_reason(self, trigger, match):
        m = _onset_model(trigger=trigger)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["T0"])
        with pytest.raises(ValueError, match=match):
            sim.run(t_span=(0, 10), n_points=11)

    def test_real_delay_still_refused(self):
        m = _onset_model(delay=1.0)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["T0"])
        with pytest.raises(ValueError, match="delay"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_fixed_time_event_model_is_untouched(self):
        """No requested parameter moves this crossing, so no ∂t*/∂p record is
        emitted and the GH #212 jump runs exactly as before."""
        from bngsim._switch_sensitivity import compute_event_time_sens

        m = _onset_model()
        res = compute_event_time_sens(m._core, ["kin"], 0.0, 10.0)
        assert res.records == []
        assert res.compensated == [0]

    def test_crossing_outside_the_window_emits_no_record(self):
        from bngsim._switch_sensitivity import compute_event_time_sens

        m = _onset_model()
        assert compute_event_time_sens(m._core, ["T0"], 0.0, 10.0).records == [(0, [1.0])]
        # t* = 2.5 is past the end of this window: the event never fires.
        assert compute_event_time_sens(m._core, ["T0"], 0.0, 1.0).records == []
