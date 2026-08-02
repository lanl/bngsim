"""GH #212 / issue #49 / issue #144: forward sensitivity through events.

GH #205 originally refused output sensitivities on *any* model with events: the
integrator reinitialises state at an event (``CVodeReInit``) but the CVODES
forward-sensitivity vectors were never reinitialised, so the columns went
silently stale at and after the first fire.

GH #212 lifted that refusal for **fixed-time** events (``g = time − T``, the
dosing/stimulation pattern). For that class the event-time sensitivity
``∂t*/∂p = 0`` and the core applies the jump ``s⁺ = J_h·s⁻ + ∂h/∂p`` plus
``CVodeSensReInit`` at each fire.

Issue #49 lifted it for the case where the crossing time ITSELF moves — an
onset written as ``time >= T0`` with ``T0`` fitted, which is the same modelling
intent as the ``piecewise(kin, time >= T0, 0)`` issue #48 already supported and
has the same gradient. The jump then carries all four terms::

    s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p

with ∂t*/∂p supplied by ``bngsim._switch_sensitivity.compute_event_time_sens``.
Same issue: a delay of literal ``0`` is not a delay, and ``persistent=false``
without a delay has no window to act in, so neither is refused any more.

Issue #144 lifted it for a **state-dependent** trigger — ``v > 30``, whose
crossing time moves with every parameter through the trajectory while naming
none of them. There is nothing to resolve ahead of the run, so the solver
differentiates the crossing where it happens, by the implicit function theorem
on the trigger's residual ``g``::

    dt*/dθ = − (∂g/∂x·S(t*⁻) + ∂g/∂p) / (∂g/∂t + ∂g/∂x·f(x⁻))

feeding the same four-term jump. That covers the initial-condition columns too,
which issue #49's parameter-only ``∂t*/∂p`` does not.

Five groups are asserted:

  * **Fixed-time allowed + correct** — fixed-time event models run and the
    ``output_sensitivities`` match an independent central finite-difference
    across the event (constant reset, additive bolus, parameter-valued reset,
    and an assignment that reads the species it writes).
  * **Still unsupported** — real delays, and triggers with no single crossing
    surface (conjunction, disjunction, negation, equality), raise with a clear
    reason.
  * **State-dependent triggers (issue #144)** — closed forms where they exist,
    finite differences on AMICI's ``neuron`` fixture where they do not, the IC
    column's own crossing shift, a derived-parameter threshold's chain rule,
    and a crossing rate too cancelled to resolve, which refuses.
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


# ── AMICI's `neuron` fixture: the Izhikevich spiking model, which is the
# reproduction issue #144 was filed on. Two rate rules
#
#     dv/dt = 0.04v² + 5v + 140 − u + I0        du/dt = a·(b·v − u)
#
# and one event `v > 30` → `v := c; u := u + d`. Every ingredient of the jump is
# live at once: a state-dependent crossing whose time moves with all four
# parameters, an assignment that RESETS a row to a parameter (∂h/∂p), and an
# assignment that READS the row it writes (∂h/∂x). ────────────────────────────
SBML_NEURON = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="neuron">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="v" compartment="cell" initialConcentration="-60"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="u" compartment="cell" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="a" value="0.02" constant="true"/>
      <parameter id="b" value="0.3" constant="true"/>
      <parameter id="c" value="-65" constant="true"/>
      <parameter id="d" value="2" constant="true"/>
      <parameter id="I0" value="10" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="v">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/>
            <apply><times/><cn>0.04</cn><apply><power/><ci>v</ci><cn>2</cn></apply></apply>
            <apply><times/><cn>5</cn><ci>v</ci></apply>
            <cn>140</cn>
            <apply><minus/><ci>u</ci></apply>
            <ci>I0</ci>
          </apply>
        </math>
      </rateRule>
      <rateRule variable="u">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
            <ci>a</ci>
            <apply><minus/><apply><times/><ci>b</ci><ci>v</ci></apply><ci>u</ci></apply>
          </apply>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="spike" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>v</ci><cn>30</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="v">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>c</ci></math>
          </eventAssignment>
          <eventAssignment variable="u">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><plus/><ci>u</ci><ci>d</ci></apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

NEURON_NOMINAL = {"a": 0.02, "b": 0.3, "c": -65.0, "d": 2.0}


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

    def test_sbml_assignment_reading_its_own_species_carries_dh_dx(self):
        """``S := 0.5·S`` at a fixed time: ∂h/∂x = 0.5, so s⁺ = s⁻/2.

        This is a fixed-time event — no ∂t*/∂p anywhere — but it was answered
        wrongly until issue #144, and the mechanism is issue #52's shadowing on
        the *assignment* side. ModelBuilder registers a species as an ExprTk
        variable only when its name is free, and an SBML model gives each
        species a same-named observable, so the ``S`` in the assignment binds to
        the observable total. The jump's ∂h/∂x difference was restricted to
        concentration addresses, matched nothing, and reported ∂h/∂x = 0 — which
        restarts the column from zero. dS/dt = −kS is linear, so zero is a fixed
        point and the whole post-event column stayed there.

        Closed form: S = 10·e^{−kt} before, 5·e^{−kt} after (the halving at t=1
        is undone by e^{+k} in the restart), so dS/dk = −10t·e^{−kt} then
        −5t·e^{−kt}. The broken answer was 0 at t=1 and only regrew from there.
        """
        m = bngsim.Model.from_sbml_string(
            SBML_EVENT.replace(
                '<math xmlns="http://www.w3.org/1998/Math/MathML"><cn>100</cn></math>',
                '<math xmlns="http://www.w3.org/1998/Math/MathML">'
                "<apply><times/><cn>0.5</cn><ci>S</ci></apply></math>",
            )
        )
        r = bngsim.Simulator(m, method="ode", sensitivity_params=["k"]).run(
            t_span=(0, 4), n_points=9, rtol=1e-11, atol=1e-13
        )
        t = np.asarray(r.time)
        ana = np.asarray(r.sensitivities)[:, 0, 0]
        expected = np.where(t < 1.0, -10.0 * t * np.exp(-0.5 * t), -5.0 * t * np.exp(-0.5 * t))
        np.testing.assert_allclose(ana, expected, rtol=1e-6, atol=1e-8)
        # The defect's signature: the column reset to 0 exactly at the event.
        assert abs(ana[t == 1.0][0]) > 3.0


# ── Still unsupported subclasses (raise loudly) ─────────────────────────────


class TestStillUnsupported:
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

    @pytest.mark.parametrize(
        "trigger, wanted",
        [
            ("S < 50 && time() >= 1", "combines conditions"),
            ("S < 50 || time() >= 1", "combines conditions"),
            ("not(S < 50)", "combines conditions"),
            ("S == 50", "equality test"),
            ("if(S < 50, 1, 0)", "not a relational comparison"),
        ],
    )
    def test_trigger_without_one_crossing_surface_raises(self, trigger, wanted):
        """Issue #144 differentiates ONE relational comparison.

        A conjunction, disjunction or negation has a true-set boundary
        assembled from several surfaces, and which one carries the rising edge
        can change as a parameter moves; an equality is satisfied on a
        measure-zero set rather than crossed. Each refusal says which of those
        it is and quotes the trigger, so a user can tell a limitation from a
        bug.
        """
        m, _ = _decay_with_event("2.0", trigger=trigger)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match=wanted):
            sim.run(t_span=(0, 10), n_points=11)

    def test_refusal_quotes_the_trigger_as_written(self):
        m, _ = _decay_with_event("2.0", trigger="S < 50 && time() >= 1")
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(ValueError, match=r"S < 50 && time\(\) >= 1"):
            sim.run(t_span=(0, 10), n_points=11)


# ── State-dependent triggers: dt*/dθ at the crossing (issue #144) ────────────
#
# The crossing time of `v > 30` moves with every parameter through the
# trajectory, so ∂t*/∂θ is non-zero and cannot be resolved before the run the
# way issue #49 resolves a clock threshold. The solver differentiates it at the
# fire instead, by the implicit function theorem on the trigger residual g:
#
#     dt*/dθ = − (∂g/∂x·S(t*⁻) + ∂g/∂p) / (∂g/∂t + ∂g/∂x·f(x⁻))
#
# Every test here checks a *number*, not just that the refusal is gone: the
# pre-#52 behaviour on these models was to answer with a tensor missing the
# event contribution, which is what makes "it runs" worthless as an assertion.


def _state_trigger_closed_form(t, k=0.1):
    """``dS/dk`` for dS/dt = −kS, S(0)=100, event ``S < 50`` → ``S := 2``.

    The crossing is at t* = ln2/k, after which S = 2·e^{−k(t−t*)} = 4·e^{−kt}.
    Both branches are exponentials in k, so the whole column is closed-form —
    and it is the ONLY thing that pins ∂t*/∂k: with the constant reset,
    ∂h/∂x = ∂h/∂p = 0 and the post-event column is exactly −f⁺·∂t*/∂k.
    """
    t_star = np.log(2.0) / k
    return np.where(t < t_star, -100.0 * t * np.exp(-k * t), -4.0 * t * np.exp(-k * t))


class TestStateDependentTrigger:
    def test_matches_the_closed_form(self):
        m, _ = _decay_with_event("2.0", trigger="S < 50")
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 20), n_points=41, rtol=1e-10, atol=1e-12)
        ana = np.asarray(r.sensitivities)[:, 0, 0]
        np.testing.assert_allclose(
            ana, _state_trigger_closed_form(np.asarray(r.time)), rtol=1e-6, atol=1e-6
        )

    def test_dropping_dtstar_would_be_visibly_wrong(self):
        """Guard against a regression that silently zeroes the crossing term.

        Without ∂t*/∂k the constant reset leaves the post-event column at 0 and
        it stays there (dS/dt = −kS is linear, so s ≡ 0 is a fixed point). The
        closed form says −4·t·e^{−kt} ≈ −13.9 just after the crossing, so the
        two answers are not near neighbours.
        """
        m, _ = _decay_with_event("2.0", trigger="S < 50")
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 20), n_points=41, rtol=1e-10, atol=1e-12)
        t = np.asarray(r.time)
        after = np.asarray(r.sensitivities)[t > np.log(2.0) / 0.1, 0, 0]
        assert np.abs(after).min() > 1.0

    def test_sbml_state_dependent_trigger_is_answered(self):
        """Issue #52's shadowing, now on the supported side.

        ModelBuilder registers a species as an ExprTk variable only when the
        name is free, and SBML models routinely give each species an observable
        of the same name — so the trigger's token binds to the observable
        total, not to ``&sp.concentration``. Before issue #52 that hid the
        state dependence and the model was answered with a tensor missing the
        event contribution (on AMICI's ``neuron``, 6x-135x off). Issue #52
        refused it; issue #144 answers it correctly, and the finite difference
        is what says which of those two this is.
        """
        m = bngsim.Model.from_sbml_string(_sbml_event_with_state_trigger())
        assert "S" in list(m._core.species_names)
        assert "S" in list(m._core.observable_names)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        r = sim.run(t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-12)
        ana = np.asarray(r.sensitivities)[:, 0, 0]

        def traj(kv):
            mm = bngsim.Model.from_sbml_string(_sbml_event_with_state_trigger())
            mm.set_param("k", kv)
            rr = bngsim.Simulator(mm, method="ode").run(
                t_span=(0, 10), n_points=21, rtol=1e-11, atol=1e-13
            )
            return np.asarray(rr.species)[:, 0]

        h = 0.5 * 1e-5
        fd = (traj(0.5 + h) - traj(0.5 - h)) / (2 * h)
        np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-6)

    def test_every_entry_point_answers(self):
        """``compute_all_sensitivities`` takes the same guard."""
        m = bngsim.Model.from_sbml_string(_sbml_event_with_state_trigger())
        out = bngsim.Simulator(m, method="ode").compute_all_sensitivities(
            t_span=(0, 10), n_points=11
        )
        assert np.all(np.isfinite(np.asarray(out.sensitivities)))

    def test_initial_condition_column_carries_the_crossing_shift(self):
        """An IC column's ∂t*/∂x(0) is non-zero, unlike issue #49's.

        Issue #49 covers parameter columns only — an initial condition cannot
        move a clock. Here it plainly can: start higher and the trajectory
        reaches the threshold later. With the constant reset the post-event
        column is exactly ``−f⁺·∂t*/∂S(0)``, and ∂t*/∂S(0) = 1/(k·S(0)) from
        t* = ln(S(0)/50)/k — so the whole column is closed-form.
        """
        m, _ = _decay_with_event("2.0", trigger="S < 50")
        sim = bngsim.Simulator(m, method="ode", sensitivity_ic=["S"])
        r = sim.run(t_span=(0, 20), n_points=41, rtol=1e-10, atol=1e-12)
        t = np.asarray(r.time)
        ana = np.asarray(r.sensitivities_ic)[:, 0, 0]
        k, s0 = 0.1, 100.0
        t_star = np.log(s0 / 50.0) / k
        # Before: S = S0·e^{−kt} ⇒ ∂S/∂S0 = e^{−kt}.
        # After:  S = 2·e^{−k(t−t*)} with ∂t*/∂S0 = 1/(k·S0) ⇒
        #         ∂S/∂S0 = 2k·e^{−k(t−t*)}/(k·S0) = 2·e^{−k(t−t*)}/S0.
        expected = np.where(t < t_star, np.exp(-k * t), 2.0 * np.exp(-k * (t - t_star)) / s0)
        np.testing.assert_allclose(ana, expected, rtol=1e-6, atol=1e-8)

    def test_unresolvable_crossing_rate_refuses_rather_than_dividing(self):
        """The transversality condition is the denominator of dt*/dθ.

        Deliberately conditioned: two zero-order productions whose rates differ
        by ``drift`` out of ``bulk``, with the trigger on ``A − B``. The
        trajectory crosses the surface perfectly cleanly — ``A − B = 1 −
        drift·t`` — but the *rate* at which it crosses is the difference of two
        rates 1e8 times larger, and dg/dt is assembled from exactly those two
        terms. Below 1e-8 of the term scale the quotient reports the finite
        differences' own noise, so it is refused with that as the stated reason
        rather than answered.

        (The other half of the guard — a denominator at the absolute noise
        floor of ``f``, i.e. a trajectory that has stopped moving through the
        surface — is a knife-edge condition on the trajectory and is not
        constructible as a fixture; it shares this code path and this message.)
        """
        b = ModelBuilder()
        b.add_parameter("bulk", 5.0e7)
        b.add_parameter("bulk_plus", 5.0e7 + 0.5)
        a = b.add_species("A", 1.0)
        bb = b.add_species("B", 0.0)
        b.add_observable("Aobs", [(a, 1.0)])
        b.add_observable("Bobs", [(bb, 1.0)])
        b.add_reaction([], [a], "elementary", "bulk")
        b.add_reaction([], [bb], "elementary", "bulk_plus")
        b.add_event("evt", "Aobs - Bobs < 0.5", [(a, "2.0")])
        m = bngsim.Model(_core=b.build())
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["bulk"])
        with pytest.raises(Exception, match="tangentially"):
            sim.run(t_span=(0, 3), n_points=13, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("param", ["a", "b", "c", "d"])
    def test_neuron_matches_finite_differences(self, param):
        """The issue #144 reproduction: AMICI's ``neuron``, all four parameters.

        The tolerance is loose on purpose — this is a spiking trajectory whose
        sensitivities run to 1e5 because every spike time moves with every
        parameter, so a fixed-``t`` comparison is inherently ill-conditioned.
        What makes it a real check is not the number but that the disagreement
        falls like h²: dropping ∂t*/∂θ, or dropping the ∂h/∂x term the ``u``
        assignment needs, leaves a residue that does not move with h at all.
        """

        def traj(overrides):
            m = bngsim.Model.from_sbml_string(SBML_NEURON)
            for k, v in overrides.items():
                m.set_param(k, v)
            r = bngsim.Simulator(m, method="ode").run(
                t_span=(0, 40), n_points=41, rtol=1e-11, atol=1e-12, max_steps=200000
            )
            return np.asarray(r.species)

        m = bngsim.Model.from_sbml_string(SBML_NEURON)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=[param])
        r = sim.run(t_span=(0, 40), n_points=41, rtol=1e-11, atol=1e-12, max_steps=200000)
        ana = np.asarray(r.sensitivities)[:, :, 0]
        assert np.all(np.isfinite(ana))

        p0 = NEURON_NOMINAL[param]
        errs = []
        for h_rel in (1e-4, 1e-5):
            h = abs(p0) * h_rel
            fd = (traj({param: p0 + h}) - traj({param: p0 - h})) / (2 * h)
            scale = max(np.max(np.abs(fd)), np.max(np.abs(ana)), 1e-12)
            errs.append(np.max(np.abs(fd - ana)) / scale)
        assert errs[-1] < 1e-5, f"relative error {errs[-1]:.2e} at the smallest step"
        # Truncation, not a missing term: a factor-10 smaller step must help.
        assert errs[-1] < errs[0]

    def test_initial_value_fire_does_not_move_with_theta(self):
        """SBML L3 §3.4.5: a t=0 fire is pinned, not located.

        The trigger is already satisfied when the run begins, so the fire
        happens at ``t_start`` for every θ in a neighbourhood and ∂t*/∂θ = 0 —
        differentiating the trigger there answers with the rate at which a
        crossing that is not happening would move. The IC column is what
        catches it: ``s⁻`` for a parameter column is 0 at t=0 so a spurious
        ∂t*/∂p multiplies out, but ∂S/∂S(0) starts at 1 and a spurious shift
        lands on it. After ``S := 2`` the true answer is exactly 0 — the reset
        forgets the initial condition — for the whole run.
        """
        m, _ = _decay_with_event("2.0", trigger="S < 200", initial_value=False)
        sim = bngsim.Simulator(m, method="ode", sensitivity_ic=["S"])
        r = sim.run(t_span=(0, 10), n_points=21, rtol=1e-10, atol=1e-12)
        ana = np.asarray(r.sensitivities_ic)[:, 0, 0]
        np.testing.assert_allclose(ana, np.zeros_like(ana), atol=1e-9)

    def test_event_triggered_by_another_event_refuses(self):
        """SBML "events triggering events" is a second jump at the same instant.

        The cascade riser's assignments run after the root batch's, so they read
        a state that already jumped, while the sensitivity jump is keyed on the
        root batch and takes every derivative at the pre-batch x⁻. Composing the
        two is real work; until then the rows the cascade writes would keep the
        sensitivity of the value they held before it, which is the GH #205
        stale-column hazard. Unreachable before issue #144 — a cascade riser
        needs a state-dependent trigger, and those were refused outright.
        """

        def build():
            b = ModelBuilder()
            b.add_parameter("k", 0.1)
            s = b.add_species("S", 100.0)
            u = b.add_species("U", 0.0)
            b.add_reaction([s], [], "elementary", "k")
            b.add_event("dose", "time() >= 2", [(s, "1000.0")])  # fixed-time
            b.add_event("sense", "S > 500", [(u, "1.0")])  # armed by the dose
            return bngsim.Model(_core=b.build())

        sim = bngsim.Simulator(build(), method="ode", sensitivity_params=["k"])
        with pytest.raises(Exception, match="another event's assignment"):
            sim.run(t_span=(0, 5), n_points=11)

        # Without sensitivities the same model still simulates normally — the
        # cascade is legal SBML, it is only its *derivative* that is missing.
        plain = bngsim.Simulator(build(), method="ode").run(t_span=(0, 5), n_points=11)
        assert np.asarray(plain.species)[-1, 1] == pytest.approx(1.0)

    def test_derived_parameter_threshold_carries_the_chain_rule(self):
        """``S < half`` with ``half = 0.5*S_ref``: perturbing ``S_ref`` moves the
        threshold, hence the crossing, hence the whole post-event column.

        The trigger references ``half``'s address and never ``S_ref``'s, so a
        difference over the *referenced* addresses alone reports ∂g/∂S_ref = 0
        and the column comes back as if the threshold were fixed.
        """
        b = ModelBuilder()
        b.add_parameter("k", 0.1)
        b.add_parameter("S_ref", 100.0)
        b.add_parameter("half", 0.0, expression="0.5 * S_ref", is_expression=True)
        s = b.add_species("S", 100.0)
        b.add_reaction([s], [], "elementary", "k")
        b.add_event("evt", "S < half", [(s, "2.0")])
        m = bngsim.Model(_core=b.build())
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["S_ref"])
        r = sim.run(t_span=(0, 20), n_points=41, rtol=1e-10, atol=1e-12)
        t = np.asarray(r.time)
        ana = np.asarray(r.sensitivities)[:, 0, 0]
        # S(0) is a literal 100, so only the threshold moves: t* = ln(100/half)/k
        # ⇒ ∂t*/∂S_ref = −1/(k·S_ref), and after the reset S = 2·e^{−k(t−t*)}
        # ⇒ ∂S/∂S_ref = 2k·e^{−k(t−t*)}·∂t*/∂S_ref = −2·e^{−k(t−t*)}/S_ref.
        k, s_ref = 0.1, 100.0
        t_star = np.log(2.0) / k
        expected = np.where(t < t_star, 0.0, -2.0 * np.exp(-k * (t - t_star)) / s_ref)
        np.testing.assert_allclose(ana, expected, rtol=1e-5, atol=1e-7)


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
