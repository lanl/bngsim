"""ODE correctness for SBML events that change a compartment's size (#74).

When an event assignment changes a compartment's size, SBML semantics keep the
*amounts* of the species it contains unchanged and recompute their
*concentrations* (so the concentrations scale by ``V_old / V_new``). bngsim
stores every species as a concentration, so it must do two things at such an
event:

  1. **Rescale concentrations** by ``V_old / V_new`` so the amount is preserved
     (e.g. tripling the volume thirds the concentration). Implemented as
     injected per-species event assignments in ``_sbml_loader``.

  2. **Divide Functional rate laws by the LIVE compartment volume** for a
     variable-volume compartment, so the post-resize derivative is
     ``d[conc]/dt = law / V_live`` rather than letting the kinetic law's
     ``compartment * f(...)`` factor leak a stale ``V`` into the rate. Mass
     action cancels the compartment analytically, so this only matters for the
     ``common_vs == 1.0`` Functional emission path.

Surfaced by ``BIOMD0000000338`` (Wajima2009 coagulation cascade), whose
``dilution_event`` triples ``compartment_1`` at ``t>0``. Before the fix bngsim
kept concentrations constant (tripling amounts) and ran the post-event cascade
3× too fast (which also made the stiff system unintegrable at the parity tol).

Oracles here are dependency-free (closed-form for mass action; a SciPy
reference integration for the Functional law), plus an optional libRoadRunner
cross-check.
"""

import bngsim
import numpy as np
import pytest

TE = 0.7  # event time; deliberately off the sample grid below

# ── Model 1: mass-action degradation in a tripling compartment ──────────────
#
# S → ∅ with law  C·k·S   (the SBML conc→amount-flux form), k = 1, V(0) = 1.
# At t > TE an event sets C := 3·C.
#
# Mass action: d(amount)/dt = -C·k·S = -C·k·[S]  ⇒  d[S]/dt = -C·k·[S]/V = -k·[S]
# (C cancels), so the concentration decays at rate k regardless of V. The event
# only contributes the instantaneous amount-preserving ÷3 rescale:
#   [S](t) = S0·e^(-k·t)                     for t ≤ TE
#   [S](t) = (S0·e^(-k·TE)/3)·e^(-k·(t-TE))  for t  > TE
SBML_MASSACTION_RESIZE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ma_resize">
    <listOfCompartments>
      <compartment id="C" size="1" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
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
      <event id="dilute" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.7</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="C">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><ci>C</ci><cn>3</cn></apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

# ── Model 2: a Functional (saturable) law in a tripling compartment ─────────
#
# S → ∅ with law  C·sat(S, k),  sat(x, kk) = kk·x/(1+x),  k = 1, V(0) = 1.
# Same dilution event. sat is NOT a power of [S], so the compartment does not
# cancel by mass-action algebra — the loader must explicitly divide by the live
# C. The correct concentration ODE is d[S]/dt = -sat(S, k) (C-independent); the
# bug (no divide) gives d[S]/dt = -C·sat(S, k) = -3·sat(S, k) after the event.
SBML_FUNCTIONAL_RESIZE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="func_resize">
    <listOfFunctionDefinitions>
      <functionDefinition id="sat">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <bvar><ci>x</ci></bvar><bvar><ci>kk</ci></bvar>
            <apply><divide/>
              <apply><times/><ci>kk</ci><ci>x</ci></apply>
              <apply><plus/><cn>1</cn><ci>x</ci></apply></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="C" size="1" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><apply><ci>sat</ci><ci>S</ci><ci>k</ci></apply></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="dilute" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.7</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="C">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><ci>C</ci><cn>3</cn></apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

# ── Model 3: an hOSU=true species must NOT be rescaled at resize ────────────
#
# An inert hasOnlySubstanceUnits=true species H (no reactions) sits beside a
# hOSU=false species S in the tripling compartment. SBML preserves H's *amount*;
# since bngsim reads an hOSU species as amount = stored × V_c (a load-time
# constant), its stored value must stay UNCHANGED across the resize (rescaling
# would corrupt the amount). The hOSU=false neighbour drops to 1/3.
SBML_HOSU_RESIZE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu_resize">
    <listOfCompartments>
      <compartment id="C" size="2" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="H" compartment="C" initialAmount="50"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
      <species id="S" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="dummy" value="0" constant="true"/>
    </listOfParameters>
    <listOfEvents>
      <event id="dilute" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.7</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="C">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><ci>C</ci><cn>3</cn></apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def _split(t):
    """Boolean mask: True where the sample is after the dilution event."""
    return t > TE


def test_massaction_resize_matches_closed_form():
    """Mass action: amount-preserving ÷3 rescale at the event, decay rate k
    unchanged by the volume (C cancels analytically)."""
    model = bngsim.Model.from_sbml_string(SBML_MASSACTION_RESIZE)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 2),
        n_points=9,
        rtol=1e-10,
        atol=1e-12,  # samples on a 0.25 grid; TE=0.7 off-grid
    )
    t = np.asarray(r.time)
    S = np.asarray(r.species)[:, list(r.species_names).index("S")]

    S0, k = 100.0, 1.0
    pre = S0 * np.exp(-k * t)
    post = (S0 * np.exp(-k * TE) / 3.0) * np.exp(-k * (t - TE))
    expected = np.where(_split(t), post, pre)
    np.testing.assert_allclose(S, expected, rtol=1e-6, atol=1e-8)

    # Regression guard: the pre-fix loader kept concentration constant across
    # the event (no rescale), so just after TE it would read ~3× the oracle.
    after = _split(t)
    assert np.all(S[after] < 0.5 * (S0 * np.exp(-k * t[after])))


def test_functional_resize_matches_scipy_reference():
    """Functional (saturable) law: the rescale AND the live-volume divide. The
    post-event decay must follow d[S]/dt = -sat(S) (C-independent), not the 3×
    rate the missing divide produced."""
    from scipy.integrate import solve_ivp

    model = bngsim.Model.from_sbml_string(SBML_FUNCTIONAL_RESIZE)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 2), n_points=9, rtol=1e-10, atol=1e-12
    )
    t = np.asarray(r.time)
    S = np.asarray(r.species)[:, list(r.species_names).index("S")]

    def sat(x):
        return 1.0 * x / (1.0 + x)

    # Correct reference: integrate the C-independent ODE, with a ÷3 rescale at TE.
    def integ(rate_sign):
        pts = t.tolist()
        sol_pre = solve_ivp(
            lambda _t, y: -rate_sign * sat(y[0]),
            (0, TE),
            [100.0],
            t_eval=[p for p in pts if p <= TE] + [TE],
            rtol=1e-11,
            atol=1e-13,
        )
        s_te = sol_pre.y[0, -1]
        sol_post = solve_ivp(
            lambda _t, y: -rate_sign * sat(y[0]),
            (TE, 2.0),
            [s_te / 3.0],
            t_eval=[TE] + [p for p in pts if p > TE],
            rtol=1e-11,
            atol=1e-13,
        )
        out = np.empty_like(t)
        out[t <= TE] = np.interp(t[t <= TE], sol_pre.t, sol_pre.y[0])
        out[t > TE] = np.interp(t[t > TE], sol_post.t, sol_post.y[0])
        return out

    expected = integ(1.0)  # correct: rate independent of C
    buggy = integ(3.0)  # pre-fix: post-event rate 3× too fast
    np.testing.assert_allclose(S, expected, rtol=1e-5, atol=1e-6)

    # The fix must be unambiguously closer to the correct oracle than the bug.
    after = t > TE
    assert np.max(np.abs(S[after] - expected[after])) < 1e-4
    assert np.max(np.abs(S[after] - buggy[after])) > 0.5


def test_amount_is_conserved_across_resize():
    """Direct invariant: amount = [S]·V is *continuous* across the event (no
    jump — the resize only redistributes it), so it follows the smooth decay
    100·e^(-k·t) everywhere; the concentration meanwhile jumps by V_old/V_new."""
    model = bngsim.Model.from_sbml_string(SBML_MASSACTION_RESIZE)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 1.4), n_points=141, rtol=1e-10, atol=1e-12
    )
    t = np.asarray(r.time)
    names = list(r.species_names)
    S = np.asarray(r.species)[:, names.index("S")]
    # GH #71: the event-resized compartment C is promoted to internal state so
    # the engine can write its size, but it is NOT a floating species — it must
    # not appear as a trajectory column. Its live size is exposed as a
    # same-named observable instead.
    assert "C" not in names
    C = np.asarray(r.observables["C"])  # promoted compartment, observable-only
    amount = S * C

    # Amount has NO discontinuity at the event: it is the continuous solution
    # of d(amount)/dt = -k·amount everywhere (the ÷3 conc rescale and the ×3
    # volume exactly cancel in the product). A spurious amount jump (the missing
    # rescale) would break this near TE.
    np.testing.assert_allclose(amount, 100.0 * np.exp(-t), rtol=1e-5, atol=1e-6)

    # The concentration, by contrast, jumps by 1/3 at the event.
    i_pre = np.where(t <= TE)[0][-1]
    i_post = np.where(t > TE)[0][0]
    assert C[i_pre] == pytest.approx(1.0)
    assert C[i_post] == pytest.approx(3.0)
    assert S[i_post] / S[i_pre] == pytest.approx(1.0 / 3.0, rel=2e-2)


def test_hosu_species_not_rescaled_at_resize():
    """An hOSU=true species's stored value is left unchanged at a resize (its
    amount = stored × V_c is preserved with no rescale); the hOSU=false
    neighbour in the same compartment drops to 1/3."""
    model = bngsim.Model.from_sbml_string(SBML_HOSU_RESIZE)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 1.4), n_points=141)
    t = np.asarray(r.time)
    names = list(r.species_names)
    H = np.asarray(r.species)[:, names.index("H")]
    S = np.asarray(r.species)[:, names.index("S")]

    i_pre = np.where(t <= TE)[0][-1]
    i_post = np.where(t > TE)[0][0]
    # hOSU species: stored value unchanged across the resize.
    assert H[i_post] == pytest.approx(H[i_pre], rel=1e-9)
    # hOSU=false neighbour: rescaled by 1/3 (amount preserved).
    assert S[i_post] == pytest.approx(S[i_pre] / 3.0, rel=1e-6)


def test_functional_resize_rejected_under_ssa():
    """An event-resized (variable-volume) compartment is now supported under SSA
    for mass-action reactions (GH #81 Tier 1): the count-preserving resize plus
    the live-volume propensity correction. But the saturable Functional law here
    is NOT a mass-action monomial, so the scalar correction does not apply and
    the reaction is refused with ``varvol_non_mass_action`` (the per-reaction
    successor to the old blanket ``compartment_event_resize`` gate). The
    mass-action variant (next test) runs. End-to-end behaviour is covered by
    test_ssa_variable_volume.py."""
    model = bngsim.Model.from_sbml_string(SBML_FUNCTIONAL_RESIZE)
    codes = {i.code for i in bngsim.validate_for_ssa(model)}
    assert "varvol_non_mass_action" in codes
    assert "compartment_event_resize" not in codes  # old blanket gate retired

    with pytest.raises(bngsim.SsaValidationError):
        bngsim.Simulator(model, method="ssa").run(t_span=(0, 2), n_points=5, seed=1)


def test_massaction_resize_runs_under_ssa():
    """The mass-action resize model (S→∅, law C·k·S) now runs under SSA: a
    unimolecular decay's propensity k·n_S is volume-independent, so the resize
    leaves the counts untouched (no rejection, no spurious rescale)."""
    model = bngsim.Model.from_sbml_string(SBML_MASSACTION_RESIZE)
    assert not [i for i in bngsim.validate_for_ssa(model) if i.severity == "error"]
    res = bngsim.Simulator(model, method="ssa").run(t_span=(0, 2), n_points=5, seed=1)
    assert np.asarray(res.species).shape[0] == 5


def test_functional_resize_matches_roadrunner():
    """Optional cross-engine lock against libRoadRunner (the literal-SBML
    reference) for the Functional resize model."""
    roadrunner = pytest.importorskip("roadrunner")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(SBML_FUNCTIONAL_RESIZE)
        path = fh.name
    try:
        rr = roadrunner.RoadRunner(path)
        rr.integrator.setValue("relative_tolerance", 1e-10)
        rr.integrator.setValue("absolute_tolerance", 1e-12)
        rr.timeCourseSelections = ["time", "[S]"]
        ref = np.asarray(rr.simulate(0, 2, 9))

        model = bngsim.Model.from_sbml_string(SBML_FUNCTIONAL_RESIZE)
        r = bngsim.Simulator(model, method="ode").run(
            t_span=(0, 2), n_points=9, rtol=1e-10, atol=1e-12
        )
        S = np.asarray(r.species)[:, list(r.species_names).index("S")]
        np.testing.assert_allclose(S, ref[:, 1], rtol=1e-6, atol=1e-7)
    finally:
        os.unlink(path)


# ── Model 4: DELAYED, useValuesFromTriggerTime=true resize (GH #248) ─────────
#
# The kitchen-sink suite case 01000 fails only in its two reacting species. The
# root cause, reduced here: a species whose *amount changes between an event's
# trigger time and its (delayed) execution time* is rescaled at the wrong
# volume when the event both (a) uses trigger-time values and (b) resizes the
# species' compartment.
#
# S → ∅ with law  C·k·S  (mass action: d(amount)/dt = -k·amount, C cancels),
# k = 1, [S](0) = 100, C = p via an assignment rule. An event triggers at
# t = 0.5, is DELAYED by 1.0 (so it executes at t = TE4 = 1.5), sets p := 4
# (so C jumps 1 → 4), and — crucially — has useValuesFromTriggerTime=true.
#
# The amount is C-independent, so amount(t) = 100·e^(-t) is CONTINUOUS across
# the resize (the ÷4 concentration rescale and the ×4 volume cancel). The
# injected #74 amount-conservation rescale must read the species' *pre-fire*
# (execution-time, t=1.5) concentration/volume. Before GH #248 the rescale was
# frozen at TRIGGER time (t=0.5, where amount ≈ 60.65) along with the user
# assignment, so it conserved the stale trigger-time amount — leaving the
# post-event amount ≈ 60.65 instead of ≈ 22.31, a ~2.7× jump.
SBML_DELAYED_UVFTT_RESIZE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="delayed_uvftt_resize">
    <listOfCompartments>
      <compartment id="C" size="1" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p" value="1" constant="false"/>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="C">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>p</ci></math>
      </assignmentRule>
    </listOfRules>
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
      <event id="grow" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><geq/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>
                  <cn>0.5</cn></apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="p">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>4</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

TE4 = 1.5  # delayed execution time: trigger 0.5 + delay 1.0


def test_delayed_uvftt_resize_conserves_amount_gh248():
    """A delayed, useValuesFromTriggerTime=true event that resizes a species'
    compartment must conserve the species' amount at *execution* time, not its
    stale trigger-time amount (GH #248 — reduced from suite case 01000)."""
    model = bngsim.Model.from_sbml_string(SBML_DELAYED_UVFTT_RESIZE)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 3), n_points=301, rtol=1e-10, atol=1e-12
    )
    t = np.asarray(r.time)
    S = np.asarray(r.species)[:, list(r.species_names).index("S")]
    C = np.asarray(r.expressions["C"])  # assignment-rule compartment
    amount = S * C

    # Amount is C-independent (mass action) ⇒ smooth 100·e^(-t) EVERYWHERE, with
    # no jump at the resize.
    np.testing.assert_allclose(amount, 100.0 * np.exp(-t), rtol=1e-5, atol=1e-6)

    # The concentration, by contrast, jumps by V_old/V_new = 1/4 at execution.
    i_pre = np.where(t < TE4)[0][-1]
    i_post = np.where(t >= TE4)[0][0]
    assert C[i_pre] == pytest.approx(1.0)
    assert C[i_post] == pytest.approx(4.0)
    assert S[i_post] / S[i_pre] == pytest.approx(0.25, rel=2e-2)

    # Regression guard: the pre-fix engine froze the injected amount-conservation
    # rescale at TRIGGER time (t=0.5, amount ≈ 60.65) instead of at execution
    # (t=1.5, amount ≈ 22.31), so the post-event amount was ~2.7× too large.
    assert amount[i_post] < 30.0  # correct ≈ 22.3; buggy ≈ 60.6


def test_delayed_uvftt_resize_matches_non_uvftt():
    """Cross-check: the injected amount-conservation rescale is a physical
    consequence of the resize at execution time, so flipping the event to
    useValuesFromTriggerTime=false (whose apply path was always correct) yields
    an identical trajectory (GH #248)."""
    uvftt_false = SBML_DELAYED_UVFTT_RESIZE.replace(
        'useValuesFromTriggerTime="true"', 'useValuesFromTriggerTime="false"'
    )
    assert uvftt_false != SBML_DELAYED_UVFTT_RESIZE

    def run(src):
        model = bngsim.Model.from_sbml_string(src)
        r = bngsim.Simulator(model, method="ode").run(
            t_span=(0, 3), n_points=301, rtol=1e-10, atol=1e-12
        )
        S = np.asarray(r.species)[:, list(r.species_names).index("S")]
        C = np.asarray(r.expressions["C"])
        return S * C

    np.testing.assert_allclose(
        run(SBML_DELAYED_UVFTT_RESIZE), run(uvftt_false), rtol=1e-9, atol=1e-9
    )
