"""Variable-volume compartments driven by an ASSIGNMENT RULE (GH #87).

A compartment whose size is set by an assignment rule — e.g. the budding-yeast
``tV := mV + dV`` of BIOMD0000000856 (Heldt2018 cell-cycle oscillator) — is
*variable-volume*, just like a rate-rule or event-resized compartment, but bngsim
recognised neither its dynamics nor its reporting as variable. Two coupled bugs
resulted, both fixed here and locked by this suite:

  1. **Dynamics.** For an amount-valued (``hasOnlySubstanceUnits=true``) species
     stored as ``amount / V_static``, the Functional storage-conversion divide
     used the *live* compartment symbol ``V(t)`` instead of the load-time numeric
     ``V_static``. Dividing the amount-rate by ``V_live(t)`` throttled every
     reaction by ``V_static / V_live(t)`` as the compartment grew — for #856 the
     SBF→CLN cascade never ignited, ``CLN/tV`` never reached ``StartThr``, no
     cell-cycle event ever fired, and the published limit cycle collapsed to a
     flat monotone line (RoadRunner and COPASI both oscillate). The same
     live-vs-static error corrupted event-assignment targets in the AR
     compartment.

  2. **Reporting.** The integrated *amount* is correct, but the reported
     *concentration* must be ``amount / V_live(t)``; bngsim reported
     ``amount / V_static``, stale by ``V_static / V_live(t)``. The new report pass
     rescales it, reading ``V_live(t)`` from the AR compartment's own
     assignment-rule *expression* column (an AR compartment has no ODE state, so
     — unlike #85's rate-rule map — the live volume is not a promoted-species
     column).

Closed-form oracles in a compartment ``tC := mC`` whose member ``mC`` grows
linearly (``dmC/dt = g``, ``mC(0) = V0`` ⇒ ``V_live(t) = V0 + g·t``):

  * **Reporting** — a boundary, amount-conserved species ``S`` (constant amount
    ``A``, no reactions) has the dependency-free concentration
    ``[S](t) = A / (V0 + g·t)``.
  * **Dynamics** — a species ``P`` produced by ``∅ → P`` with the
    concentration-form law ``tC · k`` has amount-rate ``k·V_live`` ⇒
    ``amount_P(t) = k·(V0·t + ½·g·t²)`` and concentration
    ``[P](t) = amount_P / (V0 + g·t)``.
"""

import bngsim
import numpy as np
import pytest

A = 10.0  # conserved amount of the boundary species S
V0 = 2.0  # mC size at load (so tC(0) = V0, a NON-unit volume)
G = 0.5  # mC growth rate (dmC/dt = g)
K = 0.3  # concentration-form synthesis rate constant for P


# ── tC := mC, with mC growing linearly. tC is an ASSIGNMENT-RULE compartment. ─
# The synthesis law is wrapped in a functionDefinition so it is classified
# Functional (general kinetic law) — the same path BIOMD0000000856's
# ``tV · f(…)`` laws take, where the #87 dynamics fix lives. (A bare ``tC · k``
# would classify Elementary and bake the static volume into a separate scalar.)
SBML_AR_COMP = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_comp">
    <listOfFunctionDefinitions>
      <functionDefinition id="synth">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <bvar><ci>v</ci></bvar><bvar><ci>kk</ci></bvar>
            <apply><times/><ci>v</ci><ci>kk</ci></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="mC" size="{V0}" constant="false"/>
      <compartment id="tC" size="{V0}" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="tC" initialAmount="{A}"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
      <species id="P" compartment="tC" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="{G}" constant="true"/>
      <parameter id="k" value="{K}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="mC">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math>
      </rateRule>
      <assignmentRule variable="tC">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>mC</ci></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="synP" reversible="false">
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><ci>synth</ci><ci>tC</ci><ci>k</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# ── Static control: tC := mC but mC is now a CONSTANT compartment ────────────
# V_live(t) == V0 for all t, so every reported value must be the static answer
# and the AR-compartment rescale must be a no-op (factor == 1).
SBML_AR_COMP_STATIC = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_comp_static">
    <listOfFunctionDefinitions>
      <functionDefinition id="synth">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <bvar><ci>v</ci></bvar><bvar><ci>kk</ci></bvar>
            <apply><times/><ci>v</ci><ci>kk</ci></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="mC" size="{V0}" constant="true"/>
      <compartment id="tC" size="{V0}" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="tC" initialAmount="{A}"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
      <species id="P" compartment="tC" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="{K}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="tC">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>mC</ci></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="synP" reversible="false">
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><ci>synth</ci><ci>tC</ci><ci>k</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# ── #98: bare mass-action law `tC · k` in the assignment-rule compartment ────
# No functionDefinition wrapper, so the law would classify *Elementary*
# (mass-action) and fold `comp_volumes[tC]` into a scalar rate — baking the
# static volume. The #98 guard refuses mass-action classification when a
# compartment factor is an assignment-rule (variable-volume) compartment, routing
# the reaction to the Functional path where the #87 live-aware divide applies.
# Same closed form as the Functional case: amount_P = k·(V0·t + ½·g·t²).
SBML_AR_COMP_MASSACTION = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_comp_massaction">
    <listOfCompartments>
      <compartment id="mC" size="{V0}" constant="false"/>
      <compartment id="tC" size="{V0}" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="P" compartment="tC" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="{G}" constant="true"/>
      <parameter id="k" value="{K}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="mC">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math>
      </rateRule>
      <assignmentRule variable="tC">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>mC</ci></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="synP" reversible="false">
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>tC</ci><ci>k</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _run(sbml):
    model = bngsim.Model.from_sbml_string(sbml)
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 10.0), n_points=51, rtol=1e-10, atol=1e-12
    )
    t = np.asarray(result.time)
    names = list(result.species_names)
    cols = {n: np.asarray(result.species)[:, names.index(n)] for n in names}
    return model, t, cols


def test_ar_compartment_reporting_conc_is_amount_over_live_volume():
    """Boundary conserved-amount species: [S](t) = A / (V0 + g·t)."""
    _, t, cols = _run(SBML_AR_COMP)
    expected = A / (V0 + G * t)
    np.testing.assert_allclose(cols["S"], expected, rtol=1e-6, atol=1e-9)
    # The bug reported the stale static value A/V0 (constant) — guard the drift
    # is real, not a coincidence at t=0.
    assert abs(cols["S"][-1] - A / V0) > 0.5 * (A / V0)


def test_ar_compartment_dynamics_amount_matches_closed_form():
    """Synthesised species: amount_P = k·(V0·t + ½·g·t²); the live-symbol divide
    bug would throttle production by V_static/V_live(t)."""
    _, t, cols = _run(SBML_AR_COMP)
    v_live = V0 + G * t
    amount_p = K * (V0 * t + 0.5 * G * t**2)
    expected_conc = amount_p / v_live
    np.testing.assert_allclose(cols["P"], expected_conc, rtol=1e-5, atol=1e-9)
    # Recover the amount the way a consumer would (conc · V_live) and check the
    # closed-form amount directly — this is what the dynamics fix restores.
    np.testing.assert_allclose(cols["P"] * v_live, amount_p, rtol=1e-5, atol=1e-9)


def test_ar_compartment_map_is_populated_and_scoped():
    """The AR-compartment rescale map carries the amount-valued tC species and
    points each at the compartment's assignment-rule expression column."""
    model, _, _ = _run(SBML_AR_COMP)
    amap = model._varvol_ar_conc_map
    assert set(amap) == {"S", "P"}
    for _sp, (comp_expr, v_static) in amap.items():
        assert comp_expr == "tC"
        assert v_static == pytest.approx(V0)


def test_static_ar_compartment_is_byte_identical_noop():
    """When the AR compartment's members are constant, V_live == V_static and
    the rescale is a no-op: [S] == A/V0 and [P] == k·t·V0/V0 / V0 = k·t."""
    _, t, cols = _run(SBML_AR_COMP_STATIC)
    np.testing.assert_allclose(cols["S"], A / V0, rtol=1e-9, atol=1e-12)
    # amount_P = k·V0·t (constant volume) ⇒ conc = k·t.
    np.testing.assert_allclose(cols["P"], K * t, rtol=1e-6, atol=1e-9)


def test_massaction_law_in_ar_compartment_uses_live_volume():
    """#98: a bare mass-action law `tC · k` in an assignment-rule compartment
    would classify Elementary and bake the static volume into a scalar rate. The
    guard reroutes it to the Functional path, so the same closed-form amount
    (`k·(V0·t + ½·g·t²)`) and concentration are recovered as the functionDefinition
    (Functional) variant — not the throttled `V_static·k·t` the Elementary bake
    would give."""
    _, t, cols = _run(SBML_AR_COMP_MASSACTION)
    v_live = V0 + G * t
    amount_p = K * (V0 * t + 0.5 * G * t**2)
    np.testing.assert_allclose(cols["P"], amount_p / v_live, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(cols["P"] * v_live, amount_p, rtol=1e-5, atol=1e-9)
    # Guard against the Elementary-bake regression: that path integrates
    # amount_P = V_static·k·t (no ½·g·t² term), i.e. conc → V_static·k·t/V_live.
    elem_bake = (V0 * K * t) / v_live
    assert abs(cols["P"][-1] - elem_bake[-1]) > 0.1 * elem_bake[-1]
