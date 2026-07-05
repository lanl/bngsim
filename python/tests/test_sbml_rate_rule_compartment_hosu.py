"""hOSU=true species in a RATE-RULE (continuously variable-volume) compartment (GH #114).

A compartment whose size is driven by a *rate rule* — ``dV/dt = g``, the
continuously-variable-volume case of MODEL1904020001 (a Light2019-style circadian
cell-division model) — stores an amount-valued (``hasOnlySubstanceUnits=true``)
species as ``amount / V_static``. bngsim's Functional storage-conversion divide used
the *live* compartment symbol ``V(t)`` instead of the load-time numeric ``V_static``,
throttling the amount-rate by ``V_static / V_live(t)`` as the compartment grew and
silently suppressing the dynamics — MODEL1904020001's INTF amount stayed flat at ~1.0
while it should grow to ~1.27 (off 20.6% vs both COPASI and an independent
scipy-literal integrator, which agree to machine precision).

This is the rate-rule sibling of the #87 assignment-rule-compartment fix: #87 wired
the V_static divide for AR compartments only; the ``_vd_varvol`` (``common_vs==1.0``)
and ``_vd_unified`` (``common_vs!=1.0``) §9 branches never covered rate-rule
compartments. See ``test_sbml_assignment_rule_compartment.py`` for the AR case.

Closed-form oracle, compartment ``cell`` with ``dcell/dt = g``, ``cell(0) = V0`` ⇒
``V_live(t) = V0 + g·t``:

  * **Dynamics** — an hOSU=true species ``P`` produced by ``∅ → P`` with the
    concentration-form (functionDefinition-wrapped) law ``cell · k`` has amount-rate
    ``d(amount_P)/dt = cell·k = (V0 + g·t)·k`` ⇒ ``amount_P(t) = k·(V0·t + ½·g·t²)``,
    reported as ``[P](t) = amount_P / V_live``. The live-symbol divide bug instead
    integrates ``d(amount_P)/dt = k·V_static`` (the ``cell·k / cell`` cancellation) ⇒
    the throttled ``amount_P = k·V_static·t``, missing the ``½·g·t²`` growth term.
  * **Reporting** — a boundary, conserved-amount hOSU species ``S`` (constant amount
    ``A``, no reactions) reports ``[S](t) = A / (V0 + g·t)`` (the #85 rescale, which
    the #114 dynamics fix leaves exact).

``V0 == 1.0`` exercises the ``_vd_varvol`` branch (the MODEL1904020001 INTF site, which
had no V_static guard at all); ``V0 == 2.0`` exercises the ``_vd_unified`` branch (whose
guard previously covered only ``ar_comp_targets``).
"""

import bngsim
import numpy as np
import pytest

A = 7.0  # conserved amount of the boundary species S
G = 0.5  # cell growth rate (dcell/dt = g)
K = 0.3  # concentration-form synthesis rate constant for P


# ── cell is a RATE-RULE compartment (dcell/dt = g), member species hOSU=true. ─
# The synthesis law is wrapped in a functionDefinition so it classifies Functional
# (general kinetic law) — the same §9 path MODEL1904020001's ``cell·(…)`` laws take,
# where the storage divide lives. (A bare ``cell · k`` would classify Elementary.)
def _sbml(v0):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="rate_rule_comp">
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
      <compartment id="cell" size="{v0}" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cell" initialAmount="{A}"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="{G}" constant="true"/>
      <parameter id="k" value="{K}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="cell">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="synP" reversible="false">
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><ci>synth</ci><ci>cell</ci><ci>k</ci></apply>
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


@pytest.mark.parametrize("v0", [1.0, 2.0])
def test_rate_rule_compartment_hosu_dynamics_amount_matches_closed_form(v0):
    """hOSU=true P synthesised by ∅→P with law ``cell·k``:
    amount_P(t) = k·(V0·t + ½·g·t²). v0=1.0 exercises the ``_vd_varvol`` branch (the
    MODEL1904020001 INTF site); v0=2.0 the ``_vd_unified`` branch."""
    _, t, cols = _run(_sbml(v0))
    v_live = v0 + G * t
    amount_p = K * (v0 * t + 0.5 * G * t**2)
    np.testing.assert_allclose(cols["P"], amount_p / v_live, rtol=1e-5, atol=1e-9)
    # Recover the amount the way a consumer would (conc · V_live) and check the
    # closed-form amount directly — this is what the dynamics fix restores.
    np.testing.assert_allclose(cols["P"] * v_live, amount_p, rtol=1e-5, atol=1e-9)
    # Guard against the live-symbol throttle: that bug integrates
    # d(amount)/dt = k·V_static (the ``cell·k / cell`` cancellation) ⇒ the linear
    # amount k·V0·t, missing the ½·g·t² growth term. Confirm the divergence is
    # material, not solver noise.
    throttled_amount = K * v0 * t
    assert abs((cols["P"][-1] * v_live[-1]) - throttled_amount[-1]) > 0.1 * throttled_amount[-1]


@pytest.mark.parametrize("v0", [1.0, 2.0])
def test_rate_rule_compartment_hosu_reporting_conc_is_amount_over_live_volume(v0):
    """Boundary conserved-amount hOSU species: [S](t) = A / (V0 + g·t). The #85
    rescale stays exact after the #114 dynamics fix (it assumes the amount invariant
    ``stored·V_static == amount`` that this fix restores)."""
    _, t, cols = _run(_sbml(v0))
    np.testing.assert_allclose(cols["S"], A / (v0 + G * t), rtol=1e-6, atol=1e-9)
    # The bug would report the stale static value A/V0 (constant) — guard the
    # volume drift is real, not a coincidence at t=0.
    assert abs(cols["S"][-1] - A / v0) > 0.1 * (A / v0)


@pytest.mark.parametrize("v0", [1.0, 2.0])
def test_rate_rule_compartment_volume_integrates_linearly(v0):
    """Sanity: the rate-rule compartment itself integrates correctly, cell(t) =
    V0 + g·t. The bug is in the species storage divide, not the volume — both the
    issue and this test rely on the volume being right while the amount is throttled."""
    _, t, cols = _run(_sbml(v0))
    np.testing.assert_allclose(cols["cell"], v0 + G * t, rtol=1e-8, atol=1e-10)
