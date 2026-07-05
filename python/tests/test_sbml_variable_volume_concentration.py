"""Concentration reporting for species in rate-rule-driven (variable-volume)
compartments (GH #85).

bngsim stores every species as ``amount / V_static`` using the compartment size
at load (``volume_factor``). That equals the true concentration only while the
compartment is static. When a compartment is driven by a **rate rule** its live
size ``V(t)`` diverges continuously from ``V_static``, so the reported
concentration is stale by exactly ``V_static / V_live(t)`` — the integrated
amounts are correct (for an amount-valued species the engine reads it as
``stored × V_static``), only the reported concentration is wrong.

Surfaced by MODEL1606100000 (Talemi2016 yeast osmo-homeostasis), whose
compartment ``Vos`` runs away 29.5 → 6.4e6: every ``Vos``-compartment species
was reported ~2e5× too large. The fix rescales the reported concentration to
``amount / V_live(t)`` at report time, reading ``V_live(t)`` from the
compartment's own promoted-species column.

The closed-form oracle here is a boundary (amount-conserved) species in a
linearly growing compartment: ``V(t) = V0 + g·t`` and a constant amount ``A``
give a dependency-free analytic concentration ``[S](t) = A / (V0 + g·t)``.

Two regressions are locked alongside the reporting fix:

  * **Compartment-size initialisation.** A rate-rule compartment with no
    ``initialAssignment`` is promoted to a species; before #85 it seeded to 0
    (the promotion path looked up ``getParameter`` only — ``None`` for a
    compartment), making the live volume ``g·t`` instead of ``V0 + g·t`` and
    blowing up the live-volume divide. Its column must start at the declared
    ``size``.

  * **Scope.** The rescale applies ONLY to amount-valued (hOSU=true) species in
    rate-rule compartments. Event-resized compartments already carry an injected
    per-species rescale (#74), and hOSU=false species are integrated in
    concentration space with a separate (unfixed) dilution-term gap — neither
    must be rescaled here.
"""

import bngsim
import numpy as np
import pytest

A = 10.0  # conserved amount of the boundary species
V0 = 2.0  # compartment size at load
G = 0.5  # compartment growth rate (dV/dt)

# ── Closed-form model: amount-conserved species in a growing compartment ────
#
# Compartment C grows linearly (rate rule dC/dt = g, C(0) = V0). An inert
# boundary species S (no reactions) keeps a constant amount A, so its
# concentration is A / (V0 + g·t). A second species T sits in a STATIC
# compartment D and must be reported unchanged (A_T / V_D, constant). Both
# species are hasOnlySubstanceUnits=true (amount-valued) — the case #85 fixes.
SBML_GROW = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="grow">
    <listOfCompartments>
      <compartment id="C" size="{V0}" constant="false"/>
      <compartment id="D" size="4" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="{A}"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
      <species id="T" compartment="D" initialAmount="8"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="{G}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="C">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""

# ── Static control: same species, but C is a constant compartment ───────────
SBML_STATIC = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="static">
    <listOfCompartments>
      <compartment id="C" size="{V0}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="{A}"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
  </model>
</sbml>"""

# ── hOSU=false neighbour: a concentration-valued species in C ───────────────
#
# Used only to lock the scope decision — a hOSU=false species in a rate-rule
# compartment is NOT placed in the rescale map (its variable-volume dynamics
# have a separate dilution-term gap), so the map carries S (hOSU=true) but not U.
SBML_MIXED_HOSU = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="mixed">
    <listOfCompartments>
      <compartment id="C" size="{V0}" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="{A}"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
      <species id="U" compartment="C" initialConcentration="5"
               hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="{G}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="C">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""

# ── Event resize: compartment tripled by an event (NOT a rate rule) ─────────
SBML_EVENT_RESIZE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="event_resize">
    <listOfCompartments>
      <compartment id="C" size="1" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="10"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfEvents>
      <event id="dilute" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.5</cn></apply>
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


def _run(sbml, t_end=8.0, n=9):
    model = bngsim.Model.from_sbml_string(sbml)
    r = (
        model,
        bngsim.Simulator(model, method="ode").run(
            t_span=(0, t_end), n_points=n, rtol=1e-10, atol=1e-12
        ),
    )
    return r


def test_concentration_tracks_live_volume_closed_form():
    """[S](t) = A / (V0 + g·t), not the stale A / V_static."""
    _, r = _run(SBML_GROW)
    t = np.asarray(r.time)
    names = list(r.species_names)
    S = np.asarray(r.species)[:, names.index("S")]

    expected = A / (V0 + G * t)
    np.testing.assert_allclose(S, expected, rtol=1e-7, atol=1e-9)

    # Regression guard: the pre-#85 loader reported the stored A / V_static
    # (constant A/V0) — at the final sample the live volume has grown 3×, so the
    # correct concentration is far below the stale value.
    assert S[-1] < 0.5 * (A / V0)


def test_static_neighbour_compartment_unchanged():
    """A species in a *static* compartment in the same model is reported at its
    constant concentration (amount / V_static) — the rescale must not leak."""
    _, r = _run(SBML_GROW)
    names = list(r.species_names)
    T = np.asarray(r.species)[:, names.index("T")]
    np.testing.assert_allclose(T, 8.0 / 4.0, rtol=1e-12, atol=0)  # constant 2.0


def test_amount_recovered_via_live_volume():
    """The bare-id (amount) selector returns the conserved amount A everywhere —
    recovered as conc × V_live(t), not conc × V_static."""
    _, r = _run(SBML_GROW)
    arr = r.as_roadrunner(selections=["time", "S", "[S]", "C"])
    cols = {c: i for i, c in enumerate(arr.colnames)}
    amount = arr[:, cols["S"]]
    np.testing.assert_allclose(amount, A, rtol=1e-7, atol=1e-9)
    # Consistency: amount == [S] · V_live.
    np.testing.assert_allclose(amount, arr[:, cols["[S]"]] * arr[:, cols["C"]], rtol=1e-9, atol=0)


def test_rate_rule_compartment_initialised_to_size():
    """The promoted compartment column starts at its declared ``size`` and
    follows V0 + g·t — regression for the 0-seed bug (a rate-rule compartment
    with no initialAssignment used to seed to 0, giving V_live = g·t)."""
    _, r = _run(SBML_GROW)
    t = np.asarray(r.time)
    names = list(r.species_names)
    C = np.asarray(r.species)[:, names.index("C")]
    np.testing.assert_allclose(C, V0 + G * t, rtol=1e-9, atol=0)
    assert C[0] == pytest.approx(V0)  # NOT 0


def test_static_compartment_is_noop():
    """A constant-volume model has an empty rescale map and reports the constant
    concentration — the fix is byte-for-byte inert when nothing varies."""
    model, r = _run(SBML_STATIC)
    assert model._varvol_conc_map == {}
    names = list(r.species_names)
    S = np.asarray(r.species)[:, names.index("S")]
    np.testing.assert_allclose(S, A / V0, rtol=1e-12, atol=0)  # constant 5.0


def test_map_scoped_to_amount_valued_species():
    """Only amount-valued (hOSU=true) species enter the rescale map. A
    hOSU=false species in the same rate-rule compartment is excluded (its
    variable-volume dynamics have a separate, unfixed dilution-term gap)."""
    model = bngsim.Model.from_sbml_string(SBML_MIXED_HOSU)
    assert model._varvol_conc_map == {"S": "C"}  # S in, U (hOSU=false) out


def test_event_resize_not_in_rescale_map():
    """An event-resized compartment is NOT a rate-rule compartment: its species
    already carry an injected per-event rescale (#74) and must not be rescaled
    again here. The map stays empty (regression for the BIOMD338/339 narrowing)."""
    model = bngsim.Model.from_sbml_string(SBML_EVENT_RESIZE)
    assert model._varvol_conc_map == {}


def test_concentration_matches_roadrunner():
    """Cross-engine lock against libRoadRunner (the literal-SBML reference): RR
    reports the growing-compartment species as amount / V_live too."""
    roadrunner = pytest.importorskip("roadrunner")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(SBML_GROW)
        path = fh.name
    try:
        rr = roadrunner.RoadRunner(path)
        rr.integrator.setValue("relative_tolerance", 1e-10)
        rr.integrator.setValue("absolute_tolerance", 1e-12)
        rr.timeCourseSelections = ["time", "[S]"]
        ref = np.asarray(rr.simulate(0, 8, 9))

        _, r = _run(SBML_GROW)
        S = np.asarray(r.species)[:, list(r.species_names).index("S")]
        np.testing.assert_allclose(S, ref[:, 1], rtol=1e-6, atol=1e-8)
    finally:
        os.unlink(path)
