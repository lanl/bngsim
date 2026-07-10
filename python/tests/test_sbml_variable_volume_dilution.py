"""Variable-volume *dilution* dynamics for hOSU=false species (GH #86).

For a concentration-valued (``hasOnlySubstanceUnits=false``) species ``S`` of
amount ``A = [S]·V`` in a compartment whose volume ``V(t)`` is driven by a rate
rule, the concentration ODE is::

    d[S]/dt = d(A/V)/dt = (1/V)·dA/dt − [S]·(V̇/V)
              └─ reaction term (÷V_live, #74) ─┘   └─ dilution term ─┘

bngsim integrates the stored *concentration*, but before #86 it emitted only the
reaction term — the **dilution term** ``−[S]·V̇/V`` (the concentration change
caused by the volume itself moving) was missing, so a species in a growing or
shrinking compartment was integrated as if its compartment were static. Both its
concentration and its implied amount diverged from RoadRunner (the issue's repro
was 247% off at t=10).

This is latent: no BioModels corpus model has a hOSU=false species in a rate-rule
compartment (every such species is amount-valued, handled by #85). So the safety
net here is synthetic — closed-form analytic oracles plus libRoadRunner
cross-checks — not the corpus sweep.

Oracles used:
  * **Pure dilution** (no reaction flux): amount ``A`` is conserved, so
    ``[S](t) = A / V(t)``. For a linearly growing compartment ``V = V0 + g·t``
    this is ``[S]0·V0 / (V0 + g·t)`` — exact and dependency-free.
  * **Reaction + dilution** (the issue's Michaelis–Menten repro): integrated by
    SciPy with the dilution term added, and cross-checked against RoadRunner.

Both floating and ``boundaryCondition=true`` species are covered: RoadRunner
dilutes both (amount conserved, concentration = ``amount/V_live``). A boundary
species has no reaction flux, so its full derivative is the dilution term alone —
bngsim integrates it by un-fixing the species in a rate-rule compartment.

GH #234 extends this to two more variable-volume classes:
  * **Assignment-rule compartments** whose volume is time-varying through a
    rate-ruled dependency (``C := p1·p2`` with ``p2' = r``, the SBML semantic
    suite's 00310-00318). ``V̇`` is the chain-rule time derivative of the AR RHS,
    derived symbolically (section 8c). The bare-id amount selector reads ``V_live``
    from the compartment's AR expression column (``_varvol_ar_amount_map``).
  * **``constant=true`` species** in a variable-volume compartment (suite 01117/
    01118): the amount is immutable by definition, but the concentration still
    dilutes. Un-fixed only when the species carries no reaction flux, so un-fixing
    never admits flux RoadRunner does not apply.
"""

import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest
from scipy.integrate import solve_ivp

V0 = 2.0  # compartment size at load
S0 = 5.0  # initial concentration of the diluted species


def _grow_sbml(g, boundary="false", const="false", hosu="false"):
    """A single species in a compartment growing linearly at rate ``g``."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="grow">
    <listOfCompartments>
      <compartment id="C" size="{V0}" constant="false"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="{S0}"
               hasOnlySubstanceUnits="{hosu}" boundaryCondition="{boundary}" constant="{const}"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="g" value="{g}" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="C">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


# Michaelis–Menten degradation in a growing compartment — the issue's exact repro.
MM_REPRO = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="rxn">
  <listOfCompartments><compartment id="C" size="2" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialConcentration="5"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="g" value="0.3" constant="true"/>
   <parameter id="kd" value="0.7" constant="true"/>
   <parameter id="Km" value="3" constant="true"/>
  </listOfParameters>
  <listOfRules>
   <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math></rateRule>
  </listOfRules>
  <listOfReactions><reaction id="deg" reversible="false">
   <listOfReactants>
    <speciesReference species="S" stoichiometry="1" constant="true"/>
   </listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>C</ci><apply><divide/>
     <apply><times/><ci>kd</ci><ci>S</ci></apply>
     <apply><plus/><ci>Km</ci><ci>S</ci></apply></apply></apply>
   </math></kineticLaw>
  </reaction></listOfReactions>
 </model>
</sbml>"""

# State-dependent rate rule: dC/dt = k·C ⇒ V(t) = V0·exp(k·t). With no reaction,
# the conserved amount A = S0·V0 gives [S](t) = S0 / exp(k·t). Exercises the
# inlined rate-rule RHS referencing the live compartment symbol.
EXP_GROW = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="exp">
  <listOfCompartments><compartment id="C" size="2" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialConcentration="5"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="k" value="0.2" constant="true"/></listOfParameters>
  <listOfRules>
   <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>k</ci><ci>C</ci></apply></math></rateRule>
  </listOfRules>
 </model>
</sbml>"""

# A growing compartment holding one amount-valued (St, #85 path) and one
# concentration-valued (Sf, #86 path) species — used to lock map partitioning.
MIX_HOSU = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="mix">
  <listOfCompartments><compartment id="C" size="2" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="St" compartment="C" initialAmount="10"
            hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
   <species id="Sf" compartment="C" initialConcentration="5"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="g" value="0.5" constant="true"/></listOfParameters>
  <listOfRules><rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math></rateRule></listOfRules>
 </model>
</sbml>"""


def _run(sbml, t_end=8.0, n=9, rtol=1e-11, atol=1e-13):
    model = bngsim.Model.from_sbml_string(sbml)
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0, t_end), n_points=n, rtol=rtol, atol=atol
    )
    return model, res


def _conc(res, sid="S"):
    return np.asarray(res.species)[:, list(res.species_names).index(sid)]


def _rr_sim(sbml, selections, t_end, n, rtol=1e-11, atol=1e-13):
    roadrunner = pytest.importorskip("roadrunner")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(sbml)
        path = fh.name
    try:
        rr = roadrunner.RoadRunner(path)
        rr.integrator.setValue("relative_tolerance", rtol)
        rr.integrator.setValue("absolute_tolerance", atol)
        rr.timeCourseSelections = selections
        return np.asarray(rr.simulate(0, t_end, n))
    finally:
        os.unlink(path)


# ── Closed-form pure dilution: [S](t) = S0·V0 / (V0 + g·t) ───────────────────


@pytest.mark.parametrize("boundary", ["false", "true"])
def test_pure_dilution_closed_form(boundary):
    """A hOSU=false species with no reaction flux dilutes as ``S0·V0/(V0+g·t)`` —
    both floating and boundary (the un-fix path)."""
    g = 0.5
    _, res = _run(_grow_sbml(g, boundary=boundary))
    t = np.asarray(res.time)
    S = _conc(res)
    expected = S0 * V0 / (V0 + g * t)
    np.testing.assert_allclose(S, expected, rtol=1e-8, atol=1e-10)
    # Regression guard: the pre-#86 loader left the stored concentration flat at
    # S0 (no dilution). By the final sample V has doubled, so the true value is
    # far below S0.
    assert S[-1] < 0.6 * S0


def test_shrinking_compartment_concentrates():
    """A shrinking compartment (V̇ < 0) concentrates the species: with V0=5,
    g=-0.1 the closed form S0·V0/(V0+g·t) grows from 3 toward 5 over t∈[0,20]."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="shrink">
  <listOfCompartments><compartment id="C" size="5" constant="false"/></listOfCompartments>
  <listOfSpecies><species id="S" compartment="C" initialConcentration="3"
    hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
  <listOfParameters><parameter id="r" value="-0.1" constant="true"/></listOfParameters>
  <listOfRules><rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>r</ci></math></rateRule></listOfRules>
 </model></sbml>"""
    _, res = _run(sbml, t_end=20.0, n=5)
    t = np.asarray(res.time)
    S = _conc(res)
    expected = 3.0 * 5.0 / (5.0 - 0.1 * t)
    np.testing.assert_allclose(S, expected, rtol=1e-8, atol=1e-10)
    assert S[-1] > S[0]  # concentrated, not diluted


def test_state_dependent_rate_rule_closed_form():
    """dC/dt = k·C ⇒ [S](t) = S0 / exp(k·t). Locks the inlined rate-rule RHS
    (which references the live compartment symbol C) inside the dilution term."""
    _, res = _run(EXP_GROW, t_end=5.0, n=6)
    t = np.asarray(res.time)
    S = _conc(res)
    expected = 5.0 / np.exp(0.2 * t)
    np.testing.assert_allclose(S, expected, rtol=1e-8, atol=1e-10)


# ── Reaction + dilution: the issue's Michaelis–Menten repro ──────────────────


def test_mm_reaction_plus_dilution_matches_scipy_oracle():
    """The issue's exact repro. bngsim must match a SciPy integration of the
    *full* ODE d[S]/dt = −kd·[S]/(Km+[S]) − [S]·g/C (reaction + dilution).

    Pre-#86 this species was 247% high at t=10 (reaction term only); the oracle
    is independent of RoadRunner so this test always runs.
    """
    kd, Km, g, v0 = 0.7, 3.0, 0.3, 2.0
    t_end = 10.0

    def rhs(t, y):
        s = y[0]
        C = v0 + g * t
        return [-kd * s / (Km + s) - s * g / C]

    _, res = _run(MM_REPRO, t_end=t_end, n=11)
    t = np.asarray(res.time)
    S = _conc(res)
    sol = solve_ivp(rhs, (0, t_end), [5.0], t_eval=t, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(S, sol.y[0], rtol=1e-7, atol=1e-9)


# ── Amount reporting: bare-id selector uses V_live, not V_static ─────────────


def test_amount_conserved_via_live_volume():
    """The bare-id (amount) selector returns ``conc·V_live(t)``. For a pure-
    dilution species the amount is conserved (``S0·V0``), recovered from the live
    compartment column — not the stale ``conc·V_static``."""
    g = 0.5
    _, res = _run(_grow_sbml(g))
    arr = res.as_roadrunner(selections=["time", "S", "[S]", "C"])
    cols = {c: i for i, c in enumerate(arr.colnames)}
    amount = arr[:, cols["S"]]
    np.testing.assert_allclose(amount, S0 * V0, rtol=1e-8, atol=1e-10)
    # amount == [S] · V_live, not [S] · V_static.
    np.testing.assert_allclose(amount, arr[:, cols["[S]"]] * arr[:, cols["C"]], rtol=1e-9, atol=0)


# ── Map partitioning and scope ───────────────────────────────────────────────


def test_maps_partition_by_hosu():
    """In a mixed compartment, a hOSU=true species takes the #85 concentration-
    rescale map and a hOSU=false species takes the #86 amount-only map — they
    never overlap."""
    model = bngsim.Model.from_sbml_string(MIX_HOSU)
    assert model._varvol_conc_map == {"St": "C"}
    assert model._varvol_amount_map == {"Sf": "C"}


def test_hosu_true_coexisting_species_unaffected():
    """A hOSU=true species in the same growing compartment stays #85-correct
    (reported amount/V_live) — the dilution path must not perturb it."""
    _, res = _run(MIX_HOSU)
    t = np.asarray(res.time)
    St = _conc(res, "St")  # 10 / (2 + 0.5t)
    np.testing.assert_allclose(St, 10.0 / (V0 + 0.5 * t), rtol=1e-8, atol=1e-10)


def test_static_compartment_no_dilution():
    """A static (constant) compartment emits no dilution term and an empty amount
    map — the species holds its concentration."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="static">
  <listOfCompartments><compartment id="C" size="2" constant="true"/></listOfCompartments>
  <listOfSpecies><species id="S" compartment="C" initialConcentration="5"
    hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
 </model></sbml>"""
    model, res = _run(sbml)
    assert model._varvol_amount_map == {}
    S = _conc(res)
    np.testing.assert_allclose(S, 5.0, rtol=1e-12, atol=0)  # flat — no dilution


def test_event_resize_not_in_amount_map():
    """An event-resized compartment is NOT a rate-rule compartment; its hOSU=false
    species are handled by the #74 per-event rescale, not a continuous dilution
    term. The amount map stays empty (no double-correction)."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="ev">
  <listOfCompartments><compartment id="C" size="1" constant="false"/></listOfCompartments>
  <listOfSpecies><species id="S" compartment="C" initialConcentration="5"
    hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
  <listOfEvents><event id="dilute" useValuesFromTriggerTime="false">
   <trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><gt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>0.5</cn></apply></math></trigger>
   <listOfEventAssignments><eventAssignment variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>C</ci><cn>3</cn></apply></math></eventAssignment></listOfEventAssignments>
  </event></listOfEvents>
 </model></sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    assert model._varvol_amount_map == {}


def test_suite_adapter_uses_semantic_concentration_for_hosu_event_resize():
    """The SBML-suite adapter must read ``[S]`` through as_roadrunner().

    For hOSU=true in an event-resized compartment the raw BNGsim column is
    amount/V_static, not amount/V_live. The suite grader multiplies species
    concentration by the live compartment for amount outputs, so using the raw
    column double-counts the resize (01222).
    """
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="adapter_hosu_resize">
  <listOfCompartments><compartment id="c" size="2" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="H" compartment="c" initialAmount="10"
            hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
  </listOfSpecies>
  <listOfEvents><event id="resize" useValuesFromTriggerTime="false">
   <trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><gt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>0.5</cn></apply>
   </math></trigger>
   <listOfEventAssignments>
    <eventAssignment variable="c"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>4</cn></math></eventAssignment>
   </listOfEventAssignments>
  </event></listOfEvents>
 </model></sbml>"""
    model, res = _run(sbml, t_end=1.0, n=3)
    raw = np.asarray(res.species)[:, list(res.species_names).index("H")]
    np.testing.assert_allclose(raw[-1], 5.0, rtol=1e-12, atol=0.0)

    bench_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "suites" / "sbml_test_suite"
    added = str(bench_dir) not in sys.path
    if added:
        sys.path.insert(0, str(bench_dir))
    try:
        from _engines import _BngsimSeries
    finally:
        if added:
            sys.path.remove(str(bench_dir))

    series = _BngsimSeries(model, res, n_points=3)
    conc = series.species_conc("H")
    np.testing.assert_allclose(conc[-1], 2.5, rtol=1e-12, atol=0.0)


def test_assignment_rule_species_excluded():
    """A hOSU=false species in a rate-rule compartment that is itself an
    AssignmentRule target is excluded — its own rule defines its value, so no
    dilution term is added."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="ar">
  <listOfCompartments><compartment id="C" size="2" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialConcentration="5"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
   <species id="A" compartment="C" initialConcentration="1"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="g" value="0.5" constant="true"/></listOfParameters>
  <listOfRules>
   <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>g</ci></math></rateRule>
   <assignmentRule variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>7</cn></math></assignmentRule>
  </listOfRules>
 </model></sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    # S (reaction-free, diluted) is in the map; A (AssignmentRule target) is not.
    assert model._varvol_amount_map == {"S": "C"}


def test_analytical_jacobian_complete_for_dilution():
    """The new dilution Functional law is symbolically differentiable: the
    analytical Jacobian stays complete (does not fall back to finite difference).

    GH #145: the derivation is deferred off the load path, so warm it explicitly
    (the ODE-solve setup would do the same) before asserting completeness."""
    model = bngsim.Model.from_sbml_string(EXP_GROW)
    model.prepare_analytical_jacobian()
    assert model._core.analytical_jacobian_complete is True


# ── GH #234: assignment-rule compartment with time-varying volume ────────────
#
# C := p1·p2 with a rate rule p2' = r ⇒ C(t) = p1·(p2_0 + r·t) is a time-varying
# volume even though C itself carries no rate rule. A hOSU=false species there
# needs the dilution term −[S]·V̇/V with V̇ = p1·r (the chain-rule derivative of
# the AR RHS). These mirror the SBML semantic suite's 00310-00318.

# Pure dilution in an AR compartment: a boundary species, amount conserved.
AR_PURE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="arpure">
  <listOfCompartments><compartment id="C" size="0.15" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialAmount="1.5"
            hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="p1" value="0.1" constant="true"/>
   <parameter id="p2" value="1.5" constant="false"/>
   <parameter id="r" value="0.1" constant="true"/>
  </listOfParameters>
  <listOfRules>
   <assignmentRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>p1</ci><ci>p2</ci></apply></math></assignmentRule>
   <rateRule variable="p2"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>r</ci></math></rateRule>
  </listOfRules>
 </model></sbml>"""

# Reaction + dilution in an AR compartment — the suite's 00310 exactly: a forward
# reaction S1 → S2 with rate k1·S1·C. Since S1 is read as a concentration,
# d(amount_S1)/dt = −k1·[S1]·C = −k1·amount_S1, so amount_S1(t) = A0·exp(−k1·t)
# independent of the (cancelled) volume — the dilution term is what keeps the
# stored concentration consistent with that amount.
AR_RXN = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="arrxn">
  <listOfCompartments><compartment id="C" size="0.15" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S1" compartment="C" initialAmount="1.5"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
   <species id="S2" compartment="C" initialAmount="0"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k1" value="0.9" constant="true"/>
   <parameter id="p1" value="0.1" constant="true"/>
   <parameter id="p2" value="1.5" constant="false"/>
   <parameter id="r" value="0.1" constant="true"/>
  </listOfParameters>
  <listOfRules>
   <assignmentRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>p1</ci><ci>p2</ci></apply></math></assignmentRule>
   <rateRule variable="p2"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>r</ci></math></rateRule>
  </listOfRules>
  <listOfReactions><reaction id="deg" reversible="false">
   <listOfReactants><speciesReference species="S1" stoichiometry="1" constant="true"/></listOfReactants>
   <listOfProducts><speciesReference species="S2" stoichiometry="1" constant="true"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>C</ci><ci>k1</ci><ci>S1</ci></apply></math></kineticLaw>
  </reaction></listOfReactions>
 </model></sbml>"""


AR_EVENT_RXN = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="areventrxn">
  <listOfCompartments><compartment id="C" size="1" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S1" compartment="C" initialAmount="1"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="fakeC" value="1" constant="false"/>
   <parameter id="k1" value="1" constant="true"/>
  </listOfParameters>
  <listOfRules>
   <assignmentRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>fakeC</ci></math></assignmentRule>
  </listOfRules>
  <listOfReactions><reaction id="reaction1" reversible="false">
   <listOfProducts><speciesReference species="S1" stoichiometry="1" constant="true"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><divide/><apply><times/><ci>C</ci><ci>k1</ci></apply><ci>S1</ci></apply>
   </math></kineticLaw>
  </reaction></listOfReactions>
  <listOfEvents><event id="resize_by_dependency" useValuesFromTriggerTime="false">
   <trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><gt/><ci>S1</ci><cn>2.1</cn></apply>
   </math></trigger>
   <listOfEventAssignments>
    <eventAssignment variable="fakeC"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>10</cn></math></eventAssignment>
   </listOfEventAssignments>
  </event></listOfEvents>
 </model></sbml>"""


def _explicit_species_resize_sbml(species_first=True):
    assignments = (
        """
    <eventAssignment variable="S1"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.2</cn></math></eventAssignment>
    <eventAssignment variable="C1"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.2</cn></math></eventAssignment>
"""
        if species_first
        else """
    <eventAssignment variable="C1"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.2</cn></math></eventAssignment>
    <eventAssignment variable="S1"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.2</cn></math></eventAssignment>
"""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="explicit_resize">
  <listOfCompartments><compartment id="C1" size="0.5" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S1" compartment="C1" initialConcentration="2"
            hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="x" value="0" constant="false"/></listOfParameters>
  <listOfRules>
   <assignmentRule variable="x"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>S1</ci></math></assignmentRule>
  </listOfRules>
  <listOfEvents><event id="explicit" useValuesFromTriggerTime="false">
   <trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><geq/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>0.45</cn></apply>
   </math></trigger>
   <listOfEventAssignments>{assignments}   </listOfEventAssignments>
  </event></listOfEvents>
 </model></sbml>"""


def test_ar_compartment_pure_dilution_closed_form():
    """A boundary hOSU=false species in an AR compartment C := p1·p2 (p2' = r)
    dilutes as the amount-conserving closed form ``A0 / C(t)``."""
    _, res = _run(AR_PURE, t_end=10.0, n=11)
    t = np.asarray(res.time)
    S = _conc(res, "S")
    C = 0.1 * (1.5 + 0.1 * t)  # p1·(p2_0 + r·t)
    expected = 1.5 / C  # A0 / C(t), amount conserved at 1.5
    np.testing.assert_allclose(S, expected, rtol=1e-7, atol=1e-9)
    assert S[-1] < S[0]  # diluted as C grows


def test_ar_compartment_reaction_plus_dilution_amount_decays():
    """Suite 00310: the amount of S1 decays as ``A0·exp(−k1·t)`` (volume cancels),
    and the bare-id amount selector recovers it via the live AR volume."""
    _, res = _run(AR_RXN, t_end=10.0, n=11)
    t = np.asarray(res.time)
    arr = res.as_roadrunner(selections=["time", "S1", "[S1]"])
    cols = {c: i for i, c in enumerate(arr.colnames)}
    amount = arr[:, cols["S1"]]
    conc = arr[:, cols["[S1]"]]
    expected_amount = 1.5 * np.exp(-0.9 * t)
    np.testing.assert_allclose(amount, expected_amount, rtol=1e-6, atol=1e-8)
    # amount == [S1] · C_live(t), so the stored concentration is amount/C_live.
    C = 0.1 * (1.5 + 0.1 * t)
    np.testing.assert_allclose(conc, expected_amount / C, rtol=1e-6, atol=1e-8)


def test_ar_compartment_event_dependency_rescales_and_divides_live_volume():
    """Suite 00946/00948 shape: an event changes a parameter used by an
    assignment-rule compartment. The event must rescale hOSU=false species by
    old/new volume, and Functional reaction emission must still divide by the
    live AR compartment even when V_static is 1.
    """
    model, res = _run(AR_EVENT_RXN, t_end=3.0, n=301)
    assert model._varvol_ar_amount_map == {"S1": "C"}

    t_fire = (2.1**2 - 1.0) / 2.0
    expected_conc = np.sqrt((2.1 / 10.0) ** 2 + 2.0 * (3.0 - t_fire))
    expected_amount = expected_conc * 10.0
    arr = res.as_roadrunner(selections=["S1", "[S1]"])
    np.testing.assert_allclose(arr[-1, 0], expected_amount, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(arr[-1, 1], expected_conc, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(res.expressions["C"])[-1], 10.0, rtol=1e-12)
    assert arr[-1, 1] < 2.0  # without law/V_live it overshoots to about 5.1


@pytest.mark.parametrize("species_first", [True, False])
def test_explicit_species_assignment_preserves_pre_resize_amount(species_first):
    """Suite 01779/01780 shape: a single event assigns both hOSU=false species
    concentration and compartment size. The species RHS is evaluated in the old
    compartment; after resize the stored concentration must preserve that
    explicit assignment's implied amount, independent of assignment order.
    """
    _, res = _run(_explicit_species_resize_sbml(species_first), t_end=1.0, n=11)
    c_live = np.asarray(res.observables)[:, list(res.observable_names).index("C1")]
    arr = res.as_roadrunner(selections=["S1", "[S1]"])
    np.testing.assert_allclose(c_live[-1], 0.2, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(arr[-1, 0], 0.1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(arr[-1, 1], 0.5, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(res.expressions["x"])[-1], 0.5, rtol=1e-9)


def test_ar_static_compartment_emits_no_dilution():
    """An AR compartment whose RHS is constant (V̇ ≡ 0) gets no dilution term and
    an empty amount map — byte-identical to the pre-#234 loader."""
    sbml = AR_PURE.replace(
        '<rateRule variable="p2"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>r</ci></math></rateRule>',
        "",
    ).replace(
        'value="0.1" constant="true"/>\n   <parameter id="p2" value="1.5" constant="false"/>',
        'value="0.1" constant="true"/>\n   <parameter id="p2" value="1.5" constant="true"/>',
    )
    model, res = _run(sbml, t_end=10.0, n=11)
    assert model._varvol_ar_amount_map == {}
    S = _conc(res, "S")
    np.testing.assert_allclose(S, 10.0, rtol=1e-10, atol=0)  # flat: 1.5/0.15, no dilution


def test_ar_dilution_analytical_jacobian_complete():
    """The AR dilution Functional law (its V̇ a symbolic chain-rule expression) is
    differentiable, so the analytical Jacobian stays complete."""
    model = bngsim.Model.from_sbml_string(AR_RXN)
    model.prepare_analytical_jacobian()
    assert model._core.analytical_jacobian_complete is True


# ── GH #234: constant=true species dilutes (amount immutable, concentration not) ─

# Suite 01117: a constant=true hOSU=false species in a rate-rule compartment
# (C' = 1 ⇒ V = 1 + t). The amount is fixed at [S]0·V0 = 1; the concentration
# dilutes as 1/(1+t).
CONST_SPECIES = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="constsp">
  <listOfCompartments><compartment id="C" size="1" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialConcentration="1"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="true"/>
  </listOfSpecies>
  <listOfRules>
   <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math></rateRule>
  </listOfRules>
 </model></sbml>"""


def test_constant_species_concentration_dilutes():
    """A constant=true hOSU=false species keeps its amount but its concentration
    dilutes as ``[S]0·V0 / (V0 + t)`` — un-fixed because it carries no flux."""
    _, res = _run(CONST_SPECIES, t_end=5.0, n=6)
    t = np.asarray(res.time)
    S = _conc(res, "S")
    np.testing.assert_allclose(S, 1.0 / (1.0 + t), rtol=1e-8, atol=1e-10)
    assert S[-1] < 0.2  # diluted, not held flat at 1


def test_constant_species_with_reaction_flux_stays_fixed():
    """The #234 reaction-flux guard: a constant=true species that *appears in a
    reaction* is NOT un-fixed for dilution (un-fixing would admit flux RoadRunner
    does not apply), so its concentration is held at its IC rather than diluting.
    (Such a model is degenerate per SBML — a constant non-boundary species cannot
    legally be a reactant — so this exercises the defensive guard, not real SBML.)
    """
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="constflux">
  <listOfCompartments><compartment id="C" size="1" constant="false"/></listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialConcentration="1"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="true"/>
   <species id="P" compartment="C" initialConcentration="0"
            hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="k" value="1" constant="true"/></listOfParameters>
  <listOfRules>
   <rateRule variable="C"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math></rateRule>
  </listOfRules>
  <listOfReactions><reaction id="r" reversible="false">
   <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
   <listOfProducts><speciesReference species="P" stoichiometry="1" constant="true"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>k</ci><ci>C</ci><ci>S</ci></apply></math></kineticLaw>
  </reaction></listOfReactions>
 </model></sbml>"""
    _, res = _run(sbml, t_end=3.0, n=4)
    # S participates in a reaction, so the guard leaves it fixed: its concentration
    # is held at the IC (not diluted to 1/(1+t)).
    S = _conc(res, "S")
    np.testing.assert_allclose(S, 1.0, rtol=1e-12, atol=0)


# ── Cross-engine locks against libRoadRunner ─────────────────────────────────


@pytest.mark.parametrize("boundary", ["false", "true"])
def test_pure_dilution_matches_roadrunner(boundary):
    g = 0.5
    t_end, n = 8.0, 9
    ref = _rr_sim(_grow_sbml(g, boundary=boundary), ["time", "[S]"], t_end, n)
    _, res = _run(_grow_sbml(g, boundary=boundary), t_end=t_end, n=n)
    S = _conc(res)
    np.testing.assert_allclose(S, ref[:, 1], rtol=1e-6, atol=1e-8)


def test_mm_repro_matches_roadrunner():
    t_end, n = 10.0, 11
    ref = _rr_sim(MM_REPRO, ["time", "[S]"], t_end, n)
    _, res = _run(MM_REPRO, t_end=t_end, n=n)
    S = _conc(res)
    np.testing.assert_allclose(S, ref[:, 1], rtol=1e-6, atol=1e-8)


def test_ar_compartment_dilution_matches_roadrunner():
    """The AR-compartment reaction+dilution case (00310) matches RoadRunner's
    concentration trajectory."""
    t_end, n = 10.0, 11
    ref = _rr_sim(AR_RXN, ["time", "[S1]"], t_end, n)
    _, res = _run(AR_RXN, t_end=t_end, n=n)
    S = _conc(res, "S1")
    np.testing.assert_allclose(S, ref[:, 1], rtol=1e-6, atol=1e-8)


def test_constant_species_dilution_matches_roadrunner():
    """The constant=true dilution case (01117) matches RoadRunner."""
    t_end, n = 5.0, 6
    ref = _rr_sim(CONST_SPECIES, ["time", "[S]"], t_end, n)
    _, res = _run(CONST_SPECIES, t_end=t_end, n=n)
    S = _conc(res, "S")
    np.testing.assert_allclose(S, ref[:, 1], rtol=1e-6, atol=1e-8)
