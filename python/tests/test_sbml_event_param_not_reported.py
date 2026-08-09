"""GH #71 — an SBML parameter/compartment mutated only by an event must NOT
leak into the trajectory output as a species column.

The engine writes event assignments into species slots, so a parameter or
compartment that an event mutates has to be promoted to a species to carry
per-trajectory state. But it is not a floating species, and RoadRunner does not
emit it as a trajectory column. The loader marks such a promotion
``reported=False``; the Result layer keeps it as full integrator state and as a
same-named observable (so other expressions resolve the live value) but projects
it out of ``species`` / ``species_names`` and the ``.cdat`` export.

Surfaced by MODEL1108260014, whose events assign to ``parameter_1`` (IIa*) and
``compartment_1`` — both appeared as spurious bngsim-only trajectory columns
against RoadRunner. (A rate-rule-promoted parameter, by contrast, IS a genuine
ODE variable RoadRunner reports, and stays reported — covered elsewhere.)
"""

from pathlib import Path

import bngsim
import numpy as np
import pytest

# The model GH #202 was filed against. Tracked (214 files under
# benchmarks/sbml_events), unlike the gitignored parity_checks corpus — so this
# runs in CI and in a worktree, not just on the filer's machine.
_REPRO_MODEL = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events" / "BIOMD0000000701.xml"
)

# Floating species S degrading in a constant compartment, plus a parameter P
# that is constant between events and set to 5 by an event at t > 0.5. P never
# participates in a reaction or rule — it is pure event-driven state.
SBML_EVENT_PARAM = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="event_param">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
      <parameter id="P" value="0" constant="false"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="bump" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.5</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="P">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn>5</cn>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def test_event_promoted_parameter_is_not_a_species_column():
    """P is mutated only by an event: full integrator state + observable, but
    not a floating-species trajectory column."""
    model = bngsim.Model.from_sbml_string(SBML_EVENT_PARAM)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 1.0), n_points=21, rtol=1e-10, atol=1e-12
    )
    names = list(r.species_names)

    # Only the floating species S is a trajectory column — not P.
    assert names == ["S"]
    assert "P" not in names
    assert r.species.shape[1] == 1

    # P is still exposed as a same-named observable (so referencing expressions
    # resolve its live value), and the event actually drove it 0 → 5.
    assert "P" in r.observable_names
    P = np.asarray(r.observables["P"])
    t = np.asarray(r.time)
    assert P[t <= 0.5].max() == pytest.approx(0.0)
    assert P[t > 0.5].min() == pytest.approx(5.0)

    # S itself integrates correctly (S0·e^(-k·t)), unaffected by the hidden P.
    np.testing.assert_allclose(np.asarray(r.species)[:, 0], np.exp(-t), rtol=1e-5, atol=1e-7)


def test_event_promoted_parameter_absent_from_cdat_export(tmp_path):
    """The C++ .cdat export projects to reported species too — P must not appear
    in its header or data columns."""
    model = bngsim.Model.from_sbml_string(SBML_EVENT_PARAM)
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 1.0), n_points=6, rtol=1e-10, atol=1e-12
    )
    out = tmp_path / "traj.cdat"
    r.to_cdat(out)
    text = out.read_text()
    header = text.splitlines()[0]
    assert "S" in header
    assert "P" not in header.split()
    # One time column + one species column (S), nothing else.
    assert len(text.splitlines()[1].split()) == 2


# ── GH #202 — the sensitivity tensor projects the same way ────────────────────
#
# A --k1--> B in a unit compartment, plus Q: a parameter that no reaction or
# rule touches and an event sets to `amp` at t > 0.5, i.e. the GH #71 promotion
# above. Two floating species (so a row *swap* is visible, not just a row
# count) with closed-form trajectories and derivatives:
#
#   A(t) = 2·e^(-k1·t)        dA/dk1 = -2·t·e^(-k1·t)     dA/damp = 0
#   B(t) = 2·(1 - e^(-k1·t))  dB/dk1 = +2·t·e^(-k1·t)     dB/damp = 0
#
# dA/dk1 and dB/dk1 are equal and opposite, so keeping the wrong row reads 0
# (Q's row is identically zero) and swapping the two flips the sign.
SBML_EVENT_PARAM_SENS = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="event_param_sens">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="2"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="1" constant="true"/>
      <parameter id="amp" value="5" constant="true"/>
      <parameter id="Q" value="0" constant="false"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="conv" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k1</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="bump" useValuesFromTriggerTime="false">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
                  definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.5</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="Q">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <ci>amp</ci>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""

_SENS_PARAMS = ["k1", "amp"]


@pytest.fixture
def promoted_param_sens_result():
    """Sensitivity run of SBML_EVENT_PARAM_SENS over ``k1`` and ``amp``."""
    model = bngsim.Model.from_sbml_string(SBML_EVENT_PARAM_SENS)
    assert model.species_names == ["A", "B", "Q"], (
        "fixture no longer exercises the promotion: " + repr(model.species_names)
    )
    return bngsim.Simulator(model, method="ode", sensitivity_params=_SENS_PARAMS).run(
        t_span=(0, 2.0), n_points=11, rtol=1e-11, atol=1e-13
    )


def test_sensitivity_species_axis_matches_species_columns(promoted_param_sens_result):
    """The tensor's species axis is `species_names`, not the raw state vector.

    GH #202: the promoted parameter Q is integrator state but not a trajectory
    column, and the sensitivity tensor used to keep its row while
    ``species``/``species_names`` dropped it — two arrays handed to the same
    user callback disagreeing about what a species index means.
    """
    r = promoted_param_sens_result
    assert r.species_names == ["A", "B"]
    assert r.species.shape == (11, 2)
    assert r.sensitivities.shape == (11, 2, 2)
    assert r.sensitivities.shape[:2] == r.species.shape


def test_sensitivity_rows_are_the_species_they_name(promoted_param_sens_result):
    """Row i really is d(species_names[i])/dp — closed form, both rows."""
    r = promoted_param_sens_result
    t = np.asarray(r.time)
    sens = np.asarray(r.sensitivities)
    ik1, iamp = _SENS_PARAMS.index("k1"), _SENS_PARAMS.index("amp")

    np.testing.assert_allclose(sens[:, 0, ik1], -2.0 * t * np.exp(-t), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sens[:, 1, ik1], 2.0 * t * np.exp(-t), rtol=1e-6, atol=1e-8)
    # Neither reported species depends on the promoted symbol's amplitude.
    assert np.all(sens[:, :, iamp] == 0.0)


def test_promoted_symbol_derivative_is_still_reachable(promoted_param_sens_result):
    """Dropping Q's row loses nothing: the loader gives every promoted symbol a
    same-named observable, so d(Q)/d(amp) is an ``observable:`` selector away."""
    r = promoted_param_sens_result
    t = np.asarray(r.time)
    dQ = np.asarray(r.output_sensitivities("observable:Q"))[:, 0, :]
    iamp = _SENS_PARAMS.index("amp")
    # Q := amp fires at t > 0.5, so dQ/damp steps 0 → 1 there.
    assert np.all(dQ[t <= 0.5, iamp] == 0.0)
    np.testing.assert_allclose(dQ[t > 0.5, iamp], 1.0, rtol=1e-9, atol=1e-9)


def test_gradient_composes_with_result_species(promoted_param_sens_result):
    """`Result.gradient`'s own documented workflow — a `dL/dY` built from
    ``result.species`` — runs, and returns the closed-form gradient.

    With data ≡ 0 the SSE gradient is
    ``Σ_t 2·A·(dA/dk1) + 2·B·(dB/dk1) = Σ_t 8·t·e^(-t)·(1 - 2·e^(-t))``.
    """
    r = promoted_param_sens_result
    t = np.asarray(r.time)
    data = np.zeros_like(np.asarray(r.species))

    grad = r.gradient(lambda sp, _t: 2.0 * (sp - data))
    assert grad.shape == (2,)

    expected_k1 = float(np.sum(8.0 * t * np.exp(-t) * (1.0 - 2.0 * np.exp(-t))))
    ik1, iamp = _SENS_PARAMS.index("k1"), _SENS_PARAMS.index("amp")
    assert grad[ik1] == pytest.approx(expected_k1, rel=1e-6, abs=1e-8)
    assert grad[iamp] == 0.0

    # The convenience objectives take the same species-shaped data.
    loss, sse_grad = r.sse_gradient(data)
    assert loss == pytest.approx(float(np.sum(np.asarray(r.species) ** 2)))
    np.testing.assert_allclose(sse_grad, grad, rtol=1e-12, atol=0.0)
    for method in (r.chi2_gradient, r.neg_log_likelihood_gradient):
        assert method(data, 1.0)[1].shape == (2,)


def test_species_axis_consumers_accept_n_species(promoted_param_sens_result):
    """Everything else keyed to the species axis sizes off ``n_species`` too."""
    r = promoted_param_sens_result
    n = len(r.species_names)

    # A per-species sigma vector for the FIM (GH #202: used to demand n+1).
    assert r.fisher_information(np.ones(n)).shape == (2, 2)
    # An indexed species selector (the tensor row and the name now agree).
    np.testing.assert_allclose(
        np.asarray(r.output_sensitivities("species:B"))[:, 0, :],
        np.asarray(r.sensitivities)[:, 1, :],
        rtol=0.0,
        atol=0.0,
    )
    # And the xarray view, which labels the tensor rows with `species_names`.
    xr = pytest.importorskip("xarray")
    ds = r.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["state"] == n
    assert list(ds.coords["state"].values) == r.species_names


@pytest.mark.skipif(not _REPRO_MODEL.is_file(), reason="benchmarks/sbml_events corpus not present")
def test_gradient_runs_on_the_reported_model():
    """GH #202's reproducer: a real model whose promoted symbol (`u`) made the
    documented `Result.gradient` scipy workflow raise for any data shaped like
    ``result.species``."""
    model = bngsim.Model.load(str(_REPRO_MODEL))
    assert "u" in model.species_names, "BIOMD0000000701 no longer promotes `u`"

    r = bngsim.Simulator(model, method="ode", sensitivity_params=["alpha", "konB"]).run(
        t_span=(0.0, 100.0), n_points=21
    )
    assert "u" not in r.species_names
    assert r.sensitivities.shape == (21, len(r.species_names), 2)

    data = np.zeros_like(np.asarray(r.species))
    assert r.gradient(lambda sp, _t: 2.0 * (sp - data)).shape == (2,)
