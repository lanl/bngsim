"""Regression for GH #135: the ODE integrator must tolerate a transiently
negative concentration in a rate law that is non-finite for a negative base
(e.g. a non-integer Hill power, ``pow(conc, 3.98)``).

BIOMD0000000994/995/996 (COPASI TGF-β models) failed ``CVODE flag=-4`` under the
default analytical-Jacobian config: the **correct** closed-form Jacobian is exact
enough that CVODE steps confidently, and the BDF predictor pushes a zero-pinned
fast species slightly negative — where ``pow(conc, 3.98)`` is NaN (NaN for *any*
negative base, so step reduction never escapes it). (The finite-difference Jacobian
survives because its perturbed RHS evals route through the clamped RHS callback —
RoadRunner integrates all three fine; the analytical Jacobian itself matches SymPy
and FD exactly.)

The fix has TWO halves, both in ``src/cvode_simulator.cpp``, both conditional on a
non-finite result so a model whose RHS/Jacobian is *finite* at a transiently-
negative concentration stays byte-identical (an unconditional clamp was rejected —
it froze mass-action species at small negatives and made BIOMD628/879 chatter):

  1. RHS (GH #135 first ship): the RHS callback recomputes the derivatives on a
     nonnegative-CLAMPED copy of the state, but ONLY after the *unclamped* RHS
     comes back non-finite. Resolves the original early-time NaN.
  2. Jacobian (this ship): the analytical/codegen dense+sparse Jacobian callbacks
     re-fill on the clamped state, but ONLY after the unclamped fill produces a
     non-finite entry. The first ship's premise — that the offending overshoot is
     a predictor excursion the RHS sees while the *reused* Jacobian sits at the
     last accepted nonnegative state — held for the early-time case but NOT at the
     t≈180 ligand-wash/injection dose, where CVODE re-evaluates the analytical
     Jacobian AT the negative-base state: ``d/dx (k·x^3.98) = k·3.98·x^2.98`` is
     NaN there, poisoning the Newton matrix even though the RHS clamp kept the RHS
     finite. That is why 994/995/996 still failed (now at t≈180, not t≈1.9) until
     the Jacobian was clamped too. The clamped boundary value ``k·p·0^{p-1}=0``
     (p>1) is correct.

Concentrations are physically ``>= 0``, so the clamp is the correct boundary value
(the way RoadRunner keeps such a variable cleanly positive).

Reproducing CVODE's exact predictor overshoot in a minimal model is unreliable
(it depends on the whole system's stiffness), so this fixture uses an event that
*deterministically* forces a species negative mid-integration while a
non-integer-power rate law reads it. Before the relevant half of the fix this
raised CV_CONV_FAILURE at the event time; after it, the conditional clamp keeps the
RHS (and Jacobian) finite and the solve runs to completion.
"""

from __future__ import annotations

import numpy as np
import pytest

import bngsim

# X decays slowly (a finite, well-behaved mass-action law); Y is produced at rate
# Vmax * X^3.98 — NaN for X < 0. An event at t=0.5 forces X = -1e-3, so the very
# next RHS evaluation reads a negative base.
_SBML_FRAC_POW_EVENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"><model id="frac_pow_neg">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>
<species id="X" compartment="C" initialConcentration="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
<species id="Y" compartment="C" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters>
<parameter id="kx" value="0.1" constant="true"/><parameter id="Vmax" value="3" constant="true"/>
<parameter id="hill" value="3.98592596228747" constant="true"/></listOfParameters>
<listOfReactions>
<reaction id="xdeg" reversible="false"><listOfReactants><speciesReference species="X" stoichiometry="1" constant="true"/></listOfReactants><kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>C</ci><ci>kx</ci><ci>X</ci></apply></math></kineticLaw></reaction>
<reaction id="yprod" reversible="false"><listOfProducts><speciesReference species="Y" stoichiometry="1" constant="true"/></listOfProducts><kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>C</ci><ci>Vmax</ci><apply><power/><ci>X</ci><ci>hill</ci></apply></apply></math></kineticLaw></reaction>
</listOfReactions>
<listOfEvents>
<event id="force_neg" useValuesFromTriggerTime="true">
<trigger initialValue="false" persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><geq/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>0.5</cn></apply></math></trigger>
<listOfEventAssignments><eventAssignment variable="X"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><cn>0.001</cn></apply></math></eventAssignment></listOfEventAssignments>
</event></listOfEvents>
</model></sbml>"""


def test_negative_state_in_fractional_power_rhs_recovers(monkeypatch: pytest.MonkeyPatch):
    """A non-integer-power rate law survives a (forced) negative concentration: the
    RHS would be NaN at the negative base, but the conditional nonnegative-clamp
    retry keeps it finite. Before the RHS half of the fix this raised
    CV_CONV_FAILURE at the event time. Uses the FD Jacobian so the check isolates
    the RHS path; the analytical-Jacobian half is covered by the next test."""
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", "0")
    m = bngsim.Model.from_sbml_string(_SBML_FRAC_POW_EVENT)
    r = bngsim.Simulator(m, method="ode").run(
        t_span=(0.0, 2.0), n_points=40, rtol=1e-8, atol=1e-10
    )
    assert r.time[-1] == pytest.approx(2.0)
    assert np.all(np.isfinite(np.asarray(r.species, dtype=float)))


def test_negative_state_in_fractional_power_jacobian_recovers(
    monkeypatch: pytest.MonkeyPatch,
):
    """The analytical Jacobian must survive the same forced-negative state. This is
    the half the RHS-only ship (GH #135 first cut) missed: at the dose discontinuity
    CVODE re-evaluates the analytical Jacobian AT the negative base, and
    ``d/dx (Vmax·X^3.98) = Vmax·3.98·X^2.98`` is NaN there — poisoning the Newton
    matrix even though the RHS clamp kept the RHS finite. With the analytical
    Functional Jacobian forced ON, this raised CV_CONV_FAILURE (flag=-4, |h|=hmin)
    at the event before the Jacobian clamp; after it, the conditional Jacobian
    re-fill on the clamped state keeps the matrix finite and the solve completes.
    This is the minimal mirror of BIOMD0000000994/995/996 failing at t≈180."""
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", "1")
    m = bngsim.Model.from_sbml_string(_SBML_FRAC_POW_EVENT)
    # The fractional-power derivative is exactly what makes the Jacobian non-finite;
    # confirm the analytical Jacobian is actually the one in play (not an FD fallback).
    assert m.prepare_analytical_jacobian() is True
    r = bngsim.Simulator(m, method="ode").run(
        t_span=(0.0, 2.0), n_points=40, rtol=1e-8, atol=1e-10
    )
    assert r.time[-1] == pytest.approx(2.0)
    assert np.all(np.isfinite(np.asarray(r.species, dtype=float)))


@pytest.mark.parametrize("analytical", ["1", "0"], ids=["analytical_jac", "fd_jac"])
def test_clamp_is_noop_on_a_valid_trajectory(
    monkeypatch: pytest.MonkeyPatch, analytical: str
):
    """The conditional clamp must not perturb a physical (nonnegative) solve: with
    the event removed, X decays cleanly and never goes negative, so the unclamped
    RHS is always finite and the clamp never engages. The analytical-Jacobian and
    finite-difference trajectories are therefore both clean and mutually
    consistent."""
    no_event = _SBML_FRAC_POW_EVENT.replace(
        _SBML_FRAC_POW_EVENT[
            _SBML_FRAC_POW_EVENT.index("<listOfEvents>") : _SBML_FRAC_POW_EVENT.index(
                "</listOfEvents>"
            )
            + len("</listOfEvents>")
        ],
        "",
    )
    monkeypatch.setenv("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", analytical)
    m = bngsim.Model.from_sbml_string(no_event)
    if analytical == "1":
        assert m.prepare_analytical_jacobian() is True
    r = bngsim.Simulator(m, method="ode").run(
        t_span=(0.0, 5.0), n_points=51, rtol=1e-10, atol=1e-12
    )
    sp = np.asarray(r.species, dtype=float)
    assert np.all(np.isfinite(sp)) and np.all(sp >= -1e-12)
    # X = exp(-kx t); Y = ∫ Vmax X^h dt — both smooth and well above noise.
    iX = list(r.species_names).index("X")
    assert sp[10, iX] == pytest.approx(np.exp(-0.1 * r.time[10]), rel=1e-5)
