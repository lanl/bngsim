"""The behaviour keys in ``capabilities()`` mean what they say — issue #431.

Most keys in ``capabilities()["features"]`` report what was compiled in or what
optional package is installed, and a test of one is a test of a build flag. The
four keys #431 added report something else: what this build **computes**. Each
one stands for a fix whose absence is a wrong number rather than a refusal, so
the key is a claim about arithmetic, and a claim about arithmetic is worth
exactly as much as the measurement behind it.

This file is that measurement. For each key it runs the case the key is about
and checks the answer against something independent of bngsim — a closed form,
an analytical solution, or the emitted source — and pins the key to it. The
point is that the key cannot go on being published ``True`` after the behaviour
it names has gone away: the two are asserted together, here, in one place a
reader can go to and find out what the key actually promises.

The full proofs are elsewhere and much larger — ``test_event_sensitivity.py``
(a thousand lines on the event jump), ``test_codegen_xcompartment_sens.py``,
``test_atol_vector.py``, ``test_atol_tracking.py``. Nothing here replaces them.
What is added is the link from the published key to the behaviour, which is the
thing a consumer reads the key for.

The reporting contract around the keys — always published, ``False`` carries an
explanation, the probes read the compiled extension — is in
``test_capabilities.py``.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# libsbml is a base dependency rather than an extra, so the SBML fixtures below
# are not gated on it — the same call the rest of the suite makes.


def _features() -> dict[str, bool]:
    return bngsim.capabilities()["features"]


# ── issue #144 / #146: forward sensitivities across a discrete event ─────────
#
# S decays at rate k, and at t = 1 an event halves it: `S := 0.5·S`. That
# assignment READS the species it writes, which is the repeat-dosing idiom
# (`A := A + dose`) in its smallest form, and it is the case that used to be
# answered wrongly. An SBML species token binds to its same-named observable
# rather than to the concentration address, so the jump's ∂h/∂x matched nothing
# and came out 0 — which restarts the sensitivity column from zero, and dS/dt =
# −kS is linear, so zero is a fixed point and the column stayed there.

SBML_HALVING_EVENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="halving">
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
      <event id="halve" useValuesFromTriggerTime="true">
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
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><cn>0.5</cn><ci>S</ci></apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


def _halving_event_sensitivity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t, dS/dk as computed, dS/dk in closed form)`` across the event.

    S = 10·e^{−kt} before the halving and 5·e^{−kt} after it (the factor of a
    half at t = 1 is undone by e^{+k} in the restart), so dS/dk is −10·t·e^{−kt}
    and then −5·t·e^{−kt}. No finite difference and no second bngsim run: the
    oracle is arithmetic done by hand.
    """
    model = bngsim.Model.from_sbml_string(SBML_HALVING_EVENT)
    result = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 4), n_points=9, rtol=1e-11, atol=1e-13
    )
    t = np.asarray(result.time)
    computed = np.asarray(result.sensitivities)[:, 0, 0]
    closed_form = np.where(t < 1.0, -10.0 * t * np.exp(-0.5 * t), -5.0 * t * np.exp(-0.5 * t))
    return t, computed, closed_form


class TestEventSensitivities:
    def test_the_key_is_published_true_here(self):
        """Stated first, because everything below is what it claims.

        A build that would publish ``False`` cannot reach these tests anyway:
        the compiled extension is built from this tree, and the pytest
        preflight refuses to run at all against one that has fallen behind it.
        """
        assert _features()["event_sensitivities"] is True

    def test_a_state_reading_event_assignment_matches_the_closed_form(self):
        t, computed, closed_form = _halving_event_sensitivity()
        np.testing.assert_allclose(computed, closed_form, rtol=1e-6, atol=1e-8)
        # The defect's own signature, kept as a separate assertion because it is
        # what a consumer is gating on: the column did NOT restart from zero at
        # the event. A build that dropped ∂h/∂x reported 0.0 here and regrew
        # from there, finite and plausible the whole way.
        at_event = computed[t == 1.0][0]
        assert abs(at_event) > 3.0

    def test_the_run_is_not_refused(self):
        """The other half of what makes this key necessary rather than nice.

        A build without the fix does not raise, so a consumer cannot learn the
        answer by catching something. It gets a full, finite tensor — this one,
        with a term missing.
        """
        model = bngsim.Model.from_sbml_string(SBML_HALVING_EVENT)
        result = bngsim.Simulator(model, method="ode", sensitivity_params=["k"]).run(
            t_span=(0, 4), n_points=9
        )
        assert result.sensitivities.shape == (9, model._core.n_species, 1)
        assert np.all(np.isfinite(result.sensitivities))


# ── issue #160: the analytic ∂f/∂p across a cross-compartment reaction ───────
#
# A and B live in compartments of different size, so the transport reaction
# between them is `per_species_volume_scaling`: its kinetic law is an amount per
# time while each species stores an amount per its own volume, so every row of
# it divides by a different number. One reaction like this used to make the
# whole model decline the analytic sensitivity RHS, because CVODES installs one
# callback for every column.

_XCOMP_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="xcomp">
    <listOfCompartments>
      <compartment id="C1" size="1" constant="true"/>
      <compartment id="C2" size="{v2}" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C1" initialConcentration="100" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="C2" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="kb" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="transport" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C1</ci><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="degB" reversible="false" fast="false">
        <listOfReactants><speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>C2</ci><apply><times/><ci>kb</ci><ci>B</ci></apply></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _xcomp_model(v2="5"):
    """The two-compartment model, warmed the way a real run warms it."""
    pytest.importorskip("sympy")
    model = bngsim.Model.from_sbml_string(_XCOMP_SBML.format(v2=v2))
    bngsim.Simulator(model, method="ode")
    return model


def _is_cross_compartment(model) -> bool:
    return any(
        rxn.get("per_species_volume_scaling", False)
        for rxn in model._core.codegen_data()["reactions"]
    )


def _emits_analytic_sens_rhs(model) -> bool:
    """Whether codegen produced a sensitivity RHS for this model.

    ``False`` is not an error: it is the model going back on CVODES' internal
    difference quotients, which is the correct-but-slow path this key is about.
    """
    from bngsim import _codegen

    _source, has_sens = _codegen.generate_combined_from_model(model)
    return bool(has_sens)


class TestCrossCompartmentSensitivities:
    def test_the_fixture_is_actually_cross_compartment(self):
        """The volume is what makes it so, and 1.0 is the control: at equal
        volumes there is no divide, nothing is flagged, and the fixture would
        be testing nothing."""
        assert _is_cross_compartment(_xcomp_model("5"))
        assert not _is_cross_compartment(_xcomp_model("1"))

    def test_the_key_matches_what_the_model_gets(self):
        """The key and the emitter, asserted against each other.

        Written as an equality rather than as ``is True`` so it holds in both
        configurations: with the A/B hatch set (the test below) both sides go
        False together, which is the whole claim the key makes.
        """
        assert _features()["cross_compartment_sensitivities"] == _emits_analytic_sens_rhs(
            _xcomp_model("5")
        )

    def test_true_on_this_build(self):
        assert _features()["cross_compartment_sensitivities"] is True
        assert _emits_analytic_sens_rhs(_xcomp_model("5")) is True

    def test_the_key_follows_the_ab_hatch(self, monkeypatch):
        """``BNGSIM_NO_FUNCTIONAL_SENS_RHS=1`` is the only thing that can turn
        this off, and when it does the key says so.

        A published ``False`` is an answer a consumer can act on — here, by
        expecting a run that is slower by orders of magnitude on a stiff model
        rather than one that is wrong.
        """
        monkeypatch.setenv("BNGSIM_NO_FUNCTIONAL_SENS_RHS", "1")
        assert _features()["cross_compartment_sensitivities"] is False
        assert _emits_analytic_sens_rhs(_xcomp_model("5")) is False
        why = bngsim.capabilities()["missing"]["cross_compartment_sensitivities"]
        assert "BNGSIM_NO_FUNCTIONAL_SENS_RHS" in why


# ── issues #196 and #213: the absolute tolerance across species, and in time ──


@pytest.fixture
def wide_range(data_dir: Path):
    """Two decoupled decays nine decades apart — ``wide_dynamic_range.net``."""
    model = bngsim.Model.from_net(str(data_dir / "wide_dynamic_range.net"))
    return bngsim.Simulator(model, method="ode"), model


@pytest.fixture
def deep_decay(data_dir: Path):
    """One species that decays sixteen decades — ``deep_decay.net``."""
    model = bngsim.Model.from_net(str(data_dir / "deep_decay.net"))
    return bngsim.Simulator(model, method="ode"), model


class TestPerSpeciesAtol:
    def test_the_key_is_published_true_here(self):
        assert _features()["per_species_atol"] is True

    def test_a_vector_resolves_a_species_no_scalar_can(self, wide_range):
        """S starts at 1e-9 and decays; the default scalar atol of 1e-8 sits an
        order of magnitude above it for the whole run, so it is not
        error-controlled at all and comes back negative where the analytical
        answer is positive. One vector states a tolerance for each end.
        """
        sim, model = wide_range
        idx = model.species_names.index("S()")
        exact = 1.0e-9 * np.exp(-10.0 * 0.5)

        model.reset()
        scalar = sim.run(t_span=(0.0, 0.5), n_points=6, atol=1e-8)
        assert scalar.species[-1, idx] < 0.0

        model.reset()
        vector = sim.run(
            t_span=(0.0, 0.5),
            n_points=6,
            atol=[1e-17] + [1e-8] * (model.n_species - 1),
        )
        assert vector.species[-1, idx] == pytest.approx(exact, rel=1e-3)


class TestTrackingAtol:
    def test_the_key_is_published_true_here(self):
        assert _features()["tracking_atol"] is True

    def test_the_tolerance_follows_the_trajectory(self, deep_decay):
        """D decays sixteen decades, so any fixed tolerance — vector included —
        is outgrown partway through. Measured against ``exp(-t)``: fixed comes
        back thousands of times too large, tracking gets it.
        """
        sim, model = deep_decay
        idx = model.species_names.index("D()")
        exact = float(np.exp(-30.0))

        def d_at_end(atol):
            model.reset()
            result = sim.run(
                t_span=(0.0, 30.0), n_points=31, rtol=1e-8, max_steps=1_000_000, atol=atol
            )
            return float(result.species[-1, idx])

        assert d_at_end("auto") > 1e3 * exact
        assert d_at_end(bngsim.TrackingAtol()) == pytest.approx(exact, rel=1e-4)
