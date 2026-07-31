"""``set_param`` re-resolves a species initial condition named by a parameter (#79).

``A() Stot`` in a ``.net`` species block — or an SBML ``initialAssignment`` that
is a bare ``<ci>`` — declares A's initial condition to BE that parameter. Before
this fix ``set_param("Stot", …)`` moved the parameter and nothing else: the
dependency was recorded in ``species_ic_param_refs`` and simply never consulted,
so ``get_state()`` and ``reset()`` both returned the network-generation value.
A dose scan over a total amount — the common case, since a "total" is exactly
the kind of parameter a species IC is written in terms of — silently ran every
dose at the load-time initial condition, with ``get_param`` confirming the
write. 406 of the 585 ``ode_fullnet`` corpus models have such an IC.

What the fix promises, and what each class below pins:

  * **The declared IC follows the parameter** — ``initial_conc`` always, so
    ``reset()`` rebuilds from current parameter values rather than a load-time
    snapshot, and the ``set_params(); reset()`` sequence inside ``run_batch`` /
    ``steady_state_batch`` is correct without knowing about IC parameters.
  * **Through derived parameters too** — ``R() Rtot`` with ``Rtot = 0.5*R0``
    moves when *either* ``R0`` or the parameter selecting the branch moves. The
    re-resolve runs after the derived-parameter re-evaluation and covers every
    ref, not just refs naming the written parameter.
  * **The live state follows only while it IS the IC** — a species the dynamics
    advanced holds a value the parameter no longer describes, and clobbering it
    would discard a pre-equilibration mid-protocol.
  * **A saved baseline retires the rebuild** — ``save_concentrations()``
    redefines ``initial_conc`` to a captured state, which the declared IC no
    longer describes.
  * **The #43 forward-sensitivity seed still applies** — the rebuild keeps the
    live state ON the baseline, which is exactly the condition #113 tests.
"""

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"

# `R() R0` — a species IC named by a DIRECT primary parameter. R0 is in no rate
# law, so the initial condition is its only route into the trajectory.
IC_DIRECT_NET = str(DATA_DIR / "ic_direct.net")
# `R() Rtot` with `Rtot = R0` — the same, one derived hop away.
IC_DERIVED_NET = str(DATA_DIR / "ic_derived.net")
# `R() Rtot` with `Rtot = if((sel>=1)&&(sel<10), R0, 0.5*R0)`. Two things this
# fixture buys that the others cannot: a NON-unit coefficient (drive `sel` out
# of the true branch and the IC halves), and a write to a parameter the species
# line never names — so it pins "re-resolve every ref", not "re-resolve the refs
# to the parameter just written".
IC_DERIVED_COMPOUND_NET = str(DATA_DIR / "ic_derived_compound.net")

R0 = 100.0  # the fixtures' declared R0


def _model(net=IC_DIRECT_NET) -> bngsim.Model:
    return bngsim.Model.from_net(net)


def _iR(m: bngsim.Model) -> int:
    return m.species_names.index("R()")


class TestDeclaredIcFollowsTheParameter:
    """The issue's reproducer: the write reaches ``get_state()`` and ``reset()``."""

    def test_get_state_moves_with_the_parameter(self):
        m = _model()
        i = _iR(m)
        assert m.get_state()[i] == R0
        m.set_param("R0", 1e6)
        assert m.get_param("R0") == 1e6
        assert m.get_state()[i] == 1e6

    def test_reset_rebuilds_from_current_parameter_values(self):
        """``reset()`` returns to the CURRENT IC, not a snapshot taken at load."""
        m = _model()
        i = _iR(m)
        m.set_param("R0", 1e6)
        m.set_concentration("R()", 7.0)
        m.reset()
        assert m.get_state()[i] == 1e6

    def test_initial_state_is_the_new_baseline(self):
        m = _model()
        i = _iR(m)
        m.set_param("R0", 250.0)
        assert m._core.get_initial_state()[i] == 250.0

    def test_set_params_moves_it_too(self):
        """The bulk setter funnels through the same core call."""
        m = _model()
        m.set_params({"R0": 42.0, "kf": 1.0})
        assert m.get_state()[_iR(m)] == 42.0

    def test_unrelated_parameter_leaves_the_ic_alone(self):
        m = _model()
        i = _iR(m)
        m.set_param("kf", 9.0)
        assert m.get_state()[i] == R0

    def test_a_model_with_no_parameter_named_ic_is_untouched(self):
        """No refs ⇒ the re-resolve loop has nothing to iterate."""
        m = bngsim.Model.from_net(str(DATA_DIR / "two_species_reversible.net"))
        assert list(m._core.species_ic_param_refs) == []
        before = np.asarray(m.get_state()).copy()
        for name in m.param_names:
            m.set_param(name, m.get_param(name) * 2.0 + 1.0)
        np.testing.assert_array_equal(m.get_state(), before)


class TestDerivedParameterIcs:
    """A derived (ConstantExpression) IC re-resolves through the chain."""

    def test_derived_ic_follows_its_primary(self):
        m = _model(IC_DERIVED_NET)  # Rtot = R0
        i = _iR(m)
        m.set_param("R0", 400.0)
        assert m.get_param("Rtot") == 400.0
        assert m.get_state()[i] == 400.0

    def test_writing_the_derived_parameter_itself_also_works(self):
        m = _model(IC_DERIVED_NET)
        i = _iR(m)
        m.set_param("Rtot", 33.0)  # detaches the expression, BNG setParameter semantics
        assert m.get_state()[i] == 33.0

    def test_non_unit_coefficient(self):
        """`sel` out of the true branch halves the IC — the value, not an identity."""
        m = _model(IC_DERIVED_COMPOUND_NET)
        i = _iR(m)
        assert m.get_state()[i] == R0
        m.set_param("sel", 0.5)  # false branch: Rtot = 0.5*R0
        assert m.get_param("Rtot") == 0.5 * R0
        assert m.get_state()[i] == 0.5 * R0

    def test_a_ref_the_written_parameter_does_not_name_still_re_resolves(self):
        """`sel` appears in no species line; the IC moves anyway.

        The re-resolve runs over EVERY ref after the derived-parameter
        re-evaluation, because any of them may have moved. Restricting it to the
        refs naming the written parameter would fix the direct case and leave
        this one silently broken — the shape of the original bug.
        """
        m = _model(IC_DERIVED_COMPOUND_NET)
        assert "sel" not in {m.param_names[p] for _, p in m._core.species_ic_param_refs}
        m.set_param("sel", 0.5)
        assert m.get_state()[_iR(m)] == 0.5 * R0


class TestLiveStateIsNotClobbered:
    """``concentration`` follows only while the species is still ON the baseline."""

    def test_an_advanced_state_keeps_its_value(self):
        m = _model()
        i = _iR(m)
        sim = bngsim.Simulator(m)
        sim.run(t_span=(0, 2.0), n_points=3)
        advanced = m.get_state()[i]
        assert advanced < R0  # R decays
        m.set_param("R0", 1e6)
        assert m.get_state()[i] == advanced  # the dynamics state survives
        assert m._core.get_initial_state()[i] == 1e6  # ...but the baseline moved
        m.reset()
        assert m.get_state()[i] == 1e6

    def test_a_hand_assigned_state_keeps_its_value(self):
        m = _model()
        i = _iR(m)
        m.set_concentration("R()", 7.0)
        m.set_param("R0", 1e6)
        assert m.get_state()[i] == 7.0
        assert m._core.get_initial_state()[i] == 1e6


class TestSavedBaselineRetiresTheRebuild:
    """``save_concentrations()`` redefines the baseline; the declared IC retires."""

    def test_flag_latches_on_save(self):
        m = _model()
        assert m._core.ic_baseline_saved is False
        m.save_concentrations()
        assert m._core.ic_baseline_saved is True

    def test_a_named_snapshot_does_not_latch_it(self):
        """A labeled snapshot is a side store; it never touches ``initial_conc``."""
        m = _model()
        m.save_concentrations(label="t0")
        assert m._core.ic_baseline_saved is False
        m.set_param("R0", 1e6)
        assert m.get_state()[_iR(m)] == 1e6

    def test_equilibrated_baseline_survives_a_later_set_param(self):
        m = _model()
        i = _iR(m)
        sim = bngsim.Simulator(m)
        sim.run(t_span=(0, 2.0), n_points=3)
        m.save_concentrations()
        equilibrated = m.get_state()[i]
        m.set_param("R0", 1e6)
        assert m._core.get_initial_state()[i] == equilibrated
        m.reset()
        assert m.get_state()[i] == equilibrated


class TestScanAndBatchPaths:
    """The workflows the issue reports as silently returning identical results."""

    @staticmethod
    def _t0(result, model) -> float:
        return float(np.asarray(result.observables)[0, model.observable_names.index("R_tot")])

    def test_parameter_scan_starts_each_point_at_its_own_dose(self):
        """Every point, not just the first.

        ``parameter_scan`` restores the live state per point but the IC baseline
        is model state too: without rewinding the scanned parameter first, point
        1 onward would find the species off a baseline the previous point had
        already moved, decline to touch the live value, and run at the
        invocation dose.
        """
        m = _model()
        sim = bngsim.Simulator(m)
        doses = [1.0, 10.0, 1000.0]
        results = sim.parameter_scan("R0", doses, t_span=(0, 1e-9), n_points=2)
        assert [self._t0(r, m) for r in results] == pytest.approx(doses, rel=1e-9)

    def test_parameter_scan_leaves_the_model_as_it_found_it(self):
        m = _model()
        i = _iR(m)
        sim = bngsim.Simulator(m)
        sim.parameter_scan("R0", [1.0, 10.0], t_span=(0, 1e-9), n_points=2)
        assert m.get_param("R0") == R0
        assert m.get_state()[i] == R0
        assert m._core.get_initial_state()[i] == R0

    def test_run_batch_starts_each_set_at_its_own_dose(self):
        m = _model()
        sim = bngsim.Simulator(m)
        doses = [1.0, 10.0, 1000.0]
        results = sim.run_batch((0, 1e-9), 2, params=[{"R0": d} for d in doses])
        assert [self._t0(r, m) for r in results] == pytest.approx(doses, rel=1e-9)

    def test_steady_state_batch_conserves_each_set_s_total(self):
        """R -> P with no sink: R_ss + P_ss is the dose the point started from."""
        m = _model()
        sim = bngsim.Simulator(m)
        doses = [1.0, 250.0]
        results = sim.steady_state_batch([{"R0": d} for d in doses])
        totals = [float(np.sum(r.concentrations)) for r in results]
        assert totals == pytest.approx(doses, rel=1e-6)


class TestSensitivitySeedStillApplies:
    """The rebuild must not look like the assignment that retires a #43 seed."""

    def test_dR_dR0_is_one_after_a_dose_write(self):
        """∂R(0)/∂R0 = 1 and R decays as exp(-kf t), so ∂R(t)/∂R0 = exp(-kf t).

        The #113 rule retires a parameter-graph seed when the live state is off
        the declared baseline. The rebuild moves both together, so the row stays
        live — which is what makes a gradient-based fit over a dose parameter
        work at all.
        """
        m = _model()
        m.set_param("R0", 250.0)
        sim = bngsim.Simulator(m, sensitivity_params=["R0"])
        res = sim.run(t_span=(0, 2.0), n_points=3)
        s = np.asarray(res.sensitivities)  # (n_times, n_species, n_params)
        i = _iR(m)
        kf = m.get_param("kf")
        expected = np.exp(-kf * np.asarray(res.time))
        np.testing.assert_allclose(s[:, i, 0], expected, rtol=1e-6, atol=1e-9)

    def test_matches_a_rebuild_finite_difference(self):
        """Central difference of the whole solve in R0 — the non-circular oracle.

        This is the check #79 used to make impossible: re-solving at R0 ± h did
        not move the initial condition, so the FD oracle read 0 and any analytic
        ∂/∂R0 through an IC looked like a codegen bug.

        The observables are linear in R0 here, so the difference quotient has no
        truncation error and the solver tolerance is the whole error budget:
        tightened to 1e-12/1e-14, a 1%-of-R0 step keeps the quotient's noise
        floor (~tol·|obs| / 2h) two orders below the 1e-7 asserted.
        """
        h = 0.01 * R0
        tol = dict(rtol=1e-12, atol=1e-14)

        def solve(r0: float) -> np.ndarray:
            m = _model()
            m.set_param("R0", r0)
            sim = bngsim.Simulator(m)
            return np.asarray(sim.run(t_span=(0, 2.0), n_points=3, **tol).observables)

        fd = (solve(R0 + h) - solve(R0 - h)) / (2.0 * h)

        m = _model()
        sim = bngsim.Simulator(m, sensitivity_params=["R0"])
        res = sim.run(t_span=(0, 2.0), n_points=3, **tol)
        analytic = np.asarray(res.sensitivities_observables)[:, :, 0]
        np.testing.assert_allclose(analytic, fd, rtol=1e-7, atol=1e-9)


class TestCloneCarriesTheFlag:
    """``ic_baseline_saved`` is per-instance mutable state — see the clone contract."""

    def test_clone_of_a_fresh_model_still_rebuilds(self):
        m = _model()
        c = m.clone()
        assert c._core.ic_baseline_saved is False
        c.set_param("R0", 1e6)
        assert c.get_state()[_iR(c)] == 1e6

    def test_clone_of_a_saved_model_keeps_the_baseline_retired(self):
        m = _model()
        i = _iR(m)
        m.set_concentration("R()", 7.0)
        m.save_concentrations()
        c = m.clone()
        assert c._core.ic_baseline_saved is True
        c.set_param("R0", 1e6)
        assert c._core.get_initial_state()[i] == 7.0


class TestSbmlInitialAssignment:
    """The SBML half: a bare-``<ci>`` initialAssignment is the same declaration.

    It also carries the unit conversion the ``.net`` path never needs. A
    ``hasOnlySubstanceUnits="true"`` species' symbol denotes an AMOUNT while the
    engine stores ``amount / V``, so a parameter naming its IC names an amount
    and has to be divided by the compartment volume — the same division the
    loader's ``initialAmount`` / ``initialAssignment`` branches do. The
    load-time re-resolve dropped it, loading such a species V times too large;
    ``set_param`` now applies it too, from the one shared rule.
    """

    HOSU_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="hosu">
    <listOfCompartments>
      <compartment id="c" size="5" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialAmount="1" hasOnlySubstanceUnits="%s"
               boundaryCondition="false" constant="false"/>
      <species id="P" compartment="c" initialAmount="0" hasOnlySubstanceUnits="%s"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="Stot" value="100" constant="true"/>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> Stot </ci></math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="r1" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>S</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

    @staticmethod
    def _load(tmp_path, hosu: str) -> bngsim.Model:
        pytest.importorskip("libsbml")
        path = tmp_path / f"hosu_{hosu}.xml"
        path.write_text(TestSbmlInitialAssignment.HOSU_SBML % (hosu, hosu))
        return bngsim.Model.from_sbml(str(path))

    def test_concentration_valued_species_takes_the_parameter_directly(self, tmp_path):
        m = self._load(tmp_path, "false")
        i = m.species_names.index("S")
        assert m.get_state()[i] == 100.0
        m.set_param("Stot", 500.0)
        assert m.get_state()[i] == 500.0

    def test_amount_valued_species_is_divided_by_the_compartment_volume(self, tmp_path):
        """V = 5, so an amount of 100 is stored as 20 — at load AND on a write.

        The load-time half of this was wrong before the fix: the re-resolve wrote
        the raw parameter value over the ``amount / V`` the loader had computed,
        so S loaded at 100 while an identical model spelling the same amount as
        ``initialAmount="100"`` loaded at 20.
        """
        m = self._load(tmp_path, "true")
        i = m.species_names.index("S")
        assert m.get_state()[i] == 20.0
        m.set_param("Stot", 500.0)
        assert m.get_state()[i] == 100.0
