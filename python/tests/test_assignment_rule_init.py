"""Tests for Bug #2 fix: assignment-rule resolution in initialAssignment eval_ctx.

Session 58.  When an SBML initialAssignment references a variable defined by
an assignment rule (e.g., ``param_init = B`` where ``B := A``), the numeric
evaluator must resolve assignment rules BEFORE evaluating initialAssignments.
Without this fix, ``B`` has no value in the eval_ctx → ``param_init`` gets
the wrong value → incorrect rate expressions.

Regression test: Antimony model with ``B := A; param = B`` pattern that
mirrors the real BIOMD0000001026 failure mode.
"""

import math

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")


class TestAssignmentRuleInit:
    """Assignment-rule values must be available to initialAssignment evaluator."""

    def test_param_initialized_from_assignment_rule(self):
        """Parameter whose IC references an assignment-rule variable.

        Model:  B := A  (assignment rule: B always equals A)
                param_init has initialAssignment = B
                So param_init should equal A = 5.0.
                Reaction: S' = -param_init * S
                Expected: S(t) = 10 * exp(-5*t)
        """
        text = """
        model ar_init()
            A = 5.0;
            B := A;
            var S = 10.0;
            param_init = B;
            S' = -param_init * S;
        end
        """
        model = bngsim.Model.from_antimony_string(text)
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 1.0), n_points=101)

        # S(1) = 10 * exp(-5*1) ≈ 0.0674
        expected = 10.0 * math.exp(-5.0)
        actual = result.species[-1, 0]
        assert abs(actual - expected) / expected < 1e-4, (
            f"S(1) = {actual:.6e}, expected {expected:.6e}. "
            "Assignment-rule variable B not resolved in initialAssignment."
        )

    def test_chained_assignment_rules_in_init(self):
        """Multi-level chain: C := B, B := A, param = C.

        Tests that the iterative multi-pass resolution handles chains.
        """
        text = """
        model chained()
            A = 3.0;
            B := A;
            C := B;
            var S = 1.0;
            rate_const = C;
            S' = -rate_const * S;
        end
        """
        model = bngsim.Model.from_antimony_string(text)
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 1.0), n_points=101)

        expected = math.exp(-3.0)
        actual = result.species[-1, 0]
        assert abs(actual - expected) / expected < 1e-4, (
            f"S(1) = {actual:.6e}, expected {expected:.6e}. Chained assignment rules not resolved."
        )

    def test_assignment_rule_with_expression(self):
        """Assignment rule involves an expression: B := 2*A + 1.

        param_init = B should be 2*5 + 1 = 11.
        """
        text = """
        model expr_ar()
            A = 5.0;
            B := 2*A + 1;
            var S = 10.0;
            rate_k = B;
            S' = -rate_k * S;
        end
        """
        model = bngsim.Model.from_antimony_string(text)
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 0.1), n_points=11)

        expected = 10.0 * math.exp(-11.0 * 0.1)
        actual = result.species[-1, 0]
        assert abs(actual - expected) / expected < 1e-3, (
            f"S(0.1) = {actual:.6e}, expected {expected:.6e}."
        )

    def test_difference_of_assignment_rule_vars(self):
        """Mimics BIOMD1026 pattern: Mpl = A*exp(c*t) - B*exp(d*t) where B := A.

        At t=0, Mpl should be A - A = 0, so dy/dt ≈ 0.
        Without the fix, B might not resolve to A's value.
        """
        text = """
        model metformin()
            compartment C = 1;
            species Mrbc in C = 0.0;
            A_val = 100.0;
            B_val := A_val;
            c_rate = -0.01;
            d_rate = -0.02;

            // Mpl is time-dependent assignment rule
            Mpl := A_val * exp(c_rate * time) - B_val * exp(d_rate * time);

            // Reaction: influx depends on Mpl
            Kin = 0.5;
            Kout = 0.1;
            RBC = 1.0;
            Mrbc' = RBC * Kin * Mpl - RBC * Kout * Mrbc;
        end
        """
        model = bngsim.Model.from_antimony_string(text)
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 1.0), n_points=101)

        # At t=0, Mpl = 100 - 100 = 0, so Mrbc starts near 0 and stays small
        # If bug present, B_val != A_val → Mpl(0) != 0 → Mrbc grows large
        mrbc_final = result.species[-1, 0]
        # Mrbc should be modest (Mpl grows slowly from 0)
        assert abs(mrbc_final) < 10.0, (
            f"Mrbc(1) = {mrbc_final:.6e}, expected small. "
            "B_val := A_val not resolved → Mpl(0) ≠ 0."
        )

    def test_cross_validate_with_roadrunner(self):
        """Cross-validate the assignment-rule-init pattern against RR."""
        pytest.importorskip("roadrunner")
        import roadrunner

        text = """
        model xval_ar()
            A = 5.0;
            B := A;
            var S = 10.0;
            param_init = B;
            S' = -param_init * S;
        end
        """
        # BNGsim
        model = bngsim.Model.from_antimony_string(text)
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 1.0), n_points=11)
        bg_vals = np.asarray(result.species)[:, 0]

        # RoadRunner
        import antimony

        antimony.clearPreviousLoads()
        antimony.loadString(text)
        mod = antimony.getModuleNames()[-1]
        sbml = antimony.getSBMLString(mod)
        rr = roadrunner.RoadRunner(sbml)
        rr.integrator.absolute_tolerance = 1e-12
        rr.integrator.relative_tolerance = 1e-8
        rr_result = rr.simulate(0, 1.0, 11)
        rr_vals = np.array(rr_result)[:, 1]

        max_rel_err = np.max(np.abs(bg_vals - rr_vals) / np.maximum(np.abs(rr_vals), 1e-12))
        assert max_rel_err < 1e-4, f"max_rel_err = {max_rel_err:.2e} between BNGsim and RR"


class TestInitialAssignmentTracksAssignmentRuleTarget:
    """Issue #73: an initialAssignment that references an assignment-rule target
    must use the AR's t=0 value, not the target's stale raw ``value=``.

    COPASI exports duplicate a quantity as both a raw-valued parameter and an
    assignment-rule target (``cin0`` in MODEL1606100000), then point another
    parameter's initialAssignment at it (``ModelValue_19 := cin0``). The loader's
    guarded IA-eval loop read ``cin0``'s stale raw ``value=`` (322026) instead of
    its assignment-rule value (-4807974); the post-loop AR override fixed ``cin0``
    but did not re-propagate into ``ModelValue_19``, flipping the sign of the
    boundary species ``Osmin`` away from RoadRunner and COPASI. Per SBML an
    assignment rule holds at t=0, so the AR value is the correct IA input.
    """

    # MV references cin (an AR target with a stale raw value=99); cin := MV18 - Met1;
    # Met1 := G (hOSU amount=20). Correct: cin = 15 - 20 = -5, MV = cin = -5 (not 99).
    SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="ia_ar_staleness">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="G" compartment="c" initialAmount="20"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="Osm" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="MV18" value="15" constant="true"/>
      <parameter id="Met1" value="99" constant="true"/>
      <parameter id="cin"  value="99" constant="false"/>
      <parameter id="MV"   value="99" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="Met1"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>G</ci></math></initialAssignment>
      <initialAssignment symbol="MV"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>cin</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <assignmentRule variable="cin"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><ci>MV18</ci><ci>Met1</ci></apply></math></assignmentRule>
      <assignmentRule variable="Osm"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>G</ci><ci>MV</ci></apply></math></assignmentRule>
    </listOfRules>
  </model>
</sbml>"""

    def test_ia_target_uses_assignment_rule_value_not_stale_raw(self):
        model = bngsim.Model.from_sbml_string(self.SBML)
        met1 = model.get_param("Met1")
        cin = model.get_param("cin")
        mv = model.get_param("MV")
        assert math.isclose(met1, 20.0, abs_tol=1e-9), f"Met1={met1}, expected G amount 20"
        assert math.isclose(cin, -5.0, abs_tol=1e-9), f"cin={cin}, expected MV18-Met1 = -5"
        # The bug: MV held the stale raw 99 instead of cin's AR value -5.
        assert math.isclose(mv, -5.0, abs_tol=1e-9), (
            f"MV={mv}; initialAssignment 'MV := cin' must track cin's "
            f"assignment-rule value (-5), not the stale raw value= (99)."
        )
        assert math.isclose(mv, cin, abs_tol=1e-9), "MV := cin must equal cin"
