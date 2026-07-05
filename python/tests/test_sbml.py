"""Tests for SBML model loading — Session 25.

Tests the SBML loader (_sbml_loader.py) including:
  - Direct SBML loading (from_sbml, from_sbml_string)
  - Antimony → SBML path (from_antimony uses SBML internally)
  - initialAssignment handling
  - Assignment rules on parameters
  - _safe_name renaming (ExprTk reserved words)
  - Boundary species handling
"""

import math

import bngsim
import pytest

pytest.importorskip("antimony")


# ─── Basic SBML loading via Antimony strings ──────────────────────────────────


def test_sbml_from_antimony_string():
    """Load a simple decay model via Antimony string → SBML path."""
    text = """
    model test_decay()
        S = 10.0
        k = 0.3
        J0: S -> ; k*S
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    assert model.n_species >= 1

    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 20), n_points=201)

    expected = 10.0 * math.exp(-0.3 * 20.0)
    actual = result.species[-1, 0]
    assert abs(actual - expected) < 1e-3, f"S(20) = {actual}, expected {expected}"


def test_sbml_two_species_reaction():
    """Load a two-species conversion reaction."""
    text = """
    model conversion()
        compartment C = 1;
        species A in C = 10.0;
        species B in C = 0.0;
        k = 0.5;
        J0: A -> B; k * A;
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    assert model.n_species >= 2

    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 10), n_points=101)

    # A(t) = 10*exp(-0.5*t), B(t) = 10*(1 - exp(-0.5*t))
    A_end = result.species[-1, 0]
    B_end = result.species[-1, 1]
    expected_A = 10.0 * math.exp(-0.5 * 10.0)
    expected_B = 10.0 * (1.0 - math.exp(-0.5 * 10.0))
    assert abs(A_end - expected_A) < 1e-3
    assert abs(B_end - expected_B) < 1e-3


def test_sbml_rate_rule():
    """Pure ODE model with rate rules (Antimony S' = ... syntax)."""
    text = """
    model rate_rule()
        S = 5.0
        k = 1.0
        S' = -k*S
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 3), n_points=31)

    expected = 5.0 * math.exp(-3.0)
    actual = result.species[-1, 0]
    assert abs(actual - expected) < 1e-4


def test_sbml_assignment_rule():
    """Model with an assignment rule (computed observable)."""
    text = """
    model assignment()
        S = 10.0
        k = 0.1
        total := S
        J0: S -> ; k * S
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    # Should load without error
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 5), n_points=51)
    # S should decay
    assert result.species[-1, 0] < 10.0


def test_sbml_reserved_word_species():
    """Species named with ExprTk reserved word gets renamed."""
    text = """
    model reserved()
        var default_sp = 10.0;
        k = 0.1;
        default_sp' = -k * default_sp;
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 5), n_points=51)
    assert result.species[-1, 0] < 10.0


def test_sbml_boundary_species():
    """Boundary species should remain fixed during simulation."""
    text = """
    model boundary()
        compartment C = 1;
        species $E in C = 1.0;
        species S in C = 10.0;
        species P in C = 0.0;
        k = 0.5;
        J0: S -> P; k * E * S;
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 5), n_points=51)

    # E should remain at 1.0 (boundary)
    E_vals = result.species[:, 0]
    assert abs(E_vals[-1] - 1.0) < 1e-10


def test_sbml_initial_assignment():
    """Test that initialAssignment elements are handled (Session 25).

    Creates a model where a species IC is computed from parameters.
    """
    text = """
    model init_assign()
        compartment C = 1;
        species S in C;
        species P in C = 0.0;
        // S's IC is computed from total and fraction
        total_conc = 100.0;
        fraction = 0.3;
        S = total_conc * fraction;
        k = 0.1;
        J0: S -> P; k * S;
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 10), n_points=101)

    # S should start at 30.0 and decay
    S_0 = result.species[0, 0]
    assert abs(S_0 - 30.0) < 1e-6, f"S(0) = {S_0}, expected 30.0"

    expected = 30.0 * math.exp(-0.1 * 10.0)
    actual = result.species[-1, 0]
    assert abs(actual - expected) < 1e-3


def test_sbml_piecewise():
    """Model with piecewise expression (maps to ExprTk if())."""
    text = """
    model pw()
        S = 10.0
        k1 = 0.1
        k2 = 1.0
        // Switch rate at S = 5
        S' = -piecewise(k1, S > 5, k2) * S
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 20), n_points=201)
    # Should decay (not blow up)
    assert result.species[-1, 0] > 0
    assert result.species[-1, 0] < 10.0


def test_sbml_piecewise_guarded_negative_power_folds_lazily():
    """A piecewise guarding a negative-base fractional power folds to the real
    branch, not a spurious complex.

    The real cube-root idiom ``if(x>0, x^(1/3), -(abs(x))^(1/3))`` is real for
    all x, but the *un-taken* first arm ``x^(1/3)`` is complex under Python's
    ``**`` when x<0. The load-time constant-folder must evaluate piecewise
    lazily (taken branch only) — eager folding of both arms previously surfaced
    the complex value and crashed ``from_sbml`` with a TypeError. Here dS/dt is
    the real cube root of -8 = -2, so S(1) = -2 exactly.
    """
    text = """
    model cbrt_guard()
        x = -8.0
        cr := piecewise(x^(1/3), x > 0, -(abs(x)^(1/3)))
        S = 0.0
        S' = cr
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    assert result.species[-1, 0] == pytest.approx(-2.0, abs=1e-9)


def test_sbml_clone():
    """Clone works for SBML-loaded models."""
    text = """
    model test()
        S = 5.0; k = 1.0
        S' = -k*S
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    clone = model.clone()
    assert clone.n_species == model.n_species
    assert clone.n_parameters == model.n_parameters


def test_from_sbml_string():
    """Load model directly from SBML XML string."""
    # Minimal SBML Level 3 document
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="test">
        <listOfCompartments>
          <compartment id="C" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="S" compartment="C" initialConcentration="10"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="0.3" constant="true"/>
        </listOfParameters>
        <listOfReactions>
          <reaction id="J0" reversible="false">
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
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.n_species == 1
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 10), n_points=101)
    expected = 10.0 * math.exp(-0.3 * 10.0)
    actual = result.species[-1, 0]
    assert abs(actual - expected) < 1e-3


def test_sbml_l3_initial_assignment_on_speciesreference():
    """SBML L3 lets <initialAssignment symbol="<srid>"> set the value of a
    speciesReference whose ``stoichiometry`` attribute is omitted. Without
    resolution at load time, libsbml returns NaN for the stoichiometry,
    the ODE picks up NaN coefficients and CVODE fails the corrector at
    t=0. Mirrors SBML test 01071-01076 / 01094-01095.
    """
    # S1 + 2*p1*S2 → S3 ; k*S1*S2 ; only the S2 stoichiometry comes from
    # the IA so a single-species smoke test wouldn't exercise the path.
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="iasr">
        <listOfCompartments>
          <compartment id="C" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="S1" compartment="C" initialAmount="1"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
          <species id="S2" compartment="C" initialAmount="2"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
          <species id="S3" compartment="C" initialAmount="0"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="0.5" constant="true"/>
          <parameter id="p1" value="1" constant="true"/>
        </listOfParameters>
        <listOfInitialAssignments>
          <initialAssignment symbol="srS2_ref">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><cn type="integer">2</cn><ci>p1</ci></apply>
            </math>
          </initialAssignment>
        </listOfInitialAssignments>
        <listOfReactions>
          <reaction id="J0" reversible="false">
            <listOfReactants>
              <speciesReference species="S1" stoichiometry="1" constant="true"/>
              <speciesReference id="srS2_ref" species="S2" constant="true"/>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="S3" stoichiometry="1" constant="true"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>S1</ci><ci>S2</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ode")
    # Sim must not crash with corrector failure at t=0.
    result = sim.run(t_span=(0, 0.1), n_points=11)
    s1 = result.species[:, 0]
    s2 = result.species[:, 1]
    s3 = result.species[:, 2]
    # Sanity: S2 consumed twice as fast as S1, S3 created at S1's rate.
    # The 2:1 ratio is the whole point — without the IA-resolved stoich,
    # S2 would consume at 1× S1's rate (or NaN out completely).
    ds1 = s1[0] - s1[-1]
    ds2 = s2[0] - s2[-1]
    ds3 = s3[-1] - s3[0]
    assert ds1 > 0 and ds2 > 0 and ds3 > 0, "all three should change"
    assert abs(ds2 - 2.0 * ds1) < 1e-6 * max(abs(ds1), 1.0), (
        f"S2 should be consumed at 2× S1's rate; got dS1={ds1}, dS2={ds2}"
    )
    assert abs(ds3 - ds1) < 1e-6 * max(abs(ds1), 1.0), (
        f"S3 should be produced at S1's rate; got dS3={ds3}, dS1={ds1}"
    )


def test_sbml_l2_stoichiometrymath_constant_fold():
    """SBML L2 expresses non-unity stoichiometry via the deprecated
    <stoichiometryMath> child of <speciesReference>. Constant-fold it
    over parameters at load time so the L2 form behaves identically to
    the L3 IA-on-speciesReference form. Mirrors SBML test 00388.
    """
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
      <model id="stoichmath">
        <listOfCompartments>
          <compartment id="C" size="1"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="S1" compartment="C" initialAmount="1"/>
          <species id="S2" compartment="C" initialAmount="2"/>
          <species id="S3" compartment="C" initialAmount="0"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="0.5"/>
          <parameter id="p1" value="1"/>
        </listOfParameters>
        <listOfReactions>
          <reaction id="J0" reversible="false">
            <listOfReactants>
              <speciesReference species="S1"/>
              <speciesReference species="S2">
                <stoichiometryMath>
                  <math xmlns="http://www.w3.org/1998/Math/MathML">
                    <apply><times/><cn type="integer">2</cn><ci>p1</ci></apply>
                  </math>
                </stoichiometryMath>
              </speciesReference>
            </listOfReactants>
            <listOfProducts>
              <speciesReference species="S3"/>
            </listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>k</ci><ci>S1</ci><ci>S2</ci></apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 0.1), n_points=11)
    s1 = result.species[:, 0]
    s2 = result.species[:, 1]
    ds1 = s1[0] - s1[-1]
    ds2 = s2[0] - s2[-1]
    assert abs(ds2 - 2.0 * ds1) < 1e-6 * max(abs(ds1), 1.0), (
        f"L2 stoichiometryMath should yield 2× consumption of S2; got dS1={ds1}, dS2={ds2}"
    )


def test_sbml_assignment_rule_overrides_initial_amount():
    """An assignmentRule on a species must hold at t=0, overriding the
    raw initialAmount even when the latter is inconsistent. Mirrors SBML
    test 00621: declares S3 initialAmount=3.75 but rule says
    S3 := 0.75*S2 = 0.375 at t=0.
    """
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="aroverride">
        <listOfCompartments>
          <compartment id="C" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="S2" compartment="C" initialAmount="0.5"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
          <species id="S3" compartment="C" initialAmount="3.75"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k1" value="0.75" constant="true"/>
        </listOfParameters>
        <listOfRules>
          <assignmentRule variable="S3">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><ci>k1</ci><ci>S2</ci></apply>
            </math>
          </assignmentRule>
        </listOfRules>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 1.0), n_points=11)
    # S3 should track the rule from t=0 onward, not the raw 3.75.
    s3_t0 = result.species[0, 1]
    assert abs(s3_t0 - 0.375) < 1e-6, f"S3(0)={s3_t0}: assignmentRule must override initialAmount"


def test_sbml_function_definition_in_assignment_rule_constant_folds():
    """A user-defined function call (e.g. multiply(k1, S2)) appearing
    inside an assignmentRule must be inlined when the loader resolves
    the t=0 species value — otherwise the species sticks at its raw
    initialAmount. Mirrors SBML test 00635.
    """
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="fdar">
        <listOfFunctionDefinitions>
          <functionDefinition id="multiply">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <lambda>
                <bvar><ci>x</ci></bvar>
                <bvar><ci>y</ci></bvar>
                <apply><times/><ci>x</ci><ci>y</ci></apply>
              </lambda>
            </math>
          </functionDefinition>
        </listOfFunctionDefinitions>
        <listOfCompartments>
          <compartment id="C" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="S2" compartment="C" initialAmount="0.5"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
          <species id="S3" compartment="C" initialAmount="999"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k1" value="0.75" constant="true"/>
        </listOfParameters>
        <listOfRules>
          <assignmentRule variable="S3">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <ci>multiply</ci>
                <ci>k1</ci>
                <ci>S2</ci>
              </apply>
            </math>
          </assignmentRule>
        </listOfRules>
      </model>
    </sbml>"""
    model = bngsim.Model.from_sbml_string(sbml)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 1.0), n_points=11)
    s3_t0 = result.species[0, 1]
    assert abs(s3_t0 - 0.375) < 1e-6, (
        f"S3(0)={s3_t0}: assignmentRule with user function must override initialAmount"
    )


def test_sbml_function_definition():
    """SBML function definition gets inlined correctly."""
    # Function definitions must be outside the model block in Antimony
    text = """
    function hill(x, K, n)
        x^n / (K^n + x^n)
    end

    model funcdef()
        S = 0.0
        signal = 10.0
        K = 5.0
        n = 2.0
        kprod = 1.0
        kdeg = 0.1
        S' = kprod * hill(signal, K, n) - kdeg * S
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 100), n_points=101)

    # Should reach steady state: S_ss = kprod * hill(10, 5, 2) / kdeg
    # = 1.0 * (100/125) / 0.1 = 8.0
    S_end = result.species[-1, 0]
    assert abs(S_end - 8.0) < 0.1, f"S_ss = {S_end}, expected ~8.0"
