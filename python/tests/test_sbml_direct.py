"""Tests for direct SBML .xml loading — Session 55.

Tests the SBML L2 local parameter fix and AST type handling that
enables loading raw SBML .xml files without Antimony intermediary.
Prior to this fix, only 630/978 (64%) BioModels loaded; after: 968/978 (99%).

Root causes fixed:
  1. SBML L2 local parameters: kl.getParameter(j), not kl.getLocalParameter(j)
  2. AST_NAME_AVOGADRO (type 307) → 6.02214076e23
  3. AST_FUNCTION_DELAY (type 320) → first argument (ODE approximation)

Note: as of GH #113 a non-trivial ``delay()`` is refused at load by default
(``ModelError``); the first-argument approximation in (3) now applies only under
``BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1``. See test_unsupported_constructs.py.
"""

import math

import bngsim

# ─── SBML L2 with local parameters ───────────────────────────────────────

SBML_L2_LOCAL_PARAMS = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="test_l2_local">
    <listOfCompartments>
      <compartment id="cell" size="1"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="cell" initialConcentration="10"
               boundaryCondition="false"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/>
              <ci>cell</ci><ci>k</ci><ci>S</ci>
            </apply>
          </math>
          <listOfParameters>
            <parameter id="k" value="0.3"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_sbml_l2_local_params_load():
    """SBML L2 models with KineticLaw local parameters load correctly."""
    model = bngsim.Model.from_sbml_string(SBML_L2_LOCAL_PARAMS)
    assert model.n_species >= 1
    assert model.n_reactions >= 1


def test_sbml_l2_local_params_simulate():
    """SBML L2 local parameters produce correct simulation results."""
    model = bngsim.Model.from_sbml_string(SBML_L2_LOCAL_PARAMS)
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 10), n_points=101)

    # S(t) = 10 * exp(-0.3*t), S(10) ≈ 0.498
    expected = 10.0 * math.exp(-0.3 * 10.0)
    actual = result.species[-1, 0]
    assert abs(actual - expected) < 1e-3, f"S(10) = {actual}, expected {expected}"


SBML_L2_MULTI_LOCAL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="test_l2_multi">
    <listOfCompartments>
      <compartment id="c" size="1"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="10"/>
      <species id="B" compartment="c" initialConcentration="0"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="fwd" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>c</ci><ci>kf</ci><ci>A</ci></apply>
          </math>
          <listOfParameters>
            <parameter id="kf" value="0.5"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
      <reaction id="rev" reversible="false">
        <listOfReactants>
          <speciesReference species="B" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>c</ci><ci>kr</ci><ci>B</ci></apply>
          </math>
          <listOfParameters>
            <parameter id="kr" value="0.1"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_sbml_l2_multiple_reactions_local_params():
    """Multiple reactions each with own local params don't collide."""
    model = bngsim.Model.from_sbml_string(SBML_L2_MULTI_LOCAL)
    assert model.n_species >= 2
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 50), n_points=101)
    # At steady state: kf*A = kr*B, A+B=10
    # A_ss = 10*kr/(kf+kr) = 10*0.1/0.6 ≈ 1.667
    A_end = result.species[-1, 0]
    B_end = result.species[-1, 1]
    assert abs(A_end + B_end - 10.0) < 0.1, "Conservation violated"
    assert abs(A_end - 10 * 0.1 / 0.6) < 0.2


# ─── Real SBML file loading (BIOMD0000000003 — was the first failure) ────


def test_biomd3_loads(data_dir):
    """BIOMD0000000003 (Goldbeter 1991) loads from raw SBML."""
    path = data_dir / "BIOMD0000000003.xml"
    model = bngsim.Model.from_sbml(str(path))
    assert model.n_species == 3
    assert model.n_reactions > 0


def test_biomd3_simulates(data_dir):
    """BIOMD0000000003 produces non-trivial ODE output."""
    path = data_dir / "BIOMD0000000003.xml"
    model = bngsim.Model.from_sbml(str(path))
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 100), n_points=101)
    # Should not be constant (oscillatory model)
    assert result.species.shape[0] == 101
    assert result.species.max() > result.species.min()


# ─── SBML L3 still works (regression guard) ──────────────────────────────


def test_sbml_l3_still_works():
    """SBML L3 loading via from_sbml_string still works after L2 fix."""
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core"
          level="3" version="2">
      <model id="test_l3">
        <listOfCompartments>
          <compartment id="C" size="1" constant="true"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="S" compartment="C" initialConcentration="5"
                   boundaryCondition="false" constant="false"
                   hasOnlySubstanceUnits="false"/>
        </listOfSpecies>
        <listOfParameters>
          <parameter id="k" value="1.0" constant="true"/>
        </listOfParameters>
        <listOfReactions>
          <reaction id="J0" reversible="false">
            <listOfReactants>
              <speciesReference species="S" stoichiometry="1"
                               constant="true"/>
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
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 3), n_points=31)
    expected = 5.0 * math.exp(-3.0)
    actual = result.species[-1, 0]
    assert abs(actual - expected) < 1e-4
