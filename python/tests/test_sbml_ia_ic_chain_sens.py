"""``<initialAssignment>`` chains that the IC-sensitivity lowering used to drop.

The sibling of ``test_sbml_initial_assignment_ic_sens.py``. That file covers an
IC expression over plain constant parameters; this one covers the three shapes
the predicate rejected, each of which cost the *whole* seed for every parameter
the expression reads (issue #379):

* a parameter declared ``constant="false"`` that nothing writes;
* an expression reading **another state**, meaning that state's initial value;
* a reference through an ``<assignmentRule>`` whose own expression is constant.

Plus the parameter-axis twin: a ``<parameter>``'s initialAssignment reading a
species, which froze every parameter it also read behind the section-0 fold.

All four were silent — a finite, smooth, wrong column — and all four are
invisible to a finite difference through bngsim, because ``set_param`` did not
re-resolve the initialAssignment either, so the oracle held ``x(0)`` fixed in
exactly the same way the seed did. The closed forms below are the in-repo
replacement for the second engine that separated them.

The last class is the guard, not a fix: the shape BIOMD0000000856 named, where
lowering an expression over a symbol the model can move wrote a wrong initial
condition. A wrong IC is far worse than a missing seed, so it is pinned here.
"""

import bngsim
import numpy as np
import pytest


def _run(sbml, params, t_end=4.0, n=5, rtol=1e-12, atol=1e-14):
    m = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(m, method="ode", sensitivity_params=params).run(
        t_span=(0, t_end), n_points=n, rtol=rtol, atol=atol
    )
    return m, np.asarray(r.time), np.asarray(r.sensitivities)


# ── 1. An unwritten `constant="false"` parameter ──────────────────────────────
# BIOMD0000000611's shape: `Dilution` is declared non-constant, no rule and no
# event writes it, and it divides all 17 species initialAssignments. One symbol
# failing the predicate withheld the seed from every one of them -- 18 of 106
# columns identically zero, including the largest in the tensor.
#
# S(0) = S0/D and dS/dt = -k*S, so S(t) = (S0/D)*exp(-k t):
#   dS/dS0 =  exp(-k t)/D          dS/dD = -(S0/D**2)*exp(-k t)
SBML_NONCONST_DIVISOR = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="nonconst_divisor">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="S0" value="12" constant="true"/>
      <parameter id="D" value="4" constant="false"/>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/><ci>S0</ci><ci>D</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>S</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


class TestUnwrittenNonConstantParameter:
    def test_ic_value_is_unchanged(self):
        assert bngsim.Model.from_sbml_string(SBML_NONCONST_DIVISOR)._core.get_concentration(
            "S"
        ) == pytest.approx(3.0)

    def test_both_factors_are_seeded(self):
        _m, t, s = _run(SBML_NONCONST_DIVISOR, ["S0", "D"])
        np.testing.assert_allclose(s[:, 0, 0], np.exp(-0.5 * t) / 4.0, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            s[:, 0, 1], -(12.0 / 16.0) * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9
        )

    def test_the_defect_was_a_silent_zero(self):
        """Pin the failure mode. ``S0`` reaches the trajectory only through the
        IC, so with no seed its column is identically zero -- and nothing warns."""
        _m, _t, s = _run(SBML_NONCONST_DIVISOR, ["S0"])
        assert np.max(np.abs(s[:, 0, 0])) == pytest.approx(0.25, rel=1e-6)

    def test_set_param_re_resolves_the_initial_condition(self):
        moved = bngsim.Model.from_sbml_string(SBML_NONCONST_DIVISOR)
        moved.set_param("D", 2.0)
        assert moved._core.get_concentration("S") == pytest.approx(6.0)


# ── 2. An IC expression reading another state ─────────────────────────────────
# BIOMD0000001102's shape (11 of its 12 initialAssignments), and
# BIOMD0000000838 / 643 / 644 / 645. `B(0) = A*r` means A's *initial value*.
# `C(0) = B*q` chains one step further, which is what forces the dependency
# ordering: `_ic_C` has to be declared after `_ic_B`.
#
# A(0) = 5 (a declared constant, so dA(0)/dtheta = 0), B(0) = 5r, C(0) = 5rq.
# Each decays with its own rate, so dB/dr = 5*exp(-k t) and dC/dr = 5q*exp(-k t).
SBML_IC_READS_STATE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ic_reads_state">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="5"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="r" value="3" constant="true"/>
      <parameter id="q" value="2" constant="true"/>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="B">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>A</ci><ci>r</ci></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="C">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>B</ci><ci>q</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>A</ci></apply></apply>
        </math>
      </rateRule>
      <rateRule variable="B">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>B</ci></apply></apply>
        </math>
      </rateRule>
      <rateRule variable="C">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>C</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


class TestIcExpressionReadsAnotherState:
    def test_ic_values_are_unchanged(self):
        core = bngsim.Model.from_sbml_string(SBML_IC_READS_STATE)._core
        assert core.get_concentration("B") == pytest.approx(15.0)
        assert core.get_concentration("C") == pytest.approx(30.0)

    def test_one_hop_column_is_the_closed_form(self):
        """``B(0) = A*r`` with A's IC a declared constant: dB/dr = 5*exp(-k t)."""
        m, t, s = _run(SBML_IC_READS_STATE, ["r"])
        b = list(m.species_names).index("B")
        np.testing.assert_allclose(s[:, b, 0], 5.0 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9)

    def test_two_hop_column_chains_through_the_first(self):
        """``C(0) = B*q`` and ``B(0) = A*r``, so dC/dr = 5q*exp(-k t) -- the term
        exists only if the lowering resolved ``B`` to B's own IC expression
        rather than folding it to a number."""
        m, t, s = _run(SBML_IC_READS_STATE, ["r", "q"])
        c = list(m.species_names).index("C")
        np.testing.assert_allclose(s[:, c, 0], 10.0 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(s[:, c, 1], 15.0 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9)

    def test_the_defect_was_a_silent_zero(self):
        m, _t, s = _run(SBML_IC_READS_STATE, ["r"])
        c = list(m.species_names).index("C")
        assert np.max(np.abs(s[:, c, 0])) == pytest.approx(10.0, rel=1e-6)


# ── 3. A reference through a constant assignmentRule ──────────────────────────
# BIOMD0000000807's shape: `n(0) = n0`, `n0 = N0/K`, `N0 = r_N/mu_N - 1`. A rule
# that reads no state is a constant of the built model, but the target slot is
# function-backed, so `add_species_param_ref` dropped the link and mu_N, r_N, K,
# G0 and A0 all lost their seed. The rule chain is inlined instead of referenced.
#
# S(0) = s0 = S0/K, dS/dt = -k*S  =>  dS/dS0 = exp(-k t)/K
SBML_IC_VIA_CONST_RULE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ic_via_const_rule">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="S0" value="20" constant="true"/>
      <parameter id="K" value="4" constant="true"/>
      <parameter id="k" value="0.5" constant="true"/>
      <parameter id="s0" value="0" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>s0</ci></math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <assignmentRule variable="s0">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/><ci>S0</ci><ci>K</ci></apply>
        </math>
      </assignmentRule>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>S</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


class TestIcThroughConstantAssignmentRule:
    def test_ic_value_is_unchanged(self):
        assert bngsim.Model.from_sbml_string(SBML_IC_VIA_CONST_RULE)._core.get_concentration(
            "S"
        ) == pytest.approx(5.0)

    def test_columns_are_the_closed_form(self):
        _m, t, s = _run(SBML_IC_VIA_CONST_RULE, ["S0", "K"])
        np.testing.assert_allclose(s[:, 0, 0], np.exp(-0.5 * t) / 4.0, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            s[:, 0, 1], -(20.0 / 16.0) * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9
        )

    def test_the_defect_was_a_silent_zero(self):
        _m, _t, s = _run(SBML_IC_VIA_CONST_RULE, ["S0"])
        assert np.max(np.abs(s[:, 0, 0])) == pytest.approx(0.25, rel=1e-6)


# ── 4. The parameter axis: a <parameter>'s IA reading a species ───────────────
# BIOMD0000001102's `k_MET_expression = MET*k_MET_degradation +
# MET*k_phospho_MET_basal`, a rate constant. The species reference put the whole
# expression outside the section-2 lift, which froze BOTH rate parameters behind
# the fold -- the model warns that it did, then answers a wrong column anyway.
#
# kexp = M*kd with M(0) = 4 a declared constant, dS/dt = kexp
#   =>  S(t) = 1 + 4*kd*t,  dS/dkd = 4t
SBML_PARAM_IA_READS_SPECIES = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="param_ia_reads_species">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="M" compartment="c" initialConcentration="4"
               hasOnlySubstanceUnits="false" boundaryCondition="true" constant="true"/>
      <species id="S" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kd" value="0.25" constant="true"/>
      <parameter id="kexp" value="0" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="kexp">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>M</ci><ci>kd</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>kexp</ci></math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


class TestParameterInitialAssignmentReadsSpecies:
    def test_the_folded_value_is_unchanged(self):
        assert bngsim.Model.from_sbml_string(SBML_PARAM_IA_READS_SPECIES).get_param(
            "kexp"
        ) == pytest.approx(1.0)

    def test_kd_column_is_the_closed_form(self):
        """dS/dkd = M(0)*t = 4t -- zero before the lift reached this shape."""
        m, t, s = _run(SBML_PARAM_IA_READS_SPECIES, ["kd"])
        i = list(m.species_names).index("S")
        np.testing.assert_allclose(s[:, i, 0], 4.0 * t, rtol=1e-7, atol=1e-9)

    def test_the_defect_was_a_silent_zero(self):
        m, _t, s = _run(SBML_PARAM_IA_READS_SPECIES, ["kd"])
        i = list(m.species_names).index("S")
        assert np.max(np.abs(s[:, i, 0])) == pytest.approx(16.0, rel=1e-6)

    def test_set_param_moves_the_derived_constant(self):
        moved = bngsim.Model.from_sbml_string(SBML_PARAM_IA_READS_SPECIES)
        moved.set_param("kd", 0.5)
        assert moved.get_param("kexp") == pytest.approx(2.0)


# ── 5. The guard: a symbol the model CAN move stays folded ────────────────────
# BIOMD0000000856's shape, and the reason the predicate is strict. `NSt` is
# written by an event, so this loader promotes it to a species; an IC expression
# reading it is not a parameter expression, and lowering it once evaluated the
# synthetic parameter to 0 and wrote that 0 back over the species' real initial
# condition -- a moved trajectory, which is far worse than a missing seed.
SBML_EVENT_WRITTEN_UPSTREAM = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="event_written_upstream">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="W" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="NSt" value="50" constant="false"/>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="W">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>0.66</cn><ci>NSt</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="W">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>W</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="bump" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text"
              definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
              <cn>2</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="NSt">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>10</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


class TestEventWrittenUpstreamStaysFolded:
    """The IC must be the section-0 fold, exactly, and the trajectory must be
    the one that fold implies. This is the invariant the old `getConstant()`
    filter was believed to protect; it is the subtraction of the event targets
    that actually protects it, and this pins that."""

    def test_the_initial_condition_is_correct(self):
        core = bngsim.Model.from_sbml_string(SBML_EVENT_WRITTEN_UPSTREAM)._core
        assert core.get_concentration("W") == pytest.approx(33.0)

    def test_the_trajectory_is_the_one_that_ic_implies(self):
        """W(t) = 33*exp(-k t) up to the event, which touches only NSt."""
        m = bngsim.Model.from_sbml_string(SBML_EVENT_WRITTEN_UPSTREAM)
        r = m and bngsim.Simulator(m, method="ode").run(
            t_span=(0, 1.5), n_points=4, rtol=1e-12, atol=1e-14
        )
        t = np.asarray(r.time)
        w = np.asarray(r.species)[:, list(m.species_names).index("W")]
        np.testing.assert_allclose(w, 33.0 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9)
