"""ODE correctness for time-dependent piecewise discontinuities (GH #72).

A piecewise assignment rule that switches on the SBML ``time`` csymbol (a
drug-dosing window, a scheduled stimulus) puts a discontinuity in the ODE RHS.
With the default interpolated CVODE output the integrator takes large internal
steps and can jump clean over a narrow "on" window — silently dropping the
dose. Surfaced by BIOMD0000000879 (Rodrigues2019 chemoimmunotherapy), where the
chemo schedule is 7 infusions each only 0.125 time-units wide; pre-fix bngsim
delivered just the t=0 infusion (1 of 7), the cancer escaped to carrying
capacity (N→9.6e11) instead of being driven to the immune-controlled branch
(N→~2.5, matching a segmented SciPy oracle and libRoadRunner on a fine grid).

The fix registers each ``time`` inequality found in a piecewise rate/assignment
expression as a CVODE root, so the integrator stops exactly at every pulse edge
regardless of the output grid. This test locks:

  1. the loader extracts the right number of discontinuity triggers,
  2. a narrow pulse falling *between* coarse output samples is still delivered,
     matching a closed-form oracle and being grid-independent,
  3. models with no time-dependent piecewise register zero triggers (the
     integrator path is then bit-for-bit unchanged).

Oracle is closed-form: for ``dX/dt = inp(t) - d·X`` with ``inp = kin`` only on
``[t0, t0+w]`` (else 0), the post-pulse decay is
``X(t) = (kin/d)(1 - e^{-d·w}) · e^{-d·(t - (t0+w))}`` for ``t ≥ t0+w``.
"""

import math
import os

import bngsim
import numpy as np
import pytest

# ── Minimal pulse-train model ───────────────────────────────────────────────
# X starts at 0. A single narrow production pulse inp = KIN during the window
# [T0, T0+W] (deliberately off any integer/half grid), plus first-order decay
# d·X. The pulse is far narrower than the coarse output spacing used below, so
# an integrator that does not resolve the discontinuity misses it and leaves
# X ≈ 0.
KIN, DVAL, T0, W = 100.0, 1.0, 0.7, 0.05

SBML_PULSE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="pulse_train">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kin" value="100" constant="true"/>
      <parameter id="d" value="1" constant="true"/>
      <parameter id="inp" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="inp">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece>
              <ci>kin</ci>
              <apply><and/>
                <apply><geq/>
                  <csymbol encoding="text"
                    definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.7</cn></apply>
                <apply><leq/>
                  <csymbol encoding="text"
                    definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>0.75</cn></apply>
              </apply>
            </piece>
            <otherwise><cn>0</cn></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="prod" reversible="false">
        <listOfProducts>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><ci>inp</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><ci>d</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# No time-dependent piecewise anywhere ⇒ zero discontinuity triggers.
SBML_NO_PIECEWISE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="plain_decay">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="d" value="1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><ci>d</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

_CEIL_FACTORIAL_RATE = """<apply><divide/>
  <apply><factorial/>
    <apply><ceiling/>
      <apply><times/><ci>S1</ci><ci>p1</ci></apply>
    </apply>
  </apply>
  <ci>p2</ci>
</apply>"""

_CALCULATE_FUNCDEF = """    <listOfFunctionDefinitions>
      <functionDefinition id="calculate">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>
          <bvar><ci>x</ci></bvar>
          <bvar><ci>y</ci></bvar>
          <bvar><ci>z</ci></bvar>
          <apply><divide/>
            <apply><factorial/>
              <apply><ceiling/>
                <apply><times/><ci>x</ci><ci>y</ci></apply>
              </apply>
            </apply>
            <ci>z</ci>
          </apply>
        </lambda></math>
      </functionDefinition>
    </listOfFunctionDefinitions>"""

_CALCULATE_CALL = """<apply><ci>calculate</ci><ci>S1</ci><ci>p1</ci><ci>p2</ci></apply>"""


def _ceil_reaction_model(rate_math: str, funcdefs: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="state_ceil_reaction">
{funcdefs}
    <listOfCompartments>
      <compartment id="compartment" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="compartment" initialAmount="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S2" compartment="compartment" initialAmount="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" value="4" constant="true"/>
      <parameter id="p2" value="25" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="r1" reversible="false">
        <listOfReactants>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="S2" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">{rate_math}</math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


SBML_STATE_CEIL_REACTION = _ceil_reaction_model(_CEIL_FACTORIAL_RATE)
SBML_STATE_CEIL_FUNCDEF = _ceil_reaction_model(_CALCULATE_CALL, _CALCULATE_FUNCDEF)

SBML_STATE_CEIL_RATE_RULE = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="state_ceil_rate_rule">
    <listOfParameters>
      <parameter id="S1" value="1" constant="false"/>
      <parameter id="S2" value="0" constant="false"/>
      <parameter id="p1" value="4" constant="true"/>
      <parameter id="p2" value="25" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="S1">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/>{_CEIL_FACTORIAL_RATE}</apply>
        </math>
      </rateRule>
      <rateRule variable="S2">
        <math xmlns="http://www.w3.org/1998/Math/MathML">{_CEIL_FACTORIAL_RATE}</math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


def _pulse_oracle(t):
    """Closed-form X(t) for the single-pulse model, valid for t >= T0+W."""
    x_peak = (KIN / DVAL) * (1.0 - math.exp(-DVAL * W))
    return x_peak * math.exp(-DVAL * (t - (T0 + W)))


def _X(result):
    names = list(result.species_names)
    return np.asarray(result.species)[:, names.index("X")]


def _ceil_factorial_s1_oracle(times):
    t1 = 25.0 / 96.0
    t2 = t1 + 25.0 / 24.0
    out = []
    for t in times:
        if t <= t1:
            out.append(1.0 - (24.0 / 25.0) * t)
        elif t <= t2:
            out.append(0.75 - (6.0 / 25.0) * (t - t1))
        else:
            out.append(0.5 - (2.0 / 25.0) * (t - t2))
    return np.asarray(out)


def test_loader_extracts_two_triggers_for_one_pulse():
    # `geq(time, 0.7)` and `leq(time, 0.75)` are two distinct time thresholds.
    model = bngsim.Model.from_sbml_string(SBML_PULSE)
    assert model._core.n_discontinuity_triggers == 2


def test_no_piecewise_registers_zero_triggers():
    model = bngsim.Model.from_sbml_string(SBML_NO_PIECEWISE)
    assert model._core.n_discontinuity_triggers == 0


@pytest.mark.parametrize(
    ("sbml", "selectors"),
    [
        (SBML_STATE_CEIL_REACTION, ["[S1]", "[S2]"]),
        (SBML_STATE_CEIL_RATE_RULE, ["S1", "S2"]),
        (SBML_STATE_CEIL_FUNCDEF, ["[S1]", "[S2]"]),
    ],
    ids=["reaction", "rate_rule", "funcdef_reaction"],
)
def test_state_dependent_ceiling_roots_match_piecewise_oracle(sbml, selectors):
    model = bngsim.Model.from_sbml_string(sbml)
    assert model._core.n_discontinuity_triggers == 4

    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0.0, 2.0), n_points=51, rtol=1e-8, atol=1e-12, timeout=30)
    actual = np.asarray(result.as_roadrunner(selectors))
    expected_s1 = _ceil_factorial_s1_oracle(np.asarray(result.time))

    np.testing.assert_allclose(actual[:, 0], expected_s1, rtol=0.0, atol=5e-5)
    np.testing.assert_allclose(actual[:, 1], 1.0 - expected_s1, rtol=0.0, atol=5e-5)


def test_narrow_pulse_delivered_on_coarse_grid_between_samples():
    """The pulse window [0.7, 0.75] falls between every coarse sample point;
    the fix must still deliver it (X jumps then decays), matching closed form."""
    model = bngsim.Model.from_sbml_string(SBML_PULSE)
    sim = bngsim.Simulator(model, method="ode")
    # Sample times straddle but never enter the pulse window.
    sample_times = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    r = sim.run(sample_times=sample_times, rtol=1e-10, atol=1e-12, timeout=30)
    X = _X(r)
    t = np.asarray(r.time)

    # Pulse was delivered, not stepped over: X is clearly nonzero post-pulse.
    assert X[t.tolist().index(1.0)] > 1.0

    for tv in (1.0, 2.0, 3.0, 5.0):
        got = X[t.tolist().index(tv)]
        exp = _pulse_oracle(tv)
        assert got == pytest.approx(exp, rel=2e-3), f"t={tv}: {got} vs {exp}"


def test_pulse_delivered_on_fine_grid_too():
    """A fine output grid (samples landing inside the pulse window) must also
    deliver the pulse — i.e. the fix isn't a coarse-grid-only coincidence. The
    coarse grid is the exact reference (it matches the closed form to <0.5%);
    the fine grid delivers within ~1.5% (a small CV_NORMAL boundary effect when
    dense samples coincide with the inclusive pulse edges)."""
    model = bngsim.Model.from_sbml_string(SBML_PULSE)
    sim = bngsim.Simulator(model, method="ode")
    fine = sim.run(t_span=(0.0, 5.0), n_points=5001, rtol=1e-10, atol=1e-12, timeout=30)
    Xf, tf = _X(fine), np.asarray(fine.time)
    for tv in (1.0, 2.0, 5.0):
        f = Xf[int(np.argmin(np.abs(tf - tv)))]
        assert f > 0.0  # delivered, not stepped over
        assert f == pytest.approx(_pulse_oracle(tv), rel=1.5e-2), f"t={tv}: {f}"


# ── The real model that surfaced the bug ────────────────────────────────────
_BIOMD879 = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "parity_checks",
    "rr_parity",
    "models",
    "BIOMD0000000879",
    "Rodrigues2019.xml",
)


@pytest.mark.skipif(not os.path.exists(_BIOMD879), reason="BIOMD879 SBML not present")
def test_biomd879_reaches_immune_controlled_branch():
    """The 7 chemo infusions (0.125-wide, at t=0,21,…,126) must all be
    delivered: the cancer N is driven down, NOT left to escape to carrying
    capacity (k=1e12). Pre-fix N(2000)≈9.6e11; correct is the controlled branch
    (≪1, agreeing with the segmented SciPy oracle and RoadRunner-on-a-fine-grid)."""
    model = bngsim.Model.from_sbml(_BIOMD879)
    # 1 off-edge for the t=0 pulse + 2 edges each for the 6 later pulses = 13.
    assert model._core.n_discontinuity_triggers == 13

    sim = bngsim.Simulator(model, method="ode")
    res = sim.run(t_span=(0.0, 2000.0), n_points=1001, rtol=1e-10, atol=1e-16, timeout=120)
    t = np.asarray(res.time)
    N = np.asarray(res.species)[:, list(res.species_names).index("N")]

    def N_at(tv):
        return N[int(np.argmin(np.abs(t - tv)))]

    # Controlled branch, emphatically NOT tumor escape (pre-fix N(2000)≈9.6e11,
    # max →9.6e11 ≈ carrying capacity k=1e12).
    assert N_at(2000.0) < 1e3
    assert N.max() < 2.1e10  # never climbs above the t=0 value of 2e10

    # During treatment the trajectory must track the segmented SciPy oracle
    # (N(42)≈5.18e9, all 7 infusions delivered), not the pre-fix runaway.
    assert N_at(42.0) == pytest.approx(5.18e9, rel=3e-2)
    assert N_at(126.0) == pytest.approx(3.55e8, rel=5e-2)
