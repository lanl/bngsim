"""ODE correctness for state-threshold piecewise discontinuities (GH #194).

The state twin of the GH #72 time roots. A ``piecewise`` gated on a *species*
rather than on the ``time`` csymbol —
``piecewise(k_boost, X < hi and X > lo, k_base)`` in an assignment rule or a
kinetic law — matched none of the loader's three discontinuity collectors and
registered nothing, so a window that is narrow in *time* (narrow precisely
because the boosted rate is large) was stepped straight over and the trajectory
came back bit-identical to "the window was never entered".

The tell that this is a missing root and not an accuracy shortfall: the error
**does not move with rtol/atol**. Four decades of tolerance left the reference
case at 4e-3. That property is what the first test locks.

The fix runs ``_collect_relational_edge_conditions`` — the routine event
triggers already use, including its per-atom splitting — over assignment-rule /
rate-rule math and kinetic laws, for the atoms that read integrated state. The
per-atom split is the point: over one wide step the conjunction
``(X < hi) && (X > lo)`` reads false at *both* ends, so the compound condition
never changes sign, while each half does.

This test locks:

  1. the error moves with the tolerance, and the value is the closed form
     rather than "the window was never entered";
  2. both halves of a compound state window are registered, not the conjunction;
  3. a threshold inside a called function definition is found too, under the
     call site's argument binding, including a kinetic-law ``<localParameter>``
     (whose id the emitted condition has to mangle exactly as the RHS does);
  4. what is deliberately NOT rooted — a constant threshold, and a pure-time one
     that stays :func:`_collect_time_discontinuity_conditions`'s single root;
  5. the sliding and grazing shapes, where a state surface is reached but not
     transversally crossed, still integrate.
"""

import math

import bngsim
import numpy as np
import pytest

# ── The reference model ─────────────────────────────────────────────────────
# X' = -k(X)·X with k = K_BOOST while LO < X < HI and k = K_BASE otherwise.
# K_BOOST is large, so the window is crossed in ~4e-3 time units — far inside
# one adaptive step of the surrounding K_BASE decay.
X0, K_BASE, K_BOOST, T_END = 10.0, 0.2, 20.0, 6.0
LO, HI = 4.99, 5.01

_STATE_WINDOW = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="state_window">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0" constant="false"/>
      <parameter id="k_base" value="0.2" constant="true"/>
      <parameter id="k_boost" value="20.0" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="k">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece><ci>k_boost</ci>
              <apply><and/>
                <apply><lt/><ci>X</ci><cn>5.01</cn></apply>
                <apply><gt/><ci>X</ci><cn>4.99</cn></apply>
              </apply>
            </piece>
            <otherwise><ci>k_base</ci></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>X</ci></apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# Same window, but the whole piecewise lives in a function definition the
# kinetic law calls — the shape BIOMD0000000628's `MAX(a,b)` and
# BIOMD0000000660's `rDsRc(Dna,Rc)` have, where a scan of the call site alone
# sees the arguments but never the threshold. `lo`/`hi` are kinetic-law
# LOCAL parameters, so the emitted condition must carry the loader's `_lp_<rid>_`
# mangling or it names a symbol the evaluator does not have.
_STATE_WINDOW_FUNCDEF = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="state_window_funcdef">
    <listOfFunctionDefinitions>
      <functionDefinition id="gated">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>
          <bvar><ci>s</ci></bvar>
          <bvar><ci>a</ci></bvar>
          <bvar><ci>b</ci></bvar>
          <piecewise>
            <piece><ci>k_boost</ci>
              <apply><and/>
                <apply><lt/><ci>s</ci><ci>b</ci></apply>
                <apply><gt/><ci>s</ci><ci>a</ci></apply>
              </apply>
            </piece>
            <otherwise><ci>k_base</ci></otherwise>
          </piecewise>
        </lambda></math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_base" value="0.2" constant="true"/>
      <parameter id="k_boost" value="20.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/>
              <apply><ci>gated</ci><ci>X</ci><ci>lo</ci><ci>hi</ci></apply>
              <ci>X</ci>
            </apply>
          </math>
          <listOfLocalParameters>
            <localParameter id="lo" value="4.99"/>
            <localParameter id="hi" value="5.01"/>
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# A threshold neither side of which moves with the state: `k_base < k_boost` is
# decided at load and can never change sign. Rooting it would cost a root
# evaluation per step and buy nothing, so it must not be registered.
_CONSTANT_THRESHOLD = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="constant_threshold">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_base" value="0.2" constant="true"/>
      <parameter id="k_boost" value="20.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
            <piecewise>
              <piece><ci>k_boost</ci>
                <apply><lt/><ci>k_base</ci><ci>k_boost</ci></apply></piece>
              <otherwise><ci>k_base</ci></otherwise>
            </piecewise>
            <ci>X</ci>
          </apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# One `time` threshold: GH #72 registers it, and the GH #194 scan — which
# admits only atoms reading integrated state — must not add a second root for
# the same edge.
_TIME_THRESHOLD = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="time_threshold">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_base" value="0.2" constant="true"/>
      <parameter id="k_boost" value="20.0" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/>
            <piecewise>
              <piece><ci>k_boost</ci>
                <apply><lt/>
                  <csymbol encoding="text"
                    definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
                  <cn>3</cn></apply></piece>
              <otherwise><ci>k_base</ci></otherwise>
            </piecewise>
            <ci>X</ci>
          </apply></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _sliding_or_grazing(rate_math: str, x0: float) -> str:
    """X' = <rate_math>, a rate rule on a boundary species."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="surface">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="c" initialConcentration="{x0}" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
    </listOfSpecies>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">{rate_math}</math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


def _decay_above(threshold: str) -> str:
    """piecewise(-1, X > threshold, 0) — the flow stops ON the surface."""
    return f"""<piecewise>
      <piece><apply><minus/><cn>1</cn></apply>
        <apply><gt/><ci>X</ci><cn>{threshold}</cn></apply></piece>
      <otherwise><cn>0</cn></otherwise>
    </piecewise>"""


def _closed_form(t: float) -> float:
    """Exact X(t): decay at K_BASE down to HI, at K_BOOST across the window,
    then at K_BASE again."""
    t_a = math.log(X0 / HI) / K_BASE
    t_b = math.log(HI / LO) / K_BOOST
    if t <= t_a:
        return X0 * math.exp(-K_BASE * t)
    if t <= t_a + t_b:
        return HI * math.exp(-K_BOOST * (t - t_a))
    return LO * math.exp(-K_BASE * (t - t_a - t_b))


def _window_never_entered(t: float) -> float:
    """What the pre-fix integrator returned, bit for bit: a pure K_BASE decay."""
    return X0 * math.exp(-K_BASE * t)


def _x_end(sbml: str, **kw) -> float:
    model = bngsim.Model.from_sbml_string(sbml)
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, T_END), n_points=7, timeout=60, **kw
    )
    return float(np.asarray(result.species)[-1, list(result.species_names).index("X")])


def test_narrow_state_window_registers_both_halves():
    """`X < hi` and `X > lo` are two separate roots. Rooting the conjunction
    instead would register one condition that is false at both ends of a step
    straddling the window — the sign change the integrator needs is only
    visible per atom."""
    model = bngsim.Model.from_sbml_string(_STATE_WINDOW)
    assert model._core.n_discontinuity_triggers == 2


def test_narrow_state_window_matches_closed_form_and_moves_with_tolerance():
    """The window is entered, and the residual error is now discretization —
    it shrinks with the tolerance. Pre-fix both tolerances returned
    ``X0·exp(-K_BASE·T_END)`` to the last bit: 4e-3 off the closed form, and
    the *same* 4e-3 four decades apart."""
    exact = _closed_form(T_END)
    loose = _x_end(_STATE_WINDOW, rtol=1e-8, atol=1e-8)
    tight = _x_end(_STATE_WINDOW, rtol=1e-12, atol=1e-14)

    err_loose = abs(loose - exact) / exact
    err_tight = abs(tight - exact) / exact
    assert err_loose < 1e-5, f"pre-fix this was 3.97e-03; got {err_loose:.3e}"
    assert err_tight < err_loose / 10.0, (
        "the error must move with the tolerance — a fixed error four decades "
        f"apart is the missing root, not an accuracy shortfall ({err_loose:.3e} "
        f"vs {err_tight:.3e})"
    )

    # And emphatically not the pre-fix answer, which was bit-identical to
    # never having entered the window at all.
    missed = _window_never_entered(T_END)
    assert abs(tight - missed) / missed > 1e-3


def test_narrow_state_window_is_grid_independent():
    """The crossing is located by a root, so it does not matter whether an
    output sample happens to land inside the window."""
    exact = _closed_form(T_END)
    for n_points in (7, 13, 601):
        model = bngsim.Model.from_sbml_string(_STATE_WINDOW)
        result = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, T_END), n_points=n_points, rtol=1e-10, atol=1e-12, timeout=60
        )
        got = float(np.asarray(result.species)[-1, list(result.species_names).index("X")])
        assert got == pytest.approx(exact, rel=1e-6), f"n_points={n_points}"


def test_threshold_inside_a_called_function_definition_is_found():
    """The same window, written one level down: the piecewise is a function
    definition's body and the bounds are kinetic-law local parameters. The
    condition is emitted under the call site's binding, so it must both compile
    (the local-parameter ids are mangled) and deliver the same trajectory."""
    model = bngsim.Model.from_sbml_string(_STATE_WINDOW_FUNCDEF)
    assert model._core.n_discontinuity_triggers == 2

    got = _x_end(_STATE_WINDOW_FUNCDEF, rtol=1e-10, atol=1e-12)
    assert got == pytest.approx(_closed_form(T_END), rel=1e-6)


def test_constant_threshold_registers_no_root():
    """Neither side moves with the state, so the condition can never change
    sign; a root there is pure cost. The integrator path stays unchanged."""
    model = bngsim.Model.from_sbml_string(_CONSTANT_THRESHOLD)
    assert model._core.n_discontinuity_triggers == 0


def test_time_threshold_stays_one_root():
    """GH #72 already roots `time < 3`. The state scan admits only atoms
    reading integrated state, so it neither adds a second root nor drops this
    one."""
    model = bngsim.Model.from_sbml_string(_TIME_THRESHOLD)
    assert model._core.n_discontinuity_triggers == 1


@pytest.mark.parametrize(
    ("threshold", "x0", "expected_end"),
    [("0", 2.0, 0.0), ("1", 3.0, 1.0)],
    ids=["sliding", "grazing"],
)
def test_surface_reached_but_not_crossed_still_integrates(threshold, x0, expected_end):
    """A state root, unlike a time root, is not monotone: the trajectory can
    settle *onto* the surface (`X' = -1` above it, `0` on and below) rather than
    passing through. The root fires once at the arrival and the run completes —
    it does not retrigger."""
    model = bngsim.Model.from_sbml_string(_sliding_or_grazing(_decay_above(threshold), x0))
    assert model._core.n_discontinuity_triggers == 1

    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 10.0), n_points=11, rtol=1e-8, atol=1e-10, timeout=60
    )
    x = np.asarray(result.species)[:, list(result.species_names).index("X")]
    assert x[-1] == pytest.approx(expected_end, abs=1e-8)
    # The step count is what says the root fired once rather than being chased:
    # a retrigger on a surface the flow rests on chatters into the thousands.
    #
    # Recalibrated with issue #182, which made the counters cumulative across
    # the re-init the root forces. `n_steps < 20` was measuring only the handful
    # of steps taken AFTER that re-init; the same runs report 39 (`sliding`) and
    # 32 (`grazing`) whole, nearly all of it spent walking down to the surface
    # and locating the arrival, and the bound now has to cover that. Both
    # numbers are identical on ubuntu-latest and macos-14, and stay in 22-40
    # over `n_points` from 3 to 101.
    assert result.solver_stats["n_steps"] < 80
