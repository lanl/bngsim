"""SBML ``<initialAssignment>`` over parameters: the IC's own θ-derivative.

A species whose initial condition is an expression over model parameters has a
``∂x_i(0)/∂θ`` that the forward-sensitivity seed must carry, exactly as for the
``R() R0`` parameter-named IC issue #43 covers. The SBML loader only ever
registered the trivial ``<ci>`` case, so ``u(0) = b*v0`` — AMICI's ``neuron``
fixture — produced **no seed and no warning**: the whole ``b`` column came back
short by the IC term.

Two things made it invisible:

* Nothing refused. The column was finite, smooth, and wrong.
* A trajectory finite difference agreed with it. ``set_param`` did not
  re-resolve an initialAssignment either, so the oracle held ``x(0)`` fixed in
  the same way the seed did — self-consistent, both wrong. It took a second
  engine (AMICI, on the identical document) to separate them, and the closed
  forms below are the in-repo replacement for that.

The fix lowers a compound parameter-only initialAssignment to a synthetic
*derived* parameter, because ``compute_ic_param_sens_seed`` already
differentiates a derived IC to its primaries. Registering the link also makes
``set_param`` re-resolve the IC, which is asserted here against a fresh load.
"""

import re

import bngsim
import numpy as np
import pytest

# dS/dt = -k*S with S(0) = b*v0.  S(t) = b*v0*exp(-k t), so ∂S/∂b = v0*exp(-k t)
# and ∂S/∂v0 = b*exp(-k t) — both entirely from the initial condition.
SBML_IA_PRODUCT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_product">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="u" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="b" value="0.3" constant="true"/>
      <parameter id="v0" value="-60" constant="true"/>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="u">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>b</ci><ci>v0</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="u">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>u</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


def _run(sbml, params, t_end=4.0, n=5):
    m = bngsim.Model.from_sbml_string(sbml)
    r = bngsim.Simulator(m, method="ode", sensitivity_params=params).run(
        t_span=(0, t_end), n_points=n, rtol=1e-12, atol=1e-14
    )
    return m, np.asarray(r.time), np.asarray(r.sensitivities)


class TestCompoundInitialAssignment:
    def test_product_ic_seeds_both_factors(self):
        """``u(0) = b*v0``: ∂u/∂b = v0·e^{-kt} and ∂u/∂v0 = b·e^{-kt}."""
        # Read the IC off a fresh model — `_run` leaves its own model advanced.
        fresh = bngsim.Model.from_sbml_string(SBML_IA_PRODUCT)
        assert fresh._core.get_concentration("u") == pytest.approx(-18.0)
        _m, t, s = _run(SBML_IA_PRODUCT, ["b", "v0"])
        np.testing.assert_allclose(s[:, 0, 0], -60.0 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(s[:, 0, 1], 0.3 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-8)

    def test_the_defect_was_a_silent_zero(self):
        """Pin the failure mode, so a regression is unmistakable: the seed used
        to be absent, which makes the whole column identically zero (u's only
        θ-dependence is through its initial condition)."""
        _m, _t, s = _run(SBML_IA_PRODUCT, ["b"])
        assert np.max(np.abs(s[:, 0, 0])) > 1.0

    def test_set_param_re_resolves_the_initial_condition(self):
        """Registering the link also fixes the value, not just its derivative:
        a scan over ``b`` must land where a fresh load with that ``b`` lands."""
        moved = bngsim.Model.from_sbml_string(SBML_IA_PRODUCT)
        moved.set_param("b", 0.6)
        fresh = bngsim.Model.from_sbml_string(
            SBML_IA_PRODUCT.replace('id="b" value="0.3"', 'id="b" value="0.6"')
        )
        assert moved._core.get_concentration("u") == pytest.approx(-36.0)
        assert moved._core.get_concentration("u") == pytest.approx(
            fresh._core.get_concentration("u")
        )

    def test_k_column_is_unaffected(self):
        """A parameter the IC does NOT read keeps the ordinary trajectory
        derivative: ∂u/∂k = -t·b·v0·e^{-kt}."""
        _m, t, s = _run(SBML_IA_PRODUCT, ["k"])
        np.testing.assert_allclose(
            s[:, 0, 0], -t * (0.3 * -60.0) * np.exp(-0.5 * t), rtol=1e-6, atol=1e-8
        )


# Issue #146: the same initialAssignment, on a symbol that is an SBML
# ``<parameter constant="false">`` rather than a ``<species>``. Antimony's
# pure-ODE spelling (``Virus' = ...``) emits exactly this, and it is what AMICI's
# ``nested_events`` fixture is written in. §8 of the loader promotes such a
# parameter to a species, so it IS an integrator state with an IC — but the
# initialAssignment scan keyed on the SBML species list, so the link was never
# registered and the whole ``X0`` column read zero against AMICI's 1.
SBML_IA_PROMOTED_PARAM = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ia_promoted">
    <listOfParameters>
      <parameter id="X0" value="0.1" constant="true"/>
      <parameter id="k" value="0.5" constant="true"/>
      <parameter id="X" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>X0</ci></math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci>k</ci><ci>X</ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


class TestRateRulePromotedParameter:
    """``X`` is a parameter with a rate rule, so this loader makes it a species.
    Its ``<initialAssignment>`` defines an initial condition exactly as a
    species' does and earns the same seed (issue #146)."""

    def test_the_ic_link_is_registered(self):
        m = bngsim.Model.from_sbml_string(SBML_IA_PROMOTED_PARAM)
        core = m._core
        names, params = list(core.species_names), list(core.param_names)
        assert [(names[s], params[p]) for s, p in core.species_ic_param_refs] == [("X", "X0")]

    def test_x0_column_is_the_closed_form(self):
        """X(t) = X0·e^{-kt}, so ∂X/∂X0 = e^{-kt} — the seed is the whole story."""
        _m, t, s = _run(SBML_IA_PROMOTED_PARAM, ["X0"])
        np.testing.assert_allclose(s[:, 0, 0], np.exp(-0.5 * t), rtol=1e-7, atol=1e-9)

    def test_the_defect_was_a_silent_zero(self):
        """Pin the failure mode: with no seed the column is identically zero,
        because ``X0`` reaches the trajectory only through the IC. AMICI returns
        1.0 for its peak on ``nested_events``; bngsim returned exactly 0.0."""
        _m, _t, s = _run(SBML_IA_PROMOTED_PARAM, ["X0"])
        assert np.max(np.abs(s[:, 0, 0])) == pytest.approx(1.0, rel=1e-6)

    def test_k_column_is_unaffected(self):
        _m, t, s = _run(SBML_IA_PROMOTED_PARAM, ["k"])
        np.testing.assert_allclose(s[:, 0, 0], -t * 0.1 * np.exp(-0.5 * t), rtol=1e-6, atol=1e-9)

    def test_set_param_re_resolves_the_initial_condition(self):
        """Worse than the missing seed, and fixed by the same registration:
        without the link ``set_param`` could not move ``X(0)`` either, so a plain
        parameter scan over ``X0`` ran the identical trajectory every point."""
        moved = bngsim.Model.from_sbml_string(SBML_IA_PROMOTED_PARAM)
        moved.set_param("X0", 0.4)
        fresh = bngsim.Model.from_sbml_string(
            SBML_IA_PROMOTED_PARAM.replace('id="X0" value="0.1"', 'id="X0" value="0.4"')
        )
        assert moved._core.get_concentration("X") == pytest.approx(0.4)
        assert moved._core.get_concentration("X") == pytest.approx(
            fresh._core.get_concentration("X")
        )

    def test_compound_ic_lowers_on_the_promoted_symbol_too(self):
        """The synthetic-derived-parameter branch reaches the promotion site as
        well, so ``X(0) = a*X0`` seeds both factors."""
        sbml = SBML_IA_PROMOTED_PARAM.replace(
            '<parameter id="X0" value="0.1" constant="true"/>',
            '<parameter id="X0" value="0.1" constant="true"/>\n'
            '      <parameter id="a" value="3" constant="true"/>',
        ).replace(
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>X0</ci></math>',
            '<math xmlns="http://www.w3.org/1998/Math/MathML">'
            "<apply><times/><ci>a</ci><ci>X0</ci></apply></math>",
        )
        m = bngsim.Model.from_sbml_string(sbml)
        assert m._core.get_concentration("X") == pytest.approx(0.3)
        _m, t, s = _run(sbml, ["a", "X0"])
        np.testing.assert_allclose(s[:, 0, 0], 0.1 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(s[:, 0, 1], 3.0 * np.exp(-0.5 * t), rtol=1e-7, atol=1e-9)


class TestNotLowered:
    """What must NOT be lowered. A wrong initial condition is far worse than a
    missing sensitivity seed, so the predicate is deliberately strict."""

    def test_non_constant_parameter_leaves_the_ic_alone(self):
        """``constant="false"`` parameters are promoted to species by this
        loader, so an IC reading one is not a parameter expression at all.

        BIOMD0000000856 (``WHISBF = 0.66*NSt``, ``NSt`` non-constant) is the
        case that proved it: lowering it produced a derived parameter that
        evaluated to 0 against a symbol that is a *species* in the built model,
        and the build-time IC resolution wrote that 0 over the species' real
        initial condition — moving a plain, non-sensitivity trajectory.
        """
        sbml = SBML_IA_PRODUCT.replace(
            '<parameter id="b" value="0.3" constant="true"/>',
            '<parameter id="b" value="0.3" constant="false"/>',
        )
        m = bngsim.Model.from_sbml_string(sbml)
        # The IC is still the initialAssignment's value, not 0.
        assert m._core.get_concentration("u") == pytest.approx(-18.0)
        assert not [n for n in m._core.param_names if n.startswith("_ic_")]

    def test_time_dependent_ic_is_not_lowered(self):
        sbml = SBML_IA_PRODUCT.replace(
            "<apply><times/><ci>b</ci><ci>v0</ci></apply>",
            '<apply><times/><ci>b</ci><csymbol encoding="text" '
            'definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol></apply>',
        )
        m = bngsim.Model.from_sbml_string(sbml)
        assert not [n for n in m._core.param_names if n.startswith("_ic_")]

    def test_plain_numeric_ic_is_untouched(self):
        """No initialAssignment ⇒ no synthetic parameter, and the model is
        byte-identical to what it always was."""
        sbml = re.sub(
            r"<listOfInitialAssignments>.*?</listOfInitialAssignments>",
            "",
            SBML_IA_PRODUCT,
            flags=re.S,
        )
        m = bngsim.Model.from_sbml_string(sbml)
        assert not [n for n in m._core.param_names if n.startswith("_ic_")]
        assert m._core.get_concentration("u") == pytest.approx(1.0)
