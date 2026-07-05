"""Regression tests for issue #90: SBML loader ExprTk identifier collisions.

Two distinct symbol-naming collisions surfaced by the rr_parity ODE corpus,
both in the same family as the closed reserved-word work (#24 `t`/`time`,
#18 `const`/`true`/`false`):

  (a) **Reserved math-builtin collision** — a model symbol named after an
      ExprTk builtin function (`log`, `sin`, `exp`, …). MODEL1812040006 is a
      COPASI export with a parameter literally named ``log`` whose assignment
      rule is ``log = ln(V)``; the ``<ln/>`` builtin renders to ExprTk
      ``log(...)`` and a single flat namespace cannot hold both the variable
      ``log`` and the builtin. Fixed by extending ``_safe_name`` to rename any
      model symbol whose name is an ExprTk builtin (sourced from the core's
      ``reserved_names()["functions"]``) to ``_ant_<name>``.

  (b) **u_-prefix builtin-constant collision** — bngsim registers its ``_X``
      built-in constants (Planck ``_h``, Avogadro ``_NA``, …) under the ExprTk
      key ``u_X`` (ExprTk rejects a leading ``_``). A user parameter named
      literally ``u_h`` aliased Planck's slot and failed to register.
      BIOMD0000000950 (Chitnis2012) has a parameter ``u_h``. Fixed in
      ``expression.cpp`` by reserving the ``u_X`` keys so such a parameter
      takes the transparent ``r_<name>`` mangling path (its Python-facing name
      stays ``u_h``).
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

A0 = 10.0


def _decay_sbml(param_name: str, value: float) -> str:
    """Minimal SBML: A → ∅ with first-order rate ``<param_name> * A``.

    Oracle: A(t) = A0 · exp(-value · t).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="{param_name}" value="{value}" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>{param_name}</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


# ── (a) reserved math-builtin collision: parameter named `log` ──────────────
#
# The exact MODEL1812040006 shape: a `constant=false` parameter `log` whose
# assignment rule is `log = ln(V)` (MathML `<ln/>`), then used as a first-order
# rate constant. With V = 100, log = ln(100), so A(t) = A0 · 100^(-t).
SBML_LOG_BUILTIN = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="log_builtin">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="{A0}"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="V" value="100" constant="true"/>
      <parameter id="log" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="log">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><ln/><ci>V</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>log</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def test_log_parameter_loads_and_is_renamed():
    """A parameter named `log` must load and be renamed off the builtin."""
    model = bngsim.Model.from_sbml_string(SBML_LOG_BUILTIN)
    assert "log" not in model.param_names, "raw `log` collides with the ExprTk builtin"
    assert "_ant_log" in model.param_names
    # The assignment rule `log = ln(V)` with V = 100 must still evaluate.
    assert model.get_param("_ant_log") == pytest.approx(math.log(100.0))


def test_log_parameter_drives_correct_dynamics():
    """`log` (= ln(100)) used as a first-order rate constant: the renamed
    symbol resolves in the rate law and the builtin `log(...)` from `<ln/>`
    stays distinct. A(t) = A0 · exp(-ln(100) · t)."""
    model = bngsim.Model.from_sbml_string(SBML_LOG_BUILTIN)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=11)
    t = np.asarray(r.time)
    A = np.asarray(r.species)[:, list(r.species_names).index("A")]
    np.testing.assert_allclose(A, A0 * np.exp(-math.log(100.0) * t), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("builtin", ["exp", "sin", "cos", "sqrt", "log10"])
def test_other_math_builtins_as_parameter_names_load(builtin):
    """The fix is general, not `log`-specific: any ExprTk builtin used as a
    model symbol name loads and is renamed."""
    model = bngsim.Model.from_sbml_string(_decay_sbml(builtin, 0.5))
    assert builtin not in model.param_names
    assert f"_ant_{builtin}" in model.param_names
    assert model.get_param(f"_ant_{builtin}") == pytest.approx(0.5)


# ── (b) u_-prefix builtin-constant collision: parameter named `u_h` ─────────


def test_u_h_parameter_loads():
    """A parameter literally named `u_h` must load (it aliased Planck's `_h`
    registration key `u_h` and failed pre-fix). Its Python-facing name and
    value are preserved; the `r_u_h` mangling is internal."""
    model = bngsim.Model.from_sbml_string(_decay_sbml("u_h", 0.5))
    assert "u_h" in model.param_names
    assert model.get_param("u_h") == pytest.approx(0.5)


def test_u_h_parameter_drives_correct_dynamics():
    """`u_h` used as a first-order rate constant resolves through the internal
    mangle: A(t) = A0 · exp(-u_h · t)."""
    model = bngsim.Model.from_sbml_string(_decay_sbml("u_h", 0.5))
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 5), n_points=11)
    t = np.asarray(r.time)
    A = np.asarray(r.species)[:, list(r.species_names).index("A")]
    np.testing.assert_allclose(A, A0 * np.exp(-0.5 * t), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", ["u_pi", "u_e", "u_kB", "u_NA", "u_R", "u_h", "u_F"])
def test_all_builtin_constant_key_aliases_load(name):
    """Every `u_X` key occupied by a bngsim built-in constant must be safe to
    use as a user parameter name (the class, not just `u_h`)."""
    model = bngsim.Model.from_sbml_string(_decay_sbml(name, 0.5))
    assert name in model.param_names
    assert model.get_param(name) == pytest.approx(0.5)
