"""Regression test for topological function-evaluation order (GH #76).

``evaluate_functions()`` walks ``var_param_bindings`` in sequence, writing each
function's value into its bound parameter; later functions read those
parameters. If function ``a`` references a function ``b`` that is declared
*after* it, a single declaration-order pass makes ``a`` read ``b``'s STALE bound
value (left over from the previous RHS evaluation). Because ``compute_derivs``
runs one ``evaluate_functions`` pass per RHS evaluation, the RHS becomes
path-dependent — not a pure function of ``(t, y)`` — silently corrupting
integration for any SBML model whose assignment-rule declaration order is not a
topological (dependency) order.

``ModelBuilder`` now topologically sorts ``var_param_bindings`` so one pass
converges. These tests pin that behavior with a model whose assignment rules are
declared in REVERSE dependency order:

    rule a := b        (a depends on b, declared first)
    rule b := 0.5 * S  (b depends on the species S, declared second)
    reaction S -> ,  rate = a   ⟹   dS/dt = -a = -0.5·S   ⟹   S(t) = S0·e^(-0.5t)

Without the topological sort this integrates visibly wrong (~7e-4 abs error by
t=4); with it, it matches the closed form to the integrator tolerance (~1e-7).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest


def _nontopo_sbml(rule_a_first: bool) -> str:
    rule_a = (
        '      <assignmentRule variable="a">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>b</ci></math></assignmentRule>'
    )
    rule_b = (
        '      <assignmentRule variable="b">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<apply><times/><cn>0.5</cn><ci>S</ci></apply></math></assignmentRule>"
    )
    rules = (rule_a + rule_b) if rule_a_first else (rule_b + rule_a)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="nontopo">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="a" constant="false"/>
      <parameter id="b" constant="false"/>
    </listOfParameters>
    <listOfRules>
{rules}
    </listOfRules>
    <listOfReactions>
      <reaction id="deg" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>a</ci></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _integrate_S(sbml: str) -> tuple[np.ndarray, np.ndarray]:
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(sbml), method="ode")
    r = sim.run(t_span=(0.0, 4.0), n_points=9, rtol=1e-9, atol=1e-12)
    t = np.asarray(r.time)
    s = np.asarray(r.species)[:, list(r.species_names).index("S")]
    return t, s


def test_forward_referenced_assignment_rule_matches_closed_form():
    """Assignment rule declared before its dependency must still resolve fully:
    the RHS is order-independent, so the trajectory matches S0·e^(-0.5t)."""
    t, s = _integrate_S(_nontopo_sbml(rule_a_first=True))
    exact = 10.0 * np.exp(-0.5 * t)
    err = float(np.max(np.abs(s - exact)))
    # Comfortably between the integrator tolerance (~1e-7 with the fix) and the
    # stale-read corruption (~7e-4 without it).
    assert err < 1e-5, f"non-topological RHS diverged from closed form: max err {err:g}"


def test_declaration_order_does_not_change_trajectory():
    """The two declaration orders of the same rule set must integrate
    identically — the whole point of the topological sort."""
    _, s_a_first = _integrate_S(_nontopo_sbml(rule_a_first=True))
    _, s_b_first = _integrate_S(_nontopo_sbml(rule_a_first=False))
    assert s_a_first == pytest.approx(s_b_first, abs=1e-9, rel=1e-9)
