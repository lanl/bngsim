"""GH #229 — simulate algebraic-only SBML models (no ODE state).

A model whose only dynamics are assignment rules / functions of the SBML ``time``
csymbol (plus constants) has no species to integrate, yet still defines a
trajectory over the requested grid — RoadRunner integrates these and the SBML
semantic suite grades them against an analytical reference. bngsim used to refuse
them outright (``Cannot simulate: model has no species``); it now evaluates the
observables + functions once per output row, with no CVODE state.

Also covers the n-ary relational fix the no-species path surfaced: MathML allows
3+ argument relationals (``gt(2,1,2)`` ≡ ``(2>1) and (1>2)``), and the ExprTk
translator previously emitted only the first binary pair — a latent bug that
stayed invisible while these targets were read as t=0-folded constants.
"""

import bngsim
import numpy as np
import pytest

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
)


def _model(body: str) -> str:
    return f"{_HDR}\n{body}\n</sbml>"


# p2 := 1 + time, the canonical no-species assignment-rule-of-time model (suite
# case 01317). The target is a bngsim function, exposed via result.expressions.
ASSIGN_RULE_OF_TIME = _model(
    f"""  <model id="m_ar_time">
    <listOfParameters>
      <parameter id="p1" value="5" constant="true"/>
      <parameter id="p2" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="p2">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><cn type="integer">1</cn>{_TIME}</apply>
        </math>
      </assignmentRule>
    </listOfRules>
  </model>"""
)

# x := piecewise(10, gt(2,1,2), 20). gt(2,1,2) ≡ (2>1) and (1>2) ≡ false, so the
# otherwise-branch wins ⇒ x = 20. A binary-only translation drops the third arg,
# reads gt as (2>1) ≡ true, and would wrongly yield 10.
NARY_RELATIONAL = _model(
    """  <model id="m_nary">
    <listOfParameters><parameter id="x" constant="false"/></listOfParameters>
    <listOfRules>
      <assignmentRule variable="x">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece>
              <cn type="integer">10</cn>
              <apply><gt/><cn type="integer">2</cn><cn type="integer">1</cn><cn type="integer">2</cn></apply>
            </piece>
            <otherwise><cn type="integer">20</cn></otherwise>
          </piecewise>
        </math>
      </assignmentRule>
    </listOfRules>
  </model>"""
)


def test_no_species_assignment_rule_of_time():
    model = bngsim.Model.from_sbml_string(ASSIGN_RULE_OF_TIME)
    assert model.n_species == 0
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=11)
    assert "p2" in result.expression_names
    t = np.asarray(result.time)
    np.testing.assert_allclose(np.asarray(result.expressions["p2"]), 1.0 + t, atol=1e-9)


def test_no_species_nary_relational():
    model = bngsim.Model.from_sbml_string(NARY_RELATIONAL)
    assert model.n_species == 0
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)
    # All output rows must be the otherwise-branch (20), not the binary-only 10.
    np.testing.assert_allclose(np.asarray(result.expressions["x"]), 20.0, atol=1e-12)


def test_no_species_constant_only_runs():
    # A pure-constant no-species model still simulates (empty trajectory over the
    # grid) instead of raising — the constant is read back from the parameter.
    sbml = _model(
        """  <model id="m_const">
    <listOfParameters><parameter id="k" value="3.5" constant="true"/></listOfParameters>
  </model>"""
    )
    model = bngsim.Model.from_sbml_string(sbml)
    assert model.n_species == 0
    result = bngsim.Simulator(model, method="ode").run(t_span=(0, 2), n_points=5)
    assert result.n_times == 5
    assert model.get_param("k") == pytest.approx(3.5)
