#  Copyright (c) 2026 Los Alamos National Laboratory. All Rights Reserved.
#  SPDX-License-Identifier: MIT
"""GH #239 — a reaction's ``id`` used as a rate symbol in an initialAssignment.

In SBML L3 a reaction's id is a first-class symbol whose value is that reaction's
current rate: the kinetic-law extent, in substance/time (NOT ÷V), analogous to a
species id denoting its amount. An initialAssignment that reads a reaction id must
fold to the reaction's INITIAL rate. bngsim seeds each reaction id into the numeric
context (its kinetic-law value at the initial state) and refolds it inside the IA
convergence loop, so ``p1 = J0`` and ``p1 = addone(J0)`` resolve at load, at the
initial values the other initialAssignments and rules resolve (issue #871). These
cases are initialAssignment-only; a live runtime reaction-rate binding is out of
scope (#239).
"""

from __future__ import annotations

import bngsim
import pytest

_M = "http://www.w3.org/1998/Math/MathML"


def _sbml(model_body: str) -> str:
    """Wrap a ``<model>`` body in a minimal SBML L3V2 document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
        'level="3" version="2">\n'
        f"{model_body}\n"
        "</sbml>\n"
    )


def test_reaction_id_in_initial_assignment():
    """01224: ``p1 = J0`` where J0 (``-> S1``) has kinetic law 1 ⇒ p1 folds to 1.

    The rate is the law value AS WRITTEN (substance/time), not divided by volume.
    """
    body = """
  <model id="case01224">
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>J0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer">1</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(1.0)


def test_reaction_id_through_funcdef_in_initial_assignment():
    """01233: ``p1 = addone(J0)`` with ``addone(x)=x+1`` and J0 rate 1 ⇒ p1 = 2.

    The reaction id resolves inside a function-definition argument — the funcDef
    folds its argument first, so seeding J0 into the context makes it work.
    """
    body = """
  <model id="case01233">
    <listOfFunctionDefinitions>
      <functionDefinition id="addone"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <lambda><bvar><ci>x</ci></bvar>
          <apply><plus/><ci>x</ci><cn type="integer">1</cn></apply>
        </lambda></math></functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><ci>addone</ci><ci>J0</ci></apply></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer">1</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(2.0)


def test_empty_reaction_id_in_initial_assignment():
    """01300: ``p1 = J0`` where J0 is an EMPTY reaction (no reactants/products)
    with kinetic law 3 ⇒ p1 = 3. The rate is still just the law value."""
    body = """
  <model id="case01300">
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>J0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="true">
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer">3</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(3.0)


def test_reaction_id_with_local_parameter():
    """A local parameter shadows the global context when the reaction id folds:
    ``J0`` law ``k`` with a local ``k=7`` ⇒ ``p1 = J0`` resolves to 7, exercising
    the local-parameter branch of the reaction-id seeding."""
    body = """
  <model id="m">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="c" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p1" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <ci>J0</ci></math></initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfProducts>
          <speciesReference species="S1" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci>k</ci></math>
          <listOfLocalParameters>
            <localParameter id="k" value="7"/>
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>"""
    model = bngsim.Model.from_sbml_string(_sbml(body))
    assert model.get_param("p1") == pytest.approx(7.0)


def _rate_read_by_an_ia(ias: str, params: str = "", rules: str = "") -> str:
    """p1 = J0, J0: S -> at k*S, S declared 5, k = 1."""
    return _sbml(f"""
  <model id="m">
    <listOfCompartments>
      <compartment id="c" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="5"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="false"/>
      <parameter id="p1" value="0" constant="true"/>{params}
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p1"><math xmlns="{_M}"><ci>J0</ci></math></initialAssignment>
      {ias}
    </listOfInitialAssignments>
    <listOfRules>{rules}</listOfRules>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw><math xmlns="{_M}"><apply><times/><ci>k</ci><ci>S</ci></apply></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>""")


def test_reaction_id_reads_the_resolved_initial_state():
    """The rate is taken at S's initialAssignment, 10, not its declared 5 (issue #871:
    the seed was folded once, before the initialAssignments resolved)."""
    ia = f'<initialAssignment symbol="S"><math xmlns="{_M}"><cn>10</cn></math></initialAssignment>'
    model = bngsim.Model.from_sbml_string(_rate_read_by_an_ia(ia))
    assert model._core.get_param("p1") == pytest.approx(10.0)


def test_reaction_id_reads_a_rule_resolved_at_the_initial_state():
    """k := 2*q with q = 3 by initialAssignment: J0 = 6*5 = 30."""
    ia = f'<initialAssignment symbol="q"><math xmlns="{_M}"><cn>3</cn></math></initialAssignment>'
    rule = (
        f'<assignmentRule variable="k"><math xmlns="{_M}">'
        "<apply><times/><cn>2</cn><ci>q</ci></apply></math></assignmentRule>"
    )
    q = '\n      <parameter id="q" value="1" constant="true"/>'
    model = bngsim.Model.from_sbml_string(_rate_read_by_an_ia(ia, params=q, rules=rule))
    assert model._core.get_param("p1") == pytest.approx(30.0)


# ── Issue #727: the rest of the seed's inputs ────────────────────────────────
#
# p = R, with R: S -> at its kinetic law, and Rp: -> T at p, so T(1) = p and a
# wrong p shows in the trajectory as well as in get_param. Each was folded with
# the declared state, the global context, or not at all, before #871 and #580.


def _p_reads_r(
    law: str,
    *,
    level: int = 3,
    params: str = "",
    local: dict[str, float] | None = None,
    ias: str = "",
    rules: str = "",
    s0: float = 1.0,
    p_math: str = "<ci>R</ci>",
) -> bngsim.Model:
    """p = R (declared -1), R: S -> at ``law``, Rp: -> T at p, as SBML L3v2 or L2v4.

    ``local`` is R's kinetic-law parameters, written the way each level spells them.
    """
    if level == 3:
        ns, head = "http://www.sbml.org/sbml/level3/version2/core", 'level="3" version="2"'
        sp = 'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"'
        ref = ' stoichiometry="1" constant="true"'
        lp_tag, lp_list = "localParameter", "listOfLocalParameters"
    else:
        ns, head = "http://www.sbml.org/sbml/level2/version4", 'level="2" version="4"'
        sp, ref = "", ' stoichiometry="1"'
        lp_tag, lp_list = "parameter", "listOfParameters"
    lps = "".join(f'<{lp_tag} id="{i}" value="{v}"/>' for i, v in (local or {}).items())
    locals_ = f"<{lp_list}>{lps}</{lp_list}>" if lps else ""
    # Level 2 forbids an empty listOf.
    rules_ = f"<listOfRules>{rules}</listOfRules>" if rules else ""
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="{ns}" {head}>
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="{s0}" {sp}/>
      <species id="T" compartment="c" initialConcentration="0" {sp}/>
    </listOfSpecies>
    <listOfParameters>{params}<parameter id="p" value="-1" constant="false"/></listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="p"><math xmlns="{_M}">{p_math}</math></initialAssignment>
      {ias}
    </listOfInitialAssignments>
    {rules_}
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants><speciesReference species="S"{ref}/></listOfReactants>
        <kineticLaw><math xmlns="{_M}">{law}</math>{locals_}</kineticLaw>
      </reaction>
      <reaction id="Rp" reversible="false">
        <listOfProducts><speciesReference species="T"{ref}/></listOfProducts>
        <kineticLaw><math xmlns="{_M}"><ci>p</ci></math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""
    return bngsim.Model.from_sbml_string(text)


def _p_and_t1(model: bngsim.Model) -> tuple[float, float]:
    """p as loaded, then T(1) from an ODE run."""
    p = model.get_param("p")
    result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1.0), n_points=2)
    return p, float(result.species[-1][list(result.species_names).index("T")])


_K = '<parameter id="k" value="2" constant="true"/>'
_K_TIMES_S = "<apply><times/><ci>k</ci><ci>S</ci></apply>"
# q has no value of its own; only its assignment rule, q := 2*k, gives it one.
_Q = '<parameter id="q" constant="false"/>'
_Q_RULE = (
    f'<assignmentRule variable="q"><math xmlns="{_M}">'
    "<apply><times/><cn>2</cn><ci>k</ci></apply></math></assignmentRule>"
)


def test_the_rate_reads_an_initial_assignment_to_its_rate_constant():
    """k = 5 by initialAssignment, declared 2: R = 5*1 (the seed gave 2)."""
    ia = f'<initialAssignment symbol="k"><math xmlns="{_M}"><cn>5</cn></math></initialAssignment>'
    model = _p_reads_r(_K_TIMES_S, params='<parameter id="k" value="2" constant="false"/>', ias=ia)
    assert _p_and_t1(model) == pytest.approx((5.0, 5.0))


@pytest.mark.parametrize("level", [2, 3], ids=["L2v4", "L3v2"])
def test_a_kinetic_law_local_parameter_shadows_the_global_on_either_level(level):
    """A local k = 5 beside a global k = 2: R = 5. On Level 2 the local lives
    under KineticLaw.getParameter, and the seed read only Level 3's
    getLocalParameter, so it folded R with the global, 2."""
    model = _p_reads_r(_K_TIMES_S, level=level, params=_K, local={"k": 5.0})
    assert _p_and_t1(model) == pytest.approx((5.0, 5.0))


def test_a_level_2_local_parameter_with_no_global_of_its_name():
    """kloc = 5 is only a Level 2 local, S = 2: R = 10. With no global to fall
    back on, the fold failed and the initialAssignment could not be evaluated."""
    law = "<apply><times/><ci>kloc</ci><ci>S</ci></apply>"
    model = _p_reads_r(law, level=2, local={"kloc": 5.0}, s0=2.0)
    assert _p_and_t1(model) == pytest.approx((10.0, 10.0))


def test_the_rate_reads_a_rule_target_with_no_declared_value():
    """R's law is q, and q := 2*k with k = 2: R = 4. The seed found no q, so R was
    never bound and p's initialAssignment could not be evaluated (#727 measured p
    at its declared value). In
    test_reaction_id_reads_a_rule_resolved_at_the_initial_state the rule's target
    has a declared value, so the seed did bind R there, if to a stale rate."""
    model = _p_reads_r("<ci>q</ci>", params=_K + _Q, rules=_Q_RULE)
    assert _p_and_t1(model) == pytest.approx((4.0, 4.0))


def test_a_piecewise_condition_on_a_reaction_id_takes_its_arm():
    """p = piecewise(10, R > 0, 20), with R's law the rule target q above, R = 4:
    10. #727 measured 20, the otherwise arm: the seed left R unbound, and the fold
    read the undecided condition as false. The fold now declines an undecided
    condition (test_sbml_unfoldable_initial_assignment.py pins that), so what this
    pins is R's refold."""
    cond = (
        "<piecewise><piece><cn>10</cn><apply><gt/><ci>R</ci><cn>0</cn></apply></piece>"
        "<otherwise><cn>20</cn></otherwise></piecewise>"
    )
    model = _p_reads_r("<ci>q</ci>", params=_K + _Q, rules=_Q_RULE, p_math=cond)
    assert _p_and_t1(model) == pytest.approx((10.0, 10.0))
