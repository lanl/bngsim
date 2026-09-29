"""GH #732: an initialAssignment the load-time fold cannot evaluate is not dropped.

The fold (`_eval_ast_numeric`) returns None for an IEEE special (1/0, ln(0),
0/0, an overflow, sqrt(-1)) and for a construct it does not know (a distrib
draw). The two ExprTk lifts that could have rescued it skip an IA that names
nothing or reads `time`, and swallowed the translator's refusal, so the IA was
never applied and its target silently kept its declared value.

Now a failed fold is evaluated by the engine at the initial time, with IEEE
semantics (what SBML suite case 00950 expects), and an IA the translator
refuses raises ModelError, as the same construct does in any other rule.

GH #871 closes what that left: a compartment or speciesReference target is
never put back on the engine, a fold that reads a failed target answered with
its declared value, and a piecewise fold read an undecidable condition as false.
"""

from __future__ import annotations

import math

import bngsim
import pytest
from bngsim import ModelError

_M = "http://www.w3.org/1998/Math/MathML"
_TIME = (
    "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>"
)
_NORMAL = (
    "<csymbol encoding='text' "
    "definitionURL='http://www.sbml.org/sbml/symbols/distrib/normal'>normal</csymbol>"
)
_DISTRIB_NS = (
    ' xmlns:distrib="http://www.sbml.org/sbml/level3/version1/distrib/version1"'
    ' distrib:required="true"'
)


def _doc(
    params: str,
    ias: str,
    species: str = "",
    distrib: bool = False,
    reactions: str = "",
    rules: str = "",
    compartment: str = '<compartment id="C" size="1" constant="true"/>',
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core"{_DISTRIB_NS if distrib else ""}
 level="3" version="2">
<model id="m">
<listOfCompartments>{compartment}</listOfCompartments>
<listOfSpecies>{species}</listOfSpecies>
<listOfParameters>{params}</listOfParameters>
<listOfInitialAssignments>{ias}</listOfInitialAssignments>
<listOfRules>{rules}</listOfRules>
<listOfReactions>{reactions}</listOfReactions>
</model></sbml>"""


def _ia(sym: str, body: str) -> str:
    return (
        f'<initialAssignment symbol="{sym}"><math xmlns="{_M}">{body}</math></initialAssignment>'
    )


def _param(pid: str, value: float) -> str:
    return f'<parameter id="{pid}" value="{value}" constant="true"/>'


def _div(a: str, b: str) -> str:
    return f"<apply><divide/>{a}{b}</apply>"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_div("<cn>1</cn>", "<cn>0</cn>"), math.inf),
        ("<apply><ln/><cn>0</cn></apply>", -math.inf),
        (_div("<cn>0</cn>", "<cn>0</cn>"), math.nan),
        ("<apply><power/><cn>10</cn><cn>400</cn></apply>", math.inf),
        ("<apply><root/><cn>-1</cn></apply>", math.nan),
    ],
)
def test_a_literal_non_finite_ia_lands_as_its_ieee_value(body, expected):
    m = bngsim.Model.from_sbml_string(_doc(_param("P", -999), _ia("P", body)))
    got = m._core.get_param("P")
    assert math.isnan(got) if math.isnan(expected) else got == expected


def test_a_time_bearing_ia_is_evaluated_at_the_initial_time():
    body = f"<apply><plus/>{_div('<cn>1</cn>', '<ci>x</ci>')}{_TIME}</apply>"
    m = bngsim.Model.from_sbml_string(_doc(_param("x", 0) + _param("P", 7), _ia("P", body)))
    assert m._core.get_param("P") == math.inf


def test_a_species_target_gets_its_ieee_value():
    species = (
        '<species id="S" compartment="C" initialConcentration="5" hasOnlySubstanceUnits="false"'
        ' boundaryCondition="false" constant="false"/>'
    )
    doc = _doc("", _ia("S", _div("<cn>1</cn>", "<cn>0</cn>")), species=species)
    assert bngsim.Model.from_sbml_string(doc)._core.get_concentration("S") == math.inf


def test_a_foldable_ia_is_unchanged():
    body = "<apply><times/><cn>2</cn><cn>3</cn></apply>"
    m = bngsim.Model.from_sbml_string(_doc(_param("P", 0), _ia("P", body)))
    assert m._core.get_param("P") == 6.0


@pytest.mark.parametrize("mean", ["<cn>10</cn>", "<ci>mu</ci>"])
def test_a_distrib_draw_is_refused_not_dropped(mean):
    body = f"<apply>{_NORMAL}{mean}<cn>1</cn></apply>"
    doc = _doc(_param("P", -1) + _param("mu", 10), _ia("P", body), distrib=True)
    with pytest.raises(ModelError, match="initialAssignment for 'P' cannot be evaluated"):
        bngsim.Model.from_sbml_string(doc)


# ── GH #871 ──────────────────────────────────────────────────────────────────

_S = (
    '<species id="S" compartment="C" initialConcentration="5" hasOnlySubstanceUnits="false"'
    ' boundaryCondition="false" constant="false"/>'
)
_INF = _div("<cn>1</cn>", "<cn>0</cn>")


def test_a_fold_that_reads_a_failed_target_is_refused():
    # Q's fold "answered" 6 from S's declared 5; S itself lands as inf.
    ias = _ia("S", _INF) + _ia("Q", "<apply><plus/><ci>S</ci><cn>1</cn></apply>")
    doc = _doc(_param("Q", 0), ias, species=_S)
    with pytest.raises(ModelError, match="for 'Q' cannot be evaluated: it reads 'S'"):
        bngsim.Model.from_sbml_string(doc)


def test_a_failed_compartment_ia_is_refused():
    with pytest.raises(ModelError, match="for 'C' cannot be evaluated"):
        bngsim.Model.from_sbml_string(_doc("", _ia("C", _INF)))


def test_a_failed_stoichiometry_ia_is_refused():
    reaction = (
        '<reaction id="R" reversible="false"><listOfReactants>'
        '<speciesReference id="S1" species="S" stoichiometry="1" constant="true"/>'
        f'</listOfReactants><kineticLaw><math xmlns="{_M}"><ci>k</ci></math></kineticLaw>'
        "</reaction>"
    )
    doc = _doc(_param("k", 0.1), _ia("S1", _INF), species=_S, reactions=reaction)
    with pytest.raises(ModelError, match="for 'S1' cannot be evaluated"):
        bngsim.Model.from_sbml_string(doc)


def test_a_piecewise_condition_the_fold_cannot_decide_goes_to_the_engine():
    # ExprTk: inf > 0 is true, so P = 1. The fold read the condition as false.
    body = (
        f"<piecewise><piece><cn>1</cn><apply><gt/>{_INF}<cn>0</cn></apply></piece>"
        "<otherwise><cn>2</cn></otherwise></piecewise>"
    )
    m = bngsim.Model.from_sbml_string(_doc(_param("P", 0), _ia("P", body)))
    assert m._core.get_param("P") == 1.0


def test_a_piecewise_decided_before_an_undecidable_condition_still_folds():
    body = (
        "<piecewise><piece><cn>3</cn><apply><gt/><cn>1</cn><cn>0</cn></apply></piece>"
        f"<piece><cn>4</cn><apply><gt/>{_INF}<cn>0</cn></apply></piece></piecewise>"
    )
    m = bngsim.Model.from_sbml_string(_doc(_param("P", 0), _ia("P", body)))
    assert m._core.get_param("P") == 3.0


def test_a_parameter_reading_a_failed_parameter_is_evaluated_not_refused():
    # Both are lifted, so the engine evaluates Q from P's inf.
    ias = _ia("P", _INF) + _ia("Q", "<apply><plus/><ci>P</ci><cn>1</cn></apply>")
    m = bngsim.Model.from_sbml_string(_doc(_param("P", 0) + _param("Q", 0), ias))
    assert m._core.get_param("Q") == math.inf


def test_a_species_reading_a_failed_species_is_evaluated_not_refused():
    s2 = _S.replace('id="S"', 'id="S2"')
    ias = _ia("S", _INF) + _ia("S2", "<apply><plus/><ci>S</ci><cn>1</cn></apply>")
    m = bngsim.Model.from_sbml_string(_doc("", ias, species=_S + s2))
    assert m._core.get_concentration("S2") == math.inf


# The same leak through the channels the name-only closure missed: a reaction id
# (bound to its kinetic law's initial value), an assignment rule whose own fold
# fails, a compartment an assignment rule sizes, and a constant stoichiometryMath.


def _ar(var: str, body: str) -> str:
    return f'<assignmentRule variable="{var}"><math xmlns="{_M}">{body}</math></assignmentRule>'


def _plus_one(sym: str) -> str:
    return f"<apply><plus/><ci>{sym}</ci><cn>1</cn></apply>"


_PW = (
    f"<piecewise><piece><cn>1</cn><apply><gt/>{_INF}<cn>0</cn></apply></piece>"
    "<otherwise><cn>2</cn></otherwise></piecewise>"
)


def test_a_fold_through_a_reaction_id_is_refused():
    # p1 = J0 folded k*S at S's declared 5.
    reaction = (
        '<reaction id="J0" reversible="false"><listOfReactants>'
        '<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>'
        f'<kineticLaw><math xmlns="{_M}"><apply><times/><ci>k</ci><ci>S</ci></apply></math>'
        "</kineticLaw></reaction>"
    )
    doc = _doc(
        _param("k", 1) + _param("p1", 0),
        _ia("S", _INF) + _ia("p1", "<ci>J0</ci>"),
        species=_S,
        reactions=reaction,
    )
    with pytest.raises(ModelError, match="for 'p1' cannot be evaluated: it depends through 'J0'"):
        bngsim.Model.from_sbml_string(doc)


def test_a_fold_through_a_rate_of_is_refused():
    # rateOf(S2) is k*S at S's declared 5; no name in it says so.
    rate_of = (
        "<apply><csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/rateOf'>"
        "rateOf</csymbol><ci>S2</ci></apply>"
    )
    reaction = (
        '<reaction id="J0" reversible="false"><listOfReactants>'
        '<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>'
        '<listOfProducts><speciesReference species="S2" stoichiometry="1" constant="true"/>'
        f'</listOfProducts><kineticLaw><math xmlns="{_M}"><apply><times/><ci>k</ci><ci>S</ci>'
        "</apply></math></kineticLaw></reaction>"
    )
    doc = _doc(
        _param("k", 1) + _param("p2", 0),
        _ia("S", _INF) + _ia("p2", rate_of),
        species=_S + _S.replace('id="S"', 'id="S2"'),
        reactions=reaction,
    )
    with pytest.raises(ModelError, match="for 'p2' cannot be evaluated: it depends on 'S'"):
        bngsim.Model.from_sbml_string(doc)


def test_a_failed_rule_is_not_read_as_its_declared_value():
    # C = Y + 1 folded 6 from Y's declared 5; the rule makes Y inf.
    y = '<parameter id="Y" value="5" constant="false"/>'
    doc = _doc(y, _ia("C", _plus_one("Y")), rules=_ar("Y", _INF))
    with pytest.raises(ModelError, match="for 'C' cannot be evaluated: it reads 'Y'"):
        bngsim.Model.from_sbml_string(doc)


def test_a_rule_the_fold_cannot_decide_is_not_read_as_its_declared_value():
    # The piecewise fold no longer answers 2, so C would have read Y's declared 7.
    y = '<parameter id="Y" value="7" constant="false"/>'
    doc = _doc(y, _ia("C", "<ci>Y</ci>"), rules=_ar("Y", _PW))
    with pytest.raises(ModelError, match="for 'C' cannot be evaluated: it reads 'Y'"):
        bngsim.Model.from_sbml_string(doc)


@pytest.mark.parametrize(
    ("rule", "ias", "params"),
    [
        (_plus_one("P"), _ia("P", _INF), _param("P", 0)),  # reads a failed fold
        (_PW, "", ""),  # its own fold cannot decide
    ],
)
def test_a_compartment_a_rule_sizes_over_a_failed_fold_is_refused(rule, ias, params):
    # The load converts X's amount by the folded size: 10 where 10/inf or 10/1.
    comp = '<compartment id="C" size="3" constant="false"/>'
    x = (
        '<species id="X" compartment="C" initialAmount="10" hasOnlySubstanceUnits="false"'
        ' boundaryCondition="false" constant="false"/>'
    )
    doc = _doc(params, ias, species=x, rules=_ar("C", rule), compartment=comp)
    with pytest.raises(ModelError, match="assignment rule for 'C' cannot be evaluated"):
        bngsim.Model.from_sbml_string(doc)


@pytest.mark.parametrize("body", [_INF, _PW])
def test_a_constant_stoichiometry_math_that_fails_is_refused(body):
    # It fell back to the speciesReference's stoichiometry, 1.
    doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="m">
<listOfCompartments><compartment id="C" size="1"/></listOfCompartments>
<listOfSpecies><species id="S" compartment="C" initialConcentration="5"/>
<species id="P" compartment="C" initialConcentration="0"/></listOfSpecies>
<listOfParameters><parameter id="k" value="1"/></listOfParameters>
<listOfReactions><reaction id="R" reversible="false">
<listOfReactants><speciesReference species="S"/></listOfReactants>
<listOfProducts><speciesReference species="P"><stoichiometryMath>
<math xmlns="{_M}">{body}</math></stoichiometryMath></speciesReference></listOfProducts>
<kineticLaw><math xmlns="{_M}"><ci>k</ci></math></kineticLaw></reaction></listOfReactions>
</model></sbml>"""
    with pytest.raises(ModelError, match="stoichiometryMath for 'P' in reaction 'R'"):
        bngsim.Model.from_sbml_string(doc)


def test_a_species_reading_a_failed_rule_is_evaluated_not_refused():
    y = '<parameter id="Y" value="5" constant="false"/>'
    m = bngsim.Model.from_sbml_string(
        _doc(y, _ia("S", _plus_one("Y")), species=_S, rules=_ar("Y", _INF))
    )
    assert m._core.get_concentration("S") == math.inf


def test_a_species_reading_an_undecidable_rule_is_evaluated_not_refused():
    y = '<parameter id="Y" value="7" constant="false"/>'
    m = bngsim.Model.from_sbml_string(
        _doc(y, _ia("S", "<ci>Y</ci>"), species=_S, rules=_ar("Y", _PW))
    )
    assert m._core.get_concentration("S") == 1.0
