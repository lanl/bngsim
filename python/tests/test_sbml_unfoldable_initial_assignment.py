"""GH #732: an initialAssignment the load-time fold cannot evaluate is not dropped.

The fold (`_eval_ast_numeric`) returns None for an IEEE special (1/0, ln(0),
0/0, an overflow, sqrt(-1)) and for a construct it does not know (a distrib
draw). The two ExprTk lifts that could have rescued it skip an IA that names
nothing or reads `time`, and swallowed the translator's refusal, so the IA was
never applied and its target silently kept its declared value.

Now a failed fold is evaluated by the engine at the initial time, with IEEE
semantics (what SBML suite case 00950 expects), and an IA the translator
refuses raises ModelError, as the same construct does in any other rule.
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


def _doc(params: str, ias: str, species: str = "", distrib: bool = False) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core"{_DISTRIB_NS if distrib else ""}
 level="3" version="2">
<model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>{species}</listOfSpecies>
<listOfParameters>{params}</listOfParameters>
<listOfInitialAssignments>{ias}</listOfInitialAssignments>
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
