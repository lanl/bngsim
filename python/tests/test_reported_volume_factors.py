"""T7 — narrow reported_volume_factors() accessor.

The trajectory-output layer (`Simulator._get_volume_factors`) needs per-reported-
species V_c to convert stored concentrations back to amounts for
`Result.as_roadrunner`. It used to call `codegen_data()`, which materializes a
full Python dict of every parameter / species / observable / function, then read
one field. `NetworkModel.reported_volume_factors()` returns exactly that list
(V_c for reported species, in reported-species order) without building the dict.

These tests pin the new accessor to the old `codegen_data()["species"]` filter
it replaces — for a .net model (all 1.0) and an hOSU SBML model with a non-unit
compartment volume — so the value cannot drift. `as_roadrunner()` correctness is
covered by test_result_roadrunner.py, which exercises the same V_c through the
output adapter.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest

# Minimal single-species decay SBML, hOSU=true. V1: compartment size 1 (V_c=1).
# V2: size 2 (V_c=2) → a non-unit volume factor.
_DECAY_SBML_V1 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="decay_v1">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="cell" initialAmount="100"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="X" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>X</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

_DECAY_SBML_V2 = _DECAY_SBML_V1.replace(
    'size="1" constant="true"', 'size="2" constant="true"'
).replace('id="decay_v1"', 'id="decay_v2"')


def _vf_via_codegen_data(core) -> list[float]:
    """The old path: filter the full codegen_data() species dict."""
    return [
        float(s.get("volume_factor", 1.0))
        for s in core.codegen_data()["species"]
        if s.get("reported", True)
    ]


def test_net_volume_factors_all_unit(simple_decay_net: Path) -> None:
    """A .net model: narrow accessor matches codegen_data and is all 1.0."""
    core = bngsim.Model.from_net(simple_decay_net)._core
    vf = list(core.reported_volume_factors())
    assert vf == _vf_via_codegen_data(core)
    assert vf and all(v == 1.0 for v in vf)


@pytest.mark.parametrize("sbml", [_DECAY_SBML_V1, _DECAY_SBML_V2], ids=["V1", "V2"])
def test_sbml_volume_factors_match_codegen_data(sbml: str) -> None:
    """SBML models (incl. hOSU V_c=2): narrow accessor matches the old filter."""
    core = bngsim.Model.from_sbml_string(sbml)._core
    assert list(core.reported_volume_factors()) == _vf_via_codegen_data(core)


def test_hosu_model_has_non_unit_factor() -> None:
    """The hOSU V_c=2 model exposes a non-unit volume factor (exercises ≠1.0)."""
    core = bngsim.Model.from_sbml_string(_DECAY_SBML_V2)._core
    vf = list(core.reported_volume_factors())
    assert any(v != 1.0 for v in vf), f"expected a non-unit V_c, got {vf}"
