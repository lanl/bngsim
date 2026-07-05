from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SUITE_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "suites" / "sbml_test_suite"
if str(_SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUITE_DIR))

from _grading import (  # noqa: E402
    compare_series,
    grade,
    parse_results_csv,
    read_sbml_entities,
    resolve_reaction_flux,
)

_INF = float("inf")
_NAN = float("nan")


class _Series:
    def __init__(self, values: dict[str, list[float]]):
        self.values = {k: np.asarray(v, dtype=float) for k, v in values.items()}

    def species_conc(self, sbml_id: str):
        return None

    def entity_value(self, sbml_id: str):
        return self.values.get(sbml_id)


def test_results_csv_preserves_case_sensitive_time_named_outputs(tmp_path):
    csv_path = tmp_path / "01820-results.csv"
    csv_path.write_text("Time,time,Time,TIME\n0,10,11,12\n1,10,11,12\n")

    settings = {
        "variables": ["time", "Time", "TIME"],
        "start": 0.0,
        "duration": 1.0,
        "steps": 1,
        "absolute": 1e-4,
        "relative": 1e-4,
        "amount": [],
        "concentration": [],
    }
    times, exp_data = parse_results_csv(csv_path, settings=settings)

    np.testing.assert_allclose(times, [0.0, 1.0])
    assert set(exp_data) == {"time", "Time", "TIME"}
    np.testing.assert_allclose(exp_data["time"], [10.0, 10.0])
    np.testing.assert_allclose(exp_data["Time"], [11.0, 11.0])
    np.testing.assert_allclose(exp_data["TIME"], [12.0, 12.0])

    ent = {"species": set(), "species_comp": {}, "comp_size": {}}
    series = _Series({"time": [10, 10], "Time": [11, 11], "TIME": [12, 12]})

    assert grade(settings, exp_data, series, ent)["status"] == "pass"


def test_results_csv_rejects_non_time_first_column_when_settings_provided(tmp_path):
    csv_path = tmp_path / "bad-results.csv"
    csv_path.write_text("Time,time\n10,10\n11,10\n")
    settings = {
        "start": 0.0,
        "duration": 1.0,
        "steps": 1,
    }

    with pytest.raises(ValueError, match="first column"):
        parse_results_csv(csv_path, settings=settings)


# ── GH #247 — INF / -INF / NaN sentinel grading ─────────────────────────────
#
# Expected-results cells spelled INF / -INF / NaN are exact IEEE sentinels, not
# numbers to compare within a tolerance. compare_series must grade a correct
# engine value as a match and a wrong one as a mismatch, and must keep treating
# a *finite* datum in a column merely spelled INF/NaN (01811/01813) normally.

_ATOL, _RTOL = 1e-12, 1e-8


def _cmp(actual, expected):
    return compare_series(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), _ATOL, _RTOL
    )[0]


def test_sentinel_correct_values_pass():
    assert _cmp([_INF, _INF], [_INF, _INF])
    assert _cmp([-_INF, -_INF], [-_INF, -_INF])
    assert _cmp([_NAN, _NAN], [_NAN, _NAN])


def test_sentinel_wrong_values_fail():
    assert not _cmp([1e300, 1e300], [_INF, _INF])  # finite where inf expected
    assert not _cmp([_INF], [-_INF])  # wrong sign of infinity
    assert not _cmp([0.0], [_NAN])  # finite where NaN expected
    assert not _cmp([_NAN], [_INF])  # NaN where inf expected


def test_sentinel_mixed_finite_and_nonfinite_row():
    # 01811 'd' column style: finite entries graded by tolerance, the inf entry
    # by exact match.
    assert _cmp([10.0, 10.0, _INF], [10.0, 10.0, _INF])
    assert not _cmp([10.0, 11.0, _INF], [10.0, 10.0, _INF])  # finite entry off
    assert not _cmp([10.0, 10.0, 5.0], [10.0, 10.0, _INF])  # sentinel entry off


def test_finite_comparison_unaffected():
    assert _cmp([1.0, 2.0], [1.0, 2.0])
    assert not _cmp([1.0, 2.5], [1.0, 2.0])
    assert not _cmp([1.0, _NAN], [1.0, 2.0])  # engine NaN where finite expected


def test_results_csv_parses_nonfinite_cells(tmp_path):
    # float('INF') / float('-INF') / float('NaN') must survive CSV parsing as
    # the IEEE values (cases 00950/00951 columns P/Q/R).
    csv_path = tmp_path / "00951-results.csv"
    csv_path.write_text("time,P,Q,R\n0,INF,-INF,NaN\n1,INF,-INF,NaN\n")
    settings = {"start": 0.0, "duration": 1.0, "steps": 1}
    _, exp = parse_results_csv(csv_path, settings=settings)
    assert np.all(np.isposinf(exp["P"]))
    assert np.all(np.isneginf(exp["Q"]))
    assert np.all(np.isnan(exp["R"]))


def test_grade_sentinel_columns_end_to_end():
    # Whole-grade path: an engine that reports the exact sentinels passes; one
    # that reports 0.0 for the NaN column is a value_mismatch.
    settings = {
        "variables": ["P", "Q", "R"],
        "absolute": _ATOL,
        "relative": _RTOL,
        "amount": [],
        "concentration": [],
    }
    exp = {"P": np.array([_INF, _INF]), "Q": np.array([-_INF, -_INF]), "R": np.array([_NAN, _NAN])}
    ent = {"species": set(), "species_comp": {}, "comp_size": {}}

    good = _Series({"P": [_INF, _INF], "Q": [-_INF, -_INF], "R": [_NAN, _NAN]})
    assert grade(settings, exp, good, ent)["status"] == "pass"

    bad = _Series({"P": [_INF, _INF], "Q": [-_INF, -_INF], "R": [0.0, 0.0]})
    assert grade(settings, exp, bad, ent)["status"] == "value_mismatch"


# ── GH #249 — reaction-flux outputs (comp SubmodelOutput family) ─────────────
#
# A suite output variable may be a *reaction id* whose expected value is the
# reaction rate (kinetic law), e.g. 01360's A__J2 = J1 = 5. No engine surfaces
# the rate of a reaction it does not otherwise reference, so the shared kernel
# derives it — identically for every engine — from that engine's own reported
# trajectories, and fails *closed* (var_missing, never a guess) on any law it
# cannot evaluate.

_TIMES3 = np.array([0.0, 1.0, 2.0])


def _reaction_ent(reactions, **extra):
    ent = {
        "species": set(),
        "compartments": set(),
        "parameters": set(),
        "species_comp": {},
        "comp_size": {},
        "reactions": reactions,
        "flat_species_hosu": {},
        "flat_species_comp": {},
        "flat_const_sym": {},
    }
    ent.update(extra)
    return ent


def test_reaction_flux_constant_literal():
    # J1 = 5 (01344 style): a bare literal broadcasts to the trajectory.
    ent = _reaction_ent({"J1": {"ast": ("num", 5.0), "local": {}}})
    got = resolve_reaction_flux("J1", _Series({}), ent, _TIMES3)
    np.testing.assert_allclose(got, [5.0, 5.0, 5.0])


def test_reaction_flux_references_parameter():
    # A__J2 = J1 where J1 is a parameter the engine reports (01360): read it.
    ent = _reaction_ent({"A__J2": {"ast": ("sym", "J1"), "local": {}}})
    series = _Series({"J1": [5.0, 5.0, 5.0]})
    np.testing.assert_allclose(
        resolve_reaction_flux("A__J2", series, ent, _TIMES3), [5.0, 5.0, 5.0]
    )


def test_reaction_flux_references_other_reaction():
    # A__J2 = J1 where J1 is ANOTHER reaction (01351): recurse, because the
    # engine reports nothing for the bare reaction id.
    ent = _reaction_ent(
        {
            "J1": {"ast": ("num", 5.0), "local": {}},
            "A__J2": {"ast": ("sym", "J1"), "local": {}},
        }
    )
    np.testing.assert_allclose(
        resolve_reaction_flux("A__J2", _Series({}), ent, _TIMES3), [5.0, 5.0, 5.0]
    )


def test_reaction_flux_references_constant_species_reference():
    # A__J2 = J1_sr where J1_sr is a constant <speciesReference> stoich (01387).
    ent = _reaction_ent(
        {"A__J2": {"ast": ("sym", "J1_sr"), "local": {}}},
        flat_const_sym={"J1_sr": 5.0},
    )
    np.testing.assert_allclose(
        resolve_reaction_flux("A__J2", _Series({}), ent, _TIMES3), [5.0, 5.0, 5.0]
    )


def test_reaction_flux_local_param_times_species_amount():
    # rate = k * S, k a local parameter, S a hasOnlySubstanceUnits species: the
    # kinetic law reads S as amount = conc * volume.
    ent = _reaction_ent(
        {"R": {"ast": ("call", "*", [("sym", "k"), ("sym", "S")]), "local": {"k": 2.0}}},
        flat_species_hosu={"S": True},
        flat_species_comp={"S": "C"},
        comp_size={"C": 1.0},
    )

    class _SpeciesSeries:
        def species_conc(self, sbml_id):
            return np.array([3.0, 4.0, 5.0]) if sbml_id == "S" else None

        def entity_value(self, sbml_id):
            return np.array([1.0, 1.0, 1.0]) if sbml_id == "C" else None

    np.testing.assert_allclose(
        resolve_reaction_flux("R", _SpeciesSeries(), ent, _TIMES3), [6.0, 8.0, 10.0]
    )


def test_reaction_flux_unsupported_law_fails_closed():
    ent = _reaction_ent({"R": {"ast": ("unsupported", 999), "local": {}}})
    assert resolve_reaction_flux("R", _Series({}), ent, _TIMES3) is None


def test_reaction_flux_unresolved_symbol_fails_closed():
    ent = _reaction_ent({"R": {"ast": ("sym", "ghost"), "local": {}}})
    assert resolve_reaction_flux("R", _Series({}), ent, _TIMES3) is None


def test_reaction_flux_self_reference_fails_closed():
    # A reaction whose rate references itself must not loop — grade var_missing.
    ent = _reaction_ent({"R": {"ast": ("sym", "R"), "local": {}}})
    assert resolve_reaction_flux("R", _Series({}), ent, _TIMES3) is None


def test_reaction_flux_non_reaction_returns_none():
    assert resolve_reaction_flux("nope", _Series({}), _reaction_ent({}), _TIMES3) is None


# The exact 01360 comp model (a replaced parameter used in a submodel reaction).
# Flattening yields one reaction A__J2 whose rate is the top-level parameter J1.
_CASE_01360_L3V1 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1" level="3" version="1" comp:required="true">
  <model id="case01360" name="case01360">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="1" constant="true">
        <comp:listOfReplacedElements>
          <comp:replacedElement comp:idRef="C" comp:submodelRef="A"/>
        </comp:listOfReplacedElements>
      </compartment>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S1" compartment="C" initialConcentration="3" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false">
        <comp:listOfReplacedElements>
          <comp:replacedElement comp:idRef="S1" comp:submodelRef="A"/>
        </comp:listOfReplacedElements>
      </species>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="J1" value="5" constant="true">
        <comp:listOfReplacedElements>
          <comp:replacedElement comp:idRef="J0" comp:submodelRef="A"/>
        </comp:listOfReplacedElements>
      </parameter>
    </listOfParameters>
    <comp:listOfSubmodels>
      <comp:submodel comp:id="A" comp:modelRef="sub1"/>
    </comp:listOfSubmodels>
  </model>
  <comp:listOfModelDefinitions>
    <comp:modelDefinition id="sub1" name="sub1">
      <listOfCompartments>
        <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
      </listOfCompartments>
      <listOfSpecies>
        <species id="S1" compartment="C" initialConcentration="3" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      </listOfSpecies>
      <listOfParameters>
        <parameter id="J0" value="2" constant="true"/>
      </listOfParameters>
      <listOfReactions>
        <reaction id="J2" reversible="true" fast="false">
          <listOfProducts>
            <speciesReference species="S1" stoichiometry="1" constant="true"/>
          </listOfProducts>
          <kineticLaw>
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <ci> J0 </ci>
            </math>
          </kineticLaw>
        </reaction>
      </listOfReactions>
    </comp:modelDefinition>
  </comp:listOfModelDefinitions>
</sbml>
"""


def test_gh249_case01360_reaction_flux_end_to_end(tmp_path):
    # The full path: comp-flatten the model, extract the reaction, and grade an
    # engine that reports only S1/J1/C — A__J2 must be derived from J1 and pass.
    pytest.importorskip("libsbml")
    sbml_path = tmp_path / "01360-sbml-l3v1.xml"
    sbml_path.write_text(_CASE_01360_L3V1)

    ent = read_sbml_entities(sbml_path)
    # The submodel reaction J2 flattens to A__J2 and is extracted for grading.
    assert "A__J2" in ent["reactions"]

    n = 11
    s1 = np.array([3.0 + 5.0 * t for t in range(n)])  # amount = conc·vol (vol=1)
    settings = {
        "variables": ["S1", "J1", "C", "A__J2"],
        "start": 0.0,
        "duration": 10.0,
        "steps": 10,
        "absolute": 1e-4,
        "relative": 1e-4,
        "amount": ["S1"],
        "concentration": [],
    }
    exp = {
        "S1": s1,
        "J1": np.full(n, 5.0),
        "C": np.full(n, 1.0),
        "A__J2": np.full(n, 5.0),  # the reaction rate = J1
    }

    class _Engine:
        def species_conc(self, sbml_id):
            return s1 if sbml_id == "S1" else None

        def entity_value(self, sbml_id):
            if sbml_id == "J1":
                return np.full(n, 5.0)
            if sbml_id == "C":
                return np.full(n, 1.0)
            return None  # A__J2 is NOT surfaced by the engine — it is derived

    assert grade(settings, exp, _Engine(), ent)["status"] == "pass"

    # And an engine reporting the wrong rate is a value_mismatch, not a pass.
    class _WrongEngine(_Engine):
        def entity_value(self, sbml_id):
            return np.full(n, 99.0) if sbml_id == "J1" else _Engine.entity_value(self, sbml_id)

    assert grade(settings, exp, _WrongEngine(), ent)["status"] == "value_mismatch"
