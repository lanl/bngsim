"""The SBML Test Suite harness scores a case whose event assigns a parameter or
resizes a compartment.

bngsim promotes a symbol an event assigns to integrator state, so it appears in
``Model.species_names``, but a Result does not report it as a species column
(GH #71); it reports a same-named observable instead (GH #202). The harness's
own grading indexed the Result's species block with the model's list, so every
such case (79 of the semantic suite) died with an IndexError and a full run
stopped at the first one; it also converted amounts with the compartment's t=0
volume, so after an event resized a compartment it reported the concentration
as the amount. It now grades through the shared kernel of
``benchmarks/suites/sbml_test_suite`` (GH #225), which handles both.

Each test writes one synthetic case in the suite's file layout and scores it
with ``run_single_case``, the function the harness runs per case.

A case the harness cannot score at all (its files do not parse, or grading
raises) is ``harness_error`` with its traceback (GH #700). Before, the first one
raised out of ``main()`` and the run ended with no results written, and the
kernel recorded a grading exception as a ``sim_fail``, an engine failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("libsbml")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root()
if _ROOT is None or not (_ROOT / "benchmarks" / "suites" / "sbml_test_suite" / "run.py").is_file():
    # The harness scores through that kernel; a wheel/subtree checkout has neither.
    pytest.skip(
        "harness/sbml_test_suite and its kernel are not in this checkout",
        allow_module_level=True,
    )
_HARNESS = _ROOT / "harness" / "sbml_test_suite"

_MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'
_TIME = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)


def _sbml(event_assignments: str) -> str:
    # -> S at rate k1 (amount per time), from S = 1; an event at t = 0.75, off
    # the sample grid, so no sample sits on the discontinuity.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="C" spatialDimensions="3" size="1" constant="false"/>
  </listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialAmount="1" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters><parameter id="k1" value="1" constant="false"/></listOfParameters>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfProducts>
     <speciesReference species="S" stoichiometry="1" constant="true"/>
    </listOfProducts>
    <kineticLaw><math {_MATH}><ci>k1</ci></math></kineticLaw>
   </reaction>
  </listOfReactions>
  <listOfEvents>
   <event id="E" useValuesFromTriggerTime="true">
    <trigger initialValue="false" persistent="true">
     <math {_MATH}><apply><geq/>{_TIME}<cn>0.75</cn></apply></math>
    </trigger>
    <listOfEventAssignments>{event_assignments}</listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>
"""


def _assign(var: str, value: str) -> str:
    return (
        f'<eventAssignment variable="{var}"><math {_MATH}><cn>{value}</cn></math>'
        "</eventAssignment>"
    )


def _case(
    tmp_path: Path, sbml: str, variables: list[str], rows: list[list[float]], cid: str = "90001"
) -> Path:
    d = tmp_path / cid
    d.mkdir()
    (d / f"{cid}-sbml-l3v2.xml").write_text(sbml)
    (d / f"{cid}-model.m").write_text(
        "componentTags: Compartment, EventNoDelay, Parameter, Reaction, Species\n"
        "testTags:      Amount, NonConstantParameter\n"
        "testType:      TimeCourse\n"
    )
    (d / f"{cid}-settings.txt").write_text(
        "start: 0\nduration: 2\nsteps: 4\n"
        f"variables: {', '.join(variables)}\n"
        "absolute: 1e-6\nrelative: 1e-6\namount: S\nconcentration:\n"
    )
    lines = ["time," + ",".join(variables)]
    for t, row in zip([0.0, 0.5, 1.0, 1.5, 2.0], rows, strict=True):
        lines.append(",".join(str(v) for v in [t, *row]))
    (d / f"{cid}-results.csv").write_text("\n".join(lines) + "\n")
    return d


def _harness():
    sys.path.insert(0, str(_HARNESS))
    try:
        import run_sbml_test_suite as harness
    finally:
        sys.path.remove(str(_HARNESS))
    return harness


def _score(case_dir: Path) -> dict:
    return _harness().run_single_case(case_dir, case_dir.name)


# S amount: 1 + t up to t = 0.75, then 1.75 + 3 (t - 0.75) once k1 = 3.
_S = [1.0, 1.5, 2.5, 4.0, 5.5]


def test_a_parameter_an_event_assigns_is_scored(tmp_path):
    case = _case(
        tmp_path,
        _sbml(_assign("k1", "3")),
        ["S", "k1"],
        [[s, k] for s, k in zip(_S, [1, 1, 3, 3, 3], strict=True)],
    )
    r = _score(case)
    assert r["status"] == "pass", r


def test_an_amount_after_an_event_resizes_its_compartment_uses_the_live_volume(tmp_path):
    """C goes 1 -> 2 at the event. The amount of S is unchanged by the resize
    (its concentration halves), and the t=0 volume conversion reported the
    halved concentration as the amount."""
    case = _case(
        tmp_path,
        _sbml(_assign("k1", "3") + _assign("C", "2")),
        ["S", "k1", "C"],
        [[s, k, c] for s, k, c in zip(_S, [1, 1, 3, 3, 3], [1, 1, 2, 2, 2], strict=True)],
    )
    r = _score(case)
    assert r["status"] == "pass", r


# ── A case the harness cannot score (GH #700) ────────────────────────────────


def _good_case(tmp_path: Path, cid: str = "90001") -> Path:
    return _case(
        tmp_path,
        _sbml(_assign("k1", "3")),
        ["S", "k1"],
        [[s, k] for s, k in zip(_S, [1, 1, 3, 3, 3], strict=True)],
        cid=cid,
    )


def _unreadable_case(tmp_path: Path, cid: str = "90002") -> Path:
    """A case whose settings do not parse: ``steps`` is not a number."""
    d = _good_case(tmp_path, cid)
    settings = d / f"{cid}-settings.txt"
    settings.write_text(settings.read_text().replace("steps: 4", "steps: four"))
    return d


def test_a_case_whose_files_do_not_parse_is_a_harness_error(tmp_path):
    r = _score(_unreadable_case(tmp_path))
    assert r["status"] == "harness_error", r
    assert r["error"].startswith("ValueError:"), r
    assert "parse_settings" in r["traceback"], r


def test_a_grading_exception_is_a_harness_error_not_a_sim_fail(tmp_path, monkeypatch):
    """What escapes an adapter is the harness's grading, not the engine: each
    adapter catches its own engine's load and integrate failures."""
    kernel = _harness()._kernel()

    def grading_bug(*args):
        raise KeyError("a column the grader expected")

    monkeypatch.setitem(kernel.ADAPTERS, "bngsim", grading_bug)
    r = kernel.run_case(_good_case(tmp_path), "90001", ["bngsim"])
    assert r["bngsim_status"] == r["status"] == "harness_error", r
    assert "grading_bug" in r["bngsim_traceback"] and r["traceback"] == r["bngsim_traceback"]


def test_the_harness_run_survives_a_case_it_cannot_score(tmp_path, monkeypatch):
    """main() used to raise out of the first such case, writing nothing. 90003's
    model description is not UTF-8, which this script reads itself as well as
    through the kernel, so it reaches main()'s own per-case net."""
    harness = _harness()
    suite = tmp_path / "semantic"
    suite.mkdir()
    _good_case(suite, "90001")
    _unreadable_case(suite, "90002")
    (_good_case(suite, "90003") / "90003-model.m").write_bytes(b"testType: \xff\xfe\n")
    out = tmp_path / "out.json"
    monkeypatch.setattr(harness, "SUITE_DIR", suite)
    monkeypatch.setattr(sys, "argv", ["run_sbml_test_suite.py", "--output", str(out)])
    harness.main()
    payload = json.loads(out.read_text())
    assert {c["case"]: c["status"] for c in payload["cases"]} == {
        "90001": "pass",
        "90002": "harness_error",
        "90003": "harness_error",
    }
    assert "UnicodeDecodeError" in payload["cases"][2]["traceback"]
    assert payload["summary"]["bngsim"]["harness_error"] == 2


def test_the_suite_runner_counts_a_harness_error_against_the_denominator(tmp_path, monkeypatch):
    kernel = _harness()._kernel()
    suite = tmp_path / "semantic"
    suite.mkdir()
    _good_case(suite, "90001")
    _unreadable_case(suite, "90002")
    monkeypatch.setattr(kernel, "SUITE_DIR", suite)
    monkeypatch.setattr(kernel, "RESULTS_DIR", tmp_path / "res")
    monkeypatch.setattr(sys, "argv", ["run.py", "--mode", "correctness"])
    kernel.main()
    payload = json.loads((tmp_path / "res" / "sbml_test_suite_results.json").read_text())
    s = payload["summary"]["bngsim"]
    assert (s["tested"], s["pass"], s["harness_error"]) == (2, 1, 1), s
    assert payload["n_in_scope"] == 2
    # ... and in the per-feature view, where it is a case of that tag not passed.
    assert payload["tags"]["Compartment"] == {"total": 2, "pass": 1, "fail": 1}
