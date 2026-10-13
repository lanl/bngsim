"""The per-pull-request gate on the SBML semantic suite (``ci/sbml_suite_gate.py``,
GH #700).

The gate runs the suite and holds every case's outcome to the committed baseline
the nightly also reads. These tests drive it end to end over a synthetic suite
in the real file layout (two cases, run by the real ``run.py``), and pin the two
things a gate can silently lose: the corpus it ran (a wrong or unverifiable
checkout must refuse, not compare) and its wiring into CI.

The real suite is an external corpus and is not used here; CI runs the gate
itself on every leg of python-tests.yml.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("libsbml")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ci_workflow import strip_comments  # noqa: E402
from _source_root import bngsim_source_root  # noqa: E402

_ROOT = bngsim_source_root()
if _ROOT is None or not (_ROOT / "ci" / "sbml_suite_gate.py").is_file():
    pytest.skip("ci/sbml_suite_gate.py not in this checkout", allow_module_level=True)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("_sbml_suite_gate", _ROOT / "ci" / "sbml_suite_gate.py")
V = gate._verdicts()

_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
  </listOfCompartments>
  <listOfSpecies>
   <species id="S" compartment="C" initialAmount="1" hasOnlySubstanceUnits="false"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
 </model>
</sbml>
"""


def _case(semantic: Path, cid: str, expected: float = 1.0, steps: str = "4") -> Path:
    """One case: S stays at 1. *expected* is what the results CSV claims."""
    d = semantic / cid
    d.mkdir(parents=True)
    (d / f"{cid}-sbml-l3v2.xml").write_text(_SBML)
    (d / f"{cid}-model.m").write_text(
        "componentTags: Compartment, Species\ntestTags: Amount\ntestType: TimeCourse\n"
    )
    (d / f"{cid}-settings.txt").write_text(
        f"start: 0\nduration: 1\nsteps: {steps}\nvariables: S\n"
        "absolute: 1e-6\nrelative: 1e-6\namount: S\nconcentration:\n"
    )
    rows = "".join(f"{0.25 * i},{expected}\n" for i in range(5))
    (d / f"{cid}-results.csv").write_text("time,S\n" + rows)
    return d


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """A two-case suite at ``<tmp>/checkout/cases/semantic``, and no CI summary
    file for the gate to append to."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    semantic = tmp_path / "checkout" / "cases" / "semantic"
    _case(semantic, "90001")
    _case(semantic, "90002")
    return semantic


def _gate(suite: Path, baseline: Path, *extra: str) -> int:
    return gate.main(
        ["--suite", str(suite), "--allow-unpinned", "--baseline", str(baseline), *extra]
    )


# ── the corpus it runs ──────────────────────────────────────────────────────


def test_the_suite_may_be_named_by_its_checkout_or_its_cases_directory(suite):
    assert gate.resolve_suite(str(suite)) == suite
    assert gate.resolve_suite(str(suite.parent.parent)) == suite


def test_a_directory_with_no_cases_is_refused(tmp_path):
    """run.py pointed at such a directory runs nothing and reports nothing."""
    (tmp_path / "cases" / "semantic").mkdir(parents=True)
    for d in (tmp_path, tmp_path / "missing"):
        with pytest.raises(gate.GateError, match="no SBML Test Suite semantic cases"):
            gate.resolve_suite(str(d))


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in gate._GIT_REPO_VARS}
    cfg = [
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.excludesFile=/dev/null",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
    ]
    return subprocess.run(
        ["git", *cfg, "-C", str(cwd), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_the_pin_is_checked_against_the_checkout(suite):
    checkout = suite.parent.parent
    with pytest.raises(gate.GateError, match="not a git checkout"):
        gate.check_pin(suite, "0" * 40)
    _git(checkout, "init", "-q", "--template=")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "suite")
    head = _git(checkout, "rev-parse", "HEAD")
    assert gate.check_pin(suite, head) == head
    with pytest.raises(gate.GateError, match="but the pin is"):
        gate.check_pin(suite, "0" * 40)


def test_a_copy_inside_another_repository_is_not_taken_for_a_checkout(suite):
    """``git -C`` from a plain directory answers for the repository above it."""
    outer = suite.parent.parent.parent
    _git(outer, "init", "-q", "--template=")
    _git(outer, "add", "-A")
    _git(outer, "commit", "-q", "-m", "outer")
    with pytest.raises(gate.GateError, match="not a git checkout"):
        gate.check_pin(suite, _git(outer, "rev-parse", "HEAD"))


# ── what it decides ─────────────────────────────────────────────────────────


def test_it_records_a_baseline_then_holds_the_run_to_it(suite, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    assert _gate(suite, baseline) == 2  # no baseline yet
    assert _gate(suite, baseline, "--update-baseline") == 0
    recorded = V.load_verdicts(baseline)
    assert recorded["suite"] == "sbml_semantic" and recorded["kind"] == "sbml-suite"
    assert recorded["tally"] == {"pass": 2}
    assert _gate(suite, baseline) == 0

    # The suite's expected value moves, so bngsim's unchanged answer is wrong.
    (suite / "90002" / "90002-results.csv").write_text(
        "time,S\n" + "".join(f"{0.25 * i},2.0\n" for i in range(5))
    )
    capsys.readouterr()
    fresh = tmp_path / "fresh.json"
    assert _gate(suite, baseline, "--fresh-out", str(fresh)) == 1
    out = capsys.readouterr().out
    assert "Regressed: good in the baseline, not now: 1" in out
    assert "| 90002 | pass | value_mismatch |" in out
    assert V.load_verdicts(fresh)["cases"]["90002"]["outcome"] == "value_mismatch"


def test_an_unrecorded_improvement_fails_with_the_command_that_records_it(suite, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    assert _gate(suite, baseline, "--update-baseline") == 0
    stale = V.load_verdicts(baseline)
    stale["cases"]["90001"] = {"outcome": "load_fail", "subclass": None, "note": "refused"}
    stale["tally"] = V.tally(stale["cases"])
    V.dump_verdicts(stale, baseline)
    capsys.readouterr()
    assert _gate(suite, baseline) == 1
    out = capsys.readouterr().out
    assert "Improved: failing in the baseline, good now: 1" in out
    assert "ci/sbml_suite_gate.py --update-baseline" in out


def test_a_harness_error_fails_and_is_never_recorded(suite, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    assert _gate(suite, baseline, "--update-baseline") == 0
    before = baseline.read_text()
    settings = suite / "90001" / "90001-settings.txt"
    settings.write_text(settings.read_text().replace("steps: 4", "steps: four"))
    capsys.readouterr()
    assert _gate(suite, baseline) == 1
    captured = capsys.readouterr()
    assert "The harness could not score these cases: 1" in captured.out
    assert "parse_settings" in captured.err  # the traceback reaches the log
    assert _gate(suite, baseline, "--update-baseline") == 1
    assert baseline.read_text() == before


def test_a_new_case_the_harness_cannot_score_is_never_recorded(suite, tmp_path, capsys):
    """Neither into an existing baseline, where a new case is otherwise only to
    record, nor as the first one."""
    baseline = tmp_path / "baseline.json"
    assert _gate(suite, baseline, "--update-baseline") == 0
    before = baseline.read_text()
    new = _case(suite, "90003")
    (new / "90003-settings.txt").write_text("steps: four\n")
    capsys.readouterr()
    assert _gate(suite, baseline) == 1
    assert "The harness could not score these cases: 1" in capsys.readouterr().out
    assert _gate(suite, baseline, "--update-baseline") == 1
    assert baseline.read_text() == before
    first = tmp_path / "first.json"
    assert _gate(suite, first, "--update-baseline") == 1
    assert not first.exists()


def test_a_recorded_baseline_names_the_commits_it_was_recorded_at(suite, tmp_path):
    baseline = tmp_path / "baseline.json"
    assert _gate(suite, baseline, "--update-baseline") == 0
    source = V.load_verdicts(baseline)["source"]
    assert source["bngsim_commit"] == gate.bngsim_commit()
    assert source["suite_commit"] is None  # --allow-unpinned: nothing was checked


# ── the committed baseline and the CI wiring ────────────────────────────────


def test_the_committed_baseline_is_a_clean_semantic_suite_verdict_file():
    base = V.load_verdicts(gate.BASELINE)
    assert (base["suite"], base["kind"]) == ("sbml_semantic", "sbml-suite")
    assert base["tally"] == V.tally(base["cases"]) and base["n_cases"] == len(base["cases"])
    assert "harness_error" not in base["tally"]
    # Thousands of cases: a truncated or emptied file would otherwise pass.
    assert base["tally"].get("pass", 0) > 1500
    # A recorded note must not depend on where the suite was checked out.
    leaked = [c for c, r in base["cases"].items() if "cases/semantic" in r.get("note", "")]
    assert leaked == []


def _steps(workflow: str) -> list[str]:
    """The job's steps as text blocks, one per ``- name:`` item, comments gone."""
    text = strip_comments((_ROOT / ".github" / "workflows" / workflow).read_text())
    return re.split(r"\n(?=      - name: )", text)


def test_every_python_tests_leg_runs_the_gate_on_the_pinned_corpus():
    steps = _steps("python-tests.yml")
    gate_steps = [s for s in steps if "ci/sbml_suite_gate.py" in s and "run:" in s]
    assert len(gate_steps) == 1, gate_steps
    (step,) = gate_steps
    # A step-level condition on the matrix would drop it from some legs.
    assert "matrix." not in step.split("run:")[0]
    assert "--update-baseline" not in step and "--allow-unpinned" not in step
    fetch = [s for s in steps if "fetch_semantic_suite.py" in s]
    assert len(fetch) == 1 and "--sparse" in fetch[0]
    cache = [s for s in steps if "actions/cache" in s and "sbml-test-suite" in s]
    assert len(cache) == 1 and "steps.sbml_pin.outputs.commit" in cache[0]


_ALGEBRAIC = _SBML.replace(
    " </model>",
    """  <listOfParameters><parameter id="x" value="1" constant="false"/></listOfParameters>
  <listOfRules>
   <algebraicRule>
    <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><ci>x</ci><cn>1</cn></apply></math>
   </algebraicRule>
  </listOfRules>
 </model>""",
)


def test_a_recorded_refusal_reads_the_same_wherever_the_suite_is(tmp_path):
    """bngsim's load error quotes the case file's path, and the report cuts each
    message to 200 characters. With the path left in, a deeper checkout kept
    less of the message, so a baseline recorded on another machine rewrote the
    note of every refused case."""
    kernel = _load("_sbml_suite_kernel_for_gate", _ROOT / gate.RUNNER.relative_to(gate.REPO))
    errors = []
    for where in ("a", "a/much/deeper/checkout/of/the/sbml/test/suite/than/the/other"):
        case = _case(tmp_path / where / "cases" / "semantic", "90003")
        (case / "90003-sbml-l3v2.xml").write_text(_ALGEBRAIC)
        r = kernel.run_case(case, "90003", ["bngsim"])
        assert r["status"] == "load_fail", r
        errors.append(r["error"])
    assert errors[0] == errors[1]
    assert "AlgebraicRule" in errors[0] and str(tmp_path) not in errors[0]
