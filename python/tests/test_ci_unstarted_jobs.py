"""``ci/unstarted_jobs.py``: which forced-jit jobs GitHub never started (#1488).

GitHub sometimes gives a job no runner. The job ends cancelled having run
nothing, and the run is marked failed. forced-jit-rerun.yml re-runs such a run
once, and forced-jit.yml's Report leaves the tracking issue to the re-run. Both
ask this script whether a run failed only that way. A wrong "yes" would hide a
real failure from the tracking issue until the re-run reports it, and re-run a
real failure. A wrong "no" leaves the night without its macOS result.

The job listings below are cut down from GitHub's own listings for forced-jit's
red runs of November 2026, keeping the fields the script reads.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci" / "unstarted_jobs.py"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

if not SCRIPT.is_file():  # an installed wheel carries no ci/ directory
    pytest.skip("ci/unstarted_jobs.py lives in the source root", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bngsim_ci_unstarted_jobs", SCRIPT)
assert _spec and _spec.loader
uj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uj)

PREFIX = "Python suite"
SUITE = "Python suite · every model through MIR · "


def _ran(name, conclusion="success", job_id=1):
    """A job a runner took: it has a runner and steps, whatever its outcome."""
    steps = [{"name": "Set up job", "conclusion": "success"}]
    return {"id": job_id, "name": name, "conclusion": conclusion, "runner_id": 1000129795,
            "runner_name": "GitHub Actions 1000129795", "steps": steps}  # fmt: skip


def _no_runner(name, job_id):
    """A job GitHub gave no runner, as its listing shows one."""
    return {"id": job_id, "name": name, "conclusion": "cancelled", "runner_id": 0,
            "runner_name": "", "steps": []}  # fmt: skip


def _report(conclusion=None):
    """Report, as it lists itself while it runs (conclusion None), or after."""
    return {"id": 9, "name": "Report", "conclusion": conclusion, "runner_id": 7, "steps": [{}]}


def _run(stdin_jobs, total=None):
    listing = {"total_count": len(stdin_jobs) if total is None else total, "jobs": stdin_jobs}
    return _main(json.dumps(listing))


def _main(text):
    stdin, stdout, stderr = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = io.StringIO(text), io.StringIO(), io.StringIO()
    try:
        code = uj.main(["--prefix", PREFIX])
        return code, sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr


def test_a_leg_given_no_runner_is_named_with_its_id():
    """2026-11-12: macOS 1/2 never started, the other three passed, and Report
    was running when it asked."""
    code, out, _ = _run(
        [
            _ran(SUITE + "ubuntu-latest · 1/2", job_id=1),
            _no_runner(SUITE + "macos-15 · 1/2", 131321176639),
            _ran(SUITE + "ubuntu-latest · 2/2", job_id=2),
            _ran(SUITE + "macos-15 · 2/2", job_id=3),
            _report(),
        ]
    )
    assert code == 0
    assert out == f"131321176639\t{SUITE}macos-15 · 1/2\n"


def test_two_legs_given_no_runner_are_both_named():
    """2026-11-04: both macOS shards never started."""
    code, out, _ = _run(
        [
            _no_runner(SUITE + "macos-15 · 2/2", 126684814082),
            _no_runner(SUITE + "macos-15 · 1/2", 126684814134),
            _ran(SUITE + "ubuntu-latest · 1/2", job_id=1),
            _ran(SUITE + "ubuntu-latest · 2/2", job_id=2),
            _report("failure"),
        ]
    )
    assert code == 0
    assert [line.split("\t")[0] for line in out.splitlines()] == ["126684814082", "126684814134"]


@pytest.mark.parametrize(
    "jobs",
    [
        pytest.param(  # 2026-11-07: tests failed on ubuntu-26.04
            [_ran(SUITE + "ubuntu-latest · 2/2", "failure"), _ran(SUITE + "macos-15 · 1/2")],
            id="a-leg-that-ran-and-failed",
        ),
        pytest.param(  # a real failure the same night as a missing leg
            [
                _ran(SUITE + "ubuntu-latest · 2/2", "failure"),
                _no_runner(SUITE + "macos-15 · 1/2", 5),
            ],
            id="a-failure-beside-a-leg-with-no-runner",
        ),
        pytest.param(  # cancelled once running: a timeout, or a person, mid-suite
            [_ran(SUITE + "macos-15 · 1/2", "cancelled"), _ran(SUITE + "ubuntu-latest · 1/2")],
            id="a-leg-cancelled-while-it-ran",
        ),
        pytest.param(  # 2026-11-09: every leg passed and the run was red anyway
            [_ran(SUITE + "macos-15 · 1/2"), _ran(SUITE + "ubuntu-latest · 1/2", "skipped")],
            id="every-leg-passed-or-skipped",
        ),
        pytest.param(  # a leg still running cannot be called never-started
            [{**_no_runner(SUITE + "macos-15 · 1/2", 5), "conclusion": None}],
            id="a-leg-with-no-conclusion-yet",
        ),
    ],
)
def test_a_run_with_a_result_of_its_own_names_nothing(jobs):
    """Only a run whose every missing result is a job that never started is
    left to the re-run; anything else is reported as it stands."""
    assert _run(jobs + [_report("failure")]) == (0, "", "")


def test_jobs_outside_the_prefix_are_not_judged():
    """Report is not a suite leg: its failure says nothing about the suite, and
    a Report that never started is no reason to re-run the run."""
    assert _run([_ran(SUITE + "macos-15 · 1/2"), _report("failure")]) == (0, "", "")
    assert _run([_ran(SUITE + "macos-15 · 1/2"), _no_runner("Report", 9)]) == (0, "", "")


def test_a_partial_listing_is_refused():
    """A failed job on a page this did not read would be missed, and the run
    re-run instead of reported."""
    code, out, err = _run([_no_runner(SUITE + "macos-15 · 1/2", 5)], total=150)
    assert (code, out) == (2, "")
    assert "1 of 150" in err


@pytest.mark.parametrize(
    "text",
    ["", "<!DOCTYPE html>", "[]", '{"jobs": {}}', '{"jobs": ["x"]}', '{"message": "Not Found"}'],
)
def test_what_is_not_a_job_listing_is_refused(text):
    """An error page or an API error reads as no listing, not as an empty one:
    both callers then report the run as it stands."""
    code, out, err = _main(text)
    assert (code, out) == (2, "")
    assert err.startswith("unstarted_jobs: ")


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


@pytest.mark.skipif(not WORKFLOWS.is_dir(), reason=".github/workflows is not in this checkout")
def test_the_workflows_ask_about_the_suite_jobs_by_their_name():
    """The prefix both callers pass must be how forced-jit's suite jobs are
    named; renamed, every job would fall outside it, and no run would be re-run."""
    forced = _text("forced-jit.yml")
    rerun = _text("forced-jit-rerun.yml")
    suite_name = re.search(r"^  suite:\n    name: (.+)$", forced, re.M)
    assert suite_name and suite_name.group(1).startswith(PREFIX + " ")
    asked = r'ci/unstarted_jobs\.py --prefix "([^"]+)"'
    assert re.findall(asked, forced) == [PREFIX]
    assert re.findall(asked, rerun) == [PREFIX]


@pytest.mark.skipif(not WORKFLOWS.is_dir(), reason=".github/workflows is not in this checkout")
def test_the_rerun_follows_forced_jit_by_its_name():
    """workflow_run matches a workflow by its name: line, not its file name."""
    name = re.search(r"^name: (.+)$", _text("forced-jit.yml"), re.M)
    assert name
    assert re.search(
        rf"^    workflows: \[{re.escape(name.group(1))}\]$", _text("forced-jit-rerun.yml"), re.M
    )
