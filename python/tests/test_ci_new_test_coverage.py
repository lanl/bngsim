"""ci/new_test_coverage.py warns about a new test file no check runs (issue #1764).

A test file a pull request adds is on no specialized workflow's list (mir.yml,
the Windows legs) unless someone edits it. If the file only does real work
under one of those builds, it skips on every python-tests leg, runs nowhere
else, and every check is green. #1359's MIR check was one, until review moved
it into a file mir.yml runs.

What is pinned here:

* the workflow lists the script reads agree with ``_ci_workflow``'s parser,
  the one the CI-coverage tests use, on every workflow in the checkout;
* a pytest command's selectors at their edges: node ids, globs, the whole
  suite, and ``--deselect``/``--ignore`` arguments, which run nothing;
* "ran nothing" across legs, on a real pytest run's JUnit report;
* which added files are warned about, and what the warning says;
* the added files read from git, a rename counted as new;
* the job in python-tests.yml that runs it.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ci_workflow import (  # noqa: E402
    REPO_ROOT,
    WORKFLOWS,
    expand,
    pytest_selectors,
    strip_comments,
)

SCRIPT = REPO_ROOT / "ci" / "new_test_coverage.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="ci/ not in this checkout")

if SCRIPT.is_file():
    sys.path.insert(0, str(SCRIPT.parent))
    import new_test_coverage as ntc  # noqa: E402
else:  # pragma: no cover
    ntc = None


def _c(collected: int, skipped: int = 0) -> dict[str, int]:
    return {"collected": collected, "skipped": skipped}


# --- the workflows' lists ---------------------------------------------------


def _workflows_with_pytest() -> list[Path]:
    return [
        wf
        for wf in sorted(WORKFLOWS.glob("*.yml"))
        if wf.name != "python-tests.yml" and "python -m pytest" in strip_comments(wf.read_text())
    ]


def test_it_reads_the_six_workflows_issue_1764_names():
    lists = ntc.workflow_lists()
    listed = {name for name, (sel, _) in lists.items() if sel}
    whole = {name for name, (_, w) in lists.items() if w}
    assert {"mir.yml", "windows-klu.yml", "windows-nfsim.yml", "windows-tail.yml"} <= listed
    assert {"forced-jit.yml", "sanitizers.yml"} <= whole
    assert "python-tests.yml" not in lists


@pytest.mark.parametrize("wf", _workflows_with_pytest(), ids=lambda p: p.name)
def test_its_lists_agree_with_the_coverage_tests_parser(wf):
    # _ci_workflow reads a workflow's first pytest command; every file it finds
    # there must be one this script finds (which reads them all).
    body = strip_comments(wf.read_text())
    theirs = {f"python/tests/{f}" for f in expand(pytest_selectors(body))}
    sel, _ = ntc.workflow_lists().get(wf.name, ([], False))
    tests = [f"python/tests/{p.name}" for p in (REPO_ROOT / "python" / "tests").glob("test_*.py")]
    mine = {t for t in tests if any(fnmatch.fnmatchcase(t, g) for g in sel)}
    assert theirs <= mine


def test_selectors_at_their_edges():
    tokens = [
        "python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "python/tests/test_a.py",
        "python/tests/test_b.py::test_one",
        "'python/tests/test_g*.py'",
        "--deselect",
        "python/tests/test_d.py::test_x",
        "--deselect=python/tests/test_e.py",
        "--ignore",
        "python/tests/test_f.py",
        "-q",
    ]
    sel, whole = ntc.selectors(tokens)
    assert sel == ["python/tests/test_a.py", "python/tests/test_b.py", "python/tests/test_g*.py"]
    assert not whole
    for suite in ("python/tests", "python/tests/"):
        assert ntc.selectors(["python", "-m", "pytest", suite, "-n", "auto"]) == ([], True)


def test_a_command_spans_its_continuation_lines_and_not_its_comments():
    text = """
jobs:
  j:
    steps:
      - run: |
          # python -m pytest python/tests/test_commented.py  (an old command, kept as a note)
          python -m pytest -p no:cacheprovider \\
            python/tests/test_one.py \\
            python/tests/test_two.py
          echo python/tests/test_echoed.py
"""
    (tokens,) = ntc.pytest_commands(text)
    assert ntc.selectors(tokens)[0] == ["python/tests/test_one.py", "python/tests/test_two.py"]


# --- the legs' reports ------------------------------------------------------


@pytest.fixture(scope="module")
def real_reports(tmp_path_factory):
    """Two legs' shard artifacts from a real pytest run: a file that skips at
    import, one whose tests all skip, and one with a test that runs."""
    root = tmp_path_factory.mktemp("repo")
    tests = root / "python" / "tests"
    tests.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tests / "test_import_skip.py").write_text(
        'import pytest\npytest.importorskip("no_such_module_for_1764")\n\n'
        "def test_a():\n    pass\n"
    )
    (tests / "test_all_skip.py").write_text(
        "import pytest\n\n@pytest.mark.skip(reason='needs MIR')\n"
        "@pytest.mark.parametrize('x', [1, 2])\ndef test_a(x):\n    pass\n"
    )
    (tests / "test_runs.py").write_text("def test_a():\n    pass\n")
    reports = root / "logs"
    for leg in ("ubuntu-latest", "macos-15"):
        d = reports / f"python-tests-log-{leg}-1"
        d.mkdir(parents=True)
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "python/tests",
                f"--junitxml={d / 'r.xml'}",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stdout + done.stderr
    return root, reports


def test_counts_are_read_per_leg(real_reports):
    root, reports = real_reports
    legs = ntc.counts_per_leg(reports, root)
    assert sorted(legs) == ["macos-15", "ubuntu-latest"]
    assert legs["ubuntu-latest"]["python/tests/test_all_skip.py"] == _c(2, 2)
    assert legs["ubuntu-latest"]["python/tests/test_import_skip.py"] == _c(1, 1)


def test_ran_nothing_on_a_real_report(real_reports):
    root, reports = real_reports
    legs = ntc.counts_per_leg(reports, root)
    assert ntc.ran_nothing("python/tests/test_all_skip.py", legs)
    assert ntc.ran_nothing("python/tests/test_import_skip.py", legs)
    assert ntc.ran_nothing("python/tests/test_never_collected.py", legs)
    assert not ntc.ran_nothing("python/tests/test_runs.py", legs)


def test_a_legs_shards_are_summed(tmp_path):
    # conftest puts each file in one shard, so a leg's counts are its shards' sum.
    tests = tmp_path / "python" / "tests"
    tests.mkdir(parents=True)
    for name in ("test_one", "test_two"):
        (tests / f"{name}.py").write_text("")
    for shard, name in ((1, "test_one"), (2, "test_two")):
        d = tmp_path / "logs" / f"python-tests-log-ubuntu-latest-{shard}"
        d.mkdir(parents=True)
        (d / "r.xml").write_text(
            f'<testsuites><testsuite><testcase classname="python.tests.{name}" name="t"/>'
            "</testsuite></testsuites>"
        )
    legs = ntc.counts_per_leg(tmp_path / "logs", tmp_path)
    assert legs == {
        "ubuntu-latest": {"python/tests/test_one.py": _c(1), "python/tests/test_two.py": _c(1)}
    }


def test_one_leg_that_runs_a_test_is_enough():
    legs = {"a": {"python/tests/test_x.py": _c(3, 3)}, "b": {"python/tests/test_x.py": _c(3, 2)}}
    assert not ntc.ran_nothing("python/tests/test_x.py", legs)


# --- the warning ------------------------------------------------------------

LISTS = {
    "mir.yml": (["python/tests/test_mir_equivalence.py"], False),
    "windows-nfsim.yml": (["python/tests/test_nfsim*.py"], False),
    "forced-jit.yml": ([], True),
}
LEGS = {
    "ubuntu-latest": {
        "python/tests/test_mir_new.py": _c(4, 4),
        "python/tests/test_nfsim_new.py": _c(2, 2),
        "python/tests/test_mir_equivalence.py": _c(9, 9),
        "python/tests/test_partly.py": _c(5, 3),
    },
    "macos-15": {
        "python/tests/test_mir_new.py": _c(4, 4),
        "python/tests/test_nfsim_new.py": _c(2, 2),
        "python/tests/test_mir_equivalence.py": _c(9, 9),
        "python/tests/test_partly.py": _c(5, 5),
    },
}


def test_a_new_file_that_ran_nowhere_is_warned_about():
    # #1359's shape: a MIR check in a new file mir.yml does not list.
    ((path, msg),) = ntc.warnings_for(["python/tests/test_mir_new.py"], LEGS, LISTS)
    assert path == "python/tests/test_mir_new.py"
    assert "all 4 of its tests skip on every python-tests leg (macos-15, ubuntu-latest)" in msg
    assert "mir.yml" in msg and "windows-nfsim.yml" in msg
    assert "forced-jit.yml run the whole suite, but not on this pull request" in msg


@pytest.mark.parametrize(
    "path",
    [
        "python/tests/test_nfsim_new.py",  # a list's glob names it
        "python/tests/test_mir_equivalence.py",  # a list names it as written
        "python/tests/test_partly.py",  # partly skipped: out of scope
    ],
)
def test_no_warning_where_a_list_names_it_or_some_test_ran(path):
    assert ntc.warnings_for([path], LEGS, LISTS) == []


def test_one_test_reads_as_one():
    legs = {"a": {"python/tests/test_one.py": _c(1, 1)}}
    ((_, msg),) = ntc.warnings_for(["python/tests/test_one.py"], legs, LISTS)
    assert "its one test skips on every python-tests leg (a)" in msg


def test_a_new_file_no_leg_collected_is_warned_about_as_such():
    ((_, msg),) = ntc.warnings_for(["python/tests/test_lost.py"], LEGS, LISTS)
    assert "it is not collected on every python-tests leg" in msg


def test_without_a_nightly_whole_suite_run_the_message_says_nothing_of_one():
    lists = {k: v for k, v in LISTS.items() if not v[1]}
    ((_, msg),) = ntc.warnings_for(["python/tests/test_mir_new.py"], LEGS, lists)
    assert "whole suite" not in msg


# --- the added files, and the script end to end -----------------------------


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=60
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "ci@example.invalid")
    _git(tmp_path, "config", "user.name", "ci")
    tests = tmp_path / "python" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_old.py").write_text("def test_a():\n    pass\n")
    (tests / "test_moved.py").write_text("def test_b():\n    pass\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tests / "test_new.py").write_text("def test_c():\n    pass\n")
    (tests / "helper.py").write_text("")
    (tests / "test_old.py").write_text("def test_a():\n    assert True\n")
    _git(tmp_path, "mv", "python/tests/test_moved.py", "python/tests/test_renamed.py")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "head")
    return tmp_path, base


def test_added_files_are_new_test_files_and_renames(repo):
    root, base = repo
    assert ntc.added_test_files(base, "HEAD", cwd=root) == [
        "python/tests/test_new.py",
        "python/tests/test_renamed.py",
    ]


def test_main_prints_a_warning_annotation(repo, tmp_path, monkeypatch, capsys):
    root, base = repo
    monkeypatch.setattr(ntc, "added_test_files", lambda b, h: ["python/tests/test_mir_new.py"])
    monkeypatch.setattr(ntc, "counts_per_leg", lambda reports: LEGS)
    monkeypatch.setattr(ntc, "workflow_lists", lambda: LISTS)
    assert ntc.main(["--reports", str(tmp_path), "--base", base]) == 0
    out = capsys.readouterr().out
    assert "::warning file=python/tests/test_mir_new.py::" in out
    assert "1 test file(s) added; 1 that no check on this change runs" in out


def test_main_with_no_reports_warns_and_passes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ntc, "added_test_files", lambda b, h: [])
    assert ntc.main(["--reports", str(tmp_path), "--base", "HEAD"]) == 0
    assert "no python-tests reports" in capsys.readouterr().out


# --- the wiring -------------------------------------------------------------


def test_python_tests_runs_it_after_every_leg_on_a_pull_request():
    body = strip_comments((WORKFLOWS / "python-tests.yml").read_text())
    job = body[body.index("  new-test-coverage:") :]
    assert "needs: python-tests" in job
    assert "github.event_name == 'pull_request'" in job
    assert "fetch-depth: 2" in job
    assert "pattern: python-tests-log-*" in job
    assert "python3 ci/new_test_coverage.py --reports logs --base HEAD^1 --head HEAD" in job
    # The shard artifacts it downloads carry the JUnit reports it reads.
    assert "python-tests-${{ matrix.leg.label }}-${{ matrix.shard }}.xml" in body
