"""ci/new_test_coverage.py warns about a new test file no check runs (issue #1764).

A test file a pull request adds is on no specialized workflow's list (mir.yml,
the Windows legs) unless someone edits it. If the file only does real work
under one of those builds, it skips on every python-tests leg, runs nowhere
else, and every check is green. #1359's MIR check was one, until review moved
it into a file mir.yml runs.

What is pinned here:

* the lists the script reads equal ``_ci_workflow``'s parser's, the one the
  CI-coverage tests use, on every workflow in the checkout, both ways;
* a pytest command's selectors at their edges: node ids, globs, directories,
  the whole suite, ``--deselect``/``--ignore`` arguments, a pipe, and what a
  text read cannot know (a shell array, a matrix value, ``-k``);
* whether a workflow runs on the pull request: its ``pull_request`` trigger and
  paths filter, read in order, an alias, ``paths-ignore``;
* "ran nothing" across legs, on a real pytest run's JUnit report, and what a
  missing directory, a malformed report or a missing shard does;
* which added files are warned about, and what the warning says;
* the added files read from git, a rename counted as new;
* the job in python-tests.yml that runs it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ci_workflow import (  # noqa: E402
    REPO_ROOT,
    WORKFLOWS,
    continued_block,
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


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _wf(name="w.yml", fires=True, **kw):
    w = ntc.Workflow(name, **kw)
    w.fires = fires
    w.why_not = "" if fires else "its paths filter matches no file the pull request changes"
    return w


def _cmd(command: str, empty=()):
    w = ntc.Workflow("w.yml")
    ntc.read_command(command, w, empty)
    return w


# --- the workflows' lists ---------------------------------------------------


def _workflows_with_pytest() -> list[Path]:
    if not WORKFLOWS.is_dir():
        return []
    return [wf for wf in sorted(WORKFLOWS.glob("*.yml")) if wf.name != "python-tests.yml"]


def test_it_reads_the_six_workflows_issue_1764_names():
    wfs = ntc.read_workflows([])
    listed = {name for name, w in wfs.items() if w.selectors}
    whole = {name for name, w in wfs.items() if w.whole}
    assert {"mir.yml", "windows-klu.yml", "windows-nfsim.yml", "windows-tail.yml"} <= listed
    assert {"forced-jit.yml", "sanitizers.yml"} <= whole
    assert "python-tests.yml" not in wfs
    # On a pull request that changes nothing, none of them runs.
    assert all(w.fires is False for w in wfs.values())


@pytest.mark.parametrize("wf", _workflows_with_pytest(), ids=lambda p: p.name)
def test_its_lists_equal_the_coverage_tests_parser(wf):
    # Both ways: a file this reads that _ci_workflow does not would be named
    # without running, and would silence the warning. _ci_workflow reads a
    # workflow's first pytest command, so the first is what is compared.
    body = strip_comments(_read(wf))
    commands = ntc.pytest_commands(_read(wf))
    assert bool(commands) == ("python -m pytest" in body), "the two disagree on what is a command"
    if not commands:
        return
    mine = ntc.Workflow(wf.name)
    ntc.read_command(commands[0], mine, ntc.dispatch_env(_read(wf)))
    theirs = [t for t in pytest_selectors(body) if "::" not in t]
    # pytest_selectors keeps ``python/tests/...`` words; the suite itself is read here.
    lines = body.splitlines()
    first = next(i for i, ln in enumerate(lines) if "python -m pytest" in ln)
    words = " ".join(continued_block(lines, first)).split()
    assert mine.whole == any(w.strip("\"'").rstrip("/") == "python/tests" for w in words)
    files = [
        p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "python/tests").rglob("*.py")
    ]
    their_files = {f for f in files if any(ntc.matches(f, g) for g in theirs)}
    my_files = {f for f in files if any(ntc.matches(f, g) for g in mine.selectors)}
    assert my_files == their_files
    their_nodes = {t.split("::")[0] for t in pytest_selectors(body) if "::" in t}
    assert set(mine.node_ids) <= their_nodes  # theirs also holds --deselect's node ids


def test_selectors_at_their_edges():
    w = _cmd(
        "python -m pytest -p no:cacheprovider python/tests/test_a.py "
        "python/tests/test_b.py::test_one 'python/tests/test_g*.py' "
        "--deselect python/tests/test_d.py::test_x --deselect=python/tests/test_e.py "
        "--ignore python/tests/test_f.py --rootdir python/tests -c python/tests/x.ini "
        "python/tests/windows/ -q 2>&1 | tee python/tests/test_piped.py"
    )
    assert w.selectors == [
        "python/tests/test_a.py",
        "python/tests/test_g*.py",
        "python/tests/windows/**",
    ]
    assert w.node_ids == ["python/tests/test_b.py"]
    # A deselected node id leaves the rest of its file; a deselected file goes.
    assert w.excluded == ["python/tests/test_e.py", "python/tests/test_f.py"]
    assert not w.whole and not w.unreadable
    assert w.names("python/tests/windows/test_x.py") and not w.names("python/tests/test_piped.py")
    assert not w.names("python/tests/test_gx/test_y.py")  # a shell * stays in its directory
    for suite in ("python/tests", "python/tests/"):
        assert _cmd(f"python -m pytest {suite} -n auto").whole


@pytest.mark.parametrize(
    "arg",
    ['"${files[@]}"', "$TESTS", "${{ matrix.tests }}", "-k 'not slow'", "-m mir"],
)
def test_what_a_text_read_cannot_know_is_unreadable(arg):
    w = _cmd(f"python -m pytest python/tests/test_a.py {arg}")
    assert w.unreadable
    assert not _wf(selectors=w.selectors, unreadable=w.unreadable).runs("python/tests/test_a.py")


def test_dispatch_inputs_are_empty_on_a_pull_request():
    text = "env:\n  PYTEST_ARGS: ${{ github.event.inputs.pytest_args }}\n"
    assert ntc.dispatch_env(text) == {"PYTEST_ARGS"}
    for arg in (
        "${{ inputs.pytest_args }}",
        "${{ github.event.inputs.pytest_args }}",
        "$PYTEST_ARGS",
    ):
        w = _cmd(
            f"python -m pytest python/tests {arg} --junitxml=r-${{{{ matrix.os }}}}.xml",
            {"PYTEST_ARGS"},
        )
        assert w.whole and not w.unreadable, arg


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
    (command,) = ntc.pytest_commands(text)
    assert _cmd(command).selectors == ["python/tests/test_one.py", "python/tests/test_two.py"]


# --- does a workflow run on this pull request -------------------------------

_PATHS = """
on:
  push:
    branches: [main]
    paths: &p
      - "src/**"
      - "python/tests/test_mir*.py"
      - "!python/tests/test_mir_old.py"
      - "**/*.md"  # a trailing comment
  pull_request:
    paths: *p
jobs: {}
"""


@pytest.mark.parametrize(
    "text, changed, fires",
    [
        (_PATHS, ["src/a.cpp"], True),
        (_PATHS, ["python/tests/test_mir_new.py"], True),
        (_PATHS, ["python/tests/test_mir_old.py"], False),  # a later ! takes it out
        (_PATHS, ["python/tests/test_other.py"], False),
        (_PATHS, ["docs/x/y.md"], True),
        ("on:\n  pull_request:\n    paths-ignore:\n      - 'docs/**'\n", ["docs/a.md"], False),
        ("on:\n  pull_request:\n    paths-ignore:\n      - 'docs/**'\n", ["src/a.cpp"], True),
        ("on:\n  pull_request:\n  push:\n", ["anything"], True),
        ("on: [push, pull_request]\n", ["anything"], True),
        ("on:\n  push:\n  pull_request_target:\n", ["anything"], False),
        ("on:\n  schedule:\n    - cron: '1 1 * * *'\n", ["anything"], False),
        ("on:\n  pull_request:\n    branches: [release]\n", ["anything"], None),
        ("on:\n  pull_request:\n    types: [opened]\n", ["anything"], None),
        ("on:\n  pull_request:\n    types: [opened, synchronize]\n", ["anything"], True),
    ],
)
def test_whether_a_workflow_fires_on_the_pull_request(text, changed, fires):
    assert ntc.fires_on(text, changed)[0] is fires


def test_a_list_that_names_it_counts_only_where_the_workflow_runs(tmp_path):
    # A stale list entry, or a list glob wider than the paths filter: the list
    # names the file but the workflow does not run on this pull request.
    (tmp_path / "w.yml").write_text(
        "on:\n  pull_request:\n    paths:\n      - 'src/**'\njobs:\n  j:\n    steps:\n"
        "      - run: python -m pytest python/tests/test_mir*.py\n",
        encoding="utf-8",
    )
    new = "python/tests/test_mir_new.py"
    legs = {"a": {new: _c(1, 1)}}
    (w,) = ntc.read_workflows([new], tmp_path).values()
    assert w.names(new) and w.fires is False
    ((_, msg),) = ntc.warnings_for([new], legs, {"w.yml": w})
    assert "w.yml names it, but not on this pull request: its paths filter matches no file" in msg
    (w,) = ntc.read_workflows([new, "src/a.cpp"], tmp_path).values()
    assert ntc.warnings_for([new], legs, {"w.yml": w}) == []


def test_forced_jit_runs_a_new_file_when_the_pull_request_edits_it():
    new = "python/tests/test_brand_new_1764.py"
    wfs = ntc.read_workflows([".github/workflows/forced-jit.yml", new])
    assert wfs["forced-jit.yml"].runs(new)
    assert ntc.warnings_for([new], {"a": {new: _c(2, 2)}}, wfs) == []
    # sanitizers.yml's command passes a shell array, so it never counts.
    wfs = ntc.read_workflows([".github/workflows/sanitizers.yml", new])
    assert wfs["sanitizers.yml"].fires and not wfs["sanitizers.yml"].runs(new)
    ((_, msg),) = ntc.warnings_for([new], {"a": {new: _c(2, 2)}}, wfs)
    assert "sanitizers.yml runs on this pull request, but its pytest list could not be read" in msg


# --- the legs' reports ------------------------------------------------------


@pytest.fixture(scope="module")
def real_reports(tmp_path_factory):
    """Two legs' shard artifacts from a real pytest run: a file that skips at
    import, one whose tests all skip, and one with a test that runs."""
    root = tmp_path_factory.mktemp("repo")
    tests = root / "python" / "tests"
    tests.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (tests / "test_import_skip.py").write_text(
        'import pytest\npytest.importorskip("no_such_module_for_1764")\n\n'
        "def test_a():\n    pass\n",
        encoding="utf-8",
    )
    (tests / "test_all_skip.py").write_text(
        "import pytest\n\n@pytest.mark.skip(reason='needs MIR')\n"
        "@pytest.mark.parametrize('x', [1, 2])\ndef test_a(x):\n    pass\n",
        encoding="utf-8",
    )
    (tests / "test_runs.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
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
    legs, unread = ntc.read_reports(reports, root)
    assert sorted(legs) == ["macos-15", "ubuntu-latest"] and unread == []
    assert legs["ubuntu-latest"]["python/tests/test_all_skip.py"] == _c(2, 2)
    assert legs["ubuntu-latest"]["python/tests/test_import_skip.py"] == _c(1, 1)


def test_ran_nothing_on_a_real_report(real_reports):
    root, reports = real_reports
    legs, _ = ntc.read_reports(reports, root)
    assert ntc.ran_nothing("python/tests/test_all_skip.py", legs)
    assert ntc.ran_nothing("python/tests/test_import_skip.py", legs)
    assert ntc.ran_nothing("python/tests/test_never_collected.py", legs)
    assert not ntc.ran_nothing("python/tests/test_runs.py", legs)


def _shard(logs: Path, leg: str, shard: int, xml: str) -> None:
    d = logs / f"python-tests-log-{leg}-{shard}"
    d.mkdir(parents=True)
    (d / "r.xml").write_text(xml, encoding="utf-8")


def _case(name: str) -> str:
    case = f'<testcase classname="python.tests.{name}" name="t"/>'
    return f"<testsuites><testsuite>{case}</testsuite></testsuites>"


def test_a_legs_shards_are_summed(tmp_path):
    # conftest puts each file in one shard, so a leg's counts are its shards' sum.
    tests = tmp_path / "python" / "tests"
    tests.mkdir(parents=True)
    for name in ("test_one", "test_two"):
        (tests / f"{name}.py").write_text("", encoding="utf-8")
    for shard, name in ((1, "test_one"), (2, "test_two")):
        _shard(tmp_path / "logs", "ubuntu-latest", shard, _case(name))
    legs, _ = ntc.read_reports(tmp_path / "logs", tmp_path)
    assert legs == {
        "ubuntu-latest": {"python/tests/test_one.py": _c(1), "python/tests/test_two.py": _c(1)}
    }


def test_a_malformed_report_is_set_aside_not_raised(tmp_path):
    tests = tmp_path / "python" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_one.py").write_text("", encoding="utf-8")
    _shard(tmp_path / "logs", "ubuntu-latest", 1, _case("test_one"))
    _shard(tmp_path / "logs", "ubuntu-latest", 2, "<testsuites><testsuite>")  # truncated
    legs, unread = ntc.read_reports(tmp_path / "logs", tmp_path)
    assert legs == {"ubuntu-latest": {"python/tests/test_one.py": _c(1)}}
    assert [s for s, _ in unread] == ["ubuntu-latest 2"]


def test_the_expected_shards_are_python_tests_matrix():
    expected = ntc.expected_shards(_read(WORKFLOWS / "python-tests.yml"))
    assert expected == {
        f"{leg} {s}" for leg in ("ubuntu-latest", "macos-15", "ubuntu-latest-lapack") for s in "12"
    }


def test_a_missing_shard_is_named(tmp_path):
    _shard(tmp_path, "a", 1, _case("test_one"))
    _shard(tmp_path, "a", 2, _case("test_two"))
    assert ntc.missing_shards(tmp_path, {"a 1", "a 2", "b 1"}, ["a 2"]) == ["a 2", "b 1"]


def test_one_leg_that_runs_a_test_is_enough():
    legs = {"a": {"python/tests/test_x.py": _c(3, 3)}, "b": {"python/tests/test_x.py": _c(3, 2)}}
    assert not ntc.ran_nothing("python/tests/test_x.py", legs)


# --- the warning ------------------------------------------------------------


def _workflows():
    return {
        "mir.yml": _wf("mir.yml", selectors=["python/tests/test_mir_equivalence.py"]),
        "windows-nfsim.yml": _wf("windows-nfsim.yml", selectors=["python/tests/test_nfsim*.py"]),
        "forced-jit.yml": _wf("forced-jit.yml", fires=False, whole=True),
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
    ((path, msg),) = ntc.warnings_for(["python/tests/test_mir_new.py"], LEGS, _workflows())
    assert path == "python/tests/test_mir_new.py"
    assert "all 4 of its tests skip on every python-tests leg (macos-15, ubuntu-latest)" in msg
    assert "No workflow that runs on this pull request runs it." in msg
    assert "mir.yml, windows-nfsim.yml." in msg
    assert "forced-jit.yml runs the whole suite, but not on this pull request" in msg


@pytest.mark.parametrize(
    "path",
    [
        "python/tests/test_nfsim_new.py",  # a running list's glob names it
        "python/tests/test_mir_equivalence.py",  # a running list names it as written
        "python/tests/test_partly.py",  # partly skipped: out of scope
    ],
)
def test_no_warning_where_a_running_list_names_it_or_some_test_ran(path):
    assert ntc.warnings_for([path], LEGS, _workflows()) == []


def test_a_list_that_deselects_it_does_not_count():
    wfs = {"w.yml": _wf(whole=True, excluded=["python/tests/test_x.py"])}
    ((_, msg),) = ntc.warnings_for(["python/tests/test_x.py"], {"a": {}}, wfs)
    assert "w.yml runs on this pull request, but leaves it out with --deselect or --ignore" in msg


def test_one_test_reads_as_one():
    legs = {"a": {"python/tests/test_one.py": _c(1, 1)}}
    ((_, msg),) = ntc.warnings_for(["python/tests/test_one.py"], legs, _workflows())
    assert "its one test skips on every python-tests leg (a)" in msg


def test_a_new_file_no_leg_collected_is_warned_about_as_such():
    ((_, msg),) = ntc.warnings_for(["python/tests/test_lost.py"], LEGS, _workflows())
    assert "no python-tests leg (macos-15, ubuntu-latest) collects it" in msg


def test_a_file_skipped_on_one_leg_and_uncollected_on_another_says_each():
    legs = {"a": {"python/tests/test_x.py": _c(4, 4)}, "b": {}}
    ((_, msg),) = ntc.warnings_for(["python/tests/test_x.py"], legs, _workflows())
    assert "nothing in it runs on any python-tests leg (a: 4 skipped; b: not collected)" in msg
    assert "every python-tests leg" not in msg


def test_with_no_list_carrying_workflow_it_suggests_none():
    wfs = {"forced-jit.yml": _wf("forced-jit.yml", fires=False, whole=True)}
    ((_, msg),) = ntc.warnings_for(["python/tests/test_mir_new.py"], LEGS, wfs)
    assert "add it to" not in msg and not msg.endswith(": .")


def test_without_a_whole_suite_workflow_the_message_says_nothing_of_one():
    wfs = {k: v for k, v in _workflows().items() if not v.whole}
    ((_, msg),) = ntc.warnings_for(["python/tests/test_mir_new.py"], LEGS, wfs)
    assert "whole suite" not in msg


def test_a_missing_shard_is_said_in_the_warning():
    ((_, msg),) = ntc.warnings_for(["python/tests/test_lost.py"], LEGS, _workflows(), ["b 2"])
    assert "(no report was read from b 2)" in msg


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
    (tests / "test_old.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    (tests / "test_moved.py").write_text("def test_b():\n    pass\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    for name in ("test_new.py", "helper.py", "suffix_test.py", "test_a space.py", "test_ünï.py"):
        (tests / name).write_text("def test_c():\n    pass\n", encoding="utf-8")
    (tests / "test_old.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    _git(tmp_path, "mv", "python/tests/test_moved.py", "python/tests/test_renamed.py")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "head")
    return tmp_path, base


def test_added_files_are_new_test_files_and_renames(repo):
    root, base = repo
    assert ntc.added_test_files(base, "HEAD", cwd=root) == [
        "python/tests/suffix_test.py",
        "python/tests/test_a space.py",
        "python/tests/test_new.py",
        "python/tests/test_renamed.py",
        "python/tests/test_ünï.py",
    ]


def test_pyproject_keeps_pytests_default_python_files():
    # The script's test-file pattern is pytest's default; a python_files setting
    # would change what counts as a test file.
    assert "python_files" not in _read(REPO_ROOT / "pyproject.toml")


def _main_with(monkeypatch, added, legs_unread, workflows, reports):
    monkeypatch.setattr(ntc, "changed_files", lambda b, h: [("A", f) for f in added])
    monkeypatch.setattr(ntc, "read_reports", lambda r: legs_unread)
    monkeypatch.setattr(ntc, "read_workflows", lambda changed: workflows)
    return ntc.main(["--reports", str(reports), "--base", "HEAD"])


def test_main_prints_a_warning_annotation(tmp_path, monkeypatch, capsys):
    assert (
        _main_with(
            monkeypatch, ["python/tests/test_mir_new.py"], (LEGS, []), _workflows(), tmp_path
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "::warning file=python/tests/test_mir_new.py::" in out
    assert "1 test file(s) added; 1 that no check on this change runs" in out
    # Only two legs' reports, and none of shard 2: said once, at the top.
    assert "::warning::no python-tests report was read from" in out


@pytest.mark.parametrize("make", ["missing", "empty"])
def test_main_with_no_reports_warns_and_passes(tmp_path, monkeypatch, capsys, make):
    reports = tmp_path / "logs"
    if make == "empty":
        reports.mkdir()
    monkeypatch.setattr(ntc, "changed_files", lambda b, h: [])
    assert ntc.main(["--reports", str(reports), "--base", "HEAD"]) == 0
    assert "no python-tests reports" in capsys.readouterr().out


def test_main_with_a_malformed_report_warns_and_passes(tmp_path, monkeypatch, capsys):
    _shard(tmp_path / "logs", "ubuntu-latest", 1, "<testsuites><testsuite>")
    monkeypatch.setattr(ntc, "changed_files", lambda b, h: [])
    assert ntc.main(["--reports", str(tmp_path / "logs"), "--base", "HEAD"]) == 0
    assert (
        "::warning::the python-tests report of ubuntu-latest 1 could not be read"
        in capsys.readouterr().out
    )


# --- the wiring -------------------------------------------------------------


def test_python_tests_runs_it_after_every_leg_on_a_pull_request():
    body = strip_comments(_read(WORKFLOWS / "python-tests.yml"))
    job = body[body.index("  new-test-coverage:") :]
    assert "needs: python-tests" in job
    assert "github.event_name == 'pull_request'" in job
    assert "fetch-depth: 2" in job
    assert "pattern: python-tests-log-*" in job
    assert "python3 ci/new_test_coverage.py --reports logs --base HEAD^1 --head HEAD" in job
    # The shard artifacts it downloads carry the JUnit reports it reads.
    assert "python-tests-${{ matrix.leg.label }}-${{ matrix.shard }}.xml" in body
