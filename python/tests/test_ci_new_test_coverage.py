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
from _net_corpus import _REPO_LOCATING, git_env  # noqa: E402

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
        "python/tests/test_b.py::test_one python/tests/test_g*.py 'python/tests/test_q*.py' "
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
    # An ignored file is marked: pytest ignores it only where it walks to it.
    # A deselected file is the prefix of its tests' node ids, marked as one.
    assert w.excluded == ["prefix:python/tests/test_e.py", "ignore:python/tests/test_f.py"]
    assert not w.whole and not w.unreadable
    assert w.names("python/tests/windows/test_x.py") and not w.names("python/tests/test_piped.py")
    assert not w.names("python/tests/test_gx/test_y.py")  # a shell * stays in its directory
    # A glob in quotes reaches pytest unexpanded and names nothing.
    assert not w.names("python/tests/test_q1.py")
    # Without globstar, bash reads ** as *: it does not cross a directory.
    deep = _cmd("python -m pytest python/tests/**/test_w.py")
    assert deep.names("python/tests/windows/test_w.py")
    assert not deep.names("python/tests/a/b/test_w.py")
    for suite in ("python/tests", "python/tests/"):
        assert _cmd(f"python -m pytest {suite} -n auto").whole
    # An --ignore-glob argument is left out too; it is not a second selector.
    w = _cmd("python -m pytest python/tests --ignore-glob 'python/tests/test_h*.py'")
    assert w.whole and len(w.excluded) == 1 and not w.selectors and not w.unreadable
    held = _wf(whole=True, excluded=w.excluded)
    assert not held.runs("python/tests/test_hx.py") and held.runs("python/tests/test_a.py")
    # It is an fnmatch pattern: its * crosses a directory.
    w = _cmd("python -m pytest python/tests --ignore-glob 'python/tests/*h3.py'")
    assert not _wf(whole=True, excluded=w.excluded).runs("python/tests/sub/test_h3.py")
    # fnmatch's */ needs the slash, ? crosses a directory too, and [..] is a class.
    for pattern, out, kept in (
        ("python/tests/*/test_x.py", "python/tests/sub/test_x.py", "python/tests/test_x.py"),
        ("python/tests/sub?test_x.py", "python/tests/sub/test_x.py", "python/tests/test_x.py"),
        ("python/tests/test_[ab].py", "python/tests/test_a.py", "python/tests/test_c.py"),
    ):
        w = _wf(whole=True, excluded=_cmd(f"python -m pytest --ignore-glob '{pattern}'").excluded)
        assert not w.runs(out) and w.runs(kept), pattern
    # A pattern that matches a directory leaves out what is under it, and one
    # that does not start with python/tests can match there all the same.
    for pattern in ("python/tests/su?", "python/tests/sub", "*/tests/sub/test_x.py"):
        w = _wf(whole=True, excluded=_cmd(f"python -m pytest --ignore-glob '{pattern}'").excluded)
        assert not w.runs("python/tests/sub/test_x.py"), pattern
        assert w.runs("python/tests/test_x.py"), pattern
    # A whole option in quotes is still that option.
    w = _cmd('python -m pytest python/tests "--ignore-glob=python/tests/test_h*.py"')
    assert not _wf(whole=True, excluded=w.excluded).runs("python/tests/test_hx.py")
    # --ignore and --deselect take a path or a node id as written: a quoted glob
    # names none, and pytest leaves nothing out for it.
    for opt in ("--ignore", "--deselect"):
        assert not _cmd(f"python -m pytest python/tests {opt} 'python/tests/test_h*.py'").excluded
        assert not _cmd(f'python -m pytest python/tests "{opt}=python/tests/test_h*.py"').excluded
    # Quoted, a glob is still the option's argument, and the path after it is
    # still a test path, not the argument.
    w = _cmd("python -m pytest --ignore-glob 'python/tests/test_h*.py' python/tests/test_y.py")
    assert len(w.excluded) == 1 and w.selectors == ["python/tests/test_y.py"]
    # So is a deselected node id whose test id has brackets, whatever the
    # brackets hold: the words after it are read as written.
    for test_id in ("p-1", "#1", "a|b", "a;b"):
        w = _cmd(
            "python -m pytest python/tests/test_a.py "
            f"--deselect 'python/tests/test_a.py::test_b[{test_id}]' python/tests/test_c.py"
        )
        assert w.selectors == ["python/tests/test_a.py", "python/tests/test_c.py"], test_id
        assert not w.excluded and not w.unreadable
    # A quoted node id selects its file; with the glob in its file part it is
    # a path pytest does not find.
    w = _cmd("python -m pytest 'python/tests/test_a.py::test_b[q]' python/tests/test_c.py")
    assert w.node_ids == ["python/tests/test_a.py"] and w.selectors == ["python/tests/test_c.py"]
    assert not _cmd("python -m pytest 'python/tests/test_q*.py::test_x'").node_ids
    # One outside python/tests is no test file of the suite's.
    w = _cmd("python -m pytest 'docs/test_a.py::test_b[q]'")
    assert not w.node_ids and not w.selectors and not w.whole and not w.unreadable
    # The mark's own text in a command is a word like any other.
    w = _cmd("python -m pytest @@quoted:zz@@ --ignore @@quoted:zz@@ python/tests/test_c.py")
    assert w.selectors == ["python/tests/test_c.py"] and not w.excluded and not w.unreadable


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
        # One changed file the ignore list does not cover is enough.
        (
            "on:\n  pull_request:\n    paths-ignore:\n      - 'docs/**'\n",
            ["docs/a.md", "src/a.cpp"],
            True,
        ),
        (_PATHS, ["README.md"], True),  # `**/` matches no directory as well
        ("on:\n  pull_request:\n  push:\n", ["anything"], True),
        ("on: [push, pull_request]\n", ["anything"], True),
        ("on:\n  push:\n  pull_request_target:\n", ["anything"], False),
        ("on: [push, pull_request_target]\n", ["anything"], False),
        ("on: pull_request_target\n", ["anything"], False),
        # GitHub reads ? as "zero or one of the character before", + as "one or
        # more", [..] as a class; the script reads none of the three, so it cannot tell.
        ("on:\n  pull_request:\n    paths:\n      - 'src/ab?.cpp'\n", ["src/abc.cpp"], None),
        ("on:\n  pull_request:\n    paths:\n      - 'src/a+.cpp'\n", ["src/a+.cpp"], None),
        ("on:\n  pull_request:\n    paths:\n      - 'src/[ab].cpp'\n", ["src/a.cpp"], None),
        ("on:\n  pull_request:\n    paths-ignore:\n      - 'doc?/**'\n", ["src/a.cpp"], None),
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
    # On one that changes src/, w.yml runs it, but only for that change: the
    # glob is wider than the filter, so a later change to the file alone would not.
    (w,) = ntc.read_workflows([new, "src/a.cpp"], tmp_path).values()
    assert w.runs(new)
    ((_, msg),) = ntc.warnings_for([new], legs, {"w.yml": w})
    assert "w.yml runs it on this pull request only for the pull request's other changes" in msg


def test_a_trigger_that_cannot_be_read_never_counts():
    # A branches filter, or types without synchronize: whether the workflow
    # runs on this pull request is not known, so its list does not count.
    new = "python/tests/test_mir_new.py"
    w = _wf(fires=None, selectors=[new])
    w.why_not = "its pull_request trigger has a branches filter, which is not read"
    assert not w.runs(new)
    ((_, msg),) = ntc.warnings_for([new], {"a": {new: _c(2, 2)}}, {"w.yml": w})
    assert "Whether w.yml runs on this pull request is not known: its pull_request" in msg


def test_a_quoted_argument_after_an_equals_sign_is_the_options_own():
    # --ignore='x*' reaches pytest as the glob as written, as --ignore 'x*'
    # does: it names no file and leaves nothing out (pytest 9.0.2's --co).
    for opt in ("--ignore", "--deselect"):
        for q in ("'", '"'):
            w = _cmd(f"python -m pytest python/tests {opt}={q}python/tests/test_h*.py{q}")
            assert not w.excluded and not w.unreadable, (opt, q)
            assert _wf(whole=True, excluded=w.excluded).runs("python/tests/test_h1.py")
    # --ignore-glob='x*' is still the pattern pytest matches, and the word
    # after it still a test path.
    w = _cmd("python -m pytest --ignore-glob='python/tests/test_h*.py' python/tests/test_y.py")
    assert w.excluded == ["fnmatch:python/tests/test_h*.py"]
    assert w.selectors == ["python/tests/test_y.py"]


def test_ignore_leaves_out_only_what_pytest_walks_down_to():
    # pytest collects a path it is given even where --ignore or --ignore-glob
    # matches it (pytest 9.0.2's --co); they apply below a directory it walks.
    for opt in ("--ignore-glob 'python/tests/test_h*.py'", "--ignore python/tests/test_h1.py"):
        named = _cmd(f"python -m pytest python/tests/test_h1.py python/tests/test_a.py {opt}")
        assert _wf(selectors=named.selectors, excluded=named.excluded).runs(
            "python/tests/test_h1.py"
        ), opt
        whole = _cmd(f"python -m pytest python/tests {opt}")
        assert not _wf(whole=True, excluded=whole.excluded).runs("python/tests/test_h1.py"), opt
    # A directory it is given is not asked either: only what is below it.
    sub = _cmd("python -m pytest python/tests/sub --ignore-glob 'python/tests/su?'")
    assert _wf(selectors=sub.selectors, excluded=sub.excluded).runs("python/tests/sub/test_x.py")
    whole = _cmd("python -m pytest python/tests --ignore-glob 'python/tests/su?'")
    assert not _wf(whole=True, excluded=whole.excluded).runs("python/tests/sub/test_x.py")
    # A node id is a path it is given; --deselect applies wherever a test came from.
    node = _cmd(
        "python -m pytest python/tests/test_h1.py::test_x --ignore python/tests/test_h1.py"
    )
    assert _wf(node_ids=node.node_ids, excluded=node.excluded).runs("python/tests/test_h1.py")
    gone = _cmd("python -m pytest python/tests/test_h1.py --deselect python/tests/test_h1.py")
    assert not _wf(selectors=gone.selectors, excluded=gone.excluded).runs(
        "python/tests/test_h1.py"
    )


def _runs(command: str, path: str) -> bool:
    w = _cmd(command)
    assert not w.unreadable, command
    w.fires = True
    return w.runs(path)


def test_a_path_given_under_a_directory_given_is_ignored_like_the_rest():
    # pytest 9 drops a path it is given that lies under a directory it is also
    # given, as a duplicate (9.0.2's and 9.2.0's --co): the directory is
    # walked, and --ignore applies to the path as to the rest.
    x, h3 = "python/tests/sub/test_x.py", "python/tests/sub/test_h3.py"
    for opt in (f"--ignore {x}", "--ignore-glob '*sub/test_x.py'", f"--ignore={x}"):
        for given in (x, f"{x}::test_b"):
            for suite in ("python/tests", "python/tests/sub"):
                for cmd in (f"{suite} {given} {opt}", f"{given} {opt} {suite}"):
                    assert not _runs(f"python -m pytest {cmd}", x), cmd
                    assert _runs(f"python -m pytest {cmd}", h3), cmd
            # Given alone, it is collected whatever --ignore says.
            assert _runs(f"python -m pytest {given} {opt}", x), (given, opt)
    # Of two directories given, the outer one is walked, and the inner one is
    # asked on the way down like any other.
    for cmd in ("python/tests python/tests/sub", "python/tests/sub/ python/tests/"):
        assert not _runs(f"python -m pytest {cmd} --ignore python/tests/sub", x), cmd
        assert _runs(f"python -m pytest {cmd} --ignore python/tests/sub", "python/tests/test_a.py")
    # The inner one alone is a directory it is given, and is not asked.
    assert _runs("python -m pytest python/tests/sub --ignore python/tests/sub", x)


def test_deselect_is_a_prefix_of_the_node_ids_as_written():
    # pytest leaves out each test whose node id starts with the argument
    # (9.2.0's --co): part of a file's name, or of a directory's, is enough.
    h1, hx, h3 = (
        "python/tests/test_h1.py",
        "python/tests/test_hx.py",
        "python/tests/sub/test_h3.py",
    )
    for form in ("--deselect python/tests/test_h", "--deselect=python/tests/test_h"):
        cmd = f"python -m pytest python/tests {form}"
        assert not _runs(cmd, h1) and not _runs(cmd, hx), form
        assert _runs(cmd, h3) and _runs(cmd, "python/tests/test_a.py"), form
    assert not _runs("python -m pytest python/tests --deselect python/tests/su", h3)
    assert _runs("python -m pytest python/tests --deselect python/tests/su", h1)
    assert not _runs("python -m pytest python/tests --deselect python/te", h1)
    # It applies to a path pytest is given as well.
    assert not _runs(f"python -m pytest {h1} {hx} --deselect python/tests/test_h1", h1)
    assert _runs(f"python -m pytest {h1} {hx} --deselect python/tests/test_h1", hx)
    # As written: ./ starts no node id, and a node id leaves the rest of its file.
    assert _runs("python -m pytest python/tests --deselect ./python/tests/test_h1.py", h1)
    assert _runs(f"python -m pytest python/tests --deselect {h1}::test_x", h1)
    # A glob after a space is the shell's to expand: the directory it names is
    # the prefix, and each file it names is read as left out.
    assert not _runs("python -m pytest python/tests --deselect python/tests/su?", h3)
    w = _cmd("python -m pytest python/tests --deselect python/tests/test_h*.py")
    assert w.excluded == ["python/tests/test_h*.py"]
    for opt in ("--deselect", "--ignore"):
        # After = the shell expands nothing, quoted or not, and pytest takes
        # the glob as written: nothing is left out.
        assert not _cmd(f"python -m pytest python/tests {opt}=python/tests/test_h*.py").excluded


@pytest.mark.parametrize(
    "arg,out",
    [
        ("--ignore-glob './python/tests/test_h*.py'", "python/tests/test_h1.py"),
        ("--ignore-glob 'python/tests//test_h*.py'", "python/tests/test_h1.py"),
        ("--ignore-glob 'python/tests/sub*/'", "python/tests/sub/test_x.py"),
        ("--ignore ./python/tests/test_h1.py", "python/tests/test_h1.py"),
    ],
)
def test_ignore_arguments_are_normalised_as_pytest_normalises_them(arg, out):
    # pytest makes them absolute first, which drops ./, // and a trailing /.
    w = _wf(whole=True, excluded=_cmd(f"python -m pytest python/tests {arg}").excluded)
    assert not w.runs(out)
    assert w.runs("python/tests/test_a.py")


@pytest.mark.parametrize(
    "arg", ['--deselect "$X"', "--ignore-glob '${{ matrix.x }}'", "--ignore=$Y"]
)
def test_an_excluding_option_given_a_variable_leaves_the_list_unread(arg):
    w = _cmd(f"python -m pytest python/tests/test_a.py {arg}")
    assert w.unreadable and not w.excluded
    assert not _wf(selectors=w.selectors, unreadable=w.unreadable).runs("python/tests/test_a.py")


_RUN_NEW = "jobs:\n  j:\n    steps:\n      - run: python -m pytest python/tests/test_mir_new.py\n"


def _on_paths(*paths: str) -> str:
    return "on:\n  pull_request:\n    paths:\n" + "".join(f"      - '{p}'\n" for p in paths)


def test_a_list_whose_paths_filter_does_not_match_it_is_warned_about(tmp_path):
    # The pull request adds the file to w.yml's list but not to its paths
    # filter. w.yml runs on this pull request, as its filter matches w.yml
    # itself, and runs the file; the next change to the file alone would not.
    new = "python/tests/test_mir_new.py"
    changed = [new, ".github/workflows/w.yml"]
    legs = {"a": {new: _c(2, 2)}}
    (tmp_path / "w.yml").write_text(_on_paths(".github/workflows/w.yml") + _RUN_NEW, "utf-8")
    wfs = ntc.read_workflows(changed, tmp_path)
    assert wfs["w.yml"].runs(new) and not wfs["w.yml"].fires_for(new)
    ((path, msg),) = ntc.warnings_for([new], legs, wfs)
    assert path == new
    assert "all 2 of its tests skip on every python-tests leg (a)." in msg
    assert "w.yml runs it on this pull request only for the pull request's other changes" in msg
    assert "a later change to it alone will run it nowhere. Add it to that paths filter." in msg
    # A shard with no report is said here too, as in the other warning.
    ((_, msg),) = ntc.warnings_for([new], legs, wfs, ["b 2"])
    assert "(no report was read from b 2). w.yml runs it" in msg
    # Another list that runs it and whose filter matches it: nothing to say.
    (tmp_path / "v.yml").write_text(_on_paths("python/tests/test_mir*.py") + _RUN_NEW, "utf-8")
    assert ntc.warnings_for([new], legs, ntc.read_workflows(changed, tmp_path)) == []
    (tmp_path / "v.yml").unlink()
    # With the file in w.yml's filter too, nothing to say.
    (tmp_path / "w.yml").write_text(_on_paths(".github/workflows/w.yml", new) + _RUN_NEW, "utf-8")
    assert ntc.warnings_for([new], legs, ntc.read_workflows(changed, tmp_path)) == []


def test_the_late_warning_names_what_in_the_trigger_leaves_it_out(tmp_path):
    # Not a paths filter that misses it: a paths-ignore entry, or a later `!`
    # entry, is what to change.
    new = "python/tests/test_mir_new.py"
    changed = [new, "src/a.cpp"]
    legs = {"a": {new: _c(2, 2)}}
    ignore = "on:\n  pull_request:\n    paths-ignore:\n      - 'python/tests/**'\n"
    (tmp_path / "w.yml").write_text(ignore + _RUN_NEW, "utf-8")
    ((_, msg),) = ntc.warnings_for([new], legs, ntc.read_workflows(changed, tmp_path))
    assert (
        "w.yml: its paths-ignore filter leaves it out (`python/tests/**`). "
        "Take it out of that paths-ignore filter." in msg
    )
    assert "Add it to that paths filter" not in msg
    paths = _on_paths("src/**", "python/tests/**", "!python/tests/test_mir*.py")
    (tmp_path / "w.yml").write_text(paths + _RUN_NEW, "utf-8")
    ((_, msg),) = ntc.warnings_for([new], legs, ntc.read_workflows(changed, tmp_path))
    assert "its paths filter takes it back out with `!python/tests/test_mir*.py`" in msg
    assert "Add it to that paths filter" not in msg


def test_two_lists_whose_filters_do_not_match_it_are_named_together(tmp_path):
    new = "python/tests/test_mir_new.py"
    for name in ("v.yml", "w.yml"):
        (tmp_path / name).write_text(_on_paths("src/**") + _RUN_NEW, "utf-8")
    wfs = ntc.read_workflows([new, "src/a.cpp"], tmp_path)
    ((_, msg),) = ntc.warnings_for([new], {"a": {new: _c(1, 1)}}, wfs)
    assert "v.yml, w.yml run it on this pull request only for the pull request's" in msg
    assert "no paths filter of those matches it" in msg and "those paths filters." in msg


def test_main_counts_a_file_a_check_runs_only_for_the_other_changes(tmp_path, monkeypatch, capsys):
    new = "python/tests/test_mir_new.py"
    (tmp_path / "w.yml").write_text(_on_paths("src/**") + _RUN_NEW, "utf-8")
    # main hands read_workflows every file the change touches, not the added
    # test files alone: w.yml runs on this pull request for src/a.cpp.
    real = ntc.read_workflows
    monkeypatch.setattr(ntc, "changed_files", lambda b, h: [("A", new), ("M", "src/a.cpp")])
    monkeypatch.setattr(ntc, "read_reports", lambda r: ({"a": {new: _c(1, 1)}}, []))
    monkeypatch.setattr(ntc, "read_workflows", lambda changed: real(changed, tmp_path))
    assert ntc.main(["--reports", str(tmp_path), "--base", "HEAD"]) == 0
    out = capsys.readouterr().out
    assert (
        "1 test file(s) added; 0 that no check on this change runs; "
        "1 that a check runs only for this change's other files"
    ) in out.splitlines()
    assert f"::warning file={new}::" in out and "w.yml runs it on this pull request only" in out


def test_every_file_a_list_runs_is_in_that_workflows_paths_filter():
    # The late warning's case, for the files already on a list: a change to one
    # of those files alone would fire no workflow that runs it.
    files = [
        p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "python/tests").rglob("*.py")
    ]
    lists = {n: w for n, w in ntc.read_workflows([]).items() if w.selectors or w.node_ids}
    assert {"mir.yml", "windows-klu.yml", "windows-nfsim.yml", "windows-tail.yml"} <= set(lists)
    outside = {
        n: sorted(f for f in files if w.names(f) and not w.fires_for(f)) for n, w in lists.items()
    }
    assert outside == {n: [] for n in lists}, (
        "a workflow's pytest list runs a test file its own paths filter does not match; "
        "add the file to that workflow's paths filter"
    )


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


@pytest.fixture(autouse=True)
def _no_inherited_repository(monkeypatch):
    """git exports an absolute GIT_DIR to a hook run from a linked worktree, and
    the pre-push hook runs this suite. ``ntc.main``'s own git calls and the
    child pytest below would then read that repository, so no test here sees it.
    """
    for name in _REPO_LOCATING:
        monkeypatch.delenv(name, raising=False)


def _git(cwd: Path, *args: str) -> str:
    # git_env(): an exported GIT_DIR sends every command to that repository
    # whatever the cwd, and this fixture's `git init` once made the developer's
    # checkout bare. No hooks and no signing in a scratch repository.
    cfg = ["-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"]
    return subprocess.run(
        ["git", *cfg, *args],
        cwd=cwd,
        env=git_env(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def test_a_repository_in_the_environment_is_not_written_to(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    config = other / ".git" / "config"
    before = config.read_bytes()
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _git(scratch, "init", "-q")
    _git(scratch, "config", "user.name", "ci")
    assert config.read_bytes() == before
    assert (scratch / ".git" / "config").is_file()


def test_this_module_under_a_worktree_hooks_git_dir_passes_and_leaves_that_repository_alone(
    tmp_path,
):
    """The incident, end to end: git hands a pre-push hook run from a linked
    worktree GIT_DIR=<main>/.git/worktrees/<name>. The rest of this module, run
    in that environment, must pass and leave that repository as it was."""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q")
    _git(
        main,
        "-c",
        "user.name=u",
        "-c",
        "user.email=u@x",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "a",
    )
    _git(main, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "w")
    config = main / ".git" / "config"
    before = config.read_bytes()
    heads = _git(main, "for-each-ref", "--format=%(objectname)", "refs/heads")
    env = {**os.environ, "GIT_DIR": str(main / ".git" / "worktrees" / "wt")}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            __file__,
            "-k",
            "not worktree_hooks_git_dir",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert config.read_bytes() == before
    assert _git(main, "for-each-ref", "--format=%(objectname)", "refs/heads") == heads


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
    # The whole line: nothing of a file run only for the other changes.
    assert "1 test file(s) added; 1 that no check on this change runs" in out.splitlines()
    # Only two legs' reports, and none of shard 2: said once, at the top.
    assert "::warning::no python-tests report was read from" in out


def test_main_checks_only_the_test_files_the_change_adds(tmp_path, monkeypatch, capsys):
    # An edited test file that now skips, a deleted one and a new helper are not
    # this check's: main reads the change as added_test_files does.
    changed = [
        ("M", "python/tests/test_mir_equivalence.py"),
        ("D", "python/tests/test_nfsim_new.py"),
        ("A", "python/tests/helper_new.py"),
        ("A", "python/tests/test_mir_new.py"),
    ]
    monkeypatch.setattr(ntc, "changed_files", lambda b, h: changed)
    monkeypatch.setattr(ntc, "read_reports", lambda r: (LEGS, []))
    monkeypatch.setattr(ntc, "read_workflows", lambda changed: _workflows())
    assert ntc.main(["--reports", str(tmp_path), "--base", "HEAD"]) == 0
    out = capsys.readouterr().out
    assert "1 test file(s) added; 1 that no check on this change runs" in out
    assert out.count("::warning file=") == 1


def test_main_with_a_base_git_cannot_read_warns_and_passes(repo, monkeypatch, capsys):
    # A checkout with no parent (fetch-depth 1), or any git failure: the job
    # warns and passes, as for a missing report.
    root, _ = repo
    real = ntc.changed_files
    monkeypatch.setattr(ntc, "changed_files", lambda b, h: real(b, h, cwd=root))
    monkeypatch.setenv("LC_ALL", "C")  # git's own words, whatever the locale
    assert ntc.main(["--reports", str(root), "--base", "0" * 40]) == 0
    out = capsys.readouterr().out
    assert "::warning::the files this change touches could not be read (fatal: " in out
    assert "test file(s) added" not in out


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


@pytest.mark.parametrize(
    ("stderr", "said"),
    [
        # git says what failed on its first line; hints follow it.
        (b"fatal: bad object 0\nhint: a\n", "(fatal: bad object 0); none checked"),
        # With nothing on stderr, the error as it reads.
        (b"", "(Command '['git']' returned non-zero exit status 128.); none checked"),
    ],
    ids=["first-line", "no-stderr"],
)
def test_main_says_why_git_failed_in_one_line(tmp_path, monkeypatch, capsys, stderr, said):
    def fail(base, head):
        raise subprocess.CalledProcessError(128, ["git"], stderr=stderr)

    monkeypatch.setattr(ntc, "changed_files", fail)
    assert ntc.main(["--reports", str(tmp_path), "--base", "HEAD"]) == 0
    (line,) = capsys.readouterr().out.splitlines()
    assert line == f"::warning::the files this change touches could not be read {said}"


def test_main_without_git_warns_and_passes(tmp_path, monkeypatch, capsys):
    def no_git(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(ntc.subprocess, "run", no_git)
    assert ntc.main(["--reports", str(tmp_path), "--base", "HEAD"]) == 0
    out = capsys.readouterr().out
    assert "::warning::the files this change touches could not be read (" in out
    assert "No such file or directory" in out


def test_a_workflow_file_that_is_not_utf8_is_read_not_raised_on(tmp_path):
    body = _on_paths("src/**").encode() + b"# \xff\xfe\n" + _RUN_NEW.encode()
    (tmp_path / "w.yml").write_bytes(body)
    assert ntc.read_workflows(["src/a.cpp"], tmp_path)["w.yml"].names(
        "python/tests/test_mir_new.py"
    )


def test_a_report_in_an_encoding_python_lacks_is_unread_not_raised_on(tmp_path):
    _shard(tmp_path / "logs", "ubuntu-latest", 1, '<?xml version="1.0" encoding="x-nope"?><a/>')
    legs, unread = ntc.read_reports(tmp_path / "logs")
    assert [shard for shard, _ in unread] == ["ubuntu-latest 1"]


def test_an_annotation_escapes_what_would_end_or_split_it(tmp_path, monkeypatch, capsys):
    # GitHub reads a workflow command to the end of its line, and a property
    # value to the next `:` or `,`: a file name holding them is escaped.
    odd = "python/tests/test_a\nb,c:d%.py"
    assert _main_with(monkeypatch, [odd], (LEGS, []), _workflows(), tmp_path) == 0
    out = capsys.readouterr().out
    assert "::warning file=python/tests/test_a%0Ab%2Cc%3Ad%25.py::python/tests/test_a%0Ab" in out
    assert "test_a\nb" not in out


# --- the wiring -------------------------------------------------------------


def test_python_tests_runs_it_after_every_leg_on_a_pull_request():
    body = strip_comments(_read(WORKFLOWS / "python-tests.yml"))
    job = body[body.index("  new-test-coverage:") :]
    assert "needs: python-tests" in job
    assert "if: ${{ !cancelled() && github.event_name == 'pull_request' }}" in job
    assert "fetch-depth: 2" in job
    assert "pattern: python-tests-log-*" in job
    assert "python3 ci/new_test_coverage.py --reports logs --base HEAD^1 --head HEAD" in job
    # The shard artifacts it downloads carry the JUnit reports it reads.
    assert "python-tests-${{ matrix.leg.label }}-${{ matrix.shard }}.xml" in body
