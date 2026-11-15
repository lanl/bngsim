#!/usr/bin/env python3
"""Warn when a pull request adds a test file that no check on it runs (#1764).

    python3 ci/new_test_coverage.py --reports DIR --base SHA [--head SHA]

Several workflows run a hand-picked list of test files, and only when certain
paths change: mir.yml, windows-klu.yml, windows-nfsim.yml, windows-tail.yml.
The python-tests legs build without some of what those have (MIR, KLU on
Windows, MSVC), so the tests for it skip there. A test file a pull request adds
is on none of those lists unless someone edits the workflow, so a file that
only does real work under one of them skips in python-tests, runs nowhere else,
and every check is green. Review caught one such MIR check on #1359.

This reads the JUnit reports every python-tests leg wrote, and, for each test
file the change adds, asks two things:

* Did every test in it skip on every leg? A file not collected at all counts:
  nothing in it ran. A file with any test that ran is left alone; partly
  skipped files are out of scope.
* Does any workflow other than python-tests name it in a pytest command? The
  lists are read from the workflow files, globs and plain paths alike, with
  ``--deselect`` and ``--ignore`` arguments left out. A workflow that runs the
  whole of python/tests (forced-jit.yml and sanitizers.yml, both nightly) does
  not count as naming it: it runs the file only after the merge.

A file that skipped everywhere and that no list names gets a ``::warning::``,
which says which workflows run hand-picked lists, and which run the whole
suite nightly. A warning, not a failure: a new file may skip everywhere for a
reason (a quarantine, an optional dependency no leg installs).

Stdlib only, like the rest of ci/, plus ``git`` for the added files.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_file_counts import Counts, count_reports  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: The workflow whose legs' reports are read; its own pytest commands run the
#: whole suite, so they name nothing.
SUITE_WORKFLOW = "python-tests.yml"

#: A shard's artifact directory: ``python-tests-log-<leg>-<shard>``.
SHARD_DIR = re.compile(r"^python-tests-log-(?P<leg>.+)-(?P<shard>\d+)$")

#: pytest options whose argument is a path or node id it does not run.
_EXCLUDING = frozenset({"--deselect", "--ignore", "--ignore-glob"})

_TEST_FILE = re.compile(r"^python/tests/(?:.+/)?test_[^/]*\.py$")


# --- the workflows' lists ---------------------------------------------------


def _strip_comments(text: str) -> str:
    """Whole-line ``#`` comments dropped: these files explain themselves in
    comments that name the very files being looked for."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def pytest_commands(text: str) -> list[list[str]]:
    """Each pytest command in a workflow, as its tokens, continuation lines joined."""
    lines = _strip_comments(text).splitlines()
    out: list[list[str]] = []
    for i, line in enumerate(lines):
        # Only ``python/tests`` tokens are kept, so a line that merely mentions
        # pytest (a step name, an input's description) contributes nothing.
        if not re.search(r"-m\s+pytest\b|(?:^|\s)pytest\s", line):
            continue
        block = [line]
        while block[-1].rstrip().endswith("\\") and i + len(block) < len(lines):
            block.append(lines[i + len(block)])
        out.append(" ".join(ln.rstrip().rstrip("\\") for ln in block).split())
    return out


def selectors(tokens: list[str]) -> tuple[list[str], bool]:
    """``(file selectors, runs the whole suite)`` of one pytest command.

    A node id (``file.py::test``) selects its file. An argument to
    ``--deselect`` or ``--ignore`` selects nothing. ``python/tests`` itself, or
    a directory under it, is the whole suite there.
    """
    out: list[str] = []
    whole = False
    skip_next = False
    for tok in tokens:
        tok = tok.strip("\"'")
        if skip_next:
            skip_next = False
            continue
        opt, eq, _ = tok.partition("=")
        if opt in _EXCLUDING:
            skip_next = not eq
            continue
        if not tok.startswith("python/tests"):
            continue
        path = tok.split("::", 1)[0].rstrip("/")
        if path.endswith(".py"):
            out.append(path)
        elif path == "python/tests" or "*" not in path:
            whole = True
        else:
            out.append(path)
    return out, whole


def workflow_lists(root: Path = WORKFLOWS) -> dict[str, tuple[list[str], bool]]:
    """Per workflow other than python-tests: its file selectors over every pytest
    command, and whether one of them runs the whole suite."""
    lists: dict[str, tuple[list[str], bool]] = {}
    for wf in sorted(root.glob("*.yml")):
        if wf.name == SUITE_WORKFLOW:
            continue
        sel: list[str] = []
        whole = False
        for tokens in pytest_commands(wf.read_text(encoding="utf-8")):
            s, w = selectors(tokens)
            sel += s
            whole = whole or w
        if sel or whole:
            lists[wf.name] = (sel, whole)
    return lists


def named_by(path: str, lists: dict[str, tuple[list[str], bool]]) -> list[str]:
    """The workflows whose lists name ``path``, by a glob or as written."""
    return sorted(
        name for name, (sel, _) in lists.items() if any(fnmatch.fnmatchcase(path, g) for g in sel)
    )


# --- the legs' reports ------------------------------------------------------


def counts_per_leg(reports: Path, root: Path | None = None) -> dict[str, Counts]:
    """Each python-tests leg's per-file counts, its shards' reports summed."""
    by_leg: dict[str, list[Path]] = {}
    for d in sorted(p for p in reports.iterdir() if p.is_dir()):
        m = SHARD_DIR.match(d.name)
        if m:
            by_leg.setdefault(m["leg"], []).extend(sorted(d.rglob("*.xml")))
    return {leg: count_reports(xmls, root) for leg, xmls in by_leg.items()}


def ran_nothing(path: str, legs: dict[str, Counts]) -> bool:
    """Every test in ``path`` skipped on every leg, or the file was not collected."""
    for counts in legs.values():
        c = counts.get(path, {"collected": 0, "skipped": 0})
        if c["collected"] > c["skipped"]:
            return False
    return True


# --- the check --------------------------------------------------------------


def added_test_files(base: str, head: str, cwd: Path = REPO_ROOT) -> list[str]:
    """The python/tests test files ``head`` adds over ``base``. A rename is a
    deletion and an addition (``--no-renames``), so a renamed file is checked
    as new: whatever ran it under its old name will not under the new one."""
    out = subprocess.run(
        ["git", "diff", "--no-renames", "--diff-filter=A", "--name-only", base, head],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    ).stdout
    return sorted(f for f in out.split() if _TEST_FILE.match(f))


def warnings_for(
    added: Iterable[str],
    legs: dict[str, Counts],
    lists: dict[str, tuple[list[str], bool]],
) -> list[tuple[str, str]]:
    """``(file, message)`` for each added file that no check on the change runs."""
    listed = sorted(name for name, (sel, _) in lists.items() if sel)
    nightly = sorted(name for name, (_, whole) in lists.items() if whole)
    leg_names = ", ".join(sorted(legs)) or "none reported"
    out = []
    for path in added:
        if not ran_nothing(path, legs) or named_by(path, lists):
            continue
        n = max((c.get(path, {}).get("collected", 0) for c in legs.values()), default=0)
        if n == 0:
            what = "it is not collected"
        elif n == 1:
            what = "its one test skips"
        else:
            what = f"all {n} of its tests skip"
        after = (
            f" {' and '.join(nightly)} run the whole suite, but not on this pull request."
            if nightly
            else ""
        )
        out.append(
            (
                path,
                f"{path} is new, {what} on every python-tests leg ({leg_names}), "
                "and no workflow's pytest list names it, so no check on this pull "
                f"request runs it.{after} If it needs what one of these legs builds, "
                f"add it to that workflow's list: {', '.join(listed)}.",
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", type=Path, required=True, help="the legs' shard artifacts")
    ap.add_argument("--base", required=True, help="the commit the change is measured from")
    ap.add_argument("--head", default="HEAD")
    a = ap.parse_args(argv)

    added = added_test_files(a.base, a.head)
    legs = counts_per_leg(a.reports)
    if not legs:
        print(f"::warning::no python-tests reports under {a.reports}; nothing checked")
        return 0
    found = warnings_for(added, legs, workflow_lists())
    print(f"{len(added)} test file(s) added; {len(found)} that no check on this change runs")
    for path, msg in found:
        print(f"::warning file={path}::{msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
