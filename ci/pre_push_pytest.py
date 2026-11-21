#!/usr/bin/env python3
"""The pre-push hook's pytest run: all of ``python/tests``, in parallel.

Run serially in one process, the suite took 44m29s on a 16-core Apple Silicon
Mac (2026-10-18, 16,834 tests, warm artifact cache), and every push paid it.
Under pytest-xdist with one test file per worker the same run took 4m42s on 12
workers, with the same outcome test for test. Nothing is dropped to get there:
this is still the whole directory, so the hook and python-tests.yml keep
running one suite.

``--dist worksteal``: tests go to the workers one at a time, and a worker that
runs out takes queued tests from the others. One file per worker (``loadfile``)
made the slowest file the floor: test_expression_backend_equivalence.py, about
five minutes on one worker. In a checkout without the gitignored corpora,
``loadfile`` took 304 s and ``worksteal`` 184 s on 12 workers, and every one of
the 13,573 tests had the same outcome both ways (2026-10-18). A module- or
class-scoped fixture is then built once per worker that runs one of its tests,
rather than once per file; 14 of the 475 files have one.

**The test cache is shared with the main checkout.** macOS checks a newly
written ``.dylib`` the first time anything loads it, at 0.5-2.4 s a file, and it
checks them one at a time across the whole machine: twelve fresh libraries
loaded at once by twelve processes took 11.1 s, finishing one after another.
Once a file has been checked, every later load of it, in any process, is
instant. A copy of it at another path is checked again. So a ``.claude`` or
``bngsim-wt`` worktree, whose own ``.pytest_cache`` starts empty, recompiles and
re-checks every codegen library the suite builds: the first push from one took
19m15s on 12 workers, most of it waiting on those checks. So from a worktree,
unless ``BNGSIM_TEST_CACHE_DIR`` is already set, this points it at the main
checkout's ``.pytest_cache/d/bngsim``. The main checkout itself is left on
pytest's own route to that same directory. The cache is content-addressed and
written by atomic rename, so two checkouts sharing it is safe.

The worker count defaults to the number of cores that are not efficiency cores
on Apple Silicon, and to xdist's ``auto`` (one per CPU) everywhere else. On the
same 12+4-core Mac, ``auto``'s 16 workers took 5m27s against 12 workers' 4m42s
(one file per worker, then): work dealt to an efficiency core runs slower.
``BNGSIM_PREPUSH_WORKERS`` overrides the count, and ``0`` runs the whole suite
in a single pytest process with no xdist at all, as the hook used to, for a
failure that might be about what ran before it.

Without pytest-xdist the hook stops at once and says how to get it. Running
serially instead would turn a five-minute push into a 45-minute one that
pre-commit reports only as ``Passed``, since it shows a passing hook's output
to no one.

**A push that changes only files the suite never reads does not run it.**
pre-commit passes the pushed range as ``PRE_COMMIT_FROM_REF`` and
``PRE_COMMIT_TO_REF``, and this lists the changed paths itself: ``git diff``
between the two tips' trees, with each rename split into its two paths.
pre-commit's own diff is three-dot, from the merge base, and would miss what a
force-push takes away: rewind past a test edit, add a fragment, and only the
fragment would show. A hook's ``files``/``exclude`` cannot make this decision
either. pre-commit drops paths that no longer exist before it matches them, so
a push that deletes a test file and adds a changelog fragment would show the
hook only the fragment, and the config's top-level ``exclude`` hides
``third_party/`` from every hook. When every changed path matches
``SKIPPABLE``, or the two trees are the same, the hook passes at once.
Anything else runs the suite: another path, a diff that fails, a pushed tip
that is not the commit checked out, or no range at all (``pre-commit run pytest
--hook-stage pre-push``, or this script run by hand).

This takes the commit the remote already has to have passed the suite. A
skipped push does not check that again: a branch that was already red, such as
a main merged before CI finished, is CI's to catch.

The suite tests the checkout, and pre-commit passes the range of only the first
ref a push names, which ``push.default=matching`` and ``--all`` list in
alphabetical order. A range whose tip is not ``HEAD`` therefore says nothing
about the tree the suite would test: a fragment-only branch named first must
not skip a suite the checked-out branch's own change needs.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

WORKERS_ENV = "BNGSIM_PREPUSH_WORKERS"
CACHE_ENV = "BNGSIM_TEST_CACHE_DIR"
FROM_ENV = "PRE_COMMIT_FROM_REF"
TO_ENV = "PRE_COMMIT_TO_REF"

# The variables that tell git which repository to use. git exports an absolute
# GIT_DIR, the main repository's .git/worktrees/<name>, to a hook run from a
# linked worktree (from the main checkout it exports none). Every git command a
# test runs in a scratch directory then goes to this repository instead: a
# fixture's `git init` and `git config user.name` set core.bare=true and the
# fixture's identity in the real .git/config. So the suite runs without them;
# the hook's own git calls, which read the push, keep them. The same list as
# python/tests/_net_corpus.py's _REPO_LOCATING, and test_pre_push_pytest.py
# holds the two equal.
REPO_LOCATING = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
    }
)

# The paths a push may change without running the suite. A full run that logged
# every file opened, listed or globbed in every Python process (2026-10-18)
# touched none of these files. It read changelog.d/ only in
# TestTheRepositorysOwnFragments, which runs ``ci/changelog.py check``, and the
# changelog-fragments hook runs that check on every push
# (test_changelog_fragments.py pins the hook). The suite reads nearly everything
# else: it runs the examples in README.md, SUPPORT_MATRIX.md and docs/, and
# checks CHANGELOG.md, CONTRIBUTING.md, .gitignore, .gitattributes and
# .github/workflows/. python/tests/test_pre_push_pytest.py pins this list
# against every tracked file.
SKIPPABLE = re.compile(
    r"^(changelog\.d/"
    r"|(LICENSE|NOTICE|ACKNOWLEDGMENTS\.md|CONTRIBUTORS\.md|REPRODUCING\.md"
    r"|\.readthedocs\.yaml|\.clang-format)$)"
)


def _git(*args: str) -> bytes | None:
    try:
        done = subprocess.run(["git", *args], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def pushed_paths() -> list[str] | None:
    """Every path whose content differs between the two tips, deleted ones included.

    ``None`` without a range from pre-commit, when its tip is not ``HEAD``, or
    when git cannot diff it. ``--no-renames`` lists a renamed file under its old
    name as well as its new.
    """
    old = os.environ.get(FROM_ENV, "").strip()
    new = os.environ.get(TO_ENV, "").strip()
    if not old or not new:
        return None
    tips = (_git("rev-parse", "HEAD^{commit}", f"{new}^{{commit}}") or b"").split()
    if len(tips) != 2 or tips[0] != tips[1]:
        return None
    out = _git("diff", "--name-only", "--no-renames", "--no-ext-diff", "-z", old, new)
    if out is None:
        return None
    return [os.fsdecode(name) for name in out.split(b"\0") if name]


def command(workers: str) -> list[str]:
    """The pytest invocation; ``0`` workers is the plain serial run."""
    cmd = [sys.executable, "-m", "pytest", "-q", "python/tests"]
    if workers != "0":
        cmd += ["-n", workers, "--dist", "worksteal"]
    return cmd


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def default_workers() -> str:
    """Cores that are not efficiency cores on Apple Silicon, else ``auto``.

    Read by level name, not by position: an Intel Mac reports one level (named
    "Standard"), and a chip can put another cluster ahead of its performance
    cores at ``perflevel0``.
    """
    if sys.platform != "darwin":
        return "auto"
    levels = _sysctl("hw.nperflevels")
    if not levels.isdigit() or int(levels) < 2:
        return "auto"
    cores = 0
    for level in range(int(levels)):
        if _sysctl(f"hw.perflevel{level}.name") == "Efficiency":
            continue
        count = _sysctl(f"hw.perflevel{level}.physicalcpu")
        if not count.isdigit():
            return "auto"
        cores += int(count)
    return str(cores) if cores > 0 else "auto"


def shared_cache_dir() -> Path | None:
    """The main checkout's test cache, for a run in one of its worktrees.

    ``None`` in the main checkout itself, where that directory is already the
    default, and wherever the main checkout cannot be found.
    """
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not common:
        return None
    common_dir = Path(common)
    main_root = common_dir.parent
    # Only an ordinary checkout's ``.git`` directory has the checkout beside it,
    # and it has to be bngsim's: a repository that vendors this tree has a
    # pyproject of its own.
    if common_dir.name != ".git":
        return None
    pyproject = main_root / "pyproject.toml"
    try:
        if 'name = "bngsim"' not in pyproject.read_text(encoding="utf-8"):
            return None
    except OSError:
        return None
    if Path.cwd().resolve() == main_root.resolve():
        return None
    return main_root / ".pytest_cache" / "d" / "bngsim"


def suite_env() -> dict[str, str]:
    """The environment the suite runs in: this one minus ``REPO_LOCATING``."""
    return {k: v for k, v in os.environ.items() if k not in REPO_LOCATING}


def run(cmd: list[str]) -> int:
    """Run the suite and return its exit code, letting Ctrl-C reach it first.

    ``subprocess.call`` kills its child outright on KeyboardInterrupt. The child
    is in the same process group and has the SIGINT already, so wait for it:
    xdist then stops its workers and prints what it had. The child gets
    :func:`suite_env`, so no test can reach the repository git handed the hook.
    """
    with subprocess.Popen(cmd, env=suite_env()) as proc:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            return proc.wait()


def main() -> int:
    paths = pushed_paths()
    if paths is not None and all(SKIPPABLE.search(path) for path in paths):
        shown = ", ".join(paths[:3]) + (f" and {len(paths) - 3} more" if len(paths) > 3 else "")
        what = f"only files it never reads ({shown})" if paths else "no files"
        print(f"pre-push: not running the suite; this push changes {what}.", flush=True)
        return 0
    workers = os.environ.get(WORKERS_ENV, "").strip() or default_workers()
    if workers != "0" and importlib.util.find_spec("xdist") is None:
        print(
            "pre-push: pytest-xdist is not installed, and without it the suite takes "
            "about 45 minutes instead of about 5. `uv sync --extra dev` installs it "
            f"(name any other extras and groups you use). {WORKERS_ENV}=0 git push "
            "runs the serial suite instead.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if not os.environ.get(CACHE_ENV, "").strip():
        shared = shared_cache_dir()
        if shared is not None:
            os.environ[CACHE_ENV] = str(shared)
    return run(command(workers))


if __name__ == "__main__":
    sys.exit(main())
