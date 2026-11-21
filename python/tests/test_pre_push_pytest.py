"""The pre-push hook runs the whole suite in parallel (ci/pre_push_pytest.py).

Serially the hook took 44m29s; under pytest-xdist, 4m42s with the same
result, and work stealing (``--dist worksteal``) cut a corpus-less run from
304 s to 184 s. These pin what that rests on: the hook still runs all of
``python/tests`` (nothing was dropped to get the speed), a worktree reuses the
main checkout's test cache (a cold one took 19m15s, waiting on macOS to check
each new ``.dylib``), and a venv without xdist is told so at once instead of
running for 45 minutes behind a ``Passed``.

A push skips the suite only when every path it changes, deleted ones included,
is one the suite never reads. The suite reads nearly the whole tree, docs
included, so that list is short, and it is pinned here against every tracked
file.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci" / "pre_push_pytest.py"

if not SCRIPT.is_file():  # an installed wheel carries no ci/ directory
    pytest.skip("ci/pre_push_pytest.py lives in the source root", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bngsim_ci_pre_push_pytest", SCRIPT)
assert _spec and _spec.loader
pre_push = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre_push)

SERIAL = [sys.executable, "-m", "pytest", "-q", "python/tests"]


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    *,
    have_xdist: bool,
    shared: Path | None = None,
    pushed: list[str] | None = None,
) -> tuple[int, list[list[str]], str | None]:
    """Call ``main()``; return its exit code, what it ran, and the cache env it set.

    ``pushed`` stands in for the push's changed paths. Under the hook itself the
    real push's range is in the environment, so it is never read from there.
    """
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "xdist":
            return object() if have_xdist else None
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    ran: list[list[str]] = []
    seen: list[str | None] = []

    def run(cmd: list[str]) -> int:
        ran.append(cmd)
        seen.append(os.environ.get(pre_push.CACHE_ENV))
        return 7

    monkeypatch.setattr(pre_push.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(pre_push, "run", run)
    monkeypatch.setattr(pre_push, "default_workers", lambda: "12")
    monkeypatch.setattr(pre_push, "shared_cache_dir", lambda: shared)
    monkeypatch.setattr(pre_push, "pushed_paths", lambda: pushed)
    # main() writes the cache variable into os.environ for its child; have
    # monkeypatch restore it rather than leak it into the rest of the session.
    monkeypatch.setenv(pre_push.CACHE_ENV, os.environ.get(pre_push.CACHE_ENV, ""))
    code = pre_push.main()
    return code, ran, seen[0] if seen else None


def test_with_xdist_the_whole_suite_runs_with_work_stealing(monkeypatch):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    code, ran, _ = _run_main(monkeypatch, have_xdist=True)
    assert code == 7, "main() must return the suite's own exit code"
    assert ran == [[*SERIAL, "-n", "12", "--dist", "worksteal"]]


def test_the_worker_count_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(pre_push.WORKERS_ENV, " 4 ")
    expected = [[*SERIAL, "-n", "4", "--dist", "worksteal"]]
    assert _run_main(monkeypatch, have_xdist=True)[1] == expected
    monkeypatch.setenv(pre_push.WORKERS_ENV, "")
    assert _run_main(monkeypatch, have_xdist=True)[1][0][5:7] == ["-n", "12"]


def test_zero_workers_is_the_plain_serial_run_and_needs_no_xdist(monkeypatch):
    monkeypatch.setenv(pre_push.WORKERS_ENV, "0")
    for have_xdist in (True, False):
        code, ran, _ = _run_main(monkeypatch, have_xdist=have_xdist)
        assert (code, ran) == (7, [SERIAL])


def test_without_xdist_the_push_stops_at_once_and_says_how_to_fix_it(monkeypatch, capsys):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    code, ran, _ = _run_main(monkeypatch, have_xdist=False)
    assert code == 1
    assert ran == [], "it must not fall back to a 45-minute run nobody sees"
    err = capsys.readouterr().err
    assert "pytest-xdist is not installed" in err
    assert "uv sync --extra dev" in err
    assert f"{pre_push.WORKERS_ENV}=0" in err


def test_the_suite_runs_on_the_main_checkouts_test_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    monkeypatch.delenv(pre_push.CACHE_ENV, raising=False)
    shared = tmp_path / "main" / ".pytest_cache" / "d" / "bngsim"
    assert _run_main(monkeypatch, have_xdist=True, shared=shared)[2] == str(shared)


def test_an_explicit_test_cache_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    monkeypatch.setenv(pre_push.CACHE_ENV, str(tmp_path / "mine"))
    shared = tmp_path / "main" / ".pytest_cache" / "d" / "bngsim"
    assert _run_main(monkeypatch, have_xdist=True, shared=shared)[2] == str(tmp_path / "mine")


def test_with_no_main_checkout_found_the_cache_is_the_default(monkeypatch):
    monkeypatch.delenv(pre_push.WORKERS_ENV, raising=False)
    monkeypatch.delenv(pre_push.CACHE_ENV, raising=False)
    assert _run_main(monkeypatch, have_xdist=True, shared=None)[2] == ""


def _git(cwd: Path, *args: str) -> None:
    # GIT_DIR and friends would point these at the developer's own repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


def test_a_worktree_finds_the_main_checkouts_cache(tmp_path, monkeypatch):
    main = tmp_path / "main"
    main.mkdir()
    (main / "pyproject.toml").write_text('[project]\nname = "bngsim"\n')
    _git(main, "init", "-q")
    _git(
        main,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "x",
    )
    _git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "wt" / "bngsim"))
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path / "wt" / "bngsim")
    assert pre_push.shared_cache_dir() == main.resolve() / ".pytest_cache" / "d" / "bngsim"
    # Another project's checkout, or none at all: no guess.
    (main / "pyproject.toml").write_text('[project]\nname = "pybnf"\n')
    assert pre_push.shared_cache_dir() is None
    (main / "pyproject.toml").unlink()
    assert pre_push.shared_cache_dir() is None


def test_the_main_checkout_keeps_pytests_own_route_to_its_cache(tmp_path, monkeypatch):
    """Set there, the variable would skip the ``config.cache`` route, the one
    that makes pytest write the ``.gitignore`` inside ``.pytest_cache``."""
    main = tmp_path / "main"
    main.mkdir()
    (main / "pyproject.toml").write_text('[project]\nname = "bngsim"\n')
    _git(main, "init", "-q")
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(main)
    assert pre_push.shared_cache_dir() is None


def test_the_repository_ignores_pytests_cache_directory():
    """pytest's own ``.gitignore`` there is written only if pytest creates the
    directory, and a worktree's hook run can create the main checkout's first."""
    gitignore = REPO_ROOT / ".gitignore"
    if not gitignore.is_file():
        pytest.skip(".gitignore is not in this checkout")
    assert ".pytest_cache/" in gitignore.read_text(encoding="utf-8").splitlines()


class _Sysctl:
    """A stand-in for ``subprocess.run`` answering ``sysctl -n <name>``."""

    def __init__(self, values: dict[str, str] | None) -> None:
        self.values = values
        self.asked: list[str] = []

    def __call__(self, cmd, **kwargs):
        assert cmd[:2] == ["sysctl", "-n"]
        self.asked.append(cmd[2])
        if self.values is None:
            raise FileNotFoundError(cmd[0])
        out = self.values.get(cmd[2], "")
        return subprocess.CompletedProcess(cmd, 0 if out else 1, stdout=out + "\n", stderr="")


def _levels(*levels: tuple[str, int]) -> dict[str, str]:
    values = {"hw.nperflevels": str(len(levels))}
    for i, (name, count) in enumerate(levels):
        values[f"hw.perflevel{i}.name"] = name
        values[f"hw.perflevel{i}.physicalcpu"] = str(count)
    return values


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # This machine: 16 workers took 5m27s against 12 workers' 4m42s.
        (_levels(("Performance", 12), ("Efficiency", 4)), "12"),
        # A cluster ahead of the performance cores still counts; position is not the test.
        (_levels(("Super", 6), ("Performance", 12), ("Efficiency", 4)), "18"),
        # An Intel Mac reports one "Standard" level: nothing to leave out.
        (_levels(("Standard", 8)), "auto"),
        ({}, "auto"),
        (None, "auto"),
        (_levels(("Performance", 0), ("Efficiency", 4)), "auto"),
    ],
    ids=["p-and-e", "three-levels", "intel", "no-perflevels", "no-sysctl", "zero"],
)
def test_on_a_mac_the_default_leaves_out_the_efficiency_cores(monkeypatch, values, expected):
    sysctl = _Sysctl(values)
    monkeypatch.setattr(pre_push.sys, "platform", "darwin")
    monkeypatch.setattr(pre_push.subprocess, "run", sysctl)
    assert pre_push.default_workers() == expected
    assert sysctl.asked[0] == "hw.nperflevels"


def test_elsewhere_the_default_is_xdist_auto(monkeypatch):
    sysctl = _Sysctl(_levels(("Performance", 12), ("Efficiency", 4)))
    monkeypatch.setattr(pre_push.sys, "platform", "linux")
    monkeypatch.setattr(pre_push.subprocess, "run", sysctl)
    assert pre_push.default_workers() == "auto"
    assert sysctl.asked == []


def test_run_returns_the_childs_exit_code():
    assert pre_push.run([sys.executable, "-c", "raise SystemExit(5)"]) == 5


def test_the_suite_does_not_get_the_repository_git_hands_a_worktree_hook(monkeypatch, tmp_path):
    """git exports an absolute GIT_DIR to a hook run from a linked worktree. A
    test that runs `git init` in a scratch directory under it wrote to the real
    repository and made it bare, so the suite runs without the variables that
    pick a repository, and keeps the rest of git's settings."""
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "main" / ".git" / "worktrees" / "wt"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    seen = tmp_path / "env.txt"
    probe = (
        "import os, sys; "
        "names = sorted(k for k in os.environ if k.startswith('GIT_')); "
        "open(sys.argv[1], 'w').write('\\n'.join(names))"
    )
    assert pre_push.run([sys.executable, "-c", probe, str(seen)]) == 0
    names = set(seen.read_text().split())
    assert names.isdisjoint(pre_push.REPO_LOCATING), names
    assert "GIT_CONFIG_GLOBAL" in names


def test_the_hooks_repository_variables_are_the_suites():
    """One list in three places: a name added to one and not the others would
    leave a git call there pointed at the hook's repository."""
    from _net_corpus import _REPO_LOCATING

    assert pre_push.REPO_LOCATING == _REPO_LOCATING
    gate_path = REPO_ROOT / "ci" / "sbml_suite_gate.py"
    spec = importlib.util.spec_from_file_location("bngsim_ci_sbml_suite_gate_vars", gate_path)
    assert spec and spec.loader
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert gate._GIT_REPO_VARS == _REPO_LOCATING


def test_the_push_hook_is_this_script():
    """The hook's entry is the script, and it stays a push-only hook."""
    config = REPO_ROOT / ".pre-commit-config.yaml"
    if not config.is_file():
        pytest.skip(".pre-commit-config.yaml is not in this checkout")
    hook = config.read_text(encoding="utf-8").split("- id: pytest\n")[1].split("- id: ")[0]
    assert "entry: uv run --no-sync python ci/pre_push_pytest.py" in hook
    assert "stages: [pre-push]" in hook
    # The script decides what a push may skip; pre-commit's patterns cannot see
    # deleted paths, or third_party/ past the top-level exclude.
    assert "always_run: true" in hook


# The files a push may change without running the suite, spelled out rather
# than read back from pre_push.SKIPPABLE, so a change to either is a decision.
SKIPPABLE_FILES = frozenset(
    {
        "LICENSE",
        "NOTICE",
        "ACKNOWLEDGMENTS.md",
        "CONTRIBUTORS.md",
        "REPRODUCING.md",
        ".readthedocs.yaml",
        ".clang-format",
    }
)
SKIPPABLE_DIRS = ("changelog.d/",)


@pytest.mark.parametrize(
    "path",
    [
        "python/bngsim/_model.py",
        "python/tests/conftest.py",
        "src/expression.cpp",
        "include/bngsim/model.hpp",
        "third_party/exprtk/exprtk.hpp",
        "third_party/sundials/VENDOR.json",
        "third_party/mir/LICENSE",
        "tests/data/BIOMD0000000003.xml",
        "parity_checks/bng_parity/golden/golden.json",
        "benchmarks/suites/psa/README.md",
        "harness/sbml_test_suite/SUITE_PIN.json",
        "ci/pre_push_pytest.py",
        "ci/changelog.py",
        "scripts/vendor_nfsim.py",
        "cmake/BngsimResolvePybind11.cmake",
        "CMakeLists.txt",
        "pyproject.toml",
        "uv.lock",
        "run_tests.sh",
        ".pre-commit-config.yaml",
        ".github/workflows/python-tests.yml",
        ".gitignore",
        ".gitattributes",
        "README.md",
        "SUPPORT_MATRIX.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "docs/user-guide/events.md",
        "docs/conf.py",
        # Paths nothing names yet: one added later runs the suite by default.
        "examples/new_model.py",
        "NEW_TOP_LEVEL.md",
        "LICENSES/new_dependency.txt",
    ],
)
def test_a_push_that_changes_what_the_suite_reads_runs_it(monkeypatch, path):
    assert not pre_push.SKIPPABLE.search(path)
    # Alongside a fragment too: one path the suite reads is enough.
    code, ran, _ = _run_main(monkeypatch, have_xdist=True, pushed=["changelog.d/1.fixed.md", path])
    assert (code, len(ran)) == (7, 1)


def _tracked_files() -> list[str]:
    """Every tracked path, relative to the root; skips outside a git checkout.

    Anchored on the tracked ``pyproject.toml`` rather than on the top of the work
    tree, as in ``_net_corpus.py``, for the monorepo that keeps bngsim one
    directory down.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        for args in (["--error-unmatch", "--", "pyproject.toml"], ["-z"]):
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "ls-files", *args],
                capture_output=True,
                check=True,
                env=env,
            ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout of bngsim")
    return [os.fsdecode(name) for name in out.split(b"\0") if name]


def test_the_only_tracked_files_a_push_may_skip_are_the_ones_named_here():
    """Pinned in both directions against every tracked file. A broader list
    (``docs/``, whose examples the suite runs) would skip tests a push needed; a
    narrower one only costs pushes five minutes, but it should be a decision
    taken here rather than an accident."""
    tracked = _tracked_files()
    skipped = sorted(path for path in tracked if pre_push.SKIPPABLE.search(path))
    expected = sorted(
        path for path in tracked if path in SKIPPABLE_FILES or path.startswith(SKIPPABLE_DIRS)
    )
    assert expected, "none of the skippable paths is tracked; the list has gone stale"
    assert skipped == expected


def test_a_push_of_unread_files_only_passes_without_running_the_suite(monkeypatch, capsys):
    pushed = ["changelog.d/1.fixed.md", "changelog.d/2.added.md", "LICENSE", "NOTICE"]
    code, ran, _ = _run_main(monkeypatch, have_xdist=False, pushed=pushed)
    assert (code, ran) == (0, []), "nothing the suite reads changed, xdist or not"
    out = capsys.readouterr().out
    assert "not running the suite" in out
    assert "changelog.d/1.fixed.md" in out and "and 1 more" in out


def test_a_push_whose_changes_are_unknown_runs_the_suite(monkeypatch):
    code, ran, _ = _run_main(monkeypatch, have_xdist=True, pushed=None)
    assert (code, len(ran)) == (7, 1)


def test_a_push_that_leaves_the_tree_as_it_was_passes_without_running_the_suite(
    monkeypatch, capsys
):
    """``git commit --allow-empty`` to re-run CI: the diff is between the two
    tips' trees, so an empty one means the suite would test what it tested."""
    code, ran, _ = _run_main(monkeypatch, have_xdist=True, pushed=[])
    assert (code, ran) == (0, [])
    assert "changes no files" in capsys.readouterr().out


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_a_push_that_deletes_a_file_the_suite_reads_runs_it(tmp_path, monkeypatch):
    """Why the decision is the script's and not a ``files``/``exclude`` pattern:
    pre-commit drops paths that no longer exist before matching, so this push
    (delete a test, rename a module, add a fragment) would show a pattern
    nothing but the fragment and the new name. The diff lists all four.

    changelog.d/ may skip at all only because the changelog-fragments hook runs
    the suite's one check of it on every push, which test_changelog_fragments.py
    pins (``always_run``, no ``stages:``)."""
    repo = tmp_path / "repo"
    (repo / "python" / "tests").mkdir(parents=True)
    (repo / "python" / "tests" / "test_gone.py").write_text("def test_x():\n    pass\n")
    (repo / "python" / "old_name.py").write_text("X = 1\n")
    _git(repo, "init", "-q")
    base = _commit(repo, "base")
    (repo / "python" / "tests" / "test_gone.py").unlink()
    (repo / "python" / "old_name.py").rename(repo / "python" / "new_name.py")
    (repo / "changelog.d").mkdir()
    (repo / "changelog.d" / "1.removed.md").write_text("- Gone.\n")
    head = _commit(repo, "remove")
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(repo)
    monkeypatch.setenv(pre_push.FROM_ENV, base)
    monkeypatch.setenv(pre_push.TO_ENV, head)
    assert sorted(pre_push.pushed_paths() or []) == [
        "changelog.d/1.removed.md",
        "python/new_name.py",
        "python/old_name.py",
        "python/tests/test_gone.py",
    ]
    # The same push without the deletion and the rename may skip.
    monkeypatch.setenv(pre_push.FROM_ENV, head)
    (repo / "changelog.d" / "2.removed.md").write_text("- Also gone.\n")
    monkeypatch.setenv(pre_push.TO_ENV, _commit(repo, "fragment"))
    assert pre_push.pushed_paths() == ["changelog.d/2.removed.md"]
    # A range whose tip is not the checkout says nothing about what the suite
    # would test: pre-commit passes only the first ref a push names.
    fragment_only = os.environ[pre_push.TO_ENV]
    (repo / "python" / "new_name.py").write_text("X = 2\n")
    _commit(repo, "the checked-out branch moves on")
    assert pre_push.pushed_paths() is None
    monkeypatch.setenv(pre_push.TO_ENV, "HEAD")
    assert pre_push.pushed_paths() == ["changelog.d/2.removed.md", "python/new_name.py"]
    # And a range git cannot diff is not a range with nothing in it.
    monkeypatch.setenv(pre_push.FROM_ENV, "0" * 40)
    assert pre_push.pushed_paths() is None
    monkeypatch.setenv(pre_push.FROM_ENV, head)
    monkeypatch.setenv(pre_push.TO_ENV, fragment_only + "x")
    assert pre_push.pushed_paths() is None


def test_a_force_push_that_takes_away_a_change_the_suite_reads_runs_it(tmp_path, monkeypatch):
    """The diff is between the two tips, not from their merge base as
    pre-commit's own is: rewind a branch past a test edit, add a fragment and
    force-push, and a merge-base diff shows only the fragment."""
    repo = tmp_path / "repo"
    (repo / "python" / "tests").mkdir(parents=True)
    (repo / "python" / "tests" / "test_a.py").write_text("def test_x():\n    pass\n")
    _git(repo, "init", "-q")
    base = _commit(repo, "base")
    (repo / "python" / "tests" / "test_a.py").write_text("def test_x():\n    assert 1\n")
    pushed_before = _commit(repo, "the remote has this")
    _git(repo, "checkout", "-q", "-b", "rewritten", base)
    (repo / "changelog.d").mkdir()
    (repo / "changelog.d" / "1.fixed.md").write_text("- Fixed.\n")
    rewritten = _commit(repo, "fragment on the rewound branch")
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(repo)
    # What pre-commit passes for `git push --force`: the remote's tip, and ours.
    monkeypatch.setenv(pre_push.FROM_ENV, pushed_before)
    monkeypatch.setenv(pre_push.TO_ENV, rewritten)
    assert sorted(pre_push.pushed_paths() or []) == [
        "changelog.d/1.fixed.md",
        "python/tests/test_a.py",
    ]


def test_the_test_extra_brings_xdist_so_a_synced_venv_runs_in_parallel():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    test_extra = pyproject["project"]["optional-dependencies"]["test"]
    assert any(req.startswith("pytest-xdist") for req in test_extra)
