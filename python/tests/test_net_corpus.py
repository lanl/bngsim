"""``_net_corpus.py`` splits the ``.net`` sweeps' inputs into tracked and local.

Three modules used to sweep "every ``.net`` file in the tree" with three copies
of one rglob. Two said the sweep found the committed fixtures; on a machine that
had run the benchmark and parity suites it also found 1675 gitignored files, the
copies had drifted, and nothing labelled which kind a failing case was. The
helper's docstring has the inventory and the reason the local half is kept.

The split is tested on a throwaway git repo built to carry every shape the real
tree has: tracked fixtures, a gitignored corpus, untracked sweep output, a
nested clone, build and cache directories, a tracked file deleted from disk, and
a checkout that itself lives under a directory named ``build``. Then a static
guard keeps a fourth module from growing its own rglob again.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _net_corpus  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent

#: Modules allowed to walk the tree for ``.net`` files themselves, and why each
#: is not a sweep of the corpus. Anything else that does, through ``rglob`` or a
#: ``**`` glob, is a fourth copy. A plain ``glob`` of one directory is not a
#: sweep, and an ``os.walk`` is not detected.
OWN_GLOBS = {
    "_net_corpus.py": "the one place the sweep's corpus is listed",
    "test_curated_benchmark_corpus.py": "scoped to two named suites, results/ excluded",
    "test_net_reader_param_expressions.py": "scoped to tests/data and benchmarks/models/net, "
    "which are fully tracked",
}

needs_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="not a git checkout: git is not on PATH"
)


def _git(cwd: Path, *args: str) -> str:
    """Git in ``cwd`` and nowhere else.

    Without ``GIT_*`` from the environment, so an exported ``GIT_DIR`` cannot
    turn the fixture's commit into a commit on the developer's own repository.
    Without the user's hooks or signing, so a global ``core.hooksPath`` does not
    run on a throwaway commit.
    """
    config = {
        "user.email": "t@example.com",
        "user.name": "t",
        "commit.gpgsign": "false",
        "core.hooksPath": os.devnull,
        "core.excludesFile": os.devnull,
    }
    flags = [arg for key, value in config.items() for arg in ("-c", f"{key}={value}")]
    return subprocess.run(
        ["git", *flags, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=_net_corpus.git_env(),
    ).stdout


def _touch(root: Path, rel: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("begin parameters\nend parameters\n", encoding="utf-8")
    return path


def _make_repo(base: Path, subdir: str = "") -> tuple[Path, list[Path], list[Path]]:
    """A git repo under a directory named ``build``, carrying every shape.

    With ``subdir``, the bngsim tree is that subdirectory of a larger repo, as
    in the upstream monorepo, beside a sibling project with ``.net`` files of
    its own. Returns the bngsim root, the files git tracks there, and the
    untracked ones a sweep should see.
    """
    top = base / "build" / "checkout"
    root = top / subdir if subdir else top
    root.mkdir(parents=True)
    _git(top, "init", "-q", "--template=")
    (root / ".gitignore").write_text("benchmarks/suites/*/nets/\n", encoding="utf-8")
    _touch(root, "pyproject.toml")
    if subdir:
        _touch(top, "sibling_project/tests/not_bngsims.net")
    tracked = [
        _touch(root, "benchmarks/models/net/ode/a.net"),
        _touch(root, "tests/data/b.net"),
        _touch(root, "parity_checks/fixtures/c.net"),
    ]
    gone = _touch(root, "tests/data/deleted_from_disk.net")
    _git(top, "add", "-A")
    _git(top, "commit", "-q", "-m", "fixtures")
    gone.unlink()

    local = [
        _touch(root, "benchmarks/suites/ode_fullnet/nets/gitignored.net"),
        _touch(root, "parity_checks/bng_parity/runs/2026/sweep_output.net"),
    ]
    clone = root / "parity_checks" / "bng_parity" / ".sources_cache" / "upstream@abc"
    clone.mkdir(parents=True)
    _git(clone, "init", "-q", "--template=")
    local.append(_touch(clone, "models/in_a_nested_clone.net"))
    _touch(root, "benchmarks/build/generated.net")
    _touch(root, "tests/.pytest_cache/cached.net")
    _touch(root, "elsewhere/outside_the_roots.net")

    return root, sorted(tracked), sorted(local)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root, tracked, local = _make_repo(tmp_path)
    monkeypatch.setattr(_net_corpus, "ROOT", root)
    return root, tracked, local


@needs_git
class TestTheSplit:
    def test_tracked_is_what_git_tracks_and_local_is_the_rest(self, repo):
        root, tracked, local = repo
        got_tracked, got_local, git = _net_corpus._split()
        assert git is True
        assert got_tracked == tracked
        assert got_local == local

    def test_a_file_in_a_nested_clone_is_local(self, repo):
        """bng_parity clones upstream repos into ``.sources_cache``. Git does not
        list a nested repository's files, so they are untracked here, which is
        what they are."""
        _, got_local, _ = _net_corpus._split()
        assert any(p.name == "in_a_nested_clone.net" for p in got_local)

    def test_build_and_cache_directories_are_matched_below_the_root(self, repo):
        """The copies tested ``"build" in p.parts`` on the absolute path, so a
        checkout under a directory named ``build`` swept nothing at all. The
        fixture's checkout is under one."""
        root, tracked, _ = repo
        assert "build" in root.parts
        got_tracked, got_local, _ = _net_corpus._split()
        assert got_tracked == tracked
        names = {p.name for p in got_tracked + got_local}
        assert "generated.net" not in names and "cached.net" not in names

    def test_an_exported_git_dir_neither_redirects_the_split_nor_takes_a_commit(
        self, tmp_path, monkeypatch
    ):
        """With ``GIT_DIR`` and ``GIT_WORK_TREE`` exported to another repository,
        git ignores ``-C``. The helper then listed that repository's index, and
        the fixture's commit landed on its current branch."""
        victim = tmp_path / "victim"
        victim.mkdir()
        _git(victim, "init", "-q", "--template=")
        _touch(victim, "kept.txt")
        _git(victim, "add", "-A")
        _git(victim, "commit", "-q", "-m", "the developer's work")
        monkeypatch.setenv("GIT_DIR", str(victim / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(victim))

        root, tracked, local = _make_repo(tmp_path / "fixture")
        monkeypatch.setattr(_net_corpus, "ROOT", root)
        got_tracked, got_local, git = _net_corpus._split()

        assert _git(victim, "rev-list", "--count", "HEAD").strip() == "1"
        assert (git, got_tracked, got_local) == (True, tracked, local)

    def test_a_tree_inside_another_repository_falls_back(self, tmp_path, monkeypatch):
        """A tree with no ``.git`` of its own, sitting inside some other repo,
        such as an export unpacked under a home directory kept in git. That
        repo tracks none of these files. Calling them all local would empty
        TRACKED, so this falls back as if there were no git at all."""
        outer = tmp_path / "home"
        outer.mkdir()
        _git(outer, "init", "-q", "--template=")
        _touch(outer / "export", "pyproject.toml")
        on_disk = sorted(
            [
                _touch(outer / "export", "tests/data/a.net"),
                _touch(outer / "export", "benchmarks/b.net"),
            ]
        )
        monkeypatch.setattr(_net_corpus, "ROOT", outer / "export")
        assert _net_corpus._split() == (on_disk, [], False)

    def test_a_subtree_of_a_monorepo_still_splits(self, tmp_path, monkeypatch):
        """Upstream keeps bngsim as ``bngsim/`` inside a monorepo, so ROOT is
        below the top of the work tree. The split must hold there, with paths
        relative to ROOT and none of the sibling project's files."""
        root, tracked, local = _make_repo(tmp_path, subdir="bngsim")
        monkeypatch.setattr(_net_corpus, "ROOT", root)
        assert _net_corpus._split() == (tracked, local, True)

    def test_without_git_everything_on_disk_counts_as_tracked(self, repo, monkeypatch):
        """No split is possible, so this falls back to what the rglob always
        assumed, and says so through ``GIT``."""
        root, tracked, local = repo
        monkeypatch.setattr(_net_corpus, "_git_tracked", lambda: None)
        got_tracked, got_local, git = _net_corpus._split()
        assert git is False
        assert got_local == []
        assert got_tracked == sorted(tracked + local)


class TestTheParams:
    def test_local_cases_are_labelled_and_tracked_cases_are_not(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_net_corpus, "ROOT", tmp_path)
        monkeypatch.setattr(_net_corpus, "TRACKED", [tmp_path / "tests" / "a.net"])
        monkeypatch.setattr(_net_corpus, "LOCAL", [tmp_path / "benchmarks" / "b.net"])
        ids = [p.id for p in _net_corpus.sweep_params()]
        assert ids == ["tests/a.net", "local:benchmarks/b.net"]

    def test_nothing_to_sweep_skips_under_a_declared_reason(self, monkeypatch, skip_audit):
        """An empty parameter set skips under pytest's own wording, which the
        strict skip audit rejects. The placeholder must carry a declared one."""
        monkeypatch.setattr(_net_corpus, "TRACKED", [])
        monkeypatch.setattr(_net_corpus, "LOCAL", [])
        (param,) = _net_corpus.sweep_params()
        (mark,) = param.marks
        assert mark.name == "skip"
        assert skip_audit.tier_of(mark.kwargs["reason"]) is not None


@needs_git
def test_in_this_checkout_tracked_is_exactly_git_ls_files():
    """Against the real tree: every tracked ``.net`` under the roots, and nothing
    else. In CI, LOCAL is empty, so this is the whole sweep there."""
    if not _net_corpus.GIT:
        pytest.skip("not a git checkout: the helper could not run git ls-files")
    out = subprocess.run(
        ["git", "-C", str(_net_corpus.ROOT), "ls-files", "-z", "--", *_net_corpus.ROOTS],
        capture_output=True,
        check=True,
        env=_net_corpus.git_env(),
    ).stdout.decode("utf-8")
    expected = {p for p in out.split("\0") if p.endswith(".net")}
    got = {p.relative_to(_net_corpus.ROOT).as_posix() for p in _net_corpus.TRACKED}
    assert got == {p for p in expected if (_net_corpus.ROOT / p).is_file()}
    assert len(got) > 100, f"only {len(got)} tracked .net files; the scan went blind"
    assert not {p.relative_to(_net_corpus.ROOT).as_posix() for p in _net_corpus.LOCAL} & got


def test_no_other_module_rolls_its_own_sweep():
    """Three copies of one rglob had already drifted (one swept ``.pytest_cache``)
    before they were folded into the helper. A fourth would drift the same way."""
    own = sorted(
        path.name
        for path in TESTS_DIR.glob("*.py")
        if re.search(
            r"""\.rglob\(\s*["']\*\.net["']|glob\(\s*[^)]*\*\*/\*\.net""",
            path.read_text("utf-8"),
        )
    )
    assert "_net_corpus.py" in own, "the scan no longer sees the helper's own glob"
    extra = [name for name in own if name not in OWN_GLOBS]
    assert not extra, (
        f"{extra} glob .net files themselves. Sweep _net_corpus.TRACKED / LOCAL / "
        "sweep_params() instead, or add the module to OWN_GLOBS with the reason "
        "its glob is not a corpus sweep."
    )
