"""A vendor script's clean-destination guard must run git inside a work tree.

`scripts/vendor_*.py` all derive their paths from
``REPO_ROOT = Path(__file__).resolve().parents[2]`` — the directory that
*contains* the bngsim checkout. That is the Git repo root in the monorepo layout
these scripts were written for, but in a standalone bngsim clone it is the
directory *above* the repo and is not a work tree at all.

`vendor_nfsim.py:ensure_clean_destination` ran ``git status`` with
``cwd=REPO_ROOT``, so in a standalone checkout the documented step-3 refresh died
with::

    Command '['git', 'status', '--porcelain', '--', 'bngsim/third_party/nfsim']'
    returned non-zero exit status 128

before writing anything. The guard never checked anything; it just aborted, and
the only way past it was ``--force``, which *skips* the check outright. So the
failure mode was not a broken guard but a disabled one — the refresh still ran,
with nothing standing between it and a dirty destination.

`vendor_rulemonkey.py` had already been fixed the same way (query git from
``BNGSIM_ROOT``), which is what made this a one-of-N-sites drift rather than a
plain bug. These tests pin the invariant for the whole family.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
VENDOR_NFSIM = SCRIPTS_DIR / "vendor_nfsim.py"

pytestmark = pytest.mark.skipif(
    not VENDOR_NFSIM.exists(),
    reason="scripts/ is not in this checkout (installed package)",
)


def _load(path: Path, name: str) -> types.ModuleType:
    """Import a scripts/ module by path — scripts/ is not a package."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def _vendor_nfsim() -> types.ModuleType:
    return _load(VENDOR_NFSIM, "_vendor_nfsim_under_test")


def _vendor_scripts_with_the_guard() -> list[Path]:
    return sorted(
        path
        for path in SCRIPTS_DIR.glob("vendor_*.py")
        if "def ensure_clean_destination" in path.read_text(encoding="utf-8")
    )


class TestGuardRunsGitInsideAWorkTree:
    def test_git_is_invoked_from_a_directory_git_recognizes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _vendor_nfsim()
        calls: list[tuple[list[str], object]] = []

        def fake_run(cmd: list[str], cwd: object = None, text: bool = True) -> object:
            calls.append((cmd, cwd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(mod, "run", fake_run)
        mod.ensure_clean_destination(force=False)

        assert calls, "the guard consulted git not at all"
        cmd, cwd = calls[0]
        assert cmd[:3] == ["git", "status", "--porcelain"], cmd
        assert cwd is not None, "the guard ran git with no explicit cwd"

        # The actual defect: this directory has to be one git will answer from.
        # `rev-parse` is the layout-independent form of that question -- it
        # succeeds for the repo root and for any subdirectory of the work tree,
        # so it holds in the monorepo layout too, and fails for REPO_ROOT in a
        # standalone checkout, which is exactly the bug.
        probe = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode == 0, (
            f"the guard runs git from {cwd}, which is not inside a Git work tree "
            f"({probe.stderr.strip()}). It will abort with exit 128 before "
            "checking anything, and --force is the only way past."
        )

    def test_pathspec_is_relative_to_the_cwd_it_uses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A cwd fix that left the pathspec relative to the *other* root would
        # silently scope the status query to a path that does not exist, and the
        # guard would report clean no matter how dirty the destination was.
        mod = _vendor_nfsim()
        calls: list[tuple[list[str], object]] = []

        def fake_run(cmd: list[str], cwd: object = None, text: bool = True) -> object:
            calls.append((cmd, cwd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(mod, "run", fake_run)
        mod.ensure_clean_destination(force=False)

        cmd, cwd = calls[0]
        pathspec = cmd[-1]
        assert (Path(str(cwd)) / pathspec).resolve() == mod.VENDOR_DIR.resolve(), (
            f"pathspec {pathspec!r} resolved against cwd {cwd} is not the vendor "
            f"directory {mod.VENDOR_DIR}"
        )

    def test_dirty_destination_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _vendor_nfsim()

        def fake_run(cmd: list[str], cwd: object = None, text: bool = True) -> object:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=" M third_party/nfsim/src/NFcore/reactionClass.cpp\n", stderr=""
            )

        monkeypatch.setattr(mod, "run", fake_run)
        with pytest.raises(RuntimeError, match="uncommitted changes"):
            mod.ensure_clean_destination(force=False)

    def test_force_short_circuits_without_consulting_git(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _vendor_nfsim()

        def explode(*args: object, **kwargs: object) -> object:
            raise AssertionError("--force must not shell out to git")

        monkeypatch.setattr(mod, "run", explode)
        mod.ensure_clean_destination(force=True)


class TestNoVendorScriptRunsGitFromRepoRoot:
    """The family-wide invariant, so this cannot drift back in one script.

    `REPO_ROOT` stays legitimate for building paths and for the display form
    ``bngsim/third_party/...`` in messages. It is only ever wrong as a *git
    working directory*, which is precisely what this pins.
    """

    def test_guard_scripts_do_not_use_repo_root_as_a_git_cwd(self) -> None:
        scripts = _vendor_scripts_with_the_guard()
        assert scripts, "no scripts/vendor_*.py defines ensure_clean_destination"

        offenders = [
            path.name for path in scripts if "cwd=REPO_ROOT" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, (
            f"{', '.join(offenders)} runs a git subprocess with cwd=REPO_ROOT. "
            "REPO_ROOT is the parent of the bngsim checkout and is not a work "
            "tree in a standalone clone; use BNGSIM_ROOT."
        )

    def test_every_guard_script_defines_bngsim_root(self) -> None:
        for path in _vendor_scripts_with_the_guard():
            source = path.read_text(encoding="utf-8")
            assert "BNGSIM_ROOT" in source, (
                f"{path.name} has a clean-destination guard but no BNGSIM_ROOT to run git from"
            )


def test_vendor_dir_is_unchanged_by_the_bngsim_root_refactor() -> None:
    # BNGSIM_ROOT was introduced between REPO_ROOT and the vendored paths. The
    # paths themselves must land exactly where they did before.
    mod = _vendor_nfsim()
    assert mod.VENDOR_DIR == mod.REPO_ROOT / "bngsim" / "third_party" / "nfsim"
    assert mod.PATCHES_DIR == mod.REPO_ROOT / "bngsim" / "scripts" / "nfsim_vendor_patches"
    assert mod.VENDOR_DIR.is_dir()
    assert mod.PATCHES_DIR.is_dir()
