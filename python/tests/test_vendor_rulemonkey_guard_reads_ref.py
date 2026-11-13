"""The RuleMonkey ExprTk guard judges the commit it vendors, not the checkout.

``scripts/vendor_rulemonkey.py`` exports the tree of ``--ref``, but its ExprTk
drift guard read RuleMonkey's ``third_party/bngsim_expr`` copy and ``VENDOR`` pin
from the candidate's working tree. With the checkout on one branch and ``--ref``
naming another, the guard passed or failed on files the export never used, and
the refresh recorded the checkout's pin as the vendored commit's.

That is the state a vendor refresh is in while an upstream re-sync is under
review: the candidate sits on the re-sync branch, and ``--ref main`` names a
commit whose copy is still the old one. Found by review while vendoring
RuleMonkey 3.13.0: ``--summary --ref main`` printed the branch's pin as
"matches current BNGsim tree" beside main's commit.

Each test builds a two-commit RuleMonkey stand-in whose commits hold a matching
and a stale copy of this tree's ``EXPRTK_SYNC_FILES``, checks one of them out,
and asks the guard about the other.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _net_corpus import _REPO_LOCATING, git_env  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_RULEMONKEY = REPO_ROOT / "scripts" / "vendor_rulemonkey.py"

pytestmark = pytest.mark.skipif(
    not VENDOR_RULEMONKEY.exists(),
    reason="scripts/ is not in this checkout (installed package)",
)


def _load_vendor_rulemonkey():
    name = "_vendor_rulemonkey_for_guard_reads_ref"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, VENDOR_RULEMONKEY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.excludesFile=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            *args,
        ],
        capture_output=True,
        check=True,
        text=True,
        env=git_env(),
    ).stdout.strip()


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """A RuleMonkey stand-in: commit ``synced`` matches this tree, ``stale`` does not."""
    # The script's own git calls inherit the environment.
    for name in _REPO_LOCATING:
        monkeypatch.delenv(name, raising=False)
    vendor = _load_vendor_rulemonkey()
    repo = tmp_path / "rulemonkey"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--template=")

    def write_copy(pin: str, mangle: bool) -> None:
        for rulemonkey_rel, bngsim_path in vendor.EXPRTK_SYNC_FILES.items():
            out = repo / rulemonkey_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            data = bngsim_path.read_bytes()
            if mangle and rulemonkey_rel.endswith("expression.cpp"):
                data += b"\n// stale\n"
            out.write_bytes(data)
        note = repo / vendor.RULEMONKEY_EXPRTK_VENDOR_NOTE
        note.write_text(f"{vendor.PIN_PREFIX}{pin}\n")

    write_copy("a" * 40, mangle=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "stale copy")
    stale = _git(repo, "rev-parse", "HEAD")

    write_copy("b" * 40, mangle=False)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "synced copy")
    synced = _git(repo, "rev-parse", "HEAD")
    return vendor, repo, {"stale": stale, "synced": synced}


@pytest.mark.parametrize(("checked_out", "vendored"), [("synced", "stale"), ("stale", "synced")])
def test_the_guard_reads_the_vendored_commit(stand_in, checked_out, vendored):
    vendor, repo, commits = stand_in
    _git(repo, "checkout", "--quiet", commits[checked_out])

    drift = vendor.exprtk_sync_drift(repo, commits[vendored])
    if vendored == "stale":
        assert drift == ["third_party/bngsim_expr/src/expression.cpp != bngsim/src/expression.cpp"]
        with pytest.raises(RuntimeError, match="has drifted from this BNGsim tree"):
            vendor.verify_exprtk_sync(repo, commits[vendored])
    else:
        assert drift == []
        assert vendor.verify_exprtk_sync(repo, commits[vendored]) == "b" * 40


def test_a_sync_file_the_commit_lacks_is_drift(stand_in):
    vendor, repo, commits = stand_in
    missing = "third_party/bngsim_expr/include/bngsim/dual.hpp"
    _git(repo, "checkout", "--quiet", commits["synced"])
    _git(repo, "rm", "--quiet", missing)
    _git(repo, "commit", "--quiet", "-m", "copy without dual.hpp")
    without = _git(repo, "rev-parse", "HEAD")

    assert vendor.exprtk_sync_drift(repo, without) == [f"{missing} is missing at {without}"]
