"""``ci/contributors.py``: every commit author is listed in CONTRIBUTORS.md.

CONTRIBUTING.md promises a contributor is added on the pull request that brings
their first contribution. Nothing checked it, and Ely F. Miller was missing for
three weeks and ten merged pull requests. These pin how the check decides who
is listed. The GitHub side runs through a stand-in ``gh`` here; contributors.yml
runs the real one on every pull request.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci" / "contributors.py"

if not SCRIPT.is_file():  # an installed wheel carries no ci/ directory
    pytest.skip("ci/contributors.py lives in the source root", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bngsim_ci_contributors", SCRIPT)
assert _spec and _spec.loader
contributors = importlib.util.module_from_spec(_spec)
# Registered before execution: ``@dataclass`` resolves annotations through
# ``sys.modules[cls.__module__]``.
sys.modules[_spec.name] = contributors
_spec.loader.exec_module(contributors)

Commit = contributors.Commit

LISTING = """\
# Contributors

Listed alphabetically by family name, with GitHub handle.

- Ada Lovelace (@alovelace)
- Grace Hopper (@GHopper)
- @quietdev
"""


def _commit(
    login="", name="Someone", email="someone@example.org", account_type="User", sha="a" * 40
):
    return Commit(sha, login, account_type if login else "", name, email)


def test_every_entry_in_the_repositorys_file_parses():
    """A list line the pattern skipped would leave its person unlisted, so the
    check would fail on their next commit; every entry has to be in a form it
    reads, whatever list marker or spacing it was typed with."""
    text = (REPO_ROOT / "CONTRIBUTORS.md").read_text(encoding="utf-8")
    entries = [line for line in text.splitlines() if line.lstrip()[:1] in ("-", "*", "+")]
    assert entries
    assert all(contributors.ENTRY.match(line) for line in entries), entries
    handles, _ = contributors.listed(text)
    assert len(handles) == len(entries)
    assert "elyfmiller" in handles


def test_the_two_forms_of_entry_parse_and_nothing_else_does():
    handles, names = contributors.listed(LISTING)
    assert handles == {"alovelace", "ghopper", "quietdev"}
    assert names == {"ada lovelace", "grace hopper"}
    for line in ("* Ada Lovelace (@alovelace)", "-  Ada (@alovelace)", "- (@alovelace)", "- Ada"):
        assert not contributors.ENTRY.match(line), line


def test_a_handle_only_entry_lists_its_handle():
    handles, names = contributors.listed(LISTING)
    assert contributors.unlisted([_commit("QuietDev")], handles, names) == []


def test_a_listed_handle_passes_whatever_its_case():
    handles, names = contributors.listed(LISTING)
    commits = [_commit("alovelace"), _commit("ghopper"), _commit("ALOVELACE")]
    assert contributors.unlisted(commits, handles, names) == []


def test_an_unlisted_handle_is_named_once_however_many_commits_it_has():
    handles, names = contributors.listed(LISTING)
    first = _commit("elyfmiller", sha="1" * 40)
    commits = [first, _commit("elyfmiller", sha="2" * 40), _commit("alovelace")]
    assert contributors.unlisted(commits, handles, names) == [first]


def test_a_listed_name_does_not_cover_an_unlisted_handle():
    """With an account, the handle decides: someone of the same name under
    another handle is another person."""
    handles, names = contributors.listed(LISTING)
    commit = _commit("lovelace2", name="Ada Lovelace")
    assert contributors.unlisted([commit], handles, names) == [commit]


def test_with_no_account_the_name_on_the_commit_decides():
    """PR #131's commits came from an email GitHub matched to no account."""
    handles, names = contributors.listed(LISTING)
    known = _commit(name="grace hopper", email="grace@example.org")
    stranger = _commit(name="Alan Turing", email="alan@example.org")
    again = _commit(name="Alan Turing", email="ALAN@example.org", sha="b" * 40)
    assert contributors.unlisted([known, stranger, again], handles, names) == [stranger]


def test_with_no_account_a_name_that_is_a_listed_handle_passes():
    """A git config user.name is often the person's handle: #131's opener
    committed elsewhere as ``happykawayigt``."""
    handles, names = contributors.listed(LISTING)
    commit = _commit(name="quietdev", email="q@example.org")
    assert contributors.unlisted([commit], handles, names) == []


def test_bot_accounts_are_not_people():
    handles, names = contributors.listed(LISTING)
    commits = [
        _commit("dependabot[bot]", account_type="Bot"),
        _commit("some-app", account_type="Bot"),
        _commit("renovate[bot]", account_type="User"),
    ]
    assert contributors.unlisted(commits, handles, names) == []


def test_check_names_who_is_missing_and_fails(capsys):
    commits = [_commit("alovelace"), _commit("elyfmiller", name="Ely F. Miller", sha="c" * 40)]
    assert contributors.check(commits, LISTING, "pull request #1") == 1
    out = capsys.readouterr().out
    assert "Ely F. Miller" in out and "@elyfmiller" in out and "cccccccccccc" in out
    assert contributors.check([_commit("alovelace")], LISTING, "pull request #1") == 0


def test_a_file_that_lists_nobody_fails_rather_than_passing_everyone(capsys):
    assert contributors.check([], "# Contributors\n\n* Ada (alovelace)\n", "x") == 1
    assert "lists nobody" in capsys.readouterr().out


def _fake_gh(tmp_path: Path, monkeypatch, body: str) -> Path:
    """A ``gh`` on PATH that records its arguments and runs ``body``."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "gh_args"
    gh = bindir / "gh"
    gh.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{log}"\n{body}\n')
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return log


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in gh is a POSIX shell script")
def test_a_pull_request_is_read_from_its_commits_endpoint(tmp_path, monkeypatch, capsys):
    rows = (
        '["1111111111111111111111111111111111111111","alovelace","User","Ada","a@x"]\n'
        '["2222222222222222222222222222222222222222","","","Grace Hopper","g@x"]'
    )
    log = _fake_gh(tmp_path, monkeypatch, f"cat <<'EOF'\n{rows}\nEOF")
    listing = tmp_path / "CONTRIBUTORS.md"
    listing.write_text(LISTING)
    monkeypatch.setattr(contributors, "CONTRIBUTORS", listing)

    assert contributors.main(["check", "--pr", "7", "--repo", "o/r"]) == 0
    args = log.read_text().splitlines()
    assert args[:3] == ["api", "--paginate", "repos/o/r/pulls/7/commits?per_page=100"]
    out = capsys.readouterr().out
    assert "2 commit(s) in pull request #7" in out
    assert "at most 250" not in out

    assert contributors.main(["check", "--all", "--repo", "o/r"]) == 0
    assert log.read_text().splitlines()[2] == "repos/o/r/commits?per_page=100"


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in gh is a POSIX shell script")
def test_a_pull_request_at_githubs_commit_limit_is_warned_about(tmp_path, monkeypatch, capsys):
    row = '["1111111111111111111111111111111111111111","alovelace","User","Ada","a@x"]'
    loop = f"i=0; while [ $i -lt 250 ]; do echo '{row}'; i=$((i+1)); done"
    _fake_gh(tmp_path, monkeypatch, loop)
    listing = tmp_path / "CONTRIBUTORS.md"
    listing.write_text(LISTING)
    monkeypatch.setattr(contributors, "CONTRIBUTORS", listing)
    assert contributors.main(["check", "--pr", "7", "--repo", "o/r"]) == 0
    assert "at most 250 commits" in capsys.readouterr().out
    # The default branch has no such limit: --all pages through every commit.
    assert contributors.main(["check", "--all", "--repo", "o/r"]) == 0
    assert "at most 250" not in capsys.readouterr().out


def test_without_gh_the_check_is_an_error_not_a_pass(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(SystemExit) as raised:
        contributors.main(["check", "--pr", "7", "--repo", "o/r"])
    assert raised.value.code == 2


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in gh is a POSIX shell script")
def test_a_github_that_cannot_be_read_is_an_error_not_a_pass(tmp_path, monkeypatch):
    _fake_gh(tmp_path, monkeypatch, "echo 'HTTP 404: Not Found' >&2; exit 1")
    with pytest.raises(SystemExit) as raised:
        contributors.main(["check", "--pr", "7", "--repo", "o/r"])
    assert raised.value.code == 2
