"""ci/test_file_counts.py holds each test file's count to main's (issue #1757).

It replaced four hand-kept bounds in ci/test_floors.env: the collected floors and
skip ceilings of python/tests and parity_checks/tests. Every pull request that
added tests had to raise a floor by hand, and two at once conflicted over it.

What is pinned here:

* the counts read from a JUnit report agree with pytest's own summary, on a
  real pytest run, including a module that skips at import and an xfail;
* the comparison's rules, at their edges: an untouched file may not lose a test,
  a touched, deleted or new one may, skips in untouched files may rise by the
  tolerance and not one more, and the label accepts a drop;
* the choice of main's baseline: main's own runs only (not a fork's ``main``),
  unexpired, an ancestor of the commit under test, and the distinction between
  "main has never uploaded one" (the run that adds this) and "none usable";
* the guard end to end, with GitHub faked, including the check of the reports
  against the summaries' totals;
* the wiring in python-tests.yml that the guard depends on.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ci_workflow import REPO_ROOT, WORKFLOWS, strip_comments  # noqa: E402

SCRIPT = REPO_ROOT / "ci" / "test_file_counts.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="ci/ not in this checkout")

if SCRIPT.is_file():
    sys.path.insert(0, str(SCRIPT.parent))
    import test_file_counts as tfc  # noqa: E402
else:  # pragma: no cover
    tfc = None


def _c(collected: int, skipped: int = 0) -> dict[str, int]:
    return {"collected": collected, "skipped": skipped}


# --- counting --------------------------------------------------------------

_SUITE = {
    "pkg/tests/test_a.py": """
import pytest

class TestK:
    @pytest.mark.parametrize("x", [1, 2, 3])
    def test_p(self, x):
        pass

def test_skip():
    pytest.skip("declared")

@pytest.mark.xfail(strict=False)
def test_xfail():
    assert 0

@pytest.mark.xfail(strict=False)
def test_xpass():
    pass
""",
    "pkg/tests/test_b.py": """
import pytest
pytest.importorskip("no_such_module_for_1757")

def test_1():
    pass

def test_2():
    pass
""",
    "pkg/tests/sub/test_c.py": """
def test_c():
    pass
""",
}


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """A real pytest run over a small suite: its JUnit report and its summary line."""
    root = tmp_path_factory.mktemp("suite")
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    for rel, body in _SUITE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "pkg/tests",
            "--junitxml=r.xml",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    summary = [
        ln for ln in done.stdout.splitlines() if re.search(r"\d+ passed.* in [0-9.]+s", ln)
    ][-1]
    return root, root / "r.xml", summary


def test_counts_match_pytests_own_summary(real_run):
    root, report, summary = real_run

    def n(word: str) -> int:
        m = re.search(rf"(\d+) {word}\b", summary)
        return int(m.group(1)) if m else 0

    counts = tfc.count_reports([report], root)
    collected, skipped = tfc.totals(counts)
    assert collected == n("passed") + n("skipped") + n("xfailed") + n("xpassed")
    assert skipped == n("skipped")


def test_counts_land_on_their_files(real_run):
    root, report, _ = real_run
    assert tfc.count_reports([report], root) == {
        "pkg/tests/sub/test_c.py": _c(1),
        # 3 parametrized + skip + xfail + xpass; the xfail is not a skip.
        "pkg/tests/test_a.py": _c(6, 1),
        # Skipped at import: one item, as in pytest's summary.
        "pkg/tests/test_b.py": _c(1, 1),
    }


def test_reports_are_summed(real_run):
    root, report, _ = real_run
    once = tfc.count_reports([report], root)
    twice = tfc.count_reports([report, report], root)
    assert twice == {f: _c(2 * c["collected"], 2 * c["skipped"]) for f, c in once.items()}


def test_a_name_that_is_no_file_keeps_its_name(tmp_path):
    assert tfc.file_of("nowhere.test_x.TestY", tmp_path) == "nowhere.test_x.TestY"


def test_the_longest_prefix_that_is_a_file_wins(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "test_x.py").write_text("")
    assert tfc.file_of("a.test_x.TestY", tmp_path) == "a/test_x.py"


# --- comparing -------------------------------------------------------------

BASE = {"t/test_a.py": _c(10, 1), "t/test_b.py": _c(5), "t/test_gone.py": _c(3)}


def test_the_same_counts_pass():
    assert tfc.compare(BASE, BASE, set()) == ([], [])


def test_one_test_lost_in_an_untouched_file_fails():
    head = {**BASE, "t/test_b.py": _c(4)}
    errors, _ = tfc.compare(BASE, head, set())
    assert len(errors) == 1
    assert "t/test_b.py: 5 -> 4" in errors[0]
    assert tfc.DROP_LABEL in errors[0]


def test_an_untouched_file_that_vanished_fails():
    head = {k: v for k, v in BASE.items() if k != "t/test_gone.py"}
    errors, _ = tfc.compare(BASE, head, set())
    assert errors and "t/test_gone.py: 3 -> 0" in errors[0]


@pytest.mark.parametrize("path", ["t/test_b.py", "t/test_gone.py"])
def test_a_touched_or_deleted_file_may_lose_tests(path):
    head = {**BASE, path: _c(0)}
    del head[path]
    errors, notes = tfc.compare(BASE, head, {path})
    assert errors == []
    assert any(path in n for n in notes)


def test_a_new_file_and_more_tests_pass_silently():
    head = {**BASE, "t/test_b.py": _c(50), "t/test_new.py": _c(7, 7)}
    assert tfc.compare(BASE, head, set()) == ([], [])


def test_the_label_accepts_a_drop_and_says_so():
    head = {**BASE, "t/test_b.py": _c(4)}
    errors, notes = tfc.compare(BASE, head, set(), drops_allowed=True)
    assert errors == []
    assert any(tfc.DROP_LABEL in n for n in notes)


def test_skips_may_rise_by_the_tolerance_and_not_one_more():
    tol = tfc.SKIP_TOLERANCE
    at = {**BASE, "t/test_a.py": _c(10, 1 + tol)}
    errors, notes = tfc.compare(BASE, at, set())
    assert errors == [] and notes
    past = {**BASE, "t/test_a.py": _c(10, 1 + tol + 1)}
    errors, _ = tfc.compare(BASE, past, set())
    assert len(errors) == 1 and "skip" in errors[0]


def test_skips_are_summed_over_untouched_files():
    tol = tfc.SKIP_TOLERANCE
    head = {**BASE, "t/test_a.py": _c(10, 1 + tol), "t/test_b.py": _c(5, 1)}
    errors, _ = tfc.compare(BASE, head, set())
    assert errors


def test_skips_in_a_touched_file_do_not_count():
    head = {**BASE, "t/test_a.py": _c(10, 10 + tfc.SKIP_TOLERANCE)}
    assert tfc.compare(BASE, head, {"t/test_a.py"})[0] == []


def test_the_label_does_not_accept_new_skips():
    head = {**BASE, "t/test_a.py": _c(10, 2 + tfc.SKIP_TOLERANCE)}
    assert tfc.compare(BASE, head, set(), drops_allowed=True)[0]


# --- main's baseline -------------------------------------------------------

REPO = "lanl/bngsim"
NAME = tfc.ARTIFACT_PREFIX + "ubuntu-latest" if tfc else ""


def _art(art_id, sha, created, *, branch="main", fork=False, expired=False):
    return {
        "id": art_id,
        "expired": expired,
        "created_at": created,
        "workflow_run": {
            "head_branch": branch,
            "head_sha": sha,
            "repository_id": 1,
            "head_repository_id": 2 if fork else 1,
        },
    }


class FakeApi:
    """``gh api`` with canned answers; records what was asked."""

    def __init__(self, artifacts, ancestors=(), counts=None, labels=(), push_labels=()):
        self.artifacts = artifacts
        self.ancestors = set(ancestors)
        self.counts = counts or {}
        self.labels = list(labels)
        self.push_labels = list(push_labels)
        self.calls: list[str] = []

    def __call__(self, path: str) -> bytes:
        self.calls.append(path)
        if path.startswith(f"repos/{REPO}/actions/artifacts?"):
            return json.dumps({"artifacts": self.artifacts}).encode()
        m = re.fullmatch(rf"repos/{REPO}/compare/(\w+)\.\.\.(\w+)", path)
        if m:
            return json.dumps(
                {"status": "ahead" if m.group(1) in self.ancestors else "diverged"}
            ).encode()
        m = re.fullmatch(rf"repos/{REPO}/actions/artifacts/(\d+)/zip", path)
        if m:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr(
                    tfc.COUNTS_FILE,
                    json.dumps({"sha": "x", "files": self.counts[int(m.group(1))]}),
                )
            return buf.getvalue()
        if re.fullmatch(rf"repos/{REPO}/pulls/\d+", path):
            return json.dumps({"labels": [{"name": n} for n in self.labels]}).encode()
        if re.fullmatch(rf"repos/{REPO}/commits/\w+/pulls", path):
            return json.dumps([{"labels": [{"name": n} for n in self.push_labels]}]).encode()
        raise AssertionError(f"unexpected gh api {path}")


def test_no_artifact_on_main_ever_is_the_bootstrap_case():
    assert tfc.pick_baseline(FakeApi([]), REPO, NAME, "head") is None


def test_a_forks_main_and_a_branch_are_not_main():
    arts = [_art(1, "aaa", "2026-11-14", fork=True), _art(2, "bbb", "2026-11-14", branch="fix/x")]
    assert tfc.pick_baseline(FakeApi(arts, ancestors={"aaa", "bbb"}), REPO, NAME, "head") is None


def test_the_newest_ancestor_is_chosen():
    arts = [
        _art(1, "old", "2026-11-12"),
        _art(2, "newer", "2026-11-13"),
        _art(3, "newest", "2026-11-14"),  # main moved on after the merge ref was made
    ]
    api = FakeApi(arts, ancestors={"old", "newer"})
    assert tfc.pick_baseline(api, REPO, NAME, "head")["id"] == 2


def test_the_head_itself_is_its_own_ancestor():
    arts = [_art(1, "head", "2026-11-14")]
    api = FakeApi(arts)
    assert tfc.pick_baseline(api, REPO, NAME, "head")["id"] == 1
    assert not any("/compare/" in c for c in api.calls)


def test_expired_artifacts_are_skipped_and_all_expired_fails():
    arts = [_art(1, "a", "2026-11-14", expired=True), _art(2, "b", "2026-11-13")]
    assert tfc.pick_baseline(FakeApi(arts, ancestors={"a", "b"}), REPO, NAME, "head")["id"] == 2
    with pytest.raises(RuntimeError, match="expired"):
        tfc.pick_baseline(FakeApi([arts[0]], ancestors={"a"}), REPO, NAME, "head")


def test_no_ancestor_among_mains_baselines_fails():
    arts = [_art(1, "a", "2026-11-14")]
    with pytest.raises(RuntimeError, match="merge main"):
        tfc.pick_baseline(FakeApi(arts), REPO, NAME, "head")


def test_labels_of_a_pull_request_and_of_a_push():
    api = FakeApi([], labels=["drops tests"], push_labels=["bug"])
    assert tfc.labels_of(api, REPO, 12, "sha") == {"drops tests"}
    assert tfc.labels_of(api, REPO, 0, "sha") == {"bug"}


# --- the guard end to end --------------------------------------------------

_REPORT = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="t.test_a" name="test_1"/>
<testcase classname="t.test_a" name="test_2"/>
<testcase classname="t.test_b" name="test_1"><skipped type="pytest.skip" message="x"/></testcase>
<testcase classname="t.test_b" name="test_2"><skipped type="pytest.xfail" message="x"/></testcase>
</testsuite></testsuites>"""


@pytest.fixture
def guard(tmp_path, monkeypatch):
    """Run ``guard`` against a fake GitHub; return (exit code, stdout, counts written)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tfc, "REPO_ROOT", tmp_path)
    (tmp_path / "t").mkdir()
    for name in ("test_a.py", "test_b.py"):
        (tmp_path / "t" / name).write_text("")
    (tmp_path / "logs" / "shard-1").mkdir(parents=True)
    (tmp_path / "logs" / "shard-1" / "r.xml").write_text(_REPORT)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))

    def run(api, *, changed=(), expect=(4, 1)):
        monkeypatch.setattr(tfc, "gh_api", api)
        monkeypatch.setattr(tfc, "changed_files", lambda base, head: set(changed))
        argv = ["guard", "--leg", "ubuntu-latest", "--reports", "logs", "--out", "c.json"]
        argv += ["--repo", REPO, "--sha", "head", "--pr", "7"]
        if expect:
            argv += ["--expect-collected", str(expect[0]), "--expect-skipped", str(expect[1])]
        code = tfc.main(argv)
        return code, json.loads((tmp_path / "c.json").read_text())

    return run


def test_guard_passes_and_writes_its_counts(guard, capsys):
    base = {"t/test_a.py": _c(2), "t/test_b.py": _c(2, 1)}
    api = FakeApi([_art(1, "base", "2026-11-14")], ancestors={"base"}, counts={1: base})
    code, written = guard(api)
    assert code == 0, capsys.readouterr().out
    assert written == {"sha": "head", "files": base}


def test_guard_fails_on_a_lost_test(guard, capsys):
    base = {"t/test_a.py": _c(3), "t/test_b.py": _c(2, 1)}
    api = FakeApi([_art(1, "base", "2026-11-14")], ancestors={"base"}, counts={1: base})
    code, _ = guard(api)
    out = capsys.readouterr().out
    assert code == 1
    assert "::error::tests stopped being collected" in out and "t/test_a.py: 3 -> 2" in out


def test_guard_honours_the_label(guard, capsys):
    base = {"t/test_a.py": _c(3), "t/test_b.py": _c(2, 1)}
    api = FakeApi(
        [_art(1, "base", "2026-11-14")],
        ancestors={"base"},
        counts={1: base},
        labels=[tfc.DROP_LABEL],
    )
    assert guard(api)[0] == 0


def test_guard_allows_a_loss_in_a_changed_file(guard):
    base = {"t/test_a.py": _c(3), "t/test_b.py": _c(2, 1)}
    api = FakeApi([_art(1, "base", "2026-11-14")], ancestors={"base"}, counts={1: base})
    assert guard(api, changed={"t/test_a.py"})[0] == 0


def test_guard_bootstraps_with_a_warning(guard, capsys):
    code, _ = guard(FakeApi([]))
    assert code == 0
    assert "::warning::main has no" in capsys.readouterr().out


def test_guard_fails_when_no_baseline_is_usable(guard, capsys):
    code, _ = guard(FakeApi([_art(1, "a", "2026-11-14")]))
    assert code == 1
    assert "::error::" in capsys.readouterr().out


def test_guard_fails_when_the_reports_disagree_with_the_summaries(guard, capsys):
    code, _ = guard(FakeApi([]), expect=(5, 1))
    assert code == 1
    assert "JUnit reports hold 4 tests" in capsys.readouterr().out


# --- the wiring in python-tests.yml ----------------------------------------


@pytest.fixture(scope="module")
def workflow() -> str:
    path = WORKFLOWS / "python-tests.yml"
    if not path.is_file():
        pytest.skip("python-tests.yml not in this checkout")
    return strip_comments(path.read_text())


def _job(text: str, job: str) -> str:
    m = re.search(rf"^  {job}:\n(.*?)(?=^  [\w-]+:\n|\Z)", text, re.M | re.S)
    assert m, job
    return m.group(1)


def test_the_counts_job_runs_the_guard_and_uploads_mains_baseline(workflow):
    job = _job(workflow, "python-tests-counts")
    assert "ci/test_file_counts.py guard" in job
    assert "--expect-collected" in job and "--expect-skipped" in job
    assert f"name: {tfc.ARTIFACT_PREFIX}${{{{ matrix.label }}}}" in job
    assert "path: test-file-counts.json" in job
    # Main's runs only: pull requests' uploads would crowd main's out of the
    # listing (100 per page, every branch) that pick_baseline searches.
    assert "github.ref == 'refs/heads/main'" in job
    assert "actions: read" in job and "pull-requests: read" in job


def test_the_guard_reads_both_suites_reports(workflow):
    shards = _job(workflow, "python-tests")
    assert re.search(
        r"pytest -p no:cacheprovider python/tests \\\n(?:.*\\\n)*?\s+--junitxml=python-tests-",
        shards,
    )
    assert re.search(
        r"pytest -p no:cacheprovider parity_checks/tests \\\n\s+--junitxml=parity-tests-", shards
    )
    assert "parity-tests-${{ matrix.leg.label }}-${{ matrix.shard }}.xml" in shards
    counts = _job(workflow, "python-tests-counts")
    # Both shards run the whole parity suite: one report, not two.
    assert "name: parity-tests-log-${{ matrix.label }}-1" in counts
    assert "parity-tests-log-${{ matrix.label }}-?" not in counts


def test_the_suite_bounds_are_gone_from_the_floors_file():
    text = (REPO_ROOT / "ci" / "test_floors.env").read_text()
    assert not re.search(r"^(PYTEST|PARITY)_", text, re.M)
