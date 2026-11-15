#!/usr/bin/env python3
"""Hold each test file's count to main's, so no count has to be kept by hand (#1757).

    python3 ci/test_file_counts.py count REPORT.xml [REPORT.xml ...] --out counts.json
    python3 ci/test_file_counts.py compare BASE.json HEAD.json --changed FILES.txt
    python3 ci/test_file_counts.py guard --leg LABEL --reports DIR --out counts.json

This replaces four hand-kept numbers in ci/test_floors.env: the collected floors
and skip ceilings of python/tests and parity_checks/tests. Each was a copy of a
count CI measures on every run, so it went stale by design, and #584's stale
check made every pull request that added tests edit the same line. Two such
pull requests conflicted there, and the second had to redo the sum by hand
(#1757). The four bounds moved 31 times in the first half of November 2026.

What the bounds were for is unchanged. A test that stops being collected, or
turns into a skip, leaves pytest green with a smaller denominator (#28/#36): a
conftest that deselects, a module that skips at import because an extra did not
install, a shard split that drops a file. The comparison here asks that question
of each file, against the same leg's counts on main:

* A test file the change does not touch must collect at least as many tests as
  it does on main. A file the change edits, renames or deletes may lose tests;
  that is in the diff for a reviewer to read. A file that is new is not judged.
* Across the untouched files, skips may rise by at most SKIP_TOLERANCE.

That is finer than the floors were: one lost test in an untouched file fails,
where the floor let up to its slack (500 for python/tests) go. A deliberate
drop in an untouched file, for example a parametrize list that shrinks because
its data in src/ changed, is acknowledged with the pull request label in
DROP_LABEL, after which the failed job is re-run; so is a rise in skips. The
label counts on any pull request merged since the baseline as well as on the
one under test, so a drop labelled where it was made is not charged to the
changes merged after it.

Main's counts come from the artifact the workflow uploads from a run on main,
and only when this guard passed there, so a loss main took stays red, on main
and on every pull request, until someone acts: label the pull request that
made it (or, with none to blame, the pull request that accepts it) and re-run.
That is what the floors did. The baseline is the nearest of main's recent ones
that is an ancestor of the commit under test, and the files the change touches
are ``git diff --no-renames`` between the two, so a main that moved on since the
baseline was taken is part of "the change" and is never misread as a loss. A
pull request with no baseline fails, except the one that brings this script in;
a run on main with none passes and uploads one, so main recovers by itself.

A known limit: skips are held to a baseline that moves with main, so each
merge may add up to SKIP_TOLERANCE skips in untouched files and the next
baseline takes them in. There is no absolute cap, as the skip ceilings it
replaced had.

Counts are read from pytest's ``--junitxml`` reports. A test is one
``<testcase>``. One that holds a ``<skipped>`` element that is not an xfail is
skipped. A module that skips at import is one testcase with an empty classname
and the module as its name, which is what pytest's own summary counts too.

Stdlib only, like the rest of ci/, plus ``gh`` and ``git`` for ``guard``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: How many more skips the untouched files may have than on main. Skips that
#: depend on the host are per leg, and each leg is compared with its own
#: baseline, so this is headroom for noise, not for platforms.
SKIP_TOLERANCE = 10

#: The pull request label that accepts lost tests, or more skips, in files the
#: change does not touch.
DROP_LABEL = "drops tests"

#: The artifact a green run uploads, one per leg: ``ARTIFACT_PREFIX + label``.
ARTIFACT_PREFIX = "test-file-counts-"

#: The file inside that artifact.
COUNTS_FILE = "test-file-counts.json"

#: How many of main's newest baselines to try for the nearest ancestor. About
#: a week of merges, so re-running an old pull request job still finds one.
CANDIDATES = 30

Counts = dict[str, dict[str, int]]


# --- counting --------------------------------------------------------------


def file_of(dotted: str, root: Path) -> str:
    """``python.tests.test_x.TestY`` -> ``python/tests/test_x.py``.

    The longest prefix of the dotted name that is a ``.py`` file under ``root``.
    A name that resolves to no file (a test outside the checkout) is kept as it
    is: still a stable key, so it is compared like any other.
    """
    parts = dotted.split(".")
    for k in range(len(parts), 0, -1):
        candidate = "/".join(parts[:k]) + ".py"
        if (root / candidate).is_file():
            return candidate
    return dotted


def count_reports(reports: Iterable[Path], root: Path | None = None) -> Counts:
    """Per test file: ``{"collected": n, "skipped": m}``, summed over the reports."""
    root = REPO_ROOT if root is None else root
    out: Counts = {}
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            dotted = case.get("classname") or case.get("name") or ""
            entry = out.setdefault(file_of(dotted, root), {"collected": 0, "skipped": 0})
            entry["collected"] += 1
            if any(s.get("type") != "pytest.xfail" for s in case.findall("skipped")):
                entry["skipped"] += 1
    return dict(sorted(out.items()))


# --- comparing -------------------------------------------------------------


def compare(
    base: Counts,
    head: Counts,
    changed: set[str],
    *,
    drops_allowed: bool = False,
    skip_tolerance: int = SKIP_TOLERANCE,
) -> tuple[list[str], list[str], list[str]]:
    """``(errors, warnings, notes)``: what fails the guard, and what is only reported.

    A warning is a loss in a file the change edits. It passes, since the edit is
    in the diff, but it is the likeliest accidental loss (a class that lost its
    ``Test`` prefix, a fixture that now skips everything), so the guard makes it
    an annotation a reviewer sees on the checks page rather than a log line.
    """
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    lost: list[str] = []
    skip_rise = 0
    skip_rows: list[str] = []
    for path in sorted(set(base) | set(head)):
        b = base.get(path, {"collected": 0, "skipped": 0})
        h = head.get(path, {"collected": 0, "skipped": 0})
        if path not in base:
            continue  # new: nothing to hold it to
        dc = h["collected"] - b["collected"]
        ds = h["skipped"] - b["skipped"]
        if path in changed:
            if dc < 0:
                warnings.append(
                    f"{path} collects {h['collected']} tests, {b['collected']} on main. "
                    "This change edits it, so that passes; check that the loss is meant."
                )
            continue
        if dc < 0:
            lost.append(f"{path}: {b['collected']} -> {h['collected']}")
        if ds > 0:
            skip_rise += ds
            skip_rows.append(f"{path}: {b['skipped']} -> {h['skipped']} skipped")
    if lost:
        rows = "; ".join(lost)
        if drops_allowed:
            notes.append(
                "tests lost in files this change does not touch, "
                f"accepted by the '{DROP_LABEL}' label: {rows}"
            )
        else:
            errors.append(
                f"tests stopped being collected in {len(lost)} file(s) "
                f"this change does not touch: {rows}. If that is intended, "
                f"add the '{DROP_LABEL}' label to the pull request and re-run this job."
            )
    if skip_rise > skip_tolerance and drops_allowed:
        notes.append(
            f"{skip_rise} more skips in untouched files, accepted by the "
            f"'{DROP_LABEL}' label: " + "; ".join(skip_rows)
        )
    elif skip_rise > skip_tolerance:
        errors.append(
            f"{skip_rise} more tests skip than on main in files this change does not touch "
            f"(at most {skip_tolerance} allowed): {'; '.join(skip_rows)}. "
            "A group of tests turned into skips. If that is intended, add the "
            f"'{DROP_LABEL}' label to the pull request and re-run this job."
        )
    elif skip_rows:
        notes.append(
            f"{skip_rise} more skips in untouched files, within {skip_tolerance}: "
            + "; ".join(skip_rows)
        )
    return errors, warnings, notes


def totals(counts: Counts) -> tuple[int, int]:
    return (
        sum(c["collected"] for c in counts.values()),
        sum(c["skipped"] for c in counts.values()),
    )


# --- main's baseline (guard only) ------------------------------------------

Api = Callable[[str], bytes]


def gh_api(path: str) -> bytes:
    done = subprocess.run(["gh", "api", path], capture_output=True)
    if done.returncode:
        raise RuntimeError(f"gh api {path}: {done.stderr.decode(errors='replace').strip()[:300]}")
    return done.stdout


def main_artifacts(api: Api, repo: str, name: str) -> list[dict]:
    """Main's unexpired artifacts called ``name``, newest first.

    Only main's runs upload one, so the newest page of 100 is all main's, and
    holds more than the CANDIDATES that are tried.
    """
    listing = json.loads(api(f"repos/{repo}/actions/artifacts?name={name}&per_page=100"))
    found = []
    for art in listing.get("artifacts", []):
        run = art.get("workflow_run") or {}
        same_repo = run.get("head_repository_id") == run.get("repository_id")
        if run.get("head_branch") == "main" and same_repo and not art.get("expired"):
            found.append(art)
    return sorted(found, key=lambda a: a.get("created_at", ""), reverse=True)


def commits_ahead(api: Api, repo: str, base_sha: str, head_sha: str) -> int | None:
    """How many commits ``head_sha`` is ahead of ``base_sha``; None if base is no ancestor."""
    if base_sha == head_sha:
        return 0
    cmp = json.loads(api(f"repos/{repo}/compare/{base_sha}...{head_sha}"))
    if cmp.get("status") == "identical":
        return 0
    if cmp.get("status") == "ahead":
        return int(cmp.get("ahead_by", 0))
    return None


def pick_baseline(api: Api, repo: str, name: str, head_sha: str) -> dict | None:
    """Of main's newest artifacts, the one nearest ``head_sha`` among its ancestors.

    Nearest, not newest: re-running an old main run uploads a new artifact for
    an old commit, which would widen the diff and exempt more files. None when
    main has none at all; the caller decides whether that is the bootstrap.
    RuntimeError when it has some but none is an ancestor.
    """
    arts = main_artifacts(api, repo, name)[:CANDIDATES]
    if not arts:
        return None
    best: tuple[int, dict] | None = None
    for art in arts:
        ahead = commits_ahead(api, repo, art["workflow_run"]["head_sha"], head_sha)
        if ahead is not None and (best is None or ahead < best[0]):
            best = (ahead, art)
            if ahead == 0:
                break
    if best is None:
        raise RuntimeError(
            f"none of main's {len(arts)} newest '{name}' artifacts is from an ancestor "
            f"of {head_sha[:12]}; merge main into this branch"
        )
    return best[1]


def is_bootstrap(api: Api, repo: str, head_sha: str) -> bool:
    """Whether this commit is where this script arrives: its first parent lacks it.

    On a pull request, that is the one that adds it. Any later pull request that
    finds no baseline is failed, because main uploads one from each passing
    run; main itself never is (see cmd_guard).
    """
    parents = json.loads(api(f"repos/{repo}/commits/{head_sha}")).get("parents") or []
    if not parents:
        return False
    try:
        api(f"repos/{repo}/contents/ci/test_file_counts.py?ref={parents[0]['sha']}")
    except RuntimeError as exc:
        if "404" in str(exc) or "Not Found" in str(exc):
            return True
        raise
    return False


def download_counts(api: Api, repo: str, art: dict) -> Counts:
    blob = api(f"repos/{repo}/actions/artifacts/{art['id']}/zip")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return json.loads(zf.read(COUNTS_FILE))["files"]


#: A merge commit's subject, or a squash merge's "Title (#N)".
MERGE_PR = re.compile(r"^Merge pull request #(\d+)\b|\(#(\d+)\)$")


def labels_since(api: Api, repo: str, pr: int, base_sha: str, head_sha: str) -> set[str]:
    """The labels of this pull request and of every one merged since the baseline.

    A drop labelled on the pull request that made it is accepted for every run
    whose baseline predates it, so a main run that was cancelled or red before
    it could upload newer counts does not charge that drop to the next change.
    """
    numbers = {pr} if pr else set()
    if base_sha != head_sha:
        cmp = json.loads(api(f"repos/{repo}/compare/{base_sha}...{head_sha}"))
        # At most 250 commits, the newest: a drop more than ~70 merges back is
        # missed, and that run is red until a nearer baseline is uploaded.
        for c in cmp.get("commits", []):
            subject = (c.get("commit") or {}).get("message", "").split("\n", 1)[0]
            m = MERGE_PR.search(subject)
            if m:
                numbers.add(int(m.group(1) or m.group(2)))
    labels: set[str] = set()
    for n in sorted(numbers):
        pull = json.loads(api(f"repos/{repo}/pulls/{n}"))
        labels |= {label["name"] for label in pull.get("labels", [])}
    return labels


def changed_files(base_sha: str, head_sha: str) -> set[str]:
    """Paths that differ between the two commits, fetching the base if the clone lacks it."""
    have = subprocess.run(["git", "cat-file", "-e", f"{base_sha}^{{commit}}"], capture_output=True)
    if have.returncode:
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base_sha],
            check=True,
            capture_output=True,
        )
    out = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", base_sha, head_sha],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(out.split())


# --- commands --------------------------------------------------------------


def write_counts(path: Path, counts: Counts, sha: str) -> None:
    path.write_text(
        json.dumps({"sha": sha, "files": counts}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summary(lines: list[str]) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def cmd_count(a: argparse.Namespace) -> int:
    counts = count_reports(a.reports)
    if not counts:
        print("::error::no test cases in those reports", file=sys.stderr)
        return 1
    write_counts(a.out, counts, a.sha)
    collected, skipped = totals(counts)
    print(f"{a.out}: {len(counts)} files, {collected} collected, {skipped} skipped")
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    base = json.loads(a.base.read_text())["files"]
    head = json.loads(a.head.read_text())["files"]
    changed = set(a.changed.read_text().split()) if a.changed else set()
    errors, warnings, notes = compare(base, head, changed, drops_allowed=a.drops_allowed)
    for n in notes:
        print(f"note: {n}")
    for w in warnings:
        print(f"::warning::{w}")
    for e in errors:
        print(f"::error::{e}")
    return 1 if errors else 0


def cmd_guard(a: argparse.Namespace) -> int:
    api: Api = gh_api  # looked up per call, so a test can replace it
    reports = sorted(a.reports.rglob("*.xml"))
    head = count_reports(reports)
    if not head:
        print(
            f"::error::no test cases in the JUnit reports under {a.reports}: "
            "the suites did not report."
        )
        return 1
    collected, skipped = totals(head)
    print(
        f"{a.leg}: {len(head)} test files, {collected} collected, {skipped} skipped "
        f"({len(reports)} reports)"
    )
    if a.expect_collected is not None and (collected, skipped) != (
        a.expect_collected,
        a.expect_skipped,
    ):
        print(
            f"::error::the JUnit reports hold {collected} tests, {skipped} skipped, "
            f"but pytest's summaries say {a.expect_collected}, {a.expect_skipped}: "
            "per-file counts read from them would be wrong."
        )
        return 1

    name = ARTIFACT_PREFIX + a.leg
    try:
        art = pick_baseline(api, a.repo, name, a.sha)
        if art is None and a.pr and not is_bootstrap(api, a.repo, a.sha):
            raise RuntimeError(
                f"main has no '{name}' artifact to compare with. A passing run on "
                "main uploads one: re-run this job once main's latest python-tests "
                "run has passed."
            )
    except RuntimeError as exc:
        print(f"::error::{exc}")
        return 1
    if art is None:
        # The pull request that adds this script, or a run on main. Main
        # uploads what it counted, so the next run has a baseline again; a
        # missing one cannot leave main red with no way back.
        print(
            f"::warning::main has no '{name}' artifact, so nothing was compared; "
            "this run's counts become the baseline if it is on main."
        )
        summary([f"### Test file counts · {a.leg}", "", "No baseline: nothing compared."])
        write_counts(a.out, head, a.sha)
        return 0
    base_sha = art["workflow_run"]["head_sha"]
    base = download_counts(api, a.repo, art)
    changed = changed_files(base_sha, a.sha)
    errors, warnings, notes = compare(base, head, changed)
    # Every error compare() reports is one the label accepts; ask only then.
    if errors and DROP_LABEL in labels_since(api, a.repo, a.pr, base_sha, a.sha):
        errors, warnings, notes = compare(base, head, changed, drops_allowed=True)

    b_col, b_skip = totals(base)
    print(f"main at {base_sha[:12]}: {len(base)} test files, {b_col} collected, {b_skip} skipped")
    print(f"{len(changed)} paths differ from it")
    for n in notes:
        print(f"note: {n}")
    for w in warnings:
        print(f"::warning::{w}")
    for e in errors:
        print(f"::error::{e}")
    summary(
        [
            f"### Test file counts · {a.leg}",
            "",
            f"{collected} collected ({collected - b_col:+d}), "
            f"{skipped} skipped ({skipped - b_skip:+d}) "
            f"against main at `{base_sha[:12]}`.",
            *(f"- {n}" for n in notes),
            *(f"- ⚠️ {w}" for w in warnings),
            *(f"- **{e}**" for e in errors),
        ]
    )
    if errors:
        return 1
    # Written only when the counts passed: the workflow uploads this file as
    # main's next baseline, and a failed run's counts must not become one.
    write_counts(a.out, head, a.sha)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count", help="write per-file counts from JUnit reports")
    c.add_argument("reports", nargs="+", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--sha", default="")

    m = sub.add_parser("compare", help="compare two counts files")
    m.add_argument("base", type=Path)
    m.add_argument("head", type=Path)
    m.add_argument("--changed", type=Path, help="file listing the paths the change touches")
    m.add_argument("--drops-allowed", action="store_true")

    g = sub.add_parser("guard", help="CI: count, fetch main's baseline, compare")
    g.add_argument("--leg", required=True)
    g.add_argument(
        "--reports", type=Path, required=True, help="directory holding the JUnit reports"
    )
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "lanl/bngsim"))
    g.add_argument("--sha", default=os.environ.get("GITHUB_SHA", ""))
    g.add_argument("--pr", type=int, default=0, help="pull request number; 0 on a push")
    g.add_argument(
        "--expect-collected", type=int, help="pytest's own total, to check the reports against"
    )
    g.add_argument("--expect-skipped", type=int, default=0)

    a = ap.parse_args(argv)
    return {"count": cmd_count, "compare": cmd_compare, "guard": cmd_guard}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
