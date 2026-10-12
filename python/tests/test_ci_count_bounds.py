"""Every CI test-count guard holds its count to a bound that cannot go stale.

Issue #584. Five CI jobs guard against a false green by counting what they ran.
A module that stops importing, a rename that orphans a file, or a conftest that
deselects leaves pytest green with a smaller denominator (#28/#36). Each guard
compared its count to a floor written into the workflow, and each workflow said
its floors would rise as tests were added. Nothing made them rise. When #584 was
fixed, python-tests.yml's collected floor was 3480 while CI collected 11288, so
two thirds of the suite could have vanished with the guard still green. And
native-tests.yml's CTEST_FLOOR was 5 with 6 binaries registered, so one binary
could go missing unnoticed, which is the one case it existed to catch.

The bounds now live in ``ci/test_floors.env``. Each guard passes its count to
``ci/check_test_count.sh``, which fails in both directions: past the bound, and
so far inside it that the bound has gone stale. The run that outgrows a floor is
the run that fails, and moving the floor is an edit outside
``.github/workflows/``.

**Why the staleness check runs in CI rather than here.** The issue proposed a
meta-test that collects the suite and compares. A developer checkout is not
CI's, though: three tests parametrize over gitignored .net and SBML corpora, so
the main checkout collected 14651 tests where every CI leg collected 11288, at
the same commit. A collection also cannot see passed counts, skip counts, or the
C++ suite's counts. Each guard already holds the authoritative count for its own
leg.

What this module pins is the arrangement, so that it cannot erode the way the
floors did:

* the bounds file is strict ``NAME=integer``, one direction plus a slack per
  NAME;
* no workflow hard-codes a bound, so a new guard written with the old
  ``*_FLOOR`` names, or against a literal, fails here (one that hides its bound
  under another name does not);
* every bound is checked by some workflow, and every check names a bound;
* a check can fail its step and its job, and a leg with a paths filter fires
  when the file holding its bound changes;
* both files check out with LF endings on Windows, where Git Bash runs them;
* the helper fails at exactly the edges the file declares, in both directions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ci_workflow import (  # noqa: E402
    REPO_ROOT,
    WORKFLOWS,
    continued_block,
    paths_filter_entries,
    strip_comments,
)

BOUNDS = REPO_ROOT / "ci" / "test_floors.env"
HELPER = REPO_ROOT / "ci" / "check_test_count.sh"

#: The two files a bound-reading leg must fire on, as they appear in a paths filter.
TRIGGERS = ("ci/check_test_count.sh", "ci/test_floors.env")

#: Every guard as #584 left it. The scan below is derived, so it can go blind
#: (a renamed helper, a call this regex no longer reads), and a blind scan finds
#: no calls, which every other assertion here would accept. This is the floor
#: under the floors.
KNOWN_CHECKS = {
    "python-tests.yml": {
        "PYTEST_COLLECTED",
        "PYTEST_SKIPPED",
        "PARITY_COLLECTED",
        "PARITY_SKIPPED",
    },
    "windows-nfsim.yml": {"NFSIM_PASSED"},
    "windows-klu.yml": {"KLU_PASSED"},
    "native-tests.yml": {"CTEST_BINARIES", "TEST_BNGSIM_CASES"},
}

#: A bounds-file line. Exactly what the helper's ``grep -E '^KEY='`` reads: no
#: leading space, no ``export``, no trailing comment, digits only.
LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=([0-9]+)$")
KEY = re.compile(r"^([A-Z][A-Z0-9_]*)_(FLOOR|CEILING|MAX_SLACK)$")

#: A call to the helper, and the NAME it checks.
CALL = re.compile(r"check_test_count\.sh\s+(\S+)")

#: The shape every hand-rolled guard had before #584: a bound assigned in the
#: workflow, compared in the workflow, or a count compared against a literal.
#: ``-ne 0`` / ``-eq 0`` stay legal. The Windows and native legs require zero
#: skips outright, and that check is exact by design.
HAND_ROLLED = (
    re.compile(r"\b[A-Z][A-Z0-9_]*_(?:FLOOR|CEILING)=\d"),
    re.compile(r"-(?:lt|le|gt|ge)\s+\"?\$\{?[A-Z0-9_]*_(?:FLOOR|CEILING)\b"),
    re.compile(r"-(?:lt|le|gt|ge)\s+\"?(?:[2-9]|[1-9]\d+)\b"),
)

pytestmark = pytest.mark.skipif(
    not (WORKFLOWS.is_dir() and BOUNDS.is_file()),
    reason=".github/ or ci/test_floors.env not in this checkout",
)


def _bound_lines() -> list[str]:
    """The file's non-blank, non-comment lines, as the helper would see them."""
    text = BOUNDS.read_text(encoding="utf-8")
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def _bounds() -> dict[str, tuple[str, int, int]]:
    """``NAME -> (direction, bound, max_slack)``, where direction is FLOOR or CEILING.

    Only the well-formed NAMEs. This runs at import to parametrize the tests
    below, so a malformed file must not raise here: an import error would
    replace the format tests' messages with a traceback.
    """
    values = {m.group(1): int(m.group(2)) for m in map(LINE.match, _bound_lines()) if m}
    out = {}
    for key, value in values.items():
        match = KEY.match(key)
        if not match or match.group(2) == "MAX_SLACK":
            continue
        name, kind = match.groups()
        other = "CEILING" if kind == "FLOOR" else "FLOOR"
        if f"{name}_MAX_SLACK" in values and f"{name}_{other}" not in values:
            out[name] = (kind, value, values[f"{name}_MAX_SLACK"])
    return out


class Call(NamedTuple):
    """One call to the helper, with the text around it that can defeat it."""

    name: str  #: the NAME argument
    call: str  #: the call, backslash continuations joined
    block: str  #: the step's ``run:`` block up to the call: where a ``set +e`` would be
    step: str  #: the whole step: where a ``continue-on-error`` would be
    job: str  #: the job's own keys, either side of ``steps:``: a job-level ``continue-on-error``


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _calls() -> dict[str, list[Call]]:
    """``workflow -> [Call]`` for every call to the helper in every workflow.

    Steps and jobs are found by indentation, the way the parsers in
    ``_ci_workflow.py`` read a workflow, rather than by a YAML load: PyYAML is
    only a transitive dependency here.
    """
    out: dict[str, list[Call]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        lines = strip_comments(path.read_text(encoding="utf-8")).splitlines()
        for i, line in enumerate(lines):
            match = CALL.search(line)
            if not match:
                continue
            run = max((j for j in range(i) if re.match(r"^\s*run:", lines[j])), default=0)
            step = max(
                (
                    j
                    for j in range(run + 1)
                    if re.match(r"^\s*-\s", lines[j]) and _indent(lines[j]) < _indent(lines[run])
                ),
                default=run,
            )
            end = next(
                (
                    k
                    for k in range(i + 1, len(lines))
                    if lines[k].strip() and _indent(lines[k]) <= _indent(lines[step])
                ),
                len(lines),
            )
            job = max(
                (j for j in range(step) if re.match(r"^  [A-Za-z0-9_-]+:\s*$", lines[j])),
                default=0,
            )
            job_end = next(
                (
                    k
                    for k in range(job + 1, len(lines))
                    if lines[k].strip() and _indent(lines[k]) <= 2
                ),
                len(lines),
            )
            out.setdefault(path.name, []).append(
                Call(
                    name=match.group(1),
                    call=" ".join(continued_block(lines, i)),
                    block="\n".join(lines[run : i + 1]),
                    step="\n".join(lines[step:end]),
                    job="\n".join(line for line in lines[job + 1 : job_end] if _indent(line) == 4),
                )
            )
    return out


def _run(name: str, count: object, helper: Path = HELPER) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(helper), name, str(count), "the breach text"],
        capture_output=True,
        text=True,
        check=False,
    )


_NAMES = sorted(_bounds()) if BOUNDS.is_file() else []
#: An empty parameter set skips under pytest's own wording, which the skip
#: audit has not declared, so a checkout without the file gets a declared one.
_NAME_PARAMS = _NAMES or [
    pytest.param("", marks=pytest.mark.skip(reason="ci/test_floors.env not in this checkout"))
]

needs_bash = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="POSIX-specific: ci/check_test_count.sh runs under bash",
)


class TestTheBoundsFile:
    def test_every_line_is_a_strict_assignment(self):
        """The helper greps ``^KEY=`` and reads digits. A line it cannot read is
        a bound it reports as missing at the worst moment, in CI.
        """
        bad = [line for line in _bound_lines() if not LINE.match(line)]
        assert not bad, f"ci/test_floors.env lines the helper cannot read: {bad}"

    def test_no_key_is_set_twice(self):
        keys = [m.group(1) for m in map(LINE.match, _bound_lines()) if m]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        assert not dupes, f"set more than once in ci/test_floors.env: {dupes}"

    def test_every_name_has_one_direction_and_a_slack(self):
        keys = [m.group(1) for m in map(LINE.match, _bound_lines()) if m]
        stray = [k for k in keys if not KEY.match(k)]
        assert not stray, f"keys that are not NAME_FLOOR / NAME_CEILING / NAME_MAX_SLACK: {stray}"
        kinds: dict[str, set[str]] = {}
        for key in keys:
            name, kind = KEY.match(key).groups()
            kinds.setdefault(name, set()).add(kind)
        wrong = {
            name: sorted(k)
            for name, k in kinds.items()
            if "MAX_SLACK" not in k or len(k & {"FLOOR", "CEILING"}) != 1
        }
        assert not wrong, (
            "each NAME needs exactly one of _FLOOR / _CEILING plus a _MAX_SLACK; "
            f"these do not: {wrong}"
        )

    def test_it_holds_bounds(self):
        assert _NAMES, "ci/test_floors.env defines no bounds, so every guard is unbounded"


class TestTheWorkflowsUseIt:
    def test_the_scan_still_sees_every_known_check(self):
        calls = _calls()
        missing = {
            wf: sorted(names - {c.name for c in calls.get(wf, [])})
            for wf, names in KNOWN_CHECKS.items()
            if (WORKFLOWS / wf).exists()
        }
        missing = {wf: names for wf, names in missing.items() if names}
        assert not missing, (
            f"checks the scan no longer finds: {missing}. Either a guard lost its "
            "check, or the scan went blind and every assertion below is vacuous."
        )

    def test_no_workflow_hard_codes_a_count_bound(self):
        """A bound written into a workflow is the arrangement #584 replaced:
        nothing holds it near its count, so it drifts until it guards nothing.
        """
        offenders = [
            f"{path.name}: {line.strip()}"
            for path in sorted(WORKFLOWS.glob("*.yml"))
            for line in strip_comments(path.read_text(encoding="utf-8")).splitlines()
            if any(p.search(line) for p in HAND_ROLLED)
        ]
        assert not offenders, (
            "hand-rolled count bounds: "
            + "; ".join(offenders)
            + ". Put the bound in ci/test_floors.env and check it with "
            "`bash ci/check_test_count.sh NAME COUNT 'what crossing it means'`."
        )

    def test_every_bound_is_checked_and_every_check_names_a_bound(self):
        """An unchecked bound is a number nobody reads, and it will go stale
        without failing anything. A check naming no bound exits 2 in CI, so
        catch it here first.
        """
        checked = {c.name for calls in _calls().values() for c in calls}
        assert set(_NAMES) - checked == set(), (
            f"bounds no workflow checks: {sorted(set(_NAMES) - checked)}"
        )
        assert checked - set(_NAMES) == set(), (
            f"checks with no bound in ci/test_floors.env: {sorted(checked - set(_NAMES))}"
        )

    def test_every_check_can_fail_its_step(self):
        """``bash`` rather than ``./``, so the file mode a Windows checkout
        drops does not matter. Three arguments, as the helper requires. Nothing
        that swallows the exit status: ``||``, a trailing ``&``, ``set +e``
        earlier in the same run block (GitHub runs every step's bash with -e),
        or ``continue-on-error`` on the step or its job, which turns a red guard
        into a green check.
        """
        bad = []
        unless_false = re.compile(r"^\s*(?:-\s+)?continue-on-error:\s*(?!false\b)\S", re.M)
        for wf, calls in _calls().items():
            for c in calls:
                name, call = c.name, c.call
                if not re.search(r"(?:^|\s)bash\s+ci/check_test_count\.sh\s", call):
                    bad.append(f"{wf} {name}: not invoked as `bash ci/check_test_count.sh`")
                if "||" in call or re.search(r"&\s*$", call):
                    bad.append(f"{wf} {name}: the exit status is swallowed")
                if re.search(r"^\s*set\s+\+\w*e", c.block, re.M):
                    bad.append(f"{wf} {name}: `set +e` earlier in the run block")
                if unless_false.search(c.step) or unless_false.search(c.job):
                    bad.append(f"{wf} {name}: continue-on-error on its step or job")
                args = re.search(r"check_test_count\.sh\s+\S+\s+\"\$\w+\"\s+\"[^\"]+\"\s*$", call)
                if not args:
                    bad.append(f'{wf} {name}: expected NAME "$count" "what crossing it means"')
        assert not bad, "; ".join(bad)

    def test_a_filtered_leg_fires_when_its_bound_moves(self):
        """A leg with a paths filter that does not list the bounds file lets a
        bump to its bound merge unchecked. The next unrelated PR that does fire
        the leg then goes red for a reason it did not cause.
        """
        missing = []
        for wf in _calls():
            entries = paths_filter_entries((WORKFLOWS / wf).read_text(encoding="utf-8"))
            if entries:
                missing += [f"{wf}: {t}" for t in TRIGGERS if t not in entries]
        assert not missing, f"paths filters missing the bound's files: {missing}"


class TestTheFilesSurviveAWindowsCheckout:
    def test_both_check_out_with_lf_endings(self):
        """A windows-latest checkout rewrites LF to CRLF (f562ae3). Git Bash
        then reads ``set -euo pipefail\\r`` and the helper dies before it reads
        a count, so both Windows legs fail at the guard. Neither Windows leg
        runs this module, so the attribute has to be asserted rather than
        observed.
        """
        if shutil.which("git") is None:
            pytest.skip("not a git checkout: git is not on PATH")
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-attr", "eol", "--", *TRIGGERS],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(f"not a git checkout: {result.stderr.strip()}")
        wrong = [line for line in result.stdout.splitlines() if not line.endswith(": eol: lf")]
        assert len(result.stdout.splitlines()) == len(TRIGGERS), result.stdout
        assert not wrong, (
            f"not pinned to LF by .gitattributes: {wrong}. A CRLF checkout of the "
            "helper fails both Windows legs at the guard step."
        )


@needs_bash
class TestTheHelperHoldsEachRealBound:
    """Run the helper against the real file at the edges each bound declares,
    so a bound and its slack mean what the file's notes say they mean.
    """

    @pytest.mark.parametrize("name", _NAME_PARAMS)
    def test_at_the_bound_and_one_past_it(self, name):
        kind, bound, _ = _bounds()[name]
        assert _run(name, bound).returncode == 0
        past = bound - 1 if kind == "FLOOR" else bound + 1
        if past < 0:
            return
        result = _run(name, past)
        assert result.returncode == 1, result.stdout
        side = "below its floor" if kind == "FLOOR" else "above its ceiling"
        assert f"::error::{name} is {past}, {side} of {bound}: the breach text." in result.stdout

    @pytest.mark.parametrize("name", _NAME_PARAMS)
    def test_at_the_slack_and_one_past_it(self, name):
        kind, bound, slack = _bounds()[name]
        sign = 1 if kind == "FLOOR" else -1
        edge = bound + sign * slack
        if edge < 0:
            return
        assert _run(name, edge).returncode == 0
        if edge + sign < 0:
            return
        result = _run(name, edge + sign)
        assert result.returncode == 1, result.stdout
        assert f"::error::{name}_{kind} is stale" in result.stdout
        assert "ci/test_floors.env" in result.stdout


@needs_bash
class TestTheHelperRefusesWhatItCannotRead:
    """Exit 2, not 0, for anything the helper cannot read: an unreadable bound
    that passed would be a false green inside the guard against false greens.
    """

    @staticmethod
    def _helper(tmp_path: Path, bounds: str, newline: str = "\n") -> Path:
        shutil.copy(HELPER, tmp_path / "check_test_count.sh")
        (tmp_path / "test_floors.env").write_bytes(bounds.replace("\n", newline).encode())
        return tmp_path / "check_test_count.sh"

    @pytest.mark.parametrize(
        ("bounds", "name", "count", "says"),
        [
            ("X_FLOOR=5\nX_MAX_SLACK=1\n", "Y", 5, "no Y_FLOOR or Y_CEILING"),
            ("X_FLOOR=5\nX_CEILING=9\nX_MAX_SLACK=1\n", "X", 5, "both a _FLOOR and a _CEILING"),
            ("X_FLOOR=5\n", "X", 5, "no X_MAX_SLACK"),
            ("X_FLOOR=5\nX_FLOOR=6\nX_MAX_SLACK=1\n", "X", 5, "set 2 times"),
            ("X_FLOOR=five\nX_MAX_SLACK=1\n", "X", 5, "not a non-negative integer"),
            ("X_FLOOR=5\nX_MAX_SLACK=1\n", "X", "", "nothing was counted"),
            ("X_FLOOR=5\nX_MAX_SLACK=1\n", "X", "5 tests", "nothing was counted"),
            ("X_FLOOR=5\nX_MAX_SLACK=1\n", "x", 5, "not an UPPER_CASE identifier"),
        ],
    )
    def test_exit_2(self, tmp_path, bounds, name, count, says):
        result = _run(name, count, self._helper(tmp_path, bounds))
        assert result.returncode == 2, result.stdout
        assert says in result.stdout and "::error::" in result.stdout

    def test_wrong_argument_count(self, tmp_path):
        helper = self._helper(tmp_path, "X_FLOOR=5\nX_MAX_SLACK=1\n")
        result = subprocess.run(["bash", str(helper), "X", "5"], capture_output=True, text=True)
        assert result.returncode == 2 and "usage:" in result.stdout

    def test_a_crlf_checkout_still_reads(self, tmp_path):
        """A Windows runner may check this file out with CRLF line endings, and
        two Windows legs read it. A ``\\r`` left on the value fails the integer
        check, which would take both legs down with exit 2.
        """
        helper = self._helper(tmp_path, "X_FLOOR=5\nX_MAX_SLACK=1\n", newline="\r\n")
        assert _run("X", 5, helper).returncode == 0
        assert _run("X", 4, helper).returncode == 1
        assert _run("X", 7, helper).returncode == 1

    def test_leading_zeros_are_decimal(self, tmp_path):
        """Bash arithmetic reads ``010`` as octal 8, and ``09`` as an error."""
        helper = self._helper(tmp_path, "X_FLOOR=010\nX_MAX_SLACK=0\nY_FLOOR=09\nY_MAX_SLACK=0\n")
        assert _run("X", 8, helper).returncode == 1
        assert _run("X", 10, helper).returncode == 0
        assert _run("X", "010", helper).returncode == 0
        assert _run("Y", 9, helper).returncode == 0
