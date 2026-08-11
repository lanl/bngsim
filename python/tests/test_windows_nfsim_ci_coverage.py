"""The Windows NFsim leg's own coverage is the thing under test.

Issue #291. ``windows-nfsim.yml`` triggered on the glob
``python/tests/test_nfsim*.py`` but ran a hand-maintained list of six filenames,
and the list had not been touched since the workflow was created. Three NFsim
test files added after that -- including both symmetry regression guards for the
vendored NFsim carries (#195/#278/#282 and #281/#290) -- therefore *fired* the
leg and were never executed by it. Editing one of them summoned a green check
that structurally could not fail for the change that summoned it.

The workflow now selects the cluster with the same glob the paths filter uses,
which makes that particular drift impossible. This file guards the two things a
glob does not: that the selector and the trigger stay one string, and that the
leg cannot go green while running less than it claims. Both are properties of a
file no runtime test reads, which is how the original defect survived a year --
so these are text assertions over the workflow, in the shape of
``test_lapack_dense_ci_coverage.py`` and ``test_suitesparse_build_scripts.py``.
The functional proof lives in CI; this is what keeps the proof from being
deleted.

What is deliberately NOT asserted is the shape of the list itself. The same
trigger/run mismatch exists on purpose in ``mir.yml``, whose filter matches 25
``test_codegen*.py`` files and which runs 18 of them because the other 7 test a
backend that job does not build. A blanket "every run list covers its own paths
filter" check would be wrong there. This one is scoped to the NFsim cluster,
where the leg's entire purpose is to run all of it on Windows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "windows-nfsim.yml"
TESTS_DIR = Path(__file__).resolve().parent

#: The cluster this leg exists to run, as it appears on disk.
CLUSTER_GLOB = "test_nfsim*.py"

#: Import roots a test module may reach for without the workflow installing
#: anything: bngsim's own base dependencies (pyproject ``[project] dependencies``)
#: plus bngsim itself. Anything else has to be installed explicitly or the
#: module degrades to a silent skip.
ALWAYS_PRESENT = frozenset({"bngsim", "numpy", "sympy", "libsbml", "pytest"})

pytestmark = pytest.mark.skipif(not WORKFLOW.exists(), reason=".github/ not in this checkout")


def _strip_comments(text: str) -> str:
    """Drop whole-line ``#`` comments.

    Line-based, like the #178 and #269 tests' helpers, and for the same reason:
    this workflow carries its rationale in whole-line comments that name the
    exact filenames and tokens being asserted on, so a substring search over the
    raw text would read the prose as configuration.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _body() -> str:
    return _strip_comments(WORKFLOW.read_text(encoding="utf-8"))


def _continued_block(lines: list[str], start: int) -> list[str]:
    """``lines[start]`` plus every line joined to it by a trailing backslash."""
    block = [lines[start]]
    while block[-1].rstrip().endswith("\\") and start + len(block) <= len(lines) - 1:
        block.append(lines[start + len(block)])
    return block


def _paths_filter_selectors(body: str) -> list[str]:
    """The ``python/tests/`` entries of the ``paths:`` filter (YAML list items)."""
    return re.findall(r'^\s*-\s*"(python/tests/[^"]+)"', body, re.M)


def _pytest_selectors(body: str) -> list[str]:
    """The ``python/tests/`` arguments the run step hands to pytest."""
    lines = body.splitlines()
    start = next((i for i, line in enumerate(lines) if "python -m pytest" in line), None)
    assert start is not None, "windows-nfsim.yml no longer invokes pytest"
    return [
        token
        for line in _continued_block(lines, start)
        for token in line.split()
        if token.startswith("python/tests/")
    ]


def _expand(selectors: list[str]) -> set[str]:
    """Resolve selectors (literal or globbed) against the checkout, as bash would."""
    resolved: set[str] = set()
    for selector in selectors:
        parent, _, pattern = selector.rpartition("/")
        resolved.update(path.name for path in (REPO_ROOT / parent).glob(pattern))
    return resolved


def _pip_install_text(body: str) -> str:
    """Every ``pip install`` command in the workflow, line continuations included."""
    lines = body.splitlines()
    return "\n".join(
        line
        for i, _ in enumerate(lines)
        if "pip install" in lines[i]
        for line in _continued_block(lines, i)
    )


class TestTheLegRunsWhatItTriggersOn:
    def test_every_nfsim_test_file_in_the_repo_is_executed(self):
        """The regression itself: a file on disk that this leg never runs.

        Asserted against the filesystem rather than against the paths filter, so
        it fails for a file the filter would also miss -- e.g. an NFsim test
        named off-pattern, which is invisible twice over.
        """
        on_disk = {path.name for path in TESTS_DIR.glob(CLUSTER_GLOB)}
        assert on_disk, f"no {CLUSTER_GLOB} files found -- this test's own premise is broken"
        executed = _expand(_pytest_selectors(_body()))
        missing = sorted(on_disk - executed)
        assert not missing, (
            f"windows-nfsim.yml does not execute {missing}. Those files match the "
            "workflow's paths filter, so editing one triggers the leg, the leg "
            "goes green, and it has said nothing about the file that summoned it "
            "(issue #291). Select the cluster with the glob rather than by name."
        )

    def test_the_trigger_and_the_run_step_cannot_disagree(self):
        """Everything the filter fires on must be something the run step runs.

        The converse is fine: running a file the filter ignores costs a little
        time, never a false green.
        """
        body = _body()
        triggered = _expand(_paths_filter_selectors(body))
        assert triggered, "parsed no python/tests entries from the paths filter"
        unrun = sorted(triggered - _expand(_pytest_selectors(body)))
        assert not unrun, f"windows-nfsim.yml fires on {unrun} but does not run them (issue #291)."

    def test_the_cluster_is_selected_by_glob(self):
        """One string for both, which is what makes the drift structural.

        The two assertions above catch a recurrence at the moment it happens,
        because python-tests.yml has no paths filter and so runs this file on
        every PR. This one catches the change that *makes* recurrence possible:
        going back to a hand-maintained list passes the coverage checks on the
        day it lands and fails silently on the next file added.
        """
        assert CLUSTER_GLOB in " ".join(_pytest_selectors(_body())), (
            "windows-nfsim.yml no longer selects the NFsim cluster with the same "
            f"{CLUSTER_GLOB} glob its paths filter uses. A hand-maintained list "
            "is exactly what issue #291 was."
        )


class TestTheLegCannotGoGreenHavingRunNothing:
    def test_module_level_optional_imports_are_installed(self):
        """A module-level ``importorskip`` the leg does not provision is worse
        than an untested file: it reports one skip where six tests used to be.

        This is how ``test_nfsim_exprtk_parity.py`` would have been opted in and
        still tested nothing -- the workflow installs only pytest by design, and
        that file's ``scipy.special.hyp1f1`` oracle is guarded at module scope.
        Module scope is the discriminating part: an indented ``importorskip``
        costs one test and is visible in the count, a top-level one takes the
        whole file out of collection.
        """
        installed = _pip_install_text(_body())
        pattern = re.compile(r'^[^\s#].*importorskip\(\s*["\']([\w.]+)["\']', re.M)
        offenders = []
        for path in sorted(TESTS_DIR.glob(CLUSTER_GLOB)):
            for name in pattern.findall(path.read_text(encoding="utf-8")):
                root = name.split(".")[0]
                if root not in ALWAYS_PRESENT and root not in installed:
                    offenders.append(f"{path.name} needs {root}")
        assert not offenders, (
            "windows-nfsim.yml does not install: "
            + "; ".join(offenders)
            + ". Those modules skip at collection on the leg, so the files are "
            "nominally opted in and execute nothing (issue #291)."
        )

    def test_a_build_without_nfsim_fails_the_leg(self):
        """Every test in the cluster is behind ``skipif(not HAS_NFSIM)``, so a
        build that stopped linking NFsim would skip all of them and stay green.
        Same shape as python-tests.yml's HAS_KLU assertion.
        """
        assert re.search(r"sys\.exit\([^)]*HAS_NFSIM", _body()), (
            "windows-nfsim.yml prints HAS_NFSIM but does not fail on it. A build "
            "that lost the vendored NFsim would report a green empty run."
        )

    def test_a_shrinking_run_fails_the_leg(self):
        """The count guard is the backstop for the cases above that are not
        expressible as a text assertion -- a rename that orphans a file, a
        conftest that deselects, an optional import that vanishes upstream.
        """
        body = _body()
        assert "PASS_FLOOR" in body, (
            "windows-nfsim.yml no longer floors its passed count, so a file that "
            "stopped being collected would not fail the leg."
        )
        assert re.search(r'count skipped|"\$skipped"', body), (
            "windows-nfsim.yml no longer checks its skip count. On this leg every "
            "reachable skip reason is a false green -- see the comment on that check."
        )
