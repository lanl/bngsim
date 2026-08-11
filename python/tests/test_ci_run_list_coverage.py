"""A job that fires on a test file must either run it or say why it does not.

Issue #295. ``windows-tail.yml``'s paths filter matched 43 ``python/tests``
files and its run step named 26. Editing any of the other 17 summoned a green
check that had not executed the file that summoned it -- the same defect #291
fixed in ``windows-nfsim.yml``, in a job where the #291 remedy does not apply.

The distinction this file exists to encode: **a run list narrower than its
trigger is not automatically a defect.** ``mir.yml`` omits the cc-backend tests
on purpose -- under ``BNGSIM_CODEGEN_JIT=mir`` there is no ``.so``, so a test
that caches one, shards its compile, or reaps its compiler subprocess has
nothing to assert. A blanket "every run list covers its own paths filter" check
would be wrong there, which is why ``test_windows_nfsim_ci_coverage.py`` scoped
itself to the one cluster where it is right.

So the assertion is not coverage, it is **declaration**: every file a leg fires
on is either executed by that leg or named, with a reason, in a ``not-run:``
comment in the workflow beside the list that skips it. That leaves the same two
jobs free to run different subsets and makes the difference between an intent
and an oversight readable -- to a reviewer, and to this test. The seventeen
files were not a decision anybody made; they were a list nobody updated.

Discovery is by structure, not by a table: any workflow with a ``python/tests``
paths filter and a pytest step is in scope, so a fourth leg inherits the rule
without an edit here.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

# Sibling helper import under pytest's importlib mode, as test_ssa_variable_volume.py
# does it. The parsers are shared with test_windows_nfsim_ci_coverage.py; two private
# copies of them would be the drift these tests exist to catch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ci_workflow import (  # noqa: E402
    WORKFLOWS,
    base_roots,
    expand,
    module_level_optional_imports,
    paths_filter_selectors,
    provisioned_roots,
    pytest_selectors,
    strip_comments,
)

#: ``# not-run: <file> -- <reason>``, the declaration form. Read from the raw
#: workflow text rather than the comment-stripped body, because here the comment
#: *is* the configuration: it is what distinguishes mir.yml's deliberate
#: cc-backend omission from windows-tail.yml's seventeen forgotten files.
NOT_RUN = re.compile(r"^\s*#\s*not-run:\s*(\S+\.py)\s*(?:--|—)\s*(.*)$", re.M)

#: A reason has to survive being read aloud in review. Short enough that a real
#: one-liner passes, long enough that ``-- n/a`` does not.
MIN_REASON = 40


def _in_scope() -> list[str]:
    """Workflows that fire on ``python/tests`` files and run pytest.

    Both halves are required. ``python-tests.yml`` runs pytest and has no paths
    filter at all -- it takes the whole directory, so it has nothing to disagree
    with and nothing to declare.
    """
    names = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        body = strip_comments(path.read_text(encoding="utf-8"))
        if paths_filter_selectors(body) and "python -m pytest" in body:
            names.append(path.name)
    return names


IN_SCOPE = _in_scope()

#: What the parametrized classes below iterate. An *empty* parameter set makes
#: pytest skip with a reason of its own wording, which conftest's skip audit has
#: no ``_DECLARED_SKIPS`` entry for -- so a checkout without ``.github/`` would
#: report undeclared skips (fatal under BNGSIM_SKIP_AUDIT=strict) instead of the
#: declared one the module-level skipif already supplies.
CASES = IN_SCOPE or [
    pytest.param("", marks=pytest.mark.skip(reason=".github/ not in this checkout"))
]

pytestmark = pytest.mark.skipif(not WORKFLOWS.is_dir(), reason=".github/ not in this checkout")


def _raw(workflow: str) -> str:
    return (WORKFLOWS / workflow).read_text(encoding="utf-8")


def _declared_not_run(workflow: str) -> dict[str, str]:
    return {name: reason.strip() for name, reason in NOT_RUN.findall(_raw(workflow))}


def _triggered(workflow: str) -> set[str]:
    return expand(paths_filter_selectors(strip_comments(_raw(workflow))))


def _executed(workflow: str) -> set[str]:
    return expand(pytest_selectors(strip_comments(_raw(workflow))))


def test_the_scope_is_discovered_and_is_not_empty():
    """Guard the discovery itself.

    Every assertion below is parametrized over ``IN_SCOPE``; if the pattern that
    finds these workflows stopped matching, the whole module would collect zero
    cases and report green. That is the failure mode this file is about.
    """
    assert IN_SCOPE, "no workflow matched 'has a python/tests paths filter and runs pytest'"
    known = {"windows-tail.yml", "mir.yml", "windows-nfsim.yml"}
    still_here = {name for name in known if (WORKFLOWS / name).exists()}
    assert still_here <= set(IN_SCOPE), (
        f"{sorted(still_here - set(IN_SCOPE))} used to be in scope and no longer parse as "
        "'fires on python/tests files and runs pytest'. Either the workflow changed shape "
        "or the parser did; both cost this file its teeth."
    )


@pytest.mark.parametrize("workflow", CASES)
class TestEveryTriggeredFileIsRunOrDeclared:
    def test_a_leg_does_not_fire_on_a_file_it_says_nothing_about(self, workflow):
        """The regression itself, generalized.

        The converse is deliberately not asserted: running a file the filter
        ignores costs a little wall clock and never a false green.
        """
        undeclared = sorted(
            _triggered(workflow) - _executed(workflow) - set(_declared_not_run(workflow))
        )
        assert not undeclared, (
            f"{workflow} fires on {undeclared} and neither runs them nor declares them "
            "(issue #295). Editing one of those files triggers the leg, the leg goes green, "
            "and it has said nothing about the change that summoned it. Add the file to the "
            "pytest call, or add a `# not-run: <file> -- <why>` line beside the list."
        )

    def test_a_declared_exclusion_names_a_file_that_exists(self, workflow):
        """A stale exclusion is worse than none: it reads as a decision about a
        file, and it is silently protecting nothing after a rename."""
        declared = _declared_not_run(workflow)
        gone = sorted(name for name in declared if name not in _triggered(workflow))
        assert not gone, (
            f"{workflow} declares {gone} not-run, but nothing in its paths filter matches "
            "them -- renamed, deleted, or never triggered in the first place. Drop the line."
        )

    def test_a_declared_exclusion_is_not_also_run(self, workflow):
        """Contradiction, and the direction that matters: the comment says the
        file is deliberately out while the run step runs it. A reader trusts the
        comment, so the file is effectively undocumented."""
        both = sorted(set(_declared_not_run(workflow)) & _executed(workflow))
        assert not both, f"{workflow} both runs and declares-not-run {both}."

    def test_a_declared_exclusion_states_why(self, workflow):
        """The reason is the entire mechanism. Without it this is the same
        hand-maintained list, with a comment character in front of it."""
        thin = sorted(
            f"{name} ({len(reason)} chars)"
            for name, reason in _declared_not_run(workflow).items()
            if len(reason) < MIN_REASON
        )
        assert not thin, (
            f"{workflow} declares {thin} not-run without a usable reason. "
            "An omission with a stated reason is a decision; without one it is drift "
            "that learned to spell."
        )


@pytest.mark.parametrize("workflow", CASES)
class TestOptingAFileInIsNotEnoughToRunIt:
    def test_module_level_optional_imports_are_provisioned(self, workflow):
        """A module-scope ``importorskip`` the leg does not install is worse
        than an untested file: it reports one skip where a whole module was.

        This is #291's near-miss, generalized to every leg. ``windows-nfsim``
        installed only pytest, so opting in ``test_nfsim_exprtk_parity.py`` --
        whose ``scipy.special.hyp1f1`` oracle is guarded at module scope --
        would have collected none of its six tests and reported one skip. Module
        scope is the discriminating part: an indented ``importorskip`` costs one
        test and shows up in the count.
        """
        pytest.importorskip("tomllib", reason="tomllib is 3.11+")
        body = strip_comments(_raw(workflow))
        available = base_roots() | provisioned_roots(body)
        offenders = [
            f"{name} needs {root}"
            for name in sorted(_executed(workflow))
            for root in module_level_optional_imports(name)
            if root not in available
        ]
        assert not offenders, (
            f"{workflow} runs but does not install: "
            + "; ".join(offenders)
            + ". Those modules skip at collection, so the files are nominally opted in "
            "and execute nothing (issue #291)."
        )
