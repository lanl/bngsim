"""``changelog.d/`` fragments assemble into ``CHANGELOG.md`` without losing
anything, and the checks that enforce them are not no-ops (issue #668).

Every changelog entry used to be inserted at one anchor — the top of
``## [Unreleased]`` → ``### Fixed`` — so any two open branches edited the same
region of the same file and git stopped on a conflict. 17 of the 30 merges
before the issue touched the file, and hand-resolving the same region repeatedly
is where two shipped defects came from: #662 committed literal conflict markers,
#656 duplicated its entry verbatim. ``ci/changelog.py`` replaces the anchor with
one file per change.

Three properties are worth a test, and each one is a specific way this could be
worse than what it replaced:

* **Assembly loses nothing.** ``build`` is the only writer of ``CHANGELOG.md``
  now, and it runs once per release with no one reading its diff line by line.
  A dropped entry would reach a published GitHub Release.
* **The order is the assembler's, not the contributor's.** That is the half
  ``merge=union`` cannot do — it keeps both sides verbatim and will happily put
  the newest entry second.
* **The checks fail on the shape they describe.** #664 is the cautionary case:
  the ``check-merge-conflict`` hook meant to catch #662's markers returns success
  without opening a file unless git is mid-merge, so it was green in CI while
  the markers sat in the tree. A changelog gate that cannot fail is the honour
  system with a checkmark on it.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402
from test_changelog_structure import CANONICAL  # noqa: E402

REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci" / "changelog.py"

if not SCRIPT.is_file():  # an installed wheel carries no ci/ directory
    pytest.skip("ci/changelog.py lives in the source root", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bngsim_ci_changelog", SCRIPT)
assert _spec and _spec.loader
changelog = importlib.util.module_from_spec(_spec)
# Registered before execution: ``@dataclass`` resolves annotations through
# ``sys.modules[cls.__module__]`` and raises on a module that is not there yet.
sys.modules[_spec.name] = changelog
_spec.loader.exec_module(changelog)

#: A changelog with one hand-written ``[Unreleased]`` entry in it, which is the
#: state this migration starts from and the one ``build`` has to preserve.
SYNTHETIC = """\
# Changelog

Preamble that must survive.

## [Unreleased]

Pointer paragraph.

### Fixed

- **A hand-written entry that predates fragments.** Its second line.

## [0.1.0] - 2026-01-01

### Added

- **The first release.**
"""


def _fragment(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestTheRepositorysOwnFragments:
    def test_they_are_well_formed(self):
        """The check, run against the real directory. A fragment that trips this
        would otherwise trip the `changelog` workflow on someone else's branch."""
        assert changelog.validate() == []

    def test_the_command_line_entry_point_agrees(self):
        """The functional half. `validate()` passing proves the logic; this
        proves the argument parsing, the exit code and the path resolution that
        CI and the pre-commit hook actually invoke."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "check"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert done.returncode == 0, done.stderr

    def test_the_category_vocabulary_is_keep_a_changelogs(self):
        """Two lists of the same six names, in two files. ``CATEGORIES`` names
        the filename suffixes and ``CANONICAL`` the headings they render to, so
        a category added to one and not the other is a fragment that assembles
        under a heading ``test_changelog_structure`` rejects."""
        assert [c.title() for c in changelog.CATEGORIES] == CANONICAL


class TestFragmentNames:
    @pytest.mark.parametrize(
        "name",
        [
            "659.fixed.md",
            "1.added.md",
            "659.fixed.2.md",
            "659.security.md",
            "+no-issue-number.changed.md",
            "+a.removed.md",
        ],
    )
    def test_the_documented_forms_parse(self, name):
        assert changelog.FRAGMENT_NAME.match(name), name

    @pytest.mark.parametrize(
        "name",
        [
            "659.md",  # no category
            "659.Fixed.md",  # the heading, not the suffix
            "659.tooling.md",  # outside Keep a Changelog
            "fixed.md",  # no issue
            "659.fixed.txt",  # not markdown
            "659.fixed",  # no extension
            "no-issue.fixed.md",  # a slug needs its leading +
            "+.fixed.md",  # an empty slug
        ],
    )
    def test_the_undocumented_forms_do_not(self, name):
        assert not changelog.FRAGMENT_NAME.match(name), name


class TestAssembly:
    def test_the_order_is_the_assemblers(self, tmp_path):
        """Highest issue number first, issueless fragments last. This is what
        ``merge=union`` cannot give: it keeps both sides verbatim, so whichever
        branch merged second lands second regardless of what it says."""
        for name in ("12.fixed.md", "300.fixed.md", "+zz.fixed.md", "+aa.fixed.md"):
            _fragment(tmp_path, name, f"- entry from {name}\n")
        assert [f.path.name for f in changelog.load(tmp_path)] == [
            "300.fixed.md",
            "12.fixed.md",
            "+aa.fixed.md",
            "+zz.fixed.md",
        ]

    def test_sections_come_out_in_keep_a_changelog_order(self, tmp_path):
        for category in reversed(changelog.CATEGORIES):
            _fragment(tmp_path, f"1.{category}.md", f"- a {category} entry\n")
        rendered = changelog.render(changelog.load(tmp_path))
        headings = re.findall(r"^### (.+)$", rendered, re.M)
        assert headings == CANONICAL

    def test_a_multi_line_entry_is_carried_through_verbatim(self, tmp_path):
        """Assembly is concatenation, not reflow. An entry carries nested
        ``  * `` lists and inline code, and reformatting them would put the
        assembler between what a contributor writes and what ships."""
        text = "- **A title.** Body.\n\n  * nested\n  * items\n\n  A second paragraph."
        _fragment(tmp_path, "7.added.md", text + "\n")
        assert text in changelog.render(changelog.load(tmp_path))


class TestBuildLosesNothing:
    def _built(self, tmp_path):
        for name, text in (
            ("300.fixed.md", "- **A staged fix.**\n"),
            ("300.added.md", "- **A staged addition.**\n"),
        ):
            _fragment(tmp_path, name, text)
        return changelog.build(SYNTHETIC, changelog.load(tmp_path), "0.2.0", "2026-02-02")

    def test_every_pre_existing_line_survives(self, tmp_path):
        """The property that matters most, stated the bluntest way it can be:
        ``build`` runs once per release, unwatched, and a dropped entry reaches
        a published Release."""
        built = self._built(tmp_path)
        missing = [
            line
            for line in SYNTHETIC.split("\n")
            if line.strip() and line != "## [Unreleased]" and line not in built.split("\n")
        ]
        assert not missing

    def test_hand_written_unreleased_prose_is_folded_in_not_replaced(self, tmp_path):
        """Five entries predate fragments and sit under ``[Unreleased]`` today.
        They are swept into the next release rather than orphaned — and the
        merge stays afterwards as the guarantee that this cannot silently drop
        prose someone wrote."""
        built = self._built(tmp_path)
        section = built.split("## [0.2.0] - 2026-02-02")[1].split("## [0.1.0]")[0]
        assert "- **A staged fix.**" in section
        assert "- **A hand-written entry that predates fragments.**" in section
        # Staged first, carried second: newest on top, the order the file has
        # always had.
        assert section.index("- **A staged fix.**") < section.index("- **A hand-written")

    def test_one_unreleased_heading_remains_and_it_is_empty(self, tmp_path):
        built = self._built(tmp_path)
        lines = built.split("\n")
        assert lines.count("## [Unreleased]") == 1
        unreleased = lines.index("## [Unreleased]")
        following = lines.index("## [0.2.0] - 2026-02-02")
        assert not [line for line in lines[unreleased:following] if line.startswith("### ")]
        assert "Pointer paragraph." in lines[unreleased:following]

    def test_release_notes_can_read_the_section_back(self, tmp_path):
        """The consumer. ``release.yml`` publishes exactly what
        ``ci/release_notes.py`` extracts, so an assembled section that parser
        cannot find is a release with no notes."""
        sys.path.insert(0, str(SCRIPT.parent))
        release_notes = importlib.import_module("release_notes")
        section = release_notes.section(self._built(tmp_path), "0.2.0")
        assert "- **A staged fix.**" in section
        assert "- **The first release.**" not in section


class TestTheLengthLimit:
    def test_it_is_enforced(self, tmp_path):
        _fragment(tmp_path, "1.fixed.md", "- " + "x" * changelog.MAX_FRAGMENT_CHARS + "\n")
        problems = changelog.validate(tmp_path)
        assert len(problems) == 1
        assert "over the" in problems[0]

    def test_a_fragment_at_the_limit_passes(self, tmp_path):
        _fragment(tmp_path, "1.fixed.md", "- " + "x" * (changelog.MAX_FRAGMENT_CHARS - 2) + "\n")
        assert changelog.validate(tmp_path) == []

    def test_assembly_refuses_a_release_whose_notes_would_publish_truncated(self):
        """The gate that 0.13.0 and 0.12.0 did not have. ``release_notes.py``
        truncates gracefully at a paragraph boundary and says so in a footer, so
        nothing downstream fails — which is why both of them shipped incomplete
        notes with no one noticing. Asked here, it is in front of the one person
        who can still shorten an entry."""
        sys.path.insert(0, str(SCRIPT.parent))
        release_notes = importlib.import_module("release_notes")
        wall = "\n\n".join(f"- **Entry {i}.** " + "x" * 2000 for i in range(80))
        oversized = f"# Changelog\n\n## [9.9.9] - 2026-02-02\n\n### Fixed\n\n{wall}\n"
        assert len(release_notes.section(oversized, "9.9.9")) > release_notes.LIMIT

        truncated, size, limit = changelog.would_truncate(oversized, "9.9.9")
        assert truncated and size <= limit == release_notes.LIMIT

        # And it says no when the answer is no: a gate that always fires is a
        # gate that gets passed --allow-truncation as a habit.
        small, small_size, _ = changelog.would_truncate(SYNTHETIC, "0.1.0")
        assert not small and 0 < small_size < limit

    def test_it_keeps_the_largest_plausible_release_inside_githubs_ceiling(self):
        """The arithmetic the limit comes from, asserted rather than asserted-to.

        GitHub rejects a release body over 125,000 characters and
        ``release.yml`` publishes the version's section as that body. 0.13.0 and
        0.12.0 both exceeded it and shipped notes truncated at a paragraph
        boundary. The entry count is read from the file rather than pinned, so
        a future release that carries more entries than any so far re-runs the
        sum instead of inheriting a stale one.
        """
        sys.path.insert(0, str(SCRIPT.parent))
        release_notes = importlib.import_module("release_notes")
        text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        lines = text.split("\n")
        starts = [i for i, line in enumerate(lines) if line.startswith("## [")] + [len(lines)]
        busiest = max(
            sum(1 for line in lines[a + 1 : b] if line.startswith("- "))
            for a, b in zip(starts, starts[1:], strict=False)
        )
        assert busiest >= 50, "the corpus of releases shrank; re-derive the limit"
        assert busiest * changelog.MAX_FRAGMENT_CHARS < release_notes.LIMIT / 1.5


class TestTheCheckFailsOnTheShapeItDescribes:
    """The guard on the guard (#664). Each assertion above is only worth having
    if the check reports the defect it names, so one of each is run through it."""

    @pytest.mark.parametrize(
        ("name", "text", "expected"),
        [
            ("659.tooling.md", "- fine\n", "does not parse"),
            ("659.fixed.md", "", "is empty"),
            ("659.fixed.md", "No bullet marker.\n", "must start with '- '"),
            ("659.fixed.md", "- first\nnot indented\n", "indented by two spaces"),
        ],
    )
    def test_each_defect_is_reported(self, tmp_path, name, text, expected):
        _fragment(tmp_path, name, text)
        problems = changelog.validate(tmp_path)
        assert len(problems) == 1, problems
        assert expected in problems[0]

    def test_a_duplicate_fragment_is_reported(self, tmp_path):
        """#656's defect in the new shape. Two files cannot collide on a name,
        but two names can carry one entry twice."""
        _fragment(tmp_path, "659.fixed.md", "- once\n")
        _fragment(tmp_path, "0659.fixed.md", "- once\n")
        assert any("duplicates" in p for p in changelog.validate(tmp_path))

    def test_a_well_formed_fragment_is_not_reported(self, tmp_path):
        """A check that cries wolf gets bypassed, so the negative case is part
        of the guard."""
        _fragment(tmp_path, "659.fixed.md", "- **A title.** A body.\n  A continuation.\n")
        assert changelog.validate(tmp_path) == []


class TestTheBranchMustStageItsOwnEntry:
    @pytest.mark.parametrize(
        "changed",
        [
            ["python/bngsim/_result.py"],
            ["src/cvode_simulator.cpp", "python/tests/test_x.py"],
            ["include/bngsim/model.hpp"],
        ],
    )
    def test_a_source_change_without_a_fragment_is_refused(self, changed):
        problems = changelog.required(changed)
        assert len(problems) == 1
        assert "stages no changelog entry" in problems[0]

    def test_a_source_change_with_a_fragment_passes(self):
        assert changelog.required(["python/bngsim/_result.py", "changelog.d/560.fixed.md"]) == []

    @pytest.mark.parametrize(
        "changed",
        [
            ["python/tests/test_x.py"],
            ["docs/reference/expressions.md"],
            [".github/workflows/lint.yml"],
            ["changelog.d/README.md"],
        ],
    )
    def test_a_change_with_nothing_to_say_is_not_asked_to_say_it(self, changed):
        """Test-only, docs-only and CI-only branches are exempt by construction
        rather than by label, so the label stays rare enough to mean something."""
        assert changelog.required(changed) == []

    def test_editing_the_shared_file_is_refused_even_with_a_fragment(self):
        """The half that removes the conflict rather than relocating it. A
        branch that stages a fragment *and* edits CHANGELOG.md still collides
        with every other open branch."""
        problems = changelog.required(
            ["python/bngsim/_result.py", "changelog.d/560.fixed.md", "CHANGELOG.md"]
        )
        assert len(problems) == 1
        assert "edits CHANGELOG.md" in problems[0]

    def test_a_fragment_with_an_unparseable_name_does_not_count(self):
        """Otherwise ``changelog.d/notes.md`` satisfies the gate and fails the
        check in a different job, which is a confusing way to say one thing."""
        assert changelog.required(["src/expression.cpp", "changelog.d/notes.md"])

    def test_the_command_line_entry_point_reads_a_file_list(self):
        """The CI path end to end: the workflow pipes ``gh api … --jq`` output
        into this. A flag rename here is a gate that passes on every branch."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "required", "--files", "-"],
            input="src/expression.cpp\n",
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert done.returncode == 1
        assert "stages no changelog entry" in done.stderr

    def test_an_empty_file_list_fails_closed(self):
        """The failure a gate cannot afford. A pipeline's exit status is the
        last command's, so a `gh api` that failed would feed nothing to a
        checker that then objects to nothing — green, having read nothing.
        That is #664's defect, and the workflow's `set -o pipefail` is the
        other half of this guard."""
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "required", "--files", "-"],
            input="",
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert done.returncode == 1
        assert "nothing was actually checked" in done.stderr

    def test_the_workflow_does_not_swallow_a_failed_api_call(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "changelog.yml"
        if not workflow.is_file():
            pytest.skip(".github/workflows is not in this checkout")
        body = "\n".join(
            line
            for line in workflow.read_text(encoding="utf-8").split("\n")
            if not line.lstrip().startswith("#")
        )
        assert "pipefail" in body


class TestTheGatesAreWiredUp:
    """Textual, like the other CI-coverage modules here, and for the same
    reason: a script only protects the repository if something runs it."""

    def test_the_workflow_invokes_both_halves(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "changelog.yml"
        if not workflow.is_file():
            pytest.skip(".github/workflows is not in this checkout")
        body = "\n".join(
            line
            for line in workflow.read_text(encoding="utf-8").split("\n")
            if not line.lstrip().startswith("#")
        )
        assert "ci/changelog.py check" in body
        assert "ci/changelog.py required" in body
        assert "pull_request" in body
        # The escape hatch is a label, so the gate has to re-run when one is
        # applied. Without these two types the label clears nothing until an
        # unrelated commit lands, which is a workflow asking for empty commits
        # to use its own exemption. Found on this workflow's own pull request.
        assert "labeled" in body and "unlabeled" in body

    def test_every_run_finishes_and_reads_the_label_when_it_runs(self):
        """``gh pr create --label`` fires ``opened`` and ``labeled`` together.
        Under a concurrency group one run was cancelled, and a cancelled run
        stays in the commit's rollup as a red ``fragment staged`` beside the
        one that passed (#1407, #1408). With both runs finishing, both have to
        agree, so the label comes from the API at run time and not from each
        event's payload, which is a snapshot of the moment it fired."""
        workflow = REPO_ROOT / ".github" / "workflows" / "changelog.yml"
        if not workflow.is_file():
            pytest.skip(".github/workflows is not in this checkout")
        body = "\n".join(
            line
            for line in workflow.read_text(encoding="utf-8").split("\n")
            if not line.lstrip().startswith("#")
        )
        assert "concurrency" not in body
        assert "github.event.pull_request.labels" not in body
        assert "changelog exempt" in body

    def test_the_pre_commit_hook_runs_the_check_at_both_gating_stages(self):
        """``stages:`` is absent on purpose, so the hook inherits
        ``default_stages: [pre-commit, pre-push]``. A hook that ran only on
        commit would be skipped by an amend or a ``--no-verify`` commit and
        would first be heard from in CI."""
        config = REPO_ROOT / ".pre-commit-config.yaml"
        if not config.is_file():
            pytest.skip(".pre-commit-config.yaml is not in this checkout")
        text = config.read_text(encoding="utf-8")
        hook = text.split("- id: changelog-fragments")[1].split("- id: ")[0]
        assert "ci/changelog.py check" in hook
        assert "always_run: true" in hook
        assert "stages:" not in hook

    def test_contributing_sends_contributors_to_the_fragment_directory(self):
        """It sent them to ``CHANGELOG.md`` for 579 entries; the one-line ask is
        what the whole honour system rested on."""
        contributing = REPO_ROOT / "CONTRIBUTING.md"
        if not contributing.is_file():
            pytest.skip("CONTRIBUTING.md is not in this checkout")
        text = contributing.read_text(encoding="utf-8")
        assert "changelog.d/" in text
        assert "Do not edit [`CHANGELOG.md`](CHANGELOG.md)" in text
