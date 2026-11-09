"""CI takes GitHub release downloads from the Actions cache (#1607).

On 2026-11-09 GitHub's release downloads answered 504 for half an hour, and
every leg of a pull request failed within a minute or two of starting: each one
downloaded SUNDIALS' source archive, the uv binary, BNG2.pl or a reference
archive from a GitHub release. Each now comes from somewhere that outage does
not reach:

* SUNDIALS, BNG2.pl and the reference archives through
  .github/actions/cached-download, which keys the Actions cache on the file's
  sha256 and downloads only on a miss;
* uv through setup-uv 10, which downloads from Astral's mirror and falls back to
  GitHub only if the mirror fails.

These tests read the workflows as text (see _ci_workflow.py for why) and pin
what keeps that true: every job that builds bngsim from source takes SUNDIALS
from the cache before it builds, no workflow step downloads a GitHub release
asset itself, setup-uv stays on a version with the mirror, and a workflow
filtered by path also runs when the steps it builds through change.

They read workflow YAML only. Downloads made by code the steps run are outside
them, and two remain, both behind a cache: the corpus job's BioModels mirror
(fetch_biomodels_mirror.py, on a corpus-cache miss), and golden-verify's
`--group parity` sync, whose PyBioNetGen build downloads BioNetGen from GitHub's
releases on a uv-cache miss (a uv.lock change). uv's managed Python comes from
Astral's mirror, not from GitHub.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ci_workflow import WORKFLOWS, paths_filter_entries, strip_comments  # noqa: E402

REPO_ROOT = WORKFLOWS.parent.parent
ACTIONS = REPO_ROOT / ".github" / "actions"

pytestmark = pytest.mark.skipif(
    not WORKFLOWS.is_dir(), reason=".github/workflows is not in this checkout (installed package)"
)

SUNDIALS_STEP = "uses: ./.github/actions/sundials-archive"
#: A step that builds bngsim, and so fetches SUNDIALS through CMake. The
#: from-source pip installs pass -v; the others install tools.
BUILD = re.compile(r"\buv sync\b|\bcmake -S \.|\bpip install -v\b")


def _workflows() -> dict[str, str]:
    return {p.name: strip_comments(p.read_text()) for p in sorted(WORKFLOWS.glob("*.yml"))}


def _jobs(text: str) -> dict[str, str]:
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):[ \t]*$", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _building_jobs() -> dict[str, str]:
    return {
        f"{name}::{job}": body
        for name, text in _workflows().items()
        for job, body in _jobs(text).items()
        if BUILD.search(body)
    }


def test_the_build_jobs_are_found():
    """Guards the BUILD pattern: a pattern that matched nothing would pass the
    test below with nothing checked."""
    found = set(_building_jobs())
    assert {
        "python-tests.yml::python-tests",
        "mir.yml::mir-tests",
        "native-tests.yml::native-tests",
        "sanitizers.yml::cpp",
        "windows-tail.yml::tail-tests",
        "nightly-parity.yml::rr-ssa",
        "nightly-parity.yml::differential-nf",
    } <= found
    assert len(found) >= 19


@pytest.mark.parametrize("job", sorted(_building_jobs()) if WORKFLOWS.is_dir() else [])
def test_every_source_build_takes_sundials_from_the_cache_first(job):
    body = _building_jobs()[job]
    assert SUNDIALS_STEP in body, f"{job} builds bngsim without {SUNDIALS_STEP}"
    assert body.index(SUNDIALS_STEP) < BUILD.search(body).start(), (
        f"{job} builds before it takes SUNDIALS from the cache"
    )


def test_no_step_downloads_a_github_release_asset_itself():
    """A release URL may only be a value handed to cached-download: an env
    entry whose only use is that step's ``url:`` input."""
    for name, text in _workflows().items():
        for line in text.splitlines():
            if "releases/download" not in line:
                continue
            m = re.match(r"^\s+([A-Z][A-Z0-9_]*): https://\S+/releases/download/\S+$", line)
            assert m, f"{name}: a release URL outside an env entry: {line.strip()}"
            key = m.group(1)
            uses = [ln.strip() for ln in text.splitlines() if key in ln and ln != line]
            assert uses, f"{name}: {key} is never used"
            for use in uses:
                assert use == f"url: ${{{{ env.{key} }}}}", f"{name}: {key} used as {use!r}"
            steps = re.findall(r"uses: \./\.github/actions/cached-download\n((?:\s+.+\n)+)", text)
            for block in steps:
                if f"env.{key}" in block:
                    break
            else:
                pytest.fail(f"{name}: {key} is not a cached-download url")


def test_setup_uv_downloads_from_astrals_mirror_and_prunes_its_cache():
    """setup-uv before 10 downloads uv only from GitHub's releases. Pruning keeps
    the uv cache the size it was under v5, whose default it was."""
    refs = []
    for name, text in _workflows().items():
        for m in re.finditer(r"uses: astral-sh/setup-uv@(\S+)\n((?:\s{6,}.+\n)*)", text):
            refs.append(m.group(1))
            major = re.fullmatch(r"v(\d+)(?:\.\d+){0,2}", m.group(1))
            assert major and int(major.group(1)) >= 10, f"{name}: setup-uv@{m.group(1)}"
            assert "download-from-astral-mirror: false" not in m.group(2), name
            assert "prune-cache: true" in m.group(2), f"{name}: setup-uv@{m.group(1)}"
    assert len(refs) >= 14


def _uses(text: str) -> set[str]:
    return set(re.findall(r"uses: \./\.github/actions/([a-z-]+)", text))


#: What a workflow needs in its paths filter when it uses each step: the step,
#: and what the step runs.
NEEDS = {
    "sundials-archive": {
        ".github/actions/sundials-archive/**",
        ".github/actions/cached-download/**",
        "ci/cached_download.py",
    },
    "cached-download": {".github/actions/cached-download/**", "ci/cached_download.py"},
    "reference-side": {
        ".github/actions/reference-side/**",
        ".github/actions/cached-download/**",
        "ci/cached_download.py",
    },
}


def test_a_path_filtered_workflow_runs_when_its_steps_change():
    checked = 0
    for name, text in _workflows().items():
        entries = set(paths_filter_entries(text))
        if not entries:
            continue  # runs on every pull request
        for step in _uses(text):
            missing = NEEDS[step] - entries
            assert not missing, f"{name} uses {step} but its paths filter lacks {sorted(missing)}"
            checked += 1
    assert checked >= 8


def test_every_local_step_is_known():
    """NEEDS names every step under .github/actions, so a new one is checked too."""
    assert {p.parent.name for p in ACTIONS.glob("*/action.yml")} == set(NEEDS)
    for text in _workflows().values():
        assert _uses(text) <= set(NEEDS)


def test_the_steps_read_the_pins_cmake_and_the_manifest_read():
    """sundials-archive reads the URL and sha256 CMakeLists.txt reads from
    VENDOR.json, and exports the variable source_archive.cmake reads."""
    action = (ACTIONS / "sundials-archive" / "action.yml").read_text()
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    module = (REPO_ROOT / "third_party" / "sundials" / "source_archive.cmake").read_text()
    assert '["source"]["authoritative_release_asset_url"]' in action
    assert "BNGSIM_SUNDIALS_SOURCE_URL source authoritative_release_asset_url" in cmake
    assert '["files"]["source_archive"]["sha256"]' in action
    assert "BNGSIM_SUNDIALS_SOURCE_SHA256 files source_archive sha256" in cmake
    assert "BNGSIM_SUNDIALS_ARCHIVE=" in action
    assert "$ENV{BNGSIM_SUNDIALS_ARCHIVE}" in module
    cached = (ACTIONS / "cached-download" / "action.yml").read_text()
    # Keyed on the content, so a new pin is a new entry.
    assert cached.count("key: download-${{ inputs.sha256 }}") == 2
    assert "ci/cached_download.py" in cached
