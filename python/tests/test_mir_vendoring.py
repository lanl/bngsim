"""File-based guardrails for BNGsim's pinned MIR micro-JIT vendoring (GH #78).

MIR is vendored as a pruned source tree (bngsim/third_party/mir) for the
opt-in codegen JIT backend. These checks are the regression tripwire that the
checked-in tree still matches its VENDOR.json anchors and the prune invariants —
the same role test_exprtk_vendoring.py / test_sundials_vendoring.py play for
their trees. Refresh with bngsim/scripts/vendor_mir.py (see MIR_VENDORING.md).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BNGSIM_ROOT = Path(__file__).resolve().parents[2]
MIR_DIR = BNGSIM_ROOT / "third_party" / "mir"
MIR_VENDOR_METADATA = MIR_DIR / "VENDOR.json"
CMAKE_LISTS = BNGSIM_ROOT / "CMakeLists.txt"

PINNED_COMMIT = "99c65079038f3ba9242ef646f308c266cfd7a8e5"
UPSTREAM_REMOTE = "https://github.com/vnmakarov/mir.git"

# The three translation units CMake compiles.
TRANSLATION_UNITS = ["mir.c", "mir-gen.c", "c2mir/c2mir.c"]

# Files whose SHA256 is anchored in VENDOR.json files{}.
ANCHORED_FILES = [
    "mir.c",
    "mir-gen.c",
    "c2mir/c2mir.c",
    "mir.h",
    "mir-gen.h",
    "c2mir/c2mir.h",
    "LICENSE",
]

# Standalone CLI/test drivers and build files the prune must drop.
PRUNED_FILES = [
    "sieve.c",
    "mir-bin-driver.c",
    "mir-bin-run.c",
    "mir-gen-stub.c",
    "c2mir/c2mir-driver.c",
    "CMakeLists.txt",
    "GNUmakefile",
]

# Whole upstream subtrees the prune must drop.
PRUNED_DIRS = [
    "adt-tests",
    "c-benchmarks",
    "c-tests",
    "llvm2mir",
    "mir2c",
    "mir-tests",
    "mir-utils",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mir_vendor_metadata_shape():
    metadata = json.loads(MIR_VENDOR_METADATA.read_text())

    assert metadata["name"] == "MIR"
    assert metadata["vendored_path"] == "bngsim/third_party/mir"
    assert metadata["source"]["authoritative_remote"] == UPSTREAM_REMOTE
    assert metadata["source"]["upstream_remote"] == UPSTREAM_REMOTE
    assert metadata["source"]["commit"] == PINNED_COMMIT
    assert metadata["license"]["spdx"] == "MIT"
    assert metadata["build"]["translation_units"] == TRANSLATION_UNITS
    assert metadata["local_carries"] == []
    # The hand-authored prune rationale must survive refreshes.
    assert metadata["pruning"]["included_paths"]
    assert metadata["pruning"]["excluded_paths"]


def test_mir_anchored_checksums_match_tree():
    """The VENDOR.json anchors must match the checked-in files byte-for-byte."""
    metadata = json.loads(MIR_VENDOR_METADATA.read_text())
    recorded = metadata["files"]

    for rel in ANCHORED_FILES:
        path = MIR_DIR / rel
        assert path.is_file(), f"anchored file missing from vendored tree: {rel}"
        assert recorded[rel]["sha256"] == _sha256(path), f"checksum drift for {rel}"


def test_mir_translation_units_present():
    for rel in TRANSLATION_UNITS:
        assert (MIR_DIR / rel).is_file(), f"missing translation unit: {rel}"
    assert (MIR_DIR / "LICENSE").is_file()
    # BNGsim-owned files that a refresh must preserve.
    assert (MIR_DIR / "README.md.bngsim").is_file()


def test_mir_pruned_paths_absent():
    for rel in PRUNED_FILES:
        assert not (MIR_DIR / rel).exists(), f"pruned file should be absent: {rel}"
    for rel in PRUNED_DIRS:
        assert not (MIR_DIR / rel).exists(), f"pruned subtree should be absent: {rel}"


def test_cmake_gates_mir_behind_option():
    text = CMAKE_LISTS.read_text()

    assert (
        'option(BNGSIM_ENABLE_MIR "Build the vendored MIR micro-JIT backend '
        'for the codegen RHS (GH #78, prototype)" OFF)'
    ) in text
    for rel in TRANSLATION_UNITS:
        assert f"third_party/mir/{rel}" in text
    assert "BNGSIM_HAS_MIR=1" in text
