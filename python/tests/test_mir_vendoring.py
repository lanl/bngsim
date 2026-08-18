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
import re
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
    # The hand-authored prune rationale must survive refreshes.
    assert metadata["pruning"]["included_paths"]
    assert metadata["pruning"]["excluded_paths"]


def test_mir_local_carries_are_still_applied():
    """Every declared local carry must still be present in the vendored file.

    The tree is no longer stock upstream: issue #413 carries a bounds check in
    mir-gen.c's ``try_spilled_reg_mem``. Checksum anchors cannot guard a carry —
    a refresh rewrites the file from upstream AND re-anchors it, so the patch
    vanishes with the tree still matching its own metadata. Each entry names a
    ``marker`` comment the patch itself carries; grepping for it is the one
    tripwire a re-vendor cannot satisfy by accident. Same check as
    ``scripts/vendor_mir.py --check``.
    """
    metadata = json.loads(MIR_VENDOR_METADATA.read_text())
    carries = metadata["local_carries"]
    assert isinstance(carries, list)

    for carry in carries:
        for field in ("file", "marker", "issue", "summary", "reapply"):
            assert carry.get(field), f"local carry missing {field!r}: {carry}"
        path = MIR_DIR / carry["file"]
        assert path.is_file(), f"local carry names a missing file: {carry['file']}"
        assert carry["marker"] in path.read_text(), (
            f"{carry['file']} no longer carries {carry['marker']!r} — the carry was "
            f"dropped (a refresh?). See {carry['issue']}: {carry['reapply']}"
        )


def test_mir_gen_spill_reload_table_is_bounds_checked():
    """Issue #413: the #413 carry, pinned to the property it restores.

    ``try_spilled_reg_mem`` rewrites every operand naming one spilled register
    into a memory operand and records the rewritten indices in a fixed
    ``op_nums[MAX_INSN_RELOAD_MEM_OPS]`` so it can undo them. Upstream sizes that
    table at 2 and bounds the loop with ``gen_assert`` only — i.e. ``assert``,
    which -DNDEBUG deletes from every release build — and an insn CAN name the
    register three times: ``x = x * x`` reaches the allocator as ``dmul r, r, r``.
    The third store then lands past the array, which glibc reports as
    ``*** stack smashing detected ***``.

    Both halves of the carry are asserted here, because either alone is
    insufficient. The table has to be 3 for the three-operand case to be handled
    rather than declined, and the runtime check has to stay because 3 is not a
    provable bound — the call site is reached for ``MIR_USE`` too, and ``MIR_USE``
    has unbounded ``nops``.

    A source-level assertion, because the behavior it guards needs -O2 MIR_gen
    on a spilling function; test_codegen_jit_self_multiply.py is the end-to-end
    half and runs only where the JIT is built.
    """
    src = (MIR_DIR / "mir-gen.c").read_text()
    # Comments out, because the carry's own comment quotes the upstream line it
    # replaced — matching prose here would report the fix as the bug.
    code = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    start = code.index("static int try_spilled_reg_mem")
    body = code[start : code.index("\nstatic ", start + 1)]

    assert "#define MAX_INSN_RELOAD_MEM_OPS 3" in code, (
        "the #413 carry no longer sizes op_nums for a three-operand insn naming "
        "one spilled register in every operand (`x = x * x`)"
    )
    assert "if (n >= MAX_INSN_RELOAD_MEM_OPS) {" in body, (
        "the #413 bounds check is gone from try_spilled_reg_mem — op_nums can be "
        "overrun again by an insn naming one spilled register more times than the "
        "table holds, which MIR_USE can do without limit"
    )
    # The check must not have been left as an assert(), which NDEBUG removes.
    recorded = body[body.index("int n = 0, op_nums") :]
    assert "gen_assert (n < MAX_INSN_RELOAD_MEM_OPS)" not in recorded


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
