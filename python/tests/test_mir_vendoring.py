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

PINNED_COMMIT = "439a90a2d7d9f38a70e8d9a6ebeabd900d66e920"
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

    Three carries today: issue #1215's two (in mir.c and c2mir/c2mir.c) and
    issue #1216's loop-padding NOPs in mir-gen-x86_64.c. Issue #413's was
    dropped when upstream merged it (vnmakarov/mir#468). Checksum anchors cannot
    guard a carry — a refresh rewrites the file from upstream AND re-anchors it,
    so the patch vanishes with the tree still matching its own metadata. Each
    entry names a ``marker`` comment the patch itself carries; grepping for it is
    the one tripwire a re-vendor cannot satisfy by accident. Same check as
    ``scripts/vendor_mir.py --check``.
    """
    metadata = json.loads(MIR_VENDOR_METADATA.read_text())
    carries = metadata["local_carries"]
    assert isinstance(carries, list)

    for carry in carries:
        for field in ("file", "marker", "issue", "upstream_status", "summary", "reapply"):
            assert carry.get(field), f"local carry missing {field!r}: {carry}"
        path = MIR_DIR / carry["file"]
        assert path.is_file(), f"local carry names a missing file: {carry['file']}"
        assert carry["marker"] in path.read_text(), (
            f"{carry['file']} no longer carries {carry['marker']!r} — the carry was "
            f"dropped (a refresh?). See {carry['issue']}: {carry['reapply']}"
        )


def test_mir_gen_spill_reload_table_is_bounds_checked():
    """Issue #413: the fix, now upstream, pinned to the property it restores.

    ``try_spilled_reg_mem`` rewrites every operand naming one spilled register
    into a memory operand and records the rewritten indices in a fixed
    ``op_nums[MAX_INSN_RELOAD_MEM_OPS]`` so it can undo them. Upstream used to
    size that table at 2 and bound the loop with ``gen_assert`` only — i.e.
    ``assert``, which -DNDEBUG deletes from every release build — and an insn CAN
    name the register three times: ``x = x * x`` reaches the allocator as
    ``dmul r, r, r``. The third store then landed past the array, which glibc
    reports as ``*** stack smashing detected ***``. BNGsim carried the fix until
    upstream merged it as vnmakarov/mir#468; a refresh that fails here means
    upstream lost it, and the carry has to come back (MIR_VENDORING.md, "Past
    carries").

    Both halves of the fix are asserted here, because either alone is
    insufficient. The table has to be 3 for the three-operand case to be handled
    rather than declined, and the runtime check has to stay because 3 is not a
    provable bound — the call site is reached for ``MIR_USE`` too, and ``MIR_USE``
    has unbounded ``nops``.

    A source-level assertion, because the behavior it guards needs -O2 MIR_gen
    on a spilling function; test_codegen_jit_self_multiply.py is the end-to-end
    half and runs only where the JIT is built.
    """
    src = (MIR_DIR / "mir-gen.c").read_text()
    # Comments out, so prose quoting the old line (as the #413 carry's comment
    # did) cannot report the fix as the bug.
    code = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    start = code.index("static int try_spilled_reg_mem")
    body = code[start : code.index("\nstatic ", start + 1)]

    assert "#define MAX_INSN_RELOAD_MEM_OPS 3" in code, (
        "the #413 fix no longer sizes op_nums for a three-operand insn naming "
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


# The NOP Intel recommends for each length from 0 to 7 bytes (Intel 64 and IA-32
# Architectures Software Developer's Manual, Vol. 2B, "NOP").
INTEL_RECOMMENDED_NOPS = [
    b"",
    b"\x90",
    b"\x66\x90",
    b"\x0f\x1f\x00",
    b"\x0f\x1f\x40\x00",
    b"\x0f\x1f\x44\x00\x00",
    b"\x66\x0f\x1f\x44\x00\x00",
    b"\x0f\x1f\x80\x00\x00\x00\x00",
]


def test_mir_x86_64_loop_padding_is_intels_recommended_nops():
    """Issue #1216: the NOPs MIR pads a loop head with must run under Rosetta 2.

    From -O2, ``target_translate`` aligns the head of each innermost loop to a
    ``LOOP_ALIGN`` (8) byte boundary, padding with ``nop_pats[n]`` for n = 0..7.
    Upstream's 3- and 4-byte entries are the register forms ``0F 1F C0`` (nopl
    %eax) and ``48 0F 1F C0`` (nopq %rax). An x86-64 CPU runs them, but Rosetta 2
    raises SIGILL on both, so an x86_64 Python on Apple Silicon died the first
    time it ran a 3- or 4-byte pad, and never at -O0 or -O1, which pad nothing.
    The carry (VENDOR.json ``local_carries``) uses the memory forms Intel
    recommends, as upstream's 5- to 7-byte entries already are.

    A source-level assertion, because no CI leg runs x86_64 code under Rosetta.
    """
    src = (MIR_DIR / "mir-gen-x86_64.c").read_text()
    # Comments out, so the carry's note quoting the old bytes is not read as an
    # entry.
    code = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    # Exactly one definition, of 8: a later #undef and #define, or one per #if
    # branch, would let a pad outgrow the table checked below.
    aligns = re.findall(r"^[ \t]*#[ \t]*(define|undef)[ \t]+LOOP_ALIGN\b(.*)$", code, re.M)
    assert [(d, v.strip()) for d, v in aligns] == [("define", "8")], (
        "LOOP_ALIGN changed: padding can now take more than 7 bytes, so "
        f"INTEL_RECOMMENDED_NOPS has to grow with it ({aligns})"
    )
    start = code.index("static const char *nop_pats[] = {")
    body = code[code.index("{", start) + 1 : code.index("};", start)]

    entries = []
    for item in body.split(","):
        if not item.strip():
            continue
        text = "".join(re.findall(r'"([^"]*)"', item))
        assert re.fullmatch(r"(\\x[0-9a-fA-F]{2})*", text), f"nop_pats entry {item.strip()!r}"
        entries.append(bytes(int(h, 16) for h in re.findall(r"\\x([0-9a-fA-F]{2})", text)))

    wrong = [
        f"{n} bytes: {got.hex(' ')} for {want.hex(' ')}"
        for n, (got, want) in enumerate(zip(entries, INTEL_RECOMMENDED_NOPS, strict=False))
        if got != want
    ]
    assert len(entries) == len(INTEL_RECOMMENDED_NOPS) and not wrong, (
        f"nop_pats is not Intel's recommended NOPs ({len(entries)} entries): "
        f"{'; '.join(wrong)}. Rosetta 2 raises SIGILL on the register forms "
        "0f 1f c0 and 48 0f 1f c0; see VENDOR.json local_carries (#1216)."
    )


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


# ── Issue #1215: the address space each MIR context reserves for its code ────
# third_party/mir/README.md.bngsim says why CMake sets 32 MB. The live models'
# address space is measured in test_mir_equivalence.py, which runs only on a
# MIR build: a skip here would be one run_tests.sh's walk-up probe counts
# against it.


def test_cmake_sizes_the_mir_code_reservation():
    text = CMAKE_LISTS.read_text()
    assert 'set(BNGSIM_MIR_CODE_RESERVE_MB "32" CACHE STRING' in text
    assert "MIR_CODE_RESERVE_SIZE=${BNGSIM_MIR_CODE_RESERVE_BYTES}" in text
    # PUBLIC: MirJit's code allocator tells the reservation apart by its size,
    # so it must see the size mir.c was built with.
    assert "target_compile_definitions(bngsim_mir PUBLIC\n" in text
    assert "if(BNGSIM_MIR_CODE_RESERVE_MB GREATER 2048)" in text
    # The vendored default it overrides is still upstream's, under the #ifndef
    # that lets the define override it, so a refresh that moves either is seen
    # here. bngsim_mir compiles with -w: without the guard, mir.c's own #define
    # would replace CMake's with no warning, and every context would reserve
    # 128 MB again.
    mir_c = (MIR_DIR / "mir.c").read_text()
    assert (
        "#ifndef MIR_CODE_RESERVE_SIZE\n"
        "#if UINTPTR_MAX == 0xffffffffffffffffu && !defined(_WIN32)\n"
        "#define MIR_CODE_RESERVE_SIZE (128 * 1024 * 1024)\n"
    ) in mir_c
    assert mir_c.count("#define MIR_CODE_RESERVE_SIZE") == 2


def test_c2mir_checks_every_struct_it_mallocs():
    """Issue #1215: c2mir's preprocessor allocated its macro, macro-call and #if
    records with plain malloc and wrote through the result unchecked, so running
    out there was a segfault the allocator MirJit hands MIR never sees. The
    local carry checks each; this is the check a refresh that adds another
    unchecked one fails."""
    lines = (MIR_DIR / "c2mir" / "c2mir.c").read_text().splitlines()
    struct_malloc = re.compile(r"\bmalloc \(sizeof \(struct \w+\)\);")
    mallocs = [i for i, line in enumerate(lines) if struct_malloc.search(line)]
    assert len(mallocs) >= 3
    for i in mallocs:
        var = re.search(r"(\w+) = malloc \(", lines[i]).group(1)
        following = [line for line in lines[i + 1 : i + 5] if line.strip()]
        assert following[0].strip().startswith(f"if ({var} == NULL)"), (i + 1, lines[i])
        assert "(MIR_alloc_error," in following[1], (i + 1, lines[i])
