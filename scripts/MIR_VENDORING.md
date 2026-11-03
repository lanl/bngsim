# MIR Vendoring

## Goal

BNGsim vendors a pruned copy of [MIR](https://github.com/vnmakarov/mir) (Vladimir
Makarov's lightweight JIT) as source under `bngsim/third_party/mir`. MIR ships
**c2mir**, a C11 frontend, so the *same C source `bngsim._codegen` already emits*
for the ODE RHS can be JIT-compiled in-process in ~1–2 ms instead of shelling out
to `cc -O3` + `dlopen` (~80 ms–seconds). It is a single ~1 MB C library — no LLVM,
no Rust toolchain — targeting x86-64 and aarch64 (Apple Silicon), so it adds no new
build dependency to the wheel pipeline (GH #78).

`bngsim/scripts/vendor_mir.py` is the supported path that prunes the tree and
re-anchors `bngsim/third_party/mir/VENDOR.json`. If this doc and `VENDOR.json`
disagree, trust `VENDOR.json`.

## Status And Scope

- The MIR backend is a **prototype**, gated behind the `BNGSIM_ENABLE_MIR` CMake
  option (**default OFF**). When MIR is off, the `cc`-subprocess + `dlopen`
  codegen backend and the ExprTk fallback are unchanged — nothing in a default
  build links or reads the MIR tree.
- MIR is scoped to **compiler-less distribution** (GH #142/#146): wherever a host
  C compiler exists, the `cc` codegen path wins end-to-end, so MIR stays opt-in
  and is not a default. The pin is immutable and refreshed deliberately, not
  routinely.

## Current Baseline

- `bngsim/third_party/mir/VENDOR.json` records the pinned upstream commit
  `439a90a2d7d9f38a70e8d9a6ebeabd900d66e920`, canonical remote
  `https://github.com/vnmakarov/mir.git`, branch `master`, the MIT license, the
  three built translation units, the prune set, the per-file SHA256 anchors, and
  the wrapper guardrails.
- `bngsim/third_party/mir/README.md.bngsim` is the BNGsim-specific note that lives
  alongside the tree (why it's here, what is built, the header-free-source rule).
  MIR's own upstream `README.md` is preserved next to it.
- The preferred disposable checkout is `/tmp/mir-vendor-candidate`.

## Source Of Truth

- Authoritative repository: `https://github.com/vnmakarov/mir.git`
- Authoritative branch: `master`
- Pinned commit: `439a90a2d7d9f38a70e8d9a6ebeabd900d66e920`
- `bngsim/scripts/vendor_mir.py`: only supported exporter into
  `bngsim/third_party/mir`
- `bngsim/third_party/mir/VENDOR.json`: authoritative record of the vendored
  commit, prune set, per-file checksums, and guardrails
- `bngsim/include/bngsim/mir_jit.hpp`: the `MirJit` shim that consumes MIR
- `bngsim/CMakeLists.txt`: the `BNGSIM_ENABLE_MIR` build wiring
- Local checkout paths are transport only. A local cache or clone is never
  authoritative by itself.

BNGsim pins an immutable upstream commit rather than a moving `master` because MIR
is effectively single-maintainer; vendoring must be restartable, reviewable, and
machine-independent.

## Safe Workspace Policy

- Prefer `/tmp/mir-vendor-candidate` for vendoring work.
- Prefer cloning directly from `https://github.com/vnmakarov/mir.git`.
- If you bootstrap the candidate from a local cache checkout, treat that cache as
  transport only and reset `origin` to the canonical remote before relying on it.
- Do not vendor from a dirty MIR checkout — `vendor_mir.py` refuses a checkout
  with local changes.
- Do not treat a found build tree as proof. Validate through the repo-supported
  build/test flows below.

## Rebuild The Candidate Checkout

Fresh clone, then check out the pinned commit from `VENDOR.json`:

```sh
git clone https://github.com/vnmakarov/mir.git /tmp/mir-vendor-candidate
git -C /tmp/mir-vendor-candidate checkout 439a90a2d7d9f38a70e8d9a6ebeabd900d66e920
git -C /tmp/mir-vendor-candidate remote -v
git -C /tmp/mir-vendor-candidate rev-parse HEAD
```

To move the pin to a newer commit, check out (or fetch) the new ref deliberately
and pass it to `--ref` below; record it in `VENDOR.json` only after the
validation passes.

## Standard Refresh Workflow

### 1. Preview Before Writing

From the PyBNF repo root. In a standalone checkout, run from inside it and drop
the `bngsim/` prefix: from the directory above a checkout with another name,
`bngsim/scripts/...` is some other checkout's script, and it would act on that
checkout (GH #864).

```sh
python3 bngsim/scripts/vendor_mir.py \
  --mir-repo /tmp/mir-vendor-candidate \
  --ref master \
  --summary
```

The summary verifies:

- canonical remote / candidate HEAD
- resolved source ref and commit
- the pinned baseline commit and whether this refresh moves the pin
- upstream file count, kept-after-prune count vs the current vendored tree
- which of the anchored files would change

`--check` is the offline tripwire — it needs no MIR checkout and verifies the
*checked-in* tree still matches `VENDOR.json` and the prune invariants:

```sh
python3 bngsim/scripts/vendor_mir.py --check
```

(`python/tests/test_mir_vendoring.py` runs the same checks inside the suite.)

### 2. Refresh `bngsim/third_party/mir`

```sh
python3 bngsim/scripts/vendor_mir.py \
  --mir-repo /tmp/mir-vendor-candidate \
  --ref master
```

This re-prunes the tree from the resolved commit and updates the *dynamic*
`VENDOR.json` fields (`source.branch_or_ref`, `source.commit`, `imported_at_utc`,
and the `files{}` checksums). The hand-authored `purpose` / `build` / `guardrails`
/ `pruning` / `notes` are preserved, as is `README.md.bngsim`.

### 3. Validate Inside BNGsim

MIR is off by default, so validation means building it on and exercising the JIT
codegen path. The backend is selected by `-DBNGSIM_ENABLE_MIR=ON`; pass it through
`CMAKE_ARGS`, which both the editable install and a packaged wheel honor:

```sh
CMAKE_ARGS="-DBNGSIM_ENABLE_MIR=ON" uv pip install --no-deps -e bngsim
```

The in-repo `scripts/rebuild_editable.py` is a convenience wrapper that reads the
`BNGSIM_ENABLE_MIR=1` env var and adds the same define:

```sh
BNGSIM_ENABLE_MIR=1 uv run --no-sync --directory bngsim python scripts/rebuild_editable.py
```

A successful configure prints `Building vendored MIR micro-JIT from:
third_party/mir/`. Without the define the C++ side compiles `mir_jit.hpp` to an
inert stub and the codegen dispatch never takes the JIT branch.

Then select the JIT backend at runtime with `BNGSIM_CODEGEN_JIT=mir` and run the
codegen-sensitive tests, plus the vendoring guardrail:

```sh
uv run --directory bngsim python -m pytest python/tests/test_mir_vendoring.py

BNGSIM_CODEGEN_JIT=mir uv run --directory bngsim python -m pytest \
  python/tests/test_codegen.py \
  python/tests/test_codegen_jacobian.py
```

The codegen tests must agree numerically with the default `cc` backend
(`BNGSIM_CODEGEN_JIT` unset). If editable Python results disagree with the current
C++/MIR source tree, rebuild before trusting the failure.

## Pruning Rules

The prune is a **denylist** encoded in `vendor_mir.py` (`is_vendored_path`) and
mirrored in `VENDOR.json` → `pruning`. Every upstream file is vendored except:

- whole subtrees: `.github`, `adt-tests`, `c-benchmarks`, `c-tests`, `llvm2mir`,
  `mir2c`, `mir-tests`, `mir-utils`
- standalone CLI/test drivers: `sieve.c`, `mir-bin-driver.c`, `mir-bin-run.c`,
  `mir-gen-stub.c`, `c2mir/c2mir-driver.c`
- build files (`CMakeLists.txt`, `GNUmakefile`), dotfiles (`.clang-format`,
  `.gitignore`), CI/shell scripts (`*.yml`, `*.yaml`, `*.sh`), images (`*.svg`),
  and docs (`*.md`) other than `README.md`

**All target backends are kept** (`mir-x86_64.*`, `mir-aarch64.*`, `mir-ppc64.*`,
`mir-riscv64.*`, `mir-s390x.*`, and the matching `mir-gen-<arch>.c` and
`c2mir/<arch>/`), so the same tree builds on every host arch without per-arch
pruning. The denylist is validated to reproduce the checked-in tree exactly: a
no-op re-vendor of the pinned commit changes no source file (only the dynamic
`VENDOR.json` fields) other than those with a local carry, which it reverts to
upstream's and then exits nonzero (see "Local Carries").

Change the prune in `vendor_mir.py` first if the vendor surface ever changes.

## Upgrade Guardrails

These behaviors are required by `mir_jit.hpp` and recorded under
`VENDOR.json` → `guardrails`; re-verify them on any pin bump:

- **Header-free JIT source.** c2mir's bundled preprocessor cannot parse the
  platform SDK `<math.h>` / `<stdlib.h>` (compiler-specific extensions; on macOS
  it reports *"Unsupported compiler detected"*). Because BNGsim *generates* the C
  it JITs, the JIT path strips system `#include`s and prepends `extern`
  declarations for the libc/libm symbols the RHS can call (`pow`, `exp`, `sqrt`,
  `log`, `fabs`, `fmax`, `fmin`, `memset`, …). MIR resolves those externals at
  link time via the import resolver (`dlsym(RTLD_DEFAULT, name)`).
- **Imports are limited to the prelude.** c2mir accepts a call to an undeclared
  function as an implicit `int name()` where `cc` refuses it, so a bare call the
  emitter has no C for reaches `MIR_link` as an import. MIR reports an unresolved
  import by `exit()`ing the process, and `dlsym` binds any name a loaded library
  happens to export (on macOS a private framework loaded into the Python process
  has an `equal`). `MirJit` refuses every import outside the prelude's table
  before linking; re-check on a pin bump that `MIR_link` still resolves imports
  only through the resolver.
- **Find the JIT'd function by iteration.** `MIR_get_global_item` is declared in
  `mir.h` but not defined at the pinned commit; `MirJit` locates the function by
  scanning `module->items` for the matching `MIR_func_item` (the version-stable
  approach `c2mir-driver` uses). If a refresh starts defining
  `MIR_get_global_item`, the scan still works — but verify before relying on it.
- **Single-maintainer upstream.** MIR is effectively single-maintainer and pinned
  to an immutable commit. Refresh deliberately.
- **Local carries: three** (see "Current carries" below): two for issue
  #1215, in `mir.c` and `c2mir/c2mir.c`, and one for issue #1216,
  `mir-gen-x86_64.c`'s 3- and 4-byte loop-padding NOPs. `VENDOR.json` →
  `local_carries` records each deliberate BNGsim edit, and both
  `vendor_mir.py --check` and `test_mir_vendoring.py` verify every recorded
  carry is *still applied* — see "Local Carries" below.

## Local Carries

A carry is a deliberate BNGsim edit to upstream MIR source. Prefer landing a fix
upstream; carry one only when waiting is not an option.

A carry cannot be policed by the SHA256 anchors. A refresh rewrites the file from
upstream **and re-anchors its checksum**, so a dropped carry leaves a tree that
matches its own metadata perfectly. Each entry therefore names a `marker` — a
comment string the patch itself carries — and `--check` greps the file for it.
That is the one tripwire a re-vendor cannot satisfy by accident, and it is why
`do_write` exits nonzero (after writing) when a refresh loses a carry.

Each entry records:

| field | meaning |
|---|---|
| `file` | vendored path the carry edits |
| `marker` | comment string the patch carries; `--check` greps for it |
| `issue` | the BNGsim issue that motivated it |
| `upstream_status` | reported / merged / absent, and as of when |
| `summary` | what the upstream code does wrong and what the carry does instead |
| `reapply` | what to do when a refresh drops it |

Those six are required — `--check` reports an entry missing any of them. An entry
may carry more; the #413 carry added `function` and `upstream_file_sha256` (the
hash of the file *before* the carry, so a refresh can tell "upstream unchanged,
re-apply verbatim" from "upstream rewrote this, read it first"). Note that
`files{}` is rebuilt wholesale on every refresh, so a per-file extra key there
would not survive — carry-adjacent facts belong in the carry entry.

### Current carries

Two are issue #1215, which made running out of memory or address space a
`RuntimeError` from `MirJit` instead of the end of the process, and both are
marked `BNGsim local carry (lanl/bngsim#1215)`. Both were reported upstream on
2026-10-29, the `mir.c` one as
[vnmakarov/mir#493](https://github.com/vnmakarov/mir/issues/493) and the
`c2mir.c` one as [vnmakarov/mir#494](https://github.com/vnmakarov/mir/issues/494).
Both are open as of 2026-10-30, and upstream master `dc2c763f` (2026-10-23)
still has the code they replace.

- **`mir.c` — `_MIR_get_default_code_alloc`.** MirJit hands each context a code
  allocator that wraps MIR's default one, so that a failed mapping is reported
  instead of coming back as a NULL code address MIR writes through. The default
  is static in `mir.c`; the carry adds a one-line accessor for it, so MirJit
  wraps it rather than copying its platform code. A refresh that drops it fails
  to link.
- **`c2mir/c2mir.c` — `new_macro`, `new_macro_call`, `new_ifstate`.** c2mir's
  preprocessor allocates these records with plain `malloc` and writes through
  the result unchecked, out of reach of the allocator MirJit hands MIR. The
  carry checks each and reports a null through MIR's error function as an
  allocation error, as `c2mir_calloc` already does. `test_mir_vendoring.py`
  checks every plain `malloc` of a struct in `c2mir.c`, so a refresh that adds
  another fails it too.

The third is issue #1216's, marked `BNGsim local carry (lanl/bngsim#1216)`.

- **`mir-gen-x86_64.c` — the 3- and 4-byte loop-padding NOPs (issue #1216),
  carried from 2026-10-27.** From -O2, `target_translate` aligns the head of
  each innermost loop to an 8-byte boundary with 0 to 7 bytes of NOPs from
  `nop_pats`. Upstream's 3- and 4-byte entries are the register forms
  `0F 1F C0` (`nopl %eax`) and `48 0F 1F C0` (`nopq %rax`). An x86-64 CPU runs
  them, but Rosetta 2 raises SIGILL on both, so an x86_64 Python on Apple
  Silicon died the first time it ran a 3- or 4-byte pad (one reached by falling
  into the loop; a jump to the label skips it). -O0 and -O1 pad nothing, which
  is why only -O2 failed. The carry uses the memory forms Intel recommends,
  `0F 1F 00` and `0F 1F 40 00`, as upstream's 5- to 7-byte entries already are.
  They are the same length, so no offset moves, and a NOP reads no memory
  whatever its operand. It is carried rather than waited for because it changes
  nothing a CPU does, and without it a MIR build is unusable at -O2 in an x86_64
  Python on Apple Silicon. Reported upstream on 2026-10-30 as a pull request
  with the same two entries,
  [vnmakarov/mir#497](https://github.com/vnmakarov/mir/pull/497), open as of
  2026-10-30; upstream master (`dc2c763f`) still has the register forms.
  `test_mir_vendoring.py::test_mir_x86_64_loop_padding_is_intels_recommended_nops`
  holds the table to Intel's list, and stays when upstream takes the fix.

### Past carries

- **`mir-gen.c` — `try_spilled_reg_mem` bounds check (issue #413), carried
  2026-08-18 to 2026-10-22.** MIR_gen's register allocator rewrites every operand
  naming one spilled register into a memory operand and records the rewritten
  indices in `op_nums[MAX_INSN_RELOAD_MEM_OPS]`. Upstream sized that table at 2
  and bounded it with `gen_assert` alone, which `-DNDEBUG` deletes from every
  release build; `x = x * x` reaches the allocator as the three-operand
  `dmul r, r, r`, so the third store landed past the array (glibc:
  `*** stack smashing detected ***`). The carry raised the table to 3 and added a
  runtime check that undoes the rewrites and declines. Upstream merged the same
  code as [vnmakarov/mir#468](https://github.com/vnmakarov/mir/pull/468), and the
  carry was dropped when the pin moved to `439a90a2`, which includes it.
  `test_mir_vendoring.py::test_mir_gen_spill_reload_table_is_bounds_checked`
  now guards upstream's copy against a regression, and
  `python/tests/test_codegen_jit_self_multiply.py` is still the end-to-end test.
  The minimal reproducer is two lines of C, JIT'd at -O2: a call, then
  `o[0] = t*t` (the call spills `t`; the square makes all three operands of one
  insn name it).

## Build Wiring (for reference)

`bngsim/CMakeLists.txt` compiles three translation units into a static
`bngsim_mir` library when `BNGSIM_ENABLE_MIR` is ON:

- `mir.c` — MIR core (loader, linker, interpreter; `#include`s the host target
  backend, e.g. `mir-x86_64.c` / `mir-aarch64.c`, under a target-macro guard)
- `mir-gen.c` — the optimizing JIT code generator
- `c2mir/c2mir.c` — the C11 → MIR frontend

`bngsim_mir` is linked `PRIVATE` into `bngsim` (and `_bngsim_core`) with
`BNGSIM_HAS_MIR=1`, so MIR's headers never leak into BNGsim's public ABI. The
include dirs are this directory and `c2mir/`. `C2MIR_PARALLEL` is intentionally
left undefined (single-threaded gen, no pthread dependency).

## Notes

- `bngsim/third_party/mir` stores vendored source, not metadata — refreshing it
  rewrites C/H files, not just a pin.
- The first `vendor_mir.py` refresh of the hand-authored `VENDOR.json` normalizes
  its formatting (standard 2-space `json.dumps`); the per-file checksums and
  content are unchanged.
