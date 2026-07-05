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
  `99c65079038f3ba9242ef646f308c266cfd7a8e5`, canonical remote
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
- Pinned commit: `99c65079038f3ba9242ef646f308c266cfd7a8e5`
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
git -C /tmp/mir-vendor-candidate checkout 99c65079038f3ba9242ef646f308c266cfd7a8e5
git -C /tmp/mir-vendor-candidate remote -v
git -C /tmp/mir-vendor-candidate rev-parse HEAD
```

To move the pin to a newer commit, check out (or fetch) the new ref deliberately
and pass it to `--ref` below; record it in `VENDOR.json` only after the
validation passes.

## Standard Refresh Workflow

### 1. Preview Before Writing

From the PyBNF repo root:

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
BNGSIM_ENABLE_MIR=1 uv run --directory bngsim python scripts/rebuild_editable.py
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
`VENDOR.json` fields).

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
- **Find the JIT'd function by iteration.** `MIR_get_global_item` is declared in
  `mir.h` but not defined at the pinned commit; `MirJit` locates the function by
  scanning `module->items` for the matching `MIR_func_item` (the version-stable
  approach `c2mir-driver` uses). If a refresh starts defining
  `MIR_get_global_item`, the scan still works — but verify before relying on it.
- **Single-maintainer upstream.** MIR is effectively single-maintainer and pinned
  to an immutable commit. Refresh deliberately.
- **Local carries: none.** The tree is stock upstream at the pinned commit.
  `vendor_mir.py --check` and `test_mir_vendoring.py` assert `local_carries == []`.
  If a change must survive a refresh, land it upstream first or record it
  explicitly in `VENDOR.json` → `local_carries`.

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
