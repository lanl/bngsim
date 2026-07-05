# NFsim Vendoring

## Goal

BNGsim vendors NFsim from `RuleWorld/nfsim` `master` with a small, explicit
BNGsim carry queue. `bngsim/scripts/vendor_nfsim.py` is the only supported
path that writes `bngsim/third_party/nfsim`.

This doc is BNGsim-specific. The tree under `bngsim/third_party/nfsim` is
generated source. If a change needs to survive the next vendor refresh, land
it in the NFsim checkout first and then re-run `vendor_nfsim.py`.

## Current Baseline

- `bngsim/third_party/nfsim/VENDOR.json` currently records source ref
  `bngsim/vendor` at commit `5d207e05255f66b5d9eb167350049158b7ff260a`.
- The two ExprTk shim carries (reserved-symbol remap + mratio) were rewritten
  to **forward** to the host single source `bngsim::expr_compat` rather than
  carry a hand-ported copy of the logic in `bngsim/src/expression.cpp`
  (issue #49). The shim is now a thin mu::Parser adapter; the vendored NFsim
  target links `bngsim::expression` so the shim's `#include
  <bngsim/expr_compat.hpp>` resolves. The remaining carries are required local
  carries (selector/exception/build-safety); do not treat them as a new
  upstream retarget.
- The current vendored source is based on upstream `origin/master` commit
  `4c3da839f6c1ecdf1a1ea32942c5de52904e0377`.
- Vendoring metadata should record canonical upstream remote
  `https://github.com/RuleWorld/nfsim.git`.
- There are no active upstream-PR carries. The earlier local carries behind
  RuleWorld/nfsim PRs `#69`, `#70`, `#72`, `#73`, `#74`, `#80`, `#81`, and
  `#82` are now in upstream `origin/master` and should stay out of the local
  carry queue.
- `bngsim/vendor-next` is retired. Do not use or recreate it.

If this doc and `VENDOR.json` disagree, trust `VENDOR.json`.

## Source Of Truth

- `origin/master`: upstream NFsim base
- `bngsim/vendor`: single active NFsim integration branch used for vendoring
- `bngsim/scripts/vendor_nfsim.py`: only supported exporter into
  `bngsim/third_party/nfsim`
- `bngsim/third_party/nfsim/VENDOR.json`: authoritative record of source
  commit, upstream base, and live carry queue
- `bngsim/scripts/nfsim_vendor_patches/`: restartable export of
  `origin/master..bngsim/vendor`

Historical branches such as `bngsim/connected-update`,
`bngsim/vendor-build`, `bngsim/tfun`, `bngsim/step-to-cache`,
`bngsim/windows-nauty`, `bngsim/embedding-api`, and `bngsim/library-build`
are provenance only unless you are doing archaeology.

## Live Carry Queue

The live carry queue stays explicit in both
`bngsim/scripts/vendor_nfsim.py` (`CARRY_QUEUE`) and
`bngsim/third_party/nfsim/VENDOR.json` (`carry_queue`).

Current carry topics:

- NFsim v1.14.3 selector compatibility toggle
- molecule-limit exception instead of process termination
- TFUN comment-line handling
- macOS subproject-architecture guard for embedded builds
- `exit(1)` → `throw std::runtime_error` conversion across six NFsim files
  (Session 29 restoration; was silently reverted by the May-15 refresh
  because it had bypassed this carry queue)
- product-molecularity check for multi-bond ring-opening
  (internal#57; candidate to push upstream)
- Species-observable complex tracking regardless of `-bscb`
  (internal#57; candidate to push upstream)
- ExprTk reserved-symbol remap in NFsim's mu::Parser shim — forwards to the
  host single source `bngsim::expr_compat` (issue #49)
- `mratio()` built-in in NFsim's mu::Parser shim — forwards to
  `bngsim::expr_compat::mratio` (issue #49)
- reactant-count composite scope-free evaluation guard
  (restored from PyBNF-Private commit `10d511ef`)
- hard-stop restoration for `include_products()` / `exclude_products()`
- NFtest-header pruning from the vendored `NFsim.hh` facade
- CLI-source trimming for library-only embedded NFsim builds
- `stepTo()` cache compatibility hook restoration

Dropped carries that are now upstream include connectivity determinism,
observable-output accessors, NFsim's library-build/export toggles, the old
`stepTo()` cache carries, the ExprTk snapshot refresh, and the recent
`resetConcentrations()` / selector-teardown fixes.

Keep this queue small, reviewable, and easy to drop. When upstream absorbs a
carry or BNGsim no longer depends on it, remove it from the queue and rebuild
`bngsim/vendor` from a clean `origin/master` base.

Direct edits to `bngsim/third_party/nfsim/` are not supported and have caused
two production regressions (Session 29's exit→throw work and the stepTo cache
fix were both lost). If a change needs to survive, land it in the NFsim
checkout on the `bngsim/vendor` branch first, regenerate the patch series
under `bngsim/scripts/nfsim_vendor_patches/`, and re-run `vendor_nfsim.py`.

## Safe Workspace Policy

- Prefer `/tmp/nfsim-vendor-candidate` for vendoring work.
- Prefer cloning directly from `https://github.com/RuleWorld/nfsim.git`.
- If you bootstrap the candidate from another local checkout for convenience or
  offline work, treat that checkout as transport only.
- Immediately reset the candidate checkout's `origin` remote to
  `https://github.com/RuleWorld/nfsim.git` before running
  `vendor_nfsim.py`. That keeps `VENDOR.json` metadata canonical.
- Do not treat a found build tree as proof. Validate through the repo-supported
  build/test flows below.

## Rebuild The Candidate Checkout

Fresh clone from the canonical upstream remote:

```sh
REPO_ROOT=/path/to/repo   # the checkout root that contains bngsim/
git clone https://github.com/RuleWorld/nfsim.git /tmp/nfsim-vendor-candidate
git -C /tmp/nfsim-vendor-candidate fetch origin --prune
git -C /tmp/nfsim-vendor-candidate checkout -B bngsim/vendor origin/master
git -C /tmp/nfsim-vendor-candidate am \
  "$REPO_ROOT"/bngsim/scripts/nfsim_vendor_patches/*.patch
git -C /tmp/nfsim-vendor-candidate status --short --branch
```

Optional bootstrap from a local cache checkout:

```sh
REPO_ROOT=/path/to/repo   # the checkout root that contains bngsim/
git clone /path/to/nfsim-cache /tmp/nfsim-vendor-candidate
git -C /tmp/nfsim-vendor-candidate remote set-url origin https://github.com/RuleWorld/nfsim.git
git -C /tmp/nfsim-vendor-candidate fetch origin --prune
git -C /tmp/nfsim-vendor-candidate checkout -B bngsim/vendor origin/master
git -C /tmp/nfsim-vendor-candidate am \
  "$REPO_ROOT"/bngsim/scripts/nfsim_vendor_patches/*.patch
git -C /tmp/nfsim-vendor-candidate status --short --branch
```

The local clone may inherit a different default branch from the source repo.
The `checkout -B ... origin/master` step above resets it to the intended clean
base.

## Standard Refresh Workflow

### 1. Rebuild Or Update `bngsim/vendor`

For a clean restart, prefer rebuilding the branch from `origin/master` and
replaying the explicit carries instead of stacking more ad hoc commits on top
of an old vendor branch:

```sh
REPO_ROOT=/path/to/repo   # the checkout root that contains bngsim/
git -C /tmp/nfsim-vendor-candidate fetch origin --prune
git -C /tmp/nfsim-vendor-candidate checkout -B bngsim/vendor origin/master
git -C /tmp/nfsim-vendor-candidate am \
  "$REPO_ROOT"/bngsim/scripts/nfsim_vendor_patches/*.patch
```

Expected carry order:

1. `NFsimV1143Compatibility` selector toggle
2. exception-safety / molecule-limit throw
3. TFUN comment handling
4. macOS subproject-architecture guard
5. `exit(1)` → `throw std::runtime_error` conversions
6. product-molecularity fix for multi-bond ring-opening
7. Species-observable complex tracking regardless of `-bscb`
8. ExprTk reserved-symbol remap in NFsim's mu::Parser shim
9. `mratio()` built-in in NFsim's mu::Parser shim
10. reactant-count composite scope-free guard
11. `include_products()` / `exclude_products()` hard stop
12. NFtest-header pruning from the vendored facade
13. CLI-source trimming for library-only embedded builds
14. `stepTo()` cache compatibility hook restoration

When you intentionally change the carry stack, regenerate the patch bundle
afterward.

### 2. Preview Before Writing

From the repo root:

```sh
python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --ref bngsim/vendor \
  --summary --compare-ref origin/master

python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --ref bngsim/vendor \
  --check
```

The goal is a small, understandable delta versus `origin/master`. If the delta
is unexpectedly large, stop and inspect the carry queue before rewriting the
vendored tree.

### 3. Refresh `bngsim/third_party/nfsim`

```sh
python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --ref bngsim/vendor
```

This updates:

- `bngsim/third_party/nfsim`
- `bngsim/third_party/nfsim/VENDOR.json`

### 4. Validate Inside BNGsim

Use the repo-supported flows and re-run the NFsim-specific surface:

```sh
ctest --output-on-failure -R 'nfsim_library_test|nfsim_simulator_test'
uv run --directory bngsim python -m pytest \
  python/tests/test_nfsim.py \
  python/tests/test_nfsim_session.py \
  python/tests/test_nfsim_tech_debt.py \
  python/tests/test_timeout.py
uv run python bngsim/scripts/local_ci.py wheel --python 3.12
```

Prefer `ninja`-backed builds when the local generator supports it, but treat
the test commands above as the real evidence that vendoring still works.

Editable-build reminder:

- `uv run --directory bngsim python -m pytest ...` uses the installed editable
  extension, and editable imports do not auto-rebuild.
- If Python results disagree with the current C++ source tree, compare the
  `_bngsim_core` binary timestamp to the edited source files and rebuild the
  runtime before trusting the failure.
- On this macOS setup, `uv run python bngsim/scripts/rebuild_editable.py` can
  still fail for unrelated mixed-architecture linker reasons. When that
  happens, validate the current runtime through a fresh wheel build instead of
  trusting a stale editable `.so`.

## Restart Artifacts

The restart bundle lives in `bngsim/dev/notes/`:

- `bngsim/dev/notes/NFSIM_VENDORING_HANDOFF.md`
- `bngsim/scripts/nfsim_vendor_patches/`

After any intentional change to `bngsim/vendor`, regenerate the patch export:

```sh
REPO_ROOT=/path/to/repo   # the checkout root that contains bngsim/
rm -f "$REPO_ROOT"/bngsim/scripts/nfsim_vendor_patches/*.patch
git -C /tmp/nfsim-vendor-candidate format-patch \
  -o "$REPO_ROOT"/bngsim/scripts/nfsim_vendor_patches \
  origin/master..bngsim/vendor
```

Then refresh the handoff note so the recorded tip, base, and validation
commands still match reality.

## Vendoring Commands

Refresh from the current candidate checkout:

```sh
python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --ref bngsim/vendor
```

Verify the vendored files without writing:

```sh
python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --ref bngsim/vendor \
  --check
```

## Tripwire (`--verify-clean`)

`vendor_nfsim.py --verify-clean` is the CI-side tripwire that catches
direct edits to `bngsim/third_party/nfsim/`. It performs two checks:

1. **Byte-match.** Exports the tree the candidate's `bngsim/vendor` tip
   would produce and compares it against `bngsim/third_party/nfsim/`.
   Any divergence is reported as `DIRECT EDIT DETECTED` with the list of
   changed files and the standard remediation message.
2. **Carry-queue ↔ patch series cross-check.** Confirms that the number
   of `*.patch` files under `bngsim/scripts/nfsim_vendor_patches/`
   matches `len(commit for topic in CARRY_QUEUE for commit in topic)`,
   and that every commit summary listed in `CARRY_QUEUE` matches some
   patch's `Subject:` line. Catches "added a topic to `CARRY_QUEUE`
   without re-exporting the patch series" and similar drift.

```sh
python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --verify-clean
```

Add `--strict` to additionally rebuild the candidate from scratch
(`origin/master` + `git am` of the patch series in a temp checkout) and
confirm the result matches the live candidate's `bngsim/vendor` tip —
i.e., that the patch series is reproducible end-to-end.

```sh
python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --verify-clean --strict
```

The tripwire is also wired into the local-CI pre-commit hooks:

```sh
pre-commit run --hook-stage manual local-ci-vendor-verify
pre-commit run --hook-stage manual local-ci-vendor-verify-strict
```

A paused GitHub Actions job (`vendor_verify` in `.github/workflows/build.yml`)
will run the same check automatically when the Actions quota resets in
June 2026.

When the tripwire fires, do not bypass it. Drift means a direct edit
slipped through; follow the remediation message — land the change on
`bngsim/vendor`, re-export the patch series, update `CARRY_QUEUE`,
re-run `vendor_nfsim.py`.

Preview the impact without writing:

```sh
python3 bngsim/scripts/vendor_nfsim.py \
  --nfsim-repo /tmp/nfsim-vendor-candidate \
  --ref bngsim/vendor \
  --summary --compare-ref origin/master
```

The script records source metadata in `bngsim/third_party/nfsim/VENDOR.json`.
The no-argument form targets `/tmp/nfsim-vendor-candidate` plus
`bngsim/vendor`, but the explicit `--nfsim-repo` / `--ref` form is clearer in
shared notes and transcripts.

If your NFsim checkout lives somewhere else, point the script at it:

```sh
python3 bngsim/scripts/vendor_nfsim.py --nfsim-repo /path/to/nfsim
```

If you need to bypass automatic ref resolution, an explicit fully-qualified ref
also works:

```sh
python3 bngsim/scripts/vendor_nfsim.py --ref refs/heads/bngsim/vendor
```

## Pruning Rules

The vendored import includes:

- `CMakeLists.txt`
- `LICENSE.txt`
- `README.md`
- `src/`

The vendored import excludes:

- `src/NFtest`
- `src/NFfunction/muParser`
- `src/NFfunction/exprtk` (GH #126 — see "NFsim's ExprTk Copy" below)
- `src/NFutil/MTrand/mttest.cpp`

These rules are encoded in `bngsim/scripts/vendor_nfsim.py` and should be
changed there first if the vendor surface changes.

The vendored `bngsim/third_party/nfsim/README.md` is copied from the NFsim
source branch during refresh. If you want that file's maintainer guidance to
survive the next vendor sync, update it in the NFsim source checkout first and
then re-run `vendor_nfsim.py`.

## NFsim's ExprTk Copy

- Upstream NFsim vendors its own `src/NFfunction/exprtk/exprtk.hpp`, but BNGsim
  **prunes it from the vendor export** (`PRUNED_PATHS` /
  `VENDOR.json` `excluded_paths`) — GH #126, the file-level half of the GH #49
  ExprTk convergence (ADR-005 collapsed the duplicated *logic*; #126 collapses
  the duplicated *file*).
- The NFsim build is configured with `NFSIM_USE_EXPRTK=ON`, so the `mu::Parser`
  shim (`src/NFfunction/nfsim_funcparser.h`) still does `#include "exprtk.hpp"`.
  After the prune that include resolves against the **single host snapshot**
  `bngsim/third_party/exprtk/exprtk.hpp`: the parent `CMakeLists.txt` adds
  `${BNGSIM_EXPRTK_DIR}` to the `NFsim_objects` / `nfsim` target include paths
  right where it already links `bngsim::expression` into them (issue #49 wiring).
  No vendor carry is involved — the redirect is host build configuration, the
  same place that already forces `NFSIM_USE_EXPRTK` and the library/executable
  toggles.
- The vendored `third_party/nfsim/CMakeLists.txt` still appends its (now-absent)
  `src/NFfunction/exprtk` to the include path (the `NFSIM_USE_EXPRTK` branch at
  lines ~69 / ~143). That is a harmless dead `-I` the compiler ignores — exactly
  the long-standing pattern for the already-pruned `src/NFfunction/muParser`
  dir — so it is left untouched rather than carried.
- There is now **one** `exprtk.hpp` per binary. Refreshing
  `bngsim/third_party/exprtk` (via `vendor_exprtk.py`) is the single place an
  ExprTk snapshot bump happens; the ODE/SSA engine, RuleMonkey, and the NFsim
  shim all compile against it. The former byte-drift guard
  `test_exprtk_reserved_consistency.py` is retired — there is no second copy to
  drift against.
