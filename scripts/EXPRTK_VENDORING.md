# ExprTk Vendoring

## Goal

BNGsim vendors ExprTk as a stock upstream `exprtk.hpp` with no hidden local
carry queue. `bngsim/scripts/vendor_exprtk.py` is the only supported path that
writes `bngsim/third_party/exprtk`.

This doc is BNGsim-specific. The tree under `bngsim/third_party/exprtk` is
generated vendored source. If a change needs to survive the next refresh,
prefer the wrapper layer first (`bngsim/src/expression.cpp`,
`bngsim/include/bngsim/expression.hpp`) and only carry a header patch if the
carry is explicitly documented in `VENDOR.json`.

## Current Baseline

- `bngsim/third_party/exprtk/VENDOR.json` records the pinned upstream commit,
  canonical remote, header checksum, and reserved-name guardrails.
- The preferred disposable checkout is `/tmp/exprtk-vendor-candidate`.
- If this doc and `VENDOR.json` disagree, trust `VENDOR.json`.

## Source Of Truth

- Official project homepage and download: `https://www.partow.net/programming/exprtk/index.html`
- Official source download artifact: `https://www.partow.net/downloads/exprtk.zip`
- BNGsim's authoritative vendoring source: `https://github.com/ArashPartow/exprtk.git`
- Authoritative branch: `master`
- `bngsim/scripts/vendor_exprtk.py`: only supported exporter into
  `bngsim/third_party/exprtk`
- `bngsim/third_party/exprtk/VENDOR.json`: authoritative record of the
  vendored commit, header checksum, and wrapper-sensitive guardrails

BNGsim intentionally pins the author-maintained GitHub mirror rather than a
moving raw download because vendoring must be restartable, reviewable, and
machine-independent. The Partow site remains the upstream project homepage and
official download location; the Git checkout is the canonical vendoring
transport for BNGsim.

## Safe Workspace Policy

- Prefer `/tmp/exprtk-vendor-candidate` for vendoring work.
- Prefer cloning directly from `https://github.com/ArashPartow/exprtk.git`.
- If you bootstrap the candidate from another local checkout, treat that local
  checkout as transport only.
- Immediately reset the candidate checkout's `origin` remote to
  `https://github.com/ArashPartow/exprtk.git`, then `fetch --prune`, before
  running `vendor_exprtk.py`.
- Do not vendor from a dirty ExprTk checkout.
- Do not treat a found build tree as proof. Validate through the repo-supported
  flows below.

## Rebuild The Candidate Checkout

Fresh clone from the canonical upstream mirror:

```sh
git clone --branch master --single-branch https://github.com/ArashPartow/exprtk.git /tmp/exprtk-vendor-candidate
git -C /tmp/exprtk-vendor-candidate remote -v
git -C /tmp/exprtk-vendor-candidate status --short --branch
git -C /tmp/exprtk-vendor-candidate rev-parse HEAD
```

Optional bootstrap from a local cache checkout:

```sh
git clone /path/to/exprtk-cache /tmp/exprtk-vendor-candidate
git -C /tmp/exprtk-vendor-candidate remote set-url origin https://github.com/ArashPartow/exprtk.git
git -C /tmp/exprtk-vendor-candidate fetch origin --prune
git -C /tmp/exprtk-vendor-candidate checkout -B master origin/master
git -C /tmp/exprtk-vendor-candidate status --short --branch
```

Refresh an existing candidate checkout safely:

```sh
git -C /tmp/exprtk-vendor-candidate fetch origin --prune
git -C /tmp/exprtk-vendor-candidate checkout master
git -C /tmp/exprtk-vendor-candidate pull --ff-only origin master
git -C /tmp/exprtk-vendor-candidate status --short --branch
```

## Standard Refresh Workflow

### 1. Preview Before Writing

From the PyBNF repo root:

```sh
python3 bngsim/scripts/vendor_exprtk.py \
  --exprtk-repo /tmp/exprtk-vendor-candidate \
  --ref master \
  --summary

python3 bngsim/scripts/vendor_exprtk.py \
  --exprtk-repo /tmp/exprtk-vendor-candidate \
  --ref master \
  --check
```

The summary output verifies:

- canonical remote / candidate HEAD
- resolved source ref and commit
- current vendored baseline from `VENDOR.json`
- header checksum and delta versus `bngsim/third_party/exprtk/exprtk.hpp`
- the upstream reserved-name lists BNGsim reads directly for mangling
- wrapper-critical header tokens such as `reserved_words[]`,
  `reserved_symbols[]`, `ifunction`, `allow_zero_parameters()`, and
  `set_max_stack_depth`

If the summary shows a dirty checkout, non-canonical remote, missing tokens, or
unexpected local carries, stop and fix the upstream candidate / guardrails
before writing.

### 2. Refresh `bngsim/third_party/exprtk`

```sh
python3 bngsim/scripts/vendor_exprtk.py \
  --exprtk-repo /tmp/exprtk-vendor-candidate \
  --ref master
```

This updates:

- `bngsim/third_party/exprtk/exprtk.hpp`
- `bngsim/third_party/exprtk/VENDOR.json`

### 3. Validate Inside BNGsim

Use the repo-supported flows:

```sh
uv run --directory bngsim python -m pytest \
  python/tests/test_exprtk_reserved_words.py \
  python/tests/test_exprtk_vendoring.py \
  python/tests/test_sign_as_parameter.py \
  python/tests/test_obs_zero_arg_call.py \
  python/tests/test_model_clone.py \
  python/tests/test_sbml.py

uv run python bngsim/scripts/local_ci.py wheel --python 3.12
```

Editable-build reminder:

- `uv run --directory bngsim python -m pytest ...` uses the editable
  environment.
- If Python results disagree with the current C++ source tree, rebuild or
  re-install before trusting the failure.
- Prefer the wheel smoke above as the packaging truth because it rebuilds the
  vendored extension from the current source tree.

## Wrapper Ownership

BNGsim-specific compatibility behavior belongs in the wrapper layer, not in the
vendored header:

- compile-time policy macros defined before including `exprtk.hpp`:
  `exprtk_disable_string_capabilities`,
  `exprtk_disable_rtl_io_file`,
  `exprtk_disable_rtl_vecops`,
  `exprtk_disable_caseinsensitivity`
- BNG-compatible aliases: `ln`, `rint`, `sign`, `mratio`, `time()`
- BNG identifier adaptation: underscore-prefixed names, reserved-name mangling,
  and zero-arg observable-call normalization
- clone-time parser reuse and wrapper-specific expression preprocessing

Current local carries inside `exprtk.hpp`: none

## Upgrade Guardrails

- `bngsim/src/expression.cpp` reads ExprTk's `reserved_words[]` and
  `reserved_symbols[]` directly from the vendored header, so upstream changes
  cannot silently drift away from BNGsim's mangling logic.
- `vendor_exprtk.py` fails if the header stops exposing the wrapper-critical
  tokens BNGsim depends on.
- `VENDOR.json` stores the pinned header checksum and machine-readable copies of
  the upstream reserved-name lists.
- `python/tests/test_exprtk_vendoring.py` re-parses the vendored header and
  verifies that `VENDOR.json` still matches the checked-in file.
- The targeted ExprTk-sensitive pytest set above is the behavioral evidence
  that the wrapper still matches the vendored header.

## Historical Note

Before this workflow was hardened, BNGsim carried an undocumented `exprtk.hpp`
variant plus a CMake fallback that downloaded a moving `master` raw header into
the source tree. The current workflow removes that implicit mutation path and
tracks a single pinned upstream commit instead.
