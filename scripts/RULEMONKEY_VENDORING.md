# RuleMonkey Vendoring

## Goal

BNGsim vendors RuleMonkey from `richardposner/RuleMonkey` `main` with no
BNGsim carry queue. `bngsim/scripts/vendor_rulemonkey.py` is the only
supported path that writes `bngsim/third_party/rulemonkey`.

This doc is BNGsim-specific. The tree under `bngsim/third_party/rulemonkey`
is generated source. If a change needs to survive the next vendor refresh,
land it upstream in RuleMonkey first and then re-run `vendor_rulemonkey.py`.

## Current Baseline

- `bngsim/third_party/rulemonkey/VENDOR.json` currently records authoritative
  branch `main`, resolved source ref `origin/main`, and commit
  `13e9f636068a0f1063dfe17a397685ca17620c21`.
- Vendoring metadata records canonical upstream remote
  `https://github.com/richardposner/RuleMonkey.git`.
- The preferred disposable checkout is `/tmp/rulemonkey-vendor-candidate`.
- If this doc and `VENDOR.json` disagree, trust `VENDOR.json`.

## Source Of Truth

- Authoritative repo: `https://github.com/richardposner/RuleMonkey.git`
- Authoritative branch: `main`
- `bngsim/scripts/vendor_rulemonkey.py`: only supported exporter into
  `bngsim/third_party/rulemonkey`
- `bngsim/third_party/rulemonkey/VENDOR.json`: authoritative record of the
  vendored commit, canonical upstream, and ExprTk sync metadata
- Local checkout paths are transport only. A local cache or clone is never
  authoritative by itself.

## Safe Workspace Policy

- Prefer `/tmp/rulemonkey-vendor-candidate` for vendoring work.
- Prefer cloning directly from GitHub.
- If you bootstrap the candidate from a local cache checkout, treat that cache
  as transport only.
- Immediately reset the candidate checkout's `origin` remote to
  `https://github.com/richardposner/RuleMonkey.git`, then `fetch --prune`,
  before running `vendor_rulemonkey.py`.
- Do not vendor from a dirty RuleMonkey checkout.
- Do not treat a found build tree as proof. Validate through the repo-supported
  flows below.

## Rebuild The Candidate Checkout

Fresh clone from canonical upstream:

```sh
git clone --branch main --single-branch https://github.com/richardposner/RuleMonkey.git /tmp/rulemonkey-vendor-candidate
git -C /tmp/rulemonkey-vendor-candidate remote -v
git -C /tmp/rulemonkey-vendor-candidate status --short --branch
git -C /tmp/rulemonkey-vendor-candidate rev-parse HEAD
```

Optional bootstrap from a local cache checkout:

```sh
git clone /path/to/RuleMonkey-cache /tmp/rulemonkey-vendor-candidate
git -C /tmp/rulemonkey-vendor-candidate remote set-url origin https://github.com/richardposner/RuleMonkey.git
git -C /tmp/rulemonkey-vendor-candidate fetch origin --prune
git -C /tmp/rulemonkey-vendor-candidate checkout -B main origin/main
git -C /tmp/rulemonkey-vendor-candidate status --short --branch
```

Refresh an existing candidate checkout safely:

```sh
git -C /tmp/rulemonkey-vendor-candidate fetch origin --prune
git -C /tmp/rulemonkey-vendor-candidate checkout main
git -C /tmp/rulemonkey-vendor-candidate pull --ff-only origin main
git -C /tmp/rulemonkey-vendor-candidate status --short --branch
```

## Standard Refresh Workflow

### 1. Preview Before Writing

From the PyBNF repo root:

```sh
python3 bngsim/scripts/vendor_rulemonkey.py \
  --rulemonkey-repo /tmp/rulemonkey-vendor-candidate \
  --ref main \
  --summary

python3 bngsim/scripts/vendor_rulemonkey.py \
  --rulemonkey-repo /tmp/rulemonkey-vendor-candidate \
  --ref main \
  --check
```

The summary output verifies:

- canonical remote / candidate HEAD
- resolved source ref and commit
- current vendored baseline from `VENDOR.json`
- tree delta versus `bngsim/third_party/rulemonkey`
- git-range impact on the vendored surface
- RuleMonkey's standalone `third_party/bngsim_expr` pin still matches this
  BNGsim tree

If the summary unexpectedly shows `third_party/` export, missing
`bngsim::expression` handoff logic, or ExprTk drift, stop and fix upstream /
guardrails before writing.

### 2. Refresh `bngsim/third_party/rulemonkey`

```sh
python3 bngsim/scripts/vendor_rulemonkey.py \
  --rulemonkey-repo /tmp/rulemonkey-vendor-candidate \
  --ref main
```

This updates:

- `bngsim/third_party/rulemonkey`
- `bngsim/third_party/rulemonkey/VENDOR.json`

### 3. Validate Inside BNGsim

Use the repo-supported flows:

```sh
uv run --directory bngsim python -m pytest python/tests/test_rulemonkey.py
uv run --directory bngsim python -m pytest python/tests/test_timeout.py -k rulemonkey
uv run python bngsim/scripts/local_ci.py wheel --python 3.12
```

Editable-build reminder:

- `uv run --directory bngsim python -m pytest ...` uses the editable
  environment.
- If Python results disagree with the current C++ source tree, rebuild or
  re-install before trusting the failure.
- Prefer the wheel smoke above as the packaging truth because it exercises the
  packaged `RuleMonkeySession` flow.

## Duplication Guardrails

- `third_party/` is intentionally excluded from the vendored export.
  `vendor_rulemonkey.py` fails if the export surface ever includes it.
- `vendor_rulemonkey.py` verifies upstream RuleMonkey CMake still contains the
  `if(TARGET bngsim::expression)` host-target handoff before writing.
- `vendor_rulemonkey.py` strips RuleMonkey's unconditional
  `third_party/bngsim_expr` include from the main `rulemonkey` target when
  exporting into BNGsim, because BNGsim does not vendor RuleMonkey's
  `third_party/` tree.
- `vendor_rulemonkey.py` checks that RuleMonkey's standalone
  `third_party/bngsim_expr` / `third_party/exprtk` copies still match BNGsim's
  current `expression.hpp`, `expr_compat.hpp`, `expression.cpp`, and
  `exprtk.hpp` (the `EXPRTK_SYNC_FILES` set). `expr_compat.hpp` was added to the
  set after GH #49, where `expression.cpp` began `#include`-ing it; a drift in
  it would otherwise pass the guard while breaking RuleMonkey's standalone
  `bngsim_expr` build.
- `bngsim/CMakeLists.txt` must declare `bngsim::expression` before
  `add_subdirectory(third_party/rulemonkey)` and fails configure if vendored
  RuleMonkey stops linking that host target.

## Vendoring Commands

Refresh from the candidate checkout:

```sh
python3 bngsim/scripts/vendor_rulemonkey.py \
  --rulemonkey-repo /tmp/rulemonkey-vendor-candidate \
  --ref main
```

Verify the vendored files without writing:

```sh
python3 bngsim/scripts/vendor_rulemonkey.py \
  --rulemonkey-repo /tmp/rulemonkey-vendor-candidate \
  --ref main \
  --check
```

Preview the impact without writing:

```sh
python3 bngsim/scripts/vendor_rulemonkey.py \
  --rulemonkey-repo /tmp/rulemonkey-vendor-candidate \
  --ref main \
  --summary
```

The no-argument form assumes `/tmp/rulemonkey-vendor-candidate` at upstream
`main`:

```sh
python3 bngsim/scripts/vendor_rulemonkey.py
```

If you need to bypass automatic ref resolution, an explicit fully-qualified
ref also works:

```sh
python3 bngsim/scripts/vendor_rulemonkey.py --ref refs/remotes/origin/main
```

## Pruning Rules

The vendored import includes:

- `CMakeLists.txt`
- `LICENSE`
- `README.md`
- `CHANGELOG.md`
- `cmake/`
- `cpp/`
- `docs/`
- `include/`

The vendored import excludes:

- `cpp/cli`
- `third_party/` — deliberately, do not add it. RuleMonkey's standalone
  `third_party/exprtk/` and `third_party/bngsim_expr/` copies are for
  standalone RuleMonkey builds only. Vendoring them into BNGsim would
  recompile `bngsim::ExprTkEvaluator` and reopen duplicate-symbol / ODR risk.
  Inside a BNGsim build, RuleMonkey must reuse the host
  `bngsim::expression` target instead.

These rules are encoded in `bngsim/scripts/vendor_rulemonkey.py` and should be
changed there first if the vendor surface changes.
