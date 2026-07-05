# SUNDIALS Vendoring

## Goal

BNGsim does not check SUNDIALS source into the repository. Instead, managed
builds fetch a pinned official SUNDIALS release archive described in
`bngsim/third_party/sundials/VENDOR.json`.

`bngsim/scripts/vendor_sundials.py` is the supported path for refreshing or
checking that metadata. `bngsim/CMakeLists.txt` consumes the pinned URL and
SHA256 directly from `VENDOR.json`.

If this doc and `VENDOR.json` disagree, trust `VENDOR.json`.

## Source Of Truth

- Canonical upstream repository: `https://github.com/LLNL/sundials.git`
- Canonical BNGsim fetch artifact: the official GitHub release asset
  `sundials-<version>.tar.gz`
- Current pinned release tag: `v7.2.1`
- Current pinned release asset:
  `https://github.com/LLNL/sundials/releases/download/v7.2.1/sundials-7.2.1.tar.gz`
- `bngsim/third_party/sundials/VENDOR.json`: authoritative record of the
  release tag, peeled tag commit, release URL, and archive SHA256
- `bngsim/scripts/vendor_sundials.py`: supported metadata refresh/check tool

BNGsim intentionally pins the published release tarball instead of a Git tag
inside CMake so managed builds stay reproducible even if a tag is moved,
rewritten, or rate-limited differently across machines.

## Managed Vs System Mode

- Managed mode (default): BNGsim fetches the pinned release archive from
  `VENDOR.json` through CMake `FetchContent`.
- System mode: `-DBNGSIM_USE_SYSTEM_SUNDIALS=ON` skips the pinned fetch path
  and uses `find_package(SUNDIALS ...)`.

System mode is intentionally environment-managed rather than repo-pinned. Use
it for distribution packaging or environments that already provide SUNDIALS.

## Safe Workspace Policy

- Prefer `/tmp/sundials-vendor-candidate`.
- Treat the candidate workspace as disposable.
- Verify the release tarball fingerprint before updating `VENDOR.json`.
- Do not treat a previous build tree as proof. Use the repo-supported
  validation flow after any intentional change.

## Rebuild The Candidate Workspace

Prepare the release tarball in the disposable `/tmp` workspace:

```sh
mkdir -p /tmp/sundials-vendor-candidate
gh release download v7.2.1 \
  --repo LLNL/sundials \
  --pattern sundials-7.2.1.tar.gz \
  --dir /tmp/sundials-vendor-candidate
shasum -a 256 /tmp/sundials-vendor-candidate/sundials-7.2.1.tar.gz
tar -xzf /tmp/sundials-vendor-candidate/sundials-7.2.1.tar.gz -C /tmp/sundials-vendor-candidate
```

Verify the tag object and peeled commit separately:

```sh
gh release view v7.2.1 \
  --repo LLNL/sundials \
  --json tagName,name,publishedAt,targetCommitish,assets

git ls-remote https://github.com/LLNL/sundials.git \
  refs/tags/v7.2.1 refs/tags/v7.2.1^{}
```

For the current pinned baseline the expected peeled commit is:

```text
2dcb3e018b4c4cfe824bff09eb52184ed083e368
```

## Standard Refresh Workflow

### 1. Preview Before Writing

From the PyBNF repo root:

```sh
python3 bngsim/scripts/vendor_sundials.py \
  --archive /tmp/sundials-vendor-candidate/sundials-7.2.1.tar.gz \
  --tag v7.2.1 \
  --tag-object 35a984a216f6ea1db0315a506114673d90994407 \
  --tag-commit 2dcb3e018b4c4cfe824bff09eb52184ed083e368 \
  --summary

python3 bngsim/scripts/vendor_sundials.py \
  --archive /tmp/sundials-vendor-candidate/sundials-7.2.1.tar.gz \
  --tag v7.2.1 \
  --tag-object 35a984a216f6ea1db0315a506114673d90994407 \
  --tag-commit 2dcb3e018b4c4cfe824bff09eb52184ed083e368 \
  --check
```

The summary/check path verifies:

- release tag, tag object, and peeled tag commit
- archive SHA256 and size
- archive root directory and detected SUNDIALS version
- presence of the SUNDIALS component paths BNGsim depends on
- the build options and alias-target hooks BNGsim expects to keep using:
  `BUILD_CVODES`, `BUILD_KINSOL`, `ENABLE_KLU`,
  `SUNDIALS_BUILD_WITH_MONITORING`,
  `SUNDIALS_BUILD_WITH_PROFILING`, `SUNDIALS_POSIX_TIMERS`,
  `POSIX_TIMERS_NEED_POSIX_C_SOURCE`, and the `SUNDIALS::...` alias export
  path in `SundialsAddLibrary.cmake`

If the archive fingerprint, version, or guardrail tokens drift unexpectedly,
stop and inspect the upstream release before changing `VENDOR.json`.

### 2. Refresh `bngsim/third_party/sundials/VENDOR.json`

```sh
python3 bngsim/scripts/vendor_sundials.py \
  --archive /tmp/sundials-vendor-candidate/sundials-7.2.1.tar.gz \
  --tag v7.2.1 \
  --tag-object 35a984a216f6ea1db0315a506114673d90994407 \
  --tag-commit 2dcb3e018b4c4cfe824bff09eb52184ed083e368
```

This updates:

- `bngsim/third_party/sundials/VENDOR.json`

Current local carries inside the fetched SUNDIALS source: none

### 3. Validate Inside BNGsim

Use the repo-supported build flow:

```sh
uv run --directory bngsim python -m pytest python/tests/test_sundials_vendoring.py
uv run python bngsim/scripts/local_ci.py wheel --python 3.12
```

The wheel build is the real evidence that BNGsim still configures and links
against the pinned SUNDIALS release correctly.

## Upgrade Guardrails

- `bngsim/CMakeLists.txt` reads the managed-fetch source URL and SHA256 from
  `bngsim/third_party/sundials/VENDOR.json`, so the build no longer relies on
  a mutable Git tag.
- `vendor_sundials.py` verifies the release archive still exposes the build
  knobs and component paths BNGsim depends on before `VENDOR.json` is updated.
- `python/tests/test_sundials_vendoring.py` checks the metadata shape and the
  CMake lockstep expectations from inside the repo.

## Notes

- `bngsim/third_party/sundials` intentionally stores metadata, not source code.
- If you change the pinned SUNDIALS release, update the tag object, peeled tag
  commit, archive hash, and validation transcript together.
