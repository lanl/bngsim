# Local CI for bngsim

This is the local-validation harness we use while GitHub Actions is paused
(through 2026-06-01 — see `dev/plans/CI_LOCAL_FIX_AND_ACTIONS_PAUSE_PLAN_2026-05-12.md`).
Anyone with a target platform can run it; the recipe is identical and the
report is comparable.

## Required tools

- **uv** — manages Pythons, venvs, and dependencies on every platform.
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
- **Docker** — only for the Linux-via-Docker leg (`local_ci_linux_docker.sh`).
  - On macOS: `brew install colima docker && colima start`.

Everything else (Python 3.10–3.13, build deps, test deps) is provisioned by
uv from inside the script.

## Coverage map (current pause)

| Platform | Where | Reporter | Command |
|---|---|---|---|
| macOS x86_64 | Intel mac | the project | `uv run python scripts/local_ci.py matrix` |
| macOS arm64 | M4 Max | (Bill / you) | same command |
| Linux x86_64 | Docker on Intel mac | the project | `scripts/local_ci_linux_docker.sh` (or `MATRIX=1 ...` for cp310-cp313) |
| Windows AMD64 | efm46's box | efm46 | same command as macOS, run from PowerShell or cmd |

## Subcommands

```bash
# Build + smoke one wheel for the current uv Python.
uv run python scripts/local_ci.py wheel

# Build + smoke one wheel for a specific Python.
uv run python scripts/local_ci.py wheel --python 3.13

# Build + smoke cp310/311/312/313 in sequence.
uv run python scripts/local_ci.py matrix

# Smoke-test an already-built wheel.
uv run python scripts/local_ci.py smoke --wheel path/to/wheel.whl --python 3.12

# Linux leg: build cp312 manylinux2014 wheel + host-side smoke.
scripts/local_ci_linux_docker.sh

# Linux leg with full matrix:
MATRIX=1 scripts/local_ci_linux_docker.sh
```

The matrix subcommand on a single platform takes roughly:

| Step | Time |
|---|---|
| uv python install (first run only, all 4 versions) | ~2 min |
| SUNDIALS archive fetch + bngsim wheel build, per Python | ~5–7 min |
| Wheel install + smoke, per Python | ~30 sec |
| Total per matrix run | ~25–35 min |

The Linux Docker leg adds another ~10–15 min for the manylinux2014 image
pull (~600 MB) and pinned SUNDIALS rebuild inside the container.

## Pre-commit integration

These wrap the same commands so `pre-commit` discovery / shell completion
works:

```bash
# Equivalent to: uv run python scripts/local_ci.py wheel
pre-commit run --hook-stage manual local-ci-wheel

# Equivalent to: uv run python scripts/local_ci.py matrix
pre-commit run --hook-stage manual local-ci-matrix

# Equivalent to: scripts/local_ci_linux_docker.sh
pre-commit run --hook-stage manual local-ci-linux
```

Manual stage = does not auto-run on commit or push, only when invoked.

## Reports

Each successful run writes a Markdown report to `scripts/`:

- `local_ci_report-<sys>-<arch>-cp<py>.md` — one Python on one platform.
- `local_ci_report-<sys>-<arch>-matrix.md` — the full matrix on one platform.

For example, after a macOS-arm64 matrix run you'd see:

```
scripts/local_ci_report-darwin-arm64-cp310.md
scripts/local_ci_report-darwin-arm64-cp311.md
scripts/local_ci_report-darwin-arm64-cp312.md
scripts/local_ci_report-darwin-arm64-cp313.md
scripts/local_ci_report-darwin-arm64-matrix.md
```

The matrix file is the one to paste into the PR / issue. Per-Python files
are useful when a particular Python version fails and we need detail.

## Recipe for a reporter (efm46 / arm64 / etc.)

1. Install `uv`.
2. `git clone git@github.com:lanl/bngsim.git`
3. `cd bngsim`
4. `uv run python scripts/local_ci.py matrix`
5. Send back the `local_ci_report-<sys>-<arch>-matrix.md` file (or paste its
   contents into the relevant GitHub issue / DM).

## On 2026-06-01

When the GitHub Actions quota resets:

1. Revert `.github/workflows/build.yml` to restore the `push` /
   `pull_request` triggers (commented-out at the top of the `on:` block).
2. Push a no-op commit on `feature/bngsim` to trigger one confirmation run.
3. Compare the CI matrix output against the most recent local matrix reports.
4. If they agree, the local-CI harness can keep running as a pre-merge gate
   on every contributor's box without needing GHA to be the source of truth.
