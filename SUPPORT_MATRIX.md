# bngsim Support Matrix

Reference for downstream consumers (most importantly PyBNF) of what
`pip install bngsim` actually delivers across OS × Python × backend.

## Wheel availability

CI builds prebuilt wheels via `cibuildwheel` on every push to `main` /
`feature/bngsim*` and on PRs targeting `main`. The matrix is:

| Runner OS      | Architecture | Deployment target  | Python versions       |
|----------------|--------------|--------------------|-----------------------|
| ubuntu-latest  | x86_64       | manylinux2014      | 3.10, 3.11, 3.12, 3.13|
| macos-14       | arm64        | macOS 11           | 3.10, 3.11, 3.12, 3.13|
| macos-15-intel | x86_64       | macOS 10.15        | 3.10, 3.11, 3.12, 3.13|
| windows-latest | AMD64        | Windows Server 2022| 3.10, 3.11, 3.12, 3.13|

Source distributions (sdist) are published alongside; sdist users will
build from source and need a C++17 compiler, CMake ≥ 3.20, and Ninja.

Not covered: Linux ARM64, macOS x86_64 < 10.15, Windows ARM64, musllinux,
PyPy. These are skipped by `CIBW_SKIP`.

## Bundled vs optional features

| Feature              | Availability                                  | Activation                         |
|----------------------|-----------------------------------------------|------------------------------------|
| ODE (CVODE + CVODES) | Bundled in every wheel (pinned SUNDIALS release archive via FetchContent) | always on                      |
| SSA (Gillespie)      | Bundled                                       | `method="ssa"`                     |
| PSA (partial scaling)| Bundled                                       | `method="psa"`                     |
| NFsim                | Bundled (vendored under `third_party/nfsim`)  | `HAS_NFSIM=True` everywhere        |
| RuleMonkey           | Bundled (vendored under `third_party/rulemonkey`) | `HAS_RULEMONKEY=True` everywhere |
| SBML loader          | Runtime dep `python-libsbml>=5.20`            | `HAS_LIBSBML=True` everywhere      |
| Antimony loader      | Optional extra: `pip install bngsim[antimony]`| `HAS_ANTIMONY` depends on extra    |
| KLU sparse solver    | Disabled in macOS wheels (no SuiteSparse vendoring); on by default in Linux/Windows wheels | `BNGSIM_ENABLE_KLU=ON` build flag |
| HDF5 save/load       | Optional extra: `pip install bngsim[hdf5]`    | requires `h5py`                    |
| pandas integration   | Optional extra: `pip install bngsim[pandas]`  | requires `pandas`                  |
| JAX gradient bridge  | Optional extra: `pip install bngsim[jax]`     | requires `jax`, `jaxlib`, `diffrax`|

Use `bngsim.capabilities()` at runtime to introspect what is available.

## Local validation log

GitHub Actions are paused (workflow_dispatch only) through 2026-06-01 — see
`bngsim/dev/plans/CI_LOCAL_FIX_AND_ACTIONS_PAUSE_PLAN_2026-05-12.md`. While
paused, the four-platform wheel matrix is validated via the local-CI
harness under `bngsim/scripts/` (see `bngsim/scripts/LOCAL_CI.md`); each
reporter runs `uv run python bngsim/scripts/local_ci.py matrix` on their
target platform and produces a verifiable Markdown report.

For each Python version below, the wheel is built in a fresh `uv venv`,
installed (with `[antimony,pandas]` extras when antimony has a wheel for
that platform/Python) into a *second* fresh venv, and smoke-tested via
`bngsim/scripts/local_ci_smoke.py` — covering `capabilities()`, the
`.net`/SBML/Antimony loaders, and `NfsimSession` + `RuleMonkeySession`.

### darwin-x86_64 / Python 3.10–3.13  (this development box; 2026-05-12)

bngsim 0.5.5, macOS 15.7.4 Sequoia, deployment target 10.15.

| Python | Wheel                                             | capabilities | .net+ODE | SBML+ODE | NFsim | RuleMonkey | Antimony |
|--------|---------------------------------------------------|--------------|----------|----------|-------|------------|----------|
| 3.10   | `bngsim-0.5.5-cp310-cp310-macosx_10_15_x86_64.whl` | PASS         | PASS     | PASS     | PASS  | PASS       | PASS     |
| 3.11   | `bngsim-0.5.5-cp311-cp311-macosx_10_15_x86_64.whl` | PASS         | PASS     | PASS     | PASS  | PASS       | PASS     |
| 3.12   | `bngsim-0.5.5-cp312-cp312-macosx_10_15_x86_64.whl` | PASS         | PASS     | PASS     | PASS  | PASS       | PASS     |
| 3.13   | `bngsim-0.5.5-cp313-cp313-macosx_10_15_x86_64.whl` | PASS         | PASS     | PASS     | PASS  | PASS       | PASS     |

### linux-x86_64 (manylinux2014, via Colima Docker)  (this development box; 2026-05-12)

bngsim 0.5.5, cibuildwheel 2.22.0 host-side, container image
`quay.io/pypa/manylinux2014_x86_64:2024.11.16-1`. Wheel built inside the
container, pytest + smoke run inside the container against the freshly
installed wheel.

| Python | Wheel                                                                              | build | pytest | smoke |
|--------|------------------------------------------------------------------------------------|-------|--------|-------|
| 3.12   | `bngsim-0.5.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (3.8 MB) | PASS  | 766 passed, 54 skipped, 12 deselected | PASS |

Notes: antimony and libroadrunner are unavailable *inside* the
manylinux2014 container (antimony only ships `manylinux_2_28`;
libroadrunner's wheel needs `libpython3.X.so.1.0` which manylinux2014
strips). The tests that need them skip via `pytest.importorskip`.
Outside the container, on any real Linux distro with glibc ≥ 2.28,
`pip install bngsim[antimony]` resolves cleanly.

cp310 / cp311 / cp313 in-container runs are deferred (each is ~35 min
on this hardware due to SUNDIALS rebuild); cp312 is the leg that
actually failed in CI run 25736978049 and is the one that needed
explicit validation post-fix.

### darwin-arm64 / Python 3.10–3.13 (M4 Max)

_Pending; will be filled in from the M4 Max reporter's matrix run._

### windows-amd64 / Python 3.10–3.13

_Pending; will be filled in from efm46's matrix run._

A single confirmation GitHub Actions run on or after 2026-06-01 (quota
reset) is the final validation that the local-CI matrix and the
GitHub-runner matrix agree.

## Verifying an installed wheel

The wheel build matrix runs the full `pytest` suite (skipping HDF5 tests
that need `bngsim[hdf5]`). To self-verify after installing:

```python
import bngsim
caps = bngsim.capabilities()
print(caps["version"], caps["features"])
# Expected: {"libsbml": True, "nfsim": True, "rulemonkey": True, ...}

# Load + simulate a tiny model
m = bngsim.Model.from_net("simple_decay.net")
r = bngsim.Simulator(m, method="ode").run(t_span=(0, 10), n_points=11)
assert r.n_times == 11
```

## Release checklist (for cutting a new bngsim version)

1. Bump `bngsim/pyproject.toml` `version` and `bngsim/python/bngsim/__init__.py` `__version__` (single source of truth — see issue #31).
2. Update `bngsim/CHANGELOG.md` with the user-visible diff.
3. Push to `feature/bngsim`; verify all four CI legs are green on cp310–cp313.
4. Tag `git tag v<x.y.z> && git push origin v<x.y.z>` — the `publish` job will run.
5. Confirm wheels appear on PyPI (one per OS × Python combination, plus sdist).
6. From a clean venv on each target OS (or via a temporary CI matrix), install
   with `pip install bngsim==<x.y.z>` and run the verification snippet above.
7. If a PyBNF release depends on this version, bump PyBNF's pinned bngsim version
   in its own `pyproject.toml` and run PyBNF's integration tests against the
   published wheel before announcing.

## Known caveats

- macOS wheels build with `-DBNGSIM_ENABLE_KLU=OFF` so they do not need to bundle
  Homebrew SuiteSparse dylibs that would conflict with the 10.15 deployment
  target. KLU is rarely the rate-limiting linear solver for BioNetGen-scale
  networks, but Jacobian-heavy ODE workloads may be measurably faster on
  Linux/Windows where KLU is on.
- `python-libsbml` is a hard runtime dependency. Wheels for it exist on PyPI for
  every target in the matrix above; users on exotic platforms may need to build
  it from source.
- `antimony` is optional because the published wheels lag CPython releases and
  the maintainers' release cadence is irregular. If you need Antimony support
  on a brand-new Python, fall back to `Model.from_sbml(...)` or pin to a Python
  for which `pip install antimony` succeeds.
