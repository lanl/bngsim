# bngsim Support Matrix

Reference for downstream consumers (most importantly PyBNF) of what
`pip install bngsim` actually delivers across OS × Python × backend.

## Wheel availability

`.github/workflows/wheels.yml` builds prebuilt wheels via `cibuildwheel` on
pushes to `main` that touch the build inputs (`pyproject.toml`, `CMakeLists.txt`,
`src/`, `include/`, `third_party/`, `python/bngsim/`, `ci/`) and on manual
`workflow_dispatch`. Those runs upload artifacts only; `release.yml` publishes to
PyPI on a `v*` tag. The matrix is:

| Runner OS      | Architecture | Deployment target  | Python versions       |
|----------------|--------------|--------------------|-----------------------|
| ubuntu-latest  | x86_64       | manylinux_2_28     | 3.10, 3.11, 3.12, 3.13|
| macos-14       | arm64        | macOS 11           | 3.10, 3.11, 3.12, 3.13|
| macos-15-intel | x86_64       | macOS 10.15        | 3.10, 3.11, 3.12, 3.13|
| windows-latest | AMD64        | Windows Server 2022| 3.10, 3.11, 3.12, 3.13|

Source distributions (sdist) are published alongside; sdist users will
build from source and need a C++17 compiler, CMake ≥ 3.20, and Ninja.

Not covered: Linux ARM64, macOS x86_64 < 10.15, Windows ARM64, musllinux,
PyPy. These fall out of the `build` / `skip` selectors in
`[tool.cibuildwheel]`.

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
| KLU sparse solver    | Bundled in every wheel (Linux/macOS/Windows) — SuiteSparse vendored by auditwheel/delocate/delvewheel | `HAS_KLU=True` everywhere |
| BLAS dense factor    | macOS wheels only (Accelerate); source builds where CMake finds a LAPACK | `HAS_LAPACK_DENSE` + `BNGSIM_LAPACK_DENSE=1` |
| HDF5 save/load       | Optional extra: `pip install bngsim[hdf5]`    | requires `h5py`                    |
| pandas integration   | Optional extra: `pip install bngsim[pandas]`  | requires `pandas`                  |
| JAX gradient bridge  | Optional extra: `pip install bngsim[jax]`     | requires `jax`, `jaxlib`, `diffrax`|

Use `bngsim.capabilities()` at runtime to introspect what is available.

### KLU is never optional in a published wheel

Every wheel is smoke-tested for the sparse solver before it can be published:

```
test-command = 'python -c "import bngsim; print(bngsim.__version__); assert bngsim.HAS_KLU"'
```

so a wheel that fell back to the dense solver would fail its own build. Linux
takes SuiteSparse from EPEL inside the manylinux image; macOS and Windows build
the KLU subset from source (`ci/build_suitesparse.sh`, `ci/build_suitesparse.ps1`)
— on macOS at the wheel's own deployment target, which is what makes vendoring
viable where linking Homebrew dylibs was not.

Source installs get the same guarantee: `[tool.scikit-build.cmake.define]` sets
`BNGSIM_REQUIRE_KLU=ON` for every pip build, and when no system SuiteSparse is
found CMake builds a pinned KLU subset from source (GH #209) instead of silently
degrading. `HAS_KLU=False` therefore only happens on a deliberate
`-DBNGSIM_ENABLE_KLU=OFF` build.

This matters for performance, not just capability: automatic sparse-solver
selection is the main reason BNGsim's advantage over dense-only paths grows with
network size, so it is on wherever the ODE backend is.

### The BLAS dense factor is the one feature that is macOS-only

`HAS_LAPACK_DENSE` (GH #84) is the opposite case from KLU, and the only entry in
the table above whose availability differs across published wheels. It reports
whether CMake linked a BLAS/LAPACK for the optimized dense factorization:

| Wheel               | Backend                | `HAS_LAPACK_DENSE` |
|---------------------|------------------------|--------------------|
| macOS arm64 / x86_64| Accelerate (system)    | `True`             |
| manylinux x86_64    | none found             | `False`            |
| Windows AMD64       | none found             | `False`            |

Nothing is missing on `False`: dense factorizations use the built-in LU, which
is what runs by default *everywhere* — the BLAS path is opt-in via
`BNGSIM_LAPACK_DENSE=1` and produces the same trajectory either way. So this
flag answers "will that environment variable do anything on this install?", and
the honest answer on a Linux or Windows wheel is no.

A source install on a host with LAPACK does get it: `apt-get install
liblapack-dev`, `dnf install lapack-devel` or a conda-forge `openblas` before
`pip install bngsim` is enough for `find_package(LAPACK)` to resolve, and
`bngsim.capabilities()["missing"]["lapack_dense"]` spells that out at runtime.
That configuration — an HPC or cluster box building from source against the
system LAPACK — is the population this backend actually reaches, which is why
CI gained a Linux-with-LAPACK leg on both the C++ and Python side (GH #269)
rather than relying on the macOS legs alone.

## Validation

GitHub Actions is the source of truth for the four-platform matrix. The
per-platform wheel legs (`wheels.yml`) each build with `BNGSIM_REQUIRE_KLU=ON`
and run the `cibuildwheel` `test-command` against the repaired wheel, so a leg
that is green is a wheel that imports and reports `HAS_KLU=True` in a clean
environment. `lint.yml`, `native-tests.yml`, `mir.yml`, `windows-nfsim.yml` and
`windows-tail.yml` cover the pre-commit hooks, the C++ unit suite, the MIR JIT
backend, and the Windows NFsim/RuleMonkey paths respectively.

`python-tests.yml` is the Python suite's own gate: the whole of `python/tests`,
on `ubuntu-latest` and `macos-14`, in the **default** build configuration — no
`-D` overrides, so KLU, NFsim and RuleMonkey are all on and the MIR backend is
off, which is what a wheel ships. It has no `paths:` filter and names no test
files, so it fires on every change and picks up new tests without a workflow
edit. Every other job above disables something or runs a curated list, which is
how the shipped configuration went untested on Linux and macOS until issue #169.

Check the latest results before trusting this table:

```bash
gh run list --workflow wheels.yml --limit 5
```

For pre-push validation without burning CI minutes, the local-CI harness under
`scripts/` reproduces a matrix leg on the current box — see
[scripts/LOCAL_CI.md](scripts/LOCAL_CI.md):

```bash
uv run python scripts/local_ci.py matrix
```

It builds each wheel in a fresh `uv venv`, installs it (with `[antimony,pandas]`
extras where antimony ships a wheel for that platform/Python) into a *second*
fresh venv, and smoke-tests via `scripts/local_ci_smoke.py` — `capabilities()`,
the `.net`/SBML/Antimony loaders, and `NfsimSession` + `RuleMonkeySession`.

One known environment quirk: inside the manylinux container, `antimony` and
`libroadrunner` are unavailable (antimony ships `manylinux_2_28` only;
libroadrunner's wheel needs the `libpython3.X.so.1.0` that manylinux images
strip). Tests needing them skip via `pytest.importorskip`. On any real Linux
distro with glibc ≥ 2.28, `pip install bngsim[antimony]` resolves cleanly.

## Verifying an installed wheel

Each wheel leg runs the `cibuildwheel` `test-command` — an import plus the
`HAS_KLU` assertion — against the repaired wheel in a clean environment. The full
`pytest` suite runs against the source tree, not against every wheel. To
self-verify after installing:

```python
import bngsim
caps = bngsim.capabilities()
print(caps["version"], caps["features"])
# Expected: {"libsbml": True, "nfsim": True, "rulemonkey": True, "klu": True, ...}
assert bngsim.HAS_KLU  # sparse solver is bundled in every published wheel

# Load + simulate a tiny model
m = bngsim.Model.from_net("simple_decay.net")
r = bngsim.Simulator(m, method="ode").run(t_span=(0, 10), n_points=11)
assert r.n_times == 11
```

## Release checklist (for cutting a new bngsim version)

1. Bump `version` in `pyproject.toml` — the single source of truth.
   `python/bngsim/_version.py` reads it back through `importlib.metadata`, so
   there is no second literal to keep in sync.
2. Update `CHANGELOG.md` with the user-visible diff.
3. Push to `main`; verify all four wheel legs are green on cp310–cp313.
4. Tag `git tag v<x.y.z> && git push origin v<x.y.z>` — `release.yml` builds and
   publishes to PyPI via Trusted Publishing. Rehearse first with a manual
   `workflow_dispatch` run targeting `testpypi`.
5. Confirm wheels appear on PyPI (one per OS × Python combination, plus sdist).
6. From a clean venv on each target OS (or via a temporary CI matrix), install
   with `pip install bngsim==<x.y.z>` and run the verification snippet above.
7. If a PyBNF release depends on this version, bump PyBNF's pinned bngsim version
   in its own `pyproject.toml` and run PyBNF's integration tests against the
   published wheel before announcing.

## Known caveats

- `python-libsbml` is a hard runtime dependency. Wheels for it exist on PyPI for
  every target in the matrix above; users on exotic platforms may need to build
  it from source.
- `antimony` is optional because the published wheels lag CPython releases and
  the maintainers' release cadence is irregular. If you need Antimony support
  on a brand-new Python, fall back to `Model.from_sbml(...)` or pin to a Python
  for which `pip install antimony` succeeds.
