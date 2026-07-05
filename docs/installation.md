# Installation



## From wheel (recommended)

```bash
pip install bngsim
```

Prebuilt wheels are available for:

| Platform | Architectures | Python |
|----------|---------------|--------|
| Linux | x86_64 | 3.10–3.13 |
| macOS | arm64, x86_64 | 3.10–3.13 |
| Windows | x86_64 | 3.10–3.13 |

See [`SUPPORT_MATRIX.md`](https://github.com/lanl/bngsim/blob/HEAD/SUPPORT_MATRIX.md) for the full platform × Python ×
backend matrix, what each optional extra activates, and the release checklist.

## From source

```bash
# Requires: cmake ≥ 3.20, C++17 compiler, Python ≥ 3.10
cd bngsim
pip install .
```

`pip install .` uses build isolation, so pip provisions the build backend
(`scikit-build-core`, `pybind11`) automatically from `[build-system].requires`.

Pass `--no-build-isolation` *only* if you have already installed the build
backend into the active environment yourself:

```bash
pip install "scikit-build-core>=0.10" "pybind11>=2.13"
pip install --no-build-isolation .
```

Otherwise `--no-build-isolation` fails with
`ModuleNotFoundError: No module named 'scikit_build_core'`.

**Resource-constrained builds (containers, CI).** The C++ build defaults to
compiling with all available cores. On a machine with many CPUs but limited
RAM (e.g. a 12-core container capped at ~6 GiB), parallel compilation of the
vendored SUNDIALS/NFsim translation units can exhaust memory and get
OOM-killed. Cap the parallelism with the standard CMake env var:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 pip install .
```

Managed builds fetch the pinned SUNDIALS release archive recorded in
`third_party/sundials/VENDOR.json`. ExprTk is vendored directly.
Set `-DBNGSIM_USE_SYSTEM_SUNDIALS=ON` to use an environment-managed SUNDIALS
install instead. See `scripts/SUNDIALS_VENDORING.md` for the supported
refresh/check workflow.

## Sparse solver (SuiteSparse / KLU) — required for large / genome-scale models

For models with more than ~50 species, the ODE backend routes the CVODE Newton
solve to a **sparse** direct linear solver (SuiteSparse/KLU). Without KLU the
build falls back to the **dense** solver, which factorizes the full N×N Jacobian
at O(N³). For a genome-scale model this is the difference between minutes and
hours — e.g. a 74,795-species network has a Jacobian that is 99.997% zeros
(dense storage ≈ 45 GB), so the dense path is catastrophic.

KLU is **discovered automatically at build time** from any standard install
location, conda prefix, HPC module, or `CMAKE_PREFIX_PATH` (GH #209). If
SuiteSparse is not found, the build still succeeds — dense-only — so verify
(below) after installing on a new machine, or pass `-DBNGSIM_REQUIRE_KLU=ON` to
turn a miss into a hard build failure (recommended for HPC/CI deployments).

**macOS** — found from Homebrew:

```bash
brew install suite-sparse
```

**Linux (system package)** — found from `/usr`:

```bash
sudo apt-get install libsuitesparse-dev     # Debian/Ubuntu
sudo dnf install suitesparse-devel           # RHEL/Rocky/Fedora
```

**HPC / conda (no root, modules-based clusters).** Provide SuiteSparse, then
build bngsim from source — discovery honors `$CONDA_PREFIX` and
`CMAKE_PREFIX_PATH` with no explicit paths:

```bash
conda install -c conda-forge suitesparse     # discovered via $CONDA_PREFIX
#   …or on a module cluster:  module load suitesparse

# Build from source. $CONDA_PREFIX is searched automatically; -DBNGSIM_REQUIRE_KLU=ON
# fails the build (instead of going dense-only) if SuiteSparse is somehow not found.
CMAKE_ARGS="-DBNGSIM_REQUIRE_KLU=ON" \
  pip install --no-binary=bngsim --force-reinstall bngsim
```

If SuiteSparse lives on a non-standard prefix that is neither `$CONDA_PREFIX` nor
on `CMAKE_PREFIX_PATH`, point CMake at it with `-DKLU_ROOT=<prefix>` (or
`-DSUITESPARSE_ROOT`, or an explicit `-DKLU_INCLUDE_DIR`/`-DKLU_LIBRARY_DIR`).
Watch the configure log for `BNGsim: sparse Jacobian support ENABLED (KLU)` (vs
`DISABLED (KLU not found)`).

**Verify KLU is active** in any install — no model needed:

```bash
python -c "import bngsim; print('KLU available:', bngsim.capabilities()['features']['klu'])"
# -> KLU available: True
```

If it prints `False`, the install is dense-only — large models will be orders of
magnitude slower, and `bngsim.capabilities()['missing']['klu']` carries the
rebuild recipe. A large model run on a dense-only install also emits a one-time
`bngsim.DenseSolverFallbackWarning` at `run()`. See
internal#209 for details.

## Editable rebuilds for local development

`pip install -e` for `bngsim` is only partially in-place:

- Python modules are imported from `bngsim/python/bngsim/` via `_bngsim_editable.pth`.
- The compiled runtime extension is imported from your environment's
  `site-packages/bngsim/_bngsim_core...so`.

That means rebuilding `bngsim/build/{wheel_tag}/_bngsim_core...so` is not enough on its
own. Python will keep importing the copy in `site-packages/` until you reinstall or run
`cmake --install` against the live environment. The current editable hook is the default
scikit-build-core redirect mode with `editable.rebuild = false`, so imports do not
automatically refresh the copied extension.

Use the helper below from the `bngsim/` project directory (or via `uv run --directory bngsim`)
to rebuild and reinstall the extension for the current interpreter. On macOS it also
reconfigures stale editable caches to the current interpreter architecture and rebuilds only
the `_bngsim_core` extension target:

```bash
uv run --directory bngsim python scripts/rebuild_editable.py
```

Then run a quick regression check against the installed runtime:

```bash
uv run --directory bngsim python -m pytest python/tests/test_method_normalization.py -q
```

## Optional dependencies

```bash
pip install bngsim            # core package
pip install bngsim[antimony]  # Antimony .ant loading
pip install bngsim[pandas]   # pandas DataFrame support
pip install bngsim[hdf5]     # HDF5 save/load (h5py)
pip install bngsim[dev]      # all of the above + pytest + ruff
```

## Antimony version note

- Antimony-backed model loading requires `bngsim[antimony]`.
- For repeated SBML conversion workloads (`loadSBMLFile` + `getSBMLString` loops), use
  Antimony `3.1.2` due an upstream memory-fix patch.
- Until `3.1.2` is published on PyPI, download the wheel from:
  `https://github.com/sys-bio/antimony/actions/runs/22922160697/`

## Capability introspection

Downstream tools (PyBNF, PyBioNetGen, etc.) can probe what an installed
`bngsim` actually supports without a try/except dance. Two surfaces:

**Module-level boolean flags** — match the existing pattern; safe to read
unconditionally.

```python
import bngsim
bngsim.HAS_NFSIM       # compiled C++ NFsim backend
bngsim.HAS_RULEMONKEY  # compiled C++ RuleMonkey backend
bngsim.HAS_LIBSBML     # optional Python dependency 'python-libsbml'
bngsim.HAS_ANTIMONY    # optional Python dependency 'antimony'
```

**Aggregator** — `bngsim.capabilities()` returns a structured dict whose
schema is stable across releases. Feature names will not be renamed or
removed; new features may be added.

```python
import bngsim
caps = bngsim.capabilities()
caps["version"]                       # bngsim package version, e.g. '0.4.1'
caps["features"]["sbml_ssa"]          # bool — can this install run SBML SSA?
caps["features"]["antimony_import"]   # bool — needs both libsbml AND antimony
caps["missing"]                       # dict[str, str] — only contains entries
                                      # for unavailable features
```

`caps["missing"][name]` distinguishes the two failure modes a downstream
tool needs to report differently:

- **Compiled backend gap** — message starts with "<X> backend not
  present in this install" and names both the vendored source path and
  the CMake flag (e.g. `-DBNGSIM_BUILD_RULEMONKEY=OFF`). NFsim and
  RuleMonkey are vendored in `third_party/` and built by default in
  source builds, so this typically means the install came from a wheel
  that excludes the backend (or the user explicitly built with the flag
  set OFF). Caller should tell users to reinstall a wheel/build that
  includes the backend.
- **Optional Python dependency** — message starts with "optional
  dependency" and names the PyPI package (`'python-libsbml'`,
  `'antimony'`). Caller should tell users to `pip install` it.

Stable feature keys: `nfsim`, `rulemonkey`, `libsbml`, `antimony`,
`sbml_import`, `sbml_ssa`, `sbml_psa`, `antimony_import`, `codegen`.
