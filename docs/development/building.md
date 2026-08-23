# Building & contributing

`CONTRIBUTING.md` in the repository root is the authoritative build guide. It
covers environment provisioning, the rebuild paths and their trade-offs, and the
one-build-directory-per-interpreter rule. This page is an orientation map for
finding your way around the tree.

## Build from source

```bash
git clone https://github.com/lanl/bngsim.git
cd bngsim

# Provision the venv with the test dependencies.
# `--extra dev` is a superset, adding roadrunner, jax, pandas and ruff.
uv sync --extra test
```

The editable install does **not** auto-rebuild the C++ extension. After changing
C++, the fast path is an incremental cmake rebuild that touches nothing else in
the venv:

```bash
python scripts/rebuild_editable.py
```

`uv sync --extra test --reinstall-package bngsim` also works and is the slower,
more thorough option. Name every extra on that line: `uv sync` prunes the venv to
match exactly the extras passed, so omitting them strips the rest.

## Running the tests

```bash
python -m pytest python/tests/ -q     # Python suite
./build/<wheel-tag>/tests/test_bngsim # C++ unit tests
```

Test counts are deliberately not quoted here. They moved by more than an order of
magnitude while this page said otherwise.

## Project structure

```
bngsim/
├── CMakeLists.txt          # C++ build (SUNDIALS, ExprTk, libbngsim)
├── pyproject.toml          # single source of truth for the version
├── CONTRIBUTING.md         # the authoritative build guide
├── include/bngsim/         # C++ public headers
├── src/                    # C++ implementation
│   ├── model.cpp           # NetworkModel
│   ├── cvode_simulator.cpp # ODE and forward sensitivities
│   ├── ssa_simulator.cpp   # SSA and PSA
│   ├── nfsim_simulator.cpp # in-process NFsim
│   ├── net_file_loader.cpp # .net parsing
│   ├── expression.cpp      # ExprTk wrapper, the mratio single source
│   └── _bngsim_core.cpp    # pybind11 bindings
├── python/bngsim/          # Python package
│   ├── _model.py           # Model
│   ├── _simulator.py       # Simulator
│   ├── _result.py          # Result
│   ├── _codegen.py         # generated-C emitters
│   ├── _jacobian.py        # symbolic Jacobian
│   ├── _switch_sensitivity.py  # crossing detection and dt*/dp
│   ├── _sbml_loader.py     # SBML
│   └── _bngl_loader.py     # BNGL, via BNG2.pl
├── python/tests/           # Python test suite
├── tests/                  # C++ tests and shared test data
├── third_party/            # vendored: sundials, exprtk, nfsim, rulemonkey, mir
├── scripts/                # vendoring, rebuild and local-CI tooling
├── parity_checks/          # cross-engine parity harnesses
└── docs/                   # this documentation
```

Each vendored tree under `third_party/` is generated source with its own
`VENDOR.json` and a matching `scripts/*_VENDORING.md`. Do not edit those trees
directly. Land the change upstream and re-run the vendoring script, or it is lost
at the next refresh.

## Running CI locally

```bash
python scripts/local_ci.py wheel --python 3.12
```

`scripts/local_ci_smoke.py` is the quicker variant, and
`scripts/local_ci_linux_docker.sh` runs the Linux leg in a container.
