# Contributing to bngsim

Thanks for your interest in improving `bngsim`. This file covers the essentials;
the full [development documentation](docs/development/building.md) has the project
layout, extension guides, and CI details.

## Development setup

```bash
git clone https://github.com/lanl/bngsim.git
cd bngsim

# Editable install. Drop --no-build-isolation to let pip provision the
# build backend (scikit-build-core, pybind11) automatically.
pip install --no-build-isolation -e .
```

Requires CMake ≥ 3.20, a C++17 compiler, and Python ≥ 3.10.

## Running tests

```bash
python -m pytest python/tests/ -q    # Python test suite
./build/tests/test_bngsim            # C++ unit tests
```

## Before opening a PR

- Keep new code consistent with the surrounding style (`.clang-format` for C++,
  `ruff` for Python).
- Add or update tests for behavior changes.
- Update [`CHANGELOG.md`](CHANGELOG.md) and, where user-facing, the relevant page
  under [`docs/`](docs/).

See the [development docs](docs/development/) for building wheels locally
(`cibuildwheel`) and for guides on adding built-in functions and objectives.
