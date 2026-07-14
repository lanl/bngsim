# Contributing to bngsim

Thanks for your interest in improving `bngsim`. This file covers the essentials;
the full [development documentation](docs/development/building.md) has the project
layout, extension guides, and CI details.

## Development setup

`bngsim` uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
git clone https://github.com/lanl/bngsim.git
cd bngsim

# Create the project venv (.venv), build the C++ extension, and install bngsim
# editable with the test dependencies (pytest, scipy, antimony, …).
uv sync --extra test
# ...or `--extra dev` for the full toolchain (roadrunner, jax, pandas, ruff, …).
```

Requires CMake ≥ 3.20, a C++17 compiler, and Python ≥ 3.10 (uv can provision the
interpreter). The build requires SuiteSparse/KLU (`brew install suite-sparse` /
`apt-get install libsuitesparse-dev` / `conda install -c conda-forge suitesparse`).

The editable install does **not** auto-rebuild the C++ extension
(`editable.rebuild = false`). After changing C++, refresh it:

```bash
uv sync --reinstall-package bngsim   # rebuild + reinstall the editable extension
```

## Running tests

`uv run` uses the project venv without needing `source .venv/bin/activate`:

```bash
uv run python -m pytest python/tests/ -q   # Python test suite
```

The C++ unit tests need a build configured with `-DBNGSIM_BUILD_TESTS=ON`; see the
[development docs](docs/development/building.md).

Install the git hooks (ruff / clang-format / mypy on commit; the pytest suite on
push, run via `uv run`) with:

```bash
uv run pre-commit install
```

## Before opening a PR

- Keep new code consistent with the surrounding style (`.clang-format` for C++,
  `ruff` for Python).
- Add or update tests for behavior changes.
- Update [`CHANGELOG.md`](CHANGELOG.md) and, where user-facing, the relevant page
  under [`docs/`](docs/).

See the [development docs](docs/development/) for building wheels locally
(`cibuildwheel`) and for guides on adding built-in functions and objectives.
