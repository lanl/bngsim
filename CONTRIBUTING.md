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
# editable with the test dependencies (pytest, scipy, antimony, jsonschema, …).
uv sync --extra test
# ...or `--extra dev` for the full toolchain (roadrunner, jax, pandas, ruff, …).
```

> **`uv sync` prunes — name every extra you want, every time.** It makes the venv
> match *exactly* the extras you pass, uninstalling everything else. So running
> `uv sync --extra test` in a venv provisioned with `--extra dev` silently strips
> roadrunner, jax, vivarium and the rest. Pick the line you want and stay on it
> (`--extra dev` is a superset of `--extra test`). By the same rule, a package you
> hand-install with `uv pip install` disappears at the next sync unless it is
> declared in `pyproject.toml` — if you need it, declare it.

Requires CMake ≥ 3.20, a C++17 compiler, and Python ≥ 3.10 (uv can provision the
interpreter). The build requires SuiteSparse/KLU (`brew install suite-sparse` /
`apt-get install libsuitesparse-dev` / `conda install -c conda-forge suitesparse`).

The editable install does **not** auto-rebuild the C++ extension
(`editable.rebuild = false`). After changing C++, refresh it:

```bash
uv sync --reinstall-package bngsim   # rebuild + reinstall the editable extension
```

### Legacy BioNetGen (BNG2.pl) for `parity_checks/`

bngsim has no BNGL parser, so every BNGL job shells out to BNG2.pl for network
generation before simulating in-process. Without it the `parity_checks/` engine
tests skip. Two ways to supply it — pick either:

```bash
uv sync --extra dev --group parity   # installs the pinned PyBioNetGen, which
                                     # bundles BNG2.pl + bin/run_network + bin/NFsim
export BNGPATH=/path/to/BioNetGen-2.9.3   # ...or point at an install you already have
```

Resolution order is `$BNG2_PL` → `$BNGPATH` → PyBioNetGen's bundled copy, so an
env var always **overrides** an installed package (`_core.bngpath`). `$BNGPATH`
may be the BioNetGen folder or the `BNG2.pl` script itself. When nothing
resolves, the skip message names every location that was tried — if a test says
"no usable BNG2.pl", that message is the diagnosis, not a dead end.

The `parity` group pins the same commit as
`parity_checks/requirements-pybionetgen.txt`; `test_pin_agreement.py` fails if
they drift. For a fully isolated parity/benchmark environment (rather than
adding PyBioNetGen to your dev venv), use
`parity_checks/bng_parity/bootstrap_parity_env.py` instead.

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
