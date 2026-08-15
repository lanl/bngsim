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
uv sync --extra test --reinstall-package bngsim   # rebuild + reinstall the extension
```

The extras belong on that line — `--reinstall-package` does not exempt it from
the prune rule above, so a bare `uv sync --reinstall-package bngsim` rebuilds the
extension *and* strips every extra on the way (GH #229 is what that costs).

The faster path is an incremental cmake rebuild, which touches nothing else in
the venv:

```bash
python scripts/rebuild_editable.py
```

That one needs `pybind11` importable in the venv, which is why the `dev` extra
declares it: `pybind11` is a `[build-system]` requirement, so uv installs it into
a transient isolated build env and never into `.venv`. On `--extra test` the
script falls back to whatever pybind11 the system supplies, and tells you when it
is doing that.

### Legacy BioNetGen (BNG2.pl) for `parity_checks/`

bngsim has no BNGL parser, so every BNGL job shells out to BNG2.pl for network
generation before simulating in-process — the `parity_checks/` engine tests, the
cBNGL round-trip gate, and (since #162) `Model.from_bngl` itself. Without it
those skip. Two ways to supply it — pick either:

```bash
uv sync --extra dev --group parity   # installs the pinned PyBioNetGen, which
                                     # bundles BNG2.pl + bin/run_network + bin/NFsim
export BNGPATH=/path/to/BioNetGen-2.9.3   # ...or point at an install you already have
```

Resolution order is `$BNG2_PL` → `$BNGPATH` → `BNG2.pl` on `$PATH` →
PyBioNetGen's bundled copy, so an env var always **overrides** an installed
package. `$BNGPATH` may be the BioNetGen folder or the `BNG2.pl` script itself.
When nothing resolves, the skip message names every location that was tried — if
a test says "no usable BNG2.pl", that message is the diagnosis, not a dead end.

The resolver is `bngsim._bngpath`, in the shipped package since #162 because
`Model.from_bngl` needs it; `_core.bngpath` re-exports it and adds the
`sys.exit`-ing `require_bng` the sweep entrypoints use. Do not add a local
BNG2.pl lookup anywhere — that module's header records the six near-duplicates
that disagreed about precedence, and a seventh had appeared in
`bngsim.convert._bng2` before it was folded back in.

The `parity` group pins the same commit as
`parity_checks/requirements-pybionetgen.txt`; `test_pin_agreement.py` fails if
they drift. It is **not** the `bngl` extra: that extra wants the PyPI release
for BNG2.pl, this group wants an exact commit for engine-routing provenance, and
`dev` deliberately omits `bngl` so no environment has to reconcile a range with
a git pin. For a fully isolated parity/benchmark environment (rather than adding
PyBioNetGen to your dev venv), use
`parity_checks/bng_parity/bootstrap_parity_env.py` instead.

### AMICI for `parity_checks/amici_parity/`

AMICI is the second reference engine — the SBML/CVODES oracle `amici_parity`
compares against, and the only independent check on forward *sensitivities*.

```bash
uv sync --extra dev --group amici    # builds the pinned AMICI (~1-4 min)
```

It is a group rather than an extra because the pin is a commit, and a direct
reference in published metadata would make bngsim unpublishable — the same
trade `parity` makes. The commit must match
`parity_checks/amici_parity/AMICI_PIN.json`, which is the source of truth;
`test_amici_pin_agreement.py` fails if they drift. Do not relax it to
`amici==1.0.1` off PyPI: the pin is 12 commits past that tag and they touch the
engine, not just its tests.

Building it needs **swig, a C++ toolchain, and a BLAS** (macOS: Accelerate,
nothing to install; Linux: the build fetches `scipy-openblas64` itself).

> **If the clone fails with `git-lfs: command not found`**, that is your
> machine, not AMICI. A `git-lfs` install that has since been removed leaves
> `filter.lfs.*` in `~/.gitconfig` with `required = true`, and every clone of an
> LFS-using repo then aborts. Either reinstall `git-lfs` or drop those four
> keys (`git config --global --remove-section filter.lfs`).

Leaving AMICI undeclared is what used to break this: `uv sync` prunes anything
not in the lock, so a hand-installed AMICI disappeared at the next sync and the
whole suite went quiet. Worse for correctness than for coverage — with no second
engine in the environment, a finite-difference oracle that happens to share a
defect with the engine is the only oracle left, which is exactly how a 5x-wrong
sensitivity column survived review (see CHANGELOG, issue #144 follow-up).

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

## Changing generated code

Compiled codegen artifacts are cached under `~/.cache/bngsim/codegen` and keyed by
model content plus `_CODEGEN_CACHE_KEY` — **not** by the generated C, because
hashing the C would mean regenerating it on every cache probe.

`_CODEGEN_CACHE_KEY` is `_CODEGEN_VERSION` (a hand-maintained constant in
`python/bngsim/_codegen.py`) plus a digest of the source of every module that
determines the emitted C. **`_CODEGEN_SOURCE_MODULES`, in that same file, is that
list** — read it there rather than from a copy here. This paragraph used to
restate the names and went stale the first time one was added, so from #68 until
issue #267 it told you to bump the version by hand for an edit the digest had
covered all along. The digest means an edit to any module on that list
invalidates stale artifacts on its own, so a change cannot go silently inert on a
machine with a warm cache the way issues #41 and #43 did — there, the fixes
appeared to do nothing at all, reporting `dy/dp == 0` where a cold cache gave
correct values.

Two cases the digest does not cover, where you must bump `_CODEGEN_VERSION`
yourself and say why in the comment block above it:

- a **C++** change that alters `codegen_data()` and therefore the emitted source;
- any time you want to deliberately invalidate a release's caches.

## Before opening a PR

- Keep new code consistent with the surrounding style (`.clang-format` for C++,
  `ruff` for Python).
- Add or update tests for behavior changes.
- Update [`CHANGELOG.md`](CHANGELOG.md) and, where user-facing, the relevant page
  under [`docs/`](docs/).

See the [development docs](docs/development/) for building wheels locally
(`cibuildwheel`) and for guides on adding built-in functions and objectives.
