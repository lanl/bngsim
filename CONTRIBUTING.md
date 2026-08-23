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

### One build directory per interpreter, not per configuration

`build-dir = "build/{wheel_tag}"` — the wheel tag is the Python version and the
platform. It says nothing about the virtual environment being installed into, or
about the options passed, so **every** build for one interpreter shares one CMake
cache. A second install with different options rewrites the cache the first one
is using (GH #459):

```bash
uv pip install --python /tmp/other/bin/python \
    --config-settings=cmake.define.BNGSIM_ENABLE_MIR=ON \
    --config-settings=cmake.define.BNGSIM_ENABLE_KLU=OFF .
```

Give a differently configured install its own tree and the two stay apart:

```bash
--config-settings=build-dir=/tmp/otherbuild/{wheel_tag}
```

`rebuild_editable.py` names every feature option on its configure line rather
than inheriting any of them, and stops rather than building on a tree whose cache
disagrees with what it asked for — the recovery it names is `rm -rf` on the tree
plus a reinstall, because reconfiguring an already-built tree leaves link
settings from the configuration it was built with. To change an option for one
rebuild, set the same-named environment variable:

```bash
BNGSIM_ENABLE_MIR=1 python scripts/rebuild_editable.py
```

### Re-locking after a `pyproject.toml` change

`uv.lock` records `provides-extras`, so **any** edit to this project's dependency
metadata — adding an extra, moving a pin — invalidates the lock. CI's
`uv sync --extra dev` then re-resolves instead of installing from it, and a
re-resolve has to fetch metadata for every locked package, including the
`parity` group's git-pinned PyBioNetGen. So the CI failure reads as a
PyBioNetGen build error on a PR that never touched PyBioNetGen. **Run `uv lock`
and commit the result in the same PR.**

PyBioNetGen is in `no-build-isolation-package`, which means it builds against
your `.venv` rather than an isolated one — so that venv needs the two things its
legacy `setup.py` assumes and does not declare:

```bash
uv pip install setuptools pip   # setup.py imports setuptools, then shells out
                                # to `pip install numpy`
uv lock                         # ~20 min: it downloads BNG2.pl at build time
```

Without them `uv lock` fails with `ModuleNotFoundError: No module named
'setuptools'`, or with a `CalledProcessError` from `pip install numpy` — neither
of which names the venv as the missing piece.

**Do not run `uv lock` when you have not changed a dependency.** Bumping the
version for a release does not need it: edit the `version = "..."` line under
`[[package]] name = "bngsim"` in `uv.lock` by hand, and confirm with
`uv lock --check`, which exits 0. Running `uv lock` on an unchanged tree
rewrites roughly 118 unrelated lines, dropping redundant
`python_full_version >= '3.11'` markers from inside the `amici` package block.
That churn is cosmetic — the `amici` gate itself survives, and both forms export
an identical dependency set — but which form you get depends on your uv version,
so the file ping-pongs between contributors for no reason. `uv lock --check`
runs in CI and is tolerant of the difference, so it will not fail on either form.

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

### The suite's artifact caches are its own

A pytest session redirects both of bngsim's content-addressed caches — compiled
`.so` artifacts and BNG2.pl-generated networks — away from `~/.cache/bngsim` and
into a directory the suite owns (issue #372). It is `.pytest_cache/d/bngsim/` by
default, or a per-run temp directory when the cache provider is off
(`-p no:cacheprovider`, which every CI leg passes). Both the module attribute and
the env var are set, so subprocesses that import bngsim land there too.

The directory is **persistent**, so runs stay warm: a cold full suite measured
19m19s against roughly 14m warm. Two consequences worth knowing:

```bash
uv run python -m pytest python/tests/ --cache-clear   # force a cold-cache run
```

- `pytest --cache-clear` wipes it, which is the supported way to test a cold
  compile path across the whole suite;
- `BNGSIM_TEST_CACHE_DIR=/path` relocates it — node-local scratch, a cache CI
  restores between runs, or a throwaway directory. It overrides
  `BNGSIM_CODEGEN_CACHE_DIR` / `BNGSIM_BNGL_CACHE_DIR` for the session rather
  than deferring to them: exporting those points bngsim's *real* cache somewhere,
  and the suite has no more business writing there than in `~/.cache`.

Being persistent, it still grows — every `_CODEGEN_CACHE_KEY` change orphans what
came before it. The difference is that it grows somewhere disposable, under a
directory `git clean -xdf` and `--cache-clear` both remove, and that
`bngsim-cache info -C .pytest_cache/d/bngsim/codegen` will tell you how much of
it your last emitter edit stranded.

A test that needs true isolation — a cold compile, or a cache whose entries it
counts — still monkeypatches `_codegen.CACHE_DIR` (or `_bngl_loader.CACHE_DIR`) to
a `tmp_path`, which overrides the session value and restores it afterwards. Do
that in particular when the test **invents a key**: a fabricated
`_CODEGEN_CACHE_KEY` or model hash produces an artifact that is orphaned the
moment it is written, and it must not be written where anything else will see it.

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

The key is also carried in the artifact's *name* — `rhs_<key>_<hash><suffix>`, built
by `_artifact_stem()` (issue #363) — so `bngsim-cache info` can report how much of a
cache your edit just orphaned and `bngsim-cache prune --orphaned` can sweep exactly
that. Build every artifact path through that helper: a call site that spells the name
itself is a silent cache miss, not an error.

## Fixes that need a capability key

Most fixes need no announcement beyond the changelog. A few do, and the test is
narrow (issue #431): **does a build without this fix return a wrong number
instead of refusing?** If so, add a key to `capabilities()["features"]`.

The reason is that nothing else can carry the answer. The version string
identifies a release *cycle* rather than a build — bngsim bumps `__version__` at
the start of one, so a from-source install made before your fix declares the
number of the release that carries it — and a `hasattr` probe cannot see a
change in what a build computes rather than in what it exposes. So a consumer
deciding whether to trust the result has nothing to read, and its two ways of
guessing wrong are not equally bad: guessing "absent" costs a slower path,
guessing "present" costs a plausible wrong answer nobody looks at twice.

When you add one:

- Give it a real probe where the fix can actually be missing — a binding the fix
  added, for a fix that is partly C++, since a source checkout builds the
  extension separately and it can lag the Python layer.
- Publish it on every build, `True` or `False`. An absent key means "too old to
  have been asked", which a consumer must handle as a third case.
- Give `missing[name]` a sentence that says what goes wrong and what to do.
- Measure what the key claims in `python/tests/test_behaviour_capability_keys.py`
  and assert the key against the measurement, so the key cannot outlive the
  behaviour.
- Names are permanent: `capabilities()` promises that existing keys are never
  renamed or removed.

## Before opening a PR

- Keep new code consistent with the surrounding style (`.clang-format` for C++,
  `ruff` for Python).
- Add or update tests for behavior changes.
- Update [`CHANGELOG.md`](CHANGELOG.md) and, where user-facing, the relevant page
  under [`docs/`](docs/).

See the [development docs](docs/development/) for building wheels locally
(`cibuildwheel`) and for guides on adding built-in functions and objectives.
