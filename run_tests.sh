#!/usr/bin/env bash
# bngsim/run_tests.sh — Run Python tests against the installed bngsim package
#
# Usage:
#   cd bngsim && ./run_tests.sh           # run all tests
#   cd bngsim && ./run_tests.sh -k ssa    # run only SSA tests
#   cd bngsim && ./run_tests.sh -x        # stop on first failure
#
# The source tree's python/bngsim/ can shadow the installed package.
# This script avoids that by building a stand-in for the repo in a temp dir:
#   1. python/tests/ is COPIED to <tmp>/python/tests/
#   2. every other top-level entry is SYMLINKED into <tmp>/
#   3. <tmp>/python/ therefore holds tests/ and nothing else, so `import bngsim`
#      cannot find the source package and falls through to the installed one
#
# Point 2 is what makes the tests work rather than merely run: a test that
# resolves tests/data, benchmarks/ or scripts/ relative to its own __file__
# finds the real thing, because <tmp> IS the repo as far as a walk-up can tell.
#
# Prerequisites:
#   cd bngsim && pip install --no-build-isolation .   # build + install
#   pip install pytest numpy pandas                    # test deps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_SRC="${SCRIPT_DIR}/python/tests"
DATA_DIR="${SCRIPT_DIR}/tests/data"

# Verify test data exists
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Test data directory not found: $DATA_DIR" >&2
    exit 1
fi

# Allow overriding the Python interpreter (default: python3)
PYTHON="${PYTHON:-python3}"

# Verify bngsim is importable
if ! "$PYTHON" -c "import bngsim" 2>/dev/null; then
    echo "ERROR: bngsim is not installed in $($PYTHON --version 2>&1)." >&2
    echo "  Interpreter: $PYTHON" >&2
    echo "  Install with: cd bngsim && pip install --no-build-isolation ." >&2
    echo "  Or set PYTHON=/path/to/.venv/bin/python ./run_tests.sh" >&2
    exit 1
fi

# Build the stand-in repo. Only python/ is withheld; everything else is linked,
# not copied, so this costs nothing however large the tree is.
#
# Before issue #590 the copy went to <tmp>/tests with nothing beside it, which
# hid far more than python/bngsim/: 69 modules resolve tests/data, benchmarks/
# or scripts/ by walking up from __file__, and under that layout the walk-up
# left the repo entirely. The assert-based ones failed and the .exists()-guarded
# ones skipped as "not in this checkout (installed package)" while the source
# tree was right there -- 64 failures, 181 errors and 359 spurious skips, none
# of them a real defect in bngsim.
#
# NOT named TMPDIR. Where that variable is already exported -- launchd exports it
# on macOS, and plenty of Linux CI images set it -- assigning to it silently
# redirects every CHILD's temp files into the stand-in, because Python's tempfile
# reads it. pytest's own tmp_path would then land inside a tree whose root carries
# bngsim's pyproject.toml, so a walk-up out of any temp directory would "find" the
# source root there, and the per-run artifact cache would land inside rootdir.
# The script had done this since it was written; it only became visible once the
# stand-in was the rootdir and three tests started failing that way (issue #590).
STANDIN=$(mktemp -d)
trap 'rm -rf "$STANDIN"' EXIT

# python/tests is COPIED, into a directory named `tests` under a `python` that
# holds nothing else. The names are load-bearing: python/tests/__init__.py makes
# it a package, so pytest walks up to `python` and puts THAT on sys.path -- the
# one directory whose bngsim/ we are hiding.
mkdir -p "$STANDIN/python/tests"
cp "${TEST_SRC}"/*.py "$STANDIN/python/tests/"

# Everything else is SYMLINKED. The caches are excluded so a run here cannot
# write into the developer's own (GH #372 gave the suite its own cache for that
# reason), and python/ is the point of the whole exercise.
shopt -s dotglob nullglob
linked=""
for entry in "${SCRIPT_DIR}"/*; do
    name=$(basename "$entry")
    case "$name" in
        python | .pytest_cache | .mypy_cache | .ruff_cache) continue ;;
    esac
    ln -s "$entry" "$STANDIN/$name"
    linked="$name"
done
shopt -u dotglob nullglob

# Refuse to continue if those were not real symlinks. `ln -s` copies instead of
# linking under Git Bash/MSYS unless Windows Developer Mode is on (or
# MSYS=winsymlinks:nativestrict is set), and a silent deep copy here would
# duplicate build/, third_party/, .git and .venv -- minutes of I/O and gigabytes,
# for a script whose whole cost used to be 300 small files. Fail loudly instead.
if [[ -n "$linked" && ! -L "$STANDIN/$linked" ]]; then
    echo "ERROR: $STANDIN/$linked is not a symlink — this shell copied it instead." >&2
    echo "  run_tests.sh needs real symlinks to stand the repo up cheaply." >&2
    echo "  On Windows: enable Developer Mode, or set MSYS=winsymlinks:nativestrict," >&2
    echo "  or run the suite under WSL. Elsewhere: run it in place instead —" >&2
    echo "    python -m pytest python/tests/ -q" >&2
    exit 1
fi

# Run pytest from the stand-in. BNGSIM_TEST_DATA and BNGSIM_SOURCE_ROOT point at
# the REAL tree rather than the link farm, for the modules that consult them
# (python/tests/_source_root.py) — both resolve to the same files either way, but
# a path that says what it is beats one that has to be followed to find out.
#
# pyproject.toml is among the symlinks, so unlike before #590 pytest finds the
# project config here: [tool.pytest.ini_options] applies, addopts supplies
# --import-mode=importlib, and the run is configured exactly as an in-place one.
cd "$STANDIN"
BNGSIM_TEST_DATA="$DATA_DIR" \
BNGSIM_SOURCE_ROOT="$SCRIPT_DIR" \
    "$PYTHON" -m pytest "$STANDIN/python/tests" \
    "$@"
