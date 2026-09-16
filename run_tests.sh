#!/usr/bin/env bash
# bngsim/run_tests.sh — Run Python tests against the installed bngsim package
#
# Usage:
#   cd bngsim && ./run_tests.sh           # run all tests
#   cd bngsim && ./run_tests.sh -k ssa    # run only SSA tests
#   cd bngsim && ./run_tests.sh -x        # stop on first failure
#
# The source tree's python/bngsim/ can shadow the installed package.
# This script avoids that by:
#   1. Copying test files to a temp directory
#   2. Setting BNGSIM_TEST_DATA so fixtures find the .net files
#   3. Running pytest from the temp dir (no source tree on sys.path)
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

# Copy tests to a temp directory to avoid source tree shadowing
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Copy EVERY .py, into a directory named `tests` (issue #578). python/tests is a
# package, not a loose pile of test modules: __init__.py is what makes
# test_sbml_psa.py's `from .test_ssa_psa_volume import ...` resolve, and the
# sibling helpers _ci_workflow.py and _extrande_reference.py are imported at
# module scope by five more. Copying only test_*.py + conftest.py left all six
# raising at collection, and pytest aborts the whole run on a collection error —
# so this script could never execute a single test.
#
# The directory name matters as much as the contents. With __init__.py present,
# pytest derives the package name from the directory and puts its PARENT on
# sys.path; a mktemp name like `tmp.AbC123` is not a legal package name. Naming
# it `tests` reproduces the source layout exactly, and the parent it adds to
# sys.path is $TMPDIR rather than the repo's python/ — which is the shadowing
# this script exists to avoid.
mkdir "$TMPDIR/tests"
cp "${TEST_SRC}"/*.py "$TMPDIR/tests/"

# Run pytest from the temp dir. BNGSIM_TEST_DATA lets a test find the .net
# fixtures, and BNGSIM_SOURCE_ROOT the repo itself, neither being reachable by
# the __file__ walk-up that works for an in-place run. Both are the env vars the
# suite's own `_source_root()` helpers already look for, in that order.
#
# NOTE the temp rootdir has no pyproject.toml, so [tool.pytest.ini_options] does
# not apply here: -v --tb=short are repeated below because addopts cannot supply
# them, and the run uses pytest's default `prepend` import mode rather than the
# `importlib` the in-place run gets. That is what makes the `tests` package name
# above load-bearing — prepend walks up through __init__.py to find the package
# root and puts its parent on sys.path, which is how test_sbml_psa.py's relative
# imports resolve here.
cd "$TMPDIR"
BNGSIM_TEST_DATA="$DATA_DIR" \
BNGSIM_SOURCE_ROOT="$SCRIPT_DIR" \
    "$PYTHON" -m pytest "$TMPDIR/tests" \
    -v --tb=short \
    "$@"
