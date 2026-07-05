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
trap "rm -rf $TMPDIR" EXIT

cp "${TEST_SRC}"/test_*.py "$TMPDIR/"
cp "${TEST_SRC}/conftest.py" "$TMPDIR/"

# Run pytest from the temp dir
cd "$TMPDIR"
BNGSIM_TEST_DATA="$DATA_DIR" \
    "$PYTHON" -m pytest "$TMPDIR" \
    -v --tb=short \
    "$@"
