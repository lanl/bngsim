#!/bin/sh
# SBML Test Suite wrapper shim for bngsim (GH #241).
#
# Register in the official runner (GUI or headless) as:
#     executable: <this path>/bngsim_wrapper.sh
#     arguments:  %d %n %o %l %v
#     unsupported tags: (paste from bngsim-unsupported-tags.txt)
#
# It runs the Python wrapper with the bngsim virtualenv interpreter so the
# runner does not need bngsim on its own PATH. Override the interpreter with
# BNGSIM_PYTHON if the venv lives elsewhere.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${BNGSIM_PYTHON:-$HERE/../../../../.venv/bin/python}"
exec "$PYTHON" "$HERE/bngsim_wrapper.py" "$@"
