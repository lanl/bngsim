#!/usr/bin/env bash
# Linux x86_64 wheel build + smoke locally via Docker (manylinux2014).
#
# cibuildwheel itself runs on the host and spawns its own manylinux2014_x86_64
# container via the host's Docker CLI. CIBW_TEST_COMMAND then runs pytest +
# our local_ci_smoke.py *inside* that container against the freshly built
# wheel — mirroring exactly what the GHA `Wheels • ubuntu-latest` job does
# without burning Actions minutes.
#
# Requirements:
#   - Docker (Colima, Docker Desktop, OrbStack — anything providing `docker`).
#   - `uv` on the host PATH.
#
# Output:
#   - wheelhouse-local/bngsim-<ver>-cp312-cp312-manylinux*_x86_64.whl
#   - bngsim/scripts/local_ci_report-linux-x86_64-cp312.md (written by smoke)
#
# By default builds cp312 only (the leg that actually failed in CI). To build
# the full cp310/311/312/313 matrix in-container, set MATRIX=1:
#   MATRIX=1 bngsim/scripts/local_ci_linux_docker.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
WHEELHOUSE="${REPO_ROOT}/wheelhouse-local"
mkdir -p "$WHEELHOUSE"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on host PATH"
  echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found on host PATH"
  echo "Install Colima:  brew install colima docker && colima start"
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "error: Docker daemon not reachable"
  echo "Start Colima:    colima start"
  exit 2
fi

# Provision a host-side Python + cibuildwheel via uv. cibuildwheel will then
# pull the manylinux2014_x86_64 image itself and build inside it.
echo "==> setting up host-side cibuildwheel environment"
uv python install 3.12
VENV="${REPO_ROOT}/.venv-cibuildwheel-linux"
rm -rf "$VENV"
uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" cibuildwheel==2.22.0

# Which CPython tags to build inside the container.
if [[ "${MATRIX:-0}" = "1" ]]; then
  CIBW_BUILD="cp310-* cp311-* cp312-* cp313-*"
else
  CIBW_BUILD="cp312-*"
fi

# CIBW_TEST_COMMAND runs inside the manylinux container against the installed
# wheel. We run pytest (skipping HDF5 / save_load), then the smoke script that
# generates the per-platform Markdown report. `{project}` is replaced by
# cibuildwheel with the path to the bngsim/ source dir inside the container.
read -r -d '' CIBW_TEST_COMMAND <<'PYTEST' || true
python -m pytest {project}/bngsim/python/tests -v --tb=short --import-mode=importlib -k "not hdf5 and not h5py and not save_load" && \
python {project}/bngsim/scripts/local_ci_smoke.py \
  --data-dir {project}/bngsim/tests/data \
  --antimony-fixture-dir {project}/bngsim/benchmarks/models/antimony/ssys \
  --report {project}/bngsim/scripts/_reports/smoke-linux-x86_64-cp312.json
PYTEST

echo "==> cibuildwheel: CIBW_BUILD=$CIBW_BUILD (host=$(uname -ms))"
CIBW_BUILD="$CIBW_BUILD" \
CIBW_SKIP="*-musllinux_* pp*" \
CIBW_ARCHS_LINUX="x86_64" \
CIBW_BEFORE_BUILD="pip install cmake ninja" \
CIBW_TEST_EXTRAS="pandas" \
CIBW_TEST_REQUIRES="pytest>=8.0 numpy>=1.24 libroadrunner>=2.5" \
CIBW_TEST_COMMAND="$CIBW_TEST_COMMAND" \
  "$VENV/bin/python" -m cibuildwheel \
    --platform linux \
    --output-dir "$WHEELHOUSE" \
    "$REPO_ROOT/bngsim"

echo "==> built wheel(s):"
ls -la "$WHEELHOUSE"/*manylinux*.whl 2>/dev/null || true

# Render a Markdown matrix-style report from the JSON the in-container smoke
# wrote out (the JSON path is shared between host and container via the
# cibuildwheel mount).
REPORT_JSON="$REPO_ROOT/bngsim/scripts/_reports/smoke-linux-x86_64-cp312.json"
REPORT_MD="$REPO_ROOT/bngsim/scripts/local_ci_report-linux-x86_64-cp312.md"
if [[ -f "$REPORT_JSON" ]]; then
  WHEEL=$(ls -t "$WHEELHOUSE"/*cp312-cp312-manylinux*_x86_64.whl 2>/dev/null | head -1)
  "$VENV/bin/python" - <<PY
import json, pathlib, platform
data = json.loads(pathlib.Path("$REPORT_JSON").read_text())
out = pathlib.Path("$REPORT_MD")
lines = [
    "# local_ci report: linux-x86_64 / Python 3.12 (manylinux2014 via Docker)",
    "",
    "- bngsim source: \`$REPO_ROOT/bngsim\`",
    f"- host: \`{platform.platform()}\`",
    "- container: \`quay.io/pypa/manylinux2014_x86_64\`",
    "- python (target): \`3.12\`",
    "- wheel: \`$(basename "$WHEEL")\`",
    "- build: **PASS**",
    "",
    "## smoke checks",
    "",
    "| check | result | detail |",
    "|---|---|---|",
]
for k, v in data.items():
    lines.append(f"| {k} | {'PASS' if v['ok'] else 'FAIL'} | {v.get('detail', '')} |")
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
PY
fi

echo "==> done; see bngsim/scripts/local_ci_report-linux-x86_64-cp312.md"
