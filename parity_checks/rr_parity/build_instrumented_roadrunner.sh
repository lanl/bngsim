#!/usr/bin/env bash
# Build the instrumented RoadRunner (wshlavacek/roadrunner @ timing-instrumentation)
# for arm64 macOS, per parity_checks/rr_parity/INSTRUMENTED_ROADRUNNER.md, and
# shadow the stock wheel in the bngsim venv so getLoadTimings()/__warmup_sec__
# light up the rr_parity matrix's per-model RoadRunner load timing.
#
# 3 stages (each skipped if its install marker already exists):
#   1. LLVM 13 prebuilt (arm64) → $LLVM_PREFIX
#   2. libroadrunner-deps (12 submodules, non-recursive) → $DEPS_PREFIX
#   3. roadrunner (BUILD_PYTHON) → $RR_INSTALL ; then shadow into the venv
set -uo pipefail

ROOT="${RR_BUILD_ROOT:-$HOME/rr-instrumented-build}"
VENV="${BNGSIM_VENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.venv}"
SWIG=/opt/homebrew/bin/swig
export CC=/usr/bin/clang CXX=/usr/bin/clang++
POL=-DCMAKE_POLICY_VERSION_MINIMUM=3.5
NJ=$(sysctl -n hw.ncpu)

LLVM_PREFIX=$ROOT/llvm-13.x-macos-14-arm64-Release
DEPS_SRC=$ROOT/libroadrunner-deps
DEPS_PREFIX=$ROOT/deps-install
RR_SRC=$ROOT/roadrunner
RR_INSTALL=$ROOT/rr-install

mkdir -p "$ROOT"
echo "=== build root: $ROOT  ($(date)) ===  jobs=$NJ"

# ── Stage 1: LLVM 13 prebuilt (arm64) ────────────────────────────────────────
# The sys-bio zip has NO top-level folder — it contains bin/ include/ lib/
# directly — so extract into a dedicated $LLVM_PREFIX dir.
if [ ! -d "$LLVM_PREFIX/lib/cmake" ]; then
  echo "=== [1/3] downloading LLVM 13 arm64 prebuilt ==="
  cd "$ROOT"
  gh release download --repo sys-bio/llvm-13.x --pattern 'llvm-13.x-macos-14-arm64-Release.zip' --clobber || {
    echo "LLVM download failed"; exit 11; }
  rm -rf "$LLVM_PREFIX"; mkdir -p "$LLVM_PREFIX"
  unzip -q llvm-13.x-macos-14-arm64-Release.zip -d "$LLVM_PREFIX"
  [ -d "$LLVM_PREFIX/lib/cmake" ] || { echo "LLVM extract layout unexpected"; ls "$LLVM_PREFIX"; exit 12; }
  # The zip stores binaries without the execute bit; roadrunner's FindLLVM must
  # run llvm-config, so restore +x on the whole bin/ tree.
  chmod -R +x "$LLVM_PREFIX/bin" 2>/dev/null || true
  echo "LLVM_PREFIX=$LLVM_PREFIX"
else
  echo "=== [1/3] LLVM present at $LLVM_PREFIX — skip ==="
fi

# ── Stage 2: libroadrunner-deps ──────────────────────────────────────────────
if [ ! -d "$DEPS_PREFIX/cmake" ] && [ ! -d "$DEPS_PREFIX/lib" ]; then
  echo "=== [2/3] libroadrunner-deps ==="
  [ -d "$DEPS_SRC" ] || git clone https://github.com/sys-bio/libroadrunner-deps "$DEPS_SRC"
  cd "$DEPS_SRC"
  # Submodules live under third_party/; init the 12 needed ones NON-recursively,
  # deliberately EXCLUDING third_party/llvm-13.x (the multi-GB monorepo — we use
  # the prebuilt LLVM from stage 1, BUILD_LLVM=OFF).
  git submodule update --init \
      third_party/zlib third_party/nleq1 third_party/nleq2 third_party/bzip2 \
      third_party/poco third_party/expat third_party/libsbml third_party/rr-libstruct \
      third_party/sundials third_party/NuML third_party/libSEDML third_party/SBMLNetwork || {
    echo "submodule init failed"; exit 21; }
  cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$DEPS_PREFIX" -DCMAKE_BUILD_TYPE=Release $POL || { echo "deps configure failed"; exit 22; }
  cmake --build build --target install -j"$NJ" || { echo "deps build failed"; exit 23; }
else
  echo "=== [2/3] deps present at $DEPS_PREFIX — skip ==="
fi

# ── Stage 3: roadrunner (instrumented fork) ──────────────────────────────────
if [ ! -d "$RR_INSTALL" ]; then
  echo "=== [3/3] roadrunner (timing-instrumentation) ==="
  [ -d "$RR_SRC" ] || git clone -b timing-instrumentation https://github.com/wshlavacek/roadrunner "$RR_SRC"
  cd "$RR_SRC"
  # roadrunner's CMakeLists FORCE-sets CMAKE_OSX_ARCHITECTURES=x86_64 ("compile
  # for intel chips") but guards on `NOT CMAKE_OSX_ARCHITECTURES`, so passing it
  # on the command line wins. Without this it cross-targets x86_64 and fails
  # linking the arm64 deps (expat/etc.). Wipe the stale x86_64 cache first.
  rm -rf build
  cmake -S . -B build -DBUILD_PYTHON=ON \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DLLVM_INSTALL_PREFIX="$LLVM_PREFIX" \
      -DRR_DEPENDENCIES_INSTALL_PREFIX="$DEPS_PREFIX" \
      -DPython_ROOT_DIR="$VENV" \
      -DSWIG_EXECUTABLE="$SWIG" \
      -DCMAKE_INSTALL_PREFIX="$RR_INSTALL" \
      -DCMAKE_BUILD_TYPE=Release $POL || { echo "rr configure failed"; exit 31; }
  cmake --build build --target install -j"$NJ" || { echo "rr build failed"; exit 32; }
else
  echo "=== [3/3] roadrunner installed at $RR_INSTALL — skip ==="
fi

# ── Shadow the stock wheel in the venv ───────────────────────────────────────
BUILT=$(find "$RR_INSTALL" -maxdepth 3 -type d -name roadrunner -path '*site-packages*' | head -1)
[ -z "$BUILT" ] && BUILT=$(find "$RR_SRC/build" -maxdepth 4 -type d -name roadrunner -path '*site-packages*' | head -1)
SITE="$VENV/lib/python3.12/site-packages"
echo "=== shadow: built package = $BUILT ==="
if [ -n "$BUILT" ] && [ -d "$BUILT" ]; then
  [ -d "$SITE/roadrunner.stock-backup" ] || cp -R "$SITE/roadrunner" "$SITE/roadrunner.stock-backup"
  rm -rf "$SITE/roadrunner"
  cp -R "$BUILT" "$SITE/roadrunner"
  echo "shadowed stock roadrunner (backup at roadrunner.stock-backup)"
else
  echo "WARNING: could not locate built roadrunner package; shadow skipped"; exit 41
fi

echo "=== verify ==="
"$VENV/bin/python" - <<'PY'
import roadrunner
print("version:", roadrunner.__version__)
print("has getLoadTimings:", hasattr(roadrunner.RoadRunner, "getLoadTimings"))
print("has __warmup_sec__:", hasattr(roadrunner, "__warmup_sec__"))
PY
echo "=== DONE ($(date)) ==="
