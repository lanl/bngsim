#!/usr/bin/env bash
#
# Build a pinned SuiteSparse KLU subset FROM SOURCE (shared libraries) and
# install it to a prefix, for bundling into bngsim wheels.
#
# Why from source (vs brew/EPEL/vcpkg): package-manager SuiteSparse targets the
# CI runner's OS, so on macOS `delocate` refuses to bundle a runner-OS dylib into
# a lower-tagged wheel (narrow wheels). Building here at a LOW
# CMAKE_OSX_DEPLOYMENT_TARGET yields low-min-target dylibs -> broad-compat wheels.
#
# Subset = the exact set SUNDIALS' KLU TPL needs: suitesparse_config, amd,
# colamd, btf, klu. No CHOLMOD/UMFPACK -> no BLAS dependency. SHARED (not static):
# KLU and BTF are LGPL-2.1+, so the wheel bundles the shared libs (auditwheel /
# delocate / delvewheel) rather than static-linking (which triggers LGPL relink
# obligations).
#
# SUITESPARSE_USE_FORTRAN=OFF: AMD ships an optional Fortran interface KLU never
# calls. When a Fortran compiler is on PATH (e.g. a dev box with Homebrew gcc's
# gfortran) SuiteSparse builds it, linking libamd against Homebrew's
# libgfortran/libgcc_s/libquadmath (min target 11.0/15.0) -> delocate then refuses
# the low-tagged wheel. The bare CI runners have no gfortran so they never hit it;
# disabling Fortran makes the build reproducible regardless of host toolchain.
#
# Usage: build_suitesparse.sh <install-prefix>
#   Reads MACOSX_DEPLOYMENT_TARGET from the env (set by the workflow matrix) so
#   the SuiteSparse dylibs' min target matches the wheel's; falls back to an
#   arch-derived default when unset.
set -euo pipefail

PREFIX="${1:?usage: build_suitesparse.sh <install-prefix>}"

# Pinned SuiteSparse release (tag + commit; the commit is asserted after clone
# so a moved tag can't silently change what we build).
SS_VERSION="v7.8.3"
SS_COMMIT="d3c4926d2c47fd6ae558e898bfc072ade210a2a1"

echo "==> Building SuiteSparse ${SS_VERSION} (${SS_COMMIT}) -> ${PREFIX}"

# --- ensure a recent enough CMake (SuiteSparse 7.x needs >= 3.22) -------------
if ! command -v cmake >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
        brew install cmake
    else
        python3 -m pip install --upgrade "cmake>=3.22" ninja
    fi
fi
cmake --version | head -1

# --- CMake args (single array; kept non-empty so "${args[@]}" is safe under
#     `set -u` on macOS's stock bash 3.2, where expanding an empty array errors) -
args=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_STATIC_LIBS=OFF
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;colamd;btf;klu"
    -DKLU_USE_CHOLMOD=OFF
    -DSUITESPARSE_USE_FORTRAN=OFF
    -DSUITESPARSE_USE_OPENMP=OFF
    -DSUITESPARSE_USE_CUDA=OFF
    -DSUITESPARSE_DEMOS=OFF
)

if [[ "$(uname -s)" == "Darwin" ]]; then
    arch="$(uname -m)"
    # Prefer the wheel's target so SuiteSparse dylibs and the extension agree;
    # otherwise pick each arch's broad floor (arm64 macOS starts at 11.0; on
    # x86_64, C++17 std::filesystem etc. floor at 10.15).
    if [[ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
        target="${MACOSX_DEPLOYMENT_TARGET}"
    elif [[ "${arch}" == "arm64" ]]; then
        target="11.0"
    else
        target="10.15"
    fi
    # Install names: absolute ${PREFIX}/lib by default (like Homebrew's), so
    # delocate resolves the dylibs by path at wheel-repair time. The CMake sdist
    # fallback (no delocate) sets SS_INSTALL_NAME_DIR=@rpath instead so the
    # dylibs are relocatable via the extension's rpath.
    args+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${target}"
           "-DCMAKE_OSX_ARCHITECTURES=${arch}"
           "-DCMAKE_INSTALL_NAME_DIR=${SS_INSTALL_NAME_DIR:-${PREFIX}/lib}")
    echo "==> macOS ${arch}, deployment target ${target}"
fi

# Optional install RPATH baked into the dylibs so they locate each other once
# copied beside the extension (the sdist fallback sets SS_INSTALL_RPATH to
# @loader_path on macOS / $ORIGIN on Linux). Unset by default → no change to the
# wheel path, whose dylibs are resolved by absolute name / auditwheel.
if [[ -n "${SS_INSTALL_RPATH:-}" ]]; then
    args+=("-DCMAKE_INSTALL_RPATH=${SS_INSTALL_RPATH}"
           "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON")
fi

# --- fetch (shallow clone of the pinned tag, verify the commit) ---------------
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
git clone --depth 1 --branch "${SS_VERSION}" \
    https://github.com/DrTimothyAldenDavis/SuiteSparse.git "${work}/SuiteSparse"
got="$(git -C "${work}/SuiteSparse" rev-parse HEAD)"
if [[ "${got}" != "${SS_COMMIT}" ]]; then
    echo "ERROR: SuiteSparse ${SS_VERSION} resolved to ${got}, expected ${SS_COMMIT}" >&2
    exit 1
fi

# --- configure + build + install (KLU subset, shared, no OpenMP/CUDA) ---------
cmake -B "${work}/build" -S "${work}/SuiteSparse" "${args[@]}"

cmake --build "${work}/build" --config Release -j
cmake --install "${work}/build" --config Release

echo "==> Installed SuiteSparse to ${PREFIX}:"
ls -la "${PREFIX}/lib" 2>/dev/null || ls -la "${PREFIX}/lib64" 2>/dev/null || true
ls -la "${PREFIX}/include" 2>/dev/null || true
