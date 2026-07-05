<#
    Build a pinned SuiteSparse KLU subset FROM SOURCE (shared DLLs) with MSVC and
    install it to a prefix, for bundling into bngsim wheels via delvewheel.

    Why from source on Windows: vcpkg's `suitesparse` surfaces only klu +
    suitesparseconfig, so SUNDIALS' KLU TPL fails with
    AMD/COLAMD/BTF_LIBRARY-NOTFOUND. A source build of the full KLU subset
    (suitesparse_config;amd;colamd;btf;klu) produces the amd/colamd/btf import
    libs + DLLs that SUNDIALS' find_library needs.

    SHARED (not static): KLU/BTF are LGPL-2.1+; the wheel bundles the DLLs
    (delvewheel) rather than static-linking.

    Usage: build_suitesparse.ps1 [-Prefix C:/suitesparse]
#>
param([string]$Prefix = "C:/suitesparse")

$ErrorActionPreference = "Stop"

# Pinned SuiteSparse release (tag + commit; asserted after clone).
$SsVersion = "v7.8.3"
$SsCommit  = "d3c4926d2c47fd6ae558e898bfc072ade210a2a1"

Write-Host "==> Building SuiteSparse $SsVersion ($SsCommit) -> $Prefix"
cmake --version | Select-Object -First 1

# --- fetch (shallow clone of the pinned tag, verify the commit) --------------
$work = Join-Path $env:RUNNER_TEMP "suitesparse-src"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
git clone --depth 1 --branch $SsVersion `
    https://github.com/DrTimothyAldenDavis/SuiteSparse.git $work
if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
$got = (git -C $work rev-parse HEAD).Trim()
if ($got -ne $SsCommit) {
    throw "SuiteSparse $SsVersion resolved to $got, expected $SsCommit"
}

# --- configure + build + install (KLU subset, shared, no OpenMP/CUDA) --------
# Default generator = latest Visual Studio (multi-config) -> pass --config on
# build/install. cmake locates MSVC itself; no vcvars pre-sourcing needed.
cmake -B "$work/build" -S "$work" `
    -DCMAKE_INSTALL_PREFIX="$Prefix" `
    -DBUILD_SHARED_LIBS=ON `
    -DBUILD_STATIC_LIBS=OFF `
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;colamd;btf;klu" `
    -DKLU_USE_CHOLMOD=OFF `
    -DSUITESPARSE_USE_OPENMP=OFF `
    -DSUITESPARSE_USE_CUDA=OFF `
    -DSUITESPARSE_DEMOS=OFF
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build "$work/build" --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

cmake --install "$work/build" --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake install failed" }

Write-Host "==> Installed SuiteSparse to ${Prefix}:"
Get-ChildItem "$Prefix/bin" -ErrorAction SilentlyContinue | Select-Object Name
Get-ChildItem "$Prefix/lib" -ErrorAction SilentlyContinue | Select-Object Name
