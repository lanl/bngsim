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

# --- provide a BLAS for SuiteSparse_config's REQUIRED find_package(BLAS) ------
# SuiteSparseBLAS.cmake ends in an unconditional `find_package(BLAS REQUIRED)`
# with no opt-out; macOS satisfies it via the Accelerate framework, but Windows
# has no system BLAS. None of the 5 libraries we build (config/amd/colamd/btf/klu)
# LINK a BLAS -- suitesparse_config only DETECTS one to record its integer size --
# so this OpenBLAS is used at configure time ONLY and is neither linked into nor
# bundled with the wheel (verified: no output lib references it).
$OpenBlasVer = "0.3.33"
$blasRoot = Join-Path $env:RUNNER_TEMP "openblas"
if (-not (Get-ChildItem -Path $blasRoot -Recurse -Filter "libopenblas.lib" -ErrorAction SilentlyContinue)) {
    $zip = Join-Path $env:RUNNER_TEMP "openblas.zip"
    $url = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$OpenBlasVer/OpenBLAS-$OpenBlasVer-x64.zip"
    Write-Host "==> Fetching OpenBLAS $OpenBlasVer (configure-only BLAS; not bundled)"
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $blasRoot -Force
}
$blasLib = Get-ChildItem -Path $blasRoot -Recurse -Filter "libopenblas.lib" | Select-Object -First 1
if (-not $blasLib) { throw "OpenBLAS import lib not found under $blasRoot" }
# <prefix>/lib/libopenblas.lib -> <prefix>, so find_package(BLAS) resolves it
# from CMAKE_PREFIX_PATH regardless of the zip's internal layout.
$blasPrefix = Split-Path (Split-Path $blasLib.FullName -Parent) -Parent
Write-Host "==> OpenBLAS import lib: $($blasLib.FullName) (prefix $blasPrefix)"

# --- configure + build + install (KLU subset, shared, no OpenMP/CUDA) --------
# Default generator = latest Visual Studio (multi-config) -> pass --config on
# build/install. cmake locates MSVC itself; no vcvars pre-sourcing needed.
cmake -B "$work/build" -S "$work" `
    -DCMAKE_INSTALL_PREFIX="$Prefix" `
    -DCMAKE_PREFIX_PATH="$blasPrefix" `
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
