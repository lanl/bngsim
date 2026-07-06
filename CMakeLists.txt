cmake_minimum_required(VERSION 3.20)

# Version is sourced exclusively from pyproject.toml (issue #31).
# Parse the literal here so CMake, the C++ extension, and any
# downstream CMake consumer all see the same string without manual
# duplication.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _bngsim_pyproject)
string(REGEX MATCH "\nversion[ \t]*=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
       _bngsim_version_match "${_bngsim_pyproject}")
if(NOT _bngsim_version_match)
    message(FATAL_ERROR
        "Could not extract version from pyproject.toml. "
        "Expected a top-level line of the form: version = \"X.Y.Z\"")
endif()
set(BNGSIM_VERSION "${CMAKE_MATCH_1}")
unset(_bngsim_pyproject)
unset(_bngsim_version_match)

project(bngsim VERSION ${BNGSIM_VERSION} LANGUAGES CXX C)

# Build-provenance stamp (issue #125). Capture the source commit at configure
# time so the compiled extension can answer "which source produced me?". This
# is the string half of the stale-binary guard: a separately-built _bngsim_core
# (editable.rebuild=false, #23) can silently lag the source tree and drive false
# correctness verdicts through the bespoke oracles. The Python-side guard in
# bngsim/_build_provenance.py uses mtime as the *verdict*; this commit string is
# the human-readable identity it prints. Best-effort: a build from an exported
# tarball with no .git falls back to "unknown", and a dirty tree is flagged.
set(BNGSIM_BUILD_COMMIT "unknown")
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                rev-parse --short=12 HEAD
        OUTPUT_VARIABLE _bngsim_git_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _bngsim_git_rc
        ERROR_QUIET
    )
    if(_bngsim_git_rc EQUAL 0 AND _bngsim_git_commit)
        set(BNGSIM_BUILD_COMMIT "${_bngsim_git_commit}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                    status --porcelain --untracked-files=no
            OUTPUT_VARIABLE _bngsim_git_dirty
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_bngsim_git_dirty)
            set(BNGSIM_BUILD_COMMIT "${BNGSIM_BUILD_COMMIT}+dirty")
        endif()
        unset(_bngsim_git_dirty)
    endif()
    unset(_bngsim_git_commit)
    unset(_bngsim_git_rc)
endif()
message(STATUS "BNGsim: build commit = ${BNGSIM_BUILD_COMMIT}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build shared library by default
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BNGSIM_BUILD_TESTS "Build unit tests" ON)
option(BNGSIM_BUILD_PYTHON "Build pybind11 Python bindings" OFF)
option(BNGSIM_ENABLE_KLU "Enable SuiteSparse/KLU sparse Jacobian support when available" ON)
# Opt-in hard failure (GH #209). When ON, a configure that cannot find/verify
# SuiteSparse/KLU is a FATAL_ERROR instead of a silent dense-only fallback —
# so an HPC/CI deployment can never ship a dense-only build unnoticed.
option(BNGSIM_REQUIRE_KLU "Fail the build if SuiteSparse/KLU is not found/usable" OFF)
# Gold-standard self-sufficiency (GH #209): when no system SuiteSparse/KLU is
# found, build a pinned KLU subset from source and bundle it beside the
# extension, so an sdist `pip install` on a box with no SuiteSparse still gets
# the sparse solver instead of failing (REQUIRE_KLU) or degrading to dense.
option(BNGSIM_KLU_AUTOBUILD "Build+bundle a pinned SuiteSparse/KLU subset from source when none is found" ON)

# When building Python bindings, build everything as static libraries so the
# extension module (.so/.dylib) is fully self-contained.
if(BNGSIM_BUILD_PYTHON)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# ─── SUNDIALS v7.x ─────────────────────────────────────────────────────────────
# Two modes:
#   1. Managed fetch (default) — FetchContent from the pinned official release
#      archive recorded in third_party/sundials/VENDOR.json
#   2. System SUNDIALS         — uses find_package() for environment-managed
#      builds such as conda-forge
#
# Refresh the pinned managed-fetch metadata only through
# bngsim/scripts/vendor_sundials.py. See bngsim/scripts/SUNDIALS_VENDORING.md.
# Set -DBNGSIM_USE_SYSTEM_SUNDIALS=ON to use a pre-installed SUNDIALS (≥7.0).

option(BNGSIM_USE_SYSTEM_SUNDIALS "Use system-installed SUNDIALS instead of FetchContent" OFF)
set(BNGSIM_SUNDIALS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sundials")
set(BNGSIM_SUNDIALS_VENDOR_METADATA "${BNGSIM_SUNDIALS_DIR}/VENDOR.json")

set(BNGSIM_LINK_KLU OFF)
set(BNGSIM_KLU_DISABLE_REASON "explicitly disabled")

# ─── Portable SuiteSparse/KLU discovery (GH #209) ─────────────────────────────
# Replaces the old hardcoded `foreach(_prefix /opt/homebrew /usr/local /usr)`
# probe, which ignored CMAKE_PREFIX_PATH/$CONDA_PREFIX/module trees and so left
# every Linux/HPC build silently dense-only (O(N^3) on a sparse Jacobian).
#
# find_path/find_library resolve, in precedence order:
#   1. an explicit -DKLU_INCLUDE_DIR / -DKLU_LIBRARY (already-set cache → no-op),
#   2. -DKLU_LIBRARY_DIR and the *_ROOT / $CONDA_PREFIX env hints,
#   3. CMAKE_PREFIX_PATH (honored natively by find_*),
#   4. the historical system prefixes (brew, vanilla Linux /usr).
# Nothing is FORCE-stomped, so a user-passed prefix always wins. SUNDIALS' KLU
# TPL consumes KLU_INCLUDE_DIR + KLU_LIBRARY_DIR, so we derive the library dir
# from the resolved libklu. Sets BNGSIM_KLU_FOUND ON/OFF.
macro(_bngsim_discover_klu)
    find_path(KLU_INCLUDE_DIR
        NAMES klu.h
        HINTS
            ${KLU_ROOT} ${SUITESPARSE_ROOT}
            $ENV{KLU_ROOT} $ENV{SUITESPARSE_ROOT} $ENV{CONDA_PREFIX}
        PATH_SUFFIXES include include/suitesparse suitesparse
        PATHS /opt/homebrew /usr/local /usr
    )
    find_library(KLU_LIBRARY
        NAMES klu
        HINTS
            ${KLU_LIBRARY_DIR} ${KLU_ROOT} ${SUITESPARSE_ROOT}
            $ENV{KLU_ROOT} $ENV{SUITESPARSE_ROOT} $ENV{CONDA_PREFIX}
        PATH_SUFFIXES lib lib64
        PATHS /opt/homebrew /usr/local /usr
    )
    if(KLU_INCLUDE_DIR AND KLU_LIBRARY)
        # Respect an explicit -DKLU_LIBRARY_DIR; otherwise derive it from the
        # resolved library so the SUNDIALS KLU TPL can locate libklu + friends
        # (amd, colamd, btf, suitesparseconfig live alongside it).
        if(NOT KLU_LIBRARY_DIR)
            get_filename_component(_bngsim_klu_libdir "${KLU_LIBRARY}" DIRECTORY)
            set(KLU_LIBRARY_DIR "${_bngsim_klu_libdir}"
                CACHE PATH "SuiteSparse/KLU library directory")
            unset(_bngsim_klu_libdir)
        endif()
        set(BNGSIM_KLU_FOUND ON)
    else()
        set(BNGSIM_KLU_FOUND OFF)
    endif()
endmacro()

# Shared actionable failure for -DBNGSIM_REQUIRE_KLU=ON (GH #209). Called from
# whichever SUNDIALS branch finds KLU unavailable, so a required-KLU build fails
# fast with a fix-it message instead of degrading to the dense solver.
macro(_bngsim_require_klu_or_die _reason)
    if(BNGSIM_REQUIRE_KLU)
        message(FATAL_ERROR
            "SuiteSparse/KLU is required (-DBNGSIM_REQUIRE_KLU=ON) but was not "
            "found/usable: ${_reason}.\n"
            "Large/genome-scale models route the CVODE Newton solve to the sparse "
            "KLU direct solver; without it the dense fallback factorizes the full "
            "N x N Jacobian at O(N^3) (e.g. ~45 GB / hours for a 75k-species "
            "network). Install SuiteSparse, then re-configure:\n"
            "  - macOS:         brew install suite-sparse\n"
            "  - Debian/Ubuntu: apt-get install libsuitesparse-dev\n"
            "  - RHEL/Fedora:   dnf install suitesparse-devel\n"
            "  - conda:         conda install -c conda-forge suitesparse\n"
            "  - HPC module:    module load suitesparse\n"
            "If the prefix is non-standard, point CMake at it (any one of):\n"
            "  -DCMAKE_PREFIX_PATH=<prefix>   (or export CONDA_PREFIX=<prefix>)\n"
            "  -DKLU_ROOT=<prefix>  /  -DSUITESPARSE_ROOT=<prefix>\n"
            "  -DKLU_INCLUDE_DIR=<dir>  -DKLU_LIBRARY_DIR=<dir>\n"
            "Or drop the requirement: -DBNGSIM_REQUIRE_KLU=OFF (default), or "
            "disable sparse support entirely with -DBNGSIM_ENABLE_KLU=OFF."
        )
    endif()
endmacro()

# Gold-standard fallback (GH #209): build the pinned SuiteSparse/KLU subset from
# source at configure time (ci/build_suitesparse.sh) and bundle the shared libs
# next to the extension, so `pip install` from an sdist gets KLU even with no
# system SuiteSparse — instead of failing (REQUIRE_KLU) or silently degrading to
# the dense O(N^3) solver. Reuses the same recipe as the wheels, but tells it to
# emit relocatable install names (@rpath/$ORIGIN) since no delocate/auditwheel
# runs on an sdist install; the extension gets a matching rpath below.
#
# Gated to non-Windows (Windows ships a wheel; its from-source build needs the
# MSVC/OpenBLAS handling in ci/build_suitesparse.ps1). On any failure — e.g.
# Linux with no BLAS for SuiteSparse_config's find_package(BLAS REQUIRED) — this
# leaves BNGSIM_KLU_FOUND OFF so the caller falls through to require-or-die.
macro(_bngsim_autobuild_klu)
    if(WIN32)
        message(STATUS
            "BNGsim: SuiteSparse/KLU auto-build is not supported on Windows — "
            "install the published wheel or SuiteSparse and re-configure.")
    else()
        find_program(BNGSIM_BASH bash)
        find_program(BNGSIM_GIT git)
        if(NOT BNGSIM_BASH OR NOT BNGSIM_GIT)
            message(STATUS
                "BNGsim: SuiteSparse/KLU auto-build needs both bash and git on "
                "PATH; not found — skipping the from-source fallback.")
        else()
            set(_bngsim_klu_prefix "${CMAKE_BINARY_DIR}/_bngsim_klu_autobuild")
            set(_bngsim_ss_script "${CMAKE_CURRENT_SOURCE_DIR}/ci/build_suitesparse.sh")
            # Relocatable install: @rpath install-names plus an rpath so the KLU
            # dylibs find each other once copied beside the extension.
            set(_bngsim_ss_env "")
            if(APPLE)
                list(APPEND _bngsim_ss_env
                    "SS_INSTALL_NAME_DIR=@rpath" "SS_INSTALL_RPATH=@loader_path")
                if(CMAKE_OSX_DEPLOYMENT_TARGET)
                    list(APPEND _bngsim_ss_env
                        "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
                endif()
            else()
                list(APPEND _bngsim_ss_env "SS_INSTALL_RPATH=$ORIGIN")
            endif()
            message(STATUS
                "BNGsim: SuiteSparse/KLU not found — building the pinned KLU "
                "subset from source into ${_bngsim_klu_prefix} (one-time, ~1-2 min) …")
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E env ${_bngsim_ss_env}
                        ${BNGSIM_BASH} ${_bngsim_ss_script} ${_bngsim_klu_prefix}
                RESULT_VARIABLE _bngsim_ss_rc)
            if(NOT _bngsim_ss_rc EQUAL 0)
                message(STATUS
                    "BNGsim: SuiteSparse/KLU auto-build FAILED (exit ${_bngsim_ss_rc}); "
                    "see the output above. Falling back to system discovery result.")
            else()
                # Re-discover against the freshly built prefix. Clear any cached
                # NOTFOUND from the first probe so find_* actually re-searches.
                unset(KLU_INCLUDE_DIR CACHE)
                unset(KLU_LIBRARY CACHE)
                unset(KLU_LIBRARY_DIR CACHE)
                set(KLU_ROOT "${_bngsim_klu_prefix}")
                _bngsim_discover_klu()
                if(BNGSIM_KLU_FOUND)
                    # Bundle the built dylibs beside the extension (install rule +
                    # extension rpath added after the target is defined).
                    set(BNGSIM_KLU_AUTOBUILT ON)
                    set(BNGSIM_KLU_BUNDLE_DIR "${KLU_LIBRARY_DIR}")
                    message(STATUS
                        "BNGsim: built + bundling SuiteSparse/KLU from source — "
                        "lib=${KLU_LIBRARY_DIR}")
                else()
                    message(STATUS
                        "BNGsim: SuiteSparse/KLU auto-build installed but KLU is "
                        "still not discoverable under ${_bngsim_klu_prefix}.")
                endif()
            endif()
        endif()
    endif()
endmacro()

# A required-KLU build with sparse support explicitly disabled is contradictory.
if(BNGSIM_REQUIRE_KLU AND NOT BNGSIM_ENABLE_KLU)
    message(FATAL_ERROR
        "-DBNGSIM_REQUIRE_KLU=ON contradicts -DBNGSIM_ENABLE_KLU=OFF: cannot "
        "require KLU while sparse support is disabled. Enable KLU "
        "(-DBNGSIM_ENABLE_KLU=ON, the default) or drop -DBNGSIM_REQUIRE_KLU.")
endif()

if(BNGSIM_USE_SYSTEM_SUNDIALS)
    # Request the KLU linear-solver components as OPTIONAL so the configure does
    # not hard-fail on a system SUNDIALS built without KLU; whether they actually
    # arrived is verified by target existence below (GH #209 Part 5) rather than
    # assumed from BNGSIM_ENABLE_KLU.
    find_package(SUNDIALS 7.0 REQUIRED
        COMPONENTS
            cvodes
            kinsol
            nvecserial
            sunmatrixdense
            sunlinsoldense
        OPTIONAL_COMPONENTS
            sunmatrixsparse
            sunlinsolklu
    )
    message(STATUS "Using system SUNDIALS: ${SUNDIALS_DIR}")

    # Verify the system SUNDIALS really provides KLU instead of assuming it: a
    # conda-forge/system SUNDIALS can be built without the KLU solver even when
    # SuiteSparse headers are installed on the box.
    if(BNGSIM_ENABLE_KLU)
        if(TARGET SUNDIALS::sunlinsolklu AND TARGET SUNDIALS::sunmatrixsparse)
            set(BNGSIM_LINK_KLU ON)
        else()
            set(BNGSIM_KLU_DISABLE_REASON
                "system SUNDIALS was built without the KLU linear solver")
            _bngsim_require_klu_or_die("${BNGSIM_KLU_DISABLE_REASON}")
        endif()
    endif()
else()
    include(FetchContent)

    if(NOT EXISTS "${BNGSIM_SUNDIALS_VENDOR_METADATA}")
        message(FATAL_ERROR
            "Missing SUNDIALS vendoring metadata: ${BNGSIM_SUNDIALS_VENDOR_METADATA}\n"
            "Refresh the pinned fetch metadata explicitly via\n"
            "  python3 bngsim/scripts/vendor_sundials.py --archive /tmp/sundials-vendor-candidate/<release>.tar.gz\n"
            "See bngsim/scripts/SUNDIALS_VENDORING.md for the supported workflow."
        )
    endif()

    file(READ "${BNGSIM_SUNDIALS_VENDOR_METADATA}" _bngsim_sundials_vendor_json)

    function(_bngsim_sundials_json_get out_var)
        string(JSON _value ERROR_VARIABLE _error GET "${_bngsim_sundials_vendor_json}" ${ARGN})
        if(_error)
            list(JOIN ARGN "." _path)
            message(FATAL_ERROR
                "Malformed SUNDIALS vendoring metadata in ${BNGSIM_SUNDIALS_VENDOR_METADATA}: "
                "could not read ${_path}: ${_error}"
            )
        endif()
        set(${out_var} "${_value}" PARENT_SCOPE)
    endfunction()

    _bngsim_sundials_json_get(BNGSIM_SUNDIALS_RELEASE_TAG source authoritative_release_tag)
    _bngsim_sundials_json_get(BNGSIM_SUNDIALS_SOURCE_URL source authoritative_release_asset_url)
    _bngsim_sundials_json_get(BNGSIM_SUNDIALS_SOURCE_SHA256 files source_archive sha256)
    _bngsim_sundials_json_get(BNGSIM_SUNDIALS_TAG_COMMIT source tag_commit)

    message(STATUS
        "Using pinned SUNDIALS release ${BNGSIM_SUNDIALS_RELEASE_TAG} "
        "(${BNGSIM_SUNDIALS_TAG_COMMIT})"
    )
    message(STATUS "SUNDIALS source archive: ${BNGSIM_SUNDIALS_SOURCE_URL}")

    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_CVODE OFF CACHE BOOL "" FORCE)   # Superseded by CVODES
    set(BUILD_ARKODE OFF CACHE BOOL "" FORCE)
    set(BUILD_IDA OFF CACHE BOOL "" FORCE)
    set(BUILD_IDAS OFF CACHE BOOL "" FORCE)
    set(BUILD_KINSOL ON CACHE BOOL "" FORCE)    # Required for steady-state solver
    set(BUILD_CVODES ON CACHE BOOL "" FORCE)   # Required for forward sensitivities
    set(EXAMPLES_ENABLE_C OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "" FORCE)
    set(SUNDIALS_BUILD_WITH_MONITORING OFF CACHE BOOL "" FORCE)
    set(SUNDIALS_BUILD_WITH_PROFILING OFF CACHE BOOL "" FORCE)

    # SUNDIALS' nested POSIX timer probe can leave timers disabled on the
    # universal macOS Ninja build even when clock_gettime() is available.
    # Seed the cache so sundials_profiler.c stays buildable in our local build.
    if(APPLE)
        set(SUNDIALS_POSIX_TIMERS ON CACHE BOOL "" FORCE)
        set(POSIX_TIMERS_NEED_POSIX_C_SOURCE TRUE CACHE BOOL "" FORCE)
    endif()

    # ── KLU sparse direct solver (SuiteSparse) ──────────────────────────────
    # Enable KLU in the managed SUNDIALS for sparse Jacobian support. Discovery
    # is portable (GH #209): _bngsim_discover_klu honors CMAKE_PREFIX_PATH,
    # $CONDA_PREFIX, KLU_ROOT/SUITESPARSE_ROOT, an explicit
    # -DKLU_INCLUDE_DIR/-DKLU_LIBRARY_DIR, and the historical system prefixes —
    # so brew, vanilla Linux /usr, conda, and HPC module trees all resolve with
    # no CMake edits. SUNDIALS' KLU TPL reads KLU_INCLUDE_DIR + KLU_LIBRARY_DIR
    # (set by the macro from the resolved libklu); nothing is FORCE-stomped.
    if(BNGSIM_ENABLE_KLU)
        _bngsim_discover_klu()
        # No system SuiteSparse? Build+bundle the pinned KLU subset from source
        # (self-sufficient sdist installs) before giving up. Sets BNGSIM_KLU_FOUND
        # + BNGSIM_KLU_AUTOBUILT on success.
        if(NOT BNGSIM_KLU_FOUND AND BNGSIM_KLU_AUTOBUILD)
            _bngsim_autobuild_klu()
        endif()
        if(BNGSIM_KLU_FOUND)
            set(ENABLE_KLU ON CACHE BOOL "" FORCE)
            message(STATUS "BNGsim: found SuiteSparse/KLU — "
                           "include=${KLU_INCLUDE_DIR} lib=${KLU_LIBRARY_DIR}")
        else()
            # Fail fast (before downloading/building SUNDIALS) when KLU is
            # required; otherwise warn and continue dense-only.
            _bngsim_require_klu_or_die("not found on any search prefix")
            message(WARNING
                "BNGsim: SuiteSparse/KLU not found — sparse Jacobian support "
                "DISABLED, large models fall back to the dense O(N^3) solver. "
                "Install SuiteSparse (brew/apt/dnf/conda/module) and/or pass "
                "-DCMAKE_PREFIX_PATH / -DKLU_ROOT; pass -DBNGSIM_REQUIRE_KLU=ON "
                "to make this fatal. See GH #209.")
            set(ENABLE_KLU OFF CACHE BOOL "" FORCE)
            set(BNGSIM_KLU_DISABLE_REASON "KLU not found")
        endif()
    else()
        set(ENABLE_KLU OFF CACHE BOOL "" FORCE)
        set(BNGSIM_KLU_DISABLE_REASON "explicitly disabled")
        message(STATUS "BNGsim: sparse Jacobian support explicitly disabled")
    endif()

    FetchContent_Declare(
        sundials
        URL "${BNGSIM_SUNDIALS_SOURCE_URL}"
        URL_HASH "SHA256=${BNGSIM_SUNDIALS_SOURCE_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    FetchContent_MakeAvailable(sundials)

    if(ENABLE_KLU)
        set(BNGSIM_LINK_KLU ON)
    endif()

    unset(_bngsim_sundials_vendor_json)
endif()

# ─── ExprTk (vendored header-only) ────────────────────────────────────────────
# ExprTk is a single header file vendored in third_party/exprtk/.
# Builds must stay reproducible and must not silently mutate the source tree.
# Refreshes belong in bngsim/scripts/vendor_exprtk.py, which also updates the
# vendoring metadata and upstream guardrails alongside exprtk.hpp.
#
# This block, and the bngsim::expression target below, run BEFORE
# add_subdirectory(third_party/nfsim) and add_subdirectory(third_party/rulemonkey)
# on purpose: both vendored engines probe `if(TARGET bngsim::expression)` to
# reuse the host's ExprTk evaluator / BNG-compat layer instead of compiling a
# duplicate copy of exprtk.hpp + bngsim::ExprTkEvaluator. RuleMonkey links the
# evaluator directly (its evaluator IS bngsim::ExprTkEvaluator); NFsim's
# mu::Parser shim links bngsim::expression so it can forward to the host's
# single mratio() + reserved-symbol-remap source. See issues #39 and #49.
set(BNGSIM_EXPRTK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk")
set(BNGSIM_EXPRTK_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk/exprtk.hpp")
set(BNGSIM_EXPRTK_VENDOR_METADATA "${BNGSIM_EXPRTK_DIR}/VENDOR.json")

if(NOT EXISTS "${BNGSIM_EXPRTK_HEADER}")
    message(FATAL_ERROR
        "Missing vendored ExprTk header: ${BNGSIM_EXPRTK_HEADER}\n"
        "Restore third_party/exprtk from the repository, or refresh it explicitly via\n"
        "  python3 bngsim/scripts/vendor_exprtk.py --exprtk-repo /tmp/exprtk-vendor-candidate\n"
        "See bngsim/scripts/EXPRTK_VENDORING.md for the supported vendoring workflow."
    )
endif()

if(NOT EXISTS "${BNGSIM_EXPRTK_VENDOR_METADATA}")
    message(FATAL_ERROR
        "Missing ExprTk vendoring metadata: ${BNGSIM_EXPRTK_VENDOR_METADATA}\n"
        "Refresh third_party/exprtk through bngsim/scripts/vendor_exprtk.py so exprtk.hpp\n"
        "and VENDOR.json stay pinned together."
    )
endif()

# ─── bngsim::expression — ExprTk evaluator as a standalone target ─────────────
# src/expression.cpp (bngsim::ExprTkEvaluator) is built as its own static
# target rather than as a plain entry in BNGSIM_SOURCES, so the vendored
# RuleMonkey can detect and link it via `if(TARGET bngsim::expression)` —
# guaranteeing exactly one compiled copy of the bngsim:: expression symbols
# and one exprtk.hpp per binary (the alternative is a duplicate-symbol / ODR
# hazard across two static archives). Must be declared before
# add_subdirectory(third_party/rulemonkey). See issue #39.
add_library(bngsim_expression STATIC src/expression.cpp)
add_library(bngsim::expression ALIAS bngsim_expression)
target_include_directories(bngsim_expression
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk)
target_compile_features(bngsim_expression PUBLIC cxx_std_17)
set_target_properties(bngsim_expression PROPERTIES POSITION_INDEPENDENT_CODE ON)
# ExprTk-heavy translation units can exceed MSVC's default section limit.
target_compile_options(bngsim_expression PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

# ─── NFsim (vendored static library) ───────────────────────────────────────────
# Declared after bngsim::expression so the vendored NFsim object target can link
# it (below) and the mu::Parser ExprTk shim can #include <bngsim/expr_compat.hpp>
# to forward to the host's single mratio() + reserved-symbol-remap source. See
# issue #49.
option(BNGSIM_BUILD_NFSIM "Build vendored NFsim as a static library for in-process simulation" ON)

if(BNGSIM_BUILD_NFSIM)
    # NFsim source is vendored in third_party/nfsim/src/ — no external path needed
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nfsim/src/NFcore/NFcore.hh")
        message(STATUS "Building vendored NFsim from: third_party/nfsim/src/")
        # The vendored export intentionally omits muParser, so NFsim must use
        # the ExprTk-backed compatibility parser path on every configure.
        # Upstream NFsim now exposes BUILD_LIBRARY / BUILD_EXECUTABLE toggles;
        # force the embedded-library configuration here rather than carrying a
        # local fork of NFsim's top-level build logic.
        set(NFSIM_BUILD_LIBRARY ON CACHE BOOL "" FORCE)
        set(NFSIM_BUILD_EXECUTABLE OFF CACHE BOOL "" FORCE)
        set(NFSIM_USE_EXPRTK ON CACHE BOOL "" FORCE)
        add_subdirectory(third_party/nfsim)

        # The vendored NFsim mu::Parser ExprTk shim (third_party/nfsim/src/
        # NFfunction/nfsim_funcparser.h) forwards to the host's single source of
        # the BNG-compat primitives — mratio() and the reserved-symbol remap —
        # via #include <bngsim/expr_compat.hpp> instead of carrying a hand-ported
        # copy (issue #49). Link bngsim::expression into the vendored NFsim
        # targets so:
        #   * its PUBLIC include/ usage requirement reaches the NFsim object
        #     compiles (so the shim's #include resolves), and
        #   * the final link records the dependency and orders
        #     libbngsim_expression after the NFsim objects.
        # expression.cpp is compiled exactly once (in the bngsim_expression
        # target); NFsim only *links* that archive, it does not recompile the
        # evaluator, so there is no duplicate-symbol / ODR hazard. This mirrors
        # the RuleMonkey handoff (issue #39), which links bngsim::expression for
        # the analogous reason.
        if(NOT TARGET bngsim::expression)
            message(FATAL_ERROR
                "bngsim::expression must be declared before add_subdirectory(third_party/nfsim) "
                "so the vendored NFsim mu::Parser shim can forward to the host BNG-compat "
                "primitives instead of carrying a hand-ported copy (issue #49).")
        endif()
        # GH #126: the vendored NFsim tree no longer carries its own byte-identical
        # src/NFfunction/exprtk/exprtk.hpp (pruned from the vendor export). The
        # mu::Parser shim's `#include "exprtk.hpp"` now resolves against the single
        # host snapshot in third_party/exprtk. The vendored NFsim CMakeLists still
        # appends its (now-absent) src/NFfunction/exprtk to the include path — a
        # harmless dead -I, mirroring the long-standing pruned-muParser pattern — so
        # we supply the real header directory on the NFsim targets here rather than
        # carry a redirect inside the vendored tree (keeps the carry queue empty of
        # this cleanup). bngsim::expression exposes third_party/exprtk only PRIVATEly,
        # so linking it does not propagate the dir; add it explicitly.
        if(TARGET NFsim_objects)
            target_link_libraries(NFsim_objects PRIVATE bngsim::expression)
            target_include_directories(NFsim_objects PRIVATE ${BNGSIM_EXPRTK_DIR})
            # ExprTk-heavy NFsim TUs (e.g. NFfunction/localFunction.cpp) exceed
            # MSVC's default COFF section limit; /bigobj raises it. GH #150.
            # Mirrors the same flag on bngsim_expression and bngsim below.
            target_compile_options(NFsim_objects PRIVATE
                $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
        endif()
        if(TARGET nfsim)
            target_link_libraries(nfsim PUBLIC bngsim::expression)
            target_include_directories(nfsim PUBLIC ${BNGSIM_EXPRTK_DIR})
        endif()
    else()
        message(WARNING "Vendored NFsim source not found at third_party/nfsim/src/, "
                        "disabling BNGSIM_BUILD_NFSIM")
        set(BNGSIM_BUILD_NFSIM OFF)
    endif()
endif()

# ─── RuleMonkey (vendored static library) ─────────────────────────────────────
option(BNGSIM_BUILD_RULEMONKEY "Build vendored RuleMonkey for in-process nf_exact/rm simulation" ON)

if(BNGSIM_BUILD_RULEMONKEY)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rulemonkey/include/rulemonkey/simulator.hpp")
        message(STATUS "Building vendored RuleMonkey from: third_party/rulemonkey/")
        set(RULEMONKEY_BUILD_CLI OFF CACHE BOOL "" FORCE)
        set(RULEMONKEY_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(RULEMONKEY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(RULEMONKEY_INSTALL OFF CACHE BOOL "" FORCE)
        if(NOT TARGET bngsim::expression)
            message(FATAL_ERROR
                "bngsim::expression must be declared before add_subdirectory(third_party/rulemonkey) "
                "so vendored RuleMonkey reuses the host ExprTk evaluator instead of compiling "
                "a duplicate copy.")
        endif()
        add_subdirectory(third_party/rulemonkey)
        get_target_property(_bngsim_rulemonkey_link_libraries rulemonkey LINK_LIBRARIES)
        if(NOT _bngsim_rulemonkey_link_libraries)
            set(_bngsim_rulemonkey_link_libraries "")
        endif()
        list(FIND _bngsim_rulemonkey_link_libraries bngsim::expression _bngsim_rulemonkey_expr_index)
        if(_bngsim_rulemonkey_expr_index EQUAL -1)
            message(FATAL_ERROR
                "Vendored RuleMonkey must link bngsim::expression inside BNGsim. "
                "The upstream RuleMonkey CMake handoff changed and would reintroduce duplicate "
                "ExprTkEvaluator symbols. Refresh the vendoring guardrails before proceeding.")
        endif()
        unset(_bngsim_rulemonkey_expr_index)
        unset(_bngsim_rulemonkey_link_libraries)
        set_target_properties(rulemonkey PROPERTIES POSITION_INDEPENDENT_CODE ON)
        # MSVC's <cmath> hides POSIX math constants (M_PI, M_E, ...) unless
        # _USE_MATH_DEFINES is set before <cmath> is pulled in. The vendored
        # RuleMonkey source also defines this defensively at the top of
        # expr_eval.cpp; we apply it as a target-wide compile definition so
        # any future vendored TU using <cmath> + M_PI / M_E stays portable
        # on MSVC. No-op on POSIX toolchains (Clang/GCC/Apple).
        target_compile_definitions(rulemonkey PRIVATE _USE_MATH_DEFINES)
    else()
        message(WARNING "Vendored RuleMonkey source not found at third_party/rulemonkey/, "
                        "disabling BNGSIM_BUILD_RULEMONKEY")
        set(BNGSIM_BUILD_RULEMONKEY OFF)
    endif()
endif()

# ─── MIR (vendored in-process micro-JIT for the codegen RHS, GH #78) ──────────
# Prototype backend. When ON, the code-generated ODE RHS can be JIT-compiled
# in-process with c2mir + MIR_gen (~1-2 ms) instead of shelling out to
# `cc -O3` + dlopen (~80 ms-seconds). Default OFF: the cc-subprocess + dlopen
# codegen backend and the ExprTk fallback are unchanged when MIR is disabled.
option(BNGSIM_ENABLE_MIR "Build the vendored MIR micro-JIT backend for the codegen RHS (GH #78, prototype)" OFF)

if(BNGSIM_ENABLE_MIR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mir/c2mir/c2mir.c")
        message(STATUS "Building vendored MIR micro-JIT from: third_party/mir/")
        # Three translation units; each #includes the host target backend under
        # a target-macro guard, so only the host arch's files are read. Compiled
        # as C (gnu11 — MIR uses GNU extensions). Warnings silenced: third-party.
        add_library(bngsim_mir STATIC
            third_party/mir/mir.c
            third_party/mir/mir-gen.c
            third_party/mir/c2mir/c2mir.c
        )
        set_target_properties(bngsim_mir PROPERTIES
            C_STANDARD 11
            C_EXTENSIONS ON
            POSITION_INDEPENDENT_CODE ON
        )
        # Consumers (bngsim's MirJit shim) include "mir-gen.h" and "c2mir.h".
        target_include_directories(bngsim_mir PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/mir>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/mir/c2mir>
        )
        # MIR needs the dynamic-loader lib (dlsym for external resolve) and, on
        # platforms where libm is a separate library, libm. On Apple libm is
        # part of libSystem (always linked), so an explicit -lm is redundant —
        # and because libm also arrives transitively (SUNDIALS), the redundant
        # entry trips a benign `ld: warning: ignoring duplicate libraries: -lm`.
        # On Windows (MSVC, GH #3) there is no separate libm — the math runtime
        # lives in the C runtime — and CMAKE_DL_LIBS is empty (no libdl; the MIR
        # import table is a static libm/libc table, not dlsym). So link libm only
        # on the Unix-y platforms where it is a real, separate archive.
        # C2MIR_PARALLEL is intentionally left undefined → single-threaded gen,
        # no pthread dependency.
        target_link_libraries(bngsim_mir PUBLIC ${CMAKE_DL_LIBS})
        if(NOT APPLE AND NOT WIN32)
            target_link_libraries(bngsim_mir PUBLIC m)
        endif()
        target_compile_options(bngsim_mir PRIVATE
            $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-w>
            $<$<C_COMPILER_ID:MSVC>:/w>
        )
    else()
        message(WARNING "Vendored MIR source not found at third_party/mir/, "
                        "disabling BNGSIM_ENABLE_MIR")
        set(BNGSIM_ENABLE_MIR OFF)
    endif()
endif()

# ─── Library target ────────────────────────────────────────────────────────────
# When building Python bindings, build bngsim as a static library so the
# extension module is self-contained (no separate libbngsim.dylib needed).
if(BNGSIM_BUILD_PYTHON)
    set(BUILD_SHARED_LIBS OFF)
endif()

# NB: src/expression.cpp is built as the separate bngsim::expression target
# (declared above, before the RuleMonkey subdirectory) and linked in below.
set(BNGSIM_SOURCES
    src/model.cpp
    src/model_builder.cpp
    src/net_file_loader.cpp
    src/table_function.cpp
    src/cvode_simulator.cpp
    src/ssa_simulator.cpp
    src/result.cpp
    src/bngsim_api.cpp
    src/steady_state.cpp
    src/lapack_dense_linsol.cpp
)

# Conditionally add NFsim simulator wrapper
if(BNGSIM_BUILD_NFSIM)
    list(APPEND BNGSIM_SOURCES src/nfsim_simulator.cpp)
endif()

# Conditionally add RuleMonkey simulator wrapper
if(BNGSIM_BUILD_RULEMONKEY)
    list(APPEND BNGSIM_SOURCES src/rulemonkey_simulator.cpp)
endif()

# ─── Optimized dense linear solver (GH #84) ──────────────────────────────────
# A custom SUNLinearSolver (src/lapack_dense_linsol.cpp) factors large, dense
# Jacobians with a blocked BLAS dgetrf while keeping SUNDIALS' built-in
# triangular back-solve. macOS links the Accelerate framework (always present,
# no new dependency); other platforms link a system LAPACK (e.g. OpenBLAS). The
# source always compiles — when no backend is found it falls back to the
# built-in dense LU at runtime — so the build succeeds everywhere. We resolve
# the library here and link it (with the BNGSIM_HAS_LAPACK_DENSE define) after
# the target is declared, mirroring the KLU conditional-link pattern.
option(BNGSIM_ENABLE_LAPACK_DENSE
       "Enable the optimized BLAS dgetrf dense linear solver (GH #84)" ON)
set(BNGSIM_LAPACK_DENSE OFF)
set(BNGSIM_LAPACK_DENSE_LIBS "")
set(BNGSIM_LAPACK_DENSE_BACKEND "")
if(BNGSIM_ENABLE_LAPACK_DENSE)
    if(APPLE)
        find_library(BNGSIM_ACCELERATE_FRAMEWORK Accelerate)
        if(BNGSIM_ACCELERATE_FRAMEWORK)
            set(BNGSIM_LAPACK_DENSE ON)
            set(BNGSIM_LAPACK_DENSE_LIBS "${BNGSIM_ACCELERATE_FRAMEWORK}")
            set(BNGSIM_LAPACK_DENSE_BACKEND "Accelerate")
        endif()
    else()
        # This solver calls the Fortran dgetrf_ symbol directly, so its integer
        # type must match the selected LAPACK library. Probe the selected
        # library and compile the solver for the detected ABI: 32-bit integers
        # for LP64 LAPACK, 64-bit integers for ILP64 LAPACK.
        if(NOT DEFINED BLA_SIZEOF_INTEGER)
            set(BLA_SIZEOF_INTEGER ANY)
        endif()
        find_package(LAPACK QUIET)
        if(LAPACK_FOUND)
            if(CMAKE_CROSSCOMPILING)
                message(STATUS "BNGsim: optimized dense solver DISABLED "
                               "(cannot verify LAPACK integer ABI while cross-compiling)")
            else()
                include(CheckCSourceRuns)
                set(_bngsim_saved_required_libs "${CMAKE_REQUIRED_LIBRARIES}")
                set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
                check_c_source_runs([=[
extern void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv,
                    int *info);

int main(void) {
    int n = 2;
    int lda = 2;
    int info = -999;
    int ipiv[2] = {0, 0};
    double a[4] = {0.0, 1.0, 2.0, 3.0};
    dgetrf_(&n, &n, a, &lda, ipiv, &info);
    if (info != 0) return 1;
    if (ipiv[0] != 2) return 2;
    if (ipiv[1] != 2) return 3;
    return 0;
}
]=] BNGSIM_LAPACK_DGETRF_INT32_RUNS)
                check_c_source_runs([=[
#include <stdint.h>

extern void dgetrf_(const int64_t *m, const int64_t *n, double *a, const int64_t *lda,
                    int64_t *ipiv, int64_t *info);

int main(void) {
    int64_t n = 2;
    int64_t lda = 2;
    int64_t info = -999;
    int64_t ipiv[2] = {0, 0};
    double a[4] = {0.0, 1.0, 2.0, 3.0};
    dgetrf_(&n, &n, a, &lda, ipiv, &info);
    if (info != 0) return 1;
    if (ipiv[0] != 2) return 2;
    if (ipiv[1] != 2) return 3;
    return 0;
}
]=] BNGSIM_LAPACK_DGETRF_INT64_RUNS)
                set(CMAKE_REQUIRED_LIBRARIES "${_bngsim_saved_required_libs}")
                unset(_bngsim_saved_required_libs)
                if(BNGSIM_LAPACK_DGETRF_INT32_RUNS)
                    set(BNGSIM_LAPACK_DENSE ON)
                    set(BNGSIM_LAPACK_DENSE_LIBS "${LAPACK_LIBRARIES}")
                    set(BNGSIM_LAPACK_DENSE_BACKEND "system LAPACK (32-bit integer ABI)")
                elseif(BNGSIM_LAPACK_DGETRF_INT64_RUNS)
                    set(BNGSIM_LAPACK_DENSE ON)
                    set(BNGSIM_LAPACK_DENSE_LIBS "${LAPACK_LIBRARIES}")
                    set(BNGSIM_LAPACK_DENSE_BACKEND "system LAPACK (64-bit integer ABI)")
                    set(BNGSIM_LAPACK_DENSE_INT64 ON)
                else()
                    message(STATUS "BNGsim: optimized dense solver DISABLED "
                                   "(LAPACK dgetrf_ did not pass either integer ABI check)")
                endif()
            endif()
        endif()
    endif()
endif()

add_library(bngsim ${BNGSIM_SOURCES})

# ExprTk-heavy translation units can exceed MSVC's default section limit.
target_compile_options(bngsim PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

target_include_directories(bngsim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# NFsim simulator needs NFsim headers
if(BNGSIM_BUILD_NFSIM)
    target_include_directories(bngsim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nfsim/src
    )
    target_compile_definitions(bngsim PUBLIC BNGSIM_HAS_NFSIM=1)
endif()

if(BNGSIM_BUILD_RULEMONKEY)
    target_compile_definitions(bngsim PUBLIC BNGSIM_HAS_RULEMONKEY=1)
endif()

# MIR micro-JIT backend (GH #78). Linked PRIVATE — the JIT is an internal
# implementation detail of the codegen RHS path; MIR's headers do not leak into
# bngsim's public ABI (mir_jit.hpp guards all MIR includes behind BNGSIM_HAS_MIR).
if(BNGSIM_ENABLE_MIR)
    target_link_libraries(bngsim PRIVATE bngsim_mir)
    target_compile_definitions(bngsim PRIVATE BNGSIM_HAS_MIR=1)
endif()

target_link_libraries(bngsim
    PUBLIC
        bngsim::expression
        SUNDIALS::cvodes
        SUNDIALS::kinsol
        SUNDIALS::nvecserial
        SUNDIALS::sunmatrixdense
        SUNDIALS::sunlinsoldense
)

# KLU sparse solver: link conditionally.
# SUNDIALS FetchContent creates alias targets SUNDIALS::sunlinsolklu when
# ENABLE_KLU is ON. We check ENABLE_KLU (which we set and SUNDIALS reads)
# as the authoritative signal.
if(BNGSIM_LINK_KLU)
    target_link_libraries(bngsim PUBLIC
        SUNDIALS::sunmatrixsparse
        SUNDIALS::sunlinsolklu
    )
    target_compile_definitions(bngsim PUBLIC BNGSIM_HAS_KLU=1)
    if(BNGSIM_USE_SYSTEM_SUNDIALS)
        message(STATUS "BNGsim: sparse Jacobian support ENABLED (system KLU)")
    else()
        message(STATUS "BNGsim: sparse Jacobian support ENABLED (KLU)")
    endif()
else()
    message(STATUS "BNGsim: sparse Jacobian support DISABLED (${BNGSIM_KLU_DISABLE_REASON})")
endif()

# Optimized dense linear solver (GH #84): link the resolved BLAS backend and
# define BNGSIM_HAS_LAPACK_DENSE. Absent (no Accelerate / no system LAPACK), the
# custom solver source still compiles but lapack_dense_available() returns false
# and make_dense_linear_solver() always returns the built-in dense LU.
if(BNGSIM_LAPACK_DENSE)
    target_link_libraries(bngsim PUBLIC ${BNGSIM_LAPACK_DENSE_LIBS})
    target_compile_definitions(bngsim PUBLIC BNGSIM_HAS_LAPACK_DENSE=1)
    if(BNGSIM_LAPACK_DENSE_INT64)
        target_compile_definitions(bngsim PUBLIC BNGSIM_LAPACK_DENSE_INT64=1)
    endif()
    message(STATUS "BNGsim: optimized dense solver ENABLED (${BNGSIM_LAPACK_DENSE_BACKEND})")
elseif(BNGSIM_ENABLE_LAPACK_DENSE)
    message(STATUS "BNGsim: optimized dense solver DISABLED (no BLAS/LAPACK found) — built-in dense LU")
else()
    message(STATUS "BNGsim: optimized dense solver explicitly disabled — built-in dense LU")
endif()

# Link against libnfsim when NFsim is enabled
if(BNGSIM_BUILD_NFSIM)
    target_link_libraries(bngsim PRIVATE nfsim)
endif()

# Link against RuleMonkey when enabled
if(BNGSIM_BUILD_RULEMONKEY)
    target_link_libraries(bngsim PRIVATE RuleMonkey::rulemonkey)
endif()

set_target_properties(bngsim PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME bngsim
)

# ─── Tests ─────────────────────────────────────────────────────────────────────
if(BNGSIM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Python bindings ────────────────────────────────────────────────────────────
if(BNGSIM_BUILD_PYTHON)
    # editable.rebuild = false (pyproject.toml), but the manual helper
    # bngsim/scripts/rebuild_editable.py re-runs CMake later against the
    # existing editable build directory. The original configure may have run
    # inside a uv build-isolation venv that gets deleted, leaving cached
    # Python_EXECUTABLE / pybind11_DIR paths pointing at phantoms. Drop stale
    # cache entries before find_package so the manual rebuild helper can
    # re-discover the live runtime venv. See #23.
    foreach(_var Python_EXECUTABLE Python3_EXECUTABLE PYTHON_EXECUTABLE)
        if(${_var} AND NOT EXISTS "${${_var}}")
            unset(${_var} CACHE)
            unset(_${_var} CACHE)
        endif()
    endforeach()
    if(pybind11_DIR
       AND NOT EXISTS "${pybind11_DIR}/pybind11Config.cmake"
       AND NOT EXISTS "${pybind11_DIR}/pybind11-config.cmake")
        unset(pybind11_DIR CACHE)
    endif()

    # Resolve pybind11 via the runtime Python so the editable rebuild path
    # works after the build-isolation venv is gone. Try several candidates
    # because $VIRTUAL_ENV is unset when `.venv/bin/python` is invoked
    # directly, and FindPython resolves venv symlinks to the underlying
    # interpreter (which lacks the venv's site-packages).
    if(NOT pybind11_DIR)
        set(_bngsim_py_candidates "")
        if(DEFINED ENV{BNGSIM_PYTHON_EXECUTABLE})
            list(APPEND _bngsim_py_candidates "$ENV{BNGSIM_PYTHON_EXECUTABLE}")
        endif()
        if(DEFINED ENV{VIRTUAL_ENV})
            list(APPEND _bngsim_py_candidates "$ENV{VIRTUAL_ENV}/bin/python")
        endif()
        list(APPEND _bngsim_py_candidates
            "${CMAKE_SOURCE_DIR}/.venv/bin/python"
            "${CMAKE_SOURCE_DIR}/../.venv/bin/python"
        )
        find_program(_bngsim_py_path NAMES python3 python)
        if(_bngsim_py_path)
            list(APPEND _bngsim_py_candidates "${_bngsim_py_path}")
        endif()

        foreach(_py IN LISTS _bngsim_py_candidates)
            if(EXISTS "${_py}")
                execute_process(
                    COMMAND "${_py}" -c
                        "import pybind11; print(pybind11.get_cmake_dir())"
                    OUTPUT_VARIABLE _bngsim_pybind11_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _bngsim_pybind11_lookup
                    ERROR_QUIET
                )
                if(_bngsim_pybind11_lookup EQUAL 0 AND _bngsim_pybind11_dir)
                    set(pybind11_DIR "${_bngsim_pybind11_dir}"
                        CACHE PATH "pybind11 cmake dir" FORCE)
                    break()
                endif()
            endif()
        endforeach()
    endif()

    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_bngsim_core MODULE src/_bngsim_core.cpp)
    target_link_libraries(_bngsim_core PRIVATE bngsim)
    target_include_directories(_bngsim_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    # Stamp the version into the extension via a compile define, so the
    # C++ side never holds its own literal copy (issue #31). The build commit
    # (issue #125) is stamped the same way so the loaded binary can report the
    # source it was built from — the identity half of the stale-binary guard.
    target_compile_definitions(_bngsim_core PRIVATE
        BNGSIM_VERSION_STR="${PROJECT_VERSION}"
        BNGSIM_BUILD_COMMIT_STR="${BNGSIM_BUILD_COMMIT}"
    )
    # The MIR define is PRIVATE on the bngsim static lib, so it does not reach this
    # pybind TU; without it mir_jit.hpp compiles to the inert stub and any direct
    # MirJit use here (GH #149 SSA propensity-JIT binding) throws. The MIR symbols
    # already link in transitively via the static bngsim lib; this only flips the
    # header to the real implementation. (No-op when MIR is OFF.)
    if(BNGSIM_ENABLE_MIR)
        target_compile_definitions(_bngsim_core PRIVATE BNGSIM_HAS_MIR=1)
        # Bring in MIR's PUBLIC include dirs (c2mir.h / mir-gen.h) so the header's
        # MIR includes resolve in this TU; symbols already arrive via bngsim.
        target_link_libraries(_bngsim_core PRIVATE bngsim_mir)
    endif()
    # The single pybind binding TU registers the whole API and can exceed
    # MSVC's default COFF section limit; /bigobj raises it. GH #150.
    target_compile_options(_bngsim_core PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

    # When building for Python, link bngsim statically into the extension
    # so the .so/.dylib is self-contained (no separate libbngsim needed).
    set_target_properties(bngsim PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # scikit-build-core handles install destination.
    # wheel.install-dir = "bngsim" in pyproject.toml, so DESTINATION "."
    # puts the .so at bngsim/_bngsim_core.so (inside the Python package).
    install(TARGETS _bngsim_core DESTINATION .)

    # When KLU was built from source (no system SuiteSparse — the sdist
    # fallback), delocate/auditwheel does not run, so bundle the shared libs
    # ourselves: copy them into bngsim/.dylibs and point the extension's rpath
    # there. The libs already carry @rpath/$ORIGIN install-names + an rpath to
    # find each other (ci/build_suitesparse.sh, SS_INSTALL_* env). No-op on the
    # wheel path, where system KLU is discovered and the repair tool bundles it.
    if(BNGSIM_KLU_AUTOBUILT)
        # Copy the whole KLU shared-lib set (SuiteSparse ships each as a
        # real file + a SONAME symlink + an unversioned dev symlink). We match
        # every dylib/so, not just the SONAME the extension references: matching
        # only the SONAME would select the *symlink* while excluding its real
        # target, leaving a dangling link the wheel writer drops. The writer
        # dereferences the symlinks into copies, so the bundle carries a few
        # redundant copies — acceptable for this no-system-SuiteSparse fallback.
        install(DIRECTORY "${BNGSIM_KLU_BUNDLE_DIR}/" DESTINATION .dylibs
                FILES_MATCHING PATTERN "*.dylib" PATTERN "*.so*")
        if(APPLE)
            set_property(TARGET _bngsim_core APPEND PROPERTY
                INSTALL_RPATH "@loader_path/.dylibs")
        else()
            set_property(TARGET _bngsim_core APPEND PROPERTY
                INSTALL_RPATH "$ORIGIN/.dylibs")
        endif()
    endif()
endif()

# ─── Install (C++ library — standalone builds only) ───────────────────────
# When building Python bindings, bngsim is statically linked into the
# extension module. No need to install the library or headers into the wheel.
if(NOT BNGSIM_BUILD_PYTHON)
install(TARGETS bngsim bngsim_expression
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/bngsim DESTINATION include)
endif()
