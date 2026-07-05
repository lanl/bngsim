cmake_minimum_required(VERSION 3.20)

project(RuleMonkey VERSION 3.5.0 LANGUAGES CXX)

if(CMAKE_VERSION VERSION_LESS 3.21)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(PROJECT_IS_TOP_LEVEL ON)
    else()
        set(PROJECT_IS_TOP_LEVEL OFF)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(RM_UNIVERSAL_BINARY
    "On macOS, build a universal arm64+x86_64 binary (~2x build time)" OFF)
if(APPLE AND RM_UNIVERSAL_BINARY)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Universal binary")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# ---------------------------------------------------------------------------
# librulemonkey: the core simulation engine
# ---------------------------------------------------------------------------
add_library(rulemonkey STATIC
    cpp/rulemonkey/engine.cpp
    cpp/rulemonkey/simulator.cpp
    cpp/rulemonkey/expr_eval.cpp
    cpp/rulemonkey/table_function.cpp
    cpp/rulemonkey/canonical.cpp
    cpp/rulemonkey/pattern_parser.cpp
)

target_include_directories(rulemonkey
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/rulemonkey
)

# Stable namespaced alias for in-tree consumers (`add_subdirectory`):
#   target_link_libraries(my_app PRIVATE RuleMonkey::rulemonkey)
# Mirrors the alias produced by find_package() against an install.
add_library(RuleMonkey::rulemonkey ALIAS rulemonkey)

target_compile_features(rulemonkey PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Expression evaluator backend: ExprTk, via bngsim::ExprTkEvaluator.
#
# RM's rate-law / function expression evaluator IS bngsim::ExprTkEvaluator
# (issue #6, dev/exprtk_swap_plan_2026_05_16.md).  RM is itself vendored into
# BNGsim, which already ships these exact sources as the `bngsim::expression`
# target.  Compiling RM's vendored copy *inside* a BNGsim build would emit a
# second set of bngsim:: symbols → ODR violation / linker duplicate-symbol
# error.  So the vendored copy is gated on PROJECT-level target detection:
#
#   - In a BNGsim build  -> link the host's bngsim::expression target.
#   - Standalone RM build -> compile the vendored third_party copy.
#
# The vendored expression.cpp is built as a separate OBJECT library so the
# 1.6 MB exprtk.hpp compile (~40 s) is isolated to one TU and so exprtk's
# (third-party, not warning-clean) code is not subject to RM's -Wall -Werror.
# Its objects are folded into librulemonkey.a via target_sources, so there is
# no extra target to install/export.  See third_party/bngsim_expr/VENDOR.
# ---------------------------------------------------------------------------
if(TARGET bngsim::expression)
    target_link_libraries(rulemonkey PRIVATE bngsim::expression)
else()
    add_library(rm_bngsim_expr OBJECT
        third_party/bngsim_expr/src/expression.cpp)
    target_include_directories(rm_bngsim_expr PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bngsim_expr/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/exprtk)
    target_compile_features(rm_bngsim_expr PUBLIC cxx_std_17)
    target_sources(rulemonkey PRIVATE $<TARGET_OBJECTS:rm_bngsim_expr>)
endif()

# -Werror is on by default for the top-level developer build so a warning
# shows up as a compile failure before it lands in CI.  Downstream consumers
# building RM as a subdirectory get the less brittle default and can opt in
# with -DRULEMONKEY_WARNINGS_AS_ERRORS=ON.
option(RULEMONKEY_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ${PROJECT_IS_TOP_LEVEL})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(rulemonkey PRIVATE
        -Wall -Wextra -Wno-deprecated-declarations
    )
    if(RULEMONKEY_WARNINGS_AS_ERRORS)
        target_compile_options(rulemonkey PRIVATE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# Optional dev-time profiler instrumentation.
#
# When ON, defines RM_DEV_PROFILES so engine_profile.hpp's gates resolve to
# `true`, and the engine emits per-phase profile reports to stderr at the
# end of each run.  Default OFF: profiler scaffolding compiles to nothing
# (every `if constexpr (k*Profile)` block is dead-stripped).
# ---------------------------------------------------------------------------
option(RULEMONKEY_ENABLE_DEV_PROFILES "Compile in engine dev profiler instrumentation" OFF)
if(RULEMONKEY_ENABLE_DEV_PROFILES)
    target_compile_definitions(rulemonkey PRIVATE RM_DEV_PROFILES)
endif()

# ---------------------------------------------------------------------------
# Command-line tools: rm_driver (single-trajectory batch) and rm_scan
# (parameter_scan / bifurcate parameter sweeps).
# ---------------------------------------------------------------------------
option(RULEMONKEY_BUILD_CLI "Build the rm_driver / rm_scan command-line tools" ${PROJECT_IS_TOP_LEVEL})
if(RULEMONKEY_BUILD_CLI)
    add_executable(rm_driver cpp/cli/rm_driver.cpp)
    add_executable(rm_scan cpp/cli/rm_scan.cpp)

    foreach(_rm_cli_target rm_driver rm_scan)
        target_link_libraries(${_rm_cli_target} PRIVATE rulemonkey)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(${_rm_cli_target} PRIVATE
                -Wall -Wextra -Wno-deprecated-declarations)
            if(RULEMONKEY_WARNINGS_AS_ERRORS)
                target_compile_options(${_rm_cli_target} PRIVATE -Werror)
            endif()
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Optional AddressSanitizer + UndefinedBehaviorSanitizer.
#
# Heavy index-based pool manipulation, BFS / Fenwick code, and pattern-matching
# fast paths are exactly the surface where ASan + UBSan routinely catch
# regressions that observable-trajectory parity tests miss.  Off by default
# (the perf hit is ~2x); cmake --preset asan turns it on.
#
# Both Linux and macOS are exercised in CI (the asan job's matrix).  The
# sanitizer flags themselves are platform-neutral; the CI step sets
# ASAN_OPTIONS / UBSAN_OPTIONS to keep diagnostic output uniform.
# Applied after the optional rm_driver is declared so every built target gets
# the flags.
# ---------------------------------------------------------------------------
option(RULEMONKEY_ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)
if(RULEMONKEY_ENABLE_ASAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "RULEMONKEY_ENABLE_ASAN requires Clang or GCC")
    endif()
    set(_rm_asan_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_compile_options(rulemonkey  PRIVATE ${_rm_asan_flags})
    target_link_options   (rulemonkey  PUBLIC  ${_rm_asan_flags})
    foreach(_rm_cli_target rm_driver rm_scan)
        if(TARGET ${_rm_cli_target})
            target_compile_options(${_rm_cli_target} PRIVATE ${_rm_asan_flags})
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Canonical-label cache self-check (issue #9 §2 step 5, plan decision #6).
#
# The cached-incremental canonical-label layer (per-complex cached label +
# dirty bit, invalidated by the structural mutators) has no production
# consumer yet — partial scaling (plan §7.2) is future work.  Its
# correctness is instead proven by an invariant: every cached-label read
# must equal a from-scratch recanonicalization.  The check is compiled IN
# for Debug and ASan builds, so ctest and the corpus guard tier exercise
# it, and OUT of Release, where the batch `.species` sweep stays a plain
# from-scratch walk.  A plain `assert` cannot carry this: the asan preset
# is RelWithDebInfo, which defines NDEBUG.
# ---------------------------------------------------------------------------
if(RULEMONKEY_ENABLE_ASAN OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(rulemonkey PRIVATE RULEMONKEY_CANONICAL_CACHE_SELFCHECK)
endif()

# ---------------------------------------------------------------------------
# Local-observable tracker self-check (issue #10).  evaluate_observable_on
# routes tracked Molecules-type observables through the per-mid
# obs_mol_contrib table instead of a from-scratch embedding recompute.  The
# table is delta-maintained by incremental_update_observables; correctness
# is proven by an invariant — every fast-path read must equal the
# from-scratch recompute.  Compiled IN for Debug and ASan, OUT of Release,
# matching the canonical-cache self-check above.
# ---------------------------------------------------------------------------
if(RULEMONKEY_ENABLE_ASAN OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(rulemonkey PRIVATE RULEMONKEY_LOCAL_OBS_SELFCHECK)
endif()

# ---------------------------------------------------------------------------
# Examples (off by default — host projects opt in)
# ---------------------------------------------------------------------------
option(RULEMONKEY_BUILD_EXAMPLES "Build the C++ embedding examples" OFF)
if(RULEMONKEY_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(RULEMONKEY_BUILD_TESTS "Build RuleMonkey C++ tests" ${PROJECT_IS_TOP_LEVEL})
if(RULEMONKEY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install rules
#
# Layout under CMAKE_INSTALL_PREFIX (default /usr/local on Linux/macOS):
#
#   include/rulemonkey/{simulator,types}.hpp
#   lib/librulemonkey.a
#   bin/rm_driver     (when RULEMONKEY_BUILD_CLI=ON)
#   bin/rm_scan       (when RULEMONKEY_BUILD_CLI=ON)
#   lib/cmake/RuleMonkey/RuleMonkeyConfig.cmake
#   lib/cmake/RuleMonkey/RuleMonkeyConfigVersion.cmake
#   lib/cmake/RuleMonkey/RuleMonkeyTargets.cmake
#
# Host projects (e.g. BNGsim) consume RM via:
#
#   find_package(RuleMonkey 3.1 CONFIG REQUIRED)
#   target_link_libraries(my_app PRIVATE RuleMonkey::rulemonkey)
#
# Set RULEMONKEY_INSTALL=OFF when RM is consumed via add_subdirectory()
# from a parent project that doesn't want to advertise install rules.
# ---------------------------------------------------------------------------
option(RULEMONKEY_INSTALL "Generate install() rules for RuleMonkey" ON)

if(RULEMONKEY_INSTALL AND RULEMONKEY_ENABLE_ASAN)
    # Guard against installing an asan-instrumented archive: the asan flags
    # are PRIVATE compile / PUBLIC link on the rulemonkey target, so every
    # downstream `find_package(RuleMonkey)` consumer would inherit
    # `-fsanitize=address,undefined` on its final link line.  That fails
    # outright unless the consumer also enables asan, and even when it
    # links it produces a binary that runs against the wrong runtime.
    # Asan builds are dev-only; the supported `asan` preset disables
    # install for this reason.  Refuse loudly rather than silently
    # producing a poisoned install tree.
    message(FATAL_ERROR
        "RULEMONKEY_ENABLE_ASAN=ON is incompatible with RULEMONKEY_INSTALL=ON: "
        "the asan runtime would propagate to every consumer via PUBLIC link "
        "options and break their builds.  Pass -DRULEMONKEY_INSTALL=OFF (or "
        "use the `asan` preset, which does this for you).")
endif()

if(RULEMONKEY_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Public targets: always the static library, and the rm_driver / rm_scan
    # command-line tools when RULEMONKEY_BUILD_CLI=ON.
    set(_rulemonkey_install_targets rulemonkey)
    foreach(_rm_cli_target rm_driver rm_scan)
        if(TARGET ${_rm_cli_target})
            list(APPEND _rulemonkey_install_targets ${_rm_cli_target})
        endif()
    endforeach()

    install(TARGETS ${_rulemonkey_install_targets}
        EXPORT RuleMonkeyTargets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Public headers: simulator.hpp + types.hpp under include/rulemonkey/.
    install(DIRECTORY include/rulemonkey
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp"
    )

    # Generate and install the package config files so downstream
    # `find_package(RuleMonkey CONFIG REQUIRED)` works.
    set(_rulemonkey_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/RuleMonkey)

    install(EXPORT RuleMonkeyTargets
        FILE RuleMonkeyTargets.cmake
        NAMESPACE RuleMonkey::
        DESTINATION ${_rulemonkey_cmake_dir}
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/RuleMonkeyConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RuleMonkeyConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/RuleMonkeyConfig.cmake"
        INSTALL_DESTINATION ${_rulemonkey_cmake_dir}
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/RuleMonkeyConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/RuleMonkeyConfigVersion.cmake"
        DESTINATION ${_rulemonkey_cmake_dir}
    )
endif()
