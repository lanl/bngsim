# Tests for bngsim Phase A

# Resolve test data directory at compile time
set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(test_bngsim test_bngsim.cpp)
target_link_libraries(test_bngsim PRIVATE bngsim)
target_compile_definitions(test_bngsim PRIVATE TEST_DATA_DIR="${TEST_DATA_DIR}")
add_test(NAME bngsim_tests COMMAND test_bngsim)

# Validation against run_network reference outputs
add_executable(test_validation test_validation.cpp)
target_link_libraries(test_validation PRIVATE bngsim)
target_compile_definitions(test_validation PRIVATE TEST_DATA_DIR="${TEST_DATA_DIR}")
add_test(NAME bngsim_validation COMMAND test_validation)

# ─── Optimized dense linear solver (GH #84) ─────────────────────────────────
add_executable(test_lapack_dense_linsol test_lapack_dense_linsol.cpp)
target_link_libraries(test_lapack_dense_linsol PRIVATE bngsim)
add_test(NAME lapack_dense_linsol_test COMMAND test_lapack_dense_linsol)

# ─── C API tests (Session 61, NEXT_STEPS P3) ────────────────────────────────
# test_c_api reads its model from argv[1] (a .net path) rather than a
# TEST_DATA_DIR compile define, and exits non-zero if given none. Register it
# with a minimal valid model; it is model-agnostic (the reserved-name and
# error-path cases dominate), so simple_decay.net exercises every branch.
add_executable(test_c_api test_c_api.cpp)
target_link_libraries(test_c_api PRIVATE bngsim)
add_test(NAME c_api_test COMMAND test_c_api "${TEST_DATA_DIR}/simple_decay.net")

# ─── Cold-start seed-count rounding policy (GH #51) ──────────────────────────
# Header-only target: seed_count_rounding.hpp has no link dependencies, so it
# does not pull bngsim's interface include dirs transitively. It still needs
# both src/ (for the header itself) and include/ (seed_count_rounding.hpp
# includes bngsim/platform_compat.hpp, the Windows getpid() shim from GH #150)
# on the search path, or the standalone target fails to compile.
add_executable(test_seed_count_rounding test_seed_count_rounding.cpp)
target_include_directories(test_seed_count_rounding PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include)
set_target_properties(test_seed_count_rounding PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME seed_count_rounding_test COMMAND test_seed_count_rounding)

# ─── NFsim library integration test (Phase C.5) ──────────────────────────────
if(BNGSIM_BUILD_NFSIM)
    # NFsim test data: vendored alongside the source
    set(NFSIM_TEST_DATA_DIR "${TEST_DATA_DIR}/nfsim")

    add_executable(test_nfsim_library test_nfsim_library.cpp)
    target_link_libraries(test_nfsim_library PRIVATE nfsim)
    target_compile_definitions(test_nfsim_library PRIVATE
        NFSIM_TEST_DATA_DIR="${NFSIM_TEST_DATA_DIR}"
    )
    # C++17: muParser replaced by ExprTk, no more std::auto_ptr
    set_target_properties(test_nfsim_library PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME nfsim_library_test COMMAND test_nfsim_library)

    # ─── NfsimSimulator wrapper test (Phase C.5 Step 2) ───────────────────────
    add_executable(test_nfsim_simulator test_nfsim_simulator.cpp)
    target_link_libraries(test_nfsim_simulator PRIVATE bngsim)
    target_compile_definitions(test_nfsim_simulator PRIVATE
        NFSIM_TEST_DATA_DIR="${NFSIM_TEST_DATA_DIR}"
    )
    set_target_properties(test_nfsim_simulator PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME nfsim_simulator_test COMMAND test_nfsim_simulator)

    add_executable(nfsim_connectivity_diag nfsim_connectivity_diag.cpp)
    target_link_libraries(nfsim_connectivity_diag PRIVATE nfsim)
    set_target_properties(nfsim_connectivity_diag PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endif()
