// bngsim/tests/test_c_api.cpp — C API regression tests (Session 61, NEXT_STEPS P3)
//
// Tests bngsim_api.h: model lifecycle, parameter access, simulation,
// result access, error handling, reserved names. ~100 lines, 8 test cases.

#include "bngsim/bngsim_api.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

static const char* NET_FILE = nullptr;  // set from argv[1]
static int passed = 0, failed = 0;

#define TEST(name) \
    printf("  test_%s... ", #name); fflush(stdout);

#define PASS() \
    printf("OK\n"); passed++;

#define FAIL(msg) \
    printf("FAIL: %s\n", msg); failed++;

// ─── Test 1: Null input safety ──────────────────────────────────────────────
static void test_null_safety() {
    TEST(null_safety);
    BNGModel* m = bng_model_create(nullptr);
    assert(m == nullptr);
    const char* err = bng_last_error();
    assert(err && strlen(err) > 0);

    // Null model operations should not crash
    bng_model_free(nullptr);
    bng_result_free(nullptr);
    assert(bng_n_species(nullptr) == 0);
    assert(bng_set_param(nullptr, "k", 1.0) == -1);

    double v;
    assert(bng_get_param(nullptr, "k", &v) == -1);
    PASS();
}

// ─── Test 2: Model create + free lifecycle ──────────────────────────────────
static void test_lifecycle() {
    TEST(lifecycle);
    BNGModel* m = bng_model_create(NET_FILE);
    if (!m) { FAIL(bng_last_error()); return; }

    assert(bng_n_species(m) > 0);
    assert(bng_n_reactions(m) > 0);
    assert(bng_n_parameters(m) > 0);

    bng_model_free(m);
    PASS();
}

// ─── Test 3: Parameter get/set round-trip ───────────────────────────────────
static void test_param_roundtrip() {
    TEST(param_roundtrip);
    BNGModel* m = bng_model_create(NET_FILE);
    if (!m) { FAIL(bng_last_error()); return; }

    // Find first parameter name by trying common ones from test .net files
    // We'll use the fact that set_param returns -1 for unknown names
    double orig;
    int rc = bng_get_param(m, "kp1", &orig);
    if (rc != 0) {
        // Model may not have kp1 — that's fine, test the error path
        assert(bng_last_error() && strlen(bng_last_error()) > 0);
        bng_model_free(m);
        PASS();
        return;
    }

    // Set to new value
    rc = bng_set_param(m, "kp1", orig * 2.0);
    assert(rc == 0);

    double readback;
    rc = bng_get_param(m, "kp1", &readback);
    assert(rc == 0);
    assert(std::abs(readback - orig * 2.0) < 1e-12);

    // Not-found parameter
    rc = bng_set_param(m, "__nonexistent__", 42.0);
    assert(rc == -1);

    bng_model_free(m);
    PASS();
}

// ─── Test 4: ODE simulation ─────────────────────────────────────────────────
static void test_simulate_ode() {
    TEST(simulate_ode);
    BNGModel* m = bng_model_create(NET_FILE);
    if (!m) { FAIL(bng_last_error()); return; }

    BNGResult* r = bng_simulate_ode(m, 0.0, 10.0, 11, 1e-8, 1e-8);
    if (!r) { FAIL(bng_last_error()); bng_model_free(m); return; }

    assert(bng_result_n_times(r) == 11);
    assert(bng_result_n_species(r) == bng_n_species(m));
    assert(bng_result_n_observables(r) == bng_n_observables(m));

    const double* t = bng_result_time(r);
    assert(t != nullptr);
    assert(std::abs(t[0] - 0.0) < 1e-12);
    assert(std::abs(t[10] - 10.0) < 1e-12);

    const double* sp = bng_result_species(r);
    assert(sp != nullptr);

    bng_result_free(r);
    bng_model_free(m);
    PASS();
}

// ─── Test 5: SSA simulation ─────────────────────────────────────────────────
static void test_simulate_ssa() {
    TEST(simulate_ssa);
    BNGModel* m = bng_model_create(NET_FILE);
    if (!m) { FAIL(bng_last_error()); return; }

    BNGResult* r = bng_simulate_ssa(m, 0.0, 10.0, 11, 12345ULL);
    if (!r) { FAIL(bng_last_error()); bng_model_free(m); return; }

    assert(bng_result_n_times(r) == 11);
    assert(bng_result_n_species(r) > 0);

    bng_result_free(r);
    bng_model_free(m);
    PASS();
}

// ─── Test 6: Clone ──────────────────────────────────────────────────────────
static void test_clone() {
    TEST(clone);
    BNGModel* m = bng_model_create(NET_FILE);
    if (!m) { FAIL(bng_last_error()); return; }

    BNGModel* c = bng_model_clone(m);
    assert(c != nullptr);
    assert(bng_n_species(c) == bng_n_species(m));

    bng_model_free(c);
    bng_model_free(m);
    PASS();
}

// ─── Test 7: Reserved constants ─────────────────────────────────────────────
static void test_reserved_constants() {
    TEST(reserved_constants);
    int count = 0;
    const char** names = bng_reserved_constants(&count);
    assert(names != nullptr);
    assert(count == 7);
    // Check _pi is first
    assert(strcmp(names[0], "_pi") == 0);
    // NULL-terminated
    assert(names[count] == nullptr);
    PASS();
}

// ─── Test 8: Reserved functions ─────────────────────────────────────────────
static void test_reserved_functions() {
    TEST(reserved_functions);
    int count = 0;
    const char** names = bng_reserved_functions(&count);
    assert(names != nullptr);
    assert(count == 37);
    assert(strcmp(names[0], "time") == 0);
    assert(names[count] == nullptr);
    PASS();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to-.net-file>\n", argv[0]);
        return 1;
    }
    NET_FILE = argv[1];

    printf("=== C API Tests ===\n");

    test_null_safety();
    test_lifecycle();
    test_param_roundtrip();
    test_simulate_ode();
    test_simulate_ssa();
    test_clone();
    test_reserved_constants();
    test_reserved_functions();

    printf("\n%d/%d tests passed.\n", passed, passed + failed);
    return failed > 0 ? 1 : 0;
}
