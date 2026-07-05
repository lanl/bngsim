// bngsim/tests/test_lapack_dense_linsol.cpp — GH #84
//
// Correctness gate for the custom dense linear solver: factoring with the BLAS
// dgetrf and back-solving with SUNDIALS' built-in denseGETRS must be equivalent
// to the all-built-in path. The feasibility scripts (dev/investigations/
// lu_backend/lu_verify.c) only checked dgetrf+dgetrs vs builtin+builtin — they
// never exercised this CROSS combination, which is exactly what production runs.
// So this test pins it directly: same residual on Ax=b and the two solution
// vectors agree to ~1e-12.

#include <bngsim/lapack_dense_linsol.hpp>
#include <bngsim/platform_compat.hpp> // POSIX setenv()/unsetenv() shim for Windows (GH #150)
#include <bngsim/sundials_guards.hpp>

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg)                                                                            \
    do {                                                                                            \
        if (!(cond)) {                                                                              \
            std::cerr << "  FAIL: " << msg << " [" << __FILE__ << ":" << __LINE__ << "]"            \
                      << std::endl;                                                                 \
            return 1;                                                                               \
        }                                                                                          \
    } while (0)

#define RUN_TEST(func)                                                                              \
    do {                                                                                            \
        ++tests_run;                                                                                \
        std::cout << "  " << #func << "... " << std::flush;                                         \
        int _rc = func();                                                                           \
        if (_rc == 0) {                                                                             \
            ++tests_passed;                                                                         \
            std::cout << "OK" << std::endl;                                                         \
        } else {                                                                                    \
            std::cout << "FAILED" << std::endl;                                                     \
        }                                                                                           \
    } while (0)

// Fill an n×n well-conditioned (diagonally dominant) matrix into column-major
// `data`, deterministically. Same construction as lu_verify.c.
static void fill_well_conditioned(double *data, int n, unsigned seed) {
    std::srand(seed);
    for (long i = 0; i < (long)n * n; ++i)
        data[i] = ((double)std::rand() / RAND_MAX) - 0.5;
    for (int d = 0; d < n; ++d)
        data[(long)d * n + d] += n; // strong diagonal → well conditioned
}

// ||A x - b||_inf against the ORIGINAL (unfactored) column-major A.
static double residual_inf(const double *A, int n, const double *x, const double *b) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            s += A[(long)j * n + i] * x[j];
        double r = std::fabs(s - b[i]);
        if (r > m)
            m = r;
    }
    return m;
}

// Solve Ax=b with the chosen dense backend; returns x and the LU residual.
// A_src is the pristine matrix; it is copied into a fresh SUNDenseMatrix that
// the solver factors in place.
static int solve_with(bool prefer_lapack, const double *A_src, const double *b, int n,
                      SUNContext ctx, std::vector<double> &x_out, double &resid_out) {
    bngsim::SUNMatrixGuard A(SUNDenseMatrix(n, n, ctx));
    CHECK(A, "SUNDenseMatrix alloc");
    std::copy(A_src, A_src + (long)n * n, SUNDenseMatrix_Data(A));

    bngsim::NVectorGuard bv(N_VNew_Serial(n, ctx));
    bngsim::NVectorGuard xv(N_VNew_Serial(n, ctx));
    CHECK(bv && xv, "N_Vector alloc");
    std::copy(b, b + n, N_VGetArrayPointer(bv));

    bngsim::SUNLinSolGuard LS(bngsim::make_dense_linear_solver(xv, A, ctx, prefer_lapack));
    CHECK(LS, "linear solver alloc");

    CHECK(SUNLinSolInitialize(LS) == SUN_SUCCESS, "initialize");
    int setup = SUNLinSolSetup(LS, A);
    CHECK(setup == SUN_SUCCESS, "setup (factorization) succeeded");
    int solve = SUNLinSolSolve(LS, A, xv, bv, 0.0);
    CHECK(solve == SUN_SUCCESS, "solve succeeded");

    x_out.assign(N_VGetArrayPointer(xv), N_VGetArrayPointer(xv) + n);
    resid_out = residual_inf(A_src, n, x_out.data(), b);
    return 0;
}

// The core test: built-in vs LAPACK on the same well-conditioned system, across
// the corpus-relevant size range. Residuals tiny AND the two solutions agree.
static int test_builtin_vs_lapack_parity() {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");

    const int sizes[] = {1, 2, 50, 195, 295, 400};
    for (int n : sizes) {
        std::vector<double> A((long)n * n);
        fill_well_conditioned(A.data(), n, 999u + (unsigned)n);
        std::vector<double> b(n);
        for (int i = 0; i < n; ++i)
            b[i] = 1.0 + (i % 7);

        std::vector<double> x_builtin, x_lapack;
        double r_builtin = 0.0, r_lapack = 0.0;
        if (solve_with(false, A.data(), b.data(), n, ctx, x_builtin, r_builtin) != 0)
            return 1;
        if (solve_with(true, A.data(), b.data(), n, ctx, x_lapack, r_lapack) != 0)
            return 1;

        CHECK(r_builtin < 1e-11, "built-in residual small");
        CHECK(r_lapack < 1e-11, "lapack-factor residual small");

        double dmax = 0.0;
        for (int i = 0; i < n; ++i)
            dmax = std::max(dmax, std::fabs(x_builtin[i] - x_lapack[i]));
        CHECK(dmax < 1e-11, "built-in and lapack solutions agree");
    }
    return 0;
}

// Adaptive factorization-count gate (GH #132): the custom solver factors with
// the built-in GETRF for its first K calls, then switches to dgetrf. This test
// drives one persistent solver through more than K setup+solve cycles on fresh
// pivoting systems and checks every solve matches an independent built-in solve
// of the same system — exercising the built-in calls (1..K), the switch, and
// the dgetrf calls (K+1..) in one solver, confirming the count increments and
// the mid-run backend swap never changes the answer. (Forces MIN_N=0 so the
// small test systems can take the BLAS path; K small so the switch happens.)
static int test_adaptive_switch() {
    if (!bngsim::lapack_dense_available())
        return 0; // no BLAS backend → solver is the built-in fallback; nothing to switch
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    ::setenv("BNGSIM_LAPACK_DENSE_K", "3", 1);   // built-in for 3 factorizations, then dgetrf
    ::setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0", 1);

    const int n = 64;
    bngsim::SUNMatrixGuard A(SUNDenseMatrix(n, n, ctx));
    CHECK(A, "SUNDenseMatrix alloc");
    bngsim::NVectorGuard bv(N_VNew_Serial(n, ctx));
    bngsim::NVectorGuard xv(N_VNew_Serial(n, ctx));
    CHECK(bv && xv, "N_Vector alloc");
    // One persistent adaptive solver reused across cycles (factor_count lives in it).
    bngsim::SUNLinSolGuard LS(bngsim::make_dense_linear_solver(xv, A, ctx, true));
    CHECK(LS, "adaptive solver alloc");
    CHECK(SUNLinSolInitialize(LS) == SUN_SUCCESS, "initialize");

    const int CYCLES = 8; // 3 built-in + 5 dgetrf
    for (int cyc = 0; cyc < CYCLES; ++cyc) {
        std::vector<double> A_src((long)n * n);
        std::srand(7000u + (unsigned)cyc);
        for (long i = 0; i < (long)n * n; ++i)
            A_src[i] = ((double)std::rand() / RAND_MAX) - 0.5; // non-dominant → pivots
        A_src[0] = 1e-6;
        A_src[1] = 1.0; // force a row swap at step 0
        std::vector<double> b(n);
        for (int i = 0; i < n; ++i)
            b[i] = 1.0 + (i % 5);

        // Adaptive solve (built-in for cyc<3, dgetrf for cyc>=3).
        std::copy(A_src.begin(), A_src.end(), SUNDenseMatrix_Data(A));
        std::copy(b.begin(), b.end(), N_VGetArrayPointer(bv));
        CHECK(SUNLinSolSetup(LS, A) == SUN_SUCCESS, "adaptive setup");
        CHECK(SUNLinSolSolve(LS, A, xv, bv, 0.0) == SUN_SUCCESS, "adaptive solve");
        std::vector<double> x_adaptive(N_VGetArrayPointer(xv), N_VGetArrayPointer(xv) + n);

        // Independent built-in reference solve of the same system.
        std::vector<double> x_ref;
        double r_ref = 0.0;
        if (solve_with(false, A_src.data(), b.data(), n, ctx, x_ref, r_ref) != 0)
            return 1;
        CHECK(r_ref < 1e-6, "reference residual small");
        CHECK(residual_inf(A_src.data(), n, x_adaptive.data(), b.data()) < 1e-6,
              "adaptive residual small across the backend switch");
        double dmax = 0.0;
        for (int i = 0; i < n; ++i)
            dmax = std::max(dmax, std::fabs(x_adaptive[i] - x_ref[i]));
        CHECK(dmax < 1e-9, "adaptive solve matches built-in before and after the switch");
    }
    ::unsetenv("BNGSIM_LAPACK_DENSE_K");
    ::unsetenv("BNGSIM_LAPACK_DENSE_MIN_N");
    return 0;
}

// Counter reset (GH #132): the adaptive gate's factor_count lives in the solver
// and persists across reuse, so the warm CVODE path — which reuses ONE solver
// across run()s via CVodeReInit — must reset it per run with
// lapack_dense_reset_factor_count, else a long prior run leaves the count past K
// and a later short run takes the BLAS factor on its first factorization. This
// drives one persistent solver past the switch, resets, and confirms the count
// zeroes and then climbs again from scratch — and that the reset / accessor are
// safe no-ops on a non-adaptive (built-in) solver. (The backend choice is
// answer-invariant, so the counter is the only observable that proves reset.)
static int test_adaptive_reset() {
    if (!bngsim::lapack_dense_available())
        return 0; // no BLAS backend → no adaptive solver to reset
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    ::setenv("BNGSIM_LAPACK_DENSE_K", "3", 1);
    ::setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0", 1);

    const int n = 32;
    bngsim::SUNMatrixGuard A(SUNDenseMatrix(n, n, ctx));
    CHECK(A, "SUNDenseMatrix alloc");
    bngsim::NVectorGuard bv(N_VNew_Serial(n, ctx));
    bngsim::NVectorGuard xv(N_VNew_Serial(n, ctx));
    CHECK(bv && xv, "N_Vector alloc");
    bngsim::SUNLinSolGuard LS(bngsim::make_dense_linear_solver(xv, A, ctx, true));
    CHECK(LS, "adaptive solver alloc");
    CHECK(SUNLinSolInitialize(LS) == SUN_SUCCESS, "initialize");
    CHECK(bngsim::lapack_dense_factor_count(LS) == 0, "count starts at 0");

    // One setup+solve on a fresh well-conditioned system; only the counter matters.
    auto one_cycle = [&](unsigned seed) -> bool {
        fill_well_conditioned(SUNDenseMatrix_Data(A), n, seed);
        double *bp = N_VGetArrayPointer(bv);
        for (int i = 0; i < n; ++i)
            bp[i] = 1.0 + (i % 5);
        return SUNLinSolSetup(LS, A) == SUN_SUCCESS &&
               SUNLinSolSolve(LS, A, xv, bv, 0.0) == SUN_SUCCESS;
    };

    for (int c = 0; c < 5; ++c)
        CHECK(one_cycle(4100u + (unsigned)c), "solve cycle");
    CHECK(bngsim::lapack_dense_factor_count(LS) == 5, "count tracks factorizations");

    // What run_warm does at each warm re-entry: zero the gate for the next run.
    bngsim::lapack_dense_reset_factor_count(LS);
    CHECK(bngsim::lapack_dense_factor_count(LS) == 0, "reset zeroes the count");

    for (int c = 0; c < 2; ++c)
        CHECK(one_cycle(4200u + (unsigned)c), "post-reset solve cycle");
    CHECK(bngsim::lapack_dense_factor_count(LS) == 2, "count restarts from 0 after reset");

    // The reset / accessor must be harmless no-ops on a non-adaptive solver
    // (the warm path calls reset on whatever solver it holds, incl. built-in).
    bngsim::SUNMatrixGuard A2(SUNDenseMatrix(n, n, ctx));
    bngsim::SUNLinSolGuard builtin(bngsim::make_dense_linear_solver(xv, A2, ctx, false));
    CHECK(builtin, "built-in solver alloc");
    CHECK(bngsim::lapack_dense_factor_count(builtin) == -1, "factor_count = -1 for built-in");
    bngsim::lapack_dense_reset_factor_count(builtin); // must not touch foreign content

    ::unsetenv("BNGSIM_LAPACK_DENSE_K");
    ::unsetenv("BNGSIM_LAPACK_DENSE_MIN_N");
    return 0;
}

// BLAS factorization counter (GH #132): of the run's factorizations, how many
// actually took the dgetrf path. The first K stay on the built-in GETRF, so the
// dgetrf count lags the total by K — this is what lets a benchmark tell a
// LAPACK-dense run that genuinely exercised dgetrf from one that never crossed
// the gate (reported via SolverStats::n_dense_blas_factorizations).
static int test_blas_factor_count() {
    if (!bngsim::lapack_dense_available())
        return 0; // no BLAS backend → no dgetrf path to count
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    ::setenv("BNGSIM_LAPACK_DENSE_K", "3", 1); // built-in for 3, then dgetrf
    ::setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0", 1);

    const int n = 32;
    bngsim::SUNMatrixGuard A(SUNDenseMatrix(n, n, ctx));
    bngsim::NVectorGuard bv(N_VNew_Serial(n, ctx));
    bngsim::NVectorGuard xv(N_VNew_Serial(n, ctx));
    CHECK(A && bv && xv, "alloc");
    bngsim::SUNLinSolGuard LS(bngsim::make_dense_linear_solver(xv, A, ctx, true));
    CHECK(LS, "adaptive solver alloc");
    CHECK(SUNLinSolInitialize(LS) == SUN_SUCCESS, "initialize");
    CHECK(bngsim::lapack_dense_blas_factor_count(LS) == 0, "blas count starts at 0");

    auto one_cycle = [&](unsigned seed) -> bool {
        fill_well_conditioned(SUNDenseMatrix_Data(A), n, seed);
        double *bp = N_VGetArrayPointer(bv);
        for (int i = 0; i < n; ++i)
            bp[i] = 1.0 + (i % 5);
        return SUNLinSolSetup(LS, A) == SUN_SUCCESS &&
               SUNLinSolSolve(LS, A, xv, bv, 0.0) == SUN_SUCCESS;
    };

    // First 3 factorizations are built-in (below K) → blas count stays 0.
    for (int c = 0; c < 3; ++c)
        CHECK(one_cycle(5100u + (unsigned)c), "pre-gate solve");
    CHECK(bngsim::lapack_dense_factor_count(LS) == 3, "3 factorizations so far");
    CHECK(bngsim::lapack_dense_blas_factor_count(LS) == 0, "no dgetrf before the gate");

    // The next 4 cross the gate → 4 dgetrf factorizations.
    for (int c = 0; c < 4; ++c)
        CHECK(one_cycle(5200u + (unsigned)c), "post-gate solve");
    CHECK(bngsim::lapack_dense_factor_count(LS) == 7, "7 factorizations total");
    CHECK(bngsim::lapack_dense_blas_factor_count(LS) == 4, "4 of them via dgetrf");

    // Reset zeroes the BLAS count too.
    bngsim::lapack_dense_reset_factor_count(LS);
    CHECK(bngsim::lapack_dense_blas_factor_count(LS) == 0, "reset zeroes the blas count");

    // -1 (not applicable) on a built-in solver.
    bngsim::SUNMatrixGuard A2(SUNDenseMatrix(n, n, ctx));
    bngsim::SUNLinSolGuard builtin(bngsim::make_dense_linear_solver(xv, A2, ctx, false));
    CHECK(builtin, "built-in solver alloc");
    CHECK(bngsim::lapack_dense_blas_factor_count(builtin) == -1, "blas count = -1 for built-in");

    ::unsetenv("BNGSIM_LAPACK_DENSE_K");
    ::unsetenv("BNGSIM_LAPACK_DENSE_MIN_N");
    return 0;
}

// The gate predicate is OPT-IN (GH #84): default keeps the built-in dense LU
// regardless of size/density; only BNGSIM_LAPACK_DENSE=1 engages the BLAS factor.
static int test_gate_predicate() {
    using bngsim::should_use_lapack_dense;
    if (!bngsim::lapack_dense_available()) {
        ::setenv("BNGSIM_LAPACK_DENSE", "1", 1);
        CHECK(!should_use_lapack_dense(1000, 1.0, true), "no backend → never lapack, even opted in");
        ::unsetenv("BNGSIM_LAPACK_DENSE");
        return 0;
    }
    // Default (no env): built-in regardless of size / density / force_dense.
    ::unsetenv("BNGSIM_LAPACK_DENSE");
    CHECK(!should_use_lapack_dense(1000, 1.0, true), "default → built-in (no auto-engage)");
    CHECK(!should_use_lapack_dense(300, 0.9, false), "default large+dense → built-in");
    // Explicit opt-in engages it regardless of size / density.
    ::setenv("BNGSIM_LAPACK_DENSE", "1", 1);
    CHECK(should_use_lapack_dense(8, 0.01, false), "opt-in → lapack (any size)");
    // Explicit opt-out.
    ::setenv("BNGSIM_LAPACK_DENSE", "0", 1);
    CHECK(!should_use_lapack_dense(1000, 1.0, true), "opt-out → built-in");
    ::unsetenv("BNGSIM_LAPACK_DENSE");
    return 0;
}

// Pivoting parity (closes a real gap): the diagonally-dominant matrices above
// always pivot on the diagonal, so they never exercise the ipiv→pivots rebase.
// These matrices REQUIRE row interchanges. Built-in GETRF and LAPACK dgetrf both
// do partial pivoting and on the same matrix pick the same pivots, so a correct
// rebase makes the two solutions agree to rounding; a mis-rebased pivot would
// permute the solve wrongly → large residual and disagreement.
static int test_pivoting_parity() {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    const int sizes[] = {2, 3, 16, 64, 150};
    for (int n : sizes) {
        std::vector<double> A((long)n * n);
        std::srand(31337u + (unsigned)n);
        for (long i = 0; i < (long)n * n; ++i)
            A[i] = ((double)std::rand() / RAND_MAX) - 0.5; // no diagonal boost
        A[0] = 1e-6; // tiny A(0,0) ...
        A[1] = 1.0;  // ... vs large A(1,0) → forces a row swap at step 0
        std::vector<double> b(n);
        for (int i = 0; i < n; ++i)
            b[i] = 1.0 + (i % 5);

        std::vector<double> xb, xl;
        double rb = 0.0, rl = 0.0;
        if (solve_with(false, A.data(), b.data(), n, ctx, xb, rb) != 0)
            return 1;
        if (solve_with(true, A.data(), b.data(), n, ctx, xl, rl) != 0)
            return 1;
        CHECK(rb < 1e-6, "built-in residual small (pivoting)");
        CHECK(rl < 1e-6, "lapack residual small (pivoting) — rebase correct");
        double dmax = 0.0;
        for (int i = 0; i < n; ++i)
            dmax = std::max(dmax, std::fabs(xb[i] - xl[i]));
        CHECK(dmax < 1e-9, "built-in and lapack agree under row pivoting");
    }
    return 0;
}

int main() {
    std::cout << "test_lapack_dense_linsol (GH #84)\n";
    std::cout << "  lapack_dense_available = " << (bngsim::lapack_dense_available() ? "yes" : "no")
              << "\n";
    // The kernel-parity tests must exercise the BLAS dgetrf path directly, so
    // defeat the GH #132 adaptive gate: K=0 → dgetrf from the first factorization,
    // MIN_N=0 → no small-N exclusion (these tests run N as small as 1). The
    // adaptive switch itself is covered separately by test_adaptive_switch.
    ::setenv("BNGSIM_LAPACK_DENSE_K", "0", 1);
    ::setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0", 1);
    RUN_TEST(test_builtin_vs_lapack_parity);
    RUN_TEST(test_pivoting_parity);
    ::unsetenv("BNGSIM_LAPACK_DENSE_K");
    ::unsetenv("BNGSIM_LAPACK_DENSE_MIN_N");
    RUN_TEST(test_adaptive_switch);
    RUN_TEST(test_adaptive_reset);
    RUN_TEST(test_blas_factor_count);
    RUN_TEST(test_gate_predicate);
    std::cout << tests_passed << "/" << tests_run << " passed\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
