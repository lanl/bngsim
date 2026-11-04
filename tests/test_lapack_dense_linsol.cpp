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
static int tests_skipped = 0;

// A third status alongside pass (0) and fail (1), added by GH #269. Four of the
// cases below open with an early return when the build links no BLAS backend,
// and a plain `return 0` counted every one of them as a PASS: a bare
// ubuntu-latest printed "6/6 passed" in 0.03 s, byte-identical to the summary on
// a host where all six really ran. ctest suppresses stdout on success, so even
// the `lapack_dense_available = no` line never reached the log — the whole leg
// read as coverage it did not have. 77 is ctest's own SKIP_RETURN_CODE
// convention, used here per-case rather than per-binary.
#define SKIP_RC 77

#define SKIP(reason)                                                                               \
    do {                                                                                           \
        std::cout << "SKIPPED (" << reason << ")";                                                 \
        return SKIP_RC;                                                                            \
    } while (0)

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "  FAIL: " << msg << " [" << __FILE__ << ":" << __LINE__ << "]"           \
                      << std::endl;                                                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

#define RUN_TEST(func)                                                                             \
    do {                                                                                           \
        ++tests_run;                                                                               \
        std::cout << "  " << #func << "... " << std::flush;                                        \
        int _rc = func();                                                                          \
        if (_rc == SKIP_RC) {                                                                      \
            ++tests_skipped;                                                                       \
            std::cout << std::endl;                                                                \
        } else if (_rc == 0) {                                                                     \
            ++tests_passed;                                                                        \
            std::cout << "OK" << std::endl;                                                        \
        } else {                                                                                   \
            std::cout << "FAILED" << std::endl;                                                    \
        }                                                                                          \
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
//
// This one used to RUN on a no-BLAS host, and that was the subtler half of the
// GH #269 false green: with no backend, solve_with(true, …) falls back to the
// built-in factor, so both arms of the comparison are the same code and dmax is
// identically 0. It passed by asserting the built-in solver agrees with itself.
// The fallback path it did exercise incidentally is now covered on purpose by
// test_prefer_lapack_fallback_contract below, so this can say plainly that a
// parity test with only one implementation present has nothing to compare.
static int test_builtin_vs_lapack_parity() {
    if (!bngsim::lapack_dense_available())
        SKIP("no BLAS backend — both arms would be the built-in factor");
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
        SKIP(
            "no BLAS backend — the solver factors with the built-in GETRF only, nothing to switch");
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    ::setenv("BNGSIM_LAPACK_DENSE_K", "3", 1); // built-in for 3 factorizations, then dgetrf
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
        SKIP("no BLAS backend — no adaptive solver to reset");
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
        SKIP("no BLAS backend — no dgetrf path to count");
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
//
// Deliberately NOT a SKIP on a no-BLAS host (unlike the four above): the
// no-backend branch asserts something real and host-specific — that opting in
// on a build with no backend still refuses the BLAS path rather than routing to
// a solver that is not there.
static int test_gate_predicate() {
    using bngsim::should_use_lapack_dense;
    if (!bngsim::lapack_dense_available()) {
        ::setenv("BNGSIM_LAPACK_DENSE", "1", 1);
        CHECK(!should_use_lapack_dense(1000, 1.0, true),
              "no backend → never lapack, even opted in");
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
    if (!bngsim::lapack_dense_available())
        SKIP("no BLAS backend — both arms would be the built-in factor");
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    const int sizes[] = {2, 3, 16, 64, 150};
    for (int n : sizes) {
        std::vector<double> A((long)n * n);
        std::srand(31337u + (unsigned)n);
        for (long i = 0; i < (long)n * n; ++i)
            A[i] = ((double)std::rand() / RAND_MAX) - 0.5; // no diagonal boost
        A[0] = 1e-6;                                       // tiny A(0,0) ...
        A[1] = 1.0; // ... vs large A(1,0) → forces a row swap at step 0
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

// The one case with something to assert on EVERY host, and the reason the two
// parity tests above can now skip without losing coverage (GH #269). Asking for
// the BLAS factor is a request, not a requirement: on a build that has none it
// must still hand back a working dense solver — the whole contract of the
// !BNGSIM_HAS_LAPACK_DENSE branch of make_dense_linear_solver — and on a build
// that has one it must hand back the adaptive solver. The accessors agree with
// lapack_dense_available() about which solver came back: -1 ("not applicable")
// from the plain SUNDIALS solver on a no-backend build, a real count from the
// adaptive one otherwise. A build that linked a backend but still returned the
// built-in solver would look identical from the outside without this.
static int test_prefer_lapack_fallback_contract() {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    const int n = 24;
    std::vector<double> A((long)n * n);
    fill_well_conditioned(A.data(), n, 8123u);
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i)
        b[i] = 1.0 + (i % 3);

    std::vector<double> x;
    double resid = 0.0;
    // prefer_lapack = true on any host: with a backend this is the adaptive
    // solver, without one it must degrade to the built-in GETRF, not fail.
    if (solve_with(true, A.data(), b.data(), n, ctx, x, resid) != 0)
        return 1;
    CHECK(resid < 1e-11, "prefer_lapack solve is accurate on every build");

    bngsim::SUNMatrixGuard A2(SUNDenseMatrix(n, n, ctx));
    bngsim::NVectorGuard yv(N_VNew_Serial(n, ctx));
    CHECK(A2 && yv, "alloc");
    bngsim::SUNLinSolGuard LS(bngsim::make_dense_linear_solver(yv, A2, ctx, true));
    CHECK(LS, "solver alloc");
    if (bngsim::lapack_dense_available()) {
        CHECK(bngsim::lapack_dense_factor_count(LS) == 0,
              "backend present → adaptive solver, count starts at 0");
    } else {
        CHECK(bngsim::lapack_dense_factor_count(LS) == -1,
              "no backend → the solver on the built-in factor, count not applicable");
        CHECK(bngsim::lapack_dense_blas_factor_count(LS) == -1,
              "no backend → no dgetrf count either");
    }
    return 0;
}

// ── A row the matrix holds alone ────────────────────────────────────────────
// Row r has no nonzero off its diagonal, so x_r = b_r / a_rr, and 0 where b_r is.
// Column r carries a dominant entry on another row, so partial pivoting takes
// that row as the pivot for column r and the back-solve computes x_r as the
// remainder of sums that cancel: SUNDIALS' own dense LU leaves their rounding in
// x_r, which on an exact, unstable state (MODEL1812040005's P = 0 with its
// species reversed) grew to 25.6x the trajectory's peak.
static void fill_with_row_held_alone(double *data, int n, int r, double b_r, double *b,
                                     unsigned seed) {
    fill_well_conditioned(data, n, seed);
    for (int j = 0; j < n; ++j)
        if (j != r)
            data[(long)j * n + r] = 0.0; // row r: the diagonal alone
    data[(long)r * n + r] = 0.75;
    data[(long)r * n + (r + 1) % n] = 1e8 * (1.0 + (seed % 7)); // pivot for column r
    for (int i = 0; i < n; ++i)
        b[i] = ((double)std::rand() / RAND_MAX) - 0.5;
    b[r] = b_r;
}

// One solve of A x = b with the given solver; x is written to x_out.
static int solve_on(SUNLinearSolver LS, const double *A_src, const double *b, int n, SUNContext ctx,
                    std::vector<double> &x_out) {
    bngsim::SUNMatrixGuard A(SUNDenseMatrix(n, n, ctx));
    bngsim::NVectorGuard bv(N_VNew_Serial(n, ctx));
    bngsim::NVectorGuard xv(N_VNew_Serial(n, ctx));
    CHECK(A && bv && xv, "alloc");
    std::copy(A_src, A_src + (long)n * n, SUNDenseMatrix_Data(A));
    std::copy(b, b + n, N_VGetArrayPointer(bv));
    CHECK(SUNLinSolInitialize(LS) == SUN_SUCCESS, "initialize");
    CHECK(SUNLinSolSetup(LS, A) == SUN_SUCCESS, "setup");
    CHECK(SUNLinSolSolve(LS, A, xv, bv, 0.0) == SUN_SUCCESS, "solve");
    x_out.assign(N_VGetArrayPointer(xv), N_VGetArrayPointer(xv) + n);
    return 0;
}

static bool same_bits(double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); }

// The solver gives the row held alone its own answer to the bit, and every other
// component as SUNDIALS' dense LU gives it (on the built-in factor; the BLAS
// factor's other components are dgetrf's). The stock solver has to leave a
// residue in some trial, or the case is not exercised.
static int rows_held_alone_read_exactly(bool blas) {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    int stock_left_a_residue = 0;
    const int sizes[] = {2, 3, 8, 30};
    for (int n : sizes) {
        for (unsigned trial = 0; trial < 40; ++trial) {
            const int r = (int)(trial % (unsigned)n);
            const double b_r = (trial % 2 == 0) ? 0.0 : 0.3 - 0.01 * trial;
            std::vector<double> A((long)n * n), b(n);
            fill_with_row_held_alone(A.data(), n, r, b_r, b.data(), 6100u + trial + 97u * n);

            ::setenv("BNGSIM_LAPACK_DENSE_K", "0", 1); // the BLAS arm factors with dgetrf
            ::setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0", 1);
            bngsim::SUNMatrixGuard M(SUNDenseMatrix(n, n, ctx));
            bngsim::NVectorGuard y(N_VNew_Serial(n, ctx));
            bngsim::SUNLinSolGuard ours(bngsim::make_dense_linear_solver(y, M, ctx, blas));
            bngsim::SUNLinSolGuard stock(SUNLinSol_Dense(y, M, ctx));
            ::unsetenv("BNGSIM_LAPACK_DENSE_K");
            ::unsetenv("BNGSIM_LAPACK_DENSE_MIN_N");
            CHECK(ours && stock, "solver alloc");

            std::vector<double> x, x_stock;
            if (solve_on(ours, A.data(), b.data(), n, ctx, x) != 0 ||
                solve_on(stock, A.data(), b.data(), n, ctx, x_stock) != 0)
                return 1;
            CHECK(x[r] == b_r / 0.75, "the row held alone gives x_r = b_r / a_rr exactly");
            if (x_stock[r] != x[r])
                ++stock_left_a_residue;
            if (!blas) {
                for (int i = 0; i < n; ++i)
                    CHECK(i == r || same_bits(x[i], x_stock[i]),
                          "every other component is SUNDIALS' dense LU's to the bit");
            }
            CHECK(residual_inf(A.data(), n, x.data(), b.data()) < 1e-6, "residual small");
        }
    }
    CHECK(stock_left_a_residue > 0, "the stock dense LU leaves a residue in some trial");
    return 0;
}

static int test_a_row_held_alone_is_read_exactly() { return rows_held_alone_read_exactly(false); }

static int test_a_row_held_alone_is_read_exactly_on_the_blas_factor() {
    if (!bngsim::lapack_dense_available())
        SKIP("no BLAS backend — no dgetrf factor to read it on");
    return rows_held_alone_read_exactly(true);
}

// The control: on a matrix with no row held alone, and on one whose only
// off-diagonal entry in a row is NaN (a NaN is not a zero), the built-in-factor
// solver is SUNDIALS' dense LU to the bit.
static int test_without_a_row_held_alone_the_solve_is_sunlinsol_dense() {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    for (int n : {1, 2, 5, 40}) {
        for (bool with_nan : {false, true}) {
            if (with_nan && n < 2)
                continue;
            std::vector<double> A((long)n * n), b(n);
            std::srand(7300u + (unsigned)n);
            for (long i = 0; i < (long)n * n; ++i)
                A[i] = ((double)std::rand() / RAND_MAX) - 0.5; // pivots
            for (int i = 0; i < n; ++i)
                b[i] = 1.0 + (i % 3);
            if (with_nan) {
                for (int j = 0; j < n; ++j)
                    A[(long)j * n + 0] = (j == 0) ? 1.0 : 0.0;
                A[(long)1 * n + 0] = std::nan("");
            }
            bngsim::SUNMatrixGuard M(SUNDenseMatrix(n, n, ctx));
            bngsim::NVectorGuard y(N_VNew_Serial(n, ctx));
            bngsim::SUNLinSolGuard ours(bngsim::make_dense_linear_solver(y, M, ctx, false));
            bngsim::SUNLinSolGuard stock(SUNLinSol_Dense(y, M, ctx));
            CHECK(ours && stock, "solver alloc");
            std::vector<double> x, x_stock;
            if (solve_on(ours, A.data(), b.data(), n, ctx, x) != 0 ||
                solve_on(stock, A.data(), b.data(), n, ctx, x_stock) != 0)
                return 1;
            for (int i = 0; i < n; ++i)
                CHECK(same_bits(x[i], x_stock[i]), "the same solve as SUNLinSol_Dense");
        }
    }
    return 0;
}

// ── Rows at 0 that read only each other (#1485) ─────────────────────────────
// Rows 0..L-1 read only each other off their diagonals. In a chain row k reads
// row k-1 (row 0 is held alone); in a cycle row k reads row k+1 and row L-1
// reads row 0. Each one's column has a dominant entry on a row past them, which
// partial pivoting takes as the pivot, as in #1485's models with the species
// reversed (P' = kP, Q' = kQ + P read by V' = 1 + 1e8·Q - V, and the cycle
// P' = Q' = k(P + Q)). The rows past them are dense, and read every column.
// Where their right-hand side is 0, so are their components, exactly.
enum class Held { chain, cycle };

static void fill_with_rows_held_together(double *data, int n, int L, Held shape, double *b,
                                         unsigned seed) {
    fill_well_conditioned(data, n, seed);
    for (int k = 0; k < L; ++k) {
        for (int j = 0; j < n; ++j)
            data[(long)j * n + k] = 0.0;
        data[(long)k * n + k] = 0.75;
        if (shape == Held::chain && k > 0) // below the diagonal, or a long chain is ill-conditioned
            data[(long)(k - 1) * n + k] = 0.5 + 0.005 * k;
        if (shape == Held::cycle && L > 1)
            data[(long)((k + 1) % L) * n + k] = 1.0 + 0.1 * k;
        data[(long)k * n + L + k % (n - L)] = 1e8 * (1.0 + k); // pivot for column k
    }
    for (int i = 0; i < n; ++i)
        b[i] = (i < L) ? 0.0 : ((double)std::rand() / RAND_MAX) - 0.5;
}

// The exact answer where rows 0..L-1 are at 0: x_k = 0 there, and the rest is
// the solve of the rows past them on their own columns, which read the first L
// columns only times 0. Solved by SUNDIALS' dense LU, on a matrix with no row
// at 0 to leave a residue in.
static int exact_past_zero(const std::vector<double> &A, const std::vector<double> &b, int n,
                           int L, SUNContext ctx, std::vector<double> &x_exact) {
    const int r = n - L;
    std::vector<double> Ar((long)r * r), br(b.begin() + L, b.end()), xr;
    for (int j = 0; j < r; ++j)
        for (int i = 0; i < r; ++i)
            Ar[(long)j * r + i] = A[(long)(j + L) * n + (i + L)];
    bngsim::SUNMatrixGuard M(SUNDenseMatrix(r, r, ctx));
    bngsim::NVectorGuard y(N_VNew_Serial(r, ctx));
    bngsim::SUNLinSolGuard stock(SUNLinSol_Dense(y, M, ctx));
    CHECK(stock, "solver alloc");
    if (solve_on(stock, Ar.data(), br.data(), r, ctx, xr) != 0)
        return 1;
    x_exact.assign(L, 0.0);
    x_exact.insert(x_exact.end(), xr.begin(), xr.end());
    return 0;
}

static double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b, int from) {
    double m = 0.0;
    for (size_t i = from; i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Solve with this solver and with SUNDIALS' own on the same matrix.
static int solve_both(const std::vector<double> &A, const std::vector<double> &b, int n,
                      bool blas, SUNContext ctx, std::vector<double> &x,
                      std::vector<double> &x_stock) {
    ::setenv("BNGSIM_LAPACK_DENSE_K", "0", 1); // the BLAS arm factors with dgetrf
    ::setenv("BNGSIM_LAPACK_DENSE_MIN_N", "0", 1);
    bngsim::SUNMatrixGuard M(SUNDenseMatrix(n, n, ctx));
    bngsim::NVectorGuard y(N_VNew_Serial(n, ctx));
    bngsim::SUNLinSolGuard ours(bngsim::make_dense_linear_solver(y, M, ctx, blas));
    bngsim::SUNLinSolGuard stock(SUNLinSol_Dense(y, M, ctx));
    ::unsetenv("BNGSIM_LAPACK_DENSE_K");
    ::unsetenv("BNGSIM_LAPACK_DENSE_MIN_N");
    CHECK(ours && stock, "solver alloc");
    if (solve_on(ours, A.data(), b.data(), n, ctx, x) != 0 ||
        solve_on(stock, A.data(), b.data(), n, ctx, x_stock) != 0)
        return 1;
    return 0;
}

// The rows at 0 give exactly 0, and every other component is SUNDIALS' dense
// LU's to the bit (on the built-in factor; on the BLAS factor dgetrf's, no
// further from the exact answer than SUNDIALS' are). The stock solver has to
// leave a residue in some trial, or the case is not exercised. A cycle of 40
// rows is read as a cycle of 2 is: there is no bound on how many rows read each
// other. The residual of the rows past them is not a measure here: the LU's
// components there are what they were, and were computed against its residue
// at the rows at 0, which a dominant entry multiplies; their error is what it
// was.
static int rows_held_together_read_exactly(bool blas) {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    int stock_left_a_residue = 0;
    for (Held shape : {Held::chain, Held::cycle}) {
        for (int L : {2, 3, 16, 40}) {
            for (int past : {3, 30}) {
                const int n = L + past;
                for (unsigned trial = 0; trial < 10; ++trial) {
                    std::vector<double> A((long)n * n), b(n), x, x_stock;
                    fill_with_rows_held_together(A.data(), n, L, shape, b.data(),
                                                 8100u + trial + 31u * n + 7u * L);
                    if (solve_both(A, b, n, blas, ctx, x, x_stock) != 0)
                        return 1;
                    for (int k = 0; k < L; ++k) {
                        CHECK(x[k] == 0.0, "a row held at 0 with others gives exactly 0");
                        if (x_stock[k] != 0.0)
                            ++stock_left_a_residue;
                    }
                    if (!blas)
                        for (int i = L; i < n; ++i)
                            CHECK(same_bits(x[i], x_stock[i]),
                                  "every other component is SUNDIALS' dense LU's to the bit");
                    std::vector<double> x_exact;
                    if (exact_past_zero(A, b, n, L, ctx, x_exact) != 0)
                        return 1;
                    const double scale = 1.0 + max_abs_diff(x_exact, std::vector<double>(n), 0);
                    // dgetrf's components are as far from it as SUNDIALS' are, about
                    // 1e-9 here, one or the other ahead trial by trial.
                    const double bound = blas ? 1e-8 * scale
                                              : std::max(2.0 * max_abs_diff(x_stock, x_exact, L),
                                                         1e-12 * scale);
                    CHECK(max_abs_diff(x, x_exact, L) <= bound,
                          "no component is further from the exact answer than an LU's");
                }
            }
        }
    }
    CHECK(stock_left_a_residue > 0, "the stock dense LU leaves a residue in some trial");
    return 0;
}

static int test_rows_at_zero_that_read_each_other_are_read_exactly() {
    return rows_held_together_read_exactly(false);
}

static int test_rows_at_zero_that_read_each_other_are_read_exactly_on_the_blas_factor() {
    if (!bngsim::lapack_dense_available())
        SKIP("no BLAS backend — no dgetrf factor to read it on");
    return rows_held_together_read_exactly(true);
}

// A row whose right-hand side is 0 but that reads a row whose answer is not, is
// not 0, and neither is a row that reads it: in a chain 0 <- 1 <- 2 <- 3, b is
// nonzero on row `lead` alone, so rows lead.. are the LU's to the bit and rows
// before it are 0. Nothing is read where b is nonzero on every row of a block,
// so the solve is SUNDIALS' to the bit everywhere.
static int test_a_row_at_zero_that_reads_a_row_off_zero_is_the_lus() {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    const int L = 4, n = L + 30;
    for (int lead = 0; lead <= L; ++lead) { // lead == L: no row of the chain off 0
        for (unsigned trial = 0; trial < 10; ++trial) {
            std::vector<double> A((long)n * n), b(n), x, x_stock;
            fill_with_rows_held_together(A.data(), n, L, Held::chain, b.data(), 9600u + trial);
            if (lead < L)
                b[lead] = 0.3 + 0.1 * trial;
            if (solve_both(A, b, n, false, ctx, x, x_stock) != 0)
                return 1;
            for (int k = 0; k < L; ++k) {
                if (k < lead)
                    CHECK(x[k] == 0.0, "a row reading only rows at 0 gives exactly 0");
                else if (k == 0)
                    CHECK(x[k] == b[0] / 0.75, "row 0 is held alone: b / a, as before");
                else
                    CHECK(same_bits(x[k], x_stock[k]), "a row reading a row off 0 is the LU's");
            }
            for (int i = L; i < n; ++i)
                CHECK(same_bits(x[i], x_stock[i]), "every other component is the LU's");
        }
    }
    // Every row's right-hand side nonzero: the solve is SUNDIALS' everywhere.
    for (Held shape : {Held::chain, Held::cycle}) {
        std::vector<double> A((long)n * n), b(n), x, x_stock;
        fill_with_rows_held_together(A.data(), n, L, shape, b.data(), 9700u);
        for (int k = 0; k < L; ++k)
            b[k] = 0.25 + 0.5 * k;
        if (solve_both(A, b, n, false, ctx, x, x_stock) != 0)
            return 1;
        for (int i = 0; i < n; ++i)
            CHECK(i < 1 || same_bits(x[i], x_stock[i]), "no 0 to read: the LU's to the bit");
    }
    return 0;
}

// The bound. A row with 16 entries off its diagonal, each on a row held alone at
// 0, is read, and so is a row at 0 that reads it; one with 17 is not, nor is
// its reader, and both are the LU's to the bit. Rows 0..16 are held alone,
// row 17 reads rows 0..15 (or 0..16), and row 18 reads row 17.
static int test_a_row_past_the_bound_is_left_to_the_lu() {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    const int X = 17, Y = 18, n = Y + 1 + 30;
    for (int reads : {16, 17}) {
        std::vector<double> A((long)n * n), b(n), x, x_stock;
        fill_with_rows_held_together(A.data(), n, Y + 1, Held::chain, b.data(), 9500u);
        for (int k = 0; k <= Y; ++k)
            for (int j = 0; j < n; ++j)
                if (j != k)
                    A[(long)j * n + k] = 0.0;
        for (int j = 0; j < reads; ++j)
            A[(long)j * n + X] = 1.0 + 0.01 * j;
        A[(long)X * n + Y] = 2.0;
        if (solve_both(A, b, n, false, ctx, x, x_stock) != 0)
            return 1;
        for (int k = 0; k < X; ++k)
            CHECK(x[k] == 0.0, "a row held alone is read either way");
        for (int k : {X, Y}) {
            if (reads == 16) {
                CHECK(x[k] == 0.0, "within the bound, the row and its reader are 0 exactly");
            } else {
                CHECK(x_stock[k] != 0.0, "the stock LU leaves a residue there");
                CHECK(same_bits(x[k], x_stock[k]), "past it, the LU's to the bit");
            }
        }
    }
    return 0;
}

// One solver, factoring matrices whose pattern changes from one to the next, as
// a Newton matrix's does where an entry reaches or leaves 0. The solver keeps
// what it found of the pattern while it holds; each answer must be the one a
// fresh solver gives. A chain, rows 0 <- 1 <- 2, and row 3 held alone off 0,
// ahead of 30 dense rows; then row 1 reads a dense row; then the chain again;
// then row 2 reads row 3 in place of row 1 (as many entries, so only where they
// lie tells it from the chain: it is off 0 now); then rows 0..16 held alone
// with row 17 reading 17 of them, 16, and 17 again (a row that drops to the
// bound).
static int test_a_pattern_that_changes_is_read_anew() {
    bngsim::SunContextGuard ctx;
    CHECK(ctx, "SUNContext_Create");
    const int n = 19 + 30;
    enum Step { chain, reads_dense, moved, wide17, wide16 };
    bngsim::SUNMatrixGuard M(SUNDenseMatrix(n, n, ctx));
    bngsim::NVectorGuard y(N_VNew_Serial(n, ctx));
    bngsim::SUNLinSolGuard kept(bngsim::make_dense_linear_solver(y, M, ctx, false));
    CHECK(kept, "solver alloc");
    for (Step step : {chain, reads_dense, chain, moved, chain, wide17, wide16, wide17, chain}) {
        for (bool zero : {true, false}) {
            std::vector<double> A((long)n * n), b(n);
            const bool wide = step == wide17 || step == wide16;
            fill_with_rows_held_together(A.data(), n, wide ? 19 : 4, Held::chain, b.data(), 9400u);
            if (!wide) {
                A[(long)2 * n + 3] = 0.0; // row 3 held alone, off 0
                b[3] = 0.7;
            }
            if (wide) {
                for (int k = 0; k < 19; ++k)
                    for (int j = 0; j < n; ++j)
                        if (j != k)
                            A[(long)j * n + k] = 0.0;
                for (int j = 0; j < (step == wide17 ? 17 : 16); ++j)
                    A[(long)j * n + 17] = 1.0 + 0.01 * j;
                A[(long)17 * n + 18] = 2.0;
            }
            if (step == reads_dense)
                A[(long)(4 + 5) * n + 1] = 0.5;
            if (step == moved) {
                A[(long)1 * n + 2] = 0.0;
                A[(long)3 * n + 2] = 1.3;
            }
            if (!zero)
                b[0] = 0.5; // the chain's head off 0: nothing of the chain is 0
            bngsim::SUNLinSolGuard fresh(bngsim::make_dense_linear_solver(y, M, ctx, false));
            CHECK(fresh, "solver alloc");
            std::vector<double> x, x_fresh;
            if (solve_on(kept, A.data(), b.data(), n, ctx, x) != 0 ||
                solve_on(fresh, A.data(), b.data(), n, ctx, x_fresh) != 0)
                return 1;
            for (int i = 0; i < n; ++i)
                CHECK(same_bits(x[i], x_fresh[i]), "the kept solver gives a fresh one's answer");
            if (zero && step == chain)
                for (int k = 0; k < 3; ++k)
                    CHECK(x[k] == 0.0, "the chain at 0 is read exactly");
            if (zero && step == moved)
                CHECK(x[2] != 0.0, "a row that now reads a row off 0 is not 0");
            if (zero && step == wide16)
                CHECK(x[17] == 0.0 && x[18] == 0.0, "at the bound, read exactly");
        }
    }
    return 0;
}

int main() {
    const bool have_blas = bngsim::lapack_dense_available();
    std::cout << "test_lapack_dense_linsol (GH #84)\n";
    std::cout << "  lapack_dense_available = " << (have_blas ? "yes" : "no") << "\n";
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
    RUN_TEST(test_prefer_lapack_fallback_contract);
    RUN_TEST(test_a_row_held_alone_is_read_exactly);
    RUN_TEST(test_a_row_held_alone_is_read_exactly_on_the_blas_factor);
    RUN_TEST(test_without_a_row_held_alone_the_solve_is_sunlinsol_dense);
    RUN_TEST(test_rows_at_zero_that_read_each_other_are_read_exactly);
    RUN_TEST(test_rows_at_zero_that_read_each_other_are_read_exactly_on_the_blas_factor);
    RUN_TEST(test_a_row_at_zero_that_reads_a_row_off_zero_is_the_lus);
    RUN_TEST(test_a_row_past_the_bound_is_left_to_the_lu);
    RUN_TEST(test_a_pattern_that_changes_is_read_anew);
    std::cout << tests_passed << "/" << tests_run << " passed";
    if (tests_skipped > 0)
        std::cout << ", " << tests_skipped << " SKIPPED (no BLAS dense backend)";
    std::cout << "\n";
    // GH #269: a skip is not a failure — a host with no BLAS is a supported
    // configuration and the exit code has to stay 0 there, or native-tests.yml's
    // stock ubuntu leg goes permanently red for a build option it never asked
    // for. The skip COUNT on the summary line is the signal instead; the leg
    // that is supposed to have a backend asserts it is zero.
    return (tests_passed + tests_skipped == tests_run) ? 0 : 1;
}
