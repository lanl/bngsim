// bngsim/src/lapack_dense_linsol.cpp — GH #84, adaptive gate GH #132
//
// Custom direct SUNLinearSolver with an adaptive factorization backend: it
// factors with either SUNDIALS' built-in denseGETRF or the BLAS dgetrf (chosen
// per-factorization by the GH #132 count gate) and always back-solves with the
// built-in denseGETRS. See lapack_dense_linsol.hpp for the rationale.
//
// Correctness contract (verified by tests/test_lapack_dense_linsol.cpp):
//   LAPACK dgetrf and SUNDIALS denseGETRF produce the SAME in-place L/U storage
//   (U on/above the diagonal, unit-lower multipliers below, column-major) and
//   the SAME sequential row-swap pivot semantics. They differ ONLY in pivot
//   base — LAPACK IPIV is 1-based int, SUNDIALS pivots are 0-based sunindextype.
//   So factoring with dgetrf and back-solving with denseGETRS is bit-equivalent
//   to the all-built-in path once the pivots are rebased (ipiv[k] - 1), which is
//   what lets the solver switch backends mid-run without changing the answer.

#include "bngsim/lapack_dense_linsol.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_dense.h> // SUNDlsMat_denseGETRS
#include <sundials/sundials_math.h>  // SUN_RCONST
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace bngsim {

namespace {

// BNGSIM_LAPACK_DENSE opt-in switch: "1"/"on"/"force" → on, anything else → off.
// Read fresh each call (getenv is negligible next to a factorization); reading
// fresh lets a benchmark A/B the two backends across process invocations.
bool env_opt_in() {
    const char *v = std::getenv("BNGSIM_LAPACK_DENSE");
    if (!v || !*v)
        return false;
    const std::string s(v);
    return !(s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE");
}

// ── Adaptive factorization-count gate (GH #132) ─────────────────────────────
// The custom solver delegates its first `kDefaultSwitchAfter` factorizations to
// the built-in dense GETRF and only switches to the BLAS dgetrf once a run has
// refactored more often than that — the signature of a stiff / long-horizon,
// factorization-bound integration where the O(N^3) factor finally dominates.
// `kDefaultMinN` keeps tiny systems on the built-in factor regardless of count
// (below it, dgetrf is pure call overhead). See the gate rationale below and
// dev/notes/gh84_lapack_dense_findings.md for the tuning data.
//
// Both are overridable via env (BNGSIM_LAPACK_DENSE_K / _MIN_N) so the large
// dense corpus can be re-swept to confirm the crossover; the defaults are the
// values that gate is shipped with.
constexpr long kDefaultSwitchAfter = 5; // built-in for the first 5 factorizations
constexpr long kDefaultMinN = 256;      // never take the BLAS path below this N

long env_long(const char *name, long fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v)
        return fallback;
    char *end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || parsed < 0)
        return fallback;
    return parsed;
}

} // namespace

bool lapack_dense_available() {
#ifdef BNGSIM_HAS_LAPACK_DENSE
    return true;
#else
    return false;
#endif
}

bool should_use_lapack_dense(int n, double density, bool force_dense) {
    // Opt-in only (see lapack_dense_linsol.hpp): end-to-end benchmarking showed
    // no reliable win and some regressions, so the auto path keeps the built-in
    // dense LU. n/density/force_dense are unused under this policy but kept in
    // the signature for the call sites and any future re-tuning.
    (void)n;
    (void)density;
    (void)force_dense;
    return lapack_dense_available() && env_opt_in();
}

#ifdef BNGSIM_HAS_LAPACK_DENSE

#ifdef BNGSIM_LAPACK_DENSE_INT64
using lapack_int = std::int64_t;
#else
using lapack_int = int;
#endif

// LAPACK Fortran GETRF. CMake defines lapack_int to match the selected LAPACK
// library's integer ABI (LP64 uses int; ILP64 uses int64_t). Keeping this as a
// solver-local type still avoids changing SUNDIALS' global index size.
extern "C" void dgetrf_(const lapack_int *m, const lapack_int *n, double *a, const lapack_int *lda,
                        lapack_int *ipiv, lapack_int *info);

namespace {

// Solver content: SUNDIALS 0-based pivots consumed by denseGETRS, plus a scratch
// LAPACK 1-based pivot array filled by dgetrf, plus the GH #132 adaptive-gate
// state (how many factorizations have run, and the switch thresholds).
struct LapackDenseContent {
    sunindextype N;
    sunindextype *pivots;   // 0-based (denseGETRS convention), size N
    lapack_int *ipiv;       // 1-based LAPACK scratch, size N
    long factor_count;      // # ls_setup calls so far (GH #132)
    long blas_factor_count; // # of those that actually took the BLAS dgetrf path
    long switch_after;      // K: built-in for the first K factorizations, then dgetrf
    long min_n;             // never use the BLAS factor below this N
    sunindextype last_flag;
};

SUNLinearSolver_Type ls_get_type(SUNLinearSolver) { return SUNLINEARSOLVER_DIRECT; }
SUNLinearSolver_ID ls_get_id(SUNLinearSolver) { return SUNLINEARSOLVER_CUSTOM; }

SUNErrCode ls_initialize(SUNLinearSolver S) {
    // Deliberately does NOT reset factor_count. CVODE calls this from
    // cvInitialSetup on every (re)init — including each mid-integration
    // CVodeReInit at an event crossing on the cold path. Zeroing the count here
    // would restart the gate at every event, so a genuinely factorization-bound
    // run punctuated by events could never accumulate past K. The counter must
    // span one whole run instead: the cold path rebuilds the solver per run (so
    // it starts at 0 naturally and accumulates across events within the run),
    // and the warm reuse path resets it explicitly per run via
    // lapack_dense_reset_factor_count (see CvodeSimulator::Impl::run_warm).
    static_cast<LapackDenseContent *>(S->content)->last_flag = SUN_SUCCESS;
    return SUN_SUCCESS;
}

int ls_setup(SUNLinearSolver S, SUNMatrix A) {
    auto *c = static_cast<LapackDenseContent *>(S->content);
    const sunindextype rows = SUNDenseMatrix_Rows(A);
    ++c->factor_count;

    // Adaptive backend choice (GH #132). The first `switch_after` factorizations
    // run the built-in dense GETRF — byte-identical to SUNLinSol_Dense — so a
    // few-step model pays no BLAS overhead and cannot regress. Only once the run
    // has refactored more than that (a stiff / long-horizon, factorization-bound
    // integration) does it switch to the blocked BLAS dgetrf, where the O(N^3)
    // factor dominates and dgetrf wins. A minimum N keeps tiny systems on the
    // built-in factor no matter how often they refactor (dgetrf there is pure
    // call overhead). The back-solve is the built-in denseGETRS either way, so
    // the in-place L/U storage and 0-based pivots are consumed identically.
    const bool use_blas = c->factor_count > c->switch_after && rows >= c->min_n;
    if (use_blas)
        ++c->blas_factor_count;

    if (!use_blas) {
        // Built-in dense LU: writes the same in-place L/U and 0-based pivots that
        // denseGETRS consumes. Return matches SUNLinSol_Dense's last_flag (0 ok;
        // >0 is the 1-based column of a zero pivot → singular).
        c->last_flag = SUNDlsMat_denseGETRF(SUNDenseMatrix_Cols(A), rows, rows, c->pivots);
        return c->last_flag != 0 ? SUNLS_LUFACT_FAIL : SUN_SUCCESS;
    }

    if constexpr (sizeof(lapack_int) < sizeof(sunindextype)) {
        if (rows > static_cast<sunindextype>(std::numeric_limits<lapack_int>::max())) {
            c->last_flag = SUNLS_PACKAGE_FAIL_REC;
            return SUNLS_PACKAGE_FAIL_REC;
        }
    }
    lapack_int n = static_cast<lapack_int>(rows);
    lapack_int lda = n; // dense SUNMatrix stores columns contiguously: leading dim = rows
    lapack_int info = 0;
    // dgetrf factors the column-major data in place; denseGETRS then reads the
    // same storage through SUNDenseMatrix_Cols(A).
    dgetrf_(&n, &n, SUNDenseMatrix_Data(A), &lda, c->ipiv, &info);
    // Rebase LAPACK's 1-based row-swap pivots to denseGETRS' 0-based convention.
    for (lapack_int k = 0; k < n; ++k)
        c->pivots[k] = static_cast<sunindextype>(c->ipiv[k] - 1);
    c->last_flag = static_cast<sunindextype>(info);
    // Mirror SUNLinSol_Dense: any nonzero info is a factorization failure. info>0
    // is a zero pivot (U(info,info)=0, singular); info<0 is an illegal argument,
    // which cannot occur here (n, lda, and the data pointer are always valid).
    if (info != 0)
        return SUNLS_LUFACT_FAIL;
    return SUN_SUCCESS;
}

int ls_solve(SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, sunrealtype /*tol*/) {
    auto *c = static_cast<LapackDenseContent *>(S->content);
    N_VScale(SUN_RCONST(1.0), b, x); // copy b into x, exactly as SUNLinSol_Dense does
    SUNDlsMat_denseGETRS(SUNDenseMatrix_Cols(A), SUNDenseMatrix_Rows(A), c->pivots,
                         N_VGetArrayPointer(x));
    c->last_flag = SUN_SUCCESS;
    return SUN_SUCCESS;
}

sunindextype ls_last_flag(SUNLinearSolver S) {
    return static_cast<LapackDenseContent *>(S->content)->last_flag;
}

SUNErrCode ls_space(SUNLinearSolver S, long int *lenrwLS, long int *leniwLS) {
    auto *c = static_cast<LapackDenseContent *>(S->content);
    *leniwLS = 2 + 2 * static_cast<long int>(c->N); // pivots + ipiv
    *lenrwLS = 0;
    return SUN_SUCCESS;
}

SUNErrCode ls_free(SUNLinearSolver S) {
    if (S == nullptr)
        return SUN_SUCCESS;
    if (S->content) {
        auto *c = static_cast<LapackDenseContent *>(S->content);
        std::free(c->pivots);
        std::free(c->ipiv);
        std::free(c);
        S->content = nullptr;
    }
    if (S->ops) {
        std::free(S->ops);
        S->ops = nullptr;
    }
    std::free(S);
    return SUN_SUCCESS;
}

SUNLinearSolver make_lapack_dense(SUNMatrix A, SUNContext ctx) {
    SUNLinearSolver S = SUNLinSolNewEmpty(ctx);
    if (S == nullptr)
        return nullptr;

    S->ops->gettype = ls_get_type;
    S->ops->getid = ls_get_id;
    S->ops->initialize = ls_initialize;
    S->ops->setup = ls_setup;
    S->ops->solve = ls_solve;
    S->ops->lastflag = ls_last_flag;
    S->ops->space = ls_space;
    S->ops->free = ls_free;

    auto *c = static_cast<LapackDenseContent *>(std::malloc(sizeof(LapackDenseContent)));
    if (c == nullptr) {
        SUNLinSolFreeEmpty(S);
        return nullptr;
    }
    const sunindextype n = SUNDenseMatrix_Rows(A);
    c->N = n;
    c->factor_count = 0;
    c->blas_factor_count = 0;
    c->switch_after = env_long("BNGSIM_LAPACK_DENSE_K", kDefaultSwitchAfter);
    c->min_n = env_long("BNGSIM_LAPACK_DENSE_MIN_N", kDefaultMinN);
    c->last_flag = 0;
    c->pivots =
        static_cast<sunindextype *>(std::malloc(static_cast<size_t>(n) * sizeof(sunindextype)));
    c->ipiv = static_cast<lapack_int *>(std::malloc(static_cast<size_t>(n) * sizeof(lapack_int)));
    if (c->pivots == nullptr || c->ipiv == nullptr) {
        std::free(c->pivots);
        std::free(c->ipiv);
        std::free(c);
        SUNLinSolFreeEmpty(S);
        return nullptr;
    }
    S->content = c;
    return S;
}

} // namespace

SUNLinearSolver make_dense_linear_solver(N_Vector y, SUNMatrix A, SUNContext ctx,
                                         bool prefer_lapack) {
    if (prefer_lapack) {
        SUNLinearSolver S = make_lapack_dense(A, ctx);
        if (S != nullptr)
            return S;
        // Allocation failure → fall back to the built-in solver.
    }
    return SUNLinSol_Dense(y, A, ctx);
}

// Identify the custom adaptive solver by its setup op (ls_setup has internal
// linkage but is visible here in the same TU). The built-in SUNLinSol_Dense and
// KLU solvers carry a different setup function, so these guards correctly skip
// them and never reinterpret a foreign solver's content.
void lapack_dense_reset_factor_count(SUNLinearSolver S) {
    if (S == nullptr || S->ops == nullptr || S->ops->setup != ls_setup || S->content == nullptr)
        return;
    auto *c = static_cast<LapackDenseContent *>(S->content);
    c->factor_count = 0;
    c->blas_factor_count = 0;
}

long lapack_dense_factor_count(SUNLinearSolver S) {
    if (S == nullptr || S->ops == nullptr || S->ops->setup != ls_setup || S->content == nullptr)
        return -1;
    return static_cast<LapackDenseContent *>(S->content)->factor_count;
}

long lapack_dense_blas_factor_count(SUNLinearSolver S) {
    if (S == nullptr || S->ops == nullptr || S->ops->setup != ls_setup || S->content == nullptr)
        return -1;
    return static_cast<LapackDenseContent *>(S->content)->blas_factor_count;
}

#else // !BNGSIM_HAS_LAPACK_DENSE

SUNLinearSolver make_dense_linear_solver(N_Vector y, SUNMatrix A, SUNContext ctx,
                                         bool /*prefer_lapack*/) {
    return SUNLinSol_Dense(y, A, ctx);
}

// No custom adaptive solver is built without a BLAS backend, so there is no
// counter to reset or report.
void lapack_dense_reset_factor_count(SUNLinearSolver /*S*/) {}
long lapack_dense_factor_count(SUNLinearSolver /*S*/) { return -1; }
long lapack_dense_blas_factor_count(SUNLinearSolver /*S*/) { return -1; }

#endif // BNGSIM_HAS_LAPACK_DENSE

} // namespace bngsim
