// bngsim/src/lapack_dense_linsol.cpp — GH #84, adaptive gate GH #132
//
// Custom direct SUNLinearSolver with an adaptive factorization backend: it
// factors with either SUNDIALS' built-in denseGETRF or the BLAS dgetrf (chosen
// per-factorization by the GH #132 count gate) and always back-solves with the
// built-in denseGETRS. See lapack_dense_linsol.hpp for the rationale. Every
// dense solve is this solver, the BLAS factor only where it is asked for and
// linked: each reads a row the matrix holds alone as its own equation, and
// gives exactly 0 to rows that read only rows at 0 (below).
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

#include <algorithm>
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

#endif // BNGSIM_HAS_LAPACK_DENSE

namespace {

#ifdef BNGSIM_HAS_LAPACK_DENSE
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
// values that gate is shipped with. Only the BLAS solver reads them, so they
// live inside its #ifdef: outside it, a build without LAPACK warns that all
// three are unused.
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
#endif // BNGSIM_HAS_LAPACK_DENSE

// A row with at most kMaxReads nonzeros off its diagonal is a candidate to be
// read on its own (below). A row with more is left to the LU, and so is every
// row that reads one, directly or through others: finding a candidate's entries
// is a pass over the matrix, which the bound keeps to a few entries per row.
constexpr sunindextype kMaxReads = 16;

// Solver content: SUNDIALS 0-based pivots consumed by denseGETRS, plus a scratch
// LAPACK 1-based pivot array filled by dgetrf, plus the GH #132 adaptive-gate
// state (how many factorizations have run, and the switch thresholds), plus the
// rows of the factored matrix the solve reads on their own. `adaptive` is false for a solver that
// was not asked for the BLAS factor, or whose build links none: it factors with
// the built-in GETRF only, as SUNLinSol_Dense does, and the GH #132 accessors
// treat it as not the adaptive solver (-1, no reset).
struct LapackDenseContent {
    sunindextype N;
    sunindextype *pivots; // 0-based (denseGETRS convention), size N
#ifdef BNGSIM_HAS_LAPACK_DENSE
    lapack_int *ipiv; // 1-based LAPACK scratch, size N
#endif
    bool adaptive;          // the GH #132 gate may take the BLAS dgetrf
    long factor_count;      // # ls_setup calls so far (GH #132)
    long blas_factor_count; // # of those that actually took the BLAS dgetrf path
    long switch_after;      // K: built-in for the first K factorizations, then dgetrf
    long min_n;             // never use the BLAS factor below this N
    sunindextype last_flag;
    // Per row: its nonzeros off the diagonal, counted up to kMaxReads + 1, size N.
    unsigned char *off;
    // Per row, at the last full look at the pattern: 0 for a row with more than
    // kMaxReads entries off its diagonal, 1 for one that reads such a row,
    // directly or through others, and 2 for a candidate, which reads neither.
    unsigned char *state;
    sunindextype *witness;     // for a row in state 1, a row in state 0 or 1 it reads
    sunindextype n_barred;     // the rows in state 1
    sunindextype *barred;      // their indices, size N
    unsigned char *hit;        // scratch for finding them, size N
    sunindextype *wave, *next; // likewise
    bool have_pattern;         // the fields below hold a pattern still to check
    // The candidates, and what each one's row reads.
    sunindextype n_cand;
    sunindextype *cand;    // candidate k's row, size N
    sunindextype *cand_of; // a row's candidate index, or -1, size N
    sunindextype *n_dep;   // candidate k's entries off the diagonal, size N
    sunindextype *dep_col; // their columns, kMaxReads per candidate
    sunindextype
        *reader_at;       // candidate k's readers are reader[reader_at[k] .. [k + 1]), size N + 1
    sunindextype *reader; // the candidates whose rows read candidate k's, kMaxReads per candidate
    // The rows held alone: candidates with nothing off their diagonals.
    sunindextype n_alone;
    sunindextype *alone;     // their indices, size N
    sunrealtype *alone_diag; // their diagonal entries, read before the factor
    sunrealtype *alone_x;    // a solve's answer for each, size N
    // Per solve: per candidate, whether its component is exactly 0, and a list.
    unsigned char *zero; // size N
    sunindextype *queue; // size N
};

// ── Rows the solve reads on their own ───────────────────────────────────────
// Where row i of the Newton matrix has no nonzero off its diagonal, x_i is fixed
// by that row alone: a_ii·x_i = b_i. Partial pivoting does not see that. When
// column i has a larger entry in another row, GETRF takes that row as the
// pivot, and the back-solve then computes x_i from it, as the remainder of
// sums that cancel in exact arithmetic. Their rounding is left in x_i.
//
// That residue matters where the answer is an exact, unstable state. In
// MODEL1812040005, P = 0 is a rate-rule target whose law is P·g, so P stays
// at exactly 0, but g > 0 for a while, and an e-fold every 0.1 time units
// grows any departure by 4e26 there. Its row of the matrix is the diagonal alone
// while P = 0, and V's row reads √P with a huge slope at 0. With the species
// in the file's order, P's column comes last, and its row is never a pivot
// elsewhere: x_P = 0 exactly. Reversed, P's column comes first, V's row is the
// pivot, and the 1e-16 left in x_P grew to P = 2.6e8, 25.6x the trajectory's
// peak, on x86-64. On arm64 the residue came out negative, and the band of
// #1340 held it there. BIOMD0000000908's S = 0 went the same way. So the solve
// reads each row held alone as its row gives it, b_i / a_ii.
//
// Species at 0 that feed each other went the same way (#1485): a chain,
// Q' = kQ + P, where Q's row reads P's, and a cycle, P' = Q' = k(P + Q), where
// P's and Q's rows read each other. Neither row is held alone, and a cycle's P
// reached 9.6e5 from 0 with the species reversed. But a set of rows that read
// only each other, each with b_i = 0, fixes those components at exactly 0: the
// rows are a system of their own, A_ZZ·x_Z = 0, and A_ZZ is not singular where
// A is not, since the rest of A does not reach them (A is block triangular
// with A_ZZ a block). KLU's block-triangular form finds the same answer, so the
// solve gives 0 to the largest such set. Every other component is the
// back-solve's, so a solve whose answer has no such exact 0 is SUNDIALS' dense
// LU's to the bit, as before.
//
// A NaN entry is not a zero, so a row with one reads it. Finding the rows is
// one pass over the matrix per factorization, against GETRF's N^3/3, and a
// check of the candidates' entries; a few more passes where the pattern has
// changed. Reading them is a pass over the candidates and their readers per
// solve.

// The rows that read a row with more than kMaxReads entries off its diagonal,
// directly or through others: none of them can be read on its own. Each wave
// takes the columns of the rows barred so far, in a loop over each column that
// vectorizes, and bars the rows with an entry there, recording the first such
// column as the witness. Each column is taken once, and none once every row is
// barred.
void bar_rows(LapackDenseContent *c, sunrealtype **cols, sunindextype n) {
    constexpr unsigned char kMore = static_cast<unsigned char>(kMaxReads + 1);
    sunindextype n_wave = 0;
    for (sunindextype i = 0; i < n; ++i) {
        c->state[i] = (c->off[i] < kMore) ? 2 : 0;
        if (c->state[i] == 0)
            c->wave[n_wave++] = i;
    }
    c->n_barred = 0;
    sunindextype left = n - n_wave; // rows not barred yet
    unsigned char *hit = c->hit;
    while (n_wave > 0 && left > 0) {
        for (sunindextype i = 0; i < n; ++i)
            hit[i] = 0;
        for (sunindextype q = 0; q < n_wave; ++q) {
            const sunrealtype *col = cols[c->wave[q]];
            for (sunindextype i = 0; i < n; ++i)
                hit[i] |= static_cast<unsigned char>(col[i] != SUN_RCONST(0.0));
        }
        sunindextype n_next = 0;
        for (sunindextype i = 0; i < n; ++i) {
            if (c->state[i] != 2 || hit[i] == 0)
                continue;
            sunindextype q = 0;
            while (cols[c->wave[q]][i] == SUN_RCONST(0.0))
                ++q;
            c->state[i] = 1;
            c->witness[i] = c->wave[q];
            c->barred[c->n_barred++] = i;
            c->next[n_next++] = i;
        }
        left -= n_next;
        std::swap(c->wave, c->next);
        n_wave = n_next;
    }
}

// Each candidate's entries off the diagonal, by column, and from them each
// candidate's readers. Most of a column is neither: a chunk of rows is tested
// in a loop that vectorizes, and only a chunk holding a candidate's nonzero is
// walked entry by entry.
void gather_candidates(LapackDenseContent *c, sunrealtype **cols, sunindextype n) {
    constexpr sunindextype chunk = 16;
    const unsigned char *state = c->state;
    const sunindextype m = c->n_cand;
    for (sunindextype k = 0; k < m; ++k)
        c->n_dep[k] = 0;
    for (sunindextype j = 0; j < n; ++j) {
        const sunrealtype *col = cols[j];
        for (sunindextype i0 = 0; i0 < n; i0 += chunk) {
            const sunindextype i1 = std::min(n, i0 + chunk);
            unsigned char any = 0;
            for (sunindextype i = i0; i < i1; ++i)
                any |= static_cast<unsigned char>((col[i] != SUN_RCONST(0.0)) & (state[i] == 2));
            if (any == 0)
                continue;
            for (sunindextype i = i0; i < i1; ++i) {
                if (col[i] == SUN_RCONST(0.0) || state[i] != 2 || i == j)
                    continue;
                const sunindextype k = c->cand_of[i];
                c->dep_col[k * kMaxReads + c->n_dep[k]++] = j;
            }
        }
    }
    // A candidate reads only candidates: a row reading any other is barred.
    for (sunindextype k = 0; k <= m; ++k)
        c->reader_at[k] = 0;
    for (sunindextype k = 0; k < m; ++k)
        for (sunindextype e = 0; e < c->n_dep[k]; ++e)
            ++c->reader_at[c->cand_of[c->dep_col[k * kMaxReads + e]] + 1];
    for (sunindextype k = 0; k < m; ++k)
        c->reader_at[k + 1] += c->reader_at[k];
    for (sunindextype k = 0; k < m; ++k)
        c->queue[k] = c->reader_at[k]; // the next free slot of each list
    for (sunindextype k = 0; k < m; ++k)
        for (sunindextype e = 0; e < c->n_dep[k]; ++e)
            c->reader[c->queue[c->cand_of[c->dep_col[k * kMaxReads + e]]]++] = k;
}

void note_rows_read_alone(LapackDenseContent *c, SUNMatrix A) {
    const sunindextype n = SUNDenseMatrix_Rows(A);
    sunrealtype **cols = SUNDenseMatrix_Cols(A);
    constexpr unsigned char kMore = static_cast<unsigned char>(kMaxReads + 1);
    // off[i] counts the nonzeros off row i's diagonal, column by column as the
    // matrix is stored, up to kMore, in loops with no branch either side of the
    // diagonal: the compiler vectorizes them. A branch per entry made the scan a
    // tenth of a fill-routed model's run (egfr_net, N = 356).
    unsigned char *off = c->off;
    for (sunindextype i = 0; i < n; ++i)
        off[i] = 0;
    for (sunindextype j = 0; j < n; ++j) {
        const sunrealtype *col = cols[j];
        for (sunindextype i = 0; i < j; ++i)
            off[i] = std::min<unsigned char>(
                kMore, static_cast<unsigned char>(off[i] + (col[i] != SUN_RCONST(0.0))));
        for (sunindextype i = j + 1; i < n; ++i)
            off[i] = std::min<unsigned char>(
                kMore, static_cast<unsigned char>(off[i] + (col[i] != SUN_RCONST(0.0))));
    }
    // The candidates and what they read depend on the pattern alone, which
    // changes only where an entry reaches or leaves 0. They are the last full
    // look's where it still holds: the same rows have at most kMaxReads entries
    // off their diagonals; each candidate has as many as it had, all still
    // nonzero where they were, so they are the same entries; and each row
    // barred for reading a row with more still reads the row it was barred by,
    // so it is still barred. Looking again is a few passes over the matrix that
    // test each entry, which on egfr_net took a fifth of the factor's time.
    bool same = c->have_pattern;
    if (same) {
        unsigned char moved = 0;
        for (sunindextype i = 0; i < n; ++i)
            moved |= static_cast<unsigned char>((off[i] < kMore) != (c->state[i] != 0));
        same = moved == 0;
    }
    for (sunindextype q = 0; same && q < c->n_barred; ++q) {
        const sunindextype i = c->barred[q];
        same = cols[c->witness[i]][i] != SUN_RCONST(0.0);
    }
    for (sunindextype k = 0; same && k < c->n_cand; ++k) {
        const sunindextype i = c->cand[k];
        same = c->n_dep[k] == off[i];
        for (sunindextype e = 0; same && e < c->n_dep[k]; ++e)
            same = cols[c->dep_col[k * kMaxReads + e]][i] != SUN_RCONST(0.0);
    }
    if (!same) {
        bar_rows(c, cols, n);
        sunindextype m = 0;
        for (sunindextype i = 0; i < n; ++i) {
            c->cand_of[i] = -1;
            if (c->state[i] == 2) {
                c->cand_of[i] = m;
                c->cand[m++] = i;
            }
        }
        c->n_cand = m;
        if (m > 0)
            gather_candidates(c, cols, n);
        c->have_pattern = true;
    }
    // The rows held alone, with this matrix's diagonal. A zero diagonal on one is
    // a singular matrix, which the factorization reports; there is nothing for
    // the solve to read.
    sunindextype k_alone = 0;
    for (sunindextype k = 0; k < c->n_cand; ++k) {
        const sunindextype i = c->cand[k];
        if (c->n_dep[k] == 0 && cols[i][i] != SUN_RCONST(0.0)) {
            c->alone[k_alone] = i;
            c->alone_diag[k_alone] = cols[i][i];
            ++k_alone;
        }
    }
    c->n_alone = k_alone;
}

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
    note_rows_read_alone(c, A);

    // Adaptive backend choice (GH #132). The first `switch_after` factorizations
    // run the built-in dense GETRF — SUNLinSol_Dense's factor, to the byte — so a
    // few-step model pays no BLAS overhead and cannot regress. Only once the run
    // has refactored more than that (a stiff / long-horizon, factorization-bound
    // integration) does it switch to the blocked BLAS dgetrf, where the O(N^3)
    // factor dominates and dgetrf wins. A minimum N keeps tiny systems on the
    // built-in factor no matter how often they refactor (dgetrf there is pure
    // call overhead). The back-solve is the built-in denseGETRS either way, so
    // the in-place L/U storage and 0-based pivots are consumed identically.
    const bool use_blas = c->adaptive && c->factor_count > c->switch_after && rows >= c->min_n;
    if (use_blas)
        ++c->blas_factor_count;

    if (!use_blas) {
        // Built-in dense LU: writes the same in-place L/U and 0-based pivots that
        // denseGETRS consumes. Return matches SUNLinSol_Dense's last_flag (0 ok;
        // >0 is the 1-based column of a zero pivot → singular).
        c->last_flag = SUNDlsMat_denseGETRF(SUNDenseMatrix_Cols(A), rows, rows, c->pivots);
        return c->last_flag != 0 ? SUNLS_LUFACT_FAIL : SUN_SUCCESS;
    }

#ifdef BNGSIM_HAS_LAPACK_DENSE
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
#else
    return SUNLS_LUFACT_FAIL; // unreachable: `adaptive` is never set without a backend
#endif
}

int ls_solve(SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, sunrealtype /*tol*/) {
    auto *c = static_cast<LapackDenseContent *>(S->content);
    // Each row held alone gives its component outright, and a candidate whose
    // right-hand side is 0 may be exactly 0. Read b before x is written, in case
    // the caller passes one vector as both.
    const sunrealtype *bd = N_VGetArrayPointer(b);
    for (sunindextype k = 0; k < c->n_alone; ++k)
        c->alone_x[k] = bd[c->alone[k]] / c->alone_diag[k];
    const sunindextype m = c->n_cand;
    sunindextype n_queue = 0;
    for (sunindextype k = 0; k < m; ++k) {
        c->zero[k] = static_cast<unsigned char>(bd[c->cand[k]] == SUN_RCONST(0.0));
        if (c->zero[k] == 0)
            c->queue[n_queue++] = k;
    }
    // A candidate is 0 where its own b is and every row it reads is 0: the rows
    // read, directly or through others, from a candidate whose b is not 0 are
    // not.
    for (sunindextype q = 0; q < n_queue; ++q) {
        const sunindextype k = c->queue[q];
        for (sunindextype r = c->reader_at[k]; r < c->reader_at[k + 1]; ++r) {
            const sunindextype w = c->reader[r];
            if (c->zero[w] != 0) {
                c->zero[w] = 0;
                c->queue[n_queue++] = w;
            }
        }
    }
    N_VScale(SUN_RCONST(1.0), b, x); // copy b into x, exactly as SUNLinSol_Dense does
    SUNDlsMat_denseGETRS(SUNDenseMatrix_Cols(A), SUNDenseMatrix_Rows(A), c->pivots,
                         N_VGetArrayPointer(x));
    sunrealtype *xd = N_VGetArrayPointer(x);
    for (sunindextype k = 0; k < c->n_alone; ++k)
        xd[c->alone[k]] = c->alone_x[k];
    for (sunindextype k = 0; k < m; ++k)
        if (c->zero[k] != 0)
            xd[c->cand[k]] = SUN_RCONST(0.0);
    c->last_flag = SUN_SUCCESS;
    return SUN_SUCCESS;
}

sunindextype ls_last_flag(SUNLinearSolver S) {
    return static_cast<LapackDenseContent *>(S->content)->last_flag;
}

SUNErrCode ls_space(SUNLinearSolver S, long int *lenrwLS, long int *leniwLS) {
    auto *c = static_cast<LapackDenseContent *>(S->content);
    const long int n = static_cast<long int>(c->N);
    // pivots, the candidates and what they read, plus ipiv where a BLAS factor is
    // linked (the bytes per row, off[], state[], hit[] and zero[], are left out)
#ifdef BNGSIM_HAS_LAPACK_DENSE
    *leniwLS = 3 + (11 + 2 * kMaxReads) * n;
#else
    *leniwLS = 3 + (10 + 2 * kMaxReads) * n;
#endif
    *lenrwLS = 2 * n; // alone_diag, alone_x
    return SUN_SUCCESS;
}

void free_content(LapackDenseContent *c) {
    std::free(c->pivots);
#ifdef BNGSIM_HAS_LAPACK_DENSE
    std::free(c->ipiv);
#endif
    std::free(c->off);
    std::free(c->state);
    std::free(c->hit);
    std::free(c->zero);
    std::free(c->witness);
    std::free(c->barred);
    std::free(c->wave);
    std::free(c->next);
    std::free(c->cand);
    std::free(c->cand_of);
    std::free(c->n_dep);
    std::free(c->dep_col);
    std::free(c->reader_at);
    std::free(c->reader);
    std::free(c->alone);
    std::free(c->queue);
    std::free(c->alone_diag);
    std::free(c->alone_x);
    std::free(c);
}

SUNErrCode ls_free(SUNLinearSolver S) {
    if (S == nullptr)
        return SUN_SUCCESS;
    if (S->content) {
        free_content(static_cast<LapackDenseContent *>(S->content));
        S->content = nullptr;
    }
    if (S->ops) {
        std::free(S->ops);
        S->ops = nullptr;
    }
    std::free(S);
    return SUN_SUCCESS;
}

// `adaptive` asks for the GH #132 gate onto the BLAS factor; it is honoured only
// where a backend is linked.
SUNLinearSolver make_lapack_dense(SUNMatrix A, SUNContext ctx, bool adaptive) {
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

    auto *c = static_cast<LapackDenseContent *>(std::calloc(1, sizeof(LapackDenseContent)));
    if (c == nullptr) {
        SUNLinSolFreeEmpty(S);
        return nullptr;
    }
    const sunindextype n = SUNDenseMatrix_Rows(A);
    const auto count = static_cast<size_t>(n);
    c->N = n;
    c->factor_count = 0;
    c->blas_factor_count = 0;
#ifdef BNGSIM_HAS_LAPACK_DENSE
    c->adaptive = adaptive;
    c->switch_after = env_long("BNGSIM_LAPACK_DENSE_K", kDefaultSwitchAfter);
    c->min_n = env_long("BNGSIM_LAPACK_DENSE_MIN_N", kDefaultMinN);
    c->ipiv = static_cast<lapack_int *>(std::malloc(count * sizeof(lapack_int)));
#else
    (void)adaptive;
    c->adaptive = false;
#endif
    c->last_flag = 0;
    c->n_alone = 0;
    c->n_cand = 0;
    c->n_barred = 0;
    c->have_pattern = false;
    c->pivots = static_cast<sunindextype *>(std::malloc(count * sizeof(sunindextype)));
    bool ok = c->pivots != nullptr;
    c->off = static_cast<unsigned char *>(std::malloc((count) * sizeof(unsigned char)));
    ok = ok && c->off != nullptr;
    c->state = static_cast<unsigned char *>(std::malloc((count) * sizeof(unsigned char)));
    ok = ok && c->state != nullptr;
    c->hit = static_cast<unsigned char *>(std::malloc((count) * sizeof(unsigned char)));
    ok = ok && c->hit != nullptr;
    c->zero = static_cast<unsigned char *>(std::malloc((count) * sizeof(unsigned char)));
    ok = ok && c->zero != nullptr;
    c->witness = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->witness != nullptr;
    c->barred = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->barred != nullptr;
    c->wave = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->wave != nullptr;
    c->next = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->next != nullptr;
    c->cand = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->cand != nullptr;
    c->cand_of = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->cand_of != nullptr;
    c->n_dep = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->n_dep != nullptr;
    c->dep_col =
        static_cast<sunindextype *>(std::malloc((count * kMaxReads) * sizeof(sunindextype)));
    ok = ok && c->dep_col != nullptr;
    c->reader_at = static_cast<sunindextype *>(std::malloc((count + 1) * sizeof(sunindextype)));
    ok = ok && c->reader_at != nullptr;
    c->reader =
        static_cast<sunindextype *>(std::malloc((count * kMaxReads) * sizeof(sunindextype)));
    ok = ok && c->reader != nullptr;
    c->alone = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->alone != nullptr;
    c->queue = static_cast<sunindextype *>(std::malloc((count) * sizeof(sunindextype)));
    ok = ok && c->queue != nullptr;
    c->alone_diag = static_cast<sunrealtype *>(std::malloc((count) * sizeof(sunrealtype)));
    ok = ok && c->alone_diag != nullptr;
    c->alone_x = static_cast<sunrealtype *>(std::malloc((count) * sizeof(sunrealtype)));
    ok = ok && c->alone_x != nullptr;
    for (sunindextype i = 0; ok && i < n; ++i)
        c->cand_of[i] = -1;
    if (!ok) {
        free_content(c);
        SUNLinSolFreeEmpty(S);
        return nullptr;
    }
    S->content = c;
    return S;
}

// The adaptive solver's content, or nullptr for any other solver: the built-in
// SUNLinSol_Dense and KLU carry a different setup op (ls_setup has internal
// linkage but is visible here in the same TU), and this solver built without
// the gate is not the adaptive one either.
LapackDenseContent *adaptive_content(SUNLinearSolver S) {
    if (S == nullptr || S->ops == nullptr || S->ops->setup != ls_setup || S->content == nullptr)
        return nullptr;
    auto *c = static_cast<LapackDenseContent *>(S->content);
    return c->adaptive ? c : nullptr;
}

} // namespace

SUNLinearSolver make_dense_linear_solver(N_Vector y, SUNMatrix A, SUNContext ctx,
                                         bool prefer_lapack) {
    SUNLinearSolver S = make_lapack_dense(A, ctx, prefer_lapack && lapack_dense_available());
    if (S != nullptr)
        return S;
    // Allocation failure → fall back to the built-in solver.
    return SUNLinSol_Dense(y, A, ctx);
}

SUNLinearSolver make_fill_routed_dense_linear_solver(N_Vector y, SUNMatrix A, SUNContext ctx) {
    SUNLinearSolver S = make_lapack_dense(A, ctx, lapack_dense_available());
    if (S == nullptr)
        return SUNLinSol_Dense(y, A, ctx); // allocation failure → built-in solver
    auto *c = static_cast<LapackDenseContent *>(S->content);
    c->switch_after = 0; // BLAS from the first factorization
    c->min_n = 0;
    return S;
}

void lapack_dense_reset_factor_count(SUNLinearSolver S) {
    if (LapackDenseContent *c = adaptive_content(S)) {
        c->factor_count = 0;
        c->blas_factor_count = 0;
    }
}

long lapack_dense_factor_count(SUNLinearSolver S) {
    const LapackDenseContent *c = adaptive_content(S);
    return c != nullptr ? c->factor_count : -1;
}

long lapack_dense_blas_factor_count(SUNLinearSolver S) {
    const LapackDenseContent *c = adaptive_content(S);
    return c != nullptr ? c->blas_factor_count : -1;
}

} // namespace bngsim
