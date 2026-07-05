// bngsim/include/bngsim/lapack_dense_linsol.hpp — GH #84, adaptive gate GH #132
//
// A custom *direct* SUNLinearSolver for genuinely-dense, factorization-bound
// ODE Jacobians. It factors the dense SUNMatrix with an optimized BLAS GETRF
// (Accelerate on macOS, LAPACK/OpenBLAS on Linux) and KEEPS the SUNDIALS
// built-in triangular back-solve (SUNDlsMat_denseGETRS) for the solve.
//
// The factorization backend is chosen ADAPTIVELY per call (GH #132): the first
// K factorizations use the built-in dense GETRF (byte-identical to the stock
// SUNLinSol_Dense, so few-step models never pay BLAS overhead), and only a run
// that refactors more than K times — a stiff / long-horizon, factorization-
// bound integration — switches to dgetrf for the remainder. This captures the
// per-factorization BLAS win on the workloads that have one while guaranteeing
// no regression on the many that don't. See the gate note below.
//
// Why this shape (see dev/notes/gh84_lapack_dense_kickoff.md):
//   - The built-in unblocked dense GETRF is the bottleneck on large dense
//     models; a blocked BLAS dgetrf is 2–15× faster as N grows past ~100.
//   - LAPACK's single-RHS dgetrs is *slower* than SUNDIALS' built-in back-solve
//     at the corpus-relevant N (call overhead + pivot permutation for nrhs=1),
//     so we keep denseGETRS and only swap the factorization.
//   - We do NOT use SUNLinSol_LapackDense: it ties LAPACK pivots to the global
//     SUNDIALS index type, which can force an index-size choice that also affects
//     KLU. This solver carries its OWN LAPACK pivot array using the detected
//     LAPACK integer ABI, converts it to SUNDIALS' 0-based sunindextype pivots
//     for denseGETRS, and leaves the global index size (and KLU) alone.
//
// When no BLAS backend is linked (BNGSIM_HAS_LAPACK_DENSE undefined), the
// factory falls back to the built-in SUNLinSol_Dense, so every build compiles
// and runs identically to before.

#pragma once

#include <sundials/sundials_context.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>

namespace bngsim {

// ── Opt-in gate (GH #84) ─────────────────────────────────────────────────────
// The BLAS dense factor is correct (verified bit-equivalent to the built-in LU,
// including pivoting) and faster per factorization (microbenchmark: 2–8× as N
// grows). But end-to-end benchmarking on the real rr_parity corpus showed it
// does NOT reliably win: BNG ODE models are RHS-bound, so the factorization is
// rarely the bottleneck, and neither N nor structural density predicts the
// outcome — high-density large models even REGRESSED (BIOMD574 0.90×,
// MODEL1011090000 0.86×) while the one solid win was a moderate-density
// solve-heavy model the density signal would exclude (MODEL9087474843 1.38×).
// One ill-conditioned model (BIOMD338) also integrates more slowly under
// dgetrf's pivoting and stalls where the built-in LU succeeds.
//
// So there is NO static (N, density) gate that reliably separates wins from
// regressions: the deciding quantity is the FACTORIZATION COUNT, a runtime
// property invisible at setup. A 5-factorization fully-dense N=5063 model
// regresses ~25%; an 11-factorization N=1265 model wins 2.27×. GH #132 moves
// the decision into the solver itself — it factors with the built-in GETRF for
// the first K calls (no regression on the common few-step case) and switches to
// dgetrf only once a run proves factorization-bound (count > K), capturing the
// win there. See dev/notes/gh84_lapack_dense_findings.md for the data and K.
//
// The env switch BNGSIM_LAPACK_DENSE=1 still gates whether the adaptive solver
// is used at all (default off → stock built-in dense LU, zero behavioral change
// and rr_parity DIFF 0); BNGSIM_LAPACK_DENSE_K / _MIN_N tune the adaptive gate.

// True iff this build links a BLAS GETRF backend (set by CMake via the
// BNGSIM_HAS_LAPACK_DENSE compile definition).
bool lapack_dense_available();

// Gate predicate: true iff the adaptive custom solver should be used for an
// n×n dense system. Opt-in only — returns true ONLY when the build links a BLAS
// backend AND BNGSIM_LAPACK_DENSE=1 is set; false otherwise (the default,
// keeping the built-in dense LU). The adaptive count gate inside the solver
// (GH #132) then decides per-factorization whether to use built-in or dgetrf,
// so this predicate need not — and cannot — read the runtime factorization
// count. n/density/force_dense are accepted for the call sites but not consulted.
bool should_use_lapack_dense(int n, double density, bool force_dense);

// Construct a dense linear solver for the n×n SUNDenseMatrix A.
//   prefer_lapack && a BLAS backend linked → custom adaptive-factor solver
//                                            (built-in GETRF for the first K
//                                             factorizations, then BLAS dgetrf)
//   otherwise                              → SUNDIALS built-in SUNLinSol_Dense
// Returns nullptr only if the underlying constructor fails (out of memory).
SUNLinearSolver make_dense_linear_solver(N_Vector y, SUNMatrix A, SUNContext ctx,
                                         bool prefer_lapack);

// Reset the GH #132 adaptive factorization counter to zero, so the gate next
// starts from the built-in factor again. The warm CVODE fast path
// (CvodeSimulator::Impl::run_warm) reuses ONE persistent solver across run()
// calls via CVodeReInit; without this, a long prior run would leave the counter
// above K and push a later (possibly short) run straight onto the BLAS factor.
// The cold run() path rebuilds its solver each call and so resets naturally —
// only the warm reuse needs this. No-op unless S is the custom adaptive dense
// solver (the built-in dense and KLU solvers have a different setup op), so it
// is safe to call on whatever solver the warm path is holding.
void lapack_dense_reset_factor_count(SUNLinearSolver S);

// Diagnostic: how many factorizations the adaptive solver has run since it was
// built or last reset; -1 if S is not the custom adaptive dense solver. The
// backend choice is answer-invariant (built-in GETRF and dgetrf agree bit-for-
// bit modulo pivot base), so it is not observable from a trajectory — the tests
// read this to assert the gate's count and reset behavior directly.
long lapack_dense_factor_count(SUNLinearSolver S);

// Diagnostic: of those factorizations, how many actually took the BLAS dgetrf
// path (factor_count > K and N >= min_n) rather than the built-in dense GETRF;
// -1 if S is not the custom adaptive dense solver. Surfaced as
// SolverStats::n_dense_blas_factorizations so a benchmark can tell a run that
// genuinely exercised dgetrf from one that stayed on the built-in factor because
// the K-factorization gate was never crossed (GH #132).
long lapack_dense_blas_factor_count(SUNLinearSolver S);

} // namespace bngsim
