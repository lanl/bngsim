// bngsim/include/bngsim/dense_eigenvalues.hpp -- eigenvalues of a small dense
// real matrix (issue #78)
//
// Why this exists rather than a LAPACK call: dgeev is only linked when CMake
// finds a BLAS backend (BNGSIM_HAS_LAPACK_DENSE, see lapack_dense_linsol.hpp),
// and the steady-state stability certificate that consumes this has to reach the
// same verdict on every build — a guard that silently does not run on one
// platform is worse than no guard. The routines here are the textbook
// EISPACK/LAPACK sequence (balance → Householder Hessenberg → Francis
// double-shift QR), unblocked and dependency-free.
//
// Cost is the usual O(n³) with a small constant. The caller in steady_state.cpp
// caps the size it hands over for exactly that reason.
#pragma once

namespace bngsim {

// Eigenvalues of a general real n×n matrix.
//
// `a` is COLUMN-MAJOR (a[j*n + i] = A_ij) and is DESTROYED. `wr` and `wi`
// receive the real and imaginary parts of the n eigenvalues; a complex
// conjugate pair is stored in consecutive slots with the +imaginary part first.
// The order is otherwise unspecified (it follows the QR deflation).
//
// Returns false if the QR iteration hit its step limit on some eigenvalue, or
// if the matrix holds a non-finite entry; `wr`/`wi` are then meaningless. A
// caller that cannot proceed without the spectrum must treat false as "no
// answer", never as "nothing found".
bool dense_eigenvalues(double *a, int n, double *wr, double *wi);

// Not offered, and deliberately: the determinant's SIGN.
//
// sign(det A) = (-1)^(number of negative real eigenvalues), so a matrix whose
// determinant sign differs from (-1)^n cannot be Hurwitz — an n³/3 stability
// screen instead of ~10n³, which is exactly what a caller past this file's size
// limit would want. Measured on the ode_fullnet corpus it flags 4 models whose
// true max Re(λ)/max|λ| is 1e-16 or smaller: their reduced Jacobians are
// singular to machine precision (|det| ~ 1e-21), so the sign is roundoff. An LU
// cannot tell that case apart either — see the note on lu_diag_rcond in
// steady_state.cpp, which measured that min|U_jj|/max|U_jj| is not
// rank-revealing on this corpus. A screen that is unreliable precisely where it
// would be the only screen is worse than none.

} // namespace bngsim
