// bngsim/include/bngsim/dq_jacobian.hpp -- the dense difference quotient CVODE
// forms when it is given no Jacobian, formed by bngsim instead so that a row
// whose step crosses a jump in a rate law can be mended (issue #1002).
//
// With no Jacobian callback, CVODE differences the right-hand side one column at
// a time (cvLsDenseDQJac): every dense run and steady-state march with
// `jacobian="fd"`, or with one law the closed form declines, ran on that. It
// reads Δf/h where the step crosses a threshold; fd_jacobian.hpp says what that
// did to a run. `cvode_dense_dq_past_jumps` is CVODE's loop first, whole and as
// it is, so a matrix that crosses nothing is CVODE's to the bit: the same
// increment, the same right-hand-side call, `(1/inc)·(f1 − f0)` as N_VLinearSum
// forms it, and on a failed call the same early return that leaves the probed
// component where it stands (CVODE then rewinds it). The jump test of
// fd_jacobian.hpp runs after it, over the matrix it left.
#pragma once

#include "bngsim/fd_jacobian.hpp"

#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace bngsim {

// The floor of CVODE's difference-quotient increment, minInc = 1000·|h|·uround·
// n·‖fy‖_WRMS (1 where that norm is 0), as cvLsDenseDQJac forms it, with the
// session's error weights left in `ewt`. False where CVODE gives neither the
// weights nor the step. The dense quotient below and the colored one of
// sparse_jacobian.hpp (#1254) both step on it.
inline bool cvode_dq_min_inc(void *cvode_mem, N_Vector fy, N_Vector ewt, sunindextype n,
                             sunrealtype *min_inc) {
    if (CVodeGetErrWeights(cvode_mem, ewt) != CV_SUCCESS) {
        return false;
    }
    sunrealtype h = 0.0;
    if (CVodeGetCurrentStep(cvode_mem, &h) != CV_SUCCESS) {
        return false;
    }
    const sunrealtype uround = SUN_UNIT_ROUNDOFF;
    const sunrealtype fnorm = N_VWrmsNorm(fy, ewt);
    *min_inc = (fnorm != 0.0) ? (1000.0 * SUNRabs(h) * uround * n * fnorm) : 1.0;
    return true;
}

// The dense J at CVODE's Jacobian setup, for a callback handed (y, fy, J, tmp1,
// tmp2, tmp3). CVODE's increment is max(√uround·|y_j|, minInc/ewt_j), with
// minInc = 1000·|h|·uround·n·‖fy‖_WRMS (1 where that norm is 0), from CVODE's
// own error weights and step (CVodeGetErrWeights; CVodeGetCurrentStep returns
// cv_next_h, which CVODE sets at every change of cv_h ahead of a setup). bngsim
// sets no inequality constraints, the one part of CVODE's left out.
//
//   solver_f(out)  f at `y` into `out`, as CVODE calls it; its nonzero return
//                  ends the loop and is returned;
//   side_f(out)    f at `y` into `out` for the jump test, leaving nothing a run
//                  reads afterwards; false where that is not usable (a failure,
//                  or a value that is not finite).
//
// tmp1 holds f(y + h·e_j), tmp2 the weights and tmp3 f(y + 2h·e_j); `scratch`
// (3·n, kept between calls) the two backward rows and each row's terms.
// `*n_mended` counts the entries mended.
template <class SolverF, class SideF>
inline int cvode_dense_dq_past_jumps(void *cvode_mem, N_Vector y, N_Vector fy, SUNMatrix J,
                                     N_Vector tmp1, N_Vector tmp2, N_Vector tmp3,
                                     std::vector<double> &scratch, SolverF &&solver_f,
                                     SideF &&side_f, int *n_mended) {
    const sunindextype n = SUNDenseMatrix_Columns(J);
    sunrealtype min_inc = 1.0;
    if (!cvode_dq_min_inc(cvode_mem, fy, tmp2, n, &min_inc)) {
        return -1;
    }
    const sunrealtype srur = SUNRsqrt(SUN_UNIT_ROUNDOFF);
    const double *ewt = N_VGetArrayPointer(tmp2);
    auto inc_of = [&](sunindextype j, double yj) {
        const sunrealtype by_self = srur * SUNRabs(yj);
        const sunrealtype by_floor = min_inc / ewt[j];
        return by_self > by_floor ? by_self : by_floor;
    };

    double *yd = N_VGetArrayPointer(y);
    const double *f0 = N_VGetArrayPointer(fy);
    const double *f1 = N_VGetArrayPointer(tmp1);
    for (sunindextype j = 0; j < n; ++j) {
        double *col = SUNDenseMatrix_Column(J, j);
        const double yj = yd[j];
        const double inc = inc_of(j, yj);
        yd[j] += inc;
        const int retval = solver_f(tmp1);
        if (retval != 0) {
            return retval;
        }
        yd[j] = yj;
        const double inc_inv = 1.0 / inc;
        for (sunindextype i = 0; i < n; ++i) {
            col[i] = inc_inv * (f1[i] - f0[i]);
        }
    }

    const auto un = static_cast<std::size_t>(n);
    if (scratch.size() != 3 * un) {
        scratch.assign(3 * un, 0.0);
    }
    double *f2 = N_VGetArrayPointer(tmp3);
    double *fm1 = scratch.data();
    double *fm2 = scratch.data() + un;
    double *terms = scratch.data() + 2 * un;
    for (sunindextype i = 0; i < n; ++i) {
        terms[i] = std::abs(f0[i]);
    }
    for (sunindextype j = 0; j < n; ++j) {
        const double *col = SUNDenseMatrix_Column(J, j);
        for (sunindextype i = 0; i < n; ++i) {
            terms[i] += std::abs(col[i] * yd[j]);
        }
    }
    N_Vector side = N_VCloneEmpty(tmp1);
    if (side == nullptr) {
        return -1;
    }
    std::vector<FdJumpMend> mends;
    auto side_at = [&](sunindextype j, double yj, double m, double *out) {
        yd[j] = yj + m * inc_of(j, yj);
        N_VSetArrayPointer(out, side);
        const bool ok = side_f(side);
        yd[j] = yj;
        return ok;
    };
    for (sunindextype j = 0; j < n; ++j) {
        double *col = SUNDenseMatrix_Column(J, j);
        const double yj = yd[j];
        const double inc = inc_of(j, yj);
        if (!fd_jump_testable(yj, inc) || !side_at(j, yj, 2.0, f2)) {
            continue;
        }
        // f(y + h) is f0 + h·col to the rounding of the column formula.
        auto may_at = [&](sunindextype i) {
            const double d01 = col[i] * inc;
            return fd_step_may_cross_jump(d01, f2[i] - f0[i] - d01, fd_jump_floor(terms[i]));
        };
        bool may = false;
        for (sunindextype i = 0; i < n && !may; ++i) {
            may = may_at(i);
        }
        if (!may || !side_at(j, yj, -1.0, fm1) || !side_at(j, yj, -2.0, fm2)) {
            continue;
        }
        for (sunindextype i = 0; i < n; ++i) {
            const double d01 = col[i] * inc;
            const double d12 = f2[i] - f0[i] - d01;
            const double dmm = fm1[i] - fm2[i];
            const double floor = fd_jump_floor(terms[i]);
            if (may_at(i) && fd_step_crosses_jump(dmm, d01, d12, floor)) {
                mends.push_back({static_cast<std::size_t>(j) * un + static_cast<std::size_t>(i),
                                 static_cast<long>(j), col[i]});
                col[i] = fd_entry_past_jump(dmm, f0[i] - fm1[i], d12, floor, inc);
            }
        }
    }
    if (mends.empty()) {
        N_VSetArrayPointer(nullptr, side);
        N_VDestroy(side);
        return 0;
    }
    double *jac = SUNDenseMatrix_Data(J);
    for (sunindextype i = 0; i < n; ++i) {
        terms[i] = std::abs(f0[i]);
    }
    for (sunindextype j = 0; j < n; ++j) {
        const double *col = SUNDenseMatrix_Column(J, j);
        for (sunindextype i = 0; i < n; ++i) {
            terms[i] += std::abs(col[i] * yd[j]);
        }
    }
    // One step of species j back along the flow (fd_jump_came_across): the
    // state is saved in fm1 and restored to the bit; the row there goes in f2.
    auto came_across = [&](long j, std::size_t a, std::size_t b) {
        const double inc = inc_of(j, yd[j]);
        const double tau = inc / std::abs(f0[j]);
        for (sunindextype k = 0; k < n; ++k) {
            fm1[k] = yd[k];
            yd[k] = fm1[k] - tau * f0[k];
        }
        N_VSetArrayPointer(f2, side);
        const bool ok = side_f(side);
        for (sunindextype k = 0; k < n; ++k) {
            yd[k] = fm1[k];
        }
        for (std::size_t e = a; ok && e < b; ++e) {
            const auto i =
                static_cast<sunindextype>(mends[e].at - static_cast<std::size_t>(j) * un);
            double jf = 0.0;
            for (sunindextype k = 0; k < n; ++k) {
                jf += jac[static_cast<std::size_t>(k) * un + static_cast<std::size_t>(i)] * f0[k];
            }
            if (fd_jump_came_across(f2[i], f0[i] - tau * jf, mends[e].was * inc)) {
                return true;
            }
        }
        return false;
    };
    for (std::size_t a = 0; a < mends.size();) {
        const long j = mends[a].col;
        std::size_t b = a;
        while (b < mends.size() && mends[b].col == j) {
            ++b;
        }
        bool own_row = false;
        for (std::size_t e = a; e < b; ++e) {
            own_row = own_row || mends[e].at == static_cast<std::size_t>(j) * (un + 1);
        }
        const bool keep = fd_jump_unparked(f0[j], terms[j]) &&
                          (terms[j] == 0.0 || (own_row && came_across(j, a, b)));
        for (std::size_t e = a; e < b; ++e) {
            if (keep) {
                ++*n_mended;
            } else {
                jac[mends[e].at] = mends[e].was;
            }
        }
        a = b;
    }
    N_VSetArrayPointer(nullptr, side);
    N_VDestroy(side);
    return 0;
}

} // namespace bngsim
