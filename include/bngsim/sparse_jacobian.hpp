// bngsim/include/bngsim/sparse_jacobian.hpp — the sparse (KLU) linear-solver
// route, shared by the time-course solver and the steady-state march (#128)
//
// Everything a CVODE session needs in order to factor its Newton matrix with
// KLU instead of a dense LU, in one place because there are now two sessions
// that do it: CvodeSimulator::run() and SteadyStateMarcher. Before issue #128
// the steady-state march always factored densely — ss_make_dense_linsol's own
// comment said so — while run() on the same model routed to KLU, which on the
// 1000+ species corpus models is 2.0-3.9x of wall clock.
//
// Three things live here, and they are here rather than in one of the two .cpp
// files for the same reason: a routing rule, a matrix layout and a difference
// formula that two solvers must agree on are exactly the things that drift when
// each keeps its own copy.
//
//   * route_to_sparse_linear_solver — the dense/sparse decision itself.
//   * install_csc_structure — reinstating the CSC pattern SUNMatZero_Sparse
//     clears, which every sparse Jacobian callback must do before it fills.
//   * colored_fd_jacobian — the Curtis-Powell-Reid difference quotient, for a
//     sparse-routed model with no closed-form Jacobian (KLU has no built-in
//     difference quotient to fall back on: CVODE's covers dense and banded
//     matrices only).

#pragma once

#include "bngsim/fd_jacobian.hpp"
#include "bngsim/lapack_dense_linsol.hpp"
#include "bngsim/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace bngsim {

// ── The dense/sparse decision ────────────────────────────────────────────────
//
// Species count threshold for auto-selecting the sparse solver: KLU when
// N >= SPARSE_THRESHOLD AND density < SPARSE_DENSITY_MAX. Many rule-based
// models have relatively dense Jacobians (chromatic number ≈ N, density often
// 10-30%), making KLU's sparse factorization slower than a dense LU. Only
// models with truly sparse Jacobians (metapopulation, compartmental transport)
// tend to benefit. The density cutoff reflects internal benchmarking.
//
// The choice of dense backend once this says "dense" (built-in dense LU vs the
// GH #84 BLAS dgetrf solver) is a separate question, answered by
// should_use_lapack_dense() in bngsim/lapack_dense_linsol.hpp — except for a
// model the LU-fill test below moved off KLU, which always takes the BLAS
// factor (make_fill_routed_dense_linear_solver).
inline constexpr int SPARSE_THRESHOLD = 50;
inline constexpr double SPARSE_DENSITY_MAX = 0.10; // 10%

// ── LU fill: a sparse Jacobian whose factor is not ───────────────────────────
//
// SPARSE_DENSITY_MAX tests the Jacobian, but KLU factors I - gamma*J, and what
// it costs is the density of the *factor*. On the rule-derived networks of
// benchmarks/_dev/suite_ode.json with 354-3,744 species the Jacobian is 1-7%
// dense while its LU fills to at least 20-35% of n^2 (hub species — a free
// enzyme, an adaptor, a ligand — couple to most complexes). KLU is left-looking with no
// BLAS, so on those models it spends >90% of a run in klu_l_refactor, and the
// BLAS dense factor (GH #84) is 3.6-10x faster end to end; a genuinely sparse
// network (metapop_sir_100, fill 2%) keeps KLU 6x ahead. See
// benchmarks/_dev/bench_linear_solver_fill.py for the sweep.
//
// So a model the size/density test sends to KLU goes dense instead when all of:
//   * a BLAS dense factor is linked (the built-in unblocked LU is not reliably
//     faster than KLU even at this fill — multisite_phos 2.4x slower);
//   * LU_FILL_DENSE_MIN_N <= n <= LU_FILL_DENSE_MAX_N (below, KLU and the dense
//     factor tie: SHP2_base_model, 149 species, 26% fill; above, two dense n^2
//     copies stop being a reasonable amount of memory — 400 MB at the cap);
//   * the analytical Jacobian is complete and requested (auto / analytical):
//     the dense route's only other Jacobian is CVODE's difference quotient, n
//     RHS evaluations, where the sparse route colors its differences;
//   * KLU's symbolic analysis estimates fill >= LU_FILL_DENSE_MIN (an AMD
//     estimate, low on these nets — compute_lu_fill_estimate, model_builder.cpp).
// Neither force flag is overridden: force_sparse_linear_solver keeps KLU, and
// is the escape hatch for a model this rule misjudges.
inline constexpr double LU_FILL_DENSE_MIN = 0.10;
inline constexpr int LU_FILL_DENSE_MIN_N = 256;
inline constexpr int LU_FILL_DENSE_MAX_N = 5000;

// Can the fill test change this model's route? True only for a model the size
// and density test would send to KLU, in the size window above, on a build with
// both KLU and a BLAS dense factor. Callers use it to skip the fill estimate —
// a KLU symbolic analysis — for every model it cannot affect.
inline bool lu_fill_matters(const JacobianSparsity &sp, int ns) {
#ifdef BNGSIM_HAS_KLU
    return lapack_dense_available() && !sp.empty() && ns >= SPARSE_THRESHOLD &&
           sp.density < SPARSE_DENSITY_MAX && ns >= LU_FILL_DENSE_MIN_N &&
           ns <= LU_FILL_DENSE_MAX_N;
#else
    (void)sp;
    (void)ns;
    return false;
#endif
}

// The fill estimate a routing decision needs, or -1 when the fill test does not
// apply: outside lu_fill_matters, or without a complete analytical Jacobian the
// requested strategy will use. `Model` is NetworkModel (a template only so this
// header need not include model.hpp); estimated_lu_fill() is computed once per
// model and shared across clones.
template <class Model>
double routing_lu_fill(const Model &model, int ns, const std::string &jacobian_strategy) {
    const auto &sp = model.jacobian_sparsity();
    if (!lu_fill_matters(sp, ns))
        return -1.0;
    if (!(jacobian_strategy == "auto" || jacobian_strategy == "analytical") ||
        !model.analytical_jacobian_complete())
        return -1.0;
    return model.estimated_lu_fill();
}

// Does the fill test send this model to the dense BLAS factor instead of KLU?
// `lu_fill` is routing_lu_fill()'s answer; -1 (not applicable) never does.
inline bool fill_routes_to_dense(const JacobianSparsity &sp, int ns, double lu_fill,
                                 const std::string &jacobian_strategy, bool force_dense,
                                 bool force_sparse) {
    return !force_dense && !force_sparse && jacobian_strategy != "jax" &&
           lu_fill >= LU_FILL_DENSE_MIN && lu_fill_matters(sp, ns);
}

// Should this model's Newton matrix be a CSC SUNSparseMatrix factored by KLU?
//
// Use sparse KLU when: (1) KLU is available, (2) the model is large enough,
// (3) a sparsity pattern exists, and (4) the density is low enough to benefit.
// A Jacobian denser than the cutoff is effectively dense and KLU's overhead
// makes it slower than a dense LU — which also guards models where Functional
// rate laws make the sparsity pattern nearly dense. JAX Jacobians always fill a
// dense matrix, so that strategy forces dense mode.
//
// The two force flags (GH #102, GH #29) straddle the size/density test but not
// the hard requirements around it: KLU needs a real sparsity pattern to build
// its CSC matrix, and a JAX Jacobian only ever fills a dense one, so sparse_ok
// gates both overrides. force_dense wins the (rejected upstream) both-set case
// only as a belt-and-braces default.
//
// `jacobian_strategy` is SolverOptions::jacobian / SteadyStateOptions::jacobian.
// "jax" forces dense; "fd" stays sparse and takes the colored difference
// quotient below, exactly as it does on the time-course path (routing_lu_fill
// returns -1 for it, so the fill test never moves an "fd" model).
//
// `lu_fill` is routing_lu_fill(model, ns, jacobian_strategy): a model the size
// and density test sends to KLU goes dense when fill_routes_to_dense says so.
//
// A structurally empty Jacobian (nnz == 0: species no reaction moves, an event's
// targets) has a density of 0, which the density test reads as sparse, but it
// gives KLU nothing to factor and, under jacobian="fd", nothing to color, and the
// run was refused for force_sparse_linear_solver, a flag the caller never set
// (SBML Test Suite 01530). So the size and density test sends it to the dense
// solver, as codegen's own reading of this rule already did (_codegen.py's
// fallback for a core that does not report routes_sparse). force_sparse still
// asks for KLU, and is refused by name where nothing can fill the matrix.
inline bool route_to_sparse_linear_solver(const JacobianSparsity &sp, int ns,
                                          const std::string &jacobian_strategy, bool force_dense,
                                          bool force_sparse, double lu_fill) {
#ifdef BNGSIM_HAS_KLU
    const bool sparse_ok = (jacobian_strategy != "jax") && !sp.empty();
    return sparse_ok && !force_dense &&
           (force_sparse ||
            ((ns >= SPARSE_THRESHOLD) && (sp.nnz > 0) && (sp.density < SPARSE_DENSITY_MAX) &&
             !fill_routes_to_dense(sp, ns, lu_fill, jacobian_strategy, force_dense, force_sparse)));
#else
    (void)sp;
    (void)ns;
    (void)jacobian_strategy;
    (void)force_dense;
    (void)force_sparse;
    (void)lu_fill;
    return false;
#endif
}

// ── CSC structure ────────────────────────────────────────────────────────────
//
// Copy the model's CSC pattern into a SUNSparseMatrix's index arrays. Every
// sparse Jacobian callback must call this before filling values: CVODE may call
// SUNMatZero() ahead of the callback, and SUNMatZero_Sparse clears BOTH the
// values and the structural indices. Templated on the index type so this header
// needs no SUNDIALS include; pass SUNSparseMatrix_IndexPointers(J) and
// SUNSparseMatrix_IndexValues(J).
template <typename IndexT>
inline void install_csc_structure(IndexT *col_ptrs, IndexT *row_indices,
                                  const JacobianSparsity &sp) {
    for (int j = 0; j <= sp.n; ++j) {
        col_ptrs[j] = static_cast<IndexT>(sp.col_ptrs[j]);
    }
    for (int k = 0; k < sp.nnz; ++k) {
        row_indices[k] = static_cast<IndexT>(sp.row_indices[k]);
    }
}

// ── Colored finite-difference Jacobian (Curtis-Powell-Reid) ──────────────────
//
// Computes J = ∂f/∂y into the CSC value array. Columns that share a color have
// non-overlapping sparsity patterns, so they can be perturbed simultaneously in
// a single RHS evaluation. This reduces the cost from O(N) RHS evals (one per
// column) to O(n_colors) ≈ 5-20, which is the key speedup for large sparse
// models.
//
// For each color c:
//   1. Perturb ALL columns j with color[j] == c simultaneously:
//      y_pert = y + Σ_j h_j * e_j
//   2. Evaluate f(y_pert) — one single RHS call.
//   3. Extract individual column contributions: for each column j in color c,
//      for each nonzero row i in column j:
//        J[i][j] = (f_pert[i] - f[i]) / h_j
//      This works because no other column j' in the same color has a nonzero
//      in row i (that's the coloring guarantee).
//
// Reference: Curtis, Powell, Reid (1974) "On the estimation of sparse
// Jacobian matrices", J. Inst. Math. Appl. 13, 117–119.
//
// Precondition: `sp` carries a coloring (NetworkModel::ensure_jacobian_coloring,
// not the bare accessor), `fy` is f at the unperturbed `y`, and the three
// scratch buffers are ns long — the caller owns them so a per-step Jacobian
// allocates nothing. `eval_rhs(t, y_pert, fy_pert)` evaluates the SAME right-
// hand side the session is integrating.
//
// `jump_rows`, 4·ns long, turns on the jump test of fd_jacobian.hpp (issue
// #1002), over the matrix the loop above left (which gives each row's terms):
// one more evaluation per color at twice the step and, for a color one of whose
// entries may step across a jump, two at minus one and minus two steps; an
// entry that does is given fd_entry_past_jump. A column of a color leaves the
// rows of the other columns of that color alone, so one evaluation serves them
// all, as the quotient's own does. A column is tested where fd_jump_testable
// takes it, and its mends are kept as fd_jacobian.hpp says. Null leaves the
// quotient as it was. `*n_mended` (if given) counts the entries mended.
//
// The step. With the session's error weights `ewt` (and `min_inc` from them,
// cvode_dq_min_inc), a column's step is CVODE's own dense difference
// quotient's (cvLsDenseDQJac): max(sqrt(eps)*|y_j|, min_inc/ewt_j), on the
// scale the error test judges y_j by. Without them, sqrt(eps)*max(|y_j|, 1): a
// floor of 1 in absolute terms, which moved a species of 1e-9 by fifteen times
// itself, so the matrix KLU factored was the Jacobian somewhere else, and Newton
// failed to converge or reached a state with no finite rate where dense
// differences, on the weights, ran (#1254).
template <typename EvalRhs>
inline void colored_fd_jacobian(const JacobianSparsity &sp, int ns, double t, const double *y,
                                const double *fy, double *jac_data, double *y_pert, double *fy_pert,
                                double *h_vals, EvalRhs &&eval_rhs, double *jump_rows = nullptr,
                                int *n_mended = nullptr, const double *ewt = nullptr,
                                double min_inc = 1.0) {
    // Finite difference perturbation scale
    const double sqrt_uround = 1.4901161193847656e-8; // sqrt(machine epsilon)

    // Iterate over colors (one RHS eval per color)
    for (int c = 0; c < sp.n_colors; ++c) {
        const auto &group = sp.color_groups[c];

        // 1. Build perturbed state: y_pert = y + Σ_{j in group} h_j * e_j
        std::memcpy(y_pert, y, static_cast<std::size_t>(ns) * sizeof(double));

        for (int j : group) {
            double h = ewt != nullptr ? std::max(sqrt_uround * std::abs(y[j]), min_inc / ewt[j])
                                      : sqrt_uround * std::max(std::abs(y[j]), 1.0);
            h_vals[j] = h;
            y_pert[j] += h;
        }

        // 2. Single RHS evaluation for all columns in this color group
        eval_rhs(t, y_pert, fy_pert);

        // 3. Extract Jacobian entries for each column in the group
        for (int j : group) {
            double inv_h = 1.0 / h_vals[j];
            int64_t col_start = sp.col_ptrs[j];
            int64_t col_end = sp.col_ptrs[j + 1];

            for (int64_t k = col_start; k < col_end; ++k) {
                int i = static_cast<int>(sp.row_indices[k]);
                jac_data[k] = (fy_pert[i] - fy[i]) * inv_h;
            }
        }
    }
    if (jump_rows == nullptr) {
        return;
    }

    double *f2 = jump_rows;
    double *fm1 = jump_rows + ns;
    double *fm2 = jump_rows + 2 * static_cast<std::ptrdiff_t>(ns);
    double *terms = jump_rows + 3 * static_cast<std::ptrdiff_t>(ns);
    for (int i = 0; i < ns; ++i) {
        terms[i] = std::abs(fy[i]);
    }
    for (int j = 0; j < ns; ++j) {
        for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k) {
            terms[sp.row_indices[k]] += std::abs(jac_data[k] * y[j]);
        }
    }
    // f at y + m·Σh_j·e_j over the color; false where it fails, so a probe the
    // session itself never asked for cannot end the run.
    auto side = [&](const std::vector<int> &group, double m, double *out) {
        std::memcpy(y_pert, y, static_cast<std::size_t>(ns) * sizeof(double));
        for (int j : group) {
            y_pert[j] = y[j] + m * h_vals[j];
        }
        try {
            eval_rhs(t, y_pert, out);
        } catch (...) {
            return false;
        }
        return true;
    };
    std::vector<FdJumpMend> mends;
    for (int c = 0; c < sp.n_colors; ++c) {
        const auto &group = sp.color_groups[c];
        auto mendable = [&](int j) { return fd_jump_testable(y[j], h_vals[j]); };
        if (std::none_of(group.begin(), group.end(), mendable) || !side(group, 2.0, f2)) {
            continue;
        }
        // f(y + h) is fy + h·J to the rounding of the column formula.
        auto may_at = [&](int j, int64_t k) {
            const int i = static_cast<int>(sp.row_indices[k]);
            const double d01 = jac_data[k] * h_vals[j];
            return fd_step_may_cross_jump(d01, f2[i] - fy[i] - d01, fd_jump_floor(terms[i]));
        };
        bool may = false;
        for (int j : group) {
            if (!mendable(j)) {
                continue;
            }
            for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1] && !may; ++k) {
                may = may_at(j, k);
            }
        }
        if (!may || !side(group, -1.0, fm1) || !side(group, -2.0, fm2)) {
            continue;
        }
        for (int j : group) {
            if (!mendable(j)) {
                continue;
            }
            for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k) {
                const int i = static_cast<int>(sp.row_indices[k]);
                const double d01 = jac_data[k] * h_vals[j];
                const double d12 = f2[i] - fy[i] - d01;
                const double dmm = fm1[i] - fm2[i];
                const double floor = fd_jump_floor(terms[i]);
                if (may_at(j, k) && fd_step_crosses_jump(dmm, d01, d12, floor)) {
                    mends.push_back({static_cast<std::size_t>(k), j, jac_data[k]});
                    jac_data[k] = fd_entry_past_jump(dmm, fy[i] - fm1[i], d12, floor, h_vals[j]);
                }
            }
        }
    }
    if (mends.empty()) {
        return;
    }
    for (int i = 0; i < ns; ++i) {
        terms[i] = std::abs(fy[i]);
    }
    for (int j = 0; j < ns; ++j) {
        for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k) {
            terms[sp.row_indices[k]] += std::abs(jac_data[k] * y[j]);
        }
    }
    // J·f with every mend in, for fd_jump_came_across's prediction along the flow.
    double *jf = fm1;
    std::fill(jf, jf + ns, 0.0);
    for (int j = 0; j < ns; ++j) {
        for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k) {
            jf[sp.row_indices[k]] += jac_data[k] * fy[j];
        }
    }
    auto came_across = [&](long j, std::size_t a, std::size_t b) {
        const double tau = h_vals[j] / std::abs(fy[j]);
        for (int k = 0; k < ns; ++k) {
            y_pert[k] = y[k] - tau * fy[k];
        }
        try {
            eval_rhs(t, y_pert, fm2);
        } catch (...) {
            return false;
        }
        for (std::size_t e = a; e < b; ++e) {
            const int i = static_cast<int>(sp.row_indices[mends[e].at]);
            if (fd_jump_came_across(fm2[i], fy[i] - tau * jf[i], mends[e].was * h_vals[j])) {
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
            own_row = own_row || sp.row_indices[mends[e].at] == j;
        }
        const bool keep = fd_jump_unparked(fy[j], terms[j]) &&
                          (terms[j] == 0.0 || (own_row && came_across(j, a, b)));
        for (std::size_t e = a; e < b; ++e) {
            if (!keep) {
                jac_data[mends[e].at] = mends[e].was;
            } else if (n_mended != nullptr) {
                ++*n_mended;
            }
        }
        a = b;
    }
}

} // namespace bngsim
