// bngsim/src/steady_state.cpp -- steady-state solver
//
// Default (integration): CVODE integration with early termination on the
//   BNG2.pl parity criterion ||f(y)||_2 / n_species < tol (run_network -c).
// newton: two-tier integrate-first solver (GH #27). Tier 1 is a CVODE burst
//   that carries the state into the physical root's basin; tier 2 is a KINSOL
//   Newton polish, accepted only once it is seed-stable (agrees across two
//   successively tighter bursts). Falls back to integration otherwise. See
//   solve_by_newton_two_tier for the rationale (why Newton-first returned wrong
//   / NaN roots). Since tier 1 IS the integration path, tier 2 can only add
//   work; GH #28 measured that net cost at 1.4-3.9x across six published
//   dose-response models, which is why "integration" is now the default.
// Steady-state sensitivity: dY_ss/dp = -J^{-1} * df/dp
//
// Convergence criterion: every integrate-to-steady-state path here uses the
// SAME rule as Simulator.run(steady_state=True) / BNG2.pl run_network -c:
// the L2 norm of the derivative vector divided by n_species, ||f||_2 / n.
//
// Codegen (issue #63): every RHS evaluation in this file goes through
// SteadyStateRhs, which dispatches to the code-generated `bngsim_codegen_rhs`
// when SteadyStateOptions carries a .so path or a MIR-JIT source and to
// NetworkModel::compute_derivs otherwise. The same object also exposes the
// compiled analytical Jacobian and the analytical ∂f/∂p the codegen sensitivity
// RHS emits, which is what dY_ss/dp is assembled from when they are available.
// Before #63 this file read no codegen option at all: a Simulator whose
// codegen_backend reported "cc" still solved for steady state on the interpreted
// path, and both factors of dY_ss/dp were always finite differences.

#include "bngsim/steady_state.hpp"
#include "bngsim/codegen_abi.hpp"
#include "bngsim/dynamic_library.hpp"
#include "bngsim/mir_jit.hpp"
#include "bngsim/model.hpp"
#include "bngsim/types.hpp"

#include <cvodes/cvodes.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_logger.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "bngsim/lapack_dense_linsol.hpp"
#include "bngsim/platform_compat.hpp"
#include "bngsim/sundials_guards.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bngsim {

// ---------------------------------------------------------------------------
// The RHS this solver evaluates: codegen when one is attached, else interpreted
// ---------------------------------------------------------------------------

// Every f(y) in this file is evaluated through here (issue #63).
//
// Backends. A non-empty SteadyStateOptions::codegen_c_source JIT-compiles the
// emitted C in-process (MIR); otherwise a non-empty codegen_so_path is dlopen'd;
// otherwise the interpreted NetworkModel::compute_derivs runs. The precedence
// and the symbol set match CvodeSimulator::Impl::setup_codegen_rhs, so a
// Simulator gets the same RHS from steady_state() as from run(). A load or
// compile failure THROWS rather than quietly reverting to the interpreter — a
// silent downgrade to a 10x slower path is exactly the failure mode #63 is about.
//
// Beyond f(y) this also carries the two derivative callbacks the sensitivity
// assembly wants, when the artifact provides them:
//   * `bngsim_codegen_jac` — the dense column-major analytical Jacobian, the
//     compiled mirror of NetworkModel::fill_dense_analytical_jacobian;
//   * `bngsim_codegen_sens_rhs` — evaluated at yS = 0, whose J·yS term then
//     vanishes and leaves the bare analytical ∂f/∂p column;
//   * `bngsim_codegen_output_sens` — the compiled d(func)/dθ chain rule, fed the
//     solved dY_ss/dp columns (issue #75).
//
// Parameter values reach the compiled code through a contiguous mirror rather
// than the model's Parameter vector, so any caller that mutates a parameter
// (the finite-difference ∂f/∂p fallback below) must call sync_params()
// afterwards. The interpreted backend reads the model directly and does not care.
class SteadyStateRhs {
  public:
    SteadyStateRhs(NetworkModel &model, const SteadyStateOptions &opts) : model_(model) {
        const bool use_jit = !opts.codegen_c_source.empty();
        if (!use_jit && opts.codegen_so_path.empty()) {
            return; // interpreted
        }

        // The compiled Jacobian comes in two shapes and the codegen emits exactly
        // one of them per model: dense for a dense-routed model, CSC for a
        // sparse/KLU-routed one (GH #162). The steady-state sensitivity solve
        // always factors densely, so both are resolved and the sparse one is
        // scattered into the dense buffer — otherwise every large sparse model,
        // which is precisely the case worth compiling, would silently drop back
        // to the interpreted Jacobian fill.
        if (use_jit) {
            jit_ = MirJit(opts.codegen_c_source);
            rhs_fn_ = jit_.symbol<CodegenRhsFn>("bngsim_codegen_rhs");
            jac_fn_ = jit_.try_symbol<CodegenJacFn>("bngsim_codegen_jac");
            jac_sparse_fn_ = jit_.try_symbol<CodegenJacSparseFn>("bngsim_codegen_jac_sparse");
            sens_fn_ = jit_.try_symbol<CodegenSensRhsFn>("bngsim_codegen_sens_rhs");
            output_sens_fn_ = jit_.try_symbol<CodegenOutputSensFn>("bngsim_codegen_output_sens");
            backend_ = "codegen-jit";
        } else {
            lib_ = DynamicLibrary(opts.codegen_so_path);
            rhs_fn_ = lib_.symbol<CodegenRhsFn>("bngsim_codegen_rhs");
            jac_fn_ = lib_.try_symbol<CodegenJacFn>("bngsim_codegen_jac");
            jac_sparse_fn_ = lib_.try_symbol<CodegenJacSparseFn>("bngsim_codegen_jac_sparse");
            sens_fn_ = lib_.try_symbol<CodegenSensRhsFn>("bngsim_codegen_sens_rhs");
            output_sens_fn_ = lib_.try_symbol<CodegenOutputSensFn>("bngsim_codegen_output_sens");
            backend_ = "codegen-so";
        }

        so_data_.tfun_ctx = &model_;
        so_data_.tfun_eval = tfun_eval_thunk;
        sync_params();
    }

    // Non-copyable / non-movable: so_data_.param_values points into param_buf_.
    SteadyStateRhs(const SteadyStateRhs &) = delete;
    SteadyStateRhs &operator=(const SteadyStateRhs &) = delete;

    const std::string &backend() const { return backend_; }

    // Re-mirror the model's live parameter values into the buffer the compiled
    // code reads, re-deriving constant-expression parameters first.
    //
    // The re-derivation is the same one cvode_rhs/cvode_codegen_rhs perform
    // under a sensitivity probe (issue #2): BNG2.pl encodes a rate law like
    // `chi*kon` as a derived parameter `_rateLaw{N}`, so perturbing `kon`
    // without re-evaluating `_rateLaw{N}` leaves the rate constant at its
    // nominal value and the chain-rule term silently drops out of ∂f/∂p. It runs
    // on the interpreted backend too — compute_derivs()/evaluate_functions()
    // never refresh derived parameters, only set_param() does, so before #63 the
    // finite-difference ∂f/∂p below inherited exactly that hole.
    //
    // `held` is the index of a parameter the caller has just written by hand and
    // wants left alone. Re-deriving it would overwrite the write — and when that
    // write is a finite-difference probe of a *derived* parameter, silently
    // return a zero column. Holding it and re-deriving everything else is
    // exactly NetworkModel::set_param's rule (it detaches the target from its
    // expression, then refreshes the rest), so the FD probe and an explicit
    // set_param see the same model.
    void sync_params(int held = -1) {
        auto &params = const_cast<std::vector<Parameter> &>(model_.parameters());
        auto &evaluator = model_.evaluator();
        for (size_t i = 0; i < params.size(); ++i) {
            auto &p = params[i];
            if (p.is_expression && p.evaluator_id >= 0 && static_cast<int>(i) != held) {
                p.value = evaluator.evaluate(p.evaluator_id);
            }
        }
        if (!rhs_fn_) {
            return; // interpreted backend reads the model directly
        }
        param_buf_.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            param_buf_[i] = params[i].value;
        }
        so_data_.param_values = param_buf_.data();
    }

    // f(t, y) -> ydot. The interpreted branch is compute_derivs() alone:
    // compute_derivs_core() already refreshes observable totals and
    // function-bound parameters for every model that has functions, and skips
    // both for pure mass-action models where they are dead work (the GH #106/T1
    // gate). The compiled RHS evaluates observables and functions internally.
    void eval(double t, const double *y, double *ydot) {
        if (rhs_fn_) {
            rhs_fn_(t, const_cast<double *>(y), ydot, &so_data_);
            return;
        }
        model_.compute_derivs(t, y, ydot);
    }

    // Is a closed-form Jacobian available at all — compiled or interpreted?
    // A compiled Jacobian (either shape) is emitted only when the interpreted
    // one is complete, so the model predicate alone would answer this; both are
    // checked so the two never drift.
    bool has_analytical_jacobian() const {
        return jac_fn_ != nullptr || jac_sparse_fn_ != nullptr ||
               model_.analytical_jacobian_complete();
    }
    // "codegen" / "analytical" / "finite-difference" for SteadyStateResult.
    const char *jacobian_source() const {
        if (jac_fn_ || jac_sparse_fn_)
            return "codegen";
        if (model_.analytical_jacobian_complete())
            return "analytical";
        return "finite-difference";
    }
    // Dense COLUMN-MAJOR n×n: jac[j*n + i] = ∂f_i/∂x_j. Precondition:
    // has_analytical_jacobian(). Every fill memsets the buffer itself.
    void fill_dense_jacobian(double t, const double *y, double *jac) {
        if (jac_fn_) {
            jac_fn_(t, const_cast<double *>(y), jac, &so_data_);
            return;
        }
        if (jac_sparse_fn_) {
            // CSC values → dense column-major. The emitted C zeroes the value
            // array; the dense buffer is ours to clear.
            const int ns = model_.n_species();
            const auto &sp = model_.jacobian_sparsity();
            const int nnz = sp.col_ptrs[ns];
            csc_vals_.assign(static_cast<size_t>(nnz), 0.0);
            jac_sparse_fn_(t, const_cast<double *>(y), csc_vals_.data(), &so_data_);
            std::memset(jac, 0, static_cast<size_t>(ns) * ns * sizeof(double));
            for (int col = 0; col < ns; ++col) {
                for (int k = sp.col_ptrs[col]; k < sp.col_ptrs[col + 1]; ++k) {
                    jac[static_cast<size_t>(col) * ns + sp.row_indices[k]] = csc_vals_[k];
                }
            }
            return;
        }
        model_.fill_dense_analytical_jacobian(t, y, jac);
    }

    // The analytical ∂f/∂p exists only in the compiled artifact — there is no
    // interpreted counterpart, so an absent symbol means the caller must
    // difference. Since GH #67 that is no longer "every Functional/MM model":
    // generate_combined_c falls back to the model-based emitter, which covers a
    // Functional rate law that is smooth algebra. What is left absent is
    // Michaelis-Menten and the laws carrying a condition or a non-smooth builtin.
    bool has_analytical_dfdp() const { return sens_fn_ != nullptr; }

    // ∂f/∂p_{param_index} at (t, y) into `out` (n_species).
    //
    // The emitted bngsim_codegen_sens_rhs computes ySdot = J(t,y)·yS + ∂f/∂p_iP.
    // Handing it an all-zero yS zeroes the first term exactly (it is a matrix
    // product, not a difference quotient), leaving the analytical ∂f/∂p column —
    // the same derivative CVODES integrates against on the time-course path.
    void eval_dfdp(double t, const double *y, int param_index, double *out) {
        const int ns = model_.n_species();
        zero_seed_.assign(static_cast<size_t>(ns), 0.0);
        tmp1_.assign(static_cast<size_t>(ns), 0.0);
        tmp2_.assign(static_cast<size_t>(ns), 0.0);
        ydot_scratch_.assign(static_cast<size_t>(ns), 0.0);
        plist_[0] = param_index;

        CodegenSensUserDataForSO sens_data;
        sens_data.param_values = so_data_.param_values;
        sens_data.plist = plist_;
        sens_data.n_sens = 1;

        sens_fn_(/*Ns=*/1, t, const_cast<double *>(y), ydot_scratch_.data(), /*iS=*/0,
                 zero_seed_.data(), out, &sens_data, tmp1_.data(), tmp2_.data());
    }

    // Is the compiled output-sensitivity chain rule available (issue #75)?
    //
    // Absent for three reasons, all of which the caller answers the same way (fall
    // back to finite differences): the model has no compiled artifact at all; the
    // artifact was built without `_want_output_sens` so the symbol was never
    // emitted; or `_analyze_output_sens` declined the whole model (rateOf, an
    // embedded table-function wrapper, no user-selectable functions).
    bool has_analytical_output_sens() const { return output_sens_fn_ != nullptr; }

    // d(func_m)/dθ_c into func_sens_out[c*n_func + m], from the per-column state
    // sensitivities state_sens[c][i] = dx_i/dθ_c. `plist[c]` is the differentiated
    // parameter's index in the model's Parameter vector — the same index space
    // eval_dfdp uses, since both read the codegen's `p[]` mirror.
    //
    // The emitter leaves a function it did not differentiate (one outside the
    // user-function closure) at whatever the caller pre-filled, and writes NaN for
    // one it declined; the caller must therefore pre-fill with a non-finite
    // sentinel and treat every non-finite row as "no compiled answer".
    void eval_output_sens(double t, const double *y, const double *const *state_sens,
                          const int *plist, int n_sens, double *obs_sens_out,
                          double *func_sens_out) {
        output_sens_fn_(t, y, so_data_.param_values, state_sens, plist, n_sens, obs_sens_out,
                        func_sens_out, &so_data_);
    }

  private:
    // Trampoline the codegen .so calls to evaluate a table function; ctx is the
    // owning NetworkModel (mirrors codegen_tfun_eval_thunk in cvode_simulator).
    static double tfun_eval_thunk(int tf_id, double x, void *ctx) {
        return static_cast<NetworkModel *>(ctx)->evaluate_table_function_at(tf_id, x);
    }

    NetworkModel &model_;
    DynamicLibrary lib_;
    MirJit jit_;
    CodegenRhsFn rhs_fn_ = nullptr;
    CodegenJacFn jac_fn_ = nullptr;
    CodegenJacSparseFn jac_sparse_fn_ = nullptr;
    CodegenSensRhsFn sens_fn_ = nullptr;
    CodegenOutputSensFn output_sens_fn_ = nullptr;
    CodegenUserDataForSO so_data_{};
    std::vector<double> param_buf_;
    std::string backend_ = "exprtk";
    // eval_dfdp scratch, kept on the object so a per-parameter loop does not
    // reallocate four n_species vectors per column.
    std::vector<double> zero_seed_, tmp1_, tmp2_, ydot_scratch_;
    std::vector<double> csc_vals_; // CSC → dense Jacobian scatter buffer
    int plist_[1] = {0};
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct SteadyStateUserData {
    SteadyStateRhs *rhs;
    NetworkModel *model;
};

// The subspace f(y) = 0 is solved on (issue #74).
//
// Resolved once per solve from SteadyStateOptions::steady_state_mask and then
// threaded through everything that asks a convergence question: the residual
// norm, the KINSOL polish's unknown set, and the dY_ss/dp linear system. With no
// mask `masked` is false, `included` is 0..ns-1 and every formula below reduces
// to exactly what it computed before this existed.
struct ResidualSubspace {
    std::vector<int> included; // 0-based species indices, ascending
    std::vector<int> excluded; // the complement, ascending (empty when !masked)
    std::vector<char> keep;    // per-species selector, length ns
    bool masked = false;       // false ⇒ every species (BNG2.pl parity)

    bool includes(int i) const { return keep[static_cast<size_t>(i)] != 0; }
};

// Validate the caller's mask and expand it into the form the solver indexes.
static ResidualSubspace make_subspace(const SteadyStateOptions &opts, int ns) {
    ResidualSubspace sub;
    const bool have_mask = !opts.steady_state_mask.empty();

    if (have_mask && static_cast<int>(opts.steady_state_mask.size()) != ns) {
        throw std::runtime_error(
            "steady_state_mask has " + std::to_string(opts.steady_state_mask.size()) +
            " entries but the model has " + std::to_string(ns) +
            " species: the mask is one selector per species, in species order.");
    }

    sub.keep.assign(static_cast<size_t>(ns), 1);
    if (have_mask) {
        for (int i = 0; i < ns; ++i) {
            sub.keep[static_cast<size_t>(i)] =
                opts.steady_state_mask[static_cast<size_t>(i)] ? 1 : 0;
        }
    }
    for (int i = 0; i < ns; ++i) {
        if (sub.keep[static_cast<size_t>(i)]) {
            sub.included.push_back(i);
        } else {
            sub.excluded.push_back(i);
        }
    }

    // `masked` tracks whether anything was actually EXCLUDED, not whether a mask
    // was supplied. An all-true mask therefore takes byte-for-byte the same code
    // path as no mask — including the KINSOL and dY_ss/dp full-space branches,
    // which the restricted path below deliberately does not reproduce
    // (see solve_by_newton). Without this, `mask=ones(n)` would quietly mean
    // something different from `mask=None`.
    sub.masked = !sub.excluded.empty();

    if (sub.included.empty()) {
        throw std::runtime_error("steady_state_mask excludes every species: there is no subspace "
                                 "left to solve f(y) = 0 on. Select at least one species.");
    }
    return sub;
}

// Compute the steady-state residual ||f(y)||_2 / n over the subspace.
//
// Unmasked this is the BNG2.pl parity criterion ||f(y)||_2 / n_species — the
// SAME quantity Simulator.run(steady_state=True) checks at each output point
// (Network3 network.cpp run_network -c). It is the single convergence criterion
// used by every integrate-to-steady-state path and by the post-solve
// verification of the Newton path, so there is one rule.
//
// Masked (issue #74) both the sum and the divisor run over the included species,
// so `tol` still reads as a per-species residual scale rather than shrinking with
// the number of species the caller dropped.
static double compute_residual(SteadyStateRhs &rhs, const double *y, int ns,
                               const ResidualSubspace &sub) {
    std::vector<double> f(ns, 0.0);
    rhs.eval(0.0, y, f.data()); // steady state: time irrelevant
    double sumsq = 0.0;
    for (int i : sub.included) {
        sumsq += f[i] * f[i];
    }
    const size_t n = sub.included.size();
    return (n > 0) ? std::sqrt(sumsq) / static_cast<double>(n) : 0.0;
}

// A steady state must be finite and (up to a small scale-relative slack)
// non-negative. Newton can walk a species negative, where Hill/power rate laws
// return NaN — compute_residual then returns NaN, and `NaN >= tol` is false, so
// the old convergence check accepted it (GH #27 Bug 1). This predicate rejects
// any non-finite or clearly-negative concentration. The negativity floor is
// relative to the largest concentration so a root that lands a near-zero species
// at -1e-9 (roundoff around a true zero, e.g. simple decay's A*≈0) still passes.
static bool ss_state_is_physical(const std::vector<double> &y) {
    double maxabs = 0.0;
    for (double v : y) {
        if (!std::isfinite(v))
            return false;
        maxabs = std::max(maxabs, std::abs(v));
    }
    const double neg_floor = -1e-7 * std::max(maxabs, 1e-300);
    for (double v : y) {
        if (v < neg_floor)
            return false;
    }
    return true;
}

// Do two candidate steady states agree to AGREE_RTOL? Used by the two-tier
// solver (GH #27 Bug 2) to accept a KINSOL root only once it is *seed-stable*:
// two Newton solves from successively tighter integration bursts landing on the
// same state. The floor `1e-6*max|b|` keeps near-zero species (relative diff
// explodes as a component → 0) from dominating; this is the scale-robust analog
// of the benchmark's XCHECK metric.
static bool ss_states_agree(const std::vector<double> &a, const std::vector<double> &b,
                            double agree_rtol) {
    if (a.size() != b.size())
        return false;
    double maxabs = 0.0;
    for (double v : a)
        maxabs = std::max(maxabs, std::abs(v));
    for (double v : b)
        maxabs = std::max(maxabs, std::abs(v));
    const double floor = 1e-6 * std::max(maxabs, 1e-300);
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double denom = floor + std::abs(b[i]);
        worst = std::max(worst, std::abs(a[i] - b[i]) / denom);
    }
    return worst < agree_rtol;
}

// How close to singular was the factored matrix? min|U_jj| / max|U_jj| read off
// the U diagonal of an in-place LU (both SUNLinSol_Dense and SUNLinSol_LapackDense
// leave the factors in the SUNMatrix data, column-major, so A[j*n+j] is U_jj).
//
// Why this is worth reporting (issue #63). dY_ss/dp = -J⁻¹·(∂f/∂p) is only
// defined when the (reduced) Jacobian at the root has full rank. When it is not,
// the returned matrix is whatever the LU made of a singular system, and nothing
// else on the result says so — the finite-difference Jacobian this predates
// carried ~sqrt(eps) noise that perturbed the singular direction just enough to
// return a finite, modest-looking, entirely meaningless answer. An exact
// analytical Jacobian does not launder that.
//
// Read this as a diagnostic, NOT as a rank test, and do not build a refusal on
// it. Two measurements on the 585-model ode_fullnet corpus say why:
//
//   * min|U_jj|/max|U_jj| is not rank-revealing. Before the conservation-law
//     reduction was repaired (see detect_conservation_laws in model_builder.cpp),
//     97 of 419 solvable models had a genuinely rank-deficient reduced Jacobian,
//     and the two populations OVERLAPPED across five decades: full-rank models ran
//     down to 1.5e-13 while rank-deficient ones reached 1.7e-8. The best single
//     threshold still misclassified 6; the shipped 1e-8 misclassified 10. The
//     six-orders-of-magnitude separation the first version of this comment
//     claimed was an artifact of an eight-model sample.
//   * Most of what it was flagging was not the model. IGF1R_model_v1 (rank
//     578/579), Reduced_IGF1R_hela (546/549) and fceri_fyn (1274/1276) were
//     singular because the reduction picked a dependent-species set it could not
//     solve for, not because their steady states were continua. With that fixed
//     all three are full rank and well conditioned (1.5e-3, 4.7e-3, 1.6e-4), and
//     their dY_ss/dp now matches a finite difference of the steady state itself.
//
// What remains below the floor is a real minority — models like
// tests/data/nested_derived_rate_const.net, whose equilibrium set is a line — so
// the warning still earns its place. A refusal would need a rank-revealing
// factorization or a proper condition estimator (LAPACK dgecon), not this ratio.
static double lu_diag_rcond(const double *lu, int n) {
    if (n <= 0)
        return 0.0;
    double dmin = std::abs(lu[0]);
    double dmax = dmin;
    for (int j = 1; j < n; ++j) {
        const double d = std::abs(lu[static_cast<size_t>(j) * n + j]);
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
    }
    if (!(dmax > 0.0) || !std::isfinite(dmax) || !std::isfinite(dmin))
        return 0.0;
    return dmin / dmax;
}

// Build the dense direct linear solver for an n×n steady-state system,
// applying the GH #84 gate (bngsim/lapack_dense_linsol.hpp). The steady-state
// paths have no KLU option — they always factor densely — so the density floor
// (not force_dense) is the right guard: a structurally-sparse SS Jacobian stays
// on the built-in dense LU, whose zero-skipping beats a full BLAS dgetrf, while
// large AND dense SS systems take the optimized factor. force_dense is false.
static SUNLinearSolver ss_make_dense_linsol(N_Vector v, SUNMatrix A, SUNContext ctx,
                                            NetworkModel &model, int n) {
    const bool use_lapack =
        should_use_lapack_dense(n, model.jacobian_sparsity().density, /*force_dense=*/false);
    return make_dense_linear_solver(v, A, ctx, use_lapack);
}

// ---------------------------------------------------------------------------
// Tier 1: CVODE integration with early termination
// ---------------------------------------------------------------------------

static int cvode_ss_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *ud) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    const double *yp = N_VGetArrayPointer(y);
    double *yp_dot = N_VGetArrayPointer(ydot);
    data->rhs->eval(static_cast<double>(t), yp, yp_dot);
    return 0;
}

// A CVODE session held across successive marches of the SAME trajectory.
//
// The two-tier solver's burst ladder (solve_by_newton_two_tier) walks one
// trajectory to a progressively tighter residual, so building a SUNContext,
// CVODE memory, dense SUNMatrix and linear solver per rung is pure overhead —
// and worse, it restarts BDF at order 1 with a fresh initial step, discarding
// the step-size/order history the previous rung paid to build up. Holding one
// session across the rungs makes the ladder what its header claims: a single
// march to the tightest rung, not a restart per rung. solve_by_integration is
// then just the one-leg case of the same code.
class SteadyStateMarcher {
  public:
    SteadyStateMarcher(NetworkModel &model, SteadyStateRhs &rhs, const SteadyStateOptions &opts,
                       const ResidualSubspace &sub)
        : model_(model), rhs_(rhs), opts_(opts), sub_(sub), ud_{&rhs, &model},
          ns_(model.n_species()) {

        if (!ctx_) {
            throw std::runtime_error("SUNContext_Create failed (steady_state)");
        }

        y_ = NVectorGuard(N_VNew_Serial(ns_, ctx_));
        double *y_data = y_.data();
        for (int i = 0; i < ns_; ++i) {
            y_data[i] = model.species()[i].concentration;
        }

        cvode_mem_ = CvodeMemGuard(CVodeCreate(CV_BDF, ctx_));
        if (CVodeInit(cvode_mem_, cvode_ss_rhs, 0.0, y_) != CV_SUCCESS) {
            throw std::runtime_error("CVodeInit failed (steady_state)");
        }

        CVodeSStolerances(cvode_mem_, opts.rtol, opts.atol);
        CVodeSetUserData(cvode_mem_, &ud_);
        CVodeSetMaxNumSteps(cvode_mem_, opts.max_steps);

        A_ = SUNMatrixGuard(SUNDenseMatrix(ns_, ns_, ctx_));
        LS_ = SUNLinSolGuard(ss_make_dense_linsol(y_, A_, ctx_, model, ns_));
        CVodeSetLinearSolver(cvode_mem_, LS_, A_);

        // Analytical Jacobian if available and not "fd"
        if (opts.jacobian != "fd") {
            // We reuse the existing dense analytical Jacobian callback from
            // cvode_simulator.cpp. But since we can't call that static function
            // directly, we compute derivs manually and let CVODE do FD.
            // For this implementation, we rely on CVODE's internal FD Jacobian.
            // The analytical Jacobian is used by KINSOL (Tier 2) instead.
        }
    }

    // March forward one internal CVODE step at a time, checking the BNG2.pl
    // parity criterion ||f(y)||_2 / n_species < tol after each step. This is
    // the SAME rule Simulator.run(steady_state=True) applies (run_network -c);
    // the old geometric time-horizon (t = 10, 100, 1000, ...) has been
    // removed so there is one convergence rule everywhere. Each march is capped
    // at max_time of ADDITIONAL simulated time via CVodeSetStopTime, so a
    // non-equilibrating system returns unconverged rather than running forever
    // — and a ladder rung gets exactly the budget it got back when every rung
    // built its own integrator starting from t = 0.
    bool march(double tol, double *residual_out) {
        const sunrealtype t_stop = t_ + static_cast<sunrealtype>(opts_.max_time);
        CVodeSetStopTime(cvode_mem_, t_stop);

        double *y_data = y_.data();
        bool converged = false;

        while (t_ < t_stop) {
            int flag = CVode(cvode_mem_, t_stop, y_, &t_, CV_ONE_STEP);
            if (flag < 0) {
                // Integration failed -- report unconverged.
                break;
            }

            // compute_residual evaluates f at y through the same backend the
            // integrator just used, refreshing observables/functions internally.
            double resid = compute_residual(rhs_, y_data, ns_, sub_);
            if (resid < tol) {
                converged = true;
                *residual_out = resid;
                break;
            }

            if (flag == CV_TSTOP_RETURN) {
                // Exhausted this march's time budget without converging.
                break;
            }
        }

        if (!converged) {
            *residual_out = compute_residual(rhs_, y_data, ns_, sub_);
        }
        return converged;
    }

    // Snapshot the march so far as a solver result. Step / RHS-eval counts are
    // cumulative over every march on this session, which is what a caller that
    // laddered through several rungs wants to see.
    SteadyStateResult make_result(bool converged, double residual) {
        SteadyStateResult result;
        result.method_used = "integration";
        result.species_names = model_.species_names();
        const double *y_data = y_.data();
        result.concentrations.assign(y_data, y_data + ns_);
        result.converged = converged;
        result.residual = residual;
        result.n_residual_species = static_cast<int>(sub_.included.size());
        result.excluded_species = sub_.excluded;

        long int nst = 0, nfe = 0;
        CVodeGetNumSteps(cvode_mem_, &nst);
        CVodeGetNumRhsEvals(cvode_mem_, &nfe);
        result.n_steps = static_cast<int>(nst);
        result.n_rhs_evals = static_cast<int>(nfe);
        return result;
    }

  private:
    NetworkModel &model_;
    SteadyStateRhs &rhs_;
    const SteadyStateOptions &opts_;
    const ResidualSubspace &sub_;
    SteadyStateUserData ud_;
    int ns_;
    // Declared in construction order; the guards tear down in reverse, so the
    // SUNContext outlives everything created from it.
    SunContextGuard ctx_;
    NVectorGuard y_;
    CvodeMemGuard cvode_mem_;
    SUNMatrixGuard A_;
    SUNLinSolGuard LS_;
    sunrealtype t_ = 0.0;
};

static SteadyStateResult solve_by_integration(NetworkModel &model, SteadyStateRhs &rhs,
                                              const SteadyStateOptions &opts,
                                              const ResidualSubspace &sub) {
    SteadyStateMarcher marcher(model, rhs, opts, sub);
    double residual = 0.0;
    const bool converged = marcher.march(opts.tol, &residual);
    return marcher.make_result(converged, residual);
}

// ---------------------------------------------------------------------------
// Tier 2: KINSOL Newton solver (with reduced-space for conservation laws)
// ---------------------------------------------------------------------------

// User data for reduced-space KINSOL
struct ReducedKinsolData {
    SteadyStateRhs *rhs;
    NetworkModel *model;
    const ConservationLaws *cl;
};

// Reconstruct full y from independent species y_ind using conservation laws
static void reconstruct_full(const double *y_ind, double *y_full, int ns,
                             const ConservationLaws &cl, const std::vector<Species> &species) {

    // First, copy independent species into full vector
    for (size_t k = 0; k < cl.independent.size(); ++k) {
        y_full[cl.independent[k]] = y_ind[k];
    }
    // Then, reconstruct dependent species from conservation constraints
    // Σ L[k,i] * y_full[i] = constants[k]
    // L[k,dep] * y_full[dep] = constants[k] - Σ_{i≠dep} L[k,i] * y_full[i]
    for (int k = 0; k < cl.n_laws; ++k) {
        int dep = cl.dependent[k];
        double coeff_dep = cl.coefficients[k][dep];
        if (std::abs(coeff_dep) < 1e-15)
            continue; // degenerate
        double rhs = cl.constants[k];
        for (int i = 0; i < ns; ++i) {
            if (i != dep) {
                rhs -= cl.coefficients[k][i] * y_full[i];
            }
        }
        y_full[dep] = rhs / coeff_dep;
    }
}

// Reduced-space KINSOL RHS: evaluate f(y) for independent species only
static int kinsol_reduced_rhs(N_Vector y_ind, N_Vector fval, void *ud) {
    auto *data = static_cast<ReducedKinsolData *>(ud);
    NetworkModel *model = data->model;
    const auto &cl = *data->cl;
    const int ns = model->n_species();
    const int n_ind = static_cast<int>(cl.independent.size());

    const double *y_ind_data = N_VGetArrayPointer(y_ind);
    double *f_ind = N_VGetArrayPointer(fval);

    // Reconstruct full state vector
    std::vector<double> y_full(ns, 0.0);
    // Initialize with current concentrations (for fixed species)
    const auto &species = model->species();
    for (int i = 0; i < ns; ++i)
        y_full[i] = species[i].concentration;
    reconstruct_full(y_ind_data, y_full.data(), ns, cl, species);

    // Compute full f(y)
    std::vector<double> f_full(ns, 0.0);
    data->rhs->eval(0.0, y_full.data(), f_full.data());

    // Extract independent species residuals
    for (int k = 0; k < n_ind; ++k) {
        f_ind[k] = f_full[cl.independent[k]];
    }
    return 0;
}

// Full-space KINSOL RHS (for models without conservation laws)
static int kinsol_rhs(N_Vector y, N_Vector fval, void *ud) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    const double *yp = N_VGetArrayPointer(y);
    double *fp = N_VGetArrayPointer(fval);
    int ns = data->model->n_species();

    data->rhs->eval(0.0, yp, fp);

    // Zero out fixed species residuals (the codegen RHS already zeroes them;
    // this keeps the interpreted path identical and is a no-op otherwise)
    const auto &species = data->model->species();
    for (const auto &s : species) {
        if (s.fixed) {
            fp[s.index - 1] = 0.0;
        }
    }
    return 0;
}

// `linsolv_failed`, when non-null, reports whether KINSOL failed because it
// could not set up or apply the dense linear solver at all — as opposed to
// merely failing to converge. For a *structurally* singular reduced Jacobian
// (GH #27 Bug 3, e.g. Barua 2013's 404×404 rank-deficient system) that is a
// property of the sparsity pattern, not of the seed, so every subsequent
// attempt fails identically. The two-tier ladder uses this to stop probing
// KINSOL instead of building MAX_NEWTON_ATTEMPTS dense n_ind×n_ind
// factorizations that are all doomed.
static SteadyStateResult solve_by_newton(NetworkModel &model, SteadyStateRhs &rhs,
                                         const SteadyStateOptions &opts,
                                         const ResidualSubspace &sub,
                                         bool *linsolv_failed = nullptr) {

    if (linsolv_failed) {
        *linsolv_failed = false;
    }

    const int ns = model.n_species();
    const auto &cl = model.conservation_laws();
    SteadyStateResult result;
    result.method_used = "newton";
    result.species_names = model.species_names();
    result.concentrations.resize(ns);
    result.n_residual_species = static_cast<int>(sub.included.size());
    result.excluded_species = sub.excluded;

    // ── Recompute conservation constants from CURRENT concentrations ──
    // (important: PSet may have changed ICs since model load time)
    ConservationLaws cl_copy = cl;
    if (!cl_copy.empty()) {
        for (int k = 0; k < cl_copy.n_laws; ++k) {
            double c = 0.0;
            for (int i = 0; i < ns; ++i) {
                c += cl_copy.coefficients[k][i] * model.species()[i].concentration;
            }
            cl_copy.constants[k] = c;
        }
    }

    // ── Which species Newton actually solves for (issue #74) ──────────────────
    // The conservation-law reduction already drops the dependent species; a mask
    // drops the ones whose settling was never asked about. Both mean the same
    // thing here — "not an unknown, hold it at the value integration left" —
    // because reconstruct_full() pre-fills y_full from the model's live
    // concentrations, so an excluded species keeps that value and its equation
    // leaves the system along with it. Leaving a write-only accumulator IN is
    // precisely what makes this system singular at EVERY seed: nothing consumes
    // it, so its Jacobian column is structurally zero (GH #27 Bug 3 called this
    // out on Barua 2013's 404×404 reduced system without naming the cause).
    //
    // A mask with no conservation laws drives the same reduced machinery with a
    // zero-law reduction, so there is one code path for "some species are not
    // unknowns" rather than a second restricted full-space RHS.
    //
    // Both restricted forms also drop `fixed` ($-prefixed) species: their
    // derivative is zeroed by definition, so keeping one as an unknown adds a
    // structurally zero ROW and makes the system singular for a reason that has
    // nothing to do with the model. The unmasked paths do NOT do this (kinsol_rhs
    // zeroes those rows while leaving the unknowns in place, and cl.independent
    // is taken verbatim) and are left exactly as they were — an all-true mask is
    // not masked at all (see make_subspace), so no pre-#74 solve changes path.
    const bool use_reduced = !cl_copy.empty() || sub.masked;
    if (use_reduced) {
        std::vector<int> unknowns;
        if (cl_copy.empty()) {
            cl_copy.n_species = ns;
            for (int i : sub.included) {
                if (!model.species()[i].fixed)
                    unknowns.push_back(i);
            }
        } else if (sub.masked) {
            unknowns.reserve(cl_copy.independent.size());
            for (int i : cl_copy.independent) {
                if (sub.includes(i) && !model.species()[i].fixed)
                    unknowns.push_back(i);
            }
        } else {
            unknowns = cl_copy.independent; // unmasked: verbatim, as before #74
        }
        if (unknowns.empty()) {
            // Every included species is pinned by a conservation law, so there is
            // no Newton system to build. Not an error — report it unconverged so
            // the two-tier ladder answers from integration instead.
            result.converged = false;
            model.get_state_into(result.concentrations.data());
            result.residual = compute_residual(rhs, result.concentrations.data(), ns, sub);
            return result;
        }
        cl_copy.independent = std::move(unknowns);
    }
    const int n_ind = use_reduced ? static_cast<int>(cl_copy.independent.size()) : ns;

    // RAII guards
    SunContextGuard ctx;
    if (!ctx) {
        throw std::runtime_error("SUNContext_Create failed (kinsol)");
    }

    // Route this KINSOL context's error/warning log to the null sink. In the
    // two-tier solver a KINSOL failure (non-convergence, or the unrecoverable
    // dense linear-solver setup failure on a structurally singular reduced
    // Jacobian — GH #27 Bug 3, e.g. Barua 2013's 404×404 rank-deficient system)
    // is EXPECTED and handled by falling back to integration, so its stderr
    // spam is noise. Hard misuse still surfaces via the flag checks below.
    {
        SUNLogger logger = nullptr;
        if (SUNContext_GetLogger(ctx, &logger) == SUN_SUCCESS && logger != nullptr) {
            SUNLogger_SetErrorFilename(logger, bngsim::null_device);
            SUNLogger_SetWarningFilename(logger, bngsim::null_device);
        }
    }

    NVectorGuard y(N_VNew_Serial(n_ind, ctx));
    double *y_data = y.data();

    if (use_reduced) {
        // Extract independent species concentrations
        for (int k = 0; k < n_ind; ++k) {
            y_data[k] = model.species()[cl_copy.independent[k]].concentration;
        }
    } else {
        for (int i = 0; i < ns; ++i) {
            y_data[i] = model.species()[i].concentration;
        }
    }

    NVectorGuard scale(N_VNew_Serial(n_ind, ctx));
    N_VConst(1.0, scale);

    KinsolMemGuard kin_mem(KINCreate(ctx));
    if (!kin_mem) {
        throw std::runtime_error("KINCreate failed");
    }

    int flag;
    ReducedKinsolData rd{&rhs, &model, &cl_copy};
    SteadyStateUserData ud{&rhs, &model};

    if (use_reduced) {
        flag = KINInit(kin_mem, kinsol_reduced_rhs, y);
        KINSetUserData(kin_mem, &rd);
    } else {
        flag = KINInit(kin_mem, kinsol_rhs, y);
        KINSetUserData(kin_mem, &ud);
    }

    if (flag != KIN_SUCCESS) {
        throw std::runtime_error("KINInit failed");
    }

    KINSetFuncNormTol(kin_mem, opts.tol);
    KINSetScaledStepTol(kin_mem, 1e-15); // allow very small steps
    KINSetNumMaxIters(kin_mem, 200);
    // Set max Newton step large enough for the problem scale
    double max_newton_step = 0.0;
    for (int i = 0; i < n_ind; ++i) {
        double v = std::abs(y_data[i]);
        if (v > max_newton_step)
            max_newton_step = v;
    }
    max_newton_step = std::max(max_newton_step * 100.0, 1e6);
    KINSetMaxNewtonStep(kin_mem, max_newton_step);

    SUNMatrixGuard A_guard(SUNDenseMatrix(n_ind, n_ind, ctx));
    SUNLinSolGuard LS_guard(ss_make_dense_linsol(y, A_guard, ctx, model, n_ind));
    KINSetLinearSolver(kin_mem, LS_guard, A_guard);

    // Solve — use KIN_NONE (pure Newton) for reduced systems since the
    // reduced Jacobian is non-singular by construction. For full-space
    // systems, also use KIN_NONE (the auto fallback to integration
    // handles convergence failure gracefully).
    flag = KINSol(kin_mem, y, KIN_NONE, scale, scale);

    // Distinguish "the linear solver is unusable on this system" from ordinary
    // non-convergence — see the note on `linsolv_failed` above.
    if (linsolv_failed) {
        *linsolv_failed = (flag == KIN_LINSOLV_NO_RECOVERY || flag == KIN_LINIT_FAIL ||
                           flag == KIN_LSETUP_FAIL || flag == KIN_LSOLVE_FAIL);
    }

    // Check result
    long int nfe = 0, nni = 0;
    KINGetNumFuncEvals(kin_mem, &nfe);
    KINGetNumNonlinSolvIters(kin_mem, &nni);
    result.n_steps = static_cast<int>(nni);
    result.n_rhs_evals = static_cast<int>(nfe);

    if (flag >= 0) {
        result.converged = true;
    }

    // Reconstruct full concentrations
    if (use_reduced) {
        std::vector<double> y_full(ns, 0.0);
        for (int i = 0; i < ns; ++i)
            y_full[i] = model.species()[i].concentration;
        reconstruct_full(y_data, y_full.data(), ns, cl_copy, model.species());
        for (int i = 0; i < ns; ++i)
            result.concentrations[i] = y_full[i];
    } else {
        for (int i = 0; i < ns; ++i)
            result.concentrations[i] = y_data[i];
    }

    // Compute actual residual using full state and verify convergence.
    //
    // GH #27 Bug 1: the old guard was `if (residual >= tol) converged = false`.
    // When Newton walks a species negative, Hill/power rate laws yield NaN, so
    // compute_residual returns NaN — and `NaN >= tol` is false, so a NaN result
    // was reported converged (returning conc=[nan, …]). Use the positive test
    // `!(residual < tol)` (true for NaN) and additionally reject any non-finite
    // or clearly-negative concentration, so an unphysical Newton root never
    // passes as a converged steady state.
    result.residual = compute_residual(rhs, result.concentrations.data(), ns, sub);
    if (!(result.residual < opts.tol) || !ss_state_is_physical(result.concentrations)) {
        result.converged = false;
    }

    // RAII guards handle cleanup automatically

    return result;
}

// ---------------------------------------------------------------------------
// Two-tier steady-state solver (GH #27): integrate FIRST, then Newton
// ---------------------------------------------------------------------------
//
// The manuscript's two-tier method is tier 1 = CVODE integration with early
// termination, tier 2 = KINSOL Newton polish. The previous default ran them in
// the OPPOSITE order — Newton seeded at the raw initial condition, integration
// only as a non-convergence fallback — so on any model whose f(y)=0 has several
// roots (e.g. kinetic proofreading) Newton would *converge* to a spurious root
// the dynamics never reach and the fallback never fired (GH #27 Bug 2).
//
// Here integration carries the state into the physical root's basin and Newton
// polishes from there. The open question the issue flags is how tight the burst
// must be — it is model-dependent. We make it ADAPTIVE without ever trusting a
// single unvalidated Newton solve: a KINSOL root is accepted only when it is
// *seed-stable*, i.e. two Newton solves from successively tighter integration
// bursts land on the same state (ss_states_agree). A unique-root model confirms
// on the first pair of bursts (fast — the speedup the issue wants to see); a
// multi-root model's Newton roots keep drifting with the seed until the burst
// has essentially converged, so it simply integrates to the physical root. Any
// non-finite / non-converged / structurally-singular (Bug 3) Newton attempt is
// discarded and integration continues. The result is correct on every model,
// with the root-finding speedup surfacing exactly where it is trustworthy.
//
// Burst tolerances form a scale-free ladder: the residual ||f||_2/n at the IC,
// reduced by a decade for the first burst and by two decades per rung after,
// floored at opts.tol. Each rung CONTINUES the previous burst's march on a
// single SteadyStateMarcher — same CVODE memory, so the BDF order and step size
// carry over too, not just the state (integration conserves mass, so the reduced
// Newton's conservation constants, recomputed from that state, are unchanged).
// The total integration work is therefore one march to the tightest rung, not a
// restart per rung; only the stop criterion tightens.

// Newton polish seeded from `seed`; returns the KINSOL result. The model's live
// species are set to `seed` first (solve_by_integration / solve_by_newton read
// the IC from there). KINSOL construction can throw (KINInit/KINCreate) — the
// caller treats a throw as a failed, non-accepted attempt.
static SteadyStateResult ss_newton_from(NetworkModel &model, SteadyStateRhs &rhs,
                                        const SteadyStateOptions &opts, const ResidualSubspace &sub,
                                        const std::vector<double> &seed,
                                        bool *linsolv_failed = nullptr) {
    model.set_state_from(seed.data());
    return solve_by_newton(model, rhs, opts, sub, linsolv_failed);
}

static SteadyStateResult solve_by_newton_two_tier(NetworkModel &model, SteadyStateRhs &rhs,
                                                  const SteadyStateOptions &opts,
                                                  const ResidualSubspace &sub) {
    const int ns = model.n_species();

    // Snapshot the initial condition so the model is restored on return (the
    // public contract: steady_state() without sensitivity leaves ICs intact).
    std::vector<double> ic(ns);
    model.get_state_into(ic.data());

    auto restore = [&]() { model.set_state_from(ic.data()); };

    // Accept a KINSOL result only if it converged to a finite, physical root.
    auto accept = [&](const SteadyStateResult &r) {
        return r.converged && r.method_used == "newton" && (r.residual < opts.tol) &&
               ss_state_is_physical(r.concentrations);
    };

    const double r0 = compute_residual(rhs, ic.data(), ns, sub);

    // IC already at steady state: a single Newton polish (which converges
    // immediately) reports the canonical "newton" without any integration.
    if (r0 < opts.tol) {
        SteadyStateResult r;
        try {
            r = ss_newton_from(model, rhs, opts, sub, ic);
        } catch (...) {
            r.converged = false;
        }
        if (accept(r)) {
            restore();
            return r;
        }
    }

    constexpr double AGREE_RTOL = 1e-4; // seed-stability tolerance (see header)
    constexpr int MAX_RUNGS = 14;
    // Cap on burst-seeded Newton attempts. Two agreeing attempts accept a
    // unique-root model on rung 1; a multi-root model's attempts keep drifting,
    // so after this many we stop probing and just integrate (KINSOL adds only
    // cost there). Correctness is unaffected: integration is always the answer.
    constexpr int MAX_NEWTON_ATTEMPTS = 6;

    std::vector<double> prev_newton;
    bool have_prev = false;
    int newton_attempts = 0;
    // Cleared the moment KINSOL reports it cannot factor the system at all
    // (Bug 3: Barua 2013's reduced Jacobian is structurally singular, so it is
    // singular at EVERY seed). Retrying then only buys MAX_NEWTON_ATTEMPTS
    // doomed dense n_ind×n_ind factorizations, which on a 409-species model is
    // the dominant cost of a solve that was always going to end in integration.
    bool newton_viable = true;

    std::vector<double> seed = ic;
    double bt = std::max(r0 * 0.1, opts.tol);

    // One integrator for the whole ladder: each rung CONTINUES this march (the
    // early-exit Newton probe above may have left the model's live state
    // elsewhere, so re-seed it from the IC the marcher is about to read).
    restore();
    SteadyStateMarcher marcher(model, rhs, opts, sub);

    for (int rung = 0; rung < MAX_RUNGS; ++rung) {
        // Tier 1: continue integrating from the previous rung's end state to bt.
        double burst_residual = 0.0;
        const bool burst_converged = marcher.march(bt, &burst_residual);
        SteadyStateResult burst = marcher.make_result(burst_converged, burst_residual);
        seed = burst.concentrations;

        if (burst.residual < opts.tol) {
            // Integration itself reached the parity tolerance — done.
            restore();
            return burst;
        }
        if (!burst.converged) {
            // Could not reach even this (looser) burst tolerance within max_time
            // (a slow/oscillatory system, or a residual floor above tol like
            // Barua 2013). Integration is the best available answer.
            restore();
            return burst;
        }

        // Tier 2: Newton polish from the burst state. Accept only when a second
        // attempt lands on the same root (seed-stable); the first attempt just
        // seeds the comparison. A raw-IC Newton is never trusted alone — that is
        // exactly the Bug 2 hazard.
        if (newton_viable && newton_attempts < MAX_NEWTON_ATTEMPTS) {
            ++newton_attempts;
            SteadyStateResult nr;
            bool linsolv_failed = false;
            try {
                nr = ss_newton_from(model, rhs, opts, sub, seed, &linsolv_failed);
            } catch (...) {
                // KINCreate / KINInit failed: as unrecoverable as a singular
                // factorization, and just as seed-independent.
                nr.converged = false;
                linsolv_failed = true;
            }
            if (linsolv_failed) {
                newton_viable = false;
            }
            if (accept(nr)) {
                if (have_prev && ss_states_agree(nr.concentrations, prev_newton, AGREE_RTOL)) {
                    restore();
                    return nr; // seed-stable root — accept
                }
                prev_newton = nr.concentrations;
                have_prev = true;
            }
        }

        if (bt <= opts.tol)
            break;
        bt = std::max(bt * 0.01, opts.tol);
    }

    // Ladder exhausted without a seed-stable Newton: continue the same march to
    // the full tolerance (a correct, if slower, answer).
    double fin_residual = 0.0;
    const bool fin_converged = marcher.march(opts.tol, &fin_residual);
    SteadyStateResult fin = marcher.make_result(fin_converged, fin_residual);
    restore();
    return fin;
}

// ---------------------------------------------------------------------------
// Steady-state sensitivity: dY_ss/dp = -J^{-1} * df/dp
// ---------------------------------------------------------------------------

// dY_ss/dp = -J⁻¹·(∂f/∂p) by the implicit function theorem.
//
// Both factors prefer closed form and fall back to differencing (issue #63):
//
//   J      — the compiled `bngsim_codegen_jac`, else the interpreted
//            fill_dense_analytical_jacobian when analytical_jacobian_complete,
//            else one-sided finite differences (n_species RHS evaluations).
//            This is the same "analytical when complete, FD otherwise" rule
//            jacobian="auto" applies everywhere else, and the same matrix the
//            newton path's KINSOL polish already uses.
//   ∂f/∂p  — the analytical column the codegen sensitivity RHS emits (see
//            SteadyStateRhs::eval_dfdp), else one-sided finite differences in
//            the parameter (one RHS evaluation per parameter).
//
// Before #63 both were unconditionally finite-differenced, at a fixed
// √eps step, through the *interpreted* RHS — ~1300 ExprTk RHS evaluations to
// assemble one Jacobian on a 1281-species model, for a matrix the model already
// had in closed form. Which path ran is now recorded on the result
// (sens_jacobian_source / sens_dfdp_source) rather than being invisible.
// Issue #74 — a mask restricts this solve to the same subspace the convergence
// test used. It has to: a write-only accumulator contributes a structurally zero
// Jacobian COLUMN, so including it makes -J⁻¹ singular at every root and the
// gradient comes back NaN (which is honest, but useless). Restricting the system
// to the included species, with the excluded ones held at the values integration
// left, is exact whenever nothing else's derivative reads them — which is clause
// 3 of NetworkModel::pure_sink_species(), so the recommended mask satisfies it by
// construction. Excluded species get a NaN row: a species with no steady value
// has no steady-state gradient, and 0.0 would be a confident wrong answer.
static void compute_ss_sensitivity(NetworkModel &model, SteadyStateRhs &rhs,
                                   SteadyStateResult &result,
                                   const std::vector<std::string> &param_names,
                                   const std::string &opts_jacobian, const ResidualSubspace &sub) {

    const int ns = model.n_species();
    const int np = static_cast<int>(param_names.size());
    if (ns == 0 || np == 0)
        return;

    const double *y_ss = result.concentrations.data();
    const auto &params = model.parameters();

    // Map param names to indices
    std::vector<int> pidx(np);
    for (int p = 0; p < np; ++p) {
        bool found = false;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].name == param_names[p]) {
                pidx[p] = static_cast<int>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Steady-state sensitivity: parameter '" + param_names[p] +
                                     "' not found");
        }
    }

    // ── Step 1: dense Jacobian J at y_ss, column-major (J[j*ns+i] = ∂f_i/∂x_j) ──
    std::vector<double> J(static_cast<size_t>(ns) * ns, 0.0);
    std::vector<double> f0(ns), f1(ns), y_pert(ns);
    const double eps = 1.4901161193847656e-8; // sqrt(machine eps)

    // jacobian="fd" pins the finite-difference assembly, the same escape hatch it
    // is everywhere else in the library (and the A/B lever for checking the
    // closed-form path against the one that predates #63). "jax" has no
    // steady-state analogue — there is no Python callback plumbed through here —
    // so it also takes the FD path rather than pretending otherwise.
    const bool want_analytical_jac = (opts_jacobian == "auto" || opts_jacobian == "analytical");
    result.sens_jacobian_source = want_analytical_jac ? rhs.jacobian_source() : "finite-difference";
    if (want_analytical_jac && rhs.has_analytical_jacobian()) {
        rhs.fill_dense_jacobian(0.0, y_ss, J.data());
    } else {
        // Finite differences: works for every rate-law type, at the cost of one
        // RHS evaluation per column. J[:,j] = (f(y + h·e_j) − f(y)) / h.
        rhs.eval(0.0, y_ss, f0.data());
        for (int j = 0; j < ns; ++j) {
            std::memcpy(y_pert.data(), y_ss, ns * sizeof(double));
            double h = eps * std::max(std::abs(y_ss[j]), 1.0);
            y_pert[j] += h;
            rhs.eval(0.0, y_pert.data(), f1.data());
            for (int i = 0; i < ns; ++i) {
                J[static_cast<size_t>(j) * ns + i] = (f1[i] - f0[i]) / h; // column-major
            }
        }
    }

    // ── Step 2: ∂f/∂p for each sensitivity parameter ──────────────────────────
    std::vector<double> dfdp(static_cast<size_t>(ns) * np, 0.0); // column-major: dfdp[p*ns+i]

    if (rhs.has_analytical_dfdp()) {
        result.sens_dfdp_source = "codegen";
        for (int p = 0; p < np; ++p) {
            rhs.eval_dfdp(0.0, y_ss, pidx[p], dfdp.data() + static_cast<size_t>(p) * ns);
        }
    } else {
        // Finite differences in the parameter, at the fixed steady state:
        // ∂f/∂p[:,p] = (f(y_ss; p+h) − f(y_ss; p)) / h. sync_params() re-derives
        // constant-expression parameters after each write, so a rate law stored
        // as a derived `_rateLaw{N}` picks up the chain rule (issue #2).
        result.sens_dfdp_source = "finite-difference";
        rhs.sync_params();
        rhs.eval(0.0, y_ss, f0.data());

        for (int p = 0; p < np; ++p) {
            int pi = pidx[p];
            double pval = params[pi].value;
            double h = eps * std::max(std::abs(pval), 1.0);

            // Perturb parameter, holding it against re-derivation so a probe of a
            // derived parameter is not immediately undone.
            const_cast<std::vector<Parameter> &>(params)[pi].value = pval + h;
            rhs.sync_params(pi);
            rhs.eval(0.0, y_ss, f1.data());

            // Restore parameter
            const_cast<std::vector<Parameter> &>(params)[pi].value = pval;
            rhs.sync_params(pi);

            for (int i = 0; i < ns; ++i) {
                dfdp[static_cast<size_t>(p) * ns + i] = (f1[i] - f0[i]) / h;
            }
        }
    }

    // Step 3: Solve J * sens[:,p] = -dfdp[:,p] for each parameter.
    //
    // If conservation laws are present, J is singular (rank-deficient).
    // Use reduced-space solve: extract independent rows/cols from J and df/dp,
    // solve the reduced system, then reconstruct dependent species from
    // conservation constraints (their sensitivity follows from the chain rule).

    const auto &cl = model.conservation_laws();

    result.sensitivity.resize(ns * np, 0.0);
    result.sens_param_names = param_names;
    result.n_sens_params = np;

    if (!cl.empty() || sub.masked) {
        // Reduced-space sensitivity solve.
        //
        // `solve_idx` is the unknown set: the conservation-law independents,
        // narrowed to the species the mask kept (issue #74). A mask with no
        // conservation laws lands here too, with every law loop below a no-op —
        // one code path for "some species are not unknowns", mirroring
        // solve_by_newton.
        // `fixed` species are dropped from the restricted forms for the same
        // reason as in solve_by_newton — a zeroed derivative is a zero row — and
        // their dY_ss/dp is genuinely 0: a boundary condition does not move with
        // a rate constant. Their rows stay 0 rather than NaN, since it is the
        // mask, not fixedness, that means "no answer". The unmasked branch keeps
        // cl.independent verbatim, exactly as before #74.
        std::vector<int> solve_idx;
        if (cl.empty()) {
            for (int i : sub.included) {
                if (!model.species()[i].fixed)
                    solve_idx.push_back(i);
            }
        } else if (sub.masked) {
            solve_idx.reserve(cl.independent.size());
            for (int i : cl.independent) {
                if (sub.includes(i) && !model.species()[i].fixed)
                    solve_idx.push_back(i);
            }
        } else {
            solve_idx = cl.independent;
        }
        if (solve_idx.empty()) {
            // Nothing left to differentiate (every included species is pinned by
            // a law). Report NaN rather than a zero matrix a fitter would read as
            // "this parameter does not matter".
            std::fill(result.sensitivity.begin(), result.sensitivity.end(),
                      std::numeric_limits<double>::quiet_NaN());
            return;
        }
        const int n_ind = static_cast<int>(solve_idx.size());

        // Build the reduced Jacobian J_red (n_ind × n_ind) by projecting the full
        // J through the conservation-law reconstruction, rather than running a
        // SECOND finite-difference sweep over the reduced residual:
        //
        //   J_red[i][j] = ∂f_{ind_i}/∂y_{ind_j}
        //               = J[ind_i][ind_j] + Σ_k J[ind_i][dep_k] · D[k][j]
        //
        // where D[k][j] = ∂y_{dep_k}/∂y_{ind_j} is exactly what differentiating
        // reconstruct_full() gives. That is the chain rule the old comment here
        // described and then abandoned ("for simplicity and robustness, use FD on
        // the reduced residual directly"); with a closed-form J to project it is
        // both exact and cheaper — the FD sweep cost n_ind more RHS evaluations
        // on top of the ns that had already built a full J the reduced path then
        // never looked at.
        //
        // D follows reconstruct_full()'s ordering: law k solves for its dependent
        // from every OTHER species, so it sees the dependents of laws k' < k
        // already updated (derivative D[k'][j]) and those of laws k' > k still at
        // their unperturbed values (derivative 0). A degenerate law (|L[k,dep]|
        // below the reconstruction floor) is skipped there, so its row stays 0.
        std::vector<double> D(static_cast<size_t>(cl.n_laws) * n_ind, 0.0);
        for (int k = 0; k < cl.n_laws; ++k) {
            const int dep = cl.dependent[k];
            const double coeff_dep = cl.coefficients[k][dep];
            if (std::abs(coeff_dep) < 1e-15)
                continue; // degenerate — reconstruct_full skips it too
            double *Dk = D.data() + static_cast<size_t>(k) * n_ind;
            for (int j = 0; j < n_ind; ++j) {
                // i = ind_j contributes L[k][ind_j]·1; every other independent
                // contributes 0.
                double acc = cl.coefficients[k][solve_idx[j]];
                // i = dep_{k'} for k' < k contributes L[k][dep_k']·D[k'][j].
                for (int kp = 0; kp < k; ++kp) {
                    const int dep_p = cl.dependent[kp];
                    if (dep_p == dep)
                        continue; // excluded by the i ≠ dep sum
                    acc += cl.coefficients[k][dep_p] * D[static_cast<size_t>(kp) * n_ind + j];
                }
                Dk[j] = -acc / coeff_dep;
            }
        }

        std::vector<double> J_red(static_cast<size_t>(n_ind) * n_ind, 0.0);
        for (int j = 0; j < n_ind; ++j) {
            double *col = J_red.data() + static_cast<size_t>(j) * n_ind; // column-major
            const double *Jcol_ind = J.data() + static_cast<size_t>(solve_idx[j]) * ns;
            for (int i = 0; i < n_ind; ++i) {
                col[i] = Jcol_ind[solve_idx[i]];
            }
            for (int k = 0; k < cl.n_laws; ++k) {
                const double d = D[static_cast<size_t>(k) * n_ind + j];
                if (d == 0.0)
                    continue;
                const double *Jcol_dep = J.data() + static_cast<size_t>(cl.dependent[k]) * ns;
                for (int i = 0; i < n_ind; ++i) {
                    col[i] += Jcol_dep[solve_idx[i]] * d;
                }
            }
        }

        // Build reduced df/dp
        std::vector<double> dfdp_red(n_ind * np, 0.0);
        for (int p = 0; p < np; ++p)
            for (int i = 0; i < n_ind; ++i)
                dfdp_red[p * n_ind + i] = dfdp[p * ns + solve_idx[i]];

        // Solve J_red * sens_ind = -dfdp_red using SUNDIALS with RAII guards.
        //
        // ONE factorization for all np right-hand sides. The loop used to re-copy
        // J_red into A and re-run SUNLinSolSetup on every parameter, paying np
        // dense n_ind³ LU factorizations of the SAME matrix — on a 1276-wide
        // reduced system that is the dominant cost of the whole assembly. Dense
        // LU solve does not modify the stored factors, so hoisting is a pure win.
        SunContextGuard ctx;
        SUNMatrixGuard A_guard(SUNDenseMatrix(n_ind, n_ind, ctx));
        NVectorGuard bv(N_VNew_Serial(n_ind, ctx));
        NVectorGuard xv(N_VNew_Serial(n_ind, ctx));
        SUNLinSolGuard LS_guard(ss_make_dense_linsol(xv, A_guard, ctx, model, n_ind));

        sunrealtype *A_data = SUNDenseMatrix_Data(A_guard);
        std::memcpy(A_data, J_red.data(), static_cast<size_t>(n_ind) * n_ind * sizeof(double));
        SUNLinSolSetup(LS_guard, A_guard);
        result.sens_jacobian_rcond = lu_diag_rcond(A_data, n_ind);

        for (int p = 0; p < np; ++p) {
            double *b_data = N_VGetArrayPointer(bv);
            for (int i = 0; i < n_ind; ++i)
                b_data[i] = -dfdp_red[p * n_ind + i];
            SUNLinSolSolve(LS_guard, A_guard, xv, bv, 0.0);
            const double *x_data = N_VGetArrayPointer(xv);

            // Fill the solved species' sensitivity
            for (int i = 0; i < n_ind; ++i)
                result.sensitivity[solve_idx[i] * np + p] = x_data[i];

            // Reconstruct dependent species sensitivity from conservation:
            // Σ L[k,i] * dy_i/dp = 0 → dy_dep/dp = -(1/L[k,dep]) * Σ_{i≠dep} L[k,i] * dy_i/dp
            for (int k = 0; k < cl.n_laws; ++k) {
                int dep = cl.dependent[k];
                double cd = cl.coefficients[k][dep];
                if (std::abs(cd) < 1e-15)
                    continue;
                double s = 0.0;
                for (int i = 0; i < ns; ++i)
                    if (i != dep)
                        s += cl.coefficients[k][i] * result.sensitivity[i * np + p];
                result.sensitivity[dep * np + p] = -s / cd;
            }
        }

        // A masked-out species was held fixed above, which is what made the
        // system solvable — but "held fixed" is not a gradient. Its row read 0
        // through the dependent reconstruction (correct: the law sees it as a
        // constant) and is overwritten with NaN now that the reconstruction is
        // done, because a species with no steady value has no ∂x*/∂p to report.
        for (int i : sub.excluded) {
            for (int p = 0; p < np; ++p)
                result.sensitivity[static_cast<size_t>(i) * np + p] =
                    std::numeric_limits<double>::quiet_NaN();
        }

        // RAII guards handle cleanup
    } else {
        // Full-space solve (no conservation laws, still using RAII guards)
        SunContextGuard ctx;
        SUNMatrixGuard A_guard(SUNDenseMatrix(ns, ns, ctx));
        sunrealtype *A_data = SUNDenseMatrix_Data(A_guard);
        std::memcpy(A_data, J.data(), ns * ns * sizeof(double));

        NVectorGuard b(N_VNew_Serial(ns, ctx));
        NVectorGuard x(N_VNew_Serial(ns, ctx));
        SUNLinSolGuard LS_guard(ss_make_dense_linsol(x, A_guard, ctx, model, ns));
        // One factorization, np solves — see the note on the reduced branch.
        SUNLinSolSetup(LS_guard, A_guard);
        result.sens_jacobian_rcond = lu_diag_rcond(A_data, ns);

        for (int p = 0; p < np; ++p) {
            double *b_data = N_VGetArrayPointer(b);
            for (int i = 0; i < ns; ++i)
                b_data[i] = -dfdp[p * ns + i];
            SUNLinSolSolve(LS_guard, A_guard, x, b, 0.0);
            const double *x_data = N_VGetArrayPointer(x);
            for (int i = 0; i < ns; ++i)
                result.sensitivity[i * np + p] = x_data[i];
        }

        // RAII guards handle cleanup
    }
}

// ---------------------------------------------------------------------------
// Steady-state OUTPUT sensitivities (GH #12)
// ---------------------------------------------------------------------------
//
// Project the species sensitivity dY_ss/dp (from compute_ss_sensitivity) onto
// the model's observables and global functions, so a gradient consumer can read
// d(observable)/dp and d(function)/dp directly instead of re-deriving the output
// Jacobian:
//
//   d(obs_j)/dp  = Σ_i (∂obs_j/∂x_i)·dY_ss_i/dp                (exact; linear groups)
//   d(func_m)/dp = Σ_i (∂func_m/∂x_i)·dY_ss_i/dp + ∂func_m/∂p
//
// Observables are Σ factor·x, so ∂obs/∂x is exactly the group factor and the
// observable projection is exact.
//
// The function total derivative prefers the compiled chain rule and falls back to
// finite differences per function (issue #75), the same shape ∂f/∂p has in
// compute_ss_sensitivity since #63:
//
//   preferred — `bngsim_codegen_output_sens` (GH #198), the evaluator the CVODES
//               forward-sensitivity path already uses, handed the solved dY_ss/dp
//               columns. Both terms are closed form, in the same artifact as the
//               RHS, so value and derivative cannot diverge.
//   fallback  — the finite-difference primitive compute_ss_sensitivity uses: the
//               state-chain Jacobian ∂func/∂x from per-species perturbations, plus
//               the function's explicit parameter dependence ∂func/∂p (e.g.
//               `k3/(K4+G)` w.r.t. k3) from per-parameter perturbations at the
//               fixed steady state. BOTH terms are needed for the total
//               derivative; dropping either is a confidently wrong gradient.
//
// The fallback is not vestigial. It answers a whole-model decline (no compiled
// artifact, an artifact built without `_want_output_sens`, or `_analyze_output_sens`
// declining on rateOf / an embedded table-function wrapper) AND the per-function
// cases inside a live artifact: a function the emitter marked unsupported (written
// NaN) and one it left outside the user-function closure (left untouched). Both are
// detected the same way — pre-fill the buffer with NaN, and any row that comes back
// non-finite is one the compiled evaluator did not answer. Rows it did answer are
// skipped by the FD sweep, so a model where every function is supported never pays
// for the ns species perturbations at all.
//
// The ∂func/∂p probe goes through SteadyStateRhs::sync_params for the same reason
// the ∂f/∂p probe does (issue #63, and #2 before it): neither update_observables
// nor evaluate_functions re-derives ConstantExpression parameters — only
// set_param does — so writing `kon` straight into the Parameter vector leaves a
// BNG2.pl-derived `_rateLaw{N} = chi*kon` at its nominal value, and a function
// that reads `_rateLaw{N}` silently loses the ∂func/∂_rateLaw1·∂_rateLaw1/∂kon
// chain-rule term. sync_params(pi) re-derives everything except the probed
// parameter itself, matching set_param's detach-then-refresh rule; the restore
// needs the same call, or the derived parameters stay at their perturbed values
// and corrupt every later column and the caller's model. The compiled evaluator
// needs none of that — it carries the derived-parameter chain rule in closed form
// (`plist[c]` selects the primary, and the emitter expands ∂p_derived/∂primary).
//
// Getting the symbol here at all took a Python-side re-prep: it is emitted only
// when the model carries `_want_output_sens`, which Simulator.__init__ set from its
// CONSTRUCTOR sensitivity_params, while steady_state() takes its own
// sensitivity_params as a METHOD argument. Simulator._prepare_output_sens_codegen()
// now runs the GH #205 dance (set the flag, drop a plain-RHS artifact that would
// shadow the sensitivity one, regenerate, restore on failure) for both entry
// points; without it the ordinary
// `Simulator(m, method="ode").steady_state(sensitivity_params=[...])` call lands
// here with no symbol to call.
//
// Precondition: result.sensitivity is populated and the model species are set to
// the steady state. The model is left evaluated at the steady state with the
// original parameter values on return.
static void compute_ss_output_sensitivity(NetworkModel &model, SteadyStateRhs &rhs,
                                          SteadyStateResult &result,
                                          const std::vector<std::string> &param_names) {
    const int ns = model.n_species();
    const int np = static_cast<int>(param_names.size());
    const int n_obs = model.n_observables();
    const int n_func = model.n_functions();
    if (np == 0 || result.sensitivity.empty()) {
        return;
    }

    result.observable_names = model.observable_names();
    result.function_names = model.function_names();

    const double *y_ss = result.concentrations.data();

    // ── Observables: exact linear projection through the group factors ────────
    // obs_j = Σ_{(i,f) ∈ group_j} f·v_i·x_i  ⇒  d(obs_j)/dp = Σ f·v_i·dY_ss_i/dp,
    // where v_i is the amount-valued volume factor (issue #119).
    //
    // v_i is NOT cosmetic and NOT 1 in general. update_observables() — the single
    // site defining what an observable's VALUE is — multiplies an amount-valued
    // species (SBML hasOnlySubstanceUnits="true", GH #75-SBML) by its
    // volume_factor, because such a symbol denotes the species's amount rather
    // than the stored concentration. A derivative that omits it is not the
    // derivative of the value the same result reports: it is off by the
    // compartment volume, and inversely proportional to a quantity the true answer
    // does not depend on at all. The CVODES path (obs_sens_terms) and the compiled
    // emitter (_emit_obs_sens_lines) both already fold it in, so this site was the
    // lone dissenter — and within one SteadyStateResult its `expression:` rows
    // were right (both of their paths carry the factor) while `observable:` rows
    // were wrong.
    //
    // v_i = 1 for every .net model (amount_valued is SBML-only), for V_c = 1, and
    // for hOSU=false, so this is a no-op everywhere it was previously correct.
    if (n_obs > 0) {
        result.observable_sensitivity.assign(static_cast<size_t>(n_obs) * np, 0.0);
        const auto &observables = model.observables();
        const auto &species_list = model.species();
        for (int j = 0; j < n_obs; ++j) {
            double *out = result.observable_sensitivity.data() + static_cast<size_t>(j) * np;
            for (const auto &entry : observables[j].entries) {
                const int i = entry.species_index - 1; // group entries are 1-based
                if (i < 0 || i >= ns) {
                    continue;
                }
                const auto &sp = species_list[i];
                const double weight =
                    sp.amount_valued ? entry.factor * sp.volume_factor : entry.factor;
                const double *dxi = result.sensitivity.data() + static_cast<size_t>(i) * np;
                for (int p = 0; p < np; ++p) {
                    out[p] += weight * dxi[p];
                }
            }
        }
    }

    // ── Functions: compiled chain rule, differenced where it declines ─────────
    if (n_func > 0) {
        result.function_sensitivity.assign(static_cast<size_t>(n_func) * np, 0.0);
        const double eps = 1.4901161193847656e-8; // sqrt(machine eps)
        const double nan = std::numeric_limits<double>::quiet_NaN();

        // param_names[p] was validated to exist by compute_ss_sensitivity, which
        // throws on an unknown name — so a -1 here cannot happen on the live path.
        // It is still checked below rather than asserted, since both branches have
        // a correct answer for it (skip the parameter's explicit term).
        const auto &params = model.parameters();
        std::vector<int> pidx(np, -1);
        for (int p = 0; p < np; ++p) {
            for (size_t k = 0; k < params.size(); ++k) {
                if (params[k].name == param_names[p]) {
                    pidx[p] = static_cast<int>(k);
                    break;
                }
            }
        }

        // Which function rows the compiled evaluator did not answer. Everything,
        // until it says otherwise.
        std::vector<uint8_t> need_fd(n_func, 1);
        int n_need_fd = n_func;
        bool any_codegen = false;

        // ── Preferred: the compiled d(func)/dθ (issue #75) ────────────────────
        if (rhs.has_analytical_output_sens() &&
            std::all_of(pidx.begin(), pidx.end(), [](int i) { return i >= 0; })) {
            rhs.sync_params();

            // The ABI wants one contiguous n_species column per sensitivity
            // direction; result.sensitivity is species-major (sensitivity[i*np+p]),
            // so transpose it into np columns. plist carries the differentiated
            // parameter index for each — every column here is a parameter column
            // (the >= n_params sentinel marks an IC column, which the steady-state
            // path has none of: ∂x*/∂x(0) = 0 at a stable root).
            std::vector<double> sens_cols(static_cast<size_t>(np) * ns);
            std::vector<const double *> col_ptrs(np);
            for (int p = 0; p < np; ++p) {
                double *col = sens_cols.data() + static_cast<size_t>(p) * ns;
                for (int i = 0; i < ns; ++i) {
                    col[i] = result.sensitivity[static_cast<size_t>(i) * np + p];
                }
                col_ptrs[p] = col;
            }

            // NaN pre-fill is the detector: the emitter writes NaN for a function
            // it declined and leaves one outside the user closure untouched, so a
            // finite value is exactly "the compiled evaluator answered this row".
            std::vector<double> fs(static_cast<size_t>(np) * n_func, nan);
            rhs.eval_output_sens(0.0, y_ss, col_ptrs.data(), pidx.data(), np,
                                 /*obs_sens_out=*/nullptr, fs.data());

            for (int m = 0; m < n_func; ++m) {
                bool row_ok = true;
                for (int p = 0; p < np && row_ok; ++p) {
                    row_ok = std::isfinite(fs[static_cast<size_t>(p) * n_func + m]);
                }
                if (!row_ok) {
                    continue; // leave need_fd[m] set
                }
                for (int p = 0; p < np; ++p) {
                    result.function_sensitivity[static_cast<size_t>(m) * np + p] =
                        fs[static_cast<size_t>(p) * n_func + m];
                }
                need_fd[m] = 0;
                --n_need_fd;
                any_codegen = true;
            }
        }

        result.sens_output_source =
            !any_codegen ? "finite-difference" : (n_need_fd == 0 ? "codegen" : "mixed");

        // ── Fallback: finite-difference total derivative, per declined row ────
        // Skipped entirely when the compiled evaluator answered every function —
        // that is where the ns species perturbations (a full observable + function
        // re-evaluation each) stop being paid.
        if (n_need_fd > 0) {
            // A declined row may hold the NaN the evaluator wrote; the FD sweep
            // accumulates, so clear it back to zero first.
            for (int m = 0; m < n_func; ++m) {
                if (need_fd[m]) {
                    for (int p = 0; p < np; ++p) {
                        result.function_sensitivity[static_cast<size_t>(m) * np + p] = 0.0;
                    }
                }
            }

            // Base function values at the steady state with the original
            // parameters. function_value_cache() returns a reference reused by
            // every subsequent evaluate_functions() call, so snapshot it into f0.
            // sync_params() first, so the baseline is taken against freshly
            // derived expression parameters — the same ordering
            // compute_ss_sensitivity uses for its own f0.
            rhs.sync_params();
            model.update_observables(y_ss);
            model.evaluate_functions(0.0);
            const std::vector<double> f0(model.function_value_cache());
            std::vector<double> f1;
            std::vector<double> y_pert(ns);

            // State-chain term: ∂func_m/∂x_i via one-sided FD (perturb one
            // species, re-evaluate observables + functions), folded into
            // Σ_i (∂func_m/∂x_i)·dY_ss_i/dp as each species column is produced.
            for (int i = 0; i < ns; ++i) {
                std::memcpy(y_pert.data(), y_ss, ns * sizeof(double));
                const double h = eps * std::max(std::abs(y_ss[i]), 1.0);
                y_pert[i] += h;
                model.update_observables(y_pert.data());
                model.evaluate_functions(0.0);
                f1 = model.function_value_cache();
                const double *dxi = result.sensitivity.data() + static_cast<size_t>(i) * np;
                for (int m = 0; m < n_func; ++m) {
                    if (!need_fd[m]) {
                        continue;
                    }
                    const double dfm_dxi = (f1[m] - f0[m]) / h;
                    double *out = result.function_sensitivity.data() + static_cast<size_t>(m) * np;
                    for (int p = 0; p < np; ++p) {
                        out[p] += dfm_dxi * dxi[p];
                    }
                }
            }

            // Explicit-parameter term: ∂func_m/∂p at the fixed steady state
            // (perturb one parameter, keep the state fixed). Observables are
            // functions of species only, so update_observables(y_ss) restores the
            // same totals; the function evaluator picks up the live parameter
            // value.
            for (int p = 0; p < np; ++p) {
                const int pi = pidx[p];
                if (pi < 0) {
                    continue;
                }
                const double pval = params[pi].value;
                const double h = eps * std::max(std::abs(pval), 1.0);
                // Perturb, re-deriving every expression parameter but the probed
                // one.
                const_cast<std::vector<Parameter> &>(params)[pi].value = pval + h;
                rhs.sync_params(pi);
                model.update_observables(y_ss);
                model.evaluate_functions(0.0);
                f1 = model.function_value_cache();
                // Restore, and re-derive again — otherwise the derived parameters
                // keep the perturbed values this probe just gave them.
                const_cast<std::vector<Parameter> &>(params)[pi].value = pval;
                rhs.sync_params(pi);
                for (int m = 0; m < n_func; ++m) {
                    if (!need_fd[m]) {
                        continue;
                    }
                    result.function_sensitivity[static_cast<size_t>(m) * np + p] +=
                        (f1[m] - f0[m]) / h;
                }
            }
        }

        // Leave the model evaluated at the steady state with original parameters.
        model.update_observables(y_ss);
        model.evaluate_functions(0.0);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

SteadyStateResult find_steady_state(NetworkModel &model, const SteadyStateOptions &opts) {

    const int ns = model.n_species();
    if (ns == 0) {
        throw std::runtime_error("Cannot find steady state: model has no species");
    }

    // Normalize and validate method. "kinsol" is an input alias for "newton";
    // "auto" was removed (newton already means try-Newton-then-parity-fallback).
    std::string method = opts.method;
    if (method == "kinsol") {
        method = "newton";
    }
    if (method != "integration" && method != "newton") {
        throw std::runtime_error("Invalid steady-state method '" + opts.method +
                                 "'. "
                                 "Must be \"newton\", \"integration\", or \"kinsol\" "
                                 "(alias for \"newton\").");
    }

    // Which species the convergence test covers (issue #74). Validated up front
    // so a wrong-length mask is a clear error rather than a solve that quietly
    // tested the wrong species.
    const ResidualSubspace sub = make_subspace(opts, ns);

    // Resolve the RHS backend once for the whole solve: the codegen artifact is
    // loaded (or JIT-compiled) a single time and shared by the march, the KINSOL
    // polish, the residual check and the sensitivity assembly (issue #63).
    SteadyStateRhs rhs(model, opts);

    SteadyStateResult result;

    if (method == "integration") {
        // Default: CVODE marched to the BNG2.pl parity criterion.
        result = solve_by_integration(model, rhs, opts, sub);
    } else {
        // "newton": two-tier integrate-first solver (GH #27). A short CVODE
        // burst carries the state into the physical root's basin, then KINSOL
        // polishes; the polish is accepted only once it is seed-stable (agrees
        // across two successively tighter bursts), otherwise integration
        // continues. This is correct on multi-root and NaN-prone models where
        // the old Newton-first ordering returned spurious / non-finite roots.
        // Opt in for the tighter residual the polish delivers; it costs more
        // wall clock than plain integration (GH #28).
        result = solve_by_newton_two_tier(model, rhs, opts, sub);
    }

    result.rhs_backend = rhs.backend();

    // ── Why did it fail? (issue #74) ──────────────────────────────────────────
    // A failed solve used to say only converged=false, which reads as "needs more
    // time" — the one thing that cannot help when the residual has a structural
    // floor. If any write-only accumulator was IN the convergence test and is
    // carrying flux at the returned state, name it: that is the whole answer, and
    // finding it by hand on a 409-species network is a long afternoon.
    //
    // The bar is "this species ALONE keeps the residual above tol": since
    // ||f||₂/n ≥ |f_i|/n, that is |f_i| > tol·n_included. A sink whose production
    // has genuinely stopped (every producing reaction dead at the root) is
    // therefore not reported — it is not what is holding the residual up.
    if (!result.converged) {
        const std::vector<int> sinks = model.pure_sink_species();
        if (!sinks.empty()) {
            std::vector<double> f(ns, 0.0);
            rhs.eval(0.0, result.concentrations.data(), f.data());
            const auto &names = result.species_names;
            const double floor = opts.tol * static_cast<double>(sub.included.size());
            for (int i : sinks) {
                if (sub.includes(i) && std::abs(f[i]) > floor) {
                    result.unconverged_pure_sinks.push_back(
                        i < static_cast<int>(names.size()) ? names[i] : std::to_string(i));
                }
            }
        }
    }

    // Compute sensitivity if requested and converged
    if (result.converged && !opts.sensitivity_params.empty()) {
        // Update model state to steady-state values for sensitivity
        auto &species = const_cast<std::vector<Species> &>(model.species());
        for (int i = 0; i < ns; ++i) {
            species[i].concentration = result.concentrations[i];
        }
        compute_ss_sensitivity(model, rhs, result, opts.sensitivity_params, opts.jacobian, sub);
        // GH #12 — project dY_ss/dp onto observables/functions for direct
        // d(output)/dp access (mirrors Result.output_sensitivities).
        compute_ss_output_sensitivity(model, rhs, result, opts.sensitivity_params);
    }

    return result;
}

} // namespace bngsim
