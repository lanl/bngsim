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
#include "bngsim/atol_vector.hpp"
#include "bngsim/codegen_abi.hpp"
#include "bngsim/dense_eigenvalues.hpp"
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

#ifdef BNGSIM_HAS_KLU
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>
#endif

#include "bngsim/lapack_dense_linsol.hpp"
#include "bngsim/platform_compat.hpp"
#include "bngsim/sparse_jacobian.hpp"
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
        // sparse/KLU-routed one (GH #162). Both are resolved, and each of the
        // two fills below converts from whichever shape this model has —
        // otherwise a model would silently drop back to the interpreted Jacobian
        // whenever the shape it needs is not the shape it was emitted in. That
        // happens in both directions: the sensitivity solve and the KINSOL
        // polish always factor densely (so a CSC artifact is scattered into a
        // dense buffer), and since issue #128 the march may factor sparsely
        // under force_sparse_linear_solver on a model the codegen emitted dense
        // (so a dense artifact is gathered into the CSC values).
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
    //
    // void, not int, so the compiled Jacobian's return code is dropped here —
    // as it is in fill_sparse_jacobian and eval(), and unlike the three
    // cvode_simulator.cpp mirrors, which propagate it. That is a decision and
    // not an oversight: the emitter's ONLY return statement is `return 0;`, so
    // there is no code to carry. The reasoning, and what has to change if that
    // ever stops being true, is in bngsim/codegen_abi.hpp; the invariant it
    // rests on is enforced by test_emitted_c_has_no_nonzero_return rather than
    // left to this comment.
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

    // CSC values, length jacobian_sparsity().nnz, indexed by the pattern's data
    // index — what KLU factors on the march's sparse route (issue #128). The
    // mirror of fill_dense_jacobian, and it has the same precondition
    // (has_analytical_jacobian()), the same self-zeroing contract (every branch
    // clears the buffer it fills) and the same dropped return code, for the
    // reason given there and in bngsim/codegen_abi.hpp.
    void fill_sparse_jacobian(double t, const double *y, double *vals) {
        if (jac_sparse_fn_) {
            jac_sparse_fn_(t, const_cast<double *>(y), vals, &so_data_);
            return;
        }
        if (jac_fn_) {
            // Dense column-major → CSC. The compiled dense fill is emitted for
            // models the auto rule routes DENSE, so this runs only under
            // force_sparse_linear_solver; it is still the compiled derivative,
            // which beats dropping to the interpreted sparse fill.
            const int ns = model_.n_species();
            const auto &sp = model_.jacobian_sparsity();
            dense_vals_.assign(static_cast<size_t>(ns) * ns, 0.0);
            jac_fn_(t, const_cast<double *>(y), dense_vals_.data(), &so_data_);
            for (int col = 0; col < ns; ++col) {
                for (int k = sp.col_ptrs[col]; k < sp.col_ptrs[col + 1]; ++k) {
                    vals[k] = dense_vals_[static_cast<size_t>(col) * ns + sp.row_indices[k]];
                }
            }
            return;
        }
        model_.fill_sparse_analytical_jacobian(t, y, vals);
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
    std::vector<double> csc_vals_;   // CSC → dense Jacobian scatter buffer
    std::vector<double> dense_vals_; // dense → CSC Jacobian gather buffer (#128)
    int plist_[1] = {0};
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct SteadyStateUserData {
    SteadyStateRhs *rhs;
    NetworkModel *model;
    // Scratch for the colored finite-difference sparse Jacobian (issue #128),
    // sized once when that callback is selected so a per-step Jacobian
    // allocates nothing. Empty on every other path.
    std::vector<double> fd_y_pert, fd_fy_pert, fd_h_vals;
};

// Does SteadyStateOptions::jacobian ask for the closed form?
//
// One rule, five sites: the CVODE march and the KINSOL polish install their
// Jacobian callback under it (issue #127), and dY_ss/dp, the #78 stability
// certificate and the reported source read it through ss_fill_state_jacobian.
// "fd" pins the difference quotient everywhere, as it does in the rest of the
// library. "jax" also lands on differences here — the JAX Jacobian is a Python
// callback plumbed only into CvodeSimulator, so claiming it in a steady-state
// solve would be a lie about which matrix ran.
static bool ss_want_analytical_jacobian(const std::string &strategy) {
    return strategy == "auto" || strategy == "analytical";
}

// Do the two solver tiers hand their Newton matrix a closed form (issue #127)?
//
// Both tiers ask this one question, so neither can end up differencing while the
// other does not, and SteadyStateResult::solver_jacobian_source is answered from
// the same predicate rather than from a second reading of the options.
static bool ss_install_solver_jacobian(const SteadyStateRhs &rhs, const SteadyStateOptions &opts) {
    return ss_want_analytical_jacobian(opts.jacobian) && rhs.has_analytical_jacobian();
}

// What the march and the polish actually factored: "codegen" / "analytical" when
// the callback is installed, "finite-difference" for CVODE's and KINSOL's own
// difference quotients.
static const char *ss_solver_jacobian_source(const SteadyStateRhs &rhs,
                                             const SteadyStateOptions &opts) {
    return ss_install_solver_jacobian(rhs, opts) ? rhs.jacobian_source() : "finite-difference";
}

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

// Build the dense direct linear solver for an n×n steady-state system, applying
// the GH #84 gate (bngsim/lapack_dense_linsol.hpp). This is the whole story for
// the KINSOL polish and the sensitivity solve, which factor the *reduced*
// matrix and have no sparse route; the march reaches it only when
// ss_use_sparse_linsol says dense (issue #128).
//
// `force_dense` is the caller's force_dense_linear_solver, passed on for parity
// with CvodeSimulator::Impl::choose_linear_solver_kind. The gate is opt-in and
// does not currently consult it (nor n, nor density) — it is threaded through so
// the two solvers cannot answer this question from different inputs if it ever
// does.
static SUNLinearSolver ss_make_dense_linsol(N_Vector v, SUNMatrix A, SUNContext ctx,
                                            NetworkModel &model, int n, bool force_dense = false) {
    const bool use_lapack = should_use_lapack_dense(n, model.jacobian_sparsity().density,
                                                    /*force_dense=*/force_dense);
    return make_dense_linear_solver(v, A, ctx, use_lapack);
}

// Does the CVODE march factor with KLU rather than densely (issue #128)?
//
// The shared rule run() takes (bngsim/sparse_jacobian.hpp) — same threshold,
// same density test, same force flags — and then one requirement of its own:
// something to fill the CSC values with. KLU cannot fall back on CVODE's
// built-in difference quotient, which covers dense and banded matrices only, so
// a sparse route needs either the closed form or a graph coloring, and a
// pattern with no structural nonzero has neither.
//
// run() refuses that case with a legible error instead, and rightly: there the
// only way to ask for a sparse route was force_sparse_linear_solver, so the
// refusal answers a request the caller made. Here the AUTO rule can ask for it
// unprompted — a model with 50+ species and no reactions has density
// 0 < SPARSE_DENSITY_MAX — and such a model solves immediately, f(y) ≡ 0 being
// a steady state. Failing it would be a regression, so the march declines the
// route and factors densely, exactly as it did before #128.
//
// One predicate, so the marcher and the reported name cannot disagree about
// what ran.
static bool ss_use_sparse_linsol(NetworkModel &model, const SteadyStateRhs &rhs,
                                 const SteadyStateOptions &opts, int ns) {
    const auto &sp = model.jacobian_sparsity();
    if (!route_to_sparse_linear_solver(sp, ns, opts.jacobian, opts.force_dense_linear_solver,
                                       opts.force_sparse_linear_solver)) {
        return false;
    }
    // A closed form fills any pattern; otherwise it takes a coloring, which
    // every pattern with a structural nonzero has (a fully dense one degenerates
    // to one column per color, which is correct and merely not a speedup).
    return ss_install_solver_jacobian(rhs, opts) ? sp.nnz > 0
                                                 : model.ensure_jacobian_coloring().has_coloring();
}

// What the march factored with, for SteadyStateResult::linear_solver.
static const char *ss_linear_solver_name(NetworkModel &model, const SteadyStateRhs &rhs,
                                         const SteadyStateOptions &opts, int ns) {
    if (ss_use_sparse_linsol(model, rhs, opts, ns)) {
        return "klu";
    }
    return should_use_lapack_dense(ns, model.jacobian_sparsity().density,
                                   opts.force_dense_linear_solver)
               ? "lapack-dense"
               : "dense";
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

// The march's Jacobian: the closed form the same object already carries (issue
// #127), compiled or interpreted, straight into the dense matrix's column-major
// data array — the layout SteadyStateRhs::fill_dense_jacobian writes and the one
// SUNDenseMatrix stores. Installed by SteadyStateMarcher's constructor; without
// it CVODE differences its own, at one RHS evaluation per species per setup.
//
// The mirror of CvodeSimulator's cvode_analytical_dense_jac / cvode_codegen_
// dense_jac, minus their GH #135 nonnegative-clamp retry: that guard exists
// because the time-course path's RHS is clamped too, so a state where the
// analytical Jacobian goes non-finite (a fractional power of a transiently
// negative concentration) is one the RHS still integrated through. This RHS is
// not clamped — such a state already fails the march on its f(y) alone — so
// there is no asymmetry here for the retry to repair.
static int cvode_ss_dense_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J, void *ud,
                              N_Vector /*tmp1*/, N_Vector /*tmp2*/, N_Vector /*tmp3*/) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    data->rhs->fill_dense_jacobian(static_cast<double>(t), N_VGetArrayPointer(y),
                                   SUNDenseMatrix_Data(J));
    return 0;
}

#ifdef BNGSIM_HAS_KLU
// The same Jacobian, written into the CSC values KLU factors (issue #128), for a
// march the routing rule sent to the sparse linear solver. The structure has to
// be reinstalled first: CVODE may SUNMatZero() before the callback, and
// SUNMatZero_Sparse clears the index arrays along with the values.
//
// Like cvode_ss_dense_jac, and unlike its two time-course mirrors, this carries
// no GH #135 nonnegative-clamp retry — for the reason given there: that guard
// exists because the time-course RHS is clamped and this one is not, so a state
// where the closed form goes non-finite is one the march has already failed on
// f(y) alone.
static int cvode_ss_sparse_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J, void *ud,
                               N_Vector /*tmp1*/, N_Vector /*tmp2*/, N_Vector /*tmp3*/) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    const auto &sp = data->model->jacobian_sparsity();
    install_csc_structure(SUNSparseMatrix_IndexPointers(J), SUNSparseMatrix_IndexValues(J), sp);
    data->rhs->fill_sparse_jacobian(static_cast<double>(t), N_VGetArrayPointer(y),
                                    SUNSparseMatrix_Data(J));
    return 0;
}

// A sparse-routed march with no closed form to install: the Curtis-Powell-Reid
// colored difference quotient, at one RHS evaluation per COLOR rather than per
// species. Not an optimization but a requirement — CVODE's built-in difference
// quotient covers dense and banded matrices only, so a sparse matrix with no
// Jacobian callback fails linear-solver initialization outright.
//
// It differences the march's OWN right-hand side (SteadyStateRhs::eval, compiled
// when a codegen artifact is attached), not NetworkModel::compute_derivs — the
// matrix therefore belongs to the system being integrated even when the two
// backends differ. cvode_colored_jac on the time-course path predates the
// codegen RHS and still differences the interpreted one.
static int cvode_ss_colored_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *ud,
                                N_Vector /*tmp1*/, N_Vector /*tmp2*/, N_Vector /*tmp3*/) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    const int ns = data->model->n_species();
    // Bare accessor: the marcher materialized the coloring before installing
    // this callback, and the materialized value never changes afterward.
    const auto &sp = data->model->jacobian_sparsity();

    install_csc_structure(SUNSparseMatrix_IndexPointers(J), SUNSparseMatrix_IndexValues(J), sp);

    // Sized once, when the callback was selected, so this hot path allocates
    // nothing per Jacobian evaluation.
    SteadyStateRhs *rhs = data->rhs;
    colored_fd_jacobian(
        sp, ns, static_cast<double>(t), N_VGetArrayPointer(y), N_VGetArrayPointer(fy),
        SUNSparseMatrix_Data(J), data->fd_y_pert.data(), data->fd_fy_pert.data(),
        data->fd_h_vals.data(),
        [rhs](double tt, const double *yy, double *ydot) { rhs->eval(tt, yy, ydot); });
    return 0;
}
#endif // BNGSIM_HAS_KLU

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

        // Scalar atol, or the per-species vector when the caller supplied one
        // (issue #196). The convergence test below is unaffected either way —
        // it is ||f(y)||_2/n_species against opts.tol, a norm with no
        // per-species reading.
        apply_cvode_tolerances(cvode_mem_, ctx_, opts.rtol, opts.atol, opts.atol_vec, ns_);
        CVodeSetUserData(cvode_mem_, &ud_);
        CVodeSetMaxNumSteps(cvode_mem_, opts.max_steps);

        setup_linear_solver(model, rhs, opts);
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
                // Integration failed -- report unconverged. Remembered, not just
                // broken out of: under jacobian="auto" a hard integrator failure
                // is what triggers the retry on difference quotients (issue #127,
                // mirroring GH #176 on the time-course path).
                integrator_failed_ = true;
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

    // Did CVODE give up on a march (any negative flag), as opposed to running
    // out of this march's time budget? Sticky across the ladder's rungs.
    bool integrator_failed() const { return integrator_failed_; }

  private:
    // The Newton matrix and the solver that factors it.
    //
    // Two decisions, taken the way CvodeSimulator::run takes them:
    //
    //   * dense vs sparse (issue #128) — ss_use_sparse_linsol, the shared
    //     size/density/force-flag rule. Until then this was unconditionally a
    //     SUNDenseMatrix, so the march factored densely on the very models
    //     run() routes to KLU: BaruaBCR_2012 (1122 species) solves in 1.78 s
    //     against the dense 5.59 s, fceri_fyn (1281) in 6.92 s against 13.25 s.
    //   * which Jacobian fills it (issue #127) — ss_install_solver_jacobian,
    //     the same gate tier 2 applies to KINSetJacFn, so a solve does not
    //     difference in one tier and not the other. Until then the march ran
    //     CVODE's internal difference quotient whatever opts.jacobian said, on
    //     models whose closed form is assembled, compiled and loaded in this
    //     very object.
    //
    // The two are independent — the issue #128 attribution ran all four corners
    // and every one returned the same state — except on the sparse route, where
    // "no callback" is not an option: CVODE's built-in difference quotient
    // covers dense and banded matrices only.
    void setup_linear_solver(NetworkModel &model, SteadyStateRhs &rhs,
                             const SteadyStateOptions &opts) {
#ifdef BNGSIM_HAS_KLU
        // ss_use_sparse_linsol has already materialized the coloring when the
        // colored difference quotient is the fill, so the bare accessor below
        // (and in cvode_ss_colored_jac) reads a populated pattern.
        if (ss_use_sparse_linsol(model, rhs, opts, ns_)) {
            const bool analytical = ss_install_solver_jacobian(rhs, opts);
            const auto &sp = model.jacobian_sparsity();

            A_ = SUNMatrixGuard(SUNSparseMatrix(ns_, ns_, sp.nnz, CSC_MAT, ctx_));
            if (!A_) {
                throw std::runtime_error("SUNSparseMatrix failed (steady_state)");
            }
            install_csc_structure(SUNSparseMatrix_IndexPointers(A_),
                                  SUNSparseMatrix_IndexValues(A_), sp);

            LS_ = SUNLinSolGuard(SUNLinSol_KLU(y_, A_, ctx_));
            if (!LS_) {
                throw std::runtime_error("SUNLinSol_KLU failed (steady_state)");
            }
            CVodeSetLinearSolver(cvode_mem_, LS_, A_);

            if (!analytical) {
                ud_.fd_y_pert.resize(static_cast<size_t>(ns_));
                ud_.fd_fy_pert.resize(static_cast<size_t>(ns_));
                ud_.fd_h_vals.assign(static_cast<size_t>(ns_), 0.0);
            }
            if (CVodeSetJacFn(cvode_mem_, analytical ? cvode_ss_sparse_jac
                                                     : cvode_ss_colored_jac) != CV_SUCCESS) {
                throw std::runtime_error("CVodeSetJacFn failed (steady_state, sparse)");
            }
            return;
        }
#endif
        A_ = SUNMatrixGuard(SUNDenseMatrix(ns_, ns_, ctx_));
        LS_ = SUNLinSolGuard(
            ss_make_dense_linsol(y_, A_, ctx_, model, ns_, opts.force_dense_linear_solver));
        CVodeSetLinearSolver(cvode_mem_, LS_, A_);

        if (ss_install_solver_jacobian(rhs, opts)) {
            if (CVodeSetJacFn(cvode_mem_, cvode_ss_dense_jac) != CV_SUCCESS) {
                throw std::runtime_error("CVodeSetJacFn failed (steady_state)");
            }
        }
    }

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
    bool integrator_failed_ = false;
};

static SteadyStateResult solve_by_integration(NetworkModel &model, SteadyStateRhs &rhs,
                                              const SteadyStateOptions &opts,
                                              const ResidualSubspace &sub,
                                              bool *integrator_failed = nullptr) {
    SteadyStateMarcher marcher(model, rhs, opts, sub);
    double residual = 0.0;
    const bool converged = marcher.march(opts.tol, &residual);
    if (integrator_failed) {
        *integrator_failed = marcher.integrator_failed();
    }
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
    // Scratch for the reduced Jacobian callback (issue #127): the state it fills
    // at, the full ns×ns fill, and its projection onto the unknowns. Held here so
    // a Jacobian setup allocates nothing.
    std::vector<double> jac_y_full, jac_full, jac_red;
};

// The reduced Jacobian's projection through the conservation-law reconstruction.
// Defined with ss_fill_state_jacobian further down, next to the other consumer
// of the same matrix (dY_ss/dp and the #78 certificate); declared here because
// the KINSOL polish now needs it too (issue #127).
static void ss_reduce_jacobian(const double *J, int ns, const ConservationLaws &cl,
                               const std::vector<int> &idx, std::vector<double> &J_red);

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

// Reduced-space KINSOL Jacobian: ∂f_ind/∂y_ind of the residual above (#127).
//
// Two things have to match kinsol_reduced_rhs exactly or the Newton step is a
// step for a different system:
//
//   * the STATE — the model's live concentrations, with the unknowns overwritten
//     from y_ind and the law-dependent species reconstructed from them. A
//     species that is neither (a mask-excluded one, issue #74) is a constant
//     here, which is why nothing below differentiates it.
//   * the PROJECTION — d/dy_ind of that reconstruction, which is what
//     ss_reduce_jacobian applies to the full fill. KINSOL's difference quotient
//     gets it for free by differencing the reduced residual itself; a closed-form
//     fill is of the FULL system and has to be projected by hand.
static int kinsol_reduced_jac(N_Vector y_ind, N_Vector /*fval*/, SUNMatrix J, void *ud,
                              N_Vector /*tmp1*/, N_Vector /*tmp2*/) {
    auto *data = static_cast<ReducedKinsolData *>(ud);
    NetworkModel &model = *data->model;
    const auto &cl = *data->cl;
    const int ns = model.n_species();
    const int n_ind = static_cast<int>(cl.independent.size());

    const auto &species = model.species();
    data->jac_y_full.resize(static_cast<size_t>(ns));
    for (int i = 0; i < ns; ++i) {
        data->jac_y_full[static_cast<size_t>(i)] = species[i].concentration;
    }
    reconstruct_full(N_VGetArrayPointer(y_ind), data->jac_y_full.data(), ns, cl, species);

    // Every fill_dense_jacobian branch memsets its buffer, so resize (not assign)
    // is enough — the zeroing is not skipped, it is done once instead of twice.
    data->jac_full.resize(static_cast<size_t>(ns) * ns);
    data->rhs->fill_dense_jacobian(0.0, data->jac_y_full.data(), data->jac_full.data());
    ss_reduce_jacobian(data->jac_full.data(), ns, cl, cl.independent, data->jac_red);
    std::memcpy(SUNDenseMatrix_Data(J), data->jac_red.data(),
                static_cast<size_t>(n_ind) * n_ind * sizeof(double));
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

// Full-space KINSOL Jacobian: the plain ns×ns fill (issue #127).
//
// It matches kinsol_rhs's fixed-species handling without doing anything about
// it, because both sides of the closed form already do: a fixed species'
// derivative is zero by definition, so fill_dense_analytical_jacobian zeroes its
// ROW as its last step and the emitted C does the same. That row is structurally
// zero in the difference quotient this replaces too — a full-space system with a
// fixed species is singular either way, and falls back to integration.
static int kinsol_jac(N_Vector y, N_Vector /*fval*/, SUNMatrix J, void *ud, N_Vector /*tmp1*/,
                      N_Vector /*tmp2*/) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    data->rhs->fill_dense_jacobian(0.0, N_VGetArrayPointer(y), SUNDenseMatrix_Data(J));
    return 0;
}

// Which species a restricted steady-state system treats as UNKNOWNS.
//
// Three callers have to agree on this set or they are answering questions about
// different systems: the KINSOL polish (solve_by_newton), the dY_ss/dp linear
// solve (compute_ss_sensitivity), and the stability certificate at the accepted
// root (issue #78).
//
//   * conservation laws present — the reduction's independent species; a
//     dependent one is not an unknown but a linear function of the others.
//   * mask present (issue #74) — narrowed to the species the mask kept.
//   * `fixed` species — dropped from either restricted form: their derivative is
//     zeroed by definition, so keeping one as an unknown adds a structurally
//     zero ROW and makes the system singular for a reason that has nothing to do
//     with the model.
//
// Unmasked WITH laws returns cl.independent verbatim (fixed species included),
// exactly as before #74. A caller with neither laws nor a mask builds no
// restricted system at all and does not come here.
static std::vector<int> ss_unknown_species(const NetworkModel &model, const ConservationLaws &cl,
                                           const ResidualSubspace &sub) {
    if (!cl.empty() && !sub.masked) {
        return cl.independent;
    }
    std::vector<int> unknowns;
    if (cl.empty()) {
        unknowns.reserve(sub.included.size());
        for (int i : sub.included) {
            if (!model.species()[i].fixed)
                unknowns.push_back(i);
        }
    } else {
        unknowns.reserve(cl.independent.size());
        for (int i : cl.independent) {
            if (sub.includes(i) && !model.species()[i].fixed)
                unknowns.push_back(i);
        }
    }
    return unknowns;
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
    // Both restricted forms also drop `fixed` ($-prefixed) species — see
    // ss_unknown_species, which is where that rule now lives, shared with the
    // dY_ss/dp solve and the stability certificate. The unmasked paths do NOT do
    // this (kinsol_rhs zeroes those rows while leaving the unknowns in place, and
    // cl.independent is taken verbatim) and are left exactly as they were — an
    // all-true mask is not masked at all (see make_subspace), so no pre-#74 solve
    // changes path.
    const bool use_reduced = !cl_copy.empty() || sub.masked;
    if (use_reduced) {
        if (cl_copy.empty()) {
            cl_copy.n_species = ns;
        }
        std::vector<int> unknowns = ss_unknown_species(model, cl_copy, sub);
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

    // The polish's Newton matrix (issue #127). Same gate as the march, so a
    // single solve does not difference in one tier and use the closed form in
    // the other; the reduced callback carries the projection onto the unknown
    // set that the difference quotient it replaces got for free.
    if (ss_install_solver_jacobian(rhs, opts)) {
        flag = KINSetJacFn(kin_mem, use_reduced ? kinsol_reduced_jac : kinsol_jac);
        if (flag != KIN_SUCCESS) {
            throw std::runtime_error("KINSetJacFn failed");
        }
    }

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

// ── Is a polished root one the dynamics can actually rest on? (issue #78) ────
//
// Seed stability and dynamical stability are different properties, and only the
// second is what a caller asking for a steady state wants. The ladder's
// agreement test asks whether REFINING THE SEED moves the root; near a
// separatrix the trajectory slows to a crawl, so two successively tighter bursts
// hand KINSOL near-identical seeds a few percent from the saddle, both polish to
// it, and the two agree — the guard is satisfied precisely where it is needed.
// The Gardner toggle at alpha_2 = 53.53 returns the saddle [28.245, 1.830] with
// residual 2.8e-10 and converged=True, a state one part in 1e6 either side of
// which runs away to a DIFFERENT attractor.
//
// What settles it is the Jacobian's spectrum at the root: a steady state the
// system can occupy has every eigenvalue in the closed left half-plane.
// `Undetermined` means the certificate could not answer (see the definition
// below) and the root is then accepted exactly as it was before #78 — a guard
// that cannot decide must not overturn behavior.
enum class RootStability { Undetermined, Stable, Unstable };

static RootStability certify_root_stability(NetworkModel &model, SteadyStateRhs &rhs,
                                            const SteadyStateOptions &opts,
                                            const ResidualSubspace &sub,
                                            const std::vector<double> &y);

// SteadyStateResult::root_stability spelling of a verdict.
static const char *root_stability_name(RootStability s) {
    switch (s) {
    case RootStability::Stable:
        return "stable";
    case RootStability::Unstable:
        return "unstable";
    default:
        return "undetermined";
    }
}

static SteadyStateResult solve_by_newton_two_tier(NetworkModel &model, SteadyStateRhs &rhs,
                                                  const SteadyStateOptions &opts,
                                                  const ResidualSubspace &sub,
                                                  bool *integrator_failed = nullptr) {
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

    // How many burst-seeded roots the stability certificate turned down, carried
    // onto whatever this solve ends up returning (issue #78) — otherwise a
    // rejection is invisible from the outside and the fallback to integration
    // reads as an ordinary slow solve.
    int n_rejected = 0;
    // Set once the ladder's integrator exists; every return below funnels through
    // `finish`, so reading it there reports a failed march from any rung without
    // a flag at each exit (issue #127's retry trigger).
    const SteadyStateMarcher *ladder = nullptr;
    auto finish = [&](SteadyStateResult r) {
        r.n_unstable_roots_rejected = n_rejected;
        if (integrator_failed) {
            *integrator_failed = ladder != nullptr && ladder->integrator_failed();
        }
        return r;
    };

    const double r0 = compute_residual(rhs, ic.data(), ns, sub);

    // IC already at steady state: a single Newton polish (which converges
    // immediately) reports the canonical "newton" without any integration.
    //
    // The certificate runs here but does NOT reject: the caller's own initial
    // condition is the root, so integration would return the very same state and
    // there is nothing else to fall back to. An unstable verdict is reported
    // instead of acted on — "you handed me an equilibrium the system cannot sit
    // on" is the useful answer, and it is one the caller could not previously get.
    if (r0 < opts.tol) {
        SteadyStateResult r;
        try {
            r = ss_newton_from(model, rhs, opts, sub, ic);
        } catch (...) {
            r.converged = false;
        }
        if (accept(r)) {
            r.root_stability = root_stability_name(
                certify_root_stability(model, rhs, opts, sub, r.concentrations));
            restore();
            return finish(std::move(r));
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

    // The root the certificate last turned down (issue #78). While the
    // trajectory is still creeping past a saddle, every rung's polish lands back
    // on it; without this, each landing would pay for another eigen-decomposition
    // to reach the same verdict.
    std::vector<double> rejected_root;
    bool have_rejected = false;

    std::vector<double> seed = ic;
    double bt = std::max(r0 * 0.1, opts.tol);

    // One integrator for the whole ladder: each rung CONTINUES this march (the
    // early-exit Newton probe above may have left the model's live state
    // elsewhere, so re-seed it from the IC the marcher is about to read).
    restore();
    SteadyStateMarcher marcher(model, rhs, opts, sub);
    ladder = &marcher;

    for (int rung = 0; rung < MAX_RUNGS; ++rung) {
        // Tier 1: continue integrating from the previous rung's end state to bt.
        double burst_residual = 0.0;
        const bool burst_converged = marcher.march(bt, &burst_residual);
        SteadyStateResult burst = marcher.make_result(burst_converged, burst_residual);
        seed = burst.concentrations;

        if (burst.residual < opts.tol) {
            // Integration itself reached the parity tolerance — done.
            restore();
            return finish(std::move(burst));
        }
        if (!burst.converged) {
            // Could not reach even this (looser) burst tolerance within max_time
            // (a slow/oscillatory system, or a residual floor above tol like
            // Barua 2013). Integration is the best available answer.
            restore();
            return finish(std::move(burst));
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
                if (have_rejected &&
                    ss_states_agree(nr.concentrations, rejected_root, AGREE_RTOL)) {
                    // The root the certificate already turned down. Keep
                    // integrating; do not re-certify and do not let it re-pair.
                } else if (have_prev &&
                           ss_states_agree(nr.concentrations, prev_newton, AGREE_RTOL)) {
                    // Seed-stable. Is it a state the dynamics can rest on?
                    const RootStability st =
                        certify_root_stability(model, rhs, opts, sub, nr.concentrations);
                    if (st == RootStability::Unstable) {
                        // A saddle the trajectory is merely passing near. Discard
                        // it and keep integrating: the burst leaves the saddle's
                        // neighborhood on its own, and a later rung polishes the
                        // attractor the system actually reaches (issue #78).
                        ++n_rejected;
                        rejected_root = nr.concentrations;
                        have_rejected = true;
                        prev_newton.clear();
                        have_prev = false;
                    } else {
                        nr.root_stability = root_stability_name(st);
                        restore();
                        return finish(std::move(nr)); // seed-stable and dynamically stable
                    }
                } else {
                    prev_newton = nr.concentrations;
                    have_prev = true;
                }
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
    return finish(std::move(fin));
}

// ---------------------------------------------------------------------------
// Steady-state sensitivity: dY_ss/dp = -J^{-1} * df/dp
// ---------------------------------------------------------------------------

// √(machine eps): the standard one-sided difference-quotient fraction, where the
// O(h) truncation error and the O(eps/h) cancellation error meet.
static constexpr double kFdEps = 1.4901161193847656e-8;

// One-sided step for probing a PARAMETER of value p (issue #76).
//
// Relative to the parameter itself, because the step has to stay inside the
// region where the rate law is locally linear in p and |p| is the only scale
// the model offers for that. What this replaced, `eps * max(|p|, 1)`, floored
// the step at an absolute sqrt(eps) — a small probe only for a parameter of
// order 1. `ode/before_bunching` carries KD = 1e-9, so the probe was 1500% of
// it, the difference quotient was a secant across a decade and a half of the
// rate law's curvature, and dY_ss/dKD came back 15.9x low. A parameter of
// exactly zero has no scale of its own; 1.0 is what the old floor used and
// there is nothing better to be had.
//
// The price is the other end of the tradeoff: where the parameter's own term is
// a tiny fraction of the RHS component it sits in, a step this small loses the
// response to cancellation where the old wide step did not. That is what
// `param_fd_widen()` below repairs, per component (issue #123).
static inline double param_fd_step(double p) { return kFdEps * (p != 0.0 ? std::abs(p) : 1.0); }

// The absolute step `param_fd_step` replaced. Still the right probe for a
// component whose response to the relative one is roundoff — see
// `param_fd_widen()`. Equal to the relative step for |p| >= 1, which is how the
// callers know there is no second probe to take.
static inline double param_fd_wide_step(double p) { return kFdEps * std::max(std::abs(p), 1.0); }

// Unit roundoff (DBL_EPSILON), the relative spacing of doubles.
static constexpr double kUround = 2.220446049250313e-16;

// How many times its own roundoff a probe's response has to be before the
// difference quotient counts as signal (issue #123).
//
// The quotient's relative error is about noise/response, so this is "prefer the
// wide probe once the narrow one is worse than 1%" — matched to the wide
// probe's own typical accuracy on the models where the two disagree, and flat
// in the corpus measurement across two decades either side (see the PR table:
// columns wrong by >1e-3 read 165 / 117 / 91 at C = 16 / 100 / 1000 against 231
// for the relative step alone, while the #76 fixes it preserves fall away above
// this — 184 of 191 at 100, 151 at 1e5).
static constexpr double kFdNoiseFactor = 100.0;

// The absolute roundoff floor of each component of g, given the terms g was
// assembled from (`term_scale`) and g itself.
//
// |g_i| alone is the wrong scale at a steady state: there f_i is a cancellation
// of large rate terms, so its roundoff is set by the TERMS and not by the
// near-zero sum, and a floor built from |f_i| would call every response signal.
static inline double roundoff_floor(double g_i, double term_scale_i) {
    return kUround * std::max(std::abs(g_i), std::abs(term_scale_i));
}

// Term scale of each f_i from the Jacobian row: Σ_j |J_ij|·|y_j|. A rate term of
// degree d in the species contributes d times the term itself, so the row sum is
// the size of what f_i was assembled from — which is what its roundoff scales
// with. J is column-major (J[j*ns + i] = ∂f_i/∂x_j) and already assembled, so
// this is O(n²) arithmetic over memory the caller is holding anyway.
static void rhs_term_scale(const double *J, const double *y, int ns, std::vector<double> &out) {
    out.assign(static_cast<size_t>(ns), 0.0);
    for (int j = 0; j < ns; ++j) {
        const double yj = std::abs(y[j]);
        if (yj == 0.0) {
            continue;
        }
        const double *col = J + static_cast<size_t>(j) * ns;
        for (int i = 0; i < ns; ++i) {
            out[static_cast<size_t>(i)] += std::abs(col[i]) * yj;
        }
    }
}

// Should component i take its ∂g/∂p from the wide probe instead of the narrow
// one? True exactly when the narrow probe's response does not clear that
// component's roundoff floor by `kFdNoiseFactor` — i.e. when the quotient it
// would give is roundoff rather than a derivative.
static inline bool param_fd_widen(double response, double g_i, double term_scale_i) {
    return std::abs(response) <= kFdNoiseFactor * roundoff_floor(g_i, term_scale_i);
}

// One-sided step for probing SPECIES j of a state whose largest concentration
// is `y_scale`.
//
// Relative to the species, floored at the scale of the state it belongs to
// rather than at 1.0 — unlike parameters, which have no common unit, every
// species is a concentration in the same one, so the state HAS a typical
// magnitude and a species at (or near) zero can be probed against it. The
// absolute 1.0 was wrong in both directions: a nanomolar model was probed at
// 1 molar, and a model in molecule counts (1e6) was probed at 1e-14 of itself,
// which is cancellation noise rather than a derivative. Scored against the
// analytical Jacobian over 1,066 corpus model-states, this rule beats both the
// old one (511 better vs 54 worse, at 10x) and the floor-free relative step
// (291 vs 13) — the floor is what a species far below the state's scale needs
// to stay out of the cancellation noise.
static inline double state_fd_step(double y, double y_scale) {
    return kFdEps * std::max(std::abs(y), y_scale);
}

// The perturbed value to write (`*x_plus`) for a probe of `x` by `h`, and the
// step the difference quotient must divide by — the REALIZED `(x + h) - x`,
// which differs from the requested h by a rounding. For a subnormal x no
// relative step survives the addition at all; dividing by that zero would fill
// the column with infinities, so fall back to the absolute step the old rule
// used (exact for a rate law linear in x, and nothing better exists at 1e-310).
static inline double fd_probe(double x, double h, double *x_plus) {
    double xp = x + h;
    if (xp == x) {
        xp = x + kFdEps;
    }
    *x_plus = xp;
    return xp - x;
}

// The state's own magnitude, which `state_fd_step` floors its probe at.
//
// Over the species that HAVE a steady value only. A species the caller masked
// out (issue #74) is a write-only accumulator holding whatever integration left
// it at — a quantity that grows without bound, 7.5e8 on Barua 2013 while the
// other 405 species are settled at 1e-10 — and letting it set the scale would
// drag every other species' probe up with it. `excluded` is ascending, as both
// callers' sources guarantee. An all-zero state offers no scale at all, so it
// keeps the historical 1.0.
static double state_probe_scale(const double *y, int ns, const std::vector<int> &excluded) {
    double scale = 0.0;
    size_t e = 0;
    for (int i = 0; i < ns; ++i) {
        if (e < excluded.size() && excluded[e] == i) {
            ++e;
            continue;
        }
        scale = std::max(scale, std::abs(y[i]));
    }
    return scale > 0.0 ? scale : 1.0;
}

// ---------------------------------------------------------------------------
// The Jacobian at a state, and its restriction to the unknown subspace
// ---------------------------------------------------------------------------

// Dense column-major J (ns×ns) at `y`: closed form when the model has one, else
// one-sided finite differences (one RHS evaluation per column). Returns the
// SteadyStateResult::sens_jacobian_source spelling of which ran.
//
// Shared by dY_ss/dp and the stability certificate (issue #78) so the two read
// the SAME matrix. Left as its own function rather than inlined twice: an FD
// step rule that exists at two sites is a rule that will be improved at one of
// them (the ∂f/∂p probes needed three passes — #63, #76, #123 — to get right).
static const char *ss_fill_state_jacobian(SteadyStateRhs &rhs, const double *y, int ns,
                                          const ResidualSubspace &sub, bool want_analytical,
                                          double *J) {
    if (want_analytical && rhs.has_analytical_jacobian()) {
        rhs.fill_dense_jacobian(0.0, y, J);
        return rhs.jacobian_source();
    }
    // J[:,j] = (f(y + h·e_j) − f(y)) / h.
    std::vector<double> f0(ns), f1(ns), y_pert(ns);
    const double y_scale = state_probe_scale(y, ns, sub.excluded);
    rhs.eval(0.0, y, f0.data());
    for (int j = 0; j < ns; ++j) {
        std::memcpy(y_pert.data(), y, static_cast<size_t>(ns) * sizeof(double));
        const double h = fd_probe(y[j], state_fd_step(y[j], y_scale), &y_pert[j]);
        rhs.eval(0.0, y_pert.data(), f1.data());
        for (int i = 0; i < ns; ++i) {
            J[static_cast<size_t>(j) * ns + i] = (f1[i] - f0[i]) / h; // column-major
        }
    }
    return "finite-difference";
}

// Restrict the full column-major J to the unknown species `idx`, projecting the
// dependent species out through the conservation-law reconstruction:
//
//   J_red[i][j] = ∂f_{idx_i}/∂y_{idx_j}
//               = J[idx_i][idx_j] + Σ_k J[idx_i][dep_k] · D[k][j]
//
// where D[k][j] = ∂y_{dep_k}/∂y_{idx_j} is exactly what differentiating
// reconstruct_full() gives. That is the chain rule an older comment here
// described and then abandoned ("for simplicity and robustness, use FD on the
// reduced residual directly"); with a closed-form J to project it is both exact
// and cheaper — the FD sweep cost n_ind more RHS evaluations on top of the ns
// that had already built a full J the reduced path then never looked at.
//
// D follows reconstruct_full()'s ordering: law k solves for its dependent from
// every OTHER species, so it sees the dependents of laws k' < k already updated
// (derivative D[k'][j]) and those of laws k' > k still at their unperturbed
// values (derivative 0). A degenerate law (|L[k,dep]| below the reconstruction
// floor) is skipped there, so its row stays 0. With no laws this is a plain
// submatrix, which is what a mask-only restriction wants.
static void ss_reduce_jacobian(const double *J, int ns, const ConservationLaws &cl,
                               const std::vector<int> &idx, std::vector<double> &J_red) {
    const int n_ind = static_cast<int>(idx.size());
    std::vector<double> D(static_cast<size_t>(cl.n_laws) * n_ind, 0.0);
    for (int k = 0; k < cl.n_laws; ++k) {
        const int dep = cl.dependent[k];
        const double coeff_dep = cl.coefficients[k][dep];
        if (std::abs(coeff_dep) < 1e-15)
            continue; // degenerate — reconstruct_full skips it too
        double *Dk = D.data() + static_cast<size_t>(k) * n_ind;
        for (int j = 0; j < n_ind; ++j) {
            // i = idx_j contributes L[k][idx_j]·1; every other independent
            // contributes 0.
            double acc = cl.coefficients[k][idx[j]];
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

    J_red.assign(static_cast<size_t>(n_ind) * n_ind, 0.0);
    for (int j = 0; j < n_ind; ++j) {
        double *col = J_red.data() + static_cast<size_t>(j) * n_ind; // column-major
        const double *Jcol_ind = J + static_cast<size_t>(idx[j]) * ns;
        for (int i = 0; i < n_ind; ++i) {
            col[i] = Jcol_ind[idx[i]];
        }
        for (int k = 0; k < cl.n_laws; ++k) {
            const double d = D[static_cast<size_t>(k) * n_ind + j];
            if (d == 0.0)
                continue;
            const double *Jcol_dep = J + static_cast<size_t>(cl.dependent[k]) * ns;
            for (int i = 0; i < n_ind; ++i) {
                col[i] += Jcol_dep[idx[i]] * d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Linear-stability certificate at a Newton root (issue #78)
// ---------------------------------------------------------------------------

// How far right of the imaginary axis an eigenvalue has to be, RELATIVE to the
// spectral radius, before the root is called unstable.
//
// The two populations are nowhere near each other, so this is picked from a gap
// rather than tuned. Over the 585-model ode_fullnet corpus every root the
// two-tier solver accepts has max Re(λ)/max|λ| ≤ 3.4e-17 — the zero eigenvalues
// of a conserved system, at roundoff — while the Gardner saddle this issue
// reports sits at +0.169. The floor under the threshold is this file's own
// eigensolver: cross-checked against LAPACK on ~1,000 corpus Jacobians its
// max Re(λ) agrees to 1.1e-8 of the spectral radius in the worst case (a
// defective eigenvalue, where eps^(1/m) accuracy is all any method has), so
// 1e-6 clears the measured error by a hundredfold and the observed instability
// by five decades. Erring low would cost only speed — a rejected root falls back
// to integration, which is the correct answer either way.
static constexpr double kStabilityRelTol = 1e-6;

// Above this many unknowns the certificate declines rather than pay for the
// spectrum. The eigen-decomposition is O(n³) and unblocked: 0.05 s at n=256 and
// 0.68 s at n=512 on a dense matrix, less on the structurally sparse Jacobians
// this corpus has (0.05 s at 356 unknowns, 0.26 s at 624, 2.1 s at 1281 —
// against KINSOL solves of 0.15 s, 0.55 s and 15.2 s on the same models). 512
// bounds the worst case under a second and still covers all but 8 of the
// 585-model corpus. A declined root is reported as "undetermined" and accepted,
// exactly as before #78 — the limit is visible on the result rather than silent.
static constexpr int kStabilitySpectrumMaxN = 512;

static RootStability certify_root_stability(NetworkModel &model, SteadyStateRhs &rhs,
                                            const SteadyStateOptions &opts,
                                            const ResidualSubspace &sub,
                                            const std::vector<double> &y) {
    const int ns = model.n_species();
    if (ns <= 0 || static_cast<int>(y.size()) != ns)
        return RootStability::Undetermined;

    // The unknown subspace the polish solved on — the same set, for the same
    // reasons (see ss_unknown_species). It has to be this matrix and not the full
    // one when a mask is in play: a species the caller excluded was held fixed
    // during the solve, so its equation is not part of the dynamics that decides
    // whether THIS root is an attractor. With conservation laws the two agree
    // anyway (range(J) ⊆ ker(L) makes the full spectrum the reduced one plus a
    // zero per law), but the reduced form says so exactly instead of relying on
    // the threshold to ignore those zeros.
    const auto &cl = model.conservation_laws();
    const bool use_reduced = !cl.empty() || sub.masked;
    std::vector<int> idx;
    if (use_reduced) {
        idx = ss_unknown_species(model, cl, sub);
        if (idx.empty())
            return RootStability::Undetermined; // every species pinned; no dynamics left
    }
    const int n = use_reduced ? static_cast<int>(idx.size()) : ns;
    if (n > kStabilitySpectrumMaxN)
        return RootStability::Undetermined;

    // The analytical fill takes its observables from `y`, but the FD fallback
    // runs the RHS, so put the model on the root first and hand it back after.
    // The ns×ns buffer is the same one dY_ss/dp assembles for the same matrix.
    std::vector<double> saved(ns);
    model.get_state_into(saved.data());
    model.set_state_from(y.data());
    std::vector<double> J(static_cast<size_t>(ns) * ns, 0.0);
    ss_fill_state_jacobian(rhs, y.data(), ns, sub, ss_want_analytical_jacobian(opts.jacobian),
                           J.data());
    model.set_state_from(saved.data());

    std::vector<double> M;
    if (use_reduced) {
        ss_reduce_jacobian(J.data(), ns, cl, idx, M);
    } else {
        M = std::move(J);
    }

    std::vector<double> wr(static_cast<size_t>(n)), wi(static_cast<size_t>(n));
    if (!dense_eigenvalues(M.data(), n, wr.data(), wi.data()))
        return RootStability::Undetermined;

    double max_re = -std::numeric_limits<double>::infinity();
    double radius = 0.0;
    for (int i = 0; i < n; ++i) {
        max_re = std::max(max_re, wr[i]);
        radius = std::max(radius, std::hypot(wr[i], wi[i]));
    }
    if (!(radius > 0.0)) {
        // Every eigenvalue is zero: the linearization says nothing at all about
        // this root (a fully degenerate system, or one the mask emptied).
        return RootStability::Undetermined;
    }
    return (max_re > kStabilityRelTol * radius) ? RootStability::Unstable : RootStability::Stable;
}

// dY_ss/dp = -J⁻¹·(∂f/∂p) by the implicit function theorem.
//
// Both factors prefer closed form and fall back to differencing (issue #63):
//
//   J      — the compiled `bngsim_codegen_jac`, else the interpreted
//            fill_dense_analytical_jacobian when analytical_jacobian_complete,
//            else one-sided finite differences (n_species RHS evaluations).
//            This is the same "analytical when complete, FD otherwise" rule
//            jacobian="auto" applies everywhere else, and the same matrix the
//            #78 stability certificate reads (both go through
//            ss_fill_state_jacobian) — and, since issue #127, the same one the
//            march and the polish factor, under the same gate.
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
    std::vector<double> f0(ns), f1(ns);

    // jacobian="fd" pins the finite-difference assembly, the same escape hatch it
    // is everywhere else in the library (and the A/B lever for checking the
    // closed-form path against the one that predates #63) — see
    // ss_want_analytical_jacobian, which is also what the two solver tiers gate
    // their own Jacobian callback on.
    result.sens_jacobian_source = ss_fill_state_jacobian(
        rhs, y_ss, ns, sub, ss_want_analytical_jacobian(opts_jacobian), J.data());

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
        //
        // Two probes, and each component takes the one that answered (issue
        // #123). The relative step is the derivative-preserving one, but where
        // p's own term is a small fraction of the f_i it sits in, it moves f_i by
        // less than f_i's own roundoff and the quotient is noise; those
        // components take the wide (pre-#76 absolute) probe instead, which
        // carries a secant error but not a fabricated one. Every entry is
        // therefore one of the two quotients this file already knew how to
        // compute — the rule chooses between them, it does not invent a third.
        result.sens_dfdp_source = "finite-difference";
        rhs.sync_params();
        rhs.eval(0.0, y_ss, f0.data());

        std::vector<double> term_scale;
        rhs_term_scale(J.data(), y_ss, ns, term_scale);
        std::vector<double> f_wide(ns);

        for (int p = 0; p < np; ++p) {
            int pi = pidx[p];
            double pval = params[pi].value;
            double p_plus = 0.0;
            const double h = fd_probe(pval, param_fd_step(pval), &p_plus);

            // Perturb parameter, holding it against re-derivation so a probe of a
            // derived parameter is not immediately undone.
            const_cast<std::vector<Parameter> &>(params)[pi].value = p_plus;
            rhs.sync_params(pi);
            rhs.eval(0.0, y_ss, f1.data());

            // Restore parameter
            const_cast<std::vector<Parameter> &>(params)[pi].value = pval;
            rhs.sync_params(pi);

            double *col = dfdp.data() + static_cast<size_t>(p) * ns;
            for (int i = 0; i < ns; ++i) {
                col[i] = (f1[i] - f0[i]) / h;
            }

            // The wide probe, for the components the narrow one could not
            // resolve. Skipped entirely when the two steps coincide (|p| >= 1),
            // which is where it would cost an RHS evaluation for nothing.
            const double h_wide_req = param_fd_wide_step(pval);
            if (h_wide_req <= param_fd_step(pval)) {
                continue;
            }
            double p_wide = 0.0;
            const double h_wide = fd_probe(pval, h_wide_req, &p_wide);
            const_cast<std::vector<Parameter> &>(params)[pi].value = p_wide;
            rhs.sync_params(pi);
            rhs.eval(0.0, y_ss, f_wide.data());
            const_cast<std::vector<Parameter> &>(params)[pi].value = pval;
            rhs.sync_params(pi);

            for (int i = 0; i < ns; ++i) {
                if (param_fd_widen(f1[i] - f0[i], f0[i], term_scale[i])) {
                    col[i] = (f_wide[i] - f0[i]) / h_wide;
                }
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
        // `solve_idx` is the unknown set (ss_unknown_species): the
        // conservation-law independents, narrowed to the species the mask kept
        // (issue #74), minus the `fixed` ones. A mask with no conservation laws
        // lands here too, with every law loop below a no-op — one code path for
        // "some species are not unknowns", mirroring solve_by_newton.
        // A fixed species' dY_ss/dp is genuinely 0 — a boundary condition does not
        // move with a rate constant — so its row stays 0 rather than NaN: it is
        // the mask, not fixedness, that means "no answer".
        std::vector<int> solve_idx = ss_unknown_species(model, cl, sub);
        if (solve_idx.empty()) {
            // Nothing left to differentiate (every included species is pinned by
            // a law). Report NaN rather than a zero matrix a fitter would read as
            // "this parameter does not matter".
            std::fill(result.sensitivity.begin(), result.sensitivity.end(),
                      std::numeric_limits<double>::quiet_NaN());
            return;
        }
        const int n_ind = static_cast<int>(solve_idx.size());

        // The reduced Jacobian (n_ind × n_ind), projected through the
        // conservation-law reconstruction — see ss_reduce_jacobian, which the
        // stability certificate shares.
        std::vector<double> J_red;
        ss_reduce_jacobian(J.data(), ns, cl, solve_idx, J_red);

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
                // Issue #170 stage 3: `weight` is not constant in every parameter.
                // When the amount conversion above reads a *writable* compartment
                // size, that size is the coefficient, so its own column carries a
                // direct ∂weight/∂V·x_i on top of the chain rule — the observable's
                // units move with the volume. Same term the CVODES path and the
                // compiled emitter add; this site is the third of the three, and
                // keeping them together is what stopped the #119 divergence above.
                if (sp.amount_valued && sp.volume_param_idx0 >= 0) {
                    const std::string &vname =
                        model.parameters()[static_cast<std::size_t>(sp.volume_param_idx0)].name;
                    for (int p = 0; p < np; ++p) {
                        if (param_names[static_cast<std::size_t>(p)] == vname) {
                            out[p] += entry.factor * y_ss[i];
                        }
                    }
                }
            }
        }
    }

    // ── Functions: compiled chain rule, differenced where it declines ─────────
    if (n_func > 0) {
        result.function_sensitivity.assign(static_cast<size_t>(n_func) * np, 0.0);
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
            //
            // The same sweep accumulates each function's TERM SCALE,
            // Σ_i |∂func_m/∂x_i|·|x_i| — the size of the quantities func_m is
            // assembled from, which the explicit-parameter probe below needs as
            // its roundoff floor (issue #123). It is the function-side analogue
            // of the Jacobian row sum compute_ss_sensitivity uses, and it comes
            // free: the partials are already in hand.
            const double y_scale = state_probe_scale(y_ss, ns, result.excluded_species);
            std::vector<double> func_term_scale(static_cast<size_t>(n_func), 0.0);
            for (int i = 0; i < ns; ++i) {
                std::memcpy(y_pert.data(), y_ss, ns * sizeof(double));
                const double h = fd_probe(y_ss[i], state_fd_step(y_ss[i], y_scale), &y_pert[i]);
                model.update_observables(y_pert.data());
                model.evaluate_functions(0.0);
                f1 = model.function_value_cache();
                const double *dxi = result.sensitivity.data() + static_cast<size_t>(i) * np;
                for (int m = 0; m < n_func; ++m) {
                    if (!need_fd[m]) {
                        continue;
                    }
                    const double dfm_dxi = (f1[m] - f0[m]) / h;
                    func_term_scale[static_cast<size_t>(m)] += std::abs(dfm_dxi * y_ss[i]);
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
            // Two probes here as well (issue #123): a function whose dependence
            // on p is a small fraction of its own magnitude loses the narrow
            // probe's response to roundoff, and takes the wide one instead.
            std::vector<double> f_wide;
            for (int p = 0; p < np; ++p) {
                const int pi = pidx[p];
                if (pi < 0) {
                    continue;
                }
                const double pval = params[pi].value;

                // Perturb, re-deriving every expression parameter but the probed
                // one; restore and re-derive again afterwards, or the derived
                // parameters keep the perturbed values this probe gave them.
                auto probe = [&](double h_request, std::vector<double> &into) {
                    double p_plus = 0.0;
                    const double h = fd_probe(pval, h_request, &p_plus);
                    const_cast<std::vector<Parameter> &>(params)[pi].value = p_plus;
                    rhs.sync_params(pi);
                    model.update_observables(y_ss);
                    model.evaluate_functions(0.0);
                    into = model.function_value_cache();
                    const_cast<std::vector<Parameter> &>(params)[pi].value = pval;
                    rhs.sync_params(pi);
                    return h;
                };

                const double h = probe(param_fd_step(pval), f1);
                const double h_wide_req = param_fd_wide_step(pval);
                const bool have_wide = h_wide_req > param_fd_step(pval);
                const double h_wide = have_wide ? probe(h_wide_req, f_wide) : 0.0;

                for (int m = 0; m < n_func; ++m) {
                    if (!need_fd[m]) {
                        continue;
                    }
                    const double narrow = f1[m] - f0[m];
                    double dfm_dp = narrow / h;
                    if (have_wide &&
                        param_fd_widen(narrow, f0[m], func_term_scale[static_cast<size_t>(m)])) {
                        dfm_dp = (f_wide[m] - f0[m]) / h_wide;
                    }
                    result.function_sensitivity[static_cast<size_t>(m) * np + p] += dfm_dp;
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

    // Same refusal run() makes (issue #128): the pair is a contradiction, not a
    // precedence question, and both flags exist to measure the auto rule — a
    // solve that quietly honors one of them hands a benchmark auto-selected
    // numbers under a "forced" label.
    if (opts.force_dense_linear_solver && opts.force_sparse_linear_solver) {
        throw std::invalid_argument(
            "force_dense_linear_solver and force_sparse_linear_solver are mutually "
            "exclusive; set at most one. Leave both false for the size/density "
            "auto-selection.");
    }

    // Per-species atol for the march (issue #196), checked here so a
    // wrong-length vector names itself rather than surfacing from inside the
    // marcher's constructor.
    validate_atol_vector(opts.atol_vec, ns, "find_steady_state()");

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
    bool integrator_failed = false;

    // One dispatch, so the retry below re-runs exactly what the first attempt ran.
    auto solve = [&](const SteadyStateOptions &o) {
        integrator_failed = false;
        if (method == "integration") {
            // Default: CVODE marched to the BNG2.pl parity criterion.
            return solve_by_integration(model, rhs, o, sub, &integrator_failed);
        }
        // "newton": two-tier integrate-first solver (GH #27). A short CVODE
        // burst carries the state into the physical root's basin, then KINSOL
        // polishes; the polish is accepted only once it is seed-stable (agrees
        // across two successively tighter bursts), otherwise integration
        // continues. This is correct on multi-root and NaN-prone models where
        // the old Newton-first ordering returned spurious / non-finite roots.
        // Opt in for the tighter residual the polish delivers; it costs more
        // wall clock than plain integration (GH #28).
        return solve_by_newton_two_tier(model, rhs, o, sub, &integrator_failed);
    };

    result = solve(opts);
    const SteadyStateOptions *effective = &opts;

    // ── The analytical Jacobian is a bet; this is how it is called off ────────
    //
    // Same policy as GH #176 on the time-course path, for the same failure and
    // (measured) the same model: a rate law that is genuinely discontinuous in a
    // state variable — l-type-calcium-channel-dynamics' `if((-70+V)<-20, …)`,
    // whose threshold the state approaches asymptotically — has an exact
    // derivative that omits the jump, so the closed-form Jacobian cannot warn
    // CVODE's corrector about the step. The predictor overshoots, the local
    // error test fails repeatedly and the step collapses to hmin. A difference
    // quotient straddles the discontinuity and supplies a regularizing slope,
    // which is why FD integrates the same model cleanly. On the 585-model corpus
    // that is one model, and before issue #127 installed a Jacobian at all the
    // march never met it.
    //
    // "auto" means "try the closed form" and therefore has to include calling it
    // off; an explicit jacobian="analytical" is a deliberate choice and is not
    // second-guessed — it surfaces the failure, exactly as GH #176 leaves it.
    // The trigger is a HARD integrator failure, not mere non-convergence: a model
    // that simply needs more max_time returns at its stop time with no CVODE
    // error, and retrying it would only spend the same budget twice.
    SteadyStateOptions fd_opts;
    if (!result.converged && integrator_failed && opts.jacobian == "auto" &&
        ss_install_solver_jacobian(rhs, opts)) {
        fd_opts = opts;
        fd_opts.jacobian = "fd";
        SteadyStateResult retried = solve(fd_opts);
        result = std::move(retried);
        result.solver_jacobian_retried = true;
        effective = &fd_opts;
    }

    result.rhs_backend = rhs.backend();
    // Which Newton matrix the tiers themselves factored (issue #127). Answered
    // from the predicate the install sites use, not from a second reading of
    // opts.jacobian, so it cannot describe a callback that was never installed —
    // and read off the options the ANSWER came from, so a retried solve reports
    // the difference quotient that produced it.
    result.solver_jacobian_source = ss_solver_jacobian_source(rhs, *effective);
    // And which linear solver factored it (issue #128). Read off the same
    // predicate the marcher routes on, and off the effective options for the
    // same reason as above — though a retry cannot change this one, since only
    // jacobian="jax" moves the routing and the retry sets "fd".
    result.linear_solver = ss_linear_solver_name(model, rhs, *effective, ns);

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
