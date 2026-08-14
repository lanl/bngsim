// bngsim/src/cvode_simulator.cpp — CVODE ODE simulator (SUNDIALS v7)
//
// BDF method with Newton iteration for stiff biochemical systems.
// Auto-selects dense or sparse (KLU) direct solver based on model size.
//   - Dense: SUNDenseMatrix + SUNLinSol_Dense — best for N < 50
//   - Sparse: SUNSparseMatrix (CSC) + SUNLinSol_KLU — best for N >= 50
//
// The sparse solver uses the Jacobian sparsity pattern computed at model
// load time (see net_file_loader.cpp build_jacobian_sparsity). CVODE's
// internal difference-quotient Jacobian approximation is used; the
// sparsity pattern tells KLU the nonzero structure so it only stores
// and factorizes the sparse portion.
//
// Uses Jacobian sparsity computed at model load time for sparse linear solves.

#include "bngsim/atol_vector.hpp"
#include "bngsim/codegen_abi.hpp"
#include "bngsim/functional_jac_scatter.hpp"
#include "bngsim/lapack_dense_linsol.hpp"
#include "bngsim/mm_jacobian.hpp"
#include "bngsim/model.hpp"
#include "bngsim/platform_compat.hpp" // POSIX ssize_t shim for Windows (GH #150)
#include "bngsim/result.hpp"
#include "bngsim/simulator.hpp"
#include "bngsim/sparse_jacobian.hpp"
#include "bngsim/types.hpp"
#include "bngsim/wallclock.hpp"

#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_logger.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#ifdef BNGSIM_HAS_KLU
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>
#endif

#include "bngsim/dynamic_library.hpp"
#include "bngsim/mir_jit.hpp"
#include "bngsim/sundials_guards.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bngsim {

// ─── Constants ───────────────────────────────────────────────────────────────

// SPARSE_THRESHOLD / SPARSE_DENSITY_MAX and the dense-vs-sparse decision they
// gate now live in bngsim/sparse_jacobian.hpp, shared with the steady-state
// march (issue #128) — see route_to_sparse_linear_solver there.

// Retry a CV_TOO_MUCH_WORK return, but only while the integrator is still
// getting somewhere (issue #54).
//
// CV_TOO_MUCH_WORK is ordinarily recoverable: CVODE spent its per-output-point
// step budget (max_steps) without reaching t_target, but t, y, the step size and
// the order are all intact, so calling CVode again just continues. That is why
// max_steps is documented as a batch size per output point rather than a ceiling
// on the whole run.
//
// It stops being recoverable when the step size collapses at a discontinuity —
// an `if(t >= sigma)` rate jump drives h to ~1e-15 until t + h == t, and every
// retry then buys zero progress. The loop had no exit for that: `max_steps`
// bounded a batch and never the number of batches, so raising it to 1,000,000
// changed nothing and only the wall-clock `timeout` ever ended the run. In a fit
// that spends the caller's entire per-trial budget before scoring the trial inf.
//
// A batch that advances t by nothing is that stall and no other case: a model
// that legitimately needs many steps advances every batch, however slowly. So
// retry while t moves and raise a diagnosable error the moment it does not,
// naming the t and h CVODE wedged at.
static void retry_while_advancing(void *cvode_mem, sunrealtype t_target, N_Vector y,
                                  sunrealtype *t_ret, int &flag, const char *context,
                                  const std::function<void()> &check_budget) {
    while (flag == CV_TOO_MUCH_WORK) {
        if (check_budget)
            check_budget();

        sunrealtype t_before = 0.0;
        CVodeGetCurrentTime(cvode_mem, &t_before);

        flag = CVode(cvode_mem, t_target, y, t_ret, CV_NORMAL);
        if (flag != CV_TOO_MUCH_WORK)
            return;

        sunrealtype t_after = 0.0;
        CVodeGetCurrentTime(cvode_mem, &t_after);
        if (t_after > t_before)
            continue; // still climbing, however slowly — keep going

        sunrealtype h_now = 0.0;
        CVodeGetCurrentStep(cvode_mem, &h_now);
        long n_steps = 0;
        CVodeGetNumSteps(cvode_mem, &n_steps);
        std::ostringstream msg;
        msg << "CVODE made no progress " << context << ": an entire max_steps batch advanced the "
            << "internal time not at all, at t=" << std::setprecision(17) << t_after
            << " with step size h=" << h_now << " (after " << n_steps << " steps). Retrying "
            << "cannot help — the step size has collapsed, typically at a discontinuity such as "
            << "an if(t >= sigma) rate jump, where t + h == t. A crossing at a time bngsim can "
            << "resolve is stopped on exactly (issue #305), so what is left here is one it "
            << "cannot: a threshold over live state, or one whose time is not constant over the "
            << "run. Loosen rtol/atol, or move the discontinuity onto an event — its trigger "
            << "leaves the right-hand side smooth in t and applies the jump at the root, which "
            << "an output point does NOT do (CV_NORMAL interpolates output points, so a sample "
            << "time at the crossing does not bound the step that spans it).";
        throw std::runtime_error(msg.str());
    }
}

// The choice of dense backend (built-in dense LU vs the GH #84 BLAS dgetrf
// solver) is made inside setup_linsol_and_jac via should_use_lapack_dense()
// (bngsim/lapack_dense_linsol.hpp). That factor is opt-in (BNGSIM_LAPACK_DENSE=1)
// and off by default — end-to-end benchmarking found no reliable win — so the
// default dense path here is unchanged.

// ─── CVODE RHS callback ─────────────────────────────────────────────────────
// SUNDIALS v7 signature: int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)

// ─── CVODE root function callback (event support) ────────────────────────────
// Each event has a trigger expression that evaluates to 0 or 1 (boolean).
// The root function is: g_i(t, y) = trigger_i(t, y) - 0.5
// Zero-crossing at 0.5 detects the false→true transition.
// SUNDIALS signature: int g(sunrealtype t, N_Vector y, sunrealtype* gout, void* user_data)

// The codegen calling contract (symbol signatures + the two user_data structs)
// lives in bngsim/codegen_abi.hpp: the steady-state solver resolves the same
// symbols, and two hand-maintained copies of an ABI struct diverge silently.
// The nested aliases below keep the historical CvodeUserData::Codegen*Fn spelling.

struct CvodeUserData {
    NetworkModel *model;
    // Code-generated RHS function pointer.
    // If non-null, used instead of model->compute_derivs().
    // The codegen function reads parameters from param_values.
    using CodegenRhsFn = bngsim::CodegenRhsFn;
    CodegenRhsFn codegen_fn = nullptr;
    void *codegen_dl_handle = nullptr; // dlopen handle (for cleanup)
    // Struct expected by codegen: first field is double* param_values.
    // Points into the model's live parameter values.
    double *codegen_param_values = nullptr;

    // Pre-built struct passed to the codegen RHS/Jacobian .so on every
    // callback. All three fields are run-invariant, so run() populates this
    // once and the callbacks pass &codegen_so_data instead of reconstructing
    // it on each invocation (GH #77).
    CodegenUserDataForSO codegen_so_data{};

    // Code-generated sensitivity RHS function pointer.
    // CVSensRhs1Fn-compatible: (Ns, t, y, ydot, iS, yS, ySdot, user_data, tmp1, tmp2)
    // The codegen .so expects a CodegenSensUserData struct as user_data,
    // which we build on the stack in the callback wrapper.
    using CodegenSensRhsFn = bngsim::CodegenSensRhsFn;
    CodegenSensRhsFn codegen_sens_fn = nullptr;
    // Companion to the above: Σ|term| per row of the ∂f/∂p column, for the
    // issue #177 sensitivity roundoff floor. Emitted by the same generator, so
    // non-null exactly when codegen_sens_fn is — except against a .so built
    // before emitter v28, where it is null and the floor stays as it was.
    using CodegenSensTermScaleFn = bngsim::CodegenSensTermScaleFn;
    CodegenSensTermScaleFn codegen_sens_term_scale_fn = nullptr;
    // plist for sensitivity codegen (maps iS → param index)
    int *codegen_plist = nullptr;
    int codegen_n_sens = 0;

    // Code-generated dense analytical Jacobian function pointer (GH #76 Task 4).
    // If non-null AND the model's analytical Jacobian is complete, the dense
    // Jacobian dispatch prefers this compiled mirror of fill_dense_analytical_
    // jacobian over the interpreted cvode_analytical_dense_jac. jac is the n×n
    // column-major SUNDenseMatrix data; the emitted C memsets it itself. Reads
    // params from the same CodegenUserDataForSO the RHS uses.
    using CodegenJacFn = bngsim::CodegenJacFn;
    CodegenJacFn codegen_jac_fn = nullptr;

    // Code-generated *sparse* (CSC) analytical Jacobian function pointer (GH
    // #162). The compiled mirror of NetworkModel::fill_sparse_analytical_jacobian:
    // it fills the nnz-length CSC value array (jac_data[data_idx]) rather than an
    // n×n dense buffer, so it is the only viable compiled Jacobian for the large
    // sparse/KLU models that route to the KLU solver. When non-null AND the model
    // is sparse-routed with a complete analytical Jacobian, the KLU Jacobian
    // dispatch prefers it over the interpreted cvode_analytical_jac. The emitted C
    // memsets the value array itself; the CSC structure (col_ptrs/row_indices) is
    // reinstalled by the C++ callback. Reads params from the same
    // CodegenUserDataForSO the RHS uses.
    using CodegenJacSparseFn = bngsim::CodegenJacSparseFn;
    CodegenJacSparseFn codegen_jac_sparse_fn = nullptr;

    // Code-generated observable/expression output evaluator (GH #136). When
    // non-null, the warm recording loop calls this compiled function once per
    // output row to fill obs_out[N_OBS] and func_out[N_FUNC] — replacing the
    // interpreted update_observables() + evaluate_functions() pass that
    // dominated wall time on large models. Reads params from the same
    // CodegenUserDataForSO the RHS uses (so on the warm path, where params are
    // constant, the buffer is always current). Null ⇒ interpreted recording.
    using CodegenOutputsFn = bngsim::CodegenOutputsFn;
    CodegenOutputsFn codegen_outputs_fn = nullptr;

    // Code-generated observable + expression output-sensitivity evaluator (GH
    // #198). When non-null, the cold (CVODES sensitivity) recording loop calls
    // this once per output row to fill func_sens_out[c*N_FUNC + m] = d func_m/dθ_c
    // (and obs_sens_out when non-NULL) from the per-column state sensitivities,
    // via the chain rule over the same expression graph the value codegen uses.
    // state_sens[c] is the c-th yS column (parameter-axis dx/dp columns then
    // IC-axis dx/dY(0) columns); plist[c] is the differentiated parameter index
    // for a parameter column (>= N_PARAMS for an IC column, which skips the
    // parameter term). Null ⇒ no expression output sensitivities (blocks stay
    // empty; an expression: selector raises). Reads params from the same
    // CodegenUserDataForSO the RHS uses.
    using CodegenOutputSensFn = bngsim::CodegenOutputSensFn;
    CodegenOutputSensFn codegen_output_sens_fn = nullptr;

    // Tracking absolute tolerance (issue #213). Inactive (an empty ceiling) for
    // every run that does not ask for it. When active, CVODE calls
    // cvode_tracking_ewt below at every step and that reads this — so it is
    // held BY VALUE here rather than pointed at, because the two setup paths
    // have different lifetimes for their option structs and only the user data
    // is guaranteed to outlive the integration on both.
    AtolTracking atol_tracking;

    // JAX AD Jacobian callback.
    // Stored here so the CVODE Jacobian callback can access it.
    std::function<void(double, const double *, double *, int)> jax_jac_fn;

    // CVODES sensitivity parameter array.
    // When sensitivities are active, CVODES perturbs sens_p[plist[i]] and
    // calls the RHS. The RHS must read parameters from this array, not from
    // the model's internal storage. We sync model params from this array
    // before each compute_derivs() call.
    double *sens_p = nullptr; // pointer to sens_p vector (owned by run())
    int n_params = 0;         // number of parameters in sens_p

    // Switch-time parameters held at their nominal value against CVODES'
    // finite-difference sensitivity probe (issue #48). Non-null only when a
    // fitted switch time was detected; sized n_params, 1 = pinned.
    //
    // A switch time enters f ONLY through an `if()` condition, so ∂f/∂p ≡ 0
    // inside every branch and the parameter's whole gradient is the jump applied
    // at the crossing. But when the model has no analytic sensitivity RHS (any
    // non-Elementary reaction — which every `if()`-gated rate law is), CVODES
    // falls back to its internal FD and perturbs the parameter by
    // √rtol·|p| ≈ 1e-5·|p|. That MOVES the switch, so the perturbed RHS carries
    // the kink a finite distance from t* and the solver hits it while still
    // approaching the stop time: error control then collapses h to ~1e-16 and
    // the run stalls at mxstep, exactly the issue #48 symptom. Pinning makes the
    // probe return f(p) − f(p) = 0 — the true ∂f/∂p in the branch interior — and
    // leaves the kink where the model actually puts it. The state RHS is
    // unaffected: outside a probe sens_p already holds the nominal value.
    //
    // Correct only for a parameter that appears solely in conditions; the Python
    // detector verifies that and refuses rather than pin a parameter whose
    // in-branch ∂f/∂p is genuinely non-zero.
    const char *sens_param_pinned = nullptr;
    const double *sens_param_nominal = nullptr;

    // Throwaway RHS output for the rateOf probe the root function runs before
    // evaluating event triggers (GH #106). compute_derivs() publishes the live
    // dx/dt into the model's current_derivs as a side effect; this buffer just
    // absorbs the returned RHS. Sized to n_species in run() only when the model
    // uses rateOf; left empty (and untouched) otherwise.
    std::vector<double> rateof_root_scratch;

    // Nonnegative-clamped copy of the integrator state for RHS evaluation (GH
    // #135). Lazily sized to n_species on first RHS callback. Concentrations are
    // physically nonnegative, but CVODE's predictor can push a zero-pinned fast
    // species slightly negative, where a fractional-power / sqrt / log rate law
    // evaluates to NaN; the RHS is evaluated at the clamped state so it stays
    // finite there. A no-op (numerically identical) wherever the state is already
    // nonnegative.
    std::vector<double> rhs_nonneg_scratch;

    // Colored finite-difference Jacobian scratch (T4). cvode_colored_jac is
    // called once per Jacobian evaluation and otherwise heap-allocated three
    // length-n_species vectors on every call. These persist them across calls:
    // setup_linsol_and_jac sizes them to n_species exactly when the colored-FD
    // Jacobian callback is selected, so the callback re-uses them with zero
    // per-eval allocation. Empty (and untouched) for analytical / dense / codegen
    // Jacobian paths, which allocate nothing per call already.
    std::vector<double> colored_jac_y_pert;
    std::vector<double> colored_jac_fy_pert;
    std::vector<double> colored_jac_h_vals;

    // Chatter guard (GH #95): per-event "dormant" flags, or null. A non-null
    // pointer addresses an array of length n_events; a non-zero entry suppresses
    // that event's trigger root — root_fn returns a constant so CVODE stops
    // detecting its zero-crossings. Set when an event is found to be chattering
    // (the Zeno pathology a non-negativity clamp hits once its clamped variable
    // decays far below atol and floating-point noise re-trips the trigger every
    // micro-step). Left null — and thus completely inert — for models without
    // events. Owned by run().
    const char *event_dormant = nullptr;

    // Residual expression ids of this run's state-dependent rate-law switches
    // (issue #150), in root order AFTER the event and GH #72 discontinuity
    // roots. Unlike those two — booleans, rooted as `value − 0.5` — these are
    // rooted on the residual itself, so CVODE brackets the crossing on the very
    // function the saltation jump differentiates there. Null (and the root set
    // unchanged) for every run that is not asking for sensitivities and every
    // model with no such condition. Owned by run().
    const std::vector<int> *state_switch_roots = nullptr;
};

// CVODE's error-weight callback for the tracking absolute tolerance (issue
// #213). CVODE passes whatever went to CVodeSetUserData — cvInitialSetup copies
// cv_user_data into cv_e_data for a user-supplied efun — so the cast is to this
// file's user-data type and the shared rule lives in atol_vector.hpp.
static int cvode_tracking_ewt(N_Vector y, N_Vector ewt, void *user_data) {
    return fill_tracking_ewt(static_cast<CvodeUserData *>(user_data)->atol_tracking, y, ewt);
}

// Tfun dispatch thunk: invoked by codegen .so to evaluate a table function at
// the given index value. ctx is opaque on the .so side; we set it to the
// owning NetworkModel pointer. Its type is CodegenTfunEvalFn (declared above).
static double codegen_tfun_eval_thunk(int tf_id, double x, void *ctx) {
    auto *model = static_cast<NetworkModel *>(ctx);
    return model->evaluate_table_function_at(tf_id, x);
}

// Concentrations are physically nonnegative, but CVODE's predictor can push a
// zero-pinned fast species slightly negative, where a fractional-power / sqrt /
// log rate law evaluates to NaN (e.g. `pow(conc, 3.98)` with conc < 0 — NaN for
// ANY negative base, so step reduction alone never escapes it). An exact
// analytical Jacobian lets CVODE step confidently enough to reach that excursion
// where a coarser finite-difference Jacobian would not, so the same model
// integrates under FD but fails under the analytical Jacobian with a spurious
// CV_CONV_FAILURE (GH #135: BIOMD0000000994/995/996). The RHS callbacks retry on
// a nonnegative-CLAMPED copy of the state — the correct boundary value, the way
// RoadRunner keeps such a variable cleanly positive — but ONLY after the
// unclamped RHS comes back non-finite (see cvode_rhs). That keeps a model whose
// RHS is finite at a transiently-negative concentration byte-identical: a
// mass-action law like -k·conc is finite and self-corrects toward 0, and an
// unconditional clamp would instead freeze the species slightly negative and make
// the solve chatter (mxstep). Returns a pointer to the (lazily sized) scratch.
static inline double *clamp_state_nonneg(CvodeUserData *data, const double *y_ptr) {
    const int ns = data->model->n_species();
    if (data->rhs_nonneg_scratch.size() != static_cast<std::size_t>(ns))
        data->rhs_nonneg_scratch.assign(ns, 0.0);
    double *s = data->rhs_nonneg_scratch.data();
    for (int i = 0; i < ns; ++i)
        s[i] = y_ptr[i] > 0.0 ? y_ptr[i] : 0.0;
    return s;
}

// Backstop for any RHS that is still non-finite after the nonnegative clamp
// (e.g. an inf from a 1/conc divide at conc == 0). Returning a RECOVERABLE error
// (a positive code) from the RHS callback tells CVODE to shrink the step and
// retry rather than carry a NaN/Inf into the Newton solve as a spurious
// CV_CONV_FAILURE — the standard SUNDIALS robustness contract.
static inline bool rhs_has_nonfinite(const CvodeUserData *data, const double *ydot_ptr) {
    const int ns = data->model->n_species();
    for (int i = 0; i < ns; ++i)
        if (!std::isfinite(ydot_ptr[i]))
            return true;
    return false;
}

// The Jacobian half of the GH #135 fractional-power guard. d/dx (k·x^p) = k·p·
// x^{p-1} is NaN for ANY negative base when p is non-integer (e.g. p=3.98592…
// from the COPASI TGF-β trimer law), exactly as the RHS k·x^p is — so when the
// BDF predictor pushes a zero-pinned species slightly negative near a dose event,
// the analytical Jacobian goes non-finite even though the RHS clamp (cvode_rhs)
// kept the RHS finite. The Jacobian is the Newton-iteration matrix, so a single
// NaN entry makes every corrector iterate NaN: the step collapses to |h|=hmin and
// CV_CONV_FAILURE (flag=-4) results, and step reduction can never escape it (the
// base stays negative). This was the dose-region half of GH #135 the RHS-only
// clamp missed — BIOMD0000000994/995/996 still failed at the t≈180 ligand-wash/
// injection events. The Jacobian callbacks scan their freshly-filled value buffer
// and re-fill on the nonnegative-clamped state ONLY when non-finite (count = ns²
// for a dense fill, nnz for a CSC sparse fill); the fill routines memset the
// buffer first, so the re-fill cleanly overwrites it. Where the species is clamped
// to 0 the term k·p·0^{p-1} = 0 (p>1) is the correct one-sided boundary value.
// Byte-identical for every finite-Jacobian model — mass-action / polynomial laws
// never go non-finite — so the all-Elementary parity solves an always-on clamp
// would have perturbed are untouched. A still-non-finite Jacobian after the clamp
// (e.g. a p<1 power, whose derivative is inf at 0) is left as-is: no worse than
// before, and no clamp can finitize a genuinely singular boundary Jacobian.
static inline bool jac_has_nonfinite(const double *jac_data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(jac_data[i]))
            return true;
    return false;
}

static int cvode_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
    auto *data = static_cast<CvodeUserData *>(user_data);
    const double *y_ptr = N_VGetArrayPointer(y);
    double *ydot_ptr = N_VGetArrayPointer(ydot);

    // When CVODES sensitivity FD is active, CVODES perturbs
    // sens_p[plist[i]] before calling this RHS. We must sync the model's
    // internal parameter values from sens_p so compute_derivs() sees the
    // perturbed parameter. This is the critical bridge between CVODES's
    // parameter perturbation and our ExprTk-based RHS.
    if (data->sens_p) {
        auto &params = const_cast<std::vector<Parameter> &>(data->model->parameters());
        for (int i = 0; i < data->n_params; ++i) {
            // A pinned switch-time parameter ignores the FD probe (issue #48):
            // ∂f/∂p is 0 in the branch interior, and letting the probe move the
            // switch instead drags the kink into the approach and stalls the
            // solver. See CvodeUserData::sens_param_pinned.
            params[i].value = (data->sens_param_pinned != nullptr && data->sens_param_pinned[i])
                                  ? data->sens_param_nominal[i]
                                  : data->sens_p[i];
        }
        // Re-evaluate constant-expression parameters (e.g., ``_rateLaw{N}``
        // from BNG2.pl that encode ``chi*kon`` style products) so derived
        // rate constants pick up the perturbed primary value. Without this,
        // CVODES's finite-difference sensitivity drops the chain-rule
        // contribution and produces wrong-sign sensitivities for the
        // primary parameter (issue #2).
        auto &evaluator = const_cast<NetworkModel *>(data->model)->evaluator();
        for (auto &p : params) {
            if (p.is_expression && p.evaluator_id >= 0) {
                p.value = evaluator.evaluate(p.evaluator_id);
            }
        }
    }

    data->model->compute_derivs(static_cast<double>(t), y_ptr, ydot_ptr);
    if (rhs_has_nonfinite(data, ydot_ptr)) {
        // Only now (a non-finite RHS) do we retry on the nonnegative-clamped
        // state. Conditional, so a model whose RHS is finite at a transiently-
        // negative concentration — every mass-action / polynomial law, where
        // e.g. -k·conc self-corrects back toward 0 — is byte-identical and keeps
        // that restoring behavior; clamping unconditionally would freeze such a
        // species at a small negative value and make the solve chatter.
        data->model->compute_derivs(static_cast<double>(t), clamp_state_nonneg(data, y_ptr),
                                    ydot_ptr);
        if (rhs_has_nonfinite(data, ydot_ptr))
            return 1; // still non-finite (e.g. inf from 1/conc) -> recoverable
    }
    return 0; // success
}

// ─── Codegen RHS callback ────────────────────────────────────────────────────
// CVODE calls this; we forward to the dlopen'd codegen function.
// The codegen function expects (double t, double* y, double* ydot, void* user_data)
// where user_data->param_values points to the live parameter array.
// We also need to call model->update_observables() AFTER the codegen RHS
// for the result recording to work, but the codegen RHS itself handles
// observables internally. The key insight: the codegen RHS reads params
// from the runtime array, not from the model object.

static int cvode_codegen_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
    auto *data = static_cast<CvodeUserData *>(user_data);
    double *y_ptr = N_VGetArrayPointer(y);
    double *ydot_ptr = N_VGetArrayPointer(ydot);

    // When CVODES sensitivity FD is active, CVODES perturbs sens_p[plist[i]]
    // before calling this RHS. The codegen .so reads parameters from
    // codegen_param_values (a separate buffer set up at run() time), so we
    // must mirror sens_p into it; otherwise FD perturbations are invisible
    // to the codegen RHS and df/dp ≈ 0 — every sensitivity column comes back
    // identically zero. Mirrors the ExprTk sync block in cvode_rhs above.
    //
    // We also re-evaluate constant-expression parameters (e.g., BNG2.pl's
    // ``_rateLaw{N} = chi*kon`` style derived rate constants) so the chain
    // rule lands in dfdp for the underlying primary parameter (issue #2).
    if (data->sens_p) {
        auto &params = const_cast<std::vector<Parameter> &>(data->model->parameters());
        for (int i = 0; i < data->n_params; ++i) {
            // Pinned switch-time parameters, as in cvode_rhs above (issue #48).
            params[i].value = (data->sens_param_pinned != nullptr && data->sens_param_pinned[i])
                                  ? data->sens_param_nominal[i]
                                  : data->sens_p[i];
        }
        auto &evaluator = const_cast<NetworkModel *>(data->model)->evaluator();
        for (auto &p : params) {
            if (p.is_expression && p.evaluator_id >= 0) {
                p.value = evaluator.evaluate(p.evaluator_id);
            }
        }
        for (int i = 0; i < data->n_params; ++i) {
            data->codegen_param_values[i] = params[i].value;
        }
    }

    // The codegen function expects a struct with param_values + tfun callback.
    // It is pre-built once per run() (CvodeUserData::codegen_so_data) since all
    // its fields are run-invariant; we pass it by pointer rather than rebuild it
    // on every callback (GH #77). param_values points at the live buffer, which
    // the sensitivity sync above updates in place — so the pointer stays valid.
    int rc = data->codegen_fn(static_cast<double>(t), y_ptr, ydot_ptr, &data->codegen_so_data);
    if (rc != 0)
        return rc; // codegen RHS already signalled (recoverable or fatal)
    if (rhs_has_nonfinite(data, ydot_ptr)) {
        // Conditional nonnegative-clamp retry, identical rationale to cvode_rhs:
        // byte-identical wherever the RHS is finite, engages only on a genuine
        // non-finite (e.g. pow(conc, 3.98) at a transiently-negative conc).
        rc = data->codegen_fn(static_cast<double>(t), clamp_state_nonneg(data, y_ptr), ydot_ptr,
                              &data->codegen_so_data);
        if (rc != 0)
            return rc;
        if (rhs_has_nonfinite(data, ydot_ptr))
            return 1; // still non-finite -> recoverable
    }
    return 0;
}

// ─── Codegen Sensitivity RHS callback ────────────────────────────────────────
// Bridges CVODES CVSensRhs1Fn (N_Vector args) to the codegen raw double* API.
// The codegen sens RHS expects a CodegenSensUserData struct with:
//   - param_values: contiguous parameter array
//   - plist: maps sensitivity index iS → parameter index
//   - n_sens: number of sensitivity parameters
//
// This callback is set on CVodeSensInit1 when the codegen .so provides
// bngsim_codegen_sens_rhs. Otherwise, CVODES uses its internal FD.

// CodegenSensUserDataForSO — bngsim/codegen_abi.hpp

static int cvode_codegen_sens_rhs(int Ns, sunrealtype t, N_Vector y, N_Vector ydot, int iS,
                                  N_Vector yS, N_Vector ySdot, void *user_data, N_Vector tmp1,
                                  N_Vector tmp2) {

    auto *data = static_cast<CvodeUserData *>(user_data);

    // Build the struct the codegen function expects
    CodegenSensUserDataForSO so_data;
    so_data.param_values = data->codegen_param_values;
    so_data.plist = data->codegen_plist;
    so_data.n_sens = data->codegen_n_sens;

    return data->codegen_sens_fn(Ns, static_cast<double>(t), N_VGetArrayPointer(y),
                                 N_VGetArrayPointer(ydot), iS, N_VGetArrayPointer(yS),
                                 N_VGetArrayPointer(ySdot), &so_data, N_VGetArrayPointer(tmp1),
                                 N_VGetArrayPointer(tmp2));
}

// ─── Analytical Dense Jacobian ───────────────────────────────────────────────
//
// For all-Elementary mass-action models using dense solver (N < SPARSE_THRESHOLD
// or density >= SPARSE_DENSITY_MAX), computes J analytically into a dense matrix.
// O(nnz) cost, zero RHS evaluations, exact (no FD truncation error).
// Replaces CVODE's internal FD Jacobian which costs O(N) RHS evals.

static int cvode_analytical_dense_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J,
                                      void *user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
                                      N_Vector /*tmp3*/) {

    auto *data = static_cast<CvodeUserData *>(user_data);
    NetworkModel *model = data->model;
    const double *y_ptr = N_VGetArrayPointer(y);
    double *jac = SUNDenseMatrix_Data(J);
    // Assemble Elementary closed-form + Functional symbolic contributions into
    // the dense matrix's column-major data array. The same method backs the
    // entrywise FD validation, so integration and the test exercise one path.
    model->fill_dense_analytical_jacobian(static_cast<double>(t), y_ptr, jac);
    // Conditional nonnegative-clamp retry (see jac_has_nonfinite): re-fill on the
    // clamped state ONLY when an entry is non-finite — a fractional power of a
    // transiently-negative concentration. Byte-identical otherwise.
    const std::size_t n = static_cast<std::size_t>(model->n_species());
    if (jac_has_nonfinite(jac, n * n))
        model->fill_dense_analytical_jacobian(static_cast<double>(t),
                                              clamp_state_nonneg(data, y_ptr), jac);
    return 0;
}

// ─── Codegen Dense Analytical Jacobian (GH #76 Task 4) ───────────────────────
//
// Compiled-C mirror of fill_dense_analytical_jacobian. Forwards to the dlopen'd
// bngsim_codegen_jac, which assembles the same Elementary closed-form + MM
// closed-form + Functional symbolic contributions (and zeroes fixed-species
// rows) into the column-major dense matrix, reading parameters from the same
// CodegenUserDataForSO the codegen RHS uses. The emitted C memsets the matrix
// itself. Used in place of cvode_analytical_dense_jac when a model is codegen-
// compiled and the symbol resolved; the interpreted path stays the fallback.

static int cvode_codegen_dense_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J,
                                   void *user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
                                   N_Vector /*tmp3*/) {
    auto *data = static_cast<CvodeUserData *>(user_data);
    double *y_ptr = N_VGetArrayPointer(y);
    double *jac = SUNDenseMatrix_Data(J);

    // When CVODES FD sensitivity is active, codegen_param_values is kept in sync
    // with the (possibly perturbed) parameters by cvode_codegen_rhs, which runs
    // before the Jacobian in each Newton iteration — so the buffer is current.
    // Reuse the per-run() pre-built struct rather than rebuilding it here (GH #77).
    int rc = data->codegen_jac_fn(static_cast<double>(t), y_ptr, jac, &data->codegen_so_data);
    if (rc != 0)
        return rc;
    // Conditional nonnegative-clamp retry (see jac_has_nonfinite): the compiled
    // mirror carries the same fractional-power Jacobian term, and its emitted C
    // memsets the matrix first, so the re-fill on the clamped state overwrites it.
    const std::size_t n = static_cast<std::size_t>(data->model->n_species());
    if (jac_has_nonfinite(jac, n * n))
        rc = data->codegen_jac_fn(static_cast<double>(t), clamp_state_nonneg(data, y_ptr), jac,
                                  &data->codegen_so_data);
    return rc;
}

// ─── JAX AD Dense Jacobian ───────────────────────────────────────────────────
//
// CVODE callback that delegates to a Python JAX function via std::function.
// The JAX function computes the exact Jacobian via forward-mode AD (jacfwd),
// supporting ALL rate law types (Elementary, Functional, MichaelisMenten).
// The callback fills the SUNDenseMatrix in column-major order.
//
// NOTE: This callback must acquire the GIL since it calls back into Python.
// CVODE runs with GIL released, so we must re-acquire before calling JAX.

static int cvode_jax_dense_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J,
                               void *user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
                               N_Vector /*tmp3*/) {

    auto *data = static_cast<CvodeUserData *>(user_data);
    const int ns = data->model->n_species();
    const double *y_ptr = N_VGetArrayPointer(y);

    if (!data->jax_jac_fn) {
        return -1; // no JAX callback set — shouldn't happen
    }

    // Get pointer to the dense matrix data (column-major)
    // SUNDenseMatrix stores columns contiguously: col j starts at j*ns
    sunrealtype *jac_data = SUNDenseMatrix_Data(J);

    // Call the JAX callback: fn(t, y_ptr, jac_col_major_ptr, n_species)
    // The Python callback fills jac_data in column-major order.
    data->jax_jac_fn(static_cast<double>(t), y_ptr, jac_data, ns);

    return 0;
}

#ifdef BNGSIM_HAS_KLU
// ─── Analytical Sparse Jacobian ─────────────────────────────────────────────
//
// For all-Elementary mass-action models, computes J analytically at O(nnz) cost.
// For reaction v_r = k_r * sf * ∏_j x_j^{m_j}:
//   ∂v_r/∂x_j = k_r * sf * m_j * x_j^{m_j-1} * ∏_{i≠j} x_i^{m_i}
//   J[i][j] += S[i][r] * ∂v_r/∂x_j
//
// Zero RHS evaluations needed. Exact (no truncation error). Dominant speedup
// for large models: egfr_net 356 sp → O(30K) ops vs 356 RHS evals for FD.

static int cvode_analytical_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J,
                                void *user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
                                N_Vector /*tmp3*/) {

    auto *data = static_cast<CvodeUserData *>(user_data);
    NetworkModel *model = data->model;
    const double *conc = N_VGetArrayPointer(y);

    const auto &sp = model->jacobian_sparsity();
    sunrealtype *jac_data = SUNSparseMatrix_Data(J);

    // CVODE may call SUNMatZero() before this callback, and SUNMatZero_Sparse
    // clears BOTH values and structural indices. Reinstall CSC structure first.
    install_csc_structure(SUNSparseMatrix_IndexPointers(J), SUNSparseMatrix_IndexValues(J), sp);

    // Accumulate the analytical Jacobian numeric values (Elementary + MM +
    // Functional, fixed-species rows zeroed) into the CSC data array. Single
    // source of truth shared with the dense fill and the GH #151 self-check.
    // (sunrealtype is double in this build, so jac_data aliases double*.)
    model->fill_sparse_analytical_jacobian(static_cast<double>(t), conc, jac_data);
    // Conditional nonnegative-clamp retry (see jac_has_nonfinite); the sparse fill
    // memsets the nnz value array first, so the re-fill cleanly overwrites it.
    if (jac_has_nonfinite(jac_data, static_cast<std::size_t>(sp.nnz)))
        model->fill_sparse_analytical_jacobian(static_cast<double>(t),
                                               clamp_state_nonneg(data, conc), jac_data);

    return 0; // success
}

// ─── Codegen Sparse Analytical Jacobian (GH #162) ────────────────────────────
//
// Compiled-C mirror of fill_sparse_analytical_jacobian. Like cvode_analytical_jac
// it reinstalls the CSC structure (SUNMatZero_Sparse clears the indices), but the
// O(nnz) value fill is delegated to the dlopen'd/JIT'd bngsim_codegen_jac_sparse,
// which assembles the same Elementary + MM + Functional contributions (and zeroes
// fixed-species rows) into the CSC value array — reading parameters from the same
// CodegenUserDataForSO the codegen RHS uses. The emitted C memsets the value array
// itself. Used in place of cvode_analytical_jac when a sparse-routed model is
// codegen-compiled and the symbol resolved; the interpreted path stays the
// fallback.
static int cvode_codegen_sparse_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J,
                                    void *user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
                                    N_Vector /*tmp3*/) {
    auto *data = static_cast<CvodeUserData *>(user_data);
    NetworkModel *model = data->model;

    const auto &sp = model->jacobian_sparsity();
    sunrealtype *jac_data = SUNSparseMatrix_Data(J);

    // Reinstall the CSC structure (see cvode_analytical_jac): SUNMatZero_Sparse
    // may have cleared both values and structural indices before this callback.
    install_csc_structure(SUNSparseMatrix_IndexPointers(J), SUNSparseMatrix_IndexValues(J), sp);

    // The compiled function fills the nnz-length value array (it memsets it
    // first). (sunrealtype is double in this build, so jac_data aliases double*.)
    double *y_ptr = N_VGetArrayPointer(y);
    int rc = data->codegen_jac_sparse_fn(static_cast<double>(t), y_ptr, jac_data,
                                         &data->codegen_so_data);
    if (rc != 0)
        return rc;
    // Conditional nonnegative-clamp retry (see jac_has_nonfinite); the compiled
    // sparse mirror memsets the nnz value array first.
    if (jac_has_nonfinite(jac_data, static_cast<std::size_t>(sp.nnz)))
        rc = data->codegen_jac_sparse_fn(static_cast<double>(t), clamp_state_nonneg(data, y_ptr),
                                         jac_data, &data->codegen_so_data);
    return rc;
}

// ─── Colored Finite-Difference Sparse Jacobian (Curtis-Powell-Reid) ──────────
//
// Computes J = ∂f/∂y using graph-colored finite differences.
// Columns that share a color have non-overlapping sparsity patterns, so they
// can be perturbed simultaneously in a single RHS evaluation. This reduces
// the cost from O(N) RHS evals (one per column) to O(n_colors) ≈ 5–20 RHS
// evals, which is the key speedup for large sparse models.
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
// The difference formula itself is colored_fd_jacobian in
// bngsim/sparse_jacobian.hpp, shared with the steady-state march's sparse route
// (issue #128); what stays here is the CVODE plumbing around it.

static int cvode_colored_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
                             N_Vector /*tmp1*/, N_Vector /*tmp2*/, N_Vector /*tmp3*/) {

    auto *data = static_cast<CvodeUserData *>(user_data);
    NetworkModel *model = data->model;
    const int ns = model->n_species();
    // Bare accessor, not ensure_jacobian_coloring(): setup_linsol_and_jac only
    // installs this callback after materializing the coloring and seeing
    // has_coloring(), and the materialized value never changes afterward — so
    // the coloring reads below are valid without a call_once check per Jacobian.
    const auto &sp = model->jacobian_sparsity();

    double *y_data = N_VGetArrayPointer(y);
    double *fy_data = N_VGetArrayPointer(fy);

    // Access the sparse matrix CSC arrays
    sunrealtype *jac_data = SUNSparseMatrix_Data(J);

    // CVODE may call SUNMatZero() before this callback, and SUNMatZero_Sparse
    // clears BOTH values and structural indices. Reinstall CSC structure first.
    install_csc_structure(SUNSparseMatrix_IndexPointers(J), SUNSparseMatrix_IndexValues(J), sp);

    // Workspace for perturbed state and RHS — persisted across calls in
    // user_data (T4) and sized once when this callback was selected, so this hot
    // path allocates nothing per Jacobian evaluation. Defensive resize covers
    // the (unreached) case of an unsized buffer; it is a no-op once sized.
    if (static_cast<int>(data->colored_jac_y_pert.size()) != ns) {
        data->colored_jac_y_pert.resize(ns);
        data->colored_jac_fy_pert.resize(ns);
        data->colored_jac_h_vals.assign(ns, 0.0);
    }
    double *y_pert = data->colored_jac_y_pert.data();
    double *fy_pert = data->colored_jac_fy_pert.data();
    double *h_vals = data->colored_jac_h_vals.data();

    colored_fd_jacobian(sp, ns, static_cast<double>(t), y_data, fy_data, jac_data, y_pert, fy_pert,
                        h_vals, [model](double tt, const double *yy, double *ydot) {
                            model->compute_derivs(tt, yy, ydot);
                        });

    return 0; // success
}
#endif // BNGSIM_HAS_KLU

// ─── Event / discontinuity root function ─────────────────────────────────────
//
// Root function callback: evaluate each event trigger then each
// discontinuity-trigger condition, subtracting 0.5 so a false→true (or
// true→false) flip is a sign change. Event roots occupy gout[0,n_events)
// and discontinuity roots gout[n_events, n_roots).
static int cvode_event_root_fn(sunrealtype t, N_Vector y, sunrealtype *gout, void *user_data) {
    auto *data = static_cast<CvodeUserData *>(user_data);
    auto *mdl = data->model;
    const double *y_ptr = N_VGetArrayPointer(y);
    const int nsp = mdl->n_species();

    // Sync species concentrations so ExprTk trigger expressions see current y
    auto &sp_vec = const_cast<std::vector<Species> &>(mdl->species());
    for (int i = 0; i < nsp; ++i) {
        sp_vec[i].concentration = y_ptr[i];
    }
    mdl->update_observables(y_ptr);
    mdl->evaluate_functions(static_cast<double>(t));

    // GH #106: refresh the live rateOf buffer (and re-evaluate functions
    // with live dx/dt) so triggers reading rateOf(species) — directly or
    // via a rateOf-bearing function — see the derivative at this (t, y).
    // No-op for non-rateOf models. compute_derivs publishes current_derivs
    // as a side effect; rateof_root_scratch absorbs the returned RHS.
    if (mdl->uses_rateof()) {
        mdl->compute_derivs(static_cast<double>(t), y_ptr, data->rateof_root_scratch.data());
    }

    auto &eval = mdl->evaluator();
    const auto &events = mdl->events();
    const int ne = static_cast<int>(events.size());
    for (int i = 0; i < ne; ++i) {
        if (data->event_dormant != nullptr && data->event_dormant[i]) {
            // Chatter guard (GH #95): a dormant event's root is held at a
            // constant so it never changes sign — CVODE integrates over
            // the noise floor instead of halting at every sub-atol
            // crossing of the trigger.
            gout[i] = 1.0;
            continue;
        }
        double trigger_val = eval.evaluate(events[i].trigger_expr_idx);
        gout[i] = trigger_val - 0.5;
    }
    const auto &disc = mdl->discontinuity_triggers();
    for (int j = 0; j < static_cast<int>(disc.size()); ++j) {
        gout[ne + j] = eval.evaluate(disc[j]) - 0.5;
    }
    // State-dependent rate-law switches (issue #150), rooted on the residual
    // itself rather than on a boolean: `Virus < 1` becomes `Virus − 1`, whose
    // sign change IS the crossing and whose gradient is what dt*/dθ needs.
    if (data->state_switch_roots != nullptr) {
        const auto &ss = *data->state_switch_roots;
        const int base = ne + static_cast<int>(disc.size());
        for (int j = 0; j < static_cast<int>(ss.size()); ++j) {
            gout[base + j] = eval.evaluate(ss[j]);
        }
    }
    return 0;
}

// ─── Forward-sensitivity run state ───────────────────────────────────────────
//
// Everything Impl::setup_forward_sensitivities() resolves for one run(): the
// column counts, the CVODES sensitivity vectors, and the contiguous arrays
// CVODES and the codegen sensitivity RHS read through raw pointers stored in
// CvodeUserData. Held by value in run() so those pointers stay valid for the
// whole integration (the sizes are fixed once setup returns).
struct SensitivityState {
    int n_p = 0;                         // requested parameter columns
    int n_ic = 0;                        // requested initial-condition columns
    int n_total = 0;                     // n_p + n_ic
    NVectorArrayGuard yS;                // CVODES sensitivity vectors (param cols first)
    std::vector<int> param_indices;      // 0-based parameter indices
    std::vector<int> ic_species_indices; // 0-based species indices for IC sens
    std::vector<double> pbar;            // parameter scaling factors for CVODES
    std::vector<double> p;               // contiguous parameter values (CVODES reads this)
    std::vector<int> plist;              // which indices in p to perturb
    std::vector<char> pin_mask;          // switch-time params held nominal (issue #48)
    std::vector<double> pin_nominal;     // their nominal values
    // Hoisted so the event-fire sensitivity jump (GH #212) can re-init the
    // sensitivity vectors with the same method CVodeSensInit1 was given.
    int method = CV_STAGGERED;

    // ── Sensitivity error floor (issue #177) ─────────────────────────────
    // The per-(state × column) absolute tolerance handed to
    // CVodeSensSVtolerances, kept alive past setup so it can be refreshed
    // mid-run, plus the GH #214 static floor it is a max() against.
    NVectorArrayGuard abstolS;
    std::vector<double> atolS_base; // n_sens*ns, column-major by column
    // Per (row, column): the largest |s_i| at which row i was found unresolvable
    // against its column's own noise (issue #183). A high-water mark, so the
    // relaxation is never withdrawn — see refresh_sens_error_floor.
    std::vector<double> floor_unresolvable; // n_sens*ns
    std::vector<double> floor_terms;        // ns scratch: Σ|term| of one column's row
    std::vector<double> floor_jac;          // nnz scratch: the analytical Jacobian
    double floor_tau = 0.0;                 // time scale ε·Σ|term| is multiplied by
    double floor_rtol = 0.0;                // reltolS, needed to re-tell CVODES
    bool floor_active = false;
    // Which noise sources this run floors. Requested through
    // BNGSIM_SENS_FLOOR_PARTS, then narrowed by what the model can actually
    // supply — an emitted term scale, an analytical Jacobian, or neither.
    bool floor_do_dfdp_terms = true;     // Σ|term| of ∂f/∂p, from the emitter
    bool floor_do_jac_terms = true;      // Σ_j|J_ij||s_j|, from the analytical Jacobian
    bool floor_do_col_norm = true;       // ε‖s‖∞, the column's representation floor
    bool floor_do_unresolvable = true;   // relax rows whose rtol band is under ε‖s‖∞ (#183)
    double floor_unresolvable_cap = 1e3; // ulps of ‖s‖∞ the relaxation may reach
    int floor_refreshes = 0;
    double floor_max_relax = 1.0; // largest floor/base ratio applied, for diagnostics
};

// Scratch for the issue #48 switch-time sensitivity jump: the RHS on either
// branch at the crossing state, and the state copy whose clock gets nudged
// across the threshold to select the branch. Owned by run() and sized once,
// before the integration loop, so a crossing itself allocates nothing.
struct SwitchJumpScratch {
    std::vector<double> f_minus;
    std::vector<double> f_plus;
    std::vector<double> ywork;

    void resize(int ns) {
        f_minus.resize(static_cast<size_t>(ns));
        f_plus.resize(static_cast<size_t>(ns));
        ywork.resize(static_cast<size_t>(ns));
    }

    // Non-copyable on purpose: taking this by value would compile and run,
    // and the only symptom would be the per-crossing allocation the pre-sizing
    // exists to avoid. SensitivityState gets this for free (its N_Vector guard
    // is non-copyable); this one has to ask.
    SwitchJumpScratch() = default;
    SwitchJumpScratch(const SwitchJumpScratch &) = delete;
    SwitchJumpScratch &operator=(const SwitchJumpScratch &) = delete;
};

// ─── CvodeSimulator::Impl ───────────────────────────────────────────────────

struct CvodeSimulator::Impl {
    NetworkModel &model;
    double rtol = 1e-8;
    double atol = 1e-8;
    // Simulator-level per-species absolute tolerance (issue #196). Empty keeps
    // the scalar `atol` above. A non-empty SolverOptions::atol_vec on the run
    // itself wins over this, mirroring how opts.atol wins over `atol`.
    std::vector<double> atol_vec;
    int max_steps = 20000; // Matches SolverOptions::max_steps default

    // Direct linear solver chosen by the most recent setup_linsol_and_jac()
    // (a LinearSolverKind). Both run() and run_warm() call setup just before
    // integrating, so this is fresh when the solver stats are recorded.
    int linear_solver_used = LINEAR_SOLVER_DENSE;

    // ─── Counters of the run's closed segments (issue #182) ─────────────────
    // Every CVodeGetNum* counter restarts at 0 on a (re-)initialization, and
    // this path re-initializes at every event fire, switch crossing and chatter
    // re-arm. Sampling them once at the end therefore reports only the segment
    // after the LAST re-init — which is empty, and so reads 0 steps, when an
    // event fires at the final instant (the issue's `t_ins == t_end`). What a
    // caller wants is the run, so the segments are banked here as they close:
    // reinit_cvode() adds the segment CVodeReInit is about to zero, and
    // record_solver_stats() adds the one still open. Reset at the top of each
    // run (both paths), since the counts belong to that run alone.
    struct SegmentCounters {
        long int steps = 0;
        long int rhs_evals = 0;
        long int lin_solv_setups = 0;
        long int nonlin_iters = 0;
        long int nonlin_conv_fails = 0;
        long int err_test_fails = 0;

        void operator+=(const SegmentCounters &o) {
            steps += o.steps;
            rhs_evals += o.rhs_evals;
            lin_solv_setups += o.lin_solv_setups;
            nonlin_iters += o.nonlin_iters;
            nonlin_conv_fails += o.nonlin_conv_fails;
            err_test_fails += o.err_test_fails;
        }
    };
    SegmentCounters closed_segments;

    // The counters CVODE has accumulated since its last (re-)initialization.
    static SegmentCounters read_segment_counters(void *cvode_mem);

    // The one way to re-initialize CVODE mid-run: banks the closing segment's
    // counters before CVodeReInit zeroes them (issue #182), then re-inits.
    // Returns CVodeReInit's flag so each caller keeps its own error message.
    int reinit_cvode(void *cvode_mem, sunrealtype t, N_Vector y);

    // Cached codegen library + resolved symbols (GH #77). dlopen + dlsym +
    // dlclose on every run() is the dominant fixed per-run overhead on the
    // codegen path — enough that codegen lost to ExprTk on short-horizon
    // models where there is no integration compute to amortize it against.
    // The library is loaded once and the handle/function pointers cached,
    // keyed by .so path; repeated run()s on the same simulator reuse the
    // already-mapped library instead of re-mapping it. A path change (rare:
    // a regenerated .so) triggers a one-time reload. The DynamicLibrary stays
    // open for the simulator's lifetime and is unloaded by ~Impl.
    std::string codegen_so_path_cached;
    DynamicLibrary codegen_lib;
    CvodeUserData::CodegenRhsFn codegen_fn = nullptr;
    CvodeUserData::CodegenSensRhsFn codegen_sens_fn = nullptr;
    CvodeUserData::CodegenSensTermScaleFn codegen_sens_term_scale_fn = nullptr;
    CvodeUserData::CodegenJacFn codegen_jac_fn = nullptr;
    CvodeUserData::CodegenJacSparseFn codegen_jac_sparse_fn = nullptr;
    CvodeUserData::CodegenOutputsFn codegen_outputs_fn = nullptr;
    CvodeUserData::CodegenOutputSensFn codegen_output_sens_fn = nullptr;

    // In-process MIR micro-JIT of the codegen RHS (GH #78). The analogue of the
    // dlopen path above: when SolverOptions::codegen_c_source is set, the C
    // source is JIT-compiled once and the resolved function pointers cached,
    // keyed by the source string. A source change forces a one-time rebuild.
    // The MirJit owns the JIT'd code for the simulator's lifetime (mirrors how
    // codegen_lib owns the dlopen'd library).
    std::string codegen_c_source_cached;
    MirJit codegen_jit;

    // ─── Warm CVODE state (GH #102 reaction kernel) ─────────────────────────
    // A plain CvodeSimulator::run() rebuilds *all* SUNDIALS state every call —
    // SUNContext, the N_Vector, CVODE memory, and (most expensively) the KLU
    // sparse linear solver with a fresh symbolic factorization. That fixed
    // per-call cost (~5.7/33.7/83.4 ms at 10K/50K/100K species) is negligible
    // against a long integration but dominates a hybrid splitting loop that
    // takes many *small* coupling steps. The warm path keeps these objects
    // alive on the simulator and re-enters via CVodeReInit, which reuses the
    // allocations and — critically — keeps the linear solver attached so KLU
    // does NOT redo its symbolic factorization (first_factorize stays 0); only
    // a cheap numeric refactor runs per step. Used only for the simple case
    // (no events, no sensitivities, no JAX Jacobian); the cold run() path is
    // unchanged for everything else. Set BNGSIM_NO_WARM_CVODE to force the cold
    // path (used by the microbench to measure the warm win).
    //
    // The guards are declared in dependency order so they destruct in reverse
    // (LS, A, cvode_mem, y freed before ctx) — matching the local-guard order
    // in run(). Heap-allocated (unique_ptr) so a cold-only simulator never
    // creates a SUNContext, and so &user_data is a stable pointer for CVODE.
    struct WarmCache {
        bool valid = false;
        SunContextGuard ctx; // declared first → freed last
        NVectorGuard y;
        CvodeMemGuard cvode_mem;
        SUNMatrixGuard A;
        SUNLinSolGuard LS;
        CvodeUserData user_data{nullptr}; // CVODE holds &user_data
        std::vector<double> codegen_param_buf;
        // Fingerprint of the configuration the persistent objects were built
        // for. Any mismatch forces a full teardown + rebuild before reuse.
        int ns = -1;
        double rtol = 0.0;
        double atol = 0.0;
        // Per-species atol the persistent objects were built with (issue #196).
        // Part of the fingerprint like every other tolerance: CVodeReInit does
        // not touch tolerances, so reusing memory built for a different vector
        // would silently integrate at the previous run's tolerances.
        std::vector<double> atol_vec;
        // Tracking depth the persistent objects were built with (issue #213).
        // In the fingerprint for the same reason as atol_vec: CVodeReInit does
        // not touch tolerances, so reusing memory that was put on
        // CVodeWFtolerances (or left off it) would integrate this run at the
        // previous run's rule.
        double atol_track_decades = 0.0;
        double max_step_size = -1.0;
        int max_steps = 0;
        std::string jacobian;
        std::string codegen_so_path;
        std::string codegen_c_source; // MIR-JIT source fingerprint (GH #78)
        bool force_dense = false;
        bool force_sparse = false;
        bool use_sparse = false;
        int linear_solver = LINEAR_SOLVER_DENSE;
    };
    std::unique_ptr<WarmCache> warm;

    Impl(NetworkModel &m) : model(m) {}

    // Resolve the codegen RHS (loading/caching the .so) and build the parameter
    // mirror the codegen function reads from; returns the RHS function pointer.
    // Shared by the cold run() and the warm path so the codegen ABI lives once.
    CVRhsFn setup_codegen_rhs(const SolverOptions &opts, CvodeUserData &user_data,
                              std::vector<double> &codegen_param_buf);

    // Build the dense/sparse linear solver into A_guard/LS_guard, attach it to
    // cvode_mem, and select + install the Jacobian callback. Mirrors the
    // linear-solver + Jacobian block of run(); shared by both paths so the
    // KLU/analytical/colored selection lives once.
    void setup_linsol_and_jac(void *cvode_mem, SUNContext ctx, N_Vector y, SUNMatrixGuard &A_guard,
                              SUNLinSolGuard &LS_guard, const SolverOptions &opts,
                              CvodeUserData &user_data, bool use_sparse, int ns,
                              int linear_solver_kind);

    int choose_linear_solver_kind(bool use_sparse, const SolverOptions &opts, int ns);

    // Warm fast path: persistent CVODE memory reused via CVodeReInit. Handles
    // only the no-events / no-sensitivity / non-JAX case (see WarmCache).
    Result run_warm(const TimeSpec &times, const SolverOptions &opts, bool use_sparse);

    // ─── Cold-path run() setup steps ────────────────────────────────────────
    // Named boxes for the sequential configuration run() performs before it
    // starts stepping (GH #109). Each owns one block of the former inline
    // body, with the reasoning that block carried; run() reads as the sequence
    // of these calls followed by the integration loop. None of them changes
    // behavior — they are the same statements in the same order.

    // ns == 0: no ODE state to integrate, so there is no CVODE setup at all.
    Result run_algebraic_only(const TimeSpec &times);

    // Dense vs sparse (KLU) matrix decision, including the two force flags.
    bool choose_use_sparse(const SolverOptions &opts, int ns) const;

    // Create the SUNContext / state vector / CVODE memory, seed y from the
    // model, wire the (cached) codegen RHS into user_data, and apply the
    // tolerances and step limits. The RAII guards stay owned by run() (their
    // declaration order there is the teardown order), so they are filled in
    // place here rather than returned.
    void create_cvode_core(const TimeSpec &times, const SolverOptions &opts, int ns, double rtol,
                           double atol, const std::vector<double> &atol_v, int max_steps,
                           SunContextGuard &ctx, NVectorGuard &y, CvodeMemGuard &cvode_mem,
                           CvodeUserData &user_data, std::vector<double> &codegen_param_buf);

    // Resolve the per-species absolute tolerance in force for one run (issue
    // #196): the run's own vector when it has one, else the simulator-level
    // vector set by set_tolerances, else empty — which means the scalar atol
    // and CVodeSStolerances, exactly as before #196.
    const std::vector<double> &resolve_atol_vec(const SolverOptions &opts) const {
        return opts.atol_vec.empty() ? atol_vec : opts.atol_vec;
    }

    // Reject an unknown opts.jacobian, and reject the two strategies that ask
    // for something this model/request cannot supply. Also copies the JAX
    // callback into user_data.
    void validate_jacobian_option(const SolverOptions &opts, CvodeUserData &user_data);

    // Resolve the requested sensitivity columns, seed s(0), and hand CVODES its
    // sensitivity problem. Fills `sens`, whose arrays back raw pointers in
    // user_data and in CVODES for the rest of the run.
    // ``times`` is read only for the integration horizon, which sets the time
    // scale of the issue #177 sensitivity error floor.
    void setup_forward_sensitivities(const TimeSpec &times, const SolverOptions &opts, int ns,
                                     double rtol, double atol, const std::vector<double> &atol_v,
                                     N_Vector y, void *cvode_mem, CvodeUserData &user_data,
                                     SensitivityState &sens);

    // ── Sensitivity error floor (issue #177) ─────────────────────────────
    // Decide whether the floor is armed for this run and size its scratch.
    void setup_sens_error_floor(const SolverOptions &opts, int ns, double rtol, double horizon,
                                CvodeUserData &user_data, SensitivityState &sens);

    // Re-derive atolS from the magnitudes actually being summed at (t, y, s)
    // and hand the result back to CVODES. No-op unless the floor is armed.
    void refresh_sens_error_floor(void *cvode_mem, double t, N_Vector y, CvodeUserData &user_data,
                                  SensitivityState &sens, int ns);

    // Attach cvode_event_root_fn as CVODE's root function, and silence the
    // benign tiny-step warning a discontinuity root provokes.
    void register_roots(void *cvode_mem, SUNContext ctx, int n_roots, int n_disc);

    // Size the Result and its (optional) sensitivity blocks, and name every axis.
    void allocate_run_result(Result &result, const SolverOptions &opts, int n_out, int n_sens_p,
                             int n_sens_ic);

    // Copy CVODE's counters into the Result's solver_stats block. Used by the
    // warm path too — the two recorded the same counters in the same order.
    void record_solver_stats(void *cvode_mem, SUNLinearSolver ls, Result &result);

    // Publish the final state (and the forward-sensitivity carry-over seed)
    // back onto the model for the next action in a multi-action sequence.
    void write_final_state_back(const SolverOptions &opts, int ns, const double *y_data,
                                double final_t, const SensitivityState &sens);

    // ─── Forward-sensitivity jumps at a discontinuity ───────────────────────
    // The two places a sensitivity column has to jump rather than be
    // integrated: across an event's assignments (GH #212, plus the issue #49
    // event-time term) and across a switch-time crossing (issue #48). Both
    // were function-scope lambdas in run() (GH #135); they touch none of the
    // event bookkeeping the stepping loop writes (trigger_was_true /
    // pending_events / event_dormant / event_rng), so unlike the firing
    // lambdas they need no shared mutable state to move out.
    //
    // `sens` carries what used to be six separate captures. run()'s
    // `wants_sensitivity` is not among them: it is
    // `!param_names.empty() || !ic_species_names.empty()` and `sens.n_total`
    // is the sum of those two sizes, so the lambdas' `!wants_sensitivity ||
    // n_sens == 0` guard was testing one condition twice.

    // Put the model's parameters back on the nominal point CVODES' last
    // finite-difference probe left them off. Every derivative either jump
    // reads is taken there, so both call this first.
    void restore_nominal_params(const SensitivityState &sens);

    // Re-evaluate every expression-valued parameter from the current parameter
    // values, so a finite difference over a *primary* carries into the derived
    // parameters that read it (`k := 2*kbase` must move when kbase is
    // perturbed, or its column comes back a flat zero). `skip_param_idx` is the
    // parameter being perturbed: re-deriving that one would silently undo the
    // perturbation, which is the failure this exists to prevent, so it is left
    // alone. Pass -1 to re-derive all. Function-bound parameters (an SBML
    // assignment rule) are refreshed by evaluate_functions instead, so a caller
    // sandwiches this between two of its own state syncs.
    void rederive_expression_params(int skip_param_idx);

    // Read s⁻ out of CVODES at t_evt. MUST run before the caller's
    // CVodeReInit — see the ordering note on apply_event_sensitivity_jump.
    std::vector<std::vector<double>> capture_event_sens(void *cvode_mem, int ns, double t_evt,
                                                        SensitivityState &sens);

    // Put an unjumped s back into CVODES after a bare CVodeReInit (issue #146).
    // The counterpart of capture_event_sens for a root that changes nothing:
    // the columns are continuous across it, but CVodeReInit rewinds only the
    // state stepper, so the sensitivity solver must still be restarted at the
    // same instant or it resumes from the interrupted step's end.
    void resume_sens_after_reinit(void *cvode_mem, int ns,
                                  const std::vector<std::vector<double>> &s_at_reinit,
                                  SensitivityState &sens);

    // Jump dx/dθ across an event's assignments and resume CVODES from it.
    // `at_run_start` marks the SBML L3 §3.4.5 t=0 fire, where the trigger was
    // already satisfied when the run began: the fire is pinned to t_start
    // rather than located, so ∂t*/∂θ is 0 there and must not be differentiated
    // (the same reason issue #49's detector drops a crossing at or before
    // t_start).
    void apply_event_sensitivity_jump(const SolverOptions &opts, void *cvode_mem, int ns,
                                      double t_evt, const std::vector<int> &fired,
                                      const std::vector<double> &x_minus,
                                      const std::vector<std::vector<double>> &s_minus,
                                      SensitivityState &sens, bool at_run_start = false);

    // ∂t*/∂θ at a located crossing of the surface g = 0, by the implicit
    // function theorem (issue #144, reused for the rate-law switches of issue
    // #150). `gidx`/`support` are the residual and the species a difference of
    // it has to perturb; `subject` names the crossing in the tangency refusal
    // ("event 'E1' crosses its trigger"). Throws on a tangential crossing.
    void residual_dtstar(int gidx, const std::vector<int> &support, const std::string &subject,
                         double t_evt, int ns, const std::vector<double> &x_minus,
                         const std::vector<double> &f_minus,
                         const std::vector<std::vector<double>> &s_minus,
                         const SensitivityState &sens, std::vector<double> &tau_out);

    // The event-trigger entry point to residual_dtstar. Returns false when this
    // event has no usable residual, in which case the caller leaves ∂t*/∂θ at
    // whatever the issue #49 detector supplied (zero for a fixed-time event).
    bool state_trigger_dtstar(int event_idx0, double t_evt, int ns,
                              const std::vector<double> &x_minus,
                              const std::vector<double> &f_minus,
                              const std::vector<std::vector<double>> &s_minus,
                              const SensitivityState &sens, std::vector<double> &tau_out);

    // Jump dx/dθ across a switch-time crossing and restart both steppers at
    // the kink.
    void apply_switch_sensitivity_jump(void *cvode_mem, N_Vector y, int ns, double t_evt,
                                       const SwitchTimeSens &sw, SwitchJumpScratch &scratch,
                                       SensitivityState &sens);

    // Add the saltation jump of a state-dependent rate-law switch to `s` in
    // place (issue #150). `batch` is every switch whose residual root fired at
    // this instant — usually one, and several when one crossing is written more
    // than one way (issue #153); the caller re-seeds CVODES from `s` afterwards.
    // Throws on a tangential crossing, when the two branches cannot be told
    // apart at x(t*), or when a batch does not resolve to a single crossing.
    void
    apply_state_switch_sensitivity_jump(void *cvode_mem, N_Vector y, int ns, double t_evt,
                                        const std::vector<const NetworkModel::StateSwitch *> &batch,
                                        std::vector<std::vector<double>> &s,
                                        SensitivityState &sens);
};

// ─── Shared integrator setup (used by run() and run_warm()) ──────────────────

CVRhsFn CvodeSimulator::Impl::setup_codegen_rhs(const SolverOptions &opts, CvodeUserData &user_data,
                                                std::vector<double> &codegen_param_buf) {
    CVRhsFn rhs_fn = cvode_rhs; // default: ExprTk-based RHS

    // In-process MIR micro-JIT backend (GH #78). When codegen_c_source is set,
    // JIT-compile the same C the codegen emits instead of dlopen'ing a cc-built
    // .so. Both backends resolve the identical bngsim_codegen_rhs (+ optional
    // sens/jac) symbols and feed the same cvode_codegen_rhs callback, so the
    // setup below the backend split is shared. The JIT source takes precedence
    // over a .so path if both happen to be set.
    const bool use_jit = !opts.codegen_c_source.empty();

    if (!use_jit && opts.codegen_so_path.empty()) {
        return rhs_fn;
    }

    if (use_jit) {
        // Compile + resolve once, then reuse across run()s (the cached MirJit is
        // an Impl member; a changed source forces a one-time rebuild). Mirrors
        // the dlopen caching below.
        if (!codegen_jit || opts.codegen_c_source != codegen_c_source_cached) {
            codegen_jit = MirJit(opts.codegen_c_source);
            codegen_fn = codegen_jit.symbol<CvodeUserData::CodegenRhsFn>("bngsim_codegen_rhs");
            codegen_sens_fn =
                codegen_jit.try_symbol<CvodeUserData::CodegenSensRhsFn>("bngsim_codegen_sens_rhs");
            codegen_sens_term_scale_fn =
                codegen_jit.try_symbol<CvodeUserData::CodegenSensTermScaleFn>(
                    "bngsim_codegen_sens_term_scale");
            codegen_jac_fn =
                codegen_jit.try_symbol<CvodeUserData::CodegenJacFn>("bngsim_codegen_jac");
            codegen_jac_sparse_fn = codegen_jit.try_symbol<CvodeUserData::CodegenJacSparseFn>(
                "bngsim_codegen_jac_sparse");
            codegen_outputs_fn =
                codegen_jit.try_symbol<CvodeUserData::CodegenOutputsFn>("bngsim_codegen_outputs");
            codegen_output_sens_fn = codegen_jit.try_symbol<CvodeUserData::CodegenOutputSensFn>(
                "bngsim_codegen_output_sens");
            codegen_c_source_cached = opts.codegen_c_source;
            // A subsequent switch back to the dlopen path must reload the .so.
            codegen_so_path_cached.clear();
        }
        user_data.codegen_dl_handle = nullptr; // JIT code is not a dlopen handle
    } else {
        // Load + resolve once, then reuse across run()s (the cached library is an
        // Impl member; a changed path forces a one-time reload). GH #77.
        if (!codegen_lib || opts.codegen_so_path != codegen_so_path_cached) {
            codegen_lib = DynamicLibrary(opts.codegen_so_path);
            codegen_fn = codegen_lib.symbol<CvodeUserData::CodegenRhsFn>("bngsim_codegen_rhs");
            codegen_sens_fn =
                codegen_lib.try_symbol<CvodeUserData::CodegenSensRhsFn>("bngsim_codegen_sens_rhs");
            codegen_sens_term_scale_fn =
                codegen_lib.try_symbol<CvodeUserData::CodegenSensTermScaleFn>(
                    "bngsim_codegen_sens_term_scale");
            codegen_jac_fn =
                codegen_lib.try_symbol<CvodeUserData::CodegenJacFn>("bngsim_codegen_jac");
            codegen_jac_sparse_fn = codegen_lib.try_symbol<CvodeUserData::CodegenJacSparseFn>(
                "bngsim_codegen_jac_sparse");
            codegen_outputs_fn =
                codegen_lib.try_symbol<CvodeUserData::CodegenOutputsFn>("bngsim_codegen_outputs");
            codegen_output_sens_fn = codegen_lib.try_symbol<CvodeUserData::CodegenOutputSensFn>(
                "bngsim_codegen_output_sens");
            codegen_so_path_cached = opts.codegen_so_path;
            // A subsequent switch to the JIT path must recompile the source.
            codegen_c_source_cached.clear();
        }
        user_data.codegen_dl_handle = codegen_lib.native_handle();
    }

    user_data.codegen_fn = codegen_fn;
    user_data.codegen_sens_fn = codegen_sens_fn;
    user_data.codegen_sens_term_scale_fn = codegen_sens_term_scale_fn;
    user_data.codegen_jac_fn = codegen_jac_fn;
    user_data.codegen_jac_sparse_fn = codegen_jac_sparse_fn;
    user_data.codegen_outputs_fn = codegen_outputs_fn;
    user_data.codegen_output_sens_fn = codegen_output_sens_fn;

    // Contiguous mirror of the model's live parameter values; rebuilt every
    // call so parameter edits between runs are picked up.
    const auto &params = model.parameters();
    codegen_param_buf.resize(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        codegen_param_buf[i] = params[i].value;
    }
    user_data.codegen_param_values = codegen_param_buf.data();

    user_data.codegen_so_data.param_values = user_data.codegen_param_values;
    user_data.codegen_so_data.tfun_ctx = &model;
    user_data.codegen_so_data.tfun_eval = codegen_tfun_eval_thunk;

    rhs_fn = cvode_codegen_rhs;
    return rhs_fn;
}

int CvodeSimulator::Impl::choose_linear_solver_kind(bool use_sparse, const SolverOptions &opts,
                                                    int ns) {
#ifdef BNGSIM_HAS_KLU
    if (use_sparse) {
        return LINEAR_SOLVER_KLU;
    }
#else
    (void)use_sparse;
#endif
    const bool use_lapack = should_use_lapack_dense(ns, model.jacobian_sparsity().density,
                                                    opts.force_dense_linear_solver);
    return use_lapack ? LINEAR_SOLVER_LAPACK_DENSE : LINEAR_SOLVER_DENSE;
}

void CvodeSimulator::Impl::setup_linsol_and_jac(void *cvode_mem, SUNContext ctx, N_Vector y,
                                                SUNMatrixGuard &A_guard, SUNLinSolGuard &LS_guard,
                                                const SolverOptions &opts, CvodeUserData &user_data,
                                                bool use_sparse, int ns, int linear_solver_kind) {
    int flag;
#ifdef BNGSIM_HAS_KLU
    if (use_sparse) {
        // Sparse: SUNSparseMatrix (CSC) + KLU
        const auto &sp = model.jacobian_sparsity();
        A_guard = SUNMatrixGuard(SUNSparseMatrix(ns, ns, sp.nnz, CSC_MAT, ctx));
        if (!A_guard) {
            throw std::runtime_error("SUNSparseMatrix failed");
        }

        sunindextype *col_ptrs = SUNSparseMatrix_IndexPointers(A_guard);
        sunindextype *row_indices = SUNSparseMatrix_IndexValues(A_guard);

        for (int j = 0; j <= ns; ++j) {
            col_ptrs[j] = static_cast<sunindextype>(sp.col_ptrs[j]);
        }
        for (int k = 0; k < sp.nnz; ++k) {
            row_indices[k] = static_cast<sunindextype>(sp.row_indices[k]);
        }

        LS_guard = SUNLinSolGuard(SUNLinSol_KLU(y, A_guard, ctx));
        if (!LS_guard) {
            throw std::runtime_error("SUNLinSol_KLU failed");
        }
        linear_solver_used = linear_solver_kind;
    } else
#endif
    {
        // Dense: SUNDenseMatrix + a dense direct solver. Large, genuinely-dense
        // Jacobians take the GH #84 BLAS dgetrf factor (built-in back-solve
        // retained); everything else uses the built-in dense LU. The matrix is
        // an ordinary column-major SUNDenseMatrix either way, so the analytical
        // / codegen dense Jacobian callbacks below are unaffected.
        A_guard = SUNMatrixGuard(SUNDenseMatrix(ns, ns, ctx));
        const bool use_lapack = (linear_solver_kind == LINEAR_SOLVER_LAPACK_DENSE);
        LS_guard = SUNLinSolGuard(make_dense_linear_solver(y, A_guard, ctx, use_lapack));
        if (!LS_guard) {
            throw std::runtime_error("dense linear solver creation failed");
        }
        linear_solver_used = linear_solver_kind;
    }

    flag = CVodeSetLinearSolver(cvode_mem, LS_guard, A_guard);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSetLinearSolver failed");
    }

    // ─── Jacobian callback selection (respects opts.jacobian) ────────────────
    const std::string &jac_strategy = opts.jacobian;
#ifdef BNGSIM_HAS_KLU
    if (use_sparse) {
        // Materializes the coloring if this is the first sparse setup for the
        // model (GH #29) — build() no longer computes it, and this is the only
        // consumer. Cheap and idempotent afterward, including on the warm path.
        const auto &sp = model.ensure_jacobian_coloring();
        const bool analytical_ready = model.analytical_jacobian_complete();
        CVLsJacFn jac_fn = nullptr;

        if (jac_strategy != "fd" && analytical_ready &&
            (jac_strategy == "analytical" || jac_strategy == "auto")) {
            // Prefer the compiled CSC Jacobian (GH #162) when codegen resolved it;
            // it mirrors fill_sparse_analytical_jacobian without the per-step
            // ExprTk eval. Falls back to the interpreted sparse callback otherwise.
            jac_fn =
                user_data.codegen_jac_sparse_fn ? cvode_codegen_sparse_jac : cvode_analytical_jac;
            if (std::getenv("BNGSIM_JAC_DEBUG"))
                std::fprintf(stderr, "[jac] sparse Jacobian: %s\n",
                             user_data.codegen_jac_sparse_fn
                                 ? "compiled (bngsim_codegen_jac_sparse)"
                                 : "interpreted (cvode_analytical_jac)");
        } else if (sp.has_coloring()) {
            jac_fn = cvode_colored_jac;
            // Size the colored-FD scratch once, here, so the per-eval callback
            // allocates nothing (T4). assign() zeroes h_vals to match the
            // callback's original `h_vals(ns, 0.0)` init.
            user_data.colored_jac_y_pert.resize(ns);
            user_data.colored_jac_fy_pert.resize(ns);
            user_data.colored_jac_h_vals.assign(ns, 0.0);
        }

        if (!jac_fn) {
            // Backstop, expected unreachable. CVODE's built-in difference-quotient
            // Jacobian covers SUNMATRIX_DENSE and SUNMATRIX_BAND only — with a
            // sparse matrix and no callback, CVodeSetLinearSolver's initialization
            // fails with an opaque "no Jacobian constructor available", so refuse
            // here with something legible instead.
            //
            // Every pattern with a structural nonzero now colors (GH #29 removed
            // the density < 0.5 ceiling that used to skip near-dense ones, which
            // was the last way force_sparse_linear_solver could land here), so
            // reaching this means sp.nnz == 0: a structurally empty Jacobian, with
            // no column to perturb and nothing for KLU to factorize. sparse_ok
            // only rejects n == 0, so a model with species but no reactions can
            // still get here under the flag.
            throw std::runtime_error(
                "force_sparse_linear_solver: this model cannot supply a sparse "
                "Jacobian — its analytical Jacobian is unavailable" +
                std::string(jac_strategy == "fd" ? " (suppressed by jacobian=\"fd\")" : "") +
                " and its Jacobian sparsity pattern has no structural nonzeros to "
                "color for finite differences, leaving nothing to fill the CSC "
                "matrix KLU factorizes. Drop the flag to use the dense solver, "
                "which finite-differences the full matrix directly.");
        }
        flag = CVodeSetJacFn(cvode_mem, jac_fn);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeSetJacFn failed");
        }
    } else
#endif
        if (jac_strategy == "jax") {
        flag = CVodeSetJacFn(cvode_mem, cvode_jax_dense_jac);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeSetJacFn (jax) failed");
        }
    } else if (jac_strategy != "fd") {
        if (model.analytical_jacobian_complete() &&
            (jac_strategy == "analytical" || jac_strategy == "auto")) {
            CVLsJacFn dense_jac =
                user_data.codegen_jac_fn ? cvode_codegen_dense_jac : cvode_analytical_dense_jac;
            if (std::getenv("BNGSIM_JAC_DEBUG"))
                std::fprintf(stderr, "[jac] dense Jacobian: %s\n",
                             user_data.codegen_jac_fn ? "compiled (bngsim_codegen_jac)"
                                                      : "interpreted (cvode_analytical_dense_jac)");
            flag = CVodeSetJacFn(cvode_mem, dense_jac);
            if (flag != CV_SUCCESS) {
                throw std::runtime_error("CVodeSetJacFn (dense) failed");
            }
        }
    }
}

// ─── Assignment-rule copy-back map (GH #136) ─────────────────────────────────
//
// An SBML assignment rule targeting a species is loaded as a function named
// after that (fixed) species; at each output row its value overwrites the slot
// CVODE integrated, so the recorded species column reflects the rule. The match
// is by name and static for the model's life, so resolving it once turns the
// former per-row O(n_func × n_species) name scan (which dominated wall time on
// large models — 399 funcs × 786 species × 1001 rows on BIOMD0000000470) into a
// per-row O(#assignment-rule-species) copy. Returns (func_decl_idx,
// species_idx0) pairs; the species is the lowest-indexed fixed species whose
// name equals the function name, matching the old ascending-scan-then-break.
// Empty (and thus inert) for every model without assignment-rule species.
static std::vector<std::pair<int, int>> build_assignment_rule_copyback(const NetworkModel &model) {
    std::vector<std::pair<int, int>> map;
    const auto &funcs = model.functions();
    const auto &species = model.species();
    const int ns = static_cast<int>(species.size());
    std::unordered_map<std::string, int> fixed_name_to_idx;
    for (int si = 0; si < ns; ++si) {
        if (species[si].fixed) {
            fixed_name_to_idx.emplace(species[si].name, si);
        }
    }
    if (fixed_name_to_idx.empty()) {
        return map;
    }
    for (std::size_t fi = 0; fi < funcs.size(); ++fi) {
        auto it = fixed_name_to_idx.find(funcs[fi].name);
        if (it != fixed_name_to_idx.end()) {
            map.push_back({static_cast<int>(fi), it->second});
        }
    }
    return map;
}

// ─── Observable output sensitivities (GH #197) ───────────────────────────────
//
// BNGL observables are linear in species: obs_j = Σ_i c_ji·x_i, where c_ji
// folds the GroupEntry factor and — for an amount-valued species — the
// volume scaling that update_observables() applies (model.cpp:1142). So
// d obs_j/dθ = Σ_i c_ji·dx_i/dθ: a runtime chain rule over the CVODES
// species sensitivities extracted in run(), no codegen required. The same
// coefficients drive both the parameter axis and the IC axis; only the
// source dx/dθ vector differs (yS parameter cols vs IC cols). Expression
// (global-function) sensitivities are nonlinear and are left to the codegen
// stage (#198) — those blocks stay empty here.
// Issue #170 stage 3: `weight` is not constant in every parameter. When the
// amount conversion above reads a *writable* compartment size, the coefficient
// itself is that parameter, so d obs_j/dV carries a direct Σ factor_ji·x_i on
// top of the chain rule — the observable's own units move with the volume. That
// term is the whole answer at t=0, where every dx_i/dθ is still zero.
struct ObsSensTerm {
    int obs;           // observable row (0-based, recording order)
    int species0;      // 0-based species index it reads
    double weight;     // c_ji = factor · (amount_valued ? volume_factor : 1)
    int vol_param;     // (#170) the parameter `weight`'s conversion IS, or -1
    double raw_factor; // ∂c_ji/∂V = factor, meaningful when vol_param >= 0
};

static std::vector<ObsSensTerm> build_observable_sens_terms(const NetworkModel &model) {
    std::vector<ObsSensTerm> terms;
    const int ns = model.n_species();
    const int n_obs = model.n_observables();
    const auto &obs_list = model.observables();
    const auto &spec_list = model.species();
    for (int j = 0; j < n_obs; ++j) {
        for (const auto &e : obs_list[j].entries) {
            const int idx0 = e.species_index - 1; // entries are 1-based
            if (idx0 < 0 || idx0 >= ns) {
                continue;
            }
            double weight = e.factor;
            const auto &sp = spec_list[idx0];
            int vol_param = -1;
            if (sp.amount_valued) {
                weight *= sp.volume_factor;
                vol_param = sp.volume_param_idx0;
            }
            terms.push_back({j, idx0, weight, vol_param, e.factor});
        }
    }
    return terms;
}

// ─── Warm fast path ──────────────────────────────────────────────────────────
//
// Reuses persistent SUNDIALS objects across calls via CVodeReInit. Numerically
// identical to run() for the case it covers (no events, no sensitivities, no
// JAX Jacobian): with no event roots, run()'s inner sub-step loop reduces to a
// single CVode() per output point, exactly as here. Recording mirrors run()
// (initial point with no AR copy-back; later points copy assignment-rule
// function values into fixed-species slots).
Result CvodeSimulator::Impl::run_warm(const TimeSpec &times, const SolverOptions &opts,
                                      bool use_sparse) {
    const int ns = model.n_species();
    const int n_obs = model.n_observables();
    const int n_func = model.n_functions();

    const double rtol = (opts.rtol > 0) ? opts.rtol : this->rtol;
    const double atol = (opts.atol > 0) ? opts.atol : this->atol;
    const std::vector<double> &atol_v = resolve_atol_vec(opts);
    validate_atol_vector(atol_v, ns, "run()");
    validate_atol_tracking(opts.atol_track_decades, atol_v, ns, "run()");
    const int max_steps = (opts.max_steps > 0) ? opts.max_steps : this->max_steps;

    // Validate the Jacobian strategy exactly as the cold path does. ("jax"
    // never reaches here — JAX models are routed to the cold path.)
    const std::string &jac_strategy = opts.jacobian;
    if (jac_strategy != "auto" && jac_strategy != "analytical" && jac_strategy != "fd") {
        throw std::runtime_error("Invalid jacobian option '" + jac_strategy +
                                 "'. Must be \"auto\", \"analytical\", \"fd\", or \"jax\".");
    }
    if (jac_strategy == "analytical" && !model.analytical_jacobian_complete()) {
        throw std::runtime_error(
            "jacobian=\"analytical\" requested but the analytical Jacobian is not "
            "available for this model. It covers Elementary mass-action rate laws and "
            "Functional rate laws whose derivatives could be symbolically derived; "
            "Michaelis-Menten and rate laws that fail symbolic differentiation fall "
            "back to finite differences.");
    }

    if (!warm) {
        warm = std::make_unique<WarmCache>();
    }
    WarmCache &w = *warm;
    const int desired_linear_solver = choose_linear_solver_kind(use_sparse, opts, ns);

    // Can the persistent objects be reused, or must they be rebuilt? Reuse only
    // when every setup-affecting input is unchanged since the last build.
    const bool reuse =
        w.valid && w.ns == ns && w.rtol == rtol && w.atol == atol && w.atol_vec == atol_v &&
        w.atol_track_decades == opts.atol_track_decades && w.max_steps == max_steps &&
        w.max_step_size == opts.max_step_size && w.jacobian == jac_strategy &&
        w.force_dense == opts.force_dense_linear_solver &&
        w.force_sparse == opts.force_sparse_linear_solver && w.use_sparse == use_sparse &&
        w.linear_solver == desired_linear_solver && w.codegen_so_path == opts.codegen_so_path &&
        w.codegen_c_source == opts.codegen_c_source;

    // Mark invalid up front; only a fully successful run restores validity, so
    // any throw (CVODE failure, timeout) forces a clean rebuild next call.
    w.valid = false;

    if (!reuse) {
        // ─── Full (re)build of the persistent SUNDIALS objects ───────────────
        // Release in dependency-safe order before remaking.
        w.LS = SUNLinSolGuard{};
        w.A = SUNMatrixGuard{};
        w.cvode_mem = CvodeMemGuard{};
        w.y = NVectorGuard{};
        if (!w.ctx) {
            w.ctx = SunContextGuard{};
        }
        if (!w.ctx) {
            throw std::runtime_error("SUNContext_Create failed");
        }

        w.y = NVectorGuard(N_VNew_Serial(ns, w.ctx));
        if (!w.y) {
            throw std::runtime_error("N_VNew_Serial failed");
        }
        double *y_data = w.y.data();
        for (int i = 0; i < ns; ++i) {
            y_data[i] = model.species()[i].concentration;
        }

        w.cvode_mem = CvodeMemGuard(CVodeCreate(CV_BDF, w.ctx));
        if (!w.cvode_mem) {
            throw std::runtime_error("CVodeCreate failed");
        }

        w.user_data = CvodeUserData{&model};
        CVRhsFn rhs_fn = setup_codegen_rhs(opts, w.user_data, w.codegen_param_buf);

        int flag = CVodeInit(w.cvode_mem, rhs_fn, times.t_start, w.y);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeInit failed: " + std::to_string(flag));
        }
        w.user_data.atol_tracking = make_atol_tracking(rtol, atol_v, opts.atol_track_decades);
        apply_cvode_tolerances(w.cvode_mem, w.ctx, rtol, atol, atol_v, ns,
                               w.user_data.atol_tracking.active() ? cvode_tracking_ewt : nullptr);
        CVodeSetUserData(w.cvode_mem, &w.user_data);
        CVodeSetMaxNumSteps(w.cvode_mem, max_steps);
        if (opts.max_step_size > 0) {
            CVodeSetMaxStep(w.cvode_mem, opts.max_step_size);
        }

        setup_linsol_and_jac(w.cvode_mem, w.ctx, w.y, w.A, w.LS, opts, w.user_data, use_sparse, ns,
                             desired_linear_solver);

        // Record the fingerprint these objects were built for.
        w.ns = ns;
        w.rtol = rtol;
        w.atol = atol;
        w.atol_vec = atol_v;
        w.atol_track_decades = opts.atol_track_decades;
        w.max_steps = max_steps;
        w.max_step_size = opts.max_step_size;
        w.jacobian = jac_strategy;
        w.force_dense = opts.force_dense_linear_solver;
        w.force_sparse = opts.force_sparse_linear_solver;
        w.use_sparse = use_sparse;
        w.linear_solver = desired_linear_solver;
        w.codegen_so_path = opts.codegen_so_path;
        w.codegen_c_source = opts.codegen_c_source;
    } else {
        // ─── Warm re-entry: refill y from the model, refresh params, reinit ──
        // This is the whole point: no SUNContext/CVODE/linear-solver rebuild,
        // and KLU keeps its symbolic factorization (only a numeric refactor
        // runs on the next solve).
        double *y_data = w.y.data();
        for (int i = 0; i < ns; ++i) {
            y_data[i] = model.species()[i].concentration;
        }
        if (!opts.codegen_so_path.empty() || !opts.codegen_c_source.empty()) {
            const auto &params = model.parameters();
            for (size_t i = 0; i < params.size() && i < w.codegen_param_buf.size(); ++i) {
                w.codegen_param_buf[i] = params[i].value;
            }
        }
        int flag = CVodeReInit(w.cvode_mem, times.t_start, w.y);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeReInit failed: " + std::to_string(flag));
        }
    }

    // GH #132: restart the adaptive BLAS-dense factor-count gate for this run.
    // The warm path reuses one persistent dense solver across run()s, so without
    // this a long prior run would leave factor_count > K and push this (possibly
    // short) run straight onto the BLAS factor on its first factorization. The
    // cold run() path rebuilds its solver each call and so resets naturally;
    // only the warm reuse needs this. No-op for the sparse/KLU and built-in
    // dense solvers (they carry a different setup op). Harmless on the fresh-
    // build branch above, where the counter is already 0.
    lapack_dense_reset_factor_count(w.LS);

    // Same restart, for the same reason, on the issue #182 counters: the
    // re-entry CVodeReInit above zeroed CVODE's own, and the carried half is an
    // Impl member that would otherwise still hold the previous run's segments.
    // (Nothing re-inits mid-run here — the warm path takes no events — so this
    // stays 0 through to record_solver_stats.)
    closed_segments = SegmentCounters{};

    // ─── Integration loop (no events, no sensitivities) ──────────────────────
    void *cvode_mem = w.cvode_mem;
    double *y_data = w.y.data();
    WallClockBudget budget(opts.timeout_seconds);

    std::vector<double> t_out = times.output_times();
    const int n_out = static_cast<int>(t_out.size());

    Result result;
    result.allocate(n_out, ns, n_obs);
    result.set_species_names(model.species_names());
    {
        auto reported = model.reported_species_indices();
        if (reported.size() != static_cast<std::size_t>(ns)) {
            result.set_reported_species_indices(std::move(reported));
        }
    }
    result.set_observable_names(model.observable_names());
    if (n_func > 0) {
        result.set_expression_names(model.function_names());
    }

    std::vector<double> obs_buf(n_obs);
    const auto ar_copyback = build_assignment_rule_copyback(model);

    // Per-output-row observable + function evaluation (GH #136). When a codegen
    // output evaluator resolved (large models on this event-free warm path), call
    // the compiled function once per row to fill obs_buf + func_out — far cheaper
    // than the interpreted update_observables() + evaluate_functions() ExprTk
    // pass. Otherwise fall back to the interpreted path (whose function values
    // live in model.function_value_cache()). fill_row returns a pointer to the
    // n_func function values for this row (obs always land in obs_buf), computed
    // from the current y BEFORE any assignment-rule copy-back — matching the
    // interpreted ordering exactly.
    const bool use_codegen_outputs = (w.user_data.codegen_outputs_fn != nullptr);
    std::vector<double> func_out(use_codegen_outputs ? n_func : 0);
    auto fill_row = [&](double t_row) -> const double * {
        if (use_codegen_outputs) {
            w.user_data.codegen_outputs_fn(t_row, y_data, obs_buf.data(), func_out.data(),
                                           &w.user_data.codegen_so_data);
            return func_out.data();
        }
        // GH #106/#231: a rate_of__<species> accessor reads model.current_derivs,
        // which is only refreshed as a side effect of an RHS eval. Without an
        // explicit refresh here the recorded value is the last *internal*
        // integration step's derivative — and at t=0, before any step, a stale
        // (zero) buffer. Probe dx/dt at this exact (t_row, y_data) so every
        // recorded rateOf is exact, including the initial row. No-op otherwise.
        if (model.uses_rateof()) {
            model.refresh_rateof_derivs(t_row, y_data);
        }
        model.update_observables(y_data);
        model.evaluate_functions(t_row);
        for (int j = 0; j < n_obs; ++j) {
            obs_buf[j] = model.observables()[j].total;
        }
        return model.function_value_cache().data();
    };

    // Record initial state (no AR copy-back at t=0, matching run()).
    {
        const double *fvals0 = fill_row(times.t_start);
        result.record(0, times.t_start, y_data, obs_buf.data());
        if (n_func > 0) {
            result.record_expressions(0, fvals0);
        }
    }

    const bool check_ss = opts.steady_state;
    const double ss_tol = (opts.steady_state_tol > 0.0) ? opts.steady_state_tol : atol;
    std::vector<double> ss_derivs;
    if (check_ss) {
        ss_derivs.resize(ns);
    }
    int last_recorded_index = 0;
    bool ss_reached = false;
    double ss_residual_last = 0.0;

    for (int i = 1; i < n_out; ++i) {
        if (budget.active())
            budget.check();

        sunrealtype t_ret;
        int flag = CVode(cvode_mem, t_out[i], w.y, &t_ret, CV_NORMAL);
        retry_while_advancing(cvode_mem, t_out[i], w.y, &t_ret, flag,
                              "while integrating to the next output point", [&budget] {
                                  if (budget.active())
                                      budget.check();
                              });
        if (flag < 0) {
            throw std::runtime_error("CVODE integration failed at t=" + std::to_string(t_out[i]) +
                                     " with flag=" + std::to_string(flag));
        }

        const double *fvals = fill_row(t_ret);

        // Copy assignment-rule function values into their fixed-species slots so
        // the recorded species column reflects the rule, not the stale ODE
        // value (mirrors run()'s per-point copy-back). The (func, species) pairs
        // were resolved once into ar_copyback (GH #136); the function values came
        // from fill_row just above (codegen or interpreted).
        if (!ar_copyback.empty()) {
            for (const auto &[fi, si] : ar_copyback) {
                y_data[si] = fvals[fi];
            }
        }

        result.record(i, static_cast<double>(t_ret), y_data, obs_buf.data());
        if (n_func > 0) {
            result.record_expressions(i, fvals);
        }

        last_recorded_index = i;

        if (check_ss) {
            model.compute_derivs(static_cast<double>(t_ret), y_data, ss_derivs.data());
            double sumsq = 0.0;
            for (int k = 0; k < ns; ++k) {
                sumsq += ss_derivs[k] * ss_derivs[k];
            }
            const double dx = std::sqrt(sumsq) / static_cast<double>(ns);
            ss_residual_last = dx;
            if (dx < ss_tol) {
                ss_reached = true;
                break;
            }
        }
    }

    // ─── Solver statistics ───────────────────────────────────────────────────
    // Same counters, in the same order, as the cold path — shared so the two
    // cannot drift (Impl::record_solver_stats). The BLAS factorization count
    // (GH #132) comes off the warm cache's persistent linear solver.
    record_solver_stats(cvode_mem, w.LS, result);
    if (check_ss) {
        result.solver_stats().steady_state_reached = ss_reached;
        result.solver_stats().steady_state_residual = ss_residual_last;
    }

    if (check_ss && ss_reached && last_recorded_index + 1 < n_out) {
        result.truncate(last_recorded_index + 1);
    }

    // ─── Write final state back to the model ─────────────────────────────────
    {
        auto &species = const_cast<std::vector<Species> &>(model.species());
        for (int i = 0; i < ns; ++i) {
            species[i].concentration = y_data[i];
        }
        const double final_t = (check_ss && ss_reached) ? t_out[last_recorded_index] : times.t_end;
        model.set_current_time(final_t);

        // GH #210 — the warm path never computes sensitivities, so it advances
        // the state (carry-over) without tracking dx/dθ: mark the state dirty
        // and drop any pending seed left by an earlier sensitivity run.
        model.set_ic_state_dirty(true);
        model.clear_pending_sens_seed();
    }

    // CVODE memory survived a full successful run — keep it warm for next call.
    w.valid = true;
    return result;
}

// ─── Cold-path run() setup steps (GH #109) ───────────────────────────────────
//
// One named box per block of the sequential configuration run() used to do
// inline. Declaration order below matches the call order in run().

// Algebraic-only model (GH #229): no ODE state, but assignment rules and
// functions of the SBML `time` csymbol — plus any constant outputs —
// still define a trajectory over the requested grid. RoadRunner
// integrates these; with no state to integrate, bngsim evaluates the
// observables + functions once per output row (no CVODE needed). The
// SBML semantic suite exercises this with pure parameter+assignmentRule
// models such as `p2 := 1 + time`.
//
// Events mutate state discretely at their trigger time; reproducing them
// needs the cold path's rootfinding + reinit machinery, which assumes ≥1
// integrator state to anchor the crossing — refuse those loud rather than
// silently dropping the assignment. Discontinuity triggers (piecewise /
// comparison expressions, GH #72) need NO special handling here: with no
// ODE to integrate, crossing-step accuracy is moot — evaluating the rule
// fresh at each output grid point already yields the correct piecewise
// value (exactly what RoadRunner reports).
Result CvodeSimulator::Impl::run_algebraic_only(const TimeSpec &times) {
    if (model.n_events() > 0) {
        throw std::runtime_error("Cannot simulate: model has no species but defines events "
                                 "(no integrator state to anchor the trigger crossing).");
    }
    const int n_obs = model.n_observables();
    const int n_func = model.n_functions();
    std::vector<double> t_out = times.output_times();
    const int n_out = static_cast<int>(t_out.size());

    Result result;
    result.allocate(n_out, /*n_species=*/0, n_obs);
    result.set_species_names(model.species_names());
    result.set_observable_names(model.observable_names());
    if (n_func > 0) {
        result.set_expression_names(model.function_names());
    }

    std::vector<double> obs_buf(n_obs);
    for (int i = 0; i < n_out; ++i) {
        const double t_row = t_out[i];
        // No species ⇒ update_observables never dereferences the conc pointer
        // (every observable entry's species index is out of range and is
        // skipped). evaluate_functions binds time() and the constant params.
        model.update_observables(nullptr);
        model.evaluate_functions(t_row);
        for (int j = 0; j < n_obs; ++j) {
            obs_buf[j] = model.observables()[j].total;
        }
        result.record(i, t_row, /*species_conc=*/nullptr, obs_buf.data());
        if (n_func > 0) {
            result.record_expressions(i, model.function_value_cache().data());
        }
    }
    return result;
}

// Decide dense vs sparse — the shared rule in bngsim/sparse_jacobian.hpp, which
// the steady-state march takes too (issue #128). Kept as a member so the call
// sites below read as they did; the decision itself is not duplicated.
bool CvodeSimulator::Impl::choose_use_sparse(const SolverOptions &opts, int ns) const {
    return route_to_sparse_linear_solver(model.jacobian_sparsity(), ns, opts.jacobian,
                                         opts.force_dense_linear_solver,
                                         opts.force_sparse_linear_solver);
}

// ─── SUNDIALS v7 setup ───────────────────────────────────────────────────────
//
// RAII guards handle all SUNDIALS cleanup automatically; run() owns them (its
// declaration order there is the teardown order), so they are filled in place.
void CvodeSimulator::Impl::create_cvode_core(const TimeSpec &times, const SolverOptions &opts,
                                             int ns, double rtol, double atol,
                                             const std::vector<double> &atol_v, int max_steps,
                                             SunContextGuard &ctx, NVectorGuard &y,
                                             CvodeMemGuard &cvode_mem, CvodeUserData &user_data,
                                             std::vector<double> &codegen_param_buf) {
    if (!ctx) {
        throw std::runtime_error("SUNContext_Create failed");
    }

    y = NVectorGuard(N_VNew_Serial(ns, ctx));
    if (!y) {
        throw std::runtime_error("N_VNew_Serial failed");
    }

    double *y_data = y.data();
    for (int i = 0; i < ns; ++i) {
        y_data[i] = model.species()[i].concentration;
    }

    cvode_mem = CvodeMemGuard(CVodeCreate(CV_BDF, ctx));
    if (!cvode_mem) {
        throw std::runtime_error("CVodeCreate failed");
    }

    // GH #106: size the rateOf probe scratch the event root function writes into
    // (only models that reference rateOf run that probe). Sized here, on the
    // cold/events path that owns the root function; the warm path has no roots.
    if (model.uses_rateof()) {
        user_data.rateof_root_scratch.assign(model.n_species(), 0.0);
    }

    // ─── Codegen RHS loading ────────────────────────────────────────────────
    // Resolve the (cached) codegen .so RHS / sens / Jacobian symbols and build
    // the per-run parameter mirror the codegen function reads from. Shared with
    // the warm path so the codegen ABI lives in one place (Impl::setup_codegen_
    // rhs). codegen_param_buf must outlive the integration loop in run() — the
    // user_data points into it.
    CVRhsFn rhs_fn = setup_codegen_rhs(opts, user_data, codegen_param_buf);

    int flag = CVodeInit(cvode_mem, rhs_fn, times.t_start, y);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeInit failed: " + std::to_string(flag));
    }
    // Fresh CVODE memory, so its own counters start at 0; the carried half of
    // the run's totals is an Impl member and has to be cleared (issue #182).
    closed_segments = SegmentCounters{};

    // Tracking (issue #213) is decided here rather than in run(), so the two
    // statements that have to agree — the spec the callback reads and the
    // CVodeWFtolerances that installs the callback — are one block. Depth 0
    // leaves both empty and the SS/SV dispatch below exactly as it was.
    user_data.atol_tracking = make_atol_tracking(rtol, atol_v, opts.atol_track_decades);
    apply_cvode_tolerances(cvode_mem, ctx, rtol, atol, atol_v, ns,
                           user_data.atol_tracking.active() ? cvode_tracking_ewt : nullptr);

    flag = CVodeSetUserData(cvode_mem, &user_data);
    flag = CVodeSetMaxNumSteps(cvode_mem, max_steps);

    if (opts.max_step_size > 0) {
        CVodeSetMaxStep(cvode_mem, opts.max_step_size);
    }
}

// ─── Validate Jacobian strategy ──────────────────────────────────────────────
void CvodeSimulator::Impl::validate_jacobian_option(const SolverOptions &opts,
                                                    CvodeUserData &user_data) {
    const std::string &jac_strategy = opts.jacobian;
    if (jac_strategy != "auto" && jac_strategy != "analytical" && jac_strategy != "fd" &&
        jac_strategy != "jax") {
        throw std::runtime_error("Invalid jacobian option '" + jac_strategy +
                                 "'. "
                                 "Must be \"auto\", \"analytical\", \"fd\", or \"jax\".");
    }

    // Copy the JAX callback from opts into user_data.
    if (jac_strategy == "jax") {
        if (!opts.jax_jac_fn) {
            throw std::runtime_error("jacobian=\"jax\" requested but no JAX callback was provided. "
                                     "Set opts.jax_jac_fn before calling run().");
        }
        user_data.jax_jac_fn = opts.jax_jac_fn;
    }

    // If user explicitly requests analytical but it's not available, fail fast.
    if (jac_strategy == "analytical") {
        if (!model.analytical_jacobian_complete()) {
            throw std::runtime_error(
                "jacobian=\"analytical\" requested but the analytical Jacobian is not "
                "available for this model. It covers Elementary mass-action rate laws and "
                "Functional rate laws whose derivatives could be symbolically derived; "
                "Michaelis-Menten and rate laws that fail symbolic differentiation fall "
                "back to finite differences.");
        }
    }
}

// ─── CVODES forward sensitivity setup ────────────────────────────────────────
//
// When opts.sensitivity.param_names is non-empty, initialize CVODES
// sensitivity analysis. CVODES computes dY/dp alongside the ODE integration
// using its internal finite-difference approximation of the sensitivity RHS.
// This works for ALL rate law types (Elementary, Functional, MichaelisMenten).
void CvodeSimulator::Impl::setup_forward_sensitivities(
    const TimeSpec &times, const SolverOptions &opts, int ns, double rtol, double atol,
    const std::vector<double> &atol_v, N_Vector y, void *cvode_mem, CvodeUserData &user_data,
    SensitivityState &sens) {
    sens.n_p = static_cast<int>(opts.sensitivity.param_names.size());
    sens.n_ic = static_cast<int>(opts.sensitivity.ic_species_names.size());
    sens.n_total = sens.n_p + sens.n_ic;

    const int n_sens_p = sens.n_p;
    const int n_sens_ic = sens.n_ic;
    const int n_sens = sens.n_total;
    if (n_sens == 0) {
        return;
    }

    // Aliases onto `sens`: every array below backs a raw pointer handed to
    // CVODES or to the codegen sensitivity RHS, so the storage has to outlive
    // this function. run() owns `sens` for the whole integration.
    NVectorArrayGuard &yS_guard = sens.yS;
    std::vector<int> &sens_param_indices = sens.param_indices;
    std::vector<int> &sens_ic_species_indices = sens.ic_species_indices;
    std::vector<double> &pbar = sens.pbar;
    std::vector<double> &sens_p = sens.p;
    std::vector<int> &sens_plist = sens.plist;
    std::vector<char> &sens_pin_mask = sens.pin_mask;
    std::vector<double> &sens_pin_nominal = sens.pin_nominal;
    int &sens_method = sens.method;

    double *y_data = N_VGetArrayPointer(y);
    int flag;

    const auto &params = model.parameters();

    // Resolve param sens names → indices.
    for (const auto &pname : opts.sensitivity.param_names) {
        bool found = false;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].name == pname) {
                sens_param_indices.push_back(static_cast<int>(i));
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Sensitivity parameter '" + pname +
                                     "' not found in model. "
                                     "Available: " +
                                     [&]() {
                                         std::string s;
                                         for (const auto &p : params) {
                                             if (!s.empty())
                                                 s += ", ";
                                             s += p.name;
                                         }
                                         return s;
                                     }());
        }
    }

    // Resolve IC species names → indices. IC sens uses the codegen sens
    // RHS path exclusively (CVODES internal FD has no parameter to
    // perturb, so the variational ODE source term ∂f/∂p ≡ 0 must be
    // produced analytically by bngsim_dfdp via a sentinel iP that hits
    // its `default: → zero` arm).
    if (n_sens_ic > 0 && !user_data.codegen_sens_fn) {
        throw std::runtime_error("sensitivity_ic requires codegen sensitivity RHS, but no "
                                 "codegen .so is loaded. Build the model with codegen enabled "
                                 "(or pass codegen=True / a net_path with mass-action kinetics).");
    }
    const auto &species = model.species();
    for (const auto &sname : opts.sensitivity.ic_species_names) {
        bool found = false;
        for (size_t i = 0; i < species.size(); ++i) {
            if (species[i].name == sname) {
                sens_ic_species_indices.push_back(static_cast<int>(i));
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Sensitivity IC species '" + sname + "' not found in model.");
        }
    }

    // Resolve every event trigger's residual now rather than at the first fire
    // (issue #144). It is the same answer either way — the residual is derived
    // from the trigger's own text — but this keeps every mutation of the
    // evaluator's expression table on the setup side of the integration, and it
    // is where a *clone* (whose residual cache starts empty, since expression
    // ids do not survive the move to another evaluator) resolves its own.
    for (int ei = 0; ei < model.n_events(); ++ei) {
        (void)model.event_trigger_residual_expr(ei);
    }

    // Build contiguous parameter array for CVODES. CVODES internal FD
    // perturbs p[plist[i]]; for codegen sens RHS, this array is just
    // mirrored into model params at each call so the RHS sees the
    // current state.
    sens_p.resize(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        sens_p[i] = params[i].value;
    }

    // plist[iS] = parameter index for column iS. For IC-sens columns we
    // use a sentinel ``params.size()`` (one past the end). The codegen
    // bngsim_dfdp(iP, ...) switch hits its ``default:`` arm and returns
    // dfdp=0, collapsing the variational ODE to ds/dt = J·s. CVODES does
    // not deref p[plist[iS]] when a user-supplied sens RHS is set, so
    // the sentinel is never read out of bounds.
    sens_plist.resize(n_sens);
    for (int i = 0; i < n_sens_p; ++i) {
        sens_plist[i] = sens_param_indices[i];
    }
    const int ic_plist_sentinel = static_cast<int>(params.size());
    for (int i = 0; i < n_sens_ic; ++i) {
        sens_plist[n_sens_p + i] = ic_plist_sentinel;
    }

    // pbar: |p| (or 1.0 if zero) for param cols; 1.0 for IC cols.
    pbar.resize(n_sens);
    for (int i = 0; i < n_sens_p; ++i) {
        double val = params[sens_param_indices[i]].value;
        pbar[i] = (val != 0.0) ? std::abs(val) : 1.0;
    }
    for (int i = 0; i < n_sens_ic; ++i) {
        pbar[n_sens_p + i] = 1.0;
    }

    // ── Pre-equilibration / carry-over seeding decision (GH #210) ────────
    // In a two-phase pre-equilibration (ADR-0052) the species state is
    // carried over from the equilibration phase with no reset, so the
    // measurement phase's IC is x_ss(θ) and ∂y(0)/∂θ = dx_ss/dθ — NOT the
    // fresh-start seed. We seed yS(0) from the prior phase's captured
    // dx/dθ when carry_sensitivities is set, and otherwise refuse loudly
    // rather than return silently-wrong derivatives on a dirty state.
    const bool state_dirty = model.ic_state_dirty();
    bool use_carry_seed = false;
    if (n_sens_p > 0) {
        if (opts.carry_sensitivities) {
            // Opt-in carry-over: require a pending seed whose columns match
            // the requested parameters (same names, same order).
            const auto &seed = model.pending_sens_seed();
            const auto &seed_names = model.pending_sens_seed_param_names();
            bool names_match = seed_names.size() == opts.sensitivity.param_names.size();
            for (size_t i = 0; names_match && i < seed_names.size(); ++i) {
                names_match = (seed_names[i] == opts.sensitivity.param_names[i]);
            }
            if (seed.empty() || !names_match || seed.size() != static_cast<size_t>(ns) * n_sens_p) {
                throw std::runtime_error(
                    "carry_sensitivities=True, but no matching forward-sensitivity "
                    "seed from a prior phase is available. Run the equilibration "
                    "phase on the same Simulator with the same sensitivity_params "
                    "(and no reset between phases) before the measurement phase. "
                    "(pre-equilibration output sensitivities, GH #210)");
            }
            use_carry_seed = true;
        } else if (state_dirty) {
            // Sensitivities on a carried-over / manually-advanced state
            // without opt-in would seed yS(0) as if starting from the ICs —
            // silently wrong (∂y(0)/∂θ is dx/dθ of the carried state, not 0).
            throw std::runtime_error(
                "Output sensitivities were requested on a carried-over species "
                "state (the model was advanced by a previous run() or set "
                "manually, with no reset since). Seeding the measurement phase "
                "as a fresh start would give silently wrong derivatives across "
                "the pre-equilibration boundary. Pass carry_sensitivities=True "
                "to seed from the prior phase's steady-state sensitivity, or "
                "reset() the model for a fresh start. (GH #210)");
        }
    }
    // IC (∂y/∂y_k(0)) sensitivities across a carry-over boundary are out of
    // scope: the carried reference state is no longer the model ICs, so e_k
    // is not a meaningful seed. Refuse rather than return a wrong matrix.
    if (n_sens_ic > 0 && state_dirty) {
        throw std::runtime_error("sensitivity_ic (initial-condition sensitivities) across a "
                                 "carried-over / pre-equilibration boundary is not supported: the "
                                 "carried state is no longer the model's initial condition. "
                                 "reset() for a fresh start. (GH #210)");
    }

    // Allocate sensitivity vectors and seed s(0) = ∂y(0)/∂θ.
    //   • Carry-over (use_carry_seed): param cols are seeded from the prior
    //     phase's dx/dθ; the IC-parameter identity is NOT applied (the
    //     carried state is not at the ICs, and the seed already integrated
    //     any IC-parameter dependence through the equilibration phase).
    //   • Fresh start: for param-sens cols whose param sets a species's IC
    //     directly (recorded in species_ic_param_refs by the .net loader),
    //     seed yS[iS][species_idx] = 1. Other param cols seed to zero.
    //   • For IC-sens cols, seed yS[iS][species_idx] = 1 unconditionally.
    yS_guard = NVectorArrayGuard(N_VCloneVectorArray(n_sens, y), n_sens);
    if (!yS_guard) {
        throw std::runtime_error("N_VCloneVectorArray failed for sensitivities");
    }
    for (int i = 0; i < n_sens; ++i) {
        N_VConst(0.0, yS_guard[i]);
    }
    if (use_carry_seed) {
        const auto &seed = model.pending_sens_seed(); // row-major [species*np + param]
        for (int iS = 0; iS < n_sens_p; ++iS) {
            double *col = N_VGetArrayPointer(yS_guard[iS]);
            for (int i = 0; i < ns; ++i) {
                col[i] = seed[static_cast<size_t>(i) * n_sens_p + iS];
            }
        }
    } else if (n_sens_p > 0) {
        std::unordered_map<int, int> param_to_sens_idx;
        param_to_sens_idx.reserve(static_cast<size_t>(n_sens_p));
        for (int iS = 0; iS < n_sens_p; ++iS) {
            param_to_sens_idx.emplace(sens_param_indices[iS], iS);
        }
        const auto &ic_param_sens = opts.sensitivity.ic_param_sens;
        if (!ic_param_sens.empty()) {
            // Issue #43: Python-computed ∂x_i(0)/∂p seeds. These cover BOTH
            // direct-parameter ICs (coefficient 1) and derived-parameter ICs
            // (Rtot = R0, Rtot = 2*R0, …) whose chain-rule partial the C++
            // seeding cannot compute, so when supplied they replace the
            // legacy identity-only loop entirely. `+=` accumulates in case a
            // species IC depends on the same requested primary through more
            // than one path.
            for (const auto &seed : ic_param_sens) {
                auto it = param_to_sens_idx.find(seed.primary_param_idx0);
                if (it == param_to_sens_idx.end()) {
                    continue; // primary not requested for sensitivity
                }
                const int iS = it->second;
                if (seed.species_idx0 < 0 || seed.species_idx0 >= ns) {
                    continue;
                }
                N_VGetArrayPointer(yS_guard[iS])[seed.species_idx0] += seed.d_ic_d_primary;
            }
        } else {
            // Legacy fallback (no Python injection, e.g. sympy unavailable):
            // a species IC that names a requested primary directly seeds
            // yS_species(0) = 1. Derived-parameter ICs stay unseeded here —
            // the pre-#43 behavior — but direct ICs remain correct.
            const auto &species_vec = model.species();
            for (const auto &ref : model.species_ic_param_refs()) {
                const int species_idx0 = ref.first;
                const int param_idx0 = ref.second;
                auto it = param_to_sens_idx.find(param_idx0);
                if (it == param_to_sens_idx.end()) {
                    continue; // parameter not requested for sensitivity
                }
                const int iS = it->second;
                if (species_idx0 < 0 || species_idx0 >= ns) {
                    continue;
                }
                // GH #113: the IC expression describes `initial_conc`. Once an
                // assignment has moved this species off that baseline, the
                // parameter no longer reaches its initial condition and the
                // identity seed would report a gradient through an IC the model
                // no longer has. (The Python injection path above applies the
                // same rule, plus issue #111's explicit declarations.)
                const auto &sp = species_vec[static_cast<std::size_t>(species_idx0)];
                if (sp.concentration != sp.initial_conc) {
                    continue;
                }
                N_VGetArrayPointer(yS_guard[iS])[species_idx0] = 1.0;
            }
        }
    }
    for (int k = 0; k < n_sens_ic; ++k) {
        const int species_idx0 = sens_ic_species_indices[k];
        if (species_idx0 < 0 || species_idx0 >= ns) {
            continue;
        }
        N_VGetArrayPointer(yS_guard[n_sens_p + k])[species_idx0] = 1.0;
    }

    // Initialize sensitivity analysis.
    if (opts.sensitivity.method == "simultaneous") {
        sens_method = CV_SIMULTANEOUS;
    }

    CVSensRhs1Fn sens_rhs_fn = nullptr;
    if (user_data.codegen_sens_fn) {
        user_data.codegen_plist = sens_plist.data();
        user_data.codegen_n_sens = n_sens;
        sens_rhs_fn = cvode_codegen_sens_rhs;
    }

    flag = CVodeSensInit1(cvode_mem, n_sens, sens_method, sens_rhs_fn, yS_guard.arr);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSensInit1 failed: " + std::to_string(flag));
    }

    // ── Scale-aware sensitivity error control (GH #214) ──────────────────
    // bngsim integrates concentrations (= amount / V_compartment). For the
    // sub-picoliter compartments of real cell-biology models, 1/V reaches
    // ~1e11–1e14, inflating both the state and its sensitivities by that
    // factor (Smith2013: |s|~1e18 vs AMICI's amount-based ~1e10). The default
    // CVodeSensEEtolerances derives a single scalar absolute floor per column
    // (atolS[iS] = atol / pbar[iS]); against a 1e18-magnitude sensitivity that
    // floor is ~30 orders below the variable, so the CVODES error test demands
    // sub-machine-eps relative accuracy and the step collapses across a large
    // discontinuity (Smith's t=2880 insulin restimulation → flag -3).
    //
    // Instead set a per-(state × parameter) absolute floor proportional to each
    // sensitivity's own natural magnitude scale[i]/pbar[iS], where scale[i] is a
    // characteristic size of state i (its initial magnitude, floored at 1). This
    // is the non-dimensionalizing move: error control becomes relative-per-
    // component regardless of the unit system. For a well-scaled model (every
    // state ≤ 1 ⇒ scale[i]=1) it reduces EXACTLY to the EE floor atol/pbar[iS],
    // so well-scaled models stay byte-identical; only large-magnitude states get
    // a proportionally relaxed (reachable) floor. rtol still governs the
    // relative accuracy uniformly. (CVODES clones these vectors internally, so
    // the guard's lifetime is not load-bearing.)
    //
    // Issue #196: when the state axis carries a per-species atol, each column's
    // floor is built from ROW i's tolerance rather than from the one scalar.
    // The structure is untouched — atol_i·scale[i]/pbar[iS] — so a constant
    // vector reproduces this atolS entry for entry, and a caller who asked for
    // a decade-spanning state tolerance gets sensitivities held to the same
    // per-species statement instead of having it collapsed back onto one
    // number here.
    //
    // Issue #213: when the state axis is TRACKING, atol_v is the ceiling of
    // that rule, and this block reads the ceiling — deliberately, not by
    // inheritance. Turning tracking on therefore leaves the sensitivity
    // tolerances exactly where the same vector would have put them without it.
    // The alternative, re-deriving atolS from the live state at each refresh,
    // would make this base TIGHTEN mid-run, and that is the hazard the issue
    // #183 block below had to add a high-water mark to avoid: CVODES then has
    // to re-pass an error test it already passed, at a step size chosen under
    // the looser rule. It would also only be reachable on models that armed the
    // #177 refresh at all, so it would be a mode that silently applied to some
    // models and not others. The state axis is what tracking moves.
    std::vector<double> sens_state_scale(static_cast<size_t>(ns));
    for (int i = 0; i < ns; ++i) {
        sens_state_scale[i] = std::max(std::abs(y_data[i]), 1.0);
    }
    const bool per_species_atol = !atol_v.empty();
    sens.abstolS = NVectorArrayGuard(N_VCloneVectorArray(n_sens, y), n_sens);
    if (!sens.abstolS) {
        throw std::runtime_error("N_VCloneVectorArray failed for sensitivity tolerances");
    }
    sens.atolS_base.resize(static_cast<size_t>(n_sens) * static_cast<size_t>(ns));
    for (int iS = 0; iS < n_sens; ++iS) {
        double *atolS_col = N_VGetArrayPointer(sens.abstolS[iS]);
        const double pb = (pbar[iS] != 0.0) ? pbar[iS] : 1.0;
        for (int i = 0; i < ns; ++i) {
            const double atol_i = per_species_atol ? atol_v[static_cast<size_t>(i)] : atol;
            const double a = atol_i * sens_state_scale[i] / pb;
            sens.atolS_base[static_cast<size_t>(iS) * ns + i] = a;
            atolS_col[i] = a;
        }
    }
    flag = CVodeSensSVtolerances(cvode_mem, rtol, sens.abstolS.arr);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSensSVtolerances failed: " + std::to_string(flag));
    }

    // ── The floor under that floor (issue #177) ──────────────────────────
    // Everything above is a statement about the *magnitude* a sensitivity is
    // expected to have. It says nothing about the accuracy the arithmetic can
    // deliver, and on a model whose ∂f/∂p is a difference of large fluxes those
    // are wildly different numbers: the tolerance asks for 1e-8 of a quantity
    // computed to ±4e2, and CVODES shrinks h forever chasing it. Arm the
    // refresh that lifts atolS to that roundoff wherever it sits above the
    // static floor. Never below: this is a max(), so a model whose arithmetic
    // is clean keeps exactly the tolerances it had.
    setup_sens_error_floor(opts, ns, rtol, times.t_end - times.t_start, user_data, sens);
    refresh_sens_error_floor(cvode_mem, times.t_start, y, user_data, sens, ns);

    flag = CVodeSetSensErrCon(cvode_mem, opts.sensitivity.error_control);

    flag = CVodeSetSensParams(cvode_mem, sens_p.data(), pbar.data(), sens_plist.data());
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSetSensParams failed: " + std::to_string(flag));
    }

    user_data.sens_p = sens_p.data();
    user_data.n_params = static_cast<int>(params.size());

    // Pin switch-time parameters against the internal-FD probe (issue #48).
    // Only populated when a fitted switch time was detected, so every other
    // model keeps CVODES' probe exactly as it was.
    if (!opts.sensitivity.switch_pinned_params.empty()) {
        sens_pin_mask.assign(params.size(), 0);
        sens_pin_nominal.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            sens_pin_nominal[i] = params[i].value;
        }
        for (int pidx : opts.sensitivity.switch_pinned_params) {
            if (pidx >= 0 && pidx < static_cast<int>(params.size())) {
                sens_pin_mask[static_cast<size_t>(pidx)] = 1;
            }
        }
        user_data.sens_param_pinned = sens_pin_mask.data();
        user_data.sens_param_nominal = sens_pin_nominal.data();
    }
}

// ─── Sensitivity error floor (issue #177) ────────────────────────────────────
//
// An absolute tolerance is a statement about what counts as negligible. The GH
// #214 floor above states it in units of the *variable*: atolS[iS][i] =
// atol·scale[i]/pbar[iS], where scale[i] is a characteristic size of state i.
// That is the right question for a quantity you can compute to full precision.
//
// A sensitivity is not always such a quantity. For column iS the variational
// equation is ṡ = J·s + ∂f/∂p_iS, so row i's derivative is *assembled* by
// summing terms, and in a model whose species span many orders those terms
// cancel: ∂f/∂p can be 1e18 − 1e18, reported as ~0 while carrying ~ε·2e18 of
// roundoff. Nothing in the reported value records that. A row whose own |s_i|
// has decayed to zero contributes nothing to rtol·|s_i| either, so atolS is the
// only thing holding the error weight finite — and set below that roundoff, the
// error test can never pass at any step size. CVODES shrinks h without bound,
// chasing accuracy float64 does not have. (Issue #177: 183,219 steps for a
// two-reaction model whose scalar solve takes 210, and a 0.013 s run that does
// not finish in 300 s once 16 sensitivity columns are on.)
//
// So floor atolS at that roundoff — ε · τ · Σ|term|, per row and per column:
//
//   Σ|term|_i  the magnitudes summed into row i. The emitted
//              bngsim_codegen_sens_term_scale reports the ∂f/∂p half (a value
//              says nothing about what cancelled to produce it, so this has to
//              come from the emitter, not from re-reading ∂f/∂p); Σ_j|J_ij||s_j|
//              is the J·s half, formed here from the analytical Jacobian.
//   τ          a time scale, because Σ|term| has the units of ṡ and a tolerance
//              has the units of s. It is exactly the smallest step that RHS
//              noise alone may force: a step of size h admits ~h·(noise in ṡ) of
//              error in s, so demanding ε·τ·Σ|term| ≥ that is the statement
//              "noise may not push h below τ". A genuine accuracy requirement
//              still shrinks h as far as it likes. Set to a small fraction of
//              the integration horizon (see kSensFloorTauFraction), which is the
//              only problem time scale available before the first step.
//
// Three things this shape gets right that a global static floor did not, each
// having killed an earlier attempt on the corpus:
//
//   * It is PER ROW. A norm over the whole column overstates a small row's
//     noise by however far the column spans — seven orders on BIOMD0000000072,
//     which is what turned a 3.0e-7 median FD error into 2.4e-3 there.
//   * It carries NO 1/pbar. ∂f/∂p is already in the units of s, so the division
//     that inflated a 3.32e-18-valued parameter's floor to 40% of its own
//     column's peak has nothing to divide.
//   * It is a max() against the existing floor, and Σ|term| is ~ε·(a few) for a
//     well-scaled model. Every model whose arithmetic is clean keeps exactly the
//     tolerances — and the step sequence — it had.
// The unit roundoff of the accumulation. One ε, not a safety multiple of it:
// the floor is deliberately the smallest defensible one, so that anywhere it
// binds it binds because the arithmetic genuinely cannot do better.
static constexpr double kSensFloorEps = std::numeric_limits<double>::epsilon();

// τ = this × the integration horizon. Read as: RHS noise alone may not push the
// step below a thousandth of the horizon. Picked by measurement, not by taste —
// see dev/issue177 and the issue thread for the table it came from. Larger fixes
// more and resolves less; smaller is inert.
static constexpr double kSensFloorTauFraction = 1e-3;

// Hysteresis on the issue #183 relaxation: a row is declared unresolvable at
// rtol·|s_i| < ε‖s‖∞ and released only once rtol·|s_i| climbs this far ABOVE
// ε‖s‖∞. Two failures, one on each side, fix the shape:
//
//   * Released as soon as the strict test flips, the floor drops eight orders in
//     one refresh — CVODES then has to re-pass, at a step size chosen under the
//     looser rule, an error test it already passed. Smith's columns then worked
//     at one ladder spacing and failed at both a coarser and a finer one, which
//     is a lucky refresh time rather than a rule.
//   * Never released at all, the relaxation outlives the condition. On the #55
//     SIR model that cost the analytic run 1436 steps against the <1000 its
//     step-count test pins: a row relaxed early stays relaxed while it grows
//     back into significance, and the error it is then allowed to carry
//     propagates through J·s into rows that ARE controlled.
//
// A wide band, because the two regimes are ten orders apart and the gap is where
// the hysteresis is free.
static constexpr double kSensUnresolvableRelease = 1e6;

void CvodeSimulator::Impl::setup_sens_error_floor(const SolverOptions &opts, int ns, double rtol,
                                                  double horizon, CvodeUserData &user_data,
                                                  SensitivityState &sens) {
    sens.floor_active = false;
    if (!opts.sensitivity.error_control) {
        return; // nothing to floor: the error test is off
    }
    // BNGSIM_SENS_ERROR_FLOOR=0 restores the pre-#177 tolerances from the same
    // binary and the same .so — which is what makes the corpus A/B for this
    // change a one-variable experiment rather than a two-build comparison.
    const char *hatch = std::getenv("BNGSIM_SENS_ERROR_FLOOR");
    if (hatch && std::string(hatch) == "0") {
        return;
    }
    double frac = kSensFloorTauFraction;
    if (const char *fenv = std::getenv("BNGSIM_SENS_ERROR_FLOOR_TAU")) {
        const double v = std::atof(fenv);
        if (v > 0.0) {
            frac = v;
        }
    }
    sens.floor_tau = frac * std::abs(horizon);
    sens.floor_rtol = rtol;
    if (!(sens.floor_tau > 0.0)) {
        return; // a zero-length span has no time scale to state a floor in
    }

    // Which noise sources are in play. One knob per source, because they are
    // independent statements about the arithmetic and choosing between them is a
    // measurement rather than a preference: "terms" is the assembly floor (the
    // emitted Σ|term| for ∂f/∂p plus Σ_j|J_ij||s_j| for J·s), "colnorm" is the
    // column's representation floor.
    bool want_terms = true;
    sens.floor_do_col_norm = true;
    if (const char *parts = std::getenv("BNGSIM_SENS_FLOOR_PARTS")) {
        const std::string p(parts);
        want_terms = p.find("terms") != std::string::npos;
        sens.floor_do_col_norm = p.find("colnorm") != std::string::npos;
    }

    // The ∂f/∂p half of the assembly floor needs the emitted companion; the J·s
    // half needs an analytical Jacobian. Either alone is a usable
    // (under-)estimate — every term this cannot see only makes the floor
    // smaller, i.e. closer to the behaviour that shipped.
    const auto &spat = model.jacobian_sparsity();
    sens.floor_do_dfdp_terms = want_terms && user_data.codegen_sens_term_scale_fn != nullptr;
    sens.floor_do_jac_terms =
        want_terms && model.analytical_jacobian_complete() && !spat.empty() && spat.nnz > 0;
    if (!sens.floor_do_dfdp_terms && !sens.floor_do_jac_terms && !sens.floor_do_col_norm) {
        return;
    }
    sens.floor_terms.assign(static_cast<size_t>(ns), 0.0);
    sens.floor_unresolvable.assign(static_cast<size_t>(sens.n_total) * static_cast<size_t>(ns),
                                   0.0);
    sens.floor_jac.assign(sens.floor_do_jac_terms ? static_cast<size_t>(spat.nnz) : 0, 0.0);
    // BNGSIM_SENS_FLOOR_UNRESOLVABLE=0 restores the pre-#183 tolerances from the
    // same binary, which is what makes this change's corpus A/B a one-variable
    // experiment rather than a two-build comparison (as BNGSIM_SENS_ERROR_FLOOR=0
    // is for #177 and BNGSIM_SENS_FLOOR_LADDER=off for the refresh ladder).
    if (const char *ur = std::getenv("BNGSIM_SENS_FLOOR_UNRESOLVABLE")) {
        sens.floor_do_unresolvable = std::string(ur) != "0";
    }
    if (const char *uc = std::getenv("BNGSIM_SENS_FLOOR_UNRESOLVABLE_CAP")) {
        const double v = std::atof(uc);
        if (v > 0.0) {
            sens.floor_unresolvable_cap = v;
        }
    }
    sens.floor_active = true;
}

void CvodeSimulator::Impl::refresh_sens_error_floor(void *cvode_mem, double t, N_Vector y,
                                                    CvodeUserData &user_data,
                                                    SensitivityState &sens, int ns) {
    if (!sens.floor_active) {
        return;
    }
    const int n_sens = sens.n_total;
    double *y_data = N_VGetArrayPointer(y);

    // |J| once per refresh, shared by every column. The compiled mirror when
    // there is one, so the Jacobian read here is the Jacobian the step is using.
    const bool want_jac = sens.floor_do_jac_terms;
    const auto &spat = model.jacobian_sparsity();
    if (want_jac) {
        if (user_data.codegen_jac_sparse_fn) {
            user_data.codegen_jac_sparse_fn(t, y_data, sens.floor_jac.data(),
                                            &user_data.codegen_so_data);
        } else {
            model.fill_sparse_analytical_jacobian(t, y_data, sens.floor_jac.data());
        }
    }

    CodegenSensUserDataForSO so_data;
    so_data.param_values = user_data.codegen_param_values;
    so_data.plist = user_data.codegen_plist;
    so_data.n_sens = user_data.codegen_n_sens;
    const bool want_terms = sens.floor_do_dfdp_terms && so_data.plist != nullptr;

    bool moved = false;
    for (int iS = 0; iS < n_sens; ++iS) {
        double *atolS_col = N_VGetArrayPointer(sens.abstolS[iS]);
        const double *base = sens.atolS_base.data() + static_cast<size_t>(iS) * ns;

        std::fill(sens.floor_terms.begin(), sens.floor_terms.end(), 0.0);
        if (want_terms) {
            // An IC column's plist entry is the one-past-the-end sentinel, so
            // this hits the emitted switch's default arm and returns zeros —
            // correct, an IC column has no ∂f/∂p term at all.
            user_data.codegen_sens_term_scale_fn(n_sens, t, y_data, iS, sens.floor_terms.data(),
                                                 &so_data);
        }
        const double *s = N_VGetArrayPointer(sens.yS[iS]);
        if (want_jac) {
            // Σ_j |J_ij|·|s_j| over the CSC pattern: one O(nnz) pass, no
            // cancellation, which is the whole point — the signed J·s is
            // already what CVODES computes.
            for (int j = 0; j < ns; ++j) {
                const double sj = std::abs(s[j]);
                if (!(sj > 0.0)) {
                    continue;
                }
                for (int64_t k = spat.col_ptrs[static_cast<size_t>(j)];
                     k < spat.col_ptrs[static_cast<size_t>(j) + 1]; ++k) {
                    sens.floor_terms[static_cast<size_t>(
                        spat.row_indices[static_cast<size_t>(k)])] +=
                        std::abs(sens.floor_jac[static_cast<size_t>(k)]) * sj;
                }
            }
        }

        // The column's own representation floor. Everything above is about the
        // arithmetic that *forms* ṡ; this is about the arithmetic that carries
        // s. Each BDF step solves (I − γJ)Δ = r for the whole column at once, so
        // every entry of the result is assembled from quantities of size
        // ‖s‖∞ and inherits ~ε‖s‖∞ of absolute error whatever its own size. A
        // row asked for accuracy below that is being asked for digits the column
        // does not carry. Unlike the term scale this is in the units of s
        // already, so it takes no τ — and it is zero at t=0, where the term
        // scale is what covers the run.
        double col_norm = 0.0;
        if (sens.floor_do_col_norm) {
            for (int i = 0; i < ns; ++i) {
                col_norm = std::max(col_norm, std::abs(s[i]));
            }
        }
        const double col_floor = kSensFloorEps * col_norm;

        for (int i = 0; i < ns; ++i) {
            double noise =
                std::max(kSensFloorEps * sens.floor_tau * sens.floor_terms[i], col_floor);
            // ── Rows the column cannot resolve to rtol (issue #183) ──────────
            //
            // col_floor = ε‖s‖∞ is the absolute noise every entry of the column
            // carries. A row asked for rtol·|s_i| tighter than that is being
            // asked for digits the column does not have, and no step size
            // supplies them: rtol·|s_i| < ε‖s‖∞ says exactly "row i's relative
            // band is finer than the column's own noise". #177 floored atolS at
            // the roundoff of the sum that FORMS ṡ; this is the roundoff of the
            // column that CARRIES s, and neither implies the other.
            //
            // The floor to state there is |s_i| itself — the row is not error-
            // controlled beyond its own size, which is the honest reading of a
            // value that is noise. Measured, not chosen: Smith's k7 needed
            // atolS ≈ 1.6·|s_i| on the binding row and k8 ≈ 0.73·|s_i|, from
            // columns whose norms differ by seven orders.
            //
            // The test separates the two cases by ten orders of magnitude, so it
            // is nothing like a knife edge: at Smith's t≈19.3 the binding row
            // sits 36 ulps of ‖s‖∞ above zero, while sens_scale_cancellation's
            // one live row sits 4.5e15 of them — the whole of float64. A row
            // that carries its column is never touched, which is what keeps
            // #177's accuracy test green.
            //
            // Self-correcting, too: Akt_P2's sensitivity climbs 13 orders out of
            // this regime between t=19 and t=24, and the next refresh finds it
            // resolvable again and stops relaxing it.
            //
            // Held as a high-water mark, per (row, column). A floor that TIGHTENS
            // mid-run is a step-controller hazard in its own right — CVODES has
            // to re-pass an error test it already passed, at a step size chosen
            // under the looser rule — and this rule is not naturally monotone:
            // Akt_P2 leaves the unresolvable regime by growing 13 orders, which
            // without the mark drops its floor eight orders in one refresh. That
            // showed up as the fix working at one ladder spacing and failing at
            // both a coarser and a finer one, which is the signature of a lucky
            // refresh time rather than a rule.
            const double s_i = std::abs(s[i]);
            if (sens.floor_do_unresolvable) {
                double &mark = sens.floor_unresolvable[static_cast<size_t>(iS) * ns + i];
                const double band = sens.floor_rtol * s_i;
                if (band < col_floor) {
                    // Capped at kSensUnresolvableCap ulps of the column. Without
                    // the cap the rule reads "fewer than 1/rtol digits" and hands
                    // every such row a 100% relative tolerance — an eight-order
                    // window, so a row still carrying six good digits is released
                    // from error control entirely. The rows this is for are far
                    // deeper than that: Smith's binding row sits 36 ulps of ‖s‖∞
                    // above zero, so the cap does not bind where it matters and
                    // does bind everywhere it should.
                    mark = std::max(mark, std::min(s_i, sens.floor_unresolvable_cap * col_floor));
                } else if (band > kSensUnresolvableRelease * col_floor) {
                    mark = 0.0; // resolvable again, by a margin
                }
                noise = std::max(noise, mark);
            }
            // A non-finite state makes a non-finite term scale; keep the static
            // floor there rather than handing CVODES a NaN tolerance, which it
            // would silently propagate into every error weight.
            const double want = (std::isfinite(noise) && noise > base[i]) ? noise : base[i];
            moved = moved || (want != atolS_col[i]);
            if (base[i] > 0.0) {
                sens.floor_max_relax = std::max(sens.floor_max_relax, want / base[i]);
            }
            atolS_col[i] = want;
        }
    }

    ++sens.floor_refreshes;
    // BNGSIM_SENS_FLOOR_DEBUG traces the floor against the step size, which is
    // the pair that says whether a refresh arrived in time: raising a tolerance
    // after CVODES has crawled to h~1e-13 does not recover the step controller.
    // ``lo`` is over rows with a nonzero s — a row whose sensitivity is exactly
    // zero contributes exactly zero to the WRMS norm however small its tolerance
    // is, so the unrestricted minimum is a decoy (it cost real time to learn).
    if (std::getenv("BNGSIM_SENS_FLOOR_DEBUG")) {
        long nst = 0, netf = 0, ncfn = 0, snetf = 0, sncfn = 0;
        double h = 0.0;
        CVodeGetNumSteps(cvode_mem, &nst);
        CVodeGetLastStep(cvode_mem, &h);
        CVodeGetNumErrTestFails(cvode_mem, &netf);
        CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
        CVodeGetSensNumErrTestFails(cvode_mem, &snetf);
        CVodeGetSensNumNonlinSolvConvFails(cvode_mem, &sncfn);
        double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
        for (int iS = 0; iS < n_sens; ++iS) {
            const double *c = N_VGetArrayPointer(sens.abstolS[iS]);
            const double *sv = N_VGetArrayPointer(sens.yS[iS]);
            for (int i = 0; i < ns; ++i) {
                if (sv[i] != 0.0) {
                    lo = std::min(lo, c[i]);
                }
                hi = std::max(hi, c[i]);
            }
        }
        std::fprintf(stderr,
                     "[sens-floor] #%d t=%g nst=%ld h=%.3e tau=%.3e live atolS in [%.3e, %.3e] "
                     "max relax=%.3e moved=%d etf=%ld/%ld cf=%ld/%ld\n",
                     sens.floor_refreshes, t, nst, h, sens.floor_tau, lo, hi, sens.floor_max_relax,
                     static_cast<int>(moved), snetf, netf, sncfn, ncfn);
    }
    if (!moved) {
        return; // the tolerances CVODES already holds are these tolerances
    }
    const int flag = CVodeSensSVtolerances(cvode_mem, sens.floor_rtol, sens.abstolS.arr);
    if (flag != CV_SUCCESS) {
        // A refresh is an optimization, not a correctness requirement: the
        // tolerances already installed remain in force. Disarm rather than
        // abort a run that is otherwise proceeding.
        sens.floor_active = false;
    }
}

// ─── Event rootfinding registration ──────────────────────────────────────────
void CvodeSimulator::Impl::register_roots(void *cvode_mem, SUNContext ctx, int n_roots,
                                          int n_disc) {
    // Root function callback: see cvode_event_root_fn.
    int flag = CVodeRootInit(cvode_mem, n_roots, cvode_event_root_fn);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeRootInit failed: " + std::to_string(flag));
    }

    if (n_disc > 0) {
        // A discontinuity root makes CVODE restart at each pulse edge,
        // where its first post-reinit step can be so small that t+h==t in
        // floating point — a benign SUNDIALS warning ("solver will continue
        // anyway") that RoadRunner emits on the same models. Route THIS
        // context's warning log to the null sink so a dosing-schedule model
        // doesn't spam stdout. Scoped to n_disc>0: models without time
        // piecewise keep their exact prior logging. Hard errors still throw
        // via the flag<0 checks. The context (and this logger) is freed when
        // the run's SunContextGuard goes out of scope.
        SUNLogger logger = nullptr;
        if (SUNContext_GetLogger(ctx, &logger) == SUN_SUCCESS && logger != nullptr) {
            SUNLogger_SetWarningFilename(logger, bngsim::null_device);
        }
    }
}

// ─── Allocate result ─────────────────────────────────────────────────────────
void CvodeSimulator::Impl::allocate_run_result(Result &result, const SolverOptions &opts, int n_out,
                                               int n_sens_p, int n_sens_ic) {
    const int ns = model.n_species();
    const int n_obs = model.n_observables();
    const int n_func = model.n_functions();

    result.allocate(n_out, ns, n_obs);
    result.set_species_names(model.species_names());
    // GH #71: project trajectory columns to reported species only when the
    // model has unreported state (an event-mutated parameter/compartment
    // promoted to a species). All-reported models leave the projection empty,
    // so the output column set stays byte-identical.
    {
        auto reported = model.reported_species_indices();
        if (reported.size() != static_cast<std::size_t>(model.n_species())) {
            result.set_reported_species_indices(std::move(reported));
        }
    }
    result.set_observable_names(model.observable_names());
    if (n_func > 0) {
        result.set_expression_names(model.function_names());
    }

    // Allocate sensitivity storage in the result. Param sens and IC sens
    // are stored on separate axes (Result.sensitivity_data vs
    // Result.sensitivity_ic_data) so callers can address each
    // independently without slot-collision risk.
    if (n_sens_p > 0) {
        result.allocate_sensitivities(n_out, ns, n_sens_p);
        result.set_sens_param_names(opts.sensitivity.param_names);
    }
    if (n_sens_ic > 0) {
        result.allocate_sensitivities_ic(n_out, ns, n_sens_ic);
        result.set_sens_ic_species_names(opts.sensitivity.ic_species_names);
    }
}

// ─── Solver statistics ───────────────────────────────────────────────────────
CvodeSimulator::Impl::SegmentCounters CvodeSimulator::Impl::read_segment_counters(void *cvode_mem) {
    SegmentCounters c;
    CVodeGetNumSteps(cvode_mem, &c.steps);
    CVodeGetNumRhsEvals(cvode_mem, &c.rhs_evals);
    CVodeGetNumLinSolvSetups(cvode_mem, &c.lin_solv_setups);
    CVodeGetNumNonlinSolvIters(cvode_mem, &c.nonlin_iters);
    CVodeGetNumNonlinSolvConvFails(cvode_mem, &c.nonlin_conv_fails);
    CVodeGetNumErrTestFails(cvode_mem, &c.err_test_fails);
    return c;
}

int CvodeSimulator::Impl::reinit_cvode(void *cvode_mem, sunrealtype t, N_Vector y) {
    // Bank what this segment cost before CVodeReInit zeroes every counter
    // (cvodes.c "Initialize all the counters"), so the run's totals survive the
    // restart — issue #182.
    closed_segments += read_segment_counters(cvode_mem);
    return CVodeReInit(cvode_mem, t, y);
}

void CvodeSimulator::Impl::record_solver_stats(void *cvode_mem, SUNLinearSolver ls,
                                               Result &result) {
    // The still-open segment plus every segment a mid-run re-init closed
    // (issue #182). Without the carried half, a model with events reports only
    // the tail after its last fire.
    SegmentCounters total = closed_segments;
    total += read_segment_counters(cvode_mem);

    result.solver_stats().n_steps = static_cast<int>(total.steps);
    result.solver_stats().n_rhs_evals = static_cast<int>(total.rhs_evals);
    result.solver_stats().n_jac_evals = static_cast<int>(total.lin_solv_setups);
    result.solver_stats().n_nonlin_iters = static_cast<int>(total.nonlin_iters);
    result.solver_stats().n_nonlin_conv_fails = static_cast<int>(total.nonlin_conv_fails);
    result.solver_stats().n_err_test_fails = static_cast<int>(total.err_test_fails);
    result.solver_stats().linear_solver = linear_solver_used;
    // GH #132: BLAS dgetrf factorization count for this run (0 unless LAPACK-dense
    // and the adaptive K gate was crossed). `ls` is the caller's linear solver —
    // run()'s LS_guard, or the warm cache's persistent w.LS.
    {
        const long bc = lapack_dense_blas_factor_count(ls);
        result.solver_stats().n_dense_blas_factorizations = bc > 0 ? static_cast<int>(bc) : 0;
    }
}

// ─── Write final state back to model ─────────────────────────────────────────
// After simulation, update the model's species concentrations to the
// final state. This is essential for multi-action sequences where
// saveConcentrations() or subsequent simulate() actions depend on
// the post-simulation state (matching BNG's propagate_cvode_network
// behavior which writes back to the global species array). When the
// steady-state early-stop fired, the "final time" the caller passes is the
// last sample we actually integrated to, not the originally requested
// ``t_end``.
void CvodeSimulator::Impl::write_final_state_back(const SolverOptions &opts, int ns,
                                                  const double *y_data, double final_t,
                                                  const SensitivityState &sens) {
    auto &species = const_cast<std::vector<Species> &>(model.species());
    for (int i = 0; i < ns; ++i) {
        species[i].concentration = y_data[i];
    }
    model.set_current_time(final_t);

    // GH #210 — the state is now advanced past the ICs (carry-over). Thread
    // the forward-sensitivity matrix dx/dθ at this point into the model so a
    // subsequent carry_sensitivities=True run (the measurement phase of a
    // pre-equilibration) can seed yS(0) from it. sens.yS holds the final
    // integrated point (the loop's last CVodeGetSens). Capture the
    // parameter columns only, row-major [species*np + param]. A
    // non-sensitivity run still marks the state dirty but drops any stale
    // seed (it advanced the state without tracking dx/dθ).
    model.set_ic_state_dirty(true);
    if (sens.n_p > 0) {
        std::vector<double> seed(static_cast<size_t>(ns) * sens.n_p);
        for (int iS = 0; iS < sens.n_p; ++iS) {
            const double *col = N_VGetArrayPointer(sens.yS[iS]);
            for (int i = 0; i < ns; ++i) {
                seed[static_cast<size_t>(i) * sens.n_p + iS] = col[i];
            }
        }
        model.set_pending_sens_seed(std::move(seed), opts.sensitivity.param_names);
    } else {
        model.clear_pending_sens_seed();
    }
}

// ─── Forward-sensitivity jumps at a discontinuity (GH #135) ──────────────────
//
// The two places a sensitivity column has to jump rather than be integrated.
// Both bodies moved verbatim out of run(); each opens by binding the names its
// body uses to `sens`, to the model, and (for the switch jump) to the caller's
// scratch — so the maths below is the text that was reviewed with #48/#49/#82.

// ─── Forward-sensitivity jump across a fixed-time event (GH #212) ────────────
//
// A note on "GH #212", which appears throughout this file and in
// _bngsim_core.cpp: it is an internal phase-plan label, not an issue in this
// repository (lanl/bngsim has no #212, and lanl/PyBNF#212 is unrelated). The
// phases it names are Phase 1, fixed-time events — the jump below, shipped;
// Phase 2, state-dependent triggers — **issue #144**, which is where the
// crossing-time term now comes from; and Phase 3, execution delays, still
// unimplemented and tracked in issue #144. The two user-facing refusals that
// used to cite the label were repointed at issue #144; these internal
// references are left as the historical name of the code they describe.
//
// At a fixed-time event the state jumps x⁺ = h(x⁻, p); the forward
// sensitivity vectors must jump too, or the columns go silently stale (the
// GH #205 hazard this path lifts). For the Phase-1 subclass — fixed-time,
// persistent, no-delay, enforced upstream by the Python guard via
// NetworkModel::event_sensitivity_unsupported_reason — the event-time
// sensitivity ∂t*/∂p = 0, so the jump collapses to
//     s⁺_k = Σ_j (∂c_k/∂x_j)·s⁻_j + ∂c_k/∂p      (k = assigned species)
// with non-assigned rows unchanged.
//
// The pre-event sensitivities s⁻ MUST be captured BEFORE the caller's
// CVodeReInit: ReInit resets the state stepper and a subsequent CVodeGetSens
// returns sensitivities that no longer correspond to s⁻ (empirically this
// corrupted the jump for state-referencing assignments while leaving
// constant resets — which never use s⁻ — correct). So capture_event_sens()
// pulls s⁻ into yS_guard + a copy at root-return time, and
// apply_event_sensitivity_jump() consumes that copy after the assignments
// and CVodeReInit. yS_guard still holds s⁻ for the non-assigned rows; only
// assigned rows are overwritten, then CVodeSensReInit resumes the CVODES
// solver from the jumped sensitivities.
//
// The assignment-value derivatives ∂c/∂x, ∂c/∂p are obtained by central
// finite-difference of the value expression at x⁻; the expression's
// referenced variables (ExprTk) prune the FD to the species/params it
// actually reads, so the common constant/parameter assignment costs O(1).
// Derived-parameter chains inside an assignment value (a param defined as
// f(p_l)) are a Phase-1 limitation: only the direct ∂c/∂p_l is differenced.
// CVODES leaves the model's parameter values wherever its last
// finite-difference sensitivity probe put them — the RHS callbacks mirror
// sens_p into params and nothing writes them back afterwards — so any
// derivative read here would be taken at p ± √rtol·|p|. For a term that is
// MULTIPLIED into the jump (f⁻/f⁺ against ∂t*/∂p) that scales the whole
// jump by (1 ∓ √rtol), i.e. an answer that drifts with rtol. sens_p itself
// IS nominal between probes (CVODES restores the perturbed entry), so
// re-running the callbacks' own sync is enough; the resumed integration
// re-syncs on its next RHS call, so nothing needs undoing.
void CvodeSimulator::Impl::restore_nominal_params(const SensitivityState &sens) {
    auto &eval_ref_outer = model.evaluator();
    const std::vector<double> &sens_p = sens.p;

    if (sens_p.empty()) {
        return;
    }
    auto &params_live = const_cast<std::vector<Parameter> &>(model.parameters());
    const size_t np = std::min(params_live.size(), sens_p.size());
    for (size_t i = 0; i < np; ++i) {
        params_live[i].value = sens_p[i];
    }
    for (auto &p : params_live) {
        if (p.is_expression && p.evaluator_id >= 0) {
            p.value = eval_ref_outer.evaluate(p.evaluator_id);
        }
    }
}

void CvodeSimulator::Impl::rederive_expression_params(int skip_param_idx) {
    auto &eval_ref = model.evaluator();
    auto &params_live = const_cast<std::vector<Parameter> &>(model.parameters());
    for (int i = 0; i < static_cast<int>(params_live.size()); ++i) {
        if (i == skip_param_idx) {
            continue;
        }
        if (params_live[i].is_expression && params_live[i].evaluator_id >= 0) {
            params_live[i].value = eval_ref.evaluate(params_live[i].evaluator_id);
        }
    }
}

std::vector<std::vector<double>> CvodeSimulator::Impl::capture_event_sens(void *cvode_mem, int ns,
                                                                          double t_evt,
                                                                          SensitivityState &sens) {
    const int n_sens = sens.n_total;
    NVectorArrayGuard &yS_guard = sens.yS;

    std::vector<std::vector<double>> s_minus;
    if (n_sens == 0) {
        return s_minus;
    }
    sunrealtype t_tmp = static_cast<sunrealtype>(t_evt);
    int gf = CVodeGetSens(cvode_mem, &t_tmp, yS_guard.arr);
    if (gf != CV_SUCCESS) {
        throw std::runtime_error("CVodeGetSens for event sensitivity capture failed: " +
                                 std::to_string(gf));
    }
    s_minus.resize(static_cast<size_t>(n_sens));
    for (int c = 0; c < n_sens; ++c) {
        const double *col = N_VGetArrayPointer(yS_guard[c]);
        s_minus[c].assign(col, col + ns);
    }
    return s_minus;
}

void CvodeSimulator::Impl::resume_sens_after_reinit(
    void *cvode_mem, int ns, const std::vector<std::vector<double>> &s_at_reinit,
    SensitivityState &sens) {
    const int n_sens = sens.n_total;
    if (n_sens == 0 || s_at_reinit.empty()) {
        return;
    }
    NVectorArrayGuard &yS_guard = sens.yS;
    for (int c = 0; c < n_sens; ++c) {
        double *col = N_VGetArrayPointer(yS_guard[c]);
        std::copy(s_at_reinit[static_cast<size_t>(c)].begin(),
                  s_at_reinit[static_cast<size_t>(c)].begin() + ns, col);
    }
    int rf = CVodeSensReInit(cvode_mem, sens.method, yS_guard.arr);
    if (rf != CV_SUCCESS) {
        throw std::runtime_error("CVodeSensReInit after a no-op root reinit failed: " +
                                 std::to_string(rf));
    }
}

// ─── ∂t*/∂θ for a state-dependent trigger (issue #144) ───────────────────────
//
// A trigger that reads the state — AMICI's `neuron` fires on `v > 30` — has a
// crossing time that moves with every parameter through the trajectory, even
// though it names none of them. Neither of the two ways bngsim already knows a
// ∂t*/∂p can supply it: GH #212's fixed-time events have ∂t*/∂p = 0, and issue
// #49's detector resolves a *threshold* to primary parameters, which cannot
// serve `v > 30` (the threshold is a constant; the whole dependence is in the
// trajectory). It has to be differentiated where the crossing happens.
//
// The implicit function theorem on g(x(t*), p, t*) = 0 gives it directly:
//
//     dt*/dθ = − ( ∂g/∂x·S(t*⁻) + ∂g/∂p ) / ( ∂g/∂t + ∂g/∂x·f(x⁻) )
//
// with S = ∂x/∂θ (the pre-event sensitivities the caller captured) and f the
// RHS. Every term is read at the nominal parameter point and at x⁻, by central
// finite difference of the trigger's residual (NetworkModel::
// event_trigger_residual_expr — the boolean trigger itself is a step and
// carries no derivative). ∂g/∂t is kept even though it is zero for a purely
// state-dependent trigger: a trigger may read both `time` and the state, and
// then the denominator is the *total* dg/dt along the flow, not the state part
// alone.
//
// Two properties are worth stating because they are what makes this a drop-in
// for the existing jump:
//
//   * The IC-sensitivity columns get a non-zero ∂t*/∂θ too. For a fixed-time or
//     issue #49 event the crossing cannot move with an initial condition, which
//     is why EventTimeSens covers parameter columns only; here it plainly can —
//     perturb x(0) and the trajectory reaches the threshold at a different
//     time. The numerator's ∂g/∂p term is absent for those columns (an IC is
//     not in g), the ∂g/∂x·S term is not.
//   * The denominator is the transversality condition. When it vanishes the
//     trajectory grazes the trigger surface, t*(θ) is genuinely not
//     differentiable (a small perturbation destroys the crossing outright or
//     splits it in two), and dt*/dθ is unbounded — so a denominator that is not
//     resolvable is refused rather than divided by. That is the principled
//     failure this path replaces the blanket "state-dependent trigger" refusal
//     with.
//
// ONE way the denominator stops being resolvable, and it is refused here:
//
//   It is a near-total *cancellation* of its own terms. dg/dt is assembled as
//   ∂g/∂t + Σ ∂g/∂x_j·f_j and each factor is a central difference carrying
//   ~1e-10 relative error, so once the sum falls to 1e-8 of the scale of the
//   terms it is made of, the quotient is reporting the differences' noise
//   rather than the trajectory.
//
// There was a second, ABSOLUTE arm — ε·‖f‖∞ scaled by Σ|∂g/∂x| — retired in
// issue #322. It is a valid roundoff bound but a badly loose one: ‖f‖∞ ranges
// over every state, including ones the trigger never touches, so it overstates
// the true error by ‖f‖∞/|f_support|. On BIOMD0000000711 that was a factor of
// ~1e16 and it refused a crossing whose derivative is perfectly computable.
//
// A denominator that is small but NOT cancelled is now left alone, which is the
// long-standing rule in the line below applied consistently: dt*/dθ really is
// large there, and a large derivative is an answer. It is safe to allow because
// dt*/dθ is never consumed alone — every use multiplies it by a flow term that
// carries the same smallness, so a grazing crossing cancels back to a finite
// jump. The residual risk (a product that does NOT cancel) is caught by the
// post-jump finiteness guard in apply_event_sensitivity_jump, where the overflow
// is observed rather than predicted.
static constexpr double kTransversalityRelFloor = 1e-8;

bool CvodeSimulator::Impl::state_trigger_dtstar(int event_idx0, double t_evt, int ns,
                                                const std::vector<double> &x_minus,
                                                const std::vector<double> &f_minus,
                                                const std::vector<std::vector<double>> &s_minus,
                                                const SensitivityState &sens,
                                                std::vector<double> &tau_out) {
    if (sens.n_total == 0 || !model.event_trigger_is_state_dependent(event_idx0)) {
        return false;
    }
    const int gidx = model.event_trigger_residual_expr(event_idx0);
    if (gidx < 0) {
        // Upstream guard (NetworkModel::event_sensitivity_unsupported_reason)
        // refuses this model before the run, so reaching here means the trigger
        // became unresolvable after the guard ran. Leave ∂t*/∂θ alone rather
        // than inventing one.
        return false;
    }
    residual_dtstar(gidx, model.event_trigger_residual_species(event_idx0),
                    "event '" + model.events()[event_idx0].id + "' crosses its trigger", t_evt, ns,
                    x_minus, f_minus, s_minus, sens, tau_out);
    return true;
}

void CvodeSimulator::Impl::residual_dtstar(int gidx, const std::vector<int> &support,
                                           const std::string &subject, double t_evt, int ns,
                                           const std::vector<double> &x_minus,
                                           const std::vector<double> &f_minus,
                                           const std::vector<std::vector<double>> &s_minus,
                                           const SensitivityState &sens,
                                           std::vector<double> &tau_out) {
    const int n_sens = sens.n_total;
    const int n_sens_p = sens.n_p;

    auto &eval = model.evaluator();
    auto &sp_vec = const_cast<std::vector<Species> &>(model.species());
    auto &params = const_cast<std::vector<Parameter> &>(model.parameters());

    std::vector<double> xwork(x_minus.begin(), x_minus.end());
    auto sync = [&](double t) {
        for (int i = 0; i < ns; ++i) {
            sp_vec[i].concentration = xwork[i];
        }
        model.update_observables(xwork.data());
        model.evaluate_functions(t);
        // A trigger may read rateOf(species) (GH #106), whose bound value is
        // only refreshed by a derivative probe — without this the difference
        // would report 0 through that path instead of dx/dt's own dependence.
        if (model.uses_rateof()) {
            model.refresh_rateof_derivs(t, xwork.data());
        }
    };
    // Sync after perturbing parameter `skip_idx`: the first sync refreshes the
    // rule-bound parameters a model function writes, rederive_expression_params
    // then carries the perturbation into the derived parameters, and the second
    // sync lets a function read those. A threshold written over a derived
    // parameter (`v > 2*vth`) would otherwise report a flat ∂g/∂p of zero.
    auto perturbed_sync = [&](int skip_idx, double t) {
        sync(t);
        rederive_expression_params(skip_idx);
        sync(t);
    };

    sync(t_evt);

    // ∂g/∂x, over the species that can move g.
    std::vector<double> gx(static_cast<std::size_t>(ns), 0.0);
    for (int j : support) {
        const double xj = x_minus[j];
        double h = 1e-6 * std::fabs(xj);
        if (h == 0.0) {
            h = 1e-9;
        }
        xwork[j] = xj + h;
        sync(t_evt);
        const double g_hi = eval.evaluate(gidx);
        xwork[j] = xj - h;
        sync(t_evt);
        const double g_lo = eval.evaluate(gidx);
        xwork[j] = xj; // restore this component
        gx[j] = (g_hi - g_lo) / (2.0 * h);
    }
    sync(t_evt);

    // ∂g/∂p, per requested parameter column. A trigger that names no parameter
    // (`v > 30`) leaves this zero and pays only the differences.
    std::vector<double> gp(static_cast<std::size_t>(n_sens_p), 0.0);
    for (int c = 0; c < n_sens_p; ++c) {
        const int pidx = sens.param_indices[c];
        const double p0 = params[pidx].value;
        double h = 1e-6 * std::fabs(p0);
        if (h == 0.0) {
            h = 1e-9;
        }
        params[pidx].value = p0 + h;
        perturbed_sync(pidx, t_evt);
        const double g_hi = eval.evaluate(gidx);
        params[pidx].value = p0 - h;
        perturbed_sync(pidx, t_evt);
        const double g_lo = eval.evaluate(gidx);
        params[pidx].value = p0;
        perturbed_sync(pidx, t_evt);
        gp[c] = (g_hi - g_lo) / (2.0 * h);
    }

    // ∂g/∂t — the trigger's own explicit time dependence, held at x⁻.
    const double h_t = 1e-6 * std::max(std::fabs(t_evt), 1.0);
    sync(t_evt + h_t);
    const double g_t_hi = eval.evaluate(gidx);
    sync(t_evt - h_t);
    const double g_t_lo = eval.evaluate(gidx);
    const double gt = (g_t_hi - g_t_lo) / (2.0 * h_t);

    // Back to the nominal parameter point and to (x⁻, t*), which is the state
    // the caller's own differences expect to find.
    restore_nominal_params(sens);
    sync(t_evt);

    // Transversality: the denominator is dg/dt along the flow.
    double flow = gt;
    double scale = std::fabs(gt);
    for (int j : support) {
        const double term = gx[j] * f_minus[j];
        flow += term;
        scale += std::fabs(term);
    }

    // What makes a crossing non-transversal is CANCELLATION: terms of some size
    // summing to ~0. `scale` is exactly Σ|terms|, so |flow| <= relfloor*scale is
    // that test, and it is the only test made here (issue #322).
    //
    // There used to be a second, absolute arm: eps * gx_l1 * max_i|f_i|. That is
    // a VALID bound on the roundoff in `flow` but a very loose one, because
    // max_i|f_i| ranges over every state including ones the trigger never
    // touches. It is inflated by max|f| / |f_support|, so a model with one fast
    // species poisoned the test for events on every slow species. On
    // BIOMD0000000711 the true roundoff is ~eps*1.19e-13 ≈ 3e-29 while the bound
    // computed 2.22e-12 — off by ~1e16, and it refused the crossing.
    //
    // Removing it does NOT mean a grazing crossing now yields garbage. `tau` is
    // never used alone: every consumer multiplies it by a flow term
    // ((f⁻-f⁺)·tau, f⁺·tau, f⁻·tau), and for a grazing crossing those factors
    // carry the same smallness as the denominator, so the blow-up cancels and
    // the jump lands finite. Where it genuinely does not cancel the result is
    // non-finite, and the caller's post-jump guard refuses there — at the point
    // where the damage is observable, instead of inferring it from a proxy.
    if (!std::isfinite(flow) || scale == 0.0 ||
        std::fabs(flow) <= kTransversalityRelFloor * scale) {
        std::ostringstream msg;
        msg << "Forward sensitivity: " << subject << " tangentially at t=" << t_evt
            << " — the residual's rate of change along the trajectory is " << flow
            << ", which its own terms (scale " << scale << ") cancel to within "
            << kTransversalityRelFloor
            << ". The crossing time is not differentiable there (an arbitrarily small "
               "parameter change destroys the crossing or splits it in two), so dt*/dp is "
               "unbounded and bngsim refuses rather than divide by it (issue #144). Move the "
               "threshold off the trajectory's turning point, or drop sensitivities for this "
               "run.";
        throw std::runtime_error(msg.str());
    }

    tau_out.assign(static_cast<std::size_t>(n_sens), 0.0);
    for (int c = 0; c < n_sens; ++c) {
        double num = (c < n_sens_p) ? gp[static_cast<std::size_t>(c)] : 0.0;
        const std::vector<double> &sm = s_minus[c];
        for (int j : support) {
            num += gx[j] * sm[j];
        }
        tau_out[static_cast<std::size_t>(c)] = -num / flow;
    }
}

void CvodeSimulator::Impl::apply_event_sensitivity_jump(
    const SolverOptions &opts, void *cvode_mem, int ns, double t_evt, const std::vector<int> &fired,
    const std::vector<double> &x_minus, const std::vector<std::vector<double>> &s_minus,
    SensitivityState &sens, bool at_run_start) {
    auto &eval_ref_outer = model.evaluator();
    auto &sp_vec_outer = const_cast<std::vector<Species> &>(model.species());
    const auto &events_outer = model.events();
    const int n_sens = sens.n_total;
    const int n_sens_p = sens.n_p;
    const int sens_method = sens.method;
    const std::vector<int> &sens_param_indices = sens.param_indices;
    NVectorArrayGuard &yS_guard = sens.yS;

    if (n_sens == 0 || fired.empty() || s_minus.empty()) {
        return;
    }

    auto &params = const_cast<std::vector<Parameter> &>(model.parameters());

    // Every derivative below is read at the nominal parameter point, not at
    // whatever CVODES' last FD probe left behind (see
    // restore_nominal_params). ∂t*/∂p multiplies f⁻/f⁺ into the jump, so a
    // probe-point read would make the answer drift with rtol.
    restore_nominal_params(sens);

    // ─── ∂t*/∂θ for this batch (issue #49, issue #144) ───────────────────
    // Zero (the GH #212 Phase-1 case) unless the crossing time actually moves.
    // Sized over ALL sensitivity columns, not just the parameter ones: issue
    // #49's detector covers parameter columns only (an initial condition cannot
    // move a clock), but a state-dependent trigger's crossing moves with an
    // initial condition too, and the IC columns' shift is filled in by
    // state_trigger_dtstar below. The issue #49 path leaves them at 0, which is
    // what it meant before this vector grew.
    std::vector<double> tau(static_cast<size_t>(n_sens), 0.0);
    bool tau_nonzero = false;
    int tau_event = -1; // which fired event `tau` belongs to (-1: none yet)
    // Two events firing at the SAME instant whose crossing times move
    // differently. The assignments are applied as one simultaneous batch, so
    // there is no single t*(θ) to shift the flow along and the composition is
    // genuinely ambiguous. Refuse rather than pick one and return a plausible
    // number.
    auto adopt_tau = [&](int ei, const std::vector<double> &candidate) {
        if (tau_event >= 0 && candidate != tau) {
            throw std::runtime_error(
                "Forward sensitivity: events '" + events_outer[tau_event].id + "' and '" +
                events_outer[ei].id + "' fire at the same instant t=" + std::to_string(t_evt) +
                " but their crossing times move differently with the requested "
                "parameters, so the event-time sensitivity jump is ambiguous "
                "(issue #49). Separate the trigger times, or drop the parameters "
                "that move them from sensitivity_params.");
        }
        tau_event = ei;
        tau = candidate;
        tau_nonzero = true;
    };
    // Which of the fired events get their crossing differentiated here (issue
    // #144). Resolved first, because it decides which of them the issue #49
    // detector's records still apply to — and consulted before f⁻/f⁺ are
    // computed, because the denominator of dt*/dθ needs f⁻.
    //
    // A state-dependent trigger takes this path even when the detector DID
    // resolve it (a threshold on a unit-rate counter is both), because the
    // detector's ∂t*/∂p covers parameter columns only and a counter's crossing
    // moves with that counter's initial condition too. Nothing issue #49
    // validated changes: an event both paths could claim reads live state, and
    // those were refused outright before this issue.
    //
    // Never at t_start: a t=0 fire is an SBML L3 §3.4.5 initial-value fire, not
    // a located crossing. Its trigger was already satisfied when the run began,
    // so the fire happens at t_start for every θ in a neighbourhood and
    // ∂t*/∂θ is 0 — differentiating the trigger there would answer with the
    // rate at which a crossing that is not happening would move.
    std::vector<int> state_dep_fired;
    for (int ei : fired) {
        if (!at_run_start && model.event_trigger_is_state_dependent(ei) &&
            model.event_trigger_residual_expr(ei) >= 0) {
            state_dep_fired.push_back(ei);
        }
    }
    const std::unordered_set<int> differentiated_here(state_dep_fired.begin(),
                                                      state_dep_fired.end());

    // Pass 1 — the crossing times issue #49's Python detector resolved ahead of
    // the run. Its records carry the parameter columns only, so they are
    // widened with zero IC columns here.
    for (int ei : fired) {
        if (differentiated_here.count(ei) != 0) {
            continue;
        }
        for (const auto &et : opts.sensitivity.event_times) {
            if (et.event_idx0 != ei || et.dtstar_dp.size() != static_cast<size_t>(n_sens_p)) {
                continue;
            }
            if (std::all_of(et.dtstar_dp.begin(), et.dtstar_dp.end(),
                            [](double v) { return v == 0.0; })) {
                continue;
            }
            std::vector<double> widened(static_cast<size_t>(n_sens), 0.0);
            std::copy(et.dtstar_dp.begin(), et.dtstar_dp.end(), widened.begin());
            adopt_tau(ei, widened);
        }
    }
    const bool needs_flow = tau_nonzero || !state_dep_fired.empty();

    // Save the post-event state, then drive the evaluator to x⁻ so the
    // derivative evaluations below see pre-event values. Restored at the end.
    std::vector<double> x_post(static_cast<size_t>(ns));
    for (int i = 0; i < ns; ++i) {
        x_post[i] = sp_vec_outer[i].concentration;
    }

    // f⁺ = f(t*, x⁺) while the evaluator still holds the post-event state,
    // and f⁻ = f(t*, x⁻) once it has been driven back. Only needed when the
    // crossing time actually moves; an ordinary fixed-time event pays
    // nothing. Unlike the issue #48 switch jump these are the RHS at two
    // DIFFERENT states rather than two branches at one state: an event
    // trigger is not part of f, so f is the same smooth function on both
    // sides and it is the state that jumps.
    std::vector<double> f_minus, f_plus;
    if (needs_flow) {
        f_plus.assign(static_cast<size_t>(ns), 0.0);
        f_minus.assign(static_cast<size_t>(ns), 0.0);
        model.compute_derivs(t_evt, x_post.data(), f_plus.data());
    }

    std::vector<double> xwork(x_minus.begin(), x_minus.end());
    auto sync_state = [&]() {
        for (int i = 0; i < ns; ++i) {
            sp_vec_outer[i].concentration = xwork[i];
        }
        model.update_observables(xwork.data());
        model.evaluate_functions(t_evt);
    };
    // Sync after perturbing parameter `skip_idx` — see the identically-shaped
    // helper in state_trigger_dtstar: functions, then the derived-parameter
    // chain, then functions again for anything that reads a derived parameter.
    auto perturbed_sync = [&](int skip_idx, double) {
        sync_state();
        rederive_expression_params(skip_idx);
        sync_state();
    };
    sync_state();
    if (needs_flow) {
        model.compute_derivs(t_evt, xwork.data(), f_minus.data());
        sync_state(); // compute_derivs may leave functions at its own state
    }

    // Pass 2 — differentiate the crossing of each state-dependent trigger that
    // fired, now that f⁻ is available. Leaves the evaluator back at (x⁻, p₀).
    for (int ei : state_dep_fired) {
        std::vector<double> candidate;
        if (state_trigger_dtstar(ei, t_evt, ns, x_minus, f_minus, s_minus, sens, candidate)) {
            adopt_tau(ei, candidate);
        }
    }

    // Rows the batch assigns take the ∂h/∂x·(…) + ∂h/∂p form below; every
    // other row is continuous (h_k = x_k) and reduces to the issue #48 jump
    // s⁺_k = s⁻_k + (f⁻_k − f⁺_k)·∂t*/∂p, applied here in place on yS_guard
    // (which still holds s⁻). Done before the assigned rows are written so
    // the two never see each other's output.
    std::unordered_set<int> assigned_rows;
    if (tau_nonzero) {
        for (int ei : fired) {
            for (const auto &asg : events_outer[ei].assignments) {
                assigned_rows.insert(asg.first);
            }
        }
        for (int c = 0; c < n_sens; ++c) {
            if (tau[static_cast<size_t>(c)] == 0.0) {
                continue;
            }
            double *col = N_VGetArrayPointer(yS_guard[c]);
            for (int i = 0; i < ns; ++i) {
                if (assigned_rows.count(i) == 0) {
                    col[i] += (f_minus[i] - f_plus[i]) * tau[static_cast<size_t>(c)];
                }
            }
        }
    }

    for (int ei : fired) {
        const auto &ev = events_outer[ei];
        for (const auto &asg : ev.assignments) {
            const int k = asg.first;      // assigned species (0-based)
            const int vexpr = asg.second; // value expression id
            if (k < 0 || k >= ns) {
                continue;
            }
            // Restrict the FD to what the assignment value can actually be
            // moved by — species and parameters, each followed through the
            // observables and derived/rule-bound parameters that hide them.
            // Matching on the referenced addresses alone silently zeroed both
            // halves: an SBML species token binds to its same-named observable
            // rather than to &sp.concentration, and a derived parameter names
            // neither of the primaries behind it. See
            // NetworkModel::expression_support.
            std::vector<int> x_support, p_support;
            model.expression_support(vexpr, &x_support, &p_support);
            const std::unordered_set<int> p_support_set(p_support.begin(), p_support.end());

            // ∂c/∂x_j via central FD.
            std::vector<double> dcdx(static_cast<size_t>(ns), 0.0);
            for (int j : x_support) {
                const double xj = x_minus[j];
                double h = 1e-6 * std::fabs(xj);
                if (h == 0.0) {
                    h = 1e-9;
                }
                xwork[j] = xj + h;
                sync_state();
                const double f_hi = eval_ref_outer.evaluate(vexpr);
                xwork[j] = xj - h;
                sync_state();
                const double f_lo = eval_ref_outer.evaluate(vexpr);
                xwork[j] = xj; // restore this component
                dcdx[j] = (f_hi - f_lo) / (2.0 * h);
            }
            sync_state(); // back to x⁻ for the parameter FD

            // ∂c/∂p for each parameter column (IC columns: ∂c/∂p ≡ 0).
            std::vector<double> dcdp(static_cast<size_t>(n_sens_p), 0.0);
            for (int col = 0; col < n_sens_p; ++col) {
                const int pidx = sens_param_indices[col];
                if (p_support_set.count(pidx) == 0) {
                    continue;
                }
                const double p0 = params[pidx].value;
                double h = 1e-6 * std::fabs(p0);
                if (h == 0.0) {
                    h = 1e-9;
                }
                params[pidx].value = p0 + h;
                perturbed_sync(pidx, t_evt);
                const double f_hi = eval_ref_outer.evaluate(vexpr);
                params[pidx].value = p0 - h;
                perturbed_sync(pidx, t_evt);
                const double f_lo = eval_ref_outer.evaluate(vexpr);
                params[pidx].value = p0; // restore
                perturbed_sync(pidx, t_evt);
                dcdp[col] = (f_hi - f_lo) / (2.0 * h);
            }
            sync_state(); // restore evaluator state at (x⁻, p₀)

            // Assemble s⁺_k for every sensitivity column:
            //     s⁺_k = Σ_j (∂h_k/∂x_j)·(s⁻_j + f⁻_j·∂t*/∂p)
            //            + ∂h_k/∂p − f⁺_k·∂t*/∂p
            // The pre-shift carries s⁻ along the pre-event flow by how far
            // the event time moves, the event Jacobian maps it through the
            // reset, and the post-shift carries it back along the
            // post-event flow. With ∂t*/∂p = 0 both shifts vanish and this
            // is the GH #212 jump unchanged.
            for (int c = 0; c < n_sens; ++c) {
                // IC columns carry a shift only for a state-dependent trigger
                // (issue #144); issue #49's detector leaves them at 0.
                const double tau_c = tau[static_cast<size_t>(c)];
                double acc = 0.0;
                const std::vector<double> &sm = s_minus[c];
                for (int j = 0; j < ns; ++j) {
                    if (dcdx[j] != 0.0) {
                        acc += dcdx[j] * (sm[j] + (tau_c != 0.0 ? f_minus[j] * tau_c : 0.0));
                    }
                }
                if (c < n_sens_p) {
                    acc += dcdp[c];
                }
                if (tau_c != 0.0) {
                    acc -= f_plus[k] * tau_c;
                }
                N_VGetArrayPointer(yS_guard[c])[k] = acc;
            }
        }
    }

    // Post-jump conditioning guard (issue #322). residual_dtstar now refuses
    // only genuine non-transversality (cancellation), so a GRAZING crossing —
    // small |flow|, no cancellation — reaches here with a large ∂t*/∂p. That is
    // usually harmless: every use of tau above multiplies it by a flow term
    // ((f⁻-f⁺)·tau, f⁺·tau), and those factors carry the same smallness, so the
    // ratio lands finite. BIOMD0000000711's non-negativity clamp is exactly this
    // shape and its jump reduces to the incoming s⁻.
    //
    // Where the cancellation does NOT occur the product overflows, and that is
    // the honest place to refuse: the damage is measured rather than predicted
    // from a proxy on an intermediate. Checking the OUTPUT also covers every
    // route into the jump at once, including ones a guard on tau alone would
    // miss.
    if (tau_nonzero) {
        for (int c = 0; c < n_sens; ++c) {
            const double *col = N_VGetArrayPointer(yS_guard[c]);
            for (int i = 0; i < ns; ++i) {
                if (!std::isfinite(col[i])) {
                    std::ostringstream msg;
                    msg << "Forward sensitivity: the jump across the event at t=" << t_evt
                        << " produced a non-finite dx/dp (column " << c << ", species " << i
                        << "). The crossing time moves with this parameter, and the crossing is "
                           "grazing enough that ∂t*/∂p overflows the jump rather than cancelling "
                           "against the flow difference — so the derivative does not exist in any "
                           "usable sense there (issue #322). Move the threshold off the "
                           "trajectory's turning point, or drop sensitivities for this run.";
                    throw std::runtime_error(msg.str());
                }
            }
        }
    }

    // Restore the post-event state so downstream code and resumed
    // integration see x⁺ (not the x⁻ used for differentiation).
    xwork.assign(x_post.begin(), x_post.end());
    sync_state();

    int rf = CVodeSensReInit(cvode_mem, sens_method, yS_guard.arr);
    if (rf != CV_SUCCESS) {
        throw std::runtime_error("CVodeSensReInit after event sensitivity jump failed: " +
                                 std::to_string(rf));
    }
}

// ─── Forward-sensitivity jump across a switch time (issue #48) ───────────────
// A *switch time* is a fitted parameter that sets WHEN a step in the
// dynamics happens — the `if(t>=sigma, ...)` onset times of the Lin2021
// COVID model, gated on a unit-rate counter clock. Unlike a GH #212 event,
// the state is continuous across the crossing and it is the crossing TIME
// that moves with the parameter, so the jump above (∂t*/∂p = 0, x jumps)
// becomes its mirror image (∂t*/∂p ≠ 0, x continuous):
//
//     s(t*⁺) = s(t*⁻) + (f⁻ − f⁺)·∂t*/∂p
//
// This is the ENTIRE gradient: ∂f/∂p is a clean 0 inside each smooth branch
// (sympy drops the boundary delta when the parameter appears only in the
// `if` condition), so without this jump the switch-time column comes back
// silently zero. Validated against finite differences on both a minimal
// model and the 26-species Lin2021 exemplar — see issue #48.
//
// Two mechanics are worth spelling out:
//
//   * **Reaching the crossing.** With sensitivities active CVODES fails
//     error control on the step APPROACHING the kink and collapses h to
//     ~1e-15 (mxstep, never returns) — before any root can fire, so the GH
//     #72 discontinuity root cannot break that step. The CVodeSetStopTime()
//     in run()'s stepping loop clamps the step so the last one lands exactly
//     on t* with the before-branch RHS smooth over the whole interval. That
//     is what makes the crossing reachable at all; the root machinery is
//     untouched.
//
//   * **Reading off f⁻ and f⁺.** Both must be the RHS at the SAME crossing
//     state x(t*), differing only in which branch is live. Rather than
//     rebuild the model per branch (what the Python prototype did), we nudge
//     the *clock* a few ulp either side of its threshold: the branch flips
//     exactly, while the smooth part of the RHS moves by O(ulp) and cancels
//     in the difference. This works for every comparison operator (`>=`,
//     `<`, …) because "before" is always the smaller clock value, and it
//     needs no knowledge of the condition's structure.
//
// IC-sensitivity columns are not jumped: a clock whose own initial condition
// is a fitted parameter would move t*, which the Python detector refuses
// rather than silently zeroing.
void CvodeSimulator::Impl::apply_switch_sensitivity_jump(void *cvode_mem, N_Vector y, int ns,
                                                         double t_evt, const SwitchTimeSens &sw,
                                                         SwitchJumpScratch &scratch,
                                                         SensitivityState &sens) {
    auto &sp_vec_outer = const_cast<std::vector<Species> &>(model.species());
    double *y_data = N_VGetArrayPointer(y);
    const int n_sens_p = sens.n_p;
    const int sens_method = sens.method;
    NVectorArrayGuard &yS_guard = sens.yS;
    std::vector<double> &sw_f_minus = scratch.f_minus;
    std::vector<double> &sw_f_plus = scratch.f_plus;
    std::vector<double> &sw_ywork = scratch.ywork;

    // Restore the nominal parameter point first, or the whole jump comes out
    // scaled by (1 ∓ √rtol) — see restore_nominal_params.
    restore_nominal_params(sens);

    // f⁻ / f⁺ at x(t*), branch selected by nudging the clock across its
    // threshold. eps_clock is a few ulp of the threshold: large enough that
    // threshold ± eps_clock are distinct doubles, small enough that the
    // smooth part of the RHS is unchanged to roundoff.
    std::copy(y_data, y_data + ns, sw_ywork.begin());
    const bool time_clock = (sw.clock_species_idx0 < 0);
    const double eps_clock = 64.0 * std::numeric_limits<double>::epsilon() *
                             std::max(std::fabs(time_clock ? t_evt : sw.threshold), 1.0);
    auto rhs_on_branch = [&](double offset, std::vector<double> &out) {
        if (!time_clock) {
            sw_ywork[static_cast<size_t>(sw.clock_species_idx0)] = sw.threshold + offset;
        }
        // Sync the evaluator's species symbols, as the root handler does:
        // compute_derivs() refreshes observables and functions from the
        // passed array but does not write back the bound concentrations.
        for (int i = 0; i < ns; ++i) {
            sp_vec_outer[i].concentration = sw_ywork[i];
        }
        model.compute_derivs(time_clock ? t_evt + offset : t_evt, sw_ywork.data(), out.data());
    };
    rhs_on_branch(-eps_clock, sw_f_minus);
    rhs_on_branch(+eps_clock, sw_f_plus);

    // ── Land the clock ON its threshold, not a few ulp short (issue #82) ──
    // The stop time puts t exactly on t*, but the condition is read off the
    // CLOCK SPECIES, and that clock is integrated: counter(t*) comes back
    // 1–2e-14 BELOW the threshold it is supposed to have reached. So the
    // restart below re-enters on the *before* branch and the discontinuity
    // lands inside the first step after the restart — the one place the stop
    // time exists to prevent it. CVODES then sizes h from the pre-switch RHS
    // (identically 0 on this model: no transmission, no distancing), every
    // corrector answers with the post-switch RHS, and the error test fails at
    // every h down to ~1e-10 — 7 failures, no step completed, CV_ERR_FAILURE
    // at the crossing. Which side of the threshold the last bits fall on is
    // deterministic but effectively arbitrary, which is why issue #82 looks
    // like isolated spikes in parameter space and moves non-monotonically
    // with rtol: a fit lost 25% of otherwise-integrable candidates to it.
    //
    // t* is DEFINED as the time the clock reaches `threshold` (a unit-rate
    // counter, so t* = threshold − offset exactly), so setting the clock
    // there is a correction of accumulated integration error, not a
    // perturbation. nextafter puts it on the after-branch of a strict `>` as
    // well as a `>=`, at a cost of one ulp. Discrepancies too large to be
    // roundoff are left alone: those would mean the crossing was detected in
    // the wrong place, and silently moving state would only hide it.
    if (!time_clock) {
        const size_t clk = static_cast<size_t>(sw.clock_species_idx0);
        const double drift = y_data[clk] - sw.threshold;
        const double drift_max = 1e-9 * std::max(std::fabs(sw.threshold), 1.0);
        if (drift < 0.0 && -drift <= drift_max) {
            y_data[clk] = std::nextafter(sw.threshold, std::numeric_limits<double>::infinity());
        }
    }

    // Restore the evaluator to the true crossing state so the resumed
    // integration (and any downstream reader) sees x(t*), not the nudge.
    for (int i = 0; i < ns; ++i) {
        sp_vec_outer[i].concentration = y_data[i];
    }
    model.update_observables(y_data);
    model.evaluate_functions(t_evt);
    if (model.uses_rateof()) {
        model.refresh_rateof_derivs(t_evt, y_data);
    }

    // s⁻ MUST be read before CVodeReInit — after it, CVodeGetSens no longer
    // returns the pre-crossing columns (the same ordering hazard the GH #212
    // jump documents above).
    sunrealtype t_tmp = static_cast<sunrealtype>(t_evt);
    int gf = CVodeGetSens(cvode_mem, &t_tmp, yS_guard.arr);
    if (gf != CV_SUCCESS) {
        throw std::runtime_error("CVodeGetSens for switch-time sensitivity capture failed: " +
                                 std::to_string(gf));
    }
    for (int c = 0; c < n_sens_p; ++c) {
        const double dtstar = sw.dtstar_dp[static_cast<size_t>(c)];
        if (dtstar == 0.0) {
            continue; // this parameter does not move this crossing
        }
        double *col = N_VGetArrayPointer(yS_guard[c]);
        for (int i = 0; i < ns; ++i) {
            col[i] += (sw_f_minus[i] - sw_f_plus[i]) * dtstar;
        }
    }

    // Restart the state stepper AT the kink (order drops to 1, history
    // discarded) — the same reason the GH #72 discontinuity root reinits —
    // then resume CVODES from the jumped sensitivities.
    int rf = reinit_cvode(cvode_mem, static_cast<sunrealtype>(t_evt), y);
    if (rf != CV_SUCCESS) {
        throw std::runtime_error("CVodeReInit at switch time failed: " + std::to_string(rf));
    }
    rf = CVodeSensReInit(cvode_mem, sens_method, yS_guard.arr);
    if (rf != CV_SUCCESS) {
        throw std::runtime_error("CVodeSensReInit after switch-time sensitivity jump failed: " +
                                 std::to_string(rf));
    }
}

// ─── Saltation jump at a state-dependent rate-law switch (issue #150) ────────
//
// The rate-law twin of the issue #144 event trigger, and the mirror image of the
// issue #48 clock switch. A condition like `Virus < 1` inside a piecewise rate
// law flips a branch of f at a crossing whose time t*(θ) moves with EVERY
// parameter through the trajectory. The state is continuous there, so this is
// the issue #48 jump
//
//     s(t*⁺) = s(t*⁻) + (f⁻ − f⁺)·dt*/dθ
//
// with dt*/dθ differentiated at the crossing rather than resolved ahead of the
// run — the implicit function theorem on g(x(t*), θ, t*) = 0, which is exactly
// what residual_dtstar computes. Both halves already existed; what was missing
// was a crossing to run them on, which the residual root now supplies.
//
// The columns are jumped in the caller's captured s⁻, in place; the caller
// re-seeds CVODES from it (resume_sens_after_reinit) once this returns.
//
// **Reading off f⁻ and f⁺.** Both must be the RHS at the SAME crossing state
// x(t*), differing only in which branch is live. A clock switch selects the
// branch by nudging the clock across its threshold; a state switch has no such
// variable, so nudge the whole state ALONG THE FLOW by ±δt:
//
//     x^± = x(t*) ± δt·f(x(t*))
//
// which is the same operation — for a unit-rate counter f_clock = 1 and it
// reduces to the clock nudge exactly — and moves the residual by δt·dg/dt, the
// very quantity the transversality check certifies as non-degenerate. The
// smooth part of f moves by O(δt·J·f) and cancels in the difference.
//
// δt cannot simply be a few ulp: CVODE locates a root only to
// ~100·ε·(|t| + |h|), so a nudge that small can land on the wrong side of a
// crossing the root finder already resolved to its own tolerance. So the size is
// MEASURED rather than assumed — start just past the root-location error and
// grow geometrically until the residual actually changes sign across the pair,
// then stop. Failing to flip inside the ladder means the residual is not
// behaving like a transversal crossing at all; what happens then is the next
// paragraph, and it is never a silent jump of zero (which is what "we could not
// tell the branches apart" would otherwise be indistinguishable from).
//
// **A crossing with no jump at it.** The saltation term is (f⁻ − f⁺)·dt*/dθ, so
// everything above is moot when the two branches of f agree at x(t*) — and the
// most common `piecewise` in the corpus is exactly that. A clamp is CONTINUOUS
// at its own switch: BIOMD0000000161's basal PIP synthesis is
// `piecewise(0.581·k·(exp((basal − PIP)/basal) − 1), PIP < basal, 0)`, whose
// live branch is exactly 0 where PIP = basal. There is nothing to add to s⁻
// there, however unbounded dt*/dθ may be, so the branch gap is measured FIRST
// and a continuous switch returns before any of it is formed.
//
// That also decides what a flow probe that never flips means. It means the
// trajectory does not pass through the surface — it rides it — which is a
// tangency, and a tangency is only fatal when there IS a jump. So instead of
// refusing outright, straddle the surface along a COORDINATE of the residual's
// own support, where the gradient survives a tangential flow, and ask the same
// question. Orientation does not matter for it: only |f_a − f_b| is being read.
// That probe is a Newton step onto the surface along the best-conditioned
// coordinate, then ±eta past it, so it is scale-free — a fixed relative nudge
// fails whenever the residual carries the model's units rather than a species'.
//
// **Several residuals at one instant** (issue #153). The roots are deduplicated
// by residual TEXT, which merges `X<1` with `X<=1` and nothing else, so two
// spellings of one crossing arrive as a batch: `sp_fourier_synthesizer` roots on
// `ds1` and on `3·ds1 − 12·s1²·ds1`, and `ml_hopfield` on `dS1/dt` and `dS3/dt`,
// which are equal along its trajectory because its own weight matrix leaves
// `S1 ≡ S3` invariant. Both want ONE jump, not a composition of several. The
// batch is therefore carried in together and the two halves of the saltation
// term are checked separately, because they need different things:
//
//   * (f⁻ − f⁺) has to carry EVERY branch change, which it does exactly when the
//     one flow probe crosses every residual in the batch — the ladder therefore
//     grows δt until they all flip together, and a batch that never does is not
//     one crossing but several meeting by coincidence.
//   * dt*/dθ has to be ONE vector, and flipping together does NOT establish
//     that. A common factor does: g_k = h·g_1 scales the implicit-function
//     numerator and denominator alike where g_1 vanishes, so it cancels and the
//     ratio is the same. An equality that holds only ALONG THE TRAJECTORY does
//     not — ml_hopfield's two residuals have non-parallel gradients (cosine
//     −0.30 at its crossing) and their dt*/dθ are permutations of each other
//     under the W12 ↔ W23 symmetry, i.e. the crossing SPLITS under a
//     perturbation that breaks it, and then each branch change moves with its
//     own t*. So the vector is formed from each residual in turn and compared,
//     and a batch that does not agree is refused with the numbers.
//
// The order matters and is the cheap way round: the branch gap is measured
// before any dt*/dθ, and both corpus models are the BNGL signed-rate idiom
// (`if(r>0, r, 0)` against `if(r<0, −r, 0)`, continuous where r = 0), so they
// return with no jump — and no merge decision — before the second check is
// reached at all.
static constexpr double kStateSwitchNudgeStart = 256.0; // × ε · max(|t*|, 1)
static constexpr double kStateSwitchNudgeGrowth = 8.0;
static constexpr int kStateSwitchNudgeTries = 6; // ⇒ up to ~2e-9 · max(|t*|, 1)
static constexpr double kStateSwitchContinuousRelTol = 1e-6;
// A gap that grows linearly with the probe (ratio → 0.5) is f varying smoothly
// over the displacement; a gap that does not move (ratio → 1) is a jump.
static constexpr double kStateSwitchGapRatio = 0.7;
// Two residuals name one crossing when their dt*/dθ agree to this, read against
// the largest column of the vector. The merged jump's own relative error is
// bounded by the same number (it is (Δ₁+Δ₂)·τ₁ against Δ₁·τ₁ + Δ₂·τ₂), so this
// is the accuracy the merge is worth, not a guess at the differencing noise —
// which for a genuine common factor is orders of magnitude below it.
static constexpr double kStateSwitchTauAgreeTol = 1e-4;

void CvodeSimulator::Impl::apply_state_switch_sensitivity_jump(
    void *cvode_mem, N_Vector y, int ns, double t_evt,
    const std::vector<const NetworkModel::StateSwitch *> &batch,
    std::vector<std::vector<double>> &s, SensitivityState &sens) {
    const int n_sens = sens.n_total;
    if (n_sens == 0 || s.empty() || batch.empty()) {
        return;
    }
    const std::size_t nb = batch.size();
    // The residual the crossing surface is DEFINED by. Everything below that
    // needs a single g reads it from here; the rest of the batch is checked
    // against it, and for the batches that get past those checks the choice
    // does not matter (they all describe the same crossing).
    const NetworkModel::StateSwitch &sw = *batch.front();
    double *y_data = N_VGetArrayPointer(y);

    // Every derivative below is read at the nominal parameter point, not at
    // whatever CVODES' last finite-difference probe left behind — dt*/dθ
    // multiplies f⁻/f⁺ into the jump, so a probe-point read would make the
    // answer drift with rtol (see restore_nominal_params).
    restore_nominal_params(sens);

    auto &eval = model.evaluator();
    auto &sp_vec = const_cast<std::vector<Species> &>(model.species());

    // x(t*) as the root finder located it — the state every difference below is
    // taken at, and the one the caller's captured s⁻ belongs to.
    const std::vector<double> x(y_data, y_data + ns);
    std::vector<double> xw(static_cast<std::size_t>(ns), 0.0);
    std::vector<double> f0(static_cast<std::size_t>(ns), 0.0);
    std::vector<double> f_minus(static_cast<std::size_t>(ns), 0.0);
    std::vector<double> f_plus(static_cast<std::size_t>(ns), 0.0);

    auto sync = [&](const std::vector<double> &state, double t) {
        for (int i = 0; i < ns; ++i) {
            sp_vec[i].concentration = state[i];
        }
        model.update_observables(state.data());
        model.evaluate_functions(t);
        if (model.uses_rateof()) {
            model.refresh_rateof_derivs(t, state.data());
        }
    };

    // f at the crossing, on whichever branch the located state happens to sit.
    // Only its direction is used, to step off the surface either way.
    sync(x, t_evt);
    model.compute_derivs(t_evt, x.data(), f0.data());

    // The probe point at ±δt along the flow, and EVERY residual in the batch
    // there. `t ± δt` rather than `t` on both: `x ± δt·f` IS the trajectory at
    // that instant to first order, so a condition reading both the state and
    // `time` selects its branch consistently. Either way the smooth part of f
    // moves by O(δt) and cancels in f⁻ − f⁺.
    std::vector<double> g_before(nb, 0.0);
    std::vector<double> g_after(nb, 0.0);
    auto probe = [&](double signed_dt, std::vector<double> &g_out) {
        for (int i = 0; i < ns; ++i) {
            xw[static_cast<std::size_t>(i)] =
                x[static_cast<std::size_t>(i)] + signed_dt * f0[static_cast<std::size_t>(i)];
        }
        sync(xw, t_evt + signed_dt);
        for (std::size_t k = 0; k < nb; ++k) {
            g_out[k] = eval.evaluate(batch[k]->residual_expr_idx);
        }
    };
    // How many of the batch's residuals this probe pair actually straddles. All
    // of them is what says f⁻ − f⁺ carries every branch change (issue #153);
    // for a single switch it is the sign-flip test this ladder always ran.
    auto n_straddled = [&]() {
        std::size_t n = 0;
        for (std::size_t k = 0; k < nb; ++k) {
            if (std::isfinite(g_before[k]) && std::isfinite(g_after[k]) &&
                ((g_before[k] < 0.0 && g_after[k] > 0.0) ||
                 (g_before[k] > 0.0 && g_after[k] < 0.0))) {
                ++n;
            }
        }
        return n;
    };

    const double t_scale = std::max(std::fabs(t_evt), 1.0);
    const double dt0 = kStateSwitchNudgeStart * std::numeric_limits<double>::epsilon() * t_scale;
    double dt = dt0;
    double dt_used = 0.0;
    std::size_t best_straddled = 0;
    for (int attempt = 0; attempt < kStateSwitchNudgeTries;
         ++attempt, dt *= kStateSwitchNudgeGrowth) {
        probe(-dt, g_before);
        probe(+dt, g_after);
        const std::size_t n = n_straddled();
        best_straddled = std::max(best_straddled, n);
        if (n == nb) {
            dt_used = dt;
            break;
        }
    }
    // |f⁻ − f⁺| against the scale of f itself: below this, the two branches are
    // the same function here and the saltation term is zero.
    auto branch_gap = [&](double &scale_out) {
        double gap = 0.0;
        scale_out = 0.0;
        for (int i = 0; i < ns; ++i) {
            gap = std::max(gap, std::fabs(f_minus[i] - f_plus[i]));
            scale_out = std::max(scale_out, std::max(std::fabs(f_minus[i]), std::fabs(f_plus[i])));
        }
        return gap;
    };

    // "'a', 'b' and 'c'", for the refusals a batch can reach.
    auto name_the_batch = [&]() {
        std::ostringstream names;
        for (std::size_t k = 0; k < nb; ++k) {
            names << (k == 0 ? "'" : (k + 1 == nb ? "' and '" : "', '"))
                  << batch[k]->residual_source;
        }
        names << "'";
        return names.str();
    };

    // ── A residual that is identically ZERO here is not a crossing ───────────
    //
    // CVODE reports a root the instant g reaches exactly 0.0 from a nonzero
    // value — cvRootfind's `ghi == 0 && glo != 0` — and that IS a crossing when
    // g then leaves zero. It is not one when g stays there, and nothing upstream
    // notices: `cvRcheck1` deactivates a root that is zero at (re)init, so it
    // covers a residual that starts on the surface, but not one that reaches
    // exactly zero mid-run and never leaves.
    //
    // `ml_q_learning` (issue #154) is the second kind. It writes a softmax as
    // `1 − 1/(1+exp(−u))`, and `1 + exp(−u)` rounds to exactly 1.0 for every
    // u > 53·ln2 ≈ 36.74, so that factor — and with it the whole residual
    // `alpha·(1−sigma(u))·td` — is EXACTLY 0.0 on an open region of state space
    // that its trajectory enters and never leaves. Underneath the arithmetic
    // the condition `dql > 0` is true everywhere and crosses nothing; the zero
    // is the floating-point evaluation's, not the model's.
    //
    // On such a region ∂g/∂x is zero in every direction, so the implicit
    // function theorem has no denominator and the condition's own boolean is
    // constant over the whole neighbourhood — the rate law sits on one branch
    // and stays there, which is exactly the case that needs no jump. What the
    // ladder above reports for it, though, is a failure to straddle, i.e. the
    // tangency refusal below. That reads a plateau as a surface the trajectory
    // touches, and refuses a run that has nothing wrong with it.
    //
    // Measured rather than assumed, and in the directions the jump would
    // actually differentiate along: zero at x(t*), zero at ±δt along the flow at
    // every nudge the ladder tried, and zero at ±h along every coordinate of the
    // residual's own support (the same h the coordinate probe below uses, so the
    // two agree about what "here" means). Costs nothing on a real crossing —
    // `dt_used != 0` skips it, and the first non-zero evaluation short-circuits.
    auto rides_a_zero_plateau = [&](const NetworkModel::StateSwitch &one, double g_flow_lo,
                                    double g_flow_hi) {
        if (g_flow_lo != 0.0 || g_flow_hi != 0.0 || one.species.empty()) {
            return false;
        }
        sync(x, t_evt);
        if (eval.evaluate(one.residual_expr_idx) != 0.0) {
            return false;
        }
        for (int j : one.species) {
            const double xj = x[static_cast<std::size_t>(j)];
            const double h = 1e-7 * std::max(std::fabs(xj), 1.0);
            xw.assign(x.begin(), x.end());
            for (const double step : {h, -h}) {
                xw[static_cast<std::size_t>(j)] = xj + step;
                sync(xw, t_evt);
                if (eval.evaluate(one.residual_expr_idx) != 0.0) {
                    return false;
                }
            }
        }
        return true;
    };

    if (dt_used == 0.0) {
        std::vector<const NetworkModel::StateSwitch *> crossing;
        crossing.reserve(nb);
        for (std::size_t k = 0; k < nb; ++k) {
            if (!rides_a_zero_plateau(*batch[k], g_before[k], g_after[k])) {
                crossing.push_back(batch[k]);
            }
        }
        if (crossing.size() != nb) {
            sync(x, t_evt); // the probes above left the evaluator off the state
            if (!crossing.empty()) {
                // Whatever is left is re-run on its own rather than carried
                // along: issue #153's "one step straddles all of them or these
                // are not one crossing" test counts the batch it is handed, and
                // a plateau can never be straddled, so leaving one in the batch
                // refuses a set of real crossings for the company it keeps.
                apply_state_switch_sensitivity_jump(cvode_mem, y, ns, t_evt, crossing, s, sens);
            }
            return;
        }
    }

    if (dt_used == 0.0 && nb > 1) {
        // Several residuals, and no one step across the crossing straddles them
        // all — so f⁻ − f⁺ cannot be made to carry every branch change at once,
        // and these are not one surface written several ways (issue #153). Each
        // jump would read f on the two branches of its OWN condition at x(t*),
        // which one shared step cannot separate.
        std::ostringstream msg;
        msg << "Forward sensitivity: " << nb
            << " state-dependent rate-law switches cross at the same instant t=" << t_evt
            << " (residuals " << name_the_batch()
            << "). Stepping the state along the flow by up to " << (dt / kStateSwitchNudgeGrowth)
            << " time units either way ";
        if (best_straddled == 0) {
            msg << "does not carry any of them across zero — the trajectory rides those surfaces "
                   "rather than crossing them";
        } else {
            msg << "carries " << best_straddled
                << " of them across zero but never all together — "
                   "so they are independent conditions meeting at one instant rather than one "
                   "crossing written several ways";
        }
        msg << ", and each jump reads f on the two branches of its OWN condition at x(t*). bngsim "
               "refuses rather than sum jumps that each carry the others' branch change (issue "
               "#150, issue #153). Separate the crossings, or drop sensitivities for this run.";
        throw std::runtime_error(msg.str());
    }

    if (dt_used == 0.0) {
        // The trajectory rides this surface rather than passing through it. Only
        // one question is left: do the branches differ at all? Straddle along a
        // coordinate of the residual's support, where the gradient survives a
        // tangential flow (see the note above).
        sync(x, t_evt);
        const double g_star = eval.evaluate(sw.residual_expr_idx);
        // Pick the best-conditioned coordinate — the one whose own scale moves
        // the residual most — and take a NEWTON step onto the surface, then
        // ±eta past it. A fixed relative step cannot do this job: the residual
        // carries the model's units, and on `ph_lorenz_attractor` (whose
        // condition is `X·Y − beta·Z > 0`, the sign of dZ/dt) a 1e-12-relative
        // nudge moves it by far less than its own distance from zero, so the
        // straddle silently fails and a continuous switch reads as an
        // undecidable one. Scaling by 1/(∂g/∂x_j) removes the units.
        int best = -1;
        double best_gj = 0.0;
        double best_weight = 0.0;
        for (int j : sw.species) {
            const double xj = x[static_cast<std::size_t>(j)];
            const double h = 1e-7 * std::max(std::fabs(xj), 1.0);
            xw.assign(x.begin(), x.end());
            xw[static_cast<std::size_t>(j)] = xj + h;
            sync(xw, t_evt);
            const double g_hi = eval.evaluate(sw.residual_expr_idx);
            xw[static_cast<std::size_t>(j)] = xj - h;
            sync(xw, t_evt);
            const double g_lo = eval.evaluate(sw.residual_expr_idx);
            const double gj = (g_hi - g_lo) / (2.0 * h);
            if (!std::isfinite(gj) || gj == 0.0) {
                continue;
            }
            const double weight = std::fabs(gj) * std::max(std::fabs(xj), 1.0);
            if (weight > best_weight) {
                best_weight = weight;
                best_gj = gj;
                best = j;
            }
        }
        // Straddle at ±eta about the surface, where eta clears the located
        // root's own distance from zero. The displacement is |g*|/|∂g/∂x_j|,
        // which is not always small — a tangency is located in TIME, so g* need
        // not be near zero — and the smooth part of f moves with it. So the gap
        // is measured at eta AND at 2·eta and read by how it SCALES: a gap that
        // doubles with the probe is f varying smoothly over the displacement,
        // i.e. one function on both sides; a gap that does not move is a jump.
        // That discriminator is what `ph_lorenz_attractor` needs — its condition
        // is `X·Y − beta·Z > 0`, the sign of dZ/dt, so both branches are 0 at
        // the surface and its 1.5e-6 relative "gap" is entirely the probe's.
        double f_scale = 0.0;
        double gap1 = 0.0;
        double gap2 = 0.0;
        bool straddled = false;
        if (best >= 0 && std::isfinite(g_star)) {
            const double xj = x[static_cast<std::size_t>(best)];
            const double eta0 =
                std::fabs(g_star) + 1e-9 * std::fabs(best_gj) * std::max(std::fabs(xj), 1.0);
            auto gap_at = [&](double eta, double &scale_out) {
                xw.assign(x.begin(), x.end());
                xw[static_cast<std::size_t>(best)] = xj + (-g_star - eta) / best_gj;
                sync(xw, t_evt);
                const double g_lo = eval.evaluate(sw.residual_expr_idx);
                model.compute_derivs(t_evt, xw.data(), f_minus.data());
                xw[static_cast<std::size_t>(best)] = xj + (-g_star + eta) / best_gj;
                sync(xw, t_evt);
                const double g_hi = eval.evaluate(sw.residual_expr_idx);
                model.compute_derivs(t_evt, xw.data(), f_plus.data());
                straddled = std::isfinite(g_lo) && std::isfinite(g_hi) && g_lo != 0.0 &&
                            g_hi != 0.0 && ((g_lo < 0.0) != (g_hi < 0.0));
                return branch_gap(scale_out);
            };
            double scale2 = 0.0;
            gap2 = gap_at(2.0 * eta0, scale2);
            const bool straddled2 = straddled;
            gap1 = gap_at(eta0, f_scale);
            straddled = straddled && straddled2;
            f_scale = std::max(f_scale, scale2);
        }
        sync(x, t_evt);
        if (straddled && (gap2 <= kStateSwitchContinuousRelTol * f_scale ||
                          gap1 < kStateSwitchGapRatio * gap2)) {
            return; // continuous at its own switch: no jump, and none to refuse
        }
        const double dt_max = dt / kStateSwitchNudgeGrowth;
        std::ostringstream msg;
        msg << "Forward sensitivity: the state-dependent rate-law condition with residual '"
            << sw.residual_source << "' was located as a crossing at t=" << t_evt
            << ", but stepping the state along the flow by up to " << dt_max
            << " time units either way does not change the residual's sign — the trajectory "
               "rides that surface rather than crossing it. ";
        if (straddled) {
            msg << "The two branches of the right-hand side there differ by " << gap1
                << " against a scale of " << f_scale
                << ", and doubling the probe leaves that gap at " << gap2
                << " rather than doubling it, so the jump is real";
        } else {
            msg << "Perturbing the residual's own support does not move it across zero either, "
                << "so the two branches cannot be told apart at all";
        }
        msg << ", while dt*/dθ is unbounded at a tangency — bngsim refuses rather than invent "
               "a saltation term (issue #150). A `piecewise` that is CONTINUOUS at its own "
               "switch (the usual clamp idiom) needs no jump and never reaches this. Move the "
               "threshold off the trajectory's turning point, or drop sensitivities for this "
               "run.";
        throw std::runtime_error(msg.str());
    }
    dt = dt_used;

    // ── Restart just PAST the surface, not on it (issue #82, rate-law side) ──
    // CVODE locates a root only to ~100·ε·(|t| + |h|), so x(t*) lands on either
    // side of g = 0 by a hair — or, as on Smith_BMCSystBiol2013's
    // `PI345P3 > pip3_basal`, exactly ON it, with g(x(t*)) == 0.0 bit-for-bit.
    // Restarting there puts the discontinuity inside the first step after the
    // restart — the one thing stopping at the crossing exists to prevent. CVODES
    // then sizes h from the before-branch RHS while every corrector answers with
    // the after-branch one, the error test fails at every h down to ~1e-17, and
    // the root fires a second time: a jump would be applied twice, which is the
    // same wrong answer with a different sign of the error.
    //
    // Worse, the re-fire need not converge to anything. Sitting exactly on the
    // surface, CVODES restarts at h ≈ ε·|t_end| (~3e-15 on Smith), takes one
    // step too short to move the state by even one ulp, roots on the same
    // crossing, and is re-initialized back to h ≈ ε·|t_end| — an unbounded loop
    // that advances simulated time by ~3.5e-15 per iteration and never returns
    // (issue #187). The scalar run of the same model is 0.02 s; whether the run
    // reaches this state at all depends on where the output grid lands, which is
    // why it read as an `n_points` dependence.
    //
    // `x + δt·f` is the probe point the ladder above VERIFIED is on the after
    // side of every residual that fired at this instant — g there is nonzero and
    // of the opposite sign, which is what the straddle test means — so taking it
    // as the restart state is one explicit Euler step along the flow, an error of
    // O(δt²) in the state with δt at most ~2e-9 of the run's own time scale. The
    // state's own tolerance never sees it; the branch selection does.
    //
    // This is a property of having STOPPED at a crossing, not of having jumped
    // there: a switch that turns out to be continuous applies no saltation term
    // but is standing on exactly the same surface, so it restarts the same way.
    // That is the whole of issue #187 — the continuous return below used to skip
    // this and leave the state on the root.
    auto restart_past_surface = [&]() {
        for (int i = 0; i < ns; ++i) {
            y_data[i] = x[static_cast<std::size_t>(i)] + dt * f0[static_cast<std::size_t>(i)];
        }
        // The evaluator (and everything reading concentrations downstream) must
        // see the state the integration resumes from.
        for (int i = 0; i < ns; ++i) {
            xw[static_cast<std::size_t>(i)] = y_data[i];
        }
        sync(xw, t_evt);
        const int rf = reinit_cvode(cvode_mem, static_cast<sunrealtype>(t_evt), y);
        if (rf != CV_SUCCESS) {
            throw std::runtime_error("CVodeReInit past a state-switch crossing failed: " +
                                     std::to_string(rf));
        }
    };

    // One probe pair for the whole batch: the ladder verified it crosses every
    // residual, so the branch change it reads is already the combined one and
    // there is nothing left to compose (issue #153).
    probe(-dt, g_before);
    model.compute_derivs(t_evt - dt, xw.data(), f_minus.data());
    probe(+dt, g_after);
    model.compute_derivs(t_evt + dt, xw.data(), f_plus.data());

    {
        // Same question on the transversal path, where it is free: f⁻ and f⁺ are
        // already in hand. A continuous switch gets no jump, no dt*/dθ solve, and
        // no transversality refusal — MODEL1006230090 reaches the last of those
        // with a denominator of 1e-15. Both models the batch path was written for
        // are the BNGL signed-rate idiom and leave HERE, before their several
        // residuals ever have to agree on a dt*/dθ.
        double f_scale = 0.0;
        if (branch_gap(f_scale) <= kStateSwitchContinuousRelTol * f_scale) {
            restart_past_surface();
            return;
        }
    }

    // Back to the true crossing state: residual_dtstar differentiates there,
    // and the resumed integration must not see the nudge.
    sync(x, t_evt);

    auto subject_of = [](const NetworkModel::StateSwitch &one) {
        return "the state-dependent rate-law condition with residual '" + one.residual_source +
               "' crosses";
    };
    std::vector<double> tau;
    residual_dtstar(sw.residual_expr_idx, sw.species, subject_of(sw), t_evt, ns, x, f_minus, s,
                    sens, tau);

    // ── One crossing time, or several? (issue #153) ──────────────────────────
    // There IS a jump here, so the batch has to resolve to a single t*(θ) for it
    // to be attributed to. A common factor gives that by construction — h scales
    // the implicit-function numerator and denominator alike where the residual
    // vanishes, so it cancels — while conditions whose crossings move apart do
    // not, and the sum of their jumps is not any one jump. Asking each residual
    // for the vector and comparing is that question put directly, of the
    // quantity that is actually used; it is not implied by their having flipped
    // together (see the note above the constants), and it is also weaker than
    // "one surface" on purpose: two INDEPENDENT crossings that the requested
    // columns move together merge correctly, because (Δ₁ + Δ₂)·τ is then exactly
    // Δ₁·τ₁ + Δ₂·τ₂.
    double tau_scale = 0.0;
    for (int c = 0; c < n_sens; ++c) {
        tau_scale = std::max(tau_scale, std::fabs(tau[static_cast<std::size_t>(c)]));
    }
    std::vector<double> tau_k;
    for (std::size_t k = 1; k < nb; ++k) {
        residual_dtstar(batch[k]->residual_expr_idx, batch[k]->species, subject_of(*batch[k]),
                        t_evt, ns, x, f_minus, s, sens, tau_k);
        double worst = 0.0;
        for (int c = 0; c < n_sens; ++c) {
            worst = std::max(worst, std::fabs(tau_k[static_cast<std::size_t>(c)] -
                                              tau[static_cast<std::size_t>(c)]));
            tau_scale = std::max(tau_scale, std::fabs(tau_k[static_cast<std::size_t>(c)]));
        }
        if (worst > kStateSwitchTauAgreeTol * tau_scale) {
            std::ostringstream msg;
            msg << "Forward sensitivity: " << nb
                << " state-dependent rate-law switches cross at the same instant t=" << t_evt
                << " (residuals " << name_the_batch()
                << "), the right-hand side jumps there, and their crossing times move differently "
                   "with the requested columns: dt*/dθ from '"
                << sw.residual_source << "' and from '" << batch[k]->residual_source
                << "' differ by " << worst << " against a scale of " << tau_scale
                << ". So they are separate crossings that happen to coincide rather than one "
                   "surface written twice, there is no single t*(θ) to shift the flow along, and "
                   "each saltation jump would carry the other's branch change. bngsim refuses "
                   "rather than compose them (issue #150, issue #153). Separate the crossings, or "
                   "drop the parameters that move them apart from sensitivity_params.";
            throw std::runtime_error(msg.str());
        }
    }

    for (int c = 0; c < n_sens; ++c) {
        const double tau_c = tau[static_cast<std::size_t>(c)];
        if (tau_c == 0.0) {
            continue;
        }
        std::vector<double> &col = s[static_cast<std::size_t>(c)];
        for (int i = 0; i < ns; ++i) {
            col[static_cast<std::size_t>(i)] += (f_minus[i] - f_plus[i]) * tau_c;
        }
    }

    // Restart just past the surface — see the note at the lambda above.
    restart_past_surface();
}

// ─── Public interface ────────────────────────────────────────────────────────

CvodeSimulator::CvodeSimulator(NetworkModel &model) : impl_(std::make_unique<Impl>(model)) {}

CvodeSimulator::~CvodeSimulator() = default;

void CvodeSimulator::set_tolerances(double rtol, double atol) {
    impl_->rtol = rtol;
    impl_->atol = atol;
    // A scalar tolerance replaces a per-species one rather than sitting behind
    // it (issue #196). The alternative — leaving a previously set vector in
    // force — would make this call silently do nothing to the tolerance CVODE
    // actually uses.
    impl_->atol_vec.clear();
}

void CvodeSimulator::set_tolerances(double rtol, const std::vector<double> &atol) {
    validate_atol_vector(atol, impl_->model.n_species(), "set_tolerances()");
    impl_->rtol = rtol;
    impl_->atol_vec = atol;
}

void CvodeSimulator::set_max_steps(int max_steps) { impl_->max_steps = max_steps; }

Result CvodeSimulator::run(const TimeSpec &times, const SolverOptions &opts) {
    auto &model = impl_->model;
    const int ns = model.n_species();
    const int n_obs = model.n_observables();

    // Contradictory linear-solver pin (GH #29). Checked ahead of everything
    // else because it is a property of the request, not of the model: silently
    // letting one flag win would hand a benchmark auto-selected numbers under a
    // "forced" label. Both flags exist to measure the auto rule, so a run that
    // does not honor the pin is worse than no run at all.
    if (opts.force_dense_linear_solver && opts.force_sparse_linear_solver) {
        throw std::invalid_argument(
            "force_dense_linear_solver and force_sparse_linear_solver are mutually "
            "exclusive; set at most one. Leave both false for the size/density "
            "auto-selection.");
    }

    // Per-species atol (issue #196). Checked here, alongside the pin above, for
    // the same reason: it is a property of the request. A vector whose length
    // does not match the model is a caller error that has to surface as one —
    // reaching CVODE with it would either read past the end or quietly give
    // species i the number written for species j.
    validate_atol_vector(impl_->resolve_atol_vec(opts), ns, "run()");
    // ...and the extra contract a tracking depth adds on top of it (issue
    // #213): a ceiling to track below, every entry of it strictly positive.
    validate_atol_tracking(opts.atol_track_decades, impl_->resolve_atol_vec(opts), ns, "run()");

    // Algebraic-only model (GH #229): with no ODE state there is no CVODE setup
    // at all, so the whole path lives in its own helper.
    if (ns == 0) {
        return impl_->run_algebraic_only(times);
    }

    // Wall-clock budget. Checked at each outer integration step and at each
    // pending-event sub-step (see the inner while loop below). 0 disables.
    WallClockBudget budget(opts.timeout_seconds);

    double rtol = (opts.rtol > 0) ? opts.rtol : impl_->rtol;
    double atol = (opts.atol > 0) ? opts.atol : impl_->atol;
    const std::vector<double> &atol_v = impl_->resolve_atol_vec(opts);
    int max_steps = (opts.max_steps > 0) ? opts.max_steps : impl_->max_steps;

    // Dense vs sparse (KLU) matrix, including the two force flags (GH #102,
    // GH #29) — see Impl::choose_use_sparse.
    const bool use_sparse = impl_->choose_use_sparse(opts, ns);

    // ─── Warm fast path dispatch (GH #102 reaction kernel) ───────────────────
    // The simple case — no events, no forward sensitivities, no JAX Jacobian —
    // reuses persistent CVODE memory across calls via CVodeReInit (see
    // Impl::run_warm), avoiding the full SUNDIALS/KLU rebuild that otherwise
    // dominates a hybrid splitting loop's small coupling steps. Everything else
    // (events, sensitivities, JAX) takes the cold path below, unchanged. The
    // BNGSIM_NO_WARM_CVODE escape hatch forces the cold path (used by the
    // microbench to measure the warm win, and as a safety valve).
    const bool wants_sensitivity =
        !opts.sensitivity.param_names.empty() || !opts.sensitivity.ic_species_names.empty();
    // Exclude any model that registers CVODE roots — events AND discontinuity
    // triggers (GH #72 time-dependent piecewise rate laws), both of which need
    // the cold path's rootfinding + CVodeReInit-at-crossing machinery.
    const bool has_roots = model.n_events() > 0 || model.n_discontinuity_triggers() > 0;
    const bool warm_eligible = !has_roots && !wants_sensitivity && (opts.jacobian != "jax") &&
                               !std::getenv("BNGSIM_NO_WARM_CVODE");
    if (warm_eligible) {
        return impl_->run_warm(times, opts, use_sparse);
    }

    // Create output time points
    std::vector<double> t_out = times.output_times();
    const int n_out = static_cast<int>(t_out.size());

    // ─── SUNDIALS v7 setup ───────────────────────────────────────────────────
    //
    // RAII guards handle all SUNDIALS cleanup automatically. They are declared
    // here — the declaration order IS the teardown order, reversed — and filled
    // in place by Impl::create_cvode_core, which also seeds y from the model,
    // wires the (cached) codegen RHS into user_data, and applies the tolerances
    // and step limits. codegen_param_buf must outlive the integration loop
    // below: the user_data points into it.
    SunContextGuard ctx;
    NVectorGuard y;
    CvodeMemGuard cvode_mem;
    CvodeUserData user_data{&model};       // user data for RHS callback
    std::vector<double> codegen_param_buf; // RAII: replaces new[]/delete[]
    impl_->create_cvode_core(times, opts, ns, rtol, atol, atol_v, max_steps, ctx, y, cvode_mem,
                             user_data, codegen_param_buf);
    double *y_data = y.data();

    // CVODE status flag, shared by the setup calls below and the stepping loop.
    int flag = CV_SUCCESS;

    // ─── Validate Jacobian strategy ──────────────────────────────────────────
    impl_->validate_jacobian_option(opts, user_data);

    // ─── Linear solver + Jacobian setup ──────────────────────────────────────
    // Build the dense/sparse linear solver, attach it to cvode_mem, and select
    // the Jacobian callback (analytical / colored-FD / dense-codegen / JAX,
    // respecting opts.jacobian). Shared with the warm path so the KLU and
    // Jacobian-strategy selection lives in one place (Impl::setup_linsol_and_jac).
    SUNMatrixGuard A_guard;
    SUNLinSolGuard LS_guard;
    const int desired_linear_solver = impl_->choose_linear_solver_kind(use_sparse, opts, ns);
    impl_->setup_linsol_and_jac(cvode_mem, ctx, y, A_guard, LS_guard, opts, user_data, use_sparse,
                                ns, desired_linear_solver);

    // ─── CVODES forward sensitivity setup ────────────────────────────────────
    // Resolve the requested parameter / initial-condition columns, seed
    // s(0) = ∂y(0)/∂θ, and hand CVODES the sensitivity problem — see
    // Impl::setup_forward_sensitivities. `sens` owns every array CVODES and the
    // codegen sensitivity RHS hold raw pointers into, so it is declared here,
    // alongside the other guards, and lives for the whole integration.
    SensitivityState sens;
    impl_->setup_forward_sensitivities(times, opts, ns, rtol, atol, atol_v, y, cvode_mem, user_data,
                                       sens);

    // Aliases for the names the recording blocks and the integration loop
    // below already use. The jump handlers take `sens` itself, so the method /
    // parameter-index / nominal-value aliases they were the only readers of
    // are gone (GH #135).
    const int n_sens_p = sens.n_p;
    const int n_sens_ic = sens.n_ic;
    const int n_sens = sens.n_total;
    NVectorArrayGuard &yS_guard = sens.yS;
    const std::vector<int> &sens_plist = sens.plist;

    // ─── Allocate result ─────────────────────────────────────────────────────
    // Sizes the trajectory + (optional) sensitivity blocks and names every
    // axis — see Impl::allocate_run_result.

    const int n_func = model.n_functions();

    Result result;
    impl_->allocate_run_result(result, opts, n_out, n_sens_p, n_sens_ic);

    // Temporary observable buffer
    std::vector<double> obs_buf(n_obs);
    const auto ar_copyback = build_assignment_rule_copyback(model);

    // ─── Observable output sensitivities (GH #197) ───────────────────────────
    // d obs_j/dθ = Σ_i c_ji·dx_i/dθ: a runtime chain rule over the CVODES
    // species sensitivities extracted below, no codegen required. The c_ji
    // coefficient table is built once by build_observable_sens_terms (which
    // carries the derivation); the same table drives both the parameter axis
    // and the IC axis, only the source dx/dθ vector differs. Expression
    // (global-function) sensitivities are nonlinear and are left to the codegen
    // stage (#198) — those blocks stay empty here.
    std::vector<ObsSensTerm> obs_sens_terms;
    const bool compute_obs_sens_p = (n_sens_p > 0 && n_obs > 0);
    const bool compute_obs_sens_ic = (n_sens_ic > 0 && n_obs > 0);
    // Scratch outputs laid out [col][obs] so the pointer arrays below hand
    // record_observable_sensitivities* the sens_data[c][j] view it expects.
    std::vector<double> obs_sens_p_buf, obs_sens_ic_buf;
    std::vector<const double *> obs_sens_p_ptrs, obs_sens_ic_ptrs;
    bool obs_sens_has_volume = false;
    if (compute_obs_sens_p || compute_obs_sens_ic) {
        obs_sens_terms = build_observable_sens_terms(model);
        for (const auto &term : obs_sens_terms) {
            if (term.vol_param >= 0) {
                obs_sens_has_volume = true;
                break;
            }
        }
    }
    if (compute_obs_sens_p) {
        result.allocate_observable_sensitivities(n_out, n_obs, n_sens_p);
        obs_sens_p_buf.assign(static_cast<size_t>(n_sens_p) * n_obs, 0.0);
        obs_sens_p_ptrs.resize(n_sens_p);
        for (int p = 0; p < n_sens_p; ++p) {
            obs_sens_p_ptrs[p] = obs_sens_p_buf.data() + static_cast<size_t>(p) * n_obs;
        }
    }
    if (compute_obs_sens_ic) {
        result.allocate_observable_sensitivities_ic(n_out, n_obs, n_sens_ic);
        obs_sens_ic_buf.assign(static_cast<size_t>(n_sens_ic) * n_obs, 0.0);
        obs_sens_ic_ptrs.resize(n_sens_ic);
        for (int k = 0; k < n_sens_ic; ++k) {
            obs_sens_ic_ptrs[k] = obs_sens_ic_buf.data() + static_cast<size_t>(k) * n_obs;
        }
    }
    // Compute + record observable output sensitivities for one output row from
    // its species-sensitivity pointers. sens_ptrs[0..n_sens_p) are the
    // parameter-axis dx/dp columns; sens_ptrs[n_sens_p..n_sens) the IC-axis
    // dx/dY(0) columns — the yS ordering CVodeGetSens fills (seeded above).
    auto record_observable_output_sensitivities = [&](int time_index,
                                                      const double *const *sens_ptrs) {
        if (compute_obs_sens_p) {
            std::fill(obs_sens_p_buf.begin(), obs_sens_p_buf.end(), 0.0);
            for (int p = 0; p < n_sens_p; ++p) {
                const double *ys = sens_ptrs[p];
                double *out = obs_sens_p_buf.data() + static_cast<size_t>(p) * n_obs;
                // (#170 stage 3) The differentiated parameter, so a column that
                // IS a compartment size can pick up the direct ∂c_ji/∂V term
                // below. `obs_sens_has_volume` is false for every model without
                // an amount-valued species on a writable size, which is all of
                // .net and every hOSU=false SBML model — the branch is then
                // hoisted out of the inner loop entirely.
                const int p_idx = sens_plist[p];
                for (const auto &term : obs_sens_terms) {
                    out[term.obs] += term.weight * ys[term.species0];
                    if (obs_sens_has_volume && term.vol_param == p_idx) {
                        out[term.obs] += term.raw_factor * y_data[term.species0];
                    }
                }
            }
            result.record_observable_sensitivities(time_index, obs_sens_p_ptrs.data(), n_obs,
                                                   n_sens_p);
        }
        if (compute_obs_sens_ic) {
            std::fill(obs_sens_ic_buf.begin(), obs_sens_ic_buf.end(), 0.0);
            const double *const *ic_ptrs = sens_ptrs + n_sens_p;
            for (int k = 0; k < n_sens_ic; ++k) {
                const double *ys = ic_ptrs[k];
                double *out = obs_sens_ic_buf.data() + static_cast<size_t>(k) * n_obs;
                for (const auto &term : obs_sens_terms) {
                    out[term.obs] += term.weight * ys[term.species0];
                }
            }
            result.record_observable_sensitivities_ic(time_index, obs_sens_ic_ptrs.data(), n_obs,
                                                      n_sens_ic);
        }
    };

    // ─── Expression (global-function) output sensitivities (GH #198) ──────────
    // Global functions are nonlinear in species/observables/parameters, so
    // d func_m/dθ needs the full chain rule over the expression graph — emitted
    // as compiled C (bngsim_codegen_output_sens) in the same .so as the RHS, so
    // value and derivative never diverge. The compiled evaluator recomputes
    // obs[]/func[] from the current state and folds the per-column state
    // sensitivities (the same sens_ptrs the species/observable blocks use) into
    // func_sens_out[c*N_FUNC + m]; the parameter term is the Kronecker δ plus the
    // derived-parameter chain (sens_plist[c] selects the differentiated parameter;
    // IC columns carry the params.size() sentinel and skip it). Unsupported
    // functions are written NaN by the codegen and rejected by the Result at
    // selection time (never silently wrong). A null symbol (codegen inactive or a
    // declined model) leaves the blocks empty — an expression: selector then
    // raises rather than reading zeros.
    const bool compute_expr_sens =
        (user_data.codegen_output_sens_fn != nullptr && n_func > 0 && n_sens > 0);
    const bool compute_expr_sens_p = compute_expr_sens && n_sens_p > 0;
    const bool compute_expr_sens_ic = compute_expr_sens && n_sens_ic > 0;
    std::vector<double> func_sens_buf;          // [col][func], filled by the codegen
    std::vector<const double *> func_sens_ptrs; // per-column views for recording
    if (compute_expr_sens_p) {
        result.allocate_expression_sensitivities(n_out, n_func, n_sens_p);
    }
    if (compute_expr_sens_ic) {
        result.allocate_expression_sensitivities_ic(n_out, n_func, n_sens_ic);
    }
    if (compute_expr_sens) {
        func_sens_buf.assign(static_cast<size_t>(n_sens) * n_func, 0.0);
        func_sens_ptrs.resize(n_sens);
        for (int c = 0; c < n_sens; ++c) {
            func_sens_ptrs[c] = func_sens_buf.data() + static_cast<size_t>(c) * n_func;
        }
    }
    // Compute + record expression output sensitivities for one output row from
    // all of its species-sensitivity columns. The codegen fills func_sens_buf in
    // [col][func] order; the parameter columns [0, n_sens_p) record into the
    // parameter block and the IC columns [n_sens_p, n_sens) into the IC block.
    auto record_expression_output_sensitivities = [&](int time_index, double t_row,
                                                      const double *const *sens_ptrs) {
        if (!compute_expr_sens) {
            return;
        }
        std::fill(func_sens_buf.begin(), func_sens_buf.end(), 0.0);
        user_data.codegen_output_sens_fn(
            t_row, y_data, user_data.codegen_so_data.param_values, sens_ptrs, sens_plist.data(),
            n_sens, /*obs_sens_out=*/nullptr, func_sens_buf.data(), &user_data.codegen_so_data);
        if (compute_expr_sens_p) {
            result.record_expression_sensitivities(time_index, func_sens_ptrs.data(), n_func,
                                                   n_sens_p);
        }
        if (compute_expr_sens_ic) {
            result.record_expression_sensitivities_ic(time_index, func_sens_ptrs.data() + n_sens_p,
                                                      n_func, n_sens_ic);
        }
    };

    // ─── Event rootfinding setup ─────────────────────────────────────────────
    // Register root functions for event trigger edge detection.
    // g_i(t, y) = trigger_i(t, y) - 0.5  (zero-crossing detects false→true)
    const int n_events = model.n_events();
    std::vector<bool> trigger_was_true(n_events, false);

    // ─── Random tie-break among equal-priority simultaneous events (GH #242) ──
    // SBML L3v2 §4.11.6: among events firing at the same instant with the SAME
    // (maximum) priority, one is chosen at random each round. This per-run RNG
    // (mirroring the ssa_simulator.cpp mt19937_64 pattern) supplies that choice
    // in process_firing_batch's drain. It is drawn from ONLY at a genuine tie
    // (≥2 not-done instances share the max priority), so a model with no equal-
    // priority simultaneity never advances the stream and stays byte-identical
    // to the old lowest-index behavior; a model WITH ties is reproducible for a
    // fixed opts.event_seed (default fixed → deterministic out of the box).
    std::mt19937_64 event_rng(opts.event_seed);

    // ─── Chatter guard state (GH #95) ────────────────────────────────────────
    // event_dormant[i] != 0 suppresses event i's trigger root (see
    // CvodeUserData::event_dormant and root_fn). An event is flagged dormant
    // after it fires CHATTER_LIMIT times in a row where each fire BOTH advances
    // simulated time negligibly AND changes the state by less than the
    // integrator tolerance — the signature of a non-negativity clamp re-firing
    // on floating-point noise once its variable has decayed far below atol
    // (RoadRunner keeps such a variable cleanly positive and never fires). The
    // dual criterion keeps this inert for genuine recurring events, which move
    // the state meaningfully and fire with real time gaps. A dormant event is
    // re-armed (at an output point) if its assigned species climb back above the
    // atol noise floor — so a real recovery is never permanently missed, while a
    // decay-to-zero clamp stays suppressed. See the detection and re-arm blocks
    // in the integration loop below.
    constexpr int CHATTER_LIMIT = 50;
    constexpr double REARM_TOL_FACTOR = 1024.0;
    std::vector<char> event_dormant(n_events, 0);
    std::vector<double> event_last_fire_time(n_events, 0.0);
    std::vector<int> event_chatter_count(n_events, 0);
    std::vector<double> chatter_y_before; // snapshot to size on first firing
    user_data.event_dormant = (n_events > 0) ? event_dormant.data() : nullptr;

    // Discontinuity triggers (GH #72): time-dependent inequality conditions
    // (from piecewise assignment rules / rate laws) registered as additional
    // CVODE roots AFTER the event roots, in gout indices [n_events, n_roots).
    // They carry no state assignment — a crossing only forces a CVodeReInit so
    // the integrator stops at the discontinuity instead of stepping over a
    // narrow forcing pulse. n_disc == 0 for any model without time-dependent
    // piecewise, in which case the root machinery is bit-for-bit unchanged.
    const int n_disc = model.n_discontinuity_triggers();

    // State-dependent rate-law switches (issue #150): the residual of each
    // `piecewise(..., X < c, ...)` condition, registered as a further root AFTER
    // the discontinuity ones, in gout indices [n_events + n_disc, n_roots). Two
    // things come out of the stop it forces: the crossing is LOCATED rather than
    // chased (which is what keeps a sensitivity run out of issue #82's collapsed
    // step), and there is a place to apply the saltation jump dx/dθ takes there.
    //
    // Registered only when the run asks for sensitivities. Those are the runs the
    // missing jump is wrong for, and leaving the root set alone otherwise keeps
    // every plain trajectory in the corpus bit-for-bit unchanged; adding these
    // roots unconditionally is a trajectory-accuracy change of its own and wants
    // its own measurement.
    //
    // Deduplicated by the residual's text rather than the condition's, so
    // `X<1` and `X<=1` — the same crossing, two spellings — are one root and one
    // jump instead of two coincident ones the composition would double-count.
    std::vector<const NetworkModel::StateSwitch *> state_switches;
    std::vector<int> state_switch_roots;
    if (wants_sensitivity && !opts.sensitivity.state_switch_conditions.empty()) {
        std::unordered_set<std::string> seen_residual;
        for (const std::string &cond : opts.sensitivity.state_switch_conditions) {
            const NetworkModel::StateSwitch *sw = model.state_switch(cond);
            if (sw == nullptr || !seen_residual.insert(sw->residual_source).second) {
                continue;
            }
            state_switches.push_back(sw);
            state_switch_roots.push_back(sw->residual_expr_idx);
        }
    }
    const int n_state_switch = static_cast<int>(state_switch_roots.size());
    if (n_state_switch > 0) {
        user_data.state_switch_roots = &state_switch_roots;
    }
    const int n_roots = n_events + n_disc + n_state_switch;

    // Delay queue for events with delay > 0.
    // When a delayed event fires, we store it here and apply when
    // t >= t_fire + delay. If persistent=false and trigger reverts to
    // false before delay expires, the pending event is cancelled.
    //
    // SBML L3 useValuesFromTriggerTime semantics: when the event has
    // useValuesFromTriggerTime=true (the default), assignment RHS values are
    // evaluated at trigger time and the resulting numbers are held in
    // `frozen_values` until firing time. When false, RHS is evaluated at
    // firing time and `frozen_values` is empty.
    struct PendingEvent {
        int event_idx;                     // index into model.events()
        double apply_time;                 // t_fire + delay
        std::vector<double> frozen_values; // size = n_assignments when frozen
    };
    std::vector<PendingEvent> pending_events;

    // ─── Helper: process a batch of rising-edge fires at time t ────────
    //
    // SBML L3v2 §4.11.6 simultaneous-event execution algorithm. The batch is a
    // dynamic MULTISET of execution instances (one per rising edge), NOT a fixed
    // list — after each immediate fire the state may push another event's
    // trigger false→true, which joins the SAME instant's batch (the immediate,
    // delay-0 cascade). Concretely:
    //
    //  1. Seed one instance per event in ``firing_in`` (the root-detected /
    //     T0 risers). ``prev[]`` baseline = the caller's ``trigger_was_true``.
    //     Each instance freezes its useValuesFromTriggerTime=true assignment
    //     RHS values and resolves its delay at its own trigger time — for the
    //     seed instances that is the pre-batch state (captured here), so all
    //     events sharing the trigger time read the same state.
    //  2. Drain highest-priority-first (priorities re-evaluated per fire, SBML
    //     §3.4.6 — a state-dependent priority like ``2*S2`` is not stable across
    //     fires that move its referents; test 00934). Among the not-done
    //     instances at the SAME maximum priority, pick ONE AT RANDOM via
    //     ``event_rng`` (§4.11.6, GH #242) — the draw happens only at a genuine
    //     ≥2-way tie, so distinct priorities stay deterministic and a tie-free
    //     model never advances the RNG stream.
    //  3. A picked delayed event (delay>0) is queued to ``pending_events`` (no
    //     state change → no cascade). A picked immediate event applies its
    //     assignments, then we refresh observables/functions and RE-CHECK ALL
    //     triggers against ``prev``: a rising edge enqueues a NEW instance
    //     (capturing its UVFTT values + delay now, at this sub-instant); a
    //     falling edge cancels every not-done non-persistent instance of that
    //     event (§4.11.3 — a non-persistent event whose trigger lapses before
    //     it executes does not fire). ``prev`` advances to the settled state.
    //  4. ``CASCADE_LIMIT`` backstops an algebraic loop (A arms B arms A …).
    //  5. On exit ``trigger_was_true`` is synced to the settled ``prev`` so the
    //     post-batch delayed cascade (cascade_triggered_events) is a no-op this
    //     drain subsumes, and CVODE root detection resumes from the right edge.
    //
    // Returns true if any immediate event modified y_data. Defined at function
    // scope so both the t=0 init and the runtime CV_ROOT_RETURN handler can call
    // it. No-op when firing_in is empty.
    auto &eval_ref_outer = model.evaluator();
    auto &sp_vec_outer = const_cast<std::vector<Species> &>(model.species());
    const auto &events_outer = model.events();

    // One scheduled execution of an event within a same-instant batch. Its
    // frozen UVFTT values and resolved delay are captured at its trigger time
    // (batch entry for a seed instance, the firing sub-instant for a cascade
    // instance). ``done`` marks it executed, queued-as-delayed, or cancelled.
    struct ExecInstance {
        int event_idx;
        std::vector<double> snapshot_vals; // UVFTT frozen RHS (empty if !UVFTT)
        double delay;
        bool done = false;
    };

    // Guard against a same-instant algebraic loop (mutually-arming events).
    // Far above any legitimate cascade depth (00978 fires ~11; 01533 ~106).
    constexpr int CASCADE_LIMIT = 100000;

    auto process_firing_batch = [&](double t_now, const std::vector<int> &firing_in) -> bool {
        if (firing_in.empty())
            return false;

        // Capture an event's UVFTT snapshot + resolved delay AT THE CURRENT
        // state (its trigger time). Used both to seed the batch and to enqueue
        // a cascade instance mid-drain.
        auto make_instance = [&](int ei) -> ExecInstance {
            ExecInstance inst;
            inst.event_idx = ei;
            const auto &ev = events_outer[ei];
            if (ev.use_values_from_trigger_time) {
                inst.snapshot_vals.reserve(ev.assignments.size());
                for (const auto &[sp_idx0, val_expr_idx] : ev.assignments) {
                    (void)sp_idx0;
                    inst.snapshot_vals.push_back(eval_ref_outer.evaluate(val_expr_idx));
                }
            }
            double d = ev.delay;
            if (ev.delay_expr_idx >= 0) {
                d = eval_ref_outer.evaluate(ev.delay_expr_idx);
                if (d < 0.0)
                    d = 0.0;
            }
            inst.delay = d;
            return inst;
        };

        // Local trigger baseline; synced back into trigger_was_true on exit.
        std::vector<bool> prev = trigger_was_true;

        // Seed one instance per root-detected riser (UVFTT/delays captured at
        // the shared pre-batch trigger-time state).
        std::vector<ExecInstance> queue;
        queue.reserve(firing_in.size());
        for (int ei : firing_in) {
            queue.push_back(make_instance(ei));
        }

        bool any_immediate = false;
        int fires = 0;

        auto eval_pri = [&](const ExecInstance &inst) -> double {
            const auto &ev = events_outer[inst.event_idx];
            return (ev.priority_expr_idx >= 0) ? eval_ref_outer.evaluate(ev.priority_expr_idx)
                                               : static_cast<double>(ev.priority);
        };

        while (true) {
            // Collect the not-done instances sharing the MAXIMUM priority.
            // Iterating in increasing index keeps `ties` index-ordered, so the
            // single-candidate case reproduces the old lowest-index pick without
            // touching the RNG.
            double best_pri = 0.0;
            std::vector<size_t> ties;
            for (size_t k = 0; k < queue.size(); ++k) {
                if (queue[k].done)
                    continue;
                double pk = eval_pri(queue[k]);
                if (ties.empty() || pk > best_pri) {
                    best_pri = pk;
                    ties.clear();
                    ties.push_back(k);
                } else if (pk == best_pri) {
                    ties.push_back(k);
                }
            }
            if (ties.empty())
                break;

            // Random tie-break among equal-max-priority instances (GH #242).
            // A single candidate consumes no randomness (byte-identical to the
            // old deterministic drain for tie-free models).
            size_t k;
            if (ties.size() == 1) {
                k = ties[0];
            } else {
                std::uniform_int_distribution<size_t> pick(0, ties.size() - 1);
                k = ties[pick(event_rng)];
            }
            queue[k].done = true;

            if (++fires > CASCADE_LIMIT) {
                throw std::runtime_error(
                    "Event cascade exceeded CASCADE_LIMIT at t=" + std::to_string(t_now) +
                    " (same-instant events appear to arm each other in an algebraic loop).");
            }

            const auto &ev = events_outer[queue[k].event_idx];
            double delay_now = queue[k].delay;

            if (delay_now > 0.0) {
                // Delayed: queue for a future apply_time. No state change now,
                // so no cascade re-check follows.
                PendingEvent pe;
                pe.event_idx = queue[k].event_idx;
                pe.apply_time = t_now + delay_now;
                if (ev.use_values_from_trigger_time) {
                    pe.frozen_values = queue[k].snapshot_vals;
                }
                pending_events.push_back(std::move(pe));
                continue;
            }

            // Immediate fire. Apply RHS values: snapshot for UVFTT=true;
            // otherwise evaluate now against the (possibly mutated) state.
            const auto &assigns = ev.assignments;
            std::vector<double> nv(assigns.size());
            if (ev.use_values_from_trigger_time) {
                nv = queue[k].snapshot_vals;
            } else {
                for (size_t a = 0; a < assigns.size(); ++a) {
                    nv[a] = eval_ref_outer.evaluate(assigns[a].second);
                }
            }
            for (size_t a = 0; a < assigns.size(); ++a) {
                int sp_idx0 = assigns[a].first;
                y_data[sp_idx0] = nv[a];
                sp_vec_outer[sp_idx0].concentration = nv[a];
            }
            any_immediate = true;

            // Refresh so the priority re-eval, cascade re-check, and any
            // rateOf-bearing trigger below see the post-fire state.
            model.update_observables(y_data);
            model.evaluate_functions(t_now);
            if (model.uses_rateof()) {
                model.refresh_rateof_derivs(t_now, y_data);
            }

            // Re-check every trigger against `prev`: a rising edge is a
            // same-instant cascade fire (enqueue a fresh instance); a falling
            // edge cancels the not-done non-persistent instances of that event.
            for (int ei = 0; ei < n_events; ++ei) {
                double tv = eval_ref_outer.evaluate(events_outer[ei].trigger_expr_idx);
                bool now_true = (tv > 0.5);
                bool was = prev[ei];
                // Skip a chatter-dormant event (GH #95): like the root and the
                // post-batch cascade paths, the same-instant drain must not
                // re-arm an event the chatter guard is stepping over, or a
                // multi-event model can re-fire it here and defeat suppression.
                if (now_true && !was && !event_dormant[ei]) {
                    // Forward sensitivity has no jump for this fire (issue
                    // #144). A cascade instance executes LATER in the same
                    // instant, reading a state that already jumped, while
                    // apply_event_sensitivity_jump is keyed on the caller's
                    // seed list and takes every derivative at the pre-batch x⁻.
                    // The rows this instance assigns would keep the sensitivity
                    // of the value they held before it — silently stale, which
                    // is the GH #205 hazard this whole area exists to remove.
                    // Composing two jumps at one instant is real work and
                    // nothing in the corpus needs it yet, so refuse.
                    //
                    // Unreachable before issue #144: a cascade instance is a
                    // trigger that an assignment made true, which needs a
                    // state-dependent trigger, and those were refused outright.
                    if (sens.n_total > 0) {
                        throw std::runtime_error(
                            "Forward sensitivity: event '" + events_outer[ei].id +
                            "' was triggered at t=" + std::to_string(t_now) +
                            " by another event's assignment, not by a crossing the integrator "
                            "located (SBML \"events triggering events\"). That is a second "
                            "state jump at the same instant, and bngsim has no sensitivity "
                            "jump to compose with the first, so the columns for the species it "
                            "assigns would be silently stale (GH #205). Separate the fires in "
                            "time, or drop sensitivities for this run (issue #144).");
                    }
                    queue.push_back(make_instance(ei));
                } else if (!now_true && was) {
                    if (!events_outer[ei].persistent) {
                        for (auto &inst : queue) {
                            if (!inst.done && inst.event_idx == ei) {
                                inst.done = true;
                            }
                        }
                    }
                }
                prev[ei] = now_true;
            }
        }

        // Settle: the batch has converged. Publish the final trigger states so
        // the delayed cascade_triggered_events call after us finds no new edge
        // (this drain already handled every same-instant rise) and CVODE's next
        // root pass measures crossings from the correct baseline.
        for (int ei = 0; ei < n_events; ++ei) {
            trigger_was_true[ei] = prev[ei];
        }

        return any_immediate;
    };

    // ─── Helper: fire events newly triggered by an event assignment ────────
    //
    // SBML L3 §3.4: after a batch of events executes, every trigger must be
    // re-checked; any that just transitioned false→true *because an assignment
    // moved the state it depends on* is itself a rising-edge fire ("events
    // triggering events"). The CVODE root finder cannot see these — it only
    // detects zero-crossings during continuous integration, never the discrete
    // jump an assignment makes. So after assignments are applied (at a root
    // batch or a delayed apply) we evaluate all triggers against the
    // post-assignment state, compare against trigger_was_true (the
    // pre-assignment baseline), and route the rising-edge set through
    // process_firing_batch (which honors UVFTT/priority/delay and the §4.11.6
    // random tie-break).
    //
    // Drives 01754/01758/01759 (GH #233): a persistent delayed event whose
    // assignment re-satisfies its own (and a sibling's) trigger, sustaining a
    // self-perpetuating delayed-event chain the root finder alone freezes after
    // one round. Also drives 01590's same-instant monitor (GH #242): the delayed
    // apply of one competing event fires the delay-0 maxcheck that records the
    // running max |Q−R|. process_firing_batch's own CASCADE_LIMIT backstops an
    // algebraic loop; the caller ReInits CVODE unconditionally for the
    // assignment that called us.
    //
    // Both delayed AND same-instant (delay-0) assignment-induced re-triggers are
    // handled: the rising-edge set is routed through process_firing_batch, which
    // fires the immediate ones (with the §4.11.6 seed-keyed random tie-break, GH
    // #242) and queues the delayed ones. Firing immediate re-triggers used to be
    // deferred because a DETERMINISTIC tie-break made competing same-priority
    // non-persistent events (the RandomEventExecution family) monotonically
    // diverge and spuriously trip their divergence monitors; the random tie-break
    // removes that hazard, so the immediate cascade is now correct here too.
    // This is what lets a delayed event's assignment fire a same-instant monitor
    // (01590's maxcheck) or another delay-0 event. In the root-batch path this is
    // a no-op — process_firing_batch already settled trigger_was_true before we
    // are called; only the delayed-apply path reaches here with fresh risers.
    //
    // trigger_was_true is advanced to now_true for EVERY event first (matching
    // the root-batch caller's contract that a seeded event's baseline is true),
    // then process_firing_batch re-settles it across the cascade. A falling edge
    // therefore re-arms; a persistent delayed event that re-satisfies its own or
    // a sibling's trigger schedules the next round (01754/01758/01759 chains).
    auto cascade_triggered_events = [&](double t_now) {
        model.update_observables(y_data);
        model.evaluate_functions(t_now);
        if (model.uses_rateof()) {
            model.refresh_rateof_derivs(t_now, y_data);
        }
        std::vector<int> risers;
        for (int ei = 0; ei < n_events; ++ei) {
            double v = eval_ref_outer.evaluate(events_outer[ei].trigger_expr_idx);
            bool now_true = (v > 0.5);
            // A chatter-dormant event (GH #95) is being stepped over: root_fn
            // suppresses its continuous root, so it must ALSO be excluded from
            // the same-instant cascade. Otherwise it re-fires here on every root
            // batch and the noise-floor suppression is defeated — the #242
            // cascade rework began firing immediate risers (was: delayed only),
            // which reintroduced BIOMD711's Zeno clamp chatter. Its trigger
            // baseline is still advanced so re-arm resumes from a consistent
            // state.
            if (now_true && !trigger_was_true[ei] && !event_dormant[ei]) {
                risers.push_back(ei);
            }
            trigger_was_true[ei] = now_true;
        }
        if (!risers.empty()) {
            // No sensitivity guard here: process_firing_batch's own drain
            // subsumes every *immediate* same-instant rise and refuses it when
            // sensitivities are active, so a riser reaching this point is one
            // the drain left — a delayed event, which the upstream delay guard
            // already refuses for sensitivities.
            process_firing_batch(t_now, risers);
        }
    };

    // ─── Switch-time crossings to jump across (issue #48) ────────────────────
    // Which recorded crossings this run can actually reach, and the pre-sized
    // scratch the jump differentiates into. The jump itself — and why it is the
    // ENTIRE switch-time gradient — lives with Impl::apply_switch_sensitivity_jump.
    std::vector<const SwitchTimeSens *> switch_list;
    if (wants_sensitivity && n_sens_p > 0 && !opts.sensitivity.switch_times.empty()) {
        for (const auto &sw : opts.sensitivity.switch_times) {
            // A crossing outside the reported window contributes nothing, and a
            // record whose width doesn't match this run's parameter columns is
            // stale — drop both rather than index out of range. The window is
            // half-open: a crossing ON t_end still jumps (the recorded column is
            // right-continuous, as at any interior crossing), while one on
            // t_start would jump before the run's own initial recording.
            if (sw.t_star > t_out.front() && sw.t_star <= t_out.back() &&
                sw.dtstar_dp.size() == static_cast<size_t>(n_sens_p)) {
                switch_list.push_back(&sw);
            }
        }
    }
    size_t next_switch = 0; // index into switch_list of the next crossing
    // Time tolerance for "reached / already past" a crossing, scaled to the run
    // horizon so it stays meaningful for both day-scale and second-scale models.
    const double switch_t_eps = 1e-9 * std::max(1.0, std::fabs(t_out.back() - t_out.front()));
    SwitchJumpScratch sw_scratch;
    if (!switch_list.empty()) {
        sw_scratch.resize(ns);
    }

    // ─── Fixed time-discontinuity crossings to land on (issue #305) ──────────
    // Same window filter as the issue #48 list above, minus any crossing that
    // list already stops at — a #48 crossing carries a sensitivity jump this
    // one must not pre-empt, and stopping twice at one instant would leave the
    // jump keyed on a t_ret the stop already consumed.
    std::vector<double> crossing_stops;
    for (double t_cross : opts.crossing_stop_times) {
        if (!(t_cross > t_out.front() && t_cross <= t_out.back())) {
            continue;
        }
        bool claimed_by_switch = false;
        for (const auto *sw : switch_list) {
            if (std::fabs(sw->t_star - t_cross) <= switch_t_eps) {
                claimed_by_switch = true;
                break;
            }
        }
        if (!claimed_by_switch) {
            crossing_stops.push_back(t_cross);
        }
    }
    size_t next_crossing = 0; // index into crossing_stops of the next one ahead

    if (n_roots > 0) {
        // Register the event + discontinuity roots (Impl::register_roots).
        impl_->register_roots(cvode_mem, ctx, n_roots, n_disc);

        // Two-phase t=0 trigger initialization (SBML L3 §3.4.5):
        //
        //   (1) Seed trigger_was_true from each event's `initialValue`
        //       attribute — this represents the trigger's *presumed prior
        //       state* just before simulation starts.
        //   (2) Evaluate the actual t=0 trigger expression. An event whose
        //       presumed prior state was false but whose actual t=0 value
        //       is true is a rising-edge fire AT t=0 — it is fired here,
        //       in priority order, before the initial state is recorded.
        //   (3) Update trigger_was_true to the *actual* t=0 value so that
        //       subsequent CVODE root crossings are detected correctly.
        //
        // The previous code seeded trigger_was_true purely from the t=0
        // expression value, ignoring `initialValue` entirely; that
        // suppressed legitimate t=0 fires for events declared with
        // `initialValue=false`.
        {
            model.update_observables(y_data);
            model.evaluate_functions(times.t_start);
            // GH #106: the t=0 trigger init runs OUTSIDE the root function, so
            // refresh the live rateOf buffer here too — otherwise a trigger
            // reading rateOf(species) sees the zero-initialized current_derivs
            // and can spuriously fire at t=0 (e.g. `rateOf(A) > -1` would be
            // 0 > -1 = true before the first derivative is ever computed).
            if (model.uses_rateof()) {
                model.compute_derivs(times.t_start, y_data, user_data.rateof_root_scratch.data());
            }

            std::vector<int> t0_firing;
            t0_firing.reserve(n_events);
            for (int i = 0; i < n_events; ++i) {
                trigger_was_true[i] = events_outer[i].initial_value;
                double val = eval_ref_outer.evaluate(events_outer[i].trigger_expr_idx);
                bool now_true = (val > 0.5);
                if (now_true && !trigger_was_true[i]) {
                    t0_firing.push_back(i);
                }
                trigger_was_true[i] = now_true;
            }

            // Snapshot x⁻ and s⁻ before the assignments mutate y_data and
            // before CVodeReInit, so the sensitivity jump (GH #212) can
            // differentiate at the pre-event state and consume valid pre-event
            // sensitivities. Cheap and only taken when sensitivities are active.
            std::vector<double> t0_x_minus;
            std::vector<std::vector<double>> t0_s_minus;
            if (wants_sensitivity && !t0_firing.empty()) {
                t0_x_minus.assign(y_data, y_data + ns);
                t0_s_minus = impl_->capture_event_sens(cvode_mem, ns,
                                                       static_cast<double>(times.t_start), sens);
            }

            bool t0_immediate_fired = process_firing_batch(times.t_start, t0_firing);

            if (t0_immediate_fired) {
                // Refresh observables/functions after t=0 fires so the
                // recorded initial state and the integrator setup both
                // see the post-event values.
                model.update_observables(y_data);
                model.evaluate_functions(times.t_start);
                if (model.uses_rateof()) {
                    model.compute_derivs(times.t_start, y_data,
                                         user_data.rateof_root_scratch.data());
                }
                // Re-evaluate trigger_was_true against post-fire state so
                // any event that falsified its own trigger can re-arm.
                for (int ei = 0; ei < n_events; ++ei) {
                    double v = eval_ref_outer.evaluate(events_outer[ei].trigger_expr_idx);
                    trigger_was_true[ei] = (v > 0.5);
                }
            }

            // If a t=0 immediate event mutated the state vector, tell CVODE
            // to restart from the modified y. (CVodeInit was called with
            // the original y above; without ReInit, internal state is
            // inconsistent with our writes.)
            if (t0_immediate_fired) {
                int reinit_flag = impl_->reinit_cvode(cvode_mem, times.t_start, y);
                if (reinit_flag != CV_SUCCESS) {
                    throw std::runtime_error("CVodeReInit after t=0 event failed: " +
                                             std::to_string(reinit_flag));
                }
                // Jump dx/dp across the t=0 event and re-seed CVODES sensitivity
                // vectors (GH #212). No-op unless sensitivities are active.
                impl_->apply_event_sensitivity_jump(opts, cvode_mem, ns,
                                                    static_cast<double>(times.t_start), t0_firing,
                                                    t0_x_minus, t0_s_minus, sens,
                                                    /*at_run_start=*/true);
            }
        }
    }

    // ─── Record initial state ────────────────────────────────────────────────

    model.update_observables(y_data);
    model.evaluate_functions(times.t_start);
    for (int j = 0; j < n_obs; ++j) {
        obs_buf[j] = model.observables()[j].total;
    }
    result.record(0, times.t_start, y_data, obs_buf.data());

    // Record function values at t=0
    if (n_func > 0) {
        result.record_expressions(0, model.function_value_cache().data());
    }

    // Record initial sensitivities. Param cols are zero at t=0 for a fresh
    // start (or the carried-over dx/dθ seed under carry_sensitivities, GH
    // #210); IC cols are e_k at t=0 (seeded above).
    if (n_sens > 0) {
        std::vector<const double *> sens_ptrs(n_sens);
        for (int s = 0; s < n_sens; ++s) {
            sens_ptrs[s] = N_VGetArrayPointer(yS_guard[s]);
        }
        if (n_sens_p > 0) {
            result.record_sensitivities(0, sens_ptrs.data(), ns, n_sens_p);
        }
        if (n_sens_ic > 0) {
            result.record_sensitivities_ic(0, sens_ptrs.data() + n_sens_p, ns, n_sens_ic);
        }
        record_observable_output_sensitivities(0, sens_ptrs.data());
        record_expression_output_sensitivities(0, times.t_start, sens_ptrs.data());
    }

    // ─── Integration loop ────────────────────────────────────────────────────
    //
    // Each output point t_out[i] is reached via an inner while-loop that
    // stops at the *earlier* of (a) the next pending delayed-event
    // apply_time inside (t_now, t_out[i]] or (b) t_out[i] itself. Without
    // the inner loop, CVODE integrates straight to t_out[i] and we can
    // only apply pending events at sample times — that loses the
    // post-apply decay between apply_time and the sample. For an event
    // with delay 1.0 firing at t=2.303 with sample step 0.1, the apply
    // would land at t=3.4 instead of t=3.303, missing 0.097 of decay.
    sunrealtype t_now = static_cast<sunrealtype>(times.t_start);

    // Steady-state early-termination buffers and tolerance. Matches BNG2.pl
    // ``run_network -c`` semantics: after each output point is recorded,
    // compute ``||f(t,y)||_2 / n_species`` and stop integrating once it
    // falls below ``ss_tol``. ``ss_tol`` defaults to the integrator atol — the
    // SCALAR one, also when a per-species atol is in force (issue #196): the
    // criterion is a single norm over every species and has no per-species
    // reading to take. A caller running with a vector atol and wanting the
    // early stop should say what "steady" means with steady_state_tol.
    const bool check_ss = opts.steady_state;
    const double ss_tol = (opts.steady_state_tol > 0.0) ? opts.steady_state_tol : atol;
    std::vector<double> ss_derivs;
    if (check_ss) {
        ss_derivs.resize(ns);
    }
    int last_recorded_index = 0;
    bool ss_reached = false;
    double ss_residual_last = 0.0;

    // ─── Early refresh points for the sensitivity error floor (issue #177) ───
    //
    // The floor set at t=0 is right for every row whose ∂f/∂p already has terms
    // there, but a row whose species starts empty has none yet — its terms
    // appear as the trajectory fills it in, and by the first *output* point
    // CVODES may already have crawled to h~1e-13, where raising a tolerance no
    // longer recovers the step controller (that is exactly how the "relax on a
    // demonstrated stall" attempt on this issue died). So refresh on a geometric
    // ladder of early times as well, decades below the horizon, while h is still
    // near where the solver started.
    //
    // These rungs are NOT extra CVode() output targets. They were, and that was
    // not free: CVODES sizes its first step from the distance to the first tout
    // (cvHin's hub = 0.1·tdist), so an early target rewrites h0 and with it the
    // whole step sequence — measured on 309 corpus models as moving nearly every
    // one of them, which is why the ladder shipped gated on the floor binding at
    // t=0 and so never ran for the models issue #183 is about. Single-stepping to
    // the SAME tout is free by construction: CV_ONE_STEP takes exactly the steps
    // CV_NORMAL would and simply hands each one back. Re-measured that way, the
    // only models that move are the ones whose floor genuinely binds (9 of 309),
    // because a refresh that finds no row above its static floor never calls
    // CVodeSensSVtolerances at all.
    std::vector<double> floor_times;
    const char *ladder_env = std::getenv("BNGSIM_SENS_FLOOR_LADDER");
    const std::string ladder_mode(ladder_env ? ladder_env : "");
    const bool ladder_off = ladder_mode == "off";
    double ladder_first = 1e-6;
    if (const char *lf = std::getenv("BNGSIM_SENS_FLOOR_LADDER_FIRST")) {
        const double v = std::atof(lf);
        if (v > 0.0 && v < 1.0) {
            ladder_first = v;
        }
    }
    // Rungs per decade, not one per decade. A decade-spaced ladder leaves
    // Smith's k8 a gap from t=2.4 to t=24 with the transition at t≈20 inside it,
    // and the run dies in the gap. Halving is where the ladder stops being the
    // variable: 14/16 columns at ratio 10, 16/16 at 3.16, 2 and 1.5 alike. That
    // monotonicity is the point — a rule that needs a particular spacing is
    // fitted to it, and this one only needed the mark below to become one.
    // Free in wall clock (measured at ±0.5 ms across the corpus): the rungs are
    // single steps CVODES was taking anyway, and a refresh that changes nothing
    // never calls CVodeSensSVtolerances.
    double ladder_ratio = 2.0;
    if (const char *lr = std::getenv("BNGSIM_SENS_FLOOR_LADDER_RATIO")) {
        const double v = std::atof(lr);
        if (v > 1.0) {
            ladder_ratio = v;
        }
    }
    if (sens.floor_active && !ladder_off) {
        const double span = t_out.empty() ? 0.0 : (t_out.back() - times.t_start);
        for (double f = ladder_first; f < 1.0; f *= ladder_ratio) {
            const double tf = times.t_start + f * span;
            if (tf > times.t_start) {
                floor_times.push_back(tf);
            }
        }
    }
    size_t next_floor_time = 0;
    int floor_single_steps = 0;

    for (int i = 1; i < n_out; ++i) {
        // Loop until we've reached t_out[i] (within numerical tolerance).
        while (true) {
            // Wall-clock budget check (no-op when disabled). Placed at the
            // top of the inner sub-step loop so it fires between event
            // sub-steps as well as between output points.
            if (budget.active())
                budget.check();

            // Pick the next stop: the earliest pending apply_time strictly
            // inside (t_now, t_out[i]], else t_out[i] itself.
            double t_target = t_out[i];
            bool target_is_event = false;
            if (n_events > 0) {
                for (const auto &pe : pending_events) {
                    if (pe.apply_time > static_cast<double>(t_now) + 1e-15 &&
                        pe.apply_time < t_target) {
                        t_target = pe.apply_time;
                        target_is_event = true;
                    }
                }
            }
            // …and the next issue #177 floor-refresh time. NOT as an extra
            // output target: CVODES sizes its first step from the distance to
            // the first tout (cvHin's hub = 0.1·tdist), so an early stop
            // rewrites h0 and with it the entire step sequence — measured on the
            // corpus, an extra early target moves nearly every model, which is
            // why this ladder shipped disarmed for everything that did not
            // already need it at t=0 (and therefore for 15 of Smith's 16
            // columns, issue #183). Single-stepping instead is free by
            // construction: CV_ONE_STEP takes exactly the steps CV_NORMAL would,
            // to exactly the same tout, and simply hands each one back.
            while (next_floor_time < floor_times.size() &&
                   floor_times[next_floor_time] <= static_cast<double>(t_now) + 1e-15) {
                ++next_floor_time;
            }
            // CV_ONE_STEP does not stop short for tout — it takes the step it
            // was going to take and hands it back — so single-stepping may never
            // be allowed to cross t_target. Crossing it reorders the run against
            // anything that lives exactly there: on BIOMD0000000104, whose event
            // triggers on `time > 1`, a step that ran past t=1 found that root
            // and applied the assignment BEFORE the t=1 sample was recorded, so
            // the output carried the post-event state that `time > 1` is
            // strictly false for. It reads as a 60% state difference at an
            // identical step count — the signature of a discrete jump landing on
            // the wrong side of a sample, which no tolerance change produces.
            //
            // Against CVODES' INTERNAL time, not t_now. The two are not the same
            // number: CV_NORMAL integrates past its tout and interpolates back,
            // so after an output point t_now is the interpolated t_out[i] while
            // tn already sits beyond it. Ask with t_now and this single-steps in
            // exactly the case where CV_NORMAL would have returned t_out[i+1] by
            // interpolation without stepping at all — one extra step, on a model
            // whose floor never binds. cv_next_h is the step about to be
            // attempted and a step only ever shrinks from there (error-test
            // failures), so tn + next_h bounds where it can end. Before the first
            // step next_h is 0, which is safe for its own reason: cvHin caps h0
            // at 0.1·|tout − t0|, so the opening step cannot reach t_target.
            sunrealtype tn_now = 0.0;
            double next_h = 0.0;
            CVodeGetCurrentTime(cvode_mem, &tn_now);
            CVodeGetCurrentStep(cvode_mem, &next_h);
            const bool step_for_floor =
                next_floor_time < floor_times.size() && !target_is_event &&
                static_cast<double>(tn_now) + std::abs(next_h) < t_target - 1e-15;
            const double t_before_step = static_cast<double>(t_now);

            // ─── Stop cleanly at the next switch time (issue #48) ────────────
            // CVodeSetStopTime clamps the final step to land exactly on t*, so
            // the whole approach stays on the before-branch and the kink never
            // enters an error test. Without it, CVODES collapses h at the
            // crossing once sensitivities are active and never gets across.
            // Skipped entirely when no switch-time crossing was detected, which
            // leaves every other model's stepping bit-for-bit unchanged.
            //
            // A fixed crossing (issue #305) is stopped at through the same
            // call, for the same reason one step further back: the step that
            // spans the kink is the one that cannot be taken. It differs only
            // in what happens on arrival — there is no ∂t*/∂p to jump by, so
            // the stop is spent entirely on landing the step.
            bool stop_at_switch = false;
            bool stop_at_crossing = false;
            double t_switch = 0.0;
            double t_crossing = 0.0;
            while (next_switch < switch_list.size() &&
                   switch_list[next_switch]->t_star <= static_cast<double>(t_now) + switch_t_eps) {
                ++next_switch; // defensive: a crossing we are already past
            }
            while (next_crossing < crossing_stops.size() &&
                   crossing_stops[next_crossing] <= static_cast<double>(t_now) + switch_t_eps) {
                ++next_crossing;
            }
            // Whichever comes first. Ties cannot happen: a crossing at a #48
            // switch time was dropped from crossing_stops when it was built.
            const bool have_switch = next_switch < switch_list.size();
            const bool have_crossing = next_crossing < crossing_stops.size();
            if (have_switch && (!have_crossing || switch_list[next_switch]->t_star <=
                                                      crossing_stops[next_crossing])) {
                t_switch = switch_list[next_switch]->t_star;
                int sf = CVodeSetStopTime(cvode_mem, static_cast<sunrealtype>(t_switch));
                if (sf != CV_SUCCESS) {
                    throw std::runtime_error(
                        "CVodeSetStopTime for switch time t=" + std::to_string(t_switch) +
                        " failed: " + std::to_string(sf));
                }
                stop_at_switch = true;
            } else if (have_crossing) {
                t_crossing = crossing_stops[next_crossing];
                int sf = CVodeSetStopTime(cvode_mem, static_cast<sunrealtype>(t_crossing));
                if (sf != CV_SUCCESS) {
                    throw std::runtime_error("CVodeSetStopTime for discontinuity crossing t=" +
                                             std::to_string(t_crossing) +
                                             " failed: " + std::to_string(sf));
                }
                stop_at_crossing = true;
            } else if (!switch_list.empty() || !crossing_stops.empty()) {
                // Every crossing is behind us — clear the stop time explicitly
                // rather than trusting CVODE to have cleared it. CVODE only
                // clears tstop on a CV_TSTOP_RETURN; when a root lands on the
                // same instant it returns CV_ROOT_RETURN instead and tstop stays
                // armed at a time now behind us, which the next CVode() rejects
                // outright (CV_ILL_INPUT, "tstop is behind current t"). An SBML
                // piecewise-in-time law hits this every time: the loader
                // registers a GH #72 discontinuity root at exactly the threshold
                // this stop time targets.
                int cf = CVodeClearStopTime(cvode_mem);
                if (cf != CV_SUCCESS) {
                    throw std::runtime_error(
                        "CVodeClearStopTime after the last switch time failed: " +
                        std::to_string(cf));
                }
            }

            // …and never while the issue #48 switch stop time is armed. That
            // machinery wants an undisturbed approach to t*: CVodeSetStopTime
            // clamps the final step to land exactly on the crossing so the whole
            // approach stays on the before-branch and the kink never enters an
            // error test, and a refresh landing inside that approach moves the
            // tolerances the approach is being taken under. Measured as
            // test_switch_time_sensitivity's wide-spread cases failing outright
            // (CV_ERR_FAILURE at t=29) — the switch tests are delicate about the
            // last step before t* for exactly the reasons issue #82 documents.
            // NOT suppressed by an issue #305 crossing stop, though, and the
            // difference matters: the #48 stop is armed only as the run closes
            // on t*, while a crossing stop is armed from wherever the run is to
            // wherever the next crossing is — on Smith_BMCSystBiol2013 that is
            // the whole 240-unit horizon. Suppressing the ladder for its
            // duration turns the #177 floor off for the entire run, and the
            // #183 columns it exists for then collapse the step at t≈23 and
            // spend the wall-clock budget. There is nothing to protect anyway:
            // a crossing stop carries no jump whose accuracy depends on the
            // tolerances the approach was taken under, and CVODE honours tstop
            // in CV_ONE_STEP just as it does in CV_NORMAL, so the step still
            // lands exactly on the crossing.
            const bool one_step = step_for_floor && !stop_at_switch;
            sunrealtype t_ret;
            flag = CVode(cvode_mem, t_target, y, &t_ret, one_step ? CV_ONE_STEP : CV_NORMAL);

            // CV_TOO_MUCH_WORK is normally recoverable — max_steps is a batch
            // size per output point, not a ceiling on the run — so retry, but
            // only while the integrator is actually advancing. See
            // retry_while_advancing: a batch that moves t not at all is a
            // collapsed step size at a discontinuity, where retrying forever is
            // what made this never return (issue #54). The wall-clock budget is
            // still re-checked between batches.
            retry_while_advancing(cvode_mem, t_target, y, &t_ret, flag,
                                  "while integrating to the next output point", [&budget] {
                                      if (budget.active())
                                          budget.check();
                                  });

            if (flag < 0) {
                throw std::runtime_error(
                    "CVODE integration failed at t=" + std::to_string(t_target) +
                    " with flag=" + std::to_string(flag));
            }
            t_now = t_ret;

            // Issue #177: a single step taken only so the sensitivity error
            // floor can be re-derived from the live state. Nothing is recorded
            // here — this is not an output point — and the loop simply goes
            // round again, in CV_NORMAL once the ladder is spent.
            //
            // Only on a plain CV_SUCCESS. A root or a switch stop time can end
            // the step instead, and those returns belong to the handlers below;
            // taking this branch would skip the event fire or the issue #48
            // switch jump entirely. next_floor_time is left alone in that case,
            // so the refresh simply happens on a later step.
            //
            // The step may also overshoot t_target, since CV_ONE_STEP does not
            // stop short for it. That is why nothing is recorded from this
            // branch: the next pass sees t_now ≥ t_target, drops back to
            // CV_NORMAL, and CVODES interpolates y(t_target) out of the step
            // just taken — the same value, from the same step.
            if (one_step && flag == CV_SUCCESS) {
                // A step that advanced t not at all is the collapsed step size
                // of issue #54, and single-stepping hides it: CV_ONE_STEP never
                // returns CV_TOO_MUCH_WORK, so retry_while_advancing — the thing
                // that turns that stall into a diagnosable error naming t and h
                // — is never consulted, and the run burns its whole wall-clock
                // budget instead. Abandon the rest of the ladder the moment a
                // step buys nothing, and likewise once single-stepping has cost
                // a full max_steps batch, so the fallback is always CV_NORMAL's
                // existing handling rather than an unbounded crawl.
                if (static_cast<double>(t_ret) <= t_before_step ||
                    ++floor_single_steps >= impl_->max_steps) {
                    next_floor_time = floor_times.size();
                    continue;
                }
                bool crossed = false;
                while (next_floor_time < floor_times.size() &&
                       floor_times[next_floor_time] <= static_cast<double>(t_ret) + 1e-15) {
                    ++next_floor_time;
                    crossed = true;
                }
                sunrealtype t_tmp;
                if (crossed && CVodeGetSens(cvode_mem, &t_tmp, yS_guard.arr) == CV_SUCCESS) {
                    impl_->refresh_sens_error_floor(cvode_mem, static_cast<double>(t_ret), y,
                                                    user_data, sens, ns);
                }
                continue;
            }

            // ─── Event handling: CV_ROOT_RETURN ───────────────────────────────
            // CVODE stopped at a root (event trigger zero-crossing).
            // Identify which events fired, apply assignments, reinit integrator.
            if (flag == CV_ROOT_RETURN && n_roots > 0) {
                // root_info spans both event roots [0,n_events) and
                // discontinuity roots [n_events,n_roots). The event-firing
                // loops below only scan [0,n_events), so a discontinuity
                // crossing is never misread as an event; its sole effect is
                // the unconditional CVodeReInit at the end of this block, which
                // breaks the integration step exactly at the `time` threshold
                // so the solver cannot step over a narrow forcing pulse.
                std::vector<int> root_info(n_roots);
                CVodeGetRootInfo(cvode_mem, root_info.data());

                // Sync species concentrations for trigger/assignment evaluation
                for (int si = 0; si < ns; ++si) {
                    sp_vec_outer[si].concentration = y_data[si];
                }
                model.update_observables(y_data);
                model.evaluate_functions(static_cast<double>(t_ret));
                // GH #231: refresh the live rateOf buffer so the rising-edge
                // confirmation below — and any rateOf-bearing function it reads —
                // sees dx/dt at THIS root point, not the stale value the last
                // RHS/Newton probe happened to leave. root_fn already refreshes
                // during root-finding, so CVODE *detects* the crossing; without
                // this the confirmation re-reads a stale derivative, the rising
                // edge is missed, and the event silently never fires — erratically,
                // depending on step size (01261/01293). No-op when !uses_rateof.
                if (model.uses_rateof()) {
                    model.refresh_rateof_derivs(static_cast<double>(t_ret), y_data);
                }

                // ─── First pass: identify rising-edge events ─────────────────
                // A rising-edge fire is a root-detected event whose trigger now
                // reads true and was previously false. Trigger states are also
                // refreshed here for events that crossed without rising.
                std::vector<int> firing; // event indices that just fired
                firing.reserve(n_events);
                for (int ei = 0; ei < n_events; ++ei) {
                    if (root_info[ei] == 0)
                        continue;
                    double trigger_val = eval_ref_outer.evaluate(events_outer[ei].trigger_expr_idx);
                    bool trigger_now = (trigger_val > 0.5);
                    if (trigger_now && !trigger_was_true[ei]) {
                        firing.push_back(ei);
                    }
                    trigger_was_true[ei] = trigger_now;
                }

                // Snapshot state before firing so the chatter guard (GH #95)
                // can measure whether this batch changed it by more than the
                // integrator tolerance. chatter_y_before doubles as the
                // pre-event state x⁻ for the sensitivity jump (GH #212); capture
                // s⁻ here too, before process_firing_batch and CVodeReInit.
                std::vector<std::vector<double>> evt_s_minus;
                if (!firing.empty()) {
                    chatter_y_before.assign(y_data, y_data + ns);
                }
                // s(t_ret) must be read BEFORE the CVodeReInit below whether or
                // not anything fires: ReInit rewinds the state stepper to t_ret
                // but leaves the CVODES sensitivity history at the end of the
                // internal step the root interrupted, and nothing downstream can
                // recover it (issue #146). A root that fires an event feeds this
                // into the GH #212 jump; a root that fires nothing — a GH #72
                // discontinuity root, or an event trigger that crossed without
                // rising — still needs it, to re-seed CVODES with the
                // sensitivities that belong to t_ret.
                if (sens.n_total > 0) {
                    evt_s_minus =
                        impl_->capture_event_sens(cvode_mem, ns, static_cast<double>(t_ret), sens);
                }

                bool any_event_fired = process_firing_batch(static_cast<double>(t_ret), firing);

                // ─── Chatter guard: detect Zeno re-firing (GH #95) ───────────
                // A fire that BOTH advances simulated time negligibly since this
                // event's previous fire AND changes the state by less than the
                // tolerance is re-firing on floating-point noise, not tracking
                // real dynamics — what a "keep X >= 0" clamp does once X has
                // decayed far below atol. After CHATTER_LIMIT such fires in a
                // row the event is marked dormant; root_fn then drops its root
                // so CVODE steps over the noise floor (as RoadRunner does by
                // keeping X cleanly positive). The dual criterion leaves genuine
                // recurring events — which move the state and fire with real
                // time gaps — untouched.
                if (any_event_fired && !firing.empty()) {
                    bool subtol = true;
                    for (int si = 0; si < ns; ++si) {
                        // "Below the tolerance" is a per-species statement, so
                        // it reads the per-species atol when there is one
                        // (issue #196); with none, atol_v is empty and this is
                        // the scalar test it always was.
                        const double atol_si =
                            atol_v.empty() ? atol : atol_v[static_cast<size_t>(si)];
                        if (std::fabs(y_data[si] - chatter_y_before[si]) >
                            atol_si + rtol * std::fabs(y_data[si])) {
                            subtol = false;
                            break;
                        }
                    }
                    const double t_fire = static_cast<double>(t_ret);
                    const double horizon = t_out.back() - t_out.front();
                    const double time_eps =
                        1e-6 * std::max(std::fabs(t_fire), horizon > 0.0 ? horizon : 1.0);
                    for (int ei : firing) {
                        const double gap = t_fire - event_last_fire_time[ei];
                        event_last_fire_time[ei] = t_fire;
                        if (subtol && gap <= time_eps) {
                            if (++event_chatter_count[ei] >= CHATTER_LIMIT) {
                                event_dormant[ei] = 1;
                            }
                        } else {
                            event_chatter_count[ei] = 0;
                        }
                    }
                }

                // Refresh observables and functions after
                // event assignments so that:
                //   1. Promoted-param species → observable → rate law chain sees
                //      the updated values
                //   2. Subsequent events in the same root batch see updated state
                //   3. The integrator restarts with consistent evaluator state
                if (any_event_fired) {
                    model.update_observables(y_data);
                    model.evaluate_functions(static_cast<double>(t_ret));
                    // Refresh dx/dt at the post-fire state so the cascade re-check
                    // below sees a fresh rateOf buffer (GH #231). No-op otherwise.
                    if (model.uses_rateof()) {
                        model.refresh_rateof_derivs(static_cast<double>(t_ret), y_data);
                    }
                }

                // Re-check every trigger against the POST-fire state and schedule
                // any delayed event an assignment just pushed false→true ("events
                // triggering events", GH #233) — the root finder can't see a
                // discrete jump. This also advances trigger_was_true, so an event
                // whose own assignment FALSIFIES its trigger (trigger ``S1<0.1``,
                // assignment ``S1:=1``) re-arms for the next rising edge, while
                // one whose assignment RE-SATISFIES the trigger queues its next
                // (delayed) fire. The cascade only queues, so the unconditional
                // CVodeReInit below (for the root batch's own assignments) is the
                // one that matters; sensitivities are upstream-guarded off for
                // the self-triggering event models this path serves, so the GH
                // #212 jump (keyed on the root batch `firing`) is left untouched.
                cascade_triggered_events(static_cast<double>(t_ret));

                // Reinitialize CVODE with modified state vector
                int reinit_flag = impl_->reinit_cvode(cvode_mem, t_ret, y);
                if (reinit_flag != CV_SUCCESS) {
                    throw std::runtime_error("CVodeReInit after event failed: " +
                                             std::to_string(reinit_flag));
                }
                // Jump dx/dp across the event and re-seed CVODES sensitivity
                // vectors (GH #212). chatter_y_before holds this batch's
                // pre-fire state x⁻ (snapshotted above whenever firing is
                // non-empty, which any_event_fired implies). No-op unless
                // sensitivities are active.
                //
                // A state-dependent rate-law switch crossed here too, if any of
                // its residual roots is in this batch (issue #150).
                std::vector<int> switched;
                for (int j = 0; j < n_state_switch; ++j) {
                    if (root_info[n_events + n_disc + j] != 0) {
                        switched.push_back(j);
                    }
                }
                if (any_event_fired) {
                    if (!switched.empty() && !evt_s_minus.empty()) {
                        throw std::runtime_error(
                            "Forward sensitivity: a state-dependent rate-law switch (residual '" +
                            state_switches[switched.front()]->residual_source +
                            "') crosses at exactly the instant t=" + std::to_string(t_ret) +
                            " an event fires. The event jump differentiates at the pre-event "
                            "state and the switch jump differentiates the branch of f at the "
                            "same instant, so composing them is ambiguous and bngsim refuses "
                            "rather than pick an order (issue #150). Separate the two times, or "
                            "drop sensitivities for this run.");
                    }
                    impl_->apply_event_sensitivity_jump(opts, cvode_mem, ns,
                                                        static_cast<double>(t_ret), firing,
                                                        chatter_y_before, evt_s_minus, sens);
                } else if (!evt_s_minus.empty()) {
                    // No event fired. The state is continuous across this root
                    // either way, but a state-switch crossing makes dx/dθ
                    // discontinuous there by the saltation term, so add it to
                    // the captured s⁻ in place before re-seeding (issue #150).
                    //
                    // Several residuals rooting at ONE instant go in together
                    // rather than one at a time (issue #153): on the corpus they
                    // are always one crossing surface the text dedup could not
                    // merge — `ds1` against `3·ds1 − 12·s1²·ds1` — which wants a
                    // single jump off a single probe, and telling the two cases
                    // apart is a run-time measurement at the crossing, not a
                    // property of the text.
                    if (!switched.empty()) {
                        std::vector<const NetworkModel::StateSwitch *> batch;
                        batch.reserve(switched.size());
                        for (int j : switched) {
                            batch.push_back(state_switches[j]);
                        }
                        impl_->apply_state_switch_sensitivity_jump(
                            cvode_mem, y, ns, static_cast<double>(t_ret), batch, evt_s_minus, sens);
                    }
                    // The CVodeReInit above rewound the state stepper to t_ret
                    // while leaving CVODES' sensitivity history at the end of
                    // the interrupted step, so resuming would restart dx/dθ from
                    // s(t_n) against x(t_ret). Re-seed with s(t_ret), captured
                    // before the ReInit (issue #146). Without this every GH #72
                    // discontinuity root injects an s'·(t_n − t_ret) step into
                    // every column: on AMICI's `events` fixture, whose two
                    // assignment-free triggers are roots and nothing else, that
                    // is a 5% error against the model's own finite difference.
                    impl_->resume_sens_after_reinit(cvode_mem, ns, evt_s_minus, sens);
                }
                // Inner-while-loop will continue to t_out[i] (or to the next
                // pending apply_time if one is closer) on its next pass.
            }

            // ─── Switch-time crossing: jump dx/dp (issue #48) ────────────────
            // We stopped ON the crossing (CV_TSTOP_RETURN, or CV_ROOT_RETURN if
            // an event root happened to land on the same instant — keyed on the
            // reached time rather than the flag so either is handled). Apply
            // every crossing scheduled at this instant; each is an independent
            // additive jump on the same unchanged state, so coincident switches
            // simply sum.
            if (stop_at_switch && static_cast<double>(t_ret) >= t_switch - switch_t_eps) {
                while (next_switch < switch_list.size() &&
                       switch_list[next_switch]->t_star <=
                           static_cast<double>(t_ret) + switch_t_eps) {
                    impl_->apply_switch_sensitivity_jump(
                        cvode_mem, y, ns, static_cast<double>(t_ret), *switch_list[next_switch],
                        sw_scratch, sens);
                    ++next_switch;
                }
            }

            // ─── Fixed crossing reached: restart on the after-branch (#305) ──
            // The stop landed the step exactly on t*, so the whole approach was
            // taken on the before-branch and the kink never entered an error
            // test. What is left is to drop the before-branch BDF history, or
            // the first step past the crossing is predicted from a polynomial
            // fitted entirely to the branch that just ended — the same reason
            // the GH #72 root reinits, and the reason issue #48's jump ends in
            // one too.
            //
            // A root landing on the same instant has already done exactly that
            // (and, with sensitivities active, re-seeded s(t*) across it for
            // issue #146), so this runs only when none fired: an unconditional
            // second reinit would be harmless but wasteful, while skipping the
            // s⁻ capture/resume on the path where no root fires would not be.
            if (stop_at_crossing && static_cast<double>(t_ret) >= t_crossing - switch_t_eps) {
                while (next_crossing < crossing_stops.size() &&
                       crossing_stops[next_crossing] <= static_cast<double>(t_ret) + switch_t_eps) {
                    ++next_crossing;
                }
                if (flag != CV_ROOT_RETURN) {
                    std::vector<std::vector<double>> cross_s_minus;
                    if (sens.n_total > 0) {
                        cross_s_minus = impl_->capture_event_sens(cvode_mem, ns,
                                                                  static_cast<double>(t_ret), sens);
                    }
                    int rf = impl_->reinit_cvode(cvode_mem, t_ret, y);
                    if (rf != CV_SUCCESS) {
                        throw std::runtime_error("CVodeReInit at the discontinuity crossing t=" +
                                                 std::to_string(t_crossing) +
                                                 " failed: " + std::to_string(rf));
                    }
                    if (!cross_s_minus.empty()) {
                        impl_->resume_sens_after_reinit(cvode_mem, ns, cross_s_minus, sens);
                    }
                }
            }

            // ─── Pending delayed events ──────────────────────────────────────
            // Check if any pending delayed events should fire at current time.
            // Also cancel non-persistent events whose trigger reverted to false.
            if (!pending_events.empty()) {
                const auto &events = model.events();
                auto &eval_ref = model.evaluator();
                auto &sp_vec = const_cast<std::vector<Species> &>(model.species());

                // Cancel non-persistent events whose trigger is no longer true
                for (int si = 0; si < ns; ++si) {
                    sp_vec[si].concentration = y_data[si];
                }
                model.update_observables(y_data);
                model.evaluate_functions(static_cast<double>(t_ret));
                // Fresh dx/dt for the non-persistent-cancel trigger check below
                // (a rateOf-bearing delayed-event trigger; GH #231). No-op otherwise.
                if (model.uses_rateof()) {
                    model.refresh_rateof_derivs(static_cast<double>(t_ret), y_data);
                }

                // Cancel non-persistent pending events whose trigger has already
                // lapsed. Factored out so it can also run BETWEEN individual
                // delayed applies below (SBML §4.11.3): a competing delayed event
                // that another just disabled must not still fire.
                auto cancel_lapsed_nonpersistent = [&]() {
                    pending_events.erase(
                        std::remove_if(pending_events.begin(), pending_events.end(),
                                       [&](const PendingEvent &pe) {
                                           const auto &ev = events[pe.event_idx];
                                           if (!ev.persistent) {
                                               double tv = eval_ref.evaluate(ev.trigger_expr_idx);
                                               return tv <= 0.5; // trigger reverted: cancel
                                           }
                                           return false;
                                       }),
                        pending_events.end());
                };
                cancel_lapsed_nonpersistent();

                // Apply events whose delay has expired, ONE AT A TIME in queue
                // order (which preserves the §4.11.6 random pick made at trigger
                // time). Between applies we (a) re-cancel any non-persistent
                // pending event this assignment just disabled — so competing
                // delayed events like 01590's Qinc/Rinc mutually exclude, exactly
                // one incrementing per round instead of both — and (b) run the
                // cascade so a same-instant delay-0 event the assignment triggers
                // (01590's maxcheck; a persistent chain link 01754/58/59) fires.
                // When the event was queued under useValuesFromTriggerTime=true,
                // pe.frozen_values holds the trigger-time RHS values applied
                // verbatim; otherwise the RHS is evaluated against current state.
                bool delayed_applied = false;
                while (true) {
                    int due = -1;
                    for (size_t j = 0; j < pending_events.size(); ++j) {
                        if (static_cast<double>(t_ret) >= pending_events[j].apply_time) {
                            due = static_cast<int>(j);
                            break;
                        }
                    }
                    if (due < 0) {
                        break;
                    }
                    PendingEvent pe = std::move(pending_events[static_cast<size_t>(due)]);
                    pending_events.erase(pending_events.begin() + due);

                    const auto &ev_pe = events[pe.event_idx];
                    const auto &assigns = ev_pe.assignments;
                    const bool use_frozen = !pe.frozen_values.empty();
                    for (size_t a = 0; a < assigns.size(); ++a) {
                        int sp_idx0 = assigns[a].first;
                        // The injected compartment-resize concentration rescale
                        // (ode_only, GH #74) conserves each contained species'
                        // *amount* across the resize, which physically happens at
                        // the event's execution (apply) time — so it must read
                        // pre-fire state HERE, never the trigger-time snapshot a
                        // useValuesFromTriggerTime=true event freezes for its user
                        // assignments. GH #248: for such a delayed UVFTT event a
                        // species produced by a reaction between trigger and apply
                        // time (case 01000's S2) was otherwise rescaled from its
                        // stale trigger-time amount, corrupting it by V_old/V_new
                        // at the wrong volume. Evaluating the rescale fresh here
                        // reproduces the (correct) UVFTT=false apply path exactly.
                        const bool ode_only =
                            a < ev_pe.assignment_ode_only.size() && ev_pe.assignment_ode_only[a];
                        double nv = (use_frozen && !ode_only)
                                        ? pe.frozen_values[a]
                                        : eval_ref.evaluate(assigns[a].second);
                        y_data[sp_idx0] = nv;
                        sp_vec[sp_idx0].concentration = nv;
                    }
                    delayed_applied = true;

                    model.update_observables(y_data);
                    model.evaluate_functions(static_cast<double>(t_ret));
                    if (model.uses_rateof()) {
                        model.refresh_rateof_derivs(static_cast<double>(t_ret), y_data);
                    }
                    cancel_lapsed_nonpersistent();
                    cascade_triggered_events(static_cast<double>(t_ret));
                }

                if (delayed_applied) {
                    // Reinit CVODE after delayed event(s) modified state.
                    // The only re-init on this path with no sensitivity
                    // counterpart, and deliberately so (issue #146): a model
                    // with an effective execution delay is refused upstream by
                    // NetworkModel::event_sensitivity_unsupported_reason, so
                    // sensitivities are never live here. Whoever lifts that
                    // refusal owns the jump AND the CVodeSensReInit — the state
                    // has jumped by then, so re-seeding s unchanged (what
                    // resume_sens_after_reinit does) would be wrong here.
                    int reinit_flag = impl_->reinit_cvode(cvode_mem, t_ret, y);
                    if (reinit_flag != CV_SUCCESS) {
                        throw std::runtime_error("CVodeReInit after delayed event failed: " +
                                                 std::to_string(reinit_flag));
                    }
                }
            }

            // Inner-while exit: reached t_out[i] within numerical tolerance.
            if (t_now >= static_cast<sunrealtype>(t_out[i]) - static_cast<sunrealtype>(1e-12)) {
                break;
            }
        } // end while (true) — inner integration loop

        sunrealtype t_ret = t_now; // for downstream code (recording, sensitivities)

        // ─── Chatter guard: re-arm recovered events (GH #95) ─────────────────
        // A dormant event whose assigned species have climbed back above the
        // atol noise floor is no longer pinned at the numerical fixed point
        // where chatter lives, so re-enable its trigger root. Decay-to-zero
        // never re-arms (the variable stays near zero), so suppression is
        // effectively permanent for the clamp pathology, while a genuine
        // recovery re-arms within one output interval (the model's own reporting
        // resolution). The CVodeReInit re-baselines the root function so the
        // re-enabled trigger's current sign becomes CVODE's edge-detection
        // reference.
        if (n_events > 0) {
            bool rearmed = false;
            const double rearm_floor = REARM_TOL_FACTOR * atol;
            for (int ei = 0; ei < n_events; ++ei) {
                if (event_dormant[ei] == 0) {
                    continue;
                }
                double mag = 0.0;
                for (const auto &asg : events_outer[ei].assignments) {
                    mag = std::max(mag, std::fabs(y_data[asg.first]));
                }
                if (mag > rearm_floor) {
                    event_dormant[ei] = 0;
                    event_chatter_count[ei] = 0;
                    rearmed = true;
                }
            }
            if (rearmed) {
                // Same ordering as every other re-init on this path (issue
                // #146): read s(t_now) first, because CVodeReInit rewinds the
                // state stepper and leaves the sensitivity history where the
                // last internal step left it. Nothing jumps here — re-arming an
                // event changes no state — so the columns go back unchanged.
                std::vector<std::vector<double>> rearm_s =
                    impl_->capture_event_sens(cvode_mem, ns, static_cast<double>(t_now), sens);
                int rf = impl_->reinit_cvode(cvode_mem, t_now, y);
                if (rf != CV_SUCCESS) {
                    throw std::runtime_error("CVodeReInit after chatter re-arm failed: " +
                                             std::to_string(rf));
                }
                impl_->resume_sens_after_reinit(cvode_mem, ns, rearm_s, sens);
            }
        }

        // Update observables from current state. Refresh the rateOf buffer at
        // this exact (t_ret, y_data) first so a rate_of__<species> accessor in a
        // recorded assignment-rule function reads dx/dt here, not the last
        // internal integration step's value (GH #106/#231). No-op otherwise.
        if (model.uses_rateof()) {
            model.refresh_rateof_derivs(t_ret, y_data);
        }
        model.update_observables(y_data);
        model.evaluate_functions(t_ret);

        // Copy function values back to species array for assignment-rule
        // species. When an SBML assignment rule targets a species, the
        // loader creates a function with the species name. After
        // evaluate_functions(), the function holds the correct value,
        // but the species slot in y_data still has the ODE-integrated
        // (stale) value. Copy function → y_data so the species output
        // reflects the assignment rule's computed value. The (func, species)
        // pairs were resolved once into ar_copyback (GH #136); the cached
        // function values come from the evaluate_functions() call just above.
        if (!ar_copyback.empty()) {
            const auto &fvals = model.function_value_cache();
            for (const auto &[fi, si] : ar_copyback) {
                y_data[si] = fvals[fi];
            }
        }

        for (int j = 0; j < n_obs; ++j) {
            obs_buf[j] = model.observables()[j].total;
        }

        result.record(i, static_cast<double>(t_ret), y_data, obs_buf.data());

        // Record function values
        if (n_func > 0) {
            result.record_expressions(i, model.function_value_cache().data());
        }

        // Extract sensitivities at this time point.
        if (n_sens > 0) {
            flag = CVodeGetSens(cvode_mem, &t_ret, yS_guard.arr);
            if (flag == CV_SUCCESS) {
                std::vector<const double *> sens_ptrs(n_sens);
                for (int s = 0; s < n_sens; ++s) {
                    sens_ptrs[s] = N_VGetArrayPointer(yS_guard[s]);
                }
                if (n_sens_p > 0) {
                    result.record_sensitivities(i, sens_ptrs.data(), ns, n_sens_p);
                }
                if (n_sens_ic > 0) {
                    result.record_sensitivities_ic(i, sens_ptrs.data() + n_sens_p, ns, n_sens_ic);
                }
                record_observable_output_sensitivities(i, sens_ptrs.data());
                record_expression_output_sensitivities(i, static_cast<double>(t_ret),
                                                       sens_ptrs.data());
                // Issue #177: the magnitudes being summed move with the state,
                // so re-derive the roundoff floor here — where the run holds
                // control and yS_guard already carries s(t_ret) from the
                // CVodeGetSens above. The floor set at t=0 covers the models
                // whose ∂f/∂p is large from the start (it depends on y, not on
                // s, so unlike the J·s half it is not zero there); this covers a
                // model that starts small and grows into the same problem.
                impl_->refresh_sens_error_floor(cvode_mem, static_cast<double>(t_ret), y, user_data,
                                                sens, ns);
            }
        }

        last_recorded_index = i;

        // ─── Steady-state early-stop check ───────────────────────────────────
        // Mirrors BNG2.pl ``run_network -c`` (Network3 network.cpp): after
        // every output point we compute ``||f(t,y)||_2 / n_species`` and stop
        // integrating once it falls below ``ss_tol``. The current row is
        // already recorded above so it stays in the truncated Result.
        if (check_ss) {
            model.compute_derivs(static_cast<double>(t_ret), y_data, ss_derivs.data());
            double sumsq = 0.0;
            for (int k = 0; k < ns; ++k) {
                sumsq += ss_derivs[k] * ss_derivs[k];
            }
            const double dx = std::sqrt(sumsq) / static_cast<double>(ns);
            ss_residual_last = dx;
            if (dx < ss_tol) {
                ss_reached = true;
                break;
            }
        }
    }

    // ─── Solver statistics ───────────────────────────────────────────────────

    impl_->record_solver_stats(cvode_mem, LS_guard, result);

    if (check_ss) {
        result.solver_stats().steady_state_reached = ss_reached;
        result.solver_stats().steady_state_residual = ss_residual_last;
    }

    // ─── Truncate Result on steady-state early-stop ──────────────────────────
    // When the steady-state check broke us out of the output loop, the
    // pre-allocated rows past ``last_recorded_index`` were never integrated
    // and still hold the zero-initialized values from ``allocate()``. Drop
    // them so the caller sees only the rows we actually integrated.
    if (check_ss && ss_reached && last_recorded_index + 1 < n_out) {
        result.truncate(last_recorded_index + 1);
    }

    // ─── Write final state back to model ─────────────────────────────────────
    // Publishes the final concentrations, the final time and the GH #210
    // forward-sensitivity carry-over seed onto the model — see
    // Impl::write_final_state_back. When the steady-state early-stop fired, the
    // "final time" is the last sample we actually integrated to, not the
    // originally requested ``t_end``.
    {
        const double final_t = (check_ss && ss_reached) ? t_out[last_recorded_index] : times.t_end;
        impl_->write_final_state_back(opts, ns, y_data, final_t, sens);
    }

    // ─── Cleanup ─────────────────────────────────────────────────────────────

    // All SUNDIALS resources are freed automatically by the RAII guards:
    //   sens.yS (== yS_guard), LS_guard, A_guard, cvode_mem, y, ctx,
    //   codegen_param_buf
    // The cached codegen library (impl_->codegen_lib) intentionally stays open
    // for the simulator's lifetime and is unloaded by ~Impl (GH #77).

    return result;
}

} // namespace bngsim
