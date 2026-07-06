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

#include "bngsim/functional_jac_scatter.hpp"
#include "bngsim/lapack_dense_linsol.hpp"
#include "bngsim/mm_jacobian.hpp"
#include "bngsim/model.hpp"
#include "bngsim/platform_compat.hpp" // POSIX ssize_t shim for Windows (GH #150)
#include "bngsim/result.hpp"
#include "bngsim/simulator.hpp"
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
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bngsim {

// ─── Constants ───────────────────────────────────────────────────────────────

// Species count threshold for auto-selecting sparse solver.
// Uses KLU when N >= SPARSE_THRESHOLD AND density < SPARSE_DENSITY_MAX.
// Many rule-based models have relatively dense Jacobians
// (chromatic number ≈ N, density often 10-30%), making KLU's sparse
// factorization slower than dense LAPACK LU. Only models with truly sparse
// Jacobians (metapopulation, compartmental transport) tend to benefit.
// The density cutoff reflects internal benchmarking.
static constexpr int SPARSE_THRESHOLD = 50;
static constexpr double SPARSE_DENSITY_MAX = 0.10; // 10%

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

// Tfun callback type: invoked by codegen .so to evaluate a table function at
// the given index value. ctx is opaque on the .so side; we set it to the
// owning NetworkModel pointer.
using CodegenTfunEvalFn = double (*)(int tf_id, double x, void *ctx);

// Lightweight struct matching the CodegenUserData layout expected by .so.
// MUST mirror the typedef emitted by bngsim/python/bngsim/_codegen.py
// (generate_rhs_c). Field order is part of the ABI contract between
// the codegen .so and the simulator.
struct CodegenUserDataForSO {
    double *param_values;
    void *tfun_ctx;
    CodegenTfunEvalFn tfun_eval;
};

struct CvodeUserData {
    NetworkModel *model;
    // Code-generated RHS function pointer.
    // If non-null, used instead of model->compute_derivs().
    // The codegen function reads parameters from param_values.
    using CodegenRhsFn = int (*)(double t, double *y, double *ydot, void *user_data);
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
    using CodegenSensRhsFn = int (*)(int Ns, double t, double *y, double *ydot, int iS, double *yS,
                                     double *ySdot, void *user_data, double *tmp1, double *tmp2);
    CodegenSensRhsFn codegen_sens_fn = nullptr;
    // plist for sensitivity codegen (maps iS → param index)
    int *codegen_plist = nullptr;
    int codegen_n_sens = 0;

    // Code-generated dense analytical Jacobian function pointer (GH #76 Task 4).
    // If non-null AND the model's analytical Jacobian is complete, the dense
    // Jacobian dispatch prefers this compiled mirror of fill_dense_analytical_
    // jacobian over the interpreted cvode_analytical_dense_jac. jac is the n×n
    // column-major SUNDenseMatrix data; the emitted C memsets it itself. Reads
    // params from the same CodegenUserDataForSO the RHS uses.
    using CodegenJacFn = int (*)(double t, double *y, double *jac_colmajor, void *user_data);
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
    using CodegenJacSparseFn = int (*)(double t, double *y, double *jac_data, void *user_data);
    CodegenJacSparseFn codegen_jac_sparse_fn = nullptr;

    // Code-generated observable/expression output evaluator (GH #136). When
    // non-null, the warm recording loop calls this compiled function once per
    // output row to fill obs_out[N_OBS] and func_out[N_FUNC] — replacing the
    // interpreted update_observables() + evaluate_functions() pass that
    // dominated wall time on large models. Reads params from the same
    // CodegenUserDataForSO the RHS uses (so on the warm path, where params are
    // constant, the buffer is always current). Null ⇒ interpreted recording.
    using CodegenOutputsFn = int (*)(double t, double *y, double *obs_out, double *func_out,
                                     void *user_data);
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
    using CodegenOutputSensFn = int (*)(double t, const double *y, const double *p,
                                        const double *const *state_sens, const int *plist,
                                        int n_sens, double *obs_sens_out, double *func_sens_out,
                                        void *user_data);
    CodegenOutputSensFn codegen_output_sens_fn = nullptr;

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
};

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
            params[i].value = data->sens_p[i];
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
        data->model->compute_derivs(static_cast<double>(t),
                                    clamp_state_nonneg(data, y_ptr), ydot_ptr);
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
            params[i].value = data->sens_p[i];
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

struct CodegenSensUserDataForSO {
    double *param_values;
    int *plist;
    int n_sens;
};

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
    sunindextype *jac_col_ptrs = SUNSparseMatrix_IndexPointers(J);
    sunindextype *jac_row_indices = SUNSparseMatrix_IndexValues(J);

    // CVODE may call SUNMatZero() before this callback, and SUNMatZero_Sparse
    // clears BOTH values and structural indices. Reinstall CSC structure first.
    for (int j = 0; j <= sp.n; ++j) {
        jac_col_ptrs[j] = static_cast<sunindextype>(sp.col_ptrs[j]);
    }
    for (int k = 0; k < sp.nnz; ++k) {
        jac_row_indices[k] = static_cast<sunindextype>(sp.row_indices[k]);
    }

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
    sunindextype *jac_col_ptrs = SUNSparseMatrix_IndexPointers(J);
    sunindextype *jac_row_indices = SUNSparseMatrix_IndexValues(J);

    // Reinstall the CSC structure (see cvode_analytical_jac): SUNMatZero_Sparse
    // may have cleared both values and structural indices before this callback.
    for (int j = 0; j <= sp.n; ++j) {
        jac_col_ptrs[j] = static_cast<sunindextype>(sp.col_ptrs[j]);
    }
    for (int k = 0; k < sp.nnz; ++k) {
        jac_row_indices[k] = static_cast<sunindextype>(sp.row_indices[k]);
    }

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

static int cvode_colored_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
                             N_Vector /*tmp1*/, N_Vector /*tmp2*/, N_Vector /*tmp3*/) {

    auto *data = static_cast<CvodeUserData *>(user_data);
    NetworkModel *model = data->model;
    const int ns = model->n_species();
    const auto &sp = model->jacobian_sparsity();

    double *y_data = N_VGetArrayPointer(y);
    double *fy_data = N_VGetArrayPointer(fy);

    // Access the sparse matrix CSC arrays
    sunrealtype *jac_data = SUNSparseMatrix_Data(J);
    sunindextype *jac_col_ptrs = SUNSparseMatrix_IndexPointers(J);
    sunindextype *jac_row_indices = SUNSparseMatrix_IndexValues(J);

    // CVODE may call SUNMatZero() before this callback, and SUNMatZero_Sparse
    // clears BOTH values and structural indices. Reinstall CSC structure first.
    for (int j = 0; j <= sp.n; ++j) {
        jac_col_ptrs[j] = static_cast<sunindextype>(sp.col_ptrs[j]);
    }
    for (int k = 0; k < sp.nnz; ++k) {
        jac_row_indices[k] = static_cast<sunindextype>(sp.row_indices[k]);
    }

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

    // Finite difference perturbation scale
    const double sqrt_uround = 1.4901161193847656e-8; // sqrt(machine epsilon)

    // Iterate over colors (one RHS eval per color)
    for (int c = 0; c < sp.n_colors; ++c) {
        const auto &group = sp.color_groups[c];

        // 1. Build perturbed state: y_pert = y + Σ_{j in group} h_j * e_j
        std::memcpy(y_pert, y_data, ns * sizeof(double));

        for (int j : group) {
            double h = sqrt_uround * std::max(std::abs(y_data[j]), 1.0);
            h_vals[j] = h;
            y_pert[j] += h;
        }

        // 2. Single RHS evaluation for all columns in this color group
        model->compute_derivs(static_cast<double>(t), y_pert, fy_pert);

        // 3. Extract Jacobian entries for each column in the group
        for (int j : group) {
            double inv_h = 1.0 / h_vals[j];
            int64_t col_start = sp.col_ptrs[j];
            int64_t col_end = sp.col_ptrs[j + 1];

            for (int64_t k = col_start; k < col_end; ++k) {
                int i = static_cast<int>(sp.row_indices[k]);
                jac_data[k] = (fy_pert[i] - fy_data[i]) * inv_h;
            }
        }
    }

    return 0; // success
}
#endif // BNGSIM_HAS_KLU

// ─── CvodeSimulator::Impl ───────────────────────────────────────────────────

struct CvodeSimulator::Impl {
    NetworkModel &model;
    double rtol = 1e-8;
    double atol = 1e-8;
    int max_steps = 20000; // Matches SolverOptions::max_steps default

    // Direct linear solver chosen by the most recent setup_linsol_and_jac()
    // (a LinearSolverKind). Both run() and run_warm() call setup just before
    // integrating, so this is fresh when the solver stats are recorded.
    int linear_solver_used = LINEAR_SOLVER_DENSE;

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
        double max_step_size = -1.0;
        int max_steps = 0;
        std::string jacobian;
        std::string codegen_so_path;
        std::string codegen_c_source; // MIR-JIT source fingerprint (GH #78)
        bool force_dense = false;
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
        const auto &sp = model.jacobian_sparsity();
        const bool analytical_ready = model.analytical_jacobian_complete();
        CVLsJacFn jac_fn = nullptr;

        if (jac_strategy != "fd" && analytical_ready &&
            (jac_strategy == "analytical" || jac_strategy == "auto")) {
            // Prefer the compiled CSC Jacobian (GH #162) when codegen resolved it;
            // it mirrors fill_sparse_analytical_jacobian without the per-step
            // ExprTk eval. Falls back to the interpreted sparse callback otherwise.
            jac_fn = user_data.codegen_jac_sparse_fn ? cvode_codegen_sparse_jac
                                                     : cvode_analytical_jac;
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

        if (jac_fn) {
            flag = CVodeSetJacFn(cvode_mem, jac_fn);
            if (flag != CV_SUCCESS) {
                throw std::runtime_error("CVodeSetJacFn failed");
            }
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
        w.valid && w.ns == ns && w.rtol == rtol && w.atol == atol && w.max_steps == max_steps &&
        w.max_step_size == opts.max_step_size && w.jacobian == jac_strategy &&
        w.force_dense == opts.force_dense_linear_solver && w.use_sparse == use_sparse &&
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
        flag = CVodeSStolerances(w.cvode_mem, rtol, atol);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeSStolerances failed");
        }
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
        w.max_steps = max_steps;
        w.max_step_size = opts.max_step_size;
        w.jacobian = jac_strategy;
        w.force_dense = opts.force_dense_linear_solver;
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
        while (flag == CV_TOO_MUCH_WORK) {
            if (budget.active())
                budget.check();
            flag = CVode(cvode_mem, t_out[i], w.y, &t_ret, CV_NORMAL);
        }
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
    long int nst, nfe, nsetups, nni, ncfn, netf;
    CVodeGetNumSteps(cvode_mem, &nst);
    CVodeGetNumRhsEvals(cvode_mem, &nfe);
    CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    CVodeGetNumErrTestFails(cvode_mem, &netf);
    result.solver_stats().n_steps = static_cast<int>(nst);
    result.solver_stats().n_rhs_evals = static_cast<int>(nfe);
    result.solver_stats().n_jac_evals = static_cast<int>(nsetups);
    result.solver_stats().n_nonlin_iters = static_cast<int>(nni);
    result.solver_stats().n_nonlin_conv_fails = static_cast<int>(ncfn);
    result.solver_stats().n_err_test_fails = static_cast<int>(netf);
    result.solver_stats().linear_solver = linear_solver_used;
    // GH #132: how many factorizations took the BLAS dgetrf path (0 unless this
    // is the LAPACK-dense solver and the run crossed the adaptive K gate).
    {
        const long bc = lapack_dense_blas_factor_count(w.LS);
        result.solver_stats().n_dense_blas_factorizations = bc > 0 ? static_cast<int>(bc) : 0;
    }
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

// ─── Public interface ────────────────────────────────────────────────────────

CvodeSimulator::CvodeSimulator(NetworkModel &model) : impl_(std::make_unique<Impl>(model)) {}

CvodeSimulator::~CvodeSimulator() = default;

void CvodeSimulator::set_tolerances(double rtol, double atol) {
    impl_->rtol = rtol;
    impl_->atol = atol;
}

void CvodeSimulator::set_max_steps(int max_steps) { impl_->max_steps = max_steps; }

Result CvodeSimulator::run(const TimeSpec &times, const SolverOptions &opts) {
    auto &model = impl_->model;
    const int ns = model.n_species();
    const int n_obs = model.n_observables();

    if (ns == 0) {
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
        if (model.n_events() > 0) {
            throw std::runtime_error(
                "Cannot simulate: model has no species but defines events "
                "(no integrator state to anchor the trigger crossing).");
        }
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

    // Wall-clock budget. Checked at each outer integration step and at each
    // pending-event sub-step (see the inner while loop below). 0 disables.
    WallClockBudget budget(opts.timeout_seconds);

    double rtol = (opts.rtol > 0) ? opts.rtol : impl_->rtol;
    double atol = (opts.atol > 0) ? opts.atol : impl_->atol;
    int max_steps = (opts.max_steps > 0) ? opts.max_steps : impl_->max_steps;

    // Decide dense vs sparse.
    // Use sparse KLU when: (1) KLU is available, (2) model is large enough,
    // (3) sparsity pattern exists, and (4) density is low enough to benefit.
    // If density > 50%, the Jacobian is effectively dense and KLU's overhead
    // makes it slower than optimized LAPACK dense LU.
    // Also guards against models where Functional rate laws make the sparsity
    // pattern nearly dense.
    // JAX Jacobians always produce dense matrices, so force dense mode there.
#ifdef BNGSIM_HAS_KLU
    const auto &sp = model.jacobian_sparsity();
    const bool use_sparse = !opts.force_dense_linear_solver && (opts.jacobian != "jax") &&
                            (ns >= SPARSE_THRESHOLD) && !sp.empty() &&
                            (sp.density < SPARSE_DENSITY_MAX);
#else
    const bool use_sparse = false;
#endif

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

    // RAII guards handle all SUNDIALS cleanup automatically.
    SunContextGuard ctx;
    if (!ctx) {
        throw std::runtime_error("SUNContext_Create failed");
    }

    NVectorGuard y(N_VNew_Serial(ns, ctx));
    if (!y) {
        throw std::runtime_error("N_VNew_Serial failed");
    }

    double *y_data = y.data();
    for (int i = 0; i < ns; ++i) {
        y_data[i] = model.species()[i].concentration;
    }

    CvodeMemGuard cvode_mem(CVodeCreate(CV_BDF, ctx));
    if (!cvode_mem) {
        throw std::runtime_error("CVodeCreate failed");
    }

    // User data for RHS callback
    CvodeUserData user_data{&model};

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
    // rhs). codegen_param_buf must outlive the integration loop below — the
    // user_data points into it.
    std::vector<double> codegen_param_buf; // RAII: replaces new[]/delete[]
    CVRhsFn rhs_fn = impl_->setup_codegen_rhs(opts, user_data, codegen_param_buf);

    int flag = CVodeInit(cvode_mem, rhs_fn, times.t_start, y);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeInit failed: " + std::to_string(flag));
    }

    flag = CVodeSStolerances(cvode_mem, rtol, atol);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSStolerances failed");
    }

    flag = CVodeSetUserData(cvode_mem, &user_data);
    flag = CVodeSetMaxNumSteps(cvode_mem, max_steps);

    if (opts.max_step_size > 0) {
        CVodeSetMaxStep(cvode_mem, opts.max_step_size);
    }

    // ─── Validate Jacobian strategy ──────────────────────────────────────────
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
    //
    // When opts.sensitivity.param_names is non-empty, initialize CVODES
    // sensitivity analysis. CVODES computes dY/dp alongside the ODE integration
    // using its internal finite-difference approximation of the sensitivity RHS.
    // This works for ALL rate law types (Elementary, Functional, MichaelisMenten).

    const int n_sens_p = static_cast<int>(opts.sensitivity.param_names.size());
    const int n_sens_ic = static_cast<int>(opts.sensitivity.ic_species_names.size());
    const int n_sens = n_sens_p + n_sens_ic;
    NVectorArrayGuard yS_guard;               // RAII: frees sensitivity vectors
    std::vector<int> sens_param_indices;      // 0-based parameter indices
    std::vector<int> sens_ic_species_indices; // 0-based species indices for IC sens
    std::vector<double> pbar;                 // parameter scaling factors for CVODES
    std::vector<double> sens_p;               // contiguous parameter values (CVODES reads this)
    std::vector<int> sens_plist;              // which indices in sens_p to perturb
    // Hoisted so the event-fire sensitivity jump (GH #212) can re-init the
    // sensitivity vectors with the same method CVodeSensInit1 was given.
    int sens_method = CV_STAGGERED;

    if (n_sens > 0) {
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
            throw std::runtime_error(
                "sensitivity_ic requires codegen sensitivity RHS, but no "
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
                throw std::runtime_error("Sensitivity IC species '" + sname +
                                         "' not found in model.");
            }
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
                if (seed.empty() || !names_match ||
                    seed.size() != static_cast<size_t>(ns) * n_sens_p) {
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
            throw std::runtime_error(
                "sensitivity_ic (initial-condition sensitivities) across a "
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
        } else if (n_sens_p > 0 && !model.species_ic_param_refs().empty()) {
            std::unordered_map<int, int> param_to_sens_idx;
            param_to_sens_idx.reserve(static_cast<size_t>(n_sens_p));
            for (int iS = 0; iS < n_sens_p; ++iS) {
                param_to_sens_idx.emplace(sens_param_indices[iS], iS);
            }
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
                N_VGetArrayPointer(yS_guard[iS])[species_idx0] = 1.0;
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
        std::vector<double> sens_state_scale(static_cast<size_t>(ns));
        for (int i = 0; i < ns; ++i) {
            sens_state_scale[i] = std::max(std::abs(y_data[i]), 1.0);
        }
        NVectorArrayGuard abstolS_guard(N_VCloneVectorArray(n_sens, y), n_sens);
        if (!abstolS_guard) {
            throw std::runtime_error("N_VCloneVectorArray failed for sensitivity tolerances");
        }
        for (int iS = 0; iS < n_sens; ++iS) {
            double *atolS_col = N_VGetArrayPointer(abstolS_guard[iS]);
            const double pb = (pbar[iS] != 0.0) ? pbar[iS] : 1.0;
            for (int i = 0; i < ns; ++i) {
                atolS_col[i] = atol * sens_state_scale[i] / pb;
            }
        }
        flag = CVodeSensSVtolerances(cvode_mem, rtol, abstolS_guard.arr);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeSensSVtolerances failed: " + std::to_string(flag));
        }

        flag = CVodeSetSensErrCon(cvode_mem, opts.sensitivity.error_control);

        flag = CVodeSetSensParams(cvode_mem, sens_p.data(), pbar.data(), sens_plist.data());
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeSetSensParams failed: " + std::to_string(flag));
        }

        user_data.sens_p = sens_p.data();
        user_data.n_params = static_cast<int>(params.size());
    }

    // ─── Allocate result ─────────────────────────────────────────────────────

    const int n_func = model.n_functions();

    Result result;
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

    // Temporary observable buffer
    std::vector<double> obs_buf(n_obs);
    const auto ar_copyback = build_assignment_rule_copyback(model);

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

    // ─── Observable output sensitivities (GH #197) ───────────────────────────
    // BNGL observables are linear in species: obs_j = Σ_i c_ji·x_i, where c_ji
    // folds the GroupEntry factor and — for an amount-valued species — the
    // volume scaling that update_observables() applies (model.cpp:1142). So
    // d obs_j/dθ = Σ_i c_ji·dx_i/dθ: a runtime chain rule over the CVODES
    // species sensitivities extracted below, no codegen required. The same
    // coefficients drive both the parameter axis and the IC axis; only the
    // source dx/dθ vector differs (yS parameter cols vs IC cols). Expression
    // (global-function) sensitivities are nonlinear and are left to the codegen
    // stage (#198) — those blocks stay empty here.
    struct ObsSensTerm {
        int obs;       // observable row (0-based, recording order)
        int species0;  // 0-based species index it reads
        double weight; // c_ji = factor · (amount_valued ? volume_factor : 1)
    };
    std::vector<ObsSensTerm> obs_sens_terms;
    const bool compute_obs_sens_p = (n_sens_p > 0 && n_obs > 0);
    const bool compute_obs_sens_ic = (n_sens_ic > 0 && n_obs > 0);
    // Scratch outputs laid out [col][obs] so the pointer arrays below hand
    // record_observable_sensitivities* the sens_data[c][j] view it expects.
    std::vector<double> obs_sens_p_buf, obs_sens_ic_buf;
    std::vector<const double *> obs_sens_p_ptrs, obs_sens_ic_ptrs;
    if (compute_obs_sens_p || compute_obs_sens_ic) {
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
                if (sp.amount_valued) {
                    weight *= sp.volume_factor;
                }
                obs_sens_terms.push_back({j, idx0, weight});
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
                for (const auto &term : obs_sens_terms) {
                    out[term.obs] += term.weight * ys[term.species0];
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
    auto record_expression_output_sensitivities =
        [&](int time_index, double t_row, const double *const *sens_ptrs) {
            if (!compute_expr_sens) {
                return;
            }
            std::fill(func_sens_buf.begin(), func_sens_buf.end(), 0.0);
            user_data.codegen_output_sens_fn(
                t_row, y_data, user_data.codegen_so_data.param_values, sens_ptrs,
                sens_plist.data(), n_sens, /*obs_sens_out=*/nullptr, func_sens_buf.data(),
                &user_data.codegen_so_data);
            if (compute_expr_sens_p) {
                result.record_expression_sensitivities(time_index, func_sens_ptrs.data(), n_func,
                                                       n_sens_p);
            }
            if (compute_expr_sens_ic) {
                result.record_expression_sensitivities_ic(
                    time_index, func_sens_ptrs.data() + n_sens_p, n_func, n_sens_ic);
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
    const int n_roots = n_events + n_disc;

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
            process_firing_batch(t_now, risers);
        }
    };

    // ─── Forward-sensitivity jump across a fixed-time event (GH #212) ────────
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
    auto capture_event_sens = [&](double t_evt) -> std::vector<std::vector<double>> {
        std::vector<std::vector<double>> s_minus;
        if (!wants_sensitivity || n_sens == 0) {
            return s_minus;
        }
        sunrealtype t_tmp = static_cast<sunrealtype>(t_evt);
        int gf = CVodeGetSens(cvode_mem, &t_tmp, yS_guard.arr);
        if (gf != CV_SUCCESS) {
            throw std::runtime_error(
                "CVodeGetSens for event sensitivity capture failed: " + std::to_string(gf));
        }
        s_minus.resize(static_cast<size_t>(n_sens));
        for (int c = 0; c < n_sens; ++c) {
            const double *col = N_VGetArrayPointer(yS_guard[c]);
            s_minus[c].assign(col, col + ns);
        }
        return s_minus;
    };

    auto apply_event_sensitivity_jump = [&](double t_evt, const std::vector<int> &fired,
                                            const std::vector<double> &x_minus,
                                            const std::vector<std::vector<double>> &s_minus) {
        if (!wants_sensitivity || n_sens == 0 || fired.empty() || s_minus.empty()) {
            return;
        }

        auto &params = const_cast<std::vector<Parameter> &>(model.parameters());

        // Save the post-event state, then drive the evaluator to x⁻ so the
        // derivative evaluations below see pre-event values. Restored at the end.
        std::vector<double> x_post(static_cast<size_t>(ns));
        for (int i = 0; i < ns; ++i) {
            x_post[i] = sp_vec_outer[i].concentration;
        }
        std::vector<double> xwork(x_minus.begin(), x_minus.end());
        auto sync_state = [&]() {
            for (int i = 0; i < ns; ++i) {
                sp_vec_outer[i].concentration = xwork[i];
            }
            model.update_observables(xwork.data());
            model.evaluate_functions(t_evt);
        };
        sync_state();

        for (int ei : fired) {
            const auto &ev = events_outer[ei];
            for (const auto &asg : ev.assignments) {
                const int k = asg.first;      // assigned species (0-based)
                const int vexpr = asg.second; // value expression id
                if (k < 0 || k >= ns) {
                    continue;
                }
                // Restrict the FD to the variables the assignment value reads.
                const auto refs = eval_ref_outer.referenced_variable_addresses(vexpr);
                std::unordered_set<const double *> refset(refs.begin(), refs.end());

                // ∂c/∂x_j via central FD (only for referenced species).
                std::vector<double> dcdx(static_cast<size_t>(ns), 0.0);
                for (int j = 0; j < ns; ++j) {
                    if (refset.find(&sp_vec_outer[j].concentration) == refset.end()) {
                        continue;
                    }
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
                    if (refset.find(&params[pidx].value) == refset.end()) {
                        continue;
                    }
                    const double p0 = params[pidx].value;
                    double h = 1e-6 * std::fabs(p0);
                    if (h == 0.0) {
                        h = 1e-9;
                    }
                    params[pidx].value = p0 + h;
                    model.evaluate_functions(t_evt);
                    const double f_hi = eval_ref_outer.evaluate(vexpr);
                    params[pidx].value = p0 - h;
                    model.evaluate_functions(t_evt);
                    const double f_lo = eval_ref_outer.evaluate(vexpr);
                    params[pidx].value = p0; // restore
                    dcdp[col] = (f_hi - f_lo) / (2.0 * h);
                }
                model.evaluate_functions(t_evt); // restore function state at (x⁻, p₀)

                // Assemble s⁺_k for every sensitivity column.
                for (int c = 0; c < n_sens; ++c) {
                    double acc = 0.0;
                    const std::vector<double> &sm = s_minus[c];
                    for (int j = 0; j < ns; ++j) {
                        if (dcdx[j] != 0.0) {
                            acc += dcdx[j] * sm[j];
                        }
                    }
                    if (c < n_sens_p) {
                        acc += dcdp[c];
                    }
                    N_VGetArrayPointer(yS_guard[c])[k] = acc;
                }
            }
        }

        // Restore the post-event state so downstream code and resumed
        // integration see x⁺ (not the x⁻ used for differentiation).
        xwork.assign(x_post.begin(), x_post.end());
        sync_state();

        int rf = CVodeSensReInit(cvode_mem, sens_method, yS_guard.arr);
        if (rf != CV_SUCCESS) {
            throw std::runtime_error(
                "CVodeSensReInit after event sensitivity jump failed: " + std::to_string(rf));
        }
    };

    if (n_roots > 0) {
        // Root function callback: evaluate each event trigger then each
        // discontinuity-trigger condition, subtracting 0.5 so a false→true (or
        // true→false) flip is a sign change. Event roots occupy gout[0,n_events)
        // and discontinuity roots gout[n_events, n_roots).
        auto root_fn = [](sunrealtype t, N_Vector y, sunrealtype *gout, void *user_data) -> int {
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
                mdl->compute_derivs(static_cast<double>(t), y_ptr,
                                    data->rateof_root_scratch.data());
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
            return 0;
        };

        flag = CVodeRootInit(cvode_mem, n_roots, root_fn);
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
                t0_s_minus = capture_event_sens(static_cast<double>(times.t_start));
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
                int reinit_flag = CVodeReInit(cvode_mem, times.t_start, y);
                if (reinit_flag != CV_SUCCESS) {
                    throw std::runtime_error("CVodeReInit after t=0 event failed: " +
                                             std::to_string(reinit_flag));
                }
                // Jump dx/dp across the t=0 event and re-seed CVODES sensitivity
                // vectors (GH #212). No-op unless sensitivities are active.
                apply_event_sensitivity_jump(static_cast<double>(times.t_start), t0_firing,
                                             t0_x_minus, t0_s_minus);
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
    // falls below ``ss_tol``. ``ss_tol`` defaults to the integrator atol.
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
            if (n_events > 0) {
                for (const auto &pe : pending_events) {
                    if (pe.apply_time > static_cast<double>(t_now) + 1e-15 &&
                        pe.apply_time < t_target) {
                        t_target = pe.apply_time;
                    }
                }
            }

            sunrealtype t_ret;
            flag = CVode(cvode_mem, t_target, y, &t_ret, CV_NORMAL);

            // CV_TOO_MUCH_WORK is recoverable, not a failure: CVODE merely
            // used its per-call step budget (max_steps) without reaching
            // t_target. The integrator state (t, y, step size, order) is
            // intact, so we simply call CVode again to continue -- max_steps
            // acts as a batch size, not a hard ceiling. The wall-clock
            // budget is re-checked between batches so a genuinely
            // non-terminating integration is still bounded by `timeout`.
            while (flag == CV_TOO_MUCH_WORK) {
                if (budget.active())
                    budget.check();
                flag = CVode(cvode_mem, t_target, y, &t_ret, CV_NORMAL);
            }

            if (flag < 0) {
                throw std::runtime_error(
                    "CVODE integration failed at t=" + std::to_string(t_target) +
                    " with flag=" + std::to_string(flag));
            }
            t_now = t_ret;

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
                    evt_s_minus = capture_event_sens(static_cast<double>(t_ret));
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
                        if (std::fabs(y_data[si] - chatter_y_before[si]) >
                            atol + rtol * std::fabs(y_data[si])) {
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
                int reinit_flag = CVodeReInit(cvode_mem, t_ret, y);
                if (reinit_flag != CV_SUCCESS) {
                    throw std::runtime_error("CVodeReInit after event failed: " +
                                             std::to_string(reinit_flag));
                }
                // Jump dx/dp across the event and re-seed CVODES sensitivity
                // vectors (GH #212). chatter_y_before holds this batch's
                // pre-fire state x⁻ (snapshotted above whenever firing is
                // non-empty, which any_event_fired implies). No-op unless
                // sensitivities are active.
                if (any_event_fired) {
                    apply_event_sensitivity_jump(static_cast<double>(t_ret), firing,
                                                 chatter_y_before, evt_s_minus);
                }
                // Inner-while-loop will continue to t_out[i] (or to the next
                // pending apply_time if one is closer) on its next pass.
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
                        const bool ode_only = a < ev_pe.assignment_ode_only.size() &&
                                              ev_pe.assignment_ode_only[a];
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
                    int reinit_flag = CVodeReInit(cvode_mem, t_ret, y);
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
                int rf = CVodeReInit(cvode_mem, t_now, y);
                if (rf != CV_SUCCESS) {
                    throw std::runtime_error("CVodeReInit after chatter re-arm failed: " +
                                             std::to_string(rf));
                }
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

    long int nst, nfe, nsetups, nni, ncfn, netf;
    CVodeGetNumSteps(cvode_mem, &nst);
    CVodeGetNumRhsEvals(cvode_mem, &nfe);
    CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    CVodeGetNumErrTestFails(cvode_mem, &netf);

    result.solver_stats().n_steps = static_cast<int>(nst);
    result.solver_stats().n_rhs_evals = static_cast<int>(nfe);
    result.solver_stats().n_jac_evals = static_cast<int>(nsetups);
    result.solver_stats().n_nonlin_iters = static_cast<int>(nni);
    result.solver_stats().n_nonlin_conv_fails = static_cast<int>(ncfn);
    result.solver_stats().n_err_test_fails = static_cast<int>(netf);
    result.solver_stats().linear_solver = impl_->linear_solver_used;
    // GH #132: BLAS dgetrf factorization count for this run (0 unless LAPACK-dense
    // and the adaptive K gate was crossed). LS_guard is the solver built above.
    {
        const long bc = lapack_dense_blas_factor_count(LS_guard);
        result.solver_stats().n_dense_blas_factorizations = bc > 0 ? static_cast<int>(bc) : 0;
    }

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
    // After simulation, update the model's species concentrations to the
    // final state. This is essential for multi-action sequences where
    // saveConcentrations() or subsequent simulate() actions depend on
    // the post-simulation state (matching BNG's propagate_cvode_network
    // behavior which writes back to the global species array). When the
    // steady-state early-stop fired, the "final time" is the last sample we
    // actually integrated to, not the originally requested ``t_end``.
    {
        auto &species = const_cast<std::vector<Species> &>(model.species());
        for (int i = 0; i < ns; ++i) {
            species[i].concentration = y_data[i];
        }
        const double final_t = (check_ss && ss_reached) ? t_out[last_recorded_index] : times.t_end;
        model.set_current_time(final_t);

        // GH #210 — the state is now advanced past the ICs (carry-over). Thread
        // the forward-sensitivity matrix dx/dθ at this point into the model so a
        // subsequent carry_sensitivities=True run (the measurement phase of a
        // pre-equilibration) can seed yS(0) from it. yS_guard holds the final
        // integrated point (the loop's last CVodeGetSens). Capture the
        // parameter columns only, row-major [species*np + param]. A
        // non-sensitivity run still marks the state dirty but drops any stale
        // seed (it advanced the state without tracking dx/dθ).
        model.set_ic_state_dirty(true);
        if (n_sens_p > 0) {
            std::vector<double> seed(static_cast<size_t>(ns) * n_sens_p);
            for (int iS = 0; iS < n_sens_p; ++iS) {
                const double *col = N_VGetArrayPointer(yS_guard[iS]);
                for (int i = 0; i < ns; ++i) {
                    seed[static_cast<size_t>(i) * n_sens_p + iS] = col[i];
                }
            }
            model.set_pending_sens_seed(std::move(seed), opts.sensitivity.param_names);
        } else {
            model.clear_pending_sens_seed();
        }
    }

    // ─── Cleanup ─────────────────────────────────────────────────────────────

    // All SUNDIALS resources are freed automatically by the RAII guards:
    //   yS_guard, LS_guard, A_guard, cvode_mem, y, ctx, codegen_param_buf
    // The cached codegen library (impl_->codegen_lib) intentionally stays open
    // for the simulator's lifetime and is unloaded by ~Impl (GH #77).

    return result;
}

} // namespace bngsim
