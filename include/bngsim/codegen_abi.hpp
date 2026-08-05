// bngsim/include/bngsim/codegen_abi.hpp — the codegen .so / MIR-JIT calling contract
//
// The C source emitted by ``bngsim/python/bngsim/_codegen.py`` exports a fixed
// set of symbols with a fixed set of signatures, and passes state through two
// plain structs whose FIELD ORDER IS PART OF THE ABI. Both the CVODES simulator
// (src/cvode_simulator.cpp) and the steady-state solver (src/steady_state.cpp)
// resolve those symbols, so the declarations live here rather than being
// duplicated per translation unit — a divergence between two copies of an ABI
// struct is silent memory corruption, not a compile error.
//
// Anything added here must mirror the emitter. The current contract:
//
//   bngsim_codegen_rhs        (generate_rhs_c)        — dx/dt
//   bngsim_codegen_sens_rhs   (generate_sens_rhs_c)   — ySdot = J·yS + ∂f/∂p_iS
//   bngsim_codegen_sens_term_scale (same emitter)     — Σ|term| per row of ∂f/∂p_iS
//   bngsim_codegen_jac        (generate_jacobian_*)   — dense column-major J
//   bngsim_codegen_jac_sparse (generate_jacobian_*)   — CSC value array
//   bngsim_codegen_outputs    (generate_outputs_*)    — observable/function values
//   bngsim_codegen_output_sens(generate_output_sens_*)— d(output)/dθ
//
// Only the RHS is mandatory; every other symbol is resolved with try_symbol and
// may be absent (a Functional/MM model has no analytical sens RHS, a model
// without a complete analytical Jacobian has no compiled Jacobian, etc.).
//
// ─── The int return value, and why two consumers ignore it ───────────────────
//
// Every function here returns int, and the intended meaning is CVODES': 0 on
// success, > 0 recoverable (retry the step), < 0 unrecoverable. Nothing has ever
// returned anything else. `return 0;` is the ONLY return statement the emitter
// produces, in every exported function and on every branch it has — dense and
// CSC Jacobian, flat and GH #165 chunked, Elementary / Michaelis-Menten /
// Functional. There is no early exit to carry an error out of: a domain problem
// shows up as a non-finite value in the output buffer, which is what the
// callers' jac_has_nonfinite / rhs_has_nonfinite clamp retries exist to catch.
//
// So the consumers split, deliberately, and neither side is an oversight:
//
//   * cvode_simulator.cpp propagates it (cvode_codegen_rhs,
//     cvode_codegen_dense_jac, cvode_codegen_sparse_jac all `return rc`). They
//     are CVODE callbacks whose signature already has the channel, so honoring
//     it costs one line.
//   * steady_state.cpp does NOT (SteadyStateRhs::eval, ::fill_dense_jacobian,
//     ::fill_sparse_jacobian are void). Those are library methods with five
//     callers between them, one of which — ss_fill_state_jacobian, shared by
//     dY_ss/dp and the issue #78 certificate — has no error channel to forward
//     to. Threading one through for a code that cannot be produced would add a
//     failure path nothing can reach and nothing can test.
//
// **If you ever emit a nonzero return, that split stops being safe** and
// steady_state.cpp's three methods have to grow the channel. The invariant is
// pinned rather than merely written down here:
// python/tests/test_codegen_jacobian.py::test_emitted_c_has_no_nonzero_return
// fails on the first `return 1;` anyone adds, and says what to fix.

#pragma once

namespace bngsim {

// Tfun callback type: invoked by the codegen .so to evaluate a table function at
// the given index value. ctx is opaque on the .so side; callers set it to the
// owning NetworkModel pointer.
using CodegenTfunEvalFn = double (*)(int tf_id, double x, void *ctx);

// Lightweight struct matching the CodegenUserData layout expected by the .so.
// MUST mirror the typedef emitted by _codegen.py (generate_rhs_c). Field order
// is part of the ABI contract between the codegen .so and its callers.
struct CodegenUserDataForSO {
    double *param_values;
    void *tfun_ctx;
    CodegenTfunEvalFn tfun_eval;
};

// The user_data struct the codegen *sensitivity* RHS expects. Mirrors the
// CodegenSensUserData typedef emitted by generate_sens_rhs_c; same ABI caveat.
struct CodegenSensUserDataForSO {
    double *param_values;
    int *plist; // plist[iS] = parameter index for sensitivity direction iS
    int n_sens;
};

// dx/dt. Reads parameters from CodegenUserDataForSO::param_values (NOT from the
// model object), zeroes fixed-species rows itself.
using CodegenRhsFn = int (*)(double t, double *y, double *ydot, void *user_data);

// CVSensRhs1Fn-compatible: ySdot = J(t,y)·yS + ∂f/∂p_{plist[iS]}. user_data is a
// CodegenSensUserDataForSO. tmp1/tmp2 are unused by the current emitter but are
// part of the CVODES signature.
//
// Note for non-CVODES callers: passing an all-zero yS makes the J·yS term vanish,
// so ySdot comes back as the bare analytical ∂f/∂p column — which is how the
// steady-state solver gets ∂f/∂p without differencing anything.
using CodegenSensRhsFn = int (*)(int Ns, double t, double *y, double *ydot, int iS, double *yS,
                                 double *ySdot, void *user_data, double *tmp1, double *tmp2);

// Issue #177: scale_out[i] = Σ|term| over the very contributions the function
// above sums into row i of ∂f/∂p_{plist[iS]} — the magnitude of the arithmetic
// behind a value, which the value itself does not carry. ε·scale_out[i] is the
// roundoff of that row, and a sensitivity absolute tolerance set below it asks
// for accuracy float64 does not have, so CVODES shrinks h without bound.
//
// Emitted alongside bngsim_codegen_sens_rhs and takes the same
// CodegenSensUserDataForSO, so wherever the analytic sensitivity RHS is
// resolved this is too. Resolved with try_symbol like everything else here: a
// .so built before v28 of the emitter simply does not have it, and the caller
// then keeps the unfloored tolerance rather than failing to load.
using CodegenSensTermScaleFn = int (*)(int Ns, double t, double *y, int iS, double *scale_out,
                                       void *user_data);

// Dense analytical Jacobian into an n×n COLUMN-MAJOR buffer
// (jac[j*n + i] = ∂f_i/∂x_j). The emitted C memsets the buffer itself.
using CodegenJacFn = int (*)(double t, double *y, double *jac_colmajor, void *user_data);

// Sparse (CSC) analytical Jacobian: fills the nnz-length value array indexed by
// the model's sparsity pattern. The emitted C memsets the array itself.
using CodegenJacSparseFn = int (*)(double t, double *y, double *jac_data, void *user_data);

// Observable / expression value evaluator: fills obs_out[N_OBS], func_out[N_FUNC].
using CodegenOutputsFn = int (*)(double t, double *y, double *obs_out, double *func_out,
                                 void *user_data);

// Observable + expression output-sensitivity evaluator (GH #198):
// func_sens_out[c*N_FUNC + m] = d func_m/dθ_c from the per-column state
// sensitivities. plist[c] is the differentiated parameter index for a parameter
// column (>= N_PARAMS marks an IC column, which skips the parameter term).
using CodegenOutputSensFn = int (*)(double t, const double *y, const double *p,
                                    const double *const *state_sens, const int *plist, int n_sens,
                                    double *obs_sens_out, double *func_sens_out, void *user_data);

} // namespace bngsim
