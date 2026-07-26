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
//   bngsim_codegen_jac        (generate_jacobian_*)   — dense column-major J
//   bngsim_codegen_jac_sparse (generate_jacobian_*)   — CSC value array
//   bngsim_codegen_outputs    (generate_outputs_*)    — observable/function values
//   bngsim_codegen_output_sens(generate_output_sens_*)— d(output)/dθ
//
// Only the RHS is mandatory; every other symbol is resolved with try_symbol and
// may be absent (a Functional/MM model has no analytical sens RHS, a model
// without a complete analytical Jacobian has no compiled Jacobian, etc.).

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
