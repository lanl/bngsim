// bngsim/include/bngsim/atol_vector.hpp — per-species absolute tolerance (issue #196)
//
// A scalar `atol` is one number asked to mean the same thing for every state.
// On a model whose species span decades that is not a tolerance, it is a choice
// of which end of the model to resolve — the value the smallest species needs
// makes the model unintegrable, the value the model integrates at leaves the
// smallest species under the noise floor, and no scalar sits between them.
// CVODE's answer is `CVodeSVtolerances`, a per-species absolute tolerance
// vector, which bngsim already uses one axis over (the sensitivity columns take
// `CVodeSensSVtolerances`).
//
// Two functions, shared by the CVODE march in cvode_simulator.cpp and the
// steady-state march in steady_state.cpp so the contract is stated once:
//
//   validate_atol_vector — the length/value contract, checked where the caller
//                          can still be told which call was wrong.
//   apply_cvode_tolerances — SS vs SV dispatch. An empty vector means the
//                          scalar path, byte-for-byte as before #196.

#pragma once

#include <cmath>
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bngsim/sundials_guards.hpp"

namespace bngsim {

// Check a per-species absolute tolerance vector against the model it will be
// applied to. `where` names the caller in the message (e.g. "run()").
//
// An empty vector is always valid — it means "no per-species tolerance", the
// scalar path. A non-empty one must have exactly `n_species` entries, because
// the alternative is broadcasting or truncating, and both of those silently
// hand species i the tolerance the caller wrote for species j.
inline void validate_atol_vector(const std::vector<double> &atol_vec, int n_species,
                                 const char *where) {
    if (atol_vec.empty()) {
        return;
    }
    if (static_cast<int>(atol_vec.size()) != n_species) {
        std::ostringstream msg;
        msg << where << ": per-species atol has " << atol_vec.size() << " entries but the model "
            << "has " << n_species << " species. The vector is positional — entry i is the "
            << "absolute tolerance for species i, ordered like Model.species_names — so a "
            << "length mismatch is rejected rather than broadcast or truncated.";
        throw std::invalid_argument(msg.str());
    }
    for (size_t i = 0; i < atol_vec.size(); ++i) {
        const double a = atol_vec[i];
        if (!std::isfinite(a) || a < 0.0) {
            std::ostringstream msg;
            msg << where << ": per-species atol[" << i << "] = " << a
                << " is not a usable absolute tolerance; every entry must be finite and >= 0 "
                << "(CVODE rejects a negative abstol outright, and a NaN would silently disable "
                << "error control on that species).";
            throw std::invalid_argument(msg.str());
        }
    }
}

// Hand CVODE the tolerances for this run: `CVodeSVtolerances` when `atol_vec`
// is non-empty, else `CVodeSStolerances` with the scalar. Must be called after
// CVodeInit (CVODE rejects both before its own allocation).
//
// CVODE copies the abstol vector into its own storage (N_VScale into
// cv_Vabstol), so the N_Vector built here is free to die at the end of the
// call — the tolerances outlive it.
//
// One thing NOT to expect: a constant vector is not bit-identical to the same
// scalar. The two paths compute the same error weights by different
// expressions — cvEwtSetSS scales then adds a constant, cvEwtSetSV takes one
// fused N_VLinearSum — and the second is FMA-contractable where the first is
// not, which is a ~1 ulp difference in ewt and a rounding-level difference in
// the trajectory. What IS guaranteed is that a caller who never sets a vector
// stays on the identical code path they were on before #196.
inline void apply_cvode_tolerances(void *cvode_mem, SUNContext ctx, double rtol, double atol,
                                   const std::vector<double> &atol_vec, int n_species) {
    if (atol_vec.empty()) {
        const int flag = CVodeSStolerances(cvode_mem, rtol, atol);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeSStolerances failed");
        }
        return;
    }
    validate_atol_vector(atol_vec, n_species, "CVODE setup");
    NVectorGuard abstol(N_VNew_Serial(n_species, ctx));
    if (!abstol) {
        throw std::runtime_error("N_VNew_Serial failed for the per-species atol vector");
    }
    double *a = abstol.data();
    for (int i = 0; i < n_species; ++i) {
        a[i] = atol_vec[static_cast<size_t>(i)];
    }
    const int flag = CVodeSVtolerances(cvode_mem, rtol, abstol);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSVtolerances failed: " + std::to_string(flag));
    }
}

} // namespace bngsim
