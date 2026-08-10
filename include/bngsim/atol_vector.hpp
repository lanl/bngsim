// bngsim/include/bngsim/atol_vector.hpp — per-species absolute tolerance (issue #196)
//                                          and its over-time twin (issue #213)
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
// A vector fixes the CROSS-SPECIES compromise and nothing else. Whatever number
// species i gets, it gets for the whole run — so a species that starts at order
// one and decays to something tiny outgrows its own tolerance partway through
// and stops being error-controlled from there on. That is the same failure
// reached through time rather than through the species list, and CVODE's
// construct for it is `CVodeWFtolerances`: instead of a fixed vector you hand
// the integrator a callback that computes the error weights at the state
// actually being integrated. Issue #213 is that third mode, spelled here as
// AtolTracking.
//
// Four functions, shared by the CVODE march in cvode_simulator.cpp and the
// steady-state march in steady_state.cpp so the contract is stated once:
//
//   validate_atol_vector — the length/value contract, checked where the caller
//                          can still be told which call was wrong.
//   validate_atol_tracking — the extra contract tracking adds on top of it.
//   make_atol_tracking   — build the ceiling/floor pair from a vector + depth.
//   apply_cvode_tolerances — SS vs SV vs WF dispatch. An empty vector and a
//                          null error-weight function mean the scalar path,
//                          byte-for-byte as before #196.

#pragma once

#include <cmath>
#include <cvodes/cvodes.h>
#include <limits>
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

// ─── Tracking absolute tolerance (issue #213) ────────────────────────────────
//
// The per-species vector re-evaluated against the trajectory instead of against
// the initial state:
//
//     atol_i(y) = clamp(rtol·|y_i|, lo_i, hi_i)
//     ewt_i(y)  = 1 / (rtol·|y_i| + atol_i(y))
//
// `hi` is the #196 vector — the tolerance species i would have had for the
// whole run — and `lo` is `hi` scaled down by the requested depth. So the rule
// reads: hold every species to `rtol` of the magnitude it currently has, but
// never looser than the vector already asked for, and stop tightening once it
// has fallen `decades` decades below where it started.
//
// Two properties that make it safe to reach for, and that the tests pin:
//
//   * At depth 0, lo == hi and this reduces to the #196 vector exactly. The
//     mode is a strict extension of the vector, not a different rule that
//     happens to agree somewhere.
//   * The clamp's upper end is `hi`, so tracking is never LOOSER than the
//     vector it was built from — only tighter, and by at most `decades`.
//
// And one that makes it usable at all: it is a pure function of `y`. A running
// maximum or an RMS over the trajectory would make the error weights depend on
// the step history, which depends on the error weights — the same state would
// integrate differently depending on how it was reached, and a rejected step
// would leave a mark. This rule has no state to carry across a CVodeReInit, so
// it survives every event fire and switch crossing without reset semantics.
struct AtolTracking {
    double rtol = 0.0;
    std::vector<double> hi; // ceiling: the #196 per-species vector
    std::vector<double> lo; // floor: hi scaled down by the requested depth

    bool active() const { return !hi.empty(); }
};

// The contract tracking adds on top of validate_atol_vector.
//
// `decades` <= 0 means "off" and is always valid. A positive depth needs a
// ceiling to hang off, and every entry of that ceiling must be strictly
// positive: a zero ceiling scales to a zero floor, and `atol_i == 0` at a
// species sitting at exactly zero makes ewt_i infinite. CVODE checks that for
// its own built-in weight routines and explicitly does NOT for a user-supplied
// one (cvInitialSetup skips the N_VMin test when cv_user_efun is set), so the
// check has to happen here or not at all.
inline void validate_atol_tracking(double decades, const std::vector<double> &atol_vec,
                                   int n_species, const char *where) {
    if (!(decades > 0.0)) {
        if (std::isnan(decades) || decades < 0.0) {
            std::ostringstream msg;
            msg << where << ": tracking depth " << decades
                << " is not a usable number of decades; it must be finite and >= 0 (0 turns "
                << "tracking off and leaves the per-species vector in force unchanged).";
            throw std::invalid_argument(msg.str());
        }
        return;
    }
    if (!std::isfinite(decades)) {
        std::ostringstream msg;
        msg << where << ": tracking depth is infinite. A tracking tolerance with no floor is "
            << "pure relative error control, which has no weight to give a species sitting at "
            << "exactly zero; ask for a finite number of decades instead.";
        throw std::invalid_argument(msg.str());
    }
    if (n_species == 0) {
        // No ODE state to weight (GH #229's algebraic-only model, which bails
        // out before CVODE is created at all). An empty vector is the correct
        // "auto" ceiling there, and refusing it would make tracking the one
        // atol form such a model cannot be handed.
        return;
    }
    validate_atol_vector(atol_vec, n_species, where);
    if (atol_vec.empty()) {
        std::ostringstream msg;
        msg << where << ": a tracking absolute tolerance needs a per-species ceiling to track "
            << "below, and none was supplied. The ceiling is the same n_species vector the "
            << "scalar-free path takes (issue #196) — tracking scales it down by " << decades
            << " decades as each species falls, it does not invent one.";
        throw std::invalid_argument(msg.str());
    }
    const double scale = std::pow(10.0, -decades);
    for (size_t i = 0; i < atol_vec.size(); ++i) {
        if (!(atol_vec[i] > 0.0)) {
            std::ostringstream msg;
            msg << where << ": tracking ceiling[" << i << "] = " << atol_vec[i]
                << ", but every entry must be strictly > 0 under tracking. A zero ceiling "
                << "scales to a zero floor, and a species then at exactly zero would have an "
                << "infinite error weight — the run would fail on the first step with nothing "
                << "to point at.";
            throw std::invalid_argument(msg.str());
        }
        if (!(atol_vec[i] * scale >= std::numeric_limits<double>::min())) {
            std::ostringstream msg;
            msg << where << ": tracking ceiling[" << i << "] = " << atol_vec[i] << " underflows "
                << "to " << (atol_vec[i] * scale) << " — subnormal or zero — " << decades
                << " decades down. Ask for fewer decades, or raise that entry. Rejected here "
                << "rather than passed on because CVODE's report for an unmeetable tolerance "
                << "is a corrector convergence failure at t=0, which names nothing.";
            throw std::invalid_argument(msg.str());
        }
    }
}

// Build the ceiling/floor pair. Validate first; this assumes it has passed.
inline AtolTracking make_atol_tracking(double rtol, const std::vector<double> &atol_vec,
                                       double decades) {
    AtolTracking t;
    if (!(decades > 0.0) || atol_vec.empty()) {
        return t;
    }
    const double scale = std::pow(10.0, -decades);
    t.rtol = rtol;
    t.hi = atol_vec;
    t.lo.resize(atol_vec.size());
    for (size_t i = 0; i < atol_vec.size(); ++i) {
        t.lo[i] = atol_vec[i] * scale;
    }
    return t;
}

// The body of the CVEwtFn. Each translation unit wraps this in a two-line
// callback that casts its own user-data type, because CVODE hands the error
// weight function whatever went to CVodeSetUserData and that is a different
// struct in cvode_simulator.cpp than in steady_state.cpp.
//
// Returns 0 on success and -1 if any weight came out non-positive or
// non-finite, which CVODE reports as CV_ILL_INPUT / MSGCV_EWT_FAIL rather than
// integrating against a garbage norm. validate_atol_tracking makes that
// unreachable for a spec built here; it stays because the alternative to a
// checked division is an inf silently entering the WRMS norm.
inline int fill_tracking_ewt(const AtolTracking &t, N_Vector y, N_Vector ewt) {
    const sunindextype n = static_cast<sunindextype>(t.hi.size());
    if (N_VGetLength(y) != n) {
        return -1;
    }
    const sunrealtype *yd = N_VGetArrayPointer(y);
    sunrealtype *w = N_VGetArrayPointer(ewt);
    for (sunindextype i = 0; i < n; ++i) {
        const double rel = t.rtol * std::abs(static_cast<double>(yd[i]));
        double a = rel;
        const size_t k = static_cast<size_t>(i);
        if (a < t.lo[k]) {
            a = t.lo[k];
        } else if (a > t.hi[k]) {
            a = t.hi[k];
        }
        const double denom = rel + a;
        if (!(denom > 0.0) || !std::isfinite(denom)) {
            return -1;
        }
        w[i] = static_cast<sunrealtype>(1.0 / denom);
    }
    return 0;
}

// Hand CVODE the tolerances for this run: `CVodeWFtolerances` when `ewt_fn` is
// non-null (issue #213), else `CVodeSVtolerances` when `atol_vec` is non-empty,
// else `CVodeSStolerances` with the scalar. Must be called after CVodeInit
// (CVODE rejects all three before its own allocation).
//
// CVODE copies the abstol vector into its own storage (N_VScale into
// cv_Vabstol), so the N_Vector built here is free to die at the end of the
// call — the tolerances outlive it.
//
// One thing NOT to expect: a constant vector is not bit-identical to the same
// scalar, and a depth-0 tracking spec is not bit-identical to the vector it was
// built from. All three paths compute the same error weights by different
// expressions — cvEwtSetSS scales then adds a constant, cvEwtSetSV takes one
// fused N_VLinearSum, fill_tracking_ewt writes the two roundings out — and they
// are contractable to different degrees, which is a ~1 ulp difference in ewt
// and a rounding-level difference in the trajectory. What IS guaranteed is that
// a caller who never sets a vector and never asks for tracking stays on the
// identical code path they were on before #196.
inline void apply_cvode_tolerances(void *cvode_mem, SUNContext ctx, double rtol, double atol,
                                   const std::vector<double> &atol_vec, int n_species,
                                   CVEwtFn ewt_fn = nullptr) {
    if (ewt_fn != nullptr) {
        const int flag = CVodeWFtolerances(cvode_mem, ewt_fn);
        if (flag != CV_SUCCESS) {
            throw std::runtime_error("CVodeWFtolerances failed: " + std::to_string(flag));
        }
        return;
    }
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
