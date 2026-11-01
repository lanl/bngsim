// bngsim/include/bngsim/nonneg_clamp.hpp -- which state slots the GH #135
// non-finite-RHS retry may clamp to 0, and the clamp itself.
//
// One rule, two integrators (issue #1357). The time course (cvode_simulator.cpp)
// has retried a non-finite RHS and Jacobian on the nonnegative-clamped state
// since GH #135, over the slots issue #706 allows. The steady-state run test
// (steady_state.cpp) now retries the same way, so the slots, the clamp and the
// reading that triggers it are defined here, once.
#pragma once

#include "bngsim/model.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace bngsim {

// 1 for a slot that is a concentration by construction, 0 for one that may be
// negative by design (issue #706): a fixed species, one not reported as a
// species (a promoted parameter or compartment), one declared with a negative
// initial value, and the target of a rate rule.
inline std::vector<char> nonneg_clamp_mask_of(const NetworkModel &model) {
    const int ns = model.n_species();
    std::vector<char> mask(ns, 1);
    const auto &species = model.species();
    for (int i = 0; i < ns; ++i) {
        const Species &sp = species[i];
        if (sp.fixed || !sp.reported || sp.declared_negative)
            mask[i] = 0;
    }
    for (const Reaction &rxn : model.reactions()) {
        if (!rxn.is_rate_rule_ode)
            continue;
        for (int si : rxn.product_indices) {
            if (si >= 1 && si <= ns)
                mask[si - 1] = 0;
        }
    }
    return mask;
}

// `y` with every slot the mask allows, and that is not positive, put at 0, into
// `out` (sized to the mask), but for the slots `spare` marks: the time course's
// spare_at_pole, which the run test does not use.
inline void clamp_nonneg_into(const std::vector<char> &mask, const double *y,
                              std::vector<double> &out, const char *spare = nullptr) {
    const std::size_t n = mask.size();
    if (out.size() != n)
        out.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (y[i] > 0.0 || !mask[i] || (spare != nullptr && spare[i])) ? y[i] : 0.0;
}

// What an evaluation of the right-hand side or of a closed-form Jacobian read:
// all finite, a NaN and no infinity, or an infinity. The time course retries
// either non-finite reading; the run test retries only the NaN one.
enum class ClampReading { finite, nan_only, infinite };

inline ClampReading read_values(const double *v, std::size_t n) {
    bool nan = false;
    for (std::size_t k = 0; k < n; ++k) {
        if (std::isinf(v[k]))
            return ClampReading::infinite;
        nan = nan || std::isnan(v[k]);
    }
    return nan ? ClampReading::nan_only : ClampReading::finite;
}

} // namespace bngsim
