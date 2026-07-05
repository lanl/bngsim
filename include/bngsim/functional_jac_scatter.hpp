// bngsim/include/bngsim/functional_jac_scatter.hpp — per-observable Functional
// Jacobian scatter (GH #76 task 2)
//
// Single source of truth for the `.net` per-observable Functional Jacobian
// contribution, shared by the dense path (NetworkModel::fill_dense_analytical_
// jacobian, which also backs the FD self-check and the dense CVODE callback) and
// the sparse CVODE callback (cvode_analytical_jac). Both call this template with
// a small `emit` lambda that writes one J[row][col] += value, so the product-rule
// math has exactly one implementation.
//
// For a Functional reaction with rate = func(observables) · ∏_reactants x_i^{m_i}
// (apply_species_factor=true, non-empty reactant multiset):
//   ∂rate/∂x_j = (∂func/∂x_j)·∏R + func·∂(∏R)/∂x_j
// where ∂func/∂x_j = Σ_k (∂func/∂obs_k)·(∂obs_k/∂x_j). The ∂func/∂obs_k come from
// the Python symbolic core (ExprTk ids); ∂obs_k/∂x_j and ∂(∏R)/∂x_j are assembled
// here from the model's observable groups and reactant multisets.
//
// Precondition: the caller has already called update_observables(conc) and
// set_current_time(t). This helper additionally calls evaluate_functions(t) to
// refresh the function-bound rate parameters that supply `func` (the per-species
// path does not need this, so callers keep it out of that branch).

#pragma once

#include "bngsim/model.hpp"

#include <vector>

namespace bngsim {

// emit(int64_t csc_idx, int col, int row, double value) accumulates
// J[row][col] += value. csc_idx is the CSC data index of (row, col); the dense
// caller ignores it and indexes jac[col*ns + row], the sparse caller writes
// jac_data[csc_idx].
template <typename Emit>
inline void scatter_functional_observable_terms(NetworkModel &model, const double *conc, double t,
                                                Emit &&emit) {
    const auto &fjac = model.functional_jacobian();
    if (fjac.observable_terms.empty())
        return;

    const auto &sp = model.jacobian_sparsity();
    auto &eval = model.evaluator();
    const auto &params = model.parameters();
    const auto &species = model.species();

    // Refresh function-bound rate parameters so `func` matches the RHS exactly.
    model.evaluate_functions(t);

    std::vector<double> dbuf; // ∂func/∂obs_k values for the current reaction
    for (const auto &ot : fjac.observable_terms) {
        const double func =
            (ot.func_param_idx0 >= 0 && ot.func_param_idx0 < static_cast<int>(params.size()))
                ? params[ot.func_param_idx0].value
                : 0.0;

        // ∏R = ∏_reactants v_i^{m_i}, with v_i = conc·V_i for amount-valued i.
        double P = 1.0;
        for (const auto &[s, m] : ot.reactants) {
            double v = conc[s];
            if (species[s].amount_valued)
                v *= species[s].volume_factor;
            for (int p = 0; p < m; ++p)
                P *= v;
        }

        // Pre-evaluate ∂func/∂obs_k once (each column references them by index).
        dbuf.clear();
        dbuf.reserve(ot.dfunc_dobs_eval_ids.size());
        for (int id : ot.dfunc_dobs_eval_ids)
            dbuf.push_back(eval.evaluate(id));

        for (const auto &col : ot.columns) {
            const int j = col.species_j;
            // Term A: (∂func/∂x_j)·∏R.
            double A_j = 0.0;
            for (const auto &[k_idx, g_kj] : col.a_terms)
                A_j += g_kj * dbuf[k_idx];
            double val = A_j * P;
            // Term B: func·∂(∏R)/∂x_j (mass-action product rule).
            if (col.is_reactant) {
                double dP = static_cast<double>(col.mult_j);
                for (const auto &[s, m] : ot.reactants) {
                    double v = conc[s];
                    if (species[s].amount_valued)
                        v *= species[s].volume_factor;
                    const int e = (s == j) ? m - 1 : m;
                    for (int p = 0; p < e; ++p)
                        dP *= v;
                }
                // ∂v_j/∂conc_j = V_j for an amount-valued column (chain rule).
                if (species[j].amount_valued)
                    dP *= species[j].volume_factor;
                val += func * dP;
            }
            if (val == 0.0)
                continue;
            for (const auto &[csc, coeff] : col.affected) {
                const int row = static_cast<int>(sp.row_indices[csc]);
                emit(csc, j, row, coeff * val);
            }
        }
    }
}

} // namespace bngsim
