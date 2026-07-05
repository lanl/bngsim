// bngsim/include/bngsim/mm_jacobian.hpp — Michaelis–Menten (tQSSA) closed-form
// Jacobian derivative (GH #76 task 3)
//
// Single source of truth for ∂rate/∂E and ∂rate/∂S of the tQSSA MM rate law,
// shared by the dense path (NetworkModel::fill_dense_analytical_jacobian) and the
// sparse CVODE callback (cvode_analytical_jac). Derived by hand to match
// compute_rxn_rate's MM formula exactly:
//
//   delta = S - Km - E
//   D     = sqrt(delta² + 4·Km·S)
//   sFree = ½·(delta + D)          (clamped to 0 in the RHS)
//   rate  = kcat·stat·sFree·E/(Km + sFree)
//
// Chain rule through sFree (∂delta/∂E = -1, ∂delta/∂S = +1):
//   ∂sFree/∂E = ½·(-1 - delta/D)
//   ∂sFree/∂S = ½·(+1 + (delta + 2·Km)/D)
//   ∂rate/∂E  = kcat·stat·[ sFree/(Km+sFree) + E·Km/(Km+sFree)²·∂sFree/∂E ]
//   ∂rate/∂S  = kcat·stat·E·Km/(Km+sFree)²·∂sFree/∂S
//
// Where the RHS clamps sFree to 0 (delta + D ≤ 0), the rate is identically 0 in a
// neighbourhood, so both derivatives are 0 — matching the clamped flat region.

#pragma once

#include <cmath>

namespace bngsim {

inline void mm_tqssa_derivatives(double kcat, double Km, double E, double S, double stat,
                                 double &dE, double &dS) {
    double delta = S - Km - E;
    double D = std::sqrt(delta * delta + 4.0 * Km * S);
    double sFree = 0.5 * (delta + D);
    if (sFree <= 0.0 || D <= 0.0) {
        // Clamped (rate ≡ 0 locally) or degenerate — flat, zero derivative.
        dE = 0.0;
        dS = 0.0;
        return;
    }
    double dsF_dE = 0.5 * (-1.0 - delta / D);
    double dsF_dS = 0.5 * (1.0 + (delta + 2.0 * Km) / D);
    double C = kcat * stat;
    double KpsF = Km + sFree;
    double common = C * E * Km / (KpsF * KpsF); // C·E·Km/(Km+sFree)²
    dE = C * sFree / KpsF + common * dsF_dE;
    dS = common * dsF_dS;
}

} // namespace bngsim
