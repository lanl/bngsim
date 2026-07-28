// bngsim/include/bngsim/mm_jacobian.hpp — Michaelis–Menten (tQSSA) closed form
// (GH #76 task 3, made numerically stable in GH #89)
//
// Single source of truth for the tQSSA free substrate and for ∂rate/∂E and
// ∂rate/∂S: the free substrate is shared with compute_rxn_rate's MM branch, the
// derivatives with the dense path (NetworkModel::fill_dense_analytical_jacobian)
// and the sparse CVODE callback (cvode_analytical_jac).
//
//   delta = S - Km - E
//   D     = sqrt(delta² + 4·Km·S)
//   sFree = ½·(delta + D)          (clamped to 0 in the RHS)
//   rate  = kcat·stat·sFree·E/(Km + sFree)
//
// **Stability (GH #89).** Two cancellations used to live in these lines, both
// fatal once delta < 0 with |delta| ≫ √(4·Km·S) — the deep enzyme excess regime.
//
// 1. ½·(delta + D) subtracts two nearly-equal positive numbers, losing about two
//    significant digits per decade of that ratio (no correct digit left at 1e8).
//    sFree is the positive root of x² − delta·x − Km·S = 0, so for delta < 0 the
//    conjugate form 2·Km·S/(D − delta) multiplies out to the same value with no
//    subtraction at all. For delta ≥ 0 the textbook form is already
//    cancellation-free and is kept bit-for-bit.
//
// 2. The chain rule through sFree used to be written ∂sFree/∂E = ½·(−1 − delta/D)
//    and ∂sFree/∂S = ½·(1 + (delta + 2·Km)/D), which cancel in exactly the same
//    regime — so fixing sFree alone still left ∂rate/∂E with a relative error of
//    1e+10 on a deep-saturation sweep. Differentiating the *symmetric* form of the
//    rate collapses each partial to a single subtraction-free quotient (the tQSSA
//    complex is c = ½·(A − D) with A = E + S + Km, and that same D is √(A² − 4·E·S),
//    so ∂c/∂E = (S − c)/D = sFree/D and ∂c/∂Km = −c/D):
//
//      ∂rate/∂E  = kcat·stat·sFree/D
//      ∂rate/∂S  = kcat·stat·E·Km/((Km + sFree)·D)
//      ∂rate/∂Km = −rate/D          (emitted by python/bngsim/_codegen.py)
//
//    Verified identical to sympy.diff of the shipped rate, and measured against
//    mpmath at 60 digits: machine precision in every column of all four sweeps.
//
// Where the RHS clamps sFree to 0 (delta + D ≤ 0), the rate is identically 0 in a
// neighbourhood, so both derivatives are 0 — matching the clamped flat region.

#pragma once

#include <cmath>

namespace bngsim {

struct MMTqssa {
    double delta; // S - Km - E
    double D;     // sqrt(delta² + 4·Km·S)
    double sFree; // positive root of x² - delta·x - Km·S = 0 (unclamped)
};

// Free substrate as the *stable* quadratic root — see note 1 in the header
// comment. Callers that feed a rate still apply the RHS's own `sFree < 0` clamp;
// this returns the root as computed so the derivative guard can see it.
inline MMTqssa mm_tqssa(double Km, double E, double S) {
    MMTqssa q;
    q.delta = S - Km - E;
    q.D = std::sqrt(q.delta * q.delta + 4.0 * Km * S);
    if (q.delta >= 0.0) {
        q.sFree = 0.5 * (q.delta + q.D);
    } else {
        // D - delta > 0 whenever delta < 0 and D ≥ 0, so this cannot divide by
        // zero on real inputs; the guard covers a NaN/negative-S degeneracy.
        double denom = q.D - q.delta;
        q.sFree = denom > 0.0 ? 2.0 * Km * S / denom : 0.0;
    }
    return q;
}

inline void mm_tqssa_derivatives(double kcat, double Km, double E, double S, double stat,
                                 double &dE, double &dS) {
    MMTqssa q = mm_tqssa(Km, E, S);
    if (q.sFree <= 0.0 || q.D <= 0.0) {
        // Clamped (rate ≡ 0 locally) or degenerate — flat, zero derivative.
        dE = 0.0;
        dS = 0.0;
        return;
    }
    double C = kcat * stat;
    dE = C * q.sFree / q.D;
    dS = C * E * Km / ((Km + q.sFree) * q.D);
}

} // namespace bngsim
