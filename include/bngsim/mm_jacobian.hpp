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
//   sFree = ½·(delta + D)
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
// **No clamp on sFree (GH #93).** sFree < 0 happens exactly when S < 0 — the
// substrate-exhaustion endgame, where the integrator oversteps zero and probes a
// negative concentration. The old code floored sFree to 0 in the RHS while the
// derivatives below guarded on `sFree > 0` and returned 0, so the Jacobian
// asserted "flat" over a region where the emitted RHS beside it in the same
// artifact still varied. Both are gone, because the clamp was never needed:
//
// * D is real for *every* S, positive or not — delta² + 4·Km·S factors as
//   (S + Km − E)² + 4·Km·E, which is a sum of squares for Km, E ≥ 0. So the
//   branch below is a genuine smooth continuation, not a square root falling off
//   a cliff.
// * The continuation is the *restoring* one: for S → −∞ the rate tends to
//   kcat·S (slope kcat, independent of E), i.e. dS/dt = −rate pushes S back up
//   toward 0. The clamp instead left a flat dead zone with no restoring force,
//   and put a kink at S = 0 for the error controller to trip over.
// * At S = 0 the unclamped rate is differentiable — not merely one-sided:
//   both one-sided derivatives equal kcat·E/(Km + E), which is what the ∂rate/∂S
//   quotient below returns. The old `sFree > 0` guard reported 0 there, and
//   every species with a zero initial condition sits at exactly that state.
//
// The one real degeneracy is Km + sFree ≤ 0, which is the rate's own denominator.
// Km + sFree vanishes only when Km·E = 0 (substitute x = −Km into the quadratic:
// it leaves −Km·E), so for Km, E > 0 it is positive for every S, decaying like
// Km·E/|S| as S → −∞. It covers E = 0 with S ≤ −Km, and Km = 0 with S < E —
// the latter used to evaluate 0/0 and hand CVODE a NaN. Guarding the denominator
// rather than sFree also keeps the correct kcat·E at Km = 0, S > E, which a
// guard on Km or E would have zeroed.
//
// Be clear about what that guard is: it is a *jump*, not a smooth patch. Coming
// at E = 0 from above with S < −Km the rate tends to kcat·(S + Km), not to 0. We
// take 0 anyway, because at E = 0 exactly the alternative is 0/0, and because a
// reaction with no enzyme has rate 0 by definition. The trade is worth making
// only because of where it sits: the old clamp put its kink at S = 0, which
// every substrate crosses on its way to exhaustion, whereas this one needs a
// species pinned at exactly 0 *and* a second one driven past −Km.

#pragma once

#include <cmath>

namespace bngsim {

struct MMTqssa {
    double delta; // S - Km - E
    double D;     // sqrt(delta² + 4·Km·S)
    double sFree; // positive root of x² - delta·x - Km·S = 0 (negative iff S < 0)
    double KpsF;  // Km + sFree — the rate's denominator, and the only guard
};

// Free substrate as the *stable* quadratic root — see note 1 in the header
// comment. Returned unclamped (GH #93); callers guard on `KpsF > 0` instead,
// which is the same guard the derivatives below and every emitter in
// python/bngsim/_codegen.py use.
inline MMTqssa mm_tqssa(double Km, double E, double S) {
    MMTqssa q;
    q.delta = S - Km - E;
    q.D = std::sqrt(q.delta * q.delta + 4.0 * Km * S);
    if (q.delta >= 0.0) {
        q.sFree = 0.5 * (q.delta + q.D);
    } else {
        // D - delta > 0 whenever delta < 0 and D ≥ 0, so this cannot divide by
        // zero on real inputs; the guard covers a NaN degeneracy.
        double denom = q.D - q.delta;
        q.sFree = denom > 0.0 ? 2.0 * Km * S / denom : 0.0;
    }
    q.KpsF = Km + q.sFree;
    return q;
}

inline void mm_tqssa_derivatives(double kcat, double Km, double E, double S, double stat,
                                 double &dE, double &dS) {
    MMTqssa q = mm_tqssa(Km, E, S);
    if (q.KpsF <= 0.0 || q.D <= 0.0) {
        // Degenerate exactly where compute_rxn_rate returns 0 (see the header
        // note); the rate is identically 0 there, so both partials are 0. D == 0
        // implies KpsF == 0, so the second test is belt-and-braces against a
        // float64 edge rather than a separate case.
        dE = 0.0;
        dS = 0.0;
        return;
    }
    double C = kcat * stat;
    dE = C * q.sFree / q.D;
    dS = C * E * Km / (q.KpsF * q.D);
}

} // namespace bngsim
