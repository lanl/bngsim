// bngsim/include/bngsim/fd_jacobian.hpp -- the one-sided difference-quotient
// step rule for the state Jacobian, and the dense column sweep that applies it.
//
// One rule, one place (issue #523). The steady-state solver differenced its
// Jacobian here since issue #63 and improved the step three times on the way
// (#63, #76, #123); the public Model.jacobian() fallback has to hand back the
// SAME matrix the solver factors, which it can only do by running the same
// loop. So the loop is a template over the RHS callable: steady_state.cpp
// instantiates it on SteadyStateRhs (interpreted or compiled RHS), model.cpp on
// NetworkModel::compute_derivs. The parameter-probe steps stay in
// steady_state.cpp; only kFdEps and the probe helper are shared with them.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

namespace bngsim {

// √(machine eps): the standard one-sided difference-quotient fraction, where the
// O(h) truncation error and the O(eps/h) cancellation error meet.
constexpr double kFdEps = 1.4901161193847656e-8;

// One-sided step for probing SPECIES j of a state whose largest concentration
// is `y_scale`.
//
// Relative to the species, floored at the scale of the state it belongs to
// rather than at 1.0 — unlike parameters, which have no common unit, every
// species is a concentration in the same one, so the state HAS a typical
// magnitude and a species at (or near) zero can be probed against it. The
// absolute 1.0 was wrong in both directions: a nanomolar model was probed at
// 1 molar, and a model in molecule counts (1e6) was probed at 1e-14 of itself,
// which is cancellation noise rather than a derivative. Scored against the
// analytical Jacobian over 1,066 corpus model-states, this rule beats both the
// old one (511 better vs 54 worse, at 10x) and the floor-free relative step
// (291 vs 13) — the floor is what a species far below the state's scale needs
// to stay out of the cancellation noise.
//
// What the floor costs such a species, a step of many times itself, is taken
// back in `fd_state_partials` below (issue #1002).
inline double state_fd_step(double y, double y_scale) {
    return kFdEps * std::max(std::abs(y), y_scale);
}

// The perturbed value to write (`*x_plus`) for a probe of `x` by `h`, and the
// step the difference quotient must divide by — the REALIZED `(x + h) - x`,
// which differs from the requested h by a rounding. For a subnormal x no
// relative step survives the addition at all; dividing by that zero would fill
// the column with infinities, so fall back to the absolute step the old rule
// used (exact for a rate law linear in x, and nothing better exists at 1e-310).
inline double fd_probe(double x, double h, double *x_plus) {
    double xp = x + h;
    if (xp == x) {
        xp = x + kFdEps;
    }
    *x_plus = xp;
    return xp - x;
}

// The state's own magnitude, which `state_fd_step` floors its probe at.
//
// Over the species that HAVE a steady value only. A species the caller masked
// out (issue #74) is a write-only accumulator holding whatever integration left
// it at — a quantity that grows without bound, 7.5e8 on Barua 2013 while the
// other 405 species are settled at 1e-10 — and letting it set the scale would
// drag every other species' probe up with it. `excluded` is ascending, as both
// callers' sources guarantee. An all-zero state offers no scale at all, so it
// keeps the historical 1.0.
inline double state_probe_scale(const double *y, int ns, const std::vector<int> &excluded) {
    double scale = 0.0;
    std::size_t e = 0;
    for (int i = 0; i < ns; ++i) {
        if (e < excluded.size() && excluded[e] == i) {
            ++e;
            continue;
        }
        scale = std::max(scale, std::abs(y[i]));
    }
    return scale > 0.0 ? scale : 1.0;
}

// ─── A species far below the state's scale (issue #1002) ─────────────────────
//
// `state_fd_step` floors the step at the state's largest concentration, which
// keeps a small species out of the cancellation noise of the rows it is read
// in. It also steps that species by many times itself: beside a species at 1e8,
// one at 0.87 is stepped by 1.5, and a term of second order in it is read as a
// secant across twice the species. `k·X²` came back with a slope of
// `k·(2X + h)`: dX*/ds 7% off at a ratio of 1e7 and 44% at 1e8, and a species at
// exactly 0 that dimerizes was given a decay rate it does not have.
//
// The floored quotient is kept wherever a second quotient at half the step
// agrees with it to what rounding can do to the two, which is every entry the
// species enters linearly, and those are bit for bit what they were. Where the
// two differ by more than that, the first has a truncation error that can be
// seen, and the entry is extrapolated to a step of zero from a ladder of
// halved steps (Richardson's scheme, by Neville's tableau): the big steps keep
// the noise down, and the extrapolation removes what they cost. For a
// mass-action law, a polynomial in the species, three or four steps are exact.
// A law with a scale of its own far below the first step, a Hill term whose
// half-saturation is, is followed down the ladder until two estimates agree.
// What cannot be seen is a term too small to move the first two quotients
// apart: that entry is what it was.

// How many times its own step (√eps of itself) a species has to be stepped by
// before its column is looked at again. Below this the floored step's
// truncation error is under 64·√eps/2, 5e-7 of the entry.
constexpr double kFdRefineRatio = 64.0;
// Two estimates a row's rounding cannot tell apart are one. A quotient is good
// to 2·eps of the sum of the row's terms over its step, an extrapolated one to
// about eight times that, and two of them differ by twice as much; or they
// agree to kFdRefineRtol of themselves.
constexpr double kFdNoise = 32.0;
constexpr double kFdRefineRtol = 1e-6;
// The ladder: at most this many halvings, a factor of 1e-12 in the step, and
// not below the species' own step.
constexpr int kFdMaxHalvings = 40;

// Column-major D (n_out×ns, D[j*n_out + i] = ∂g_i/∂y_j) of any `eval(y, g)`
// that fills n_out values from a state of ns species. One `eval` per column
// plus one for the base point, and more for the column of a species far below
// the state's scale where a row is not linear in it (see above). `excluded`
// (0-based, ascending) names the species that must not set the probe scale.
template <class Eval>
inline void fd_state_partials(Eval &&eval, const double *y, int ns, int n_out,
                              const std::vector<int> &excluded, double *D) {
    const auto un = static_cast<std::size_t>(ns);
    const auto um = static_cast<std::size_t>(n_out);
    std::vector<double> g0(um), g1(um), y_pert(un), h0(un);
    const double y_scale = state_probe_scale(y, ns, excluded);
    eval(y, g0.data());
    for (int j = 0; j < ns; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        std::memcpy(y_pert.data(), y, un * sizeof(double));
        h0[uj] = fd_probe(y[j], state_fd_step(y[j], y_scale), &y_pert[uj]);
        eval(y_pert.data(), g1.data());
        double *col = D + uj * um;
        for (std::size_t i = 0; i < um; ++i) {
            col[i] = (g1[i] - g0[i]) / h0[uj];
        }
    }
    // What a row's values are assembled from: its own value and each entry
    // times its species. Rounding in a row is eps of this.
    std::vector<double> row_terms(um);
    bool any_small = false;
    for (std::size_t i = 0; i < um; ++i) {
        row_terms[i] = std::abs(g0[i]);
    }
    for (int j = 0; j < ns; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        const double yj = std::abs(y[j]);
        any_small = any_small || h0[uj] > kFdRefineRatio * kFdEps * yj;
        if (yj == 0.0) {
            continue;
        }
        const double *col = D + uj * um;
        for (std::size_t i = 0; i < um; ++i) {
            row_terms[i] += std::abs(col[i]) * yj;
        }
    }
    if (!any_small) {
        return;
    }
    const double eps = std::numeric_limits<double>::epsilon();
    const auto width = static_cast<std::size_t>(kFdMaxHalvings) + 1;
    std::vector<double> steps(width);
    std::vector<std::size_t> open; // rows whose entry is still being asked
    std::vector<double> table;     // per open row: the last row of its tableau
    std::vector<double> cur(width);
    for (int j = 0; j < ns; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        const double yj = std::abs(y[j]);
        if (!(h0[uj] > kFdRefineRatio * kFdEps * yj)) {
            continue;
        }
        double *col = D + uj * um;
        steps[0] = h0[uj];
        open.clear();
        std::memcpy(y_pert.data(), y, un * sizeof(double));
        for (int k = 1; k <= kFdMaxHalvings; ++k) {
            const auto uk = static_cast<std::size_t>(k);
            const double want = steps[uk - 1] / 2.0;
            // Not below the species' own step: there the quotient is the
            // plain relative one, which is where an entry that has not
            // settled is left.
            if (k > 1 && want < kFdEps * yj) {
                break;
            }
            y_pert[uj] = y[j] + want;
            const double h = y_pert[uj] - y[j];
            if (!(h > 0.0)) {
                break;
            }
            steps[uk] = h;
            eval(y_pert.data(), g1.data());
            if (k == 1) {
                for (std::size_t i = 0; i < um; ++i) {
                    const double q0 = col[i];
                    const double q1 = (g1[i] - g0[i]) / h;
                    if (std::abs(q1 - q0) >
                        kFdNoise * eps * row_terms[i] / h + kFdRefineRtol * std::abs(q0)) {
                        open.push_back(i);
                    }
                    // Otherwise one quotient to rounding, and the entry stays.
                }
                if (open.empty()) {
                    break;
                }
                table.assign(open.size() * width, 0.0);
                for (std::size_t r = 0; r < open.size(); ++r) {
                    table[r * width] = col[open[r]];
                }
            }
            bool any_open = false;
            for (std::size_t r = 0; r < open.size(); ++r) {
                if (open[r] == um) {
                    continue; // settled
                }
                const std::size_t i = open[r];
                double *prev = table.data() + r * width;
                cur[0] = (g1[i] - g0[i]) / h;
                for (std::size_t m = 1; m <= uk; ++m) {
                    // The polynomial through the quotients at steps k-m..k, at 0.
                    const double far_step = steps[uk - m];
                    cur[m] = (cur[m - 1] * far_step - prev[m - 1] * h) / (far_step - h);
                }
                // Settled where one more step leaves the estimate where it
                // was. Until then the entry is the plain quotient at the
                // smallest step taken, which is what it is left as where the
                // ladder ends first: a law that is not smooth at that scale.
                const bool settled = k > 1 && std::abs(cur[uk] - prev[uk - 1]) <=
                                                  kFdNoise * eps * row_terms[i] / h +
                                                      kFdRefineRtol * std::abs(cur[uk]);
                const double entry = settled ? cur[uk] : cur[0];
                if (std::isfinite(entry)) {
                    col[i] = entry;
                }
                if (settled) {
                    open[r] = um;
                } else {
                    any_open = true;
                    std::memcpy(prev, cur.data(), (uk + 1) * sizeof(double));
                }
            }
            if (!any_open) {
                break;
            }
        }
        y_pert[uj] = y[j];
    }
}

// Dense COLUMN-MAJOR J (ns×ns, J[j*ns + i] = ∂f_i/∂y_j) at (t, y) by one-sided
// differences: J[:,j] = (f(t, y + h·e_j) − f(t, y)) / h, extrapolated for a
// species far below the state's scale (see `fd_state_partials`).
// `eval(t, y, ydot)` is the RHS; `excluded` (0-based, ascending) names the
// species that must not set the probe scale.
template <class Eval>
inline void fd_dense_state_jacobian(Eval &&eval, double t, const double *y, int ns,
                                    const std::vector<int> &excluded, double *J) {
    fd_state_partials([&](const double *yy, double *f) { eval(t, yy, f); }, y, ns, ns, excluded, J);
}

} // namespace bngsim
