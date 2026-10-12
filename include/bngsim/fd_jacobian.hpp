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
// `k·(2X + h)`: dX*/ds 7% off at a ratio of 1e7 and 44% at 1e8.
//
// For such a species the row is evaluated three more times: at its own step
// (√eps of itself, the step it would have were it alone in the state), at
// twice that, and a thousandth beyond the own step. An entry becomes the
// quotient at the own step where all of this holds, and is the first
// quotient, bit for bit as it was, where any of it does not:
//
// - the row's rounding over the own step is under kFdSettled of the quotient:
//   it is known there;
// - the quotient at twice the step agrees with it to kFdSettled of itself, and
//   both steps moved the row: the row is as smooth as a difference needs at
//   that scale;
// - the thousandth beyond moves the row by a thousandth of what the step did,
//   to a tenth: the row resolves the step, and the quotient is not a count of
//   the row's own last bits. (`k·((A + X) − A)` with A at 1e8 moves in steps
//   of 1.5e-8 whatever X does, a rounding inside one rate law that neither
//   the row's terms nor a small move of the state shows. Its quotients at X's
//   own step and at twice it are the same wrong number, 1.14 for 1.)
// - it is further from the first quotient than twice what it is known to (its
//   rounding, the difference from the quotient at twice the step, and
//   kFdRefineRtol of itself for what a one-sided step leaves out). Otherwise
//   the first quotient may be the nearer of the two.
//
// Nothing is extrapolated. An earlier cut of this change took the entry from a
// ladder of halved steps below the floored one, extrapolated to a step of
// zero. That shows what the row does down to the ladder's last step and no
// further: a law with a scale of its own between that step and the species'
// own (a kink 1.7e-5 of the species away, a ripple 1e-4 of it wide) came back
// with the slope of everything else in the row, 11% off and with the wrong
// sign. The own step is where a difference quotient of this species is
// defined; what it does not see is a scale under that step, which no
// difference quotient of a species sees, at any size of the state.
//
// So an entry is not changed where the row's rounding is too large for the
// species' own step, which is what the floor is for: a species at 1e-9 in a
// row whose terms are of order 1 keeps the first quotient. Nor is the column
// of a species at exactly 0, which has no step of its own: a dimerization's
// entry there is still the secant.
//
// A row's rounding is eps of its value and of each entry times its species,
// at the least, and what moves of the whole state show beyond that. The sum
// of a row's terms is not to be had from its value and its entries:
// `L·(r − δ − φ(L))` with `φ → r − δ` is terms of 1e6 that cancel in the value
// and in every entry, and a quotient of it at a step of 1e-8 is all rounding
// (BIOMD0000000884). So the row is evaluated with every species moved by a
// few parts in 1e14, up and the same down, in two patterns: the mean of the
// two differs from the row at the state by its rounding and by nothing else
// to first order. (The entry of a small species is taken at its own step
// here: the first quotient, a secant, times the species is no term of the
// row, and beside 1e12 it is a million times one.)

// How many times its own step (√eps of itself) a species has to be stepped by
// before its column is looked at again. Below this the floored step's
// truncation error is under 64·√eps/2, 5e-7 of the entry, for a term of
// second order.
constexpr double kFdRefineRatio = 64.0;
// A row's rounding over a step is taken this many times over.
constexpr double kFdNoise = 32.0;
// What a one-sided quotient at the species' own step leaves out, of itself.
constexpr double kFdRefineRtol = 1e-6;
// A quotient is known where the row's rounding over its step is under this
// much of it, and two quotients agree where they are no further apart.
constexpr double kFdSettled = 1e-4;
// The relative move of the whole state a row's rounding is sampled at, in
// kFdNoisePairs patterns: species by species up and down in turn, and the
// other way round. (All one way scales a ratio of two sums by nothing.)
constexpr double kFdNoiseMove = 0x1p-44;
constexpr int kFdNoisePairs = 2;
// How far beyond the own step the row is asked whether it resolves it, as a
// part of the step, and how near what the quotient says its answer has to be.
constexpr double kFdResolveStep = 0x1p-10;
constexpr double kFdResolveRtol = 0.1;

// Column-major D (n_out×ns, D[j*n_out + i] = ∂g_i/∂y_j) of any `eval(y, g)`
// that fills n_out values from a state of ns species. One `eval` per column
// plus one for the base point; for a state with a species far below its scale
// four more, and three for the column of each such species (see above).
// `excluded` (0-based, ascending) names the species that must not set the
// probe scale.
template <class Eval>
inline void fd_state_partials(Eval &&eval, const double *y, int ns, int n_out,
                              const std::vector<int> &excluded, double *D) {
    const auto un = static_cast<std::size_t>(ns);
    const auto um = static_cast<std::size_t>(n_out);
    std::vector<double> g0(um), g1(um), g_fine(um), y_pert(un);
    std::vector<int> small; // the species far below the state's scale
    const double y_scale = state_probe_scale(y, ns, excluded);
    eval(y, g0.data());
    for (int j = 0; j < ns; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        std::memcpy(y_pert.data(), y, un * sizeof(double));
        const double h = fd_probe(y[j], state_fd_step(y[j], y_scale), &y_pert[uj]);
        eval(y_pert.data(), g1.data());
        double *col = D + uj * um;
        for (std::size_t i = 0; i < um; ++i) {
            col[i] = (g1[i] - g0[i]) / h;
        }
        // A species at 0 has no step of its own.
        if (y[j] != 0.0 && h > kFdRefineRatio * kFdEps * std::abs(y[j])) {
            small.push_back(j);
        }
    }
    if (small.empty()) {
        return;
    }
    // The quotients of each such species at its own step and at twice it, and
    // what each step moved the row by.
    const std::size_t n_small = small.size();
    std::vector<double> own(n_small * um), twice(n_small * um), step(n_small, 0.0);
    std::vector<char> moved(n_small * um, 0), resolved(n_small * um, 0);
    for (std::size_t s = 0; s < n_small; ++s) {
        const auto uj = static_cast<std::size_t>(small[s]);
        const double base = kFdEps * std::abs(y[uj]);
        std::memcpy(y_pert.data(), y, un * sizeof(double));
        y_pert[uj] = y[uj] + base;
        const double h1 = y_pert[uj] - y[uj];
        y_pert[uj] = y[uj] + 2.0 * base;
        const double h2 = y_pert[uj] - y[uj];
        if (!(h1 > 0.0) || !(h2 > h1)) {
            continue; // no step of its own that a double holds
        }
        step[s] = h1;
        eval(y_pert.data(), g1.data());
        for (std::size_t i = 0; i < um; ++i) {
            twice[s * um + i] = (g1[i] - g0[i]) / h2;
            moved[s * um + i] = g1[i] != g0[i] ? 1 : 0;
        }
        y_pert[uj] = y[uj] + base;
        eval(y_pert.data(), g1.data());
        for (std::size_t i = 0; i < um; ++i) {
            own[s * um + i] = (g1[i] - g0[i]) / h1;
            moved[s * um + i] = moved[s * um + i] != 0 && g1[i] != g0[i] ? 1 : 0;
        }
        // A thousandth beyond: what the row does there against what the
        // quotient says it should.
        y_pert[uj] = y[uj] + base * (1.0 + kFdResolveStep);
        const double beyond = y_pert[uj] - (y[uj] + base);
        if (beyond > 0.0) {
            eval(y_pert.data(), g_fine.data());
            for (std::size_t i = 0; i < um; ++i) {
                const double expect = own[s * um + i] * beyond;
                const double off = std::abs((g_fine[i] - g1[i]) - expect);
                resolved[s * um + i] = off <= kFdResolveRtol * std::abs(expect) ? 1 : 0;
            }
        }
    }
    // Each row's rounding: eps of its own value and of each entry times its
    // species at the least (a small species' entry at its own step), and what
    // moves of the whole state, up and the same down, show beyond that.
    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> rounding(um), g_up(um), g_down(um);
    for (std::size_t i = 0; i < um; ++i) {
        rounding[i] = std::abs(g0[i]);
    }
    {
        std::size_t s = 0;
        for (int j = 0; j < ns; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            const bool is_small = s < n_small && small[s] == j;
            const double *col = is_small && step[s] > 0.0 ? own.data() + s * um : D + uj * um;
            for (std::size_t i = 0; i < um; ++i) {
                const double term = col[i] * y[j];
                if (std::isfinite(term)) {
                    rounding[i] += std::abs(term);
                }
            }
            if (is_small) {
                ++s;
            }
        }
    }
    for (std::size_t i = 0; i < um; ++i) {
        rounding[i] *= eps;
    }
    for (int pair = 0; pair < kFdNoisePairs; ++pair) {
        for (int sign = 0; sign < 2; ++sign) {
            for (int c = 0; c < ns; ++c) {
                const bool up = ((c + pair) % 2 == 0) == (sign == 0);
                y_pert[static_cast<std::size_t>(c)] =
                    y[c] * (1.0 + (up ? kFdNoiseMove : -kFdNoiseMove));
            }
            eval(y_pert.data(), sign == 0 ? g_up.data() : g_down.data());
        }
        for (std::size_t i = 0; i < um; ++i) {
            const double beyond = std::abs(0.5 * (g_up[i] + g_down[i]) - g0[i]);
            if (beyond > rounding[i]) {
                rounding[i] = beyond;
            }
        }
    }
    // An entry is the quotient at the species' own step where that is known,
    // agrees with the one at twice the step, and is further from the first
    // quotient than it is known to.
    for (std::size_t s = 0; s < n_small; ++s) {
        if (!(step[s] > 0.0)) {
            continue;
        }
        double *col = D + static_cast<std::size_t>(small[s]) * um;
        for (std::size_t i = 0; i < um; ++i) {
            const double q = own[s * um + i];
            const double apart = std::abs(twice[s * um + i] - q);
            const double noise = kFdNoise * rounding[i] / step[s];
            const double size = std::abs(q);
            // (A quotient that is no number fails the last of these: nothing
            // is further from anything than an infinity.)
            if (moved[s * um + i] != 0 && resolved[s * um + i] != 0 && noise <= kFdSettled * size &&
                apart <= kFdSettled * size &&
                std::abs(q - col[i]) > 2.0 * (kFdRefineRtol * size + noise + apart)) {
                col[i] = q;
            }
        }
    }
}

// Dense COLUMN-MAJOR J (ns×ns, J[j*ns + i] = ∂f_i/∂y_j) at (t, y) by one-sided
// differences: J[:,j] = (f(t, y + h·e_j) − f(t, y)) / h, at the species' own
// step for one far below the state's scale (see `fd_state_partials`).
// `eval(t, y, ydot)` is the RHS; `excluded` (0-based, ascending) names the
// species that must not set the probe scale.
template <class Eval>
inline void fd_dense_state_jacobian(Eval &&eval, double t, const double *y, int ns,
                                    const std::vector<int> &excluded, double *J) {
    fd_state_partials([&](const double *yy, double *f) { eval(t, yy, f); }, y, ns, ns, excluded, J);
}

} // namespace bngsim
