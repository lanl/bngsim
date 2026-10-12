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
// For such a species a second quotient is taken at half the step. Where the
// two agree to what the row's rounding can do to them, which is every entry
// the species enters linearly, the entry is the first, bit for bit as it was.
// Where they do not, the first has a truncation error that can be seen, and
// the entry is asked of a ladder of halved steps, extrapolated to a step of
// zero (Richardson's scheme, by Neville's tableau): the big steps keep the
// noise down, and the extrapolation removes what they cost. For a mass-action
// law, a polynomial in the species, three or four steps are exact.
//
// An entry changes only where the ladder shows what it is. That is one of two
// things, and anything else leaves the first quotient, as it was.
//
// - Two successive estimates agree to kFdRefineRtol of themselves, the row's
//   rounding over the step is under kFdSettled of them, and every estimate
//   after them, all the way down the ladder, stays where they are. The ladder
//   goes down to the species' own step, or to where the row's rounding over
//   the step has grown past kFdSettled of the estimate and nothing finer can
//   be told from it. It does not stop where the estimates first agree: a law
//   with a scale of its own far below the first step, `V·X/(K + X)` with K
//   at 1e-6 beside steps of 1, is flat to every step above K, and two
//   estimates there agree on the slope of everything else in the row. The
//   size the agreement is asked to is the estimate's own. Asked to the size
//   of the first quotient, a secant that can be orders above the entry, an
//   estimate that was still falling by a factor a step was taken for one
//   that had settled.
// - The species is at 0, the ladder has run all of its kFdMaxHalvings steps
//   without the row's rounding overtaking it, and over them the row has moved
//   in proportion to the square of the step: the quotient at the last step is
//   under twice its share, `step/first step`, of the first. The slope at the
//   species is then under 2e-12 of the first quotient, which is what a law of
//   second or higher order in a species at 0 has, and the entry is the
//   extrapolated value. A slope that is not 0 is told from this by the first
//   rule, down to 1e-12 of the first quotient.
//
// So an entry that is under the row's rounding over the steps is not changed,
// whether its slope is 0 or not: the two cannot be told apart there, and the
// first quotient is at least what it was. Neither is one whose estimates do
// not come to rest: a law that is not smooth at the species, or a row whose
// rounding overtakes the ladder first. What cannot be seen at all is a term
// too small to move the first two quotients apart, or a scale of the law's
// own that lies below the row's rounding at every step.
//
// A row's rounding is measured and not worked out. The sum of a row's terms
// is not to be had from its value and its entries: `L·(r − δ − φ(L))` with
// `φ → r − δ` is terms of 1e6 that cancel in the value and in every entry, and
// a quotient of it at a step of 1e-8 is all rounding (BIOMD0000000884, where a
// bound from the entries called that rounding a curvature). So the row is
// evaluated at the state with every species moved by a few parts in 1e14,
// three times and not all the same way, and what the entries do not account
// for of the change is its rounding. To that is added, at each step, the
// rounding of what the step itself moved the row by. A step that leaves a
// row's value where it was, to the last bit, is under that row's rounding
// whatever was measured, and ends its ladder.

// How many times its own step (√eps of itself) a species has to be stepped by
// before its column is looked at again. Below this the floored step's
// truncation error is under 64·√eps/2, 5e-7 of the entry.
constexpr double kFdRefineRatio = 64.0;
// Two quotients differ where they are further apart than this many times the
// row's rounding over the step, and kFdRefineRtol of themselves. Two
// estimates agree where they are no further apart than that.
constexpr double kFdNoise = 32.0;
constexpr double kFdRefineRtol = 1e-6;
// An estimate is known where the row's rounding over the step, kFdNoise times
// over, is under this much of it: a changed entry is then within 1e-4 of
// itself of the slope, and within 3e-6 where the rounding is what was
// measured.
constexpr double kFdSettled = 1e-4;
// The relative move of the whole state a row's rounding is sampled at, three
// times: species by species up and down in turn, the other way round, and all
// up. (All one way alone scales a ratio of two sums by nothing at all.)
constexpr double kFdNoiseMove = 0x1p-44;
constexpr int kFdNoiseProbes = 3;
// The ladder: at most this many halvings, a factor of 1e-12 in the step, and
// not below the species' own step.
constexpr int kFdMaxHalvings = 40;

// Column-major D (n_out×ns, D[j*n_out + i] = ∂g_i/∂y_j) of any `eval(y, g)`
// that fills n_out values from a state of ns species. One `eval` per column
// plus one for the base point; for a state with a species far below its scale
// three more, and more for the column of each such species (see above).
// `excluded` (0-based, ascending) names the species that must not set the
// probe scale.
template <class Eval>
inline void fd_state_partials(Eval &&eval, const double *y, int ns, int n_out,
                              const std::vector<int> &excluded, double *D) {
    const auto un = static_cast<std::size_t>(ns);
    const auto um = static_cast<std::size_t>(n_out);
    std::vector<double> g0(um), g1(um), y_pert(un), h0(un);
    const double y_scale = state_probe_scale(y, ns, excluded);
    eval(y, g0.data());
    bool any_small = false;
    for (int j = 0; j < ns; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        std::memcpy(y_pert.data(), y, un * sizeof(double));
        h0[uj] = fd_probe(y[j], state_fd_step(y[j], y_scale), &y_pert[uj]);
        eval(y_pert.data(), g1.data());
        double *col = D + uj * um;
        for (std::size_t i = 0; i < um; ++i) {
            col[i] = (g1[i] - g0[i]) / h0[uj];
        }
        any_small = any_small || h0[uj] > kFdRefineRatio * kFdEps * std::abs(y[j]);
    }
    if (!any_small) {
        return;
    }
    // Each row's rounding: eps of its own value and of each entry times its
    // species at the least, and what three small moves of the whole state
    // change in it beyond what the entries say they should.
    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> rounding(um), along(um);
    for (std::size_t i = 0; i < um; ++i) {
        rounding[i] = std::abs(g0[i]);
    }
    for (int j = 0; j < ns; ++j) {
        const double *col = D + static_cast<std::size_t>(j) * um;
        for (std::size_t i = 0; i < um; ++i) {
            const double term = col[i] * y[j];
            if (std::isfinite(term)) {
                rounding[i] += std::abs(term);
            }
        }
    }
    for (std::size_t i = 0; i < um; ++i) {
        rounding[i] *= eps;
    }
    for (int probe = 0; probe < kFdNoiseProbes; ++probe) {
        std::fill(along.begin(), along.end(), 0.0);
        for (int j = 0; j < ns; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            const double move = probe == 2 || (j + probe) % 2 == 0 ? kFdNoiseMove : -kFdNoiseMove;
            y_pert[uj] = y[j] * (1.0 + move);
            const double moved = y_pert[uj] - y[j];
            const double *col = D + uj * um;
            for (std::size_t i = 0; i < um; ++i) {
                const double term = col[i] * moved;
                if (std::isfinite(term)) {
                    along[i] += term;
                }
            }
        }
        eval(y_pert.data(), g1.data());
        for (std::size_t i = 0; i < um; ++i) {
            const double beyond = std::abs(g1[i] - g0[i] - along[i]);
            if (beyond > rounding[i]) {
                rounding[i] = beyond;
            }
        }
    }
    const auto width = static_cast<std::size_t>(kFdMaxHalvings) + 1;
    std::vector<double> steps(width);
    std::vector<std::size_t> open; // rows whose entry is still being asked
    std::vector<double> table;     // per open row: the last row of its tableau
    std::vector<double> first;     // per open row: its first quotient
    std::vector<double> rest;      // per open row: the estimate the ladder came to rest at
    std::vector<double> rest_noise;
    std::vector<char> at_rest;
    std::vector<double> last; // per open row: the quotient at the last step taken
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
        int taken = 0; // steps of the ladder below the first
        for (int k = 1; k <= kFdMaxHalvings; ++k) {
            const auto uk = static_cast<std::size_t>(k);
            const double want = steps[uk - 1] / 2.0;
            if (k > 1 && want < kFdEps * yj) {
                break; // not below the species' own step
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
                        kFdNoise * rounding[i] / h + kFdRefineRtol * std::abs(q0)) {
                        open.push_back(i);
                    }
                    // Otherwise one quotient to rounding, and the entry stays.
                }
                if (open.empty()) {
                    break;
                }
                table.assign(open.size() * width, 0.0);
                first.assign(open.size(), 0.0);
                rest.assign(open.size(), 0.0);
                rest_noise.assign(open.size(), 0.0);
                at_rest.assign(open.size(), 0);
                last.assign(open.size(), 0.0);
                for (std::size_t r = 0; r < open.size(); ++r) {
                    table[r * width] = col[open[r]];
                    first[r] = col[open[r]];
                }
            }
            taken = k;
            bool any_open = false;
            for (std::size_t r = 0; r < open.size(); ++r) {
                if (open[r] == um) {
                    continue; // changed, or left as it was
                }
                const std::size_t i = open[r];
                const double moved = g1[i] - g0[i];
                if (moved == 0.0) {
                    // The step does not move the row at all: nothing finer is
                    // to be seen. An estimate that held this far has held.
                    if (at_rest[r] != 0) {
                        col[i] = rest[r];
                    }
                    open[r] = um;
                    continue;
                }
                double *prev = table.data() + r * width;
                cur[0] = moved / h;
                for (std::size_t m = 1; m <= uk; ++m) {
                    // The polynomial through the quotients at steps k-m..k, at 0.
                    const double far_step = steps[uk - m];
                    cur[m] = (cur[m - 1] * far_step - prev[m - 1] * h) / (far_step - h);
                }
                const double est = cur[uk];
                if (!std::isfinite(est)) {
                    open[r] = um;
                    continue;
                }
                // What rounding can do to a quotient at this step: the row's
                // own, and that of what the step moved it by.
                const double noise = kFdNoise * (rounding[i] + eps * std::abs(moved)) / h;
                if (at_rest[r] != 0) {
                    if (std::abs(est - rest[r]) >
                        kFdRefineRtol * std::abs(rest[r]) + noise + rest_noise[r]) {
                        at_rest[r] = 0; // it did not hold: the law has a scale below here
                    } else if (noise > kFdSettled * std::abs(rest[r])) {
                        col[i] = rest[r]; // held down to the row's rounding
                        open[r] = um;
                        continue;
                    }
                }
                if (at_rest[r] == 0) {
                    // (At the first halving the estimate before this one is
                    // the first quotient, which an opened row is not at.)
                    if (noise <= kFdSettled * std::abs(est) &&
                        std::abs(est - prev[uk - 1]) <= kFdRefineRtol * std::abs(est) + noise) {
                        at_rest[r] = 1;
                        rest[r] = est;
                        rest_noise[r] = noise;
                    } else if (noise > kFdSettled * std::max(std::abs(cur[0]), std::abs(est))) {
                        open[r] = um; // the row's rounding has overtaken the ladder
                        continue;
                    }
                }
                last[r] = cur[0];
                any_open = true;
                std::memcpy(prev, cur.data(), (uk + 1) * sizeof(double));
            }
            if (!any_open) {
                break;
            }
        }
        // The ladder is at its end. An estimate that held to it has held.
        // For a species at 0, one that never came to rest is the slope where
        // the row has moved as the square of the step all the way down.
        for (std::size_t r = 0; r < open.size(); ++r) {
            if (open[r] == um) {
                continue;
            }
            const std::size_t i = open[r];
            if (at_rest[r] != 0) {
                col[i] = rest[r];
                continue;
            }
            if (y[j] != 0.0) {
                continue;
            }
            const auto ut = static_cast<std::size_t>(taken);
            const double est = table[r * width + ut];
            const double share = 2.0 * std::abs(first[r]) * (steps[ut] / steps[0]);
            if (std::abs(last[r]) <= share && std::abs(est) <= std::abs(last[r])) {
                col[i] = est;
            }
        }
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
