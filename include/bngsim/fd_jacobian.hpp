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
// What the floor costs a species far below the state's scale, a step of many
// times itself, is checked by `fd_own_step_columns` and `fd_half_step_columns`
// below (issue #1002).
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
// secant across twice the species. `k·X²` comes back with a slope of
// `k·(2X + h)`: dX*/ds 7% off at a ratio of 1e7 and 44% at 1e8.
//
// The species' own step (√eps of itself) has the opposite failure. A term that
// reads the species through a sum with something much larger, `C·((K + X) − K)`
// or a total minus what is bound, answers in steps of that sum's last bit; where
// that grain is wider than the own step and narrower than the floored one, the
// own step does not see the term at all and the floored step sees all of it.
// No sampling of a row tells the two apart in every case (the plan on #1002
// records four rules that were tried), so neither matrix is returned in place of
// the other. `fd_own_step_columns` gives the second matrix, and a caller works
// its answer out with both: where they agree the floored one is returned as it
// was, and where they do not the answer is refused. Curvature moves the floored
// step's answer and grain moves the own step's, and the two steps are at least
// kFdRefineRatio apart, so a result both give is neither failure's.
//
// The second matrix is the first except in the column of a species stepped by
// more than kFdRefineRatio times its own step, and there only where all of
// this holds of an entry, at the own step:
//
// - the own step moved the row, and the row's rounding over it is under
//   kFdSettled of the quotient: it is known there;
// - the quotient at kFdRefineRatio times the step agrees with it to
//   kFdSettled of itself: the slope is the row's over two decades of step;
// - a thousandth beyond the own step moves the row by a thousandth of what the
//   step did, to a tenth: the row resolves the step, and the quotient is not a
//   count of the row's own last bits;
// - it is further from the first quotient than twice what it is known to.
//
// A column is taken where any entry of it is. The second matrix is never
// returned, only compared with: a column with one entry at the own step and
// the same term's other entry at the floored one is not a column of secants
// at one step (rows that add up to nothing in the right-hand side need not in
// it, and MODEL1911130007's solve with such a matrix was 43% off), so a solve
// with it can disagree where the floored one is right, which refuses. It
// cannot agree the floored answer into being returned where that is wrong,
// which is what taking a column whole or not at all did: a row the own step
// sees nothing of, `C·((K + X) − K)` with the grain of K + X wider than
// kFdRefineRatio own steps, vetoed the column, and the curvature in another
// row of it, `k·X²`, came back 44% off with no error, on the issue's own
// model with that one reaction added; and a function the compiled evaluator
// answered, `1e8 + X`, vetoed the column for one it declined beside it,
// `abs(X)·X`, 85% off with no error.
//
// A row's rounding is taken as eps of its value and of each entry times its
// species, a small species' entry at its own step: the floored quotient, a
// secant, times the species is no term of the row.

// How many times its own step a species has to be stepped by before its column
// is taken again, and the step the second quotient is taken at.
constexpr double kFdRefineRatio = 64.0;
// A row's rounding over a step is taken this many times over.
constexpr double kFdNoise = 32.0;
// What a one-sided quotient at the species' own step leaves out, of itself.
constexpr double kFdRefineRtol = 1e-6;
// A quotient is known where the row's rounding over its step is under this
// much of it, and two quotients agree where they are no further apart.
constexpr double kFdSettled = 1e-4;
// How far beyond the own step the row is asked whether it resolves it, as a
// part of the step, and how near what the quotient says its answer has to be.
constexpr double kFdResolveStep = 0x1p-10;
constexpr double kFdResolveRtol = 0.1;

// The second matrix for `D` (column-major n_out×ns, D[j*n_out + i] =
// ∂g_i/∂y_j, as the floored sweep took it from `eval(y, g)` with g(y) = `g0`):
// `D_own` is D with the columns of species far below the state's scale taken
// at their own step where the rules above allow. Returns the species whose
// columns differ (0-based, ascending); empty where none does, and then D_own
// is D to the bit. Three more `eval` per such species.
template <class Eval>
inline std::vector<int> fd_own_step_columns(Eval &&eval, const double *y, int ns, int n_out,
                                            const std::vector<int> &excluded, const double *g0,
                                            const double *D, double *D_own) {
    const auto un = static_cast<std::size_t>(ns);
    const auto um = static_cast<std::size_t>(n_out);
    std::memcpy(D_own, D, un * um * sizeof(double));
    std::vector<int> changed;
    std::vector<int> small; // the species far below the state's scale
    const double y_scale = state_probe_scale(y, ns, excluded);
    for (int j = 0; j < ns; ++j) {
        double unused = 0.0;
        const double h = fd_probe(y[j], state_fd_step(y[j], y_scale), &unused);
        // A species at 0 has no step of its own.
        if (y[j] != 0.0 && h > kFdRefineRatio * kFdEps * std::abs(y[j])) {
            small.push_back(j);
        }
    }
    if (small.empty()) {
        return changed;
    }
    std::vector<double> g1(um), g_fine(um), y_pert(un);
    // The quotients of each such species at its own step and at
    // kFdRefineRatio times it, whether the own step moved the row, and
    // whether the row resolves it.
    const std::size_t n_small = small.size();
    std::vector<double> own(n_small * um), wide(n_small * um);
    std::vector<double> step(n_small, 0.0);
    std::vector<char> moved(n_small * um, 0), resolved(n_small * um, 0);
    for (std::size_t s = 0; s < n_small; ++s) {
        const auto uj = static_cast<std::size_t>(small[s]);
        const double base = kFdEps * std::abs(y[uj]);
        std::memcpy(y_pert.data(), y, un * sizeof(double));
        y_pert[uj] = y[uj] + base;
        const double h1 = y_pert[uj] - y[uj];
        y_pert[uj] = y[uj] + kFdRefineRatio * base;
        const double h_wide = y_pert[uj] - y[uj];
        if (!(h1 > 0.0) || !(h_wide > h1)) {
            continue; // no step of its own that a double holds
        }
        step[s] = h1;
        eval(y_pert.data(), g1.data());
        for (std::size_t i = 0; i < um; ++i) {
            wide[s * um + i] = (g1[i] - g0[i]) / h_wide;
        }
        y_pert[uj] = y[uj] + base;
        eval(y_pert.data(), g1.data());
        for (std::size_t i = 0; i < um; ++i) {
            own[s * um + i] = (g1[i] - g0[i]) / h1;
            moved[s * um + i] = g1[i] != g0[i] ? 1 : 0;
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
    // species (a small species' entry at its own step).
    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> rounding(um);
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
    std::vector<char> take(um);
    for (std::size_t s = 0; s < n_small; ++s) {
        if (!(step[s] > 0.0)) {
            continue;
        }
        const double *col = D + static_cast<std::size_t>(small[s]) * um;
        bool any = false;
        for (std::size_t i = 0; i < um; ++i) {
            const double q = own[s * um + i];
            take[i] = 0;
            const double apart = std::abs(wide[s * um + i] - q);
            const double noise = kFdNoise * rounding[i] / step[s];
            const double size = std::abs(q);
            const double gap = std::abs(q - col[i]);
            // (A quotient that is no number is not taken: nothing is that near
            // an infinity.)
            if (moved[s * um + i] != 0 && resolved[s * um + i] != 0 && noise <= kFdSettled * size &&
                apart <= kFdSettled * size && gap > 2.0 * (kFdRefineRtol * size + noise + apart)) {
                take[i] = 1;
                any = true;
            }
        }
        if (!any) {
            continue;
        }
        double *out = D_own + static_cast<std::size_t>(small[s]) * um;
        for (std::size_t i = 0; i < um; ++i) {
            if (take[i] != 0) {
                out[i] = own[s * um + i];
            }
        }
        changed.push_back(small[s]);
    }
    return changed;
}

// ─── Curvature in a row the own step does not read (issue #1002) ─────────────
//
// fd_own_step_columns takes an entry again only where the own step reads its
// row, and a row that a much larger species is in is not read there: its
// rounding over the own step is more than the entry. A term of second order in
// the small species in such a row, `k·X²` in the rate of a species at 1e8, is
// a secant across the floored step that nothing compares with anything:
// dG*/ds came back 66.05 for 58.96 with jacobian="fd", and no error.
//
// So the column of each species far below the state's scale is also taken
// again at half the floored step, over which a row's rounding comes to twice
// the quotient it comes to over the floored step, and no more. A secant moves
// with its step (k·(2X + h) for k·X²), and a slope does not: an answer that
// both steps give is not the curvature's. A column is taken whole, every entry
// at the one step, so that rows that add up to nothing in the right-hand side
// add up to nothing in it. A species at 0 is taken again too: the floored step
// is all the step it has, and k·X² there reads k·h for 0.
//
// `D_half` is `D` with those columns at half the floored step. Returns the
// species whose columns were taken (0-based, ascending). One more `eval` per
// such species. A species `excluded` names is taken too: the solve reads the
// column of one that is a conservation law's dependent, or a fixed species a
// parameter sets.
template <class Eval>
inline std::vector<int> fd_half_step_columns(Eval &&eval, const double *y, int ns, int n_out,
                                             const std::vector<int> &excluded, const double *g0,
                                             const double *D, double *D_half) {
    const auto un = static_cast<std::size_t>(ns);
    const auto um = static_cast<std::size_t>(n_out);
    std::memcpy(D_half, D, un * um * sizeof(double));
    std::vector<int> taken;
    const double y_scale = state_probe_scale(y, ns, excluded);
    std::vector<double> g1(um), y_pert(y, y + un);
    for (int j = 0; j < ns; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        const double h_req = state_fd_step(y[j], y_scale);
        double unused = 0.0;
        const double h = fd_probe(y[j], h_req, &unused);
        if (y[j] != 0.0 && !(h > kFdRefineRatio * kFdEps * std::abs(y[j]))) {
            continue; // stepped by about its own step: no curvature to speak of
        }
        const double h_half = fd_probe(y[j], 0.5 * h_req, &y_pert[uj]);
        if (!(h_half > 0.0) || !(h_half < h)) {
            y_pert[uj] = y[j];
            continue; // no smaller step that a double holds
        }
        eval(y_pert.data(), g1.data());
        y_pert[uj] = y[j];
        double *col = D_half + uj * um;
        for (std::size_t i = 0; i < um; ++i) {
            col[i] = (g1[i] - g0[i]) / h_half;
        }
        taken.push_back(j);
    }
    return taken;
}

// Dense COLUMN-MAJOR J (ns×ns, J[j*ns + i] = ∂f_i/∂y_j) at (t, y) by one-sided
// differences: J[:,j] = (f(t, y + h·e_j) − f(t, y)) / h, one `eval` per column
// plus one for the base point. `eval(t, y, ydot)` is the RHS; `excluded` (0-based,
// ascending) names the species that must not set the probe scale.
template <class Eval>
inline void fd_dense_state_jacobian(Eval &&eval, double t, const double *y, int ns,
                                    const std::vector<int> &excluded, double *J) {
    std::vector<double> f0(static_cast<std::size_t>(ns)), f1(static_cast<std::size_t>(ns)),
        y_pert(static_cast<std::size_t>(ns));
    const double y_scale = state_probe_scale(y, ns, excluded);
    eval(t, y, f0.data());
    for (int j = 0; j < ns; ++j) {
        std::memcpy(y_pert.data(), y, static_cast<std::size_t>(ns) * sizeof(double));
        const double h = fd_probe(y[j], state_fd_step(y[j], y_scale), &y_pert[j]);
        eval(t, y_pert.data(), f1.data());
        double *col = J + static_cast<std::size_t>(j) * ns;
        for (int i = 0; i < ns; ++i) {
            col[i] = (f1[i] - f0[i]) / h;
        }
    }
}

// ─── A quotient that steps across a jump (issue #1002) ───────────────────────
//
// A rate law with a jump in it (`if(X > 5, a, b)`, a comparison, `ceil`,
// `rint`) differenced from a state within one step of the threshold reads the
// jump over the step, Δf/h, where the row has a slope. Beside X = 5,
// `if(X > 5, 0.04, 0.02)·X` came back with −1.3e6 for −0.02. CVODE's Newton
// iteration under that matrix takes corrections a millionth of their size,
// calls them converged at once, and the run follows its predictor (X = 5 − 0.1t,
// 166% off at t = 60, with no error). A species held at 0 under `if(I > 0, …)`
// is stepped by the solver's floor, far under 1e-12; the entry is larger still,
// and the LU's rounding through it leaks 1e-36 into I, which turns the
// condition true (490% off).
//
// What tells a jump from a slope is the row over the steps beside the
// quotient's own. A slope, a kink, an extremum or an inflection moves the row
// over one step by no more than a small factor of what it moves over the steps
// either side; a jump moves it by the jump, however short the step. So the
// quotient's difference d = f(y + h) − f(y) reads a jump where |d| is more than
// kFdJumpRatio times both f(y + 2h) − f(y + h) and f(y − h) − f(y − 2h). The
// outer steps, not f(y) − f(y − h): on `X == 0` at X = 0 both inner steps cross.
//
// And more than kFdJumpNoise last bits of the row's terms (`fd_jump_floor`):
// a row whose terms cancel answers a short step in counts of their last bit,
// which is a jump at that grain and none in the model. A branch of a rate law
// changes a term by a part of itself. The terms are the row's value and each
// entry times its species, from the matrix the solver formed: BIOMD0000000091,
// mass action alone, had two entries taken for jumps against a floor of the
// row's value.
//
// `fd_step_may_cross_jump` is the half of the test the forward step gives; only
// where it holds are the two backward points worth evaluating.
//
// A column is tested only where `fd_jump_testable` holds, and of the mends it
// takes, only some are kept:
//
// - Where nothing moves the species (its row is zero), every mend is kept: its
//   column reaches Newton only through rounding, which a jump-sized entry
//   turns into a flip of the law.
// - Where it moves away, at more than kFdJumpParked of the terms of its own row
//   in the mended matrix (`fd_jump_unparked`), its mends are kept if its own
//   row is one of them and the run came from across the jump
//   (`fd_jump_came_across`). A large entry on the diagonal is what freezes a
//   species in Newton's iteration, so the run follows its predictor; one in
//   another species' row, of a species that moves, turns its corrections into
//   that row's, which fails the iteration and forms the matrix again. And it is
//   what lets CVODE hold a species that parks just short of a jump in another
//   species' law: the #176 fixture parks V 1e-11 under the step of `v_rec`,
//   the closed form and a mended matrix both collapse the step there, and the
//   straddled quotient integrates it to 2.5e-7 of the exact branch. In #903's
//   `split`, V + V2 parks there while V leaves at 0.38 and V2 arrives at 0.38:
//   V's step crosses, in those rows, and mended there the run crossed (0.633
//   off at rtol 1e-7, where main is right). A run that slides along a moving
//   threshold is wrong as on main (in the PR's "Left alone").
//   The mended matrix, because a species' own straddled entry swells its
//   row's terms: X at 0.1 beside `if(X > 5, ...)·X` had terms of 6.7e6.
// - Parked, a species' rate is its solver's noise and points either way: with
//   V's asymptote 9e-11 under the step and tolerances of 1e-8, the march read V
//   moving away at t = 27.7 and, mended, never converged (residual 18 where
//   main reaches 5e-10).
// - The species is more than two steps from 0, or exactly at 0. A step wider
//   than the species (CVODE's floor: 2.5e-2 beside a species at 1.6e-8, 7.8e13
//   beside one at 1.7e-4 late in a march, 3.2 beside one at -3.7e-7) reads the
//   model on the other side of 0, where a saturating law is nothing like
//   itself; six corpus models had a saturation or a pole taken for a jump that
//   way. A species at exactly 0 stays there unless something makes it, and
//   then it moves toward its step; its column reaches Newton only through
//   rounding, which a jump-sized entry turns into a flip of the law.
//
// A row that cancels inside one rate law has terms its matrix does not show,
// and can still have a last bit taken for a jump: BIOMD0000000520 had an entry
// of 1e-9 mended, against an iteration that cannot feel it (the run was bit
// for bit). Weighing an entry by the solver's step does not tell those apart:
// a run's first step is 1.9e-10 beside `if(X > 5, ...)` at X = 5, and the
// matrix formed there is used at steps 1e10 times longer.
constexpr double kFdJumpRatio = 16.0;
constexpr double kFdJumpNoise = 1024.0;
constexpr double kFdJumpParked = 1e-6;

// Is the column of species j, at `y` and stepped by `inc`, tested for a jump?
inline bool fd_jump_testable(double y, double inc) {
    return y == 0.0 || std::abs(y) > 2.0 * std::abs(inc);
}

// Are the mends of a tested column kept, its species' own rate `f_self` out of
// its row's terms `self_terms` in the mended matrix?
inline bool fd_jump_unparked(double f_self, double self_terms) {
    return self_terms == 0.0 || std::abs(f_self) > kFdJumpParked * self_terms;
}

// Did the run come from across the jump a mended row reads? `back` is the row
// at the state one step of the column's species back along the flow, y −
// τ·f(y) with τ = inc/|f_j|, `predicted` what the row would read there on the
// state's own branch (f(y) − τ·J·f(y), J mended), `d01` the quotient's own
// difference. A species leaving a jump came from across it; a member of a sum
// that parks did not, though it moves, where the sum is in its own row too.
inline bool fd_jump_came_across(double back, double predicted, double d01) {
    return std::isfinite(back) && std::abs(back - predicted) > 0.5 * std::abs(d01);
}

// A mend made, and undone where its column's species turns out parked, or
// where the run did not come from across the jump.
struct FdJumpMend {
    std::size_t at; // the entry, in the solver's matrix data
    long col;       // its column
    double was;     // what the quotient had read
};

// The least change a row reads as a jump, from the sum of its terms' sizes.
inline double fd_jump_floor(double row_terms) {
    return kFdJumpNoise * std::numeric_limits<double>::epsilon() * row_terms;
}

// The tests over one row, in the row's differences: d01 the quotient's own,
// d12 one step further on, dmm two steps back, dm the step back.
inline bool fd_step_may_cross_jump(double d01, double d12, double floor) {
    return std::isfinite(d01) && std::isfinite(d12) &&
           std::abs(d01) > kFdJumpRatio * std::abs(d12) + floor;
}

inline bool fd_step_crosses_jump(double dmm, double d01, double d12, double floor) {
    return fd_step_may_cross_jump(d01, d12, floor) && std::isfinite(dmm) &&
           std::abs(d01) > kFdJumpRatio * std::max(std::abs(dmm), std::abs(d12)) + floor;
}

// The entry a row whose quotient crosses a jump takes instead: the backward
// quotient dm/h, the slope of the branch the state is on, where that step
// crosses nothing by the same test against the same outer steps; 0 where it
// does, as on `X == 0` at X = 0, whose rows have no slope to give.
inline double fd_entry_past_jump(double dmm, double dm, double d12, double floor, double h) {
    const bool clean =
        std::isfinite(dm) &&
        std::abs(dm) <= kFdJumpRatio * std::max(std::abs(dmm), std::abs(d12)) + floor;
    return clean ? dm / h : 0.0;
}

} // namespace bngsim
