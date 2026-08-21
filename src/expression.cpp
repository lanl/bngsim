// bngsim/src/expression.cpp — ExprTk-based expression evaluator
//
// ExprTk replaces muParser and provides BNG-compatible aliases.
// Supports underscore-prefixed constants, built-in functions, and time().
//
// Note: BNG2.pl's muParser exposes simulation time as the zero-arg function
// time() and leaves t free as an ordinary identifier. We mirror that here so
// that BNGL models defining a parameter / observable / species named `t`
// (a common counter pattern, e.g. `Molecules t counter()`) load successfully.

// ExprTk is a large header — compile once here.
// Disable some ExprTk features we don't need to speed up compilation.
#define exprtk_disable_string_capabilities
#define exprtk_disable_rtl_io_file
#define exprtk_disable_rtl_vecops
// BNG is case-sensitive for parameter names (e.g., k3 ≠ K3).
// ExprTk defaults to case-insensitive, which silently merges k3/K3
// into the same variable and produces incorrect trajectories.
#define exprtk_disable_caseinsensitivity
#include "exprtk.hpp"

#include "bngsim/expr_compat.hpp"
#include "bngsim/expression.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bngsim {

// ─── Mratio: confluent hypergeometric ratio M(a+1,b+1,z)/M(a,b,z) ───────────
//
// Direct port of BNG2.pl Perl2/Expression.pm `sub Mratio`
// (Fortran by W. S. Hlavacek 2018; Perl by L. A. Harris 2019).
//
// Uses Gauss's continued fraction for the ratio of contiguous Kummer 1F1
// functions, evaluated by the modified Lentz method
// [Lentz 1976 Applied Optics 15:668-671; Thompson & Barnett 1986 J Comput
// Phys 64:490-509].
//
// CF coefficients (q_j = 1 for all j; q_0 = 0):
//   p_1 = 1
//   p_2 = z * [a - (b+0)] / [(b+0) * (b+1)]
//   p_3 = z * (a + 1)     / [(b+1) * (b+2)]
//   p_4 = z * [a - (b+1)] / [(b+2) * (b+3)]
//   p_5 = z * (a + 2)     / [(b+3) * (b+4)]
//   ...
//
// Why Lentz rather than the direct power-series for M: Lentz works with
// the per-step ratio Δ_j = C_j·D_j ≈ 1 and never accumulates the partial
// sums of M(a,b,z) and M(a+1,b+1,z) themselves. For BNG inputs with large
// negative-integer `a` and large `|z|` (e.g. test_Mratio_1: a=-1000, b=9001,
// z=-10000) the equivalent series partial sums peak around 1.5e308 — past
// double's representable range — and the ratio becomes inf/inf = nan. Lentz
// stays O(1) throughout and converges in a few hundred iterations even on
// those inputs.
//
// This is the single source of truth for mratio across BNGsim: the host
// MratioFunction adapter below and the vendored NFsim mu::Parser ExprTk shim
// both call it (the shim via <bngsim/expr_compat.hpp>). See issue #49.
// ─── Where the continued fraction below can be trusted (issue #453) ─────────
//
// Outside a certain region the fraction converges to something that is not the
// ratio. It does not fail or hang: `|Delta - 1|` decays smoothly and
// geometrically into a false limit, exactly as it does into the true one, and
// then the answer is returned. Errors reach a factor of a thousand and in
// places the sign is wrong.
//
// Three things follow from that, all of them measured rather than assumed:
//
//   * No stopping test can fix it. Requiring more consecutive hits, watching
//     the error history for a plateau, demanding a minimum iteration count —
//     none of them separate a false stop from a real one, because on the way in
//     the two look the same.
//   * Iterating longer cannot fix it either. At 120 digits the same recurrence
//     does reach the true value, but only after about 1600 steps where double
//     precision settles at about 84. By then rounding has frozen the iterates.
//   * So the only honest answer is to decide beforehand whether the arguments
//     are ones the fraction handles, and to refuse when they are not.
//
// The region, from 1599 argument triples checked against a 40 to 60 digit
// reference:
//
//   * `a` a non-positive integer: safe by construction, not by measurement. The
//     odd partial numerator carries z*(a+k) and reaches exactly zero at k = -a,
//     so the fraction TERMINATES. What it computes there is a finite exact sum.
//   * `a <= 0` with `z <= 0`: 487 triples, none wrong. This is also where BNG
//     models live, since a model builds a = -min(AT,BT) and z = -1/Keq. It
//     stays true when a fit moves the counts off the integers, which is why the
//     rule is written this way rather than as a bound on |z*a|/b^2 alone: such
//     a bound refuses test_Mratio_1's own fit path.
//   * |z| <= 20: the fraction was right at every one of 3400 hostile triples
//     with |z| up to 50, chosen with small b and a on both sides of zero. The
//     first wrong answer anywhere appears at |z| = 60 and is marginal (2e-08);
//     the first serious one at |z| = 75. Twenty leaves a factor of three.
//   * b^2 >= 64*|z*a|, and for a positive z also 2*z <= b: the odd partial
//     numerators are z*(a+k)/((b+k-2)(b+k-1)), so the first half of this is the
//     statement that they start small and the fraction is a contraction from
//     the first step. The factor 64 is where leaks reached zero on the grid,
//     with a margin: 32 was the last value that leaked nothing and 16 still let
//     two through. The second half bounds the even partial numerators, which
//     the first half does not reach at all; see the note on issue #456 below.
//
// Everything else is refused, unless the asymptotic route below can vouch for
// an answer of its own. That still gives up cases the fraction would have got
// right, since it is right most of the time in the uncertain region, but there
// is no way to tell which ones, and a loud refusal is worth more than a number
// that is usually right.
//
// The `2*z <= b` companion on the last clause is issue #456. Without it the
// clause admitted a corner the grid behind issue #453 never sampled: positive
// z with a large b, where the fraction was wrong 60 times out of 3191 answers.
// `mratio(1, 901, 3000)` returned -0.4287 where the ratio is 630.7. The reason
// the clause missed it is that b*b >= 64*|z*a| bounds only the odd partial
// numerators, which carry z*(a+k); the even ones carry z*(a-b-k), so for a
// large b they are about z/b however small z*a is. Where that turns is sharp
// and it is at z = b: over 597 triples the fraction is clean through z/b =
// 1.02 and 12 of 47 are wrong at 1.05. Negative z does not need the companion
// and does not get it, measured clean out to |z|/b = 300, because there the
// partial numerators alternate in sign and the fraction stays a contraction.
// Half of b leaves a factor of two under a threshold whose position is
// understood rather than merely observed.
static bool mratio_cf_is_trustworthy(double a, double b, double z) {
    if (a <= 0.0 && a == std::floor(a)) {
        return true;
    }
    if (a <= 0.0 && z <= 0.0) {
        return true;
    }
    if (std::abs(z) <= 20.0) {
        return true;
    }
    if (b * b < 64.0 * std::abs(z * a)) {
        return false;
    }
    return z <= 0.0 || 2.0 * z <= b;
}

// ─── The asymptotic route, for arguments the fraction is refused (issue #456) ─
//
// Refusing everything outside the region above costs answers that are perfectly
// obtainable, just not by that fraction. For a large |z| the ratio has an
// asymptotic expansion in which the Gamma factors cancel between numerator and
// denominator, and that cancellation is the whole point: the individual Kummer
// functions overflow a double long before their ratio does.
//
//   z -> -inf:  R(a,b,z) = (b/(-z)) * 2F0(a+1, a-b+1; ; -1/z)
//                                   / 2F0(a,   a-b+1; ; -1/z)
//   z -> +inf:  R(a,b,z) = (b/a)    * 2F0(b-a, -a;    ;  1/z)
//                                   / 2F0(b-a, 1-a;   ;  1/z)
//
// Both series are divergent, so each is summed to its smallest term, which is
// where a Poincare expansion comes closest to the function it represents. That
// term is then an estimate of how close, and it is what makes this route safe
// to add at all: it certifies its own accuracy, so it is used only where the
// estimate is small and the refusal is kept everywhere else. The estimate was
// measured against a 40 to 60 digit reference and is honest to within about a
// factor of ten, so the 1e-10 below is worth roughly 1e-9 in the worst case.
//
// Over 5955 argument triples this turns 1179 refusals into answers, every one
// of them correct, and the worst error among them is 2.7e-11. Refusals fall
// from 46.4% of the grid to 28.5%.
//
// The fraction keeps priority wherever it is trusted. This route only ever
// looks at arguments that were going to be refused, so it cannot change an
// answer that is being given today, only replace a refusal with a value.

// isfinite() is a <math.h> macro, and the MIR JIT strips the system headers
// before handing the generated C to c2mir, so the copy of this routine in the
// generated source has no isfinite to call. Both copies spell the test out the
// same way instead, which is what keeps them numerically identical. NaN fails
// it because every comparison against a NaN is false, and so does an infinity.
// A magnitude past 1e308 is refused rather than classified, which is safe --
// refusing is always allowed -- and no argument a model produces is near it.
static bool mratio_is_finite(double v) { return v <= 1.0e308 && v >= -1.0e308; }

// Sum 2F0(alpha, beta; ; x) to its smallest term. Returns the sum; *last_term
// receives the magnitude of the final term added, the accuracy estimate.
//
// A term of exactly zero ends the series early, which happens whenever alpha or
// beta passes through a non-positive integer. The estimate is still the last
// term that was added rather than the zero that stopped it. Reporting zero
// there would be a claim the series never made: one that stops after a single
// term has demonstrated no decay at all, and calling that exact collapses the
// answer to b/(-z): at a=300, b=301, z=-21 it returned 14.33 where the ratio is
// 0.9998. Carrying the last real term forward refuses those and keeps the cases
// where the series did decay before it terminated.
static double mratio_2f0(double alpha, double beta, double x, double *last_term) {
    // A backstop rather than a tuning knob: raising it to 4000 or lowering it to
    // 200 changes nothing about which arguments get an answer, because the ones
    // that reach it are refused on the dropped branch instead. The median series
    // here is one term long and the 90th percentile is 200.
    constexpr int max_terms = 500;
    double t = 1.0;
    double s = 1.0;
    for (int k = 0; k < max_terms; ++k) {
        const double next = t * (alpha + k) * (beta + k) * x / (k + 1.0);
        if (next == 0.0 || std::abs(next) >= std::abs(t)) {
            break;
        }
        s += next;
        t = next;
    }
    *last_term = std::abs(t);
    return s;
}

// The size of the branch the expansion drops, relative to the one it keeps, as
// a logarithm. For a large |z| the Kummer function is a sum of an algebraic
// branch carrying 1/Gamma(b-a) and an exponential one carrying exp(z)/Gamma(a);
// each formula above keeps whichever dominates and drops the other, so the
// dropped branch is the error the smallest term does not account for.
//
// Working in logs is what keeps this finite: the two branches differ by
// hundreds of orders of magnitude in the cases that matter.
//
// This is also what handles the two arguments where the expansion has no
// algebraic term to keep. When b-a is a non-positive integer, 1/Gamma(b-a) is
// zero, so for z < 0 the branch being kept is absent and what is left is the
// branch being dropped. lgamma is +inf at exactly those points, which sends the
// ratio to +inf and refuses, with no separate test needed. The mirror case for
// z > 0 is a non-positive integer a, and lgamma handles that one the same way.
// Without this, M(a,a,z) over itself came back as 5e-06 where it is exactly 1.
static double mratio_dropped_branch_log(double a, double b, double z) {
    const double log_z = std::log(std::abs(z));
    double den;
    double num;
    if (z < 0.0) {
        den = std::lgamma(b - a) - std::lgamma(a) + z + (2.0 * a - b) * log_z;
        num = std::lgamma(b - a) - std::lgamma(a + 1.0) + z + (2.0 * a - b + 1.0) * log_z;
    } else {
        den = std::lgamma(a) - std::lgamma(b - a) - z + (b - 2.0 * a) * log_z;
        num = std::lgamma(a + 1.0) - std::lgamma(b - a) - z + (b - 2.0 * a - 1.0) * log_z;
    }
    // Written so that a NaN refuses. A NaN loses every comparison, so taking the
    // larger of the two would quietly drop it and hand back the other one, and
    // `!(x <= 0.0)` is true for a NaN as well as for anything positive. A
    // positive value is a refusal in any case, since it says the branch being
    // dropped is the larger of the two -- that is the +inf a Gamma pole gives.
    // A -inf is the opposite and is welcome: the dropped branch is nothing.
    if (!(den <= 0.0) || !(num <= 0.0)) {
        return 1.0;
    }
    return den > num ? den : num;
}

// Returns true and writes the ratio to *out when the expansion can vouch for
// it, false when it cannot. Every comparison is written so that a NaN fails it
// and refuses, rather than passing by accident.
static bool mratio_asymptotic(double a, double b, double z, double *out) {
    constexpr double tol = 1.0e-10;
    if (z == 0.0 || !mratio_is_finite(a) || !mratio_is_finite(b) || !mratio_is_finite(z)) {
        return false;
    }

    double sum_num = 0.0;
    double sum_den = 0.0;
    double err_num = 0.0;
    double err_den = 0.0;
    double coeff = 0.0;
    if (z < 0.0) {
        const double x = -1.0 / z;
        sum_num = mratio_2f0(a + 1.0, a - b + 1.0, x, &err_num);
        sum_den = mratio_2f0(a, a - b + 1.0, x, &err_den);
        coeff = b / (-z);
    } else {
        const double x = 1.0 / z;
        sum_num = mratio_2f0(b - a, -a, x, &err_num);
        sum_den = mratio_2f0(b - a, 1.0 - a, x, &err_den);
        // a == 0 would divide by zero here, but it is a non-positive integer, so
        // the fraction is trusted there and this is never reached with it. The
        // dropped-branch test below refuses it in any case, since lgamma(0) is
        // +inf.
        coeff = b / a;
    }
    if (sum_num == 0.0 || sum_den == 0.0 || !mratio_is_finite(sum_num) ||
        !mratio_is_finite(sum_den)) {
        return false;
    }

    // Anything above zero means the dropped branch is at least as large as the
    // kept one, which is past useless and also keeps the exp() below in range.
    const double dropped = mratio_dropped_branch_log(a, b, z);
    if (!(dropped <= 0.0)) {
        return false;
    }
    const double relative =
        err_num / std::abs(sum_num) + err_den / std::abs(sum_den) + std::exp(dropped);
    if (!(relative <= tol)) {
        return false;
    }

    const double value = coeff * sum_num / sum_den;
    if (!mratio_is_finite(value)) {
        return false;
    }
    *out = value;
    return true;
}

double expr_compat::mratio(double a, double b, double z) {
    if (!mratio_cf_is_trustworthy(a, b, z)) {
        double asymptotic = 0.0;
        if (mratio_asymptotic(a, b, z, &asymptotic)) {
            return asymptotic;
        }
        throw std::runtime_error(
            "mratio(a, b, z): the continued fraction this uses is not reliable for these "
            "arguments and would return a wrong value without saying so, and the asymptotic "
            "expansion that covers part of the rest cannot vouch for an answer here either (a=" +
            std::to_string(a) + ", b=" + std::to_string(b) + ", z=" + std::to_string(z) +
            "). The fraction is reliable when a is a non-positive integer, when a <= 0 and "
            "z <= 0, when |z| <= 20, or when b*b >= 64*|z*a| and z is either negative or no "
            "more than half of b. BNG builds a = -min(AT,BT) and z = -1/Keq, which is inside "
            "that region, so a model reaching this is using mratio in its own way. See "
            "lanl/bngsim issues #453 and #456.");
    }
    constexpr double eps = 1.0e-16;
    constexpr double tiny = 1.0e-32;
    // Safety cap so a pathological non-converging case fails loud rather
    // than hanging. BNG's reference has no cap; in practice the supported
    // parameter ranges converge in well under this bound.
    constexpr int max_iter = 100000;

    // Initialize per the modified-Lentz recipe: f_0 = q_0, but q_0 = 0
    // here, so substitute `tiny`. C_0 = f_0, D_0 = 0.
    double fsave = tiny;
    double Csave = fsave;
    double Dsave = 0.0;
    double err = 1.0 + eps;

    // Parity bookkeeping: even-indexed and odd-indexed CF terms use
    // different formulas for p_j. `odd` alternates after every step.
    //
    // This used to swap `odd` with a second flag that nothing ever read, which
    // is the same thing written at more length. It is a toggle here because the
    // generated C carries a copy of this routine (issue #451) and c2mir, the
    // frontend of the MIR JIT backend, miscompiles the swap: the JIT returned a
    // wrong number for every argument. Keeping the two spellings identical is
    // what makes this function the single source of truth for both.
    int odd = 1;
    int iodd = 0;
    int ieven = 0;
    double f = 0.0;

    int j = 0;
    while (err > eps) {
        ++j;
        if (j > max_iter) {
            throw std::runtime_error("mratio: modified-Lentz continued fraction failed to converge "
                                     "within " +
                                     std::to_string(max_iter) +
                                     " iterations "
                                     "(a=" +
                                     std::to_string(a) + ", b=" + std::to_string(b) +
                                     ", z=" + std::to_string(z) + ")");
        }

        double p;
        if (j == 1) {
            p = 1.0;
        } else {
            const double den = (b + (j - 2)) * (b + (j - 1));
            double num;
            if (odd == 1) {
                ++iodd;
                num = z * (a + iodd);
            } else {
                ++ieven;
                num = z * (a - (b + (ieven - 1)));
            }
            p = num / den;
        }
        constexpr double q = 1.0;

        double D = q + p * Dsave;
        if (std::abs(D) < tiny) {
            D = tiny;
        }
        double C = q + p / Csave;
        if (std::abs(C) < tiny) {
            C = tiny;
        }
        D = 1.0 / D;

        const double Delta = C * D;
        f = Delta * fsave;
        err = std::abs(Delta - 1.0);

        fsave = f;
        Csave = C;
        Dsave = D;
        odd = 1 - odd;
    }
    return f;
}

// ─── Non-finite return diagnostic ─────────────────────────────────────────────
//
// Custom functions handed to ExprTk that return nan/inf produce silent
// propagation through every downstream expression — exactly how issue #42
// went undiagnosed for so long (mratio overflowed to nan, and U1_U0 /
// C_mean / C_sdev / C_theory all just became nan with no logging).
//
// This helper traps the next such bug by stamping a one-time warning on
// stderr whenever a registered function returns a non-finite value, then
// deduplicating by (function name + argument bit-pattern) so that a
// long-running ODE that repeatedly evaluates the same bad input prints
// once rather than once-per-step. Argument bits are compared verbatim
// (no value equality), so nan-valued inputs deduplicate cleanly.
//
// Lives on ExprTkEvaluator::Impl; each adapter holds a back-pointer that
// is null only in default-constructed temporaries.
class NonFiniteWarningSet {
  public:
    void warn_if_nonfinite(const char *fname, std::initializer_list<double> args, double result) {
        if (std::isfinite(result)) {
            return;
        }
        std::string key(fname);
        for (double a : args) {
            std::uint64_t bits;
            std::memcpy(&bits, &a, sizeof(bits));
            char buf[20];
            std::snprintf(buf, sizeof(buf), ":%016llx", static_cast<unsigned long long>(bits));
            key.append(buf);
        }
        if (!seen_.insert(std::move(key)).second) {
            return;
        }
        std::cerr << "bngsim: warning: '" << fname << "(";
        const char *sep = "";
        for (double a : args) {
            std::cerr << sep << a;
            sep = ", ";
        }
        std::cerr << ")' returned " << result
                  << "; the value will propagate through any expression that "
                     "references this call. Further occurrences with the same "
                     "arguments will be silent."
                  << std::endl;
    }

  private:
    std::unordered_set<std::string> seen_;
};

// ─── ExprTk custom function adapters ─────────────────────────────────────────
//
// ExprTk requires inheriting from ifunction for custom functions. Each
// adapter holds a back-pointer to a NonFiniteWarningSet so a function
// that returns nan/inf is surfaced once on stderr instead of silently
// propagating (issue #42 follow-up).

// 3-arg: mratio(a, b, z)
template <typename T> struct MratioFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    MratioFunction() : exprtk::ifunction<T>(3) {
        exprtk::ifunction<T>::allow_zero_parameters() = false;
    }
    T operator()(const T &a, const T &b, const T &z) override {
        const double da = static_cast<double>(a);
        const double db = static_cast<double>(b);
        const double dz = static_cast<double>(z);
        const double r = expr_compat::mratio(da, db, dz);
        if (warner) {
            warner->warn_if_nonfinite("mratio", {da, db, dz}, r);
        }
        return static_cast<T>(r);
    }
};

// 1-arg aliases for backward compat
template <typename T> struct LnFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    LnFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = std::log(dx);
        if (warner) {
            warner->warn_if_nonfinite("ln", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct RintFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    RintFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = std::round(dx);
        if (warner) {
            warner->warn_if_nonfinite("rint", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct SignFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    SignFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = (dx > 0.0) ? 1.0 : ((dx < 0.0) ? -1.0 : 0.0);
        if (warner) {
            warner->warn_if_nonfinite("sign", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

// tgamma(x) = Γ(x); the SBML loader emits ``tgamma((n)+1)`` for MathML
// factorial(n) (ExprTk has no gamma builtin). std::tgamma matches the C
// library form native codegen emits, so interpreted and codegen agree.
template <typename T> struct TgammaFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    TgammaFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = std::tgamma(dx);
        if (warner) {
            warner->warn_if_nonfinite("tgamma", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

// 0-arg: time() — reads from a bound double*. No warner: the simulator
// owns the time pointer and a non-finite t would be a higher-level bug
// flagged by the integrator, not by this layer.
template <typename T> struct TimeFunction : public exprtk::ifunction<T> {
    double *time_ptr = nullptr;
    TimeFunction() : exprtk::ifunction<T>(0) {
        exprtk::ifunction<T>::allow_zero_parameters() = true;
    }
    T operator()() override { return time_ptr ? static_cast<T>(*time_ptr) : T(0); }
};

// Adapter for std::function-based custom functions (0-3 args). The
// user-supplied function name is stored so the warning message can
// identify the offender — define_function() copies it from the
// registration argument.
template <typename T> struct StdFunc0Adapter : public exprtk::ifunction<T> {
    std::function<double()> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc0Adapter(std::function<double()> f) : exprtk::ifunction<T>(0), fn(std::move(f)) {
        exprtk::ifunction<T>::allow_zero_parameters() = true;
    }
    T operator()() override {
        const double r = fn();
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct StdFunc1Adapter : public exprtk::ifunction<T> {
    std::function<double(double)> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc1Adapter(std::function<double(double)> f) : exprtk::ifunction<T>(1), fn(std::move(f)) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = fn(dx);
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {dx}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct StdFunc2Adapter : public exprtk::ifunction<T> {
    std::function<double(double, double)> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc2Adapter(std::function<double(double, double)> f)
        : exprtk::ifunction<T>(2), fn(std::move(f)) {}
    T operator()(const T &x, const T &y) override {
        const double dx = static_cast<double>(x);
        const double dy = static_cast<double>(y);
        const double r = fn(dx, dy);
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {dx, dy}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct StdFunc3Adapter : public exprtk::ifunction<T> {
    std::function<double(double, double, double)> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc3Adapter(std::function<double(double, double, double)> f)
        : exprtk::ifunction<T>(3), fn(std::move(f)) {}
    T operator()(const T &x, const T &y, const T &z) override {
        const double dx = static_cast<double>(x);
        const double dy = static_cast<double>(y);
        const double dz = static_cast<double>(z);
        const double r = fn(dx, dy, dz);
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {dx, dy, dz}, r);
        }
        return static_cast<T>(r);
    }
};

// ─── ExprTk implementation (pimpl) ───────────────────────────────────────────

// ─── Name remapping ──────────────────────────────────────────────────────────
//
// Two transparent transformations bridge BNG identifier conventions onto
// ExprTk's identifier rules:
//
//   1. Underscore-prefixed names: ExprTk rejects identifiers starting with
//      '_'. BNG built-ins use _pi, _e, _NA, etc. We map "_X" → "u_X" on
//      registration and rewrite the same tokens in compiled expressions.
//      This mapping is unconditional (no built-in ExprTk symbol starts with
//      '_'), so the rewrite is safe to apply token-by-token in expressions.
//
//   2. ExprTk reserved-word collisions: ExprTk rejects add_variable() for
//      names matching its reserved-word/reserved-symbol lists (e.g., a BNG
//      parameter literally named `const`, `true`, `false`). We register the
//      user's variable under "r_<name>" and rewrite references in compiled
//      expressions. Unlike (1) this rewrite is *conditional* — built-in
//      functions like `sin`, `if`, `time` must remain literal tokens — so
//      we only rewrite identifiers that were actually mangled at
//      registration. The mangling map is per-evaluator and lives on Impl.

static const std::unordered_set<std::string> &exprtk_reserved_identifiers() {
    static const std::unordered_set<std::string> reserved = [] {
        std::unordered_set<std::string> names;
        names.reserve(exprtk::details::reserved_words_size +
                      exprtk::details::reserved_symbols_size);
        for (std::size_t i = 0; i < exprtk::details::reserved_words_size; ++i) {
            names.insert(exprtk::details::reserved_words[i]);
        }
        for (std::size_t i = 0; i < exprtk::details::reserved_symbols_size; ++i) {
            names.insert(exprtk::details::reserved_symbols[i]);
        }
        return names;
    }();
    return reserved;
}

static const std::unordered_set<std::string> &bngsim_exprtk_aliases() {
    static const std::unordered_set<std::string> aliases = {"ln", "rint", "sign", "mratio", "time"};
    return aliases;
}

// Registration keys occupied by bngsim's built-in constants. init_builtins()
// registers each "_X" constant (Planck's `_h`, Avogadro's `_NA`, …) under the
// ExprTk key "u_X" because ExprTk rejects a leading '_'. A user parameter
// named literally "u_h" / "u_pi" / … maps to that same key (it does not start
// with '_', so compute_registration_name() would leave it unchanged) and
// would collide with the constant slot at registration (GH #90: Chitnis2012
// / BIOMD0000000950 has a parameter `u_h`). Treat these keys as reserved so
// such names take the r_ mangling path like any other reserved-word collision.
// Must stay in sync with the "_X" constants registered in init_builtins().
static const std::unordered_set<std::string> &bngsim_remapped_constant_keys() {
    static const std::unordered_set<std::string> keys = {"u_pi", "u_e", "u_kB", "u_NA",
                                                         "u_R",  "u_h", "u_F"};
    return keys;
}

bool expr_compat::is_exprtk_reserved(const std::string &name) {
    // Names that, if registered as a user variable, would collide with
    // a name already taken by the symbol table. Two sources:
    //
    //   1. ExprTk's current reserved_words[] + reserved_symbols[] lists,
    //      read directly from the vendored upstream header so bumps cannot
    //      silently drift away from our mangling assumptions.
    //
    //   2. bngsim-specific function aliases registered in
    //      ExprTkEvaluator::Impl::init_builtins() (`ln`, `rint`, `sign`,
    //      `mratio`, `time`). ExprTk would reject add_variable() on any
    //      of these once the function is registered, but the names are
    //      not on ExprTk's reserved_symbols[] list, so we have to track
    //      them here ourselves. BNG2.pl's parser already rejects
    //      `ln`/`rint`/`mratio`/`time` as parameter names upstream, so
    //      models reaching bngsim via `generate_network` can only hit
    //      the `sign` collision in practice — but we mangle all five
    //      for symmetry and to handle hand-crafted .net inputs.
    //
    //   3. The registration keys bngsim's built-in constants occupy after the
    //      unconditional "_X" → "u_X" remap (`u_pi`, …, `u_h`, `u_F`). A user
    //      parameter named literally `u_h` would otherwise alias Planck's
    //      constant slot and fail to register (GH #90).
    //
    // Comparison is exact (case-sensitive) because we build with
    // exprtk_disable_caseinsensitivity, so e.g. `Const` is not reserved.
    const auto &exprtk_reserved = exprtk_reserved_identifiers();
    const auto &bngsim_aliases = bngsim_exprtk_aliases();
    const auto &constant_keys = bngsim_remapped_constant_keys();
    return exprtk_reserved.find(name) != exprtk_reserved.end() ||
           bngsim_aliases.find(name) != bngsim_aliases.end() ||
           constant_keys.find(name) != constant_keys.end();
}

// Unconditional leading-underscore remap "_X" → "u_X" (see expr_compat.hpp).
std::string expr_compat::remap_name(const std::string &name) {
    if (!name.empty() && name[0] == '_') {
        return "u_" + name.substr(1);
    }
    return name;
}

// Compute the symbol-table key for `name` at registration time.
// Combines the unconditional underscore remap with reserved-word mangling.
std::string expr_compat::compute_registration_name(const std::string &name) {
    std::string underscore_mapped = remap_name(name);
    if (underscore_mapped != name) {
        return underscore_mapped;
    }
    if (is_exprtk_reserved(name)) {
        return "r_" + name;
    }
    return name;
}

struct ExprTkEvaluator::Impl {
    using SymbolTable = exprtk::symbol_table<double>;
    using Expression = exprtk::expression<double>;
    using Parser = exprtk::parser<double>;

    SymbolTable symbol_table;
    // The parser belongs to THIS evaluator and to no other (issue #257), and is
    // built on first compile rather than in the constructor.
    //
    // It used to be a shared_ptr handed to every clone, with a mutex alongside
    // it, because compile() is not reentrant: it drives a lexer, a token scanner
    // and an error list that all live in the parser, so two threads compiling
    // through one parser corrupt each other (issue #201 — SIGSEGV/SIGABRT/SIGBUS,
    // or an ERR244 "unregistered symbol" when one compile's symbol resolution is
    // clobbered by another). Serializing compile() was necessary but not
    // sufficient, and the residue was the ~10% flake in
    // test_expression_parser_thread_safety.py: exprtk's compile() ends with
    //
    //     symtab_store_.symtab_list_ = expr.get_symbol_table_list();
    //
    // and never clears it, so the parser holds a strong handle on the symbol
    // table of the last expression compiled through it. symbol_table's refcount
    // is a plain std::size_t. Two evaluators behind one parser therefore means
    // thread B's compile (inside the mutex) dropping thread A's symbol table
    // while thread A churns that same counter outside it — in
    // register_symbol_table() below, in the growth of `expressions`, and in its
    // own destructor. A lost update runs clear() on a symbol table whose
    // variable addresses are already baked into A's compiled nodes, and the
    // crash surfaces wherever the corrupted heap is next touched: the reported
    // one is a malloc freelist trap (SIGTRAP) in an unrelated allocation.
    //
    // Nothing about that was fixable with another lock, because the counter is
    // touched on paths no evaluator API sees (vector growth, destruction). What
    // is fixable is the sharing, and the sharing bought nothing: NetworkModel's
    // default constructor builds an evaluator that clone() immediately replaces,
    // so the parser construction the sharing "saved" was being paid and
    // discarded on every clone anyway. Lazy construction genuinely saves it.
    //
    // Still NOT a hot path: evaluate() reads `expressions` and never touches the
    // parser, so the integration loop is unaffected. Compilation happens at
    // build, at clone, and in the lazy memos in model.cpp.
    std::unique_ptr<Parser> parser;
    std::vector<Expression> expressions;

    // Cached preprocessed expression strings for efficient clone().
    // Indexed in parallel with `expressions`.
    std::vector<std::string> preprocessed_strings;

    // Owned custom function objects (must outlive the symbol table)
    MratioFunction<double> mratio_func;
    LnFunction<double> ln_func;
    RintFunction<double> rint_func;
    SignFunction<double> sign_func;
    TgammaFunction<double> tgamma_func;
    TimeFunction<double> time_func;

    // User-registered custom functions (owned, heap-allocated)
    std::vector<std::unique_ptr<exprtk::ifunction<double>>> user_functions;

    // Diagnostic state: every custom-function adapter carries a back-pointer
    // to this set, so a function that returns nan/inf gets one warning on
    // stderr the first time a given (name, args) tuple misbehaves. Per
    // evaluator (no globals); not copied across clone_empty().
    NonFiniteWarningSet nonfinite_warner;

    // Names that were mangled at registration to avoid ExprTk reserved-word
    // collisions (key: original BNG name, value: ExprTk symbol-table key).
    // Underscore-prefixed names (e.g., _pi) are NOT recorded here — they
    // remap unconditionally via compute_registration_name() in both
    // directions, so no per-evaluator state is needed.
    std::unordered_map<std::string, std::string> mangled_user_names;

    // BNG-source names registered as scalar variables/constants on this
    // evaluator (i.e., everything that went through define_variable /
    // add_remapped_constant). Used by strip_empty_parens() to rewrite
    // `obs()` → `obs` for observable references that the BNG parser
    // accepts (Expression.pm:870-927 — Observable as zero-arg call) but
    // ExprTk's grammar would reject as `obs * ()`.
    std::unordered_set<std::string> scalar_variable_names;

    // Look up the symbol-table key for `name` when rewriting an expression.
    // Mirrors compute_registration_name() but only mangles reserved words
    // that were actually registered on this evaluator, so built-in tokens
    // (sin, if, time, ...) pass through unchanged.
    std::string remap_token(const std::string &name) const {
        std::string underscore_mapped = expr_compat::remap_name(name);
        if (underscore_mapped != name) {
            return underscore_mapped;
        }
        auto it = mangled_user_names.find(name);
        if (it != mangled_user_names.end()) {
            return it->second;
        }
        return name;
    }

    // Rewrite `name()` → `name` for any identifier registered as a scalar
    // variable on this evaluator. BNGL's grammar (per BNG2.pl's
    // bng2/Perl2/Expression.pm:870-927) accepts an Observable as a
    // zero-arg call (`obs()`), and BNG2.pl preserves that syntax verbatim
    // when emitting .net rate laws / parameter expressions / event
    // expressions. ExprTk's grammar would parse `obs()` as the implicit
    // multiplication `obs * ()` and reject the empty parens with ERR248.
    // We close the gap by stripping the trailing `()` for any name we
    // know is a scalar — leaving function names (built-ins like `sin`,
    // `time`, user-defined Func0/1/2/3) untouched, since those go
    // through add_function and are not in scalar_variable_names.
    std::string strip_empty_parens(const std::string &expr) const {
        if (scalar_variable_names.empty())
            return expr;
        std::string result;
        result.reserve(expr.size());
        size_t i = 0;
        while (i < expr.size()) {
            const bool at_boundary =
                (i == 0) ||
                (!std::isalnum(static_cast<unsigned char>(expr[i - 1])) && expr[i - 1] != '_');
            const bool ident_start =
                (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_');
            if (at_boundary && ident_start) {
                size_t start = i;
                i++;
                while (i < expr.size() &&
                       (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_')) {
                    i++;
                }
                std::string ident = expr.substr(start, i - start);
                if (i + 1 < expr.size() && expr[i] == '(' && expr[i + 1] == ')' &&
                    scalar_variable_names.count(ident)) {
                    result += ident;
                    i += 2;
                } else {
                    result += ident;
                }
                continue;
            }
            result += expr[i];
            i++;
        }
        return result;
    }

    // Rewrite all identifier tokens in `expr` through remap_token().
    std::string remap_expression(const std::string &expr) const {
        std::string result;
        result.reserve(expr.size() + 16);
        size_t i = 0;
        while (i < expr.size()) {
            const bool at_boundary =
                (i == 0) ||
                (!std::isalnum(static_cast<unsigned char>(expr[i - 1])) && expr[i - 1] != '_');
            const bool ident_start =
                (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_');
            if (at_boundary && ident_start) {
                size_t start = i;
                i++;
                while (i < expr.size() &&
                       (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_')) {
                    i++;
                }
                const std::string token = expr.substr(start, i - start);
                // A declared symbol that collides with an ExprTk reserved name
                // and is *also* used in call form (`frac(x)`) is genuinely
                // ambiguous in a single flat namespace — raise the same clear,
                // deterministic error the NFsim mu::Parser shim does, rather
                // than rewriting to r_<name> and letting ExprTk emit a cryptic
                // "not a function" message. strip_empty_parens() has already
                // removed legitimate zero-arg scalar calls (`obs()`), so any
                // `name(` left for a mangled symbol is a real call form. Only
                // reserved-mangled names are tracked in mangled_user_names;
                // underscore remaps are not, so they fall through untouched.
                if (mangled_user_names.count(token)) {
                    size_t j = i;
                    while (j < expr.size() && std::isspace(static_cast<unsigned char>(expr[j]))) {
                        j++;
                    }
                    if (j < expr.size() && expr[j] == '(') {
                        throw std::runtime_error(
                            "identifier '" + token +
                            "' is both a declared model "
                            "symbol and used as a function call '" +
                            token +
                            "(...)'; "
                            "ExprTk reserves '" +
                            token +
                            "' as a built-in and a single "
                            "flat namespace cannot hold both meanings — rename the "
                            "model symbol");
                    }
                }
                result += remap_token(token);
                continue;
            }
            result += expr[i];
            i++;
        }
        return result;
    }

    void add_remapped_constant(const std::string &name, double value) {
        std::string mapped = expr_compat::compute_registration_name(name);
        if (mapped != name && name[0] != '_') {
            mangled_user_names[name] = mapped;
        }
        symbol_table.add_constant(mapped, value);
        scalar_variable_names.insert(name);
    }

    void init_builtins() {
        // Register built-in constants
        // Note: ExprTk rejects '_' prefix, so we remap to "u_" internally
        add_remapped_constant("_pi", 3.14159265358979323846);
        add_remapped_constant("_e", 2.71828182845904523536);
        add_remapped_constant("_NA", 6.02214076e23);
        add_remapped_constant("_kB", 1.380649e-23);
        add_remapped_constant("_R", 8.314462618153241);
        add_remapped_constant("_h", 6.62607015e-34);
        add_remapped_constant("_F", 96485.33212331002);

        // Wire the non-finite-return diagnostic into every owned adapter.
        // time_func is intentionally omitted (see TimeFunction comment).
        mratio_func.warner = &nonfinite_warner;
        ln_func.warner = &nonfinite_warner;
        rint_func.warner = &nonfinite_warner;
        sign_func.warner = &nonfinite_warner;
        tgamma_func.warner = &nonfinite_warner;

        // Register backward-compatible aliases
        symbol_table.add_function("ln", ln_func);
        symbol_table.add_function("rint", rint_func);
        symbol_table.add_function("sign", sign_func);
        symbol_table.add_function("tgamma", tgamma_func);

        // Register built-in functions
        // Note: ExprTk has a built-in `if` keyword (grammar-level) that handles
        // if(cond, true_val, false_val) natively. No custom function needed.
        symbol_table.add_function("mratio", mratio_func);

        // time() — pointer set later via set_time_ptr().
        // We do NOT register `t` here so that `t` is free for use as a model
        // identifier (matches BNG2.pl convention).
        symbol_table.add_function("time", time_func);
    }

    Impl() { init_builtins(); }

    // Build the parser on demand. An evaluator that never compiles anything —
    // every discarded default-constructed one on the clone path, and every
    // expression-free model — never pays for it.
    Parser &acquire_parser() {
        if (!parser) {
            parser = std::make_unique<Parser>();
            // Increase max stack depth for deeply nested if() expressions.
            // ExprTk default is 400 (~200 nested if()), muParser handled 2000.
            parser->settings().set_max_stack_depth(4096);
        }
        return *parser;
    }

    void set_time_ptr(double *ptr) { time_func.time_ptr = ptr; }
};

ExprTkEvaluator::ExprTkEvaluator() : impl_(std::make_unique<Impl>()) {}

ExprTkEvaluator::~ExprTkEvaluator() = default;

void ExprTkEvaluator::define_variable(const std::string &name, double *addr) {
    std::string mapped = expr_compat::compute_registration_name(name);
    if (!impl_->symbol_table.add_variable(mapped, *addr)) {
        const bool reserved = expr_compat::is_exprtk_reserved(name);
        std::string detail;
        if (reserved) {
            // The mangled form already exists — most often because another
            // BNG name collides with the mangled key (e.g., user has both
            // `const` and a separate `r_const`).
            detail = "name '" + name + "' is an ExprTk reserved word; bngsim mangles it to '" +
                     mapped +
                     "', and that mangled key is already registered. Rename one of the "
                     "conflicting parameters.";
        } else {
            detail = "name '" + name + "' (mapped: '" + mapped +
                     "') is already registered. Check the .net file for duplicate "
                     "parameter / observable / species names (case-sensitive).";
        }
        throw std::runtime_error("ExprTk: failed to register variable '" + name + "'. " + detail);
    }
    if (mapped != name && (name.empty() || name[0] != '_')) {
        impl_->mangled_user_names[name] = mapped;
    }
    impl_->scalar_variable_names.insert(name);
}

void ExprTkEvaluator::define_constant(const std::string &name, double value) {
    impl_->add_remapped_constant(name, value);
}

void ExprTkEvaluator::define_function(const std::string &name, Func0 fn) {
    auto adapter = std::make_unique<StdFunc0Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func1 fn) {
    auto adapter = std::make_unique<StdFunc1Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func2 fn) {
    auto adapter = std::make_unique<StdFunc2Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func3 fn) {
    auto adapter = std::make_unique<StdFunc3Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

// ─── C-style logical operator replacement ────────────────────────────────────
// BNG2.pl emits C-style && and || in if() conditions, but ExprTk uses
// 'and' and 'or' keywords. Replace before compilation.
static std::string replace_logical_operators(const std::string &expr) {
    std::string result;
    result.reserve(expr.size() + 16);
    for (size_t i = 0; i < expr.size(); ++i) {
        if (i + 1 < expr.size()) {
            if (expr[i] == '&' && expr[i + 1] == '&') {
                result += " and ";
                i++; // skip second '&'
                continue;
            }
            if (expr[i] == '|' && expr[i + 1] == '|') {
                result += " or ";
                i++; // skip second '|'
                continue;
            }
        }
        result += expr[i];
    }
    return result;
}

int ExprTkEvaluator::compile(const std::string &expr) {
    // Replace C-style logical operators before any other processing
    std::string preprocessed = replace_logical_operators(expr);

    // Strip `obs()` → `obs` for any name registered as a scalar variable.
    // BNGL accepts Observable references as zero-arg calls; ExprTk does
    // not. Run before remap_expression so we match against BNG-source
    // names, not their post-mangling forms.
    std::string stripped = impl_->strip_empty_parens(preprocessed);

    // Remap identifiers before ExprTk compilation:
    //   - Unconditional: "_X" → "u_X" (ExprTk rejects '_' prefix)
    //   - Conditional:   reserved-word names registered on this evaluator
    //                    (e.g., user's `const` → `r_const`)
    std::string remapped = impl_->remap_expression(stripped);

    // Delegate to compile_preprocessed (which also caches the string)
    return compile_preprocessed(remapped);
}

int ExprTkEvaluator::compile_preprocessed(const std::string &preprocessed_expr) {
    Impl::Expression expression;
    expression.register_symbol_table(impl_->symbol_table);

    // No lock: the parser belongs to this evaluator alone (issue #257), so the
    // error() read below can only be reporting this compile()'s own failure.
    // With the parser shared, the two had to be one critical section, because
    // error() reads state the next compile() on that parser overwrites (#201).
    Impl::Parser &parser = impl_->acquire_parser();
    if (!parser.compile(preprocessed_expr, expression)) {
        throw std::runtime_error("ExprTk compilation failed for expression: '" + preprocessed_expr +
                                 "' — " + parser.error());
    }

    int id = static_cast<int>(impl_->expressions.size());
    impl_->expressions.push_back(std::move(expression));
    impl_->preprocessed_strings.push_back(preprocessed_expr);
    return id;
}

const std::string &ExprTkEvaluator::preprocessed_expr(int expr_id) const {
    if (expr_id < 0 || expr_id >= static_cast<int>(impl_->preprocessed_strings.size())) {
        throw std::runtime_error("Invalid expression ID: " + std::to_string(expr_id));
    }
    return impl_->preprocessed_strings[expr_id];
}

std::vector<const double *> ExprTkEvaluator::referenced_variable_addresses(int expr_id) const {
    std::vector<const double *> out;
    if (expr_id < 0 || expr_id >= static_cast<int>(impl_->preprocessed_strings.size())) {
        return out;
    }
    // Collect the variable identifiers referenced by the (already remapped)
    // preprocessed string, then resolve each through this evaluator's symbol
    // table to the address it was bound to via define_variable. Names not
    // registered as variables (constants, functions) resolve to null and are
    // skipped, so the result contains only model-variable addresses.
    std::vector<std::string> names;
    // Pass the symbol table so the collector resolves built-in/user functions
    // (e.g. time()) instead of bailing out when it meets an unknown token — the
    // symbol-table-less overload returns nothing for a trigger like
    // `time() >= t_dose`, which would silently hide a parameter reference.
    if (!exprtk::collect_variables(impl_->preprocessed_strings[expr_id], impl_->symbol_table,
                                   names)) {
        return out;
    }
    out.reserve(names.size());
    for (const std::string &nm : names) {
        auto *var = impl_->symbol_table.get_variable(nm);
        if (var != nullptr) {
            out.push_back(&var->ref());
        }
    }
    return out;
}

int ExprTkEvaluator::n_expressions() const { return static_cast<int>(impl_->expressions.size()); }

double ExprTkEvaluator::evaluate(int expr_id) {
    if (expr_id < 0 || expr_id >= static_cast<int>(impl_->expressions.size())) {
        throw std::runtime_error("Invalid expression ID: " + std::to_string(expr_id));
    }
    return impl_->expressions[expr_id].value();
}

void ExprTkEvaluator::set_time_ptr(double *time_addr) { impl_->set_time_ptr(time_addr); }

// ─── Clone support ───────────────────────────────────────────────────────────

ExprTkEvaluator::ExprTkEvaluator(ExprTkEvaluator &&) noexcept = default;
ExprTkEvaluator &ExprTkEvaluator::operator=(ExprTkEvaluator &&) noexcept = default;

const void *ExprTkEvaluator::parser_identity() const { return impl_->parser.get(); }

std::unique_ptr<ExprTkEvaluator> ExprTkEvaluator::clone_empty() const {
    // Nothing is carried over — not the parser (issue #257), not the symbol
    // table, not the expression list. The caller re-registers and re-compiles.
    return std::make_unique<ExprTkEvaluator>();
}

// ─── Reserved names ──────────────────────────────────────────────────────────

ReservedNames reserved_names() {
    ReservedNames names;
    names.constants = {"_pi", "_e", "_kB", "_NA", "_R", "_h", "_F"};
    names.functions = {"time",  "sin",   "cos",   "tan",    "asin",  "acos", "atan",  "sinh",
                       "cosh",  "tanh",  "asinh", "acosh",  "atanh", "exp",  "log",   "ln",
                       "log2",  "log10", "sqrt",  "abs",    "floor", "ceil", "round", "rint",
                       "trunc", "min",   "max",   "clamp",  "avg",   "sum",  "erf",   "erfc",
                       "sign",  "sgn",   "if",    "mratio", "tgamma"};
    return names;
}

} // namespace bngsim
