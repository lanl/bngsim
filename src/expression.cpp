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

#include "bngsim/c_locale_numbers.hpp"
#include "bngsim/expr_compat.hpp"
#include "bngsim/expression.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <iostream>
#include <limits>
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
//   * `a` a negative integer: safe by construction, not by measurement. The
//     odd partial numerator carries z*(a+k) and reaches exactly zero at k = -a,
//     so the fraction TERMINATES. What it computes there is a finite exact sum.
//     Not `a == 0`: the odd numerators start at k = 1, so for a = 0 none of them
//     vanishes and the fraction runs on (issue #768). mratio(0, 1, 100) is
//     (e^100 - 1)/100 = 2.69e41 and this returned 1.28e15; mratio(0, 0.5, 40)
//     came back negative.
//   * `a <= 0` with `z <= 0`: 487 triples, none wrong. This is also where BNG
//     models live, since a model builds a = -min(AT,BT) and z = -1/Keq. It
//     stays true when a fit moves the counts off the integers, which is why the
//     rule is written this way rather than as a bound on |z*a|/b^2 alone: such
//     a bound refuses test_Mratio_1's own fit path.
//     Except past z = -20 with b - a < 1, which the grid never reached (issue
//     #1080). There the fraction converges to the ratio but loses about
//     10*eps*sqrt(-z)/(b - a) of it, so with both b and |a| tiny it is wrong:
//     with b in [1e-12, 10], |a| in [1e-15, 10] and z in [-2000, 0], 2605 of
//     20000 triples were off by more than 1e-8, the worst by 1.2e-2, and past
//     z = -1e5 by factors. mratio(-1e-12, 1e-9, -25) returned 5.56205069e-11
//     where the ratio is 5.56205472e-11. That corner is summed directly up to
//     z = -700 (mratio_series below) and refused here, which leaves anything
//     further out to the asymptotic route. With b - a >= 1 the loss is below
//     1e-13 out to z = -1e6, and what BNG builds, a = -min(AT,BT) with
//     b = |AT-BT| + 1, has b - a >= 1 in any case.
//   * |z| <= 20: the fraction was right at every one of 3400 hostile triples
//     with |z| up to 50, chosen with small b and a on both sides of zero. The
//     first wrong answer anywhere appears at |z| = 60 and is marginal (2e-08);
//     the first serious one at |z| = 75. Twenty leaves a factor of three.
//     No false limit there, but not every digit either (issue #1080): with
//     |a| tiny next to a b below 1 the fraction lost up to 4.4e-7 near z = 20,
//     and with b itself tiny just over 1e-8 near z = -20, here and in the
//     clause above. Those arguments never reach this rule now; see
//     mratio_series below.
//   * b^2 >= 64*|z*a|, and for a positive z also 2*z <= b: the odd partial
//     numerators are z*(a+k)/((b+k-2)(b+k-1)), so the first half of this is the
//     statement that they start small and the fraction is a contraction from
//     the first step. The factor 64 is where leaks reached zero on the grid,
//     with a margin: 32 was the last value that leaked nothing and 16 still let
//     two through. The second half bounds the even partial numerators, which
//     the first half does not reach at all; see the note on issue #456 below.
//
// All of it for `b > 0` only (issue #768). The grids behind every clause had a
// positive b, and with b negative and not an integer the fraction settles on a
// false limit inside each of them: 21 of 200 triples wrong at a < 0 and z <= 0
// (mratio(-22.52, -25.18, -12.00) came back positive where the ratio is
// -2.337), 28 of 200 at |z| <= 20, and 16 of 200 even where a is a negative
// integer and the fraction terminates, out to the wrong sign
// (mratio(-20, -23.34, -10.87) returned 5.935 where the ratio is -2.073). BNG
// builds b = |AT-BT| + 1, so no model it writes has b <= 0.
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
    if (!(b > 0.0)) {
        return false;
    }
    if (a < 0.0 && a == std::floor(a)) {
        return true;
    }
    if (a <= 0.0 && z < -20.0 && b - a < 1.0) {
        return false;
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
        // a == 0 is M(0, b, z) = 1, which is all its expansion says: every term
        // after the first carries a, and with 1/Gamma(a) = 0 it drops no branch.
        // The zero-term rule above would report the unit first term as the
        // error and refuse; it guards a sum that stopped without showing it had
        // decayed, and this one is exact. It matters for a = 0 past z = -700
        // with 0 < b < 1 (issue #1080), the one a = 0, z < 0, b > 0 the rule
        // refuses, which the fraction got right unless b was tiny. A b <= 0
        // asks here too and is answered on the same terms as any other.
        if (a == 0.0) {
            sum_den = 1.0;
            err_den = 0.0;
        } else {
            sum_den = mratio_2f0(a, a - b + 1.0, x, &err_den);
        }
        coeff = b / (-z);
    } else {
        // a == 0 would divide by zero below. With z > 0 it is summed exactly
        // before the fraction is asked (issue #768); refused here in any case.
        if (a == 0.0) {
            return false;
        }
        const double x = 1.0 / z;
        sum_num = mratio_2f0(b - a, -a, x, &err_num);
        sum_den = mratio_2f0(b - a, 1.0 - a, x, &err_den);
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

double expr_compat::rint(double x) { return std::floor(x + 0.5); }

// ─── Numeric literals, read as strtod reads them ─────────────────────────────
//
// ExprTk's string_to_real accumulates the digits in a double and then scales by
// a power of ten, rounding at each step, where strtod — and so a C compiler,
// and so codegen — rounds once. About a quarter of six-digit literals with a
// negative exponent came out one ulp off, so the interpreter and the compiled
// path disagreed silently, and at a threshold that is a whole step:
// round(0.49999999999999997) was 1 interpreted and 0 compiled. The header is
// vendored unmodified (test_exprtk_vendoring.py pins it), so the literal is
// respelled before ExprTk sees it instead.

namespace {

bool exprtk_reads(const std::string &text, double &value) {
    return exprtk::details::string_to_real(text, value);
}

// A spelling of the finite, non-negative double `v` that ExprTk reads as `v`.
std::string exact_spelling(double v) {
    // Its shortest decimal digits as an integer D and an exponent E. ExprTk
    // reads an integer below 2^53 exactly and scales it by one exactly
    // representable power of ten (up to 1e22) in one multiply or divide, which
    // rounds once, to the double nearest D*10^E — strtod's answer, and v,
    // since those digits round-trip. Checked rather than assumed, because a
    // 16- or 17-digit D or a larger exponent falls outside that.
    for (int digits = 1; digits <= 17; ++digits) {
        char buf[40];
        c_snprintf(buf, sizeof(buf), "%.*e", digits - 1, v);
        if (c_strtod(buf, nullptr) != v) {
            continue;
        }
        const char *e = std::strchr(buf, 'e');
        std::string mantissa;
        for (const char *c = buf; c != e; ++c) {
            if (*c != '.') {
                mantissa += *c;
            }
        }
        int exponent = std::atoi(e + 1) - (digits - 1);
        while (mantissa.size() > 1 && mantissa.back() == '0') {
            mantissa.pop_back();
            ++exponent;
        }
        const std::string candidate =
            exponent == 0 ? mantissa : mantissa + "e" + std::to_string(exponent);
        double read = 0.0;
        if (exprtk_reads(candidate, read) && read == v) {
            return candidate;
        }
        break;
    }
    // Exactly: v is M*2^e with M an odd integer below 2^53. Every factor is an
    // integer ExprTk reads exactly, and a multiply or divide by a power of two
    // is exact (each partial result lies between M and v), so the constant
    // ExprTk folds is v itself.
    int exp2 = 0;
    const double frac = std::frexp(v, &exp2);
    auto mant = static_cast<long long>(std::ldexp(frac, 53));
    int scale = exp2 - 53;
    while (mant != 0 && mant % 2 == 0) {
        mant /= 2;
        ++scale;
    }
    std::string out = "(" + std::to_string(mant);
    for (int left = std::abs(scale); left > 0;) {
        const int step = std::min(left, 52);
        out += (scale < 0 ? '/' : '*') + std::to_string(1LL << step);
        left -= step;
    }
    return out + ")";
}

// End of the number token ExprTk's lexer would scan from `i` (scan_number),
// or npos where it would report a malformed number.
std::size_t number_token_end(const std::string &expr, std::size_t i) {
    bool dot = false, exp = false, exp_sign = false, exp_digit = false;
    const auto digit = [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; };
    while (i < expr.size()) {
        const char c = expr[i];
        if (c == '.') {
            if (dot) {
                return std::string::npos;
            }
            dot = true;
        } else if (c == 'e' || c == 'E') {
            if (i + 1 >= expr.size() ||
                (expr[i + 1] != '+' && expr[i + 1] != '-' && !digit(expr[i + 1]))) {
                return std::string::npos;
            }
            exp = true;
        } else if (exp && (c == '+' || c == '-') && !exp_digit) {
            if (exp_sign) {
                return std::string::npos;
            }
            exp_sign = true;
        } else if (exp && digit(c)) {
            exp_digit = true;
        } else if (!digit(c)) {
            break;
        }
        ++i;
    }
    return i;
}

} // namespace

std::string expr_compat::exact_literals(const std::string &expr) {
    const auto letter = [](char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; };
    const auto word = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
    };
    const auto digit = [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; };
    std::string out;
    out.reserve(expr.size());
    std::size_t i = 0;
    while (i < expr.size()) {
        const char c = expr[i];
        if (letter(c) || c == '_') {
            // A symbol, copied whole so its digits are never read as a number.
            // ExprTk's scan_symbol: word characters, and a '.' that another
            // word character follows.
            std::size_t j = i + 1;
            while (j < expr.size() && (word(expr[j]) || (expr[j] == '.' && j + 1 < expr.size() &&
                                                         word(expr[j + 1])))) {
                ++j;
            }
            out.append(expr, i, j - i);
            i = j;
            continue;
        }
        if (digit(c) || (c == '.' && i + 1 < expr.size() && digit(expr[i + 1]))) {
            const std::size_t j = number_token_end(expr, i);
            if (j == std::string::npos) {
                out.append(expr, i, std::string::npos); // ExprTk reports it
                break;
            }
            const std::string token = expr.substr(i, j - i);
            char *end = nullptr;
            const double v = c_strtod(token.c_str(), &end);
            double read = 0.0;
            // Below the normal range ExprTk reads nothing at all (it refuses
            // the exponent), where strtod and the C compiler give the subnormal.
            if (end == token.c_str() + token.size() && std::isfinite(v) &&
                !(exprtk_reads(token, read) && read == v)) {
                out += exact_spelling(v);
            } else {
                out += token;
            }
            i = j;
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

// a == 0 has a closed form (issue #768): M(0, b, z) = 1, so the ratio is
// M(1, b+1, z) = sum_k z^k / (b+1)_k. For b > 0 and z > 0 every term is
// positive, so the sum carries no cancellation, where the fraction runs on at
// a = 0 (no odd numerator vanishes) and is only marginal inside |z| <= 20:
// mratio(0, 1, 20) came out 8.7e-9 off. Not quite good to rounding, since each
// term adds one: against mpmath over b up to 1e4 and z up to 2e4 the worst
// was 2.6e-13. Past the largest term, at k = z - b - 1, the terms fall faster
// than geometrically, so a term below eps of the sum ends it. A sum past the
// largest double is +inf, which is the ratio rounded; the non-finite warner
// reports it as such. The cap is reached only with b past about 5e8 and z
// near b, where the terms fall too slowly (b = 5e8 with z = b + 8.3e5 needs
// 1.06e6 terms, and its ratio is a finite e^704); the NaN then sends the call
// on to the rule above, which refuses a = 0 with 2*z > b.
//
// The inf and NaN tests are comparisons rather than isinf/isnan because the
// generated copy has to spell them that way (see mratio_is_finite), and the two
// copies are kept spelled alike.
static double mratio_a_zero_series(double b, double z) {
    double term = 1.0;
    double sum = 1.0;
    for (int k = 0; k < 1000000; ++k) {
        term *= z / (b + 1.0 + k);
        sum += term;
        if (sum > 1.7976931348623157e308) { // +inf
            return sum;
        }
        if (z < b + 1.0 + k && term <= 1.0e-17 * sum) {
            return sum;
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// ─── Both series summed, where the fraction loses digits (issue #1080) ───────
//
// Inside |z| <= 20 the fraction converges to the ratio, but not to all of it.
// It loses digits in proportion to how large the ratio is, about 1e-14*|R| for
// a positive z. The ratio is large there when a is tiny next to b and b < 1:
// M(a, b, z) stays near 1 while M(a+1, b+1, z) grows like e^z/b, so R reaches
// 1e8 to 1e9 as z approaches 20. mratio(-1e-12, 0.0266641568, 19.3105334) came
// out 2.235481406e8 where the ratio is 2.235481880e8. A tiny b with z near -20
// loses the same way at the other end, where R is about 1e-9. Against mpmath at
// 60 digits, over 20000 triples with |a| in [1e-15, 1e-2], b in [1e-9, 10] and
// z in (-20, 20], 210 were off by more than 1e-8, the worst by 4.4e-7.
//
// Both Kummer functions are summed directly instead, when b > 0 and |alpha| < 1:
//
//   z > 0:  alpha = a,      R = M(alpha+1, b+1, x) / M(alpha, b, x),  x = z
//   z < 0:  alpha = b - a,  R = M(alpha,   b+1, x) / M(alpha, b, x),  x = -z
//
// The second line is Kummer's transformation M(a, b, z) = e^z M(b-a, b, -z)
// applied to both functions, whose e^z factors cancel. In that box the
// numerator's terms all share one sign from the second onward, and so do the
// denominator's: (alpha)_k keeps the sign of alpha once alpha + 1 > 0, and the
// numerator's first parameter is alpha + 1 > 0 when z > 0. So the sums cancel
// only where a Kummer function is itself near zero. That is a pole or a zero of
// the ratio, where the problem is ill-conditioned whatever computes it.
//
// The denominator is summed times b, which keeps its second term alpha*x finite
// however small b is. The fraction has no such care: mratio(0.5, 1e-300, 10)
// came back as 1e-32, its Lentz floor, where the ratio is 2.1e-300.
// Nothing overflows: for |alpha| < 1 and x <= 20 the numerator is below
// (1+x)*e^x, and the denominator's terms after b add up to less than x*e^x.
//
// Measured against the same reference: none of 60000 triples drawn as above is
// off by more than 1e-8 (worst 1.0e-12, beside a pole), and none of 20000 over
// the whole box with b from 1e-300 to 1e8 (worst 2.7e-14). Placed within 1e-15
// to 1e-3 of a pole, where the fraction is off by factors, the value is the
// exact ratio at an `a` 1 to 16 ulps away, 4 or fewer in 1994 of 2193 cases.
// That error is the problem's conditioning, not the method's.
//
// The box keeps the fraction everywhere else, which is where it measured clean
// (0 of 20000 off with |a| in [1e-2, 10]): a negative integer a (BNG's) never
// enters the box, and neither does a z < 0 with b >= 1 + a.
//
// Past z = -20 the fraction's loss goes on, and grows, where a <= 0 and
// b - a < 1 (see mratio_cf_is_trustworthy). That corner is in the box out to
// z = -700. With a <= 0, alpha = b - a is at least b, so every term of both
// series is positive and nothing cancels at all. The bounds above become e^x
// for the numerator and b + alpha*x*e^x for the denominator, under the largest
// double for x <= 700. Past that the asymptotic route answers. There the branch
// it drops is about x*e^(-x)/(b - a) of the one it keeps, so it vouches for
// every b - a above about 1e-290; below that, out to z = -775 at most, the call
// is refused, where the fraction returned its Lentz floor. Against mpmath, none
// of 60000 triples with b in [1e-12, 10], |a| in [1e-15, 10] and z in
// [-2000, 0] is off by more than 1e-8 (worst 4.2e-14; the fraction was off by
// more than 1e-8 in 2605 of 20000, by up to 1.2e-2), and none of 6000 with b
// down to 1e-300 and z out to -1e9 (worst 1.2e-14). It is slower than the
// fraction was: about 7 us at z = -700 for the series and just past it for the
// asymptotic route, against 2 us, and the same 6 us by z = -5000.
//
// The sum stops once a term is below 1e-17 of its series' absolute sum and the
// ratio of successive terms is at most 1/2 for good. Each ratio is below
// 2*x/(k+1), so from k + 1 >= 4*x on the rest of a series is no larger than its
// last term. That is about 100 terms inside |z| <= 20 and 2800 at z = -700.
//
// The NaN test is a comparison, not isnan, for the reason given at
// mratio_is_finite: the generated copy has to spell it so, and the two copies
// are kept alike. A computed denominator of exactly zero gives an infinity,
// which is a pole rounded, and the non-finite warner reports it.
static bool mratio_series(double a, double b, double z, double *out) {
    if (!(b > 0.0) || !mratio_is_finite(b) || z == 0.0) {
        return false;
    }
    const double x = z > 0.0 ? z : -z;
    const double alpha = z > 0.0 ? a : b - a;
    const double shift = z > 0.0 ? 1.0 : 0.0;
    if (!(alpha > -1.0 && alpha < 1.0)) {
        return false;
    }
    if (!(x <= 20.0) && !(z < 0.0 && a <= 0.0 && x <= 700.0)) {
        return false;
    }
    double tn = 1.0; // a term of M(alpha + shift, b + 1, x)
    double sn = 1.0;
    double an = 1.0;
    double td = b; // a term of b * M(alpha, b, x)
    double sd = b;
    double ad = b;
    for (int k = 0; k < 3000; ++k) {
        tn = tn * ((alpha + shift + k) / (b + 1.0 + k)) * (x / (k + 1.0));
        if (k == 0) {
            td = alpha * x;
        } else {
            td = td * ((alpha + k) / (b + k)) * (x / (k + 1.0));
        }
        sn += tn;
        sd += td;
        an += std::abs(tn);
        ad += std::abs(td);
        if (k + 1.0 >= 4.0 * x && std::abs(tn) <= 1.0e-17 * an && std::abs(td) <= 1.0e-17 * ad) {
            // b * sn would overflow for a b near the largest double, and sd / b
            // for a tiny one, so the division is ordered by which side of 1 b is.
            const double value = b < 1.0 ? b * sn / sd : sn / (sd / b);
            if (!(value == value)) { // NaN
                return false;
            }
            *out = value;
            return true;
        }
    }
    return false;
}

double expr_compat::mratio(double a, double b, double z) {
    if (a == 0.0 && b > 0.0 && z > 0.0) {
        const double summed = mratio_a_zero_series(b, z);
        if (summed == summed) { // not NaN
            return summed;
        }
    }
    double series = 0.0;
    if (mratio_series(a, b, z, &series)) {
        return series;
    }
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
            "). The fraction is reliable for b > 0, and then when a is a negative integer, "
            "when a <= 0 and z <= 0 (past z = -20 only with b - a >= 1), when |z| <= 20, or "
            "when b*b >= 64*|z*a| and z is either negative or no more than half of b. BNG "
            "builds a = -min(AT,BT), b = |AT-BT|+1 and z = -1/Keq, which is inside that "
            "region, so a model reaching this is using mratio in its own way. See "
            "lanl/bngsim issues #453, #456, #768 and #1080.");
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
        // C_1 = 1 + 1/tiny stands for infinity, so the step after it is
        // C_2 = 1 exactly. Taking p/C_1 literally added p*1e-32, with
        // p_2 = z*(a-b)/(b*(b+1)), and a b below about 1e-22 makes that
        // anything at all: mratio(-4.12, 2.4e-279, -1.45) returned 1e-32, the
        // floor, where the ratio is 8.5e-280 (issue #1080). While |p_2| is
        // below 5.5e15, 1 + p*1e-32 rounds to 1, so nothing there changes by a
        // bit; for BNG's arguments (b >= 1) passing it takes an equilibrium
        // constant below 1e-16 times the larger count, 1e-13 at a thousand
        // copies, and there it moves within rounding. A p_2 that overflowed
        // keeps the old arithmetic, which makes the NaN it always made there.
        // (Not mratio_is_finite, which stops at 1e308: a p_2 just under the
        // largest double is a ratio near 1e-308, which still wants this step.)
        double C = (j == 2 && std::abs(p) <= 1.7976931348623157e308) ? q : q + p / Csave;
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
        if (muted || std::isfinite(result)) {
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

    // Set while an expression is probed at times of the caller's choosing
    // (evaluate_over_times): a value that is not finite there is not one the
    // run computed, and is neither reported nor remembered as reported.
    bool muted = false;

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
        // BNG's floor(x + 0.5), not std::round (issue #771): the two differ at
        // every negative half, where std::round goes away from zero.
        const double r = expr_compat::rint(dx);
        if (warner) {
            warner->warn_if_nonfinite("rint", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

// round(x): std::round, a half away from zero — the C `round` codegen emits.
// ExprTk's own round is floor(x + 0.5), or ceil(x - 0.5) below zero, and that
// addition can itself round: it gave round(0.49999999999999994) = 1 and
// round(2^52 + 1) = 2^52 + 2 while codegen gave 0 and 2^52 + 1 (issue #684).
// configure_parser() disables the builtin so this one is the `round` parsed.
template <typename T> struct RoundFunction : public exprtk::ifunction<T> {
    RoundFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override { return static_cast<T>(std::round(static_cast<double>(x))); }
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
    //      `ln`/`rint`/`mratio`/`time` as *parameter* names upstream, but
    //      it accepts an *observable* named `time`, so models reaching
    //      bngsim via `generate_network` can hit the `sign` and `time`
    //      collisions in practice. We mangle all five for symmetry and to
    //      handle hand-crafted .net inputs. A mangled `time` stays the
    //      clock in call form (`time()`); see is_zero_arg_builtin (#776).
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
    RoundFunction<double> round_func;
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

    // ExprTk's dependency collector reports constants from the symbol table
    // alongside variables. Keep their registration keys so dependency queries
    // can return only model variables, as the public interface promises.
    std::unordered_set<std::string> constant_symbol_names;

    // Built-in functions that take no arguments. Their call form `name()` means
    // the built-in even when the model declares a scalar of the same name,
    // which is then reachable only bare (issue #776). `time` is the only one.
    static bool is_zero_arg_builtin(const std::string &name) { return name == "time"; }

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

    // Issue #1415: the first call in `expr` that a bracket follows, as the
    // call's text, or "" where there is none. ExprTk multiplies a variable, or
    // a bracketed group, against a bracket after it: `A(B)` and `(A)(B)` are
    // A·B. After a call it does not. The bracket opens a second statement, and
    // the expression's value is the last statement's: `k*exp(-A)(B)` came back
    // B, and a .net rate law written so ran as its last factor. A call is a
    // name no variable or constant is registered under, then its argument
    // list: a built-in, a model function, a table, `time()`, or `if`/`not`.
    // ExprTk's binary word operators are not: `(A > 1) and (B)(C)` is
    // (A > 1) and B·C. BNG2.pl rejects the same text as a syntax error.
    std::string call_before_bracket(const std::string &expr) const {
        const auto ident = [](char c) {
            return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
        };
        const std::size_t n = expr.size();
        std::size_t i = 0;
        while (i < n) {
            const char c = expr[i];
            if (std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '.') {
                // A number, exponent included: `1e5` is not the name `e5`.
                while (i < n &&
                       (std::isdigit(static_cast<unsigned char>(expr[i])) != 0 || expr[i] == '.')) {
                    ++i;
                }
                if (i < n && (expr[i] == 'e' || expr[i] == 'E')) {
                    std::size_t k = i + 1;
                    if (k < n && (expr[k] == '+' || expr[k] == '-')) {
                        ++k;
                    }
                    if (k < n && std::isdigit(static_cast<unsigned char>(expr[k])) != 0) {
                        i = k;
                        while (i < n && std::isdigit(static_cast<unsigned char>(expr[i])) != 0) {
                            ++i;
                        }
                    }
                }
                continue;
            }
            if (!(std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_')) {
                ++i;
                continue;
            }
            const std::size_t start = i;
            while (i < n && ident(expr[i])) {
                ++i;
            }
            const std::string name = expr.substr(start, i - start);
            std::size_t open = i;
            while (open < n && std::isspace(static_cast<unsigned char>(expr[open])) != 0) {
                ++open;
            }
            static const std::unordered_set<std::string> kBinaryWords = {
                "and", "or", "xor", "nand", "nor", "xnor", "mand", "mor", "in", "like", "ilike"};
            if (open >= n || expr[open] != '(' || symbol_table.is_variable(name) ||
                symbol_table.is_constant_node(name) || kBinaryWords.count(name) != 0) {
                continue;
            }
            int depth = 0;
            std::size_t close = open;
            for (; close < n; ++close) {
                if (expr[close] == '(') {
                    ++depth;
                } else if (expr[close] == ')' && --depth == 0) {
                    break;
                }
            }
            if (close >= n) {
                return "";
            }
            std::size_t next = close + 1;
            while (next < n && std::isspace(static_cast<unsigned char>(expr[next])) != 0) {
                ++next;
            }
            if (next < n && (expr[next] == '(' || expr[next] == '[' || expr[next] == '{')) {
                return expr.substr(start, close + 1 - start);
            }
        }
        return "";
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
    //
    // A built-in can end up in scalar_variable_names after all: a model may
    // declare a scalar under a built-in's name, which define_variable registers
    // under the mangled key and records by its source name. BNG2.pl rejects a
    // *parameter* named `time` but accepts an *observable* named `time`, and a
    // hand-written .net can declare either. `time()` must still read the clock
    // then, so a zero-argument built-in call is never stripped (issue #776):
    // stripping it turned the clock into the model scalar, silently.
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
                    scalar_variable_names.count(ident) && !is_zero_arg_builtin(ident)) {
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
                    if (j < expr.size() && expr[j] == '(' && is_zero_arg_builtin(token)) {
                        // `time()` with a model scalar also named `time`: the
                        // call form is the built-in clock, the bare name is the
                        // scalar (r_time). Not ambiguous — BNG gives `time()`
                        // no other meaning — so emit the built-in unmapped
                        // rather than raise (issue #776).
                        result += token;
                        continue;
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
        constant_symbol_names.insert(mapped);
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
        // time_func is intentionally omitted (see TimeFunction comment), and
        // round_func needs none: std::round of a finite value is finite.
        mratio_func.warner = &nonfinite_warner;
        ln_func.warner = &nonfinite_warner;
        rint_func.warner = &nonfinite_warner;
        sign_func.warner = &nonfinite_warner;
        tgamma_func.warner = &nonfinite_warner;

        // Register backward-compatible aliases
        symbol_table.add_function("ln", ln_func);
        symbol_table.add_function("rint", rint_func);
        // A reserved name: ExprTk's builtin is disabled in configure_parser().
        symbol_table.add_reserved_function("round", round_func);
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
            configure_parser(*parser);
        }
        return *parser;
    }

    // Every parser this evaluator uses is configured here, so the one that
    // compiles an expression and the one that later re-reads it to collect its
    // variables (referenced_variable_addresses) accept exactly the same inputs.
    // They did not: the collector was ExprTk's own collect_variables(), which
    // builds a default parser, and an expression deeper than that parser's
    // limit compiled here and came back as reading no variables (issue #779).
    static void configure_parser(Parser &p) {
        // Increase max stack depth for deeply nested if() expressions.
        // ExprTk default is 400 (~200 nested if()), muParser handled 2000.
        p.settings().set_max_stack_depth(4096);
        // `round` resolves to RoundFunction in the symbol table (issue #684).
        p.settings().disable_base_function(Parser::settings_t::e_bf_round);
    }

    void set_time_ptr(double *ptr) { time_func.time_ptr = ptr; }
};

ExprTkEvaluator::ExprTkEvaluator() : impl_(std::make_unique<Impl>()) {}

ExprTkEvaluator::~ExprTkEvaluator() = default;

bool ExprTkEvaluator::try_define_variable(const std::string &name, double *addr) {
    std::string mapped = expr_compat::compute_registration_name(name);
    if (!impl_->symbol_table.add_variable(mapped, *addr))
        return false;
    if (mapped != name && (name.empty() || name[0] != '_')) {
        impl_->mangled_user_names[name] = mapped;
    }
    impl_->scalar_variable_names.insert(name);
    return true;
}

void ExprTkEvaluator::define_variable(const std::string &name, double *addr) {
    if (!try_define_variable(name, addr)) {
        const std::string mapped = expr_compat::compute_registration_name(name);
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
        } else if (impl_->symbol_table.symbol_exists(mapped)) {
            detail = "name '" + name + "' (mapped: '" + mapped +
                     "') is already registered. Check the .net file for duplicate "
                     "parameter / observable / species names (case-sensitive).";
        } else {
            // add_variable gives the same `false` for a name that is no ExprTk
            // symbol as for a duplicate, and a name like `k-1` or `2k` was told
            // to look for a duplicate that is not there. Nothing holds the name,
            // so the name itself was refused.
            detail = "name '" + name + "' (mapped: '" + mapped +
                     "') is not a valid identifier: one starts with a letter or an "
                     "underscore and holds only letters, digits, underscores and "
                     "periods (not as its last character). Rename it.";
        }
        throw std::runtime_error("ExprTk: failed to register variable '" + name + "'. " + detail);
    }
}

void ExprTkEvaluator::define_constant(const std::string &name, double value) {
    impl_->add_remapped_constant(name, value);
}

// symbol_table.add_function() refuses a name that is already registered, or
// that is not a valid ExprTk symbol, and the expression that later reads the
// name binds whatever is there. define_variable throws on the same refusal; a
// function has to as well, or a second table under one name (issue #772) is
// dropped while the model goes on as if it had been registered.
static void throw_function_not_registered(const std::string &name) {
    throw std::runtime_error("ExprTk: failed to register function '" + name +
                             "': the name is already registered or is not a valid "
                             "identifier.");
}

void ExprTkEvaluator::define_function(const std::string &name, Func0 fn) {
    auto adapter = std::make_unique<StdFunc0Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    if (!impl_->symbol_table.add_function(name, *adapter))
        throw_function_not_registered(name);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func1 fn) {
    auto adapter = std::make_unique<StdFunc1Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    if (!impl_->symbol_table.add_function(name, *adapter))
        throw_function_not_registered(name);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func2 fn) {
    auto adapter = std::make_unique<StdFunc2Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    if (!impl_->symbol_table.add_function(name, *adapter))
        throw_function_not_registered(name);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func3 fn) {
    auto adapter = std::make_unique<StdFunc3Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    if (!impl_->symbol_table.add_function(name, *adapter))
        throw_function_not_registered(name);
    impl_->user_functions.push_back(std::move(adapter));
}

// ─── C-style logical operator replacement ────────────────────────────────────
// BNG2.pl emits C-style && and || in if() conditions, but ExprTk uses
// 'and' and 'or' keywords. Replaced in compile(), after the identifier remap
// (issue #770: see the note there).
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
    // Strip `obs()` → `obs` for any name registered as a scalar variable.
    // BNGL accepts Observable references as zero-arg calls; ExprTk does
    // not. Run before remap_expression so we match against BNG-source
    // names, not their post-mangling forms.
    std::string stripped = impl_->strip_empty_parens(expr);

    // Remap identifiers before ExprTk compilation:
    //   - Unconditional: "_X" → "u_X" (ExprTk rejects '_' prefix)
    //   - Conditional:   reserved-word names registered on this evaluator
    //                    (e.g., user's `const` → `r_const`)
    std::string remapped = impl_->remap_expression(stripped);

    // Replace C-style logical operators LAST, after the identifier remap
    // (issue #770). The replacement inserts the ExprTk
    // keywords `and` / `or`, which are identifiers as far as the remap is
    // concerned: run first, a model symbol named `or` or `and` (mangled to
    // `r_or` / `r_and` at registration) captured the operator just inserted,
    // so `(x>2) || on` compiled as the implicit product `(x>2) r_or on` and
    // `(a)||(b)` tripped the "declared symbol used as a function call" guard.
    // `&&` and `||` contain no identifier characters, so the remap passes them
    // through untouched and the keywords inserted here are never rewritten.
    // Every expression without such a symbol compiles to exactly the text it
    // did before.
    std::string preprocessed = replace_logical_operators(remapped);

    // Then respell any literal ExprTk would read off by an ulp, so the
    // interpreter computes with the same numbers codegen does.
    preprocessed = expr_compat::exact_literals(preprocessed);

    // Issue #1415: refused, where ExprTk would keep only the bracket after it.
    const std::string call = impl_->call_before_bracket(preprocessed);
    if (!call.empty()) {
        throw std::runtime_error(
            "Expression '" + expr + "': the bracket after the call '" + call +
            "' is not multiplied by it. ExprTk reads it as a second statement, and the "
            "expression's value would be that statement's alone, which drops the call and "
            "everything before it (issue #1415). BioNetGen rejects it as a syntax error. Write "
            "the product with '*': '" +
            call + "*(...)'.");
    }

    // Delegate to compile_preprocessed (which also caches the string)
    return compile_preprocessed(preprocessed);
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
    //
    // The text is re-parsed against this evaluator's own symbol table (so
    // built-in and user functions such as time() resolve instead of stopping
    // the parse — a trigger like `time() >= t_dose` must still report t_dose)
    // by a parser configured exactly as the one that compiled it. ExprTk's
    // exprtk::collect_variables() cannot be used: it builds a default parser
    // whose stack-depth limit is ~10x lower than ours, and on failure it
    // returns false, which this function used to turn into "reads no
    // variables". Every caller takes an empty result to mean the expression is
    // independent of the model — a fixed-time trigger, a zero jump sensitivity,
    // a constant delay — so an expression past ~197 nested if()s or ~65 nested
    // parentheses silently lost its dt*/dp term or its jump (issue #779).
    // A re-parse that fails now raises instead: an unknown dependency set must
    // not read as an empty one.
    Impl::Parser collector;
    Impl::configure_parser(collector);
    collector.dec().collect_variables() = true;
    Impl::Expression scratch;
    scratch.register_symbol_table(impl_->symbol_table);
    const std::string &text = impl_->preprocessed_strings[expr_id];
    if (!collector.compile(text, scratch)) {
        throw std::runtime_error("ExprTk: could not re-parse expression '" + text +
                                 "' to collect the model variables it reads (" + collector.error() +
                                 "). Its dependencies are unknown, so it cannot be classified "
                                 "for sensitivity analysis.");
    }
    std::deque<Impl::Parser::dependent_entity_collector::symbol_t> symbols;
    collector.dec().symbols(symbols);
    std::unordered_set<const double *> seen;
    out.reserve(symbols.size());
    for (const auto &sym : symbols) {
        if (impl_->constant_symbol_names.count(sym.first) != 0) {
            continue;
        }
        auto *var = impl_->symbol_table.get_variable(sym.first);
        if (var != nullptr && seen.insert(&var->ref()).second) {
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

void ExprTkEvaluator::set_nonfinite_warnings_muted(bool muted) {
    impl_->nonfinite_warner.muted = muted;
}

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
