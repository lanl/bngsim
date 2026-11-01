// bngsim/include/bngsim/dual.hpp — a dual number, for derivatives that are
// exact where bngsim has no closed form (issue #1330)
//
// A Dual carries a value and its derivative along one direction: v + d·ε with
// ε² = 0. Evaluating a rate law on duals whose seeds are a species' unit vector
// gives that law's value and its partial in the species, to the rounding of the
// value itself and with no step: a difference quotient reads a secant, which
// beside a species at 1e8 stepped a species at 0 by 1.5 (issue #1002), and no
// pair of steps catches every law whose secants agree at both.
//
// The value of every operation is the double the interpreter computes for it,
// operation for operation, so a branch a dual takes is the branch the
// integration took. The derivative is that branch's.
//
// ─── A tie ───────────────────────────────────────────────────────────────────
//
// A state can sit exactly where a law changes branch: `if(X > c, …)` at X = c,
// abs(X) or max(X, 0) at X = 0, floor() at a whole number. There the derivative
// of the branch taken is one of two one-sided derivatives, which differ. Each
// comparison of two duals whose values are equal and whose derivatives are not
// is such a point: dual_context().tied is set, and how it is decided depends on
// dual_context().tie:
//
//   DualTie::Value  as the doubles decide it: the branch the integration took,
//                   and its derivative. abs'(0) is sign(0) = 0, as the closed
//                   form takes it.
//   DualTie::Above  as at the state moved by an infinitesimal step along the
//                   direction (ε > 0): the one-sided derivative from that side,
//                   and the one-sided value of a law that jumps there.
//   DualTie::Below  the same from the other side (ε < 0).
//
// So a caller evaluates with Value, and where that set `tied`, evaluates again
// from Above and from Below: where the answers it builds agree, the derivative
// did not depend on the side, and where they do not, the answer has none.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace bngsim {

enum class DualTie : int { Value = 0, Above = 1, Below = -1 };

struct DualContext {
    DualTie tie = DualTie::Value;
    bool tied = false; // a tie was met since this was last cleared
};

// One per thread: a NetworkModel is evaluated on one thread at a time (every
// fan-out clones it), and the context belongs to the evaluation in progress.
inline DualContext &dual_context() {
    thread_local DualContext context;
    return context;
}

// A product's derivative a'·b + a·b'. A term whose derivative is 0 is left out
// rather than multiplied, so that a value that is not finite times a factor
// that does not move is not NaN: (k·X)' at k = inf, X' = 0 has no derivative to
// give, but k' = 0 and X = 0 is 0·inf in the sum, which the value already is.
inline double dual_product_derivative(double av, double ad, double bv, double bd) {
    double out = 0.0;
    if (ad != 0.0) {
        out += ad * bv;
    }
    if (bd != 0.0) {
        out += av * bd;
    }
    return out;
}

struct Dual;
// How a compares with b, ties included: see the definition below.
inline int dual_order(const Dual &a, const Dual &b);

struct Dual {
    double v = 0.0; // the value
    double d = 0.0; // its derivative along the seeded direction

    constexpr Dual() = default;
    // Implicit, so that a literal or a constant reads as a dual with no
    // derivative, as the evaluator's templates write them (T(0), T(1)).
    constexpr Dual(double value) : v(value) {}
    constexpr Dual(double value, double derivative) : v(value), d(derivative) {}

    explicit constexpr operator double() const { return v; }
    explicit constexpr operator float() const { return static_cast<float>(v); }
    explicit constexpr operator int() const { return static_cast<int>(v); }
    explicit constexpr operator long() const { return static_cast<long>(v); }
    explicit constexpr operator long long() const { return static_cast<long long>(v); }
    explicit constexpr operator unsigned long() const { return static_cast<unsigned long>(v); }
    explicit constexpr operator unsigned long long() const {
        return static_cast<unsigned long long>(v);
    }
    explicit constexpr operator bool() const { return v != 0.0; }

    Dual &operator+=(const Dual &o) {
        v += o.v;
        d += o.d;
        return *this;
    }
    Dual &operator-=(const Dual &o) {
        v -= o.v;
        d -= o.d;
        return *this;
    }
    Dual &operator*=(const Dual &o);
    Dual &operator/=(const Dual &o);
    Dual &operator++() {
        v += 1.0;
        return *this;
    }
    Dual &operator--() {
        v -= 1.0;
        return *this;
    }
    Dual operator++(int) {
        Dual was = *this;
        v += 1.0;
        return was;
    }
    Dual operator--(int) {
        Dual was = *this;
        v -= 1.0;
        return was;
    }

    // Hidden friends, found by argument-dependent lookup only: a declaration
    // at namespace scope would hide the built-in and library overloads every
    // other unqualified call in namespace bngsim reaches.
    friend Dual operator+(const Dual &a) { return a; }
    friend Dual operator-(const Dual &a) { return Dual(-a.v, -a.d); }
    friend Dual operator+(const Dual &a, const Dual &b) { return Dual(a.v + b.v, a.d + b.d); }
    friend Dual operator-(const Dual &a, const Dual &b) { return Dual(a.v - b.v, a.d - b.d); }
    friend Dual operator*(const Dual &a, const Dual &b) {
        return Dual(a.v * b.v, dual_product_derivative(a.v, a.d, b.v, b.d));
    }
    // (a/b)' = (a' − (a/b)·b')/b. Over a divisor that has overflowed, the
    // quotient of a finite value is 0 and so is its derivative: 1/(1 + u) for
    // a u = x^n or exp(x) past the largest double has the derivative
    // −u'/(1 + u)², which is −n/(x·u) or −1/u and goes to 0, where the formula
    // reads ∞/∞.
    friend Dual operator/(const Dual &a, const Dual &b) {
        const double q = a.v / b.v;
        if (b.d == 0.0) {
            return Dual(q, a.d == 0.0 ? 0.0 : a.d / b.v);
        }
        if (std::isinf(b.v) && std::isfinite(a.v) && std::isfinite(a.d)) {
            return Dual(q, 0.0);
        }
        const double num = (a.d == 0.0 ? 0.0 : a.d) - q * b.d;
        return Dual(q, num / b.v);
    }
    friend bool operator<(const Dual &a, const Dual &b) { return dual_order(a, b) == -1; }
    friend bool operator>(const Dual &a, const Dual &b) { return dual_order(a, b) == 1; }
    friend bool operator<=(const Dual &a, const Dual &b) {
        const int o = dual_order(a, b);
        return o == -1 || o == 0;
    }
    friend bool operator>=(const Dual &a, const Dual &b) {
        const int o = dual_order(a, b);
        return o == 1 || o == 0;
    }
    friend bool operator==(const Dual &a, const Dual &b) { return dual_order(a, b) == 0; }
    friend bool operator!=(const Dual &a, const Dual &b) { return dual_order(a, b) != 0; }
};

inline Dual &Dual::operator*=(const Dual &o) { return *this = *this * o; }
inline Dual &Dual::operator/=(const Dual &o) { return *this = *this / o; }

// ─── Comparison, and the tie ─────────────────────────────────────────────────

// -1, 0 or +1 as a is below, level with or above b at the state moved along the
// direction by an infinitesimal step on the side `tie` says; Value decides a
// tie as level. A NaN on either side compares as nothing does (unordered = 2).
inline int dual_order(const Dual &a, const Dual &b) {
    if (a.v < b.v) {
        return -1;
    }
    if (a.v > b.v) {
        return 1;
    }
    if (!(a.v == b.v)) {
        return 2; // unordered: a NaN
    }
    if (a.d == b.d) {
        return 0;
    }
    DualContext &ctx = dual_context();
    ctx.tied = true;
    switch (ctx.tie) {
    case DualTie::Above:
        return a.d < b.d ? -1 : 1;
    case DualTie::Below:
        return a.d < b.d ? 1 : -1;
    case DualTie::Value:
    default:
        return 0;
    }
}

// ─── Functions of a dual ─────────────────────────────────────────────────────
//
// In namespace dualmath, not bngsim, so that they hide nothing: a declaration
// of `floor` or `abs` in namespace bngsim would be what every unqualified call
// there finds first, through a conversion from double, instead of <cmath>'s.

namespace dualmath {

// ─── Steps: a law constant between jumps ─────────────────────────────────────

// The value of a step function f (floor, ceil, …) of `x` with derivative 0, at
// the state, or, on a side `tie` names, just off it: f(x) where f is constant
// near x, and where x sits on a jump with x' ≠ 0, the tie is met and Above
// (Below) reads f just past the jump in the direction x moves (against it).
// `jumps(x)` says whether x is a jump of f; `f_after(x, up)` is f just above
// (up) or just below x.
template <class F, class Jumps, class After>
inline Dual dual_step(const Dual &x, F &&f, Jumps &&jumps, After &&after) {
    const double value = f(x.v);
    if (x.d == 0.0 || !std::isfinite(x.v) || !jumps(x.v)) {
        return Dual(value);
    }
    DualContext &ctx = dual_context();
    ctx.tied = true;
    switch (ctx.tie) {
    case DualTie::Above:
        return Dual(after(x.v, x.d > 0.0));
    case DualTie::Below:
        return Dual(after(x.v, x.d < 0.0));
    case DualTie::Value:
    default:
        return Dual(value);
    }
}

inline bool dual_is_whole(double x) { return std::floor(x) == x; }

inline Dual floor(const Dual &x) {
    return dual_step(
        x, [](double u) { return std::floor(u); }, dual_is_whole,
        [](double u, bool up) { return up ? u : u - 1.0; });
}
inline Dual ceil(const Dual &x) {
    return dual_step(
        x, [](double u) { return std::ceil(u); }, dual_is_whole,
        [](double u, bool up) { return up ? u + 1.0 : u; });
}
inline Dual trunc(const Dual &x) {
    return dual_step(
        x, [](double u) { return std::trunc(u); },
        [](double u) { return dual_is_whole(u) && u != 0.0; },
        [](double u, bool up) {
            // Toward zero: just above a positive whole number it is the number,
            // just below it the one before; the reverse below zero.
            if (u > 0.0) {
                return up ? u : u - 1.0;
            }
            return up ? u + 1.0 : u;
        });
}

// ─── Smooth functions ────────────────────────────────────────────────────────

// f(x) with derivative f'(x)·x'. The chain factor is not formed where x' = 0,
// so a derivative that is not finite at a point no direction moves is not NaN
// (sqrt(0) read as a constant).
inline Dual dual_chain(double value, double fprime, double xd) {
    return Dual(value, xd == 0.0 ? 0.0 : fprime * xd);
}

inline Dual exp(const Dual &x) {
    const double e = std::exp(x.v);
    return dual_chain(e, e, x.d);
}
inline Dual expm1(const Dual &x) { return dual_chain(std::expm1(x.v), std::exp(x.v), x.d); }
inline Dual log(const Dual &x) { return dual_chain(std::log(x.v), 1.0 / x.v, x.d); }
inline Dual log10(const Dual &x) {
    return dual_chain(std::log10(x.v), 1.0 / (x.v * 2.302585092994045684), x.d);
}
inline Dual log1p(const Dual &x) { return dual_chain(std::log1p(x.v), 1.0 / (1.0 + x.v), x.d); }
inline Dual sqrt(const Dual &x) {
    const double r = std::sqrt(x.v);
    return dual_chain(r, 0.5 / r, x.d);
}
inline Dual sin(const Dual &x) { return dual_chain(std::sin(x.v), std::cos(x.v), x.d); }
inline Dual cos(const Dual &x) { return dual_chain(std::cos(x.v), -std::sin(x.v), x.d); }
inline Dual tan(const Dual &x) {
    const double t = std::tan(x.v);
    return dual_chain(t, 1.0 + t * t, x.d);
}
inline Dual asin(const Dual &x) {
    return dual_chain(std::asin(x.v), 1.0 / std::sqrt(1.0 - x.v * x.v), x.d);
}
inline Dual acos(const Dual &x) {
    return dual_chain(std::acos(x.v), -1.0 / std::sqrt(1.0 - x.v * x.v), x.d);
}
inline Dual atan(const Dual &x) { return dual_chain(std::atan(x.v), 1.0 / (1.0 + x.v * x.v), x.d); }
inline Dual sinh(const Dual &x) { return dual_chain(std::sinh(x.v), std::cosh(x.v), x.d); }
inline Dual cosh(const Dual &x) { return dual_chain(std::cosh(x.v), std::sinh(x.v), x.d); }
inline Dual tanh(const Dual &x) {
    const double t = std::tanh(x.v);
    return dual_chain(t, 1.0 - t * t, x.d);
}
inline Dual asinh(const Dual &x) {
    return dual_chain(std::asinh(x.v), 1.0 / std::sqrt(x.v * x.v + 1.0), x.d);
}
inline Dual acosh(const Dual &x) {
    return dual_chain(std::acosh(x.v), 1.0 / std::sqrt(x.v * x.v - 1.0), x.d);
}
inline Dual atanh(const Dual &x) {
    return dual_chain(std::atanh(x.v), 1.0 / (1.0 - x.v * x.v), x.d);
}
inline Dual erf(const Dual &x) {
    // d/dx erf(x) = 2/√π·e^(−x²)
    return dual_chain(std::erf(x.v), 1.12837916709551257390 * std::exp(-x.v * x.v), x.d);
}
inline Dual erfc(const Dual &x) {
    return dual_chain(std::erfc(x.v), -1.12837916709551257390 * std::exp(-x.v * x.v), x.d);
}
// ψ(x), the digamma function, for Γ'(x) = Γ(x)·ψ(x): the recurrence
// ψ(x) = ψ(x + 1) − 1/x up to x ≥ 10, then the asymptotic series, which is
// right to rounding there (its next term is 691/(32760·x¹²)); reflection
// below 0.
inline double dual_digamma(double x) {
    constexpr double pi = 3.14159265358979323846;
    if (x <= 0.0 && std::floor(x) == x) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x < 0.0) {
        return dual_digamma(1.0 - x) - pi / std::tan(pi * x);
    }
    double shift = 0.0;
    while (x < 10.0) {
        shift -= 1.0 / x;
        x += 1.0;
    }
    const double r = 1.0 / x;
    const double r2 = r * r;
    return shift + std::log(x) - 0.5 * r -
           r2 * (1.0 / 12 - r2 * (1.0 / 120 - r2 * (1.0 / 252 - r2 * (1.0 / 240 - r2 / 132))));
}
inline Dual tgamma(const Dual &x) {
    const double g = std::tgamma(x.v);
    return Dual(g, x.d == 0.0 ? 0.0 : g * dual_digamma(x.v) * x.d);
}

// |x|, as (x < 0) ? −x : x: the value is the interpreter's to the sign of a
// zero, and at x = 0 with x' ≠ 0 the tie is met. Value takes sign(0) = 0 for the
// derivative there, as the closed form does (issue #507); Above and Below take
// the one-sided derivative, |x'| and −|x'|.
inline Dual abs(const Dual &x) {
    const double value = (x.v < 0.0) ? -x.v : x.v;
    if (x.v > 0.0) {
        return Dual(value, x.d);
    }
    if (x.v < 0.0) {
        return Dual(value, -x.d);
    }
    if (x.d == 0.0 || !(x.v == 0.0)) {
        return Dual(value, x.v == 0.0 ? 0.0 : x.d); // NaN stays NaN
    }
    DualContext &ctx = dual_context();
    ctx.tied = true;
    switch (ctx.tie) {
    case DualTie::Above:
        return Dual(value, std::abs(x.d));
    case DualTie::Below:
        return Dual(value, -std::abs(x.d));
    case DualTie::Value:
    default:
        return Dual(value, 0.0);
    }
}

// sign(x): 1, 0 or −1, constant but at 0, where it jumps.
inline Dual sgn(const Dual &x) {
    return dual_step(
        x, [](double u) { return u > 0.0 ? 1.0 : (u < 0.0 ? -1.0 : 0.0); },
        [](double u) { return u == 0.0; }, [](double, bool up) { return up ? 1.0 : -1.0; });
}

// a^b. The derivative is b·a^(b−1)·a' + a^b·ln(a)·b', each term formed only
// where its seed is not 0: a power with a constant exponent, at a base of 0,
// has the derivative of b·a^(b−1)·a' alone (0 at b > 1, a' at b = 1, not finite
// below), and ln(0) does not enter it. That is the closed form's account of a
// zero base too (issue #541). std::pow(a, b) is the value, as the interpreter's.
inline Dual pow(const Dual &a, const Dual &b) {
    const double value = std::pow(a.v, b.v);
    double d = 0.0;
    if (a.d != 0.0) {
        // b·a^(b−1), with a^(b−1) from pow so that a = 0 at b = 1 gives 1.
        d += b.v * std::pow(a.v, b.v - 1.0) * a.d;
    }
    if (b.d != 0.0) {
        d += value * std::log(a.v) * b.d;
    }
    return Dual(value, d);
}

// fmod(a, b) = a − trunc(a/b)·b: slope 1 in a and −trunc(a/b) in b, jumping
// where a/b is a nonzero whole number. Off a jump the value is std::fmod's, as
// the interpreter's; just past one, on a side, the side's.
inline Dual fmod(const Dual &a, const Dual &b) {
    const Dual q = trunc(a / b);
    if (q.v == std::trunc(a.v / b.v)) {
        return Dual(std::fmod(a.v, b.v), a.d - q.v * b.d);
    }
    return Dual(a.v - q.v * b.v, a.d - q.v * b.d);
}

inline Dual atan2(const Dual &y, const Dual &x) {
    const double r2 = x.v * x.v + y.v * y.v;
    double d = 0.0;
    if (y.d != 0.0) {
        d += x.v / r2 * y.d;
    }
    if (x.d != 0.0) {
        d -= y.v / r2 * x.d;
    }
    return Dual(std::atan2(y.v, x.v), d);
}

inline Dual hypot(const Dual &a, const Dual &b) { return sqrt(a * a + b * b); }

} // namespace dualmath
} // namespace bngsim

namespace std {
// The evaluator asks numeric_limits of its number type (a NaN, an infinity, an
// epsilon); a dual's are those of its value, with no derivative.
template <> class numeric_limits<bngsim::Dual> : public numeric_limits<double> {
  public:
    static constexpr bngsim::Dual min() noexcept { return numeric_limits<double>::min(); }
    static constexpr bngsim::Dual max() noexcept { return numeric_limits<double>::max(); }
    static constexpr bngsim::Dual lowest() noexcept { return numeric_limits<double>::lowest(); }
    static constexpr bngsim::Dual epsilon() noexcept { return numeric_limits<double>::epsilon(); }
    static constexpr bngsim::Dual round_error() noexcept {
        return numeric_limits<double>::round_error();
    }
    static constexpr bngsim::Dual infinity() noexcept { return numeric_limits<double>::infinity(); }
    static constexpr bngsim::Dual quiet_NaN() noexcept {
        return numeric_limits<double>::quiet_NaN();
    }
    static constexpr bngsim::Dual signaling_NaN() noexcept {
        return numeric_limits<double>::signaling_NaN();
    }
    static constexpr bngsim::Dual denorm_min() noexcept {
        return numeric_limits<double>::denorm_min();
    }
};
} // namespace std
