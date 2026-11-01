// bngsim/src/exprtk_dual.hpp — ExprTk on bngsim::Dual (issue #1330)
//
// ExprTk is templated on its number type and dispatches every function it
// evaluates through a tag, number_type<T>::type, to an overload of `*_impl`
// for that tag. A Dual gets its own tag, and an overload of each: the value
// is the double ExprTk's real-type overload computes, written as it writes it,
// and the derivative is that value's along the dual's direction (dual.hpp).
// Every call ExprTk makes with the tag is unqualified, so these are found by
// argument-dependent lookup where the templates are instantiated for Dual.
//
// Included by src/expression.cpp only, after exprtk.hpp, so that the one
// translation unit that compiles ExprTk compiles it for both types.
#pragma once

#include "exprtk_dual_pre.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace exprtk {
namespace details {
namespace numeric {
namespace details {

struct dual_type_tag {
    dual_type_tag() {}
};

template <> struct number_type<bngsim::Dual> {
    typedef dual_type_tag type;
    number_type() {}
};

// ExprTk's real epsilon, for equal() and not_equal().
template <> struct epsilon_type<bngsim::Dual> {
    static inline bngsim::Dual value() { return bngsim::Dual(0.000000000100); }
};

using bngsim::Dual;

inline bool is_nan_impl(const Dual v, dual_type_tag) { return v.v != v.v; }
inline int to_int32_impl(const Dual v, dual_type_tag) { return static_cast<int>(v.v); }
inline _int64_t to_int64_impl(const Dual v, dual_type_tag) { return static_cast<_int64_t>(v.v); }
inline _uint64_t to_uint64_impl(const Dual v, dual_type_tag) { return static_cast<_uint64_t>(v.v); }
inline bool is_integer_impl(const Dual &v, dual_type_tag) { return std::fmod(v.v, 1.0) == 0.0; }

inline Dual abs_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::abs(v); }
// std::min / std::max, as ExprTk's: the first argument unless the second is
// strictly below (above) it; a tie is met in the comparison.
inline Dual min_impl(const Dual v0, const Dual v1, dual_type_tag) { return std::min<Dual>(v0, v1); }
inline Dual max_impl(const Dual v0, const Dual v1, dual_type_tag) { return std::max<Dual>(v0, v1); }

// equal() and not_equal(): equal to within ExprTk's epsilon of the larger
// magnitude (at least 1). Constant on either side of the band's edge, which no
// state is on but by accident, so a derivative of 0 and no tie.
inline bool dual_within_epsilon(const Dual &v0, const Dual &v1) {
    const double eps = 0.000000000100;
    const double a = (v0.v - v1.v) < 0.0 ? -(v0.v - v1.v) : (v0.v - v1.v);
    const double m0 = v0.v < 0.0 ? -v0.v : v0.v;
    const double m1 = v1.v < 0.0 ? -v1.v : v1.v;
    return a <= std::max(1.0, std::max(m0, m1)) * eps;
}
inline Dual equal_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return dual_within_epsilon(v0, v1) ? Dual(1.0) : Dual(0.0);
}
inline Dual nequal_impl(const Dual v0, const Dual v1, dual_type_tag) {
    const double eps = 0.000000000100;
    const double a = (v0.v - v1.v) < 0.0 ? -(v0.v - v1.v) : (v0.v - v1.v);
    const double m0 = v0.v < 0.0 ? -v0.v : v0.v;
    const double m1 = v1.v < 0.0 ? -v1.v : v1.v;
    return (a > std::max(1.0, std::max(m0, m1)) * eps) ? Dual(1.0) : Dual(0.0);
}

inline Dual modulus_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return bngsim::dualmath::fmod(v0, v1);
}
inline Dual pow_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return bngsim::dualmath::pow(v0, v1);
}
inline Dual logn_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return bngsim::dualmath::log(v0) / bngsim::dualmath::log(v1);
}
inline Dual trunc_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::trunc(v); }
inline Dual root_impl(const Dual v0, const Dual v1, dual_type_tag) {
    if (v0 < Dual(0.0)) {
        return (v1 == trunc_impl(v1, dual_type_tag())) &&
                       (modulus_impl(v1, Dual(2.0), dual_type_tag()) != Dual(0.0))
                   ? -bngsim::dualmath::pow(abs_impl(v0, dual_type_tag()), Dual(1.0) / v1)
                   : Dual(std::numeric_limits<double>::quiet_NaN());
    }
    return bngsim::dualmath::pow(v0, Dual(1.0) / v1);
}
inline Dual round_impl(const Dual v, dual_type_tag) {
    return (v < Dual(0.0)) ? bngsim::dualmath::ceil(v - Dual(0.5))
                           : bngsim::dualmath::floor(v + Dual(0.5));
}
inline Dual roundn_impl(const Dual v0, const Dual v1, dual_type_tag) {
    const int index =
        std::max<int>(0, std::min<int>(static_cast<int>(exprtk::details::pow10_size) - 1,
                                       static_cast<int>(std::floor(v1.v))));
    const Dual p10 = Dual(exprtk::details::pow10[index]);
    if (v0 < Dual(0.0)) {
        return bngsim::dualmath::ceil((v0 * p10) - Dual(0.5)) / p10;
    }
    return bngsim::dualmath::floor((v0 * p10) + Dual(0.5)) / p10;
}
inline Dual hypot_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return bngsim::dualmath::sqrt((v0 * v0) + (v1 * v1));
}
inline Dual atan2_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return bngsim::dualmath::atan2(v0, v1);
}
inline Dual shr_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return v0 * (Dual(1.0) / Dual(std::pow(2.0, static_cast<double>(static_cast<int>(v1.v)))));
}
inline Dual shl_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return v0 * Dual(std::pow(2.0, static_cast<double>(static_cast<int>(v1.v))));
}
inline Dual sgn_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::sgn(v); }

// The logical operators read their operands' truth as ExprTk does, through
// operator!= against 0, which meets a tie at an operand of exactly 0 that
// moves.
inline bool dual_true(const Dual &v) { return exprtk::details::is_true(v); }
inline Dual and_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return (dual_true(v0) && dual_true(v1)) ? Dual(1.0) : Dual(0.0);
}
inline Dual nand_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return (!dual_true(v0) || !dual_true(v1)) ? Dual(1.0) : Dual(0.0);
}
inline Dual or_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return (dual_true(v0) || dual_true(v1)) ? Dual(1.0) : Dual(0.0);
}
inline Dual nor_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return (!dual_true(v0) && !dual_true(v1)) ? Dual(1.0) : Dual(0.0);
}
inline Dual xor_impl(const Dual v0, const Dual v1, dual_type_tag) {
    return (!dual_true(v0) != !dual_true(v1)) ? Dual(1.0) : Dual(0.0);
}
inline Dual xnor_impl(const Dual v0, const Dual v1, dual_type_tag) {
    const bool t0 = dual_true(v0);
    const bool t1 = dual_true(v1);
    return ((t0 && t1) || (!t0 && !t1)) ? Dual(1.0) : Dual(0.0);
}

inline Dual erf_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::erf(v); }
inline Dual erfc_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::erfc(v); }
inline Dual ncdf_impl(const Dual v, dual_type_tag) {
    return Dual(0.5) * erfc_impl(-(v / Dual(numeric::constant::sqrt2)), dual_type_tag());
}
inline Dual sinc_impl(const Dual v, dual_type_tag) {
    if (std::abs(v.v) >= std::numeric_limits<double>::epsilon()) {
        return bngsim::dualmath::sin(v) / v;
    }
    return Dual(1.0);
}
inline Dual acosh_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::acosh(v); }
inline Dual asinh_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::asinh(v); }
inline Dual atanh_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::atanh(v); }
inline Dual expm1_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::expm1(v); }
inline Dual log1p_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::log1p(v); }
inline Dual acos_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::acos(v); }
inline Dual asin_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::asin(v); }
inline Dual atan_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::atan(v); }
inline Dual ceil_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::ceil(v); }
inline Dual cos_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::cos(v); }
inline Dual cosh_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::cosh(v); }
inline Dual exp_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::exp(v); }
inline Dual floor_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::floor(v); }
inline Dual log_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::log(v); }
inline Dual log10_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::log10(v); }
inline Dual log2_impl(const Dual v, dual_type_tag) {
    return bngsim::dualmath::log(v) / Dual(numeric::constant::log2);
}
inline Dual neg_impl(const Dual v, dual_type_tag) { return -v; }
inline Dual pos_impl(const Dual v, dual_type_tag) { return +v; }
inline Dual sin_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::sin(v); }
inline Dual sinh_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::sinh(v); }
inline Dual sqrt_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::sqrt(v); }
inline Dual tan_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::tan(v); }
inline Dual tanh_impl(const Dual v, dual_type_tag) { return bngsim::dualmath::tanh(v); }
inline Dual cot_impl(const Dual v, dual_type_tag) { return Dual(1.0) / bngsim::dualmath::tan(v); }
inline Dual sec_impl(const Dual v, dual_type_tag) { return Dual(1.0) / bngsim::dualmath::cos(v); }
inline Dual csc_impl(const Dual v, dual_type_tag) { return Dual(1.0) / bngsim::dualmath::sin(v); }
inline Dual r2d_impl(const Dual v, dual_type_tag) { return v * Dual(numeric::constant::_180_pi); }
inline Dual d2r_impl(const Dual v, dual_type_tag) { return v * Dual(numeric::constant::pi_180); }
inline Dual d2g_impl(const Dual v, dual_type_tag) { return v * Dual(10.0 / 9.0); }
inline Dual g2d_impl(const Dual v, dual_type_tag) { return v * Dual(9.0 / 10.0); }
inline Dual notl_impl(const Dual v, dual_type_tag) { return dual_true(v) ? Dual(0.0) : Dual(1.0); }
inline Dual frac_impl(const Dual v, dual_type_tag) { return v - trunc_impl(v, dual_type_tag()); }

template <typename T> inline T const_pi_impl(dual_type_tag) { return T(numeric::constant::pi); }
template <typename T> inline T const_e_impl(dual_type_tag) { return T(numeric::constant::e); }
template <typename T> inline T const_qnan_impl(dual_type_tag) {
    return T(std::numeric_limits<double>::quiet_NaN());
}

// A literal reads as ExprTk reads it for a double, with no derivative, so that
// a number in a law is the same number on both.
template <typename Iterator>
inline bool string_to_real(Iterator &itr_external, const Iterator end, Dual &t, dual_type_tag) {
    double value = 0.0;
    const bool ok = exprtk::details::string_to_real<Iterator, double>(itr_external, end, value,
                                                                      real_type_tag());
    t = Dual(value);
    return ok;
}

} // namespace details
} // namespace numeric
} // namespace details
} // namespace exprtk
