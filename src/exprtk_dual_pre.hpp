// bngsim/src/exprtk_dual_pre.hpp — what ExprTk on bngsim::Dual needs declared
// before exprtk.hpp (issue #1330)
//
// ExprTk calls a few functions by qualified name (`details::is_true(x)` in its
// logical and conditional nodes), which finds only the overloads declared
// where the calling template is defined. The truth of a dual goes here, ahead
// of exprtk.hpp; everything ExprTk finds by argument-dependent lookup is in
// exprtk_dual.hpp, after it. src/expression.cpp includes the three in order.
#pragma once

#include "bngsim/dual.hpp"

namespace exprtk {
namespace details {
// The truth of a condition, as ExprTk reads a double's: not equal to 0, which
// meets a tie at a condition of exactly 0 that moves.
inline bool is_true(const bngsim::Dual &v) { return bngsim::Dual(0.0) != v; }
inline bool is_false(const bngsim::Dual &v) { return bngsim::Dual(0.0) == v; }
} // namespace details
} // namespace exprtk
