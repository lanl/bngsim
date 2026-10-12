// bngsim/include/bngsim/expr_compat.hpp — BNG ↔ ExprTk compatibility primitives
//
// Single source of truth for the BNG-compatibility logic shared by the two
// ExprTk-backed expression layers in BNGsim:
//
//   * the host evaluator  — bngsim::ExprTkEvaluator (src/expression.cpp), used
//     by the ODE/SSA engines and NfsimSimulator::evaluate_expression; and
//   * the vendored NFsim mu::Parser shim (third_party/nfsim/src/NFfunction/
//     nfsim_funcparser.h), which evaluates NFsim function columns in-process.
//
// Both layers must agree on (a) the BNGL built-in mratio(), and (b) how a model
// identifier is mapped onto an ExprTk-legal symbol name (the unconditional
// leading-underscore remap and the conditional reserved-symbol mangle). Before
// issue #49 each feature was hand-ported into both layers and kept in sync by
// guard tests; the shim now forwards to these definitions instead, so the logic
// lives in exactly one place (the bngsim::expression static archive that the
// vendored NFsim target links). See issue #49 and bngsim ADR-005
// (bngsim/dev/adr/ADR-005-nfsim-exprtk-shim-single-source.md).
//
// The declarations here are deliberately free of any ExprTk dependency: the
// reserved-symbol set is built from exprtk.hpp inside the single translation
// unit that defines is_exprtk_reserved(), so neither the host call sites nor
// the NFsim shim need to include the 1.6 MB ExprTk header to call these.
#pragma once

#include <string>

namespace bngsim {
namespace expr_compat {

// mratio(a, b, z): the BNGL built-in confluent-hypergeometric ratio
// M(a+1, b+1, z) / M(a, b, z), evaluated by Gauss's continued fraction with the
// modified-Lentz method so it stays finite on BNG's large-|z| range (issue #42).
double mratio(double a, double b, double z);

// rint(x): BNG's round-to-nearest, floor(x + 0.5) — the definition BNG2.pl's
// run_network evaluates (muParser's Rint). A half rounds up, toward +inf, so
// rint(-2.5) is -2 and rint(-0.5) is 0 (issue #771). This is NOT C's rint
// (half to even under the default rounding mode), std::round (half away from
// zero) or ExprTk's round (away from zero below 0); all four agree except at
// halves and where x +/- 0.5 itself rounds. Written as the literal expression,
// so such an argument (0.49999999999999994 -> 1) gives what run_network gives.
double rint(double x);

// `expr` with every numeric literal ExprTk would misread respelled so that it
// reads as the double strtod gives — what a C compiler gives the generated
// code. ExprTk's own number parser is not correctly rounded: it sums digits in
// a double and scales by powers of ten, so 7.77e-05 came back
// 7.769999999999999e-05 and 0.49999999999999997 came back 0.5, against
// codegen's 7.77e-05 and 0.49999999999999994. A literal ExprTk reads right is
// left as written, and so is anything that is not a literal.
std::string exact_literals(const std::string &expr);

// Unconditional leading-underscore remap: "_X" → "u_X" (ExprTk rejects an
// identifier starting with '_'). Identity for any name not starting with '_'.
// No ExprTk built-in starts with '_', so this rewrite is always safe to apply
// token-by-token to an expression string.
std::string remap_name(const std::string &name);

// True if `name`, registered as an ExprTk variable, would collide with a name
// the symbol table already owns: an ExprTk reserved word/symbol, a bngsim
// function alias (ln/rint/sign/mratio/time), or one of the "u_*" keys the
// built-in constants occupy after the underscore remap (GH #90).
bool is_exprtk_reserved(const std::string &name);

// Symbol-table key for `name` at registration time: the unconditional
// underscore remap first, then the conditional reserved-symbol mangle
// ("X" → "r_X"). Identity if neither applies.
std::string compute_registration_name(const std::string &name);

} // namespace expr_compat
} // namespace bngsim
