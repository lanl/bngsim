#pragma once

#include <string>
#include <vector>

// BNGL math-expression support for RuleMonkey.
//
// As of issue #6 the actual parsing + evaluation of rate-law / function /
// parameter expressions is done by `bngsim::ExprTkEvaluator` (vendored
// ExprTk wrapper — see third_party/bngsim_expr/ and
// dev/exprtk_swap_plan_2026_05_16.md).  The hand-rolled recursive-descent
// parser + tree-walking evaluator that used to live here is gone.
//
// The one piece RM still needs in its own namespace is a lightweight
// dependency scanner: ExprTk compiles an expression against a fixed
// symbol table, but RM has to know which parameters / observables /
// functions an expression references *before* it can build that table
// and decide function settle order / rate-dependent-observable marking.

namespace rulemonkey::expr {

// Return the identifier tokens referenced by a BNGL math expression, in
// first-seen order, de-duplicated.
//
// Used only for dependency ordering — never for evaluation.  The scan is
// deliberately permissive: it returns every maximal
// `[A-Za-z_][A-Za-z0-9_]*` run that is not part of a numeric literal.
// Builtin / function names (`sin`, `exp`, `if`, `time`, …) and
// local-function arguments may appear in the result; that is harmless,
// because callers only ever *match* these tokens against known
// parameter / observable / function names — an unmatched token
// contributes nothing to the dependency graph.  The contract that
// matters is the converse: every genuine parameter / observable /
// function reference IS returned.
std::vector<std::string> collect_variables(const std::string& expr);

} // namespace rulemonkey::expr
