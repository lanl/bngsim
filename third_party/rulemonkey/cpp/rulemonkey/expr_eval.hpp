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

// Identifiers of a BNGL expression split by whether they are written
// applied to one of `tag_names`.  Each list is de-duplicated and in
// first-seen order; a name written both ways appears in BOTH.
struct TagScopedNames {
  std::vector<std::string> tag_applied; // occurs at least once as `Name(tag)`
  std::vector<std::string> bare;        // occurs at least once in any other form
};

// Classify every identifier in `expr` by tag application: `Name(tag)`,
// with `tag` (trimmed) one of `tag_names`, is tag-applied; every other
// occurrence — bare `Name`, or `Name(something_else)` — is bare.
//
// This is what separates a local function's locally-evaluated observables
// from the global ones it also references (issue #38).  BNG2 emits both
// kinds identically in `<ListOfReferences>` —
//
//     <Reference name="Cnt_Wz"  type="Observable"/>   <!-- local  -->
//     <Reference name="Obs_Src" type="Observable"/>   <!-- global -->
//
// — so the only surviving evidence is the expression itself:
//
//     (kcat*Obs_Src)+(0*Cnt_Wz(x))
//
// with `x` the function's own `<Argument id=>`.  Pass that function's own
// argument names as `tag_names`: in a chain of local functions each callee
// is tagged with its OWN parameter, so the classification is per function.
//
// Like collect_variables the scan is deliberately permissive — called
// functions (`flipUp(x)`), constants and builtins all come back too.
// Callers match the result against a known name set, so the extra tokens
// contribute nothing.
TagScopedNames classify_by_tag_application(const std::string& expr,
                                           const std::vector<std::string>& tag_names);

} // namespace rulemonkey::expr
