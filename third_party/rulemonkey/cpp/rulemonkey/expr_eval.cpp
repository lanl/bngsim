#include "expr_eval.hpp"

#include <algorithm>
#include <cctype>

namespace rulemonkey::expr {

std::vector<std::string> collect_variables(const std::string& expr) {
  std::vector<std::string> out;
  const size_t n = expr.size();
  size_t i = 0;
  while (i < n) {
    auto const c = static_cast<unsigned char>(expr[i]);

    // Numeric literal: a run of digits / '.', plus an optional exponent
    // (`e`/`E` followed by an optional sign and at least one digit).
    // Consumed whole so the exponent marker is never mistaken for an
    // identifier — `1e-5` must NOT yield a spurious variable named `e`.
    if (std::isdigit(c) ||
        (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(expr[i + 1])))) {
      ++i;
      while (i < n && (std::isdigit(static_cast<unsigned char>(expr[i])) || expr[i] == '.'))
        ++i;
      if (i < n && (expr[i] == 'e' || expr[i] == 'E')) {
        size_t j = i + 1;
        if (j < n && (expr[j] == '+' || expr[j] == '-'))
          ++j;
        if (j < n && std::isdigit(static_cast<unsigned char>(expr[j]))) {
          i = j;
          while (i < n && std::isdigit(static_cast<unsigned char>(expr[i])))
            ++i;
        }
      }
      continue;
    }

    // Identifier: [A-Za-z_][A-Za-z0-9_]*
    if (std::isalpha(c) || c == '_') {
      size_t const start = i;
      ++i;
      while (i < n && (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_'))
        ++i;
      std::string ident = expr.substr(start, i - start);
      if (std::find(out.begin(), out.end(), ident) == out.end())
        out.push_back(std::move(ident));
      continue;
    }

    ++i; // operator / paren / whitespace — skip
  }
  return out;
}

TagScopedNames classify_by_tag_application(const std::string& expr,
                                           const std::vector<std::string>& tag_names) {
  TagScopedNames out;
  auto record = [](std::vector<std::string>& bucket, const std::string& name) {
    if (std::find(bucket.begin(), bucket.end(), name) == bucket.end())
      bucket.push_back(name);
  };

  const size_t n = expr.size();
  size_t i = 0;
  while (i < n) {
    auto const c = static_cast<unsigned char>(expr[i]);

    // Identifier: [A-Za-z_][A-Za-z0-9_]*.  Anything else is skipped a
    // character at a time, so unlike collect_variables this scan does not
    // consume numeric literals whole and `1e-5` contributes a spurious
    // bare `e`.  Harmless: callers only test membership of names they
    // already know, and no exponent fragment can be followed by `(tag)`,
    // so nothing spurious can reach the tag_applied bucket.
    if (std::isalpha(c) == 0 && c != '_') {
      ++i;
      continue;
    }
    size_t const start = i;
    ++i;
    while (i < n && (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_'))
      ++i;
    std::string const ident = expr.substr(start, i - start);

    // Applied form?  `Name (`, allowing whitespace before the paren.
    // Anything else is a bare occurrence.
    size_t j = i;
    while (j < n && std::isspace(static_cast<unsigned char>(expr[j])))
      ++j;
    if (j >= n || expr[j] != '(') {
      record(out.bare, ident);
      continue;
    }

    // Take the balanced argument list and test it against the tags.
    // Resuming the outer scan from just after the identifier, rather than
    // past the argument list, is deliberate: a nested `f(g(x))` must
    // still see `g` applied to `x`.
    size_t const open = j;
    int depth = 0;
    size_t close = std::string::npos;
    for (; j < n; ++j) {
      if (expr[j] == '(') {
        ++depth;
      } else if (expr[j] == ')') {
        if (--depth == 0) {
          close = j;
          break;
        }
      }
    }
    if (close == std::string::npos) {
      // Unbalanced — leave the diagnosis to the ExprTk compile.
      record(out.bare, ident);
      continue;
    }

    std::string arg = expr.substr(open + 1, close - open - 1);
    auto const first = arg.find_first_not_of(" \t\n\r");
    auto const last = arg.find_last_not_of(" \t\n\r");
    if (first != std::string::npos)
      arg = arg.substr(first, last - first + 1);
    else
      arg.clear();

    record(std::find(tag_names.begin(), tag_names.end(), arg) != tag_names.end() ? out.tag_applied
                                                                                 : out.bare,
           ident);
  }
  return out;
}

} // namespace rulemonkey::expr
