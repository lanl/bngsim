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

} // namespace rulemonkey::expr
