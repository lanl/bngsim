#pragma once

// Reading a number the way BNG2 writes it, whatever locale the host process
// has set (issue #122).

#include <cerrno>
#include <clocale>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <locale.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <xlocale.h>
#endif

namespace rulemonkey {

// All of `text` as a double, read in the "C" locale.
//
// strtod and std::stod use the decimal point of the C locale the process has
// set, and iostreams that of the C++ global locale.  Under a decimal-comma
// locale (de_DE, fr_FR, ...) stod stops at the `.` of "0.5", returns 0 and
// reports nothing, and an iostream may read the `.` as a thousands separator.
// A program that embeds RuleMonkey can set either.  This reads with a "C"
// locale of its own, so it returns the double strtod returns in the "C"
// locale, bit for bit, and refuses a token that is only partly a number
// rather than keeping the part it could read.  `what` names the value in the
// error message.
inline double parse_c_double(const std::string& text, const std::string& what) {
#ifdef _WIN32
  static auto* const c_locale = _create_locale(LC_NUMERIC, "C");
#else
  static auto* const c_locale = newlocale(LC_NUMERIC_MASK, "C", static_cast<locale_t>(nullptr));
#endif
  if (c_locale == nullptr)
    throw std::runtime_error(what + ": cannot create the C locale to read '" + text + "'");
  const char* const begin = text.c_str();
  // strtod_l's endptr is `char**`, so this cannot be `const char*` however
  // read-only it is here; misc-const-correctness does not model that.
  char* end = nullptr; // NOLINT(misc-const-correctness)
  errno = 0;
#ifdef _WIN32
  double const value = _strtod_l(begin, &end, c_locale);
#else
  double const value = strtod_l(begin, &end, c_locale);
#endif
  if (end == begin || *end != '\0')
    throw std::runtime_error(what + ": '" + text + "' is not a number");
  if (errno == ERANGE)
    throw std::runtime_error(what + ": '" + text + "' is out of the range of a double");
  return value;
}

} // namespace rulemonkey
