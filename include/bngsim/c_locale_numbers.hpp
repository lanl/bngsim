#pragma once

// Issue #1235 — numbers read and written in the "C" locale, whatever the
// process's LC_NUMERIC.
//
// strtod, std::stod, atof and snprintf follow the C locale's decimal point. In a
// process that has called setlocale(LC_ALL, "") on a German or French machine,
// "0.5" read as 0 and 0.5 was written as "0,5". Python leaves LC_NUMERIC at "C"
// unless something calls setlocale, and its own float() reads a point
// whatever the locale, so only this side changed: a homodimer's stat factor
// `0.5*kf1` was refused as an unknown parameter, a group factor `0.5*1,2`
// read 0 with no error, and an inline tfun's 2.5 read 2. ExprTk parses its
// own numbers and is not affected.
//
// Each helper takes the same arguments and gives the same answer as the call
// it replaces, read or written in "C". Where a "C" locale cannot be made (it
// always can on the platforms bngsim builds for), they fall back to the
// process's own.

#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <locale.h>
#include <stdexcept>
#include <string>
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <xlocale.h>
#endif

namespace bngsim {

namespace detail {
#if defined(_WIN32)
inline _locale_t c_numeric_locale() {
    static const _locale_t loc = _create_locale(LC_ALL, "C");
    return loc;
}
#else
inline locale_t c_numeric_locale() {
    static const locale_t loc = newlocale(LC_ALL_MASK, "C", static_cast<locale_t>(0));
    return loc;
}
#endif
} // namespace detail

// strtod, read in "C".
inline double c_strtod(const char *s, char **end) {
    const auto loc = detail::c_numeric_locale();
    if (!loc)
        return std::strtod(s, end);
#if defined(_WIN32)
    return _strtod_l(s, end, loc);
#else
    return strtod_l(s, end, loc);
#endif
}

// atof, read in "C".
inline double c_atof(const char *s) { return c_strtod(s, nullptr); }

// std::stod, read in "C": the same leading whitespace, the same count of
// characters read in *pos, std::invalid_argument where nothing converts and
// std::out_of_range where strtod reports ERANGE (as libstdc++ and libc++ do).
inline double c_stod(const std::string &s, std::size_t *pos = nullptr) {
    const char *begin = s.c_str();
    char *end = nullptr;
    const int saved = errno;
    errno = 0;
    const double v = c_strtod(begin, &end);
    const int got = errno;
    errno = saved;
    if (end == begin)
        throw std::invalid_argument("stod");
    if (got == ERANGE)
        throw std::out_of_range("stod");
    if (pos)
        *pos = static_cast<std::size_t>(end - begin);
    return v;
}

// snprintf, written in "C". The locale is switched for this thread only, for
// the length of the call.
inline int c_snprintf(char *buf, std::size_t n, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const auto loc = detail::c_numeric_locale();
    int r;
#if defined(_WIN32)
    if (loc)
        r = _vsnprintf_s_l(buf, n, _TRUNCATE, fmt, loc, ap);
    else
        r = std::vsnprintf(buf, n, fmt, ap);
#else
    if (loc) {
        const locale_t old = uselocale(loc);
        r = std::vsnprintf(buf, n, fmt, ap);
        uselocale(old);
    } else {
        r = std::vsnprintf(buf, n, fmt, ap);
    }
#endif
    va_end(ap);
    return r;
}

} // namespace bngsim
