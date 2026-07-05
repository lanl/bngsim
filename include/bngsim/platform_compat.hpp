// bngsim/include/bngsim/platform_compat.hpp — POSIX shims for Windows/MSVC (GH #150)
//
// bngsim's simulators lean on a small set of POSIX symbols that MSVC's CRT does
// not expose under their standard spelling: the signed-size type `ssize_t`,
// `getpid()`, and `setenv()`/`unsetenv()`. On Unix these come straight from
// <unistd.h>/<stdlib.h>; on Windows we map them onto the MSVC equivalents so the
// call sites compile unchanged. Include this instead of <unistd.h>.

#pragma once

#ifdef _WIN32

#include <cstddef>   // std::ptrdiff_t
#include <cstdlib>   // std::getenv, _putenv_s
#include <process.h> // _getpid

// POSIX `ssize_t` — the signed counterpart to size_t. MSVC ships `SSIZE_T`
// (<BaseTsd.h>) but not the lowercase POSIX spelling; std::ptrdiff_t is
// layout-compatible and is what the call sites cast to.
using ssize_t = std::ptrdiff_t;

// POSIX `getpid()` is spelled `_getpid()` in <process.h> on Windows. Defining
// the macro after the include leaves <process.h>'s own declarations intact and
// only rewrites bngsim's call sites to the underscored CRT name.
#define getpid _getpid

// POSIX `setenv()`/`unsetenv()` — emulated via the MSVC `_putenv_s()` CRT call.
// Declared at global scope so existing `::setenv` / `::unsetenv` calls resolve.
inline int setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) {
        return 0;
    }
    return _putenv_s(name, value);
}

inline int unsetenv(const char *name) { return _putenv_s(name, ""); }

#else // !_WIN32

#include <cstdlib>  // ::setenv, ::unsetenv (POSIX, declared in <stdlib.h>)
#include <unistd.h> // ::getpid, ssize_t

#endif // _WIN32
