// bngsim/include/bngsim/cc_jit.hpp — system-cc compile-to-.so, MirJit surface
//
// GH #190 experiment. Compiles a generated C source string (e.g. the SSA
// value-specialized propensity vector from NetworkModel::emit_ssa_propensity_
// source) to a shared object via the system C compiler — `cc -O3 -shared -fPIC`
// + dlopen, exactly the path bngsim._codegen already uses for the ODE RHS — and
// exposes MirJit's surface (construct from source, then `.symbol<T>(name)`), so
// the SSA loop swaps backends with a one-line change.
//
// Unlike MIR's in-process ~1 ms JIT, cc pays the compiler's ~100 ms ONCE, but
// the .so is cached by a content hash under the codegen cache dir, so an SSA
// ENSEMBLE (many replicates of one model — the normal stochastic workflow)
// compiles only on the first replicate and dlopen's the cached .so thereafter,
// while cc's -O3 code can beat MIR's -O2. The point of the experiment: decide
// whether cc (already the shipped codegen backend, no MIR needed) carries the
// SSA recompute-all win — see issue-190 / issue-142-146.
//
// POSIX-only (uses cc + std::system). On Windows it compiles to a throwing stub
// so translation units that include it still build (mirrors MirJit's stub).

#pragma once

#include "bngsim/dynamic_library.hpp"

#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <atomic>
#include <cstdint>
#include <cstdio>  // std::snprintf, std::rename, std::remove
#include <cstdlib> // std::getenv, std::system
#include <filesystem>
#include <fstream>
#include <pwd.h> // getpwnam, getpwuid
#include <sys/stat.h>
#include <system_error>
#include <unistd.h> // getpid, getuid
#endif

namespace bngsim {

class CcJit {
  public:
    CcJit() = default;

#ifndef _WIN32
    /// Bumped whenever the compile command or the library's contract changes, so
    /// a library an older bngsim built is not reused (issue #1927).
    static constexpr const char *kFormat = "v2";

    /// The machine this process runs as, named as Python's platform.machine()
    /// and the Python codegen's arch field spell it (issue #1927). Read from the
    /// compiler's target macros, so it is the process's own architecture: exact
    /// under Rosetta, where Apple's cc would otherwise build for the hardware.
    static const char *machine() {
#if defined(__x86_64__)
        return "x86_64";
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
        return "arm64";
#elif defined(__aarch64__)
        return "aarch64";
#else
        return "unknown";
#endif
    }

    /// The compile command for `cpath` → `sotmp`. On macOS it passes -arch with
    /// this process's machine (issue #1927), as the Python codegen does since
    /// #1759; a CC naming another -arch then builds a universal library.
    static std::string compile_command(const std::string &cpath, const std::string &sotmp) {
        // A CC of only whitespace is unset, as the Python codegen reads it.
        const char *cc_env = std::getenv("CC");
        std::string cmd = trim(cc_env ? cc_env : "");
        if (cmd.empty())
            cmd = "cc";
#if defined(__APPLE__)
        if (std::string(machine()) != "unknown")
            cmd += std::string(" -arch ") + machine();
#endif
        cmd += " -O3 -shared -fPIC -o " + shell_quote(sotmp) + " " + shell_quote(cpath) + " -lm";
        return cmd;
    }

    /// `s` as one POSIX shell word: single-quoted, with each ' written '\''.
    static std::string shell_quote(const std::string &s) {
        std::string out = "'";
        for (char c : s) {
            if (c == '\'')
                out += "'\\''";
            else
                out += c;
        }
        return out + "'";
    }

    static std::string trim(const std::string &s) {
        const char *ws = " \t\n\r\f\v";
        const auto b = s.find_first_not_of(ws);
        return b == std::string::npos ? "" : s.substr(b, s.find_last_not_of(ws) - b + 1);
    }

    /// The home directory: HOME, else the password database, as Python's
    /// Path.home() reads it.
    static std::string home_dir() {
        const char *home = std::getenv("HOME");
        if (home && *home)
            return home;
        if (const passwd *pw = getpwuid(getuid()); pw && pw->pw_dir)
            return pw->pw_dir;
        return "/tmp";
    }

    /// A leading ~ or ~user expanded as Python's Path.expanduser() does; a user
    /// the password database does not know leaves the path as it is.
    static std::string expand_user(const std::string &path) {
        if (path.empty() || path[0] != '~')
            return path;
        const auto slash = path.find('/');
        const std::string user =
            path.substr(1, slash == std::string::npos ? std::string::npos : slash - 1);
        const std::string rest = slash == std::string::npos ? "" : path.substr(slash);
        if (user.empty())
            return home_dir() + rest;
        if (const passwd *pw = getpwnam(user.c_str()); pw && pw->pw_dir)
            return std::string(pw->pw_dir) + rest;
        return path;
    }

    /// The cache directory: BNGSIM_CODEGEN_CACHE_DIR when set (trimmed, with a
    /// leading ~ expanded), as the Python codegen reads it, else
    /// ~/.cache/bngsim/codegen.
    static std::string cache_dir() {
        const char *env = std::getenv("BNGSIM_CODEGEN_CACHE_DIR");
        const std::string dir = trim(env ? env : "");
        if (dir.empty())
            return home_dir() + "/.cache/bngsim/codegen";
        return expand_user(dir);
    }

    /// Stable cache path (no compile). Lets a caller invalidate a cached .so
    /// before timing a fresh compile. The name carries the machine and kFormat,
    /// and the hash covers the source and the compile command (with CC), so two
    /// architectures sharing a cache, or two compilers, never name one library
    /// alike (issue #1927).
    static std::string cache_path(const std::string &src, const std::string &tag = "ssa_prop") {
        uint64_t h = 1469598103934665603ULL; // FNV-1a 64-bit
        auto mix = [&h](const std::string &s) {
            for (unsigned char c : s) {
                h ^= c;
                h *= 1099511628211ULL;
            }
            h ^= 0xffU; // separator, so ("ab","c") and ("a","bc") differ
            h *= 1099511628211ULL;
        };
        mix(src);
        mix(compile_command("", ""));
        char hex[20];
        std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
        return cache_dir() + "/" + tag + "_" + machine() + "_" + kFormat + "_" + hex + ".so";
    }

    /// Compile `src` to a cached .so (cc -O3 -shared -fPIC) and load it. A cache
    /// hit skips the compile and just dlopen's. Throws std::runtime_error on any
    /// write/compile/load failure.
    explicit CcJit(const std::string &src, const std::string &tag = "ssa_prop") {
        const std::string so = cache_path(src, tag);
        struct stat st;
        if (stat(so.c_str(), &st) != 0) {
            // A failed mkdir surfaces as the failed write below, which says
            // which file.
            std::error_code ec;
            std::filesystem::create_directories(so.substr(0, so.find_last_of('/')), ec);
            // Temps unique to this process and this compile, so concurrent
            // ensembles, and the threads of one run_replicates, don't collide
            // (one thread removed the source another was compiling); the
            // finished .so is renamed into place atomically.
            static std::atomic<unsigned long> counter{0};
            const std::string stem = so.substr(0, so.size() - 3) + "." +
                                     std::to_string(static_cast<long>(getpid())) + "_" +
                                     std::to_string(counter.fetch_add(1));
            const std::string cpath = stem + ".c";
            const std::string sotmp = stem + ".so.tmp";
            {
                std::ofstream f(cpath);
                if (!f)
                    throw std::runtime_error("CcJit: cannot write source " + cpath);
                f << src;
            }
            const std::string cmd = compile_command(cpath, sotmp);
            const int rc = std::system(cmd.c_str());
            std::remove(cpath.c_str());
            if (rc != 0) {
                std::remove(sotmp.c_str());
                throw std::runtime_error("CcJit: compile failed: " + cmd);
            }
            if (std::rename(sotmp.c_str(), so.c_str()) != 0) {
                // A concurrent ensemble may have installed it first — accept that.
                if (stat(so.c_str(), &st) != 0) {
                    std::remove(sotmp.c_str());
                    throw std::runtime_error("CcJit: install (rename) failed for " + so);
                }
                std::remove(sotmp.c_str());
            }
        }
        lib_ = DynamicLibrary(so);
        so_path_ = so;
    }
#else
    explicit CcJit(const std::string &, const std::string & = "ssa_prop") {
        throw std::runtime_error("CcJit: system-cc backend is POSIX-only (not built on Windows)");
    }
    static std::string cache_path(const std::string &, const std::string & = "ssa_prop") {
        return "";
    }
    static const char *machine() { return "unknown"; }
#endif

    template <typename T> T symbol(const std::string &name) const { return lib_.symbol<T>(name); }

    const std::string &so_path() const { return so_path_; }

  private:
    DynamicLibrary lib_;
    std::string so_path_;
};

} // namespace bngsim
