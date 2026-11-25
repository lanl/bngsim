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
#include <cstdint>
#include <cstdio>  // std::snprintf, std::rename, std::remove
#include <cstdlib> // std::getenv, std::system
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h> // getpid
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
        const char *cc_env = std::getenv("CC");
        std::string cmd = (cc_env && *cc_env) ? cc_env : "cc";
#if defined(__APPLE__)
        if (std::string(machine()) != "unknown")
            cmd += std::string(" -arch ") + machine();
#endif
        cmd += " -O3 -shared -fPIC -o '" + sotmp + "' '" + cpath + "' -lm";
        return cmd;
    }

    /// The cache directory: BNGSIM_CODEGEN_CACHE_DIR when set, as the Python
    /// codegen reads it, else ~/.cache/bngsim/codegen.
    static std::string cache_dir() {
        const char *env = std::getenv("BNGSIM_CODEGEN_CACHE_DIR");
        std::string dir = env ? env : "";
        const auto b = dir.find_first_not_of(" \t\n\r\f\v");
        dir = b == std::string::npos ? ""
                                     : dir.substr(b, dir.find_last_not_of(" \t\n\r\f\v") - b + 1);
        const char *home = std::getenv("HOME");
        const std::string h = home ? home : "/tmp";
        if (dir.empty())
            return h + "/.cache/bngsim/codegen";
        if (dir == "~")
            return h;
        if (dir.rfind("~/", 0) == 0)
            return h + dir.substr(1);
        return dir;
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
            // Process-unique temps so concurrent ensembles don't collide; the
            // finished .so is renamed into place atomically.
            const std::string stem =
                so.substr(0, so.size() - 3) + "." + std::to_string(static_cast<long>(getpid()));
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
