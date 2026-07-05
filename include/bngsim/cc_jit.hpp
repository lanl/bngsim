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
// the .so is cached by a content hash under ~/.cache/bngsim/codegen, so an SSA
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
#include <fstream>
#include <sys/stat.h>
#include <unistd.h> // getpid
#endif

namespace bngsim {

class CcJit {
  public:
    CcJit() = default;

#ifndef _WIN32
    /// Content hash → stable cache path (no compile). Lets a caller invalidate a
    /// cached .so before timing a fresh compile.
    static std::string cache_path(const std::string &src, const std::string &tag = "ssa_prop") {
        uint64_t h = 1469598103934665603ULL; // FNV-1a 64-bit
        for (unsigned char c : src) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        char hex[20];
        std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
        const char *home = std::getenv("HOME");
        std::string dir = std::string(home ? home : "/tmp") + "/.cache/bngsim/codegen";
        return dir + "/" + tag + "_" + hex + ".so";
    }

    /// Compile `src` to a cached .so (cc -O3 -shared -fPIC) and load it. A cache
    /// hit skips the compile and just dlopen's. Throws std::runtime_error on any
    /// write/compile/load failure.
    explicit CcJit(const std::string &src, const std::string &tag = "ssa_prop") {
        const std::string so = cache_path(src, tag);
        struct stat st;
        if (stat(so.c_str(), &st) != 0) {
            const std::string dir = so.substr(0, so.find_last_of('/'));
            (void) std::system(("mkdir -p '" + dir + "'").c_str());
            // Process-unique temps so concurrent ensembles don't collide; the
            // finished .so is renamed into place atomically.
            const std::string stem = so.substr(0, so.size() - 3) + "." +
                                     std::to_string(static_cast<long>(getpid()));
            const std::string cpath = stem + ".c";
            const std::string sotmp = stem + ".so.tmp";
            {
                std::ofstream f(cpath);
                if (!f)
                    throw std::runtime_error("CcJit: cannot write source " + cpath);
                f << src;
            }
            const char *cc_env = std::getenv("CC");
            const std::string cc = cc_env ? cc_env : "cc";
            const std::string cmd =
                cc + " -O3 -shared -fPIC -o '" + sotmp + "' '" + cpath + "' -lm";
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
#endif

    template <typename T> T symbol(const std::string &name) const { return lib_.symbol<T>(name); }

    const std::string &so_path() const { return so_path_; }

  private:
    DynamicLibrary lib_;
    std::string so_path_;
};

} // namespace bngsim
