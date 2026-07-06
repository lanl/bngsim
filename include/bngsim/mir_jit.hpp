// bngsim/include/bngsim/mir_jit.hpp — In-process micro-JIT for the codegen RHS
//
// GH #78 (prototype). Compiles the C source bngsim._codegen emits *in-process*
// via MIR's c2mir C11 frontend + MIR_gen, instead of shelling out to `cc -O3`
// and dlopen'ing the resulting .so. The JIT'd RHS is native-quality (~MIR -O2)
// and compiles in ~1-2 ms instead of ~80 ms-seconds.
//
// MirJit deliberately mirrors DynamicLibrary's surface — construct from input,
// then `.symbol<T>(name)` / `.try_symbol<T>(name)` — so the codegen RHS dispatch
// in cvode_simulator.cpp switches backends with almost no change.
//
// All MIR-specific code is guarded by BNGSIM_HAS_MIR (set by the CMake build
// only when BNGSIM_ENABLE_MIR is ON). When MIR is not built, MirJit still
// compiles but throws on construction, so the codegen dispatch compiles
// unconditionally and simply never takes the JIT branch.

#pragma once

#include <stdexcept>
#include <string>

#if defined(BNGSIM_HAS_MIR)
#include <cstddef>
#include <cstdio>  // open_memstream/tmpfile, FILE, fflush, fclose, stderr
#include <cstdlib> // free
#include <cstring> // std::memset, std::memcpy, std::strcmp
#include <cmath>   // libm addresses for the Windows import table (GH #3)
#ifndef _WIN32
#include <dlfcn.h> // dlsym(RTLD_DEFAULT) — POSIX import resolver
#endif
extern "C" {
#include "c2mir.h"
#include "mir-gen.h"
#include "mir.h"
}
#endif

namespace bngsim {

#if defined(BNGSIM_HAS_MIR)
namespace detail {

// Import resolver for MIR_link: maps an external symbol name (libc/libm
// functions the generated RHS calls — pow, exp, sqrt, memset, ...) to its
// runtime address. Must have C linkage / no captures.
//
// POSIX: RTLD_DEFAULT searches every globally-loaded symbol, which includes
// libSystem/libm already resident in this process, so dlsym resolves the whole
// libm/libc surface the codegen can emit.
//
// Windows/MSVC (GH #3): there is no dlsym/RTLD_DEFAULT. We resolve against a
// static table of exactly the libm/libc functions the codegen can emit — which
// is complete by construction: the JIT source can only *call* functions that
// jit_prelude() forward-declares, so the set of MIR import symbols is a subset
// of the prelude set. Taking &std::pow (etc.) yields a valid out-of-line
// address on every ABI, so the table needs no OS symbol lookup at all. Keep it
// in step with jit_prelude() below.
#ifdef _WIN32
inline void *mir_import_resolver(const char *name) {
    struct Entry {
        const char *name;
        void *addr;
    };
    // Cast through the exact C signature to pick the right <cmath> overload.
#define BNGSIM_MIR_D1(fn)                                                                          \
    {#fn, reinterpret_cast<void *>(static_cast<double (*)(double)>(&std::fn))}
#define BNGSIM_MIR_D2(fn)                                                                          \
    {#fn, reinterpret_cast<void *>(static_cast<double (*)(double, double)>(&std::fn))}
    static const Entry table[] = {
        BNGSIM_MIR_D2(pow),      BNGSIM_MIR_D1(exp),      BNGSIM_MIR_D1(exp2),
        BNGSIM_MIR_D1(expm1),    BNGSIM_MIR_D1(log),      BNGSIM_MIR_D1(log2),
        BNGSIM_MIR_D1(log10),    BNGSIM_MIR_D1(log1p),    BNGSIM_MIR_D1(logb),
        BNGSIM_MIR_D1(sqrt),     BNGSIM_MIR_D1(cbrt),     BNGSIM_MIR_D2(hypot),
        BNGSIM_MIR_D1(fabs),     BNGSIM_MIR_D2(fmax),     BNGSIM_MIR_D2(fmin),
        BNGSIM_MIR_D2(fmod),     BNGSIM_MIR_D2(copysign), BNGSIM_MIR_D1(floor),
        BNGSIM_MIR_D1(ceil),     BNGSIM_MIR_D1(round),    BNGSIM_MIR_D1(trunc),
        BNGSIM_MIR_D1(rint),     BNGSIM_MIR_D1(nearbyint), BNGSIM_MIR_D1(sin),
        BNGSIM_MIR_D1(cos),      BNGSIM_MIR_D1(tan),      BNGSIM_MIR_D1(asin),
        BNGSIM_MIR_D1(acos),     BNGSIM_MIR_D1(atan),     BNGSIM_MIR_D2(atan2),
        BNGSIM_MIR_D1(sinh),     BNGSIM_MIR_D1(cosh),     BNGSIM_MIR_D1(tanh),
        BNGSIM_MIR_D1(asinh),    BNGSIM_MIR_D1(acosh),    BNGSIM_MIR_D1(atanh),
        BNGSIM_MIR_D1(erf),      BNGSIM_MIR_D1(erfc),     BNGSIM_MIR_D1(tgamma),
        BNGSIM_MIR_D1(lgamma),
        {"memset",
         reinterpret_cast<void *>(static_cast<void *(*)(void *, int, std::size_t)>(&std::memset))},
        {"memcpy", reinterpret_cast<void *>(
                       static_cast<void *(*)(void *, const void *, std::size_t)>(&std::memcpy))},
    };
#undef BNGSIM_MIR_D1
#undef BNGSIM_MIR_D2
    for (const auto &e : table) {
        if (std::strcmp(e.name, name) == 0)
            return e.addr;
    }
    return nullptr;
}
#else
inline void *mir_import_resolver(const char *name) { return dlsym(RTLD_DEFAULT, name); }
#endif

// Portable capture of c2mir's diagnostic stream into a std::string. POSIX has
// open_memstream (a growable in-memory FILE*); MSVC's CRT does not, so on
// Windows we fall back to a tmpfile() read back after compilation. Diagnostics
// only matter on a compile *failure* (which valid generated source never hits),
// so a null stream degrades gracefully to stderr rather than aborting.
class DiagStream {
  public:
    DiagStream() {
#ifdef _WIN32
        file_ = std::tmpfile();
#else
        file_ = open_memstream(&buf_, &len_);
#endif
    }
    ~DiagStream() {
        if (file_ != nullptr)
            std::fclose(file_);
#ifndef _WIN32
        if (buf_ != nullptr)
            std::free(buf_);
#endif
    }
    DiagStream(const DiagStream &) = delete;
    DiagStream &operator=(const DiagStream &) = delete;

    // Stream handed to c2mir as message_file; falls back to stderr if the
    // in-memory/temp stream could not be created.
    FILE *file() const { return file_ != nullptr ? file_ : stderr; }

    // Finalize and return whatever c2mir wrote. Call after c2mir_compile.
    std::string str() {
        if (file_ == nullptr)
            return {};
        std::fflush(file_);
#ifdef _WIN32
        long end = std::ftell(file_);
        if (end <= 0)
            return {};
        std::rewind(file_);
        std::string out(static_cast<size_t>(end), '\0');
        size_t n = std::fread(&out[0], 1, out.size(), file_);
        out.resize(n);
        return out;
#else
        return buf_ != nullptr ? std::string(buf_, len_) : std::string();
#endif
    }

  private:
    FILE *file_ = nullptr;
#ifndef _WIN32
    char *buf_ = nullptr;
    size_t len_ = 0;
#endif
};

// getc callback for c2mir_compile: streams a NUL-terminated C source string.
struct MirSourceReader {
    const char *p;
};
inline int mir_source_getc(void *data) {
    auto *r = static_cast<MirSourceReader *>(data);
    int c = static_cast<unsigned char>(*r->p);
    if (c == 0)
        return EOF;
    r->p++;
    return c;
}

} // namespace detail
#endif

/// In-process JIT compiler for a code-generated C RHS source string.
///
/// Usage (mirrors DynamicLibrary):
///   MirJit jit(c_source);                                  // throws on failure
///   auto rhs = jit.symbol<RhsFn>("bngsim_codegen_rhs");    // throws if absent
///   auto jac = jit.try_symbol<JacFn>("bngsim_codegen_jac");// nullptr if absent
///   // jit owns the JIT'd code; keep it alive while the function pointers are used.
class MirJit {
  public:
    MirJit() = default;

    /// JIT-compile a generated C RHS source string in-process.
    /// opt_level selects MIR_gen's optimization level: 0-3 forces a level;
    /// -1 (default) auto-selects by source size (see kAuto). MIR's -O2
    /// register allocation/optimization is superlinear on a single very large
    /// flat RHS function (measured: a 3 MB / 1026-species RHS takes ~71 s at
    /// -O2 but ~4 s at -O0), so auto downgrades the level as the source grows,
    /// mirroring how bngsim._codegen.compile_rhs downgrades cc's -O flag.
    /// Throws std::runtime_error on c2mir/MIR failure (with captured diagnostics).
    static constexpr int kAuto = -1;

    /// True iff this build embeds the MIR backend (configured with
    /// -DBNGSIM_ENABLE_MIR=ON). Lets callers gate an in-process MIR fallback
    /// without attempting (and catching) a constructor that would only throw on a
    /// non-MIR build — e.g. the SSA propensity path's compiler-less fallback.
    static constexpr bool available() {
#if defined(BNGSIM_HAS_MIR)
        return true;
#else
        return false;
#endif
    }
    explicit MirJit(const std::string &c_source, int opt_level = kAuto) {
#if !defined(BNGSIM_HAS_MIR)
        (void)c_source;
        (void)opt_level;
        throw std::runtime_error("MirJit: bngsim was built without the MIR backend (configure with "
                                 "-DBNGSIM_ENABLE_MIR=ON).");
#else
        std::string jit_source = make_jit_source(c_source);
        compile(jit_source, opt_level == kAuto ? auto_opt_level(jit_source.size()) : opt_level);
#endif
    }

    ~MirJit() { close(); }

    // Non-copyable
    MirJit(const MirJit &) = delete;
    MirJit &operator=(const MirJit &) = delete;

    // Movable
    MirJit(MirJit &&other) noexcept {
#if defined(BNGSIM_HAS_MIR)
        ctx_ = other.ctx_;
        gen_inited_ = other.gen_inited_;
        other.ctx_ = nullptr;
        other.gen_inited_ = false;
#else
        (void)other;
#endif
    }
    MirJit &operator=(MirJit &&other) noexcept {
        if (this != &other) {
            close();
#if defined(BNGSIM_HAS_MIR)
            ctx_ = other.ctx_;
            gen_inited_ = other.gen_inited_;
            other.ctx_ = nullptr;
            other.gen_inited_ = false;
#endif
        }
        return *this;
    }

    /// True if a module has been JIT-compiled and linked.
    explicit operator bool() const {
#if defined(BNGSIM_HAS_MIR)
        return ctx_ != nullptr;
#else
        return false;
#endif
    }

    /// Resolve a JIT'd function by name. Throws if not found.
    template <typename T> T symbol(const std::string &name) const {
        void *addr = raw_symbol(name);
        if (addr == nullptr) {
            throw std::runtime_error("MirJit: symbol '" + name + "' not found in JIT module");
        }
        return reinterpret_cast<T>(addr);
    }

    /// Resolve a JIT'd function by name, returning nullptr if not found.
    template <typename T> T try_symbol(const std::string &name) const noexcept {
        return reinterpret_cast<T>(raw_symbol(name));
    }

  private:
#if defined(BNGSIM_HAS_MIR)
    MIR_context_t ctx_ = nullptr;

    // MIR_gen_init() runs only AFTER c2mir succeeds (see compile()); a c2mir
    // failure calls close() before gen_init was ever reached. Calling
    // MIR_gen_finish() without a matching MIR_gen_init() hard-aborts the whole
    // process ("Calling MIR_gen_finish before MIR_gen_init -- good bye"), turning
    // a catchable compile error into a crash. Track whether gen_init ran so
    // close() only finishes the generator when it was actually started.
    bool gen_inited_ = false;

    // c2mir cannot parse the platform SDK <math.h>/<stdlib.h>/<string.h> (they
    // use compiler-specific extensions; on macOS c2mir reports "Unsupported
    // compiler detected"). Since bngsim *generates* the C it JITs, the JIT path
    // strips the system #includes and prepends extern declarations for the
    // libc/libm symbols the RHS can call; MIR resolves them at link time via the
    // import resolver (dlsym on POSIX; a static libm/libc table on Windows — see
    // detail::mir_import_resolver). The RHS *body* — everything that affects
    // numerics — is byte-identical to what the cc-subprocess backend compiles.
    // Only the header preamble differs.
    static std::string make_jit_source(const std::string &src) {
        std::string out;
        out.reserve(src.size() + 2048);
        out += jit_prelude();
        // Strip lines of the form: optional ws, '#', optional ws, "include", ws, '<'.
        size_t i = 0, n = src.size();
        while (i < n) {
            size_t eol = src.find('\n', i);
            if (eol == std::string::npos)
                eol = n;
            if (!is_system_include_line(src, i, eol)) {
                out.append(src, i, eol - i);
                if (eol < n)
                    out.push_back('\n');
            }
            i = eol + 1;
        }
        return out;
    }

    static bool is_system_include_line(const std::string &s, size_t begin, size_t end) {
        size_t j = begin;
        auto skip_ws = [&]() {
            while (j < end && (s[j] == ' ' || s[j] == '\t'))
                j++;
        };
        skip_ws();
        if (j >= end || s[j] != '#')
            return false;
        j++;
        skip_ws();
        static const char kw[] = "include";
        for (const char *k = kw; *k; ++k) {
            if (j >= end || s[j] != *k)
                return false;
            j++;
        }
        skip_ws();
        return j < end && s[j] == '<';
    }

    // Forward declarations for every libc/libm function the codegen can emit
    // (see _BUILTIN_IDENT_MAP, the ^→pow lowering, the MM sqrt, and ExprTk
    // builtins that pass through to <math.h>). M_PI/M_E are still provided by
    // the generated source's own #ifndef/#define blocks. The memset/memcpy size
    // argument must match c2mir's built-in size_t width (what sizeof yields in
    // the generated call): unsigned long on the LP64 targets (macOS/Linux
    // x86-64 + aarch64) and unsigned long long on Windows (LLP64) — mirroring
    // c2mir/x86_64/mirc_x86_64_stddef.h. Getting this wrong on Windows would
    // narrow the byte count passed to memset (GH #3).
    static const char *jit_prelude() {
        return "/* bngsim MIR-JIT prelude (forward-declare libc/libm; resolved via the "
               "import resolver) */\n"
               "extern double pow(double, double);\n"
               "extern double exp(double); extern double exp2(double); extern double "
               "expm1(double);\n"
               "extern double log(double); extern double log2(double); extern double "
               "log10(double);\n"
               "extern double log1p(double); extern double logb(double);\n"
               "extern double sqrt(double); extern double cbrt(double); extern double "
               "hypot(double, double);\n"
               "extern double fabs(double); extern double fmax(double, double); extern double "
               "fmin(double, double);\n"
               "extern double fmod(double, double); extern double copysign(double, double);\n"
               "extern double floor(double); extern double ceil(double); extern double "
               "round(double);\n"
               "extern double trunc(double); extern double rint(double); extern double "
               "nearbyint(double);\n"
               "extern double sin(double); extern double cos(double); extern double tan(double);\n"
               "extern double asin(double); extern double acos(double); extern double "
               "atan(double);\n"
               "extern double atan2(double, double);\n"
               "extern double sinh(double); extern double cosh(double); extern double "
               "tanh(double);\n"
               "extern double asinh(double); extern double acosh(double); extern double "
               "atanh(double);\n"
               "extern double erf(double); extern double erfc(double);\n"
               "extern double tgamma(double); extern double lgamma(double);\n"
#ifdef _WIN32
               // Windows is LLP64 — c2mir's size_t is unsigned long long (64-bit).
               "extern void *memset(void *, int, unsigned long long);\n"
               "extern void *memcpy(void *, const void *, unsigned long long);\n";
#else
               // LP64 — c2mir's size_t is unsigned long (64-bit).
               "extern void *memset(void *, int, unsigned long);\n"
               "extern void *memcpy(void *, const void *, unsigned long);\n";
#endif
    }

    // Auto opt-level by JIT source size. MIR -O2's optimizer is superlinear on
    // one giant flat function, so step the level down as the source grows. The
    // thresholds bracket the measured pathology (3 MB → 71 s at -O2) with margin;
    // the intermediate band GH #78 targets (≤256 species, ~tens of KB) stays at
    // -O2 for best code quality. The genuinely optimal policy needs RUNTIME
    // benchmarking (compile-vs-integration trade-off) — the gate the issue
    // describes — so these are deliberately conservative defaults, overridable
    // via the opt_level argument.
    static int auto_opt_level(size_t source_bytes) {
        constexpr size_t kO2Max = 512u * 1024u;       // ≤512 KB: -O2
        constexpr size_t kO1Max = 4u * 1024u * 1024u; // ≤4 MB: -O1, else -O0
        if (source_bytes <= kO2Max)
            return 2;
        if (source_bytes <= kO1Max)
            return 1;
        return 0;
    }

    void compile(const std::string &jit_source, int opt_level) {
        ctx_ = MIR_init();
        if (ctx_ == nullptr)
            throw std::runtime_error("MirJit: MIR_init failed");
        c2mir_init(ctx_);

        // Capture c2mir diagnostics for error reporting (portable in-memory
        // stream on POSIX, tmpfile on Windows — see detail::DiagStream).
        detail::DiagStream diag;

        struct c2mir_options opts;
        std::memset(&opts, 0, sizeof(opts));
        opts.message_file = diag.file();

        detail::MirSourceReader reader{jit_source.c_str()};
        int ok = c2mir_compile(ctx_, &opts, detail::mir_source_getc, &reader, "bngsim_rhs_module",
                               nullptr);
        std::string diagnostics = diag.str();

        if (!ok) {
            close();
            throw std::runtime_error("MirJit: c2mir failed to compile generated RHS source:\n" +
                                     diagnostics);
        }

        for (MIR_module_t m = DLIST_HEAD(MIR_module_t, *MIR_get_module_list(ctx_)); m != nullptr;
             m = DLIST_NEXT(MIR_module_t, m)) {
            MIR_load_module(ctx_, m);
        }

        MIR_gen_init(ctx_);
        gen_inited_ = true;
        if (opt_level < 0)
            opt_level = 0;
        if (opt_level > 3)
            opt_level = 3;
        MIR_gen_set_optimize_level(ctx_, static_cast<unsigned>(opt_level));
        MIR_link(ctx_, MIR_set_gen_interface, detail::mir_import_resolver);
    }

    // Find a JIT'd function item by name across loaded modules and return its
    // linked native code address. MIR_get_global_item is declared but not
    // defined in the pinned commit, so scan items (the c2mir-driver approach).
    void *raw_symbol(const std::string &name) const {
        if (ctx_ == nullptr)
            return nullptr;
        for (MIR_module_t m = DLIST_HEAD(MIR_module_t, *MIR_get_module_list(ctx_)); m != nullptr;
             m = DLIST_NEXT(MIR_module_t, m)) {
            for (MIR_item_t it = DLIST_HEAD(MIR_item_t, m->items); it != nullptr;
                 it = DLIST_NEXT(MIR_item_t, it)) {
                if (it->item_type == MIR_func_item && it->u.func != nullptr &&
                    name == it->u.func->name) {
                    return it->addr;
                }
            }
        }
        return nullptr;
    }

    void close() {
        if (ctx_ != nullptr) {
            // Only finish the generator if it was started — see gen_inited_.
            // c2mir_init + MIR_init always ran by the time ctx_ is non-null, so
            // their finishers are unconditional.
            if (gen_inited_) {
                MIR_gen_finish(ctx_);
                gen_inited_ = false;
            }
            c2mir_finish(ctx_);
            MIR_finish(ctx_);
            ctx_ = nullptr;
        }
    }
#else // !BNGSIM_HAS_MIR — stub so the codegen dispatch compiles unconditionally.
    void *raw_symbol(const std::string &) const { return nullptr; }
    void close() {}
#endif
};

} // namespace bngsim
