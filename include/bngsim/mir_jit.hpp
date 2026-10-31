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

#include <memory>
#include <stdexcept>
#include <string>

#if defined(BNGSIM_HAS_MIR)
#include "bngsim/c_locale_numbers.hpp" // c2mir reads literals with strtod (#1235)

#include <cerrno>
#include <cmath>   // libm addresses for the prelude's import table (GH #3)
#include <csetjmp> // MIR's failures return to compile() by longjmp (issue #1215)
#include <cstdarg>
#include <cstddef>
#include <cstdint> // intptr_t
#include <cstdio>  // open_memstream/tmpfile, FILE, fflush, fclose, stderr
#include <cstdlib> // free, malloc, calloc, realloc, exit
#include <cstring> // std::memset, std::memcpy, std::strcmp
#ifndef _WIN32
#include <dlfcn.h> // dlsym(RTLD_DEFAULT) — POSIX import resolver
#endif
extern "C" {
#include "c2mir.h"
#include "mir-gen.h"
#include "mir.h"
// MIR's default code allocator, from the local carry in mir.c (issue #1215).
MIR_code_alloc_t _MIR_get_default_code_alloc(void);
}
#ifndef MIR_CODE_RESERVE_SIZE
#error "CMake defines MIR_CODE_RESERVE_SIZE for bngsim_mir and what links it (issue #1215)"
#endif
#endif

namespace bngsim {

#if defined(BNGSIM_HAS_MIR)
namespace detail {

// The libc/libm functions jit_prelude() declares, by name, with their
// addresses: the whole import surface a generated source is meant to have.
// Keep it in step with jit_prelude() below. Taking &std::pow (etc.) yields a
// valid out-of-line address on every ABI.
//
// It is not the surface a source *can* have. c2mir takes a call to an
// undeclared function as an implicit `int name()`, the C89 rule, where cc
// refuses it; so a rate law that calls an engine function the emitter has no C
// for (frac, inrange, a table function's name) reaches MIR_link as one more
// import. MirJit::compile refuses those by name before linking (see there).
inline void *mir_prelude_address(const char *name) {
    struct Entry {
        const char *name;
        void *addr;
    };
    // Cast through the exact C signature to pick the right <cmath> overload.
#define BNGSIM_MIR_D1(fn)                                                                          \
    { #fn, reinterpret_cast<void *>(static_cast<double (*)(double)>(&std::fn)) }
#define BNGSIM_MIR_D2(fn)                                                                          \
    { #fn, reinterpret_cast<void *>(static_cast<double (*)(double, double)>(&std::fn)) }
    static const Entry table[] = {
        BNGSIM_MIR_D2(pow),
        BNGSIM_MIR_D1(exp),
        BNGSIM_MIR_D1(exp2),
        BNGSIM_MIR_D1(expm1),
        BNGSIM_MIR_D1(log),
        BNGSIM_MIR_D1(log2),
        BNGSIM_MIR_D1(log10),
        BNGSIM_MIR_D1(log1p),
        BNGSIM_MIR_D1(logb),
        BNGSIM_MIR_D1(sqrt),
        BNGSIM_MIR_D1(cbrt),
        BNGSIM_MIR_D2(hypot),
        BNGSIM_MIR_D1(fabs),
        BNGSIM_MIR_D2(fmax),
        BNGSIM_MIR_D2(fmin),
        BNGSIM_MIR_D2(fmod),
        BNGSIM_MIR_D2(copysign),
        BNGSIM_MIR_D1(floor),
        BNGSIM_MIR_D1(ceil),
        BNGSIM_MIR_D1(round),
        BNGSIM_MIR_D1(trunc),
        BNGSIM_MIR_D1(rint),
        BNGSIM_MIR_D1(nearbyint),
        BNGSIM_MIR_D1(sin),
        BNGSIM_MIR_D1(cos),
        BNGSIM_MIR_D1(tan),
        BNGSIM_MIR_D1(asin),
        BNGSIM_MIR_D1(acos),
        BNGSIM_MIR_D1(atan),
        BNGSIM_MIR_D2(atan2),
        BNGSIM_MIR_D1(sinh),
        BNGSIM_MIR_D1(cosh),
        BNGSIM_MIR_D1(tanh),
        BNGSIM_MIR_D1(asinh),
        BNGSIM_MIR_D1(acosh),
        BNGSIM_MIR_D1(atanh),
        BNGSIM_MIR_D1(erf),
        BNGSIM_MIR_D1(erfc),
        BNGSIM_MIR_D1(tgamma),
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

// Import resolver for MIR_link: maps a name jit_prelude() declares to its
// runtime address, and every other name to nullptr. Must have C linkage / no
// captures.
//
// POSIX: dlsym(RTLD_DEFAULT) searches every globally-loaded symbol, which
// includes libSystem/libm already resident in this process. It is asked only
// about the prelude's names: any other name it found would be some unrelated
// global of the same spelling, called through an implicit `int name()`.
//
// Windows/MSVC (GH #3): there is no dlsym/RTLD_DEFAULT, so the table above is
// the resolver.
#ifdef _WIN32
inline void *mir_import_resolver(const char *name) { return mir_prelude_address(name); }
#else
inline void *mir_import_resolver(const char *name) {
    return mir_prelude_address(name) != nullptr ? dlsym(RTLD_DEFAULT, name) : nullptr;
}
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

// Issue #1215: MIR's answer to running out of memory or address space was to
// end the process. Its default error function calls exit(1), its growable
// arrays call exit(1) themselves when an allocation fails, and a failed code
// mapping comes back as a NULL code address that most callers write through.
// Under a cap on the address space (`ulimit -v`) that is where enough live
// MIR-compiled models end up.
//
// So every MIR context MirJit makes allocates through mir_alloc, maps its code
// through a MirCodeMaps, and reports through mir_error, and all three, while
// MirJit::compile has a call into MIR under way, longjmp back to it with the
// message. (MIR.md asks only that an error function never return.) compile()
// then throws (see MirJit::throw_mir_failure for what becomes of the context).
// Nothing between the setjmp and the longjmp may own a destructor: the frames
// in between are MIR's C and the steps compile() hands to mir_guarded, which
// keep only trivially destructible locals.
struct MirFailure {
    std::jmp_buf env;
    MirFailure *outer;
    bool out_of_memory;
    char message[256];
};

// The failure that compile()'s call into MIR under way on this thread returns
// to, or nullptr when none is.
inline thread_local MirFailure *mir_failure_target = nullptr;

// True while MIR_init2 runs on this thread: its first mapping is the reservation.
inline thread_local bool mir_initializing = false;

// Test hooks, which _bngsim_core._mir_jit_compile sets: whether this thread's
// compiles are counted at all (allocations through the counting table mir_alloc
// picks, so that a compile outside the test pays nothing for it); the
// allocations MIR has asked for, and the index of one to fail as if the C heap
// were out; the code mappings it has made besides its reservation, the index of
// one to refuse, which also refuses the reservation so that there are such
// mappings, and the errno to refuse it with (ENOMEM: as if the address space
// were out). -1 fails none.
inline thread_local bool mir_test_hooks = false;
inline thread_local long long mir_allocation_count = 0;
inline thread_local long long mir_failing_allocation = -1;
inline thread_local long long mir_mapping_count = 0;
inline thread_local long long mir_failing_mapping = -1;
inline thread_local int mir_failing_mapping_errno = ENOMEM;

// MIR's error function. With no call under way it does what MIR's default does.
// MIR_NO_RETURN rather than [[noreturn]]: GCC and clang make the attribute part
// of the type MIR_error_func_t names.
inline void MIR_NO_RETURN mir_error(MIR_error_type_t type, const char *format, ...) {
    MirFailure *failure = mir_failure_target;
    va_list ap;
    va_start(ap, format);
    if (failure == nullptr) {
        std::vfprintf(stderr, format, ap);
        va_end(ap);
        std::fputc('\n', stderr);
        std::exit(1);
    }
    failure->out_of_memory = type == MIR_alloc_error;
    std::vsnprintf(failure->message, sizeof failure->message, format, ap);
    va_end(ap);
    std::longjmp(failure->env, 1);
}

[[noreturn]] inline void mir_report(MirFailure *failure, bool out_of_memory, const char *message) {
    failure->out_of_memory = out_of_memory;
    std::snprintf(failure->message, sizeof failure->message, "%s", message);
    std::longjmp(failure->env, 1);
}

// What mir_alloc hands MIR. An allocation that came back empty while a call is
// under way is reported, so MIR never sees the null; with none under way MIR
// gets it, as it would from its default allocator.
inline void *mir_allocated(void *mem, size_t size) {
    if (mem != nullptr || size == 0)
        return mem;
    MirFailure *failure = mir_failure_target;
    if (failure == nullptr)
        return mem;
    mir_report(failure, true, "out of memory");
}

inline bool mir_allocation_injected_to_fail() {
    return mir_allocation_count++ == mir_failing_allocation;
}

inline void *mir_malloc(size_t size, void *) { return mir_allocated(std::malloc(size), size); }
inline void *mir_calloc(size_t num, size_t size, void *) {
    return mir_allocated(std::calloc(num, size), num * size);
}
inline void *mir_realloc(void *ptr, size_t, size_t new_size, void *) {
    return mir_allocated(std::realloc(ptr, new_size), new_size);
}
inline void mir_free(void *ptr, void *) { std::free(ptr); }

inline void *mir_malloc_counted(size_t size, void *) {
    return mir_allocated(mir_allocation_injected_to_fail() ? nullptr : std::malloc(size), size);
}
inline void *mir_calloc_counted(size_t num, size_t size, void *) {
    return mir_allocated(mir_allocation_injected_to_fail() ? nullptr : std::calloc(num, size),
                         num * size);
}
// A failed realloc leaves ptr allocated and the caller still holding it, so an
// injected failure skips the call rather than undoing it.
inline void *mir_realloc_counted(void *ptr, size_t, size_t new_size, void *) {
    return mir_allocated(mir_allocation_injected_to_fail() ? nullptr : std::realloc(ptr, new_size),
                         new_size);
}

// c2mir frees some of what it allocates through the context's allocator with
// plain free(), so mir_alloc must stay on the C heap.
inline MIR_alloc_t mir_alloc() {
    static struct MIR_alloc plain = {mir_malloc, mir_calloc, mir_realloc, mir_free, nullptr};
    static struct MIR_alloc counted = {mir_malloc_counted, mir_calloc_counted, mir_realloc_counted,
                                       mir_free, nullptr};
    return mir_test_hooks ? &counted : &plain;
}

// One context's code mappings. MIR is handed `alloc`, whose user_data points
// here; MIR's default code allocator does the mapping. A failed mapping while a
// call is under way is reported, as mir_alloc reports a failed allocation. The
// one exception is the reservation MIR_init2 asks for first: when that is
// refused MIR maps its code page by page instead, so it gets the refusal.
//
// The mappings are recorded so that a context abandoned after a failure can
// give its address space back (MirJit::throw_mir_failure). MIR unmaps code only
// in MIR_finish, which an abandoned context never reaches, so the record needs
// no removals.
struct MirCodeMaps {
    struct Map {
        void *addr;
        size_t len;
    };
    struct MIR_code_alloc alloc;
    Map *maps = nullptr;
    size_t n = 0;
    size_t cap = 0;

    MirCodeMaps();
    MirCodeMaps(const MirCodeMaps &) = delete;
    MirCodeMaps &operator=(const MirCodeMaps &) = delete;
    ~MirCodeMaps() { std::free(maps); }

    // A mapping the record has no room for stays mapped if its context is
    // abandoned; nothing else goes wrong.
    void record(void *addr, size_t len) {
        if (n == cap) {
            size_t grown = cap == 0 ? 16 : 2 * cap;
            auto *more = static_cast<Map *>(std::realloc(maps, grown * sizeof(Map)));
            if (more == nullptr)
                return;
            maps = more;
            cap = grown;
        }
        maps[n++] = {addr, len};
    }

    void unmap_all() {
        MIR_code_alloc_t base = _MIR_get_default_code_alloc();
        for (size_t i = 0; i < n; ++i)
            base->mem_unmap(maps[i].addr, maps[i].len, base->user_data);
        n = 0;
    }
};

// What MIR's default code allocator returns for a refused mapping: mmap's
// MAP_FAILED, or VirtualAlloc's NULL.
inline void *mir_map_failed() {
#ifdef _WIN32
    return nullptr;
#else
    return reinterpret_cast<void *>(static_cast<std::intptr_t>(-1));
#endif
}

inline void *mir_mem_map(size_t len, void *user_data) {
    auto *code = static_cast<MirCodeMaps *>(user_data);
    MIR_code_alloc_t base = _MIR_get_default_code_alloc();
    const bool reservation = mir_initializing && len == MIR_CODE_RESERVE_SIZE;
    bool refused = false;
    if (mir_test_hooks)
        refused =
            reservation ? mir_failing_mapping >= 0 : mir_mapping_count++ == mir_failing_mapping;
    void *mem = refused ? mir_map_failed() : base->mem_map(len, base->user_data);
    if (mem != nullptr && mem != mir_map_failed()) {
        code->record(mem, len);
        return mem;
    }
    // mmap says why; a refusal other than ENOMEM is a policy against
    // executable memory (SELinux deny_execmem, a hardened runtime without the
    // JIT entitlement), which no cap explains. VirtualAlloc reports through
    // GetLastError, and is taken to have run out.
#ifdef _WIN32
    const int why = refused ? mir_failing_mapping_errno : ENOMEM;
#else
    const int why = refused ? mir_failing_mapping_errno : errno;
#endif
    MirFailure *failure = mir_failure_target;
    if (reservation || failure == nullptr)
        return mem;
    if (why == ENOMEM)
        mir_report(failure, true, "no address space to map code");
    char message[sizeof failure->message];
    std::snprintf(message, sizeof message, "cannot map executable memory for code: %s",
                  std::strerror(why));
    mir_report(failure, false, message);
}
inline int mir_mem_unmap(void *addr, size_t len, void *) {
    MIR_code_alloc_t base = _MIR_get_default_code_alloc();
    return base->mem_unmap(addr, len, base->user_data);
}
inline int mir_mem_protect(void *addr, size_t len, MIR_mem_protect_t prot, void *) {
    MIR_code_alloc_t base = _MIR_get_default_code_alloc();
    return base->mem_protect(addr, len, prot, base->user_data);
}

inline MirCodeMaps::MirCodeMaps() : alloc{mir_mem_map, mir_mem_unmap, mir_mem_protect, this} {}

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
        code_maps_ = std::move(other.code_maps_);
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
            code_maps_ = std::move(other.code_maps_);
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
    // ctx_'s code allocator. MIR keeps a pointer to it, so it lives on the heap,
    // where a move leaves it, and outlives ctx_.
    std::unique_ptr<detail::MirCodeMaps> code_maps_;

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
    // import resolver (dlsym on POSIX, for the prelude's names only; a static
    // libm/libc table on Windows — see detail::mir_import_resolver). The RHS
    // *body* — everything that affects
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
    // the generated source's own #ifndef/#define blocks: every source carrying
    // its own #include <math.h> defines both. The sensitivity source did not,
    // and got away with it only because it is concatenated after an RHS that
    // does — python/tests/test_codegen_jit_prelude.py now asks each source on
    // its own (GH #470).
    //
    // Stripping the headers also strips the *types and macros* they define, and
    // the generated source names four of those (GH #85 — a missing one is a
    // compile failure of the whole module, not a graceful decline):
    //
    //   size_t  the (size_t) index casts in bngsim_codegen_output_sens (#198).
    //           Unknown as a typedef name, `(size_t)_c` stops being a cast and
    //           the whole declaration fails to parse — which is why #85 showed
    //           up as "syntax error on double (expected '<statement>')".
    //   NULL    the no-observables function-block call args (both the sens RHS
    //           and #198 pass NULL for `obs` when the model has no observables).
    //   NAN     the unsupported-function sentinel in the #198 block.
    //   INFINITY sympy's C99 printer for oo, which can land in the untaken
    //           branch of a printed derivative (k = if(p > 1, p, p*abs(1/0))).
    //
    // c2mir bundles its own <stddef.h> (it is in every target's
    // TARGET_STD_INCLUDES; only <math.h>/<stdlib.h>/<string.h> have no bundled
    // copy and fall through to the unparseable platform SDK), so including it
    // here is what defines size_t — at the width c2mir's own sizeof yields, on
    // every target, with no hand-maintained LP64/LLP64 table to get wrong. That
    // is also why memset/memcpy below can now just say size_t: matching c2mir's
    // width is what GH #3 needed, and this gets it by construction rather than
    // by mirroring c2mir/x86_64/mirc_x86_64_stddef.h from the outside.
    //
    // NULL, NAN and INFINITY are defined here rather than taken from that
    // header: c2mir's stddef.h omits NULL on macOS-x86_64 and Windows, and no
    // bundled header defines NAN or INFINITY at all. All three are
    // #ifndef-guarded so a target that does define them wins.
    static const char *jit_prelude() {
        return "/* bngsim MIR-JIT prelude (forward-declare libc/libm; resolved via the "
               "import resolver) */\n"
               "#include <stddef.h>\n" // size_t (+ NULL where c2mir defines it)
               "#ifndef NULL\n"
               "#define NULL ((void *)0)\n"
               "#endif\n"
               "#ifndef NAN\n"
               "#define NAN (0.0 / 0.0)\n"
               "#endif\n"
               "#ifndef INFINITY\n"
               "#define INFINITY (1.0 / 0.0)\n"
               "#endif\n"
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
               // size_t comes from c2mir's own <stddef.h> above, so these match
               // whatever the generated sizeof() call passes on this target —
               // unsigned long on LP64, unsigned long long on Windows (GH #3).
               "extern void *memset(void *, int, size_t);\n"
               "extern void *memcpy(void *, const void *, size_t);\n";
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

    // Runs step() with a failure MIR reports returning here (see
    // detail::MirFailure): false, with failure filled in, if one did. step must
    // keep only trivially destructible locals of its own.
    template <typename Step> static bool mir_guarded(detail::MirFailure &failure, Step &&step) {
        failure.outer = detail::mir_failure_target;
        if (setjmp(failure.env) != 0) {
            detail::mir_failure_target = failure.outer;
            return false;
        }
        detail::mir_failure_target = &failure;
        step();
        detail::mir_failure_target = failure.outer;
        return true;
    }

    // A context MIR reported a failure in is abandoned, not finished. MIR does
    // not keep its state whole through an error that returns: a function it had
    // begun has no name yet, the inliner's flags are still set where MIR_finish
    // frees data, and MIR_finish crashes on them. So its heap memory stays
    // allocated; its code mappings, the address space a cap counts, are given
    // back, since nothing can reach the code in them.
    [[noreturn]] void throw_mir_failure(const detail::MirFailure &failure) {
        code_maps_->unmap_all();
        code_maps_.reset();
        ctx_ = nullptr;
        gen_inited_ = false;
        if (failure.out_of_memory) {
            throw std::runtime_error(
                std::string("MirJit: MIR ran out of memory or address space compiling the "
                            "generated source (") +
                failure.message +
                "). Under a cap on the address space (ulimit -v), every live MIR-compiled "
                "model counts against it; free some or raise the cap (issue #1215).");
        }
        throw std::runtime_error(
            std::string("MirJit: MIR failed compiling the generated source: ") + failure.message);
    }

    void compile(const std::string &jit_source, int opt_level) {
        code_maps_ = std::make_unique<detail::MirCodeMaps>();
        detail::MirFailure failure;
        detail::mir_initializing = true;
        const bool inited = mir_guarded(
            failure, [this] { ctx_ = MIR_init2(detail::mir_alloc(), &code_maps_->alloc); });
        detail::mir_initializing = false;
        if (!inited)
            throw_mir_failure(failure);
        MIR_set_error_func(ctx_, detail::mir_error);
        if (!mir_guarded(failure, [this] { c2mir_init(ctx_); }))
            throw_mir_failure(failure);

        // Capture c2mir diagnostics for error reporting (portable in-memory
        // stream on POSIX, tmpfile on Windows — see detail::DiagStream).
        detail::DiagStream diag;

        struct c2mir_options opts;
        std::memset(&opts, 0, sizeof(opts));
        opts.message_file = diag.file();

        detail::MirSourceReader reader{jit_source.c_str()};
        int ok = 0;
        bool reported = false;
        {
            // c2mir converts a float literal with strtod and refuses whatever
            // strtod leaves, so under a decimal-comma LC_NUMERIC every `0.5` in
            // the source was a "wrong number" and the compile failed (#1235).
            const ScopedCNumericLocale c_numbers;
            reported = !mir_guarded(failure, [&] {
                ok = c2mir_compile(ctx_, &opts, detail::mir_source_getc, &reader,
                                   "bngsim_rhs_module", nullptr);
            });
        }
        if (reported)
            throw_mir_failure(failure);
        std::string diagnostics = diag.str();

        // c2mir's parse of the source is spent once it has made the MIR module,
        // and for a large model it is most of the context: fceri_fyn's live
        // context held 852 MB of heap with it, 388 MB without (issue #1215).
        c2mir_finish(ctx_);

        if (!ok) {
            close();
            throw std::runtime_error("MirJit: c2mir failed to compile generated RHS source:\n" +
                                     diagnostics);
        }

        if (!mir_guarded(failure, [this] {
                for (MIR_module_t m = DLIST_HEAD(MIR_module_t, *MIR_get_module_list(ctx_));
                     m != nullptr; m = DLIST_NEXT(MIR_module_t, m)) {
                    MIR_load_module(ctx_, m);
                }
            }))
            throw_mir_failure(failure);

        // An import the resolver cannot answer is a call c2mir declared
        // implicitly (see detail::mir_prelude_address). MIR_link would report it
        // through MIR's error function, as a failure of MIR's rather than of the
        // source, so refuse it here, as cc refuses it.
        std::string unresolved = first_unresolved_import(ctx_);
        if (!unresolved.empty()) {
            close();
            throw std::runtime_error(
                "MirJit: the generated source refers to " + unresolved +
                ", which it does not define and which is not one of the libc/libm functions "
                "the JIT prelude declares (c2mir accepts a call to an undeclared function; "
                "the cc backend refuses it)");
        }

        if (opt_level < 0)
            opt_level = 0;
        if (opt_level > 3)
            opt_level = 3;
        if (!mir_guarded(failure, [this, opt_level] {
                MIR_gen_init(ctx_);
                gen_inited_ = true;
                MIR_gen_set_optimize_level(ctx_, static_cast<unsigned>(opt_level));
                MIR_link(ctx_, MIR_set_gen_interface, detail::mir_import_resolver);
            }))
            throw_mir_failure(failure);
    }

    // The first import across the loaded modules that the import resolver has no
    // address for, or "" if every one resolves.
    static std::string first_unresolved_import(MIR_context_t ctx) {
        for (MIR_module_t m = DLIST_HEAD(MIR_module_t, *MIR_get_module_list(ctx)); m != nullptr;
             m = DLIST_NEXT(MIR_module_t, m)) {
            for (MIR_item_t it = DLIST_HEAD(MIR_item_t, m->items); it != nullptr;
                 it = DLIST_NEXT(MIR_item_t, it)) {
                if (it->item_type == MIR_import_item &&
                    detail::mir_import_resolver(it->u.import_id) == nullptr) {
                    return it->u.import_id;
                }
            }
        }
        return {};
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
            // compile() finished c2mir before anything closes, and MIR_init2
            // always returned by the time ctx_ is non-null.
            if (gen_inited_) {
                MIR_gen_finish(ctx_);
                gen_inited_ = false;
            }
            MIR_finish(ctx_);
            ctx_ = nullptr;
            code_maps_.reset();
        }
    }
#else // !BNGSIM_HAS_MIR — stub so the codegen dispatch compiles unconditionally.
    void *raw_symbol(const std::string &) const { return nullptr; }
    void close() {}
#endif
};

} // namespace bngsim
