// bngsim/src/expression.cpp — ExprTk-based expression evaluator
//
// ExprTk replaces muParser and provides BNG-compatible aliases.
// Supports underscore-prefixed constants, built-in functions, and time().
//
// Note: BNG2.pl's muParser exposes simulation time as the zero-arg function
// time() and leaves t free as an ordinary identifier. We mirror that here so
// that BNGL models defining a parameter / observable / species named `t`
// (a common counter pattern, e.g. `Molecules t counter()`) load successfully.

// ExprTk is a large header — compile once here.
// Disable some ExprTk features we don't need to speed up compilation.
#define exprtk_disable_string_capabilities
#define exprtk_disable_rtl_io_file
#define exprtk_disable_rtl_vecops
// BNG is case-sensitive for parameter names (e.g., k3 ≠ K3).
// ExprTk defaults to case-insensitive, which silently merges k3/K3
// into the same variable and produces incorrect trajectories.
#define exprtk_disable_caseinsensitivity
#include "exprtk.hpp"

#include "bngsim/expr_compat.hpp"
#include "bngsim/expression.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bngsim {

// ─── Mratio: confluent hypergeometric ratio M(a+1,b+1,z)/M(a,b,z) ───────────
//
// Direct port of BNG2.pl Perl2/Expression.pm `sub Mratio`
// (Fortran by W. S. Hlavacek 2018; Perl by L. A. Harris 2019).
//
// Uses Gauss's continued fraction for the ratio of contiguous Kummer 1F1
// functions, evaluated by the modified Lentz method
// [Lentz 1976 Applied Optics 15:668-671; Thompson & Barnett 1986 J Comput
// Phys 64:490-509].
//
// CF coefficients (q_j = 1 for all j; q_0 = 0):
//   p_1 = 1
//   p_2 = z * [a - (b+0)] / [(b+0) * (b+1)]
//   p_3 = z * (a + 1)     / [(b+1) * (b+2)]
//   p_4 = z * [a - (b+1)] / [(b+2) * (b+3)]
//   p_5 = z * (a + 2)     / [(b+3) * (b+4)]
//   ...
//
// Why Lentz rather than the direct power-series for M: Lentz works with
// the per-step ratio Δ_j = C_j·D_j ≈ 1 and never accumulates the partial
// sums of M(a,b,z) and M(a+1,b+1,z) themselves. For BNG inputs with large
// negative-integer `a` and large `|z|` (e.g. test_Mratio_1: a=-1000, b=9001,
// z=-10000) the equivalent series partial sums peak around 1.5e308 — past
// double's representable range — and the ratio becomes inf/inf = nan. Lentz
// stays O(1) throughout and converges in a few hundred iterations even on
// those inputs.
//
// This is the single source of truth for mratio across BNGsim: the host
// MratioFunction adapter below and the vendored NFsim mu::Parser ExprTk shim
// both call it (the shim via <bngsim/expr_compat.hpp>). See issue #49.
double expr_compat::mratio(double a, double b, double z) {
    constexpr double eps = 1.0e-16;
    constexpr double tiny = 1.0e-32;
    // Safety cap so a pathological non-converging case fails loud rather
    // than hanging. BNG's reference has no cap; in practice the supported
    // parameter ranges converge in well under this bound.
    constexpr int max_iter = 100000;

    // Initialize per the modified-Lentz recipe: f_0 = q_0, but q_0 = 0
    // here, so substitute `tiny`. C_0 = f_0, D_0 = 0.
    double fsave = tiny;
    double Csave = fsave;
    double Dsave = 0.0;
    double err = 1.0 + eps;

    // Parity bookkeeping: even-indexed and odd-indexed CF terms use
    // different formulas for p_j. The flags alternate after every step.
    int odd = 1;
    int even = 0;
    int iodd = 0;
    int ieven = 0;
    double f = 0.0;

    int j = 0;
    while (err > eps) {
        ++j;
        if (j > max_iter) {
            throw std::runtime_error("mratio: modified-Lentz continued fraction failed to converge "
                                     "within " +
                                     std::to_string(max_iter) +
                                     " iterations "
                                     "(a=" +
                                     std::to_string(a) + ", b=" + std::to_string(b) +
                                     ", z=" + std::to_string(z) + ")");
        }

        double p;
        if (j == 1) {
            p = 1.0;
        } else {
            const double den = (b + (j - 2)) * (b + (j - 1));
            double num;
            if (odd == 1) {
                ++iodd;
                num = z * (a + iodd);
            } else {
                ++ieven;
                num = z * (a - (b + (ieven - 1)));
            }
            p = num / den;
        }
        constexpr double q = 1.0;

        double D = q + p * Dsave;
        if (std::abs(D) < tiny) {
            D = tiny;
        }
        double C = q + p / Csave;
        if (std::abs(C) < tiny) {
            C = tiny;
        }
        D = 1.0 / D;

        const double Delta = C * D;
        f = Delta * fsave;
        err = std::abs(Delta - 1.0);

        fsave = f;
        Csave = C;
        Dsave = D;
        std::swap(odd, even);
    }
    return f;
}

// ─── Non-finite return diagnostic ─────────────────────────────────────────────
//
// Custom functions handed to ExprTk that return nan/inf produce silent
// propagation through every downstream expression — exactly how issue #42
// went undiagnosed for so long (mratio overflowed to nan, and U1_U0 /
// C_mean / C_sdev / C_theory all just became nan with no logging).
//
// This helper traps the next such bug by stamping a one-time warning on
// stderr whenever a registered function returns a non-finite value, then
// deduplicating by (function name + argument bit-pattern) so that a
// long-running ODE that repeatedly evaluates the same bad input prints
// once rather than once-per-step. Argument bits are compared verbatim
// (no value equality), so nan-valued inputs deduplicate cleanly.
//
// Lives on ExprTkEvaluator::Impl; each adapter holds a back-pointer that
// is null only in default-constructed temporaries.
class NonFiniteWarningSet {
  public:
    void warn_if_nonfinite(const char *fname, std::initializer_list<double> args, double result) {
        if (std::isfinite(result)) {
            return;
        }
        std::string key(fname);
        for (double a : args) {
            std::uint64_t bits;
            std::memcpy(&bits, &a, sizeof(bits));
            char buf[20];
            std::snprintf(buf, sizeof(buf), ":%016llx", static_cast<unsigned long long>(bits));
            key.append(buf);
        }
        if (!seen_.insert(std::move(key)).second) {
            return;
        }
        std::cerr << "bngsim: warning: '" << fname << "(";
        const char *sep = "";
        for (double a : args) {
            std::cerr << sep << a;
            sep = ", ";
        }
        std::cerr << ")' returned " << result
                  << "; the value will propagate through any expression that "
                     "references this call. Further occurrences with the same "
                     "arguments will be silent."
                  << std::endl;
    }

  private:
    std::unordered_set<std::string> seen_;
};

// ─── ExprTk custom function adapters ─────────────────────────────────────────
//
// ExprTk requires inheriting from ifunction for custom functions. Each
// adapter holds a back-pointer to a NonFiniteWarningSet so a function
// that returns nan/inf is surfaced once on stderr instead of silently
// propagating (issue #42 follow-up).

// 3-arg: mratio(a, b, z)
template <typename T> struct MratioFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    MratioFunction() : exprtk::ifunction<T>(3) {
        exprtk::ifunction<T>::allow_zero_parameters() = false;
    }
    T operator()(const T &a, const T &b, const T &z) override {
        const double da = static_cast<double>(a);
        const double db = static_cast<double>(b);
        const double dz = static_cast<double>(z);
        const double r = expr_compat::mratio(da, db, dz);
        if (warner) {
            warner->warn_if_nonfinite("mratio", {da, db, dz}, r);
        }
        return static_cast<T>(r);
    }
};

// 1-arg aliases for backward compat
template <typename T> struct LnFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    LnFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = std::log(dx);
        if (warner) {
            warner->warn_if_nonfinite("ln", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct RintFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    RintFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = std::round(dx);
        if (warner) {
            warner->warn_if_nonfinite("rint", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct SignFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    SignFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = (dx > 0.0) ? 1.0 : ((dx < 0.0) ? -1.0 : 0.0);
        if (warner) {
            warner->warn_if_nonfinite("sign", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

// tgamma(x) = Γ(x); the SBML loader emits ``tgamma((n)+1)`` for MathML
// factorial(n) (ExprTk has no gamma builtin). std::tgamma matches the C
// library form native codegen emits, so interpreted and codegen agree.
template <typename T> struct TgammaFunction : public exprtk::ifunction<T> {
    NonFiniteWarningSet *warner = nullptr;
    TgammaFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = std::tgamma(dx);
        if (warner) {
            warner->warn_if_nonfinite("tgamma", {dx}, r);
        }
        return static_cast<T>(r);
    }
};

// 0-arg: time() — reads from a bound double*. No warner: the simulator
// owns the time pointer and a non-finite t would be a higher-level bug
// flagged by the integrator, not by this layer.
template <typename T> struct TimeFunction : public exprtk::ifunction<T> {
    double *time_ptr = nullptr;
    TimeFunction() : exprtk::ifunction<T>(0) {
        exprtk::ifunction<T>::allow_zero_parameters() = true;
    }
    T operator()() override { return time_ptr ? static_cast<T>(*time_ptr) : T(0); }
};

// Adapter for std::function-based custom functions (0-3 args). The
// user-supplied function name is stored so the warning message can
// identify the offender — define_function() copies it from the
// registration argument.
template <typename T> struct StdFunc0Adapter : public exprtk::ifunction<T> {
    std::function<double()> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc0Adapter(std::function<double()> f) : exprtk::ifunction<T>(0), fn(std::move(f)) {
        exprtk::ifunction<T>::allow_zero_parameters() = true;
    }
    T operator()() override {
        const double r = fn();
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct StdFunc1Adapter : public exprtk::ifunction<T> {
    std::function<double(double)> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc1Adapter(std::function<double(double)> f) : exprtk::ifunction<T>(1), fn(std::move(f)) {}
    T operator()(const T &x) override {
        const double dx = static_cast<double>(x);
        const double r = fn(dx);
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {dx}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct StdFunc2Adapter : public exprtk::ifunction<T> {
    std::function<double(double, double)> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc2Adapter(std::function<double(double, double)> f)
        : exprtk::ifunction<T>(2), fn(std::move(f)) {}
    T operator()(const T &x, const T &y) override {
        const double dx = static_cast<double>(x);
        const double dy = static_cast<double>(y);
        const double r = fn(dx, dy);
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {dx, dy}, r);
        }
        return static_cast<T>(r);
    }
};

template <typename T> struct StdFunc3Adapter : public exprtk::ifunction<T> {
    std::function<double(double, double, double)> fn;
    NonFiniteWarningSet *warner = nullptr;
    std::string fname;
    StdFunc3Adapter(std::function<double(double, double, double)> f)
        : exprtk::ifunction<T>(3), fn(std::move(f)) {}
    T operator()(const T &x, const T &y, const T &z) override {
        const double dx = static_cast<double>(x);
        const double dy = static_cast<double>(y);
        const double dz = static_cast<double>(z);
        const double r = fn(dx, dy, dz);
        if (warner) {
            warner->warn_if_nonfinite(fname.c_str(), {dx, dy, dz}, r);
        }
        return static_cast<T>(r);
    }
};

// ─── ExprTk implementation (pimpl) ───────────────────────────────────────────

// ─── Name remapping ──────────────────────────────────────────────────────────
//
// Two transparent transformations bridge BNG identifier conventions onto
// ExprTk's identifier rules:
//
//   1. Underscore-prefixed names: ExprTk rejects identifiers starting with
//      '_'. BNG built-ins use _pi, _e, _NA, etc. We map "_X" → "u_X" on
//      registration and rewrite the same tokens in compiled expressions.
//      This mapping is unconditional (no built-in ExprTk symbol starts with
//      '_'), so the rewrite is safe to apply token-by-token in expressions.
//
//   2. ExprTk reserved-word collisions: ExprTk rejects add_variable() for
//      names matching its reserved-word/reserved-symbol lists (e.g., a BNG
//      parameter literally named `const`, `true`, `false`). We register the
//      user's variable under "r_<name>" and rewrite references in compiled
//      expressions. Unlike (1) this rewrite is *conditional* — built-in
//      functions like `sin`, `if`, `time` must remain literal tokens — so
//      we only rewrite identifiers that were actually mangled at
//      registration. The mangling map is per-evaluator and lives on Impl.

static const std::unordered_set<std::string> &exprtk_reserved_identifiers() {
    static const std::unordered_set<std::string> reserved = [] {
        std::unordered_set<std::string> names;
        names.reserve(exprtk::details::reserved_words_size +
                      exprtk::details::reserved_symbols_size);
        for (std::size_t i = 0; i < exprtk::details::reserved_words_size; ++i) {
            names.insert(exprtk::details::reserved_words[i]);
        }
        for (std::size_t i = 0; i < exprtk::details::reserved_symbols_size; ++i) {
            names.insert(exprtk::details::reserved_symbols[i]);
        }
        return names;
    }();
    return reserved;
}

static const std::unordered_set<std::string> &bngsim_exprtk_aliases() {
    static const std::unordered_set<std::string> aliases = {"ln", "rint", "sign", "mratio", "time"};
    return aliases;
}

// Registration keys occupied by bngsim's built-in constants. init_builtins()
// registers each "_X" constant (Planck's `_h`, Avogadro's `_NA`, …) under the
// ExprTk key "u_X" because ExprTk rejects a leading '_'. A user parameter
// named literally "u_h" / "u_pi" / … maps to that same key (it does not start
// with '_', so compute_registration_name() would leave it unchanged) and
// would collide with the constant slot at registration (GH #90: Chitnis2012
// / BIOMD0000000950 has a parameter `u_h`). Treat these keys as reserved so
// such names take the r_ mangling path like any other reserved-word collision.
// Must stay in sync with the "_X" constants registered in init_builtins().
static const std::unordered_set<std::string> &bngsim_remapped_constant_keys() {
    static const std::unordered_set<std::string> keys = {"u_pi", "u_e", "u_kB", "u_NA",
                                                         "u_R",  "u_h", "u_F"};
    return keys;
}

bool expr_compat::is_exprtk_reserved(const std::string &name) {
    // Names that, if registered as a user variable, would collide with
    // a name already taken by the symbol table. Two sources:
    //
    //   1. ExprTk's current reserved_words[] + reserved_symbols[] lists,
    //      read directly from the vendored upstream header so bumps cannot
    //      silently drift away from our mangling assumptions.
    //
    //   2. bngsim-specific function aliases registered in
    //      ExprTkEvaluator::Impl::init_builtins() (`ln`, `rint`, `sign`,
    //      `mratio`, `time`). ExprTk would reject add_variable() on any
    //      of these once the function is registered, but the names are
    //      not on ExprTk's reserved_symbols[] list, so we have to track
    //      them here ourselves. BNG2.pl's parser already rejects
    //      `ln`/`rint`/`mratio`/`time` as parameter names upstream, so
    //      models reaching bngsim via `generate_network` can only hit
    //      the `sign` collision in practice — but we mangle all five
    //      for symmetry and to handle hand-crafted .net inputs.
    //
    //   3. The registration keys bngsim's built-in constants occupy after the
    //      unconditional "_X" → "u_X" remap (`u_pi`, …, `u_h`, `u_F`). A user
    //      parameter named literally `u_h` would otherwise alias Planck's
    //      constant slot and fail to register (GH #90).
    //
    // Comparison is exact (case-sensitive) because we build with
    // exprtk_disable_caseinsensitivity, so e.g. `Const` is not reserved.
    const auto &exprtk_reserved = exprtk_reserved_identifiers();
    const auto &bngsim_aliases = bngsim_exprtk_aliases();
    const auto &constant_keys = bngsim_remapped_constant_keys();
    return exprtk_reserved.find(name) != exprtk_reserved.end() ||
           bngsim_aliases.find(name) != bngsim_aliases.end() ||
           constant_keys.find(name) != constant_keys.end();
}

// Unconditional leading-underscore remap "_X" → "u_X" (see expr_compat.hpp).
std::string expr_compat::remap_name(const std::string &name) {
    if (!name.empty() && name[0] == '_') {
        return "u_" + name.substr(1);
    }
    return name;
}

// Compute the symbol-table key for `name` at registration time.
// Combines the unconditional underscore remap with reserved-word mangling.
std::string expr_compat::compute_registration_name(const std::string &name) {
    std::string underscore_mapped = remap_name(name);
    if (underscore_mapped != name) {
        return underscore_mapped;
    }
    if (is_exprtk_reserved(name)) {
        return "r_" + name;
    }
    return name;
}

struct ExprTkEvaluator::Impl {
    using SymbolTable = exprtk::symbol_table<double>;
    using Expression = exprtk::expression<double>;
    using Parser = exprtk::parser<double>;

    SymbolTable symbol_table;
    // Parser is shared across clones to avoid re-constructing the ~100KB
    // template object. The parser is stateless between compile() calls.
    std::shared_ptr<Parser> parser;
    std::vector<Expression> expressions;

    // Cached preprocessed expression strings for efficient clone().
    // Indexed in parallel with `expressions`.
    std::vector<std::string> preprocessed_strings;

    // Owned custom function objects (must outlive the symbol table)
    MratioFunction<double> mratio_func;
    LnFunction<double> ln_func;
    RintFunction<double> rint_func;
    SignFunction<double> sign_func;
    TgammaFunction<double> tgamma_func;
    TimeFunction<double> time_func;

    // User-registered custom functions (owned, heap-allocated)
    std::vector<std::unique_ptr<exprtk::ifunction<double>>> user_functions;

    // Diagnostic state: every custom-function adapter carries a back-pointer
    // to this set, so a function that returns nan/inf gets one warning on
    // stderr the first time a given (name, args) tuple misbehaves. Per
    // evaluator (no globals); not copied across clone_empty().
    NonFiniteWarningSet nonfinite_warner;

    // Names that were mangled at registration to avoid ExprTk reserved-word
    // collisions (key: original BNG name, value: ExprTk symbol-table key).
    // Underscore-prefixed names (e.g., _pi) are NOT recorded here — they
    // remap unconditionally via compute_registration_name() in both
    // directions, so no per-evaluator state is needed.
    std::unordered_map<std::string, std::string> mangled_user_names;

    // BNG-source names registered as scalar variables/constants on this
    // evaluator (i.e., everything that went through define_variable /
    // add_remapped_constant). Used by strip_empty_parens() to rewrite
    // `obs()` → `obs` for observable references that the BNG parser
    // accepts (Expression.pm:870-927 — Observable as zero-arg call) but
    // ExprTk's grammar would reject as `obs * ()`.
    std::unordered_set<std::string> scalar_variable_names;

    // Look up the symbol-table key for `name` when rewriting an expression.
    // Mirrors compute_registration_name() but only mangles reserved words
    // that were actually registered on this evaluator, so built-in tokens
    // (sin, if, time, ...) pass through unchanged.
    std::string remap_token(const std::string &name) const {
        std::string underscore_mapped = expr_compat::remap_name(name);
        if (underscore_mapped != name) {
            return underscore_mapped;
        }
        auto it = mangled_user_names.find(name);
        if (it != mangled_user_names.end()) {
            return it->second;
        }
        return name;
    }

    // Rewrite `name()` → `name` for any identifier registered as a scalar
    // variable on this evaluator. BNGL's grammar (per BNG2.pl's
    // bng2/Perl2/Expression.pm:870-927) accepts an Observable as a
    // zero-arg call (`obs()`), and BNG2.pl preserves that syntax verbatim
    // when emitting .net rate laws / parameter expressions / event
    // expressions. ExprTk's grammar would parse `obs()` as the implicit
    // multiplication `obs * ()` and reject the empty parens with ERR248.
    // We close the gap by stripping the trailing `()` for any name we
    // know is a scalar — leaving function names (built-ins like `sin`,
    // `time`, user-defined Func0/1/2/3) untouched, since those go
    // through add_function and are not in scalar_variable_names.
    std::string strip_empty_parens(const std::string &expr) const {
        if (scalar_variable_names.empty())
            return expr;
        std::string result;
        result.reserve(expr.size());
        size_t i = 0;
        while (i < expr.size()) {
            const bool at_boundary =
                (i == 0) ||
                (!std::isalnum(static_cast<unsigned char>(expr[i - 1])) && expr[i - 1] != '_');
            const bool ident_start =
                (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_');
            if (at_boundary && ident_start) {
                size_t start = i;
                i++;
                while (i < expr.size() &&
                       (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_')) {
                    i++;
                }
                std::string ident = expr.substr(start, i - start);
                if (i + 1 < expr.size() && expr[i] == '(' && expr[i + 1] == ')' &&
                    scalar_variable_names.count(ident)) {
                    result += ident;
                    i += 2;
                } else {
                    result += ident;
                }
                continue;
            }
            result += expr[i];
            i++;
        }
        return result;
    }

    // Rewrite all identifier tokens in `expr` through remap_token().
    std::string remap_expression(const std::string &expr) const {
        std::string result;
        result.reserve(expr.size() + 16);
        size_t i = 0;
        while (i < expr.size()) {
            const bool at_boundary =
                (i == 0) ||
                (!std::isalnum(static_cast<unsigned char>(expr[i - 1])) && expr[i - 1] != '_');
            const bool ident_start =
                (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_');
            if (at_boundary && ident_start) {
                size_t start = i;
                i++;
                while (i < expr.size() &&
                       (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_')) {
                    i++;
                }
                const std::string token = expr.substr(start, i - start);
                // A declared symbol that collides with an ExprTk reserved name
                // and is *also* used in call form (`frac(x)`) is genuinely
                // ambiguous in a single flat namespace — raise the same clear,
                // deterministic error the NFsim mu::Parser shim does, rather
                // than rewriting to r_<name> and letting ExprTk emit a cryptic
                // "not a function" message. strip_empty_parens() has already
                // removed legitimate zero-arg scalar calls (`obs()`), so any
                // `name(` left for a mangled symbol is a real call form. Only
                // reserved-mangled names are tracked in mangled_user_names;
                // underscore remaps are not, so they fall through untouched.
                if (mangled_user_names.count(token)) {
                    size_t j = i;
                    while (j < expr.size() && std::isspace(static_cast<unsigned char>(expr[j]))) {
                        j++;
                    }
                    if (j < expr.size() && expr[j] == '(') {
                        throw std::runtime_error(
                            "identifier '" + token +
                            "' is both a declared model "
                            "symbol and used as a function call '" +
                            token +
                            "(...)'; "
                            "ExprTk reserves '" +
                            token +
                            "' as a built-in and a single "
                            "flat namespace cannot hold both meanings — rename the "
                            "model symbol");
                    }
                }
                result += remap_token(token);
                continue;
            }
            result += expr[i];
            i++;
        }
        return result;
    }

    void add_remapped_constant(const std::string &name, double value) {
        std::string mapped = expr_compat::compute_registration_name(name);
        if (mapped != name && name[0] != '_') {
            mangled_user_names[name] = mapped;
        }
        symbol_table.add_constant(mapped, value);
        scalar_variable_names.insert(name);
    }

    void init_builtins() {
        // Register built-in constants
        // Note: ExprTk rejects '_' prefix, so we remap to "u_" internally
        add_remapped_constant("_pi", 3.14159265358979323846);
        add_remapped_constant("_e", 2.71828182845904523536);
        add_remapped_constant("_NA", 6.02214076e23);
        add_remapped_constant("_kB", 1.380649e-23);
        add_remapped_constant("_R", 8.314462618153241);
        add_remapped_constant("_h", 6.62607015e-34);
        add_remapped_constant("_F", 96485.33212331002);

        // Wire the non-finite-return diagnostic into every owned adapter.
        // time_func is intentionally omitted (see TimeFunction comment).
        mratio_func.warner = &nonfinite_warner;
        ln_func.warner = &nonfinite_warner;
        rint_func.warner = &nonfinite_warner;
        sign_func.warner = &nonfinite_warner;
        tgamma_func.warner = &nonfinite_warner;

        // Register backward-compatible aliases
        symbol_table.add_function("ln", ln_func);
        symbol_table.add_function("rint", rint_func);
        symbol_table.add_function("sign", sign_func);
        symbol_table.add_function("tgamma", tgamma_func);

        // Register built-in functions
        // Note: ExprTk has a built-in `if` keyword (grammar-level) that handles
        // if(cond, true_val, false_val) natively. No custom function needed.
        symbol_table.add_function("mratio", mratio_func);

        // time() — pointer set later via set_time_ptr().
        // We do NOT register `t` here so that `t` is free for use as a model
        // identifier (matches BNG2.pl convention).
        symbol_table.add_function("time", time_func);
    }

    Impl() : parser(std::make_shared<Parser>()) {
        // Increase max stack depth for deeply nested if() expressions.
        // ExprTk default is 400 (~200 nested if()), muParser handled 2000.
        parser->settings().set_max_stack_depth(4096);
        init_builtins();
    }

    // Constructor that shares an existing parser (for clone_empty)
    explicit Impl(std::shared_ptr<Parser> shared_parser) : parser(std::move(shared_parser)) {
        init_builtins();
    }

    void set_time_ptr(double *ptr) { time_func.time_ptr = ptr; }
};

ExprTkEvaluator::ExprTkEvaluator() : impl_(std::make_unique<Impl>()) {}

ExprTkEvaluator::~ExprTkEvaluator() = default;

void ExprTkEvaluator::define_variable(const std::string &name, double *addr) {
    std::string mapped = expr_compat::compute_registration_name(name);
    if (!impl_->symbol_table.add_variable(mapped, *addr)) {
        const bool reserved = expr_compat::is_exprtk_reserved(name);
        std::string detail;
        if (reserved) {
            // The mangled form already exists — most often because another
            // BNG name collides with the mangled key (e.g., user has both
            // `const` and a separate `r_const`).
            detail = "name '" + name + "' is an ExprTk reserved word; bngsim mangles it to '" +
                     mapped +
                     "', and that mangled key is already registered. Rename one of the "
                     "conflicting parameters.";
        } else {
            detail = "name '" + name + "' (mapped: '" + mapped +
                     "') is already registered. Check the .net file for duplicate "
                     "parameter / observable / species names (case-sensitive).";
        }
        throw std::runtime_error("ExprTk: failed to register variable '" + name + "'. " + detail);
    }
    if (mapped != name && (name.empty() || name[0] != '_')) {
        impl_->mangled_user_names[name] = mapped;
    }
    impl_->scalar_variable_names.insert(name);
}

void ExprTkEvaluator::define_constant(const std::string &name, double value) {
    impl_->add_remapped_constant(name, value);
}

void ExprTkEvaluator::define_function(const std::string &name, Func0 fn) {
    auto adapter = std::make_unique<StdFunc0Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func1 fn) {
    auto adapter = std::make_unique<StdFunc1Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func2 fn) {
    auto adapter = std::make_unique<StdFunc2Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

void ExprTkEvaluator::define_function(const std::string &name, Func3 fn) {
    auto adapter = std::make_unique<StdFunc3Adapter<double>>(std::move(fn));
    adapter->warner = &impl_->nonfinite_warner;
    adapter->fname = name;
    impl_->symbol_table.add_function(name, *adapter);
    impl_->user_functions.push_back(std::move(adapter));
}

// ─── C-style logical operator replacement ────────────────────────────────────
// BNG2.pl emits C-style && and || in if() conditions, but ExprTk uses
// 'and' and 'or' keywords. Replace before compilation.
static std::string replace_logical_operators(const std::string &expr) {
    std::string result;
    result.reserve(expr.size() + 16);
    for (size_t i = 0; i < expr.size(); ++i) {
        if (i + 1 < expr.size()) {
            if (expr[i] == '&' && expr[i + 1] == '&') {
                result += " and ";
                i++; // skip second '&'
                continue;
            }
            if (expr[i] == '|' && expr[i + 1] == '|') {
                result += " or ";
                i++; // skip second '|'
                continue;
            }
        }
        result += expr[i];
    }
    return result;
}

int ExprTkEvaluator::compile(const std::string &expr) {
    // Replace C-style logical operators before any other processing
    std::string preprocessed = replace_logical_operators(expr);

    // Strip `obs()` → `obs` for any name registered as a scalar variable.
    // BNGL accepts Observable references as zero-arg calls; ExprTk does
    // not. Run before remap_expression so we match against BNG-source
    // names, not their post-mangling forms.
    std::string stripped = impl_->strip_empty_parens(preprocessed);

    // Remap identifiers before ExprTk compilation:
    //   - Unconditional: "_X" → "u_X" (ExprTk rejects '_' prefix)
    //   - Conditional:   reserved-word names registered on this evaluator
    //                    (e.g., user's `const` → `r_const`)
    std::string remapped = impl_->remap_expression(stripped);

    // Delegate to compile_preprocessed (which also caches the string)
    return compile_preprocessed(remapped);
}

int ExprTkEvaluator::compile_preprocessed(const std::string &preprocessed_expr) {
    Impl::Expression expression;
    expression.register_symbol_table(impl_->symbol_table);

    if (!impl_->parser->compile(preprocessed_expr, expression)) {
        throw std::runtime_error("ExprTk compilation failed for expression: '" + preprocessed_expr +
                                 "' — " + impl_->parser->error());
    }

    int id = static_cast<int>(impl_->expressions.size());
    impl_->expressions.push_back(std::move(expression));
    impl_->preprocessed_strings.push_back(preprocessed_expr);
    return id;
}

const std::string &ExprTkEvaluator::preprocessed_expr(int expr_id) const {
    if (expr_id < 0 || expr_id >= static_cast<int>(impl_->preprocessed_strings.size())) {
        throw std::runtime_error("Invalid expression ID: " + std::to_string(expr_id));
    }
    return impl_->preprocessed_strings[expr_id];
}

std::vector<const double *> ExprTkEvaluator::referenced_variable_addresses(int expr_id) const {
    std::vector<const double *> out;
    if (expr_id < 0 || expr_id >= static_cast<int>(impl_->preprocessed_strings.size())) {
        return out;
    }
    // Collect the variable identifiers referenced by the (already remapped)
    // preprocessed string, then resolve each through this evaluator's symbol
    // table to the address it was bound to via define_variable. Names not
    // registered as variables (constants, functions) resolve to null and are
    // skipped, so the result contains only model-variable addresses.
    std::vector<std::string> names;
    // Pass the symbol table so the collector resolves built-in/user functions
    // (e.g. time()) instead of bailing out when it meets an unknown token — the
    // symbol-table-less overload returns nothing for a trigger like
    // `time() >= t_dose`, which would silently hide a parameter reference.
    if (!exprtk::collect_variables(impl_->preprocessed_strings[expr_id], impl_->symbol_table,
                                   names)) {
        return out;
    }
    out.reserve(names.size());
    for (const std::string &nm : names) {
        auto *var = impl_->symbol_table.get_variable(nm);
        if (var != nullptr) {
            out.push_back(&var->ref());
        }
    }
    return out;
}

int ExprTkEvaluator::n_expressions() const { return static_cast<int>(impl_->expressions.size()); }

double ExprTkEvaluator::evaluate(int expr_id) {
    if (expr_id < 0 || expr_id >= static_cast<int>(impl_->expressions.size())) {
        throw std::runtime_error("Invalid expression ID: " + std::to_string(expr_id));
    }
    return impl_->expressions[expr_id].value();
}

void ExprTkEvaluator::set_time_ptr(double *time_addr) { impl_->set_time_ptr(time_addr); }

// ─── Efficient clone support ─────────────────────────────────────────────────

ExprTkEvaluator::ExprTkEvaluator(std::shared_ptr<void> shared_parser)
    : impl_(std::make_unique<Impl>(std::static_pointer_cast<Impl::Parser>(shared_parser))) {}

ExprTkEvaluator::ExprTkEvaluator(ExprTkEvaluator &&) noexcept = default;
ExprTkEvaluator &ExprTkEvaluator::operator=(ExprTkEvaluator &&) noexcept = default;

std::unique_ptr<ExprTkEvaluator> ExprTkEvaluator::clone_empty() const {
    // Share the parser (heavyweight, stateless between calls)
    // but create a fresh symbol table and expression list.
    return std::unique_ptr<ExprTkEvaluator>(new ExprTkEvaluator(impl_->parser));
}

// ─── Reserved names ──────────────────────────────────────────────────────────

ReservedNames reserved_names() {
    ReservedNames names;
    names.constants = {"_pi", "_e", "_kB", "_NA", "_R", "_h", "_F"};
    names.functions = {"time",  "sin",   "cos",   "tan",    "asin",  "acos", "atan",  "sinh",
                       "cosh",  "tanh",  "asinh", "acosh",  "atanh", "exp",  "log",   "ln",
                       "log2",  "log10", "sqrt",  "abs",    "floor", "ceil", "round", "rint",
                       "trunc", "min",   "max",   "clamp",  "avg",   "sum",  "erf",   "erfc",
                       "sign",  "sgn",   "if",    "mratio", "tgamma"};
    return names;
}

} // namespace bngsim
