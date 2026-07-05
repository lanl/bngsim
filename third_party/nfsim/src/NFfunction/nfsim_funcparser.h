// nfsim_funcparser.h — ExprTk-based drop-in replacement for mu::Parser in NFsim
//
// Maintained by bngsim to keep expression handling aligned between the
// ODE/SSA engines and the vendored NFsim runtime.
//
// This header provides a `mu::Parser` class with the same API surface as muParser
// but backed by ExprTk. All existing NFsim code that uses `mu::Parser` works without
// changes to variable names, namespace references, or calling patterns.
//
// Key design decisions:
// - "Constants" are stored as mutable variables in internal storage, so
//   DefineConst() can be called after SetExpr() (used by updateParameters/fileUpdate).
// - ExprTk's `log` is natural log (matching bngsim). BNG's `ln` is aliased to natural log.
// - Built-in functions match bngsim's expression.cpp: if(), ln(), rint(), sign(), etc.
// - Constants: _PI, _e, _Na (NFsim convention) plus bngsim's _pi, _kB, _NA, _R, _h, _F.
//
// SINGLE SOURCE OF TRUTH (issue #49): the BNG-compatibility logic this shim needs
// — the mratio() built-in and the BNG-identifier → ExprTk-symbol mapping (the
// unconditional leading-underscore remap and the conditional reserved-symbol
// mangle) — is owned by the host evaluator and exposed via
// <bngsim/expr_compat.hpp>. This shim FORWARDS to those definitions instead of
// carrying a hand-ported copy, so the logic lives in exactly one place (the
// bngsim::expression archive that the vendored NFsim target links). The shim
// stays a thin mu::Parser → ExprTk adapter; see issue #49 and bngsim ADR-005
// (bngsim/dev/adr) for the rationale.
//
// IMPORTANT: two transparent transformations bridge BNG identifier conventions
// onto ExprTk's identifier rules. The caller (NFsim) never sees either one —
// it continues to DefineConst()/DefineVar()/SetExpr() with the original BNG names.
//
//   1. Underscore-prefixed names: ExprTk rejects identifiers starting with '_'.
//      NFsim uses _PI, _e, _Na as constant names, and BNG XML files may define
//      parameters with leading underscores (e.g., __FREE parameters in PyBNF,
//      __TFUN_VAL__ placeholders for time-dependent functions). The host remaps
//      "_X" → "u_X" in both symbol registration and expression preprocessing.
//      This mapping is unconditional (no built-in ExprTk symbol starts with
//      '_'), so the rewrite is safe to apply token-by-token in expressions.
//
//   2. ExprTk reserved-symbol collisions: ExprTk reserves a large built-in
//      name set (frac, min, max, sum, avg, mod, root, hypot, erf, …) and
//      rejects add_variable() for any of them. muParser had a much smaller
//      reserved set, so legacy BNG2.pl→NFsim models that declare a parameter /
//      observable / function whose name equals one of these (e.g. a parameter
//      literally named `frac`) silently failed to register here. The host
//      registers the user's symbol under "r_<name>"; this shim rewrites
//      references to it in compiled expressions. Unlike (1) this rewrite is
//      *conditional* — a genuine built-in call like `frac(x)` in a model that
//      never declared a `frac` symbol must stay literal — so we only rewrite
//      identifiers that were actually mangled at registration. A single flat
//      namespace cannot hold both meanings of a name: if a declared symbol is
//      *also* used in call form (`frac(...)`), SetExpr() raises a clear,
//      deterministic error rather than silently resolving to the built-in. This
//      mirrors — and now shares one source with — the ODE engine's reserved-word
//      handling in bngsim/src/expression.cpp.
//
#ifndef NFSIM_FUNCPARSER_H_
#define NFSIM_FUNCPARSER_H_

// ExprTk compilation options — disable features we don't need.
#define exprtk_disable_string_capabilities
#define exprtk_disable_rtl_io_file
#define exprtk_disable_rtl_vecops
// BNG is case-sensitive for parameter names (e.g., k3 ≠ K3).
// ExprTk defaults to case-insensitive, which silently merges k3/K3.
#define exprtk_disable_caseinsensitivity
#include "exprtk.hpp"

// Host single source of truth for mratio() + the BNG↔ExprTk identifier mapping
// (issue #49). Linked into the vendored NFsim target via bngsim::expression.
#include "bngsim/expr_compat.hpp"

#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mu {

// ─── Custom ExprTk function adapters ─────────────────────────────────────────

namespace detail {

// 1-arg: ln(x) — natural logarithm (backward-compat alias)
template <typename T>
struct LnFunction : public exprtk::ifunction<T> {
    LnFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T& x) override { return std::log(x); }
};

// 1-arg: rint(x) — round to nearest integer
template <typename T>
struct RintFunction : public exprtk::ifunction<T> {
    RintFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T& x) override { return std::round(x); }
};

// 1-arg: sign(x) — signum function
template <typename T>
struct SignFunction : public exprtk::ifunction<T> {
    SignFunction() : exprtk::ifunction<T>(1) {}
    T operator()(const T& x) override {
        return (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0));
    }
};

// 3-arg: mratio(a, b, z) — BNGL built-in confluent hypergeometric ratio.
// Forwards to the host single source bngsim::expr_compat::mratio (issue #49),
// so the NFsim shim and the ODE/SSA engine evaluate it from one implementation.
template <typename T>
struct MratioFunction : public exprtk::ifunction<T> {
    MratioFunction() : exprtk::ifunction<T>(3) {
        exprtk::ifunction<T>::allow_zero_parameters() = false;
    }
    T operator()(const T& a, const T& b, const T& z) override {
        return static_cast<T>(bngsim::expr_compat::mratio(static_cast<double>(a),
                                                          static_cast<double>(b),
                                                          static_cast<double>(z)));
    }
};

}  // namespace detail

// ─── mu::Parser — ExprTk-based drop-in replacement ──────────────────────────

class Parser {
public:
    struct exception_type : public std::runtime_error {
        exception_type(const std::string& msg) : std::runtime_error(msg) {}
        std::string GetMsg() const { return what(); }
    };

    Parser()
        : compiled_(false)
    {
        // Register built-in constants (NFsim convention: _PI, _e, _Na)
        DefineConst("_PI", 3.14159265358979323846);
        DefineConst("_e",  2.71828182845904523536);
        DefineConst("_Na", 6.02214076e23);

        // Additional constants supported by bngsim: _pi, _NA, _kB, _R, _h, _F
        DefineConst("_pi", 3.14159265358979323846);
        DefineConst("_NA", 6.02214076e23);
        DefineConst("_kB", 1.380649e-23);
        DefineConst("_R",  8.314462618153241);
        DefineConst("_h",  6.62607015e-34);
        DefineConst("_F",  96485.33212331002);

        // Register backward-compatible aliases. These names are reserved by the
        // host (bngsim::expr_compat::is_exprtk_reserved), so a model symbol that
        // collides with one is mangled at registration just like `frac`.
        symbol_table_.add_function("ln",     ln_func_);
        symbol_table_.add_function("rint",   rint_func_);
        symbol_table_.add_function("sign",   sign_func_);
        symbol_table_.add_function("mratio", mratio_func_);
    }

    ~Parser() = default;

    // Non-copyable (symbol_table holds pointers to internal storage)
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    // ─── DefineVar: bind a variable name to an external double* ─────────
    void DefineVar(const std::string& name, double* ptr) {
        std::string mapped = registerName(name);
        symbol_table_.add_variable(mapped, *ptr);
        if (compiled_) {
            recompile();
        }
    }

    // ─── DefineConst: store value internally, register as variable ──────
    void DefineConst(const std::string& name, double value) {
        std::string mapped = registerName(name);
        auto it = const_storage_.find(mapped);
        if (it != const_storage_.end()) {
            *(it->second) = value;
        } else {
            auto ptr = std::make_unique<double>(value);
            double* raw = ptr.get();
            const_storage_[mapped] = std::move(ptr);
            symbol_table_.add_variable(mapped, *raw);
        }
    }

    // ─── SetExpr: compile the expression ────────────────────────────────
    void SetExpr(const std::string& expr) {
        original_expr_string_ = expr;
        compile();
    }

    // ─── Eval: evaluate the compiled expression ─────────────────────────
    double Eval() {
        if (!compiled_) {
            throw exception_type("Expression not compiled (call SetExpr first)");
        }
        return expression_.value();
    }

    // ─── GetExpr: return the ORIGINAL expression string ─────────────────
    std::string GetExpr() const {
        return original_expr_string_;
    }

private:
    // Compute the symbol-table key for `name` at registration time and record
    // any reserved-symbol mangling so expressions can be rewritten to match.
    // The registration-key decision is forwarded to the host single source
    // (bngsim::expr_compat::compute_registration_name), which combines the
    // unconditional underscore remap with the conditional reserved-symbol
    // mangle. Only reserved-symbol mangles are recorded here for
    // remapExpression(); underscore-prefixed names remap unconditionally and
    // need no per-name state.
    std::string registerName(const std::string& name) {
        std::string mapped = bngsim::expr_compat::compute_registration_name(name);
        if (mapped == name) {
            return name;  // neither underscore- nor reserved-mangled
        }
        if (!name.empty() && name[0] == '_') {
            return mapped;  // unconditional underscore remap — no state to track
        }
        mangled_reserved_[name] = mapped;  // reserved-symbol mangle: record "r_<name>"
        return mapped;
    }

    // Rewrite all identifier tokens in `expr` to match registration:
    //   - "_X"            → "u_X"  (unconditional, via host remap_name)
    //   - declared symbol → "r_X"  (only names actually mangled at registration)
    //   - everything else → unchanged (genuine built-ins, function/local refs)
    //
    // A declared symbol that collides with a reserved name and is used in call
    // form (`frac(...)`) is genuinely ambiguous in a single flat namespace; we
    // raise a clear error rather than silently resolving to the ExprTk built-in.
    std::string remapExpression(const std::string& expr) const {
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
                std::string token = expr.substr(start, i - start);
                if (!token.empty() && token[0] == '_') {
                    result += bngsim::expr_compat::remap_name(token);
                    continue;
                }
                auto it = mangled_reserved_.find(token);
                if (it != mangled_reserved_.end()) {
                    // Peek past whitespace: a declared symbol used in call form
                    // cannot coexist with the same-named ExprTk built-in.
                    size_t j = i;
                    while (j < expr.size() &&
                           std::isspace(static_cast<unsigned char>(expr[j]))) {
                        j++;
                    }
                    if (j < expr.size() && expr[j] == '(') {
                        throw exception_type(
                            "identifier '" + token + "' is both a declared model "
                            "symbol and used as a function call '" + token + "(...)'; "
                            "ExprTk reserves '" + token + "' as a built-in and a single "
                            "flat namespace cannot hold both meanings — rename the "
                            "model symbol");
                    }
                    result += it->second;
                    continue;
                }
                result += token;
                continue;
            }
            result += expr[i];
            i++;
        }
        return result;
    }

    void compile() {
        // Recompute the remapped expression from the original each time so that
        // any symbol registered before this (re)compile is reflected.
        expr_string_ = remapExpression(original_expr_string_);

        expression_ = exprtk::expression<double>();  // reset
        expression_.register_symbol_table(symbol_table_);

        exprtk::parser<double> parser;
        // Increase max stack depth for deeply nested if() expressions.
        // ExprTk default is 400 (~200 nested if()), muParser handled 2000.
        parser.settings().set_max_stack_depth(4096);
        if (!parser.compile(expr_string_, expression_)) {
            throw exception_type(
                "ExprTk compilation failed for '" + original_expr_string_ +
                "' (remapped: '" + expr_string_ + "'): " +
                parser.error());
        }
        compiled_ = true;
    }

    void recompile() {
        if (!original_expr_string_.empty()) {
            compile();
        }
    }

    // ExprTk objects
    exprtk::symbol_table<double> symbol_table_;
    exprtk::expression<double> expression_;
    std::string expr_string_;           // remapped expression (what ExprTk sees)
    std::string original_expr_string_;  // original expression (what caller sees)
    bool compiled_;

    // Internal storage for "constants" (mutable via DefineConst after SetExpr)
    std::unordered_map<std::string, std::unique_ptr<double>> const_storage_;

    // Names mangled at registration to dodge ExprTk reserved-symbol collisions
    // (key: original BNG name, value: "r_<name>" symbol-table key). Used by
    // remapExpression() to rewrite only the names actually registered, leaving
    // genuine built-in tokens untouched. Underscore-prefixed names are NOT
    // recorded here — they remap unconditionally in both directions.
    std::unordered_map<std::string, std::string> mangled_reserved_;

    // Custom function objects (must outlive symbol_table_)
    detail::LnFunction<double> ln_func_;
    detail::RintFunction<double> rint_func_;
    detail::SignFunction<double> sign_func_;
    detail::MratioFunction<double> mratio_func_;
};

}  // namespace mu

#endif  // NFSIM_FUNCPARSER_H_
