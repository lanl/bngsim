// bngsim/include/bngsim/expression.hpp — Pluggable expression evaluator interface
//
// Defines the expression-evaluator interface and the default ExprTk backend.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bngsim {

// ─── Abstract expression evaluator interface ─────────────────────────────────
class ExpressionEvaluator {
  public:
    virtual ~ExpressionEvaluator() = default;

    // Bind a named variable to a memory location (updated externally)
    virtual void define_variable(const std::string &name, double *addr) = 0;

    // Define a named constant
    virtual void define_constant(const std::string &name, double value) = 0;

    // Register a user function with 0-3 args
    using Func0 = std::function<double()>;
    using Func1 = std::function<double(double)>;
    using Func2 = std::function<double(double, double)>;
    using Func3 = std::function<double(double, double, double)>;

    virtual void define_function(const std::string &name, Func0 fn) = 0;
    virtual void define_function(const std::string &name, Func1 fn) = 0;
    virtual void define_function(const std::string &name, Func2 fn) = 0;
    virtual void define_function(const std::string &name, Func3 fn) = 0;

    // Compile an expression string. Returns an ID for later evaluation.
    // Throws std::runtime_error on parse failure.
    virtual int compile(const std::string &expr) = 0;

    // Evaluate a previously compiled expression by ID.
    virtual double evaluate(int expr_id) = 0;

    // Return the addresses of the bound model variables (species
    // concentrations / parameter values registered via define_variable) that a
    // compiled expression references. Used to classify event triggers for
    // forward-sensitivity support (GH #212): a trigger referencing a species is
    // state-dependent, and one referencing a requested sensitivity parameter
    // has a parameter-dependent crossing time — both are beyond Phase-1
    // (fixed-time) event sensitivity and keep raising. Constants and built-in
    // functions (e.g. time()) are not variables and are not reported.
    virtual std::vector<const double *> referenced_variable_addresses(int expr_id) const = 0;
};

// ─── ExprTk-based implementation ─────────────────────────────────────────────
//
// Registers BNG-compatible aliases:
//   ln → log, rint → round, sign → sgn
// Registers built-in constants:
//   _pi, _e, _kB, _NA, _R, _h, _F
// Registers built-in functions:
//   if(cond, true_val, false_val), mratio(a, b, z), time()
// Note: `t` is intentionally NOT registered (BNG2.pl uses time() only),
// leaving `t` free as a model identifier.
class ExprTkEvaluator : public ExpressionEvaluator {
  public:
    ExprTkEvaluator();
    ~ExprTkEvaluator() override;

    // Non-copyable, movable
    ExprTkEvaluator(const ExprTkEvaluator &) = delete;
    ExprTkEvaluator &operator=(const ExprTkEvaluator &) = delete;
    ExprTkEvaluator(ExprTkEvaluator &&) noexcept;
    ExprTkEvaluator &operator=(ExprTkEvaluator &&) noexcept;

    void define_variable(const std::string &name, double *addr) override;
    void define_constant(const std::string &name, double value) override;

    void define_function(const std::string &name, Func0 fn) override;
    void define_function(const std::string &name, Func1 fn) override;
    void define_function(const std::string &name, Func2 fn) override;
    void define_function(const std::string &name, Func3 fn) override;

    int compile(const std::string &expr) override;
    double evaluate(int expr_id) override;
    std::vector<const double *> referenced_variable_addresses(int expr_id) const override;

    // Bind the time() and t() functions to a double* that tracks simulation time.
    // Must be called after construction, before compiling time-dependent expressions.
    void set_time_ptr(double *time_addr);

    // ─── Efficient clone support ─────────────────────────────────────────
    //
    // clone_empty() creates a new evaluator that shares the heavyweight
    // ExprTk parser with this evaluator, but has its own symbol table and
    // expression list. The caller must register variables and compile
    // expressions on the returned evaluator.
    //
    // This avoids re-constructing the ~100KB exprtk::parser<double>
    // template object on every model clone.
    std::unique_ptr<ExprTkEvaluator> clone_empty() const;

    // compile_preprocessed() compiles an already-preprocessed expression
    // (logical operators and underscore names already remapped). Skips the
    // preprocessing steps that compile() performs.
    int compile_preprocessed(const std::string &preprocessed_expr);

    // Return the cached preprocessed string for a compiled expression.
    // Useful for model clone: cache strings at build time, re-compile
    // from cached strings on clone.
    const std::string &preprocessed_expr(int expr_id) const;

    // Number of compiled expressions.
    int n_expressions() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Private constructor for clone_empty() — shares parser
    explicit ExprTkEvaluator(std::shared_ptr<void> shared_parser);
};

// ─── Reserved names introspection ────────────────────────────────────────────
struct ReservedNames {
    std::vector<std::string> constants;
    std::vector<std::string> functions;
};

ReservedNames reserved_names();

} // namespace bngsim
