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
//
// ─── THREAD SAFETY (issues #201, #257) ───────────────────────────────────────
//
// An ExprTkEvaluator owns ALL of its ExprTk state and shares NONE of it with any
// other evaluator — parser, symbol table, expression list, function adapters.
// That is a deliberate design decision, not an accident of implementation, and
// `parser_identity()` exists so it can be asserted rather than merely believed.
//
// The evaluator itself is *not* internally synchronized: it is per-model-instance
// state, and a NetworkModel is not thread-safe (every fan-out clones — see the
// THREAD SAFETY note in model.cpp and
// python/tests/test_parallel_workers_clone_the_model.py). One thread per
// evaluator is the contract; concurrent compile()/evaluate() on ONE evaluator is
// a caller bug that no lock in here would make correct anyway, because
// compile() grows `expressions` under an evaluate() that is reading it.
//
// What made the shared parser unsalvageable rather than merely lock-hungry:
// exprtk::parser::compile() ends with
//     symtab_store_.symtab_list_ = expr.get_symbol_table_list();
// and never clears it, so a parser keeps a *strong handle* on the symbol table of
// the last expression compiled through it — and exprtk's symbol_table refcount is
// a plain std::size_t. With one parser behind two evaluators, thread B's next
// compile drops thread A's symbol table while thread A is still incrementing and
// decrementing that same counter outside any lock (register_symbol_table, growth
// of the expression vector, destruction). A lost update frees the symbol table's
// data underneath the variable pointers already baked into A's compiled nodes.
// #201's mutex serialized compile() and could not have covered that, which is why
// the regression test it added stayed ~10% flaky until #257.
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

    // ─── Clone support ───────────────────────────────────────────────────
    //
    // clone_empty() creates a new evaluator that shares NOTHING with this one:
    // its own parser, symbol table, expression list and function adapters. The
    // caller must register variables and compile expressions on the returned
    // evaluator (NetworkModel::clone() does, from the cached preprocessed
    // strings this one exposes via preprocessed_expr()).
    //
    // It used to hand the new evaluator this one's exprtk::parser<double>, to
    // save re-constructing the ~100KB template object per model clone. Issue
    // #257 retired that: the parser retains a strong handle on the last symbol
    // table compiled through it (see the class comment), so sharing it couples
    // the lifetime of one evaluator's symbol table to another evaluator's next
    // compile, across threads, through a non-atomic refcount.
    //
    // The saving was illusory in any case. `NetworkModel copy;` default-
    // constructs an evaluator before clone() overwrites it, so the "avoided"
    // parser construction was being paid and thrown away on every clone. The
    // parser is now built lazily on first compile, which means that discarded
    // evaluator never builds one: measured, a clone pays the same single ~51 µs
    // parser construction it always did, and a model with no expressions at all
    // now pays none.
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

    // Identity of this evaluator's exprtk::parser, or nullptr if it has not
    // compiled anything yet (the parser is constructed lazily on first
    // compile). Two distinct evaluators must never report the same non-null
    // value — that is the whole of the issue #257 invariant, and this accessor
    // is what makes it testable instead of merely documented. Not for
    // dereferencing: the value is an identity, and the type says so.
    const void *parser_identity() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ─── Reserved names introspection ────────────────────────────────────────────
struct ReservedNames {
    std::vector<std::string> constants;
    std::vector<std::string> functions;
};

ReservedNames reserved_names();

} // namespace bngsim
