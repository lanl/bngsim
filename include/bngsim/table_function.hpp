// bngsim/include/bngsim/table_function.hpp — Table function (tfun) support
//
// Piecewise-linear or step interpolation of tabular data.
// Registered as a zero-argument ExprTk function with constant extrapolation
// beyond the endpoints. Includes step interpolation compatible with BioNetGen
// tfun syntax.

#pragma once

#include <cctype>
#include <string>
#include <vector>

namespace bngsim {

/// Strip a trailing "()" from a tfun index/header token, if present.
/// Mirrors BNG's `TfunReader.pm:84` (`$idx_col =~ s/\(\)$//`): `.tfun`
/// header columns and index names may be written either as bare identifiers
/// (`time`, `myParam`) or with empty parens (`time()`, `myParam()`).
inline std::string strip_paren_suffix(const std::string &name) {
    if (name.size() >= 2 && name.compare(name.size() - 2, 2, "()") == 0) {
        return name.substr(0, name.size() - 2);
    }
    return name;
}

/// Case-insensitive ASCII equality on two strings.
inline bool iequals_ascii(const std::string &a, const std::string &b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    }
    return true;
}

/// True iff `name` denotes the time index. Matches BNG's
/// `TfunReader.pm:87-88` semantics: case-insensitive `time`/`t` after
/// stripping a trailing "()" suffix. So `time`, `T`, `Time()`, `t()` all
/// canonicalize to the time index.
inline bool is_time_index(const std::string &name) {
    std::string n = strip_paren_suffix(name);
    return iequals_ascii(n, "time") || iequals_ascii(n, "t");
}

/// Interpolation method for table functions.
/// Matches BioNetGen tfun-dev branch TfunMethod enum.
enum class InterpolationMethod {
    Linear, ///< Piecewise-linear interpolation (default)
    Step    ///< Step function (left-continuous: value at left endpoint)
};

/// A table function: sorted (x, y) pairs with interpolation.
///
/// The index pointer (`index_ptr_`) points to the variable whose current value
/// provides the lookup key (e.g., `&model.current_time` for time-indexed tfuns,
/// or `&parameter.value` for parameter-indexed tfuns).
///
/// Registered as a zero-argument ExprTk function: when evaluated, it reads
/// `*index_ptr_`, performs binary search + interpolation, and returns
/// the interpolated value.
class TableFunction {
  public:
    /// Construct from parallel vectors of x and y values.
    /// @param name       Function name (e.g., "cumNcases")
    /// @param xs         Sorted (monotonically increasing) x values
    /// @param ys         Corresponding y values
    /// @param index_name Name of the index variable (e.g., "time", "drug_conc")
    /// @param method     Interpolation method (Linear or Step)
    /// @throws std::runtime_error if xs.size() != ys.size(), xs.size() < 2,
    ///         or xs is not monotonically increasing
    TableFunction(const std::string &name, std::vector<double> xs, std::vector<double> ys,
                  const std::string &index_name = "time",
                  InterpolationMethod method = InterpolationMethod::Linear);

    /// Evaluate the table function at the current value of *index_ptr_.
    /// Binary search + interpolation (linear or step).
    /// Constant extrapolation beyond endpoints.
    double evaluate() const;

    /// Evaluate at a caller-supplied index value, bypassing index_ptr_.
    /// Used by the codegen RHS path, where the index variable (time,
    /// parameter, or observable) is already a local in the generated C
    /// and we don't want to round-trip through the model's mutable state.
    double evaluate_at(double x) const;

    /// Set the pointer to the index variable.
    /// Must be called before evaluate(). The pointed-to double must remain
    /// valid for the lifetime of this TableFunction.
    void set_index_ptr(double *ptr) { index_ptr_ = ptr; }

    /// Get the pointer to the index variable (may be nullptr before binding).
    double *index_ptr() const { return index_ptr_; }

    // ─── Accessors ───────────────────────────────────────────────────────

    const std::string &name() const { return name_; }
    const std::string &index_name() const { return index_name_; }
    const std::vector<double> &xs() const { return xs_; }
    const std::vector<double> &ys() const { return ys_; }
    int size() const { return static_cast<int>(xs_.size()); }
    InterpolationMethod method() const { return method_; }

    /// Load a .tfun file and create a TableFunction.
    /// File format (GDAT-style):
    ///   # index_col_header  value_col_header
    ///   x0  y0
    ///   x1  y1
    ///   ...
    ///
    /// @param name        Runtime function name (the TableFunction's internal
    ///                    identifier; used to build the ExprTk callable name)
    /// @param filepath    Path to .tfun file
    /// @param index_name  Expected index name (must match column 1 header, or
    ///                    "time"/"t"/"time()"/"t()" for time-indexed)
    /// @param method      Interpolation method (Linear or Step)
    /// @param header_name Expected value-column header in the .tfun file. Defaults
    ///                    to `name`. Distinct from `name` only when the runtime
    ///                    table is registered under a synthetic ID (wrapper-form
    ///                    `(tfun('drive') + ...)/k`) while the .tfun file's
    ///                    column-2 header still records the original BNG function
    ///                    name.
    /// @throws std::runtime_error on parse errors
    static TableFunction from_file(const std::string &name, const std::string &filepath,
                                   const std::string &index_name = "time",
                                   InterpolationMethod method = InterpolationMethod::Linear,
                                   const std::string &header_name = "");

  private:
    std::string name_;
    std::string index_name_;
    std::vector<double> xs_;
    std::vector<double> ys_;
    InterpolationMethod method_ = InterpolationMethod::Linear;
    double *index_ptr_ = nullptr;
};

} // namespace bngsim
