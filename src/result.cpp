// bngsim/src/result.cpp — Simulation result container implementation

#include "bngsim/result.hpp"

#include "bngsim/function_columns.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace bngsim {

Result::Result() = default;
Result::~Result() = default;

void Result::allocate(int n_times, int n_species, int n_observables) {
    n_times_ = n_times;
    n_species_ = n_species;
    n_observables_ = n_observables;

    time_.resize(n_times, 0.0);
    species_.resize(n_times * n_species, 0.0);
    observables_.resize(n_times * n_observables, 0.0);
}

void Result::record(int time_index, double t, const double *species_conc,
                    const double *observable_vals) {
    if (time_index < 0 || time_index >= n_times_) {
        throw std::out_of_range("time_index out of range in Result::record");
    }

    time_[time_index] = t;

    if (species_conc && n_species_ > 0) {
        std::memcpy(&species_[time_index * n_species_], species_conc, n_species_ * sizeof(double));
    }
    if (observable_vals && n_observables_ > 0) {
        std::memcpy(&observables_[time_index * n_observables_], observable_vals,
                    n_observables_ * sizeof(double));
    }
}

void Result::record_expressions(int time_index, const double *expr_vals) {
    if (time_index < 0 || time_index >= n_times_ || n_expressions_ == 0)
        return;
    if (expr_vals) {
        std::memcpy(&expressions_[time_index * n_expressions_], expr_vals,
                    n_expressions_ * sizeof(double));
    }
}

int Result::n_times() const { return n_times_; }
int Result::n_species() const { return n_species_; }
int Result::n_observables() const { return n_observables_; }

const std::vector<double> &Result::time() const { return time_; }
const std::vector<double> &Result::species_data() const { return species_; }
const std::vector<double> &Result::observable_data() const { return observables_; }
const std::vector<double> &Result::expression_data() const { return expressions_; }
int Result::n_expressions() const { return n_expressions_; }

void Result::set_species_names(const std::vector<std::string> &names) { species_names_ = names; }
void Result::set_observable_names(const std::vector<std::string> &names) {
    observable_names_ = names;
}
void Result::set_expression_names(const std::vector<std::string> &names) {
    expression_names_ = names;
    n_expressions_ = static_cast<int>(names.size());
    if (n_times_ > 0 && n_expressions_ > 0) {
        expressions_.resize(n_times_ * n_expressions_, 0.0);
    }
}
const std::vector<std::string> &Result::species_names() const { return species_names_; }
const std::vector<std::string> &Result::observable_names() const { return observable_names_; }
const std::vector<std::string> &Result::expression_names() const { return expression_names_; }

void Result::set_reported_species_indices(std::vector<std::size_t> indices) {
    reported_species_indices_ = std::move(indices);
}
const std::vector<std::size_t> &Result::reported_species_indices() const {
    return reported_species_indices_;
}

SolverStats &Result::solver_stats() { return stats_; }
const SolverStats &Result::solver_stats() const { return stats_; }

SsaDiagnostics &Result::ssa_diagnostics() { return ssa_diag_; }
const SsaDiagnostics &Result::ssa_diagnostics() const { return ssa_diag_; }

// ─── Sensitivity data ────────────────────────────────────────────────────────

void Result::allocate_sensitivities(int n_times, int n_species, int n_params) {
    n_sens_params_ = n_params;
    sensitivities_.resize(n_times * n_species * n_params, 0.0);
}

void Result::record_sensitivities(int time_index, const double *const *sens_data, int n_species,
                                  int n_params) {
    if (time_index < 0 || time_index >= n_times_ || n_sens_params_ == 0)
        return;
    // sens_data[p][i] = dY_i/dp_p at this time point
    // We store as [time][species][param] = sensitivities_[t*ns*np + i*np + p]
    int offset = time_index * n_species * n_params;
    for (int p = 0; p < n_params; ++p) {
        for (int i = 0; i < n_species; ++i) {
            sensitivities_[offset + i * n_params + p] = sens_data[p][i];
        }
    }
}

void Result::set_sens_param_names(const std::vector<std::string> &names) {
    sens_param_names_ = names;
}

const std::vector<double> &Result::sensitivity_data() const { return sensitivities_; }
int Result::n_sens_params() const { return n_sens_params_; }
const std::vector<std::string> &Result::sens_param_names() const { return sens_param_names_; }

// ─── IC sensitivity data ─────────────────────────────────────────────────────

void Result::allocate_sensitivities_ic(int n_times, int n_species, int n_ic) {
    n_sens_ic_species_ = n_ic;
    sensitivities_ic_.resize(n_times * n_species * n_ic, 0.0);
}

void Result::record_sensitivities_ic(int time_index, const double *const *sens_data, int n_species,
                                     int n_ic) {
    if (time_index < 0 || time_index >= n_times_ || n_sens_ic_species_ == 0)
        return;
    int offset = time_index * n_species * n_ic;
    for (int p = 0; p < n_ic; ++p) {
        for (int i = 0; i < n_species; ++i) {
            sensitivities_ic_[offset + i * n_ic + p] = sens_data[p][i];
        }
    }
}

void Result::set_sens_ic_species_names(const std::vector<std::string> &names) {
    sens_ic_species_names_ = names;
}

const std::vector<double> &Result::sensitivity_ic_data() const { return sensitivities_ic_; }
int Result::n_sens_ic_species() const { return n_sens_ic_species_; }
const std::vector<std::string> &Result::sens_ic_species_names() const {
    return sens_ic_species_names_;
}

// ─── Observable / expression output sensitivities (GH #196) ──────────────────
//
// Storage only — these mirror the species blocks above. record_* is a no-op
// until allocate_* has sized the buffer (the empty-buffer guard), so a stray
// record on an unpopulated block can never index out of range.

void Result::allocate_observable_sensitivities(int n_times, int n_observables, int n_params) {
    observable_sensitivities_.assign(static_cast<size_t>(n_times) * n_observables * n_params, 0.0);
}

void Result::record_observable_sensitivities(int time_index, const double *const *sens_data,
                                             int n_observables, int n_params) {
    if (time_index < 0 || time_index >= n_times_ || observable_sensitivities_.empty())
        return;
    size_t offset = static_cast<size_t>(time_index) * n_observables * n_params;
    for (int p = 0; p < n_params; ++p) {
        for (int i = 0; i < n_observables; ++i) {
            observable_sensitivities_[offset + static_cast<size_t>(i) * n_params + p] =
                sens_data[p][i];
        }
    }
}

const std::vector<double> &Result::observable_sensitivity_data() const {
    return observable_sensitivities_;
}

void Result::allocate_expression_sensitivities(int n_times, int n_expressions, int n_params) {
    expression_sensitivities_.assign(static_cast<size_t>(n_times) * n_expressions * n_params, 0.0);
}

void Result::record_expression_sensitivities(int time_index, const double *const *sens_data,
                                             int n_expressions, int n_params) {
    if (time_index < 0 || time_index >= n_times_ || expression_sensitivities_.empty())
        return;
    size_t offset = static_cast<size_t>(time_index) * n_expressions * n_params;
    for (int p = 0; p < n_params; ++p) {
        for (int i = 0; i < n_expressions; ++i) {
            expression_sensitivities_[offset + static_cast<size_t>(i) * n_params + p] =
                sens_data[p][i];
        }
    }
}

const std::vector<double> &Result::expression_sensitivity_data() const {
    return expression_sensitivities_;
}

void Result::allocate_observable_sensitivities_ic(int n_times, int n_observables, int n_ic) {
    observable_sensitivities_ic_.assign(static_cast<size_t>(n_times) * n_observables * n_ic, 0.0);
}

void Result::record_observable_sensitivities_ic(int time_index, const double *const *sens_data,
                                                int n_observables, int n_ic) {
    if (time_index < 0 || time_index >= n_times_ || observable_sensitivities_ic_.empty())
        return;
    size_t offset = static_cast<size_t>(time_index) * n_observables * n_ic;
    for (int k = 0; k < n_ic; ++k) {
        for (int i = 0; i < n_observables; ++i) {
            observable_sensitivities_ic_[offset + static_cast<size_t>(i) * n_ic + k] =
                sens_data[k][i];
        }
    }
}

const std::vector<double> &Result::observable_sensitivity_ic_data() const {
    return observable_sensitivities_ic_;
}

void Result::allocate_expression_sensitivities_ic(int n_times, int n_expressions, int n_ic) {
    expression_sensitivities_ic_.assign(static_cast<size_t>(n_times) * n_expressions * n_ic, 0.0);
}

void Result::record_expression_sensitivities_ic(int time_index, const double *const *sens_data,
                                                int n_expressions, int n_ic) {
    if (time_index < 0 || time_index >= n_times_ || expression_sensitivities_ic_.empty())
        return;
    size_t offset = static_cast<size_t>(time_index) * n_expressions * n_ic;
    for (int k = 0; k < n_ic; ++k) {
        for (int i = 0; i < n_expressions; ++i) {
            expression_sensitivities_ic_[offset + static_cast<size_t>(i) * n_ic + k] =
                sens_data[k][i];
        }
    }
}

const std::vector<double> &Result::expression_sensitivity_ic_data() const {
    return expression_sensitivities_ic_;
}

// ─── Truncate (steady-state early-stop) ──────────────────────────────────────

void Result::truncate(int new_n_times) {
    if (new_n_times < 0 || new_n_times > n_times_) {
        throw std::out_of_range("Result::truncate: new_n_times (" + std::to_string(new_n_times) +
                                ") out of range [0, " + std::to_string(n_times_) + "]");
    }
    if (new_n_times == n_times_) {
        return;
    }

    time_.resize(new_n_times);
    if (n_species_ > 0) {
        species_.resize(static_cast<size_t>(new_n_times) * n_species_);
    }
    if (n_observables_ > 0) {
        observables_.resize(static_cast<size_t>(new_n_times) * n_observables_);
    }
    if (n_expressions_ > 0) {
        expressions_.resize(static_cast<size_t>(new_n_times) * n_expressions_);
    }
    if (n_sens_params_ > 0) {
        sensitivities_.resize(static_cast<size_t>(new_n_times) * n_species_ * n_sens_params_);
    }
    if (n_sens_ic_species_ > 0) {
        sensitivities_ic_.resize(static_cast<size_t>(new_n_times) * n_species_ *
                                 n_sens_ic_species_);
    }

    // GH #196 output sensitivities: row/depth counts aren't tracked here, so
    // shrink each block by its per-time-point stride (size / old n_times_).
    // n_times_ > 0 is guaranteed (new_n_times == n_times_ returned above).
    auto shrink = [&](std::vector<double> &block) {
        if (!block.empty()) {
            size_t per_time = block.size() / static_cast<size_t>(n_times_);
            block.resize(static_cast<size_t>(new_n_times) * per_time);
        }
    };
    shrink(observable_sensitivities_);
    shrink(expression_sensitivities_);
    shrink(observable_sensitivities_ic_);
    shrink(expression_sensitivities_ic_);

    n_times_ = new_n_times;
}

// ─── Export helpers ──────────────────────────────────────────────────────────

void Result::to_gdat(const std::string &path, bool print_functions, bool print_rate_laws) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    out << std::setprecision(12) << std::scientific;

    // Optional function (expression) columns, in declared order: user-named
    // functions when print_functions, plus the auto-generated _rateLawN columns
    // when print_rate_laws. Headers are always bare (no "()"), so the output is
    // byte-identical across every simulation method.
    const std::vector<size_t> func_cols =
        gdat_function_indices(expression_names_, print_functions, print_rate_laws);

    // User-facing observable columns, dropping the loader's internal
    // network-rewrite scaffolding observables (issue #61). run_network never
    // emits these, so including them would break .gdat column-set parity.
    const std::vector<size_t> obs_cols = public_observable_indices(observable_names_);

    // Header line
    out << "#          time";
    for (size_t c : obs_cols) {
        out << "          " << observable_names_[c];
    }
    for (size_t c : func_cols) {
        out << "          " << expression_names_[c];
    }
    out << "\n";

    // Data lines
    for (int t = 0; t < n_times_; ++t) {
        out << " " << std::setw(18) << time_[t];
        for (size_t c : obs_cols) {
            out << " " << std::setw(18) << observables_[t * n_observables_ + c];
        }
        for (size_t c : func_cols) {
            out << " " << std::setw(18) << expressions_[t * n_expressions_ + c];
        }
        out << "\n";
    }
}

void Result::to_cdat(const std::string &path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    out << std::setprecision(12) << std::scientific;

    // GH #71: project to the reported species subset. Empty ⇒ all species
    // (every existing model — byte-identical column set and ordering).
    std::vector<std::size_t> cols = reported_species_indices_;
    if (cols.empty()) {
        cols.reserve(n_species_);
        for (int j = 0; j < n_species_; ++j) {
            cols.push_back(static_cast<std::size_t>(j));
        }
    }

    // Header line
    out << "#          time";
    for (std::size_t j : cols) {
        out << "          " << species_names_[j];
    }
    out << "\n";

    // Data lines
    for (int t = 0; t < n_times_; ++t) {
        out << " " << std::setw(18) << time_[t];
        for (std::size_t j : cols) {
            out << " " << std::setw(18) << species_[t * n_species_ + j];
        }
        out << "\n";
    }
}

// ─── Non-finite forward sensitivities (issue #384) ───────────────────────────
// Rationale in result.hpp.

void refuse_nonfinite_sensitivity_block(const std::vector<double> &data,
                                        const std::vector<double> &times, int n_times,
                                        int n_species, int n_cols,
                                        const std::vector<std::string> &col_names,
                                        const std::string &axis) {
    if (n_cols <= 0 || n_species <= 0 || data.empty()) {
        return;
    }
    // Report the FIRST bad output point and every column implicated there: the
    // time localizes the event for a bisection, and the column names what to
    // drop to get a usable run out of the same model.
    const size_t stride = static_cast<size_t>(n_species) * static_cast<size_t>(n_cols);
    for (int ti = 0; ti < n_times; ++ti) {
        std::set<int> bad_cols;
        size_t n_bad = 0;
        for (int si = 0; si < n_species; ++si) {
            for (int ci = 0; ci < n_cols; ++ci) {
                const size_t k = static_cast<size_t>(ti) * stride +
                                 static_cast<size_t>(si) * static_cast<size_t>(n_cols) +
                                 static_cast<size_t>(ci);
                if (k < data.size() && !std::isfinite(data[k])) {
                    bad_cols.insert(ci);
                    ++n_bad;
                }
            }
        }
        if (bad_cols.empty()) {
            continue;
        }
        std::ostringstream os;
        os << "Forward sensitivity returned a non-finite value: " << n_bad
           << " cell(s) at the first affected output point t="
           << (ti < static_cast<int>(times.size()) ? times[ti] : 0.0) << " (index " << ti << " of "
           << n_times << "), in " << axis << " ";
        bool first = true;
        int shown = 0;
        for (int ci : bad_cols) {
            if (shown++ == 6) {
                os << ", … (" << (bad_cols.size() - 6) << " more)";
                break;
            }
            os << (first ? "'" : ", '")
               << (ci < static_cast<int>(col_names.size()) ? col_names[ci] : std::to_string(ci))
               << "'";
            first = false;
        }
        // The floor is ONE cause, not the cause: issue #388 measured 14 corpus
        // models that stay non-finite with it switched off, so promising it as
        // the remedy would send those callers down a dead end.
        os << ". The state trajectory and the solver's own counters can both be clean "
              "when this happens — CVODES' error test cannot reject a value that is "
              "already NaN, because every comparison against NaN is false. Check "
              "n_sens_err_test_fails in Result.solver_stats for what the sensitivity "
              "solve actually rejected. A tighter atol often resolves it, and "
              "BNGSIM_SENS_ERROR_FLOOR=0 disables the issue #177 tolerance floor, whose "
              "relaxation admits the blow-up on some models — though not on the ones that "
              "fail at the first output point, whose derivative was never defined "
              "(GH #384, GH #388).";
        throw std::runtime_error(os.str());
    }
}

void refuse_nonfinite_sensitivities(const Result &result) {
    const int n_p = result.n_sens_params();
    const int n_ic = result.n_sens_ic_species();
    if (n_p <= 0 && n_ic <= 0) {
        return;
    }
    const int n_t = result.n_times();
    const int ns = result.n_species();
    refuse_nonfinite_sensitivity_block(result.sensitivity_data(), result.time(), n_t, ns, n_p,
                                       result.sens_param_names(), "parameter column");
    refuse_nonfinite_sensitivity_block(result.sensitivity_ic_data(), result.time(), n_t, ns, n_ic,
                                       result.sens_ic_species_names(), "initial-condition column");
}

} // namespace bngsim
