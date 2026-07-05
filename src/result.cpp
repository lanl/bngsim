// bngsim/src/result.cpp — Simulation result container implementation

#include "bngsim/result.hpp"

#include "bngsim/function_columns.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
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

} // namespace bngsim
