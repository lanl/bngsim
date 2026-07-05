// bngsim/src/rulemonkey_simulator.cpp — RuleMonkey simulator wrapper

#include "bngsim/rulemonkey_simulator.hpp"
#include "bngsim/wallclock.hpp"
#include "seed_count_rounding.hpp"

#include <rulemonkey/simulator.hpp>
#include <rulemonkey/types.hpp>

#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bngsim {

namespace {

// Build a CancelCallback that returns `false` (= cancel) once the
// WallClockBudget is exhausted. Returns an empty std::function when the
// budget is disabled so the upstream engine's no-callback fast path is
// preserved verbatim (one extra branch per 1024-event stride otherwise).
// Captures by value so the lambda owns its own clock snapshot — the
// caller's budget can go out of scope safely.
rulemonkey::CancelCallback make_cancel_callback(const WallClockBudget &budget) {
    if (!budget.active()) {
        return {};
    }
    return [budget]() noexcept { return budget.elapsed() <= budget.limit_seconds(); };
}

std::string
unsupported_feature_summary(const std::vector<rulemonkey::UnsupportedFeature> &features) {
    std::ostringstream os;
    bool first = true;
    for (const auto &feature : features) {
        if (feature.severity != rulemonkey::Severity::Error) {
            continue;
        }
        if (!first) {
            os << "; ";
        }
        first = false;
        os << feature.element << ": " << feature.feature;
    }
    return os.str();
}

void reject_unsupported_errors(const rulemonkey::RuleMonkeySimulator &sim) {
    std::string errors = unsupported_feature_summary(sim.unsupported_features());
    if (!errors.empty()) {
        throw std::runtime_error("RuleMonkey cannot honor this BNG XML: " + errors);
    }
}

std::vector<double> uniform_time_labels(double t_start, double t_end, int n_points) {
    std::vector<double> labels(n_points);
    const double dt = (n_points > 1) ? (t_end - t_start) / (n_points - 1) : 0.0;
    for (int i = 0; i < n_points; ++i) {
        labels[i] = t_start + i * dt;
    }
    return labels;
}

int rulemonkey_interval_count(int bngsim_n_points) {
    if (bngsim_n_points < 1) {
        throw std::runtime_error("RuleMonkeySimulator: n_points must be positive");
    }
    return bngsim_n_points - 1;
}

Result convert_rulemonkey_result(const rulemonkey::Result &rm_result,
                                 const std::vector<double> *time_labels = nullptr) {
    const int n_times = static_cast<int>(rm_result.n_times());
    const int n_obs = static_cast<int>(rm_result.n_observables());
    const int n_funcs = static_cast<int>(rm_result.n_functions());

    if (time_labels && static_cast<int>(time_labels->size()) != n_times) {
        throw std::runtime_error("RuleMonkey result conversion received mismatched time labels");
    }
    if (static_cast<int>(rm_result.observable_data.size()) != n_obs) {
        throw std::runtime_error("RuleMonkey result has inconsistent observable_data shape");
    }
    if (static_cast<int>(rm_result.function_data.size()) != n_funcs) {
        throw std::runtime_error("RuleMonkey result has inconsistent function_data shape");
    }
    if (!time_labels && static_cast<int>(rm_result.time.size()) != n_times) {
        throw std::runtime_error("RuleMonkey result has inconsistent time shape");
    }

    Result result;
    result.allocate(n_times, /*n_species=*/0, n_obs);
    result.set_observable_names(rm_result.observable_names);
    // BNGL `begin functions` globals (Clusters(), pre1_dose(), ...) are the
    // quantities PyBNF fits against. RuleMonkey 3.2.0 reports them on Result
    // alongside the observables; surface them as bngsim Result expression
    // columns, parallel to the NFsim backend's global/composite function
    // columns. function_names omits local (molecule-scoped) functions, which
    // have no scalar value.
    result.set_expression_names(rm_result.function_names);

    std::vector<double> obs_buf(n_obs, 0.0);
    std::vector<double> func_buf(n_funcs, 0.0);
    for (int i = 0; i < n_times; ++i) {
        for (int j = 0; j < n_obs; ++j) {
            if (static_cast<int>(rm_result.observable_data[j].size()) != n_times) {
                throw std::runtime_error("RuleMonkey result has ragged observable_data columns");
            }
            obs_buf[j] = rm_result.observable_data[j][i];
        }
        for (int j = 0; j < n_funcs; ++j) {
            if (static_cast<int>(rm_result.function_data[j].size()) != n_times) {
                throw std::runtime_error("RuleMonkey result has ragged function_data columns");
            }
            func_buf[j] = rm_result.function_data[j][i];
        }
        const double t = time_labels ? (*time_labels)[i] : rm_result.time[i];
        result.record(i, t, nullptr, obs_buf.data());
        result.record_expressions(i, func_buf.data());
    }

    result.solver_stats().n_steps = static_cast<int>(rm_result.event_count);
    result.solver_stats().n_rhs_evals = static_cast<int>(rm_result.event_count);
    return result;
}

rulemonkey::TimeSpec to_rulemonkey_times(const TimeSpec &times) {
    rulemonkey::TimeSpec rm_times;
    rm_times.t_start = times.t_start;
    rm_times.t_end = times.t_end;
    if (!times.sample_times.empty()) {
        // Explicit, possibly non-uniform output instants (GH #169). Upstream
        // RuleMonkey #16 records at exactly these times in a single run_ssa pass
        // and ignores n_points when sample_times is set, so pass the sorted list
        // through and leave n_points at its default. The Python layer guarantees
        // sample_times[0] == t_start and t_end == the largest sample time, which
        // satisfies upstream's "t_end bounds the SSA loop" contract.
        rm_times.sample_times = times.sample_times;
    } else {
        rm_times.n_points = rulemonkey_interval_count(times.n_points);
    }
    return rm_times;
}

} // namespace

struct RuleMonkeySimulator::Impl {
    explicit Impl(std::string path)
        : xml_path(std::move(path)),
          // Round fractional seed-species concentrations to integer counts at
          // the bngsim handoff so RuleMonkey's truncating loader sees an
          // already-integer value (GH #51). Fail-safe: no temp → load the
          // original XML unchanged. `seed_rounded` is declared before `sim`, so
          // it is destroyed *after* sim — the temp file outlives every read
          // RuleMonkey makes of it.
          seed_rounded(bngsim::seedround::write_seed_rounded_xml(xml_path)),
          sim(seed_rounded.path_or(xml_path), rulemonkey::Method::NfExact) {
        reject_unsupported_errors(sim);
    }

    std::string xml_path;
    bngsim::seedround::TempXmlFile seed_rounded;
    rulemonkey::RuleMonkeySimulator sim;
};

RuleMonkeySimulator::RuleMonkeySimulator(const std::string &xml_path)
    : impl_(std::make_unique<Impl>(xml_path)) {}

RuleMonkeySimulator::~RuleMonkeySimulator() = default;

Result RuleMonkeySimulator::run(const TimeSpec &times, uint64_t seed, double timeout_seconds) {
    WallClockBudget budget(timeout_seconds);
    try {
        // Both the uniform grid and explicit sample_times (GH #169) go through
        // upstream's stateless run(): RuleMonkey #16 honors TimeSpec::sample_times
        // in a single non-invasive run_ssa pass — output at exactly the requested
        // instants, bit-identical to the uniform grid where the two schedules
        // share a time, and event_count preserved. The recorder labels each row
        // with the requested time, so convert_rulemonkey_result needs no explicit
        // time labels and solver_stats().n_steps now reports for both paths.
        auto rm_result =
            impl_->sim.run(to_rulemonkey_times(times), seed, make_cancel_callback(budget));
        return convert_rulemonkey_result(rm_result);
    } catch (const rulemonkey::Cancelled &) {
        // Upstream raises Cancelled iff our callback returned false, which it
        // does iff the WallClockBudget elapsed past the limit. Translate to
        // the typed bngsim exception that pybind11 maps to SimulationTimeout.
        throw TimeoutError(budget.limit_seconds(), budget.elapsed());
    }
}

void RuleMonkeySimulator::initialize(uint64_t seed) { impl_->sim.initialize(seed); }

void RuleMonkeySimulator::step_to(double time, double timeout_seconds) {
    WallClockBudget budget(timeout_seconds);
    try {
        impl_->sim.step_to(time, make_cancel_callback(budget));
    } catch (const rulemonkey::Cancelled &) {
        throw TimeoutError(budget.limit_seconds(), budget.elapsed());
    }
}

Result RuleMonkeySimulator::simulate(double t_start, double t_end, int n_points,
                                     double timeout_seconds,
                                     const std::vector<double> &sample_times) {
    WallClockBudget budget(timeout_seconds);
    const double internal_start = impl_->sim.current_time();

    try {
        if (!sample_times.empty()) {
            // Explicit output instants (GH #184) on a stateful session. Map the
            // caller's absolute sample times onto the session's internal clock by
            // their offset from the segment start (sample_times[0]); the live
            // session continues from internal_start, so the i-th sample lands at
            // internal_start + (sample_times[i] - sample_times[0]). Hand the
            // engine a TimeSpec carrying those instants: upstream's session
            // simulate(TimeSpec) records at exactly them in a single run_ssa pass
            // (RuleMonkey #16), continuing from the current session state —
            // bit-identical to the uniform grid where the two schedules share a
            // time, with event_count preserved. The Result is relabelled back to
            // the caller's absolute sample_times.
            const double origin = sample_times.front();
            rulemonkey::TimeSpec rm_ts;
            rm_ts.sample_times.reserve(sample_times.size());
            for (double t : sample_times) {
                rm_ts.sample_times.push_back(internal_start + (t - origin));
            }
            rm_ts.t_start = internal_start;
            rm_ts.t_end = rm_ts.sample_times.back();
            auto rm_result = impl_->sim.simulate(rm_ts, make_cancel_callback(budget));
            return convert_rulemonkey_result(rm_result, &sample_times);
        }

        const double internal_end = internal_start + (t_end - t_start);
        auto rm_result =
            impl_->sim.simulate(internal_start, internal_end, rulemonkey_interval_count(n_points),
                                make_cancel_callback(budget));
        auto labels = uniform_time_labels(t_start, t_end, n_points);
        return convert_rulemonkey_result(rm_result, &labels);
    } catch (const rulemonkey::Cancelled &) {
        throw TimeoutError(budget.limit_seconds(), budget.elapsed());
    }
}

void RuleMonkeySimulator::add_molecules(const std::string &molecule_type_name, int count) {
    impl_->sim.add_molecules(molecule_type_name, count);
}

int RuleMonkeySimulator::get_molecule_count(const std::string &molecule_type_name) const {
    return impl_->sim.get_molecule_count(molecule_type_name);
}

int RuleMonkeySimulator::get_species_count(const std::string &pattern) const {
    return impl_->sim.get_species_count(pattern);
}

void RuleMonkeySimulator::add_species(const std::string &pattern, int count) {
    impl_->sim.add_species(pattern, count);
}

void RuleMonkeySimulator::remove_species(const std::string &pattern, int count) {
    impl_->sim.remove_species(pattern, count);
}

void RuleMonkeySimulator::set_species_count(const std::string &pattern, int count) {
    impl_->sim.set_species_count(pattern, count);
}

void RuleMonkeySimulator::save_species(const std::string &path) const {
    impl_->sim.write_species_file(path);
}

double
RuleMonkeySimulator::evaluate_expression(const std::string &expr,
                                         const std::unordered_map<std::string, double> &extra) {
    return impl_->sim.evaluate_expression(expr, extra);
}

std::vector<std::string> RuleMonkeySimulator::get_observable_names() const {
    return impl_->sim.observable_names();
}

std::vector<double> RuleMonkeySimulator::get_observable_values() {
    return impl_->sim.get_observable_values();
}

double RuleMonkeySimulator::get_parameter(const std::string &name) const {
    return impl_->sim.get_parameter(name);
}

bool RuleMonkeySimulator::has_session() const { return impl_->sim.has_session(); }

double RuleMonkeySimulator::current_time() const { return impl_->sim.current_time(); }

void RuleMonkeySimulator::destroy_session() { impl_->sim.destroy_session(); }

void RuleMonkeySimulator::save_state(const std::string &path) const { impl_->sim.save_state(path); }

void RuleMonkeySimulator::load_state(const std::string &path) { impl_->sim.load_state(path); }

void RuleMonkeySimulator::set_param(const std::string &name, double value) {
    impl_->sim.set_param(name, value);
}

void RuleMonkeySimulator::clear_param_overrides() { impl_->sim.clear_param_overrides(); }

void RuleMonkeySimulator::set_molecule_limit(int limit) { impl_->sim.set_molecule_limit(limit); }

void RuleMonkeySimulator::set_block_same_complex_binding(bool enabled) {
    impl_->sim.set_block_same_complex_binding(enabled);
}

const std::string &RuleMonkeySimulator::xml_path() const { return impl_->xml_path; }

} // namespace bngsim
