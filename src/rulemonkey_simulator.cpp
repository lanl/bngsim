// bngsim/src/rulemonkey_simulator.cpp — RuleMonkey simulator wrapper

#include "bngsim/rulemonkey_simulator.hpp"
#include "bngsim/wallclock.hpp"
#include "seed_count_rounding.hpp" // round_half_up_count — the GH #51 policy

#include <rulemonkey/simulator.hpp>
#include <rulemonkey/types.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    if (n_points > 1)
        labels.back() = t_end; // as TimeSpec::output_times() (issue #693)
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

// A bngsim run starts from the initial state at t_start and never sees [0,
// t_start] (issue #703). For a model that does not read the clock only elapsed
// time matters, so the request is shifted to elapsed time (t_start -> 0) and
// run() labels the rows back with the times asked for. That keeps the engine's
// clock small, where its own time tolerances are, however late the run starts.
rulemonkey::TimeSpec elapsed_times(const TimeSpec &times) {
    rulemonkey::TimeSpec rm_times;
    const double t0 = times.t_start;
    rm_times.t_start = 0.0;
    rm_times.t_end = times.t_end - t0;
    if (!times.sample_times.empty()) {
        // Explicit, possibly non-uniform output instants (GH #169). Upstream
        // RuleMonkey #16 records at exactly these times in a single run_ssa pass
        // and ignores n_points when sample_times is set, so pass the sorted list
        // through and leave n_points at its default. The Python layer guarantees
        // sample_times[0] == t_start and t_end == the largest sample time, which
        // satisfies upstream's "t_end bounds the SSA loop" contract.
        rm_times.sample_times.reserve(times.sample_times.size());
        for (double t : times.sample_times)
            rm_times.sample_times.push_back(t - t0);
    } else {
        rm_times.n_points = rulemonkey_interval_count(times.n_points);
    }
    return rm_times;
}

// For a model that reads the clock the request goes through unchanged: since
// RuleMonkey 3.11.0 the seed state sits at t_start on the model's clock
// (richardposner/RuleMonkey#98), so a rate law that reads time() reads absolute
// model time, as ode and ssa do (issue #1022).
rulemonkey::TimeSpec absolute_times(const TimeSpec &times) {
    rulemonkey::TimeSpec rm_times;
    rm_times.t_start = times.t_start;
    rm_times.t_end = times.t_end;
    if (!times.sample_times.empty()) {
        rm_times.sample_times = times.sample_times;
    } else {
        rm_times.n_points = rulemonkey_interval_count(times.n_points);
    }
    return rm_times;
}

// Whether `seg_start` is the session's clock, by the test upstream's simulate()
// applies before it accepts a segment.
bool on_session_clock(double seg_start, double session_t) {
    const double tol = 1e-9 * std::max(1.0, std::fabs(session_t));
    return std::fabs(seg_start - session_t) <= tol;
}

void require_finite(const char *where, double t) {
    if (!std::isfinite(t)) {
        std::ostringstream msg;
        msg << where << ": a time that is not a finite number (" << t << ")";
        throw std::runtime_error(msg.str());
    }
}

} // namespace

struct RuleMonkeySimulator::Impl {
    explicit Impl(std::string path) : xml_path(std::move(path)) {
        sim = std::make_unique<rulemonkey::RuleMonkeySimulator>(xml_path,
                                                                rulemonkey::Method::NfExact);
        reject_unsupported_errors(*sim);
        apply_seed_rounding();
    }

    // Round every fractional seed-species amount half-up, so RuleMonkey's
    // truncating instantiation matches bngsim's cold-start seed policy — 5.7
    // seeds 6 molecules, not 5 (GH #51).
    //
    // The engine already resolves each `<Species concentration=>` expression
    // against the current parameter overrides and reports the *real-valued*
    // result, so the policy is a pin on top of that number rather than a
    // rewrite of the XML the engine parses. Pins outrank the concentration
    // expression and deliberately survive `set_param`, so a stale pin would
    // shadow the very re-derivation an override is asking for: clear the
    // previous round first and re-derive from whatever the expressions now
    // resolve to. Call after anything that can move a seed amount.
    //
    // Keyed by `<Species id=>` because ids are unique where BNGL patterns need
    // not be (upstream resolves a key by pattern first, then by id; no `id` in
    // 223 corpus XMLs is also some other species' pattern).
    void apply_seed_rounding() {
        sim->clear_initial_amount_overrides();
        for (const auto &row : sim->initial_species()) {
            // Fail-safe, matching the overlay this replaced: an amount that is
            // not a finite non-negative number is left for the engine to
            // reject or truncate as it sees fit.
            if (!std::isfinite(row.amount) || row.amount < 0.0 ||
                std::floor(row.amount) == row.amount) {
                continue;
            }
            sim->set_initial_amount(
                row.id, static_cast<double>(bngsim::seedround::round_half_up_count(row.amount)));
        }
    }

    // Advance the live session from `start` to `end` and return the rows at
    // `samples` (sorted, all within [start, end]), one engine call per stretch.
    //
    // The engine integrates a clock-reading rate in panels of up to a
    // thousandth of a call and accepts a panel whose nodes all read the same,
    // so a window narrower than the gap between two nodes is not seen: a pulse
    // of area 1 and width 0.1 fires nothing in a call of 1000. A call that
    // ends on every instant the rate changes at has no such window inside it.
    // Each stretch then starts with its rate already on the new branch, or
    // jumps to it within a few ulps of its first panel, which the engine
    // resolves. Ending a call spends its last waiting-time draw, which is
    // exact (richardposner/RuleMonkey#89).
    //
    // A stretch also ends on an output time too close after its start for the
    // engine to tell the two apart: run_ssa records every sample within
    // 1e-12*|t_end| of a call's start as the state at the start, whatever
    // fires in between, and at t = 1e9 that is a stretch of 0.001. A row is
    // therefore always read as the first row of a call, or at least that far
    // into one, and a row at `end` is read after the last stretch.
    //
    // The engine can stop short of a stretch's end: where nothing more can
    // fire (and no rate reads the clock), and where the molecule limit is
    // reached. Its clock then stays where it stopped, as after any upstream
    // call, and the rows still owed are read there, each at its own time.
    rulemonkey::Result advance(double start, double end, const std::vector<double> &samples,
                               const rulemonkey::CancelCallback &should_continue) {
        rulemonkey::Result out;
        const auto keep = [&out](const rulemonkey::Result &piece, std::size_t row) {
            if (out.observable_data.empty() && out.function_data.empty()) {
                out.observable_names = piece.observable_names;
                out.function_names = piece.function_names;
                out.observable_data.resize(piece.observable_data.size());
                out.function_data.resize(piece.function_data.size());
            }
            out.time.push_back(piece.time.at(row));
            for (std::size_t j = 0; j < out.observable_data.size(); ++j) {
                out.observable_data[j].push_back(piece.observable_data.at(j).at(row));
            }
            for (std::size_t j = 0; j < out.function_data.size(); ++j) {
                out.function_data[j].push_back(piece.function_data.at(j).at(row));
            }
        };
        std::size_t next = 0; // the first sample not yet recorded
        auto stop = std::upper_bound(breakpoints.begin(), breakpoints.end(), start);
        double a = start;
        while (a < end) {
            stop = std::upper_bound(stop, breakpoints.end(), a);
            double b = (stop != breakpoints.end() && *stop < end) ? *stop : end;
            const auto after = std::upper_bound(samples.begin() + next, samples.end(), a);
            const double t_eps = std::max(1e-15, 1e-12 * std::max(std::fabs(a), std::fabs(b)));
            if (after != samples.end() && *after < b && *after <= a + t_eps) {
                b = *after;
            }
            rulemonkey::TimeSpec ts;
            ts.sample_times.push_back(a);
            // A sample on `a` is the first row of this call, so that a function
            // column reads the branch that starts there too.
            std::vector<std::size_t> rows;
            while (next < samples.size() && samples[next] < b) {
                if (samples[next] > a) {
                    ts.sample_times.push_back(samples[next]);
                }
                rows.push_back(ts.sample_times.size() - 1);
                ++next;
            }
            ts.sample_times.push_back(b);
            ts.t_start = a;
            ts.t_end = b;
            const auto piece = sim->simulate(ts, should_continue);
            for (std::size_t row : rows) {
                keep(piece, row);
            }
            out.event_count = piece.event_count;
            if (sim->current_time() < b) {
                break; // stopped short: nothing further fires in this call
            }
            a = b;
        }
        if (next < samples.size()) {
            // The rows still owed, at the end itself or past where the engine
            // stopped: the state it now holds, in a call that goes nowhere and
            // draws nothing, which the engine records at each row's own time.
            const double here = sim->current_time();
            rulemonkey::TimeSpec ts;
            ts.sample_times.push_back(here);
            for (std::size_t i = next; i < samples.size(); ++i) {
                ts.sample_times.push_back(std::max(samples[i], here));
            }
            ts.t_start = here;
            ts.t_end = here;
            const auto piece = sim->simulate(ts, should_continue);
            for (std::size_t row = 1; next < samples.size(); ++next, ++row) {
                keep(piece, row);
            }
            out.event_count = piece.event_count;
        }
        if (out.observable_names.empty() && out.function_names.empty()) {
            out.observable_names = sim->observable_names();
            out.observable_data.resize(out.observable_names.size());
        }
        return out;
    }

    std::string xml_path;
    std::unique_ptr<rulemonkey::RuleMonkeySimulator> sim;
    // How the model reads the clock, as the caller has worked it out
    // (set_clock_profile). Until told otherwise it is taken to read it, with no
    // known instants at which its rates change.
    bool reads_clock = true;
    std::vector<double> breakpoints;
};

RuleMonkeySimulator::RuleMonkeySimulator(const std::string &xml_path)
    : impl_(std::make_unique<Impl>(xml_path)) {}

RuleMonkeySimulator::~RuleMonkeySimulator() = default;

void RuleMonkeySimulator::set_clock_profile(bool reads_clock, std::vector<double> breakpoints) {
    for (double x : breakpoints) {
        require_finite("RuleMonkeySimulator::set_clock_profile", x);
    }
    std::sort(breakpoints.begin(), breakpoints.end());
    breakpoints.erase(std::unique(breakpoints.begin(), breakpoints.end()), breakpoints.end());
    impl_->reads_clock = reads_clock;
    impl_->breakpoints = std::move(breakpoints);
}

Result RuleMonkeySimulator::run(const TimeSpec &times, uint64_t seed, double timeout_seconds) {
    require_finite("RuleMonkeySimulator::run", times.t_start);
    require_finite("RuleMonkeySimulator::run", times.t_end);
    for (double t : times.sample_times) {
        require_finite("RuleMonkeySimulator::run", t);
    }
    WallClockBudget budget(timeout_seconds);
    // The requested instants themselves, as every other backend reports them:
    // the engine's uniform grid is t_start + i*dt, which can miss the last by
    // an ulp (issue #693).
    const auto requested = times.output_times();
    try {
        // Both the uniform grid and explicit sample_times (GH #169) go through
        // upstream's stateless run(): RuleMonkey #16 honors TimeSpec::sample_times
        // in a single non-invasive run_ssa pass — output at exactly the requested
        // instants, bit-identical to the uniform grid where the two schedules
        // share a time, and event_count preserved.
        if (!impl_->reads_clock) {
            auto rm_result =
                impl_->sim->run(elapsed_times(times), seed, make_cancel_callback(budget));
            // Back from elapsed time to the labels the caller asked for.
            std::vector<double> labels(rm_result.time.begin(), rm_result.time.end());
            for (double &t : labels)
                t += times.t_start;
            if (requested.size() == labels.size())
                labels = requested;
            return convert_rulemonkey_result(rm_result, &labels);
        }

        // A model that reads the clock runs on it, one engine call per stretch
        // between the instants its rates change at, on a session of this run's
        // own.
        if (impl_->sim->has_session()) {
            throw std::runtime_error(
                "RuleMonkeySimulator::run: a model that reads the clock is run on a session "
                "of the run's own, and one is already active; destroy it first");
        }
        if (times.t_end < times.t_start) {
            throw std::runtime_error("RuleMonkeySimulator::run: t_end is earlier than t_start");
        }
        struct EndSession {
            rulemonkey::RuleMonkeySimulator &sim;
            ~EndSession() { sim.destroy_session(); }
        };
        impl_->sim->initialize(seed, times.t_start);
        const EndSession end_session{*impl_->sim};
        const auto rm_result =
            impl_->advance(times.t_start, times.t_end, requested, make_cancel_callback(budget));
        return convert_rulemonkey_result(rm_result, &requested);
    } catch (const rulemonkey::Cancelled &) {
        // Upstream raises Cancelled iff our callback returned false, which it
        // does iff the WallClockBudget elapsed past the limit. Translate to
        // the typed bngsim exception that pybind11 maps to SimulationTimeout.
        throw TimeoutError(budget.limit_seconds(), budget.elapsed());
    }
}

void RuleMonkeySimulator::initialize(uint64_t seed, double t_start) {
    impl_->sim->initialize(seed, t_start);
}

void RuleMonkeySimulator::step_to(double time, double timeout_seconds) {
    require_finite("RuleMonkeySimulator::step_to", time);
    const double session_t = impl_->sim->current_time();
    if (time < session_t) {
        // Upstream runs nothing and says nothing.
        std::ostringstream msg;
        msg.precision(17);
        msg << "RuleMonkeySimulator::step_to: t=" << time
            << " is earlier than the session's clock (" << session_t
            << "); a session cannot go back";
        throw std::runtime_error(msg.str());
    }
    WallClockBudget budget(timeout_seconds);
    try {
        if (impl_->reads_clock) {
            impl_->advance(session_t, time, {}, make_cancel_callback(budget));
        } else {
            impl_->sim->step_to(time, make_cancel_callback(budget));
        }
    } catch (const rulemonkey::Cancelled &) {
        throw TimeoutError(budget.limit_seconds(), budget.elapsed());
    }
}

bool RuleMonkeySimulator::starts_on_clock(double seg_start) const {
    return on_session_clock(seg_start, impl_->sim->current_time());
}

Result RuleMonkeySimulator::simulate(double t_start, double t_end, int n_points,
                                     double timeout_seconds,
                                     const std::vector<double> &sample_times) {
    require_finite("RuleMonkeySimulator::simulate", t_start);
    require_finite("RuleMonkeySimulator::simulate", t_end);
    for (double t : sample_times) {
        require_finite("RuleMonkeySimulator::simulate", t);
    }
    if (sample_times.empty()) {
        rulemonkey_interval_count(n_points); // refuses n_points < 1
        if (n_points < 2) {
            // One row cannot carry both ends of a segment, and upstream would
            // run the segment before the mismatch came to light.
            throw std::runtime_error(
                "RuleMonkeySimulator::simulate: n_points must be at least 2 (the start of "
                "the segment and its end)");
        }
    }
    WallClockBudget budget(timeout_seconds);
    const double session_t = impl_->sim->current_time();
    const double seg_start = sample_times.empty() ? t_start : sample_times.front();
    const bool on_clock = on_session_clock(seg_start, session_t);

    if (!on_clock && impl_->reads_clock) {
        // The session runs on the model's clock (issue #1022), and a rate law
        // that reads time() reads the session's clock, not the labels of the
        // rows. Running this segment from session_t under labels that start at
        // seg_start would return a trajectory whose time axis is not the time
        // its rates saw.
        std::ostringstream msg;
        msg.precision(17);
        msg << "RuleMonkeySimulator::simulate: the segment starts at t=" << seg_start
            << " but the session's clock reads " << session_t
            << ". Start the segment at current_time(), or put the seed state there with "
               "initialize(seed, t_start). A segment off the clock is run as elapsed time "
               "only for a model declared not to read the clock (issue #1022).";
        throw std::runtime_error(msg.str());
    }

    try {
        if (!on_clock) {
            // A model that does not read the clock sees only elapsed time: the
            // session advances by the segment's span from where it is, and the
            // rows carry the caller's times. Explicit instants (GH #184) land
            // at session_t + (sample_times[i] - sample_times[0]).
            if (!sample_times.empty()) {
                rulemonkey::TimeSpec rm_ts;
                rm_ts.sample_times.reserve(sample_times.size());
                for (double t : sample_times) {
                    rm_ts.sample_times.push_back(session_t + (t - seg_start));
                }
                rm_ts.t_start = rm_ts.sample_times.front();
                rm_ts.t_end = rm_ts.sample_times.back();
                auto rm_result = impl_->sim->simulate(rm_ts, make_cancel_callback(budget));
                return convert_rulemonkey_result(rm_result, &sample_times);
            }
            auto rm_result = impl_->sim->simulate(session_t, session_t + (t_end - t_start),
                                                  rulemonkey_interval_count(n_points),
                                                  make_cancel_callback(budget));
            const auto labels = uniform_time_labels(t_start, t_end, n_points);
            return convert_rulemonkey_result(rm_result, &labels);
        }

        // On the clock. Upstream's session simulate(TimeSpec) records at exactly
        // the instants it is given in a single run_ssa pass (RuleMonkey #16),
        // continuing from the current session state.
        if (impl_->reads_clock) {
            const auto samples =
                sample_times.empty() ? uniform_time_labels(t_start, t_end, n_points) : sample_times;
            if (samples.back() < seg_start) {
                throw std::runtime_error(
                    "RuleMonkeySimulator::simulate: the segment ends before it starts");
            }
            const auto rm_result =
                impl_->advance(seg_start, samples.back(), samples, make_cancel_callback(budget));
            return convert_rulemonkey_result(rm_result, &samples);
        }
        if (!sample_times.empty()) {
            rulemonkey::TimeSpec rm_ts;
            rm_ts.sample_times = sample_times;
            rm_ts.t_start = sample_times.front();
            rm_ts.t_end = sample_times.back();
            auto rm_result = impl_->sim->simulate(rm_ts, make_cancel_callback(budget));
            return convert_rulemonkey_result(rm_result, &sample_times);
        }
        auto rm_result = impl_->sim->simulate(t_start, t_end, rulemonkey_interval_count(n_points),
                                              make_cancel_callback(budget));
        const auto labels = uniform_time_labels(t_start, t_end, n_points);
        return convert_rulemonkey_result(rm_result, &labels);
    } catch (const rulemonkey::Cancelled &) {
        throw TimeoutError(budget.limit_seconds(), budget.elapsed());
    }
}

void RuleMonkeySimulator::add_molecules(const std::string &molecule_type_name, int count) {
    impl_->sim->add_molecules(molecule_type_name, count);
}

int RuleMonkeySimulator::get_molecule_count(const std::string &molecule_type_name) const {
    return impl_->sim->get_molecule_count(molecule_type_name);
}

int RuleMonkeySimulator::get_species_count(const std::string &pattern) const {
    return impl_->sim->get_species_count(pattern);
}

void RuleMonkeySimulator::add_species(const std::string &pattern, int count) {
    impl_->sim->add_species(pattern, count);
}

void RuleMonkeySimulator::remove_species(const std::string &pattern, int count) {
    impl_->sim->remove_species(pattern, count);
}

void RuleMonkeySimulator::set_species_count(const std::string &pattern, int count) {
    impl_->sim->set_species_count(pattern, count);
}

void RuleMonkeySimulator::save_species(const std::string &path) const {
    impl_->sim->write_species_file(path);
}

double
RuleMonkeySimulator::evaluate_expression(const std::string &expr,
                                         const std::unordered_map<std::string, double> &extra) {
    return impl_->sim->evaluate_expression(expr, extra);
}

std::vector<std::string> RuleMonkeySimulator::get_observable_names() const {
    return impl_->sim->observable_names();
}

std::vector<double> RuleMonkeySimulator::get_observable_values() {
    return impl_->sim->get_observable_values();
}

double RuleMonkeySimulator::get_parameter(const std::string &name) const {
    return impl_->sim->get_parameter(name);
}

std::vector<std::string> RuleMonkeySimulator::warnings() const {
    std::vector<std::string> out;
    for (const auto &feature : impl_->sim->unsupported_features()) {
        if (feature.severity == rulemonkey::Severity::Warn) {
            out.push_back(feature.element + ": " + feature.feature);
        }
    }
    return out;
}

bool RuleMonkeySimulator::has_session() const { return impl_->sim->has_session(); }

double RuleMonkeySimulator::current_time() const { return impl_->sim->current_time(); }

void RuleMonkeySimulator::destroy_session() { impl_->sim->destroy_session(); }

void RuleMonkeySimulator::save_state(const std::string &path) const {
    impl_->sim->save_state(path);
}

void RuleMonkeySimulator::load_state(const std::string &path) { impl_->sim->load_state(path); }

void RuleMonkeySimulator::set_param(const std::string &name, double value) {
    // Upstream owns the whole cascade since RuleMonkey v3.7.0: the override
    // re-derives from `<Parameter expr=>`, so it reaches derived parameters,
    // the rate constants they feed, and — through any `<Species
    // concentration=>` expression naming them — the seed-species amounts a
    // scan point starts from (GH #44, richardposner/RuleMonkey#23). It also
    // validates the name and enforces "no override while a session is active",
    // so a throw here leaves the seed pins untouched.
    impl_->sim->set_param(name, value);
    // Seed amounts may have just moved; re-derive the half-up pins over them.
    impl_->apply_seed_rounding();
}

void RuleMonkeySimulator::clear_param_overrides() {
    impl_->sim->clear_param_overrides();
    impl_->apply_seed_rounding();
}

void RuleMonkeySimulator::set_molecule_limit(int limit) { impl_->sim->set_molecule_limit(limit); }

void RuleMonkeySimulator::set_block_same_complex_binding(bool enabled) {
    impl_->sim->set_block_same_complex_binding(enabled);
}

const std::string &RuleMonkeySimulator::xml_path() const { return impl_->xml_path; }

} // namespace bngsim
