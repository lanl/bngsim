// bngsim/include/bngsim/rulemonkey_simulator.hpp — RuleMonkey simulator wrapper
//
// Wraps RuleMonkey as an in-process library for exact network-free simulation.

#pragma once

#include "bngsim/result.hpp"
#include "bngsim/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bngsim {

class RuleMonkeySimulator {
  public:
    explicit RuleMonkeySimulator(const std::string &xml_path);
    ~RuleMonkeySimulator();

    RuleMonkeySimulator(const RuleMonkeySimulator &) = delete;
    RuleMonkeySimulator &operator=(const RuleMonkeySimulator &) = delete;

    // Stateless run from the parsed XML plus current configuration.
    //
    // @param timeout_seconds  Wall-clock budget in seconds; `<= 0` disables
    //                         it. When the budget is exceeded between SSA
    //                         events (upstream RuleMonkey polls every 1024
    //                         events), throws `bngsim::TimeoutError`.
    Result run(const TimeSpec &times, uint64_t seed = 42, double timeout_seconds = 0.0);

    // Stateful session API for multi-action workflows.
    //
    // `step_to` and `simulate` accept `timeout_seconds` with the same
    // semantics as `run()`. On timeout, the upstream RuleMonkey session is
    // left at the last completed SSA event; subsequent state is undefined
    // for bngsim purposes, so callers should `destroy_session()` before
    // reusing the simulator.
    void initialize(uint64_t seed = 42);
    void step_to(double time, double timeout_seconds = 0.0);
    // `sample_times`, when non-empty, overrides the uniform t_start..t_end /
    // n_points grid: the returned Result is labelled with exactly these times
    // (sorted ascending, absolute in the caller's frame, sample_times[0] the
    // segment start) and the live session advances by
    // (sample_times.back() - sample_times.front()) from its current time
    // (GH #184). An empty list is the uniform path, byte-identical to before.
    Result simulate(double t_start, double t_end, int n_points, double timeout_seconds = 0.0,
                    const std::vector<double> &sample_times = {});
    void add_molecules(const std::string &molecule_type_name, int count);
    int get_molecule_count(const std::string &molecule_type_name) const;

    // Exact-species queries & mutations (RuleMonkey #9 §1). Each accepts an
    // exact, fully-specified, connected BNGL species pattern, parsed and
    // canonicalized by the engine's runtime pattern parser. All require an
    // active session; the engine throws std::runtime_error otherwise.
    int get_species_count(const std::string &pattern) const;
    void add_species(const std::string &pattern, int count);
    void remove_species(const std::string &pattern, int count);
    void set_species_count(const std::string &pattern, int count);

    // Writes the live pool to `path` as a BNG-format `.species` file
    // (`readNFspecies`-compatible), via the engine's write_species_file.
    // The bngsim/NFsim-facing name is `save_species` for cross-backend
    // parity with NfsimSimulator::save_species.
    void save_species(const std::string &path) const;

    // Evaluates a BNG expression against the active session's current state
    // (RuleMonkey #9 §1). `extra` layers name=value bindings on top of the
    // model namespace for this evaluation only. Non-const: the engine
    // recomputes observables before evaluating.
    double evaluate_expression(const std::string &expr,
                               const std::unordered_map<std::string, double> &extra = {});

    std::vector<std::string> get_observable_names() const;
    std::vector<double> get_observable_values();
    double get_parameter(const std::string &name) const;
    bool has_session() const;
    double current_time() const;
    void destroy_session();
    void save_state(const std::string &path) const;
    void load_state(const std::string &path);

    // Configuration.
    void set_param(const std::string &name, double value);
    void clear_param_overrides();
    void set_molecule_limit(int limit);
    void set_block_same_complex_binding(bool enabled);

    const std::string &xml_path() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bngsim
