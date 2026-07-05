#pragma once

// Interface-only contract snapshot for the RuleMonkey <-> BNGsim boundary.
// This header is intentionally free of engine code.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rulemonkey/types.hpp"

namespace rulemonkey {

enum class Method {
  NfExact,
};

// Public embedding surface for in-process callers.
//
// Construction parses and validates the XML immediately. Each simulator owns
// its parsed model, parameter overrides, and molecule-limit configuration; no
// prior run state is reused across `run()` calls.
class RuleMonkeySimulator {
public:
  // Parses and validates `xml_path` immediately.
  //
  // Throws std::runtime_error if the path is empty or missing, the method is
  // unsupported, or the XML cannot be imported by the active runtime.
  explicit RuleMonkeySimulator(const std::string& xml_path, Method method = Method::NfExact);
  ~RuleMonkeySimulator();

  RuleMonkeySimulator(const RuleMonkeySimulator&) = delete;
  RuleMonkeySimulator& operator=(const RuleMonkeySimulator&) = delete;

  // Move is canonical pImpl: only the unique_ptr<Impl> is rebound, no
  // engine state is copied.  Defined out-of-line in simulator.cpp because
  // Impl is incomplete in this header.
  RuleMonkeySimulator(RuleMonkeySimulator&&) noexcept;
  RuleMonkeySimulator& operator=(RuleMonkeySimulator&&) noexcept;

  // Executes a fresh simulation from the parsed model plus the current
  // instance-local parameter overrides and molecule limit.
  //
  // Each call seeds a new RNG from `seed`; repeated calls with the same XML,
  // method, TimeSpec, overrides, molecule limit, and seed are reproducible
  // within the same runtime build. Prior successful or failed runs do not
  // mutate reusable simulator state.
  //
  // `should_continue`, if non-empty, is polled periodically from the SSA loop
  // (see CancelCallback in types.hpp).  A `false` return throws
  // `rulemonkey::Cancelled` out of this call; the simulator instance is left
  // in a usable state and a subsequent run() may be issued.
  Result run(const TimeSpec& ts, std::uint64_t seed = 42,
             const CancelCallback& should_continue = {});

  // Runs a parameter sweep: for each value of `spec.parameter`, simulates the
  // model over `spec.per_point` and records the endpoint observable and
  // global-function values (the values at `per_point.t_end`).  This is the
  // RM equivalent of BNG's `parameter_scan` action.
  //
  // With `spec.reset_conc == true` (the default) every point is an
  // independent run from the model's seed species — the dose-response case.
  // With `reset_conc == false` each point continues from the previous
  // point's final molecular state (BNG `reset_conc=>0`); the sweep is then
  // one continuous-time trajectory, so time-dependent rate laws see a
  // cumulative clock rather than a per-point reset.
  //
  // Every point uses the same `seed`: as in BNG, the seed is a run-level
  // setting, not per-point, so the points share a random stream and differ
  // only by the swept parameter.
  //
  // Throws std::runtime_error if a session is active, if `spec.parameter` is
  // not a declared parameter, or if the spec is otherwise invalid (empty
  // value set, `n_points < 1`, `n_points < 2` with distinct `par_min`/
  // `par_max`, or `log_scale` with a non-positive bound).
  ScanResult parameter_scan(const ScanSpec& spec, std::uint64_t seed = 42);

  // Runs `spec` as a forward sweep (par_min -> par_max) immediately followed
  // by a backward sweep (par_max -> par_min), as one continuous trajectory:
  // molecular state carries over across every point and across the
  // forward/backward turn, so a bistable model traces out a hysteresis loop.
  // This is the RM equivalent of BNG's `bifurcate` action.
  //
  // `spec.reset_conc` is ignored — bifurcation requires carry-over.  The
  // returned `BifurcateResult` aligns both branches to the same ascending
  // parameter axis (see types.hpp).
  //
  // Throws std::runtime_error under the same conditions as parameter_scan().
  BifurcateResult bifurcate(const ScanSpec& spec, std::uint64_t seed = 42);

  // Creates or resets a live session from the parsed model plus the current
  // stored parameter overrides and molecule limit.
  void initialize(std::uint64_t seed = 42);

  // Advances the active session to absolute logical time `time` and
  // discards the sampled trajectory.  Internally this still records
  // observable values at the segment endpoints (current_time and
  // `time`) — the in-process engine has no zero-sample fast path —
  // but the caller sees no `Result`.  Suitable for "equilibrate then
  // perturb then sample" flows where the equilibration trajectory
  // is uninteresting.
  //
  // `should_continue`, if non-empty, enables cooperative cancellation; a
  // `false` return throws `rulemonkey::Cancelled` mid-advance.  The session's
  // `current_time()` then reflects the last fully-applied event, so the
  // caller may inspect partial state, resume by calling step_to / simulate
  // again, or call `destroy_session()` to discard the run.
  void step_to(double time, const CancelCallback& should_continue = {});

  // Samples a segment from the active session starting at the current session
  // time and ending at `t_end`.
  //
  // `should_continue` has the same cooperative-cancellation semantics as on
  // `run()`; if it returns false mid-segment, `Cancelled` is thrown and the
  // session is left at the time of the last completed event.  No partial
  // Result is returned in that case.
  Result simulate(double t_start, double t_end, int n_points,
                  const CancelCallback& should_continue = {});

  // Samples a segment from the active session honoring an explicit `TimeSpec`
  // — in particular `ts.sample_times`, recording at exactly those sorted
  // output instants in a single SSA pass.  This is the stateful counterpart of
  // `run(TimeSpec)`: it continues from the current session state (no re-seed /
  // reset) and routes through the same `run_ssa` path, which already honors
  // `TimeSpec::sample_times` (see issue #16).  The segment must start at the
  // current session time: `ts.sample_times.front()` — or `ts.t_start` when
  // `sample_times` is empty — must equal `current_time()`.  `should_continue`
  // has the same cooperative-cancellation semantics as the other session calls.
  Result simulate(const TimeSpec& ts, const CancelCallback& should_continue = {});

  // Adds `count` default, unbound molecules of the named imported
  // `MoleculeType` to the active session.
  void add_molecules(const std::string& molecule_type_name, int count);

  // Enumerates the live species in the active session: every complex in
  // the pool is canonicalized and graph-isomorphic complexes are grouped
  // into one `SpeciesRow` with a summed instance count, sorted by the
  // canonical species string.  This is a one-shot pool walk — intended
  // to be called while the simulation is paused (between `simulate()`
  // segments or after a run), not per event.
  // Throws std::runtime_error if no session is active.
  std::vector<SpeciesRow> enumerate_species() const;

  // Writes `enumerate_species()` to `path` as a BNG-format `.species`
  // file: `#` comment header followed by one `<pattern>  <count>` line
  // per species, readable by BNG2.pl's `readNFspecies` (issue #9 §2).
  // Throws std::runtime_error if no session is active or `path` cannot
  // be opened for writing.
  void write_species_file(const std::string& path) const;

  // Returns the number of live complex instances of the species whose
  // canonical BNGL string equals `canonical_species`.
  //
  // `canonical_species` must be a string that RuleMonkey itself emitted
  // — a `SpeciesRow::species` from `enumerate_species()` or a data-line
  // pattern from a written `.species` file.  This method does NOT parse
  // or canonicalize: a string that is not byte-identical to a canonical
  // label RM would emit yields 0, even when it denotes the same
  // species.  For pattern-keyed lookup from an arbitrary hand-written
  // BNGL species string, use `get_species_count(pattern)` below — it
  // parses and canonicalizes the pattern internally.
  //
  // This is a batch query — internally a full pool walk, the same cost
  // as `enumerate_species()`.  To read many species at once, call
  // `enumerate_species()` once and index its rows rather than calling
  // this per species.
  // Throws std::runtime_error if no session is active.
  long species_count(const std::string& canonical_species) const;

  // Returns the total number of live complexes in the active session's
  // pool — the network-free analogue of the total complex/particle
  // population.  Equals the sum of every `enumerate_species()` row
  // count, but is computed without canonicalization and so is cheaper
  // than `enumerate_species()` when only the total is needed.
  // Throws std::runtime_error if no session is active.
  long total_complex_count() const;

  // --- Pattern-keyed species methods (issue #9 §1) -----------------------
  //
  // These four methods accept a runtime BNGL species-pattern string —
  // e.g. `"A(b!1).B(a!1)"` — parsed against the loaded molecule types
  // by RuleMonkey's runtime pattern parser.  Unlike `species_count()`
  // above, the string need NOT be a canonical label RuleMonkey emitted:
  // it is parsed and canonicalized internally.
  //
  // Scope is exact, fully-specified, connected species (issue #9 §1
  // design decision A): every molecule lists every component, every
  // stateful component carries a concrete `~state`, bonds are numeric
  // labels, and the molecules form one complex.  A malformed,
  // under-specified, or wildcard pattern (`!+`, `!?`, omitted
  // components) throws std::runtime_error naming the offending token.
  //
  // All four throw std::runtime_error if no session is active.  They
  // are paused-session calls — none advances logical time or touches
  // the SSA event loop.

  // Returns the number of live complexes that are the species denoted
  // by `pattern`.
  int get_species_count(const std::string& pattern) const;

  // Instantiates `count` fresh copies of the species `pattern` into the
  // active session's pool.  Throws if `count <= 0`.
  void add_species(const std::string& pattern, int count);

  // Removes `count` live copies of the species `pattern` from the pool.
  // Throws if `count <= 0`, or if fewer than `count` copies are live.
  void remove_species(const std::string& pattern, int count);

  // Drives the live count of species `pattern` to exactly `count`,
  // adding or removing the difference.  Throws if `count < 0`.
  void set_species_count(const std::string& pattern, int count);

  // Returns the current active-session molecule count for the named imported
  // `MoleculeType`.
  int get_molecule_count(const std::string& molecule_type_name) const;

  // Returns the current active-session observable values in the same order
  // as `observable_names()`.  Non-const because the engine recomputes lazily
  // — calling this between SSA events forces a fresh evaluation of every
  // observable against current pool state.
  std::vector<double> get_observable_values();

  // Returns the current active-session global-function values in the same
  // order as `function_names()`.  These are the BNGL `begin functions`
  // entries with no local (per-molecule) arguments — the derived
  // quantities models commonly use as their measured outputs.  Non-const
  // for the same reason as `get_observable_values()`: a between-event call
  // forces a fresh evaluation of the observables the functions depend on.
  // Throws std::runtime_error if no session is active.
  std::vector<double> get_function_values();

  // Evaluates the BNGL expression `expr` against the active session's
  // current state and returns its numeric value (issue #9 §1).
  //
  // Resolvable symbols: every declared parameter, the bare clock `t` and
  // the `time()` builtin, every observable, and every global function —
  // all settled against the current pool, exactly as a rate law would
  // see them between events.  The `extra` map supplies additional
  // name=value bindings layered on top; an `extra` name shadows a model
  // symbol on a clash, so callers can probe "what would this expression
  // be if X were Y" without mutating session state.
  //
  // Non-const for the same reason as get_observable_values(): a
  // between-event call recomputes observables (and the functions derived
  // from them) before evaluating.
  // Throws std::runtime_error if no session is active or if `expr` does
  // not compile (syntax error / unresolved identifier).
  double evaluate_expression(const std::string& expr,
                             const std::unordered_map<std::string, double>& extra = {});

  // Returns the numeric parameter value that the simulator would currently use
  // for the named declared parameter, accounting for any active overrides.
  // Derived parameter expressions (e.g., `B = 2*A` declared in the BNGL)
  // re-resolve when their inputs are overridden, so this reflects the
  // post-cascade value, not the parsed-at-load-time value.
  double get_parameter(const std::string& name) const;

  // Saves the active session state to a file. The session must be active.
  // The file embeds a 64-bit fingerprint of the model schema (molecule
  // types, components, allowed states) which `load_state` verifies on
  // read; parameter values, rate constants, and seed species do not
  // participate, so callers may legitimately mutate those between save
  // and load.
  //
  // *Portability caveat*: the RNG state is serialized via the C++ stdlib's
  // `operator<<` for `std::mt19937_64`, which is required by the standard
  // to round-trip within the same implementation but is NOT specified to
  // be byte-identical across different stdlib implementations (libc++ vs
  // libstdc++ vs MSVC STL).  Save/load is reliably reproducible only
  // between RuleMonkey binaries built against the same stdlib.  Crossing
  // stdlibs (e.g. saving with libstdc++ and loading with libc++) may
  // throw on read or, worse, succeed and continue from a divergent RNG
  // state.  If you need cross-stdlib state portability, generate the
  // checkpoint and resume on the same toolchain — RuleMonkey does not
  // currently hand-serialize the Mersenne Twister state.
  void save_state(const std::string& path) const;

  // Creates a new session by loading state from a file. Replaces any
  // existing session.  Throws std::runtime_error if the file's schema
  // fingerprint does not match the model XML this simulator was
  // constructed from (the pool serialization is keyed by molecule-type
  // and component indices, so a mismatched XML would silently produce
  // corrupt trajectories).  The schema must match exactly; non-schema
  // fields (parameter values, rate constants, seed concentrations) may
  // differ between save and load.
  //
  // *Important caveat*: the fingerprint covers molecule-type schema only
  // — molecule type names, component names, and allowed states.  It does
  // NOT cover ReactionRule patterns/operators or Observable patterns.
  // Loading a session into a simulator constructed from a different XML
  // that adds, removes, or restructures rules / observables (while keeping
  // the molecule-type schema unchanged) will succeed silently.  The pool
  // state is valid in absolute terms but the trajectory continuation
  // will reflect the *new* simulator's rules / observables, which may
  // not be what the caller intended.  Do not rely on save/load to pin
  // down a frozen rule set; instead, treat the source XML as the
  // canonical contract and only resume sessions across simulators built
  // from the same XML (or a strict superset that intentionally extends
  // the rule list).
  void load_state(const std::string& path);

  // Reports whether a live runtime/session state is currently active.
  bool has_session() const;

  // Returns the active session's current logical time.  Useful when
  // resuming after `load_state` to feed `simulate(t_start, …)` with
  // the matching segment-start time.  Throws if no session is active.
  double current_time() const;

  // Destroys any live runtime/session state while preserving parsed-model
  // metadata and stored configuration. Safe to call repeatedly.
  void destroy_session();

  // Sets an instance-local override for a declared parameter. The override is
  // applied to subsequent `run()` calls and future `initialize()` calls, and
  // is reflected immediately in `get_parameter()` (including any derived
  // parameters that reference `name`).
  // Throws std::runtime_error if `name` is not a parameter declared in the
  // loaded XML, or if a session is currently active; call `destroy_session()`
  // first to mutate overrides, then re-`initialize()`.
  void set_param(const std::string& name, double value);

  // Clears all instance-local parameter overrides.
  // Throws std::runtime_error if a session is currently active.
  void clear_param_overrides();

  // Sets an instance-local global molecule limit for subsequent runs and
  // future `initialize()` calls.
  // Throws std::runtime_error if a session is currently active.
  void set_molecule_limit(int limit);

  // When true, bimolecular rules only fire between molecules in different
  // complexes (equivalent to NFsim's -bscb flag).  Default: true (strict
  // BNGL semantics).
  // Throws std::runtime_error if a session is currently active.
  void set_block_same_complex_binding(bool value);

  // Returns observable names in XML declaration order captured at
  // construction. The returned vector is a copy.
  std::vector<std::string> observable_names() const;

  // Returns global-function names in XML declaration order captured at
  // construction — the `begin functions` entries with no local arguments,
  // parallel to a Result's `function_names`.  Local functions are
  // excluded; the vector is empty for a model without global functions.
  // The returned vector is a copy.
  std::vector<std::string> function_names() const;

  // Returns parameter names in XML declaration order captured at
  // construction. The returned vector is a copy.
  std::vector<std::string> parameter_names() const;

  // Returns the original constructor XML path string.
  const std::string& xml_path() const;

  // Returns the validated runtime method for this simulator instance.
  Method method() const;

  // Returns the list of unsupported BNGL features detected in the XML at
  // load time.  Each entry has a `Severity` of `Warn` (best-effort run, may
  // still produce useful output) or `Error` (RM cannot honor BNGL semantics
  // for this construct — the rm_driver CLI refuses to run such models
  // unless --ignore-unsupported is passed; embedders should inspect the
  // severities themselves before deciding to call run()).
  const std::vector<UnsupportedFeature>& unsupported_features() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rulemonkey
