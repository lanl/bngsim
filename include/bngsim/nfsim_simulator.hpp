// bngsim/include/bngsim/nfsim_simulator.hpp — NFsim stochastic simulator wrapper
//
// Wraps NFsim as an in-process library for rule-based stochastic simulation.
// Eliminates subprocess overhead for method=>"nf" models.
//
// API mirrors SsaSimulator: NfsimSimulator::run(TimeSpec, seed) → Result
//
// Supports in-process NFsim simulation with the same table-function behavior
// exposed by the rest of bngsim.

#pragma once

#include "bngsim/result.hpp"
#include "bngsim/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bngsim {

// ─── NFsim Stochastic Simulator ──────────────────────────────────────────────
//
// Wraps the vendored NFsim library for in-process rule-based simulation.
// Each run() call re-parses the XML (~1ms) for a clean state, seeds the RNG,
// and captures observable counts into a bngsim Result (no file I/O).
//
// Thread-safety: A static mutex in StreamSuppressor serializes all NFsim
// operations that touch cout/cerr. This means concurrent NFsim instances
// run correctly but are serialized (not parallel). ODE/SSA paths are
// unaffected and remain fully parallel.
//
// Same-complex binding blocking (NFsim CLI -bscb) defaults ON in bngsim
// for correctness on BLBR-style aggregation models (NFsim CLI defaults it
// off; bngsim deliberately differs). Use set_block_same_complex_binding(false)
// to opt out.
//
// Error handling: NFsim errors throw std::runtime_error instead of calling
// exit(). The Python wrapper catches these as RuntimeError/SimulationError.
//
// Usage:
//   NfsimSimulator sim("model.xml");
//   auto result = sim.run({0.0, 10.0, 101}, /*seed=*/42);
//   // result.observable_names(), result.observable_data() — all in memory
//
class NfsimSimulator {
  public:
    // Construct with path to BNG XML file.
    // The XML is NOT parsed until run() is called.
    explicit NfsimSimulator(const std::string &xml_path);
    ~NfsimSimulator();

    // Non-copyable, non-movable (owns NFsim System*)
    NfsimSimulator(const NfsimSimulator &) = delete;
    NfsimSimulator &operator=(const NfsimSimulator &) = delete;

    // ─── Simulation ──────────────────────────────────────────────────────────

    /// Run a single simulation trajectory (stateless — re-parses XML each call).
    ///
    /// @param times            Time specification (t_start, t_end, n_points)
    /// @param seed             RNG seed for deterministic reproducibility
    /// @param timeout_seconds  Wall-clock budget in seconds; `<= 0` disables
    ///                         it. When the budget is exceeded between output
    ///                         points, the call throws `bngsim::TimeoutError`.
    /// @return Result with observable time courses (no species-level data for NFsim)
    ///
    /// Each call re-parses XML for clean initial state. Observable counts are
    /// captured at evenly-spaced sample points using stepTo().
    Result run(const TimeSpec &times, uint64_t seed = 42, double timeout_seconds = 0.0);

    // ─── Session API (stateful, multi-action workflows) ──────────────────────
    //
    // For models that need state persistence between actions (e.g., T4:
    // equilibrate → setConcentration → simulate), use the session API instead
    // of run(). The session keeps the NFsim System alive between calls.
    //
    // Usage:
    //   sim.initialize(42);           // parse XML, prepare, seed
    //   sim.step_to(600.0);           // equilibrate (state preserved)
    //   sim.add_molecules("L", 1266); // add ligand molecules
    //   auto result = sim.simulate(0.0, 60.0, 13);  // simulate with output
    //   sim.destroy_session();        // cleanup

    /// Initialize a simulation session. Parses XML, seeds RNG, prepares system.
    /// Must be called before step_to() / simulate() / add_molecules().
    void initialize(uint64_t seed);

    /// Advance simulation to target time (no output captured). State preserved.
    void step_to(double time);

    /// Run simulation from t_start to t_end, capturing n_points output snapshots.
    /// Returns Result with observable time courses.
    ///
    /// @param t_start          Segment start time (user-facing label)
    /// @param t_end            Segment end time (user-facing label)
    /// @param n_points         Number of output snapshots (>= 2)
    /// @param timeout_seconds  Wall-clock budget in seconds; `<= 0` disables
    ///                         it. When the budget is exceeded between output
    ///                         points, the call throws `bngsim::TimeoutError`.
    ///                         The session state remains usable for
    ///                         `destroy_session()` after the exception
    ///                         propagates.
    /// @param relative_time    When true, the time axis stamped onto the
    ///                         returned Result is offset to start at 0 (i.e.
    ///                         elapsed time since t_start, running to
    ///                         t_end - t_start), matching BNG2.pl's
    ///                         simulate_nf output convention. The internal
    ///                         NFsim clock and session_logical_time are
    ///                         unaffected, so multi-segment threading still
    ///                         works. Default false preserves the
    ///                         absolute-time labelling used elsewhere in
    ///                         bngsim.
    /// @param sample_times     Optional explicit output instants (GH #184).
    ///                         When non-empty, overrides the uniform
    ///                         t_start..t_end / n_points grid: the Result is
    ///                         labelled with exactly these times (sorted
    ///                         ascending, treated as absolute in the caller's
    ///                         frame) and the live NFsim clock is advanced from
    ///                         the current session_logical_time by
    ///                         (sample_times[i] - sample_times[0]). This gives
    ///                         the stateful session the same explicit-time
    ///                         capability run(TimeSpec) already has, without
    ///                         re-seeding or resetting state. An empty list is
    ///                         the uniform path, byte-identical to before.
    Result simulate(double t_start, double t_end, int n_points, double timeout_seconds = 0.0,
                    bool relative_time = false, const std::vector<double> &sample_times = {});

    /// Add count molecules of the named MoleculeType in default (unbound) state.
    /// Used to implement setConcentration() for NFsim models.
    void add_molecules(const std::string &molecule_type_name, int count);

    /// Get current count of a MoleculeType by name.
    int get_molecule_count(const std::string &molecule_type_name) const;

    /// Get current count of an exact single-molecule BNGL species pattern.
    ///
    /// Example: "X(p~0,y)" counts live X molecules with p state 0 and y
    /// unbound. Patterns must be exact enough to identify one instantiable
    /// species; ambiguous or multi-molecule patterns fail with a clear error.
    int get_species_count(const std::string &pattern) const;

    /// Add count copies of an exact single-molecule BNGL species pattern.
    void add_species(const std::string &pattern, int count);

    /// Remove count live copies matching an exact single-molecule BNGL species pattern.
    void remove_species(const std::string &pattern, int count);

    /// Set the live count for an exact single-molecule BNGL species pattern.
    void set_species_count(const std::string &pattern, int count);

    /// Get current observable names (session must be active).
    std::vector<std::string> get_observable_names() const;

    /// Get current observable values (session must be active).
    std::vector<double> get_observable_values() const;

    /// Write the live System's molecular species to a BNG-format
    /// ``.species`` file. Mirrors the artifact BNG2.pl emits when
    /// running ``simulate({method=>"nf", get_final_state=>1, ...})``.
    /// The file lists one species pattern per line with its count;
    /// PyBioNetGen consumes this for state writeback between NF
    /// segments. The session is unchanged.
    void save_species(const std::string &path);

    /// Get a parameter value by name (session must be active).
    /// Used to evaluate expressions like "EGF_copy_number" for setConcentration.
    double get_parameter(const std::string &name) const;

    /// Snapshot the live System's full molecular state (counts, component
    /// states, and bonds) into the named slot ``label`` for later in-process
    /// restore. Mirrors the BNG ``saveConcentrations()`` /
    /// ``saveConcentrations("name")`` action. Each distinct label owns its own
    /// snapshot, so multiple named states coexist; saving to the same label
    /// again overwrites just that slot. The empty label ``""`` is the default
    /// (unlabeled) slot. The session is otherwise unchanged.
    void save_concentrations(const std::string &label = "");

    /// Restore the molecular state captured by save_concentrations() into the
    /// named slot ``label`` (``""`` = default/unlabeled). Mirrors the BNG
    /// ``resetConcentrations()`` / ``resetConcentrations("name")`` action.
    /// Throws std::runtime_error if no snapshot has been saved under ``label``.
    void restore_concentrations(const std::string &label = "");

    /// Whether a save_concentrations() snapshot is available to restore under
    /// the named slot ``label`` (``""`` = default/unlabeled).
    bool has_saved_concentrations(const std::string &label = "") const;

    /// Names of every currently-held save_concentrations() slot, including the
    /// default/unlabeled slot as ``""`` when present. Sorted ascending.
    std::vector<std::string> saved_concentration_labels() const;

    /// Whether a session is currently active.
    bool has_session() const;

    /// Destroy the current session, freeing the NFsim System.
    void destroy_session();

    // ─── Configuration ───────────────────────────────────────────────────────

    /// Set a parameter override. Applied before prepareForSimulation() on each run.
    /// Also re-evaluates every dependent parameter from its BNG XML expression
    /// (via ExprTk) so cascades propagate. Safe to call before *or* after
    /// initialize(); when called post-init, the live NFsim System is updated
    /// in place and reaction rates are refreshed.
    ///
    /// Pre-init writes are baked into the XML that NFsim parses, so
    /// `<Species concentration="X">` seed-species expressions and
    /// `<RateConstant value="_rateLawN"/>` references both pick up the
    /// override-resolved values when the agent population is created.
    /// Post-init writes do NOT alter the live agent population — they only
    /// refresh reaction rates from the new parameter namespace.
    /// @param name  Parameter name (must exist in XML)
    /// @param value New value
    void set_param(const std::string &name, double value);

    /// Clear all parameter overrides.
    void clear_param_overrides();

    /// Evaluate an arbitrary BNG expression with the current overrides applied.
    /// Useful for re-rendering BNG functions or rate laws against the
    /// override namespace without piping every expression through Python.
    ///
    /// @param expr        ExprTk-compatible expression string
    /// @param extra       Additional name=value bindings layered on top of
    ///                    the simulator's current overrides
    /// @return Numeric value
    double evaluate_expression(const std::string &expr,
                               const std::unordered_map<std::string, double> &extra = {}) const;

    /// Set global molecule limit (default: INT_MAX, no artificial limit).
    void set_molecule_limit(int limit);

    /// Enable or disable NFsim connectivity inference at XML initialization.
    /// Default behavior is the conservative compatibility path (`false`).
    void set_connectivity(bool enabled);

    /// Opt into NFsim v1.14.3 selector compatibility.
    /// Default behavior uses the clean single-draw selector path.
    void set_nfsim_v1143_compat(bool enabled);

    /// Block same-complex binding (NFsim CLI ``-bscb``).
    /// When enabled, two reactant patterns in a bimolecular rule cannot
    /// match molecules in the same complex. Implies complex bookkeeping
    /// (``-cb``). Default: ``true`` in bngsim (NFsim CLI default is off,
    /// but bngsim defaults it on for correctness on BLBR/aggregation models).
    void set_block_same_complex_binding(bool enabled);

    /// Set the universal traversal limit (NFsim CLI ``-utl N``).
    /// Negative values use NFsim's auto-computed suggested limit
    /// (``ReactionClass::NO_LIMIT``). Default: -1 (auto).
    void set_traversal_limit(int limit);

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Path to the XML file.
    const std::string &xml_path() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bngsim
