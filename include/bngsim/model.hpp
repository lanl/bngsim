// bngsim/include/bngsim/model.hpp — Instance-based network model
//
// All state is per-instance. No globals. Thread-safe for independent instances.

#pragma once

#include "bngsim/expression.hpp"
#include "bngsim/types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bngsim {

// Forward declarations
class ModelBuilder;
class NetFileLoader;
class TableFunction;
struct ModelImplData;

/// Resolved metadata for one registered TableFunction, used by the
/// model-based codegen path to emit the C call for a tfun-backed BNGL
/// function. The index kind is "time", "parameter", or "observable";
/// the corresponding index field carries the 0-based array index for
/// parameter/observable indexing (and is -1 for time-indexed tfuns).
struct TableFunctionSpec {
    std::string name;
    std::string index_kind;
    int index_param_idx = -1;
    int index_obs_idx = -1;
};

class NetworkModel {
    friend class ModelBuilder; // Sole construction path for all input formats
    friend class NetFileLoader;

  public:
    // Forward-declared here; defined in src/model_impl.hpp (internal).
    // Public so that internal helpers (parse functions) can reference the type.
    struct Impl;

    NetworkModel();
    ~NetworkModel();

    // Non-copyable, movable
    NetworkModel(const NetworkModel &) = delete;
    NetworkModel &operator=(const NetworkModel &) = delete;
    NetworkModel(NetworkModel &&) noexcept;
    NetworkModel &operator=(NetworkModel &&) noexcept;

    // ─── Factory ─────────────────────────────────────────────────────────────
    // Load from a .net file.
    static NetworkModel from_net(const std::string &path);

    // ─── Clone ───────────────────────────────────────────────────────────────
    // Deep copy for parallel workers (each worker gets its own model instance)
    NetworkModel clone() const;

    // ─── Parameter access ────────────────────────────────────────────────────
    void set_param(const std::string &name, double value);
    double get_param(const std::string &name) const;
    std::vector<std::string> param_names() const;

    // ─── State management ────────────────────────────────────────────────────
    void reset(); // restore species to initial concentrations

    // Snapshot current species concentrations as the new "initial" state.
    // Subsequent reset() calls restore to this snapshot.
    // Mirrors the BNG saveConcentrations() action.
    void save_concentrations();

    // Set a single species concentration by name.
    // Mirrors the BNG setConcentration() action.
    void set_concentration(const std::string &name, double value);

    // Get a single species concentration by name.
    double get_concentration(const std::string &name) const;

    // ─── Bulk state-vector access (GH #102) ──────────────────────────────────
    // Copy / assign all n_species() concentrations as one contiguous block, in
    // species()/species_names() order, for low-overhead per-step state exchange
    // with an external orchestrator (a hybrid SSA/ODE kernel driving bngsim
    // per-step). O(n_species) with no per-name hash lookups; `out`/`in` must
    // each point to at least n_species() doubles. set_state_from only writes the
    // species concentrations — observables and other derived state are left as
    // they were and get recomputed by the next RHS / observable evaluation.
    void get_state_into(double *out) const;
    void set_state_from(const double *in);

    // ─── Accessors ───────────────────────────────────────────────────────────
    int n_species() const;
    int n_reactions() const;
    int n_observables() const;
    int n_parameters() const;
    int n_functions() const;
    int n_events() const;
    // Number of discontinuity triggers (GH #72): time-dependent piecewise
    // conditions registered as CVODE root functions so the integrator stops
    // at each `time` threshold and cannot step over a narrow forcing pulse.
    int n_discontinuity_triggers() const;

    const std::vector<Species> &species() const;
    const std::vector<Reaction> &reactions() const;
    const std::vector<Observable> &observables() const;
    const std::vector<Parameter> &parameters() const;
    const std::vector<Function> &functions() const;
    const std::vector<Event> &events() const;

    // Forward-sensitivity support classification for this model's events
    // (GH #212). Returns a human-readable reason string when the model has at
    // least one event that is NOT in the Phase-1 (fixed-time) subclass for the
    // requested sensitivity parameters, or std::nullopt when every event is
    // Phase-1-safe (so the integrator can propagate dx/dp across the
    // discontinuity via the s⁺ = J_h·s⁻ + ∂h/∂p jump). An event is Phase-1-safe
    // iff it is persistent, has no delay, its trigger references no species
    // (fixed-time, not state-dependent), and its trigger references none of the
    // requested sensitivity parameters (so the crossing time ∂t*/∂p = 0). Names
    // are resolved against parameters(); an unknown name yields a reason rather
    // than throwing. A model with no events returns std::nullopt.
    std::optional<std::string>
    event_sensitivity_unsupported_reason(const std::vector<std::string> &sens_param_names) const;

    // ExprTk expression-table indices of the discontinuity triggers (GH #72).
    const std::vector<int> &discontinuity_triggers() const;
    const std::vector<StoichEntry> &stoichiometry() const;
    const JacobianSparsity &jacobian_sparsity() const;
    const AnalyticalJacobianData &analytical_jacobian() const;
    const ConservationLaws &conservation_laws() const;

    // ─── Functional analytical Jacobian (GH #76) ─────────────────────────────
    // Per-instance symbolically-derived ∂(rate)/∂x for Functional rate laws.
    const FunctionalJacobianData &functional_jacobian() const;

    // True iff the analytical Jacobian covers every reaction: the shared
    // Elementary structure is available AND (there are no Functional reactions
    // OR their per-instance derivative terms are populated). This is the
    // predicate the CVODE strategy dispatch uses to choose analytical vs FD.
    bool analytical_jacobian_complete() const;

    // Read-only context for the Python sympy differentiator: every Functional
    // reaction's rate-law expression, the function map (for inlining), the
    // observable groups, per-species amount/volume metadata, and the
    // constant-parameter names. The engine NEVER calls Python; Python reads
    // this, differentiates, and writes back via set_functional_jacobian.
    FunctionalJacobianContext functional_jacobian_context() const;

    // Compile the supplied per-reaction derivative expressions into this
    // instance's evaluator and populate the functional Jacobian. Returns true
    // on success; false (leaving the model on the FD path) if any expression
    // fails to compile or references a Jacobian entry outside the sparsity
    // pattern. All-or-nothing: callers pass terms for every Functional reaction.
    bool set_functional_jacobian(const std::vector<FunctionalJacobianInput> &terms);

    // Assemble the full dense analytical Jacobian at (t, conc) into a
    // column-major n×n buffer (jac[j*n + i] = ∂f_i/∂x_j): Elementary closed-form
    // + Functional symbolic contributions + fixed-species row zeroing. The CVODE
    // dense callback delegates here so a single implementation is exercised by
    // both integration and the entrywise FD validation. GH #76.
    void fill_dense_analytical_jacobian(double t, const double *conc, double *jac);

    // Assemble the analytical Jacobian at (t, conc) into a CSC numeric-value
    // buffer of length nnz (jacobian_sparsity().col_ptrs[n]), indexed by the
    // sparsity pattern's data index — the same math as the dense fill, written
    // sparsely. Zeroes fixed-species rows. The CVODE sparse callback delegates
    // here (after reinstalling the CSC structure), and the large-model branch of
    // the set_functional_jacobian self-check uses it to validate without ever
    // allocating a dense n×n matrix (GH #151). Caller supplies a zero-or-not
    // buffer; this memsets it to 0 first.
    void fill_sparse_analytical_jacobian(double t, const double *conc, double *vals);

    // (species_idx0, param_idx0) pairs for species whose initial
    // concentration is set directly by a parameter (.net "begin species"
    // entries with a parameter name in the IC column). Used by forward
    // sensitivity to seed s(0) = ∂y(0)/∂p.
    const std::vector<std::pair<int, int>> &species_ic_param_refs() const;

    std::vector<std::string> species_names() const;
    // 0-based indices of species with `reported == true`, in species order
    // (GH #71). The trajectory-output layer projects Result species columns to
    // this subset. When every species is reported (the common case) the caller
    // skips wiring it and the projection is a no-op.
    std::vector<std::size_t> reported_species_indices() const;
    // Per-reported-species volume_factor (V_c), in reported-species order — the
    // narrow accessor the trajectory-output layer needs to convert stored
    // concentrations back to amounts (Result.as_roadrunner). Avoids
    // materializing the whole model as a Python dict via codegen_data() just to
    // read this one field (T7). Structure-fixed, so the caller may cache it.
    std::vector<double> reported_volume_factors() const;
    std::vector<std::string> observable_names() const;
    std::vector<std::string> function_names() const;
    std::vector<std::string> load_warnings() const;

    // Evaluate all functions and return their current values (for recording in Result).
    std::vector<double> function_values() const;

    // Function values from the most recent evaluate_functions() call, indexed by
    // function declaration order (parallel to function_names()). Populated as a
    // side effect of evaluate_functions(); reading it avoids a second interpreted
    // ExprTk pass over every function at each output row (GH #136). Valid only
    // after evaluate_functions() has run; sized to n_functions() once it has.
    const std::vector<double> &function_value_cache() const;

    // ─── RHS evaluation (used by simulators) ─────────────────────────────────
    // Compute dx/dt for all species. conc and derivs are n_species-sized arrays.
    // Uses 0-based indexing for the arrays.
    //
    // GH #106: for a model that references rateOf(species), this first runs a
    // derivative *probe* (compute the RHS into a scratch buffer and publish it
    // to the live current_derivs the rate_of__<species> accessors read), then
    // computes the real RHS with rateOf evaluating to the just-published dx/dt.
    // One probe is exact because every corpus rateOf argument is a species whose
    // derivative is independent of the rateOf consumers (no algebraic loop).
    // Byte-identical to the single-pass body for models without rateOf.
    void compute_derivs(double t, const double *conc, double *derivs);

    // True iff the model references rateOf(species) (GH #106). Simulators use
    // this to refresh current_derivs before evaluating rateOf-bearing triggers
    // (CVODE root fn) and to reject rateOf under SSA.
    bool uses_rateof() const;

    // Refresh the live rateOf derivative buffer (current_derivs) from a probe at
    // (t, conc) without returning the RHS. No-op when !uses_rateof(). Used by the
    // CVODE root function before evaluating event triggers that read rateOf.
    void refresh_rateof_derivs(double t, const double *conc);

    // Compute propensity for a single reaction (for SSA).
    // conc is 0-based n_species array.
    double compute_propensity(int rxn_index, const double *conc);

    // GH #190 — structure-specialized propensity vector: reads each reaction's
    // rate constant from a runtime params array `p[]` (signature gains
    // `const double* p`) rather than baking it, so the source/.so cache key
    // depends only on model structure — one compile per model, reused across all
    // parameter points (a fit) and replicates (an ensemble). Returns
    // {source, n_unsupported}; reactions not representable as a static-volume
    // mass-action monomial emit `a[r]=0.0` and increment n_unsupported, so the
    // caller should only trust the kernel when n_unsupported == 0.
    std::pair<std::string, int> emit_ssa_propensity_source_structure() const;

    // Update observable group totals from current species concentrations.
    // conc is 0-based n_species array.
    void update_observables(const double *conc);

    // Evaluate all functions (updates variable parameters).
    void evaluate_functions(double t);

    // ─── RHS observable/function-eval gate instrumentation (T1) ──────────────
    // compute_derivs_core() refreshes observable totals + function-bound
    // parameters only when has_functions() — the ExprTk evaluator (the sole RHS
    // consumer of observable sums) runs solely inside evaluate_functions(), so
    // for a pure mass-action model both passes are dead work and are skipped.
    // These counters expose that gate for tests/benchmarks: rhs_eval_count() is
    // every RHS call; rhs_observable_eval_count() is the subset that ran the two
    // passes. Mass-action ⇒ the latter stays 0; functional ⇒ the two are equal.
    bool rhs_evaluates_observables() const; // the build-time gate decision
    std::uint64_t rhs_eval_count() const;
    std::uint64_t rhs_observable_eval_count() const;
    void reset_rhs_counters();

    // ─── System time ────────────────────────────────────────────────────────
    double current_time() const;
    void set_current_time(double t);

    // ─── Pre-equilibration / carry-over sensitivity state (GH #210) ──────────
    // A two-phase pre-equilibration (ADR-0052) runs the same persistent
    // Simulator across two run() calls with NO reset between them: the species
    // state is carried over so the equilibration steady state is the
    // measurement phase's initial condition. These hooks let the CVODE
    // simulator (a) know the current species state is carried-over rather than
    // the fresh load/reset ICs, and (b) thread the prior phase's final forward-
    // sensitivity matrix dx/dθ into the next phase's yS(0) seed.

    // True iff the current species state is carried-over dynamics — i.e. it was
    // produced by integrating a previous run() (whose θ-derivative dx/dθ is
    // generally nonzero), as opposed to a fresh initial condition. The simulator
    // sets it at each run's state write-back; reset()/save_concentrations()
    // clear it. set_concentration()/set_state_from() (literal/external IC
    // assignment, θ-independent) do NOT set it. Forward sensitivities requested
    // on a dirty state without carry_sensitivities would be silently wrong
    // (fresh seeding assumes ∂x(0)/∂θ = 0), so the simulator raises in that case.
    bool ic_state_dirty() const;
    void set_ic_state_dirty(bool dirty);

    // The prior phase's final species forward-sensitivity matrix dx/dθ, stored
    // row-major as [species_idx * n_params + param_idx] (n_species × n_params),
    // captured by the simulator at the end of a parameter-sensitivity run. The
    // accompanying parameter-name vector identifies the columns and must match
    // the next run's sensitivity_params for the seed to be consumed. Empty when
    // no seed is pending. Cleared by reset()/save_concentrations()/
    // set_concentration()/set_state_from() and by any non-sensitivity run
    // (which advances state without tracking dx/dθ, invalidating the seed).
    const std::vector<double> &pending_sens_seed() const;
    const std::vector<std::string> &pending_sens_seed_param_names() const;
    void set_pending_sens_seed(std::vector<double> seed, std::vector<std::string> param_names);
    void clear_pending_sens_seed();

    // ─── Table functions ────────────────────────────────────────────────────

    /// Add a table function from a .tfun file.
    /// @param name        Function name (e.g., "cumNcases")
    /// @param filepath    Path to .tfun file
    /// @param index_name  Index variable: "time" (default), or a parameter/observable name
    /// @param method      Interpolation method: "linear" (default) or "step"
    /// @param header_name Optional .tfun column-2 header validation name; defaults
    ///                    to `name`. Differs from `name` only for embedded-form
    ///                    tfun where the runtime ID is synthetic but the .tfun
    ///                    file still labels its value column with the original
    ///                    BNG function name.
    void add_table_function(const std::string &name, const std::string &filepath,
                            const std::string &index_name = "time",
                            const std::string &method = "linear",
                            const std::string &header_name = "");

    /// Add a table function from in-memory data.
    /// @param name       Function name
    /// @param xs         Sorted x values (monotonically increasing)
    /// @param ys         Corresponding y values
    /// @param index_name Index variable name
    /// @param method     Interpolation method: "linear" (default) or "step"
    void add_table_function(const std::string &name, const std::vector<double> &xs,
                            const std::vector<double> &ys, const std::string &index_name = "time",
                            const std::string &method = "linear");

    /// Number of registered table functions.
    int n_table_functions() const;

    /// Names of all registered table functions.
    std::vector<std::string> table_function_names() const;

    /// Evaluate the tf_id-th table function at index value x. Used by the
    /// codegen RHS path: the C thunk receives a void* pointing at this model
    /// and dispatches by tf_id (0-based, in the order tfun-bodied functions
    /// appear in the .net "begin functions" block).
    /// @throws std::out_of_range if tf_id is out of bounds.
    double evaluate_table_function_at(int tf_id, double x) const;

    /// Specs for every registered table function, in the same order as
    /// table_function_names() and the runtime dispatch ID. The model-based
    /// codegen path uses this to emit the right index expression
    /// (t / p[idx] / obs[idx]) at each tfun call site.
    std::vector<TableFunctionSpec> table_function_specs() const;

    // ─── Expression evaluator access ─────────────────────────────────────────
    ExpressionEvaluator &evaluator();

  private:
    std::unique_ptr<Impl> impl_;
    void set_load_warnings_(std::vector<std::string> warnings);

    /// The single-pass RHS body (GH #106). compute_derivs() and
    /// refresh_rateof_derivs() wrap this: the rateOf probe and the real pass are
    /// both `compute_derivs_core` calls, differing only in which buffer they read
    /// (current_derivs, via the rate_of__<species> accessors) and write.
    void compute_derivs_core(double t, const double *conc, double *derivs);

    /// Internal helper: register a TableFunction with the expression evaluator
    /// and bind its index pointer to the appropriate model variable.
    void register_table_function_(TableFunction &tf);
};

} // namespace bngsim
