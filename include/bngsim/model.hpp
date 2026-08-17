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
    // Writes the parameter, re-evaluates every expression-valued parameter that
    // reads it, and re-resolves every species initial condition that names one
    // of those parameters (issue #79) — `A() Stot` declares A's IC to BE Stot,
    // so a dose scan over Stot has to move it. See refresh_param_ref_ics() for
    // which of `initial_conc` / `concentration` follows and when.
    //
    // Issue #188 — writing a *derived* (expression-backed) parameter overrides
    // its expression, BNG `setParameter` style, and the override is keyed on the
    // value: it lasts exactly while the parameter does not hold what its
    // expression produces, so writing that value back lifts it. `force_override`
    // pins the parameter independently of its value, for the one caller that
    // needs "treat this as an independent input" to survive an identity write —
    // `bngsim.jax.differentiable_solve(flat=True)`, whose documented legacy
    // semantics are that every parameter, derived included, is its own axis.
    // That mode used to get it for free from an unconditional detach; asking for
    // it by name is what lets an ordinary write round-trip.
    void set_param(const std::string &name, double value, bool force_override = false);
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

    // The IC baseline `species[].initial_conc` as one contiguous block, same
    // ordering (GH #113). reset() returns the live state to exactly this, so
    // comparing the two answers "is this species still at the initial condition
    // the model declares, or has an assignment superseded it?" — which is what
    // decides whether a parameter-graph ∂x_i(0)/∂p seed (issue #43) still applies.
    void get_initial_state_into(double *out) const;

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
    // (GH #212, widened by issue #49 and issue #144). Returns a human-readable
    // reason string when the model has at least one event that forward
    // sensitivity cannot propagate through for the requested sensitivity
    // parameters, or std::nullopt when every event is covered by the jump
    //     s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p
    // the integrator applies at each fire. An event is covered iff it has no
    // *effective* delay and its ∂t*/∂p is known, which holds when
    //   * its trigger reads no species/observable/rate and none of the
    //     requested sensitivity parameters, so ∂t*/∂p = 0 (GH #212 Phase 1), or
    //   * its index is in `event_time_compensated`, meaning the Python detector
    //     resolved the trigger's threshold and will supply ∂t*/∂p (issue #49), or
    //   * its trigger reduces to a single relational comparison, so the solver
    //     can differentiate the crossing itself at each fire — the implicit
    //     function theorem on g(x(t*), p, t*) = 0 (issue #144). This is the
    //     case that covers a *state-dependent* trigger like `v > 30`, whose
    //     crossing time moves with every parameter through the trajectory.
    // Persistence only matters when there IS a delay: per SBML L3v2 §4.11.3 a
    // non-persistent trigger can only cancel a fire during the window between
    // trigger time and execution time, which a zero-delay event has none of.
    // Names are resolved against parameters(); an unknown name yields a reason
    // rather than throwing. A model with no events returns std::nullopt.
    std::optional<std::string>
    event_sensitivity_unsupported_reason(const std::vector<std::string> &sens_param_names,
                                         const std::vector<int> &event_time_compensated = {}) const;

    // ─── Event-trigger residuals for a moving crossing (issue #144) ──────────
    //
    // The CVODE root function is the *boolean* trigger offset by 0.5, which is
    // a step and carries no derivative. Differentiating the crossing time needs
    // the trigger's residual instead: for `lhs ⋈ rhs` (⋈ one of < <= > >=) that
    // is `lhs − rhs`, whose zero set is the crossing surface. Only the ratio
    //     dt*/dp = −(∂g/∂x·S + ∂g/∂p) / (∂g/∂t + ∂g/∂x·f)
    // is ever formed, so the residual's overall sign is irrelevant and no
    // orientation is imposed.
    //
    // event_trigger_residual_expr() returns the ExprTk expression id of that
    // residual for event `event_idx0`, compiling it into this model's evaluator
    // on first use, or -1 when the trigger is not a single relational
    // comparison (a conjunction, a negation, an equality, or a bare boolean).
    // `why`, when non-null, receives the reason for a -1. Results are cached
    // per event; the cache is derived from the trigger's own text, so a clone
    // re-derives it rather than copying an evaluator id across evaluators.
    int event_trigger_residual_expr(int event_idx0, std::string *why = nullptr) const;

    // Species whose perturbation can move that residual — the finite-difference
    // support of ∂g/∂x. Tight (the species the residual names, plus the species
    // behind any observable it names) when every other symbol it reads is
    // state-free; every species otherwise, because a rateOf accessor or a
    // parameter carrying an SBML assignment rule can read the whole state.
    // Empty when event_trigger_residual_expr() is -1.
    const std::vector<int> &event_trigger_residual_species(int event_idx0) const;

    // Does this event's trigger read live state (a species concentration, an
    // observable total, or a rateOf accessor)? Such a trigger's crossing time
    // moves with the parameters *through the trajectory* even when it names
    // none of them, so its ∂t*/∂p is non-zero and must be differentiated at the
    // crossing rather than resolved ahead of the run (issue #52 / issue #144).
    bool event_trigger_is_state_dependent(int event_idx0) const;

    // Indices of the events whose ∂t*/∂p the solver differentiates at each fire
    // (state-dependent trigger + a usable residual). The Python guard subtracts
    // these from its own "blocked" set so the two classifications cannot drift.
    std::vector<int> events_with_runtime_event_time_sens() const;

    // Which species and which parameters a central finite difference over
    // `expr_idx` has to perturb — following observables to the species behind
    // them and derived/rule-bound parameters to the primaries behind them, so a
    // pruned difference cannot report a zero where there is a derivative. Pass
    // nullptr for either output to skip it. See model.cpp for the two
    // silent-zero shapes this exists to close.
    void expression_support(int expr_idx, std::vector<int> *species_out,
                            std::vector<int> *params_out) const;

    // ─── Rate-law switch conditions that read model state (issue #150) ───────
    //
    // The rate-law twin of the state-dependent event trigger issue #144 covers.
    // A condition such as `Virus < 1` inside a `piecewise`/`if()` rate law flips
    // a branch of f at a crossing whose time moves with every parameter through
    // the trajectory, so dx/dθ is DISCONTINUOUS there by the saltation term
    // (f⁻ − f⁺)·dt*/dθ. Nothing carries that term on its own: the analytic
    // sensitivity RHS differentiates the `Piecewise` to a clean in-branch value
    // with no boundary delta, and CVODES' internal difference quotient
    // integrates the variational equation straight across the crossing.
    //
    // What the solver needs is the same object issue #144 builds for a trigger —
    // a residual `lhs − rhs` whose zero set is the crossing surface — reached
    // from a condition's source text instead of an event index. That residual is
    // registered as a CVODE root (so the crossing is LOCATED rather than stepped
    // over, which is also what keeps the integrator out of issue #82's pit) and
    // differentiated there.
    //
    // Resolving one compiles the residual into this model's evaluator; results
    // are cached by source text and, like the trigger-residual cache, are NOT
    // copied by clone() — an expression id means something else in another
    // evaluator. Returns nullptr and sets `why` when the condition is not one
    // this machinery can locate and differentiate: a conjunction, a negation, a
    // comparison one of whose sides is itself a comparison (its residual would be
    // a difference of two booleans — a step, with no gradient to root on), or a
    // comparison that reads no live state at all.
    //
    // An EQUALITY is admitted here where event_trigger_residual_expr() above
    // refuses it (issue #381). The two want different things from the same text:
    // an event needs a rising edge, which `x == c` does not have, while a switch
    // needs the surface its branch can change across, which for `x == c` is the
    // `x − c = 0` that `x < c` names too. A lone equality then measures a zero
    // branch gap at that root and applies no jump, and a redundant one —
    // MODEL2003190004 spells `APC <= 0.2` as `(APC == 0.2) or (APC < 0.2)` —
    // resolves to the residual its sibling atom already registered.
    struct StateSwitch {
        // Compiled `(lhs)-(rhs)`. Rooted on directly — a smooth residual, not
        // the boolean-minus-0.5 step the event and GH #72 roots use, so CVODE
        // brackets the crossing on the same function that is differentiated.
        int residual_expr_idx = -1;
        // Species a central difference of the residual has to perturb
        // (NetworkModel::expression_support).
        std::vector<int> species;
        // The residual's preprocessed text. Identifies the *crossing* rather
        // than its spelling, so `X<1` and `X<=1` resolve to one root.
        std::string residual_source;
    };
    const StateSwitch *state_switch(const std::string &condition_src,
                                    std::string *why = nullptr) const;

    // ExprTk expression-table indices of the discontinuity triggers (GH #72).
    const std::vector<int> &discontinuity_triggers() const;
    const std::vector<StoichEntry> &stoichiometry() const;
    const JacobianSparsity &jacobian_sparsity() const;

    // Same pattern jacobian_sparsity() returns, with the Curtis-Powell-Reid
    // coloring materialized (computed on first call, then cached and shared
    // across clones). build() does not color — only the sparse colored-FD
    // Jacobian callback consumes it — so this is the only accessor whose result
    // has n_colors / colors / color_groups populated. Every pattern with at
    // least one structural nonzero gets a coloring, however dense: a fully dense
    // one degenerates to one column per color, i.e. plain FD.
    const JacobianSparsity &ensure_jacobian_coloring() const;

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

    // The divisor `resolve_ic_from_param` applies to each species_ic_param_refs
    // entry, in the same order (issue #170 stage 3). An amount_valued species
    // stores amount/V_c, so the parameter that names its IC names an *amount*
    // and the stored value — and therefore ∂(stored IC)/∂p — carries a 1/V_c
    // the parameter graph on the Python side cannot see. 1.0 for every other
    // ref, which is every `.net` model and every hOSU=false SBML species.
    std::vector<double> species_ic_param_ref_divisors() const;

    // ∂(stored initial condition)/∂(compartment size), for the species whose
    // stored IC moves when a compartment size is written (issue #170 stage 3).
    // Returns (species_idx0, volume_param_idx0, d_ic_d_volume) triples; empty
    // for `.net`, for a model with no writable compartment size, and once
    // save_concentrations() has redefined the baseline. See model.cpp.
    std::vector<std::tuple<int, int, double>> compartment_ic_sens_seeds() const;

    std::vector<std::string> species_names() const;
    // 0-based indices of species with `reported == true`, in species order
    // (GH #71). The trajectory-output layer projects Result species columns to
    // this subset. When every species is reported (the common case) the caller
    // skips wiring it and the projection is a no-op.
    std::vector<std::size_t> reported_species_indices() const;

    // ─── Pure sinks: write-only accumulator species (issue #74) ──────────────
    // 0-based ascending indices of the species this network only ever writes to:
    // the "degraded" / "produced" / "secreted" pools a BNGL model carries to
    // count cumulative flux. Such a species has a constant non-zero derivative
    // for as long as its producing reactions fire, so ||f(y)||₂/n can never
    // reach tol and steady_state() reports failure however long it integrates —
    // even when every other species has settled. This is what a caller passes
    // (negated) as SteadyStateOptions::steady_state_mask to solve f(y) = 0 on
    // the subspace that HAS a steady state.
    //
    // Three clauses, all structural — no user annotation, nothing measured:
    //   1. the species appears as a product of at least one reaction, and
    //   2. as a reactant of none, and
    //   3. its Jacobian column is structurally empty, i.e. no species'
    //      derivative depends on it.
    //
    // (1)+(2) are the issue's definition. (3) is what makes excluding it
    // provably harmless to the rest of the system, and it is not implied: an
    // Elementary rate law reads only its reactants, but a Functional one reads
    // observables, so a product-only species named in an observable a rate law
    // consumes still feeds back into the dynamics (as does a promoted
    // compartment-volume species, GH #171). jacobian_sparsity() already
    // resolves that dependency transitively through function references
    // (GH #164), so clause 3 is one column-pointer comparison.
    //
    // `fixed` ($-prefixed boundary-condition) species are excluded: compute_derivs
    // zeroes their derivative, so they contribute nothing to the residual and
    // never block convergence in the first place.
    std::vector<int> pure_sink_species() const;

    // Per-reported-species volume_factor (V_c), in reported-species order — the
    // narrow accessor the trajectory-output layer needs to convert stored
    // concentrations back to amounts (Result.as_roadrunner). Avoids
    // materializing the whole model as a Python dict via codegen_data() just to
    // read this one field (T7). Structure-fixed, so the caller may cache it.
    std::vector<double> reported_volume_factors() const;
    std::vector<std::string> observable_names() const;
    std::vector<std::string> function_names() const;
    // Defining expression of each function, parallel to function_names(). Exists
    // so the GH #333 guard can test every rate law for a logarithm without
    // building the whole functional_jacobian_context(), whose function_map runs
    // to tens of thousands of entries on a genome-scale model and would be paid
    // on every Model construction.
    std::vector<std::string> function_expressions() const;
    std::vector<std::string> load_warnings() const;

    // Per-function evaluation expression, parallel to function_names(). Empty
    // where the value is computed from the declared expression, which is every
    // function the GH #333 guard did not rewrite.
    std::vector<std::string> function_eval_expressions() const;

    // Point a function's *value* at a different expression and recompile its
    // evaluator (GH #333), leaving the declared expression — the one that gets
    // differentiated — untouched. Returns false if no function by that name
    // exists; throws if the replacement does not compile, leaving the original
    // in place.
    //
    // The one caller is the Python-side zero-base logarithm guard, which rewrites
    // ``S^n*ln(S)`` to its limit at ``S == 0``. That rewrite is symbolic and
    // therefore lives in Python (sympy), but ``.net`` models are built entirely
    // in C++ via ModelBuilder, so there is no build-time seam to apply it at —
    // hence a post-build replacement here, which both the interpreted ExprTk
    // evaluator and the codegen C emitter pick up because both read this field.
    bool set_function_eval_expression(const std::string &name, const std::string &expression);

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

    // True iff the current species state's θ-derivative is NOT the fresh-start
    // seed — i.e. the state was produced by integrating a previous run() (whose
    // dx/dθ is generally nonzero) rather than being a θ-independent initial
    // condition. The simulator sets it at each run's state write-back; reset()
    // clears it unless the IC baseline itself carries a derivative (GH #81), and
    // save_concentrations() clears it only when there is no carried derivative to
    // hand the new baseline. set_concentration()/set_state_from()
    // (literal/external IC assignment, θ-independent) do NOT set it. Forward
    // sensitivities requested on a dirty state without carry_sensitivities would
    // be silently wrong (fresh seeding assumes ∂x(0)/∂θ = 0), so the simulator
    // raises in that case.
    bool ic_state_dirty() const;
    void set_ic_state_dirty(bool dirty);

    // The prior phase's final species forward-sensitivity matrix dx/dθ, stored
    // row-major as [species_idx * n_params + param_idx] (n_species × n_params),
    // captured by the simulator at the end of a parameter-sensitivity run. The
    // accompanying parameter-name vector identifies the columns and must match
    // the next run's sensitivity_params for the seed to be consumed. Empty when
    // no seed is pending. Cleared by set_concentration()/set_state_from() and by
    // any non-sensitivity run (which advances state without tracking dx/dθ,
    // invalidating the seed); reset() re-seeds it from the IC baseline's own
    // derivative when there is one and clears it otherwise, and
    // save_concentrations() hands it to the new baseline (GH #81).
    const std::vector<double> &pending_sens_seed() const;
    const std::vector<std::string> &pending_sens_seed_param_names() const;
    void set_pending_sens_seed(std::vector<double> seed, std::vector<std::string> param_names);
    void clear_pending_sens_seed();

    // True iff the IC baseline (``species[].initial_conc``, as redefined by
    // save_concentrations()) carries its own dx/dθ — i.e. reset() returns to a
    // θ-dependent initial condition and restores that derivative with it. False
    // for the literal .net/SBML ICs (GH #81).
    bool has_baseline_sens_seed() const;

    // True once save_concentrations() has redefined the IC baseline to a
    // captured state, so `species[].initial_conc` is no longer the initial
    // condition the .net / SBML input declares. set_param() stops re-resolving
    // parameter-named ICs from that point on (issue #79) rather than overwrite
    // a pre-equilibrated baseline. Latching, per model instance; carried by
    // clone(). Exposed for introspection and for the clone contract test.
    bool ic_baseline_saved() const;

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

    /// Re-resolve every species IC that a parameter names, from the current
    /// parameter values (issue #79). Called at the end of set_param(), after
    /// the derived-parameter re-evaluation. See model.cpp for the rules.
    void refresh_param_ref_ics();

    /// Re-derive the storage convention (`Species::volume_factor`, and an
    /// amount-declared `initial_conc = amount/V`) from the current compartment
    /// sizes (issue #170). Called from set_param() when the written parameter is
    /// a compartment size, after the derived-parameter re-evaluation. See
    /// model.cpp for why only the species-side halves live here.
    void refresh_compartment_volume_state();

    /// Does this bound address carry live state — a species concentration, an
    /// observable total, or a rateOf accessor (issue #52)? The single
    /// definition of "moves with the trajectory"; the event-sensitivity guard
    /// and the state-dependent-trigger classification both read it, so the two
    /// cannot disagree about what makes a trigger state-dependent.
    bool is_state_address(const double *addr) const;
};

} // namespace bngsim
