// bngsim/include/bngsim/model_builder.hpp — Programmatic model construction
//
// General-purpose builder for constructing NetworkModel instances from
// external parsers (Antimony, SBML, .net, or any future format) without
// going through .net file serialization. This makes ALL input formats
// first-class inputs to BNGsim with identical optimization pipelines.
//
// The build() method runs the full setup pipeline: ExprTk evaluator,
// stoichiometry, Jacobian sparsity, analytical Jacobian, graph coloring,
// SSA precompute. Every model goes through the same code path.
//
// Usage:
//   ModelBuilder builder;
//   builder.add_parameter("k", 0.3);
//   builder.add_species("S", 10.0);
//   builder.add_function("_rhs_S", "-k*S");
//   builder.add_reaction({0}, {0}, RateLawType::Functional, "_rhs_S");
//   auto model = builder.build();
//
// Used by NetFileLoader and other front ends so every model input goes through
// the same setup pipeline. Also carries .net-specific metadata such as tfun
// specs, net_file_dir, and species-parameter references.

#pragma once

#include "bngsim/model.hpp"
#include "bngsim/types.hpp"

#include <string>
#include <vector>

namespace bngsim {

class ModelBuilder {
  public:
    ModelBuilder();
    ~ModelBuilder();

    // Non-copyable, movable
    ModelBuilder(const ModelBuilder &) = delete;
    ModelBuilder &operator=(const ModelBuilder &) = delete;
    ModelBuilder(ModelBuilder &&) noexcept;
    ModelBuilder &operator=(ModelBuilder &&) noexcept;

    // ─── Add model elements ──────────────────────────────────────────────

    /// Add a parameter. Returns the 0-based parameter index.
    /// Parameters must be added before species/reactions/functions that
    /// reference them.
    int add_parameter(const std::string &name, double value, const std::string &expression = "",
                      bool is_expression = false);

    /// Add a species. Returns the 0-based species index.
    /// volume_factor: storage→amount conversion factor used by the SSA fire
    /// step (`Δstorage = ±1/volume_factor` per fire). SBML loaders pass the
    /// species's compartment volume V_c; default 1.0 leaves `.net` and V=1
    /// SBML behavior unchanged.
    /// amount_valued: when true, the species participates in every reaction's
    /// species factor by its *amount* (stored × volume_factor) rather than the
    /// stored concentration (GH #75 — hasOnlySubstanceUnits=true semantics).
    /// Default false leaves `.net`, V=1 SBML, and hOSU=false behavior
    /// byte-identical.
    /// reported: when false the species is full integrator state but is omitted
    /// from the trajectory output columns (GH #71). SBML loaders pass false for
    /// an event-mutated parameter/compartment promoted to a species. Default
    /// true is byte-identical (every reported species appears in the output).
    int add_species(const std::string &name, double init_conc, bool fixed = false,
                    double volume_factor = 1.0, bool amount_valued = false, bool reported = true);

    /// Add an observable group. Returns the 0-based observable index.
    /// entries: vector of (0-based species index, factor) pairs.
    int add_observable(const std::string &name, const std::vector<std::pair<int, double>> &entries);

    /// Add a function (named expression). Returns the 0-based function index.
    /// The expression can reference parameter names, observable names, and
    /// built-in functions (time(), sin(), etc.).
    int add_function(const std::string &name, const std::string &expression);

    /// Add a reaction. Returns the 0-based reaction index.
    /// reactants/products: vectors of 0-based species indices. Empty = null
    /// (for creation/degradation reactions).
    /// rate_law: parameter name (Elementary) or function name (Functional).
    /// For MichaelisMenten: rate_law = "MM", and mm_params must be set.
    /// apply_species_factor: BNGL-convention default true — engine multiplies
    /// the rate by the reactant species factor. SBML loaders that emit a
    /// kinetic law whose expression already includes the species factor
    /// should pass false to avoid double-counting.
    /// ssa_volume_factor: SSA-only multiplier converting an ODE-units rate
    /// (storage/time) to amount/time propensity. SBML loaders pass V_c (the
    /// reaction's compartment volume); default 1.0 leaves `.net` and V=1
    /// SBML behavior unchanged. ODE path is unaffected by this field.
    /// per_species_volume_scaling: cross-compartment ODE accumulator. When
    /// true, compute_derivs divides the rate by each affected species's
    /// volume_factor rather than accumulating a single scalar rate. SBML
    /// loaders pass true for unified emission of mixed-V_s reactions.
    /// Default false preserves `.net` and uniform-V_s SBML behavior.
    /// is_rate_rule_ode: GH #81 — mark a Functional reaction `[] → [X]` that
    /// was compiled from an SBML rate rule (`dX/dt = f`). The ODE path is
    /// unaffected; the SSA/PSA loop excludes it from stochastic selection and
    /// integrates X deterministically (forward Euler) instead. Default false.
    /// ssa_live_volume_idx0 / ssa_live_volume_exp: GH #81 — live
    /// compartment-volume correction for SSA propensities in a variable-volume
    /// compartment. idx0 is the 0-based species index whose stored value is the
    /// live volume V_live(t) (the promoted compartment species); exp is
    /// `n_h - 1` (hOSU=false species-factor count minus one). The SSA propensity
    /// is multiplied by `(ssa_volume_factor/V_live)^exp`. ODE path is
    /// unaffected. Default idx0=-1 ⇒ no correction (`.net` / static-V identical).
    int add_reaction(const std::vector<int> &reactants, const std::vector<int> &products,
                     RateLawType type, const std::string &rate_law, double stat_factor = 1.0,
                     bool apply_species_factor = true, double ssa_volume_factor = 1.0,
                     bool per_species_volume_scaling = false, bool is_rate_rule_ode = false,
                     int ssa_live_volume_idx0 = -1, double ssa_live_volume_exp = 0.0,
                     bool ode_only = false);

    /// GH #81: set the SSA live-volume correction on an already-added reaction.
    /// The SBML loader promotes an event-resized compartment to a species only
    /// in its event-emission pass — AFTER the reaction-emission pass — so the
    /// live-volume species index is not known when the reaction is added. The
    /// loader records the (reaction, compartment) pair during emission and calls
    /// this once the compartment species index is resolved. `rxn_idx0` is the
    /// 0-based index returned by add_reaction. No-op if out of range.
    void set_reaction_live_volume(int rxn_idx0, int ssa_live_volume_idx0,
                                  double ssa_live_volume_exp);

    /// GH #144 (case 4): set the CROSS-COMPARTMENT ODE live-volume divide on a
    /// species. For an hOSU=false species in a variable-volume compartment, the
    /// per_species_volume_scaling accumulation in compute_derivs must divide by
    /// the LIVE compartment volume (= conc[live_idx0], the promoted compartment
    /// species) rather than the static volume_factor (= V_static). Like
    /// set_reaction_live_volume, the loader records this during emission and
    /// resolves live_idx0 only after the compartment is promoted to a species.
    /// No-op if species_idx0 is out of range.
    void set_species_ode_live_volume(int species_idx0, int live_idx0);

    /// GH #231 (rateOf sub-cluster 3): mark a hasOnlySubstanceUnits=true species
    /// in a CONSTANT-volume compartment so its rateOf csymbol reports the
    /// amount-rate (volume_factor·d(conc)/dt) instead of the stored d(conc)/dt.
    /// The same flag also serves a hOSU species in a VARIABLE-volume compartment:
    /// the integrator stores amount/V_static, so ×volume_factor recovers the
    /// amount-rate there too (suite 01463). See Species::report_rateof_amount.
    /// No-op if species_idx0 is out of range.
    void set_species_rateof_amount(int species_idx0);

    /// GH #144 (case 4): append a CROSS-COMPARTMENT SSA live-volume term to an
    /// already-added reaction. compute_rxn_rate multiplies the SSA propensity by
    /// `(v_static / V_live)^exp`, with `V_live = conc[live_idx0]` (the promoted
    /// compartment species). One term per variable-volume compartment the reaction
    /// touches (exp = the hOSU=false reactant law-factor count in that compartment).
    /// No-op if rxn_idx0 is out of range.
    void add_reaction_live_volume_term(int rxn_idx0, int live_idx0, double v_static, double exp);

    /// Enable/disable conservation-law detection in build() (GH #102).
    /// The detector runs dense O(n_species^3) Gaussian elimination on the
    /// stoichiometry matrix and is consumed only by the steady-state solver —
    /// the ODE/SSA integration paths never read it. Disable it to keep setup
    /// O(reactions) for very large ODE-only networks (~100K species), where the
    /// dense factorization is intractable. Default true preserves the behavior
    /// of every existing front end. When disabled, the built model carries an
    /// empty ConservationLaws (steady_state will then find no laws).
    void set_compute_conservation_laws(bool enabled);

    /// Enable SBML rateOf csymbol support (GH #106). Call when the model
    /// references rateOf(species) (event triggers, rate-rule / assignment-rule
    /// functions). build() then sizes the live derivative buffer and registers
    /// the rate_of__<species> accessor variables those expressions read. No-op
    /// to omit for models without rateOf — the built model is byte-identical.
    void enable_rateof();

    // ─── .net-specific support ───────────────────────────────────────────

    /// Set the directory of the source .net file (for resolving relative
    /// paths to .tfun files and other external references).
    void set_net_file_dir(const std::string &dir);

    /// Record that a species' initial concentration should be resolved from
    /// a parameter value. Called when .net files use parameter names instead
    /// of numeric ICs (e.g., "A0" instead of "100.0").
    /// species_idx0: 0-based species index.
    /// param_name: name of the parameter to resolve from.
    void add_species_param_ref(int species_idx0, const std::string &param_name);

    /// Register a table function specification to be loaded during build().
    /// func_name: the table function's runtime identifier (forms the ExprTk
    ///   callable name "tfun_<func_name>"). For whole-body tfun this matches a
    ///   BNG function name; for embedded/wrapper-form tfun the loader passes a
    ///   synthetic name like "<bng_func>__tfun<k>".
    /// filepath: relative or absolute path to the .tfun file.
    /// index_name: "time" (default) or a parameter/observable name.
    /// header_name: optional .tfun column-2 header validation name. Defaults to
    ///   func_name. Set to the original BNG function name for embedded-form
    ///   tfun so the file's header (which still labels its column by the BNG
    ///   name) is accepted.
    void add_table_function_spec(const std::string &func_name, const std::string &filepath,
                                 const std::string &index_name = "time",
                                 const std::string &method = "linear",
                                 const std::string &header_name = "");

    /// Register an inline table function specification (data embedded in .net).
    /// Aligns with BioNetGen tfun-dev syntax: tfun([x0,x1,...],[y0,y1,...],index)
    /// func_name: the function name (must match a function added via add_function())
    /// xs, ys: data arrays (must be same length, xs monotonically increasing)
    /// index_name: "time" (default) or a parameter/observable name
    /// method: "linear" (default) or "step"
    void add_inline_table_function_spec(const std::string &func_name, const std::vector<double> &xs,
                                        const std::vector<double> &ys,
                                        const std::string &index_name = "time",
                                        const std::string &method = "linear");

    // ─── Events ──────────────────────────────────────────────────────────

    /// Add a discrete event.
    /// @param id           Unique event identifier
    /// @param trigger_expr ExprTk boolean expression (e.g., "time() >= 10")
    /// @param assignments  Vector of (0-based species index, value expression string)
    /// @param delay        Constant delay before applying assignments (used when delay_expr is
    /// empty)
    /// @param priority     Constant priority (used when priority_expr is empty; higher = first)
    /// @param persistent   SBML L3: trigger must stay true through delay
    /// @param initial_value Trigger state at t=0 (true = treat as already true; false = can fire at
    /// t=0)
    /// @param use_values_from_trigger_time SBML L3: if true, RHS values are
    ///        evaluated at trigger time and held until apply time
    /// @param delay_expr   Optional ExprTk expression evaluated at trigger time
    ///        (overrides the constant `delay` when non-empty)
    /// @param priority_expr Optional ExprTk expression evaluated at trigger time
    ///        (overrides the constant `priority` when non-empty)
    /// @param assignment_ode_only GH #81: parallel to `assignments`; entries
    ///        true are applied under ODE only and skipped under SSA (the
    ///        compartment-resize concentration rescale, which must not perturb
    ///        molecule counts). Empty (default) ⇒ every assignment applies in
    ///        both modes.
    void add_event(const std::string &id, const std::string &trigger_expr,
                   const std::vector<std::pair<int, std::string>> &assignments, double delay = 0.0,
                   int priority = 0, bool persistent = true, bool initial_value = true,
                   bool use_values_from_trigger_time = true, const std::string &delay_expr = "",
                   const std::string &priority_expr = "",
                   const std::vector<bool> &assignment_ode_only = {});

    // ─── Discontinuity triggers (GH #72) ───────────────────────────────────

    /// Register a time-dependent inequality condition (e.g. "time()<=0.125")
    /// as a discontinuity trigger. At build time it is compiled into the
    /// evaluator and registered as a CVODE root function so the integrator
    /// stops exactly at the `time` threshold crossing and cannot step over a
    /// narrow forcing pulse encoded by a piecewise assignment rule. Unlike an
    /// event it carries no state assignment — its only effect is to break the
    /// integration step at the discontinuity. Duplicate expression strings are
    /// ignored.
    /// @param condition_expr ExprTk boolean expression in `time()`
    void add_discontinuity_trigger(const std::string &condition_expr);

    // ─── Build ───────────────────────────────────────────────────────────

    /// Finalize and build the NetworkModel. Runs the full setup pipeline:
    /// ExprTk evaluator, stoichiometry, Jacobian sparsity, graph coloring,
    /// SSA precompute. The builder is consumed (moved from).
    ///
    /// Throws std::runtime_error if the model is invalid.
    NetworkModel build();

  private:
    struct BuilderImpl;
    std::unique_ptr<BuilderImpl> bimpl_;
};

} // namespace bngsim
