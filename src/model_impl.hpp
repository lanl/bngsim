// bngsim/src/model_impl.hpp — Definition of NetworkModel::Impl
//
// Shared between model.cpp and net_file_loader.cpp.
// NOT part of the public API — lives in src/, not include/.

#pragma once

#include "bngsim/expression.hpp"
#include "bngsim/model.hpp"
#include "bngsim/table_function.hpp"
#include "bngsim/types.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bngsim {

// ─── Shared immutable model data ─────────────────────────────────────────────
//
// All data that is IMMUTABLE after build() completes. Shared across clones
// via shared_ptr<const> to eliminate deep-copying of large vectors.
// For fceri_gamma (3744 sp, 58K rxns), this avoids copying ~10MB of
// reaction data per clone() call.
//
// Safety contract: after ModelBuilder::build() populates this struct and
// wraps it in shared_ptr<const>, no code path modifies it. All const refs
// returned by NetworkModel accessors point into this struct.
struct SharedModelData {
    std::vector<Reaction> reactions;
    std::vector<StoichEntry> stoichiometry;

    // Name → index lookups (0-based)
    std::unordered_map<std::string, int> param_name_to_idx;
    std::unordered_map<std::string, int> observable_name_to_idx;
    std::unordered_map<std::string, int> function_name_to_idx;
    std::unordered_map<std::string, int> species_name_to_idx;

    // Variable parameters: indices of parameters whose values come from functions.
    // (func_idx, param_idx) — both 0-based into their respective vectors.
    std::vector<std::pair<int, int>> var_param_bindings;

    // Species whose initial concentration is set by a parameter (.net "begin
    // species" entries with a parameter name in the IC column). Stored as
    // (species_idx0, param_idx0) pairs. Used by forward-sensitivity setup to
    // seed s(0) = ∂y(0)/∂p with the IC Jacobian column.
    std::vector<std::pair<int, int>> species_ic_param_refs;

    // Directory from which the .net file was loaded (for resolving relative
    // paths to .tfun files).
    std::string net_file_dir;

    // Jacobian sparsity pattern (CSC format) for sparse ODE solver.
    //
    // The CSC structure (n/nnz/col_ptrs/row_indices/density) is computed by
    // build() and immutable thereafter. The Curtis-Powell-Reid coloring fields
    // inside it are NOT: they are lazily materialized (GH #29) by
    // ensure_jacobian_coloring() on first use, because only the sparse-FD
    // Jacobian callback consumes them and only a minority of models ever take
    // that path. `mutable` covers exactly that one deferred write — the same
    // sanctioned exception to the "immutable after build()" contract above as
    // conservation_laws below, with the same guarantee: the coloring is written
    // exactly once under jac_coloring_once and never changes afterward, so every
    // const ref returned by jacobian_sparsity() stays valid across it. The CSC
    // structure is never touched after build(), so a reader holding `jac_sparsity`
    // for its col_ptrs/row_indices races with no one; readers of the *coloring*
    // must go through ensure_jacobian_coloring().
    mutable JacobianSparsity jac_sparsity;
    mutable std::once_flag jac_coloring_once;

    // Analytical Jacobian pre-computed structure.
    AnalyticalJacobianData analytical_jac;

    // Conservation laws detected from the stoichiometry matrix.
    //
    // Lazily materialized (GH #102): the detector is dense O(ns^3) Gaussian
    // elimination, consumed ONLY by the steady-state solver and the public
    // conservation_laws() accessor — never by ODE/SSA integration. It is NOT
    // computed at build(); ensure_conservation_laws() computes it on first
    // access and caches it here (once, thread-safe, shared across clones), so
    // ODE/SSA-only runs never pay for it. This is the one sanctioned exception
    // to the "immutable after build()" contract above: the write happens exactly
    // once under conservation_laws_once and the materialized value never changes
    // afterward, so every const ref returned by conservation_laws() stays valid.
    // When conservation_laws_enabled is false (set_compute_conservation_laws(
    // false)) it is never computed and stays empty (n_species only), for callers
    // that need the full unreduced system.
    mutable ConservationLaws conservation_laws;
    mutable std::once_flag conservation_laws_once;
    bool conservation_laws_enabled = true;
};

// Lazily compute (once) and return the model's conservation laws. On the first
// call with conservation_laws_enabled it runs the dense stoichiometric
// null-space detector, caches the result in `sd`, and returns it; later calls
// return the cached value; a disabled model returns the empty laws immediately.
// `species` supplies only the species count and fixed flags — both structural
// and identical across clones, so any instance may trigger the shared compute.
// Defined in model_builder.cpp next to detect_conservation_laws().
const ConservationLaws &ensure_conservation_laws(const SharedModelData &sd,
                                                 const std::vector<Species> &species);

// Lazily compute (once) the Curtis-Powell-Reid coloring of the Jacobian
// sparsity pattern and return the pattern with it materialized. The coloring is
// consumed ONLY by the sparse colored-FD Jacobian callback, so build() does not
// compute it; the first caller that needs it pays, later calls and other clones
// reuse it. A pattern with no structural nonzeros is left uncolored (n_colors
// stays 0) — there is nothing to perturb. Defined in model_builder.cpp next to
// compute_coloring().
const JacobianSparsity &ensure_jacobian_coloring(const SharedModelData &sd);

// The value to STORE in `species[i].concentration` / `.initial_conc` when
// parameter `param_value` names species `sp`'s initial condition — i.e. one
// entry of SharedModelData::species_ic_param_refs resolved at the current
// parameter values.
//
// Two callers must agree on this, which is the whole reason it is a function:
// ModelBuilder::build() resolves the refs once at load, and
// NetworkModel::set_param() re-resolves them whenever a parameter moves
// (issue #79). A rule that lived in only one of them is the drift that #79's
// dose scan ran into from the other side.
//
// The conversion is the SBML loader's, inverted. A `.net` IC column and every
// hOSU=false SBML species store a concentration and the parameter names it
// directly, so the factor is 1 and this is the identity. A
// hasOnlySubstanceUnits=true species's symbol denotes an AMOUNT (the loader's
// `amount_valued`), and the engine stores amount/V_static — so a parameter that
// names its IC names an amount and has to be divided by the compartment volume,
// exactly as the loader's initialAmount / initialAssignment branches do. V = 0
// is left undivided, matching those branches.
inline double resolve_ic_from_param(const Species &sp, double param_value) {
    if (sp.amount_valued && sp.volume_factor != 0.0)
        return param_value / sp.volume_factor;
    return param_value;
}

// Full definition of NetworkModel::Impl.
// model.hpp forward-declares it; this header provides the body.
//
// ── CLONE() CONTRACT — read this when adding a new field below ──────────────
//
// `NetworkModel::clone()` is hand-rolled (model.cpp:60). Every per-instance
// field of `Impl` must be addressed there explicitly. There is no
// compile-time enforcement; the canonical exercise is
// `python/tests/test_model_clone.py` — extend it when you add a field.
//
// For each new mutable field, decide:
//
//   1. **Deep copy** vs share-pointer. If the field is mutated after build()
//      it must be deep-copied. If it's immutable post-build, prefer adding
//      it to `SharedModelData` instead.
//
//   2. **Evaluator rebinding**. If the field exposes addresses to the
//      ExprTk evaluator (e.g., `&p.value`, `&obs.total`, `&sp.concentration`),
//      the cloned evaluator must rebind those addresses to the clone's
//      copy of the field via `define_variable`.
//
//   3. **Expression re-compile**. If the field stores `evaluator_id` indices
//      (parameter expressions, function bodies, event trigger / delay /
//      priority / assignment-RHS expressions, table-function callable
//      registrations), each index is local to the *original* evaluator. The
//      clone must read the cached preprocessed string from the original
//      evaluator's `preprocessed_expr(idx)` and re-compile it via
//      `compile_preprocessed` to get a fresh index in the cloned evaluator.
//
//   4. **Order dependency**. Re-compilation order matters: a parameter
//      expression that references a function-derived parameter slot needs
//      the parameter rebound first; a function expression that references
//      a table function needs the table function registered first; an
//      event trigger that references a species needs the species rebound
//      first. The current ordering in `clone()` is documented at its top.
//
// Past silent-correctness bugs caught only after a real symptom:
//   * `events` was never copied at all (Phase 5b). DSMTS event cases
//     fired 0/N reps because cloned models had empty event lists.
//   * Species names were not rebound to the cloned evaluator. Event
//     trigger expressions referencing species failed to compile in the
//     clone with `ExprTk ERR239 — Undefined symbol`.
//
// Both bugs were latent because the .net regression corpus has no events
// and no species-referencing trigger expressions. Convention is the only
// thing keeping this file consistent — the test file enforces it.
struct NetworkModel::Impl {
    // ── Shared immutable data (shared across clones) ─────────────────────
    std::shared_ptr<const SharedModelData> shared;

    // ── Mutable per-instance data ────────────────────────────────────────
    std::vector<Species> species;
    std::vector<Observable> observables;
    std::vector<Parameter> parameters;
    std::vector<Function> functions;

    // Whether model has any functional rate laws or table functions.
    // Mutable because register_table_function_() can set it post-build.
    bool has_functions = false;

    // Cache of the most recent function values, indexed by function
    // *declaration* index (parallel to `functions`). Populated as a side effect
    // of evaluate_functions() so the output-recording path can read function
    // values without a second interpreted ExprTk pass over every function (GH
    // #136 — observable/expression evaluation dominated large-model wall time).
    // Lazily sized on the first evaluate_functions() call; a function whose
    // evaluator_id < 0 keeps its 0.0 slot, matching function_values(). Pure
    // recomputable cache: clone() leaves it default-empty (re-filled on the
    // clone's next evaluate_functions()), so it is exempt from the clone
    // contract below.
    std::vector<double> function_value_cache;

    // System time
    double current_time = 0.0;

    // ── Pre-equilibration / carry-over sensitivity state (GH #210) ───────────
    // ic_state_dirty: true once the species state diverges from the load/reset
    // ICs (advanced by a run() or set manually). pending_sens_seed: the prior
    // phase's final dx/dθ (row-major [species*np + param]) plus its column
    // parameter names, threaded into the next phase's yS(0) seed when
    // carry_sensitivities is set. clone() copies these (a clone is a faithful
    // snapshot of live state, like `species`/`current_time`); reset() and the
    // manual-state mutators clear them.
    bool ic_state_dirty = false;
    std::vector<double> pending_sens_seed;
    std::vector<std::string> pending_sens_seed_param_names;

    // ── Has save_concentrations() redefined the IC baseline? (issue #79) ──────
    // set_param() re-resolves every species IC that names the written parameter
    // (or a derived parameter that reads it), because `A() Stot` says A's
    // initial condition IS Stot and a dose scan over Stot has to move it. That
    // claim only holds while `initial_conc` is still the DECLARED initial
    // condition. save_concentrations() redefines it to the current — typically
    // pre-equilibrated — state, at which point the .net/SBML IC no longer
    // describes the baseline and re-resolving it would silently discard the
    // equilibration. Latching here retires the rebuild for the model's
    // remaining life; there is no un-save. clone() copies it with the baseline
    // it describes.
    bool ic_baseline_saved = false;

    // ── The IC baseline's own θ-derivative (GH #81) ───────────────────────────
    // save_concentrations() redefines initial_conc to the *current* state, which
    // in a pre-equilibration protocol is x_ss(θ) — a baseline whose ∂x(0)/∂θ is
    // the carried dx/dθ, not zero. Stashing that derivative alongside the new
    // baseline lets reset() restore the baseline WITH its derivative, instead of
    // silently reverting a θ-dependent initial condition to fresh-start seeding.
    // Empty ⇔ the baseline is θ-independent (literal .net ICs, or a
    // save_concentrations() taken with no carried derivative), which is the
    // pre-#81 behavior. clone() copies these with the baseline they describe.
    std::vector<double> baseline_sens_seed;
    std::vector<std::string> baseline_sens_seed_param_names;

    // RHS instrumentation counters (T1 — gate ODE RHS observable/function eval).
    // Diagnostics only: rhs_eval_count counts every compute_derivs_core() call;
    // rhs_obs_func_eval_count counts the subset of those calls that actually ran
    // update_observables() + evaluate_functions() (i.e. has_functions). For a
    // pure mass-action model the second stays 0 while the first grows — the
    // observable/function passes are dead work and are skipped. Pure
    // recomputable per-instance state: clone() builds a fresh Impl so both reset
    // to 0 in the clone (same exemption as function_value_cache), and nothing
    // numeric depends on them — exempt from the clone contract below.
    std::uint64_t rhs_eval_count = 0;
    std::uint64_t rhs_obs_func_eval_count = 0;

    // ── SBML rateOf csymbol support (GH #106) ────────────────────────────
    // True iff the model references rateOf(species) anywhere (event triggers,
    // rate-rule / assignment-rule functions). When false, compute_derivs() is
    // byte-identical to pre-#106 and these buffers stay empty.
    bool uses_rateof = false;
    // Live instantaneous species derivatives dx/dt. The `rate_of__<species>`
    // accessor variables are ExprTk-bound to &current_derivs[species_idx0], so
    // this vector is sized once at build() (to n_species, zero-init) and never
    // resized — the bound addresses must stay stable. Refreshed by the probe in
    // compute_derivs() / refresh_rateof_derivs() before any rateOf read.
    std::vector<double> current_derivs;
    // Scratch output for the probe pass so it never writes the accessor source
    // (current_derivs) while compute_derivs_core reads it. Published into
    // current_derivs atomically once the probe completes. Sized with
    // current_derivs.
    std::vector<double> rateof_scratch;

    // Expression evaluator (ExprTk)
    std::unique_ptr<ExprTkEvaluator> evaluator;

    // Table functions
    // Stored as unique_ptrs so the TableFunction objects have stable addresses
    // for the ExprTk zero-arg function adapter pointers.
    std::vector<std::unique_ptr<TableFunction>> table_functions;

    // Events
    // Discrete state assignments triggered by boolean conditions.
    std::vector<Event> events;

    // ── Event-trigger residuals for forward sensitivity (issue #144) ─────
    // Per event, parallel to `events`: the ExprTk expression id of the
    // trigger's residual `lhs − rhs`, the species that can move it, and the
    // reason there is none. Sentinels are deliberately distinct so "not looked
    // at yet" can never be read as "looked at, and there is none":
    //     -2 → not resolved yet (the lazy state; also the state a clone starts in)
    //     -1 → resolved, and the trigger is not a single relational comparison
    //    >=0 → the compiled residual
    // A pure recomputable cache: every entry is a function of the trigger's own
    // preprocessed text, so clone() re-derives rather than copying an
    // expression id that means nothing in the clone's evaluator. Sized lazily
    // by NetworkModel::event_trigger_residual_expr().
    std::vector<int> event_trigger_residual_idx;
    std::vector<std::vector<int>> event_trigger_residual_species;
    std::vector<std::string> event_trigger_residual_reason;

    // Memo for NetworkModel::expression_support(), keyed by expression id:
    // (species support, parameter support). The event-jump differences ask for
    // the same handful of expressions at every fire, and the walk builds
    // address→index maps over the whole model each time — on a spiking model
    // that is the difference between O(1) and O(n_species) per fire.
    //
    // Safe to memo because the answer is a property of the expression graph,
    // which build() fixes. The one post-build mutation that can touch it,
    // set_param()'s detach of an expression-backed parameter, can only ever
    // make a support SMALLER (the parameter stops expanding to the primaries
    // behind it) — so a stale entry over-differences, costing time and writing
    // the zero it would have written anyway, and never under-differences.
    // Recomputable, so clone() leaves it empty (same exemption as
    // function_value_cache).
    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> expression_support_cache;

    // ── Rate-law state-switch residuals (issue #150) ─────────────────────
    // Memo for NetworkModel::state_switch(), keyed by the condition's SOURCE
    // text. `first` is the resolved switch (residual_expr_idx -1 when the
    // condition is not one), `second` the reason there is none. Holds an
    // evaluator expression id, so clone() leaves it empty and re-derives from
    // the same text — the identical contract to event_trigger_residual_idx.
    std::unordered_map<std::string, std::pair<NetworkModel::StateSwitch, std::string>>
        state_switch_cache;

    // Discontinuity triggers (GH #72)
    // ExprTk expression-table indices for time-dependent inequality
    // conditions found in piecewise expressions that feed the ODE RHS
    // (e.g. a chemo-dosing assignment rule that is nonzero only during
    // narrow `[t0, t0+w]` windows). They carry NO state assignment — their
    // sole purpose is to be registered as CVODE root functions so the
    // adaptive integrator stops exactly at each `time` threshold crossing
    // and cannot step over a narrow pulse. Empty for the overwhelming
    // majority of models (no time-dependent piecewise), in which case the
    // integrator behaves byte-identically to pre-#72.
    std::vector<int> discontinuity_trigger_expr_idx;

    // User-visible diagnostics produced while loading a model.
    std::vector<std::string> load_warnings;

    // Functional analytical Jacobian (GH #76). Per-instance because its
    // ExprTk derivative-expression ids are local to `evaluator`. Populated
    // after build() by set_functional_jacobian() (Python-driven). clone()
    // re-compiles each derivative expression into the cloned evaluator via the
    // preprocessed-string cache (same contract as functions/events) — see
    // model.cpp clone(). Empty (populated=false) ⇒ the model uses the FD
    // Jacobian for its Functional reactions, exactly as pre-#76.
    FunctionalJacobianData functional_jac;

    Impl()
        : shared(std::make_shared<SharedModelData>()),
          evaluator(std::make_unique<ExprTkEvaluator>()) {}
};

// ─── rateOf accessor registration (GH #106) ──────────────────────────────────
//
// Register the per-species `rate_of__<species>` ExprTk variables, each bound to
// &current_derivs[i], so a compiled expression that references rateOf(species)
// reads the live dx/dt the probe publishes into current_derivs. The naming
// convention (prefix `rate_of__` + the species name, which is already the
// loader's _safe_name form) lives here so ModelBuilder::build() and
// NetworkModel::clone() stay in lock-step with the loader's token emission.
//
// Preconditions: current_derivs is already sized to species.size() and will not
// be resized afterwards (the bound addresses must stay stable), and this runs
// BEFORE any expression containing a rate_of__ token is compiled.
inline void register_rateof_accessors(ExprTkEvaluator &eval, std::vector<Species> &species,
                                      std::vector<double> &current_derivs) {
    for (std::size_t i = 0; i < species.size(); ++i) {
        eval.define_variable("rate_of__" + species[i].name, &current_derivs[i]);
    }
}

} // namespace bngsim
