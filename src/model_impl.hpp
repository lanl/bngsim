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
    JacobianSparsity jac_sparsity;

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
