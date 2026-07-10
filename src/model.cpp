// bngsim/src/model.cpp — Instance-based network model implementation
//
// Core RHS evaluation: update_observables → evaluate_functions → compute rates.
// Ported from BNG's derivs_network() (network.cpp) to instance-based design.

#include "bngsim/functional_jac_scatter.hpp"
#include "bngsim/mm_jacobian.hpp"
#include "bngsim/net_file_loader.hpp"
#include "bngsim/table_function.hpp"
#include "model_impl.hpp" // defines NetworkModel::Impl

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace bngsim {

namespace {

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static InterpolationMethod parse_interpolation_method(const std::string &method,
                                                      const std::string &tf_name) {
    std::string normalized = to_lower_ascii(method);
    if (normalized.empty() || normalized == "linear") {
        return InterpolationMethod::Linear;
    }
    if (normalized == "step") {
        return InterpolationMethod::Step;
    }
    throw std::runtime_error("TableFunction '" + tf_name + "': unsupported interpolation method '" +
                             method + "' (expected 'linear' or 'step')");
}

} // namespace

// ─── Constructors / move ─────────────────────────────────────────────────────

NetworkModel::NetworkModel() : impl_(std::make_unique<Impl>()) {}
NetworkModel::~NetworkModel() = default;

NetworkModel::NetworkModel(NetworkModel &&) noexcept = default;
NetworkModel &NetworkModel::operator=(NetworkModel &&) noexcept = default;

// ─── Factory ─────────────────────────────────────────────────────────────────

NetworkModel NetworkModel::from_net(const std::string &path) {
    NetFileLoader loader;
    return loader.load(path);
}

std::vector<std::string> NetworkModel::load_warnings() const { return impl_->load_warnings; }

void NetworkModel::set_load_warnings_(std::vector<std::string> warnings) {
    impl_->load_warnings = std::move(warnings);
}

// ─── Clone ───────────────────────────────────────────────────────────────────

NetworkModel NetworkModel::clone() const {
    // Deep copy with shared ExprTk parser and pre-processed expression cache.
    //
    // clone_empty() shares the heavyweight
    // ExprTk parser (~100KB template object) with the original model.
    // compile_preprocessed() skips the logical-operator and underscore
    // remapping that compile() does — the cached strings are already
    // preprocessed. This eliminates parser construction + string
    // preprocessing overhead on every clone.
    //
    // ── ORDERING CONTRACT — preserve when modifying ─────────────────────────
    //
    // The per-instance field handling below is order-dependent. If you add
    // a new step or move an existing one, verify against this contract:
    //
    //   1. `shared` (shared_ptr copy)
    //   2. Deep-copy POD-ish per-instance vectors (species, observables,
    //      parameters, functions) and scalars (has_functions, current_time)
    //   3. Create empty evaluator + bind time pointer to the COPY's
    //      current_time
    //   4. Define parameter / observable / species names as ExprTk
    //      variables in the cloned evaluator
    //   4b. Re-bind rateOf accessors (GH #106): size the copy's current_derivs
    //      and bind each rate_of__<species> to it — must precede step 5+ since
    //      any re-compiled expression may reference a rate_of__ token
    //   5. Re-compile expression-valued parameters (needs step 4)
    //   6. Re-create table functions (registers them as ExprTk callables;
    //      needed before function expressions are re-compiled)
    //   7. Re-compile function expressions (may reference table functions
    //      from step 6 and parameter slots from step 4)
    //   8. Deep-copy events and re-compile event expressions (may
    //      reference everything from steps 4, 6, 7)
    //
    // `python/tests/test_model_clone.py` is the canonical exercise of this
    // contract. See model_impl.hpp Impl declaration for the four-step
    // checklist when adding a new per-instance field.

    NetworkModel copy;

    // Share immutable data (single pointer copy — zero-cost for 58K reactions)
    copy.impl_->shared = impl_->shared;

    // Deep-copy mutable per-instance data
    copy.impl_->species = impl_->species;
    copy.impl_->observables = impl_->observables;
    copy.impl_->parameters = impl_->parameters;
    copy.impl_->functions = impl_->functions;
    copy.impl_->has_functions = impl_->has_functions;
    copy.impl_->current_time = impl_->current_time;

    // Carry-over sensitivity state (GH #210): a clone is a faithful snapshot of
    // the live species/current_time state copied just above, so it must carry
    // the same "is this state advanced past the ICs?" flag and pending dx/dθ
    // seed. (Batch/parallel workers clone() then reset(), which clears both.)
    copy.impl_->ic_state_dirty = impl_->ic_state_dirty;
    copy.impl_->pending_sens_seed = impl_->pending_sens_seed;
    copy.impl_->pending_sens_seed_param_names = impl_->pending_sens_seed_param_names;

    // Create evaluator that shares the parser with the original
    copy.impl_->evaluator = impl_->evaluator->clone_empty();

    // Bind time()/t() to the COPY's current_time
    copy.impl_->evaluator->set_time_ptr(&copy.impl_->current_time);

    // Re-bind all variables to the copy's data
    for (auto &p : copy.impl_->parameters) {
        copy.impl_->evaluator->define_variable(p.name, &p.value);
    }
    for (auto &obs : copy.impl_->observables) {
        copy.impl_->evaluator->define_variable(obs.name, &obs.total);
    }
    // Species names are registered for event trigger / assignment-RHS
    // expressions (matches the post-build registration in
    // model_builder.cpp). define_variable throws on duplicate, so guard
    // against species names that collide with observables.
    for (auto &sp : copy.impl_->species) {
        try {
            copy.impl_->evaluator->define_variable(sp.name, &sp.concentration);
        } catch (...) {
            // Already registered as a parameter or observable — skip.
        }
    }

    // Re-bind the rateOf accessors (GH #106) BEFORE any expression that may
    // reference a rate_of__<species> token is re-compiled. The copy gets its own
    // current_derivs (sized to n_species, zeroed) and rateof_scratch, and each
    // rate_of__<species> variable is bound to the COPY's buffer — mirrors the
    // time-pointer rebinding above and the build()-time registration.
    copy.impl_->uses_rateof = impl_->uses_rateof;
    if (copy.impl_->uses_rateof) {
        copy.impl_->current_derivs.assign(copy.impl_->species.size(), 0.0);
        copy.impl_->rateof_scratch.assign(copy.impl_->species.size(), 0.0);
        register_rateof_accessors(*copy.impl_->evaluator, copy.impl_->species,
                                  copy.impl_->current_derivs);
    }

    // Re-compile expression-valued parameters using cached preprocessed strings
    for (auto &p : copy.impl_->parameters) {
        if (p.is_expression && !p.expression.empty() && p.evaluator_id >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(p.evaluator_id);
            p.evaluator_id = copy.impl_->evaluator->compile_preprocessed(cached);
            p.value = copy.impl_->evaluator->evaluate(p.evaluator_id);
        }
    }

    // Re-create table functions in the copy (deep copy with fresh bindings)
    for (const auto &tf_ptr : impl_->table_functions) {
        auto new_tf = std::make_unique<TableFunction>(tf_ptr->name(), tf_ptr->xs(), tf_ptr->ys(),
                                                      tf_ptr->index_name(), tf_ptr->method());
        copy.register_table_function_(*new_tf);
        copy.impl_->table_functions.push_back(std::move(new_tf));
    }

    // Re-compile functions using cached preprocessed strings
    // (AFTER table functions are registered, since function expressions
    // may reference table function names)
    for (auto &func : copy.impl_->functions) {
        if (func.evaluator_id >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(func.evaluator_id);
            func.evaluator_id = copy.impl_->evaluator->compile_preprocessed(cached);
        }
    }

    // Re-compile functional analytical-Jacobian derivative expressions (GH #76)
    // into the cloned evaluator. The structure (species index, affected CSC
    // entries, coefficients, populated flag) is instance-independent and copied
    // verbatim; only the ExprTk ids are local to each evaluator. Same
    // preprocessed-string contract as functions/events above.
    copy.impl_->functional_jac = impl_->functional_jac;
    for (auto &st : copy.impl_->functional_jac.species_terms) {
        if (st.deriv_eval_id >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(st.deriv_eval_id);
            st.deriv_eval_id = copy.impl_->evaluator->compile_preprocessed(cached);
        }
    }
    // Per-observable terms (GH #76 task 2): same recompile contract for the
    // ∂func/∂obs_k ids; the columns/reactants/coeffs are instance-independent.
    for (auto &ot : copy.impl_->functional_jac.observable_terms) {
        for (auto &id : ot.dfunc_dobs_eval_ids) {
            if (id >= 0) {
                const auto &cached = impl_->evaluator->preprocessed_expr(id);
                id = copy.impl_->evaluator->compile_preprocessed(cached);
            }
        }
    }

    // Deep-copy events and re-compile their expressions in the new evaluator.
    // Each event carries indices into the evaluator's expression table for
    // its trigger, optional delay, optional priority, and per-assignment
    // value expressions. These indices are evaluator-specific; without
    // re-compilation in the cloned evaluator, the cloned model's events
    // would be silent no-ops (DSMTS event cases 00028/29/32/33 hit this
    // pre-Phase-5b but the SSA simulator did not consult events at all,
    // so the bug was latent).
    copy.impl_->events = impl_->events;
    for (auto &ev : copy.impl_->events) {
        if (ev.trigger_expr_idx >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(ev.trigger_expr_idx);
            ev.trigger_expr_idx = copy.impl_->evaluator->compile_preprocessed(cached);
        }
        if (ev.delay_expr_idx >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(ev.delay_expr_idx);
            ev.delay_expr_idx = copy.impl_->evaluator->compile_preprocessed(cached);
        }
        if (ev.priority_expr_idx >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(ev.priority_expr_idx);
            ev.priority_expr_idx = copy.impl_->evaluator->compile_preprocessed(cached);
        }
        for (auto &assign : ev.assignments) {
            if (assign.second >= 0) {
                const auto &cached = impl_->evaluator->preprocessed_expr(assign.second);
                assign.second = copy.impl_->evaluator->compile_preprocessed(cached);
            }
        }
    }
    // Re-compile discontinuity triggers (GH #72) into the cloned evaluator,
    // same preprocessed-string contract as events above. Without this a clone
    // (used by every parallel worker) would carry stale expression-table
    // indices and silently lose its pulse-edge stops.
    copy.impl_->discontinuity_trigger_expr_idx = impl_->discontinuity_trigger_expr_idx;
    for (auto &idx : copy.impl_->discontinuity_trigger_expr_idx) {
        if (idx >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(idx);
            idx = copy.impl_->evaluator->compile_preprocessed(cached);
        }
    }
    copy.impl_->load_warnings = impl_->load_warnings;

    return copy;
}

// ─── Parameter access ────────────────────────────────────────────────────────

void NetworkModel::set_param(const std::string &name, double value) {
    auto it = impl_->shared->param_name_to_idx.find(name);
    if (it == impl_->shared->param_name_to_idx.end()) {
        throw std::runtime_error("Parameter not found: " + name);
    }
    auto &param = impl_->parameters[it->second];
    param.value = value;

    // If this parameter was expression-backed (e.g., "d = d__FREE"),
    // detach it so the explicit value isn't overwritten by re-evaluation.
    // This matches BNG's setParameter() semantics: the literal value
    // overrides the expression for the remainder of the action sequence.
    if (param.is_expression) {
        param.is_expression = false;
        param.evaluator_id = -1;
        // Keep param.expression for debugging/introspection
    }

    // Re-evaluate remaining expression-valued parameters (e.g., "a = a__FREE").
    // This ensures that derived parameters pick up the new value.
    for (auto &p : impl_->parameters) {
        if (p.is_expression && p.evaluator_id >= 0) {
            p.value = impl_->evaluator->evaluate(p.evaluator_id);
        }
    }
}

double NetworkModel::get_param(const std::string &name) const {
    auto it = impl_->shared->param_name_to_idx.find(name);
    if (it == impl_->shared->param_name_to_idx.end()) {
        throw std::runtime_error("Parameter not found: " + name);
    }
    return impl_->parameters[it->second].value;
}

std::vector<std::string> NetworkModel::param_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->parameters.size());
    for (const auto &p : impl_->parameters) {
        names.push_back(p.name);
    }
    return names;
}

// ─── State management ────────────────────────────────────────────────────────

void NetworkModel::reset() {
    for (auto &s : impl_->species) {
        s.concentration = s.initial_conc;
    }
    impl_->current_time = 0.0;
    // Fresh ICs again: no carry-over, no pending sensitivity seed (GH #210).
    impl_->ic_state_dirty = false;
    clear_pending_sens_seed();
}

void NetworkModel::save_concentrations() {
    for (auto &s : impl_->species) {
        s.initial_conc = s.concentration;
    }
    // The current state is now the baseline ICs, so a subsequent reset()
    // returns here — treat it as a fresh start and drop any carried-over
    // sensitivity seed, which referenced the old IC baseline (GH #210).
    impl_->ic_state_dirty = false;
    clear_pending_sens_seed();
}

void NetworkModel::set_concentration(const std::string &name, double value) {
    auto it = impl_->shared->species_name_to_idx.find(name);
    if (it == impl_->shared->species_name_to_idx.end()) {
        throw std::runtime_error("Species not found: " + name);
    }
    impl_->species[it->second].concentration = value;
    // GH #210 — setting a species to a literal value is a fresh initial-condition
    // assignment (its θ-derivative is 0), NOT a carried-over dynamics state, so
    // it does NOT mark the state dirty: a reset()+set_concentration() IC setup
    // must still get fresh forward-sensitivity seeding. It does invalidate any
    // pending carry-over seed, since that seed assumed this species' prior value
    // (carry-over perturbations are set_param, not set_concentration — ADR-0052).
    clear_pending_sens_seed();
}

double NetworkModel::get_concentration(const std::string &name) const {
    auto it = impl_->shared->species_name_to_idx.find(name);
    if (it == impl_->shared->species_name_to_idx.end()) {
        throw std::runtime_error("Species not found: " + name);
    }
    return impl_->species[it->second].concentration;
}

void NetworkModel::get_state_into(double *out) const {
    const auto &sp = impl_->species;
    for (std::size_t i = 0; i < sp.size(); ++i)
        out[i] = sp[i].concentration;
}

void NetworkModel::set_state_from(const double *in) {
    auto &sp = impl_->species;
    for (std::size_t i = 0; i < sp.size(); ++i)
        sp[i].concentration = in[i];
    // GH #210 — external state injection (GH #102 hybrid orchestrator) is treated
    // as a fresh initial condition for sensitivity seeding (we have no dx/dθ for
    // an externally-supplied state). It invalidates any pending carry-over seed
    // but does not, by itself, mark the state as carried-over dynamics: callers
    // that need carry-over use the persistent-Simulator integration flow, which
    // sets the dirty flag at the run write-back.
    clear_pending_sens_seed();
}

// ─── Pre-equilibration / carry-over sensitivity state (GH #210) ──────────────

bool NetworkModel::ic_state_dirty() const { return impl_->ic_state_dirty; }

void NetworkModel::set_ic_state_dirty(bool dirty) { impl_->ic_state_dirty = dirty; }

const std::vector<double> &NetworkModel::pending_sens_seed() const {
    return impl_->pending_sens_seed;
}

const std::vector<std::string> &NetworkModel::pending_sens_seed_param_names() const {
    return impl_->pending_sens_seed_param_names;
}

void NetworkModel::set_pending_sens_seed(std::vector<double> seed,
                                         std::vector<std::string> param_names) {
    impl_->pending_sens_seed = std::move(seed);
    impl_->pending_sens_seed_param_names = std::move(param_names);
}

void NetworkModel::clear_pending_sens_seed() {
    impl_->pending_sens_seed.clear();
    impl_->pending_sens_seed_param_names.clear();
}

// ─── Accessors ───────────────────────────────────────────────────────────────

int NetworkModel::n_species() const { return static_cast<int>(impl_->species.size()); }
int NetworkModel::n_reactions() const { return static_cast<int>(impl_->shared->reactions.size()); }
int NetworkModel::n_observables() const { return static_cast<int>(impl_->observables.size()); }
int NetworkModel::n_parameters() const { return static_cast<int>(impl_->parameters.size()); }
int NetworkModel::n_functions() const { return static_cast<int>(impl_->functions.size()); }
int NetworkModel::n_events() const { return static_cast<int>(impl_->events.size()); }
int NetworkModel::n_discontinuity_triggers() const {
    return static_cast<int>(impl_->discontinuity_trigger_expr_idx.size());
}
const std::vector<int> &NetworkModel::discontinuity_triggers() const {
    return impl_->discontinuity_trigger_expr_idx;
}

const std::vector<Species> &NetworkModel::species() const { return impl_->species; }
const std::vector<Reaction> &NetworkModel::reactions() const { return impl_->shared->reactions; }
const std::vector<Observable> &NetworkModel::observables() const { return impl_->observables; }
const std::vector<Parameter> &NetworkModel::parameters() const { return impl_->parameters; }
const std::vector<Function> &NetworkModel::functions() const { return impl_->functions; }
const std::vector<Event> &NetworkModel::events() const { return impl_->events; }

std::optional<std::string> NetworkModel::event_sensitivity_unsupported_reason(
    const std::vector<std::string> &sens_param_names) const {
    const auto &events = impl_->events;
    if (events.empty()) {
        return std::nullopt;
    }
    const auto &species = impl_->species;
    const auto &params = impl_->parameters;
    const ExpressionEvaluator &eval = *impl_->evaluator;

    // Resolve requested sensitivity-parameter names → bound value addresses.
    // An unknown name is reported (the run path throws on unknown sensitivity
    // params; here we surface the same condition as a reason so the Python
    // guard can raise a single, clear error).
    std::unordered_set<const double *> sens_param_addrs;
    sens_param_addrs.reserve(sens_param_names.size());
    for (const std::string &pname : sens_param_names) {
        const Parameter *match = nullptr;
        for (const Parameter &p : params) {
            if (p.name == pname) {
                match = &p;
                break;
            }
        }
        if (match == nullptr) {
            return "sensitivity parameter '" + pname + "' was not found in the model.";
        }
        sens_param_addrs.insert(&match->value);
    }

    // State-variable addresses (species concentrations the trigger may read).
    std::unordered_set<const double *> species_addrs;
    species_addrs.reserve(species.size());
    for (const Species &sp : species) {
        species_addrs.insert(&sp.concentration);
    }

    for (const Event &ev : events) {
        const std::string id = ev.id.empty() ? std::string("<unnamed>") : ev.id;
        if (!ev.persistent) {
            return "event '" + id +
                   "' is non-persistent; forward sensitivity through non-persistent "
                   "triggers is not yet supported (GH #212 Phase 3).";
        }
        if (ev.delay != 0.0 || ev.delay_expr_idx >= 0) {
            return "event '" + id +
                   "' has an execution delay; forward sensitivity through delayed events "
                   "is not yet supported (GH #212 Phase 3).";
        }
        const std::vector<const double *> refs =
            eval.referenced_variable_addresses(ev.trigger_expr_idx);
        for (const double *addr : refs) {
            if (species_addrs.count(addr) != 0) {
                return "event '" + id +
                       "' has a state-dependent trigger (it reads a species concentration); "
                       "only fixed-time triggers are supported for forward sensitivity so far "
                       "(GH #212 Phase 2).";
            }
        }
        for (const double *addr : refs) {
            if (sens_param_addrs.count(addr) != 0) {
                std::string pname;
                for (const Parameter &p : params) {
                    if (&p.value == addr) {
                        pname = p.name;
                        break;
                    }
                }
                return "event '" + id +
                       "' has a trigger whose crossing time depends on the requested "
                       "sensitivity parameter '" +
                       pname +
                       "' (the event-time sensitivity dt*/dp is non-zero); event-time "
                       "sensitivity is not yet supported (GH #212 Phase 2). Drop '" +
                       pname +
                       "' from the requested sensitivity parameters, or treat the trigger "
                       "time as fixed.";
            }
        }
    }
    return std::nullopt;
}
const std::vector<StoichEntry> &NetworkModel::stoichiometry() const {
    return impl_->shared->stoichiometry;
}
const JacobianSparsity &NetworkModel::jacobian_sparsity() const {
    return impl_->shared->jac_sparsity;
}
const AnalyticalJacobianData &NetworkModel::analytical_jacobian() const {
    return impl_->shared->analytical_jac;
}
const FunctionalJacobianData &NetworkModel::functional_jacobian() const {
    return impl_->functional_jac;
}
bool NetworkModel::analytical_jacobian_complete() const {
    const auto &ajd = impl_->shared->analytical_jac;
    if (!ajd.available)
        return false; // empty sparsity, or an unsupported reaction type
    // Functional reactions must have their per-instance derivative terms.
    return ajd.n_functional == 0 || impl_->functional_jac.populated;
}
const ConservationLaws &NetworkModel::conservation_laws() const {
    // Lazily materialized on first access (see model_impl.hpp / GH #102): the
    // dense O(ns^3) detector is skipped entirely for ODE/SSA-only runs that
    // never call this accessor, and computed once (shared across clones) for
    // steady-state / introspection callers that do.
    return ensure_conservation_laws(*impl_->shared, impl_->species);
}

// ─── Functional analytical Jacobian (GH #76) ─────────────────────────────────

FunctionalJacobianContext NetworkModel::functional_jacobian_context() const {
    FunctionalJacobianContext ctx;
    const auto &sd = *impl_->shared;

    for (int r = 0; r < static_cast<int>(sd.reactions.size()); ++r) {
        const auto &rxn = sd.reactions[r];
        if (rxn.rate_law_type != RateLawType::Functional)
            continue;
        auto fit = sd.function_name_to_idx.find(rxn.function_name);
        if (fit == sd.function_name_to_idx.end())
            continue; // no resolvable rate-law expression → leave on FD
        FunctionalJacobianContext::Rxn cr;
        cr.rxn_idx = r;
        cr.rate_expr = impl_->functions[fit->second].expression;
        cr.apply_species_factor = rxn.apply_species_factor;
        for (int si : rxn.reactant_indices)
            cr.reactant_idx0.push_back(si - 1);
        for (int si : rxn.product_indices)
            cr.product_idx0.push_back(si - 1);
        cr.stat_factor = rxn.stat_factor;
        cr.per_species_volume_scaling = rxn.per_species_volume_scaling;
        ctx.functional_reactions.push_back(std::move(cr));
    }

    for (const auto &f : impl_->functions)
        ctx.function_map.emplace_back(f.name, f.expression);

    for (const auto &obs : impl_->observables) {
        std::vector<std::pair<int, double>> grp;
        for (const auto &e : obs.entries)
            grp.emplace_back(e.species_index - 1, e.factor);
        ctx.observables.emplace_back(obs.name, std::move(grp));
    }

    ctx.species_meta.reserve(impl_->species.size());
    for (const auto &s : impl_->species)
        ctx.species_meta.emplace_back(s.amount_valued, s.volume_factor);

    // Constant parameters: every parameter whose name is NOT a function
    // (function-bound synthetic params are time/state-varying and are inlined
    // via function_map instead). Expression-valued (derived) params are
    // constant during integration, so they stay in this set.
    for (const auto &p : impl_->parameters)
        if (sd.function_name_to_idx.find(p.name) == sd.function_name_to_idx.end())
            ctx.constant_names.push_back(p.name);

    return ctx;
}

bool NetworkModel::set_functional_jacobian(const std::vector<FunctionalJacobianInput> &terms) {
    const auto &sd = *impl_->shared;
    const auto &sp = sd.jac_sparsity;
    if (sp.empty())
        return false;

    auto find_csc = [&](int row, int col) -> int64_t {
        int64_t lo = sp.col_ptrs[col], hi = sp.col_ptrs[col + 1];
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (sp.row_indices[mid] < row)
                lo = mid + 1;
            else if (sp.row_indices[mid] > row)
                hi = mid;
            else
                return mid;
        }
        return -1;
    };

    // Build into a local payload; only commit on full success so a partial
    // failure leaves the model cleanly on the FD path.
    FunctionalJacobianData fjac;

    for (const auto &in : terms) {
        if (in.rxn_idx < 0 || in.rxn_idx >= static_cast<int>(sd.reactions.size()))
            return false;
        const auto &rxn = sd.reactions[in.rxn_idx];
        const bool dbg = std::getenv("BNGSIM_JAC_DEBUG") != nullptr;

        // Net stoichiometry per affected species (0-based). Folds stat_factor and
        // a 1/V_i divide (per_species_volume_scaling) into the per-row coeff;
        // shared by both the per-species and the per-observable scatter.
        std::unordered_map<int, double> net;
        for (int si : rxn.reactant_indices)
            if (si - 1 >= 0)
                net[si - 1] -= 1.0;
        for (int si : rxn.product_indices)
            if (si - 1 >= 0)
                net[si - 1] += 1.0;
        auto row_coeff = [&](int i, double c_i) -> double {
            double coeff = rxn.stat_factor * c_i;
            if (rxn.per_species_volume_scaling && i >= 0 &&
                i < static_cast<int>(impl_->species.size()))
                coeff /= impl_->species[i].volume_factor;
            return coeff;
        };

        if (in.per_observable) {
            // .net per-observable path (GH #76 task 2): rate = func(obs) · ∏R.
            // deriv_terms carry ∂func/∂obs_k keyed by observable index; func
            // itself is read at runtime from the reaction's bound rate parameter.
            FunctionalJacobianData::ObservableTerm ot;
            ot.func_param_idx0 = rxn.rate_param_idx0;
            // Reactant multiset (unique species → multiplicity) for ∏R.
            std::unordered_map<int, int> rmult;
            for (int si : rxn.reactant_indices)
                if (si - 1 >= 0)
                    rmult[si - 1] += 1;
            for (const auto &[s, m] : rmult)
                ot.reactants.emplace_back(s, m);

            // Accumulate columns keyed by species j; build affected rows lazily.
            std::unordered_map<int, FunctionalJacobianData::ObservableTerm::Column> cols;
            auto ensure_col = [&](int j) -> FunctionalJacobianData::ObservableTerm::Column & {
                auto it = cols.find(j);
                if (it != cols.end())
                    return it->second;
                FunctionalJacobianData::ObservableTerm::Column c;
                c.species_j = j;
                for (const auto &[i, c_i] : net) {
                    if (c_i == 0.0)
                        continue;
                    int64_t csc = find_csc(i, j);
                    if (csc < 0)
                        throw std::out_of_range("jac csc"); // signalled below
                    c.affected.emplace_back(csc, row_coeff(i, c_i));
                }
                return cols.emplace(j, std::move(c)).first->second;
            };

            try {
                // Term A: one ∂func/∂obs_k per observable index in deriv_terms.
                for (const auto &[obs_idx, deriv_str] : in.deriv_terms) {
                    if (obs_idx < 0 || obs_idx >= static_cast<int>(impl_->observables.size()))
                        return false;
                    int k_idx;
                    try {
                        k_idx = static_cast<int>(ot.dfunc_dobs_eval_ids.size());
                        ot.dfunc_dobs_eval_ids.push_back(impl_->evaluator->compile(deriv_str));
                    } catch (const std::exception &e) {
                        if (dbg)
                            std::fprintf(stderr,
                                         "[jac] rxn %d obs %d compile FAILED: %s\n  expr: %s\n",
                                         in.rxn_idx, obs_idx, e.what(), deriv_str.c_str());
                        return false;
                    }
                    const auto &obs = impl_->observables[obs_idx];
                    for (const auto &entry : obs.entries) {
                        int j = entry.species_index - 1;
                        if (j < 0 || j >= static_cast<int>(impl_->species.size()))
                            continue;
                        double g = entry.factor;
                        if (impl_->species[j].amount_valued)
                            g *= impl_->species[j].volume_factor;
                        if (g == 0.0)
                            continue;
                        ensure_col(j).a_terms.emplace_back(k_idx, g);
                    }
                }
                // Term B: each reactant species is a product-rule column.
                for (const auto &[s, m] : rmult) {
                    auto &c = ensure_col(s);
                    c.is_reactant = true;
                    c.mult_j = m;
                }
            } catch (const std::out_of_range &) {
                if (dbg)
                    std::fprintf(stderr, "[jac] rxn %d: per-observable entry outside sparsity\n",
                                 in.rxn_idx);
                return false; // a derivative entry fell outside the sparsity pattern
            }

            for (auto &[j, c] : cols)
                ot.columns.push_back(std::move(c));
            fjac.observable_terms.push_back(std::move(ot));
            continue;
        }

        // Per-affected-row volume metadata (GH #171). The volume divide is NOT
        // folded into coeff: it is deferred to the runtime scatter so a
        // variable-volume compartment uses the LIVE volume conc[live_idx]. A
        // static-volume / non-varvol row divides by static_divisor (1.0 for a
        // non-per_species_volume_scaling reaction — the byte-identical no-op).
        const int nsp = static_cast<int>(impl_->species.size());
        auto live_idx_of = [&](int i) -> int {
            if (!rxn.per_species_volume_scaling || i < 0 || i >= nsp)
                return -1;
            return impl_->species[i].ode_live_volume_idx0;
        };
        auto static_divisor_of = [&](int i) -> double {
            if (rxn.per_species_volume_scaling && i >= 0 && i < nsp)
                return impl_->species[i].volume_factor;
            return 1.0;
        };

        for (const auto &[species_j, deriv_str] : in.deriv_terms) {
            int eval_id;
            try {
                eval_id = impl_->evaluator->compile(deriv_str);
            } catch (const std::exception &e) {
                if (dbg)
                    std::fprintf(stderr, "[jac] rxn %d species %d compile FAILED: %s\n  expr: %s\n",
                                 in.rxn_idx, species_j, e.what(), deriv_str.c_str());
                return false;
            }
            FunctionalJacobianData::SpeciesTerm st;
            st.species_idx = species_j;
            st.deriv_eval_id = eval_id;
            for (const auto &[i, c_i] : net) {
                if (c_i == 0.0)
                    continue;
                double coeff = rxn.stat_factor * c_i;
                int64_t csc = find_csc(i, species_j);
                if (csc < 0) {
                    if (dbg)
                        std::fprintf(stderr, "[jac] rxn %d: J[%d][%d] outside sparsity pattern\n",
                                     in.rxn_idx, i, species_j);
                    return false; // derivative entry outside the sparsity pattern
                }
                st.affected.push_back({csc, coeff, live_idx_of(i), static_divisor_of(i)});
            }
            if (!st.affected.empty())
                fjac.species_terms.push_back(std::move(st));
        }

        // GH #171: the new ∂/∂V_live column. Independent of in.deriv_terms (the
        // bare-law case k·A·B has no ∂func/∂V_live term at all — the whole column
        // is this contribution), so build it once per per_species_volume_scaling
        // reaction from the affected rows directly, at (row i, col live_idx). The
        // sparsity edge for these was added in build_jac_sparsity; a missing CSC
        // here means the two builders disagree — fail to the FD path.
        if (rxn.per_species_volume_scaling) {
            FunctionalJacobianData::VolumeColumnTerm vt;
            vt.func_param_idx0 = rxn.rate_param_idx0;
            for (const auto &[i, c_i] : net) {
                if (c_i == 0.0)
                    continue;
                int L = live_idx_of(i);
                if (L < 0 || L >= nsp)
                    continue; // static-volume row: no live column
                int64_t csc = find_csc(i, L);
                if (csc < 0) {
                    if (dbg)
                        std::fprintf(stderr,
                                     "[jac] rxn %d: varvol column J[%d][%d] outside sparsity\n",
                                     in.rxn_idx, i, L);
                    return false;
                }
                vt.entries.push_back({csc, rxn.stat_factor * c_i, L});
            }
            if (!vt.entries.empty())
                fjac.volume_terms.push_back(std::move(vt));
        }
    }

    fjac.populated = true;
    impl_->functional_jac = std::move(fjac);

    // Debug escape hatch: trust the assembled analytical Jacobian and skip the
    // FD self-check entirely. Used to isolate a self-check FALSE-fail (a correct
    // Jacobian dropped by a too-tight FD gate) from a genuine symbolic↔engine
    // mismatch — with the check off, a wrong Jacobian surfaces as an integration
    // divergence rather than being silently swapped for FD. Never set in
    // production; the self-check is the correctness backstop.
    if (std::getenv("BNGSIM_JAC_NO_SELFCHECK") != nullptr) {
        if (std::getenv("BNGSIM_JAC_DEBUG"))
            std::fprintf(stderr, "[jac] BNGSIM_JAC_NO_SELFCHECK set: trusting analytical "
                                 "Jacobian without FD validation\n");
        return true;
    }

    // ── Self-validation gate ─────────────────────────────────────────────────
    // The symbolic derivatives can diverge from the engine's actual RHS for
    // rate-law constructs the symbolic core does not yet handle exactly. (The
    // former declaration-order function-evaluation divergence is now fixed at the
    // source: var_param_bindings is topologically sorted in ModelBuilder so a
    // single evaluate_functions pass converges.) Cross-check the assembled
    // analytical Jacobian against reliability-gated finite differences at the
    // initial state plus a few spread-out probe states and DISCARD it (fall back
    // to FD) on any trustworthy mismatch — never ship a wrong Jacobian. One-shot,
    // at load.
    {
        const int ns = n_species();
        std::vector<double> y0(ns);
        bool finite = true;
        for (int i = 0; i < ns; ++i) {
            y0[i] = impl_->species[i].concentration;
            if (!std::isfinite(y0[i]))
                finite = false;
        }
        // Probe at the initial state AND a deterministically spread-out state:
        // some symbolic↔engine divergences are state-dependent (e.g. cancel at
        // the initial point) and only surface away from it.
        std::vector<std::vector<double>> probes;
        if (finite && ns > 0) {
            probes.push_back(y0);
            // Per-index-varied probes: state-dependent symbolic↔engine
            // divergences (e.g. ones tied to a species ratio) survive uniform
            // scaling, so each species gets a DIFFERENT deterministic
            // multiplier from a low-discrepancy (golden-ratio) sequence.
            for (double key : {0.6180339887, 0.3819660113, 0.7548776662}) {
                std::vector<double> yk(ns);
                for (int i = 0; i < ns; ++i) {
                    double frac = (i + 1) * key;
                    frac -= std::floor(frac);
                    yk[i] = std::abs(y0[i]) * (0.4 + 1.8 * frac) + 1e-3;
                }
                probes.push_back(std::move(yk));
            }
        }
        bool ok = true;
        // Reliability-gated central differences. A single-step, 1%-band FD check
        // FALSE-FAILS correct Jacobians on stiff models: at a probe state a
        // large-magnitude RHS loses all significance in (fp-fm) to catastrophic
        // cancellation, or high curvature makes one-step FD inaccurate, so a
        // CORRECT analytical entry reads >1% off and a perfectly good Jacobian is
        // needlessly dropped to FD. The fix is to judge an entry ONLY where the
        // finite difference is itself trustworthy — finite, free of cancellation,
        // and CONVERGED across two well-separated step sizes (Richardson). A
        // genuine symbolic↔engine divergence then surfaces as a reliable FD entry
        // that still disagrees with the analytical value; FD artifacts are simply
        // not judged. Validated across the BioModels SBML corpus (1597 models):
        // zero wrong attaches, ~129 needless fall-backs recovered — see
        // dev/notes/gh76_investigation_findings.md.
        //
        // Per-entry verdict for a converged central difference at two step sizes:
        // true ⇒ a trustworthy analytical↔FD mismatch (reject the Jacobian).
        auto entry_is_mismatch = [](double an, double fp1i, double fm1i, double fp2i, double fm2i,
                                    double h1, double h2) -> bool {
            double d1 = fp1i - fm1i, d2 = fp2i - fm2i;
            double fd1 = d1 / (2.0 * h1), fd2 = d2 / (2.0 * h2);
            if (!std::isfinite(fd1) || !std::isfinite(fd2))
                return false;
            double sc1 = std::max(std::max(std::abs(fp1i), std::abs(fm1i)), 1.0);
            double sc2 = std::max(std::max(std::abs(fp2i), std::abs(fm2i)), 1.0);
            if (std::abs(d1) <= 1e-9 * sc1 || std::abs(d2) <= 1e-9 * sc2)
                return false; // catastrophic cancellation: slope is float noise
            double cden = std::max(std::max(std::abs(fd1), std::abs(fd2)), 1e-6);
            if (std::abs(fd1 - fd2) > 5e-3 * cden)
                return false; // not Richardson-converged: FD is not a trustworthy oracle
            double ad = std::abs(an - fd2);
            double denom = std::max(std::max(std::abs(an), std::abs(fd2)), 1e-6);
            return ad > 1e-6 && ad > 1e-2 * denom; // symmetric 1% band on the converged estimate
        };

        std::vector<double> f0(ns);
        std::vector<double> fp1(ns), fm1(ns), fp2(ns), fm2(ns);

        // Dense exhaustive validation costs O(ns²) memory + O(ns) RHS evals per
        // probe — fine for the modest models it was validated on, but infeasible
        // for very large sparse models (a genome-scale model's dense Jacobian is
        // many gigabytes and the per-column FD loop is ~10^5 RHS evals). Above a
        // crossover
        // (BNGSIM_JAC_SELFCHECK_DENSE_MAX, default 4096) validate SPARSELY (GH
        // #151): assemble into the nnz-length CSC value array (no dense buffer),
        // check EVERY analytical entry for finiteness, and reliability-gate FD on
        // a deterministic sample of columns (BNGSIM_JAC_SELFCHECK_SAMPLE, default
        // 256) — each sampled column compared across ALL rows, so a wrong value or
        // a missing structural entry is still caught where sampled. The native
        // saturable derivatives (GH #151) are independently FD-validated and match
        // SymPy to ~1e-14, so a column sample (offset per probe ⇒ wider coverage)
        // is a sound trade for a model that otherwise gets no analytical Jacobian.
        auto env_int = [](const char *name, long dflt) -> long {
            const char *e = std::getenv(name);
            if (!e)
                return dflt;
            char *end = nullptr;
            long v = std::strtol(e, &end, 10);
            if (end == e || v < 0)
                return dflt;
            return v;
        };
        const long dense_max = env_int("BNGSIM_JAC_SELFCHECK_DENSE_MAX", 4096);
        const bool use_sparse = static_cast<long>(ns) > dense_max;

        if (!use_sparse) {
            std::vector<double> Ja(static_cast<size_t>(ns) * ns);
            for (const auto &y : probes) {
                // Skip a probe whose RHS is non-finite (degenerate point) rather
                // than failing the whole check on it.
                compute_derivs(0.0, y.data(), f0.data());
                bool rhs_finite = true;
                for (double v : f0)
                    if (!std::isfinite(v))
                        rhs_finite = false;
                if (!rhs_finite)
                    continue;
                fill_dense_analytical_jacobian(0.0, y.data(), Ja.data());
                // A non-finite analytical entry at a state whose RHS is finite
                // would poison the integrator's Newton solve (e.g.
                // ∂(k·sqrt(A))/∂A = k/(2·sqrt(A)) → ∞ at A=0 while the rate is a
                // finite 0). FD tolerates this, so such a model must stay on FD.
                for (double v : Ja)
                    if (!std::isfinite(v)) {
                        ok = false;
                        break;
                    }
                for (int j = 0; j < ns && ok; ++j) {
                    // Scale-RELATIVE FD step (GH #168). A step floored at 1.0
                    // (the former max(|y[j]|,1.0)) is ~1e7× the value for
                    // amount-scaled models (species ~1e-12 from a converted SBML
                    // compartmental model), which both leaves the linear regime
                    // AND drives a nonnegative species across the converter-
                    // emitted division guards `if(X>1e-300, expr/X, 0)` — the
                    // downward half drops X below the guard, the central
                    // difference reads exactly HALF the true slope, BOTH steps
                    // cross identically so the Richardson guard is fooled, and a
                    // CORRECT analytical Jacobian is rejected. A step relative to
                    // |y[j]| keeps the perturbation a small fraction of the value:
                    // it stays in the linear regime and a nonnegative species
                    // never crosses zero into a guard branch. For |y[j]|≥1 this is
                    // byte-identical to the old floored step (max(|y|,1)==|y|), so
                    // O(1)-scale models — and the corrupted-Jacobian safety test —
                    // are unchanged. A zero species has no central difference, so
                    // SKIP it (never reject): the per-entry finiteness check and
                    // the independently FD-validated native derivatives (GH #151)
                    // still guard correctness, and where |y[j]| is merely too
                    // small to difference reliably the catastrophic-cancellation
                    // guard in entry_is_mismatch already drops the noise-level
                    // slope — erring toward a smaller step is safe.
                    double aj = std::abs(y[j]);
                    if (!(aj > 0.0))
                        continue; // zero species ⇒ no FD oracle; skip, don't reject
                    double h1 = 1e-5 * aj, h2 = 5e-7 * aj; // two well-separated steps
                    std::vector<double> yp = y, ym = y;
                    yp[j] = y[j] + h1;
                    ym[j] = y[j] - h1;
                    compute_derivs(0.0, yp.data(), fp1.data());
                    compute_derivs(0.0, ym.data(), fm1.data());
                    yp[j] = y[j] + h2;
                    ym[j] = y[j] - h2;
                    compute_derivs(0.0, yp.data(), fp2.data());
                    compute_derivs(0.0, ym.data(), fm2.data());
                    for (int i = 0; i < ns; ++i) {
                        double an = Ja[static_cast<size_t>(j) * ns + i];
                        if (entry_is_mismatch(an, fp1[i], fm1[i], fp2[i], fm2[i], h1, h2)) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (!ok)
                    break;
            }
        } else {
            const auto &sp = impl_->shared->jac_sparsity;
            const int64_t nnz = sp.nnz;
            long sample = env_int("BNGSIM_JAC_SELFCHECK_SAMPLE", 256);
            if (sample > ns)
                sample = ns;
            std::vector<double> vals(static_cast<size_t>(nnz));
            std::vector<double> acol(ns); // dense scratch for one sampled column
            for (size_t pi = 0; pi < probes.size() && ok; ++pi) {
                const auto &y = probes[pi];
                compute_derivs(0.0, y.data(), f0.data());
                bool rhs_finite = true;
                for (double v : f0)
                    if (!std::isfinite(v))
                        rhs_finite = false;
                if (!rhs_finite)
                    continue;
                fill_sparse_analytical_jacobian(0.0, y.data(), vals.data());
                // Finiteness over EVERY analytical entry (the dangerous failure
                // mode — a NaN/Inf poisons the whole Newton solve), full coverage.
                int64_t nonfinite = 0, first_bad = -1;
                for (int64_t k = 0; k < nnz; ++k)
                    if (!std::isfinite(vals[k])) {
                        ++nonfinite;
                        if (first_bad < 0)
                            first_bad = k;
                    }
                if (nonfinite > 0) {
                    if (std::getenv("BNGSIM_JAC_DEBUG"))
                        std::fprintf(stderr,
                                     "[jac] sparse self-check probe %zu: %lld non-finite "
                                     "analytical entries (first at csc=%lld, row=%d, value=%g)\n",
                                     pi, (long long)nonfinite, (long long)first_bad,
                                     (int)sp.row_indices[first_bad], vals[first_bad]);
                    ok = false;
                    break;
                }
                std::vector<double> yp = y, ym = y;
                for (long s = 0; s < sample && ok; ++s) {
                    // Golden-ratio low-discrepancy column pick, offset per probe so
                    // successive probes sample DIFFERENT columns (wider coverage).
                    double frac = (static_cast<double>(s) + 1.0 + 0.123456789 * pi) * 0.6180339887;
                    frac -= std::floor(frac);
                    int j = static_cast<int>(frac * ns);
                    if (j >= ns)
                        j = ns - 1;
                    // Scale-relative FD step, identical rationale to the dense
                    // path above (GH #168): relative to |y[j]| so an amount-scaled
                    // species is not perturbed across its rate-law guard branches,
                    // byte-identical for |y[j]|≥1, and a zero species is skipped
                    // (not rejected). Skipping before mutating yp/ym keeps them
                    // restored for the next sampled column.
                    double aj = std::abs(y[j]);
                    if (!(aj > 0.0))
                        continue; // zero species ⇒ no FD oracle; skip, don't reject
                    double h1 = 1e-5 * aj, h2 = 5e-7 * aj;
                    yp[j] = y[j] + h1;
                    ym[j] = y[j] - h1;
                    compute_derivs(0.0, yp.data(), fp1.data());
                    compute_derivs(0.0, ym.data(), fm1.data());
                    yp[j] = y[j] + h2;
                    ym[j] = y[j] - h2;
                    compute_derivs(0.0, yp.data(), fp2.data());
                    compute_derivs(0.0, ym.data(), fm2.data());
                    yp[j] = y[j]; // restore for the next sampled column
                    ym[j] = y[j];
                    // Analytical column j as a dense vector (0 outside the
                    // pattern) so out-of-pattern rows are checked too — catches a
                    // missing structural entry, matching the dense path's rigor.
                    std::fill(acol.begin(), acol.end(), 0.0);
                    for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k)
                        acol[sp.row_indices[k]] = vals[k];
                    for (int i = 0; i < ns; ++i) {
                        if (entry_is_mismatch(acol[i], fp1[i], fm1[i], fp2[i], fm2[i], h1, h2)) {
                            if (std::getenv("BNGSIM_JAC_DEBUG")) {
                                double fd2 = (fp2[i] - fm2[i]) / (2.0 * h2);
                                std::fprintf(stderr,
                                             "[jac] sparse self-check probe %zu: FD mismatch at "
                                             "(row=%d, col=%d) analytical=%g fd=%g\n",
                                             pi, i, j, acol[i], fd2);
                            }
                            ok = false;
                            break;
                        }
                    }
                }
            }
        }
        // compute_derivs left observables/params/time at the last probe;
        // re-sync to the initial state so the model is clean post-attach.
        if (!probes.empty()) {
            update_observables(y0.data());
            evaluate_functions(0.0);
        }
        if (!ok) {
            if (std::getenv("BNGSIM_JAC_DEBUG"))
                std::fprintf(stderr, "[jac] analytical Jacobian failed FD self-check; "
                                     "falling back to finite differences\n");
            impl_->functional_jac = FunctionalJacobianData{}; // populated=false
            return false;
        }
    }
    return true;
}

void NetworkModel::fill_dense_analytical_jacobian(double t, const double *conc, double *jac) {
    // Single source of truth for the dense analytical Jacobian: the CVODE dense
    // callback delegates here, and the test hook calls it directly so the
    // entrywise FD check validates the *real* assembly. Column-major
    // (jac[j*ns + i] = ∂f_i/∂x_j), matching SUNDenseMatrix_Data.
    const int ns = n_species();
    const auto &ajd = impl_->shared->analytical_jac;
    const auto &params = impl_->parameters;
    const auto &sp = impl_->shared->jac_sparsity;

    std::memset(jac, 0, static_cast<size_t>(ns) * ns * sizeof(double));

    // Elementary mass-action contributions (closed form).
    for (const auto &rxn_terms : ajd.reactions) {
        if (rxn_terms.rate_param_idx0 < 0 || rxn_terms.reactants.empty())
            continue;
        double k = params[rxn_terms.rate_param_idx0].value;
        double k_sf = k * rxn_terms.stat_factor * rxn_terms.amount_factor;
        if (k_sf == 0.0)
            continue;
        for (const auto &pr : rxn_terms.reactants) {
            double dv_dxj = k_sf * static_cast<double>(pr.multiplicity);
            double xj = conc[pr.species_idx];
            for (int p = 0; p < pr.multiplicity - 1; ++p)
                dv_dxj *= xj;
            for (const auto &other : pr.others) {
                double xi = conc[other.species_idx];
                for (int p = 0; p < other.multiplicity; ++p)
                    dv_dxj *= xi;
            }
            for (const auto &[data_idx, stoich] : pr.affected) {
                int row = static_cast<int>(sp.row_indices[data_idx]);
                jac[pr.species_idx * ns + row] += stoich * dv_dxj;
            }
        }
    }

    // Michaelis–Menten (tQSSA) closed-form contributions (GH #76 task 3). Reads
    // the live state directly (E, S are species, not observables), so no
    // observable refresh is needed. The sparse CVODE callback scatters the
    // identical math (cvode_simulator.cpp).
    for (const auto &mt : ajd.mm_reactions) {
        double dE, dS;
        mm_tqssa_derivatives(params[mt.kcat_param_idx0].value, params[mt.km_param_idx0].value,
                             conc[mt.e_idx], conc[mt.s_idx], mt.stat_factor, dE, dS);
        if (dE != 0.0)
            for (const auto &[data_idx, coeff] : mt.e_affected)
                jac[mt.e_idx * ns + sp.row_indices[data_idx]] += coeff * dE;
        if (dS != 0.0)
            for (const auto &[data_idx, coeff] : mt.s_affected)
                jac[mt.s_idx * ns + sp.row_indices[data_idx]] += coeff * dS;
    }

    // Functional rate-law contributions (symbolic derivatives, GH #76).
    if (impl_->functional_jac.populated) {
        update_observables(conc);
        set_current_time(t);
        auto &eval = *impl_->evaluator;
        for (const auto &st : impl_->functional_jac.species_terms) {
            double dv_dxj = eval.evaluate(st.deriv_eval_id);
            if (dv_dxj == 0.0)
                continue;
            for (const auto &a : st.affected) {
                // GH #171: divide by the LIVE compartment volume for a
                // variable-volume row, else the static divisor (1.0 for a
                // non-varvol row ⇒ byte-identical to the pre-#171 folded coeff).
                double divisor = a.static_divisor;
                if (a.live_idx >= 0 && conc[a.live_idx] > 0.0)
                    divisor = conc[a.live_idx];
                int row = static_cast<int>(sp.row_indices[a.csc]);
                jac[st.species_idx * ns + row] += a.coeff * dv_dxj / divisor;
            }
        }
        // GH #171: the new ∂/∂V_live column for cross-compartment variable-volume
        // reactions — −(stat·netstoichᵢ·func)/V_live² at (row i, col live_idx).
        // func is a function-bound rate parameter, so refresh it (exact RHS
        // parity) before reading; paid only when a varvol term exists.
        if (!impl_->functional_jac.volume_terms.empty()) {
            evaluate_functions(t);
            for (const auto &vt : impl_->functional_jac.volume_terms) {
                if (vt.func_param_idx0 < 0 ||
                    vt.func_param_idx0 >= static_cast<int>(params.size()))
                    continue;
                double func = params[vt.func_param_idx0].value;
                if (func == 0.0)
                    continue;
                for (const auto &e : vt.entries) {
                    double V = conc[e.live_idx];
                    if (!(V > 0.0))
                        continue; // RHS used the static divisor here ⇒ ∂/∂V = 0
                    int row = static_cast<int>(sp.row_indices[e.csc]);
                    jac[e.live_idx * ns + row] += -e.coeff * func / (V * V);
                }
            }
        }
        // Per-observable (.net) product-rule contributions (GH #76 task 2). The
        // helper refreshes the function-bound rate params it reads for `func`; the
        // sparse CVODE callback scatters the identical math (cvode_simulator.cpp).
        scatter_functional_observable_terms(
            *this, conc, t,
            [&](int64_t /*csc*/, int col, int row, double v) { jac[col * ns + row] += v; });
    }

    // Fixed (boundary-condition) species have ∂f_i/∂x_j = 0 for all j.
    for (const auto &s : impl_->species) {
        if (s.fixed) {
            int row = s.index - 1;
            for (int j = 0; j < ns; ++j)
                jac[j * ns + row] = 0.0;
        }
    }
}

void NetworkModel::fill_sparse_analytical_jacobian(double t, const double *conc, double *vals) {
    // Sparse mirror of fill_dense_analytical_jacobian: identical math, written by
    // CSC data index instead of (col·n + row). This is the canonical sparse
    // assembly — the CVODE sparse callback (cvode_simulator.cpp) reinstalls the
    // CSC structure and then delegates here, and the large-model self-check (GH
    // #151) uses it to validate without a dense n×n buffer.
    const int ns = n_species();
    const auto &ajd = impl_->shared->analytical_jac;
    const auto &params = impl_->parameters;
    const auto &sp = impl_->shared->jac_sparsity;
    const int64_t nnz = sp.nnz;

    std::memset(vals, 0, static_cast<size_t>(nnz) * sizeof(double));

    // Elementary mass-action contributions (closed form).
    for (const auto &rxn_terms : ajd.reactions) {
        if (rxn_terms.rate_param_idx0 < 0 || rxn_terms.reactants.empty())
            continue;
        double k = params[rxn_terms.rate_param_idx0].value;
        double k_sf = k * rxn_terms.stat_factor * rxn_terms.amount_factor;
        if (k_sf == 0.0)
            continue;
        for (const auto &pr : rxn_terms.reactants) {
            double dv_dxj = k_sf * static_cast<double>(pr.multiplicity);
            double xj = conc[pr.species_idx];
            for (int p = 0; p < pr.multiplicity - 1; ++p)
                dv_dxj *= xj;
            for (const auto &other : pr.others) {
                double xi = conc[other.species_idx];
                for (int p = 0; p < other.multiplicity; ++p)
                    dv_dxj *= xi;
            }
            for (const auto &[data_idx, stoich] : pr.affected)
                vals[data_idx] += stoich * dv_dxj;
        }
    }

    // Michaelis–Menten (tQSSA) closed-form contributions (GH #76 task 3).
    for (const auto &mt : ajd.mm_reactions) {
        double dE, dS;
        mm_tqssa_derivatives(params[mt.kcat_param_idx0].value, params[mt.km_param_idx0].value,
                             conc[mt.e_idx], conc[mt.s_idx], mt.stat_factor, dE, dS);
        if (dE != 0.0)
            for (const auto &[data_idx, coeff] : mt.e_affected)
                vals[data_idx] += coeff * dE;
        if (dS != 0.0)
            for (const auto &[data_idx, coeff] : mt.s_affected)
                vals[data_idx] += coeff * dS;
    }

    // Functional rate-law contributions (symbolic derivatives, GH #76).
    if (impl_->functional_jac.populated) {
        update_observables(conc);
        set_current_time(t);
        auto &eval = *impl_->evaluator;
        for (const auto &st : impl_->functional_jac.species_terms) {
            double dv_dxj = eval.evaluate(st.deriv_eval_id);
            if (dv_dxj == 0.0)
                continue;
            for (const auto &a : st.affected) {
                // GH #171: live-volume divide (mirror of the dense path).
                double divisor = a.static_divisor;
                if (a.live_idx >= 0 && conc[a.live_idx] > 0.0)
                    divisor = conc[a.live_idx];
                vals[a.csc] += a.coeff * dv_dxj / divisor;
            }
        }
        // GH #171: the new ∂/∂V_live column (sparse mirror of the dense path).
        if (!impl_->functional_jac.volume_terms.empty()) {
            evaluate_functions(t);
            for (const auto &vt : impl_->functional_jac.volume_terms) {
                if (vt.func_param_idx0 < 0 ||
                    vt.func_param_idx0 >= static_cast<int>(params.size()))
                    continue;
                double func = params[vt.func_param_idx0].value;
                if (func == 0.0)
                    continue;
                for (const auto &e : vt.entries) {
                    double V = conc[e.live_idx];
                    if (!(V > 0.0))
                        continue; // RHS used the static divisor here ⇒ ∂/∂V = 0
                    vals[e.csc] += -e.coeff * func / (V * V);
                }
            }
        }
        scatter_functional_observable_terms(
            *this, conc, t,
            [&](int64_t csc, int /*col*/, int /*row*/, double v) { vals[csc] += v; });
    }

    // Fixed (boundary-condition) species: zero their row across all columns.
    for (const auto &s : impl_->species) {
        if (!s.fixed)
            continue;
        int row = s.index - 1;
        for (int j = 0; j < ns; ++j)
            for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k)
                if (static_cast<int>(sp.row_indices[k]) == row)
                    vals[k] = 0.0;
    }
}
const std::vector<std::pair<int, int>> &NetworkModel::species_ic_param_refs() const {
    return impl_->shared->species_ic_param_refs;
}

std::vector<std::string> NetworkModel::species_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->species.size());
    for (const auto &s : impl_->species) {
        names.push_back(s.name);
    }
    return names;
}

std::vector<std::size_t> NetworkModel::reported_species_indices() const {
    std::vector<std::size_t> idx;
    idx.reserve(impl_->species.size());
    for (std::size_t i = 0; i < impl_->species.size(); ++i) {
        if (impl_->species[i].reported) {
            idx.push_back(i);
        }
    }
    return idx;
}

std::vector<double> NetworkModel::reported_volume_factors() const {
    // V_c for each reported species, in species order — mirrors the
    // codegen_data()["species"] filter `[s.volume_factor for s if s.reported]`
    // the output layer used to build a whole-model dict for (T7).
    std::vector<double> vf;
    vf.reserve(impl_->species.size());
    for (const auto &s : impl_->species) {
        if (s.reported) {
            vf.push_back(s.volume_factor);
        }
    }
    return vf;
}

std::vector<std::string> NetworkModel::observable_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->observables.size());
    for (const auto &o : impl_->observables) {
        names.push_back(o.name);
    }
    return names;
}

std::vector<std::string> NetworkModel::function_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->functions.size());
    for (const auto &f : impl_->functions) {
        names.push_back(f.name);
    }
    return names;
}

std::vector<double> NetworkModel::function_values() const {
    // Return current evaluated values of all functions.
    // evaluate_functions() must have been called first.
    std::vector<double> vals;
    vals.reserve(impl_->functions.size());
    for (const auto &f : impl_->functions) {
        if (f.evaluator_id >= 0) {
            vals.push_back(impl_->evaluator->evaluate(f.evaluator_id));
        } else {
            vals.push_back(0.0);
        }
    }
    return vals;
}

// ─── RHS evaluation ──────────────────────────────────────────────────────────

void NetworkModel::update_observables(const double *conc) {
    const auto &species_list = impl_->species;
    for (auto &obs : impl_->observables) {
        double total = 0.0;
        for (const auto &entry : obs.entries) {
            // species_index is 1-based, conc array is 0-based
            int idx0 = entry.species_index - 1;
            if (idx0 >= 0 && idx0 < static_cast<int>(species_list.size())) {
                // Amount-valued species (GH #75): a hOSU symbol denotes the
                // species's *amount* (stored × volume_factor), not the stored
                // concentration. The per-species observable IS the ExprTk
                // symbol an SBML rate law / observable-expression resolves to
                // (the loader registers a same-named observable that shadows
                // the species variable), so restoring the amount here is the
                // single read site that makes Functional laws and
                // observable-sums read amounts — replacing the loader's
                // per-emission `_wrap_hosu_amounts` / `_hosu_amount_factor`
                // rewrites. Default false / volume_factor 1.0 ⇒ stored value
                // unchanged (byte-identical for .net, V=1 SBML, hOSU=false).
                double v = conc[idx0];
                const auto &sp = species_list[idx0];
                if (sp.amount_valued) {
                    v *= sp.volume_factor;
                }
                total += entry.factor * v;
            }
        }
        obs.total = total;
    }
}

void NetworkModel::evaluate_functions(double t) {
    impl_->current_time = t;

    if (!impl_->has_functions)
        return;

    // Function values are also cached by declaration index so the recording
    // path can read them without a redundant ExprTk pass (GH #136). Lazily
    // sized; slots for functions with evaluator_id < 0 stay 0.0 (matching
    // function_values()).
    auto &cache = impl_->function_value_cache;
    if (cache.size() != impl_->functions.size())
        cache.assign(impl_->functions.size(), 0.0);

    // Evaluate each function expression and update its bound parameter
    for (const auto &[func_idx, param_idx] : impl_->shared->var_param_bindings) {
        const auto &func = impl_->functions[func_idx];
        if (func.evaluator_id >= 0) {
            double val = impl_->evaluator->evaluate(func.evaluator_id);
            impl_->parameters[param_idx].value = val;
            cache[func_idx] = val;
        }
    }
}

const std::vector<double> &NetworkModel::function_value_cache() const {
    return impl_->function_value_cache;
}

// ─── RHS observable/function-eval gate instrumentation (T1) ──────────────────

bool NetworkModel::rhs_evaluates_observables() const { return impl_->has_functions; }
std::uint64_t NetworkModel::rhs_eval_count() const { return impl_->rhs_eval_count; }
std::uint64_t NetworkModel::rhs_observable_eval_count() const {
    return impl_->rhs_obs_func_eval_count;
}
void NetworkModel::reset_rhs_counters() {
    impl_->rhs_eval_count = 0;
    impl_->rhs_obs_func_eval_count = 0;
}

// Helper to compute species population factor.
// For ODE (discrete=false): simple product of concentrations.
// For SSA (discrete=true): falling factorial for repeated reactants.
//
// ODE path uses 1-based reactant_indices (original format).
// SSA path uses pre-computed reactant_multiplicities (0-based).
//
// Amount-valued species (GH #75): when `species_list[si].amount_valued` is
// true, the species participates by its *amount* (stored value × volume_factor)
// rather than the stored concentration. `amount_valued` defaults false and
// `volume_factor` defaults 1.0, so the read is the stored value unchanged for
// `.net` models, V=1 SBML, and every hOSU=false species (byte-identical). The
// SSA falling factorial is taken over the amount, which is the physically
// correct discrete population for an hOSU=true species.
static double compute_species_factor_ode(const std::vector<int> &reactant_indices,
                                         const double *conc, int n_species,
                                         const std::vector<Species> &species_list) {
    double factor = 1.0;
    for (int ri : reactant_indices) {
        int si = ri - 1;
        if (si >= 0 && si < n_species) {
            double v = conc[si];
            if (species_list[si].amount_valued) {
                v *= species_list[si].volume_factor;
            }
            factor *= v;
        }
    }
    return factor;
}

// SSA version: uses pre-computed multiplicities (zero heap allocation)
static inline double
compute_species_factor_ssa(const std::vector<std::pair<int, int>> &multiplicities,
                           const double *conc, const std::vector<Species> &species_list) {
    double factor = 1.0;
    for (const auto &[si, count] : multiplicities) {
        double n = conc[si];
        if (species_list[si].amount_valued) {
            n *= species_list[si].volume_factor;
        }
        for (int k = 0; k < count; ++k) {
            factor *= (n - k);
        }
    }
    return factor;
}

// Compute a single reaction rate (for both ODE and SSA)
static double
compute_rxn_rate(const Reaction &rxn, const std::vector<Parameter> &params, const double *conc,
                 int n_species, const std::vector<Species> &species_list,
                 bool discrete // true for SSA (integer populations), false for ODE (continuous)
) {
    double rate = 0.0;

    switch (rxn.rate_law_type) {
    case RateLawType::Elementary:
    case RateLawType::Functional: {
        // rate = k_or_func_value * stat_factor [* species_factor]
        // The species-factor multiplication is the BNGL convention; it can
        // be disabled per-reaction (apply_species_factor=false) when the
        // function expression already encodes the full propensity, as the
        // SBML loader's unified emission does.
        int k_idx = rxn.rate_param_idx0; // pre-computed 0-based index
        if (k_idx < 0 || k_idx >= static_cast<int>(params.size()))
            return 0.0;

        rate = params[k_idx].value * rxn.stat_factor;
        if (rxn.apply_species_factor) {
            if (discrete) {
                rate *= compute_species_factor_ssa(rxn.reactant_multiplicities, conc, species_list);
            } else {
                rate *=
                    compute_species_factor_ode(rxn.reactant_indices, conc, n_species, species_list);
            }
        }
        // SSA-only: convert ODE-units rate (storage/time) to amount/time
        // propensity. Default 1.0 → no-op for .net and V=1 SBML.
        if (discrete) {
            rate *= rxn.ssa_volume_factor;
            // GH #81: live compartment-volume correction. For a mass-action
            // reaction in a variable-volume compartment the baked
            // ssa_volume_factor (= V_static) no longer matches the live volume;
            // each hOSU=false concentration factor is stale by V_static/V_live,
            // so the exact monomial correction is (V_static/V_live)^(n_h-1).
            // idx0<0 (the default) ⇒ no-op, so .net / static-V stay identical.
            if (rxn.ssa_live_volume_idx0 >= 0 && rxn.ssa_live_volume_exp != 0.0) {
                double v_live = conc[rxn.ssa_live_volume_idx0];
                if (v_live > 0.0) {
                    rate *= std::pow(rxn.ssa_volume_factor / v_live, rxn.ssa_live_volume_exp);
                }
            }
            // GH #144 (case 4): cross-compartment correction — a product of
            // per-varvol-compartment (V_static/V_live)^m_c ratios. Empty by default
            // (the scalar path / single-compartment / .net stay byte-identical). A
            // reaction populates either the scalar above or this vector, not both.
            for (const auto &lt : rxn.ssa_live_volume_terms) {
                if (lt.exp == 0.0)
                    continue;
                double v_live = conc[lt.live_idx0];
                if (v_live > 0.0) {
                    rate *= std::pow(lt.v_static / v_live, lt.exp);
                }
            }
        }
        break;
    }

    case RateLawType::MichaelisMenten: {
        // MM kcat Km: tQSSA (total quasi-steady-state approximation)
        //
        // Matches NFsim's MMRxnClass::update_a() formula:
        //   sFree = 0.5 * ((S - Km - E) + sqrt((S - Km - E)^2 + 4*Km*S))
        //   rate  = kcat * sFree * E / (Km + sFree)
        //
        // This is more accurate than sQSSA (rate = kcat*E*S/(Km+S)) across
        // all E/S ratios, and reduces to sQSSA when E << S + Km.
        // E is first reactant, S is second reactant.
        if (rxn.rate_law_param_indices.size() < 2)
            return 0.0;
        int kcat_idx = rxn.rate_law_param_indices[0] - 1;
        int km_idx = rxn.rate_law_param_indices[1] - 1;
        if (kcat_idx < 0 || km_idx < 0)
            return 0.0;

        double kcat = params[kcat_idx].value;
        double Km = params[km_idx].value;

        if (rxn.reactant_indices.size() >= 2) {
            int e_si = rxn.reactant_indices[0] - 1;
            int s_si = rxn.reactant_indices[1] - 1;
            if (e_si >= 0 && e_si < n_species && s_si >= 0 && s_si < n_species) {
                double E = conc[e_si];
                double S = conc[s_si];

                // tQSSA: compute free substrate
                double delta = S - Km - E;
                double sFree = 0.5 * (delta + std::sqrt(delta * delta + 4.0 * Km * S));

                // Guard against numerical issues (sFree should be non-negative)
                if (sFree < 0.0)
                    sFree = 0.0;

                rate = kcat * rxn.stat_factor * sFree * E / (Km + sFree);
            }
        }
        break;
    }
    } // switch

    return rate;
}

void NetworkModel::compute_derivs(double t, const double *conc, double *derivs) {
    // GH #106: refresh the live rateOf buffer before the real RHS so any
    // rate_of__<species> accessor (in a rate-rule / assignment-rule function)
    // reads dx/dt at this (t, conc). One probe is exact — see the header note
    // and dev/notes/gh106_rateof_kickoff.md. No-op when !uses_rateof.
    refresh_rateof_derivs(t, conc);
    compute_derivs_core(t, conc, derivs);
}

bool NetworkModel::uses_rateof() const { return impl_->uses_rateof; }

void NetworkModel::refresh_rateof_derivs(double t, const double *conc) {
    if (!impl_->uses_rateof) {
        return;
    }
    // Probe into a separate scratch so compute_derivs_core never reads the
    // accessor source (current_derivs) mid-accumulation, then publish the whole
    // vector atomically. The probe itself reads whatever current_derivs holds
    // (stale), which is harmless: a rateOf argument is a species whose
    // derivative is independent of every rateOf consumer, so the argument
    // entries it produces are exact regardless of the stale reads.
    compute_derivs_core(t, conc, impl_->rateof_scratch.data());
    // GH #231 (sub-cluster 3 / 01463): rateof_scratch[i] is d(stored)/dt, and the
    // stored value is amount/V_static regardless of live volume (the integrator
    // works in static-concentration units; the Simulator rescales to live volume
    // separately). A hasOnlySubstanceUnits=true species's symbol denotes its
    // amount, so its rateOf csymbol is d(amount)/dt = V_static·d(stored)/dt =
    // volume_factor·rateof_scratch[i] — for CONSTANT and VARIABLE volume alike (the
    // conc·V̇ term is already absorbed by the static-units storage; 01455/01457
    // constant-V, 01463 rate-rule varvol). Scale the published buffer for those
    // species so every rate_of__<species> accessor reads the amount-rate. Untouched
    // (factor 1) for .net, V=1, hOSU=false (report_rateof_amount stays false).
    const std::size_t ns = impl_->species.size();
    for (std::size_t i = 0; i < ns; ++i) {
        if (impl_->species[i].report_rateof_amount)
            impl_->rateof_scratch[i] *= impl_->species[i].volume_factor;
    }
    impl_->current_derivs = impl_->rateof_scratch;
}

void NetworkModel::compute_derivs_core(double t, const double *conc, double *derivs) {
    const int ns = n_species();
    const int nr = n_reactions();

    ++impl_->rhs_eval_count;

    // Zero derivatives
    std::memset(derivs, 0, ns * sizeof(double));

    // 1+2. Refresh observable totals + function-bound parameters — but ONLY for
    // models that have functions. The rate loop below reads species
    // concentrations and parameter values directly; the ONLY RHS consumer of an
    // observable sum (obs.total) is the ExprTk evaluator, and the evaluator runs
    // on the RHS solely inside evaluate_functions() (which iterates
    // var_param_bindings and early-returns when !has_functions). So for a pure
    // mass-action model — no functions, hence no Functional rate law and no
    // function-bound rate constant — both passes are pure dead work × the CVODE
    // RHS count. Skipping them is byte-identical: obs.total goes stale but is
    // never read, and the only side effect we must preserve is
    // evaluate_functions()'s current_time update (some out-of-RHS readers and
    // the rateOf probe expect t to be current). The SSA hot loop already gates
    // these same two calls (ssa_simulator.cpp, on has_functional_rates); this is
    // the ODE counterpart. Functional / function-bound-MM / event- or
    // assignment-rule- / table-function- / rateOf-driven models all carry
    // functions ⇒ has_functions ⇒ both passes run exactly as before.
    if (impl_->has_functions) {
        ++impl_->rhs_obs_func_eval_count;
        update_observables(conc);
        evaluate_functions(t);
    } else {
        impl_->current_time = t;
    }

    // 3. Compute reaction rates and accumulate derivatives
    const auto &species_list = impl_->species;
    for (int r = 0; r < nr; ++r) {
        const auto &rxn = impl_->shared->reactions[r];
        double rate = compute_rxn_rate(rxn, impl_->parameters, conc, ns, species_list, false);

        if (rate == 0.0)
            continue;

        // Cross-compartment unified emission: divide by each species's
        // volume_factor at accumulation time. The kinetic-law function
        // already evaluates to amount/time, but storage is `amount/V_c`,
        // so dStorage[s]/dt = (c-b) * kinetic_law / V_c(s) — and V_c(s)
        // can differ across species, so it cannot be folded into a single
        // scalar rate. Default false preserves the .net and uniform-V_s
        // SBML behavior (single rate, no per-species divide).
        if (rxn.per_species_volume_scaling) {
            // GH #144 (case 4): for an hOSU=false species in a variable-volume
            // compartment, storage tracks live concentration `amount/V_live`, so
            // the per-species divide must use V_live (= conc[ode_live_volume_idx0],
            // the promoted compartment species) rather than the static
            // volume_factor. idx0 < 0 (the default, and every static-compartment /
            // hOSU=true / .net species) ⇒ volume_factor, byte-identical to pre-#144.
            auto species_divisor = [&](int si) -> double {
                int lv = species_list[si].ode_live_volume_idx0;
                if (lv >= 0 && lv < ns) {
                    double v_live = conc[lv];
                    if (v_live > 0.0)
                        return v_live;
                }
                return species_list[si].volume_factor;
            };
            for (int ri : rxn.reactant_indices) {
                int si = ri - 1;
                if (si >= 0 && si < ns) {
                    derivs[si] -= rate / species_divisor(si);
                }
            }
            for (int pi : rxn.product_indices) {
                int si = pi - 1;
                if (si >= 0 && si < ns) {
                    derivs[si] += rate / species_divisor(si);
                }
            }
            continue;
        }

        // Subtract from reactants
        for (int ri : rxn.reactant_indices) {
            int si = ri - 1;
            if (si >= 0 && si < ns) {
                derivs[si] -= rate;
            }
        }
        // Add to products
        for (int pi : rxn.product_indices) {
            int si = pi - 1;
            if (si >= 0 && si < ns) {
                derivs[si] += rate;
            }
        }
    }

    // 4. Zero derivatives for fixed (boundary condition) species
    for (const auto &s : impl_->species) {
        if (s.fixed) {
            derivs[s.index - 1] = 0.0;
        }
    }
}

double NetworkModel::compute_propensity(int rxn_index, const double *conc) {
    if (rxn_index < 0 || rxn_index >= static_cast<int>(impl_->shared->reactions.size())) {
        return 0.0;
    }
    return compute_rxn_rate(impl_->shared->reactions[rxn_index], impl_->parameters, conc,
                            n_species(), impl_->species, true);
}

std::pair<std::string, int> NetworkModel::emit_ssa_propensity_source_structure() const {
    // GH #190 — STRUCTURE-specialized propensity vector: rate constants are read
    // from a runtime parameter array `p[]` instead of baked as literals; only the
    // structural factors (fixed by topology) are baked. Because no parameter VALUE
    // appears in the source, the emitted text — and therefore the .so cache key —
    // depends only on the model STRUCTURE: one compile per model, reused across
    // every parameter point (a fit) and replicate (an ensemble), with no
    // per-point recompile and no value-keyed cache explosion. Signature:
    //   void bngsim_ssa_propensities(const double* x, const double* p, double* a)
    //
    // Covered rate laws (each emitted to match compute_rxn_rate's SSA path
    // bit-for-bit, so the codegen realization equals the interpreted one):
    //   • Elementary mass-action  K·∏∏(x−m),  K = p[k]·stat·ssa_volume_factor
    //   • MichaelisMenten (tQSSA)  kcat·stat·sFree·E/(Km+sFree)
    // Functional / MM-without-two-params / live-variable-volume reactions emit
    // a[r]=0.0 and increment n_unsupported (the caller trusts the kernel only when
    // n_unsupported == 0). Functional covers BNG Sat/Hill (rewritten to Functional
    // expressions over observables) — not codegen'd here; those stay interpreted.
    const auto &reactions = impl_->shared->reactions;
    const auto &params = impl_->parameters;
    const auto &species = impl_->species;
    const int nr = static_cast<int>(reactions.size());
    const int ns = static_cast<int>(species.size());

    auto lit = [](double v) {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return std::string(buf);
    };

    std::string body;
    body.reserve(static_cast<size_t>(nr) * 52 + 64);
    int n_unsupported = 0;
    bool needs_sqrt = false;
    for (int r = 0; r < nr; ++r) {
        const Reaction &rxn = reactions[r];
        const bool elem_ok = rxn.rate_law_type == RateLawType::Elementary &&
                             rxn.apply_species_factor && rxn.rate_param_idx0 >= 0 &&
                             rxn.rate_param_idx0 < static_cast<int>(params.size()) &&
                             rxn.ssa_live_volume_idx0 < 0 && rxn.ssa_live_volume_terms.empty();
        if (elem_ok) {
            // Runtime rate constant × baked structural factor (omitted when 1.0).
            const double C = rxn.stat_factor * rxn.ssa_volume_factor;
            body += "  a[" + std::to_string(r) + "] = p[" + std::to_string(rxn.rate_param_idx0) +
                    "]";
            if (C != 1.0)
                body += " * " + lit(C);
            for (const auto &mult : rxn.reactant_multiplicities) {
                const int si = mult.first;
                const int count = mult.second;
                const double vf = species[si].amount_valued ? species[si].volume_factor : 1.0;
                const std::string n =
                    vf != 1.0 ? ("(" + lit(vf) + " * x[" + std::to_string(si) + "])")
                              : ("x[" + std::to_string(si) + "]");
                for (int m = 0; m < count; ++m) {
                    if (m == 0) {
                        body += " * " + n;
                    } else {
                        body += " * (" + n + " - " + lit(static_cast<double>(m)) + ")";
                    }
                }
            }
            body += ";\n";
            continue;
        }

        // MichaelisMenten tQSSA — mirrors compute_rxn_rate's SSA MM branch exactly
        // (raw conc for E/S, no ssa_volume_factor): kcat=p[k0], Km=p[km0],
        // E=x[e0], S=x[s0]; sFree = max(0, 0.5·((S−Km−E) + √((S−Km−E)²+4·Km·S))).
        const bool mm_shape = rxn.rate_law_type == RateLawType::MichaelisMenten &&
                              rxn.rate_law_param_indices.size() >= 2 &&
                              rxn.reactant_indices.size() >= 2;
        const int kcat0 = mm_shape ? rxn.rate_law_param_indices[0] - 1 : -1;
        const int km0 = mm_shape ? rxn.rate_law_param_indices[1] - 1 : -1;
        const int e0 = mm_shape ? rxn.reactant_indices[0] - 1 : -1;
        const int s0 = mm_shape ? rxn.reactant_indices[1] - 1 : -1;
        const bool mm_ok = mm_shape && kcat0 >= 0 && kcat0 < static_cast<int>(params.size()) &&
                           km0 >= 0 && km0 < static_cast<int>(params.size()) && e0 >= 0 &&
                           e0 < ns && s0 >= 0 && s0 < ns;
        if (mm_ok) {
            needs_sqrt = true;
            body += "  { double Km = p[" + std::to_string(km0) + "]; double E = x[" +
                    std::to_string(e0) + "]; double S = x[" + std::to_string(s0) + "];\n";
            body += "    double d = S - Km - E;\n";
            body += "    double sf = 0.5 * (d + sqrt(d * d + 4.0 * Km * S));\n";
            body += "    if (sf < 0.0) sf = 0.0;\n";
            body += "    a[" + std::to_string(r) + "] = p[" + std::to_string(kcat0) + "]";
            if (rxn.stat_factor != 1.0)
                body += " * " + lit(rxn.stat_factor);
            body += " * sf * E / (Km + sf); }\n";
            continue;
        }

        body += "  a[" + std::to_string(r) +
                "] = 0.0; /* unsupported: Functional / non-mass-action / live-varvol */\n";
        ++n_unsupported;
    }

    std::string src;
    src.reserve(body.size() + 256);
    // Portable export decoration so the cc-compiled propensity library exports
    // bngsim_ssa_propensities on Windows. An MSVC/MinGW DLL exports nothing
    // unless the symbol is tagged __declspec(dllexport), so the by-name lookup in
    // ssa_simulator.cpp / _bngsim_core.cpp (DynamicLibrary::symbol ->
    // GetProcAddress) would fail to resolve it and the run would silently fall
    // back to the interpreted propensity path — the same class of bug the Python
    // codegen prelude's BNGSIM_EXPORT fixed for the RHS/Jacobian entry points
    // (lanl/bngsim #5, this is #6). Unix ELF/Mach-O export global symbols by
    // default, so the macro expands to nothing there and the source stays
    // functionally identical; it is likewise a no-op on the in-process MIR/cc
    // JIT paths, which resolve the symbol from the JIT rather than from a DLL.
    src += "#if defined(_WIN32)\n";
    src += "#define BNGSIM_EXPORT __declspec(dllexport)\n";
    src += "#else\n";
    src += "#define BNGSIM_EXPORT\n";
    src += "#endif\n";
    // sqrt declaration only when an MM reaction needs it, so pure mass-action
    // sources stay byte-identical (cc accepts the extern; MIR strips #include but
    // keeps this decl and resolves sqrt via its libm import resolver).
    if (needs_sqrt)
        src += "extern double sqrt(double);\n";
    src += "BNGSIM_EXPORT void bngsim_ssa_propensities(const double* x, const double* p, double* "
           "a) {\n";
    src += body;
    src += "}\n";
    return {src, n_unsupported};
}

// ─── System time ─────────────────────────────────────────────────────────────

double NetworkModel::current_time() const { return impl_->current_time; }
void NetworkModel::set_current_time(double t) { impl_->current_time = t; }

// ─── Expression evaluator access ─────────────────────────────────────────────

ExpressionEvaluator &NetworkModel::evaluator() { return *impl_->evaluator; }

// ─── Table functions ──────────────────────────────────────────────────────────

// Canonicalization helpers (`is_time_index`, `strip_paren_suffix`) come from
// include/bngsim/table_function.hpp so the .tfun header reader and this
// runtime index resolver agree on case-insensitivity and trailing "()"
// stripping.

void NetworkModel::register_table_function_(TableFunction &tf) {
    // Resolve the index pointer based on the index name. The name is
    // normalized BNG-style (case-insensitive `time`/`t`; trailing "()"
    // tolerated on any index) so users can write `Time`, `myParam()`, etc.
    const auto &idx_name = tf.index_name();

    if (is_time_index(idx_name)) {
        // Time-indexed: point to model's current_time
        tf.set_index_ptr(&impl_->current_time);
    } else {
        const std::string lookup = strip_paren_suffix(idx_name);
        // Try parameter
        auto pit = impl_->shared->param_name_to_idx.find(lookup);
        if (pit != impl_->shared->param_name_to_idx.end()) {
            tf.set_index_ptr(&impl_->parameters[pit->second].value);
        } else {
            // Try observable
            auto oit = impl_->shared->observable_name_to_idx.find(lookup);
            if (oit != impl_->shared->observable_name_to_idx.end()) {
                tf.set_index_ptr(&impl_->observables[oit->second].total);
            } else {
                throw std::runtime_error("TableFunction '" + tf.name() + "': index variable '" +
                                         idx_name +
                                         "' not found as time, parameter, or observable");
            }
        }
    }

    // Register as a zero-arg ExprTk function under an internal name to avoid
    // collision with the synthetic parameter variable of the same name.
    // The expression evaluator will reference "tfun_NAME()" in compiled exprs.
    std::string internal_name = "tfun_" + tf.name();
    TableFunction *tf_ptr = &tf;
    impl_->evaluator->define_function(
        internal_name,
        ExpressionEvaluator::Func0([tf_ptr]() -> double { return tf_ptr->evaluate(); }));

    // Mark model as having functions (table functions participate in
    // evaluate_functions → rate computation)
    impl_->has_functions = true;
}

void NetworkModel::add_table_function(const std::string &name, const std::string &filepath,
                                      const std::string &index_name, const std::string &method,
                                      const std::string &header_name) {
    InterpolationMethod interpolation = parse_interpolation_method(method, name);
    auto tf = std::make_unique<TableFunction>(
        TableFunction::from_file(name, filepath, index_name, interpolation, header_name));
    register_table_function_(*tf);
    impl_->table_functions.push_back(std::move(tf));
}

void NetworkModel::add_table_function(const std::string &name, const std::vector<double> &xs,
                                      const std::vector<double> &ys, const std::string &index_name,
                                      const std::string &method) {
    InterpolationMethod interpolation = parse_interpolation_method(method, name);
    auto tf = std::make_unique<TableFunction>(name, xs, ys, index_name, interpolation);
    register_table_function_(*tf);
    impl_->table_functions.push_back(std::move(tf));
}

int NetworkModel::n_table_functions() const {
    return static_cast<int>(impl_->table_functions.size());
}

std::vector<std::string> NetworkModel::table_function_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->table_functions.size());
    for (const auto &tf : impl_->table_functions) {
        names.push_back(tf->name());
    }
    return names;
}

double NetworkModel::evaluate_table_function_at(int tf_id, double x) const {
    if (tf_id < 0 || tf_id >= static_cast<int>(impl_->table_functions.size())) {
        throw std::out_of_range("evaluate_table_function_at: tf_id " + std::to_string(tf_id) +
                                " out of range [0, " +
                                std::to_string(impl_->table_functions.size()) + ")");
    }
    return impl_->table_functions[tf_id]->evaluate_at(x);
}

std::vector<TableFunctionSpec> NetworkModel::table_function_specs() const {
    std::vector<TableFunctionSpec> specs;
    specs.reserve(impl_->table_functions.size());
    const auto &param_map = impl_->shared->param_name_to_idx;
    const auto &obs_map = impl_->shared->observable_name_to_idx;
    for (const auto &tf : impl_->table_functions) {
        TableFunctionSpec spec;
        spec.name = tf->name();
        const auto &idx_name = tf->index_name();
        const std::string lookup = strip_paren_suffix(idx_name);
        if (is_time_index(idx_name)) {
            spec.index_kind = "time";
        } else if (auto pit = param_map.find(lookup); pit != param_map.end()) {
            spec.index_kind = "parameter";
            spec.index_param_idx = pit->second;
        } else if (auto oit = obs_map.find(lookup); oit != obs_map.end()) {
            spec.index_kind = "observable";
            spec.index_obs_idx = oit->second;
        } else {
            // Should be unreachable: register_table_function_ rejects an
            // unresolved index_name, so by the time table_function_specs()
            // can be called the binding has already passed that gate.
            spec.index_kind = "unknown";
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

} // namespace bngsim
