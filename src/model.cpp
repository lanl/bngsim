// bngsim/src/model.cpp — Instance-based network model implementation
//
// Core RHS evaluation: update_observables → evaluate_functions → compute rates.
// Ported from BNG's derivs_network() (network.cpp) to instance-based design.

#include "bngsim/c_locale_numbers.hpp"
#include "bngsim/fd_jacobian.hpp"
#include "bngsim/functional_jac_scatter.hpp"
#include "bngsim/mm_jacobian.hpp"
#include "bngsim/net_file_loader.hpp"
#include "bngsim/table_function.hpp"
#include "model_impl.hpp" // defines NetworkModel::Impl

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
    // Deep copy. The clone's evaluator shares NOTHING with this one (issue
    // #257): clone_empty() hands back an evaluator with its own parser, symbol
    // table and expression list, and the loop below rebuilds it from the
    // pre-processed expression cache. compile_preprocessed() skips the
    // logical-operator and underscore remapping that compile() does — the
    // cached strings are already preprocessed — so the clone still avoids the
    // string-preprocessing pass, which is the part of the saving that was real.
    //
    // The evaluator this constructs and then throws away one line below is
    // deliberate leverage, not waste: because the parser is built lazily on
    // first compile, that discarded evaluator never builds one, so the clone
    // pays for exactly one parser — the same count it paid when the parser was
    // shared and the default-constructed one was discarded fully built.
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
    copy.impl_->volume_laws = impl_->volume_laws; // issue #758: found once, for every clone
    copy.impl_->volume_laws_at = impl_->volume_laws_at;
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
    // Likewise the IC baseline's own dx/dθ (GH #81): the clone copies
    // `species[].initial_conc` above, so it must copy that baseline's derivative
    // too or the clone's reset() would forget it.
    copy.impl_->baseline_sens_seed = impl_->baseline_sens_seed;
    copy.impl_->baseline_sens_seed_param_names = impl_->baseline_sens_seed_param_names;
    // ...and the event state that goes with the copied state and clock (issue
    // #693).
    copy.impl_->event_carry = impl_->event_carry;
    copy.impl_->event_carry_history = impl_->event_carry_history;
    copy.impl_->event_carry_evicted_t = impl_->event_carry_evicted_t;
    // ...and whether that baseline is a saved state rather than the declared IC
    // (issue #79), or the clone's set_param() would re-resolve a parameter-named
    // IC over a baseline the original had already retired.
    copy.impl_->ic_baseline_saved = impl_->ic_baseline_saved;

    // Create evaluator that shares the parser with the original
    copy.impl_->evaluator = impl_->evaluator->clone_empty();

    // Bind time() to the COPY's current_time. One symbol: `t` is left free
    // as an ordinary model identifier (src/expression.cpp), never bound here.
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
    // model_builder.cpp). A name already registered as a parameter or
    // observable, or one that is no symbol (a .net species `A()`), is skipped.
    for (auto &sp : copy.impl_->species)
        copy.impl_->evaluator->try_define_variable(sp.name, &sp.concentration);

    // Re-bind the rateOf accessors (GH #106) BEFORE any expression that may
    // reference a rate_of__<species> token is re-compiled. The copy gets its own
    // current_derivs (sized to n_species, zeroed) and rateof_scratch, and each
    // rate_of__<species> variable is bound to the COPY's buffer — mirrors the
    // time-pointer rebinding above and the build()-time registration.
    copy.impl_->uses_rateof = impl_->uses_rateof;
    // Issue #654 — a syntactic property of the function expressions, which the
    // clone re-compiles verbatim, so it carries over as plain data, for a
    // function that names `time` and one that calls a table read on the clock
    // alike (register_table_function_ below does not set it).
    copy.impl_->functions_use_time = impl_->functions_use_time;
    if (copy.impl_->uses_rateof) {
        // What reads which rateOf is the model's structure, and a clone's is
        // this one's: asked here once and copied, where each clone asked for
        // itself (every rate law parsed again, 80 us a reaction, on the first
        // run of each of a batch's entries).
        ensure_rateof_reads_();
        copy.impl_->derivative_rateof_reads = impl_->derivative_rateof_reads;
        copy.impl_->derivative_rateof_reads_state = impl_->derivative_rateof_reads_state;
        copy.impl_->rateof_read_elsewhere = impl_->rateof_read_elsewhere;
        copy.impl_->rateof_nesting = impl_->rateof_nesting;
        copy.impl_->rateof_flow_nesting = impl_->rateof_flow_nesting;
        copy.impl_->rateof_ring = impl_->rateof_ring;
        copy.impl_->rateof_reads_ready.store(true, std::memory_order_release);
        copy.impl_->current_derivs.assign(copy.impl_->species.size(), 0.0);
        copy.impl_->rateof_scratch.assign(copy.impl_->species.size(), 0.0);
        register_rateof_accessors(*copy.impl_->evaluator, copy.impl_->species,
                                  copy.impl_->current_derivs);
    }

    // Re-compile expression-valued parameters using cached preprocessed strings.
    //
    // Issue #188 — the gate is "has an expression", not "is currently attached to
    // it". A parameter whose expression `set_param` has overridden keeps its
    // evaluator precisely so the override stays reversible, and a clone that
    // dropped it would be the one copy of the model that could never be put back.
    // Only an attached parameter takes its value from the expression; an
    // overridden one carries the literal it was written with.
    //
    // Issue #626 — two passes, not one. The recompile is about evaluator ids and
    // walks declaration order; the re-derivation is about values and must walk
    // `derived_param_order` like every other re-evaluation pass (issue #568),
    // or a dependent declared before the parameter it reads takes a stale link
    // whenever the source is not at its expression fixed point (a probe's held
    // window, or rate_of__ inputs the clone has just zeroed). The set is the
    // same: after build() an evaluator is only ever dropped, never attached, so
    // every parameter recompiled here is in `derived_param_order`, and
    // refresh_derived_params() applies the same is_expression guard.
    for (auto &p : copy.impl_->parameters) {
        if (!p.expression.empty() && p.evaluator_id >= 0) {
            const auto &cached = impl_->evaluator->preprocessed_expr(p.evaluator_id);
            p.evaluator_id = copy.impl_->evaluator->compile_preprocessed(cached);
        }
    }
    copy.refresh_derived_params();

    // Re-create table functions in the copy (deep copy with fresh bindings)
    // (A copy of each table, so whether the model reads it comes along;
    // register_table_function_ binds its index to the copy's slot.)
    for (const auto &tf_ptr : impl_->table_functions) {
        auto new_tf = std::make_unique<TableFunction>(*tf_ptr);
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
    // The last attach's verdict (issue #534) is plain data: the clone reports
    // the same decline its parent does, as Model.clone() carries the status.
    copy.impl_->functional_jac_decline = impl_->functional_jac_decline;

    // Deep-copy events and re-compile their expressions in the new evaluator.
    // Each event carries indices into the evaluator's expression table for
    // its trigger, optional delay, optional priority, per-assignment value
    // expressions, and a resize rescale's compartment size (issue #936).
    // These indices are evaluator-specific; without
    // re-compilation in the cloned evaluator, the cloned model's events
    // would be silent no-ops (DSMTS event cases 00028/29/32/33 hit this
    // pre-Phase-5b but the SSA simulator did not consult events at all,
    // so the bug was latent).
    // The issue #144 trigger-residual cache is deliberately NOT copied: its
    // expression ids index the *source* evaluator's table and mean something
    // else in the clone's. Leaving it at the unresolved sentinel makes the
    // clone re-derive each residual from its own recompiled trigger, which is
    // the same text and therefore the same answer. The issue #150 state-switch
    // cache is left empty for the identical reason — it is keyed by condition
    // text, so the clone re-resolves the same conditions into its own evaluator.
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
        // ...and a resize rescale's compartment size (issue #936). Left as the
        // source's index, every clone read some other expression as the size.
        for (auto &idx : ev.assignment_rescale_size_expr) {
            if (idx >= 0) {
                const auto &cached = impl_->evaluator->preprocessed_expr(idx);
                idx = copy.impl_->evaluator->compile_preprocessed(cached);
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

void NetworkModel::set_param(const std::string &name, double value, bool force_override) {
    auto it = impl_->shared->param_name_to_idx.find(name);
    if (it == impl_->shared->param_name_to_idx.end()) {
        throw std::runtime_error("Parameter not found: " + name);
    }
    auto &param = impl_->parameters[it->second];

    // Issue #164 / #170 — an SBML compartment size is TWO representations: the
    // `p[]` a kinetic law reads, which a write has always moved, and the *storage
    // convention* (bngsim stores `amount/V_c`), which used to be folded at load
    // into constants nothing re-derived. #164 refused the write outright because
    // which halves landed was not even uniform inside one model. #170 put the
    // volume back into every one of those folds — the mass-action scalar carries
    // it as a live ratio on the rate parameter, the SSA propensity and the
    // per-species Jacobian divide read it from `params`, the Functional storage
    // divide is emitted against the compartment symbol, and
    // `refresh_compartment_volume_state` below re-derives `volume_factor` and an
    // amount-declared IC — so the write is honored here.
    //
    // What survives is a per-compartment residue the loader could not resolve,
    // flagged at load and named in its own message rather than covered by a
    // blanket refusal. Writing the value it already holds stays legal even then:
    // `set_params(dict(zip(param_names, vec)))` round-trips a full parameter
    // vector through here, and a write that changes nothing has nothing to desync.
    // A synthesized slot is not a knob (`primary_param_names` excludes it), so
    // reaching one at all is a mistake worth naming. An unchanged write stays
    // legal, for the same round-trip reason as below. Two kinds, and the message
    // says which — they are wrong for different reasons:
    //
    // Issue #170 — `_V0_<comp>` records the compartment size *as it was at load*
    // so the rate parameter's `(C / _V0_C)` ratio is exactly 1.0 there. Moving it
    // would rescale every rate in that compartment while the volume itself stayed
    // put, and would leave the next write to `C` computing its ratio against the
    // wrong baseline.
    //
    // Issue #227 — a function's backing slot holds whatever
    // `evaluate_functions()` last wrote there, and it writes again before every
    // RHS evaluation. So the write used to be *accepted*: `get_param` echoed the
    // new value back, and the trajectory did not move. That is the worst shape a
    // no-op can take, and it is why a `jax.grad` over `primary_param_names` used
    // to spend a coordinate returning exactly 0.0 forever.
    if (param.is_internal && value != param.value) {
        if (impl_->shared->function_name_to_idx.count(param.name) != 0) {
            throw CompartmentSizeWriteError(
                "Cannot set '" + param.name +
                "': a function of that name owns this slot — bngsim stores the function's "
                "evaluated value here and recomputes it from the function's own expression "
                "before every derivative evaluation, so a write here is discarded at the next "
                "one and the trajectory does not move (issue #227). If '" +
                param.name +
                "' also appears in the parameters block, that row is the seed this slot holds "
                "until the function first evaluates, not a knob (issue #266). "
                "Set the parameters the function's expression reads instead; "
                "Model.primary_param_names omits this name for that reason.");
        }
        throw CompartmentSizeWriteError(
            "Cannot set '" + param.name +
            "': it is not a parameter of the model but bngsim's record of a compartment's "
            "size at load time, which the reaction rate constants in that compartment are "
            "normalised against (issue #170). Writing it would rescale those rates without "
            "moving the volume. Set the compartment size itself instead; "
            "Model.primary_param_names omits this one for that reason.");
    }

    if (param.volume_write_refused && value != param.value) {
        std::ostringstream cur;
        cur << param.value;
        throw CompartmentSizeWriteError(
            "Cannot set '" + param.name + "': it is an SBML compartment size (currently " +
            cur.str() +
            ") this model cannot resolve to a live volume. One of three reasons, all "
            "decided at load: an assignment rule recomputes its size every step, so a "
            "write would not survive the next evaluation; a reaction "
            "divides two compartments' species by it as a single scalar, which is exact "
            "only while those compartments have equal size; or an <initialAssignment> "
            "folds it into a quantity no parameter holds — a species amount, a reaction "
            "rate — and SBML evaluates that expression once, at load. "
            "Load the model at the size you want instead: Model.from_sbml(path, "
            "compartment_sizes={'" +
            param.name +
            "': <value>}). Model.unwritable_compartment_size_params lists these; every "
            "other compartment size in this model is an ordinary writable parameter.");
    }

    // Issue #188 — a parameter that has a defining expression is expression-backed
    // exactly while it holds the value that expression produces.
    //
    // BNG's setParameter() semantics still hold, and a value-CHANGING write still
    // implements them: `d = d__FREE` written to 27 stops tracking `d__FREE` for
    // the remainder of the action sequence. What changed is that the override is
    // keyed on the *value* rather than latched by the act of writing, so it is
    // reversible — writing back the value the expression produces re-attaches it.
    //
    // Latching it broke two things. A write of the value the parameter already
    // held detached it, so `set_params(dict(zip(param_names, vec)))` — the whole
    // parameter-vector round trip a fitting harness performs every iteration, and
    // the one `Result.gradient`'s own docstring recommends — quietly turned every
    // derived parameter in the model into an independent constant (9,524 of them
    // across the 279 corpus models that have any). And detaching is not
    // bookkeeping: a derived parameter that stops being derived stops
    // contributing its chain-rule term to the primaries underneath it, so on
    // BIOMD0000000701 that identity round trip took d(x)/d(alpha) from 1.0e-3 to
    // exactly zero — a silently wrong sensitivity column for a parameter the
    // caller never wrote. Neither reset() nor clone() could put it back, and nor
    // could writing the primary underneath.
    //
    // The comparison is exact equality, and it is exact by construction rather
    // than by luck: the loop below is the only writer of an attached derived
    // parameter's value, so such a parameter always holds precisely
    // `evaluate(expression)` — verified bit-for-bit on all 9,524.
    //
    // `evaluator_id`, not `expression`, is what says "there is an expression
    // here": the .net reader stores the source text of every parameter,
    // constants included (`kf 0.5` keeps "0.5"), and only an expression-valued
    // one is ever compiled.
    //
    // `force_override` is the escape hatch for a caller whose whole contract is
    // "this parameter is an independent input" — see the header. It pins
    // regardless of the value, which is what the unconditional detach used to do
    // for every caller.
    const bool was_expression = param.is_expression;
    const int was_evaluator_id = param.evaluator_id;
    double from_expr = 0.0;
    bool expr_live = false;
    if (!force_override && param.evaluator_id >= 0 && !param.expression.empty()) {
        try {
            from_expr = impl_->evaluator->evaluate(param.evaluator_id);
            expr_live = true;
        } catch (...) {
            expr_live = false;
        }
    }

    param.value = value;

    if (expr_live) {
        param.is_expression = (value == from_expr);
    } else if (force_override ? param.evaluator_id >= 0 : param.is_expression) {
        // Two ways here, and both want the pre-#188 latch — detach, permanently.
        //
        // `force_override`: the caller asked for a pin that no later write can
        // lift by accident, which is the legacy semantics exactly. It keys on the
        // evaluator, not on is_expression (issue #769): a parameter an ordinary
        // write already overrode has is_expression == false but still holds its
        // evaluator, which is what makes that override reversible. Testing
        // is_expression here skipped the detach for exactly that parameter, so
        // a later write equal to the expression's value re-attached a "pinned"
        // parameter and the next primary write moved it.
        //
        // Or an expression that will not evaluate — compilation failed at build
        // time (model_builder swallows that case), or it reads something
        // unavailable here — which has nothing to compare against.
        //
        // Dropping the evaluator is what makes this state distinguishable from
        // the reattachable override above.
        param.is_expression = false;
        param.evaluator_id = -1;
        // Keep param.expression for debugging/introspection
    }

    // expression_support() follows an is_expression parameter to whatever its
    // expression reads, so flipping the flag either way changes the support of
    // every expression that reads this parameter. Since #188 made an override
    // reversible, a re-attach GROWS the support again, and a memo entry filled
    // while detached would keep the primaries out, so an event jump writes
    // dh/dp = 0 for them (issue #773). Drop the whole memo: the walk is
    // transitive, so the affected entries are not just this parameter's own.
    if (param.is_expression != was_expression || param.evaluator_id != was_evaluator_id)
        impl_->expression_support_cache.clear();

    // Re-evaluate remaining expression-valued parameters (e.g., "a = a__FREE"),
    // so derived parameters pick up the new value — the whole chain of them,
    // however deep and in whatever order they were declared (issue #568).
    refresh_derived_params();

    // ...the storage convention that a compartment size decides (issue #170),
    // which must also come after the re-evaluation above: an amount-declared IC
    // reads the volume, and the volume itself can be the target of an
    // initialAssignment carried by a derived parameter.
    if (param.is_compartment_size)
        refresh_compartment_volume_state();

    // ...and the species initial conditions those parameters name (issue #79).
    // Must come AFTER the refresh above: a species IC may name a *derived*
    // parameter (`R() Rtot` with `Rtot = 2*R0`), whose value only just moved.
    refresh_param_ref_ics();
}

// Re-derive every expression-valued parameter, in dependency order (issue
// #568). See the header for why the order is fixed at build() and what
// `skip_param_idx` is for.
//
// `derived_param_order` lists exactly the parameters build() left with an
// evaluator, so the guard here is what each of them is still *attached* to its
// expression: issue #188's value-keyed override keeps the evaluator so the
// override stays reversible, and re-deriving an overridden parameter would
// discard the caller's write on the next set_param of anything at all.
const std::vector<int> &NetworkModel::derived_param_order() const {
    return impl_->shared->derived_param_order;
}

void NetworkModel::refresh_derived_params(int skip_param_idx) {
    for (int pi : impl_->shared->derived_param_order) {
        if (pi == skip_param_idx)
            continue;
        auto &p = impl_->parameters[pi];
        if (p.is_expression && p.evaluator_id >= 0) {
            p.value = impl_->evaluator->evaluate(p.evaluator_id);
        }
    }
}

// Re-resolve every species initial condition that a parameter names, from the
// current parameter values (issue #79).
//
// `A() Stot` in the .net species block — or an SBML initialAssignment that is a
// bare <ci> — says A's initial condition IS that parameter, so a later
// set_param("Stot", …) has to move it. Without this the common dose scan over a
// total amount silently ran every dose at the load-time value: get_param
// confirmed the write, get_state() did not move, and nothing warned.
//
// Two fields, two different rules:
//
//   * `initial_conc` — the declared IC — always follows the parameter. That is
//     what makes reset() rebuild from current parameter values (issue #79's
//     other half) and what makes the `set_params(); reset()` sequence in
//     run_batch() / steady_state_batch() correct without their knowing anything
//     about IC parameters.
//   * `concentration` — the live state — follows only while the species is
//     still sitting ON that baseline. A species the dynamics advanced (or a
//     caller assigned) holds a value this parameter no longer describes, and
//     overwriting it would discard a pre-equilibration mid-protocol. The next
//     reset() picks the new IC up regardless. The same
//     `concentration == initial_conc` test decides whether a parameter still
//     reaches a species IC in the #113 sensitivity seeding, so the two agree.
//
// `concentration == initial_conc` alone is a VALUE test standing in for a
// PROVENANCE question, and the two part company when a run leaves a species
// exactly where it started (issue #141). `kinetics_mb1n.bngl` does that: free
// antigen is not consumed while its `m()` switch is 0, so its carried value is
// bit-identical to the declared IC, the value test reads "still at baseline",
// and the next setParameter() zeroes an integrated pool mid-protocol. So the
// live state additionally requires that no run has advanced it —
// `ic_state_dirty`, GH #210's carry-over marker, which write_final_state_back
// sets and reset() clears.
//
// Read, never written, here: setting it from set_param is what issue #79
// proposed and it is wrong — it makes the next forward-sensitivity run raise
// instead of rebuilding anything.
//
// This deliberately does NOT extend to a caller who assigns a species the same
// number its own declared IC produces (set_concentration/set_state on a model
// that has not run — neither marks the state dirty). There the assignment and
// the expression agree numerically and which was meant is genuinely ambiguous;
// #113's `_superseded_ic_rows` documents that same case as ambiguous and keeps
// the row, with declare_ic_sensitivity (#111) as the way to say either. Those
// two sites still agree. An advanced run is not ambiguous, which is why only it
// is carved out.
//
// Every ref is re-resolved, not just the refs naming the written parameter:
// after the derived-parameter re-evaluation above, ANY ref may have moved, and
// re-resolving one that did not is a no-op. Empty for all but a handful of
// models — the loop is over species_ic_param_refs, not species.
void NetworkModel::refresh_param_ref_ics() {
    // save_concentrations() redefined the baseline to the current state; the
    // declared IC no longer describes it (see Impl::ic_baseline_saved).
    if (impl_->ic_baseline_saved)
        return;
    for (const auto &ref : impl_->shared->species_ic_param_refs) {
        auto &sp = impl_->species[static_cast<std::size_t>(ref.first)];
        const double val = resolve_ic_from_param(
            sp, impl_->parameters[static_cast<std::size_t>(ref.second)].value);
        const bool at_baseline = (sp.concentration == sp.initial_conc) && !impl_->ic_state_dirty;
        sp.initial_conc = val;
        sp.declared_negative = val < 0.0; // issue #706
        if (at_baseline)
            sp.concentration = val;
    }
}

// Re-derive the storage convention from the compartment sizes (issue #170).
//
// A compartment volume plays two roles in a loaded model: a *symbol* in kinetic
// laws, which is an ordinary `p[]` a write already moves, and the *storage
// convention* — bngsim stores `amount/V_c`, so V_c decides the amount↔storage
// conversion (`volume_factor`) and, for an amount-declared species, the stored
// initial condition itself. Issue #164 shipped a refusal because that second role
// was folded at load and nothing re-derived it; this is what re-derives it.
//
// Both fields live on `Species`, which is per-instance state — a model clone gets
// its own copy — so writing them here cannot leak into a sibling model the way
// rescaling a shared `Reaction` would. That is why the reaction-side halves of
// the same fold (the mass-action scalar, the SSA propensity volume) are handled
// by reading a parameter instead of by rewriting a cached number.
//
// `initial_conc` follows the #79 discipline exactly, for the same reason: the
// declared IC always follows the volume (so `reset()` rebuilds from it), but the
// live `concentration` follows only while the species is still sitting ON that
// baseline and no run has advanced it — otherwise a mid-protocol volume change
// would discard an integrated state. The invariant here is the declared *amount*
// rather than a parameter value, which is why it is a separate loop and not a
// `species_ic_param_refs` entry: `amount/V` is not "this IC IS that parameter".
void NetworkModel::refresh_compartment_volume_state() {
    for (auto &sp : impl_->species) {
        if (sp.volume_param_idx0 < 0)
            continue;
        const double v = impl_->parameters[static_cast<std::size_t>(sp.volume_param_idx0)].value;
        sp.volume_factor = v;
        // The amount→storage divide is not an initial condition and is not
        // subject to the baseline rules — it is how every stored value in this
        // compartment is *read*, so it moves unconditionally.
        if (std::isnan(sp.initial_amount) || impl_->ic_baseline_saved)
            continue;
        const double ic = (v != 0.0) ? sp.initial_amount / v : sp.initial_amount;
        const bool at_baseline = (sp.concentration == sp.initial_conc) && !impl_->ic_state_dirty;
        sp.initial_conc = ic;
        sp.declared_negative = ic < 0.0; // issue #706
        if (at_baseline)
            sp.concentration = ic;
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
    // A fresh start for the events too (issue #693).
    clear_event_carry();
    // Back at the IC baseline. When that baseline is θ-independent (the literal
    // .net ICs) this is a fresh start: no carry-over, no pending seed (GH #210).
    // But save_concentrations() can have redefined the baseline to a
    // pre-equilibrated x_ss(θ), in which case the baseline carries its own
    // dx/dθ — restore that with it rather than silently reverting a
    // θ-dependent initial condition to fresh-start seeding (GH #81).
    if (impl_->baseline_sens_seed.empty()) {
        impl_->ic_state_dirty = false;
        clear_pending_sens_seed();
    } else {
        impl_->pending_sens_seed = impl_->baseline_sens_seed;
        impl_->pending_sens_seed_param_names = impl_->baseline_sens_seed_param_names;
        impl_->ic_state_dirty = true;
    }
}

void NetworkModel::save_concentrations() {
    for (auto &s : impl_->species) {
        s.initial_conc = s.concentration;
    }
    // The current state is now the baseline ICs, so a subsequent reset() returns
    // here. GH #81: the state itself did not change, so neither did its
    // θ-derivative — the new baseline INHERITS the carried dx/dθ (a BNG
    // pre-equilibrate → saveConcentrations → scan protocol otherwise loses it
    // and silently re-seeds a θ-dependent IC as a fresh start). Stash it as the
    // baseline's own derivative for reset(), and keep it pending so the next
    // phase can carry. With no carried derivative the baseline is θ-independent
    // literal ICs and this is the pre-#81 fresh start.
    impl_->baseline_sens_seed = impl_->pending_sens_seed;
    impl_->baseline_sens_seed_param_names = impl_->pending_sens_seed_param_names;
    if (impl_->baseline_sens_seed.empty()) {
        impl_->ic_state_dirty = false;
    }
    // The baseline is now this state, not the declared IC — so a later
    // set_param() must NOT re-resolve a parameter-named IC over it and discard
    // the equilibration it just captured (issue #79).
    impl_->ic_baseline_saved = true;
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

void NetworkModel::get_initial_state_into(double *out) const {
    const auto &sp = impl_->species;
    for (std::size_t i = 0; i < sp.size(); ++i)
        out[i] = sp[i].initial_conc;
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

bool NetworkModel::has_baseline_sens_seed() const { return !impl_->baseline_sens_seed.empty(); }

bool NetworkModel::ic_baseline_saved() const { return impl_->ic_baseline_saved; }

// ─── Event state carried between runs (issue #693) ───────────────────────────

const NetworkModel::EventCarry &NetworkModel::event_carry() const { return impl_->event_carry; }

const std::deque<NetworkModel::EventCarry> &NetworkModel::event_carry_history() const {
    return impl_->event_carry_history;
}

double NetworkModel::event_carry_evicted_t() const { return impl_->event_carry_evicted_t; }

namespace {
void check_event_carry(const NetworkModel::EventCarry &carry, const std::vector<Event> &evs) {
    if (!carry.valid)
        return;
    if (!std::isfinite(carry.t))
        throw std::invalid_argument("set_event_carry: the time is not finite");
    if (carry.trigger.size() != evs.size())
        throw std::invalid_argument("set_event_carry: " + std::to_string(carry.trigger.size()) +
                                    " trigger values for a model with " +
                                    std::to_string(evs.size()) + " events");
    for (const auto &pe : carry.pending) {
        if (pe.event_idx < 0 || pe.event_idx >= static_cast<int>(evs.size()))
            throw std::invalid_argument("set_event_carry: no event " +
                                        std::to_string(pe.event_idx));
        if (!std::isfinite(pe.apply_time))
            throw std::invalid_argument("set_event_carry: an apply time is not finite");
        const auto n = evs[static_cast<std::size_t>(pe.event_idx)].assignments.size();
        if (!pe.frozen_values.empty() && pe.frozen_values.size() != n)
            throw std::invalid_argument(
                "set_event_carry: " + std::to_string(pe.frozen_values.size()) +
                " frozen values for an event with " + std::to_string(n) + " assignments");
    }
}
} // namespace

void NetworkModel::set_event_carry(EventCarry carry) {
    check_event_carry(carry, impl_->events);
    impl_->event_carry = std::move(carry);
}

void NetworkModel::set_event_carry_state(EventCarry carry, std::deque<EventCarry> history,
                                         double evicted_t) {
    check_event_carry(carry, impl_->events);
    for (const auto &h : history)
        check_event_carry(h, impl_->events);
    impl_->event_carry = std::move(carry);
    impl_->event_carry_history = std::move(history);
    impl_->event_carry_evicted_t = evicted_t;
}

void NetworkModel::publish_event_carry(EventCarry carry, bool continued) {
    auto &h = impl_->event_carry_history;
    if (!continued) {
        h.clear();
        impl_->event_carry_evicted_t = -std::numeric_limits<double>::infinity();
    } else {
        // The run continued from where the trajectory stood at its start; the
        // leg ends after that belong to a branch it has left.
        const double t0 = impl_->event_carry.t;
        while (!h.empty() && h.back().t > t0)
            h.pop_back();
    }
    h.push_back(carry);
    if (h.size() > kEventCarryHistory) {
        impl_->event_carry_evicted_t = h.front().t;
        h.pop_front();
    }
    impl_->event_carry = std::move(carry);
}

void NetworkModel::clear_event_carry() {
    impl_->event_carry = EventCarry{};
    impl_->event_carry_history.clear();
    impl_->event_carry_evicted_t = -std::numeric_limits<double>::infinity();
}

int NetworkModel::rewind_event_carry(double t) {
    auto &h = impl_->event_carry_history;
    for (auto it = h.rbegin(); it != h.rend(); ++it)
        if (it->valid && it->t == t) {
            impl_->event_carry = *it;
            return 1;
        }
    return t <= impl_->event_carry_evicted_t ? -1 : 0;
}

const NetworkModel::EventCarry *NetworkModel::event_carry_for(double t_start, int n_events) const {
    const auto &c = impl_->event_carry;
    if (!c.valid || c.t != t_start || static_cast<int>(c.trigger.size()) != n_events)
        return nullptr;
    return &c;
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

// ─── Event-trigger residuals (issue #144) ────────────────────────────────────
//
// Reduce an event trigger to the residual whose sign change IS the crossing, so
// the solver can differentiate the crossing time. The trigger the evaluator
// holds is a *boolean*: `v > 30` evaluates to 1 or 0 and its derivative is a
// delta, which is why the CVODE root function (trigger − 0.5) can locate the
// crossing but not differentiate it. `lhs − rhs` is the same surface with a
// gradient on it.
//
// Only a single relational comparison qualifies. A conjunction, disjunction or
// negation does not: its true-set boundary is assembled from several surfaces
// and which one carries the rising edge can change as a parameter moves (the
// same reasoning that restricts issue #49's clock-threshold recognizer to
// conjunctions of half-lines, one step stricter because here there is no
// `time` ordering to pick the active atom by). All three are refused by name.
//
// `NetworkModel::state_switch` reduces a *rate-law* condition through the same
// splitter, and the two differ on exactly one operator; `ResidualUse` below is
// that difference.
namespace {

// What a residual is being built FOR. It decides one thing: whether an equality
// test names a surface the caller can use.
//
// An event trigger (issue #144) fires on a RISING EDGE, and `x == c` has none a
// root finder can straddle — a continuous trajectory satisfies it on a
// measure-zero set, so there is no false→true instant to locate, let alone to
// differentiate. Refused, as it always was.
//
// A rate-law switch (issue #150) asks a different question: WHERE CAN THIS
// BRANCH CHANGE. For `x == c` the answer is the surface `x − c = 0` — the same
// one `x < c` names, and the boundary of the equality's own true-set. Whether
// the branch really changes across it is not this function's business: it is
// measured at the root, by the branch-gap probe in
// `apply_state_switch_sensitivity_jump`, which returns with no jump when the two
// sides evaluate the same. A LONE equality is exactly that case (both sides take
// the `otherwise` branch), so admitting it costs a root and a zero jump.
//
// What it buys is the crossing being bracketed at all. MODEL2003190004 spells
// `APC <= 0.2` as `(APC == 0.2) or (APC < 0.2)`, an `<or/>` of `<eq/>` and
// `<lt/>` over one pair of operands (issue #381). The splitter judges the two
// atoms separately, so refusing the equality half declined the analytic
// sensitivity RHS for the WHOLE model — and CVODES' difference quotient, which
// then answers every column, stalled at that very crossing. Both atoms reduce to
// `(APC)-(0.2)`, so the run-time detector deduplicates them into the one root
// the `<` half already earned.
enum class ResidualUse {
    // Issue #144: the rising edge of an event trigger.
    EventRisingEdge,
    // Issue #150: the surface a rate-law branch can change across.
    SwitchSurface,
};

// Logical connectives ExprTk accepts as words. `&&`/`||` are rewritten to
// ` and `/` or ` by the evaluator's preprocessing before an expression is
// compiled, so the word forms are what a preprocessed trigger actually
// carries — but the single-character forms are legal ExprTk too and a `.net`
// or SBML source can spell them either way, so both are scanned for.
bool is_logical_word(const std::string &w) {
    static const std::unordered_set<std::string> kWords = {"and", "or",   "not",  "nand", "nor",
                                                           "xor", "xnor", "mand", "mor"};
    return kWords.count(w) != 0;
}

bool is_ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_'; }

// Strip parentheses that enclose the whole expression: SBML triggers routinely
// arrive as `(v>30)`, whose relational operator would otherwise never be seen
// at depth 0. Only strips a pair that really does span the string — `(a)>(b)`
// opens and closes before the end and is left alone.
std::string strip_enclosing_parens(std::string s) {
    auto trim = [](std::string &v) {
        const auto b = v.find_first_not_of(" \t\n\r");
        if (b == std::string::npos) {
            v.clear();
            return;
        }
        v = v.substr(b, v.find_last_not_of(" \t\n\r") - b + 1);
    };
    trim(s);
    while (s.size() >= 2 && s.front() == '(' && s.back() == ')') {
        int depth = 0;
        bool spans = true;
        for (std::size_t i = 0; i + 1 < s.size(); ++i) {
            if (s[i] == '(')
                ++depth;
            else if (s[i] == ')')
                --depth;
            if (depth == 0) {
                spans = false; // the opening paren closed before the end
                break;
            }
        }
        if (!spans) {
            break;
        }
        s = s.substr(1, s.size() - 2);
        trim(s);
    }
    return s;
}

// True when `s`, once any enclosing parentheses are stripped, IS itself a
// comparison — so it evaluates to 0 or 1 and a residual built from it is a
// difference of two BOOLEANS: a step, with no gradient to root on and none to
// differentiate `dt*/dθ` with. `(x>0) != (y>0)`, the `<xor/>` idiom, is the
// shape (issue #381).
//
// Deliberately top-level only. An operand that merely *contains* a comparison
// deeper down is usually a perfectly good smooth function — every corpus
// residual of the form `(... if(PO2AMB>80, 80, PO2AMB) ...) < 0` is one, and
// those are surfaces the trajectory really does cross. What is refused here is
// the operand whose own outermost operator is relational.
bool is_itself_a_comparison(const std::string &operand) {
    const std::string s = strip_enclosing_parens(operand);
    int depth = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '(') {
            ++depth;
            continue;
        }
        if (c == ')') {
            --depth;
            continue;
        }
        if (depth != 0) {
            continue;
        }
        if (is_ident_char(c)) {
            std::size_t j = i;
            while (j < s.size() && is_ident_char(s[j])) {
                ++j;
            }
            i = j - 1;
            continue;
        }
        // A bare `!` is negation; only `!=` is a comparison.
        if (c == '<' || c == '>' || c == '=') {
            return true;
        }
        if (c == '!' && i + 1 < s.size() && s[i + 1] == '=') {
            return true;
        }
    }
    return false;
}

// Build `(<lhs>)-(<rhs>)` from a preprocessed trigger, or return nullopt with
// `why` describing what defeated the split.
std::optional<std::string> trigger_residual_source(const std::string &trigger, std::string &why,
                                                   ResidualUse use) {
    const std::string s = strip_enclosing_parens(trigger);
    if (s.empty()) {
        why = "it is empty";
        return std::nullopt;
    }

    std::size_t op_pos = std::string::npos;
    std::size_t op_len = 0;
    int depth = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '(') {
            ++depth;
            continue;
        }
        if (c == ')') {
            --depth;
            continue;
        }
        if (depth != 0) {
            continue;
        }
        if (is_ident_char(c)) {
            // Consume the whole identifier so a logical word is recognized as a
            // word (and so `nor` inside `normalize` is not).
            std::size_t j = i;
            while (j < s.size() && is_ident_char(s[j])) {
                ++j;
            }
            const std::string word = s.substr(i, j - i);
            if (is_logical_word(word)) {
                why = "it combines conditions with '" + word +
                      "', and which of the combined surfaces carries the rising edge can "
                      "change as a parameter moves";
                return std::nullopt;
            }
            i = j - 1;
            continue;
        }
        if (c == '&' || c == '|') {
            why = std::string("it combines conditions with '") + c +
                  "', and which of the combined surfaces carries the rising edge can change "
                  "as a parameter moves";
            return std::nullopt;
        }
        if (c == '=' || c == '!') {
            // `==`, `!=` and ExprTk's single `=` equality.
            if (use == ResidualUse::EventRisingEdge) {
                why = "it is an equality test, which a continuous trajectory satisfies only on a "
                      "measure-zero set rather than crossing transversally";
                return std::nullopt;
            }
            // A bare `!` is negation, not a comparison. Unreachable from a
            // compiled expression (this build's ExprTk rejects `!` outright, and
            // the `not(` spelling is caught as a logical word above), but the
            // scan must not read the `!` of a negation as half an operator.
            if (c == '!' && (i + 1 >= s.size() || s[i + 1] != '=')) {
                why = "it negates a condition, so its true-set boundary is the negated "
                      "condition's and not a comparison of its own";
                return std::nullopt;
            }
            if (op_pos != std::string::npos) {
                why = "it chains more than one comparison, so its true-set boundary is "
                      "assembled from several surfaces";
                return std::nullopt;
            }
            op_pos = i;
            op_len = (i + 1 < s.size() && s[i + 1] == '=') ? 2 : 1;
            i += op_len - 1;
            continue;
        }
        if (c == '<' || c == '>') {
            if (op_pos != std::string::npos) {
                why = "it chains more than one comparison, so its true-set boundary is "
                      "assembled from several surfaces";
                return std::nullopt;
            }
            op_pos = i;
            op_len = (i + 1 < s.size() && s[i + 1] == '=') ? 2 : 1;
            i += op_len - 1;
        }
    }

    if (op_pos == std::string::npos) {
        why = "it is not a relational comparison, so it has no crossing surface to "
              "differentiate";
        return std::nullopt;
    }
    const std::string lhs = s.substr(0, op_pos);
    const std::string rhs = s.substr(op_pos + op_len);
    if (lhs.find_first_not_of(" \t\n\r") == std::string::npos ||
        rhs.find_first_not_of(" \t\n\r") == std::string::npos) {
        why = "one side of its comparison is empty";
        return std::nullopt;
    }
    if (is_itself_a_comparison(lhs) || is_itself_a_comparison(rhs)) {
        why = "one side of its comparison is itself a comparison, so the residual would be a "
              "difference of two booleans — a step with no gradient, rather than a surface "
              "with one";
        return std::nullopt;
    }
    // Orientation is not imposed: dt*/dp is a ratio of two derivatives of g, so
    // negating g leaves it unchanged.
    return "(" + lhs + ")-(" + rhs + ")";
}

} // namespace

int NetworkModel::event_trigger_residual_expr(int event_idx0, std::string *why) const {
    const auto &events = impl_->events;
    if (event_idx0 < 0 || event_idx0 >= static_cast<int>(events.size())) {
        if (why != nullptr) {
            *why = "no such event";
        }
        return -1;
    }
    auto &cache = impl_->event_trigger_residual_idx;
    if (cache.size() != events.size()) {
        cache.assign(events.size(), -2);
        impl_->event_trigger_residual_species.assign(events.size(), {});
        impl_->event_trigger_residual_reason.assign(events.size(), std::string());
    }
    if (cache[event_idx0] == -2) {
        cache[event_idx0] = -1;
        const Event &ev = events[event_idx0];
        std::string reason;
        if (ev.trigger_expr_idx < 0) {
            reason = "it has no trigger expression";
        } else {
            const std::string &pre = impl_->evaluator->preprocessed_expr(ev.trigger_expr_idx);
            const std::optional<std::string> src =
                trigger_residual_source(pre, reason, ResidualUse::EventRisingEdge);
            if (src) {
                try {
                    // Compiling the *preprocessed* halves keeps every identifier
                    // in the spelling the evaluator's symbol table already
                    // binds; running them back through compile() would remap
                    // names that are already remapped.
                    cache[event_idx0] = impl_->evaluator->compile_preprocessed(*src);
                } catch (const std::exception &e) {
                    reason = std::string("its residual '") + *src + "' did not compile (" +
                             e.what() + ")";
                }
            }
        }
        if (cache[event_idx0] >= 0) {
            expression_support(cache[event_idx0],
                               &impl_->event_trigger_residual_species[event_idx0], nullptr);
        } else {
            impl_->event_trigger_residual_reason[event_idx0] = reason;
        }
    }
    if (why != nullptr && cache[event_idx0] < 0) {
        *why = impl_->event_trigger_residual_reason[event_idx0];
    }
    return cache[event_idx0];
}

// THREAD SAFETY (issues #201, #257). This and event_trigger_residual_expr above are
// lazy memos: `const` methods that compile into `impl_->evaluator` on first use, so
// two threads calling them on ONE NetworkModel race. But do not read that as "the
// memos are the fragile part" — they are not, and treating them as the hazard
// understates it. An integration writes impl_->species concentrations and
// impl_->current_time straight back onto the model, so two threads integrating one
// NetworkModel corrupt each other's state long before either reaches a lazy compile.
// (Measured: sharing the model across run_batch's workers does not produce a subtly
// wrong answer, it kills CVODE with flag=-3 on the first step.)
//
// The invariant is therefore the blunt one: a NetworkModel is NOT thread-safe, and
// every fan-out must give each worker its own clone().
// python/tests/test_parallel_workers_clone_the_model.py enforces it at runtime over
// every parallel entry point, so a new fan-out that forgets to clone fails there
// rather than in a user's fit.
//
// What #201 got wrong, and #257 corrected, is the *other* half: that clones could
// "legitimately share" the ExprTk parser as long as compile() was serialized. They
// could not. A parser retains a strong handle on the symbol table of the last
// expression compiled through it, so sharing one made a clone's symbol table
// reachable — and droppable, through a non-atomic refcount — from whichever thread
// compiled next. Clones now share no ExprTk state at all; see the class comment on
// ExprTkEvaluator.
const NetworkModel::StateSwitch *NetworkModel::state_switch(const std::string &condition_src,
                                                            std::string *why) const {
    auto it = impl_->state_switch_cache.find(condition_src);
    if (it == impl_->state_switch_cache.end()) {
        StateSwitch sw;
        std::string reason;
        // Compile the condition itself first, only to obtain its *preprocessed*
        // text: the residual halves have to be spelled the way the evaluator's
        // symbol table already binds them, exactly as the trigger path does.
        // It doubles as the check that the condition compiles at all.
        int cond_idx = -1;
        try {
            cond_idx = impl_->evaluator->compile(condition_src);
        } catch (const std::exception &e) {
            reason = std::string("it did not compile (") + e.what() + ")";
        }
        if (cond_idx >= 0) {
            const std::string &pre = impl_->evaluator->preprocessed_expr(cond_idx);
            const std::optional<std::string> src =
                trigger_residual_source(pre, reason, ResidualUse::SwitchSurface);
            if (src) {
                try {
                    sw.residual_expr_idx = impl_->evaluator->compile_preprocessed(*src);
                    sw.residual_source = *src;
                } catch (const std::exception &e) {
                    reason = std::string("its residual '") + *src + "' did not compile (" +
                             e.what() + ")";
                }
            }
        }
        if (sw.residual_expr_idx >= 0) {
            // A condition over parameters and literals alone has a crossing at a
            // time the trajectory does not move — that is the issue #48 clock
            // switch (or no crossing at all), and running the saltation jump on
            // it would double-count what the clock path already applies.
            bool reads_state = false;
            for (const double *addr :
                 impl_->evaluator->referenced_variable_addresses(sw.residual_expr_idx)) {
                if (is_state_address(addr)) {
                    reads_state = true;
                    break;
                }
            }
            if (reads_state) {
                expression_support(sw.residual_expr_idx, &sw.species, nullptr);
            } else {
                sw.residual_expr_idx = -1;
                sw.residual_source.clear();
                reason = "it reads no live model state, so its crossing time does not move with "
                         "the trajectory";
            }
        }
        it = impl_->state_switch_cache
                 .emplace(condition_src, std::make_pair(std::move(sw), std::move(reason)))
                 .first;
    }
    if (it->second.first.residual_expr_idx < 0) {
        if (why != nullptr) {
            *why = it->second.second;
        }
        return nullptr;
    }
    return &it->second.first;
}

int NetworkModel::compiled_expression(const std::string &expression_src) const {
    auto it = impl_->compiled_expression_cache.find(expression_src);
    if (it == impl_->compiled_expression_cache.end()) {
        // A text that does not compile throws here and is not cached; compile()
        // appends nothing for it, so asking again costs a compile, not memory.
        it = impl_->compiled_expression_cache
                 .emplace(expression_src, impl_->evaluator->compile(expression_src))
                 .first;
    }
    return it->second;
}

const std::vector<int> &NetworkModel::event_trigger_residual_species(int event_idx0) const {
    static const std::vector<int> kEmpty;
    if (event_trigger_residual_expr(event_idx0) < 0) {
        return kEmpty;
    }
    return impl_->event_trigger_residual_species[event_idx0];
}

// ─── What a compiled expression can be differentiated with respect to ────────
//
// The support of a central finite difference over an expression: which species
// and which parameters can move it. Every finite difference the event-jump
// machinery takes — ∂g/∂x and ∂g/∂p of a trigger residual (issue #144), ∂h/∂x
// and ∂h/∂p of an event assignment (GH #212) — prunes its perturbations with
// this, so all four agree about what "reads" means.
//
// Naming the addresses an expression references is NOT enough, and the two ways
// it falls short are both silent-zero bugs rather than slow ones:
//
//   * ModelBuilder registers a species as an ExprTk variable only when its name
//     is free, and SBML models routinely give each species an observable of the
//     same name — so in `u + d` the token `u` binds to the *observable total*,
//     never to &sp.concentration. A difference restricted to concentration
//     addresses therefore perturbs nothing and reports ∂h/∂u = 0 on every SBML
//     model. (This is issue #52's shadowing, on the assignment side.)
//   * A derived parameter hides its primaries. `2*Ktot` with `Ktot = k1 + k2`
//     references Ktot's address and neither k1's nor k2's, so a requested k1
//     looks absent and its column comes back flat.
//
// So the walk follows both: an observable expands to the species behind it, and
// a parameter that is an expression — or that a model function writes, which is
// how an SBML assignment rule arrives — expands to whatever ITS body reads.
// Only a rateOf accessor is opaque: dx_i/dt can read the whole state and any
// parameter, so it widens both supports to everything (issue #764 — it used to
// widen only the species, and every parameter column of an event assignment
// reading rateOf was skipped as unsupported).
void NetworkModel::expression_reads_(const std::vector<std::vector<int>> &groups,
                                     std::vector<std::vector<int>> &species_out,
                                     std::vector<std::vector<int>> &rateof_out,
                                     std::vector<char> &known) const {
    const std::size_t n_groups = groups.size();
    species_out.assign(n_groups, {});
    rateof_out.assign(n_groups, {});
    known.assign(n_groups, 1);
    const auto &species = impl_->species;
    const int ns = static_cast<int>(species.size());
    // A table function's index is not among the addresses its call reads.
    // One indexed by the clock reads nothing of the state. Any other is read
    // by the expressions that call it (the name `tfun_<name>` in their text,
    // whatever stands after it or before: ExprTk compiles `tfun_kf ()`, a
    // bare `tfun_kf` and `2tfun_kf()` as the same call), with
    // what its index reads: a parameter, or the species behind an
    // observable. (Not by every expression, and not "nothing is known": one
    // table over a constant, read by nothing, then made every write feed
    // every trigger; one over a rule on a rateOf made every derivative read
    // that rateOf, its own species' among them, a ring.)
    struct Table {
        std::string call;
        int param = -1;
        std::vector<int> species;
    };
    std::vector<Table> tables;
    for (const auto &tf : impl_->table_functions) {
        if (is_time_index(tf->index_name())) {
            continue;
        }
        Table table;
        table.call = "tfun_" + tf->name();
        const std::string key = strip_paren_suffix(tf->index_name());
        const auto &sd = *impl_->shared;
        if (auto pit = sd.param_name_to_idx.find(key); pit != sd.param_name_to_idx.end()) {
            table.param = pit->second;
        } else if (auto oit = sd.observable_name_to_idx.find(key);
                   oit != sd.observable_name_to_idx.end() && oit->second >= 0 &&
                   oit->second < static_cast<int>(impl_->observables.size())) {
            for (const auto &entry : impl_->observables[oit->second].entries) {
                const int idx0 = entry.species_index - 1;
                if (idx0 >= 0 && idx0 < ns) {
                    table.species.push_back(idx0);
                }
            }
        } else {
            known.assign(n_groups, 0);
            return;
        }
        tables.push_back(std::move(table));
    }
    std::unordered_map<std::string, int> table_of;
    for (std::size_t k = 0; k < tables.size(); ++k) {
        table_of.emplace(tables[k].call, static_cast<int>(k));
    }
    // The identifiers of a text, a numeric literal stepped over whole (as
    // ModelBuilder's for_each_identifier reads them).
    const auto identifiers = [](const std::string &text) {
        std::vector<std::string> out;
        const std::size_t n = text.size();
        const auto digit = [&](std::size_t k) {
            return k < n && std::isdigit(static_cast<unsigned char>(text[k])) != 0;
        };
        std::size_t i = 0;
        while (i < n) {
            const auto c = static_cast<unsigned char>(text[i]);
            if (std::isdigit(c) != 0 || (text[i] == '.' && digit(i + 1))) {
                while (i < n && (digit(i) || text[i] == '.')) {
                    ++i;
                }
                if (i < n && (text[i] == 'e' || text[i] == 'E')) {
                    std::size_t j = i + 1;
                    if (j < n && (text[j] == '+' || text[j] == '-')) {
                        ++j;
                    }
                    if (digit(j)) {
                        i = j;
                        while (digit(i)) {
                            ++i;
                        }
                    }
                }
            } else if (std::isalpha(c) != 0 || text[i] == '_') {
                // (A dot inside a name is part of it, as ExprTk reads it: a
                // table named `k.f` is called as `tfun_k.f`.)
                const auto part = [&](std::size_t k) {
                    return k < n && (std::isalnum(static_cast<unsigned char>(text[k])) != 0 ||
                                     text[k] == '_');
                };
                const std::size_t start = i;
                while (part(i) || (i < n && text[i] == '.' && part(i + 1))) {
                    ++i;
                }
                out.push_back(text.substr(start, i - start));
            } else {
                ++i;
            }
        }
        return out;
    };
    std::unordered_map<const double *, int> species_of;
    species_of.reserve(species.size());
    for (int i = 0; i < ns; ++i) {
        species_of.emplace(&species[i].concentration, i);
    }
    std::unordered_map<const double *, const Observable *> obs_of;
    obs_of.reserve(impl_->observables.size());
    for (const Observable &obs : impl_->observables) {
        obs_of.emplace(&obs.total, &obs);
    }
    std::unordered_map<const double *, int> param_of;
    param_of.reserve(impl_->parameters.size());
    for (int i = 0; i < static_cast<int>(impl_->parameters.size()); ++i) {
        param_of.emplace(&impl_->parameters[i].value, i);
    }
    std::unordered_map<const double *, int> rateof_of;
    for (int i = 0; i < static_cast<int>(impl_->current_derivs.size()); ++i) {
        rateof_of.emplace(&impl_->current_derivs[static_cast<std::size_t>(i)], i);
    }
    std::unordered_map<int, int> written_by_function;
    for (const auto &[func_idx, param_idx] : impl_->shared->var_param_bindings) {
        written_by_function.emplace(param_idx, func_idx);
    }

    for (std::size_t g = 0; g < n_groups; ++g) {
        std::unordered_set<int> named;
        std::unordered_set<int> rated;
        bool all = true;
        std::unordered_set<int> visited; // expression ids, so a cycle terminates
        std::function<void(int)> visit = [&](int eid) {
            if (eid < 0 || !visited.insert(eid).second) {
                return;
            }
            if (!tables.empty()) {
                // (By the names in the text, each looked up among the
                // tables: every table asked of every expression, 2,000 of
                // each took a quarter longer to read.)
                std::unordered_set<int> called;
                for (const std::string &name :
                     identifiers(impl_->evaluator->preprocessed_expr(eid))) {
                    if (auto it = table_of.find(name); it != table_of.end()) {
                        called.insert(it->second);
                    }
                }
                for (int which : called) {
                    const Table &table = tables[static_cast<std::size_t>(which)];
                    named.insert(table.species.begin(), table.species.end());
                    const int pidx = table.param;
                    if (pidx < 0 || pidx >= static_cast<int>(impl_->parameters.size())) {
                        continue;
                    }
                    if (auto fn = written_by_function.find(pidx); fn != written_by_function.end()) {
                        visit(impl_->functions[fn->second].evaluator_id);
                    } else if (impl_->parameters[pidx].is_expression) {
                        visit(impl_->parameters[pidx].evaluator_id);
                    }
                }
            }
            for (const double *addr : impl_->evaluator->referenced_variable_addresses(eid)) {
                if (auto sp = species_of.find(addr); sp != species_of.end()) {
                    named.insert(sp->second);
                } else if (auto ob = obs_of.find(addr); ob != obs_of.end()) {
                    for (const auto &entry : ob->second->entries) {
                        const int idx0 = entry.species_index - 1;
                        if (idx0 >= 0 && idx0 < ns) {
                            named.insert(idx0);
                        }
                    }
                } else if (auto ra = rateof_of.find(addr); ra != rateof_of.end()) {
                    rated.insert(ra->second);
                } else if (auto pa = param_of.find(addr); pa != param_of.end()) {
                    const int pidx = pa->second;
                    if (auto fn = written_by_function.find(pidx); fn != written_by_function.end()) {
                        visit(impl_->functions[fn->second].evaluator_id);
                    } else if (impl_->parameters[pidx].is_expression) {
                        visit(impl_->parameters[pidx].evaluator_id);
                    }
                } else {
                    all = false; // something unmodelled
                }
            }
        };
        for (int eid : groups[g]) {
            visit(eid);
        }
        known[g] = all ? 1 : 0;
        species_out[g].assign(named.begin(), named.end());
        std::sort(species_out[g].begin(), species_out[g].end());
        rateof_out[g].assign(rated.begin(), rated.end());
        std::sort(rateof_out[g].begin(), rateof_out[g].end());
    }
}

int NetworkModel::rateof_flow_nesting() const {
    ensure_rateof_reads_();
    return impl_->rateof_flow_nesting;
}

std::string NetworkModel::rateof_ring_refusal() const {
    if (!impl_->uses_rateof) {
        return {};
    }
    ensure_rateof_reads_();
    const auto &ring = impl_->rateof_ring;
    if (ring.empty()) {
        return {};
    }
    const auto name = [&](int si) { return impl_->species[static_cast<std::size_t>(si)].name; };
    // A ring of rate rules is named whole up to six, and by its first six
    // beyond that.
    const std::size_t shown = std::min<std::size_t>(ring.size(), 6);
    std::string who;
    std::string how;
    for (std::size_t i = 0; i < shown; ++i) {
        who += (i == 0 ? "" : i + 1 == ring.size() ? " and " : ", ") + name(ring[i]);
        // What it reads of the ring: the next named where it reads that one,
        // and the first of the ring it does read otherwise.
        int read = -1;
        const auto &reads = impl_->derivative_rateof_reads[static_cast<std::size_t>(ring[i])];
        const int next = ring[(i + 1) % ring.size()];
        if (std::find(reads.begin(), reads.end(), next) != reads.end()) {
            read = next;
        } else {
            for (int x : reads) {
                if (std::find(ring.begin(), ring.end(), x) != ring.end()) {
                    read = x;
                    break;
                }
            }
        }
        if (read >= 0) {
            how += (how.empty() ? "" : ", ") +
                   ("d(" + name(ring[i]) + ")/dt reads rateOf(" + name(read) + ")");
        }
    }
    if (ring.size() > shown) {
        who += " and " + std::to_string(ring.size() - shown) + " more";
        how += ", ...";
    }
    if (ring.size() == 1) {
        return "The rate of change of " + who + " is written in terms of itself (" + how +
               "). That is an algebraic loop: the rate is defined by its own value at the same "
               "instant, and bngsim does not solve one (lanl/bngsim#991; libRoadRunner, Antimony "
               "and libSBML refuse such a model too). Write the rate without its own rateOf.";
    }
    return "The rates of change of " + who + " are written in terms of each other (" + how +
           "). That is an algebraic loop: each rate is defined by the others at the same "
           "instant, and bngsim does not solve one (lanl/bngsim#991; libRoadRunner, Antimony and "
           "libSBML refuse such a model too). Write the rates so that none reads, through any "
           "chain of rateOf, its own.";
}

int NetworkModel::rateof_nesting() const {
    ensure_rateof_reads_();
    return impl_->rateof_nesting;
}

void NetworkModel::ensure_rateof_reads_() const {
    if (impl_->rateof_reads_ready.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->rateof_reads_mutex);
    if (impl_->rateof_reads_ready.load(std::memory_order_relaxed)) {
        return;
    }
    {
        const auto &sd = *impl_->shared;
        const std::size_t ns = impl_->species.size();
        const int np = static_cast<int>(impl_->parameters.size());
        impl_->derivative_rateof_reads.assign(ns, {});
        impl_->rateof_read_elsewhere.clear();
        impl_->derivative_rateof_reads_state = 1;
        if (uses_rateof()) {
            std::unordered_map<int, int> written_by_function;
            for (const auto &[func_idx, param_idx] : sd.var_param_bindings) {
                written_by_function.emplace(param_idx, func_idx);
            }
            // A reaction's rate law, as the expressions behind its rate
            // parameters (a plain parameter has none, and reads no rateOf).
            std::vector<std::vector<int>> groups(sd.reactions.size());
            for (std::size_t r = 0; r < sd.reactions.size(); ++r) {
                const Reaction &rxn = sd.reactions[r];
                std::vector<int> params{rxn.rate_param_idx0, rxn.ssa_volume_param_idx0};
                for (int pi : rxn.rate_law_param_indices) {
                    params.push_back(pi - 1);
                }
                for (int p : params) {
                    if (p < 0 || p >= np) {
                        continue;
                    }
                    if (auto fn = written_by_function.find(p); fn != written_by_function.end()) {
                        groups[r].push_back(impl_->functions[fn->second].evaluator_id);
                    } else if (impl_->parameters[p].is_expression) {
                        groups[r].push_back(impl_->parameters[p].evaluator_id);
                    }
                }
            }
            // One group more for everything else that is evaluated: each
            // event's expressions, and the functions and expression
            // parameters that are no reaction's rate law (an output, an
            // assignment rule). What those reach is followed from them.
            std::unordered_set<int> rate_laws;
            for (const auto &group : groups) {
                rate_laws.insert(group.begin(), group.end());
            }
            std::vector<int> elsewhere;
            const auto also = [&](int eid) {
                if (eid >= 0 && rate_laws.count(eid) == 0) {
                    elsewhere.push_back(eid);
                }
            };
            for (const Event &ev : impl_->events) {
                also(ev.trigger_expr_idx);
                also(ev.delay_expr_idx);
                also(ev.priority_expr_idx);
                for (const auto &assignment : ev.assignments) {
                    also(assignment.second);
                }
                for (int eid : ev.assignment_rescale_size_expr) {
                    also(eid);
                }
            }
            for (const auto &fn : impl_->functions) {
                also(fn.evaluator_id);
            }
            for (const auto &param : impl_->parameters) {
                if (param.is_expression) {
                    also(param.evaluator_id);
                }
            }
            groups.push_back(std::move(elsewhere));
            std::vector<std::vector<int>> named;
            std::vector<std::vector<int>> rated;
            std::vector<char> known;
            expression_reads_(groups, named, rated, known);
            if (known.back() == 0) {
                impl_->derivative_rateof_reads_state = 2;
            }
            impl_->rateof_read_elsewhere = rated.back();
            for (std::size_t r = 0; r < sd.reactions.size(); ++r) {
                if (known[r] == 0) {
                    impl_->derivative_rateof_reads_state = 2;
                    break;
                }
                if (rated[r].empty()) {
                    continue;
                }
                // Every species the reaction changes. One that stands on
                // both sides as often (a catalyst) has nothing of the rate in
                // its derivative: the right-hand side takes zero times the
                // rate off it, and on a probe not that (compute_derivs_core).
                // While the two sides were applied one after the other it
                // had, `(1 - 1e17) + 1e17` being 0, and was given the rate
                // law's reads; which let a catalyst close a ring of rate
                // rules that is none.
                //
                // Nor has a species that is held fixed, whose derivative is
                // zero whatever is added to it: given the reads, `$X` made
                // by a reaction at `rateOf(Y)` beside `Y' = rateOf(X)` was a
                // ring, and refused, where Y' is a number.
                //
                // A reaction that changes no species has its rate law read by
                // no derivative, and the law is still a column of the result.
                const Reaction &rxn = sd.reactions[r];
                bool read = false;
                for (const auto *changed : {&rxn.net_loss, &rxn.net_gain}) {
                    for (const auto &[si, m] : *changed) {
                        if (m != 0.0 && si >= 0 && static_cast<std::size_t>(si) < ns &&
                            !impl_->species[static_cast<std::size_t>(si)].fixed) {
                            auto &reads = impl_->derivative_rateof_reads[si];
                            reads.insert(reads.end(), rated[r].begin(), rated[r].end());
                            read = true;
                        }
                    }
                }
                if (!read) {
                    auto &elsewhere = impl_->rateof_read_elsewhere;
                    elsewhere.insert(elsewhere.end(), rated[r].begin(), rated[r].end());
                }
            }
            for (auto &reads : impl_->derivative_rateof_reads) {
                std::sort(reads.begin(), reads.end());
                reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
            }
            {
                auto &elsewhere = impl_->rateof_read_elsewhere;
                std::sort(elsewhere.begin(), elsewhere.end());
                elsewhere.erase(std::unique(elsewhere.begin(), elsewhere.end()), elsewhere.end());
            }
        }
    }
    // How deep rateOf is nested, from the lists just filled.
    impl_->rateof_nesting = 0;
    impl_->rateof_flow_nesting = 0;
    impl_->rateof_ring.clear();
    if (impl_->derivative_rateof_reads_state != 1) {
        // (The right-hand side keeps its one probe where nothing can be said:
        // three more on every call would be four times the run.)
        impl_->rateof_nesting = 3;
    } else {
        const auto &reads = impl_->derivative_rateof_reads;
        const std::size_t ns = reads.size();
        // The longest chain of derivatives each reading the next one's
        // rateOf, however long: ten rate rules each on the rateOf of the one
        // before take nine probes more than one, and counted to eight (the
        // cap was on every chain) the tenth came out 0.087 for 0.562.
        //
        // Only a ring of them has no last link: `X' = rateOf(Y)` with
        // `Y' = rateOf(X)` defines each rate by the other at the same instant,
        // an algebraic loop that no number of probes reads (taken as eight
        // deep, such a model was integrated with rates a probe late, 0.1% to
        // 12% off). The model is refused for it by name
        // (rateof_ring_refusal), as libRoadRunner, Antimony and libSBML
        // refuse it. The rings are found as wholes (Tarjan's strongly
        // connected components, which come out with each one after
        // everything it reads), and the depth of everything else is counted
        // over them.
        //
        // (With a stack of its own. One call for each link, a chain of
        // 120,000 rate rules ended the process; 7,000 did on a thread with
        // half a megabyte.)
        std::vector<int> depth(ns, 0);
        {
            std::vector<int> order(ns, -1);
            std::vector<int> low(ns, 0);
            std::vector<int> component(ns, -1);
            std::vector<char> on_stack(ns, 0);
            std::vector<std::size_t> stack;
            std::vector<std::pair<std::size_t, std::size_t>> walk; // node, next read
            std::vector<std::size_t> members;
            int visited = 0;
            int components = 0;
            for (std::size_t root = 0; root < ns; ++root) {
                if (order[root] >= 0) {
                    continue;
                }
                walk.emplace_back(root, 0);
                while (!walk.empty()) {
                    const std::size_t v = walk.back().first;
                    if (walk.back().second == 0) {
                        order[v] = low[v] = visited++;
                        stack.push_back(v);
                        on_stack[v] = 1;
                    }
                    if (walk.back().second < reads[v].size()) {
                        const int x = reads[v][walk.back().second++];
                        if (x < 0 || static_cast<std::size_t>(x) >= ns) {
                            continue;
                        }
                        const auto w = static_cast<std::size_t>(x);
                        if (order[w] < 0) {
                            walk.emplace_back(w, 0);
                        } else if (on_stack[w] != 0) {
                            low[v] = std::min(low[v], order[w]);
                        }
                        continue;
                    }
                    if (low[v] == order[v]) {
                        members.clear();
                        for (;;) {
                            const std::size_t w = stack.back();
                            stack.pop_back();
                            on_stack[w] = 0;
                            component[w] = components;
                            members.push_back(w);
                            if (w == v) {
                                break;
                            }
                        }
                        bool reads_any = false;
                        bool ring = members.size() > 1;
                        int below = 0;
                        for (std::size_t m : members) {
                            for (int x : reads[m]) {
                                if (x < 0 || static_cast<std::size_t>(x) >= ns) {
                                    continue;
                                }
                                reads_any = true;
                                if (component[static_cast<std::size_t>(x)] == components) {
                                    ring = true;
                                } else {
                                    below = std::max(below, depth[static_cast<std::size_t>(x)]);
                                }
                            }
                        }
                        if (ring && impl_->rateof_ring.empty()) {
                            // In the order each reads the next: from the
                            // first declared, to a member it reads that has
                            // not been named, until none is left to name.
                            std::vector<std::size_t> todo(members);
                            std::sort(todo.begin(), todo.end());
                            std::size_t at = todo.front();
                            while (true) {
                                impl_->rateof_ring.push_back(static_cast<int>(at));
                                todo.erase(std::find(todo.begin(), todo.end(), at));
                                if (todo.empty()) {
                                    break;
                                }
                                std::size_t next = todo.front();
                                for (int x : reads[at]) {
                                    if (x >= 0 &&
                                        std::find(todo.begin(), todo.end(),
                                                  static_cast<std::size_t>(x)) != todo.end()) {
                                        next = static_cast<std::size_t>(x);
                                        break;
                                    }
                                }
                                at = next;
                            }
                        }
                        const int own = reads_any ? 1 : 0;
                        for (std::size_t m : members) {
                            depth[m] = own + below;
                        }
                        ++components;
                    }
                    walk.pop_back();
                    if (!walk.empty()) {
                        low[walk.back().first] = std::min(low[walk.back().first], low[v]);
                    }
                }
            }
        }
        const auto of = [&](int x) {
            return x >= 0 && static_cast<std::size_t>(x) < ns ? depth[static_cast<std::size_t>(x)]
                                                              : 0;
        };
        // Over the species whose rateOf is read: by a derivative, which is
        // what the right-hand side waits on, and by anything else. (Not over
        // every derivative that reads one. `Y' = rateOf(X)` with no rateOf(Y)
        // anywhere is one probe, and counted as two it made every row, stop
        // and event of such a model a third dearer.)
        int flow = 0;
        for (std::size_t i = 0; i < ns; ++i) {
            for (int x : reads[i]) {
                flow = std::max(flow, of(x));
            }
        }
        int nesting = flow;
        for (int x : impl_->rateof_read_elsewhere) {
            nesting = std::max(nesting, of(x));
        }
        impl_->rateof_flow_nesting = flow;
        impl_->rateof_nesting = nesting;
    }
    impl_->rateof_reads_ready.store(true, std::memory_order_release);
}

void NetworkModel::expression_support(int expr_idx, std::vector<int> *species_out,
                                      std::vector<int> *params_out) const {
    {
        auto memo = impl_->expression_support_cache.find(expr_idx);
        if (memo != impl_->expression_support_cache.end()) {
            if (species_out != nullptr) {
                *species_out = memo->second.first;
            }
            if (params_out != nullptr) {
                *params_out = memo->second.second;
            }
            return;
        }
    }
    const auto &species = impl_->species;
    const int ns = static_cast<int>(species.size());

    std::unordered_map<const double *, int> species_of;
    species_of.reserve(species.size());
    for (int i = 0; i < ns; ++i) {
        species_of.emplace(&species[i].concentration, i);
    }
    std::unordered_map<const double *, const Observable *> obs_of;
    obs_of.reserve(impl_->observables.size());
    for (const Observable &obs : impl_->observables) {
        obs_of.emplace(&obs.total, &obs);
    }
    std::unordered_map<const double *, int> param_of;
    param_of.reserve(impl_->parameters.size());
    for (int i = 0; i < static_cast<int>(impl_->parameters.size()); ++i) {
        param_of.emplace(&impl_->parameters[i].value, i);
    }
    // param index → the model function that writes it, if any.
    std::unordered_set<const double *> rateof_addrs;
    for (const double &deriv : impl_->current_derivs) {
        rateof_addrs.insert(&deriv);
    }
    std::unordered_map<int, int> written_by_function;
    for (const auto &[func_idx, param_idx] : impl_->shared->var_param_bindings) {
        written_by_function.emplace(param_idx, func_idx);
    }

    std::unordered_set<int> species_support;
    std::unordered_set<int> param_support;
    bool all_species = false;
    bool all_params = false;
    std::unordered_set<int> visited; // expression ids, so a cycle terminates

    std::function<void(int)> visit = [&](int eid) {
        if (eid < 0 || !visited.insert(eid).second) {
            return;
        }
        for (const double *addr : impl_->evaluator->referenced_variable_addresses(eid)) {
            auto sp = species_of.find(addr);
            if (sp != species_of.end()) {
                species_support.insert(sp->second);
                continue;
            }
            auto ob = obs_of.find(addr);
            if (ob != obs_of.end()) {
                for (const auto &entry : ob->second->entries) {
                    const int idx0 = entry.species_index - 1;
                    if (idx0 >= 0 && idx0 < ns) {
                        species_support.insert(idx0);
                    }
                }
                continue;
            }
            auto pa = param_of.find(addr);
            if (pa == param_of.end()) {
                all_species = true; // a rateOf accessor, or something unmodelled
                all_params = all_params || rateof_addrs.count(addr) != 0;
                continue;
            }
            const int pidx = pa->second;
            param_support.insert(pidx);
            auto fn = written_by_function.find(pidx);
            if (fn != written_by_function.end()) {
                visit(impl_->functions[fn->second].evaluator_id);
            } else if (impl_->parameters[pidx].is_expression) {
                visit(impl_->parameters[pidx].evaluator_id);
            }
        }
    };
    visit(expr_idx);

    std::vector<int> sp_out;
    if (all_species) {
        sp_out.resize(static_cast<std::size_t>(ns));
        std::iota(sp_out.begin(), sp_out.end(), 0);
    } else {
        sp_out.assign(species_support.begin(), species_support.end());
        std::sort(sp_out.begin(), sp_out.end());
    }
    std::vector<int> pa_out;
    if (all_params) {
        pa_out.resize(impl_->parameters.size());
        std::iota(pa_out.begin(), pa_out.end(), 0);
    } else {
        pa_out.assign(param_support.begin(), param_support.end());
        std::sort(pa_out.begin(), pa_out.end());
    }

    if (species_out != nullptr) {
        *species_out = sp_out;
    }
    if (params_out != nullptr) {
        *params_out = pa_out;
    }
    impl_->expression_support_cache.emplace(expr_idx,
                                            std::make_pair(std::move(sp_out), std::move(pa_out)));
}

// The end of the number starting at e[i]: digits and a point, then an exponent
// only when one is well formed (`1e-5`), so the `e` of `2exp(x)` is a name.
static size_t skip_number(const std::string &e, size_t i) {
    const auto digit = [&](size_t k) {
        return k < e.size() && std::isdigit(static_cast<unsigned char>(e[k])) != 0;
    };
    while (i < e.size() && (digit(i) || e[i] == '.'))
        ++i;
    if (i < e.size() && (e[i] == 'e' || e[i] == 'E')) {
        size_t j = i + 1;
        if (j < e.size() && (e[j] == '+' || e[j] == '-'))
            ++j;
        if (digit(j)) {
            i = j;
            while (digit(i))
                ++i;
        }
    }
    return i;
}

// What a rate (or a trigger) depends on through the model's definitions
// (issue #719): the species it reads and whether it reads the clock. A worklist
// over parameters: each one's defining function or expression is read as text
// (for the clock, a rate accessor, a table-function call) and through
// expression_support() (for the addresses it reads, which names the species
// behind each observable and every other parameter it reads, whose definitions
// join the worklist). A table function's index is not in the text of its call,
// so a call adds what the index reads: the clock, a parameter (whose definition
// joins the worklist) or an observable's species.
//
// The clock is the call `time()`, and a bare `time` unless the model declares a
// scalar of that name (issue #776): ExprTk calls a zero-argument function
// without its parentheses.
//
// A function in `pc_functions` (by index) reads the clock only inside
// conditions whose crossings are breakpoints, so it is constant between them:
// its own text's clock is not counted (a table or a rate accessor it calls
// still is).
NetworkModel::RateDeps
NetworkModel::rate_dependencies_(std::vector<int> params, const std::vector<std::string> &texts,
                                 const std::vector<char> *pc_functions) const {
    RateDeps deps;
    const auto &sd = *impl_->shared;
    const int ns = static_cast<int>(impl_->species.size());
    const int np = static_cast<int>(impl_->parameters.size());
    const bool time_declared =
        sd.param_name_to_idx.count("time") != 0 || sd.observable_name_to_idx.count("time") != 0;
    std::unordered_map<std::string, const TableFunction *> tables;
    for (const auto &tf : impl_->table_functions) {
        tables.emplace(tf->name(), tf.get());
        tables.emplace("tfun_" + tf->name(), tf.get());
    }
    std::vector<int> written_by(static_cast<std::size_t>(np), -1);
    for (const auto &[func_idx, param_idx] : sd.var_param_bindings)
        if (param_idx >= 0 && param_idx < np)
            written_by[param_idx] = func_idx;
    const auto add_obs = [&](int oi) {
        for (const auto &entry : impl_->observables[oi].entries)
            if (entry.species_index >= 1 && entry.species_index <= ns)
                deps.species.insert(entry.species_index - 1);
    };

    std::vector<char> seen(static_cast<std::size_t>(np), 0);
    const auto scan = [&](const std::string &e, bool clock = true) {
        if (e.find("rate_of__") != std::string::npos) {
            deps.time = true; // the running derivatives: every species, at every time
            deps.unknown = true;
        }
        for (size_t i = 0; i < e.size();) {
            const unsigned char c = static_cast<unsigned char>(e[i]);
            if (std::isdigit(c) || c == '.') {
                // A number, exponent included, and no further: ExprTk reads
                // `2time()` as 2*time(), so a name may follow a number directly.
                i = skip_number(e, i);
                continue;
            }
            if (!(std::isalpha(c) || c == '_')) {
                ++i;
                continue;
            }
            size_t j = i;
            while (j < e.size() && (std::isalnum(static_cast<unsigned char>(e[j])) || e[j] == '_'))
                ++j;
            const std::string id = e.substr(i, j - i);
            size_t k = j;
            while (k < e.size() && std::isspace(static_cast<unsigned char>(e[k])))
                ++k;
            const bool call = k < e.size() && e[k] == '(';
            if (clock && id == "time" && (call || !time_declared))
                deps.time = true;
            auto tit = tables.find(id);
            if (tit != tables.end()) {
                const std::string &idx = tit->second->index_name();
                const std::string key = strip_paren_suffix(idx);
                if (is_time_index(idx)) {
                    deps.time = true;
                } else if (auto pit = sd.param_name_to_idx.find(key);
                           pit != sd.param_name_to_idx.end()) {
                    params.push_back(pit->second);
                } else if (auto oit = sd.observable_name_to_idx.find(key);
                           oit != sd.observable_name_to_idx.end()) {
                    add_obs(oit->second);
                } else {
                    deps.unknown = true;
                }
            }
            i = j;
        }
    };
    for (const auto &t : texts)
        scan(t);
    std::vector<int> sp, pa;
    while (!params.empty()) {
        const int p = params.back();
        params.pop_back();
        if (p < 0 || p >= np || seen[p])
            continue;
        seen[p] = 1;
        int eid = -1;
        if (written_by[p] >= 0) {
            const Function &f = impl_->functions[written_by[p]];
            const bool clock =
                !(pc_functions && written_by[p] < static_cast<int>(pc_functions->size()) &&
                  (*pc_functions)[written_by[p]]);
            scan(f.expression, clock);
            scan(f.eval_expression); // a guarded form (#333) is not what was classified

            eid = f.evaluator_id;
        } else if (impl_->parameters[p].is_expression) {
            scan(impl_->parameters[p].expression);
            eid = impl_->parameters[p].evaluator_id;
        }
        if (eid < 0)
            continue;
        expression_support(eid, &sp, &pa);
        deps.species.insert(sp.begin(), sp.end());
        params.insert(params.end(), pa.begin(), pa.end());
    }
    return deps;
}

// A reaction's propensity, as compute_rxn_rate reads it: its reactants, the
// species of its SSA falling factorial, a live compartment volume, and its
// rate parameters' definitions.
NetworkModel::RateDeps
NetworkModel::reaction_rate_dependencies_(int rxn_idx0,
                                          const std::vector<char> *pc_functions) const {
    const Reaction &rxn = impl_->shared->reactions[rxn_idx0];
    std::vector<int> params{rxn.rate_param_idx0, rxn.ssa_volume_param_idx0};
    for (int pi : rxn.rate_law_param_indices)
        params.push_back(pi - 1);
    RateDeps deps = rate_dependencies_(std::move(params), {}, pc_functions);
    const int ns = static_cast<int>(impl_->species.size());
    const auto add = [&](int si) {
        if (si >= 0 && si < ns)
            deps.species.insert(si);
    };
    for (int ri : rxn.reactant_indices)
        add(ri - 1);
    for (const auto &[si, m] : rxn.ssa_falling_factorial)
        add(si);
    add(rxn.ssa_live_volume_idx0);
    for (const auto &lt : rxn.ssa_live_volume_terms)
        add(lt.live_idx0);
    return deps;
}

bool NetworkModel::reaction_rate_species_support(int rxn_idx0, std::vector<int> &out) const {
    out.clear();
    if (rxn_idx0 < 0 || rxn_idx0 >= static_cast<int>(impl_->shared->reactions.size()))
        return false;
    const RateDeps deps = reaction_rate_dependencies_(rxn_idx0);
    out.assign(deps.species.begin(), deps.species.end());
    return !deps.unknown;
}

bool NetworkModel::reaction_rate_reads_functions(int rxn_idx0) const {
    const auto &reactions = impl_->shared->reactions;
    if (rxn_idx0 < 0 || rxn_idx0 >= static_cast<int>(reactions.size()))
        return false;
    const Reaction &rxn = reactions[rxn_idx0];
    const int np = static_cast<int>(impl_->parameters.size());
    std::vector<char> written(static_cast<std::size_t>(np), 0);
    for (const auto &[func_idx, param_idx] : impl_->shared->var_param_bindings)
        if (param_idx >= 0 && param_idx < np)
            written[param_idx] = 1;
    std::vector<int> params{rxn.rate_param_idx0, rxn.ssa_volume_param_idx0};
    for (int pi : rxn.rate_law_param_indices)
        params.push_back(pi - 1);
    std::vector<int> support;
    for (int pidx : params) {
        if (pidx < 0 || pidx >= np)
            continue;
        if (written[pidx])
            return true;
        if (!impl_->parameters[pidx].is_expression)
            continue;
        expression_support(impl_->parameters[pidx].evaluator_id, nullptr, &support);
        for (int q : support)
            if (q >= 0 && q < np && written[q])
                return true;
    }
    return false;
}

bool NetworkModel::reaction_rate_reads_time(int rxn_idx0,
                                            const std::vector<char> *pc_functions) const {
    if (rxn_idx0 < 0 || rxn_idx0 >= static_cast<int>(impl_->shared->reactions.size()))
        return false;
    return reaction_rate_dependencies_(rxn_idx0, pc_functions).time;
}

bool NetworkModel::event_trigger_reads_time(int event_idx0) const {
    const auto &events = impl_->events;
    if (event_idx0 < 0 || event_idx0 >= static_cast<int>(events.size()))
        return false;
    const Event &ev = events[event_idx0];
    std::vector<int> params;
    if (ev.trigger_expr_idx >= 0) {
        std::vector<int> sp;
        expression_support(ev.trigger_expr_idx, &sp, &params);
    }
    return rate_dependencies_(std::move(params), {ev.trigger_source}).time;
}

bool NetworkModel::state_switch_reads_time(const StateSwitch &sw) const {
    std::vector<int> params;
    if (sw.residual_expr_idx >= 0) {
        std::vector<int> sp;
        expression_support(sw.residual_expr_idx, &sp, &params);
    }
    return rate_dependencies_(std::move(params), {sw.residual_source}).time;
}

bool NetworkModel::event_trigger_is_state_dependent(int event_idx0) const {
    const auto &events = impl_->events;
    if (event_idx0 < 0 || event_idx0 >= static_cast<int>(events.size())) {
        return false;
    }
    const Event &ev = events[event_idx0];
    if (ev.trigger_expr_idx < 0) {
        return false;
    }
    // Through expression_support(), not the trigger's direct references (issue
    // #775). An SBML assignment rule arrives as a function-written parameter
    // and a non-linear rule on a species as a parameter slot too, so `x < 1`
    // with `x := 2*S` names no state address at all, and this used to answer
    // "time-only": the solver then jumped at a fixed t* (dt*/dp = 0) and the
    // and/or/not refusal below never ran. The support walk already follows
    // observables, function-written and expression-valued parameters down to
    // the species behind them, and widens to every species for a rateOf
    // accessor, so it is a superset of what is_state_address() accepted.
    std::vector<int> species;
    expression_support(ev.trigger_expr_idx, &species, nullptr);
    return !species.empty();
}

std::vector<int> NetworkModel::events_with_runtime_event_time_sens() const {
    std::vector<int> out;
    for (int ei = 0; ei < static_cast<int>(impl_->events.size()); ++ei) {
        if (event_trigger_is_state_dependent(ei) && event_trigger_residual_expr(ei) >= 0) {
            out.push_back(ei);
        }
    }
    return out;
}

// Every address that carries live state, i.e. anything whose value moves with
// the trajectory. Issue #52: this deliberately covers more than species
// concentrations. ModelBuilder registers a species as an ExprTk variable only
// when its name is not already taken (model_builder.cpp), and SBML models
// routinely give each species an observable of the same name — so in `v > 30`
// the token `v` binds to the *observable total*, not to &sp.concentration.
// Checking concentrations alone therefore missed every SBML state-dependent
// trigger and answered those models instead of refusing them: on AMICI's
// `neuron` (Izhikevich, trigger `v > 30`) the returned sensitivities were
// 6x-135x off, in one direction, across every parameter. An observable total is
// a linear functional of the state and a rateOf accessor is dx/dt, so both move
// with the trajectory just as a concentration does.
bool NetworkModel::is_state_address(const double *addr) const {
    for (const Species &sp : impl_->species) {
        if (&sp.concentration == addr) {
            return true;
        }
    }
    for (const Observable &obs : impl_->observables) {
        if (&obs.total == addr) {
            return true;
        }
    }
    for (const double &deriv : impl_->current_derivs) {
        if (&deriv == addr) {
            return true;
        }
    }
    return false;
}

bool NetworkModel::event_delay_is_fixed_zero(int event_idx, std::vector<int> *reads) const {
    const Event &ev = impl_->events.at(static_cast<std::size_t>(event_idx));
    if (reads != nullptr) {
        reads->clear();
    }
    if (ev.delay_expr_idx < 0) {
        return ev.delay == 0.0;
    }
    const auto &params = impl_->parameters;
    const ExpressionEvaluator &eval = *impl_->evaluator;
    for (const double *addr : eval.referenced_variable_addresses(ev.delay_expr_idx)) {
        // A delay that reads trajectory state, time, or a derived parameter can
        // become positive after the run starts. A plain parameter cannot: its
        // value is fixed for the run (issue #835).
        int found = -1;
        for (std::size_t pi = 0; pi < params.size(); ++pi) {
            if (&params[pi].value == addr) {
                found = static_cast<int>(pi);
                break;
            }
        }
        if (found < 0 || params[static_cast<std::size_t>(found)].is_expression) {
            return false;
        }
        if (reads != nullptr) {
            reads->push_back(found);
        }
    }
    // evaluate() is non-const on the evaluator but has no observable effect
    // here; the model is logically unchanged. For a parameter-only expression
    // this reads the current run's value.
    ExpressionEvaluator &mut = const_cast<ExpressionEvaluator &>(eval);
    return mut.evaluate(ev.delay_expr_idx) == 0.0;
}

std::optional<std::string> NetworkModel::event_sensitivity_unsupported_reason(
    const std::vector<std::string> &sens_param_names,
    const std::vector<int> &event_time_compensated) const {
    const auto &events = impl_->events;
    if (events.empty()) {
        return std::nullopt;
    }
    const auto &params = impl_->parameters;

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

    // A trigger reading live state (see is_state_address, issue #52) has a
    // crossing time that depends on the sensitivity parameters *through the
    // state*, even when the trigger names no parameter at all, so dt*/dp is
    // non-zero exactly as for an explicitly parameter-dependent trigger. Issue
    // #144 differentiates that crossing at each fire instead of refusing it,
    // for the triggers that reduce to one relational comparison.

    // Issue #49 sub-finding: 25 of the 88 corpus events refused for a moving
    // crossing time ALSO tripped the delay / persistence checks — and in every
    // one of those cases the delay was a literal 0. A zero delay is not a
    // delay: process_firing_batch takes the immediate path for `delay_now <=
    // 0`, so nothing is queued and no time passes between trigger and
    // execution. There is still an ORDER within the instant: an earlier fire
    // of the same batch can change what a useValuesFromTriggerTime=false
    // assignment reads, or cancel a non-persistent instance. The sensitivity
    // jump composes the batch in the order it executed (issue #722), so that
    // needs no refusal here. Normalize it here rather than in the builder, so the
    // runtime's own delay handling is untouched. A delay *expression* counts as
    // vacuous when it reads only fixed parameters, none of them a requested
    // sensitivity parameter, and evaluates to 0 now; those parameter values can
    // be changed before a run but not during one. The verdict therefore holds
    // for this run's values only — the runtime re-checks at every fire (the
    // delayed-queue guard in CvodeSimulator::run), so a batch row that sets the
    // parameter nonzero is refused there rather than answered.
    //
    // zero_delay_sens_param is set when the only thing making a zero delay
    // effective is that it reads a requested sensitivity parameter.
    const Parameter *zero_delay_sens_param = nullptr;
    auto has_effective_delay = [&](int ei) {
        zero_delay_sens_param = nullptr;
        std::vector<int> reads;
        if (!event_delay_is_fixed_zero(ei, &reads)) {
            return true;
        }
        // A zero delay is still one to differentiate when it reads a requested
        // parameter: the execution time is t* + d(p), so ∂t_exec/∂p carries
        // ∂d/∂p, which the immediate path never adds. On `x' = -x`, reset
        // `x = 1` at `d` after `time > 1` with d = 0, the true ∂x/∂d is x
        // (right-sided), and passing this gate reported 0.
        for (const int pidx : reads) {
            if (sens_param_addrs.count(&params[pidx].value) != 0) {
                zero_delay_sens_param = &params[pidx];
                return true;
            }
        }
        return false;
    };

    const std::unordered_set<int> compensated(event_time_compensated.begin(),
                                              event_time_compensated.end());

    for (std::size_t ei = 0; ei < events.size(); ++ei) {
        const Event &ev = events[ei];
        const std::string id = ev.id.empty() ? std::string("<unnamed>") : ev.id;
        const bool delayed = has_effective_delay(static_cast<int>(ei));
        if (delayed && zero_delay_sens_param != nullptr) {
            const std::string &pname = zero_delay_sens_param->name;
            return "event '" + id +
                   "' has a delay that is 0 for this run but reads the requested " +
                   "sensitivity parameter '" + pname +
                   "', so its execution time moves with that parameter; forward sensitivity "
                   "through event delays is not yet supported (issue #144 tracks what is). "
                   "Drop '" +
                   pname + "' from the requested sensitivity parameters.";
        }
        if (delayed) {
            return "event '" + id +
                   "' has an execution delay; forward sensitivity through delayed events "
                   "is not yet supported (issue #144 tracks what is).";
        }
        // No persistence check follows: `persistent` governs whether a fire
        // queued at trigger time is cancelled when the trigger reverts before
        // execution time (SBML L3v2 §4.11.3). With no effective delay — the
        // only case that reaches here — trigger time IS execution time, so the
        // flag has no window to act in and refusing on it refused nothing real.
        // Ghanbari2020 and Zongo2020 were blocked by exactly this.
        // One definition of "reads state", shared with the runtime gate
        // (events_with_runtime_event_time_sens), so the two cannot drift again
        // (issue #775: this was a second copy of the direct-address loop).
        const bool state_dependent = event_trigger_is_state_dependent(static_cast<int>(ei));
        if (state_dependent && compensated.count(static_cast<int>(ei)) == 0) {
            // Issue #144: the crossing time moves with the trajectory, so
            // ∂t*/∂p is non-zero and cannot be resolved before the run — it is
            // differentiated at the fire, by the implicit function theorem on
            // the trigger's residual g. That needs a residual to differentiate.
            std::string why;
            if (event_trigger_residual_expr(static_cast<int>(ei), &why) < 0) {
                return "event '" + id +
                       "' has a state-dependent trigger (it reads a species concentration, an "
                       "observable, or a rate), so its crossing time depends on the "
                       "sensitivity parameters through the trajectory and the event-time "
                       "sensitivity dt*/dp is non-zero — but bngsim cannot differentiate this "
                       "trigger's crossing because " +
                       why + " (trigger: '" + ev.trigger_source + "'; issue #144).";
            }
            continue;
        }
        if (compensated.count(static_cast<int>(ei)) != 0) {
            // Issue #49: the Python detector resolved this trigger's threshold
            // to the model's primary parameters and supplies ∂t*/∂p, so the
            // crossing-time term is carried by the jump rather than dropped.
            continue;
        }
        // The parameters the trigger reaches, not just the ones it names: a
        // derived parameter or an assignment rule hides its primaries from the
        // trigger's own references (issue #775), and a requested primary behind
        // one moves the crossing just as a named one does.
        std::vector<int> trigger_params;
        if (ev.trigger_expr_idx >= 0) {
            expression_support(ev.trigger_expr_idx, nullptr, &trigger_params);
        }
        for (const int pidx : trigger_params) {
            if (sens_param_addrs.count(&params[pidx].value) != 0) {
                const std::string &pname = params[pidx].name;
                return "event '" + id +
                       "' has a trigger whose crossing time depends on the requested "
                       "sensitivity parameter '" +
                       pname +
                       "' (the event-time sensitivity dt*/dp is non-zero), and bngsim "
                       "could not resolve the trigger to a threshold on simulation time "
                       "or on a unit-rate counter, so it has no dt*/dp to jump by "
                       "(issue #49). Drop '" +
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
const JacobianSparsity &NetworkModel::ensure_jacobian_coloring() const {
    // Qualified: the member name hides the namespace-scope helper it forwards to.
    return bngsim::ensure_jacobian_coloring(*impl_->shared);
}
double NetworkModel::estimated_lu_fill() const {
    return bngsim::ensure_lu_fill_estimate(*impl_->shared);
}
const AnalyticalJacobianData &NetworkModel::analytical_jacobian() const {
    return impl_->shared->analytical_jac;
}
const FunctionalJacobianData &NetworkModel::functional_jacobian() const {
    return impl_->functional_jac;
}

const FunctionalJacobianDecline &NetworkModel::last_functional_jacobian_decline() const {
    return impl_->functional_jac_decline;
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
    const SharedModelData &sd = *impl_->shared;
    if (!sd.conservation_laws_enabled || !conservation_laws_follow_volumes(sd, impl_->species))
        return ensure_conservation_laws(sd, impl_->species);
    // Issue #758 — laws with compartment sizes in them, for the sizes this
    // model has now.
    std::vector<double> volumes;
    volumes.reserve(impl_->species.size());
    for (const auto &sp : impl_->species)
        volumes.push_back(sp.volume_factor);
    // NaN is a size like another here: a model at one is not found again at
    // every call.
    auto same = [](const std::vector<double> &a, const std::vector<double> &b) {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i]))))
                return false;
        }
        return true;
    };
    if (!impl_->volume_laws || !same(volumes, impl_->volume_laws_at)) {
        impl_->volume_laws =
            std::make_shared<const ConservationLaws>(conservation_laws_at(sd, impl_->species));
        impl_->volume_laws_at = std::move(volumes);
    }
    return *impl_->volume_laws;
}

std::vector<std::vector<int>> NetworkModel::conservation_law_members() const {
    return conservation_law_members_of(conservation_laws());
}

std::vector<std::vector<int>> conservation_law_members_of(const ConservationLaws &cl) {
    const int ns = cl.n_species;
    std::vector<std::vector<int>> members(cl.n_laws);
    std::vector<double> weight(ns);
    for (int k = 0; k < cl.n_laws; ++k) {
        const std::vector<double> &row = cl.coefficients[k];
        double largest = 0.0;
        for (int i = 0; i < ns; ++i) {
            const double w =
                static_cast<size_t>(i) < cl.species_weight.size() ? cl.species_weight[i] : 1.0;
            weight[i] = std::fabs(row[i]) / w;
            largest = std::max(largest, weight[i]);
        }
        for (int i = 0; i < ns; ++i) {
            if (weight[i] > 1e-9 * largest)
                members[k].push_back(i);
        }
    }
    return members;
}

NetworkModel::ConservationLawDrift NetworkModel::conservation_law_drift() {
    ConservationLawDrift found;
    const ConservationLaws &cl = conservation_laws();
    const int ns = n_species();
    if (cl.n_laws == 0 || ns == 0)
        return found;
    // From `cl` itself and not through the accessor again: `cl` is held here.
    const std::vector<std::vector<int>> members = conservation_law_members_of(cl);
    std::vector<double> here(ns), state(ns), net(ns), gross(ns);
    get_state_into(here.data());
    const double t_now = impl_->current_time;
    // What an evaluation writes, kept to be put back: the observable totals,
    // the value of every parameter (a function writes the one it is bound
    // to) and the functions' own cache. A model with no function writes none.
    std::vector<double> kept_totals, kept_values, kept_cache;
    if (impl_->has_functions) {
        kept_totals.reserve(impl_->observables.size());
        for (const auto &obs : impl_->observables)
            kept_totals.push_back(obs.total);
        kept_values.reserve(impl_->parameters.size());
        for (const auto &par : impl_->parameters)
            kept_values.push_back(par.value);
        kept_cache = impl_->function_value_cache;
    }
    for (int pass = 0; pass < 2 && found.law < 0; ++pass) {
        for (int i = 0; i < ns; ++i) {
            const double spread = 1.0 + 0.5 * std::sin(1.0 + i);
            const double shift = 1.0 + 0.25 * std::cos(static_cast<double>(i));
            state[i] = pass == 0 ? std::fabs(here[i]) * spread + 0.37 * shift
                                 : std::fabs(here[i]) / spread + 1.3 * shift;
        }
        // Each species' rate as compute_derivs sums it, and the same terms
        // with their absolute values summed. The rate is rounded from the
        // second: a species one reaction makes at 1e6 and another uses at
        // 1e6 has a rate that is 1e-10 of rounding whatever else moves it, so
        // the total is measured against the fluxes and not the net rates.
        compute_flux_split(t_now, state.data(), nullptr, net.data(), gross.data());
        for (int k = 0; k < cl.n_laws; ++k) {
            const std::vector<double> &row = cl.coefficients[k];
            double total = 0.0, size = 0.0;
            for (int i : members[k]) {
                total += row[i] * net[i];
                size += std::fabs(row[i]) * gross[i];
            }
            // A member whose rate is not finite makes the total NaN, which
            // compares false: that law is not asked at this state.
            if (std::fabs(total) > 1e-10 * size) {
                found.law = k;
                found.drift = std::fabs(total);
                found.size = size;
                break;
            }
        }
    }
    if (impl_->has_functions) {
        for (size_t i = 0; i < kept_totals.size(); ++i)
            impl_->observables[i].total = kept_totals[i];
        for (size_t i = 0; i < kept_values.size(); ++i)
            impl_->parameters[i].value = kept_values[i];
        impl_->function_value_cache = std::move(kept_cache);
    }
    return found;
}

// ─── Table function index kinds ──────────────────────────────────────────────

namespace {
// What a table function's index reads, written into `spec`'s index fields:
// "time"; "parameter", a slot no function owns (index_param_idx); "function",
// the slot a function owns (index_func_idx); "observable" (index_obs_idx); or
// "unknown". A function's slot holds its value only where the interpreter's
// evaluate_functions has just written it, so a reader that takes it for a
// parameter reads the value it had at load: compiled code did, and froze the
// trajectory (issue #965). This is the one place that tells the two apart.
void resolve_table_index(const TableFunction &tf, const SharedModelData &sd,
                         TableFunctionSpec &spec) {
    const auto &idx_name = tf.index_name();
    const std::string lookup = strip_paren_suffix(idx_name);
    if (is_time_index(idx_name)) {
        spec.index_kind = "time";
    } else if (sd.param_name_to_idx.count(lookup) != 0) {
        if (auto fit = sd.function_name_to_idx.find(lookup); fit != sd.function_name_to_idx.end()) {
            spec.index_kind = "function";
            spec.index_func_idx = fit->second;
        } else {
            spec.index_kind = "parameter";
            spec.index_param_idx = sd.param_name_to_idx.at(lookup);
        }
    } else if (auto oit = sd.observable_name_to_idx.find(lookup);
               oit != sd.observable_name_to_idx.end()) {
        spec.index_kind = "observable";
        spec.index_obs_idx = oit->second;
    } else {
        // Should be unreachable: register_table_function_ rejects an
        // unresolved index_name, so by the time a spec can be asked for the
        // binding has already passed that gate.
        spec.index_kind = "unknown";
    }
}
} // namespace

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
    ctx.species_volume_param.reserve(impl_->species.size());
    for (const auto &s : impl_->species) {
        ctx.species_meta.emplace_back(s.amount_valued, s.volume_factor);
        ctx.species_volume_param.emplace_back(
            s.volume_param_idx0 >= 0
                ? impl_->parameters[static_cast<std::size_t>(s.volume_param_idx0)].name
                : std::string());
    }

    // Constant parameters: every parameter whose name is NOT a function
    // (function-bound synthetic params are time/state-varying and are inlined
    // via function_map instead). Expression-valued (derived) params are
    // constant during integration, so they stay in this set.
    for (const auto &p : impl_->parameters)
        if (sd.function_name_to_idx.find(p.name) == sd.function_name_to_idx.end())
            ctx.constant_names.push_back(p.name);

    // Table functions that do not read the state: indexed by time, or by a
    // constant parameter. The evaluator registers each as the zero-argument
    // function `tfun_<name>` (register_table_function_), which is how it reaches
    // the rate-law text, so that is the name the symbolic core must be allowed
    // to treat as a constant — exactly as it treats `time()`. Its value at the
    // current time is what the derivative needs; no derivative of the table
    // itself is ever taken, because nothing it depends on is a species. A table
    // indexed by an observable depends on the state and stays out, so a rate
    // law that reads one keeps declining to finite differences, and so does
    // one indexed by a function (issue #965).
    for (const auto &tf : impl_->table_functions) {
        TableFunctionSpec index;
        resolve_table_index(*tf, sd, index);
        if (index.index_kind == "time" || index.index_kind == "parameter")
            ctx.constant_names.push_back("tfun_" + tf->name());
    }

    return ctx;
}

bool NetworkModel::set_functional_jacobian(const std::vector<FunctionalJacobianInput> &terms) {
    const auto &sd = *impl_->shared;
    const auto &sp = sd.jac_sparsity;

    // Issue #534: every `return false` below first records its verdict on the
    // instance, for last_functional_jacobian_decline() — which term or entry,
    // at which probe, both values — so the reason reaches the Python status as
    // text and not only the BNGSIM_JAC_DEBUG=1 stderr line. Reset here, so the
    // record describes THIS call: a call that returns true leaves kind None.
    using DeclineKind = FunctionalJacobianDecline::Kind;
    FunctionalJacobianDecline &verdict = impl_->functional_jac_decline;
    verdict = FunctionalJacobianDecline{};

    if (sp.empty()) {
        verdict.kind = DeclineKind::NoSparsity;
        return false;
    }

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
        if (in.rxn_idx < 0 || in.rxn_idx >= static_cast<int>(sd.reactions.size())) {
            verdict.kind = DeclineKind::BadReactionIndex;
            verdict.rxn_idx = in.rxn_idx;
            return false;
        }
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

        // (#170) `row_coeff` above bakes the psvs `1/V_i` divide, and only the
        // per-observable branch below uses it — the per-species branch defers the
        // divide to the scatter (GH #171) and is the one that can be made live.
        // The two never co-occur in practice: a cross-compartment reaction is
        // emitted with apply_species_factor=false, which is exactly what routes it
        // to the per-species branch. Fail the whole model's analytical Jacobian to
        // the FD path rather than bake a volume a write can move, if that ever
        // changes.
        if (in.per_observable && rxn.per_species_volume_scaling) {
            for (const auto &[i, c_i] : net) {
                if (c_i != 0.0 && i >= 0 && i < static_cast<int>(impl_->species.size()) &&
                    impl_->species[i].volume_param_idx0 >= 0) {
                    verdict.kind = DeclineKind::BakedVolume;
                    verdict.rxn_idx = in.rxn_idx;
                    verdict.per_observable = true;
                    verdict.row = i;
                    return false;
                }
            }
        }

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
            int outside_row = -1, outside_col = -1; // the entry ensure_col found no slot for
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
                    if (csc < 0) {
                        outside_row = i;
                        outside_col = j;
                        throw std::out_of_range("jac csc"); // signalled below
                    }
                    c.affected.emplace_back(csc, row_coeff(i, c_i));
                }
                return cols.emplace(j, std::move(c)).first->second;
            };

            try {
                // Term A: one ∂func/∂obs_k per observable index in deriv_terms.
                for (const auto &[obs_idx, deriv_str] : in.deriv_terms) {
                    if (obs_idx < 0 || obs_idx >= static_cast<int>(impl_->observables.size())) {
                        verdict.kind = DeclineKind::BadObservableIndex;
                        verdict.rxn_idx = in.rxn_idx;
                        verdict.per_observable = true;
                        verdict.target_idx = obs_idx;
                        return false;
                    }
                    int k_idx;
                    try {
                        k_idx = static_cast<int>(ot.dfunc_dobs_eval_ids.size());
                        ot.dfunc_dobs_eval_ids.push_back(impl_->evaluator->compile(deriv_str));
                    } catch (const std::exception &e) {
                        if (dbg)
                            std::fprintf(stderr,
                                         "[jac] rxn %d obs %d compile FAILED: %s\n  expr: %s\n",
                                         in.rxn_idx, obs_idx, e.what(), deriv_str.c_str());
                        verdict.kind = DeclineKind::CompileFailed;
                        verdict.rxn_idx = in.rxn_idx;
                        verdict.per_observable = true;
                        verdict.target_idx = obs_idx;
                        verdict.detail = e.what();
                        return false;
                    }
                    const auto &obs = impl_->observables[obs_idx];
                    for (const auto &entry : obs.entries) {
                        int j = entry.species_index - 1;
                        if (j < 0 || j >= static_cast<int>(impl_->species.size()))
                            continue;
                        // (#170) The `× V_j` an amount-valued column carries is
                        // applied by the scatter instead of baked here — it
                        // already reads `species[j].volume_factor` live for the
                        // product-rule term, and in the same `(g·V_j)·∂func/∂obs`
                        // order, so the value is unchanged and a written volume
                        // reaches it. A zero group factor is still skipped; a zero
                        // volume is not a case (the loader rejects V=0 outright).
                        const double g = entry.factor;
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
                    std::fprintf(stderr,
                                 "[jac] rxn %d: per-observable entry J[%d][%d] outside sparsity\n",
                                 in.rxn_idx, outside_row, outside_col);
                verdict.kind = DeclineKind::TermOutsidePattern;
                verdict.rxn_idx = in.rxn_idx;
                verdict.per_observable = true;
                verdict.row = outside_row;
                verdict.col = outside_col;
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
        // (#170) …and which parameter that divisor IS, so a written compartment
        // size moves the psvs divide. -1 for a non-psvs row and for a promoted
        // compartment (whose live volume already comes through `live_idx`).
        auto divisor_param_of = [&](int i) -> int {
            if (rxn.per_species_volume_scaling && i >= 0 && i < nsp)
                return impl_->species[i].volume_param_idx0;
            return -1;
        };

        for (const auto &[species_j, deriv_str] : in.deriv_terms) {
            int eval_id;
            try {
                eval_id = impl_->evaluator->compile(deriv_str);
            } catch (const std::exception &e) {
                if (dbg)
                    std::fprintf(stderr, "[jac] rxn %d species %d compile FAILED: %s\n  expr: %s\n",
                                 in.rxn_idx, species_j, e.what(), deriv_str.c_str());
                verdict.kind = DeclineKind::CompileFailed;
                verdict.rxn_idx = in.rxn_idx;
                verdict.target_idx = species_j;
                verdict.detail = e.what();
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
                    verdict.kind = DeclineKind::TermOutsidePattern;
                    verdict.rxn_idx = in.rxn_idx;
                    verdict.target_idx = species_j;
                    verdict.row = i;
                    verdict.col = species_j;
                    return false; // derivative entry outside the sparsity pattern
                }
                st.affected.push_back(
                    {csc, coeff, live_idx_of(i), static_divisor_of(i), divisor_param_of(i)});
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
                    verdict.kind = DeclineKind::TermOutsidePattern;
                    verdict.rxn_idx = in.rxn_idx;
                    verdict.row = i;
                    verdict.col = L;
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
        // and CONVERGED across well-separated step sizes (Richardson). A
        // genuine symbolic↔engine divergence then surfaces as a reliable FD entry
        // that still disagrees with the analytical value; FD artifacts are simply
        // not judged. Validated across the BioModels SBML corpus (1597 models):
        // zero wrong attaches, ~129 needless fall-backs recovered — see
        // dev/notes/gh76_investigation_findings.md. The same rule judges an
        // entry only where a derivative EXISTS: the two one-sided differences
        // (free, given the unperturbed RHS) must agree, or the probe sits on a
        // switching surface of a piecewise rate law and no finite difference is a
        // reference there (issue #511, below).
        //
        // Convergence is tested across THREE step sizes, 20x apart, not two
        // (issue #533). Two steps cannot see a plateau: a rate law written in a
        // cancelling form — BIOMD0000000393's quadratic root
        // (N/2)·(sqrt(1 + 4x/N) − 1) with 4x/N ~ 1e-6 resolves to 8e-10 relative in
        // doubles — has a rounding structure that is the SAME at 1e-5·|y| and at
        // 5e-7·|y|, so the two central differences agreed to 1e-6 while both
        // sat 3% off the true slope (an entry whose derivative is small against
        // the function's rounding floor), and a correct Jacobian was rejected.
        // Rounding error in a central difference grows as the step shrinks and
        // truncation error shrinks, so a difference that is right at one step
        // and wrong at another moves between 2e-4·|y| and 1e-5·|y|; requiring
        // the coarser step to agree as well exposes the plateau, and the entry
        // is skipped as an FD artifact rather than judged. A smooth function
        // converges across all three (the truncation error at 2e-4·|y| is
        // ~1e-7 relative for O(1) curvature), so a genuine mismatch is still
        // judged there.
        //
        // Per-entry verdict for a central difference converged across the three
        // step sizes: true ⇒ a trustworthy analytical↔FD mismatch (reject the
        // Jacobian).
        auto entry_is_mismatch = [](double an, double f0i, double fp0i, double fm0i, double fp1i,
                                    double fm1i, double fp2i, double fm2i, double h0, double h1,
                                    double h2) -> bool {
            double d0 = fp0i - fm0i, d1 = fp1i - fm1i, d2 = fp2i - fm2i;
            double fd0 = d0 / (2.0 * h0), fd1 = d1 / (2.0 * h1), fd2 = d2 / (2.0 * h2);
            if (!std::isfinite(fd0) || !std::isfinite(fd1) || !std::isfinite(fd2))
                return false;
            double sc0 = std::max(std::max(std::abs(fp0i), std::abs(fm0i)), 1.0);
            double sc1 = std::max(std::max(std::abs(fp1i), std::abs(fm1i)), 1.0);
            double sc2 = std::max(std::max(std::abs(fp2i), std::abs(fm2i)), 1.0);
            if (std::abs(d0) <= 1e-9 * sc0 || std::abs(d1) <= 1e-9 * sc1 ||
                std::abs(d2) <= 1e-9 * sc2)
                return false; // catastrophic cancellation: slope is float noise
            double cden = std::max(std::max(std::abs(fd0), std::abs(fd1)), std::abs(fd2));
            cden = std::max(cden, 1e-6);
            if (std::abs(fd1 - fd2) > 5e-3 * cden || std::abs(fd0 - fd1) > 5e-3 * cden)
                return false; // not converged across the three steps: FD is not a trustworthy
                              // oracle
            // Issue #511: a piecewise rate law — if(), a min/max or a division
            // guard — switches on a quantity that is exactly 0 at the seed state
            // whenever the species it compares are still at their seed values, so
            // the initial probe sits ON the switching surface. The forward and
            // backward probes then evaluate different branches: the central
            // difference is the mean of two one-sided slopes, converged in h (both
            // steps straddle the same surface) and equal to neither the analytical
            // value — the slope of the branch the point itself is on — nor anything
            // a derivative could be, because none exists there. Five corpus models
            // (a wave equation, a Schrödinger discretization, three BioModels) lost
            // a correct Jacobian to that mean. Where the one-sided slopes disagree
            // there is nothing to check at this probe; the spread-out probes below
            // do not sit on the seed's surfaces and judge the entry there.
            double fw = (fp2i - f0i) / h2, bw = (f0i - fm2i) / h2;
            double oden = std::max(std::max(std::abs(fw), std::abs(bw)), 1e-6);
            if (std::abs(fw - bw) > 5e-3 * oden)
                return false; // one-sided slopes disagree: on a switching surface, no reference
            double ad = std::abs(an - fd2);
            double denom = std::max(std::max(std::abs(an), std::abs(fd2)), 1e-6);
            return ad > 1e-6 && ad > 1e-2 * denom; // symmetric 1% band on the converged estimate
        };

        std::vector<double> f0(ns);
        std::vector<double> fp0(ns), fm0(ns), fp1(ns), fm1(ns), fp2(ns), fm2(ns);

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
            for (size_t pi = 0; pi < probes.size(); ++pi) {
                const auto &y = probes[pi];
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
                // Every entry is scanned so the verdict can say how many were
                // non-finite; it names the first (issue #534).
                int64_t n_nonfinite = 0;
                for (size_t k = 0; k < Ja.size(); ++k)
                    if (!std::isfinite(Ja[k])) {
                        const int row = static_cast<int>(k % ns), col = static_cast<int>(k / ns);
                        if (n_nonfinite++ == 0) {
                            if (std::getenv("BNGSIM_JAC_DEBUG"))
                                std::fprintf(stderr,
                                             "[jac] dense self-check probe %zu: non-finite "
                                             "analytical entry at (row=%d, col=%d) value=%g\n",
                                             pi, row, col, Ja[k]);
                            verdict.kind = DeclineKind::NonfiniteEntry;
                            verdict.probe = static_cast<int>(pi);
                            verdict.row = row;
                            verdict.col = col;
                            verdict.analytical = Ja[k];
                        }
                    }
                if (n_nonfinite > 0) {
                    verdict.n_nonfinite = n_nonfinite;
                    ok = false;
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
                    // three well-separated steps, 20x apart (issue #533)
                    double h0 = 2e-4 * aj, h1 = 1e-5 * aj, h2 = 5e-7 * aj;
                    std::vector<double> yp = y, ym = y;
                    yp[j] = y[j] + h0;
                    ym[j] = y[j] - h0;
                    compute_derivs(0.0, yp.data(), fp0.data());
                    compute_derivs(0.0, ym.data(), fm0.data());
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
                        if (entry_is_mismatch(an, f0[i], fp0[i], fm0[i], fp1[i], fm1[i], fp2[i],
                                              fm2[i], h0, h1, h2)) {
                            const double fd = (fp2[i] - fm2[i]) / (2.0 * h2);
                            if (std::getenv("BNGSIM_JAC_DEBUG")) {
                                std::fprintf(stderr,
                                             "[jac] dense self-check probe %zu: FD mismatch at "
                                             "(row=%d, col=%d) analytical=%g fd=%g\n",
                                             pi, i, j, an, fd);
                            }
                            verdict.kind = DeclineKind::FdMismatch;
                            verdict.probe = static_cast<int>(pi);
                            verdict.row = i;
                            verdict.col = j;
                            verdict.analytical = an;
                            verdict.finite_difference = fd;
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
                    // The column of the first bad entry is the one whose CSC
                    // pointer range holds its data index.
                    const int first_row = static_cast<int>(sp.row_indices[first_bad]);
                    const int first_col =
                        static_cast<int>(
                            std::upper_bound(sp.col_ptrs.begin(), sp.col_ptrs.end(), first_bad) -
                            sp.col_ptrs.begin()) -
                        1;
                    if (std::getenv("BNGSIM_JAC_DEBUG"))
                        std::fprintf(stderr,
                                     "[jac] sparse self-check probe %zu: %lld non-finite "
                                     "analytical entries (first at csc=%lld, row=%d, col=%d, "
                                     "value=%g)\n",
                                     pi, (long long)nonfinite, (long long)first_bad, first_row,
                                     first_col, vals[first_bad]);
                    verdict.kind = DeclineKind::NonfiniteEntry;
                    verdict.probe = static_cast<int>(pi);
                    verdict.row = first_row;
                    verdict.col = first_col;
                    verdict.analytical = vals[first_bad];
                    verdict.n_nonfinite = nonfinite;
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
                    double h0 = 2e-4 * aj, h1 = 1e-5 * aj, h2 = 5e-7 * aj; // issue #533
                    yp[j] = y[j] + h0;
                    ym[j] = y[j] - h0;
                    compute_derivs(0.0, yp.data(), fp0.data());
                    compute_derivs(0.0, ym.data(), fm0.data());
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
                        if (entry_is_mismatch(acol[i], f0[i], fp0[i], fm0[i], fp1[i], fm1[i],
                                              fp2[i], fm2[i], h0, h1, h2)) {
                            const double fd = (fp2[i] - fm2[i]) / (2.0 * h2);
                            if (std::getenv("BNGSIM_JAC_DEBUG")) {
                                std::fprintf(stderr,
                                             "[jac] sparse self-check probe %zu: FD mismatch at "
                                             "(row=%d, col=%d) analytical=%g fd=%g\n",
                                             pi, i, j, acol[i], fd);
                            }
                            verdict.kind = DeclineKind::FdMismatch;
                            verdict.probe = static_cast<int>(pi);
                            verdict.row = i;
                            verdict.col = j;
                            verdict.analytical = acol[i];
                            verdict.finite_difference = fd;
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

// Issue #170 — the amount_valued reactant volume product for one Elementary
// reaction. `amount_factor` is baked into the SHARED analytical-Jacobian data at
// build time, so a compartment size written afterwards cannot move it; when any
// factor came from a compartment-size *parameter* the loader records the same
// factors in the same order in `amount_volume_terms` and this reproduces the
// product from the live parameter values. Empty (the `.net` / V=1 / hOSU=false
// case, and any model with no writable compartment) ⇒ the baked value, unchanged.
static inline double live_amount_factor(const AnalyticalJacobianData::ReactionTerms &terms,
                                        const std::vector<Parameter> &params) {
    if (terms.amount_volume_terms.empty())
        return terms.amount_factor;
    double af = 1.0;
    for (const auto &[pidx, static_v] : terms.amount_volume_terms)
        af *= (pidx >= 0 && pidx < static_cast<int>(params.size())) ? params[pidx].value : static_v;
    return af;
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
        double k_sf = k * rxn_terms.stat_factor * live_amount_factor(rxn_terms, params);
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
                double divisor = a.static_divisor_param_idx0 >= 0
                                     ? params[a.static_divisor_param_idx0].value
                                     : a.static_divisor;
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
                if (vt.func_param_idx0 < 0 || vt.func_param_idx0 >= static_cast<int>(params.size()))
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

void NetworkModel::fill_dense_fd_jacobian(double t, const double *conc, double *jac) {
    // No excluded species: the probe scale is the whole state's magnitude. A
    // masked steady-state solve keeps its accumulators out of that scale
    // (issue #74); an arbitrary-state evaluation has no mask to honor.
    static const std::vector<int> kNoExcluded;
    fd_dense_state_jacobian(
        [this](double tt, const double *y, double *f) { compute_derivs(tt, y, f); }, t, conc,
        n_species(), kNoExcluded, jac);
}

std::vector<int> NetworkModel::fill_dense_fd_jacobian_own_step(double t, const double *conc,
                                                               const double *floored, double *jac) {
    static const std::vector<int> kNoExcluded;
    const int ns = n_species();
    std::vector<double> f0(static_cast<size_t>(ns));
    compute_derivs(t, conc, f0.data());
    return fd_own_step_columns([this, t](const double *y, double *f) { compute_derivs(t, y, f); },
                               conc, ns, ns, kNoExcluded, f0.data(), floored, jac);
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
        double k_sf = k * rxn_terms.stat_factor * live_amount_factor(rxn_terms, params);
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
                double divisor = a.static_divisor_param_idx0 >= 0
                                     ? params[a.static_divisor_param_idx0].value
                                     : a.static_divisor;
                if (a.live_idx >= 0 && conc[a.live_idx] > 0.0)
                    divisor = conc[a.live_idx];
                vals[a.csc] += a.coeff * dv_dxj / divisor;
            }
        }
        // GH #171: the new ∂/∂V_live column (sparse mirror of the dense path).
        if (!impl_->functional_jac.volume_terms.empty()) {
            evaluate_functions(t);
            for (const auto &vt : impl_->functional_jac.volume_terms) {
                if (vt.func_param_idx0 < 0 || vt.func_param_idx0 >= static_cast<int>(params.size()))
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

// The 1/V_c that `resolve_ic_from_param` puts between "the parameter that names
// this IC" and "the number stored" (issue #170 stage 3). Read from the same
// species fields that function reads, in the ref order, so a caller building the
// chain rule ∂(stored IC)/∂p cannot use a different convention than the loader
// and set_param() use to produce the value.
std::vector<double> NetworkModel::species_ic_param_ref_divisors() const {
    const auto &refs = impl_->shared->species_ic_param_refs;
    std::vector<double> out;
    out.reserve(refs.size());
    for (const auto &ref : refs) {
        const auto &sp = impl_->species[static_cast<std::size_t>(ref.first)];
        // `std::isfinite` is not in resolve_ic_from_param's guard, which would
        // divide by a NaN volume and store the NaN. Since issue #353 the loader
        // substitutes a unit volume for a non-finite declared size, so a model
        // like MODEL2002070001 no longer arrives here with a NaN volume_factor;
        // this guard remains the runtime backstop for any volume_factor that is
        // still non-finite, keeping the *seed* the finite number it was rather
        // than a NaN yS(0) that poisons a whole CVODES column. Same call as
        // build_per_species_sympy's degenerate-coefficient check, for the same
        // reason.
        const bool usable =
            sp.amount_valued && sp.volume_factor != 0.0 && std::isfinite(sp.volume_factor);
        out.push_back(usable ? sp.volume_factor : 1.0);
    }
    return out;
}

// ∂(stored IC)/∂V — the forward-sensitivity seed a writable compartment size
// needs (issue #170 stage 3).
//
// bngsim stores amount/V_c, so for the species whose declared initial condition
// is an AMOUNT the stored number moves when the volume does, and the state's own
// meaning moves with the parameter: x(0) = A/V ⇒ ∂x(0)/∂V = −A/V² = −x(0)/V.
// Without this seed the column starts at zero and is wrong from t=0 by a term
// that never decays. The two ways a stored IC can be an amount over a volume are
// the two `refresh_*` functions above, and this returns a row for each:
//
//   * `initial_amount` non-NaN — an `initialAmount` or an amount-valued
//     `<initialAssignment>`, refreshed by refresh_compartment_volume_state();
//   * an `amount_valued` species whose IC a parameter names, refreshed by
//     refresh_param_ref_ics() through resolve_ic_from_param's divide. Its
//     *numerator* may move with the volume too (an `<initialAssignment>` reading
//     the compartment is carried as a derived parameter); that half is the
//     ordinary parameter-graph chain rule, scaled by
//     species_ic_param_ref_divisors(). This is only the explicit divide.
//
// A concentration-declared species is not here: its stored value IS the declared
// number at every volume, so the seed is a true zero rather than a missing row.
// Neither is anything at all once save_concentrations() has redefined the
// baseline — the same latch that stops both refresh functions, for the same
// reason (issue #79/#81: the carried state is not the declared IC any more).
std::vector<std::tuple<int, int, double>> NetworkModel::compartment_ic_sens_seeds() const {
    std::vector<std::tuple<int, int, double>> out;
    if (impl_->ic_baseline_saved)
        return out;
    std::vector<char> is_param_ref(impl_->species.size(), 0);
    for (const auto &ref : impl_->shared->species_ic_param_refs)
        is_param_ref[static_cast<std::size_t>(ref.first)] = 1;
    for (std::size_t i = 0; i < impl_->species.size(); ++i) {
        const auto &sp = impl_->species[i];
        if (sp.volume_param_idx0 < 0)
            continue;
        const double v = impl_->parameters[static_cast<std::size_t>(sp.volume_param_idx0)].value;
        if (v == 0.0)
            continue; // both refresh paths leave a zero volume undivided
        const bool amount_declared = !std::isnan(sp.initial_amount);
        const bool amount_param_ref = sp.amount_valued && is_param_ref[i];
        if (!amount_declared && !amount_param_ref)
            continue;
        const double seed = -sp.initial_conc / v;
        // Since issue #353 a non-finite declared compartment size is substituted
        // with a unit volume at load, so a model like MODEL2002070001 arrives here
        // with finite numbers and this column is emitted (and FD-correct). The
        // guard stays as the backstop for any seed still non-finite through
        // another path: emitting a NaN row would put a NaN in yS(0) and fail the
        // whole solve, so leave it out.
        if (!std::isfinite(seed))
            continue;
        out.emplace_back(static_cast<int>(i), sp.volume_param_idx0, seed);
    }
    return out;
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

std::vector<int> NetworkModel::pure_sink_species() const {
    const int ns = static_cast<int>(impl_->species.size());
    if (ns == 0) {
        return {};
    }

    // Clauses 1 and 2 in one pass over the reaction list (see model.hpp).
    std::vector<char> is_reactant(ns, 0);
    std::vector<char> is_product(ns, 0);
    for (const auto &rxn : impl_->shared->reactions) {
        for (int si_1 : rxn.reactant_indices) {
            const int si_0 = si_1 - 1;
            if (si_0 >= 0 && si_0 < ns)
                is_reactant[si_0] = 1;
        }
        for (int si_1 : rxn.product_indices) {
            const int si_0 = si_1 - 1;
            if (si_0 >= 0 && si_0 < ns)
                is_product[si_0] = 1;
        }
    }

    // Clause 3: a structurally empty Jacobian column. The pattern is CSC, so
    // column j is empty iff it spans no entries. A model whose sparsity was
    // never built (n == 0) cannot answer clause 3, so it answers no sinks rather
    // than asserting inertness it has not checked.
    const JacobianSparsity &sp = jacobian_sparsity();
    if (sp.n != ns || static_cast<int>(sp.col_ptrs.size()) != ns + 1) {
        return {};
    }

    std::vector<int> sinks;
    for (int j = 0; j < ns; ++j) {
        if (!is_product[j] || is_reactant[j] || impl_->species[j].fixed)
            continue;
        if (sp.col_ptrs[j + 1] != sp.col_ptrs[j])
            continue; // something's derivative reads it — not inert
        sinks.push_back(j);
    }
    return sinks;
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

std::vector<std::string> NetworkModel::function_expressions() const {
    std::vector<std::string> exprs;
    exprs.reserve(impl_->functions.size());
    for (const auto &f : impl_->functions) {
        exprs.push_back(f.expression);
    }
    return exprs;
}

std::vector<std::string> NetworkModel::function_eval_expressions() const {
    std::vector<std::string> exprs;
    exprs.reserve(impl_->functions.size());
    for (const auto &f : impl_->functions) {
        exprs.push_back(f.eval_expression);
    }
    return exprs;
}

bool NetworkModel::set_function_eval_expression(const std::string &name,
                                                const std::string &expression) {
    for (auto &f : impl_->functions) {
        if (f.name != name) {
            continue;
        }
        // Compile before publishing either field, so a replacement that does not
        // parse leaves the function exactly as it was rather than half-updated.
        // The superseded evaluator entry stays in the evaluator's expression list
        // — compile() only appends — which costs one dead slot per replaced
        // function and keeps every other function's id stable.
        int compiled = -1;
        try {
            compiled = impl_->evaluator->compile(expression);
        } catch (const std::exception &e) {
            throw std::runtime_error("set_function_eval_expression: function '" + name +
                                     "' replacement did not compile: " + expression + " — " +
                                     e.what());
        }
        f.evaluator_id = compiled;
        f.eval_expression = expression;
        // A body that names the time makes the function block one that reads
        // it, as the build decides it from the functions' text (issue #654).
        // Left as it was built, a steady-state solve of the model was not
        // refused and a time course's early stop was not held (issue #710).
        const auto in_a_word = [&](std::size_t k) {
            return k < expression.size() &&
                   (std::isalnum(static_cast<unsigned char>(expression[k])) != 0 ||
                    expression[k] == '_');
        };
        for (std::size_t at = expression.find("time"); at != std::string::npos;
             at = expression.find("time", at + 1)) {
            if ((at == 0 || !in_a_word(at - 1)) && !in_a_word(at + 4)) {
                impl_->functions_use_time = true;
                break;
            }
        }
        // The support walk follows a function-written parameter into its
        // function's body, so a new body can change the support of anything
        // that reads it (issue #773).
        impl_->expression_support_cache.clear();
        return true;
    }
    return false;
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

    // Evaluate each function expression and update its bound parameter, one
    // strongly-connected group at a time and in the order build() sorted them
    // (GH #76). A group of one — every function in every corpus model but one —
    // is a single evaluation, so this is the same walk it has always been.
    const auto &starts = impl_->shared->function_scc_starts;
    const auto &bindings = impl_->shared->var_param_bindings;
    const auto eval_one = [&](size_t bi) {
        const auto &[func_idx, param_idx] = bindings[bi];
        const auto &func = impl_->functions[func_idx];
        if (func.evaluator_id < 0)
            return;
        const double val = impl_->evaluator->evaluate(func.evaluator_id);
        impl_->parameters[param_idx].value = val;
        cache[func_idx] = val;
    };

    for (size_t g = 0; g + 1 < starts.size(); ++g) {
        const size_t lo = static_cast<size_t>(starts[g]), hi = static_cast<size_t>(starts[g + 1]);
        if (hi - lo == 1) {
            eval_one(lo);
            continue;
        }
        solve_function_cycle(lo, hi);
    }
}

// Gaussian elimination with partial pivoting, in place: solves A x = b for the
// small dense systems solve_function_cycle() builds (k is a function-cycle size
// — four on the one corpus model that has one). Returns false when the pivot
// column is numerically empty, which the caller treats as "this group did not
// solve" rather than pressing on with a fabricated step. A dedicated routine
// rather than the LAPACK path: that one is wired to SUNDIALS' SUNLinearSolver
// for the Jacobian and carries setup this does not need at k of a handful.
static bool solve_dense_in_place(std::vector<double> &a, std::vector<double> &b, size_t k) {
    for (size_t col = 0; col < k; ++col) {
        size_t piv = col;
        for (size_t r = col + 1; r < k; ++r)
            if (std::abs(a[r * k + col]) > std::abs(a[piv * k + col]))
                piv = r;
        if (!(std::abs(a[piv * k + col]) > 0.0))
            return false;
        if (piv != col) {
            for (size_t c = 0; c < k; ++c)
                std::swap(a[col * k + c], a[piv * k + c]);
            std::swap(b[col], b[piv]);
        }
        const double d = a[col * k + col];
        for (size_t r = col + 1; r < k; ++r) {
            const double m = a[r * k + col] / d;
            if (m == 0.0)
                continue;
            for (size_t c = col; c < k; ++c)
                a[r * k + c] -= m * a[col * k + c];
            b[r] -= m * b[col];
        }
    }
    for (size_t i = k; i-- > 0;) {
        double acc = b[i];
        for (size_t c = i + 1; c < k; ++c)
            acc -= a[i * k + c] * b[c];
        b[i] = acc / a[i * k + i];
    }
    return true;
}

// Solve one strongly-connected group of functions — a set that reads itself, so
// no order evaluates it one at a time (issue #621).
//
// What this replaces: the group used to be swept once per RHS evaluation, in
// whatever order the sort's cycle fallback left it. That is a Gauss-Seidel
// iteration with no convergence check and no fixed number of steps, so the
// value a function held depended on how many times the RHS had been called —
// `Model.rhs(y, t)` returned a different number on every call at the same state,
// and `compute_derivs` was not a function of (t, y) at all. Below a loop gain of
// 1 it crept toward the answer without arriving; above 1 it ran away. On
// MODEL1006230117, the one corpus model with such a group, the gain is ~36: it
// reached `inf` before t = 1e-13 and the solve died there.
//
// Newton on the residual g(x) = F(x) - x, not iteration of F. That choice is
// forced by the same model: its group is LINEAR in its unknowns (`R` reads
// `LR`, `LRG`, `RG`; each of those reads only `R`), so the system has one exact
// solution — and a linear system is precisely where Newton lands on it in a
// single step whatever the gain, while fixed-point iteration diverges for any
// gain above 1. Groups are small (four here; the Jacobian is dense k*k with k
// the group size), and nothing pays for this unless it has such a group.
//
// F is evaluated Jacobi-style — every member read from the SAME x — because a
// residual built from half-updated values is not g(x) and its difference
// quotient is not J.
void NetworkModel::solve_function_cycle(size_t lo, size_t hi) {
    const auto &bindings = impl_->shared->var_param_bindings;
    const size_t k = hi - lo;
    auto &scratch = impl_->fn_cycle_scratch;
    scratch.x.assign(k, 0.0);
    scratch.f.assign(k, 0.0);
    scratch.g.assign(k, 0.0);
    scratch.jac.assign(k * k, 0.0);

    // Seed from zero, NOT from what the slots happen to hold. The slots hold the
    // previous call's answer, which makes the starting point a function of
    // history — and Newton stops as soon as the residual is under tolerance, so
    // a different start lands on a different point inside that band. Warm-
    // starting left `Model.rhs()` returning seven distinct values across twelve
    // calls at one state, spread 1.7e-12 relative: the issue #621 defect
    // surviving twelve orders of magnitude smaller, which is still not a
    // function of (t, y). A constant seed makes every step downstream of it a
    // function of the state alone, so the same state gives the same answer
    // bit for bit.
    std::fill(scratch.x.begin(), scratch.x.end(), 0.0);

    // F(x): write x into every bound slot, then read all members off it.
    const auto eval_F = [&](const std::vector<double> &x, std::vector<double> &out) {
        for (size_t i = 0; i < k; ++i)
            impl_->parameters[bindings[lo + i].second].value = x[i];
        for (size_t i = 0; i < k; ++i) {
            const auto &func = impl_->functions[bindings[lo + i].first];
            out[i] = func.evaluator_id >= 0 ? impl_->evaluator->evaluate(func.evaluator_id) : x[i];
        }
    };

    // Convergence and the FD step are both RELATIVE to the group's own
    // magnitude, max(|x|, |F(x)|) (issue #782). The old test,
    // `norm <= kTol*(1 + max|x|)`, was absolute at the x = 0 seed — max|F(0)|
    // <= 1e-12 — so any group whose values were all ~1e-12 or smaller was
    // declared solved at F(0) without a single Newton step: the wrong answer,
    // marked converged, so the NaN guard below never fired. A cyclic model is
    // scale-invariant in exactly the way that test was not. The only absolute
    // floor left is DBL_MIN, which admits nothing but a group that is zero.
    //
    // For the same reason the seed is never accepted unless its residual is
    // exactly zero (x = 0 is then a true fixed point). Otherwise at least one
    // Newton step is taken, which is exact for a linear group whatever its
    // scale.
    constexpr int kMaxIter = 50;
    constexpr double kTol = 1e-12;
    constexpr double kFloor = std::numeric_limits<double>::min();
    bool converged = false;
    for (int iter = 0; iter < kMaxIter && !converged; ++iter) {
        eval_F(scratch.x, scratch.f);
        double norm = 0.0, scale = 0.0;
        for (size_t i = 0; i < k; ++i) {
            scratch.g[i] = scratch.f[i] - scratch.x[i];
            norm = std::max(norm, std::abs(scratch.g[i]));
            scale = std::max({scale, std::abs(scratch.x[i]), std::abs(scratch.f[i])});
        }
        if (!std::isfinite(norm))
            break;
        if (iter == 0 ? norm == 0.0 : norm <= kTol * scale + kFloor) {
            converged = true;
            break;
        }
        // Dense FD Jacobian of g, column by column. The step scales with the
        // group (issue #782): a fixed `1e-7*(|x|+1)` is ~1e6 times a 1e-13
        // group, which turns the Jacobian of a nonlinear group into a secant
        // taken far from the point.
        std::vector<double> xp = scratch.x, fp(k);
        for (size_t j = 0; j < k; ++j) {
            const double h = 1e-7 * std::max({std::abs(scratch.x[j]), scale, kFloor});
            const double keep = xp[j];
            xp[j] = keep + h;
            eval_F(xp, fp);
            for (size_t i = 0; i < k; ++i) {
                const double gp = fp[i] - xp[i];
                scratch.jac[i * k + j] = (gp - scratch.g[i]) / h;
            }
            xp[j] = keep;
        }
        if (!solve_dense_in_place(scratch.jac, scratch.g, k))
            break; // singular: leave the last iterate, below
        for (size_t i = 0; i < k; ++i)
            scratch.x[i] -= scratch.g[i];
    }

    // Write the result back, converged or not. A group that would not solve is
    // left holding its last iterate and its functions are marked non-finite, so
    // the RHS guards that already exist report it rather than a silently
    // state-independent number reaching the integrator.
    eval_F(scratch.x, scratch.f);
    for (size_t i = 0; i < k; ++i) {
        const double val = converged ? scratch.f[i] : std::numeric_limits<double>::quiet_NaN();
        impl_->parameters[bindings[lo + i].second].value = val;
        impl_->function_value_cache[bindings[lo + i].first] = val;
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
// SSA falling factorial is taken over the molecule count of every species,
// whichever units it is stored in (see compute_species_factor_ssa).
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
//
// The falling factorial is over the molecule count n = c·V, where c is the
// stored value and V the species' volume_factor (issue #692). It is formed in
// storage units, ∏_{j<m} (c − j/V) = ∏(n − j)/V^m, so an hOSU=false species
// keeps the concentration units its rate law is written in, and an amount-
// valued one multiplies each term back by V. It used to subtract j from c
// itself, which is the count only at V = 1: at V = 10 the propensity of
// 2A -> B was k·n(n − V)/V, zero at n = V and negative below it.
//
// j/V is the same division the SSA uses to store a count (ssa_simulator.cpp
// stores n as n/V), so at n = j the term is exactly 0 rather than an ulp off
// it. At V = 1 every term is c − j, unchanged for `.net` models.
static inline double
compute_species_factor_ssa(const std::vector<std::pair<int, int>> &multiplicities,
                           const double *conc, const std::vector<Species> &species_list) {
    double factor = 1.0;
    for (const auto &[si, count] : multiplicities) {
        const double c = conc[si];
        const double vf = species_list[si].volume_factor;
        const bool amount = species_list[si].amount_valued;
        for (int k = 0; k < count; ++k) {
            double term = k == 0 ? c : c - static_cast<double>(k) / vf;
            if (amount)
                term *= vf;
            factor *= term;
        }
    }
    return factor;
}

// A reaction whose rate parameter never resolved (issue #589). This used to
// return a propensity of 0.0, an ordinary value, so the reaction was dead for
// the whole run with nothing reporting it. ModelBuilder::build() refuses such a
// model, so a built one cannot get here; this is the backstop for any path that
// does. It is safe to throw on the ODE path because every CVODE callback that
// reaches this runs under guard_cvode_callback (src/cvode_simulator.cpp), which
// keeps the exception out of SUNDIALS' C frames and rethrows it afterwards.
[[noreturn]] static void throw_unresolved_rate_param(const Reaction &rxn, int param_idx0,
                                                     size_t n_params) {
    const std::string which = rxn.function_name.empty()
                                  ? std::string("a reaction with no rate name")
                                  : "the reaction with rate '" + rxn.function_name + "'";
    throw std::runtime_error(which + " has no resolvable rate parameter (index " +
                             std::to_string(param_idx0) + " of " + std::to_string(n_params) +
                             " parameters); refusing to evaluate its rate as 0.");
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
            throw_unresolved_rate_param(rxn, k_idx, params.size());

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
            // (#170) Read the compartment size from `params` when the loader could
            // name it: reactions are shared across model clones, so this factor
            // cannot be rescaled in place on a write. Same number at load ⇒ no-op.
            rate *= (rxn.ssa_volume_param_idx0 >= 0 &&
                     rxn.ssa_volume_param_idx0 < static_cast<int>(params.size()))
                        ? params[rxn.ssa_volume_param_idx0].value
                        : rxn.ssa_volume_factor;
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
            // A law evaluated as written holds n^m for a species it reads m
            // times; the SSA's count of distinct m-tuples is n(n−1)…(n−m+1).
            // Each factor (c − j/V)/c is (n − j)/n in storage units, formed with
            // the j/V a count is stored by, so it is exactly 0 at n = j. At
            // c = 0 the law's own c^m has made the rate 0 already.
            for (const auto &[si, m] : rxn.ssa_falling_factorial) {
                const double c = conc[si];
                if (c == 0.0)
                    continue;
                const double vf = species_list[si].volume_factor;
                // Stops at the first zero factor (j = n): past it every term
                // would only repeat the 0, m − n times over.
                for (int j = 1; j < m && rate != 0.0; ++j)
                    rate *= (c - static_cast<double>(j) / vf) / c;
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
            throw_unresolved_rate_param(rxn, -1, params.size());
        int kcat_idx = rxn.rate_law_param_indices[0] - 1;
        int km_idx = rxn.rate_law_param_indices[1] - 1;
        const int np = static_cast<int>(params.size());
        if (kcat_idx < 0 || kcat_idx >= np)
            throw_unresolved_rate_param(rxn, kcat_idx, params.size());
        if (km_idx < 0 || km_idx >= np)
            throw_unresolved_rate_param(rxn, km_idx, params.size());

        double kcat = params[kcat_idx].value;
        double Km = params[km_idx].value;

        if (rxn.reactant_indices.size() >= 2) {
            int e_si = rxn.reactant_indices[0] - 1;
            int s_si = rxn.reactant_indices[1] - 1;
            if (e_si >= 0 && e_si < n_species && s_si >= 0 && s_si < n_species) {
                double E = conc[e_si];
                double S = conc[s_si];

                // tQSSA: compute free substrate. mm_tqssa takes the conjugate
                // root when delta < 0 — writing it as ½(delta + D) there loses
                // ~2 digits per decade of |delta|/√(4·Km·S) (GH #89).
                MMTqssa q = mm_tqssa(Km, E, S);

                // GH #93: sFree is NOT clamped to 0. It goes negative exactly
                // when S does, and the tQSSA rate continues smoothly — and
                // restoringly — through there; the clamp used to contradict the
                // `sFree > 0` guard in mm_tqssa_derivatives and in every
                // emitter. The rate's own denominator is the only degeneracy.
                // See the header note in include/bngsim/mm_jacobian.hpp.
                rate = q.KpsF > 0.0 ? kcat * rxn.stat_factor * q.sFree * E / q.KpsF : 0.0;
                // Issue #809 (review): a NaN or infinite Km, kcat, E or S fails
                // both guards above (`denom > 0` in mm_tqssa, `KpsF > 0` here)
                // and came out as a rate of exactly 0, so the reaction silently
                // stopped, under ODE and SSA alike. x − x is 0 for a finite x
                // and NaN otherwise, so this adds exactly 0.0 to a finite
                // rate and hands a non-finite input on to the callers that
                // refuse it. Every emitter adds the same term.
                rate += (kcat - kcat) + (Km - Km) + (E - E) + (S - S);
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
    // and dev/notes/gh106_rateof_kickoff.md — unless a derivative reads the
    // rateOf of a species whose own derivative reads one: each such level
    // takes a probe more (rateof_flow_nesting()). No-op when !uses_rateof.
    if (impl_->uses_rateof) {
        const int probes = rateof_flow_nesting();
        if (!impl_->rateof_ring.empty()) {
            // A ring of derivatives on each other's rateOf has no value to
            // read (rateof_ring_refusal, which whoever runs the model raises
            // first). Not a number, so that a caller who did not ask is
            // stopped by its solver and not answered.
            std::fill(derivs, derivs + n_species(), std::numeric_limits<double>::quiet_NaN());
            return;
        }
        for (int pass = probes; pass >= 0; --pass) {
            refresh_rateof_derivs_once_(t, conc);
        }
    }
    compute_derivs_core(t, conc, derivs);
}

bool NetworkModel::uses_rateof() const { return impl_->uses_rateof; }

bool NetworkModel::functions_use_time() const { return impl_->functions_use_time; }

void NetworkModel::refresh_rateof_derivs(double t, const double *conc) {
    if (!impl_->uses_rateof) {
        return;
    }
    // A probe reads the buffer the probe before it left, so a rateOf whose
    // species' derivative reads another rateOf is exact one probe later.
    for (int pass = rateof_nesting(); pass >= 0; --pass) {
        refresh_rateof_derivs_once_(t, conc);
    }
}

void NetworkModel::refresh_rateof_derivs_once_(double t, const double *conc) {
    if (!impl_->uses_rateof) {
        return;
    }
    // Probe into a separate scratch so compute_derivs_core never reads the
    // accessor source (current_derivs) mid-accumulation, then publish the whole
    // vector atomically. The probe itself reads whatever current_derivs holds
    // (stale). Where a rateOf argument is a species whose derivative reads no
    // rateOf, the entries the probe produces for it are exact regardless; a
    // derivative that does read one is exact a probe later, which is why the
    // callers make one more for each level of nesting.
    // (As a probe, the last of them too. Its rate laws read what the probe
    // before it left, and for a species nothing is counted a probe deep for,
    // that is the call before: zeros in a model first asked, the rates of
    // before an event after one. A catalyst is such a species. Given zero
    // times a law that is infinite for that alone, the rate of change a row
    // or a trigger read of it was not a number, and `at (rateOf(E) < -0.1)`
    // fired on the instant of another event. A rate that is not finite at
    // the model's own values fails the run in compute_derivs.)
    compute_derivs_core(t, conc, impl_->rateof_scratch.data(), true);
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

void NetworkModel::compute_derivs_core(double t, const double *conc, double *derivs, bool probe) {
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
            // One update per species, `net * term` (issue #801), by the net
            // change of each and not one side after the other (see
            // Reaction::net_loss; a net of zero is `0 * term`, nothing unless
            // the rate is not finite). For a species on one side alone it is
            // the statement it was, bit for bit.
            for (const auto &[si, m] : rxn.net_loss) {
                if (si < ns && (m != 0.0 || !probe)) {
                    const double term = rate / species_divisor(si);
                    derivs[si] -= m == 1.0 ? term : m * term;
                }
            }
            for (const auto &[si, m] : rxn.net_gain) {
                if (si < ns) {
                    const double term = rate / species_divisor(si);
                    derivs[si] += m == 1.0 ? term : m * term;
                }
            }
            continue;
        }

        // Each species' net change times the rate: one update per species, not
        // one per unit of stoichiometry (issue #801: an SBML coefficient of 1e6
        // was a million updates per evaluation, and their rounding drift), and
        // not one per side. A species on both sides of the reaction in equal
        // numbers gets `0 * rate`: `(x - rate) + rate` left the rounding of the
        // rate in a catalyst's derivative, 0 for a synthesis of 0.01 beside a
        // rate of 1e15. Zero times a finite rate is nothing; times a rate that
        // is not finite it is NaN, which fails the run as it did. For a species
        // on one side alone this is the statement it was, bit for bit.
        // (On a probe a net of zero is left alone: the rate law reads rateOfs
        // the probes have not filled yet, and may be infinite for that alone.)
        for (const auto &[si, m] : rxn.net_loss) {
            if (si < ns) {
                if (m == 1.0) {
                    derivs[si] -= rate;
                } else if (m != 0.0 || !probe) {
                    derivs[si] -= m * rate;
                }
            }
        }
        for (const auto &[si, m] : rxn.net_gain) {
            if (si < ns) {
                derivs[si] += m == 1.0 ? rate : m * rate;
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

void NetworkModel::compute_flux_split(double t, const double *conc, const std::vector<int> *rxns,
                                      double *net, double *gross) {
    const int ns = n_species();
    const int nr = n_reactions();

    std::memset(net, 0, ns * sizeof(double));
    if (gross != nullptr) {
        std::memset(gross, 0, ns * sizeof(double));
    }

    if (impl_->has_functions) {
        update_observables(conc);
        evaluate_functions(t);
    } else {
        impl_->current_time = t;
    }

    // compute_derivs_core's loop term for term, over the subset, summing the
    // signed term into `net` and |term| into `gross`.
    const auto &species_list = impl_->species;
    const int n_loop = rxns != nullptr ? static_cast<int>(rxns->size()) : nr;
    for (int k = 0; k < n_loop; ++k) {
        const int r = rxns != nullptr ? (*rxns)[static_cast<std::size_t>(k)] : k;
        if (r < 0 || r >= nr)
            continue;
        const auto &rxn = impl_->shared->reactions[r];
        const double rate = compute_rxn_rate(rxn, impl_->parameters, conc, ns, species_list, false);
        if (rate == 0.0)
            continue;

        auto divisor = [&](int si) -> double {
            if (!rxn.per_species_volume_scaling)
                return 1.0;
            int lv = species_list[si].ode_live_volume_idx0;
            if (lv >= 0 && lv < ns) {
                double v_live = conc[lv];
                if (v_live > 0.0)
                    return v_live;
            }
            return species_list[si].volume_factor;
        };
        for (const auto &[si, m] : rxn.net_loss) {
            if (si < ns) {
                const double term = m * rate / divisor(si);
                net[si] -= term;
                if (gross != nullptr) {
                    gross[si] += std::fabs(term);
                }
            }
        }
        for (const auto &[si, m] : rxn.net_gain) {
            if (si < ns) {
                const double term = m * rate / divisor(si);
                net[si] += term;
                if (gross != nullptr) {
                    gross[si] += std::fabs(term);
                }
            }
        }
    }

    for (const auto &s : impl_->species) {
        if (s.fixed) {
            net[s.index - 1] = 0.0;
            if (gross != nullptr) {
                gross[s.index - 1] = 0.0;
            }
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

void NetworkModel::compute_propensities(double t, const double *conc, double *out) {
    // The same refresh the SSA loop's sync_state runs before its propensity
    // pass, under the gate compute_derivs_core applies to the RHS: a pure
    // mass-action model reads no observable and binds no function, so both
    // passes would be dead work there (and evaluate_functions early-returns
    // anyway). A Functional rate law reads its function-bound parameter, which
    // is only current once the observables it sums are.
    if (impl_->has_functions) {
        update_observables(conc);
        evaluate_functions(t);
    } else {
        impl_->current_time = t;
    }
    const int nr = n_reactions();
    for (int r = 0; r < nr; ++r) {
        out[r] = compute_propensity(r, conc);
    }
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

    // Written in "C" (issue #1235): under a decimal-comma locale "%.17g" wrote
    // 0,5 into the C source, which cc and MIR read as two arguments.
    auto lit = [](double v) {
        char buf[40];
        c_snprintf(buf, sizeof(buf), "%.17g", v);
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
            //
            // (#170) …except that the propensity volume is not structural: it is
            // the compartment size, which `set_param` moves. This source is
            // generated once, when the SSA propensity library is prepared, so a
            // volume folded into the literal here would go stale on the first
            // write — the interpreted propensity reads it from `params` and the
            // JIT'd one would not, which is exactly the two-backends-disagree
            // failure issue #164 refused over. Emit it as its own `p[]` read,
            // parenthesised with the stat factor so the product is formed from
            // the same two doubles in the same order the fold used and the
            // emitted propensity is unchanged at the nominal point.
            const bool live_vol = rxn.ssa_volume_param_idx0 >= 0 &&
                                  rxn.ssa_volume_param_idx0 < static_cast<int>(params.size());
            const double C = live_vol ? rxn.stat_factor : rxn.stat_factor * rxn.ssa_volume_factor;
            body +=
                "  a[" + std::to_string(r) + "] = p[" + std::to_string(rxn.rate_param_idx0) + "]";
            if (live_vol)
                body +=
                    " * (" + lit(C) + " * p[" + std::to_string(rxn.ssa_volume_param_idx0) + "])";
            else if (C != 1.0)
                body += " * " + lit(C);
            // The species factor mirrors compute_species_factor_ssa: a falling
            // factorial over the count, ∏(x − j/V), each term times V for an
            // amount-valued species (issue #692). V is the species' compartment
            // size, and like the propensity volume above it is a `p[]` read
            // wherever the loader named its parameter: a literal would keep the
            // load-time size after a write while the interpreted path followed
            // it (issue #723). A species with no volume parameter cannot be
            // resized, so its literal is exact, and at V = 1 it is omitted,
            // which leaves the `.net` source byte-identical.
            for (const auto &mult : rxn.reactant_multiplicities) {
                const int si = mult.first;
                const int count = mult.second;
                const Species &sp = species[si];
                const bool amount = sp.amount_valued;
                std::string vol;
                if (sp.volume_param_idx0 >= 0 &&
                    sp.volume_param_idx0 < static_cast<int>(params.size()))
                    vol = "p[" + std::to_string(sp.volume_param_idx0) + "]";
                else if (sp.volume_factor != 1.0)
                    vol = lit(sp.volume_factor);
                const std::string x = "x[" + std::to_string(si) + "]";
                for (int m = 0; m < count; ++m) {
                    std::string term;
                    if (m == 0)
                        term = x;
                    else if (vol.empty())
                        term = "(" + x + " - " + lit(static_cast<double>(m)) + ")";
                    else // "m.0": `lit` prints 2.0 as "2", and "2 / 10" is integer 0 in C
                        term = "(" + x + " - " + std::to_string(m) + ".0 / " + vol + ")";
                    if (amount && !vol.empty())
                        term = "(" + term + " * " + vol + ")";
                    body += " * " + term;
                }
            }
            body += ";\n";
            continue;
        }

        // MichaelisMenten tQSSA — mirrors compute_rxn_rate's SSA MM branch exactly
        // (raw conc for E/S, no ssa_volume_factor): kcat=p[k0], Km=p[km0],
        // E=x[e0], S=x[s0]; sFree = root of x²−d·x−Km·S with d = S−Km−E, taken
        // through mm_tqssa's stable branch (GH #89), unclamped and guarded on its
        // own denominator (GH #93). SSA populations are non-negative, so the
        // negative-sFree branch is unreachable here — but the two must stay
        // mirrors, which is the property #93 was filed about.
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
            body += "    double D = sqrt(d * d + 4.0 * Km * S);\n";
            body += "    double sf = (d >= 0.0) ? 0.5 * (d + D)\n";
            body +=
                "                           : ((D - d) > 0.0 ? 2.0 * Km * S / (D - d) : 0.0);\n";
            body += "    double den = Km + sf;\n";
            body +=
                "    a[" + std::to_string(r) + "] = (den > 0.0 ? p[" + std::to_string(kcat0) + "]";
            if (rxn.stat_factor != 1.0)
                body += " * " + lit(rxn.stat_factor);
            // Non-finite inputs propagate, as in compute_rxn_rate (issue #809).
            body += " * sf * E / den : 0.0) + ((p[" + std::to_string(kcat0) + "] - p[" +
                    std::to_string(kcat0) + "]) + (Km - Km) + (E - E) + (S - S)); }\n";
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

// A second table under a name the model already has was appended beside the
// first (issue #772): the evaluator kept the first `tfun_<name>`, so the rate
// laws read the old data, while table_function_names listed the name twice and
// codegen, resolving names last-wins, reported the new one. A table is now one
// per name, and new data under a table's name goes into that table, which the
// evaluator and compiled code already call. A function's
// body cannot be replaced, so a function's name is refused before anything
// changes. The one table that takes a function's name is a `.net` tfun, which
// ModelBuilder registers as the body of the function it is declared in
// (`backs_a_function`).
TableFunction *NetworkModel::table_named_(const std::string &name) const {
    for (const auto &tf : impl_->table_functions)
        if (tf->name() == name)
            return tf.get();
    return nullptr;
}

// The function whose body calls the table `name`: the one a `.net` tfun is
// written in, which names a whole-body table and is the `<func>` of an
// embedded `<func>__tfun<k>`. Empty for a table nothing calls.
std::string NetworkModel::function_calling_table_(const std::string &name) const {
    const std::string call = "tfun_" + name + "()";
    for (const auto &f : impl_->functions) {
        for (auto at = f.expression.find(call); at != std::string::npos;
             at = f.expression.find(call, at + 1)) {
            const char before = at == 0 ? ' ' : f.expression[at - 1];
            if (std::isalnum(static_cast<unsigned char>(before)) == 0 && before != '_')
                return f.name;
        }
    }
    return "";
}

void NetworkModel::require_new_table_name_(const std::string &name, bool backs_a_function) const {
    if (!backs_a_function && impl_->shared && impl_->shared->function_name_to_idx.count(name) != 0)
        throw std::invalid_argument("the model already has a function named '" + name +
                                    "', and a table cannot replace a function's body; add "
                                    "the table under a new name");
}

// What a table is read over is compiled into the model's code (_tfun_call_map
// writes `t`, `p[i]`, `obs[i]` or `func[i]` into each call), so new data has to
// be read over the same thing. Spelling may differ as it may for any index:
// `Time` and `t` are the clock, and `k()` is `k`.
void NetworkModel::require_same_index_(const TableFunction &held,
                                       const std::string &index_name) const {
    const bool clock = is_time_index(held.index_name());
    const bool same = clock
                          ? is_time_index(index_name)
                          : !is_time_index(index_name) && strip_paren_suffix(index_name) ==
                                                              strip_paren_suffix(held.index_name());
    if (!same) {
        const std::string over = clock ? "time" : strip_paren_suffix(held.index_name());
        throw std::invalid_argument("table function '" + held.name() + "' is read over '" + over +
                                    "', and what it is read over is compiled into the model's "
                                    "code; its new data has to be read over '" +
                                    over + "' too, not '" + index_name + "'");
    }
}

// Register `tf` and keep it. A table the model reads (one a model source
// declares) and that is read on the clock makes every function that calls it
// move with time (issue #654); set only once the evaluator has the table, so a
// name it refuses leaves the model as it was. A table added after load is read
// by nothing, and marked it as reading the time all the same, which since #710
// refused its steady_state().
void NetworkModel::keep_table_function_(std::unique_ptr<TableFunction> tf, bool backs_a_function) {
    register_table_function_(*tf);
    tf->set_read_by_model(backs_a_function);
    if (backs_a_function && is_time_index(tf->index_name()))
        impl_->functions_use_time = true;
    impl_->table_functions.push_back(std::move(tf));
}

void NetworkModel::add_table_function(const std::string &name, const std::string &filepath,
                                      const std::string &index_name, const std::string &method,
                                      const std::string &header_name) {
    add_table_function_(name, filepath, index_name, method, header_name,
                        /*backs_a_function=*/false);
}

void NetworkModel::add_table_function(const std::string &name, const std::vector<double> &xs,
                                      const std::vector<double> &ys, const std::string &index_name,
                                      const std::string &method) {
    add_table_function_(name, xs, ys, index_name, method, /*backs_a_function=*/false);
}

void NetworkModel::add_table_function_(const std::string &name, const std::string &filepath,
                                       const std::string &index_name, const std::string &method,
                                       const std::string &header_name, bool backs_a_function) {
    if (TableFunction *held = table_named_(name)) {
        const InterpolationMethod interpolation = method_for_(held, method, name);
        require_replaceable_(*held, index_name, backs_a_function);
        // New data is built and checked (knots, values, file and header)
        // before the table takes it, so a refusal leaves the table as it was.
        // Only the data and the method move: the evaluator's callable, the
        // compiled code's tf_id, the index binding and the reader stay. The
        // file's value column may be labelled with the table's name or, as a
        // model's own file is, with the function the table is written in.
        held->take_data_of(TableFunction::from_file(name, filepath, held->index_name(),
                                                    interpolation, name,
                                                    function_calling_table_(name)));
        return;
    }
    const InterpolationMethod interpolation = method_for_(nullptr, method, name);
    require_new_table_name_(name, backs_a_function);
    keep_table_function_(std::make_unique<TableFunction>(TableFunction::from_file(
                             name, filepath, index_name, interpolation, header_name)),
                         backs_a_function);
}

void NetworkModel::add_table_function_(const std::string &name, const std::vector<double> &xs,
                                       const std::vector<double> &ys, const std::string &index_name,
                                       const std::string &method, bool backs_a_function) {
    if (TableFunction *held = table_named_(name)) {
        const InterpolationMethod interpolation = method_for_(held, method, name);
        require_replaceable_(*held, index_name, backs_a_function);
        held->take_data_of(TableFunction(name, xs, ys, held->index_name(), interpolation));
        return;
    }
    const InterpolationMethod interpolation = method_for_(nullptr, method, name);
    require_new_table_name_(name, backs_a_function);
    keep_table_function_(std::make_unique<TableFunction>(name, xs, ys, index_name, interpolation),
                         backs_a_function);
}

// The interpolation of new data: the one asked for, or with none ("") the
// table's own. A `.tfun` file carries no method, so new data for a `.net`
// table declared `method=>"step"` read from a file stayed a step only if this
// kept it; a default of "linear" made it linear, and the run changed with no
// word. A new table with none is linear, as a `.net` tfun is.
InterpolationMethod NetworkModel::method_for_(const TableFunction *held, const std::string &method,
                                              const std::string &name) const {
    if (method.empty() && held != nullptr)
        return held->method();
    return parse_interpolation_method(method, name);
}

// ModelBuilder registers each of a source's tables once; two under one name in
// a source are two declarations of one name, not new data.
void NetworkModel::require_replaceable_(const TableFunction &held, const std::string &index_name,
                                        bool backs_a_function) const {
    if (backs_a_function)
        throw std::invalid_argument("table function '" + held.name() +
                                    "' is declared twice; a model holds one table per name");
    require_same_index_(held, index_name);
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

std::vector<double> NetworkModel::time_table_knots() const {
    std::vector<double> out;
    for (const auto &tf : impl_->table_functions)
        if (tf->read_by_model() && is_time_index(tf->index_name()))
            out.insert(out.end(), tf->xs().begin(), tf->xs().end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<double> NetworkModel::evaluate_over_times(const std::string &expression,
                                                      const std::vector<double> &times,
                                                      double shift) const {
    if (!impl_->scan_shift_defined) {
        impl_->evaluator->define_variable("bngsim_scan_shift_", &impl_->scan_shift);
        impl_->scan_shift_defined = true;
    }
    impl_->scan_shift = shift;
    auto it = impl_->clock_expression_cache.find(expression);
    if (it == impl_->clock_expression_cache.end()) {
        it =
            impl_->clock_expression_cache.emplace(expression, impl_->evaluator->compile(expression))
                .first;
    }
    // The clock is the model's own, which the evaluator's time() and every
    // time-indexed table read through a pointer, so it is put back whatever
    // happens below. A builtin's non-finite value at a time of the caller's
    // choosing is not the run's, so it is not reported either.
    struct Restore {
        double *clock;
        double saved;
        ExprTkEvaluator *evaluator;
        ~Restore() {
            *clock = saved;
            evaluator->set_nonfinite_warnings_muted(false);
        }
    } restore{&impl_->current_time, impl_->current_time, impl_->evaluator.get()};
    impl_->evaluator->set_nonfinite_warnings_muted(true);
    std::vector<double> out(times.size());
    for (size_t i = 0; i < times.size(); ++i) {
        impl_->current_time = times[i];
        out[i] = impl_->evaluator->evaluate(it->second);
    }
    return out;
}

std::vector<TableFunctionSpec> NetworkModel::table_function_specs() const {
    std::vector<TableFunctionSpec> specs;
    specs.reserve(impl_->table_functions.size());
    for (const auto &tf : impl_->table_functions) {
        TableFunctionSpec spec;
        spec.name = tf->name();
        spec.step = tf->method() == InterpolationMethod::Step;
        resolve_table_index(*tf, *impl_->shared, spec);
        spec.method = tf->method() == InterpolationMethod::Step ? "step" : "linear";
        spec.xs = tf->xs();
        spec.ys = tf->ys();
        spec.read = tf->read_by_model();
        specs.push_back(std::move(spec));
    }
    return specs;
}

} // namespace bngsim
