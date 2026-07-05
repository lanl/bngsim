// bngsim/src/model_builder.cpp — Programmatic model construction
//
// Implements ModelBuilder: the SOLE model construction API. All input
// formats (.net, Antimony, SBML, programmatic Python) go through
// ModelBuilder → build(). The build() method runs the full setup pipeline:
// ExprTk evaluator, stoichiometry, Jacobian sparsity, analytical Jacobian,
// graph coloring, SSA precompute.
//
// Used by NetFileLoader and other front ends so every model input shares the
// same build pipeline. Also carries .net-specific metadata such as tfun specs,
// net_file_dir, and species-parameter references.

#include "bngsim/model_builder.hpp"
#include "bngsim/expression.hpp"
#include "bngsim/types.hpp"
#include "model_impl.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bngsim {

// ─── BuilderImpl ─────────────────────────────────────────────────────────────

struct ModelBuilder::BuilderImpl {
    std::vector<Parameter> parameters;
    std::vector<Species> species;
    std::vector<Observable> observables;
    std::vector<Function> functions;
    std::vector<Reaction> reactions;

    std::unordered_map<std::string, int> param_name_to_idx;
    std::unordered_map<std::string, int> species_name_to_idx;
    std::unordered_map<std::string, int> observable_name_to_idx;
    std::unordered_map<std::string, int> function_name_to_idx;

    // Whether build() runs conservation-law detection (GH #102). The detector
    // does dense O(n_species^3) Gaussian elimination on the stoichiometry
    // matrix, which is intractable for very large networks (~100K species) and
    // is only consumed by the steady-state solver — the ODE/SSA integration
    // paths never read it. Front ends that build large ODE-only models can opt
    // out so setup stays O(reactions). Default true preserves every existing
    // caller's behavior.
    bool compute_conservation_laws = true;

    // SBML rateOf csymbol support (GH #106). Set by enable_rateof() when the
    // loader detects any rateOf(species) reference. build() then sizes the live
    // current_derivs buffer and registers the rate_of__<species> accessors.
    bool enable_rateof = false;

    // .net-specific support
    std::string net_file_dir;

    struct SpeciesParamRef {
        int species_idx0;
        std::string param_name;
    };
    std::vector<SpeciesParamRef> species_param_refs;

    struct EventSpec {
        std::string id;
        std::string trigger_expr;
        std::vector<std::pair<int, std::string>> assignments; // (species_idx0, value_expr)
        std::vector<bool> assignment_ode_only;                // GH #81; parallel to assignments
        double delay = 0.0;
        std::string delay_expr; // optional; takes precedence when non-empty
        int priority = 0;
        std::string priority_expr; // optional; takes precedence when non-empty
        bool persistent = true;
        bool initial_value = true;
        bool use_values_from_trigger_time = true;
    };
    std::vector<EventSpec> event_specs;

    // Discontinuity-trigger condition strings (GH #72), insertion-ordered and
    // de-duplicated. Compiled to evaluator expression indices in build().
    std::vector<std::string> discontinuity_trigger_specs;

    struct TfunSpec {
        std::string func_name;
        std::string header_name; // file column-2 validation name; empty falls back to func_name
        std::string filepath;    // empty for inline mode
        std::string index_name;
        std::string method; // "linear" or "step"
        // Inline data (populated when filepath is empty)
        std::vector<double> xs;
        std::vector<double> ys;
        bool is_inline = false;
    };
    std::vector<TfunSpec> tfun_specs;
};

// ─── Constructor / destructor / move ─────────────────────────────────────────

ModelBuilder::ModelBuilder() : bimpl_(std::make_unique<BuilderImpl>()) {}
ModelBuilder::~ModelBuilder() = default;
ModelBuilder::ModelBuilder(ModelBuilder &&) noexcept = default;
ModelBuilder &ModelBuilder::operator=(ModelBuilder &&) noexcept = default;

// ─── Add model elements ──────────────────────────────────────────────────────

int ModelBuilder::add_parameter(const std::string &name, double value,
                                const std::string &expression, bool is_expression) {
    int idx = static_cast<int>(bimpl_->parameters.size());
    Parameter p;
    p.index = idx + 1; // 1-based for .net compatibility
    p.name = name;
    p.value = value;
    p.expression = expression;
    p.is_expression = is_expression;
    p.evaluator_id = -1;

    bimpl_->param_name_to_idx[name] = idx;
    bimpl_->parameters.push_back(std::move(p));
    return idx;
}

int ModelBuilder::add_species(const std::string &name, double init_conc, bool fixed,
                              double volume_factor, bool amount_valued, bool reported) {
    int idx = static_cast<int>(bimpl_->species.size());
    Species s;
    s.index = idx + 1; // 1-based
    s.name = name;
    s.concentration = init_conc;
    s.initial_conc = init_conc;
    s.fixed = fixed;
    s.volume_factor = volume_factor;
    s.amount_valued = amount_valued;
    s.reported = reported;

    bimpl_->species_name_to_idx[name] = idx;
    bimpl_->species.push_back(std::move(s));
    return idx;
}

int ModelBuilder::add_observable(const std::string &name,
                                 const std::vector<std::pair<int, double>> &entries) {
    int idx = static_cast<int>(bimpl_->observables.size());
    Observable obs;
    obs.index = idx + 1;
    obs.name = name;
    obs.total = 0.0;

    for (const auto &[sp_idx0, factor] : entries) {
        GroupEntry ge;
        ge.species_index = sp_idx0 + 1; // convert 0-based → 1-based
        ge.factor = factor;
        obs.entries.push_back(ge);
    }

    bimpl_->observable_name_to_idx[name] = idx;
    bimpl_->observables.push_back(std::move(obs));
    return idx;
}

int ModelBuilder::add_function(const std::string &name, const std::string &expression) {
    int idx = static_cast<int>(bimpl_->functions.size());
    Function func;
    func.index = idx + 1;
    func.name = name;
    func.expression = expression;
    func.evaluator_id = -1;

    bimpl_->function_name_to_idx[name] = idx;
    bimpl_->functions.push_back(std::move(func));
    return idx;
}

int ModelBuilder::add_reaction(const std::vector<int> &reactants, const std::vector<int> &products,
                               RateLawType type, const std::string &rate_law, double stat_factor,
                               bool apply_species_factor, double ssa_volume_factor,
                               bool per_species_volume_scaling, bool is_rate_rule_ode,
                               int ssa_live_volume_idx0, double ssa_live_volume_exp,
                               bool ode_only) {
    int idx = static_cast<int>(bimpl_->reactions.size());
    Reaction rxn;
    rxn.index = idx + 1;
    rxn.rate_law_type = type;
    rxn.stat_factor = stat_factor;
    rxn.apply_species_factor = apply_species_factor;
    rxn.ssa_volume_factor = ssa_volume_factor;
    rxn.per_species_volume_scaling = per_species_volume_scaling;
    rxn.is_rate_rule_ode = is_rate_rule_ode;
    rxn.ssa_live_volume_idx0 = ssa_live_volume_idx0;
    rxn.ssa_live_volume_exp = ssa_live_volume_exp;
    rxn.ode_only = ode_only;

    // Convert 0-based → 1-based species indices
    for (int ri : reactants) {
        rxn.reactant_indices.push_back(ri + 1);
    }
    for (int pi : products) {
        rxn.product_indices.push_back(pi + 1);
    }

    // Resolve rate law reference
    if (type == RateLawType::Functional) {
        rxn.function_name = rate_law;
        // Will resolve param index in build()
        rxn.rate_law_param_indices.push_back(-1);
    } else if (type == RateLawType::Elementary) {
        rxn.function_name = rate_law;
        auto it = bimpl_->param_name_to_idx.find(rate_law);
        if (it != bimpl_->param_name_to_idx.end()) {
            rxn.rate_law_param_indices.push_back(bimpl_->parameters[it->second].index);
        } else {
            rxn.rate_law_param_indices.push_back(-1);
        }
    } else if (type == RateLawType::MichaelisMenten) {
        // For MM, rate_law should be "kcat_name,km_name"
        auto comma = rate_law.find(',');
        if (comma != std::string::npos) {
            std::string kcat_name = rate_law.substr(0, comma);
            std::string km_name = rate_law.substr(comma + 1);
            auto k1 = bimpl_->param_name_to_idx.find(kcat_name);
            auto k2 = bimpl_->param_name_to_idx.find(km_name);
            if (k1 != bimpl_->param_name_to_idx.end())
                rxn.rate_law_param_indices.push_back(bimpl_->parameters[k1->second].index);
            if (k2 != bimpl_->param_name_to_idx.end())
                rxn.rate_law_param_indices.push_back(bimpl_->parameters[k2->second].index);
        }
    }

    bimpl_->reactions.push_back(std::move(rxn));
    return idx;
}

void ModelBuilder::set_reaction_live_volume(int rxn_idx0, int ssa_live_volume_idx0,
                                            double ssa_live_volume_exp) {
    if (rxn_idx0 < 0 || rxn_idx0 >= static_cast<int>(bimpl_->reactions.size()))
        return;
    bimpl_->reactions[rxn_idx0].ssa_live_volume_idx0 = ssa_live_volume_idx0;
    bimpl_->reactions[rxn_idx0].ssa_live_volume_exp = ssa_live_volume_exp;
}

void ModelBuilder::set_species_ode_live_volume(int species_idx0, int live_idx0) {
    if (species_idx0 < 0 || species_idx0 >= static_cast<int>(bimpl_->species.size()))
        return;
    bimpl_->species[species_idx0].ode_live_volume_idx0 = live_idx0;
}

void ModelBuilder::set_species_rateof_amount(int species_idx0) {
    if (species_idx0 < 0 || species_idx0 >= static_cast<int>(bimpl_->species.size()))
        return;
    bimpl_->species[species_idx0].report_rateof_amount = true;
}

void ModelBuilder::add_reaction_live_volume_term(int rxn_idx0, int live_idx0, double v_static,
                                                 double exp) {
    if (rxn_idx0 < 0 || rxn_idx0 >= static_cast<int>(bimpl_->reactions.size()))
        return;
    bimpl_->reactions[rxn_idx0].ssa_live_volume_terms.push_back({live_idx0, v_static, exp});
}

// ─── .net-specific support ───────────────────────────────────────────────────

void ModelBuilder::set_compute_conservation_laws(bool enabled) {
    bimpl_->compute_conservation_laws = enabled;
}

void ModelBuilder::set_net_file_dir(const std::string &dir) { bimpl_->net_file_dir = dir; }

void ModelBuilder::enable_rateof() { bimpl_->enable_rateof = true; }

void ModelBuilder::add_species_param_ref(int species_idx0, const std::string &param_name) {
    bimpl_->species_param_refs.push_back({species_idx0, param_name});
}

void ModelBuilder::add_table_function_spec(const std::string &func_name,
                                           const std::string &filepath,
                                           const std::string &index_name, const std::string &method,
                                           const std::string &header_name) {
    BuilderImpl::TfunSpec spec;
    spec.func_name = func_name;
    spec.header_name = header_name;
    spec.filepath = filepath;
    spec.index_name = index_name;
    spec.method = method;
    spec.is_inline = false;
    bimpl_->tfun_specs.push_back(std::move(spec));
}

// ─── Events ──────────────────────────────────────────────────────────────────

void ModelBuilder::add_event(const std::string &id, const std::string &trigger_expr,
                             const std::vector<std::pair<int, std::string>> &assignments,
                             double delay, int priority, bool persistent, bool initial_value,
                             bool use_values_from_trigger_time, const std::string &delay_expr,
                             const std::string &priority_expr,
                             const std::vector<bool> &assignment_ode_only) {
    BuilderImpl::EventSpec spec;
    spec.id = id;
    spec.trigger_expr = trigger_expr;
    spec.assignments = assignments;
    spec.assignment_ode_only = assignment_ode_only;
    spec.delay = delay;
    spec.delay_expr = delay_expr;
    spec.priority = priority;
    spec.priority_expr = priority_expr;
    spec.persistent = persistent;
    spec.initial_value = initial_value;
    spec.use_values_from_trigger_time = use_values_from_trigger_time;
    bimpl_->event_specs.push_back(std::move(spec));
}

void ModelBuilder::add_discontinuity_trigger(const std::string &condition_expr) {
    auto &specs = bimpl_->discontinuity_trigger_specs;
    // De-duplicate: the same threshold can appear in several rate laws /
    // assignment rules, and one root per distinct crossing is enough.
    if (std::find(specs.begin(), specs.end(), condition_expr) != specs.end()) {
        return;
    }
    specs.push_back(condition_expr);
}

void ModelBuilder::add_inline_table_function_spec(const std::string &func_name,
                                                  const std::vector<double> &xs,
                                                  const std::vector<double> &ys,
                                                  const std::string &index_name,
                                                  const std::string &method) {
    BuilderImpl::TfunSpec spec;
    spec.func_name = func_name;
    spec.index_name = index_name;
    spec.method = method;
    spec.xs = xs;
    spec.ys = ys;
    spec.is_inline = true;
    bimpl_->tfun_specs.push_back(std::move(spec));
}

// ─── Build pipeline helper functions ─────────────────────────────────────────

namespace {

// Build stoichiometry from reactions (same logic as net_file_loader.cpp)
std::vector<StoichEntry> build_stoich(const std::vector<Reaction> &reactions) {
    std::vector<StoichEntry> entries;
    for (const auto &rxn : reactions) {
        std::unordered_map<int, double> net;
        for (int si : rxn.reactant_indices)
            net[si] -= 1.0;
        for (int si : rxn.product_indices)
            net[si] += 1.0;
        for (const auto &[si, coeff] : net) {
            if (coeff != 0.0)
                entries.push_back({si, rxn.index, coeff});
        }
    }
    return entries;
}

// Extract identifiers from expression string
std::vector<std::string> extract_ids(const std::string &expr) {
    std::vector<std::string> ids;
    size_t i = 0;
    while (i < expr.size()) {
        if (std::isalpha(expr[i]) || expr[i] == '_') {
            size_t start = i;
            while (i < expr.size() && (std::isalnum(expr[i]) || expr[i] == '_')) {
                i++;
            }
            ids.push_back(expr.substr(start, i - start));
        } else {
            i++;
        }
    }
    return ids;
}

// Build Jacobian sparsity (same algorithm as net_file_loader.cpp)
JacobianSparsity build_jac_sparsity(const std::vector<Reaction> &reactions, int n_species,
                                    const std::vector<Observable> &observables,
                                    const std::unordered_map<std::string, int> &obs_name_to_idx,
                                    const std::vector<Function> &functions,
                                    const std::unordered_map<std::string, int> &func_name_to_idx,
                                    const std::vector<Species> &species) {

    if (n_species == 0)
        return {};

    std::vector<std::vector<int>> obs_species(observables.size());
    for (int oi = 0; oi < static_cast<int>(observables.size()); ++oi) {
        for (const auto &entry : observables[oi].entries) {
            int si_0 = entry.species_index - 1;
            if (si_0 >= 0 && si_0 < n_species)
                obs_species[oi].push_back(si_0);
        }
    }

    // Transitive species dependencies per function (GH #164). A Functional rate
    // law is compiled over parameters + observables only; a function reference
    // resolves to that function's bound parameter. So the species a rate law
    // depends on flow in *exclusively* through observables — directly named in
    // the expression, or named in a function it references (transitively). This
    // precomputes, for each function, the union of obs_species over every
    // observable reachable from its expression, following function→function
    // references. It REPLACES the previous "a function reference means this rate
    // depends on ALL species" fallback, which densified the Jacobian sparsity
    // toward n_species² nonzeros (e.g. genome-scale models with ~37K rate laws
    // referencing assignment-rule functions → ~2.8B nonzeros / tens of GB; GH
    // #164). The resolved set is the exact dependency superset, so the analytical
    // and finite-difference Jacobians lose no nonzero — only the artificial
    // density. Iterative post-order DFS with an on-stack cycle guard keeps it
    // O(total expr size + total deps) and avoids deep recursion; SBML
    // assignment/rate-rule graphs are acyclic, the guard only protects against
    // malformed cyclic input.
    const int nf = static_cast<int>(functions.size());
    std::vector<std::vector<int>> func_direct_obs(nf); // species from observables in expr
    std::vector<std::vector<int>> func_refs(nf);       // functions referenced in expr
    for (int fi = 0; fi < nf; ++fi) {
        for (const auto &id : extract_ids(functions[fi].expression)) {
            auto oit = obs_name_to_idx.find(id);
            if (oit != obs_name_to_idx.end())
                for (int si_0 : obs_species[oit->second])
                    func_direct_obs[fi].push_back(si_0);
            auto fjt = func_name_to_idx.find(id);
            if (fjt != func_name_to_idx.end() && fjt->second != fi)
                func_refs[fi].push_back(fjt->second);
        }
    }
    std::vector<std::vector<int>> func_deps(nf); // transitive sorted-unique species
    {
        std::vector<char> fstate(nf, 0); // 0=unvisited, 1=on-stack, 2=done
        std::vector<int> stk;
        for (int s = 0; s < nf; ++s) {
            if (fstate[s] != 0)
                continue;
            stk.push_back(s);
            while (!stk.empty()) {
                int u = stk.back();
                if (fstate[u] == 0) {
                    fstate[u] = 1;
                    for (int v : func_refs[u])
                        if (fstate[v] == 0)
                            stk.push_back(v);
                } else if (fstate[u] == 1) {
                    std::vector<int> acc = func_direct_obs[u];
                    for (int v : func_refs[u])
                        if (fstate[v] == 2) // skip back-edges into the active path (cycles)
                            acc.insert(acc.end(), func_deps[v].begin(), func_deps[v].end());
                    std::sort(acc.begin(), acc.end());
                    acc.erase(std::unique(acc.begin(), acc.end()), acc.end());
                    func_deps[u] = std::move(acc);
                    fstate[u] = 2;
                    stk.pop_back();
                } else {
                    stk.pop_back();
                }
            }
        }
    }

    // Per-column row adjacency (GH #102). Sparse construction avoids the dense
    // n_species×n_species bool matrix the previous implementation allocated —
    // O(n^2) memory and time, intractable at ~100K species. Each column gathers
    // O(reactions touching it) rows (Elementary reactions and observable-scoped
    // functionals stay sparse), so build is O(total nonzeros). The CSC emitted
    // below is byte-identical to the dense path: rows sorted ascending and
    // de-duplicated per column.
    std::vector<std::vector<int>> col_rows(n_species);

    for (const auto &rxn : reactions) {
        std::unordered_map<int, double> net;
        for (int si : rxn.reactant_indices)
            net[si] -= 1.0;
        for (int si : rxn.product_indices)
            net[si] += 1.0;

        std::vector<int> affected;
        for (const auto &[si_1, coeff] : net) {
            if (coeff != 0.0) {
                int si_0 = si_1 - 1;
                if (si_0 >= 0 && si_0 < n_species)
                    affected.push_back(si_0);
            }
        }
        if (affected.empty())
            continue;

        std::vector<int> rate_deps;
        if (rxn.rate_law_type == RateLawType::Functional) {
            auto fit = func_name_to_idx.find(rxn.function_name);
            if (fit != func_name_to_idx.end()) {
                // Transitive species the rate law's value depends on (via the
                // observables it / its referenced functions read), plus the
                // reactants the engine's species factor scales by (GH #164).
                rate_deps = func_deps[fit->second];
                for (int ri : rxn.reactant_indices) {
                    int si_0 = ri - 1;
                    if (si_0 >= 0 && si_0 < n_species)
                        rate_deps.push_back(si_0);
                }
                std::sort(rate_deps.begin(), rate_deps.end());
                rate_deps.erase(std::unique(rate_deps.begin(), rate_deps.end()), rate_deps.end());
            } else {
                // Unknown rate-law function (malformed input): can't resolve its
                // dependencies, so fall back to the safe all-species pattern.
                rate_deps.reserve(n_species);
                for (int j = 0; j < n_species; ++j)
                    rate_deps.push_back(j);
            }
        } else {
            for (int ri : rxn.reactant_indices) {
                int si_0 = ri - 1;
                if (si_0 >= 0 && si_0 < n_species)
                    rate_deps.push_back(si_0);
            }
        }

        for (int i : affected)
            for (int j : rate_deps)
                col_rows[j].push_back(i);

        // GH #171: a cross-compartment variable-volume reaction divides each
        // affected species i's derivative by the LIVE compartment volume
        // conc[ode_live_volume_idx0] (a real ODE state — the promoted compartment
        // species). So ∂(dSᵢ/dt)/∂V_live = −(varvol RHS of i)/V_live is a
        // structural nonzero at (row i, col L). Without this edge the new column
        // falls outside the sparsity pattern and set_functional_jacobian rejects
        // it (reverting the whole model to FD). Guarded on
        // per_species_volume_scaling + a live divisor ⇒ the CSC is byte-identical
        // for every static-volume / .net model. The dedup below folds the
        // redundant edge when the rate law already reads V_live directly (the
        // explicit-compartment-factor case, GH #172).
        if (rxn.per_species_volume_scaling) {
            for (int i : affected) {
                if (i < 0 || i >= n_species)
                    continue;
                int L = species[i].ode_live_volume_idx0;
                if (L >= 0 && L < n_species)
                    col_rows[L].push_back(i);
            }
        }
    }

    // Sort + de-duplicate each column, then assemble CSC. Ascending row order
    // within a column matches the previous row-major scan, so downstream
    // consumers (analytical-Jacobian binary search, KLU) see identical input.
    JacobianSparsity sp;
    sp.n = n_species;
    sp.col_ptrs.resize(n_species + 1, 0);
    int64_t nnz = 0;
    for (int j = 0; j < n_species; ++j) {
        auto &rows = col_rows[j];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        nnz += static_cast<int64_t>(rows.size());
    }

    sp.nnz = static_cast<int>(nnz);
    sp.row_indices.resize(nnz);
    sp.density = (n_species > 0)
                     ? static_cast<double>(nnz) / (static_cast<double>(n_species) * n_species)
                     : 0.0;

    int64_t idx = 0;
    for (int j = 0; j < n_species; ++j) {
        sp.col_ptrs[j] = idx;
        for (int i : col_rows[j])
            sp.row_indices[idx++] = i;
    }
    sp.col_ptrs[n_species] = idx;
    return sp;
}

// Build analytical Jacobian (same as net_file_loader.cpp)
AnalyticalJacobianData build_anal_jac(const std::vector<Reaction> &reactions, int n_species,
                                      int n_params, const JacobianSparsity &sp,
                                      const std::vector<Species> &species) {

    AnalyticalJacobianData ajd;
    ajd.available = false;
    if (n_species == 0 || sp.empty())
        return ajd;

    // GH #76: Functional rate laws are no longer a hard gate. Their analytical
    // ∂v/∂x is supplied per-instance after build (set_functional_jacobian), so
    // here we only build the closed-form Elementary terms, push an inert
    // placeholder for each Functional reaction (keeping ajd.reactions aligned
    // one-per-reaction), and count them. A reaction type we still cannot cover
    // analytically (MichaelisMenten — handled by a later increment) clears the
    // structural-availability flag so the whole model uses the FD Jacobian.
    bool has_unsupported = false;
    int n_functional = 0;

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

    ajd.reactions.reserve(reactions.size());
    for (const auto &rxn : reactions) {
        if (rxn.rate_law_type == RateLawType::Functional) {
            // Covered by the per-instance functional path; inert placeholder
            // here keeps the per-reaction alignment of ajd.reactions.
            ++n_functional;
            ajd.reactions.emplace_back();
            continue;
        }
        if (rxn.rate_law_type == RateLawType::MichaelisMenten) {
            // MichaelisMenten (tQSSA) closed-form analytical Jacobian (GH #76
            // task 3). Inert placeholder keeps ajd.reactions one-per-reaction;
            // the contribution comes from ajd.mm_reactions. If the kcat/Km/E/S
            // indices cannot be resolved, fall back to FD for the whole model.
            ajd.reactions.emplace_back();
            bool built = false;
            if (rxn.reactant_indices.size() >= 2 && rxn.rate_law_param_indices.size() >= 2) {
                int e_i = rxn.reactant_indices[0] - 1;
                int s_i = rxn.reactant_indices[1] - 1;
                int kcat_i = rxn.rate_law_param_indices[0] - 1;
                int km_i = rxn.rate_law_param_indices[1] - 1;
                if (e_i >= 0 && e_i < n_species && s_i >= 0 && s_i < n_species && kcat_i >= 0 &&
                    kcat_i < n_params && km_i >= 0 && km_i < n_params) {
                    AnalyticalJacobianData::MMTerm mt;
                    mt.e_idx = e_i;
                    mt.s_idx = s_i;
                    mt.kcat_param_idx0 = kcat_i;
                    mt.km_param_idx0 = km_i;
                    mt.stat_factor = rxn.stat_factor;
                    std::unordered_map<int, double> net;
                    for (int si : rxn.reactant_indices) {
                        int s0 = si - 1;
                        if (s0 >= 0 && s0 < n_species)
                            net[s0] -= 1.0;
                    }
                    for (int si : rxn.product_indices) {
                        int s0 = si - 1;
                        if (s0 >= 0 && s0 < n_species)
                            net[s0] += 1.0;
                    }
                    bool ok = true;
                    for (const auto &[i, c] : net) {
                        if (c == 0.0)
                            continue;
                        int64_t csc_e = find_csc(i, e_i);
                        int64_t csc_s = find_csc(i, s_i);
                        if (csc_e < 0 || csc_s < 0) {
                            ok = false;
                            break;
                        }
                        mt.e_affected.emplace_back(csc_e, c);
                        mt.s_affected.emplace_back(csc_s, c);
                    }
                    if (ok) {
                        ajd.mm_reactions.push_back(std::move(mt));
                        built = true;
                    }
                }
            }
            if (!built)
                has_unsupported = true;
            continue;
        }
        if (rxn.rate_law_type != RateLawType::Elementary) {
            // Any other still-unsupported reaction type — FD for the whole model.
            has_unsupported = true;
            ajd.reactions.emplace_back();
            continue;
        }

        AnalyticalJacobianData::ReactionTerms terms;
        terms.stat_factor = rxn.stat_factor;
        // Amount-valued reactant product (GH #75): mirror the `x·V_c` reads the
        // RHS species factor performs for amount_valued reactants, as a single
        // constant multiplier on the rate (and thus on every ∂v_r/∂x_j).
        terms.amount_factor = 1.0;
        for (int si : rxn.reactant_indices) {
            int si_0 = si - 1;
            if (si_0 >= 0 && si_0 < n_species && species[si_0].amount_valued)
                terms.amount_factor *= species[si_0].volume_factor;
        }
        terms.rate_param_idx0 = -1;
        if (!rxn.rate_law_param_indices.empty()) {
            int k_idx = rxn.rate_law_param_indices[0] - 1;
            if (k_idx >= 0 && k_idx < n_params)
                terms.rate_param_idx0 = k_idx;
        }
        if (terms.rate_param_idx0 < 0 || terms.rate_param_idx0 >= n_params) {
            ajd.reactions.push_back(std::move(terms));
            continue;
        }

        std::unordered_map<int, double> net_stoich;
        for (int si : rxn.reactant_indices) {
            int si_0 = si - 1;
            if (si_0 >= 0 && si_0 < n_species)
                net_stoich[si_0] -= 1.0;
        }
        for (int si : rxn.product_indices) {
            int si_0 = si - 1;
            if (si_0 >= 0 && si_0 < n_species)
                net_stoich[si_0] += 1.0;
        }

        std::vector<std::pair<int, double>> affected;
        for (const auto &[si_0, coeff] : net_stoich)
            if (coeff != 0.0)
                affected.emplace_back(si_0, coeff);

        if (affected.empty()) {
            ajd.reactions.push_back(std::move(terms));
            continue;
        }

        std::unordered_map<int, int> mult_map;
        for (int ri : rxn.reactant_indices) {
            int si_0 = ri - 1;
            if (si_0 >= 0 && si_0 < n_species)
                mult_map[si_0]++;
        }

        for (const auto &[j, m_j] : mult_map) {
            AnalyticalJacobianData::PerReactant pr;
            pr.species_idx = j;
            pr.multiplicity = m_j;
            for (const auto &[other_j, other_m] : mult_map)
                if (other_j != j)
                    pr.others.push_back({other_j, other_m});
            for (const auto &[i, coeff] : affected) {
                int64_t data_idx = find_csc(i, j);
                if (data_idx >= 0)
                    pr.affected.emplace_back(data_idx, coeff);
            }
            terms.reactants.push_back(std::move(pr));
        }
        ajd.reactions.push_back(std::move(terms));
    }
    // Structurally available unless a still-unsupported reaction type appears.
    // n_functional > 0 does NOT clear this — the model-level completeness
    // predicate additionally requires the per-instance functional terms (see
    // NetworkModel::analytical_jacobian_complete).
    ajd.n_functional = n_functional;
    ajd.available = !has_unsupported;
    return ajd;
}

// Detect conservation laws via Gaussian elimination on S^T
//
// The left null space of the stoichiometry matrix S satisfies L * S = 0,
// meaning L * (dy/dt) = 0, so L * y = const for all time.
// We find L by computing the reduced row echelon form of S^T and extracting
// the null space from the non-pivot rows.
ConservationLaws detect_conservation_laws(const std::vector<Reaction> &reactions,
                                          const std::vector<Species> &species) {

    const int ns = static_cast<int>(species.size());
    const int nr = static_cast<int>(reactions.size());
    ConservationLaws cl;
    cl.n_species = ns;
    if (ns == 0 || nr == 0)
        return cl;

    // Build dense stoichiometry matrix S (ns × nr) stored row-major.
    // S[i][r] = net stoichiometric coefficient of species i in reaction r.
    std::vector<std::vector<double>> S(ns, std::vector<double>(nr, 0.0));
    for (int r = 0; r < nr; ++r) {
        const auto &rxn = reactions[r];
        for (int ri : rxn.reactant_indices) {
            int si = ri - 1;
            if (si >= 0 && si < ns)
                S[si][r] -= 1.0;
        }
        for (int pi : rxn.product_indices) {
            int si = pi - 1;
            if (si >= 0 && si < ns)
                S[si][r] += 1.0;
        }
    }

    // Skip fixed species rows — they don't participate in conservation.
    // Zero out their rows in S so they don't contribute.
    for (int i = 0; i < ns; ++i) {
        if (species[i].fixed) {
            for (int r = 0; r < nr; ++r)
                S[i][r] = 0.0;
        }
    }

    // Gaussian elimination with partial pivoting on S to find its rank
    // and the left null space. We work on S directly (ns rows, nr columns).
    // After RREF, rows that are all-zero correspond to conservation laws
    // in the original species space.
    //
    // We track row operations via an augmented identity: [S | I_ns].
    // The null space rows of S correspond to rows of the transformed I
    // where the S part is zero.

    // Augmented matrix: ns rows × (nr + ns) columns, row-major
    std::vector<std::vector<double>> aug(ns, std::vector<double>(nr + ns, 0.0));
    for (int i = 0; i < ns; ++i) {
        for (int r = 0; r < nr; ++r)
            aug[i][r] = S[i][r];
        aug[i][nr + i] = 1.0; // identity augmentation
    }

    const double tol = 1e-12;
    std::vector<int> pivot_row(nr, -1); // which row is pivot for each column
    int rank = 0;

    for (int col = 0; col < nr && rank < ns; ++col) {
        // Find pivot: largest absolute value in column col, rows [rank, ns)
        int best = -1;
        double best_val = 0.0;
        for (int i = rank; i < ns; ++i) {
            double v = std::abs(aug[i][col]);
            if (v > best_val) {
                best_val = v;
                best = i;
            }
        }
        if (best_val < tol)
            continue; // zero column, skip

        // Swap rows
        if (best != rank)
            std::swap(aug[best], aug[rank]);

        // Scale pivot row
        double pivot = aug[rank][col];
        for (int j = 0; j < nr + ns; ++j)
            aug[rank][j] /= pivot;

        // Eliminate all other rows
        for (int i = 0; i < ns; ++i) {
            if (i == rank)
                continue;
            double factor = aug[i][col];
            if (std::abs(factor) < tol)
                continue;
            for (int j = 0; j < nr + ns; ++j)
                aug[i][j] -= factor * aug[rank][j];
        }

        pivot_row[col] = rank;
        ++rank;
    }

    int n_laws = ns - rank;
    if (n_laws <= 0)
        return cl; // full rank, no conservation laws

    // Extract conservation law coefficients from zero rows of transformed S.
    // Row i is a zero row if all S columns are zero.
    cl.n_laws = n_laws;
    cl.coefficients.reserve(n_laws);
    cl.dependent.reserve(n_laws);

    // Track which species are already chosen as dependent (avoid collision)
    std::vector<bool> already_dep(ns, false);

    // Rows [0..rank-1] are pivot rows → independent species
    // Rows [rank..ns-1] are null space → conservation laws
    for (int i = rank; i < ns; ++i) {
        // Verify the S part is indeed zero
        bool all_zero = true;
        for (int r = 0; r < nr; ++r) {
            if (std::abs(aug[i][r]) > tol) {
                all_zero = false;
                break;
            }
        }
        if (!all_zero)
            continue; // numerical noise, skip

        // The conservation law coefficients are in columns [nr, nr+ns)
        std::vector<double> coeffs(ns);
        for (int j = 0; j < ns; ++j)
            coeffs[j] = aug[i][nr + j];

        // Choose dependent species: the one with largest absolute coefficient
        // that hasn't already been chosen (for numerical stability + uniqueness)
        int dep = -1;
        double dep_val = 0.0;
        for (int j = 0; j < ns; ++j) {
            if (species[j].fixed)
                continue;
            if (already_dep[j])
                continue; // already used by another law
            double v = std::abs(coeffs[j]);
            if (v > dep_val) {
                dep_val = v;
                dep = j;
            }
        }
        if (dep < 0)
            continue; // degenerate law

        already_dep[dep] = true;
        cl.coefficients.push_back(std::move(coeffs));
        cl.dependent.push_back(dep);
    }

    // Adjust n_laws in case some were skipped
    cl.n_laws = static_cast<int>(cl.dependent.size());
    if (cl.n_laws == 0)
        return cl;

    // Build independent species list (all non-dependent, non-fixed)
    std::vector<bool> is_dep(ns, false);
    for (int d : cl.dependent)
        is_dep[d] = true;
    for (int i = 0; i < ns; ++i) {
        if (!is_dep[i])
            cl.independent.push_back(i);
    }

    // Compute conservation constants from initial conditions
    cl.constants.resize(cl.n_laws);
    for (int k = 0; k < cl.n_laws; ++k) {
        double c = 0.0;
        for (int i = 0; i < ns; ++i) {
            c += cl.coefficients[k][i] * species[i].initial_conc;
        }
        cl.constants[k] = c;
    }

    return cl;
}

// Graph coloring (same as net_file_loader.cpp)
void compute_coloring(JacobianSparsity &sp) {
    const int n = sp.n;
    if (n == 0 || sp.nnz == 0)
        return;

    std::vector<std::vector<int>> row_to_cols(n);
    for (int j = 0; j < n; ++j)
        for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k)
            row_to_cols[static_cast<int>(sp.row_indices[k])].push_back(j);

    sp.colors.assign(n, -1);
    int max_color = -1;
    std::vector<int> forbidden(n, -1);

    for (int j = 0; j < n; ++j) {
        for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k) {
            int row = static_cast<int>(sp.row_indices[k]);
            for (int neighbor : row_to_cols[row])
                if (neighbor != j && sp.colors[neighbor] >= 0)
                    forbidden[sp.colors[neighbor]] = j;
        }
        int c = 0;
        while (c < n && forbidden[c] == j)
            ++c;
        sp.colors[j] = c;
        if (c > max_color)
            max_color = c;
    }

    sp.n_colors = max_color + 1;
    sp.color_groups.resize(sp.n_colors);
    for (int j = 0; j < n; ++j)
        sp.color_groups[sp.colors[j]].push_back(j);
}

} // anonymous namespace

// Lazily materialize the conservation laws (declared in model_impl.hpp). Kept
// here next to the detector so detect_conservation_laws() can stay in the
// anonymous namespace (internal linkage); this thin wrapper has external
// linkage so NetworkModel::conservation_laws() (model.cpp) can reach it.
const ConservationLaws &ensure_conservation_laws(const SharedModelData &sd,
                                                 const std::vector<Species> &species) {
    if (sd.conservation_laws_enabled) {
        std::call_once(sd.conservation_laws_once, [&] {
            sd.conservation_laws = detect_conservation_laws(sd.reactions, species);
        });
    }
    return sd.conservation_laws;
}

// ─── Build ───────────────────────────────────────────────────────────────────

NetworkModel ModelBuilder::build() {
    if (!bimpl_) {
        throw std::runtime_error("ModelBuilder::build() called on moved-from builder");
    }

    // ── 0. Validate model structure ──────────────────────────────────────
    // Catch malformed models early with clear error messages. Without this,
    // bad indices/names crash during simulation rather than at build time.
    {
        const auto &b = *bimpl_;
        const int ns = static_cast<int>(b.species.size());
        const int np = static_cast<int>(b.parameters.size());

        // 0a. Duplicate species names
        {
            std::unordered_map<std::string, int> seen;
            for (int i = 0; i < ns; ++i) {
                auto [it, inserted] = seen.emplace(b.species[i].name, i);
                if (!inserted) {
                    throw std::runtime_error("ModelBuilder::validate: duplicate species name '" +
                                             b.species[i].name + "' (indices " +
                                             std::to_string(it->second) + " and " +
                                             std::to_string(i) + ")");
                }
            }
        }

        // 0b. Duplicate parameter names
        {
            std::unordered_map<std::string, int> seen;
            for (int i = 0; i < np; ++i) {
                auto [it, inserted] = seen.emplace(b.parameters[i].name, i);
                if (!inserted) {
                    throw std::runtime_error("ModelBuilder::validate: duplicate parameter name '" +
                                             b.parameters[i].name + "' (indices " +
                                             std::to_string(it->second) + " and " +
                                             std::to_string(i) + ")");
                }
            }
        }

        // 0c. Reaction species indices in range + rate law resolution
        for (int ri = 0; ri < static_cast<int>(b.reactions.size()); ++ri) {
            const auto &rxn = b.reactions[ri];

            for (int si : rxn.reactant_indices) {
                if (si < 1 || si > ns) {
                    throw std::runtime_error("ModelBuilder::validate: reaction " +
                                             std::to_string(ri) + " has reactant species index " +
                                             std::to_string(si) + " out of range [1, " +
                                             std::to_string(ns) + "]");
                }
            }
            for (int si : rxn.product_indices) {
                if (si < 1 || si > ns) {
                    throw std::runtime_error("ModelBuilder::validate: reaction " +
                                             std::to_string(ri) + " has product species index " +
                                             std::to_string(si) + " out of range [1, " +
                                             std::to_string(ns) + "]");
                }
            }

            if (rxn.rate_law_type == RateLawType::Elementary) {
                // Note: .net parser may initially mark Functional reactions
                // as Elementary; build() step 5 reclassifies them later.
                // So accept if the name is in params OR functions.
                if (!rxn.function_name.empty() &&
                    b.param_name_to_idx.find(rxn.function_name) == b.param_name_to_idx.end() &&
                    b.function_name_to_idx.find(rxn.function_name) ==
                        b.function_name_to_idx.end()) {
                    throw std::runtime_error(
                        "ModelBuilder::validate: reaction " + std::to_string(ri) +
                        " (Elementary) references unknown parameter '" + rxn.function_name + "'");
                }
            } else if (rxn.rate_law_type == RateLawType::Functional) {
                if (!rxn.function_name.empty() &&
                    b.function_name_to_idx.find(rxn.function_name) ==
                        b.function_name_to_idx.end() &&
                    b.param_name_to_idx.find(rxn.function_name) == b.param_name_to_idx.end()) {
                    throw std::runtime_error(
                        "ModelBuilder::validate: reaction " + std::to_string(ri) +
                        " (Functional) references unknown function '" + rxn.function_name + "'");
                }
            } else if (rxn.rate_law_type == RateLawType::MichaelisMenten) {
                if (rxn.rate_law_param_indices.size() < 2 || rxn.rate_law_param_indices[0] < 1 ||
                    rxn.rate_law_param_indices[1] < 1) {
                    throw std::runtime_error("ModelBuilder::validate: reaction " +
                                             std::to_string(ri) +
                                             " (MichaelisMenten) has unresolved kcat/Km "
                                             "parameters (expected 'kcat_name,km_name')");
                }
            }
        }

        // 0d. Observable group species indices in range
        for (int oi = 0; oi < static_cast<int>(b.observables.size()); ++oi) {
            const auto &obs = b.observables[oi];
            for (const auto &entry : obs.entries) {
                if (entry.species_index < 1 || entry.species_index > ns) {
                    throw std::runtime_error("ModelBuilder::validate: observable '" + obs.name +
                                             "' references species index " +
                                             std::to_string(entry.species_index) +
                                             " out of range [1, " + std::to_string(ns) + "]");
                }
            }
        }
    }

    NetworkModel model;
    auto &impl = *model.impl_;

    // ── 1. Build shared immutable data ───────────────────────────────────
    auto sd = std::make_shared<SharedModelData>();

    sd->reactions = std::move(bimpl_->reactions);
    sd->param_name_to_idx = std::move(bimpl_->param_name_to_idx);
    sd->species_name_to_idx = std::move(bimpl_->species_name_to_idx);
    sd->observable_name_to_idx = std::move(bimpl_->observable_name_to_idx);
    sd->function_name_to_idx = std::move(bimpl_->function_name_to_idx);

    // Copy mutable data into model Impl
    impl.species = std::move(bimpl_->species);
    impl.parameters = std::move(bimpl_->parameters);
    impl.observables = std::move(bimpl_->observables);
    impl.functions = std::move(bimpl_->functions);

    // Assign shared data early so that register_table_function_() (called
    // during tfun processing in step 3d) can find param/observable names.
    // sd remains mutable through the local shared_ptr<SharedModelData>;
    // impl.shared is shared_ptr<const> so the const contract holds after build.
    impl.shared = sd;

    // ── 2. Set up function → parameter bindings ──────────────────────────
    impl.has_functions = !impl.functions.empty();

    const int nf = static_cast<int>(impl.functions.size());

    // Bind each function to its (existing or synthetic) parameter slot, in
    // declaration order so parameter indices stay deterministic.
    std::vector<int> func_param_idx(nf);
    for (int fi = 0; fi < nf; ++fi) {
        auto pit = sd->param_name_to_idx.find(impl.functions[fi].name);
        if (pit != sd->param_name_to_idx.end()) {
            func_param_idx[fi] = pit->second;
        } else {
            // Create synthetic parameter for this function
            Parameter synth;
            synth.index = static_cast<int>(impl.parameters.size()) + 1;
            synth.name = impl.functions[fi].name;
            synth.value = 0.0;
            synth.expression = "";
            synth.is_expression = false;
            synth.evaluator_id = -1;

            int new_idx = static_cast<int>(impl.parameters.size());
            sd->param_name_to_idx[synth.name] = new_idx;
            impl.parameters.push_back(std::move(synth));
            func_param_idx[fi] = new_idx;
        }
    }

    // Evaluate functions in DEPENDENCY (topological) order, not declaration
    // order. evaluate_functions() walks var_param_bindings in sequence and
    // writes each function's value into its bound parameter; consumers read that
    // parameter. If function A's expression references a function B declared
    // *after* A, a single declaration-order pass reads B's STALE bound-param
    // value (from the previous RHS evaluation), so compute_derivs becomes
    // path-dependent rather than a pure function of (t, y). That silently
    // corrupts the RHS for such models and makes the analytical Jacobian (which
    // assumes fully-resolved function values) diverge from the engine (GH #76).
    // Ordering every function after the functions whose bound params it
    // references makes one pass converge. SBML assignment/rate-rule dependency
    // graphs are acyclic; any residual cycle (malformed input) falls back to
    // declaration order for the nodes involved.
    std::vector<std::vector<int>> successors(nf); // fj -> functions depending on fj
    std::vector<int> in_degree(nf, 0);
    for (int fi = 0; fi < nf; ++fi) {
        const std::string &expr = impl.functions[fi].expression;
        std::unordered_set<int> deps;
        size_t i = 0, n = expr.size();
        while (i < n) {
            char c = expr[i];
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                size_t start = i;
                while (i < n &&
                       (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_'))
                    ++i;
                auto it = sd->function_name_to_idx.find(expr.substr(start, i - start));
                if (it != sd->function_name_to_idx.end() && it->second != fi && it->second >= 0 &&
                    it->second < nf)
                    deps.insert(it->second);
            } else {
                ++i;
            }
        }
        for (int fj : deps) {
            successors[fj].push_back(fi);
            ++in_degree[fi];
        }
    }

    // Kahn topological sort; seed ready nodes in ascending index so an already
    // dependency-ordered model keeps its original (byte-identical) order.
    std::vector<int> order;
    order.reserve(nf);
    std::vector<char> placed(nf, 0);
    std::vector<int> queue;
    queue.reserve(nf);
    for (int fi = 0; fi < nf; ++fi)
        if (in_degree[fi] == 0)
            queue.push_back(fi);
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        int u = queue[qi];
        order.push_back(u);
        placed[u] = 1;
        for (int v : successors[u])
            if (--in_degree[v] == 0)
                queue.push_back(v);
    }
    // Cycle fallback: append any unplaced functions in declaration order.
    for (int fi = 0; fi < nf; ++fi)
        if (!placed[fi])
            order.push_back(fi);

    sd->var_param_bindings.reserve(nf);
    for (int fi : order)
        sd->var_param_bindings.push_back({fi, func_param_idx[fi]});

    // ── 3. Set up ExprTk evaluator ───────────────────────────────────────
    auto &eval = *impl.evaluator;
    eval.set_time_ptr(&impl.current_time);

    // Register ALL parameters (including synthetics) as variables
    for (auto &p : impl.parameters)
        eval.define_variable(p.name, &p.value);

    // Register observable totals as variables
    for (auto &obs : impl.observables)
        eval.define_variable(obs.name, &obs.total);

    // Register the rateOf accessors (GH #106) BEFORE any expression is compiled
    // (parameter expressions just below, then functions and event triggers
    // later all may reference a rate_of__<species> token). impl.species is fully
    // populated at this point; current_derivs is sized once and never resized so
    // the bound &current_derivs[i] addresses stay stable for the model's life
    // (the heap Impl survives build()'s return move). Gated on the loader's
    // enable_rateof() ⇒ zero added symbols/cost for the 1593 non-rateOf models.
    if (bimpl_->enable_rateof) {
        impl.uses_rateof = true;
        impl.current_derivs.assign(impl.species.size(), 0.0);
        impl.rateof_scratch.assign(impl.species.size(), 0.0);
        register_rateof_accessors(eval, impl.species, impl.current_derivs);
    }

    // Compile expression-valued parameters
    for (auto &p : impl.parameters) {
        if (p.is_expression && !p.expression.empty()) {
            try {
                p.evaluator_id = eval.compile(p.expression);
                p.value = eval.evaluate(p.evaluator_id);
            } catch (...) {
                // May reference functions — handle below
            }
        }
    }

    // ── 3b. Resolve species parameter references ────────────────────────
    // When a .net file uses a parameter name as a species initial
    // concentration (e.g., "A0" instead of "100.0"), re-resolve the
    // species IC from the now-evaluated parameter value. Also persist the
    // (species, param) mapping on shared data so that forward sensitivity
    // setup can seed s(0) = ∂y(0)/∂p with the IC Jacobian column.
    for (const auto &ref : bimpl_->species_param_refs) {
        auto pit = sd->param_name_to_idx.find(ref.param_name);
        if (pit != sd->param_name_to_idx.end()) {
            double val = impl.parameters[pit->second].value;
            impl.species[ref.species_idx0].concentration = val;
            impl.species[ref.species_idx0].initial_conc = val;
            sd->species_ic_param_refs.emplace_back(ref.species_idx0, pit->second);
        }
    }

    // ── 3c. Set net_file_dir ────────────────────────────────────────────
    sd->net_file_dir = bimpl_->net_file_dir;

    // ── 3d. Process tfun specs ───────────────────────────────────────────
    // Load/create table functions and register them with the model BEFORE
    // function compilation, since function expressions may reference tfuns.
    // Supports inline data and step interpolation compatible with BioNetGen
    // tfun syntax.
    for (const auto &spec : bimpl_->tfun_specs) {
        try {
            if (spec.is_inline) {
                // Inline data: create TableFunction directly from arrays
                model.add_table_function(spec.func_name, spec.xs, spec.ys, spec.index_name,
                                         spec.method);
            } else {
                // File-based: resolve path relative to .net file directory
                std::string tfun_path = spec.filepath;
                if (!tfun_path.empty() && tfun_path[0] != '/' && !bimpl_->net_file_dir.empty()) {
                    tfun_path = bimpl_->net_file_dir + "/" + spec.filepath;
                }
                model.add_table_function(spec.func_name, tfun_path, spec.index_name, spec.method,
                                         spec.header_name);
            }
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to create table function '" + spec.func_name +
                                     "': " + e.what());
        }
        // Replace the function expression with the internal tfun_ name
        // so ExprTk evaluates it as a zero-arg function call.
        auto fit = sd->function_name_to_idx.find(spec.func_name);
        if (fit != sd->function_name_to_idx.end()) {
            impl.functions[fit->second].expression = "tfun_" + spec.func_name + "()";
        }
    }

    // ── 4. Resolve inter-function references ─────────────────────────────
    // Replace "funcName()" → "funcName" for known function names
    {
        std::unordered_map<std::string, bool> func_names;
        for (const auto &f : impl.functions)
            func_names[f.name] = true;

        for (auto &func : impl.functions) {
            std::string &expr = func.expression;
            std::string result;
            result.reserve(expr.size());
            size_t i = 0;
            while (i < expr.size()) {
                if (std::isalpha(expr[i]) || expr[i] == '_') {
                    size_t start = i;
                    while (i < expr.size() && (std::isalnum(expr[i]) || expr[i] == '_'))
                        i++;
                    std::string ident = expr.substr(start, i - start);
                    if (i + 1 < expr.size() && expr[i] == '(' && expr[i + 1] == ')' &&
                        func_names.count(ident)) {
                        result += ident;
                        i += 2;
                    } else {
                        result += ident;
                    }
                } else {
                    result += expr[i];
                    i++;
                }
            }
            expr = result;
        }
    }

    // Compile functions
    for (auto &func : impl.functions) {
        try {
            func.evaluator_id = eval.compile(func.expression);
        } catch (const std::exception &e) {
            throw std::runtime_error("ModelBuilder: failed to compile function '" + func.name +
                                     "': " + func.expression + " — " + e.what());
        }
    }

    // ── 5. Resolve Functional reaction param indices ─────────────────────
    for (auto &rxn : sd->reactions) {
        if (!rxn.function_name.empty()) {
            if (sd->function_name_to_idx.count(rxn.function_name)) {
                rxn.rate_law_type = RateLawType::Functional;
                if (!rxn.rate_law_param_indices.empty() && rxn.rate_law_param_indices[0] == -1) {
                    auto pit = sd->param_name_to_idx.find(rxn.function_name);
                    if (pit != sd->param_name_to_idx.end()) {
                        rxn.rate_law_param_indices[0] = impl.parameters[pit->second].index;
                    }
                }
            }
        }
    }

    // ── 6. Build stoichiometry ───────────────────────────────────────────
    sd->stoichiometry = build_stoich(sd->reactions);

    // ── 7. Jacobian sparsity + analytical Jacobian + coloring ────────────
    const int ns = static_cast<int>(impl.species.size());
    const int np = static_cast<int>(impl.parameters.size());

    sd->jac_sparsity =
        build_jac_sparsity(sd->reactions, ns, impl.observables, sd->observable_name_to_idx,
                           impl.functions, sd->function_name_to_idx, impl.species);

    sd->analytical_jac = build_anal_jac(sd->reactions, ns, np, sd->jac_sparsity, impl.species);

    if (!sd->jac_sparsity.empty() && sd->jac_sparsity.density < 0.5) {
        compute_coloring(sd->jac_sparsity);
    }

    // ── 7b. Conservation laws ───────────────────────────────────────────
    // Deferred (GH #102): the detector is dense O(ns^3) Gaussian elimination and
    // is consumed ONLY by the steady-state solver and the conservation_laws()
    // accessor — never by ODE/SSA integration. build() therefore does NOT run it;
    // ensure_conservation_laws() materializes it on first access (once, shared
    // across clones), so ODE/SSA-only runs (and the large networks that the cubic
    // detector previously walled off) never pay for it. compute_conservation_laws
    // == false keeps it permanently disabled (empty, n_species set), exactly as
    // before, for callers that need the full unreduced system.
    sd->conservation_laws.n_species = ns;
    sd->conservation_laws_enabled = bimpl_->compute_conservation_laws;

    // ── 8. Pre-compute SSA propensity data ───────────────────────────────
    for (auto &rxn : sd->reactions) {
        std::unordered_map<int, int> mult_map;
        for (int ri : rxn.reactant_indices)
            mult_map[ri]++;
        rxn.reactant_multiplicities.clear();
        rxn.reactant_multiplicities.reserve(mult_map.size());
        for (const auto &[si_1, count] : mult_map) {
            int si_0 = si_1 - 1;
            if (si_0 >= 0 && si_0 < ns)
                rxn.reactant_multiplicities.emplace_back(si_0, count);
        }

        if (!rxn.rate_law_param_indices.empty()) {
            int k_idx = rxn.rate_law_param_indices[0] - 1;
            if (k_idx >= 0 && k_idx < np)
                rxn.rate_param_idx0 = k_idx;
        }

        if (rxn.rate_law_type == RateLawType::MichaelisMenten) {
            if (rxn.rate_law_param_indices.size() >= 2) {
                int kcat = rxn.rate_law_param_indices[0] - 1;
                int km = rxn.rate_law_param_indices[1] - 1;
                if (kcat >= 0 && kcat < np)
                    rxn.mm_kcat_idx0 = kcat;
                if (km >= 0 && km < np)
                    rxn.mm_km_idx0 = km;
            }
            if (rxn.reactant_indices.size() >= 2) {
                int e_si = rxn.reactant_indices[0] - 1;
                int s_si = rxn.reactant_indices[1] - 1;
                if (e_si >= 0 && e_si < ns)
                    rxn.mm_enzyme_idx0 = e_si;
                if (s_si >= 0 && s_si < ns)
                    rxn.mm_substrate_idx0 = s_si;
            }
        }
    }

    // ── 8b. Compile events ───────────────────────────────────────────────
    // Event trigger and assignment expressions are compiled AFTER all
    // parameters, observables, and functions are registered with the
    // evaluator (steps 3-4), so they can reference any model symbol.
    // Also registers species concentrations as ExprTk variables so
    // trigger expressions like "S >= 10" can read species values.
    {
        // Register species concentrations as ExprTk variables so trigger
        // expressions can reference them (e.g., "S >= threshold").
        for (auto &sp : impl.species) {
            // Only register if not already registered (species names may
            // collide with observable names in some SBML models).
            // Use try-catch since define_variable throws on duplicate.
            try {
                eval.define_variable(sp.name, &sp.concentration);
            } catch (...) {
                // Already registered (e.g., as observable) — skip
            }
        }

        for (const auto &espec : bimpl_->event_specs) {
            Event ev;
            ev.id = espec.id;
            ev.delay = espec.delay;
            ev.priority = espec.priority;
            ev.persistent = espec.persistent;
            ev.initial_value = espec.initial_value;
            ev.use_values_from_trigger_time = espec.use_values_from_trigger_time;

            // Compile trigger expression
            try {
                ev.trigger_expr_idx = eval.compile(espec.trigger_expr);
            } catch (const std::exception &e) {
                throw std::runtime_error("ModelBuilder: failed to compile event trigger '" +
                                         espec.id + "': " + espec.trigger_expr + " — " + e.what());
            }

            // Compile optional delay expression
            if (!espec.delay_expr.empty()) {
                try {
                    ev.delay_expr_idx = eval.compile(espec.delay_expr);
                } catch (const std::exception &e) {
                    throw std::runtime_error("ModelBuilder: failed to compile event delay '" +
                                             espec.id + "': " + espec.delay_expr + " — " +
                                             e.what());
                }
            }

            // Compile optional priority expression
            if (!espec.priority_expr.empty()) {
                try {
                    ev.priority_expr_idx = eval.compile(espec.priority_expr);
                } catch (const std::exception &e) {
                    throw std::runtime_error("ModelBuilder: failed to compile event priority '" +
                                             espec.id + "': " + espec.priority_expr + " — " +
                                             e.what());
                }
            }

            // Compile assignment value expressions
            for (std::size_t ai = 0; ai < espec.assignments.size(); ++ai) {
                const auto &[sp_idx0, val_expr] = espec.assignments[ai];
                if (sp_idx0 < 0 || sp_idx0 >= ns) {
                    throw std::runtime_error("ModelBuilder: event '" + espec.id +
                                             "' assigns to species index " +
                                             std::to_string(sp_idx0) + " out of range [0, " +
                                             std::to_string(ns - 1) + "]");
                }
                int val_id = -1;
                try {
                    val_id = eval.compile(val_expr);
                } catch (const std::exception &e) {
                    throw std::runtime_error("ModelBuilder: failed to compile event assignment '" +
                                             espec.id + "': " + val_expr + " — " + e.what());
                }
                ev.assignments.emplace_back(sp_idx0, val_id);
                // GH #81: carry the parallel ode_only flag (default false when
                // the spec omitted it, so every existing event is unchanged).
                ev.assignment_ode_only.push_back(ai < espec.assignment_ode_only.size() &&
                                                 espec.assignment_ode_only[ai]);
            }

            impl.events.push_back(std::move(ev));
        }

        // Compile discontinuity-trigger conditions (GH #72) into the same
        // evaluator. Each becomes a CVODE root in the ODE simulator.
        for (const auto &cond : bimpl_->discontinuity_trigger_specs) {
            try {
                impl.discontinuity_trigger_expr_idx.push_back(eval.compile(cond));
            } catch (const std::exception &e) {
                throw std::runtime_error("ModelBuilder: failed to compile discontinuity trigger '" +
                                         cond + "' — " + e.what());
            }
        }
    }

    // ── 9. Freeze shared data and assign to model ────────────────────────
    impl.shared = std::move(sd);

    // Invalidate builder
    bimpl_.reset();

    return model;
}

} // namespace bngsim
