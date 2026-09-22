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

// Call `fn(token)` once per identifier in `expr`, in source order, under
// ExprTk's symbol grammar ([A-Za-z_][A-Za-z0-9_]*). Repeats are not collapsed.
//
// Scanned against the *source* text, so the names handed to `fn` are the
// model's own — the evaluator mangles a name before registering it (`_X` →
// `u_X`, reserved words → `r_<name>`), and matching post-mangling would miss
// them.
//
// A numeric literal is stepped over as a unit, exponent included, so `1e-5`
// yields nothing and `2.0e3` does not yield `e3`. That distinction was
// invisible while the only consumer was `references_model_symbol`, where a
// phantom token merely kept a parameter flagged derived — but issue #617 gave
// the same scanner two consumers that REFUSE a model, and `e` and `E` are
// ordinary parameter names (`e` appears in four `.net` files in this tree, `E`
// is the enzyme in every Michaelis-Menten model). `E = 1E-5*V` was read as
// `E` defining itself and the build was aborted. An exponent that is not
// well-formed (`1e`, `1eX`) is left for the identifier branch below, which is
// what ExprTk itself does with it.
template <typename F> static void for_each_identifier(const std::string &expr, F &&fn) {
    const auto digit = [&](size_t k) {
        return k < expr.size() && std::isdigit(static_cast<unsigned char>(expr[k])) != 0;
    };
    size_t i = 0, n = expr.size();
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(expr[i]);
        if (std::isdigit(c) || (expr[i] == '.' && digit(i + 1))) {
            while (i < n && (digit(i) || expr[i] == '.'))
                ++i;
            if (i < n && (expr[i] == 'e' || expr[i] == 'E')) {
                size_t j = i + 1;
                if (j < n && (expr[j] == '+' || expr[j] == '-'))
                    ++j;
                if (digit(j)) {
                    i = j;
                    while (digit(i))
                        ++i;
                }
            }
            continue;
        }
        if (std::isalpha(c) || expr[i] == '_') {
            size_t start = i;
            while (i < n && (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_'))
                ++i;
            fn(expr.substr(start, i - start));
        } else {
            ++i;
        }
    }
}

// Kahn topological sort of `n` nodes given each node's successors and in-degree:
// every node comes out after the nodes it depends on.
//
// Two dependency graphs in build() need exactly this and need it to behave the
// same way — function evaluation order (GH #76) and derived-parameter
// evaluation order (issue #568). Ready nodes are seeded in ascending index,
// which keeps the output deterministic and, among nodes that become ready
// together, in declaration order. It does NOT preserve declaration order
// wholesale: this is FIFO Kahn, so every root is emitted ahead of every
// non-root — `a`, `b = f(a)`, `c` comes out `a, c, b` even though the input was
// already a dependency order. Only a min-heap variant has that property. The
// values are unaffected, because nodes that swap are by construction
// independent of each other; what moves is report and emit order.
//
// A cycle is a graph no order satisfies. Its nodes are still appended, in
// declaration order, so the sort itself never drops a node — what to DO about a
// cycle is the caller's decision, and the two callers decide differently (see
// each).
//
// `has_cycle` is the caller's gate, NOT `cycle.empty()`: the two are the same
// today, but a caller that refused on the reconstruction being non-empty would
// silently BUILD a cyclic model on any future day the walk failed to close —
// the one failure this reporting exists to prevent, arriving with no
// diagnostic. `cycle` is a presentation detail on top; it holds one cycle as
// `v0, v1, …, v0`, the first node repeated at the end.
struct DependencyOrder {
    std::vector<int> order;
    std::vector<int> cycle; // one cycle, v0 … v0; empty if none was reconstructed
    bool has_cycle = false; // a node was left unplaced — authoritative
};

static DependencyOrder dependency_order(int n, const std::vector<std::vector<int>> &successors,
                                        std::vector<int> in_degree, bool want_cycle = false) {
    std::vector<int> order;
    order.reserve(static_cast<size_t>(n));
    std::vector<char> placed(static_cast<size_t>(n), 0);
    std::vector<int> queue;
    queue.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        if (in_degree[i] == 0)
            queue.push_back(i);
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int u = queue[qi];
        order.push_back(u);
        placed[u] = 1;
        for (int v : successors[u])
            if (--in_degree[v] == 0)
                queue.push_back(v);
    }
    DependencyOrder out;
    for (int i = 0; i < n; ++i)
        if (!placed[i]) {
            order.push_back(i);
            out.has_cycle = true;
        }
    out.order = std::move(order);

    if (!want_cycle || !out.has_cycle)
        return out;
    // Kahn decrements a node's in-degree once per PLACED predecessor, so a node
    // left unplaced kept at least one predecessor that was itself never placed.
    // Walking predecessors from such a node therefore never leaves the unplaced
    // set, and in a finite graph must revisit one — the segment between the two
    // visits is a cycle. (The unplaced set is wider than the cycle: it also
    // holds everything downstream of one. This reports the cycle itself.)
    //
    // Every unplaced start is tried rather than only the first. Under the
    // in-degree argument above the first always closes, but that argument is
    // the caller's to keep (it needs `in_degree` to count the edges in
    // `successors`), and a start that dead-ends is not a reason to report
    // nothing — at least one unplaced node lies ON a cycle, and from there the
    // walk always closes.
    std::vector<int> pred(static_cast<size_t>(n), -1);
    for (int u = 0; u < n; ++u)
        if (!placed[u])
            for (int v : successors[u])
                if (!placed[v] && pred[v] < 0)
                    pred[v] = u;
    for (int start = 0; start < n && out.cycle.empty(); ++start) {
        if (placed[start])
            continue;
        std::vector<int> seen_at(static_cast<size_t>(n), -1);
        std::vector<int> walk;
        for (int v = start; v >= 0; v = pred[v]) {
            if (seen_at[v] >= 0) {
                // pred[x] is something x READS, so consecutive entries of
                // `walk` already run in "reads" direction. The cycle is the
                // segment from this node's first visit to here, closed back
                // onto itself.
                out.cycle.assign(walk.begin() + seen_at[v], walk.end());
                out.cycle.push_back(v);
                break;
            }
            seen_at[v] = static_cast<int>(walk.size());
            walk.push_back(v);
        }
    }
    return out;
}

// Tarjan strongly-connected components, returned in TOPOLOGICAL order (every
// group after the groups it reads). A singleton group is an ordinary function;
// a group of two or more is a set of functions that read each other, which no
// evaluation order can resolve one at a time.
//
// The derived-PARAMETER sort above answers a cycle by refusing (issue #617):
// a parameter denotes one number and a cyclic definition denotes none. A
// FUNCTION cycle is a different object — a simultaneous system over values the
// engine recomputes every step, which can have a perfectly good solution — so
// it is grouped here and solved at evaluation time (issue #621) rather than
// refused. That is why this exists beside `dependency_order` instead of
// replacing it: the two callers need different answers to the same question.
//
// Iterative rather than recursive: a genome-scale model's assignment-rule graph
// can be tens of thousands of nodes deep and this runs at load on every model.
static std::vector<std::vector<int>>
strongly_connected_components(int n, const std::vector<std::vector<int>> &reads) {
    std::vector<int> index(static_cast<size_t>(n), -1), low(static_cast<size_t>(n), 0);
    std::vector<char> on_stack(static_cast<size_t>(n), 0);
    std::vector<int> stack;
    std::vector<std::vector<int>> out;
    int next_index = 0;

    struct Frame {
        int v;
        size_t edge;
    };
    std::vector<Frame> call;

    for (int root = 0; root < n; ++root) {
        if (index[root] >= 0)
            continue;
        call.push_back({root, 0});
        index[root] = low[root] = next_index++;
        stack.push_back(root);
        on_stack[root] = 1;
        while (!call.empty()) {
            Frame &f = call.back();
            if (f.edge < reads[f.v].size()) {
                const int w = reads[f.v][f.edge++];
                if (index[w] < 0) {
                    index[w] = low[w] = next_index++;
                    stack.push_back(w);
                    on_stack[w] = 1;
                    call.push_back({w, 0});
                } else if (on_stack[w]) {
                    low[f.v] = std::min(low[f.v], index[w]);
                }
                continue;
            }
            const int v = f.v;
            call.pop_back();
            if (!call.empty())
                low[call.back().v] = std::min(low[call.back().v], low[v]);
            if (low[v] == index[v]) {
                std::vector<int> group;
                for (;;) {
                    const int w = stack.back();
                    stack.pop_back();
                    on_stack[w] = 0;
                    group.push_back(w);
                    if (w == v)
                        break;
                }
                std::sort(group.begin(), group.end()); // declaration order within a group
                out.push_back(std::move(group));
            }
        }
    }
    // Tarjan emits a group only after everything it reads, i.e. already in
    // topological order for "reads" edges. No reversal.
    return out;
}

// Issue #227 — does `expr` name anything in this model whose value can move
// after load? That, and only that, is what makes a parameter *derived*: an
// expression that names another parameter carries `∂p_d/∂θ` into every rate law
// that reads it, while `gamma = 1/7` or `pi = 2*asin(1)` names nothing and is a
// constant written as arithmetic — which is exactly the line BNG2.pl draws when
// it annotates the first `# ConstantExpression` and the second `# Constant`, and
// the rule GH #181 gave the codegen `.net` parser.
//
// Species are checked even though a parameter expression that reads one cannot
// compile at the point this is called (species become evaluator variables later
// in build()) — a caller that reorders those steps should not silently start
// folding a species reference into a constant.
static bool references_model_symbol(const std::string &expr, const std::string &self_name,
                                    const SharedModelData &sd) {
    bool found = false;
    for_each_identifier(expr, [&](const std::string &token) {
        // `time()` is a registered function and `rate_of__<species>` a
        // registered variable; both read the running solve, so an expression
        // naming either is not a load-time constant however it is annotated.
        if (token == "time" || token.rfind("rate_of__", 0) == 0)
            found = true;
        else if (token != self_name && sd.param_name_to_idx.count(token))
            found = true;
        else if (sd.observable_name_to_idx.count(token) || sd.species_name_to_idx.count(token))
            found = true;
    });
    return found;
}

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
                                const std::string &expression, bool is_expression,
                                bool is_compartment_size, bool is_internal) {
    int idx = static_cast<int>(bimpl_->parameters.size());
    Parameter p;
    p.index = idx + 1; // 1-based for .net compatibility
    p.name = name;
    p.value = value;
    p.expression = expression;
    p.is_expression = is_expression;
    p.evaluator_id = -1;
    p.is_compartment_size = is_compartment_size;
    p.is_internal = is_internal;

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

void ModelBuilder::set_species_volume_param(int species_idx0, int param_idx0,
                                            double initial_amount) {
    if (species_idx0 < 0 || species_idx0 >= static_cast<int>(bimpl_->species.size()))
        return;
    bimpl_->species[species_idx0].volume_param_idx0 = param_idx0;
    bimpl_->species[species_idx0].initial_amount = initial_amount;
}

void ModelBuilder::set_reaction_ssa_volume_param(int rxn_idx0, int param_idx0) {
    if (rxn_idx0 < 0 || rxn_idx0 >= static_cast<int>(bimpl_->reactions.size()))
        return;
    bimpl_->reactions[rxn_idx0].ssa_volume_param_idx0 = param_idx0;
}

void ModelBuilder::set_param_volume_write_refused(const std::string &name) {
    for (auto &p : bimpl_->parameters) {
        if (p.name == name) {
            p.volume_write_refused = true;
            return;
        }
    }
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
        bool any_live_volume = false;
        for (int si : rxn.reactant_indices) {
            int si_0 = si - 1;
            if (si_0 >= 0 && si_0 < n_species && species[si_0].amount_valued) {
                terms.amount_factor *= species[si_0].volume_factor;
                // (#170) Same factors, same order, but tagged with the
                // compartment-size parameter each came from so the scatter can
                // re-read a written volume. Kept only if at least one is live.
                terms.amount_volume_terms.emplace_back(species[si_0].volume_param_idx0,
                                                       species[si_0].volume_factor);
                any_live_volume |= species[si_0].volume_param_idx0 >= 0;
            }
        }
        if (!any_live_volume)
            terms.amount_volume_terms.clear();
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
//
// The rows alone are not the whole answer: the consumers eliminate one species
// per law, so which species each law is solved FOR has to make that elimination
// well posed. L is therefore also row-reduced over the species columns, and the
// dependents are its pivots — see the long note at that step below.
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
    //
    // Rows [0..rank-1] are pivot rows → independent species
    // Rows [rank..ns-1] are null space → conservation laws
    std::vector<std::vector<double>> laws;
    laws.reserve(n_laws);
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
        laws.push_back(std::move(coeffs));
    }

    // ── Choose the dependent species by ROW-REDUCING the law matrix ───────────
    //
    // Every consumer of these laws eliminates dependent species one law at a
    // time: reconstruct_full() solves law k for y[dep_k] from the *current* value
    // of every other species, and compute_ss_sensitivity()'s D matrix forward-
    // substitutes the same walk to get ∂y_dep/∂y_ind. Both are exact only if
    // L[:, dependent] is the identity — one pass cannot satisfy law k once a
    // LATER law overwrites a dependent that law k also constrains.
    //
    // The old rule picked each law's dependent greedily (largest |coefficient|
    // not already claimed), which guarantees neither. On real models it does not
    // merely lose accuracy, it picks a set for which L[:, dependent] is SINGULAR
    // — the elimination has no solution at all. Measured on the ode_fullnet
    // corpus, that silently violated the constraints it was enforcing
    // (|L_ind + L_dep·D| = 1.0 instead of 0) and handed the reduced Jacobian a
    // null space of the reduction's own making: IGF1R_model_v1 came back rank
    // 578/579, Reduced_IGF1R_hela 546/549, fceri_fyn 1274/1276, so
    // steady_state(sensitivity_params=...) reported dY_ss/dp from a system it
    // could not invert (issue #63). With a valid dependent set the same three are
    // full rank and well conditioned (min|U|/max|U| of 1.5e-3, 4.7e-3, 7.1e-5).
    //
    // Row-reducing L over the species columns and taking the PIVOT columns as the
    // dependents makes L[:, dependent] = I by construction, so the invariant both
    // consumers assume is now a property of the data instead of a hope. Pivots are
    // selected by largest magnitude over all remaining rows and admissible columns
    // (full pivoting), which also keeps the elimination away from tiny divisors.
    // n_laws is at most a few dozen even on the largest corpus networks, so this
    // costs nothing next to the O(ns·(nr+ns)) elimination above that produced the
    // rows in the first place.
    const int nl = static_cast<int>(laws.size());
    cl.coefficients.reserve(nl);
    cl.dependent.reserve(nl);
    std::vector<bool> is_pivot_col(ns, false);
    for (int r = 0; r < nl; ++r) {
        int best_row = -1, best_col = -1;
        double best_val = 0.0;
        for (int i = r; i < nl; ++i) {
            for (int j = 0; j < ns; ++j) {
                // A fixed species is held constant, so it is never eliminated.
                if (species[j].fixed || is_pivot_col[j])
                    continue;
                const double v = std::abs(laws[i][j]);
                if (v > best_val) {
                    best_val = v;
                    best_row = i;
                    best_col = j;
                }
            }
        }
        // No admissible pivot left: the remaining rows are supported entirely on
        // fixed species, are linearly dependent on the rows already taken (row
        // reduction has zeroed them), or are numerical noise — none of which
        // constrains anything solvable. The old code dropped these one at a time
        // via `dep < 0`; dropping the whole tail here is the same set, since row
        // reduction has already eliminated every pivot column from them. The tol
        // floor matters because dividing a noise-level row by its own noise-level
        // pivot would promote round-off into an O(1) "law".
        if (best_row < 0 || best_val < tol)
            break;

        if (best_row != r)
            std::swap(laws[best_row], laws[r]);

        // Normalize the pivot to 1, then clear the pivot column from every other
        // row, so the finished matrix has L[k][dependent[k']] = δ_kk'.
        const double pivot = laws[r][best_col];
        for (int j = 0; j < ns; ++j)
            laws[r][j] /= pivot;
        laws[r][best_col] = 1.0; // exact, against round-off in the divide
        for (int i = 0; i < nl; ++i) {
            if (i == r)
                continue;
            const double factor = laws[i][best_col];
            if (factor == 0.0)
                continue;
            for (int j = 0; j < ns; ++j)
                laws[i][j] -= factor * laws[r][j];
            laws[i][best_col] = 0.0;
        }

        is_pivot_col[best_col] = true;
        cl.dependent.push_back(best_col);
    }

    for (size_t k = 0; k < cl.dependent.size(); ++k)
        cl.coefficients.push_back(std::move(laws[k]));

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

// Curtis-Powell-Reid graph coloring of the Jacobian sparsity pattern: two
// columns may share a color iff their patterns share no row, so all columns of
// a color can be perturbed in one RHS eval. Greedy first-fit, so n_colors is an
// upper bound on the chromatic number — for a fully dense pattern every column
// gets its own color and colored FD degenerates to plain column-by-column FD,
// which is correct, just not a speedup. Called only via
// ensure_jacobian_coloring() below.
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

// Lazily materialize the Jacobian coloring (declared in model_impl.hpp). Same
// shape as ensure_conservation_laws above, and here for the same reason:
// compute_coloring() has internal linkage, this wrapper does not, so
// NetworkModel::ensure_jacobian_coloring() (model.cpp) can reach it.
const JacobianSparsity &ensure_jacobian_coloring(const SharedModelData &sd) {
    std::call_once(sd.jac_coloring_once, [&] { compute_coloring(sd.jac_sparsity); });
    return sd.jac_sparsity;
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
            // Issue #266 — the slot the function binds to is the function's
            // storage whether bngsim synthesized it (below) or the model
            // declared it, because `evaluate_functions()` walks
            // `var_param_bindings` and overwrites the bound slot either way. So
            // a declared row that a same-named function shadows is a discarded
            // seed, not a knob, and carries the same flag as the synthesized
            // one. #227 dropped the synthesized slots and left these, which is
            // the residue: an SBML `<assignmentRule>` converted to `.net`
            // emits both rows — the parameter keeps the species' *initial*
            // value and the rule becomes the function — so `set_param` on it
            // was accepted and discarded one step later, and an optimizer
            // handed the name through `primary_param_names` fitted nothing.
            impl.parameters[pit->second].is_internal = true;
        } else {
            // Create synthetic parameter for this function.
            //
            // Issue #227 — this slot is storage for the function's evaluated
            // value, not a knob of the model: `evaluate_functions()` overwrites
            // it before every RHS evaluation, so a write to it survives exactly
            // until the next one and its sensitivity column is identically zero
            // forever. `is_internal` is what says so — the same flag #170 gave
            // `_V0_<comp>`, and for the same reason: a synthesized slot an
            // optimizer handed would fit a quantity with no meaning. It keeps
            // the name out of `primary_param_names` and makes `set_param`
            // refuse the write rather than accept one that does nothing.
            Parameter synth;
            synth.index = static_cast<int>(impl.parameters.size()) + 1;
            synth.name = impl.functions[fi].name;
            synth.value = 0.0;
            synth.expression = "";
            synth.is_expression = false;
            synth.is_internal = true;
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
    // references makes one pass converge.
    //
    // A cyclic function graph keeps the fallback order and still builds — the
    // opposite of what the derived-PARAMETER sort below does with a cycle, and
    // deliberately so (issue #617). SBML forbids a circular assignment-rule
    // graph, but MODEL1006230117 in the vendored corpus has one, and it is not
    // the nonsense the parameter case is: its two rules are
    //
    //     R   = R_Total - LR - LRG - RG
    //     LRG = (L_iso * R * Gs) / (K_H * K_C)
    //
    // — mutually recursive, but LINEAR in R and LRG and so a simultaneous system
    // with one solution. The engine does not find it. `evaluate_functions()`
    // runs one Gauss-Seidel sweep per RHS evaluation with no convergence check,
    // so the RHS is a function of how many times it has been called; on this
    // very model the loop gain is ~36 and it diverges, reaching `inf` before
    // t = 1e-13 (issue #621). So refusing a function cycle would cost this model
    // its LOAD, not its results — it has none today. It is left building because
    // the fix is to solve the system or to iterate it to a checked fixed point,
    // which is a different change from this sort, and not because the relaxation
    // works. A parameter cycle has no such open question: it is refused.
    std::vector<std::vector<int>> successors(nf); // fj -> functions depending on fj
    std::vector<int> in_degree(nf, 0);
    for (int fi = 0; fi < nf; ++fi) {
        std::unordered_set<int> deps;
        for_each_identifier(impl.functions[fi].expression, [&](const std::string &token) {
            auto it = sd->function_name_to_idx.find(token);
            if (it != sd->function_name_to_idx.end() && it->second != fi && it->second >= 0 &&
                it->second < nf)
                deps.insert(it->second);
        });
        for (int fj : deps) {
            successors[fj].push_back(fi);
            ++in_degree[fi];
        }
    }

    // `successors[u]` holds the functions that read u; Tarjan wants the edges
    // the other way round (what each function reads), so invert once.
    std::vector<std::vector<int>> reads(static_cast<size_t>(nf));
    for (int u = 0; u < nf; ++u)
        for (int v : successors[u])
            reads[static_cast<size_t>(v)].push_back(u);

    const std::vector<std::vector<int>> sccs = strongly_connected_components(nf, reads);

    sd->var_param_bindings.reserve(nf);
    sd->function_scc_starts.reserve(sccs.size() + 1);
    for (const auto &group : sccs) {
        sd->function_scc_starts.push_back(static_cast<int>(sd->var_param_bindings.size()));
        for (int fi : group)
            sd->var_param_bindings.push_back({fi, func_param_idx[fi]});
    }
    sd->function_scc_starts.push_back(static_cast<int>(sd->var_param_bindings.size()));

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
    //
    // Issue #227 — and then demote the ones that turn out to be constants. A
    // front end has to guess `is_expression` from the value text alone (the
    // `.net` reader asks whether `std::stod` consumes the whole token), so
    // `gamma = 1/7` arrives here flagged derived — and it is not: it references
    // nothing, so there is no chain rule to carry and no primary underneath it
    // to be fitted instead. Left flagged, `primary_param_names` drops it, and
    // the recovery rate of `SIR.net` is silently absent from the vector the
    // accessor's own docstring says to hand an optimizer.
    //
    // The demotion has to happen *here*, after the evaluation: the guess is what
    // gets the expression compiled at all, and without that `gamma` would keep
    // the 1.0 that a partial `stod("1/7")` left behind. Dropping `evaluator_id`
    // with it is what makes the parameter an ordinary constant rather than a
    // derived one that happens to have no primaries — otherwise writing it its
    // own nominal value would re-attach it (issue #188's value-keyed override)
    // and take it back out of `primary_param_names`.
    //
    // A function-bound parameter is not a knob whatever its expression says: an
    // SBML `<assignmentRule>` arrives as a function that overwrites this slot
    // every step, so the slot is not a knob even when its `<initialAssignment>`
    // is referenceless arithmetic. It is not *derived* either, though — nothing
    // recomputes it from primaries, the function simply overwrites it — so the
    // fact is carried by `is_internal` (set at bind time above) and the derived
    // flag comes off. Issue #266: the two flags have to stay disjoint, since
    // `primary_param_names` is the residue of subtracting both and a row
    // carrying each would be reported under neither reason. The expression is
    // still compiled and evaluated first, because that is what seeds the value
    // this slot holds until the function's first evaluation.
    //
    // ...and evaluate them in DEPENDENCY order, not declaration order (issue
    // #568). This is GH #76's defect one field over: a single declaration-order
    // pass over a chain (`bb = a*3` declared before `a = base*2`) reads the
    // link below it at whatever value it happens to hold, so `bb` takes the
    // front end's seed here rather than `a*3` — and every later pass moves the
    // chain exactly one link, which is why a second, value-identical
    // `set_param("base", …)` used to change the answer. Because a derived
    // parameter is routinely a rate constant, the failure mode is a plausible
    // wrong number: the model integrates at the pre-write rate and reports
    // success.
    //
    // Nothing enforced the order it assumed. add_parameter() appends in call
    // order, every parameter is an ExprTk variable before any expression is
    // compiled, and the `.net`/SBML/`ModelBuilder` front ends all declare in
    // their own source's order. The one-pass assumption was a precondition on a
    // public API that the API never stated and no caller could see — the SBML
    // loader is the only front end that ever knew to hand-sort for it.
    //
    // The order is computed once here and stored on the shared data, because
    // every pass that re-derives these parameters (`set_param`, the CVODES
    // sensitivity RHS syncs, the steady-state FD probe) must walk the same one:
    // a chain is only ever as converged as the least-ordered pass over it.
    // Edges run between derived parameters only. A constant is already at its
    // final value, and a function-bound slot is not derived from anything —
    // `evaluate_functions()` overwrites it every step — so neither can be
    // stale.
    std::unordered_set<int> function_bound(func_param_idx.begin(), func_param_idx.end());
    std::vector<int> derived_param_idx;              // node -> parameter index
    std::unordered_map<int, int> derived_param_node; // parameter index -> node
    std::vector<int> function_bound_expr;            // compiled, but not graph nodes
    for (int pi = 0; pi < static_cast<int>(impl.parameters.size()); ++pi) {
        const auto &p = impl.parameters[pi];
        if (!p.is_expression || p.expression.empty())
            continue;
        // A function-bound slot is storage, not a definition: `evaluate_functions()`
        // overwrites it before every derivative evaluation, so its expression is
        // only the seed it holds until the function's first pass and it cannot go
        // stale, cannot drift, and cannot be unsatisfiable. It is therefore NOT a
        // node — it is exempt from the ordering AND from the cycle and
        // self-reference refusals below, which would otherwise refuse the very
        // shape the function sort above deliberately admits: the `.net` spelling
        // of an SBML `<assignmentRule>` emits both a `# ConstantExpression`
        // parameter row and a same-named function, so `R = R_Total-LRG` with
        // `LRG = 0.5*R` was refused as a parameter cycle even though both slots
        // belong to functions (issue #266's shape; MODEL1006230117's). It still
        // gets compiled and evaluated, after the sorted set, so the seed reads
        // operands that already hold their own values.
        if (function_bound.count(pi)) {
            function_bound_expr.push_back(pi);
            continue;
        }
        derived_param_node.emplace(pi, static_cast<int>(derived_param_idx.size()));
        derived_param_idx.push_back(pi);
    }
    const int nd = static_cast<int>(derived_param_idx.size());
    std::vector<std::vector<int>> p_successors(nd); // node -> derived params reading it
    std::vector<int> p_in_degree(nd, 0);
    for (int k = 0; k < nd; ++k) {
        const Parameter &pk = impl.parameters[derived_param_idx[k]];
        std::unordered_set<int> deps;
        bool self_reference = false;
        for_each_identifier(pk.expression, [&](const std::string &token) {
            if (token == pk.name) {
                self_reference = true;
                return;
            }
            auto it = sd->param_name_to_idx.find(token);
            if (it == sd->param_name_to_idx.end())
                return;
            auto dit = derived_param_node.find(it->second);
            if (dit != derived_param_node.end() && dit->second != k)
                deps.insert(dit->second);
        });
        // A parameter defined in terms of itself (issue #617). There is no value
        // it denotes, so there is nothing to build: `s = s*2` has no solution
        // unless s is 0, and what bngsim used to do with it was quieter than a
        // wrong answer deserves — `references_model_symbol` skips the
        // parameter's own name, so the row was demoted to an ordinary
        // `Constant` holding `seed*2`, with the expression dropped and nothing
        // to show that the number came from a definition that does not define
        // anything. BNG2.pl refuses the same model outright: "ABORT: Parameter s
        // is defined recursively."
        if (self_reference)
            throw std::runtime_error("ModelBuilder: parameter '" + pk.name +
                                     "' is defined in terms of itself: " + pk.name + " = " +
                                     pk.expression +
                                     ". A parameter's expression cannot read the parameter it "
                                     "defines — there is no value that satisfies it.");
        for (int d : deps) {
            p_successors[d].push_back(k);
            ++p_in_degree[k];
        }
    }

    // A reference CYCLE among derived parameters is the same defect one step
    // wider, and is refused for the same reason (issue #617). `x = y+1` with
    // `y = x+1` has no solution at all; `a = 10-b` with `b = a` has one but is
    // not something a single evaluation pass can find. Either way the sort has
    // no order to produce, and what the fallback order then produced was a
    // number manufactured from the front end's *seeds* — 2.0, 1.0 or 101.0 for
    // the same model on three different seed pairs — which then moved by a
    // fixed step on every later `set_param` of any parameter at all, because
    // each pass relaxes the cycle one more step. A rate constant that drifts
    // with the number of writes is not a model. This is the residue #568 left:
    // that fix made an acyclic chain converge in one pass, and a cycle was the
    // one place the drift survived.
    //
    // BNG2.pl refuses every parameter dependency cycle, solvable or not
    // ("ABORT: Parameter y has a dependency cycle y->x->y"), and matching it is
    // what keeps a `.bngl` that BNG2.pl rejects from loading here. Nothing in
    // the corpus is affected: of the 2,774 `.net` files in the tree and the
    // 1,291 vendored BioModels SBML documents that load, zero have a parameter
    // cycle or a self-reference.
    const DependencyOrder p_sorted =
        dependency_order(nd, p_successors, p_in_degree, /*want_cycle=*/true);
    if (p_sorted.has_cycle) {
        // Gated on has_cycle, not on the reconstruction: a cycle the walk could
        // not name is still a cycle, and building it is the outcome this refuses.
        std::string where;
        if (p_sorted.cycle.empty()) {
            where = "among its derived parameters";
        } else {
            std::string path;
            for (size_t ci = 0; ci < p_sorted.cycle.size(); ++ci) {
                if (ci)
                    path += " -> ";
                path += impl.parameters[derived_param_idx[p_sorted.cycle[ci]]].name;
            }
            where = "(" + path + ", each reading the next)";
        }
        throw std::runtime_error(
            "ModelBuilder: the model's derived parameters contain a reference cycle " + where +
            ". No single evaluation of the parameters satisfies it, so the values would depend "
            "on the order they were declared in and on how many times they had been "
            "re-derived.");
    }

    std::vector<int> compile_order;
    compile_order.reserve(p_sorted.order.size() + function_bound_expr.size());
    for (int k : p_sorted.order)
        compile_order.push_back(derived_param_idx[k]);
    compile_order.insert(compile_order.end(), function_bound_expr.begin(),
                         function_bound_expr.end());

    for (int pi : compile_order) {
        auto &p = impl.parameters[pi];
        try {
            p.evaluator_id = eval.compile(p.expression);
            p.value = eval.evaluate(p.evaluator_id);
        } catch (const std::exception &e) {
            // Previously swallowed with `continue` and the note "May reference
            // functions — handle below". There is no below: nothing re-compiles
            // a parameter skipped here, and there is nothing to re-compile for,
            // since every function already owns a parameter slot (func_param_idx
            // above creates one when the names do not already meet) and all
            // parameters are registered as variables before this loop — so a
            // function name is a symbol the compile can already see. What the
            // swallow actually caught was an expression naming something that
            // exists nowhere, and it left the parameter holding whatever the
            // front end's partial `std::stod` had scraped off the front: `k1 =
            // 2*kbse` (a typo for `kbase`) stayed 2.0 and the model integrated
            // at twice the intended rate, reporting success. run_network refuses
            // the same file ("Could not find parameter kbse. Exiting."), and a
            // function body that will not compile already throws a few lines
            // below; a parameter now says so too (issue #602).
            throw std::runtime_error("ModelBuilder: failed to compile parameter '" + p.name +
                                     "': " + p.expression + " — " + e.what());
        }
        if (function_bound.count(pi) || !references_model_symbol(p.expression, p.name, *sd)) {
            p.is_expression = false;
            p.evaluator_id = -1;
        }
    }

    // Publish the order every later re-evaluation walks — the survivors of the
    // demotion above, which are exactly the parameters that still hold
    // `evaluate(expression)` and can therefore go stale.
    sd->derived_param_order.reserve(static_cast<size_t>(nd));
    for (int pi : compile_order)
        if (impl.parameters[pi].evaluator_id >= 0)
            sd->derived_param_order.push_back(pi);

    // ── 3b. Resolve species parameter references ────────────────────────
    // When a .net file uses a parameter name as a species initial
    // concentration (e.g., "A0" instead of "100.0"), re-resolve the
    // species IC from the now-evaluated parameter value. Also persist the
    // (species, param) mapping on shared data so that forward sensitivity
    // setup can seed s(0) = ∂y(0)/∂p with the IC Jacobian column, and so
    // NetworkModel::set_param() can re-resolve the same ICs when a parameter
    // moves after load (issue #79).
    for (const auto &ref : bimpl_->species_param_refs) {
        auto pit = sd->param_name_to_idx.find(ref.param_name);
        if (pit != sd->param_name_to_idx.end()) {
            auto &sp = impl.species[ref.species_idx0];
            const double val = resolve_ic_from_param(sp, impl.parameters[pit->second].value);
            sp.concentration = val;
            sp.initial_conc = val;
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

    // ── Time dependence of the function block (issue #654) ───────────────
    //
    // Whether any function value can move with the clock alone. The SSA gates
    // its piecewise-constant sub-stepping on this: a rate law that reads a
    // purely time-dependent function has no species to trigger a propensity
    // refresh, so without sub-stepping it is held at its t_start value for the
    // whole run.
    //
    // Answered from the expression text, not by evaluating anything. The
    // previous owner of this decision sampled every function at three times and
    // called it constant when the values agreed, which is not a property three
    // samples can establish: a rate of period 5 over [0, 10] is probed at 0, 5
    // and 10 — one whole period apart each time — reads as constant, and the
    // trajectory is then computed against a rate that is not the model's, with
    // no warning (issue #654). `time` is the only clock symbol in the grammar
    // (`t` is deliberately left free as an ordinary identifier, see
    // src/expression.cpp), and a time-indexed table function is the other way
    // in — that one is flagged by register_table_function_() when the tfun is
    // registered, above and post-build alike.
    //
    // No transitive closure is needed: the flag is about the function block as
    // a whole, so a function that reads a time-dependent one is already covered
    // by the one it reads. Over-reporting is safe (sub-stepping a model that
    // did not need it costs time, not correctness); under-reporting is the bug.
    for (const auto &func : impl.functions) {
        if (impl.functions_use_time)
            break;
        for_each_identifier(func.expression, [&](const std::string &token) {
            if (token == "time")
                impl.functions_use_time = true;
        });
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

    // ── 7. Jacobian sparsity + analytical Jacobian ───────────────────────
    const int ns = static_cast<int>(impl.species.size());
    const int np = static_cast<int>(impl.parameters.size());

    sd->jac_sparsity =
        build_jac_sparsity(sd->reactions, ns, impl.observables, sd->observable_name_to_idx,
                           impl.functions, sd->function_name_to_idx, impl.species);

    sd->analytical_jac = build_anal_jac(sd->reactions, ns, np, sd->jac_sparsity, impl.species);

    // The Curtis-Powell-Reid coloring is NOT computed here (GH #29): it is
    // consumed only by the sparse colored-FD Jacobian callback, so
    // ensure_jacobian_coloring() materializes it on first use (once, shared
    // across clones). See model_impl.hpp for the immutability contract.

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

            // Compile trigger expression. The source is retained verbatim
            // (issue #49): the event-time sensitivity detector differentiates
            // the trigger's threshold symbolically and must see the model's own
            // spelling, not the evaluator's preprocessed form.
            ev.trigger_source = espec.trigger_expr;
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
